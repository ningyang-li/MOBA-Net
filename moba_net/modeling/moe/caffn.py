

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..pixel_decoder.ops.functions import MSDeformAttnFunction
from ..pixel_decoder.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from .base.parallel_experts import ParallelExperts


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    # probs: original probs
    # top_k_gates: logits of topK gating function
    # top_k_indices: indices of topK exprets
    
    # create an array containing the topk logits only
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    
    # flat for sorting
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    
    # exclude zero elements and sort the remaining
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    
    # count
    # number of selected experts
    expert_size = (gates > 0).long().sum(0)
    # indices of selected and sorted experts
    index_sorted_experts = nonzeros[_index_sorted_experts]
    # map sorted indices to original indices
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    # get corrssponding gating logits (sorted)
    batch_gates = top_k_gates[index_sorted_experts]
    
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class CAFFN(nn.Module):
    def __init__(self, n_experts=6, k=2, d_model=256, d_ffn=1024, noisy_gating=True, n_ranks=[4, 8, 16, 32, 64, 128], acc_aux_loss=True, dropout=0.):
        """
        Complexity-Aware Feed-Forward Network (CAFFN)

        Different experts exhibit different ranks
        
        :param n_experts    number of experts (for both offset layer and attention layer)
        :param k            number of selected expert, k must be 1 to avoid to modify the cuda implemtntation of MSDA
        :param d_model      input dimension
        :param d_ffn        hidden dimension
        :param noisy_gating whether add noise to clear logits of gate
        :ranks n_ranks of each expert
        :param acc_aux_loss save the usage of experts
        :param dropout      dropout rate of all linear layers
        """
        super().__init__()
        assert n_experts >=1 and n_experts <= len(n_ranks)
        
        self.im2col_step = 128

        self.n_experts = n_experts
        self.k = k
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.noisy_gating = noisy_gating
        self.n_ranks = n_ranks
        self.acc_aux_loss = acc_aux_loss

        # rank-aware gate layer, it receive [number of valid patches, maximum value, diversity, sum] of each query
        self.rank_aware_gate = nn.Linear(4, 2 * n_experts if noisy_gating else n_experts, bias=False)

        # low-rank linear layers
        self.lora_start_experts = nn.ModuleList()
        for i in range(n_experts):
            layers = None
            layers = [nn.Linear(d_model, self.n_ranks[i], bias=False),
                      nn.Linear(self.n_ranks[i], d_ffn, bias=True),
                      nn.SiLU(),
                      nn.Dropout(dropout)]
            layers = nn.Sequential(*layers)
            self.lora_start_experts.append(layers)
        
        self.lora_end_experts = nn.ModuleList()
        for i in range(n_experts):
            layers = None
            layers = [nn.Linear(d_ffn, self.n_ranks[i], bias=False),
                      nn.Linear(self.n_ranks[i], d_model, bias=True),
                      nn.SiLU(),
                      nn.Dropout(dropout)]
            layers = nn.Sequential(*layers)
            self.lora_end_experts.append(layers)
        
        self._reset_parameters()
        self.init_aux_statistics()

    def _reset_parameters(self):
        constant_(self.rank_aware_gate.weight.data, 0.)

        for i in range(self.n_experts):
            xavier_uniform_(self.lora_start_experts[i][0].weight.data) # Wa
            xavier_uniform_(self.lora_start_experts[i][1].weight.data) # Wb  
            constant_(self.lora_start_experts[i][1].bias.data, 0.)     # Bb
            xavier_uniform_(self.lora_end_experts[i][0].weight.data)   # Wa
            xavier_uniform_(self.lora_end_experts[i][1].weight.data)   # Wb
            constant_(self.lora_end_experts[i][1].bias.data, 0.)       # Bb
            
    def init_aux_statistics(self, clear=True):
        # initialize the statistics of probality, topk probality, frequency of router
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.topk_acc_probs = 0.
    
    def update_aux_statistics(self, logits, probs, gates):
        # update existing statistics of router
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)
        
        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

    def get_topk_loss_and_clear(self):
        # select topk probs and corresponding indices
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.k, dim=0)
        # create the array with the same shape containing the topk probs only
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        # squre of topk probs and original probs (MSE)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()
        
        # reset topk_acc_probs
        self.topk_acc_probs = 0.
        return {'topk_loss': topk_loss}
    
    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''
        # compute losses
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.n_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        zloss = torch.where(
            torch.isinf(zloss) | torch.isnan(zloss),
            torch.tensor(1e6, device=zloss.device),
            zloss
        )
        
        self.init_aux_statistics(clear=False)
        return {'switch_loss': switchloss, 'z_loss': zloss}
        
    def compute_switchloss(self, probs, freqs):
        # load-banlance loss
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.n_experts
        
    def compute_zloss(self, logits):
        # mean(log(sum(e(x)))^2)
        # suppress the maximum logit
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        zloss = torch.where(
            torch.isinf(zloss) | torch.isnan(zloss),
            torch.tensor(1e6, device=zloss.device),
            zloss
        )
        return zloss
    
    def top_k_gating(self, x, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # get original gating logits
        clean_logits = self.rank_aware_gate(x)
        # noisy gating during training
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        # noisy gating during test
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        # no noisy gating
        else:
            logits = clean_logits
        # activate original logits
        probs = torch.softmax(logits, dim=1) + 1e-4
        
        # top-1
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        
        # sort selected logits and get corresponding indices
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates
        
        # compute losses
        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.switchloss * self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        loss = torch.Tensor([[[loss,]]])
        return loss

    def get_rank_features(self, attn_weights_aggregated):
        # merge heads
        # (N, Len_q, Len_in)
        N, Len_q, Len_in = attn_weights_aggregated.shape
        routing_features = torch.cat([
            (attn_weights_aggregated > 0.01).float().sum(dim=-1, keepdim=True),  # number of valid patches
            attn_weights_aggregated.max(dim=-1, keepdim=True)[0],    # maximum value
            attn_weights_aggregated.std(dim=-1, keepdim=True),       # diversity
            attn_weights_aggregated.sum(dim=-1, keepdim=True),       # sum
        ], dim=-1)  # [N, Len_q, 4]

        return routing_features
    
    def forward(self, x, attn_weights_aggregated):
        """
        :param x                       (N, Length_{x}, C)
        :attn_weights_aggregated       (N, Length_{x}, C) # both expert and head dimensions have been aggregated

        :return output                 (N, Length_{x}, C)

        :::aggragate the expert and attention head dimensions of attn_weights
        :::infer gates based on aggregated attn_weights
        :::distribute queries to experts with different ranks
        :::aggregate outputs of all experts
        """
        N, Len_x, _ = x.shape
        
        # each query is the basic unit for an expert, there are N*Len_x queries
        x = x.reshape((N*Len_x, self.d_model))
        
        loss = self.top_k_gating(self.get_rank_features(attn_weights_aggregated).reshape((N*Len_x, 4)))
                    
        # routing
        # Original Index ==> Expert Index
        expert_input = x[self.batch_index]

        # Start low-rank experts
        output = []
        start_index = 0
        end_index = 0
        for i in range(self.n_experts):
            if self.expert_size[i] > 0:
                end_index = start_index + self.expert_size[i]
                h_cur_sampling = expert_input[start_index:end_index]
                h = self.lora_start_experts[i](h_cur_sampling)
                lora_output = self.lora_end_experts[i](h)
                output.append(lora_output)
                start_index = end_index

        out = output[0]
        for i in range(1, len(output)):
            out = torch.concat((out, output[i]), dim=0)

        out = out * self.batch_gates[:, None]
        # Expert Index ==> Original Index
        # transform indexes of patches to Original Space
        zeros = torch.zeros((N*Len_x, self.d_model), dtype=out.dtype, device=out.device)
        out = zeros.index_add(0, self.batch_index, out)
        out = out.view(N, Len_x, self.d_model)

        losses = {}
        losses.update(self.get_topk_loss_and_clear())
        losses.update(self.get_aux_loss_and_clear())

        return out, losses








        