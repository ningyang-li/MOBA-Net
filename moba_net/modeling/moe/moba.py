# Mixture-of-Sampling (MoS) Module

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
from ...utils.utils import inverse_sigmoid


from ..pixel_decoder.modules import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP,
)
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


class MOBA(nn.Module):
    def __init__(self, n_experts=4, k=1, d_model=256, d_mask=None, n_levels=4, n_heads=8, noisy_gating=True, acc_aux_loss=True,
                 boltzmann_sampling={"mask_threshold": 0.5, "do_boltzmann": False, "sample_ratio": 0.1, "base_temp": 1, "use_sparse": False},
                 inside_box_head=False):
        """
        Mixture-of-Boltzmann-Attention (MOBA) Module (based on Boltzmann Sampling and Multi-Head Self-Attention Module)
        All experts have the same structure
        
        :param n_experts    number of experts (for both offset layer and attention layer)
        :param k            number of selected expert, k must be 1 to avoid to modify the cuda implemtntation of MSDA
        :param d_model      hidden dimension
        :param d_mask       boltzmann mask dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param noisy_gating whether add noise to clear logits of gate
        :param acc_aux_loss save the usage of experts
        :param boltzmann_sampling 
                            Boltzmann sampling on attention mask
                            "mask_threshold"   original threshold for masked attention
                            "do_boltzmann"     whether to do Boltzmann sampling
                            "sample_ratio"     number of iid samples as a ratio of total number of masked tokens
                            "base_temp"        base temperature for Boltzmann sampling
                            "gaussian_sampling" use gaussian sampling after boltzmann probability
        :param inside_box_head use box heads in experts
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        assert n_experts >= 1
        
        self.im2col_step = 128
        self.n_experts = n_experts
        self.k = k
        self.d_model = d_model
        d_mask = d_mask if d_mask != None else d_model
        self.d_mask = d_mask
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.noisy_gating = noisy_gating
        self.acc_aux_loss = acc_aux_loss
        self.boltzmann_sampling = boltzmann_sampling
        self.inside_box_head = inside_box_head

        # gate layer
        self.gate = nn.Linear(d_model, 2 * n_experts if noisy_gating else n_experts, bias=False)

        # expert layers  ParallelExperts(self, num_experts, input_size, output_size, bias=False)
        self.boltzmann_embed_experts_1 = ParallelExperts(n_experts, d_model, d_model, bias=True)
        self.boltzmann_embed_experts_2 = ParallelExperts(n_experts, d_model, d_model, bias=True)
        self.boltzmann_embed_experts_3 = ParallelExperts(n_experts, d_model, d_mask, bias=True)

        self.q_experts = ParallelExperts(n_experts, d_model, d_model, bias=True)

        if inside_box_head:
            self.box_embed_experts_1 = ParallelExperts(n_experts, d_model, d_model, bias=True)
            self.box_embed_experts_2 = ParallelExperts(n_experts, d_model, d_model, bias=True)
            self.box_embed_experts_3 = ParallelExperts(n_experts, d_model, 4, bias=True)

        self.decoder_norm = nn.LayerNorm(d_mask)
        
        # regular layers
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        self.init_aux_statistics()
        
        # expert frequency
        self.expert_frequency = torch.zeros((n_experts,), device='cuda')

    def _reset_parameters(self):
        constant_(self.gate.weight.data, 0.)
        
        xavier_uniform_(self.boltzmann_embed_experts_1.w.data)
        constant_(self.boltzmann_embed_experts_1.b.data, 0.)
        xavier_uniform_(self.boltzmann_embed_experts_2.w.data)
        constant_(self.boltzmann_embed_experts_2.b.data, 0.)
        xavier_uniform_(self.boltzmann_embed_experts_3.w.data)
        constant_(self.boltzmann_embed_experts_3.b.data, 0.)
        xavier_uniform_(self.q_experts.w.data)
        constant_(self.q_experts.b.data, 0.)
        
        if self.inside_box_head:
            xavier_uniform_(self.box_embed_experts_1.w.data)
            constant_(self.box_embed_experts_1.b.data, 0.)
            xavier_uniform_(self.box_embed_experts_2.w.data)
            constant_(self.box_embed_experts_2.b.data, 0.)
            xavier_uniform_(self.box_embed_experts_3.w.data)
            constant_(self.box_embed_experts_3.b.data, 0.)
        
        xavier_uniform_(self.key_proj.weight.data)
        constant_(self.key_proj.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

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
        
        # print expert frequency
        # self.expert_frequency += self.acc_freq
        # print(id(self.expert_frequency), self.expert_frequency)
        
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
        clean_logits = self.gate(x)
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
        
        return loss, logits, probs, gates

    def gaussian_sampling(self, masked_prob, ref_points, layer_id, N, Len_q, new_size):
        # generate gaussian distribution based on ref_points
        # ref_points: [N, Len_q, n_level, 4] (cx, cy, w, h) in [0,1]
        cur_level = layer_id % self.n_levels
        ref_cur = ref_points[:, :, cur_level, :]  # [N, Len_q, 4]
        masked_prob = masked_prob.view(N*self.k, self.n_heads, Len_q, new_size, new_size)
        h, w = masked_prob.shape[-2], masked_prob.shape[-1]
        device = masked_prob.device

        # 1. normalized centers [h, w, 2]
        y_coords = torch.linspace(0.5/h, 1-0.5/h, h, device=device)
        x_coords = torch.linspace(0.5/w, 1-0.5/w, w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        patch_coords = torch.stack([xx, yy], dim=-1)  # [h, w, 2]
        
        # 2. squared distance between center and each patch [N, Len_q, h, w]
        centers = ref_cur[:, :, :2].unsqueeze(2).unsqueeze(2)   # [N, Len_q, 1, 1, 2]
        patch_coords = patch_coords.unsqueeze(0).unsqueeze(0) # [1, 1, h, w, 2]
        dist_sq = ((patch_coords - centers) ** 2).sum(-1)      # [N, Len_q, h, w]
        
        # 3. Gaussian probability
        wh = ref_cur[:, :, 2:4].clamp(min=1e-6)
        sigma_sq = (wh.prod(-1) * 3.0).view(N, Len_q, 1, 1)   #
        spatial_prior = torch.exp(-dist_sq / sigma_sq)        # [N, Len_q, h, w]
        
        # 4. align shape of attn_mask [N*k, Head, Len_q, h, w] 
        spatial_prior = spatial_prior.unsqueeze(1).unsqueeze(1).repeat(1, self.k, self.n_heads, 1, 1, 1)
        spatial_prior = spatial_prior.view(N * self.k, self.n_heads, Len_q, h, w)
        # multiplication
        masked_prob = masked_prob * spatial_prior
        
        # 5. limited range
        half_w = (ref_cur[:, :, 2:3] / 2).unsqueeze(-1)  # [N, Len_q, 1, 1]
        half_h = (ref_cur[:, :, 3:4] / 2).unsqueeze(-1)  # [N, Len_q, 1, 1]
        cx = ref_cur[:, :, 0:1].unsqueeze(-1)            # [N, Len_q, 1, 1]
        cy = ref_cur[:, :, 1:2].unsqueeze(-1)            # [N, Len_q, 1, 1]
        
        # coordinates of each patch [1, 1, h, w]
        px = patch_coords[..., 0]  # [1, 1, h, w]
        py = patch_coords[..., 1]
        
        margin = 5.0
        # [1,1,h,w] vs [N,Len_q,1,1] -> [N,Len_q,h,w]
        in_range = (px > cx - half_w * margin) & (px < cx + half_w * margin) & \
                   (py > cy - half_h * margin) & (py < cy + half_h * margin)
        # in_range: [N, Len_q, h, w]
        in_range = in_range.unsqueeze(1).unsqueeze(1).repeat(1, self.k, self.n_heads, 1, 1, 1).view(N*self.k, self.n_heads, Len_q, h, w)
        # sampling
        boltzmann_mask = masked_prob.masked_fill(~in_range, -10.0) * (-1.)
        boltzmann_mask = boltzmann_mask.flatten(-2)
        
        return boltzmann_mask
        
    def dense_attention(self, query_embed, key, value, attn_mask,):
        # [N, k, n_heads, Len_q, d_model//n_heads] @ [N, n_heads, d_model//n_heads, Len_in] -> [N, k, n_heads, Len_q, Len_in]
        attn_scores = torch.matmul(query_embed, key) / math.sqrt(self.d_model//self.n_heads)
        attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        all_masked = attn_mask.all(dim=-1, keepdim=True)  # [N, k, n_heads, Len_q, 1]
        attn_weights = torch.where(all_masked, 
                                   torch.zeros_like(attn_weights),
                                   attn_weights)
        # [N, k, n_heads, Len_q, Len_in] @ [N, n_heads, Len_in, d_model//n_heads] -> [N, k, n_heads, Len_q, d_model//n_heads]
        output = torch.matmul(attn_weights, value)

        return output, attn_weights

    def forward(self, query, input_flatten, mask_features, attn_mask_target_size, layer_id=-1, ref_points=None):
        """
        :param query                       (N, Length_{query}, C) for self-attention, Length_{query}=H_i*W_i
        :param input_flatten               (N, H_i*W_i, C)
        :param mask_features               (N, H_i*W_i, C)
        :param attn_mask_target_size       (H_{i+1}, W_{i+1})
        :param layer_id                    i
        :ref_points                        reference points of queries
        
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # each query is the basic unit for an expert, there are N*Len_q queries
        query = query.reshape((N*Len_q, self.d_model))
        query = self.decoder_norm(query)
        assert not torch.isnan(
            query
        ).any(), f"NaN detected after query in layer {layer_id}"
        
        # gating
        loss, logits, probs, gates = self.top_k_gating(query)
        
        # routing
        # Original Index ==> Expert Index
        expert_input = query[self.batch_index]
        # boltzmann_embed_experts
        # ==================================================================================================
        mask_embed_h = F.silu(self.boltzmann_embed_experts_1(expert_input, self.expert_size))
        mask_embed_h = F.silu(self.boltzmann_embed_experts_2(mask_embed_h, self.expert_size))
        mask_embed = self.boltzmann_embed_experts_3(mask_embed_h, self.expert_size)

        # # box_embed_experts
        # # ==================================================================================================
        if self.inside_box_head:
            box_embed_h = F.silu(self.box_embed_experts_1(expert_input, self.expert_size))
            box_embed_h = F.silu(self.box_embed_experts_2(box_embed_h, self.expert_size))
            box_embed = self.box_embed_experts_3(box_embed_h, self.expert_size)
            zeros = torch.zeros((N*Len_q*self.k, 4), dtype=box_embed.dtype, device=box_embed.device)
            box_embed = zeros.index_add(0, self.index_sorted_experts, box_embed)
            box_embed = box_embed.view(N, Len_q, self.k, 4)
            # box_embedtorch.Size([1, 1200, 2, 4]), ref_points[:, :, layer_id:layer_id+1]: torch.Size([1, 1200, 1, 4])
            outputs_box = box_embed + inverse_sigmoid(ref_points[:, :, layer_id:layer_id+1]).to(ref_points.device)
            outputs_box = outputs_box.sigmoid()
            outputs_box = outputs_box.permute(0, 2, 1, 3) # bkq4
            outputs_box = outputs_box.view(N*self.k, Len_q, 4)
        else:
            outputs_box = None

        # mask_embed = mask_embed * self.batch_gates[:, None]
        zeros = torch.zeros((N*Len_q*self.k, self.d_model), dtype=mask_embed.dtype, device=mask_embed.device)
        mask_embed = zeros.index_add(0, self.index_sorted_experts, mask_embed)
        mask_embed = mask_embed.view(N, Len_q, self.k, self.d_model)
        # mask_embed.shape [N, Len_q, k, d_model], mask_features.shape [N, n=h*w, d_model]
        # every expert will ocnduct boltzmann sampling exclusively
        outputs_mask = torch.einsum("bqkc,bcn->bqkn", mask_embed, mask_features.transpose(1, 2)) 
        outputs_mask = outputs_mask.permute(0, 2, 1, 3) # bkqn
        old_size = int(np.sqrt(outputs_mask.shape[-1]))
        outputs_mask = outputs_mask.view(N, self.k, Len_q, old_size, old_size)  # bkqn->bkqhw
        outputs_mask = outputs_mask.view(N*self.k, Len_q, old_size, old_size)  # bkqHW->(bk)qhw always be 1/4 scale
        # down-sampling
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )   # (bk)qhw->(bk)qhw
        new_size = attn_mask.shape[-1]

        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)                       # (bk)qhw->(bk)qn
            .unsqueeze(1)                     # (bk)qn->(bk)1qn
            .repeat(1, self.n_heads, 1, 1)    # (bk)1qn->(bk)Hqn
        ).detach()

        threshold = self.boltzmann_sampling["mask_threshold"]
        do_boltzmann = self.boltzmann_sampling["do_boltzmann"]
        sample_ratio = self.boltzmann_sampling["sample_ratio"]
        base_temp = self.boltzmann_sampling["base_temp"]
        gaussian_sampling = self.boltzmann_sampling["gaussian_sampling"]

        if do_boltzmann:
            # probability of Boltzman sampling
            Temp = base_temp / (
                1 + layer_id
            )  # temperature decays with layer number (first layer from id -1)
            boltzmann_prob = torch.exp(attn_mask / Temp)
            
            boltzmann_prob = (
                boltzmann_prob * (attn_mask < threshold).float()
            )  # remove unmasked regions
            boltzmann_prob = boltzmann_prob / (boltzmann_prob.sum(dim=-1, keepdim=True) + 1e-6)

            # sample from Boltzman distribution n times
            n_samples = int(
                attn_mask.shape[-1] * sample_ratio
            )  # number of iid samples on the tokens
            masked_prob = (
                1 - boltzmann_prob
            ) ** n_samples  # probability that each token is still masked after n iid samples

            if ref_points is not None and layer_id >= 0 and gaussian_sampling:
                # gaussian sampling
                boltzmann_mask = self.gaussian_sampling(masked_prob, ref_points, layer_id, N, Len_q, new_size)                
            else:
                # random sampling
                boltzmann_mask = (torch.rand_like(boltzmann_prob) < masked_prob).bool()
                
            # combine with original mask
            attn_mask = torch.logical_and(
                (attn_mask < threshold).bool(), boltzmann_mask
            )
        else:
            attn_mask = (attn_mask < threshold).bool()
        
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        
        assert not torch.isnan(
            attn_mask
        ).any(), f"NaN detected in attn_mask in layer {layer_id}"
        # query experts
        # ====================================================================================================
        
        # cross-attention
        query_embed = self.q_experts(expert_input, self.expert_size)
        zeros = torch.zeros((N*Len_q*self.k, self.d_model), dtype=query_embed.dtype, device=query_embed.device)
        query_embed = zeros.index_add(0, self.index_sorted_experts, query_embed)
        query_embed = query_embed.view(N, Len_q, self.k, self.n_heads, self.d_model//self.n_heads)
        query_embed = query_embed.permute(0, 2, 3, 1, 4)  # (N, self.k, self.n_heads, Len_q, self.d_model//self.n_heads)
        # do not use moe
        key = self.key_proj(input_flatten)
        key = key.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        key = key.permute(0, 2, 1, 3)
        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = value.permute(0, 2, 1, 3)

        output, attn_weights = self.dense_attention(query_embed, key.transpose(-2, -1), value, attn_mask.view(N, self.k, self.n_heads, Len_q, Len_in))
        
        # output (N, k, n_heads, Len_q, d_head)
        # merge heads
        output = output.transpose(2, 3).contiguous().view(N, self.k, Len_q, self.d_model)
        output = output.permute(0, 2, 1, 3)    # [N, Len_q, self.k, self.d_model]
        # merge experts
        output = output.reshape(-1, self.d_model) # [N*Len_q*self.k, self.d_model]
        output = output[self.index_sorted_experts]
        output = output * self.batch_gates[:, None]
        zeros = torch.zeros((N*Len_q, self.d_model), dtype=query_embed.dtype, device=query_embed.device)
        output = zeros.index_add(0, self.batch_index, output)
        output = output.view(N, Len_q, self.d_model)
    
        output = self.output_proj(output)

        # attn_weights (N, k, n_heads, Len_q, Len_in)
        # merge heads
        attn_weights_aggregated = attn_weights.mean(dim=2)
        attn_weights_aggregated = attn_weights_aggregated.permute(0, 2, 1, 3)    # [N, Len_q, self.k, Len_in]
        attn_weights_aggregated2 = attn_weights_aggregated
        # merge experts
        attn_weights_aggregated = attn_weights_aggregated.reshape(-1, Len_in) # [N*Len_q*self.k, Len_in]
        attn_weights_aggregated = attn_weights_aggregated[self.index_sorted_experts]
        attn_weights_aggregated = attn_weights_aggregated * self.batch_gates[:, None]
        zeros = torch.zeros((N*Len_q, Len_in), dtype=query_embed.dtype, device=query_embed.device)
        attn_weights_aggregated = zeros.index_add(0, self.batch_index, attn_weights_aggregated)
        attn_weights_aggregated = attn_weights_aggregated.view(N, Len_q, Len_in)
        
        # from PIL import Image
        # for query_id in range(1200):
        #     matrix = attn_weights_aggregated2[0, query_id, 0].reshape(int(np.sqrt(attn_weights_aggregated.shape[-1])), int(np.sqrt(attn_weights_aggregated.shape[-1]))).float().cpu().numpy()
        #     normalized_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        #     if layer_id < 2:
        #         gray = (255 * ((matrix<=matrix.mean()))).astype(np.uint8)
        #     else:
        #         gray = (255 * ((matrix>matrix.mean()))).astype(np.uint8)
        #     gray = np.repeat(np.repeat(gray, 4*(2**layer_id), axis=0), 4*(2**layer_id), axis=1) # nearest sampling
        #     Image.fromarray(gray, 'L').save('boltzmann_attn/boltzmann_attn-' + str(query_id) + '-' + str(layer_id) + '.png')

        outputs_mask = outputs_mask.flatten(-2).reshape(N*self.k*Len_q, old_size*old_size)
        outputs_mask = outputs_mask * self.batch_gates[:, None]
        zeros = torch.zeros((N*Len_q, old_size*old_size), dtype=query_embed.dtype, device=query_embed.device)
        outputs_mask = zeros.index_add(0, self.batch_index, outputs_mask)
        outputs_mask = outputs_mask.view(N, Len_q, old_size*old_size).view(N, Len_q, old_size, old_size)

        if self.inside_box_head:
            outputs_box = outputs_box.flatten(-2).reshape(N*self.k*Len_q, 4)
            outputs_box = outputs_box * self.batch_gates[:, None]
            zeros = torch.zeros((N*Len_q, 4), dtype=query_embed.dtype, device=query_embed.device)
            outputs_box = zeros.index_add(0, self.batch_index, outputs_box)
            outputs_box = outputs_box.view(N, Len_q, 4)

        losses = {}
        losses.update(self.get_topk_loss_and_clear())
        losses.update(self.get_aux_loss_and_clear())

        return output, losses, [attn_weights, attn_weights_aggregated], outputs_mask, outputs_box








        