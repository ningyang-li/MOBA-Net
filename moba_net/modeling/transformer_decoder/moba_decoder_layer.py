# ELSE-MoS Decoder Layer

from typing import Optional, List, Union
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast

from ...utils.utils import MLP, _get_clones, _get_activation_fn, gen_sineembed_for_position, inverse_sigmoid
from ..moe import MOBA, MoFFN, CAFFN


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MOBA):
                m._reset_parameters()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]
        outputs_mask_list = []
        outputs_box_list = [reference_points.transpose(0, 1)]
        # outputs_box_list = []
        losses_moe = {'topk_loss': torch.zeros((1,), device=tgt.device), 'switch_loss': torch.zeros((1,), device=tgt.device), 'z_loss': torch.zeros((1,), device=tgt.device)}
        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            # =======================================================================================================================
            # use certain scale of features successively
            level_index = layer_id % layer.num_feature_levels
            
            reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, level_index, :]) # nq, bs, 256*2

            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            attn_mask_target_size = tuple(spatial_shapes[(layer_id) % layer.num_feature_levels])
            query_start_index = level_start_index[level_index]
            query_end_index = level_start_index[level_index+1] if level_index != layer.num_feature_levels-1 else memory.shape[0]
            # fetch query from flattened inputs
            cur_memory = memory[query_start_index:query_end_index]
            memory_mask = memory_mask.flatten(2)
            
            output, loss_mos, loss_moe, outputs_mask, outputs_box = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,

                memory=cur_memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,

                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                
                attn_mask_target_size=attn_mask_target_size,
                layer_id=layer_id,
            )

            # aggregate losses of all mos and moe layers
            for k in losses_moe.keys():
                losses_moe[k] += loss_mos[k]
                losses_moe[k] += loss_moe[k]

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
                # ref_points.append(reference_points_input[:, :, layer_id:layer_id+1].transpose(0, 1).squeeze(0).contiguous().to(device))
        
            intermediate.append(self.norm(output))
            outputs_mask_list.append(outputs_mask)
            outputs_box_list.append(outputs_box)
            

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            losses_moe, outputs_mask_list, outputs_box_list
        ]


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 # moe args
                 attn_type="moba", ffn_type="caffn",
                 n_experts=4, k=1,
                 boltzmann_sampling={"mask_threshold": 0.5, "do_boltzmann": True, "sample_ratio": 0.1, "base_temp": 1, "use_sparse": False},
                 noisy_gating=True, acc_aux_loss=True, shared_gate=True, n_ranks=[4, 8, 16, 32, 64, 128], inside_box_head=False):
        super().__init__()
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.num_feature_levels = n_levels
        
        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            if attn_type == "moba":
                self.cross_attn = MOBA(n_experts=n_experts, k=k, d_model=d_model, n_levels=n_levels, n_heads=n_heads,
                                       boltzmann_sampling=boltzmann_sampling, noisy_gating=noisy_gating, acc_aux_loss=acc_aux_loss,
                                       inside_box_head=inside_box_head)
            else:
                self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        if ffn_type == "moffn":
            self.ffn = MoFFN(n_experts=n_experts, k=k, d_model=d_model, d_ffn=d_ffn, noisy_gating=noisy_gating,
                             acc_aux_loss=acc_aux_loss, dropout=dropout)
        elif ffn_type == "caffn":
            self.ffn = CAFFN(n_experts=n_experts, k=k, d_model=d_model, d_ffn=d_ffn, noisy_gating=noisy_gating, n_ranks=n_ranks,
                             acc_aux_loss=acc_aux_loss, dropout=dropout)
        else:
            # ffn
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn, d_model),
                nn.Dropout(dropout)
            )
        
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                
                # boltzmann
                attn_mask_target_size: tuple = (),
                layer_id: int = 0,
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        loss_mos = {'topk_loss': torch.zeros((1,), device=tgt.device), 'switch_loss': torch.zeros((1,), device=tgt.device), 'z_loss': torch.zeros((1,), device=tgt.device)}
        loss_moe = {'topk_loss': torch.zeros((1,), device=tgt.device), 'switch_loss': torch.zeros((1,), device=tgt.device), 'z_loss': torch.zeros((1,), device=tgt.device)}
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        if self.attn_type == "moba":
            # query, input_flatten, mask_features, attn_mask_target_size, layer_id
            tgt2, loss_mos, attn_weights_list, outputs_mask, outputs_box = self.cross_attn(query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                                                                             input_flatten=memory.transpose(0, 1),
                                                                                             mask_features=cross_attn_mask.transpose(1, 2),
                                                                                             attn_mask_target_size=attn_mask_target_size,
                                                                                             layer_id=layer_id,
                                                                                             ref_points=tgt_reference_points.transpose(0, 1).contiguous(),
                                                                                             )
        else:
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                                       tgt_reference_points.transpose(0, 1).contiguous(),
                                                       memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                                                       memory_key_padding_mask)
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ffn
        if self.ffn_type == "moffn":
            tgt3, loss_moe = self.ffn(tgt)
        elif self.ffn_type == "caffn":
            tgt3, loss_moe = self.ffn(tgt, attn_weights_list[1])
        else:
            tgt3 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt3)

        return tgt, loss_mos, loss_moe, outputs_mask, outputs_box









