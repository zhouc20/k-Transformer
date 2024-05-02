import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from graphgps.layer.performer_layer import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from typing import Final, List, Tuple, Optional
from torch import Tensor
import torch.nn.functional as F
import math

from torch_geometric.graphgym.config import cfg
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.rwgnn_layer import RW_MPNN_layer, RW_Transformer_layer
from graphgps.layer.Exphormer import ExphormerAttention
import logging


class Simplicial_01_dense_Layer(nn.Module):
    """ simplicial transformer containing 0-simplicial and 1-simplicial
        dense version
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, exp_edges_cfg=None, log_attn_weights=False, num_layer_MPNN=1, similarity_type='cos',
                 inference_mode='original', mp_threshold=0.0, force_undirected=False, complex_type='original',
                 complex_pool_type='add', cluster_threshold=0.1, complex_max_distance=5, focusing_factor=1):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.num_layer_MPNN = num_layer_MPNN
        self.similarity_type = similarity_type
        self.inference_mode = inference_mode
        self.force_undirected = force_undirected
        self.mp_threshold = mp_threshold

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type != 'Transformer':
            raise NotImplementedError(
                "Logging of attention weights is only supported for "
                "Transformer global attention model."
            )
        if global_model_type == 'GINE_RW':
            self.inference_mode = 'original'

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
                if self.inference_mode in ['inter_intra', 'original_inter']:
                    gin_nn_intra = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                           self.activation(),
                                           Linear_pyg(dim_h, dim_h))
                    self.local_model_intra = GINEConvESLapPE(gin_nn_intra)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
                if self.inference_mode in ['inter_intra', 'original_inter']:
                    gin_nn_intra = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                                 self.activation(),
                                                 Linear_pyg(dim_h, dim_h))
                    self.local_model_intra = pygnn.GINEConv(gin_nn_intra)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        elif local_gnn_type == 'GINE_RW':
            gin_nn = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            gin_nn_2 = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                                  self.activation(),
                                                  Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            self.local_model = RW_MPNN_layer(gin_nn, gin_nn_2, num_layer_MPNN, similarity_type, inference_mode, mp_threshold, force_undirected)

        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = MultiheadAttention(self.dim_h, self.num_heads, dropout=self.attn_dropout, batch_first=True)
            self.attn_bias = AttentionBias(self.num_heads)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads, generalized_attention=True,
                dropout=self.attn_dropout, causal=False, focusing_factor=focusing_factor)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif global_model_type == 'Exphormer':
            self.self_attn = ExphormerAttention(dim_h, dim_h, num_heads,
                                                use_bias=False,
                                                use_virt_nodes=exp_edges_cfg.num_virt_node > 0,
                                                use_local_neighbors=exp_edges_cfg.get("use_local_neighbors", False))
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            pass

        # Multi-head attention.
        assert self.self_attn is not None
        if self.global_model_type in ['ETransformer', 'Exphormer']:
            h_attn = self.self_attn(batch)
        else:
            h_dense, mask = to_dense_batch(h, batch.batch_simplex)
            b, Nm, hid_dim = h_dense.shape[0], h_dense.shape[1], h_dense.shape[2]
            assert b == batch.num_graphs and hid_dim == self.dim_h

            if self.global_model_type == 'Transformer':
                hodge_laplacian = torch.zeros([b, Nm, Nm, 1], dtype=torch.long, device=h_dense.device)
                accumulate_tokens = 0
                for i in range(b):
                    n_i = batch.num_node_per_graph[i] + batch.num_edge_per_graph[i]
                    ln_edge_attr = batch.local_neighbor_edge_attr[accumulate_tokens: accumulate_tokens + n_i ** 2]
                    hodge_laplacian[i, :n_i, :n_i, :] = ln_edge_attr.reshape(n_i, n_i, 1)
                    accumulate_tokens += n_i ** 2
                attn_bias = self.attn_bias(hodge_laplacian.long().squeeze(3))
                # logging.info(attn_bias.shape)
                # logging.info(mask.dtype)
                h_attn = self.self_attn(h_dense, h_dense, h_dense, attn_bias, attn_mask=None,
                                        key_padding_mask=~mask,
                                        need_weights=False)[0][mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s


class Simplicial_01_sparse_Layer(nn.Module):
    """ simplicial transformer containing 0-simplicial and 1-simplicial
        sparse version
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, exp_edges_cfg=None, log_attn_weights=False, num_layer_MPNN=1, similarity_type='cos',
                 inference_mode='original', mp_threshold=0.0, force_undirected=False, complex_type='original',
                 complex_pool_type='add', cluster_threshold=0.1, complex_max_distance=5, focusing_factor=1):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.num_layer_MPNN = num_layer_MPNN
        self.similarity_type = similarity_type
        self.inference_mode = inference_mode
        self.force_undirected = force_undirected
        self.mp_threshold = mp_threshold

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type != 'Transformer':
            raise NotImplementedError(
                "Logging of attention weights is only supported for "
                "Transformer global attention model."
            )
        if local_gnn_type != 'None':
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'Exphormer':
            self.self_attn = ExphormerAttention(dim_h, dim_h, num_heads,
                                                use_bias=False,
                                                use_virt_nodes=exp_edges_cfg.num_virt_node > 0,
                                                use_local_neighbors=exp_edges_cfg.get("use_local_neighbors", True))
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.


        # Multi-head attention.
        assert self.self_attn is not None
        if self.global_model_type in ['ETransformer', 'Exphormer']:
            h_attn = self.self_attn(batch)
        else:
            raise RuntimeError(f"Unexpected {self.global_model_type}")

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        batch_first: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        # self.dropout_module = FairseqDropout(
        #     dropout, module_name=self.__class__.__name__
        # )
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.batch_first = batch_first

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # self.k_proj = quant_noise(
        #     nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        # self.v_proj = quant_noise(
        #     nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        # self.q_proj = quant_noise(
        #     nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        #
        # self.out_proj = quant_noise(
        #     nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        # )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        if self.batch_first:
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        tgt_len, bsz, embed_dim = query.size()
        # bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.reshape(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.softmax(
            attn_weights, dim=-1 #, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, self.dropout)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if self.batch_first:
            attn = attn.permute(1, 0, 2)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


from torch_geometric.graphgym.config import cfg

class AttentionBias(nn.Module):
    def __init__(self,
                 n_head: int,
                 # max_degree: int = 10,
                 max_distance: int = 1,
                 max_bond_type: int = 20):  # cfg.dataset.get('max_degree', 15) + 5
        super().__init__()

        self.n_head = n_head
        # self.max_degree = max_degree
        self.max_distance = max_distance
        self.max_bond_type = cfg.dataset.get('max_degree', 20)

        self.edge_encode = nn.Embedding(self.max_bond_type, self.n_head, padding_idx=0)
        self.distance_encode = nn.Embedding(self.max_distance, self.n_head)

    def forward(self, adj=None, distance=None):
        # logging.info(torch.max(adj))
        # logging.info(self.max_bond_type)
        x1 = self.edge_encode(adj).permute(0, 3, 1, 2) if adj is not None else 0
        x2 = self.distance_encode(distance).permute(0, 3, 1, 2) if distance is not None else 0
        x = x1 + x2
        return x
