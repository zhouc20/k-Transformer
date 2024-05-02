import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

# from grit.utils import negate_edge_index
from torch_geometric.graphgym.register import *
import opt_einsum as oe

from yacs.config import CfgNode as CN

import warnings
import logging

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out



class MultiHeadCrossAttentionLayer(nn.Module):
    """
        Cross Attention layer
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.E = nn.Linear(in_dim, in_dim, bias=True)
        self.T = nn.Linear(in_dim, in_dim, bias=True)
        self.P = nn.Linear(3 * in_dim, in_dim, bias=True)
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.A = nn.Embedding(3, out_dim * num_heads)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.T.weight)
        nn.init.xavier_normal_(self.P.weight)

        # self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        # nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.batch_key_idx]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.batch_query_idx]     # (num relative) x num_heads x out_dim
        score = src * dest / np.sqrt(self.out_dim)    # element-wise multiplication
        c = self.A(batch.connectivity).reshape(-1, self.num_heads, self.out_dim)
        score = score * c
        score = score.sum(-1, keepdim=True)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, batch.batch_query_idx)  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.batch_key_idx] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.Q_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.batch_query_idx, dim=0, out=batch.wV, reduce='add')

    def forward(self, batch):
        x1 = self.act(self.T(batch.x))[batch.edge_index[0]]
        x2 = self.act(self.T(batch.x))[batch.edge_index[1]]
        if batch.get("edge_attr", None) is not None:
            batch.E = self.act(self.E(batch.edge_attr))
        else:
            batch.E = None
        Z = self.act(self.P(torch.cat([x1, x2, batch.E], dim=1)))

        Q_h = self.Q(batch.x)
        K_h = self.K(Z)
        V_h = self.V(Z)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV

        return h_out, Z


@register_layer("Tensorized_12_Layer")
class Tensorized_12_Layer(nn.Module):
    """
        Proposed Transformer Layer for cross attention bewteen 1-tuples and 2-tuples
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = cfg.get("update_e", False)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadCrossAttentionLayer(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=True,
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn =cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch):
        h = batch.x
        # logging.info('there is x')
        num_nodes = batch.num_nodes

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            log_deg = get_log_deg(batch)
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg


