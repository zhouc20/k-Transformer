import torch
from torch import Tensor
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.layer.tensorized_transformer_layer import Tensorized_12_Layer
import logging


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

        self.exp_edge_fixer = CrossAttnIndexFixer()

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class CrossAttnIndexFixer(torch.nn.Module):
    def __init__(self):
        super(CrossAttnIndexFixer, self).__init__()

    def forward(self, batch):
        batch.batch_query_idx = num2batch(batch.query_idx.reshape(-1))

        data_list = batch.to_data_list()
        batch_key_idx = []
        cumulative_num_nodes = 0
        for data in data_list:
            batch_key_idx.append(data.key_idx + cumulative_num_nodes)
            cumulative_num_nodes += data.num_edge_per_graph
        batch_key_idx = torch.cat(batch_key_idx, dim=0)
        batch.batch_key_idx = batch_key_idx.long()
        return batch


def num2batch(num_edge: Tensor):
    offset = cumsum_pad0(num_edge)
    # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
    batch = torch.zeros((offset[-1] + num_edge[-1]),
                        device=offset.device,
                        dtype=offset.dtype)
    batch[offset] = 1
    batch[0] = 0
    batch = batch.cumsum_(dim=0)
    return batch


def cumsum_pad0(num: Tensor):
    ret = torch.empty_like(num)
    ret[0] = 0
    ret[1:] = torch.cumsum(num[:-1], dim=0)
    return ret


@register_network('TensorizedTransformer')
class CrossAttnTensorizedTransformerModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(Tensorized_12_Layer(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=True,
                norm_e=cfg.gt.attn.norm_e,
                O_e=cfg.gt.attn.O_e,
                cfg=cfg.gt
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
            # logging.info(batch)
        return batch
