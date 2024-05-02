import torch
from torch import Tensor
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.exp_edge_fixer import ExpanderEdgeFixer
from graphgps.layer.simplicial_transformer_layer import Simplicial_01_sparse_Layer, Simplicial_01_dense_Layer
import logging


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dense=False):
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

        self.tuple_encoder = SimplicialEncoder(dense=dense)

        if 'Exphormer' in cfg.gt.layer_type:
            self.exp_edge_fixer = ExpanderEdgeFixer(add_edge_index=cfg.prep.add_edge_index,
                                                    num_virt_node=cfg.prep.num_virt_node)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class SimplicialEncoder(torch.nn.Module):
    """
    Encoding simplicial complex features

    Args:
        dense (bool): indicate dense or sparse implementation
    """
    def __init__(self, dense=False):
        super(SimplicialEncoder, self).__init__()
        self.use_local_neighbors = cfg.prep.get("use_local_neighbors", True)
        self.dense = dense

    def forward(self, batch):
        data_list = batch.to_data_list()
        local_neighbor_edges = []
        cumulative_num_nodes = 0
        x_list = []
        for data in data_list:
            x_list.append(data.x)
            x_list.append(data.edge_attr)
            if self.use_local_neighbors and not self.dense:
                local_neighbor_edges.append(data.local_neighbor_edge_index + cumulative_num_nodes)
                cumulative_num_nodes += (data.num_edge_per_graph + data.num_node_per_graph)
        batch.x = torch.cat(x_list, dim=0)
        if self.use_local_neighbors and not self.dense:
            local_neighbor_edges = torch.cat(local_neighbor_edges, dim=1)
            batch.local_neighbor_edge_index = local_neighbor_edges.long()

        batch.batch_original = batch.batch.clone()
        batch.batch_simplex = num2batch(batch.num_edge_per_graph + batch.num_node_per_graph)
        batch.batch = batch.batch_simplex
        if self.use_local_neighbors and self.dense:
            batch.batch_attn_bias = num2batch((batch.num_edge_per_graph + batch.num_node_per_graph) ** 2)
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


class PrePooling(torch.nn.Module):
    """
    pooling layer for different orders of simplices

    """
    def __init__(self):
        super(PrePooling, self).__init__()
        self.pooling_type = cfg.model.get("simplex_pooling", 'all')
        assert self.pooling_type in ['all', 'node']

    def forward(self, batch):
        if self.pooling_type == 'node':
            node_index = torch.zeros(batch.batch_simplex.shape, dtype=batch.batch_simplex.dtype, device=batch.batch_simplex.device)
            accumulate = 0
            for i in range(batch.num_graphs):
                node_index[accumulate: accumulate + batch.num_node_per_graph[i]] = 1
                accumulate += (batch.num_node_per_graph[i] + batch.num_edge_per_graph[i])
            batch.x = batch.x[node_index.bool()]
            batch.batch = batch.batch_original
        return batch



@register_network('kSimplicialTransformerSparse')
class kSimplicialTransformerModelSparse(torch.nn.Module):
    """
        Sparse k-simplicial transformer
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, dense=False)
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
            layers.append(Simplicial_01_sparse_Layer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                exp_edges_cfg=cfg.prep,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                num_layer_MPNN=cfg.gnn.num_layer_MPNN,
                similarity_type=cfg.gnn.similarity_type,
                inference_mode=cfg.gnn.inference_mode,
                mp_threshold=cfg.gnn.mp_threshold,
                force_undirected=cfg.gnn.force_undirected,
                focusing_factor=cfg.gt.focusing_factor
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.pre_pooling = PrePooling()

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('kSimplicialTransformerDense')
class kSimplicialTransformerModelDense(torch.nn.Module):
    """
        Dense k-simplicial transformer
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, dense=True)
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
            layers.append(Simplicial_01_dense_Layer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                exp_edges_cfg=cfg.prep,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                num_layer_MPNN=cfg.gnn.num_layer_MPNN,
                similarity_type=cfg.gnn.similarity_type,
                inference_mode=cfg.gnn.inference_mode,
                mp_threshold=cfg.gnn.mp_threshold,
                force_undirected=cfg.gnn.force_undirected,
                focusing_factor=cfg.gt.focusing_factor
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.pre_pooling = PrePooling()

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
