import torch
from torch import Tensor
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.exp_edge_fixer import ExpanderEdgeFixer
from graphgps.layer.k_transformer_layer import k_Transformer_Layer
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
        assert cfg.model.type == 'kTransformer'
        k = cfg.model.get('k', 2)
        if k == 2:
            self.tuple_encoder = Tuple2Encoder(cfg.gt.dim_hidden)
        elif k == 3:
            self.tuple_encoder = Tuple3Encoder(cfg.gt.dim_hidden)
        else:
            raise NotImplementedError

        if 'Exphormer' in cfg.gt.layer_type:
            self.exp_edge_fixer = ExpanderEdgeFixer(add_edge_index=cfg.prep.add_edge_index,
                                                    num_virt_node=cfg.prep.num_virt_node)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class Tuple2Encoder(torch.nn.Module):
    """
    Encoding node and edge features to 2-tuples

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, hid_dim):
        super(Tuple2Encoder, self).__init__()
        self.mode = cfg.gt.get('tuple_encoder_mode', 'add')
        assert self.mode in ['add', 'concat']
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hid_dim * 3, hid_dim * 3), torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(hid_dim * 3, hid_dim))

    def forward(self, batch):
        x1 = batch.x[batch.edge_index[0]]
        x2 = batch.x[batch.edge_index[1]]
        if self.mode == 'concat':
            x = torch.cat([x1, x2, batch.edge_attr], dim=1)
            # logging.info(x.shape)
            x = self.mlp(x)
        else:
            x = x1 + x2 + batch.edge_attr
        batch.x = x
        batch.batch_original = batch.batch.clone()
        batch.batch_tuple = num2batch(batch.num_edge_per_graph)
        batch.batch = batch.batch_tuple
        return batch


class Tuple3Encoder(torch.nn.Module):
    """
    Encoding node and edge features to 3-tuples

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, hid_dim):
        super(Tuple3Encoder, self).__init__()
        self.mode = cfg.gt.get('tuple_encoder_mode', 'add')
        assert self.mode in ['add', 'concat']
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hid_dim * 6, hid_dim * 6), torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(hid_dim * 6, hid_dim * 3), torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(hid_dim * 3, hid_dim))

    def forward(self, batch):
        x1 = batch.x[batch.tuple_index[0]]
        x2 = batch.x[batch.tuple_index[1]]
        x3 = batch.x[batch.tuple_index[2]]
        if self.mode == 'concat':
            batch.edge_attr = batch.edge_attr.reshape(batch.tuple_index.shape[1], batch.edge_attr.shape[1] * 3)
            x = torch.cat([x1, x2, x3, batch.edge_attr], dim=1)
            x = self.mlp(x)
        else:
            idx1 = torch.arange(batch.tuple_index.shape[1], dtype=torch.long, device=batch.x.device) * 3
            idx2 = idx1 + 1
            idx3 = idx2 + 1
            x = x1 + x2 + x3 + batch.edge_attr[idx1] + batch.edge_attr[idx2] + batch.edge_attr[idx3]
        batch.x = x
        batch.batch_original = batch.batch.clone()
        batch.batch_tuple = num2batch(batch.num_tuple_per_graph)
        batch.batch = batch.batch_tuple
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


@register_network('kTransformer')
class kTransformerModel(torch.nn.Module):
    """
        Order-k transformer.
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
            layers.append(k_Transformer_Layer(
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

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
