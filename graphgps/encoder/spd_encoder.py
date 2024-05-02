'''
    The RRWP encoder for GRIT (ours)
'''
import torch
from torch import nn
from torch.nn import functional as F
from ogb.utils.features import get_bond_feature_dims
import torch_sparse

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from torch_scatter import scatter
import warnings

def full_edge_index(edge_index, batch=None, total_nodes=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        if total_nodes is None: total_nodes = edge_index.max().item() + 1
        batch = edge_index.new_zeros(total_nodes)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full




@register_edge_encoder('spd_emb')
class SPDEdgeEncoder(torch.nn.Module):
    '''
        Shortest-path-distance (SPD) Embedding Encoder
    '''
    def __init__(self, in_dim, out_dim,
                 batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True,
                 pe_name="spd",
                 pad_0th=False, # 0-th embdding is a fixed zero-vector; For the case that 0 indicate the padding.
                 overwrite_old_attr=False,
                 ):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pe_name = pe_name
        if pad_0th:
            self.spd_emb = nn.Embedding(in_dim+1, out_dim, padding_idx=0)
        else:
            self.spd_emb = nn.Embedding(in_dim, out_dim)

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.pad_to_full_graph = pad_to_full_graph
        self.overwrite_old_attr = overwrite_old_attr

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        k_idx, k_val = f'{self.pe_name}_index', f'{self.pe_name}_val'
        rel_idx = batch[k_idx].type(torch.long)
        rel_val = batch[k_val].type(torch.long)
        edge_index = batch.edge_index.type(torch.long)
        edge_attr = batch.get('edge_attr', None)
        
        rel_val = self.spd_emb(rel_val)

        if edge_attr is None:
            edge_attr = rel_val.new_full((edge_index.size(1), rel_val.size(1)), 0)
            # zero padding for non-existing edges

        # if self.overwrite_old_attr:
        out_idx, out_val = rel_idx, rel_val
        # else:
        #     out_idx, out_val = torch_sparse.coalesce(
        #         torch.cat([edge_index, rel_idx], dim=1),
        #         torch.cat([edge_attr, rel_val], dim=0),
        #         batch.num_nodes, batch.num_nodes,
        #         op="add"
        #     )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch, total_nodes=batch.num_nodes)
            edge_attr_pad = rel_val.new_full((edge_index_full.size(1), rel_val.size(1)), 0)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"{self.fc.__repr__()})"


# @register_edge_encoder('Hodge1Lap_emb')
# class Hodge1LapEdgeEncoder(torch.nn.Module):
#     '''
#         Merge RRWP with given edge-attr and Zero-padding to all pairs of node
#         FC_1(RRWP) + FC_2(edge-attr)
#         - FC_2 given by the TypedictEncoder in same cases
#         - Zero-padding for non-existing edges in fully-connected graph
#         - (optional) add node-attr as the E_{i,i}'s attr
#             note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     '''
#     def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  pad_to_full_graph=True, fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc = nn.Linear(1, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         self.pad_to_full_graph = pad_to_full_graph
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         Hodge_idx = batch.edge_index
#         Hodge_val = batch.pestat_HodgeLap1PE
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#         Hodge_val = self.fc(Hodge_val[:, 0])
#
#         if edge_attr is None:
#             edge_attr = torch.zeros(edge_index.size(1), Hodge_val.size(1))
#             # zero padding for non-existing edges
#
#         if self.overwrite_old_attr:
#             out_idx, out_val = Hodge_idx, Hodge_val
#         else:
#             edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
#             out_idx, out_val = torch_sparse.coalesce(
#                 torch.cat([edge_index, Hodge_idx], dim=1),
#                 torch.cat([edge_attr, Hodge_val], dim=0),
#                 batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#
#
#         if self.pad_to_full_graph:
#             edge_index_full = full_edge_index(out_idx, batch=batch.batch)
#             edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
#             # zero padding to fully-connected graphs
#             out_idx = torch.cat([out_idx, edge_index_full], dim=1)
#             out_val = torch.cat([out_val, edge_attr_pad], dim=0)
#             out_idx, out_val = torch_sparse.coalesce(
#                out_idx, out_val, batch.num_nodes, batch.num_nodes,
#                op="add"
#             )
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}," \
#                f"{self.fc.__repr__()})"


@register_edge_encoder('Hodge1Lap_emb')
class Hodge1LapEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(1, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        Hodge_idx = batch.edge_index
        Hodge_val = batch.pestat_HodgeLap1PE
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        Hodge_val = self.fc(Hodge_val[:, 0].unsqueeze(1))

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), Hodge_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = Hodge_idx, Hodge_val
        else:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, Hodge_idx], dim=1),
                torch.cat([edge_attr, Hodge_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )


        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"


@register_edge_encoder('EdgeRWSE_emb')
class EdgeRWSEEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        Hodge_idx = batch.edge_index
        Hodge_val = batch.pestat_EdgeRWSE
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        Hodge_val = self.fc(Hodge_val)

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), Hodge_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = Hodge_idx, Hodge_val
        else:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, Hodge_idx], dim=1),
                torch.cat([edge_attr, Hodge_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )


        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"


@register_edge_encoder('RD_emb')
class RDEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(1, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        RD_idx = batch.RD_index
        RD_val = batch.RD_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        RD_val = self.fc(RD_val)

        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), RD_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = RD_idx, RD_val
        else:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, RD_idx], dim=1),
                torch.cat([edge_attr, RD_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )


        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"
