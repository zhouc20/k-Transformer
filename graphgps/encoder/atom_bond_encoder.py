import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

@register_node_encoder('Atom_custom')
class AtomEncoder_custom(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])
        batch.x = encoded_features
        return batch



@register_edge_encoder('Bond_custom')
class BondEncoder_custom(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](batch.edge_attr[:, i])
        batch.edge_attr = bond_embedding
        return batch