"""
SignNet https://arxiv.org/abs/2202.13013
based on https://github.com/cptq/SignNet-BasisNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.nn import GINConv, MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import unbatch
from typing import List, Callable, Optional


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 use_bn=True, dropout=0.5, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
                             2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1) # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(one, batch_index, dim=0, dim_size=batch_size,
                          reduce='add')  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        # print(f"     - mask: {mask.shape} {mask}")
        # print(f"     - num_nodes: {num_nodes}")
        # print(f"     - batched_num_nodes: {batched_num_nodes.shape} {batched_num_nodes}")
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


@register_node_encoder('SignNet')
class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_SignNet
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        model_type = pecfg.model  # Encoder NN model type for SignNet
        if model_type not in ['MLP', 'DeepSet']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        sign_inv_layers = pecfg.layers  # Num. layers in \phi GNN part
        rho_layers = pecfg.post_layers  # Num. layers in \rho MLP/DeepSet
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(f"SignNet PE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Sign invariant neural network.
        if self.model_type == 'MLP':
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg.phi_hidden_dim,
                out_channels=pecfg.phi_out_dim,
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        elif self.model_type == 'DeepSet':
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg.phi_hidden_dim,
                out_channels=pecfg.phi_out_dim,
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch):
        if not (hasattr(batch, 'eigvals_sn') and hasattr(batch, 'eigvecs_sn')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_SignNet.enable' to True")
        # eigvals = batch.eigvals_sn
        eigvecs = batch.eigvecs_sn

        # pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = eigvecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1

        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, batch.batch)  # (Num nodes) x (pos_enc_dim)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_SignNet = pos_enc
        return batch


class StableExpressivePE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, phi: nn.Module, psi_list: List[nn.Module]) -> None:
        super().__init__()
        self.phi = phi
        self.psi_list = nn.ModuleList(psi_list)

    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        Lambda = Lambda.unsqueeze(dim=2)   # [B, D_pe, 1]
#        Lambda = torch.cat([torch.cat([torch.cos(Lambda / 10000**(i/37)), torch.sin(Lambda / 10000**(i / 37))], dim=-1)
#                            for i in range(37)], dim=-1)
        Z = torch.stack([
            psi(Lambda).squeeze(dim=2)     # [B, D_pe]
            for psi in self.psi_list
        ], dim=2)                          # [B, D_pe, M]

        V_list = unbatch(V, batch, dim=0)   # [N_i, D_pe] * B
        Z_list = list(Z)                    # [D_pe, M] * B

        W_list = []                        # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):   # [N_i, D_pe] and [D_pe, M]
            V = V.unsqueeze(dim=0)         # [1, N_i, D_pe]
            Z = Z.permute(1, 0)            # [M, D_pe]
            Z = Z.diag_embed()             # [M, D_pe, D_pe]
            V_T = V.mT                     # [1, D_pe, N_i]
            W = V.matmul(Z).matmul(V_T)    # [M, N_i, N_i]
            # W = V.matmul(V_T).repeat([Z.size(0), 1, 1])
            W = W.permute(1, 2, 0)         # [N_i, N_i, M]
            W_list.append(W)

        return self.phi(W_list, edge_index)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims


class MLPs(nn.Module):
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool = True, activation: str = 'relu',
        dropout_prob: float = 0.0, norm_type: str = "batch"
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, use_bn, activation, dropout_prob, norm_type)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features


class MLPLayer(nn.Module):
    """
    Based on https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """
    fc: nn.Linear
    bn: Optional[nn.BatchNorm1d]
    activation: nn.Module
    dropout: nn.Dropout

    def __init__(self, in_dims: int, out_dims: int, use_bn: bool, activation: str,
                 dropout_prob: float, norm_type: str = "batch") -> None:
        super().__init__()
        # self.fc = nn.Linear(in_dims, out_dims, bias=not use_bn)
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dims) if norm_type == "batch" else nn.LayerNorm(out_dims)
        else:
            self.bn = None
        # self.bn = nn.BatchNorm1d(out_dims) if use_bn else None
        # self.ln = nn.LayerNorm(out_dims) if use_bn else None

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        X = self.fc(X)                     # [***, D_out]
        # import logging
        # logging.info(X.shape)
        if self.bn is not None:
#            if X.ndim == 3:
#                # X = self.bn(X.transpose(2, 1)).transpose(2, 1)
#                X = self.ln(X)
#            else:
#                X = self.bn(X)
            shape = X.size()
            X = X.reshape(-1, shape[-1])   # [prod(***), D_out]
            X = self.bn(X)                 # [prod(***), D_out]
            X = X.reshape(shape)           # [***, D_out]
        X = self.activation(X)             # [***, D_out]
#        if self.bn is not None:
#            if X.ndim == 3:
#                X = self.bn(X.transpose(2, 1)).transpose(2, 1)
#            else:
#                X = self.bn(X)
        X = self.dropout(X)                # [***, D_out]
        return X


class DeepSets(nn.Module):
    layers: nn.ModuleList

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = DeepSetsLayer(in_dims, hidden_dims, activation)
            self.layers.append(layer)
            in_dims = hidden_dims

        # layer = DeepSetsLayer(hidden_dims, out_dims, activation='id') # drop last activation
        layer = DeepSetsLayer(hidden_dims, out_dims, activation)
        self.layers.append(layer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :return: Output feature matrix. [B, N, D_out]
        """
        for layer in self.layers:
            X = layer(X)   # [B, N, D_hid] or [B, N, D_out]
        return X           # [B, N, D_out]


class DeepSetsLayer(nn.Module):
    fc_one: nn.Linear
    fc_all: nn.Linear
    activation: nn.Module

    def __init__(self, in_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()
        self.fc_curr = nn.Linear(in_dims, out_dims, bias=True)
        self.fc_all = nn.Linear(in_dims, out_dims, bias=False)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :return: Output feature matrix. [B, N, D_out]
        """
        Z_curr = self.fc_curr(X)                          # [B, N, D_out]
        Z_all = self.fc_all(X.sum(dim=1, keepdim=True))   # [B, 1, D_out]
        # Z_all = self.fc_all(X.max(dim=1, keepdim=True)[0])   # [B, 1, D_out]
        X = Z_curr + Z_all                                # [B, N, D_out]
        return self.activation(X)                         # [B, N, D_out]


class MaskedDeepSets(nn.Module):
    layers: nn.ModuleList

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MaskedDeepSetsLayer(in_dims, hidden_dims, activation)
            self.layers.append(layer)
            in_dims = hidden_dims

        layer = MaskedDeepSetsLayer(hidden_dims, out_dims, activation)
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :param mask: Fake node masking. [B, N, 1]
        :return: Output feature matrix. [B, N, D_out]
        """
        for layer in self.layers:
            X = layer(X, mask)   # [B, N, D_hid] or [B, N, D_out]
        return X           # [B, N, D_out]


class MaskedDeepSetsLayer(nn.Module):
    fc_one: nn.Linear
    fc_all: nn.Linear
    activation: nn.Module

    def __init__(self, in_dims: int, out_dims: int, activation: str) -> None:
        super().__init__()
        self.fc_curr = nn.Linear(in_dims, out_dims, bias=True)
        self.fc_all = nn.Linear(in_dims, out_dims, bias=False)

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'id':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [B, N, D_in]
        :param mask: Fake node masking. [B, N, 1]
        :return: Output feature matrix. [B, N, D_out]
        """
        Z_curr = self.fc_curr(X)                          # [B, N, D_out]
        Z_all = self.fc_all((X * mask).sum(dim=1, keepdim=True))   # [B, 1, D_out]
        X = Z_curr + Z_all                                # [B, N, D_out]
        return self.activation(X)                         # [B, N, D_out]


class GINs(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
            bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(MLPs(2, in_dims, hidden_dims, hidden_dims))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = GINLayer(MLPs(2, hidden_dims, hidden_dims, out_dims))
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X.ndim == 3:
                    X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                else:
                    X = self.batch_norms[i](X)
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLPs

    def __init__(self, mlp: MLPs) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims


@register_node_encoder('StableExpressivePE')
class StableExpressivePENodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_StableExpressivePE
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        model_type = pecfg.model  # Encoder NN model type for SignNet
        n_psi = pecfg.get("n_psi", 6)
        if model_type not in ['MLP', 'DeepSet']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        n_psi_layers = pecfg.get('layers', 4)  # Num. layers in \psi MLP/DeepSet
        rho_layers = pecfg.get('post_layers', 8)  # Num. layers in \phi GNN part
        psi_hidden_dim = pecfg.get('psi_hidden_dim', 16)
        phi_hidden_dim = pecfg.get('phi_hidden_dim', 128)
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        self.D_pe = max_freqs
        assert dim_pe == max_freqs
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(f"SignNet PE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Sign invariant neural network.
        self.phi = GINPhi(n_layers=pecfg.post_layers, in_dims=n_psi, hidden_dims=phi_hidden_dim, out_dims=dim_pe)

        if self.model_type == 'DeepSet':
            psi_list = [DeepSets(n_psi_layers, 1, psi_hidden_dim, 1, "relu") for _ in range(n_psi)]
        elif self.model_type == 'MaskedDeepSet':
            psi_list = [MaskedDeepSets(n_psi_layers, 1, psi_hidden_dim, 1, "relu") for _ in range(n_psi)]
        elif self.model_type == 'MLP':
            psi_list = [MLP(1, psi_hidden_dim, 1, n_psi_layers) for _ in range(n_psi)]
        else:
            raise NotImplementedError
        self.psi_list = nn.ModuleList(psi_list)
        # if self.model_type == 'MLP':
        #     self.sign_inv_net = GINDeepSigns(
        #         in_channels=1,
        #         hidden_channels=pecfg.phi_hidden_dim,
        #         out_channels=pecfg.phi_out_dim,
        #         num_layers=sign_inv_layers,
        #         k=max_freqs,
        #         dim_pe=dim_pe,
        #         rho_num_layers=rho_layers,
        #         use_bn=True,
        #         dropout=0.0,
        #         activation='relu'
        #     )
        # elif self.model_type == 'DeepSet':
        #     self.sign_inv_net = MaskedGINDeepSigns(
        #         in_channels=1,
        #         hidden_channels=pecfg.phi_hidden_dim,
        #         out_channels=pecfg.phi_out_dim,
        #         num_layers=sign_inv_layers,
        #         dim_pe=dim_pe,
        #         rho_num_layers=rho_layers,
        #         use_bn=True,
        #         dropout=0.0,
        #         activation='relu'
        #     )
        # else:
        #     raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch):
        if not (hasattr(batch, 'eigvals_sn') and hasattr(batch, 'eigvecs_sn')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_SignNet.enable' to True")
        """
                :param Lambda: Eigenvalue vectors. [B, D_pe]
                :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
                :param edge_index: Graph connectivity in COO format. [2, E_sum]
                :param batch: Batch index vector. [N_sum]
                :return: Positional encoding matrix. [N_sum, D_pe]
                """
        # TODO: reshape lambda
        eigvals = batch.eigvals_sn
        eigvecs = batch.eigvecs_sn
        V = eigvecs
        Lambda = eigvals.reshape(batch.num_graphs, self.D_pe)
        Lambda = Lambda.unsqueeze(dim=2)  # [B, D_pe, 1]
        #        Lambda = torch.cat([torch.cat([torch.cos(Lambda / 10000**(i/37)), torch.sin(Lambda / 10000**(i / 37))], dim=-1)
        #                            for i in range(37)], dim=-1)
        Z = torch.stack([
            psi(Lambda).squeeze(dim=2)  # [B, D_pe]
            for psi in self.psi_list
        ], dim=2)  # [B, D_pe, M]

        V_list = unbatch(V, batch.batch, dim=0)  # [N_i, D_pe] * B
        Z_list = list(Z)  # [D_pe, M] * B

        W_list = []  # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):  # [N_i, D_pe] and [D_pe, M]
            V = V.unsqueeze(dim=0)  # [1, N_i, D_pe]
            Z = Z.permute(1, 0)  # [M, D_pe]
            Z = Z.diag_embed()  # [M, D_pe, D_pe]
            V_T = V.mT  # [1, D_pe, N_i]
            W = V.matmul(Z).matmul(V_T)  # [M, N_i, N_i]
            # W = V.matmul(V_T).repeat([Z.size(0), 1, 1])
            W = W.permute(1, 2, 0)  # [N_i, N_i, M]
            W_list.append(W)

        pos_enc = self.phi(W_list, batch.edge_index_original)  # [N_sum, D_pe]

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_SignNet = pos_enc
        return batch


class GINPhi(nn.Module):
    gin: GINs

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int) -> None:
        super().__init__()
        self.gin = GINs(n_layers, in_dims, hidden_dims, out_dims)
        # self.mlp = create_mlp(out_dims, out_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        # import logging
        # logging.info(W.shape)
        # logging.info(edge_index.shape)
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
        PE = self.gin(W, edge_index)       # [N_sum, N_max, D_pe]
        PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
        return PE               # [N_sum, D_pe]
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims

