import numpy as np
import torch
from torch_sparse import SparseTensor
from torch.linalg import det
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# compute div(w), outflow is positive
def divergence(adj):
    outflow = torch.sum(adj, dim=1)
    inflow = torch.sum(adj, dim=0)
    div = outflow - inflow
    return div

def Delta_0(alpha, A):
    w_g = A * alpha.unsqueeze(0).repeat(A.shape[-1], 1) - A * alpha.unsqueeze(1).repeat(1, A.shape[-1])
    return w_g


# directly compute graph Laplacian L0 (0-order Hodge Laplacian)
def compute_graph_Laplacian_0(edge_index, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    A1 = (A + A.permute(1, 0)).bool().float()
    D = torch.diag_embed(torch.sum(A1, dim=0))
    L0 = D - A1
    return L0


# edge is directed
def compute_gradient(edge_index, edge_attr, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()

    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    div = divergence(adj)
    L0 = compute_graph_Laplacian_0(edge_index, Nm)
    alpha = torch.linalg.solve(L0, -div)
    w_g = Delta_0(alpha, A)
    return w_g


def extract_triangle(edge_index, Nm=None, directed=True):
    triangle_set = []
    if directed:
        A = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=torch.ones(edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        A = (A + A.permute(1, 0)).bool().float()
        edge_index, _ = dense_to_sparse(A)
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    for i in range(Nm):
        subset, sub_edge_idx, mapping, edge_mask = k_hop_subgraph(i, 1, edge_index, relabel_nodes=False)
        for j in range(sub_edge_idx.shape[1]):
            if sub_edge_idx[0, j] > i and sub_edge_idx[1, j] > i and sub_edge_idx[0, j] < sub_edge_idx[1, j]:
                tri_node, _ = torch.sort(torch.tensor([i, sub_edge_idx[0, j], sub_edge_idx[1, j]]))
                triangle_set.append(tri_node)
    return triangle_set


def delta_1_delta_1_star(triangle_set):
    d1d1s = torch.diag_embed(torch.ones(len(triangle_set)) * 3.)
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        pc = [torch.tensor([c[0], c[1]]), torch.tensor([c[1], c[2]]), torch.tensor([c[2], c[0]])]
        nc = [torch.tensor([c[1], c[0]]), torch.tensor([c[2], c[1]]), torch.tensor([c[0], c[2]])]
        for j in range(i+1, len(triangle_set)):
            a = triangle_set[j]
            pa = [torch.tensor([a[0], a[1]]), torch.tensor([a[1], a[2]]), torch.tensor([a[2], a[0]])]
            na = [torch.tensor([a[1], a[0]]), torch.tensor([a[2], a[1]]), torch.tensor([a[0], a[2]])]
            for x in pc:
                for y in pa:
                    if torch.equal(x, y):
                        d1d1s[i, j] = 1
                        d1d1s[j, i] = 1
                        break
                for y in na:
                    if torch.equal(x, y):
                        d1d1s[i, j] = -1
                        d1d1s[j, i] = -1
                        break
    return d1d1s


def curl(triangle_set, edge_index, edge_attr, Nm=None):
    curl_ls = []
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    adj = adj - adj.permute(1, 0)

    for tri in triangle_set:
        accumulate_curl = 0
        accumulate_curl += adj[tri[0], tri[1]]
        accumulate_curl += adj[tri[1], tri[2]]
        accumulate_curl += adj[tri[2], tri[0]]
        curl_ls.append(accumulate_curl)
    return torch.tensor(curl_ls)


def delta_1_star(gamma, triangle_set, edge_index, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = torch.zeros(Nm, Nm)
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        adj[c[0], c[1]] += gamma[i]
        adj[c[1], c[0]] -= gamma[i]
        adj[c[1], c[2]] += gamma[i]
        adj[c[2], c[1]] -= gamma[i]
        adj[c[2], c[0]] += gamma[i]
        adj[c[0], c[2]] -= gamma[i]
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    return A * adj



def compute_solenoidal(edge_index, edge_attr, Nm=None, directed=True):
    triangle_set = extract_triangle(edge_index, Nm, directed)
    d1d1s = delta_1_delta_1_star(triangle_set)
    curls = curl(triangle_set, edge_index, edge_attr, Nm)
    gamma = torch.linalg.solve(d1d1s, curls)
    w_s = delta_1_star(gamma, triangle_set, edge_index, Nm)
    return w_s


def Hodge_decomposition(edge_index, edge_attr, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    w_g = compute_gradient(edge_index, edge_attr, Nm)
    w_s = compute_solenoidal(edge_index, edge_attr, Nm, directed)
    w_h = adj - w_s - w_g
    return w_g, w_s, w_h


def compute_Hodge_0_Laplacian(edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    if not directed:
        A = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=torch.ones(edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        # convert to standard directed graph
        # this won't affect result for directed graph
        # but should be used for undirected graphs
        A = (A + A.permute(1, 0)).bool().float()
        mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
        A = A * mask
        edge_index, _ = dense_to_sparse(A)

    B1 = torch.zeros([Nm, edge_index.shape[1]], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        B1[edge_index[0, i], i] = -1
        B1[edge_index[1, i], i] = 1

    return torch.matmul(B1, B1.permute(1, 0))


def compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    # convert to standard directed graph
    # this won't affect result for directed graph
    # but should be used for undirected graphs, and convenient for computing B2
    A = (A + A.permute(1, 0)).bool().float()
    mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
    A = A * mask
    edge_index, _ = dense_to_sparse(A)
    print(edge_index)

    B1 = torch.zeros([Nm, edge_index.shape[1]], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        B1[edge_index[0, i], i] = -1
        B1[edge_index[1, i], i] = 1

    # s1 = torch.matmul(B1.permute(1, 0), B1)
    # print(s1)
    # eigenvalue, eigenvector = torch.linalg.eig(s1)
    # eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    # eigenvalue = torch.view_as_real(eigenvalue)[:, 0]
    # print(eigenvalue)
    # print(eigenvector)
    # B1[:, 1] = B1[:, 1] * (-1)
    # s1 = torch.matmul(B1.permute(1, 0), B1)
    # print(s1)
    # eigenvalue, eigenvector = torch.linalg.eig(s1)
    # eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    # eigenvalue = torch.view_as_real(eigenvalue)[:, 0]
    # print(eigenvalue)
    # print(eigenvector)

    triangle_set = extract_triangle(edge_index, Nm, True)
    print(triangle_set)
    B2 = torch.zeros([edge_index.shape[1], len(triangle_set)])
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[0] and edge_index[1, j] == c[1]:
                B2[j, i] = 1
                break
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[1] and edge_index[1, j] == c[2]:
                B2[j, i] = 1
                break
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[0] and edge_index[1, j] == c[2]:
                B2[j, i] = -1
                break
    return torch.matmul(B1.permute(1, 0), B1) + torch.matmul(B2, B2.permute(1, 0))


def display_L1_eigen(index, edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    G = nx.Graph()
    G.clear()
    G.add_nodes_from(np.arange(0, Nm))
    edge_list = []
    for j in range(edge_index.shape[1]):
        # print((data.edge_index[0][j], data.edge_index[1][j]))
        edge_list.append((edge_index[0][j].item(), edge_index[1][j].item()))
    # print(edge_list)
    G.add_edges_from(edge_list)

    L1 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm, directed)
    eigenvalue, eigenvector = torch.linalg.eig(L1)
    eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    eigenvalue = torch.view_as_real(eigenvalue)[:, 0]

    # cmap = plt.cm.get_cmap('Blues')
    cmap = matplotlib.colormaps['Blues']
    edge_vmin, edge_vmax = 0, torch.max(eigenvector).item() * 0.9
    print(eigenvalue, eigenvector)
    for i in range(edge_index.shape[1]):
        print(eigenvalue[i], eigenvector[:, i])
        plt.figure(i, figsize=(20, 20))
        nx.draw_networkx(G, node_size=300, edge_color=np.abs(eigenvector[:, i].numpy()), width=5.0, edge_cmap=cmap, edge_vmin=edge_vmin, edge_vmax=edge_vmax)
        plt.title(str(index) + '_' + str(eigenvalue[i].item()))
        plt.savefig('graph_figure/' + str(index) + '_' + str(eigenvalue[i].item()) + '.png')


def edge_random_walk(edge_index, Nm):  # This is the edge_down version
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    A = (A + A.permute(1, 0)).bool().float()
    mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
    A = A * mask
    directed_edge_index, _ = dense_to_sparse(A)
    B1 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
    for i in range(directed_edge_index.shape[1]):
        B1[directed_edge_index[0, i], i] = 1
        B1[directed_edge_index[1, i], i] = 1
    P_down = torch.matmul(B1.permute(1, 0), B1)
    P_down = P_down - torch.eye(directed_edge_index.shape[1]) * 2.
    # B = torch.matmul(torch.diag_embed(1. / torch.sum(B, dim=0)), B)
    for j in range(P_down.shape[0]):
        if torch.sum(P_down[j]) > 0:
            P_down[j] = P_down[j] / torch.sum(P_down[j])
    # print(P_down)

    triangle_set = extract_triangle(directed_edge_index, Nm, True)
    # print(directed_edge_index)
    # print(triangle_set)
    P_up = torch.zeros([directed_edge_index.shape[1], directed_edge_index.shape[1]], dtype=torch.float)
    for triangle in triangle_set:
        edge0 = torch.index_select(triangle, 0, torch.tensor([0, 1]))
        edge1 = torch.index_select(triangle, 0, torch.tensor([0, 2]))
        edge2 = torch.index_select(triangle, 0, torch.tensor([1, 2]))
        # print(edge0, edge1, edge2)
        idx0, idx1, idx2 = -1, -1, -1
        for i in range(directed_edge_index.shape[1]):
            if directed_edge_index[0, i] == edge0[0] and directed_edge_index[1, i] == edge0[1]:
                idx0 = i
            if directed_edge_index[0, i] == edge1[0] and directed_edge_index[1, i] == edge1[1]:
                idx1 = i
            if directed_edge_index[0, i] == edge2[0] and directed_edge_index[1, i] == edge2[1]:
                idx2 = i
        # print(idx0, idx1, idx2)
        P_up[idx0, idx1] = 1
        P_up[idx0, idx2] = 1
        P_up[idx1, idx0] = 1
        P_up[idx1, idx2] = 1
        P_up[idx2, idx0] = 1
        P_up[idx2, idx1] = 1
    # print(P_up[0])
    for j in range(P_up.shape[0]):
        if torch.sum(P_up[j]) > 0:
            P_up[j] = P_up[j] / torch.sum(P_up[j])
    P = 0.5 * (P_up + P_down)


    Bk = torch.eye(directed_edge_index.shape[1])
    prob = []
    for walks in range(40):
        Bk = torch.matmul(P_up, Bk)
        prob.append(torch.diagonal(Bk).unsqueeze(0))
    prob = torch.cat(prob, dim=0).permute(1, 0)
    print(prob[0])


if __name__ == '__main__':
    # edge_index = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    #                            [2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]])
    # edge_index = edge_index - 1
    # edge_index[1, :26] += 8
    # edge_index[0, 26:] += 8
    # # edge_attr = torch.tensor([-1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, -1, -2, -1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9,
    # #      4.1, -1, -2, 3.1, 5.1, 23.9, 13.1, 13.1, 27.9, -3, 9.3])
    # edge_attr = torch.tensor([1, 4.2, 2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, 1, 2, 1, 4.2, 2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1,
    #  1, 2, 5.1, 11.1, 27.9, 13.1, 13.1, 29.9, 3, 13.3])
    #
    # Nm = 16

    # example in Tutorial
    # edge_index = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7],
    #                            [2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8]])
    # edge_index = edge_index - 1
    # edge_attr = torch.tensor(
    #     [-1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, -1, -2])
    # Nm = 8

    # example in Hodge Laplacian
    # edge_index = torch.tensor([[1, 2, 3, 3, 3, 4, 5],
    #                            [2, 3, 4, 5, 6, 1, 6]])
    # edge_index -= 1
    # print(edge_index)
    # Nm = 6
    # directed_edge_index = edge_index

    # edge_index = torch.tensor([[0, 1, 2, 3, 4, 4, 5, 6, 6, 8, 9, 9, 10, 11, 11],
    #                            [1, 2, 3, 4, 5, 14,6, 7, 8, 9, 10,14,11, 12, 13]])
    # Nm = 15
    #
    # A = SparseTensor(row=edge_index[0],
    #                    col=edge_index[1],
    #                    value=torch.ones(edge_index.shape[1], dtype=torch.float),
    #                    sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    # A = (A + A.permute(1, 0)).bool().float()
    # mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
    # A = A * mask
    # directed_edge_index, _ = dense_to_sparse(A)
    # print(directed_edge_index)
    #
    # B1 = torch.zeros([directed_edge_index.shape[1], Nm], dtype=torch.float)
    # B2 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
    # for i in range(directed_edge_index.shape[1]):
    #     B1[i, directed_edge_index[1, i]] = 1
    #     B2[directed_edge_index[0, i], i] = 1
    # P = torch.matmul(B1, B2)
    # # print(P)
    # for j in range(P.shape[0]):
    #     if torch.sum(P[j]) > 0:
    #         P[j] = P[j] / torch.sum(P[j])
    #
    # Bk = torch.eye(edge_index.shape[1])
    # for i in range(5):
    #     Bk = torch.matmul(Bk, P)
    #     print(Bk)

    # B1 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
    # for i in range(directed_edge_index.shape[1]):
    #     B1[directed_edge_index[0, i], i] = 1
    #     B1[directed_edge_index[1, i], i] = 1
    # print(B1)
    # delta_1 = B1 / 2
    # delta_1_s = B1
    # for i in range(Nm):
    #     if int(torch.sum(B1[i])) <= 1:
    #         pass
    #     else:
    #         delta_1_s[i] = delta_1_s[i] / (torch.sum(delta_1_s[i]) - 1)
    # B = torch.matmul(delta_1.permute(1, 0), delta_1_s)
    # B = B - torch.diag_embed(torch.diagonal(B))
    # # B = torch.matmul(B1.permute(1, 0), B1)
    # # B = B - torch.eye(Nm) * 2.
    # B = torch.matmul(torch.diag_embed(1 / torch.sum(B, dim=0)), B)
    # print(B)
    # prob = B
    # idx = 0
    #
    # adj = (A + A.permute(1, 0)).bool().float()
    # undir_edge_index, _ = dense_to_sparse(adj)
    # print(undir_edge_index)
    # prob_undirected = torch.zeros([undir_edge_index.shape[1], directed_edge_index.shape[1]], dtype=torch.float)
    # for i in range(undir_edge_index.shape[1]):
    #     if undir_edge_index[0, i] < undir_edge_index[1, i]:
    #         prob_undirected[i] = prob[idx]
    #         idx += 1
    #     else:
    #         for j in range(i):
    #             if undir_edge_index[0, j] == undir_edge_index[1, i] and undir_edge_index[1, j] == \
    #                     undir_edge_index[0, i]:
    #                 prob_undirected[i] = prob_undirected[j]
    #                 break
    # print(prob_undirected)

    # Bk = torch.eye(7)
    # prob = []
    # for i in range(100):
    #     print(i+1)
    #     Bk = torch.matmul(B, Bk)
    #     prob.append(torch.diagonal(Bk).unsqueeze(0))
    #     if i % 5 == 0:
    #         print(Bk)
    # prob = torch.cat(prob, dim=0).permute(1, 0)
    # print(prob)

    # edge_index = torch.tensor([[1, 2, 3, 3, 4, 4, 6],
    #                            [2, 3, 4, 5, 1, 6, 2]])
    # edge_index -= 1
    # Nm = 6

    # edge_index = torch.tensor([[0, 1, 2],
    #                            [1, 2, 0]])
    # edge_attr = torch.tensor([1., 1.1, 0.9])
    # Nm = 3

    # example zinc molecular graph 1002
    # index = 1002
    # edge_index = torch.tensor([[0, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 11, 13, 13, 14, 15, 16, 17, 19],
    #                            [1, 2, 3, 20,4, 5, 6, 19,7, 10,8, 9, 10,11, 12, 13, 14, 18, 15, 16, 17, 18, 20]])
    # Nm = 21

    # example zinc molecular graph 798
    # index = 798
    # edge_index = torch.tensor([[0, 1, 1,  2, 3, 3, 5, 6, 6, 7, 8, 9, 9, 11, 11, 12, 12, 16, 16, 17, 18, 19, 19, 20, 21, 22, 23],
    #                            [1, 2, 16, 3, 4, 5, 6, 7, 15,8, 9, 10,11,12, 15, 13, 14, 17, 24, 18, 19, 20, 24, 21, 22, 23, 24]])
    # Nm = 25

    # example zinc molecular graph 488
    # index = 488
    # edge_index = torch.tensor([[0, 1, 2, 3, 4, 4, 5, 6, 6, 8, 9, 9, 10, 11, 11],
    #                            [1, 2, 3, 4, 5, 14,6, 7, 8, 9, 10,14,11, 12, 13]])
    # Nm = 15
    # w_g, w_s, w_h = Hodge_decomposition(edge_index, edge_attr, Nm, True)
    # print(w_g[:8, 8:])
    # print(w_s[:8, 8:])
    # print(w_h[:8, 8:])
    # print((w_g + w_h + w_s)[:8, 8:])
    # L0 = compute_graph_Laplacian_0(edge_index, Nm)
    # print(L0)
    # L0 = compute_Hodge_0_Laplacian(edge_index, Nm)
    # print(L0)

    # L1 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm, True)
    # print(L1)
    # eigenvalue, eigenvector = torch.linalg.eigh(L1)
    # # eigenvalue = torch.view_as_real(eigenvalue)[:, 0]
    # # eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    # print(eigenvalue, eigenvector)
    # for i in range(15):
    #     print(eigenvalue[i], eigenvector[:, i])
    # display_L1_eigen(index, edge_index, Nm, True)
    #
    # values, indices = torch.sort(eigenvalue, descending=False)
    # max_low_freqs = [3, 6, 9, 12, 15]
    # figure_num = 0
    # for max_low_freq in max_low_freqs:
    #     figure_num += 1
    #     accumulate_num = 0
    #     lowest_vectors = []
    #     for i in range(values.shape[0]):
    #         print(values[i])
    #         print(eigenvector[:, indices[i]])
    #         if values[i] > 1e-4:
    #             lowest_vectors.append(eigenvector[:, indices[i]].unsqueeze(0))
    #             accumulate_num += 1
    #         if accumulate_num == max_low_freq:
    #             break
    #
    #     lowest_vectors = torch.cat(lowest_vectors, dim=0)
    #     proj_low = torch.matmul(lowest_vectors.permute(1, 0), lowest_vectors)
    #     print(proj_low)
    #     proj_low = torch.abs(proj_low)
    #     unit = torch.ones([edge_index.shape[1], 1], dtype=torch.float)
    #     unit = unit / torch.sqrt(torch.tensor(edge_index.shape[1], dtype=torch.float))
    #     weight = torch.matmul(torch.abs(proj_low), unit).reshape(-1)
    #
    # zero_vec = []
    # for i in range(len(eigenvalue)):
    #     if torch.abs(eigenvalue[i]) < 1e-4:
    #         zero_vec.append(eigenvector[:, i].unsqueeze(0))
    # zero_vecs = torch.cat(zero_vec, dim=0)
    # zero_vecs = zero_vecs.permute(1, 0)  # (num_edge, num_zero_eval)
    # proj = torch.matmul(zero_vecs, zero_vecs.permute(1, 0))
    # print(zero_vecs)
    # # print(proj)
    # proj = torch.abs(proj)
    # print(proj)
    # unit = torch.ones([edge_index.shape[1], 1], dtype=torch.float)
    # # unit = unit / torch.sqrt(torch.tensor(edge_index.shape[1], dtype=torch.float))
    # weight = torch.matmul(proj, unit).reshape(-1)
    # print(edge_index)
    # print(weight)
    #     G = nx.Graph()
    #     G.clear()
    #     G.add_nodes_from(np.arange(0, Nm))
    #     edge_list = []
    #     for j in range(edge_index.shape[1]):
    #         # print((data.edge_index[0][j], data.edge_index[1][j]))
    #         edge_list.append((edge_index[0][j].item(), edge_index[1][j].item()))
    #     # print(edge_list)
    #     G.add_edges_from(edge_list)
    #     plt.figure(figure_num, figsize=(20, 20))
    #     cmap = matplotlib.colormaps['Blues']
    #     edge_vmin, edge_vmax = 0, 1
    #     nx.draw_networkx(G, node_size=300, edge_color=np.abs(weight.numpy()), width=5.0, edge_cmap=cmap, edge_vmin=edge_vmin, edge_vmax=edge_vmax)
    #     plt.title(str(index) + '_projector')
    #     plt.savefig('graph_figure/' + str(index) + '_low_abs_' + str(max_low_freq) + '_projector.png')

    # 4*4 rook
    adj1 = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                         [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                         [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0]])
    # Shrikhande
    adj2 = torch.tensor([[0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                         [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                         [1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                         [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                         [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                         [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0]])

    edge_index_1 = dense_to_sparse(adj1)[0]
    edge_index_2 = dense_to_sparse(adj2)[0]
    Nm = 16
    edge_random_walk(edge_index_1, Nm)
    edge_random_walk(edge_index_2, Nm)
    L1 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index_1, 16, False)
    eigenvalue1, eigenvector1 = torch.linalg.eigh(L1)
    # print(eigenvalue1, eigenvector1)
    # eigenvector1 = torch.view_as_real(eigenvector1)[:, :, 0]
    # eigenvalue1 = torch.view_as_real(eigenvalue1)[:, 0]
    values1, indices1 = torch.sort(eigenvalue1, descending=True)
    # print('L1', values1)
    # print(eigenvalue1)
    # print(eigenvector1)
    L2 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index_2, 16, False)
    eigenvalue2, eigenvector2 = torch.linalg.eigh(L2)
    # print(eigenvalue2, eigenvector2)
    # eigenvector2 = torch.view_as_real(eigenvector2)[:, :, 0]
    # eigenvalue2 = torch.view_as_real(eigenvalue2)[:, 0]
    values2, indices2 = torch.sort(eigenvalue2, descending=True)
    # print('L2', values2)
    # L10 = compute_Hodge_0_Laplacian(edge_index_1, 16, False)
    # eigenvalue10, eigenvector10 = torch.linalg.eig(L10)
    # eigenvector10 = torch.view_as_real(eigenvector10)[:, :, 0]
    # eigenvalue10 = torch.view_as_real(eigenvalue10)[:, 0]
    # values10, indices10 = torch.sort(eigenvalue10, descending=True)
    # print('L10', values10)
    # L20 = compute_Hodge_0_Laplacian(edge_index_2, 16, False)
    # eigenvalue20, eigenvector20 = torch.linalg.eig(L20)
    # eigenvector20 = torch.view_as_real(eigenvector20)[:, :, 0]
    # eigenvalue20 = torch.view_as_real(eigenvalue20)[:, 0]
    # values20, indices20 = torch.sort(eigenvalue20, descending=True)
    # print('L20', values20)
    # # print(eigenvalue2)
    # # print(eigenvector2)
    # adj3 = torch.tensor([[0, 1, 1, 0, 0, 0],
    #                      [1, 0, 1, 0, 0, 0],
    #                      [1, 1, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 1, 1],
    #                      [0, 0, 0, 1, 0, 1],
    #                      [0, 0, 0, 1, 1, 0]])
    adj3 = torch.tensor([[0, 1, 0, 0, 0, 1],
                         [1, 0, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0],
                         [0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0, 1],
                         [1, 0, 0, 0, 1, 0]])
    adj4 = torch.tensor([[0, 1, 0, 0, 0, 1],
                         [1, 0, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0],
                         [0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0],
                         [1, 0, 0, 0, 0, 0]])
    # adj4 = torch.tensor([[0, 1, 1, 0, 0, 0],
    #                      [1, 0, 1, 0, 0, 0],
    #                      [1, 1, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 1, 1],
    #                      [0, 0, 0, 1, 0, 0],
    #                      [0, 0, 0, 1, 0, 0]])
    edge_index_3 = dense_to_sparse(adj3)[0]
    edge_index_4 = dense_to_sparse(adj4)[0]
    L3 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index_3, 6, False)
    print('L3', L3)
    eigenvalue3, eigenvector3 = torch.linalg.eigh(L3)
    # print(eigenvalue3, eigenvector3)
    # eigenvector3 = torch.view_as_real(eigenvector3)[:, :, 0]
    # eigenvalue3 = torch.view_as_real(eigenvalue3)[:, 0]
    values3, indices3 = torch.sort(eigenvalue3, descending=True)
    print('L3', values3)
    L4 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index_4, 6, False)
    print('L4', L4)
    eigenvalue4, eigenvector4 = torch.linalg.eigh(L4)
    # print(eigenvalue4, eigenvector4)
    # eigenvector4 = torch.view_as_real(eigenvector4)[:, :, 0]
    # eigenvalue4 = torch.view_as_real(eigenvalue4)[:, 0]
    values4, indices4 = torch.sort(eigenvalue4, descending=True)
    print('L4', values4)
    for i in range(10):
        x = torch.randn([6, 4]) * 10
        A3 = L3 + torch.nn.ReLU()(torch.einsum('ad, bd -> ab', x, x))
        A4 = L4 + torch.nn.ReLU()(torch.einsum('ad, bd -> ab', x[:5], x[:5]))
        eigenvalue3, eigenvector3 = torch.linalg.eigh(A3)
        values3, indices3 = torch.sort(eigenvalue3, descending=True)
        print('A3', A3, values3)
        # print('A3', values3)
        eigenvalue4, eigenvector4 = torch.linalg.eigh(A4)
        values4, indices4 = torch.sort(eigenvalue4, descending=True)
        print('A4', A4, values4)
    # print(torch.matmul(eigenvector4, eigenvector4.T))
    # print(torch.matmul(eigenvector4.T, eigenvector4))

    # L30 = compute_Hodge_0_Laplacian(edge_index_3, 6, False)
    # eigenvalue30, eigenvector30 = torch.linalg.eig(L30)
    # eigenvector30 = torch.view_as_real(eigenvector30)[:, :, 0]
    # eigenvalue30 = torch.view_as_real(eigenvalue30)[:, 0]
    # values30, indices30 = torch.sort(eigenvalue30, descending=True)
    # print('L30', values30)
    # L40 = compute_Hodge_0_Laplacian(edge_index_4, 6, False)
    # eigenvalue40, eigenvector40 = torch.linalg.eig(L40)
    # eigenvector40 = torch.view_as_real(eigenvector40)[:, :, 0]
    # eigenvalue40 = torch.view_as_real(eigenvalue40)[:, 0]
    # values40, indices40 = torch.sort(eigenvalue40, descending=True)
    # print('L40', values40)

    # Nm = 28
    # edge_index_1 = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    #                              [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,5, 6, 4, 8, 9, 7, 11, 12, 10, 14, 15, 13, 17, 18, 19, 20, 21, 16, 23, 24, 25, 26, 27, 22]])
    # edge_index_2 = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    #                              [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,5, 6, 4, 8, 9, 7, 11, 12, 13, 14, 15, 10, 17, 18, 16, 20, 21, 19, 23, 24, 25, 26, 27, 22]])
    # A1 = SparseTensor(row=edge_index_1[0],
    #                  col=edge_index_1[1],
    #                  value=torch.ones(edge_index_1.shape[1], dtype=torch.float),
    #                  sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    # # convert to standard directed graph
    # # this won't affect result for directed graph
    # # but should be used for undirected graphs, and convenient for computing B2
    # A1 = (A1 + A1.permute(1, 0)).bool().float()
    # A1 = torch.matmul(torch.diag_embed(1. / torch.sum(A1, dim=0)), A1)
    # A2 = SparseTensor(row=edge_index_2[0],
    #                   col=edge_index_2[1],
    #                   value=torch.ones(edge_index_2.shape[1], dtype=torch.float),
    #                   sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    # # convert to standard directed graph
    # # this won't affect result for directed graph
    # # but should be used for undirected graphs, and convenient for computing B2
    # A2 = (A2 + A2.permute(1, 0)).bool().float()
    # A2 = torch.matmul(torch.diag_embed(1. / torch.sum(A2, dim=0)), A2)
    # P1 = torch.eye(Nm)
    # P2 = torch.eye(Nm)
    # SE1 = []
    # SE2 = []
    # for i in range(8):
    #     P1 = torch.matmul(A1, P1)
    #     P2 = torch.matmul(A2, P2)
    #     SE1.append(torch.diagonal(P1).unsqueeze(0))
    #     SE2.append(torch.diagonal(P2).unsqueeze(0))
    # prob1 = torch.cat(SE1, dim=0).permute(1, 0)
    # prob2 = torch.cat(SE2, dim=0).permute(1, 0)
    # print(prob1)
    # print(prob2)




