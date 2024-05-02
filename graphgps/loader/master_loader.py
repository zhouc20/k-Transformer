import logging
import os.path as osp
import time
from functools import partial

import math
import numpy as np
import copy
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from numpy.random import default_rng
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (Amazon, Coauthor, GNNBenchmarkDataset, Planetoid, TUDataset,
                                      WebKB, WikipediaNetwork, ZINC)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader
from scipy.sparse.csgraph import floyd_warshall

from graphgps.loader.planetoid import Planetoid
from graphgps.loader.dataset.aqsol_molecules import AQSOL
from graphgps.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgps.loader.dataset.malnet_tiny import MalNetTiny
from graphgps.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgps.loader.dataset.SRDataset import SRDataset
from graphgps.loader.dataset.GraphCountDataset import GraphCountDataset
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import (pre_transform_in_memory, generate_splits,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size, move_node_feat_to_x, extractsubset)
from graphgps.transform.expander_edges import generate_random_expander
from graphgps.transform.dist_transforms import (add_dist_features, add_reverse_edges,
                                                 add_self_loops, effective_resistances,
                                                 effective_resistance_embedding,
                                                 effective_resistances_from_embedding)
from graphgps.transform.hodge_decomposition import *


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)
        print(dataset_dir)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'Amazon':
            dataset = Amazon(dataset_dir, name)
            if name == "photo" or name == "computers":
                pre_transform_in_memory(dataset, partial(generate_splits, g_split = cfg.dataset.split[0]))
                pre_transform_in_memory(dataset, partial(add_reverse_edges))
                if cfg.prep.add_self_loops:
                  pre_transform_in_memory(dataset, partial(add_self_loops))

        elif pyg_dataset_id == 'Coauthor':
            dataset = Coauthor(dataset_dir, name)
            if name == "physics" or name == "cs":
                pre_transform_in_memory(dataset, partial(generate_splits, g_split = cfg.dataset.split[0]))
                pre_transform_in_memory(dataset, partial(add_reverse_edges))
                if cfg.prep.add_self_loops:
                  pre_transform_in_memory(dataset, partial(add_self_loops))

        elif pyg_dataset_id == 'Planetoid':
            # dataset = Planetoid(dataset_dir, name)
            dataset = Planetoid(dataset_dir, name, split='random', train_percent= cfg.prep.train_percent)
            # dataset = Planetoid(dataset_dir, name, split='random', num_train_per_class = 4725, num_val = 1575, num_test = 1575)
            # Citeseer
            # dataset = Planetoid(dataset_dir, name, split='random', num_train_per_class = 1996, num_val = 665, num_test = 666)
            if name == "PubMed":
                pre_transform_in_memory(dataset, partial(typecast_x, type_str='float'))
            if cfg.prep.add_reverse_edges == True:
              pre_transform_in_memory(dataset, partial(add_reverse_edges))
            if cfg.prep.add_self_loops == True:
              pre_transform_in_memory(dataset, partial(add_self_loops))


        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)


        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id == 'SR':
            dataset = preformat_SR(dataset_dir)

        elif pyg_dataset_id == 'GraphCount':
            dataset = preformat_GraphCount(dataset_dir)
            # return dataset

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            print(dataset_dir, name)
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('ogbn'):
            dataset = preformat_ogbn(dataset_dir, name)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable and (not key.startswith('posenc_ER')):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

        # Other preprocessings:
        # adding expander edges:
        if cfg.prep.exp:
            for j in range(cfg.prep.exp_count):
                start = time.perf_counter()
                logging.info(f"Adding expander edges (round {j}) ...")
                pre_transform_in_memory(dataset,
                                        partial(generate_random_expander,
                                                degree=cfg.prep.exp_deg,
                                                algorithm=cfg.prep.exp_algorithm,
                                                rng=None,
                                                max_num_iters=cfg.prep.exp_max_num_iters,
                                                exp_index=j),
                                        show_progress=True
                                        )
                elapsed = time.perf_counter() - start
                timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                          + f'{elapsed:.2f}'[-3:]
                logging.info(f"Done! Took {timestr}")

        # adding shortest path features
        if cfg.prep.dist_enable:
            start = time.perf_counter()
            logging.info(f"Precalculating node distances and shortest paths ...")
            is_undirected = dataset[0].is_undirected()
            Max_N = max([data.num_nodes for data in dataset])
            pre_transform_in_memory(dataset,
                                    partial(add_dist_features,
                                            max_n=Max_N,
                                            is_undirected=is_undirected,
                                            cutoff=cfg.prep.dist_cutoff),
                                    show_progress=True
                                    )
            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")

        # adding effective resistance features
        if cfg.posenc_ERN.enable or cfg.posenc_ERE.enable:
            start = time.perf_counter()
            logging.info(f"Precalculating effective resistance for graphs ...")

            MaxK = max(
                [
                    min(
                        math.ceil(data.num_nodes // 2),
                        math.ceil(8 * math.log(data.num_edges) / (cfg.posenc_ERN.accuracy ** 2))
                    )
                    for data in dataset
                ]
            )

            cfg.posenc_ERN.er_dim = MaxK
            logging.info(f"Choosing ER pos enc dim = {MaxK}")

            pre_transform_in_memory(dataset,
                                    partial(effective_resistance_embedding,
                                            MaxK=MaxK,
                                            accuracy=cfg.posenc_ERN.accuracy,
                                            which_method=0),
                                    show_progress=True
                                    )

            pre_transform_in_memory(dataset,
                                    partial(effective_resistances_from_embedding,
                                            normalize_per_node=False),
                                    show_progress=True
                                    )

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")

    if cfg.model.type == 'kTransformer':
        k = cfg.model.get('k', 2)
        if k == 2:
            pre_transform_in_memory(dataset,
                                    partial(pretransform_2tuple),
                                    show_progress=True)
        elif k == 3:
            pre_transform_in_memory(dataset,
                                    partial(pretransform_3tuple),
                                    show_progress=True)
        else:
            raise NotImplementedError

    if cfg.model.type == 'TensorizedTransformer':
        pre_transform_in_memory(dataset,
                                partial(pretransform_1_2tuple),
                                show_progress=True)

    if cfg.model.type == 'kSimplicialTransformerSparse':
        pre_transform_in_memory(dataset,
                                partial(pretransform_2simplex, dense=False),
                                show_progress=True)
    if cfg.model.type == 'kSimplicialTransformerDense':
        pre_transform_in_memory(dataset,
                                partial(pretransform_2simplex, dense=True),
                                show_progress=True)

    if name == 'ogbn-arxiv' or name == 'ogbn-proteins' or name == 'SR25' or name == 'Count':
      return dataset

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name in ['PATTERN', 'CLUSTER', 'CSL']:
        tf_list = []
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")
    if name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        dataset = join_dataset_splits(
            [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
            for split in ['train', 'val', 'test']]
        )
        pre_transform_in_memory(dataset, T.Compose(tf_list))
    elif name == 'CSL':
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name)

    # dataset = join_dataset_splits(
    #     [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
    #      for split in ['train', 'val', 'test']]
    # )
    # pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_SR(dataset_dir):
    dataset = SRDataset(dataset_dir)
    dataset.data.x = dataset.data.x.long()
    dataset.data.y = torch.arange(len(dataset.data.y)).long()
    dataset.data['train_graph_index'] = torch.arange(len(dataset.data.y)).long()
    dataset.data['val_graph_index'] = torch.arange(len(dataset.data.y)).long()
    dataset.data['test_graph_index'] = torch.arange(len(dataset.data.y)).long()
    logging.info(dataset.data['train_graph_index'])
    return dataset


class CountSubstructureTransform(object):
    def __init__(self, target=0):
        self.target = target

    def __call__(self, data):
        data.y = data.y[:, int(self.target)]
        if cfg.model.type == 'kTransformer':
            data = pretransform_2tuple(data)
        elif cfg.model.type == 'kSimplicialTransformerSparse':
            data = pretransform_2simplex(data, dense=False)
        elif cfg.model.type == 'kSimplicialTransformerDense':
            data = pretransform_2simplex(data, dense=True)
        return data


def preformat_GraphCount(dataset_dir):
    pretransform = CountSubstructureTransform(cfg.dataset.get('target', 0))
    dataset = GraphCountDataset(dataset_dir)
    dataset.data.y = dataset.data.y[:, cfg.dataset.get("target", 0)]
    dataset.data.y = dataset.data.y / torch.std(dataset.data.y)
    # logging.info(dataset.data)
    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from graphgps.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        def add_num_nodes(data):
            data.num_all_nodes = torch.tensor([data.num_nodes], dtype=torch.long).unsqueeze(0)
            return data
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len), add_num_nodes])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]

    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.

        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]

    elif name == 'inference':
        split_idxs = [split_idx['valid'],  # The original labeled 'valid' set.
                      split_idx['test-dev'],  # Held-out unlabeled test dev.
                      split_idx['test-challenge']  # Held-out challenge test set.
                      ]

        dataset = dataset[torch.cat(split_idxs)]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(split_idxs[0]), len(split_idxs[1]), len(split_idxs[2])
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        # Check prediction targets.
        assert(all([not torch.isnan(dataset[i].y)[0] for i in split_idxs[0]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[1]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[2]]))

    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgps.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgps.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgps.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS', 'TRIANGLES', 'alchemy_full']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    dataset.data.x = torch.argmax(dataset.data.x, dim=-1).unsqueeze(1).long()
    logging.info(dataset.data.x)
    dataset.data.edge_attr = dataset.data.edge_attr.long()
    dataset.data.edge_index = dataset.data.edge_index.long()
    dataset.data.y = (dataset.data.y - dataset.data.y.mean(dim=0)) / dataset.data.y.std(dim=0)
    # if name == 'alchemy_full':
    #     dataset.data['train_graph_index'] = torch.arange(10000).long()
    #     dataset.data['val_graph_index'] = torch.arange(10000, 11000).long()
    #     dataset.data['test_graph_index'] = torch.arange(11000, len(dataset)).long()
    #     logging.info(dataset.data['test_graph_index'])
    return dataset

def preformat_ogbn(dataset_dir, name):
  if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
    dataset = PygNodePropPredDataset(name=name)
    if name == 'ogbn-arxiv':
      pre_transform_in_memory(dataset, partial(add_reverse_edges))
      if cfg.prep.add_self_loops:
        pre_transform_in_memory(dataset, partial(add_self_loops))
    if name == 'ogbn-proteins':
      pre_transform_in_memory(dataset, partial(move_node_feat_to_x))
      pre_transform_in_memory(dataset, partial(typecast_x, type_str='float'))
    split_dict = dataset.get_idx_split()
    split_dict['val'] = split_dict.pop('valid')
    dataset.split_idx = split_dict
    return dataset


     #  We do not need to store  these separately.
     # storing separatelymight simplify the duplicated logger code in main.py
     # s_dict = dataset.get_idx_split()
     # dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]
     # convert the adjacency list to an edge_index list.
     # data = dataset[0]
     # coo = data.adj_t.coo()
     # data is only a deep copy.  Need to write to the dataset object itself.
     # dataset[0].edge_index = torch.stack(coo[:2])
     # del dataset[0]['adj_t'] # remove the adjacency list after the edge_index is created.

     # return dataset
  else:
     ValueError(f"Unknown ogbn dataset '{name}'.")

def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


def pretransform_2tuple(data):
    """
        Pretransform 2-tuples; supports full attention, A^{ngbh}, A^{ngbh+}, A^{LN}
    """
    N = copy.deepcopy(data.num_nodes)
    data.edge_index_original = data.edge_index.clone()
    if data.edge_attr is None:
        data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=torch.long)
    data.edge_attr_original = data.edge_attr.clone()
    if len(data.edge_attr.shape) == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)
    if cfg.prep.get("use_local_neighbors", False) and data.get("local_neighbor_edge_index", None) is None:
        local_k_edge = torch.zeros([N * N, N * N])
        # A^{ngbh} (use all global neighbors)
        if cfg.prep.get("use_global_neighbors", False):
            for i in range(N):
                for j in range(N):
                    delta_a = [N * i + k for k in range(N)]
                    delta_b = [N * j + k for k in range(N)]
                    local_k_edge[delta_a, delta_b] = 1
                    delta_c = [N * l + i for l in range(N)]
                    delta_d = [N * l + j for l in range(N)]
                    local_k_edge[delta_c, delta_d] = 2
        # A^{ngbh}+ (use global neighbors and local neighbors)
        if cfg.prep.get("use_global_neighbors_delta", False):
            for i in range(N):
                for j in range(N):
                    delta_a = [N * i + k for k in range(N)]
                    delta_b = [N * j + k for k in range(N)]
                    local_k_edge[delta_a, delta_b] = 3
                    delta_c = [N * l + i for l in range(N)]
                    delta_d = [N * l + j for l in range(N)]
                    local_k_edge[delta_c, delta_d] = 4
        # A^{LN} (use only local neighbors)
        for idx in range(data.edge_index.shape[1]):
            i = data.edge_index[0, idx]
            j = data.edge_index[1, idx]
            delta_a = [N * i + k for k in range(N)]
            delta_b = [N * j + k for k in range(N)]
            local_k_edge[delta_a, delta_b] = 1
            delta_c = [N * l + i for l in range(N)]
            delta_d = [N * l + j for l in range(N)]
            local_k_edge[delta_c, delta_d] = 2
        local_neighbor_edge_index, local_neighbor_edge_attr = dense_to_sparse(local_k_edge)
        data.local_neighbor_edge_index = local_neighbor_edge_index.long()
        data.local_neighbor_edge_attr = local_neighbor_edge_attr.long()
    A = SparseTensor(row=data.edge_index[0],
                     col=data.edge_index[1],
                     value=data.edge_attr + 1,  # start from 1 to 4
                     sparse_sizes=(N, N, data.edge_attr.shape[1])).coalesce().to_dense()
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)
    data.num_edge_per_graph = torch.tensor((N**2), dtype=torch.long)
    adj = torch.ones([N, N], dtype=torch.long)
    edge_index = dense_to_sparse(adj)[0]
    data.edge_index = edge_index
    data.edge_attr = A.reshape(-1, data.edge_attr.shape[1]).long()
    if data.get('pestat_EdgeRWSE', None) is not None:
        PE = SparseTensor(row=data.edge_index_original[0],
                          col=data.edge_index_original[1],
                          value=data.pestat_EdgeRWSE,  # start from 1 to 4
                          sparse_sizes=(N, N, data.pestat_EdgeRWSE.shape[1])).coalesce().to_dense()
        data.pestat_EdgeRWSE = PE.reshape(-1, data.pestat_EdgeRWSE.shape[1])
    # data.num_nodes = N ** 2
    return data


def pretransform_1_2tuple(data):
    """
        Pretransform cross attentin between 1-tuples and 2-tuples
    """
    N = copy.deepcopy(data.num_nodes)
    data.edge_index_original = data.edge_index.clone()
    if data.edge_attr is None:
        data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=torch.long)
    data.edge_attr_original = data.edge_attr.clone()
    if len(data.edge_attr.shape) == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)
    A = SparseTensor(row=data.edge_index[0],
                     col=data.edge_index[1],
                     value=data.edge_attr + 1,  # start from 1 to 4
                     sparse_sizes=(N, N, data.edge_attr.shape[1])).coalesce().to_dense()
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)
    data.num_edge_per_graph = torch.tensor((N**2), dtype=torch.long)
    adj = torch.ones([N, N], dtype=torch.long)
    edge_index = dense_to_sparse(adj)[0]
    data.edge_index = edge_index
    data.edge_attr = A.reshape(-1, data.edge_attr.shape[1]).long()
    if data.get('pestat_EdgeRWSE', None) is not None:
        PE = SparseTensor(row=data.edge_index_original[0],
                          col=data.edge_index_original[1],
                          value=data.pestat_EdgeRWSE,  # start from 1 to 4
                          sparse_sizes=(N, N, data.pestat_EdgeRWSE.shape[1])).coalesce().to_dense()
        data.pestat_EdgeRWSE = PE.reshape(-1, data.pestat_EdgeRWSE.shape[1])

    # local or global version
    local = cfg.model.get('local', True)
    if local:
        data.query_idx = torch.ones([N], dtype=torch.long) * N * 2
        key_idx = torch.zeros([2 * N * N], dtype=torch.long)
        for i in range(N):
            key_idx[2 * i * N: 2 * i * N + N] = torch.arange(i*N, (i+1)*N)
            key_idx[2 * i * N + N: 2 * (i+1) * N] = torch.arange(i, i + N * N, N)
        data.key_idx = key_idx.long()
        connectivity1 = torch.zeros([N, N], dtype=torch.long)
        connectivity2 = torch.ones([N, N], dtype=torch.long)
        data.connectivity = torch.cat([connectivity1, connectivity2], dim=1).reshape(-1).long()
    else:
        data.query_idx = torch.ones([N], dtype=torch.long) * N * N
        data.key_idx = torch.arange(N*N, dtype=torch.long).unsqueeze(0).repeat(N, 1).reshape(-1)
        connectivity = torch.zeros([N ** 3], dtype=torch.long)
        for i in range(N):
            connectivity[i * N * N + i * N: i * N * N + (i+1) * N] = 1
            for j in range(N):
                connectivity[i * N * N + i + j * N] = 2
        data.connectivity = connectivity
    return data


def pretransform_3tuple(data):
    """
        Pretransform connected 3-tuples
    """
    N = copy.deepcopy(data.num_nodes)
    data.edge_index_original = data.edge_index.clone()
    if data.edge_attr is None:
        data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=torch.long)
    data.edge_attr_original = data.edge_attr.clone()
    if len(data.edge_attr.shape) == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)

    # sample connected 3-tuples to reduce tuple numbers
    distance_constrainu = cfg.prep.get("distlimitu", 2)
    distance_constrainl = cfg.prep.get("distlimitl", -1)
    k = 3
    subgs = extractsubset(torch.arange(N), k)  # (ns, k)
    if distance_constrainu >= 0 or distance_constrainl >= 0:
        tadj = SparseTensor(row=data.edge_index[0],
                            col=data.edge_index[1],
                            sparse_sizes=(N, N)).to_dense().numpy()
        tadj = floyd_warshall(tadj, directed=False, unweighted=True)
        tadj = torch.from_numpy(tadj)
        mask = torch.ones_like(tadj, dtype=torch.bool)
        if distance_constrainu >= 0:
            mask &= tadj <= distance_constrainu
        if distance_constrainl >= 0:
            mask &= tadj >= distance_constrainl
        mask |= torch.eye(N, dtype=torch.bool)
        tadj = mask.flatten()
        subgsmask = (subgs.unsqueeze(-1) * N + subgs.unsqueeze(-2)).flatten(
            -2, -1)
        subgsmask = tadj[subgsmask].all(dim=-1)
        subgs = subgs[subgsmask]

    data.tuple_index = subgs.permute(1, 0)
    # logging.info(subgs.shape)

    A = SparseTensor(row=data.edge_index[0],
                     col=data.edge_index[1],
                     value=data.edge_attr + 1,  # start from 1 to 4
                     sparse_sizes=(N, N, data.edge_attr.shape[1])).coalesce().to_dense()
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)
    data.num_edge_per_graph = torch.tensor((subgs.shape[0]), dtype=torch.long)
    data.num_tuple_per_graph = torch.tensor((subgs.shape[0]), dtype=torch.long)
    edge_attr = []
    pestat = []
    if data.get('pestat_EdgeRWSE', None) is not None:
        PE = SparseTensor(row=data.edge_index[0],
                          col=data.edge_index[1],
                          value=data.pestat_EdgeRWSE,  # start from 1 to 4
                          sparse_sizes=(N, N, data.pestat_EdgeRWSE.shape[1])).coalesce().to_dense()
    for i in range(subgs.shape[0]):
        edge_attr.append(A[subgs[i][0]][subgs[i][1]].unsqueeze(0))
        edge_attr.append(A[subgs[i][0]][subgs[i][2]].unsqueeze(0))
        edge_attr.append(A[subgs[i][1]][subgs[i][2]].unsqueeze(0))
        if data.get('pestat_EdgeRWSE', None) is not None:
            pestat.append(PE[subgs[i][0]][subgs[i][1]].unsqueeze(0))
            pestat.append(PE[subgs[i][0]][subgs[i][2]].unsqueeze(0))
            pestat.append(PE[subgs[i][1]][subgs[i][2]].unsqueeze(0))
    if len(subgs) == 0:
        data.edge_attr = None
    else:
        edge_attr = torch.cat(edge_attr, dim=0)
        # logging.info(edge_attr.shape)
        data.edge_attr = edge_attr.reshape(-1, data.edge_attr.shape[1]).long()
    if data.get('pestat_EdgeRWSE', None) is not None:
        pestat = torch.cat(pestat, dim=0)
        data.pestat_EdgeRWSE = pestat
    # data.num_nodes = N ** 2
    return data


def pretransform_2simplex(data, dense=True):
    """
        Pretransform 0, 1, and 2-simplices
    """
    N = copy.deepcopy(data.num_nodes)
    if data.edge_attr is None:
        data.edge_attr = torch.ones([data.edge_index.shape[1], 1], dtype=torch.long)
    if len(data.edge_attr.shape) == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)
    data.edge_index_original = data.edge_index.clone()
    data.edge_attr_original = data.edge_attr.clone()
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)

    directed = cfg.dataset.get('directed', False)
    data.num_edge_per_graph = torch.tensor((int(data.edge_index.shape[1] / 2)), dtype=torch.long) if not directed else torch.tensor((int(data.edge_index.shape[1])), dtype=torch.long)
    # A = SparseTensor(row=data.edge_index[0],
    #                  col=data.edge_index[1],
    #                  value=data.edge_attr + 1,
    #                  sparse_sizes=(N, N, data.edge_attr.shape[1])).coalesce().to_dense()
    if not directed:
        mask = torch.triu(torch.ones(N, N), diagonal=1)
        # A_mask = A * mask.unsqueeze(2).reshape(-1, data.edge_attr.shape[1])
        directed_edge_attr = data.edge_attr[data.edge_index[0] < data.edge_index[1]]
        if data.get('pestat_EdgeRWSE', None) is not None:
            data.pestat_EdgeRWSE = data.pestat_EdgeRWSE[data.edge_index[0] < data.edge_index[1]]
        # directed_edge_index = data.edge_index[data.edge_index[0] < data.edge_index[1]]

        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(N, N)).coalesce().to_dense()
        A_mask = A * mask
        directed_edge_index, _ = dense_to_sparse(A_mask)

        B = torch.abs(compute_Helmholtzians_Hodge_1_Laplacian(directed_edge_index, N, False))
        offset = torch.max(B)
        d = torch.sum(A, dim=1)
        A *= (offset + 3)
        A += torch.diag_embed(d + offset + 4)
        B1 = torch.zeros([N, directed_edge_index.shape[1]], dtype=torch.float)
        for i in range(directed_edge_index.shape[1]):
            B1[directed_edge_index[0, i], i] = 1
            B1[directed_edge_index[1, i], i] = 1
    # B = torch.matmul(B1.permute(1, 0), B1)
    else:
        # A_mask = A * mask.unsqueeze(2).reshape(-1, data.edge_attr.shape[1])
        directed_edge_attr = data.edge_attr
        directed_edge_index = data.edge_index
        if data.get('pestat_EdgeRWSE', None) is not None:
            data.pestat_EdgeRWSE = data.pestat_EdgeRWSE
        # directed_edge_index = data.edge_index[data.edge_index[0] < data.edge_index[1]]
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(N, N)).coalesce().to_dense()

        B = torch.abs(compute_Helmholtzians_Hodge_1_Laplacian(data.edge_index, N, True))
        offset = torch.max(B)
        d = torch.sum(A, dim=1)
        A *= (offset + 3)
        A += torch.diag_embed(d + offset + 4)
        B1 = torch.zeros([N, data.edge_index.shape[1]], dtype=torch.float)
        for i in range(data.edge_index.shape[1]):
            B1[data.edge_index[0, i], i] = 1
            B1[data.edge_index[1, i], i] = 1

    P_u = torch.cat([A, B1 * (offset + 2)], dim=1)
    P_d = torch.cat([B1.permute(1, 0) * (offset + 1), B], dim=1)
    L = torch.cat([P_u, P_d], dim=0).long()

    data.edge_attr = directed_edge_attr
    data.edge_index = directed_edge_index

    if cfg.prep.get("use_local_neighbors", True):
        if dense:
            data.local_neighbor_edge_attr = L.reshape(-1, 1).long()
        else:
            local_neighbor_edge_index, local_neighbor_edge_attr = dense_to_sparse(L)
            data.local_neighbor_edge_index = local_neighbor_edge_index.long()
            data.local_neighbor_edge_attr = local_neighbor_edge_attr.long()

    # logging.info(data)
    return data
