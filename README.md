# k-Transformer

Official code repo for AISTATS 2024 paper: On the Theoretical Expressive Power and Design Space of High-Order Graph Transformers. Paper: https://proceedings.mlr.press/v238/zhou24a.html

![k-Transformer](./k-Transformer.png)

In this paper, we provide a systematic study of the theoretical expressive power of order- $k$ graph transformers and sparse variants. We first show that, an order- $k$ graph transformer without additional structural information is less expressive than the $k$-Weisfeiler Lehman ($k$-WL) test despite its high computational cost. We then explore strategies to both sparsify and enhance the higher-order graph transformers, aiming to improve both their efficiency and expressiveness. Indeed, sparsification based on neighborhood information can enhance the expressive power, as it provides additional information about input graph structures. In particular, we show that a natural neighborhood-based sparse order- $k$ transformer model is not only computationally efficient, but also expressive – as expressive as $k$-WL test. We further study several other sparse graph attention models that are computationally efficient and provide their expressiveness analysis.

We implement 2-transformers with full attention ($\mathcal A_2$), kernelized attention ($\mathcal A_2-Performer$), neighbor attention ($\mathcal A_k^{\mathsf{Ngbh}}$ and $\mathcal A_k^{\mathsf{Ngbh+}}$), local neighbor attention ($\mathcal A_k^{\mathsf{LN}}$) and virtual tuple attention ($\mathcal A_k^{\mathsf{VT}}$). For ablation study, we also implement cross attention between $1$-tuples and $2$-tuples, as well as sampling connected $3$-tuples. For simplicial transformers, we implement both dense version for order- $0,1,2$ simplices ($\mathcal{AS}_ {0:2}$) which could use Hodge Laplacians as attention biases, and sparse version with simplex neighbor attention ($\mathcal{AS}_ {0:2}^{\mathsf{SN}}$) and virtual simplex attention ($\mathcal{AS}_{0:2}^{\mathsf{VS}}$).

### Python environment setup with Conda

```bash
conda create -n exphormer python=3.9
conda activate exphormer

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb

conda clean --all
```


### Running Exphormer

```bash
conda activate exphormer

# Running Exphormer for LRGB Datasets
python main.py --cfg configs/Exphormer_LRGB/peptides-struct-EX.yaml  wandb.use False

# Running Exphormer for Cifar10
python main.py --cfg configs/Exphormer/cifar10.yaml  wandb.use False
```

You can also set your wandb settings and use wandb.

### Guide on configs files

Most of the configs are shared with [GraphGPS](https://github.com/rampasek/GraphGPS) code. You can change the following parameters in the config files for different parameters and variants of the Exphormer:

```
prep:
  exp: True  # Set True for using expander graphs, set False otherwise. 
    # Alternatively you can set use_exp_edges to False.
    # In this case expander graphs will be calculated but not used in the Exphormer. 
  exp_deg: 5 # Set the degree of the expander graph.
    # Please note that if you set this to d, the algorithm will use d permutations 
    # or d Hamiltonian cycles, so the actual degree of the expander graph will be 2d
  exp_algorithm: 'Random-d' # Options are ['Random-d', 'Random-d2', 'Hamiltonian].
    # Default value is 'Random-d'
  add_edge_index: True # Set True if you want to add real edges beside expander edges
  num_virt_node: 1 # Set 0 for not using virtual nodes 
    # otherwise set the number of virtual nodes you want to use.
```


## Citation

```
@InProceedings{pmlr-v238-zhou24a,
  title = 	 { On the Theoretical Expressive Power and the Design Space of Higher-Order Graph Transformers },
  author =       {Zhou, Cai and Yu, Rose and Wang, Yusu},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2179--2187},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/zhou24a/zhou24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/zhou24a.html},
  abstract = 	 { Graph transformers have recently received significant attention in graph learning, partly due to their ability to capture more global interaction via self-attention. Nevertheless, while higher-order graph neural networks have been reasonably well studied, the exploration of extending graph transformers to higher-order variants is just starting. Both theoretical understanding and empirical results are limited. In this paper, we provide a systematic study of the theoretical expressive power of order-$k$ graph transformers and sparse variants. We first show that, an order-$k$ graph transformer without additional structural information is less expressive than the $k$-Weisfeiler Lehman ($k$-WL) test despite its high computational cost. We then explore strategies to both sparsify and enhance the higher-order graph transformers, aiming to improve both their efficiency and expressiveness. Indeed, sparsification based on neighborhood information can enhance the expressive power, as it provides additional information about input graph structures. In particular, we show that a natural neighborhood-based sparse order-$k$ transformer model is not only computationally efficient, but also expressive – as expressive as $k$-WL test. We further study several other sparse graph attention models that are computationally efficient and provide their expressiveness analysis. Finally, we provide experimental results to show the effectiveness of the different sparsification strategies. }
}
```


