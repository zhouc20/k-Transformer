# Exphormer
CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/Exphormer/pcqm-contact-GatedGCN+Exphormer.yaml  seed 1 wandb.use False  # charlie tmux 1
CUDA_VISIBLE_DEVICES=2 python main.py --cfg configs/Exphormer/cifar10-GatedGCN+Exphormer.yaml --repeat 3  wandb.use False  #  charlie tmux 2
CUDA_VISIBLE_DEVICES=3 python main.py --cfg configs/Exphormer/mnist-GatedGCN+Exphormer.yaml  --repeat 3 wandb.use False  # charlie tmux 3
CUDA_VISIBLE_DEVICES=4 python main.py --cfg configs/Exphormer/peptides-struct-GatedGCN+Exphormer.yaml  --repeat 3 wandb.use False  # charlie tmux 4
CUDA_VISIBLE_DEVICES=5 python main.py --cfg configs/Exphormer/peptides-func-GatedGCN+Exphormer.yaml  --repeat 3 wandb.use False  # charlie tmux 5



