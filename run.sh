# #!/bin/bash

# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.3 --drop_feat_p 0.2  --dataset Cora
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset CiteSeer --drop_edge_p 0.5 --drop_feat_p 0.2
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset PubMed
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset WikiCS
# python flow-on-hypersphere.py --gamma 2.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.3 --drop_feat_p 0.2 --dataset Computers
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset Photo
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset CS	
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --dataset Physics

# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.5 --drop_feat_p 0.2 --dataset Cora
# python flow-on-hypersphere.py --gamma 2.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.5 --drop_feat_p 0.2 --dataset CiteSeer
# python flow-on-hypersphere.py --gamma 3.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.8 --drop_feat_p 0.2 --dataset PubMed
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.5 --drop_feat_p 0.2 --dataset WikiCS
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0 --drop_feat_p 0 --dataset Computers
# python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0 --drop_feat_p 0 --dataset Photo
# python flow-on-hypersphere.py --gamma 3.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.8 --drop_feat_p 0.2 --dataset Physics
# python flow-on-hypersphere.py --gamma 3.0 --alpha 1.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.8 --drop_feat_p 0.2 --dataset CS


# CUDA_VISIBLE_DEVICES=1 python flow-on-hypersphere.py --gamma 1.0 --alpha 1.0 --beta 0.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.5 --drop_feat_p 0.2 --dataset WikiCS
# CUDA_VISIBLE_DEVICES=1 python flow-on-hypersphere.py --gamma 1.0 --alpha 0.0 --beta 1.0  --num_layer 1 --num_epochs 1000 --hidden_dim 1024 --drop_edge_p 0.5 --drop_feat_p 0.2 --dataset WikiCS

# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset PubMed --k 1 --alpha 3.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset PubMed --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset PubMed --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset WikiCS --k 1 --alpha 3.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset WikiCS --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python flow-on-hypersphere.py --dataset WikiCS --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1

# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset PubMed --k 1 --alpha 3.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset PubMed --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset PubMed --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset Computers --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset Computers --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python flow-on-hypersphere2.py --dataset Computers --k 1 --alpha 3.0 --drop_edge_p 0.8 --drop_feat_p 0.1

# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset Cora --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset Cora --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 2
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset Cora --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 3

# CUDA_VISIBLE_DEVICES=1 python flow-on-hypersphere2.py --dataset CS --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1
# CUDA_VISIBLE_DEVICES=1 python flow-on-hypersphere2.py --dataset CS --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 2
# CUDA_VISIBLE_DEVICES=1 python flow-on-hypersphere2.py --dataset CS --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 3

# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset squirrel_filtered --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1 --trials 3
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset chameleon_filtered --k 1 --alpha 2.0 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1 --trials 3
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset amazon_ratings --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1 --trials 3
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset roman_empire --k 1 --alpha 1.5 --drop_edge_p 0.8 --drop_feat_p 0.1 --num_layers 1 --trials 3

# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset chameleon_filtered --k 1 --alpha 2.0 --drop_edge_p 0.0 --drop_feat_p 0.0 --num_layers 1 --trials 1
# CUDA_VISIBLE_DEVICES=0 python flow-on-hypersphere2.py --dataset chameleon_filtered --k 1 --alpha 2.0 --drop_edge_p 0.1 --drop_feat_p 0.1 --num_layers 3 --trials 1

# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --align_mode mean --run_noise_eval --trials 5 --h_target 2.0
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --align_mode direct --run_noise_eval --trials 5 --h_target 2.0
# python flow-on-hypersphere2-anchor_stability.py --dataset CS --align_mode mean --run_noise_eval --trials 5
# python flow-on-hypersphere2-anchor_stability.py --dataset CS --align_mode direct --run_noise_eval --trials 5
# python flow-on-hypersphere2-anchor_stability.py --dataset Cora --align_mode mean --trials 1 --h_target 1.5

# python flow-on-hypersphere2-anchor_stability.py --dataset Cora --repulsion_mode negative_sampling --num_neg 30 --h_target 1.5 --trials 1
# python flow-on-hypersphere2-anchor_stability.py --dataset Cora --repulsion_mode pairwise --num_neg 16 --h_target 1.5 --trials 1

# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode mean_field --use_adaptive_alpha --h_target 2.0 --trials 5
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 5 --h_target 2.0 --trials 5
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 10 --h_target 2.0 --trials 1
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 15 --h_target 2.0 --trials 1
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 20 --h_target 2.0 --trials 1
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 5 --h_target 2.0 --trials 1 --alpha_fix 2.0
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 5 --h_target 2.0 --trials 1 --alpha_fix 1.5
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 5 --h_target 2.0 --trials 1 --alpha_fix 0.5
# python flow-on-hypersphere2-anchor_stability.py --dataset WikiCS --repulsion_mode negative_sampling --num_neg 5 --h_target 2.0 --trials 1 --alpha_fix 0.1

# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 1 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 5 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 10 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 20 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 30 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 40 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset Cora --loss_mode nash --warmup_epochs 50 --nash_use_softplus --trials 3 --num_layers 1

# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 1 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 5 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 10 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 20 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 30 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 40 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CiteSeer --loss_mode nash --warmup_epochs 50 --nash_use_softplus --trials 3 --num_layers 1

# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 1 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 5 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 10 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 20 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 30 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 40 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset PubMed --loss_mode nash --warmup_epochs 50 --nash_use_softplus --trials 3 --num_layers 1

# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 1 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 5 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 10 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 20 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 30 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 40 --nash_use_softplus --trials 3 --num_layers 1
# python flow-on-hypersphere2-nash.py --dataset CS --loss_mode nash --warmup_epochs 50 --nash_use_softplus --trials 3 --num_layers 1


# python train-bayes.py --dataset Cora --backbone transformer --hidden_dim 1024 --loss_mode bayes --lr 1e-3 --thermo_lr 1e-3 --init_kappa 1.0 --init_kappa_align -1 --trials 3
# python train-bayes.py --dataset CiteSeer --backbone transformer --hidden_dim 1024 --loss_mode bayes --lr 1e-3 --thermo_lr 1e-3 --init_kappa 1.0 --init_kappa_align -1 --trials 3

# python train-bayes.py --dataset WikiCS --backbone transformer --hidden_dim 1024 --loss_mode bayes --lr 1e-3 --thermo_lr 1e-3 --init_kappa 1.0 --init_kappa_align -1 --trials 3
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Cora --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset CiteSeer --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset PubMed --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset WikiCS --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Computers --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Photo --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset CS --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Physics --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.3 --drop_feat_p 0.2

# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Cora --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset CiteSeer --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset PubMed --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset WikiCS --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Computers --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Photo --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset CS --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset Physics --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1

# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset WikiCS --backbone transformer --loss_mode bayes --init_kappa 1.2 --trials 1 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset WikiCS --backbone transformer --loss_mode bayes --init_kappa 0.8 --trials 1 --drop_edge_p 0.2 --drop_feat_p 0.1

# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset Cora --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.8 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset CiteSeer --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset PubMed --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=3 python train-bayes.py --dataset WikiCS --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset Computers --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset Photo --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset CS --backbone transformer --loss_mode bayes --init_kappa 1.0  --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1
# CUDA_VISIBLE_DEVICES=2 python train-bayes.py --dataset Physics --backbone transformer --loss_mode bayes --init_kappa 1.0 --trials 3 --drop_edge_p 0.2 --drop_feat_p 0.1

# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-4
# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-4
# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-3
# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-3
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-4
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-4
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-3
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-3
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-2
# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 1e-2
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-4
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-4
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-3
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-3
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-2
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-4
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-4
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-3
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-3
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 1e-2

# python train-bayes.py --dataset Cora --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-5
# python train-bayes.py --dataset Texas --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-5
# python train-bayes.py --dataset Cornell --loss_mode bayes --init_kappa 0.2 --trials 1 --thermo_mode fixed_T --fixed_T 5e-5
# python train-bayes.py --dataset CS --loss_mode bayes --init_kappa 1.0 --trials 1 --thermo_mode fixed_T --fixed_T 5e-5

# python train-bayes.py --dataset CS --loss_mode bayes --thermo_mode adaptive --trials 1
# python train-bayes.py --dataset Texas --loss_mode bayes --thermo_mode adaptive --trials 1

CUDA_VISIBLE_DEVICES=1 python train.py --dataset Cora --init_kappa 1.0 --trials 1
