# SelfGRL

Minimal implementation of **SelfGRL: Self-Organized Graph Representation Learning Inspired by Helmholtz Free Energy Minimization**.

---

## ⚙️ Dependencies

| Package | Version |
|----------|----------|
| Python | 3.10 |
| torch | 2.4.1 |
| torch-geometric | 2.6.1 |
| torch_sparse | 0.6.18 |
| torch_scatter | 2.1.2 |
| torch_cluster | 1.6.3 |
| torch_spline_conv | 1.2.2 |
| numpy | 2.1.2 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| pandas | 2.3.2 |
| matplotlib | 3.10.5 |
| tqdm | 4.67.1 |
| ogb | 1.3.6 |
| umap-learn | 0.5.9.post2 |


---

## 🚀 Run

Example command to train on **Cora** and **Texa**:

```bash
python train.py   --dataset Cora --dataset_dir [path_to_datasets] --init_kappa 1.0
python train.py   --dataset Cora --dataset_dir [path_to_datasets] --init_kappa 0.2
```

You can modify the arguments to experiment with other datasets such as `CiteSeer`, `PubMed`, or `Photo`.

---

## 📁 Project Structure

```
SelfGRL_code/
├── train.py         # Encoder, training loop, and evaluation entry point
├── loss.py          # Bayesian thermostat objective
├── data.py          # Dataset loading and evaluation utilities
├── transforms.py    # Edge dropout and feature masking
├── README.md
├── datasets/        # Downloaded datasets
├── checkpoints/     # Generated encoder checkpoints
└── results/         # Generated evaluation summaries
```
