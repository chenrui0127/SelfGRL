import argparse
import os
import re

import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn import metrics
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS, CoraFull, WebKB, Actor
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, SAGEConv
from torch_geometric.transforms import ToUndirected, NormalizeFeatures, NormalizeScale, NormalizeRotation
import torch_geometric.transforms as T

# from alpha_decay import combined_cosine_decay
from transforms import get_graph_drop_transform
from utils import *
from vis import plot_umap_hypersphere, plot_tsne_hypersphere, plot_loss_with_moving_avg
from loss import BayesianThermodynamicsLoss
from data import load_dataset, get_wiki_cs, get_splits, eval_logistic_regression, eval_logistic_regression_random_splits, eval_logistic_regression_preset_splits, eval_clustering
from torch_geometric.utils import is_undirected
import torch
import torch_geometric
from sklearn.cluster import KMeans
import pandas as pd
from utils_metrics import compute_geometry_metrics

torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])

torch.manual_seed(3456)
np.random.seed(3456)
random.seed(3456)

def compute_alpha_t(alpha_min, alpha_max, beta, H_target, H_proxy_t):
    """
    Compute alpha_t based on the Entropy-Guided Adaptive Balancing formula.

    Parameters:
    - alpha_min (float): Minimum value for alpha.
    - alpha_max (float): Maximum value for alpha.
    - beta (float): Sensitivity hyperparameter.
    - H_target (float): Target entropy value.
    - H_proxy_t (float): Proxy entropy at time t.

    Returns:
    - float: Computed alpha_t.
    """
    if H_target == 0:
        raise ValueError("H_target cannot be zero to avoid division by zero.")

    normalized_diff = (H_target - H_proxy_t) / H_target
    sigmoid_arg = beta * normalized_diff
    sigmoid = 1 / (1 + torch.exp(-sigmoid_arg))
    alpha_t = alpha_min + (alpha_max - alpha_min) * sigmoid
    return alpha_t

def l2_normalize(x, eps=1e-16):
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

# ---------------- models ----------------
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, backbone='gcn'):
        super().__init__()

        self.activation = torch.nn.SiLU()
        self.layers = torch.nn.ModuleList()
        backbone = backbone.lower()

        if backbone == 'gcn':
            self.layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        elif backbone == 'gat':
            # 首层GAT，将输入维度映射到hidden_dim，通过heads控制
            self.layers.append(GATConv(input_dim, hidden_dim // 8, heads=8))
            # 中间层GAT，保持维度为hidden_dim
            for _ in range(num_layers - 1):
                # 第二层及以后的输入维度是上一层的输出维度，即hidden_dim
                # 为了保持输出维度不变，每个头的输出维度仍为 hidden_dim // num_heads
                self.layers.append(GATConv(hidden_dim, hidden_dim // 8, heads=8))

        elif backbone == 'sage':
            self.layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        elif backbone == 'transformer':
            self.layers.append(TransformerConv(input_dim, hidden_dim // 16, heads=16, beta=False))
            # beta为残差链接
            for _ in range(num_layers - 1):
                self.layers.append(TransformerConv(hidden_dim, hidden_dim // 4, heads=4, beta=True))

    def forward(self, x, edge_index):
        z = x
        outputs = []
        for i, conv in enumerate(self.layers):
            h = conv(z, edge_index)
            if i > 0:
                z = h + z
            else:
                z = h
            z = self.activation(z)
            outputs.append(z)
        return z, outputs
    

# ---------------- helpers ----------------
def neighbor_mean(z, edge_index, num_nodes, add_self=False):
    row, col = edge_index
    dev = z.device
    acc = torch.zeros((num_nodes, z.shape[1]), device=dev)
    cnt = torch.zeros((num_nodes, 1), device=dev)
    acc.index_add_(0, row, z[col])
    cnt.index_add_(0, row, torch.ones((row.shape[0], 1), device=dev))
    if add_self:
        acc = acc + z
        cnt = cnt + 1.0
    return acc / (cnt + 1e-9), cnt

def compute_loss_components(z, edge_index, num_nodes, k=1):
    nb_mean = z
    nb_count = torch.zeros((num_nodes, 1), device=z.device)
    for _ in range(k):
        nb_mean, nb_count = neighbor_mean(nb_mean, edge_index, num_nodes, add_self=True)
        nb_mean = l2_normalize(nb_mean)

    cos_sim = (z * nb_mean).sum(dim=1)
    w = torch.pow(torch.sigmoid(nb_count), 5)
    align_loss = (w * (1. - cos_sim)).mean()

    collapse_metric = z.mean(dim=0)
    unif_loss = (collapse_metric * collapse_metric).sum()

    H_proxy = -torch.log(unif_loss + 1e-6)

    components = {
        "L_align": align_loss.item(),
        "L_unif": unif_loss.item(),
        "H_proxy": H_proxy.item(),
    }
    return align_loss, unif_loss, H_proxy, components

# ---------------- load data ----------------
def run(args, trial):

    if args.dataset in ['WikiCS']:
        dataset = load_dataset(args.dataset, args.dataset_dir)
        num_classes = dataset.num_classes
        # dataset, num_classes, train_masks, val_masks, test_masks = get_wiki_cs(args.dataset_dir)
    else:
        dataset = load_dataset(args.dataset, args.dataset_dir)
        num_classes = dataset.num_classes

    # dataset = load_dataset(args.dataset, args.dataset_dir)

    data = dataset[0]
    print(is_undirected(data.edge_index, num_nodes=data.num_nodes))

    # prepare transforms
    transform = get_graph_drop_transform(drop_edge_p=args.drop_edge_p, drop_feat_p=args.drop_feat_p)
    dataset_name = getattr(dataset, "name", args.dataset)
    print(dataset_name, dataset[0], '\n')
    # print(dataset.name, dataset[0], '\n')

    # if len(dataset[0].y.shape) > 1:
    #     dataset[0].y = dataset[0].y.squeeze(1)

    # num_nodes = dataset.x.shape[0]
    # input_dim = dataset.x.shape[1]
    # data = ToUndirected()(dataset[0]).to(args.device)

    if len(data.y.shape) > 1:
        data.y = data.y.squeeze(1)

    num_nodes = data.x.shape[0]
    input_dim = data.x.shape[1]

    data = ToUndirected()(data).to(args.device)

    encoder = GraphEncoder(input_dim, args.hidden_dim, num_layers=args.num_layers, backbone=args.backbone).to(args.device)
    criterion = None
    if args.loss_mode == 'bayes':
        criterion = BayesianThermodynamicsLoss(
            embedding_dim=args.hidden_dim,
            init_kappa=args.init_kappa,
            init_kappa_align=args.init_kappa_align if args.init_kappa_align > 0 else None
        ).to(args.device)

        if args.thermo_mode == "fixed_T":
            criterion.set_fixed_temperature(
                fixed_T=args.fixed_T,
                freeze_unif=True,
            )

        # optimizer = torch.optim.Adam([
        #     {'params': encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        #     {'params': criterion.parameters(), 'lr': args.thermo_lr, 'weight_decay': 0.0}
        # ])

        # Only optimize trainable thermostat parameters.
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]

        param_groups = [
            {
                'params': encoder.parameters(),
                'lr': args.lr,
                'weight_decay': args.weight_decay,
            }
        ]

        if len(criterion_params) > 0:
            param_groups.append({
                'params': criterion_params,
                'lr': args.thermo_lr,
                'weight_decay': 0.0,
            })

        optimizer = torch.optim.Adam(param_groups)


    else:
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- training ----------------
    epochs = args.num_epochs
    loss_hist = []
    nmi_hist = {}
    ari_hist = {}
    geometry_hist = []

    best_loss = 1e9
    best_epoch = 0
    patience = 50

    alpha_max = 5
    alpha_min = 0.03
    alpha = alpha_max
    gamma = 0.1

    for epoch in range(epochs):
        encoder.train()

        arg_data = transform(data)

        h, o = encoder(arg_data.x, arg_data.edge_index)  # [N,z_dim]
        z = l2_normalize(h)

        align_loss, unif_loss, H_proxy_t, comps = compute_loss_components(
            z, arg_data.edge_index, data.num_nodes, k=args.k
        )

        U_t = -align_loss
        S_t = H_proxy_t

        # ---------------- loss mode ----------------
        if args.loss_mode == 'original':
            loss = align_loss + alpha * unif_loss

            # 原始 thermostat 更新
            alpha_pred = compute_alpha_t(alpha_min, alpha_max, 5, args.h_target, H_proxy_t.detach())
            alpha = (1 - gamma) * alpha + gamma * alpha_pred

            comps["loss"] = loss.item()
            comps["alpha"] = float(alpha)

        elif args.loss_mode == 'bayes':
            # align_loss: 局部结构绑定项
            # collapse_metric: 全局均匀性 / collapse 项
            loss = criterion(align_loss, unif_loss)

            thermo_stats = criterion.current_thermodynamics_status

            comps["loss"] = loss.item()
            comps["kappa_align"] = thermo_stats["kappa_align"]
            comps["kappa"] = thermo_stats["kappa"]
            comps["T_eff"] = thermo_stats["effective_T"]
            comps["w_align"] = thermo_stats["w_align"]
            comps["w_unif"] = thermo_stats["w_unif"]
        else:
            raise ValueError(f"Unknown loss_mode: {args.loss_mode}")

        loss_hist.append(comps)


        if (epoch + 1) % 10 == 0 or epoch == 0:
            # loss_msg = " | ".join([f"{k} {v:.5f}" for k, v in loss_hist[-1].items()])
            loss_msg = " | ".join([
                f"{k} {v:.8f}" if isinstance(v, (int, float)) else f"{k} {v}"
                for k, v in loss_hist[-1].items()
            ])

            nmi, ari = eval_clustering(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), num_classes)
            nmi_hist[epoch] = nmi
            ari_hist[epoch] = ari

            if args.loss_mode == 'original':
                print(f"Epoch {epoch + 1:>3d}/{epochs} | {loss_msg} | NMI {nmi:.4f} | ARI {ari:.4f} | alpha {alpha:.4f}")
            else:
                print(f"Epoch {epoch + 1:>3d}/{epochs} | {loss_msg} | NMI {nmi:.4f} | ARI {ari:.4f}")

        if best_loss > loss.item():
            # print("Reset patience")
            patience = 50
            best_loss = loss.item()
            best_epoch = epoch

            ckpt_dir = args.ckpt_dir
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_path = f"{ckpt_dir}/model_{dataset_name}_{trial}_{best_epoch + 1}.pth"
            torch.save(encoder.state_dict(), ckpt_path)
        else:
            # print(f'patience {patience}')
            patience = patience - 1
            if patience <= 0:
                print(f"\nEarly stopping at epoch {epoch + 1}/{epochs}, best epoch {best_epoch + 1}.")
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_path = f"{ckpt_dir}/model_{dataset_name}_{trial}_{best_epoch + 1}.pth"
    print(f"Load BEST model: {best_path}, loss: {best_loss:.5f}, epoch: {best_epoch + 1}")
    state_dict = torch.load(best_path, weights_only=True)
    encoder.load_state_dict(state_dict)

    h, o = encoder(data.x, data.edge_index)  # [N,z_dim]

    z = l2_normalize(h)

    print("\nEvaluating ...\n")

    x = z.cpu().detach().numpy()
    y = data.y.cpu().numpy()

    nmi, ari = eval_clustering(x, y, num_classes)
    print(f"NMI: {nmi:.4f} \nARI: {ari:.4f}")

    # if args.dataset in ['WikiCS']:
    #     acc_scores, f1_scores = eval_logistic_regression(x, y, repeat=3)
        # acc_scores, f1_scores = eval_logistic_regression_preset_splits(x, y, train_masks, val_masks, test_masks)
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'CS', 'Physics']:
        acc_scores, f1_scores = eval_logistic_regression(x, y, repeat=3)
        # splits = get_splits(num_samples=x.shape[0], train_ratio=0.1, valid_ratio=0.1, test_ratio=0.8, num_splits=3, seed=3456)
        # acc_scores, f1_scores = eval_logistic_regression_random_splits(x, y, splits)
    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin','Actor']:
        splits = get_splits(num_samples=x.shape[0], train_ratio=0.48, valid_ratio=0.32, test_ratio=0.20, num_splits=3, seed=3456)
        acc_scores, f1_scores = eval_logistic_regression_random_splits(x, y, splits)
    else:
        raise ValueError('Unknown dataset!!!') 

    # acc_scores, f1_scores = eval_logistic_regression(x, y, repeat=5)
    acc = np.mean(acc_scores)
    var = np.var(acc_scores)
    print(f"ACC: {acc:.4f} {var:.6f}")

    f1 = np.mean(f1_scores)
    var = np.var(f1_scores)
    print(f" F1: {f1:.4f} {var:.6f}\n")

    # if num_nodes < 50000:
    #     os.makedirs(f"figures", exist_ok=True)
    #     fn = f"{dataset.name}_t{trial}-e{best_epoch + 1}-k{args.k}-tau{args.tau}-alpha{args.alpha}-drop_e{args.drop_edge_p:.2f}-drop_f{args.drop_feat_p:.2f}-acc{acc:.4f}"
    #     plot_tsne_hypersphere(x, labels=y, n_components=2, save_path=f"figures/{fn}-tsne.png")
        # plot_umap_hypersphere(x, labels=y, n_components=2, save_path=f"figures/{fn}-umap.png")
        # plot_loss_with_moving_avg(
        #     [it['loss'] for it in loss_hist], window=10,
        #     save_path=f"figures/{dataset.name}-{trial}-loss.png")

    return acc, f1, nmi, ari


def main(args):
    acc_list, f1_list, nmi_list, ari_list = [], [], [], []
    for trial in range(args.trials):
        print(f"\nTrial {trial + 1:>3d}/{args.trials}")
        acc, f1, nmi, ari = run(args, trial + 1)

        acc_list.append(acc)
        f1_list.append(f1)
        nmi_list.append(nmi)
        ari_list.append(ari)

    acc_list = np.array(acc_list)
    f1_list = np.array(f1_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/results_{args.dataset}.txt"

    with open(result_path, 'a') as f:
        cfg = f"""
            backbone: {args.backbone}
            num_layers: {args.num_layers}
            hidden_dim: {args.hidden_dim}
            trials: {args.trials}
            loss_mode: {args.loss_mode}
            init_kappa: {args.init_kappa}
            init_kappa_align: {args.init_kappa_align}
            \n
            """
        cfg = re.sub(r"\n", "", cfg)
        cfg = re.sub(r"\s+", " ", cfg).strip()
        f.write(cfg + "\n")
        f.write(f"Acc: {np.mean(acc_list):.4f} {np.std(acc_list):.4f}\n")
        f.write(f"F1 : {np.mean(f1_list):.4f} {np.std(f1_list):.4f}\n")
        f.write(f"NMI: {np.mean(nmi_list):.4f} {np.std(nmi_list):.4f}\n")
        f.write(f"ARI: {np.mean(ari_list):.4f} {np.std(ari_list):.4f}\n")

    print("\n======= Summary =======")
    print(cfg)
    print(f"Acc: {np.mean(acc_list):.4f} {np.std(acc_list):.4f}")
    print(f"F1 : {np.mean(f1_list):.4f} {np.std(f1_list):.4f}")
    print(f"NMI: {np.mean(nmi_list):.4f} {np.std(nmi_list):.4f}")
    print(f"ARI: {np.mean(ari_list):.4f} {np.std(ari_list):.4f}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hierarchical Scatter On Hypersphere')
    parser.add_argument('--dataset', type=str, default='Photo', help='dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/home/chenrui/datasets/chenrui/graph/datasets/', help='dataset dir')
    parser.add_argument('--ckpt_dir', type=str, default='/home/chenrui/datasets/chenrui/graph/SGRL/checkpoints', help='checkpoint dir')
    parser.add_argument('--result_dir', type=str, default='results_bayes', help='directory to save result files')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=1500, help='max epochs')
    parser.add_argument('--k', type=int, default=1, help='k order neighbor mean')
    parser.add_argument('--h_target', type=float, default=1.5, help='target entropy')
    parser.add_argument('--drop_edge_p', type=float, default=0.8, help='drop_edge_p')
    parser.add_argument('--drop_feat_p', type=float, default=0.1, help='drop_feat_p')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--trials', type=int, default=3, help='trials')
    parser.add_argument('--cluster_method', type=str, default="kmeans")
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='cuda or cpu')
    
    parser.add_argument('--loss_mode', type=str, default='original', choices=['original', 'bayes'], help='training loss mode')
    parser.add_argument('--backbone', type=str, default='transformer', choices=['gcn', 'gat', 'sage', 'transformer'], help='Backbone encoder type')

    parser.add_argument('--thermo_lr', type=float, default=1e-3, help='learning rate for Bayesian thermodynamics loss parameters')
    parser.add_argument('--init_kappa', type=float, default=1.0, help='initial KDE bandwidth kappa')
    parser.add_argument('--init_kappa_align', type=float, default=-1.0, help='initial alignment concentration; if <=0, use hidden_dim')

    parser.add_argument("--thermo_mode", type=str, default="adaptive", choices=["adaptive", "fixed_T"],)
    parser.add_argument("--fixed_T", type=float, default=5e-4,)

    parsed_args = parser.parse_args()
    main(parsed_args)
    exit(0)
