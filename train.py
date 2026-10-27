import argparse
import os
import re
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.transforms import ToUndirected
from transforms import get_graph_drop_transform
from loss import BayesianThermodynamicsLoss
from data import load_dataset, get_splits, eval_logistic_regression_random_splits, eval_logistic_regression, eval_clustering
from torch_geometric.utils import is_undirected
import torch
import random
import torch_geometric
import numpy as np

torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])

torch.manual_seed(3456)
np.random.seed(3456)
random.seed(3456)

def l2_normalize(x, eps=1e-16):
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

# ---------------- models ----------------
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()

        self.activation = torch.nn.SiLU()
        self.layers = torch.nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim // 16, heads=16, beta=False))
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

def compute_loss_components(z, edge_index, num_nodes, k=1, tau=5):
    nb_mean = z
    nb_count = torch.zeros((num_nodes, 1), device=z.device)
    for _ in range(k):
        nb_mean, nb_count = neighbor_mean(nb_mean, edge_index, num_nodes, add_self=True)
        nb_mean = l2_normalize(nb_mean)

    cos_sim = (z * nb_mean).sum(dim=1)
    w = torch.pow(torch.sigmoid(nb_count), tau)
    align_loss = (w * (1. - cos_sim)).mean()

    collapse_metric = z.mean(dim=0)
    unif_loss = (collapse_metric * collapse_metric).sum()

    H_proxy = -torch.log(unif_loss + 1e-6)

    components = {
        "L_align": align_loss.item(),
        "L_unif": unif_loss.item(),
    }
    return align_loss, unif_loss, components

# ---------------- load data ----------------
def run(args, trial):

    dataset = load_dataset(args.dataset, args.dataset_dir)
    num_classes = dataset.num_classes
    data = dataset[0]
    print(is_undirected(data.edge_index, num_nodes=data.num_nodes))

    # prepare transforms
    transform = get_graph_drop_transform(drop_edge_p=args.drop_edge_p, drop_feat_p=args.drop_feat_p)
    dataset_name = getattr(dataset, "name", args.dataset)
    print(dataset_name, dataset[0], '\n')

    if len(data.y.shape) > 1:
        data.y = data.y.squeeze(1)

    num_nodes = data.x.shape[0]
    input_dim = data.x.shape[1]

    data = ToUndirected()(data).to(args.device)

    encoder = GraphEncoder(input_dim, args.hidden_dim, num_layers=args.num_layers).to(args.device)
    criterion = BayesianThermodynamicsLoss(
        embedding_dim=args.hidden_dim,
        init_kappa=args.init_kappa,
        init_kappa_align=args.init_kappa_align if args.init_kappa_align > 0 else None
    ).to(args.device)

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

    # ---------------- training ----------------
    epochs = args.num_epochs
    loss_hist = []
    nmi_hist = {}
    ari_hist = {}

    best_loss = 1e9
    best_epoch = 0
    patience = 50

    for epoch in range(epochs):
        encoder.train()

        arg_data = transform(data)

        h, o = encoder(arg_data.x, arg_data.edge_index)  # [N,z_dim]
        z = l2_normalize(h)

        align_loss, unif_loss, comps = compute_loss_components(
            z, arg_data.edge_index, data.num_nodes, k=args.k, tau=args.tau
        )

        loss = criterion(align_loss, unif_loss)
        thermo_stats = criterion.current_thermodynamics_status

        comps["loss"] = loss.item()
        comps["kappa_align"] = thermo_stats["kappa_align"]
        comps["kappa"] = thermo_stats["kappa"]
        comps["T_eff"] = thermo_stats["effective_T"]
        comps["w_align"] = thermo_stats["w_align"]
        comps["w_unif"] = thermo_stats["w_unif"]

        loss_hist.append(comps)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            loss_msg = " | ".join([
                f"{k} {v:.8f}" if isinstance(v, (int, float)) else f"{k} {v}"
                for k, v in loss_hist[-1].items()
            ])

            nmi, ari = eval_clustering(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), num_classes)
            nmi_hist[epoch] = nmi
            ari_hist[epoch] = ari
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

    if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'CS', 'Physics']:
        acc_scores, f1_scores = eval_logistic_regression(x, y, repeat=3)
    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin','Actor']:
        splits = get_splits(num_samples=x.shape[0], train_ratio=0.48, valid_ratio=0.32, test_ratio=0.20, num_splits=3, seed=3456)
        acc_scores, f1_scores = eval_logistic_regression_random_splits(x, y, splits)
    else:
        raise ValueError('Unknown dataset!!!') 

    acc = np.mean(acc_scores)
    var = np.var(acc_scores)
    print(f"ACC: {acc:.4f} {var:.6f}")

    f1 = np.mean(f1_scores)
    var = np.var(f1_scores)
    print(f" F1: {f1:.4f} {var:.6f}\n")

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
            num_layers: {args.num_layers}
            hidden_dim: {args.hidden_dim}
            init_kappa: {args.init_kappa}
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
    parser = argparse.ArgumentParser('SeofGRL')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset dir')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint dir')
    parser.add_argument('--result_dir', type=str, default='results', help='directory to save result files')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=1500, help='max epochs')
    parser.add_argument('--k', type=int, default=1, help='k order neighbor mean')
    parser.add_argument('--tau', type=int, default=5, help='sensitivity to neighborhood density')
    parser.add_argument('--drop_edge_p', type=float, default=0.8, help='drop_edge_p')
    parser.add_argument('--drop_feat_p', type=float, default=0.1, help='drop_feat_p')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--trials', type=int, default=5, help='trials')
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='cuda or cpu')
    parser.add_argument('--thermo_lr', type=float, default=1e-3, help='learning rate for Bayesian thermodynamics loss parameters')
    parser.add_argument('--init_kappa', type=float, default=1.0, help='initial KDE bandwidth kappa')
    parser.add_argument('--init_kappa_align', type=float, default=-1.0, help='initial alignment concentration; if <=0, use hidden_dim')

    parsed_args = parser.parse_args()
    main(parsed_args)
    exit(0)
