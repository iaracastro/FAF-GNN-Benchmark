import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
    to_undirected,
    add_self_loops,
    remove_self_loops,
    coalesce,
    degree,
)

import json
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_scatter import scatter
import matplotlib.pyplot as plt


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


cora    = Planetoid(root='data/Cora',   name='Cora')
pubmed  = Planetoid(root='data/PubMed', name='PubMed')
amazon  = Amazon(root='data/Amazon',    name='Photo')

cora_canonical_train_mask = cora[0].train_mask.clone()
cora_canonical_val_mask   = cora[0].val_mask.clone()
cora_canonical_test_mask  = cora[0].test_mask.clone()

pubmed_canonical_train_mask = pubmed[0].train_mask.clone()
pubmed_canonical_val_mask   = pubmed[0].val_mask.clone()
pubmed_canonical_test_mask  = pubmed[0].test_mask.clone()

def make_split(data, num_train_per_class=20, num_val=500, num_test=1000, seed=42):
    torch.manual_seed(seed)
    n = data.num_nodes
    C = data.y.max().item() + 1
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    for c in range(C):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx))]
        train_mask[perm[:num_train_per_class]] = True

    # Remaining nodes for validation and test according to the specified numbers
    remaining_idx = (~train_mask).nonzero(as_tuple=True)[0]
    perm = remaining_idx[torch.randperm(len(remaining_idx))]
    val_mask[perm[:num_val]] = True
    test_mask[perm[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask

amazon_train_mask, amazon_val_mask, amazon_test_mask = make_split(amazon[0], seed=0)

data = cora[0]
print(data)
print(pubmed[0])
print(amazon[0])
# Atributos relevantes: data.x, data.edge_index, data.y
# data.x:          [num_nodes, num_features]
# data.edge_index: [2, num_edges]  — formato COO
# data.y:          [num_nodes]


def inspect(data, name):
    G = to_networkx(data, to_undirected=True)
    print(f"\n=== {name} ===")
    print(f"Nós:           {data.num_nodes}")
    print(f"Arestas:       {data.num_edges // 2}")
    print(f"Features/nó:  {data.num_features}")
    print(f"Classes:       {data.y.max().item() + 1}")
    print(f"Grau médio:    {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    # Homofilia: fração de arestas entre nós da mesma classe
    src, dst = data.edge_index
    homofilia = (data.y[src] == data.y[dst]).float().mean().item()
    print(f"Homofilia:     {homofilia:.3f}")


def compute_faf(data, K=2, reducers=('mean', 'sum', 'max', 'min')):
    """
    Computes Fixed Aggregation Features:
        z_v = x_v ⊕_{r in R} ⊕_{k=1}^K h_v^{(k,r)}

    The aggregation is fixed and non-trainable.
    """
    x           = data.x.float()

    edge_index    = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index    = coalesce(edge_index, num_nodes=data.num_nodes)

    src, dst    = edge_index          # aresta dst <- src
    num_nodes   = data.num_nodes

    # Pré-calcular graus para normalização (redutor 'mean')
    # deg = degree(dst, num_nodes=num_nodes).clamp(min=1)

    features = [x]                   # começa com atributos originais
    h_prev = {r: x.clone() for r in reducers}

    for k in range(1, K + 1):
        h_curr = {}
        for r in reducers:
            h_in = h_prev[r]         # representação do hop anterior

            if r == 'mean':
                # Σ h_u / grau(v)  para u ∈ N(v)
                agg = scatter(h_in[src], dst, dim=0,
                              dim_size=num_nodes, reduce='sum')
                
                deg = degree(dst, num_nodes=num_nodes).clamp(min=1)
                agg = agg / deg.unsqueeze(1)

            elif r == 'sum':
                agg = scatter(h_in[src], dst, dim=0,
                              dim_size=num_nodes, reduce='sum')

            elif r == 'max':
                agg = scatter(h_in[src], dst, dim=0,
                              dim_size=num_nodes, reduce='max')

            elif r == 'min':
                agg = scatter(h_in[src], dst, dim=0,
                              dim_size=num_nodes, reduce='min')

            else:
                raise ValueError(f"Unknown reducer: {r}")

            h_curr[r] = agg
            features.append(agg)    # concatena h_v^(k,r)

        h_prev = h_curr

    z = torch.cat(features, dim=1)  # [num_nodes, d*(1 + K*|R|)]
    return z


def standardize_by_train(x, train_mask, eps=1e-8):
    """
    Standardizes features using only training nodes.
    This prevents test leakage.
    """
    mean = x[train_mask].mean(dim=0, keepdim=True)
    std = x[train_mask].std(dim=0, keepdim=True)
    std = std.clamp(min=eps)
    return (x - mean) / std


def randomize_graph(data, seed=42, nswap_factor=10):
    """
    Degree-preserving randomization using NetworkX double-edge swaps.

    Keeps:
        - same nodes
        - same node features
        - same labels
        - same train/val/test masks

    Changes:
        - edge topology
    """

    G = to_networkx(data, to_undirected=True)
    G_rand = G.copy()

    nswap     = nswap_factor * G.number_of_edges()
    max_tries = max(100, 20 * nswap)   # sempre maior que nswap — evita o NetworkXError

    nx.double_edge_swap(G_rand, nswap=nswap, max_tries=max_tries, seed=seed)

    edge_index            = from_networkx(G_rand).edge_index
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=data.num_nodes)

    data_rand = copy.copy(data)
    data_rand.edge_index = edge_index

    return data_rand

 
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_dim, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
 
 
def train_and_eval_gcn(model, data, epochs=500, lr=0.01, wd=5e-4):
    """Separate loop for GCN: forward needs edge_index as extra argument."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test = -1.0, -1.0
    x = data.x.float()
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()

        out  = model(x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        opt.step()

        train_losses.append(loss.item())
 
        model.eval()
        with torch.no_grad():
            out_eval  = model(x, data.edge_index)
            pred = out_eval.argmax(dim=1)
            val_acc  = (pred[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val  = val_acc
                best_test = test_acc
            
            val_loss = F.cross_entropy(out_eval[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())
 
    return best_test, train_losses, val_losses
 


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

def train_and_eval_mlp(model, x, data, epochs=500, lr=0.01, wd=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test = -1.0, -1.0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(x)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            out_eval = model(x)
            pred = out_eval.argmax(dim=1)
            val_acc  = (pred[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val  = val_acc
                best_test = test_acc
            
            val_loss = F.cross_entropy(out_eval[data.val_mask], data.y[data.val_mask]).item()
            val_losses.append(val_loss)

    return best_test, train_losses, val_losses


## ── Cell: run all four models on one dataset ────────────────────────────────
## Usage: change `data` to cora[0], pubmed[0], or amazon[0] as needed.
## For Amazon Photo, replace the masks with make_split() output first.
 
def run_experiment(
        data,
        hidden_dim=64,
        epochs=300,
        K=2,
        reducers=("mean", "sum", "max", "min"),
        seed=0,
        normalize_faf=True,
        device='cpu',
        run_mlp_x=True,
        run_faf_true=True,
        run_faf_rand=True,
        run_gcn=True,
    ):
    """
    Trains and evaluates MLP-X, FAF-True, FAF-Random, GCN on `data`.
    Returns a dict with accuracies and the three delta values.
 
    Parameters
    ----------
    data       : PyG Data object (with .x, .edge_index, .y, train/val/test masks)
    hidden_dim : hidden layer width — kept identical across all MLP heads
    epochs     : number of training epochs
    K          : number of aggregation hops for FAF
    seed       : seed for reproducibility
    """
    set_seed(seed)
    data = data.to(device)

    results = {}
    loss_hist = {}
 
    num_classes = int(data.y.max().item() + 1)
    num_features = data.num_features

    # ── 1. MLP-X  (no graph information) ────────────────────────────────────
    if run_mlp_x:
        x_mlp = data.x.float()
        mlp_x = MLP(in_dim=num_features,
                    hidden_dim=hidden_dim,
                    out_dim=num_classes)
        num_params = sum(p.numel() for p in mlp_x.parameters() if p.requires_grad)
        mlp_x.to(device)
        x_mlp = x_mlp.to(device)
        start_time = time.time()
        acc_mlp_x, train_losses, val_losses = train_and_eval_mlp(mlp_x, x_mlp, data, epochs=epochs)
        execution_time = time.time() - start_time
        loss_hist['mlp_x'] = {'train': train_losses, 'val': val_losses}
        print(f"MLP-X          acc = {acc_mlp_x:.4f}\ttime: {execution_time:.2f} s\tparams: {num_params}")
        results['mlp_x'] = {
            'acc': acc_mlp_x,
            'time': execution_time,
            'params': num_params,
        }
        del mlp_x, x_mlp
 
    # ── 2. FAF-True + MLP  (fixed aggregation on the real graph) ────────────
    z_true   = compute_faf(data, K=K, reducers=reducers)          # [N, d*(1 + K*|R|)]
    faf_num_features = z_true.shape[1]
    
    if run_faf_true:
        if normalize_faf:
            z_true = standardize_by_train(z_true, data.train_mask)

        faf_true = MLP(in_dim=faf_num_features,
                    hidden_dim=hidden_dim,
                    out_dim=num_classes)
        num_params = sum(p.numel() for p in faf_true.parameters() if p.requires_grad)
        faf_true.to(device)
        start_time = time.time()
        acc_faf_true, train_losses, val_losses = train_and_eval_mlp(faf_true, z_true, data, epochs=epochs)
        execution_time = time.time() - start_time
        loss_hist['faf_true'] = {'train': train_losses, 'val': val_losses}
        print(f"FAF-True+MLP   acc = {acc_faf_true:.4f}\ttime: {execution_time:.2f} s\tparams: {num_params}")
        results['faf_true'] = {
            'acc': acc_faf_true,
            'time': execution_time,
            'params': num_params,
        }
        del faf_true, z_true
 
    # ── 3. FAF-Random + MLP  (same aggregation on a degree-preserving random graph)
    if run_faf_rand:
        data_rand  = randomize_graph(data, seed=seed)
        data_rand = data_rand.to(device)
        z_rand     = compute_faf(data_rand, K=K, reducers=reducers)

        if normalize_faf:
            z_rand = standardize_by_train(z_rand, data.train_mask)

        faf_rand   = MLP(in_dim=faf_num_features,
                        hidden_dim=hidden_dim,
                        out_dim=num_classes)
        num_params = sum(p.numel() for p in faf_rand.parameters() if p.requires_grad)
        faf_rand.to(device)

        # NOTE: train/val/test masks come from the original data — labels are unchanged
        start_time = time.time()
        acc_faf_rand, train_losses, val_losses = train_and_eval_mlp(faf_rand, z_rand, data, epochs=epochs)
        execution_time = time.time() - start_time
        loss_hist['faf_rand'] = {'train': train_losses, 'val': val_losses}
        print(f"FAF-Random+MLP acc = {acc_faf_rand:.4f}\ttime: {execution_time:.2f} s\tparams: {num_params}")
        results['faf_rand'] = {
            'acc': acc_faf_rand,
            'time': execution_time,
            'params': num_params,
        }
        del faf_rand, z_rand, data_rand

    # ── 4. GCN  (learned message passing) ───────────────────────────────────
    if run_gcn:
        gcn     = GCN(in_dim=num_features,
                    hidden_dim=hidden_dim,
                    out_dim=num_classes)
        num_params = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
        gcn.to(device)
        start_time = time.time()
        acc_gcn, train_losses, val_losses = train_and_eval_gcn(gcn, data, epochs=epochs)
        execution_time = time.time() - start_time
        loss_hist['gcn'] = {'train': train_losses, 'val': val_losses}
        print(f"GCN            acc = {acc_gcn:.4f}\ttime: {execution_time:.2f} s\tparams: {num_params}")
        results['gcn'] = {
            'acc': acc_gcn,
            'time': execution_time,
            'params': num_params,
        }
        del gcn

    # ── Delta decomposition ──────────────────────────────────────────────────
    if run_mlp_x and run_faf_true and run_faf_rand and run_gcn:
        delta_graph     = acc_faf_true - acc_mlp_x
        delta_structure = acc_faf_true - acc_faf_rand
        delta_random_vs_x = acc_faf_rand - acc_mlp_x
        delta_learned   = acc_gcn      - acc_faf_true
        results['deltas'] = {
            'graph': delta_graph,
            'structure': delta_structure,
            'random_vs_x': delta_random_vs_x,
            'learned': delta_learned,
        }

        residuo = delta_graph - delta_structure - delta_random_vs_x
    
        print(f"\nΔ_graph     = {delta_graph:+.4f}  (graph info vs. no graph)")
        print(f"Δ_structure = {delta_structure:+.4f}  (real topology vs. random graph)")
        print(f"Δ_random_vs_x  = {delta_random_vs_x:+.4f}  (FAF-Random - MLP-X)")
        print(f"Δ_learned      = {delta_learned:+.4f}  (GCN - FAF-True)")
        print(f"Resíduo        = {residuo:.6f}  (should be 0)")
 
    return results, loss_hist


data_cora = cora[0]
data_pubmed = pubmed[0]
data_amazon = amazon[0]

datasets = {
    'cora': data_cora,
    'pubmed': data_pubmed,
    'amazon': data_amazon,
}

ablation_results = {}
for k in range(1, 4):
    print(f"\n=== K = {k} ===")
    ablation_results[k] = {}
    for seed in range(30):
        ablation_results[k][seed] = {}
        for dataset_name, data in datasets.items():
            data.train_mask, data.val_mask, data.test_mask = make_split(data, seed=seed)
            results, _ = run_experiment(
                data,
                K=k,
                seed=seed,
                device=device,
                run_mlp_x=False,
                run_gcn=False,
            )
            ablation_results[k][seed][dataset_name] = results

# Save ablation results to a JSON file
with open('results/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=1)


data_cora.train_mask, data_cora.val_mask, data_cora.test_mask = cora_canonical_train_mask.clone(), cora_canonical_val_mask.clone(), cora_canonical_test_mask.clone()
data_pubmed.train_mask, data_pubmed.val_mask, data_pubmed.test_mask = pubmed_canonical_train_mask.clone(), pubmed_canonical_val_mask.clone(), pubmed_canonical_test_mask.clone()
data_amazon.train_mask, data_amazon.val_mask, data_amazon.test_mask = amazon_train_mask.clone(), amazon_val_mask.clone(), amazon_test_mask.clone()

final_results = {}

for seed in range(30):
    final_results[seed] = {}
    for dataset_name, data in datasets.items():
        results, _ = run_experiment(
            data,
            seed=seed,
            device=device,
        )
        final_results[seed][dataset_name] = results

# Save final results to a JSON file
with open('results/final_results.json', 'w') as f:
    json.dump(final_results, f, indent=1)