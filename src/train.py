# train_eval_baseline.py
import os
import glob
import random
import math
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# torch-geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

import numpy as np

# ----------------------------
# Dataset: load .pt и place clustering (greedy)
# ----------------------------

# helper загрузки нормализации (разово при инициализации)
def load_norms(norms_path):
    if not os.path.isfile(norms_path):
        print(f"[norms] warning: {norms_path} not found -> skipping normalization")
        return None
    with open(norms_path, 'r', encoding='utf-8') as fh:
        norms = json.load(fh)
    # convert lists -> numpy arrays
    if 'node_mean' in norms:
        norms['node_mean'] = np.array(norms['node_mean'], dtype=float)
        norms['node_std']  = np.array(norms['node_std'], dtype=float)
    if 'edge_mean' in norms:
        norms['edge_mean'] = np.array(norms['edge_mean'], dtype=float)
        norms['edge_std']  = np.array(norms['edge_std'], dtype=float)
    # indices optionally present (from compute_norms_filtered)
    if 'node_cont_indices' in norms:
        norms['node_cont_indices'] = list(norms['node_cont_indices'])
    if 'edge_cont_indices' in norms:
        norms['edge_cont_indices'] = list(norms['edge_cont_indices'])
    return norms



class GraphFilesDataset:
    """
    Лёгкая обёртка: держит список загруженных dict'ов и массив pose, place_labels.
    НЕ наследует torch.utils.data.Dataset потому что мы будем делать Subset-объекты.
    """
    def __init__(self, root_dir, pos_threshold=2.0):
        files = sorted(glob.glob(os.path.join(root_dir, '*.pt')))
        assert len(files) > 0, f"No .pt files in {root_dir}"
        self.files = files
        self.data_list = []
        self.poses = []
        for p in files:
            d = torch.load(p)
            # defensive: ensure tensor x
            if isinstance(d.get('x'), np.ndarray):
                d['x'] = torch.tensor(d['x'], dtype=torch.float32)
            self.data_list.append(d)
            pose = d.get('pose')
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            self.poses.append(np.array(pose, dtype=float))
        self.poses = np.array(self.poses)  # (N,3)
        self.pos_threshold = float(pos_threshold)
        self.place_labels = self._build_place_labels()
        self.label2indices = self._make_label2indices()

    def _build_place_labels(self):
        N = len(self.poses)
        labels = np.full(N, -1, dtype=int)
        unassigned = set(range(N))
        cur = 0
        while unassigned:
            i = unassigned.pop()
            center = self.poses[i]
            rem = np.array(list(unassigned)) if len(unassigned) > 0 else np.array([], dtype=int)
            if rem.size > 0:
                dists = np.linalg.norm(self.poses[rem] - center[None, :], axis=1)
                close = rem[dists <= self.pos_threshold]
            else:
                close = np.array([], dtype=int)
            members = np.concatenate(([i], close)) if close.size > 0 else np.array([i], dtype=int)
            for m in members:
                if m in unassigned:
                    unassigned.remove(int(m))
                labels[int(m)] = cur
            labels[i] = cur
            cur += 1
        print(f"[Dataset] {len(self.files)} samples -> {cur} places (pos_threshold={self.pos_threshold}m)")
        return labels

    def _make_label2indices(self):
        d = defaultdict(list)
        for idx, lab in enumerate(self.place_labels):
            d[int(lab)].append(idx)
        return d

# ----------------------------
# Subset wrapper (по индексам) — возвращает pyg.Data и обновлённые метки
# ----------------------------
class SubsetGraphDataset(Dataset):
    def __init__(self, base: GraphFilesDataset, indices: list, norms_path=None):
        self.base = base
        self.indices = list(indices)
        # build local place labels and mapping
        self.place_labels = np.array([int(base.place_labels[i]) for i in self.indices])
        # remap place ids to compact 0..P-1 (important for label2indices in subset)
        unique_places = sorted(set(self.place_labels.tolist()))
        self.place_map = {p: idx for idx, p in enumerate(unique_places)}
        self.remapped_labels = np.array([self.place_map[p] for p in self.place_labels])
        # build label->local idx mapping
        d = defaultdict(list)
        self.norms = load_norms(norms_path) if norms_path is not None else None
        for local_idx, lab in enumerate(self.remapped_labels):
            d[int(lab)].append(local_idx)
        self.label2indices = d

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        orig_idx = self.indices[local_idx]
        d = self.base.data_list[orig_idx]
        x = d['x']  # torch.Tensor or numpy
        ea = d.get('edge_attr', None)
        # ensure numpy arrays for math
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.array(x, dtype=float)

        # apply normalization if available
        if self.norms is not None and 'node_mean' in self.norms:
            # default: if no cont_indices provided -> normalize all numeric except index 0 (class)
            cont_idx = self.norms.get('node_cont_indices', None)
            if cont_idx is None:
                # default heuristic: skip index 0 (class_idx)
                cont_idx = list(range(1, x_np.shape[1]))
            mean = self.norms['node_mean']
            std  = self.norms['node_std']
            # guard shapes: mean/std length might be smaller (if you used filtered compute)
            # map cont_idx -> positions in mean/std
            for j, feat_idx in enumerate(cont_idx):
                if feat_idx < len(mean):
                    x_np[:, feat_idx] = (x_np[:, feat_idx] - mean[j]) / (std[j] if std[j] > 1e-8 else 1.0)

        # same for edge_attr
        if ea is not None:
            if isinstance(ea, torch.Tensor):
                ea_np = ea.cpu().numpy()
            else:
                ea_np = np.array(ea, dtype=float)
            if self.norms is not None and 'edge_mean' in self.norms:
                cont_eidx = self.norms.get('edge_cont_indices', None)
                if cont_eidx is None:
                    cont_eidx = list(range(0, ea_np.shape[1]-1))  # default: skip last label
                emean = self.norms['edge_mean']; estd = self.norms['edge_std']
                for j, feat_idx in enumerate(cont_eidx):
                    if feat_idx < ea_np.shape[1] and j < len(emean):
                        ea_np[:, feat_idx] = (ea_np[:, feat_idx] - emean[j]) / (estd[j] if estd[j] > 1e-8 else 1.0)
            # convert back to tensor
            ea = torch.tensor(ea_np, dtype=torch.float32)
        # convert x back to tensor
        x = torch.tensor(x_np, dtype=torch.float32)

        # build Data object
        edge_index = d.get('edge_index', None)
        if edge_index is None:
            edge_index = torch.empty((2,0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=ea, pose=torch.tensor(d.get('pose', None)) if d.get('pose', None) is not None else None)
        label = int(self.remapped_labels[local_idx])  # как у тебя
        return data, label

    def get_label2indices(self):
        return self.label2indices

# ----------------------------
# Triplet dataset (на базе SubsetGraphDataset)
# ----------------------------
class TripletGraphDataset(Dataset):
    def __init__(self, subset_dataset: SubsetGraphDataset):
        self.base = subset_dataset
        self.label2idx = self.base.get_label2indices()
        self.indices = list(range(len(self.base)))

    def __len__(self):
        return len(self.base)

    def sample_pos(self, anchor_local_idx, anchor_label):
        candidates = self.label2idx[anchor_label]
        candidates = [i for i in candidates if i != anchor_local_idx]
        if len(candidates) == 0:
            return anchor_local_idx
        return random.choice(candidates)

    def sample_neg(self, anchor_label):
        other_labels = [l for l in self.label2idx.keys() if l != anchor_label]
        if not other_labels:
            return random.choice(self.indices)
        neg_label = random.choice(other_labels)
        return random.choice(self.label2idx[neg_label])

    def __getitem__(self, idx):
        anchor, anchor_label = self.base[idx]
        pos_idx = self.sample_pos(idx, anchor_label)
        neg_idx = self.sample_neg(anchor_label)
        pos, _ = self.base[pos_idx]
        neg, _ = self.base[neg_idx]
        return anchor, pos, neg

# ----------------------------
# collate
# ----------------------------
def triplet_collate(batch):
    anchors, poss, negs = zip(*batch)
    batch_a = Batch.from_data_list(list(anchors))
    batch_p = Batch.from_data_list(list(poss))
    batch_n = Batch.from_data_list(list(negs))
    return batch_a, batch_p, batch_n

# ----------------------------
# simple GNN
# ----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2, proj_dim=128, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pool_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(self.pool_dim, self.pool_dim),
            nn.ReLU(),
            nn.Linear(self.pool_dim, proj_dim)
        )

    def forward(self, batch_data: Batch):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.drop(h)
        hg = global_mean_pool(h, batch)
        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        return z

# ----------------------------
# helpers: split places -> train/test indices (places целиком)
# ----------------------------
def split_train_test_by_place(base_ds: GraphFilesDataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    place2idx = base_ds.label2indices
    places = list(place2idx.keys())
    n_test_places = max(1, int(len(places) * test_ratio))
    test_places = set(random.sample(places, n_test_places))
    train_places = set(places) - test_places

    train_idxs = []
    test_idxs = []
    for p in train_places:
        train_idxs.extend(place2idx[p])
    for p in test_places:
        test_idxs.extend(place2idx[p])

    train_idxs = sorted(train_idxs)
    test_idxs = sorted(test_idxs)
    print(f"Split: {len(train_idxs)} train samples, {len(test_idxs)} test samples, test_places={len(test_places)}")
    return train_idxs, test_idxs, sorted(list(test_places))

# ----------------------------
# build gallery/query from test indices
# ----------------------------
def build_gallery_query(base_ds: GraphFilesDataset, test_indices, seed=123):
    """
    Для каждого place в тесте: если place имеет >=2 сэмплов -> выбрать 1 как query, остальные -> gallery.
    Если place имеет 1 сэмпл -> только gallery.
    Возвращает lists: gallery_indices (orig idx), query_indices (orig idx)
    """
    random.seed(seed)
    place2idx = base_ds.label2indices
    test_places = sorted(set([base_ds.place_labels[i] for i in test_indices]))
    gallery = []
    queries = []
    for p in test_places:
        idxs = [i for i in place2idx[p] if i in test_indices]
        if len(idxs) >= 2:
            q = random.choice(idxs)
            queries.append(q)
            for i in idxs:
                if i != q:
                    gallery.append(i)
        elif len(idxs) == 1:
            gallery.append(idxs[0])
        else:
            # shouldn't happen
            pass
    print(f"Built gallery ({len(gallery)}) and queries ({len(queries)}) from test set")
    return gallery, queries

# ----------------------------
# compute descriptors for a given list of original indices
# ----------------------------
def compute_descriptors_for_indices(model, base_ds: GraphFilesDataset, indices, device, batch_size=64):
    model.eval()
    descs = []
    idx_order = []
    loader_items = []
    for idx in indices:
        d = base_ds.data_list[idx]
        x = d['x']; edge_index = d.get('edge_index', None)
        if edge_index is None:
            edge_index = torch.empty((2,0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=d.get('edge_attr', None))
        loader_items.append(data)
        idx_order.append(idx)
    loader = DataLoader(loader_items, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: Batch.from_data_list(batch))
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            z = model(b)  # (B,D)
            descs.append(z.cpu())
    descs = torch.cat(descs, dim=0)  # (M,D) where M = len(indices)
    return descs, idx_order

# ----------------------------
# Recall@K: queries vs gallery (по original indices)
# ----------------------------
def compute_recall_at_k(model, base_ds: GraphFilesDataset, gallery_idxs, query_idxs, device, ks=(1,5,10)):
    model.eval()
    # descriptors
    gallery_descs, gallery_order = compute_descriptors_for_indices(model, base_ds, gallery_idxs, device)
    query_descs, query_order = compute_descriptors_for_indices(model, base_ds, query_idxs, device)

    # mapping from orig idx to place label
    place_labels = base_ds.place_labels  # original place labels
    # compute pairwise distances (queries x gallery)
    dists = torch.cdist(query_descs, gallery_descs, p=2)  # (Q, G)
    Q = dists.shape[0]
    recalls = {}
    # for each query check top-k
    if Q == 0 or gallery_descs.shape[0] == 0:
        for k in ks:
            recalls[k] = 0.0
        return recalls

    sorted_idxs = torch.argsort(dists, dim=1)  # ascending
    for k in ks:
        correct = 0
        for i_q, q_orig in enumerate(query_order):
            topk = sorted_idxs[i_q, :k].cpu().numpy()
            topk_orig = [gallery_order[t] for t in topk]
            q_label = place_labels[q_orig]
            # positive if any of topk has same place label
            found = any(place_labels[g] == q_label for g in topk_orig)
            if found:
                correct += 1
        recalls[k] = correct / max(1, Q)
    return recalls

# ----------------------------
# training loop
# ----------------------------
def train_baseline(root_dir,
                   epochs=10,
                   batch_size=8,
                   lr=1e-3,
                   margin=0.3,
                   pos_threshold=2.0,
                   test_ratio=0.2,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Device:", device)
    base_ds = GraphFilesDataset(root_dir, pos_threshold=pos_threshold)

    # split places -> train/test (no leakage)
    train_idxs, test_idxs, test_places = split_train_test_by_place(base_ds, test_ratio=test_ratio, seed=42)

    # build gallery and queries from test
    gallery_idxs, query_idxs = build_gallery_query(base_ds, test_idxs, seed=123)

    # Subset datasets
    train_subset = SubsetGraphDataset(base_ds, train_idxs, norms_path=os.path.join(root_dir, "norms.json"))
    triplet_ds = TripletGraphDataset(train_subset)
    loader = DataLoader(triplet_ds, batch_size=batch_size, shuffle=True, collate_fn=triplet_collate, num_workers=4)

    # model
    sample_data, _ = train_subset[0]
    in_dim = sample_data.x.shape[1]
    model = SimpleGNN(in_dim=in_dim, hidden_dim=128, n_layers=2, proj_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        it = 0
        for batch_a, batch_p, batch_n in loader:
            batch_a = batch_a.to(device); batch_p = batch_p.to(device); batch_n = batch_n.to(device)
            z_a = model(batch_a); z_p = model(batch_p); z_n = model(batch_n)
            loss = criterion(z_a, z_p, z_n)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item()); it += 1
        avg_loss = total_loss / max(1, it)
        # eval on test
        recalls = compute_recall_at_k(model, base_ds, gallery_idxs, query_idxs, device, ks=(1,5,10))
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | Recall@1={recalls[1]:.4f} Recall@5={recalls[5]:.4f} Recall@10={recalls[10]:.4f}")

    torch.save(model.state_dict(), "vpr_baseline_split_model.pt")
    print("Saved vpr_baseline_split_model.pt")
    return model, base_ds, gallery_idxs, query_idxs

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="путь к папке с .pt (res)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--pos", type=float, default=2.0, help="threshold (m) для clustering places")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="доля мест для теста (по place)")
    args = parser.parse_args()

    train_baseline(args.root, epochs=args.epochs, batch_size=args.batch, lr=args.lr, margin=args.margin, pos_threshold=args.pos, test_ratio=args.test_ratio)