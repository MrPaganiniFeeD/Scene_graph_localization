# train_eval_baseline.py  (updated)
import os
import glob
import random
import math
from collections import defaultdict
import json
import argparse

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

def load_norms(norms_path):
    if not os.path.isfile(norms_path):
        print(f"[norms] warning: {norms_path} not found -> skipping normalization")
        return None
    with open(norms_path, 'r', encoding='utf-8') as fh:
        norms = json.load(fh)
    if 'node_mean' in norms:
        norms['node_mean'] = np.array(norms['node_mean'], dtype=float)
        norms['node_std']  = np.array(norms['node_std'], dtype=float)
    if 'edge_mean' in norms:
        norms['edge_mean'] = np.array(norms['edge_mean'], dtype=float)
        norms['edge_std']  = np.array(norms['edge_std'], dtype=float)
    if 'node_cont_indices' in norms:
        norms['node_cont_indices'] = list(norms['node_cont_indices'])
    if 'edge_cont_indices' in norms:
        norms['edge_cont_indices'] = list(norms['edge_cont_indices'])
    return norms


class GraphFilesDataset:
    """
    Хранит список файлов .pt и предзагруженные словари.
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
        self.place_labels = np.array([int(base.place_labels[i]) for i in self.indices])
        unique_places = sorted(set(self.place_labels.tolist()))
        self.place_map = {p: idx for idx, p in enumerate(unique_places)}
        self.remapped_labels = np.array([self.place_map[p] for p in self.place_labels])
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
            cont_idx = self.norms.get('node_cont_indices', None)
            if cont_idx is None:
                cont_idx = list(range(1, x_np.shape[1]))
            mean = self.norms['node_mean']
            std  = self.norms['node_std']
            for j, feat_idx in enumerate(cont_idx):
                if feat_idx < len(mean):
                    x_np[:, feat_idx] = (x_np[:, feat_idx] - mean[j]) / (std[j] if std[j] > 1e-8 else 1.0)

        if ea is not None:
            if isinstance(ea, torch.Tensor):
                ea_np = ea.cpu().numpy()
            else:
                ea_np = np.array(ea, dtype=float)
            if self.norms is not None and 'edge_mean' in self.norms:
                cont_eidx = self.norms.get('edge_cont_indices', None)
                if cont_eidx is None:
                    cont_eidx = list(range(0, ea_np.shape[1]-1))
                emean = self.norms['edge_mean']; estd = self.norms['edge_std']
                for j, feat_idx in enumerate(cont_eidx):
                    if feat_idx < ea_np.shape[1] and j < len(emean):
                        ea_np[:, feat_idx] = (ea_np[:, feat_idx] - emean[j]) / (estd[j] if estd[j] > 1e-8 else 1.0)
            ea = torch.tensor(ea_np, dtype=torch.float32)
        x = torch.tensor(x_np, dtype=torch.float32)

        edge_index = d.get('edge_index', None)
        if edge_index is None:
            edge_index = torch.empty((2,0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=ea,
                    pose=torch.tensor(d.get('pose', None)) if d.get('pose', None) is not None else None)

        # --- DEBUG / bookkeeping attributes ---
        # original index in base dataset and filename (full path)
        data.orig_idx = torch.tensor(orig_idx, dtype=torch.long)
        data.filename = os.path.basename(self.base.files[orig_idx])

        label = int(self.remapped_labels[local_idx])
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
def _ensure_nonempty(data, feat_dim):
    if not hasattr(data, 'x') or data.x is None or data.x.numel() == 0 or data.x.shape[0] == 0:
        data.x = torch.zeros((1, feat_dim), dtype=torch.float32)
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            cols = data.edge_attr.shape[1] if (data.edge_attr is not None and data.edge_attr.ndim == 2) else 0
            data.edge_attr = torch.empty((0, cols), dtype=torch.float32)
    return data

def triplet_collate_with_padding(batch, feat_dim):
    anchors, poss, negs = zip(*batch)
    anchors = [_ensure_nonempty(a, feat_dim) for a in anchors]
    poss    = [_ensure_nonempty(p, feat_dim) for p in poss]
    negs    = [_ensure_nonempty(n, feat_dim) for n in negs]
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
            z = model(b)
            descs.append(z.cpu())
    descs = torch.cat(descs, dim=0)
    return descs, idx_order

# ----------------------------
# Recall@K
# ----------------------------
def compute_recall_at_k(model, base_ds: GraphFilesDataset, gallery_idxs, query_idxs, device,
                        ks=(1,5,10), radius=None, radius_mode='or', use_2d=False):
    model.eval()
    gallery_descs, gallery_order = compute_descriptors_for_indices(model, base_ds, gallery_idxs, device)
    query_descs, query_order = compute_descriptors_for_indices(model, base_ds, query_idxs, device)
    if query_descs.shape[0] == 0 or gallery_descs.shape[0] == 0:
        return {k: 0.0 for k in ks}
    dists = torch.cdist(query_descs, gallery_descs, p=2)
    pose_pdists = None
    if radius is not None:
        gallery_poses = torch.tensor(base_ds.poses[gallery_idxs], dtype=torch.float32)
        query_poses = torch.tensor(base_ds.poses[query_idxs], dtype=torch.float32)
        if use_2d:
            gallery_poses = gallery_poses[:, :2]
            query_poses = query_poses[:, :2]
        pose_pdists = torch.cdist(query_poses, gallery_poses, p=2)
    place_labels = base_ds.place_labels
    Q = dists.shape[0]
    sorted_idxs = torch.argsort(dists, dim=1)
    recalls = {}
    for k in ks:
        correct = 0
        for i_q, q_orig in enumerate(query_order):
            topk = sorted_idxs[i_q, :k].cpu().numpy()
            topk_orig = [gallery_order[t] for t in topk]
            q_label = int(place_labels[q_orig])
            found = False
            for g_orig, g_pos_in_gallery in zip(topk_orig, topk):
                place_match = (int(place_labels[g_orig]) == q_label)
                radius_match = False
                if radius is not None and pose_pdists is not None:
                    if float(pose_pdists[i_q, int(g_pos_in_gallery)]) <= float(radius):
                        radius_match = True
                if radius is None:
                    if place_match:
                        found = True; break
                else:
                    if radius_mode == 'or':
                        if place_match or radius_match:
                            found = True; break
                    elif radius_mode == 'only':
                        if radius_match:
                            found = True; break
                    elif radius_mode == 'and':
                        if place_match and radius_match:
                            found = True; break
                    else:
                        raise ValueError("radius_mode must be one of {'or','only','and'}")
            if found:
                correct += 1
        recalls[k] = correct / max(1, Q)
    return recalls

# ----------------------------
# training loop
# ----------------------------
def train_baseline(train_root,
                   test_root=None,
                   epochs=10,
                   batch_size=8,
                   lr=1e-3,
                   margin=0.3,
                   pos_threshold=2.0,
                   test_ratio=0.2,
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   refresh_every=100,
                   negs_num_per_query=10,
                   mining_strategy='random',
                   debug_batch=False,
                   debug_print_n=6):
    print("Device:", device)
    if test_root is None:
        test_root = train_root

    # 0) prepare datasets
    base_train = GraphFilesDataset(train_root, pos_threshold=pos_threshold)
    base_test = GraphFilesDataset(test_root, pos_threshold=pos_threshold)

    # for training we will use all train samples (no split)
    train_idxs = list(range(len(base_train.files)))
    # build gallery/query from test folder
    test_idxs = list(range(len(base_test.files)))
    gallery_idxs, query_idxs = build_gallery_query(base_test, test_idxs, seed=123)

    train_subset = SubsetGraphDataset(base_train, train_idxs, norms_path=os.path.join(train_root, "norms.json"))
    if len(train_subset) == 0:
        raise RuntimeError("train_subset is empty — check your dataset and pos_threshold split.")

    sample_data, _ = train_subset[0]
    in_dim = sample_data.x.shape[1]
    print(f"[info] node feature dim (in_dim) = {in_dim}")

    from functools import partial
    collate = partial(triplet_collate_with_padding, feat_dim=in_dim)

    # create model
    model = SimpleGNN(in_dim=in_dim, hidden_dim=128, n_layers=2, proj_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin)

    # initial mining: use train dataset for mining
    from triplets_graph import TripletsGraphDataset, SimpleTripletFromGlobal

    trip_ds = TripletsGraphDataset(base_train, negs_num_per_query=negs_num_per_query, mining=mining_strategy)

    mining_args = {
        'sampled_queries_num': min(500, len(train_idxs)),
        'neg_samples_num': 1000,
        'infer_batch_size': 64,
        'device': device,
        'train_positives_dist_threshold': max(1.0, pos_threshold * 1.5)
    }

    print(f"[mining] Running initial triplet mining (strategy={mining_strategy}) ...")
    model.eval()
    try:
        trip_ds = trip_ds.compute_triplets(mining_args, model)
    except Exception as e:
        print("[mining] compute_triplets failed with error:", e)
        print("[mining] Falling back to random mining strategy.")
        trip_ds = TripletsGraphDataset(base_train, negs_num_per_query=negs_num_per_query, mining='random')
        trip_ds = trip_ds.compute_triplets(mining_args, model)

    if trip_ds.triplets_global_indexes is None or len(trip_ds.triplets_global_indexes) == 0:
        print("[mining] No triplets mined. Trying random mining with larger sampled_queries_num.")
        mining_args['sampled_queries_num'] = min(2000, len(train_idxs))
        trip_ds = TripletsGraphDataset(base_train, negs_num_per_query=negs_num_per_query, mining='random')
        trip_ds = trip_ds.compute_triplets(mining_args, model)

    print("Triplets mined:", None if trip_ds.triplets_global_indexes is None else trip_ds.triplets_global_indexes.shape)
    if trip_ds.triplets_global_indexes is not None and len(trip_ds.triplets_global_indexes) > 0:
        print("example triplet:", trip_ds.triplets_global_indexes[0])

    if trip_ds.triplets_global_indexes is None or len(trip_ds.triplets_global_indexes) == 0:
        raise RuntimeError("No triplets mined. Try using mining_strategy='random' or increase sampled_queries_num.")

    simple_triplet_ds = SimpleTripletFromGlobal(base_train, trip_ds.triplets_global_indexes)
    train_loader = DataLoader(simple_triplet_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4)

    # sanity batch
    try:
        it = iter(train_loader)
        batch_a, batch_p, batch_n = next(it)
        assert batch_a.num_graphs == batch_p.num_graphs == batch_n.num_graphs
        print("Batch graphs (sanity):", batch_a.num_graphs)
    except Exception as e:
        print("Warning: cannot fetch a batch from train_loader for sanity check:", e)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        iters = 0
        for batch_a, batch_p, batch_n in train_loader:
            # --- DEBUG: print filenames & pose distances before moving to device ---
            if debug_batch:
                # batch_a/.. are Batch objects; filename was preserved as a list attribute
                a_fnames = getattr(batch_a, 'filename', None)
                p_fnames = getattr(batch_p, 'filename', None)
                n_fnames = getattr(batch_n, 'filename', None)
                # orig_idx aggregated into tensor (per-graph)
                a_idxs = getattr(batch_a, 'orig_idx', None)
                p_idxs = getattr(batch_p, 'orig_idx', None)
                n_idxs = getattr(batch_n, 'orig_idx', None)
                # print up to debug_print_n triplets
                to_print = min(debug_print_n, batch_a.num_graphs)
                print("----- Batch triplets preview -----")
                for ii in range(to_print):
                    a_name = a_fnames[ii] if a_fnames is not None else "?"
                    p_name = p_fnames[ii] if p_fnames is not None else "?"
                    n_name = n_fnames[ii] if n_fnames is not None else "?"
                    # compute pose distances if original indices available
                    a_pose = None; p_pose = None; n_pose = None
                    if a_idxs is not None:
                        a_pose = base_train.poses[int(a_idxs[ii].item())]
                    if p_idxs is not None:
                        p_pose = base_train.poses[int(p_idxs[ii].item())]
                    if n_idxs is not None:
                        n_pose = base_train.poses[int(n_idxs[ii].item())]
                    ap_dist = (np.linalg.norm(a_pose - p_pose) if (a_pose is not None and p_pose is not None) else None)
                    an_dist = (np.linalg.norm(a_pose - n_pose) if (a_pose is not None and n_pose is not None) else None)
                    print(f"A[{ii}] = {a_name} | P = {p_name} | N = {n_name} | ap_dist={ap_dist:.3f} an_dist={an_dist:.3f}")
                print("----- end preview -----")

            # move to device and forward
            batch_a = batch_a.to(device); batch_p = batch_p.to(device); batch_n = batch_n.to(device)
            za = model(batch_a); zp = model(batch_p); zn = model(batch_n)
            loss = criterion(za, zp, zn)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item()); iters += 1

        avg_loss = total_loss / max(1, iters)

        # optional: recompute triplets every refresh_every epochs (use current model)
        if refresh_every > 0 and epoch % refresh_every == 0:
            print(f"[mining] Refreshing triplets at epoch {epoch} (strategy={mining_strategy}) ...")
            model.eval()
            try:
                trip_ds = trip_ds.compute_triplets(mining_args, model)
            except Exception as e:
                print("[mining] compute_triplets failed during refresh with error:", e)
                print("[mining] Skipping refresh this epoch.")
                trip_ds = trip_ds
            if trip_ds.triplets_global_indexes is None or len(trip_ds.triplets_global_indexes) == 0:
                print("[mining] Warning: no triplets after refresh — keeping previous triplets")
            else:
                simple_triplet_ds = SimpleTripletFromGlobal(base_train, trip_ds.triplets_global_indexes)
                train_loader = DataLoader(simple_triplet_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4)
                print(f"[mining] New triplets: {trip_ds.triplets_global_indexes.shape}")

        # evaluation on test set
        recalls = compute_recall_at_k(model, base_test, gallery_idxs, query_idxs, device, ks=(1,5,10), radius=3.0, radius_mode='or')
        print(f"Epoch {epoch:02d} | avg_loss={avg_loss:.4f} | Recall@1={recalls[1]:.4f} Recall@5={recalls[5]:.4f} Recall@10={recalls[10]:.4f}")

    torch.save(model.state_dict(), "vpr_baseline_split_model.pt")
    print("Saved vpr_baseline_split_model.pt")
    return model, base_train, base_test, gallery_idxs, query_idxs

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="путь к папке с .pt для обучения")
    parser.add_argument("--test_root", type=str, default=None, help="путь к папке с .pt для теста (если не задан, используется train_root)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--pos", type=float, default=2.0, help="threshold (m) для clustering places")
    parser.add_argument("--test_ratio", type=float, default=0.4, help="(unused) retained for API compatibility")
    parser.add_argument("--refresh_every", type=int, default=100)
    parser.add_argument("--mining_strategy", type=str, default='random')
    parser.add_argument("--debug_batch", action='store_true', help="печать имён файлов в батче и расстояний (debug)")
    parser.add_argument("--debug_print_n", type=int, default=6, help="сколько триплетов печатать из каждого батча")
    args = parser.parse_args()

    train_baseline(args.train_root,
                   test_root=args.test_root,
                   epochs=args.epochs,
                   batch_size=args.batch,
                   lr=args.lr,
                   margin=args.margin,
                   pos_threshold=args.pos,
                   test_ratio=args.test_ratio,
                   refresh_every=args.refresh_every,
                   mining_strategy=args.mining_strategy,
                   debug_batch=args.debug_batch,
                   debug_print_n=args.debug_print_n)