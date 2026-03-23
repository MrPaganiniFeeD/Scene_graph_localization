# train_eval_baseline_v2.py
import os
import glob
import random
import math
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

import numpy as np

# ----------------------------
# Model (VPRGraphEncoder, same как в предложенном патче)
# ----------------------------
class VPRGraphEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 n_layers=3,
                 proj_dim=128,
                 num_node_classes=None,
                 node_emb_dim=64,
                 dropout=0.1):
        super().__init__()
        self.use_node_class = (num_node_classes is not None)
        self.node_emb = None
        if self.use_node_class:
            self.node_emb = nn.Embedding(num_node_classes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
        eff_in_dim = in_dim + (node_emb_dim if self.use_node_class else 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(eff_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.pool_out_dim = hidden_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(self.pool_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        self._proj_dim = proj_dim

    def forward(self, batch: Batch):
        x = batch.x
        if self.use_node_class and hasattr(batch, 'node_class') and batch.node_class is not None:
            node_cls = batch.node_class.long().to(x.device)
            emb = self.node_emb(node_cls)
            x = torch.cat([x, emb], dim=1)
        h = self.input_mlp(x)
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = self.act(h)
            h = self.drop(h)
        hg_mean = global_mean_pool(h, batch.batch)
        hg_max  = global_max_pool(h, batch.batch)
        hg = torch.cat([hg_mean, hg_max], dim=1)
        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        return z

    @property
    def out_dim(self):
        return self._proj_dim

# ----------------------------
# Helpers: load .pt -> pyg.Data
# ----------------------------
def dict_to_pyg_data(d):
    # expects d to be loaded from .pt and contain keys: 'x', optionally 'node_class', 'edge_index', 'edge_attr', 'pose'
    x = d.get('x', None)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x is None:
        x = torch.zeros((1, 1), dtype=torch.float32)
    edge_index = d.get('edge_index', None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long) if not torch.is_tensor(edge_index) else edge_index.long()
    edge_attr = d.get('edge_attr', None)
    if edge_attr is not None and not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.array(edge_attr, dtype=float), dtype=torch.float32)
    node_class = d.get('node_class', None)
    if node_class is not None and not torch.is_tensor(node_class):
        node_class = torch.tensor(node_class, dtype=torch.long)
    #pose = d.get('pose', None)
    #if pose is not None and not torch.is_tensor(pose):
    #    pose = torch.tensor(np.array(pose, dtype=float), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_class=node_class)
    return data

# ----------------------------
# SceneGraphDataset: scans root/<scene>/{queries,database}
# ----------------------------
class SceneGraphDataset:
    """
    root: path to train_root (contains scene subfolders).
    Each scene folder must contain 'queries' and 'database' subfolders with .pt files.
    """
    def __init__(self, root):
        self.root = root
        self.scenes = []
        # entries: list of dict {path, data, scene, role}
        self.entries = []
        # maps
        self.scene_to_db_indices = defaultdict(list)
        self.scene_to_query_indices = defaultdict(list)
        self._scan()

    def _scan(self):
        scenes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if len(scenes) == 0:
            raise RuntimeError(f"No scenes found under {self.root}")
        self.scenes = scenes
        idx = 0
        for s in scenes:
            scene_dir = os.path.join(self.root, s)
            db_dir = os.path.join(scene_dir, 'database')
            q_dir  = os.path.join(scene_dir, 'queries')
            db_files = sorted(glob.glob(os.path.join(db_dir, '*.pt'))) if os.path.isdir(db_dir) else []
            q_files  = sorted(glob.glob(os.path.join(q_dir, '*.pt')))  if os.path.isdir(q_dir) else []
            # load db
            for p in db_files:
                d = torch.load(p, weights_only=False)
                self.entries.append({'path': p, 'data': d, 'scene': s, 'role': 'db'})
                self.scene_to_db_indices[s].append(idx)
                idx += 1
            # load queries
            for p in q_files:
                d = torch.load(p, weights_only=False)
                self.entries.append({'path': p, 'data': d, 'scene': s, 'role': 'query'})
                self.scene_to_query_indices[s].append(idx)
                idx += 1
        # build convenience arrays
        self.n = len(self.entries)
        # simple diagnostics
        n_db = sum(len(v) for v in self.scene_to_db_indices.values())
        n_q  = sum(len(v) for v in self.scene_to_query_indices.values())
        print(f"[SceneGraphDataset] root={self.root} scenes={len(self.scenes)} entries={self.n} (db={n_db}, queries={n_q})")

    def get_entry(self, idx):
        return self.entries[idx]

    def get_pyg_data(self, idx):
        return dict_to_pyg_data(self.entries[idx]['data'])

    def all_db_indices(self):
        return [i for i,e in enumerate(self.entries) if e['role']=='db']

    def all_query_indices(self):
        return [i for i,e in enumerate(self.entries) if e['role']=='query']

    def db_indices_by_scene(self, scene):
        return list(self.scene_to_db_indices.get(scene, []))

    def query_indices_by_scene(self, scene):
        return list(self.scene_to_query_indices.get(scene, []))

    def scene_of(self, idx):
        return self.entries[idx]['scene']

# ----------------------------
# Triplet dataset (anchors = queries). Positive from same scene DB randomly, negative from DB of other scenes randomly.
# ----------------------------
class TripletGraphSceneDataset(Dataset):
    def __init__(self, scene_ds: SceneGraphDataset):
        self.scene_ds = scene_ds
        # build list of anchors (indices of all queries that have at least one DB in the same scene)
        anchors = []
        for s in scene_ds.scenes:
            q_idxs = scene_ds.query_indices_by_scene(s)
            db_idxs = scene_ds.db_indices_by_scene(s)
            if len(db_idxs) == 0:
                # no positive can be sampled for this scene -> skip queries
                continue
            for q in q_idxs:
                anchors.append(q)
        if len(anchors) == 0:
            raise RuntimeError("No query anchors with positive DB entries found in dataset.")
        self.anchors = anchors
        self.scenes = scene_ds.scenes

        # precompute per-scene DB lists for negatives
        self.scene_db_lists = {s: scene_ds.db_indices_by_scene(s) for s in self.scenes}
        # scenes that have at least one DB (for negative sampling)
        self.scenes_with_db = [s for s,l in self.scene_db_lists.items() if len(l) > 0]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        # anchor_idx is an index into scene_ds.entries (a query)
        anchor_idx = self.anchors[idx]
        anchor_data = self.scene_ds.get_pyg_data(anchor_idx)
        scene = self.scene_ds.scene_of(anchor_idx)
        # sample positive from same scene DB
        pos_candidates = self.scene_ds.db_indices_by_scene(scene)
        # fallback (shouldn't happen because anchors ensured positive exists)
        if len(pos_candidates) == 0:
            pos_idx = anchor_idx
        else:
            pos_idx = random.choice(pos_candidates)
        pos_data = self.scene_ds.get_pyg_data(pos_idx)
        # sample negative: choose random scene different from anchor's scene that has db entries
        other_scenes = [s for s in self.scenes_with_db if s != scene]
        if len(other_scenes) == 0:
            # fallback to random db from same scene (degenerate case)
            neg_candidates = pos_candidates
        else:
            neg_scene = random.choice(other_scenes)
            neg_candidates = self.scene_ds.db_indices_by_scene(neg_scene)
        neg_idx = random.choice(neg_candidates)
        neg_data = self.scene_ds.get_pyg_data(neg_idx)
        return anchor_data, pos_data, neg_data

# ----------------------------
# ensure non-empty helper and collate (same as before)
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
# compute descriptors for indices (gallery/query) — helper used in evaluation
# ----------------------------
def compute_descriptors_for_indices(model, scene_ds: SceneGraphDataset, indices, device, batch_size=64):
    model.eval()
    descs = []
    loader_items = []
    for idx in indices:
        d = scene_ds.get_pyg_data(idx)
        loader_items.append(d)
    loader = DataLoader(loader_items, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: Batch.from_data_list(batch))
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            z = model(b)
            descs.append(z.cpu())
    if len(descs) == 0:
        return torch.empty((0, model.out_dim)), []
    descs = torch.cat(descs, dim=0)
    return descs, indices

# ----------------------------
# compute recall@k (place equality by scene)
# ----------------------------
def compute_recall_at_k(model, scene_ds: SceneGraphDataset, gallery_idxs, query_idxs, device, ks=(1,5,10)):
    model.eval()
    gallery_descs, _ = compute_descriptors_for_indices(model, scene_ds, gallery_idxs, device)
    query_descs, _   = compute_descriptors_for_indices(model, scene_ds, query_idxs, device)
    if query_descs.shape[0] == 0 or gallery_descs.shape[0] == 0:
        return {k: 0.0 for k in ks}
    dists = torch.cdist(query_descs, gallery_descs, p=2).cpu().numpy()
    sorted_idxs = np.argsort(dists, axis=1)  # shape [Q, G]
    recalls = {}
    # map gallery position -> original index
    gallery_order = list(gallery_idxs)
    for k in ks:
        correct = 0
        for i_q, q_idx in enumerate(query_idxs):
            topk = sorted_idxs[i_q, :k]
            found = False
            q_scene = scene_ds.scene_of(q_idx)
            for gpos in topk:
                g_orig = gallery_order[int(gpos)]
                if scene_ds.scene_of(g_orig) == q_scene:
                    found = True
                    break
            if found:
                correct += 1
        recalls[k] = correct / max(1, len(query_idxs))
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
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   refresh_every=100,
                   debug_batch=False,
                   debug_print_n=6):
    print("Device:", device)
    if test_root is None:
        test_root = train_root

    # prepare datasets
    base_train = SceneGraphDataset(train_root)
    base_test = SceneGraphDataset(test_root)

    # training triplet dataset: anchors = queries
    trip_ds = TripletGraphSceneDataset(base_train)
    # need feature dim for collate: take first DB entry x dim (fallback to first entry)
    sample_entry = None
    for e in base_train.entries:
        if e['role'] == 'db' and e['data'].get('x') is not None:
            sample_entry = e['data']
            break
    if sample_entry is None:
        sample_entry = base_train.entries[0]['data']
    in_dim = sample_entry.get('x').shape[1] if hasattr(sample_entry.get('x'), 'shape') else sample_entry.get('x').size(1)
    print(f"[info] node feature dim (in_dim) = {in_dim}")

    collate = lambda batch: triplet_collate_with_padding(batch, feat_dim=in_dim)

    # model
    model = VPRGraphEncoder(in_dim=in_dim, hidden_dim=256, n_layers=3, num_node_classes=528, proj_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin)

    train_loader = DataLoader(trip_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)

    # sanity
    try:
        it = iter(train_loader)
        batch_a, batch_p, batch_n = next(it)
        print("Sanity batch graphs:", batch_a.num_graphs)
    except Exception as e:
        print("Warning: cannot fetch a batch from train_loader for sanity check:", e)

    # gallery & queries for evaluation (take from test dataset)
    gallery_idxs = base_test.all_db_indices()
    query_idxs = base_test.all_query_indices()
    print(f"[Eval] Gallery {len(gallery_idxs)} queries {len(query_idxs)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        iters = 0
        for batch_a, batch_p, batch_n in train_loader:
            if debug_batch:
                a_names = getattr(batch_a, 'filename', None)
                # printing filenames may not be serialized into Batch; we can print counts
                print(f"[DEBUG] batch graphs: a={batch_a.num_graphs}, p={batch_p.num_graphs}, n={batch_n.num_graphs}")
            batch_a = batch_a.to(device); batch_p = batch_p.to(device); batch_n = batch_n.to(device)
            za = model(batch_a); zp = model(batch_p); zn = model(batch_n)
            loss = criterion(za, zp, zn)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item()); iters += 1
        avg_loss = total_loss / max(1, iters)

        # evaluation
        recalls = compute_recall_at_k(model, base_test, gallery_idxs, query_idxs, device, ks=(1,5,10))
        print(f"Epoch {epoch:02d} | avg_loss={avg_loss:.4f} | Recall@1={recalls[1]:.4f} Recall@5={recalls[5]:.4f} Recall@10={recalls[10]:.4f}")

    torch.save(model.state_dict(), "vpr_scene_model.pt")
    print("Saved vpr_scene_model.pt")
    return model, base_train, base_test

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="path to train folder (contains scene subfolders)")
    parser.add_argument("--test_root", type=str, default=None, help="path to test folder (if omitted, use train_root)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--refresh_every", type=int, default=100)
    parser.add_argument("--debug_batch", action='store_true')
    parser.add_argument("--debug_print_n", type=int, default=6)
    args = parser.parse_args()

    train_baseline(args.train_root,
                   test_root=args.test_root,
                   epochs=args.epochs,
                   batch_size=args.batch,
                   lr=args.lr,
                   margin=args.margin,
                   refresh_every=args.refresh_every,
                   debug_batch=args.debug_batch,
                   debug_print_n=args.debug_print_n)