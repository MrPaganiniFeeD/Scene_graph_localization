import os, json, glob, argparse
import numpy as np
import torch
from features import iou2d_xyxy, aabb_iou_3d, angle_sin_cos
import pandas as pd
import re
import math


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter

#root_dir = '/home/amelekhin96/pinkin_ek/data/sber'                     # корень с папками map1, map2, ...
#output_dir = '/home/amelekhin96/pinkin_ek/data/graph/test' # общая папка для результатов

#os.makedirs(output_dir, exist_ok=True)


T_2_1 = np.array([
    [ 0.956711209165,  0.290813077207, -0.011463698546, -0.153455820707],
    [-0.290812256493,  0.956778490454,  0.001775296591,  0.688947242214],
    [ 0.011484499655,  0.001635337894,  0.999932713705, -0.077340220372],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_3_1 = np.array([
    [ 0.994962719425, -0.100235832042, -0.001401759667,  0.514291026771],
    [ 0.100237186633,  0.994963134041,  0.000931834714,  0.461334808929],
    [ 0.001301295964, -0.001067649246,  0.999998583376, -0.130197717405],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_4_1 = np.array([
    [ 0.802577217171, -0.595935887259, -0.027022745116,  0.386908564150],
    [ 0.595360363748,  0.803014107814, -0.026727886735,  0.886000197368],
    [ 0.037627752456,  0.005362921595,  0.999277434608, -0.097609346688],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_5_1 = np.array([
    [-0.314015910006, -0.949409051338, -0.004057277569,  18.696274552193],
    [ 0.949408164245, -0.313990708099, -0.005828627066,  11.209595424855],
    [ 0.004259803837, -0.005682294081,  0.999974782485, -0.462489823486],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_6_1 = np.array([
    [-0.666876352700, -0.745157342238,  0.004057772695,  18.889920284467],
    [ 0.745168270604, -0.666869447050,  0.003064159890,  11.280045723329],
    [ 0.000422723393,  0.005067139233,  0.999987072619, -0.427276360094],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_7_1 = np.array([
    [-0.608371581503, -0.793649920732, -0.001955029598,  18.883837684948],
    [ 0.793644427264, -0.608374670851,  0.002963602522,  11.397353413000],
    [-0.003541453394,  0.000251373208,  0.999993697440, -0.461398363017],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_8_1 = np.array([
    [-0.989549287520, -0.144184982476,  0.001702467792,  16.858101263295],
    [ 0.144180168093, -0.989547836670, -0.002675456918, -11.745636334855],
    [ 0.002070434030, -0.002402034395,  0.999994971754, -0.467016966506],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)

Ts = [T_2_1, T_3_1, T_4_1, T_5_1, T_6_1, T_7_1, T_8_1]


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_label_maps(root_dir):
    map_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'^map\d+$', d)]
    edge_labels = []
    print(map_dirs)
    for map_dir in map_dirs:
        graphs_dir = os.path.join(root_dir, map_dir, 'graphs')
        print(graphs_dir)
        json_files = glob.glob(os.path.join(graphs_dir, '*.json'))
        json_files.sort()  # для порядка
        for p in json_files:
            j = load_json(p)
            for e in j.get('links', []):
                edge_labels.append(e.get('data', {}).get('label', 'unknown'))
        edge_labels = sorted(set(edge_labels))
        edge_label2idx = {c:i for i,c in enumerate(edge_labels)}
    return edge_label2idx


def get_position(df, frame_id, map_id):
    px = df.loc[frame_id, 'px']
    py = df.loc[frame_id, 'py']
    pz = df.loc[frame_id, 'pz']
    
    if map_id == 1:
        return px, py, pz
    
    idx = map_id - 2
    if idx < 0 or idx >= len(Ts):
        raise ValueError(f"map_id {map_id} не поддерживается. Допустимые значения: 1..{len(Ts)+1}")
    
    T = Ts[idx]
    
    point = np.array([px, py, pz, 1.0], dtype=np.float32)
    
    point_transformed = T @ point  # или np.dot(T, point)
    
    return point_transformed[0], point_transformed[1], point_transformed[2]



def convert_one(json_path, poses, out_path, edge_label2idx):
    j = load_json(json_path)
    nodes = j.get('nodes', [])
    links = j.get('links', [])
    node_ids = [n['id'] for n in nodes]
    id2idx = {nid:i for i, nid in enumerate(node_ids)}
    N = len(nodes)

    # 16 dims
    node_cont_feats = []   # непрерывные признаки (без class_idx)
    node_class_idx = []    # только индекс класса
    node_meta = []
    node_x = torch.tensor(np.array(node_cont_feats, dtype=np.float32))
    for n in nodes:
        d = n.get('data', {})
        cname = d.get('class_name', 'unknown')
        class_idx = float(d.get('class_id', -1))
        b2 = d.get('bbox_2d', {})
        xyxy = b2.get('xyxy', [0, 0, 0, 0])
        x1,y1,x2,y2 = xyxy
        cxcy = b2.get('center', [0, 0])
        cx, cy = cxcy
        w = max(0.0, x2-x1)
        h = max(0.0, y2-y1)
        conf_raw = b2.get('conf', None)
        if isinstance(conf_raw, list) and len(conf_raw)>0:
            conf = float(conf_raw[0])
        elif isinstance(conf_raw, (int,float)):
            conf = float(conf_raw)
        else:
            conf = 0.0
        b3 = d.get('bbox_3d', {}).get('obb', {})
        center3 = b3.get('center', [0.0,0.0,0.0])
        extent = b3.get('extent', [0.0,0.0,0.0])
        center3 = [float(x) for x in center3]
        extent = [float(x) for x in extent]
        visible = 1.0 if d.get('visible_current_frame', False) else 0.0
        obs = float(d.get('observation_count', 0))
        vec = [cx, cy, w, h, conf,
               center3[0], center3[1], center3[2],
               extent[0], extent[1], extent[2],
               visible, obs]
        node_cont_feats.append(vec)
        node_meta.append({'id': n['id'], 'class_name': d.get('class_name','unknown'), 'xyxy':xyxy, 'center2':[cx,cy]})
        node_class_idx.append(int(class_idx))   # целочисленный индекс
    # теперь node_x формируем после цикла
    if len(node_cont_feats) > 0:
        node_x = torch.tensor(np.array(node_cont_feats, dtype=np.float32))
    else:
        # fallback: пустой тензор с ожидаемым числом признаков
        # если хочешь другой размер — измени feat_dim
        feat_dim = len(node_cont_feats[0]) if len(node_cont_feats) > 0 else 13
        node_x = torch.empty((0, feat_dim), dtype=torch.float32)

    # global centers for normalization
    if len(node_meta) >= 2:
        centers2 = np.array([nm['center2'] for nm in node_meta], dtype=float)
        max_dist2 = float(np.nanmax(np.linalg.norm(centers2[:,None,:] - centers2[None,:,:], axis=-1)))
        max_dist2 = max(max_dist2, 1e-6)
    else:
        max_dist2 = 1.0

    # global centers for normalizationтщву
    centers2 = np.array([nm['center2'] for nm in node_meta], dtype=float)
    max_dist2 = float(np.nanmax(np.linalg.norm(centers2[:,None,:] - centers2[None,:,:], axis=-1))) if N>=2 else 1.0
    max_dist2 = max(max_dist2, 1e-6)

    # edges
    edge_src = []; edge_dst = []; edge_attr = []; edge_label_idx = []; edge_meta = []
    edge_u_class_idx = []
    edge_v_class_idx = []
    for e in links:
        u = e['source']; v = e['target']
        if u not in id2idx or v not in id2idx:
            continue
        ui = id2idx[u]; vi = id2idx[v]
        edge_src.append(ui); edge_dst.append(vi)
        ed = e.get('data', {})
        u_node = nodes[ui]['data']; v_node = nodes[vi]['data']
        u_box2 = [float(x) for x in u_node.get('bbox_2d', {}).get('xyxy', [0,0,0,0])]
        v_box2 = [float(x) for x in v_node.get('bbox_2d', {}).get('xyxy', [0,0,0,0])]
        u_cidx = float(u_node.get('class_id', -1))
        v_cidx = float(v_node.get('class_id', -1))
        u_center2 = u_node.get('bbox_2d', {}).get('center', [0.0,0.0]); v_center2 = v_node.get('bbox_2d', {}).get('center', [0.0,0.0])
        dx2 = float(v_center2[0]) - float(u_center2[0]); dy2 = float(v_center2[1]) - float(u_center2[1])
        rel_dist2 = float(np.sqrt(dx2*dx2 + dy2*dy2))
        rel_dist2_norm = rel_dist2 / max_dist2
        sin_t, cos_t = angle_sin_cos(dx2, dy2)
        iou2 = iou2d_xyxy(u_box2, v_box2)
        ux1,uy1,ux2,uy2 = u_box2; vx1,vy1,vx2,vy2 = v_box2
        ua = max(0.0, ux2-ux1) * max(0.0, uy2-uy1)
        va = max(0.0, vx2-vx1) * max(0.0, vy2-vy1)
        ix1 = max(ux1, vx1); iy1 = max(uy1, vy1); ix2 = min(ux2, vx2); iy2 = min(uy2, vy2)
        iw = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1); inter = iw*ih
        overlap_rel_min = inter / (min(ua,va) + 1e-8) if min(ua,va) > 0 else 0.0
        area_ratio = (va / (ua + 1e-8)) if ua > 0 else 0.0
        # 3d
        u_aabb = u_node.get('bbox_3d', {}).get('aabb', {}); v_aabb = v_node.get('bbox_3d', {}).get('aabb', {})
        u_aabb_min = u_aabb.get('min'); u_aabb_max = u_aabb.get('max'); v_aabb_min = v_aabb.get('min'); v_aabb_max = v_aabb.get('max')
        if None not in (u_aabb_min, u_aabb_max, v_aabb_min, v_aabb_max):
            iou3 = aabb_iou_3d(u_aabb_min, u_aabb_max, v_aabb_min, v_aabb_max)
        else:
            iou3 = 0.0
        # same_track
        same_track = 1.0 if (u_node.get('track_id') is not None and v_node.get('track_id') is not None and int(u_node.get('track_id')) == int(v_node.get('track_id'))) else 0.0
        label = ed.get('label', 'unknown'); label_idx = float(edge_label2idx.get(label, -1))
        edge_attr_vec = [rel_dist2_norm, sin_t, cos_t, iou2, overlap_rel_min, area_ratio, iou3, same_track, label_idx]
        edge_attr.append([float(x) for x in edge_attr_vec])
        edge_label_idx.append(int(label_idx))
        edge_meta.append({'u':u,'v':v,'label':label,'label_class':ed.get('label_class','unknown')})
        edge_u_class_idx.append(int(u_cidx))
        edge_v_class_idx.append(int(v_cidx))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if len(edge_src)>0 else torch.empty((2,0),dtype=torch.long)
    edge_attr_t = torch.tensor(np.array(edge_attr, dtype=np.float32)) if len(edge_attr)>0 else torch.empty((0,9),dtype=torch.float32)
    edge_label_t = torch.tensor(np.array(edge_label_idx, dtype=np.int64)) if len(edge_label_idx)>0 else torch.empty((0,),dtype=torch.int64)
    edge_u_cls = torch.tensor(np.array(edge_u_class_idx, dtype=np.int64))
    edge_v_cls = torch.tensor(np.array(edge_v_class_idx, dtype=np.int64))

    out = {
        'x': node_x,
        'node_class': torch.tensor(node_class_idx, dtype=torch.long),
        'edge_index': edge_index,
        'edge_attr': edge_attr_t,
        'edge_label': edge_label_t,
        'edge_u_class': edge_u_cls,
        'edge_v_class': edge_v_cls,
        'node_meta': node_meta,
        'edge_meta': edge_meta,
        'edge_label2idx': edge_label2idx,
        'json_path': json_path,
        'pose': torch.tensor(poses, dtype=torch.float32),  # (3,)
        'scene_name': j.get('graph', {}).get('scene_name', None)
    }
    torch.save(out, out_path)
    return out

#!/usr/bin/env python3
"""
split_and_convert.py

Scan maps in root_dir, convert JSON graph files to .pt using convert_one(), and split into
`database` and `queries` according to chosen strategy.

Strategies:
  - simple: every k-th frame is a candidate query, then filter by spatial min_sep (median*factor)
  - cluster: cluster DB into M clusters and pick one query per cluster (closest to center)
  - kcenter: choose queries by k-center greedy on precomputed query features (requires --queries_feats)

Saves:
  - converted .pt under output_dir/{database,queries}
  - split metadata JSON: output_dir/splits.json with lists and parameters

Usage examples:
  python split_and_convert.py --root_dir /data/sber --output_dir /out/graph --strategy simple
  python split_and_convert.py --strategy cluster --k_clusters 150
  python split_and_convert.py --strategy kcenter --kcenter_k 150 --queries_feats /path/q_feats.npy

"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/home/amelekhin96/pinkin_ek/data/sber')
    p.add_argument('--output_dir', type=str, default='/home/amelekhin96/pinkin_ek/data/graph/test')
    p.add_argument('--map', type=str, default=None, help='Specific map directory name, e.g. map8. If omitted process last map by default')
    p.add_argument('--strategy', type=str, choices=['simple', 'cluster', 'kcenter'], default='simple')
    p.add_argument('--k', type=int, default=3, help='for simple: every k-th is candidate')
    p.add_argument('--min_sep_factor', type=float, default=2.0, help='factor * median_nearest_dist to decide min_sep')
    p.add_argument('--k_clusters', type=int, default=150, help='for cluster strategy: number of clusters')
    p.add_argument('--kcenter_k', type=int, default=150, help='for kcenter strategy: number of queries to select')
    p.add_argument('--queries_feats', type=str, default=None, help='path to numpy .npy with queries features (Q x D) required for kcenter')
    p.add_argument('--save_splits_json', action='store_true', help='save splits metadata to output_dir/splits.json')
    p.add_argument('--force_overwrite', action='store_true', help='overwrite existing .pt files')
    return p.parse_args()


def collect_json_files(graphs_dir):
    json_files = glob.glob(os.path.join(graphs_dir, '*.json'))
    json_files.sort()
    return json_files


def gather_all_entries(json_files, map_num, poses_df):
    all_entries = []
    for idx, json_path in enumerate(json_files):
        base = os.path.basename(json_path)
        file_stem = base.split('.')[0]
        out_filename = f"map{map_num}_{file_stem}.pt"
        try:
            frame_id = int(file_stem)
            poses = get_position(poses_df, frame_id, map_num)
        except Exception as e:
            print(f"[WARN] cannot get pose for {json_path}: {e}")
            poses = None
        all_entries.append({
            'idx': idx,
            'json_path': json_path,
            'base': base,
            'file_stem': file_stem,
            'out_filename': out_filename,
            'pose': poses
        })
    return all_entries


def simple_selection(all_entries, k=3, min_sep_factor=2.0, output_dir=None, poses_df=None):
    candidate_query_idxs = [e['idx'] for e in all_entries if e['idx'] % k == 0]
    candidate_db_idxs = [e['idx'] for e in all_entries if e['idx'] % k != 0]

    db_with_pos = [(i, float(all_entries[i]['pose'][0]), float(all_entries[i]['pose'][1]))
                   for i in candidate_db_idxs if all_entries[i]['pose'] is not None]
    q_with_pos = [(i, float(all_entries[i]['pose'][0]), float(all_entries[i]['pose'][1]))
                  for i in candidate_query_idxs if all_entries[i]['pose'] is not None]

    if len(db_with_pos) == 0 or len(q_with_pos) == 0:
        print('[WARN] Not enough poses; falling back to naive split')
        return candidate_query_idxs, candidate_db_idxs, {'min_sep': None, 'median_dist': None}

    db_xy = np.array([[p[1], p[2]] for p in db_with_pos])
    q_xy = np.array([[p[1], p[2]] for p in q_with_pos])
    db_idx_map = [p[0] for p in db_with_pos]
    q_idx_map = [p[0] for p in q_with_pos]

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(db_xy)
    dists, _ = nn.kneighbors(q_xy, return_distance=True)
    dists = dists.ravel()
    median_dist = float(np.median(dists))
    min_sep = median_dist * min_sep_factor

    keep_query_global_idxs = []
    move_to_db_global_idxs = []
    q_map = {gidx: dist for gidx, dist in zip(q_idx_map, dists)}
    for gidx in candidate_query_idxs:
        if gidx in q_map:
            if q_map[gidx] >= min_sep:
                keep_query_global_idxs.append(gidx)
            else:
                move_to_db_global_idxs.append(gidx)
        else:
            move_to_db_global_idxs.append(gidx)

    final_db_set = set(candidate_db_idxs) | set(move_to_db_global_idxs)
    final_query_set = set(keep_query_global_idxs)
    final_db_idxs = sorted(list(final_db_set - final_query_set))
    final_query_idxs = sorted(list(final_query_set))

    # auto relax if too few queries
    min_queries_desired = max(50, int(0.15 * (len(all_entries))))
    if len(final_query_idxs) < min_queries_desired:
        print(f"[WARN] too few queries ({len(final_query_idxs)}) after filtering; relaxing min_sep")
        min_sep_relaxed = min_sep * 0.5
        keep_query_global_idxs = [gidx for gidx in candidate_query_idxs if (gidx in q_map and q_map[gidx] >= min_sep_relaxed)]
        final_query_idxs = sorted(keep_query_global_idxs)
        final_db_idxs = sorted(list(set(range(len(all_entries))) - set(final_query_idxs)))

    params = {'median_dist': median_dist, 'min_sep': min_sep, 'min_sep_factor': min_sep_factor}
    return final_query_idxs, final_db_idxs, params


def cluster_selection(all_entries, k_clusters=150):
    # cluster on database XY, then for each cluster choose closest query (if any)
    candidate_query_idxs = [e['idx'] for e in all_entries if e['idx'] % 3 == 0]
    candidate_db_idxs = [e['idx'] for e in all_entries if e['idx'] % 3 != 0]

    db_with_pos = [(i, float(all_entries[i]['pose'][0]), float(all_entries[i]['pose'][1]))
                   for i in candidate_db_idxs if all_entries[i]['pose'] is not None]
    q_with_pos = [(i, float(all_entries[i]['pose'][0]), float(all_entries[i]['pose'][1]))
                  for i in candidate_query_idxs if all_entries[i]['pose'] is not None]

    if len(db_with_pos) == 0:
        print('[WARN] No pos for DB - fallback to simple selection')
        return simple_selection(all_entries)

    db_xy = np.array([[p[1], p[2]] for p in db_with_pos])
    q_xy = np.array([[p[1], p[2]] for p in q_with_pos]) if len(q_with_pos) > 0 else np.zeros((0,2))
    db_idx_map = [p[0] for p in db_with_pos]
    q_idx_map = [p[0] for p in q_with_pos]

    # cluster db
    k = min(k_clusters, len(db_xy))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(db_xy)
    centers = kmeans.cluster_centers_

    final_query_idxs = set()
    if q_xy.shape[0] > 0:
        closest_idx_to_center, _ = pairwise_distances_argmin_min(centers, q_xy)
        for qi in closest_idx_to_center:
            final_query_idxs.add(q_idx_map[int(qi)])

    final_db_idxs = sorted(list(set(range(len(all_entries))) - final_query_idxs))
    params = {'k_clusters': k}
    return sorted(list(final_query_idxs)), final_db_idxs, params


def kcenter_selection(all_entries, queries_feats_path, kcenter_k=150):
    # requires numpy .npy file with queries features in the order of candidate queries
    candidate_query_idxs = [e['idx'] for e in all_entries if e['idx'] % 3 == 0]
    if not os.path.isfile(queries_feats_path):
        raise FileNotFoundError(f"queries_feats not found: {queries_feats_path}")
    feats = np.load(queries_feats_path)  # shape [Qcand, D]
    if feats.shape[0] != len(candidate_query_idxs):
        print(f"[WARN] queries_feats rows ({feats.shape[0]}) != candidate_query count ({len(candidate_query_idxs)})")

    # k-center greedy on candidate queries features
    N = feats.shape[0]
    k = min(kcenter_k, N)
    centers = [0]
    dist = np.linalg.norm(feats - feats[0:1], axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(dist))
        centers.append(idx)
        newd = np.linalg.norm(feats - feats[idx:idx+1], axis=1)
        dist = np.minimum(dist, newd)

    selected_query_global_idxs = [candidate_query_idxs[i] for i in centers]
    final_query_idxs = sorted(selected_query_global_idxs)
    final_db_idxs = sorted(list(set(range(len(all_entries))) - set(final_query_idxs)))
    params = {'kcenter_k': kcenter_k, 'selected_k': len(final_query_idxs), 'queries_feats_path': queries_feats_path}
    return final_query_idxs, final_db_idxs, params


def save_split_metadata(output_dir, map_dir, final_query_idxs, final_db_idxs, all_entries, params):
    splits = {
        'map_dir': map_dir,
        'num_total': len(all_entries),
        'num_queries': len(final_query_idxs),
        'num_database': len(final_db_idxs),
        'queries': [all_entries[i]['out_filename'] for i in final_query_idxs],
        'database': [all_entries[i]['out_filename'] for i in final_db_idxs],
        'params': params
    }
    out_path = os.path.join(output_dir, 'splits.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(splits, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] splits metadata saved to {out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # build edge labels map once
    edge_label2idx = build_label_maps(args.root_dir)

    # expected explicit mapping (user-provided order)
    # train: db = map1, queries = map2..map7
    # test:  db = map1, queries = map8
    train_db_map = 'map1'
    train_query_maps = [f"map{i}" for i in range(2, 8)]  # map2..map7
    test_db_map = 'map1'
    test_query_maps = ['map8']

    # discover maps present under root_dir
    existing_maps = {d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d)) and re.match(r'^map\d+$', d)}
    print(f"[INFO] maps found under root_dir: {sorted(existing_maps)}")

    # helper to validate maps
    def ensure_map_exists(map_name):
        if map_name not in existing_maps:
            raise RuntimeError(f"Expected map directory '{map_name}' not found under {args.root_dir}")

    # validate required maps exist (give informative error)
    required_maps = set([train_db_map] + train_query_maps + test_query_maps)
    missing = required_maps - existing_maps
    if missing:
        print(f"[WARN] Some expected maps are missing: {sorted(list(missing))}. The script will process only existing ones.")

    # prepare output directories
    train_db_dir = os.path.join(args.output_dir, 'train', 'database')
    train_q_dir = os.path.join(args.output_dir, 'train', 'queries')
    test_db_dir = os.path.join(args.output_dir, 'test', 'database')
    test_q_dir = os.path.join(args.output_dir, 'test', 'queries')
    for d in (train_db_dir, train_q_dir, test_db_dir, test_q_dir):
        os.makedirs(d, exist_ok=True)

    # small helper to process one map: convert all jsons and save as either db or queries or both
    def process_map_full(map_name, save_as_db=False, save_as_queries=False, dest_db_dir=None, dest_q_dir=None):
        if map_name not in existing_maps:
            print(f"[SKIP] map {map_name} not present under root_dir")
            return

        graphs_dir = os.path.join(args.root_dir, map_name, 'graphs')
        poses_path = os.path.join(args.root_dir, map_name, 'poses.csv')
        poses_df = None
        if os.path.isfile(poses_path):
            try:
                poses_df = pd.read_csv(poses_path)
            except Exception as e:
                print(f"[WARN] failed to read poses.csv for {map_name}: {e}; proceeding with default poses")
                poses_df = None
        else:
            print(f"[INFO] no poses.csv for {map_name} -> using default [0,0,0] poses")

        json_files = collect_json_files(graphs_dir)
        if len(json_files) == 0:
            print(f"[WARN] no json graph files found in {graphs_dir}")
            return

        all_entries = gather_all_entries(json_files, int(re.search(r'\d+', map_name).group()), poses_df if poses_df is not None else pd.DataFrame())

        def save_entry_as_pt_local(entry):
            src_json = entry['json_path']
            out_fn = entry['out_filename']
            # compute a stable pose vector (3 floats). If poses_df missing, fallback to zeros.
            try:
                frame_id = int(entry['file_stem'])
                if poses_df is not None and not poses_df.empty:
                    poses = get_position(poses_df, frame_id, int(re.search(r'\d+', map_name).group()))
                else:
                    poses = [0.0, 0.0, 0.0]
            except Exception:
                poses = [0.0, 0.0, 0.0]
            # save to destination(s)
            if save_as_db and dest_db_dir is not None:
                dest_path = os.path.join(dest_db_dir, out_fn)
                if os.path.exists(dest_path) and not args.force_overwrite:
                    print(f"[SKIP] {dest_path} exists (use --force_overwrite to overwrite)")
                else:
                    try:
                        convert_one(src_json, poses, dest_path, edge_label2idx)
                    except Exception as e:
                        print(f"[ERROR] convert_one failed for {src_json} -> {dest_path}: {e}")
            if save_as_queries and dest_q_dir is not None:
                dest_path = os.path.join(dest_q_dir, out_fn)
                if os.path.exists(dest_path) and not args.force_overwrite:
                    print(f"[SKIP] {dest_path} exists (use --force_overwrite to overwrite)")
                else:
                    try:
                        convert_one(src_json, poses, dest_path, edge_label2idx)
                    except Exception as e:
                        print(f"[ERROR] convert_one failed for {src_json} -> {dest_path}: {e}")

        # iterate entries and save
        for entry in all_entries:
            save_entry_as_pt_local(entry)

        print(f"[INFO] processed map {map_name}: saved to DB? {save_as_db} Q? {save_as_queries}")

    # --- Build train split: db = map1, queries = map2..map7 ---
    if train_db_map in existing_maps:
        # process map1 as database for train
        process_map_full(train_db_map, save_as_db=True, save_as_queries=False, dest_db_dir=train_db_dir, dest_q_dir=None)
    else:
        print(f"[WARN] train DB map {train_db_map} not present -> train database will be empty")

    for m in train_query_maps:
        if m in existing_maps:
            process_map_full(m, save_as_db=False, save_as_queries=True, dest_db_dir=None, dest_q_dir=train_q_dir)
        else:
            print(f"[INFO] train query map {m} not present -> skipping")

    # --- Build test split: db = map1, queries = map8 ---
    # reuse map1 as test database as well (convert again or skip if already present)
    if test_db_map in existing_maps:
        process_map_full(test_db_map, save_as_db=True, save_as_queries=False, dest_db_dir=test_db_dir, dest_q_dir=None)
    else:
        print(f"[WARN] test DB map {test_db_map} not present -> test database will be empty")

    for m in test_query_maps:
        if m in existing_maps:
            process_map_full(m, save_as_db=False, save_as_queries=True, dest_db_dir=None, dest_q_dir=test_q_dir)
        else:
            print(f"[INFO] test query map {m} not present -> skipping")

    # final summary
    def summary_counts(dir_path):
        return len(glob.glob(os.path.join(dir_path, '*.pt')))
    print(f"[DONE] train DB: {summary_counts(train_db_dir)}, train Q: {summary_counts(train_q_dir)}, "
          f"test DB: {summary_counts(test_db_dir)}, test Q: {summary_counts(test_q_dir)})")

if __name__ == '__main__':
    main()