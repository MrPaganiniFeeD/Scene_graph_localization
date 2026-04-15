import json

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print("Файл не найден")
    except json.JSONDecodeError as e:
        print(f"Ошибка в формате JSON: {e}")


def build_links2idx(path_to_links_type):
    type2idx = {}
    idx = 0
    with open(path_to_links_type, "r", encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n')
            type2idx[line] = idx
            idx += 1
            
    return type2idx


import math
import numpy as np

EPS = 1e-8

def iou2d_xyxy(a, b, eps=EPS):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    max_x_left, max_y_left = max(ax1, bx1), max(ay1, by1)
    min_x_right, min_y_right = min(ax2, bx2), min(ay2, by2)
    iw = max(0, min_x_right - max_x_left)
    ih = max(0, min_y_right - max_y_left)
    inter = iw * ih
    EPS = 1e-8

def iou2d_xyxy(a, b, eps=EPS):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + eps
    return float(inter / union)

def aabb_iou_3d(min_a, max_a, min_b, max_b, eps=EPS):
    min_a = np.array(min_a, dtype=float); max_a = np.array(max_a, dtype=float)
    min_b = np.array(min_b, dtype=float); max_b = np.array(max_b, dtype=float)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
    union = vol_a + vol_b - inter_vol + eps
    return float(inter_vol / union)


def angle_sin_cos(dx, dy):
    theta = math.atan2(dy, dx)
    return float(math.sin(theta)), float(math.cos(theta))


def direction_bin(dx, dy, n_bins=8):
    theta = math.atan2(dy, dx)
    t = theta if theta >= 0 else (theta + 2*math.pi)
    bin_idx = int(math.floor(t/(2*math.pi) * n_bins)) % n_bins
    onehot = [0] * n_bins 
    onehot[bin_idx] = 1
    return onehot, bin_idx

def convert_one(json_path, scan_id, out_path, edge_label2idx):
    j = load_json(json_path)
    nodes = j.get('nodes', [])
    links = j.get('links', [])
    node_ids = [n['id'] for n in nodes]
    id2idx = {nid:i for i, nid in enumerate(node_ids)}
    N = len(nodes)

    node_cont_feats = []   # непрерывные признаки (без class_idx)
    node_class_idx = []    # только индекс класса
    node_meta = []
    node_x = torch.tensor(np.array(node_cont_feats, dtype=np.float32))
    for n in nodes:
        d = n.get('data', {})
        cname = d.get('class_name', 'unknown')
        class_idx = float(d.get('class_id', 0))
        b2 = d.get('bbox_2d', {})
        xyxy = b2.get('xyxy', [0, 0, 0, 0])
        x1,y1,x2,y2 = xyxy
        cxcy = b2.get('center', [0, 0])
        cx, cy = cxcy
        w = max(0.0, x2-x1)
        h = max(0.0, y2-y1)
        vec = [cx, cy, w, h]
        node_cont_feats.append(vec)
        node_meta.append({'id': n['id'], 'class_name': d.get('class_name','unknown'), 'xyxy':xyxy, 'center2':[cx,cy]})
        node_class_idx.append(int(class_idx))   # целочисленный индекс
    # теперь node_x формируем после цикла
    if len(node_cont_feats) > 0:
        node_x = torch.tensor(np.array(node_cont_feats, dtype=np.float32))
    else:
        feat_dim = len(node_cont_feats[0]) if len(node_cont_feats) > 0 else 4
        node_x = torch.empty((0, feat_dim), dtype=torch.float32)


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
        u_node = nodes[ui]['data']; v_node = nodes[vi]['data']
        u_box2 = [float(x) for x in u_node.get('bbox_2d', {}).get('xyxy', [0,0,0,0])]
        v_box2 = [float(x) for x in v_node.get('bbox_2d', {}).get('xyxy', [0,0,0,0])]
        u_cidx = float(u_node.get('class_id', 0))
        v_cidx = float(v_node.get('class_id', 0))
        u_center2 = u_node.get('bbox_2d', {}).get('center', [0.0,0.0]); v_center2 = v_node.get('bbox_2d', {}).get('center', [0.0,0.0])
        dx2 = float(v_center2[0]) - float(u_center2[0]); dy2 = float(v_center2[1]) - float(u_center2[1])
        rel_dist2 = float(np.sqrt(dx2*dx2 + dy2*dy2))
        sin_t, cos_t = angle_sin_cos(dx2, dy2)
        iou2 = iou2d_xyxy(u_box2, v_box2)
        ux1,uy1,ux2,uy2 = u_box2; vx1,vy1,vx2,vy2 = v_box2
        ua = max(0.0, ux2-ux1) * max(0.0, uy2-uy1)
        va = max(0.0, vx2-vx1) * max(0.0, vy2-vy1)
        ix1 = max(ux1, vx1); iy1 = max(uy1, vy1); ix2 = min(ux2, vx2); iy2 = min(uy2, vy2)
        iw = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1); inter = iw*ih
        overlap_rel_min = inter / (min(ua,va) + 1e-8) if min(ua,va) > 0 else 0.0
        area_ratio = (va / (ua + 1e-8)) if ua > 0 else 0.0
        # same_track
        label = e.get('label'); label_idx = float(edge_label2idx.get(label, 0))

        edge_attr_vec = [rel_dist2, sin_t, cos_t, iou2, overlap_rel_min, area_ratio, label_idx]
        edge_attr.append([float(x) for x in edge_attr_vec])
        edge_label_idx.append(int(label_idx))
        edge_meta.append({'u':u,'v':v,'label':label})
        edge_u_class_idx.append(int(u_cidx))
        edge_v_class_idx.append(int(v_cidx))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if len(edge_src)>0 else torch.empty((2,0),dtype=torch.long)
    edge_attr_t = torch.tensor(np.array(edge_attr, dtype=np.float32)) if len(edge_attr)>0 else torch.empty((0,7),dtype=torch.float32)
    edge_label_t = torch.tensor(np.array(edge_label_idx, dtype=np.int64)) if len(edge_label_idx)>0 else torch.empty((0,),dtype=torch.int64)
    edge_u_cls = torch.tensor(np.array(edge_u_class_idx, dtype=np.int64))
    edge_v_cls = torch.tensor(np.array(edge_v_class_idx, dtype=np.int64))

    data = {
        'x':node_x, # [N, 4] N - количество объектов, x-  геометрическая характеристика [cx, cy, w, h]
        'edge_index':edge_index, # [2, V] V - количетсво связей
        'edge_attr':edge_attr_t, # list(rel_dist2, sin_t, cos_t, iou2, overlap_rel_min, area_ratio, label_idx) geomtric property [V, 7]
        'node_class':torch.tensor(node_class_idx, dtype=torch.long), # [N, int] class object
        'edge_label':edge_label_t, # индекс связи
        'edge_u_class':edge_u_cls, # Для отладки
        'edge_v_class':edge_v_cls, # Для отладки
        'node_meta':node_meta, # Для отладки
        'edge_meta':edge_meta, # Для отладки
        'edge_label2idx':edge_label2idx, # Для отладки
        'json_path':json_path, # Для отладки
        'scan_id':scan_id # Для отладки
    }

    # Сохраняем в файл
    torch.save(data, out_path)
    return data

import os
import torch
import glob
import numpy as np

root_dir = '/mnt/external_usb_hdd/6YL/Datasets/3RScan/SceneGraphs_real_classes/'
out_dir = '/mnt/external_usb_hdd/6YL/Datasets/3RScan/SceneGraphs_real_classes_pt/'

all_scenes_path = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
all_count_scens = len(all_scenes_path)
all_scenes_path.sort()

type_links_path = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/relationships.txt"
edge_label2idx = build_links2idx(type_links_path)
print(edge_label2idx)
minim = 10**10
summa = 0
# train
for f in all_scenes_path:
    scan_id = os.path.basename(f)
    all_graphs = glob.glob(os.path.join(f, '*.json'))
    all_graphs.sort()
    for graph in all_graphs:
        final_path2pt = os.path.join(out_dir, scan_id, os.path.basename(graph).split('.')[0] + '.pt')
        os.makedirs(os.path.join(out_dir, scan_id), exist_ok=True)
        convert_one(graph, scan_id, final_path2pt, edge_label2idx)
