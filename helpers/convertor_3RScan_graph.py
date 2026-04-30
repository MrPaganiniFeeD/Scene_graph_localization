import json
import math
import numpy as np
import torch

EPS = 1e-8

def build_links2idx(path_to_links_type):
    type2idx = {}
    idx = 0
    with open(path_to_links_type, "r", encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n')
            type2idx[line] = idx
            idx += 1
            
    return type2idx


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Файл не найден")
    except json.JSONDecodeError as e:
        print(f"Ошибка в формате JSON: {e}")
    return None


def rotate_cxcywh_clockwise_90(cx, cy, w, h):
    """
    Поворот на 90° по часовой стрелке для нормализованных координат [0, 1]:
      [cx, cy, w, h] -> [1 - cy, cx, h, w]
    """
    return 1.0 - cy, cx, h, w


def cxcywh_to_xyxy(cx, cy, w, h):
    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(1.0, cx + w / 2.0)
    y2 = min(1.0, cy + h / 2.0)
    return [x1, y1, x2, y2]


def iou2d_xyxy(a, b, eps=EPS):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + eps
    return float(inter / union)


def angle_sin_cos(dx, dy):
    theta = math.atan2(dy, dx)
    return float(math.sin(theta)), float(math.cos(theta))


def convert_one(json_path, scan_id, out_path, edge_label2idx, edge_mode="compact"):
    j = load_json(json_path)
    nodes = j.get("nodes", [])
    links = j.get("links", [])

    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    node_cont_feats = []
    node_class_idx = []
    node_meta = []

    rotated_nodes = []

    for n in nodes:
        d = n.get("data", {})
        class_idx = int(d.get("class_id", 0))

        b2 = d.get("bbox_2d", {})
        xyxy = b2.get("xyxy", [0.0, 0.0, 0.0, 0.0])
        x1, y1, x2, y2 = [float(x) for x in xyxy]

        cx, cy = b2.get("center", [0.0, 0.0])
        cx = float(cx)
        cy = float(cy)

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        cx_r, cy_r, w_r, h_r = rotate_cxcywh_clockwise_90(cx, cy, w, h)

        node_cont_feats.append([cx_r, cy_r, w_r, h_r])
        node_class_idx.append(class_idx)

        rotated_xyxy = cxcywh_to_xyxy(cx_r, cy_r, w_r, h_r)

        node_meta.append({
            "id": n["id"],
            "class_name": d.get("class_name", "unknown"),
            "xyxy_raw": [x1, y1, x2, y2],
            "xyxy_rot": rotated_xyxy,
            "center_rot": [cx_r, cy_r],
            "wh_rot": [w_r, h_r],
        })

        rotated_nodes.append({
            "class_idx": class_idx,
            "center": [cx_r, cy_r],
            "wh": [w_r, h_r],
            "xyxy": rotated_xyxy,
            "raw_data": d,
        })

    if len(node_cont_feats) > 0:
        node_x = torch.tensor(np.array(node_cont_feats, dtype=np.float32))
    else:
        node_x = torch.empty((0, 4), dtype=torch.float32)

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    edge_src = []
    edge_dst = []
    edge_attr = []
    edge_label_idx = []
    edge_meta = []
    edge_u_class_idx = []
    edge_v_class_idx = []

    for e in links:
        u = e["source"]
        v = e["target"]
        if u not in id2idx or v not in id2idx:
            continue

        ui = id2idx[u]
        vi = id2idx[v]
        edge_src.append(ui)
        edge_dst.append(vi)

        u_node = rotated_nodes[ui]
        v_node = rotated_nodes[vi]

        u_box = u_node["xyxy"]
        v_box = v_node["xyxy"]

        ux1, uy1, ux2, uy2 = u_box
        vx1, vy1, vx2, vy2 = v_box

        ucx, ucy = u_node["center"]
        vcx, vcy = v_node["center"]

        uw, uh = u_node["wh"]
        vw, vh = v_node["wh"]

        dx = float(vcx - ucx)
        dy = float(vcy - ucy)

        rel_dist = float(np.sqrt(dx * dx + dy * dy))
        sin_t, cos_t = angle_sin_cos(dx, dy)

        iou2 = iou2d_xyxy(u_box, v_box)

        ua = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
        va = max(0.0, vx2 - vx1) * max(0.0, vy2 - vy1)

        ix1 = max(ux1, vx1)
        iy1 = max(uy1, vy1)
        ix2 = min(ux2, vx2)
        iy2 = min(uy2, vy2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        overlap_rel_min = inter / (min(ua, va) + 1e-8) if min(ua, va) > 0 else 0.0
        area_ratio = (va / (ua + 1e-8)) if ua > 0 else 0.0

        w_ratio = (vw / (uw + 1e-8)) if uw > 0 else 0.0
        h_ratio = (vh / (uh + 1e-8)) if uh > 0 else 0.0

        log_area_ratio = math.log1p(area_ratio)
        log_w_ratio = math.log1p(w_ratio)
        log_h_ratio = math.log1p(h_ratio)

        center_in_u = float(ux1 <= vcx <= ux2 and uy1 <= vcy <= uy2)
        center_in_v = float(vx1 <= ucx <= vx2 and vy1 <= ucy <= vy2)

        if edge_mode == "compact":
            edge_attr_vec = [
                rel_dist,
                dx,
                dy,
                sin_t,
                cos_t,
                iou2,
                overlap_rel_min,
                log_area_ratio,
                log_w_ratio,
                log_h_ratio,
            ]

        elif edge_mode == "extended":
            dx_norm = dx / (uw + 1e-8)
            dy_norm = dy / (uh + 1e-8)

            edge_attr_vec = [
                rel_dist,
                dx,
                dy,
                dx_norm,
                dy_norm,
                sin_t,
                cos_t,
                iou2,
                overlap_rel_min,
                log_area_ratio,
                log_w_ratio,
                log_h_ratio,
                math.log1p(inter),
                math.log1p(inter / (ua + 1e-8)),
                math.log1p(inter / (va + 1e-8)),
                center_in_u,
                center_in_v,
            ]
        else:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")

        edge_attr.append([float(x) for x in edge_attr_vec])

        label = e.get("label")
        label_idx = int(edge_label2idx.get(label, 0))
        edge_label_idx.append(label_idx)

        edge_meta.append({"u": u, "v": v, "label": label})
        edge_u_class_idx.append(int(u_node["class_idx"]))
        edge_v_class_idx.append(int(v_node["class_idx"]))

    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if len(edge_src) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

    if len(edge_attr) > 0:
        edge_attr_t = torch.tensor(np.array(edge_attr, dtype=np.float32), dtype=torch.float32)
    else:
        feat_dim = 10 if edge_mode == "compact" else 17
        edge_attr_t = torch.empty((0, feat_dim), dtype=torch.float32)

    edge_label_t = (
        torch.tensor(np.array(edge_label_idx, dtype=np.int64), dtype=torch.long)
        if len(edge_label_idx) > 0
        else torch.empty((0,), dtype=torch.long)
    )

    edge_u_cls = (
        torch.tensor(np.array(edge_u_class_idx, dtype=np.int64), dtype=torch.long)
        if len(edge_u_class_idx) > 0
        else torch.empty((0,), dtype=torch.long)
    )

    edge_v_cls = (
        torch.tensor(np.array(edge_v_class_idx, dtype=np.int64), dtype=torch.long)
        if len(edge_v_class_idx) > 0
        else torch.empty((0,), dtype=torch.long)
    )

    data = {
        "x": node_x,  # уже повёрнутые [cx, cy, w, h]
        "edge_index": edge_index,
        "edge_attr": edge_attr_t,
        "node_class": torch.tensor(node_class_idx, dtype=torch.long),
        "edge_label": edge_label_t,
        "edge_u_class": edge_u_cls,
        "edge_v_class": edge_v_cls,
        "node_meta": node_meta,
        "edge_meta": edge_meta,
        "edge_label2idx": edge_label2idx,
        "json_path": json_path,
        "scan_id": scan_id,
        "graph_rotated": True,
    }

    torch.save(data, out_path)
    return data

import os
import torch
import glob
import numpy as np

root_dir = '/workspace/tmp/dataset/3RScan/SceneGraphs_real_classes/'
out_dir = '/workspace/tmp/dataset/3RScan/SceneGraphs_real_classes_pt_rot_extended/'

all_scenes_path = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
all_count_scens = len(all_scenes_path)
all_scenes_path.sort()

type_links_path = "/workspace/tmp/dataset/3RScan/files/relationships.txt"
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
        convert_one(graph, scan_id, final_path2pt, edge_label2idx, edge_mode="compact")
