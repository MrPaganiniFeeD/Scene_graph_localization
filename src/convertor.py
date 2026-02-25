import os, json, glob, argparse
import numpy as np
import torch
from features import iou2d_xyxy, aabb_iou_3d, angle_sin_cos
import pandas as pd
import re

root_dir = r'C:\Users\Egor\VsCode project\Scene_graph_localization\data'                     # корень с папками map1, map2, ...
output_dir = r'C:\Users\Egor\VsCode project\Scene_graph_localization\res'  # общая папка для результатов
os.makedirs(output_dir, exist_ok=True)


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
    node_feats = []
    node_meta = []
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
        vec = [class_idx, cx, cy, w, h, conf,
               center3[0], center3[1], center3[2],
               extent[0], extent[1], extent[2],
               visible, obs]
        node_feats.append(vec)
        node_meta.append({'id': n['id'], 'class_name': d.get('class_name','unknown'), 'xyxy':xyxy, 'center2':[cx,cy]})
        node_x = torch.tensor(np.array(node_feats, dtype=np.float32))

    # global centers for normalization
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

edge_label2idx = build_label_maps(root_dir)

# Получаем все подпапки, имена которых начинаются с "map" и содержат цифры
map_dirs = [d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'^map\d+$', d)]

for map_dir in map_dirs:
    # Извлекаем номер карты (например, "map3" -> 3)
    map_num = re.search(r'\d+', map_dir).group()
    map_num = int(map_num)

    # Путь к папке с JSON-файлами графов
    graphs_dir = os.path.join(root_dir, map_dir, 'graphs')
    if not os.path.isdir(graphs_dir):
        print(f"Предупреждение: {graphs_dir} не существует, пропускаем")
        continue

    # Загружаем poses для текущей карты (предполагается файл poses.csv в папке карты)
    poses_path = os.path.join(root_dir, map_dir, 'poses.csv')
    if os.path.isfile(poses_path):
        poses_df = pd.read_csv(poses_path)
    else:
        print(f"Предупреждение: {poses_path} не найден, передаю пустой массив")
        poses = np.array([], dtype=np.float32)   # или можно выйти с ошибкой

    # Находим все JSON-файлы в папке graphs
    json_files = glob.glob(os.path.join(graphs_dir, '*.json'))
    json_files.sort()  # для порядка

    for json_path in json_files:
        # Извлекаем имя файла без расширения (например, "00001")
        base = os.path.basename(json_path)
        file_stem = os.path.splitext(base)[0]
        print("base",base)

        # Формируем выходное имя: mapX_00001.pt
        out_filename = f"map{map_num}_{file_stem}.pt"
        out_path = os.path.join(output_dir, out_filename)

        print(f"Обработка {json_path} -> {out_path}")
        frame_id = int(base.split('.')[0])
        poses = get_position(poses_df, frame_id, map_num)
        convert_one(json_path, poses, out_path, edge_label2idx)

print("Готово!")