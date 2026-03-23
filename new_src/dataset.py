import os
import json
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
from scipy.spatial import KDTree
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

from scipy.spatial.distance import pdist
import math

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.44420420130352495, 0.41322746532289134, 0.3678658064565412], std=[0.24352604373543688, 0.24045797651069503, 0.24250136992133814]),
])


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print("Файл не найден")
    except json.JSONDecodeError as e:
        print(f"Ошибка в формате JSON: {e}")




def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images, 
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images                  = torch.cat([e[0] for e in batch])
    triplets_local_indexes  = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


#def extract_sequence_id(path):



class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path) :
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
    def __getitem__(self, index):
        return base_transform(path_to_pil_img(self.images_paths[index]))
    def __len__(self):
        return len(self.images_paths)

def make_item(img_path, graph_path=None, scene=None, pose=None):
    return {"img": img_path, "graph": graph_path, "scene": scene, "pose": pose}

class BaseDataset(data.Dataset):
    """
    Dataset with images from database and queries, used for inference (testing and building cache).
    Refactored: вынесены вспомогательные методы для поддержки нескольких модальностей.
    При несоответствии длин модальностей: печатаем информацию и ПРОПУСКАЕМ сцену.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="3RScan", split="train"):
        super().__init__()
        self.args = args
        self.modalites = getattr(args, 'modalites', ['image', 'graph', 'pose'])
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        # контейнеры для элементов (items)
        self.database_items = []
        self.queries_items = []

        meta_data = join(self.dataset_folder, 'files')
        scene_data_path = join(meta_data, '3RScan_small.json')
        scene_data = load_json(scene_data_path) or []


        # ---- Построение словарей соответствий ----
        self.transforms = {}          # scan_name -> 4x4 matrix to reference
        self.scan_to_ref = {}          # scan_name -> reference_name
        self.ref_to_scans = {}         # reference_name -> list of scan_names

        for entry in scene_data:
            ref = entry['reference']
            self.scan_to_ref[ref] = ref
            self.ref_to_scans[ref] = []
            self.transforms[ref] = np.eye(4)  # reference сам в себе

            for scan in entry.get('scans', []):
                scan_name = scan['reference']
                self.scan_to_ref[scan_name] = ref
                self.ref_to_scans[ref].append(scan_name)
                # Матрица из JSON (список 16 чисел) -> 4x4
                if 'transform' in scan:
                    mat = np.array(scan['transform']).reshape(4, 4)
                else:
                    mat = np.eye(4)
                self.transforms[scan_name] = mat
        print("SCAN LEN", len(self.scan_to_ref))

        # загружаем метаданные конкретного датасета (3RScan)
        if dataset_name == "3RScan":
            if split == 'train':
                split_scans_path = join(meta_data, "train_scans_small.txt")
            elif split == 'test':
                split_scans_path = join(meta_data, "test_resplit_scans_small.txt")
            else:
                split_scans_path = join(meta_data, f"{split}_scans.txt")

            with open(split_scans_path, 'r', encoding='utf-8') as f:
                scene_names = [l.strip() for l in f if l.strip()]
            
            print(len(scene_names))
            # обработка каждого сцены
            for scene_name in scene_names:
                # собираем элементы для database (самой сцены)

                is_ref = (scene_name in self.scan_to_ref and self.scan_to_ref[scene_name] == scene_name)
                if is_ref:
                    db_items = self._build_items_for_scene(scene_name)
                    if db_items:
                        self.database_items.extend(db_items)
                    else:
                        print(f"[SKIP] Пропускаю scene '{scene_name}' для database (несоответствие модальностей).")
                else:
                    print(f"ОШИБКА. {scene_name} не является reference сценой")
                    continue
                
                for query_scene in self.ref_to_scans[scene_name]:
                    # Добавляем в queries
                    q_items = self._build_items_for_scene(query_scene)
                    if q_items:
                        self.queries_items.extend(q_items)
                    else:
                        print(f"[SKIP] Пропускаю scene '{query_scene}' для query (несоответствие модальностей).")

                """
                db_items = self._build_items_for_scene(scene_name)
                if not db_items:
                    print(f"[SKIP] Пропускаю scene '{scene_name}' для database (несоответствие модальностей).")
                else:
                    self.database_items.extend(db_items)

                # ищем запись в scene_data и собираем элементы для связанных scan-queries
                for scene_entry in scene_data:
                    if scene_entry.get('reference') != scene_name:
                        continue
                    scans_queries = scene_entry.get('scans', [])
                    for scene_queries_data in scans_queries:
                        scene_queries_name = scene_queries_data.get('reference')
                        if not scene_queries_name:
                            continue
                        q_items = self._build_items_for_scene(scene_queries_name)
                        if not q_items:
                            print(f"[SKIP] Пропускаю scene '{scene_queries_name}' для queries (несоответствие модальностей).")
                        else:
                            self.queries_items.extend(q_items)
                """

        self.items = self.database_items + self.queries_items
        
        self.soft_positives_per_query = self._build_soft_positives(radius=1)  # радиус в метрах

    # ----------------- вспомогательные методы -----------------

    def _list_scene_files(self, scene_name):
        """
        Вернёт словарь списков путей для запрошенных модальностей.
        keys: 'images', 'poses', 'graphs' (в зависимости от modalites)
        """
        files = {}
        scene_graph_path = join(self.dataset_folder, "Splited_graphs", scene_name)
        scene_image_path = join(self.dataset_folder, 'scenes', scene_name, 'sequence')

        if 'image' in self.modalites:
            files['images'] = sorted([
                p for p in glob(join(scene_image_path, "**", "frame-*.color.jpg"), recursive=True)
                if '.rendered.' not in p
            ])
        if 'pose' in self.modalites:
            files['poses'] = sorted(glob(join(scene_image_path, "**", "*.pose.txt"), recursive=True))
        if 'graph' in self.modalites:
            files['graphs'] = sorted(glob(join(scene_graph_path, "**", "*.pt"), recursive=True))

        return files

    def _basename_key(self, path):
        """
        Попытка получить "ключ" из имени файла — например 'frame-0001' из 'frame-0001.color.jpg',
        'frame-0001' из 'frame-0001.pose.txt' или 'frame-0001' из 'frame-0001.pt'.
        """
        if path is None:
            return None
        name = os.path.basename(path)
        # Берём часть до первого "." — для твоего формата это даёт 'frame-XXXX'
        return name.split('.')[0]

    def _validate_scene_alignment(self, files, scene_name=None):
        """
        Проверка согласованности длин списков.
        Если есть рассинхронизация — печатаем подробную информацию и возвращаем False (чтобы сцена была пропущена).
        Возвращает True, если всё согласовано.
        """
        def _sample_names(lst, max_show=6):
            if not lst:
                return []
            if len(lst) <= max_show:
                return [os.path.basename(p) for p in lst]
            k = max_show // 2
            head = [os.path.basename(p) for p in lst[:k]]
            tail = [os.path.basename(p) for p in lst[-(max_show-k):]]
            return head + ["..."] + tail

        # собираем модальности, которые присутствуют
        present = [(k, len(files.get(k, []))) for k in ('images', 'poses', 'graphs') if k in files]
        if len(present) <= 1:
            return True  # нечего сравнивать — считаем OK

        # выберем опорную модальность: images если есть, иначе первая
        names_lengths = {name: ln for name, ln in present}
        base_name = 'images' if 'images' in names_lengths else present[0][0]
        base_len = names_lengths[base_name]

        # найдём несовпадения
        mismatches = [(name, ln) for name, ln in present if ln != base_len]
        if not mismatches:
            return True

        # Формируем подробный вывод
        scene_info = f" (scene_id='{scene_name}')" if scene_name is not None else ""
        lines = [f"[MODALITY MISMATCH] Несовпадение длин файлов для сцены{scene_info}:"]
        lines.append(f"  Опорная модальность: '{base_name}' -> {base_len} элементов.")
        for name, ln in present:
            lines.append(f"  - {name}: {ln}")
        lines.append("")

        # Показать примеры имён файлов
        for name, _ in present:
            sample = _sample_names(list(files.get(name, [])))
            if sample:
                lines.append(f"Примеры для '{name}' (total={len(files.get(name, []))}): {', '.join(sample)}")

        # Попытка показать отсутствующие ключи (frame ids)
        # Собираем ключи (basename до первой точки) для каждой модальности
        key_sets = {}
        for name in ('images', 'poses', 'graphs'):
            lst = files.get(name, [])
            if lst:
                key_sets[name] = set(self._basename_key(p) for p in lst if p is not None)

        if key_sets:
            lines.append("")
            lines.append("Анализ ключей (frame-ids):")
            all_keys = set().union(*key_sets.values())
            for name in sorted(key_sets.keys()):
                missing = sorted(list(all_keys - key_sets[name]))
                extra = sorted(list(key_sets[name] - all_keys))
                # Покажем до 6 отсутствующих/лишних ключей для наглядности
                def _show(xs):
                    if not xs:
                        return "[]"
                    if len(xs) <= 6:
                        return str(xs)
                    return str(xs[:3]) + " ... " + str(xs[-3:])
                lines.append(f"  - {name}: missing keys w.r.t. union: {_show(missing)}")

        lines.append("")
        lines.append(f"Активные модальности: {self.modalites}")

        msg = "\n".join(lines)
        # логируем/печатаем и возвращаем False — сцена будет пропущена
        """
        try:
            logging.warning(msg)
        except Exception:
            print(msg)
        return False
        """

    def _build_items_for_scene(self, scene_name):
        """
        Построить список item'ов (dict) для одной сцены.
        Если модальности не согласованы — вернёт пустой список (с печатью причины).
        """
        files = self._list_scene_files(scene_name)

        # Валидация: если не ок — пропускаем сцену
        ok = self._validate_scene_alignment(files, scene_name)
        if not ok:
            return []  # сцена пропускается

        images = files.get('images', [])
        poses = files.get('poses', [])
        graphs = files.get('graphs', [])

        # определяем число элементов (если есть images — используем их, иначе poses, иначе graphs)
        if images:
            n = len(images)
        elif poses:
            n = len(poses)
        elif graphs:
            n = len(graphs)
        else:
            return []  # сцена пустая / ничего не найдено

        transform_mat = self.transforms.get(scene_name, np.eye(4))
        transform_mat = np.eye(4)

        items = []
        for i in range(n):
            img_path = images[i] if i < len(images) else None
            graph_path = graphs[i] if i < len(graphs) else None

            camera_center = None
            if poses and i < len(poses):
                try:
                    pose_raw = self.read_pose_file(poses[i])
                    center_local = self.compute_camera_center_from_T(pose_raw)
                    center_hom = np.append(center_local, 1.0)
                    center_ref = (transform_mat @ center_hom)[:3]
                    camera_center = center_ref
                except Exception as e:
                    # Если чтение позы упало — логируем и продолжаем без камеры
                    logging.warning(f"Failed to read pose for {poses[i]}: {e}")
                    pose_raw = None
                    camera_center = None

            items.append({
                "img": img_path,
                "graph": graph_path,
                "scene": scene_name,
                "pose": camera_center,
            })
        return items

    # ----------------- оставшиеся публичные методы (пример) -----------------

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        sample = {
            "index": index,
            "scene": item.get("scene"),
            "pose": item.get("pose"),
            "img_path": item.get("img"),
            "graph_path": item.get("graph"),
        }

        if item.get("img") is not None:
            img = path_to_pil_img(item["img"])
            img_rotated_cw = img.transpose(Image.ROTATE_270)
            sample["image"] = base_transform(img_rotated_cw)

        if item.get("graph") is not None:
            sample["graph"] = torch.load(item["graph"], map_location="cpu")

        return sample

    # ----------------- оставляем твои полезные методы -----------------

    def _build_soft_positives(self, radius=5.0):
        """
        Для каждого query-элемента находит индексы database-элементов из того же reference,
        чьи центры камер находятся в пределах radius метров.
        Возвращает список списков индексов (по длине self.queries_items).
        """
        print('ХХХХХХХХХХХХХХХХХХХХХХХХХХХХХХХХХХ')
        # Группируем database-элементы по reference
        db_by_ref = {}
        for idx, item in enumerate(self.database_items):
            ref = item['scene']  # reference совпадает с именем скана для database
            if ref not in db_by_ref:
                db_by_ref[ref] = {'indices': [], 'centers': []}
            if item['pose'] is not None:
                db_by_ref[ref]['indices'].append(idx)
                db_by_ref[ref]['centers'].append(item['pose'])

        # Для каждой группы строим KD-Tree
        print("BUILD_SOFT_POSITIVES, DB", len(db_by_ref))
        trees = {}
        for ref, data in db_by_ref.items():
            if len(data['centers']) > 0:
                trees[ref] = {
                    'tree': KDTree(np.array(data['centers'])),
                    'indices': data['indices']
                }

        # Для каждого query ищем соседей
        print("TREES", len(trees))
        print("queries", len(self.queries_items))
        soft_positives = []
        for q_item in self.queries_items:
            if q_item['pose'] is None:
                soft_positives.append([])
                continue
            ref = self.scan_to_ref[q_item['scene']]  # вместо q_item.get('ref')
            if ref not in trees:
                soft_positives.append([])
                continue
            tree_info = trees[ref]
            # Ищем все точки в радиусе radius
            idxs_local = tree_info['tree'].query_ball_point(q_item['pose'], r=radius)
            # Преобразуем в глобальные индексы database_items
            global_idxs = [tree_info['indices'][i] for i in idxs_local]
            soft_positives.append(global_idxs)
        return soft_positives

    def read_pose_file(self, pose_path: str) -> np.ndarray:
        """Read pose file and return 4x4 numpy array (supports 16 numbers in one line or 4x4 layout)."""
        try:
            arr = np.loadtxt(pose_path)
        except Exception:
            arr = np.genfromtxt(pose_path)
        arr = np.array(arr).flatten()
        if arr.size != 16:
            raise ValueError(f"File {pose_path} doesn't contain 16 numbers (found {arr.size}).")
        return arr.reshape(4, 4).astype(np.float64)

    def compute_camera_center_from_T(self, T: np.ndarray, convention: str = 'cam2world') -> np.ndarray:
        """
        Return 3D camera center in world coords.
        convention:
        - 'cam2world' (default): T stores cam->world, center = T[:3,3]
        - 'world2cam': T stores world->cam, center = -R^T t
        """
        if T.shape != (4,4):
            raise ValueError("T must be 4x4")
        Rmat = T[:3,:3]
        t = T[:3,3]
        if convention == 'cam2world':
            return t.copy()
        elif convention == 'world2cam':
            return - Rmat.T.dot(t)
        else:
            raise ValueError("convention must be 'cam2world' or 'world2cam'")

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = transforms.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img

    def __repr__(self):
        return  (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {len(self.database_items)}; #queries: {len(self.queries_items)} >")
    def get_positives(self):
        return self.soft_positives_per_query

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import faiss
from functools import partial
from torch_geometric.data import Data, HeteroData, Batch as PyGBatch
import logging


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_graph_file(graph_path):
    if graph_path is None:
        return None
    try:
        return torch.load(graph_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(graph_path, map_location="cpu")

from torch_geometric.data import Batch

def _graph_to_list(g):
    if g is None:
        return []
    if isinstance(g, (list, tuple)):
        return [x for x in g if x is not None]
    return [g]

GRAPH_KEEP_KEYS = {
    "x",
    "edge_index",
    "edge_attr",
    "node_class",
    "edge_label",
    "edge_u_class",
    "edge_v_class",
}

def _sanitize_graph_obj(g, feat_dim=4):
    """
    Приводит graph к PyG Data с ЕДИНОЙ схемой полей.
    Все графы будут иметь одинаковые ключи:
      x, edge_index, edge_attr, edge_label, edge_u_class, edge_v_class, node_class
    """
    if g is None:
        return None

    if isinstance(g, PyGBatch):
        g = g.to_data_list()
        return g

    if isinstance(g, dict):
        g = dict_to_pyg_data(g, feat_dim=feat_dim)

    if not isinstance(g, (Data, HeteroData)):
        return g

    # x
    x = getattr(g, "x", None)
    if x is None:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
    elif not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    if x.ndim != 2 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)

    num_nodes = x.shape[0]

    # node_class
    node_class = getattr(g, "node_class", None)
    if node_class is None:
        node_class = torch.zeros((num_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        node_class = node_class.view(-1).long()
        if node_class.numel() != num_nodes:
            fixed = torch.zeros((num_nodes,), dtype=torch.long)
            m = min(num_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    # edge_index
    edge_index = getattr(g, "edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = edge_index.long()

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    # edge-level fields
    edge_attr = getattr(g, "edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
    else:
        if not torch.is_tensor(edge_attr):
            edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
        else:
            edge_attr = edge_attr.float()
        if edge_attr.ndim != 2:
            edge_attr = torch.empty((0, 7), dtype=torch.float32)

    edge_label = getattr(g, "edge_label", None)
    if edge_label is None:
        edge_label = torch.empty((0,), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_label):
            edge_label = torch.tensor(edge_label, dtype=torch.long)
        else:
            edge_label = edge_label.long()
        edge_label = edge_label.view(-1)

    edge_u_class = getattr(g, "edge_u_class", None)
    if edge_u_class is None:
        edge_u_class = torch.empty((0,), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_u_class):
            edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
        else:
            edge_u_class = edge_u_class.long()
        edge_u_class = edge_u_class.view(-1)

    edge_v_class = getattr(g, "edge_v_class", None)
    if edge_v_class is None:
        edge_v_class = torch.empty((0,), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_v_class):
            edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
        else:
            edge_v_class = edge_v_class.long()
        edge_v_class = edge_v_class.view(-1)

    # если есть ребра — выкидываем выходящие за диапазон
    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        if not valid.all():
            edge_index = edge_index[:, valid]

            if edge_attr.shape[0] == valid.shape[0]:
                edge_attr = edge_attr[valid]
            else:
                edge_attr = edge_attr[:edge_index.shape[1]]

            if edge_label.shape[0] == valid.shape[0]:
                edge_label = edge_label[valid]
            else:
                edge_label = edge_label[:edge_index.shape[1]]

            if edge_u_class.shape[0] == valid.shape[0]:
                edge_u_class = edge_u_class[valid]
            else:
                edge_u_class = edge_u_class[:edge_index.shape[1]]

            if edge_v_class.shape[0] == valid.shape[0]:
                edge_v_class = edge_v_class[valid]
            else:
                edge_v_class = edge_v_class[:edge_index.shape[1]]

    # ВАЖНО: возвращаем Data со ВСЕМИ ключами
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )

from torch_geometric.data import Batch as PyGBatch

def _collate_graph_samples(samples, feat_dim=4):
    """
    Collate для inference/cache режима.
    Возвращает dict:
      {
        "graph": PyG Batch,
        "scene": list[str] | None
      }
    """
    out = {}

    # metadata
    indices = [s.get("index") for s in samples if s.get("index") is not None]
    scenes = [s.get("scene") for s in samples if s.get("scene") is not None]

    out["index"] = torch.as_tensor(indices, dtype=torch.long) if len(indices) > 0 else None
    out["scene"] = scenes if len(scenes) > 0 else None

    # graphs
    graph_values = [s.get("graph") for s in samples if s.get("graph") is not None]
    flat_graphs = []

    for g in graph_values:
        g = _sanitize_graph_obj(g, feat_dim=feat_dim)

        if g is None:
            continue

        if isinstance(g, list):
            for x in g:
                x = _ensure_nonempty(_sanitize_graph_obj(x, feat_dim=feat_dim), feat_dim)
                flat_graphs.append(x)
        else:
            flat_graphs.append(_ensure_nonempty(g, feat_dim))

    if len(flat_graphs) == 0:
        out["graph"] = None
    else:
        out["graph"] = PyGBatch.from_data_list(flat_graphs)

    return out



def dict_to_pyg_data(d, feat_dim=4, edge_attr_dim=7):
    if d is None:
        return None

    # x
    x = d.get("x", None)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif x is not None and not torch.is_tensor(x):
        x = torch.tensor(np.asarray(x), dtype=torch.float32)

    if x is None or x.numel() == 0 or x.ndim != 2 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
    else:
        x = x.float()

    num_nodes = x.shape[0]

    # edge_index
    edge_index = d.get("edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = (
            torch.tensor(edge_index, dtype=torch.long)
            if not torch.is_tensor(edge_index)
            else edge_index.long()
        )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    # edge_attr
    edge_attr = d.get("edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()

    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(0)

    # node_class
    node_class = d.get("node_class", None)
    if node_class is None:
        node_class = torch.zeros((num_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        else:
            node_class = node_class.long()
        node_class = node_class.view(-1)

        if node_class.numel() != num_nodes:
            fixed = torch.zeros((num_nodes,), dtype=torch.long)
            m = min(num_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    # Optional edge-level fields
    edge_label = d.get("edge_label", None)
    if edge_label is not None and not torch.is_tensor(edge_label):
        edge_label = torch.tensor(edge_label, dtype=torch.long)
    if torch.is_tensor(edge_label):
        edge_label = edge_label.view(-1)

    edge_u_class = d.get("edge_u_class", None)
    if edge_u_class is not None and not torch.is_tensor(edge_u_class):
        edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
    if torch.is_tensor(edge_u_class):
        edge_u_class = edge_u_class.view(-1)

    edge_v_class = d.get("edge_v_class", None)
    if edge_v_class is not None and not torch.is_tensor(edge_v_class):
        edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
    if torch.is_tensor(edge_v_class):
        edge_v_class = edge_v_class.view(-1)

    # Remove invalid edges
    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )

        if not valid.all():
            edge_index = edge_index[:, valid]

            if edge_attr is not None:
                edge_attr = edge_attr[valid] if edge_attr.shape[0] == valid.shape[0] else edge_attr[:edge_index.shape[1]]

            if edge_label is not None:
                edge_label = edge_label[valid] if edge_label.shape[0] == valid.shape[0] else edge_label[:edge_index.shape[1]]

            if edge_u_class is not None:
                edge_u_class = edge_u_class[valid] if edge_u_class.shape[0] == valid.shape[0] else edge_u_class[:edge_index.shape[1]]

            if edge_v_class is not None:
                edge_v_class = edge_v_class[valid] if edge_v_class.shape[0] == valid.shape[0] else edge_v_class[:edge_index.shape[1]]

    json_path = d.get("json_path", None)
    if node_class.numel() != x.shape[0]:
        raise ValueError(
            f"Malformed graph: x has {x.shape[0]} nodes, "
            f"but node_class has {node_class.numel()} elements. "
            f"json_path={json_path}"
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_class=node_class,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
    )


def _flatten_graphs(graphs):
    """
    Приводит вход к плоскому списку Data-объектов.
    """
    flat = []
    for g in graphs:
        g = _sanitize_graph_obj(g)
        if g is None:
            continue
        if isinstance(g, list):
            flat.extend([x for x in g if x is not None])
        else:
            flat.append(g)
    return flat


def _collate_graph_objects(graphs):
    """
    Collate графов в Batch.
    """
    graphs = _flatten_graphs(graphs)
    if len(graphs) == 0:
        return None

    first = graphs[0]

    if isinstance(first, (Data, HeteroData)):
        return PyGBatch.from_data_list(graphs)

    if torch.is_tensor(first):
        if all(torch.is_tensor(g) and g.shape == first.shape for g in graphs):
            return torch.stack(graphs, dim=0)
        return graphs

    return graphs

def graph_collate_fn(batch, feat_dim=4):
    """
    Unified collate for graph-only pipeline.

    Inference/cache mode:
        batch = list[dict]
        returns dict with graph batch + global indices

    Training mode:
        batch = list[(sample_dict, triplets_local_indexes, triplets_global_indexes)]
        returns:
            batch_samples, triplets_local_indexes, triplets_global_indexes
    """
    first = batch[0]

    # -------------------------
    # Inference / cache mode
    # -------------------------
    if isinstance(first, dict):
        return _collate_graph_samples(batch, feat_dim=feat_dim)

    # -------------------------
    # Training mode
    # -------------------------
    samples, triplets_local_indexes, triplets_global_indexes = zip(*batch)

    graphs = []
    for sample in samples:
        g = _sanitize_graph_obj(sample["graph"], feat_dim=feat_dim)
        if g is None:
            continue
        if isinstance(g, list):
            for x in g:
                graphs.append(_ensure_nonempty(_sanitize_graph_obj(x, feat_dim=feat_dim), feat_dim))
        else:
            graphs.append(_ensure_nonempty(g, feat_dim))

    batch_graph = PyGBatch.from_data_list(graphs) if len(graphs) > 0 else None

    triplets_local_indexes = torch.cat([x[None] for x in triplets_local_indexes], dim=0)
    triplets_global_indexes = torch.cat([x[None] for x in triplets_global_indexes], dim=0)

    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i

    batch_samples = {"graph": batch_graph}
    return batch_samples, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes



def _collate_samples(samples):
    """
    Collate list of sample dicts into a batch dict.

    Each sample can contain:
      - image: Tensor[C,H,W] or Tensor[V,C,H,W] or None
      - graph: graph object or list/tuple of graph objects or None
      - metadata fields: index, scene, pose, img_path, graph_path

    Returns a dict with the same keys, where modality keys are batched.
    """
    out = {}

    # collect metadata always as lists
    for meta_key in ["index", "scene", "pose", "img_path", "graph_path"]:
        values = [s.get(meta_key) for s in samples]
        if any(v is not None for v in values):
            out[meta_key] = values

    # images
    image_values = [s.get("image") for s in samples if s.get("image") is not None]
    if len(image_values) > 0:
        out["image"] = torch.cat(image_values, dim=0)
    else:
        out["image"] = None

    # graphs
    graph_values = [s.get("graph") for s in samples if s.get("graph") is not None]
    if len(graph_values) == 0:
        out["graph"] = None
    else:
        flat_graphs = []
        for g in graph_values:
            if isinstance(g, (list, tuple)):
                flat_graphs.extend([x for x in g if x is not None])
            else:
                flat_graphs.append(g)
        out["graph"] = _collate_graph_objects(flat_graphs)

    return out


def collate_fn(batch):
    """
    Triplet collate.

    Expects batch items:
        (sample_dict, triplets_local_indexes, triplets_global_indexes)

    sample_dict may contain image, graph, or both.

    Returns:
        batch_samples: dict with keys image / graph / metadata
        triplets_local_indexes: Tensor[batch_size * num_neg, 3]
        triplets_global_indexes: Tensor[batch_size, 2 + num_neg]
    """
    samples, triplets_local_indexes, triplets_global_indexes = zip(*batch)

    batch_samples = _collate_samples(list(samples))

    triplets_local_indexes = torch.cat([x[None] for x in triplets_local_indexes], dim=0)
    triplets_global_indexes = torch.cat([x[None] for x in triplets_global_indexes], dim=0)

    # same logic as in your original collate:
    # shift local indexes by the sample size
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i

    return batch_samples, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes

def compute_cache_graph_only(args, model, subset_ds, cache_shape):
    subset_dl = DataLoader(
        dataset=subset_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        shuffle=False,
        pin_memory=(args.device == "cuda"),
        collate_fn=graph_collate_fn
    )

    model = model.eval()
    cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)

    with torch.no_grad():
        cursor = 0
        subset_indices = getattr(subset_ds, "indices", None)

        for batch in tqdm(subset_dl, ncols=100):
            graph_batch = batch["graph"].to(args.device)

            global_features = model(graph_batch)  # [B, D]
            batch_size = global_features.shape

            if subset_indices is None:
                batch_indices = np.arange(cursor, cursor + batch_size, dtype=np.int32)
            else:
                batch_indices = np.array(subset_indices[cursor:cursor + batch_size], dtype=np.int32)

            cache[batch_indices] = global_features.cpu().numpy()
            cursor += batch_size

    return cache

def graph_collate_fn_cache(batch, feat_dim=4):
    """
    Compatibility wrapper.
    Returns only PyG Batch, but reuses the same sanitization logic.
    """
    out = graph_collate_fn(batch, feat_dim=feat_dim)
    if not isinstance(out, dict):
        raise ValueError("graph_collate_fn_cache expects inference/cache batch of dict samples.")
    if out["graph"] is None:
        raise ValueError("No graphs found in cache batch.")
    return out["graph"]

def _ensure_nonempty(data, feat_dim):
    if data is None:
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 7), dtype=torch.float32),
            edge_label=torch.empty((0,), dtype=torch.long),
            edge_u_class=torch.empty((0,), dtype=torch.long),
            edge_v_class=torch.empty((0,), dtype=torch.long),
            node_class=torch.full((1,), 0, dtype=torch.long),
        )

    # x
    x = getattr(data, "x", None)
    if x is None or not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32) if x is not None else None

    synthetic_node = False
    if x is None or x.numel() == 0 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
        synthetic_node = True
    else:
        x = x.float()

    n_nodes = x.shape[0]

    # node_class
    node_class = getattr(data, "node_class", None)
    if node_class is None:
        node_class = torch.full((n_nodes,), 0, dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        node_class = node_class.view(-1)

        if node_class.numel() != n_nodes:
            fixed = torch.full((n_nodes,), 0, dtype=torch.long)
            m = min(n_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    # edge_index
    if synthetic_node:
        # если узел синтетический, ребра надо сбросить полностью
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
        edge_label = torch.empty((0,), dtype=torch.long)
        edge_u_class = torch.empty((0,), dtype=torch.long)
        edge_v_class = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            if not torch.is_tensor(edge_index):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            else:
                edge_index = edge_index.long()

            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        # edge_attr / labels
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and not torch.is_tensor(edge_attr):
            edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)

        edge_label = getattr(data, "edge_label", None)
        if edge_label is not None and not torch.is_tensor(edge_label):
            edge_label = torch.tensor(edge_label, dtype=torch.long)

        edge_u_class = getattr(data, "edge_u_class", None)
        if edge_u_class is not None and not torch.is_tensor(edge_u_class):
            edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)

        edge_v_class = getattr(data, "edge_v_class", None)
        if edge_v_class is not None and not torch.is_tensor(edge_v_class):
            edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)

        # если есть ребра — проверим диапазон индексов и выкинем битые
        if edge_index.numel() > 0:
            valid = (
                (edge_index[0] >= 0) & (edge_index[0] < n_nodes) &
                (edge_index[1] >= 0) & (edge_index[1] < n_nodes)
            )

            if not valid.all():
                edge_index = edge_index[:, valid]

                if edge_attr is not None and torch.is_tensor(edge_attr) and edge_attr.shape[0] == valid.shape[0]:
                    edge_attr = edge_attr[valid]
                elif edge_attr is not None and torch.is_tensor(edge_attr):
                    edge_attr = edge_attr[:edge_index.shape[1]]

                if edge_label is not None and torch.is_tensor(edge_label) and edge_label.shape[0] == valid.shape[0]:
                    edge_label = edge_label[valid]
                elif edge_label is not None and torch.is_tensor(edge_label):
                    edge_label = edge_label[:edge_index.shape[1]]

                if edge_u_class is not None and torch.is_tensor(edge_u_class) and edge_u_class.shape[0] == valid.shape[0]:
                    edge_u_class = edge_u_class[valid]
                elif edge_u_class is not None and torch.is_tensor(edge_u_class):
                    edge_u_class = edge_u_class[:edge_index.shape[1]]

                if edge_v_class is not None and torch.is_tensor(edge_v_class) and edge_v_class.shape[0] == valid.shape[0]:
                    edge_v_class = edge_v_class[valid]
                elif edge_v_class is not None and torch.is_tensor(edge_v_class):
                    edge_v_class = edge_v_class[:edge_index.shape[1]]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )
# ---------------------------------------------------------------------
# TripletsDataset
# ---------------------------------------------------------------------

class TripletsDataset(BaseDataset):
    """
    Triplets dataset adapted to the current BaseDataset structure.

    Works with:
      - image-only
      - graph-only
      - image + graph

    Expected BaseDataset fields:
      - self.database_items
      - self.queries_items
      - self.items
      - self.soft_positives_per_query (will be recomputed here)
      - self.scan_to_ref
      - self.ref_to_scans

    Each item is expected to be a dict like:
      {
        "img": "...",
        "graph": "...",
        "scene": "...",
        "pose": np.ndarray([3]) or None
      }
    """

    def __init__(self, args, datasets_folder="datasets", dataset_name="3RScan", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)

        self.mining = getattr(args, "mining", "full")
        self.neg_samples_num = getattr(args, "neg_samples_num", 1000)
        self.negs_num_per_query = negs_num_per_query
        self.is_inference = False

        self.resize = getattr(args, "resize", None)
        self.train_positives_dist_threshold = getattr(args, "train_positives_dist_threshold", 10.0)
        self.soft_positives_radius = getattr(args, "soft_positives_radius", 1.0)

        self.modalities = tuple(
            getattr(args, "modalities", getattr(args, "modalites", ["image", "graph"]))
        )
        self.use_images = "image" in self.modalities
        self.use_graphs = "graph" in self.modalities

        identity_transform = transforms.Lambda(lambda x: x)

        if self.resize is not None:
            self.resized_transform = transforms.Compose([
                transforms.Resize(self.resize),
                base_transform
            ])
        else:
            self.resized_transform = base_transform

        q_transforms = [
            transforms.ColorJitter(brightness=args.brightness) if getattr(args, "brightness", None) is not None else identity_transform,
            transforms.ColorJitter(contrast=args.contrast) if getattr(args, "contrast", None) is not None else identity_transform,
            transforms.ColorJitter(saturation=args.saturation) if getattr(args, "saturation", None) is not None else identity_transform,
            transforms.ColorJitter(hue=args.hue) if getattr(args, "hue", None) is not None else identity_transform,
            transforms.RandomPerspective(args.rand_perspective) if getattr(args, "rand_perspective", None) is not None else identity_transform,
            transforms.RandomResizedCrop(size=self.resize, scale=(1 - args.random_resized_crop, 1))
            if getattr(args, "random_resized_crop", None) is not None and self.resize is not None else identity_transform,
            transforms.RandomRotation(degrees=args.random_rotation) if getattr(args, "random_rotation", None) is not None else identity_transform,
            self.resized_transform,
        ]
        self.query_transform = transforms.Compose(q_transforms)

        # Build helper views from BaseDataset storage
        self._sync_views()

        # Hard positives + query filtering
        self._filter_queries_without_soft_positives()

        # Recompute soft positives after query filtering
        self.soft_positives_per_query = self._build_soft_positives(radius=self.soft_positives_radius)

        if self.mining == "full":
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]

        self.triplets_global_indexes = torch.empty((0, 2 + self.negs_num_per_query), dtype=torch.long)



    # -----------------------------------------------------------------
    # Views / indexing
    # -----------------------------------------------------------------

    def _sync_views(self):
        self.database_num = len(self.database_items)
        self.queries_num = len(self.queries_items)

        self.database_paths = [item.get("img") for item in self.database_items]
        self.queries_paths = [item.get("img") for item in self.queries_items]
        self.images_paths = self.database_paths + self.queries_paths

        self.database_utms = np.array(
            [item["pose"] for item in self.database_items if item.get("pose") is not None],
            dtype=np.float32
        ) if any(item.get("pose") is not None for item in self.database_items) else np.empty((0, 3), dtype=np.float32)

        self.queries_utms = np.array(
            [item["pose"] for item in self.queries_items if item.get("pose") is not None],
            dtype=np.float32
        ) if any(item.get("pose") is not None for item in self.queries_items) else np.empty((0, 3), dtype=np.float32)

        self.database_pose_indices = np.array(
            [i for i, item in enumerate(self.database_items) if item.get("pose") is not None],
            dtype=np.int32
        )
        self.queries_pose_indices = np.array(
            [i for i, item in enumerate(self.queries_items) if item.get("pose") is not None],
            dtype=np.int32
        )

        self.items = self.database_items + self.queries_items

    def _build_item_sample(self, item, is_query=False, with_meta=True):
        """
        Build a sample dict for one item.
        The returned dict always contains image/graph keys (possibly None),
        plus metadata if with_meta=True.
        """
        sample = {}

        if with_meta:
            sample["scene"] = item.get("scene")
            sample["pose"] = item.get("pose")
            sample["img_path"] = item.get("img")
            sample["graph_path"] = item.get("graph")

        if self.use_images:
            img_path = item.get("img")
            if img_path is not None:
                img = path_to_pil_img(img_path)
                img = self.query_transform(img) if is_query else self.resized_transform(img)
                sample["image"] = img
            else:
                sample["image"] = None
        else:
            sample["image"] = None

        if self.use_graphs:
            graph_path = item.get("graph")
            sample["graph"] = _load_graph_file(graph_path) if graph_path is not None else None
        else:
            sample["graph"] = None

        return sample


    def _filter_queries_without_soft_positives(self):
        keep = []
        for i, q_item in enumerate(self.queries_items):
            has_pose = q_item.get("pose") is not None
            has_positives = len(self.soft_positives_per_query[i]) > 0
            if has_pose and has_positives:
                keep.append(i)

        removed = len(self.queries_items) - len(keep)
        if removed > 0:
            logging.info(
                f"Removed {removed} queries without soft positives or without pose. "
                f"They will not be used for training."
            )

        self.queries_items = [self.queries_items[i] for i in keep]
        self.queries_paths = [self.queries_paths[i] for i in keep]
        self.soft_positives_per_query = [
            np.asarray(self.soft_positives_per_query[i], dtype=np.int32)
            for i in keep
        ]

        self.queries_num = len(self.queries_items)
        self.items = self.database_items + self.queries_items

    def _complete_negative_indexes(self, query_index, num_negatives=None):
        """
        Sample negatives only from a random scene whose reference != query reference.
        All negatives are taken from the SAME randomly chosen foreign reference scene.
        """
        if num_negatives is None:
            num_negatives = self.negs_num_per_query

        query_item = self.queries_items[query_index]
        query_scene = query_item["scene"]
        query_ref = self.scan_to_ref.get(query_scene, query_scene)

        # all references except the query reference
        other_refs = [ref for ref in self.ref_to_scans.keys() if ref != query_ref]
        if len(other_refs) == 0:
            raise RuntimeError(f"No foreign reference scenes found for query ref='{query_ref}'")

        # try several times to find a non-empty foreign reference scene
        for _ in range(50):
            chosen_ref = np.random.choice(other_refs)

            candidate_db_indexes = [
                db_idx
                for db_idx, db_item in enumerate(self.database_items)
                if self.scan_to_ref.get(db_item["scene"], db_item["scene"]) == chosen_ref
            ]

            if len(candidate_db_indexes) == 0:
                continue

            replace = len(candidate_db_indexes) < num_negatives
            neg_indexes = np.random.choice(
                candidate_db_indexes,
                size=num_negatives,
                replace=replace
            ).astype(np.int32)

            return neg_indexes

        raise RuntimeError(
            f"Could not sample negatives for query_index={query_index} "
            f"because no non-empty foreign reference scene was found."
        )

    # -----------------------------------------------------------------
    # Dataset API
    # -----------------------------------------------------------------

    def __len__(self):
        if self.is_inference:
            return len(self.items)
        return len(self.triplets_global_indexes)

    def __getitem__(self, index):
        if self.is_inference:
            item = self.items[index]
            return self._build_item_sample(item, is_query=(index >= self.database_num), with_meta=True)

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index],
            (1, 1, self.negs_num_per_query)
        )

        query_index = int(query_index.item())
        best_positive_index = int(best_positive_index.item())
        neg_indexes = [int(i.item()) for i in neg_indexes]

        query_item = self.queries_items[query_index]
        positive_item = self.database_items[best_positive_index]
        negative_items = [self.database_items[i] for i in neg_indexes]

        items = [query_item, positive_item, *negative_items]

        sample = {}

        if self.use_images:
            images = [
                self._build_item_sample(items[0], is_query=True, with_meta=False)["image"],
                self._build_item_sample(items[1], is_query=False, with_meta=False)["image"],
                *[self._build_item_sample(it, is_query=False, with_meta=False)["image"] for it in items[2:]]
            ]
            sample["image"] = torch.stack(images, dim=0)
        else:
            sample["image"] = None

        if self.use_graphs:
            graphs = [
                self._build_item_sample(items[0], is_query=True, with_meta=False)["graph"],
                self._build_item_sample(items[1], is_query=False, with_meta=False)["graph"],
                *[self._build_item_sample(it, is_query=False, with_meta=False)["graph"] for it in items[2:]]
            ]
            sample["graph"] = graphs
        else:
            sample["graph"] = None

        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int64)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3))
            )

        return sample, triplets_local_indexes, self.triplets_global_indexes[index]

    def __repr__(self):
        return  (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {len(self.database_items)}; #queries: {len(self.queries_items)} >")

    def get_positives(self):
        return self.soft_positives_per_query

    # -----------------------------------------------------------------
    # Cache / model utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _move_to_device(obj, device):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: TripletsDataset._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(TripletsDataset._move_to_device(x, device) for x in obj)
        if hasattr(obj, "to"):
            try:
                return obj.to(device)
            except Exception:
                return obj
        return obj

    def _prepare_model_input(self, batch):
        """
        Extract only modality tensors/objects from a collated batch dict.
        If there is a single modality, returns that object directly.
        If there are multiple modalities, returns a dict.
        """
        if not isinstance(batch, dict):
            return batch

        model_inputs = {}
        if self.use_images and batch.get("image") is not None:
            model_inputs["image"] = batch["image"]
        if self.use_graphs and batch.get("graph") is not None:
            model_inputs["graph"] = batch["graph"]

        if len(model_inputs) == 1:
            return next(iter(model_inputs.values()))

        return model_inputs

    @staticmethod
    def _split_model_output(output):
        """
        Accepts:
          - (local_features, global_features)
          - {"local_features": ..., "global_features": ...}
          - {"local": ..., "global": ...}
          - global tensor only

        Returns: local_features, global_features
        """
        local_features = None
        global_features = None

        if isinstance(output, dict):
            local_features = output.get("local_features", output.get("local", None))
            global_features = output.get("global_features", output.get("global", None))
            if global_features is None and len(output) == 1:
                global_features = next(iter(output.values()))
        elif isinstance(output, (tuple, list)):
            if len(output) == 1:
                global_features = output[0]
            elif len(output) >= 2:
                local_features, global_features = output[0], output[1]
        else:
            global_features = output

        return local_features, global_features

    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """
        Compute the cache containing global embeddings, used to find best positives
        and hardest negatives.

        The dataset used here should be in inference mode, i.e. __getitem__ returns sample dicts.
        """
        subset_dl = DataLoader(
            dataset=subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=(args.device == "cuda"),
            collate_fn=partial(graph_collate_fn_cache, feat_dim=4)
        )
        model = model.eval()

        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)

        # If the model returns local feature maps, we create the local cache lazily.
        with torch.no_grad():
            cursor = 0
            subset_indices = getattr(subset_ds, "indices", None)

            for batch in tqdm(subset_dl, ncols=100):
                #print("batch, cache", batch)
                batch = TripletsDataset._move_to_device(batch, args.device)

                output = model(batch)
                global_features = output

                if global_features is None:
                    raise RuntimeError("Model must return global features for cache computation.")


                batch_size = global_features.shape[0]

                if subset_indices is None:
                    batch_indices = np.arange(cursor, cursor + batch_size, dtype=np.int32)
                else:
                    batch_indices = np.array(subset_indices[cursor:cursor + batch_size], dtype=np.int32)

                cache[batch_indices] = global_features.detach().cpu().numpy()

                cursor += batch_size

        return cache
        

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(
                f"For query {self.queries_paths[query_index]} with index {query_index} "
                f"features have not been computed."
            )
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_indexes = self.soft_positives_per_query[query_index]
        positives_features = cache[positives_indexes]

        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)

        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = positives_indexes[best_positive_num[0][0]]
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_samples = np.asarray(neg_samples, dtype=np.int32).reshape(-1)
        if len(neg_samples) == 0:
            return np.empty((0,), dtype=np.int32)

        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)

        k = min(self.negs_num_per_query, len(neg_samples))
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), k)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes

    # -----------------------------------------------------------------
    # Triplet mining
    # -----------------------------------------------------------------

    def compute_triplets(self, args, model):
        self.is_inference = True

        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining in ["partial", "msls_weighted"]:
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
        else:
            raise ValueError(f"Unknown mining method: {self.mining}")

        self.is_inference = False

    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []

        refresh_rate = min(self.queries_num, getattr(args, "cache_refresh_rate", self.queries_num))
        sampled_queries_indexes = np.random.choice(self.queries_num, refresh_rate, replace=False)

        positives_indexes = [self.soft_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        positives_indexes = list(np.unique(positives_indexes))

        subset_indices = positives_indexes + list(sampled_queries_indexes + self.database_num)
        subset_ds = Subset(self, subset_indices)

        cache = self.compute_cache(args, model, subset_ds, (len(self.items), args.features_dim))

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)

            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = self._complete_negative_indexes(query_index)

            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []

        refresh_rate = min(self.queries_num, getattr(args, "cache_refresh_rate", self.queries_num))
        sampled_queries_indexes = np.random.choice(self.queries_num, refresh_rate, replace=False)

        database_indexes = list(range(self.database_num))
        subset_indices = database_indexes + list(sampled_queries_indexes + self.database_num)
        subset_ds = Subset(self, subset_indices)

        cache = self.compute_cache(args, model, subset_ds, (len(self.items), args.features_dim))

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)

            neg_indexes = np.random.choice(
                self.database_num,
                size=min(self.neg_samples_num, self.database_num),
                replace=False
            )

            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = self._complete_negative_indexes(query_index)

            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []

        refresh_rate = min(self.queries_num, getattr(args, "cache_refresh_rate", self.queries_num))
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(self.queries_num, refresh_rate, replace=False)
        elif self.mining == "msls_weighted":
            sampled_queries_indexes = np.random.choice(self.queries_num, refresh_rate, replace=False, p=self.weights)
        else:
            raise ValueError(f"Unsupported mining method in partial stage: {self.mining}")

        db_neg_count = min(self.database_num, self.neg_samples_num)
        sampled_database_indexes = np.random.choice(self.database_num, db_neg_count, replace=False)

        positives_indexes = [self.soft_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]

        database_indexes = list(np.unique(list(sampled_database_indexes) + positives_indexes))

        subset_indices = database_indexes + list(sampled_queries_indexes + self.database_num)
        subset_ds = Subset(self, subset_indices)

        cache = self.compute_cache(args, model, subset_ds, cache_shape=(len(self.items), args.features_dim))

        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)

            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = self._complete_negative_indexes(query_index)

            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)