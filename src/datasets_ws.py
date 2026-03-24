import json
import logging
import math
import os
from glob import glob
from os.path import basename, join

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data, HeteroData

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_MEAN = [0.44420420130352495, 0.41322746532289134, 0.3678658064565412]
DEFAULT_STD = [0.24352604373543688, 0.24045797651069503, 0.24250136992133814]

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ]
)


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except json.JSONDecodeError as e:
        print(f"JSON error in {path}: {e}")
    return None


def make_item(img_path, graph_path=None, scene=None, pose=None):
    return {"img": img_path, "graph": graph_path, "scene": scene, "pose": pose}


def _identity():
    return transforms.Lambda(lambda x: x)


def _stack_images(images):
    """
    Accepts a list of image tensors.
    - If tensors are [C,H,W], returns [B,C,H,W]
    - If tensors are already [N,C,H,W], concatenates on dim=0
    """
    images = [x for x in images if x is not None]
    if not images:
        return None

    first = images[0]
    if not torch.is_tensor(first):
        return images

    if first.ndim == 3:
        if all(torch.is_tensor(x) and x.ndim == 3 for x in images):
            return torch.stack(images, dim=0)
        return torch.cat([x.unsqueeze(0) if x.ndim == 3 else x for x in images], dim=0)

    if first.ndim == 4:
        return torch.cat(images, dim=0)

    return torch.stack(images, dim=0)


# ---------------------------------------------------------------------
# Graph sanitization / batching
# ---------------------------------------------------------------------

GRAPH_KEEP_KEYS = {
    "x",
    "edge_index",
    "edge_attr",
    "node_class",
    "edge_label",
    "edge_u_class",
    "edge_v_class",
}


def _graph_to_list(g):
    if g is None:
        return []
    if isinstance(g, (list, tuple)):
        return [x for x in g if x is not None]
    return [g]


def dict_to_pyg_data(d, feat_dim=4, edge_attr_dim=7):
    if d is None:
        return None

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

    edge_index = d.get("edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = edge_index.long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = d.get("edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(0)

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

    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        if not valid.all():
            edge_index = edge_index[:, valid]
            edge_count = edge_index.shape[1]

            if torch.is_tensor(edge_attr):
                edge_attr = edge_attr[valid] if edge_attr.shape[0] == valid.shape[0] else edge_attr[:edge_count]
            if torch.is_tensor(edge_label):
                edge_label = edge_label[valid] if edge_label.shape[0] == valid.shape[0] else edge_label[:edge_count]
            if torch.is_tensor(edge_u_class):
                edge_u_class = edge_u_class[valid] if edge_u_class.shape[0] == valid.shape[0] else edge_u_class[:edge_count]
            if torch.is_tensor(edge_v_class):
                edge_v_class = edge_v_class[valid] if edge_v_class.shape[0] == valid.shape[0] else edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_class=node_class,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
    )


def _ensure_nonempty(data_obj, feat_dim=4):
    if data_obj is None:
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 7), dtype=torch.float32),
            edge_label=torch.empty((0,), dtype=torch.long),
            edge_u_class=torch.empty((0,), dtype=torch.long),
            edge_v_class=torch.empty((0,), dtype=torch.long),
            node_class=torch.zeros((1,), dtype=torch.long),
        )

    if isinstance(data_obj, dict):
        data_obj = dict_to_pyg_data(data_obj, feat_dim=feat_dim)

    if not isinstance(data_obj, (Data, HeteroData)):
        return data_obj

    x = getattr(data_obj, "x", None)
    if x is None or not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32) if x is not None else None

    synthetic_node = False
    if x is None or x.numel() == 0 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
        synthetic_node = True
    else:
        x = x.float()

    n_nodes = x.shape[0]

    node_class = getattr(data_obj, "node_class", None)
    if node_class is None:
        node_class = torch.zeros((n_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        else:
            node_class = node_class.long()
        node_class = node_class.view(-1)
        if node_class.numel() != n_nodes:
            fixed = torch.zeros((n_nodes,), dtype=torch.long)
            m = min(n_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    if synthetic_node:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
        edge_label = torch.empty((0,), dtype=torch.long)
        edge_u_class = torch.empty((0,), dtype=torch.long)
        edge_v_class = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = getattr(data_obj, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            if not torch.is_tensor(edge_index):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            else:
                edge_index = edge_index.long()
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_attr = getattr(data_obj, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.empty((0, 7), dtype=torch.float32)
        elif not torch.is_tensor(edge_attr):
            edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
        else:
            edge_attr = edge_attr.float()

        edge_label = getattr(data_obj, "edge_label", None)
        if edge_label is None:
            edge_label = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_label):
            edge_label = torch.tensor(edge_label, dtype=torch.long)
        else:
            edge_label = edge_label.long()
        edge_label = edge_label.view(-1)

        edge_u_class = getattr(data_obj, "edge_u_class", None)
        if edge_u_class is None:
            edge_u_class = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_u_class):
            edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
        else:
            edge_u_class = edge_u_class.long()
        edge_u_class = edge_u_class.view(-1)

        edge_v_class = getattr(data_obj, "edge_v_class", None)
        if edge_v_class is None:
            edge_v_class = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_v_class):
            edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
        else:
            edge_v_class = edge_v_class.long()
        edge_v_class = edge_v_class.view(-1)

        if edge_index.numel() > 0:
            valid = (
                (edge_index[0] >= 0) & (edge_index[0] < n_nodes) &
                (edge_index[1] >= 0) & (edge_index[1] < n_nodes)
            )
            if not valid.all():
                edge_index = edge_index[:, valid]
                edge_count = edge_index.shape[1]
                if edge_attr.shape[0] == valid.shape[0]:
                    edge_attr = edge_attr[valid]
                else:
                    edge_attr = edge_attr[:edge_count]
                if edge_label.shape[0] == valid.shape[0]:
                    edge_label = edge_label[valid]
                else:
                    edge_label = edge_label[:edge_count]
                if edge_u_class.shape[0] == valid.shape[0]:
                    edge_u_class = edge_u_class[valid]
                else:
                    edge_u_class = edge_u_class[:edge_count]
                if edge_v_class.shape[0] == valid.shape[0]:
                    edge_v_class = edge_v_class[valid]
                else:
                    edge_v_class = edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )


def _sanitize_graph_obj(g, feat_dim=4):
    if g is None:
        return None

    if isinstance(g, PyGBatch):
        # batch -> list[Data]
        return [x for x in g.to_data_list()]

    if isinstance(g, dict):
        g = dict_to_pyg_data(g, feat_dim=feat_dim)

    if isinstance(g, list):
        out = []
        for x in g:
            sx = _sanitize_graph_obj(x, feat_dim=feat_dim)
            if sx is None:
                continue
            if isinstance(sx, list):
                out.extend(sx)
            else:
                out.append(sx)
        return out

    if not isinstance(g, (Data, HeteroData)):
        return g

    # normalize tensors
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

    node_class = getattr(g, "node_class", None)
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

    edge_attr = getattr(g, "edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(0)

    edge_label = getattr(g, "edge_label", None)
    if edge_label is None:
        edge_label = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_label):
        edge_label = torch.tensor(edge_label, dtype=torch.long)
    else:
        edge_label = edge_label.long()
    edge_label = edge_label.view(-1)

    edge_u_class = getattr(g, "edge_u_class", None)
    if edge_u_class is None:
        edge_u_class = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_u_class):
        edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
    else:
        edge_u_class = edge_u_class.long()
    edge_u_class = edge_u_class.view(-1)

    edge_v_class = getattr(g, "edge_v_class", None)
    if edge_v_class is None:
        edge_v_class = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_v_class):
        edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
    else:
        edge_v_class = edge_v_class.long()
    edge_v_class = edge_v_class.view(-1)

    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        if not valid.all():
            edge_index = edge_index[:, valid]
            edge_count = edge_index.shape[1]
            if edge_attr.shape[0] == valid.shape[0]:
                edge_attr = edge_attr[valid]
            else:
                edge_attr = edge_attr[:edge_count]
            if edge_label.shape[0] == valid.shape[0]:
                edge_label = edge_label[valid]
            else:
                edge_label = edge_label[:edge_count]
            if edge_u_class.shape[0] == valid.shape[0]:
                edge_u_class = edge_u_class[valid]
            else:
                edge_u_class = edge_u_class[:edge_count]
            if edge_v_class.shape[0] == valid.shape[0]:
                edge_v_class = edge_v_class[valid]
            else:
                edge_v_class = edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )


def _ensure_graph_list(graphs, feat_dim=4):
    flat = []
    for g in _graph_to_list(graphs):
        sg = _sanitize_graph_obj(g, feat_dim=feat_dim)
        if sg is None:
            continue
        if isinstance(sg, list):
            for x in sg:
                x = _ensure_nonempty(_sanitize_graph_obj(x, feat_dim=feat_dim), feat_dim=feat_dim)
                flat.append(x)
        else:
            flat.append(_ensure_nonempty(sg, feat_dim=feat_dim))
    return flat


def _collate_graph_objects(graphs, feat_dim=4):
    graphs = _ensure_graph_list(graphs, feat_dim=feat_dim)
    if len(graphs) == 0:
        return None
    if isinstance(graphs[0], (Data, HeteroData)):
        return PyGBatch.from_data_list(graphs)
    if torch.is_tensor(graphs[0]):
        if all(torch.is_tensor(g) and g.shape == graphs[0].shape for g in graphs):
            return torch.stack(graphs, dim=0)
    return graphs


# ---------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------

def _collate_samples(samples, feat_dim=4):
    """
    Collate list of sample dicts into a batch dict.
    """
    out = {}

    for meta_key in ["index", "scene", "pose", "img_path", "graph_path"]:
        values = [s.get(meta_key) for s in samples]
        if any(v is not None for v in values):
            out[meta_key] = values

    image_values = [s.get("image") for s in samples if s.get("image") is not None]
    out["image"] = _stack_images(image_values) if image_values else None

    graph_values = [s.get("graph") for s in samples if s.get("graph") is not None]
    out["graph"] = _collate_graph_objects(graph_values, feat_dim=feat_dim) if graph_values else None

    return out


def collate_fn(batch, feat_dim=4):
    """
    Unified collate:
      - inference/cache mode: batch = list[dict]
      - training mode: batch = list[(sample_dict, local_idx, global_idx)]
    """
    first = batch[0]

    if isinstance(first, dict):
        return _collate_samples(batch, feat_dim=feat_dim)

    samples, triplets_local_indexes, triplets_global_indexes = zip(*batch)
    batch_samples = _collate_samples(list(samples), feat_dim=feat_dim)

    triplets_local_indexes = torch.cat([x[None] for x in triplets_local_indexes], dim=0)
    triplets_global_indexes = torch.cat([x[None] for x in triplets_global_indexes], dim=0)

    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i

    return batch_samples, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


# ---------------------------------------------------------------------
# Dataset for PCA / feature extraction
# ---------------------------------------------------------------------

class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))

    def __getitem__(self, index):
        return base_transform(path_to_pil_img(self.images_paths[index]))

    def __len__(self):
        return len(self.images_paths)


# ---------------------------------------------------------------------
# Loader / preprocessor
# ---------------------------------------------------------------------

class SampleLoader:
    """
    Класс, который выполняет подготавливает данные для TripletDataset: 
      - читает image
      - читает graph
      - image поворачивает на 90° по часовой
      - для query применяются аугментации
      - делается resize
      - graph x поворачивается на 90° по часовой
    """

    def __init__(self, args, use_images=True, use_graphs=True):
        self.args = args
        self.use_images = use_images
        self.use_graphs = use_graphs

        self.resize = args.resize
        self.graph_rotate = getattr(args, "graph_rotate", True)
        self.feat_dim = args.in_dim_graph

        self.image_mean = DEFAULT_MEAN
        self.image_std = DEFAULT_STD

        self.query_aug = self._build_query_augmentations()

    def _rotate_pil_clockwise_90(self, img):
        return img.transpose(Image.Transpose.ROTATE_270)

    def _build_query_augmentations(self):
        ops = []

        if getattr(self.args, "brightness", None) is not None:
            ops.append(transforms.ColorJitter(brightness=self.args.brightness))
        if getattr(self.args, "contrast", None) is not None:
            ops.append(transforms.ColorJitter(contrast=self.args.contrast))
        if getattr(self.args, "saturation", None) is not None:
            ops.append(transforms.ColorJitter(saturation=self.args.saturation))
        if getattr(self.args, "hue", None) is not None:
            ops.append(transforms.ColorJitter(hue=self.args.hue))

        return transforms.Compose(ops) if ops else None

    def _final_image_transform(self, img):
        """
        Финальный preprocessing после rotate/augment:
          resize -> ToTensor -> Normalize
        """
        if self.resize is not None:
            img = transforms.functional.resize(img, self.resize)
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=self.image_mean, std=self.image_std)
        return img

    def _load_pil_image(self, img_path):
        return path_to_pil_img(img_path)


    def rotate_graph_features(self, graph):
        """
        Поворот graph['x'] на 90° по часовой стрелке.
        [1 - y, x, h, w]
        """
        if graph is None or not hasattr(graph, "x") or graph.x is None:
            return graph

        x = graph.x
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.ndim != 2 or x.shape[1] < 2:
            graph.x = x
            return graph

        new_x = x.clone()

        if x.shape[1] >= 4:
            new_x[:, 0] = 1.0 - x[:, 1]
            new_x[:, 1] = x[:, 0]
            new_x[:, 2] = x[:, 3]
            new_x[:, 3] = x[:, 2]
        else:
            new_x[:, 0] = 1.0 - x[:, 1]
            new_x[:, 1] = x[:, 0]

        graph.x = new_x
        return graph

    def load_image(self, pil_img, is_query=False):
        if pil_img is None:
            return None

        img = self._rotate_pil_clockwise_90(pil_img)

        if is_query and self.query_aug is not None:
            img = self.query_aug(img)

        img = self._final_image_transform(img)
        return img

    def load_graph(self, graph_path, original_size=None):
        if graph_path is None:
            return None

        graph = torch.load(graph_path, map_location="cpu")
        graph = _sanitize_graph_obj(graph, feat_dim=self.feat_dim)

        if isinstance(graph, list):
            out = []
            for g in graph:
                g = _ensure_nonempty(g, feat_dim=self.feat_dim)
                if self.graph_rotate:
                    g = self.rotate_graph_features(g)
                out.append(g)
            return out

        graph = _ensure_nonempty(graph, feat_dim=self.feat_dim)

        if self.graph_rotate:
            graph = self.rotate_graph_features(graph)

        return graph

    def load(self, item, is_query=False, with_meta=True):
        sample = {}

        img_path = item.get("img")
        graph_path = item.get("graph")

        pil_img = None
        original_size = None

        if self.use_images and img_path is not None:
            pil_img = self._load_pil_image(img_path)
            original_size = pil_img.size  # (W, H)

        if with_meta:
            sample["scene"] = item.get("scene")
            sample["pose"] = item.get("pose")
            sample["img_path"] = img_path
            sample["graph_path"] = graph_path

        if self.use_images:
            sample["image"] = self.load_image(pil_img, is_query=is_query)
        else:
            sample["image"] = None

        if self.use_graphs:
            sample["graph"] = self.load_graph(graph_path, original_size=original_size)
        else:
            sample["graph"] = None

        return sample

# ---------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------

class BaseDataset(data.Dataset):
    """
    Dataset with database + query items.

    Responsibility:
      - discover scenes
      - build aligned raw item lists
      - store mappings scan->ref
      - build soft positives

    Important:
      BaseDataset.__getitem__ returns RAW item dict.
      Use .load_sample(item, is_query=...) to get loaded sample.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="3RScan", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.dataset_folder = join(datasets_folder, dataset_name)

        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.modalities = tuple(getattr(args, "modalities", ["image", "graph", "pose"]))
        self.use_images = "image" in self.modalities
        self.use_graphs = "graph" in self.modalities
        self.use_pose = "pose" in self.modalities
        self.resize = getattr(args, "resize", None)

        self.database_items = []
        self.queries_items = []
        self.items = []

        # scene metadata
        meta_data = join(self.dataset_folder, "files")
        scene_data_path = join(meta_data, "3RScan_small.json")
        scene_data = load_json(scene_data_path) or []

        # mappings
        self.transforms = {}
        self.scan_to_ref = {}
        self.ref_to_scans = {}

        for entry in scene_data:
            ref = entry["reference"]
            self.scan_to_ref[ref] = ref
            self.ref_to_scans[ref] = []
            self.transforms[ref] = np.eye(4)

            for scan in entry.get("scans", []):
                scan_name = scan["reference"]
                self.scan_to_ref[scan_name] = ref
                self.ref_to_scans[ref].append(scan_name)
                if "transform" in scan:
                    mat = np.array(scan["transform"]).reshape(4, 4)
                else:
                    mat = np.eye(4)
                self.transforms[scan_name] = mat

        print("SCAN LEN", len(self.scan_to_ref))

        # loader/preprocessor
        self.loader = SampleLoader(args, use_images=self.use_images, use_graphs=self.use_graphs)

        # split files
        if dataset_name == "3RScan":
            if split == "train":
                split_scans_path = join(meta_data, "train_scans_small.txt")
            elif split == "test":
                split_scans_path = join(meta_data, "test_resplit_scans_small.txt")
            else:
                split_scans_path = join(meta_data, f"{split}_scans.txt")

            with open(split_scans_path, "r", encoding="utf-8") as f:
                scene_names = [l.strip() for l in f if l.strip()]

            print(len(scene_names))

            for scene_name in scene_names:
                is_ref = (scene_name in self.scan_to_ref and self.scan_to_ref[scene_name] == scene_name)
                if is_ref:
                    db_items = self._build_items_for_scene(scene_name)
                    if db_items:
                        self.database_items.extend(db_items)
                    else:
                        print(f"[SKIP] database scene '{scene_name}' due to modality mismatch.")
                else:
                    print(f"[SKIP] {scene_name} is not a reference scene.")
                    continue

                for query_scene in self.ref_to_scans.get(scene_name, []):
                    q_items = self._build_items_for_scene(query_scene)
                    if q_items:
                        self.queries_items.extend(q_items)
                    else:
                        print(f"[SKIP] query scene '{query_scene}' due to modality mismatch.")

        self.database_num = len(self.database_items)
        self.queries_num = len(self.queries_items)
        self.items = self.database_items + self.queries_items

        self.soft_positives_per_query = self._build_soft_positives(radius=1.0)


    def _list_scene_files(self, scene_name):
        files = {}
        scene_graph_path = join(self.dataset_folder, "Splited_graphs", scene_name)
        scene_image_path = join(self.dataset_folder, "scenes", scene_name, "sequence")

        if "image" in self.modalities:
            files["images"] = sorted(
                [
                    p
                    for p in glob(join(scene_image_path, "**", "frame-*.color.jpg"), recursive=True)
                    if ".rendered." not in p
                ]
            )
        if "pose" in self.modalities:
            files["poses"] = sorted(glob(join(scene_image_path, "**", "*.pose.txt"), recursive=True))
        if "graph" in self.modalities:
            files["graphs"] = sorted(glob(join(scene_graph_path, "**", "*.pt"), recursive=True))

        return files

    def _basename_key(self, path):
        if path is None:
            return None
        return basename(path).split(".")[0]

    def _validate_scene_alignment(self, files, scene_name=None):
        """
        Returns True if the scene is usable, False if lengths are inconsistent.
        """
        present = [(k, len(files.get(k, []))) for k in ("images", "poses", "graphs") if k in files]
        if len(present) <= 1:
            return True

        lengths = {name: ln for name, ln in present}
        base_name = "images" if "images" in lengths else present[0][0]
        base_len = lengths[base_name]

        mismatches = [(name, ln) for name, ln in present if ln != base_len]
        if not mismatches:
            return True

        scene_info = f" (scene_id='{scene_name}')" if scene_name is not None else ""
        lines = [f"[MODALITY MISMATCH] Different file counts for scene{scene_info}:"]
        lines.append(f"  Reference modality: '{base_name}' -> {base_len} items.")
        for name, ln in present:
            lines.append(f"  - {name}: {ln}")

        for name, _ in present:
            lst = list(files.get(name, []))
            if not lst:
                continue
            sample = [basename(p) for p in (lst if len(lst) <= 6 else lst[:3] + ["..."] + lst[-3:])]
            lines.append(f"  samples {name}: {sample}")

        logging.warning("\n".join(lines))
        return False

    def _build_items_for_scene(self, scene_name):
        files = self._list_scene_files(scene_name)

        if not self._validate_scene_alignment(files, scene_name):
            return []

        images = files.get("images", [])
        poses = files.get("poses", [])
        graphs = files.get("graphs", [])

        if images:
            n = len(images)
        elif poses:
            n = len(poses)
        elif graphs:
            n = len(graphs)
        else:
            return []

        transform_mat = self.transforms.get(scene_name, np.eye(4))
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
                    camera_center = (transform_mat @ center_hom)[:3]
                except Exception as e:
                    logging.warning(f"Failed to read pose {poses[i]}: {e}")
                    camera_center = None

            items.append(
                {
                    "img": img_path,
                    "graph": graph_path,
                    "scene": scene_name,
                    "pose": camera_center,
                }
            )

        return items

    def load_sample(self, item, is_query=False, with_meta=True):
        return self.loader.load(item, is_query=is_query, with_meta=with_meta)

    def __len__(self):
        return len(self.items)

    def get_raw_item(self, index):
        return self.items[index]

    def __getitem__(self, index):
        item = self.items[index]
        is_query = index >= self.database_num
        return self.load_sample(item, is_query=is_query, with_meta=True)

    def _build_soft_positives(self, radius=5.0):
        db_by_ref = {}
        for idx, item in enumerate(self.database_items):
            ref = item["scene"]
            if ref not in db_by_ref:
                db_by_ref[ref] = {"indices": [], "centers": []}
            if item["pose"] is not None:
                db_by_ref[ref]["indices"].append(idx)
                db_by_ref[ref]["centers"].append(item["pose"])

        trees = {}
        for ref, data_ref in db_by_ref.items():
            if len(data_ref["centers"]) > 0:
                trees[ref] = {
                    "tree": KDTree(np.array(data_ref["centers"])),
                    "indices": data_ref["indices"],
                }

        soft_positives = []
        for q_item in self.queries_items:
            if q_item["pose"] is None:
                soft_positives.append([])
                continue

            ref = self.scan_to_ref.get(q_item["scene"], q_item["scene"])
            if ref not in trees:
                soft_positives.append([])
                continue

            tree_info = trees[ref]
            idxs_local = tree_info["tree"].query_ball_point(q_item["pose"], r=radius)
            global_idxs = [tree_info["indices"][i] for i in idxs_local]
            soft_positives.append(global_idxs)

        return soft_positives

    def read_pose_file(self, pose_path):
        try:
            arr = np.loadtxt(pose_path)
        except Exception:
            arr = np.genfromtxt(pose_path)
        arr = np.array(arr).flatten()
        if arr.size != 16:
            raise ValueError(f"File {pose_path} doesn't contain 16 numbers (found {arr.size}).")
        return arr.reshape(4, 4).astype(np.float64)

    def compute_camera_center_from_T(self, T, convention="cam2world"):
        if T.shape != (4, 4):
            raise ValueError("T must be 4x4")
        Rmat = T[:3, :3]
        t = T[:3, 3]
        if convention == "cam2world":
            return t.copy()
        if convention == "world2cam":
            return -Rmat.T.dot(t)
        raise ValueError("convention must be 'cam2world' or 'world2cam'")

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {len(self.database_items)}; #queries: {len(self.queries_items)} >"

    def get_positives(self):
        return self.soft_positives_per_query


# ---------------------------------------------------------------------
# Triplet dataset
# ---------------------------------------------------------------------

class TripletsDataset(BaseDataset):
    """
    Triplets dataset using BaseDataset raw items + SampleLoader.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="3RScan", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)

        self.mining = getattr(args, "mining", "full")
        self.neg_samples_num = getattr(args, "neg_samples_num", 1000)
        self.negs_num_per_query = negs_num_per_query
        self.is_inference = False

        self.soft_positives_radius = getattr(args, "soft_positives_radius", 1.0)
        self.train_positives_dist_threshold = getattr(args, "train_positives_dist_threshold", 10.0)

        self._filter_queries_without_soft_positives()
        self.soft_positives_per_query = self._build_soft_positives(radius=self.soft_positives_radius)

        self.triplets_global_indexes = torch.empty((0, 2 + self.negs_num_per_query), dtype=torch.long)
        self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)] if self.mining == "full" else None

        self.weights = None
        if self.mining == "msls_weighted":
            self._build_msls_weights()

    # -------- views / filtering --------

    def _filter_queries_without_soft_positives(self):
        keep = []
        for i, q_item in enumerate(self.queries_items):
            has_pose = q_item.get("pose") is not None
            has_positives = len(self.soft_positives_per_query[i]) > 0
            if has_pose and has_positives:
                keep.append(i)

        removed = len(self.queries_items) - len(keep)
        if removed > 0:
            logging.info(f"Removed {removed} queries without pose/soft positives.")

        self.queries_items = [self.queries_items[i] for i in keep]
        self.soft_positives_per_query = [np.asarray(self.soft_positives_per_query[i], dtype=np.int32) for i in keep]

        self.queries_num = len(self.queries_items)
        self.items = self.database_items + self.queries_items

    def _build_msls_weights(self):
        counts = np.array([max(1, len(p)) for p in self.soft_positives_per_query], dtype=np.float32)
        inv = 1.0 / counts
        inv /= inv.sum() if inv.sum() > 0 else 1.0
        self.weights = inv

    # -------- negatives --------

    def _complete_negative_indexes(self, query_index, num_negatives=None):
        if num_negatives is None:
            num_negatives = self.negs_num_per_query

        query_item = self.queries_items[query_index]
        query_scene = query_item["scene"]
        query_ref = self.scan_to_ref.get(query_scene, query_scene)

        other_refs = [ref for ref in self.ref_to_scans.keys() if ref != query_ref]
        if len(other_refs) == 0:
            raise RuntimeError(f"No foreign reference scenes found for query ref='{query_ref}'")

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
            neg_indexes = np.random.choice(candidate_db_indexes, size=num_negatives, replace=replace).astype(np.int32)
            return neg_indexes

        raise RuntimeError(
            f"Could not sample negatives for query_index={query_index} because no non-empty foreign reference scene was found."
        )

    # -------- dataset API --------

    def __len__(self):
        if self.is_inference:
            return len(self.items)
        return len(self.triplets_global_indexes)

    def __getitem__(self, index):
        if self.is_inference:
            item = self.items[index]
            return self.load_sample(item, is_query=(index >= self.database_num), with_meta=True)

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index],
            (1, 1, self.negs_num_per_query),
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
                self.load_sample(items[0], is_query=True, with_meta=False)["image"],
                self.load_sample(items[1], is_query=False, with_meta=False)["image"],
                *[self.load_sample(it, is_query=False, with_meta=False)["image"] for it in items[2:]],
            ]
            sample["image"] = torch.stack(images, dim=0)
        else:
            sample["image"] = None

        if self.use_graphs:
            graphs = [
                self.load_sample(items[0], is_query=True, with_meta=False)["graph"],
                self.load_sample(items[1], is_query=False, with_meta=False)["graph"],
                *[self.load_sample(it, is_query=False, with_meta=False)["graph"] for it in items[2:]],
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
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {len(self.database_items)}; #queries: {len(self.queries_items)} >"

    def get_positives(self):
        return self.soft_positives_per_query

    # -------- cache / model utils --------

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
    def _empty_cache(shape):
        return np.full(shape, np.nan, dtype=np.float32)

    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        subset_dl = DataLoader(
            dataset=subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=(args.device == "cuda"),
            collate_fn=collate_fn,
        )

        model = model.eval()

        cache = TripletsDataset._empty_cache(cache_shape)

        with torch.no_grad():
            cursor = 0
            subset_indices = getattr(subset_ds, "indices", None)

            for batch in subset_dl:
                batch = TripletsDataset._move_to_device(batch, args.device)
                model_input = batch
                if isinstance(batch, dict):
                    model_input = TripletsDataset._move_to_device(subset_ds.dataset._prepare_model_input(batch) if hasattr(subset_ds, "dataset") and hasattr(subset_ds.dataset, "_prepare_model_input") else batch, args.device)

                batch_graph = model_input["graph"].to(args.device)
                batch_image = model_input["image"].to(args.device)

                outputs = model(
                    graph=batch_graph,
                    image=batch_image,
                    mode=args.mode,   # "graph" / "fusion"
                    return_parts=True,
                )

                global_features = outputs["fused"]

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
        if query_features is None or np.isnan(query_features).all():
            raise RuntimeError(
                f"For query {self.queries_items[query_index].get('img', 'N/A')} with index {query_index} features have not been computed."
            )
        return query_features

    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_indexes = self.soft_positives_per_query[query_index]
        if len(positives_indexes) == 0:
            raise RuntimeError(f"No positives available for query {query_index}")

        positives_features = cache[positives_indexes]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features.astype(np.float32))
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1).astype(np.float32), 1)
        best_positive_index = positives_indexes[best_positive_num[0][0]]
        return int(best_positive_index)

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_samples = np.asarray(neg_samples, dtype=np.int32).reshape(-1)
        if len(neg_samples) == 0:
            return np.empty((0,), dtype=np.int32)

        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features.astype(np.float32))

        k = min(self.negs_num_per_query, len(neg_samples))
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1).astype(np.float32), k)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes

    # -------- triplet mining --------

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

        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
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

        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            neg_indexes = self._complete_negative_indexes(query_index)
            if self.neg_cache is not None:
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

        for query_index in sampled_queries_indexes:
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            neg_indexes = self._complete_negative_indexes(query_index)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)

