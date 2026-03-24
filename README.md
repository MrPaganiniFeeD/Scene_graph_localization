# Scene_graph_localization
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MrPaganiniFeeD/Scene_graph_localization.git
cd Scene_graph_localization

```

### 2. Create and activate virtual environment


```bash
python3 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
```

### 3. Install dependencies


```bash
pip install -r requirements.txt
```

## Data Preparation

The code expects a dataset structured as follows (example for `3RScan`):

```text
datasets_folder/
└── 3RScan/
    ├── files/
    │   ├── 3RScan_small.json          # scene metadata (reference, scans, transformations)
    │   ├── train_scans_small.txt      # list of scene names for training
    │   ├── test_resplit_scans_small.txt
    │   └── ...
    ├── scenes/                         # image data
    │   └── <scene_name>/
    │       └── sequence/
    │           ├── frame-000000.color.jpg
    │           ├── frame-000000.pose.txt  # 4x4 camera pose matrix
    │           └── ...
    └── Splited_graphs/                 # precomputed scene graphs (.pt files)
        └── <scene_name>/
            ├── frame-000000.pt
            └── ...
```

- **`3RScan_small.json`** must contain at least:
    
    - `"reference"`: reference scene name.
        
    - `"scans"`: list of scans belonging to that reference, each with `"reference"` and optionally `"transform"` (4x4 matrix).
        
- **Graph files** are expected to be serialized `torch_geometric.data.Data` objects (or dictionaries) with keys: `x`, `edge_index`, `edge_attr`, `node_class` etc. If they are dictionaries, the code converts them automatically.
    
- **Pose files** are text files with 16 numbers (row-major 4x4 matrix).
    

### Training

Run `train.py`:

```bash
python train.py \
    --datasets_folder /path/to/datasets \
    --dataset_name 3RScan \
    --mode image \
    --train_batch_size 4 \
    --epochs_num 20 \
    --lr 1e-5 \
    --mining partial \
    --cache_refresh_rate 500 \
    --queries_per_epoch 2000 \
    --save_dir my_experiment
```

### Evaluation

Run `eval.py`:

```bash
python eval.py \
    --datasets_folder /path/to/datasets \
    --dataset_name 3RScan \
    --mode fusion \
    --resume /path/to/best_model.pth \
    --save_dir eval_results
```


The script extracts embeddings for database and queries, builds a FAISS index, computes recall@k (default k = 1,5,10,20) and prints/saves the results.

# Dataset

This document explains how the dataset pipeline is organized, how samples are discovered and loaded, how graphs and images are preprocessed, and how triplets are mined during training.

---

## High-level overview

The dataset code is built around **three layers**:

1. **`BaseDataset`**  
   Responsible for discovering the raw dataset structure, building aligned database/query item lists, and computing soft positives.

2. **`SampleLoader`**  
   Responsible for loading and preprocessing a single sample:
   - reading the image,
   - reading the graph,
   - applying image normalization and query augmentations,
   - rotating image and graph features by 90° clockwise,
   - returning a unified sample dictionary.

3. **`TripletsDataset`**  
   Extends `BaseDataset` and turns the raw database/query samples into triplets for training:
   - query,
   - best positive,
   - negatives,
   with periodic feature caching and triplet mining.

The code is designed for **multimodal visual place recognition** with:
- **images**,
- **scene graphs**,
- **pose information** for spatial positive mining.

---

## Dataset layout expected on disk

The code assumes a dataset folder like:

```text
datasets/
└── 3RScan/
    ├── files/
    │   ├── 3RScan_small.json
    │   ├── train_scans_small.txt
    │   ├── test_resplit_scans_small.txt
    │   └── ...
    ├── scenes/
    │   └── <scene_name>/
    │       └── sequence/
    │           ├── frame-xxxx.color.jpg
    │           ├── frame-xxxx.pose.txt
    │           └── ...
    └── Splited_graphs/
        └── <scene_name>/
            ├── *.pt
            └── ...
```

### Main metadata file

`files/3RScan_small.json` contains scene grouping information:
- one **reference** scene,
- multiple associated **scan** scenes,
- optional rigid transforms from scan to reference.

This file is used to build:
- `scan_to_ref`
- `ref_to_scans`
- `transforms`

These mappings are central to the whole pipeline.

---

## What a single raw item looks like

Each item in the dataset is stored as a Python dictionary:

```python
{
    "img": <path to image or None>,
    "graph": <path to graph or None>,
    "scene": <scene identifier>,
    "pose": <camera center in world coordinates or None>,
}
```

In practice, `BaseDataset` builds these items from file lists on disk.

---

## Modalities

The dataset supports three modalities:

- `image`
- `graph`
- `pose`

They are controlled by:

```python
self.modalities = tuple(getattr(args, "modalities", ["image", "graph", "pose"]))
self.use_images = "image" in self.modalities
self.use_graphs = "graph" in self.modalities
self.use_pose = "pose" in self.modalities
```

### Important consequence

The dataset can work in several modes:

- **image only**
- **graph only**
- **image + graph**
- **image + graph + pose**

Pose is not directly fed to the model, but it is used to build positives.

---

## `BaseDataset`: raw dataset construction

`BaseDataset` is the foundation of the whole pipeline.

### Responsibilities

It:
- reads metadata,
- builds scene mappings,
- discovers files per scene,
- checks alignment between modalities,
- constructs `database_items`,
- constructs `queries_items`,
- concatenates them into `items`,
- computes soft positives.


## How scenes are grouped

The dataset uses `3RScan_small.json` to map scans to reference scenes.

For each entry:

- `entry["reference"]` is treated as the **reference scene**
- `entry["scans"]` contains scan variants

This creates:

### `scan_to_ref`
Maps any scene to its reference scene:

```python
scan_to_ref[scan_name] = ref
```

For a reference scene itself:

```python
scan_to_ref[ref] = ref
```

### `ref_to_scans`
Maps a reference scene to the list of scan scenes associated with it:

```python
ref_to_scans[ref] = [scan1, scan2, ...]
```

### `transforms`
Stores optional 4×4 transforms from scan to reference:

```python
self.transforms[scan_name] = mat
```

If no transform is provided, identity is used.

---

## How the split is chosen

For `dataset_name == "3RScan"` the split file is chosen like this:

- `train` → `files/train_scans_small.txt`
- `test` → `files/test_resplit_scans_small.txt`
- otherwise → `files/{split}_scans.txt`

The file contains scene names. For each scene:

- if it is a **reference scene**, it is added to `database_items`
- its associated scans are added to `queries_items`

If a scene does not match the expected modality alignment rules, it is skipped.

---

## `SampleLoader`: loading and preprocessing

`SampleLoader` is the only place where actual file loading and preprocessing happen.

It is responsible for:

- loading the image from disk,
- loading the graph from disk,
- applying query-only augmentations,
- resizing and normalizing the image,
- rotating image and graph features,
- ensuring the graph is valid even if empty.

## Graph preprocessing

Graphs are loaded from `.pt` files with:

```python
torch.load(graph_path, map_location="cpu")
```

They are then sanitized and converted to a `torch_geometric.data.Data` object.

### Expected graph fields

The code accepts the following keys:

- `x`
- `edge_index`
- `edge_attr`
- `node_class`
- `edge_label`
- `edge_u_class`
- `edge_v_class`

### Node features `x`

`x` is expected to contain node features of shape:

```python
[num_nodes, feat_dim]
```

If missing or empty, the code creates a synthetic node:

```python
torch.zeros((1, feat_dim))
```

### Edges

`edge_index` is expected to have shape:

```python
[2, num_edges]
```

Invalid edge indices are filtered out.

### Edge attributes

If present, `edge_attr` is kept and aligned with the filtered edges.

### Node classes

If `node_class` is missing, it is filled with zeros.

If its length does not match the number of nodes, it is padded or truncated.

---
## `BaseDataset` item structure

After initialization:

```python
self.database_items
self.queries_items
self.items = self.database_items + self.queries_items
```

This means:

- indices `[0, database_num - 1]` are **database**
- indices `[database_num, len(items)-1]` are **queries**

This split is used everywhere in evaluation and triplet mining.

---

## `BaseDataset.__getitem__`

`BaseDataset.__getitem__(index)` returns a fully loaded sample dict.

It does:

```python
item = self.items[index]
is_query = index >= self.database_num
return self.load_sample(item, is_query=is_query, with_meta=True)
```

So each returned sample includes:

- `image` if enabled
- `graph` if enabled
- metadata:
  - `scene`
  - `pose`
  - `img_path`
  - `graph_path`

This is the inference/evaluation view of the data.

---

## `TripletsDataset`: training view

`TripletsDataset` extends `BaseDataset` and is designed for triplet training.

Instead of returning raw loaded samples, it returns:

- one query,
- one selected positive,
- several negatives.

The triplets are regenerated periodically using the current model embeddings.

## Training item returned by `TripletsDataset.__getitem__`

In training mode, one dataset item corresponds to one mined triplet set.

It returns:

```python
(sample, triplets_local_indexes, triplets_global_indexes)
```

### `sample`
Contains loaded data for:
- query
- positive
- negatives

### `triplets_local_indexes`
Local index triplets used inside the stacked batch.

### `triplets_global_indexes`
Global dataset indices of the selected query, positive, and negatives.

This is why the training `collate_fn` has a special branch for triplet batches.

---

## Why `collate_fn` has two modes

The collate function is unified for both inference and training.

### Inference mode
If each dataset item is a `dict`, it returns a simple batch dict.

### Training mode
If each dataset item is a tuple:
```python
(sample, triplets_local_indexes, triplets_global_indexes)
```
then it:
- collates all samples,
- stacks local triplet indices,
- stacks global triplet indices.

This allows the same dataset code to support both training and evaluation.

---

## 30. `is_inference`

`TripletsDataset` has a flag:

```python
self.is_inference = False
```

When set to `True`, `__getitem__` returns the same raw loaded sample format as `BaseDataset`, which is necessary for:
- feature extraction,
- caching,
- retrieval evaluation.

During triplet mining the code temporarily sets:

```python
self.is_inference = True
```

so the dataset can be iterated in inference mode.

## Full data flow summary

Here is the complete data flow:

### On initialization
1. Read metadata JSON.
2. Build scene mappings.
3. Discover file paths.
4. Align modalities.
5. Build raw items.
6. Compute soft positives.

### For inference / retrieval
1. `__getitem__` returns a sample dict.
2. `collate_fn` stacks images and graphs.
3. Model produces embeddings.
4. FAISS retrieves nearest database items.
5. Recall is computed.

### For training
1. Triplets are mined from cached embeddings.
2. `TripletsDataset.__getitem__` returns query/positive/negative samples.
3. `collate_fn` assembles triplet batches.
4. Model computes embeddings.
5. Triplet loss is optimized.

