# Scene_graph_localization
# Dataset README

This document explains how the dataset pipeline is organized, how samples are discovered and loaded, how graphs and images are preprocessed, and how triplets are mined during training.

---

## 1. High-level overview

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

## 2. Dataset layout expected on disk

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

## 3. What a single raw item looks like

Each item in the dataset is stored as a Python dictionary:

```python
{
    "img": <path to image or None>,
    "graph": <path to graph or None>,
    "scene": <scene identifier>,
    "pose": <camera center in world coordinates or None>,
}
```

This is produced by:

```python
make_item(img_path, graph_path=None, scene=None, pose=None)
```

In practice, `BaseDataset` builds these items from file lists on disk.

---

## 4. Modalities

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

## 5. `BaseDataset`: raw dataset construction

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

### Key attributes

After initialization you get:

- `self.database_items`
- `self.queries_items`
- `self.items`
- `self.database_num`
- `self.queries_num`
- `self.scan_to_ref`
- `self.ref_to_scans`
- `self.transforms`
- `self.soft_positives_per_query`

---

## 6. How scenes are grouped

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

## 7. How the split is chosen

For `dataset_name == "3RScan"` the split file is chosen like this:

- `train` → `files/train_scans_small.txt`
- `test` → `files/test_resplit_scans_small.txt`
- otherwise → `files/{split}_scans.txt`

The file contains scene names. For each scene:

- if it is a **reference scene**, it is added to `database_items`
- its associated scans are added to `queries_items`

If a scene does not match the expected modality alignment rules, it is skipped.

---

## 8. File discovery for each scene

The helper `_list_scene_files(scene_name)` searches:

### Images
```python
datasets/3RScan/scenes/<scene_name>/sequence/**/frame-*.color.jpg
```

### Poses
```python
datasets/3RScan/scenes/<scene_name>/sequence/**/*.pose.txt
```

### Graphs
```python
datasets/3RScan/Splited_graphs/<scene_name>/**/*.pt
```

The images exclude rendered frames:

```python
if ".rendered." not in p
```

This keeps only the intended camera images.

---

## 9. Alignment check across modalities

Before a scene is accepted, `_validate_scene_alignment()` checks whether the number of files is consistent across modalities.

For example, if a scene has:

- 120 images
- 120 poses
- 119 graphs

then it is considered inconsistent and skipped.

This is important because the dataset assumes a 1-to-1 correspondence between:
- image frame,
- pose file,
- graph file.

### Why this matters

If counts do not match, any attempt to align an image with the wrong graph or pose would corrupt training and evaluation.

The code logs a detailed warning, including sample file names.

---

## 10. How raw items are built

`_build_items_for_scene(scene_name)` creates one item per frame index.

For each `i`:

- `img_path = images[i]` if available
- `graph_path = graphs[i]` if available
- `pose` is computed from the pose file if present

### Pose computation

The code reads a `4x4` pose matrix from the `.pose.txt` file:

```python
pose_raw = self.read_pose_file(poses[i])
```

Then computes the camera center:

```python
center_local = self.compute_camera_center_from_T(pose_raw)
```

If the scene has a transform to reference space, the center is transformed:

```python
camera_center = (transform_mat @ center_hom)[:3]
```

The final item stores that `camera_center` in the `pose` field.

---

## 11. What `pose` means here

In this dataset, `pose` is used as a **3D camera center** in world/reference coordinates, not as a full 4x4 pose matrix.

It is used to compute spatial distance between query and database items.

This is the basis of **soft positive mining**.

---

## 12. `SampleLoader`: loading and preprocessing

`SampleLoader` is the only place where actual file loading and preprocessing happen.

It is responsible for:

- loading the image from disk,
- loading the graph from disk,
- applying query-only augmentations,
- resizing and normalizing the image,
- rotating image and graph features,
- ensuring the graph is valid even if empty.

---

## 13. Image preprocessing

### Image loading
The image is opened with PIL:

```python
Image.open(path).convert("RGB")
```

### Clockwise rotation
Every image is rotated by **90° clockwise**:

```python
img.transpose(Image.Transpose.ROTATE_270)
```

This is important because your graph coordinates are also rotated to match image orientation.

### Query-only photometric augmentation
For query images, optional augmentations may be applied:

- brightness
- contrast
- saturation
- hue

These are built from `torchvision.transforms.ColorJitter`.

### Final image transform
After rotation and augmentation:

1. optional resize,
2. conversion to tensor,
3. normalization with dataset mean/std.

The default statistics are:

```python
DEFAULT_MEAN = [0.44420420130352495, 0.41322746532289134, 0.3678658064565412]
DEFAULT_STD  = [0.24352604373543688, 0.24045797651069503, 0.24250136992133814]
```

---

## 14. Graph preprocessing

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

## 15. Why graph sanitization is needed

Graph files can often be irregular:
- missing fields,
- empty node lists,
- inconsistent edge lengths,
- bad indices after preprocessing.

The sanitization functions ensure the dataset never crashes on such samples and always returns a usable graph object.

This is especially important when training with batching.

---

## 16. Graph rotation

The loader rotates graph coordinates to stay aligned with the rotated image.

For nodes with at least 4 dimensions in `x`, the code applies:

```python
new_x[:, 0] = 1.0 - x[:, 1]
new_x[:, 1] = x[:, 0]
new_x[:, 2] = x[:, 3]
new_x[:, 3] = x[:, 2]
```

This corresponds to a clockwise 90° rotation in normalized coordinates.

If the graph only has 2 coordinates, the fallback rotation is:

```python
new_x[:, 0] = 1.0 - x[:, 1]
new_x[:, 1] = x[:, 0]
```

### Important assumption

This rotation is correct only if the first two coordinates in `x` are normalized to `[0, 1]`.

---

## 17. `_ensure_nonempty()`

This helper guarantees that a graph is never empty.

If the graph is missing or has no nodes, it creates a synthetic one-node graph with:
- zero features,
- empty edge tensors,
- zero node_class.

This prevents downstream code from failing on empty graphs.

---

## 18. `BaseDataset` item structure

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

## 19. `BaseDataset.__getitem__`

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

## 20. `BaseDataset.get_raw_item()`

This method returns the raw item dictionary before loading:

```python
dataset.get_raw_item(index)
```

Use this when you need:
- original file paths,
- scene labels,
- raw poses,
- visualization without preprocessing.

This is the safest choice for plotting or debugging retrieval.

---

## 21. Soft positives

Soft positives are built from spatial proximity in reference space.

### Idea

For each query:
- find database items from the same reference scene,
- measure Euclidean distance between query pose and database pose,
- keep database items within radius `r`.

This is implemented with `scipy.spatial.KDTree`.

### Result

`self.soft_positives_per_query` is a list of arrays:

```python
soft_positives_per_query[q_idx] = [db_idx_1, db_idx_2, ...]
```

Each entry contains database indices considered positive for that query.

---

## 22. Why soft positives are called “soft”

They are not a single exact match.

Instead, the query may have several valid positives:
- nearby viewpoints,
- slightly different camera positions,
- same scene structure.

This is useful for retrieval, where strict one-to-one matching is often too harsh.

---

## 23. `TripletsDataset`: training view

`TripletsDataset` extends `BaseDataset` and is designed for triplet training.

Instead of returning raw loaded samples, it returns:

- one query,
- one selected positive,
- several negatives.

The triplets are regenerated periodically using the current model embeddings.

---

## 24. Triplet dataset initialization

During `TripletsDataset.__init__()` the following happens:

1. `BaseDataset` is initialized.
2. mining mode is set:
   - `"full"`
   - `"partial"`
   - `"msls_weighted"`
   - `"random"`
3. queries without valid soft positives are removed.
4. soft positives are recomputed.
5. storage for mined triplets is initialized:
   ```python
   self.triplets_global_indexes
   ```
6. optional negative cache and sampling weights are prepared.

---

## 25. Removing unusable queries

`_filter_queries_without_soft_positives()` removes queries that have:
- no pose,
- or no soft positives.

This is important because such queries cannot participate in mining.

After filtering:
- `self.queries_items` is shortened,
- `self.soft_positives_per_query` is filtered accordingly,
- `self.queries_num` is updated,
- `self.items` is rebuilt.

---

## 26. Weighting queries for MSLS-style mining

If `mining == "msls_weighted"`, the code computes weights inversely proportional to the number of positives per query.

Queries with fewer positives get higher sampling probability.

This encourages the training loop to focus more on harder / rarer queries.

---

## 27. Negative sampling logic

`_complete_negative_indexes(query_index)` samples negatives from **foreign reference scenes** only.

### Why this is important

Negatives should not come from the same reference scene, otherwise they may be false negatives.

### How it works

For the query:
1. determine its reference scene,
2. collect all other reference scenes,
3. sample one foreign reference scene,
4. sample database items from that scene.

If there are not enough candidates, sampling may be done with replacement.

---

## 28. Training item returned by `TripletsDataset.__getitem__`

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

## 29. Why `collate_fn` has two modes

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

---

## 31. Feature cache in triplet mining

The triplet mining loop uses a **feature cache**.

### Why cache exists

The model is expensive to run.  
To mine good triplets, the code needs embeddings for many items.

Instead of recomputing embeddings for every query-positive-negative comparison, it:

1. runs the model once on a subset,
2. stores embeddings in `cache`,
3. mines positives/negatives based on the cached embeddings.

### Cache contents

The cache is a NumPy array:

```python
cache.shape == (len(self.items), args.features_dim)
```

Uncomputed locations are filled with `NaN`.

---

## 32. What `compute_cache()` does

`compute_cache(args, model, subset_ds, cache_shape)`:

1. creates a DataLoader over `subset_ds`,
2. sends batches to device,
3. runs the model in eval mode with `torch.no_grad()`,
4. extracts `outputs["fused"]`,
5. writes these embeddings into the correct positions of the cache.

This is the backbone of mining.

---

## 33. Why the cache is recomputed periodically

The model changes during training, so the embeddings become stale.

That is why the triplet mining loop periodically recomputes the cache:

- `cache_refresh_rate` determines how often to update the mined triplets.

This keeps mining aligned with the current model state.

---

## 34. Triplet mining modes

The code supports several mining strategies.

### `full`
Uses all database items and sampled queries.

### `partial`
Uses only a subset of database items and sampled queries.

### `msls_weighted`
Same as partial, but query sampling is weighted by the number of positives.

### `random`
Uses random positives and random negatives without a hard-mining style selection.

---

## 35. `compute_triplets_full()`

This mode:

1. samples a subset of queries,
2. builds a cache over:
   - all database items,
   - sampled queries,
3. finds the best positive for each query,
4. samples negatives from foreign scenes,
5. stores the mined triplets.

This is the most exhaustive mode.

---

## 36. `compute_triplets_partial()`

This mode is cheaper.

It:
1. samples queries,
2. samples a subset of database items for the cache,
3. includes all positives for the sampled queries,
4. builds a cache only over the selected subset,
5. mines positives/negatives from that subset.

This reduces computation and memory usage.

---

## 37. `compute_triplets_random()`

This mode is the most lightweight.

It:
1. samples queries,
2. collects their positives,
3. builds a cache over that subset,
4. samples negatives randomly from foreign scenes.

This is useful when you want a simpler or faster baseline.

---

## 38. Positive selection

For each query, the code chooses the “best positive” among all soft positives by nearest embedding distance:

```python
faiss.IndexFlatL2(args.features_dim)
```

The positive embedding closest to the query embedding becomes the selected positive.

This creates a harder and more informative training pair.

---

## 39. Negative selection

Negatives are sampled from other scenes first, then optionally mined using embeddings.

In the current code path, `_complete_negative_indexes()` already ensures negatives come from a foreign reference scene.

If you want harder negatives, `get_hardest_negatives_indexes()` can rank them by embedding distance.

---

## 40. Retrieval/evaluation split

For testing and retrieval evaluation, the code treats:

- `database_items` as the gallery
- `queries_items` as the query set

The split is:

```python
db_indices = range(database_num)
q_indices = range(database_num, len(items))
```

This is the standard retrieval setup.

---

## 41. How evaluation embeddings are produced

For evaluation, the dataset should be in inference mode or use `BaseDataset`, so that each item yields a regular sample dict.

The model is then called with:

```python
model(graph=graph, image=image, mode=args.mode, return_parts=True)
```

The evaluation uses `outputs["fused"]`.

---

## 42. Retrieval metric logic

The code computes Recall@K as follows:

1. Embed all database items.
2. Embed all query items.
3. Normalize embeddings.
4. Search nearest database items using FAISS inner product.
5. Convert scene names to reference IDs using `scan_to_ref`.
6. Check whether at least one of the top-K retrieved database items belongs to the same reference scene as the query.

### This means

The evaluation metric is **scene/reference-level recall**, not exact item-level match.

That is consistent with localization-style retrieval evaluation.

---

## 43. Why `scan_to_ref` is used in evaluation

Different scans can belong to the same reference scene.

So a retrieval may still be correct even if the raw scene names differ, as long as they map to the same reference scene.

This is why evaluation compares:

```python
dataset.scan_to_ref.get(scene_name, scene_name)
```

rather than raw scene names only.

---

## 44. Important note about query rotations

Both image and graph are rotated by 90° clockwise in the loader.

This means:
- your model sees consistently oriented data,
- graph coordinates stay aligned with rotated image frames.

If you visualize raw files, remember that the dataset pipeline may have already rotated them.

---

## 45. Important note about graph input shape

The graph encoder in your project expects:
- `x`
- `edge_index`
- possibly `node_class`

The dataset converts dictionaries or loaded tensors into `torch_geometric.data.Data` objects.

If the graph is already a `Data`, it is sanitized and repaired.

---

## 46. Important note about empty graphs

Empty graphs are replaced by a one-node synthetic graph.

This avoids downstream crashes in:
- batching,
- GCN layers,
- pooling.

It also keeps the shape of the pipeline stable.

---

## 47. What `collate_fn` does with graphs

During batching, graphs are collected into a `torch_geometric.data.Batch` using:

```python
PyGBatch.from_data_list(graphs)
```

This lets the GNN process multiple scene graphs in one forward pass.

---

## 48. What `collate_fn` does with images

Images are stacked into a tensor of shape:

```python
[B, C, H, W]
```

If the tensors already have an extra batch dimension, they are concatenated appropriately.

---

## 49. What `collate_fn` does with metadata

For metadata fields like:
- `scene`
- `pose`
- `img_path`
- `graph_path`

the collate function keeps them as lists.

This is useful for:
- visualization,
- debugging,
- evaluation,
- mining.

---

## 50. Full data flow summary

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

---

## 51. Practical notes for using this dataset

### If you train
Use `TripletsDataset`.

### If you evaluate
Use:
- `BaseDataset`, or
- `TripletsDataset` with `is_inference = True`.

### If you visualize
Use raw items from:
```python
dataset.get_raw_item(index)
```

### If you need camera-position positives
Use `pose` and `soft_positives_per_query`.

---

## 52. Common pitfalls

### 1. Graph/image misalignment
If file counts do not match, the scene is skipped.

### 2. Wrong mode during evaluation
If `args.mode` does not match the model inputs, the forward pass can fail.

### 3. Forgetting `is_inference`
If `TripletsDataset` is not switched to inference mode, it will return triplets instead of raw samples.

### 4. Using stale cache
Triplet mining cache should be recomputed periodically, otherwise mined pairs become outdated.

### 5. Interpreting FAISS output
With `IndexFlatIP` on normalized vectors, FAISS returns similarity scores, not distances.

---

## 53. Suggested README structure for the repository

If you place this in the repository root, a good final structure is:

```text
README.md
datasets_ws.py
graph_model.py
train.py
test.py
visualization.py
```

With the following responsibilities:

- `datasets_ws.py` — all dataset logic
- `graph_model.py` — graph and multimodal encoders
- `train.py` — training loop and loss
- `test.py` — retrieval evaluation
- `visualization.py` — qualitative retrieval plots

---

## 54. Minimal usage examples

### Build a training dataset

```python
triplets_ds = TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
```

### Build a test dataset

```python
test_ds = BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
```

### Get a loaded sample

```python
sample = test_ds[0]
```

### Get a raw item

```python
raw_item = test_ds.get_raw_item(0)
```

### Get positives for a query

```python
positives = test_ds.get_positives()
```

---

## 55. Summary

This dataset pipeline is built for multimodal retrieval with graph and image inputs.  
It separates concerns cleanly:

- `BaseDataset` discovers and organizes data,
- `SampleLoader` preprocesses samples,
- `TripletsDataset` mines triplets for training,
- `collate_fn` supports both inference and triplet batches,
- feature caching makes mining efficient,
- soft positives and reference-scene mappings provide a robust retrieval target.

The design is flexible, but it depends on correct alignment of files, consistent scene metadata, and periodic cache recomputation.

