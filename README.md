# Scene_graph_localization
Graph Place Recognition Encoder
<img width="2030" height="467" alt="GPRe_cropped 2" src="https://github.com/user-attachments/assets/5e27fd4b-c5c1-4bd4-bace-e93e121cc3f8" />
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


