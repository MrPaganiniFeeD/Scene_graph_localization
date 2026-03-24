import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image



def _load_img(path):
    if path is None or not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


def _room_id(dataset, scene_name):
    return dataset.scan_to_ref.get(scene_name, scene_name)


def _get_graph_x_from_sample(sample):
    g = sample.get("graph", None)
    if g is None:
        return None

    if isinstance(g, dict):
        x = g.get("x", None)
    else:
        x = getattr(g, "x", None)

    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 1:
        x = x.view(1, -1)

    if x.shape[-1] < 2:
        return None

    return x.detach().cpu().float()


def _get_image_for_display(sample, mean=None, std=None):
    img = sample.get("image", None)

    if img is None:
        img_path = sample.get("img_path", None)
        if img_path is None:
            return None
        pil_img = _load_img(img_path)
        if pil_img is None:
            return None
        pil_img = pil_img.transpose(Image.Transpose.ROTATE_270)  # 90° clockwise
        return pil_img

    if isinstance(img, Image.Image):
        return img

    if torch.is_tensor(img):
        x = img.detach().cpu()

        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.float()

            if mean is not None and std is not None:
                mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
                std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
                if mean_t.shape[0] == x.shape[0]:
                    x = x * std_t + mean_t

            x = x.clamp(0, 1)
            x = x.permute(1, 2, 0).numpy()
            return x

        return x.numpy()

    return img


def _show_sample(
    ax,
    sample,
    title="",
    max_boxes=None,
    coords_normalized=True,
    mean=None,
    std=None,
):
    img = _get_image_for_display(sample, mean=mean, std=std)

    if img is None:
        ax.set_title(title + "\n[no image]")
        ax.axis("off")
        return

    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if isinstance(img, Image.Image):
        W, H = img.size
    else:
        H, W = img.shape[:2]

    x = _get_graph_x_from_sample(sample)
    if x is None:
        return

    n = x.shape[0]
    if max_boxes is not None:
        n = min(n, max_boxes)

    for i in range(n):
        row = x[i].tolist()
        cx, cy = row[0], row[1]
        bw = row[2] if len(row) > 2 else None
        bh = row[3] if len(row) > 3 else None

        if coords_normalized:
            cx_px = cx * W
            cy_px = cy * H
        else:
            cx_px = cx
            cy_px = cy

        if bw is not None and bh is not None:
            if coords_normalized:
                bw_px = bw * W
                bh_px = bh * H
            else:
                bw_px = bw
                bh_px = bh

            x1 = cx_px - bw_px / 2.0
            y1 = cy_px - bh_px / 2.0

            rect = patches.Rectangle(
                (x1, y1),
                max(1.0, bw_px),
                max(1.0, bh_px),
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.plot(cx_px, cy_px, marker="o", markersize=4, color="red")
        ax.text(cx_px + 3, cy_px + 3, str(i), color="red", fontsize=8)


def _get_sample_via_getitem(dataset, global_index):
    old_mode = getattr(dataset, "is_inference", None)
    if old_mode is not None:
        dataset.is_inference = True
    try:
        sample = dataset[global_index]
    finally:
        if old_mode is not None:
            dataset.is_inference = old_mode
    return sample


def visualize_triplet_images(
    dataset,
    triplets_global_indexes,
    save_dir="triplet_vis",
    num_triplets_to_show=1,
    max_boxes=30,
    coords_normalized=True,
    mean=None,
    std=None,
    ):
    """
    Сохраняет каждый triplet в отдельный .jpg файл.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not torch.is_tensor(triplets_global_indexes):
        triplets_global_indexes = torch.as_tensor(triplets_global_indexes)

    B, K = triplets_global_indexes.shape
    num_triplets_to_show = min(num_triplets_to_show, B)

    for b in range(num_triplets_to_show):
        gidx = triplets_global_indexes[b].detach().cpu().tolist()

        q_local_idx = int(gidx[0])
        p_db_idx = int(gidx[1])
        neg_db_idx = [int(x) for x in gidx[2:]]

        q_global_idx = dataset.database_num + q_local_idx

        q_sample = _get_sample_via_getitem(dataset, q_global_idx)
        p_sample = _get_sample_via_getitem(dataset, p_db_idx)
        n_samples = [_get_sample_via_getitem(dataset, i) for i in neg_db_idx]

        q_scene = q_sample.get("scene")
        p_scene = p_sample.get("scene")
        n_scenes = [s.get("scene") for s in n_samples]

        q_ref = _room_id(dataset, q_scene)
        p_ref = _room_id(dataset, p_scene)
        n_refs = [_room_id(dataset, s) for s in n_scenes]

        print("=" * 120)
        print(f"TRIPLET #{b}")
        print(
            f"QUERY    | q_local={q_local_idx} | q_global={q_global_idx} | "
            f"scene={q_scene} | ref={q_ref} | img={q_sample.get('img_path')}"
        )
        print(
            f"POSITIVE | db_idx={p_db_idx}     | scene={p_scene} | "
            f"ref={p_ref} | img={p_sample.get('img_path')}"
        )
        for i, s in enumerate(n_samples):
            print(
                f"NEGATIVE_{i+1} | db_idx={neg_db_idx[i]} | "
                f"scene={n_scenes[i]} | ref={n_refs[i]} | img={s.get('img_path')}"
            )

        n_plots = 2 + len(n_samples)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        _show_sample(
            axes[0],
            q_sample,
            title=f"QUERY\nscene={q_scene}\nref={q_ref}\nidx={q_local_idx}",
            max_boxes=max_boxes,
            coords_normalized=coords_normalized,
            mean=mean,
            std=std,
        )
        _show_sample(
            axes[1],
            p_sample,
            title=f"POSITIVE\nscene={p_scene}\nref={p_ref}\nidx={p_db_idx}",
            max_boxes=max_boxes,
            coords_normalized=coords_normalized,
            mean=mean,
            std=std,
        )

        for j, s in enumerate(n_samples):
            _show_sample(
                axes[2 + j],
                s,
                title=f"NEGATIVE_{j+1}\nscene={n_scenes[j]}\nref={n_refs[j]}\nidx={neg_db_idx[j]}",
                max_boxes=max_boxes,
                coords_normalized=coords_normalized,
                mean=mean,
                std=std,
            )

        plt.tight_layout()

        out_path = os.path.join(save_dir, f"triplet_{b:04d}.jpg")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"saved: {out_path}")

def _add_border(ax, color, linewidth=4):
    rect = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(rect)


def _format_title(prefix, scene_name, room_name, score=None):
    lines = [prefix, f"scene: {scene_name}", f"ref: {room_name}"]
    if score is not None:
        lines.append(f"score: {score:.4f}")
    return "\n".join(lines)


def visualize_retrieval(
    dataset,
    nn_idx,
    nn_scores,
    q_indices,
    db_indices,
    save_dir="vis",
    top_k=5,
    num_queries=5,
    random_sample=True,
):
    """
    Visualization of retrieval results.

    Args:
        dataset: BaseDataset
        nn_idx: np.ndarray [num_queries, max_k], nearest db local indices
        nn_scores: np.ndarray [num_queries, max_k], FAISS scores (IP after normalization)
        q_indices: global indices of queries in dataset
        db_indices: global indices of database items in dataset
        save_dir: output directory
        top_k: number of retrieved results to show
        num_queries: number of queries to visualize
        random_sample: if True, pick random queries; else first num_queries
    """
    os.makedirs(save_dir, exist_ok=True)

    num_total_queries = len(q_indices)
    if num_total_queries == 0:
        print("No queries to visualize.")
        return

    if random_sample:
        chosen = random.sample(range(num_total_queries), min(num_queries, num_total_queries))
    else:
        chosen = list(range(min(num_queries, num_total_queries)))

    for vis_rank, qi in enumerate(chosen):
        q_global_idx = q_indices[qi]
        q_item = dataset.get_raw_item(q_global_idx)

        q_img = _load_img(q_item.get("img"))
        q_scene = q_item.get("scene")
        q_room = _room_id(dataset, q_scene)

        fig, axes = plt.subplots(
            1,
            top_k + 1,
            figsize=(4 * (top_k + 1), 4),
            squeeze=False,
        )
        axes = axes[0]

        # Query
        ax = axes[0]
        ax.axis("off")
        if q_img is not None:
            ax.imshow(q_img)
        ax.set_title(_format_title("QUERY", q_scene, q_room), fontsize=10)
        _add_border(ax, color="blue", linewidth=5)

        # Retrieved items
        for k in range(top_k):
            db_local_idx = int(nn_idx[qi, k])
            db_global_idx = db_indices[db_local_idx]
            db_item = dataset.get_raw_item(db_global_idx)

            db_img = _load_img(db_item.get("img"))
            db_scene = db_item.get("scene")
            db_room = _room_id(dataset, db_scene)
            score = float(nn_scores[qi, k])

            is_correct = (db_room == q_room)
            border_color = "green" if is_correct else "red"

            ax = axes[k + 1]
            ax.axis("off")
            if db_img is not None:
                ax.imshow(db_img)

            ax.set_title(
                _format_title(f"TOP {k+1}", db_scene, db_room, score=score),
                fontsize=10,
            )
            _add_border(ax, color=border_color, linewidth=5)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"query_{vis_rank:03d}_datasetidx_{q_global_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[VIS] saved: {save_path}")