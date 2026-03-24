from torch.utils.data import DataLoader, Subset
import numpy as np
import faiss
import torch
from tqdm import tqdm
import datasets_ws
import visualize
from os.path import basename, join

def _build_subset_loader(dataset, indices, args, shuffle=False):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=args.infer_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=datasets_ws.collate_fn,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

@torch.no_grad()
def _extract_embeddings(model, loader, device, args):
    model.eval()
    all_embs = []
    all_scenes = []

    for batch_samples in tqdm(loader, desc="Extract embeddings", ncols=100):
        #print("TEST BATCH[0]", batch[0])        
        graph = batch_samples.get("graph", None)
        image = batch_samples.get("image", None)

        if graph is not None:
            graph = graph.to(device)
        if image is not None:
            image = image.to(device)
        outputs = model(
            graph=graph,
            image=image,
            mode=args.mode,
            return_parts=True,
        )

        emb = outputs["fused"].detach().float().cpu()

        scenes = batch_samples["scene"]
        if isinstance(scenes, str):
            scenes = [scenes]

        all_embs.append(emb)
        all_scenes.extend(list(scenes))

    all_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    return all_embs, all_scenes

def test(args, dataset, model, device=None, ks=(1, 5, 10, 20)):
    if device is None:
        device = args.device

    # database = первые database_items
    db_indices = list(range(len(dataset.database_items)))
    # queries = всё остальное
    q_indices = list(range(len(dataset.database_items), len(dataset.items)))

    db_loader = _build_subset_loader(dataset, db_indices, args, shuffle=False)
    q_loader = _build_subset_loader(dataset, q_indices, args, shuffle=False)

    db_embs, db_scenes = _extract_embeddings(model, db_loader, device, args)
    q_embs, q_scenes = _extract_embeddings(model, q_loader, device, args)

    faiss.normalize_L2(db_embs)
    faiss.normalize_L2(q_embs)

    index = faiss.IndexFlatIP(db_embs.shape[1])
    index.add(db_embs)

    max_k = max(ks)
    nn_scores, nn_idx = index.search(q_embs, max_k)

    if getattr(args, "visualize", False):
        save_path = join(args.save_dir, "test")
        visualize.visualize_retrieval(
            dataset=dataset,
            nn_idx=nn_idx,
            nn_scores=nn_scores,
            q_indices=q_indices,
            db_indices=db_indices,
            save_dir=save_path,
            top_k=5,
            num_queries=getattr(args, "vis_num_queries", 5),
            random_sample=True,
        )


    def room_id(scene_name):
        return dataset.scan_to_ref.get(scene_name, scene_name)

    q_rooms = [room_id(s) for s in q_scenes]
    db_rooms = [room_id(s) for s in db_scenes]

    recalls = {}
    for k in ks:
        hit = 0
        for qi in range(len(q_rooms)):
            retrieved_rooms = [db_rooms[j] for j in nn_idx[qi, :k]]
            if q_rooms[qi] in retrieved_rooms:
                hit += 1
        recalls[f"R@{k}"] = hit / max(1, len(q_rooms))

    recalls_str = ", ".join([f"R@{k}: {recalls[f'R@{k}']*100:.2f}" for k in ks])
    return recalls, recalls_str, nn_idx