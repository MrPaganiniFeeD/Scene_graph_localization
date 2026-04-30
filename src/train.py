import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import datasets_ws
import network
import parser
import visualize
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark=True

import util
import commons
import warnings
import test
warnings.filterwarnings('ignore')
import os

DEFAULT_MEAN = [0.44420420130352495, 0.41322746532289134, 0.3678658064565412]
DEFAULT_STD = [0.24352604373543688, 0.24045797651069503, 0.24250136992133814]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")


@torch.no_grad()
def compute_edge_attr_stats_from_dataset(ds, max_items=None):
    sum_ = None
    sum_sq = None
    count = 0

    items = ds.items if max_items is None else ds.items[:max_items]

    for item in tqdm(items, desc="Edge stats"):
        graph_path = item["graph"]
        if graph_path is None:
            continue

        g = ds.loader.load_graph(graph_path)

        graphs = g if isinstance(g, list) else [g]
        for gi in graphs:
            if gi is None or gi.edge_attr is None or gi.edge_attr.numel() == 0:
                continue

            ea = gi.edge_attr.detach().float().cpu()   # [E, F]

            if sum_ is None:
                sum_ = torch.zeros(ea.shape[1], dtype=torch.float64)
                sum_sq = torch.zeros(ea.shape[1], dtype=torch.float64)

            sum_ += ea.double().sum(dim=0)
            sum_sq += (ea.double() ** 2).sum(dim=0)
            count += ea.shape[0]

    if count == 0:
        raise RuntimeError("No edge_attr found.")

    mean = sum_ / count
    var = sum_sq / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    print("count:", count)
    print("mean per feature:", mean)
    print("std per feature:", std)

    return mean, std


#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)

logging.info(f"Train query set: {triplets_ds}")

# val_ds = BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
# logging.info(f"Val set: {val_ds}")


test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### normalizer graph
if args.mode == "graph" or args.mode == "fusion":
    graph_normalizer = network.EdgeAttrNormalizer()
    for item in triplets_ds.items:
        graph_path = item["graph"]
        if graph_path is None:
            continue
        
        g = torch.load(graph_path, map_location="cpu")
        g = datasets_ws._sanitize_graph_obj(g, args.in_dim_graph, args.edge_attr_dim)

        if isinstance(g, list):
            for gi in g:
                if gi is not None and gi.edge_attr is not None and gi.edge_attr.numel() > 0:
                    graph_normalizer.update(gi.edge_attr)
        else:
            if g.edge_attr is not None and g.edge_attr.numel() > 0:
                graph_normalizer.update(g.edge_attr)

    graph_normalizer.finalize()
    util.save_edge_normalizer(args, graph_normalizer.mean, graph_normalizer.std, graph_normalizer.log_indices, "edge_normalizer.pt")
    triplets_ds.loader.edge_normalizer = graph_normalizer
    test_ds.loader.edge_normalizer = graph_normalizer
    compute_edge_attr_stats_from_dataset(triplets_ds, max_items=1000)



util.save_networks(args)

#### Initialize model
graph_encoder = network.VPRGraphEncoder(
    in_dim=args.in_dim_graph,
    hidden_dim=args.graph_hidden_dim,
    n_layers=args.graph_layers,
    num_node_classes=args.num_obj_classes, 
    node_emb_dim=args.node_emb_dim,
    num_edge_classes=args.num_edge_classes,
    edge_emb_dim=args.edge_emb_dim,
    proj_dim=args.graph_proj,
    dropout=args.graph_dropout).to(args.device)

GAT_graph_encoder = network.GATGraphEncoder(
    in_dim=args.in_dim_graph,
    hidden_dim=args.graph_hidden_dim,
    n_layers=args.graph_layers,
    num_node_classes=args.num_obj_classes, 
    node_emb_dim=args.node_emb_dim,
    num_edge_classes=args.num_edge_classes,
    edge_emb_dim=args.edge_emb_dim,
    proj_dim=args.graph_proj,
    edge_cont_dim=args.edge_attr_dim,
    dropout=args.graph_dropout,
    heads=args.graph_head).to(args.device)

    
megaloc = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
image_encoder = megaloc.to(args.device)

if args.load_state_mode == "graph":
    gat_graph_encoder = util.load_model(args, args.load_model)
elif args.load_state_mode == "image":
    pass
elif args.load_state_mode == "multimodal":
    pass

model = network.FusionMLP(
    graph_encoder=graph_encoder,
    image_encoder=image_encoder if "image" in args.modalities else None,
    graph_out_dim=256,
    image_out_dim=8448,
    fusion_dim=4096,
    freeze_image_encoder=True,
    train_only_aggregator=False,
    normalize=True,
).to(args.device)

args.graph_model_name = model.graph_encoder.__class__.__name__
args.image_model_name = model.image_encoder.__class__.__name__
args.multimodel_model_name = model.__class__.__name__

### Setup Optimizer and Loss
if args.optim == "adam":
    if args.mode == "graph":
        optimizer = torch.optim.Adam(model.graph_encoder.parameters(), lr=args.graph_lr, weight_decay=1e-4)
    elif args.mode == "image":
        # trainable_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        # optimizer = torch.optim.Adam(model.image_encoder.aggregator.parameters(), lr=args.lr * 0.1)
        optimizer = torch.optim.Adam(
            model.image_encoder.aggregator.linear.parameters(),
            lr=3e-7,
            weight_decay=1e-4
        )
    elif args.mode == "fusion":
        optimizer = torch.optim.AdamW(
            [
                # graph encoder
                {"params": model.graph_encoder.parameters(), "lr": args.graph_lr},
                # fusion
                {"params": model.mlp.parameters(), "lr": args.lr},
            ],
            weight_decay=1e-4
        )

        
        if model.image_encoder is not None and False:
            optimizer.add_param_group({
                "params": model.image_encoder.aggregator.linear.parameters(),
                "lr": 3e-7
            })
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(base_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2, reduction=args.loss_reduction)

#### Resume model, optimizer, and other training parameters
if args.resume:
    #model, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
else:
    best_r5 = start_epoch_num = not_improved_num = 0

best_r5 = 0
#logging.info(f"Output dimension of the model is {args.features_dim}")

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False
        
        
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                collate_fn=lambda batch: datasets_ws.collate_fn(batch, args.in_dim_graph, args.edge_attr_dim),
                                pin_memory=(args.device=="cuda"),
                                drop_last=True)
        
        model = model.train()
        print(len(triplets_ds), "Количество триплетов")
        print(len(triplets_dl))
        triplets_number = 0
        for batch_samples, triplets_local_indexes, triplets_global_indexes in tqdm(triplets_dl, ncols=100):
            # Compute features of all images (images contains queries, positives and negatives)
            if epoch_num == 0 and loop_num == 0 and triplets_number == 0:
                visualize.visualize_triplet_images(
                    dataset=triplets_ds,
                    triplets_global_indexes=triplets_global_indexes,
                    save_dir=args.save_dir,       
                    num_triplets_to_show=20,
                    max_boxes=30,
                    coords_normalized=True,   # если graph['x'] в [0,1]
                    mean=DEFAULT_MEAN,
                    std=DEFAULT_STD,
                )
            
            if "graph" in args.modalities:
                batch_graph = batch_samples["graph"].to(args.device)
            else:
                batch_graph = None
            if "image" in args.modalities:
                batch_image = batch_samples["image"].to(args.device)
            else:
                batch_image = None
            
            outputs = model(
                graph=batch_graph,
                image=batch_image,
                mode=args.mode,   # "graph" / "fusion"
                return_parts=True,
            )
            
            

            embeddings = outputs["fused"]

            total_loss = 0
            """
            if args.criterion == "triplet":    
                print()   
                N = args.negs_num_per_query
                B = embeddings.shape[0] // (2 + N)

                embeddings = embeddings.view(B, 2 + N, -1)
                queries = embeddings[:, 0]
                positives = embeddings[:, 1]
                negatives = embeddings[:, 2:]  # [B, N, D]

                anchor = queries[:, None, :].expand(-1, N, -1).reshape(-1, embeddings.size(-1))
                positive = positives[:, None, :].expand(-1, N, -1).reshape(-1, embeddings.size(-1))
                negative = negatives.reshape(-1, embeddings.size(-1))

                total_loss = triplet_loss(anchor, positive, negative)
            """
            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T

                    total_loss += triplet_loss(embeddings[queries_indexes],
                                                      embeddings[positives_indexes],
                                                      embeddings[negatives_indexes])



            total_loss /= (args.train_batch_size * args.negs_num_per_query)

                                                    


            del embeddings

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_loss = total_loss.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del total_loss
            triplets_number += 1

        
        
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                    f"current batch triplet loss = {batch_loss:.4f}, " +
                    f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        recalls, recalls_str, _ = test.test(args, test_ds, model)
        logging.info(f"Recalls on val set {test_ds}: {recalls_str}")

    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"average epoch triplet loss = {epoch_losses.mean():.4f}")


    # Compute recalls on validation set
    
    if loops_num != 1:
        recalls, recalls_str, nn_idx = test.test(args, test_ds, model)
        logging.info(f"Recalls on {test_ds}: {recalls_str}")

    is_best = recalls['R@5'] > best_r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num,
        "multimodal_state_dict": model.state_dict(),
        "graph_state_dict": model.graph_encoder.state_dict() if model.graph_encoder != None else None,
        "image_state_dict": model.image_encoder.state_dict() if model.image_encoder != None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": recalls,
        "best_r5": best_r5,
        "not_improved_num": not_improved_num,
        "mode": args.mode,
        "graph_init_args": (model.graph_encoder.init_args if model.graph_encoder != None else None),
        "multimodal_init_args": (model.init_args if model != None else None),
        "image_init_args": (None if model.image_encoder else None)
    }, is_best, filename="last_model.pth")
    
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {(recalls['R@5']):.1f}")
        best_r5 = recalls['R@5']
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {(recalls['R@5']):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break



logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str, _ = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")