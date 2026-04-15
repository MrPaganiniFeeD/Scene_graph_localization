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


#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)

logging.info(f"Train query set: {triplets_ds}")

# val_ds = BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
# logging.info(f"Val set: {val_ds}")


test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
graph_encoder = network.VPRGraphEncoder(in_dim=args.in_dim_graph, hidden_dim=args.graph_hidden_dim, n_layers=args.graph_layers, num_node_classes=args.num_obj_classes, num_edge_classes=args.num_edge_classes, proj_dim=args.graph_proj).to(args.device)

megaloc = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
image_encoder = megaloc.to(args.device)

model = network.MultiModalVPRGraphEncoder(
    graph_encoder=graph_encoder,
    image_encoder=image_encoder,
    image_out_dim=8448,
    graph_out_dim=256,
    fusion_dim=8448,
    normalize=True,
    graph_fusion_scale=0.05,
    freeze_image_encoder=True,

).to(args.device)

### Setup Optimizer and Loss
if args.optim == "adam":
    if args.mode == "graph":
        optimizer = torch.optim.Adam(model.graph_encoder.parameters(), lr=args.graph_lr, weight_decay=1e-4)
    elif args.mode == "image":
        # trainable_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model.image_encoder.aggregator.parameters(), lr=args.lr * 0.1)
    elif args.mode == "fusion":
        optimizer = torch.optim.Adam(
            [
                {"params": model.graph_encoder.parameters(), "lr": args.lr},
                {"params": model.graph_proj.parameters(), "lr": args.lr},
                {"params": model.graph_gate.parameters(), "lr": args.lr},
                {"params": model.image_proj.parameters(), "lr": args.lr},
                {"params": model.fuse_norm.parameters(), "lr": args.lr},
                {"params": model.fuse_mlp.parameters(), "lr": args.lr},
            ],
            weight_decay=1e-4
        )

elif args.optim == "sgd":
    optimizer = torch.optim.SGD(base_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

triplet_loss = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

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
                                collate_fn=datasets_ws.collate_fn,
                                pin_memory=(args.device=="cuda"),
                                drop_last=True)
        
        model = model.train()
        print(len(triplets_ds), "Количество триплетов")
        # images shape: (train_batch_size*4)*3*H*W
        # print(triplets_ds[0], "triplets_ds[0]")
        print(len(triplets_dl))
        for batch_samples, triplets_local_indexes, triplets_global_indexes in tqdm(triplets_dl, ncols=100):
            # Compute features of all images (images contains queries, positives and negatives)
            if epoch_num == 0 and loop_num == 0 and False:
                visualize.visualize_triplet_images(
                    dataset=triplets_ds,
                    triplets_global_indexes=triplets_global_indexes,
                    save_dir="/workspace/tmp/projects/Scene_graph_localization/data/triplets_vis",       
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
        
        
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                    f"current batch triplet loss = {batch_loss:.4f}, " +
                    f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        recalls, recalls_str, _ = test.test(args, test_ds, model)
        #recalls_dist, recalls_str_dist, _ = test_patched.test(args, test_ds, graph_encoder)
        logging.info(f"Recalls on val set {test_ds}: {recalls_str}")
        #logging.info(f"Recalls on val set (radius=2.0m) {test_ds}: {recalls_str_dist}")

    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"average epoch triplet loss = {epoch_losses.mean():.4f}")


    # Compute recalls on validation set
    
    if loops_num != 1:
        recalls, recalls_str, nn_idx = test.test(args, test_ds, model)
        logging.info(f"Recalls on {test_ds}: {recalls_str}")

    is_best = recalls['R@5'] > best_r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num, "mode": args.mode,
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