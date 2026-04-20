import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime

import test
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
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

checkpoint = torch.load("/workspace/tmp/projects/Scene_graph_localization/data/2026-04-19_22-01-57/best_model.pth", map_location='cpu')

model.load_state_dict(checkpoint["model_state_dict"])
# print("model", model)


model = torch.nn.DataParallel(model)

if args.resume != None:
    state_dict = torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)


######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
norm_ckpt = torch.load(
    "/workspace/tmp/projects/Scene_graph_localization/data/2026-04-19_22-01-57/edge_normalizer.pt",
    map_location="cpu"
)

graph_normalizer = network.EdgeAttrNormalizer(
    log_indices=norm_ckpt["log_indices"]
)
graph_normalizer.mean = norm_ckpt["mean"]
graph_normalizer.std = norm_ckpt["std"]

test_ds.loader.edge_normalizer = graph_normalizer
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str, nn_idx = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")