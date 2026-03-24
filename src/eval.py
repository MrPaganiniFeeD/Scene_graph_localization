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
graph_encoder = network.VPRGraphEncoder(in_dim=args.in_dim_graph, hidden_dim=256, n_layers=3, num_node_classes=528 + 1, proj_dim=128).to(args.device)

megaloc = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
image_encoder = megaloc.to(args.device)

model = network.MultiModalVPRGraphEncoder(
    graph_encoder=graph_encoder,
    image_encoder=image_encoder,
    image_out_dim=8448,
    shared_dim=128,
    fusion_hidden_dim=256,
    normalize=True,
).to(args.device)

model = torch.nn.DataParallel(model)

if args.resume != None:
    state_dict = torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)

"""
if args.pca_dim == None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)
"""

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str, nn_idx = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")