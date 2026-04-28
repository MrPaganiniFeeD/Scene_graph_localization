import os
import sys
import torch
import parser
import logging
import sklearn
import util
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
if args.load_state_mode == "graph":
    gat_graph_encoder = util.load_model(args, args.load_model)
elif args.load_state_mode == "image":
    pass
elif args.load_state_mode == "multimodal":
    loaded_multimodal = util.load_model(args, args.load_model)

model = loaded_multimodal

model = torch.nn.DataParallel(model)

if args.resume != None:
    state_dict = torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)


######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
if args.mode == "fusion" or args.mode == "graph":
    norm_ckpt = torch.load(
        os.path.join(args.load_model, "edge_normalizer.pt"),
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