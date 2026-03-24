import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for VPRGraphEncoder with triplet loss', allow_abbrev=False)
    
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=4,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--mode", type=str, default="image", help="_", choices=["graph", "image", "fusion"])
    parser.add_argument("--cache_refresh_rate", type=int, default=4,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=2,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument("--registers", action='store_true', help="_")
    parser.add_argument("--features_dim", type=int, default=128, help="_")
    parser.add_argument("--in_dim_graph", type=int, default=4, help="_")

    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
#    parser.add_argument("--foundation_model_path", type=str, default=None,
#                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322,322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--dense_feature_map_size', type=int, default=[61,61,128], nargs=3, 
                        help="size of dense feature map (a 61x61 grid 128-dim local features)")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1,5,10,20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--rerank_num", type=int, default=100, help="_")
    # Data augmentation parameters
    parser.add_argument("--modalities", nargs='+', choices=['image', 'graph', 'pose'], 
                    default=['pose', 'image'], help="List of modalities")
    parser.add_argument("--visualize", type=bool, default=True, help='_')
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default="/mnt/external_usb_hdd/6YL/Datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="3RScan", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="/home/pinkin_ek/projects/Scene_graph_localization/data",
                        help="Folder name of the current run (saved in ./logs/)")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignored unknown arguments: {unknown}")

    return args