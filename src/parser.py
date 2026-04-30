import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for VPRGraphEncoder with triplet loss', allow_abbrev=False)
    
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.5,
                        help="margin for the triplet loss") # L=max(d(query, positive) - d(query, negative) + margin, 0) margin - минимаьлное расстояние до негатива
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--loss_reduction", type=str, default="sum", help="_", choices=["mean"]) # лос по батчам складывается (sum) или усредняется (mean)
    parser.add_argument("--mode", type=str, default="fusion", help="_", choices=["graph", "image", "fusion"]) # режим обучения
    parser.add_argument("--cache_refresh_rate", type=int, default=2000,
                        help="How often to refresh cache, in number of queries") # размера кэша 
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=2,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=5000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"]) # random - random positive and negative, partial - hardest negative
    parser.add_argument("--load_state_mode", type=str, default="None", help="_", choices=["graph", "image", "multimodal", "None"])
    # Model parameters
    parser.add_argument("--features_dim", type=int, default=256 + 8448, help="_") # Размер выходного эмбединга
    parser.add_argument("--in_dim_graph", type=int, default=4, help="_") # Размерность геометрических фичей в node
    parser.add_argument("--edge_attr_dim", type=int, default=10, help="_") # Размерность геометрических атрибутов 
    parser.add_argument("--graph_rotate", type=bool, default=False, help="_") # Graph rotate
    parser.add_argument("--soft_positives_radius", type=float, default=0.5, help="_") # 
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--visualize", type=bool, default=True, help='_') # Визуализация (например, для триплетов)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=1, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--dense_feature_map_size', type=int, default=[61,61,128], nargs=3, 
                        help="size of dense feature map (a 61x61 grid 128-dim local features)")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1,5,10,25], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--rerank_num", type=int, default=100, help="_")
    
    # GRAPH ENCODER PARAMETRS:
    parser.add_argument("--graph_hidden_dim", type=int, default=512, help="_") # Размерность скрытого слоя 
    parser.add_argument("--node_emb_dim", type=int, default=128, help="_") # Размерность node эмбединга
    parser.add_argument("--edge_emb_dim", type=int, default=128, help="_") # Размерность edge эмбединга 
    parser.add_argument("--graph_layers", type=int, default=1, help="_") # Количество слоёв
    parser.add_argument("--num_obj_classes", type=int, default=528 + 1, help="_") # Количество классов node 
    parser.add_argument("--num_edge_classes", type=int, default=41, help="_") # Количество edge классов 
    parser.add_argument("--graph_proj", type=int, default=256, help="_") # выходной эмбединг 
    parser.add_argument("--graph_dropout", type=float, default=0.1, help="_") # dropout
    parser.add_argument("--graph_lr", type=float, default=0.00001, help="_") # 
    parser.add_argument("--graph_head", type=int, default=4, help="_") # Количество голов (для graph attention)


    # Data augmentation parameters
    parser.add_argument("--modalities", nargs='+', choices=['image', 'graph', 'pose'], 
                    default=['pose', "graph", "image"], help="List of modalities") # Модальность для datasets_ws. Если режим fusion: ['pose', "graph", "image"], graph: ['pose', "graph"], image: ['pose', "image"] 
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default="/mnt/external_usb_hdd/6YL/Datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="3RScan", help="Relative path of the dataset")
    parser.add_argument("--graph_dataset_name", type=str, default="SceneGraphs_real_classes_pt_compact", help="_")
    parser.add_argument("--save_dir", type=str, default="/home/pinkin_ek/projects/Scene_graph_localization/data",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--load_model", type=str, default="./data/2026-04-27_14-03-02")
    parser.add_argument("--graph_model_name", type=str, default="GATGraphEncoder", help="_")
    parser.add_argument("--multimodel_model_name", type=str, default="MultiModalVPRGraphEncoder", help="_")
    parser.add_argument("--image_mdoel_name", type=str, default=None, help="_")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignored unknown arguments: {unknown}")

    return args