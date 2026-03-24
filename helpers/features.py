import math
import numpy as np

EPS = 1e-8

def iou2d_xyxy(a, b, eps=EPS):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    max_x_left, max_y_left = max(ax1, bx1), max(ay1, by1)
    min_x_right, min_y_right = min(ax2, bx2), min(ay2, by2)
    iw = max(0, min_x_right - max_x_left)
    ih = max(0, min_y_right - max_y_left)
    inter = iw * ih
    EPS = 1e-8

def iou2d_xyxy(a, b, eps=EPS):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + eps
    return float(inter / union)

def aabb_iou_3d(min_a, max_a, min_b, max_b, eps=EPS):
    min_a = np.array(min_a, dtype=float); max_a = np.array(max_a, dtype=float)
    min_b = np.array(min_b, dtype=float); max_b = np.array(max_b, dtype=float)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
    union = vol_a + vol_b - inter_vol + eps
    return float(inter_vol / union)


def angle_sin_cos(dx, dy):
    theta = math.atan2(dy, dx)
    return float(math.sin(theta)), float(math.cos(theta))


def direction_bin(dx, dy, n_bins=8):
    theta = math.atan2(dy, dx)
    t = theta if theta >= 0 else (theta + 2*math.pi)
    bin_idx = int(math.floor(t/(2*math.pi) * n_bins)) % n_bins
    onehot = [0] * n_bins 
    onehot[bin_idx] = 1
    return onehot, bin_idx
