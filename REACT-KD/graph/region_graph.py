import numpy as np
import torch
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
import os

def compute_centroids(mask):
    """
    Compute centroids from a binary 3D mask.
    Return list of (z, y, x) tuples in float (not int).
    """
    labeled = mask.astype(int)
    props = regionprops(labeled)
    return [prop.centroid for prop in props]

def build_region_graph(liver_mask, tumor_mask, feature_volume=None):
    """
    Construct a region-level graph.
    Args:
        liver_mask: binary [D, H, W]
        tumor_mask: binary [D, H, W]
        feature_volume: [C, D, H, W] (encoder feature of one sample)

    Returns:
        dict with 'node_feats', 'adj_matrix', 'meta'
    """
    liver_centroids = compute_centroids(liver_mask)
    tumor_centroids = compute_centroids(tumor_mask)

    if len(liver_centroids) == 0 or len(tumor_centroids) == 0:
        return None

    nodes = liver_centroids + tumor_centroids
    num_nodes = len(nodes)

    # 1. 构造邻接矩阵（非自环全连接）
    coords = np.array(nodes)
    dists = cdist(coords, coords)
    adj = (dists > 0).astype(float)

    # 2. 节点特征：在 feature_volume 上取 [C, D, H, W] 的对应位置
    node_feats = []
    for c in coords:
        z, y, x = np.round(c).astype(int)
        z = np.clip(z, 0, feature_volume.shape[1] - 1)
        y = np.clip(y, 0, feature_volume.shape[2] - 1)
        x = np.clip(x, 0, feature_volume.shape[3] - 1)
        feat = feature_volume[:, z, y, x]  # shape: [C]
        node_feats.append(feat.detach().cpu().numpy())

    node_feats = np.stack(node_feats)  # [N, C]
    feat_tensor = torch.tensor(node_feats, dtype=torch.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    return {
        "node_feats": feat_tensor,
        "adj_matrix": adj_tensor,
        "meta": {
            "num_liver": len(liver_centroids),
            "num_tumor": len(tumor_centroids),
        }
    }

def save_region_graph(graph, pid, epoch, fold, save_dir_root):
    """
    Save graph dict as .pt file: Fold_{fold}/Epoch_{epoch}/graph_{pid}.pt
    """
    if graph is None:
        return
    save_dir = os.path.join(save_dir_root, f"Fold_{fold}", f"Epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"graph_{pid}.pt")
    torch.save(graph, save_path)
