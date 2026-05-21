"""
Core mask-to-graph conversion.

Returns a 3-tuple (data, skeleton, nx_graph):
  - data      : PyG Data object — save this to disk
  - skeleton  : np.ndarray uint8 — for visualisation only, do NOT save to .pt
  - nx_graph  : networkx Graph  — for visualisation only, do NOT save to .pt
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import sknw
from torch_geometric.data import Data

from .features import (
    compute_distance_transform,
    compute_node_features,
    compute_edge_features,
)


def mask_to_graph(mask_path: str, split: str = 'train') -> tuple:
    """
    Convert a binary crack mask PNG to a PyG Data object.

    Args:
        mask_path: path to a binary mask (white = crack, black = background)
        split:     dataset split label stored on the Data object ('train' / 'test')

    Returns:
        (data, skeleton, nx_graph)  or  (None, None, None) if mask has no cracks.
    """
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return None, None, None

    _, binary_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
    H, W = binary_mask.shape

    skeleton = skeletonize(binary_mask).astype(np.uint8)
    nx_graph = sknw.build_sknw(skeleton)

    if len(nx_graph.nodes()) == 0:
        return None, None, None

    dist_map = compute_distance_transform(binary_mask)

    x, node_index = compute_node_features(nx_graph, dist_map, H, W)
    edge_index, edge_attr = compute_edge_features(nx_graph, dist_map, node_index)

    crack_pixels = int(binary_mask.sum())
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # ── metadata (stored as graph-level attributes) ──────────────────
        filename=os.path.basename(mask_path),
        split=split,
        img_h=H,
        img_w=W,
        crack_pixels=crack_pixels,
        crack_density=float(crack_pixels) / float(H * W),
    )

    return data, skeleton, nx_graph
