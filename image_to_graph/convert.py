"""
Core mask-to-graph conversion.

Returns a 4-tuple (data, skeleton, nx_graph, skip_reason):
  - data        : PyG Data object — save this to disk
  - skeleton    : np.ndarray uint8 — for visualisation only, do NOT save to .pt
  - nx_graph    : networkx Graph  — for visualisation only, do NOT save to .pt
  - skip_reason : None on success, or one of:
                    'empty_mask'   — could not read mask or no crack pixels
                    'empty_graph'  — sknw produced 0 nodes
                    'degenerate'   — graph too small after spur pruning
"""

import copy
import os
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
import sknw
from torch_geometric.data import Data

from .features import (
    compute_distance_transform,
    compute_node_features,
    compute_edge_features,
)


def _prune_spurs(graph, prune_ratio: float):
    """
    Remove leaf branches shorter than prune_ratio × longest_branch_length.

    A leaf branch is an edge where at least one endpoint has degree 1 (spur
    artifact from rough mask boundaries). Iterates until stable so that
    newly-exposed leaves are also removed. Topology-preserving: only leaf
    edges are ever removed, never interior edges that would split the graph.

    Returns a pruned copy; does not mutate the input.
    """
    g = copy.deepcopy(graph)

    while True:
        if g.number_of_edges() == 0:
            break

        # Geometric branch length for every edge
        lengths = {}
        for u, v, edata in g.edges(data=True):
            pts  = edata.get('pts', np.array([]))
            y_u  = float(g.nodes[u]['o'][0]);  x_u = float(g.nodes[u]['o'][1])
            y_v  = float(g.nodes[v]['o'][0]);  x_v = float(g.nodes[v]['o'][1])
            wpts = ([(y_u, x_u)]
                    + [(float(p[0]), float(p[1])) for p in pts]
                    + [(y_v, x_v)])
            L = sum(
                np.sqrt((wpts[i+1][0] - wpts[i][0])**2 +
                        (wpts[i+1][1] - wpts[i][1])**2)
                for i in range(len(wpts) - 1)
            )
            lengths[(u, v)] = L

        threshold = prune_ratio * max(lengths.values())

        to_remove = [
            (u, v) for (u, v), L in lengths.items()
            if L < threshold and (g.degree(u) == 1 or g.degree(v) == 1)
        ]
        if not to_remove:
            break

        for u, v in to_remove:
            if g.has_edge(u, v):
                g.remove_edge(u, v)
        g.remove_nodes_from([n for n in list(g.nodes()) if g.degree(n) == 0])

    return g


def mask_to_graph(
    mask_path: str,
    split: str = 'train',
    prune_ratio: float = 0.1,
    min_nodes: int = 3,
) -> tuple:
    """
    Convert a binary crack mask PNG to a PyG Data object.

    Args:
        mask_path   : path to a binary mask (white = crack, black = background)
        split       : dataset split label ('train' / 'test')
        prune_ratio : remove leaf branches shorter than this fraction of the
                      longest branch (0.0 disables pruning)
        min_nodes   : drop graphs with fewer nodes than this after pruning

    Returns:
        (data, skeleton, nx_graph, skip_reason)
        skip_reason is None on success, otherwise a string describing why
        the mask was skipped (caller uses this for the curation log).
    """
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return None, None, None, 'empty_mask'

    _, binary_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
    if binary_mask.sum() == 0:
        return None, None, None, 'empty_mask'

    H, W = binary_mask.shape

    skeleton = skeletonize(binary_mask).astype(np.uint8)
    nx_graph = sknw.build_sknw(skeleton)

    if len(nx_graph.nodes()) == 0:
        return None, None, None, 'empty_graph'

    # ── Spur pruning (Issue 3) ────────────────────────────────────────────────
    if prune_ratio > 0.0:
        nx_graph = _prune_spurs(nx_graph, prune_ratio)

    # ── Degenerate graph filter (Issue 4) ────────────────────────────────────
    if nx_graph.number_of_nodes() < min_nodes or nx_graph.number_of_edges() == 0:
        return None, None, None, 'degenerate'

    dist_map = compute_distance_transform(binary_mask)

    x, node_index = compute_node_features(nx_graph, dist_map, H, W)
    edge_index, edge_attr = compute_edge_features(nx_graph, dist_map, node_index)

    # ── Node pixel positions (x_pixel, y_pixel) for Stage 3 hard negatives ──
    # sknw stores positions as (row, col) = (y, x); we store (x, y) here to
    # match the (x_norm, y_norm) ordering in the feature matrix.
    ordered_nids = list(nx_graph.nodes())
    pos = torch.tensor(
        [[float(nx_graph.nodes[nid]['o'][1]), float(nx_graph.nodes[nid]['o'][0])]
         for nid in ordered_nids],
        dtype=torch.float,
    )

    crack_pixels = int(binary_mask.sum())
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        # ── metadata ─────────────────────────────────────────────────────────
        filename=os.path.basename(mask_path),
        split=split,
        img_h=H,
        img_w=W,
        crack_pixels=crack_pixels,
        crack_density=float(crack_pixels) / float(H * W),
    )

    return data, skeleton, nx_graph, None
