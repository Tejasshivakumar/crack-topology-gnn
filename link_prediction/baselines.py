"""
Tier 1.5 — Coordinates-only baseline (critical sanity check).

Predicts link likelihood purely from spatial distance between node pixel
positions (graph.pos), with NO message passing.

We WANT this to lose.  If it rivals the GNN, the hard negatives in splits.py
are not hard enough — return to sample_hard_negatives() and increase k_near.
Its losing is the proof that topology, not proximity, drives the GNN's result.
"""

import numpy as np
import torch


def coords_score(pos: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    """
    Score candidate pairs by 1/(1 + Euclidean pixel distance).

    Closer pairs receive higher scores.  No training required.

    Args:
        pos        : [N, 2] node pixel positions (x, y) from graph.pos
        edge_index : [2, E] candidate pairs to score

    Returns:
        scores : float32 array of length E, each in (0, 1]
    """
    pos_np = pos.numpy()
    ei     = edge_index.numpy()
    u_pos  = pos_np[ei[0]]   # [E, 2]
    v_pos  = pos_np[ei[1]]   # [E, 2]
    dists  = np.linalg.norm(u_pos - v_pos, axis=1)   # [E]
    return (1.0 / (1.0 + dists)).astype(np.float32)


def score_split(split: dict, data) -> tuple:
    """
    Score val and test pairs using coordinates only.

    Args:
        split : dict from transductive_split()
        data  : original PyG graph (for graph.pos)

    Returns:
        (val_scores, val_labels, test_scores, test_labels) as numpy arrays
    """
    pos = data.pos
    val_scores  = coords_score(pos, split['val_ei'])
    test_scores = coords_score(pos, split['test_ei'])
    return (
        val_scores,  split['val_labels'].numpy(),
        test_scores, split['test_labels'].numpy(),
    )
