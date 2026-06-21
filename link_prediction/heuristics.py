"""
Tier 1 — Non-learning structural heuristics (the floor the GNN must beat).

Common Neighbors, Adamic-Adar, Resource Allocation.
All three operate on the observed (message-passing) graph only — they see
exactly what the GNN sees, so the comparison is fair.

Note: crack skeleton graphs are near-planar and locally tree-like, so raw
CN/AA/RA may be weak (crack tips typically share zero common neighbours).
That is fine — a low floor the GNN clears convincingly is a clean result.
"""

import math
import numpy as np
import networkx as nx
import torch


def _data_to_nx(train_data) -> nx.Graph:
    """Build an undirected NetworkX graph from a split's observed edge_index."""
    G = nx.Graph()
    G.add_nodes_from(range(train_data.x.size(0)))
    ei = train_data.edge_index.numpy()
    for u, v in zip(ei[0], ei[1]):
        G.add_edge(int(u), int(v))
    return G


def _cn_score(G: nx.Graph, u: int, v: int) -> float:
    return float(len(list(nx.common_neighbors(G, u, v))))


def _aa_score(G: nx.Graph, u: int, v: int) -> float:
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        d = G.degree(w)
        if d > 1:
            score += 1.0 / math.log(d)
    return score


def _ra_score(G: nx.Graph, u: int, v: int) -> float:
    score = 0.0
    for w in nx.common_neighbors(G, u, v):
        d = G.degree(w)
        if d > 0:
            score += 1.0 / d
    return score


_SCORERS = {
    'cn': _cn_score,
    'aa': _aa_score,
    'ra': _ra_score,
}


def score_pairs(train_data, edge_index: torch.Tensor, method: str) -> np.ndarray:
    """
    Score candidate pairs using a structural heuristic.

    Args:
        train_data  : PyG Data with observed edges (message-passing graph)
        edge_index  : [2, E] candidate pairs to score
        method      : 'cn', 'aa', or 'ra'

    Returns:
        scores : float32 array of length E
    """
    if method not in _SCORERS:
        raise ValueError(f"method must be 'cn', 'aa', or 'ra'; got {method!r}")
    fn = _SCORERS[method]
    G  = _data_to_nx(train_data)
    ei = edge_index.numpy()
    return np.array(
        [fn(G, int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])],
        dtype=np.float32,
    )


def score_split(split: dict, method: str) -> tuple:
    """
    Score val and test candidate pairs from a transductive_split dict.

    Returns (val_scores, val_labels, test_scores, test_labels) as numpy arrays.
    """
    train_d = split['train_data']
    val_scores  = score_pairs(train_d, split['val_ei'],  method)
    test_scores = score_pairs(train_d, split['test_ei'], method)
    return (
        val_scores,  split['val_labels'].numpy(),
        test_scores, split['test_labels'].numpy(),
    )
