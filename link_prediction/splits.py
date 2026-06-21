"""
Transductive edge split with both critical fixes for the topology proof.

Fix 1 — Feature recomputation (Issue 1, prevents label leakage):
    degree/is_endpoint/is_junction (x cols 3-5) were computed on the full
    graph during Stage 2.  After hiding edges, those values are stale and
    fingerprint where the hidden edges are.  We recompute from the observed
    (message-passing) edges only, so the model sees the graph it actually has.

Fix 2 — Hard negative sampling (Issue 2, forces topology use):
    Random negatives are almost always far apart.  Because node features
    include position (x_norm, y_norm), a model can win by scoring near=real,
    far=fake — no topology needed.  Hard negatives are unconnected pairs that
    are spatially near each other, so distance is no longer a giveaway.
    The SAME negative set is used by every approach (heuristics, baselines,
    GNNs) so comparisons are fair.
"""

import random
import torch
import numpy as np

from torch_geometric.transforms import RandomLinkSplit


# ── Fix 1: feature recompute ──────────────────────────────────────────────────

def recompute_structural_features(x: torch.Tensor,
                                   edge_index_observed: torch.Tensor,
                                   num_nodes: int) -> torch.Tensor:
    """
    Recompute degree (col 3), is_endpoint (col 4), is_junction (col 5) from
    the observed edge set only and overwrite those columns in x.

    Cols 0-2 (x_norm, y_norm, thickness) are never touched.
    """
    deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index_observed.size(1) > 0:
        deg.scatter_add_(
            0,
            edge_index_observed[0],
            torch.ones(edge_index_observed.size(1), dtype=torch.float),
        )
    x = x.clone()
    x[:, 3] = deg
    x[:, 4] = (deg == 1).float()   # is_endpoint
    x[:, 5] = (deg >= 3).float()   # is_junction
    return x


# ── Fix 2: hard negative sampling ────────────────────────────────────────────

def _make_edge_set(edge_index: torch.Tensor) -> set:
    """Canonical (min, max) pair set for fast membership tests."""
    ei = edge_index.numpy()
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in zip(ei[0], ei[1])
    }


def sample_hard_negatives(
    pos: torch.Tensor,
    existing_edges_set: set,
    num_neg: int,
    k_near: int = 10,
    rng: random.Random = None,
) -> list:
    """
    Sample up to num_neg unconnected pairs that are spatially near each other.

    Strategy: for each node i, look at its k_near nearest neighbours in pixel
    space (using pos); keep pairs that are NOT already edges.  Deduplicate,
    shuffle, return up to num_neg pairs.

    Falls back gracefully if the graph has too few nodes or not enough
    near-non-edges — returns however many are available.

    Args:
        pos               : [N, 2] pixel positions (x, y)
        existing_edges_set: set of (min_u, max_v) pairs — real edges to exclude
        num_neg           : desired number of negatives
        k_near            : neighbourhood size to search per anchor node
        rng               : optional seeded random.Random for reproducibility
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for hard negative sampling: pip install scipy")

    N = pos.size(0)
    if N < 2 or num_neg <= 0:
        return []

    if rng is None:
        rng = random.Random()

    tree = cKDTree(pos.numpy())
    candidates = set()

    for i in range(N):
        k = min(k_near + 1, N)
        _, idx = tree.query(pos[i].numpy(), k=k)
        for j in idx[1:]:          # skip self
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if (a, b) not in existing_edges_set:
                candidates.add((a, b))

    candidates = list(candidates)
    rng.shuffle(candidates)
    return candidates[:num_neg]


# ── Core split function ───────────────────────────────────────────────────────

def transductive_split(
    data,
    num_val: float = 0.10,
    num_test: float = 0.10,
    k_near: int = 30,
    seed: int = None,
    num_test_min_edges: int = 4,
) -> dict | None:
    """
    Transductively split one graph's edges and apply both critical fixes.

    Edge partition (80/10/10 by default):
        - message-passing edges: used for GNN convolution
        - val positives: hidden edges for validation scoring
        - test positives: hidden edges for test scoring
        - val/test negatives: hard near-non-edges matching positive counts

    Returns None if the graph is too small to produce a meaningful split
    (< 4 undirected edges).

    Return dict:
        train_data       — PyG Data with observed edges + recomputed x
        val_pos_ei       — [2, n_val] val positive edge_index
        val_neg_ei       — [2, n_val] val hard-negative edge_index
        test_pos_ei      — [2, n_test] test positive edge_index
        test_neg_ei      — [2, n_test] test hard-negative edge_index
        x_observed       — [N, 6] features recomputed from observed edges
        val_labels        — 1/0 labels for [val_pos, val_neg] concatenated
        test_labels       — 1/0 labels for [test_pos, test_neg] concatenated
    """
    n_edges_undirected = data.edge_index.size(1) // 2
    if n_edges_undirected < num_test_min_edges:
        return None

    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    splitter = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
    )

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_d, val_d, test_d = splitter(data)
    except Exception:
        return None

    # Allow num_val=0 and num_test=0 in isolation (one-sided eval / training use)
    if num_test > 0 and test_d.edge_label_index.size(1) == 0:
        return None
    if num_val > 0 and val_d.edge_label_index.size(1) == 0:
        return None

    # ── Fix 1: recompute structural features from observed edges ──────────────
    num_nodes = data.x.size(0)
    x_obs = recompute_structural_features(data.x, train_d.edge_index, num_nodes)
    train_d.x = x_obs

    # ── Fix 2: hard negative sampling ────────────────────────────────────────
    pos = data.pos                               # [N, 2]
    full_edge_set = _make_edge_set(data.edge_index)

    # Positive edge_indices (only the positive half from RandomLinkSplit)
    if num_test > 0:
        test_pos_mask = test_d.edge_label.bool()
        test_pos_ei   = test_d.edge_label_index[:, test_pos_mask]
        n_test = test_pos_ei.size(1)
    else:
        test_pos_ei = torch.zeros(2, 0, dtype=torch.long)
        n_test = 0

    if num_val > 0:
        val_pos_mask = val_d.edge_label.bool()
        val_pos_ei   = val_d.edge_label_index[:, val_pos_mask]
    else:
        val_pos_ei = torch.zeros(2, 0, dtype=torch.long)
    n_val = val_pos_ei.size(1)

    # Also exclude the supervision positives from negative candidates
    sup_set = _make_edge_set(
        torch.cat([val_pos_ei, test_pos_ei], dim=1)
    ) if (n_val + n_test) > 0 else set()
    exclude_set = full_edge_set | sup_set

    all_negs = sample_hard_negatives(
        pos, exclude_set,
        num_neg=(n_val + n_test) * 5 + 50,   # oversample generously
        k_near=k_near,
        rng=rng,
    )

    val_neg_pairs  = all_negs[:n_val]
    test_neg_pairs = all_negs[n_val:n_val + n_test]

    def _pairs_to_ei(pairs):
        if not pairs:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).T  # [2, P]

    val_neg_ei  = _pairs_to_ei(val_neg_pairs)
    test_neg_ei = _pairs_to_ei(test_neg_pairs)

    # Build label tensors: [pos..., neg...]  (1 = real, 0 = fake)
    val_labels  = torch.cat([
        torch.ones(val_pos_ei.size(1)),
        torch.zeros(val_neg_ei.size(1)),
    ])
    test_labels = torch.cat([
        torch.ones(test_pos_ei.size(1)),
        torch.zeros(test_neg_ei.size(1)),
    ])

    val_ei  = torch.cat([val_pos_ei,  val_neg_ei],  dim=1)
    test_ei = torch.cat([test_pos_ei, test_neg_ei], dim=1)

    return {
        'train_data':    train_d,
        'val_ei':        val_ei,
        'val_labels':    val_labels,
        'test_ei':       test_ei,
        'test_labels':   test_labels,
        'val_pos_ei':    val_pos_ei,
        'val_neg_ei':    val_neg_ei,
        'test_pos_ei':   test_pos_ei,
        'test_neg_ei':   test_neg_ei,
        'x_observed':    x_obs,
    }


# ── Batch helper ──────────────────────────────────────────────────────────────

def prepare_dataset(graphs: list, num_val=0.10, num_test=0.10,
                    k_near=10, seed=0) -> list:
    """
    Apply transductive_split to every graph; return list of split dicts.
    Graphs that are too small (< 4 undirected edges) are skipped.
    Every graph gets a deterministic seed derived from the base seed so
    results are reproducible but varied across graphs.
    """
    splits = []
    for i, g in enumerate(graphs):
        s = transductive_split(g, num_val=num_val, num_test=num_test,
                               k_near=k_near, seed=seed + i)
        if s is not None:
            splits.append(s)
    return splits
