"""
Graph masking utilities for the two topology-understanding tasks.

Task 1 — Edge masking (link prediction):
    Wraps PyG's RandomLinkSplit.  Removes a fraction of edges; the model must
    predict which disconnected node pairs are actually connected.
    Analogous to a crack covered by mud: the physical crack exists but is
    hidden from the sensor.

Task 2 — Node masking (missing-node prediction):
    Removes a random subset of nodes (filtered by node_type) and all their
    incident edges.  Labels remaining visible nodes as:
        1  — was connected to a removed node (has a hidden neighbour)
        0  — no hidden neighbours
    The model must identify topologically incomplete nodes without ever seeing
    the removed node.

    node_type options:
        'endpoint' (default) — masks degree-1 nodes (crack tips).
                               Baseline / research-justified strategy: crack tips
                               are where evolution happens, so hiding them simulates
                               a crack tip that existed in the past.
        'junction'           — masks degree-3+ nodes (branching points).
                               Tests whether the model understands crack branching
                               topology; a harder structural task.
        'random'             — masks any node regardless of structural role.
                               Pure-random ablation baseline with no topological bias.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from .splits import recompute_structural_features

# Node feature column indices (matches image_to_graph/features.py)
_COL_IS_ENDPOINT = 4   # 1.0 if degree == 1
_COL_IS_JUNCTION = 5   # 1.0 if degree >= 3


# ── Task 1: Edge masking ──────────────────────────────────────────────────────

def make_edge_splitter(num_val: float = 0.1, num_test: float = 0.2) -> RandomLinkSplit:
    """
    Returns a configured RandomLinkSplit transform.
    Call splitter(graph) → (train_data, val_data, test_data).
    """
    return RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,   # balanced 1:1 positive:negative
    )


# ── Task 2: Node masking ──────────────────────────────────────────────────────

def apply_node_mask(
    data: Data,
    mask_frac: float = 0.20,
    seed: int = None,
    node_type: str = 'endpoint',
) -> tuple:
    """
    Hides a random fraction of nodes (filtered by node_type) from the graph.

    Parameters
    ----------
    data      : PyG Data object with node features x [N, 6].
    mask_frac : Fraction of the candidate pool to hide (0.0 – 1.0).
    seed      : RNG seed for reproducibility.
    node_type : Which nodes are candidates for hiding:
                  'endpoint' — degree-1 nodes (crack tips).   [DEFAULT / BASELINE]
                  'junction' — degree-3+ nodes (branch pts).  [ablation]
                  'random'   — any node.                      [ablation]

    Returns
    -------
    masked_data  — PyG Data with hidden nodes zeroed and their edges removed.
    node_labels  — FloatTensor [N], 1 = this visible node had a hidden neighbour.
    hidden_mask  — BoolTensor [N], True for nodes that were hidden.
    eval_mask    — BoolTensor [N], True for nodes to evaluate (visible & non-isolated).
    """
    rng = np.random.default_rng(seed)

    N = data.x.size(0)
    x = data.x.clone()
    edge_index = data.edge_index.clone()

    # 1. Identify candidate nodes to hide based on node_type
    if node_type == 'endpoint':
        candidate_ids = (x[:, _COL_IS_ENDPOINT] == 1.0).nonzero(as_tuple=True)[0].numpy()
    elif node_type == 'junction':
        candidate_ids = (x[:, _COL_IS_JUNCTION] == 1.0).nonzero(as_tuple=True)[0].numpy()
    elif node_type == 'random':
        candidate_ids = np.arange(N)
    else:
        raise ValueError(f"node_type must be 'endpoint', 'junction', or 'random'; got {node_type!r}")

    if len(candidate_ids) == 0 or data.edge_index.size(1) == 0:
        node_labels = torch.zeros(N, dtype=torch.float)
        hidden_mask = torch.zeros(N, dtype=torch.bool)
        eval_mask   = torch.ones(N, dtype=torch.bool)
        masked_data = Data(x=x, edge_index=edge_index,
                           edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None)
        return masked_data, node_labels, hidden_mask, eval_mask

    # 2. Sample which candidates to hide
    n_hide = max(1, int(len(candidate_ids) * mask_frac))
    n_hide = min(n_hide, len(candidate_ids))
    hidden_ids = rng.choice(candidate_ids, size=n_hide, replace=False)
    hidden_set = set(hidden_ids.tolist())

    hidden_mask = torch.zeros(N, dtype=torch.bool)
    hidden_mask[list(hidden_set)] = True

    # 3. Build node labels: 1 for visible nodes connected to a hidden node
    node_labels = torch.zeros(N, dtype=torch.float)
    adj = {i: [] for i in range(N)}
    ei = edge_index.numpy()
    for u, v in zip(ei[0], ei[1]):
        adj[u].append(v)

    for h in hidden_set:
        for neighbour in adj[h]:
            if neighbour not in hidden_set:
                node_labels[neighbour] = 1.0

    # 4. Zero out features of hidden nodes
    x[list(hidden_set)] = 0.0

    # 5. Remove edges incident to hidden nodes
    src, dst = edge_index[0], edge_index[1]
    keep = ~(hidden_mask[src] | hidden_mask[dst])
    new_edge_index = edge_index[:, keep]
    new_edge_attr  = data.edge_attr[keep] if data.edge_attr is not None else None

    # 6. Recompute degree / is_endpoint / is_junction (cols 3-5) for ALL nodes
    # from the post-masking edge set. This prevents the GNN from reading a
    # visible node's stale degree and inferring "degree > visible edges → hidden
    # neighbour exists" without doing any topology reasoning.
    x = recompute_structural_features(x, new_edge_index, N)
    # Re-zero all features of hidden nodes (cols 0-2 were already zeroed in
    # step 4; recompute correctly zeroes cols 3-5 since hidden nodes have no
    # edges, but we explicitly enforce the full zero to be safe).
    x[list(hidden_set)] = 0.0

    masked_data = Data(
        x=x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=N,
    )

    # Eval mask: visible nodes that still have at least one edge
    visible = ~hidden_mask
    has_edge = torch.zeros(N, dtype=torch.bool)
    if new_edge_index.size(1) > 0:
        has_edge[new_edge_index[0]] = True
        has_edge[new_edge_index[1]] = True
    eval_mask = visible & has_edge

    return masked_data, node_labels, hidden_mask, eval_mask


# ── Minimum graph size checks ─────────────────────────────────────────────────

def is_valid_for_edge_task(data: Data, min_edges: int = 4) -> bool:
    """Graph must have enough edges for a meaningful train/val/test split."""
    return data.edge_index.size(1) >= min_edges * 2   # *2 — both directions stored


def is_valid_for_node_task(data: Data, min_nodes: int = 2,
                           node_type: str = 'endpoint') -> bool:
    """
    Graph must have at least min_nodes candidate nodes to mask.
    node_type must match what will be passed to apply_node_mask.
    """
    if node_type == 'endpoint':
        return int((data.x[:, _COL_IS_ENDPOINT] == 1.0).sum().item()) >= min_nodes
    elif node_type == 'junction':
        return int((data.x[:, _COL_IS_JUNCTION] == 1.0).sum().item()) >= min_nodes
    elif node_type == 'random':
        return data.x.size(0) >= min_nodes
    return False
