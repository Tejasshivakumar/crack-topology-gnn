"""
Graph masking utilities for the two topology-understanding tasks.

Task 1 — Edge masking (link prediction):
    Wraps PyG's RandomLinkSplit.  Removes a fraction of edges; the model must
    predict which disconnected node pairs are actually connected.
    Analogous to a crack covered by mud: the physical crack exists but is
    hidden from the sensor.

Task 2 — Node masking (missing-node prediction):
    Removes a random subset of endpoint nodes (degree == 1) and all their
    incident edges.  Labels remaining visible nodes as:
        1  — was connected to a removed endpoint (has a hidden neighbour)
        0  — no hidden neighbours
    The model must identify topologically incomplete nodes without ever seeing
    the removed node.  This proves the GNN understands what a complete crack
    network looks like.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit


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

def apply_node_mask(data: Data, mask_frac: float = 0.20, seed: int = None) -> tuple:
    """
    Hides a random fraction of endpoint nodes from the graph.

    Steps:
      1. Identify endpoint nodes (is_endpoint feature == 1, i.e., col 4 of x).
      2. Randomly select `mask_frac` of them as hidden.
      3. Zero out their node features (makes them invisible to message passing).
      4. Remove all edges incident to hidden nodes.
      5. Build node-level labels: 1 if the node had a hidden neighbour, else 0.

    Returns:
        masked_data  — PyG Data with hidden nodes zeroed and their edges removed
        node_labels  — FloatTensor [N] of binary labels (1 = has hidden neighbour)
        hidden_mask  — BoolTensor [N], True for nodes that were hidden
        eval_mask    — BoolTensor [N], True for nodes to evaluate (not hidden, non-isolated)
    """
    rng = np.random.default_rng(seed)

    N = data.x.size(0)
    x = data.x.clone()
    edge_index = data.edge_index.clone()

    # 1. Find endpoint nodes (is_endpoint == 1)
    endpoint_ids = (x[:, 4] == 1.0).nonzero(as_tuple=True)[0].numpy()

    if len(endpoint_ids) == 0 or data.edge_index.size(1) == 0:
        # Nothing to mask — return as-is with all-zero labels
        node_labels = torch.zeros(N, dtype=torch.float)
        hidden_mask = torch.zeros(N, dtype=torch.bool)
        eval_mask   = torch.ones(N, dtype=torch.bool)
        masked_data = Data(x=x, edge_index=edge_index,
                           edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None)
        return masked_data, node_labels, hidden_mask, eval_mask

    # 2. Sample which endpoints to hide
    n_hide = max(1, int(len(endpoint_ids) * mask_frac))
    hidden_ids = rng.choice(endpoint_ids, size=n_hide, replace=False)
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

    masked_data = Data(
        x=x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=N,
    )

    # Eval mask: evaluate only on visible, non-isolated nodes
    visible = ~hidden_mask
    has_edge = torch.zeros(N, dtype=torch.bool)
    if new_edge_index.size(1) > 0:
        has_edge[new_edge_index[0]] = True
        has_edge[new_edge_index[1]] = True
    eval_mask = visible & has_edge

    return masked_data, node_labels, hidden_mask, eval_mask


# ── Minimum graph size check ──────────────────────────────────────────────────

def is_valid_for_edge_task(data: Data, min_edges: int = 4) -> bool:
    """Graph must have enough edges for a meaningful train/val/test split."""
    return data.edge_index.size(1) >= min_edges * 2   # *2 because both directions stored


def is_valid_for_node_task(data: Data, min_endpoints: int = 2) -> bool:
    """Graph must have at least min_endpoints endpoint nodes to mask any."""
    return int((data.x[:, 4] == 1.0).sum().item()) >= min_endpoints
