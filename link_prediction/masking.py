"""
Graph masking utilities for the topology-understanding tasks.

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

Task 3 — Frontier masking (physically-motivated crack evolution simulation):
    Two complementary strategies that specifically target the GROWTH FRONT of
    the crack (the outermost tips and their connecting edges), rather than
    randomly hiding any part of the graph.

    apply_frontier_mask:
        Hides crack tip NODES (degree-1) and their complete connecting edges.
        Physically: "the crack grew into this region since the last photo; the
        tip node and the last growth segment are both unobserved."
        The complete edge is removed because edges are binary in this graph
        (no half-edge representation).  This is physically conservative —
        the full last growth segment is hidden, making the task harder.

    frontier_edge_split:
        Hides the frontier EDGES only — the edges connected to endpoint nodes.
        The tip node STAYS VISIBLE but becomes isolated (degree → 0).
        Physically: "you can see the crack tip, but the segment connecting it
        back to the established network is obscured (mud, shadow, annotation
        gap).  Predict which visible isolated tip connects where."
        This is the most direct simulation of crack link prediction:
        given a visible tip, predict the direction and target of next growth.
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


# ── Task 3: Frontier masking ──────────────────────────────────────────────────

def apply_frontier_mask(
    data: Data,
    mask_frac: float = 0.30,
    seed: int = None,
) -> tuple:
    """
    Frontier masking — hide crack tip nodes AND their complete connecting edges.

    Physical interpretation
    -----------------------
    A crack tip (endpoint node, degree=1) is where crack propagation occurs.
    Hiding the tip node plus its full connecting edge simulates:
        "This growth segment and its tip formed after the last observation.
         The model must infer from the remaining graph structure where growth
         occurred and from which base node it originated."

    On complete vs. partial edge removal
    -------------------------------------
    Crack segment edges are binary in this representation (present or absent).
    There is no "half-edge" — subdividing an edge would require inserting a
    synthetic midpoint node.  Removing the complete edge is the physically
    correct minimum unit: it hides the full last growth segment.  This is
    conservative (harder for the model) and honest — the edge feature
    path_length still encodes how long that hidden segment was.

    Relationship to apply_node_mask(node_type='endpoint')
    ------------------------------------------------------
    Structurally identical.  The difference is:
      apply_node_mask     : generic "hide nodes, prove the model notices"
      apply_frontier_mask : physically framed as crack growth simulation;
                            also returns hidden_edge_index so the hidden edges
                            can be used directly as positives in the edge task,
                            forming a joint frontier evaluation.

    Why this proves GNN topology understanding
    ------------------------------------------
    After masking, the base node of a hidden growth segment looks like an
    ordinary endpoint (degree reduced by 1).  An MLP sees only the node's
    own (recomputed) features and cannot distinguish "always terminated here"
    from "lost a growth edge."  A GNN can aggregate evidence from neighbours:
    crack width of adjacent segments, tortuosity pattern, branching density —
    structural context that reveals whether a node is an active growth base.

    Parameters
    ----------
    data      : PyG Data object.
    mask_frac : Fraction of endpoint nodes to hide (default 0.30 — slightly
                higher than node-task baseline because frontier edges are fewer).
    seed      : RNG seed for reproducibility.

    Returns
    -------
    masked_data       — PyG Data: tip nodes zeroed, their edges removed,
                        structural features recomputed from remaining edges.
    node_labels       — FloatTensor [N]: 1 = visible node that was the base of
                        a hidden growth segment (lost a frontier-edge neighbour).
    hidden_mask       — BoolTensor [N]: True for hidden tip nodes.
    eval_mask         — BoolTensor [N]: True for visible nodes with ≥ 1 edge.
    hidden_edge_index — LongTensor [2, K]: removed frontier edges (both
                        directions stored); use as positives in joint edge eval.
    """
    rng = np.random.default_rng(seed)

    N          = data.x.size(0)
    x          = data.x.clone()
    edge_index = data.edge_index.clone()

    # 1. Identify crack tip candidates (degree-1 endpoint nodes)
    candidate_ids = (x[:, _COL_IS_ENDPOINT] == 1.0).nonzero(as_tuple=True)[0].numpy()

    # Edge case: no endpoints or empty graph
    if len(candidate_ids) == 0 or edge_index.size(1) == 0:
        empty = torch.zeros(N, dtype=torch.float)
        empty_b = torch.zeros(N, dtype=torch.bool)
        empty_ei = torch.zeros(2, 0, dtype=torch.long)
        masked = Data(x=x, edge_index=edge_index,
                      edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None)
        return masked, empty, empty_b, torch.ones(N, dtype=torch.bool), empty_ei

    # 2. Sample which tips to hide (these represent "grew since last observation")
    n_hide   = max(1, int(len(candidate_ids) * mask_frac))
    n_hide   = min(n_hide, len(candidate_ids))
    hidden_ids = rng.choice(candidate_ids, size=n_hide, replace=False)
    hidden_set = set(hidden_ids.tolist())

    hidden_mask = torch.zeros(N, dtype=torch.bool)
    hidden_mask[list(hidden_set)] = True

    # 3. Build adjacency to label the base nodes of hidden growth segments
    adj = {i: [] for i in range(N)}
    ei_np = edge_index.numpy()
    for u, v in zip(ei_np[0], ei_np[1]):
        adj[u].append(v)

    # Label visible nodes that were connected to a hidden tip:
    # these are the "growth bases" — the last visible node before the hidden tip.
    node_labels = torch.zeros(N, dtype=torch.float)
    for h in hidden_set:
        for neighbour in adj[h]:
            if neighbour not in hidden_set:
                node_labels[neighbour] = 1.0

    # 4. Zero features of hidden tip nodes
    x[list(hidden_set)] = 0.0

    # 5. Remove ALL edges incident to hidden tips.
    #    Each hidden tip has degree=1, so exactly one edge per tip is removed.
    #    That one edge IS the last growth segment — its complete removal is
    #    intentional and physically motivated (see docstring).
    src, dst   = edge_index[0], edge_index[1]
    keep_mask  = ~(hidden_mask[src] | hidden_mask[dst])
    new_ei     = edge_index[:, keep_mask]
    new_ea     = data.edge_attr[keep_mask] if data.edge_attr is not None else None

    # Collect the removed frontier edges for joint edge-task use
    hidden_edge_index = edge_index[:, ~keep_mask]

    # 6. Recompute structural features from post-masking edge set.
    #    Without this, the base node's degree still reflects the hidden edge,
    #    leaking its label to the model.
    x = recompute_structural_features(x, new_ei, N)
    x[list(hidden_set)] = 0.0  # re-zero hidden nodes after recompute

    masked_data = Data(x=x, edge_index=new_ei, edge_attr=new_ea, num_nodes=N)

    # 7. Eval mask: visible nodes that still have at least one edge
    has_edge = torch.zeros(N, dtype=torch.bool)
    if new_ei.size(1) > 0:
        has_edge[new_ei[0]] = True
        has_edge[new_ei[1]] = True
    eval_mask = (~hidden_mask) & has_edge

    return masked_data, node_labels, hidden_mask, eval_mask, hidden_edge_index


def frontier_edge_split(
    data: Data,
    mask_frac: float = 0.30,
    k_near: int = 30,
    seed: int = None,
) -> dict | None:
    """
    Frontier edge split — hide frontier edges while keeping tip nodes visible.

    A frontier edge is an edge where at least one endpoint has degree=1 (a
    crack tip).  After hiding, the tip node remains in the graph but becomes
    isolated (no connecting edges).  The model must predict which isolated
    tip reconnects to which base node — directly simulating:
        "Given the current crack network and its visible tips, which tip will
         grow and in which direction?"

    Why tip nodes stay visible (unlike apply_frontier_mask)
    --------------------------------------------------------
    In a real road survey, you CAN see the crack tip in the image (the end of
    the visible white crack pixel region).  What you CANNOT see is the new
    segment that formed since the last survey.  This split models exactly that:
    the tip exists, the connecting growth segment is hidden.

    Why complete edge removal (not partial)
    ----------------------------------------
    Edges are binary in this graph.  The full segment from junction/midpoint
    to the crack tip is removed.  Edge features (path_length, avg_thickness)
    encoded on the hidden edge capture how significant that growth segment was.

    Hard negatives
    --------------
    Uses the same KD-tree hard-negative protocol as transductive_split:
    spatially-near non-edges (k_near=30) replace trivially-far random negatives,
    so the model cannot win by proximity alone.

    Parameters
    ----------
    data      : PyG Data object with data.pos ([N, 2] pixel coordinates).
    mask_frac : Fraction of frontier (tip-adjacent) undirected edges to hide.
    k_near    : Neighbourhood size for hard negative sampling.
    seed      : RNG seed.

    Returns None if the graph has fewer than 2 undirected frontier edges or
    if data.pos is not available.

    Returns dict with keys
    ----------------------
    train_data  — PyG Data for message passing (non-frontier + kept frontier
                  edges; structural features recomputed).
    label_ei    — [2, P+N] edge_index of positive + negative candidate pairs
                  (one direction only — undirected scoring).
    labels      — FloatTensor [P+N]: 1 = real frontier edge, 0 = hard negative.
    n_frontier  — number of frontier edges hidden (= number of positives).
    """
    import random as _random
    from .splits import recompute_structural_features, sample_hard_negatives, _make_edge_set

    if data.pos is None:
        return None

    rng   = _random.Random(seed)
    N     = data.x.size(0)
    x     = data.x.clone()
    ei    = data.edge_index          # [2, 2E]
    ea    = data.edge_attr           # [2E, 7] or None

    # 1. Identify frontier edges: at least one endpoint is a crack tip (degree=1)
    is_ep = x[:, _COL_IS_ENDPOINT]  # [N]
    src, dst = ei[0], ei[1]
    is_frontier_col = (is_ep[src] == 1.0) | (is_ep[dst] == 1.0)  # [2E] bool

    # 2. Build undirected frontier pairs: canonical key (min_u, max_v) → [col_a, col_b]
    #    PyG stores both (u,v) and (v,u); we need to track both column indices
    #    so we can hide both directions together.
    ei_np = ei.numpy()
    pair_to_cols: dict = {}
    for col in range(ei.size(1)):
        if not is_frontier_col[col].item():
            continue
        u, v = int(ei_np[0, col]), int(ei_np[1, col])
        key  = (min(u, v), max(u, v))
        pair_to_cols.setdefault(key, []).append(col)

    # Keep only pairs with both directions present (skip degenerate self-loops)
    frontier_pairs = [(key, cols) for key, cols in pair_to_cols.items()
                      if len(cols) == 2]

    if len(frontier_pairs) < 2:
        return None

    # 3. Sample fraction to hide
    n_hide = max(1, int(len(frontier_pairs) * mask_frac))
    idx_list = list(range(len(frontier_pairs)))
    rng.shuffle(idx_list)
    hide_idx = set(idx_list[:n_hide])

    # Determine which edge_index columns to hide and collect positive pairs
    hide_cols: set = set()
    pos_pairs: list = []
    for i, (key, cols) in enumerate(frontier_pairs):
        if i in hide_idx:
            hide_cols.update(cols)
            pos_pairs.append(key)   # canonical (min_u, max_v) for edge task

    if not pos_pairs:
        return None

    # 4. Build keep mask and split edge_index / edge_attr
    keep_mask = torch.ones(ei.size(1), dtype=torch.bool)
    for col in hide_cols:
        keep_mask[col] = False

    mp_ei = ei[:, keep_mask]
    mp_ea = ea[keep_mask] if ea is not None else None

    # 5. Recompute structural features from message-passing edges.
    #    Tip nodes that lost their only edge now show degree=0 and
    #    is_endpoint=0 — an honest representation of their observed state.
    x_obs = recompute_structural_features(x, mp_ei, N)
    train_data = Data(x=x_obs, edge_index=mp_ei, edge_attr=mp_ea, num_nodes=N)

    # 6. Hard negative sampling: spatially-near non-edges
    full_edge_set = _make_edge_set(ei)
    neg_pairs = sample_hard_negatives(
        data.pos,
        full_edge_set,
        num_neg=len(pos_pairs) * 5 + 10,
        k_near=k_near,
        rng=rng,
    )
    neg_pairs = neg_pairs[:len(pos_pairs)]   # match positive count (1:1 ratio)

    if not neg_pairs:
        return None

    # 7. Build label tensors (one direction per pair — undirected scoring)
    def _pairs_to_ei(pairs):
        return torch.tensor(pairs, dtype=torch.long).T  # [2, P]

    pos_ei_t = _pairs_to_ei(pos_pairs)   # [2, n_hide]
    neg_ei_t = _pairs_to_ei(neg_pairs)   # [2, n_neg]

    label_ei = torch.cat([pos_ei_t, neg_ei_t], dim=1)
    labels   = torch.cat([
        torch.ones(pos_ei_t.size(1),  dtype=torch.float),
        torch.zeros(neg_ei_t.size(1), dtype=torch.float),
    ])

    return {
        'train_data': train_data,
        'label_ei':   label_ei,
        'labels':     labels,
        'n_frontier': len(pos_pairs),
    }


def is_valid_for_frontier_task(data: Data, min_frontier_edges: int = 2) -> bool:
    """
    Graph must have at least min_frontier_edges undirected frontier edges
    (edges connected to endpoint/degree-1 nodes) for a meaningful split.
    """
    is_ep    = data.x[:, _COL_IS_ENDPOINT]
    src, dst = data.edge_index[0], data.edge_index[1]
    n_frontier_cols = int(
        ((is_ep[src] == 1.0) | (is_ep[dst] == 1.0)).sum().item()
    )
    n_frontier_undirected = n_frontier_cols // 2
    return n_frontier_undirected >= min_frontier_edges
