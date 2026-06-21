"""
Evaluation metrics for both tasks.

─── BASELINE (unchanged) ────────────────────────────────────────────────────
Edge task metrics:
    AUC-ROC  — overall discriminative ability
    AP       — Average Precision (area under PR curve); better for imbalanced sets
    Hits@K   — fraction of true edges ranked in top-K predictions per graph

Node task metrics:
    AUC-ROC         — discriminative ability for missing-neighbour classification
    AP              — Average Precision
    F1 @ 0.5        — harmonic mean at fixed threshold (reported for comparability)
    F1 @ opt thresh — F1 at the threshold that maximises it on the eval set
    Balanced Acc    — (sensitivity + specificity) / 2; robust to class imbalance

─── ABLATION ────────────────────────────────────────────────────────────────
ablation_node_masking  — sweep mask_frac × node_type
ablation_edge_masking  — sweep edge test fraction
run_ablation_study     — combined runner that prints a formatted table
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
)

from .masking import (
    make_edge_splitter, apply_node_mask,
    is_valid_for_edge_task, is_valid_for_node_task,
)
from .splits import recompute_structural_features


# ── Shared helper ─────────────────────────────────────────────────────────────

def _find_optimal_threshold(labels: np.ndarray, probs: np.ndarray,
                             n_steps: int = 81) -> tuple:
    """
    Find the decision threshold that maximises F1 on the given predictions.

    Scans `n_steps` evenly-spaced thresholds in [0.05, 0.95].
    Returns (best_threshold, best_f1).
    """
    thresholds = np.linspace(0.05, 0.95, n_steps)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t), float(best_f1)


def _node_metrics_from_arrays(all_labels: np.ndarray,
                               all_probs: np.ndarray) -> dict:
    """Compute the full node-task metric suite from pooled arrays."""
    opt_t, f1_opt = _find_optimal_threshold(all_labels, all_probs)
    preds_fixed = (all_probs >= 0.5).astype(int)
    preds_opt   = (all_probs >= opt_t).astype(int)
    return {
        'node_auc':          float(roc_auc_score(all_labels, all_probs)),
        'node_ap':           float(average_precision_score(all_labels, all_probs)),
        'node_f1':           float(f1_score(all_labels, preds_fixed, zero_division=0)),
        'node_f1_opt':       float(f1_opt),
        'node_thresh_opt':   float(opt_t),
        'node_bal_acc':      float(balanced_accuracy_score(all_labels, preds_opt)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE — do NOT modify signatures or return keys; downstream code depends
# on them.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_mrr(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """
    Mean Reciprocal Rank within a batch of scored (positive + negative) pairs.

    For each positive pair, rank = #(negatives scored >= positive) + 1.
    MRR = mean(1 / rank) over all positives.
    Returns None when no valid pairs are available.
    """
    pos_scores = probs[labels == 1]
    neg_scores = probs[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    mrr_vals = [1.0 / (float(np.sum(neg_scores >= ps)) + 1.0) for ps in pos_scores]
    return float(np.mean(mrr_vals))


def evaluate_edge_task(encoder, edge_pred, dataset, device):
    """
    Evaluate link prediction on a held-out dataset.

    Uses hard negatives (k_near=30 spatially-near non-edges) and feature
    recompute after split. Reports AUC-ROC and AP only — Hits@K and MRR
    are excluded because many crack graphs are too small to guarantee a
    negative pool of size K, making those metrics mechanically inflated.
    """
    import warnings
    from .splits import transductive_split
    from .metrics import graph_metrics, aggregate

    encoder.eval(); edge_pred.eval()
    per_graph = []

    with torch.no_grad():
        for g in dataset:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                s = transductive_split(g, num_val=0.0, num_test=0.20, seed=0)
            if s is None:
                continue

            td  = s['train_data']
            x   = td.x.float().to(device)
            ei  = td.edge_index.to(device)
            ea  = td.edge_attr.float().to(device) if td.edge_attr is not None else None
            eli = s['test_ei'].to(device)

            z   = encoder(x, ei, ea)
            sc  = torch.sigmoid(edge_pred(z, eli)).cpu().numpy()
            lbl = s['test_labels'].numpy()

            per_graph.append(graph_metrics(sc, lbl, hits_ks=()))

    agg = aggregate(per_graph, hits_ks=())
    return {
        'edge_auc': agg.get('auc_mean') or 0.0,
        'edge_ap':  agg.get('ap_mean')  or 0.0,
    }


def evaluate_node_task(encoder, node_pred, dataset, device,
                       mask_frac=0.20, threshold=0.5, seed=0):
    """
    Evaluate missing-node prediction on a held-out dataset.
    Baseline strategy: endpoint nodes, 20% masking fraction.

    Now returns extended metrics (node_f1_opt, node_thresh_opt, node_bal_acc)
    in addition to the original keys (node_auc, node_ap, node_f1) — fully
    backward-compatible.
    """
    encoder.eval(); node_pred.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for i, g in enumerate(dataset):
            if not is_valid_for_node_task(g):
                continue

            masked, node_labels, _, eval_mask = apply_node_mask(
                g, mask_frac=mask_frac, seed=seed + i
            )
            if eval_mask.sum() == 0:
                continue

            labels_ev = node_labels[eval_mask]
            if len(torch.unique(labels_ev)) < 2:
                continue

            x   = masked.x.to(device)
            ei  = masked.edge_index.to(device)
            ea  = masked.edge_attr.to(device) if masked.edge_attr is not None else None

            z     = encoder(x, ei, ea)
            probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()
            lbls  = labels_ev.numpy()

            all_probs.extend(probs)
            all_labels.extend(lbls)

    if not all_labels:
        return {
            'node_auc': 0.5, 'node_ap': 0.0, 'node_f1': 0.0,
            'node_f1_opt': 0.0, 'node_thresh_opt': 0.5, 'node_bal_acc': 0.5,
        }

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = _node_metrics_from_arrays(all_labels, all_probs)
    # Keep the fixed-threshold F1 under the original key for backward compat
    metrics['node_f1'] = float(f1_score(all_labels,
                                         (all_probs >= threshold).astype(int),
                                         zero_division=0))
    return metrics


def full_evaluation(encoder, edge_pred, node_pred, dataset, device):
    """Baseline combined evaluation — endpoint masking at 20%."""
    edge_metrics = evaluate_edge_task(encoder, edge_pred, dataset, device)
    node_metrics = evaluate_node_task(encoder, node_pred, dataset, device)
    return {**edge_metrics, **node_metrics}


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION — parametric sweeps over mask_frac and node_type
# ══════════════════════════════════════════════════════════════════════════════

def ablation_node_masking(
    encoder, node_pred, dataset, device,
    mask_fracs=(0.10, 0.20, 0.30, 0.40, 0.50),
    node_types=('endpoint', 'junction', 'random'),
    seed: int = 0,
) -> dict:
    """
    Sweep node masking across fractions and node types.

    For each (mask_frac, node_type) combination, pool predictions across the
    dataset and compute the full metric suite.

    Parameters
    ----------
    mask_fracs  : Iterable of fractions to evaluate.
    node_types  : Iterable of node_type strings ('endpoint', 'junction', 'random').
    seed        : Base seed; per-graph seed = seed + graph_index.

    Returns
    -------
    results : dict keyed by (mask_frac, node_type) → metrics dict containing:
                node_auc, node_ap, node_f1 (@ 0.5), node_f1_opt,
                node_thresh_opt, node_bal_acc, n_graphs (graphs contributed).
    """
    encoder.eval(); node_pred.eval()
    results = {}

    for node_type in node_types:
        for mask_frac in mask_fracs:
            all_probs, all_labels = [], []
            n_graphs = 0

            with torch.no_grad():
                for i, g in enumerate(dataset):
                    if not is_valid_for_node_task(g, node_type=node_type):
                        continue

                    masked, node_labels, _, eval_mask = apply_node_mask(
                        g, mask_frac=mask_frac, seed=seed + i, node_type=node_type,
                    )
                    if eval_mask.sum() == 0:
                        continue

                    labels_ev = node_labels[eval_mask]
                    if len(torch.unique(labels_ev)) < 2:
                        continue

                    x  = masked.x.to(device)
                    ei = masked.edge_index.to(device)
                    ea = masked.edge_attr.to(device) if masked.edge_attr is not None else None

                    z     = encoder(x, ei, ea)
                    probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()
                    lbls  = labels_ev.numpy()

                    all_probs.extend(probs)
                    all_labels.extend(lbls)
                    n_graphs += 1

            key = (mask_frac, node_type)
            if not all_labels or len(np.unique(all_labels)) < 2:
                results[key] = {
                    'node_auc': 0.5, 'node_ap': 0.0,
                    'node_f1': 0.0, 'node_f1_opt': 0.0,
                    'node_thresh_opt': 0.5, 'node_bal_acc': 0.5,
                    'n_graphs': n_graphs,
                }
                continue

            all_probs  = np.array(all_probs)
            all_labels = np.array(all_labels)

            m = _node_metrics_from_arrays(all_labels, all_probs)
            m['node_f1']  = float(f1_score(all_labels,
                                            (all_probs >= 0.5).astype(int),
                                            zero_division=0))
            m['n_graphs'] = n_graphs
            results[key] = m

    return results


def ablation_edge_masking(
    encoder, edge_pred, dataset, device,
    test_fracs=(0.10, 0.20, 0.30, 0.40, 0.50),
    hits_k=(10, 20),
) -> dict:
    """
    Sweep edge masking across test fractions.

    For each test_frac, RandomLinkSplit hides that fraction of edges; the model
    predicts them using the remaining (1 - test_frac) visible edges.

    Returns
    -------
    results : dict keyed by test_frac → metrics dict containing:
                edge_auc, edge_ap, edge_hits@K, n_graphs.
    """
    encoder.eval(); edge_pred.eval()
    results = {}

    for test_frac in test_fracs:
        splitter = make_edge_splitter(num_val=0.0, num_test=test_frac)
        all_probs, all_labels = [], []
        hits_at = {k: [] for k in hits_k}
        n_graphs = 0

        with torch.no_grad():
            for g in dataset:
                if not is_valid_for_edge_task(g):
                    continue
                try:
                    train_d, _, test_d = splitter(g)
                except Exception:
                    continue
                if test_d.edge_label_index.size(1) == 0:
                    continue

                x   = test_d.x.to(device)
                ei  = train_d.edge_index.to(device)
                ea  = train_d.edge_attr.to(device) if train_d.edge_attr is not None else None
                eli = test_d.edge_label_index.to(device)

                z     = encoder(x, ei, ea)
                probs = torch.sigmoid(edge_pred(z, eli)).cpu().numpy()
                lbls  = test_d.edge_label.numpy()

                if len(np.unique(lbls)) < 2:
                    continue

                all_probs.extend(probs)
                all_labels.extend(lbls)
                n_graphs += 1

                pos_mask = lbls == 1
                n_pos    = pos_mask.sum()
                for k in hits_k:
                    if n_pos == 0:
                        continue
                    top_k_idx = np.argsort(probs)[::-1][:k]
                    hits_at[k].append(
                        float(pos_mask[top_k_idx].sum()) / float(n_pos)
                    )

        if not all_labels:
            results[test_frac] = {
                'edge_auc': 0.5, 'edge_ap': 0.0, 'n_graphs': 0,
                **{f'edge_hits@{k}': 0.0 for k in hits_k},
            }
            continue

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)

        m = {
            'edge_auc':  float(roc_auc_score(all_labels, all_probs)),
            'edge_ap':   float(average_precision_score(all_labels, all_probs)),
            'n_graphs':  n_graphs,
        }
        for k in hits_k:
            m[f'edge_hits@{k}'] = float(np.mean(hits_at[k])) if hits_at[k] else 0.0
        results[test_frac] = m

    return results


def run_ablation_study(
    encoder, edge_pred, node_pred, dataset, device,
    mask_fracs=(0.10, 0.20, 0.30, 0.40, 0.50),
    node_types=('endpoint', 'junction', 'random'),
    seed: int = 0,
) -> dict:
    """
    Run the full ablation study and print formatted result tables.

    Covers:
      1. Node masking ablation — mask_frac × node_type
      2. Edge masking ablation — test_frac sweep

    Returns
    -------
    {
        'node': { (mask_frac, node_type): metrics_dict, ... },
        'edge': { test_frac: metrics_dict, ... },
    }
    """
    print('\n' + '═' * 72)
    print('  ABLATION: Node Masking — mask_frac × node_type')
    print('═' * 72)
    print(f"  Strategies : {node_types}")
    print(f"  Fractions  : {mask_fracs}")
    print(f"  Dataset    : {len(dataset)} graphs\n")

    node_results = ablation_node_masking(
        encoder, node_pred, dataset, device,
        mask_fracs=mask_fracs, node_types=node_types, seed=seed,
    )

    # Print node ablation table — one sub-table per node_type
    for nt in node_types:
        print(f"  ── node_type = '{nt}' ──")
        print(f"  {'frac':>6}  {'AUC-ROC':>8}  {'Avg Prec':>9}  "
              f"{'F1@0.5':>7}  {'F1@opt':>7}  {'Bal-Acc':>8}  {'Thresh':>7}  {'Graphs':>6}")
        print('  ' + '-' * 66)
        for frac in mask_fracs:
            m = node_results.get((frac, nt), {})
            marker = ' ◄ baseline' if (frac == 0.20 and nt == 'endpoint') else ''
            print(f"  {frac:>6.2f}  {m.get('node_auc', 0):>8.4f}  "
                  f"{m.get('node_ap', 0):>9.4f}  {m.get('node_f1', 0):>7.4f}  "
                  f"{m.get('node_f1_opt', 0):>7.4f}  {m.get('node_bal_acc', 0):>8.4f}  "
                  f"{m.get('node_thresh_opt', 0.5):>7.3f}  "
                  f"{m.get('n_graphs', 0):>6}{marker}")
        print()

    print('═' * 72)
    print('  ABLATION: Edge Masking — test fraction sweep')
    print('═' * 72)

    edge_results = ablation_edge_masking(
        encoder, edge_pred, dataset, device,
        test_fracs=mask_fracs,
    )

    print(f"  {'frac':>6}  {'AUC-ROC':>8}  {'Avg Prec':>9}  "
          f"{'Hits@10':>8}  {'Hits@20':>8}  {'Graphs':>6}")
    print('  ' + '-' * 52)
    for frac in mask_fracs:
        m = edge_results.get(frac, {})
        marker = ' ◄ baseline' if frac == 0.20 else ''
        print(f"  {frac:>6.2f}  {m.get('edge_auc', 0):>8.4f}  "
              f"{m.get('edge_ap', 0):>9.4f}  {m.get('edge_hits@10', 0):>8.4f}  "
              f"{m.get('edge_hits@20', 0):>8.4f}  {m.get('n_graphs', 0):>6}{marker}")
    print()

    return {'node': node_results, 'edge': edge_results}


# ══════════════════════════════════════════════════════════════════════════════
# HEADLINE TABLE — all approaches on the same splits (fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

def headline_table(
    splits: list,
    graphs: list,
    encoder=None,
    edge_pred=None,
    device: torch.device = torch.device('cpu'),
    hits_ks: tuple = (20, 50),
) -> dict:
    """
    Evaluate every approach on the SAME pre-computed splits and return the
    headline results table.

    Approaches included:
        Tier 1  : Common Neighbors (CN), Adamic-Adar (AA), Resource Allocation (RA)
        Tier 1.5: Coordinates-only baseline
        Tier 2  : GNN encoder (if encoder + edge_pred are provided)

    Args:
        splits    : list of dicts from splits.prepare_dataset()
        graphs    : list of original PyG Data objects (same order as splits)
        encoder   : trained GNN encoder (optional; set None to skip Tier 2)
        edge_pred : trained edge predictor (optional)
        device    : torch device for GNN inference

    Returns:
        dict mapping approach_name → aggregated metrics dict
          {auc_mean, auc_std, ap_mean, ap_std, hits@K_mean, hits@K_std,
           mrr_mean, mrr_std, n_graphs}

    All metrics are AUC-ROC, AP (primary), Hits@K, MRR — per-graph-averaged.
    AUC-ROC is always reported for comparability with prior work.
    """
    from .heuristics import score_pairs as heuristic_score
    from .baselines  import coords_score
    from .metrics    import graph_metrics, aggregate

    # Map splits back to their source graphs (same order)
    # splits may be fewer than graphs (small graphs are dropped)
    # We need the original graph for pos (coordinates baseline) and for
    # constructing the NX graph (heuristics).
    # splits[i]['train_data'].x is already recomputed — use it.

    # Build a lookup from split index to original graph
    # Note: prepare_dataset() preserves order so splits[i] corresponds
    # to graphs that passed the size filter, in order.
    graph_idx = 0
    split_to_graph = {}
    g_iter = list(graphs)
    used = []
    gi = 0
    for s in splits:
        # Match by looking for a graph with the right node count
        n = s['train_data'].x.size(0)
        while gi < len(g_iter) and g_iter[gi].x.size(0) != n:
            gi += 1
        if gi < len(g_iter):
            used.append(gi)
            gi += 1
        else:
            used.append(None)

    def _get_graph(i):
        idx = used[i] if i < len(used) else None
        return g_iter[idx] if idx is not None else None

    approaches = {}

    # ── Tier 1: structural heuristics ────────────────────────────────────────
    for method, label in [('cn', 'Common Neighbors'),
                           ('aa', 'Adamic-Adar'),
                           ('ra', 'Resource Allocation')]:
        per_graph = []
        for i, s in enumerate(splits):
            try:
                val_s  = heuristic_score(s['train_data'], s['val_ei'],  method)
                test_s = heuristic_score(s['train_data'], s['test_ei'], method)
                per_graph.append(
                    graph_metrics(test_s, s['test_labels'].numpy(), hits_ks=hits_ks)
                )
            except Exception:
                per_graph.append(None)
        approaches[label] = aggregate(per_graph, hits_ks=hits_ks)

    # ── Tier 1.5: coordinates-only ────────────────────────────────────────────
    per_graph = []
    for i, s in enumerate(splits):
        g = _get_graph(i)
        if g is None or not hasattr(g, 'pos'):
            per_graph.append(None)
            continue
        try:
            test_s = coords_score(g.pos, s['test_ei'])
            per_graph.append(
                graph_metrics(test_s, s['test_labels'].numpy(), hits_ks=hits_ks)
            )
        except Exception:
            per_graph.append(None)
    approaches['Coordinates-only'] = aggregate(per_graph, hits_ks=hits_ks)

    # ── Tier 2: GNN encoder ───────────────────────────────────────────────────
    if encoder is not None and edge_pred is not None:
        encoder.eval(); edge_pred.eval()
        per_graph = []
        with torch.no_grad():
            for s in splits:
                try:
                    td   = s['train_data']
                    x    = td.x.float().to(device)
                    ei   = td.edge_index.to(device)
                    ea   = td.edge_attr.float().to(device) if td.edge_attr is not None else None
                    z    = encoder(x, ei, ea)
                    eli  = s['test_ei'].to(device)
                    sc   = torch.sigmoid(edge_pred(z, eli)).cpu().numpy()
                    lbl  = s['test_labels'].numpy()
                    per_graph.append(graph_metrics(sc, lbl, hits_ks=hits_ks))
                except Exception:
                    per_graph.append(None)
        approaches['GNN (GAE)'] = aggregate(per_graph, hits_ks=hits_ks)

    return approaches


def print_headline_table(approaches: dict, hits_ks: tuple = (20, 50)):
    """Print the headline comparison table to stdout."""
    from .metrics import format_results

    print('\n' + '═' * 80)
    print('  HEADLINE RESULTS — link prediction, per-graph-averaged')
    print('  Metrics: AUC-ROC (comparability) | AP = primary | Hits@K | MRR')
    print('═' * 80)

    col_w = 26
    print(f"  {'Approach':<{col_w}}  Results")
    print('  ' + '─' * 76)
    for name, agg in approaches.items():
        print(f"  {name:<{col_w}}  {format_results(agg, hits_ks=hits_ks)}")
    print('═' * 80)
    print('  AP = Average Precision (primary under imbalance)')
    print('  AUC reported for comparability — do not weight conclusions on it\n')
