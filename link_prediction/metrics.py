"""
Per-graph-then-average evaluation metrics for link prediction.

Critical design rule: compute each metric PER GRAPH, then average across
graphs.  Do NOT pool all pairs and compute once.  Graphs range from ~3 to
~1003 nodes — pooling lets giant graphs dominate and hides failure on small
ones.

Metrics reported:
    AUC-ROC  — always (comparability anchor to prior papers; flatters under
                imbalance so don't lean conclusions on it, but its absence
                looks odd to reviewers)
    AP       — primary metric (honest under imbalance; weight conclusions here)
    Hits@K   — fraction of top-K predicted links that are real positives
    MRR      — mean reciprocal rank (optional but informative)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Per-pair metric helpers ───────────────────────────────────────────────────

def _hits_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float | None:
    """Fraction of true positives ranked in the top-K predicted pairs."""
    n_pos = int(labels.sum())
    if n_pos == 0 or k <= 0:
        return None
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].sum()) / float(n_pos)


def _mrr(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean Reciprocal Rank. Rank = # negatives scored >= positive + 1."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    ranks = [float(np.sum(neg_scores >= ps)) + 1.0 for ps in pos_scores]
    return float(np.mean([1.0 / r for r in ranks]))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _ap(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


# ── Per-graph evaluation ──────────────────────────────────────────────────────

def graph_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    hits_ks: tuple = (20, 50),
) -> dict | None:
    """
    Compute all metrics for one graph's scored pairs.

    Returns None if the graph contributes no valid pairs (e.g. only one class).
    """
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None

    m = {
        'auc': _auc(scores, labels),
        'ap':  _ap(scores,  labels),
        'mrr': _mrr(scores, labels),
    }
    for k in hits_ks:
        m[f'hits@{k}'] = _hits_at_k(scores, labels, k)

    return m


# ── Dataset-level aggregation ─────────────────────────────────────────────────

def aggregate(per_graph: list, hits_ks: tuple = (20, 50)) -> dict:
    """
    Aggregate a list of per-graph metric dicts into mean ± std.

    Keys in output: {metric}_mean, {metric}_std, n_graphs.
    None values are excluded from aggregation.

    Args:
        per_graph : list of dicts from graph_metrics()
        hits_ks   : K values used — must match what graph_metrics() produced
    """
    keys = ['auc', 'ap', 'mrr'] + [f'hits@{k}' for k in hits_ks]
    collected = {k: [] for k in keys}

    for m in per_graph:
        if m is None:
            continue
        for k in keys:
            v = m.get(k)
            if v is not None:
                collected[k].append(v)

    result = {'n_graphs': sum(1 for m in per_graph if m is not None)}
    for k in keys:
        vals = collected[k]
        if vals:
            result[f'{k}_mean'] = float(np.mean(vals))
            result[f'{k}_std']  = float(np.std(vals))
        else:
            result[f'{k}_mean'] = None
            result[f'{k}_std']  = None

    return result


# ── Convenience: evaluate a scorer over a list of splits ─────────────────────

def evaluate_scorer(
    splits: list,
    score_fn,
    split_key: str = 'test',
    hits_ks: tuple = (20, 50),
) -> dict:
    """
    Evaluate an arbitrary scoring function over a list of pre-computed splits.

    Args:
        splits    : list of dicts from splits.transductive_split()
        score_fn  : callable(split) → (scores: np.ndarray, labels: np.ndarray)
                    for the chosen split (val or test)
        split_key : 'val' or 'test'
        hits_ks   : Hits@K values to compute

    Returns:
        aggregated metrics dict (keys: {metric}_mean, {metric}_std, n_graphs)
    """
    per_graph = []
    for s in splits:
        try:
            scores, labels = score_fn(s)
        except Exception:
            per_graph.append(None)
            continue
        per_graph.append(graph_metrics(scores, labels, hits_ks=hits_ks))
    return aggregate(per_graph, hits_ks=hits_ks)


# ── Pretty printer ────────────────────────────────────────────────────────────

def format_results(agg: dict, hits_ks: tuple = (20, 50)) -> str:
    """Format an aggregated metrics dict as a human-readable string."""
    def _f(k):
        v = agg.get(f'{k}_mean')
        s = agg.get(f'{k}_std')
        if v is None:
            return '   —  '
        if s is not None:
            return f'{v:.4f}±{s:.3f}'
        return f'{v:.4f}'

    parts = [
        f"AUC={_f('auc')}",
        f"AP={_f('ap')}",
    ]
    for k in hits_ks:
        parts.append(f"H@{k}={_f(f'hits@{k}')}")
    parts.append(f"MRR={_f('mrr')}")
    parts.append(f"n={agg.get('n_graphs', 0)}")
    return '  '.join(parts)
