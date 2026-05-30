"""
Evaluation metrics for both tasks.

Edge task metrics:
    AUC-ROC  — overall discriminative ability
    AP       — Average Precision (area under PR curve); better for imbalanced test sets
    Hits@K   — fraction of true edges ranked in top-K predictions per graph

Node task metrics:
    AUC-ROC  — discriminative ability for missing-neighbour classification
    F1       — harmonic mean of precision and recall
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .masking import make_edge_splitter, apply_node_mask, is_valid_for_edge_task, is_valid_for_node_task


# ── Edge task ─────────────────────────────────────────────────────────────────

def evaluate_edge_task(encoder, edge_pred, dataset, device, hits_k=(10, 20)):
    """
    Evaluate link prediction on a held-out dataset.
    For each graph, 20% of edges are hidden; the model is given the 80% visible
    edges and must predict which pairs are actually connected.

    Returns dict of metrics.
    """
    splitter = make_edge_splitter(num_val=0.0, num_test=0.20)
    encoder.eval(); edge_pred.eval()

    all_probs, all_labels = [], []
    hits_at = {k: [] for k in hits_k}

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

            # Encode using visible (train) edges
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

            # Hits@K: among positive edges, how many land in top-K scores?
            pos_mask = lbls == 1
            n_pos    = pos_mask.sum()
            for k in hits_k:
                if n_pos == 0:
                    continue
                top_k_idx = np.argsort(probs)[::-1][:k]
                hit = float(pos_mask[top_k_idx].sum()) / float(n_pos)
                hits_at[k].append(hit)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    results = {
        'edge_auc': roc_auc_score(all_labels, all_probs),
        'edge_ap':  average_precision_score(all_labels, all_probs),
    }
    for k in hits_k:
        results[f'edge_hits@{k}'] = float(np.mean(hits_at[k])) if hits_at[k] else 0.0

    return results


# ── Node task ─────────────────────────────────────────────────────────────────

def evaluate_node_task(encoder, node_pred, dataset, device,
                       mask_frac=0.20, threshold=0.5, seed=0):
    """
    Evaluate missing-node prediction on a held-out dataset.
    For each graph, mask_frac of endpoint nodes are hidden; the model must
    identify which visible nodes have a hidden neighbour.

    Returns dict of metrics.
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
        return {'node_auc': 0.5, 'node_ap': 0.0, 'node_f1': 0.0}

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_bin  = (all_probs >= threshold).astype(int)

    return {
        'node_auc': roc_auc_score(all_labels, all_probs),
        'node_ap':  average_precision_score(all_labels, all_probs),
        'node_f1':  f1_score(all_labels, preds_bin, zero_division=0),
    }


# ── Combined report ───────────────────────────────────────────────────────────

def full_evaluation(encoder, edge_pred, node_pred, dataset, device):
    edge_metrics = evaluate_edge_task(encoder, edge_pred, dataset, device)
    node_metrics = evaluate_node_task(encoder, node_pred, dataset, device)
    return {**edge_metrics, **node_metrics}
