#!/usr/bin/env python3
"""
M5 — GINE Edge Feature Importance Ablation.

Loads the GINE 200-epoch checkpoint (eval-only, no retraining) and runs
node-task evaluation 6 times, each time zeroing a different group of the
7 raw edge features to measure each group's contribution to node AP.

Raw edge feature layout [E, 7]:
  0: path_length      — skeleton path length along segment
  1: euclidean_dist   — straight-line node-to-node distance
  2: tortuosity       — path_length / euclidean_dist
  3: angle_sym        — crack orientation (degrees) → encoded as sin/cos inside GINE
  4: avg_thickness    — mean crack width along segment
  5: min_thickness    — thinnest point
  6: max_thickness    — widest point

Ablation groups tested:
  full          — all 7 features (baseline)
  no_tortuosity — drop index 2
  no_thickness  — drop indices 4, 5, 6
  no_angle      — drop index 3
  no_geometry   — drop indices 0, 1 (path_length + euclidean_dist)
  no_edge_feats — drop all 7 (GINE degrades to GCN-like behaviour)

Usage:
    cd crack-topology-gnn
    source ../.venv/bin/activate

    python3 link_prediction/edge_ablation.py \\
        --ckpt   outputs/linkpred_200ep/gine/best_model.pt \\
        --graphs outputs/clean_graphs/graphs/test_graphs.pt \\
        --output outputs/edge_ablation_results.json
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_prediction.model   import build_encoder, MLPNodePredictor
from link_prediction.masking import apply_node_mask, is_valid_for_node_task

DEVICE = torch.device('cpu')

# ── Edge feature groups to zero out ──────────────────────────────────────────
# Each entry: (name, display_label, list_of_column_indices_to_zero)
ABLATION_GROUPS = [
    ('full',          'Full edge features (baseline)',          []),
    ('no_tortuosity', 'Drop tortuosity',                        [2]),
    ('no_thickness',  'Drop thickness (avg/min/max)',           [4, 5, 6]),
    ('no_angle',      'Drop angle encoding',                    [3]),
    ('no_geometry',   'Drop geometry (path_len + euclid_dist)', [0, 1]),
    ('no_edge_feats', 'Drop ALL edge features (GCN-like)',      [0, 1, 2, 3, 4, 5, 6]),
]


def zero_columns(ea: torch.Tensor, cols: list) -> torch.Tensor:
    if not cols:
        return ea
    ea = ea.clone()
    for c in cols:
        ea[:, c] = 0.0
    return ea


def evaluate_node_ap(encoder, node_pred, dataset, zero_cols, mask_frac=0.20, seed=0):
    encoder.eval()
    node_pred.eval()

    per_ap, per_auc = [], []

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

            x  = masked.x.float().to(DEVICE)
            ei = masked.edge_index.to(DEVICE)
            ea = masked.edge_attr.float().to(DEVICE) if masked.edge_attr is not None else None

            if ea is not None:
                ea = zero_columns(ea, zero_cols)

            z     = encoder(x, ei, ea)
            probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()
            lbls  = labels_ev.cpu().numpy()

            per_ap.append(float(average_precision_score(lbls, probs)))
            per_auc.append(float(roc_auc_score(lbls, probs)))

    return {
        'node_ap':      float(np.mean(per_ap))  if per_ap  else 0.0,
        'node_auc':     float(np.mean(per_auc)) if per_auc else 0.5,
        'n_graphs':     len(per_ap),
    }


def load_gine(ckpt_path):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved   = ckpt.get('args', {})
    hidden  = saved.get('hidden',  128)
    out_dim = saved.get('out_dim', 64)
    heads   = saved.get('heads',   4)
    dropout = saved.get('dropout', 0.3)

    encoder   = build_encoder('gine', in_channels=6, hidden=hidden,
                               out_dim=out_dim, heads=heads, dropout=dropout)
    node_pred = MLPNodePredictor(out_dim, hidden=hidden // 2, dropout=dropout)

    encoder.load_state_dict(ckpt['encoder'])
    node_pred.load_state_dict(ckpt['node_pred'])
    encoder.eval()
    node_pred.eval()

    print(f'GINE loaded  |  best epoch: {ckpt.get("best_epoch", "?")}')
    return encoder, node_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',   default='outputs/linkpred_200ep/gine/best_model.pt')
    p.add_argument('--graphs', default='outputs/clean_graphs/graphs/test_graphs.pt')
    p.add_argument('--output', default='outputs/edge_ablation_results.json')
    p.add_argument('--mask-frac', type=float, default=0.20)
    p.add_argument('--seed',      type=int,   default=0)
    args = p.parse_args()

    encoder, node_pred = load_gine(args.ckpt)

    print(f'Loading test graphs: {args.graphs}')
    dataset = torch.load(args.graphs, map_location='cpu', weights_only=False)
    print(f'  {len(dataset)} graphs\n')

    results = {}
    baseline_ap = None

    print(f'{"Ablation":<45} {"Node AP":>8}  {"vs Full":>8}  {"n_graphs":>9}')
    print('-' * 75)

    for key, label, zero_cols in ABLATION_GROUPS:
        r = evaluate_node_ap(encoder, node_pred, dataset,
                              zero_cols=zero_cols,
                              mask_frac=args.mask_frac,
                              seed=args.seed)
        r['zeroed_columns'] = zero_cols
        r['label']          = label
        results[key]        = r

        if key == 'full':
            baseline_ap = r['node_ap']
            delta_str = '  —'
        else:
            delta = r['node_ap'] - baseline_ap
            delta_str = f'{delta:+.4f}'

        print(f'{label:<45} {r["node_ap"]:>8.4f}  {delta_str:>8}  {r["n_graphs"]:>9}')

    print()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved → {args.output}')

    # Summary interpretation
    print('\n── Key takeaways ──')
    ranked = sorted(
        [(k, v) for k, v in results.items() if k != 'full'],
        key=lambda x: x[1]['node_ap']
    )
    print(f'  Biggest drop when removed:  {results[ranked[0][0]]["label"]}  '
          f'(AP {results[ranked[0][0]]["node_ap"]:.4f}, '
          f'drop {results[ranked[0][0]]["node_ap"] - baseline_ap:+.4f})')
    print(f'  Smallest drop when removed: {results[ranked[-1][0]]["label"]}  '
          f'(AP {results[ranked[-1][0]]["node_ap"]:.4f}, '
          f'drop {results[ranked[-1][0]]["node_ap"] - baseline_ap:+.4f})')


if __name__ == '__main__':
    main()
