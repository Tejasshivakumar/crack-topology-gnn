#!/usr/bin/env python3
"""
Frontier masking evaluation — all model checkpoints.

Runs two physically-motivated evaluations on existing trained models:

  Frontier Edge Task
      frontier_edge_split: frontier edges (edges touching crack tips) are hidden;
      the tip node stays VISIBLE but isolated.  Model must predict which isolated
      tip reconnects where — directly simulating crack growth direction prediction.

  Frontier Node Task
      apply_frontier_mask: crack tip nodes AND their complete connecting edges are
      hidden.  Model must identify which visible base nodes lost a growth tip.

Results are compared side-by-side against the random-masking baseline from
outputs/linkpred_clean_50ep/comparison_results.json.

Usage
-----
    cd crack-topology-gnn
    source ../.venv/bin/activate

    python3 link_prediction/eval_frontier.py

    # Custom paths / mask fraction
    python3 link_prediction/eval_frontier.py \\
        --ckpt-dir    outputs/linkpred_clean_50ep \\
        --test-graphs outputs/clean_graphs/graphs/test_graphs.pt \\
        --mask-frac   0.30 \\
        --output      outputs/frontier_eval_results.json
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_prediction.model import build_encoder, MLPEdgePredictor, MLPNodePredictor
from link_prediction.masking import (
    apply_frontier_mask,
    frontier_edge_split,
    is_valid_for_frontier_task,
    is_valid_for_node_task,
)

DEVICE = torch.device('cpu')   # evaluate on CPU — avoids MPS OOM on large test sets


# ── Model loading ─────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, model_name: str):
    """
    Load encoder + edge_pred + node_pred from a saved checkpoint.
    Reads hyperparameters from the stored args dict so no manual config needed.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    saved_args = ckpt.get('args', {})
    hidden  = saved_args.get('hidden',  128)
    out_dim = saved_args.get('out_dim', 64)
    heads   = saved_args.get('heads',   4)
    dropout = saved_args.get('dropout', 0.3)

    # mlp_pos uses the MLP architecture with a stripped feature set
    arch = 'mlp' if model_name == 'mlp_pos' else model_name

    encoder   = build_encoder(arch, in_channels=6, hidden=hidden,
                               out_dim=out_dim, heads=heads, dropout=dropout)
    edge_pred = MLPEdgePredictor(out_dim, hidden=hidden,     dropout=dropout)
    node_pred = MLPNodePredictor(out_dim, hidden=hidden // 2, dropout=dropout)

    encoder.load_state_dict(ckpt['encoder'])
    edge_pred.load_state_dict(ckpt['edge_pred'])
    node_pred.load_state_dict(ckpt['node_pred'])

    encoder.eval(); edge_pred.eval(); node_pred.eval()

    return encoder, edge_pred, node_pred, ckpt.get('best_epoch', '?')


# ── Frontier edge evaluation ──────────────────────────────────────────────────

def evaluate_frontier_edge(encoder, edge_pred, graphs,
                            mask_frac: float = 0.30, seed: int = 0) -> dict:
    """
    Frontier edge task: hide edges touching crack tip nodes; tip stays visible.
    Model predicts which isolated tip reconnects to which base node.

    Aggregation: per-graph AP then averaged (consistent with canonical edge eval).
    """
    per_ap  = []
    per_auc = []

    with torch.no_grad():
        for i, g in enumerate(graphs):
            if not is_valid_for_frontier_task(g, min_frontier_edges=2):
                continue

            split = frontier_edge_split(g, mask_frac=mask_frac, seed=seed + i)
            if split is None:
                continue

            td  = split['train_data']
            x   = td.x.float().to(DEVICE)
            ei  = td.edge_index.to(DEVICE)
            ea  = td.edge_attr.float().to(DEVICE) if td.edge_attr is not None else None
            eli = split['label_ei'].to(DEVICE)
            lbl = split['labels'].numpy()

            if len(np.unique(lbl)) < 2:
                continue

            z      = encoder(x, ei, ea)
            scores = torch.sigmoid(edge_pred(z, eli)).cpu().numpy()

            per_ap.append(float(average_precision_score(lbl, scores)))
            per_auc.append(float(roc_auc_score(lbl, scores)))

    n = len(per_ap)
    if n == 0:
        return {'frontier_edge_ap': 0.0, 'frontier_edge_auc': 0.0,
                'frontier_edge_ap_std': 0.0, 'n_frontier_edge_graphs': 0}

    return {
        'frontier_edge_ap':      float(np.mean(per_ap)),
        'frontier_edge_ap_std':  float(np.std(per_ap)),
        'frontier_edge_auc':     float(np.mean(per_auc)),
        'n_frontier_edge_graphs': n,
    }


# ── Frontier node evaluation ──────────────────────────────────────────────────

def evaluate_frontier_node(encoder, node_pred, graphs,
                            mask_frac: float = 0.30, seed: int = 0) -> dict:
    """
    Frontier node task: hide tip nodes + their complete edges (growth segments).
    Model identifies which visible base nodes lost a growth-tip neighbour.

    Aggregation: global pooling (consistent with canonical node eval).
    """
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for i, g in enumerate(graphs):
            if not is_valid_for_node_task(g, min_nodes=2, node_type='endpoint'):
                continue

            result = apply_frontier_mask(g, mask_frac=mask_frac, seed=seed + i)
            masked, node_labels, _, eval_mask, _ = result

            if eval_mask.sum() == 0:
                continue
            labels_ev = node_labels[eval_mask]
            if len(torch.unique(labels_ev)) < 2:
                continue

            x   = masked.x.float().to(DEVICE)
            ei  = masked.edge_index.to(DEVICE)
            ea  = masked.edge_attr.float().to(DEVICE) if masked.edge_attr is not None else None

            z     = encoder(x, ei, ea)
            probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels_ev.numpy())

    if not all_labels or len(np.unique(all_labels)) < 2:
        return {'frontier_node_ap': 0.0, 'frontier_node_auc': 0.0}

    ap  = float(average_precision_score(all_labels, all_probs))
    auc = float(roc_auc_score(all_labels, all_probs))
    return {'frontier_node_ap': ap, 'frontier_node_auc': auc}


# ── Comparison table ──────────────────────────────────────────────────────────

def print_table(all_results: dict, baseline: dict, model_order: list,
                mask_frac: float):
    W = 108
    print('\n' + '═' * W)
    print('  FRONTIER MASKING EVALUATION — Physically-Motivated Crack Growth Simulation')
    print(f'  mask_frac = {mask_frac}  |  hard negatives k_near=30  |  evaluated on CPU')
    print('─' * W)
    print('  WHAT EACH TASK TESTS:')
    print('  Frontier Edge : tip VISIBLE but isolated — predict which tip connects where (growth direction)')
    print('  Frontier Node : tip + edge HIDDEN        — predict which base node lost a growth segment')
    print('═' * W)
    print(f"  {'Model':<8}  {'Ep':>3}  "
          f"│  {'Rand Edge AP':>12}  {'Front Edge AP':>13}  {'ΔEdge':>6}  "
          f"│  {'Rand Node AP':>12}  {'Front Node AP':>13}  {'ΔNode':>6}")
    print('  ' + '─' * (W - 2))

    for m in model_order:
        if m not in all_results:
            continue
        r  = all_results[m]
        b  = baseline.get(m, {})
        ep = str(r.get('best_epoch', '?'))

        re_ap = b.get('edge_ap', 0.0)
        fe_ap = r.get('frontier_edge_ap', 0.0)
        de    = fe_ap - re_ap

        rn_ap = b.get('node_ap', 0.0)
        fn_ap = r.get('frontier_node_ap', 0.0)
        dn    = fn_ap - rn_ap

        fe_std = r.get('frontier_edge_ap_std', 0.0)

        print(f"  {m:<8}  {ep:>3}  "
              f"│  {re_ap:>12.4f}  {fe_ap:>13.4f}  {de:>+6.3f}  "
              f"│  {rn_ap:>12.4f}  {fn_ap:>13.4f}  {dn:>+6.3f}")

    print('═' * W)
    print('  Δ = Frontier AP − Random AP')
    print('  Expected: ΔEdge < 0 (frontier task is harder — fewer structural shortcuts at crack tips)')
    print('  Key finding: does GNN maintain or WIDEN its advantage over MLP on frontier task?')
    print()

    # Print GNN vs MLP gap analysis
    if 'mlp' in all_results and any(m in all_results for m in ['gine', 'sage']):
        mlp_fe  = all_results['mlp'].get('frontier_edge_ap', 0.0)
        mlp_fn  = all_results['mlp'].get('frontier_node_ap', 0.0)
        mlp_re  = baseline.get('mlp', {}).get('edge_ap', 0.0)
        mlp_rn  = baseline.get('mlp', {}).get('node_ap', 0.0)

        print('  ── GNN vs MLP gap (primary proof of topology understanding) ──')
        print(f"  {'Model':<8}  {'Rand Edge AP':>12}  {'→':>2}  {'Front Edge AP':>13}  "
              f"│  {'Rand Node AP':>12}  {'→':>2}  {'Front Node AP':>13}  {'GNN advantage?'}")
        print('  ' + '─' * 78)

        for m in ['gine', 'sage', 'gcn', 'gat']:
            if m not in all_results:
                continue
            r = all_results[m]
            b = baseline.get(m, {})

            rand_e_gap  = b.get('edge_ap', 0.0) - mlp_re
            front_e_gap = r.get('frontier_edge_ap', 0.0) - mlp_fe
            rand_n_gap  = b.get('node_ap', 0.0) - mlp_rn
            front_n_gap = r.get('frontier_node_ap', 0.0) - mlp_fn

            e_sign = '✓ widens' if front_e_gap > rand_e_gap else ('✓ holds' if front_e_gap > 0 else '✗ loses')
            n_sign = '✓ widens' if front_n_gap > rand_n_gap else ('✓ holds' if front_n_gap > 0 else '✗ loses')

            print(f"  {m:<8}  {rand_e_gap:>+12.4f}  {'':>2}  {front_e_gap:>+13.4f}  "
                  f"│  {rand_n_gap:>+12.4f}  {'':>2}  {front_n_gap:>+13.4f}  {n_sign}")

        print()
        print('  Gap = GNN AP − MLP AP  |  ✓ = GNN still beats MLP on frontier task')
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate all checkpoints on frontier masking tasks.'
    )
    p.add_argument('--ckpt-dir',    default='outputs/linkpred_clean_50ep',
                   help='Directory containing per-model checkpoint subdirs')
    p.add_argument('--test-graphs', default='outputs/clean_graphs/graphs/test_graphs.pt')
    p.add_argument('--baseline',    default='outputs/linkpred_clean_50ep/comparison_results.json',
                   help='Random-masking baseline JSON for comparison')
    p.add_argument('--mask-frac',   type=float, default=0.30,
                   help='Fraction of frontier edges/nodes to hide (default 0.30)')
    p.add_argument('--seed',        type=int,   default=0)
    p.add_argument('--models',      nargs='+',
                   default=['mlp', 'gcn', 'sage', 'gine', 'gat'])
    p.add_argument('--output',      default=None,
                   help='Optional path to save results JSON')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'\nDevice          : {DEVICE}')
    print(f'Checkpoint dir  : {args.ckpt_dir}')
    print(f'Test graphs     : {args.test_graphs}')
    print(f'Mask fraction   : {args.mask_frac}')
    print(f'Models          : {args.models}')

    print(f'\nLoading test graphs...')
    test_graphs = torch.load(args.test_graphs, map_location='cpu', weights_only=False)
    print(f'  {len(test_graphs)} graphs loaded')

    # Count how many graphs are valid for each task
    valid_edge = sum(1 for g in test_graphs if is_valid_for_frontier_task(g, 2))
    valid_node = sum(1 for g in test_graphs
                     if is_valid_for_node_task(g, 2, 'endpoint'))
    print(f'  Valid for frontier edge task : {valid_edge}')
    print(f'  Valid for frontier node task : {valid_node}')

    # Load random-masking baseline
    baseline = {}
    if os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)
        print(f'\nBaseline loaded : {args.baseline}')
    else:
        print(f'\nNo baseline JSON found — Δ columns will show N/A')

    all_results = {}

    for model_name in args.models:
        ckpt_path = os.path.join(args.ckpt_dir, model_name, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f'\n  [{model_name.upper()}] checkpoint not found at {ckpt_path} — skipping')
            continue

        print(f'\n{"─" * 60}')
        print(f'  Model: {model_name.upper()}')
        print(f'{"─" * 60}')

        try:
            encoder, edge_pred, node_pred, best_ep = load_checkpoint(ckpt_path, model_name)
        except Exception as e:
            print(f'  ERROR loading checkpoint: {e}')
            continue

        print(f'  Checkpoint loaded  (best epoch: {best_ep})')

        # ── Frontier edge task ─────────────────────────────────────────────
        print(f'  Running frontier edge task...')
        e_metrics = evaluate_frontier_edge(
            encoder, edge_pred, test_graphs,
            mask_frac=args.mask_frac, seed=args.seed,
        )
        n_eg = e_metrics['n_frontier_edge_graphs']
        print(f'  Frontier Edge AP  = {e_metrics["frontier_edge_ap"]:.4f}'
              f'  (±{e_metrics["frontier_edge_ap_std"]:.4f},  {n_eg} graphs)')
        print(f'  Frontier Edge AUC = {e_metrics["frontier_edge_auc"]:.4f}')
        if baseline.get(model_name):
            delta = e_metrics['frontier_edge_ap'] - baseline[model_name].get('edge_ap', 0)
            print(f'  vs Random Edge AP = {baseline[model_name].get("edge_ap", 0):.4f}'
                  f'  →  Δ = {delta:+.4f}')

        # ── Frontier node task ─────────────────────────────────────────────
        print(f'  Running frontier node task...')
        n_metrics = evaluate_frontier_node(
            encoder, node_pred, test_graphs,
            mask_frac=args.mask_frac, seed=args.seed,
        )
        print(f'  Frontier Node AP  = {n_metrics["frontier_node_ap"]:.4f}')
        print(f'  Frontier Node AUC = {n_metrics["frontier_node_auc"]:.4f}')
        if baseline.get(model_name):
            delta = n_metrics['frontier_node_ap'] - baseline[model_name].get('node_ap', 0)
            print(f'  vs Random Node AP = {baseline[model_name].get("node_ap", 0):.4f}'
                  f'  →  Δ = {delta:+.4f}')

        all_results[model_name] = {
            'best_epoch': best_ep,
            **e_metrics,
            **n_metrics,
        }

    # ── Comparison table ───────────────────────────────────────────────────
    print_table(all_results, baseline, args.models, args.mask_frac)

    # ── Save results ───────────────────────────────────────────────────────
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'Results saved → {args.output}')

    return all_results


if __name__ == '__main__':
    main()
