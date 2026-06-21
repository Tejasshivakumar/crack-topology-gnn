#!/usr/bin/env python3
"""
Multi-model comparison for crack topology link prediction.

Trains all requested encoder architectures sequentially on the same data
and produces a side-by-side comparison table with all evaluation metrics.

Architectures compared
──────────────────────
  mlp   MLP (no-graph baseline) — ignores graph structure entirely
  gcn   GCN — basic message passing, no edge features
  sage  GraphSAGE — inductive mean aggregation, no edge features
  gine  GINE — GIN + edge features (most expressive)
  gat   CrackGATEncoder — attention + edge features  ← original

Metrics reported
────────────────
  Node task : AUC-ROC, Avg Precision, F1@0.5, F1@opt, Balanced Accuracy
  Edge task : AUC-ROC, Avg Precision, MRR, Hits@10, Hits@20

Usage
─────
    cd crack-topology-gnn
    source ../.venv/bin/activate

    TRAIN="outputs/crack_seg_graphs_train/graphs/train_graphs.pt"
    TEST="outputs/crack_seg_graphs_test/graphs/test_graphs.pt"

    # Run all five models
    python3 link_prediction/compare.py \\
        --train-graphs "$TRAIN" \\
        --test-graphs  "$TEST"  \\
        --output-dir   outputs/comparison \\
        --epochs 200

    # Run a subset
    python3 link_prediction/compare.py \\
        --train-graphs "$TRAIN" \\
        --test-graphs  "$TEST"  \\
        --models gat gcn sage \\
        --epochs 200

    # Quick sanity check (subsample training data)
    python3 link_prediction/compare.py \\
        --train-graphs "$TRAIN" \\
        --test-graphs  "$TEST"  \\
        --num-train-graphs 500 \\
        --epochs 50
"""

import os
import sys
import json
import argparse
import random

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_prediction.train     import train
from link_prediction.evaluate  import full_evaluation
from link_prediction.model     import MODEL_NAMES, count_parameters
from link_prediction.visualize import plot_training_curves

# Extended model list — mlp_pos uses MLP architecture but with position-only
# features (x_norm, y_norm only; degree/endpoint/junction zeroed out).
# This establishes the floor: what spatial coordinates alone can achieve,
# before any structural features or message passing are added.
EXTENDED_MODEL_NAMES = ['mlp_pos'] + MODEL_NAMES


def _pos_only_graphs(graphs):
    """Return graph copies with x[:,2:] zeroed — position+thickness only."""
    result = []
    for g in graphs:
        g2 = g.clone()
        g2.x = g.x.clone().float()
        g2.x[:, 2:] = 0.0
        result.append(g2)
    return result


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Comparison table printer ──────────────────────────────────────────────────

def _fmt(v, is_best=False):
    s = f'{v:.4f}'
    return f'*{s}*' if is_best else f' {s} '


def print_comparison_table(results: dict, model_order: list):
    """Print a formatted side-by-side comparison table to stdout."""
    node_keys = ['node_auc', 'node_ap', 'node_f1', 'node_f1_opt', 'node_bal_acc']
    edge_keys = ['edge_auc', 'edge_ap']

    # Find best per metric
    best = {}
    for key in node_keys + edge_keys:
        vals = {m: results[m]['metrics'].get(key, 0.0) for m in model_order if m in results}
        best[key] = max(vals, key=vals.get) if vals else None

    W = 84
    print('\n' + '═' * W)
    print('  MODEL COMPARISON — crack topology link prediction')
    print('═' * W)
    print(f"  {'Model':<7}  {'Params':>7}  │"
          f"  {'Node AUC':>8}  {'Node AP':>8}  {'F1':>6}  {'F1-opt':>7}  {'BalAcc':>7}  │"
          f"  {'Edge AUC':>8}  {'Edge AP':>8}")
    print('  ' + '─' * (W - 2))

    labels = {
        'mlp_pos': 'mlp-pos',
        'mlp':     'mlp',
        'gcn':     'gcn',
        'sage':    'sage',
        'gine':    'gine',
        'gat':     'gat',
    }
    for m in model_order:
        if m not in results:
            continue
        r = results[m]
        mets = r['metrics']
        params_k = r['params'] // 1000
        label = labels.get(m, m)

        def v(key):
            val = mets.get(key, 0.0)
            marker = '◄' if best.get(key) == m else ' '
            return f'{val:.4f}{marker}'

        print(f"  {label:<7}  {params_k:>6}k  │"
              f"  {v('node_auc'):>9}  {v('node_ap'):>9}  {v('node_f1'):>7}  "
              f"{v('node_f1_opt'):>8}  {v('node_bal_acc'):>8}  │"
              f"  {v('edge_auc'):>9}  {v('edge_ap'):>9}")

    print('═' * W)
    print('  ◄ = best in column\n')


# ── Training curve plot ───────────────────────────────────────────────────────

def plot_comparison_curves(all_history: dict, save_path: str):
    """
    Two-panel figure: node val-AUC and edge val-AUC over epochs,
    one line per model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (name, history) in enumerate(all_history.items()):
        epochs    = [h['epoch']        for h in history]
        node_auc  = [h['node_val_auc'] for h in history]
        edge_auc  = [h['edge_val_auc'] for h in history]
        c = colours[idx % len(colours)]
        axes[0].plot(epochs, node_auc, label=name, color=c)
        axes[1].plot(epochs, edge_auc, label=name, color=c)

    for ax, title in zip(axes, ['Node Val AUC (missing crack tips)',
                                  'Edge Val AUC (missing segments)']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC-ROC')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison curves saved → {save_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare multiple GNN architectures for crack topology link prediction.'
    )

    p.add_argument('--train-graphs', type=str,
                   default='outputs/clean_graphs/graphs/train_graphs.pt')
    p.add_argument('--test-graphs',  type=str,
                   default='outputs/clean_graphs/graphs/test_graphs.pt')
    p.add_argument('--output-dir',   type=str, default='outputs/comparison')

    p.add_argument('--models', nargs='+', default=EXTENDED_MODEL_NAMES,
                   choices=EXTENDED_MODEL_NAMES,
                   help='Which encoder architectures to compare (default: all)')

    # Shared hyper-parameters (same for every model)
    p.add_argument('--hidden',           type=int,   default=128)
    p.add_argument('--out-dim',          type=int,   default=64)
    p.add_argument('--heads',            type=int,   default=4,
                   help='Attention heads (GAT only)')
    p.add_argument('--dropout',          type=float, default=0.3)
    p.add_argument('--epochs',           type=int,   default=200)
    p.add_argument('--lr',               type=float, default=5e-4)
    p.add_argument('--weight-decay',     type=float, default=1e-4)
    p.add_argument('--edge-loss-weight', type=float, default=0.5)
    p.add_argument('--node-mask-frac',   type=float, default=0.20)
    p.add_argument('--accum-steps',      type=int,   default=8)
    p.add_argument('--warmup-epochs',    type=int,   default=10)
    p.add_argument('--log-every',        type=int,   default=20)

    p.add_argument('--patience',          type=int,   default=20,
                   help='Early-stop patience in epochs (0 = disabled)')
    p.add_argument('--num-train-graphs', type=int, default=None,
                   help='Subsample N training graphs (useful for quick experiments)')
    p.add_argument('--num-test-graphs',  type=int, default=None,
                   help='Subsample N test graphs for evaluation (default: use all)')
    p.add_argument('--seed',             type=int, default=42)
    p.add_argument('--device',           type=str, default=None,
                   help='Force device: cpu, cuda, mps (default: auto-detect)')

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device else get_device()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f'Device: {device}')
    print(f'Models to compare: {args.models}')

    print(f'\nLoading training graphs: {args.train_graphs}')
    train_dataset = torch.load(args.train_graphs, weights_only=False)
    print(f'Loading test graphs:     {args.test_graphs}')
    test_dataset  = torch.load(args.test_graphs,  weights_only=False)

    if args.num_train_graphs is not None and args.num_train_graphs < len(train_dataset):
        rng = random.Random(args.seed)
        train_dataset = rng.sample(train_dataset, args.num_train_graphs)
        print(f'Subsampled training set → {len(train_dataset)} graphs')

    if args.num_test_graphs is not None and args.num_test_graphs < len(test_dataset):
        test_dataset = test_dataset[:args.num_test_graphs]
        print(f'Subsampled test set → {len(test_dataset)} graphs')

    print(f'  Train: {len(train_dataset)} graphs  |  Test: {len(test_dataset)} graphs\n')

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    all_history = {}

    for model_name in args.models:
        print('\n' + '═' * 60)
        print(f'  TRAINING: {model_name.upper()}')
        print('═' * 60)

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # mlp_pos: position-only baseline — strips structural features so the
        # model only sees x_norm, y_norm (+ zeroed cols). Uses MLP architecture.
        if model_name == 'mlp_pos':
            train_data_use = _pos_only_graphs(train_dataset)
            test_data_use  = _pos_only_graphs(test_dataset)
            arch_name      = 'mlp'
        else:
            train_data_use = train_dataset
            test_data_use  = test_dataset
            arch_name      = model_name

        result = train(
            train_dataset    = train_data_use,
            epochs           = args.epochs,
            hidden           = args.hidden,
            out_dim          = args.out_dim,
            heads            = args.heads,
            lr               = args.lr,
            weight_decay     = args.weight_decay,
            edge_loss_weight = args.edge_loss_weight,
            node_mask_frac   = args.node_mask_frac,
            dropout          = args.dropout,
            accum_steps      = args.accum_steps,
            warmup_epochs    = args.warmup_epochs,
            patience         = args.patience,
            device           = device,
            log_every        = args.log_every,
            model_name       = arch_name,
        )

        encoder   = result['encoder']
        edge_pred = result['edge_pred']
        node_pred = result['node_pred']

        # Save checkpoint
        ckpt_path = os.path.join(model_dir, 'best_model.pt')
        torch.save({
            'encoder':        encoder.state_dict(),
            'edge_pred':      edge_pred.state_dict(),
            'node_pred':      node_pred.state_dict(),
            'best_val_score': result['best_val_score'],
            'best_epoch':     result['best_epoch'],
            'model_name':     model_name,
            'args':           vars(args),
        }, ckpt_path)

        # Training curves
        plot_training_curves(
            result['history'],
            save_path=os.path.join(model_dir, 'training_curves.png'),
        )

        # Evaluate on CPU — avoids MPS OOM on large test sets with heavy models (e.g. GAT)
        print(f'\n  Evaluating {model_name} on test set (CPU)...')
        encoder.cpu(); edge_pred.cpu(); node_pred.cpu()
        metrics = full_evaluation(encoder, edge_pred, node_pred, test_data_use,
                                  torch.device('cpu'))
        if device.type == 'mps':
            torch.mps.empty_cache()

        n_params = count_parameters(encoder, edge_pred, node_pred)

        print(f'  Node AUC={metrics["node_auc"]:.4f}  AP={metrics["node_ap"]:.4f}'
              f'  F1={metrics["node_f1"]:.4f}  F1-opt={metrics["node_f1_opt"]:.4f}')
        print(f'  Edge AUC={metrics["edge_auc"]:.4f}  AP={metrics["edge_ap"]:.4f}')

        metrics['best_val_score'] = result['best_val_score']
        metrics['best_epoch']     = result['best_epoch']

        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results[model_name] = {'metrics': metrics, 'params': n_params}
        all_history[model_name] = result['history']

    # ── Final comparison ──────────────────────────────────────────────────────
    print_comparison_table(all_results, args.models)

    # Save aggregated results
    comparison_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(comparison_path, 'w') as f:
        # metrics dicts are already JSON-serialisable
        json.dump({m: r['metrics'] | {'params': r['params']}
                   for m, r in all_results.items()}, f, indent=2)
    print(f'Comparison results saved → {comparison_path}')

    # Training curves (all models on one figure)
    plot_comparison_curves(
        all_history,
        save_path=os.path.join(args.output_dir, 'comparison_curves.png'),
    )


if __name__ == '__main__':
    main()
