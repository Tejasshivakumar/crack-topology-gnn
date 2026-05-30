#!/usr/bin/env python3
"""
Phase 3: GNN Link + Node Prediction — main entry point.

Proves the model understands crack topology through two tasks:
  1. Edge prediction  — reconstruct hidden crack segments (mud/dirt analogy)
  2. Node prediction  — identify nodes with hidden neighbours (incomplete crack tips)

Usage:
    cd crack-topology-gnn
    python3 link_prediction/run.py \\
        --train-graphs outputs/graphs/graphs/train_graphs.pt \\
        --test-graphs  outputs/graphs/graphs/test_graphs.pt  \\
        --output-dir   outputs/link_prediction \\
        --epochs 150

Full options: python3 link_prediction/run.py --help
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_prediction.train     import train, build_models
from link_prediction.evaluate  import full_evaluation
from link_prediction.visualize import visualize_predictions, plot_training_curves


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3: crack topology GNN training + evaluation')

    p.add_argument('--train-graphs', type=str,
                   default='outputs/graphs/graphs/train_graphs.pt')
    p.add_argument('--test-graphs',  type=str,
                   default='outputs/graphs/graphs/test_graphs.pt')
    p.add_argument('--output-dir',   type=str,
                   default='outputs/link_prediction')

    # Model
    p.add_argument('--hidden',   type=int,   default=128)
    p.add_argument('--out-dim',  type=int,   default=64)
    p.add_argument('--heads',    type=int,   default=4)
    p.add_argument('--dropout',  type=float, default=0.3)

    # Training
    p.add_argument('--epochs',           type=int,   default=300)
    p.add_argument('--lr',               type=float, default=5e-4)
    p.add_argument('--weight-decay',     type=float, default=1e-4)
    p.add_argument('--edge-loss-weight', type=float, default=1.0,
                   help='Weight λ for edge prediction loss (node + λ*edge)')
    p.add_argument('--node-mask-frac',   type=float, default=0.20,
                   help='Fraction of endpoint nodes to hide per graph during node task training')
    p.add_argument('--accum-steps',      type=int,   default=8,
                   help='Gradient accumulation steps (number of graphs per optimizer step)')
    p.add_argument('--warmup-epochs',    type=int,   default=10,
                   help='Linear LR warmup epochs before cosine annealing')

    # Misc
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--vis-n',     type=int, default=10,
                   help='Number of test graphs to visualise (-1 = all)')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training — load checkpoint from --output-dir and run evaluation only')

    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    torch.manual_seed(args.seed)

    print(f'Device: {device}')
    print(f'Loading training graphs: {args.train_graphs}')
    train_dataset = torch.load(args.train_graphs, weights_only=False)
    print(f'Loading test graphs:     {args.test_graphs}')
    test_dataset  = torch.load(args.test_graphs,  weights_only=False)
    print(f'  Train: {len(train_dataset)} graphs  |  Test: {len(test_dataset)} graphs\n')

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir   = os.path.join(args.output_dir, 'visualizations')
    ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
    os.makedirs(vis_dir, exist_ok=True)

    if args.eval_only:
        # ── Load existing checkpoint ──────────────────────────────────────────
        print(f'Loading checkpoint: {ckpt_path}')
        ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
        in_ch     = train_dataset[0].x.size(1)
        encoder, edge_pred, node_pred = build_models(
            in_ch, args.hidden, args.out_dim, args.heads, args.dropout
        )
        encoder.load_state_dict(ckpt['encoder'])
        edge_pred.load_state_dict(ckpt['edge_pred'])
        node_pred.load_state_dict(ckpt['node_pred'])
        encoder.to(device); edge_pred.to(device); node_pred.to(device)
        print(f'  Best epoch: {ckpt.get("best_epoch", "?")}  |  '
              f'Best val score: {ckpt.get("best_val_score", "?")}\n')
    else:
        # ── Training ──────────────────────────────────────────────────────────
        print('=' * 55)
        print('  TRAINING')
        print('=' * 55)
        result = train(
            train_dataset    = train_dataset,
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
            device           = device,
            log_every        = args.log_every,
        )

        encoder   = result['encoder']
        edge_pred = result['edge_pred']
        node_pred = result['node_pred']

        torch.save({
            'encoder':        encoder.state_dict(),
            'edge_pred':      edge_pred.state_dict(),
            'node_pred':      node_pred.state_dict(),
            'best_val_score': result['best_val_score'],
            'best_epoch':     result['best_epoch'],
            'args':           vars(args),
        }, ckpt_path)
        print(f'\nCheckpoint saved → {ckpt_path}')

        plot_training_curves(
            result['history'],
            save_path=os.path.join(args.output_dir, 'training_curves.png'),
        )

    # ── Evaluation on test set ────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print('  EVALUATION ON UNSEEN TEST SET')
    print('=' * 55)

    metrics = full_evaluation(encoder, edge_pred, node_pred, test_dataset, device)

    print('\n  ── Task 1: Node Prediction (missing crack tips — primary) ──')
    print(f'  AUC-ROC  : {metrics["node_auc"]:.4f}')
    print(f'  Avg Prec : {metrics["node_ap"]:.4f}')
    print(f'  F1 Score : {metrics["node_f1"]:.4f}')

    print('\n  ── Task 2: Edge Prediction (missing crack segments — secondary) ──')
    print(f'  AUC-ROC  : {metrics["edge_auc"]:.4f}')
    print(f'  Avg Prec : {metrics["edge_ap"]:.4f}')
    for k in [10, 20]:
        key = f'edge_hits@{k}'
        if key in metrics:
            print(f'  Hits@{k:<3} : {metrics[key]:.4f}')

    if not args.eval_only:
        metrics['best_val_score'] = result['best_val_score']
        metrics['best_epoch']     = result['best_epoch']

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics saved → {metrics_path}')

    # ── Visualisations ────────────────────────────────────────────────────────
    print(f'\nGenerating {args.vis_n if args.vis_n > 0 else "all"} test visualisations...')
    vis_graphs = test_dataset if args.vis_n < 0 else test_dataset[:args.vis_n]

    for g in vis_graphs:
        fname = getattr(g, 'filename', 'graph')
        stem  = os.path.splitext(fname)[0]
        path  = os.path.join(vis_dir, stem + '.png')
        try:
            visualize_predictions(g, encoder, edge_pred, node_pred, device,
                                  save_path=path)
        except Exception as e:
            print(f'  Skipping {fname}: {e}')

    print(f'Visualisations saved → {vis_dir}')
    print('\nDone.')


if __name__ == '__main__':
    main()
