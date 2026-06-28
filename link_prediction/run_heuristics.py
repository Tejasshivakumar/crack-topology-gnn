#!/usr/bin/env python3
"""
Run heuristic baselines (CN, AA, RA, Coordinates-only) on the same test splits
used for GNN evaluation and compare against the best GNN checkpoint.

Usage (from crack-topology-gnn/):
    python3 link_prediction/run_heuristics.py \
        --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
        --gnn-dir      outputs/linkpred_clean_50ep/gine \
        --model        gine \
        --output-dir   outputs/linkpred_clean_50ep/heuristics

All approaches are evaluated on IDENTICAL pre-computed splits so the
comparison is strictly fair (same positives, same hard negatives).
"""

import os
import sys
import json
import argparse
import warnings
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_prediction.train    import build_models
from link_prediction.splits   import transductive_split
from link_prediction.evaluate import headline_table, print_headline_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-graphs', default='outputs/clean_graphs/graphs/test_graphs.pt')
    p.add_argument('--gnn-dir',     default='outputs/linkpred_clean_50ep/gine',
                   help='Directory containing best_model.pt for the GNN comparison row')
    p.add_argument('--model',       default='gine',
                   choices=['mlp', 'gcn', 'sage', 'gine', 'gat'],
                   help='Architecture name matching the checkpoint in --gnn-dir')
    p.add_argument('--hidden',  type=int, default=128)
    p.add_argument('--out-dim', type=int, default=64)
    p.add_argument('--heads',   type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--hits-ks', type=int, nargs='+', default=[20, 50])
    p.add_argument('--output-dir', default='outputs/linkpred_clean_50ep/heuristics')
    p.add_argument('--no-gnn', action='store_true',
                   help='Skip GNN row (heuristics + coords only)')
    return p.parse_args()


def build_splits(dataset):
    """Build transductive splits for every test graph, dropping invalids."""
    splits, graphs = [], []
    for g in dataset:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = transductive_split(g, num_val=0.0, num_test=0.20, seed=0)
        if s is None:
            continue
        splits.append(s)
        graphs.append(g)
    return splits, graphs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading test graphs: {args.test_graphs}')
    test_dataset = torch.load(args.test_graphs, weights_only=False)
    print(f'  {len(test_dataset)} graphs loaded')

    print('\nBuilding transductive splits (same seed=0 as GNN evaluation)...')
    splits, graphs = build_splits(test_dataset)
    print(f'  {len(splits)} graphs produced valid splits')

    encoder = edge_pred = None
    if not args.no_gnn:
        ckpt_path = os.path.join(args.gnn_dir, 'best_model.pt')
        print(f'\nLoading GNN checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        in_ch = test_dataset[0].x.size(1)
        encoder, edge_pred, _ = build_models(
            in_ch, args.hidden, args.out_dim, args.heads, args.dropout,
            model_name=args.model,
        )
        encoder.load_state_dict(ckpt['encoder'])
        edge_pred.load_state_dict(ckpt['edge_pred'])
        encoder.eval(); edge_pred.eval()
        print(f'  Best epoch: {ckpt.get("best_epoch","?")}  '
              f'val_score: {ckpt.get("best_val_score","?"):.4f}')

    hits_ks = tuple(args.hits_ks)
    print('\nRunning all approaches on identical splits...')
    approaches = headline_table(
        splits, graphs,
        encoder=encoder, edge_pred=edge_pred,
        device=torch.device('cpu'),
        hits_ks=hits_ks,
    )

    print_headline_table(approaches, hits_ks=hits_ks)

    # Save to JSON (replace None with null-safe strings)
    def _clean(d):
        return {k: (v if v is not None else 'null') for k, v in d.items()}

    out = {name: _clean(agg) for name, agg in approaches.items()}
    out_path = os.path.join(args.output_dir, 'heuristic_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved → {out_path}')


if __name__ == '__main__':
    main()
