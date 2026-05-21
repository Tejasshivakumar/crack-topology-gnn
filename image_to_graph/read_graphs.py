#!/usr/bin/env python3
"""
Inspect saved .pt graph datasets.

Usage:
    python read_graphs.py --pt ../outputs/graphs/train_graphs.pt
    python read_graphs.py --pt ../outputs/graphs/test_graphs.pt --n 3
"""

import argparse
import json
import os
import torch
import numpy as np


NODE_FEATURES = [
    'x_norm', 'y_norm', 'thickness', 'degree', 'is_endpoint', 'is_junction',
]
EDGE_FEATURES = [
    'path_length', 'euclidean_dist', 'tortuosity',
    'angle_sym', 'avg_thickness', 'min_thickness', 'max_thickness',
]


def print_graph(data, idx: int) -> None:
    print(f'\n── Graph [{idx}] ─────────────────────────────────')
    print(f'  file          : {getattr(data, "filename", "n/a")}')
    print(f'  split         : {getattr(data, "split", "n/a")}')
    print(f'  image size    : {getattr(data, "img_h", "?")} × {getattr(data, "img_w", "?")}')
    crack_px = getattr(data, 'crack_pixels', 'n/a')
    density  = getattr(data, 'crack_density', 'n/a')
    print(f'  crack pixels  : {crack_px}  (density={density:.4f})'
          if isinstance(density, float) else f'  crack pixels  : {crack_px}')
    print(f'  nodes         : {data.x.size(0)}')
    print(f'  edges (undir) : {data.edge_index.size(1) // 2}')

    print(f'\n  Node features  x.shape={list(data.x.shape)}')
    print(f'  {"col":>4}  {"name":<16}  {"mean":>8}  {"min":>8}  {"max":>8}')
    for i, name in enumerate(NODE_FEATURES):
        col = data.x[:, i]
        print(f'  {i:>4}  {name:<16}  {col.mean().item():>8.3f}  '
              f'{col.min().item():>8.3f}  {col.max().item():>8.3f}')

    print(f'\n  Edge features  edge_attr.shape={list(data.edge_attr.shape)}')
    # Use only unique edges (every other row since both directions stored)
    ea = data.edge_attr[::2]
    print(f'  {"col":>4}  {"name":<16}  {"mean":>8}  {"min":>8}  {"max":>8}')
    for i, name in enumerate(EDGE_FEATURES):
        col = ea[:, i]
        print(f'  {i:>4}  {name:<16}  {col.mean().item():>8.3f}  '
              f'{col.min().item():>8.3f}  {col.max().item():>8.3f}')


def parse_args():
    p = argparse.ArgumentParser(description='Inspect saved crack graph .pt files.')
    p.add_argument('--pt', type=str, required=True, help='Path to .pt graph file.')
    p.add_argument('--n',  type=int, default=2,     help='Number of graphs to display.')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Loading: {args.pt}')
    dataset = torch.load(args.pt, weights_only=False)
    print(f'Total graphs: {len(dataset)}')

    # Overall dataset stats
    node_counts = [d.x.size(0) for d in dataset]
    edge_counts = [d.edge_index.size(1) // 2 for d in dataset]
    print(f'\n── Dataset overview ─────────────────────────')
    print(f'  nodes  mean={np.mean(node_counts):.1f}  '
          f'median={np.median(node_counts):.0f}  '
          f'min={min(node_counts)}  max={max(node_counts)}')
    print(f'  edges  mean={np.mean(edge_counts):.1f}  '
          f'median={np.median(edge_counts):.0f}  '
          f'min={min(edge_counts)}  max={max(edge_counts)}')

    n = min(args.n, len(dataset))
    for i in range(n):
        print_graph(dataset[i], i)


if __name__ == '__main__':
    main()
