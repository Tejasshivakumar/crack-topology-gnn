#!/usr/bin/env python3
"""
Stage 2: Batch mask-to-graph conversion.

Processes binary crack masks → PyG Data objects with full node/edge features
and saves them as .pt files ready for Stage 3 (GNN link prediction).

Usage — DeepCrack convenience flag (processes train + test automatically):
    python build_dataset.py \\
        --deepcrack-root "/Users/tejasskamar/Practicum/Data Set/DeepCrack" \\
        --output-dir ../outputs/graphs \\
        --vis

Usage — arbitrary mask directory:
    python build_dataset.py \\
        --mask-dir /path/to/masks \\
        --image-dir /path/to/images \\   # optional, for richer visualisations
        --split test \\
        --output-dir ../outputs/graphs \\
        --vis --vis-n 50

Output layout:
    output_dir/
      graphs/
        train_graphs.pt       # list of PyG Data objects
        test_graphs.pt
        feature_schema.json   # column names for x and edge_attr
        stats.json            # per-split dataset statistics
      visualizations/
        train/<filename>.png
        test/<filename>.png
"""

import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_to_graph.convert import mask_to_graph
from image_to_graph.visualize import visualize_graph


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

NODE_FEATURE_NAMES = [
    'x_norm', 'y_norm', 'thickness', 'degree', 'is_endpoint', 'is_junction',
]
EDGE_FEATURE_NAMES = [
    'path_length', 'euclidean_dist', 'tortuosity',
    'angle_sym', 'avg_thickness', 'min_thickness', 'max_thickness',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_image_match(filename: str, image_dir: str) -> str | None:
    """Find the original image for a given mask filename (any extension)."""
    if image_dir is None:
        return None
    stem = os.path.splitext(filename)[0]
    for ext in IMG_EXTS:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def compute_stats(dataset: list) -> dict:
    if not dataset:
        return {}
    node_counts = [d.x.size(0) for d in dataset]
    edge_counts = [d.edge_index.size(1) // 2 for d in dataset]
    densities   = [float(d.crack_density) for d in dataset]
    return {
        'num_graphs':       len(dataset),
        'nodes_mean':       float(np.mean(node_counts)),
        'nodes_median':     float(np.median(node_counts)),
        'nodes_max':        int(np.max(node_counts)),
        'nodes_min':        int(np.min(node_counts)),
        'edges_mean':       float(np.mean(edge_counts)),
        'edges_median':     float(np.median(edge_counts)),
        'edges_max':        int(np.max(edge_counts)),
        'edges_min':        int(np.min(edge_counts)),
        'crack_density_mean': float(np.mean(densities)),
        'crack_density_max':  float(np.max(densities)),
    }


# ── Core processing ───────────────────────────────────────────────────────────

def process_split(
    mask_dir: str,
    split: str,
    image_dir: str,
    graphs_dir: str,
    vis_dir: str,
    do_vis: bool,
    vis_n: int,
) -> list:
    """
    Process all masks in mask_dir for one split.
    Returns the list of saved PyG Data objects.
    """
    mask_files = sorted(
        f for f in glob.glob(os.path.join(mask_dir, '*'))
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if not mask_files:
        print(f'[{split}] No mask files found in {mask_dir}')
        return []

    print(f'\n[{split}] Processing {len(mask_files)} masks from: {mask_dir}')

    if do_vis:
        os.makedirs(vis_dir, exist_ok=True)

    dataset = []
    skipped = 0
    vis_count = 0

    for mask_path in tqdm(mask_files, desc=f'{split}'):
        data, skeleton, nx_graph = mask_to_graph(mask_path, split=split)

        if data is None:
            skipped += 1
            continue

        dataset.append(data)

        # Visualisation
        if do_vis and (vis_n < 0 or vis_count < vis_n):
            img_path = find_image_match(data.filename, image_dir)
            original = None
            if img_path:
                bgr = cv2.imread(img_path)
                if bgr is not None:
                    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)

            stem = os.path.splitext(data.filename)[0]
            save_path = os.path.join(vis_dir, stem + '.png')
            visualize_graph(binary_mask, skeleton, nx_graph, data,
                            original_image=original, save_path=save_path)
            vis_count += 1

    print(f'[{split}] Converted: {len(dataset)}  |  Skipped (empty): {skipped}')
    if do_vis:
        print(f'[{split}] Visualisations saved: {vis_count} → {vis_dir}')

    # Save .pt
    pt_path = os.path.join(graphs_dir, f'{split}_graphs.pt')
    torch.save(dataset, pt_path)
    print(f'[{split}] Graphs saved → {pt_path}')

    return dataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Stage 2: Convert crack masks to PyG graphs for link prediction.'
    )

    # DeepCrack convenience
    p.add_argument('--deepcrack-root', type=str, default=None,
                   help='Root of the DeepCrack dataset. Auto-detects train_lab/, '
                        'test_lab/, train_img/, test_img/.')

    # Generic mask source
    p.add_argument('--mask-dir',  type=str, default=None,
                   help='Directory of binary mask PNGs (use with --split).')
    p.add_argument('--image-dir', type=str, default=None,
                   help='Directory of original images (optional, for richer vis).')
    p.add_argument('--split', type=str, default='train',
                   choices=['train', 'test'],
                   help='Split label to attach to each graph.')

    # Output
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'outputs', 'graphs'
                   ),
                   help='Root output directory.')

    # Visualisation
    p.add_argument('--vis',    action='store_true', default=True,
                   help='Save visualisations (default: True).')
    p.add_argument('--no-vis', dest='vis', action='store_false',
                   help='Disable visualisations.')
    p.add_argument('--vis-n', type=int, default=-1,
                   help='Max number of visualisations per split (-1 = all).')

    return p.parse_args()


def main():
    args = parse_args()

    graphs_dir = os.path.join(args.output_dir, 'graphs')
    vis_root   = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(graphs_dir, exist_ok=True)

    # Write feature schema once
    schema_path = os.path.join(graphs_dir, 'feature_schema.json')
    with open(schema_path, 'w') as f:
        json.dump({
            'node_features': NODE_FEATURE_NAMES,
            'edge_features': EDGE_FEATURE_NAMES,
        }, f, indent=2)
    print(f'Feature schema saved → {schema_path}')

    all_stats = {}

    if args.deepcrack_root:
        root = args.deepcrack_root
        splits_cfg = [
            ('train', os.path.join(root, 'train_lab'), os.path.join(root, 'train_img')),
            ('test',  os.path.join(root, 'test_lab'),  os.path.join(root, 'test_img')),
        ]
        for split, mask_dir, image_dir in splits_cfg:
            if not os.path.isdir(mask_dir):
                print(f'Skipping {split} — not found: {mask_dir}')
                continue
            dataset = process_split(
                mask_dir=mask_dir,
                split=split,
                image_dir=image_dir if os.path.isdir(image_dir) else None,
                graphs_dir=graphs_dir,
                vis_dir=os.path.join(vis_root, split),
                do_vis=args.vis,
                vis_n=args.vis_n,
            )
            all_stats[split] = compute_stats(dataset)

    elif args.mask_dir:
        dataset = process_split(
            mask_dir=args.mask_dir,
            split=args.split,
            image_dir=args.image_dir,
            graphs_dir=graphs_dir,
            vis_dir=os.path.join(vis_root, args.split),
            do_vis=args.vis,
            vis_n=args.vis_n,
        )
        all_stats[args.split] = compute_stats(dataset)

    else:
        print('Error: provide --deepcrack-root or --mask-dir.')
        raise SystemExit(1)

    # Save stats
    stats_path = os.path.join(graphs_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nDataset statistics saved → {stats_path}')

    # Print summary
    print('\n══ Summary ══════════════════════════════════')
    for split, stats in all_stats.items():
        if not stats:
            continue
        print(f'\n{split.upper()}:')
        print(f'  graphs          : {stats["num_graphs"]}')
        print(f'  nodes  mean/max : {stats["nodes_mean"]:.1f} / {stats["nodes_max"]}')
        print(f'  edges  mean/max : {stats["edges_mean"]:.1f} / {stats["edges_max"]}')
        print(f'  crack density   : mean={stats["crack_density_mean"]:.4f}  '
              f'max={stats["crack_density_max"]:.4f}')
    print()


if __name__ == '__main__':
    main()
