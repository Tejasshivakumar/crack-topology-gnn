#!/usr/bin/env python3
"""
Train and compare both segmentation models on crack_seg_clean.

Usage:
    cd /Users/tejasskamar/Practicum/PHASE\ 2/crack-topology-gnn/segmentation
    python compare_seg.py --max-epochs 50 --no-wandb
"""

import os
import sys
import json
import argparse
import subprocess

CLEAN = '/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean'

MODELS = [
    ('hybrid',   'HybridGraphUNet (pretrained ResNet34d + GNN)'),
    ('plain',    'PlainResNetUNet  (pretrained ResNet34d, no GNN)'),
    ('enhanced', 'EnhancedGraphUNet (scratch CNN + GNN)'),
]

METRICS = [
    ('test/crack_iou',      'Crack IoU      ↑'),
    ('test/crack_dice',     'Crack Dice     ↑'),
    ('test/tol_crack_iou',  'Tol Crack IoU  ↑'),
    ('test/tol_crack_dice', 'Tol Crack Dice ↑'),
    ('test/crack_prec',     'Crack Prec     ↑'),
    ('test/crack_rec',      'Crack Rec      ↑'),
    ('test/cldice',         'clDice         ↑'),
    ('test/iou',            'Mean IoU       ↑'),
    ('test/acc',            'Accuracy       ↑'),
    ('test/loss',           'Loss           ↓'),
]


# All metric keys that must be present for a cached result to be considered valid.
# If any are missing the cache is stale (produced by old code) and is retraining triggered.
REQUIRED_KEYS = {m[0] for m in METRICS}


def parse_args():
    p = argparse.ArgumentParser(description='Compare both segmentation models on crack_seg_clean')
    p.add_argument('--train-img-dir',  default=f'{CLEAN}/train/images')
    p.add_argument('--train-mask-dir', default=f'{CLEAN}/train/masks')
    p.add_argument('--test-img-dir',   default=f'{CLEAN}/test/images')
    p.add_argument('--test-mask-dir',  default=f'{CLEAN}/test/masks')
    p.add_argument('--max-epochs',     type=int,   default=150)
    p.add_argument('--image-size',     type=int,   default=448)
    p.add_argument('--batch-size',     type=int,   default=8)
    p.add_argument('--use-tversky',    action='store_true')
    p.add_argument('--use-cldice',     action='store_true')
    p.add_argument('--cldice-weight',  type=float, default=0.3)
    p.add_argument('--output-dir',     default='../outputs/seg_compare')
    p.add_argument('--no-wandb',       action='store_true', default=True)
    p.add_argument('--force',          action='store_true',
                   help='Ignore any cached metrics.json and retrain all models')
    return p.parse_args()


def _cache_is_valid(metrics_path):
    """Return True only if the cache exists and contains every required metric key."""
    if not os.path.exists(metrics_path):
        return False
    with open(metrics_path) as f:
        cached = json.load(f)
    missing = REQUIRED_KEYS - cached.keys()
    if missing:
        print(f'  Cache stale — missing keys: {sorted(missing)}')
        return False
    return True


def run_model(model_key, args):
    out_dir = os.path.join(args.output_dir, model_key)
    metrics_path = os.path.join(out_dir, 'metrics.json')

    if not args.force and _cache_is_valid(metrics_path):
        print(f'\n[{model_key}] Valid cache found — skipping training, loading existing results.')
        with open(metrics_path) as f:
            return json.load(f)

    if args.force and os.path.exists(metrics_path):
        print(f'\n[{model_key}] --force set — ignoring cached results, retraining.')

    cmd = [
        sys.executable, 'train.py',
        '--model',          model_key,
        '--train-img-dir',  args.train_img_dir,
        '--train-mask-dir', args.train_mask_dir,
        '--test-img-dir',   args.test_img_dir,
        '--test-mask-dir',  args.test_mask_dir,
        '--max-epochs',     str(args.max_epochs),
        '--image-size',     str(args.image_size),
        '--batch-size',     str(args.batch_size),
        '--cldice-weight',  str(args.cldice_weight),
        '--output-dir',     out_dir,
    ]
    if args.no_wandb:
        cmd.append('--no-wandb')
    if args.use_tversky:
        cmd.append('--use-tversky')
    if args.use_cldice:
        cmd.append('--use-cldice')

    print(f'\n{"=" * 65}')
    print(f'  Training: {model_key.upper()} — {dict(MODELS)[model_key]}')
    print(f'{"=" * 65}\n')
    subprocess.run(cmd, check=True)

    with open(metrics_path) as f:
        return json.load(f)


def print_table(all_results):
    col_w = 22
    width = 25 + col_w * len(MODELS)

    # Warn if any model result is missing required keys (should not happen after cache fix)
    for mk, _ in MODELS:
        missing = REQUIRED_KEYS - all_results.get(mk, {}).keys()
        if missing:
            print(f'\n  WARNING: {mk} is missing metrics {sorted(missing)} — results invalid, rerun with --force\n')

    import math
    print('\n' + '=' * width)
    print('  SEGMENTATION COMPARISON — crack_seg_clean')
    print('=' * width)

    header = f"{'Metric':<25}" + ''.join(f"{name:<{col_w}}" for _, name in MODELS)
    print(header)
    print('-' * width)

    for key, label in METRICS:
        row = f'{label:<25}'
        vals = [all_results.get(mk, {}).get(key, float('nan')) for mk, _ in MODELS]
        finite = [v for v in vals if not math.isnan(v)]
        best   = (max(finite) if '↑' in label else min(finite)) if finite else None
        for v in vals:
            if math.isnan(v):
                row += f'{"nan":<{col_w}}'
            else:
                marker = ' ✓' if (best is not None and abs(v - best) < 1e-9) else '  '
                row += f'{v:.4f}{marker:<{col_w - 6}}'
        print(row)

    print('=' * width)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Dataset  : {CLEAN}')
    print(f'Epochs   : {args.max_epochs}')
    print(f'Img size : {args.image_size}×{args.image_size}')
    print(f'Batch    : {args.batch_size}')
    print(f'Output   : {args.output_dir}')

    all_results = {}
    for model_key, model_name in MODELS:
        print(f'\n>>> Starting {model_name}')
        all_results[model_key] = run_model(model_key, args)

    print_table(all_results)

    summary_path = os.path.join(args.output_dir, 'comparison.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved → {summary_path}')


if __name__ == '__main__':
    main()
