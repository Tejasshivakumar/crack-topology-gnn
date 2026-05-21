#!/usr/bin/env python3
"""
Build a reproducible 80/10/10 train/val/test split from SematicSeg_Dataset.

Usage:
    python prepare_split.py

What it does:
  1. Reads all valid image-mask pairs from SematicSeg_Dataset
     (skips desktop.ini, skips orphan labels with no matching image)
  2. Shuffles with seed=42 for reproducibility
  3. Splits 80 / 10 / 10 → train / val / test
  4. Copies files into:
       split/train/images, split/train/masks
       split/val/images,   split/val/masks
       split/test/images,  split/test/masks

Run this ONCE before training.  Re-running overwrites the split.
"""

import os
import shutil
import random

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_ROOT = '/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY'
SRC_IMAGES   = os.path.join(DATASET_ROOT, 'SematicSeg_Dataset', 'Original Image')
SRC_MASKS    = os.path.join(DATASET_ROOT, 'SematicSeg_Dataset', 'Labels')
SPLIT_ROOT   = os.path.join(DATASET_ROOT, 'split')

VALID_IMG_EXT  = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SEED           = 42
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
# TEST_RATIO   = remaining (0.10)


def collect_pairs(img_dir, mask_dir):
    """Return sorted list of (img_path, mask_path) for valid matched pairs."""
    img_stems  = {}
    for f in os.listdir(img_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in VALID_IMG_EXT:
            img_stems[stem] = os.path.join(img_dir, f)

    pairs = []
    for f in os.listdir(mask_dir):
        stem, ext = os.path.splitext(f)
        # skip orphan labels that end in ".jpg" (e.g. "29407.jpg" inside Labels/)
        if ext.lower() not in VALID_IMG_EXT or '.jpg' in stem or '.jpeg' in stem:
            continue
        if stem in img_stems:
            pairs.append((img_stems[stem], os.path.join(mask_dir, f)))

    pairs.sort(key=lambda p: p[0])
    return pairs


def copy_split(pairs, subset_name):
    img_dir  = os.path.join(SPLIT_ROOT, subset_name, 'images')
    mask_dir = os.path.join(SPLIT_ROOT, subset_name, 'masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Clear existing files
    for d in (img_dir, mask_dir):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    for img_path, mask_path in pairs:
        shutil.copy2(img_path,  os.path.join(img_dir,  os.path.basename(img_path)))
        shutil.copy2(mask_path, os.path.join(mask_dir, os.path.basename(mask_path)))

    print(f'  {subset_name:<6}: {len(pairs):>4} pairs  ->  {img_dir}')


def main():
    pairs = collect_pairs(SRC_IMAGES, SRC_MASKS)
    print(f'Total valid image-mask pairs found: {len(pairs)}')

    random.seed(SEED)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train : n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f'\nSplit (seed={SEED}):')
    copy_split(train_pairs, 'train')
    copy_split(val_pairs,   'val')
    copy_split(test_pairs,  'test')

    print(f'\nTotal used: {n_train + n_val + n_test} / {n}')
    print('Done. Split saved to:', SPLIT_ROOT)


if __name__ == '__main__':
    main()
