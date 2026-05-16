#!/usr/bin/env python3
"""
Evaluate EnhancedGraphUNet on the crack test split.

Run:
    cd /Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/segmentation
    python test.py \\
        --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt \\
        --test-img-dir  /path/to/DeepCrack/test_img \\
        --test-mask-dir /path/to/DeepCrack/test_lab
"""

import os
import sys
import argparse

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import EnhancedGraphUNet
from train import SegModule
from dataset import CrackDataset, get_transforms


def parse_args():
    p = argparse.ArgumentParser(description='Test EnhancedGraphUNet on crack dataset')
    p.add_argument('--test-img-dir',    type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/images')
    p.add_argument('--test-mask-dir',   type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/masks')
    p.add_argument('--checkpoint-path', type=str, required=True)
    p.add_argument('--image-size',      type=int, default=256)
    p.add_argument('--batch-size',      type=int, default=8)
    p.add_argument('--num-workers',     type=int, default=2)
    p.add_argument('--precision',       type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)

    base_model = EnhancedGraphUNet(in_channels=3, out_channels=2, features=(32, 64, 128, 256))

    print(f'Loading checkpoint: {args.checkpoint_path}')
    module = SegModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=base_model,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_epochs=100,
    )
    module.eval()

    _, val_tf = get_transforms(args.image_size)
    test_ds   = CrackDataset(
        img_dir=args.test_img_dir, mask_dir=args.test_mask_dir, transform=val_tf
    )
    test_dl   = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if torch.backends.mps.is_available():
        accelerator, devices = 'mps', 1
    elif torch.cuda.is_available():
        accelerator, devices = 'gpu', 1
    else:
        accelerator, devices = 'cpu', 1

    trainer = pl.Trainer(
        accelerator=accelerator, devices=devices,
        precision=args.precision, logger=False,
    )

    results = trainer.test(module, test_dl)

    print('\n' + '=' * 50)
    print('  FINAL TEST RESULTS  (DeepCrack)')
    print('=' * 50)
    for key, val in results[0].items():
        print(f'  {key:<20s}: {val:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
