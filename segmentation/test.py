#!/usr/bin/env python3
"""
Evaluate a trained HybridGraphUNet on the held-out test split.

Run:
    cd /Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/segmentation
    python test.py --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt
"""

import os
import sys
import argparse

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HybridGraphUNet
from train import SegModule
from dataset import CrackDataset, get_transforms


def parse_args():
    p = argparse.ArgumentParser(description='Test HybridGraphUNet on held-out crack test set')
    p.add_argument('--test-img-dir',    type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/test/images')
    p.add_argument('--test-mask-dir',   type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/test/masks')
    p.add_argument('--checkpoint-path', type=str, required=True,
                   help='Path to .ckpt file saved during training')
    p.add_argument('--encoder-name',    type=str, default='resnet34d')
    p.add_argument('--image-size',      type=int, default=512)
    p.add_argument('--batch-size',      type=int, default=8)
    p.add_argument('--num-workers',     type=int, default=2)
    p.add_argument('--precision',       type=str, default='16-mixed')
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)

    base_model = HybridGraphUNet(
        encoder_name=args.encoder_name,
        pretrained=False,   # weights come from the checkpoint
        out_channels=2,
    )

    print(f'Loading checkpoint: {args.checkpoint_path}')
    module = SegModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=base_model,
        strict=False,
    )
    module.eval()

    _, test_tf = get_transforms(args.image_size)
    test_ds    = CrackDataset(
        img_dir=args.test_img_dir, mask_dir=args.test_mask_dir, transform=test_tf
    )
    pin_memory = torch.cuda.is_available()
    test_dl    = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
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
    print('  FINAL TEST RESULTS  (held-out test set)')
    print('=' * 50)
    for k, v in results[0].items():
        print(f'  {k:<25s}: {v:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
