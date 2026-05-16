#!/usr/bin/env python3
"""
Train EnhancedGraphUNet (Hybrid CNN-GNN) on road crack dataset.

Pipeline: Image Segmentation -> Graph Conversion -> GNN Link Prediction
This script covers Stage 1: semantic mask prediction (crack vs background).

Default dataset: DeepCrack
  Layout:
    root/train_img/  <- RGB images
    root/train_lab/  <- binary masks (white=crack, black=background)
    root/test_img/
    root/test_lab/

Run:
    cd /Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/segmentation
    python train.py \\
        --train-img-dir  /path/to/DeepCrack/train_img \\
        --train-mask-dir /path/to/DeepCrack/train_lab \\
        --val-img-dir    /path/to/DeepCrack/test_img \\
        --val-mask-dir   /path/to/DeepCrack/test_lab
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# make graph_layers importable when running from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import EnhancedGraphUNet
from dataset import build_dataloaders


# ── Loss ─────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """0.5 * CrossEntropy + 0.5 * SoftDice — handles class imbalance for thin cracks."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss()
        self.smooth = smooth

    def _dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n = logits.shape[1]
        total = 0.0
        for cls in range(n):
            p     = probs[:, cls]
            t     = (targets == cls).float()
            inter = (p * t).sum()
            total += (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 1.0 - total / n

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self._dice(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred, target, num_classes=2, smooth=1e-6):
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).float().sum()
        union = ((pred == cls) | (target == cls)).float().sum()
        ious.append((inter + smooth) / (union + smooth))
    return sum(ious) / len(ious)


def compute_dice(pred, target, num_classes=2, smooth=1e-6):
    scores = []
    for cls in range(num_classes):
        p     = (pred   == cls).float()
        t     = (target == cls).float()
        inter = (p * t).sum()
        scores.append((2 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return sum(scores) / len(scores)


def compute_accuracy(pred, target):
    return (pred == target).float().sum() / target.numel()


# ── Lightning Module ───────────────────────────────────────────────────────────

class SegModule(pl.LightningModule):
    def __init__(self, model, learning_rate=3e-4, weight_decay=1e-4, max_epochs=100):
        super().__init__()
        self.model         = model
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.max_epochs    = max_epochs
        self.criterion     = CombinedLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        loss = self.criterion(self(imgs), masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        iou  = compute_iou(preds, masks)
        dice = compute_dice(preds, masks)
        acc  = compute_accuracy(preds, masks)

        self.log('val/loss', loss,  prog_bar=True, sync_dist=True)
        self.log('val/iou',  iou,   prog_bar=True, sync_dist=True)
        self.log('val/dice', dice,  prog_bar=True, sync_dist=True)
        self.log('val/acc',  acc,   prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.logger:
            self._log_images(imgs, preds, masks)

        return {'val_loss': loss, 'iou': iou, 'dice': dice, 'acc': acc}

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        iou  = compute_iou(preds, masks)
        dice = compute_dice(preds, masks)
        acc  = compute_accuracy(preds, masks)

        self.log('test/loss', loss, sync_dist=True)
        self.log('test/iou',  iou,  sync_dist=True)
        self.log('test/dice', dice, sync_dist=True)
        self.log('test/acc',  acc,  sync_dist=True)
        return {'test_loss': loss, 'test_iou': iou, 'test_dice': dice, 'test_acc': acc}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}

    def _log_images(self, imgs, preds, masks):
        imgs_norm = (imgs + 1.0) / 2.0
        self.logger.experiment.log({
            'val/images':      [wandb.Image(img.cpu()) for img in imgs_norm[:4]],
            'val/predictions': [wandb.Image(p.unsqueeze(0).float().cpu()) for p in preds[:4]],
            'val/masks':       [wandb.Image(m.unsqueeze(0).float().cpu()) for m in masks[:4]],
            'global_step': self.global_step,
        })


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train EnhancedGraphUNet for road crack segmentation')
    p.add_argument('--train-img-dir',  type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/train/images')
    p.add_argument('--train-mask-dir', type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/train/masks')
    p.add_argument('--val-img-dir',    type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/images')
    p.add_argument('--val-mask-dir',   type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/masks')
    p.add_argument('--image-size',      type=int,   default=256)
    p.add_argument('--batch-size',      type=int,   default=8)
    p.add_argument('--num-workers',     type=int,   default=2)
    p.add_argument('--learning-rate',   type=float, default=3e-4)
    p.add_argument('--weight-decay',    type=float, default=1e-4)
    p.add_argument('--max-epochs',      type=int,   default=100)
    p.add_argument('--precision',       type=int,   default=16)
    p.add_argument('--output-dir',      type=str,   default='../outputs/segmentation')
    p.add_argument('--experiment-name', type=str,   default='crack_graphunet')
    p.add_argument('--checkpoint-path', type=str,   default=None)
    p.add_argument('--no-wandb',        action='store_true', help='Disable WandB logging')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = EnhancedGraphUNet(in_channels=3, out_channels=2, features=(32, 64, 128, 256))

    if args.checkpoint_path:
        print(f'Resuming from: {args.checkpoint_path}')
        module = SegModule.load_from_checkpoint(
            args.checkpoint_path, model=base_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
        )
    else:
        module = SegModule(
            model=base_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
        )

    train_dl, val_dl = build_dataloaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir,   args.val_mask_dir,
        args.batch_size, args.num_workers, args.image_size,
    )

    ckpt_cb  = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}-{{epoch:02d}}-val_iou={{val/iou:.4f}}',
        monitor='val/iou', mode='max', save_top_k=3,
    )
    early_cb = EarlyStopping(monitor='val/iou', mode='max', patience=20, verbose=True)
    callbacks = [ckpt_cb, early_cb]

    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.experiment_name,
            name=f'graphunet-bs{args.batch_size}-lr{args.learning_rate}-sz{args.image_size}',
            save_dir=args.output_dir,
        )
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    if torch.backends.mps.is_available():
        accelerator, devices = 'mps', 1
    elif torch.cuda.is_available():
        accelerator, devices = 'gpu', 1
    else:
        accelerator, devices = 'cpu', 1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dl, val_dl)
    print(f'\nBest checkpoint : {ckpt_cb.best_model_path}')
    print(f'Best val/iou    : {ckpt_cb.best_model_score:.4f}')


if __name__ == '__main__':
    main()
