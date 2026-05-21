#!/usr/bin/env python3
"""
Train HybridGraphUNet (Pretrained CNN encoder + GNN bottleneck) on road crack dataset.

Key changes over v1 (scratch-trained EnhancedGraphUNet):
  - Pretrained ResNet34d encoder (ImageNet) — biggest win for small datasets
  - Focal Loss replaces plain CE — focuses gradient on hard/misclassified pixels
  - Differential LR: encoder gets 10x lower LR to preserve pretrained weights
  - Gradient clipping, persistent workers, correct mixed-precision string

Run:
    cd /Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/segmentation
    python train.py --no-wandb --image-size 512
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HybridGraphUNet
from dataset import build_dataloaders, CrackDataset, get_transforms
from torch.utils.data import DataLoader


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss — down-weights easy background pixels, focuses on hard crack pixels."""

    def __init__(self, gamma: float = 2.0, crack_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('class_weights', torch.tensor([1.0, crack_weight]))

    def forward(self, logits, targets):
        ce      = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        p_t     = (probs * one_hot).sum(dim=1)
        return ((1 - p_t) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    """0.5 * FocalLoss + 0.5 * CrackDice.

    Focal handles class imbalance at the pixel level.
    CrackDice drives overlap specifically on the thin crack class.
    """

    def __init__(self, smooth: float = 1e-6, crack_weight: float = 10.0, focal_gamma: float = 2.0):
        super().__init__()
        self.focal  = FocalLoss(gamma=focal_gamma, crack_weight=crack_weight)
        self.smooth = smooth

    def _crack_dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        t     = (targets == 1).float()
        inter = (probs * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self._crack_dice(logits, targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred, target, num_classes=2, smooth=1e-6):
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).float().sum()
        union = ((pred == cls) | (target == cls)).float().sum()
        ious.append((inter + smooth) / (union + smooth))
    return sum(ious) / len(ious)


def compute_crack_iou(pred, target, smooth=1e-6):
    """IoU for crack class only (class=1)."""
    inter = ((pred == 1) & (target == 1)).float().sum()
    union = ((pred == 1) | (target == 1)).float().sum()
    return (inter + smooth) / (union + smooth)


def compute_dice(pred, target, num_classes=2, smooth=1e-6):
    scores = []
    for cls in range(num_classes):
        p     = (pred   == cls).float()
        t     = (target == cls).float()
        inter = (p * t).sum()
        scores.append((2 * inter + smooth) / (p.sum() + t.sum() + smooth))
    return sum(scores) / len(scores)


def compute_crack_dice(pred, target, smooth=1e-6):
    """Dice for crack class only (class=1)."""
    p     = (pred   == 1).float()
    t     = (target == 1).float()
    inter = (p * t).sum()
    return (2 * inter + smooth) / (p.sum() + t.sum() + smooth)


def compute_accuracy(pred, target):
    return (pred == target).float().sum() / target.numel()


# ── Lightning Module ───────────────────────────────────────────────────────────

class SegModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        weight_decay: float  = 1e-4,
        max_epochs: int      = 150,
        crack_weight: float  = 10.0,
        encoder_lr_scale: float = 0.1,
    ):
        super().__init__()
        self.model            = model
        self.learning_rate    = learning_rate
        self.weight_decay     = weight_decay
        self.max_epochs       = max_epochs
        self.encoder_lr_scale = encoder_lr_scale
        self.criterion        = CombinedLoss(crack_weight=crack_weight)

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

        iou        = compute_iou(preds, masks)
        crack_iou  = compute_crack_iou(preds, masks)
        dice       = compute_dice(preds, masks)
        crack_dice = compute_crack_dice(preds, masks)
        acc        = compute_accuracy(preds, masks)

        self.log('val/loss',       loss,       prog_bar=True,  sync_dist=True)
        self.log('val/iou',        iou,        prog_bar=False, sync_dist=True)
        self.log('val/crack_iou',  crack_iou,  prog_bar=True,  sync_dist=True)
        self.log('val/dice',       dice,       prog_bar=False, sync_dist=True)
        self.log('val/crack_dice', crack_dice, prog_bar=True,  sync_dist=True)
        self.log('val/acc',        acc,        prog_bar=False, sync_dist=True)

        if batch_idx == 0 and self.logger:
            self._log_images(imgs, preds, masks)

        return {'val_loss': loss, 'iou': iou, 'crack_iou': crack_iou,
                'dice': dice, 'crack_dice': crack_dice, 'acc': acc}

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        iou        = compute_iou(preds, masks)
        crack_iou  = compute_crack_iou(preds, masks)
        dice       = compute_dice(preds, masks)
        crack_dice = compute_crack_dice(preds, masks)
        acc        = compute_accuracy(preds, masks)

        self.log('test/loss',       loss,       sync_dist=True)
        self.log('test/iou',        iou,        sync_dist=True)
        self.log('test/crack_iou',  crack_iou,  sync_dist=True)
        self.log('test/dice',       dice,       sync_dist=True)
        self.log('test/crack_dice', crack_dice, sync_dist=True)
        self.log('test/acc',        acc,        sync_dist=True)
        return {'test_loss': loss, 'test_iou': iou, 'test_crack_iou': crack_iou,
                'test_dice': dice, 'test_crack_dice': crack_dice, 'test_acc': acc}

    def configure_optimizers(self):
        # Differential LR: pretrained encoder gets a 10x lower rate
        enc_params = self.model.encoder_params() if hasattr(self.model, 'encoder_params') else []
        dec_params = self.model.decoder_params() if hasattr(self.model, 'decoder_params') else list(self.model.parameters())

        param_groups = [
            {'params': enc_params, 'lr': self.learning_rate * self.encoder_lr_scale},
            {'params': dec_params, 'lr': self.learning_rate},
        ]
        opt   = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
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
    p = argparse.ArgumentParser(description='Train HybridGraphUNet for road crack segmentation')
    p.add_argument('--train-img-dir',  type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/train/images')
    p.add_argument('--train-mask-dir', type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/train/masks')
    p.add_argument('--val-img-dir',    type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/images')
    p.add_argument('--val-mask-dir',   type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/masks')
    p.add_argument('--test-img-dir',   type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/test/images')
    p.add_argument('--test-mask-dir',  type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/test/masks')
    p.add_argument('--encoder-name',    type=str,   default='resnet34d',
                   help='timm encoder name (e.g. resnet34d, efficientnet_b4, convnext_tiny)')
    p.add_argument('--no-pretrained',   action='store_true', help='Train encoder from scratch')
    p.add_argument('--image-size',      type=int,   default=512)
    p.add_argument('--batch-size',      type=int,   default=8)
    p.add_argument('--num-workers',     type=int,   default=2)
    p.add_argument('--learning-rate',   type=float, default=1e-4)
    p.add_argument('--encoder-lr-scale',type=float, default=0.1,
                   help='Encoder LR = learning-rate * encoder-lr-scale')
    p.add_argument('--weight-decay',    type=float, default=1e-4)
    p.add_argument('--crack-weight',    type=float, default=10.0,
                   help='Class weight for crack pixels in Focal loss')
    p.add_argument('--max-epochs',      type=int,   default=150)
    p.add_argument('--precision',       type=str,   default='16-mixed')
    p.add_argument('--output-dir',      type=str,   default='../outputs/segmentation')
    p.add_argument('--experiment-name', type=str,   default='crack_hybrid_gnn')
    p.add_argument('--checkpoint-path', type=str,   default=None)
    p.add_argument('--no-wandb',        action='store_true', help='Disable WandB logging')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = HybridGraphUNet(
        encoder_name=args.encoder_name,
        pretrained=not args.no_pretrained,
        out_channels=2,
    )

    if args.checkpoint_path:
        print(f'Resuming from: {args.checkpoint_path}')
        module = SegModule.load_from_checkpoint(
            args.checkpoint_path,
            model=base_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            crack_weight=args.crack_weight,
            encoder_lr_scale=args.encoder_lr_scale,
        )
    else:
        module = SegModule(
            model=base_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            crack_weight=args.crack_weight,
            encoder_lr_scale=args.encoder_lr_scale,
        )

    train_dl, val_dl = build_dataloaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir,   args.val_mask_dir,
        args.batch_size, args.num_workers, args.image_size,
    )

    ckpt_cb   = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}-ep{{epoch:02d}}-ciou={{val/crack_iou:.4f}}',
        monitor='val/crack_iou', mode='max', save_top_k=3, save_last=True,
    )
    early_cb  = EarlyStopping(monitor='val/crack_iou', mode='max', patience=25, verbose=True)
    callbacks = [ckpt_cb, early_cb]

    logger = False
    if not args.no_wandb:
        logger = WandbLogger(
            project=args.experiment_name,
            name=f'{args.encoder_name}-bs{args.batch_size}-lr{args.learning_rate}-sz{args.image_size}',
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
        gradient_clip_val=1.0,
    )

    trainer.fit(module, train_dl, val_dl)
    print(f'\nBest checkpoint : {ckpt_cb.best_model_path}')
    print(f'Best val/crack_iou : {ckpt_cb.best_model_score:.4f}')

    # ── Test evaluation on held-out test set ──────────────────────────────────
    if os.path.isdir(args.test_img_dir) and os.path.isdir(args.test_mask_dir):
        print('\nRunning evaluation on held-out test set...')
        _, test_tf = get_transforms(args.image_size)
        test_ds    = CrackDataset(img_dir=args.test_img_dir, mask_dir=args.test_mask_dir,
                                  transform=test_tf)
        pin_memory = torch.cuda.is_available()
        test_dl    = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin_memory)

        results = trainer.test(module, test_dl, ckpt_path='best')
        print('\n' + '=' * 50)
        print('  FINAL TEST RESULTS')
        print('=' * 50)
        for k, v in results[0].items():
            print(f'  {k:<25s}: {v:.4f}')
        print('=' * 50)
    else:
        print(f'\nNo test set found at {args.test_img_dir} — skipping test evaluation.')


if __name__ == '__main__':
    main()
