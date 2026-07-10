#!/usr/bin/env python3
"""
Phase 1 GraphUNet — training on crack_seg_clean.

Reproduces the Phase 1 training setup (BCE + Dice loss, scratch training,
cosine LR, AdamW) on the same crack_seg_clean dataset used in Phase 2
so all segmentation models are directly comparable.

Usage:
    cd crack-topology-gnn/segmentation/phase_1
    source ../../../.venv/bin/activate

    python train.py

    # Custom paths:
    python train.py \\
        --train-img-dir  "/path/to/crack_seg_clean/clean/train/images" \\
        --train-mask-dir "/path/to/crack_seg_clean/clean/train/masks" \\
        --val-img-dir    "/path/to/crack_seg_clean/clean/val/images" \\
        --val-mask-dir   "/path/to/crack_seg_clean/clean/val/masks" \\
        --test-img-dir   "/path/to/crack_seg_clean/clean/test/images" \\
        --test-mask-dir  "/path/to/crack_seg_clean/clean/test/masks" \\
        --output-dir     ../../outputs/seg_phase1
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import dataset utilities from the parent segmentation directory (no changes to parent files)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CrackDataset, get_transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GraphUNet


CLEAN      = '/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean'
CLEAN_TEST = f'{CLEAN}/test'   # used as val too (no separate val split)
CLEAN_TRAIN = f'{CLEAN}/train'


# ── Loss (Phase 1: BCE + Dice, 50/50) ────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    def __init__(self, crack_weight: float = 10.0, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.register_buffer('weights', torch.tensor([1.0, crack_weight]))

    def forward(self, logits, targets):
        bce  = F.cross_entropy(logits, targets, weight=self.weights)
        prob = F.softmax(logits, dim=1)[:, 1]
        t    = (targets == 1).float()
        inter = (prob * t).sum()
        dice  = 1.0 - (2.0 * inter + self.smooth) / (prob.sum() + t.sum() + self.smooth)
        return 0.5 * bce + 0.5 * dice


# ── Metrics ───────────────────────────────────────────────────────────────────

def _dilate_binary(mask, radius=2):
    k = 2 * radius + 1
    return F.max_pool2d(mask.float().unsqueeze(1), k, stride=1, padding=radius).squeeze(1).bool()


def buf_to_metrics(tp, fp, fn, tn, smooth=1e-6):
    crack_iou  = (tp + smooth) / (tp + fp + fn + smooth)
    crack_dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    crack_prec = (tp + smooth) / (tp + fp + smooth)
    crack_rec  = (tp + smooth) / (tp + fn + smooth)
    bg_iou     = (tn + smooth) / (tn + fp + fn + smooth)
    mean_iou   = 0.5 * (crack_iou + bg_iou)
    acc        = (tp + tn) / (tp + tn + fp + fn + smooth)
    return dict(crack_iou=crack_iou, crack_dice=crack_dice,
                crack_prec=crack_prec, crack_rec=crack_rec,
                mean_iou=mean_iou, acc=acc)


# ── Lightning Module ───────────────────────────────────────────────────────────

class Phase1Module(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float  = 1e-3,
        weight_decay: float   = 1e-4,
        max_epochs: int       = 100,
        crack_weight: float   = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model     = GraphUNet(in_channels=3, out_channels=2)
        self.criterion = BCEDiceLoss(crack_weight=crack_weight)
        self._reset_buf('val')
        self._reset_buf('test')

    def _reset_buf(self, split):
        setattr(self, f'_{split}_tp', 0)
        setattr(self, f'_{split}_fp', 0)
        setattr(self, f'_{split}_fn', 0)
        setattr(self, f'_{split}_tn', 0)
        setattr(self, f'_{split}_loss', 0.0)
        setattr(self, f'_{split}_n', 0)
        # Tolerant counts
        setattr(self, f'_{split}_tol_tp', 0)
        setattr(self, f'_{split}_tol_fp', 0)
        setattr(self, f'_{split}_tol_fn', 0)

    def _accumulate(self, split, preds, masks, loss):
        c_pred = (preds == 1)
        c_true = (masks == 1)
        setattr(self, f'_{split}_tp', getattr(self, f'_{split}_tp') + int((c_pred &  c_true).sum()))
        setattr(self, f'_{split}_fp', getattr(self, f'_{split}_fp') + int((c_pred & ~c_true).sum()))
        setattr(self, f'_{split}_fn', getattr(self, f'_{split}_fn') + int((~c_pred & c_true).sum()))
        setattr(self, f'_{split}_tn', getattr(self, f'_{split}_tn') + int((~c_pred & ~c_true).sum()))
        setattr(self, f'_{split}_loss', getattr(self, f'_{split}_loss') + loss.item())
        setattr(self, f'_{split}_n',    getattr(self, f'_{split}_n') + 1)
        # Tolerant (2px dilation)
        c_tol = _dilate_binary(c_true, radius=2)
        setattr(self, f'_{split}_tol_tp', getattr(self, f'_{split}_tol_tp') + int((c_pred &  c_tol).sum()))
        setattr(self, f'_{split}_tol_fp', getattr(self, f'_{split}_tol_fp') + int((c_pred & ~c_tol).sum()))
        setattr(self, f'_{split}_tol_fn', getattr(self, f'_{split}_tol_fn') + int((~c_pred & c_true).sum()))

    def _log_metrics(self, split):
        tp = getattr(self, f'_{split}_tp')
        fp = getattr(self, f'_{split}_fp')
        fn = getattr(self, f'_{split}_fn')
        tn = getattr(self, f'_{split}_tn')
        m  = buf_to_metrics(tp, fp, fn, tn)
        t_tp = getattr(self, f'_{split}_tol_tp')
        t_fp = getattr(self, f'_{split}_tol_fp')
        t_fn = getattr(self, f'_{split}_tol_fn')
        smooth = 1e-6
        tol_iou  = (t_tp + smooth) / (t_tp + t_fp + t_fn + smooth)
        tol_dice = (2*t_tp + smooth) / (2*t_tp + t_fp + t_fn + smooth)
        self.log(f'{split}/crack_iou',      m['crack_iou'],  prog_bar=(split=='val'), sync_dist=True)
        self.log(f'{split}/crack_dice',     m['crack_dice'], prog_bar=(split=='val'), sync_dist=True)
        self.log(f'{split}/crack_prec',     m['crack_prec'], sync_dist=True)
        self.log(f'{split}/crack_rec',      m['crack_rec'],  sync_dist=True)
        self.log(f'{split}/mean_iou',       m['mean_iou'],   sync_dist=True)
        self.log(f'{split}/acc',            m['acc'],        sync_dist=True)
        self.log(f'{split}/tol_crack_iou',  tol_iou,         sync_dist=True)
        self.log(f'{split}/tol_crack_dice', tol_dice,        sync_dist=True)
        return m

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        loss = self.criterion(self(imgs), masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._reset_buf('val')

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        self._accumulate('val', logits.argmax(dim=1), masks, loss)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self._log_metrics('val')

    def on_test_epoch_start(self):
        self._reset_buf('test')

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        self._accumulate('test', logits.argmax(dim=1), masks, loss)
        self.log('test/loss', loss, sync_dist=True)

    def on_test_epoch_end(self):
        self._log_metrics('test')

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(),
                                  lr=self.hparams.learning_rate,
                                  weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Phase 1 GraphUNet on crack_seg_clean')
    p.add_argument('--train-img-dir',  default=f'{CLEAN_TRAIN}/images')
    p.add_argument('--train-mask-dir', default=f'{CLEAN_TRAIN}/masks')
    p.add_argument('--val-img-dir',    default=f'{CLEAN_TEST}/images')
    p.add_argument('--val-mask-dir',   default=f'{CLEAN_TEST}/masks')
    p.add_argument('--test-img-dir',   default=f'{CLEAN_TEST}/images')
    p.add_argument('--test-mask-dir',  default=f'{CLEAN_TEST}/masks')
    p.add_argument('--output-dir',     default='../../outputs/seg_phase1')
    p.add_argument('--image-size',     type=int,   default=448)
    p.add_argument('--batch-size',     type=int,   default=8)
    p.add_argument('--num-workers',    type=int,   default=2)
    p.add_argument('--learning-rate',  type=float, default=1e-3)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--crack-weight',   type=float, default=10.0)
    p.add_argument('--max-epochs',     type=int,   default=100)
    p.add_argument('--patience',       type=int,   default=25,
                   help='Early stopping patience on val/crack_iou')
    p.add_argument('--precision',      default='16-mixed')
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    train_tf, val_tf = get_transforms(args.image_size)
    pin = torch.cuda.is_available()

    from torch.utils.data import DataLoader
    train_ds = CrackDataset(img_dir=args.train_img_dir, mask_dir=args.train_mask_dir, transform=train_tf)
    val_ds   = CrackDataset(img_dir=args.val_img_dir,   mask_dir=args.val_mask_dir,   transform=val_tf)
    test_ds  = CrackDataset(img_dir=args.test_img_dir,  mask_dir=args.test_mask_dir,  transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=(args.num_workers > 0), drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=(args.num_workers > 0))
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=(args.num_workers > 0))

    module = Phase1Module(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        crack_weight=args.crack_weight,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='phase1-{epoch:03d}-ciou={val/crack_iou:.4f}',
        monitor='val/crack_iou',
        mode='max',
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    stop_cb = EarlyStopping(
        monitor='val/crack_iou',
        patience=args.patience,
        mode='max',
        verbose=True,
    )

    accelerator = 'mps' if torch.backends.mps.is_available() else ('gpu' if torch.cuda.is_available() else 'cpu')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=args.precision,
        callbacks=[ckpt_cb, stop_cb],
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    trainer.fit(module, train_dl, val_dl)

    print(f'\nBest checkpoint: {ckpt_cb.best_model_path}')
    print('Running test set evaluation...')
    results = trainer.test(module, test_dl, ckpt_path='best')[0]

    out_path = os.path.join(args.output_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Test metrics saved → {out_path}')
    print(f'\n  crack_iou  : {results.get("test/crack_iou", "?"):.4f}')
    print(f'  crack_dice : {results.get("test/crack_dice", "?"):.4f}')
    print(f'  crack_prec : {results.get("test/crack_prec", "?"):.4f}')
    print(f'  crack_rec  : {results.get("test/crack_rec", "?"):.4f}')


if __name__ == '__main__':
    main()
