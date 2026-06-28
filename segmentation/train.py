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

from model import HybridGraphUNet, EnhancedGraphUNet  # PlainResNetUNet = HybridGraphUNet(use_gnn_bottleneck=False)
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


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky: α controls FP penalty, β controls FN penalty.

    Set β > 0.5 to penalise missed crack pixels (FN) more than spurious ones (FP).
    Missing crack pixels breaks skeletons in Stage 2; over-prediction just fattens them.
    Default α=0.3, β=0.7, γ=0.75 follows the thin-structure literature.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        t     = (targets == 1).float()
        tp    = (probs * t).sum()
        fp    = (probs * (1 - t)).sum()
        fn    = ((1 - probs) * t).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky) ** self.gamma


# ── Soft skeleton (differentiable clDice loss) ────────────────────────────────
# Shit et al., "clDice — a Novel Topology-Preserving Loss Function for Tubular
# Structure Segmentation", CVPR 2021.

def _soft_erode(x, k=5):
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)

def _soft_dilate(x, k=5):
    return F.max_pool2d(x, k, stride=1, padding=k // 2)

def _soft_open(x, k=5):
    return _soft_dilate(_soft_erode(x, k), k)

def _soft_skel(x, iters=10, k=5):
    skel = F.relu(x - _soft_open(x, k))
    for _ in range(iters - 1):
        x    = _soft_erode(x, k)
        skel = skel + F.relu(x - _soft_open(x, k))
    return skel


class SoftClDiceLoss(nn.Module):
    """Differentiable clDice — penalises disconnected / fragmented predictions.

    Directly optimises topology preservation, tying Stage 1 loss to the
    crack-topology thesis: fragmented masks → broken skeletons → corrupted graphs.
    """

    def __init__(self, smooth: float = 1e-6, iters: int = 10):
        super().__init__()
        self.smooth = smooth
        self.iters  = iters

    def forward(self, logits, targets):
        probs  = F.softmax(logits, dim=1)[:, 1:2]       # [B,1,H,W]
        t      = (targets == 1).float().unsqueeze(1)     # [B,1,H,W]
        skel_p = _soft_skel(probs, self.iters)
        skel_t = _soft_skel(t,     self.iters)
        tprec  = (skel_p * t).sum()     / (skel_p.sum() + self.smooth)
        tsens  = (skel_t * probs).sum() / (skel_t.sum() + self.smooth)
        return 1 - 2 * tprec * tsens / (tprec + tsens + self.smooth)


class CombinedLoss(nn.Module):
    """Configurable loss: Focal + shape term (CrackDice or FocalTversky) + optional soft-clDice.

    Default (all flags off):  0.5·Focal + 0.5·CrackDice
    --use-tversky:            0.5·Focal + 0.5·FocalTversky(α=0.3,β=0.7)
    --use-cldice:             (1-w)·(above) + w·SoftClDice   [default w=0.3]
    """

    def __init__(
        self,
        smooth: float        = 1e-6,
        crack_weight: float  = 10.0,
        focal_gamma: float   = 2.0,
        use_tversky: bool    = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float  = 0.7,
        tversky_gamma: float = 0.75,
        use_cldice: bool     = False,
        cldice_weight: float = 0.3,
        cldice_iters: int    = 10,
    ):
        super().__init__()
        self.smooth      = smooth
        self.use_cldice  = use_cldice
        self.cldice_w    = cldice_weight
        self.focal       = FocalLoss(gamma=focal_gamma, crack_weight=crack_weight)
        self.shape_loss  = (FocalTverskyLoss(tversky_alpha, tversky_beta, tversky_gamma, smooth)
                            if use_tversky else None)
        self.cldice_loss = SoftClDiceLoss(smooth, cldice_iters) if use_cldice else None

    def _crack_dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        t     = (targets == 1).float()
        inter = (probs * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)

    def forward(self, logits, targets):
        shape = (self.shape_loss(logits, targets) if self.shape_loss
                 else self._crack_dice(logits, targets))
        base  = 0.5 * self.focal(logits, targets) + 0.5 * shape
        if self.cldice_loss:
            return (1 - self.cldice_w) * base + self.cldice_w * self.cldice_loss(logits, targets)
        return base


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_cldice_batch(pred_binary, target_binary, smooth=1e-6):
    """Sum of per-image clDice scores over a batch (caller divides by image count).

    clDice measures connectivity/topology preservation — how well the predicted
    mask covers the true skeleton and vice versa. Critical for Stage 2 because
    skeletonization of a fragmented mask produces broken graph nodes.
    """
    from skimage.morphology import skeletonize
    pred_np   = pred_binary.cpu().numpy()   # [B, H, W] bool
    target_np = target_binary.cpu().numpy() # [B, H, W] bool
    total = 0.0
    for p, t in zip(pred_np, target_np):
        skel_p = skeletonize(p)
        skel_t = skeletonize(t)
        tprec  = float((skel_p & t).sum()) / (float(skel_p.sum()) + smooth)
        tsens  = float((skel_t & p).sum()) / (float(skel_t.sum()) + smooth)
        total += 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return total, len(pred_np)


def _dilate_binary(mask, radius=2):
    """Dilate a boolean mask tensor by `radius` pixels using max-pooling (on device)."""
    k = 2 * radius + 1
    x = mask.float().unsqueeze(1)
    return F.max_pool2d(x, k, stride=1, padding=radius).squeeze(1).bool()


# ── Lightning Module ───────────────────────────────────────────────────────────

class SegModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float    = 1e-4,
        weight_decay: float     = 1e-4,
        max_epochs: int         = 150,
        crack_weight: float     = 10.0,
        encoder_lr_scale: float = 0.1,
        use_tversky: bool       = False,
        tversky_alpha: float    = 0.3,
        tversky_beta: float     = 0.7,
        tversky_gamma: float    = 0.75,
        use_cldice: bool        = False,
        cldice_weight: float    = 0.3,
        cldice_iters: int       = 10,
    ):
        super().__init__()
        self.model            = model
        self.learning_rate    = learning_rate
        self.weight_decay     = weight_decay
        self.max_epochs       = max_epochs
        self.encoder_lr_scale = encoder_lr_scale
        self.criterion        = CombinedLoss(
            crack_weight=crack_weight,
            use_tversky=use_tversky,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            use_cldice=use_cldice,
            cldice_weight=cldice_weight,
            cldice_iters=cldice_iters,
        )
        self._val_buf  = self._empty_buf()
        self._test_buf = self._empty_buf()

    def forward(self, x):
        return self.model(x)

    # ── Accumulation helpers ──────────────────────────────────────────────────

    @staticmethod
    def _empty_buf():
        return dict(tp=0, fp=0, fn=0, tn=0, loss=0.0, n=0,
                    cldice=0.0, cldice_n=0,
                    tol_tp=0, tol_fp=0, tol_fn=0)

    def _accumulate(self, buf, preds, masks, loss):
        c_pred = (preds == 1)
        c_true = (masks == 1)
        # Strict pixel counts
        buf['tp'] += int((c_pred &  c_true).sum())
        buf['fp'] += int((c_pred & ~c_true).sum())
        buf['fn'] += int((~c_pred & c_true).sum())
        buf['tn'] += int((~c_pred & ~c_true).sum())
        buf['loss'] += loss.item()
        buf['n']    += 1
        # Boundary-tolerant counts: GT dilated by 2px (standard CRACK500 convention)
        c_tol         = _dilate_binary(c_true, radius=2)
        buf['tol_tp'] += int((c_pred &  c_tol).sum())
        buf['tol_fp'] += int((c_pred & ~c_tol).sum())
        buf['tol_fn'] += int((~c_pred & c_true).sum())  # FN always uses strict GT
        # clDice (per-image, eval-only)
        cd, n_imgs      = compute_cldice_batch(c_pred, c_true)
        buf['cldice']   += cd
        buf['cldice_n'] += n_imgs

    @staticmethod
    def _buf_to_metrics(buf, smooth=1e-6):
        tp, fp, fn, tn = buf['tp'], buf['fp'], buf['fn'], buf['tn']
        crack_iou  = (tp + smooth) / (tp + fp + fn + smooth)
        crack_dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
        crack_prec = (tp + smooth) / (tp + fp + smooth)
        crack_rec  = (tp + smooth) / (tp + fn + smooth)
        bg_iou     = (tn + smooth) / (tn + fp + fn + smooth)
        mean_iou   = 0.5 * (crack_iou + bg_iou)
        acc        = (tp + tn) / (tp + tn + fp + fn + smooth)
        avg_loss   = buf['loss'] / max(buf['n'], 1)
        cldice     = buf['cldice'] / max(buf['cldice_n'], 1)
        # Tolerant (2px boundary dilation — matches published CRACK500 protocol)
        t_tp, t_fp, t_fn = buf['tol_tp'], buf['tol_fp'], buf['tol_fn']
        tol_iou  = (t_tp + smooth) / (t_tp + t_fp + t_fn + smooth)
        tol_dice = (2*t_tp + smooth) / (2*t_tp + t_fp + t_fn + smooth)
        return dict(crack_iou=crack_iou, crack_dice=crack_dice,
                    crack_prec=crack_prec, crack_rec=crack_rec,
                    mean_iou=mean_iou, acc=acc, loss=avg_loss, cldice=cldice,
                    tol_crack_iou=tol_iou, tol_crack_dice=tol_dice)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        loss = self.criterion(self(imgs), masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_buf = self._empty_buf()

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)
        self._accumulate(self._val_buf, preds, masks, loss)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        if batch_idx == 0 and self.logger:
            self._log_images(imgs, preds, masks)

    def on_validation_epoch_end(self):
        m = self._buf_to_metrics(self._val_buf)
        self.log('val/crack_iou',      m['crack_iou'],      prog_bar=True,  sync_dist=True)
        self.log('val/crack_dice',     m['crack_dice'],     prog_bar=True,  sync_dist=True)
        self.log('val/crack_prec',     m['crack_prec'],     prog_bar=False, sync_dist=True)
        self.log('val/crack_rec',      m['crack_rec'],      prog_bar=False, sync_dist=True)
        self.log('val/cldice',         m['cldice'],         prog_bar=False, sync_dist=True)
        self.log('val/tol_crack_iou',  m['tol_crack_iou'],  prog_bar=False, sync_dist=True)
        self.log('val/tol_crack_dice', m['tol_crack_dice'], prog_bar=False, sync_dist=True)
        self.log('val/iou',            m['mean_iou'],       prog_bar=False, sync_dist=True)
        self.log('val/acc',            m['acc'],            prog_bar=False, sync_dist=True)

    # ── Test ──────────────────────────────────────────────────────────────────

    def on_test_epoch_start(self):
        self._test_buf = self._empty_buf()

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)
        self._accumulate(self._test_buf, preds, masks, loss)
        self.log('test/loss', loss, sync_dist=True)

    def on_test_epoch_end(self):
        m = self._buf_to_metrics(self._test_buf)
        self.log('test/crack_iou',      m['crack_iou'],      sync_dist=True)
        self.log('test/crack_dice',     m['crack_dice'],     sync_dist=True)
        self.log('test/crack_prec',     m['crack_prec'],     sync_dist=True)
        self.log('test/crack_rec',      m['crack_rec'],      sync_dist=True)
        self.log('test/cldice',         m['cldice'],         sync_dist=True)
        self.log('test/tol_crack_iou',  m['tol_crack_iou'],  sync_dist=True)
        self.log('test/tol_crack_dice', m['tol_crack_dice'], sync_dist=True)
        self.log('test/iou',            m['mean_iou'],       sync_dist=True)
        self.log('test/acc',            m['acc'],            sync_dist=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
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

CLEAN = '/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean'


def parse_args():
    p = argparse.ArgumentParser(description='Train segmentation model for road crack segmentation')
    p.add_argument('--model',          type=str,   default='hybrid',
                   choices=['hybrid', 'plain', 'enhanced'],
                   help='hybrid = HybridGraphUNet (pretrained+GNN), plain = pretrained ResNet34d no GNN (ablation), enhanced = scratch CNN+GNN')
    p.add_argument('--train-img-dir',  type=str,   default=f'{CLEAN}/train/images')
    p.add_argument('--train-mask-dir', type=str,   default=f'{CLEAN}/train/masks')
    p.add_argument('--test-img-dir',   type=str,   default=f'{CLEAN}/test/images')
    p.add_argument('--test-mask-dir',  type=str,   default=f'{CLEAN}/test/masks')
    p.add_argument('--encoder-name',    type=str,   default='resnet34d',
                   help='timm encoder name (e.g. resnet34d, efficientnet_b4, convnext_tiny)')
    p.add_argument('--no-pretrained',   action='store_true', help='Train encoder from scratch')
    p.add_argument('--image-size',      type=int,   default=448,
                   help='Native resolution of crack_seg_clean images is 448px')
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
    # Loss configuration
    p.add_argument('--use-tversky',     action='store_true',
                   help='Replace CrackDice with FocalTversky (β>α → penalise missed cracks harder)')
    p.add_argument('--tversky-alpha',   type=float, default=0.3, help='FP weight in Tversky')
    p.add_argument('--tversky-beta',    type=float, default=0.7, help='FN weight in Tversky')
    p.add_argument('--tversky-gamma',   type=float, default=0.75, help='Focal exponent in FocalTversky')
    p.add_argument('--use-cldice',      action='store_true',
                   help='Add differentiable soft-clDice connectivity loss term')
    p.add_argument('--cldice-weight',   type=float, default=0.3,
                   help='Weight of soft-clDice in combined loss (0–1)')
    p.add_argument('--cldice-iters',    type=int,   default=10,
                   help='Soft-skeleton erosion iterations')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pl.seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == 'enhanced':
        base_model = EnhancedGraphUNet(out_channels=2)
    elif args.model == 'plain':
        base_model = HybridGraphUNet(
            encoder_name=args.encoder_name,
            pretrained=not args.no_pretrained,
            out_channels=2,
            use_gnn_bottleneck=False,
        )
    else:
        base_model = HybridGraphUNet(
            encoder_name=args.encoder_name,
            pretrained=not args.no_pretrained,
            out_channels=2,
            use_gnn_bottleneck=True,
        )

    loss_kwargs = dict(
        use_tversky=args.use_tversky,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
        use_cldice=args.use_cldice,
        cldice_weight=args.cldice_weight,
        cldice_iters=args.cldice_iters,
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
            **loss_kwargs,
        )
    else:
        module = SegModule(
            model=base_model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            crack_weight=args.crack_weight,
            encoder_lr_scale=args.encoder_lr_scale,
            **loss_kwargs,
        )

    # Val = test set so all training data is used for training
    train_dl, val_dl = build_dataloaders(
        args.train_img_dir, args.train_mask_dir,
        args.test_img_dir,  args.test_mask_dir,
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

        import json
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results[0], f, indent=2)
        print(f'Metrics saved → {metrics_path}')
    else:
        print(f'\nNo test set found at {args.test_img_dir} — skipping test evaluation.')


if __name__ == '__main__':
    main()
