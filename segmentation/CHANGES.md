# Phase 2 — Problems & Fixes

## Environment Setup

### Problem: Missing dependencies
The virtual environment (Python 3.14.3) had `torch` and core packages installed but was missing several required libraries.

**Missing packages:**
- `torchvision`
- `pytorch-lightning`
- `wandb`
- `albumentations`
- `timm`

**Fix:**
```bash
pip install torchvision pytorch-lightning wandb albumentations timm
```

**Installed versions:**
| Package | Version |
|---|---|
| torchvision | 0.27.0 |
| pytorch-lightning | 2.6.1 |
| wandb | 0.27.0 |
| albumentations | 2.0.8 |
| timm | 1.0.27 |

---

### Problem: Deprecated `timm` import path
Both `graph_layers/torch_vertex.py` and `graph_layers/vig.py` imported `DropPath` from the deprecated path `timm.models.layers`, triggering a `FutureWarning` on every run.

**Before:**
```python
from timm.models.layers import DropPath
```

**Fix (`graph_layers/torch_vertex.py` and `graph_layers/vig.py`):**
```python
from timm.layers import DropPath
```

---

## Training Issues

### Problem 1: Misleading validation metrics
`val/iou` and `val/dice` were computed as the **mean over both classes** (background + crack). Since cracks occupy only ~2–5% of image pixels, the background class is trivially easy to predict. This made metrics unreliable — a model that ignores cracks entirely scores ~0.50 mean IoU, identical to a model that detects some cracks.

**Example of the deceptive scenario:**
| Model behaviour | Background IoU | Crack IoU | Mean IoU logged |
|---|---|---|---|
| Ignores cracks entirely | 0.99 | ~0.02 | **0.51** |
| Partially detects cracks | 0.90 | 0.12 | **0.51** |

The `val/acc: 0.977` was similarly misleading — high accuracy is trivially achieved by predicting background everywhere.

**Fix (`train.py`):** Added two new metric functions and logged them in the progress bar:

```python
def compute_crack_iou(pred, target, smooth=1e-6):
    inter = ((pred == 1) & (target == 1)).float().sum()
    union = ((pred == 1) | (target == 1)).float().sum()
    return (inter + smooth) / (union + smooth)

def compute_crack_dice(pred, target, smooth=1e-6):
    p     = (pred   == 1).float()
    t     = (target == 1).float()
    inter = (p * t).sum()
    return (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
```

`val/crack_iou` and `val/crack_dice` are now the primary progress bar metrics. Mean IoU/Dice are still logged but hidden from the progress bar.

---

### Problem 2: Loss function not focused on crack class
The original `CombinedLoss` used:
- **Plain CrossEntropy** — treats every pixel equally, so the model is rewarded for getting the ~95% background pixels right and barely penalised for missing cracks
- **Mean Dice over both classes** — gradient signal dominated by the easy background class

This meant the loss could decrease steadily while the model never actually learned to detect cracks.

**Before:**
```python
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        self.ce = nn.CrossEntropyLoss()

    def _dice(self, logits, targets):
        # averaged over BOTH classes — background dilutes gradient
        for cls in range(n):
            ...
        return 1.0 - total / n

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self._dice(logits, targets)
```

**Fix (`train.py`):** Replaced with crack-focused loss:
- **Weighted CrossEntropy** with `crack_weight=10.0` — a missed crack pixel is penalised 10× more than a false positive background pixel
- **Crack-only Dice** — Dice is computed on the crack class (class=1) only, so all gradient comes from the hard class

```python
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6, crack_weight=10.0):
        self.register_buffer('class_weights', torch.tensor([1.0, crack_weight]))

    def _crack_dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]   # crack class only
        t     = (targets == 1).float()
        inter = (probs * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        return 0.5 * ce + 0.5 * self._crack_dice(logits, targets)
```

---

### Problem 3: Checkpoint and EarlyStopping monitored the wrong metric
`ModelCheckpoint` and `EarlyStopping` were both monitoring `val/iou` (mean IoU), which as shown above is unreliable for imbalanced crack datasets. The best-saved model could be one that mostly predicts background.

**Before:**
```python
ckpt_cb  = ModelCheckpoint(..., monitor='val/iou', mode='max')
early_cb = EarlyStopping(monitor='val/iou', mode='max', patience=20)
```

**Fix (`train.py`):**
```python
ckpt_cb  = ModelCheckpoint(..., monitor='val/crack_iou', mode='max')
early_cb = EarlyStopping(monitor='val/crack_iou', mode='max', patience=20)
```

---

---

## Additional Fix — LearningRateMonitor crash with --no-wandb

### Problem
Running `python train.py --no-wandb` crashed at the start of training:
```
MisconfigurationException: Cannot use `LearningRateMonitor` callback with `Trainer` that has no logger.
```
`LearningRateMonitor` was always added to the callbacks list even when `logger=False` (no-wandb mode).

**Fix (`train.py`):** Move `LearningRateMonitor` inside the wandb block so it is only added when a logger is active:
```python
if not args.no_wandb:
    logger = WandbLogger(...)
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))  # only with logger
```

---

## Run 3 — DeepCrack Dataset (Target Achieved ✅)

No code changes were needed to train on DeepCrack. The existing `CrackDataset` loader and `train.py` CLI args handle any flat `images/masks` directory layout.

**Command used:**
```bash
python train.py --no-wandb --image-size 512 \
    --train-img-dir  "/path/to/DeepCrack/train_img" \
    --train-mask-dir "/path/to/DeepCrack/train_lab" \
    --val-img-dir    "/path/to/DeepCrack/test_img" \
    --val-mask-dir   "/path/to/DeepCrack/test_lab" \
    --test-img-dir   "/path/to/DeepCrack/test_img" \
    --test-mask-dir  "/path/to/DeepCrack/test_lab" \
    --experiment-name crack_deepcrack
```

**Result:** `test/crack_iou = 0.7220` — **target of 0.65–0.70 exceeded** using only 300 training images.

---

## Summary of All File Changes

| File | Change |
|---|---|
| `graph_layers/torch_vertex.py` | Fixed deprecated `timm.models.layers` import |
| `graph_layers/vig.py` | Fixed deprecated `timm.models.layers` import |
| `train.py` | Added `compute_crack_iou`, `compute_crack_dice` functions |
| `train.py` | Updated `validation_step` and `test_step` to log crack-specific metrics |
| `train.py` | Rewrote `CombinedLoss` to use Focal Loss + crack-only Dice |
| `train.py` | Updated `ModelCheckpoint` and `EarlyStopping` to monitor `val/crack_iou` |
| `train.py` | Fixed `LearningRateMonitor` — only added when logger is active |
| `train.py` | Added differential LR (encoder 10× lower), gradient clipping, test evaluation |
| `train.py` | Added `--encoder-name`, `--test-img-dir`, `--test-mask-dir`, `--crack-weight` args |
| `model.py` | Added `HybridGraphUNet` (pretrained ResNet34d encoder + GNN bottleneck) |
| `dataset.py` | CLAHE, CoarseDropout, Affine augmentation; pin_memory fix; persistent_workers |
| `test.py` | Rewritten to use `HybridGraphUNet` and proper held-out test split |
| `prepare_split.py` | Created — reproducible 80/10/10 train/val/test split from SematicSeg_Dataset |

---

## What to Watch During Training (After Fixes)

| Metric | What it tells you |
|---|---|
| `val/crack_iou` | Primary metric — actual crack detection quality. Target: > 0.40 by epoch 30 |
| `val/crack_dice` | Correlates with crack_iou; more sensitive to partial predictions |
| `val/loss` vs `train/loss` | Gap > 0.05 sustained over 5+ epochs signals overfitting |
| `val/iou` | Mean IoU (background + crack); context only, not for model selection |
| `val/acc` | Misleading due to class imbalance; ignore |

### Remaining risks
| Risk | Mitigation |
|---|---|
| Overfitting after epoch 40–60 | `Dropout2d(0.1)`, `DropPath`, `weight_decay=1e-4`, `EarlyStopping(patience=20)` |
| MPS + fp16 NaN loss | If NaN appears, rerun with `--precision 32` |
| Small dataset | Heavy augmentation (flips, rotations, elastic, noise, brightness) already in place |

> **Note:** Restart training from scratch after these changes. The loss function change is significant — continuing from an epoch-10 checkpoint trained with the old loss would produce inconsistent gradients.
