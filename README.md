# Crack Topology GNN — Phase 2

Road crack evolution analysis using a hybrid CNN-GNN segmentation model.

## Project Pipeline

```
Stage 1: Image Segmentation   ←  this repo
Stage 2: Image-to-Graph Conversion
Stage 3: GNN Link Prediction  (how cracks evolve and connect)
```

This repository implements **Stage 1** — generating binary crack masks from road surface images using an `EnhancedGraphUNet` model (hybrid CNN encoder/decoder with a GNN bottleneck).

---

## Model: EnhancedGraphUNet

A hybrid architecture combining a standard UNet with ViG (Vision GNN) graph convolution layers in the bottleneck.

```
Input (RGB image)
    │
    ▼
Encoder — 4x DoubleConv + MaxPool
    [3 → 32 → 64 → 128 → 256 channels]
    │
    ▼
Bottleneck (GNN)
    Grapher(k=9, dilation=1)   ← graph conv over spatial tokens
    FFN (×4 expansion)
    Grapher(k=9, dilation=2)   ← wider receptive field
    FFN (×4 expansion)
    Dropout2d(0.1)
    │
    ▼
Decoder — 4x ConvTranspose2d + skip connections + DoubleConv
    [256 → 128 → 64 → 32 channels]
    │
    ▼
Head — 1×1 Conv → 2 classes (background, crack)
```

**Key design choices:**
- Lighter feature widths `(32, 64, 128, 256)` reduce overfitting on small crack datasets
- Dual Grapher+FFN with different dilations captures multi-scale crack topology
- Edge convolution (`conv='edge'`) in the Grapher block
- Combined loss: `0.5 × CrossEntropy + 0.5 × SoftDice` handles class imbalance (cracks are thin/sparse)
- Optimizer: AdamW + CosineAnnealingLR

---

## Repository Structure

```
crack-topology-gnn/
├── segmentation/
│   ├── graph_layers/          ← ViG graph building blocks (self-contained)
│   │   ├── __init__.py
│   │   ├── torch_nn.py        ← BasicConv, act_layer, batched_index_select
│   │   ├── torch_edge.py      ← DenseDilatedKnnGraph (KNN graph construction)
│   │   ├── pos_embed.py       ← 2D sinusoidal relative positional embeddings
│   │   ├── torch_vertex.py    ← Grapher (dynamic graph conv block)
│   │   └── vig.py             ← FFN (feed-forward block with DropPath)
│   ├── model.py               ← EnhancedGraphUNet architecture
│   ├── dataset.py             ← CrackDataset loader + augmentation pipeline
│   ├── train.py               ← training script (PyTorch Lightning)
│   ├── test.py                ← evaluation from checkpoint
│   └── generate_masks.py      ← inference → binary PNG masks (feeds Stage 2)
├── outputs/
│   └── segmentation/
│       └── checkpoints/       ← best model checkpoints saved here
└── requirements.txt
```

---

## Dataset

**CrackDataset_DL_HY** — binary road crack segmentation dataset.

```
CrackDataset_DL_HY/split/
├── train/
│   ├── images/    ← 374 RGB images (.jpg)
│   └── masks/     ← 374 binary masks (.png)  white=crack, black=background
└── val/
    ├── images/    ← 94 RGB images (.jpg)
    └── masks/     ← 94 binary masks (.png)
```

Masks are stored as 1-bit PNG files (PIL mode `1`). The dataset loader converts them to binary `[0, 1]` integer tensors automatically.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `torch`, `torchvision`, `pytorch-lightning`, `albumentations`, `opencv-python`, `timm`, `wandb`, `tqdm`, `Pillow`, `numpy`

---

## Usage

All commands should be run from inside the `segmentation/` directory:

```bash
cd segmentation
```

### Train

```bash
python train.py
```

This uses the default dataset paths. All arguments are optional:

```bash
python train.py \
    --train-img-dir  /path/to/train/images \
    --train-mask-dir /path/to/train/masks \
    --val-img-dir    /path/to/val/images \
    --val-mask-dir   /path/to/val/masks \
    --image-size     256 \
    --batch-size     8 \
    --max-epochs     100 \
    --learning-rate  3e-4 \
    --no-wandb
```

| Argument | Default | Description |
|---|---|---|
| `--image-size` | `256` | Resize input images to this square size |
| `--batch-size` | `8` | Training batch size |
| `--max-epochs` | `100` | Max training epochs (early stop at patience=20) |
| `--learning-rate` | `3e-4` | AdamW learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--precision` | `16` | Mixed precision (`16` or `32`) |
| `--output-dir` | `../outputs/segmentation` | Where to save checkpoints |
| `--experiment-name` | `crack_graphunet` | WandB run name prefix |
| `--checkpoint-path` | `None` | Resume from a checkpoint |
| `--no-wandb` | off | Disable WandB logging |

Checkpoints are saved to `outputs/segmentation/checkpoints/`. Top-3 models by `val/iou` are kept.

### Evaluate

```bash
python test.py --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt
```

Prints loss, IoU, Dice, and accuracy on the validation split.

### Generate Masks (Stage 2 input)

```bash
python generate_masks.py --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt
```

Writes binary PNG masks to `outputs/masks/`. These are the input for Stage 2 (image-to-graph conversion).

```bash
python generate_masks.py \
    --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt \
    --input-dir  /path/to/images \
    --output-dir /path/to/save/masks \
    --image-size 256
```

---

## Training Metrics

| Metric | Description |
|---|---|
| `val/iou` | Mean IoU across background and crack classes (primary metric) |
| `val/dice` | Mean Dice score |
| `val/acc` | Pixel accuracy |
| `val/loss` | Combined CE + Dice loss |

---

## Credits

Graph layer code (`graph_layers/`) adapted from the ViG (Vision GNN) architecture:
> Han et al., *"Vision GNN: An Image is Worth Graph of Nodes"*, NeurIPS 2022.

`EnhancedGraphUNet` architecture and training pipeline adapted from Aryan's TokenCutSeg experiments (Phase 1), with modifications:
- Lighter feature widths for crack datasets
- Dual dilated Grapher+FFN bottleneck for multi-scale topology capture
- Adapted dataset loader for CrackDataset_DL_HY
