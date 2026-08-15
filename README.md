# From Pixels to Topology: A Three-Stage Pipeline for Road Crack Graph Analysis Using Graph Neural Networks

**Authors:** Tejas Shivakumar, Aishwarya Shekar — DCU Practicum 2025–2026

End-to-end pipeline for road crack topology analysis — from raw images to graph-structured data to GNN-based topology understanding.

## Pipeline

```
Raw road surface images
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1 — Segmentation         │  ← segmentation/
│  HybridGraphUNet                │
│  Input: RGB image               │
│  Output: binary crack mask (PNG)│
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Stage 2 — Image-to-Graph       │  ← image_to_graph/
│  Skeleton → PyG graph           │
│  Input: binary mask             │
│  Output: PyG Data (.pt)         │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Stage 3 — GNN Link Prediction  │  ← link_prediction/
│  Predict crack topology         │
│  Input: PyG graphs              │
│  Output: missing nodes & edges  │
└─────────────────────────────────┘
```

**All 3 stages are complete.**

---

## Repository Structure

```
crack-topology-gnn/
│
├── segmentation/                    ← Stage 1
│   ├── graph_layers/                ← ViG graph building blocks (self-contained)
│   │   ├── torch_nn.py              ← BasicConv, activation layers
│   │   ├── torch_edge.py            ← DenseDilatedKnnGraph (KNN graph construction)
│   │   ├── pos_embed.py             ← 2D sinusoidal positional embeddings
│   │   ├── torch_vertex.py          ← Grapher (dynamic graph conv block)
│   │   └── vig.py                   ← FFN (feed-forward block with DropPath)
│   ├── model.py                     ← EnhancedGraphUNet + HybridGraphUNet architectures
│   ├── dataset.py                   ← CrackDataset loader + augmentation pipeline
│   ├── train.py                     ← PyTorch Lightning training script
│   ├── test.py                      ← evaluation from checkpoint
│   ├── generate_masks.py            ← inference → binary PNG masks (Stage 2 input)
│   └── prepare_split.py             ← dataset split utility
│
├── image_to_graph/                  ← Stage 2
│   ├── features.py                  ← distance transform, node/edge feature extraction
│   ├── convert.py                   ← core mask_to_graph() function
│   ├── visualize.py                 ← 4-panel per-image visualisation
│   ├── build_dataset.py             ← CLI batch processor
│   ├── read_graphs.py               ← inspection utility for saved .pt files
│   ├── IMPLEMENTATION.md            ← design decisions and approach
│   └── RESULTS.md                   ← dataset statistics and feature analysis
│
├── link_prediction/                 ← Stage 3
│   ├── model.py                     ← CrackGATEncoder, MLPEdgePredictor, MLPNodePredictor
│   ├── masking.py                   ← edge masking and node masking utilities
│   ├── train.py                     ← joint training loop (node task + edge task)
│   ├── evaluate.py                  ← AUC-ROC, Average Precision, Hits@K, F1
│   ├── visualize.py                 ← per-graph prediction visualisation + training curves
│   ├── run.py                       ← CLI entry point (train + eval, or --eval-only)
│   └── IMPLEMENTATION.md            ← architecture, training design, results
│
└── outputs/                         ← gitignored
    ├── segmentation/checkpoints/    ← Stage 1 model checkpoints
    ├── graphs/graphs/               ← train_graphs.pt, test_graphs.pt
    ├── graphs/visualizations/       ← per-image graph overlay PNGs
    └── link_prediction/             ← best_model.pt, metrics.json, training_curves.png
```

---

## Stage 1 — Segmentation

### Model: HybridGraphUNet

A hybrid CNN-GNN architecture. Pretrained ResNet34d encoder with skip connections, a ViG (Vision GNN) bottleneck, and a UNet decoder.

```
Input (RGB image)
    │
Encoder — ResNet34d (ImageNet pretrained)    [64 → 64 → 128 → 256 → 512 channels]
    │
Bottleneck (GNN)
    Grapher(k=9, dilation=1) → FFN      ← local crack topology
    Grapher(k=9, dilation=2) → FFN      ← wider receptive field
    Dropout2d(0.1)
    │
Decoder — 4× ConvTranspose2d + skip + DoubleConv + final upsample
    │
Head — 1×1 Conv → 2 classes (background, crack)
```

**Also available:** `EnhancedGraphUNet` — a lighter scratch-trained variant with feature widths `(32, 64, 128, 256)` (no pretrained encoder).

**Key design choices:**
- Pretrained ResNet34d encoder — provides edge, texture, and curve detectors without requiring a large dataset
- Dual dilated Grapher+FFN — captures crack topology at two spatial scales
- Loss: `0.5 × FocalLoss(γ=2, crack_weight=10) + 0.5 × CrackDice` — handles class imbalance
- Differential learning rate: encoder 1e-5, decoder 1e-4
- Optimizer: AdamW + CosineAnnealingLR

### Results (DeepCrack)

| Metric | Value |
|---|---|
| test/crack_iou | **0.722** ✅ |
| test/crack_dice | 0.834 |
| test/iou (mean) | 0.854 |
| test/acc | 0.987 |

### Stage 1 Usage

Run from inside the `segmentation/` directory:

```bash
cd segmentation
```

**Train:**
```bash
python train.py \
    --train-img-dir  /path/to/DeepCrack/train_img \
    --train-mask-dir /path/to/DeepCrack/train_lab \
    --val-img-dir    /path/to/DeepCrack/test_img \
    --val-mask-dir   /path/to/DeepCrack/test_lab \
    --image-size 512 --batch-size 8 --max-epochs 150 --no-wandb
```

**Evaluate:**
```bash
python test.py --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt
```

**Generate masks for Stage 2:**
```bash
python generate_masks.py \
    --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt \
    --input-dir  /path/to/images \
    --output-dir /path/to/save/masks
```

---

## Stage 2 — Image-to-Graph Conversion

Converts binary crack masks into graph-structured data for GNN link prediction. No model is trained here — this is a deterministic preprocessing step.

### Approach

```
Binary mask
    │
    ▼
Skeletonise (skimage)        ← collapses crack region to 1-px centreline
    │
    ▼
Build NetworkX graph (sknw)  ← junctions and endpoints become nodes,
    │                            crack segments become edges with pixel paths
    ▼
Distance transform (OpenCV)  ← measures true crack width at every pixel
    │
    ▼
Extract node/edge features   ← 6 node features, 7 edge features
    │
    ▼
PyG Data object              ← saved to .pt for Stage 3
```

### Graph Features

**Node features `x [N, 6]`:**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x_norm` | x coordinate / image width |
| 1 | `y_norm` | y coordinate / image height |
| 2 | `thickness` | crack width at node (px, from distance transform) |
| 3 | `degree` | number of connected edges |
| 4 | `is_endpoint` | 1.0 if degree == 1 (crack tip) |
| 5 | `is_junction` | 1.0 if degree ≥ 3 (branching point) |

**Edge features `edge_attr [E, 7]`:**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `path_length` | geometric length of skeleton path (px) |
| 1 | `euclidean_dist` | straight-line distance between nodes (px) |
| 2 | `tortuosity` | path_length / euclidean_dist — ≥ 1.0 |
| 3 | `angle_sym` | crack orientation in [0°, 180°) |
| 4 | `avg_thickness` | mean crack width along segment (px) |
| 5 | `min_thickness` | thinnest point along segment (px) |
| 6 | `max_thickness` | widest point along segment (px) |

### Dataset Statistics (DeepCrack)

| | Train | Test |
|--|------:|-----:|
| Graphs | 300 | 237 |
| Nodes mean / max | 24.7 / 205 | 34.8 / 180 |
| Edges mean / max | 22.2 / 184 | 31.4 / 183 |
| Crack density mean | 2.9% | 4.3% |
| Endpoint nodes | 58.2% | 58.8% |
| Junction nodes | 35.5% | 36.1% |

### Stage 2 Usage

Run from the project root:

```bash
cd crack-topology-gnn
```

**Process DeepCrack (train + test in one command):**
```bash
python3 image_to_graph/build_dataset.py \
    --deepcrack-root "/path/to/DeepCrack" \
    --output-dir outputs/graphs \
    --vis
```

**Inspect saved graphs:**
```bash
python3 image_to_graph/read_graphs.py \
    --pt outputs/graphs/graphs/train_graphs.pt --n 3
```

**Load graphs in code:**
```python
import torch
dataset = torch.load('outputs/graphs/graphs/train_graphs.pt', weights_only=False)
graph = dataset[0]
# graph.x           — node features  [N, 6]
# graph.edge_index  — connectivity   [2, 2E]
# graph.edge_attr   — edge features  [E, 7]
# graph.filename, graph.split, graph.crack_density, ...
```

---

## Stage 3 — GNN Link & Node Prediction

Proves the model understands crack topology through two jointly-trained tasks:

1. **Node prediction** — identify crack tips whose neighbours have been removed (primary task)
2. **Edge prediction** — reconstruct hidden crack segments between nodes (secondary task)

The conceptual framing: we take a present-state crack graph, remove parts of it to simulate a past state, and train the model to reconstruct what was removed. A model that can "see through the mud" has internalised the structural grammar of crack networks.

### Model Architecture

```
CrackGATEncoder (3-layer GAT):
  Layer 1: GATConv(6 → 128, heads=4, concat) → [N, 512]  + LayerNorm + ELU + Dropout
  Layer 2: GATConv(512 → 128, heads=4, concat) → [N, 512] + LayerNorm + ELU + Dropout
  Layer 3: GATConv(512 → 64, heads=1) + Residual projection → [N, 64] + LayerNorm
  Edge features: 7-dim → 8-dim (angle encoded as [sin(2θ), cos(2θ)])

MLPEdgePredictor:  [z_u ‖ z_v ‖ z_u ⊙ z_v] → 192 → 128 → 64 → 1
MLPNodePredictor:  z → 64 → 32 → 1
```

### Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| LR schedule | Linear warmup (10 ep) + cosine annealing |
| Epochs | 300 |
| Gradient accumulation | 8 graphs/step |
| Validation metric | 0.5 × node_auc + 0.5 × edge_auc |
| Edge loss weight | 1.0 (equal weight both tasks) |

### Results (DeepCrack test set, 237 graphs)

| Task | Metric | Value |
|---|---|---|
| Node prediction | AUC-ROC | **0.8321** |
| Node prediction | Avg Precision | 0.4230 |
| Node prediction | F1 Score | 0.2518 |
| Edge prediction | AUC-ROC | **0.7247** |
| Edge prediction | Avg Precision | 0.6863 |
| Edge prediction | Hits@10 | 0.8500 |
| Edge prediction | Hits@20 | **0.9622** |

Best checkpoint: epoch 260 / 300 — `outputs/link_prediction/best_model.pt`

### Stage 3 Usage

Run from the project root:

```bash
cd crack-topology-gnn
```

**Train + evaluate (all defaults tuned):**
```bash
python3 link_prediction/run.py \
    --train-graphs outputs/graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/graphs/graphs/test_graphs.pt \
    --output-dir   outputs/link_prediction
```

**Evaluate from saved checkpoint (no retraining):**
```bash
python3 link_prediction/run.py \
    --train-graphs outputs/graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/graphs/graphs/test_graphs.pt \
    --output-dir   outputs/link_prediction \
    --eval-only
```

**Key CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 300 | Training epochs |
| `--hidden` | 128 | GAT hidden dimension per head |
| `--out-dim` | 64 | Node embedding output dimension |
| `--heads` | 4 | Number of attention heads |
| `--dropout` | 0.3 | Dropout rate |
| `--lr` | 5e-4 | AdamW learning rate |
| `--edge-loss-weight` | 1.0 | Weight λ for edge task loss |
| `--node-mask-frac` | 0.20 | Fraction of endpoints hidden per graph |
| `--accum-steps` | 8 | Gradient accumulation steps |
| `--warmup-epochs` | 10 | Linear LR warmup before cosine |
| `--eval-only` | off | Load checkpoint and skip training |
| `--vis-n` | 10 | Number of test graphs to visualise |

---

## Setup

```bash
pip install torch torchvision pytorch-lightning
pip install torch-geometric
pip install albumentations opencv-python timm wandb tqdm Pillow numpy
pip install scikit-image scikit-learn sknw matplotlib scipy networkx
```

---

## Datasets

| Dataset | Used in | Notes |
|---------|---------|-------|
| DeepCrack | Stage 1 (training) + Stage 2 | Primary benchmark dataset |
| CrackDataset_DL_HY | Stage 1 experiments | Multi-source curated dataset |

Dataset files are not committed to this repository. Place datasets in a local directory and update paths in the relevant scripts accordingly.

---

## Credits

Graph layer code (`segmentation/graph_layers/`) adapted from the ViG (Vision GNN) architecture:
> Han et al., *"Vision GNN: An Image is Worth Graph of Nodes"*, NeurIPS 2022.

Phase 1 baseline (segmentation experiments, ImageToGraph reference implementation):
> Aryan's TokenCutSeg repository.
