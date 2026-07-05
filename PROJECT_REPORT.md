# Crack Detection & Topology Analysis — Project Report

## Overview

This project explores crack detection in infrastructure images using a hybrid of image segmentation and graph neural networks. The goal is to move beyond pixel-level crack detection towards understanding the structural topology of cracks — their branching patterns, continuity, and connectivity — which matters far more for structural health monitoring than knowing which pixels are cracked.

The work spans two phases: Phase 1 focused on segmentation, Phase 2 on graph construction and GNN-based crack topology reasoning.

---

## Phase 1: Image Segmentation

### Starting Point — Aryan's Repository

We began with an existing codebase that implemented two segmentation models:

**Standard UNet** (`models/unet.py`)
- Classic encoder–decoder architecture
- 4 downsampling blocks: 64 → 128 → 256 → 512 channels
- MaxPool2d for downsampling, ConvTranspose2d for upsampling
- Skip connections at each level
- Standard baseline for binary segmentation

**Hybrid GraphUNet** (`models/hybrid_unet.py`)
- Same encoder–decoder structure as UNet
- Added a `graph_conv` block at the bottleneck: `Conv2d(1024, 1024, 3) → BatchNorm2d → ReLU`
- Intent: the bottleneck operates on a spatially compressed feature map (like a graph of local regions), capturing long-range spatial relationships before decoding
- This is a lightweight approximation of GNN reasoning applied to image features

**Training Configuration**
- Dataset: DeepCrack (`train_img/`, `train_lab/`, `test_img/`, `test_lab/`)
- Image size: 512 × 512
- Epochs: 100, Batch size: 8, LR: 1e-3 with cosine annealing and 5-epoch warmup
- Loss: combined BCE + Dice (50/50 weight)
- Augmentations: HorizontalFlip, VerticalFlip, RandomRotate90
- Optimizer: AdamW with weight decay 1e-4
- Device: Apple Silicon MPS

**Results on DeepCrack**

| Model | Val IoU | Test IoU |
|---|---|---|
| Standard UNet | ~0.65 | ~0.65 |
| Hybrid GraphUNet | ~0.67–0.68 | ~0.67–0.68 |

The hybrid bottleneck gave a small but consistent improvement over plain UNet, confirming that spatial relational reasoning at the bottleneck helps even with a simple Conv2d proxy for graph operations.

### Image-to-Graph Conversion (Phase 1)

Alongside segmentation, we built an early image-to-graph pipeline converting crack masks into graphs. This version used:
- Skeletonization of the binary mask
- `sknw` library to extract a NetworkX graph from the skeleton
- Basic node and edge attributes (position, degree)
- OBIA (Object-Based Image Analysis) for region-level graph construction

This was a proof-of-concept — the feature set was minimal and the graphs were not yet used for any downstream learning task.

---

## Dataset Problem — Why DeepCrack Was Not Enough

DeepCrack provided ~500 training images. For a practicum-scale project aiming to train GNNs on crack graphs, 500 images translates to roughly 500 graphs. That is too few for:
- Learning generalizable GNN representations
- Covering diverse crack types (linear, branching, network, alligator, etc.)
- Avoiding overfitting in both segmentation and graph learning

This forced a dataset search.

---

## Dataset Search & Dead Ends

### PaveDistress Dataset — Dropped

We investigated the PaveDistress dataset (~8,000 files). After analysis:
- Only **846 files** were crack-related out of ~8,000 total
- Significant duplicates further reduced usable data
- **No mask data** — only bounding box labels
- Without masks, we could not perform pixel-level segmentation or skeleton-based graph extraction

PaveDistress was dropped entirely.

### Side Project: Automated Mask Generation

With PaveDistress having no masks, we attempted to generate masks programmatically:

**Approach**
1. Used the trained DeepCrack GraphUNet model to predict segmentation masks on PaveDistress images
2. Post-processed predictions with OpenCV (thresholding, morphological operations) to clean up the masks
3. Built a **web interface** to let us review and manually correct generated masks — displaying the crack image alongside its predicted mask with editing tools

**Why it failed**
- The model trained on DeepCrack did not generalize well to PaveDistress images (different image conditions, pavement types, lighting)
- Predicted masks were noisy and required too much manual correction per image
- The volume of images made full manual correction infeasible
- Quality of generated masks was not sufficient for training reliable downstream models

This side project was abandoned, but it produced useful infrastructure (the web mask editor) and sharpened our understanding of domain shift in crack datasets.

---

## Dataset Resolution — crack_segmentation_dataset

We found and adopted the `crack_segmentation_dataset`, a consolidated collection aggregating **11 public crack datasets** with paired images and binary masks.

### Cleaning Pipeline

**Raw inventory:** 11,298 images across 11 sources

**Stage 1 — Scope filter**
- Retained only crack images with corresponding masks
- Removed non-crack images and images without mask pairs

**Stage 2 — Deduplication**
- Detected duplicate images across sources (same image appearing in multiple datasets under different names)
- Dropped duplicates, keeping one representative

**Final cleaned dataset:** ~4,769 images

**Output structure:**
```
crack_seg_clean/clean/
    train/
        images/   (4,071 images)
        masks/    (4,071 masks)
    test/
        images/   (698 images)
        masks/    (698 masks)
```

This 8× increase over DeepCrack gave us a viable base for both segmentation retraining and graph construction.

---

## Phase 2: Graph Construction & GNN Link Prediction

### Image-to-Graph Conversion Pipeline (Rebuilt)

The Phase 2 conversion pipeline (`image_to_graph/`) is a full rewrite of the Phase 1 proof-of-concept with rich, carefully engineered features.

**Pipeline steps:**
1. Load binary crack mask
2. Skeletonize with `skimage.morphology.skeletonize`
3. Build NetworkX graph with `sknw.build_sknw`
4. **Spur pruning** — iteratively remove leaf branches shorter than `prune_ratio × longest_branch`, eliminating skeleton noise from rough mask boundaries
5. Extract node and edge features
6. Package as a PyTorch Geometric `Data` object

**Node features [N, 6]:**

| Index | Feature | Description |
|---|---|---|
| 0 | `x_norm` | Normalized x coordinate (0–1) |
| 1 | `y_norm` | Normalized y coordinate (0–1) |
| 2 | `thickness` | Crack width at node via distance transform |
| 3 | `degree` | Number of connected edges |
| 4 | `is_endpoint` | 1.0 if degree == 1 (crack tip) |
| 5 | `is_junction` | 1.0 if degree ≥ 3 (branching point) |

**Edge features [E, 7]:**

| Index | Feature | Description |
|---|---|---|
| 0 | `path_length` | Skeleton pixel path length along segment |
| 1 | `euclidean_dist` | Straight-line distance between endpoints |
| 2 | `tortuosity` | path_length / euclidean_dist (1.0 = straight) |
| 3 | `angle_sym` | Crack orientation [0°, 180°), symmetric |
| 4 | `avg_thickness` | Mean crack width along segment |
| 5 | `min_thickness` | Thinnest point along segment |
| 6 | `max_thickness` | Widest point along segment |

**Design decisions and why:**
- **Distance transform for thickness**: More robust than pixel counting; gives a continuous width estimate at every skeleton point
- **Tortuosity**: Captures how curved a crack segment is — straight cracks vs. meandering ones carry different structural significance
- **Spur pruning**: Masks are never perfectly clean; skeletonization of rough boundaries creates short leaf branches that are noise, not real crack endpoints. Removing them makes the graph topology meaningful
- **Symmetric angle**: Since edges are undirected, `arctan2 % 180` gives the same angle for (u→v) and (v→u)

### Angle Encoding Fix

The raw `angle_sym` feature ranges [0°, 180°) with a discontinuity at 0°/180° — two edges at nearly the same orientation could have features of ~1° and ~179°, which look very different numerically but are geometrically almost identical.

**Fix:** Replace the raw angle with circular encoding `[sin(2θ), cos(2θ)]`. This maps 0° and 180° to the same point, smoothly encoding the periodicity of undirected edge orientation. This expands edge features from 7 → 8 dimensions without changing the saved graph files.

### GNN Models — Link Prediction

The task is **link prediction**: given a crack graph with some edges removed (simulating broken/occluded crack connections), can the model predict which nodes should be reconnected?

Two prediction heads:
- **MLPEdgePredictor**: Scores candidate edges using `[z_u, z_v, z_u ⊙ z_v]` — the Hadamard product captures pairwise interaction
- **MLPNodePredictor**: Classifies nodes as topologically incomplete (had a hidden neighbour removed)

Five encoder architectures were compared:

| Encoder | Description | Edge Features |
|---|---|---|
| **MLP** | No-graph baseline; pure MLP on node features | ✗ |
| **GCN** | 3-layer Graph Convolutional Network | ✗ |
| **GraphSAGE** | 3-layer inductive mean aggregation | ✗ |
| **GINE** | 3-layer GIN extended with edge features | ✓ |
| **GAT** | 3-layer Graph Attention Network, multi-head | ✓ |

All GNN encoders use:
- 3 layers with LayerNorm
- Residual connection between layer 2 and layer 3
- Dropout for regularisation

**Quick comparison results (500 graphs, 50 epochs):**

| Encoder | Node AP | Edge AP |
|---|---|---|
| MLP (baseline) | — | **0.862** |
| GCN | — | — |
| GraphSAGE | — | — |
| GINE | **0.945** | — |
| GAT | — | — |

GINE achieving 0.945 Node AP confirms that edge features (especially tortuosity and thickness) are critical for the node-level task. MLP leading on Edge AP is a known phenomenon in link prediction — the structural bias of some GNNs can hurt when the task is purely relational; the MLP baseline is a strong lower bound.

---

## Phase 2: Segmentation Model Evolution

In parallel with GNN work, the segmentation model was also upgraded in Phase 2 to reflect what we learned from Phase 1.

### EnhancedGraphUNet (Scratch-Trained)

Built in `crack-topology-gnn/segmentation/model.py`. Replaces the simple Conv2d bottleneck from Phase 1 with a real GNN:

- Encoder: 4 DoubleConv blocks (32 → 64 → 128 → 256 channels), scratch-trained
- Bottleneck: Two `Grapher` (ViG-style) blocks with dilation 1 and 2, plus FFN layers
- The `Grapher` layer dynamically constructs a k-nearest-neighbour graph in feature space at the bottleneck, performing true graph convolution on spatial feature maps
- Decoder: standard UNet upsampling with skip connections

**Problem:** With only 374 training samples (after train/val split from the available labelled data at that time), the scratch-trained encoder underfits badly. The encoder had not seen enough crack images to learn useful low/mid-level features.

### HybridGraphUNet (Pretrained Encoder)

The fix was to replace the scratch encoder with a pretrained CNN:

- Encoder: `timm` `resnet34d` pretrained on ImageNet — returns 5 multi-scale feature maps at strides 2, 4, 8, 16, 32
- Bottleneck: Same two-stage `Grapher` + `FFN` blocks (unchanged from EnhancedGraphUNet)
- Decoder: Built dynamically from encoder channel dimensions with skip connections; final upsample back to input resolution

**Why this works better:**
- ImageNet-pretrained features already encode edges, textures, curves, and gradients — exactly what a crack detector needs as low-level features
- The GNN bottleneck still captures crack topology in the compressed representation
- The pretrained encoder removes the need for thousands of labelled crack images to learn basic image features from scratch
- Training becomes: fine-tune the encoder slowly (low LR) while training the GNN bottleneck and decoder faster

---

## Current Work — Phase 1 Model on Larger Dataset

Having the clean 4,769-image dataset, we are now retraining the original Phase 1 `GraphUNet` model on it. The motivation is to establish a clean baseline:

- Does the simple Conv2d bottleneck model, which got 67–68% IoU on DeepCrack, scale to a larger and more diverse dataset?
- This gives a fair comparison point before evaluating the more advanced `HybridGraphUNet` (pretrained encoder + real Grapher GNN bottleneck)

**Training config:**
- Model: `GraphUNet` (Phase 1 architecture, unchanged)
- Dataset: `crack_seg_clean` — 3,461 train / 610 val / 698 test
- Image size: 512 × 512
- Epochs: 10 (quick comparison run)
- Batch size: 8, LR: 1e-3, cosine annealing, 5-epoch warmup
- Loss: combined BCE + Dice
- Device: Apple Silicon MPS

---

## Summary of Decisions

| Decision | Why |
|---|---|
| Dropped PaveDistress | Only 846 crack files, no masks, low quality |
| Built mask generator + web editor | Attempted to solve the no-mask problem via automation; domain shift made it unreliable |
| Switched to crack_segmentation_dataset | 11 sources, paired masks, 8× larger than DeepCrack |
| Rebuilt image-to-graph pipeline | Phase 1 graph had minimal features; needed rich geometry for GNN learning |
| Added spur pruning | Skeleton noise from mask boundaries created fake crack tips |
| Circular angle encoding | Raw angle has 0°/180° discontinuity; sin/cos encoding is smooth |
| Added tortuosity as edge feature | Distinguishes straight cracks (structural) from winding ones (fatigue) |
| Used pretrained ResNet34d encoder | With few training images, scratch encoder underfits; ImageNet features transfer well to edge/texture detection |
| Compared 5 GNN encoders | Needed to isolate contributions of graph structure vs. edge features vs. attention |

---

## File Map

```
PHASE 1/
  Graph_construction/
    models/unet.py                     Standard UNet
    models/hybrid_unet.py              GraphUNet (Conv2d bottleneck)
    datasets/crack_dataset.py          DeepCrack data loader
    train.py                           Training pipeline (Phase 1)

PHASE 2/
  crack-topology-gnn/
    image_to_graph/
      convert.py                       mask → PyG Data (skeletonize → sknw → features)
      features.py                      Node [N,6] and edge [E,7] feature extraction
      build_dataset.py                 Batch convert all masks to .pt graph files
    segmentation/
      model.py                         EnhancedGraphUNet + HybridGraphUNet (Grapher bottleneck)
      graph_layers/                    ViG Grapher, FFN, edge conv layers
    link_prediction/
      model.py                         5 encoders + MLPEdgePredictor + MLPNodePredictor

  segmentation_crack_seg/
    models/unet.py                     Phase 1 DoubleConv + UNET
    models/hybrid_unet.py              Phase 1 GraphUNet (unchanged)
    crack_dataset.py                   Loader for crack_seg_clean dataset
    train.py                           Training pipeline for Phase 1 model on new dataset

  Data Set/
    crack_seg_clean/
      clean/train/images+masks         4,071 training pairs
      clean/test/images+masks          698 test pairs
```
