# Stage 2: Image-to-Graph Conversion

## Overview

Stage 2 sits between the segmentation model (Stage 1) and the GNN link predictor (Stage 3).  
It takes binary crack masks — either ground-truth labels or model-generated masks from Stage 1 — and converts them into graph-structured data that captures the full topology, geometry, and severity of each crack network.

```
Stage 1: HybridGraphUNet (segmentation) → binary crack mask (PNG)
                                    ↓
Stage 2: image_to_graph   → PyG Data objects (.pt)  ← THIS STAGE
                                    ↓
Stage 3: GNN link prediction (topology understanding proof)
```

---

## Directory Structure

```
image_to_graph/
  __init__.py          — package entry point, exports mask_to_graph
  features.py          — distance transform, node feature extraction, edge feature extraction
  convert.py           — core mask_to_graph() function
  visualize.py         — 4-panel per-image visualisation
  build_dataset.py     — CLI batch processor (produces .pt files + visualisations)
  read_graphs.py       — inspection utility for saved .pt datasets
  IMPLEMENTATION.md    — this file
```

Outputs written to `outputs/clean_graphs/` (crack_seg_clean run):
```
outputs/clean_graphs/
  graphs/
    train_graphs.pt       — list of 3728 PyG Data objects
    test_graphs.pt        — list of 636 PyG Data objects
    feature_schema.json   — column names for x and edge_attr
    stats.json            — per-split dataset statistics + curation funnel
  visualizations/
    train/  *.png
    test/   *.png
```

---

## Pipeline Steps

### Step 1 — Load and Binarise the Mask

```python
mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
```

The mask is thresholded to `{0, 1}` values (1 = crack pixel, 0 = background). Both ground-truth DeepCrack labels and model-generated masks from Stage 1 use white = crack convention, so no special handling is needed per source.

### Step 2 — Skeletonise

```python
from skimage.morphology import skeletonize
skeleton = skeletonize(binary_mask).astype(np.uint8)
```

Skeletonisation collapses the variable-width crack region down to a 1-pixel-wide centreline. This separates **structure** (topology, connectivity, branching) from **severity** (thickness, width) — structure lives in the skeleton graph, severity is recovered separately from the original mask via the distance transform.

**Why skeletonise rather than work directly on the mask?**  
Crack masks have irregular, variable-width shapes. Building a graph directly on the raw mask would produce meaningless node positions and edges that conflate geometry with width. The skeleton gives a canonical, topology-preserving centreline that is independent of crack width.

### Step 3 — Build the NetworkX Graph (sknw)

```python
import sknw
nx_graph = sknw.build_sknw(skeleton)
```

`sknw` analyses the skeleton and identifies:
- **Junction pixels** (where 3+ skeleton branches meet) → nodes
- **Endpoint pixels** (skeleton tips) → nodes  
- **Path pixels** between junctions/endpoints → edge with `pts` attribute (ordered array of (y, x) pixel coordinates)

This gives a compact graph where nodes are topologically significant points and edges carry the full pixel-level path between them.

### Step 3b — Spur Pruning *(new)*

Raw skeletons from noisy or rough mask boundaries produce short "spur" branches — false leaf branches that create spurious endpoint nodes. Since `is_endpoint` is the most important feature for link prediction (endpoints are crack tips), spurious ones corrupt the signal.

```python
nx_graph = _prune_spurs(nx_graph, prune_ratio=0.1)
```

**Algorithm:** iteratively remove leaf edges (edges where at least one endpoint has degree 1) whose geometric path length is less than `prune_ratio × longest_branch_length`. The ratio is relative to the longest branch in the *current* graph so the threshold is scale-aware — it adapts to the size of each individual crack network. Iteration continues until no more prunable edges remain.

**Why iterate?** Removing a spur can expose a previously interior node as a new leaf. One pass is not enough for chains of short branches.

**Why relative threshold?** A fixed pixel threshold would over-prune small graphs and under-prune large ones. Using `prune_ratio × max_branch` gives a consistent behaviour across the ~5k masks which vary widely in crack density and image scale.

**Topology-preserving:** only leaf edges are ever removed. Interior edges (where both endpoints have degree ≥ 2) are never touched, so the global connectivity of the crack network is preserved.

Controlled by `--prune-ratio` (default `0.1`). Set to `0.0` to disable.

### Step 3c — Degenerate Graph Filter *(new)*

After pruning, some graphs may be too small to be meaningful for link prediction (a single node with no edges, or an isolated pair). These are dropped and logged.

```python
if nx_graph.number_of_nodes() < min_nodes or nx_graph.number_of_edges() == 0:
    return None, None, None, 'degenerate'
```

Controlled by `--min-nodes` (default `3`). The curation funnel (raw → empty_mask → empty_graph → degenerate → converted) is written to `stats.json` and printed at the end of each run.

### Step 4 — Distance Transform for Thickness

```python
radius_map = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
dist_map = radius_map * 2.0  # radius → diameter (full crack width)
```

**Why distance transform instead of pixel counting (Phase 1 approach)?**  
Phase 1 measured thickness by counting crack pixels in a fixed 5×5 window. This is sensitive to window size and overestimates thickness at diagonal cracks. The Euclidean distance transform computes, at every crack pixel, the radius of the largest inscribed circle — giving the true local crack width in pixels, independent of orientation. Multiplying by 2 converts radius to diameter (full width). This is the geometrically correct measurement.

### Step 5 — Node Feature Extraction

Each node gets a 6-dimensional feature vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x_norm` | x coordinate / image width — normalised to [0, 1] |
| 1 | `y_norm` | y coordinate / image height — normalised to [0, 1] |
| 2 | `thickness` | crack width at node (px), from distance transform |
| 3 | `degree` | number of edges connected to this node |
| 4 | `is_endpoint` | 1.0 if degree == 1 (crack tip), else 0.0 |
| 5 | `is_junction` | 1.0 if degree ≥ 3 (branching point), else 0.0 |

**Design decisions:**

- **Normalised coordinates** rather than raw pixel coordinates — GNNs are sensitive to feature scale; [0,1] works across images of any resolution and keeps the spatial features on the same scale as the binary flags.
- **degree, is_endpoint, is_junction** are all included. `degree` is continuous; `is_endpoint` and `is_junction` are explicit binary flags that make the structural role of each node immediately clear to the GNN without needing to learn a threshold on `degree`. These are the most important signals for link prediction: endpoints are crack tips that may propagate, junctions are branching points that have already connected.

> **Important for Stage 3:** `degree`, `is_endpoint`, and `is_junction` (columns 3–5) are computed on the **full graph** here and saved to disk. Stage 3 must **recompute these three columns** from only the observed (message-passing) edges after the edge split — otherwise stale degree values leak the location of hidden edges to the model. Columns 0–2 (`x_norm`, `y_norm`, `thickness`) are never overwritten.

### Step 6 — Edge Feature Extraction

Each edge gets a 7-dimensional feature vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `path_length` | geometric length of the skeleton path (px) |
| 1 | `euclidean_dist` | straight-line distance between the two nodes (px) |
| 2 | `tortuosity` | `path_length / euclidean_dist` — always ≥ 1.0 |
| 3 | `angle_sym` | crack orientation in [0°, 180°), symmetric for undirected edges |
| 4 | `avg_thickness` | mean crack width along the segment (px) |
| 5 | `min_thickness` | thinnest point along the segment (px) |
| 6 | `max_thickness` | widest point along the segment (px) |

**Design decisions:**

- **Geometric path length vs pixel count** — the path length is computed as the sum of step distances along the waypoints `[node_u] + path_pixels + [node_v]`, not just `len(pts)`. Counting pixels alone underestimates length because it excludes the node-to-edge-pixel gaps and treats diagonal steps as equal to cardinal steps. The geometric sum ensures tortuosity ≥ 1.0 by the triangle inequality, which is the physically correct constraint (no path can be shorter than the straight line).

- **Tortuosity** — this feature does not exist in Phase 1. For link prediction, tortuosity indicates how curved a crack segment is. A straight crack (tortuosity ≈ 1.0) is likely under uniform stress; a highly curved crack (tortuosity >> 1.0) indicates more complex fracture propagation. This is a strong predictor of where new connections might form.

- **Symmetric angle [0°, 180°)** — Phase 1 stored the [0°, 360°) angle and assigned the same value to both directions (u→v) and (v→u), which is inconsistent. For an undirected graph, the crack orientation should be the same regardless of which endpoint is called "u" or "v". Using `angle % 180.0` achieves true symmetry: an angle of 45° and its reverse 225° both map to 45°. This makes the edge representation geometrically correct for undirected link prediction.

- **min and max thickness** alongside `avg_thickness` — average alone loses information about whether a segment has a uniform width or has a pinch point (locally thin section, more vulnerable to fracture) or a bulge. `min_thickness` in particular flags structurally weak points along an edge, which is relevant to predicting where cracks will break or merge.

- **Both directions stored** — PyG convention for undirected graphs stores each edge twice: `(u, v)` and `(v, u)`. Both copies receive the same feature vector (since angle is symmetric). This allows message passing in both directions without any asymmetric information.

### Step 7 — Graph-Level Metadata

Each `Data` object also carries:

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `pos` | FloatTensor | [N, 2] | Node pixel positions **(x, y)** — used by Stage 3 for hard-negative sampling (spatially-near unconnected pairs). Stored separately from `x` so Stage 3 can read positions even after overwriting feature columns 3–5. |
| `filename` | str | — | source mask filename |
| `split` | str | — | `'train'` or `'test'` |
| `img_h`, `img_w` | int | — | original image dimensions |
| `crack_pixels` | int | — | total number of crack pixels in the mask |
| `crack_density` | float | — | `crack_pixels / (H × W)` |

`pos` stores `(x_pixel, y_pixel)` — i.e. `(column, row)` in image coordinates, consistent with the `(x_norm, y_norm)` ordering in the feature matrix. Stage 3 computes pairwise Euclidean distances from `pos` to identify candidate hard negatives.

---

## Why This Graph Representation?

The fundamental choice is to represent each crack image as a **skeleton graph** rather than alternatives like:

- **Superpixel graphs** — would merge many irrelevant background regions and lose crack topology
- **Grid graphs** (pixel-level) — too large (384×544 = 208K nodes), too dense, no topological meaning
- **Region adjacency graphs** — same issue as superpixels; background dominates

The skeleton graph is the minimal representation that preserves everything needed for link prediction:
1. Where cracks are (node positions)
2. How cracks connect (edges)
3. Where cracks branch (junction nodes)
4. Where cracks end (endpoint nodes — the most likely places for new connections)
5. How wide cracks are (thickness features)
6. How curved each segment is (tortuosity)

For link prediction specifically, the key question is: **which pairs of nodes that are currently unconnected will become connected as the crack evolves?** Endpoint nodes are the natural candidates (crack tips propagate), and junctions show where connections have already formed. The spatial features (normalised x, y) allow the GNN to reason about proximity, and the thickness and tortuosity features provide crack severity context.

---

## Dataset Statistics (crack_seg_clean, prune_ratio=0.1, min_nodes=3)

**Curation funnel:**

| | Train | Test |
|--|-------|------|
| Raw masks | 4071 | 698 |
| Skipped: empty mask | 3 | 3 |
| Skipped: empty graph | 0 | 0 |
| Skipped: degenerate (< 3 nodes or 0 edges) | 340 | 59 |
| **Converted (saved to disk)** | **3728** | **636** |

**Graph statistics (converted graphs only):**

| | Train | Test |
|--|-------|------|
| Graphs | 3728 | 636 |
| Nodes (mean / max) | 25.8 / 625 | 28.0 / 1003 |
| Edges (mean / max) | 16.9 / 238 | 17.8 / 259 |
| Crack density (mean) | 0.0508 | 0.0525 |
| Crack density (max) | 0.3152 | 0.4195 |

The high degenerate-filter rate (~8% train, ~8% test) reflects the diversity of the crack_seg_clean sources — many images contain very sparse or isolated crack marks that collapse to trivially small graphs after spur pruning. These are correctly excluded: link prediction requires at least 3 nodes to produce meaningful hidden-edge samples.

---

## Usage

**Process clean crack_seg_clean dataset (train + test, one call per split):**
```bash
cd crack-topology-gnn

python3 image_to_graph/build_dataset.py \
    --mask-dir  "/path/to/crack_seg_clean/clean/train/masks" \
    --image-dir "/path/to/crack_seg_clean/clean/train/images" \
    --split train \
    --output-dir outputs/clean_graphs \
    --no-vis

python3 image_to_graph/build_dataset.py \
    --mask-dir  "/path/to/crack_seg_clean/clean/test/masks" \
    --image-dir "/path/to/crack_seg_clean/clean/test/images" \
    --split test \
    --output-dir outputs/clean_graphs \
    --no-vis
```

**Key CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--prune-ratio` | `0.1` | Remove leaf branches shorter than this × longest branch. `0.0` disables. |
| `--min-nodes` | `3` | Drop graphs with fewer nodes after pruning. |
| `--no-vis` | — | Skip visualisation (strongly recommended for large runs). |
| `--vis-n` | `-1` | Max visualisations per split (`-1` = all). |

**Inspect a saved dataset:**
```bash
python3 image_to_graph/read_graphs.py --pt outputs/clean_graphs/graphs/train_graphs.pt --n 3
```

**Use in code:**
```python
import torch
dataset = torch.load('outputs/clean_graphs/graphs/train_graphs.pt', weights_only=False)
graph = dataset[0]

# graph.x          — node features [N, 6]  (cols 3-5 will be overwritten by Stage 3)
# graph.edge_index — edge connectivity [2, 2E]
# graph.edge_attr  — edge features [2E, 7]
# graph.pos        — node pixel positions [N, 2]  (x_pixel, y_pixel)
# graph.filename, graph.split, graph.img_h, graph.img_w, graph.crack_density
```

---

## Reference

- Phase 1 baseline: `/Users/tejasskamar/Practicum/PHASE 1/Aryan_Repo/TokenCutSeg/ImageToGraph/`
- **Current dataset:** crack_seg_clean — `clean/train/` (4071 masks → 3728 graphs), `clean/test/` (698 masks → 636 graphs)
  - Path: `/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/`
- Skeletonisation: `skimage.morphology.skeletonize` (Zhang-Suen thinning)
- Graph extraction: `sknw` (skeleton network analysis)
- Graph format: PyTorch Geometric `Data` objects
- Pipeline reference: `PIPELINE_AND_IMPLEMENTATION.md` (defines the structural-inference claim, all 6 known issues, and Stage 3 implementation tiers)
