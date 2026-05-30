# Stage 2: Image-to-Graph Conversion — Results

## What This Is

These results describe the **graph dataset produced by Stage 2**, not the output of any trained model. Stage 2 is a deterministic preprocessing step: it converts binary crack masks from the DeepCrack dataset into graph-structured data (PyTorch Geometric `Data` objects) ready for Stage 3 (GNN link prediction). No learning or model inference is involved here.

---

## Dataset Summary

| | Train | Test |
|--|------:|-----:|
| Source folder | `train_lab/` | `test_lab/` |
| Images processed | 300 | 237 |
| Images skipped (no crack) | 0 | 0 |
| Total nodes extracted | 7,397 | 8,247 |
| Total edges extracted | 6,663 | 7,434 |
| Visualisations saved | 300 | 237 |
| Runtime | ~2 min 12 s | ~1 min 51 s |

Every image in the DeepCrack dataset contains at least one crack — nothing was skipped. All 537 images were fully visualised.

---

## Graph Size Statistics

| Metric | Train | Test |
|--------|------:|-----:|
| Nodes — mean per graph | 24.7 | 34.8 |
| Nodes — median per graph | 18 | 24 |
| Nodes — min | 2 | 2 |
| Nodes — max | 205 | 180 |
| Edges — mean per graph | 22.2 | 31.4 |
| Edges — median per graph | 15 | 20 |
| Edges — min | 1 | 1 |
| Edges — max | 184 | 183 |

**What this tells us:** Crack networks are sparse. On average, a graph has roughly as many edges as nodes (edges ≈ nodes − 1), which is consistent with tree-like or lightly looped crack structures rather than dense meshes. The test set has slightly larger graphs on average, reflecting that the test images tend to have more complex, interconnected crack patterns.

---

## Crack Density Statistics

Crack density = fraction of image pixels that are crack pixels.

| Metric | Train | Test |
|--------|------:|-----:|
| Mean crack density | 0.0291 (2.9%) | 0.0433 (4.3%) |
| Max crack density | 0.1994 (19.9%) | 0.1925 (19.3%) |

**What this tells us:** Cracks are thin relative to the full image — on average less than 3–4% of pixels are crack. The most severe images approach 20% crack coverage. The test set has noticeably higher average density than train, meaning test images tend to be more heavily cracked, which is the harder regime for link prediction.

---

## Node Feature Statistics

Each node has a 6-dimensional feature vector `x[N, 6]`.

### Train Set (7,397 nodes total)

| Index | Feature | Mean | Std | Min | Max |
|-------|---------|-----:|----:|----:|----:|
| 0 | `x_norm` | 0.503 | 0.279 | 0.000 | 0.998 |
| 1 | `y_norm` | 0.489 | 0.239 | 0.000 | 0.997 |
| 2 | `thickness` (px) | 7.117 | 7.673 | 0.000 | 78.332 |
| 3 | `degree` | 1.802 | 1.000 | 0.000 | 6.000 |
| 4 | `is_endpoint` | 0.582 | 0.493 | 0.0 | 1.0 |
| 5 | `is_junction` | 0.355 | 0.479 | 0.0 | 1.0 |

### Test Set (8,247 nodes total)

| Index | Feature | Mean | Std | Min | Max |
|-------|---------|-----:|----:|----:|----:|
| 0 | `x_norm` | 0.487 | 0.275 | 0.000 | 0.998 |
| 1 | `y_norm` | 0.516 | 0.280 | 0.000 | 0.998 |
| 2 | `thickness` (px) | 7.529 | 8.754 | 0.000 | 102.301 |
| 3 | `degree` | 1.803 | 1.008 | 0.000 | 5.000 |
| 4 | `is_endpoint` | 0.588 | 0.492 | 0.0 | 1.0 |
| 5 | `is_junction` | 0.361 | 0.480 | 0.0 | 1.0 |

### Key Observations

- **`x_norm` and `y_norm` are well spread across [0, 1]** with mean ≈ 0.5 — nodes are distributed across the whole image rather than clustered, which is expected.
- **`thickness` has high variance (std ≈ 7–8 px)** — crack widths vary enormously across images and within a single image. A few very wide cracks reach 78–102 px (large surface fractures). The zero minimum occurs at degenerate sknw skeleton nodes at image borders where the distance transform is 0.
- **`degree` mean ≈ 1.8** — most nodes are close to degree 2 (chain nodes) or degree 1 (endpoints), confirming sparse, branching crack trees.
- **`is_endpoint` mean 0.58** — about 58% of all nodes are crack tips. This is the most important class for link prediction: these are the nodes where new crack connections are most likely to form as the crack propagates.
- **`is_junction` mean 0.36** — about 36% of nodes are branching junctions (degree ≥ 3). These nodes already represent places where multiple cracks have converged.

---

## Node Type Breakdown

| Node type | Train (mean/graph) | Train (% of nodes) | Test (mean/graph) | Test (% of nodes) |
|-----------|-------------------:|-------------------:|------------------:|------------------:|
| Endpoint (degree = 1) | 14.3 | 58.2% | 20.4 | 58.8% |
| Junction (degree ≥ 3) | 8.8 | 35.5% | 12.6 | 36.1% |
| Chain (degree = 2) | 1.6 | 6.3% | 1.8 | 5.1% |

**What this tells us:** The majority of nodes are either crack tips or junction points. Chain nodes (degree exactly 2, sitting in the middle of a straight segment) are rare because sknw merges long straight sections into a single edge — the intermediate pixels become the edge's `pts` path rather than individual nodes. This is the desired behaviour: the graph is compact and only retains topologically meaningful nodes.

The endpoint/junction ratio is remarkably consistent between train and test (~58% / ~36%), showing stable structural properties of the DeepCrack dataset across splits.

---

## Edge Feature Statistics

Each edge has a 7-dimensional feature vector `edge_attr[E, 7]`. Both directions are stored (PyG undirected convention), so the table below uses unique edges only (every other row).

### Train Set (6,663 unique edges)

| Index | Feature | Mean | Std | Min | Max |
|-------|---------|-----:|----:|----:|----:|
| 0 | `path_length` (px) | 39.509 | 61.999 | 2.000 | 709.377 |
| 1 | `euclidean_dist` (px) | 34.610 | 54.427 | 0.000 | 642.436 |
| 2 | `tortuosity` | 1.215 | 0.901 | 1.000 | 49.799 |
| 3 | `angle_sym` (°) | 86.387 | 57.105 | 0.000 | 179.869 |
| 4 | `avg_thickness` (px) | 8.763 | 7.627 | 0.667 | 68.076 |
| 5 | `min_thickness` (px) | 5.756 | 6.505 | 0.000 | 66.726 |
| 6 | `max_thickness` (px) | 12.788 | 10.015 | 2.000 | 83.119 |

### Test Set (7,434 unique edges)

| Index | Feature | Mean | Std | Min | Max |
|-------|---------|-----:|----:|----:|----:|
| 0 | `path_length` (px) | 30.813 | 49.241 | 2.000 | 730.836 |
| 1 | `euclidean_dist` (px) | 27.072 | 43.304 | 0.000 | 583.648 |
| 2 | `tortuosity` | 1.217 | 0.763 | 1.000 | 27.136 |
| 3 | `angle_sym` (°) | 89.201 | 55.160 | 0.000 | 179.215 |
| 4 | `avg_thickness` (px) | 9.608 | 8.671 | 0.667 | 94.384 |
| 5 | `min_thickness` (px) | 6.055 | 7.479 | 0.000 | 93.482 |
| 6 | `max_thickness` (px) | 14.130 | 11.416 | 2.000 | 110.332 |

### Key Observations

- **`path_length` > `euclidean_dist` in virtually all cases** — crack segments curve, so the path along the skeleton is longer than the straight line between endpoints. This gap is captured by tortuosity.
- **`tortuosity` mean ≈ 1.21, min = 1.0** — most crack segments are relatively straight (close to 1.0). The max of ~50 (train) and ~27 (test) comes from highly curved or looped crack segments. A crack with tortuosity of 2.0 is twice as long as the straight line between its endpoints, indicating a strongly curved fracture.
- **`angle_sym` is near-uniform across [0°, 180°)** with mean ≈ 86–89° and high std ≈ 55–57° — crack orientations are not biased toward any particular direction, which is consistent with random surface fracture patterns.
- **Thickness spread: `max_thickness` / `min_thickness` ≈ 2.2×** — crack segments typically widen by a factor of 2 from their thinnest to thickest point. `min_thickness` is the most structurally critical feature: pinch points (locally thin sections) are where cracks are most likely to break or propagate further.
- **Test edges are shorter on average** (path_length 30.8 vs 39.5) but thicker (avg_thickness 9.6 vs 8.8) — test cracks tend to be wider and more densely connected.

---

## Sanity Checks Passed

| Check | Status |
|-------|--------|
| All tortuosity values ≥ 1.0 | ✓ |
| All `is_endpoint`, `is_junction` values in {0.0, 1.0} | ✓ |
| `x_norm`, `y_norm` in [0, 1] | ✓ |
| `angle_sym` in [0°, 180°) | ✓ |
| `path_length` ≥ `euclidean_dist` for all edges | ✓ |
| Zero empty graphs (all 537 images converted) | ✓ |
| Both edge directions stored consistently | ✓ |

---

## Output Files

| File | Size | Contents |
|------|------|----------|
| `outputs/graphs/graphs/train_graphs.pt` | — | 300 PyG Data objects |
| `outputs/graphs/graphs/test_graphs.pt` | — | 237 PyG Data objects |
| `outputs/graphs/graphs/feature_schema.json` | — | Column names for `x` and `edge_attr` |
| `outputs/graphs/graphs/stats.json` | — | Per-split summary statistics |
| `outputs/graphs/visualizations/train/*.png` | — | 300 4-panel visualisations (all train images) |
| `outputs/graphs/visualizations/test/*.png` | — | 237 4-panel visualisations (all test images) |

---

## What Goes Into Stage 3

The `Data` objects saved here feed directly into the Stage 3 GNN link predictor. The most relevant fields:

- **`data.x`** `[N, 6]` — node features; the GNN uses these as initial node embeddings
- **`data.edge_index`** `[2, 2E]` — existing crack connections; defines the message-passing graph
- **`data.edge_attr`** `[2E, 7]` — edge features; used by edge-conditioned GNN layers
- **`data.filename`** — links predictions back to the original image for evaluation

Stage 3 trains two jointly-optimised tasks on these graphs:
1. **Node prediction** — identify endpoint nodes whose neighbours have been removed (missing crack tips)
2. **Edge prediction** — reconstruct hidden crack segments between nodes (link prediction)

### Stage 3 Results (DeepCrack test set)

| Task | Metric | Value |
|---|---|---|
| Node prediction | AUC-ROC | **0.8321** |
| Node prediction | Avg Precision | 0.4230 |
| Edge prediction | AUC-ROC | **0.7247** |
| Edge prediction | Hits@20 | **0.9622** |

See `link_prediction/IMPLEMENTATION.md` for full details.
