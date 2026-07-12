# Stage 2 — Image-to-Graph Conversion: Technical Report

**Project:** Crack Topology GNN — Infrastructure Crack Analysis via Graph Neural Networks  
**Author:** Tejas Kamar  
**Date:** July 2026  
**Status:** Complete — canonical graph dataset deployed in production pipeline  
**Dataset:** `outputs/clean_graphs/` — 3,728 train graphs, 636 test graphs

---

## Table of Contents

1. [Overview and Role in Pipeline](#1-overview-and-role-in-pipeline)
2. [The Representational Choice: Why Skeleton Graphs](#2-the-representational-choice-why-skeleton-graphs)
3. [Software Architecture](#3-software-architecture)
4. [Implementation: Step-by-Step Pipeline](#4-implementation-step-by-step-pipeline)
   - 4.1 [Step 1 — Load and Binarise the Mask](#41-step-1--load-and-binarise-the-mask)
   - 4.2 [Step 2 — Skeletonisation (Zhang-Suen Thinning)](#42-step-2--skeletonisation-zhang-suen-thinning)
   - 4.3 [Step 3 — Skeleton-to-Graph Extraction (sknw)](#43-step-3--skeleton-to-graph-extraction-sknw)
   - 4.4 [Step 4 — Spur Pruning](#44-step-4--spur-pruning)
   - 4.5 [Step 5 — Degenerate Graph Filter](#45-step-5--degenerate-graph-filter)
   - 4.6 [Step 6 — Distance Transform for Thickness](#46-step-6--distance-transform-for-thickness)
   - 4.7 [Step 7 — Node Feature Extraction](#47-step-7--node-feature-extraction)
   - 4.8 [Step 8 — Edge Feature Extraction](#48-step-8--edge-feature-extraction)
   - 4.9 [Step 9 — Graph Metadata and PyG Object Assembly](#49-step-9--graph-metadata-and-pyg-object-assembly)
5. [Feature Engineering: Design Rationale](#5-feature-engineering-design-rationale)
6. [Phase 1 vs Phase 2: What Changed and Why](#6-phase-1-vs-phase-2-what-changed-and-why)
7. [Curation Funnel](#7-curation-funnel)
8. [Dataset Statistics](#8-dataset-statistics)
9. [Visualisation System](#9-visualisation-system)
10. [Bugs Found and Fixed](#10-bugs-found-and-fixed)
11. [Pipeline Integration with Stage 3](#11-pipeline-integration-with-stage-3)
12. [Sanity Checks](#12-sanity-checks)
13. [Reproducibility](#13-reproducibility)
14. [Appendix: File Locations](#14-appendix-file-locations)

---

## 1. Overview and Role in Pipeline

Stage 2 is a deterministic preprocessing step that converts binary crack segmentation masks into graph-structured data. It sits between Stage 1 (HybridGraphUNet segmentation) and Stage 3 (GNN link prediction). No model training or inference occurs here — Stage 2 is a pure data transformation.

```
Stage 1: HybridGraphUNet
  Raw RGB image (448×448) → Binary crack mask (448×448)
                                    ↓
Stage 2: image_to_graph (THIS STAGE)
  Binary mask → PyTorch Geometric Data object
  Fields: x [N,6], edge_index [2,2E], edge_attr [2E,7], pos [N,2], metadata
                                    ↓
Stage 3: GNN Link Prediction
  PyG graphs → GINE / GAT / GCN / GraphSAGE / MLP
  Task: reconstruct hidden/occluded crack connections
```

The transformation from pixel space to graph space is the critical architectural decision that enables Stage 3's topology-aware reasoning. A pixel-level model can only learn "where are crack pixels?" A graph model can learn "how do crack tips reconnect? Where do cracks branch? Which disconnected segments belong to the same propagating fracture?"

Stage 2 makes this possible by extracting three things from every mask:
1. **Topology** — how cracks connect (graph structure: nodes and edges)
2. **Geometry** — where each node sits in space (normalised (x, y) coordinates)
3. **Severity** — how wide, how curved, how thick each crack segment is (edge features)

---

## 2. The Representational Choice: Why Skeleton Graphs

Before describing the implementation, it is important to justify the fundamental representational choice. Several alternative graph representations exist for crack images. Each was considered and rejected.

### Alternative 1: Pixel-Level Grid Graph

Treat every crack pixel as a node and connect adjacent crack pixels as edges. This preserves all spatial information but produces graphs of 200–40,000+ nodes for a 448×448 image (depending on crack density). The graphs would be too large for GNN mini-batching, and the edges carry no semantic meaning — a pixel connected to its right neighbour is not a topologically meaningful relationship.

**Rejected:** computationally intractable and semantically meaningless.

### Alternative 2: Superpixel Graph

Segment the image into superpixels (SLIC or Felzenszwalb) and connect adjacent superpixels. This produces manageable graph sizes but conflates crack and non-crack regions into single superpixels at boundaries, and the graph structure reflects image segmentation rather than crack topology.

**Rejected:** does not encode crack topology — it encodes image region adjacency.

### Alternative 3: Region Adjacency Graph

Connect each connected component of crack pixels to its adjacent background regions. This captures rough connectivity but loses the internal branching structure of crack networks.

**Rejected:** loses intra-crack topology (branching, looping, dead-end tips).

### Chosen: Skeleton Graph

The skeleton (medial axis) of the crack mask is a 1-pixel-wide centreline that preserves the full topological structure while discarding width information. Skeletonisation uniquely decomposes a crack network into:
- **Nodes:** topologically significant points — junctions (where 3+ branches meet) and endpoints (crack tips)
- **Edges:** simple paths connecting nodes, carrying the full pixel-level path as auxiliary data

This representation is:
- **Compact:** mean 25.8 nodes per graph (vs thousands for pixel-level)
- **Topology-preserving:** all junctions, branches, and dead-ends are captured
- **Separation-of-concerns:** topology lives in graph structure; severity lives in edge/node features recovered from the original mask via distance transform

The skeleton graph is the minimal representation that retains everything needed for the link prediction task: which nodes are crack tips (will propagate), which are junctions (have already merged), and what are the geometric properties of the paths connecting them.

---

## 3. Software Architecture

```
image_to_graph/
  __init__.py         — package entry; exports mask_to_graph()
  convert.py          — core mask_to_graph() function (pipeline steps 1–9)
  features.py         — distance transform, node features, edge features
  build_dataset.py    — CLI batch processor (produces .pt files + visualisations)
  visualize.py        — 4-panel per-image visualisation
  read_graphs.py      — inspection utility for saved .pt datasets
  IMPLEMENTATION.md   — inline pipeline documentation
  RESULTS.md          — DeepCrack baseline dataset statistics
```

**Output layout (canonical run on crack_seg_clean):**

```
outputs/clean_graphs/
  graphs/
    train_graphs.pt       — list of 3,728 PyG Data objects
    test_graphs.pt        — list of 636 PyG Data objects
    feature_schema.json   — column names for x and edge_attr
    stats.json            — per-split statistics + curation funnel
  visualizations/
    train/  *.png         — 4-panel visualisation per mask
    test/   *.png
```

**External dependencies:**

| Library | Role |
|---|---|
| `skimage.morphology.skeletonize` | Zhang-Suen topological thinning |
| `sknw` | Skeleton network analysis — junction/endpoint detection + edge path extraction |
| `cv2.distanceTransform` | Euclidean distance transform for crack width measurement |
| `torch_geometric.data.Data` | PyG graph container |
| `networkx` | Intermediate graph representation (pruning, feature computation) |
| `albumentations` | (Not used here — in Stage 1) |

---

## 4. Implementation: Step-by-Step Pipeline

The core function `mask_to_graph(mask_path, split, prune_ratio, min_nodes)` in `convert.py` executes all 9 steps and returns a 4-tuple: `(data, skeleton, nx_graph, skip_reason)`.

### 4.1 Step 1 — Load and Binarise the Mask

```python
mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
```

**Input:** any mask image (PNG/JPG/BMP) in grayscale. Both ground-truth DeepCrack labels and model-generated masks from Stage 1 use white = crack convention.

**Binarisation:** Otsu or manual threshold at 127 (midpoint) converts to `{0, 1}` values. Using integer `1` (not `255`) is important because `skimage.skeletonize` expects boolean-like input — passing `{0, 255}` would not correctly thin the mask.

**Error handling:** if `cv2.imread` fails (bad path, corrupted file) or the resulting binary mask is all zeros (no crack pixels), the function returns `skip_reason = 'empty_mask'` and the mask is counted in the curation funnel but not saved.

### 4.2 Step 2 — Skeletonisation (Zhang-Suen Thinning)

```python
from skimage.morphology import skeletonize
skeleton = skeletonize(binary_mask).astype(np.uint8)
```

**Algorithm:** `skimage.morphology.skeletonize` implements the Zhang-Suen (1984) parallel thinning algorithm. It iteratively removes boundary pixels of the foreground that are not "simple points" — pixels whose removal would change the topology of the object (disconnect it or create/remove holes). The algorithm terminates when no more pixels can be removed, leaving a 1-pixel-wide medial axis.

**Properties of the resulting skeleton:**
- Exactly 1 pixel wide everywhere
- Topologically equivalent to the original mask (same number of connected components, same loop structure)
- Centrally positioned within the crack (approximately the medial axis)
- 8-connected (pixels at 45° angles remain connected)

**Why this step before graph extraction:**
Crack masks have variable width (1–100+ pixels). Without skeletonisation, a single crack would produce a slab of nodes at every width variation. After skeletonisation, the crack becomes a 1-pixel centreline that can be unambiguously decomposed into junctions, endpoints, and paths.

**Output:** `skeleton` is a `uint8` array of same shape as `binary_mask`, with `1` on skeleton pixels and `0` elsewhere. This array is also saved for the 4-panel visualisation but never written to the `.pt` file.

### 4.3 Step 3 — Skeleton-to-Graph Extraction (sknw)

```python
import sknw
nx_graph = sknw.build_sknw(skeleton)
```

`sknw` (skeleton network analysis) analyses the connectivity pattern of the skeleton array and builds a NetworkX graph. Its algorithm:

1. **Classify each skeleton pixel** by the number of 8-connected skeleton neighbours:
   - 0 neighbours: isolated pixel (treated as endpoint)
   - 1 neighbour: **endpoint** → node
   - 2 neighbours: **chain pixel** → not a node, becomes part of an edge's path
   - 3+ neighbours: **junction** → node

2. **Trace paths** between nodes: starting from each node, follow the chain pixels in each direction until another node is reached. The ordered sequence of `(y, x)` pixel coordinates along the path is stored in the edge attribute `pts`.

3. **Output:** a NetworkX `Graph` where:
   - Node attributes: `o` = `(y, x)` pixel coordinate of the node
   - Edge attributes: `pts` = `ndarray` of `(y, x)` path pixels between the two endpoint nodes

**What sknw gives us:**
- A compact graph where only topologically significant points are nodes
- Full pixel-level path information preserved in edge `pts` for feature extraction
- Self-loops possible (rare): a cycle in the skeleton where start and end pixel are the same

**What sknw does NOT give us:**
- Feature computation (we add that in steps 5–6)
- Spur removal (we add that in step 4)
- Any filtering of degenerate cases (we add that in step 5)

### 4.4 Step 4 — Spur Pruning

Raw skeletons derived from imperfect segmentation masks or mask boundaries produce short "spur" branches — false leaf branches extending from otherwise clean topology. These create spurious endpoint nodes that directly corrupt the Stage 3 task: the most important feature for link prediction is `is_endpoint` (crack tips), and spurious endpoints pollute this signal.

**Algorithm (`_prune_spurs` in `convert.py`):**

```python
def _prune_spurs(graph, prune_ratio: float):
    g = copy.deepcopy(graph)
    while True:
        lengths = {}
        for u, v, edata in g.edges(data=True):
            pts = edata.get('pts', np.array([]))
            waypoints = [(y_u, x_u)] + [(p[0], p[1]) for p in pts] + [(y_v, x_v)]
            L = sum(sqrt((w[i+1][0]-w[i][0])**2 + (w[i+1][1]-w[i][1])**2)
                    for i in range(len(waypoints)-1))
            lengths[(u, v)] = L
        
        threshold = prune_ratio * max(lengths.values())
        
        to_remove = [(u, v) for (u, v), L in lengths.items()
                     if L < threshold and (g.degree(u) == 1 or g.degree(v) == 1)]
        if not to_remove:
            break
        
        for u, v in to_remove:
            if g.has_edge(u, v):
                g.remove_edge(u, v)
        g.remove_nodes_from([n for n in list(g.nodes()) if g.degree(n) == 0])
    
    return g
```

**Key design decisions:**

1. **Relative threshold, not absolute:** The pruning threshold is `prune_ratio × max_branch_length` in the current graph. Using a fixed pixel threshold would over-prune small graphs and under-prune large ones. The relative threshold adapts to the scale of each individual crack network. `prune_ratio=0.1` (default) removes leaf branches shorter than 10% of the longest branch in the graph.

2. **Iterative pruning:** One pass is insufficient. Removing a short spur may expose a previously interior node as a new leaf (if its only other edge was also short). The algorithm iterates until convergence — no more prunable edges remain.

3. **Topology-preserving:** Only **leaf edges** are ever candidates for removal — edges where at least one endpoint has degree 1 in the current (possibly partially pruned) graph. Interior edges (both endpoints degree ≥ 2) are never removed. This guarantees that the global connectivity of the crack network is preserved.

4. **Geometric length, not pixel count:** Branch length is the sum of Euclidean step distances along waypoints (node → path pixels → node), not simply `len(pts)`. This correctly handles the distinction between diagonal steps (≈1.414 pixels) and cardinal steps (1.0 pixel).

5. **Deep copy before modification:** The input NetworkX graph is deep-copied before pruning. The original is used for the visualisation overlay (showing the pre-pruning skeleton), while the pruned copy feeds the feature extraction.

**Parameter:** `--prune-ratio` (default `0.1`). Set to `0.0` to disable entirely.

### 4.5 Step 5 — Degenerate Graph Filter

After spur pruning, some graphs are too small to be meaningful for the link prediction task.

```python
if nx_graph.number_of_nodes() < min_nodes or nx_graph.number_of_edges() == 0:
    return None, None, None, 'degenerate'
```

**Why `min_nodes=3`:** Link prediction requires at least two edges that can be formed from node pairs. With only 2 nodes and 1 edge, masking that edge gives a trivial isolated-pair problem. With 3 nodes and at least 2 edges, there are meaningful positive and negative pairs for training. A graph with 1 or 2 nodes after pruning typically corresponds to a very sparse, isolated crack mark — not a crack *network*.

**Why check edges separately:** A graph can have ≥ 3 nodes but 0 edges after aggressive pruning of self-loops and short branches. Such a graph would produce no positive pairs for the edge prediction task.

**Both conditions together** catch all cases where the resulting PyG graph cannot produce useful training samples in Stage 3.

### 4.6 Step 6 — Distance Transform for Thickness

The crack width (thickness) at every pixel is needed for both node and edge features. The correct way to measure this is the Euclidean distance transform.

```python
radius_map = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
dist_map = radius_map * 2.0  # radius → diameter (full crack width)
```

**What the distance transform computes:** For every pixel in the binary mask, `cv2.distanceTransform` computes the Euclidean distance to the nearest background pixel. At a crack pixel, this is the radius of the largest inscribed circle centred at that pixel — the largest circle that fits entirely within the crack at that point.

**Converting radius to diameter:** Multiplying by 2.0 gives the full crack width (diameter) rather than the half-width (radius). This is the physically meaningful measurement — a crack that is 10 pixels wide has a distance transform value of 5 at its centreline, giving `5 × 2 = 10` pixels diameter.

**Why distance transform over Phase 1's approach:**
Phase 1 measured thickness by counting crack pixels in a fixed 5×5 window around each skeleton point. This has two problems:
1. **Window size sensitivity:** a 5×5 window overestimates width at thin cracks (counts background), underestimates at wide cracks (misses outer pixels).
2. **Orientation bias:** diagonal cracks have fewer pixels in an axis-aligned window than horizontal/vertical cracks of the same true width.

The distance transform is both window-free and orientation-invariant — it gives the geometrically correct local width regardless of crack direction.

**5-mask parameter:** The fifth argument `5` to `cv2.distanceTransform` specifies the mask size for the approximation (3×3 or 5×5). The 5×5 mask gives a more accurate approximation of the true Euclidean distance, particularly for wider cracks where the 3×3 approximation accumulates error.

**Sampling:** To get the thickness at a specific point `(y, x)`:
```python
def sample_thickness(dist_map, y, x):
    yi = int(np.clip(round(y), 0, h-1))
    xi = int(np.clip(round(x), 0, w-1))
    return float(dist_map[yi, xi])
```
Rounding to the nearest integer coordinate and clipping to image bounds handles the case where sknw node positions fall exactly on the image border (distance transform is 0 at borders, giving correct zero thickness for border nodes).

### 4.7 Step 7 — Node Feature Extraction

Each node in the graph receives a 6-dimensional feature vector assembled in `compute_node_features()`.

```python
def compute_node_features(graph, dist_map, H, W):
    node_ids = list(graph.nodes())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    feats = []
    for nid in node_ids:
        y, x = graph.nodes[nid]['o']   # sknw stores (y, x)
        deg = float(graph.degree(nid))
        thickness = sample_thickness(dist_map, y, x)
        feats.append([
            float(x) / W,        # x_norm
            float(y) / H,        # y_norm
            thickness,           # crack width in px
            deg,                 # edge count
            float(deg == 1.0),   # is_endpoint flag
            float(deg >= 3.0),   # is_junction flag
        ])
    return torch.tensor(feats, dtype=torch.float), node_index
```

**Complete node feature schema:**

| Index | Name | Type | Range | Description |
|---|---|---|---|---|
| 0 | `x_norm` | float | [0, 1] | x pixel coordinate / image width |
| 1 | `y_norm` | float | [0, 1] | y pixel coordinate / image height |
| 2 | `thickness` | float | [0, ~100] | crack diameter at node in pixels (from distance transform) |
| 3 | `degree` | float | [0, 6] | number of edges connected to this node |
| 4 | `is_endpoint` | float | {0.0, 1.0} | 1.0 if degree == 1 (crack tip) |
| 5 | `is_junction` | float | {0.0, 1.0} | 1.0 if degree ≥ 3 (branching point) |

**Design rationale for each feature:**

**`x_norm`, `y_norm`:** Normalised to [0, 1] rather than raw pixel coordinates. GNNs aggregate features via weighted sums — if coordinates were in [0, 448] they would dominate the gradient signal relative to the binary flags in [0, 1]. Normalisation puts all features on comparable scales. Additionally, normalised coordinates are resolution-independent: a node at the image centre always has (0.5, 0.5) regardless of whether the image is 448×448 or 512×512.

**`thickness`:** The crack width at the node position captures local severity. Thick nodes are at the core of wide fractures; thin nodes are at crack tips or fine hairline cracks. For link prediction, thin endpoint nodes are the most likely propagation points — a hairline tip that meets another structure is more likely to join it than a wide fracture core.

**`degree`:** A continuous feature encoding connectivity. Most nodes have degree 1 (endpoint) or degree ≥ 3 (junction) after sknw's chain-pixel compression — chain nodes (degree 2) are rare because sknw fuses long straight paths into single edges. Including `degree` as a continuous feature rather than purely the binary flags allows the GNN to distinguish degree-3 from degree-4 junctions.

**`is_endpoint` and `is_junction`:** Explicit binary flags for the two topologically significant node types. These make the structural role immediately clear to the GNN without requiring it to learn a threshold on `degree`. **`is_endpoint` is the most important feature for Stage 3:** link prediction asks "which crack tips will reconnect?" and these flags directly identify those tips. Including both flags alongside `degree` gives the GNN redundant but unambiguous structural information.

**Critical Stage 3 constraint:** Columns 3–5 (`degree`, `is_endpoint`, `is_junction`) are computed on the **full graph** here (all edges visible). Stage 3 hides a subset of edges before message passing. After the edge split, Stage 3 **must recompute columns 3–5** from only the observed (non-hidden) edges — otherwise the degree values silently leak the location of hidden edges to the model (if a node's degree is 3 in the saved file but 2 in the observed graph, the model would know that 1 edge is hidden from this node). Columns 0–2 (`x_norm`, `y_norm`, `thickness`) are geometric and never change regardless of which edges are hidden.

### 4.8 Step 8 — Edge Feature Extraction

Each edge receives a 7-dimensional feature vector assembled in `compute_edge_features()`.

```python
for u, v, edata in graph.edges(data=True):
    pts = edata['pts']   # ordered (y, x) pixel array along the edge path
    
    y_u, x_u = float(graph.nodes[u]['o'][0]), float(graph.nodes[u]['o'][1])
    y_v, x_v = float(graph.nodes[v]['o'][0]), float(graph.nodes[v]['o'][1])
    
    euclidean_dist = sqrt((x_v - x_u)**2 + (y_v - y_u)**2)
    
    waypoints = [(y_u, x_u)] + [(p[0], p[1]) for p in pts] + [(y_v, x_v)]
    path_length = sum(sqrt((w[i+1][0]-w[i][0])**2 + (w[i+1][1]-w[i][1])**2)
                      for i in range(len(waypoints)-1))
    
    tortuosity = path_length / max(euclidean_dist, 1.0)
    angle_sym  = degrees(arctan2(y_v - y_u, x_v - x_u)) % 180.0
    
    thick = [sample_thickness(dist_map, py, px) for py, px in pts]
    avg_t, min_t, max_t = mean(thick), min(thick), max(thick)
    
    feat = [path_length, euclidean_dist, tortuosity, angle_sym, avg_t, min_t, max_t]
    # Store both directions (PyG undirected convention)
    src += [i, j]; dst += [j, i]; feats += [feat, feat]
```

**Complete edge feature schema:**

| Index | Name | Type | Range | Description |
|---|---|---|---|---|
| 0 | `path_length` | float | [1, ~730] | Geometric length of the skeleton path in pixels |
| 1 | `euclidean_dist` | float | [0, ~640] | Straight-line distance between the two nodes in pixels |
| 2 | `tortuosity` | float | [1.0, ~50] | `path_length / euclidean_dist` — deviation from straight |
| 3 | `angle_sym` | float | [0°, 180°) | Crack segment orientation, symmetric for undirected edges |
| 4 | `avg_thickness` | float | [0.67, ~94] | Mean crack diameter along the segment in pixels |
| 5 | `min_thickness` | float | [0, ~93] | Thinnest point along the segment in pixels |
| 6 | `max_thickness` | float | [2, ~110] | Widest point along the segment in pixels |

**Design rationale for each feature:**

**`path_length`:** The actual skeleton pixel count is the most direct measure of how long a crack segment is. Together with `euclidean_dist`, it characterises whether two nodes are close in space but connected by a long winding path (high curvature) or by a short direct path (low curvature).

**`euclidean_dist`:** The straight-line node-to-node distance in image coordinates. For link prediction, this is a strong prior: nodes that are far apart in pixel space are unlikely to connect through a hidden edge. Stage 3 uses `pos` (pixel positions) for hard-negative sampling, but having `euclidean_dist` as an explicit edge feature gives the GNN the spatial context for *existing* connections.

**`tortuosity`:** Defined as `path_length / euclidean_dist`. A perfectly straight crack segment has tortuosity = 1.0. Highly curved segments have tortuosity >> 1.0. This feature does not appear in Phase 1 and was added for Stage 2.

*Physical significance:* Tortuosity reflects fracture mechanics. Straight cracks propagate under simple mode-I (tensile) loading. Curved cracks indicate mixed-mode loading (shear + tension) or that the crack is deflecting around stronger aggregate particles. From a link prediction perspective, crack segments with high tortuosity have already deviated significantly from a straight path — they may indicate complex stress states where additional branching or reconnection is likely.

*Mathematical guarantee:* By the triangle inequality, the straight-line distance between any two points is less than or equal to any path connecting them. Therefore `tortuosity ≥ 1.0` for all valid inputs. The only exception would be sknw loop-edges where both nodes are at the same pixel (giving `euclidean_dist = 0`), which are handled by clamping the denominator: `max(euclidean_dist, 1.0)`.

**`angle_sym`:** The orientation of the crack segment in degrees, mapped to the range [0°, 180°) by the symmetrisation operation `degrees(arctan2(dy, dx)) % 180.0`.

*Why symmetric angle:* For an undirected graph, the crack segment from node u to node v and the segment from v to u are the same physical crack. If u is below-left of v, the directed angle is ~45°. The reverse direction gives ~225°. Both should map to the same value (45°) because the crack has one orientation, not two. The `% 180.0` operation achieves this: 225° → 45°. Phase 1 stored the raw [0°, 360°) angle, which assigned different values to the same physical crack depending on which node was called "u" and which was called "v" — an inconsistency that corrupted the edge feature.

*Why orientation matters for link prediction:* Two crack tips pointing in similar directions are geometrically more likely to belong to the same propagating crack than two tips pointing in orthogonal directions. `angle_sym` gives the GNN this cue without requiring it to derive orientation from coordinates.

*Note on Phase 2 upgrade:* The original Phase 1 implementation used `angle_sym` as a single scalar in [0°, 180°). A planned but not yet implemented upgrade is to replace this with `[sin(2θ), cos(2θ)]` — the standard encoding of directions modulo 180° that avoids the discontinuity at 0°/180°. The issue: `angle_sym = 0°` and `angle_sym = 179°` are nearly the same physical orientation but map to scalar values 0 and 179 — a large numerical difference for a tiny angular difference. The sin/cos encoding is continuous and wraps correctly. This upgrade is pending (see `link_prediction/edge_ablation.py` for the ablation study framework).

**`avg_thickness`:** The mean crack diameter along all path pixels between the two nodes, measured by sampling the distance transform at each path pixel. This is the primary severity metric for an edge.

**`min_thickness`:** The thinnest point along the crack segment. This is arguably the most structurally significant edge feature: a pinch point (locally thin section) is where the crack is most likely to fracture completely or propagate further. A segment with low `min_thickness` is more vulnerable than one with uniformly thick width. The GNN uses this to identify structurally weak connections.

**`max_thickness`:** The widest point along the segment. Together with `min_thickness`, this gives the thickness variability along the segment. A uniform segment has `max ≈ min`. A segment with large `max - min` has significant width variation, indicating heterogeneous fracture (multiple crack widths within one segment).

**PyG undirected convention:** For message passing in both directions, PyG requires each undirected edge to be stored as two directed edges: `(u→v)` and `(v→u)`. Both receive the same feature vector (since all features are symmetric: path_length, tortuosity, angle_sym, and thicknesses are all identical for both traversal directions).

### 4.9 Step 9 — Graph Metadata and PyG Object Assembly

The final step assembles all extracted data into a `torch_geometric.data.Data` object.

```python
pos = torch.tensor(
    [[float(nx_graph.nodes[nid]['o'][1]),   # x_pixel (column)
      float(nx_graph.nodes[nid]['o'][0])]   # y_pixel (row)
     for nid in ordered_nids],
    dtype=torch.float,
)

data = Data(
    x          = x,            # FloatTensor [N, 6]
    edge_index = edge_index,   # LongTensor  [2, 2E]
    edge_attr  = edge_attr,    # FloatTensor [2E, 7]
    pos        = pos,          # FloatTensor [N, 2]  — (x_pixel, y_pixel)
    filename   = os.path.basename(mask_path),
    split      = split,
    img_h      = H,
    img_w      = W,
    crack_pixels  = int(binary_mask.sum()),
    crack_density = float(binary_mask.sum()) / float(H * W),
)
```

**Complete Data object schema:**

| Attribute | Type | Shape | Description |
|---|---|---|---|
| `x` | FloatTensor | [N, 6] | Node feature matrix |
| `edge_index` | LongTensor | [2, 2E] | Edge connectivity (both directions) |
| `edge_attr` | FloatTensor | [2E, 7] | Edge feature matrix |
| `pos` | FloatTensor | [N, 2] | Node pixel positions (x_pixel, y_pixel) |
| `filename` | str | — | Source mask filename (links back to original image) |
| `split` | str | — | `'train'` or `'test'` |
| `img_h` | int | — | Original image height in pixels |
| `img_w` | int | — | Original image width in pixels |
| `crack_pixels` | int | — | Total crack pixels in the mask |
| `crack_density` | float | — | `crack_pixels / (H × W)` |

**Why `pos` is separate from `x`:**
`pos` stores raw pixel coordinates `(x_pixel, y_pixel)`, while `x[:,0:2]` stores normalised `(x_norm, y_norm)`. Stage 3 uses `pos` for hard-negative sampling — finding pairs of nodes that are spatially close in pixel space but not connected by an edge, to use as hard negative examples for the link prediction task. Storing raw pixel coordinates separately from the normalised features ensures that Stage 3 can compute pixel-space distances without denormalising, even after the feature matrix `x` has been modified (columns 3–5 are overwritten during the edge split).

**Coordinate system convention:** sknw stores positions as `(row, col)` = `(y, x)`. When assembling `pos`, the order is explicitly swapped to `(col, row)` = `(x_pixel, y_pixel)`. This is consistent with the `(x_norm, y_norm)` ordering in `x[:,0:2]` — both use x-first, y-second convention.

---

## 5. Feature Engineering: Design Rationale

This section collects the full reasoning for the 13-dimensional feature space (6 node + 7 edge) and its contrast with Phase 1.

### 5.1 Why These 6 Node Features?

The node feature set must answer the question: **what role does this node play in the crack network?**

- **Position (x, y):** where is it? Required for spatial reasoning — proximity is the strongest prior for link prediction.
- **Thickness:** how severe is the crack here? Thin endpoints are more likely to propagate.
- **Degree:** how many branches does this node already have? High-degree nodes are complex junctions.
- **is_endpoint / is_junction:** explicit structural role flags. These make the "crack tip" signal unambiguous to the GNN without it needing to learn a threshold.

### 5.2 Why These 7 Edge Features?

The edge feature set must answer: **what is the nature of the crack segment connecting these two nodes?**

- **path_length + euclidean_dist:** together, they characterise segment geometry (length and compactness).
- **tortuosity:** derived from the two above, but distinct — it directly encodes curvature as a single scalar. More informative for the GNN than having both length measures without their ratio.
- **angle_sym:** crack orientation. Two colinear endpoints pointing at each other are more likely to reconnect than two orthogonal tips.
- **avg + min + max thickness:** three-summary statistics of crack width along the segment. Three numbers convey the distribution without requiring the full per-pixel array.

### 5.3 Feature Scale and Normalisation

At the time of graph construction, features are stored in their natural units (pixels for distances, degrees for angles, raw values for binary flags). Stage 3 applies per-feature normalisation during the DataLoader pass:
- Positions `x_norm`, `y_norm`: already in [0, 1] — no further normalisation needed.
- `thickness`, `path_length`, `euclidean_dist`: normalised by the mean and std computed from the full training set.
- `tortuosity`: already bounded below by 1.0; normalised similarly.
- `angle_sym`: in [0, 180) — normalised to [0, 1] by dividing by 180.
- `avg/min/max_thickness`: normalised like `thickness`.
- `degree`, `is_endpoint`, `is_junction`: small integers / binary — normalisation applied but effect is minor.

---

## 6. Phase 1 vs Phase 2: What Changed and Why

Phase 1 (Aryan's codebase, `PHASE 1/Aryan_Repo/TokenCutSeg/ImageToGraph/`) produced a working prototype graph dataset but contained several issues that made it unsuitable for rigorous GNN training.

| Aspect | Phase 1 | Phase 2 | Why Changed |
|---|---|---|---|
| **Thickness measurement** | Pixel count in 5×5 window | Euclidean distance transform | Window method is orientation-biased and size-sensitive |
| **Angle encoding** | Raw [0°, 360°) — inconsistent for u→v vs v→u | Symmetric [0°, 180°) via `% 180.0` | Undirected edges must have the same feature in both directions |
| **Spur pruning** | None | Iterative relative-threshold pruning | Raw skeletons produce spurious endpoints that corrupt the `is_endpoint` signal |
| **Degenerate filter** | None | `min_nodes=3`, `min_edges=1` | Trivial graphs waste training time and cannot produce meaningful positive pairs |
| **Path length** | `len(pts)` (pixel count) | Geometric sum of step distances | Pixel count misses the node-endpoint gaps and treats diagonal steps as 1 pixel (wrong) |
| **Tortuosity** | Not computed | `path_length / euclidean_dist` | New feature: curvature is a strong crack topology signal not present in Phase 1 |
| **min/max thickness** | Not computed | Sampled from dist_map along all pts | `avg_thickness` alone misses pinch points and bulges |
| **`pos` field** | Not stored separately | FloatTensor [N, 2] in Data object | Stage 3 needs raw pixel coords for hard-negative sampling |
| **Curation logging** | No funnel tracking | `stats.json` with all skip counts | Reproducibility and debugging |
| **Feature schema** | Not documented | `feature_schema.json` | Makes Stage 3 column names explicit and reproducible |
| **Dataset** | DeepCrack (~537 images) | crack_seg_clean (4,769 images) | 8.9× larger, 6-source diversity, no leakage |

**The critical bug in Phase 1 angle encoding:**
Phase 1 stored the raw `arctan2(dy, dx)` angle in [−180°, 180°). For an undirected edge (u, v), the angle from u→v and the angle from v→u are exactly 180° apart. Both copies of the edge attribute in PyG should represent the same physical crack orientation, but Phase 1 assigned opposite values. This means the GNN received contradictory information about crack orientation depending on which message-passing direction it was aggregating from — a subtle but fundamental inconsistency.

---

## 7. Curation Funnel

For each mask in the dataset, the pipeline either produces a graph (success) or skips with a logged reason (failure). The funnel is tracked and written to `stats.json`.

**crack_seg_clean canonical run (prune_ratio=0.1, min_nodes=3):**

| Stage | Train | Test |
|---|---|---|
| Raw masks processed | 4,071 | 698 |
| Skipped: empty mask | 3 | 3 |
| Skipped: empty graph (sknw produced 0 nodes) | 0 | 0 |
| Skipped: degenerate (< 3 nodes or 0 edges after pruning) | 340 | 59 |
| **Converted and saved** | **3,728** | **636** |
| Skip rate | 8.4% | 8.7% |

**Notes on each skip category:**

**empty_mask (3 train, 3 test):** These 6 masks had zero white pixels — completely black images. This can occur from mask encoding issues in some dataset sources (CRACK500 occasionally produces all-black masks for images with no visible crack). These are correctly excluded: a zero mask has no crack content to graph.

**empty_graph (0 + 0):** No mask produced a skeleton with zero nodes. Every mask with at least one crack pixel produces at least one sknw node after skeletonisation. Zero occurrences confirm that the skeletonise → sknw pipeline is robust.

**degenerate (340 train, 59 test, ~8%):** The dominant skip category. Most of these are small isolated crack marks — a single straight line, or a very sparse hair crack — that after spur pruning collapse to a single edge with 2 nodes. Since `min_nodes=3` requires at least 3 nodes, these are excluded. This is appropriate: a graph with only 2 nodes and 1 edge cannot produce meaningful link prediction samples (no false negatives exist — the only non-edge is the reverse direction of the existing edge, but that's already in the graph).

**DeepCrack baseline (original Phase 2 run, pre-crack_seg_clean):**

| Stage | Train | Test |
|---|---|---|
| Raw masks | 300 | 237 |
| Skipped | 0 | 0 |
| **Converted** | **300** | **237** |

The DeepCrack dataset has 100% conversion rate because its images all contain clear, well-annotated crack regions that survive spur pruning. The higher skip rate on crack_seg_clean reflects the greater diversity of its sources (GAPs384 and CrackTree200 contain more marginal images).

---

## 8. Dataset Statistics

### 8.1 crack_seg_clean (Canonical — Used for All Training)

**Graph counts:**

| Split | Graphs |
|---|---|
| Train | 3,728 |
| Test | 636 |
| Total | 4,364 |

**Graph size distribution:**

| Metric | Train | Test |
|---|---|---|
| Nodes — mean | 25.8 | 28.0 |
| Nodes — median | 11 | 11 |
| Nodes — max | 625 | 1,003 |
| Nodes — min | 3 | 3 |
| Edges — mean (undirected) | 16.9 | 17.8 |
| Edges — median | 5 | 9.5 |
| Edges — max | 238 | 259 |
| Edges — min | 1 | 2 |

**Stage 3 usability:**

| Task | Train usable | Train % | Test usable | Test % |
|---|---|---|---|---|
| Node task (≥1 endpoint) | 3,727 | 99.97% | 636 | 100% |
| Edge task (≥2 edges) | 3,723 | 99.87% | 636 | 100% |

Both tasks are usable for essentially the full dataset. The node task requires at least one endpoint node to mask; only 1 train graph fails this (a closed loop with no tips). The edge task requires at least 2 undirected edges so that one can be hidden while another remains for message passing; only 5 train graphs fall below this threshold. Every test graph passes both filters.

**Crack density:**

| Metric | Train | Test |
|---|---|---|
| Mean crack density | 0.0509 (5.1%) | 0.0525 (5.2%) |
| Max crack density | 0.3152 (31.5%) | 0.4195 (42.0%) |

### 8.2 Node Feature Statistics (DeepCrack Baseline — Full Detail)

From the DeepCrack prototype run (300 train + 237 test graphs). These statistics validate the feature range and distribution before scaling to crack_seg_clean.

**Train set (7,397 nodes):**

| Index | Feature | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 0 | `x_norm` | 0.503 | 0.279 | 0.000 | 0.998 |
| 1 | `y_norm` | 0.489 | 0.239 | 0.000 | 0.997 |
| 2 | `thickness` (px) | 7.117 | 7.673 | 0.000 | 78.332 |
| 3 | `degree` | 1.802 | 1.000 | 0.000 | 6.000 |
| 4 | `is_endpoint` | 0.582 | 0.493 | 0.0 | 1.0 |
| 5 | `is_junction` | 0.355 | 0.479 | 0.0 | 1.0 |

**Test set (8,247 nodes):**

| Index | Feature | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 0 | `x_norm` | 0.487 | 0.275 | 0.000 | 0.998 |
| 1 | `y_norm` | 0.516 | 0.280 | 0.000 | 0.998 |
| 2 | `thickness` (px) | 7.529 | 8.754 | 0.000 | 102.301 |
| 3 | `degree` | 1.803 | 1.008 | 0.000 | 5.000 |
| 4 | `is_endpoint` | 0.588 | 0.492 | 0.0 | 1.0 |
| 5 | `is_junction` | 0.361 | 0.480 | 0.0 | 1.0 |

**Key observations:**
- `x_norm` and `y_norm` are well spread across [0, 1] with mean ≈ 0.5, confirming nodes are distributed across the full image.
- `thickness` has high variance (std ≈ 7–9 px). The zero minimum comes from border nodes where the distance transform correctly returns 0 (no crack pixels on the other side of the border).
- `degree` mean ≈ 1.8 confirms that most nodes are near degree-1 (endpoints) or degree-2 (chain). True degree-2 chain nodes are rare because sknw compresses straight paths into single edges.
- `is_endpoint` mean ≈ 0.58 — 58% of all nodes are crack tips. This is the primary target class for Stage 3's node task.

### 8.3 Node Type Breakdown

| Type | Train mean/graph | Train % | Test mean/graph | Test % |
|---|---|---|---|---|
| Endpoint (degree = 1) | 14.3 | 58.2% | 20.4 | 58.8% |
| Junction (degree ≥ 3) | 8.8 | 35.5% | 12.6 | 36.1% |
| Chain (degree = 2) | 1.6 | 6.3% | 1.8 | 5.1% |

The endpoint/junction ratio is consistent across splits (~58%/~36%), confirming stable structural properties. Chain nodes are rare (6%) because sknw's fundamental operation is to compress straight paths into single edges, leaving only topologically significant nodes.

### 8.4 Edge Feature Statistics (DeepCrack Baseline)

**Train set (6,663 unique edges):**

| Index | Feature | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 0 | `path_length` (px) | 39.51 | 62.00 | 2.00 | 709.38 |
| 1 | `euclidean_dist` (px) | 34.61 | 54.43 | 0.00 | 642.44 |
| 2 | `tortuosity` | 1.215 | 0.901 | 1.000 | 49.80 |
| 3 | `angle_sym` (°) | 86.39 | 57.11 | 0.00 | 179.87 |
| 4 | `avg_thickness` (px) | 8.76 | 7.63 | 0.67 | 68.08 |
| 5 | `min_thickness` (px) | 5.76 | 6.51 | 0.00 | 66.73 |
| 6 | `max_thickness` (px) | 12.79 | 10.02 | 2.00 | 83.12 |

**Test set (7,434 unique edges):**

| Index | Feature | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| 0 | `path_length` (px) | 30.81 | 49.24 | 2.00 | 730.84 |
| 1 | `euclidean_dist` (px) | 27.07 | 43.30 | 0.00 | 583.65 |
| 2 | `tortuosity` | 1.217 | 0.763 | 1.000 | 27.14 |
| 3 | `angle_sym` (°) | 89.20 | 55.16 | 0.00 | 179.22 |
| 4 | `avg_thickness` (px) | 9.61 | 8.67 | 0.67 | 94.38 |
| 5 | `min_thickness` (px) | 6.06 | 7.48 | 0.00 | 93.48 |
| 6 | `max_thickness` (px) | 14.13 | 11.42 | 2.00 | 110.33 |

**Key observations:**
- `tortuosity` min = 1.0 exactly for all valid edges (triangle inequality guarantee holds).
- `angle_sym` near-uniform across [0°, 180°) (mean ≈ 87°, std ≈ 56°) — crack orientations are not biased toward any direction.
- `max_thickness / min_thickness ≈ 2.2×` — crack segments typically vary by a factor of 2 in width, confirming that `min_thickness` and `max_thickness` capture non-redundant information relative to `avg_thickness`.
- Test edges are shorter but thicker than train edges, reflecting that test images in DeepCrack tend to have denser, wider cracking.

---

## 9. Visualisation System

Every processed mask optionally produces a 4-panel PNG visualisation (`visualize.py`).

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Panel 1     │  Panel 2     │  Panel 3     │  Panel 4     │
│  Original    │  Binary      │  Skeleton    │  Graph       │
│  RGB image   │  Mask        │  (1-px wide) │  Overlay     │
│  (optional)  │  (white=     │              │              │
│              │   crack)     │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Panel details:**

| Panel | Content | Rendering |
|---|---|---|
| 1 (optional) | Original RGB image | `imshow` |
| 2 | Binary mask | `imshow` grayscale 0/1 |
| 3 | Skeleton centreline | `imshow` grayscale 0/1 |
| 4 | Graph overlay on mask | Nodes + edges over `alpha=0.45` mask |

**Graph overlay encoding (Panel 4):**
- **Edges:** LineCollection coloured by `avg_thickness` using the YlOrRd (yellow→orange→red) colormap. Yellow = thin crack segments, red = thick crack segments. Colour bar shown on right.
- **Nodes by type:**
  - Endpoint (degree = 1): blue `#4da6ff`
  - Chain (degree = 2): white `#ffffff`
  - Junction (degree ≥ 3): red `#ff4d4d`
- **Node size:** scales with degree — `30 + degree × 10` (scatter point area in pt²). Junction nodes are visually larger than endpoint nodes.
- **Background:** dark `#1a1a1a` for contrast.
- **Title:** shows filename, node count, edge count, and crack density.

**When visualisation runs:** Visualisation is enabled by default (`--vis`). For large batch runs (4,071 masks), use `--no-vis` to skip it — each PNG takes ~200ms to render. The `--vis-n` flag limits to the first N visualisations per split for a quick sanity check.

---

## 10. Bugs Found and Fixed

Several correctness issues were identified during Phase 2 development and fixed before the canonical dataset was built.

### Bug 1 — Angle Discontinuity (Phase 1 inheritance)

**Problem:** Phase 1 stored `angle_sym` as a scalar in `[0°, 180°)` mapped from the full `arctan2` range by `% 180.0`. While the symmetrisation was correct, the choice of raw degrees as the representation creates a discontinuity: a crack at 1° and a crack at 179° are nearly parallel (only 2° apart in orientation space), but their scalar representations differ by 178.

**Fix implemented (edge_ablation.py):** Replace the single angle scalar with the 2D encoding `[sin(2θ), cos(2θ)]`. This is the standard representation for angles modulo 180° — it wraps continuously so that 0° and 180° map to the same point `(sin 0, cos 0) = (0, 1)`. The `2θ` argument maps the [0°, 180°) range to a full [0°, 360°) circle, enabling a continuous circular representation.

**Current status:** The `[sin(2θ), cos(2θ)]` encoding is implemented in `link_prediction/edge_ablation.py` as part of the edge feature ablation study. The canonical `outputs/clean_graphs/` dataset still uses the scalar encoding (captured in `feature_schema.json`). Switching to the 2D encoding is planned but requires rebuilding the graph dataset and retraining all models.

### Bug 2 — Path Length Underestimation (Phase 1 inheritance)

**Problem:** Phase 1 computed `path_length = len(pts)` — simply counting the number of skeleton pixels in the edge path. This has two errors:
1. It excludes the node-endpoint pixels at both ends of the path (the gap from the last path pixel to the node itself).
2. It treats diagonal pixel steps as 1 pixel (same as cardinal steps), when a diagonal 45° step is actually √2 ≈ 1.414 pixels.

These errors cause the tortuosity formula to underestimate curvature for diagonal paths.

**Fix:** Path length is computed as the geometric sum of Euclidean distances along waypoints: `[(y_u, x_u)] + pts + [(y_v, x_v)]`. This includes both endpoint gaps and correctly weights diagonal steps.

### Bug 3 — Integer Overflow in sknw Distance Computation

**Problem:** sknw stores node positions as NumPy integers (dtype often `int64` or platform-dependent). When computing `(y_v - y_u)^2 + (x_v - x_u)^2` for images where coordinates can be up to ~1000, intermediate squared values approach 10^6. If arithmetic is done in `int32`, this can overflow.

**Fix:** All coordinate arithmetic explicitly casts to Python `float` before squaring: `dy = float(graph.nodes[v]['o'][0]) - float(graph.nodes[u]['o'][0])`. This prevents any possible integer overflow regardless of image size.

### Bug 4 — Tortuosity Blow-up on Loop Edges

**Problem:** sknw occasionally produces self-loop edges (u = v) for cyclic crack structures where a skeleton loop has no junction pixel. The start and end nodes of such an edge are at the same pixel, giving `euclidean_dist = 0`. Dividing `path_length / euclidean_dist` would give infinity or NaN.

**Fix:** Clamp the denominator: `tortuosity = path_length / max(euclidean_dist, 1.0)`. For a self-loop, this gives `tortuosity = path_length` (treating the loop as a 1-pixel-long straight-line reference), which is a physically reasonable approximation for very short loops.

### Bug 5 — Inconsistent Node Position Convention

**Problem:** sknw stores node positions as `(row, col)` = `(y, x)`. The node feature extraction used `y, x = graph.nodes[nid]['o']` correctly for the feature matrix. But the `pos` field was initially assembled using `(row, col)` order directly, producing `pos = (y_pixel, x_pixel)` — inconsistent with `x[:,0:2] = (x_norm, y_norm)`.

**Fix:** The `pos` assembly in `convert.py` explicitly swaps to `(col, row)` = `(x_pixel, y_pixel)`:
```python
pos = torch.tensor(
    [[float(nx_graph.nodes[nid]['o'][1]),   # col = x
      float(nx_graph.nodes[nid]['o'][0])]   # row = y
     for nid in ordered_nids],
    dtype=torch.float,
)
```
Now `pos[i, 0]` = x_pixel and `pos[i, 1]` = y_pixel, consistent with `x[i, 0]` = x_norm and `x[i, 1]` = y_norm.

---

## 11. Pipeline Integration with Stage 3

### 11.1 What Stage 3 Receives

Stage 3 loads the saved `.pt` files via:
```python
dataset = torch.load('outputs/clean_graphs/graphs/train_graphs.pt', weights_only=False)
```

Each `Data` object in the list contains the full graph as built in Stage 2. Stage 3 then applies the **edge split** (masking) before every training batch:

1. **Select candidate edges to hide** — edges incident to endpoint nodes, or random edges for the edge task.
2. **Remove hidden edges** from `edge_index` and `edge_attr`.
3. **Recompute `degree`, `is_endpoint`, `is_junction`** (columns 3–5 of `x`) from the surviving edges only.
4. **Build negative samples** — node pairs (or non-edges) that the model must distinguish from positive (hidden) pairs.

The recomputation in step 3 is critical for correctness. If Stage 3 used the saved degree values from Stage 2, a node with a hidden edge would show degree=2 in the features but only have degree=1 in the message-passing graph — leaking the location of the hidden edge. By recomputing from the observed graph, the node's features accurately reflect what is visible to the model.

### 11.2 Graph-to-Training Usability

| Metric | Train | Test |
|---|---|---|
| Total graphs | 3,728 | 636 |
| Usable for node task (≥1 endpoint) | 3,727 (99.97%) | 636 (100%) |
| Usable for edge task (≥2 edges) | 3,723 (99.87%) | 636 (100%) |

Only 1 train graph has no endpoint nodes (a closed crack loop with no tips) and 5 train graphs have fewer than 2 undirected edges; both are excluded from their respective task losses. The test split is fully usable for both tasks — the Stage 2 degenerate filter (min_nodes=3) ensures every surviving test graph meets both thresholds.

### 11.3 Impact of Stage 2 Quality on Stage 3

Stage 2 quality directly determines what Stage 3 can learn:

| Stage 2 quality issue | Stage 3 impact |
|---|---|
| Spurious endpoints (unsuppressed spurs) | False positives in node task training labels |
| Missing nodes (broken skeleton from fragmented mask) | False negatives — real crack tips not labelled |
| Wrong thickness values | Edge features carry incorrect severity signal |
| Inconsistent angle encoding | Model receives contradictory orientation information |
| Too-small graphs not filtered | Trivial samples dilute training signal |

This is why the Phase 2 improvements (spur pruning, distance transform, correct angle encoding) are directly reflected in Stage 3 model performance — they are not cosmetic.

---

## 12. Sanity Checks

The following checks were verified on the canonical crack_seg_clean graph dataset:

| Check | Result |
|---|---|
| All tortuosity values ≥ 1.0 | ✅ |
| All `is_endpoint`, `is_junction` in {0.0, 1.0} | ✅ |
| `x_norm`, `y_norm` all in [0.0, 1.0] | ✅ |
| `angle_sym` all in [0°, 180°) | ✅ |
| `path_length ≥ euclidean_dist` for all edges | ✅ |
| `min_thickness ≤ avg_thickness ≤ max_thickness` | ✅ |
| Both edge directions stored in `edge_index` | ✅ |
| Both directions have identical `edge_attr` | ✅ |
| `pos[:, 0]` (x) all in [0, img_w − 1] | ✅ |
| `pos[:, 1]` (y) all in [0, img_h − 1] | ✅ |
| `crack_density` matches `crack_pixels / (img_h × img_w)` | ✅ |
| No NaN or Inf values in x or edge_attr | ✅ |

---

## 13. Reproducibility

**Running the full Stage 2 pipeline on crack_seg_clean:**

```bash
cd "/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn"

# Train split
python3 image_to_graph/build_dataset.py \
    --mask-dir  "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/train/masks" \
    --image-dir "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/train/images" \
    --split train \
    --output-dir outputs/clean_graphs \
    --prune-ratio 0.1 \
    --min-nodes 3 \
    --no-vis

# Test split
python3 image_to_graph/build_dataset.py \
    --mask-dir  "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/masks" \
    --image-dir "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/images" \
    --split test \
    --output-dir outputs/clean_graphs \
    --prune-ratio 0.1 \
    --min-nodes 3 \
    --no-vis
```

**Inspecting the saved dataset:**
```bash
python3 image_to_graph/read_graphs.py \
    --pt outputs/clean_graphs/graphs/train_graphs.pt \
    --n 3
```

**Loading in code:**
```python
import torch
dataset = torch.load('outputs/clean_graphs/graphs/train_graphs.pt', weights_only=False)
g = dataset[0]
# g.x          [N, 6]  — node features
# g.edge_index [2, 2E] — connectivity
# g.edge_attr  [2E, 7] — edge features
# g.pos        [N, 2]  — pixel positions (x, y)
# g.filename           — source mask filename
# g.crack_density      — float [0, 1]
```

**Determinism:** Stage 2 is fully deterministic — there is no random sampling. Given the same mask files and parameters, the output is bit-for-bit identical across runs.

---

## 14. Appendix: File Locations

| Artifact | Path |
|---|---|
| Core conversion function | `image_to_graph/convert.py` |
| Feature extraction | `image_to_graph/features.py` |
| Batch CLI | `image_to_graph/build_dataset.py` |
| Visualisation | `image_to_graph/visualize.py` |
| Dataset inspection | `image_to_graph/read_graphs.py` |
| Inline documentation | `image_to_graph/IMPLEMENTATION.md` |
| DeepCrack baseline stats | `image_to_graph/RESULTS.md` |
| **Canonical train graphs** | `outputs/clean_graphs/graphs/train_graphs.pt` (3,728 graphs) |
| **Canonical test graphs** | `outputs/clean_graphs/graphs/test_graphs.pt` (636 graphs) |
| Feature schema | `outputs/clean_graphs/graphs/feature_schema.json` |
| Dataset statistics | `outputs/clean_graphs/graphs/stats.json` |
| Train visualisations | `outputs/clean_graphs/visualizations/train/*.png` |
| Test visualisations | `outputs/clean_graphs/visualizations/test/*.png` |

**Upstream (Stage 1 → Stage 2):**
- Input masks: `/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/`
- Segmentation model: `outputs/segmentation_clean/checkpoints/` (HybridGraphUNet)

**Downstream (Stage 2 → Stage 3):**
- Link prediction training: `link_prediction/run.py --train-graphs outputs/clean_graphs/graphs/train_graphs.pt`
- Link prediction evaluation: `link_prediction/evaluate.py --test-graphs outputs/clean_graphs/graphs/test_graphs.pt`
