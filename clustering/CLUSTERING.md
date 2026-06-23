# Crack Graph Clustering — Implementation

## Purpose

Clusters crack skeleton graphs by structural shape to (a) test whether discrete crack types exist in this dataset, and (b) stratify GNN link-prediction evaluation by crack morphology. Labels are attached to PyG graph objects as `g.cluster_label` for downstream GNN analysis.

---

## Paper Summary (honest framing)

> We clustered all 4,364 crack skeleton graphs using 11 scale-invariant structural features — ratios and per-node densities (junctions-per-node, endpoints-per-node, cyclomatic density, branching ratio, coefficient-of-variation of segment length, spatial density) — so that graph size cannot drive cluster assignment. Winsorisation at the 99th percentile and RobustScaler normalisation were applied before spectral clustering (k-NN affinity, k=15) and K-Means validation. The size-correlation diagnostic confirmed success: corr(cluster label, n_nodes) = −0.024, near zero, versus 0.85+ in the size-confounded run. However, silhouette scores dropped to 0.36/0.30 and method agreement between spectral and K-Means fell to 27.1%, indicating that **crack topology in this dataset varies as a continuum rather than in discrete morphological types**. Spectral clustering isolated 332 graphs (7.6%) with zero junction nodes, average degree ≈ 1, and no cyclomatic structure — degenerate skeleton artifacts excluded from subsequent evaluation. The remaining graphs were stratified by branching character (K-Means stratum 0: low-branching; stratum 1: high-branching) for per-stratum link-prediction analysis. This matches the topology-manifold finding in prior crack GNN literature and provides honest context for our GNN evaluation.

---

## Pipeline Overview

```
.pt graph files (outputs/clean_graphs/graphs/)
      │
      ▼
PyG Data → NetworkX conversion (to_networkx + position attachment)
      │
      ▼
11 scale-invariant shape features per graph
(ratios and per-node densities — no raw counts)
      │
      ▼
Winsorize at 99th percentile (safety net)
RobustScaler (median/IQR normalisation)
      │
      ▼
Silhouette sweep (k=2..7, K-Means) → auto-select best K
      │
      ├──► Spectral Clustering, k-NN affinity (primary)
      └──► K-Means (validation + stratified eval)
            │
            ▼
  crack_pseudo_labels.csv   — filename, spectral_label, kmeans_label, all features
  cluster_plot.png          — PCA 2D projection coloured by cluster
  train_graphs_clustered.pt — original graphs + g.cluster_label (spectral)
  test_graphs_clustered.pt  — original graphs + g.cluster_label (spectral)
            │
            ▼ (stratified_eval.py)
  stratified_results.json   — per-stratum node AP / edge AP for all models
```

---

## Input

| Source | Path |
|--------|------|
| Train graphs | `outputs/clean_graphs/graphs/train_graphs.pt` |
| Test graphs | `outputs/clean_graphs/graphs/test_graphs.pt` |
| Total graphs | 4,364 (3,728 train + 636 test) |

Graphs are PyG `Data` objects produced by the Stage 2 image-to-graph conversion pipeline. Each graph represents one crack mask skeleton, with nodes at junctions/endpoints and edges along crack segments.

---

## Graph Conversion (PyG → NetworkX)

```python
nx_graph = to_networkx(pyg_data, to_undirected=True)

# Attach pixel positions to nodes
for idx, p in enumerate(pyg_data.pos):
    nx_graph.nodes[idx]['o'] = p   # 'o' = origin, sknw convention

# Compute edge weights from Euclidean distance between node positions
for u, v in nx_graph.edges():
    pu = nx_graph.nodes[u]['o']
    pv = nx_graph.nodes[v]['o']
    nx_graph.edges[u, v]['weight'] = ||pu - pv||
```

---

## Feature Extraction

**11 scale-invariant shape features** are computed per graph. Every raw count is divided by `n_nodes` (or another count) to strip out graph size. A small alligator crack and a large alligator crack now have the same features and land in the same cluster.

Raw counts (`n_nodes`, `n_edges`, `total_length`, etc.) are kept in the CSV as diagnostics but are **not fed to the clustering model**.

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `avg_degree` | `mean(degree)` | Average branching per node |
| `max_degree` | `max(degree)` | Highest junction order |
| `branching_ratio` | `n_junctions / max(n_endpoints, 1)` | Network vs linear character |
| `junctions_per_node` | `n_junctions / n_nodes` | How branchy per unit graph |
| `endpoints_per_node` | `n_endpoints / n_nodes` | How tip-heavy per unit graph |
| `cyclomatic_per_node` | `cyclomatic / n_nodes` | Loop density (alligator signal) |
| `edges_per_node` | `n_edges / n_nodes` | Connectivity density |
| `components_per_node` | `components / n_nodes` | Fragmentation rate |
| `avg_tortuosity` | `mean(path_length / euclidean_dist)` | Crack curvature/winding |
| `cv_length` | `std_length / max(avg_length, ε)` | Segment-length irregularity (scale-free) |
| `node_density` | `n_nodes / bounding_box_area` | Spatial compactness |

### Epsilon guards

Every ratio uses `max(denominator, 1)` or `max(denominator, 1e-6)` to prevent division by zero:

```python
branching_ratio    = n_junctions / max(n_endpoints, 1)   # integer guard: cap at n_junctions
junctions_per_node = n_junctions / max(n_nodes, 1e-6)
cv_length          = std_length  / max(avg_length, 1e-6)
```

Note: `branching_ratio` uses `max(..., 1)` not `max(..., 1e-6)`. Using `1e-6` when `n_endpoints = 0` creates million-scale ratios that corrupt the feature space.

### Pre-processing before scaling

```python
# Winsorize at 99th percentile — safety net for any remaining edge cases
p99 = np.percentile(X, 99, axis=0)
X   = np.clip(X, None, p99)

# RobustScaler: median/IQR, resistant to outliers
X_scaled = RobustScaler().fit_transform(X)
```

Winsorization is mild with ratio features (ratios are naturally bounded) but retained as a safety net. RobustScaler is preferred over StandardScaler because crack graph size distributions remain skewed even after taking ratios.

---

## Optimal K Selection

A silhouette score sweep over k=2..7 is run using K-Means on the scaled shape features:

```python
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=20)
    score = silhouette_score(X_scaled, km.fit_predict(X_scaled))
```

**Result on crack_seg_clean (11 scale-invariant features):**

| k | Silhouette |
|---|-----------|
| **2** | **0.3579** ← chosen |
| 3 | 0.3489 |
| 4 | 0.2436 |
| 5 | 0.2540 |
| 6 | 0.2431 |
| 7 | 0.2244 |

Score of 0.358 indicates **weak cluster separation** — crack shapes form a continuum, not discrete types. k=2 is still selected (highest score) but this reflects a coarse split along a gradient, not two clean morphological groups. This is the honest result.

*For reference, the size-confounded run (raw count features) gave k=2 silhouette = 0.932, agreement 98.9%, corr(label, n_nodes) = 0.85+ — those clusters were measuring size, not shape, and are discarded.*

---

## Clustering Methods

### Spectral Clustering (primary)

```python
SpectralClustering(
    n_clusters=k,
    affinity='nearest_neighbors',  # sparse k-NN graph, no gamma parameter
    n_neighbors=15,
    assign_labels='kmeans',
    n_init=20,
    n_jobs=-1,
)
```

Builds a sparse k-NN affinity graph in the 11-feature scaled space and clusters in the resulting graph Laplacian embedding. Better than K-Means for non-convex cluster shapes.

**Why `nearest_neighbors` not `rbf`?** With 4364 data points after RobustScaler, the RBF kernel `exp(-γ‖xi-xj‖²)` with γ=1.0 collapses to near zero for any pair of graphs that differ by more than ~2σ in any feature. This makes the affinity matrix nearly singular and spectral decomposition degenerates (4363 vs 1 cluster split). The k-NN graph avoids the gamma sensitivity entirely.

### K-Means (validation + stratified eval)

```python
KMeans(n_clusters=k, n_init=30)
```

Run alongside spectral to validate robustness. Also used as the primary basis for **Path A stratified evaluation** — the K-Means split gives a more balanced, continuously-motivated "linear-ish vs branched" partition than the spectral result.

**Agreement metric:** `mean(spectral_labels == kmeans_labels) × 100%` = **27.1%** on shape features.
Low agreement (< 50%) confirms the weak cluster structure — spectral and K-Means found different things because the data doesn't have strong clusters to find.

### Size-correlation diagnostic

```python
n_nodes_arr = np.array([r["_n_nodes"] for r in records])
corr = np.corrcoef(sp_labels, n_nodes_arr)[0, 1]
```

Printed after clustering. If `|corr| > 0.5`, size leaked back in. Current run: **corr = −0.024** ✅.

---

## Cluster Interpretation

### Shape-based run (CURRENT)

corr(cluster, n_nodes) = −0.024 ✅ — size is not driving clusters.

| Cluster | Method | Count | What it found |
|---------|--------|-------|---------------|
| Spectral 0 | Spectral | 4,032 | Normal crack population — any junctions or cycles present |
| **Spectral 1** | Spectral | **332** | **Degenerate artifacts** — zero junctions, max_degree = 1, avg_degree ≈ 1. Every node is an isolated endpoint. No branching, no cycles. Bad skeletons that slipped past `min_nodes` filter. |
| K-Means 0 | K-Means | 1,514 | Low-branching / linear-ish — low avg_degree, high components_per_node, low junction density |
| K-Means 1 | K-Means | 2,850 | High-branching / networked — higher avg_degree, more junctions per node, lower fragmentation |

**Key finding:** Silhouette = 0.36 (K-Means) / 0.30 (Spectral), method agreement = 27.1%.  
Crack graphs in this dataset **do not form discrete structural types** once size is removed. Shape variation is continuous.

**Use Spectral Cluster 1 as a quality filter (Path B):** 332 graphs with zero junctions are skeleton artifacts, not a crack type. Remove them and re-run evaluation to confirm the topology-proof result holds on clean data only.

**Use K-Means strata for stratified evaluation (Path A):** Breaks link-prediction AP down by structural complexity — tests whether GNN advantage over MLP concentrates in topologically complex (high-branching) cracks.

### Size-based run (SUPERSEDED — do not cite)

Used raw count features. Silhouette 0.932, agreement 98.9% — but corr(cluster, n_nodes) = 0.85+. The clusters were measuring graph size, not crack shape. **Discarded.**

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/clustering/crack_pseudo_labels.csv` | One row per graph: `filename`, `split`, `spectral_label`, `kmeans_label`, 11 shape features, `_n_nodes`, `_n_edges` |
| `outputs/clustering/cluster_plot.png` | PCA 2D projection coloured by spectral (left) and K-Means (right) labels |
| `outputs/clustering/train_graphs_clustered.pt` | Train graphs with `g.cluster_label` (spectral) added |
| `outputs/clustering/test_graphs_clustered.pt` | Test graphs with `g.cluster_label` (spectral) added |
| `outputs/clustering/stratified_results.json` | Per-stratum node AP / edge AP for all 5 models (written by `stratified_eval.py`) |

### Using `cluster_label` in GNN training

```python
graphs = torch.load("outputs/clustering/train_graphs_clustered.pt")

# Filter to non-artifact graphs only (remove spectral cluster 1)
clean = [g for g in graphs if g.cluster_label != 1]

# Filter to high-branching stratum (use kmeans_label from CSV instead)
# Or filter by cluster_label == 0 for the majority normal population
branched = [g for g in graphs if g.cluster_label == 0]
```

`cluster_label == -1` means the graph was too small/degenerate to be clustered.

---

## Scripts

### `cluster.py` — main clustering pipeline

```bash
cd "/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn"
source ../.venv/bin/activate
python3 clustering/cluster.py
```

Options:
```bash
python3 clustering/cluster.py --n_clusters 3          # force k=3
python3 clustering/cluster.py --train <path> --test <path> --output <dir>
```

### `stratified_eval.py` — Path A + Path B analyses

Loads trained model checkpoints from `outputs/linkpred_clean_50ep/` and evaluates:

- **Path A (stratified):** Re-runs all 5 models on each K-Means stratum separately. Shows whether GNN node AP advantage over MLP concentrates in high-branching (topologically complex) cracks.
- **Path B (quality filter):** Re-runs all 5 models on graphs before and after removing the 332 degenerate spectral-cluster-1 artifacts. Confirms whether artifacts were depressing the topology-proof numbers.

```bash
python3 clustering/stratified_eval.py                          # both analyses
python3 clustering/stratified_eval.py --analysis stratified    # Path A only
python3 clustering/stratified_eval.py --analysis quality_filter # Path B only
python3 clustering/stratified_eval.py --models gine sage mlp   # subset of models
```

Output: `outputs/clustering/stratified_results.json`

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_DIR` | `outputs/clean_graphs/graphs/train_graphs.pt` | Train graph source |
| `TEST_DIR` | `outputs/clean_graphs/graphs/test_graphs.pt` | Test graph source |
| `OUTPUT_DIR` | `outputs/clustering` | Where to save results |
| `N_CLUSTERS` | `None` | Force fixed k, or `None` = auto-detect |
| `RANDOM_STATE` | `42` | RNG seed for reproducibility |
| `PRUNE_RATIO` | `0.10` | Spur pruning threshold (image-mode only) |
| `MIN_NODES` | `3` | Drop graphs with fewer than this many nodes |

---

## Key Design Decisions

**Why scale-invariant (ratio) features, not raw counts?**
Raw counts (`n_nodes`, `n_junctions`, `total_length`) scale with graph size. A large alligator crack and a small alligator crack have different raw counts but the same ratios. Using ratios ensures two cracks of the same *shape* but different *scale* cluster together. Every raw count is divided by `n_nodes` (or another count) to strip size out.

**Why k=2 when silhouette is only 0.36?**
k=2 is the highest silhouette in the sweep (0.36 vs 0.35 at k=3). It's selected as "least bad" — the data has weak cluster structure, and any k would give a low score. Forcing k=3 or higher would be overclaiming. The low silhouette is itself the finding: crack shapes are a continuum.

**Why `nearest_neighbors` affinity in spectral clustering?**
With 4364 data points and RobustScaler, the RBF kernel `exp(-γ‖xi-xj‖²)` collapses to near zero for distant graph pairs regardless of γ, making the affinity matrix nearly singular. The k-NN graph builds connectivity locally and doesn't require gamma tuning. See Fixes table for the full failure history.

**Why RobustScaler?**
Even with ratio features, crack graphs have skewed distributions (most graphs are small patches; some are large complex networks). RobustScaler (median/IQR) is less sensitive to the remaining skew than StandardScaler (mean/std).

**Why add `cluster_label` to `.pt` files?**
The GNN training pipeline (`compare.py`, `train.py`) loads `.pt` graph lists directly. Embedding `g.cluster_label` as a PyG attribute means zero changes to training code — you can filter, stratify, or use the label as a feature without touching the existing pipeline.

---

## Fixes Applied During Development

| Issue | Symptom | Fix |
|-------|---------|-----|
| Raw count features dominated by graph size | Spectral found large vs small graphs (corr(label, n_nodes) = 0.85+), not crack types | Replaced all raw counts with per-node ratios and densities |
| `branching_ratio` exploding to millions | When n_endpoints = 0, dividing by eps=1e-6 gave ratios of 10⁶+; mean branching_ratio = 496k in one cluster | Changed guard from `max(n_endpoints, 1e-6)` to `max(n_endpoints, 1)` |
| Extreme outlier (1003-node graph) dominated RBF kernel | Spectral: 4363 vs 1 graph, silhouette 0.621, 4% agreement | Winsorize at 99th percentile before scaling |
| StandardScaler sensitive to skewed distribution | Outlier inflated mean/std, distorted feature space | Replaced with RobustScaler (median/IQR) |
| Spectral RBF affinity collapse after scaling | exp(-γ‖xi-xj‖²) ≈ 0 for distant graph pairs; spectral decomposition degenerated (4363 vs 1 again) | Switched `affinity='rbf'` → `affinity='nearest_neighbors'` (k=15) |
| `filename` attribute not found in PyG graphs | All graphs named `pt_graph_0`, `pt_graph_1`, etc. | Fixed lookup: `getattr(g, "filename", None)` on original PyG object before NX conversion |
