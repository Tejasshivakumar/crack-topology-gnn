/Users/tejasskamar/Downloads/CLUSTERING_IMPLEMENTATION.md# Clustering the Crack Dataset — Research Review & Recommended Approach
**Date:** 2026-06-20  
**Project:** Road Crack Topology Prediction (Phase 2)  
**Scope:** How to cluster 4,769 clean crack images/graphs to understand dataset composition, improve training, and evaluate model behaviour per crack type

---

## 1. Why Cluster?

Our cleaned dataset (4,769 images, 6 source datasets) contains road cracks that differ substantially in type, severity, and topology — but all images are treated identically during training. Clustering serves four concrete purposes in this research:

| Purpose | How clustering helps |
|---------|---------------------|
| **Dataset understanding** | Know whether we have balanced coverage of crack types (longitudinal, transverse, alligator/fatigue, block) |
| **Train/test stratification** | Ensure every split contains the same proportion of each crack type — prevents a model that only learns one type |
| **GNN evaluation per cluster** | Report node AP and edge AP broken down by crack type/severity, not just overall — gives richer research results |
| **Curriculum learning** | Order training by complexity (simple longitudinal → complex alligator) — may improve convergence |

Clustering is also the **only way to discover hidden subgroups** in a dataset assembled from 6 different sources with no crack-type labels attached.

---

## 2. What Do We Mean by "Crack Type"?

Road pavement cracks are classified in civil engineering standards (ASTM D6433, SHRP) into:

| Type | Description | Topology signature |
|------|-------------|-------------------|
| **Longitudinal** | Single line parallel to road axis | Low junction count, high endpoint ratio, low tortuosity |
| **Transverse** | Single line perpendicular to road axis | Same as longitudinal but different orientation |
| **Block** | Rectangular pattern from thermal cycling | Medium junction count, near-right-angle branching |
| **Alligator / Fatigue** | Dense polygonal network from load fatigue | High junction count, low endpoint ratio, high graph density, high tortuosity |
| **Edge crack** | Crumbling along pavement edge | Irregular, high tortuosity |

These types map directly onto **graph topology features** we already compute in Stage 2:
- Endpoint ratio → `is_endpoint` node feature
- Junction ratio → `is_junction` node feature  
- Tortuosity → `tortuosity` edge feature
- Graph density → `num_edges / num_nodes`

This means our Stage 2 graphs already contain the features needed to cluster by crack type — no additional feature engineering required.

---

## 3. Three Levels of Clustering — Research Landscape

### Level 1 — Handcrafted Graph Feature Clustering

**What it is:** Extract summary statistics from each crack graph (already computed in Stage 2), then apply a classical clustering algorithm.

**Features to extract per graph:**
```
num_nodes, num_edges,
endpoint_ratio   = nodes with degree==1 / total nodes
junction_ratio   = nodes with degree>=3 / total nodes
mean_tortuosity  = mean of edge tortuosity values
max_tortuosity
crack_density    = crack_pixels / (H × W)   [stored on Data object]
graph_density    = num_edges / num_nodes
mean_thickness   = mean of avg_thickness edge values
mean_degree      = 2 × num_edges / num_nodes
```

**Algorithms:**
- **K-means (k=4 or 5)** — forced to pre-specify number of clusters; interpretable centroids; fast
- **HDBSCAN** — density-based, discovers clusters of variable size and shape, handles outliers (sparse/noisy graphs become noise points rather than forced into a cluster). Preferred over DBSCAN because it is robust to varying density.
- **Spectral clustering** — graph-Laplacian-based; good when clusters are non-convex but expensive at scale

**Relevant papers:**
- Comparison of clustering algorithms for pavement crack segmentation (PMC 2022) — compares k-means, FCM (Fuzzy C-Means), spectral on crack pixel data
- Fuzzy C-Means for crack detection (Expert Systems with Applications 2024) — FCM on edge-enhanced crack images outperforms hard k-means for low-contrast cracks
- HDBSCAN vs k-means for infrastructure GNN embeddings (arXiv 2025) — HDBSCAN achieves Silhouette=0.626 vs k-means 0.315 on graph embedding clustering

**Why this is the right starting point for us:**  
Interpretable, fast, requires no trained model, directly maps to civil engineering crack taxonomy. The cluster centroids can be described in physical terms ("low-junction, low-tortuosity = longitudinal crack").

---

### Level 2 — GNN Embedding Clustering

**What it is:** Use our trained Stage 3 encoder (GAT or GINE) to produce a graph-level embedding via global pooling, then cluster those embeddings.

**How it works:**
```python
# After training the Stage 3 encoder
encoder.eval()
graph_embeddings = []
for g in dataset:
    x  = g.x.float().to(device)
    ei = g.edge_index.to(device)
    ea = g.edge_attr.float().to(device)
    z  = encoder(x, ei, ea)          # [N, out_dim] node embeddings
    g_emb = z.mean(dim=0)            # [out_dim]  global mean pooling
    graph_embeddings.append(g_emb.cpu().numpy())

embeddings = np.stack(graph_embeddings)   # [num_graphs, out_dim]
# → HDBSCAN / k-means / t-SNE visualisation
```

**The CrackGNN finding (BNAIC 2025):**  
The most directly relevant paper to our work. CrackGNN applied GNN encoding to crack graphs and visualised the embeddings with t-SNE:
> *"CrackGNN embeddings visualised through t-SNE reveal organisation into a spiral-shaped manifold where severity increases monotonically... High-severity clusters exhibit elevated junction counts, fractal dimensions, and tortuosity — signatures expected for advanced alligator cracking."*

This means a GNN trained on crack graphs will naturally organise its embedding space by crack severity without any explicit severity label — the topology carries the signal.

**Algorithms:**
- **HDBSCAN** on raw embeddings (64-dim) — handles outliers, no cluster count needed
- **t-SNE / UMAP** for 2D visualisation of the embedding manifold
- **K-means (k=3 to 6)** as baseline

**Relevant papers:**
- CrackGNN: Revealing a Crack Severity Manifold, BNAIC 2025 — closest work to ours
- Graph Clustering with Graph Neural Networks (DMoN), Tsitsulin et al., JMLR 2023 — theoretical framework for GNN-based clustering
- An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders, arXiv 2024 — benchmarks DINO/MAE/CLIP embeddings for clustering

**Why this is valuable for our research:**  
It answers "what does our GNN actually learn to separate?" and provides a natural evaluation: do high-severity graphs (many junctions, high density) cluster together? If yes, the encoder has learned meaningful crack topology representations.

---

### Level 3 — Image-Level Self-Supervised Embedding Clustering (DINOv2)

**What it is:** Extract features from crack images using a pretrained DINOv2 ViT (no fine-tuning), then cluster.

**Why DINOv2:**  
DINOv2 (Meta, 2023) produces dense visual features that generalise across domains without task-specific fine-tuning. Its patch-level attention maps resemble segmentation maps — they naturally attend to crack-like linear structures. Importantly, DINOv2 was trained on a curated dataset that used k-means clustering to ensure diversity, making its features well-suited for downstream clustering.

**How it works:**
```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model     = AutoModel.from_pretrained('facebook/dinov2-base')

embeddings = []
for img_path in image_paths:
    img     = Image.open(img_path).convert('RGB')
    inputs  = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token as global image embedding
    emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    embeddings.append(emb)

embeddings = np.stack(embeddings)   # [num_images, 768]
# → UMAP to 2D → HDBSCAN
```

**Relevant papers:**
- DINOv2: Learning Robust Visual Features without Supervision, Oquab et al., TMLR 2024 — DINO features cluster without labels, segmentation-quality attention maps
- Adaptive deep clustering integrating DINOv2 embeddings, graph attention, and bio-inspired optimisation, Scientific Reports 2025 — DINOv2 + GAT + clustering pipeline
- Contrastive Learning-Based Approach for Automated Road Damage Detection, 2025 — contrastive SSL for road infrastructure

**When to use Level 3:**  
Best for visual clustering — grouping images by texture, lighting, crack colour contrast. Less useful for topology clustering (a thin hairline crack and a wide alligator crack may look visually different but both are "bad" for our GNN). Use Level 3 to check dataset visual diversity, not crack structural diversity.

---

## 4. Recommended Approach for This Project

### Recommended pipeline: 3-stage clustering

```
Stage 2 Graphs (.pt files)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP A — Handcrafted Graph Feature Clustering          │
│  Features: endpoint_ratio, junction_ratio, tortuosity,  │
│  crack_density, graph_density, mean_thickness           │
│  Algorithm: HDBSCAN (primary) + k-means k=4 (baseline) │
│  Output: cluster label per graph, cluster centroids     │
│  Purpose: dataset understanding, stratified splits      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP B — GNN Embedding Clustering (post-training)      │
│  Source: trained GAT or GINE encoder                    │
│  Pooling: global mean of node embeddings                │
│  Algorithm: HDBSCAN + UMAP visualisation                │
│  Output: severity manifold plot, cluster APs            │
│  Purpose: show GNN learns meaningful topology structure │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP C — Cross-validate: do graph clusters ≈ GNN clusters? │
│  Compute alignment (ARI / NMI) between Step A and B     │
│  Expected: high junction_ratio graphs → high-severity   │
│  GNN cluster (CrackGNN finding)                         │
└─────────────────────────────────────────────────────────┘
```

### Why this order

1. **Step A first** — can be run immediately on existing Stage 2 `.pt` files, before any Stage 3 training. Gives dataset insight now. Takes minutes.
2. **Step B second** — requires a trained encoder. Answers the research question "does our GNN learn crack severity structure?" Publishable finding.
3. **Step C** — cross-validation. If Steps A and B produce similar clusters, it confirms the GNN has learned physically meaningful representations (not just pattern matching).

---

## 5. Algorithm Comparison

| Algorithm | Best for | Needs k? | Handles noise? | Speed | Use here |
|-----------|----------|----------|----------------|-------|----------|
| **K-means** | Baseline, interpretable centroids | ✅ Yes | ❌ No | Fast | Always run as baseline |
| **HDBSCAN** | Variable-density clusters, noisy graphs | ❌ No | ✅ Yes (outliers) | Medium | Primary for graph clustering |
| Spectral | Non-convex cluster shapes | ✅ Yes | ❌ No | Slow (O(n³)) | Skip at 4,769 graphs |
| Fuzzy C-Means | Soft cluster membership | ✅ Yes | Partial | Fast | Optional if overlap expected |
| **UMAP + HDBSCAN** | High-dim embeddings (GNN/DINO) | ❌ No | ✅ Yes | Fast | Step B visualisation |
| Agglomerative | Hierarchical structure | ✅ dendrog. | ❌ No | Medium | If hierarchy wanted |

**Bottom line:** HDBSCAN on UMAP-reduced embeddings is the current best practice for clustering neural network embeddings (graph or image) in infrastructure applications (confirmed by HDBSCAN infra paper 2025, DINOv2 curated data pipeline).

---

## 6. Features to Use — Step A Details

Extract these directly from the Stage 2 `.pt` graph files (already available):

```python
import torch
import numpy as np

def extract_graph_features(data):
    x         = data.x                     # [N, 6]
    ea        = data.edge_attr             # [E, 7]
    N         = x.size(0)
    E         = data.edge_index.size(1) // 2   # undirected

    endpoint_ratio = float((x[:, 4] == 1).sum()) / N   # col 4 = is_endpoint
    junction_ratio = float((x[:, 5] == 1).sum()) / N   # col 5 = is_junction
    mean_degree    = float(x[:, 3].mean())              # col 3 = degree
    mean_thickness = float(x[:, 2].mean())              # col 2 = thickness

    mean_tortuosity  = float(ea[:, 2].mean()) if E > 0 else 0.  # col 2 = tortuosity
    max_tortuosity   = float(ea[:, 2].max())  if E > 0 else 0.
    mean_path_length = float(ea[:, 0].mean()) if E > 0 else 0.  # col 0 = path_length

    graph_density    = E / N if N > 0 else 0.
    crack_density    = float(data.crack_density)

    return {
        'num_nodes':        N,
        'num_edges':        E,
        'endpoint_ratio':   endpoint_ratio,
        'junction_ratio':   junction_ratio,
        'mean_degree':      mean_degree,
        'mean_thickness':   mean_thickness,
        'mean_tortuosity':  mean_tortuosity,
        'max_tortuosity':   max_tortuosity,
        'mean_path_length': mean_path_length,
        'graph_density':    graph_density,
        'crack_density':    crack_density,
    }
```

**Feature → crack type mapping (expected):**

| Cluster | endpoint_ratio | junction_ratio | mean_tortuosity | crack_density | Likely crack type |
|---------|---------------|---------------|----------------|---------------|-----------------|
| High endpoint, low junction, low tortuosity | High | Low | Low | Low | Longitudinal / Transverse |
| High junction, low endpoint, medium tortuosity | Low | High | Medium | Medium | Block |
| Low endpoint, very high junction, high tortuosity | Very low | Very high | High | High | Alligator / Fatigue |
| Any, very low density | Any | Any | Any | Very low | Early-stage / hairline |

---

## 7. How Number of Clusters Relates to Crack Research

For ASTM-standard crack classification, **k=4** (longitudinal, transverse, block, alligator) is the natural choice for k-means. However, our dataset likely has:
- Different severity levels within each type (hairline vs wide)
- Mixed/transitional cracks

**Recommendation:**
- Run **k-means with k=3, 4, 5, 6** and select by silhouette score
- Run **HDBSCAN** with `min_cluster_size=30` and let the algorithm discover k
- Compare both — if HDBSCAN finds ~4 clusters, it validates the ASTM taxonomy on our data

**From CrackGNN (2025):** GNN embeddings organise into a **1D severity manifold** rather than discrete clusters — severity is a continuum, not 4 discrete bins. This suggests t-SNE/UMAP visualisation may be more informative than a hard cluster count for the GNN embedding step.

---

## 8. What This Gives Us for the Research Paper

Running this 3-step clustering pipeline produces:

1. **Dataset composition table** — "X% longitudinal-type, Y% alligator-type, Z% transitional" by cluster membership
2. **Cluster-stratified splits** — train/test sets balanced by crack type, not just source
3. **Per-cluster evaluation metrics** — Node AP and Edge AP broken down by crack type cluster → "the model performs better on X-type than Y-type cracks because..."
4. **Severity manifold figure** (Step B) — t-SNE of GNN embeddings coloured by cluster label — a direct analogue of the CrackGNN finding, adapted for our topology task
5. **Cluster alignment score** (Step C) — ARI/NMI between handcrafted and GNN clusters → proves GNN has learned physically meaningful structure

---

## 9. References

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Tsitsulin A. et al., "Graph Clustering with Graph Neural Networks" (DMoN), *JMLR* vol. 24, 2023 | Theoretical foundation for GNN-based graph clustering; DMoN loss function |
| 2 | CrackGNN: Revealing a Crack Severity Manifold, *BNAIC 2025* | Most directly relevant — GNN embeddings on crack graphs form a severity manifold; HDBSCAN + t-SNE approach |
| 3 | Oquab M. et al., "DINOv2: Learning Robust Visual Features without Supervision," *TMLR* 2024 | DINOv2 for image-level clustering; k-means for curated pretraining data |
| 4 | "Adaptive deep clustering integrating DINOv2 embeddings, graph attention, and bio-inspired optimisation," *Scientific Reports* 2025 | DINOv2 + GAT + clustering pipeline, closest to our Level 3 |
| 5 | "Comparison and Analysis of Several Clustering Algorithms for Pavement Crack Segmentation," *PMC / Electronics* 2022 | K-means vs FCM vs spectral on pavement crack pixel data |
| 6 | "Fuzzy C-Means clustering based selective edge enhancement for crack detection," *Expert Systems with Applications* 2024 | FCM outperforms k-means for low-contrast crack clustering |
| 7 | "Large-scale spatiotemporal pavement crack evaluation incorporating clustering analysis," *International Journal of Pavement Engineering* 2026 | GIS + DNN + cluster analysis pipeline for large-scale road surveys |
| 8 | "Topology-informed deep learning for pavement crack detection: Preserving consistent crack structure and connectivity," *Automation in Construction* 2025 | Persistent homology for crack connectivity — TDA angle on crack topology |
| 9 | "Multilayer GNN for Predictive Maintenance and Clustering in Power Grids," *arXiv* 2025 | HDBSCAN on GNN embeddings for infrastructure (Silhouette=0.626 vs k-means 0.315) |
| 10 | "An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders," *arXiv* 2024 | Benchmarks DINO/MAE/CLIP for downstream clustering without fine-tuning |
| 11 | "Feature-based morphological analysis of shape graph data," *arXiv* 2025 | Low-dimensional invariant features of branching graphs (road networks, neurons) for clustering — directly applicable to crack graphs |
| 12 | McInnes L. et al., "HDBSCAN: Hierarchical Density Based Clustering," *JOSS* 2017 | HDBSCAN algorithm — preferred over k-means for variable-density graph embedding clusters |
| 13 | McInnes L. et al., "UMAP: Uniform Manifold Approximation and Projection," *arXiv* 2018 | Dimensionality reduction before clustering — faster and more faithful than t-SNE at scale |

---

## 10. Implementation Plan (What to Build Next)

### Script: `data_cleaning/cluster.py`

**Inputs:**
- `outputs/clean_graphs_train/graphs/train_graphs.pt` (Stage 2 output on clean data)
- `outputs/clean_graphs_test/graphs/test_graphs.pt`

**Outputs:**
- `cluster_features.csv` — per-graph extracted features
- `cluster_labels_kmeans.csv` — k-means assignments (k=4)
- `cluster_labels_hdbscan.csv` — HDBSCAN assignments
- `cluster_summary.md` — cluster centroids mapped to crack type interpretation
- `umap_plot.png` — 2D UMAP of graph features coloured by cluster
- `stratified_split.csv` — train/val/test assignment balanced by cluster

**Dependencies (all already in venv):**
```
scikit-learn     ← k-means, silhouette score, ARI/NMI
hdbscan          ← may need: pip install hdbscan
umap-learn       ← may need: pip install umap-learn
matplotlib       ← plotting
pandas           ← CSV handling
torch            ← loading .pt files
```

**Prerequisite:** Run Stage 2 on clean data first to get the `.pt` graph files.

```bash
# First: Stage 2 on clean data
TRAIN_MASKS="data_cleaning/outputs/clean/train/masks"
TRAIN_IMGS="data_cleaning/outputs/clean/train/images"
TEST_MASKS="data_cleaning/outputs/clean/test/masks"
TEST_IMGS="data_cleaning/outputs/clean/test/images"

python3 image_to_graph/build_dataset.py \
    --mask-dir  "$TRAIN_MASKS" \
    --image-dir "$TRAIN_IMGS" \
    --split train \
    --output-dir outputs/clean_graphs_train \
    --no-vis

python3 image_to_graph/build_dataset.py \
    --mask-dir  "$TEST_MASKS" \
    --image-dir "$TEST_IMGS" \
    --split test \
    --output-dir outputs/clean_graphs_test \
    --no-vis

# Then: Clustering
python3 data_cleaning/cluster.py \
    --train-graphs outputs/clean_graphs_train/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs_test/graphs/test_graphs.pt \
    --output-dir   data_cleaning/outputs/clustering \
    --k 4
```
