# Stage 3: GNN Link Prediction — Implementation

## The Claim

This stage proves a **GNN can understand the topology of a crack graph** — it can infer structural connectivity from the rest of the graph's shape, beyond what distance or non-learning heuristics achieve.

Method: **edge-masking link prediction**. Hide part of each graph's connectivity; ask models to recover it from the remaining structure. This is **structural inference, not temporal forecasting** — no time/load-resolved crack dataset exists, and documenting that data gap is itself a project contribution.

The proof requires a controlled comparison where structure is the only variable:
1. GNN beats non-learning structural heuristics (CN/AA/RA) → learning adds value over fixed topology rules.
2. Everything beats a coordinates-only baseline → the task genuinely needs topology, not just spatial proximity.
3. GNN still wins in structure-only mode (no position features) → the model reads topology, not handed-in features.

---

## Pipeline Position

```
Stage 1: HybridGraphUNet (segmentation) → binary crack mask (PNG)
                    ↓
Stage 2: image_to_graph → PyG Data objects, outputs/clean_graphs/
                    ↓
Stage 3: link_prediction (this stage) → topology understanding proof
```

**Data contract from Stage 2** — each graph provides:

| Field | Shape | Notes |
|-------|-------|-------|
| `x` | [N, 6] | cols 0–2: `x_norm, y_norm, thickness` (keep); cols 3–5: `degree, is_endpoint, is_junction` (**MUST be recomputed after edge split**) |
| `edge_index` | [2, 2E] | undirected, both directions stored |
| `edge_attr` | [2E, 7] | `path_length, euclidean_dist, tortuosity, angle_sym, avg_thickness, min_thickness, max_thickness` |
| `pos` | [N, 2] | `(x_pixel, y_pixel)` — used for hard-negative sampling |

---

## Directory Structure

```
link_prediction/
  __init__.py        — package exports
  splits.py          — transductive edge split + Fix 1 (feature recompute) + Fix 2 (hard negatives)
  heuristics.py      — Tier 1: CN, AA, RA structural scorers
  baselines.py       — Tier 1.5: coordinates-only baseline (critical sanity check)
  metrics.py         — per-graph-then-average: AUC, AP, Hits@K, MRR
  model.py           — 5 GNN encoders (MLP/GCN/SAGE/GINE/GAT) + edge/node predictor heads
  masking.py         — node masking utilities (endpoint/junction/random hiding)
  train.py           — joint training loop (edge task + node task); uses Fix 1
  evaluate.py        — headline table + ablation sweeps
  visualize.py       — per-graph prediction visualisation + training curves
  run.py             — CLI: train + eval a single model
  compare.py         — CLI: train all 5 encoders, print side-by-side comparison
  MODEL_RESEARCH.md  — architecture research and model selection justification
  IMPLEMENTATION.md  — this file
```

**Inputs:** `outputs/clean_graphs/graphs/{train,test}_graphs.pt`
**Outputs:** `outputs/linkpred/`

---

## The Two Critical Fixes

Both fixes live in `splits.py`. Getting them wrong makes results look good but mean nothing.

### Fix 1 — Feature recomputation after masking (prevents label leakage)

`degree`, `is_endpoint`, `is_junction` (x cols 3–5) are computed on the **full** graph during Stage 2. After hiding edges, a node that lost an edge still reports its old (higher) degree — a fingerprint of where the hidden edge was. The GNN can "predict" hidden edges by reading the stale degree instead of understanding topology.

**Fix:** after the edge split, recompute cols 3–5 from the **observed (message-passing) edges only**. Cols 0–2 are never touched.

```python
# splits.py
def recompute_structural_features(x, edge_index_observed, num_nodes):
    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, edge_index_observed[0], torch.ones(edge_index_observed.size(1)))
    x = x.clone()
    x[:, 3] = deg
    x[:, 4] = (deg == 1).float()   # is_endpoint
    x[:, 5] = (deg >= 3).float()   # is_junction
    return x
```

**Verification:** for any graph with hidden edges, assert that recomputed x[:,3:6] ≠ stored values. If identical, the recompute is not wired in (see sanity check §6).

This fix is applied in both `splits.py` (for heuristics/baselines/GNN evaluation) and `train.py` (during training).

### Fix 2 — Hard negative sampling (forces topology use, not distance)

Random negative pairs are almost always far apart. Because node features include position, a model scores "near = real, far = fake" and wins without topology. That defeats the proof.

**Fix:** negatives are **unconnected node pairs that are spatially near each other**, drawn from `pos`. The SAME negative set is used by every model (heuristics, baselines, GNNs) so comparisons are fair.

```python
# splits.py
def sample_hard_negatives(pos, existing_edges_set, num_neg, k_near=10):
    # for each anchor, consider its k nearest neighbours in pixel space
    # keep unconnected pairs; dedup, shuffle, take num_neg
    ...
```

If the coordinates-only baseline rivals the GNN, the negatives are still too easy — increase `k_near` or check that `pos` is being used.

---

## Split Protocol (`splits.py`)

Transductive setup (recommended, start here): for each graph, split undirected edges 80/10/10 into message-passing / val positives / test positives. Generate hard negatives for val/test. Recompute node features from observed edges.

```python
from link_prediction.splits import transductive_split, prepare_dataset

splits = prepare_dataset(graphs, num_val=0.10, num_test=0.10, k_near=10, seed=42)
# splits[i] is a dict: train_data, val_ei, val_labels, test_ei, test_labels, ...
```

`RandomLinkSplit` partitions the edges; `splits.py` replaces its random negatives with hard negatives and applies the feature recompute. Graphs with fewer than 4 undirected edges are skipped.

---

## Approaches (two tiers)

### Tier 1 — Structural heuristics (`heuristics.py`)
Non-learning, topology only. The floor the GNN must beat.

| Heuristic | Score for pair (u, v) |
|-----------|----------------------|
| **Common Neighbors (CN)** | `|Γ(u) ∩ Γ(v)|` |
| **Adamic-Adar (AA)** | `Σ_{w∈CN} 1/log|Γ(w)|` |
| **Resource Allocation (RA)** | `Σ_{w∈CN} 1/|Γ(w)|` |

Scored on the observed (message-passing) graph. Crack skeleton graphs are locally tree-like, so CN/AA/RA may be weak (few common neighbours between crack tips). That is fine — a low floor that the GNN clears convincingly is a clean result.

### Tier 1.5 — Coordinates-only baseline (`baselines.py`)
Score pairs by `1/(1 + Euclidean pixel distance)`, no message passing. **We want this to lose.** Its losing proves the task needs topology, not proximity. If it rivals the GNN, return to Fix 2.

### Tier 2a — GAE-style GNN encoder + decoder (`model.py`, `train.py`)
Node-based: encode all nodes via GNN (GraphSAGE/GCN/GINE/GAT), then score a pair by combining their embeddings with the MLP decoder. Fix 1 applied before every forward pass.

### Tier 2b — SEAL subgraph classifier (`seal.py` — future)
SEAL extracts the h-hop enclosing subgraph around each target link and classifies it with a GNN using the double-radius node labelling trick. Most direct "model uses local topology" architecture; strongest claim if it beats GAE. Planned; not yet implemented.

### Structure-only ablation
Run the GNN twice: once with full x, once with x[:,0:3] replaced by a constant (spatial/feature information stripped). If the GNN still beats the heuristics with no informative node features, the proof is cleanest: *the model reads topology, not features.*

---

## Model Architecture (`model.py`)

Five encoder architectures, all sharing the same `MLPEdgePredictor` and `MLPNodePredictor` heads.

| Encoder | Graph structure | Edge features | Use |
|---------|----------------|---------------|-----|
| `MLPEncoder` | No | No | No-graph lower bound |
| `GCNEncoder` | Yes | No | Bare topology, no edge attrs |
| `SAGEEncoder` | Yes | No | Inductive aggregation, no edge attrs |
| `GINEEncoder` | Yes | Yes (8-dim) | Expressive + edge features |
| `CrackGATEncoder` | Yes + attention | Yes (8-dim) | Heterogeneous crack junctions |

**CrackGATEncoder** — 3 layers, 4 attention heads, hidden=128, out_dim=64, residual connection between layers 2 and 3, LayerNorm after each layer.

**Angle encoding** — the raw `angle_sym` column (degrees, [0°,180°)) has a boundary discontinuity. It is encoded as `[sin(2θ), cos(2θ)]` inside the encoder (7 → 8 edge dims), mapping the range onto a smooth circle.

**`MLPEdgePredictor`** — scores pair (u,v) using `MLP([z_u ‖ z_v ‖ z_u⊙z_v])`. The Hadamard product captures non-linear interaction between endpoint embeddings, stronger than inner product alone.

**`MLPNodePredictor`** — binary classifier per node for the node masking task (does this node have a hidden neighbour?). Kept for joint training but not part of the headline link prediction comparison.

---

## Training (`train.py`)

Joint training of the encoder + edge predictor + node predictor. Fix 1 (feature recompute) is applied to every edge split before the forward pass.

```
Loss = node_loss + edge_loss_weight × edge_loss   (default weight = 0.5)
```

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | AdamW | weight_decay=1e-4 |
| LR | 5e-4 | with linear warmup |
| Warmup | 10 epochs | 0 → lr, then cosine to lr×0.01 |
| Gradient accumulation | 8 graphs/step | more stable estimates on tiny graphs |
| Checkpoint metric | 0.5×node_AUC + 0.5×edge_AUC | best of both tasks |
| Epochs (run.py) | 400 | |
| Epochs (compare.py) | 200 | |

Note: training still uses `RandomLinkSplit` negatives internally (fast). For the **final reported evaluation** — the headline table — `splits.py` hard negatives are used.

---

## Evaluation Metrics (`metrics.py`, `evaluate.py`)

Link prediction is **highly imbalanced** (few real edges, many possible non-edges).

| Metric | Report? | Why |
|--------|---------|-----|
| **AUC-ROC** | **Always** | Comparability anchor to prior papers; every LP paper reports it. Flatters under imbalance — don't lean conclusions on it, but its absence looks odd to reviewers. |
| **Average Precision (AP)** | **Yes — primary** | Honest under imbalance. Weight conclusions here. |
| **Hits@K** (K=20, 50) | Yes | "Of top-K predicted links, how many are real?" Used by Neo-GNNs / OGB; maps directly to our question. |
| **MRR** | Yes | Single ranking-quality summary. |
| **Accuracy** | **No** | Meaningless under imbalance (predict "no edge" everywhere → ~95%). |

**Critical rule — per-graph then average.** Graphs range from 3 to 1003 nodes. Pooling all candidate pairs lets giant graphs dominate and hides failure on small ones. Compute each metric **per graph, then average across graphs** (report mean ± std). `metrics.py` enforces this.

**Fairness rule.** Every approach is scored on the **same edge split and the same hard-negative set**. `splits.py` builds the split once; all approaches consume it.

### Headline results table

`evaluate.headline_table()` runs all approaches on the same splits and returns:

| Approach | AUC | AP | Hits@20 | Hits@50 | MRR |
|----------|-----|----|---------|---------|-----|
| Common Neighbors | | | | | |
| Adamic-Adar | | | | | |
| Resource Allocation | | | | | |
| Coordinates-only | | | | | |
| GNN (GAE) | | | | | |
| GNN (structure-only) | | | | | |

Story the table should tell: GNNs > heuristics; all > coordinates-only; SEAL ≥ GAE (once implemented).

---

## Mandatory Sanity Checks

1. **Coordinates-only must lose.** If it rivals the GNN → negatives too easy → fix `k_near` in `splits.py`.
2. **Leak check.** For any graph with hidden edges, confirm recomputed x[:,3:6] ≠ Stage 2 stored values. If identical → Fix 1 not wired in.
3. **Negative reuse.** Confirm identical negative sets (from `splits.py`) are used across all models for the same graph.
4. **Per-graph aggregation.** Confirm metrics are averaged per-graph, not pooled. `metrics.aggregate()` does this.
5. **Tiny-graph behaviour.** Graphs with < 5 edges yield unstable per-graph metrics. Report how they are handled (include with care, or report alongside a min-edge subset).

---

## Usage

**Train and evaluate a single model:**
```bash
cd crack-topology-gnn
python3 link_prediction/run.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --output-dir   outputs/linkpred \
    --model gat \
    --epochs 400
```

**Multi-model comparison (all 5 encoders):**
```bash
python3 link_prediction/compare.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --output-dir   outputs/comparison \
    --epochs 200
```

**Quick sanity check (subsample):**
```bash
python3 link_prediction/compare.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --num-train-graphs 500 \
    --epochs 50
```

**Key CLI flags (`run.py`):**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gat` | `mlp`, `gcn`, `sage`, `gine`, `gat` |
| `--epochs` | 400 | Training epochs |
| `--hidden` | 128 | Hidden dimension |
| `--out-dim` | 64 | Node embedding dimension |
| `--edge-loss-weight` | 0.5 | λ for edge task loss |
| `--node-mask-frac` | 0.20 | Fraction of endpoints hidden per graph |
| `--accum-steps` | 8 | Graphs per optimizer step |
| `--warmup-epochs` | 10 | Linear LR warmup |
| `--eval-only` | off | Load checkpoint, skip training |
| `--ablation` | off | Run mask_frac × node_type sweep |

**Output files:**
```
outputs/linkpred/
  best_model.pt          — best checkpoint (val combined AUC)
  metrics.json           — all test set metrics
  training_curves.png
  visualizations/        — per-graph prediction panels

outputs/comparison/
  comparison_results.json
  comparison_curves.png
  <model_name>/best_model.pt, metrics.json, training_curves.png
```

---

## References

| Paper | Use |
|-------|-----|
| Zhang & Chen, SEAL (NeurIPS 2018) | Subgraph LP; planned Tier 2b |
| Yun et al., Neo-GNNs (NeurIPS 2021) | CN/AA/RA baselines; optional model |
| Kipf & Welling, GAE (NIPS Workshop 2016) | GAE node-based LP baseline |
| Hamilton et al., GraphSAGE (NeurIPS 2017) | Encoder |
| Hu et al., OGB (NeurIPS 2020) | Hits@K / MRR evaluation protocol |
| Adamic & Adar (Social Networks 2003) | Adamic-Adar heuristic |
| Zhou, Lü, Zhang (Eur. Phys. J. B 2009) | Resource Allocation heuristic |
