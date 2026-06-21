# Phase 3 — GNN Link Prediction: Model Comparison Results
**Date:** 2026-06-19

---

## What We Did

1. **Image-to-Graph (Stage 2) on new dataset** — Converted the `crack_segmentation_dataset` (9,603 train / 1,695 test crack mask images) to PyG graph objects using skeletonization → sknw → node/edge feature extraction. Each node carries 6 features (position, thickness, degree, endpoint/junction flags); each edge carries 7 geometric features (path length, tortuosity, angle, thickness stats). 8,370 train graphs and 1,474 test graphs were produced (remainder skipped — empty masks).

2. **Multi-model link prediction comparison (Stage 3)** — Trained five encoder architectures on the same data with identical hyperparameters and evaluated on the held-out test set. The two tasks are: (1) **node task** — identify crack tip nodes whose neighbours were hidden (missing endpoint detection); (2) **edge task** — reconstruct hidden crack segments (link prediction). Quick-run settings: 500 subsampled training graphs, 50 epochs per model.

---

## Dataset Stats

| Split | Graphs | Avg Nodes | Avg Edges | Crack Density (mean) |
|-------|--------|-----------|-----------|----------------------|
| Train | 8,370  | 237.0     | 221.6     | 0.040                |
| Test  | 1,474  | 241.1     | 224.9     | 0.040                |

Graphs are ~10× larger than the previous DeepCrack dataset (24.7 nodes avg), making this a substantially harder and more realistic benchmark.

---

## Model Comparison Results
> **Quick-run only** — 500 of 8,370 training graphs, 50 epochs. Treat as architecture screening, not final numbers.

### Node Task — Missing Crack Tip Detection
*(Primary task. Class imbalance ~11:1 — AP and F1-opt are the meaningful metrics, not AUC-ROC)*

| Model  | Params | Node AUC | **Node AP** | F1 @ 0.5 | **F1 @ opt** | Bal. Acc | Best Epoch |
|--------|--------|----------|-------------|----------|--------------|----------|------------|
| MLP    | ~25k   | 0.819    | 0.282       | 0.302    | 0.308        | 0.620    | 25         |
| GCN    | ~30k   | 0.859    | 0.336       | 0.331    | 0.365        | 0.732    | 42         |
| SAGE   | ~35k   | 0.894    | 0.441       | 0.441    | 0.442        | 0.720    | 33         |
| GINE   | ~40k   | 0.995    | **0.945**   | 0.578    | **0.890**    | **0.946**| 29         |
| GAT    | 385k   | 0.829    | 0.273       | 0.324    | 0.335        | 0.686    | 49 *(still climbing)* |

### Edge Task — Missing Crack Segment Reconstruction
*(Secondary task. 1:1 pos:neg balance from RandomLinkSplit — AUC is reasonable here)*

| Model  | Edge AUC | **Edge AP** | **MRR**  | Hits@10 | Hits@20 |
|--------|----------|-------------|----------|---------|---------|
| MLP    | **0.867**| **0.862**   | **0.506**| **0.485**| **0.573**|
| GCN    | 0.779    | 0.816       | 0.469    | 0.474   | 0.572   |
| SAGE   | 0.759    | 0.785       | 0.418    | 0.464   | 0.557   |
| GINE   | 0.789    | 0.805       | 0.414    | 0.472   | 0.569   |
| GAT    | 0.834    | 0.855       | 0.514    | 0.480   | 0.576   |

---

## What the Results Tell Us

### 1. AUC-ROC is not the right primary metric for the node task
With ~11:1 class imbalance, AUC-ROC overstates model quality. MLP achieves node AUC=0.82 but node AP=0.28 — a massive gap showing the model ranks positives slightly above negatives but is nearly useless in practice. **Average Precision is the correct primary metric** for the node task.

### 2. GINE is the strongest node detector — but likely overfitting here
GINE achieves node AP=0.945 and F1-opt=0.890, far above all others. However, it was trained on only 500 graphs for 50 epochs and its node val-AUC hit 0.9988 by epoch 29 while edge AUC *dropped* (0.79→0.72). This is a clear overfitting signal on the small subsample. Its numbers must be validated on the full 8,370-graph training run before drawing conclusions.

### 3. MLP beats all GNNs on edge prediction — a red flag
The no-graph baseline achieves the best edge AP (0.862) and MRR (0.506). This reveals that **spatial node features (`x_norm`, `y_norm`) dominate edge prediction** — nearby nodes in space tend to be connected, and the MLP exploits this directly without any graph reasoning. The edge evaluation with random negatives (from RandomLinkSplit) is too easy because distant node pairs are trivially distinguishable by position alone. **Hard negative sampling** (negatives from spatially proximate but disconnected nodes) is needed for a meaningful edge evaluation.

### 4. Node–edge performance tradeoff
As encoder expressiveness increases (MLP → GCN → SAGE → GINE), node AP improves consistently but edge AP degrades. The joint loss (node + 0.5×edge) becomes dominated by the node task once models improve at it, effectively diverting capacity away from edge prediction.

### 5. GAT is under-trained
GAT (385k params, 73s/epoch) reached its best checkpoint at epoch 49 with the val score still rising. It needs significantly more epochs and training data to leverage its attention mechanism. At 50 epochs on 500 graphs it behaves like a poorly initialised GCN. Node AP=0.273 (lowest of all models) and edge AP=0.855 are not representative of its capability.

### 6. Hits@K is flat across models
All models achieve Hits@10 ≈ 0.47 and Hits@20 ≈ 0.57, with less than 3% spread. This metric is insensitive at this scale — edge ranking quality is similar regardless of architecture, again pointing to spatial features as the dominant signal.

---

## Key Limitations of This Run

- **500/8370 training graphs, 50 epochs** — screening only. All numbers should be treated as preliminary.
- **GAT not converged** — best epoch was the final epoch; more training is needed.
- **GINE result likely inflated** — near-perfect metrics on a subsample suggest memorisation.
- **Random negative sampling in edge task** — inflates all edge metrics and masks genuine differences between models. Hard negatives needed.
- **No test-time augmentation or threshold tuning** — F1@0.5 is pessimistic; F1@opt is the honest comparison.

---

## Recommended Next Steps

1. **Full training run** — all 8,370 train graphs, 200 epochs, all five models (`compare.py` without `--num-train-graphs`).
2. **Hard negative sampling** — replace RandomLinkSplit negatives with spatially proximate non-edges for a meaningful edge evaluation.
3. **Reduce GAT heads to 2** (`--heads 2`) to cut training time ~2× and make the full run tractable.
4. **Elevate AP as primary metric** in all reporting; demote AUC-ROC to secondary.
5. **Watch GINE on full data** — if node AP holds above 0.80 it is the best architecture; if it drops to SAGE range (~0.44), the small-run result was noise.
