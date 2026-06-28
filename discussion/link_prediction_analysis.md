# Link Prediction (Stage 3) — Strict Professor Analysis
**Date:** 2026-06-28  
**Files reviewed:** `model.py`, `masking.py`, `train.py`, `evaluate.py`, `splits.py`, `compare.py`, `run.py`, `visualize.py`, `heuristics.py`, `baselines.py`, `metrics.py`  
**Canonical results:** `outputs/linkpred_clean_50ep/` and `outputs/linkpred_clean_50ep/comparison_results.json`

---

## Section 1 — What Was Actually Done (The Honest Inventory)

Before criticising, here is what is genuinely present in the code:

| Component | Status |
|-----------|--------|
| 5 encoder architectures (MLP, GCN, SAGE, GINE, GAT) | Implemented and trained |
| Hard negative sampling (KD-tree, k_near=30) | Implemented, working |
| Feature recomputation after masking | Implemented, working |
| Separate backward passes for node/edge tasks | Implemented, justified |
| Dynamic pos_weight for class imbalance | Implemented |
| Gradient accumulation + clipping | Implemented |
| Warmup + cosine annealing LR schedule | Implemented |
| Structural heuristics: Common Neighbors, Adamic-Adar, Resource Allocation | Implemented in `heuristics.py` |
| Coordinates-only spatial baseline | Implemented in `baselines.py` |
| Position-only MLP baseline (`mlp_pos`) | Implemented in `compare.py` |
| Masking ablation (fraction sweep × node type sweep) | Implemented in `evaluate.py` |
| Per-graph-then-average metric aggregation for edge task | Implemented in `metrics.py` |
| Angle encoding (sin/cos circular for undirected edges) | Implemented, correct |
| Visualization with 3-panel layout | Implemented |

This is a lot of work. The **design is thoughtful**. The problems below are about what was not run and what is inconsistent.

---

## Section 2 — Critical Issues (Would Fail a Review)

---

### CRITICAL ISSUE 1: The Structural Heuristic Baselines Were Never Run

**What the code has:** `heuristics.py` fully implements three classical link prediction heuristics:
- **Common Neighbors (CN)** — two nodes likely connect if they share neighbours
- **Adamic-Adar (AA)** — same as CN but down-weights high-degree shared neighbours
- **Resource Allocation (RA)** — normalises by degree of shared neighbours

`evaluate.py` has `headline_table()` which runs all three against the GNN on the same hard-negative splits. 

**What actually happened:** The canonical `comparison_results.json` contains only `mlp, gcn, sage, gine, gat`. The heuristics were **never run**. There are no heuristic results anywhere in the outputs.

**Why this is a research failure:** In link prediction research, the standard evaluation ladder is:
1. Heuristics (the floor — can be beaten without machine learning)
2. MLP (no-graph baseline — can be beaten without graph structure)  
3. GNNs (the claim being made)

Without running heuristics, you cannot show that the GNN outperforms even basic graph analysis rules. A reviewer will ask: "Does GINE (node AP 0.501) beat Common Neighbors?" — and you cannot answer.

**Note on crack graphs specifically:** These are near-planar, locally tree-like graphs. Crack tips (the nodes you are predicting) often have **zero common neighbours** with any other node. CN/AA/RA may score near zero on this task, which would actually make the GNN victory very clean and convincing. But you need to run and show this.

**Fix:** Run `evaluate.py`'s `headline_table()` on the test set for GINE and include the results.

---

### CRITICAL ISSUE 2: The `mlp_pos` Baseline Was Never Run

**What the code has:** `compare.py` defines `EXTENDED_MODEL_NAMES = ['mlp_pos'] + MODEL_NAMES` as the default. The `mlp_pos` model takes position-only features (`x[:,2:]` zeroed — so only `x_norm` and `y_norm` remain) and passes them through the MLP architecture. It answers: "How much does just knowing where the node is (coordinates) contribute to the prediction?"

**What actually happened:** The canonical results only contain results for `mlp, gcn, sage, gine, gat`. `mlp_pos` was never run.

**Why this matters:** You already know the MLP (full features) gets node AP 0.341. If `mlp_pos` (coordinates only) gets something like node AP 0.28, that tells you structural features (degree, endpoint flag) add AP 0.06 on top of position. Then GINE gets 0.501 — adding another 0.16 from graph topology. That decomposition is the core research story. Without `mlp_pos`, you cannot cleanly separate position from structure from topology.

**Fix:** Run `compare.py --models mlp_pos` and add the result to the comparison table.

---

### CRITICAL ISSUE 3: The Ablation Study Was Never Executed

**What the code has:** `evaluate.py` has a complete `run_ablation_study()` function that sweeps:
- **Node masking fraction**: 10%, 20%, 30%, 40%, 50%
- **Node type masked**: `endpoint`, `junction`, `random`
- **Edge masking fraction**: 10%, 20%, 30%, 40%, 50%

`run.py` has a `--ablation` flag to trigger this. It would produce a table showing how performance changes as you hide more of the graph and hide different types of nodes.

**What actually happened:** There are zero ablation result files anywhere in `outputs/`. The `--ablation` flag was never passed. No ablation was run.

**Why this matters:** The ablation answers fundamental questions that a research paper must address:
- Why 20% masking? What happens at 10% or 30%?
- Why endpoint masking? What if you mask junction nodes (branch points) instead?
- Does the GNN advantage grow or shrink as more of the graph is hidden?

Without this, the choice of 20% endpoint masking looks arbitrary. A professor will ask: "Is your result specifically tuned to 20% or does it hold across masking levels?"

**Fix:** Run `python3 link_prediction/run.py --eval-only --ablation --output-dir outputs/linkpred_clean_50ep` on the best model (GINE checkpoint).

---

### CRITICAL ISSUE 4: Checkpoint Selection Uses the Wrong Metric

**In `train.py` line 269:**
```python
val_score = 0.5 * node_val_auc + 0.5 * edge_val_auc
```

The best model checkpoint is chosen by this combined **AUC-ROC** score.

**In `RESULTS.md` and all reporting:** The primary metric is stated to be **Average Precision (AP)**. The justification is that AUC-ROC overstates quality under class imbalance (11:1 ratio in the node task).

**The inconsistency:** The model that maximizes AUC-ROC on the validation set is NOT necessarily the model that maximizes AP on the test set. Under class imbalance, AUC and AP can disagree significantly — this is exactly why you chose AP over AUC in the first place.

**What this means:** Your "best" checkpoint was selected by a metric you have already argued is unreliable. The model you are reporting results for may not actually be the best model for your stated goal.

**Fix:** Change checkpoint selection to use `0.5 × node_val_AP + 0.5 × edge_val_AP`. This requires computing AP on the validation splits each epoch (slower but correct). Alternatively, checkpoint by node AP alone, since node task is primary.

---

### CRITICAL ISSUE 5: No Statistical Significance — Single Seed Per Model

Every model was trained exactly once. There are no:
- Error bars
- Confidence intervals
- Repeated runs with different random seeds

**Concrete problem:** The difference between GINE (node AP 0.501) and SAGE (node AP 0.489) is **0.012**. This is the difference between claiming "GINE is the best topology model" and "GINE and SAGE are essentially tied." Without at least 3 runs per model, you cannot make that distinction.

Even the larger gap between GINE (0.501) and MLP (0.341) should be confirmed over multiple seeds. If one training run got unlucky (bad initialization), the entire comparison could be misleading.

**Fix:** Run each model at minimum 3 times with different `--seed` values (e.g., 42, 7, 123). Report mean ± std. Even 3 runs per model produces defensible error bars.

---

## Section 3 — High Priority Issues (Methodological Inconsistencies)

---

### ISSUE 6: Node Task Uses Global Pooling, Edge Task Uses Per-Graph Average

**Node task evaluation (`evaluate.py` line 166-189):**
```python
all_probs.extend(probs)
all_labels.extend(lbls)
```
All predictions are pooled into one big list, then AUC/AP computed globally.

**Edge task evaluation (`evaluate_edge_task`, using `metrics.py`):**
```python
per_graph.append(graph_metrics(sc, lbl, hits_ks=()))
agg = aggregate(per_graph, ...)
```
Metrics computed per graph, then averaged.

**Why this is a problem:** The graphs have very different sizes (3 to ~1,000 nodes). With global pooling for the node task, a single large graph with 1,000 nodes has 50× more influence on the final AP than a small graph with 20 nodes. The node task metrics are dominated by the large, complex crack images. The edge task treats all graphs equally.

**For a fair comparison and a valid research paper, both tasks should use the same aggregation strategy.** Either both pool globally, or both average per graph. Per-graph averaging is the scientifically correct choice (it reports the expected performance on a random graph, not on a random node).

**Fix:** Implement per-graph node task evaluation (compute node AP per graph, then average across graphs) consistent with the edge task.

---

### ISSUE 7: Validation Node AUC Is Computed on Training Graphs

**In `train.py`:**
```python
def _node_auc(encoder, node_pred, graphs, device, mask_frac=0.20, seed=99):
    ...
    for i, g in enumerate(graphs):  # ← these are train_dataset graphs
        masked, node_labels, _, eval_mask = apply_node_mask(g, mask_frac=mask_frac, seed=seed + i)
```

This function is called with the **training graphs** to compute the validation node AUC used for checkpoint selection. The model being validated has ALREADY trained on these same graphs (just with different random masking each epoch).

**This is not a proper validation set.** The model has seen every graph in `node_graphs` during training. The "validation" AUC is an optimistic, in-distribution estimate. If the model overfits to specific graph structures in the training set, this validation metric will not detect it.

**Proper approach:** Hold out 10-15% of training graphs as a validation set (e.g., 370 of the 3,728 graphs) and never train on them. Use only those for validation.

**Note:** This is less urgent than Issues 1-5 because the model IS evaluated on the separate held-out test set (636 graphs) for final reporting. The harm is that early stopping may have saved a checkpoint that over-fits to training graphs.

---

### ISSUE 8: `run.py` Always Prints MRR = 0 (Dead Code Bug)

**In `run.py` line 188:**
```python
print(f'  MRR      : {metrics.get("edge_mrr", 0):.4f}')
```

`full_evaluation()` calls `evaluate_edge_task()` which returns only `edge_auc` and `edge_ap`. The key `"edge_mrr"` does not exist in the metrics dict. So this line always silently prints `MRR : 0.0000` regardless of the actual model quality. It is dead code that creates a false impression of an MRR value.

**Fix:** Remove this line (MRR was correctly dropped from the canonical metrics) or add a `None` check:
```python
mrr = metrics.get("edge_mrr")
if mrr is not None:
    print(f'  MRR      : {mrr:.4f}')
```

---

### ISSUE 9: Visualization Uses Random Negatives, Not Hard Negatives

**In `visualize.py` line 24:**
```python
_EDGE_SPLITTER = make_edge_splitter(num_val=0.0, num_test=0.20)
```
`make_edge_splitter` returns `RandomLinkSplit`, which selects random negative pairs.

**The reported metrics** use `transductive_split` with hard negatives (k_near=30 spatially-near non-edges).

**The inconsistency:** The visualizations you show do not reflect the evaluation conditions you report. The pictures show easy random negatives; the numbers reflect hard negatives. If a colleague looks at a visualization and then at the AP numbers, they will think the visualization is representative — but it is evaluating a strictly easier version of the task.

**Fix:** Replace `make_edge_splitter` in `visualize.py` with `transductive_split` from `splits.py`, so the visualization reflects the same hard-negative evaluation protocol.

---

## Section 4 — Medium Priority Issues (Training Quality)

---

### ISSUE 10: Early Stopping Never Triggered

With `patience=20` and `epochs=50`, early stopping could only trigger at epoch ≥ 30+20 = epoch 50 at the earliest. The results show every model hit its best score between epoch 43 and 50 — right at the end. Early stopping was functionally disabled.

**What this means:** The "patience=20" claim in the implementation docs is misleading. In practice, every model trained for exactly 50 epochs. This also confirms the earlier point — models are not converged.

**Fix:** Either run for 200 epochs (where early stopping can actually kick in) or document honestly that early stopping did not trigger at 50 epochs.

---

### ISSUE 11: pos_weight Capped at 5× Without Justification

**In `train.py` lines 188-190:**
```python
pw_full = _compute_node_pos_weight(train_dataset, node_mask_frac)
pw = min(pw_full, 5.0)
```

The full class imbalance is ~10.3x, but it is capped at 5x with the comment: "5× balances precision and recall better." There is no experiment supporting this. The cap changes the gradient signal from the node task in a significant way — with full pos_weight, the model is strongly pushed to detect all positives (high recall); capping to 5× means more false negatives are tolerated.

**Fix:** Run node AP at pos_weight=5x, 7x, and 10x (uncapped) and report the results. The optimal cap is an empirical question.

---

### ISSUE 12: Edge Splits Are Fixed, Node Masks Are Re-randomized Each Epoch

**Edge task:** `edge_splits` is computed **once** before training with fixed seeds:
```python
s = transductive_split(g, num_val=0.10, num_test=0.10, seed=i)
```
The same hidden edges and hard negatives are used in every training epoch.

**Node task:** `apply_node_mask(g, mask_frac=node_mask_frac)` is called with `seed=None` each epoch, so different nodes are hidden each epoch.

**The asymmetry:** The node task benefits from data augmentation (different masks each epoch forces generalization). The edge task uses the same positives and negatives every epoch — the model could in principle memorize which specific edges were hidden.

For a cleaner experiment, either both should be re-randomized each epoch or both should be fixed. The current setup introduces an asymmetry that is not documented and could subtly favor the node task.

---

## Section 5 — What Is Genuinely Well Done

To be balanced, these things are done correctly and will hold up under scrutiny:

1. **Hard negative sampling is correctly implemented.** The KD-tree approach with k_near=30 is state-of-the-art for link prediction. Crucially, the same hard negatives are used in training supervision, validation, and final test — making the comparison between models completely fair.

2. **Feature recomputation after masking is correct.** Recomputing degree, is_endpoint, is_junction from the observed (post-masking) edge set before message passing is the right fix. Many papers skip this and introduce silent leakage.

3. **Training uses hidden edges as supervision (not visible edges).** The fix from "train on visible, test on hidden" to "train on hidden edges from the test split of training graphs" is non-obvious and was done correctly. Without this, the train/eval distribution would be mismatched.

4. **Angle encoding is correct.** Using sin(2θ)/cos(2θ) for undirected crack segments is the mathematically correct circular encoding. A raw angle in [0°,180°) has a discontinuity at 0°/180° that would confuse attention mechanisms.

5. **Per-graph-then-average for edge metrics is correct.** Pooling across graphs of wildly different sizes (3 nodes to 1,000 nodes) would let giant graphs dominate. Per-graph averaging treats all images equally.

6. **The MLP baseline is a legitimate no-graph baseline.** It sees the same node features as the GNNs but performs no message passing. The +47% AP gap between GINE and MLP is therefore attributable to graph structure, not better features.

7. **Gradient accumulation and clipping are correctly applied.** `clip_grad_norm_(params, max_norm=1.0)` is called before each optimizer step. This prevents training instability on highly irregular crack graphs.

---

## Section 6 — Summary Scorecard (Stage 3)

| Component | Grade | Notes |
|-----------|-------|-------|
| Code quality & architecture design | A | Clean, modular, well-documented |
| Hard negative sampling | A | Correct KD-tree approach, consistent train/val/test |
| Feature recomputation fix | A | Correct and documented |
| Training supervision protocol | A | Hidden edges used, no distribution shift |
| Angle encoding | A | Mathematically correct circular encoding |
| Checkpoint selection metric | C | Uses AUC, but AP is stated as primary |
| Heuristic baselines (CN/AA/RA) | F | Implemented but never run |
| mlp_pos baseline | F | In default model list, never run |
| Ablation study | F | Fully implemented, never executed |
| Statistical significance | F | Single seed per model, no error bars |
| Node/edge evaluation consistency | C | Different aggregation strategies |
| Visualization protocol | C | Random negatives, inconsistent with metrics |
| run.py MRR output | D | Always prints 0, dead code |
| Early stopping | D | Never triggered at 50 epochs |

**Overall Stage 3 research readiness: 55% — the engineering is solid but the experimental evidence is incomplete.**

---

## Section 7 — Strict To-Do List for Stage 3

### MUST DO (without these, the research claim cannot be defended)

**TODO S3-1: Run Structural Heuristic Baselines**
```bash
# After loading models, call headline_table() from evaluate.py
# This runs CN, AA, RA, coordinates-only, and GNN on the same splits
```
- Shows whether GNN beats basic graph analysis rules (it should, given tree-like crack topology)
- Expected: CN/AA/RA will have very low AP on endpoint nodes (crack tips share no common neighbours)
- If heuristics beat GNN — the GNN is not learning topology, just structure.

**TODO S3-2: Run mlp_pos Baseline**
```bash
python3 link_prediction/compare.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --models mlp_pos \
    --epochs 200 \
    --output-dir outputs/linkpred_mlp_pos
```
- Establishes the floor: position coordinates alone, no structural features
- Decomposition of performance: mlp_pos → mlp → gnn = position → features → topology

**TODO S3-3: Execute the Ablation Study**
```bash
python3 link_prediction/run.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --output-dir   outputs/linkpred_clean_50ep \
    --model gine \
    --eval-only \
    --ablation
```
- Run for both GINE and SAGE (the two best models)
- Produces: mask_frac × node_type table (endpoint vs junction vs random) and edge fraction sweep
- Answers: Is 20% masking the right choice? Does the model work on junction nodes too?

**TODO S3-4: Fix Checkpoint Selection to Use AP**

In `train.py`, change the val_score computation (lines 265-269):
```python
# CURRENT (wrong — uses AUC):
node_val_auc = _node_auc(encoder, node_pred, node_graphs, device, ...)
edge_val_auc = _edge_auc(encoder, edge_pred, edge_splits, device)
val_score    = 0.5 * node_val_auc + 0.5 * edge_val_auc

# SHOULD BE (uses AP, consistent with primary metric):
node_val_ap = _node_ap(encoder, node_pred, node_graphs, device, ...)
edge_val_ap = _edge_ap(encoder, edge_pred, edge_splits, device)
val_score   = 0.5 * node_val_ap + 0.5 * edge_val_ap
```
- Implement `_node_ap` and `_edge_ap` analogous to `_node_auc`/`_edge_auc` but using `average_precision_score`
- Re-run training to get checkpoints selected by the right metric

**TODO S3-5: Run Multiple Seeds for Statistical Significance**
```bash
for seed in 42 7 123; do
    python3 link_prediction/compare.py \
        --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
        --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
        --models gine sage mlp \
        --epochs 200 \
        --seed $seed \
        --output-dir outputs/linkpred_seed$seed
done
```
- Run at minimum: GINE, SAGE, MLP (the three key comparison points) × 3 seeds
- Report mean ± std for node AP and edge AP
- This is what makes the +47% claim defensible in a paper

---

### SHOULD DO (raises research quality significantly)

**TODO S3-6: Standardize Node Task Evaluation to Per-Graph-Average**

In `evaluate.py`, replace the global pooling in `evaluate_node_task` with per-graph computation:
```python
# Instead of extending all_probs / all_labels across graphs,
# compute node_ap per graph, then average
per_graph_aps = []
for g in dataset:
    ...
    ap = average_precision_score(lbls, probs)
    per_graph_aps.append(ap)
node_ap = np.mean(per_graph_aps)
```
This makes node and edge metrics use the same aggregation strategy.

**TODO S3-7: Fix Visualization to Use Hard Negatives**

In `visualize.py`, replace `_EDGE_SPLITTER = make_edge_splitter(...)` with `transductive_split` from `splits.py`, so visualizations reflect the same hard-negative protocol as the reported numbers.

**TODO S3-8: Fix the MRR Dead Code in run.py**

Remove or guard the MRR print statement at `run.py` line 188:
```python
# Remove this line — edge_mrr is never in metrics dict:
print(f'  MRR      : {metrics.get("edge_mrr", 0):.4f}')
```

**TODO S3-9: Run Full 200-Epoch Training**
```bash
python3 link_prediction/compare.py \
    --train-graphs outputs/clean_graphs/graphs/train_graphs.pt \
    --test-graphs  outputs/clean_graphs/graphs/test_graphs.pt \
    --epochs 200 \
    --output-dir outputs/linkpred_200ep
```
- All models still converging at 50 epochs (GAT and GINE both best at epoch 49-50)
- Expected: GINE node AP → 0.55+, GAT may close gap with SAGE
- This is the number that goes in a published paper

---

### SHOULD DO IF TIME ALLOWS

**TODO S3-10: Proper Held-Out Validation Graph Set**

Split the 3,728 training graphs into 3,300 train / 428 validation BEFORE training:
```python
# In compare.py or run.py, split train_dataset:
val_graphs   = train_dataset[-428:]
train_graphs = train_dataset[:-428]
```
Then use `val_graphs` for checkpoint selection (never see them during training). This gives an honest early-stopping signal.

**TODO S3-11: Ablate pos_weight Cap**

Run GINE with pos_weight = 5x (current), 7x, and full ~10x (uncapped) and report node AP for each. The optimal cap is empirical, not theoretical.

**TODO S3-12: Fix Edge Split Randomization Asymmetry**

Either fix edge splits per epoch (add re-seeding), or document explicitly that edge splits are frozen. Ideally: resample hard negatives every K epochs (e.g., every 10) to prevent memorization.

---

## Section 8 — What Can Be Claimed Right Now vs What Needs More Work

### Can claim now:
- "GINE and SAGE outperform the no-graph MLP by +47%/+44% Average Precision on the node task under hard-negative evaluation."
- "The edge task is dominated by structural features — MLP leads — confirming it is a control result, not the main proof."
- "This gap survives spatially-hard negative evaluation (k_near=30) where position shortcuts are eliminated."

### Cannot claim yet without more experiments:
- "GNNs outperform classical link prediction heuristics" — heuristics never run
- "GINE is significantly better than SAGE" — 0.012 AP difference, single seed
- "20% masking is the right experimental setting" — ablation never run
- "The model generalizes across crack types" — ablation by node_type never run

---

*End of Stage 3 analysis. Next action: TODO S3-1 (run heuristics) and TODO S3-3 (run ablation) can be done in under 2 hours using existing checkpoints with `--eval-only`. These have the highest return on time investment.*
