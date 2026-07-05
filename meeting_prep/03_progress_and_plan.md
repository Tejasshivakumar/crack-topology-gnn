# Progress Report & Plan — vs the Professor's Mandate (corrected)

**Updated:** 2026-06-29 with the latest run results (Stage 1 ablation, corrected Stage 3 evaluation, heuristics, baselines, methodology fixes).

---

## 1. What the professor asked for

| Source | Ask |
|--------|-----|
| 24 Oct | Hybrid (Segmentation + GNN); **pure-CV vs hybrid** — does the GNN help segmentation? |
| 07 Nov | Crack → graph; **link prediction** (evolution) and/or **node classification** (severity/type); feasibility depends on dataset |
| Recent email | Follow minutes, understand & tune the model, **test on PaveDistress**, meet by mid-June |

---

## 2. Status against each thread (updated)

| Thread | Status | Result |
|--------|--------|--------|
| 3-stage pipeline (seg → graph → GNN) | Built, works | ✅ |
| Segmentation (DeepCrack) | Done | crack IoU **0.722** ✅ |
| **Pure-CV vs Hybrid (Stage 1 ablation)** | **Done** | ❗ **GNN does NOT help on crack_seg_clean** (plain 0.6375 ≥ hybrid 0.6299) |
| Link prediction (topology completion proxy) | Done | ⚠️ small GNN gain after correction (see below) |
| GNN baselines + heuristics + ablation | **Done** | ✅ closes the earlier credibility gap |
| Node classification (crack type) | Not started | ❌ PaveDistress fits it — best remaining upside |
| Test on PaveDistress | In progress | ⏳ |

---

## 3. Corrected Stage 3 results (the important change)

Switching node-task evaluation from global pooling to **per-graph AP** (the correct estimator) changed the picture:

| Model | Node AP (per-graph) | Edge AP |
|-------|--------------------:|--------:|
| mlp_pos (position only) | ~0.000 (task undefined w/o structural features) | 0.945 |
| MLP (full features, no message passing) | **0.670** | ~0.94 |
| GCN | ~0.600 | 0.912 |
| GAT | 0.609 | 0.931 |
| **SAGE** | **0.688** | 0.938 |
| GINE | 0.687 | 0.936 |
| Common Neighbors / Adamic-Adar / Resource Allocation | — | ~0.52 (near random) |
| Coordinates-only | — | **0.949** |

**Takeaways:**
- The earlier **+47%** MLP→GINE gap was a size-confounded global-pooling artifact. Corrected gap is **~+0.02** (MLP 0.670 → SAGE 0.688).
- Structural node features carry most of the signal; message passing adds a small, consistent increment.
- Classical heuristics are near-random — crack graphs are locally tree-like — so the learned models clear the floor easily.
- The edge task is dominated by spatial coordinates (coords-only 0.949 > GNN) → it's a control, not evidence of topology understanding.

---

## 4. Stage 1 segmentation ablation (crack_seg_clean, 50 ep, 448px)

| Model | Crack IoU | Crack Dice | Tol IoU | clDice | Prec | Rec |
|-------|----------:|-----------:|--------:|-------:|-----:|----:|
| **Plain ResNet34d (no GNN)** | **0.6375** | 0.7786 | 0.7335 | 0.7521 | 0.7272 | 0.8379 |
| Hybrid (ResNet34d + GNN) | 0.6299 | 0.7730 | 0.7279 | 0.7452 | 0.7140 | 0.8426 |
| Enhanced (scratch + GNN) | 0.6268 | 0.7706 | 0.7270 | **0.7540** | 0.7131 | 0.8382 |

**Finding:** the GNN bottleneck adds no segmentation benefit on this dataset; pretraining dominates. (On DeepCrack the hybrid reached 0.722.) Evaluation was also hardened: global micro-aggregation, precision/recall, clDice, and 2px boundary-tolerant metrics.

---

## 5. What's genuinely strong (lead with these)

1. Complete, working 3-stage pipeline.
2. **Rigorous, leakage-free evaluation** — found and fixed stale-feature leakage, easy-negative shortcuts, train/eval distribution shift, AUC-vs-AP checkpoint inconsistency, and the size-confounded pooling bug that inflated the early result.
3. Baselines, heuristics, and ablation now complete.
4. Physically-motivated **frontier-masking** protocol (targets the crack growth front).
5. Rigorous data cleaning (11,298 → 4,769) and the **CrackMark** annotation tool (solves the no-masks bottleneck).

---

## 6. Gaps, ranked (updated)

| # | Gap | Severity | Effort |
|---|-----|----------|--------|
| 1 | Headline thesis is now weak (small GNN gain) — needs reframe decision | High | Decision, then writing |
| 2 | Single seed — no error bars on the (now small) gap | High | ~4 h GPU |
| 3 | 50-epoch results not converged (200-epoch started, unfinished) | Medium | 2–4 h GPU |
| 4 | Pipeline not run truly end-to-end | High | ~2 h |
| 5 | PaveDistress crack-type classification unused | Med-High (best upside) | Medium |
| 6 | No unified written report | High (grade) | Days |

---

## 7. Prioritised plan

### Immediate (this week)
1. **Decide the thesis framing** with the professor (topology claim, framed honestly, vs pipeline/methodology/CrackMark lead).
2. **Multi-seed runs** (3 seeds, SAGE/GINE/MLP) → error bars on the +0.02 gap. This decides whether the GNN gain is even significant.
3. **End-to-end demo** — 10–20 raw images through all 3 stages.

### Next
4. **200-epoch canonical run** (started; finish it).
5. **PaveDistress** generalisation + (if approved) crack-type classification head — the most likely source of a clean positive result.

### Then
6. **Write the final report** consolidating corrected results, the methodology-fix story, clustering, and CrackMark.

---

## 8. Decisions requested at the meeting

1. **Thesis direction** given the small GNN gain (keep topology claim honestly, or lead with pipeline/methodology/tooling).
2. **Sign off the link-prediction reframe** (topology completion as Phase 2 deliverable).
3. **PaveDistress** — generalisation only, or add crack-type classification.

---

## 9. Publication outlook (honest)

The corrected results don't support a strong "GNNs understand crack topology" claim. A credible **domain-journal** paper (Automation in Construction / NDT & E International) is still realistic if framed as a complete, rigorously-evaluated pipeline with an honest baseline study and the CrackMark tooling — i.e. a systems + methodology contribution, not a novel-architecture-wins claim. The crack-type classification result, if positive, would meaningfully strengthen it.
