
# Strict Pre-Submission Review — Crack Topology GNN
**Files reviewed:** literature_review.md, Report.md, segmentation_report.md, stage2_report.md, stage3_report.md  
**Reviewer role:** Professor reviewing for IEEE / ScienceDirect submission  
**Date:** July 2026

---

## OVERALL VERDICT

The technical content is strong and publication-ready. The experiments are rigorous, the methodology is sound, and the frontier masking result is genuinely novel. However, **4 factual errors** and **3 structural gaps** must be fixed before anyone can write the paper — because as it stands, two of the five files contain wrong numbers and all three technical reports are missing the most recent results.

---

## PART 1 — FACTUAL ERRORS (Fix First)

### ERROR 1 — Report.md has wrong GCN and GAT single-seed numbers

**Location:** Report.md, Section 9 — "Full 200-Epoch Training on Full Dataset"

Current table in Report.md:
| Encoder | Node AP |
|---------|---------|
| MLP | 0.662 |
| GCN | **0.701** ← WRONG |
| GraphSAGE | 0.700 |
| GINE | 0.739 |
| GAT | **0.723** ← WRONG |

Actual canonical single-seed values from `outputs/linkpred_200ep/*/metrics.json`:
| Encoder | Node AP (actual) |
|---------|---------|
| MLP | 0.6605 |
| GCN | **0.4948** |
| SAGE | 0.6932 |
| GINE | 0.7265 |
| GAT | **0.6310** |

GCN is off by +0.206 (over 40% relative error). GAT is off by +0.092. These are not rounding errors — they are wrong numbers that will produce contradictions in the paper if copied from Report.md.

**Fix:** Replace the Report.md section 9 table with the correct canonical values above.

---

### ERROR 2 — Report.md Section 9 also has inconsistent multi-seed section

**Location:** Report.md, Section 9 (multi-seed paragraph)

The text says:
> "GINE multi-seed node AP: 0.739 ± 0.004 — MLP multi-seed node AP: 0.662 ± 0.003"

And says:
> "Note: GAT and GCN multi-seed runs were added later and are still completing (running in tmux)."

Both statements are outdated. Multi-seed runs are complete for all 5 models. The correct complete table:

| Model | Seed 42 | Seed 100 | Seed 2024 | Mean ± Std |
|-------|---------|----------|-----------|------------|
| MLP | 0.6581 | 0.6656 | 0.6620 | 0.662 ± 0.003 |
| GCN | 0.6028 | 0.5936 | 0.6008 | 0.599 ± 0.004 |
| SAGE | 0.7031 | 0.6966 | 0.7003 | 0.700 ± 0.003 |
| GINE | 0.7343 | 0.7382 | 0.7441 | 0.739 ± 0.004 |
| GAT | 0.6317 | 0.6291 | 0.6345 | 0.632 ± 0.002 |

**Fix:** Update Report.md with the complete 5-model multi-seed table and remove the "still completing" note.

---

### ERROR 3 — Report.md Sections 14 and 15 still list M1–M5 as pending

**Location:** Report.md, Sections 14 and 15

Section 14 says M3 is "Pending", M4 is "Partially addressed", M5 is "Not started".  
Section 15 "What's In Progress" says GAT + GCN multi-seed is "Running in tmux".  
Section 15 "What's Pending" lists M3, M4, M5.

All of these are done. A paper writer reading this as their source of truth would incorrectly think experiments need to be run.

**Fix:** Update Sections 14 and 15 to reflect current completion state.

---

### ERROR 4 — stage3_report.md Section 11 says GCN/GAT multi-seed is "pending"

**Location:** stage3_report.md, Section 11, paragraph before Table 11.1

Text says:
> "Three models (MLP, SAGE, GINE) were trained with seeds {42, 100, 2024}... GCN and GAT multi-seed runs are pending (M1 reviewer comment)."

This is outdated. GCN and GAT are done.

**Fix:** Update stage3_report.md Section 11 with the complete 5-model multi-seed table.

---

## PART 2 — STRUCTURAL GAPS (Missing Results)

These are completed experiments whose results exist in JSON files but are not written into any of the 5 source files. A paper writer has no way to use these results without reading raw JSON.

### GAP 1 — End-to-End Evaluation (C3) not in any source file

**Data exists in:** `outputs/e2e_results.json`  
**Results:**
- Oracle path (GT mask → graph → GINE): Node AP = **0.655**
- Predicted path (HybridGraphUNet → graph → GINE): Node AP = **0.800**
- Gap (oracle − predicted): **−0.145** (predicted is higher — explained by GT mask jaggedness creating noisier graphs)
- Paired valid graphs: 60

**Why it matters for the paper:** This is the answer to reviewer C3 — does the pipeline actually work end-to-end? The counterintuitive result (predicted > oracle) needs 2–3 sentences of explanation: GT masks have jagged, rough edges that create artificially noisy skeletons with many short spur nodes; predicted masks from HybridGraphUNet are smoother and produce cleaner graphs.

**Missing from:** ALL 5 source files.

**Action:** Add a new section to stage3_report.md titled "End-to-End Pipeline Evaluation" with this result and the explanation.

---

### GAP 2 — Edge Feature Ablation (M5) not in any source file

**Data exists in:** `outputs/edge_ablation_results.json`  
**Results:**

| Ablation | Node AP | Drop from full |
|----------|---------|----------------|
| Full GINE | 0.7265 | — |
| No angle | 0.7061 | −0.020 |
| No tortuosity | 0.6663 | −0.060 |
| No geometry (length+dist) | 0.5228 | −0.204 |
| No thickness (avg+min+max) | 0.3793 | −0.347 |
| No edge features at all | 0.3761 | −0.350 |

**Key finding:** Dropping thickness alone (−0.347) is nearly equivalent to dropping ALL edge features (−0.350). Thickness is the dominant edge feature. This empirically validates why GINE outperforms GCN/SAGE — it's the thickness signal, not just the graph structure, that drives crack tip prediction.

**Missing from:** ALL 5 source files.

**Action:** Add a new section to stage3_report.md titled "Edge Feature Ablation" with this table and the interpretation.

---

### GAP 3 — Complete frontier results not fully integrated into stage3_report.md

**Data exists in:** `outputs/linkpred_200ep/frontier_results.json`

The stage3_report.md mentions frontier masking in the masking strategies section (Section 7.3) but does not have a dedicated results section showing the complete frontier table.

**Full frontier results:**
| Model | Standard Node AP | Frontier Node AP | Gap |
|-------|-----------------|-----------------|-----|
| MLP | 0.662 | 0.403 | −0.259 |
| GCN | 0.599 | 0.315 | −0.284 |
| SAGE | 0.700 | 0.550 | −0.150 |
| GINE | 0.739 | 0.614 | −0.125 |
| GAT | 0.632 | 0.504 | −0.128 |

GINE−MLP gap: standard = +0.077, frontier = +0.210. The gap widens 2.7× under the harder evaluation. This is the strongest result in the paper.

**Action:** Add a "Frontier Masking Results" section to stage3_report.md with this complete table.

---

## PART 3 — LITERATURE REVIEW GAPS

### LIT GAP 1 — Two placeholder DOIs (BLOCKER)

Papers 12 and 13 have `xxx` in their DOIs. A submission with placeholder DOIs gets desk-rejected.

- **Paper 12** (Wang et al. IJRMMS 2025): `10.1016/j.ijrmms.2025.105xxx`
- **Paper 13** (Chen et al. Materialia 2025): `10.1016/j.mtla.2025.102xxx`

**Action:** Find real DOIs or replace with verified alternatives.

---

### LIT GAP 2 — Paper 2 is a Hindawi journal (weak venue)

**Bhatti et al. — IJIS, Wiley 2023, DOI 10.1155/2023/8342104**

The `10.1155` DOI prefix belongs to Hindawi, not Wiley proper. Hindawi's journals were flagged by Clarivate and IEEE in 2023 for editorial issues. An IEEE reviewer will notice.

**Recommended replacement:**
> Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S. (2021). "A Comprehensive Study of Graph Neural Networks." **IEEE Transactions on Neural Networks and Learning Systems**, 32(1), 4–24. DOI: 10.1109/TNNLS.2020.2978386

This is IEEE Transactions (impact factor ~14), covers all five architectures you compare (GCN, SAGE, GAT, GIN, MLP), and specifically discusses expressiveness limits — a better fit than the current survey.

---

### LIT GAP 3 — 8 foundational papers not in literature_review.md

These papers are cited in the technical reports (inline references like "Kipf & Welling 2017") but are not formally documented in literature_review.md. Every paper cited in the final paper must have an entry here.

**Must add:**

| Paper | Venue | Year | Why needed |
|-------|-------|------|-----------|
| Ronneberger et al. — U-Net | Springer MICCAI | 2015 | Stage 1 encoder-decoder backbone |
| Kipf & Welling — GCN | ICLR | 2017 | Stage 3 GCN encoder |
| Hamilton et al. — GraphSAGE | NeurIPS | 2017 | Stage 3 SAGE encoder |
| Veličković et al. — GAT | ICLR | 2018 | Stage 3 GAT encoder |
| Xu et al. — GIN (WL theorem) | ICLR | 2019 | Foundation for GINE |
| Hu et al. — OGB / GINE | NeurIPS | 2020 | Stage 3 GINE encoder |
| Zhang et al. — CRACK500 | IEEE ICIP | 2016 | Dataset (one of 11 sources) |
| Al-Huda et al. — EfficientCrackNet | IEEE Xplore | 2024 | Segmentation SOTA comparison |

Note: GIN (Xu et al. ICLR 2019) is already in literature_review.md as Paper 1. The others are missing.

---

### LIT GAP 4 — Missing SHM GNN context (Recommended)

The paper has no structural health monitoring + GNN citation. A reviewer coming from the SHM community will notice that the motivation section never cites any SHM GNN prior work.

**Recommended addition:**
> Chen, J. et al. (2023). "A Novel Structural Damage Detection Method via Multisensor Spatial–Temporal Graph-Based Features and Deep Graph Convolutional Network." **IEEE Transactions on Instrumentation and Measurement**, 72. DOI: 10.1109/TIM.2023.3237648. [IEEE Xplore: doc/10021664]

This paper applies GCN to structural damage detection, uses vibration signal graphs, and directly motivates graph-based approaches to infrastructure inspection — exactly the application context of your Stage 3.

---

### LIT GAP 5 — Missing SoftClDice citation in segmentation_report.md

**Location:** segmentation_report.md, Section 4.5

The report describes SoftClDice loss and credits "Shit et al. (CVPR 2021)" inline, but this paper is not in literature_review.md.

**Must add:**
> Shit, S. et al. (2021). "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." **CVPR 2021**. DOI: 10.1109/CVPR46437.2021.01629

This is IEEE CVPR (top-tier CV conference). If you use SoftClDice in training, you must cite it.

---

## PART 4 — WHAT IS STRONG (Do Not Change)

These sections are genuinely excellent and paper-ready:

| File | Strong Section | Why |
|------|----------------|-----|
| stage3_report.md | Sections 6–9 (masking, training protocol) | Hard negative sampling, feature recomputation, pos_weight capping — all correctly implemented and documented |
| stage3_report.md | Section 11.3–11.5 (gap analysis) | The 24.8× gap statistic and zero-overlap claim are rigorous and compelling |
| stage2_report.md | Sections 4.4–4.8 (spur pruning, feature extraction) | Distance transform for thickness is the right choice; design rationale is convincing |
| stage2_report.md | Section 4.2 (skeletonization) | Properties of Zhang-Suen thinning correctly described |
| segmentation_report.md | Sections 4, 5, 6 (loss, augmentation, training) | Focal+Dice loss, differential LR, CLAHE — all motivated and correct |
| segmentation_report.md | Sections 9–10 (ablation + findings) | Three-model comparison cleanly decomposes pretraining vs. GNN bottleneck effect |
| literature_review.md | All 17 paper entries | Why-it-fits / gap / how-we-fill-it structure is clear and consistent |
| literature_review.md | Coverage map | Shows clearly that no prior paper traverses all 3 stages |

---

## PART 5 — COMPLETE ACTION CHECKLIST

Ordered by priority. Complete in this order.

### MUST DO (Blockers)

- [ ] **Fix Report.md GCN node AP**: change 0.701 → 0.495 (canonical) in Section 9 table
- [ ] **Fix Report.md GAT node AP**: change 0.723 → 0.631 in Section 9 table
- [ ] **Update Report.md multi-seed section**: add GCN (0.599±0.004) and GAT (0.632±0.002) rows; remove "still completing" note
- [ ] **Update Report.md Sections 14/15**: change all M1–M5 status to Complete; remove "What's Pending" items
- [ ] **Update stage3_report.md Section 11**: add GCN and GAT multi-seed rows; update M1 status
- [ ] **Add E2E results section to stage3_report.md**: Oracle AP=0.655, Predicted AP=0.800, Gap=−0.145, n=60
- [ ] **Add edge ablation section to stage3_report.md**: full table + "thickness is dominant" finding
- [ ] **Add frontier results section to stage3_report.md**: complete 5-model table
- [ ] **Fix literature_review.md Paper 12 DOI**: replace `105xxx` with real DOI or substitute paper
- [ ] **Fix literature_review.md Paper 13 DOI**: replace `102xxx` with real DOI or substitute paper
- [ ] **Replace Paper 2 (Bhatti/Hindawi)** with Wu et al. IEEE TNNLS 2021 (DOI: 10.1109/TNNLS.2020.2978386)

### SHOULD DO (Strengthen paper)

- [ ] **Add 7 missing foundational papers to literature_review.md**: U-Net, GCN, SAGE, GAT, GINE/OGB, CRACK500, EfficientCrackNet
- [ ] **Add SoftClDice citation**: Shit et al. CVPR 2021 — used in segmentation training
- [ ] **Add SHM GNN context paper**: Chen et al. IEEE TIM 2023

### NICE TO HAVE

- [ ] Add PyTorch Geometric citation (Fey & Lenssen, ICLR Workshop 2019)
- [ ] Note GINE convergence finding (still improving at epoch 198–199) explicitly as a limitation that opens future work

---

## PART 6 — TOTAL PAPER COVERAGE MAP

After fixing the above, these 5 files together give a paper writer everything they need:

| Paper Section | Source File | Status |
|--------------|-------------|--------|
| Abstract | All results from stage3_report | ⚠️ Numbers exist but must use correct canonical values |
| Introduction + Motivation | Report.md Section 1 | ✅ Ready |
| Related Work | literature_review.md | ⚠️ Fix DOIs + add missing papers |
| Dataset | segmentation_report.md Section 2 | ✅ Ready |
| Stage 1 Architecture | segmentation_report.md Section 3 | ✅ Ready |
| Stage 1 Results | segmentation_report.md Section 8 | ✅ Ready |
| Stage 2 Pipeline | stage2_report.md Sections 3–8 | ✅ Ready |
| Stage 3 Architecture | stage3_report.md Sections 4–6 | ✅ Ready |
| Stage 3 Training | stage3_report.md Sections 7–8 | ✅ Ready |
| Results — Single seed | stage3_report.md Section 10 | ✅ Ready (use 200ep values, not Report.md) |
| Results — Multi-seed | stage3_report.md Section 11 | ⚠️ Missing GCN/GAT rows |
| Results — Frontier | stage3_report.md | ⚠️ No dedicated results section |
| Results — E2E | — | ❌ Not written anywhere |
| Results — Edge ablation | — | ❌ Not written anywhere |
| Results — Clustering | Report.md Section 11 | ✅ Ready |
| Results — Masking ablation | Report.md + stage3 | ✅ Data exists, needs integration |
| Discussion | Report.md Section 12 + stage3 | ⚠️ Needs E2E and ablation discussion |
| Conclusion | Report.md | ✅ Ready |
| References | literature_review.md | ⚠️ Fix DOIs + add 7 missing papers |
