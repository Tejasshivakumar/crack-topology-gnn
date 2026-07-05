# Q&A Defense Sheet — Anticipated Hard Questions (corrected results)

**Updated:** 2026-06-29. Each answer leads with the honest concession, then the defense. The corrected numbers are weaker than the early ones — owning that proactively is your strongest move.

---

## A. The core findings

**Q1. "Did the GNN actually help segmentation?"**
> Honest answer: no, not on `crack_seg_clean`. We ran the ablation — plain pretrained ResNet34d without the GNN bottleneck gets crack IoU 0.6375, slightly *above* the hybrid with GNN at 0.6299. Pretraining is the dominant factor at this data scale. The GNN bottleneck did help on the cleaner DeepCrack benchmark (0.722), but we can't claim a general segmentation benefit. This is a real, reportable result — negative ablations are still findings.

**Q2. "You reported a 47% GNN advantage earlier — what happened to it?"**
> That number was an artifact and we caught it. The original node-task AP used global pooling, which let large graphs dominate the score. When we switched to the methodologically correct per-graph averaging, the no-graph MLP rose to 0.670 and the best GNN (SAGE) to 0.688 — a +0.02 gain, not +47%. I'd rather show you the corrected number than defend an inflated one. Finding and fixing this is part of the contribution.

**Q3. "So is there any GNN benefit on the graph stage?"**
> A small, consistent one. SAGE/GINE (~0.688) beat the structure-aware MLP (0.670) on the crack-tip node task. More tellingly, classical link-prediction heuristics — Common Neighbors, Adamic-Adar, Resource Allocation — are near-random (~0.52) because crack graphs are locally tree-like and crack tips share no common neighbours. So the learned models clear the heuristic floor comfortably; message passing adds a little on top of the node features.

**Q4. "Then what is carrying the signal?"**
> The structural node features — degree, is_endpoint, is_junction. A position-only model can't even define the task; add structural features and you get 0.670; add message passing and you get 0.688. The decomposition is honest: features do most of the work, topology propagation adds a small increment.

---

## B. The reframe / thesis

**Q5. "If the GNN barely helps, what's the research contribution?"**
> Three things that survive the corrected numbers: (1) a leakage-free evaluation protocol — we found and fixed stale-feature leakage, easy-negative shortcuts, train/eval distribution shift, an AUC-vs-AP checkpoint inconsistency, and a size-confounded pooling bug; (2) a physically-motivated frontier-masking protocol that targets the crack growth front; (3) the CrackMark annotation tool that addresses the field's no-masks data bottleneck. The honest scientific finding — that structural features dominate and message passing adds a small gain — is itself worth reporting.

**Q6. "How does masking prove anything about crack evolution?"**
> The bridge: new crack segments form at existing tips. Our node task hides crack tips and asks which visible nodes lost a neighbour — the next-growth-direction signal. We never claim to predict evolution; we test the prerequisite (can a model read current topology). With no temporal dataset in existence, that prerequisite is the honest scope.

**Q7. "Is this enough for a 50%-weighted practicum?"**
> The engineering and rigor are strong; the headline scientific claim is modest. That's exactly why I want your steer today on whether to (a) keep the topology thesis framed honestly, or (b) lead with the pipeline + methodology + CrackMark tooling, with the GNN comparison as a supporting (and partly negative) result.

---

## C. Methodology rigor (your strongest ground)

**Q8. "Why Average Precision and per-graph averaging?"**
> Node task has ~11:1 imbalance, so AP is the honest metric over AUC. And graphs range from 3 to ~1000 nodes — global pooling lets the big ones dominate, which is exactly what inflated our early gap. Per-graph averaging weights each crack image equally. We also fixed checkpoint selection to use AP instead of AUC for consistency.

**Q9. "Did you compare against heuristics and simpler baselines?"**
> Yes, now. CN/AA/RA near-random (~0.52); coordinates-only strong on the edge task (0.95, above the GNN's 0.94); a position-only MLP can't do the node task at all. We also ran the masking-fraction × node-type ablation (endpoint/junction/random, 10–50%). Those are all done.

**Q10. "Results are single-seed and 50 epochs — and now the gap is small."**
> Correct, and that matters more now precisely because the gap is ~0.02. Multi-seed runs and the full 200-epoch run are the immediate next step to put error bars on whether SAGE/GINE's edge over the MLP is significant. I won't claim significance until that's done.

---

## D. Pipeline & data

**Q11. "Have you run the whole pipeline end-to-end?"**
> Not yet — Stage 3 used pre-made masks, not Stage 1's output. The end-to-end demo (10–20 raw images through all three stages) is planned. Concede cleanly.

**Q12. "PaveDistress has crack-type labels — are you using them?"**
> Not yet, and that's the most promising open thread — a graph-level crack-type classifier (longitudinal/transverse/map) from topology alone. Given the small node-task gain, this is where a clearly positive novel result is most likely. I'd like your steer on prioritising it.

**Q13. "What about the segmentation evaluation itself?"**
> We hardened it: global micro-aggregation of TP/FP/FN (the earlier per-batch averaging made IoU and Dice mathematically inconsistent), added precision/recall, clDice for connectivity, and 2px boundary-tolerant metrics matching the CRACK500 convention. On tolerant Dice we're ~0.84, competitive with published ResNet34-UNet baselines on related data.

---

## E. Recovery lines

- On the corrected numbers: *"I'd rather you see the honest figure — we caught the inflation ourselves."*
- On significance: *"That's single-seed; I'll have error bars before I claim it."*
- On thesis direction: *"That's exactly the steer I'm asking for today."*
- On anything unknown: *"Let me get you the exact number rather than guess."*
