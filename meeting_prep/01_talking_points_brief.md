# Practicum Meeting — Talking-Points Brief

**Project:** Road Crack Topology Understanding with GNNs (Phase 2)
**Meeting length:** 30 min · **Updated:** 2026-06-29 (with corrected results)

---

## One-line framing (say this first)

> "We built a complete image → graph → GNN pipeline, then did a rigorous round of baselines and methodology fixes. The honest finding is more nuanced than our first numbers suggested — and I want to walk you through it and get your steer on the direction."

Lead with rigour and honesty. The corrected results are weaker than the early ones, but the work that produced them is the strongest part of the project.

---

## Suggested 30-minute agenda

| Min | Segment | Goal |
|-----|---------|------|
| 0–3 | Recap of the mandate + scope | Align |
| 3–9 | What's done: 3-stage pipeline + Stage 1 ablation result | Show progress |
| 9–17 | Corrected Stage 3 findings + the methodology fixes that produced them | Land the honest claim |
| 17–24 | What the results mean for the thesis (3 decisions) | Get steer |
| 24–30 | Plan to final submission | Show direction |

---

## Status in 30 seconds

- **Stage 1 (Segmentation):** done. Hybrid (ResNet34d+GNN) = **0.722 crack IoU on DeepCrack**. On the larger `crack_seg_clean` set all variants land ~0.63. ✅
- **Stage 1 GNN ablation (NEW):** plain ResNet34d *without* GNN (**0.6375**) slightly *beats* the hybrid *with* GNN (0.6299) on `crack_seg_clean`. **The GNN bottleneck does not help segmentation on this dataset.**
- **Stage 2 (Image→Graph):** done — deterministic skeleton→graph, 6 node + 7 edge features. ✅
- **Stage 3 (GNN topology proof):** done, but the headline changed. Under corrected per-graph AP evaluation, the no-graph MLP scores **0.670** and the best GNN (SAGE) **0.688** — a small (+0.02) gain, not the +47% we first reported. Heuristics (CN/AA/RA) are near-random (~0.52). ✅ baselines + ablation now run.

---

## The numbers to memorize (corrected)

1. **Stage 1: crack IoU 0.722 on DeepCrack** — still solid. But **GNN ablation is negative on crack_seg_clean** (plain 0.6375 ≥ hybrid 0.6299).
2. **Stage 3 node task (per-graph AP): MLP 0.670 → SAGE 0.688** — message passing adds ~+0.02, not +47%. The earlier +47% was a size-confounded global-pooling artifact, now fixed.
3. **Heuristics near-random (CN 0.52); coordinates dominate the edge task (0.95 > GNN 0.94).** Crack graphs are locally tree-like — classical link prediction doesn't apply.

---

## What we can honestly claim now

- Structural node features (degree, endpoint/junction flags) carry most of the crack-topology signal.
- Message-passing GNNs give a **small but consistent** gain on the node (crack-tip) task over a structure-aware MLP.
- Classical link-prediction heuristics fail on crack graphs (near-random), and the edge task is dominated by spatial proximity — so it's a control, not evidence.
- The real strengths of the work: a **leakage-free evaluation protocol** (we found and fixed multiple bugs), a **physically-motivated frontier-masking protocol**, rigorous **data cleaning**, and the **CrackMark** annotation tool.

---

## Three decisions I need from the professor

1. **Given the small GNN gain — is "GNNs understand crack topology" still the thesis, or do we reframe?** Options: (a) keep it, framed honestly as a small-but-consistent gain + the heuristic-floor result; (b) pivot the headline to the *pipeline + methodology + CrackMark tooling* contribution. I'd like your view.

2. **Sign off the link-prediction reframe** (topology completion as the Phase 2 deliverable, since no temporal data exists). Still the right scope?

3. **PaveDistress direction** — test generalisation only, or add a crack-type classification head (longitudinal/transverse/map) using its labels? This is the one place a clearly *positive* novel result is still on the table.

> Note: the original "does the GNN help segmentation?" question is now **answered** (no, on crack_seg_clean) — so it's a finding to present, not an open decision.

---

## If asked "so did the GNN actually help?"

> "Two honest answers. In segmentation, no — a plain pretrained ResNet34d matches or beats the GNN-bottleneck version on crack_seg_clean. In the graph stage, a little — message passing adds about +0.02 AP on the crack-tip task over a structure-aware MLP, and it cleanly beats classical heuristics, which are near-random on these tree-like graphs. The bigger story is the methodology: our first +47% was inflated by size-confounded pooling, and we caught and corrected it."

---

## Weaknesses to disclose proactively

- Early +47% claim was wrong (global-pooling artifact) — corrected to ~+0.02; be the one who raises this.
- Results still at 50 epochs (200-epoch run was started, not finished).
- Pipeline not yet run truly end-to-end (Stage 3 used pre-made masks, not Stage 1 output).
- Single seed — no error bars yet on the small GNN gap (which now matters *more*, since the gap is small).
- No unified written report yet.
