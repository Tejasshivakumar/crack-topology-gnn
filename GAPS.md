# Practicum Research — Gap Analysis

This document records the gaps between what the professor originally asked for (per the meeting
minutes) and the current state of the project, plus the recommended actions before the mid-June
meeting.

---

## Source: What the Professor Asked For

### From the 24 Oct minutes (original mandate)

> *"Focus of the practicum is to use a Hybrid architecture (Segmentation + GNN) to validate
> **how well spatial features captured in GNN can help segmentation of road cracks.**"*

> *"...comparison of results between a **pure computer vision approach vs a hybrid approach**."*

### From the 07 Nov minutes (broadened direction)

> *"...transformed into a graph... used for e.g. **link-prediction** (predict how a crack evolves
> or whether I need to do some intervention in certain points) or **node classification** (identify
> the severity of different portions of the crack). The feasibility of this depends on the dataset
> you found."*

### From the most recent professor email (mid-June deadline)

> Follow the points in the minutes, understand the function and play with the parameters, test it on
> PaveDistress, then contact for a meeting **no later than mid-June**.

### The three research threads on the table

1. **Pure U-Net vs Hybrid U-Net+GNN** — does the GNN actually help segmentation?
2. **Link prediction** — predict how a crack evolves.
3. **Node classification** — severity / type of crack portions.

---

## Current Project State

| Thread | Status | Aligned? |
|---|---|---|
| 3-stage pipeline (segmentation → graph → GNN) | Built, works | ✅ Strong engineering |
| Segmentation on DeepCrack (crack IoU 0.722) | Done | ✅ Good result |
| Link prediction (Node AUC 0.83, Edge AUC 0.72) | Done | ⚠️ Proxy, not true evolution |
| Pure-CV vs Hybrid comparison | **Missing** | ❌ The original core question |
| Node classification (severity / type) | **Not started** | ❌ Unused; PaveDistress fits it |
| Test on PaveDistress | In progress (mask generation) | ✅ Exactly what was asked |

---

## Gap 1 — The Original Research Question Was Never Answered

### The problem
The whole practicum started with one question: **does adding the GNN help segmentation versus
plain computer vision?** This requires a **pure U-Net (no GNN) baseline** compared against the
Hybrid U-Net+GNN.

In `segmentation/RESULTS.md`, all three runs are **hybrid Graph-U-Net variants**:
- Run 1 — EnhancedGraphUNet (scratch-trained)
- Run 2 — HybridGraphUNet (pretrained ResNet34d)
- Run 3 — HybridGraphUNet on DeepCrack

The comparisons made are **scratch-vs-pretrained** and **dataset-vs-dataset** — never
**hybrid-vs-pure-CV**. There is no pure U-Net baseline anywhere in the project.

### The risk
If the professor asks *"so did the GNN actually help?"* at the meeting, there is currently no number
to point to. This is the foundational deliverable of the practicum and it is unanswered.

### Severity
**High** — this is the core research question, not an add-on.

### Recommended action
Train one **pure U-Net** (remove the GNN bottleneck from the architecture) on DeepCrack with
identical training config, then place the two crack-IoU numbers side by side. Low effort, high value.

| Model | Crack IoU | Mean IoU | Crack Dice |
|---|---|---|---|
| Pure U-Net (no GNN) | *(to fill)* | *(to fill)* | *(to fill)* |
| Hybrid U-Net + GNN | 0.722 | 0.854 | 0.834 |

---

## Gap 2 — Link Prediction Is a Proxy, Not the Asked-For Evolution Prediction

### The problem
The professor's link-prediction vision was to **"predict how a crack evolves."** This requires
**time-series data** — the same crack photographed over time — which does not exist in any public
dataset (including DeepCrack and PaveDistress).

The project pivoted to **topology completion**: randomly mask nodes/edges from a present-state graph
and train the model to reconstruct them, as a *proxy* for evolution understanding. This is documented
honestly in `link_prediction/IMPLEMENTATION.md` (Past → Present → Future framing).

### The risk
This is a **reframe** of what the professor asked for. It is defensible (no time-series data is
available, and the 07 Nov minutes note feasibility "depends on the dataset you found"), but it should
be **put in front of the professor explicitly for sign-off** rather than assumed accepted.

### Severity
**Medium** — defensible pivot, but needs explicit approval.

### Recommended action
Prepare a clear one-slide / one-paragraph justification for the meeting:
- Why true evolution prediction is impossible (no temporal dataset exists)
- Why topology completion is a valid proxy (reconstruction ability proves topology understanding)
- The explicit claim: *topology understanding is the prerequisite for evolution prediction.*

---

## Gap 3 — PaveDistress's Crack-Type Labels Are Going Unused

### The problem
PaveDistress has **crack-type category labels** — Longitudinal, Transverse, Map, Alligator, Sealed.
This is exactly the data the professor's **node / graph classification** suggestion (severity / type)
requires.

The current plan only uses PaveDistress for the **same masking task** as DeepCrack — discarding the
category labels that make PaveDistress distinct and valuable.

### The opportunity
Converting a PaveDistress crack to a graph and classifying **which type of crack it is** (from its
topology alone) would:
- Directly hit the professor's third research thread (node/graph classification)
- Be genuinely novel — topology-based crack typing
- Use a feature DeepCrack does not have (unlocking why PaveDistress matters)

### Severity
**Medium-High** — a missed opportunity that is also low-hanging fruit and a strong differentiator.

### Recommended action
Add a **graph-level classification head** that predicts crack type (Longitudinal / Transverse / Map)
from the graph topology. Even presenting this as a *plan* at the meeting shows the opportunity was
recognised.

---

## Gap 4 — No Baselines or Ablations Within the GNN

### The problem
The GNN results (Node AUC 0.83, Edge AUC 0.72) have no comparison points:
- No simpler baseline (MLP with no graph structure, GCN, GraphSAGE) to prove GAT is justified
- No ablation showing which design choices matter (edge features on/off, residual on/off, layer count,
  masking fraction, Hadamard product in the edge predictor)

Without these, the work is an implementation, not a research contribution.

### Severity
**Medium** — important for research credibility, less urgent than Gaps 1–3 for the meeting.

### Recommended action
- Add 1–2 baselines (a no-graph MLP and a GCN) for the same tasks.
- Produce a small ablation table (edge features on/off, residual on/off, 1/2/3 layers).
- See `link_prediction/MODEL_RESEARCH.md` for the GATv2 upgrade and structural-feature additions that
  would strengthen this further.

---

## Gap 5 — Node F1 = 0.25 Will Draw Objection

### The problem
The node-prediction F1 of 0.2518 is low. The current explanation (class imbalance, "trust AUC") is
correct but a reviewer will push back on a fixed 0.5 threshold.

### Severity
**Low** — easily addressed.

### Recommended action
Find the F1-maximising threshold on the validation set instead of hardcoding 0.5, and/or report
balanced accuracy alongside F1. Roughly 10 lines of code.

---

## Gap 6 — Single-Dataset Evaluation (Being Addressed)

### The problem
All training and testing to date is on DeepCrack only. Generalisation to a different dataset is
unproven.

### Status
**In progress** — PaveDistress mask generation is the active work that closes this gap. Running
Stage 2 + Stage 3 on PaveDistress graphs (no retraining of the GNN required) will demonstrate
cross-dataset generalisation, which is the core proof the professor is waiting for.

### Severity
**Low (actively being closed)**.

---

## Priority Order for the Mid-June Meeting

The mid-June deadline is essentially now. Recommended order:

| # | Action | Effort | Why |
|---|---|---|---|
| 1 | Run Stage 2 + 3 on PaveDistress | Medium | The literal ask — have numbers ready |
| 2 | Add a pure U-Net baseline (Gap 1) | Low | Answers the original research question |
| 3 | Frame the link-prediction pivot for sign-off (Gap 2) | Very Low | Avoids assuming acceptance |
| 4 | Propose crack-type classification on PaveDistress (Gap 3) | Low (as a plan) | Shows the opportunity was seen |
| 5 | Add GNN baselines + ablation (Gap 4) | Medium | Research credibility |
| 6 | Optimal threshold for node F1 (Gap 5) | Very Low | Removes a reviewer objection |

---

## Open Question to Resolve First

Were the Oct/Nov minutes **superseded** by a more recent email or conversation that explicitly
dropped the pure-CV-vs-hybrid comparison?

- **If yes** — Gap 1 disappears and the project is much better aligned than this document suggests.
- **If no** — Gap 1 stands as the highest-priority item, since it is the foundational deliverable.

This should be confirmed before finalising the meeting agenda.
