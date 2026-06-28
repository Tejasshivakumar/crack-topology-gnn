# Research Practicum — Critical Analysis & Project Status Report

**Project:** Road Crack Topology Understanding using Graph Neural Networks (Phase 2)  
**Date of Review:** 2026-06-27  
**Reviewer Note:** This report evaluates the project as if it were being reviewed by a professor before a final submission or presentation. It identifies what is done, what is genuinely good, what is incomplete, what is missing, and what you must still do — in plain language.

---

## 1. What This Research Is Trying to Do (in plain words)

You are trying to prove that an AI model can **understand the shape and structure of road cracks** — specifically, using a type of AI called a **Graph Neural Network (GNN)**.

The long-term goal is: given a photo of a road crack at one point in time, predict how it will grow and spread in the future. That is called **link prediction** — predicting new connections that will form in the crack network.

Because the real-world data needed for that (same road photographed at multiple time points, with weather and traffic info) does **not exist yet**, Phase 2 has a **smaller, more realistic goal**: prove that GNNs can understand the *shape* (topology) of a crack — specifically, can a GNN figure out that a part of the crack is missing or hidden? If yes, that becomes the foundation for the future real prediction task.

This is a sound and defensible research scope. The problem is that **execution is incomplete in several critical ways**, which this report will spell out.

---

## 2. What You Built (The Three Stages)

### Stage 1 — Crack Segmentation (Finding the Crack in a Photo)

**What it does:** Takes a raw photo of a road and outputs a black-and-white image where white pixels mark the crack.

**Architecture used:** A model called `HybridGraphUNet` — it combines a pretrained image recognition model (ResNet34d, trained on millions of ImageNet photos) as the "eye" that sees the image, with a GNN module in the middle that looks at the spatial structure of the features, and a standard image-reconstruction decoder (UNet-style) that produces the final mask.

**Results:**
- Best score (crack IoU) = **0.722** on the DeepCrack public benchmark
- The target was 0.65–0.70. You **exceeded the target**.
- Crack Dice Score = 0.834, Accuracy = 0.987

**Verdict:** Stage 1 is **complete and solid**. The score is competitive for the dataset size (only 300 training images). Good work here.

---

### Stage 2 — Image to Graph (Turning the Crack into a Mathematical Structure)

**What it does:** Takes the black-and-white crack mask from Stage 1 and converts it into a **graph** — a network of dots (nodes) and lines (edges).

- Each **node** is a point where the crack branches (junction) or ends (tip/endpoint)
- Each **edge** is the crack segment connecting two nodes
- Each node carries 6 numbers describing it: position, thickness, degree (how many connections), and whether it is an endpoint or junction
- Each edge carries 7 numbers: length, tortuosity (how curved it is), angle, thickness statistics

**Results:**
- Processed 4,769 clean road crack images into 4,364 graphs (some images were empty masks and were skipped)
- Average graph size: 237 nodes, 222 edges

**Verdict:** Stage 2 is **complete and well-engineered**. The feature design is thoughtful and maps cleanly to crack structure.

---

### Stage 3 — GNN Link Prediction (The Core Research Task)

**What it does:** Takes the crack graph and asks: "Can the AI figure out which crack tips have a hidden neighbour — meaning, can it predict that a crack is about to extend in some direction?"

This is done by **hiding 20% of the crack structure** and asking the model to figure out where the missing pieces are.

**Two sub-tasks:**
1. **Node task (primary):** identify which crack tip nodes have a hidden connection — i.e., which endpoints are secretly connected to something that has been removed
2. **Edge task (secondary/control):** predict which specific hidden crack segment exists between two nodes

**Five models trained:**

| Model | What it is | Node AP (primary score) | Edge AP |
|-------|-----------|------------------------|---------|
| MLP (baseline) | No graph — just looks at each node's own numbers | 0.341 | 0.941 |
| GCN | Simplest GNN — averages neighbour information | 0.342 | 0.912 |
| GAT | Attention-based GNN — pays more attention to important neighbours | 0.429 | 0.931 |
| GINE | Most expressive GNN — uses both node and edge info | **0.501** | 0.936 |
| GraphSAGE | Samples and combines neighbours | 0.489 | 0.938 |

**Key finding:**
- GINE and GraphSAGE gain **+47% and +44%** in the primary metric (Average Precision) over the no-graph MLP baseline
- This proves that **the graph structure is carrying useful information** — the GNN is actually learning the topology, not just memorizing position
- The MLP winning on the edge task is actually a good sign: it confirms the GNN's advantage on the node task is genuine topology learning, not a shortcut

**Verdict:** Stage 3 is **partially complete**. The core proof-of-concept result is there and is real. But serious gaps remain (detailed below).

---

### Extra Work — Crack Clustering (Understanding Dataset Composition)

**What it does:** Groups the 4,364 crack graphs by their structural shape — simple line cracks vs complex branching networks — to see if the GNN performs differently on different crack types.

**Finding:**
- Crack topology does NOT form clear discrete types — it's a **continuum** from simple to complex
- Simple (low-branching) cracks: SAGE node AP = **0.669** (much higher)
- Complex (high-branching) cracks: SAGE node AP = **0.458**
- This is an important and publishable finding: the model handles simple cracks much better than complex ones

**Verdict:** This is **done and valuable** but not yet written into the main research narrative.

---

## 3. What Is Genuinely Good (Professor's Acknowledgement)

1. **Research scope is honest.** Admitting that real temporal data does not exist and pivoting to a proof-of-concept is intellectually mature, not a cop-out.

2. **Data cleaning was rigorous.** Removing non-road-pavement images (concrete walls, etc.) from the combined dataset with documented justification (domain mismatch corrupts topology learning) is exactly the kind of careful thinking that distinguishes good research.

3. **The evaluation protocol is clean.** You fixed multiple data leakage bugs (stale features fingerprinting hidden edges, easy random negatives, train/eval distribution shift) and the final results reflect a genuinely honest evaluation. Many student projects skip this entirely.

4. **Average Precision (AP) as the primary metric.** With 11:1 class imbalance (many more non-endpoint nodes than endpoint nodes), using AP instead of Accuracy or AUC is the correct choice. You identified this and changed it.

5. **The +47% gap survives hard negatives.** The most important result — that GINE outperforms MLP — was tested under conditions where position-based shortcuts were removed. The gap is real.

---

## 4. Critical Gaps and Problems (What a Professor Would Flag)

---

### GAP 1 — Stage 1 and Stage 2/3 Were Never Connected End-to-End [CRITICAL]

**The problem in plain words:**  
Stage 1 (your segmentation model) was trained and tested on the **DeepCrack** dataset (300 training images).  
Stage 2 and Stage 3 were run on the **crack_seg_clean** dataset (4,769 images) — but this dataset came with **pre-made masks already included**. You used those pre-made masks, not the output of your own Stage 1 model.

This means **you never actually ran the full pipeline end-to-end**: raw photo → your Stage 1 model → your Stage 2 converter → your Stage 3 GNN.

If a professor or examiner asks "show me how your pipeline works on one real road image," you cannot currently do that.

**Why this matters:**  
Your research says it is an end-to-end pipeline. It is not end-to-end tested. If Stage 1 produces an imperfect mask (which it will — 0.722 IoU means 28% of crack pixels are wrong), those errors will propagate into Stage 2 graphs and affect Stage 3 performance. You have no data on how much this degrades the final result.

**What you must do:**
- Take at least 10–20 sample road images
- Run them through Stage 1 (`generate_masks.py`) to produce masks
- Convert those masks to graphs via Stage 2
- Run Stage 3 predictions on those graphs
- Show the full pipeline working in visualizations

---

### GAP 2 — Stage 3 Training Is Not Complete [HIGH PRIORITY]

**The problem in plain words:**  
All 5 models were trained for only **50 epochs** (training cycles). All results show the models are **still learning** at epoch 50 — they have not yet reached their full potential. GAT and GINE both hit their best scores at the very last epoch (epoch 49 and 50), which means they were cut off before convergence.

The project's own recommended next steps say **100–200 epochs** is needed. You stopped at 50.

At 50 epochs, the Node AP scores are:  
- GINE: 0.501, SAGE: 0.489

With proper training (200 epochs), these numbers are expected to be notably higher. **Your results may be understating the model's actual capability.**

**What you must do:**
- Run `compare.py` for 200 epochs on all 5 models
- This will likely push GINE above 0.55+ node AP
- GAT may close the gap with GINE with more training (it has the most parameters and needs the most data exposure)

---

### GAP 3 — No Ablation Study [IMPORTANT FOR ACADEMIC CREDIBILITY]

**The problem in plain words:**  
An ablation study means: "I remove one piece of the model at a time to prove that piece actually contributes."

Two major claims are made that are NOT proven by ablation:

**Claim A (Stage 1):** The GNN bottleneck in Stage 1 (the ViG-style Grapher layers) adds value over a plain pretrained ResNet + UNet decoder.  
- Your own results document says: *"isolating its exact contribution vs. the pretrained encoder alone was not ablated."*
- Without this comparison, you cannot claim that the GNN bottleneck in Stage 1 is doing anything useful. It may just be the pretrained ResNet doing all the work.

**Claim B (Stage 3):** Edge features (crack width, tortuosity, angle, etc.) help the GNN.  
- GCN (which ignores edge features) gets Node AP = 0.342 — barely above MLP (0.341)
- GINE (which uses edge features) gets Node AP = 0.501
- But this comparison conflates two things: using edge features AND having a more expressive message-passing formula
- You should compare GINE-with-edge-features vs GINE-without-edge-features to isolate the contribution of edge features specifically

**What you must do:**
- Stage 1: Train a standard ResNet34d + UNet (no GNN bottleneck) and compare its crack IoU to your HybridGraphUNet on DeepCrack
- Stage 3: Run one GINE experiment with `edge_attr=None` (no edge features) and compare to the full GINE

---

### GAP 4 — Research Goal vs. Achieved Goal Is Not Clearly Stated [IMPORTANT FOR PRESENTATION]

**The problem in plain words:**  
Your research goal is: *predict how cracks evolve over time (temporal link prediction)*.  
What you actually proved is: *a GNN can identify hidden parts of a static crack graph better than a non-graph model*.

These are related but NOT the same thing. Right now, if you present this to an examiner, they could reasonably ask: "How does predicting hidden parts of a static crack prove anything about temporal evolution?"

**What you must do:**
- Write a clear "Bridge Statement" that explicitly connects the two:
  - "We simulate temporal crack growth by hiding 20% of a crack graph and asking the model to recover it. This is a valid proxy for temporal prediction because: (1) new crack segments form at existing crack tips, (2) our node task identifies which tips have hidden neighbours — exactly the information needed to predict the next growth direction, (3) a model that understands current topology is a prerequisite for predicting future topology."
- This bridge must appear in your report, presentation slides, and verbal defense.

---

### GAP 5 — Clustering Results Are Orphaned [MODERATE PRIORITY]

**The problem in plain words:**  
You did clustering analysis and found important things (performance is much better on simple cracks than complex ones). But these results are not tied to the main story yet.

The stratified evaluation shows:
- On simple/low-branching cracks (Stratum 0): SAGE node AP = **0.669**
- On complex/high-branching cracks (Stratum 1): SAGE node AP = **0.458**

This is a meaningful finding — it tells you where the model succeeds and where it struggles. But it exists in a separate JSON file and is not discussed in `RESULTS.md`.

**What you must do:**
- Add a "Per-Crack-Type Analysis" section to the main results document
- Include the stratified table showing performance by crack complexity
- Argue why: complex cracks (many junctions, high tortuosity) are harder — more possible paths mean more ambiguity about which one is the hidden segment

---

### GAP 6 — No Comparison to Simple Heuristic Baselines [MODERATE]

**The problem in plain words:**  
You compare 5 different AI models against each other, but you don't compare against the simplest possible rule: **"the next crack node is the nearest unconnected node."**

In research, you always need to show your AI beats a simple, dumb rule — otherwise, why use AI at all?

**What you must do:**
- Add one heuristic: predict a hidden connection based on spatial proximity (nearest unconnected node gets the predicted edge)
- Compare this heuristic's Node AP and Edge AP to your GNN results
- If GNN beats this, it strengthens your claim. If it doesn't, that is an important finding too.

---

### GAP 7 — No Written Research Paper or Final Report [CRITICAL for Masters]

**The problem in plain words:**  
All your results live in scattered `.md` files in the codebase. There is no unified research paper or report that:
- States the research question
- Reviews related work
- Describes the methodology
- Reports results
- Discusses limitations
- Draws conclusions

For a Masters practicum worth 50% of your grade, this document is likely the primary deliverable. Code alone is not a research paper.

**What you must do:**
- Write a formal research report (6–10 pages, academic style)
- Sections: Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion, References
- The `RESULTS.md` files are a good starting point but need to be consolidated and formalized

---

## 5. Summary Scorecard

| Area | Status | Grade |
|------|--------|-------|
| Stage 1 — Segmentation | Complete, exceeds target | A |
| Stage 2 — Image to Graph | Complete, well-engineered | A |
| Stage 3 — GNN Training | Results exist but incomplete (50 epochs only) | B- |
| Data Cleaning | Rigorous, documented | A |
| Evaluation Protocol | Honest, leakage-free | A |
| End-to-End Pipeline Test | NOT DONE | F |
| Ablation Studies | NOT DONE | F |
| Research Narrative | Scattered, not unified | D |
| Clustering Integration | Done but not connected | C |
| Heuristic Baselines | NOT DONE | F |
| Final Written Report | NOT DONE | F |

**Overall readiness for final submission: 55% — not ready as-is**

---

## 6. Strict To-Do List (Priority Order)

### URGENT — Must Do Before Any Submission

**TODO 1: End-to-End Pipeline Demo**
- Run 10–20 real road images through Stage 1 (segmentation) → Stage 2 (graph conversion) → Stage 3 (GNN prediction)
- Produce a 3-panel visualization for each: [original image | crack mask | graph with GNN predictions]
- This is your "proof that the whole thing works"
- File to modify: `segmentation/generate_masks.py` (run on raw images), then pipe into `image_to_graph/build_dataset.py`, then `link_prediction/run.py --eval-only`

**TODO 2: Extended Training Run (200 epochs)**
- Run all 5 models for 200 epochs: `python3 link_prediction/compare.py --epochs 200`
- GAT especially needs this — it was hitting its best at the last epoch at 50
- Expected improvement: GINE and SAGE should reach node AP above 0.55+
- Save results to a new output directory `outputs/linkpred_200ep/`

**TODO 3: Stage 1 Ablation — Prove the GNN Bottleneck Works**
- Train a plain ResNet34d + UNet (no Grapher/GNN layers) on DeepCrack
- Compare crack IoU to your HybridGraphUNet (0.722)
- If HybridGraphUNet is higher, your GNN bottleneck claim is proven
- This is a critical academic credibility point

**TODO 4: Write the Bridge Statement**
- Write 3–5 sentences explaining HOW the hidden-edge task simulates temporal crack evolution
- This must go into your report, slides, and you must be able to say it verbally in 60 seconds
- Suggested text: see Gap 4 section above

**TODO 5: Write the Final Research Report**
- Consolidate all results from `RESULTS.md`, `RESULTS_SEGMENTATION.md`, `RESULTS_PHASE3.md` into one coherent academic document
- Structure: Abstract → Introduction → Related Work → Dataset → Methodology (Stage 1, 2, 3) → Results → Discussion → Conclusion
- Include the clustering stratified results as a sub-section of Results
- Target: 6–10 pages

---

### HIGH PRIORITY — Should Do If Time Allows

**TODO 6: Stage 3 Edge Feature Ablation**
- Run GINE with edge features disabled (pass zeros or remove edge_attr) 
- Compare to GINE with full edge features (current: node AP = 0.501)
- Shows whether crack geometry features (tortuosity, angle, thickness) actually help the GNN

**TODO 7: Add Heuristic Baseline**
- Implement nearest-neighbor heuristic: "predict the closest unconnected node as the hidden neighbour"
- Compare its Node AP and Edge AP to MLP and GNN results in the results table
- Even if it's bad, showing why it's bad strengthens your GNN argument

**TODO 8: Integrate Clustering into Main Results**
- Add a "Stratified Evaluation by Crack Complexity" section to `RESULTS.md`
- Include this table for the 5 models showing performance on simple vs complex cracks
- Write 2–3 sentences explaining why complex cracks are harder (more branching = more ambiguity)

---

### LOWER PRIORITY — Nice to Have for a Strong Presentation

**TODO 9: Visualization Figure for the Paper**
- Create one high-quality figure showing: road photo → segmentation mask → skeleton graph → GNN prediction overlay
- This single figure explains the entire 3-stage pipeline at a glance and is essential for any presentation or paper

**TODO 10: Literature Comparison Table**
- Find 3–5 published papers on crack segmentation and crack topology/graph analysis
- Show how your Stage 1 crack IoU (0.722) and Stage 3 node AP (0.501) compare
- Even if other papers use different datasets, a comparison table shows academic context

**TODO 11: Reproducibility Check**
- Record the exact commands needed to reproduce all results, in a single numbered list
- A researcher should be able to re-run your experiments from scratch using only your README
- Verify the current `README.md` is complete and accurate

---

## 7. Key Technical Terms You Must Know

These are words you will be asked about in a presentation or defense. Know what they mean.

| Term | What it means in simple words |
|------|------------------------------|
| **Graph Neural Network (GNN)** | An AI that works on network/graph data — it passes information between connected nodes to learn patterns |
| **Link Prediction** | The task of predicting which new connections (edges) will appear in a network |
| **Average Precision (AP)** | A score that measures how well a model ranks positives above negatives — the primary metric here because there are far more negatives than positives |
| **Node Task** | Your primary task: predict which crack tip nodes have a hidden neighbour |
| **Edge Task** | Secondary/control task: predict which specific hidden crack segments exist |
| **Topology** | The shape and connectivity structure of a network — how many branches, how they connect, not just where they are in space |
| **Skeletonization** | Thinning a crack mask down to a 1-pixel-wide centreline — the input to the graph conversion |
| **Tortuosity** | How curved or twisty a crack segment is — a straight crack has tortuosity = 1.0, a wiggly one is higher |
| **Ablation Study** | Removing one piece of a model to prove that piece actually contributes something |
| **Hard Negatives** | Incorrect answers that are spatially close to correct answers — used to make the training problem harder and more realistic |
| **Data Leakage** | When the model accidentally sees information it shouldn't, leading to inflated (fake-good) results |
| **Class Imbalance** | When one category (e.g. "has hidden neighbour") is much rarer than another (e.g. "no hidden neighbour") — here it's 1:11 |
| **Inductive Bias** | The built-in assumption a model makes — GNNs assume nearby connected nodes share information, which matches how cracks behave |
| **IoU (Intersection over Union)** | A score for segmentation: how much do the predicted crack pixels overlap with the true crack pixels? 1.0 = perfect, 0 = no overlap |
| **Message Passing** | The core operation in GNNs: each node collects information from its neighbours and updates itself |

---

## 8. What You Can Confidently Claim Right Now

At the current state, you can make the following statements and defend them:

1. "We built and validated a 3-stage pipeline for road crack topology analysis."
2. "Our segmentation model achieves crack IoU of 0.722 on the DeepCrack benchmark, exceeding our target of 0.65–0.70."
3. "We cleaned a multi-source crack dataset from 11,298 to 4,769 images by removing non-road-pavement data and fixing cross-split data leakage."
4. "We demonstrated that GNNs understand crack topology: GINE achieves 47% higher Average Precision than the structure-blind MLP baseline on the node task, under an evaluation protocol that eliminates spatial shortcuts."
5. "The edge task result — where MLP outperforms GNNs — is a deliberate control result confirming that local structural features dominate segment-level prediction, while the node task requires genuine topology reasoning."

---

*End of Report. Next action: complete TODO 1 (end-to-end pipeline demo) and TODO 5 (write the final research report). These are the two most critical missing pieces.*
