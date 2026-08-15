# Section I — Introduction

> **Reference numbers** use `00_references.md` numbering.  
> **Target:** ~600–750 words body text (~1.0–1.5 IEEE double-column pages).

---

## I. Introduction

Civil infrastructure — bridges, roads, tunnels, and retaining structures — is subject to progressive material degradation manifesting as surface and subsurface cracking. Undetected crack propagation leads to costly structural failure; regular inspection is safety-critical but labour-intensive at the scale of national infrastructure networks. Wettewa et al. [1] survey 111 papers applying Graph Neural Networks to infrastructure operation and maintenance, establishing that automated structural health monitoring is an active and open research frontier.

Deep learning has transformed crack detection. Convolutional segmentation networks — from U-Net [10] through DeepCrack [2] — now achieve pixel-level crack localisation competitive with human annotation, enabling fully automated inspection pipelines. Yet pixel-level output answers only the binary question *"is this pixel a crack?"* It cannot answer: *Where does this crack end? Which disconnected segments are the same physical crack, interrupted by occlusion or poor contrast? Which tip nodes are at the active propagation front?* These questions require **structural topology reasoning** — understanding cracks not as a set of pixels but as a graph of interconnected segments with tip nodes, junction nodes, and directional propagation paths.

The structural health relevance of crack topology has been recognised in the fracture mechanics and materials science literature. Ji et al. [15] demonstrate that skeleton-based graph construction — skeletonisation of crack masks followed by graph extraction with degree-based node classification — yields meaningful geometric measurements about fracture networks in rock samples. Topology-preserving loss functions [11] and topology-informed segmentation architectures [13] show that preserving connectivity during segmentation directly improves downstream structural interpretability. However, all of these works treat topology as a property to be preserved or measured, not as an object of GNN-based learning. None apply link prediction to image-derived crack topology graphs.

GNN-based approaches to crack analysis exist but are structurally different from ours. Perera et al. [4] apply message-passing GNNs to crack tip coalescence prediction in finite element method (FEM) meshes under controlled simulation conditions — not real image data, with no skeletonisation, and with crack tips manually placed. Djenouri et al. [14] construct graphs from SIFT feature correlations between image regions for GCN-based road crack detection, with no skeleton-level primitives. Zhang and Chen [7] establish link prediction on graph neural networks as a principled framework for inferring missing connections in sparse graphs, demonstrating that local subgraph structure is sufficient for high-accuracy prediction. Wu et al. [3] survey GNN families and confirm that spatial message-passing methods consistently outperform heuristic predictors on irregular real-world graphs. Yet none of this work is applied to crack topology graphs derived from real inspection images. Critically, no work among the 111 papers surveyed by Wettewa et al. [1] addresses GNN link prediction on image-derived crack topology graphs — making this sub-task an open benchmark gap in the infrastructure GNN literature.

We directly fill this gap. This paper presents a three-stage pipeline for crack topology reasoning: a topology-preserving segmentation model (Stage 1), a skeleton-to-attributed-graph converter (Stage 2), and a multi-encoder GNN link prediction benchmark (Stage 3). The pipeline is evaluated on 4,769 road-pavement crack images curated from 6 sources (selected from 11 candidates after domain filtering and deduplication), resulting in 4,364 valid topology graphs for GNN training and evaluation.

The contributions of this work are as follows:

1. **HybridGraphUNet** — a CNN–GNN hybrid segmentation model combining a pretrained ResNet34d encoder with a ViG-style GNN bottleneck under a combined Focal + CrackDice loss, achieving crack IoU of **0.722** on the DeepCrack benchmark, competitive with DeepCrack's own reported F1 of 0.741 [2].

2. **Crack topology graph construction** — a fully reproducible Stage 2 pipeline converting crack masks to attributed PyG graphs via morphological skeletonisation, sknw graph extraction, spur pruning, 6-dimensional node features (position, thickness, degree, endpoint/junction flags), and 8-dimensional edge features (path length, Euclidean distance, tortuosity, crack width, circular angle encoding).

3. **First benchmark for GNN link prediction on image-derived crack topology graphs** — evaluating five encoder architectures (MLP, GCN, GraphSAGE, GINE, GAT) with a fully honest evaluation protocol: structural feature recomputation after masking, hard negative sampling via KD-tree (k = 30), and no Hits@K / MRR metrics that mechanically inflate on small graphs.

4. **Statistical proof of topology understanding** — GINE achieves mean node Average Precision **0.739 ± 0.004** versus **0.662 ± 0.003** for the structure-blind MLP baseline, a gap of **+0.077** that is 24.8× larger than seed variance with zero distributional overlap across three independent seeds. The gain decomposes cleanly: +0.038 from message passing alone (MLP → GraphSAGE) and +0.039 from geometric edge features (SAGE → GINE).

5. **Frontier masking** — a structurally-targeted evaluation protocol that hides tip nodes at the active crack propagation front, amplifying the GINE–MLP gap from +0.060 to **+0.210** (3.5× amplification), demonstrating that topology-aware models excel precisely where structural reasoning is hardest.

6. **End-to-end validation** — full pipeline evaluation on 100 held-out images confirms that the HybridGraphUNet predicted mask path achieves node AP **0.800**, exceeding the ground-truth oracle path (0.655), validating the complete Stage 1 → Stage 2 → Stage 3 integration.

The remainder of the paper is organised as follows. Section II surveys related work on GNN theory, crack segmentation, image-to-graph approaches, and GNN encoders for physical graphs. Section III describes the methodology of all three stages. Section IV presents experimental results. Section V concludes with limitations and future directions.

---

> **Word count (body text):** ~760 words  
> **Target in final paper:** ~1.0–1.5 IEEE double-column pages  
> **Papers cited in this section:** [1], [2], [3], [4], [7], [10], [11], [13], [14], [15] — 10 papers
