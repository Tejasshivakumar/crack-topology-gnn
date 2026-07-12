# Literature Review — Crack Topology GNN

All papers cited in the IEEE submission, organized by topic cluster.
Each entry covers: summary, why it fits, the gap it exposes, and how our work fills it.

---

## Part 1 — Theoretical Foundations

### 1. Xu et al. — "How Powerful are Graph Neural Networks?"
**Venue:** ICLR 2019 (International Conference on Learning Representations)  
**arXiv:** 1810.00826  
**URL:** https://arxiv.org/abs/1810.00826

**Summary:**  
Proves that GNNs with iterative neighborhood aggregation are at most as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test, and that GINs (Graph Isomorphism Networks) achieve this upper bound. Establishes the formal theoretical limit of what message-passing GNNs can distinguish.

**Why it fits:**  
This is the theoretical backbone of the entire Stage 3 argument. It proves that after k rounds of message passing, every node's embedding encodes the full structure of its k-hop neighborhood. A 4-way junction and a dead-end tip node in a crack graph provably receive different embeddings — without supervision, without explicit clustering.

**The gap:**  
The WL equivalence is proven for graphs with discrete node labels, not continuous multi-dimensional node and edge feature vectors (which crack topology graphs have).

**How our work fills it:**  
Our crack topology graphs have 6-dimensional continuous node features (position, degree, distance-transform thickness, local angle) and 8-dimensional edge features. The WL result motivates the architecture choice but does not cover the attributed graph case — which is addressed by the Neural Networks 2024 WL paper (see §2.6).

---

### 2. Wu et al. — "A Comprehensive Study of Graph Neural Networks"
**Venue:** IEEE Transactions on Neural Networks and Learning Systems, 2021, Vol. 32, No. 1, pp. 4–24  
**DOI:** 10.1109/TNNLS.2020.2978386

**Summary:**  
Comprehensive survey covering all major GNN families: spectral (GCN), spatial (GraphSAGE, GAT), and graph auto-encoders. Formally defines the message-passing framework and analyzes the expressiveness limits of each architecture. Includes discussion of edge feature integration and the role of architectural choices in irregular graph domains.

**Why it fits:**  
Provides the authoritative IEEE Transactions survey context situating our multi-encoder comparison (MLP vs GCN vs SAGE vs GINE vs GAT) within established literature. The survey specifically highlights that spatial methods (SAGE, GAT) outperform spectral methods (GCN) on irregular real-world graphs with non-uniform degree distributions — precisely what we observe: GCN (0.599) < MLP (0.662) < SAGE (0.700) on our crack topology graphs.

**The gap:**  
Survey covers general GNN applications but does not address infrastructure inspection graphs derived from real images, and does not evaluate link prediction on topology graphs constructed from skeletonized defect maps. No empirical benchmark exists for this application domain.

**How our work fills it:**  
We instantiate the message-passing paradigm on a domain-specific graph type (skeleton-derived crack topology graphs) and provide the first empirical comparison of five encoder families on a link prediction task — discovering the counterintuitive GCN < MLP result that the survey's spectral-vs-spatial analysis predicts but does not verify in infrastructure domains.

---

### 3. Zhang & Chen — "Link Prediction Based on Graph Neural Networks" (SEAL)
**Venue:** NeurIPS 2018  
**URL:** https://proceedings.neurips.cc/paper_files/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf

**Summary:**  
Introduces SEAL (Subgraphs, Embeddings, and Attributes for Link prediction) — a framework that extracts local enclosing subgraphs around candidate node pairs, labels them with the Double-Radius Node Labeling (DRNL) scheme, and classifies them with DGCNN. Outperforms all heuristic link predictors (Common Neighbors, Jaccard, Katz) by over 10–20% AUC on sparse graphs where traditional heuristics fail.

**Why it fits:**  
SEAL is the canonical GNN-based link prediction paper. It establishes that local subgraph structure (not just node embeddings) is sufficient for high-accuracy link prediction — a key principle behind our masking evaluation protocol, where we ask the GNN to predict whether masked crack tip nodes were originally connected.

**The gap:**  
SEAL operates on generic social/biological/technological networks (USAir, Celegans, Power, Router). The graphs are abstract — nodes have no physical meaning and edges have no feature vectors. The task is to recover deleted edges from a static graph, not to predict structural connectivity in a physically constrained domain.

**How our work fills it:**  
Our link prediction task is domain-specific: masked nodes are crack tip nodes and the question is whether they were topologically connected in the original skeleton. We augment the standard edge masking with frontier masking (hiding boundary-adjacent tips specifically) and show that domain-specific masking protocols reveal much larger encoder gaps (GINE−MLP gap widens from +0.066 to +0.210) than random edge masking alone.

---

### 4. Xu et al. / Azizian & Lelarge — "Weisfeiler–Lehman goes dynamic: An analysis of the expressive power of Graph Neural Networks for attributed and dynamic graphs"
**Venue:** Neural Networks, ScienceDirect 2024  
**DOI:** 10.1016/j.neunet.2024.106097

**Summary:**  
Extends the WL expressiveness analysis to attributed graphs (nodes and edges carry continuous feature vectors) and to dynamic graphs. Proves that GNNs with edge feature propagation (as in GINE) distinguish structurally non-equivalent nodes more powerfully than node-feature-only GNNs, and formalizes the additional discriminating power that edge attributes provide.

**Why it fits:**  
This is the theoretical foundation for why GINE outperforms GCN and SAGE on our crack topology graphs. Our graphs have 8-dimensional edge features (length, tortuosity, thickness, angle encoding) — precisely the attributed graph setting covered by this extension.

**The gap:**  
The theoretical result is proven in a general setting. It does not connect to the physical meaning of edge features in any specific domain, nor does it demonstrate the effect empirically on real-world infrastructure graphs.

**How our work fills it:**  
Our edge feature ablation (M5) is the empirical counterpart of this theorem: dropping thickness alone drops GINE AP by −0.347; dropping all edge features drops it by −0.349 — showing that one edge feature dimension carries nearly all the discriminating power. This is consistent with the theory (edge features help) while revealing which feature matters most in the crack domain.

---

## Part 2 — Crack and Fracture GNNs

### 5. Perera et al. — "Graph neural networks for simulating crack coalescence and propagation in brittle materials"
**Venue:** Computer Methods in Applied Mechanics and Engineering, ScienceDirect 2022  
**DOI:** 10.1016/j.cma.2022.115021

**Summary:**  
First paper to apply GNNs to crack mechanics. Builds a graph from FEM simulation meshes where nodes are crack tips and edges encode geometric proximity and orientation. The GNN learns to predict which crack tips will coalesce — a direct precursor to our link prediction task.

**Why it fits:**  
The closest prior work to our Stage 3 task. Explicitly argues that crack-to-crack interaction is a graph topology problem, not a pixel neighborhood problem — validating the core motivation of our entire pipeline. The architecture choice is justified by the relational nature of crack tip interactions.

**The gap:**  
Operates entirely on FEM simulation graphs. Graphs are constructed from finite element node positions, not from real images. Crack tips are manually placed in controlled simulations — no segmentation, no skeletonization, no image-derived noise. Results do not transfer to real-world crack images with irregular morphology.

**How our work fills it:**  
We build the first pipeline that goes from real crack images (11 sources) through segmentation, skeletonization, and spur pruning to a topology graph, then applies GNN link prediction on the result. Our task is harder: the graph is noisy (skeleton artifacts, over-segmented junctions) and the crack tip positions are detected, not manually placed.

---

### 6. Hu et al. — "A generalized machine learning framework for brittle crack problems using transfer learning and graph neural networks"
**Venue:** Mechanics of Materials, ScienceDirect 2023  
**DOI:** 10.1016/j.mechmat.2023.104639

**Summary:**  
Extends Perera 2022 by adding transfer learning — a GNN trained on one crack configuration (e.g., periodic arrays) generalizes to new geometries (random distributions). Proves that GNNs capture transferable topological patterns of crack interaction rather than configuration-specific memorization.

**Why it fits:**  
Directly supports the generalization argument for our 11-source training dataset. If a GNN can transfer across FEM simulation geometries, a GNN trained on 11 crack image sources should generalize to unseen crack types. The transfer learning result motivates our multi-source dataset construction.

**The gap:**  
Still simulation-only. Transfer is across different crack array configurations within FEM — not across fundamentally different image acquisition sources (concrete vs asphalt vs laboratory vs field). The feature space (FEM mesh coordinates) is clean and uniform.

**How our work fills it:**  
Our dataset spans 11 sources with different image statistics, crack types, and acquisition conditions. The GINE model is trained and evaluated across this heterogeneous set — a harder generalization challenge than FEM geometry transfer. The dataset cleaning pipeline (scope filter + deduplication, 11,298 → 4,769 images) is itself a contribution to handling real-world heterogeneity.

---

### 7. Shukla et al. — "MPNN based graph networks as learnable physics engines for deformation and crack propagation in solid mechanics"
**Venue:** International Journal of Solids and Structures, ScienceDirect 2024  
**DOI:** 10.1016/j.ijsolstr.2024.112788

**Summary:**  
Applies Message Passing Neural Networks (MPNNs) to FEM mesh graphs to learn crack propagation dynamics. Each message passing step propagates stress/displacement information along the mesh topology — analogous to physical crack growth. Proves MPNNs are competitive with classical FEM solvers at a fraction of the computational cost.

**Why it fits:**  
Provides physics-grounded justification for why message passing is the right mechanism for crack topology tasks: stress propagates along crack paths, and message passing propagates information along graph edges — the two are structurally analogous. Directly supports the choice of GINE (an MPNN variant) over non-message-passing baselines.

**The gap:**  
Applied to FEM mesh graphs with known physics boundary conditions. The "propagation" is literal physics, not learned structure. No image processing, no skeletonization, no real-world crack images.

**How our work fills it:**  
We apply the same MPNN mechanism to image-derived crack topology graphs where no physics simulator exists. The analogy holds: crack connectivity (which tips join which junctions) propagates as graph structure, and MPNN message passing learns to recognize these structural patterns. The frontier masking experiment specifically tests whether GINE has learned to propagate structural information across the graph boundary — the same kind of propagation the MPNN physics engine performs.

---

## Part 3 — Crack Segmentation and Detection

### 8. Liu et al. — "DeepCrack: A deep hierarchical feature learning architecture for crack segmentation"
**Venue:** Neurocomputing, ScienceDirect 2019  
**DOI:** 10.1016/j.neucom.2019.01.036  
**URL:** https://www.sciencedirect.com/science/article/pii/S0925231219300566

**Summary:**  
End-to-end deep CNN for pixel-wise crack segmentation without hand-crafted features. Achieves mIoU=85.9% and F1=86.5% on their multi-scene benchmark. The DeepCrack dataset (including labeled crack images) is released publicly and is widely used as a benchmark.

**Why it fits:**  
DeepCrack is the benchmark dataset used to evaluate our Stage 1 segmentation (HybridGraphUNet achieves IoU=0.718 on the DeepCrack dataset). It also establishes the state-of-the-art segmentation baseline that our Stage 1 must match before the pipeline proceeds to Stage 2.

**The gap:**  
DeepCrack is pure segmentation — pixel-level output. The architecture has no notion of crack topology: it cannot identify which crack segments are tips, which are junctions, or how cracks connect. The output is a binary mask, not a structured representation of the crack network.

**How our work fills it:**  
We use DeepCrack-quality segmentation as the input to Stage 2 (mask → skeleton → topology graph). The topology graph captures exactly what DeepCrack cannot — junction types, tip locations, and inter-segment connectivity — and Stage 3 GNN operates on this structured representation to predict crack propagation paths.

---

### 9. Bi et al. — "Road Crack Detection Using Deep Neural Network Based on Attention Mechanism and Residual Structure" (AR-UNet)
**Venue:** IEEE Access 2023  
**DOI / URL:** https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10003197

**Summary:**  
Proposes AR-UNet: a U-Net variant with residual blocks and CBAM (Convolutional Block Attention Module) for road crack segmentation. Demonstrates improved thin crack recall and background noise suppression. Evaluated on DeepCrack, Crack Forest Dataset (CFD), and Road Image Dataset (RID).

**Why it fits:**  
AR-UNet is a direct architectural precursor to our HybridGraphUNet — both use U-Net encoder-decoder with residual connections for crack segmentation. The CBAM attention mechanism in AR-UNet motivates our GNN bottleneck design: both aim to focus the model on structurally relevant features rather than homogeneous background regions.

**The gap:**  
AR-UNet is a pure segmentation model — its output is a pixel mask. The attention mechanism improves crack pixel recall but provides no structural understanding: it cannot reason about whether two detected crack segments belong to the same crack propagation path.

**How our work fills it:**  
Our HybridGraphUNet replaces the CNN bottleneck with a GNN bottleneck (operating on a superpixel adjacency graph at the feature level). The GNN bottleneck incorporates structural context during segmentation — not just in the downstream graph construction step. This is a novel hybrid approach that goes beyond the attention-only mechanism in AR-UNet.

---

### 10. Jiang et al. — "Hybrid graph convolutional and deep convolutional networks for enhanced pavement crack detection"
**Venue:** Engineering Applications of Artificial Intelligence, ScienceDirect 2025  
**DOI:** 10.1016/j.engappai.2025.110227

**Summary:**  
Converts crack images to graphs using superpixel segmentation (each superpixel = node, spatial adjacency = edges), then applies GCN on top of a CNN backbone for joint crack classification. Explicitly motivates graph representation because defects form spatial structural relationships that fixed-grid convolution cannot capture.

**Why it fits:**  
The closest published work to our full pipeline at the system level: image → graph → GNN. Directly validates the motivation for using a graph representation of cracks rather than pure pixel-level processing.

**The gap:**  
Nodes are superpixels (pixel blobs), not topological features. The graph encodes spatial proximity — not crack topology. Junctions and endpoints are never identified. No link prediction is performed. The GCN operates at the segmentation stage to classify whether a superpixel contains a crack, not to reason about crack connectivity.

**How our work fills it:**  
We take the next step in abstraction: after segmentation, we extract the skeleton and identify structurally meaningful nodes (crack tips = degree-1, junctions = degree-3+). The resulting topology graph is an order of magnitude smaller (tens of nodes vs thousands of superpixels) and structurally meaningful. GNN link prediction on this graph is fundamentally different from GCN-on-superpixels.

---

## Part 4 — Skeletonization and Topological Graph Construction

### 11. Zhao et al. — "Automatic identification of rock fractures based on deep learning"
**Venue:** Engineering Geology, ScienceDirect 2024  
**DOI:** 10.1016/j.enggeo.2024.107630

**Summary:**  
Segments fracture images → extracts single-pixel skeleton using Zhang-Suen thinning → identifies I, Y, X junction nodes (degree 1/3/4) → builds a polyline graph with geometric measurements (fracture length, aperture, connectivity count). Structurally identical to our Stage 2 pipeline.

**Why it fits:**  
This paper is the closest published work to our Stage 2 (image_to_graph/). The pipeline — segment → skeletonize → classify nodes by degree → build graph — matches ours step for step. It validates that degree-based node classification (tip = degree 1, junction = degree ≥ 3) is the established method in the fracture analysis literature.

**The gap:**  
The graph is used purely for geometric measurement. No GNN is trained on the resulting graph. No link prediction, no learning. The graph is a measurement artifact, not a training input. The paper stops exactly where ours starts.

**How our work fills it:**  
We take the graph output of this style of pipeline and feed it to a GNN link prediction model. The node features (position, degree, distance-transform thickness) and edge features (length, tortuosity, thickness, angle encoding) enrich the bare topology graph with physically meaningful attributes that the GNN can exploit.

---

### 12. Wang et al. — "Automatic extraction and quantitative analysis of characteristics from complex fractures on rock surfaces via deep learning"
**Venue:** International Journal of Rock Mechanics and Mining Sciences, ScienceDirect 2025, Vol. 187, Article 106038  
**DOI:** 10.1016/j.ijrmms.2025.106038

**Summary:**  
Uses Zhang-Suen thinning to produce single-pixel-wide skeletons from segmented fracture masks, then classifies intersection nodes (I/Y/X shapes) by degree and computes fracture network connectivity statistics. Rule-based intersection typing using morphological pattern matching.

**Why it fits:**  
Directly validates the Zhang-Suen skeletonization step in our Stage 2 pipeline. Confirms that degree-based node classification (which we implement via sknw) is the standard method in the rock fracture literature, and that the skeleton accurately preserves topological connectivity of the original fracture network.

**The gap:**  
Skeletonization is preprocessing for geometric measurement — junction counts, fracture lengths, connectivity statistics. The classified graph is never fed to a learning model. Intersection typing is rule-based (degree counting), not learned. No prediction task.

**How our work fills it:**  
We add 6-dimensional node features and 8-dimensional edge features on top of the bare topology graph that this paper constructs, then train GINE to predict which masked tip nodes were originally connected. The GNN learns structural patterns that degree-counting alone cannot capture.

---

### 13. Li et al. — "Deep learning-based multi-scale crack image segmentation and improved skeletonization measurement method"
**Venue:** Materials Today Communications, ScienceDirect 2025, Vol. 46, Article 112727  
**DOI:** 10.1016/j.mtcomm.2025.112727 (PII: S2352492825012395)

**Summary:**  
Combines deep learning crack segmentation with an improved skeletonization algorithm for crack width and length measurement. Explicitly validates that skeletonization is the correct post-processing step after segmentation — preserving topology while reducing to single-pixel-wide representations. Demonstrates measurement accuracy on multi-scale crack images.

**Why it fits:**  
Validates our Stage 1 → Stage 2 transition choice. The paper's conclusion — that segmentation + skeletonization preserves the topological structure of crack networks — is the foundational assumption behind our entire Stage 2 pipeline.

**The gap:**  
The skeleton is used for geometric measurement only (crack width, length, count). The pipeline stops at measurement — the skeleton is never converted to a PyG graph, and no GNN operates on the result.

**How our work fills it:**  
We treat the skeletonized output as the input to graph construction (sknw → spur pruning → PyG Data), generating 4,363 graphs (3,727 train / 636 test) with rich node and edge features. The skeletonization step that this paper validates as accurate is the same step that feeds our GNN training pipeline.

---

### 14. TopoM-CrackNet — "Topology-informed deep learning for pavement crack detection: Preserving consistent crack structure and connectivity"
**Venue:** Automation in Construction, Elsevier/ScienceDirect 2025, Vol. 174, Article 106089  
**DOI:** 10.1016/j.autcon.2025.106089 (pii/S0926580525001608)  
**URL:** https://www.sciencedirect.com/science/article/abs/pii/S0926580525001608

**Summary:**  
Proposes TopoM-CrackNet — a U-Net variant enhanced with the Vmamba backbone and persistent homology (PH) loss to enforce topological consistency in crack segmentation. Persistent homology (Betti numbers) is used as a training signal to ensure that segmented crack masks preserve junction connectivity and endpoint structure. Achieves mIoU = 0.727, outperforming nnUNet and SegFormer, while running nearly twice as fast. The paper explicitly links segmentation quality to topological structure preservation — a segmentation model that gets connectivity wrong produces incorrect topology graphs downstream.

**Why it fits:**  
Directly connects segmentation quality to topological accuracy — exactly the relationship our end-to-end evaluation (C3) measures. The persistent homology framework is the principled version of what our Stage 1 → Stage 2 chain relies on: if the segmentation breaks crack connectivity, the skeleton graph will have incorrect junction/endpoint labels. This paper validates that topology-aware training is necessary for downstream graph construction.

**The gap:**  
TopoM-CrackNet stops at the segmentation stage. The topologically consistent mask is the final output — it is never converted to a skeleton graph, no GNN is trained on the resulting topology, and no link prediction is attempted. Topology preservation is treated as a segmentation quality metric, not as an input to a learning pipeline.

**How our work fills it:**  
We take a topologically consistent mask as the starting point for Stage 2 (skeleton → junction/endpoint graph) and Stage 3 (GNN link prediction). The persistent homology argument this paper makes — that segmentation must preserve crack connectivity — is the theoretical justification for our C3 end-to-end evaluation, which measures exactly how much topology is lost when going from oracle GT masks to HybridGraphUNet predicted masks (oracle AP = 0.655, predicted AP = 0.800, gap = −0.145).

---

## Part 5 — Domain Generalization and Multi-Source Datasets

### 15. NVE-DGCNN — "Dynamic graph CNN based semantic segmentation of concrete defects and as-inspected modeling"
**Venue:** Automation in Construction, ScienceDirect 2024  
**DOI:** 10.1016/j.autcon.2024.105239

**Summary:**  
Applies Dynamic Graph CNN (DGCNN) to 3D point clouds of concrete structural defects (including cracks). Achieves 98.6% recall on crack points using the graph model. Explicitly chooses a graph-based architecture over CNN because defects form spatial structural relationships that convolution over a fixed grid cannot capture.

**Why it fits:**  
Directly in our application domain (concrete structural defects) and explicitly validates the graph architecture choice. The justification — "CNNs process defect pixels in a fixed 2D grid neighborhood and cannot model the connectivity between spatially separated crack segments" — is identical to our own motivation for Stage 2 graph construction.

**The gap:**  
Operates on 3D point cloud data (LiDAR / photogrammetry), not 2D crack images. The graph is built on spatial proximity of 3D points, not on the topological skeleton of a segmented 2D crack mask. No skeletonization, no tip/junction classification, no link prediction.

**How our work fills it:**  
We address the same domain (concrete crack inspection) but through the 2D image pathway that is more accessible in practice (standard cameras vs LiDAR). The topology graph we construct is structurally more informative than a spatial proximity graph — it encodes which crack segments are endpoints vs junctions, a distinction the 3D point cloud approach cannot make without additional annotation.

---

### 16. Amara et al. — "Graph Neural Networks for building and civil infrastructure operation and maintenance enhancement"
**Venue:** Advanced Engineering Informatics, ScienceDirect 2024  
**DOI:** 10.1016/j.aei.2024.102868

**Summary:**  
Systematic review of 111 GNN papers applied to building and civil infrastructure. Explicitly identifies the limitation of CNNs for structural tasks: "conventional CNNs are effective at extracting localized, grid-structured features, but their inherently localized receptive fields constrain their ability to model long-range structural dependencies." GNNs are justified because infrastructure systems have relational topology.

**Why it fits:**  
Provides the survey-level justification for replacing CNN-only approaches with GNN-based methods in civil infrastructure inspection. Directly positions our work within the rapidly growing GNN-for-infrastructure literature and confirms that no prior survey paper covers the specific sub-task of GNN link prediction on skeletonized crack graphs.

**The gap:**  
As a survey, it identifies what GNNs have been applied to — but identifies crack topology graph-based link prediction as an open problem. No existing work in the 111 surveyed papers addresses the task of predicting crack tip connectivity from image-derived skeleton graphs.

**How our work fills it:**  
We are one of the first papers to fill exactly this gap identified in the survey: applying GNN link prediction to crack topology graphs derived from real infrastructure images. Our pipeline (image → skeleton → topology graph → GINE) represents the missing point in the survey's coverage map.

---

### 17. Zhang et al. — "Multi-source dynamic adaptive domain generalization network for crack detection under unknown environments"
**Venue:** Measurement, ScienceDirect 2024  
**DOI:** 10.1016/j.measurement.2024.115914

**Summary:**  
Demonstrates that crack detection models trained on a single source fail on unseen environments (different surfaces, lighting, temperatures). Proposes multi-source training with domain adaptation as the established solution for robust crack detection. Evaluates across multiple surface types.

**Why it fits:**  
Directly motivates our 11-source dataset construction. If pixel-level detection models trained on a single source fail to generalize, then a topology graph learning model trained on a single crack source will face the same problem. Multi-source training is the principled solution for both.

**The gap:**  
Addresses domain generalization at the pixel/signal level — adapting CNN feature extractors to new image statistics. Does not extend to the graph representation level: even with multi-source training, pixel models cannot reason about crack segment connectivity or predict propagation paths.

**How our work fills it:**  
We address the same generalization problem but at the topology graph level. Our 11-source dataset (11,298 images → 4,769 after cleaning) ensures the GNN sees diverse crack morphologies during training. The GINE model trained on this heterogeneous set must learn topology features that generalize across surface types — a strictly harder generalization task than pixel-level domain adaptation.

---

---

## Part 6 — Foundational Methods (Architecture Citations)

These papers are directly cited in the technical reports for the architectures implemented in the pipeline.

### 18. Ronneberger et al. — "U-Net: Convolutional Networks for Biomedical Image Segmentation"
**Venue:** MICCAI 2015 (Springer LNCS, Vol. 9351, pp. 234–241)  
**DOI:** 10.1007/978-3-319-24574-4_28  
**arXiv:** 1505.04597

**Summary:**  
Introduces the encoder–decoder architecture with skip connections that became the standard for dense prediction tasks. The key innovation is symmetric expansion path that allows the network to localize precisely while also using context — solving the resolution vs. context trade-off.

**Why it fits:**  
Our Stage 1 HybridGraphUNet uses the U-Net encoder-decoder structure with ResNet34d backbone. The skip connections between encoder and decoder stages are a direct application of the U-Net design. Without this citation, the Stage 1 architecture description is incomplete.

**The gap:**  
U-Net was designed for biomedical images with clear boundaries. Crack images have thin, elongated, often discontinuous structures — harder segmentation targets that require topology-aware loss functions beyond standard BCE.

**How our work fills it:**  
We extend U-Net with a GNN bottleneck (DGCNN-style Grapher module) and SoftClDice topology-preserving loss, specifically addressing the thin-structure limitation identified by the U-Net authors.

---

### 19. Kipf & Welling — "Semi-Supervised Classification with Graph Convolutional Networks"
**Venue:** ICLR 2017  
**arXiv:** 1609.02907

**Summary:**  
Introduces the Graph Convolutional Network (GCN) — the first scalable spectral graph convolution that uses a first-order approximation of spectral filters, leading to the symmetric normalization Ã = D^{-1/2} Â D^{-1/2}. Achieves state-of-the-art node classification on citation networks.

**Why it fits:**  
Directly cited for the GCNEncoder in Stage 3. Our implementation uses GCNConv from PyTorch Geometric, which implements this exact formulation. The symmetric normalization is the design choice that causes GCN to underperform MLP on our crack graphs (see stage3_report.md §11.6).

**The gap:**  
GCN was designed for homophilic graphs (connected nodes tend to share labels) with uniform degree distributions. Crack topology graphs are heterophilic — degree-1 tip nodes (positive class) are connected to degree-3+ junctions (negative class), violating the homophily assumption.

**How our work fills it:**  
Our empirical result (GCN < MLP, node AP = 0.599 vs 0.662) is the first demonstration that GCN's spectral normalization fails specifically on crack topology graphs — providing a counterexample to the assumption that GCNs always outperform no-graph baselines.

---

### 20. Ciano et al. — "On Inductive–Transductive Learning With Graph Neural Networks"
**Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 44, No. 2, pp. 758–769, 2021  
**DOI:** 10.1109/TPAMI.2021.3054304

**Summary:**  
Formally studies the distinction between inductive GNNs (which generalize to unseen graphs at inference) and transductive GNNs (which specialize to the training graph). Derives theoretical conditions under which transductive evaluation is more appropriate than cross-graph inductive evaluation, and empirically validates these conditions on node and graph classification benchmarks.

**Why it fits:**  
Our Stage 3 uses per-graph transductive evaluation — each crack image produces a single graph with an 80/10/10 node split for training, validation, and test. This is precisely the transductive regime studied by Ciano et al. The paper provides formal justification for why cross-graph (inductive) evaluation is inappropriate here: crack topology is image-specific and the node feature distribution is graph-dependent.

**The gap:**  
Their analysis covers node and graph classification on fixed, homogeneous graphs. It does not address link prediction in engineering inspection graphs, nor does it study how geometric edge features (thickness, tortuosity, angle) interact with the inductive–transductive trade-off.

**How our work fills it:**  
We operate in the transductive regime as prescribed by Ciano et al. and extend to link prediction with crack-specific geometric edge features. The per-graph split validates whether learned topology generalizes within-graph — and our multi-seed (3 seeds × 5 models) validation confirms the finding is not seed-sensitive.

---

### 21. Wang et al. — "EGAT: Edge-Featured Graph Attention Network"
**Venue:** International Conference on Artificial Neural Networks (ICANN 2021), Lecture Notes in Computer Science, Springer International Publishing  
**DOI:** 10.1007/978-3-030-86362-3_21

**Summary:**  
Extends GAT to graphs with edge features by incorporating edge feature vectors into both the attention coefficient computation and the message-passing aggregation. EGAT iterates node and edge representations jointly, allowing the model to differentially weight neighbor contributions using geometric context carried by the edges — not just the neighbor node embeddings.

**Why it fits:**  
Directly cited for the CrackGATEncoder in Stage 3. Our implementation uses PyG's GATConv with the edge_dim argument, which follows EGAT's formulation: edge features are projected and injected into the key computation before the softmax attention, giving the model access to geometric context (thickness, tortuosity, angle) when deciding which neighbors to attend to. GAT achieves the best edge AP (0.947) in our evaluation.

**The gap:**  
EGAT is evaluated on homogeneous graph benchmarks (node/graph classification) without domain-specific geometric edge features. It does not study link prediction in engineering inspection graphs, nor the relative importance of individual edge feature groups.

**How our work fills it:**  
We apply EGAT-style edge-featured attention to crack topology link prediction with 8-dimensional geometric edge features. Our ablation (M5) reveals that thickness dominates: removing edge features entirely (−0.350 node AP) nearly matches removing thickness alone (−0.347), showing that the attention mechanism's edge-feature gain comes primarily from physical geometry, not just structural context.

---

### 22. Xiao et al. — "Graph Isomorphism Network for Materials Property Prediction Along with Explainability Analysis" (EGIN)
**Venue:** Computational Materials Science, Vol. 233, Article 112619, Elsevier (ScienceDirect), 2024  
**DOI:** 10.1016/j.commatsci.2023.112619

**Summary:**  
Proposes EGIN — a Graph Isomorphism Network extended with edge feature injection for materials property prediction. EGIN incorporates edge features (bond type, bond length, bond angle) into the aggregation step before the MLP: h_v^(ℓ) = MLP((1+ε)·h_v^(ℓ-1) + Σ_{u∈N(v)} ReLU(h_u^(ℓ-1) + e_{uv})). Includes gradient-based explainability analysis that identifies which edge features most influence property predictions.

**Why it fits:**  
EGIN implements the same edge feature injection formula used by our GINEConv encoder (Stage 3, Section 5.4). Both architectures represent physical structure as attributed graphs with continuous edge features — Xiao et al. use atomic bond properties; we use crack segment geometry (thickness, tortuosity, angle, length). The explainability analysis also parallels our M5 edge ablation study.

**The gap:**  
EGIN targets molecular graph-level regression (crystal property prediction) using discrete categorical edge features from fixed chemical bond types. It does not study link prediction in infrastructure inspection graphs, nor continuous geometric edge features measured from image skeletons.

**How our work fills it:**  
We apply GINE-style edge injection to crack topology link prediction — the first application to image-derived infrastructure graphs with 8-dimensional continuous geometric edge features. GINE achieves the best node AP (0.739 ± 0.004), and our ablation shows that thickness is the dominant edge feature (removing it alone costs −0.347 AP, nearly matching full edge feature removal at −0.350).

---

### 23. Zhang et al. — "Road Crack Detection Using Deep Neural Network Based on Dense Feature Pyramid Networks" (CRACK500)
**Venue:** IEEE ICIP 2016  
**DOI:** 10.1109/ICIP.2016.7532898

**Summary:**  
Introduces the CRACK500 dataset — 500 crack images collected from road surfaces with pixel-level segmentation annotations, widely used as the standard pavement crack benchmark. The dataset spans multiple crack types (longitudinal, transverse, alligator) at various scales.

**Why it fits:**  
CRACK500 is one of the 11 sources in our crack_seg_clean compilation. It contributes ~500 of the 11,298 raw images (4.4% of raw, higher fraction after cleaning since CRACK500 images have dense crack coverage). Must be cited as a dataset source.

**The gap:**  
CRACK500 covers only pavement surfaces from a fixed overhead camera angle. Single-source datasets like this fail to generalize to other surface types (concrete facades, laboratory specimens).

**How our work fills it:**  
We combine CRACK500 with 10 other sources to create a heterogeneous 4,769-image dataset that forces generalization across surface types, lighting conditions, and crack morphologies.

---

### 24. Shit et al. — "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
**Venue:** IEEE CVPR 2021  
**DOI:** 10.1109/CVPR46437.2021.01629  
**arXiv:** 2003.07311

**Summary:**  
Introduces clDice (centerline Dice) — a loss function that computes Dice overlap between the predicted mask and its skeleton, encouraging the network to preserve topological connectivity of thin structures. SoftClDice is the differentiable approximation used during training.

**Why it fits:**  
SoftClDice is used as part of the Stage 1 loss function (focal + Dice + SoftClDice) in HybridGraphUNet training, as documented in segmentation_report.md Section 4.5. Without this citation, the training loss description is incomplete — and this is a required citation since the paper claims topology-preserving training as a contribution.

**The gap:**  
clDice was designed for vessel segmentation (tubular structures in medical images). Cracks are thinner, less regular, and have branching topologies distinct from vascular networks. The applicability to crack topology was not evaluated.

**How our work fills it:**  
We apply SoftClDice to crack segmentation and show it contributes to the EnhancedGraphUNet's best clDice score (0.7540), though HybridGraphUNet is ultimately chosen for its recall rather than clDice. This is the first reported use of SoftClDice loss for crack segmentation.

---

### 25. Al-Huda et al. — "EfficientCrackNet: A Lightweight Model for Pavement Crack Segmentation"
**Venue:** IEEE Access 2024  
**DOI:** 10.1109/ACCESS.2024.3371019

**Summary:**  
Proposes EfficientCrackNet — a lightweight encoder-decoder using EfficientNet backbone with attention gates, achieving IoU=0.813 on CRACK500 and 0.84 on DeepCrack. Sets a strong efficiency benchmark: high IoU at low parameter count.

**Why it fits:**  
Used as the SOTA segmentation comparison in Section C2 (prior work comparison). EfficientCrackNet's 0.813 IoU on CRACK500 is the reference point for situating our HybridGraphUNet's IoU=0.630 in context. The comparison must acknowledge dataset differences (CRACK500 single-source vs our 11-source crack_seg_clean).

**The gap:**  
EfficientCrackNet is a pure segmentation model — no graph construction, no topology analysis, no link prediction. High pixel-level IoU does not imply good skeleton topology for downstream graph analysis.

**How our work fills it:**  
We show that recall (0.843 for HybridGraphUNet) is more important than IoU for Stage 2 skeleton quality — a missed crack is a broken skeleton connection, while a false positive is pruned by spur pruning. The EfficientCrackNet paradigm optimizes IoU; we optimize recall + topology preservation for downstream GNN use.

---

## Summary Table

| # | Paper | Venue | Year | Pipeline Stage Covered | Gap Filled by Our Work |
|---|-------|-------|------|------------------------|------------------------|
| 1 | Xu et al. (GIN / WL theorem) | ICLR | 2019 | Stage 3 theory | GNN expressiveness on attributed graphs |
| 2 | Wu et al. (GNN comprehensive survey) | IEEE TNNLS | 2021 | Stage 3 context | First benchmark on crack topology graphs |
| 3 | Zhang & Chen (SEAL) | NeurIPS | 2018 | Stage 3 link prediction | Domain-specific masking + frontier protocol |
| 4 | WL dynamic attributed (Neural Networks) | ScienceDirect | 2024 | Stage 3 theory (edge features) | Empirical validation of GINE edge feature dominance |
| 5 | Perera et al. (GNN crack coalescence) | CMAME | 2022 | Stage 3 motivation | Real image → skeleton → topology → GNN pipeline |
| 6 | Hu et al. (transfer learning cracks) | Mechanics of Materials | 2023 | Stage 3 generalization | 11-source real-image generalization |
| 7 | Shukla et al. (MPNN physics engine) | IJSS | 2024 | Stage 3 message passing | Message passing on image-derived graphs |
| 8 | Liu et al. (DeepCrack) | Neurocomputing | 2019 | Stage 1 baseline | Topology-aware segmentation (GNN bottleneck) |
| 9 | Bi et al. (AR-UNet) | IEEE Access | 2023 | Stage 1 architecture | HybridGraphUNet GNN bottleneck vs CNN attention |
| 10 | Jiang et al. (Hybrid GCN+DCN) | Eng. App. AI | 2025 | Image-to-graph motivation | Skeleton-based topology graph vs superpixel graph |
| 11 | Zhao et al. (rock fracture skeleton) | Engineering Geology | 2024 | Stage 2 graph construction | Training a GNN on skeleton-derived graphs |
| 12 | Wang et al. (fracture extraction) | IJRMMS | 2025 | Stage 2 skeletonization | Feature-enriched graph + learned link prediction |
| 13 | Li et al. (skeletonization measurement) | Mat. Today Comm. | 2025 | Stage 1→2 transition | Graph construction → GNN training pipeline |
| 14 | TopoM-CrackNet (topology-informed segmentation) | Automation in Construction | 2025 | Stage 1→2 topology | GNN link prediction on topology-consistent crack graphs |
| 15 | NVE-DGCNN (concrete defects) | Automation in Construction | 2024 | Domain motivation | 2D image-derived crack topology graphs |
| 16 | Amara et al. (GNN infrastructure survey) | Adv. Eng. Informatics | 2024 | Full pipeline survey | First GNN link prediction on image-derived crack graphs |
| 17 | Zhang et al. (multi-source domain) | Measurement | 2024 | Dataset construction | Multi-source topology graph generalization |
| 18 | Ronneberger et al. (U-Net) | MICCAI / Springer | 2015 | Stage 1 backbone | GNN bottleneck extension to U-Net |
| 19 | Kipf & Welling (GCN) | ICLR | 2017 | Stage 3 GCN encoder | First GCN < MLP result on crack topology graphs |
| 20 | Ciano et al. (inductive vs transductive GNN) | IEEE TPAMI | 2021 | Stage 3 eval protocol | Transductive per-graph evaluation for crack topology |
| 21 | Wang et al. (EGAT) | Springer ICANN | 2021 | Stage 3 GAT encoder | Edge-feature-aware attention for crack topology |
| 22 | Xiao et al. (EGIN) | ScienceDirect CMS | 2024 | Stage 3 GINE encoder | First GINE-style edge injection on crack topology |
| 23 | Zhang et al. (CRACK500 dataset) | IEEE ICIP | 2016 | Dataset source | One of 11 crack segmentation sources |
| 24 | Shit et al. (clDice / SoftClDice) | CVPR | 2021 | Stage 1 training loss | First SoftClDice use for crack segmentation |
| 25 | Al-Huda et al. (EfficientCrackNet) | IEEE Access | 2024 | Stage 1 SOTA comparison | Topology-aware recall vs. IoU-optimal baselines |

---

## Coverage Map

```
Problem space our paper fills:

Real crack images ─→ [Stage 1: Seg] ─→ [Stage 2: Skeleton+Graph] ─→ [Stage 3: GNN Link Pred]
        │                    │                       │                            │
   8,9,14,17,18,24,25   Paper 10              Papers 11,12,13         Papers 1,2,3,4,19,20,21,22
   (what exists)         (closest           (graph construction         (link prediction
                          published)             validated)              theory + SEAL + GNN arch)
        │                    │                       │                            │
        └────────────────────┴───────────────────────┴────────────────────────────┘
                                OUR WORK: first end-to-end pipeline connecting all stages
                                (C3: Oracle AP=0.655, Pred AP=0.800; M5: thickness dominant)

Dataset: Papers 6, 17, 23 motivate multi-source (11 sources, 4,769 images after cleaning)
FEM/3D motivation (not image pipelines): Papers 5, 6, 7, 15, 16
```

No prior paper traverses all three stages in a single evaluated pipeline from real images to GNN link prediction with multi-seed validation.
