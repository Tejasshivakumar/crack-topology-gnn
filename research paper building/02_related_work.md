# Section II — Related Work

> **Reference numbers** use `00_references.md` numbering.

---

## II. Related Work

Whereas our initial review emphasised the direct application of Graph Neural Networks to the link-prediction problem for structural forecasting, the methodology developed here prioritises *topological comprehension*. Rather than forecasting connectivity directly with standard architectures, the proposed pipeline first ensures that the model explicitly understands the physical crack topology; by establishing this accurate geometric foundation, it is better equipped to handle the complex structural dependencies that accurate downstream link prediction requires.

### A. GNN Theoretical Foundations

Graph Neural Networks (GNNs) operate through iterative message passing, where each node aggregates representations from its neighbours to build a structural embedding. Wu et al. [3] provide a comprehensive survey of GNN families — spectral, spatial, and graph auto-encoders — establishing that spatial methods consistently outperform spectral approaches on irregular real-world graphs with non-uniform degree distributions. The expressiveness of message-passing GNNs is formally bounded by the Weisfeiler–Lehman (WL) graph isomorphism test; Beddar-Wiesing et al. [6] extend this bound to attributed graphs, proving that GNNs with edge feature propagation — such as GINE — distinguish structurally non-equivalent nodes with strictly greater power than node-feature-only architectures. This theoretical result is the core justification for including 8-dimensional geometric edge features in our Stage 3 graph representation and selecting GINE as our primary encoder.

For link prediction specifically, Zhang and Chen [7] establish SEAL as the canonical GNN-based framework, demonstrating that local enclosing subgraph structure is sufficient for high-accuracy link prediction on sparse graphs — outperforming all heuristic predictors (Common Neighbours, Jaccard, Katz) by 10–20% AUC. Wang et al. [8] further formalise that structurally-induced negative samples — non-edges between topologically proximate nodes — produce substantially better learned representations than random negatives, because they force the model to discriminate true connectivity from plausible-but-absent connections. This result directly justifies our hard negative sampling strategy (k_near = 30) in Stage 3.

### B. Crack Segmentation and Topology Preservation

The established CNN baseline for crack segmentation is DeepCrack [2], which introduced hierarchical multi-scale feature fusion to achieve pixel-wise precision. While it produces high-quality binary masks, its output has no structural awareness: it cannot distinguish crack tips from junctions, identify propagation paths, or reason about how segments connect. Ronneberger et al. [10] established the encoder–decoder architecture with skip connections that underpins all modern segmentation backbones, including our Stage 1 HybridGraphUNet.

To address the topological fragility of thin crack structures, Shit et al. [11] introduced SoftClDice — a differentiable topology-preserving loss that penalises skeleton connectivity errors rather than pixel overlap. TopoM-CrackNet [13] extends this direction by incorporating persistent homology loss to enforce junction and endpoint consistency during training, empirically demonstrating that topology-aware segmentation is a prerequisite for accurate downstream graph construction. This directly motivates our topology-aware training objective in Stage 1, combining Focal loss with a crack-class-only Dice term (CrackDice) to concentrate gradient signal on crack pixels and preserve segmentation quality for downstream skeletonisation.

Closely related to our Stage 1 design, Singh et al. [28] induce graph-based learning inside a U-Net by constructing a graph over image-feature regions, showing that this UNet–GNN hybrid captures long-range spatial dependencies that convolutional and transformer baselines (U-Net, U-Net++, SwinUNet) miss. Our HybridGraphUNet adopts the same principle, embedding a GNN bottleneck within an encoder–decoder segmentation network.

### C. Image-to-Graph Approaches and Their Limitations

Song and Tian [12] represent the closest published approach to our image-to-graph paradigm: crack images are converted to Region Adjacency Graphs over superpixels, and GCN is applied for crack classification. Djenouri et al. [14] construct graphs using SIFT feature-correlation between image regions for GCN-based detection. Both approaches, however, connect nodes by spatial proximity or visual feature similarity rather than physical structural continuity. Neither identifies crack tips, junctions, or propagation paths — the skeleton-level primitives that are structurally meaningful for link prediction. A node in a superpixel graph encodes a pixel cluster; a node in our topology graph encodes a crack endpoint or branching junction, with degree encoding its structural role.

### D. Skeleton-Based Graph Construction

In the fracture analysis literature, Ji et al. [15] establish that skeleton-based graph construction — segmentation → Zhang-Suen thinning → degree-based node classification (degree-1 tips, degree-3+ junctions) → polyline graph with geometric measurements — is the validated standard method. Their pipeline is structurally identical to our Stage 2. The critical distinction is scope: their graph is a measurement artefact used for counting and geometry statistics, whereas ours is a training input to a GNN link prediction model.

At the domain level, Wettewa et al. [1] survey 111 GNN papers applied to civil infrastructure; no work among those surveyed addresses GNN link prediction on image-derived crack topology graphs. Our pipeline directly fills this gap, providing the first empirical benchmark for the task.

### E. GNN Encoders for Attributed Physical Graphs

Kipf and Welling [16] introduce Graph Convolutional Networks (GCN) with symmetric degree normalisation $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$. While effective on homophilic citation graphs where connected nodes share labels, this normalisation dilutes signals at low-degree nodes when they are structurally adjacent to high-degree nodes — a regime that characterises crack topology graphs, where degree-1 tip nodes (the positive class for link prediction) are directly connected to degree-3+ junctions. This predicts GCN underperformance relative to a no-graph baseline, which our experiments confirm. For edge-featured GNNs, Xiao et al. [17] demonstrate on materials property prediction that the GINE edge injection formula:

$$h_v^{(\ell)} = \text{MLP}\!\left((1{+}\varepsilon)\,h_v^{(\ell-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\!\left(h_u^{(\ell-1)} + e_{uv}\right)\right)$$

substantially improves performance on attributed graphs where edge features carry physical meaning — directly paralleling our use of GINEConv with 8-dimensional crack geometry features. From the FEM simulation domain, Perera et al. [4] apply GNNs to crack tip coalescence prediction, establishing that crack-to-crack interaction is fundamentally a graph topology problem and validating message passing as the right mechanism. However, their graphs are constructed from finite element meshes with manually placed crack tips under controlled physics conditions — no image processing, no real-world noise, no skeletonization.

### F. Future Direction: Positive-Unlabelled Link Prediction

A fundamental challenge in crack propagation forecasting is the Positive-Unlabelled (PU) nature of the observation: connections that exist in the crack graph are confirmed positives, but absent connections are not confirmed negatives — some will form as the material degrades under load. Mao et al. [9] formalise this as a PU-AUC optimisation problem and derive a theoretically grounded objective that outperforms standard binary negative sampling on sparse graphs where the PU assumption holds structurally. Our work establishes the binary-classification performance baseline (GINE node AP: 0.739 ± 0.004) that future PU-AUC work on crack topology graphs will need to surpass — a direct extension path once time-series inspection datasets with contextual metadata (material type, load cycles, climate exposure) become available.

### G. Datasets and the Data Bottleneck

High-fidelity data has been identified as a prerequisite for topology-aware crack modelling. Liu et al. [29] release PaveDistress, a high-resolution pavement-distress dataset intended to capture the fine branching topology of structural failure, while Ramírez-Villanueva et al. [30] introduce PY-CrackDB, a dataset deliberately curated with environmental variation (lighting, shadow, surface texture) to train context-aware models. Rather than optimising for single-dataset resolution, we prioritise source diversity: our pipeline is trained on a curated multi-source pavement collection (Section III-A) drawn from six public datasets, exposing the model to varied acquisition conditions and annotation styles.

---

> **Word count (body text):** ~720 words  
> **Target in final paper:** ~1.0–1.5 IEEE double-column pages  
> **Papers cited in this section:** [1], [2], [3], [4], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [28], [29], [30] — 19 papers
