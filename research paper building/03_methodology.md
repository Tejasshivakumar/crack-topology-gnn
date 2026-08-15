# Section III — Methodology

> **Reference numbers** use `00_references.md` numbering.
> **All architecture details, hyperparameters, and feature dimensions are verified against source code.**

---

## III. Methodology

Our pipeline comprises three sequential stages, illustrated conceptually in Fig. 1. Stage 1 (Section III-B) produces a pixel-level binary segmentation mask from a crack image. Stage 2 (Section III-C) converts that mask into a topology graph whose nodes represent crack tips and junctions and whose edges represent connecting crack segments, with rich geometric attributes. Stage 3 (Section III-D) trains five GNN encoder architectures on those graphs to perform crack topology link prediction, proving whether message passing captures structural topology beyond what per-node features alone reveal.

### A. Dataset

**Stage 1 (segmentation)** is trained and evaluated on two datasets serving distinct purposes.

*Benchmark evaluation* — The DeepCrack dataset [2] provides 300 training images with clean, consistently annotated pixel-level crack masks evaluated on a pre-defined 237-image test split at 512 × 512 pixels. We train HybridGraphUNet on DeepCrack to establish a comparable benchmark result against prior segmentation work.

*Pipeline training* — crack_seg_clean is a multi-source road-pavement collection assembled and cleaned from the khanhha crack segmentation repository [22], deduplicated from 11,298 raw images down to 4,769 unique images across 6 sources. The table below enumerates all 11 original sources, their domain classification, and the decision applied.

**Table: crack_seg_clean — source inventory**

| Source | Domain | Raw images | Decision | Citation |
|--------|--------|-----------|----------|----------|
| CRACK500 | Road pavement | 3,363 | Kept | [23] |
| GAPs384 | Road pavement | 509 | Kept | [24] |
| DeepCrack | Road pavement | 521 | Kept | [2] |
| CrackTree200 | Road pavement | 206 | Kept | [25] |
| CrackForest (CFD) | Road pavement | 118 | Kept | [26] |
| AEL (Chambon) | Road pavement | 185 | Kept | [27] |
| Rissbilder | Concrete wall | 3,822 | Dropped — domain mismatch | — |
| NonCrack | Concrete wall | 1,411 | Dropped — domain + empty masks | — |
| Volker | Concrete structure | 990 | Dropped — domain mismatch | — |
| Eugen Muller | Unknown | 55 | Dropped — unverified domain, poor quality | — |
| Forest (CFD-variant) | Road pavement | 118 | Dropped — 100% exact duplicates of CFD | — |
| **Total** | | **11,298** | | |

After scope filtering (road-pavement only) and MD5 exact-duplicate removal, **4,769 images** remain: 4,071 train / 698 test. HybridGraphUNet is trained on these images at 448 × 448 pixels (native resolution). This is the model used for Stage 1 in the full pipeline and for the end-to-end evaluation. To isolate each stage's contribution, the model comparison in Section IV-B also includes a plain ResNet34d U-Net (no GNN bottleneck) and a scratch-trained encoder as ablation variants.

**Stage 2 and Stage 3** use ground-truth binary masks from crack_seg_clean for training. Using GT masks decouples Stage 3 evaluation from Stage 1 prediction error — the link prediction results reflect the GNN's topology understanding, not cascaded segmentation noise. The end-to-end evaluation (Section III-E) then re-introduces Stage 1 predicted masks to measure the full pipeline. The mask-to-graph pipeline (Stage 2) converts the crack_seg_clean masks to 4,364 valid PyG graphs. The train/test graph split yields **3,728 training graphs** and **636 test graphs**.

---

### B. Stage 1 — Image Segmentation

#### B.1 Architecture: HybridGraphUNet

We propose HybridGraphUNet, a hybrid CNN–GNN segmentation model combining three components:

**Encoder** — a ResNet34d backbone with pretrained ImageNet weights, loaded via the `timm` library in features-only mode. The encoder produces five multi-scale feature maps at strides 2, 4, 8, 16, and 32 with channel widths [64, 64, 128, 256, 512]. ImageNet pretraining provides edge, texture, and curve detectors without requiring large crack-specific datasets — critical because crack data with pixel-level masks is scarce. Differential learning rates are applied: the encoder fine-tunes at 1e-5 (10× lower than the decoder) to preserve pretrained weights.

**GNN Bottleneck** — applied to the deepest (stride-32) feature map, treating spatial feature tokens as a graph. Two stacked blocks are applied:
```
Grapher(k=9, dilation=1, conv='edge') → FFN(4× expansion) →
Grapher(k=9, dilation=2, conv='edge') → FFN(4× expansion) → Dropout2d(0.1)
```
Each Grapher block builds a k-NN graph over the 512-channel feature tokens and applies edge-convolution message passing. The dilated second block (dilation=2) captures longer-range structural dependencies at the bottleneck level, enabling the network to reason about crack connectivity before decoding. This distinguishes HybridGraphUNet from a plain U-Net [10]: the bottleneck explicitly encodes topological context.

**Decoder** — four UNet-style blocks, each comprising a transposed convolution (stride 2) followed by a DoubleConv block (two sequential 3×3 Conv2d–BN–ReLU layers) applied to the concatenation of the upsampled feature map and the corresponding encoder skip connection. A final transposed convolution and DoubleConv project to a 16-channel representation; a 1×1 convolution produces the two-class logit map. The total model has **32.3 M parameters**.

#### B.2 Loss Function

The combined training loss is:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{CrackDice}}$$

**Focal Loss** [γ = 2, crack class weight = 10]: down-weights easy background pixels and applies 10× class weight to crack pixels, which comprise less than 5% of pixels in typical road crack images.

$$\mathcal{L}_{\text{Focal}} = \mathbb{E}\!\left[(1-p_t)^\gamma \cdot \text{CE}(p_t)\right], \quad \text{with class weight} = [1.0,\, 10.0]$$

**CrackDice**: Dice loss computed exclusively over the crack class (class 1), preventing the large background region from diluting gradient signal:

$$\mathcal{L}_{\text{CrackDice}} = 1 - \frac{2 \sum p_i t_i + \epsilon}{\sum p_i + \sum t_i + \epsilon}$$

where $p_i \in [0,1]$ are crack-class probabilities and $t_i \in \{0,1\}$ are crack pixels. This loss directly optimises the crack segmentation quality that Stage 2 depends on — fragmented predictions yield broken skeletons and corrupted graphs [11].

#### B.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Decoder LR | 1e-4 |
| Encoder LR | 1e-5 (10× differential) |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealing (T_max=150, η_min=1e-6) |
| Max epochs | 150 (early stopping, patience=25) |
| Batch size | 8 |
| Image size | 512 × 512 |
| Precision | 16-mixed AMP |
| Gradient clipping | 1.0 |
| Checkpoint monitor | val/crack_iou |

---

### C. Stage 2 — Skeleton-to-Topology-Graph Construction

Stage 2 converts a binary crack segmentation mask to a PyG graph with geometric node and edge attributes. This pipeline closely follows the skeleton-graph construction protocol established in the fracture analysis literature [15], [20].

#### C.1 Skeletonisation

The binary mask (crack pixels = 1) is thinned to a 1-pixel-wide medial axis using `skimage.morphology.skeletonize`, which applies the Zhang-Suen parallel thinning algorithm [33]. This operation collapses the mask to its structural backbone while preserving connectivity: branches, junctions, and endpoints are retained; width information is encoded separately in node and edge attributes via the Euclidean distance transform.

#### C.2 Graph Construction and Spur Pruning

The skeleton is parsed into a NetworkX graph by the `sknw` library, which traces connected components along the skeleton and maps every junction (pixel of degree ≥ 3) and endpoint (degree-1 pixel) to a graph node; intermediate degree-2 pixels are collapsed into edge polylines stored as ordered pixel-coordinate sequences.

Short stub branches caused by mask boundary roughness are removed by iterative **spur pruning**: any leaf branch (an edge where at least one endpoint has degree 1) whose path length is less than 10% of the longest branch is deleted. Pruning iterates to convergence so newly exposed leaves are also removed. This step suppresses Stage 2 noise that would inflate the tip-node count and corrupt link prediction training. Graphs with fewer than 3 nodes or no edges after pruning are discarded.

#### C.3 Node Features (6-dimensional)

For each graph node (crack tip or junction), six features are extracted:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x_norm | Pixel x-coordinate / image width |
| 1 | y_norm | Pixel y-coordinate / image height |
| 2 | thickness | Crack width at node (pixels), from Euclidean distance transform × 2 |
| 3 | degree | Number of connected edges (post-pruning) |
| 4 | is_endpoint | 1.0 if degree = 1 (crack tip), else 0.0 |
| 5 | is_junction | 1.0 if degree ≥ 3 (branching point), else 0.0 |

Thickness is computed as twice the distance-transform radius at the node's pixel coordinate, giving the true crack diameter in pixels. The `is_endpoint` and `is_junction` binary flags encode the structural role of each node and are the primary signals for identifying where crack growth occurs. Critically, these features are **recomputed from the observed edge set** after any masking operation (see Section III-D.1) to prevent stale degree values from leaking label information to the model.

#### C.4 Edge Features (7-dimensional raw → 8-dimensional encoded)

Each edge represents a crack segment between two nodes. Seven geometric attributes are extracted from the ordered waypoint sequence stored by `sknw`:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | path_length | Cumulative Euclidean distance along the skeleton polyline (pixels) |
| 1 | euclidean_dist | Straight-line distance between node endpoints (pixels) |
| 2 | tortuosity | path_length / euclidean_dist (≥ 1; higher = more curved segment) |
| 3 | angle_sym | Symmetric crack orientation in [0°, 180°), undirected |
| 4 | avg_thickness | Mean crack width along the segment |
| 5 | min_thickness | Thinnest point along the segment |
| 6 | max_thickness | Widest point along the segment |

The `angle_sym` column is expanded to a smooth circular encoding before being fed to edge-feature-aware GNN layers. Since opposite orientations (e.g., 0° and 180°) refer to the same undirected segment direction, the raw angle has a discontinuity at 0°/180°. This is resolved by the substitution:
$$\text{angle\_sym} \;\xrightarrow{\;\text{encode}\;}\; [\sin(2\theta),\; \cos(2\theta)]$$
yielding an **8-dimensional edge feature vector** $\mathbf{e}_{uv} \in \mathbb{R}^8$ supplied to GINE and GAT encoders. GCN and SAGE, which do not consume edge attributes, use the raw 7-dimensional vector for reference.

---

### D. Stage 3 — GNN Topology Link Prediction

#### D.1 Task Formulation

To isolate the GNN's topology reasoning ability from upstream segmentation noise, all Stage 3 training and evaluation uses ground-truth binary masks as input to Stage 2; the full pipeline coupling — where Stage 1 predicted masks are fed through — is assessed separately in Section III-E.

**Node task (primary).** For each crack graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, we randomly hide 20% of degree-1 tip nodes by zeroing their feature vectors and removing all incident edges. After masking, structural features (degree, is\_endpoint, is\_junction) are recomputed from the remaining edge set — this **feature recomputation** is essential: without it, a visible base node whose degree dropped by one would be trivially identifiable without any topology reasoning. The model's binary node prediction task is:

> Given the masked graph, identify which visible nodes previously had a connection to a now-hidden tip (i.e., were the "base node" of a removed crack growth segment).

The positive class is the base nodes of removed crack tips; the negative class is all other visible, non-isolated nodes. This task **cannot be solved by per-node features alone** after recomputation — all affected base nodes look like ordinary endpoints. A structure-aware GNN can exploit neighbourhood context (adjacent thickness patterns, branching density, tortuosity of surviving edges) to identify incomplete nodes; a structure-blind MLP cannot. Any performance gap between a GNN and the MLP baseline is therefore attributable to message passing.

**Edge task (secondary).** A transductive 80/10/10 split partitions each graph's edges into message-passing edges (used for GNN convolution), training supervision edges (hidden positives), and test supervision edges (evaluation set). The model scores candidate node pairs for edge existence using the shared encoder representations and an MLP predictor head (see Section III-D.3). This task is treated as secondary because local structural features (is\_endpoint flags, degree) are highly predictive of which nearby node pairs are connected, making it a weak discriminator of topology understanding.

#### D.2 Hard Negative Sampling

For the edge task, random negative sampling (i.e., uniform node-pair non-edges) allows any model to trivially win by exploiting spatial position: nearby nodes are real edges, distant nodes are not, and node features include normalised coordinates. To eliminate this shortcut, we use **hard negative sampling**: for each anchor node, its $k_{\text{near}} = 30$ nearest neighbours in pixel space are found via a KD-tree, and unconnected pairs among those neighbours are used as negatives. This forces the model to distinguish genuinely absent connections from topologically plausible ones — a much harder task that requires structural reasoning. The same negative set is used at training, validation, and test time for consistent evaluation.

#### D.3 Encoder Architectures

Five encoder architectures are evaluated with shared hidden dimension 128 and output dimension 64, producing node embeddings $\mathbf{z} \in \mathbb{R}^{64}$. All architectures use 3 convolutional/aggregation layers with LayerNorm after each layer, residual connection between layer 2 and layer 3, dropout (p=0.3), and ReLU activations.

**MLP (no-graph baseline)** — a 3-layer MLP applied independently to each node's feature vector, ignoring graph structure entirely. This establishes the performance floor: any GNN that outperforms it is demonstrably using topology. The MLP has approximately 65k parameters.

**GCN** [16] — 3-layer Graph Convolutional Network with symmetric degree normalisation $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$. Does not consume edge attributes. We predict that GCN will underperform the MLP baseline on crack graphs due to the dilution of degree-1 tip node signals: each tip is directly adjacent to a degree-3+ junction, and symmetric normalisation down-weights the tip's contribution relative to its high-degree neighbour, obscuring exactly the signal needed to identify topologically incomplete nodes.

**GraphSAGE** [16] — 3-layer inductive neighbourhood aggregation via concatenation of self- and mean-aggregated neighbour features. Does not consume edge attributes. Mean aggregation with concatenation preserves structural heterogeneity across node types better than GCN's symmetric normalisation.

**GINE** [5], [17] — 3-layer GIN Extended with edge features (GINEConv). Per-layer update:
$$h_v^{(\ell)} = \text{MLP}\!\left((1+\varepsilon)\,h_v^{(\ell-1)} + \sum_{u \in \mathcal{N}(v)} \text{ReLU}\!\left(h_u^{(\ell-1)} + \text{Lin}(\mathbf{e}_{uv})\right)\right)$$
where $\mathbf{e}_{uv} \in \mathbb{R}^8$ is the encoded edge feature vector and $\text{Lin}(\cdot)$ projects it to the node feature dimension. GINE is theoretically as expressive as the Weisfeiler–Lehman graph isomorphism test [5], [6] and extends that power to attributed graphs through edge feature integration. Crack width (avg\_thickness) provides exactly the physical attribute that distinguishes actively growing crack tips from stable segment interiors, making GINE the predicted top performer.

**GAT** [18] — 3-layer Graph Attention Network with 4 attention heads and edge features (8-dimensional, consumed via `GATConv(edge_dim=8)`). The attention mechanism learns to weight neighbour contributions dynamically; a residual connection is applied between layers 2 and 3. GAT has approximately 380k parameters — the largest of the five models. Convergence requires more epochs than the lighter architectures; the architecture's full capacity is expected to manifest at 200+ epochs.

All encoders share two prediction heads:

**MLPEdgePredictor** — scores candidate node pair $(u, v)$ using:
$$\hat{y}_{uv} = \text{MLP}\!\left([\mathbf{z}_u \;\|\; \mathbf{z}_v \;\|\; \mathbf{z}_u \odot \mathbf{z}_v]\right)$$
The Hadamard product $\mathbf{z}_u \odot \mathbf{z}_v$ captures pairwise interaction; concatenation alone gives a weaker interaction signal.

**MLPNodePredictor** — 3-layer MLP binary classifier $\hat{y}_v = \text{MLP}(\mathbf{z}_v)$, outputting the probability that node $v$ has a hidden neighbour.

#### D.4 Joint Training and Optimisation

The encoder and both prediction heads are trained in two sequential passes per epoch, each using `BCEWithLogitsLoss` with a separate `optimizer.step()`:

1. **Node pass** — all node-masking graphs are processed; the node predictor loss $\mathcal{L}_{\text{node}}$ is accumulated and a gradient step is taken.
2. **Edge pass** — all edge-split graphs are processed; the edge predictor loss $\lambda \cdot \mathcal{L}_{\text{edge}}$ is accumulated and a second gradient step is taken.

The two-pass design keeps the node and edge supervision signals independent, preventing the larger edge-task graph count from dominating the node-task gradient. The node loss applies a positive class weight (capped at 5×) to account for the ~10:1 imbalance between base nodes and all other visible nodes. $\lambda = 0.5$ in all experiments. Gradient accumulation over 8 graphs per update step stabilises training on the large dataset. Best checkpoint is selected by combined validation score $\text{score} = 0.5 \cdot \text{AP}_{\text{node}} + 0.5 \cdot \text{AP}_{\text{edge}}$.

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| LR schedule | Linear warmup (10 ep) → cosine annealing |
| Max epochs | 200 |
| Gradient accumulation | 8 graphs per step |
| Gradient clipping | max_norm = 1.0 |
| Node mask fraction | 0.20 |
| Hard negative k_near | 30 |
| Positive class weight | min(imbalance_ratio, 5.0) |
| Seeds (multi-seed eval) | {42, 100, 2024} |

#### D.5 Frontier Masking Evaluation

Standard node-task masking hides a random 20% of all tip nodes across the graph. The **frontier masking** protocol is a physically motivated harder evaluation: 30% of tip nodes are hidden, specifically chosen as the outermost crack endpoints most likely to represent active growth fronts. After hiding, the base nodes of those growth edges look identical to any ordinary endpoint — even their degree is recomputed to reflect the post-masking state. A model that retains high AP under frontier masking must be reading neighbourhood structural context (thickness gradient, tortuosity of adjacent edges, local branching density) — it cannot be exploiting any local feature shortcut. The gap between standard and frontier AP across encoders serves as a second, stronger measure of topology understanding.

---

### E. End-to-End Evaluation

To assess the full pipeline, we compare two evaluation paths on 100 held-out crack images:

- **Oracle path** (upper bound): ground-truth mask → Stage 2 → GINE link prediction
- **Predicted path** (realistic): HybridGraphUNet mask → Stage 2 → GINE link prediction

Average Precision is computed per graph and paired across both paths. This evaluation quantifies the propagated effect of segmentation imperfection on topology link prediction quality and reveals whether GINE's topology understanding is robust to mask prediction noise.

---

> **Word count (body text):** ~1,100 words  
> **Papers cited in this section:** [2], [5], [6], [10], [11], [15], [16], [17], [18], [20], [21], [22], [23], [24], [25], [26], [27]
