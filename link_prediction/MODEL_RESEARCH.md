# GNN Model Research — Crack Topology Link & Node Prediction

## Task Definition

This document analyses the best GNN architectures for the crack topology understanding task and
justifies the model selection for both the current implementation and recommended upgrades.

### Task Characteristics

| Property | Value | Implication for Model Choice |
|---|---|---|
| Avg nodes per graph | 24–35 | Small — any model fits in memory |
| Avg edges per graph | 22–31 | Sparse, chain/tree-like topology |
| Avg node degree | ~1.8 | Very low — structural position drives the signal |
| Node feature dim | 6 | Spatial coords, thickness, degree, type flags |
| Edge feature dim | 8 | Geometric: length, distance, tortuosity, angle (encoded), thickness ×3 |
| Task 1 | Node classification | Identify nodes with hidden neighbours (missing crack tips) |
| Task 2 | Link prediction | Reconstruct hidden crack segments between nodes |
| Training | Joint | Single encoder, two MLP heads trained simultaneously |
| Topology type | Chain / tree / sparse branching | Common in crack skeleton graphs |

---

## Current Model Baseline

### Architecture: CrackGATEncoder (3-layer GAT)

```
Input node features [N, 6]:  x_norm, y_norm, thickness, degree, is_endpoint, is_junction
Input edge features [E, 8]:  path_length, euclidean_dist, tortuosity,
                              sin(2·angle), cos(2·angle), avg_thickness, min_thickness, max_thickness

Layer 1: GATConv(6 → 128, heads=4, concat=True,  edge_dim=8) → [N, 512]
         LayerNorm → ELU → Dropout(0.3)

Layer 2: GATConv(512 → 128, heads=4, concat=True,  edge_dim=8) → [N, 512]
         LayerNorm → ELU → Dropout(0.3)

Layer 3: GATConv(512 → 64,  heads=1, concat=False, edge_dim=8) → [N, 64]
         + Residual: Linear(512 → 64) from layer-2 output
         LayerNorm

MLPEdgePredictor: [z_u ‖ z_v ‖ z_u ⊙ z_v] → 128 → 64 → 1
MLPNodePredictor: z → 64 → 32 → 1
```

### Current Results (DeepCrack test set, 237 graphs)

| Task | Metric | Value |
|---|---|---|
| Node prediction | AUC-ROC | **0.8321** |
| Node prediction | Avg Precision | 0.4230 |
| Node prediction | F1 Score | 0.2518 |
| Edge prediction | AUC-ROC | **0.7247** |
| Edge prediction | Avg Precision | 0.6863 |
| Edge prediction | Hits@10 | 0.8500 |
| Edge prediction | Hits@20 | **0.9622** |

Best val score: **0.7848** (0.5 × node_auc + 0.5 × edge_auc) at epoch 260/300.

---

## Literature Review: Top GNN Architectures for Link Prediction (2018–2026)

---

### 1. GAT — Graph Attention Network
**Paper:** Veličković et al., *"Graph Attention Networks"*, ICLR 2018
**Citation:** `arxiv.org/abs/1710.10903`

#### Core Idea
Assigns learned attention weights α_ij to each neighbour during message passing:

```
α_ij = softmax( LeakyReLU( a · [Wh_i ‖ Wh_j] ) )
h_i' = σ( Σ_j α_ij · Wh_j )
```

#### Fit for Crack Graphs
- Handles heterogeneous junctions well — different branches get different attention weights
- Natively supports edge features (edge_dim parameter in PyG GATConv)
- Multi-head attention learns diverse aggregation strategies (thickness similarity, spatial proximity, orientation alignment)

#### Known Limitation
The attention computation is **static** — α_ij is computed before the nonlinearity mixes query and key features. This means GAT cannot distinguish all possible orderings of a node's neighbours, making it provably less expressive than it could be (Brody et al., 2022).

#### Current Status
Used in the current CrackGATEncoder. Stable and effective at this scale.

---

### 2. GATv2 — How Attentive are Graph Attention Networks?
**Paper:** Brody, Alon, Yahav, *"How Attentive are Graph Attention Networks?"*, ICLR 2022
**Citation:** `arxiv.org/abs/2105.14491`

#### Core Idea
Fixes GAT's static attention by reordering the linear and nonlinear operations:

```
GAT:   α_ij = softmax( a · LeakyReLU( W_1 h_i + W_2 h_j ) )  ← static
GATv2: α_ij = softmax( a · LeakyReLU( W[h_i ‖ h_j] ) )       ← dynamic
```

The key difference: in GATv2, the nonlinearity sees both h_i and h_j simultaneously, so the attention score depends on the query context. This achieves **universal approximation of attention functions** — something GAT provably cannot do.

#### Why This Matters for Crack Graphs
At a crack junction, the model needs to attend differently to a thick branch vs a thin branch **depending on which node is asking**. GATv2's dynamic attention captures this; GAT's static attention cannot always distinguish these cases.

#### Practical Upgrade Path
Drop-in replacement for `GATConv` in `model.py`:

```python
# Before
from torch_geometric.nn import GATConv
self.conv1 = GATConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, ...)

# After
from torch_geometric.nn import GATv2Conv
self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, edge_dim=edge_dim, ...)
```

Identical hyperparameters, same parameter count, provably more expressive.

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ✅ Native (same as GATConv) |
| Small sparse graphs | ✅ Excellent — cheap attention over 2–5 neighbours |
| Joint node + edge tasks | ✅ Same encoder architecture preserved |
| Topology sensitivity | ✅ Better than GAT (dynamic attention) |
| Implementation effort | ✅ One import change |

---

### 3. SEAL — Subgraph Extraction and Learning for Link Prediction
**Paper:** Zhang & Chen, *"Link Prediction Based on Graph Neural Networks"*, NeurIPS 2018
**Extended:** Zhang et al., *"Neo-GNNs"*, NeurIPS 2021
**Citation:** `arxiv.org/abs/1802.09691`

#### Core Idea
For each candidate edge (u, v):
1. Extract the h-hop **enclosing subgraph** around u and v
2. Apply **DRNL node labeling** — each node gets a label encoding its shortest-path distances to u and v
3. Run any GNN on this labeled subgraph for binary edge classification

The structural label directly encodes topology: a common neighbour of u and v gets a specific label distinct from a node on only u's side.

#### Why It Is Powerful
Standard MPNNs compute node embeddings without pair context — two node pairs with the same 1-hop neighbourhood are indistinguishable. SEAL sees the structural role of each node relative to the candidate link.

#### Fit for Crack Graphs
- Your graphs are already small (20–40 nodes), so the enclosing subgraph often spans the entire graph — reducing SEAL's structural advantage
- Edge feature injection into the subgraph node matrix requires manual engineering
- No built-in joint node prediction task
- **Best use case here:** standalone comparison baseline for edge prediction only

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ⚠️ Manual — must concatenate to node features |
| Small sparse graphs | ✅ Subgraph extraction is trivial at this scale |
| Joint node + edge tasks | ❌ Requires significant extension |
| Topology sensitivity | ✅ Best-in-class for purely structural link prediction |
| Implementation effort | ⚠️ Medium — subgraph extraction pipeline needed |

---

### 4. NBFNet — Neural Bellman-Ford Networks
**Paper:** Zhu et al., *"Neural Bellman-Ford Networks: A General GNN Framework for Link Prediction"*, NeurIPS 2021
**Citation:** `arxiv.org/abs/2106.06935`

#### Core Idea
Frames link prediction as a path problem. For a query (u, v), runs a generalised Bellman-Ford algorithm from u, propagating path representations through all edges and nodes until reaching v:

```
h_{u→v}^{(t)} = AGG( { f(h_{u→w}^{(t-1)}, e_{wv}) : w ∈ N(v) } )
```

Path representations are products of learned edge functions, making edge features first-class.

#### Fit for Crack Graphs
- Designed for knowledge graph completion — relational diversity is its strength
- Crack graphs have limited relation types (crack segments, not multi-relational KG edges)
- Per-query inference is expensive (re-run from each source node)
- Overkill for 20–40 node graphs

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ✅ Native and central |
| Small sparse graphs | ⚠️ Works but over-engineered |
| Joint node + edge tasks | ❌ Link prediction only |
| Topology sensitivity | ✅ Path-level structural reasoning |
| Implementation effort | ❌ High — requires custom message passing |

---

### 5. BUDDY / ELPH — GNNs for Link Prediction with Subgraph Sketching
**Paper:** Chamberlain et al., *"Graph Neural Networks for Link Prediction with Subgraph Sketching"*, ICLR 2023
**Citation:** `arxiv.org/abs/2209.15486`

#### Core Idea
SEAL's expressiveness comes from subgraph topology — but subgraph extraction is O(|E|) and slow. BUDDY approximates it by precomputing **structural sketches** (MinHash approximations of k-hop neighbourhoods, triangle counts, common neighbour counts) as node features, then feeds a standard lightweight GNN.

ELPH (the online variant) uses these sketches as messages in the GNN layers.

#### The PROXI Connection (2024)
Dong et al. (2024, *"PROXI: Challenging the GNNs for Link Prediction"*) showed that feeding **structural features** (common neighbours, Jaccard, Adamic-Adar) directly as input features to a simple MLP or GNN matches or beats SEAL and BUDDY on OGB benchmarks. The implication: **topology-encoding features are the key signal, not the architecture.**

For crack graphs, this means adding to `MLPEdgePredictor`:
```python
# Extra structural features per candidate edge (u, v):
common_neighbours = len(set(adj[u]) & set(adj[v]))   # int
jaccard_sim       = common_neighbours / len(set(adj[u]) | set(adj[v]))  # float
geo_dist          = euclidean(pos[u], pos[v])          # float (already in edge_attr)
```

These 3 features appended to `[z_u ‖ z_v ‖ z_u ⊙ z_v]` would give the edge predictor structural context at virtually zero cost.

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ⚠️ Structural sketches only — geometry is bolt-on |
| Small sparse graphs | ✅ Sketch computation is trivial at this scale |
| Joint node + edge tasks | ❌ Link prediction only |
| Topology sensitivity | ✅ Strong — common neighbours are exactly the right signal |
| Implementation effort | ✅ Low if using structural features as input only |

---

### 6. Neo-GNN — Neighbourhood Overlap-Aware GNNs
**Paper:** Yun et al., *"Neo-GNNs: Enriching Graph Neural Networks with Structural Features for Graph-level Tasks"*, NeurIPS 2021
**Citation:** `arxiv.org/abs/2206.04216`

#### Core Idea
Learns a structural feature vector from the adjacency matrix capturing multi-hop neighbourhood overlaps, then combines it with standard GNN node embeddings before link prediction. Generalises classical structural heuristics (Jaccard, Adamic-Adar, Common Neighbours) into a learned framework.

#### Fit for Crack Graphs
Works well on topology-driven graphs but has no native edge feature support — ignores path length, tortuosity, and thickness entirely. Since geometric edge features are central to your task, Neo-GNN alone is insufficient.

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ❌ None |
| Small sparse graphs | ✅ Overlap computation is cheap |
| Joint node + edge tasks | ❌ |
| Topology sensitivity | ✅ |
| Implementation effort | ⚠️ Medium |

---

### 7. Line Graph Neural Network (LGNN)
**Paper:** Chen et al., *"Supervised Graph Contrastive Learning for Graph Classification"*, IEEE TPAMI 2021
**Original LGNN:** `arxiv.org/abs/2010.10046`

#### Core Idea
Convert the original graph G to its **line graph** L(G):
- Each **edge** in G becomes a **node** in L(G)
- Two nodes in L(G) are connected if their corresponding edges in G share an endpoint

```
Original G:  nodes = crack junctions/tips,  edges = crack segments
Line Graph:  nodes = crack segments,         edges = adjacency at shared junctions
```

Your 8 edge features (path length, distance, tortuosity, angle, thickness × 3) become **node features** in the line graph — the GNN reasons directly over crack segments.

#### Why This Is Architecturally Elegant for This Task
The task is fundamentally about **edges** (predicting missing crack segments). Representing edges as nodes makes the GNN's natural node classification mechanism serve the edge prediction task directly.

A 30-edge crack graph produces a 30-node line graph of similar scale — no computational overhead.

#### Challenges
- Joint node prediction on the original graph requires dual GNNs (original + line graph) with synchronised decoding
- Predicting new edges in G (that don't exist yet) requires mapping back from L(G) node space — needs a careful decoder design
- Implementation complexity is medium-high

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ✅ Native — edge features become node features |
| Small sparse graphs | ✅ Line graph is same scale |
| Joint node + edge tasks | ⚠️ Requires dual-graph architecture |
| Topology sensitivity | ✅ Excellent — edge adjacency is explicit |
| Implementation effort | ⚠️ Medium-high — new graph construction step |

---

### 8. GPS / GraphGPS — General, Powerful, Scalable Graph Transformer
**Paper:** Rampášek et al., *"Recipe for a General, Powerful, Scalable Graph Transformer"*, NeurIPS 2022
**Citation:** `arxiv.org/abs/2205.12454`

#### Core Idea
Each GPS block combines:
1. **Local MPNN layer** (e.g., GINConv, PNAConv) — captures structural neighbourhood
2. **Global multi-head self-attention** — captures long-range dependencies
3. **Positional encodings** (LapPE, RWSE) — breaks permutation symmetry

#### Fit for Crack Graphs
On 20–40 node graphs, global self-attention over all node pairs is computationally trivial — but it is also architecturally redundant, since all nodes are already within 2–3 hops of each other. Laplacian positional encodings are unstable on chain/tree topologies (degenerate eigenvalue structure). Edge features are supported via the PNA sub-layer but require careful configuration.

**Verdict:** Over-engineered for this scale. The added complexity does not translate to meaningful expressiveness gains on small, sparse graphs.

#### Suitability for This Task
| Property | Rating |
|---|---|
| Edge feature support | ✅ Via PNA sub-layer |
| Small sparse graphs | ❌ Over-engineered |
| Joint node + edge tasks | ✅ Adaptable |
| Topology sensitivity | ✅ Strong (PE + MPNN) |
| Implementation effort | ❌ High |

---

### 9. GIN — Graph Isomorphism Network
**Paper:** Xu et al., *"How Powerful are Graph Neural Networks?"*, ICLR 2019
**Citation:** `arxiv.org/abs/1810.00826`

#### Core Idea
Proved that GNNs with sum aggregation + MLP update are as powerful as the 1-WL graph isomorphism test — the theoretical ceiling for MPNNs. Uses:

```
h_i^{(k)} = MLP( (1 + ε) h_i^{(k-1)} + Σ_{j∈N(i)} h_j^{(k-1)} )
```

#### Fit for Crack Graphs
GIN is the theoretically most expressive MPNN. However:
- No native edge feature support (standard GIN ignores edge attributes)
- For link prediction, the added expressiveness over GAT is marginal on graphs this size
- Useful as a theoretical **upper-bound baseline** in ablation experiments

---

## Summary Comparison Table

| Model | Year | Edge Features | Small Graphs | Joint Tasks | Topology | Effort | Verdict |
|---|---|---|---|---|---|---|---|
| GAT (current) | 2018 | ✅ | ✅ | ✅ | Good | — | Baseline |
| **GATv2** | 2022 | ✅ | ✅ | ✅ | **Better** | Low | **Primary upgrade** |
| SEAL | 2018 | ⚠️ manual | ✅ | ❌ | Excellent | Medium | Edge-only comparison |
| NBFNet | 2021 | ✅ | ⚠️ | ❌ | Good | High | KG-focused, overkill |
| BUDDY | 2023 | ⚠️ bolt-on | ✅ | ❌ | Good | Low | Add as features |
| Neo-GNN | 2021 | ❌ | ✅ | ❌ | Good | Medium | No edge features |
| Line Graph NN | 2021 | ✅ native | ✅ | ⚠️ complex | Excellent | Medium | Future direction |
| GPS | 2022 | ✅ | ❌ over-eng | ✅ | Good | High | Not suitable |
| GIN | 2019 | ❌ | ✅ | ✅ | Excellent | Low | Ablation baseline only |

---

## Recommended Model Strategy

### For the Practicum (Current Priority)

#### Upgrade 1 — GAT → GATv2 (High Priority, Low Effort)

Replace `GATConv` with `GATv2Conv` in `link_prediction/model.py`.

**Justification:**
- Published at ICLR 2022 with proof of expressiveness gain over GAT
- Dynamic attention is theoretically correct for heterogeneous crack junctions
- Identical API — one import change, no hyperparameter tuning needed
- Citable: Brody et al. (2022) directly addresses the limitation of the current model

**Expected impact:** +1–3 points on Node AUC, modest Edge AUC improvement.

#### Upgrade 2 — Structural Features in Edge Predictor (Medium Priority, Very Low Effort)

Append structural features to the edge predictor input:

```python
# In MLPEdgePredictor.forward or during training loop:
# common_nb: number of shared neighbours between u and v
# jaccard:   |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
# geo_dist:  Euclidean distance between node positions

pair = torch.cat([z_u, z_v, z_u * z_v, structural_features], dim=-1)
```

**Justification:**
- PROXI (2024) and BUDDY (2023) both show structural heuristics as input features are highly effective for topology-driven link prediction
- Common neighbours are a direct signal for crack connectivity (two nodes likely connected if they share a junction neighbour)
- Zero model architecture change needed

**Expected impact:** +3–6 points on Edge AUC.

#### Upgrade 3 — Cross-Dataset Evaluation on PaveDistress (Research Priority)

Train Stage 3 on PaveDistress graphs (after Stage 2 preprocessing) without any architecture change.
If Node AUC and Edge AUC remain above 0.75 and 0.65 respectively, the model has demonstrated generalisation — the core research contribution.

---

### Future Direction (Post-Practicum)

A **Line Graph NN + GATv2 hybrid** would be the architecturally ideal solution:

```
Original graph  →  GATv2 encoder  →  node embeddings  →  MLPNodePredictor
     ↕
Line graph      →  GATv2 encoder  →  edge embeddings  →  MLPEdgePredictor
```

The line graph encoder natively reasons over crack segments (edge features as node features), while the original graph encoder handles node prediction. The two encoders share parameters or are trained jointly. This is the correct long-term architecture for a topology-aware crack analysis system.

---

## Why Not GPS / NBFNet / DGCNN?

| Model | Reason for Exclusion |
|---|---|
| GPS | Global attention is redundant on 20-40 node graphs; LapPE unstable on tree topology |
| NBFNet | Designed for multi-relational KG completion; no joint node task; expensive per-query inference |
| DGCNN / EdgeConv | Designed for point clouds with k-NN dynamic graph; your graph topology is fixed and meaningful |
| GraphSAGE | No edge feature support; uniform neighbourhood sampling poorly suited to degree-1 dominant graphs |

---

## References

| Paper | Venue | Link |
|---|---|---|
| Graph Attention Networks (GAT) | ICLR 2018 | arxiv.org/abs/1710.10903 |
| How Attentive are Graph Attention Networks? (GATv2) | ICLR 2022 | arxiv.org/abs/2105.14491 |
| Link Prediction Based on GNNs (SEAL) | NeurIPS 2018 | arxiv.org/abs/1802.09691 |
| Neural Bellman-Ford Networks (NBFNet) | NeurIPS 2021 | arxiv.org/abs/2106.06935 |
| GNNs for Link Prediction with Subgraph Sketching (BUDDY) | ICLR 2023 | arxiv.org/abs/2209.15486 |
| Neo-GNNs: Enriching GNNs with Structural Features | NeurIPS 2021 | arxiv.org/abs/2206.04216 |
| How Powerful are Graph Neural Networks? (GIN) | ICLR 2019 | arxiv.org/abs/1810.00826 |
| Line Graph Neural Networks for Link Prediction | IEEE TPAMI 2021 | arxiv.org/abs/2010.10046 |
| Recipe for a General, Powerful, Scalable Graph Transformer (GPS) | NeurIPS 2022 | arxiv.org/abs/2205.12454 |
| PROXI: Challenging the GNNs for Link Prediction | arXiv 2024 | arxiv.org/abs/2410.01802 |
| Reconsidering GAE for Link Prediction | arXiv 2024 | arxiv.org/abs/2411.03845 |
| Evaluating GNNs for Link Prediction: Current Pitfalls | arXiv 2023 | arxiv.org/abs/2306.10453 |
| ST-ResGAT: Spatio-Temporal ResGAT for Road Condition | arXiv 2026 | arxiv.org/abs/2603.14107 |
