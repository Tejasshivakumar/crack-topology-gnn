# Section — Abstract

> **Target:** 150–250 words (IEEE double-column format).  
> **Reference numbers** use `00_references.md` numbering.

---

## Abstract

Automated crack monitoring in civil infrastructure demands more than pixel-level crack detection: it requires structural reasoning about how crack segments connect, where tips are located, and which disconnected segments are likely to join under continued loading. We present a three-stage pipeline that bridges deep segmentation and graph topology reasoning for this purpose. Stage 1 introduces HybridGraphUNet, a CNN–GNN hybrid segmentation model that couples a pretrained ResNet34d encoder with a ViG-style GNN bottleneck under a combined Focal and CrackDice loss, achieving crack IoU of 0.722 on the DeepCrack benchmark. Stage 2 converts binary crack masks into attributed topology graphs via morphological skeletonisation, graph extraction with sknw, spur pruning, and the computation of 6-dimensional node features and 8-dimensional geometric edge features encoding crack thickness, tortuosity, and orientation. Stage 3 trains and evaluates five GNN encoder architectures — MLP, GCN, GraphSAGE, GINE, and GAT — on a link prediction task that requires identifying which crack tip nodes have hidden neighbours. On 636 test graphs derived from 4,769 labelled crack images, GINE achieves a mean node Average Precision of 0.739 ± 0.004 versus 0.662 ± 0.003 for a structure-blind MLP baseline — a gap of +0.077 with zero distributional overlap across three independent seeds. Under frontier masking, which targets active growth fronts, the GINE–MLP gap amplifies to +0.210, demonstrating that topology-aware message passing becomes increasingly valuable exactly where structural reasoning is hardest. End-to-end evaluation confirms that the pipeline is viable, with the predicted-mask path achieving node AP of 0.800 versus 0.655 for the ground-truth oracle path.

---

> **Word count:** ~235 words  
> **Key numbers present:** 0.722, 0.739 ± 0.004, 0.662 ± 0.003, +0.077, +0.210, 0.800 vs 0.655
