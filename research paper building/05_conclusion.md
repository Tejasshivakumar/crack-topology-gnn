# Section V — Conclusion

> **Reference numbers** use `00_references.md` numbering.  
> **Target:** ~400–500 words body text (~0.5–0.75 IEEE double-column pages).

---

## V. Conclusion

This paper presented a three-stage pipeline for crack topology reasoning in civil infrastructure inspection images, bridging pixel-level segmentation and graph-based structural analysis.

**Stage 1** introduced HybridGraphUNet, combining a pretrained ResNet34d encoder with a ViG-style GNN bottleneck under a combined Focal + CrackDice loss. The model achieves crack IoU of **0.722** on the DeepCrack benchmark, competitive with DeepCrack's own reported F1 of 0.741 [2]. A three-model ablation on the larger, more heterogeneous crack_seg_clean dataset revealed that the pretrained encoder is the dominant performance driver: both pretrained variants outperform the scratch-trained encoder, while the GNN bottleneck improves topological coherence (highest clDice 0.754 for EnhancedGraphUNet) without consistently improving pixel-level IoU. This finding motivates topology-aware metrics as the primary segmentation target in future pipeline-oriented training.

**Stage 2** constructed 4,364 valid attributed topology graphs from crack masks via morphological skeletonisation, sknw graph extraction, spur pruning, 6-dimensional node features, and 8-dimensional geometric edge features encoding crack thickness, tortuosity, and orientation. The structural feature recomputation protocol — recomputing degree, endpoint, and junction flags from the post-masking observed edge set — ensures that no feature shortcut survives to the model; all heuristic baselines (including the semantically most relevant `is_endpoint` flag) score at or below the random floor of 0.091 Average Precision.

**Stage 3** established the first empirical benchmark for GNN link prediction on image-derived crack topology graphs, comparing five encoder architectures under a rigorous evaluation protocol with hard negative sampling (k = 30, KD-tree), multi-seed validation, and frontier masking. GINE achieves mean node AP of **0.739 ± 0.004** versus **0.662 ± 0.003** for the structure-blind MLP — a gap of **+0.077** with zero distributional overlap across three independent seeds. The gain decomposes cleanly: message passing alone contributes +0.038 (MLP → SAGE); geometric edge features contribute a further +0.039 (SAGE → GINE). Under frontier masking, which targets active propagation fronts, the GINE–MLP gap amplifies to **+0.210**, confirming that topology-aware message passing is most valuable exactly where structural reasoning is hardest. Edge feature ablation identifies crack thickness as the dominant feature group, accounting for −0.347 of the −0.350 total edge feature gain. End-to-end evaluation on 100 held-out images confirms that the complete pipeline is viable, with HybridGraphUNet predicted masks yielding higher link prediction AP (0.800) than ground-truth oracle masks (0.655).

### Limitations

The current pipeline operates on single inspection time points: no temporal sequence of inspections is used, so the model infers connectivity from static structural context rather than observed propagation. Stage 3 is trained on ground-truth masks to isolate its contribution; full end-to-end joint training is not explored. The crack_seg_clean dataset, aggregated from 6 road-pavement sources (curated from 11 candidates), exhibits variable annotation styles and imaging conditions across sources that constrain segmentation IoU (0.630) relative to the clean, single-source DeepCrack benchmark (0.722).

### Future Work

Three natural extensions follow from this work. First, **Positive-Unlabelled link prediction** [9]: absent edges in a static crack graph are not confirmed negatives — some will form under continued loading. A PU-AUC objective [9] provides a theoretically grounded framework for this assumption and represents the direct next step for Stage 3 beyond binary classification. Second, **temporal propagation forecasting**: time-series inspection data with repeated mask annotations would allow the model to predict which currently-absent connections will form next, transforming topology link prediction from a static connectivity task to a dynamic structural prognosis tool. Third, **end-to-end topology-aware training**: jointly optimising Stage 1 segmentation under a clDice [11]-weighted loss specifically targeting downstream graph quality — rather than pixel overlap — could close the gap between oracle and predicted paths and unify the pipeline into a single differentiable model.

---

> **Word count (body text):** ~530 words  
> **Target in final paper:** ~0.5–0.75 IEEE double-column pages  
> **Papers cited in this section:** [2], [9], [11]
