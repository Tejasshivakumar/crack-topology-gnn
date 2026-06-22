# Dataset Cleaning — Full Documentation
**Project:** Road Crack Topology Prediction (Phase 2 Practicum)  
**Date:** 2026-06-19  
**Dataset:** `crack_segmentation_dataset` (khanhha, GitHub)  
**Script:** `data_cleaning/clean.py`  

---

## 1. Why We Needed to Clean

The pipeline needs pixel-level crack masks to:

1. **Stage 1 (Segmentation)** — train a model to detect cracks in road images
2. **Stage 2 (Image-to-Graph)** — convert masks to topology graphs (nodes = junctions/endpoints, edges = crack segments)
3. **Stage 3 (GNN Link Prediction)** — predict how cracks will propagate

The dataset we started with (`crack_segmentation_dataset`, ~11,200 images) is a **community-merged collection** assembled from 12 different source datasets spanning multiple materials and research domains. It was originally built for generic crack segmentation benchmarking — not for road-specific topology prediction. Using it as-is would introduce three problems:

**Problem 1 — Domain mismatch (material)**  
Cracks in road pavement behave differently from cracks in concrete walls, building facades, or ceramic tiles. Road cracks are driven by traffic load, thermal expansion, and water ingress on *horizontal* surfaces and form characteristic patterns (longitudinal, transverse, alligator). Wall/concrete cracks are driven by shrinkage, tension, and corrosion on *vertical* surfaces with completely different branching topology. Training a topology-prediction model on mixed-domain data teaches the GNN patterns that don't apply to roads.

**Problem 2 — No-crack images producing empty graphs**  
Some images in the dataset contain no cracks at all (the `noncrack` source). Their masks are all black. In Stage 2, images with no crack pixels produce no skeleton, no graph, and are silently skipped. Including them wastes processing time and can distort dataset statistics.

**Problem 3 — Exact duplicates causing data leakage**  
The same image appearing in both the train and test splits means the model is evaluated on data it was trained on — an invisible form of data leakage that inflates test metrics. We found 56 such cross-split duplicates that needed to be removed.

---

## 2. The Source Dataset — What It Is

**Name:** `crack_segmentation_dataset`  
**Maintainer:** Khanh Ha (khanhha89@gmail.com), GitHub: `khanhha/crack_segmentation`  
**Total images:** ~11,200 (all resized to 448×448 pixels)  
**Format:** JPG images + JPG binary masks (white = crack, black = background)  
**Pre-split:** train and test folders provided, stratified by source dataset  

The dataset merges images from **12 original sources**. Each image's filename begins with a **prefix that encodes its origin dataset** — this is the key to the entire cleaning process.

From the dataset readme:
> *"The name prefix of each image is assigned to the corresponding dataset that the image belongs to."*

---

## 3. Step 1 — Source Inventory

### 3.1 Method

We wrote a Python script (`clean.py`) that:
1. Scans all filenames in `train/images/` and `test/images/`
2. Extracts the leading alphabetic characters as the **prefix** (e.g. `CRACK500_20160222_...jpg` → prefix `CRACK`)
3. Maps each prefix to its source dataset using a hand-built catalogue

We found **11 distinct prefixes** across 11,298 total images.

### 3.2 Full Source Catalogue

Each source was researched via its original publication and verified against the dataset readme.

---

#### `CRACK` → CRACK500
- **Full name:** CRACK500
- **Domain / Material:** Road pavement — asphalt
- **Images in dataset:** 3,363
- **Original size:** 500 images (~2000×1500 px), cropped into 16 non-overlapping patches; only patches with >1,000 crack pixels retained → 3,368 images
- **Capture:** Cell phones on the main campus of Temple University, Philadelphia
- **Annotation:** Pixel-level binary masks
- **Crack types:** Mixed (longitudinal, transverse, alligator)
- **Reference:** Zhang et al., "Road crack detection using deep convolutional neural network," IEEE ICIP 2016; Yang et al., "Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection," arXiv 2019
- **Verdict:** ✅ KEEP — standard road pavement benchmark, largest single component

---

#### `GAPS` → GAPs384
- **Full name:** German Asphalt Pavement distress dataset — 384 crack subset
- **Domain / Material:** Road pavement — asphalt (German highways)
- **Images in dataset:** 509
- **Original size:** GAPs contains 1,969 grayscale images (1920×1080, 1.2 mm/pixel resolution) with multiple distress classes (cracks, potholes, inlaid patches). GAPs384 is the 384-image subset containing only crack class, pixel-wise annotated, each cropped into 6 non-overlapping 640×544 patches
- **Annotation:** Pixel-level binary masks
- **Reference:** Eisenbach et al., "How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach," IJCNN 2017
- **Verdict:** ✅ KEEP — high-resolution, professionally captured asphalt pavement

---

#### `CFD` → CrackForest Dataset (CFD)
- **Full name:** CrackForest Dataset
- **Domain / Material:** Road pavement — cement road surface
- **Images in dataset:** 118
- **Original size:** 118 images, 480×320 pixels
- **Capture:** iPhone 5 on cement road surfaces in Beijing, China
- **Annotation:** Pixel-level binary masks, manually annotated
- **Crack types:** 5 alligator, 95 transverse, 10 longitudinal labels
- **Challenges:** Shadows, water stains, illumination variation; crack widths 1–3 mm
- **Reference:** Shi et al., "Automatic Road Crack Detection Using Random Structured Forests," IEEE Transactions on Intelligent Transportation Systems, vol. 17, no. 12, 2016
- **Verdict:** ✅ KEEP — clean road pavement benchmark with diverse crack types

---

#### `cracktree` → CrackTree200
- **Full name:** CrackTree200
- **Domain / Material:** Road pavement — cement pavement
- **Images in dataset:** 206
- **Original size:** 206 images, 800×600 pixels
- **Annotation:** Pixel-wise labels
- **Challenges:** Shadows, occlusions, low contrast, noise, environmental effects
- **Reference:** Zou et al., "CrackTree: Automatic crack detection from pavement images," Pattern Recognition Letters, vol. 33, no. 3, pp. 227–238, 2012
- **Verdict:** ✅ KEEP — widely used pavement benchmark, strong community validation

---

#### `DeepCrack` → DeepCrack
- **Full name:** DeepCrack
- **Domain / Material:** Road pavement — asphalt AND concrete pavement (both are road surfaces, not walls)
- **Images in dataset:** 521
- **Original size:** 537 images, 544×384 pixels; split 300 train / 237 test
- **Annotation:** Pixel-level, human-annotated
- **Textures:** Three — bare, dirty, rough
- **Crack widths:** 1 pixel to 180 pixels (wide range)
- **Reference:** Liu et al., "DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation," Neurocomputing 2019; GitHub: `yhlleo/DeepCrack`
- **Note:** This is the same DeepCrack dataset used in our Stage 1 segmentation experiments. Including it in the larger cleaned set is intentional — it is road pavement.
- **Verdict:** ✅ KEEP — high-quality, pre-split road pavement benchmark covering both asphalt and concrete road surfaces

---

#### `Sylvie` → AEL Dataset (Chambon)
- **Full name:** AEL (Amhaz-Eisenbach-Labayrade) dataset
- **Domain / Material:** Road pavement
- **Images in dataset:** 185 (before deduplication)
- **Original size:** Combination of three sub-databases (38 + 15 + 5 = 58 images total); 63 labels: 23 transverse, 37 longitudinal, 3 healthy pavement
- **Reference:** Amhaz, Chambon, Idier, Baltazart, "Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection," IEEE Transactions on Intelligent Transportation Systems, 2016
- **Verdict:** ✅ KEEP — road pavement source, but heavily affected by duplication (see Step 3)

---

#### `forest` → Forest / CFD-variant
- **Full name:** Unknown (not publicly documented; suspected variant of CFD)
- **Domain / Material:** Road pavement (visually)
- **Images in dataset:** 118 (before deduplication)
- **Research context:** The CrackSeg9k paper (2022) explicitly flagged this source:
  > *"Forest dataset was excluded as it was found to be very similar to the CFD dataset"*
- **Our finding:** After MD5-based deduplication, **all 118 forest images were exact duplicates** of images already present in the CFD source. Zero unique images survived.
- **Verdict:** ✅ Classified as road-pavement (kept in scope filter), but ❌ fully eliminated by deduplication — confirms CrackSeg9k's assessment

---

#### `Rissbilder` → Rissbilder
- **Full name:** Rissbilder (German: "crack images")
- **Domain / Material:** Concrete wall / building facade — NOT road pavement
- **Images in dataset:** 3,822 (34% of the entire dataset)
- **Capture context:** Images collected by a **wall-climbing inspection robot** examining concrete building facades. Vertical surface inspection, not horizontal road survey.
- **Characteristics:** Concrete shrinkage and tension cracks on vertical building surfaces; very different texture, lighting, and crack morphology from road pavement
- **Reference:** Referenced in crack detection robot papers; merged into CrackSeg9k (2022) Harvard Dataverse
- **Why it matters:** This is the single largest source in the raw dataset. Keeping it would mean 34% of training data has a fundamentally different crack topology domain.
- **Verdict:** ❌ DROP — concrete wall domain, wrong material for road topology prediction

---

#### `Volker` → Volker
- **Full name:** Volker (personal/institutional dataset, not publicly published)
- **Domain / Material:** Concrete structure / building — NOT road pavement
- **Images in dataset:** 990
- **Capture context:** Associated with Bauhaus University (Weimar, Germany), a university known for architecture and construction engineering. Filename pattern `Volker_DSC01608_*` suggests DSC (Digital Still Camera) photos of structural surfaces, not road surveys.
- **Domain assessment:** Consistent with structural/building inspection imagery based on filename conventions, institutional context, and visual characteristics reported by researchers using this dataset. Domain not confirmed by a citable publication — this is the main uncertainty.
- **Decision rationale:** When domain is uncertain and we have a road-only research scope, the correct conservative decision is to exclude. If Volker turns out to be road pavement, it can be re-added with a one-line change to the keep-list.
- **Verdict:** ❌ DROP — unconfirmed domain + domain-shift risk + conservative scope filter

---

#### `Eugen` → Eugen Muller
- **Full name:** Eugen Muller dataset (personal/institutional, not publicly published)
- **Domain / Material:** Unknown / structural — NOT confirmed road pavement
- **Images in dataset:** 55
- **Quality:** Poor. The CrackSeg9k (2022) paper, which assembled a similar merged dataset, explicitly states:
  > *"Smaller datasets like Eugen Muller and Sylvie Chambon were discarded due to poor quality and much fewer images"*
- **Verdict:** ❌ DROP — unverified domain + poor quality (confirmed by independent research) + too small to matter even if kept

---

#### `noncrack` → NonCrack (Concrete Wall)
- **Full name:** NonCrack negative samples
- **Domain / Material:** Concrete wall — NOT road pavement
- **Images in dataset:** 1,411
- **Filename pattern:** `noncrack_noncrack_concrete_wall_0_0.jpg.jpg` (double extension is present in the original files)
- **Content:** Images of **concrete wall surfaces with no cracks** — pure background negatives
- **Two reasons to drop:**
  1. **Wrong domain:** Concrete wall, not road pavement
  2. **No crack signal:** All-black masks. In Stage 2 (`image_to_graph`), images with no crack pixels produce no skeleton → no graph → silently skipped. They contribute nothing to the pipeline.
- **Verdict:** ❌ DROP — wrong domain AND produces empty graphs

---

### 3.3 Inventory Summary

| Prefix | Source | Domain | Images | Decision |
|--------|--------|--------|--------|----------|
| `CRACK` | CRACK500 | road-pavement | 3,363 | ✅ KEEP |
| `Rissbilder` | Rissbilder | concrete-wall | 3,822 | ❌ DROP |
| `noncrack` | NonCrack | concrete-wall | 1,411 | ❌ DROP |
| `Volker` | Volker | concrete-structure | 990 | ❌ DROP |
| `GAPS` | GAPs384 | road-pavement | 509 | ✅ KEEP |
| `DeepCrack` | DeepCrack | road-pavement | 521 | ✅ KEEP |
| `cracktree` | CrackTree200 | road-pavement | 206 | ✅ KEEP |
| `Sylvie` | AEL (Chambon) | road-pavement | 185 | ✅ KEEP |
| `CFD` | CrackForest | road-pavement | 118 | ✅ KEEP |
| `forest` | Forest/CFD-variant | road-pavement | 118 | ✅ KEEP* |
| `Eugen` | Eugen Muller | unknown | 55 | ❌ DROP |

*`forest` passed the scope filter but was entirely eliminated in deduplication.

---

## 4. Step 2 — Scope Filter

### 4.1 Decision: Road-Pavement Only

**Research scope:** Our pipeline predicts road crack topology evolution — how cracks on road surfaces spread and connect over time. The GNN in Stage 3 learns structural patterns of crack graphs (branching, merging, extension of tips). These patterns are domain-specific.

**The rule:** Keep only sources where `domain = road-pavement`.

```python
KEEP_DOMAINS = {'road-pavement'}   # one-line change to go cross-material
```

### 4.2 Why Not Keep Cross-Material Data?

The literature is clear on this. Domain shift between crack materials is a known and documented problem:

**Evidence 1 — CDE-Crack (2023)**  
Models trained on pavement crack data show significant performance degradation when applied to concrete wall or ceramic tile crack data without domain adaptation. The materials have different:
- Surface texture (smooth asphalt vs. rough concrete aggregate)
- Crack width distribution (road: typically 1–5 mm hairline; wall: can be much wider shrinkage cracks)
- Crack topology (road: network patterns from load distribution; wall: dendritic/branching patterns from tension)

**Evidence 2 — Unsupervised Domain Adaptation for Crack Segmentation (Automation in Construction, 2023)**  
*"Due to differences in construction materials, imaging conditions, and environmental interference, there exists obvious distribution difference between crack images collected from different civil infrastructures, namely domain shift."*

**Evidence 3 — Deep Domain Adaptation for Pavement Crack Detection (2021)**  
Cross-material crack detection requires explicit domain adaptation techniques; naive mixing degrades both segmentation IoU and structural accuracy.

**What this means for our GNN:**  
Stage 3 learns the topology of crack graphs. If trained on wall-crack topology (which branches differently due to material stress patterns), it will learn incorrect priors for predicting how road cracks propagate. The graph structure — node degrees, edge tortuosity, branching angles — differs between materials.

### 4.3 Filter Result

| | Count |
|--|--|
| Passed scope filter (road-pavement) | 5,020 |
| Dropped (wrong domain / no signal) | 6,278 |

**55% of the original dataset was discarded at this step.** The three largest non-pavement sources (Rissbilder 3,822 + noncrack 1,411 + Volker 990 = 6,223) account for 99% of this.

---

## 5. Step 3 — Exact Duplicate Removal

### 5.1 Why Deduplication Matters

The merged dataset was assembled from 12 independent sources. When different research groups download and re-package public datasets, the same image can appear under different filenames or in different sub-collections. This causes:

- **Wasted training** — the model sees the same image multiple times per epoch with no new information
- **Data leakage** — if the same image appears in both train and test splits, the model is evaluated on images it trained on, which artificially inflates test metrics

### 5.2 Method

**Algorithm:** MD5 cryptographic hash of raw image file bytes.

Two files are exact duplicates if and only if their MD5 hashes match. This is a byte-level comparison — it catches identical files regardless of filename. It does NOT catch near-duplicates (slight crops, rotations, or recolourisations of the same image). For our purposes, exact deduplication is sufficient and conservative.

**Process:**
1. For each image that passed the scope filter (5,020 images), compute its MD5 hash
2. Build a hash → first-seen mapping
3. If a hash is seen again, mark the second occurrence as a duplicate
4. For cross-split duplicates (same hash in train and test), remove from the test split (preserve training data, protect test set integrity)

### 5.3 Results

| | Count |
|--|--|
| Duplicate pairs found | 251 |
| Intra-train duplicates | 195 |
| Intra-test duplicates | 0 |
| **Cross-split duplicates (data leakage)** | **56** |
| Images removed | 251 |

### 5.4 Which Sources Had Duplicates?

| Source | Duplicates Removed | Original Count | Unique Remaining |
|--------|-------------------|----------------|-----------------|
| Forest (CFD-variant) | **118** (100%) | 118 | **0** |
| AEL / Sylvie | 115 | 185 | 70 |
| CrackForest (CFD) | 18 | 118 | 100 |
| All others | 0 | — | unchanged |

**The `forest` source was entirely eliminated** — all 118 of its images were exact duplicates of images from the CrackForest (CFD) source. This independently confirms what CrackSeg9k (2022) found: *"Forest dataset was excluded as it was found to be very similar to the CFD dataset."* In our case, they are not merely similar — they are byte-for-byte identical.

**AEL / Sylvie** had 115 of 185 images duplicated — suggesting the sub-database merger that formed AEL included images already present in CFD or CRACK500. 70 unique AEL images remain.

**56 cross-split duplicates were data leakage** — the same image existed in both train and test splits. These were removed from the test split.

---

## 6. Final Clean Dataset

### 6.1 Numbers

| | Before cleaning | After cleaning |
|--|--|--|
| Total images | 11,298 | **4,769** |
| Train | 9,603 | **4,071** |
| Test | 1,695 | **698** |
| Sources | 11 | **5** (+ forest removed by dedup) |

### 6.2 Source Breakdown (Final)

| Source | Train | Test | Total | % of clean set |
|--------|-------|------|-------|----------------|
| CRACK500 | 2,858 | 505 | 3,363 | 70.5% |
| DeepCrack | 443 | 78 | 521 | 10.9% |
| GAPs384 | 433 | 76 | 509 | 10.7% |
| CrackTree200 | 175 | 31 | 206 | 4.3% |
| CrackForest (CFD) | 100 | 0 | 100 | 2.1% |
| AEL (Chambon) | 62 | 8 | 70 | 1.5% |
| **Total** | **4,071** | **698** | **4,769** | 100% |

### 6.3 What Was Dropped and Why (Summary Table)

| Source | Count | Primary Reason | Secondary Reason |
|--------|-------|---------------|-----------------|
| Rissbilder | 3,822 | Wrong domain (concrete wall) | Wall-climbing robot inspection imagery |
| noncrack | 1,411 | No crack signal (empty masks) | Wrong domain (concrete wall) |
| Volker | 990 | Uncertain domain (concrete structure) | Domain-shift risk |
| Eugen Muller | 55 | Poor quality (CrackSeg9k confirmed) | Unverified domain |
| Forest | 118 | 100% exact duplicates of CFD | — |
| AEL duplicates | 115 | Exact duplicates within set | — |
| CFD duplicates | 18 | Exact duplicates within set | — |
| Cross-split dups | 56 | Data leakage (train=test overlap) | — |
| **Total dropped** | **6,585** | | |

*Note: 6,278 dropped by scope filter + 251 by deduplication = 6,529 unique drop events (56 cross-split duplicates also passed the scope filter, so they are counted in both)*

---

## 7. Implementation

### 7.1 Script

`data_cleaning/clean.py` — single self-contained script, ~300 lines.

**Key design decisions:**
- **No hardcoded paths** — `--dataset-root` and `--output-dir` are CLI arguments
- **Manifest-first** — the primary output is `kept.csv` (a list of image/mask paths), not copied files. File copying is opt-in via `--copy` to avoid mandatory disk duplication.
- **Extensible scope filter** — `KEEP_DOMAINS = {'road-pavement'}` at the top of the file. Changing to `{'road-pavement', 'concrete-pavement'}` enables cross-material experiments in one line.
- **Conservative on unknowns** — any prefix not in the catalogue is dropped by default

### 7.2 Running the Script

```bash
cd "/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn"

# Generate manifests and report only (no file copying — fast)
python3 data_cleaning/clean.py

# Generate manifests AND copy clean images to a new directory
python3 data_cleaning/clean.py --copy

# Copy to a custom location outside the repo
python3 data_cleaning/clean.py \
    --output-dir "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean" \
    --copy
```

### 7.3 Outputs

| File | Description |
|------|-------------|
| `outputs/inventory.csv` | All 11,298 images: prefix, source, domain, keep flag, notes |
| `outputs/kept.csv` | 4,769 clean images with full paths — feed directly to Stage 2 |
| `outputs/dropped.csv` | 6,529 dropped images with per-image reason |
| `outputs/duplicates.csv` | 251 duplicate pairs, flagged by cross-split status |
| `outputs/report.md` | Auto-generated summary tables |
| `outputs/clean/` | Copied images and masks (only if `--copy` used) |

### 7.4 Using the Clean Data in Stage 2

```python
import csv

with open("data_cleaning/outputs/kept.csv") as f:
    rows = list(csv.DictReader(f))

train_masks = [r["mask_path"] for r in rows if r["split"] == "train"]
test_masks  = [r["mask_path"] for r in rows if r["split"] == "test"]
```

Or use the copied directory directly with `build_dataset.py`:

```bash
TRAIN_MASKS="data_cleaning/outputs/clean/train/masks"
TRAIN_IMGS="data_cleaning/outputs/clean/train/images"
TEST_MASKS="data_cleaning/outputs/clean/test/masks"
TEST_IMGS="data_cleaning/outputs/clean/test/images"

python3 image_to_graph/build_dataset.py \
    --mask-dir  "$TRAIN_MASKS" \
    --image-dir "$TRAIN_IMGS" \
    --split train \
    --output-dir outputs/clean_graphs
```

---

## 8. References

| # | Citation | Relevance |
|---|----------|-----------|
| 1 | Zhang L. et al., "Road crack detection using deep convolutional neural network," *IEEE ICIP*, 2016 | CRACK500 source dataset |
| 2 | Yang F. et al., "Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection," *arXiv:1901.06340*, 2019 | CRACK500 extended paper |
| 3 | Eisenbach M. et al., "How to Get Pavement Distress Detection Ready for Deep Learning?" *IJCNN*, 2017 | GAPs384 source dataset |
| 4 | Shi Y. et al., "Automatic Road Crack Detection Using Random Structured Forests," *IEEE ITS*, vol. 17, no. 12, 2016 | CrackForest (CFD) source |
| 5 | Zou Q. et al., "CrackTree: Automatic crack detection from pavement images," *Pattern Recognition Letters*, vol. 33, no. 3, 2012 | CrackTree200 source |
| 6 | Liu W. et al., "DeepCrack: A Deep Hierarchical Feature Learning Architecture," *Neurocomputing*, 2019 | DeepCrack source |
| 7 | Amhaz R. et al., "Automatic Crack Detection on Two-Dimensional Pavement Images," *IEEE ITS*, 2016 | AEL / Sylvie Chambon source |
| 8 | Kulkarni A. et al., "CrackSeg9k: A Collection and Benchmark for Crack Segmentation Datasets and Frameworks," *ECCV Workshops*, 2022 | Identified forest=CFD duplicates; confirmed Eugen Muller poor quality; material domain breakdown |
| 9 | "Unsupervised Domain Adaptation for Crack Segmentation," *Automation in Construction*, 2023 | Justification for dropping cross-material sources |
| 10 | "Deep Domain Adaptation for Pavement Crack Detection," *arXiv:2111.10101*, 2021 | Quantified domain-shift performance degradation |
| 11 | khanhhha, `crack_segmentation_dataset`, GitHub, 2020 | The merged dataset we cleaned |

---

## 9. Decisions Written Down

For the record — decisions made during this cleaning that may need to be revisited:

| Decision | Choice Made | How to Change |
|----------|-------------|---------------|
| Material scope | Road-pavement only | Edit `KEEP_DOMAINS` in `clean.py` |
| Volker source | Dropped (uncertain domain) | Add `'Volker': {..., 'keep': True}` if domain confirmed as road |
| Duplicate resolution | Keep first occurrence, remove subsequent | Change priority logic in `run()` |
| Cross-split dedup | Remove from test split | Swap to remove from train if preferred |
| Near-duplicates | Not checked (MD5 exact only) | Add perceptual hash (pHash) step if needed |
| Empty-mask images | Dropped (noncrack source) | Keep and use as hard negatives if training segmentation model |
