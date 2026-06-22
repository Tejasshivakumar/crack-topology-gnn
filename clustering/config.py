"""
crack_clustering/config.py
===========================
Edit the paths and parameters here before running.
"""

import os

class Config:

    # Default dataset paths (point to a folder containing image masks).
    # Edit these to the folders that contain your train/test mask images
    # or override via command-line args when running `cluster.py`.
    TRAIN_DIR  = "clustering/dataset"   # <-- CHANGE THIS if you have separate train/test folders
    TEST_DIR   = "clustering/dataset"   # <-- CHANGE THIS if you have separate train/test folders
    OUTPUT_DIR = "outputs"       # where CSV + plots are saved

    # ── CLUSTERING ─────────────────────────────────────────────────
    # Set N_CLUSTERS to None to auto-detect the best K (recommended).
    # Set to an integer (e.g. 3) to force a fixed number of clusters.
    N_CLUSTERS   = None    # None = auto-detect via silhouette sweep
    RANDOM_STATE = 42

    # ── GRAPH QUALITY FILTERS ──────────────────────────────────────
    # Prune spur branches shorter than PRUNE_RATIO × longest branch.
    # Skip graphs with fewer than MIN_NODES nodes.
    PRUNE_RATIO = 0.10     # 10% of longest branch = spur threshold
    MIN_NODES   = 3        # drop degenerate (near-empty) graphs
