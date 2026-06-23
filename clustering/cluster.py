"""
crack_clustering/src/cluster.py
================================
Crack Graph Clustering Pipeline — VS Code version

Usage:
    python src/cluster.py --train data/train --test data/test --n_clusters 3

Or just set the paths in config.py and run:
    python src/cluster.py
"""

import argparse
import os
import sys
import csv
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from skimage import io, morphology, color
from skimage.util import img_as_bool
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import sknw

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
import pickle

# Optional import of torch for .pt files
try:
    import torch
except Exception:
    torch = None


# ──────────────────────────────────────────────
# IMAGE → BINARY MASK
# ──────────────────────────────────────────────
def load_mask(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img_as_bool(img).astype(np.uint8)


# ──────────────────────────────────────────────
# MASK → SKELETON → GRAPH
# ──────────────────────────────────────────────
def prune_spurs(graph: nx.Graph, prune_ratio: float) -> nx.Graph:
    """Remove short leaf branches (spurious noise from rough mask edges)."""
    if graph.number_of_edges() == 0:
        return graph

    edge_lengths = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    threshold = prune_ratio * max(edge_lengths)

    g = graph.copy()
    changed = True
    while changed:
        changed = False
        leaves = [n for n in g.nodes() if g.degree(n) == 1]
        for leaf in leaves:
            nbrs = list(g.neighbors(leaf))
            if not nbrs:
                continue
            if g.edges[leaf, nbrs[0]].get("weight", 1.0) < threshold:
                g.remove_node(leaf)
                changed = True
    return g


def mask_to_graph(binary: np.ndarray, prune_ratio: float) -> nx.Graph:
    """Binary mask → pruned NetworkX graph via skeletonization."""
    skeleton = morphology.skeletonize(binary > 0)
    graph = sknw.build_sknw(skeleton.astype(np.uint16))

    # Compute edge weights = path length in pixels
    for u, v, data in graph.edges(data=True):
        pts = data.get("pts", np.array([]))
        if len(pts) > 1:
            diffs = np.diff(pts, axis=0)
            length = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
        else:
            pos_u = graph.nodes[u].get("o", np.zeros(2))
            pos_v = graph.nodes[v].get("o", np.zeros(2))
            length = float(np.linalg.norm(pos_u - pos_v))
        data["weight"] = max(length, 1.0)

    return prune_spurs(graph, prune_ratio)


# ──────────────────────────────────────────────
# GRAPH → FEATURE VECTOR
# ──────────────────────────────────────────────
def extract_features(graph: nx.Graph) -> dict:
    """
    11 scale-invariant structural features for crack type clustering.
    All raw counts are converted to per-node ratios so that a small
    alligator crack and a large alligator crack land in the same cluster.
    Raw counts (n_nodes, n_edges, total_length, …) are excluded.
    """
    n_nodes    = graph.number_of_nodes()
    n_edges    = graph.number_of_edges()
    components = nx.number_connected_components(graph)
    cyclomatic = max(n_edges - n_nodes + components, 0)

    degrees     = [d for _, d in graph.degree()]
    avg_degree  = float(np.mean(degrees)) if degrees else 0.0
    max_degree  = float(max(degrees))     if degrees else 0.0
    n_endpoints = sum(1 for d in degrees if d == 1)
    n_junctions = sum(1 for d in degrees if d >= 3)

    eps = 1e-6  # guard against division by zero in all ratios

    # --- shape ratios (the clustering signal) ---
    branching_ratio    = n_junctions / max(n_endpoints, 1)  # cap at n_junctions when no endpoints
    junctions_per_node = n_junctions / max(n_nodes,    eps)
    endpoints_per_node = n_endpoints / max(n_nodes,    eps)
    cyclomatic_per_node = cyclomatic / max(n_nodes,    eps)
    edges_per_node     = n_edges     / max(n_nodes,    eps)
    components_per_node = components / max(n_nodes,    eps)

    edge_lengths = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    avg_length   = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    std_length   = float(np.std(edge_lengths))  if edge_lengths else 0.0
    # coefficient of variation — segment-length irregularity, scale-free
    cv_length = std_length / max(avg_length, eps)

    tortuosities = []
    for u, v, data in graph.edges(data=True):
        pos_u = graph.nodes[u].get("o", np.zeros(2))
        pos_v = graph.nodes[v].get("o", np.zeros(2))
        euc   = float(np.linalg.norm(pos_u - pos_v))
        tortuosities.append(data.get("weight", 1.0) / max(euc, eps))
    avg_tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0

    # node_density = nodes per bounding-box pixel² — measures spatial compactness
    node_density = 0.0
    if n_nodes > 1 and "o" in graph.nodes[list(graph.nodes)[0]]:
        positions = np.array([graph.nodes[n]["o"] for n in graph.nodes()])
        bbox_area = (
            (positions[:, 0].max() - positions[:, 0].min() + 1) *
            (positions[:, 1].max() - positions[:, 1].min() + 1)
        )
        node_density = n_nodes / max(bbox_area, 1.0)

    return {
        "avg_degree":          avg_degree,
        "max_degree":          max_degree,
        "branching_ratio":     branching_ratio,
        "junctions_per_node":  junctions_per_node,
        "endpoints_per_node":  endpoints_per_node,
        "cyclomatic_per_node": cyclomatic_per_node,
        "edges_per_node":      edges_per_node,
        "components_per_node": components_per_node,
        "avg_tortuosity":      avg_tortuosity,
        "cv_length":           cv_length,
        "node_density":        node_density,
        # raw counts kept for diagnostic / CSV reference but NOT fed to clustering
        "_n_nodes":  n_nodes,
        "_n_edges":  n_edges,
    }


# Scale-invariant feature keys fed to the clustering model
SHAPE_FEAT_KEYS = (
    "avg_degree", "max_degree", "branching_ratio",
    "junctions_per_node", "endpoints_per_node", "cyclomatic_per_node",
    "edges_per_node", "components_per_node",
    "avg_tortuosity", "cv_length", "node_density",
)
# Diagnostic raw counts written to CSV but excluded from feature matrix
DIAG_KEYS = ("_n_nodes", "_n_edges")


# ──────────────────────────────────────────────
# PROCESS A FOLDER OF IMAGES
# ──────────────────────────────────────────────
def process_folder(folder: str, split: str, cfg: Config) -> list:
    exts    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths   = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    records = []
    skipped = 0

    print(f"\n[{split.upper()}] {len(paths)} images found in: {folder}")

    for path in tqdm(paths, desc=f"  {split}"):
        try:
            binary = load_mask(str(path))
            if binary.sum() == 0:
                skipped += 1
                continue

            graph = mask_to_graph(binary, cfg.PRUNE_RATIO)

            if graph.number_of_nodes() < cfg.MIN_NODES or graph.number_of_edges() == 0:
                skipped += 1
                continue

            records.append({"filename": path.name, "split": split,
                             **extract_features(graph)})
        except Exception as e:
            print(f"  ✗ Skipped {path.name}: {e}")
            skipped += 1

    print(f"  ✓ Valid: {len(records)}   Skipped/degenerate: {skipped}")
    return records


def load_graphs_from_pt(pt_path: str):
    """Load graphs from a .pt file using torch (preferred) or pickle as fallback.

    Returns a list-like container of graphs (NetworkX Graphs or similar objects).
    """
    # Require PyTorch to load .pt files saved via torch.save().
    try:
        import torch as _torch
    except Exception:
        raise RuntimeError(
            "PyTorch is required to load .pt files. Please install it (e.g. `pip install torch`)."
        )

    # Use safe options: map to CPU and allow arbitrary objects (weights_only=False)
    try:
        data = _torch.load(pt_path, map_location="cpu", weights_only=False)
        print(f"Loaded .pt with torch: {type(data)}")
    except TypeError:
        # Older PyTorch versions don't support weights_only argument
        try:
            data = _torch.load(pt_path, map_location="cpu")
            print(f"Loaded .pt with torch (no weights_only): {type(data)}")
        except Exception as e:
            raise RuntimeError(f"torch.load failed: {e}")
    except Exception as e:
        raise RuntimeError(f"torch.load failed: {e}")

    # Heuristics to extract graphs from common containers
    if isinstance(data, dict):
        # common keys: 'graphs', 'data', 'dataset'
        for key in ("graphs", "data", "dataset", "graphs_list"):
            if key in data:
                return data[key]
        # maybe the dict itself is a single graph-like mapping
        return [data]

    if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
        return list(data)

    # fallback: wrap single object
    return [data]


def graphs_to_records(graphs, split: str, cfg: Config) -> list:
    records = []
    skipped_counts = {
        "missing_attr": 0,
        "too_few_nodes": 0,
        "zero_edges": 0,
        "exception": 0,
    }
    skipped_examples = []

    for i, g in enumerate(graphs):
        try:
            graph = g
            # If item is a dict with 'graph' key, unwrap
            if isinstance(g, dict) and "graph" in g:
                graph = g["graph"]

            # torch_geometric Data -> convert to NetworkX
            try:
                from torch_geometric.data import Data as _TGData
                from torch_geometric.utils import to_networkx as _to_nx
            except Exception:
                _TGData = None
                _to_nx = None

            if _TGData is not None and isinstance(graph, _TGData):
                # Convert PyG Data to networkx graph
                try:
                    nx_graph = _to_nx(graph, to_undirected=True)
                    # attach positions if present
                    if hasattr(graph, 'pos') and graph.pos is not None:
                        pos = graph.pos.cpu().numpy()
                        for idx, p in enumerate(pos):
                            if idx in nx_graph.nodes:
                                nx_graph.nodes[idx]['o'] = p
                    # compute edge weights from positions if available
                    if hasattr(graph, 'pos') and graph.pos is not None:
                        for u, v in nx_graph.edges():
                            pu = nx_graph.nodes[u].get('o', None)
                            pv = nx_graph.nodes[v].get('o', None)
                            if pu is not None and pv is not None:
                                nx_graph.edges[u, v]['weight'] = float(np.linalg.norm(pu - pv))
                            else:
                                nx_graph.edges[u, v]['weight'] = 1.0
                    graph = nx_graph
                except Exception as e:
                    print(f"  ✗ Failed to convert PyG Data -> NX for item #{i}: {e}")
                    skipped_counts['exception'] += 1
                    continue

            if not hasattr(graph, "number_of_nodes"):
                skipped_counts["missing_attr"] += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(graph)
                continue

            if graph.number_of_nodes() < cfg.MIN_NODES:
                skipped_counts["too_few_nodes"] += 1
                continue

            if graph.number_of_edges() == 0:
                skipped_counts["zero_edges"] += 1
                continue

            fname = getattr(g, "filename", None) or getattr(graph, "name", None) or f"pt_graph_{i}"
            records.append({"filename": fname, "split": split,
                             **extract_features(graph)})
        except Exception as e:
            print(f"  ✗ Skipped object #{i}: {e}")
            skipped_counts["exception"] += 1

    total_skipped = sum(skipped_counts.values())
    print(f"  ✓ Valid: {len(records)}   Skipped/degenerate: {total_skipped}")
    for k, v in skipped_counts.items():
        if v:
            print(f"    - {k}: {v}")

    # If nothing valid found, print diagnostic examples to help debug .pt contents
    if len(records) == 0 and skipped_examples:
        print("\n[Diagnostic] Sample skipped items (showing up to 5):")
        for i, item in enumerate(skipped_examples):
            print(f"  Item #{i}  type={type(item)}")
            try:
                if isinstance(item, dict):
                    print(f"    dict keys: {list(item.keys())}")
                else:
                    # show some useful attributes if present
                    attrs = []
                    for a in ("number_of_nodes", "nodes", "edges", "name", "graph"):
                        if hasattr(item, a):
                            attrs.append(a)
                    if attrs:
                        print(f"    has attrs: {attrs}")
                    # for networkx graphs, print small summary
                    try:
                        import networkx as _nx
                        if isinstance(item, _nx.Graph):
                            print(f"    networkx.Graph  nodes={item.number_of_nodes()}  edges={item.number_of_edges()}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"    (diagnostic failed: {e})")

    return records


# ──────────────────────────────────────────────
# FIND OPTIMAL K
# ──────────────────────────────────────────────
def find_optimal_k(X: np.ndarray, max_k: int, random_state: int) -> int:
    print("\n[K Search] Silhouette scores:")
    scores = {}
    for k in range(2, min(max_k + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        scores[k] = silhouette_score(X, km.fit_predict(X))
        print(f"  k={k}  →  {scores[k]:.4f}")
    best = max(scores, key=scores.get)
    print(f"  → Best k = {best}  (silhouette = {scores[best]:.4f})")
    return best


# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────
def spectral_cluster(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    print(f"\n[Spectral Clustering]  k={k}")
    # nearest_neighbors affinity: build k-NN graph instead of full RBF matrix.
    # RBF with gamma=1.0 collapses to near-zero for graphs far from the median
    # in scaled feature space, making spectral decomposition degenerate.
    n_neighbors = min(15, len(X) - 1)
    sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                            n_neighbors=n_neighbors,
                            assign_labels="kmeans", random_state=random_state,
                            n_init=20, n_jobs=-1)
    labels = sc.fit_predict(X)
    print(f"  Silhouette: {silhouette_score(X, labels):.4f}")
    return labels


def kmeans_cluster(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    print(f"\n[K-Means Validation]   k={k}")
    km     = KMeans(n_clusters=k, random_state=random_state, n_init=30)
    labels = km.fit_predict(X)
    print(f"  Silhouette: {silhouette_score(X, labels):.4f}")
    return labels


# ──────────────────────────────────────────────
# SAVE LABELS CSV
# ──────────────────────────────────────────────
def save_labels(records, sp_labels, km_labels, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "crack_pseudo_labels.csv")
    # shape features first, then raw diagnostic counts (_n_nodes, _n_edges)
    all_feat_keys = list(SHAPE_FEAT_KEYS) + list(DIAG_KEYS)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "split", "spectral_label", "kmeans_label", *all_feat_keys
        ])
        writer.writeheader()
        for i, rec in enumerate(records):
            writer.writerow({
                "filename":       rec["filename"],
                "split":          rec["split"],
                "spectral_label": int(sp_labels[i]),
                "kmeans_label":   int(km_labels[i]),
                **{k: rec[k] for k in all_feat_keys}
            })

    print(f"\n  Labels saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────
# VISUALISE
# ──────────────────────────────────────────────
def visualise(X, sp_labels, km_labels, out_dir: str):
    pca  = PCA(n_components=2, random_state=42)
    X2d  = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, labels, title in zip(
        axes,
        [sp_labels, km_labels],
        ["Spectral Clustering (Primary)", "K-Means (Validation)"]
    ):
        sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=labels,
                        cmap="tab10", s=15, alpha=0.75)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
        plt.colorbar(sc, ax=ax, label="Cluster")

    plt.suptitle("Crack Graph Clusters — PCA 2D Projection", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cluster_plot.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved   → {out_path}")


# ──────────────────────────────────────────────
# CLUSTER SUMMARY
# ──────────────────────────────────────────────
def print_summary(records, labels, method: str):
    print(f"\n{'='*55}\n  {method} — Cluster Summary\n{'='*55}")
    # shape features + raw counts for inspection
    display_keys = list(SHAPE_FEAT_KEYS) + ["_n_nodes", "_n_edges"]
    for lbl in sorted(set(labels)):
        idx = np.where(labels == lbl)[0]
        print(f"\n  Cluster {lbl}  ({len(idx)} graphs)")
        for fk in display_keys:
            vals = [records[i][fk] for i in idx]
            print(f"    {fk:<22}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")


# ──────────────────────────────────────────────
# WRITE CLUSTER LABELS BACK TO .pt FILES
# ──────────────────────────────────────────────
def save_clustered_pt(train_pt: str, test_pt: str, records: list,
                      sp_labels: np.ndarray, out_dir: str):
    """
    Load the original train/test .pt files, attach g.cluster_label to every
    graph (spectral label, int), and save as new .pt files in out_dir.

    Graphs that were skipped during feature extraction (too small / degenerate)
    get cluster_label = -1 so they are easily filtered out at training time.
    """
    import torch as _torch

    # Build filename → cluster label lookup from records
    label_map = {rec["filename"]: int(sp_labels[i]) for i, rec in enumerate(records)}

    os.makedirs(out_dir, exist_ok=True)

    for pt_path, split in [(train_pt, "train"), (test_pt, "test")]:
        graphs = _torch.load(pt_path, map_location="cpu", weights_only=False)
        n_labelled = n_skipped = 0

        for g in graphs:
            fname = getattr(g, "filename", None)
            if fname and fname in label_map:
                g.cluster_label = label_map[fname]
                n_labelled += 1
            else:
                g.cluster_label = -1   # degenerate / not clustered
                n_skipped += 1

        out_path = os.path.join(out_dir, f"{split}_graphs_clustered.pt")
        _torch.save(graphs, out_path)
        print(f"  Saved {split} clustered graphs → {out_path}")
        print(f"    labelled={n_labelled}  skipped/degenerate={n_skipped}")

        # Print cluster distribution
        from collections import Counter
        dist = Counter(g.cluster_label for g in graphs)
        for lbl, cnt in sorted(dist.items()):
            tag = "degenerate" if lbl == -1 else f"cluster {lbl}"
            print(f"    {tag}: {cnt} graphs")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crack Graph Clustering Pipeline")
    parser.add_argument("--train",      default=None, help="Path to train folder")
    parser.add_argument("--test",       default=None, help="Path to test folder")
    parser.add_argument("--n_clusters", default=None, type=int,
                        help="Fixed number of clusters (omit = auto-detect)")
    parser.add_argument("--output",     default=None, help="Output folder")
    args = parser.parse_args()

    cfg = Config()
    if args.train:      cfg.TRAIN_DIR  = args.train
    if args.test:       cfg.TEST_DIR   = args.test
    if args.n_clusters: cfg.N_CLUSTERS = args.n_clusters
    if args.output:     cfg.OUTPUT_DIR = args.output

    print("=" * 55)
    print("  Crack Graph Clustering Pipeline")
    print(f"  Train : {cfg.TRAIN_DIR}")
    print(f"  Test  : {cfg.TEST_DIR}")
    print(f"  Output: {cfg.OUTPUT_DIR}")
    print("=" * 55)

    # 1. Build graphs + extract features
    def load_split(path, split):
        # If path points to a .pt file, load graphs directly
        if os.path.isfile(path) and str(path).lower().endswith(".pt"):
            graphs = load_graphs_from_pt(path)
            return graphs_to_records(graphs, split, cfg)
        # Otherwise treat path as a folder of mask images
        return process_folder(path, split, cfg)

    records = load_split(cfg.TRAIN_DIR, "train") + load_split(cfg.TEST_DIR, "test")
    print(f"\nTotal valid graphs: {len(records)}")

    if len(records) < 4:
        sys.exit("Not enough valid graphs — check your dataset paths.")

    # 2. Feature matrix — scale-invariant shape features only
    X = np.array([[r[k] for k in SHAPE_FEAT_KEYS] for r in records])
    # Sanity check: no NaN/inf (would indicate a ratio blew up despite eps guards)
    bad = np.sum(~np.isfinite(X))
    if bad:
        print(f"  WARNING: {bad} non-finite values in feature matrix — check epsilon guards")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Ratios are naturally bounded so winsorization is mild; clip at 99th pct
    # only as a safety net for any remaining edge cases
    p99 = np.percentile(X, 99, axis=0)
    X   = np.clip(X, None, p99)

    X_scaled = RobustScaler().fit_transform(X)
    print(f"Feature matrix: {X.shape}  shape-only features, winsorized at 99th pct, RobustScaler")

    # 3. Choose K
    k = cfg.N_CLUSTERS if cfg.N_CLUSTERS else find_optimal_k(
        X_scaled, max_k=7, random_state=cfg.RANDOM_STATE
    )

    # 4. Cluster
    sp_labels = spectral_cluster(X_scaled, k, cfg.RANDOM_STATE)
    km_labels = kmeans_cluster(X_scaled,   k, cfg.RANDOM_STATE)

    # 5. Print summaries
    print_summary(records, sp_labels, "Spectral Clustering")
    print_summary(records, km_labels, "K-Means")

    # 6. Agreement
    agreement = np.mean(sp_labels == km_labels) * 100
    print(f"\n  Method agreement: {agreement:.1f}%  "
          f"(high = clusters are stable and reliable)")

    # 7. Size-correlation diagnostic — if |r| > 0.5 size leaked back in
    n_nodes_arr = np.array([r["_n_nodes"] for r in records], dtype=float)
    corr = float(np.corrcoef(sp_labels.astype(float), n_nodes_arr)[0, 1])
    print(f"\n[Size-correlation check]  corr(spectral_label, n_nodes) = {corr:+.3f}")
    if abs(corr) > 0.5:
        print("  WARNING: high size correlation — clusters may be driven by graph size, not shape")
    else:
        print("  OK: low size correlation — clusters reflect crack shape, not size")

    # 8. Save CSV + visualise
    save_labels(records, sp_labels, km_labels, cfg.OUTPUT_DIR)
    visualise(X_scaled, sp_labels, km_labels, cfg.OUTPUT_DIR)

    # 9. Write cluster labels back into the original .pt graph files
    #    Each graph gets g.cluster_label (int) added as an attribute.
    #    New files saved alongside the CSV so the GNN training pipeline
    #    can load them directly and filter/stratify by cluster.
    if (os.path.isfile(cfg.TRAIN_DIR) and cfg.TRAIN_DIR.endswith(".pt") and
            os.path.isfile(cfg.TEST_DIR)  and cfg.TEST_DIR.endswith(".pt")):
        save_clustered_pt(
            train_pt   = cfg.TRAIN_DIR,
            test_pt    = cfg.TEST_DIR,
            records    = records,
            sp_labels  = sp_labels,
            out_dir    = cfg.OUTPUT_DIR,
        )

    print("\n✓ Done.")
    print("  spectral_label in crack_pseudo_labels.csv  — for analysis")
    print("  cluster_label  in *_clustered.pt files     — for GNN training")


if __name__ == "__main__":
    main()
