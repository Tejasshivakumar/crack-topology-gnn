"""
clustering/stratified_eval.py
==============================
Two analyses on the clustered crack graphs:

  Path A — Stratified evaluation
    Loads trained model checkpoints from outputs/linkpred_clean_50ep/ and re-runs
    evaluate_node_task / evaluate_edge_task separately on each k-means stratum
    (Stratum 0: low-branching / linear-ish; Stratum 1: high-branching / networked).
    Reports AP/AUC per stratum so we can see whether GNN advantage concentrates in
    topologically complex cracks.

  Path B — Quality filter
    Spectral Cluster 1 (332 graphs, zero junctions, avg_degree ≈ 1) are degenerate
    skeleton artifacts. This script identifies and excludes them, then re-runs the
    best model (GINE) on the cleaned set. Confirms whether removing artifacts changes
    the topology-proof result.

Usage
-----
    cd "/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn"
    source ../.venv/bin/activate

    # Both analyses (default)
    python3 clustering/stratified_eval.py

    # Path A only
    python3 clustering/stratified_eval.py --analysis stratified

    # Path B only
    python3 clustering/stratified_eval.py --analysis quality_filter

    # Custom checkpoint dir
    python3 clustering/stratified_eval.py --ckpt-dir outputs/linkpred_clean_50ep

Output
------
    outputs/clustering/stratified_results.json   — per-stratum AP/AUC for all models
    outputs/clustering/quality_filter_results.json — GINE results before/after filter
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Make sure link_prediction module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from link_prediction.evaluate import evaluate_node_task, evaluate_edge_task
from link_prediction.train import build_models


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CKPT_DIR      = "outputs/linkpred_clean_50ep"
DEFAULT_CLUSTERED_DIR = "outputs/clustering"
MODEL_NAMES           = ["mlp", "gcn", "sage", "gine", "gat"]

# Hyperparameters matching the canonical clean run
HIDDEN  = 128
OUT_DIM = 64
HEADS   = 4
DROPOUT = 0.3


# ── Model loading ──────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: str, model_name: str, device: torch.device):
    """Load encoder + heads from saved checkpoint. Returns (encoder, edge_pred, node_pred)."""
    ckpt_path = os.path.join(ckpt_dir, model_name, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder, edge_pred, node_pred = build_models(
        in_channels=6, hidden=HIDDEN, out_dim=OUT_DIM,
        heads=HEADS, dropout=DROPOUT, model_name=model_name,
    )
    encoder.load_state_dict(ckpt["encoder"])
    edge_pred.load_state_dict(ckpt["edge_pred"])
    node_pred.load_state_dict(ckpt["node_pred"])
    encoder.eval(); edge_pred.eval(); node_pred.eval()
    return encoder.to(device), edge_pred.to(device), node_pred.to(device)


# ── Path A: Stratified evaluation ─────────────────────────────────────────────

def run_stratified(ckpt_dir: str, clustered_dir: str, device: torch.device,
                   model_names: list = MODEL_NAMES) -> dict:
    """
    Evaluate all models on each k-means stratum separately.
    Stratum 0: low-branching (linear-ish, 1514 train / ~253 test)
    Stratum 1: high-branching (networked,  2850 train / ~476 test)
    """
    test_pt = os.path.join(clustered_dir, "test_graphs_clustered.pt")
    if not os.path.exists(test_pt):
        sys.exit(f"ERROR: clustered test file not found at {test_pt}\n"
                 f"Run 'python3 clustering/cluster.py' first.")

    all_graphs = torch.load(test_pt, map_location="cpu", weights_only=False)

    # Split by k-means label (attribute written by cluster.py as spectral_label;
    # we want k-means here for Path A — reload from CSV for the k-means column)
    kmeans_csv = os.path.join(clustered_dir, "crack_pseudo_labels.csv")
    if os.path.exists(kmeans_csv):
        import csv
        km_map = {}
        with open(kmeans_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                km_map[row["filename"]] = int(row["kmeans_label"])
        strata = {0: [], 1: []}
        for g in all_graphs:
            fname = getattr(g, "filename", None)
            lbl   = km_map.get(fname, -1)
            if lbl in strata:
                strata[lbl].append(g)
    else:
        # Fall back to cluster_label (spectral) if CSV not present
        print("  (no CSV found — using cluster_label attribute for strata)")
        strata = {0: [], 1: []}
        for g in all_graphs:
            lbl = int(getattr(g, "cluster_label", -1))
            if lbl in strata:
                strata[lbl].append(g)

    print(f"\n[Path A — Stratified Evaluation]")
    print(f"  Stratum 0 (low-branching):  {len(strata[0])} test graphs")
    print(f"  Stratum 1 (high-branching): {len(strata[1])} test graphs")

    results = {}
    for model_name in model_names:
        print(f"\n  {model_name.upper()}")
        try:
            encoder, edge_pred, node_pred = load_checkpoint(ckpt_dir, model_name, device)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        results[model_name] = {}
        for stratum_id, graphs in strata.items():
            if len(graphs) < 4:
                print(f"    Stratum {stratum_id}: too few graphs ({len(graphs)}), skip")
                results[model_name][f"stratum_{stratum_id}"] = None
                continue

            node_m = evaluate_node_task(encoder, node_pred, graphs, device)
            edge_m = evaluate_edge_task(encoder, edge_pred, graphs, device)
            m = {**node_m, **edge_m, "n_graphs": len(graphs)}
            results[model_name][f"stratum_{stratum_id}"] = m
            print(f"    Stratum {stratum_id} (n={len(graphs):4d}): "
                  f"node_ap={m['node_ap']:.4f}  edge_ap={m['edge_ap']:.4f}")

    # Print summary comparison table
    print("\n" + "═" * 70)
    print("  STRATIFIED RESULTS — Node AP")
    print(f"  {'Model':<8}  {'Stratum 0 (linear)':<22}  {'Stratum 1 (branched)':<22}  {'Gap'}")
    print("─" * 70)
    for mn, res in results.items():
        s0 = res.get("stratum_0") or {}
        s1 = res.get("stratum_1") or {}
        ap0 = s0.get("node_ap", float("nan"))
        ap1 = s1.get("node_ap", float("nan"))
        gap = ap1 - ap0 if (s0 and s1) else float("nan")
        print(f"  {mn:<8}  {ap0:<22.4f}  {ap1:<22.4f}  {gap:+.4f}")
    print("═" * 70)

    return results


# ── Path B: Quality filter ─────────────────────────────────────────────────────

def run_quality_filter(ckpt_dir: str, clustered_dir: str,
                       device: torch.device,
                       model_names: list = MODEL_NAMES) -> dict:
    """
    Spectral Cluster 1 graphs (zero junctions, avg_degree ≈ 1) are degenerate
    skeleton artifacts. Run GINE evaluation before and after filtering them out.
    """
    test_pt = os.path.join(clustered_dir, "test_graphs_clustered.pt")
    if not os.path.exists(test_pt):
        sys.exit(f"ERROR: clustered test file not found at {test_pt}\n"
                 f"Run 'python3 clustering/cluster.py' first.")

    all_graphs = torch.load(test_pt, map_location="cpu", weights_only=False)

    # cluster_label is spectral label set by save_clustered_pt()
    degenerate = [g for g in all_graphs if getattr(g, "cluster_label", -1) == 1]
    clean      = [g for g in all_graphs if getattr(g, "cluster_label", -1) != 1]

    print(f"\n[Path B — Quality Filter]")
    print(f"  Total test graphs: {len(all_graphs)}")
    print(f"  Spectral Cluster 1 (degenerate, no junctions): {len(degenerate)}")
    print(f"  Remaining after filter:                        {len(clean)}")

    results = {}
    for model_name in model_names:
        print(f"\n  {model_name.upper()}")
        try:
            encoder, edge_pred, node_pred = load_checkpoint(ckpt_dir, model_name, device)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        node_all  = evaluate_node_task(encoder, node_pred, all_graphs, device)
        edge_all  = evaluate_edge_task(encoder, edge_pred, all_graphs, device)
        node_clean = evaluate_node_task(encoder, node_pred, clean, device)
        edge_clean = evaluate_edge_task(encoder, edge_pred, clean, device)

        results[model_name] = {
            "all_graphs":   {**node_all,  **edge_all,  "n_graphs": len(all_graphs)},
            "clean_graphs": {**node_clean, **edge_clean, "n_graphs": len(clean)},
        }
        print(f"    All    (n={len(all_graphs)}): node_ap={node_all['node_ap']:.4f}  "
              f"edge_ap={edge_all['edge_ap']:.4f}")
        print(f"    Clean  (n={len(clean)}): node_ap={node_clean['node_ap']:.4f}  "
              f"edge_ap={edge_clean['edge_ap']:.4f}")
        delta_node = node_clean["node_ap"] - node_all["node_ap"]
        print(f"    Δ node_ap = {delta_node:+.4f}  "
              f"({'improved' if delta_node > 0 else 'unchanged'} after removing artifacts)")

    print("\n" + "═" * 70)
    print("  QUALITY FILTER SUMMARY — Node AP (all vs clean)")
    print(f"  {'Model':<8}  {'All graphs':<14}  {'After filter':<14}  {'Delta'}")
    print("─" * 70)
    for mn, res in results.items():
        ap_all   = res["all_graphs"]["node_ap"]
        ap_clean = res["clean_graphs"]["node_ap"]
        print(f"  {mn:<8}  {ap_all:<14.4f}  {ap_clean:<14.4f}  {ap_clean - ap_all:+.4f}")
    print("═" * 70)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stratified + quality-filter evaluation")
    parser.add_argument("--ckpt-dir",      default=DEFAULT_CKPT_DIR)
    parser.add_argument("--clustered-dir", default=DEFAULT_CLUSTERED_DIR)
    parser.add_argument("--analysis",      default="both",
                        choices=["both", "stratified", "quality_filter"])
    parser.add_argument("--models",        nargs="+", default=MODEL_NAMES)
    args = parser.parse_args()

    model_names = args.models
    device = torch.device("cpu")  # CPU eval avoids MPS OOM on large eval sets

    os.makedirs(args.clustered_dir, exist_ok=True)
    out = {}

    if args.analysis in ("both", "stratified"):
        out["stratified"] = run_stratified(args.ckpt_dir, args.clustered_dir, device, model_names)

    if args.analysis in ("both", "quality_filter"):
        out["quality_filter"] = run_quality_filter(args.ckpt_dir, args.clustered_dir, device, model_names)

    out_path = os.path.join(args.clustered_dir, "stratified_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
