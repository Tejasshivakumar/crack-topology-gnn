#!/usr/bin/env python3
"""
C3 — End-to-End Pipeline Evaluation.

Measures the AP degradation when using Stage 1 predicted masks vs oracle GT masks.

For each test image:
  Oracle path:    GT mask → Stage 2 (mask_to_graph) → Stage 3 (GINE) → node AP
  Predicted path: HybridGraphUNet → Stage 2 → Stage 3 → node AP

The gap (oracle AP − predicted AP) is the pipeline error budget from segmentation noise.

Usage:
    cd "/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn"
    source "../.venv/bin/activate"

    python3 eval_e2e.py \\
        --seg-ckpt  outputs/seg_compare/hybrid/checkpoints/crack_hybrid_gnn-epepoch=46-ciou=val/crack_iou=0.6299.ckpt \\
        --gnn-ckpt  outputs/linkpred_200ep/gine/best_model.pt \\
        --img-dir   "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/images" \\
        --mask-dir  "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/masks" \\
        --output    outputs/e2e_results.json \\
        --n         100
"""

import os
import sys
import json
import random
import argparse

import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segmentation'))

from segmentation.model       import HybridGraphUNet
from image_to_graph.convert   import mask_to_graph
from link_prediction.model    import build_encoder, MLPNodePredictor
from link_prediction.masking  import apply_node_mask, is_valid_for_node_task

# ImageNet normalisation (matches segmentation training)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 448


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Segmentation model ────────────────────────────────────────────────────────

def load_seg_model(ckpt_path: str, device: torch.device):
    """Load HybridGraphUNet from a Lightning checkpoint."""
    model = HybridGraphUNet(
        encoder_name='resnet34d',
        pretrained=False,           # weights come from checkpoint
        out_channels=2,
        use_gnn_bottleneck=True,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    # Strip Lightning's 'model.' prefix
    sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [seg] Missing keys: {missing[:3]}{"..." if len(missing)>3 else ""}')
    model.to(device).eval()
    return model


def predict_mask(seg_model, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run HybridGraphUNet on one image; return binary mask (uint8, 0/255)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img_norm = (img_resized - IMG_MEAN) / IMG_STD                   # [H, W, 3]
    tensor   = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = seg_model(tensor)          # [1, 1, H, W]  or  [1, 2, H, W]
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            binary = (prob > 0.5).astype(np.uint8) * 255
        else:
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy()
            binary = (pred * 255).astype(np.uint8)

    return binary                           # [448, 448] uint8


# ── GNN model ─────────────────────────────────────────────────────────────────

def load_gnn(ckpt_path: str):
    ckpt    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved   = ckpt.get('args', {})
    hidden  = saved.get('hidden',  128)
    out_dim = saved.get('out_dim', 64)
    heads   = saved.get('heads',   4)
    dropout = saved.get('dropout', 0.3)

    encoder   = build_encoder('gine', in_channels=6, hidden=hidden,
                               out_dim=out_dim, heads=heads, dropout=dropout)
    node_pred = MLPNodePredictor(out_dim, hidden=hidden // 2, dropout=dropout)

    encoder.load_state_dict(ckpt['encoder'])
    node_pred.load_state_dict(ckpt['node_pred'])
    encoder.eval()
    node_pred.eval()
    print(f'GINE loaded  |  best epoch: {ckpt.get("best_epoch", "?")}')
    return encoder, node_pred


# ── Graph → node AP ───────────────────────────────────────────────────────────

def graph_node_ap(encoder, node_pred, graph, mask_frac=0.20, seed=0):
    """Run node masking eval on one graph; return AP or None if unusable."""
    if not is_valid_for_node_task(graph):
        return None

    masked, node_labels, _, eval_mask = apply_node_mask(
        graph, mask_frac=mask_frac, seed=seed
    )
    if eval_mask.sum() == 0:
        return None

    labels_ev = node_labels[eval_mask]
    if len(torch.unique(labels_ev)) < 2:
        return None

    x  = masked.x.float()
    ei = masked.edge_index
    ea = masked.edge_attr.float() if masked.edge_attr is not None else None

    with torch.no_grad():
        z     = encoder(x, ei, ea)
        probs = torch.sigmoid(node_pred(z))[eval_mask].cpu().numpy()

    lbls = labels_ev.cpu().numpy()
    return float(average_precision_score(lbls, probs))


# ── Mask → graph (with error handling) ───────────────────────────────────────

def mask_to_graph_safe(mask_uint8: np.ndarray):
    """Write mask to a temp file, call mask_to_graph, return PyG Data or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, mask_uint8)
        data, _, _, skip_reason = mask_to_graph(tmp_path)
        os.unlink(tmp_path)
        if skip_reason is not None or data is None:
            return None
        if data.x is None or data.x.shape[0] < 3:
            return None
        return data
    except Exception:
        return None


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seg-ckpt',  default='outputs/seg_compare/hybrid/checkpoints/'
                                           'crack_hybrid_gnn-epepoch=46-ciou=val/crack_iou=0.6299.ckpt')
    p.add_argument('--gnn-ckpt',  default='outputs/linkpred_200ep/gine/best_model.pt')
    p.add_argument('--img-dir',   default='/Users/tejasskamar/Practicum/Data Set/'
                                          'crack_seg_clean/clean/test/images')
    p.add_argument('--mask-dir',  default='/Users/tejasskamar/Practicum/Data Set/'
                                          'crack_seg_clean/clean/test/masks')
    p.add_argument('--output',    default='outputs/e2e_results.json')
    p.add_argument('--n',         type=int, default=100,
                   help='Number of test images to evaluate (default: 100)')
    p.add_argument('--mask-frac', type=float, default=0.20)
    p.add_argument('--seed',      type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    device = get_device()
    print(f'Device: {device}')

    # Load models
    print('\nLoading segmentation model...')
    seg_model = load_seg_model(args.seg_ckpt, device)

    print('Loading GNN model...')
    encoder, node_pred = load_gnn(args.gnn_ckpt)

    # Collect test images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if args.n < len(img_files):
        random.shuffle(img_files)
        img_files = img_files[:args.n]
    print(f'\nEvaluating {len(img_files)} images...\n')

    results_per_image = []
    oracle_aps, pred_aps = [], []
    skipped_oracle = 0
    skipped_pred   = 0

    for fname in tqdm(img_files):
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(args.img_dir, fname)

        # Try common mask name patterns
        mask_path = None
        for ext in ['.png', '.jpg', '.bmp']:
            candidate = os.path.join(args.mask_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            continue

        img_bgr  = cv2.imread(img_path)
        mask_gt  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gt is None:
            continue

        h, w = img_bgr.shape[:2]

        # ── Oracle path (GT mask → graph → node AP) ──────────────────────────
        gt_binary = (mask_gt > 127).astype(np.uint8) * 255
        gt_resized = cv2.resize(gt_binary, (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_NEAREST)
        oracle_graph = mask_to_graph_safe(gt_resized)

        oracle_ap = None
        if oracle_graph is not None:
            oracle_ap = graph_node_ap(encoder, node_pred, oracle_graph,
                                      mask_frac=args.mask_frac,
                                      seed=args.seed + len(results_per_image))

        # ── Predicted path (HybridGraphUNet → graph → node AP) ───────────────
        pred_mask   = predict_mask(seg_model, img_bgr, device)
        pred_graph  = mask_to_graph_safe(pred_mask)

        pred_ap = None
        if pred_graph is not None:
            pred_ap = graph_node_ap(encoder, node_pred, pred_graph,
                                    mask_frac=args.mask_frac,
                                    seed=args.seed + len(results_per_image))

        # Track
        row = {
            'image':     fname,
            'oracle_ap': oracle_ap,
            'pred_ap':   pred_ap,
            'gap':       (oracle_ap - pred_ap) if (oracle_ap is not None and pred_ap is not None) else None,
        }
        results_per_image.append(row)

        if oracle_ap is not None:
            oracle_aps.append(oracle_ap)
        else:
            skipped_oracle += 1

        if pred_ap is not None:
            pred_aps.append(pred_ap)
        else:
            skipped_pred += 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    paired = [(o, p) for o, p in zip(
        [r['oracle_ap'] for r in results_per_image],
        [r['pred_ap']   for r in results_per_image]
    ) if o is not None and p is not None]

    paired_oracle = [x[0] for x in paired]
    paired_pred   = [x[1] for x in paired]
    paired_gaps   = [o - p for o, p in paired]

    summary = {
        'n_images_evaluated':   len(img_files),
        'n_oracle_valid':        len(oracle_aps),
        'n_pred_valid':          len(pred_aps),
        'n_both_valid':          len(paired),
        'skipped_oracle':        skipped_oracle,
        'skipped_pred':          skipped_pred,

        'oracle_node_ap_mean':   float(np.mean(oracle_aps))  if oracle_aps  else None,
        'oracle_node_ap_std':    float(np.std(oracle_aps))   if oracle_aps  else None,

        'pred_node_ap_mean':     float(np.mean(pred_aps))    if pred_aps    else None,
        'pred_node_ap_std':      float(np.std(pred_aps))     if pred_aps    else None,

        'paired_oracle_ap_mean': float(np.mean(paired_oracle)) if paired else None,
        'paired_pred_ap_mean':   float(np.mean(paired_pred))   if paired else None,
        'mean_gap':              float(np.mean(paired_gaps))    if paired else None,
        'std_gap':               float(np.std(paired_gaps))     if paired else None,
        'pct_gap':               float(np.mean(paired_gaps) / np.mean(paired_oracle) * 100)
                                 if paired and np.mean(paired_oracle) > 0 else None,
    }

    output = {'summary': summary, 'per_image': results_per_image}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('END-TO-END PIPELINE EVALUATION RESULTS')
    print('='*60)
    print(f"Images evaluated:        {len(img_files)}")
    print(f"Both paths valid:        {len(paired)}")
    if paired:
        print(f"Oracle node AP (mean):   {summary['paired_oracle_ap_mean']:.4f}")
        print(f"Predicted node AP (mean):{summary['paired_pred_ap_mean']:.4f}")
        print(f"Mean gap (oracle−pred):  {summary['mean_gap']:.4f}  ({summary['pct_gap']:.1f}% relative drop)")
        print(f"Std of gap:              {summary['std_gap']:.4f}")
    else:
        print("No paired results — check mask paths and graph conversion.")
    print(f"\nSkipped (oracle graph degenerate): {skipped_oracle}")
    print(f"Skipped (pred graph degenerate):   {skipped_pred}")
    print(f"\nResults saved → {args.output}")


if __name__ == '__main__':
    main()
