#!/usr/bin/env python3
"""
End-to-end pipeline demo: raw image → graph → GNN prediction → figure.

Demonstrates all three stages of the crack topology pipeline:
  Stage 1: Raw image + ground-truth mask  (segmentation output proxy)
  Stage 2: mask_to_graph()  →  PyG Data  (graph construction)
  Stage 3: GINE encoder     →  link prediction scores

Output: one 4-panel PNG per image saved to --output/

Usage
-----
    cd crack-topology-gnn
    source ../.venv/bin/activate

    python3 pipeline_demo.py \\
        --mask-dir "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/masks" \\
        --img-dir  "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/images" \\
        --ckpt     outputs/linkpred_200ep/gine/best_model.pt \\
        --n        8 \\
        --output   outputs/pipeline_demo
"""

import os
import sys
import random
import argparse

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from link_prediction.model   import build_encoder, MLPEdgePredictor, MLPNodePredictor
from link_prediction.masking import apply_node_mask
from link_prediction.splits  import transductive_split
from image_to_graph.convert  import mask_to_graph

DEVICE = torch.device('cpu')


# ── Model loading ──────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, arch: str = 'gine'):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved   = ckpt.get('args', {})
    hidden  = saved.get('hidden',  128)
    out_dim = saved.get('out_dim', 64)
    heads   = saved.get('heads',   4)
    dropout = saved.get('dropout', 0.3)

    encoder   = build_encoder(arch, in_channels=6, hidden=hidden,
                               out_dim=out_dim, heads=heads, dropout=dropout)
    edge_pred = MLPEdgePredictor(out_dim, hidden=hidden,     dropout=dropout)
    node_pred = MLPNodePredictor(out_dim, hidden=hidden // 2, dropout=dropout)

    encoder.load_state_dict(ckpt['encoder'])
    edge_pred.load_state_dict(ckpt['edge_pred'])
    node_pred.load_state_dict(ckpt['node_pred'])

    encoder.eval(); edge_pred.eval(); node_pred.eval()
    return encoder, edge_pred, node_pred, ckpt.get('best_epoch', '?')


# ── Per-graph inference ────────────────────────────────────────────────────────

def run_inference(encoder, edge_pred, node_pred, data, mask_frac=0.30, seed=42):
    with torch.no_grad():
        # Node task: hide crack tips
        masked, node_labels, hidden_mask, eval_mask = apply_node_mask(
            data, mask_frac=mask_frac, seed=seed, node_type='endpoint'
        )
        x  = masked.x.float()
        ei = masked.edge_index
        ea = masked.edge_attr.float() if masked.edge_attr is not None else None
        z  = encoder(x, ei, ea)
        node_scores = torch.sigmoid(node_pred(z)).squeeze(-1)

        # Edge task: transductive split
        split = transductive_split(data, num_val=0.0, num_test=0.30,
                                   seed=seed, num_test_min_edges=2)
        if split is None:
            return None

        train_data      = split['train_data']
        label_ei        = split['test_ei']
        edge_labels     = split['test_labels']
        hidden_edge_ei  = split['test_pos_ei']   # the edges that were actually hidden

        x2  = train_data.x.float()
        ei2 = train_data.edge_index
        ea2 = train_data.edge_attr.float() if train_data.edge_attr is not None else None
        z2  = encoder(x2, ei2, ea2)
        edge_scores = torch.sigmoid(edge_pred(z2, label_ei)).squeeze(-1)

    return {
        'original':       data,
        'masked':         masked,
        'node_labels':    node_labels.numpy(),
        'eval_mask':      eval_mask.numpy(),
        'hidden_mask':    hidden_mask.numpy(),
        'node_scores':    node_scores.numpy(),
        'train_data':     train_data,
        'label_ei':       label_ei.numpy(),
        'edge_scores':    edge_scores.numpy(),
        'edge_labels':    edge_labels.numpy(),
        'hidden_edge_ei': hidden_edge_ei.numpy(),
    }


# ── Figure generation ──────────────────────────────────────────────────────────

def _node_pos(data, H, W):
    xs = data.x[:, 0].numpy() * W
    ys = data.x[:, 1].numpy() * H
    return xs, ys


def draw_figure(result, binary_mask, skeleton, orig_img, save_path, title=''):
    H, W     = binary_mask.shape
    has_img  = orig_img is not None
    n_cols   = 5 if has_img else 4

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(title, color='white', fontsize=9)

    col = 0

    # ── Shared references ─────────────────────────────────────────────────────
    orig_data   = result['original']
    hidden_mask = result['hidden_mask']
    hei         = result['hidden_edge_ei']

    oxs = orig_data.x[:, 0].numpy() * W
    oys = orig_data.x[:, 1].numpy() * H
    oei = orig_data.edge_index.numpy()

    # Set of edge-task hidden pairs (on original node ids)
    hidden_edge_set = set()
    for i in range(hei.shape[1]):
        hidden_edge_set.add((min(int(hei[0,i]), int(hei[1,i])),
                             max(int(hei[0,i]), int(hei[1,i]))))

    def _draw_skel_bg(ax):
        ax.imshow(binary_mask, cmap='gray', alpha=0.25, extent=[0, W, H, 0])

    # ── Panel 1: original image ────────────────────────────────────────────────
    if has_img:
        ax = axes[col]; col += 1
        ax.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        ax.set_title('Panel 1 — Raw Crack Image\n(Stage 1 segmentation output)',
                     color='white', fontsize=9)
        ax.axis('off')

    # ── Panel 2: segmentation + full graph ────────────────────────────────────
    ax = axes[col]; col += 1
    ax.set_facecolor('#1a1a1a')
    ax.imshow(binary_mask, cmap='gray', alpha=0.4, extent=[0, W, H, 0])
    skel_rgba = np.zeros((H, W, 4), dtype=np.float32)
    skel_rgba[skeleton > 0] = [0.0, 0.9, 1.0, 0.9]
    ax.imshow(skel_rgba, extent=[0, W, H, 0])

    drawn = set()
    for i in range(oei.shape[1]):
        u, v = int(oei[0,i]), int(oei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([oxs[u], oxs[v]], [oys[u], oys[v]],
                color='cyan', lw=0.9, alpha=0.6, zorder=3)
    for i in range(orig_data.num_nodes):
        c = '#4da6ff' if int(orig_data.x[i, 4].item()) == 1 else '#ffffff'
        ax.scatter(oxs[i], oys[i], c=c, s=22, zorder=4,
                   edgecolors='black', linewidths=0.4)

    leg2 = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#4da6ff',
               markersize=7, label='Endpoint (crack tip)'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='white',
               markersize=7, label='Chain / junction'),
    ]
    ax.legend(handles=leg2, loc='best', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.5)
    ax.set_title('Panel 2 — Segmentation + Graph\nmask → skeleton → nodes & edges',
                 color='white', fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    # ── Panel 3: combined hidden ground truth ─────────────────────────────────
    # Shows what was hidden for BOTH tasks on the full graph:
    #   Yellow X  + yellow dashed edge  → node task  (hidden crack tip + its edges)
    #   Magenta dashed edge             → edge task  (hidden crack segment)
    ax = axes[col]; col += 1
    ax.set_facecolor('#1a1a1a')
    _draw_skel_bg(ax)

    drawn = set()
    for i in range(oei.shape[1]):
        u, v = int(oei[0,i]), int(oei[1,i])
        key  = (min(u,v), max(u,v))
        if key in drawn: continue
        drawn.add(key)
        node_hidden  = hidden_mask[u] or hidden_mask[v]
        edge_hidden  = key in hidden_edge_set
        if node_hidden:
            c, lw, ls, zo = '#ffdd00', 2.0, '--', 4   # yellow — node task removal
        elif edge_hidden:
            c, lw, ls, zo = '#ff44cc', 2.0, '--', 4   # magenta — edge task removal
        else:
            c, lw, ls, zo = '#555555', 0.8, '-',  2   # grey — visible
        ax.plot([oxs[u], oxs[v]], [oys[u], oys[v]],
                color=c, lw=lw, linestyle=ls, alpha=0.9, zorder=zo)

    for i in range(orig_data.num_nodes):
        if hidden_mask[i]:
            ax.scatter(oxs[i], oys[i], marker='x', c='#ffdd00', s=130,
                       linewidths=2.5, zorder=6)
        else:
            c = '#4da6ff' if int(orig_data.x[i, 4].item()) == 1 else '#ffffff'
            ax.scatter(oxs[i], oys[i], c=c, s=22, zorder=4,
                       edgecolors='black', linewidths=0.4)

    leg3 = [
        Line2D([0],[0], marker='x', color='none', markerfacecolor='#ffdd00',
               markersize=9, markeredgecolor='#ffdd00', lw=2,
               label='Hidden node (node task)'),
        Line2D([0],[0], color='#ffdd00', lw=2, linestyle='--',
               label='Removed edge (node task)'),
        Line2D([0],[0], color='#ff44cc', lw=2, linestyle='--',
               label='Hidden edge (edge task)'),
        Line2D([0],[0], color='#555555', lw=1,
               label='Visible edge'),
    ]
    ax.legend(handles=leg3, loc='best', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.0)
    ax.set_title('Panel 3 — Ground Truth: What Was Hidden\nyellow=node-task  magenta=edge-task',
                 color='white', fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    # ── Panel 4: node prediction ───────────────────────────────────────────────
    ax = axes[col]; col += 1
    ax.set_facecolor('#1a1a1a')
    _draw_skel_bg(ax)

    masked_data = result['masked']
    mxs = masked_data.x[:, 0].numpy() * W
    mys = masked_data.x[:, 1].numpy() * H
    mei = masked_data.edge_index.numpy()

    eval_mask   = result['eval_mask']
    node_labels = result['node_labels']
    node_scores = result['node_scores']

    drawn = set()
    for i in range(mei.shape[1]):
        u, v = int(mei[0,i]), int(mei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([mxs[u], mxs[v]], [mys[u], mys[v]],
                color='gray', lw=0.6, alpha=0.4, zorder=2)

    for i in range(masked_data.num_nodes):
        px, py = mxs[i], mys[i]
        if hidden_mask[i]:
            ax.scatter(px, py, marker='x', c='#ffdd00', s=100, linewidths=2.0, zorder=6)
        elif eval_mask[i]:
            score  = float(np.atleast_1d(node_scores)[i])
            is_tp  = int(node_labels[i]) == 1 and score >= 0.5
            is_fn  = int(node_labels[i]) == 1 and score <  0.5
            is_fp  = int(node_labels[i]) == 0 and score >= 0.5
            c = '#00ee55' if is_tp else '#ff8800' if is_fn else '#ff3333' if is_fp else '#999999'
            ax.scatter(px, py, c=c, s=60, marker='D', zorder=4,
                       edgecolors='black', linewidths=0.4)
        else:
            ax.scatter(px, py, c='white', s=20, zorder=3,
                       edgecolors='black', linewidths=0.3)

    leg4 = [
        Line2D([0],[0], marker='x', color='none', markerfacecolor='#ffdd00',
               markersize=8, markeredgecolor='#ffdd00', lw=2, label='Hidden tip (GT)'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='#00ee55',
               markersize=7, label='TP — correctly flagged'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='#ff8800',
               markersize=7, label='FN — missed'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='#ff3333',
               markersize=7, label='FP — false alarm'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='#999999',
               markersize=7, label='TN — correctly silent'),
    ]
    ax.legend(handles=leg4, loc='best', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.0)
    ax.set_title('Panel 4 — Node Prediction\n(which visible node lost a crack tip?)',
                 color='white', fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    # ── Panel 5: edge prediction ───────────────────────────────────────────────
    ax = axes[col]; col += 1
    ax.set_facecolor('#1a1a1a')
    _draw_skel_bg(ax)

    td  = result['train_data']
    txs = td.x[:, 0].numpy() * W
    tys = td.x[:, 1].numpy() * H
    tei = td.edge_index.numpy()

    # Visible edges in blue
    drawn = set()
    for i in range(tei.shape[1]):
        u, v = int(tei[0,i]), int(tei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([txs[u], txs[v]], [tys[u], tys[v]],
                color='royalblue', lw=1.0, alpha=0.7, zorder=2)

    # Hidden edges as magenta dashed (matches panel 3 colour)
    for u, v in hidden_edge_set:
        if u < len(txs) and v < len(txs):
            ax.plot([txs[u], txs[v]], [tys[u], tys[v]],
                    color='#ff44cc', lw=1.5, linestyle='--', alpha=0.5, zorder=3)

    # Model predictions overlaid
    lei     = result['label_ei']
    escores = np.atleast_1d(result['edge_scores'])
    elabels = np.atleast_1d(result['edge_labels'])
    drawn   = set()
    for i in range(len(elabels)):
        u, v = int(lei[0,i]), int(lei[1,i])
        key  = (min(u,v), max(u,v))
        if key in drawn: continue
        drawn.add(key)
        if u >= len(txs) or v >= len(txs): continue
        is_hidden = int(elabels[i]) == 1
        predicted = escores[i] >= 0.5
        if   is_hidden and     predicted: c, lw, ls, zo = '#00ee55', 2.5, '-',  5  # TP
        elif is_hidden and not predicted: c, lw, ls, zo = '#ff8800', 1.5, '--', 4  # FN
        elif not is_hidden and predicted: c, lw, ls, zo = '#ff3333', 1.5, '--', 4  # FP
        else: continue
        ax.plot([txs[u], txs[v]], [tys[u], tys[v]],
                color=c, lw=lw, linestyle=ls, alpha=0.9, zorder=zo)

    for i in range(td.num_nodes):
        ax.scatter(txs[i], tys[i], c='white', s=18, zorder=6,
                   edgecolors='black', linewidths=0.3)

    leg5 = [
        Line2D([0],[0], color='royalblue', lw=1.5,               label='Visible edges'),
        Line2D([0],[0], color='#ff44cc',   lw=1.5, linestyle='--', label='Hidden (GT)'),
        Line2D([0],[0], color='#00ee55',   lw=2.5,               label='TP — correctly predicted'),
        Line2D([0],[0], color='#ff8800',   lw=1.5, linestyle='--', label='FN — missed'),
        Line2D([0],[0], color='#ff3333',   lw=1.5, linestyle='--', label='FP — false alarm'),
    ]
    ax.legend(handles=leg5, loc='best', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.0)
    ax.set_title('Panel 5 — Edge Prediction\n(magenta=hidden GT, green=TP, orange=FN)',
                 color='white', fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mask-dir', required=True,
                   help='Directory of binary crack masks (test split)')
    p.add_argument('--img-dir',  default=None,
                   help='Directory of original RGB images (same filenames)')
    p.add_argument('--ckpt',     default='outputs/linkpred_200ep/gine/best_model.pt',
                   help='GINE checkpoint (best_model.pt)')
    p.add_argument('--arch',     default='gine',
                   choices=['mlp', 'gcn', 'sage', 'gine', 'gat'])
    p.add_argument('--n',        type=int, default=8,
                   help='Number of images to process')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--mask-frac', type=float, default=0.30)
    p.add_argument('--output',   default='outputs/pipeline_demo')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print(f'Loading checkpoint: {args.ckpt}')
    encoder, edge_pred, node_pred, best_ep = load_checkpoint(args.ckpt, arch=args.arch)
    print(f'  {args.arch.upper()} loaded — best epoch {best_ep}')

    masks = sorted([f for f in os.listdir(args.mask_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    random.shuffle(masks)
    masks = masks[:args.n]

    done = 0
    for fname in masks:
        mask_path = os.path.join(args.mask_dir, fname)
        data, skeleton, nx_graph, skip = mask_to_graph(mask_path, split='test')
        if skip:
            print(f'  Skip {fname}: {skip}')
            continue

        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(binary_mask, 127, 1, cv2.THRESH_BINARY)

        orig_img = None
        if args.img_dir:
            img_path = os.path.join(args.img_dir, fname)
            if os.path.exists(img_path):
                orig_img = cv2.imread(img_path)

        result = run_inference(encoder, edge_pred, node_pred,
                               data, mask_frac=args.mask_frac, seed=args.seed)
        if result is None:
            print(f'  Skip {fname}: graph too small for edge split')
            continue

        stem      = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output, stem + '.png')
        draw_figure(result, binary_mask,
                    skeleton if skeleton is not None else np.zeros_like(binary_mask),
                    orig_img, save_path, title=fname)
        print(f'  [{done+1}/{args.n}] {fname}  →  {save_path}')
        done += 1

    print(f'\nDone — {done} figures saved to {args.output}/')


if __name__ == '__main__':
    main()
