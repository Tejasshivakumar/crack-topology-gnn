#!/usr/bin/env python3
"""
Frontier prediction visualizer — for qualitative figures (C4).

Produces a 3-panel figure per graph:
  Panel 1 — Full graph reference  (all nodes + edges on mask background)
  Panel 2 — After frontier masking (tip visible but ISOLATED — no edges)
  Panel 3 — Model predictions     (scored edges from isolated tip to base nodes)

Selects a mix of success (TP) and failure (FN) cases so both are represented.

Usage
-----
    cd crack-topology-gnn
    source ../.venv/bin/activate

    python3 visualize_frontier.py \\
        --ckpt      outputs/linkpred_200ep/gine/best_model.pt \\
        --graphs    outputs/clean_graphs/graphs/test_graphs.pt \\
        --mask-dir  "/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean/test/masks" \\
        --n         10 \\
        --output    outputs/frontier_figures
"""

import os, sys, argparse, random
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from link_prediction.model   import build_encoder, MLPEdgePredictor
from link_prediction.masking import frontier_edge_split, is_valid_for_frontier_task

DEVICE = torch.device('cpu')


def load_checkpoint(ckpt_path, arch='gine'):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    saved   = ckpt.get('args', {})
    hidden  = saved.get('hidden',  128)
    out_dim = saved.get('out_dim', 64)
    heads   = saved.get('heads',   4)
    dropout = saved.get('dropout', 0.3)
    encoder   = build_encoder(arch, in_channels=6, hidden=hidden,
                               out_dim=out_dim, heads=heads, dropout=dropout)
    edge_pred = MLPEdgePredictor(out_dim, hidden=hidden, dropout=dropout)
    encoder.load_state_dict(ckpt['encoder'])
    edge_pred.load_state_dict(ckpt['edge_pred'])
    encoder.eval(); edge_pred.eval()
    return encoder, edge_pred, ckpt.get('best_epoch', '?')


def load_background(fname, mask_dir):
    if mask_dir is None:
        return None, None
    path = os.path.join(mask_dir, fname)
    if not os.path.exists(path):
        return None, None
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    skel = skeletonize(binary).astype(np.uint8)
    return binary, skel


def node_xy(data, H=1, W=1):
    xs = data.x[:, 0].numpy() * W
    ys = data.x[:, 1].numpy() * H
    return xs, ys


def draw_frontier_figure(graph, split, scores, mask, skeleton, save_path, title=''):
    H = W = 1
    has_bg = mask is not None
    if has_bg:
        H, W = mask.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(title, color='white', fontsize=9)

    def bg(ax):
        ax.set_facecolor('#1a1a1a')
        if has_bg:
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[mask > 0, :3] = 0.35; rgba[mask > 0, 3] = 0.30
            rgba[skeleton > 0, 0] = 0.0
            rgba[skeleton > 0, 1] = 0.85
            rgba[skeleton > 0, 2] = 0.85
            rgba[skeleton > 0, 3] = 0.60
            ax.imshow(rgba, extent=[0, W, H, 0], origin='upper', aspect='auto', zorder=1)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    oxs, oys = node_xy(graph, H, W)
    is_ep = graph.x[:, 4].numpy()   # is_endpoint feature
    oei   = graph.edge_index.numpy()

    # ── Identify which nodes are the isolated tips ─────────────────────────────
    # hidden frontier edges: positives in label_ei where label==1
    lei    = split['label_ei'].numpy()
    labels = split['labels'].numpy()
    pos_mask = labels == 1
    pos_ei   = lei[:, pos_mask]     # [2, n_frontier]

    # Frontier tip nodes = nodes in pos_ei that are crack tips
    tip_nodes = set()
    for i in range(pos_ei.shape[1]):
        u, v = int(pos_ei[0, i]), int(pos_ei[1, i])
        if is_ep[u]:
            tip_nodes.add(u)
        if is_ep[v]:
            tip_nodes.add(v)

    # ── Panel 1: Full graph reference ──────────────────────────────────────────
    ax = axes[0]
    bg(ax)
    drawn = set()
    for i in range(oei.shape[1]):
        u, v = int(oei[0,i]), int(oei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([oxs[u], oxs[v]], [oys[u], oys[v]],
                color='royalblue', lw=1.2, alpha=0.7, zorder=2)
    for i in range(graph.num_nodes):
        c = '#ffdd00' if i in tip_nodes else ('#4da6ff' if is_ep[i] else '#ffffff')
        s = 80 if i in tip_nodes else 28
        ax.scatter(oxs[i], oys[i], c=c, s=s, zorder=4,
                   edgecolors='black', linewidths=0.5)
    ax.set_title('Panel 1 — Full Graph (reference)\nyellow = frontier crack tips',
                 color='white', fontsize=9)
    leg1 = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#ffdd00',
               markersize=8, label='Frontier tip (will be isolated)'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#4da6ff',
               markersize=7, label='Other endpoint'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='white',
               markersize=7, label='Chain / junction'),
        Line2D([0],[0], color='royalblue', lw=1.5, label='Crack edges'),
    ]
    ax.legend(handles=leg1, loc='lower right', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.5)

    # ── Panel 2: After frontier masking ───────────────────────────────────────
    ax = axes[1]
    bg(ax)
    td  = split['train_data']
    txs, tys = node_xy(td, H, W)
    tei = td.edge_index.numpy()

    drawn = set()
    for i in range(tei.shape[1]):
        u, v = int(tei[0,i]), int(tei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([txs[u], txs[v]], [tys[u], tys[v]],
                color='royalblue', lw=1.2, alpha=0.7, zorder=2)

    # Show hidden frontier edges as dashed yellow
    drawn = set()
    for i in range(pos_ei.shape[1]):
        u, v = int(pos_ei[0,i]), int(pos_ei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([oxs[u], oxs[v]], [oys[u], oys[v]],
                color='#ffdd00', lw=1.5, linestyle='--', alpha=0.6, zorder=3)

    for i in range(graph.num_nodes):
        c = '#ffdd00' if i in tip_nodes else ('#4da6ff' if is_ep[i] else '#ffffff')
        s = 80 if i in tip_nodes else 22
        ax.scatter(oxs[i], oys[i], c=c, s=s, zorder=4,
                   edgecolors='black', linewidths=0.5)
        if i in tip_nodes:
            # Draw isolation ring around isolated tip
            ax.scatter(oxs[i], oys[i], s=200, facecolors='none',
                       edgecolors='#ffdd00', linewidths=1.5, zorder=3)

    ax.set_title('Panel 2 — After Frontier Masking\nyellow tip = isolated (edges hidden)',
                 color='white', fontsize=9)
    leg2 = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#ffdd00',
               markersize=8, label='Isolated tip (visible, no edges)'),
        Line2D([0],[0], color='#ffdd00', lw=1.5, linestyle='--',
               label='Hidden edges (ground truth)'),
        Line2D([0],[0], color='royalblue', lw=1.5, label='Visible edges'),
    ]
    ax.legend(handles=leg2, loc='lower right', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.5)

    # ── Panel 3: Model predictions ─────────────────────────────────────────────
    ax = axes[2]
    bg(ax)

    # Visible edges
    drawn = set()
    for i in range(tei.shape[1]):
        u, v = int(tei[0,i]), int(tei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        ax.plot([txs[u], txs[v]], [tys[u], tys[v]],
                color='royalblue', lw=1.0, alpha=0.5, zorder=2)

    # Score all candidate pairs — colour by score, thicker = more confident
    sc_arr = np.atleast_1d(scores)
    lb_arr = np.atleast_1d(labels)
    drawn  = set()
    for i in range(len(lb_arr)):
        u, v = int(lei[0,i]), int(lei[1,i])
        if (min(u,v), max(u,v)) in drawn: continue
        drawn.add((min(u,v), max(u,v)))
        if u >= graph.num_nodes or v >= graph.num_nodes: continue
        s_val   = float(sc_arr[i])
        is_pos  = int(lb_arr[i]) == 1
        predicted = s_val >= 0.5
        if   is_pos and     predicted: c, lw, ls, zo = '#00ee55', 2.5, '-',  5  # TP
        elif is_pos and not predicted: c, lw, ls, zo = '#ff8800', 1.5, '--', 4  # FN
        elif not is_pos and predicted: c, lw, ls, zo = '#ff3333', 1.2, '--', 3  # FP
        else:
            if s_val > 0.3:  # show high-confidence TN as faint red
                c, lw, ls, zo = '#ff3333', 0.6, ':', 2
            else:
                continue
        ax.plot([oxs[u], oxs[v]], [oys[u], oys[v]],
                color=c, lw=lw, linestyle=ls, alpha=0.85, zorder=zo)

    for i in range(graph.num_nodes):
        c = '#ffdd00' if i in tip_nodes else ('#4da6ff' if is_ep[i] else '#ffffff')
        s = 80 if i in tip_nodes else 22
        ax.scatter(oxs[i], oys[i], c=c, s=s, zorder=6,
                   edgecolors='black', linewidths=0.5)

    ax.set_title('Panel 3 — Frontier Prediction\ngreen=TP  orange=FN  red=FP',
                 color='white', fontsize=9)
    leg3 = [
        Line2D([0],[0], color='#00ee55', lw=2.5,               label='TP — correctly predicted reconnection'),
        Line2D([0],[0], color='#ff8800', lw=1.5, linestyle='--', label='FN — missed reconnection'),
        Line2D([0],[0], color='#ff3333', lw=1.2, linestyle='--', label='FP — false reconnection'),
        Line2D([0],[0], color='royalblue', lw=1.0,              label='Visible edges'),
    ]
    ax.legend(handles=leg3, loc='lower right', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     default='outputs/linkpred_200ep/gine/best_model.pt')
    p.add_argument('--arch',     default='gine')
    p.add_argument('--graphs',   default='outputs/clean_graphs/graphs/test_graphs.pt')
    p.add_argument('--mask-dir', default=None)
    p.add_argument('--n',        type=int, default=10)
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--mask-frac', type=float, default=0.40)
    p.add_argument('--output',   default='outputs/frontier_figures')
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    encoder, edge_pred, best_ep = load_checkpoint(args.ckpt, arch=args.arch)
    print(f'{args.arch.upper()} loaded — best epoch {best_ep}')

    graphs = torch.load(args.graphs, map_location='cpu', weights_only=False)
    valid  = [g for g in graphs if is_valid_for_frontier_task(g, min_frontier_edges=2)]
    print(f'{len(valid)} / {len(graphs)} graphs valid for frontier task')

    random.shuffle(valid)
    done = 0

    for g in valid:
        if done >= args.n:
            break

        split = frontier_edge_split(g, mask_frac=args.mask_frac, seed=args.seed + done)
        if split is None:
            continue

        td  = split['train_data']
        lei = split['label_ei']
        lbl = split['labels']

        if len(np.unique(lbl.numpy())) < 2:
            continue

        with torch.no_grad():
            x  = td.x.float()
            ei = td.edge_index
            ea = td.edge_attr.float() if td.edge_attr is not None else None
            z  = encoder(x, ei, ea)
            scores = torch.sigmoid(edge_pred(z, lei)).squeeze(-1)

        fname = getattr(g, 'filename', f'graph_{done}.jpg')
        mask, skel = load_background(fname, args.mask_dir)

        stem      = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output, stem + '_frontier.png')

        draw_frontier_figure(g, split, scores.numpy(), mask, skel,
                             save_path, title=fname)
        print(f'  [{done+1}/{args.n}] {fname}')
        done += 1

    print(f'\nDone — {done} frontier figures saved to {args.output}/')


if __name__ == '__main__':
    main()
