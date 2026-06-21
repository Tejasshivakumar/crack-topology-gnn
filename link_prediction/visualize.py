"""
Visualisation for Phase 3 predictions.

Produces a figure per graph:
  Panel 0 (if mask_dir given) — Reference: crack mask + skeleton overlay
  Panel 1 — Node prediction: nodes coloured by TP/FP/TN/FN for missing-tip detection
  Panel 2 — Edge prediction: visible/predicted-missing/missed edges

When mask_dir is provided the skeleton is also drawn as a faint background
in the prediction panels so the graph topology can be judged in spatial context.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .masking import make_edge_splitter, apply_node_mask


_EDGE_SPLITTER = make_edge_splitter(num_val=0.0, num_test=0.20)


def _node_positions(data):
    """Extract (x, y) positions from normalised node features."""
    x_arr = data.x[:, 0].numpy()   # x_norm
    y_arr = data.x[:, 1].numpy()   # y_norm  (flip for image coords)
    return {i: (x_arr[i], 1.0 - y_arr[i]) for i in range(data.x.size(0))}


def _load_background(filename, mask_dir):
    """
    Load binary mask and compute skeleton for the given graph filename.
    Returns (mask_uint8 [H,W], skeleton_uint8 [H,W]) or (None, None).
    """
    if mask_dir is None:
        return None, None
    mask_path = os.path.join(mask_dir, filename)
    if not os.path.exists(mask_path):
        return None, None
    try:
        import cv2
        from skimage.morphology import skeletonize
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            return None, None
        _, binary = cv2.threshold(mask_gray, 127, 1, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary).astype(np.uint8)
        return binary, skeleton
    except Exception:
        return None, None


def _draw_reference_panel(ax, mask, skeleton):
    """Draw the mask + skeleton reference panel."""
    ax.set_facecolor('#1a1a1a')
    ax.set_title('Reference — Crack Mask & Skeleton\n'
                 'grey=crack region  cyan=skeleton centreline',
                 color='white', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[mask > 0] = [0.30, 0.30, 0.30]   # crack region: dark grey
    rgb[skeleton > 0] = [0.0, 0.85, 0.85]  # skeleton: cyan
    # extent + origin='upper': row 0 → y=1 (top), matching _node_positions flip
    ax.imshow(rgb, extent=[0, 1, 0, 1], origin='upper', aspect='auto', zorder=2)


def _draw_skeleton_bg(ax, mask, skeleton):
    """
    Draw crack mask + skeleton as a background on a prediction panel.
    Mask region shows as dim grey; skeleton centreline shows as brighter cyan.
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    # Crack mask region: dim grey
    crack_px = mask > 0
    rgba[crack_px, :3] = 0.45
    rgba[crack_px, 3]  = 0.35
    # Skeleton centreline on top: brighter cyan
    skel_px = skeleton > 0
    rgba[skel_px, 0] = 0.0
    rgba[skel_px, 1] = 0.85
    rgba[skel_px, 2] = 0.85
    rgba[skel_px, 3] = 0.65
    ax.imshow(rgba, extent=[0, 1, 0, 1], origin='upper', aspect='auto', zorder=1)


def visualize_predictions(
    graph,
    encoder,
    edge_pred,
    node_pred,
    device,
    save_path: str = None,
    edge_threshold: float = 0.5,
    node_threshold: float = 0.3,
    node_mask_frac: float = 0.20,
    dpi: int = 130,
    mask_dir: str = None,
):
    """
    Generate a figure showing edge and node predictions for one graph.

    When mask_dir is given a 3-panel layout is used:
        [Reference: mask+skeleton | Node prediction | Edge prediction]
    and the skeleton is drawn as a faint background in both prediction panels.
    Without mask_dir the original 2-panel layout is used.
    """
    encoder.eval(); edge_pred.eval(); node_pred.eval()

    fname = getattr(graph, 'filename', 'graph')

    # ── Optionally load mask/skeleton background ──────────────────────────────
    mask, skeleton = _load_background(fname, mask_dir)
    has_bg = mask is not None

    # ── Edge prediction data ──────────────────────────────────────────────────
    try:
        train_d, _, test_d = _EDGE_SPLITTER(graph)
    except Exception:
        return

    pos = _node_positions(graph)

    with torch.no_grad():
        x   = test_d.x.to(device)
        ei  = train_d.edge_index.to(device)
        ea  = train_d.edge_attr.to(device) if train_d.edge_attr is not None else None
        eli = test_d.edge_label_index.to(device)
        z        = encoder(x, ei, ea)
        e_probs  = torch.sigmoid(edge_pred(z, eli)).cpu().numpy()
        e_labels = test_d.edge_label.numpy()
        e_edges  = test_d.edge_label_index.numpy()

    # ── Node prediction data ──────────────────────────────────────────────────
    masked_d, node_labels, hidden_mask, eval_mask = apply_node_mask(
        graph, mask_frac=node_mask_frac, seed=42
    )

    with torch.no_grad():
        x   = masked_d.x.float().to(device)
        ei  = masked_d.edge_index.to(device)
        ea  = masked_d.edge_attr.float().to(device) if masked_d.edge_attr is not None else None
        z        = encoder(x, ei, ea)
        n_probs  = torch.sigmoid(node_pred(z)).cpu().numpy()

    # ── Figure layout ─────────────────────────────────────────────────────────
    ncols   = 3 if has_bg else 2
    figw    = 21 if has_bg else 14
    fig, axes = plt.subplots(1, ncols, figsize=(figw, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f'{fname}  |  Phase 3: Topology Understanding', color='white', fontsize=11)

    if has_bg:
        ax_ref, ax_node, ax_edge = axes
        _draw_reference_panel(ax_ref, mask, skeleton)
    else:
        ax_node, ax_edge = axes

    # ── Node prediction panel ─────────────────────────────────────────────────
    ax_node.set_facecolor('#1a1a1a')
    ax_node.set_title('Task 1 — Missing Crack Tip Detection (primary)\n'
                      'green=TP  red=FP  grey=TN  orange=FN  ✕=removed tip',
                      color='white', fontsize=9)
    ax_node.set_xlim(-0.05, 1.05); ax_node.set_ylim(-0.05, 1.05)
    ax_node.axis('off')

    if has_bg:
        _draw_skeleton_bg(ax_node, mask, skeleton)

    mei = masked_d.edge_index.numpy()
    for i in range(0, mei.shape[1], 2):
        u, v = mei[0, i], mei[1, i]
        xs = [pos[u][0], pos[v][0]]; ys = [pos[u][1], pos[v][1]]
        ax_node.plot(xs, ys, color='royalblue', linewidth=1.5, alpha=0.6, zorder=3)

    node_labels_np = node_labels.numpy()
    for i, (px, py) in pos.items():
        if hidden_mask[i]:
            ax_node.scatter(px, py, c='yellow', s=80, marker='x', zorder=6, linewidths=2)
            continue
        if not eval_mask[i]:
            ax_node.scatter(px, py, c='#555555', s=20, zorder=5)
            continue
        true_lbl = node_labels_np[i]
        pred_lbl = float(n_probs[i] >= node_threshold)
        if   true_lbl == 1 and pred_lbl == 1: color = '#44ff44'
        elif true_lbl == 0 and pred_lbl == 1: color = '#ff4444'
        elif true_lbl == 1 and pred_lbl == 0: color = '#ff8800'
        else:                                  color = '#aaaaaa'
        ax_node.scatter(px, py, c=color, s=50, zorder=5, edgecolors='black', linewidths=0.4)

    legend_nodes = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#44ff44', markersize=8,
               markeredgecolor='black', label='TP: correctly identified missing neighbour'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#ff4444', markersize=8,
               markeredgecolor='black', label='FP: wrongly flagged'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#aaaaaa', markersize=8,
               markeredgecolor='black', label='TN: correctly not flagged'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#ff8800', markersize=8,
               markeredgecolor='black', label='FN: missed hidden neighbour'),
        Line2D([0], [0], marker='x', color='yellow', markersize=10, lw=2, label='Removed crack tip (hidden)'),
    ]
    ax_node.legend(handles=legend_nodes, loc='lower right', facecolor='#2a2a2a',
                   edgecolor='gray', labelcolor='white', fontsize=6.5, framealpha=0.8)

    # ── Edge prediction panel ─────────────────────────────────────────────────
    ax_edge.set_facecolor('#1a1a1a')
    ax_edge.set_title('Task 2 — Missing Crack Segment Recovery (secondary)\n'
                      'blue=visible  red=predicted missing  grey=missed',
                      color='white', fontsize=9)
    ax_edge.set_xlim(-0.05, 1.05); ax_edge.set_ylim(-0.05, 1.05)
    ax_edge.axis('off')

    if has_bg:
        _draw_skeleton_bg(ax_edge, mask, skeleton)

    vis_ei = train_d.edge_index.numpy()
    for i in range(0, vis_ei.shape[1], 2):
        u, v = vis_ei[0, i], vis_ei[1, i]
        xs = [pos[u][0], pos[v][0]]; ys = [pos[u][1], pos[v][1]]
        ax_edge.plot(xs, ys, color='royalblue', linewidth=1.5, alpha=0.7, zorder=3)

    for i in range(len(e_probs)):
        if e_labels[i] != 1:
            continue
        u, v = e_edges[0, i], e_edges[1, i]
        if u not in pos or v not in pos:
            continue
        xs = [pos[u][0], pos[v][0]]; ys = [pos[u][1], pos[v][1]]
        predicted = e_probs[i] >= edge_threshold
        color = 'red' if predicted else '#666666'
        lw    = 2.5  if predicted else 1.5
        ax_edge.plot(xs, ys, color=color, linewidth=lw, linestyle='--', alpha=0.9, zorder=4)

    for i, (px, py) in pos.items():
        ax_edge.scatter(px, py, c='white', s=25, zorder=5, edgecolors='black', linewidths=0.5)

    legend_edges = [
        Line2D([0], [0], color='royalblue', lw=2, label='Visible crack segments'),
        Line2D([0], [0], color='red',       lw=2, linestyle='--', label='Correctly predicted missing segments'),
        Line2D([0], [0], color='#666666',   lw=2, linestyle='--', label='Missed segments (false negative)'),
    ]
    ax_edge.legend(handles=legend_edges, loc='lower right', facecolor='#2a2a2a',
                   edgecolor='gray', labelcolor='white', fontsize=7, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close(fig)


def plot_training_curves(history: list, save_path: str = None, dpi: int = 120):
    """Plot losses and validation AUCs over training epochs."""
    epochs        = [h['epoch']     for h in history]
    edge_loss     = [h['edge_loss'] for h in history]
    node_loss     = [h['node_loss'] for h in history]
    node_val_auc  = [h.get('node_val_auc', h.get('val_auc', 0.5)) for h in history]
    edge_val_auc  = [h.get('edge_val_auc', h.get('val_auc', 0.5)) for h in history]
    val_score     = [h.get('val_score',    h.get('val_auc', 0.5)) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a1a')

    ax1.set_facecolor('#1a1a1a')
    ax1.plot(epochs, node_loss, color='orange',    label='Node loss (primary task)')
    ax1.plot(epochs, edge_loss, color='royalblue', label='Edge loss (secondary task)')
    ax1.set_xlabel('Epoch', color='white'); ax1.set_ylabel('Loss', color='white')
    ax1.set_title('Training Loss', color='white')
    ax1.tick_params(colors='white'); ax1.legend(labelcolor='white', facecolor='#2a2a2a')
    for spine in ax1.spines.values(): spine.set_edgecolor('#444')

    ax2.set_facecolor('#1a1a1a')
    ax2.plot(epochs, node_val_auc, color='orange',    label='Node val AUC (primary)')
    ax2.plot(epochs, edge_val_auc, color='royalblue', label='Edge val AUC (secondary)')
    ax2.plot(epochs, val_score,    color='lime',      label='Combined score (best model)')
    ax2.axhline(0.5, color='#666', linestyle='--', linewidth=1, label='Random baseline')
    ax2.set_xlabel('Epoch', color='white'); ax2.set_ylabel('AUC-ROC', color='white')
    ax2.set_title('Validation AUC', color='white')
    ax2.tick_params(colors='white'); ax2.legend(labelcolor='white', facecolor='#2a2a2a', fontsize=7)
    ax2.set_ylim(0.4, 1.0)
    for spine in ax2.spines.values(): spine.set_edgecolor('#444')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close(fig)
