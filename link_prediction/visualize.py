"""
Visualisation for Phase 3 predictions.

Produces a 2-panel figure per graph:
  Left  — Edge prediction: visible edges (blue), predicted missing edges (red dashed),
           missed edges (grey dashed)
  Right — Node prediction: nodes coloured by whether the model correctly identified
           them as having a hidden neighbour
"""

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


def visualize_predictions(
    graph,
    encoder,
    edge_pred,
    node_pred,
    device,
    save_path: str = None,
    edge_threshold: float = 0.5,
    node_threshold: float = 0.5,
    node_mask_frac: float = 0.20,
    dpi: int = 130,
):
    """
    Generate a 2-panel figure showing edge and node predictions for one graph.
    """
    encoder.eval(); edge_pred.eval(); node_pred.eval()

    # ── Panel 1: Edge prediction ──────────────────────────────────────────────
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fname = getattr(graph, 'filename', 'graph')
    fig.suptitle(f'{fname}  |  Phase 3: Topology Understanding', color='white', fontsize=11)

    # ── Panel 1: Node prediction (primary task — left panel) ─────────────────
    masked_d, node_labels, hidden_mask, eval_mask = apply_node_mask(
        graph, mask_frac=node_mask_frac, seed=42
    )

    with torch.no_grad():
        x   = masked_d.x.float().to(device)
        ei  = masked_d.edge_index.to(device)
        ea  = masked_d.edge_attr.float().to(device) if masked_d.edge_attr is not None else None
        z        = encoder(x, ei, ea)
        n_probs  = torch.sigmoid(node_pred(z)).cpu().numpy()

    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    ax.set_title('Task 1 — Missing Crack Tip Detection (primary)\n'
                 'green=TP  red=FP  grey=TN  orange=FN  ✕=removed tip',
                 color='white', fontsize=9)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # Draw remaining (masked) edges
    mei = masked_d.edge_index.numpy()
    for i in range(0, mei.shape[1], 2):
        u, v = mei[0, i], mei[1, i]
        xs = [pos[u][0], pos[v][0]]; ys = [pos[u][1], pos[v][1]]
        ax.plot(xs, ys, color='royalblue', linewidth=1.5, alpha=0.6)

    # Draw nodes with prediction colouring
    node_labels_np = node_labels.numpy()
    for i, (px, py) in pos.items():
        if hidden_mask[i]:
            ax.scatter(px, py, c='yellow', s=80, marker='x', zorder=6, linewidths=2)
            continue
        if not eval_mask[i]:
            ax.scatter(px, py, c='#555555', s=20, zorder=5)
            continue
        true_lbl = node_labels_np[i]
        pred_lbl = float(n_probs[i] >= node_threshold)
        if   true_lbl == 1 and pred_lbl == 1: color = '#44ff44'
        elif true_lbl == 0 and pred_lbl == 1: color = '#ff4444'
        elif true_lbl == 1 and pred_lbl == 0: color = '#ff8800'
        else:                                  color = '#aaaaaa'
        ax.scatter(px, py, c=color, s=50, zorder=5, edgecolors='black', linewidths=0.4)

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
    ax.legend(handles=legend_nodes, loc='lower right', facecolor='#2a2a2a',
              edgecolor='gray', labelcolor='white', fontsize=6.5, framealpha=0.8)

    # ── Panel 2: Edge prediction (secondary task — right panel) ──────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    ax2.set_title('Task 2 — Missing Crack Segment Recovery (secondary)\nblue=visible  red=predicted missing  grey=missed',
                  color='white', fontsize=9)
    ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
    ax2.axis('off')

    # Visible edges
    vis_ei = train_d.edge_index.numpy()
    for i in range(0, vis_ei.shape[1], 2):
        u, v = vis_ei[0, i], vis_ei[1, i]
        xs = [pos[u][0], pos[v][0]]; ys = [pos[u][1], pos[v][1]]
        ax2.plot(xs, ys, color='royalblue', linewidth=1.5, alpha=0.7)

    # Hidden true edges — predicted or missed
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
        ax2.plot(xs, ys, color=color, linewidth=lw, linestyle='--', alpha=0.9)

    # Nodes
    for i, (px, py) in pos.items():
        ax2.scatter(px, py, c='white', s=25, zorder=5, edgecolors='black', linewidths=0.5)

    legend_edges = [
        Line2D([0], [0], color='royalblue', lw=2, label='Visible crack segments'),
        Line2D([0], [0], color='red',       lw=2, linestyle='--', label='Correctly predicted missing segments'),
        Line2D([0], [0], color='#666666',   lw=2, linestyle='--', label='Missed segments (false negative)'),
    ]
    ax2.legend(handles=legend_edges, loc='lower right', facecolor='#2a2a2a',
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
