"""
Graph visualisation utilities.

Produces a 3- or 4-panel figure per image:
  Panel 1 (optional): original RGB image
  Panel 2: binary mask
  Panel 3: skeleton (1-px centerline)
  Panel 4: graph overlay on mask
             — edges coloured by avg_thickness (YlOrRd)
             — nodes coloured by type: endpoint=blue, chain=white, junction=red
             — node size scales with degree
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


_NODE_COLORS = {
    'endpoint': '#4da6ff',   # blue  — degree == 1
    'chain':    '#ffffff',   # white — degree == 2
    'junction': '#ff4d4d',   # red   — degree >= 3
}


def _node_color(deg: int) -> str:
    if deg == 1:
        return _NODE_COLORS['endpoint']
    if deg == 2:
        return _NODE_COLORS['chain']
    return _NODE_COLORS['junction']


def _node_size(deg: int) -> float:
    return 30.0 + deg * 10.0


def visualize_graph(
    binary_mask: np.ndarray,
    skeleton: np.ndarray,
    nx_graph,
    pyg_data,
    original_image: np.ndarray = None,
    save_path: str = None,
    dpi: int = 130,
) -> None:
    """
    Save a multi-panel visualization of the crack graph.

    Args:
        binary_mask:    uint8 array (0/1) — crack mask
        skeleton:       uint8 array (0/1) — skeletonized mask
        nx_graph:       networkx Graph with per-edge 'avg_thickness' attribute
        pyg_data:       PyG Data object (used for metadata in title)
        original_image: optional RGB uint8 array
        save_path:      if given, save PNG there; otherwise plt.show()
        dpi:            output resolution
    """
    n_panels = 4 if original_image is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.patch.set_facecolor('#1a1a1a')

    fname = getattr(pyg_data, 'filename', '')
    n_nodes = pyg_data.x.size(0)
    n_edges = pyg_data.edge_index.size(1) // 2
    density = getattr(pyg_data, 'crack_density', 0.0)
    fig.suptitle(
        f'{fname}  |  nodes={n_nodes}  edges={n_edges}  crack_density={density:.3f}',
        color='white', fontsize=10, y=1.01,
    )

    panel = 0

    # ── Panel 1 (optional): original image ───────────────────────────────
    if original_image is not None:
        axes[panel].imshow(original_image)
        axes[panel].set_title('Original Image', color='white', fontsize=9)
        axes[panel].axis('off')
        panel += 1

    # ── Panel 2: binary mask ─────────────────────────────────────────────
    axes[panel].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[panel].set_title('Binary Mask', color='white', fontsize=9)
    axes[panel].axis('off')
    panel += 1

    # ── Panel 3: skeleton ────────────────────────────────────────────────
    axes[panel].imshow(skeleton, cmap='gray', vmin=0, vmax=1)
    axes[panel].set_title('Skeleton', color='white', fontsize=9)
    axes[panel].axis('off')
    panel += 1

    # ── Panel 4: graph overlay ───────────────────────────────────────────
    ax = axes[panel]
    ax.imshow(binary_mask, cmap='gray', vmin=0, vmax=1, alpha=0.45)
    ax.set_facecolor('#1a1a1a')
    ax.set_title('Graph Overlay\n(edges=avg thickness | nodes=endpoint/chain/junction)',
                 color='white', fontsize=9)
    ax.axis('off')

    # Collect edge segments and their avg_thickness for colouring
    segments, thicknesses = [], []
    for u, v, edata in nx_graph.edges(data=True):
        pts = edata.get('pts', np.empty((0, 2)))
        if len(pts) < 2:
            continue
        # pts are (y, x); matplotlib wants (x, y)
        seg = [(p[1], p[0]) for p in pts]
        segments.append(seg)
        thicknesses.append(edata.get('avg_thickness', 0.0))

    if segments:
        thick_arr = np.array(thicknesses, dtype=np.float32)
        vmin_t, vmax_t = thick_arr.min(), max(thick_arr.max(), 1e-6)
        norm = Normalize(vmin=vmin_t, vmax=vmax_t)
        cmap = cm.get_cmap('YlOrRd')
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.9)
        lc.set_array(thick_arr)
        ax.add_collection(lc)

        cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('avg thickness (px)', color='white', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    # Draw nodes
    node_ids = list(nx_graph.nodes())
    for nid in node_ids:
        y, x = nx_graph.nodes[nid]['o']
        deg = nx_graph.degree(nid)
        ax.scatter(x, y,
                   c=_node_color(deg),
                   s=_node_size(deg),
                   zorder=5,
                   edgecolors='black',
                   linewidths=0.4)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='endpoint (deg=1)',
               markerfacecolor='#4da6ff', markersize=7, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='none', label='chain (deg=2)',
               markerfacecolor='white', markersize=7, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='none', label='junction (deg≥3)',
               markerfacecolor='#ff4d4d', markersize=7, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              facecolor='#2a2a2a', edgecolor='gray',
              labelcolor='white', fontsize=7, framealpha=0.8)

    for a in axes:
        a.set_facecolor('#1a1a1a')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    else:
        plt.show()

    plt.close(fig)
