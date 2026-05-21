"""
Feature extraction for crack graph nodes and edges.

Node features [N, 6]:
    0  x_norm        — x coordinate / image width  (0–1)
    1  y_norm        — y coordinate / image height (0–1)
    2  thickness     — crack width at node in pixels (distance-transform-based)
    3  degree        — number of connected edges
    4  is_endpoint   — 1.0 if degree == 1 (crack tip), else 0.0
    5  is_junction   — 1.0 if degree >= 3 (branching point), else 0.0

Edge features [E, 7]:
    0  path_length     — skeleton pixel count along the crack segment
    1  euclidean_dist  — straight-line distance between the two nodes (pixels)
    2  tortuosity      — path_length / euclidean_dist  (1.0 = straight, >1 = curved)
    3  angle_sym       — crack orientation in [0°, 180°), symmetric for undirected edges
    4  avg_thickness   — mean crack width along the segment
    5  min_thickness   — thinnest point along the segment
    6  max_thickness   — widest point along the segment
"""

import numpy as np
import cv2
import torch


# ── Distance Transform ────────────────────────────────────────────────────────

def compute_distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    """
    Full crack width (diameter) at every pixel via Euclidean distance transform.
    binary_mask: uint8 array with 1=crack, 0=background.
    Returns array of same shape; value at crack pixel = local crack width in pixels.
    """
    radius_map = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
    return radius_map * 2.0  # radius → diameter


def sample_thickness(dist_map: np.ndarray, y: float, x: float) -> float:
    h, w = dist_map.shape
    yi = int(np.clip(round(y), 0, h - 1))
    xi = int(np.clip(round(x), 0, w - 1))
    return float(dist_map[yi, xi])


# ── Node Features ─────────────────────────────────────────────────────────────

def compute_node_features(
    graph,
    dist_map: np.ndarray,
    H: int,
    W: int,
) -> tuple:
    """
    Returns:
        x_tensor   — FloatTensor [N, 6]
        node_index — dict: node_id → row index
    """
    node_ids = list(graph.nodes())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    feats = []
    for nid in node_ids:
        y, x = graph.nodes[nid]['o']  # sknw stores (y, x)
        deg = float(graph.degree(nid))
        thickness = sample_thickness(dist_map, y, x)
        feats.append([
            float(x) / W,       # x_norm
            float(y) / H,       # y_norm
            thickness,          # crack width
            deg,                # connectivity
            float(deg == 1.0),  # is_endpoint
            float(deg >= 3.0),  # is_junction
        ])
    return torch.tensor(feats, dtype=torch.float), node_index


# ── Edge Features ─────────────────────────────────────────────────────────────

def compute_edge_features(
    graph,
    dist_map: np.ndarray,
    node_index: dict,
) -> tuple:
    """
    Adds computed features back onto each edge in graph.edges(data=True) for
    later use by the visualiser.

    Returns:
        edge_index — LongTensor [2, 2E]  (both directions for undirected PyG)
        edge_attr  — FloatTensor [2E, 7]
    """
    src, dst, feats = [], [], []

    for u, v, edata in graph.edges(data=True):
        pts = edata['pts']  # array of (y, x) skeleton pixel coords along edge

        # Cast to Python float before arithmetic to avoid numpy integer overflow
        y_u = float(graph.nodes[u]['o'][0])
        x_u = float(graph.nodes[u]['o'][1])
        y_v = float(graph.nodes[v]['o'][0])
        x_v = float(graph.nodes[v]['o'][1])
        dy = y_v - y_u
        dx = x_v - x_u
        euclidean_dist = float(np.sqrt(dx ** 2 + dy ** 2))

        # Geometric path length: node_u → path pixels → node_v
        # This ensures tortuosity >= 1 (triangle inequality holds)
        waypoints = (
            [(y_u, x_u)]
            + [(float(p[0]), float(p[1])) for p in pts]
            + [(y_v, x_v)]
        )
        path_length = float(sum(
            np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 +
                    (waypoints[i+1][1] - waypoints[i][1])**2)
            for i in range(len(waypoints) - 1)
        ))
        # Clamp euclidean_dist to ≥ 1 px before dividing.
        # Degenerate sknw loop edges can have euclidean_dist ≈ 0 (both nodes
        # at same pixel), which would blow tortuosity to infinity.
        tortuosity = path_length / max(euclidean_dist, 1.0)

        # Symmetric orientation [0°, 180°) — same for (u→v) and (v→u)
        angle_sym = float(np.degrees(np.arctan2(dy, dx)) % 180.0)

        if len(pts) > 0:
            thick = [sample_thickness(dist_map, py, px) for py, px in pts]
            avg_t = float(np.mean(thick))
            min_t = float(np.min(thick))
            max_t = float(np.max(thick))
        else:
            avg_t = min_t = max_t = 0.0

        # Store on the nx edge for the visualiser to read
        edata['path_length'] = path_length
        edata['euclidean_dist'] = euclidean_dist
        edata['tortuosity'] = tortuosity
        edata['angle_sym'] = angle_sym
        edata['avg_thickness'] = avg_t
        edata['min_thickness'] = min_t
        edata['max_thickness'] = max_t

        feat = [path_length, euclidean_dist, tortuosity, angle_sym, avg_t, min_t, max_t]
        i, j = node_index[u], node_index[v]
        src += [i, j]
        dst += [j, i]
        feats += [feat, feat]

    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(feats, dtype=torch.float)

    return edge_index, edge_attr
