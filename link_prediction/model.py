"""
Phase 3 model components.

CrackGATEncoder   — 3-layer GAT with edge features
MLPEdgePredictor  — scores node pairs for link prediction
MLPNodePredictor  — classifies nodes as having a hidden neighbour (node prediction task)

Angle encoding: the raw angle_sym feature (col 3 of edge_attr, range [0°,180°)) is
replaced inside the encoder by [sin(2θ), cos(2θ)] so opposite directions map
smoothly without the 0°/180° discontinuity.  This expands effective edge_dim from
7 → 8 without changing the saved .pt files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ── Angle encoding helper ─────────────────────────────────────────────────────

def encode_angle(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Replace angle_sym column (index 3, degrees in [0,180)) with
    [sin(2θ), cos(2θ)] — smooth circular encoding for undirected edges.
    Output dim = edge_attr.size(1) + 1  (7 → 8).
    """
    angle_deg = edge_attr[:, 3]
    angle_rad = angle_deg * (torch.pi / 180.0)
    sin_2t = torch.sin(2.0 * angle_rad).unsqueeze(1)
    cos_2t = torch.cos(2.0 * angle_rad).unsqueeze(1)
    return torch.cat([
        edge_attr[:, :3],   # path_length, euclidean_dist, tortuosity
        sin_2t, cos_2t,     # circular angle encoding
        edge_attr[:, 4:],   # avg_thickness, min_thickness, max_thickness
    ], dim=1)               # → [E, 8]


# ── Encoder ───────────────────────────────────────────────────────────────────

class CrackGATEncoder(nn.Module):
    """
    3-layer GAT encoder with:
      - edge features (8-dim after angle encoding)
      - multi-head attention
      - residual connections between layers 2 and 3
      - dropout for regularisation

    Node features in  [N, 6]: x_norm, y_norm, thickness, degree, is_endpoint, is_junction
    Node embeddings out [N, out_dim]
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden: int = 64,
        out_dim: int = 32,
        heads: int = 4,
        edge_dim: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: in_channels → hidden  (multi-head, concat → hidden*heads)
        self.conv1 = GATConv(in_channels, hidden, heads=heads,
                             concat=True, edge_dim=edge_dim, dropout=dropout)

        # Layer 2: hidden*heads → hidden  (multi-head, concat → hidden*heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads,
                             concat=True, edge_dim=edge_dim, dropout=dropout)

        # Layer 3: hidden*heads → out_dim  (single head, no concat)
        self.conv3 = GATConv(hidden * heads, out_dim, heads=1,
                             concat=False, edge_dim=edge_dim, dropout=dropout)

        # Projection for residual between layer-2 input and layer-3 output
        self.res_proj = nn.Linear(hidden * heads, out_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Encode angle column before feeding into attention layers
        if edge_attr is not None:
            edge_attr = encode_angle(edge_attr)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x2 = F.elu(x)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Layer 3 + residual
        x3 = self.conv3(x2, edge_index, edge_attr=edge_attr)
        x3 = x3 + self.res_proj(x2)   # residual connection
        x3 = self.norm3(x3)

        return x3   # [N, out_dim]


# ── Edge predictor ────────────────────────────────────────────────────────────

class MLPEdgePredictor(nn.Module):
    """
    Scores a candidate edge (u, v) using a 3-layer MLP on the concatenation of
    [z_u, z_v, z_u * z_v].  The Hadamard product captures interaction between
    the two endpoint embeddings, giving the MLP more signal than a pure dot product.
    """

    def __init__(self, in_dim: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        pair = torch.cat([src, dst, src * dst], dim=-1)  # [E, 3*in_dim]
        return self.mlp(pair).squeeze(-1)                # [E]


# ── Node predictor ────────────────────────────────────────────────────────────

class MLPNodePredictor(nn.Module):
    """
    Binary classifier per node: 1 = this node had a neighbour that was hidden
    (crack tip removed from the visible graph), 0 = no hidden neighbours.

    This is the "missing node" head — it proves the model understands which
    nodes are topologically incomplete even without seeing the missing node.
    """

    def __init__(self, in_dim: int = 64, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)   # [N]
