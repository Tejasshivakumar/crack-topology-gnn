"""
Phase 3 model components.

Encoders (all share the same MLPEdgePredictor / MLPNodePredictor heads):
  MLPEncoder      — no-graph baseline: pure MLP on node features
  GCNEncoder      — 3-layer GCN, no edge features
  SAGEEncoder     — 3-layer GraphSAGE, no edge features
  GINEEncoder     — 3-layer GIN with edge features
  CrackGATEncoder — 3-layer GAT with edge features  ← original model

Predictors:
  MLPEdgePredictor  — scores node pairs for link prediction
  MLPNodePredictor  — classifies nodes as having a hidden neighbour

Angle encoding: the raw angle_sym feature (col 3 of edge_attr, range [0°,180°)) is
replaced inside the encoder by [sin(2θ), cos(2θ)] so opposite directions map
smoothly without the 0°/180° discontinuity.  This expands effective edge_dim from
7 → 8 without changing the saved .pt files.  Used by GINEEncoder and CrackGATEncoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINEConv


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


# ── Additional encoders for model comparison ──────────────────────────────────

class MLPEncoder(nn.Module):
    """
    No-graph baseline: pure MLP on node features — ignores graph structure entirely.
    Establishes the lower bound; any GNN should beat this by using edge information.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 128,
                 out_dim: int = 64, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.net(x)


class GCNEncoder(nn.Module):
    """
    3-layer GCN (Kipf & Welling 2017) — uses graph structure but no edge features.
    Isolates the contribution of edge features vs. bare topology.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 128,
                 out_dim: int = 64, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, out_dim)
        self.res_proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x2 = F.relu(x)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = x3 + self.res_proj(x2)
        x3 = self.norm3(x3)
        return x3


class SAGEEncoder(nn.Module):
    """
    3-layer GraphSAGE (Hamilton et al. 2017) — inductive mean aggregation,
    no edge features.  Compared with GCN to test aggregation strategy.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 128,
                 out_dim: int = 64, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_dim)
        self.res_proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x2 = F.relu(x)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = x3 + self.res_proj(x2)
        x3 = self.norm3(x3)
        return x3


class GINEEncoder(nn.Module):
    """
    3-layer GINE (Hu et al. 2020) — GIN extended with edge features via linear
    projection.  Theoretically as expressive as the Weisfeiler-Leman test while
    also leveraging geometric edge attributes.
    """

    def __init__(self, in_channels: int = 6, hidden: int = 128,
                 out_dim: int = 64, edge_dim: int = 8, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.dropout = dropout

        def _mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.LayerNorm(out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c),
            )

        self.conv1 = GINEConv(_mlp(in_channels, hidden), edge_dim=edge_dim)
        self.conv2 = GINEConv(_mlp(hidden, hidden),      edge_dim=edge_dim)
        self.conv3 = GINEConv(_mlp(hidden, out_dim),     edge_dim=edge_dim)
        self.res_proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            edge_attr = encode_angle(edge_attr)   # 7 → 8

        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x2 = F.relu(x)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = x3 + self.res_proj(x2)
        x3 = self.norm3(x3)
        return x3


# ── Factory ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'mlp':  MLPEncoder,
    'gcn':  GCNEncoder,
    'sage': SAGEEncoder,
    'gine': GINEEncoder,
    'gat':  CrackGATEncoder,
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def build_encoder(name: str, in_channels: int = 6, hidden: int = 128,
                  out_dim: int = 64, heads: int = 4, dropout: float = 0.3) -> nn.Module:
    """Instantiate an encoder by name.  GAT uses heads; others ignore it."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")
    if name == 'gat':
        return cls(in_channels, hidden, out_dim, heads=heads, dropout=dropout)
    return cls(in_channels, hidden, out_dim, dropout=dropout)


def count_parameters(encoder, edge_pred, node_pred) -> int:
    """Total trainable parameters across all three components."""
    return sum(
        p.numel() for p in
        list(encoder.parameters()) + list(edge_pred.parameters()) + list(node_pred.parameters())
        if p.requires_grad
    )
