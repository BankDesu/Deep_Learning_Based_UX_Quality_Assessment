import torch
from torch import nn


class SimpleGNNEncoder(nn.Module):
    """
    Lightweight graph encoder using repeated normalized adjacency propagation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU(),
            )
            for i in range(layers)
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.numel() == 0:
            return node_features

        deg = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_adj = adjacency / deg
        h = node_features
        for layer in self.layers:
            h = norm_adj @ h
            h = layer(h)
        return h
