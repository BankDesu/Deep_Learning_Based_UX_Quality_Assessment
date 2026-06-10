import torch
from torch import nn


class SaliencyPredictor(nn.Module):
    """Saliency heatmap predictor scaffold."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, visual_feature_map: torch.Tensor) -> torch.Tensor:
        return self.head(visual_feature_map)
