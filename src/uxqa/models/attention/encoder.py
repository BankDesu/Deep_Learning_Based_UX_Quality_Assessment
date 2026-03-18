import torch
from torch import nn


class AttentionMapEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, out_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, attention_map: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(attention_map)
        return feat.flatten(1)
