import torch
from torch import nn


class UXPredictionHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.ux_score_head = nn.Linear(in_dim, 1)
        self.layout_quality_head = nn.Linear(in_dim, 1)
        self.attention_alignment_head = nn.Linear(in_dim, 1)

    def forward(self, fused_cls: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "ux_score": self.ux_score_head(fused_cls),
            "layout_quality": self.layout_quality_head(fused_cls),
            "attention_alignment": self.attention_alignment_head(fused_cls),
        }
