from dataclasses import dataclass

import torch
from torch import nn

from .encoder import AttentionMapEncoder
from .saliency import SaliencyPredictor


@dataclass(slots=True)
class AttentionBranchOutput:
    heatmap: torch.Tensor
    attention_embedding: torch.Tensor
    attention_token: torch.Tensor


class AttentionBranch(nn.Module):
    def __init__(self, visual_dim: int, out_dim: int) -> None:
        super().__init__()
        self.saliency_predictor = SaliencyPredictor(visual_dim)
        self.attention_encoder = AttentionMapEncoder(out_dim)
        self.token_proj = nn.Linear(out_dim, out_dim)

    def forward(self, visual_feature_map: torch.Tensor) -> AttentionBranchOutput:
        heatmap = self.saliency_predictor(visual_feature_map)
        attention_embedding = self.attention_encoder(heatmap)
        attention_token = self.token_proj(attention_embedding).unsqueeze(1)
        return AttentionBranchOutput(
            heatmap=heatmap,
            attention_embedding=attention_embedding,
            attention_token=attention_token,
        )
