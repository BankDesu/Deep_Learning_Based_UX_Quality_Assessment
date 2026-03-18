import torch
from torch import nn


class CrossModalTransformer(nn.Module):
    """
    Cross-modal fusion:
    - layout-aware bias by scaling layout token with graph density
    - attention-guided weighting on visual tokens before fusion
    """

    def __init__(
        self,
        dim: int,
        layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        layout_token: torch.Tensor,
        attention_token: torch.Tensor,
        graph_density: torch.Tensor,
    ) -> torch.Tensor:
        b = visual_tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        density_scale = (1.0 + graph_density).reshape(b, 1, 1)
        scaled_layout = layout_token * density_scale
        tokens = torch.cat([cls, visual_tokens, scaled_layout, attention_token], dim=1)
        fused = self.encoder(tokens)
        return self.norm(fused)
