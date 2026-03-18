import torch
from torch import nn


class SwinTransformerEncoder(nn.Module):
    """
    Lightweight Swin-like encoder scaffold:
    patch embedding + transformer encoder over patch tokens + 2D reshape.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)
