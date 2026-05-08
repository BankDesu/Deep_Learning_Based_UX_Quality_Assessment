import torch
import torch.nn.functional as F
from torch import nn


class CrossModalTransformer(nn.Module):
    """
    Cross-modal fusion:
    - layout-aware bias by scaling layout token with graph density
    - attention-guided weighting on visual tokens before fusion
    - max_visual_tokens caps sequence length via adaptive avg-pool
      to keep attention memory tractable (O(n²) cost)
    """

    def __init__(
        self,
        dim: int,
        layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_visual_tokens: int = 256,
    ) -> None:
        self.max_visual_tokens = max_visual_tokens
        super().__init__()
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.layout_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
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

        # Cap sequence length: [B, S, D] → pool → [B, max_S, D]
        # CPU fallback required: MPS does not support adaptive_avg_pool1d
        # when input length is not divisible by output length (pytorch#96056).
        if visual_tokens.shape[1] > self.max_visual_tokens:
            dev = visual_tokens.device
            vt = visual_tokens.transpose(1, 2).cpu()                    # [B, D, S] on CPU
            vt = F.adaptive_avg_pool1d(vt, self.max_visual_tokens)      # [B, D, max_S]
            visual_tokens = vt.transpose(1, 2).to(dev)                  # [B, max_S, D]

        # Attention-guided feature weighting from the attention branch token.
        attention_scale = self.attention_gate(attention_token.squeeze(1)).unsqueeze(1)
        guided_visual = visual_tokens * (1.0 + attention_scale)

        # Layout-aware bias with graph density modulating layout and visual influence.
        density_scale = (1.0 + graph_density).reshape(b, 1, 1)
        scaled_layout = layout_token * density_scale
        layout_scale = self.layout_gate(layout_token.squeeze(1)).unsqueeze(1)
        guided_visual = guided_visual * (1.0 + (layout_scale * density_scale))

        tokens = torch.cat([cls, guided_visual, scaled_layout, attention_token], dim=1)
        fused = self.encoder(tokens)
        return self.norm(fused)
