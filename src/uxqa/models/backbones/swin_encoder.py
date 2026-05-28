import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinTransformerEncoder(nn.Module):
    """
    Swin-T (ImageNet-1k pretrained, torchvision) as visual encoder.

    Wraps `torchvision.models.swin_t` and projects the final 768-channel
    feature map down to `embed_dim` to preserve the downstream contract.

    Input  : [B, 3, H, W]  (H, W expected to be 224 to match pretraining)
    Output : [B, embed_dim, H/32, W/32]   (e.g. 7×7 for H=W=224)

    The unused constructor args (patch_size, depth, num_heads, ff_dim, dropout)
    are kept for backward compatibility with existing ModelConfig fields.
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
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("Swin-T pretrained requires in_channels=3")

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_t(weights=weights)
        self.features = swin.features
        self.norm = swin.norm
        self.proj = nn.Conv2d(768, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)            # [B, H', W', 768] (channels-last)
        feat = self.norm(feat)             # [B, H', W', 768]
        feat = feat.permute(0, 3, 1, 2)    # [B, 768, H', W']
        return self.proj(feat)             # [B, embed_dim, H', W']
