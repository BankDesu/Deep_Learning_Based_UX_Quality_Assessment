"""
PixelStructureEncoder — interpretable layout feature extractor.

Computes 19 handcrafted features directly from screenshot pixels without
requiring UI hierarchy annotations or external detectors. Features are
grouped into two categories:

  Pixel Rules (5)   — from pixel_rules.py (contrast, whitespace, balance,
                       simplicity, reading_flow)
  Extended (14)     — color diversity, saturation, vertical symmetry,
                       3x3 spatial grid edge density (9), global lum std,
                       and luminance variance

All features are in [0, 1] after normalization. The encoder projects the
concatenated feature vector to a dense embedding via a small MLP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.pixel_rules import compute_pixel_rules

N_STRUCT_FEATURES = 19


def _luminance(img: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] → [B,1,H,W]"""
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def _sobel_edges(lum: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] → edge magnitude [B,1,H,W]"""
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    gx = F.conv2d(lum, sx, padding=1)
    gy = F.conv2d(lum, sy, padding=1)
    return (gx.pow(2) + gy.pow(2)).sqrt()


def _color_diversity(img: torch.Tensor) -> torch.Tensor:
    """Entropy of hue histogram (16 bins) as proxy for color variety.  [B]"""
    B = img.shape[0]
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    cmax = torch.stack([r, g, b], dim=1).max(dim=1).values
    cmin = torch.stack([r, g, b], dim=1).min(dim=1).values
    delta = (cmax - cmin).clamp(min=1e-6)

    hue = torch.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 1e-6)
    mask_g = (cmax == g) & (delta > 1e-6)
    mask_b = (cmax == b) & (delta > 1e-6)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    hue = (hue / 6.0).clamp(0.0, 1.0)  # normalize to [0,1]

    H, W = img.shape[2], img.shape[3]
    hue_flat = hue.reshape(B, H * W)
    n_bins = 16
    scores = []
    for i in range(B):
        hist = torch.histc(hue_flat[i], bins=n_bins, min=0.0, max=1.0)
        prob = hist / (hist.sum() + 1e-8)
        entropy = -(prob * (prob + 1e-8).log()).sum()
        scores.append(entropy / (n_bins * 0.0 + 2.773))  # normalize by log(16)
    return torch.stack(scores).clamp(0.0, 1.0)


def _saturation_stats(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of HSV saturation.  Both in [0,1].  Returns (mean, std)."""
    cmax = img.max(dim=1).values              # [B,H,W]
    cmin = img.min(dim=1).values
    sat = torch.where(cmax > 1e-6, (cmax - cmin) / cmax.clamp(min=1e-6),
                      torch.zeros_like(cmax))  # [B,H,W]
    mean_s = sat.mean(dim=[1, 2])
    std_s  = sat.std(dim=[1, 2])
    return mean_s.clamp(0.0, 1.0), (std_s * 4.0).clamp(0.0, 1.0)


def _vertical_symmetry(img: torch.Tensor) -> torch.Tensor:
    """Pearson r between left and right halves of luminance image.  [B]"""
    lum = _luminance(img).squeeze(1)          # [B,H,W]
    W = lum.shape[2]
    mid = W // 2
    left  = lum[:, :, :mid].reshape(lum.shape[0], -1)   # [B, H*mid]
    right = lum[:, :, W - mid:].flip(dims=[1]).reshape(lum.shape[0], -1)

    left_c  = left  - left.mean(dim=1, keepdim=True)
    right_c = right - right.mean(dim=1, keepdim=True)
    num = (left_c * right_c).sum(dim=1)
    denom = (left_c.pow(2).sum(dim=1) * right_c.pow(2).sum(dim=1)).sqrt().clamp(min=1e-6)
    return ((num / denom + 1.0) / 2.0).clamp(0.0, 1.0)  # map [-1,1] → [0,1]


def _spatial_grid_density(img: torch.Tensor, grid: int = 3) -> torch.Tensor:
    """Edge density in each cell of a grid×grid grid.  [B, grid*grid]"""
    lum = _luminance(img)
    edges = _sobel_edges(lum).squeeze(1)      # [B,H,W]
    B, H, W = edges.shape
    ph, pw = H // grid, W // grid
    cells = []
    for r in range(grid):
        for c in range(grid):
            cell = edges[:, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            cells.append(cell.mean(dim=[1, 2]))
    density = torch.stack(cells, dim=1)       # [B, grid*grid]
    # Normalize per image: divide by max across all cells
    d_max = density.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    return (density / d_max).clamp(0.0, 1.0)


def _global_lum_std(img: torch.Tensor) -> torch.Tensor:
    """Global luminance std, normalized to [0,1].  [B]"""
    lum = _luminance(img).squeeze(1)
    return (lum.std(dim=[1, 2]) * 4.0).clamp(0.0, 1.0)


def compute_structural_features(img: torch.Tensor) -> torch.Tensor:
    """
    Compute all 19 structural layout features.

    Parameters
    ----------
    img : FloatTensor [B, 3, H, W]  values in [0, 1]

    Returns
    -------
    feats : FloatTensor [B, 19]  all values in [0, 1]
    """
    pixel_feats = compute_pixel_rules(img)                    # [B, 5]
    color_div   = _color_diversity(img).unsqueeze(1)          # [B, 1]
    sat_mean, sat_std = _saturation_stats(img)                # [B], [B]
    vsym        = _vertical_symmetry(img).unsqueeze(1)        # [B, 1]
    grid_dens   = _spatial_grid_density(img, grid=3)          # [B, 9]
    lum_std     = _global_lum_std(img).unsqueeze(1)           # [B, 1]

    return torch.cat([
        pixel_feats,                     # 5
        color_div,                       # 1
        sat_mean.unsqueeze(1),           # 1
        sat_std.unsqueeze(1),            # 1
        vsym,                            # 1
        grid_dens,                       # 9
        lum_std,                         # 1
    ], dim=1)                            # [B, 19]


class PixelStructureEncoder(nn.Module):
    """
    Encode 19 pixel-based structural features into a dense embedding.

    Projects via a 2-layer MLP with GELU activation and dropout.
    """

    def __init__(self, out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N_STRUCT_FEATURES, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img [B,3,H,W] in [0,1] → embedding [B, out_dim]"""
        with torch.no_grad():
            feats = compute_structural_features(img)  # [B, 19]
        return self.norm(self.mlp(feats))
