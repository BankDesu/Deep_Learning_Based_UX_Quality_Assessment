"""
Pixel-based UX rule approximations — no bounding boxes required.

Computes 5 design-quality proxy scores directly from screenshot pixels.
Used as pseudo-supervision for large-scale pretraining on unlabeled UI datasets
(e.g., RICO 66K) where element detectors are not available or unreliable.

Each score is in [0, 1], higher = better design quality for that dimension.

Scores
------
  0  contrast     — local luminance contrast (proxy for WCAG readability)
  1  whitespace   — background fraction in [0.25, 0.60] ideal range
  2  balance      — horizontal intensity centroid near screen center
  3  simplicity   — low edge density (anti-clutter proxy)
  4  reading_flow — more content weight in upper third (F/Z pattern)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_PIXEL_RULES = 5

PIXEL_RULE_NAMES: list[str] = [
    "contrast",
    "whitespace",
    "balance",
    "simplicity",
    "reading_flow",
]

# Sobel kernels registered once as module buffers when using the nn.Module version
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)


def _luminance(img: torch.Tensor) -> torch.Tensor:
    """img [B,3,H,W] → lum [B,1,H,W], standard ITU-R BT.601 coefficients."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def pixel_contrast(img: torch.Tensor) -> torch.Tensor:
    """
    Score 0 — Local contrast proxy.

    Computes per-patch luminance standard deviation over a 16×16 grid,
    then averages. Score peaks at a mid-contrast level typical of readable UIs
    (high contrast = too jarring; near-zero = low readability).

    Target range empirically set to std ∈ [0.08, 0.30] for a score of 1.
    """
    lum = _luminance(img)                           # [B,1,H,W]
    B, _, H, W = lum.shape
    p = 16
    ph, pw = H // p, W // p
    if ph == 0 or pw == 0:
        return torch.ones(B, device=img.device)

    patches = lum[:, :, :ph * p, :pw * p].reshape(B, 1, ph, p, pw, p)
    patch_std = patches.std(dim=[3, 5]).mean(dim=[1, 2, 3])  # [B]

    # Map std to score: peaks at 0.15, 0 below 0.02 or above 0.40
    score = 1.0 - ((patch_std - 0.15).abs() / 0.25).clamp(0.0, 1.0)
    return score


def pixel_whitespace(img: torch.Tensor, bright_thresh: float = 0.85) -> torch.Tensor:
    """
    Score 1 — Whitespace / breathing room.

    Bright pixels (near white) approximate background/empty space.
    Score peaks when whitespace fraction is in [0.25, 0.60].
    """
    lum = _luminance(img).squeeze(1)               # [B,H,W]
    ws = (lum > bright_thresh).float().mean(dim=[1, 2])   # [B]

    lo, hi = 0.25, 0.60
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    score = (1.0 - ((ws - mid).abs() / (half + 1e-6))).clamp(0.0, 1.0)
    return score


def pixel_balance(img: torch.Tensor) -> torch.Tensor:
    """
    Score 2 — Horizontal visual balance.

    Computes intensity-weighted horizontal centroid.  Score = 1 when centroid
    is at x=0.5 (perfectly balanced), falls off toward edges.
    """
    lum = _luminance(img).squeeze(1)                # [B,H,W]
    B, H, W = lum.shape
    xs = torch.linspace(0.0, 1.0, W, device=img.device)  # [W]
    weight = lum / (lum.sum(dim=[1, 2], keepdim=True) + 1e-6)
    cx = (weight * xs.view(1, 1, W)).sum(dim=[1, 2])  # [B]
    score = (1.0 - 2.0 * (cx - 0.5).abs()).clamp(0.0, 1.0)
    return score


def pixel_simplicity(img: torch.Tensor) -> torch.Tensor:
    """
    Score 3 — Layout simplicity (anti-clutter proxy).

    Edge density via Sobel magnitude serves as a proxy for element count /
    visual busyness.  Low edge density = cleaner layout = higher score.
    Edge density is mapped through a sigmoid so moderate complexity still scores
    reasonably.
    """
    lum = _luminance(img)                           # [B,1,H,W]
    sx = _SOBEL_X.view(1, 1, 3, 3).to(img.device)
    sy = _SOBEL_Y.view(1, 1, 3, 3).to(img.device)
    gx = F.conv2d(lum, sx, padding=1)
    gy = F.conv2d(lum, sy, padding=1)
    edge_density = (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=[1, 2, 3])  # [B]
    # Typical UI edge density ~0.05–0.20; map to score via soft threshold at 0.10
    score = (1.0 - (edge_density / 0.15).clamp(0.0, 1.0))
    return score


def pixel_reading_flow(img: torch.Tensor) -> torch.Tensor:
    """
    Score 4 — Reading flow (F/Z pattern proxy).

    UIs following natural reading patterns place key content in the upper
    region.  Proxy: mean edge density (content weight) in top third should
    exceed that of the bottom third.
    """
    lum = _luminance(img)                           # [B,1,H,W]
    sx = _SOBEL_X.view(1, 1, 3, 3).to(img.device)
    sy = _SOBEL_Y.view(1, 1, 3, 3).to(img.device)
    gx = F.conv2d(lum, sx, padding=1)
    gy = F.conv2d(lum, sy, padding=1)
    edge_mag = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(1)  # [B,H,W]

    H = edge_mag.shape[1]
    top_third  = edge_mag[:, :H // 3, :].mean(dim=[1, 2])
    bot_third  = edge_mag[:, 2 * H // 3:, :].mean(dim=[1, 2])
    diff = top_third - bot_third                    # positive = top-heavy = good
    score = torch.sigmoid(diff * 20.0)              # soft threshold around 0
    return score


def compute_pixel_rules(img: torch.Tensor) -> torch.Tensor:
    """
    Compute all 5 pixel-based UX rule scores.

    Parameters
    ----------
    img : FloatTensor [B, 3, H, W] in [0, 1]

    Returns
    -------
    scores : FloatTensor [B, 5]  all values in [0, 1]
    """
    return torch.stack([
        pixel_contrast(img),
        pixel_whitespace(img),
        pixel_balance(img),
        pixel_simplicity(img),
        pixel_reading_flow(img),
    ], dim=1)
