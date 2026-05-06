"""
RuleChecker — algorithmic UX rule evaluation module.

Each rule function:
  - Takes detected element bounding boxes and/or screenshot pixels
  - Returns a score tensor of shape [B] in range [0, 1]
  - Score 1.0 = rule satisfied (good UX)
  - Score 0.0 = rule violated (bad UX)

Rules removed (require UI hierarchy / accessibility tree):
  - touch_target      (was rule 5)
  - heading_hierarchy (was rule 8)

Remaining 7 rules — 3 are heatmap-enhanced (★):
  0  contrast        — WCAG AA contrast proxy
  1  whitespace      — empty space balance
  2  visual_balance  — horizontal centroid near center      ★
  3  density         — element count vs max
  4  alignment       — left-edge grid consistency
  5  cta_prominence  — button area vs average + attention   ★
  6  reading_flow    — top-weighted elements in upper area  ★

When heatmap=None, ★ rules fall back to geometry-only scoring.
Blend ratio: 0.5 geometric + 0.5 attention.
"""
from __future__ import annotations

import torch
from torch import nn

from .definitions import (
    N_RULES,
    WCAG_AA_CONTRAST,
    WHITESPACE_HI,
    WHITESPACE_LO,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_batch_boxes(boxes: torch.Tensor, B: int) -> torch.Tensor:
    """Ensure boxes is [B, N, 4]. Expand if necessary."""
    if boxes.dim() == 2:
        return boxes.unsqueeze(0).expand(B, -1, -1)
    return boxes


def _heatmap_overlap_with_box(
    heatmap: torch.Tensor,
    box: torch.Tensor,
    top_pct: float = 0.20,
) -> torch.Tensor:
    """
    Compute fraction of top-salient pixels that fall inside `box`.

    Parameters
    ----------
    heatmap : FloatTensor [H, W]  — single-image saliency map in [0, 1]
    box     : FloatTensor [4]     — normalised xyxy
    top_pct : float               — top percentile threshold (default 20%)

    Returns
    -------
    overlap : scalar Tensor in [0, 1]
    """
    H, W = heatmap.shape
    threshold = heatmap.flatten().quantile(1.0 - top_pct)
    top_mask = (heatmap >= threshold).float()

    x0, y0, x1, y1 = box
    px0 = int((x0 * W).clamp(0, W - 1).item())
    py0 = int((y0 * H).clamp(0, H - 1).item())
    px1 = int((x1 * W).clamp(0, W).item())
    py1 = int((y1 * H).clamp(0, H).item())

    region_mask = torch.zeros_like(top_mask)
    region_mask[py0:py1, px0:px1] = 1.0

    overlap = (top_mask * region_mask).sum()
    total_top = top_mask.sum().clamp_min(1e-6)
    return (overlap / total_top).clamp(0.0, 1.0)


def _saliency_map_2d(heatmap: torch.Tensor, b: int) -> torch.Tensor:
    """Extract single-image saliency map as [H, W] from [B, 1, H, W]."""
    return heatmap[b, 0]


def _relative_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """
    Compute relative luminance per the WCAG definition.
    rgb : FloatTensor [..., 3] with values in [0, 1]
    """
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def rule_contrast(screenshot: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Rule 0 — Color Contrast (WCAG AA proxy).

    Samples centre and border pixels for each detected element and computes
    the contrast ratio. Scores are averaged and clamped to [0,1].
    """
    B, C, H, W = screenshot.shape
    boxes_b = _to_batch_boxes(boxes, B)
    scores: list[torch.Tensor] = []

    for b in range(B):
        img = screenshot[b].float().clamp(0.0, 1.0)
        bboxes = boxes_b[b]
        if bboxes.numel() == 0:
            scores.append(torch.tensor(0.5, device=screenshot.device))
            continue

        ratios: list[torch.Tensor] = []
        for box in bboxes[:12]:
            x0, y0, x1, y1 = box.unbind()
            cx = ((x0 + x1) * 0.5 * (W - 1)).long().clamp(0, W - 1)
            cy = ((y0 + y1) * 0.5 * (H - 1)).long().clamp(0, H - 1)
            inner_rgb = img[:, cy, cx]

            bx = (x0 * (W - 1) - 2).long().clamp(0, W - 1)
            by = (y0 * (H - 1) - 2).long().clamp(0, H - 1)
            outer_rgb = img[:, by, bx]

            L1 = _relative_luminance(inner_rgb)
            L2 = _relative_luminance(outer_rgb)
            light = torch.max(L1, L2)
            dark  = torch.min(L1, L2)
            ratio = (light + 0.05) / (dark + 0.05)
            ratios.append(ratio)

        mean_ratio = torch.stack(ratios).mean()
        scores.append((mean_ratio / WCAG_AA_CONTRAST).clamp(0.0, 1.0))

    return torch.stack(scores)


def rule_whitespace(boxes: torch.Tensor, grid: int = 32) -> torch.Tensor:
    """
    Rule 1 — Whitespace Balance.

    Rasterises bounding boxes onto a coarse grid and computes the fraction
    of unoccupied cells. Score peaks when whitespace is in [WHITESPACE_LO, WHITESPACE_HI].
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B = boxes_b.shape[0]
    scores: list[torch.Tensor] = []

    midpoint = (WHITESPACE_LO + WHITESPACE_HI) / 2.0
    half_range = (WHITESPACE_HI - WHITESPACE_LO) / 2.0

    for b in range(B):
        occupied = torch.zeros(grid, grid, device=boxes.device)
        for box in boxes_b[b]:
            gx0 = int((box[0] * grid).clamp(0, grid - 1).item())
            gy0 = int((box[1] * grid).clamp(0, grid - 1).item())
            gx1 = int((box[2] * grid).clamp(0, grid).item())
            gy1 = int((box[3] * grid).clamp(0, grid).item())
            occupied[gy0:gy1, gx0:gx1] = 1.0

        ws = 1.0 - occupied.mean()
        score = 1.0 - ((ws - midpoint).abs() / (half_range + 1e-6)).clamp(0.0, 1.0)
        scores.append(score)

    return torch.stack(scores)


def rule_visual_balance(
    boxes: torch.Tensor,
    heatmap: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rule 2 — Visual Balance.  ★ heatmap-enhanced

    Geometric: horizontal centroid of elements close to screen centre (x=0.5).
    Heatmap  : saliency-weighted centroid should also be centred horizontally.
    Blend    : 0.5 × geometric + 0.5 × attention (when heatmap provided).
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B = boxes_b.shape[0]
    scores: list[torch.Tensor] = []

    for b in range(B):
        bboxes = boxes_b[b]

        if bboxes.numel() == 0:
            geo_score = torch.tensor(0.5, device=boxes.device)
        else:
            cx_geo = ((bboxes[:, 0] + bboxes[:, 2]) / 2.0).mean()
            geo_score = (1.0 - 2.0 * (cx_geo - 0.5).abs()).clamp(0.0, 1.0)

        if heatmap is not None:
            h_map = _saliency_map_2d(heatmap, b)
            W = h_map.shape[1]
            xs = torch.linspace(0.0, 1.0, W, device=heatmap.device)
            weights = h_map / (h_map.sum() + 1e-6)
            cx_attn = (weights * xs.unsqueeze(0)).sum()
            attn_score = (1.0 - 2.0 * (cx_attn - 0.5).abs()).clamp(0.0, 1.0)
            score = 0.5 * geo_score + 0.5 * attn_score
        else:
            score = geo_score

        scores.append(score)

    return torch.stack(scores)


def rule_density(boxes: torch.Tensor, max_elements: int = 16) -> torch.Tensor:
    """
    Rule 3 — Element Density (anti-clutter).

    Score falls quadratically as the UI gets crowded above max_elements.
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B, N, _ = boxes_b.shape
    ratio = N / max(max_elements, 1)
    score = (1.0 - ratio ** 2).clamp(0.0, 1.0)
    return torch.full((B,), score, device=boxes.device, dtype=torch.float32)


def rule_alignment(boxes: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """
    Rule 4 — Edge Alignment Consistency.

    Elements sharing common left-edge x-positions indicate intentional structure.
    Score = fraction of x-edges that have at least one neighbour in the same bin.
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B = boxes_b.shape[0]
    scores: list[torch.Tensor] = []

    for b in range(B):
        bboxes = boxes_b[b]
        N = bboxes.shape[0]
        if N < 2:
            scores.append(torch.tensor(1.0, device=boxes.device))
            continue

        x0_q = (bboxes[:, 0] * bins).round()
        x1_q = (bboxes[:, 2] * bins).round()
        all_edges = torch.cat([x0_q, x1_q])

        _, counts = all_edges.unique(return_counts=True)
        aligned = (counts > 1).float().sum()
        score = (aligned / counts.numel()).clamp(0.0, 1.0)
        scores.append(score)

    return torch.stack(scores)


def rule_cta_prominence(
    boxes: torch.Tensor,
    button_mask: torch.Tensor | None,
    heatmap: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rule 5 — CTA / Button Prominence.  ★ heatmap-enhanced

    Geometric: CTA buttons should be larger than average element (area ratio).
    Heatmap  : top-salient pixels should overlap with button bounding boxes.
    Blend    : 0.5 × geometric + 0.5 × attention (when heatmap provided).

    Falls back to first element as proxy CTA when button_mask is None.
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B, N, _ = boxes_b.shape
    scores: list[torch.Tensor] = []

    for b in range(B):
        bboxes = boxes_b[b]
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        areas = w * h

        if button_mask is not None:
            mask = (button_mask[b][:N] if button_mask.dim() == 2 else button_mask[:N])
            button_boxes = bboxes[mask]
            button_areas = areas[mask]
        else:
            button_boxes = bboxes[:1]
            button_areas = areas[:1]

        if button_areas.numel() == 0 or areas.numel() == 0:
            scores.append(torch.tensor(0.5, device=boxes.device))
            continue

        mean_area = areas.mean()
        ratio = button_areas.mean() / (mean_area + 1e-6)
        geo_score = (ratio / 2.0).clamp(0.0, 1.0)

        if heatmap is not None:
            h_map = _saliency_map_2d(heatmap, b)
            overlaps = [_heatmap_overlap_with_box(h_map, box) for box in button_boxes]
            attn_score = torch.stack(overlaps).mean() if overlaps else torch.tensor(0.5, device=boxes.device)
            score = 0.5 * geo_score + 0.5 * attn_score
        else:
            score = geo_score

        scores.append(score)

    return torch.stack(scores)


def rule_reading_flow(
    boxes: torch.Tensor,
    heatmap: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rule 6 — Reading Flow (F / Z Pattern).  ★ heatmap-enhanced

    Geometric: top-weighted elements (large, near top-left) should be in
               the upper portion of the screen (cy < 0.4).
    Heatmap  : top-20% salient pixels should concentrate in upper half.
    Blend    : 0.5 × geometric + 0.5 × attention (when heatmap provided).
    """
    boxes_b = _to_batch_boxes(boxes, max(boxes.shape[0] if boxes.dim() == 3 else 1, 1))
    B, N, _ = boxes_b.shape
    scores: list[torch.Tensor] = []

    for b in range(B):
        bboxes = boxes_b[b]

        if N < 2:
            geo_score = torch.tensor(1.0, device=boxes.device)
        else:
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            importance = (1.0 - cy) * (1.0 - cx)
            top_k = max(N // 3, 1)
            top_idx = importance.topk(top_k).indices
            top_cy = cy[top_idx].mean()
            geo_score = (1.0 - top_cy / 0.4).clamp(0.0, 1.0)

        if heatmap is not None:
            h_map = _saliency_map_2d(heatmap, b)
            H_px = h_map.shape[0]
            threshold = h_map.flatten().quantile(0.80)
            top_mask = h_map >= threshold

            if top_mask.any():
                rows = torch.arange(H_px, device=heatmap.device).float() / H_px
                top_rows = rows.unsqueeze(1).expand_as(h_map)[top_mask]
                mean_row = top_rows.mean()
                attn_score = (1.0 - mean_row / 0.45).clamp(0.0, 1.0)
            else:
                attn_score = torch.tensor(0.5, device=heatmap.device)

            score = 0.5 * geo_score + 0.5 * attn_score
        else:
            score = geo_score

        scores.append(score)

    return torch.stack(scores)


# ---------------------------------------------------------------------------
# RuleChecker module
# ---------------------------------------------------------------------------

class RuleChecker(nn.Module):
    """
    Computes all 7 UX rule scores algorithmically (no learned parameters).

    forward() returns rule_scores : FloatTensor [B, N_RULES]  (N_RULES = 7)
        Each dimension corresponds to RULE_NAMES in definitions.py.
        Values are in [0, 1], higher = better UX.

    Rules that previously required hierarchy data (touch_target,
    heading_hierarchy) have been removed. Only screenshot + detected
    boxes + optional button_mask + optional heatmap are needed.

    Parameters
    ----------
    max_elements : int
        Used for density rule normalisation.
    """

    def __init__(self, max_elements: int = 16) -> None:
        super().__init__()
        self.max_elements = max_elements

    def forward(
        self,
        screenshot: torch.Tensor,               # [B, 3, H, W]
        boxes: torch.Tensor,                     # [N, 4] or [B, N, 4] normalised xyxy
        button_mask: torch.Tensor | None = None, # [N] or [B, N] bool — for cta_prominence
        heatmap: torch.Tensor | None = None,     # [B, 1, H', W'] from AttentionBranch
    ) -> torch.Tensor:
        """
        Returns
        -------
        rule_scores : FloatTensor [B, 7]
            Each value in [0, 1]. Higher = better UX for that rule.
            Rules 2, 5, 6 (★) are enhanced by heatmap when provided.
        """
        B = screenshot.shape[0]
        dev = screenshot.device

        if boxes.dim() == 2:
            boxes_b = boxes.unsqueeze(0).expand(B, -1, -1)
        else:
            boxes_b = boxes

        s0 = rule_contrast(screenshot, boxes_b)
        s1 = rule_whitespace(boxes_b)
        s2 = rule_visual_balance(boxes_b, heatmap=heatmap)              # ★
        s3 = rule_density(boxes_b, self.max_elements)
        s4 = rule_alignment(boxes_b)
        s5 = rule_cta_prominence(boxes_b, button_mask, heatmap=heatmap) # ★
        s6 = rule_reading_flow(boxes_b, heatmap=heatmap)                # ★

        return torch.stack([s0, s1, s2, s3, s4, s5, s6], dim=1).to(
            device=dev, dtype=torch.float32
        )
