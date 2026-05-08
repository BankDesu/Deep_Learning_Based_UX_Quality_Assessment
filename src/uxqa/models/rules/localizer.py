"""
IssueLocalizer — maps UX rule violations to element-level bounding boxes.

Each ViolationInstance carries:
  - bbox         : which element/region on screen has the issue
  - rule_name    : which rule was violated
  - severity     : "high" / "medium" / "low"
  - description  : human-readable explanation

Rule → localization strategy (7 rules):
  Element-level  (per bbox):  contrast, cta_prominence, alignment
  Image-level    (full bbox): whitespace, density
  Flow-level     (top elements in wrong position): reading_flow
  Balance-level  (centroid region): visual_balance
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .definitions import RULE_NAMES, WCAG_AA_CONTRAST


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ViolationInstance:
    """A single localized UX issue."""
    bbox: list[float]        # normalised [x0, y0, x1, y1] — the problem region
    rule_idx: int            # index into RULE_NAMES
    rule_name: str           # human-readable rule name
    score: float             # rule score ∈ [0,1]; lower = worse violation
    is_violation: bool       # True when score < threshold
    severity: str            # "high" / "medium" / "low"
    description: str         # actionable explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIOLATION_THRESHOLD = 0.5   # score < 0.5 → flagged as violation

def _severity(score: float) -> str:
    if score < 0.25:
        return "high"
    elif score < 0.5:
        return "medium"
    else:
        return "low"


def _relative_luminance_np(r: float, g: float, b: float) -> float:
    """WCAG relative luminance for a single RGB pixel (values in [0,1])."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio_at_bbox(
    img: torch.Tensor,   # [3, H, W] float [0,1]
    box: torch.Tensor,   # [4] normalised xyxy
) -> float:
    H, W = img.shape[1], img.shape[2]
    x0, y0, x1, y1 = box.tolist()
    cx = int(((x0 + x1) / 2) * (W - 1))
    cy = int(((y0 + y1) / 2) * (H - 1))
    bx = max(int(x0 * (W - 1)) - 2, 0)
    by = max(int(y0 * (H - 1)) - 2, 0)
    cx, cy = min(cx, W - 1), min(cy, H - 1)
    bx, bx = min(bx, W - 1), min(bx, W - 1)

    inner = img[:, cy, cx].tolist()
    outer = img[:, by, bx].tolist()
    L1 = _relative_luminance_np(*inner)
    L2 = _relative_luminance_np(*outer)
    light, dark = max(L1, L2), min(L1, L2)
    return (light + 0.05) / (dark + 0.05)


# ---------------------------------------------------------------------------
# IssueLocalizer
# ---------------------------------------------------------------------------

class IssueLocalizer:
    """
    Produces a list of ViolationInstance objects for a single image.

    Usage
    -----
    localizer = IssueLocalizer()
    violations = localizer.localize(
        screenshot=img,          # [3, H, W] float tensor
        boxes=boxes,             # [N, 4] normalised xyxy
        rule_scores=scores,      # [9] float tensor from RuleChecker
        interactive_mask=imask,  # [N] bool or None
        button_mask=bmask,       # [N] bool or None
        heading_levels=hlevels,  # [N] long or None
    )
    """

    def __init__(self, threshold: float = _VIOLATION_THRESHOLD) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        screenshot: torch.Tensor,               # [3, H, W]
        boxes: torch.Tensor,                     # [N, 4]
        rule_scores: torch.Tensor,               # [7]
        button_mask: torch.Tensor | None = None,
    ) -> list[ViolationInstance]:
        """
        Returns all ViolationInstances for a single image.
        Only instances with is_violation=True are actionable; the full list
        is returned so callers can filter by severity or rule as needed.
        """
        img = screenshot.float().clamp(0.0, 1.0)
        violations: list[ViolationInstance] = []

        # Rule 0 — Contrast (element-level)
        violations += self._check_contrast(img, boxes, rule_scores[0])

        # Rule 1 — Whitespace (image-level)
        violations += self._check_whitespace(rule_scores[1])

        # Rule 2 — Visual Balance (centroid-region)
        violations += self._check_visual_balance(boxes, rule_scores[2])

        # Rule 3 — Density (image-level)
        violations += self._check_density(rule_scores[3])

        # Rule 4 — Alignment (element-level)
        violations += self._check_alignment(boxes, rule_scores[4])

        # Rule 5 — CTA Prominence (per button element)
        violations += self._check_cta_prominence(boxes, button_mask, rule_scores[5])

        # Rule 6 — Reading Flow (top-important elements)
        violations += self._check_reading_flow(boxes, rule_scores[6])

        return violations

    # ------------------------------------------------------------------
    # Per-rule localization methods
    # ------------------------------------------------------------------

    def _check_contrast(
        self,
        img: torch.Tensor,
        boxes: torch.Tensor,
        global_score: torch.Tensor,
    ) -> list[ViolationInstance]:
        """Element-level: flag each element whose contrast ratio < WCAG AA."""
        results = []
        for box in boxes:
            ratio = _contrast_ratio_at_bbox(img, box)
            score = min(ratio / WCAG_AA_CONTRAST, 1.0)
            sev = _severity(score)
            results.append(ViolationInstance(
                bbox=box.tolist(),
                rule_idx=0,
                rule_name="contrast",
                score=score,
                is_violation=score < self.threshold,
                severity=sev,
                description=(
                    f"Contrast ratio ≈ {ratio:.1f} "
                    f"(WCAG AA requires ≥ {WCAG_AA_CONTRAST}). "
                    "Increase colour difference between element and background."
                ) if score < self.threshold else "Contrast OK.",
            ))
        return results

    def _check_whitespace(self, score: torch.Tensor) -> list[ViolationInstance]:
        """Image-level: whole-screen annotation when whitespace is off."""
        s = score.item()
        return [ViolationInstance(
            bbox=[0.0, 0.0, 1.0, 1.0],
            rule_idx=1,
            rule_name="whitespace",
            score=s,
            is_violation=s < self.threshold,
            severity=_severity(s),
            description=(
                "Whitespace is outside the ideal range (25–60%). "
                "Too cluttered: remove elements. Too sparse: add content or reduce empty space."
            ) if s < self.threshold else "Whitespace balance OK.",
        )]

    def _check_visual_balance(
        self,
        boxes: torch.Tensor,
        score: torch.Tensor,
    ) -> list[ViolationInstance]:
        """Centroid-region: annotate a small box at the visual centroid."""
        s = score.item()
        if boxes.numel() == 0:
            cx = 0.5
        else:
            cx = float(((boxes[:, 0] + boxes[:, 2]) / 2.0).mean().item())
        cy = 0.5
        # Mark centroid region (5% of screen width)
        region = [
            max(cx - 0.025, 0.0), max(cy - 0.05, 0.0),
            min(cx + 0.025, 1.0), min(cy + 0.05, 1.0),
        ]
        offset_desc = "left-heavy" if cx < 0.45 else "right-heavy" if cx > 0.55 else "balanced"
        return [ViolationInstance(
            bbox=region,
            rule_idx=2,
            rule_name="visual_balance",
            score=s,
            is_violation=s < self.threshold,
            severity=_severity(s),
            description=(
                f"Visual centroid is {offset_desc} (cx ≈ {cx:.2f}). "
                "Redistribute elements for a more symmetric layout."
            ) if s < self.threshold else "Visual balance OK.",
        )]

    def _check_density(self, score: torch.Tensor) -> list[ViolationInstance]:
        """Image-level: whole-screen when UI is too cluttered."""
        s = score.item()
        return [ViolationInstance(
            bbox=[0.0, 0.0, 1.0, 1.0],
            rule_idx=3,
            rule_name="density",
            score=s,
            is_violation=s < self.threshold,
            severity=_severity(s),
            description=(
                "Too many elements detected — UI appears cluttered. "
                "Consider grouping, hiding secondary content, or adding whitespace."
            ) if s < self.threshold else "Element density OK.",
        )]

    def _check_alignment(
        self,
        boxes: torch.Tensor,
        score: torch.Tensor,
    ) -> list[ViolationInstance]:
        """Element-level: flag elements whose left-edge doesn't align with any other."""
        if boxes.shape[0] < 2:
            return []
        results = []
        x0s = boxes[:, 0]
        bins = 10
        x0_q = (x0s * bins).round() / bins    # quantised left edges

        for i, box in enumerate(boxes):
            matches = ((x0_q - x0_q[i]).abs() < 1e-3).sum().item()
            # Only one element at this x → potentially misaligned
            if matches <= 1:
                s = max(score.item() - 0.1, 0.0)  # slightly worse than global
                results.append(ViolationInstance(
                    bbox=box.tolist(),
                    rule_idx=4,
                    rule_name="alignment",
                    score=s,
                    is_violation=True,
                    severity=_severity(s),
                    description=(
                        f"Element left-edge (x ≈ {box[0]:.2f}) has no alignment "
                        "partner. Snap to the nearest grid column."
                    ),
                ))
        return results

    def _check_cta_prominence(
        self,
        boxes: torch.Tensor,
        button_mask: torch.Tensor | None,
        score: torch.Tensor,
    ) -> list[ViolationInstance]:
        """Per button element: flag CTAs that are too small relative to average."""
        N = boxes.shape[0]
        if N == 0:
            return []

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        mean_area = areas.mean().item()

        if button_mask is not None:
            mask = button_mask[:N]
            btn_boxes = boxes[mask]
            btn_areas = areas[mask]
        else:
            btn_boxes = boxes[:1]
            btn_areas = areas[:1]

        results = []
        for box, area in zip(btn_boxes, btn_areas):
            ratio = area.item() / (mean_area + 1e-6)
            s = min(ratio / 2.0, 1.0)
            results.append(ViolationInstance(
                bbox=box.tolist(),
                rule_idx=5,
                rule_name="cta_prominence",
                score=s,
                is_violation=s < self.threshold,
                severity=_severity(s),
                description=(
                    f"CTA/button area is {ratio:.1f}× average element area "
                    "(target ≥ 2×). Increase button size or reduce surrounding elements."
                ) if s < self.threshold else "CTA prominence OK.",
            ))
        return results

    def _check_reading_flow(
        self,
        boxes: torch.Tensor,
        score: torch.Tensor,
    ) -> list[ViolationInstance]:
        """Flag top-important elements that are positioned too low on screen."""
        N = boxes.shape[0]
        if N < 2:
            return []

        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        importance = (1.0 - cy) * (1.0 - cx)
        top_k = max(N // 3, 1)
        top_idx = importance.topk(top_k).indices

        results = []
        for idx in top_idx:
            box = boxes[idx]
            elem_cy = cy[idx].item()
            # Flag if important element is in lower half
            if elem_cy > 0.4:
                s = max(1.0 - elem_cy / 0.4, 0.0) * score.item()
                results.append(ViolationInstance(
                    bbox=box.tolist(),
                    rule_idx=6,
                    rule_name="reading_flow",
                    score=s,
                    is_violation=True,
                    severity=_severity(s),
                    description=(
                        f"High-importance element found at y ≈ {elem_cy:.2f} "
                        "(should be in upper 40% for F/Z reading pattern). "
                        "Move key content toward the top."
                    ),
                ))
        return results

