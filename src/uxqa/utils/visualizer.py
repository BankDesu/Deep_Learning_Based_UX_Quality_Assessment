"""
ViolationVisualizer — draws annotated UI screenshots with localized UX issues.

Output: PIL Image with colour-coded bounding boxes and labels.

Severity colour scheme:
  high   → red     #DC2626
  medium → orange  #EA580C
  low    → yellow  #CA8A04
  ok     → green   #16A34A  (when drawing passing rules)

Usage
-----
from uxqa.utils import ViolationVisualizer
from uxqa.models.rules.localizer import IssueLocalizer

localizer   = IssueLocalizer()
visualizer  = ViolationVisualizer()

violations  = localizer.localize(screenshot, boxes, rule_scores, ...)
image       = visualizer.annotate(screenshot, violations, image_hw=(812, 375))
image.save("result.png")
"""
from __future__ import annotations

from pathlib import Path

import torch

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ..models.rules.localizer import ViolationInstance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "high":   (220, 38,  38),   # red
    "medium": (234, 88,  12),   # orange
    "low":    (202, 138,  4),   # amber
    "ok":     ( 22, 163, 74),   # green
}

_RULE_SHORT_LABELS: dict[str, str] = {
    "contrast":          "Contrast",
    "whitespace":        "Whitespace",
    "visual_balance":    "Balance",
    "density":           "Density",
    "alignment":         "Alignment",
    "touch_target":      "Touch Target",
    "cta_prominence":    "CTA Size",
    "reading_flow":      "Reading Flow",
    "heading_hierarchy": "Heading Order",
}

_BOX_ALPHA    = 140    # box fill transparency (0–255)
_BORDER_WIDTH = 2


# ---------------------------------------------------------------------------
# ViolationVisualizer
# ---------------------------------------------------------------------------

class ViolationVisualizer:
    """
    Draws localized UX violations as annotated bounding boxes on a screenshot.

    Parameters
    ----------
    show_ok : bool
        If True, draw passing rules in green. Default False (violations only).
    min_severity : str
        Minimum severity to draw. One of "high", "medium", "low".
        Default "low" (draw all violations).
    font_size : int
        Label font size in pixels. Default 12.
    """

    _SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

    def __init__(
        self,
        show_ok: bool = False,
        min_severity: str = "low",
        font_size: int = 12,
    ) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for ViolationVisualizer. "
                "Install with: pip install Pillow"
            )
        self.show_ok = show_ok
        self.min_severity = min_severity
        self.font_size = font_size
        self._font = self._load_font(font_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        screenshot: torch.Tensor | "Image.Image",
        violations: list[ViolationInstance],
        image_hw: tuple[int, int] = (812, 375),
    ) -> "Image.Image":
        """
        Draw violation bounding boxes on the screenshot.

        Parameters
        ----------
        screenshot : Tensor [3, H, W] float [0,1]  OR  PIL Image
        violations : output of IssueLocalizer.localize()
        image_hw   : (height, width) of the rendered image in pixels.
                     Normalised coordinates are scaled to this size.

        Returns
        -------
        PIL Image — annotated screenshot ready to save or display.
        """
        H, W = image_hw

        # Convert screenshot to PIL
        if isinstance(screenshot, torch.Tensor):
            base = self._tensor_to_pil(screenshot, W, H)
        else:
            base = screenshot.resize((W, H))

        # Separate draw layers: base image + transparent overlay
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_base = ImageDraw.Draw(base.convert("RGBA"))
        draw_overlay = ImageDraw.Draw(overlay)

        # Filter by severity
        min_rank = self._SEVERITY_RANK[self.min_severity]
        to_draw = [
            v for v in violations
            if (v.is_violation and self._SEVERITY_RANK.get(v.severity, 2) <= min_rank)
            or (self.show_ok and not v.is_violation)
        ]

        # Sort: high severity on top
        to_draw.sort(key=lambda v: self._SEVERITY_RANK.get(v.severity, 2))

        for v in to_draw:
            self._draw_violation(draw_overlay, v, W, H)

        # Composite overlay onto base
        combined = Image.alpha_composite(base.convert("RGBA"), overlay)

        # Draw labels on top (no alpha issues with text)
        draw_labels = ImageDraw.Draw(combined)
        for v in to_draw:
            self._draw_label(draw_labels, v, W, H)

        return combined.convert("RGB")

    def save(
        self,
        screenshot: torch.Tensor | "Image.Image",
        violations: list[ViolationInstance],
        path: str | Path,
        image_hw: tuple[int, int] = (812, 375),
    ) -> Path:
        """Annotate and save to disk. Returns the saved path."""
        img = self.annotate(screenshot, violations, image_hw=image_hw)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return out

    def summary(self, violations: list[ViolationInstance]) -> str:
        """Return a plain-text summary of all violations."""
        active = [v for v in violations if v.is_violation]
        if not active:
            return "No UX violations detected."
        lines = [f"Found {len(active)} violation(s):\n"]
        for v in sorted(active, key=lambda x: self._SEVERITY_RANK.get(x.severity, 2)):
            label = _RULE_SHORT_LABELS.get(v.rule_name, v.rule_name)
            lines.append(
                f"  [{v.severity.upper():6s}] {label:16s} "
                f"(score={v.score:.2f})  bbox={[f'{b:.2f}' for b in v.bbox]}\n"
                f"           → {v.description}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_violation(
        self,
        draw: "ImageDraw.ImageDraw",
        v: ViolationInstance,
        W: int,
        H: int,
    ) -> None:
        color = _SEVERITY_COLORS.get(v.severity if v.is_violation else "ok", (100, 100, 100))
        x0, y0, x1, y1 = self._denorm(v.bbox, W, H)

        # Filled semi-transparent rectangle
        fill = color + (_BOX_ALPHA,)
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=color + (255,), width=_BORDER_WIDTH)

    def _draw_label(
        self,
        draw: "ImageDraw.ImageDraw",
        v: ViolationInstance,
        W: int,
        H: int,
    ) -> None:
        color = _SEVERITY_COLORS.get(v.severity if v.is_violation else "ok", (100, 100, 100))
        x0, y0, x1, y1 = self._denorm(v.bbox, W, H)

        label = _RULE_SHORT_LABELS.get(v.rule_name, v.rule_name)
        sev_mark = {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(v.severity, "")
        text = f"{sev_mark} {label}"

        # Background pill for readability
        try:
            bbox_text = draw.textbbox((0, 0), text, font=self._font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=self._font)  # older Pillow

        pad = 3
        tx = max(x0, 0)
        ty = max(y0 - th - pad * 2 - 2, 0)
        draw.rectangle(
            [tx, ty, tx + tw + pad * 2, ty + th + pad * 2],
            fill=color + (220,) if len(color) == 3 else color,
        )
        draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=self._font)

    @staticmethod
    def _denorm(
        bbox: list[float], W: int, H: int
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        return (
            int(x0 * W), int(y0 * H),
            int(x1 * W), int(y1 * H),
        )

    @staticmethod
    def _tensor_to_pil(img: torch.Tensor, W: int, H: int) -> "Image.Image":
        """Convert [3, H, W] float tensor → PIL Image resized to (W, H)."""
        import numpy as np
        arr = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        arr = arr.transpose(1, 2, 0)   # [H, W, 3]
        pil = Image.fromarray(arr, mode="RGB")
        return pil.resize((W, H))

    @staticmethod
    def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
        """Load a font, falling back to default if none available."""
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except (IOError, OSError):
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
            except (IOError, OSError):
                return ImageFont.load_default()
