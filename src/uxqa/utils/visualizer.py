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
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ..models.rules.localizer import ViolationInstance


# ---------------------------------------------------------------------------
# compute_cam — shared multi-scale head-weight CAM
# ---------------------------------------------------------------------------

def _get_backbone(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, 'visual_backbone', None) or model.backbone


def _get_head(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, 'visual_head', None) or model.head


def _norm_np(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _postprocess_cam(
    cam: np.ndarray,
    out_size: int,
    smooth_ksize: int,
    smooth_sigma: float,
    clip_percentile: tuple[float, float] | None,
    edge_taper: float,
    edge_power: float,
    edge_margin: float,
) -> np.ndarray:
    import cv2

    if cam.shape[0] != out_size or cam.shape[1] != out_size:
        cam = cv2.resize(cam, (out_size, out_size))

    if smooth_ksize and smooth_ksize > 1:
        k = smooth_ksize + 1 if smooth_ksize % 2 == 0 else smooth_ksize
        cam = cv2.GaussianBlur(cam, (k, k), smooth_sigma)

    taper = None
    window = None
    if edge_taper and edge_taper > 0:
        taper = float(np.clip(edge_taper, 0.0, 1.0))
        window = np.outer(np.hanning(out_size), np.hanning(out_size)).astype(np.float32)
        if edge_power and edge_power > 1:
            window = window ** float(edge_power)
        cam = cam * ((1.0 - taper) + taper * window)

    m = int(round(out_size * float(np.clip(edge_margin, 0.0, 0.25))))
    inner = cam[m:out_size - m, m:out_size - m] if m * 2 < out_size else cam

    if clip_percentile is None:
        cam = (cam - inner.min()) / (inner.max() - inner.min() + 1e-8)
    else:
        lo, hi = np.percentile(inner, clip_percentile)
        cam = (cam - lo) / (hi - lo + 1e-8)

    if edge_taper and edge_taper > 0:
        cam = cam * ((1.0 - taper) + taper * window)

    if m > 0:
        cam[:m, :] = 0.0
        cam[-m:, :] = 0.0
        cam[:, :m] = 0.0
        cam[:, -m:] = 0.0

    return np.clip(cam, 0.0, 1.0).astype(np.float32)


def compute_cam(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    out_size: int = 224,
    smooth_ksize: int = 7,
    smooth_sigma: float = 0.0,
    clip_percentile: tuple[float, float] | None = (5.0, 95.0),
    edge_taper: float = 0.45,
    edge_power: float = 2.0,
    edge_margin: float = 0.06,
) -> np.ndarray:
    """Multi-scale head-weight CAM for EfficientNet-B4 based models.

    Combines two feature layers weighted by the quality head's linear weights.
    No backward pass required — avoids corner bias caused by frozen-backbone
    Grad-CAM with GAP (spatially uniform gradients).

    Blend weights:
      80%  features[-1] 7×7   — head-weight CAM  (what the quality head cares about)
      20%  features[5]  14×14 — head-weight CAM  (finer spatial detail)

    Note: activation-energy terms (squared feature magnitudes) were removed because
    they reflect backbone firing patterns, not quality-head decisions, and cause a
    systematic top-right corner bias due to EfficientNet zero-padding artifacts.

    Parameters
    ----------
    model    : MultiModalQualityModel or legacy linear probe
    img_t    : FloatTensor [3, H, W] ImageNet-normalized
    device   : torch device
    out_size : output spatial resolution (square)
    smooth_ksize   : Gaussian blur kernel size (odd). 0/1 disables smoothing.
    smooth_sigma   : Gaussian blur sigma (0 = auto)
    clip_percentile: robust normalization (lo, hi). None uses min-max.
    edge_taper     : 0..1 cosine window blend to reduce corner bias
    edge_power     : exponent for edge window (>=1 boosts edge suppression)
    edge_margin    : hard mask ratio for borders (0..0.25 recommended)

    Returns
    -------
    cam : float32 ndarray [out_size, out_size] in [0, 1]
    """
    import cv2

    bb   = _get_backbone(model)
    head = _get_head(model)

    acts_last: list = [None]
    acts_mid1: list = [None]

    def _h_last(_, __, out): acts_last[0] = out.detach().cpu()
    def _h_mid1(_, __, out): acts_mid1[0] = out.detach().cpu()

    h1 = bb.features[-1].register_forward_hook(_h_last)
    h3 = bb.features[5].register_forward_hook(_h_mid1)

    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)
    h1.remove(); h3.remove()

    # Head weight vector: w = W_out @ W_hidden  [feat_dim]
    W_out    = head[3].weight.detach().cpu()   # [1, 256]
    W_hidden = head[0].weight.detach().cpu()   # [256, feat_dim]
    w = (W_out @ W_hidden).squeeze(0)          # [feat_dim]

    def _head_weight_cam(acts: torch.Tensor, w_vec: torch.Tensor) -> np.ndarray:
        A = acts.squeeze(0)  # [C, H, W]
        # Project w onto the channel dim of this layer (truncate or pad if needed)
        c = A.shape[0]
        wc = w_vec[:c] if c <= w_vec.shape[0] else torch.cat(
            [w_vec, w_vec.new_zeros(c - w_vec.shape[0])]
        )
        # Subtract per-channel spatial mean so the CAM shows WHERE each channel
        # is more active than its own average, removing static corner/padding biases
        # from EfficientNet zero-padding and from dataset-level positional patterns
        # (e.g. Android status bar always present at top-right of every screenshot).
        A_centered = A - A.mean(dim=(1, 2), keepdim=True)
        return _norm_np(torch.relu((wc.view(-1, 1, 1) * A_centered).sum(dim=0)).numpy())

    # Layer 1: head-weight CAM on features[-1]  (7×7)
    cam1 = _head_weight_cam(acts_last[0], w)

    # Layer 2: head-weight CAM on features[5]   (14×14, finer detail)
    cam3 = _head_weight_cam(acts_mid1[0], w)

    def _up(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (out_size, out_size))

    cam = 0.8 * _up(cam1) + 0.2 * _up(cam3)

    return _postprocess_cam(
        cam,
        out_size=out_size,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
        clip_percentile=clip_percentile,
        edge_taper=edge_taper,
        edge_power=edge_power,
        edge_margin=edge_margin,
    )


def compute_layercam(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    out_size: int = 224,
    target_layers: tuple[int, ...] = (5, 3),
    blend: tuple[float, ...] = (0.6, 0.4),
    smooth_ksize: int = 3,
    smooth_sigma: float = 0.0,
    clip_percentile: tuple[float, float] | None = (20.0, 99.0),
) -> np.ndarray:
    """LayerCAM (Jiang et al., TIP 2021) for EfficientNet-B4 based models.

    Unlike the 7×7 head-weight `compute_cam`, LayerCAM uses **real per-element
    gradients** ∂score/∂A as spatial weights, so it can localize on high-resolution
    intermediate layers without relying on the (mismatched) quality-head weights.
    This yields sharp maps that track actual UI elements instead of blurry blobs.

    Default blends two layers, both gradient-weighted (hence both valid):
      features[5]  14×14  (semantic, weight 0.6)
      features[3]  28×28  (fine detail, weight 0.4)

    Requires a backward pass (grad-enabled); frozen backbone params are fine —
    activation gradients still flow through the visual head → score path.

    Returns
    -------
    cam : float32 ndarray [out_size, out_size] in [0, 1]
    """
    import cv2

    bb = _get_backbone(model)
    acts: dict[int, torch.Tensor] = {}
    handles = []
    for li in target_layers:
        def _mk(layer_idx):
            def _h(_, __, out):
                out.retain_grad()
                acts[layer_idx] = out
            return _h
        handles.append(bb.features[li].register_forward_hook(_mk(li)))

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    x = img_t.unsqueeze(0).to(device).requires_grad_(True)
    score = model(x)
    if isinstance(score, tuple):
        score = score[0]
    score.sum().backward()
    for h in handles:
        h.remove()

    cam_acc = np.zeros((out_size, out_size), dtype=np.float32)
    for li, wgt in zip(target_layers, blend):
        A = acts[li].detach().squeeze(0)          # [C, H, W]
        G = acts[li].grad.detach().squeeze(0)     # [C, H, W]
        # LayerCAM: positive gradient as per-element channel weight
        cam = torch.relu((torch.relu(G) * A).sum(dim=0)).cpu().numpy()  # [H, W]
        cam_acc += float(wgt) * cv2.resize(_norm_np(cam), (out_size, out_size))

    if was_training:
        model.train()

    return _postprocess_cam(
        cam_acc,
        out_size=out_size,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
        clip_percentile=clip_percentile,
        edge_taper=0.0,
        edge_power=1.0,
        edge_margin=0.0,
    )


def compute_eigencam(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    device: torch.device,
    out_size: int = 224,
    smooth_ksize: int = 7,
    smooth_sigma: float = 0.0,
    clip_percentile: tuple[float, float] | None = (5.0, 95.0),
    edge_taper: float = 0.45,
    edge_power: float = 2.0,
    edge_margin: float = 0.06,
) -> np.ndarray:
    """Eigen-CAM using PCA of activation maps (no gradients required)."""
    bb = _get_backbone(model)

    acts_last: list = [None]

    def _h_last(_, __, out):
        acts_last[0] = out.detach().cpu()

    h1 = bb.features[-1].register_forward_hook(_h_last)

    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)
    h1.remove()

    A = acts_last[0].squeeze(0).numpy()  # [C, H, W]
    A_centered = A - A.mean(axis=(1, 2), keepdims=True)
    A_flat = A_centered.reshape(A_centered.shape[0], -1)
    try:
        U, _, _ = np.linalg.svd(A_flat, full_matrices=False)
        w = U[:, 0]
    except np.linalg.LinAlgError:
        w = np.ones(A_flat.shape[0], dtype=A_flat.dtype)

    cam = np.tensordot(w, A_centered, axes=(0, 0))  # [H, W], sign is arbitrary

    # Orient the principal component so the highlighted region aligns with where
    # the backbone actually fires (per-location activation energy). PCA sign is
    # undefined, so without this the map flips randomly between images — causing
    # "all-blue" heatmaps on low-variance screenshots (ReLU kills the wrong half)
    # and inconsistent attention across samples.
    energy = (A_centered ** 2).sum(axis=0)  # [H, W] >= 0
    f = cam.ravel() - cam.mean()
    e = energy.ravel() - energy.mean()
    if float(np.dot(f, e)) < 0:
        cam = -cam

    cam = np.maximum(cam, 0.0)
    cam = _norm_np(cam)

    return _postprocess_cam(
        cam,
        out_size=out_size,
        smooth_ksize=smooth_ksize,
        smooth_sigma=smooth_sigma,
        clip_percentile=clip_percentile,
        edge_taper=edge_taper,
        edge_power=edge_power,
        edge_margin=edge_margin,
    )


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
