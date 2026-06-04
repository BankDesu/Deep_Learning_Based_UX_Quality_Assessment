"""
Gradio web demo for UX Quality Assessment.

Upload any mobile screenshot → get:
  - Predicted quality score (0-100%)
  - Quality tier label (Poor / Fair / Good / Excellent)
    - Grad-CAM explanation heatmap (post-hoc model focus)
  - 19 structural feature breakdown

Usage
-----
  python scripts/app.py
  python scripts/app.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt
  python scripts/app.py --share   # public URL for sharing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as _pd  # noqa: F401
except ImportError:
    pass

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.utils.pixel_rules import PIXEL_RULE_NAMES
from uxqa.models.structural_encoder import compute_structural_features
from uxqa.utils.visualizer import compute_layercam

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

STRUCT_NAMES = (
    PIXEL_RULE_NAMES
    + ["color_diversity", "sat_mean", "sat_std", "vert_symmetry"]
    + [f"grid_{r}{c}" for r in "top mid bot".split() for c in "L C R".split()]
    + ["lum_std"]
)

QUALITY_TIERS = [
    (0.00, 0.30, "Poor",      "#e74c3c"),
    (0.30, 0.50, "Fair",      "#e67e22"),
    (0.50, 0.70, "Good",      "#f1c40f"),
    (0.70, 1.00, "Excellent", "#2ecc71"),
]


# ── Model ────────────────────────────────────────────────────────────────────

_model = None
_device: torch.device | None = None
_model_info: dict = {}   # stores backbone name, tau, etc. for UI display


def _get_model(ckpt_path: Path) -> tuple:
    global _model, _device, _model_info
    if _model is not None:
        return _model, _device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "backbone" in ckpt and ckpt.get("ablation", "full") == "full":
        # New: MultiModalQualityModel with learned gate fusion
        from uxqa.models.multimodal_quality_model import MultiModalQualityModel
        model = MultiModalQualityModel(
            backbone=ckpt["backbone"], mode="probe", pretrained=False
        )
        model.load_state_dict(ckpt["model"])
        _model_info = {
            "name": "Learned Gate Fusion (MultiModal)",
            "backbone": ckpt["backbone"],
            "tau": 0.2227,
            "ci": "[0.092, 0.351]",
            "is_gate": True,
        }
    else:
        # Legacy: EfficientNet-B4 linear probe
        from torchvision.models import efficientnet_b4

        class _LegacyProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                base = efficientnet_b4(weights=None)
                feat_dim = base.classifier[1].in_features
                base.classifier = torch.nn.Identity()
                self.backbone = base
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(feat_dim, 256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, 1),
                )
            def forward(self, x):
                return torch.sigmoid(self.head(self.backbone(x)))

        model = _LegacyProbe()
        model.load_state_dict(ckpt["model"])
        _model_info = {
            "name": "EfficientNet-B4 linear probe (legacy)",
            "backbone": "efficientnet_b4",
            "tau": 0.215,
            "ci": "[0.088, 0.344]",
            "is_gate": False,
        }

    model.to(device).eval()
    _model, _device = model, device
    return model, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

def _letterbox(pil_img: Image.Image, size: int = 224) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas

def _letterbox_params(w: int, h: int, size: int = 224) -> tuple[int, int, int, int]:
    """Returns (nw, nh, left, top) — the valid content region in the letterboxed canvas."""
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    left = (size - nw) // 2
    top = (size - nh) // 2
    return nw, nh, left, top

def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    return _transform(_letterbox(pil_img, size=224))


# ── Multi-scale CAM ───────────────────────────────────────────────────────────



def _compute_gradcam(model, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    """Thin wrapper — delegates to shared LayerCAM utility (same recipe as paper Fig. 5).

    LayerCAM uses real per-element gradients on the 14×14 + 28×28 backbone layers,
    giving sharp, element-aligned maps instead of the blurry 7×7 head-weight CAM.
    """
    return compute_layercam(
        model,
        img_t,
        device,
        target_layers=(5, 3),
        blend=(0.6, 0.4),
        smooth_ksize=3,
        clip_percentile=(20.0, 99.0),
    )


# ── Structural features ───────────────────────────────────────────────────────

def _compute_struct(img_t: torch.Tensor) -> np.ndarray:
    mean = img_t.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = img_t.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    img_01 = (img_t * std + mean).clamp(0, 1).unsqueeze(0)
    with torch.no_grad():
        feats = compute_structural_features(img_01).squeeze(0).cpu().numpy()
    return feats


# ── Plot builders ─────────────────────────────────────────────────────────────

def _plot_gradcam(pil_img: Image.Image, cam: np.ndarray) -> np.ndarray:
    img = pil_img.convert("RGB")
    w, h = img.size
    img_np = np.array(img) / 255.0

    # Crop the valid content region from the CAM (strips letterbox padding)
    nw, nh, left, top = _letterbox_params(w, h, size=224)
    cam_valid = cam[top:top + nh, left:left + nw]
    cam_resized = cv2.resize(cam_valid, (w, h))

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 1)
    return (overlay * 255).astype(np.uint8)


def _plot_struct(feats: np.ndarray) -> np.ndarray:
    names = STRUCT_NAMES[:len(feats)]
    n = len(names)
    colors = ["#3498db" if v >= 0.5 else "#95a5a6" for v in feats]

    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.32)))
    y = np.arange(n)
    bars = ax.barh(y, feats, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Feature value [0 = low, 1 = high]", fontsize=9)
    ax.set_title("Structural Layout Features", fontsize=11, fontweight="bold")
    ax.axvline(0.5, color="#bdc3c7", linestyle="--", linewidth=0.8)
    ax.invert_yaxis()

    for bar, val in zip(bars, feats):
        ax.text(min(val + 0.02, 0.96), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5, color="#2c3e50")

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return buf


# ── Main inference function ───────────────────────────────────────────────────

def _ckpt_path() -> Path:
    # Priority 1: Learned Gate Fusion (best model, τ=0.2227)
    for p in sorted(Path("checkpoints").glob("multimodal-efficientnet_b4-probe-*/best.pt"),
                    reverse=True):
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            if c.get("backbone") and c.get("ablation", "full") == "full":
                return p
        except Exception:
            continue
    # Priority 2: finetune gate fusion
    for p in sorted(Path("checkpoints").glob("multimodal-efficientnet_b4-finetune-*/best.pt"),
                    reverse=True):
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            if c.get("backbone") and c.get("ablation", "full") == "full":
                return p
        except Exception:
            continue
    # Priority 3: legacy linear probe
    for p in [
        Path("checkpoints/baselines-effb4-original/efficientnet_b4/best.pt"),
        Path("checkpoints/baselines/efficientnet_b4/best.pt"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("No checkpoint found. Run train_multimodal.py first.")


CKPT: Path = None  # set at startup


def predict_quality(image: Image.Image):
    if image is None:
        return None, None, "Upload a screenshot to get started.", None

    model, device = _get_model(CKPT)
    img_t = _preprocess(image)

    # Score + optional gate value
    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            score_t, alpha_t = model(x, return_gate=True)
            score = score_t.item()
            alpha = alpha_t.item()
            gate_line = (
                f"\n\n**Gate α = {alpha:.3f}** "
                f"({'relies on visual' if alpha > 0.6 else 'balances both' if alpha > 0.4 else 'relies on structural'})"
            )
        except TypeError:
            score = model(x).item()
            gate_line = ""

    # Tier
    tier_label, tier_color = "Unknown", "#95a5a6"
    for lo, hi, label, color in QUALITY_TIERS:
        if lo <= score < hi or (score >= 0.70 and hi == 1.00):
            tier_label, tier_color = label, color
            break

    score_pct = score * 100
    score_md = (
        f"## Quality Score: **{score_pct:.1f}%**\n\n"
        f"<span style='font-size:1.3em; color:{tier_color}'>● {tier_label}</span>"
        f"{gate_line}\n\n"
        f"| Tier | Range |\n|---|---|\n"
        f"| Poor | 0–30% |\n"
        f"| Fair | 30–50% |\n"
        f"| Good | 50–70% |\n"
        f"| **Excellent** | 70–100% |"
    )

    # CAM + original image
    cam = _compute_gradcam(model, img_t, device)
    original_np = np.array(image.convert("RGB"))
    gradcam_img = _plot_gradcam(image, cam)

    # Structural features
    feats = _compute_struct(img_t)
    struct_img = _plot_struct(feats)

    return original_np, gradcam_img, score_md, struct_img


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> "gr.Blocks":
    import gradio as gr

    with gr.Blocks(title="UX Quality Assessment") as demo:
        info = _model_info
        gr.Markdown(
            f"""
            # UX Quality Assessment
            **Deep Learning-Based UX Quality Prediction from Mobile Screenshots**

            Upload a mobile app screenshot to receive an automated quality score
            based on a model trained on human expert ratings (UICrit dataset, UIST 2024).

            **Model:** {info.get('name','Learned Gate Fusion')} ·
            Kendall τ = {info.get('tau', 0.2227):.3f} · 95% CI {info.get('ci','[0.092, 0.351]')}
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload Screenshot", height=400)
                btn = gr.Button("Analyze", variant="primary", size="lg")
                gr.Markdown("*Drag & drop any mobile screenshot to analyze*")

            with gr.Column(scale=2):
                score_out = gr.Markdown(
                    "Upload a screenshot and click **Analyze**.",
                    elem_classes=["score-box"],
                )
                with gr.Tabs():
                    with gr.Tab("Grad-CAM Explanation"):
                        with gr.Row(equal_height=True):
                            original_out = gr.Image(
                                label="Original",
                                show_label=True,
                            )
                            gradcam_out = gr.Image(
                                label="Grad-CAM",
                                show_label=True,
                            )
                    with gr.Tab("Structural Features"):
                        struct_out = gr.Image(
                            label="19 pixel-derived layout features",
                            show_label=False,
                        )

        btn.click(
            fn=predict_quality,
            inputs=[inp],
            outputs=[original_out, gradcam_out, score_out, struct_out],
        )

        gr.Markdown(
            """
            ---
            **How to interpret:**
            - **Grad-CAM Explanation** — post-hoc interpretability map; warmer colors (red/yellow) show regions that contribute most to the quality score (head-weight CAM). Not a model input — explanation only.
            - **Structural Features** — blue bars ≥ 0.5 indicate well-designed layout properties
            - **Gate α** — learned weight (0–1): α≈1 means the model relies on visual features; α≈0 means structural features dominate
            - Score is calibrated against UICrit human ratings (1–7 scale, normalized to 0–1)
            """
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   default=None, help="Checkpoint path (auto-detected if omitted)")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true", help="Create public Gradio URL")
    return p.parse_args()


def main():
    global CKPT
    args = parse_args()

    CKPT = Path(args.ckpt) if args.ckpt else _ckpt_path()
    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}")
        sys.exit(1)

    print(f"Checkpoint : {CKPT}")
    print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Pre-load model so first inference is fast
    _get_model(CKPT)
    print("Model loaded. Starting server...\n")

    import gradio as gr
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
