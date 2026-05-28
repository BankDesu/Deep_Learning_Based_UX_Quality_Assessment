"""
Gradio web demo for UX Quality Assessment.

Upload any mobile screenshot → get:
  - Predicted quality score (0-100%)
  - Quality tier label (Poor / Fair / Good / Excellent)
  - Grad-CAM attention heatmap
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

class _DemoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
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


_model: _DemoModel | None = None
_device: torch.device | None = None


def _get_model(ckpt_path: Path) -> tuple[_DemoModel, torch.device]:
    global _model, _device
    if _model is not None:
        return _model, _device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _DemoModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _model, _device = model, device
    return model, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB"))


# ── Multi-scale CAM ───────────────────────────────────────────────────────────

def _norm(t: torch.Tensor) -> torch.Tensor:
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def _compute_gradcam(model, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    """Multi-scale CAM: combines three layers for richer spatial coverage.
      - features[-1] (7x7)  : head-weight CAM — semantic contribution per region
      - features[-2] (7x7)  : activation energy — where the network fires strongly
      - features[6]  (14x14): mid-level detail — finer spatial structure
    """
    acts_last = [None]
    acts_mid2 = [None]
    acts_mid1 = [None]

    def _h_last(_, __, out): acts_last[0] = out.detach()
    def _h_mid2(_, __, out): acts_mid2[0] = out.detach()
    def _h_mid1(_, __, out): acts_mid1[0] = out.detach()

    h1 = model.backbone.features[-1].register_forward_hook(_h_last)
    h2 = model.backbone.features[-2].register_forward_hook(_h_mid2)
    h3 = model.backbone.features[5].register_forward_hook(_h_mid1)  # 14x14

    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)
    h1.remove(); h2.remove(); h3.remove()

    # --- Layer 1: head-weight CAM (semantic, 7x7) ---
    W_out    = model.head[3].weight          # [1, 256]
    W_hidden = model.head[0].weight          # [256, feat_dim]
    w = (W_out @ W_hidden).squeeze(0)        # [feat_dim]
    A_last = acts_last[0].squeeze(0)         # [feat_dim, 7, 7]
    cam1 = _norm((w.view(-1, 1, 1) * A_last).sum(dim=0))  # [7, 7]

    # --- Layer 2: activation energy from features[-2] (7x7) ---
    A_mid2 = acts_mid2[0].squeeze(0)         # [C, 7, 7]
    cam2 = _norm((A_mid2 ** 2).mean(dim=0))  # [7, 7]

    # --- Layer 3: activation energy from features[6] (14x14) ---
    A_mid1 = acts_mid1[0].squeeze(0)         # [C, 14, 14]
    cam3 = _norm((A_mid1 ** 2).mean(dim=0))  # [14, 14]

    # Upsample all to 224x224 and blend
    def _up(t):
        return cv2.resize(t.detach().cpu().numpy(), (224, 224))

    cam = 0.2 * _up(cam1) + 0.3 * _up(cam2) + 0.5 * _up(cam3)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


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
    img_np = np.array(pil_img.resize((224, 224))) / 255.0
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_np);  axes[0].axis("off"); axes[0].set_title("Original", fontsize=11)
    axes[1].imshow(overlay); axes[1].axis("off"); axes[1].set_title("Grad-CAM Attention", fontsize=11)
    fig.tight_layout(pad=1.0)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return buf


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
    candidates = list(Path("checkpoints").glob("baselines/efficientnet_b4/best.pt"))
    if not candidates:
        raise FileNotFoundError("Checkpoint not found. Run train_baselines.py first.")
    return candidates[0]


CKPT: Path = None  # set at startup


def predict_quality(image: Image.Image):
    if image is None:
        return None, "Upload a screenshot to get started.", None

    model, device = _get_model(CKPT)
    img_t = _preprocess(image)

    # Score
    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x).item()

    # Tier
    tier_label, tier_color = "Unknown", "#95a5a6"
    for lo, hi, label, color in QUALITY_TIERS:
        if lo <= score < hi or (score >= 0.70 and hi == 1.00):
            tier_label, tier_color = label, color
            break

    score_pct = score * 100
    score_md = (
        f"## Quality Score: **{score_pct:.1f}%**\n\n"
        f"<span style='font-size:1.3em; color:{tier_color}'>● {tier_label}</span>\n\n"
        f"| | |\n|---|---|\n"
        f"| Poor | 0–30% |\n"
        f"| Fair | 30–50% |\n"
        f"| Good | 50–70% |\n"
        f"| **Excellent** | 70–100% |"
    )

    # Grad-CAM
    cam = _compute_gradcam(model, img_t, device)
    gradcam_img = _plot_gradcam(image.resize((224, 224)), cam)

    # Structural features
    feats = _compute_struct(img_t)
    struct_img = _plot_struct(feats)

    return gradcam_img, score_md, struct_img


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> "gr.Blocks":
    import gradio as gr

    with gr.Blocks(title="UX Quality Assessment") as demo:
        gr.Markdown(
            """
            # UX Quality Assessment
            **Deep Learning-Based UX Quality Prediction from Mobile Screenshots**

            Upload a mobile app screenshot to receive an automated quality score
            based on a model trained on human expert ratings (UICrit dataset).
            Model: EfficientNet-B4 · Kendall τ = 0.201 · 95% CI [0.068, 0.321]
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
                    with gr.Tab("Attention Map"):
                        gradcam_out = gr.Image(
                            label="What the model focuses on",
                            show_label=False,
                        )
                    with gr.Tab("Structural Features"):
                        struct_out = gr.Image(
                            label="19 pixel-derived layout features",
                            show_label=False,
                        )

        btn.click(
            fn=predict_quality,
            inputs=[inp],
            outputs=[gradcam_out, score_out, struct_out],
        )

        gr.Markdown(
            """
            ---
            **How to interpret:**
            - **Attention Map** — warmer colors (red/yellow) show regions that contribute most to the quality score (multi-scale: semantic + spatial detail)
            - **Structural Features** — blue bars ≥ 0.5 indicate well-designed layout properties
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
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
