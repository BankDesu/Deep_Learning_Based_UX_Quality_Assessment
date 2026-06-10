"""
Demo script for UX Quality Assessment model.

Generates 4 visualizations saved as PNG files ready for screen recording:
  1. demo_ranking.png    — test UIs ranked by predicted quality vs human rating
    2. demo_new_ui.png     — predict score on a custom screenshot (--image)
    3. demo_gradcam.png    — CAM heatmap showing model focus regions
  4. demo_structural.png — 19 structural feature breakdown as bar chart

Usage
-----
  python scripts/demo.py
  python scripts/demo.py --image path/to/screenshot.jpg
  python scripts/demo.py --n 8 --out demo_output/
"""
from __future__ import annotations

import argparse
import random
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
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import val_transforms
from uxqa.utils.metrics import kendall_tau
from uxqa.utils.pixel_rules import PIXEL_RULE_NAMES
from uxqa.models.structural_encoder import compute_structural_features
from uxqa.utils.visualizer import compute_cam, compute_layercam

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

STRUCT_FEATURE_NAMES = (
    PIXEL_RULE_NAMES
    + ["color_diversity", "saturation_mean", "saturation_std", "vert_symmetry"]
    + [f"grid_{r}{c}" for r in "ABC" for c in "123"]
    + ["lum_std_global"]
)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load MultiModalQualityModel (gate fusion) or legacy linear probe checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "backbone" in ckpt:
        # New: MultiModalQualityModel with learned gate fusion
        sys.path.insert(0, str(SRC))
        from uxqa.models.multimodal_quality_model import MultiModalQualityModel
        model = MultiModalQualityModel(
            backbone=ckpt["backbone"],
            mode="probe",
            pretrained=False,
        )
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        print(f"  Model: MultiModalQualityModel (backbone={ckpt['backbone']}, "
              f"ablation={ckpt.get('ablation','full')})")
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
        model.to(device).eval()
        print(f"  Model: EfficientNet-B4 linear probe (legacy checkpoint)")
    return model


def predict(model, img_tensor: torch.Tensor, device: torch.device) -> tuple[float, float]:
    """Returns (score, gate_alpha) for a single image tensor [3,H,W] normalized.

    gate_alpha is the learned gate value α ∈ (0,1):
      α ≈ 1 → model relies on visual features
      α ≈ 0 → model relies on structural features
    Returns 0.0 for legacy checkpoints that don't support return_gate.
    """
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            score, alpha = model(x, return_gate=True)
            return score.item(), alpha.item()
        except TypeError:
            score = model(x)
            return score.item(), 0.0


# Thin wrapper around the shared compute_cam utility (head-weight CAM)
# so demo.py and app.py use identical logic. Head-weight CAM projects the
# quality head's weights onto backbone activations — it highlights regions
# that actually drive the quality score, unlike unsupervised Eigen-CAM (PCA)
# which produced diffuse/empty maps on many UI screenshots.

class ActivationCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __call__(self, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        # LayerCAM at high-res layers (14×14 + 28×28) with real gradients —
        # sharp, element-aligned maps instead of the blurry 7×7 head-weight CAM.
        return compute_layercam(
            self.model,
            img_tensor,
            device,
            target_layers=(5, 3),
            blend=(0.6, 0.4),
            smooth_ksize=3,
            clip_percentile=(20.0, 99.0),
        )


# ── Visualization helpers ─────────────────────────────────────────────────────

def _load_pil(path: Path, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("RGB")
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


def _load_pil_original(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _letterbox_params(w: int, h: int, size: int = 224) -> tuple[int, int, int, int]:
    """Returns (nw, nh, left, top) — the valid content region in the letterboxed canvas."""
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    left = (size - nw) // 2
    top = (size - nh) // 2
    return nw, nh, left, top


def _find_rico_image(rico_id: str) -> Path | None:
    for p in [
        Path("data/raw/rico/combined") / f"{rico_id}.jpg",
        Path("data/raw/rico/combined_224") / f"{rico_id}.jpg",
    ]:
        if p.exists():
            return p
    return None


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = T.ToTensor()(img)
    return T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t)


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = (t * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()



# ── Demo 1: Ranking ───────────────────────────────────────────────────────────

def demo_ranking(model, test_ds, device, n: int, out_dir: Path) -> None:
    print(f"[1/4] Generating ranking demo (n={n})...")
    indices = list(range(min(n, len(test_ds))))
    items = []
    for i in indices:
        item = test_ds[i]
        score, _ = predict(model, item["screenshot"], device)
        items.append({
            "img": item["screenshot"],
            "pred": score,
            "human": item["quality_score"].item(),
            "rico_id": test_ds._items[i]["rico_id"],
        })

    items.sort(key=lambda x: x["pred"], reverse=True)

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 4.2))
    axes = np.array(axes).flatten()
    fig.suptitle("UI Quality Ranking — Predicted vs Human Rating", fontsize=14, fontweight="bold", y=1.01)

    for rank, (ax, item) in enumerate(zip(axes, items)):
        ax.imshow(_denorm(item["img"]))
        ax.axis("off")
        pred_pct = item["pred"] * 100
        human_pct = item["human"] * 100
        match = abs(item["pred"] - item["human"]) < 0.15
        color = "#2ecc71" if match else "#e74c3c"
        ax.set_title(
            f"#{rank+1}  Pred: {pred_pct:.0f}%\nHuman: {human_pct:.0f}%",
            fontsize=9, color=color, pad=4
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for ax in axes[len(items):]:
        ax.axis("off")

    green = mpatches.Patch(color="#2ecc71", label="|pred - human| < 15%")
    red   = mpatches.Patch(color="#e74c3c", label="|pred - human| >= 15%")
    fig.legend(handles=[green, red], loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = out_dir / "demo_ranking.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ── Demo 2: New UI ────────────────────────────────────────────────────────────

def demo_new_ui(model, img_path: Path | None, test_ds, device, out_dir: Path) -> None:
    print("[2/4] Generating new-UI demo...")
    if img_path and img_path.exists():
        pil = _load_pil(img_path)
        title = f"Custom image: {img_path.name}"
        human_str = "N/A (not in dataset)"
    else:
        # Use a test sample not in the ranking demo
        idx = min(len(test_ds) - 1, 6)
        item = test_ds[idx]
        img_t = item["screenshot"]
        human_val = item["quality_score"].item()
        pil = _load_pil(
            Path("data/raw/rico/combined_224") / f"{test_ds._items[idx]['rico_id']}.jpg"
            if (Path("data/raw/rico/combined_224")).exists() else
            Path("data/raw/rico/combined") / f"{test_ds._items[idx]['rico_id']}.jpg"
        )
        title = f"RICO ID: {test_ds._items[idx]['rico_id']}"
        human_str = f"{human_val*100:.0f}%"

    img_t = _to_tensor(pil)
    score, delta = predict(model, img_t, device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax.imshow(np.array(pil))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)

    gate_str = f"{delta:.3f}" if delta > 0 else "N/A (legacy)"
    textstr = (
        f"Predicted Quality:  {score*100:.1f}%\n"
        f"Gate α (visual weight):  {gate_str}\n"
        f"Human Rating:  {human_str}"
    )
    props = dict(boxstyle="round", facecolor="#2c3e50", alpha=0.85)
    fig.text(0.5, 0.02, textstr, transform=fig.transFigure,
             fontsize=10, color="white", verticalalignment="bottom",
             horizontalalignment="center", bbox=props)

    plt.tight_layout()
    path = out_dir / "demo_new_ui.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ── Demo 3: CAM ───────────────────────────────────────────────────────────────

def demo_gradcam(model, test_ds, device, out_dir: Path) -> None:
    print("[3/4] Generating CAM demo...")
    # Pick 2 high-quality and 2 low-quality from test set (randomized)
    scored = []
    for i in range(min(50, len(test_ds))):
        item = test_ds[i]
        s, _ = predict(model, item["screenshot"], device)
        scored.append((s, i))
    scored.sort()
    k = 2
    band = max(k, len(scored) // 4)
    low_pool = scored[:band]
    high_pool = scored[-band:]
    low_idxs = [idx for _, idx in random.sample(low_pool, k)]
    high_idxs = [idx for _, idx in random.sample(high_pool, k)]
    selected  = low_idxs + high_idxs
    labels    = ["Low Quality", "Low Quality", "High Quality", "High Quality"]

    grad_cam = ActivationCAM(model)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.suptitle("Grad-CAM Explanation: Model Focus Regions for UI Quality Prediction", fontsize=13, fontweight="bold")

    for col, (idx, label) in enumerate(zip(selected, labels)):
        item = test_ds[idx]
        img_t = item["screenshot"]
        human = item["quality_score"].item()
        rico_id = test_ds._items[idx]["rico_id"]
        img_path = _find_rico_image(rico_id)

        model.zero_grad()
        cam = grad_cam(img_t, device)
        img_np = _denorm(img_t)
        if img_path is not None:
            img_vis = _load_pil_original(img_path)
            orig_w, orig_h = img_vis.size
            img_vis_np = np.array(img_vis) / 255.0
            # Crop valid content region from CAM (strips letterbox padding)
            nw, nh, left, top = _letterbox_params(orig_w, orig_h, size=224)
            cam_valid = cam[top:top + nh, left:left + nw]
            cam_resized = cv2.resize(cam_valid, (orig_w, orig_h))
        else:
            img_vis_np = img_np
            cam_resized = cam  # already 224×224, no crop needed

        axes[0, col].imshow(img_vis_np)
        axes[0, col].axis("off")
        pred_s, _ = predict(model, img_t, device)
        color = "#27ae60" if "High" in label else "#c0392b"
        axes[0, col].set_title(f"{label}\nPred:{pred_s*100:.0f}% | Human:{human*100:.0f}%",
                                fontsize=9, color=color)

        axes[1, col].imshow(cam_resized, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1, col].axis("off")
        axes[1, col].set_title("Heatmap", fontsize=8, color="gray")

        axes[2, col].imshow(img_vis_np)
        axes[2, col].imshow(cam_resized, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[2, col].contour(cam_resized, levels=[0.6], colors="white", linewidths=1)
        axes[2, col].axis("off")
        axes[2, col].set_title("Overlay", fontsize=8, color="gray")

    plt.tight_layout()
    path = out_dir / "demo_gradcam.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ── Demo 4: Structural Features ───────────────────────────────────────────────

def demo_structural(test_ds, device, out_dir: Path) -> None:
    print("[4/4] Generating structural features demo...")
    # Compare high vs low quality UI
    scored = []
    for i in range(min(50, len(test_ds))):
        item = test_ds[i]
        scored.append((item["quality_score"].item(), i))
    scored.sort()
    low_idx  = scored[0][1]
    high_idx = scored[-1][1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Structural Feature Comparison: High vs Low Quality UI", fontsize=13, fontweight="bold")

    for ax, idx, label, color in zip(
        axes,
        [low_idx, high_idx],
        ["Low Quality UI", "High Quality UI"],
        ["#e74c3c", "#2ecc71"],
    ):
        item = test_ds[idx]
        img_t = item["screenshot"]
        human = item["quality_score"].item()

        mean = img_t.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = img_t.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_01 = (img_t * std + mean).clamp(0, 1).unsqueeze(0)

        with torch.no_grad():
            feats = compute_structural_features(img_01).squeeze(0).cpu().numpy()

        names = STRUCT_FEATURE_NAMES[:len(feats)]
        y_pos = np.arange(len(feats))
        bars = ax.barh(y_pos, feats, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Feature Value [0–1]", fontsize=9)
        ax.set_title(f"{label}\nHuman Rating: {human*100:.0f}%", fontsize=10, color=color, fontweight="bold")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "demo_structural.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default=None,
                   help="Model checkpoint path (default: auto-detect best)")
    p.add_argument("--uicrit",  default="data/raw/uicrit")
    p.add_argument("--rico",    default="data/raw/rico")
    p.add_argument("--image",   default=None,
                   help="Custom screenshot for demo_new_ui (optional)")
    p.add_argument("--n",       type=int, default=8,
                   help="Number of UIs to show in ranking demo")
    p.add_argument("--out",     default="demo_output",
                   help="Output directory for PNG files")
    return p.parse_args()


def _find_best_ckpt() -> Path | None:
    import torch as _torch

    def _is_full_gate(p: Path) -> bool:
        """Return True only for full gate fusion checkpoints (not ablations)."""
        try:
            c = _torch.load(p, map_location="cpu", weights_only=False)
            return c.get("backbone") is not None and c.get("ablation", "full") == "full"
        except Exception:
            return False

    # Priority 1: MultiModalQualityModel probe, full gate fusion (τ=0.2227)
    for p in sorted(Path("checkpoints").glob("multimodal-efficientnet_b4-probe-*/best.pt"),
                    reverse=True):
        if p.exists() and _is_full_gate(p):
            return p
    # Priority 2: multimodal finetune, full gate fusion
    for p in sorted(Path("checkpoints").glob("multimodal-efficientnet_b4-finetune-*/best.pt"),
                    reverse=True):
        if p.exists() and _is_full_gate(p):
            return p
    # Priority 3: legacy EfficientNet-B4 linear probe
    for p in [
        Path("checkpoints/baselines-effb4-original/efficientnet_b4/best.pt"),
        Path("checkpoints/baselines/efficientnet_b4/best.pt"),
    ]:
        if p.exists():
            return p
    return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = Path(args.ckpt) if args.ckpt else _find_best_ckpt()
    if ckpt is None or not ckpt.exists():
        print("ERROR: No checkpoint found. Run train_multimodal.py first, or specify --ckpt")
        sys.exit(1)
    print(f"Checkpoint: {ckpt}")

    model = load_model(ckpt, device)

    test_ds = UICritDataset(
        args.uicrit, args.rico, split="test",
        transform=val_transforms(224),
        image_size=(224, 224),
    )
    print(f"Test set: {len(test_ds)} samples\n")

    demo_ranking(model, test_ds, device, n=args.n, out_dir=out_dir)
    demo_new_ui(model, Path(args.image) if args.image else None, test_ds, device, out_dir)
    demo_gradcam(model, test_ds, device, out_dir)
    demo_structural(test_ds, device, out_dir)

    print(f"\nAll done! Output files in: {out_dir}/")
    print("  demo_ranking.png    — UI ranking grid")
    print("  demo_new_ui.png     — single UI prediction")
    print("  demo_gradcam.png    — Grad-CAM explanation maps")
    print("  demo_structural.png — structural feature comparison")
    print("\nScreen record these 4 images for your demo video.")


if __name__ == "__main__":
    main()
