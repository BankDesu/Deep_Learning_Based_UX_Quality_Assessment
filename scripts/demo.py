"""
Demo script for UX Quality Assessment model.

Generates 4 visualizations saved as PNG files ready for screen recording:
  1. demo_ranking.png    — test UIs ranked by predicted quality vs human rating
  2. demo_new_ui.png     — predict score on a custom screenshot (--image)
  3. demo_gradcam.png    — Grad-CAM heatmap showing model focus regions
  4. demo_structural.png — 19 structural feature breakdown as bar chart

Usage
-----
  python scripts/demo.py
  python scripts/demo.py --image path/to/screenshot.jpg
  python scripts/demo.py --n 8 --out demo_output/
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
import matplotlib.patches as mpatches
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

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

STRUCT_FEATURE_NAMES = (
    PIXEL_RULE_NAMES
    + ["color_diversity", "saturation_mean", "saturation_std", "vert_symmetry"]
    + [f"grid_{r}{c}" for r in "ABC" for c in "123"]
    + ["lum_std_global"]
)


# ── Demo model (mirrors baseline architecture: backbone + head) ───────────────

class _DemoModel(torch.nn.Module):
    """Minimal wrapper matching the baseline checkpoint structure (backbone + head)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.backbone(x)))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    model = _DemoModel().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict(model, img_tensor: torch.Tensor, device: torch.device) -> tuple[float, float]:
    """Returns (score, delta=0) for a single image tensor [3,H,W] normalized."""
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x)
    return score.item(), 0.0


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook on last conv block of EfficientNet-B4 (features[-1])
        target = model.backbone.features[-1]
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        x = img_tensor.unsqueeze(0).to(device)
        self.model.zero_grad()
        score = self.model(x)
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [H, W]
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Visualization helpers ─────────────────────────────────────────────────────

def _load_pil(path: Path, size: int = 224) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)


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

    textstr = (
        f"Predicted Quality:  {score*100:.1f}%\n"
        f"Structural Correction:  {delta:+.4f}\n"
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


# ── Demo 3: Grad-CAM ──────────────────────────────────────────────────────────

def demo_gradcam(model, test_ds, device, out_dir: Path) -> None:
    print("[3/4] Generating Grad-CAM demo...")
    # Pick 2 high-quality and 2 low-quality from test set
    scored = []
    for i in range(min(50, len(test_ds))):
        item = test_ds[i]
        s, _ = predict(model, item["screenshot"], device)
        scored.append((s, i))
    scored.sort()
    low_idxs  = [scored[0][1], scored[1][1]]
    high_idxs = [scored[-1][1], scored[-2][1]]
    selected  = low_idxs + high_idxs
    labels    = ["Low Quality", "Low Quality", "High Quality", "High Quality"]

    grad_cam = GradCAM(model)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Grad-CAM: Model Attention on UI Quality Prediction", fontsize=13, fontweight="bold")

    for col, (idx, label) in enumerate(zip(selected, labels)):
        item = test_ds[idx]
        img_t = item["screenshot"]
        human = item["quality_score"].item()

        model.zero_grad()
        cam = grad_cam(img_t, device)
        img_np = _denorm(img_t)

        axes[0, col].imshow(img_np)
        axes[0, col].axis("off")
        pred_s, _ = predict(model, img_t, device)
        color = "#27ae60" if "High" in label else "#c0392b"
        axes[0, col].set_title(f"{label}\nPred:{pred_s*100:.0f}% | Human:{human*100:.0f}%",
                                fontsize=9, color=color)

        import cv2
        cam_resized = cv2.resize(cam, (224, 224))
        axes[1, col].imshow(img_np)
        axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[1, col].axis("off")
        axes[1, col].set_title("Attention Map", fontsize=8, color="gray")

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
    candidates = []
    # Best: baseline EffNet-B4 (τ=0.201)
    for p in sorted(Path("checkpoints").glob("baselines/efficientnet_b4/best.pt")):
        candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
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
    print("  demo_gradcam.png    — Grad-CAM attention maps")
    print("  demo_structural.png — structural feature comparison")
    print("\nScreen record these 4 images for your demo video.")


if __name__ == "__main__":
    main()
