"""
Evaluate a saved checkpoint on any data split.

Reports Kendall's τ, Spearman ρ, Pearson r, and MAE with 95% bootstrap CI.
Saves per-image predictions to a CSV for qualitative analysis.

Supports checkpoints from:
  - train_j.py (SwinMLP, arch="swin_t_mlp")
  - train_baselines.py (LinearProbeModel, arch stored in checkpoint)

Usage
-----
  python scripts/eval.py --ckpt checkpoints/swin-j-<run>/best.pt
  python scripts/eval.py --ckpt checkpoints/baselines/resnet50/best.pt
  python scripts/eval.py --ckpt <path> --split val
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import val_transforms
from uxqa.utils.metrics import (
    kendall_tau, spearman_rho, pearson_r, mean_absolute_error, bootstrap_ci,
)


# ── Model builders (standalone, no import of train scripts) ──────────────────

def _build_swin_mlp(dropout: float = 0.0) -> nn.Module:
    from torchvision.models import swin_t

    class SwinMLP(nn.Module):
        def __init__(self):
            super().__init__()
            swin = swin_t(weights=None)
            self.features = swin.features
            self.norm = swin.norm
            self.head = nn.Sequential(
                nn.Linear(768, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
            )
        def forward(self, x):
            feat = self.features(x)
            feat = self.norm(feat)
            feat = feat.mean(dim=(1, 2))
            return torch.sigmoid(self.head(feat))

    return SwinMLP()


def _build_linear_probe(arch: str, dropout: float = 0.0) -> nn.Module:
    from torchvision.models import (
        resnet50, efficientnet_b0, efficientnet_b4, efficientnet_v2_s,
        vit_b_16, swin_t,
    )

    _feat_dims = {
        "resnet50": 2048, "efficientnet_b0": 1280, "efficientnet_b4": 1792,
        "efficientnet_v2_s": 1280, "convnextv2_base": 1024,
        "vit_b_16": 768,  "swin_t": 768,
        "dinov2_vits14": 384, "dinov2_vitb14": 768,
    }

    class Probe(nn.Module):
        def __init__(self, backbone, feat_dim):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
            )
        def forward(self, x):
            with torch.no_grad():
                feat = self.backbone(x)
            return torch.sigmoid(self.head(feat))

    if arch == "resnet50":
        m = resnet50(weights=None); m.fc = nn.Identity()
    elif arch == "efficientnet_b0":
        m = efficientnet_b0(weights=None); m.classifier = nn.Identity()
    elif arch == "efficientnet_b4":
        m = efficientnet_b4(weights=None); m.classifier = nn.Identity()
    elif arch == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=None); m.classifier = nn.Identity()
    elif arch == "convnextv2_base":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "convnextv2_base requires timm. Install with: pip install timm"
            ) from exc
        m = timm.create_model("convnextv2_base", pretrained=False, num_classes=0)
    elif arch == "vit_b_16":
        m = vit_b_16(weights=None); m.heads = nn.Identity()
    elif arch == "swin_t":
        m = swin_t(weights=None); m.head = nn.Identity()
    elif arch in ("dinov2_vits14", "dinov2_vitb14"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DINOv2 requires timm. pip install timm") from exc
        tag = ("vit_small_patch14_dinov2.lvd142m" if arch == "dinov2_vits14"
               else "vit_base_patch14_dinov2.lvd142m")
        m = timm.create_model(tag, pretrained=False, num_classes=0, dynamic_img_size=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return Probe(m, _feat_dims[arch])


def load_model(ckpt_path: Path, arch_override: str | None, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = arch_override or ckpt.get("arch")

    # MultiModalQualityModel checkpoints saved by train_multimodal.py
    if "backbone" in ckpt and arch_override is None:
        from uxqa.models.multimodal_quality_model import MultiModalQualityModel
        from uxqa.models.structural_encoder import compute_structural_features
        backbone = ckpt["backbone"]
        ablation = ckpt.get("ablation", "full")
        model = MultiModalQualityModel(backbone=backbone, mode="probe", pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        # Re-apply ablation gate override for correct inference
        if ablation == "visual-only":
            def _fwd(img, return_gate=False):
                feat_v  = model.visual_backbone(img)
                score   = torch.sigmoid(model.visual_head(feat_v))
                return (score, torch.ones(img.shape[0], 1, device=img.device)) if return_gate else score
            model.forward = _fwd
        elif ablation == "struct-only":
            def _fwd(img, return_gate=False):
                img_01 = model._denorm(img)
                with torch.no_grad():
                    sf = compute_structural_features(img_01)
                score = torch.sigmoid(model.struct_head(sf))
                return (score, torch.zeros(img.shape[0], 1, device=img.device)) if return_gate else score
            model.forward = _fwd
        print(f"  backbone={backbone}  mode={ckpt.get('mode','?')}  ablation={ablation}")
        return model, ckpt

    # train_j.py checkpoints have an "args" dict but no "arch" key
    if arch is None or arch == "swin_t_mlp":
        model = _build_swin_mlp(dropout=0.0)
    else:
        model = _build_linear_probe(arch, dropout=0.0)

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a data split")
    p.add_argument("--ckpt", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--arch", default=None,
                   help="Override arch (swin_t_mlp | resnet50 | efficientnet_b0 | "
                        "efficientnet_b4 | efficientnet_v2_s | convnextv2_base | "
                        "vit_b_16 | swin_t | dinov2_vits14 | dinov2_vitb14)")
    p.add_argument("--uicrit", default="data/raw/uicrit")
    p.add_argument("--rico", default="data/raw/rico")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap iterations")
    p.add_argument("--out", default=None,
                   help="Predictions CSV path (default: <ckpt_dir>/predictions_<split>.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    model, ckpt = load_model(ckpt_path, args.arch, device)
    print(f"Checkpoint : {ckpt_path}")
    print(f"  epoch    : {ckpt.get('epoch', '?')}")
    if "tau" in ckpt:
        print(f"  val tau  : {ckpt['tau']:.4f}")

    ds = UICritDataset(args.uicrit, args.rico, split=args.split,
                       transform=val_transforms(args.image_size),
                       image_size=(args.image_size, args.image_size))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"\nEvaluating on {args.split} split ({len(ds)} samples)...")

    preds, targs, rico_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1)
            preds.append(model(x).cpu())
            targs.append(y)
            rico_ids.extend(batch.get("rico_id", ["?"] * len(y)))

    p = torch.cat(preds)
    t = torch.cat(targs)

    tau, tau_lo, tau_hi = bootstrap_ci(p, t, kendall_tau,    n_boot=args.n_boot)
    rho, rho_lo, rho_hi = bootstrap_ci(p, t, spearman_rho,   n_boot=args.n_boot)
    pr,  pr_lo,  pr_hi  = bootstrap_ci(p, t, pearson_r,      n_boot=args.n_boot)
    mae = mean_absolute_error(p, t)

    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  {'Metric':<18}  {'Value':>8}  {'95% CI':>22}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*22}")
    print(f"  {'Kendall tau':<18}  {tau:>8.4f}  [{tau_lo:.4f}, {tau_hi:.4f}]")
    print(f"  {'Spearman rho':<18}  {rho:>8.4f}  [{rho_lo:.4f}, {rho_hi:.4f}]")
    print(f"  {'Pearson r':<18}  {pr:>8.4f}  [{pr_lo:.4f}, {pr_hi:.4f}]")
    print(f"  {'MAE':<18}  {mae:>8.4f}")
    print(f"  {'pred std':<18}  {p.std().item():>8.4f}")
    print(sep)

    out_path = Path(args.out) if args.out else ckpt_path.parent / f"predictions_{args.split}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rico_id", "target", "predicted", "abs_error"])
        for rid, tgt, pred in zip(rico_ids, t.squeeze().tolist(), p.squeeze().tolist()):
            w.writerow([rid, f"{tgt:.6f}", f"{pred:.6f}", f"{abs(tgt - pred):.6f}"])
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
