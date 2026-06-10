"""
Multi-architecture baseline comparison for UX quality prediction.

Trains a frozen-backbone + MLP head (linear probe) on multiple ImageNet-
pretrained architectures and prints a ranked comparison table.  Each run
also evaluates on the test split so results are directly reportable.

Architectures supported
-----------------------
  resnet50, efficientnet_b0, efficientnet_b4, efficientnet_v2_s,
  convnextv2_base, vit_b_16, swin_t,
  dinov2_vits14, dinov2_vitb14          ← new: self-supervised DINOv2

Usage
-----
  # Single architecture
  python scripts/train_baselines.py --arch dinov2_vits14

  # All architectures sequentially (for paper Table 2)
  python scripts/train_baselines.py --arch all --epochs 20

  # Custom subset
  python scripts/train_baselines.py --arch efficientnet_b4 dinov2_vits14

  # Disable strong augmentation (not recommended for small datasets)
  python scripts/train_baselines.py --arch all --no-strong-aug
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import train_transforms, val_transforms
from uxqa.utils.metrics import (
    kendall_tau,
    spearman_rho,
    pearson_r,
    mean_absolute_error,
    pairwise_ranking_loss,
    soft_rank_loss,
    bootstrap_ci,
)

ALL_ARCHS = [
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "convnextv2_base",
    "vit_b_16",
    "swin_t",
    "dinov2_vits14",
    "dinov2_vitb14",
]

_ARCH_META: dict[str, int] = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "efficientnet_b4": 1792,
    "efficientnet_v2_s": 1280,
    "convnextv2_base": 1024,
    "vit_b_16": 768,
    "swin_t": 768,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


def _build_backbone(arch: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (backbone with classifier removed, feature_dim). Backbone fully frozen."""
    if arch == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

    elif arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()

    elif arch == "efficientnet_b4":
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        m = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()

    elif arch == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()

    elif arch == "convnextv2_base":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "convnextv2_base requires timm. Install with: pip install timm"
            ) from exc
        m = timm.create_model("convnextv2_base", pretrained=pretrained, num_classes=0)
        feat_dim = m.num_features

    elif arch == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.heads.head.in_features
        m.heads = nn.Identity()

    elif arch == "swin_t":
        from torchvision.models import swin_t, Swin_T_Weights
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.head.in_features
        m.head = nn.Identity()

    elif arch == "dinov2_vits14":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("dinov2_vits14 requires timm. pip install timm") from exc
        m = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        feat_dim = m.num_features  # 384

    elif arch == "dinov2_vitb14":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("dinov2_vitb14 requires timm. pip install timm") from exc
        m = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        feat_dim = m.num_features  # 768

    else:
        raise ValueError(f"Unknown arch: {arch}. Choose from {ALL_ARCHS}")

    for p in m.parameters():
        p.requires_grad = False
    return m, feat_dim


class LinearProbeModel(nn.Module):
    def __init__(self, arch: str, dropout: float = 0.5, pretrained: bool = True) -> None:
        super().__init__()
        self.arch = arch
        self.backbone, feat_dim = _build_backbone(arch, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
        return torch.sigmoid(self.head(feat))


def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1)
            preds.append(model(x).cpu())
            targs.append(y)
    p = torch.cat(preds)
    t = torch.cat(targs)
    tau, tau_lo, tau_hi = bootstrap_ci(p, t, kendall_tau)
    return {
        "tau": tau, "tau_lo": tau_lo, "tau_hi": tau_hi,
        "rho": spearman_rho(p, t),
        "r":   pearson_r(p, t),
        "mae": mean_absolute_error(p, t),
        "pred_std": p.std().item(),
    }


def train_one(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    head_lr: float,
    weight_decay: float,
    ux_weight: float,
    rank_weight: float,
    rank_margin: float,
    spearman_weight: float,
    patience: int,
    dropout: float,
    ckpt_root: Path,
    pretrained: bool,
) -> dict:
    print(f"\n{'='*72}")
    print(f"  Arch: {arch}  (linear probe, frozen backbone + MLP head)")
    print(f"{'='*72}")

    model = LinearProbeModel(arch, dropout=dropout, pretrained=pretrained).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"  trainable params (head only): {n_head:,}")

    optim = AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-7)
    mse = nn.MSELoss()

    ckpt_dir = ckpt_root / arch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "train_log.csv"
    csv_fields = ["epoch", "train_loss", "val_tau", "val_rho", "val_mae", "pred_std", "is_best"]
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_w.writeheader()

    best_tau = -1.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1).to(device)
            pred = model(x)
            loss = (
                ux_weight    * mse(pred, y)
                + rank_weight    * pairwise_ranking_loss(pred, y, margin=rank_margin)
                + spearman_weight * soft_rank_loss(pred, y)
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        scheduler.step()

        val_m = _evaluate(model, val_loader, device)
        tau = val_m["tau"]
        is_best = tau > best_tau
        if is_best:
            best_tau = tau
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "tau": tau,
                        "arch": arch, "pretrained": pretrained},
                       ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        n = len(train_loader)
        csv_w.writerow({"epoch": epoch, "train_loss": f"{total_loss/n:.4f}",
                        "val_tau": f"{tau:.4f}", "val_rho": f"{val_m['rho']:.4f}",
                        "val_mae": f"{val_m['mae']:.4f}", "pred_std": f"{val_m['pred_std']:.4f}",
                        "is_best": is_best})
        csv_f.flush()

        marker = " *" if is_best else ""
        print(f"  ep {epoch:3d}/{epochs}  loss={total_loss/n:.4f}  "
              f"tau={tau:.4f}  rho={val_m['rho']:.4f}  MAE={val_m['mae']:.4f}{marker}")

        if patience > 0 and patience_counter >= patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    csv_f.close()
    print(f"  Done — best val tau={best_tau:.4f}  time={_fmt(time.time()-t0)}")

    # Test evaluation using best checkpoint
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = _evaluate(model, test_loader, device)
    print(f"  Test  tau={test_m['tau']:.4f} [{test_m['tau_lo']:.4f}, {test_m['tau_hi']:.4f}]  "
          f"rho={test_m['rho']:.4f}  MAE={test_m['mae']:.4f}")

    return {"arch": arch, "best_val_tau": best_tau, **{f"test_{k}": v for k, v in test_m.items()}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-architecture linear probe baselines")
    p.add_argument("--arch", nargs="+", default=["all"],
                   help="Architecture(s) to train. Use 'all' for all supported architectures.")
    p.add_argument("--uicrit", default="data/raw/uicrit")
    p.add_argument("--rico", default="data/raw/rico")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--ux-weight", type=float, default=0.1)
    p.add_argument("--rank-weight", type=float, default=5.0)
    p.add_argument("--rank-margin", type=float, default=0.15)
    p.add_argument("--spearman-weight", type=float, default=1.0,
                   help="Weight for soft Spearman rank loss (0 to disable)")
    p.add_argument("--no-strong-aug", action="store_true",
                   help="Use basic augmentation instead of strong (not recommended for small datasets)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--ckpt-root", default="checkpoints/baselines",
                   help="Root directory for baseline checkpoints")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Initialize backbone randomly. Use only for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    archs = ALL_ARCHS if args.arch == ["all"] else args.arch
    for a in archs:
        if a not in ALL_ARCHS:
            raise ValueError(f"Unknown arch '{a}'. Supported: {ALL_ARCHS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = args.image_size

    from uxqa.data.transforms import train_transforms_strong
    aug = train_transforms(image_size) if args.no_strong_aug else train_transforms_strong(image_size)
    if not args.no_strong_aug:
        print("  Using strong augmentation (--no-strong-aug to disable)")

    train_ds = UICritDataset(args.uicrit, args.rico, split="train",
                             transform=aug,
                             image_size=(image_size, image_size))
    val_ds   = UICritDataset(args.uicrit, args.rico, split="val",
                             transform=val_transforms(image_size),
                             image_size=(image_size, image_size))
    test_ds  = UICritDataset(args.uicrit, args.rico, split="test",
                             transform=val_transforms(image_size),
                             image_size=(image_size, image_size))
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, **kw)

    ckpt_root = Path(args.ckpt_root)
    results = []
    for arch in archs:
        r = train_one(
            arch=arch,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            ux_weight=args.ux_weight,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            spearman_weight=args.spearman_weight,
            patience=args.patience,
            dropout=args.dropout,
            ckpt_root=ckpt_root,
            pretrained=not args.no_pretrained,
        )
        results.append(r)

    # Summary table
    results.sort(key=lambda x: x["test_tau"], reverse=True)
    print(f"\n{'='*72}")
    print("  BASELINE COMPARISON TABLE (ranked by test Kendall tau)")
    print(f"{'='*72}")
    print(f"  {'Architecture':<20}  {'Val tau':>7}  {'Test tau':>8}  {'95% CI':>16}  {'rho':>7}  {'MAE':>7}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*16}  {'-'*7}  {'-'*7}")
    for r in results:
        ci = f"[{r['test_tau_lo']:.3f},{r['test_tau_hi']:.3f}]"
        print(f"  {r['arch']:<20}  {r['best_val_tau']:>7.4f}  {r['test_tau']:>7.4f}  {ci:>16}  "
              f"{r['test_rho']:>7.4f}  {r['test_mae']:>7.4f}")

    # Save summary CSV
    summary_path = ckpt_root / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
