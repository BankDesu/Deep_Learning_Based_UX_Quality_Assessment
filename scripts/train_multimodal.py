"""
Train MultiModalQualityModel (Visual + Structural gated fusion) on UICrit.

Main model for IEEE Access:
  Visual Branch:     DINOv2 ViT-S/14 (default) or any supported backbone
  Structural Branch: 19 pixel-derived layout features → MLP head
  Fusion:            Learned per-sample gate α
                     score = sigmoid(α·logit_v + (1−α)·logit_s)

Modes
-----
  probe    — backbone fully frozen; train heads + gate only
  finetune — unfreeze last N backbone blocks + full training

Ablation
--------
  full         both branches (default)
  visual-only  gate forced to α=1  → visual branch only
  struct-only  gate forced to α=0  → structural branch only

Usage
-----
  python scripts/train_multimodal.py --mode probe
  python scripts/train_multimodal.py --mode finetune --backbone dinov2_vitb14
  python scripts/train_multimodal.py --mode probe --ablation visual-only
  python scripts/train_multimodal.py --mode probe --ablation struct-only
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import sys

try:
    import pandas as _pd  # noqa: F401  — avoid PyArrow DLL conflict on Windows
except ImportError:
    pass

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
from uxqa.data.transforms import train_transforms, train_transforms_strong, val_transforms
from uxqa.models.multimodal_quality_model import MultiModalQualityModel
from uxqa.utils.metrics import (
    kendall_tau, spearman_rho, pearson_r, mean_absolute_error,
    pairwise_ranking_loss, soft_rank_loss, bootstrap_ci,
)


def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1)
            preds.append(model(x).cpu())
            targs.append(y)
    return torch.cat(preds), torch.cat(targs)


def _make_ablation_forward(model: MultiModalQualityModel, ablation: str):
    """Wrap model.forward to force gate to 1 (visual-only) or 0 (struct-only)."""
    if ablation == "visual-only":
        def _fwd(img, return_gate=False):
            feat_v  = model.visual_backbone(img)
            logit_v = model.visual_head(feat_v)
            score   = torch.sigmoid(logit_v)
            return (score, torch.ones(img.shape[0], 1, device=img.device)) if return_gate else score
        return _fwd

    if ablation == "struct-only":
        def _fwd(img, return_gate=False):
            img_01   = model._denorm(img)
            with torch.no_grad():
                from uxqa.models.structural_encoder import compute_structural_features
                struct_feats = compute_structural_features(img_01)
            logit_s = model.struct_head(struct_feats)
            score   = torch.sigmoid(logit_s)
            return (score, torch.zeros(img.shape[0], 1, device=img.device)) if return_gate else score
        return _fwd

    return None  # full: use original forward


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Visual+Structural MultiModal model on UICrit"
    )
    p.add_argument("--backbone",  default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "efficientnet_b4", "efficientnet_v2_s", "convnextv2_tiny"],
                   help="Visual backbone (default: dinov2_vits14)")
    p.add_argument("--mode",      default="probe", choices=["probe", "finetune"])
    p.add_argument("--ablation",  default="full",
                   choices=["full", "visual-only", "struct-only"])
    p.add_argument("--uicrit",   default="data/raw/uicrit")
    p.add_argument("--rico",     default="data/raw/rico")
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--head-lr",      type=float, default=1e-3)
    p.add_argument("--backbone-lr",  type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--unfreeze-last-n", type=int, default=3)
    p.add_argument("--ux-weight",    type=float, default=0.1)
    p.add_argument("--rank-weight",  type=float, default=5.0)
    p.add_argument("--rank-margin",  type=float, default=0.15)
    p.add_argument("--spearman-weight", type=float, default=1.0,
                   help="Weight for soft Spearman loss (0 to disable)")
    p.add_argument("--no-strong-aug", action="store_true",
                   help="Use basic augmentation (not recommended for 800-sample dataset)")
    p.add_argument("--patience",     type=int, default=7)
    p.add_argument("--image-size",   type=int, default=224)
    p.add_argument("--ckpt-dir",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"multimodal-{args.backbone}-{args.mode}"
    if args.ablation != "full":
        tag += f"-{args.ablation}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("checkpoints") / f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  backbone={args.backbone}  mode={args.mode}  ablation={args.ablation}")
    print(f"epochs={args.epochs}  batch={args.batch}  patience={args.patience}")

    aug = train_transforms(args.image_size) if args.no_strong_aug else train_transforms_strong(args.image_size)
    if not args.no_strong_aug:
        print("  augmentation: strong (--no-strong-aug to use basic)")

    nw = 0 if sys.platform == "win32" else 4
    kw = dict(num_workers=nw, pin_memory=True)
    train_ds = UICritDataset(args.uicrit, args.rico, split="train",
                             transform=aug, image_size=(args.image_size, args.image_size))
    val_ds   = UICritDataset(args.uicrit, args.rico, split="val",
                             transform=val_transforms(args.image_size),
                             image_size=(args.image_size, args.image_size))
    test_ds  = UICritDataset(args.uicrit, args.rico, split="test",
                             transform=val_transforms(args.image_size),
                             image_size=(args.image_size, args.image_size))
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, **kw)

    model = MultiModalQualityModel(
        backbone=args.backbone,
        mode=args.mode,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)

    ablation_fwd = _make_ablation_forward(model, args.ablation)
    if ablation_fwd is not None:
        model.forward = ablation_fwd
        print(f"  [ablation] {args.ablation}: gate overridden")

    n_vis = sum(p.numel() for p in model.visual_params())
    n_nv  = sum(p.numel() for p in model.non_visual_params())
    print(f"trainable: visual={n_vis:,}  non-visual={n_nv:,}")

    param_groups = [{"params": model.non_visual_params(), "lr": args.head_lr}]
    if args.mode == "finetune" and model.visual_params():
        param_groups.append({"params": model.visual_params(), "lr": args.backbone_lr})

    optim     = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    mse_fn    = nn.MSELoss()

    csv_path = ckpt_dir / "train_log.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "epoch", "train_loss", "val_tau", "val_rho", "val_mae", "is_best"])
    csv_w.writeheader()

    best_tau = -1.0
    patience_counter = 0
    t0 = time.time()
    print(f"\n{'-'*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1).to(device)
            pred = model(x)
            loss = (
                args.ux_weight       * mse_fn(pred, y)
                + args.rank_weight   * pairwise_ranking_loss(pred, y, margin=args.rank_margin)
                + args.spearman_weight * soft_rank_loss(pred, y)
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        scheduler.step()

        p_val, t_val = _evaluate(model, val_loader, device)
        tau = kendall_tau(p_val, t_val)
        rho = spearman_rho(p_val, t_val)
        mae = mean_absolute_error(p_val, t_val)

        is_best = tau > best_tau
        if is_best:
            best_tau = tau
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "tau": tau,
                 "backbone": args.backbone, "mode": args.mode, "ablation": args.ablation},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        n = len(train_loader)
        csv_w.writerow({"epoch": epoch, "train_loss": f"{total_loss/n:.4f}",
                        "val_tau": f"{tau:.4f}", "val_rho": f"{rho:.4f}",
                        "val_mae": f"{mae:.4f}", "is_best": is_best})
        csv_f.flush()

        marker = " *" if is_best else ""
        print(f"ep {epoch:3d}/{args.epochs}  loss={total_loss/n:.4f}  "
              f"tau={tau:.4f}  rho={rho:.4f}  MAE={mae:.4f}{marker}")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stop at epoch {epoch}.")
            break

    csv_f.close()
    print(f"\n{'-'*60}")
    print(f"Done — best val tau={best_tau:.4f}  total={_fmt(time.time()-t0)}")

    # Test evaluation
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    p_test, t_test = _evaluate(model, test_loader, device)
    tau_t, tau_lo, tau_hi = bootstrap_ci(p_test, t_test, kendall_tau)
    rho_t = spearman_rho(p_test, t_test)
    mae_t = mean_absolute_error(p_test, t_test)

    print(f"\nTEST RESULTS  backbone={args.backbone}  mode={args.mode}  ablation={args.ablation}")
    print(f"  tau = {tau_t:.4f}  95% CI [{tau_lo:.4f}, {tau_hi:.4f}]")
    print(f"  rho = {rho_t:.4f}")
    print(f"  MAE = {mae_t:.4f}")
    print(f"\nCheckpoint saved: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
