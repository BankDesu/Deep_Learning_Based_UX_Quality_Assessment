"""
Train MultiModalQualityModel (Visual + Structural gated fusion) on UICrit.

Novel architecture for IEEE Access:
  Visual Branch:     EfficientNet-B4 (ImageNet) frozen/partial-unfreeze
  Structural Branch: PixelStructureEncoder (19 pixel-derived layout features)
  Fusion:            Learned gated fusion → quality score

Modes
-----
  probe    — freeze backbone; train structural encoder + fusion + head
  finetune — unfreeze last 3 backbone blocks + full training

Usage
-----
  python scripts/train_multimodal.py --mode probe
  python scripts/train_multimodal.py --mode finetune
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
    pairwise_ranking_loss, bootstrap_ci,
)


def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def _evaluate(model, loader, device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["screenshot"].to(device)
            y = batch["quality_score"].unsqueeze(1)
            preds.append(model(x).cpu())
            targs.append(y)
    return torch.cat(preds), torch.cat(targs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Visual+Structural MultiModal model on UICrit"
    )
    p.add_argument("--mode",     default="probe", choices=["probe", "finetune"])
    p.add_argument("--ablation", default="full",
                   choices=["full", "visual-only", "struct-only"],
                   help="Ablation: full=both branches, visual-only, struct-only")
    p.add_argument("--uicrit",  default="data/raw/uicrit")
    p.add_argument("--rico",    default="data/raw/rico")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--head-lr",     type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay",type=float, default=0.1)
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--unfreeze-last-n", type=int, default=3)
    p.add_argument("--ux-weight",   type=float, default=0.1)
    p.add_argument("--rank-weight", type=float, default=5.0)
    p.add_argument("--rank-margin", type=float, default=0.15)
    p.add_argument("--patience",    type=int, default=7)
    p.add_argument("--image-size",  type=int, default=224)
    p.add_argument("--strong-aug",  action="store_true")
    p.add_argument("--ckpt-dir",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"multimodal-{args.mode}"
    if args.ablation != "full":
        tag += f"-{args.ablation}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("checkpoints") / f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  mode={args.mode}  ablation={args.ablation}")
    print(f"epochs={args.epochs}  batch={args.batch}  patience={args.patience}")

    aug = train_transforms_strong(args.image_size) if args.strong_aug else train_transforms(args.image_size)
    train_ds = UICritDataset(args.uicrit, args.rico, split="train",
                             transform=aug, image_size=(args.image_size, args.image_size))
    val_ds   = UICritDataset(args.uicrit, args.rico, split="val",
                             transform=val_transforms(args.image_size),
                             image_size=(args.image_size, args.image_size))
    test_ds  = UICritDataset(args.uicrit, args.rico, split="test",
                             transform=val_transforms(args.image_size),
                             image_size=(args.image_size, args.image_size))
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    nw = 0 if sys.platform == "win32" else 4
    kw = dict(num_workers=nw, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, **kw)

    model = MultiModalQualityModel(
        mode=args.mode,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)

    # Ablation: zero out one branch by overriding forward
    if args.ablation == "visual-only":
        _orig_fwd = model.forward
        def _visual_only_fwd(img, return_gate=False):
            feat_v = model.visual_backbone(img)
            # Route through proj_v + head without structural branch
            fused = model.fusion.proj_v(feat_v)
            score = torch.sigmoid(model.head(fused))
            return (score, None) if return_gate else score
        model.forward = _visual_only_fwd
        print("  [ablation] visual-only: structural branch disabled")

    elif args.ablation == "struct-only":
        _orig_fwd = model.forward
        def _struct_only_fwd(img, return_gate=False):
            mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_01 = (img * std + mean).clamp(0.0, 1.0)
            feat_s = model.struct_encoder(img_01)
            fused = model.fusion.proj_s(feat_s)
            score = torch.sigmoid(model.head(fused))
            return (score, None) if return_gate else score
        model.forward = _struct_only_fwd
        print("  [ablation] struct-only: visual backbone disabled")

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
            loss = (args.ux_weight * mse_fn(pred, y) +
                    args.rank_weight * pairwise_ranking_loss(pred, y, margin=args.rank_margin))
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
                 "mode": args.mode, "ablation": args.ablation},
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
    print(f"Done - best val tau={best_tau:.4f}  total={_fmt(time.time()-t0)}")

    # Test evaluation
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    p_test, t_test = _evaluate(model, test_loader, device)
    tau_t, tau_lo, tau_hi = bootstrap_ci(p_test, t_test, kendall_tau)
    rho_t = spearman_rho(p_test, t_test)
    mae_t = mean_absolute_error(p_test, t_test)

    print(f"\nTEST RESULTS ({args.mode}, ablation={args.ablation})")
    print(f"  tau = {tau_t:.4f}  95% CI [{tau_lo:.4f}, {tau_hi:.4f}]")
    print(f"  rho = {rho_t:.4f}")
    print(f"  MAE = {mae_t:.4f}")
    print(f"\nCheckpoint: {ckpt_dir / 'best.pt'}")
    print(f"\nAblation study commands:")
    print(f"  python scripts/train_multimodal.py --mode probe --ablation visual-only")
    print(f"  python scripts/train_multimodal.py --mode probe --ablation struct-only")


if __name__ == "__main__":
    main()
