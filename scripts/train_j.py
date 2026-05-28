"""
Recipe J: Swin-T + MLP head + partial unfreeze.

Architecture
------------
  screenshot [B,3,224,224]
    -> Swin-T (early stages frozen, tail trainable)
    -> norm -> global avg pool -> [B, 768]
    -> Linear(768, 256) -> GELU -> Dropout
    -> Linear(256, 1) -> sigmoid -> quality_score [B, 1]

Default flags match J4 (20 epochs, unfreeze_from=7, dropout=0.5, wd=0.1).
Override with CLI args for other experiments.

Usage
-----
  python scripts/train_j.py                          # J4 defaults
  python scripts/train_j.py --epochs 10 --dropout 0.2 --unfreeze-from 5
  python scripts/train_j.py --strong-aug --patience 5
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import swin_t, Swin_T_Weights

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import train_transforms, train_transforms_strong, val_transforms
from uxqa.utils.metrics import (
    kendall_tau,
    spearman_rho,
    pearson_r,
    mean_absolute_error,
    pairwise_ranking_loss,
)


class SwinMLP(nn.Module):
    def __init__(self, unfreeze_from: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.features = swin.features
        self.norm = swin.norm

        for i, stage in enumerate(self.features):
            req = i >= unfreeze_from
            for p in stage.parameters():
                p.requires_grad = req
        for p in self.norm.parameters():
            p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def backbone_params(self):
        return [p for p in list(self.features.parameters()) + list(self.norm.parameters()) if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.norm(feat)
        feat = feat.mean(dim=(1, 2))
        return torch.sigmoid(self.head(feat))


def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recipe J: Swin-T + MLP head")
    p.add_argument("--uicrit", default="data/raw/uicrit")
    p.add_argument("--rico", default="data/raw/rico")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-from", type=int, default=7,
                   help="Unfreeze Swin stages >= this index (0=all, 7=last stage only)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--ux-weight", type=float, default=0.1, help="MSE loss weight")
    p.add_argument("--rank-weight", type=float, default=5.0, help="Ranking loss weight")
    p.add_argument("--rank-margin", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience on val tau (0=disabled)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--strong-aug", action="store_true",
                   help="Use stronger data augmentation (recommended for small datasets)")
    p.add_argument("--ckpt-dir", default=None,
                   help="Checkpoint directory (auto-timestamped if not set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("checkpoints") / f"swin-j-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}  epochs={args.epochs}  batch={args.batch}  "
        f"unfreeze_from={args.unfreeze_from}  dropout={args.dropout}  "
        f"wd={args.weight_decay}  patience={args.patience}  "
        f"strong_aug={args.strong_aug}"
    )

    aug = train_transforms_strong(args.image_size) if args.strong_aug else train_transforms(args.image_size)
    train_ds = UICritDataset(args.uicrit, args.rico, split="train",
                             transform=aug,
                             image_size=(args.image_size, args.image_size))
    val_ds = UICritDataset(args.uicrit, args.rico, split="val",
                           transform=val_transforms(args.image_size),
                           image_size=(args.image_size, args.image_size))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = SwinMLP(unfreeze_from=args.unfreeze_from, dropout=args.dropout).to(device)
    n_bb = sum(p.numel() for p in model.backbone_params())
    n_hd = sum(p.numel() for p in model.head_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable: backbone={n_bb:,}  head={n_hd:,}  (total={n_total:,})")

    optim = AdamW(
        [
            {"params": model.backbone_params(), "lr": args.backbone_lr},
            {"params": model.head_params(),     "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    mse = nn.MSELoss()

    csv_path = ckpt_dir / "train_log.csv"
    csv_fields = ["epoch", "train_loss", "train_q", "train_r",
                  "val_tau", "val_rho", "val_r", "val_mae", "pred_std", "is_best"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    best_tau = -1.0
    patience_counter = 0
    train_start = time.time()

    print(f"\n{'-'*72}")
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        model.train()
        total_loss = total_q = total_r = 0.0
        for batch_data in train_loader:
            x = batch_data["screenshot"].to(device)
            y = batch_data["quality_score"].unsqueeze(1).to(device)
            pred = model(x)
            q_loss = mse(pred, y)
            r_loss = pairwise_ranking_loss(pred, y, margin=args.rank_margin)
            loss = args.ux_weight * q_loss + args.rank_weight * r_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
            total_q += q_loss.item()
            total_r += r_loss.item()
        scheduler.step()
        n = len(train_loader)

        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data["screenshot"].to(device)
                y = batch_data["quality_score"].unsqueeze(1)
                preds.append(model(x).cpu())
                targs.append(y)
        preds_t = torch.cat(preds)
        targs_t = torch.cat(targs)
        tau = kendall_tau(preds_t, targs_t)
        rho = spearman_rho(preds_t, targs_t)
        pr  = pearson_r(preds_t, targs_t)
        mae = mean_absolute_error(preds_t, targs_t)
        pred_std = preds_t.std().item()

        is_best = tau > best_tau
        if is_best:
            best_tau = tau
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "tau": tau, "args": vars(args)},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        writer.writerow({
            "epoch": epoch,
            "train_loss": f"{total_loss/n:.4f}",
            "train_q": f"{total_q/n:.4f}",
            "train_r": f"{total_r/n:.4f}",
            "val_tau": f"{tau:.4f}",
            "val_rho": f"{rho:.4f}",
            "val_r": f"{pr:.4f}",
            "val_mae": f"{mae:.4f}",
            "pred_std": f"{pred_std:.4f}",
            "is_best": is_best,
        })
        csv_file.flush()

        elapsed = _fmt(time.time() - train_start)
        ep_time = _fmt(time.time() - ep_start)
        print(f"Epoch {epoch:3d}/{args.epochs}  elapsed={elapsed}  ep_time={ep_time}")
        print(f"  train | loss={total_loss/n:.4f}  quality={total_q/n:.4f}  ranking={total_r/n:.4f}")
        print(f"  val   | tau={tau:.4f}  rho={rho:.4f}  r={pr:.4f}  MAE={mae:.4f}  pred_std={pred_std:.4f}")
        if is_best:
            print(f"  * best  tau={best_tau:.4f}")
        print(f"{'-'*72}")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stop — no tau improvement for {args.patience} consecutive epochs.")
            break

    csv_file.close()
    print(f"\nDone — best val tau={best_tau:.4f}  total={_fmt(time.time()-train_start)}")
    print(f"Checkpoint : {ckpt_dir / 'best.pt'}")
    print(f"Train log  : {csv_path}")


if __name__ == "__main__":
    main()
