"""
Stage 1: UI-Domain Pretraining on RICO via Pixel-Rule Pseudo-Supervision.

Trains EfficientNet-B4 (ImageNet init) to predict 5 pixel-based UX rule
scores for 66K RICO screenshots.  The backbone learns UI-specific visual
representations that transfer better to quality assessment than pure
ImageNet features.

Requires: data/raw/rico/pixel_rules.csv  (from generate_pseudo_labels.py)

Usage
-----
  python scripts/pretrain_rico.py
  python scripts/pretrain_rico.py --epochs 30 --batch 128
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

# pandas must be imported before torch/torchvision to avoid PyArrow DLL conflict on Windows
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.utils.pixel_rules import N_PIXEL_RULES, PIXEL_RULE_NAMES

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Dataset ──────────────────────────────────────────────────────────────────

class RICORuleDataset(Dataset):
    """RICO screenshots with precomputed pseudo-labels (pixel rules or CLIP scores)."""

    def __init__(
        self,
        rico_root: Path,
        labels_csv: Path,
        label_cols: list[str],
        image_size: int = 224,
        augment: bool = True,
    ) -> None:
        df = pd.read_csv(labels_csv)
        # Use pre-resized directory if available (10x faster data loading)
        presized = rico_root / f"combined_{image_size}"
        self.rico_root = presized if presized.exists() else rico_root / "combined"
        self.presized  = presized.exists()
        self.rico_ids  = df["rico_id"].astype(str).tolist()
        self.labels    = torch.tensor(
            df[label_cols].values, dtype=torch.float32
        )
        resize_tf = [] if self.presized else [T.Resize((image_size, image_size))]
        if augment:
            self.tf = T.Compose(resize_tf + [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        else:
            self.tf = T.Compose(resize_tf + [
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.rico_ids)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.rico_root / f"{self.rico_ids[idx]}.jpg"
        img = Image.open(img_path).convert("RGB")
        return {"img": self.tf(img), "label": self.labels[idx]}


# ── Model ─────────────────────────────────────────────────────────────────────

class EfficientNetRulePredictor(nn.Module):
    """EfficientNet-B4 backbone + MLP head predicting n_targets pseudo-label scores."""

    def __init__(self, n_targets: int, dropout: float = 0.3) -> None:
        super().__init__()
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features  # 1792
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_targets),
            nn.Sigmoid(),              # scores in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain EfficientNet-B4 on RICO pixel rules")
    p.add_argument("--rico",   default="data/raw/rico")
    p.add_argument("--labels", default="data/raw/rico/pixel_rules.csv",
                   help="CSV with pseudo-labels (pixel_rules.csv or clip_labels.csv)")
    p.add_argument("--label-col", default=None,
                   help="Single column name to use as label (default: all 5 pixel rule cols). "
                        "Use 'clip_quality' for CLIP-based pretraining.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch",  type=int, default=64,
                   help="batch=64 fits 12GB VRAM with AMP; batch=128 causes VRAM swap")
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--image-size",   type=int, default=224)
    p.add_argument("--val-frac",     type=float, default=0.02,
                   help="Fraction of RICO to use as pretraining validation (default 2%)")
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--ckpt-dir",     default="checkpoints/pretrain-rico-effb4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_csv = Path(args.labels)
    if not labels_csv.exists():
        print(f"ERROR: {labels_csv} not found.")
        print("Run:  python scripts/generate_pseudo_labels.py  first.")
        sys.exit(1)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve label columns
    label_cols = [args.label_col] if args.label_col else PIXEL_RULE_NAMES
    n_targets  = len(label_cols)
    print(f"device={device}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"labels={labels_csv.name}  cols={label_cols}  n_targets={n_targets}")

    rico_root = Path(args.rico)
    full_ds = RICORuleDataset(rico_root, labels_csv, label_cols, image_size=args.image_size, augment=True)
    val_ds  = RICORuleDataset(rico_root, labels_csv, label_cols, image_size=args.image_size, augment=False)

    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, _ = random_split(full_ds,  [n_train, n_val],
                               generator=torch.Generator().manual_seed(42))
    _,        val_sub = random_split(val_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_sub,  batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    presized = "yes" if full_ds.presized else "no (using originals)"
    print(f"train={n_train:,}  val={n_val:,}  pre-resized={presized}")

    model = EfficientNetRulePredictor(n_targets=n_targets).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {use_amp}")

    optim     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    mse       = nn.MSELoss()

    csv_path = ckpt_dir / "pretrain_log.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=["epoch", "train_mse", "val_mse"])
    csv_w.writeheader()

    best_val = float("inf")
    t0 = time.time()
    print(f"\n{'-'*60}")

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs   = batch["img"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(imgs)
                loss = mse(pred, labels)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["img"].to(device)
                labels = batch["label"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_loss += mse(model(imgs), labels).item()

        train_mse = total_loss / len(train_loader)
        val_mse   = val_loss   / len(val_loader)
        is_best   = val_mse < best_val
        if is_best:
            best_val = val_mse
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_mse": val_mse},
                ckpt_dir / "best.pt",
            )

        csv_w.writerow({"epoch": epoch,
                        "train_mse": f"{train_mse:.6f}",
                        "val_mse":   f"{val_mse:.6f}"})
        csv_f.flush()

        marker = " *" if is_best else ""
        ep_time = _fmt(time.time() - ep_start)
        print(f"ep {epoch:3d}/{args.epochs}  "
              f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}  "
              f"ep_time={ep_time}{marker}")

    csv_f.close()
    print(f"\nDone — best val MSE={best_val:.5f}  total={_fmt(time.time()-t0)}")
    print(f"Pretrained checkpoint: {ckpt_dir / 'best.pt'}")
    print("\nNext step:")
    print(f"  python scripts/train_proposed.py --pretrain {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
