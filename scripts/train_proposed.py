"""
Stage 2: Fine-tune RICO-pretrained EfficientNet-B4 on UICrit quality labels.

Loads the pretrained backbone from pretrain_rico.py, attaches a quality MLP
head, and fine-tunes with quality regression + pairwise ranking loss.

Supports two modes:
  --mode probe      Frozen backbone + MLP head only (linear probe)
  --mode finetune   Partial backbone unfreeze + MLP head (full fine-tune)

Compare against ImageNet-only baseline:
  python scripts/train_proposed.py --mode probe    # RICO pretrained probe
  python scripts/train_proposed.py --mode finetune # RICO pretrained fine-tune

Usage
-----
  python scripts/train_proposed.py --pretrain checkpoints/pretrain-rico-effb4/best.pt
  python scripts/train_proposed.py --pretrain <path> --mode finetune --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

# pandas (via pixel_rules dep chain) must load before torch/torchvision on Windows
# to avoid PyArrow DLL conflict (exit -1073741819)
try:
    import pandas as _pd  # noqa: F401
except ImportError:
    pass

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import train_transforms, train_transforms_strong, val_transforms
from uxqa.utils.metrics import (
    kendall_tau, spearman_rho, pearson_r, mean_absolute_error,
    pairwise_ranking_loss, bootstrap_ci,
)
from uxqa.utils.pixel_rules import N_PIXEL_RULES

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_backbone(pretrain_ckpt: Path | None) -> nn.Module:
    """
    Build EfficientNet-B4 backbone with classifier removed.

    If pretrain_ckpt is given: load RICO-pretrained weights from
    EfficientNetRulePredictor (backbone + rule head). Extract backbone only.
    Otherwise: use ImageNet weights as control condition.
    """
    from torchvision.models import EfficientNet_B4_Weights

    base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    feat_dim = base.classifier[1].in_features  # 1792
    base.classifier = nn.Identity()

    if pretrain_ckpt is not None and pretrain_ckpt.exists():
        # EfficientNetRulePredictor state_dict has keys: backbone.*, head.*
        # We need only backbone.*
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        state = ckpt["model"]
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = base.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"  [pretrain] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [pretrain] unexpected keys: {len(unexpected)}")
        print(f"  Loaded RICO-pretrained backbone from {pretrain_ckpt}")
    else:
        print("  Using ImageNet-only backbone (no pretrain checkpoint)")

    return base, feat_dim


class ProposedModel(nn.Module):
    """RICO-pretrained EfficientNet-B4 + quality MLP head."""

    def __init__(
        self,
        pretrain_ckpt: Path | None = None,
        mode: str = "probe",       # "probe" | "finetune"
        dropout: float = 0.5,
        unfreeze_last_n: int = 3,  # finetune mode: unfreeze last N feature blocks
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_backbone(pretrain_ckpt)

        if mode == "probe":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif mode == "finetune":
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze last N blocks of features (EfficientNet-B4 has 9 feature blocks)
            blocks = list(self.backbone.features.children())
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Always unfreeze the final conv + bn
            for p in self.backbone.features[-1].parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return torch.sigmoid(self.head(feat))


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    p = torch.cat(preds)
    t = torch.cat(targs)
    return p, t


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: Fine-tune RICO-pretrained EfficientNet-B4 on UICrit"
    )
    p.add_argument("--pretrain", default="checkpoints/pretrain-rico-effb4/best.pt",
                   help="RICO-pretrained checkpoint from pretrain_rico.py")
    p.add_argument("--mode", default="probe", choices=["probe", "finetune"],
                   help="probe=frozen backbone; finetune=partial unfreeze")
    p.add_argument("--uicrit", default="data/raw/uicrit")
    p.add_argument("--rico",   default="data/raw/rico")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--head-lr",      type=float, default=1e-3)
    p.add_argument("--backbone-lr",  type=float, default=1e-5,
                   help="Only used in finetune mode")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--unfreeze-last-n", type=int, default=3,
                   help="Finetune mode: unfreeze last N EfficientNet feature blocks")
    p.add_argument("--ux-weight",    type=float, default=0.1)
    p.add_argument("--rank-weight",  type=float, default=5.0)
    p.add_argument("--rank-margin",  type=float, default=0.15)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--image-size",   type=int, default=224)
    p.add_argument("--strong-aug",   action="store_true")
    p.add_argument("--ckpt-dir",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_ckpt = Path(args.pretrain) if args.pretrain else None
    tag = f"proposed-{args.mode}"
    from datetime import datetime
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (
        Path("checkpoints") / f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  mode={args.mode}  epochs={args.epochs}  "
          f"batch={args.batch}  patience={args.patience}")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, **kw)

    model = ProposedModel(
        pretrain_ckpt=pretrain_ckpt,
        mode=args.mode,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    n_bb = sum(p.numel() for p in model.backbone_params())
    n_hd = sum(p.numel() for p in model.head_params())
    print(f"trainable: backbone={n_bb:,}  head={n_hd:,}")

    param_groups = [{"params": model.head_params(), "lr": args.head_lr}]
    if args.mode == "finetune" and model.backbone_params():
        param_groups.append({"params": model.backbone_params(), "lr": args.backbone_lr})

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
                 "pretrain": str(pretrain_ckpt), "mode": args.mode},
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

    print(f"\nTEST RESULTS ({args.mode}, pretrain={'yes' if pretrain_ckpt and pretrain_ckpt.exists() else 'no'})")
    print(f"  tau = {tau_t:.4f}  95% CI [{tau_lo:.4f}, {tau_hi:.4f}]")
    print(f"  rho = {rho_t:.4f}")
    print(f"  MAE = {mae_t:.4f}")
    print(f"\nCheckpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
