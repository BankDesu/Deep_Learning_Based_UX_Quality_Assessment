"""
Minimal sanity baseline: Swin-T (ImageNet1k, frozen) + Linear regression head.

Purpose
-------
Diagnose whether the data has a meaningful learnable signal at all.
If even this stripped-down model achieves τ ~ 0.07 (matching the full UX model),
the bottleneck is the data, not the architecture.

Architecture
------------
  screenshot [B,3,224,224]
    → Swin-T (frozen, ImageNet pretrained)
    → global avg pool → [B, 768]
    → Linear(768, 1) → sigmoid → quality_score [B, 1]
"""
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import swin_t, Swin_T_Weights

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
)


class FrozenSwinLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.features = swin.features
        self.norm = swin.norm
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        self.head = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)        # [B, H', W', 768]
            feat = self.norm(feat)
            feat = feat.mean(dim=(1, 2))   # global avg pool → [B, 768]
        return torch.sigmoid(self.head(feat))  # [B, 1]


def _fmt(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def main(
    uicrit: str = "data/raw/uicrit",
    rico: str = "data/raw/rico",
    epochs: int = 10,
    batch: int = 8,
    lr: float = 1e-3,
    ckpt_dir: str = "checkpoints/baseline",
    image_size: int = 224,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  epochs={epochs}  batch={batch}  lr={lr}")

    train_ds = UICritDataset(
        uicrit, rico, split="train",
        transform=train_transforms(image_size),
        image_size=(image_size, image_size),
    )
    val_ds = UICritDataset(
        uicrit, rico, split="val",
        transform=val_transforms(image_size),
        image_size=(image_size, image_size),
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    model = FrozenSwinLinear().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable={n_train}  total={n_total}")

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    mse = nn.MSELoss()

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_tau = -1.0
    train_start = time.time()

    print(f"\n{'-'*72}")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        total_loss = total_q = total_r = 0.0
        for batch_data in train_loader:
            x = batch_data["screenshot"].to(device)
            y = batch_data["quality_score"].unsqueeze(1).to(device)
            pred = model(x)
            q_loss = mse(pred, y)
            r_loss = pairwise_ranking_loss(pred, y)
            loss = 0.1 * q_loss + 5.0 * r_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_q += q_loss.item()
            total_r += r_loss.item()
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
        pr = pearson_r(preds_t, targs_t)
        mae = mean_absolute_error(preds_t, targs_t)
        pred_std = preds_t.std().item()

        is_best = tau > best_tau
        if is_best:
            best_tau = tau
            torch.save({"epoch": epoch, "model": model.state_dict(), "tau": tau}, Path(ckpt_dir) / "best.pt")

        elapsed = _fmt(time.time() - train_start)
        ep_time = _fmt(time.time() - epoch_start)
        print(f"Epoch {epoch:3d}/{epochs}  elapsed={elapsed}  ep_time={ep_time}")
        print(f"  train │ loss={total_loss/n:.4f}  quality={total_q/n:.4f}  ranking={total_r/n:.4f}")
        print(f"  val   │ tau={tau:.4f}  rho={rho:.4f}  r={pr:.4f}  MAE={mae:.4f}  pred_std={pred_std:.4f}")
        if is_best:
            print(f"  * best (tau={best_tau:.4f})")
        print(f"{'-'*72}")

    print(f"\nDone — best val Kendall's Tau = {best_tau:.4f}  total={_fmt(time.time()-train_start)}")


if __name__ == "__main__":
    main()
