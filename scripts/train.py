from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import train_transforms, val_transforms
from uxqa.models import UXAssessmentModel
from uxqa.models.rules.definitions import RULE_NAMES
from uxqa.utils.metrics import kendall_tau, spearman_rho, pearson_r, mean_absolute_error, rule_f1


def _fmt_seconds(secs: float) -> str:
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _compute_loss(
    out: dict,
    targets: dict,
    quality_criterion: nn.Module,
    rule_criterion: nn.Module,
    ux_w: float,
    rule_w: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    q_loss = quality_criterion(out["quality_score"], targets["quality_score"])

    # rule_scores: 1 = good UX;  rule_labels: 1 = violated
    pred_viol = (1.0 - out["rule_scores"]).clamp(1e-7, 1 - 1e-7)
    r_loss = rule_criterion(pred_viol, targets["rule_labels"])

    total = ux_w * q_loss + rule_w * r_loss
    return total, {"quality": q_loss.detach(), "rules": r_loss.detach()}


def run_train(
    config_path: str = "configs/default.yaml",
    uicrit_root: str = "data/raw/uicrit",
    rico_root: str = "data/raw/rico",
    checkpoint_dir: str = "checkpoints",
    run_name: str | None = "UXQA-Run",
) -> None:
    cfg = load_config(Path(config_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    total_epochs = cfg.train.epochs

    print(f"device={device}  epochs={total_epochs}  batch={cfg.train.batch_size}")

    # ── Datasets ───────────────────────────────────────────────────────────
    img_size = cfg.model.image_size
    train_ds = UICritDataset(
        uicrit_root=uicrit_root,
        rico_root=rico_root,
        split="train",
        transform=train_transforms(img_size),
        image_size=(img_size, img_size),
    )
    val_ds = UICritDataset(
        uicrit_root=uicrit_root,
        rico_root=rico_root,
        split="val",
        transform=val_transforms(img_size),
        image_size=(img_size, img_size),
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # ── Model & optimiser ──────────────────────────────────────────────────
    model = UXAssessmentModel(cfg.model).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    quality_criterion = nn.MSELoss()
    rule_criterion    = nn.BCELoss()

    # ── WandB ──────────────────────────────────────────────────────────────
    run = None
    if cfg.train.use_wandb:
        run = wandb.init(
            project=cfg.train.wandb_project,
            name=run_name,
            config={
                "lr":              cfg.train.learning_rate,
                "epochs":          total_epochs,
                "batch_size":      cfg.train.batch_size,
                "train_size":      len(train_ds),
                "val_size":        len(val_ds),
                "detector_backend": cfg.model.detector_backend,
                "image_size":      img_size,
                "output_dim":      cfg.model.output_dim,
            },
        )

    # ── Training loop ──────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_tau = -1.0
    epoch_times: list[float] = []
    train_start = time.time()

    print(f"\n{'─'*72}")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = total_q = total_r = 0.0

        for batch in train_loader:
            x = batch["screenshot"].to(device)
            targets = {
                "quality_score": batch["quality_score"].unsqueeze(1).to(device),
                "rule_labels":   batch["rule_labels"].to(device),
            }

            out = model(x)
            loss, losses = _compute_loss(
                out, targets, quality_criterion, rule_criterion,
                cfg.train.ux_loss_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_q    += losses["quality"].item()
            total_r    += losses["rules"].item()

        scheduler.step()
        n = len(train_loader)
        avg_loss = total_loss / n
        avg_q    = total_q / n
        avg_r    = total_r / n

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        preds_list, targets_list, rule_sc_list, rule_lb_list = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["screenshot"].to(device)
                out = model(x)
                preds_list.append(out["quality_score"].cpu())
                targets_list.append(batch["quality_score"].unsqueeze(1))
                rule_sc_list.append(out["rule_scores"].cpu())
                rule_lb_list.append(batch["rule_labels"])

        preds_t   = torch.cat(preds_list)
        targets_t = torch.cat(targets_list)
        rule_sc   = torch.cat(rule_sc_list)
        rule_lb   = torch.cat(rule_lb_list)

        tau  = kendall_tau(preds_t, targets_t)
        rho  = spearman_rho(preds_t, targets_t)
        pr   = pearson_r(preds_t, targets_t)
        mae  = mean_absolute_error(preds_t, targets_t)
        f1   = rule_f1(rule_sc, rule_lb, rule_names=RULE_NAMES)

        # ── Timing & ETA ───────────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch_time * (total_epochs - epoch)
        total_elapsed = time.time() - train_start

        is_best = tau > best_tau
        if is_best:
            best_tau = tau
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "tau": tau},
                ckpt_dir / "best.pt",
            )

        # ── Terminal output ────────────────────────────────────────────────
        progress = epoch / total_epochs
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"Epoch {epoch:3d}/{total_epochs}  [{bar}]  "
            f"elapsed={_fmt_seconds(total_elapsed)}  ETA={_fmt_seconds(remaining)}"
        )
        print(
            f"  train │ loss={avg_loss:.4f}  quality={avg_q:.4f}  rule={avg_r:.4f}"
        )
        print(
            f"  val   │ tau={tau:.4f}  rho={rho:.4f}  r={pr:.4f}  MAE={mae:.4f}"
        )
        print(
            f"  rules │ macro_F1={f1['macro_f1']:.4f}  "
            + "  ".join(f"{n[:4]}={f1[f'f1_{n}']:.3f}" for n in RULE_NAMES)
        )
        if is_best:
            print(f"  ✓ best checkpoint saved (tau={best_tau:.4f})")
        print(f"{'─'*72}")

        # ── WandB logging ──────────────────────────────────────────────────
        if run:
            log = {
                "epoch":              epoch,
                "train/loss":         avg_loss,
                "train/quality_loss": avg_q,
                "train/rule_loss":    avg_r,
                "val/kendall_tau":    tau,
                "val/spearman_rho":   rho,
                "val/pearson_r":      pr,
                "val/mae":            mae,
                "val/macro_f1":       f1["macro_f1"],
                "val/macro_precision": f1["macro_precision"],
                "val/macro_recall":   f1["macro_recall"],
                "best_tau":           best_tau,
                "epoch_time_s":       epoch_elapsed,
            }
            for name in RULE_NAMES:
                log[f"val/f1_{name}"] = f1[f"f1_{name}"]
            wandb.log(log)

    if run:
        run.finish()
    print(f"\nDone — best val Kendall's Tau = {best_tau:.4f}  total={_fmt_seconds(time.time()-train_start)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--uicrit",   default="data/raw/uicrit")
    parser.add_argument("--rico",     default="data/raw/rico")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--name",     default="UXQA-run", help="WandB run name")
    args = parser.parse_args()
    run_train(
        config_path=args.config,
        uicrit_root=args.uicrit,
        rico_root=args.rico,
        checkpoint_dir=args.ckpt_dir,
        run_name=args.name,
    )
