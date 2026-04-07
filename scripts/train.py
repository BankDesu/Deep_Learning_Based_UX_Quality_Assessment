from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW
import wandb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.data import load_dataset_specs
from uxqa.models import UXAssessmentModel


def _compute_multitask_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    criterion: nn.Module,
    ux_w: float,
    layout_w: float,
    attention_w: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ux_loss = criterion(out["ux_score"], targets["ux_score"])
    layout_loss = criterion(out["layout_quality"], targets["layout_quality"])
    attention_loss = criterion(out["attention_alignment"], targets["attention_alignment"])
    total = (ux_w * ux_loss) + (layout_w * layout_loss) + (attention_w * attention_loss)
    return total, {
        "ux": ux_loss.detach(),
        "layout": layout_loss.detach(),
        "attention": attention_loss.detach(),
    }


def _resolve_training_dataset(datasets_path: str) -> None:
    specs = load_dataset_specs(Path(datasets_path))
    target_key = "ux_score_training"
    if target_key not in specs:
        raise ValueError(f"Dataset key '{target_key}' not found in {datasets_path}")

    spec = specs[target_key]
    train_manifest = spec.split_files.get("train", "")
    print(
        " ".join(
            [
                f"dataset={spec.dataset_name}",
                f"task={spec.task}",
                f"image_dir={spec.image_dir}",
                f"train_manifest={train_manifest}",
            ]
        )
    )


def run_train(
    config_path: str = "configs/default.yaml",
    datasets_path: str = "configs/datasets.yaml",
) -> None:
    cfg = load_config(Path(config_path))
    _resolve_training_dataset(datasets_path)
    model = UXAssessmentModel(cfg.model)
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.MSELoss()

    run = None
    if cfg.train.use_wandb:
        run = wandb.init(
            # Set the wandb project where this run will be logged.
            project=cfg.train.wandb_project,
            config={
                "learning_rate": cfg.train.learning_rate,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "ux_loss_weight": cfg.train.ux_loss_weight,
                "layout_loss_weight": cfg.train.layout_loss_weight,
                "attention_loss_weight": cfg.train.attention_loss_weight,
                "detector_backend": cfg.model.detector_backend,
            },
        )

    model.train()
    for epoch in range(cfg.train.epochs):
        x = torch.randn(cfg.train.batch_size, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
        targets = {
            "ux_score": torch.rand(cfg.train.batch_size, 1),
            "layout_quality": torch.rand(cfg.train.batch_size, 1),
            "attention_alignment": torch.rand(cfg.train.batch_size, 1),
        }

        out = model(x)
        loss, losses = _compute_multitask_loss(
            out=out,
            targets=targets,
            criterion=criterion,
            ux_w=cfg.train.ux_loss_weight,
            layout_w=cfg.train.layout_loss_weight,
            attention_w=cfg.train.attention_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "total_loss": loss.item(),
                    "ux_loss": losses["ux"].item(),
                    "layout_loss": losses["layout"].item(),
                    "attention_loss": losses["attention"].item(),
                }
            )
        print(
            " ".join(
                [
                    f"epoch={epoch + 1}",
                    f"total_loss={loss.item():.4f}",
                    f"ux_loss={losses['ux'].item():.4f}",
                    f"layout_loss={losses['layout'].item():.4f}",
                    f"attention_loss={losses['attention'].item():.4f}",
                ]
            )
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    run_train()
