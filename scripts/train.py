from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.models import UXAssessmentModel


def run_train(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(Path(config_path))
    model = UXAssessmentModel(cfg.model)
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(cfg.train.epochs):
        x = torch.randn(cfg.train.batch_size, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
        y = torch.rand(cfg.train.batch_size, 1)

        out = model(x)
        loss = criterion(out["ux_score"], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.4f}")


if __name__ == "__main__":
    run_train()
