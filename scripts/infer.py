from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.models import UXAssessmentModel


def run_infer(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(Path(config_path))
    model = UXAssessmentModel(cfg.model)
    model.eval()

    x = torch.randn(1, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size)
    with torch.no_grad():
        out = model(x)

    print("ux_score:", out["ux_score"].squeeze().item())
    print("layout_quality:", out["layout_quality"].squeeze().item())
    print("attention_alignment:", out["attention_alignment"].squeeze().item())


if __name__ == "__main__":
    run_infer()
