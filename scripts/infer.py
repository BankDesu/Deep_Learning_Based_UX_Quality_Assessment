from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.models import UXAssessmentModel
from uxqa.models.rules.definitions import RULE_NAMES


def run_infer(
    config_path: str = "configs/default.yaml",
    checkpoint: str | None = None,
) -> None:
    cfg = load_config(Path(config_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = UXAssessmentModel(cfg.model).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {checkpoint}")

    model.eval()

    x = torch.randn(1, cfg.model.in_channels, cfg.model.image_size, cfg.model.image_size).to(device)
    with torch.no_grad():
        out = model(x)

    print(f"quality_score : {out['quality_score'].squeeze().item():.4f}")
    print("rule_scores:")
    for name, score in zip(RULE_NAMES, out["rule_scores"][0].tolist()):
        flag = "✓" if score >= 0.5 else "✗"
        print(f"  {flag} {name:<20} {score:.4f}")


if __name__ == "__main__":
    run_infer()
