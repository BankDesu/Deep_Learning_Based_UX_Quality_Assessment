from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.config import load_config
from uxqa.data.uicrit_dataset import UICritDataset
from uxqa.data.transforms import val_transforms
from uxqa.models import UXAssessmentModel
from uxqa.models.rules.definitions import RULE_NAMES
from uxqa.utils.metrics import (
    kendall_tau, spearman_rho, pearson_r, mean_absolute_error, rule_f1,
)


def run_eval(
    checkpoint: str = "checkpoints/best.pt",
    config_path: str = "configs/default.yaml",
    uicrit_root: str = "data/raw/uicrit",
    rico_root: str = "data/raw/rico",
) -> None:
    cfg = load_config(Path(config_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = UXAssessmentModel(cfg.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}  "
        f"val_tau={ckpt.get('tau', float('nan')):.4f}"
    )

    test_ds = UICritDataset(
        uicrit_root=uicrit_root,
        rico_root=rico_root,
        split="test",
        transform=val_transforms(cfg.model.image_size),
        image_size=(cfg.model.image_size, cfg.model.image_size),
    )
    loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2
    )
    print(f"Test set: {len(test_ds)} samples")

    preds_list, targets_list, rule_sc_list, rule_lb_list = [], [], [], []

    with torch.no_grad():
        for batch in loader:
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

    tau = kendall_tau(preds_t, targets_t)
    rho = spearman_rho(preds_t, targets_t)
    r   = pearson_r(preds_t, targets_t)
    mae = mean_absolute_error(preds_t, targets_t)
    f1  = rule_f1(rule_sc, rule_lb, rule_names=RULE_NAMES)

    print("\n── Quality Score Correlation (Test) ──────────────────────")
    print(f"  Kendall's Tau τ  : {tau:.4f}   (target > 0.30)")
    print(f"  Spearman's ρ     : {rho:.4f}")
    print(f"  Pearson r        : {r:.4f}")
    print(f"  MAE              : {mae:.4f}")
    print("\n── Per-Rule F1 (Test) ────────────────────────────────────")
    print(f"  Macro Precision  : {f1['macro_precision']:.4f}")
    print(f"  Macro Recall     : {f1['macro_recall']:.4f}")
    print(f"  Macro F1         : {f1['macro_f1']:.4f}")
    for name in RULE_NAMES:
        print(f"  {name:<20} F1={f1[f'f1_{name}']:.4f}")


if __name__ == "__main__":
    run_eval()
