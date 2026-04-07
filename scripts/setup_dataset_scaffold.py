from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    "data/raw/publaynet/images",
    "data/raw/publaynet/annotations",
    "data/raw/webui/images",
    "data/raw/webui/annotations",
    "data/raw/salicon/images",
    "data/raw/salicon/fixations",
    "data/raw/salicon/maps",
    "data/raw/wdqd/images",
    "data/raw/wdqd/labels",
    "data/raw/webpage_aesthetics/images",
    "data/raw/webpage_aesthetics/labels",
    "data/interim/layout_detection",
    "data/interim/ui_detection",
    "data/interim/attention_modeling",
    "data/interim/ux_score_training",
    "data/interim/ux_evaluation_benchmark",
    "data/processed/layout_detection",
    "data/processed/ui_detection",
    "data/processed/attention_modeling",
    "data/processed/ux_score_training",
    "data/processed/ux_evaluation_benchmark",
    "data/manifests",
]


def ensure_structure() -> None:
    for rel in REQUIRED_DIRS:
        path = ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not any(path.iterdir()):
            keep.touch(exist_ok=True)
    print("Dataset scaffold ready")


if __name__ == "__main__":
    ensure_structure()
