from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetSpec:
    key: str
    task: str
    dataset_name: str
    raw_root: str
    image_dir: str
    split_files: dict[str, str]
    annotation_dir: str | None = None
    fixation_dir: str | None = None
    saliency_map_dir: str | None = None
    label_dir: str | None = None


def _to_spec(key: str, raw: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        key=key,
        task=raw["task"],
        dataset_name=raw["dataset_name"],
        raw_root=raw["raw_root"],
        image_dir=raw["image_dir"],
        split_files=raw.get("split_files", {}),
        annotation_dir=raw.get("annotation_dir"),
        fixation_dir=raw.get("fixation_dir"),
        saliency_map_dir=raw.get("saliency_map_dir"),
        label_dir=raw.get("label_dir"),
    )


def load_dataset_specs(path: str | Path) -> dict[str, DatasetSpec]:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets", {})
    return {key: _to_spec(key, value) for key, value in datasets.items()}
