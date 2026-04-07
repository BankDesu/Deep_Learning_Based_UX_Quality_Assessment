import csv
import json
from pathlib import Path

from .dataset_config import DatasetSpec


def _parse_meta(row: dict[str, str]) -> dict[str, object]:
    raw_meta = row.get("meta", "")
    if not raw_meta:
        return {}
    try:
        parsed = json.loads(raw_meta)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"raw_meta": raw_meta}


def to_unified_record(row: dict[str, str], spec: DatasetSpec, split: str) -> dict[str, object]:
    meta = _parse_meta(row)
    return {
        "id": row.get("id", ""),
        "source_dataset": spec.dataset_name,
        "task": spec.task,
        "split": split,
        "image_path": row.get("image_path", ""),
        "label_path": row.get("label_path", ""),
        "boxes": meta.get("boxes", []),
        "classes": meta.get("classes", []),
        "ux_score": meta.get("ux_score"),
        "aesthetic_score": meta.get("aesthetic_score"),
        "saliency_map_path": meta.get("saliency_map_path"),
        "fixation_path": meta.get("fixation_path"),
        "extra": meta.get("extra", {}),
    }


def convert_manifest_to_unified_jsonl(
    input_csv: Path,
    output_jsonl: Path,
    spec: DatasetSpec,
    split: str,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_csv.open("r", encoding="utf-8", newline="") as src, output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        for row in reader:
            unified = to_unified_record(row, spec=spec, split=split)
            dst.write(json.dumps(unified, ensure_ascii=True) + "\n")
            count += 1
    return count
