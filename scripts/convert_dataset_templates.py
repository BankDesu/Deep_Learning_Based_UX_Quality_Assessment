from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.data import load_dataset_specs
from uxqa.data.converters import convert_manifest_to_unified_jsonl


def run_convert(datasets_path: str = "configs/datasets.yaml") -> None:
    specs = load_dataset_specs(ROOT / datasets_path)

    total = 0
    for key, spec in specs.items():
        for split, csv_path in spec.split_files.items():
            src = ROOT / csv_path
            out = ROOT / "data" / "interim" / spec.task / f"{key}_{split}.jsonl"
            if not src.exists():
                print(f"skip missing manifest: {src}")
                continue

            count = convert_manifest_to_unified_jsonl(
                input_csv=src,
                output_jsonl=out,
                spec=spec,
                split=split,
            )
            total += count
            print(f"converted {key}:{split} -> {out} rows={count}")

    print(f"done total_rows={total}")


if __name__ == "__main__":
    run_convert()
