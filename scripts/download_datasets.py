from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _download_kaggle(dataset_slug: str, out_dir: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle CLI not found. Install with: pip3 install kaggle")

    out_dir.mkdir(parents=True, exist_ok=True)
    _run([kaggle, "datasets", "download", "-d", dataset_slug, "-p", str(out_dir), "--unzip"])


def _download_url(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1] or "dataset_archive.zip"
    target = out_dir / filename
    _run(["curl", "-L", url, "-o", str(target)])


def run_download(config_path: str = "configs/dataset_sources.yaml") -> None:
    cfg = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8")) or {}
    sources = cfg.get("sources", {})

    for key, source in sources.items():
        method = (source.get("method") or "manual").lower()
        out_dir = ROOT / source.get("output_dir", f"data/raw/{key}")

        if method == "kaggle":
            dataset_slug = (source.get("kaggle_dataset") or "").strip()
            if not dataset_slug:
                print(f"skip {key}: missing kaggle_dataset")
                continue
            try:
                _download_kaggle(dataset_slug, out_dir)
            except Exception as exc:
                print(f"failed {key}: {exc}")

        elif method == "url":
            url = (source.get("url") or "").strip()
            if not url:
                print(f"skip {key}: missing url")
                continue
            try:
                _download_url(url, out_dir)
            except Exception as exc:
                print(f"failed {key}: {exc}")

        else:
            note = source.get("note", "manual setup required")
            print(f"skip {key}: {note}")


if __name__ == "__main__":
    run_download()
