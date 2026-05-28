"""
Generate pixel-based UX pseudo-labels for all RICO screenshots.

Computes the 5 pixel_rules scores for every JPG in data/raw/rico/combined/
and writes a CSV: rico_id, contrast, whitespace, balance, simplicity, reading_flow

The CSV is consumed by scripts/pretrain_rico.py for self-supervised pretraining.

Usage
-----
  python scripts/generate_pseudo_labels.py
  python scripts/generate_pseudo_labels.py --rico data/raw/rico --batch 256
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uxqa.utils.pixel_rules import compute_pixel_rules, PIXEL_RULE_NAMES


class RICOImageDataset(Dataset):
    def __init__(self, rico_root: Path, image_size: int = 224) -> None:
        self.paths = sorted((rico_root / "combined").glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No JPGs found in {rico_root / 'combined'}")
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return {"tensor": self.tf(img), "rico_id": p.stem}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate pixel-rule pseudo-labels for RICO")
    p.add_argument("--rico", default="data/raw/rico")
    p.add_argument("--out", default="data/raw/rico/pixel_rules.csv")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rico_root = Path(args.rico)
    out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ds = RICOImageDataset(rico_root, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Found {len(ds):,} RICO images — computing pixel rules...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rico_id"] + PIXEL_RULE_NAMES)

        processed = 0
        for batch in loader:
            imgs = batch["tensor"].to(device)
            with torch.no_grad():
                scores = compute_pixel_rules(imgs).cpu()  # [B, 5]

            for i, rico_id in enumerate(batch["rico_id"]):
                row = [rico_id] + [f"{v:.6f}" for v in scores[i].tolist()]
                writer.writerow(row)

            processed += len(batch["rico_id"])
            if processed % 5000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(ds) - processed) / (rate + 1e-6)
                print(f"  {processed:>6,}/{len(ds):,}  {rate:.0f} img/s  ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"\nDone — {len(ds):,} images in {elapsed/60:.1f}m  →  {out_path}")


if __name__ == "__main__":
    main()
