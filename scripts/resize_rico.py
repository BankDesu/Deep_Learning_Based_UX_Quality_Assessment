"""
Pre-resize RICO screenshots to 224x224 JPEG for fast data loading.

Saves resized images to data/raw/rico/combined_224/ (preserves originals).
One-time operation: ~10-15 min for 66K images, cuts per-epoch training time 10x.

Usage
-----
  python scripts/resize_rico.py
  python scripts/resize_rico.py --size 256 --quality 90
  python scripts/resize_rico.py --workers 8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def resize_one(src: Path, dst: Path, size: int, quality: int) -> bool:
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        img.save(dst, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rico",    default="data/raw/rico")
    p.add_argument("--size",    type=int, default=224)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    src_dir = Path(args.rico) / "combined"
    dst_dir = Path(args.rico) / f"combined_{args.size}"
    dst_dir.mkdir(exist_ok=True)

    srcs = list(src_dir.glob("*.jpg"))
    print(f"Found {len(srcs):,} images  -> {dst_dir}")

    done = skipped = errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for src in srcs:
            dst = dst_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            futures[ex.submit(resize_one, src, dst, args.size, args.quality)] = src

        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                done += 1
            else:
                errors += 1
            if i % 5000 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (len(futures) - i) / rate
                print(f"  {i:,}/{len(futures):,}  {rate:.0f} img/s  ETA {remaining/60:.1f}m")

    total = time.time() - t0
    print(f"\nDone: {done:,} resized  {skipped:,} skipped  {errors:,} errors  "
          f"time={total/60:.1f}m  ({(done+skipped)/total:.0f} img/s)")
    print(f"\nUpdate pretrain_rico.py to use: --rico {Path(args.rico)}  "
          f"(point --rico to parent, dataset will use combined_{args.size}/)")


if __name__ == "__main__":
    main()
