"""
Generate CLIP-based UI quality pseudo-labels for RICO screenshots.

Uses CLIP ViT-B/32 to score each image against curated positive/negative
text prompts describing good/bad UI design.  Score = mean(pos_sim) - mean(neg_sim),
normalized to [0, 1] across all RICO images.

Also validates correlation with UICrit human ratings on the test split.

Output: data/raw/rico/clip_labels.csv  (columns: rico_id, clip_quality)

Usage
-----
  python scripts/generate_clip_labels.py
  python scripts/generate_clip_labels.py --model openai/clip-vit-l-14 --batch 64
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import ssl
import urllib.request

# Bypass SSL for university/corporate networks with SSL inspection
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
_orig_client_init = httpx.Client.__init__
def _client_no_ssl(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _client_no_ssl

import torch
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Prompt design ─────────────────────────────────────────────────────────────
# Validated against UICrit quality scale (1-7 → 0-1)
# Positive: describe properties of high-rated UIs
# Negative: describe properties of low-rated UIs

POSITIVE_PROMPTS = [
    "a clean and well-organized mobile app interface",
    "a professional mobile UI with clear visual hierarchy",
    "an aesthetically pleasing and readable mobile screen",
    "a high quality user interface with good spacing and layout",
    "a polished mobile app with consistent design",
]

NEGATIVE_PROMPTS = [
    "a cluttered and poorly designed mobile app",
    "a confusing mobile user interface with bad layout",
    "an overwhelming and hard to read mobile screen",
    "a low quality user interface with poor design",
    "a disorganized mobile app with inconsistent elements",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device):
    """Load CLIP via open_clip (downloads from GitHub, avoids HF SSL issues)."""
    import open_clip
    # model_name like "openai/clip-vit-b-32" → arch="ViT-B-32", pretrained="openai"
    if "/" in model_name:
        org, name = model_name.split("/", 1)
        arch = name.replace("clip-", "").replace("-", "-").upper()
        # openai/clip-vit-b-32 → ViT-B-32
        arch = arch.replace("VIT-", "ViT-")
        pretrained = org
    else:
        arch, pretrained = model_name, "openai"

    print(f"Loading CLIP via open_clip: arch={arch}  pretrained={pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def encode_texts(model, tokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [N, D]


def encode_images_batch(model, preprocess, img_paths: list[Path], device: torch.device) -> torch.Tensor:
    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in img_paths]).to(device)
    with torch.no_grad():
        feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [B, D]


def validate_on_uicrit(
    scores: dict[str, float],
    uicrit_root: Path,
    rico_root: Path,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Kendall tau between CLIP scores and UICrit ratings on test split."""
    from uxqa.data.uicrit_dataset import UICritDataset
    from uxqa.utils.metrics import kendall_tau, spearman_rho
    import torch

    ds = UICritDataset(uicrit_root, rico_root, split="test", seed=seed)
    clip_s, human_s = [], []
    missing = 0
    for item in ds._items:
        rid = str(item["rico_id"])
        if rid in scores:
            clip_s.append(scores[rid])
            human_s.append(item["quality_score"])
        else:
            missing += 1

    if missing:
        print(f"  [validate] {missing} test items missing CLIP score")

    p = torch.tensor(clip_s).unsqueeze(1)
    t = torch.tensor(human_s).unsqueeze(1)
    tau = kendall_tau(p, t)
    rho = spearman_rho(p, t)
    return {"tau": tau, "rho": rho, "n": len(clip_s)}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rico",    default="data/raw/rico")
    p.add_argument("--uicrit", default="data/raw/uicrit")
    p.add_argument("--model",  default="openai/clip-vit-b-32",
                   help="HuggingFace CLIP model ID")
    p.add_argument("--batch",  type=int, default=128)
    p.add_argument("--out",    default="data/raw/rico/clip_labels.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}  batch={args.batch}")

    model, preprocess, tokenizer = load_model(args.model, device)

    # Encode prompts once
    print("Encoding prompts...")
    pos_feats = encode_texts(model, tokenizer, POSITIVE_PROMPTS, device)  # [5, D]
    neg_feats = encode_texts(model, tokenizer, NEGATIVE_PROMPTS, device)  # [5, D]
    print(f"  pos prompts: {len(POSITIVE_PROMPTS)}  neg prompts: {len(NEGATIVE_PROMPTS)}")

    # Find all images
    rico_root = Path(args.rico)
    img_dir = rico_root / "combined_224" if (rico_root / "combined_224").exists() else rico_root / "combined"
    img_paths = sorted(img_dir.glob("*.jpg"))
    print(f"Images: {len(img_paths):,}  from {img_dir}")

    # Process in batches
    raw_scores: dict[str, float] = {}
    t0 = time.time()
    n = len(img_paths)

    for i in range(0, n, args.batch):
        batch_paths = img_paths[i : i + args.batch]
        try:
            img_feats = encode_images_batch(model, preprocess, batch_paths, device)  # [B, D]
        except Exception as e:
            print(f"  [skip batch {i}] {e}")
            continue

        pos_sim = (img_feats @ pos_feats.T).mean(dim=-1)  # [B]
        neg_sim = (img_feats @ neg_feats.T).mean(dim=-1)  # [B]
        scores = (pos_sim - neg_sim).cpu().tolist()

        for path, score in zip(batch_paths, scores):
            rico_id = path.stem
            raw_scores[rico_id] = score

        if (i // args.batch) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch_paths)) / elapsed
            eta = (n - i - len(batch_paths)) / rate
            print(f"  {i+len(batch_paths):,}/{n:,}  {rate:.0f} img/s  ETA {eta/60:.1f}m")

    print(f"\nScored {len(raw_scores):,} images in {(time.time()-t0)/60:.1f}m")

    # Normalize to [0, 1]
    vals = list(raw_scores.values())
    lo, hi = min(vals), max(vals)
    print(f"Raw score range: [{lo:.4f}, {hi:.4f}]  mean={sum(vals)/len(vals):.4f}")
    scores_norm = {rid: (s - lo) / (hi - lo + 1e-8) for rid, s in raw_scores.items()}

    # Save CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rico_id", "clip_quality"])
        w.writeheader()
        for rico_id, score in scores_norm.items():
            w.writerow({"rico_id": rico_id, "clip_quality": f"{score:.6f}"})
    print(f"Saved: {out_path}  ({len(scores_norm):,} rows)")

    # Validate against UICrit test set
    print("\nValidating against UICrit test ratings...")
    try:
        val = validate_on_uicrit(
            scores_norm,
            uicrit_root=Path(args.uicrit),
            rico_root=rico_root,
        )
        tau, rho = val["tau"], val["rho"]
        print(f"  CLIP vs UICrit test (n={val['n']}): tau={tau:.4f}  rho={rho:.4f}")
        if tau > 0.10:
            print("  -> Good pseudo-labels (tau > 0.10), proceed to pretraining.")
        elif tau > 0.05:
            print("  -> Weak but nonzero signal. May still help pretraining.")
        else:
            print("  -> Very low correlation. Consider revising prompts before pretraining.")
    except Exception as e:
        print(f"  Validation skipped: {e}")

    print(f"\nNext step:")
    print(f"  python scripts/pretrain_rico.py --label-col clip_quality --labels {out_path}")


if __name__ == "__main__":
    main()
