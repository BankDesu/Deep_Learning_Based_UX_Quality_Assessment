---
name: UXQA dataset layout and split behavior
description: Where UICrit/RICO live on disk, how the dataset auto-splits, and which scaffold dirs are unused
type: project
---

Active datasets (only these two are used by current code):
- `data/raw/uicrit/uicrit_public.csv` — UICrit annotations (rico_id, design_quality_rating 1-7, comments). Cloned from google-research-datasets/uicrit.
- `data/raw/rico/combined/<rico_id>.jpg` — RICO screenshots. ~66k JPGs on disk; only ~1000 are paired with UICrit annotations.

Split behavior: `UICritDataset` does its own seeded 80/10/10 split in-memory (`uicrit_dataset.py:300-315`). Resulting sizes: train=800 / val=100 / test=100. The CSVs in `data/manifests/uicrit_{train,val,test}.csv` exist but are NOT read by the dataset — they are stale.

**Why:** This avoids manifest drift; the split is reproducible from `seed=42`.
**How to apply:** Do not regenerate manifests or treat `data/manifests/` as the source of truth. If a deterministic external split is ever needed, refactor `UICritDataset._load_and_split` to accept a manifest path.

Stale scaffold dirs (created by `scripts/setup_dataset_scaffold.py` but unused by code): publaynet, webui, salicon, wdqd, webpage_aesthetics, plus most of `data/interim/` and `data/processed/`. Safe to ignore; may be cleaned up later.
