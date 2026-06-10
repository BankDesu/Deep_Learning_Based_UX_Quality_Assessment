"""
UICritDataset — PyTorch Dataset for UICrit + RICO.

UICrit (UIST 2024, Google Research):
  - 2980 rows across ~983 mobile UI screenshots from RICO
  - Per-row design critiques with bounding boxes
  - Per-row quality ratings (1–7) across 5 dimensions

RICO:
  - Source of the actual screenshot images
  - Paired with UICrit annotations via rico_id

Dataset file on disk
---------------------
<uicrit_root>/
    uicrit_public.csv     ← UICrit annotation file (cloned from GitHub)

<rico_root>/
    combined/
        <rico_id>.jpg     ← RICO screenshot images

Usage
-----
from uxqa.data.uicrit_dataset import UICritDataset, CritiqueRuleMapper

dataset = UICritDataset(
    uicrit_root="data/raw/uicrit",
    rico_root="data/raw/rico",
    split="train",
)
sample = dataset[0]
# sample["screenshot"]      FloatTensor [3, H, W]
# sample["quality_score"]   float  — normalized to [0, 1]
# sample["rule_labels"]     FloatTensor [7]  — 1 = rule violated
# sample["rico_id"]         str

References
----------
UICrit: https://arxiv.org/abs/2407.08850
RICO:   https://interactionmining.org/rico
"""
from __future__ import annotations

import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rule name constants (must match definitions.py)
# ---------------------------------------------------------------------------

RULE_NAMES = [
    "contrast",       # 0
    "whitespace",     # 1
    "visual_balance", # 2
    "density",        # 3
    "alignment",      # 4
    "cta_prominence", # 5
    "reading_flow",   # 6
]

N_RULES = len(RULE_NAMES)   # 7


# ---------------------------------------------------------------------------
# Critique → Rule mapper
# ---------------------------------------------------------------------------

_RULE_KEYWORDS: dict[str, list[str]] = {
    "contrast": [
        "contrast", "readability", "legibility", "color", "colour",
        "dark", "light", "visibility", "hard to read", "text color",
    ],
    "whitespace": [
        "whitespace", "white space", "spacing", "padding", "crowded",
        "empty", "margin", "gap", "breathing room", "too tight",
    ],
    "visual_balance": [
        "balance", "symmetry", "symmetric", "centered", "heavy",
        "weight", "lopsided", "unbalanced", "one-sided",
    ],
    "density": [
        "cluttered", "clutter", "too many", "overwhelming",
        "busy", "overloaded", "elements", "crowded", "information overload",
    ],
    "alignment": [
        "aligned", "alignment", "misaligned", "grid", "column",
        "edge", "offset", "inconsistent", "not lined up",
    ],
    "cta_prominence": [
        "cta", "call to action", "button", "prominent", "visible",
        "primary action", "click", "tap", "small button",
    ],
    "reading_flow": [
        "hierarchy", "order", "flow", "scan", "reading",
        "f-pattern", "z-pattern", "above the fold", "first look",
        "visual order", "priority",
    ],
}


class CritiqueRuleMapper:
    """
    Maps UICrit critique text to rule violation labels.

    Returns a binary vector of shape [N_RULES] where
    label[i] = 1 means rule i was mentioned in the critique.
    """

    def __init__(self) -> None:
        self._kw: dict[str, list[str]] = {
            k: [w.lower() for w in ws]
            for k, ws in _RULE_KEYWORDS.items()
        }

    def map(self, comment_strings: list[str]) -> torch.Tensor:
        """
        Parameters
        ----------
        comment_strings : list of critique text strings from UICrit CSV.

        Returns
        -------
        labels : FloatTensor [N_RULES]  — 1.0 = rule violated
        """
        labels = torch.zeros(N_RULES, dtype=torch.float32)
        for text in comment_strings:
            text_lower = text.lower()
            for rule_idx, rule_name in enumerate(RULE_NAMES):
                if any(kw in text_lower for kw in self._kw[rule_name]):
                    labels[rule_idx] = 1.0
        return labels

    def map_text(self, text: str) -> torch.Tensor:
        """Map a single critique text string."""
        return self.map([text])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UICritDataset(Dataset):
    """
    PyTorch Dataset that pairs RICO screenshots with UICrit quality
    ratings and rule violation labels derived from critique text.

    Rows in the CSV are grouped by rico_id — quality scores are averaged
    across tasks for the same UI, and all comments are pooled together.

    Parameters
    ----------
    uicrit_root : str | Path
        Directory containing uicrit_public.csv.
    rico_root : str | Path
        Root of the RICO dataset. Screenshots expected at:
        <rico_root>/combined/<rico_id>.jpg
    split : {"train", "val", "test"}
        Data split (deterministic seeded shuffle).
    transform : callable, optional
        Torchvision transform applied to the PIL image.
    image_size : tuple[int, int]
        (height, width) to resize screenshots to. Default (256, 128).
    train_ratio : float
        Fraction of data for train split. Default 0.8.
    val_ratio : float
        Fraction of data for val split. Default 0.1.
    seed : int
        Random seed for deterministic split. Default 42.
    """

    # UICrit design_quality_rating scale: 1–7
    _QUALITY_MIN = 1.0
    _QUALITY_MAX = 7.0

    def __init__(
        self,
        uicrit_root: str | Path,
        rico_root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: tuple[int, int] = (256, 128),
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"

        self.uicrit_root = Path(uicrit_root)
        self.rico_root = Path(rico_root)
        self.split = split
        self.transform = transform
        self.image_size = image_size

        self._mapper = CritiqueRuleMapper()
        self._items = self._load_and_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_csv(self) -> list[dict[str, Any]]:
        """Load uicrit_public.csv and return a list of row dicts."""
        csv_path = self.uicrit_root / "uicrit_public.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"UICrit CSV not found: {csv_path}\n"
                "Clone from: https://github.com/google-research-datasets/uicrit"
            )
        rows = []
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def _group_by_rico_id(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Group CSV rows by rico_id.

        Per UI:
          - quality_score = mean(design_quality_rating) normalized to [0, 1]
          - comments      = all comment strings pooled across tasks
        """
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"ratings": [], "comments": []}
        )

        for row in rows:
            rico_id = str(row["rico_id"])
            # Parse quality rating
            try:
                rating = float(row.get("design_quality_rating") or 4.0)
            except ValueError:
                rating = 4.0
            grouped[rico_id]["ratings"].append(rating)

            # Parse comments column — stored as Python list repr
            raw_comments = row.get("comments", "[]")
            try:
                comment_list: list[str] = ast.literal_eval(raw_comments)
            except Exception:
                comment_list = []
            grouped[rico_id]["comments"].extend(comment_list)

        items = []
        for rico_id, data in grouped.items():
            avg_rating = sum(data["ratings"]) / len(data["ratings"])
            quality_score = (avg_rating - self._QUALITY_MIN) / (self._QUALITY_MAX - self._QUALITY_MIN)
            items.append({
                "rico_id":       rico_id,
                "quality_score": float(quality_score),
                "comments":      data["comments"],
            })

        return items

    def _load_and_split(
        self,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Load, group, filter to existing screenshots, then split."""
        manifest_items = self._load_manifest_split()
        if manifest_items is not None:
            return manifest_items

        rows = self._load_csv()
        all_items = self._group_by_rico_id(rows)

        # Filter: only keep items whose RICO screenshot exists on disk
        valid = []
        for item in all_items:
            img_path = self.rico_root / "combined" / f"{item['rico_id']}.jpg"
            if img_path.exists():
                valid.append({**item, "img_path": img_path})

        if not valid:
            raise RuntimeError(
                "No valid UICrit items found. Check that RICO screenshots exist at "
                f"{self.rico_root}/combined/<rico_id>.jpg"
            )

        rng = random.Random(seed)
        indices = list(range(len(valid)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if self.split == "train":
            chosen = indices[:n_train]
        elif self.split == "val":
            chosen = indices[n_train : n_train + n_val]
        else:
            chosen = indices[n_train + n_val :]

        return [valid[i] for i in chosen]

    def _load_manifest_split(self) -> list[dict[str, Any]] | None:
        """Load an explicit split manifest when available.

        The repository ships UICrit manifests used by the paper experiments.
        Prefer them over re-shuffling the CSV so training/evaluation scripts
        reproduce the documented split exactly.
        """
        manifest_path = Path("data") / "manifests" / f"uicrit_{self.split}.csv"
        if not manifest_path.exists():
            return None

        items: list[dict[str, Any]] = []
        with open(manifest_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img_path = Path(row["image_path"])
                if not img_path.is_absolute():
                    img_path = Path.cwd() / img_path
                if not img_path.exists():
                    continue

                meta_raw = row.get("meta") or "{}"
                try:
                    meta = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta = {}

                rating = meta.get("ux_score", meta.get("design_quality_rating", 4.0))
                quality_score = (float(rating) - self._QUALITY_MIN) / (
                    self._QUALITY_MAX - self._QUALITY_MIN
                )

                comments = meta.get("raw_comments", [])
                if isinstance(comments, str):
                    comments = [comments]

                items.append({
                    "rico_id": str(row["id"]),
                    "quality_score": float(quality_score),
                    "comments": comments,
                    "img_path": img_path,
                })

        if not items:
            raise RuntimeError(
                f"Manifest found but no valid images were loaded: {manifest_path}"
            )
        return items

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._items[idx]

        img = Image.open(item["img_path"]).convert("RGB")
        H, W = self.image_size
        img = img.resize((W, H))

        if self.transform is not None:
            img = self.transform(img)
        else:
            import numpy as np
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        rule_labels = self._mapper.map(item["comments"])

        return {
            "screenshot":    img,
            "quality_score": torch.tensor(item["quality_score"], dtype=torch.float32),
            "rule_labels":   rule_labels,
            "rico_id":       item["rico_id"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def class_weights(self) -> torch.Tensor:
        """Per-rule positive class weights for weighted BCE loss."""
        labels = torch.stack(
            [self._mapper.map(item["comments"]) for item in self._items]
        )
        pos = labels.sum(dim=0).clamp_min(1.0)
        neg = (len(self._items) - pos).clamp_min(1.0)
        return (neg / pos).clamp(1.0, 10.0)

    def quality_stats(self) -> dict[str, float]:
        """Mean and std of quality scores."""
        scores = torch.tensor([item["quality_score"] for item in self._items])
        return {"mean": scores.mean().item(), "std": scores.std().item()}

    def __repr__(self) -> str:
        return (
            f"UICritDataset(split={self.split!r}, "
            f"n={len(self)}, "
            f"image_size={self.image_size})"
        )
