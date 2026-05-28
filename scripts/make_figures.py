"""Generate Figure 1 (architecture) and Figure 2 (score distribution + samples)."""
import sys
sys.path.insert(0, "src")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path
from PIL import Image

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
C_BLUE   = "#2980B9"
C_GREEN  = "#27AE60"
C_ORANGE = "#E67E22"
C_PURPLE = "#8E44AD"
C_GRAY   = "#7F8C8D"
C_RED    = "#C0392B"
C_LIGHT  = "#ECF0F1"
C_DARK   = "#2C3E50"


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture Diagram
# ════════════════════════════════════════════════════════════════════════════

def draw_box(ax, xy, w, h, label, sublabel="", color=C_BLUE, fontsize=9, alpha=0.92):
    x, y = xy
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=color, facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.045, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color="white", alpha=0.88, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_DARK, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=5)


fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Input image ──────────────────────────────────────────────────────────────
draw_box(ax, (0.05, 0.65), 0.08, 0.14, "Mobile UI\nScreenshot",
         "224×224 RGB", C_GRAY, fontsize=8)

# ── EfficientNet-B4 ──────────────────────────────────────────────────────────
draw_box(ax, (0.20, 0.65), 0.10, 0.14, "EfficientNet-B4",
         "ImageNet pretrained", C_BLUE, fontsize=8)
arrow(ax, 0.09, 0.65, 0.15, 0.65)

# GAP
draw_box(ax, (0.33, 0.65), 0.07, 0.11, "GAP",
         "1792-dim", C_BLUE, fontsize=8)
arrow(ax, 0.25, 0.65, 0.295, 0.65)

# ── Structural path ──────────────────────────────────────────────────────────
draw_box(ax, (0.05, 0.28), 0.08, 0.12, "Structural\nFeatures",
         "19-dim", C_GREEN, fontsize=8)

draw_box(ax, (0.20, 0.28), 0.10, 0.12, "Struct Encoder",
         "MLP + LayerNorm", C_GREEN, fontsize=8)
arrow(ax, 0.09, 0.28, 0.15, 0.28)

draw_box(ax, (0.33, 0.28), 0.07, 0.11, "Projection",
         "64-dim", C_GREEN, fontsize=8)
arrow(ax, 0.25, 0.28, 0.295, 0.28)

# ── Pixel rules label ────────────────────────────────────────────────────────
ax.text(0.05, 0.14, "19 features: contrast · saturation\nsymmetry · grid density · lum std",
        ha="center", va="center", fontsize=7.5, color=C_GRAY,
        style="italic", zorder=4)

# ── Gating ───────────────────────────────────────────────────────────────────
draw_box(ax, (0.50, 0.47), 0.10, 0.30, "Gating Network",
         "Linear(1856→256)\nSigmoid", C_ORANGE, fontsize=8)

# arrows from GAP and Struct Projection into gate
arrow(ax, 0.365, 0.65, 0.445, 0.55)
arrow(ax, 0.365, 0.28, 0.445, 0.40)

# Visual projection
draw_box(ax, (0.50, 0.76), 0.10, 0.11, "proj_v",
         "Linear(1792→256)", C_BLUE, fontsize=8)
arrow(ax, 0.365, 0.68, 0.445, 0.76)

# Struct projection
draw_box(ax, (0.50, 0.20), 0.10, 0.11, "proj_s",
         "Linear(64→256)", C_GREEN, fontsize=8)
arrow(ax, 0.365, 0.25, 0.445, 0.20)

# ── Fusion ───────────────────────────────────────────────────────────────────
draw_box(ax, (0.66, 0.47), 0.09, 0.14, "Fusion",
         "g·proj_v + (1-g)·proj_s\n256-dim", C_PURPLE, fontsize=8)

arrow(ax, 0.555, 0.76, 0.615, 0.54)
arrow(ax, 0.555, 0.20, 0.615, 0.41)
arrow(ax, 0.555, 0.47, 0.615, 0.47)

# ── Head ─────────────────────────────────────────────────────────────────────
draw_box(ax, (0.81, 0.47), 0.09, 0.20, "Head",
         "LayerNorm → Linear\n256→128→1", C_DARK, fontsize=8)
arrow(ax, 0.705, 0.47, 0.765, 0.47)

# ── Output ───────────────────────────────────────────────────────────────────
draw_box(ax, (0.95, 0.47), 0.07, 0.12, "Quality\nScore",
         "σ ∈ [0, 1]", C_RED, fontsize=8.5)
arrow(ax, 0.855, 0.47, 0.915, 0.47)

# ── Visual-only ablation note ────────────────────────────────────────────────
ax.annotate("visual-only ablation:\ndisable struct branch",
            xy=(0.66, 0.62), xytext=(0.72, 0.80),
            fontsize=7, color=C_ORANGE, style="italic",
            arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.8))

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_BLUE,   label="Visual pathway (EfficientNet-B4)"),
    mpatches.Patch(color=C_GREEN,  label="Structural pathway (19 pixel features)"),
    mpatches.Patch(color=C_ORANGE, label="Gating network"),
    mpatches.Patch(color=C_PURPLE, label="Gated Fusion"),
    mpatches.Patch(color=C_DARK,   label="Prediction head"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
          framealpha=0.9, edgecolor=C_GRAY, ncol=3)

ax.set_title("Figure 1: Multi-Modal Gated Fusion Architecture for UX Quality Prediction",
             fontsize=11, fontweight="bold", pad=10, color=C_DARK)

fig.tight_layout()
fig.savefig(OUT / "fig1_architecture.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved: figures/fig1_architecture.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Score Distribution + Sample UIs
# ════════════════════════════════════════════════════════════════════════════

import pandas as pd

dfs = []
for split in ("train", "val", "test"):
    csv = Path(f"data/manifests/uicrit_{split}.csv")
    if csv.exists():
        df = pd.read_csv(csv)
        df["split"] = split
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
scores = np.array([json.loads(r["meta"])["ux_score"] / 7.0
                   for _, r in df_all.iterrows()])
splits = df_all["split"].values

# Pick sample UIs: lowest 3, middle 3, highest 3
idx_sorted = np.argsort(scores)
n = len(scores)
sample_idx = list(idx_sorted[:3]) + list(idx_sorted[n//2-1:n//2+2]) + list(idx_sorted[-3:])
sample_scores = scores[sample_idx]
sample_paths  = [Path(df_all.iloc[i]["image_path"]) for i in sample_idx]

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("white")

# ── Left: histogram ───────────────────────────────────────────────────────────
ax_hist = fig.add_axes([0.04, 0.12, 0.38, 0.76])

tier_ranges = [
    (0.00, 0.30, "Poor",      "#E74C3C"),
    (0.30, 0.50, "Fair",      "#E67E22"),
    (0.50, 0.70, "Good",      "#F1C40F"),
    (0.70, 1.00, "Excellent", "#2ECC71"),
]

# shade tiers
for lo, hi, label, color in tier_ranges:
    ax_hist.axvspan(lo, hi, alpha=0.12, color=color)
    ax_hist.text((lo+hi)/2, 0.5, label, ha="center", va="bottom",
                 fontsize=8, color=color, fontweight="bold",
                 transform=ax_hist.get_xaxis_transform())

# per-split histograms
colors_split = {"train": C_BLUE, "val": C_GREEN, "test": C_ORANGE}
labels_split = {"train": f"Train (n={sum(splits=='train')})",
                "val":   f"Val   (n={sum(splits=='val')})",
                "test":  f"Test  (n={sum(splits=='test')})"}

bins = np.linspace(0, 1, 21)
for sp in ("train", "val", "test"):
    mask = splits == sp
    ax_hist.hist(scores[mask], bins=bins, alpha=0.55,
                 color=colors_split[sp], label=labels_split[sp], edgecolor="white")

ax_hist.axvline(scores.mean(), color=C_DARK, linestyle="--", lw=1.5,
                label=f"Mean = {scores.mean():.2f}")
ax_hist.set_xlabel("Normalized UX Quality Score (0–1)", fontsize=10)
ax_hist.set_ylabel("Count", fontsize=10)
ax_hist.set_title("Score Distribution — UICrit Dataset", fontsize=11,
                  fontweight="bold", color=C_DARK)
ax_hist.legend(fontsize=8.5, framealpha=0.9)
ax_hist.set_xlim(0, 1)
ax_hist.spines[["top", "right"]].set_visible(False)

# score stats box
stats_txt = (
    f"n = {len(scores)}\n"
    f"Mean = {scores.mean():.3f}\n"
    f"Std  = {scores.std():.3f}\n"
    f"Min  = {scores.min():.3f}\n"
    f"Max  = {scores.max():.3f}"
)
ax_hist.text(0.97, 0.97, stats_txt, transform=ax_hist.transAxes,
             va="top", ha="right", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor=C_LIGHT, alpha=0.8, edgecolor=C_GRAY))

# ── Right: 3×3 sample grid ────────────────────────────────────────────────────
tier_labels = ["Poor", "Fair", "Good", "Excellent"]

def get_tier(s):
    if s < 0.30: return "Poor",      "#E74C3C"
    if s < 0.50: return "Fair",      "#E67E22"
    if s < 0.70: return "Good",      "#F1C40F"
    return "Excellent", "#2ECC71"

col_titles = ["Low Quality\n(score < 0.30)", "Mid Quality\n(0.40–0.60)", "High Quality\n(score > 0.70)"]
row_y = [0.90, 0.57, 0.24]
col_x = [0.47, 0.64, 0.81]

for col_i, cx in enumerate(col_x):
    fig.text(cx + 0.065, 0.91, col_titles[col_i], ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color=C_DARK)

for i, (path, score) in enumerate(zip(sample_paths, sample_scores)):
    row = i % 3
    col = i // 3
    left = col_x[col]
    bottom = row_y[2 - row] - 0.30
    ax_img = fig.add_axes([left, bottom, 0.13, 0.28])

    try:
        img = Image.open(path).convert("RGB")
        # crop to center square
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img = img.resize((128, 128), Image.LANCZOS)
        ax_img.imshow(np.array(img))
    except Exception:
        ax_img.set_facecolor("#DDDDDD")
        ax_img.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax_img.transAxes)

    tier_name, tier_color = get_tier(score)
    ax_img.set_title(f"{score:.2f}  [{tier_name}]",
                     fontsize=7.5, color=tier_color, fontweight="bold", pad=2)
    for spine in ax_img.spines.values():
        spine.set_edgecolor(tier_color)
        spine.set_linewidth(2)
    ax_img.set_xticks([]); ax_img.set_yticks([])

fig.text(0.68, 0.96,
         "Figure 2b: Sample UIs — Low / Mid / High Quality",
         ha="center", fontsize=9, fontweight="bold", color=C_DARK)

fig.suptitle("Figure 2: UICrit Dataset Overview", fontsize=13,
             fontweight="bold", color=C_DARK, y=0.99)

fig.savefig(OUT / "fig2_dataset.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved: figures/fig2_dataset.png")
print("Done.")
