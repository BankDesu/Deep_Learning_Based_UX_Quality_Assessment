"""
EfficientNet-B4 architecture comparison figures.

Generates three side-by-side diagrams contrasting:
  1) fig_effnet_original.png   — vanilla ImageNet EfficientNet-B4 (1000-class classifier)
  2) fig_effnet_finetuned.png  — UICrit fine-tuned: RICO pretrain + partial unfreeze + quality MLP head
                                  (+ optional structural correction branch as used in this project)
  3) fig_effnet_compare.png    — side-by-side comparison highlighting differences

Run:
  python scripts/make_effnet_comparison.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Palette
CB  = "#1A5F7A"   # visual/backbone
CLB = "#3498DB"
CG  = "#1E8449"   # structural
CO  = "#D35400"   # head / output
CP  = "#6C3483"   # fusion
CR  = "#922B21"   # frozen
CFR = "#7F8C8D"   # frozen-grey
CTR = "#27AE60"   # trainable-green
CWH = "#FDFEFE"
CGR = "#F2F3F4"
CDK = "#1C2833"


def box(ax, cx, cy, w, h, color, text, text_color=CWH, fontsize=10, alpha=0.95, bold=True):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.08",
                       linewidth=1.4, edgecolor=color,
                       facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center",
            color=text_color, fontsize=fontsize,
            fontweight="bold" if bold else "normal", zorder=4)


def outline_box(ax, cx, cy, w, h, edge, text, fontsize=10, text_color=CDK, lw=1.6, ls="-"):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.08",
                       linewidth=lw, edgecolor=edge,
                       facecolor=CWH, linestyle=ls, zorder=3)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=CDK, lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14,
                        color=color, linewidth=lw, zorder=2)
    ax.add_patch(a)


def block_label(ax, cx, cy, text, color=CDK, fontsize=9):
    ax.text(cx, cy, text, ha="center", va="center",
            color=color, fontsize=fontsize, fontstyle="italic", zorder=4)


# ───────────────────────────────────────────────────────────────────────────────
# Figure 1: Original EfficientNet-B4 (ImageNet, vanilla)
# ───────────────────────────────────────────────────────────────────────────────
def draw_original(ax, title=True):
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    if title:
        ax.text(7, 5.55, "Original EfficientNet-B4 (ImageNet)",
                ha="center", fontsize=14, fontweight="bold", color=CDK)
        ax.text(7, 5.15, "torchvision.models.efficientnet_b4 — IMAGENET1K_V1 weights",
                ha="center", fontsize=9, fontstyle="italic", color="#555")

    # Input
    box(ax, 1.0, 3.0, 1.4, 1.0, CO, "Image\n224×224×3", fontsize=9)

    # 9 feature blocks
    bx = [2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4]
    for i, x in enumerate(bx):
        box(ax, x, 3.0, 0.85, 1.0, CFR,
            f"Block\n{i}", fontsize=8)
    # Bracket label
    ax.annotate("", xy=(10.85, 4.25), xytext=(1.95, 4.25),
                arrowprops=dict(arrowstyle="-", color=CFR, lw=1.2))
    ax.text(6.4, 4.55, "MBConv feature blocks  (frozen — ImageNet pretrained)",
            ha="center", fontsize=9, color=CFR, fontstyle="italic")

    # Pool + classifier
    box(ax, 11.5, 3.0, 0.95, 1.0, CFR, "Avg\nPool", fontsize=9)
    box(ax, 12.7, 3.0, 1.2, 1.0, CO, "Linear\n1792→1000", fontsize=9)
    box(ax, 13.7, 3.0, 0.75, 1.0, CR, "Soft\nmax", fontsize=8)

    # Output
    ax.text(13.7, 1.7, "1000-class\nImageNet probs",
            ha="center", fontsize=9, color=CR, fontweight="bold")

    # Arrows
    xs = [1.7, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.5, 12.7]
    targets = [2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.5, 12.7, 13.7]
    for s, t in zip(xs, targets):
        arrow(ax, s + 0.43, 3.0, t - 0.43, 3.0, color="#888", lw=1.0)
    arrow(ax, 13.7, 2.5, 13.7, 2.1, color=CR)

    # Footer note
    ax.text(7, 0.7,
            "Trainable params: ~19.3M  (all)   |   Output: argmax over 1000 ImageNet classes",
            ha="center", fontsize=9, color=CDK)
    ax.text(7, 0.3,
            "Loss: cross-entropy   |   Pretrained on 1.28M ImageNet images",
            ha="center", fontsize=8.5, color="#555", fontstyle="italic")


# ───────────────────────────────────────────────────────────────────────────────
# Figure 2: Fine-tuned (this project)
# ───────────────────────────────────────────────────────────────────────────────
def draw_finetuned(ax, title=True):
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    if title:
        ax.text(7, 7.55, "Fine-tuned EfficientNet-B4 (UX Quality, this project)",
                ha="center", fontsize=14, fontweight="bold", color=CDK)
        ax.text(7, 7.15,
                "Stage 1: RICO pixel-rule pretrain  →  Stage 2: UICrit fine-tune + structural correction",
                ha="center", fontsize=9, fontstyle="italic", color="#555")

    # Input
    box(ax, 1.0, 4.5, 1.4, 1.0, CO, "Screenshot\n224×224×3", fontsize=9)

    # Backbone — 9 blocks, last 3 unfrozen
    bx = [2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4]
    for i, x in enumerate(bx):
        col = CFR if i < 6 else CTR
        box(ax, x, 4.5, 0.85, 1.0, col, f"Block\n{i}", fontsize=8)
    ax.text(4.4, 5.85, "frozen (RICO-pretrained)", ha="center",
            fontsize=8.5, color=CFR, fontstyle="italic")
    ax.text(9.4, 5.85, "unfrozen (--unfreeze-last-n 3)", ha="center",
            fontsize=8.5, color=CTR, fontstyle="italic")

    # Pool — classifier REMOVED (Identity)
    box(ax, 11.5, 4.5, 0.95, 1.0, CFR, "Avg\nPool", fontsize=9)
    outline_box(ax, 12.7, 4.5, 1.25, 1.0, "#bbb",
                "Identity\n(classifier\nremoved)", fontsize=7.5, text_color="#888", ls="--")

    # Arrows along backbone
    src = [1.7, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.5]
    tgt = [2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.5, 12.7]
    for s, t in zip(src, tgt):
        arrow(ax, s + 0.43, 4.5, t - 0.43, 4.5, color="#888", lw=1.0)

    # Features → visual head
    ax.text(13.6, 4.5, "feat\n[1792]", ha="center", va="center",
            fontsize=8.5, color=CDK)
    arrow(ax, 13.2, 4.5, 13.9, 4.5, color=CDK)

    # Visual quality head (NEW)
    box(ax, 13.7, 3.1, 1.7, 0.8, CO, "Linear 1792→256", fontsize=8.5)
    box(ax, 13.7, 2.35, 1.7, 0.55, CLB, "GELU + Dropout 0.5", fontsize=8)
    box(ax, 13.7, 1.65, 1.7, 0.55, CO, "Linear 256→1", fontsize=8.5)
    arrow(ax, 13.7, 4.0, 13.7, 3.5, color=CO)
    arrow(ax, 13.7, 2.7, 13.7, 2.65, color=CO)
    arrow(ax, 13.7, 2.05, 13.7, 1.95, color=CO)

    # Structural correction branch (lower)
    ax.text(1.0, 3.0, "Pixel-derived\nfeatures (19)",
            ha="center", fontsize=8.5, color=CG, fontweight="bold")
    box(ax, 1.0, 2.0, 1.6, 0.65, CG, "Linear 19→8", fontsize=8.5)
    box(ax, 1.0, 1.25, 1.6, 0.55, CG, "GELU", fontsize=8)
    box(ax, 1.0, 0.55, 1.6, 0.55, CG, "Linear 8→1  (zero-init)", fontsize=7.5)
    arrow(ax, 1.0, 2.55, 1.0, 2.35, color=CG)
    arrow(ax, 1.0, 1.7, 1.0, 1.55, color=CG)
    arrow(ax, 1.0, 0.95, 1.0, 0.85, color=CG)

    # Fusion (sigmoid of logit_v + delta)
    box(ax, 7.0, 1.0, 2.6, 0.85, CP,
        "score = σ(logit_v + δ)", fontsize=10)
    arrow(ax, 13.7, 1.35, 13.7, 1.15, color=CO)
    arrow(ax, 13.7, 1.0, 8.3, 1.0, color=CO, lw=1.2)
    arrow(ax, 1.8, 0.55, 5.7, 1.0, color=CG, lw=1.2)

    # Output
    box(ax, 7.0, 0.25, 2.6, 0.45, CR, "UX quality ∈ [0, 1]", fontsize=9)
    arrow(ax, 7.0, 0.55, 7.0, 0.5, color=CR)

    # Header bracket
    ax.text(6.4, 6.3, "EfficientNet-B4 backbone   (Stage 1: RICO 66K, 5 pixel-rule pseudo-labels)",
            ha="center", fontsize=9, color=CB, fontweight="bold")


# ───────────────────────────────────────────────────────────────────────────────
# Figure 3: Side-by-side comparison
# ───────────────────────────────────────────────────────────────────────────────
def draw_comparison():
    fig, axes = plt.subplots(2, 1, figsize=(16, 13))
    fig.patch.set_facecolor(CWH)

    # Top: original (compact)
    axes[0].set_facecolor(CWH)
    draw_original(axes[0], title=False)
    axes[0].text(7, 5.55, "(a)  Original EfficientNet-B4 — ImageNet classification",
                 ha="center", fontsize=13, fontweight="bold", color=CDK)

    # Bottom: fine-tuned
    axes[1].set_facecolor(CWH)
    draw_finetuned(axes[1], title=False)
    axes[1].text(7, 7.55, "(b)  Fine-tuned EfficientNet-B4 — UX quality regression (this work)",
                 ha="center", fontsize=13, fontweight="bold", color=CDK)

    # Side-by-side delta legend
    legend_text = (
        "Differences:   "
        "(1) Classifier 1792→1000 replaced by 1792→256→1 quality MLP    "
        "(2) Output: 1000-way softmax → scalar sigmoid score ∈ [0,1]    "
        "(3) Last 3 MBConv blocks unfrozen; earlier blocks frozen    "
        "(4) Stage-1 RICO pretrain on 5 pixel-rule pseudo-labels (66K UI screens)    "
        "(5) Loss: cross-entropy → MSE + pairwise ranking    "
        "(6) Structural correction δ from 19 handcrafted layout features added to visual logit"
    )
    fig.text(0.5, 0.015, legend_text, ha="center", fontsize=8.5,
             color=CDK, wrap=True)

    plt.subplots_adjust(hspace=0.15, top=0.97, bottom=0.05, left=0.02, right=0.98)
    out = OUT / "fig_effnet_compare.png"
    plt.savefig(out, dpi=170, facecolor=CWH, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ───────────────────────────────────────────────────────────────────────────────
def main():
    # Figure 1
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(CWH); ax.set_facecolor(CWH)
    draw_original(ax)
    out = OUT / "fig_effnet_original.png"
    plt.savefig(out, dpi=170, facecolor=CWH, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # Figure 2
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(CWH); ax.set_facecolor(CWH)
    draw_finetuned(ax)
    out = OUT / "fig_effnet_finetuned.png"
    plt.savefig(out, dpi=170, facecolor=CWH, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # Figure 3
    draw_comparison()


if __name__ == "__main__":
    main()
