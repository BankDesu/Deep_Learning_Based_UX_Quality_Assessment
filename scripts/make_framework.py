"""Full pipeline framework diagram — input to output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from PIL import Image

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
CB  = "#1A5F7A"   # deep blue      – visual pathway
CLB = "#3498DB"   # lighter blue
CG  = "#1E8449"   # dark green     – structural pathway
CLG = "#27AE60"
CO  = "#D35400"   # orange         – gating
CP  = "#6C3483"   # purple         – fusion
CR  = "#922B21"   # red            – output
CY  = "#B7950B"   # gold           – preprocessing
CDK = "#1C2833"   # near-black
CWH = "#FDFEFE"   # near-white
CGR = "#F2F3F4"   # light grey bg

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(CWH)
ax.set_facecolor(CWH)
ax.set_xlim(0, 18); ax.set_ylim(0, 10)
ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def box(cx, cy, w, h, color, alpha=0.93, radius=0.18):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={radius}",
                       linewidth=1.8, edgecolor=color,
                       facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(b)

def box_outline(cx, cy, w, h, edge_color, fill=CWH, alpha=0.95, lw=2.0, radius=0.15):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad={radius}",
                       linewidth=lw, edgecolor=edge_color,
                       facecolor=fill, alpha=alpha, zorder=3)
    ax.add_patch(b)

def label(cx, cy, text, size=9.5, color=CWH, bold=True, va="center", ha="center"):
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=size,
            color=color, fontweight=weight, zorder=5,
            fontfamily="DejaVu Sans")

def sublabel(cx, cy, text, size=8, color=CWH, alpha=0.88):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size,
            color=color, alpha=alpha, zorder=5, style="italic",
            fontfamily="DejaVu Sans")

def arrow(x0, y0, x1, y1, color=CDK, lw=2.0, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=14), zorder=6)

def dashed_arrow(x0, y0, x1, y1, color=CDK, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=0.0"), zorder=6)

def section_bg(x, y, w, h, color, alpha=0.07, label_text="", label_size=8):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.1",
                       linewidth=1.2, edgecolor=color,
                       facecolor=color, alpha=alpha, zorder=1)
    ax.add_patch(b)
    if label_text:
        ax.text(x + w/2, y + h + 0.05, label_text,
                ha="center", va="bottom", fontsize=label_size,
                color=color, fontweight="bold", zorder=2, alpha=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# Section backgrounds
# ═══════════════════════════════════════════════════════════════════════════════
section_bg(0.25, 0.5, 2.2, 9.0, CDK, alpha=0.05, label_text="① INPUT")
section_bg(2.7,  3.5, 2.6, 5.8, CY,  alpha=0.06, label_text="② PREPROCESSING")
section_bg(5.5,  5.5, 4.2, 3.8, CB,  alpha=0.06, label_text="③ VISUAL PATHWAY")
section_bg(5.5,  1.0, 4.2, 3.8, CG,  alpha=0.06, label_text="④ STRUCTURAL PATHWAY")
section_bg(10.0, 3.0, 2.8, 4.0, CO,  alpha=0.06, label_text="⑤ GATING")
section_bg(13.1, 3.2, 1.9, 3.6, CP,  alpha=0.06, label_text="⑥ FUSION")
section_bg(15.2, 3.0, 2.5, 6.8, CR,  alpha=0.06, label_text="⑦ OUTPUT")


# ═══════════════════════════════════════════════════════════════════════════════
# ① INPUT — Mobile UI Screenshot
# ═══════════════════════════════════════════════════════════════════════════════
# Try to load a real RICO image
img_paths = list(Path("data/raw/rico/combined").glob("*.jpg"))
if img_paths:
    img_paths_sorted = sorted(img_paths)
    # pick a middle-quality one
    img = Image.open(img_paths_sorted[42]).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, 0, (w+s)//2, s))
    img = img.resize((120, 200), Image.LANCZOS)
    ax_img = fig.add_axes([0.028, 0.38, 0.065, 0.28])
    ax_img.imshow(np.array(img))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_edgecolor(CDK); sp.set_linewidth(2)

box_outline(1.35, 5.0, 1.8, 3.6, CDK, fill=CGR)
label(1.35, 6.6, "Mobile UI", size=9, color=CDK)
label(1.35, 6.2, "Screenshot", size=9, color=CDK)
sublabel(1.35, 5.7, "Any resolution\nRGB image", size=8, color=CDK, alpha=0.7)
label(1.35, 5.0, "e.g. 1080×1920", size=7.5, color=CDK, bold=False)


# ═══════════════════════════════════════════════════════════════════════════════
# ② PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
arrow(2.25, 5.0, 2.8, 7.2)   # → resize
arrow(2.25, 5.0, 2.8, 4.8)   # → structural features

# Resize + Normalize
box(3.5, 7.5, 1.8, 0.8, CY)
label(3.5, 7.62, "Resize  224×224", size=8.5, color=CWH)
sublabel(3.5, 7.28, "Bilinear interpolation", size=7.5, color=CWH)

box(3.5, 6.5, 1.8, 0.8, CY)
label(3.5, 6.62, "Normalize", size=8.5, color=CWH)
sublabel(3.5, 6.28, "ImageNet μ, σ", size=7.5, color=CWH)

arrow(3.5, 7.1, 3.5, 6.9)

# Pixel rule extraction
box(3.5, 4.8, 1.8, 1.2, CLG)
label(3.5, 5.1, "Pixel Feature", size=8.5, color=CWH)
label(3.5, 4.72, "Extraction", size=8.5, color=CWH)
sublabel(3.5, 4.35, "19 handcrafted\nstructural features", size=7.5, color=CWH)


# ═══════════════════════════════════════════════════════════════════════════════
# ③ VISUAL PATHWAY
# ═══════════════════════════════════════════════════════════════════════════════
arrow(4.4, 6.5, 5.6, 7.2)

# EfficientNet blocks (stacked)
block_colors = [CLB, CB, CLB, CB, CLB, CB, CLB, CB]
block_labels = ["Stem\n112×112", "MBConv×2\n112×112",
                "MBConv×4\n56×56",  "MBConv×4\n28×28",
                "MBConv×6\n14×14",  "MBConv×6\n14×14",
                "MBConv×6\n7×7",    "Conv Head\n7×7"]
bx = 6.0; by_start = 8.9; bh = 0.58; gap = 0.03

for i, (bc, bl) in enumerate(zip(block_colors, block_labels)):
    by = by_start - i * (bh + gap)
    box(bx, by, 1.5, bh, bc, alpha=0.88, radius=0.06)
    ax.text(bx, by + 0.08, bl.split("\n")[0], ha="center", va="center",
            fontsize=6.5, color=CWH, fontweight="bold", zorder=5)
    ax.text(bx, by - 0.1, bl.split("\n")[1], ha="center", va="center",
            fontsize=6, color=CWH, alpha=0.82, zorder=5)

# EfficientNet-B4 label
ax.text(6.0, 9.55, "EfficientNet-B4", ha="center", va="bottom",
        fontsize=10, color=CB, fontweight="bold", zorder=5)
ax.text(6.0, 9.35, "(ImageNet pretrained, frozen / fine-tuned last 3)", ha="center",
        va="bottom", fontsize=7, color=CB, style="italic", zorder=5)

# GAP
arrow(6.0, by_start - 7*(bh+gap) - bh/2, 7.1, 7.2)
box(7.6, 7.2, 1.2, 0.65, CB)
label(7.6, 7.32, "Global Avg Pool", size=8, color=CWH)
sublabel(7.6, 7.0, "1792-dim vector", size=7.5, color=CWH)

arrow(8.2, 7.2, 9.3, 7.2)

# Visual projection
box(9.6, 7.2, 0.9, 0.65, CLB)
label(9.6, 7.32, "proj_v", size=8.5, color=CWH)
sublabel(9.6, 7.0, "1792→256", size=7.5, color=CWH)


# ═══════════════════════════════════════════════════════════════════════════════
# ④ STRUCTURAL PATHWAY
# ═══════════════════════════════════════════════════════════════════════════════
arrow(4.4, 4.8, 5.6, 3.8)

# 19 features breakdown
feat_groups = [
    ("Pixel Rules", "5 features", CLG),
    ("Color Stats", "3 features", CG),
    ("Symmetry",    "1 feature",  CLG),
    ("Grid Density","9 features", CG),
    ("Lum Variance","1 feature",  CLG),
]
fx = 6.2; fy_start = 4.65; fh = 0.52
for i, (name, cnt, fc) in enumerate(feat_groups):
    fy = fy_start - i * (fh + 0.05)
    box(fx, fy, 1.5, fh, fc, alpha=0.85, radius=0.06)
    ax.text(fx, fy + 0.09, name, ha="center", va="center",
            fontsize=7, color=CWH, fontweight="bold", zorder=5)
    ax.text(fx, fy - 0.1, cnt, ha="center", va="center",
            fontsize=6.5, color=CWH, alpha=0.85, zorder=5)

ax.text(6.2, 5.28, "19 Structural Features", ha="center", va="bottom",
        fontsize=9.5, color=CG, fontweight="bold", zorder=5)

# Struct encoder
arrow(6.95, 2.75, 7.7, 2.75)
box(8.2, 2.75, 1.4, 1.0, CG)
label(8.2, 3.05, "Struct Encoder", size=8.5, color=CWH)
sublabel(8.2, 2.72, "Linear(19→64)", size=7.5, color=CWH)
sublabel(8.2, 2.47, "GELU · LayerNorm", size=7.5, color=CWH)

arrow(8.9, 2.75, 9.3, 2.75)

# Struct projection
box(9.6, 2.75, 0.9, 0.65, CLG)
label(9.6, 2.87, "proj_s", size=8.5, color=CWH)
sublabel(9.6, 2.62, "64→256", size=7.5, color=CWH)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ GATING NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
# raw visual + raw struct → concat → gate
arrow(8.2, 7.2, 10.5, 6.2, color=CB)   # fv raw → gate
arrow(8.2, 2.75, 10.5, 4.5, color=CG)  # fs raw → gate

box(10.8, 5.3, 1.6, 2.0, CO, radius=0.15)
label(10.8, 6.1,  "Gate", size=11, color=CWH)
sublabel(10.8, 5.72, "Linear(1856→256)", size=8, color=CWH)
sublabel(10.8, 5.45, "[fv; fs] concat", size=8, color=CWH)
sublabel(10.8, 5.18, "Sigmoid → g", size=8, color=CWH)

# gate formulas
ax.text(10.8, 4.82, "g = σ(W·[fᵥ; fₛ])", ha="center", va="center",
        fontsize=8.5, color=CO, style="italic", fontweight="bold", zorder=6)

# proj_v, proj_s → gate output side
arrow(10.05, 7.2,  11.6, 6.1, color=CLB)
arrow(10.05, 2.75, 11.6, 4.55, color=CLG)
arrow(11.6, 5.3, 12.4, 5.0, color=CO)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑥ FUSION
# ═══════════════════════════════════════════════════════════════════════════════
box(13.0, 5.0, 1.6, 1.8, CP)
label(13.0, 5.65, "Fusion", size=11, color=CWH)
sublabel(13.0, 5.28, "g·proj_v(fᵥ)", size=8, color=CWH)
sublabel(13.0, 5.02, "+ (1-g)·proj_s(fₛ)", size=8, color=CWH)
sublabel(13.0, 4.76, "256-dim", size=8, color=CWH)

arrow(13.8, 5.0, 14.5, 5.0, color=CP)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑦ OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
# Prediction head
box(15.2, 7.8, 2.0, 0.75, CDK)
label(15.2, 7.93, "LayerNorm", size=8.5, color=CWH)
sublabel(15.2, 7.68, "256-dim", size=7.5, color=CWH)

box(15.2, 6.8, 2.0, 0.75, CDK)
label(15.2, 6.93, "Linear  256 → 128", size=8.5, color=CWH)
sublabel(15.2, 6.68, "GELU", size=7.5, color=CWH)

box(15.2, 5.8, 2.0, 0.75, CDK)
label(15.2, 5.93, "Linear  128 → 1", size=8.5, color=CWH)
sublabel(15.2, 5.68, "Sigmoid  →  ŷ ∈ [0,1]", size=7.5, color=CWH)

arrow(15.2, 7.42, 15.2, 7.18)
arrow(15.2, 6.42, 15.2, 6.18)

# connect fusion → head
arrow(14.5, 5.0, 15.2, 7.42, color=CP)

# Quality score badge
box(15.7, 4.5, 1.2, 0.8, CR)
label(15.7, 4.68, "Quality Score", size=8.5, color=CWH)
sublabel(15.7, 4.34, "0.0 – 1.0", size=8, color=CWH)
arrow(15.2, 5.42, 15.7, 4.9, color=CR)

# Tiers
tier_data = [
    (4.3, "#27AE60", "Excellent", "> 70%"),
    (3.6, "#F1C40F", "Good",      "50–70%"),
    (2.9, "#E67E22", "Fair",      "30–50%"),
    (2.2, "#E74C3C", "Poor",      "< 30%"),
]
ax.text(16.9, 5.1, "Quality Tier", ha="center", fontsize=9,
        fontweight="bold", color=CDK, zorder=5)
for ty, tc, tname, trange in tier_data:
    box_outline(16.9, ty, 1.8, 0.55, tc, fill=tc, alpha=0.18, lw=2.5, radius=0.1)
    ax.add_patch(FancyBboxPatch((16.0, ty-0.27), 0.35, 0.54,
                                boxstyle="round,pad=0.05", facecolor=tc,
                                linewidth=0, zorder=4, alpha=0.9))
    ax.text(16.45, ty, tname, ha="left", va="center", fontsize=8.5,
            color=CDK, fontweight="bold", zorder=5)
    ax.text(17.65, ty, trange, ha="right", va="center", fontsize=8,
            color=CDK, zorder=5)

arrow(16.0, 4.5, 16.0, 4.3, color=CR)

# Attention map output
box(15.2, 1.5, 2.0, 0.75, "#566573")
label(15.2, 1.63, "Attention Map", size=8.5, color=CWH)
sublabel(15.2, 1.38, "Multi-scale CAM", size=7.5, color=CWH)
dashed_arrow(15.2, 5.42, 15.2, 1.88)


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation note
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(10.8, 4.2, "Ablation modes:", ha="center", fontsize=8,
        color=CDK, fontweight="bold", zorder=5)
for i, (mode, col) in enumerate([("visual-only (g=1)", CB),
                                  ("struct-only (g=0)", CG),
                                  ("full fusion", CO)]):
    ax.text(10.8, 3.92 - i*0.28, f"• {mode}", ha="center", fontsize=7.5,
            color=col, zorder=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════════════════
legend_patches = [
    mpatches.Patch(color=CY,  label="Preprocessing"),
    mpatches.Patch(color=CB,  label="Visual pathway (EfficientNet-B4)"),
    mpatches.Patch(color=CG,  label="Structural pathway (19 features)"),
    mpatches.Patch(color=CO,  label="Gating network"),
    mpatches.Patch(color=CP,  label="Fusion (g·proj_v + (1-g)·proj_s)"),
    mpatches.Patch(color=CDK, label="Prediction head (MLP)"),
    mpatches.Patch(color=CR,  label="Output: quality score + tier"),
]
ax.legend(handles=legend_patches, loc="lower left",
          bbox_to_anchor=(0.0, 0.0), fontsize=8.5,
          framealpha=0.95, edgecolor=CDK, ncol=4,
          title="Component Legend", title_fontsize=9)

# Title
ax.text(9.0, 10.1,
        "Framework: Deep Learning-Based UX Quality Prediction Pipeline",
        ha="center", va="bottom", fontsize=14, fontweight="bold",
        color=CDK, zorder=6)
ax.text(9.0, 9.82,
        "Input Screenshot  →  Preprocessing  →  Visual + Structural Pathways  "
        "→  Gated Fusion  →  Quality Score & Tier",
        ha="center", va="bottom", fontsize=9, color=CDK,
        style="italic", zorder=6)

fig.tight_layout(rect=[0, 0.05, 1, 1])
out_path = OUT / "fig0_framework.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=CWH)
plt.close(fig)
print(f"Saved: {out_path}")
