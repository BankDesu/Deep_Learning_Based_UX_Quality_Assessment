"""Generate Figure 3: Model comparison bar chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

models = [
    ("EffNet-B4\nvisual-only*", 0.215, 0.088, 0.344, "#27AE60", True),
    ("EffNet-B4\nfinetune",     0.183, 0.046, 0.306, "#2980B9", True),
    ("EffNet-B4\nprobe",        0.165, 0.038, 0.284, "#2980B9", True),
    ("Swin-T\nprobe",           0.120,-0.014, 0.258, "#8E44AD", False),
    ("ResNet-50\nprobe",        0.088,-0.041, 0.211, "#7F8C8D", False),
    ("CLIP->\nprobe",           0.063,-0.070, 0.196, "#E67E22", False),
    ("CLIP->\nfinetune",        0.007,-0.140, 0.149, "#E67E22", False),
    ("Gated Fusion\n(finetune)",0.183, 0.046, 0.306, "#9B59B6", True),
    ("Gated Fusion\n(probe)",   0.024,-0.112, 0.155, "#E74C3C", False),
    ("Struct-only\n(probe)",    0.002,-0.142, 0.147, "#BDC3C7", False),
]

labels  = [m[0] for m in models]
taus    = [m[1] for m in models]
ci_lo   = [m[2] for m in models]
ci_hi   = [m[3] for m in models]
colors  = [m[4] for m in models]
sig     = [m[5] for m in models]
err_lo  = [t - lo for t, lo in zip(taus, ci_lo)]
err_hi  = [hi - t for t, hi in zip(taus, ci_hi)]

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor("white")

x = np.arange(len(models))
bars = ax.bar(x, taus, color=colors, alpha=0.85, edgecolor="white",
              linewidth=0.8, zorder=3)
ax.errorbar(x, taus, yerr=[err_lo, err_hi], fmt="none",
            ecolor="#2C3E50", elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

# significance stars
for i, (t, s) in enumerate(zip(taus, sig)):
    if s:
        ax.text(i, t + err_hi[i] + 0.008, "*", ha="center", va="bottom",
                fontsize=14, color="#27AE60", fontweight="bold")

# zero line
ax.axhline(0, color="#7F8C8D", lw=1.2, linestyle="--", zorder=2)

# best model annotation
ax.annotate("Best (sig.)\nτ = 0.215", xy=(0, 0.215),
            xytext=(1.2, 0.24),
            fontsize=8.5, color="#27AE60", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.2)
ax.set_ylabel("Kendall's tau (rank correlation)", fontsize=10)
ax.set_title("Figure 3: Model Comparison — Kendall's tau with 95% Bootstrap CI\n"
             "* = statistically significant (CI excludes zero)",
             fontsize=10.5, fontweight="bold", color="#2C3E50")
ax.set_ylim(-0.22, 0.32)
ax.set_xlim(-0.6, len(models) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, zorder=0)

legend_handles = [
    mpatches.Patch(color="#27AE60", label="Best / EfficientNet-B4 visual"),
    mpatches.Patch(color="#8E44AD", label="Swin Transformer"),
    mpatches.Patch(color="#7F8C8D", label="ResNet-50"),
    mpatches.Patch(color="#E67E22", label="CLIP pretrained"),
    mpatches.Patch(color="#9B59B6", label="Gated Fusion (full)"),
    mpatches.Patch(color="#E74C3C", label="Gated Fusion collapsed"),
    mpatches.Patch(color="#BDC3C7", label="Structural-only"),
]
ax.legend(handles=legend_handles, fontsize=7.8, loc="upper right",
          framealpha=0.9, ncol=2)

fig.tight_layout()
fig.savefig(OUT / "fig3_results_comparison.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: figures/fig3_results_comparison.png")
