# Paper Figures

All figures are in the `figures/` folder as PNG files (180 DPI).

To regenerate all figures:
```bash
python scripts/make_framework.py        # Fig 0 — full pipeline
python scripts/make_figures.py          # Fig 1, Fig 2 — architecture + dataset
python scripts/make_results_figure.py   # Fig 3 — main results bar chart
python scripts/make_ablation_figure.py  # Fig 4 — ablation + gate analysis
python scripts/make_qualitative_figure.py # Fig 5 — comprehensive analysis
```

---

## Figure 0 — Full Framework Pipeline (`fig0_framework.png`)

**Caption:** Overview of the proposed framework for automated UX quality prediction from mobile screenshots. The pipeline processes an input mobile UI screenshot through five stages: (1) image preprocessing (resize 224×224, ImageNet normalization, data augmentation); (2) visual feature extraction via EfficientNet-B4 pretrained on ImageNet (1,792-dim embedding); (3) optional structural feature computation (19 pixel-derived features: pixel rules, color statistics, symmetry, grid density, luminance variance); (4) optional Gated Fusion combining visual and structural branches; (5) quality score prediction with multi-scale attention map visualization. Best result: visual-only probe τ = 0.215 (significant).

---

## Figure 1 — Model Architecture Diagram (`fig1_architecture.png`)

**Caption:** Detailed architecture of the multi-modal UX quality prediction model. Left: EfficientNet-B4 backbone produces 1,792-dimensional visual features through compound-scaled MBConv blocks followed by global average pooling. Center: Structural encoder maps 19 pixel-derived features through two linear layers with GELU, dropout, and LayerNorm to 64-dim structural features. Right: Gated Fusion network computes gate g ∈ (0,1) from the concatenated [1792; 64]-dim vector and blends visual and structural projections. Bottom: Prediction head applies LayerNorm, two linear layers with GELU and dropout, and sigmoid to produce a score in [0, 1].

---

## Figure 2 — Dataset Statistics (`fig2_dataset.png`)

**Caption:** UICrit dataset characteristics. (Left) Quality score distribution across 983 annotated samples, approximately normal centered at ~0.45. (Right) Train/validation/test split sizes (686/98/100), with the quality score range [0.14, 0.86]. Stratified splitting by score quartile ensures balanced coverage in all partitions.

---

## Figure 3 — Model Comparison Bar Chart (`fig3_results_comparison.png`)

**Caption:** Kendall's τ with 95% bootstrap confidence intervals for all evaluated models on the UICrit test set (n=100). Stars (*) indicate statistically significant results (CI excludes zero). EfficientNet-B4 visual-only probe achieves the best τ = 0.215, confirming that ImageNet-pretrained representations are sufficient for UX quality ranking. CLIP-based pretraining and structural-only models all fail to achieve significance.

---

## Figure 4 — Ablation and Gate Analysis (`fig4_ablation_gate.png`)

**Caption:** (a) Structural feature group ablation. Each bar shows Kendall's τ for a structural-only probe using features from only one group. No individual group achieves statistical significance. Spatial grid density (τ=0.039) shows the highest single-group correlation. (b) Distribution of gate activation values g for the Gated Fusion (fine-tune) model across the test set. Mean gate value = 0.847 indicates the model overwhelmingly relies on visual features, with structural branch contributing only ~15% of the fused representation.

---

## Figure 5 — Comprehensive Analysis (`fig5_analysis.png`)

**Caption:** Multi-panel experimental analysis. (a) UICrit score distribution (983 samples), approximately normal at ~0.45. (b) CLIP pseudo-labels vs expert scores scatter (τ = 0.089), demonstrating the weak alignment explaining CLIP pretraining failure. (c) Best model predictions vs expert scores (τ = 0.215, p<0.05). (d) Correctly ranked pairs out of 4,950 total for each model. (e) Per-quartile concordance rates. (f) Validation τ training curves showing visual probe converging to τ ≈ 0.20 while CLIP-pretrained and structural models fail to improve.

---

## Figure–Section Mapping

| Figure | File | Section | Purpose |
|--------|------|---------|---------|
| Fig 0 | `fig0_framework.png` | Cover / Overview | Full pipeline overview |
| Fig 1 | `fig1_architecture.png` | Section IV-B | Architecture detail |
| Fig 2 | `fig2_dataset.png` | Section III-A | Dataset statistics |
| Fig 3 | `fig3_results_comparison.png` | Section V-B | Main results visualization |
| Fig 4 | `fig4_ablation_gate.png` | Section V-F, V-G | Ablation + gate analysis |
| Fig 5 | `fig5_analysis.png` | Section V / VI | Comprehensive analysis |
