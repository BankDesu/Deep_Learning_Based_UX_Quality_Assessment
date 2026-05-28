# Deep Learning-Based UX Quality Prediction from Mobile Screenshots

**IEEE Access Submission** | King Mongkut's University of Technology Thonburi (KMUTT)

> Automated UX quality scoring of mobile UI screenshots using EfficientNet-B4 visual features and 19 pixel-derived structural cues, evaluated on the UICrit dataset.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Demo](#running-the-demo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

This project investigates automated UX quality prediction from mobile app screenshots using three complementary approaches:

1. **Visual backbone** — EfficientNet-B4 pretrained on ImageNet
2. **CLIP pseudo-label pretraining** — Domain adaptation using CLIP-generated labels on 66,261 RICO screenshots
3. **Multi-modal Gated Fusion** — Combining visual features with 19 pixel-derived structural features (color, symmetry, grid density)

**Key finding:** ImageNet visual features alone are sufficient (Kendall tau = 0.215). Domain pretraining and structural fusion both fail to improve performance.

---

## Key Results

| Model | Kendall tau | 95% CI | Significant? |
|-------|-------------|--------|-------------|
| **EfficientNet-B4 visual-only (probe)** | **0.215** | **[0.088, 0.344]** | **Yes** |
| EfficientNet-B4 probe (standalone) | 0.165 | [0.038, 0.284] | Yes |
| Gated Fusion full (finetune) | 0.183 | [0.046, 0.306] | Yes |
| Swin-T probe | 0.120 | [-0.014, 0.258] | No |
| CLIP->EfficientNet-B4 probe | 0.063 | [-0.070, 0.196] | No |
| Gated Fusion full (probe) | 0.024 | [-0.112, 0.155] | No (collapse) |
| Structural features only | 0.002 | [-0.142, 0.147] | No |

---

## Installation

**Requirements:** Python 3.10+, CUDA 11.8+ (recommended)

```bash
# Clone the repository
git clone https://github.com/BankDesu/Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment.git
cd Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm open-clip-torch gradio opencv-python matplotlib pandas scipy scikit-learn python-docx python-pptx
```

---

## Dataset Setup

```
data/
  manifests/
    uicrit_train.csv   (686 samples)
    uicrit_val.csv     (98 samples)
    uicrit_test.csv    (100 samples)
  raw/
    rico/
      combined/        (UI screenshot images .jpg)
```

Each CSV row contains `id`, `image_path`, and `meta` (JSON with `ux_score` on 1-7 scale).

For RICO pretraining: download from interactionmining.org/rico and place images in `data/raw/rico/combined/`.

---

## Running the Demo

```bash
# Run Gradio web demo (auto-detects checkpoint)
python scripts/app.py

# Specify checkpoint manually
python scripts/app.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt

# Create public sharing URL
python scripts/app.py --share
```

Opens at `http://127.0.0.1:7860` — upload any mobile screenshot to get:
- Quality Score (0-100%) with tier label (Poor / Fair / Good / Excellent)
- Multi-scale Attention Map
- 19 Structural Feature breakdown

---

## Training

```bash
# Best model: EfficientNet-B4 linear probe
python scripts/train_baselines.py --backbone efficientnet_b4 --mode probe

# Fine-tune last 3 stages
python scripts/train_baselines.py --backbone efficientnet_b4 --mode finetune

# Other backbones: efficientnet_b0, resnet50, swin_t, vit_b_16
python scripts/train_baselines.py --backbone resnet50 --mode probe

# Generate CLIP pseudo-labels for RICO
python scripts/generate_clip_labels.py --data_dir data/raw/rico/combined --output data/clip_labels_rico.csv

# Pretrain on RICO
python scripts/pretrain_rico.py

# Gated Fusion multi-modal training
python scripts/train_multimodal.py --ablation full --mode probe
python scripts/train_multimodal.py --ablation visual-only --mode probe
python scripts/train_multimodal.py --ablation struct-only --mode probe
```

---

## Evaluation

```bash
python scripts/eval.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt
# Output: Kendall tau, Spearman rho, MSE, MAE, bootstrap 95% CI
```

---

## Project Structure

```
configs/                          Training configuration YAML files
checkpoints/
  baselines/
    efficientnet_b4/best.pt       Best model checkpoint
    efficientnet_b0/best.pt
    resnet50/best.pt
    swin_t/best.pt
  clip-probe-*/                   CLIP pretrained checkpoints
  multimodal-*/                   Gated Fusion checkpoints
data/
  manifests/                      Train/val/test CSV splits
  raw/rico/combined/              UI screenshot images
figures/                          Generated paper figures (PNG)
scripts/
  app.py                          Gradio web demo
  train_baselines.py              Baseline training
  train_multimodal.py             Gated Fusion training
  pretrain_rico.py                CLIP pretraining on RICO
  generate_clip_labels.py         CLIP pseudo-label generation
  eval.py                         Evaluation
  make_figures.py                 Generate paper figures
  make_paper_docx.py              Generate paper .docx
src/uxqa/
  models/
    multimodal_quality_model.py   Gated Fusion model
    structural_encoder.py         19 structural features module
  utils/
    metrics.py                    Kendall tau + bootstrap CI
    pixel_rules.py                Pixel-level UX rule features
PAPER_IEEE_ACCESS.docx            IEEE Access paper draft
FIGURES.md                        Paper figures with captions
EXPERIMENTS.md                    Experiment log and results
README.md
```

---

## Structural Features (19-dim)

| Group | Features | Count |
|-------|----------|-------|
| Pixel Rules | contrast ratio, readability, touch target density, color count, alignment | 5 |
| Color Statistics | color diversity, HSV saturation mean, HSV saturation std | 3 |
| Symmetry | vertical symmetry (luminance channel) | 1 |
| Spatial Grid Density | Canny edge density in 3x3 grid (top/mid/bot x L/C/R) | 9 |
| Luminance Variance | std of Y channel (YCbCr) | 1 |
| **Total** | | **19** |

---

## Citation

```bibtex
@article{uxquality2025,
  title   = {Deep Learning-Based UX Quality Prediction from Mobile Screenshots
             Using Visual and Structural Cues},
  author  = {[Authors]},
  journal = {IEEE Access},
  year    = {2025}
}
```

---

## Acknowledgements

- UICrit dataset: Duan et al., UIST 2024 (doi: 10.1145/3654777.3676381)
- RICO dataset: Deka et al., UIST 2017 (doi: 10.1145/3126594.3126651)
- EfficientNet: Tan & Le, ICML 2019
- CLIP: Radford et al., ICML 2021
