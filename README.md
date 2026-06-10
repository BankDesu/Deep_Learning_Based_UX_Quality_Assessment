# UX Quality Prediction from Mobile Screenshots

**IEEE Access Submission** · King Mongkut's University of Technology Thonburi (KMUTT)

Automated UX quality scoring of mobile UI screenshots using EfficientNet-B4 visual features, 19 pixel-derived structural cues, and CLIP-based domain pretraining — evaluated on the UICrit dataset (1,000 annotated screens).

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Web Demo](#web-demo)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

This project investigates automated UX quality prediction from mobile app screenshots using three complementary approaches, all evaluated on the **UICrit** dataset — 1,000 mobile UI screenshots each annotated by seven experienced UX designers.

| Approach | Description |
|---|---|
| **Visual Baseline** | EfficientNet-B4 pretrained on ImageNet, used as a frozen feature extractor (linear probe) or partially fine-tuned |
| **CLIP Pretraining** | Domain adaptation by pretraining on 65,305 RICO screenshots using CLIP-generated pseudo-quality labels |
| **Gated Fusion** | Combining visual features with 19 pixel-derived structural features through a learned additive correction |

**Main finding:** ImageNet visual features alone are sufficient (Kendall τ = 0.215). Neither CLIP pretraining nor structural fusion provides a statistically significant improvement at this dataset scale.

---

## Key Results

Results on UICrit test set (n = 100). τ = Kendall rank correlation with bootstrap 95% CI (B = 10,000 resamples).

| Model | Mode | τ | 95% CI | Significant? |
|---|---|---|---|---|
| **EfficientNet-B4 visual-only** | probe | **0.215** | **[0.088, 0.344]** | **Yes** |
| Gated Fusion full | finetune | 0.183 | [0.046, 0.306] | Yes |
| EfficientNet-B4 | finetune | 0.183 | [0.046, 0.306] | Yes |
| EfficientNet-B4 | probe | 0.165 | [0.038, 0.284] | Yes |
| Swin-T | probe | 0.120 | [-0.014, 0.258] | No |
| ResNet-50 | probe | 0.088 | [-0.041, 0.211] | No |
| CLIP→EfficientNet-B4 | probe | 0.063 | [-0.070, 0.196] | No |
| CLIP→EfficientNet-B4 | finetune | 0.007 | [-0.140, 0.149] | No |
| Gated Fusion full | probe | 0.024 | [-0.112, 0.155] | No (collapse) |
| Structural-only | probe | 0.002 | [-0.142, 0.147] | No |

---

## Architecture

### Main Model: EfficientNet-B4 with Optional Structural Correction

```
Mobile Screenshot [B, 3, 224, 224]
         │
         ▼
┌─────────────────────────────┐
│  EfficientNet-B4 Backbone   │  ← ImageNet pretrained, 19M params
│  (frozen or partial finetune)│
└────────────┬────────────────┘
             │ visual features [B, 1792]
             │
    ┌────────┴───────────────────┐
    │                            │
    ▼                            ▼
Visual Head                Structural Encoder
Linear(1792→256)→GELU      compute_structural_features(img)
→Dropout→Linear(256→1)     → 19 pixel-derived features [B, 19]
visual_logit [B, 1]        → tiny MLP correction [B, 1]
                                structural_delta [B, 1]
    └────────┬───────────────────┘
             │  logit = visual_logit + structural_delta
             ▼
          Sigmoid → quality_score ∈ [0, 1]
```

The structural branch provides a **learned additive correction** initialized at zero, so training always starts from the visual-only baseline. The structural features only contribute if they carry genuine signal beyond the visual representation.

### 19 Structural Features

| Group | Features | Count |
|---|---|---|
| Pixel Rules | contrast, whitespace, balance, simplicity, reading_flow | 5 |
| Color Statistics | color diversity (hue entropy), HSV saturation mean, HSV saturation std | 3 |
| Vertical Symmetry | luminance-channel flip difference | 1 |
| Spatial Grid Density | Sobel edge density in 3×3 grid (top/mid/bot × L/C/R) | 9 |
| Luminance Variance | std of Y-channel (YCbCr) | 1 |
| **Total** | | **19** |

All features are computed via differentiable PyTorch operations (`conv2d`, `avg_pool2d`) and normalized to [0, 1].

**Pixel Rule details:**

| Feature | What it measures |
|---|---|
| `contrast` | Mean per-patch luminance std over 16×16 grid — proxy for readability |
| `whitespace` | Background fraction within ideal range [0.25, 0.60] |
| `balance` | Horizontal intensity centroid proximity to screen center |
| `simplicity` | Inverse of Sobel edge density — anti-clutter proxy |
| `reading_flow` | Content weight in upper third — F/Z-pattern proxy |

### CLIP Pretraining Pipeline

```
RICO Screenshots (65,305 after excluding UICrit overlap)
         │
         ▼
CLIP ViT-B/32  (frozen weights, no gradient)
  ├── 6 positive prompts: "a well-designed mobile app with clean layout..."
  └── 4 negative prompts: "a cluttered and poorly designed mobile interface..."
         │
         ▼
pseudo_score = σ( mean_positive_similarity − mean_negative_similarity )
         │
         ▼
Pretrain EfficientNet-B4 (MSE loss vs pseudo_score, 50 epochs)
         │
         ▼
Transfer pretrained backbone → fine-tune / probe on UICrit
```

### Multi-Scale LayerCAM Explanation

At inference, as a **post-hoc interpretability** step (not a model input), the model produces a spatial explanation map using **LayerCAM** (Jiang et al., TIP 2021). Unlike a global head-weight CAM, LayerCAM weights each spatial location by its own positive gradient of the quality score, so the map stays sharp on high-resolution intermediate layers instead of collapsing into blurry blobs:

```
gᶜⁱʲ = ReLU( ∂y / ∂Aᶜⁱʲ )                 ← per-element positive gradient
Lⁱʲ  = ReLU( Σ_c  gᶜⁱʲ · Aᶜⁱʲ )           ← LayerCAM map for one layer

CAM  = norm( 0.6·↑L⁽¹⁴ˣ¹⁴⁾  +  0.4·↑L⁽²⁸ˣ²⁸⁾ )
```

where `y` is the predicted quality score, `A` the activations of EfficientNet-B4 `features[5]` (14×14, semantic) and `features[3]` (28×28, fine detail), `↑` bilinear upsampling to 224×224, and `norm(·)` rescales to [0, 1]. Using real per-element gradients on the 14×14 and 28×28 layers localizes on actual UI elements (buttons, CTAs, content blocks), far sharper than the coarse 7×7 final layer. Implemented in `compute_layercam` (`src/uxqa/utils/visualizer.py`); regenerate Fig. 5 with `python scripts/make_fig5_gradcam.py`.

---

## Installation

**Requirements:** Python 3.10+, CUDA 11.8+ (recommended)

```bash
git clone https://github.com/BankDesu/Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment.git
cd Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment

# Core
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm open-clip-torch scipy Pillow pyyaml wandb

# For the web demo
pip install gradio opencv-python matplotlib

# Optional — only needed for the experimental UXAssessmentModel (not the paper model)
pip install ultralytics>=8.2.0
```

Or install as a package:

```bash
pip install -e .
```

---

## Dataset Setup

### Directory Layout

```
data/
  manifests/
    uicrit_train.csv        800 samples (80%)
    uicrit_val.csv          100 samples (10%)
    uicrit_test.csv         100 samples (10%)
  raw/
    uicrit/
      uicrit_public.csv     annotation file (rico_id, ratings, comments)
    rico/
      combined/             RICO screenshot .jpg files
```

### UICrit

Download `uicrit_public.csv` from the [UICrit project page](https://github.com/google-research/google-research/tree/master/uicrit) and place it at `data/raw/uicrit/uicrit_public.csv`. The three manifest CSVs in `data/manifests/` are included in this repository.

Each manifest row has:
- `id` — RICO screen ID
- `image_path` — relative path to screenshot
- `label_path` — points to `uicrit_public.csv`
- `meta` — JSON with `ux_score` (mean `design_quality_rating` across annotators, scale 1–7)

**Quality score normalization:** `y = (ux_score − 1) / 6`

The train/val/test split is **deterministic** (seed=42, stratified random shuffle). Running any training script always produces the same split.

| Split | N | Mean score | Std | Min | Max |
|---|---|---|---|---|---|
| Train | 800 | 0.800 | 0.105 | 0.389 | 1.000 |
| Validation | 100 | 0.808 | 0.114 | 0.444 | 1.000 |
| Test | 100 | 0.808 | 0.101 | 0.556 | 1.000 |

### RICO (for pretraining only)

Download RICO screenshots from [interactionmining.org/rico](http://interactionmining.org/rico) and place `.jpg` files in `data/raw/rico/combined/`. Of the 66,261 total RICO screens, 956 overlap with UICrit and are automatically excluded from pretraining (leaving 65,305).

---

## Training Pipeline

Each step saves checkpoints to `checkpoints/`. Steps 1–2 are optional.

### Step 1 (Optional): Generate CLIP Pseudo-Labels

```bash
python scripts/generate_clip_labels.py \
    --data_dir data/raw/rico/combined \
    --output data/raw/rico/clip_labels_rico.csv
```

Runs CLIP ViT-B/32 over all 65,305 RICO screenshots and writes a `pseudo_score` column. Takes ~2–3 hours on a single GPU.

Alternatively, generate pixel-rule pseudo-labels (faster, no CLIP required):

```bash
python scripts/generate_pseudo_labels.py \
    --data_dir data/raw/rico/combined \
    --output data/raw/rico/pixel_rules.csv
```

### Step 2 (Optional): RICO Pretraining

```bash
python scripts/pretrain_rico.py
python scripts/pretrain_rico.py --epochs 30 --batch 128
```

Pretrains EfficientNet-B4 on RICO using the pseudo-labels from Step 1. Saves to `checkpoints/rico-pretrain/best.pt`.

### Step 3: Train Visual Baselines

```bash
# EfficientNet-B4 linear probe (best result in paper)
python scripts/train_baselines.py --arch efficientnet_b4

# Fine-tune last 3 backbone blocks
python scripts/train_baselines.py --arch efficientnet_b4 --mode finetune

# Other backbones
python scripts/train_baselines.py --arch resnet50
python scripts/train_baselines.py --arch swin_t
python scripts/train_baselines.py --arch vit_b_16

# All backbones sequentially (reproduces TABLE III in paper)
python scripts/train_baselines.py --arch all --epochs 100
```

Saves to `checkpoints/baselines/<arch>/best.pt`.

### Step 4: Train Gated Fusion Model

```bash
# Full model: visual + structural features
python scripts/train_multimodal.py --mode probe
python scripts/train_multimodal.py --mode finetune

# Ablations
python scripts/train_multimodal.py --mode probe --ablation visual-only   # g=1, structural off
python scripts/train_multimodal.py --mode probe --ablation struct-only   # g=0, visual off
```

Saves to `checkpoints/multimodal-<ablation>-<mode>/best.pt`.

### Hyperparameters

| Parameter | Probe | Fine-tune | RICO Pretrain |
|---|---|---|---|
| Backbone LR | Frozen | 1×10⁻⁵ | 1×10⁻⁴ |
| Head LR | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ |
| Weight Decay | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |
| Batch Size | 32 | 32 | 64 |
| Warmup Epochs | 5 | 5 | 3 |
| Max Epochs | 100 | 100 | 50 |
| Early Stop Patience | 20 ep. | 20 ep. | — |
| Head Dropout | 0.5 | 0.5 | 0.5 |
| Optimizer | AdamW | AdamW | AdamW |
| GPU | RTX 4070 Super | RTX 4070 Super | RTX 4070 Super |

LR schedule: cosine decay with linear warmup over the first 5 epochs.

---

## Evaluation

```bash
# Test split (default)
python scripts/eval.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt

# Validation split
python scripts/eval.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt --split val
```

**Reported metrics:**

| Metric | Description |
|---|---|
| **Kendall's τ** | Primary metric — rank correlation, robust to outliers |
| **Spearman's ρ** | Monotonic correlation |
| **MSE / MAE** | Absolute prediction error |
| **Bootstrap 95% CI** | B = 10,000 resamples; result is significant if CI excludes zero |
| **Pairwise ranking accuracy (PA)** | Fraction of pairs where model correctly orders the higher-quality UI |

Output also saves per-image predictions to a CSV for qualitative analysis.

---

## Web Demo

```bash
# Auto-detect latest checkpoint
python scripts/app.py

# Specify checkpoint manually
python scripts/app.py --ckpt checkpoints/baselines/efficientnet_b4/best.pt

# Generate a public sharing URL
python scripts/app.py --share
```

Opens at `http://127.0.0.1:7860`. Upload any mobile screenshot to receive:

- **Quality Score (0–100%)** with tier: Poor / Fair / Good / Excellent
- **Multi-Scale Grad-CAM** — post-hoc explanation heatmap of what the model focused on
- **19 Structural Feature Breakdown** — bar chart with values per feature group

Quality tiers: Poor (0–30%) · Fair (30–50%) · Good (50–70%) · Excellent (70–100%)

---

## Project Structure

```
├── src/uxqa/
│   ├── models/
│   │   ├── multimodal_quality_model.py    EfficientNet-B4 + structural correction (paper model)
│   │   ├── structural_encoder.py           19 pixel-derived structural features module
│   │   ├── ux_assessment_model.py          Full Swin+GNN+Attention model (experimental)
│   │   ├── attention/
│   │   │   ├── saliency.py                 SaliencyPredictor (convolutional heatmap)
│   │   │   ├── encoder.py                  AttentionMapEncoder (heatmap → token)
│   │   │   └── branch.py                   AttentionBranch orchestrator
│   │   ├── backbones/
│   │   │   └── swin_encoder.py             Swin Transformer visual encoder
│   │   ├── fusion/
│   │   │   └── cross_modal_transformer.py  Cross-Modal Transformer (experimental)
│   │   ├── heads/
│   │   │   ├── ux_head.py                  Quality scoring head
│   │   │   └── rule_head.py                Per-rule violation head
│   │   ├── layout/
│   │   │   ├── detector.py                 YOLOv8 UI element detector adapter
│   │   │   ├── graph_builder.py            Spatial adjacency matrix (proximity graph)
│   │   │   ├── gnn_encoder.py              2-layer GNN → layout token
│   │   │   └── branch.py                   Layout branch orchestrator
│   │   └── rules/
│   │       └── definitions.py              7 algorithmic UX rules (contrast, whitespace, …)
│   ├── data/
│   │   ├── uicrit_dataset.py               Dataset loader + 800/100/100 deterministic split
│   │   ├── transforms.py                   ImageNet augmentation pipelines
│   │   └── converters.py                   Critique text → rule label keyword mapping
│   └── utils/
│       ├── metrics.py                      Kendall τ, Spearman ρ, bootstrap CI, PA
│       └── pixel_rules.py                  5 differentiable pixel-rule scores
│
├── scripts/
│   ├── app.py                              Gradio web demo
│   ├── train_baselines.py                  Visual baseline training (Step 3)
│   ├── train_multimodal.py                 Gated Fusion training (Step 4)
│   ├── pretrain_rico.py                    RICO domain pretraining (Step 2)
│   ├── generate_clip_labels.py             CLIP pseudo-label generation (Step 1)
│   ├── generate_pseudo_labels.py           Pixel-rule pseudo-label generation (Step 1 alt)
│   ├── eval.py                             Checkpoint evaluation + CSV output
│   ├── infer.py                            Single-image inference script
│   ├── make_figures.py                     Reproduce all paper figures
│   ├── make_ablation_figure.py             Ablation bar chart (Fig. 5a)
│   ├── make_results_figure.py              CI plot (Fig. 4)
│   └── make_qualitative_figure.py          Score distribution + split sizes (Fig. 1)
│
├── data/
│   ├── manifests/
│   │   ├── uicrit_train.csv                800 training samples
│   │   ├── uicrit_val.csv                  100 validation samples
│   │   └── uicrit_test.csv                 100 test samples
│   └── raw/
│       ├── uicrit/uicrit_public.csv        UICrit annotations (download separately)
│       └── rico/combined/                  RICO screenshots (download separately)
│
├── checkpoints/                            Saved model weights (auto-created)
├── configs/                                YAML training configuration files
├── figures/                                Generated paper figures (PNG)
├── PAPER_IEEE_ACCESS_equations_v5.docx     Paper draft (latest)
├── EXPERIMENTS.md                          Full experiment log with all run results
├── ARCHITECTURE.md                         Detailed architecture design notes
├── METRICS.md                              Metric definitions and implementation notes
└── requirements.txt
```

---

## Citation

```bibtex
@article{uxquality2025,
  title   = {Deep Learning-Based UX Quality Prediction from Mobile Screenshots:
             Structural Feature Fusion, CLIP Pretraining, and Multi-Scale Grad-CAM Analysis},
  author  = {[Author Names]},
  journal = {IEEE Access},
  year    = {2025}
}
```

**Dataset citations:**

- UICrit: Duan et al., "UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset," ACM UIST 2024. doi: [10.1145/3654777.3676381](https://doi.org/10.1145/3654777.3676381)
- RICO: Deka et al., "Rico: A Mobile App Dataset for Building Data-Driven Design Applications," ACM UIST 2017. doi: [10.1145/3126594.3126651](https://doi.org/10.1145/3126594.3126651)
