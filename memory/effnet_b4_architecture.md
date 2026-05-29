---
name: effnet-b4-architecture
description: "How the original ImageNet EfficientNet-B4 works vs the project's fine-tuned multi-modal UX-quality model, and the exact differences between them"
metadata: 
  node_type: memory
  type: project
  originSessionId: 35133a3e-8e27-4e33-bd5c-55431e9aa474
---

# EfficientNet-B4: original vs this project's fine-tuned model

## Original EfficientNet-B4 (torchvision, ImageNet)
- Source: `torchvision.models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)`
- Pipeline: `Image 224×224×3 → 9 MBConv feature blocks → AvgPool → Linear(1792→1000) → Softmax → argmax over 1000 ImageNet classes`
- All ~19.3M params trainable, trained on 1.28M ImageNet images with cross-entropy loss
- Purpose: generic object classification

## This project's fine-tuned model (multimodal_quality_model.py / train_proposed.py)
Two-stage training pipeline producing a UX-quality scalar score in [0,1].

**Stage 1 — RICO pixel-rule pretraining (`scripts/pretrain_rico.py`)**
- Start: EfficientNet-B4 ImageNet weights
- Replace classifier with `Linear(1792→512) → GELU → Dropout(0.3) → Linear(512→5) → Sigmoid`
- Train on RICO 66K screenshots to predict 5 pixel-rule pseudo-labels (contrast, whitespace, balance, simplicity, reading_flow)
- Output: backbone with UI-domain features → saved as `checkpoints/pretrain-rico-effb4/best.pt`

**Stage 2 — UICrit fine-tuning (`scripts/train_proposed.py`, `MultiModalQualityModel`)**
- Load Stage-1 backbone; `base.classifier = Identity()`
- Visual head: `Linear(1792→256) → GELU → Dropout(0.5) → Linear(256→1)` → logit_v
- Freeze policy (mode=finetune, `--unfreeze-last-n 3`): freeze blocks 0–5, unfreeze blocks 6–8 + final conv/bn
- Structural correction branch (parallel): 19 handcrafted pixel features (5 pixel rules + color diversity + saturation mean/std + vertical symmetry + 3×3 grid edge density + luminance std) → `Linear(19→8) → GELU → Linear(8→1, zero-init)` → δ
- Fusion: `score = sigmoid(logit_v + δ)` (additive; δ≈0 at init so training starts from visual baseline)
- Loss: `0.1 · MSE + 5.0 · pairwise_ranking_loss(margin=0.15)`
- Optimizer: AdamW with discriminative LRs — head=1e-3, backbone=1e-5, cosine schedule

## Key differences (original → fine-tuned)
| Aspect | Original | This project |
|---|---|---|
| Head | `Linear(1792→1000)` + softmax | `Linear(1792→256→1)` + sigmoid |
| Output | argmax over 1000 ImageNet classes | scalar UX quality ∈ [0,1] |
| Pretraining | ImageNet only | ImageNet → RICO pixel-rule (UI-domain) |
| Trainable | all ~19.3M params | last 3 MBConv blocks + heads (rest frozen) |
| Loss | cross-entropy | MSE + pairwise ranking |
| Modality | image only | image + 19 handcrafted structural features (additive δ correction, ~160 extra params) |
| Interpretability | none | δ>0 = structure boosts quality, δ<0 = structure reduces it |

**Why:** UICrit has only ~800 training samples; the partial unfreeze + RICO pretraining + tiny structural branch are chosen to avoid overfitting while injecting UI-domain priors and interpretability.

**How to apply:** When asked about the model architecture or differences, reference these specifics. When proposing changes, respect the design constraints — small data (don't unfreeze more), additive δ (don't make structural branch dominant), ranking-first metric (Kendall τ is primary).

Related: [[uicrit-dataset-size]], [[two-stage-training-rationale]]
