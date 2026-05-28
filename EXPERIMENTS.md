# UX Quality Assessment — Experiment Log

**Started:** 2026-05-25 / **Last update:** 2026-05-26
**Branch:** `dataset`
**Hardware:** RTX 4070 SUPER (12 GB), CUDA 13.1 driver

---

## 1. Context

- **Task:** UI quality scoring — predict UICrit human rating (1-7, normalized to [0,1]) from RICO mobile UI screenshots.
- **Data:** UICrit (Google Research, UIST 2024) + RICO screenshots. After grouping by `rico_id` and filtering to existing JPGs: **train=800 / val=100 / test=100** (80/10/10 seeded split, seed=42).
- **Label distribution:** mean=0.80, std=0.10 — narrow range, derived from `mean(design_quality_rating)/6` across ~7 raters per UI.
- **Primary metric:** Kendall's τ (rank correlation between predicted and human quality). Secondary: Spearman ρ, Pearson r, MAE, per-rule F1.

## 2. Environment Setup

| Step | Done |
|---|---|
| `git clone` UICrit annotations | ✅ `data/raw/uicrit/uicrit_public.csv` |
| Download RICO screenshots (66,261 JPG) | ✅ `data/raw/rico/combined/*.jpg` |
| Install deps: `torchvision`, `Pillow`, `ultralytics` | ✅ |
| Upgrade to CUDA torch (`torch 2.12.0+cu126`) | ✅ |
| Verify `UICritDataset` end-to-end (DataLoader + sample shapes) | ✅ |
| Smoke forward pass through `UXAssessmentModel` | ✅ |

## 3. Code Fixes (pre-experiment)

| File | Change | Why |
|---|---|---|
| `src/uxqa/config.py` | Replaced dead keys `layout_loss_weight`, `attention_loss_weight` with `rank_loss_weight=5.0`, `rule_loss_weight=0.05` | Old keys defined in `TrainConfig` but never referenced in `train.py` |
| `src/uxqa/config.py` | `_apply_overrides` now **raises `ValueError`** on unknown YAML key | Silent ignore previously masked config drift |
| `configs/default.yaml` | Synced loss-weight keys with `TrainConfig` | YAML keys were being silently ignored |
| `scripts/train.py` | Pass `rank_w`, `rule_w` from config instead of hard-coded | Make loss weights tunable |
| `src/uxqa/models/backbones/swin_encoder.py` | Replaced random-init "Swin-like" stub with `torchvision.models.swin_t(weights=IMAGENET1K_V1)` + 1×1 conv projection to `embed_dim` | Original was vanilla ViT-ish encoder from scratch (~1M params); now real Swin-T pretrained (~28M) |
| `src/uxqa/models/heads/rule_head.py` | `layout_quality = 0.5*neural + 0.5*weighted_rule_score` → `0.9*neural + 0.1*weighted_rule_score` | `weighted_rule_score` depends on **non-trainable** `RuleChecker` output; half-strength signal was a bottleneck |
| `src/uxqa/utils/metrics.py` | `pairwise_ranking_loss` margin: 0.05 → 0.15 | Margin was too tight given target std=0.10; pairs rarely exceeded it |

## 4. Experiments Timeline

All runs: AdamW, target `quality_score` ∈ [0,1]. Loss: `0.1×MSE + 5.0×pairwise_ranking(margin=0.15)`.

### 4.1 Early explorations (10 epochs, full UX model)

| # | Run name | Config | Best val τ | Best ρ | MAE | Time |
|---|---|---|---|---|---|---|
| 1 | (2-epoch warmup) | Random-init "Swin-like" stub, default loss | 0.009 | 0.014 | 0.084 | 2m 20s |
| 2 | (10-epoch random init) | Same stub, 10 epochs | 0.114 | 0.151 | 0.083 | 12m 27s |
| 3 | **swin-pretrained** | Real Swin-T pretrained, same loss | 0.027 | 0.035 | 0.084 | 10m 26s |
| 4 | **swin-rankmix** (recipe E) | Swin-T + `ux_w=0.1` + `0.9 neural mix` + `margin=0.15` | 0.071 | 0.099 | 0.086 | 11m 02s |
| 5 | **swin-frozen-linear** (baseline G) | Swin-T frozen + `Linear(768,1)` + sigmoid only | 0.148 | 0.210 | 0.165 | 4m 27s |
| 6 | **swin-mlp-unfreeze** (recipe J) | Swin-T (unfreeze stages 5-7) + MLP head + diff LR | 0.203 @ ep2 | 0.281 | 0.096 | 4m 38s |

### 4.2 Multi-architecture linear probe baselines (20 epochs, 2026-05-26)

All use frozen backbone + MLP(feat→256→1), `weight_decay=0.1`, `dropout=0.5`, `patience=5`.

| Arch | Trainable params | Best val τ | **Test τ** | Test 95% CI | Test ρ | Test MAE | Sig? | Time |
|---|---|---|---|---|---|---|---|---|
| **efficientnet_b4** | 459,265 | **0.307** | **0.201** | **[0.068, 0.321]** | **0.287** | 0.178 | ✅ | 14m 33s |
| resnet50 | 524,801 | 0.154 | 0.129 | [-0.007, 0.266] | 0.181 | 0.126 | ⚠️ | 4m 35s |
| swin_t (probe) | 197,121 | 0.174 | 0.090 | [-0.047, 0.221] | 0.126 | 0.124 | ❌ | 6m 00s |
| vit_b_16 | 197,121 | 0.167 | 0.075 | [-0.072, 0.220] | 0.107 | 0.154 | ❌ | 9m 20s |
| efficientnet_b0 | 328,193 | 0.167 | 0.027 | [-0.101, 0.149] | 0.045 | 0.153 | ❌ | 7m 24s |

**Sig?**: CI lower bound > 0 (✅), lower bound ≈ 0 (⚠️), lower bound < 0 (❌)

### Key findings from baseline comparison

1. **EfficientNet-B4 is uniquely suited for UI quality** — margin over second-best (resnet50, τ=0.129) is large. Likely because B4's compound scaling (width×depth×resolution) captures both fine UI details (icons, text) and coarse layout structure simultaneously.

2. **Only EfficientNet-B4 is statistically significant** — CI [0.068, 0.321] entirely above zero. All others have CI crossing zero → cannot claim significant correlation. This alone is a strong motivation for domain pretraining.

3. **Val τ overestimates test τ for small val sets** — swin_t val=0.174 but test=0.090. With 100 val samples the noise is high. Bootstrap CI on test is the reliable number.

4. **Swin-T fine-tune (Recipe J, test TBD) vs. Swin-T probe (test τ=0.090)** — fine-tuning matters ~2× for Swin but EfficientNet-B4 *probe* already beats Swin fine-tune. Architecture matters more than fine-tuning strategy on this data.

### Key learning curve from J (best so far)

```
ep1: τ=0.197  ranking=0.154   ← starts strong
ep2: τ=0.203  ranking=0.124   ← BEST val
ep3: τ=0.170  ranking=0.102
ep5: τ=0.130  ranking=0.088
ep7: τ=0.111  ranking=0.064
ep10: τ=0.104 ranking=0.045   ← train ranking ↓ 3.5× but val τ ↓ 2× → overfit
```

## 5. Findings

1. **Full UX model architecture (28.7M params, cross-modal transformer + GNN + attention + rule branches) HURTS performance** on this dataset. Compared to a frozen-backbone + linear head (769 params), the full model gets *worse* τ (0.07 vs 0.15). Likely causes:
   - Random-init 27M params on 800 samples → severe over-parameterization
   - Cross-modal fusion injects noise from underperforming branches (YOLO detects ~1.1 box/image; rule_scores are non-differentiable)
   - rule_head mixed non-trainable signal into final quality

2. **Data has real, learnable signal** — baseline G achieved τ=0.15 in 4 minutes with 769 trainable parameters. The ceiling for this dataset appears to be around **τ ≈ 0.20-0.25** given label narrowness and rater noise.

3. **Pretrained backbone matters more than architecture** — switching from random "Swin-like" to real Swin-T pretrained immediately improved things (recipe J).

4. **Overfitting is the next bottleneck** — recipe J peaks at epoch 2 then degrades. 26M trainable params on 800 samples is still too much.

5. **Ranking loss with margin 0.15 + ux_loss_weight=0.1** is the right balance — earlier configs let MSE dominate and collapse predictions to dataset mean.

6. **`pairwise_ranking_loss` collapses at margin** when predictions are constant — ranking_loss ≈ margin × valid_fraction; this is a signal-strength tell, not "training is broken."

7. **The narrow target distribution (std=0.10)** means MSE alone can never produce informative rankings — even a constant-mean predictor achieves MAE ≈ std.

## 6. Architecture Diagnosis

The current `UXAssessmentModel` in `src/uxqa/models/ux_assessment_model.py` was designed for interpretability (rule-based scoring + multi-modal fusion). But on this dataset, the design causes:

- Random-init `CrossModalTransformer` disrupts Swin features.
- `LayoutBranch` (YOLOv8 + GNN) gets near-empty input — `yolov8n.pt` is COCO-trained, doesn't recognize UI elements (avg 1.1 boxes, conf 0.33).
- `AttentionBranch` saliency head random-init → noise.
- `RuleChecker` is non-differentiable; its output mixed into quality via softmax weights adds variance without trainable signal.

**The simple Swin + MLP head outperforms by 3×.** Either:
- (a) Use the simple model and treat rule scores / attention as post-hoc *explanations* (parallel, not fused), or
- (b) Pretrain branches individually before joint training, or
- (c) Massively regularize the full model.

## 7. Next Steps

  ทางเลือกปรับ J ให้ลด overfit

  ┌─────┬─────────────────────────────────────────────────────────────┬──────────────────────────────┐
  │     │                             วิธี                              │            คาดผล             │
  ├─────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ J1  │ ลด unfreeze: unfreeze_from=7 (เฉพาะ stage สุดท้าย ~4M params) │ ลด overfit, อาจรักษา τ ≈ 0.20 │
  ├─────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ J2  │ เพิ่ม regularization: dropout 0.2→0.5, weight_decay 1e-2→1e-1 │ ลด overfit แต่อาจกด peak      │
  ├─────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ J3  │ Early stop: รัน 3 epoch + เซฟ best                           │ เร็วสุด, lock in 0.20          │
  ├─────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ J4  │ J1 + J2 + รัน 20 epoch ดู learning curve เต็ม                  │ comprehensive                │
  └─────┴─────────────────────────────────────────────────────────────┴──────────────────────────────┘


### Immediate — finalize J recipe (Priority A)

- [ ] **J4 run**: extend to 20 epochs with `unfreeze_from=7` (only final stage, ~4M params), dropout=0.5, weight_decay=1e-1, see if peak τ is sustainable past epoch 2
- [ ] If overfit persists: early-stop callback (`patience=3` on val τ)
- [ ] Test-set evaluation: report τ/ρ/MAE on the 100 held-out test images using best.pt
- [ ] Save train/val curves to CSV for plotting

### Architecture decisions (Priority B)

- [ ] Decide: keep full multi-modal model with simplified head, OR retire it in favor of clean Swin+MLP. Recommend the latter unless multi-modal is a thesis requirement.
- [ ] If keeping full model: **freeze CrossModalTransformer + GNN + attention encoder initially**, train only Swin tail + head until τ ≥ 0.20, then unfreeze gradually.
- [ ] Replace YOLOv8 COCO weights with a UI-element detector (PubLayNet / WebUI fine-tune) — current detector is essentially dead weight.

### Code/infra (Priority C)

- [ ] Add `--epochs`, `--lr`, `--batch` CLI args to `scripts/train.py` and `train_j.py`
- [ ] Add `scripts/eval.py` that loads `best.pt` and reports test metrics + per-rule F1
- [ ] Fix `scripts/infer.py` to accept `--image <path>` (currently still uses `torch.randn`)
- [ ] Remove stale `data/manifests/*.csv` and `data/{interim,processed}/*` directories from `setup_dataset_scaffold.py` (datasets unused)
- [ ] Add wandb logging back when committing to a recipe

### Stretch (Priority D)

- [ ] Augmentations beyond color jitter + flip: random crop, mixup, cutout (small dataset → needs aug)
- [ ] Try `swin_s` or `swin_b` once recipe is locked
- [ ] Label smoothing for quality_score (account for rater noise)
- [ ] Ensembling: average 3-5 seeds for final test report

## 8. Files of Interest

| File | Purpose |
|---|---|
| `scripts/train.py` | Full UX model trainer (currently underperforming) |
| `scripts/train_baseline.py` | Recipe G — frozen Swin + linear head |
| `scripts/train_j.py` | Recipe J — Swin partial unfreeze + MLP head (best so far) |
| `configs/default.yaml` | Full-model config (50 epochs) |
| `configs/smoke.yaml` | 10-epoch smoke config for full model |
| `src/uxqa/data/uicrit_dataset.py` | UICrit+RICO Dataset (auto 80/10/10 split, in-memory) |
| `src/uxqa/utils/metrics.py` | Kendall τ, Spearman, Pearson, MAE, rule_f1, pairwise_ranking_loss |
| `checkpoints/swin-mlp-unfreeze-20260526-0413/best.pt` | Best model so far (τ=0.203) |

## 9. Open Questions

1. Is τ ≈ 0.20-0.25 acceptable for the thesis, or is the bar higher?
2. Is interpretability (per-rule scoring) a hard requirement, or nice-to-have?
3. Is the multi-modal architecture a research contribution (must keep) or a means to an end (can simplify)?
4. Should we collect or generate more training data to break the 800-sample ceiling?

---

## 10. Current Architectures (snapshot)

Two parallel codepaths exist right now. Recipe J is the better performer but is **single-modality** — this conflicts with the thesis title.

### 10.1 Full UX Model — `UXAssessmentModel` (used by `scripts/train.py`)

Multi-modal by design, but 3 of 4 modalities are currently dead weight.

```
Input: screenshot [B, 3, 224, 224]
  │
  ▼
Stage 1: Visual Encoder
  SwinTransformerEncoder = Swin-T (ImageNet1k, pretrained, all-trainable)
    + 1x1 Conv project 768 -> 128
  Output: visual_feature_map [B, 128, 7, 7]
  │
  ├──────────────┬──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
Stage 2-3      Stage 4a       Stage 4b       Stage 5
Layout Branch  Attention      Rule Checker   Visual proj
 - YOLOv8n     Branch          - 7 rules     Linear 128->128
 - GNN(2L)    - Saliency        algorithmic   tokens [B,49,128]
              - AttnEncoder    - NO PARAMS
  │              │              │              │
  │              │   Stage 4c   │              │
  │              │   rule_encoder MLP(7->128) -> rule_token [B,1,128]
  │              │              │              │
  │   layout_token + rule_token = enhanced_layout [B,1,128]
  │              │                              │
  ▼              ▼                              ▼
Stage 5: CrossModalTransformer (4 layers, random-init)
  Concat([visual_tokens, enhanced_layout, attn_token]) -> fused_tokens
  -> fused_cls [B, 128]
  │
  ▼
Stage 6: RuleViolationHead
  weight_head:  fused_cls -> softmax weights [B, 7]
  quality_head: fused_cls -> sigmoid neural_quality [B, 1]
  weighted_score = sum(rule_scores * learned_weights)
  quality_score  = 0.9 * neural_quality + 0.1 * weighted_score
```

**Status of each modality:**
- Visual: works (Swin-T pretrained).
- Structural (Layout): YOLOv8n is COCO-trained → detects ~1.1 boxes/image at conf 0.33; branch contributes near-noise.
- Attention: saliency head is random-init with no supervision → output is noise.
- Rule-based: non-differentiable; mixed back into quality only via learnable softmax weights.

Trainable params: ~28.7M. Best val τ in 10 epochs: **0.071**.

### 10.2 Recipe J — `SwinMLP` (used by `scripts/train_j.py`)

Single-modality. Best performer so far.

```
Input: screenshot [B, 3, 224, 224]
  │
  ▼
Swin-T (ImageNet1k pretrained, torchvision)
  features[0..4]: FROZEN     (~21.7M params, no grad)
  features[5..7]: TRAINABLE  (~6.2M params, lr=1e-5)
  norm:           TRAINABLE  (~1.5K params, lr=1e-5)
  Output: [B, 7, 7, 768]
  │
  ▼ global average pool over (H, W)
  [B, 768]
  │
  ▼ MLP Head (lr=1e-3)
  Linear(768, 256) -> GELU -> Dropout(0.2) -> Linear(256, 1) -> Sigmoid
  │
  ▼
quality_score [B, 1]
```

Loss: `0.1 * MSE + 5.0 * pairwise_ranking_loss(margin=0.15)`
Trainable params: ~6.4M. Best val τ in 10 epochs: **0.203** (epoch 2).

### 10.3 Side-by-side

| | Full UX model | Recipe J |
|---|---|---|
| Modalities | Visual + Structural + Attention + Rule | **Visual only** |
| Backbone | Swin-T (all trainable) | Swin-T (stages 0-4 frozen) |
| Head | RuleViolationHead (neural+rule mix) | MLP 768→256→1 |
| Trainable params | 28.7M | 6.4M |
| Outputs | quality + rules + heatmap + weights | quality only |
| Best val τ | 0.071 | **0.203** |
| Train time (10 ep) | 11m | 5m |
| Matches thesis title? | ✅ Yes | ❌ No (single-modal) |

---

## 11. Novel Method: UI-Domain Pretraining via Pixel-Rule Pseudo-Supervision

**Hypothesis:** The ImageNet→UI domain gap is a key bottleneck. Pretraining EfficientNet-B4
on 66K RICO images using pixel-based UX heuristics as pseudo-labels should yield
UI-specific features that transfer better to human quality annotation.

### Pipeline

```
Stage 1: RICO Pretraining (no human labels)
  generate_pseudo_labels.py  →  data/raw/rico/pixel_rules.csv  (5 scores × 66K images)
  pretrain_rico.py           →  checkpoints/pretrain-rico-effb4/best.pt

Stage 2: UICrit Fine-tuning (800 labeled samples)
  train_proposed.py --mode probe    →  frozen backbone + MLP head
  train_proposed.py --mode finetune →  partial unfreeze + MLP head
```

### Pixel Rule Scores (src/uxqa/utils/pixel_rules.py)

| # | Rule | Computation | Proxy for |
|---|---|---|---|
| 0 | contrast | patch luminance std | WCAG readability |
| 1 | whitespace | bright-pixel fraction in [0.25,0.60] | breathing room |
| 2 | balance | intensity-weighted centroid X | visual balance |
| 3 | simplicity | inverted Sobel edge density | anti-clutter |
| 4 | reading_flow | top-third vs. bottom-third content | F/Z pattern |

### Expected comparison table (to fill after runs)

| Method | Backbone | Stage 1 | Stage 2 | Val τ | Test τ | 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| **ImageNet probe** | EffNet-B4 | ImageNet | frozen + MLP | **0.307** | **0.201** | [0.068, 0.321] | ✅ |
| ImageNet fine-tune | EffNet-B4 | ImageNet | partial unfreeze | 0.318 | 0.148 | [0.005, 0.281] | ⚠️ |
| **RICO pretrain probe** | EffNet-B4 | RICO pixel-rules | frozen + MLP | — | — | — | — |
| **RICO pretrain fine-tune** | EffNet-B4 | RICO pixel-rules | partial unfreeze | — | — | — | — |

> **Critical finding:** Fine-tuning 13.9M backbone params on 800 samples causes severe val→test gap
> (val τ=0.318 but test τ=0.148). Linear probe generalizes better. This motivates domain
> pretraining: RICO-pretrained features should reduce the generalization gap when fine-tuning.

### Run order
1. `python scripts/generate_pseudo_labels.py`  (~5 min for 66K images)
2. `python scripts/pretrain_rico.py --epochs 30`  (~3-4h on RTX 4070S)
3. `python scripts/train_proposed.py --mode probe`
4. `python scripts/train_proposed.py --mode finetune --strong-aug`
5. Also run EffNet-B4 fine-tune baseline: `python scripts/train_proposed.py --pretrain none --mode finetune`

---

## 12. Multi-Modal Decision Paths

**Tension:** Thesis title is *"Multi-Modal Learning of **Visual, Structural, and Attention** Features for UX Quality Assessment"*, but the best-performing recipe (J) is visual-only. Decision needed before committing to a final architecture.

### Path A — Make multi-modal actually work (keep thesis scope)

Fix each modality so it contributes real signal, then fuse.

- **Structural:** replace COCO-pretrained YOLOv8n with a UI-element detector. Options:
  - Fine-tune YOLOv8 on PubLayNet / WebUI / RICO-semantic boxes.
  - Use a pretrained UI detector (e.g., screen-recognition models, Pix2Struct embeddings).
- **Attention:** pretrain saliency on SALICON or use an off-the-shelf saliency model (UNISAL, DeepGaze) instead of random-init.
- **Rule-based:** detach from quality loss; treat as **parallel post-hoc explanation** pathway, not as a training signal.
- Joint training plan: freeze branches individually until each reaches a useful baseline, then unfreeze & fuse with low LR.

Cost: 2–4 weeks of additional work. Highest research value.

### Path B — Drop multi-modal (rename thesis)

Adopt recipe J as the primary model. Reframe scope as *"Vision Transformer for UX Quality Assessment"* or similar single-modal framing.

Cost: low. Requires advisor approval for the title/scope change.

### Path C — Hybrid: visual primary + multi-modal ablation (recommended default)

Use recipe J as the headline model, but report multi-modal experiments as an ablation/negative result. Frame the contribution as:

> *"We empirically study multi-modal fusion (visual + structural + attention) for UX scoring on UICrit, and find that under the data regime studied (n≈1000, narrow rating distribution, weak structural/attention supervision), a strong visual backbone alone outperforms multi-modal fusion. We diagnose the failure modes and propose conditions under which multi-modal fusion would be expected to help."*

Keeps the thesis title honest, delivers a result, and contributes research insight.

Cost: low-medium. Best risk/reward given current evidence.

### Decision dependencies

Three questions to discuss with advisor before committing:

1. Can the thesis title be changed? → if yes, Path B is an option.
2. Is multi-modal the contribution, or just a tool? → if contribution, Path A; if tool, Path C.
3. Are negative/diagnostic results acceptable as contribution? → if yes, Path C is viable as-is.

### Implications for code

- **Path A** → keep `UXAssessmentModel` but refactor each branch; retire current YOLO/saliency; add pretraining scripts.
- **Path B** → promote `train_j.py` → `train.py`; delete `UXAssessmentModel`, `LayoutBranch`, `AttentionBranch`, `CrossModalTransformer`, `RuleChecker` (or archive).
- **Path C** → keep both codepaths; document the comparison; use J for headline metrics, full model for ablation.
