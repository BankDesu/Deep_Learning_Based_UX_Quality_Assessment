# Experiment Metrics — UX Quality Assessment

## ความหมายของ Metrics แต่ละตัว

### Primary Metric

| Metric | สูตร | ความหมาย | Range | ตีความ |
|---|---|---|---|---|
| **Kendall's τ (tau)** | concordant − discordant pairs / total pairs | rank correlation ระหว่าง predicted score กับ human rating | [-1, 1] | 0 = ไม่มี correlation, 1 = perfect rank order, ใช้เป็น primary เพราะ robust กับ outlier และ dataset เล็ก |

**ทำไมใช้ Kendall τ เป็นหลัก:** task ของเราคือ *ranking* UIs ตาม quality ไม่ใช่ predict ค่า absolute ดังนั้น rank correlation สำคัญกว่า absolute error

### Secondary Metrics

| Metric | สูตร | ความหมาย | Range | ตีความ |
|---|---|---|---|---|
| **Spearman's ρ (rho)** | Pearson r ของ ranks | rank correlation แบบอีกวิธี | [-1, 1] | สัมพันธ์กับ τ แต่ sensitive กว่าต่อ outlier, ρ ≈ 1.5 × τ โดยประมาณ |
| **Pearson r** | cov(pred, target) / (σ_pred × σ_target) | linear correlation | [-1, 1] | วัด linear relationship ไม่ใช่ rank — ถ้า pred มี scale ผิด Pearson จะต่ำแม้ rank ถูก |
| **MAE** | mean(|pred − target|) | absolute error เฉลี่ย | [0, 1] | target std = 0.10 → MAE ≤ 0.10 แปลว่าดีกว่า constant mean predictor, MAE > 0.15 แย่กว่า naive baseline |
| **95% Bootstrap CI** | percentile bootstrap 1000 samples | ช่วงความเชื่อมั่นของ test τ | — | ถ้า CI lower bound > 0 = significant (❌ ถ้า CI crosses zero) |
| **pred_std** | std(predictions บน test set) | ความกระจายของ predictions | [0, 1] | ถ้าใกล้ 0 = model collapse (predict ค่าเดียว), target std = 0.10 |

### Pretraining Metric

| Metric | สูตร | ความหมาย |
|---|---|---|
| **val_mse** | mean((pred − pseudo_label)²) | MSE ระหว่าง predicted pseudo-label กับ label จริง, ใช้เฉพาะ Stage 1 pretrain ไม่ใช่ downstream quality metric |

---

## ผล Experiments ทั้งหมด

### Architecture Comparison (ImageNet probe — frozen backbone + MLP head)

> Dataset: UICrit 800 train / 100 val / 100 test  
> Setup: backbone frozen, head = Linear(feat_dim→256)→GELU→Dropout(0.5)→Linear(256→1)→Sigmoid  
> Loss: 0.1 × MSE + 5.0 × pairwise_ranking_loss(margin=0.15)  
> Checkpoint: `checkpoints/baselines/<arch>/best.pt`

| Architecture | Params (head) | Best val τ | Test τ | 95% CI | Test ρ | Test MAE | Significant? |
|---|---|---|---|---|---|---|---|
| **efficientnet_b4** | 459,265 | **0.307** | **0.201** | **[0.068, 0.321]** | **0.287** | 0.178 | ✅ Yes |
| resnet50 | 524,801 | 0.154 | 0.129 | [-0.007, 0.266] | 0.181 | 0.126 | ⚠️ Borderline |
| swin_t | 197,121 | 0.174 | 0.090 | [-0.047, 0.221] | 0.126 | 0.124 | ❌ No |
| vit_b_16 | 197,121 | 0.167 | 0.075 | [-0.072, 0.220] | 0.107 | 0.154 | ❌ No |
| efficientnet_v2_s | 328,193 | 0.137 | 0.067 | [-0.083, 0.211] | 0.094 | 0.140 | ❌ No |
| efficientnet_b0 | 328,193 | 0.167 | 0.027 | [-0.101, 0.149] | 0.045 | 0.153 | ❌ No |

**Checkpoint keys:** `epoch`, `tau` (best val τ), `arch`, `model` (state_dict)  
**Log:** `checkpoints/baselines/<arch>/train_log.csv` — columns: `epoch, train_loss, val_tau, val_rho, val_mae, pred_std, is_best`

---

### RICO CLIP Pseudo-Label Generation

> `scripts/generate_clip_labels.py` — CLIP ViT-B/32 (open_clip) บน 66K RICO images  
> Score = mean(pos_sim) − mean(neg_sim) normalized to [0, 1]  
> Output: `data/raw/rico/clip_labels.csv`

| Metric | Value |
|---|---|
| Images scored | 66,261 |
| Speed | ~350 img/s (GPU batch=256) |
| Raw score range | [-0.0261, 0.0120] |
| **CLIP vs UICrit test τ** | **0.089** |
| CLIP vs UICrit test ρ | 0.123 |
| Signal quality | Weak but nonzero (τ > 0.05) |

---

### RICO Pixel-Rule Pretraining (Stage 1 → Stage 2)

> Stage 1: EfficientNet-B4 + MLP head predicting 5 pixel-rule scores บน 66K RICO images  
> Stage 2: Load backbone, freeze/unfreeze, train MLP head บน UICrit

**Stage 1 pretrain** — `checkpoints/pretrain-rico-effb4/`

| File | Keys | ความหมาย |
|---|---|---|
| `best.pt` | `epoch=30`, `val_mse=0.00135`, `model` | Best checkpoint ตาม val MSE |
| `pretrain_log.csv` | `epoch, train_mse, val_mse` | Learning curve 30 epochs |

val_mse ลดจาก 0.009 → **0.00135** (converged ดี)

**Stage 2 downstream** — `checkpoints/proposed-*/`

| Run | Mode | Best val τ | Test τ | 95% CI | Test ρ | Test MAE | Significant? |
|---|---|---|---|---|---|---|---|
| `proposed-probe-20260527-1542` | probe (frozen) | 0.164 | 0.016 | [-0.133, 0.178] | 0.018 | 0.522 | ❌ No |
| `proposed-finetune-20260527-1545` | finetune (partial unfreeze) | 0.187 | 0.051 | [-0.097, 0.194] | 0.072 | 0.124 | ❌ No |

**Checkpoint keys:** `epoch`, `tau` (best val τ), `pretrain` (path), `mode`, `model`  
**Log:** `train_log.csv` — columns: `epoch, train_loss, val_tau, val_rho, val_mae, is_best`

**Finding:** RICO pixel-rule pretraining *hurt* performance vs ImageNet baseline (test τ 0.016–0.051 vs 0.201). Pixel-based pseudo-labels ไม่ align กับ human quality judgment.

---

### RICO CLIP Pretraining (Stage 1 → Stage 2)

> Stage 1: EfficientNet-B4 + MLP head predicting CLIP quality score (1 target) บน 66K RICO images  
> Stage 2: Load backbone, freeze/unfreeze, train MLP head บน UICrit  
> CLIP labels: τ=0.089 vs UICrit test (weak signal)

**Stage 1 pretrain** — `checkpoints/pretrain-rico-clip/`

val_mse ลดจาก 0.00959 → **0.00636** (converged, 30 epochs, 3h)

**Stage 2 downstream**

| Run | Mode | Best val τ | Test τ | 95% CI | Test ρ | Test MAE | Significant? |
|---|---|---|---|---|---|---|---|
| `clip-probe-20260528-0332` | probe (frozen) | 0.236 | 0.019 | [-0.114, 0.150] | 0.033 | 0.231 | ❌ No |
| `clip-finetune-20260528-1128` | finetune (partial unfreeze) | 0.248 | -0.053 | [-0.206, 0.098] | -0.065 | 0.241 | ❌ No |

**Finding:** CLIP-based pretraining บน RICO ก็ *hurt* performance เช่นกัน (test τ -0.053–0.019 vs 0.201). แม้ CLIP labels จะมี signal กับ human ratings (τ=0.089) แต่ไม่เพียงพอให้ backbone เรียนรู้ features ที่ transfer ได้ดี. **ImageNet pretraining ยังคงเป็น best approach สำหรับ task นี้.**

---

### Multi-Modal Fusion (Visual + Structural)

> Visual: EffNet-B4 (ImageNet frozen/partial-unfreeze)  
> Structural: 19 pixel-derived layout features (contrast, whitespace, color entropy, symmetry, 3×3 grid density, etc.)  
> Scripts: `scripts/train_multimodal.py`

**Ablation — probe mode (backbone frozen)**

| Run | Fusion | Test τ | 95% CI | Sig? |
|---|---|---|---|---|
| `multimodal-probe-visual-only` | visual only (same params) | 0.201 | [0.083, 0.321] | ✅ |
| `multimodal-probe-struct-only` | structural only | 0.003 | [-0.133, 0.142] | ❌ |
| `multimodal-probe` (GatedFusion 990K) | gated fusion | 0.079 | [-0.065, 0.218] | ❌ |
| `multimodal-probe` (additive 160 params) | additive correction | 0.067 | [-0.083, 0.204] | ❌ |

**Ablation — finetune mode (partial backbone unfreeze)**

| Run | Fusion | Test τ | 95% CI | Sig? |
|---|---|---|---|---|
| `multimodal-finetune` (GatedFusion) | gated fusion | 0.164 | [0.028, 0.292] | ✅ |
| `multimodal-finetune` (additive) | additive correction | 0.140 | [-0.013, 0.276] | ⚠️ |

**Finding:** Structural features คนเดียวไม่มี signal (τ=0.003). การรวม structural เข้า fusion ทุกแบบไม่ดีกว่า visual-only (0.201). สาเหตุ: dataset เล็ก (800 train) + structural features ไม่ align กับ human quality judgment → **UX quality prediction เป็น semantic task ที่ visual features dominate**

---

### Other Runs (Early Exploration — Swin-T backbone)

> ใช้ UXAssessmentModel (full multi-modal) หรือ SwinMLP — ดูรายละเอียดใน EXPERIMENTS.md

| Checkpoint | Best val τ | Test τ | หมายเหตุ |
|---|---|---|---|
| `swin-mlp-unfreeze-20260526-0413` | 0.203 | *(ยังไม่ได้รัน test eval)* | Recipe J: Swin-T partial unfreeze, peak at epoch 2 |
| `swin-rankmix-20260526-0333` | 0.071 | — | Full UX model + rank loss |

---

## ตาราง Comparison สมบูรณ์ (สำหรับ Paper)

| Method | Backbone | Features | Stage 2 | Test τ | 95% CI | Sig? |
|---|---|---|---|---|---|---|
| **ImageNet probe** | EffNet-B4 | visual only | frozen+MLP | **0.201** | [0.068, 0.321] | ✅ |
| ImageNet probe | ResNet-50 | visual only | frozen+MLP | 0.129 | [-0.007, 0.266] | ⚠️ |
| ImageNet probe | Swin-T | visual only | frozen+MLP | 0.090 | [-0.047, 0.221] | ❌ |
| ImageNet probe | ViT-B/16 | visual only | frozen+MLP | 0.075 | [-0.072, 0.220] | ❌ |
| ImageNet probe | EffNet-V2-S | visual only | frozen+MLP | 0.067 | [-0.083, 0.211] | ❌ |
| ImageNet probe | EffNet-B0 | visual only | frozen+MLP | 0.027 | [-0.101, 0.149] | ❌ |
| RICO pixel-rule finetune | EffNet-B4 | pixel rules (RICO) | partial unfreeze | 0.051 | [-0.097, 0.194] | ❌ |
| RICO pixel-rule probe | EffNet-B4 | pixel rules (RICO) | frozen+MLP | 0.016 | [-0.133, 0.178] | ❌ |
| RICO CLIP probe | EffNet-B4 | CLIP scores (RICO) | frozen+MLP | 0.019 | [-0.114, 0.150] | ❌ |
| RICO CLIP finetune | EffNet-B4 | CLIP scores (RICO) | partial unfreeze | -0.053 | [-0.206, 0.098] | ❌ |
| Multi-modal finetune (gated) | EffNet-B4 | visual+structural | partial unfreeze | 0.164 | [0.028, 0.292] | ✅ |
| Multi-modal finetune (additive) | EffNet-B4 | visual+structural | partial unfreeze | 0.140 | [-0.013, 0.276] | ⚠️ |
| Multi-modal probe (gated) | EffNet-B4 | visual+structural | frozen+MLP | 0.079 | [-0.065, 0.218] | ❌ |
| Struct-only probe | EffNet-B4 | structural only | frozen | 0.003 | [-0.133, 0.142] | ❌ |

---

## ไฟล์ที่เกี่ยวข้อง

| Path | ประเภท | ใช้สำหรับ |
|---|---|---|
| `checkpoints/baselines/summary.csv` | CSV | ตาราง baseline ทุก arch รวม |
| `checkpoints/baselines/<arch>/best.pt` | PyTorch | Model weights ที่ดีที่สุดของแต่ละ arch |
| `checkpoints/baselines/<arch>/train_log.csv` | CSV | Learning curve per epoch |
| `checkpoints/pretrain-rico-effb4/best.pt` | PyTorch | RICO-pretrained EffNet-B4 backbone |
| `checkpoints/pretrain-rico-effb4/pretrain_log.csv` | CSV | Pretrain learning curve |
| `checkpoints/proposed-*/best.pt` | PyTorch | Downstream models (probe/finetune) |
| `checkpoints/proposed-*/train_log.csv` | CSV | Downstream learning curves |
| `data/raw/rico/pixel_rules.csv` | CSV | Pseudo-labels: 5 pixel-rule scores × 66K images |
| `data/raw/rico/clip_labels.csv` | CSV | CLIP quality scores × 66,261 images (τ=0.089 vs UICrit test) |
| `checkpoints/pretrain-rico-clip/best.pt` | PyTorch | CLIP-pretrained EffNet-B4 backbone |
| `checkpoints/pretrain-rico-clip/pretrain_log.csv` | CSV | CLIP pretrain learning curve |
