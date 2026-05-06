# UX Assessment Model — Architecture & Dataset

**Multi-Modal Learning of Visual, Structural, and Attention Features for UX Quality Assessment**

Task: **UI Quality Scoring** — predict a holistic UX quality score aligned with human expert ratings, with interpretable per-rule violation breakdown.

Dataset: **UICrit** (UIST 2024) — 983 mobile UIs from RICO with human quality ratings and 3,059 element-level design critiques.

---

## Task Framing

```
Input  : Mobile UI screenshot  (+ optional UI hierarchy)
Output : quality_score ∈ [0,1]   →  correlated with human expert rating
         rule_scores   [9]        →  per-rule violation breakdown
         attention_heatmap        →  predicted saliency (for visualization)

Evaluation:
  Primary   — Kendall's Tau τ,  Spearman's ρ   (quality score vs. human rating)
  Secondary — Precision / Recall / F1 per rule  (rule_scores vs. critique labels)
  Optional  — Localization IoU ≥ 0.5            (violation bbox vs. critique bbox)
```

---

## Architecture Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Inputs                                                                  │
│  ┌─────────────────────────┐   ┌──────────────────────────────────────┐  │
│  │  Screenshot             │   │  UI Hierarchy  (optional)            │  │
│  │  FloatTensor [B,3,H,W]  │   │  List[List[Dict]]                    │  │
│  │  from RICO via UICrit   │   │  XML / Accessibility Tree            │  │
│  └────────────┬────────────┘   └─────────────────┬────────────────────┘  │
└───────────────┼─────────────────────────────────┼──────────────────────┘
                │                                 │
                ▼                                 ▼
┌──────────────────────────────┐   ┌─────────────────────────────────────┐
│  STAGE 1 — Visual Encoding   │   │  STAGE 2b — Hierarchy Parser        │
│  SwinTransformerEncoder      │   │  HierarchyParser                    │
│  visual_feature_map          │   │  → bboxes, types, interactive       │
│  [B, C, H, W]                │   │  → button_mask, heading_levels      │
└──────────┬───────────────────┘   └────────────────┬────────────────────┘
           │                                        │
           ▼                                        │
┌──────────────────────────────┐                    │
│  STAGE 2–3 — Layout Branch   │                    │
│  YOLOv8 Detector             │                    │
│  → BBoxes + Element Classes  │                    │
│  Graph Builder               │◄───────────────────┘
│  → Adjacency Matrix          │   (hierarchy bboxes override detector
│  GNN Encoder                 │    bboxes when available)
│  → layout_token [B,1,dim]    │
│  → graph_density [B]         │
└──────────┬───────────────────┘
           │                       ┌──────────────────────────────────────┐
           │                       │  STAGE 4a — Attention Branch         │
           ├──────────────────────►│  SaliencyPredictor (Conv)            │
           │  visual_feature_map   │  → heatmap [B, 1, H', W']  ★        │
           │                       │  AttentionMapEncoder                  │
           │                       │  → attention_token [B, 1, dim]       │
           │                       └────────┬─────────────────┬───────────┘
           │                                │ heatmap         │ attention_token
           │                                ▼                 │
           │              ┌─────────────────────────────────┐ │
           │              │  STAGE 4b — Rule Checker        │ │
           │              │  (algorithmic — no gradients)   │ │
           │              │                                 │ │
           │              │  Rule 0  contrast        (geo)  │ │
           │              │  Rule 1  whitespace      (geo)  │ │
           │              │  Rule 2  visual_balance  (★)    │ │
           │              │  Rule 3  density         (geo)  │ │
           │              │  Rule 4  alignment       (geo)  │ │
           │              │  Rule 5  cta_prominence  (★)    │ │
           │              │  Rule 6  reading_flow    (★)    │ │
           │              │                                 │ │
           │              │  rule_scores [B, 7] ∈ [0,1]    │ │
           │              └──────────────┬──────────────────┘ │
           │                             │                     │
           │                             ▼                     │
           │              ┌──────────────────────────────────┐ │
           │              │  STAGE 4c — Rule Token Encoder   │ │
           │              │  MLP: rule_scores → rule_token   │ │
           │              │  rule_token [B, 1, dim]          │ │
           │              └──────────────┬───────────────────┘ │
           │                             │                     │
           │              enhanced_layout_token =              │
           │              layout_token + rule_token            │
           │                             │                     │
           ▼                             ▼                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Cross-Modal Transformer                                       │
│                                                                          │
│  Inputs:                                                                 │
│    visual_tokens         [B, HW, dim]  ← from Swin (projected)          │
│    enhanced_layout_token [B,  1, dim]  ← layout + rule info             │
│    attention_token       [B,  1, dim]  ← saliency-guided gate           │
│    graph_density         [B]           ← layout complexity scalar       │
│                                                                          │
│  attention_gate(attention_token) × visual_tokens   (saliency guidance)  │
│  layout_gate(layout_token) × graph_density         (layout awareness)   │
│  TransformerEncoder → fused_tokens [B, S, dim]                          │
│  fused_cls = fused_tokens[:, 0, :]    [B, dim]                          │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ fused_cls + rule_scores
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Quality Head                                                  │
│                                                                          │
│  weight_head(fused_cls)  → learned_weights [B, 7]  (softmax)            │
│  quality_head(fused_cls) → neural_quality  [B, 1]  (sigmoid)            │
│                                                                          │
│  weighted_score  = Σ( rule_scores × learned_weights )                   │
│  quality_score   = 0.5 × neural_quality + 0.5 × weighted_score          │
│                                                                          │
│  Training target: UICrit human quality rating (normalized to [0,1])     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
                ┌──────────────┴─────────────────┐
                ▼                                 ▼
  quality_score [B,1]                  rule_scores [B,9]
  → correlated with UICrit rating      → matched against critique labels
  → Kendall's Tau, Spearman's ρ        → Precision / Recall / F1 per rule
```

---

## Dataset: UICrit

| Property | Detail |
|----------|--------|
| Source | Google Research (UIST 2024) |
| Base dataset | RICO (mobile UI screenshots) |
| Size | 983 mobile UI screenshots |
| Critiques | 3,059 design critiques with bounding boxes |
| Annotators | 7 professional designers (≥ 1 yr experience) |
| Quality rating | Per-UI holistic score (1–5 scale) |
| Critique format | Natural language + bbox [x0, y0, x1, y1] |
| Access | github.com/google-research-datasets/uicrit |

### Critique-to-Rule Mapping (Preprocessing)

UICrit critique text is mapped to our 9 rule categories via keyword matching,
enabling evaluation of `rule_scores` against human critique labels.

| Rule | Keywords in critique text |
|------|--------------------------|
| `contrast` | contrast, readability, legibility, color, dark, light |
| `whitespace` | spacing, padding, crowded, empty, margin, gap |
| `visual_balance` | balance, symmetry, centered, heavy, weight |
| `density` | cluttered, too many, overwhelming, busy, elements |
| `alignment` | aligned, misaligned, grid, column, edge, offset |
| `cta_prominence` | CTA, call to action, button, prominent, visible |
| `reading_flow` | hierarchy, order, flow, scan, F-pattern, Z-pattern |

### Data Split

| Split | UIs | Critiques |
|-------|-----|-----------|
| Train | 786 (80%) | ~2,450 |
| Val   |  98 (10%) |   ~305 |
| Test  |  99 (10%) |   ~304 |

---

## Evaluation Protocol

### Primary — Quality Score Correlation

Compare model `quality_score` against UICrit human quality rating (1–5, normalized to [0,1]).

| Metric | Description |
|--------|-------------|
| **Kendall's Tau τ** | Rank correlation — standard for quality scoring tasks |
| **Spearman's ρ** | Monotonic correlation |
| **Pearson r** | Linear correlation |
| **MAE** | Mean Absolute Error against normalized rating |

Target: τ > 0.30 is considered strong for UI quality prediction.

### Secondary — Per-Rule F1

After mapping critiques → rule labels, evaluate per-rule violation detection.

| Metric | Description |
|--------|-------------|
| **Precision** | Of flagged violations, how many match human critiques |
| **Recall** | Of human critiques, how many our model flags |
| **F1** | Harmonic mean |

Threshold: `rule_score < 0.5` → classified as violation.

### Optional — Localization

| Metric | Description |
|--------|-------------|
| **IoU ≥ 0.5** | Violation bbox overlaps with critique bbox |
| **Localization Recall** | % of critique bboxes hit by our violation bboxes |

---

## Baseline Comparison

| Baseline | Description | Primary Metric |
|----------|-------------|----------------|
| **UIClip** | CLIP-based UI quality model (UIST 2024) | Kendall's Tau |
| **ResNet-50 Regression** | Plain CNN → scalar quality score | Kendall's Tau |
| **Rule-based Only** | No ML — only rule_scores averaged | Kendall's Tau |
| **Ours (full)** | Multi-modal + rules + saliency | Kendall's Tau |

---

## Ablation Study

| Variant | Removed Component |
|---------|------------------|
| w/o Attention Branch | No heatmap, no attention_token; 3 ★ rules use geometry only |
| w/o GNN | layout_token = mean-pooled detector features |
| w/o Rule Head | quality_score = pure neural (no weighted rule blend) |
| **Full model** | All components active |

---

## Training

| Property | Detail |
|----------|--------|
| Primary loss | MSE(quality_score, normalized_uicrit_rating) |
| Secondary loss | BCE(rule_scores, critique_rule_labels) × 0.5 |
| Total loss | `L = L_quality + 0.5 × L_rules` |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-2 |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Epochs | 50 |

---

## Output Summary

| Output | Shape | Type | Used For |
|--------|-------|------|----------|
| `quality_score` | `[B, 1]` | learned (hybrid) | Primary metric — correlated with UICrit rating |
| `rule_scores` | `[B, 7]` | algorithmic | Per-rule F1 evaluation + interpretability |
| `rule_weights` | `[B, 7]` | learned | Ablation — shows which rules matter most |
| `attention_heatmap` | `[B, 1, H', W']` | predicted | Visualization + enhances 3 rules (★) |
| `layout_embedding` | `[B, dim]` | learned | Feature analysis |

---

## Design Decisions

**Why UICrit over building a new dataset?**
UICrit is a publicly available, peer-reviewed dataset from UIST 2024 with professional designer annotations and element-level bounding boxes. It provides both quality ratings (for primary evaluation) and critique text with bboxes (for secondary rule-level evaluation), avoiding the need for costly human annotation.

**Why keep AttentionBranch?**
The predicted saliency heatmap provides spatial evidence for 3 rules that geometry alone cannot reliably assess (visual balance, CTA prominence, reading flow). The 50/50 blend ensures geometric rules still function without hierarchy data.

**Why hybrid scoring (50% neural + 50% rule-weighted)?**
Pure rule-based scoring misses holistic quality aspects. Pure neural scoring is a black box. The hybrid preserves interpretability while learning patterns beyond hand-crafted rules — and allows ablation that isolates each contribution.

**Why Kendall's Tau as primary metric?**
Kendall's Tau measures rank agreement, which is what quality scoring fundamentally requires — not exact score prediction. It is also the standard metric in UIClip and related works, enabling direct comparison.

---

## File Structure

```
src/uxqa/
├── models/
│   ├── backbones/
│   │   └── swin_encoder.py          Stage 1 — Visual Encoder
│   ├── layout/
│   │   ├── detector.py              Stage 2 — YOLOv8 adapter
│   │   ├── graph_builder.py         Stage 2 — Adjacency matrix
│   │   ├── gnn_encoder.py           Stage 3 — GNN
│   │   └── branch.py                Stage 2–3 orchestrator
│   ├── attention/
│   │   ├── saliency.py              Stage 4a — SaliencyPredictor
│   │   ├── encoder.py               Stage 4a — AttentionMapEncoder
│   │   └── branch.py                Stage 4a — AttentionBranch
│   ├── rules/
│   │   ├── definitions.py           Rule constants & weights
│   │   ├── checker.py               Stage 4b — 9 UX rules
│   │   └── __init__.py
│   ├── fusion/
│   │   └── cross_modal_transformer.py  Stage 5 — Fusion
│   ├── heads/
│   │   └── rule_head.py             Stage 6 — Quality Head
│   └── ux_assessment_model.py       Main model orchestrator
├── data/
│   ├── uicrit_dataset.py            UICrit loader + critique-to-rule mapper
│   └── transforms.py                Screenshot augmentations
└── utils/
    └── metrics.py                   Kendall's Tau, Spearman's ρ, F1 per rule
```
