# การเรียนรู้แบบหลายโมดอลจากคุณสมบัติด้านภาพ โครงสร้าง และความสนใจ สำหรับการประเมินคุณภาพ UX

**Multi-Modal Learning of Visual, Structural, and Attention Features for UX Quality Assessment**

---

## ภาพรวมของงานวิจัย

งานวิจัยนี้นำเสนอโมเดลแบบ Multi-Modal สำหรับการประเมินคุณภาพประสบการณ์ผู้ใช้ (UX) ของ UI บนมือถือ โดยอัตโนมัติ โมเดลรับ screenshot ของ UI เป็น input แล้วทำนายคะแนนคุณภาพ UX แบบองค์รวม (holistic quality score) พร้อมอธิบายได้ว่าผิดกฎการออกแบบข้อใดบ้าง

แนวคิดหลักคือการรวม **3 มุมมอง** เข้าด้วยกัน:

| มุมมอง | Branch | คำอธิบาย |
|--------|--------|-----------|
| ด้านภาพ (Visual) | Swin Transformer Encoder | เข้าใจรูปลักษณ์ภาพรวมของ UI |
| ด้านโครงสร้าง (Structural) | Layout Branch + GNN | เข้าใจตำแหน่งและความสัมพันธ์ของ UI elements |
| ด้านความสนใจ (Attention) | Attention Branch | ทำนายว่าผู้ใช้จะมองที่ใดบนหน้าจอ |

---

## ที่มาและแรงจูงใจ

การออกแบบ UI ที่ดีส่งผลโดยตรงต่อ user engagement และ conversion rate แต่การประเมินคุณภาพ UX ด้วยมนุษย์ใช้เวลานานและมีค่าใช้จ่ายสูง งานวิจัยนี้จึงพัฒนาระบบประเมินอัตโนมัติที่:

- **วัดได้ (Quantifiable)** — ให้คะแนนตัวเลขสอดคล้องกับการประเมินของผู้เชี่ยวชาญ
- **อธิบายได้ (Interpretable)** — บอกได้ว่าผิดกฎการออกแบบข้อใด
- **รวดเร็ว** — ประเมินได้ทันทีจาก screenshot เพียงภาพเดียว

---

## ชุดข้อมูล: UICrit

| คุณสมบัติ | รายละเอียด |
|-----------|-----------|
| แหล่งที่มา | Google Research (ตีพิมพ์ใน UIST 2024) |
| ชุดข้อมูลพื้นฐาน | RICO (ฐานข้อมูล UI มือถือ) |
| จำนวน UI | 983 ภาพ |
| จำนวน critiques | 3,059 รายการ (มี bounding box กำกับ) |
| ผู้ annotate | นักออกแบบมืออาชีพ 7 คน (ประสบการณ์ ≥ 1 ปี) |
| คะแนน | ระดับ 1–5 ต่อ UI หนึ่งภาพ |

### การแบ่ง Dataset

| ชุด | จำนวน UI | จำนวน Critiques |
|-----|---------|----------------|
| Train | 786 (80%) | ~2,450 |
| Val | 98 (10%) | ~305 |
| Test | 99 (10%) | ~304 |

---

## สถาปัตยกรรมโมเดล (6 ขั้นตอน)

```
Screenshot [B,3,H,W]  +  UI Hierarchy (optional)
       │                          │
       ▼                          ▼
[Stage 1] Visual Encoder   [Parser] Hierarchy Parser
       │                          │
       ▼                          ▼
[Stage 2-3] Layout Branch ◄──────┘
       │
       ├──────────────────► [Stage 4a] Attention Branch
       │                          │
       │                    [Stage 4b] Rule Checker (7 rules)
       │                          │
       │                    [Stage 4c] Rule Token Encoder
       │                          │
       └──────────────────────────┘
                    │
             [Stage 5] Cross-Modal Transformer
                    │
             [Stage 6] Quality Head
                    │
         quality_score + rule_scores
```

---

## รายละเอียดแต่ละขั้นตอน

### Stage 1 — Visual Encoder (Swin Transformer)

**ทำอะไร:** แปลง screenshot เป็น visual feature map ที่มีความหมายเชิงพื้นที่

**ใช้อะไร:** Swin Transformer (hierarchical vision transformer)

**ทำอย่างไร:**
- รับ screenshot รูปแบบ FloatTensor `[B, 3, H, W]`
- Swin Transformer แบ่งภาพเป็น patches แล้วคำนวณ self-attention ภายใน windows เลื่อนที่
- ผลลัพธ์คือ feature map `[B, C, H', W']` ที่เข้ารหัสทั้งรูปแบบภาพและบริบทเชิงพื้นที่
- Feature map นี้ถูกส่งต่อไปทั้ง Layout Branch และ Attention Branch

**ทำไมใช้ Swin:** Swin Transformer เหมาะกับ UI เพราะ UI มีโครงสร้างแบบลำดับชั้น (headers, buttons, content) ซึ่ง window-based attention จับรูปแบบในระดับท้องถิ่นและโกลบอลได้พร้อมกัน

---

### Stage 2 — Layout Detector (YOLOv8)

**ทำอะไร:** ตรวจจับ UI elements (ปุ่ม, ข้อความ, รูปภาพ ฯลฯ) และให้พิกัด bounding box

**ใช้อะไร:** YOLOv8 detector (รองรับทั้ง pretrained model และ placeholder)

**ทำอย่างไร:**
- รับ screenshot ต้นฉบับและ visual feature map
- YOLOv8 ทำนาย bounding boxes `[N, 4]` และ confidence scores `[N]` ในรูป normalized xyxy
- ถ้ามี UI Hierarchy จาก accessibility tree จะใช้พิกัดจาก hierarchy แทน detector
- จำกัดสูงสุด `max_elements = 16` element ต่อภาพ

**Fallback:** ถ้าไม่มี YOLOv8 ระบบจะ fallback เป็น placeholder ที่ใช้ activation peaks จาก visual feature map แทน

---

### Stage 2 (ต่อ) — Graph Builder

**ทำอะไร:** สร้าง adjacency matrix แทนความสัมพันธ์เชิงพื้นที่ระหว่าง UI elements

**ใช้อะไร:** Euclidean distance บน element centers

**ทำอย่างไร:**
```
centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5   # หา center ของแต่ละ box
dist = torch.cdist(centers, centers, p=2)         # คำนวณ pairwise Euclidean distance
adjacency = 1.0 / (1.0 + dist)                    # แปลงเป็น proximity (ใกล้ → ค่าสูง)
```
- Elements ที่อยู่ใกล้กันมี edge weight สูง
- Diagonal = 1.0 (self-loop)

---

### Stage 3 — GNN Encoder

**ทำอะไร:** ประมวลผล graph ของ UI elements เพื่อสร้าง layout token

**ใช้อะไร:** Simple GNN (Graph Neural Network) แบบ normalized adjacency propagation

**ทำอย่างไร:**
- รับ node features `[N, in_dim]` และ adjacency matrix `[N, N]`
- ทำ graph propagation 2 รอบ:
  ```
  norm_adj = adjacency / degree          # normalize adjacency
  h = norm_adj @ h                        # aggregate neighbors
  h = Linear(h) → GELU()                 # transform
  ```
- Mean-pool node features → layout token `[B, 1, dim]`
- คำนวณ `graph_density` = จำนวน elements / พื้นที่หน้าจอ (วัดความหนาแน่นของ UI)

---

### Stage 4a — Attention Branch

**ทำอะไร:** ทำนาย saliency heatmap (แผนที่ความสนใจของผู้ใช้) และสร้าง attention token

**ใช้อะไร:**
- `SaliencyPredictor` — Convolutional network
- `AttentionMapEncoder` — Convolutional encoder + Global Average Pooling

**ทำอย่างไร:**

**SaliencyPredictor:**
```
Conv2d(in_dim → in_dim/2, 3×3) → GELU
Conv2d(in_dim/2 → 1, 1×1) → Sigmoid
```
- รับ visual feature map `[B, C, H', W']`
- ผลลัพธ์คือ heatmap `[B, 1, H', W']` ค่า 0–1 (สว่าง = ดึงดูดความสนใจมาก)

**AttentionMapEncoder:**
```
Conv2d(1 → out_dim/2, 3×3) → GELU
Conv2d(out_dim/2 → out_dim, 3×3) → GELU
AdaptiveAvgPool2d(1) → Flatten
```
- แปลง heatmap เป็น attention embedding `[B, out_dim]`
- Project เป็น attention token `[B, 1, dim]`

---

### Stage 4b — Rule Checker (7 กฎ UX)

**ทำอะไร:** ประเมิน 7 กฎการออกแบบ UX แบบ algorithmic ให้คะแนน 0–1

**ใช้อะไร:** Geometric/algorithmic rules (ไม่มี gradient — เป็น hard rules)

| Index | กฎ | คำอธิบาย | วิธีคำนวณ |
|-------|-----|---------|---------|
| 0 | `contrast` | ความต่างของสี (WCAG AA) | color contrast ratio ≥ 4.5 |
| 1 | `whitespace` | พื้นที่ว่างรอบ elements | พื้นที่ว่าง / พื้นที่รวม อยู่ใน [0.25, 0.60] |
| 2 | `visual_balance` ★ | สมดุลภาพ | centroid ของ elements อยู่ใกล้กึ่งกลางแนวนอน |
| 3 | `density` | ความหนาแน่นของ elements | จำนวน elements / พื้นที่ |
| 4 | `alignment` | การจัดเรียง elements ตาม grid | edges ของ elements align กัน |
| 5 | `cta_prominence` ★ | ปุ่ม CTA ต้องเด่นชัด | ขนาด CTA > ค่าเฉลี่ย elements |
| 6 | `reading_flow` ★ | ลำดับการอ่าน (F/Z pattern) | elements สำคัญอยู่บน-ซ้าย |

**★ = ใช้ heatmap จาก Attention Branch เสริม** — ทำให้ประเมินได้แม่นขึ้น

---

### Stage 4c — Rule Token Encoder

**ทำอะไร:** แปลง rule_scores เป็น rule_token เพื่อส่งเข้า Transformer

**ใช้อะไร:** MLP (Multi-Layer Perceptron)

```
rule_scores [B, 7] → MLP → rule_token [B, 1, dim]
enhanced_layout_token = layout_token + rule_token
```

---

### Stage 5 — Cross-Modal Transformer

**ทำอะไร:** รวมข้อมูลจากทุก branch เข้าด้วยกันผ่าน attention mechanism

**ใช้อะไร:** Transformer Encoder

**Inputs:**
| Token | Shape | แหล่งที่มา |
|-------|-------|-----------|
| visual_tokens | `[B, HW, dim]` | Swin Transformer (projected) |
| enhanced_layout_token | `[B, 1, dim]` | layout + rule info |
| attention_token | `[B, 1, dim]` | saliency-guided gate |
| graph_density | `[B]` | layout complexity scalar |

**กลไกสำคัญ:**
- **Attention Gate:** `attention_gate(attention_token) × visual_tokens` — ให้น้ำหนัก visual features ตามบริเวณที่ผู้ใช้สนใจ
- **Layout Gate:** `layout_gate(layout_token) × graph_density` — ปรับตาม layout complexity
- Transformer Encoder ประมวลผล → `fused_tokens [B, S, dim]`
- ใช้ CLS token: `fused_cls = fused_tokens[:, 0, :]`

---

### Stage 6 — Quality Head

**ทำอะไร:** แปลง fused features เป็นคะแนนคุณภาพ UX สุดท้าย

**ใช้อะไร:** Linear layers + Sigmoid

**ทำอย่างไร:**
```
weight_head(fused_cls)  → learned_weights [B, 7]   (Softmax)
quality_head(fused_cls) → neural_quality  [B, 1]   (Sigmoid)

weighted_score = Σ(rule_scores × learned_weights)
quality_score  = 0.5 × neural_quality + 0.5 × weighted_score
```

**คะแนนสุดท้ายเป็น Hybrid 50/50:**
- 50% จาก neural network (จับรูปแบบที่ซับซ้อน)
- 50% จาก rule-weighted score (อธิบายได้, interpretable)

---

## Outputs ของโมเดล

| Output | Shape | ประเภท | ใช้สำหรับ |
|--------|-------|--------|---------|
| `quality_score` | `[B, 1]` | learned (hybrid) | ผลลัพธ์หลัก — เปรียบกับ UICrit rating |
| `rule_scores` | `[B, 7]` | algorithmic | ประเมิน F1 ต่อกฎ + อธิบายผล |
| `rule_weights` | `[B, 7]` | learned | ablation study |
| `attention_heatmap` | `[B, 1, H', W']` | predicted | visualize + เสริม 3 กฎ ★ |
| `layout_embedding` | `[B, dim]` | learned | วิเคราะห์ features |

---

## การประเมินผล (Evaluation)

### Primary — คะแนนสหสัมพันธ์กับผู้เชี่ยวชาญ

| Metric | คำอธิบาย | เป้าหมาย |
|--------|---------|---------|
| **Kendall's Tau τ** | Rank correlation (มาตรฐานสำหรับ quality scoring) | τ > 0.30 ถือว่าดี |
| **Spearman's ρ** | Monotonic correlation | — |
| **Pearson r** | Linear correlation | — |
| **MAE** | Mean Absolute Error | — |

### Secondary — Per-Rule F1

ประเมินว่าโมเดลตรวจจับการละเมิดกฎแต่ละข้อได้แม่นแค่ไหน โดยเทียบกับ critiques ของผู้เชี่ยวชาญ

- **threshold:** `rule_score < 0.5` → จัดเป็น violation
- ประเมิน Precision, Recall, F1 ต่อกฎ

---

## การเปรียบเทียบกับ Baseline

| โมเดล | คำอธิบาย |
|-------|---------|
| **UIClip** | CLIP-based UI quality model (UIST 2024) |
| **ResNet-50 Regression** | CNN ธรรมดา → คะแนน scalar |
| **Rule-based Only** | ใช้ rule_scores เฉลี่ยอย่างเดียว ไม่มี ML |
| **Ours (full)** | Multi-modal + rules + saliency (งานนี้) |

---

## Ablation Study

| Variant | สิ่งที่ถอดออก | ผลที่คาดหวัง |
|---------|-------------|-------------|
| w/o Attention Branch | ไม่มี heatmap, ไม่มี attention_token | กฎ ★ ทั้ง 3 ใช้ geometry เท่านั้น |
| w/o GNN | layout_token = mean-pool features | สูญเสีย graph structure |
| w/o Rule Head | quality_score = neural เท่านั้น | สูญเสีย interpretability |
| **Full model** | ทุก component ทำงาน | performance ดีที่สุด |

---

## การฝึกโมเดล (Training)

| Parameter | ค่า |
|-----------|-----|
| Primary loss | MSE(quality_score, normalized_uicrit_rating) |
| Secondary loss | BCE(rule_scores, critique_rule_labels) × 0.5 |
| Loss รวม | `L = L_quality + 0.5 × L_rules` |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Epochs | 50 |

---

## โครงสร้างไฟล์

```
src/uxqa/
├── models/
│   ├── backbones/
│   │   └── swin_encoder.py          Stage 1 — Visual Encoder
│   ├── layout/
│   │   ├── detector.py              Stage 2 — YOLOv8 adapter
│   │   ├── graph_builder.py         Stage 2 — สร้าง adjacency matrix
│   │   ├── gnn_encoder.py           Stage 3 — GNN encoder
│   │   └── branch.py                Stage 2–3 orchestrator
│   ├── attention/
│   │   ├── saliency.py              Stage 4a — SaliencyPredictor (Conv)
│   │   ├── encoder.py               Stage 4a — AttentionMapEncoder
│   │   └── branch.py                Stage 4a — AttentionBranch
│   ├── rules/
│   │   └── definitions.py           กฎ UX 7 ข้อ + ค่าน้ำหนัก
│   ├── fusion/
│   │   └── cross_modal_transformer.py  Stage 5 — Cross-Modal Transformer
│   ├── heads/
│   │   └── ux_head.py               Stage 6 — Quality Head
│   └── hierarchy/
│       └── parser.py                HierarchyParser — แปลง accessibility tree
├── data/
│   ├── dataset_config.py            UICrit loader + critique-to-rule mapping
│   └── converters.py                แปลง critique text → rule labels
└── utils/
    └── visualizer.py                แสดงผล heatmap + rule violations
scripts/
├── download_datasets.py             ดาวน์โหลด UICrit dataset
├── setup_dataset_scaffold.py        สร้างโครงสร้าง dataset
└── convert_dataset_templates.py     แปลงรูปแบบ dataset
```

---

## Critique-to-Rule Mapping

UICrit critique text จะถูก map เป็นหมวดกฎด้วย keyword matching:

| กฎ | Keyword ที่ใช้ตรวจจับ |
|----|---------------------|
| `contrast` | contrast, readability, legibility, color, dark, light |
| `whitespace` | spacing, padding, crowded, empty, margin, gap |
| `visual_balance` | balance, symmetry, centered, heavy, weight |
| `density` | cluttered, too many, overwhelming, busy, elements |
| `alignment` | aligned, misaligned, grid, column, edge, offset |
| `cta_prominence` | CTA, call to action, button, prominent, visible |
| `reading_flow` | hierarchy, order, flow, scan, F-pattern, Z-pattern |

---

## เหตุผลของการออกแบบ (Design Decisions)

**ทำไมไม่สร้าง dataset ใหม่?**
UICrit เป็น dataset ที่ผ่าน peer-review จาก UIST 2024 มีทั้งคะแนนคุณภาพ (สำหรับ primary metric) และ critique text พร้อม bounding box (สำหรับ secondary evaluation) ครบในชุดเดียว

**ทำไมต้องมี Attention Branch?**
Saliency heatmap ให้หลักฐานเชิงพื้นที่สำหรับ 3 กฎที่ geometry อย่างเดียวไม่เพียงพอ ได้แก่ visual balance, CTA prominence, reading flow

**ทำไมใช้ hybrid scoring (50% neural + 50% rule)?**
- Neural เพียงอย่างเดียว: black box, ไม่ interpretable
- Rule เพียงอย่างเดียว: ตรวจไม่ได้รูปแบบที่ซับซ้อน
- Hybrid: สมดุลระหว่างประสิทธิภาพและความสามารถอธิบายผล

**ทำไมใช้ Kendall's Tau เป็น primary metric?**
วัด rank agreement ซึ่งคือสิ่งที่ quality scoring ต้องการจริงๆ ไม่ใช่การทำนายค่าตัวเลขที่แม่นยำ และเป็น metric มาตรฐานใน UIClip ทำให้เปรียบเทียบโดยตรงได้

---

## เทคโนโลยีที่ใช้

| เทคโนโลยี | เวอร์ชัน | บทบาท |
|-----------|---------|-------|
| Python | 3.12+ | ภาษาโปรแกรมหลัก |
| PyTorch | 2.x | Deep learning framework |
| Swin Transformer | — | Visual backbone |
| YOLOv8 (ultralytics) | — | UI element detector |
| GNN (custom) | — | Graph-based layout encoding |
| RICO dataset | — | ฐานข้อมูล UI screenshots |
| UICrit dataset | UIST 2024 | ชุดข้อมูลหลักพร้อม human ratings |
