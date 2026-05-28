"""Generate IEEE-style PowerPoint presentation."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
from pathlib import Path

C_BLUE   = RGBColor(0x1A, 0x5F, 0x7A)   # deep blue
C_ACCENT = RGBColor(0x39, 0xB5, 0xE0)   # light blue
C_GREEN  = RGBColor(0x27, 0xAE, 0x60)
C_DARK   = RGBColor(0x2C, 0x3E, 0x50)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_GRAY   = RGBColor(0xEC, 0xF0, 0xF1)
C_RED    = RGBColor(0xC0, 0x39, 0x2B)
C_ORANGE = RGBColor(0xE6, 0x7E, 0x22)

FIG = Path("figures")
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]   # completely blank


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, l, t, w, h, fill=None, line=None, line_w=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.fill.background()
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        if line_w:
            shape.line.width = Pt(line_w)
    return shape


def txt(slide, text, l, t, w, h,
        size=18, bold=False, color=C_DARK, align=PP_ALIGN.LEFT,
        italic=False, wrap=True):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.word_wrap = wrap
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return tb


def header_bar(slide, title, subtitle=""):
    rect(slide, 0, 0, 13.33, 1.2, fill=C_BLUE)
    txt(slide, title, 0.3, 0.05, 12, 0.7,
        size=28, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        txt(slide, subtitle, 0.3, 0.72, 12, 0.42,
            size=14, color=C_ACCENT, align=PP_ALIGN.LEFT)


def footer(slide, num):
    rect(slide, 0, 7.1, 13.33, 0.4, fill=C_DARK)
    txt(slide, "Deep Learning-Based UX Quality Prediction | KMUTT | IEEE Access",
        0.3, 7.12, 11, 0.3, size=10, color=C_GRAY, align=PP_ALIGN.LEFT)
    txt(slide, str(num), 12.8, 7.12, 0.5, 0.3,
        size=10, color=C_GRAY, align=PP_ALIGN.RIGHT)


def bullet(slide, items, l, t, w, size=16, color=C_DARK, spacing=0.38):
    for i, item in enumerate(items):
        indent = item.startswith("  ")
        text = item.lstrip()
        bullet_char = "  •  " if not indent else "     –  "
        txt(slide, bullet_char + text, l, t + i * spacing,
            w, spacing + 0.05, size=size if not indent else size - 1,
            color=color if not indent else RGBColor(0x55, 0x66, 0x77))


def add_image(slide, path, l, t, w, h=None):
    p = Path(path)
    if not p.exists():
        return
    if h:
        slide.shapes.add_picture(str(p), Inches(l), Inches(t), Inches(w), Inches(h))
    else:
        slide.shapes.add_picture(str(p), Inches(l), Inches(t), Inches(w))


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
rect(sl, 0, 0, 13.33, 7.5, fill=C_BLUE)
rect(sl, 0, 2.8, 13.33, 2.0, fill=C_DARK)

txt(sl, "Deep Learning-Based UX Quality Prediction",
    0.5, 1.0, 12.3, 1.0, size=30, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txt(sl, "from Mobile Screenshots Using Visual and Structural Cues",
    0.5, 1.85, 12.3, 0.8, size=22, bold=False, color=C_ACCENT, align=PP_ALIGN.CENTER)

txt(sl, "IEEE Access Manuscript Presentation",
    0.5, 3.0, 12.3, 0.5, size=16, color=C_GRAY, align=PP_ALIGN.CENTER)
txt(sl, "King Mongkut's University of Technology Thonburi (KMUTT)",
    0.5, 3.45, 12.3, 0.5, size=14, color=C_GRAY, align=PP_ALIGN.CENTER)

txt(sl, "Dataset: UICrit (983 mobile UIs)  |  Metric: Kendall's tau  |  Best: tau = 0.215",
    0.5, 5.5, 12.3, 0.5, size=13, italic=True,
    color=C_ACCENT, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem & Motivation
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Problem & Motivation", "Why automate UX quality assessment?")
footer(sl, 2)

items_l = [
    "UX quality assessment is expensive and slow",
    "  Requires 7+ expert designers per UI (UICrit protocol)",
    "  Time-consuming, difficult to scale to millions of apps",
    "",
    "Manual heuristics are too rigid",
    "  Rule-based checks miss holistic design quality",
    "  Cannot capture semantic visual aesthetics",
    "",
    "Research gap",
    "  No systematic study on visual vs structural features",
    "  CLIP pretraining effect on UX quality unknown",
]
bullet(sl, items_l, 0.5, 1.4, 6.0, size=15, spacing=0.35)

rect(sl, 7.0, 1.4, 5.8, 5.5, fill=C_GRAY, line=C_ACCENT, line_w=1.5)
txt(sl, "Research Questions", 7.2, 1.5, 5.4, 0.45,
    size=14, bold=True, color=C_BLUE)

rqs = [
    ("RQ1", "Can ImageNet visual features predict UX quality significantly?"),
    ("RQ2", "Does CLIP pretraining on RICO improve performance?"),
    ("RQ3", "Do structural pixel features help via multi-modal fusion?"),
]
for i, (tag, q) in enumerate(rqs):
    rect(sl, 7.2, 2.1 + i * 1.5, 5.3, 1.2,
         fill=C_BLUE if i == 0 else (C_ORANGE if i == 1 else C_GREEN))
    txt(sl, tag, 7.3, 2.12 + i * 1.5, 1.0, 0.4,
        size=14, bold=True, color=C_WHITE)
    txt(sl, q, 7.3, 2.45 + i * 1.5, 5.0, 0.7,
        size=12, color=C_WHITE, wrap=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Dataset
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Dataset: UICrit + RICO",
           "983 expert-annotated mobile UIs + 66,261 unlabeled RICO screens")
footer(sl, 3)

txt(sl, "UICrit (Duan et al., UIST 2024)", 0.4, 1.35, 6.5, 0.4,
    size=15, bold=True, color=C_BLUE)
items_uicrit = [
    "983 mobile UI screenshots from RICO",
    "7 experienced UX designers per UI",
    "Ratings: aesthetics, usability, overall (1-7 scale)",
    "Split: 686 train / 98 val / 100 test",
    "Score distribution: approx. normal, mean ~0.45",
]
bullet(sl, items_uicrit, 0.4, 1.75, 6.2, size=13.5, spacing=0.33)

txt(sl, "RICO (Deka et al., UIST 2017)", 0.4, 3.75, 6.5, 0.4,
    size=15, bold=True, color=C_GREEN)
items_rico = [
    "72,219 unique Android UI screens",
    "9,772 applications across 27 categories",
    "Used for CLIP pseudo-label pretraining only",
    "No quality annotations — labels generated by CLIP",
]
bullet(sl, items_rico, 0.4, 4.15, 6.2, size=13.5, spacing=0.33)

if (FIG / "fig2_dataset.png").exists():
    add_image(sl, FIG / "fig2_dataset.png", 6.8, 1.3, 6.2, 5.5)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Methodology
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Methodology", "Three complementary approaches investigated")
footer(sl, 4)

txt(sl, "Approach 1 — Visual Backbone (ImageNet pretrained)",
    0.4, 1.35, 12.5, 0.4, size=14, bold=True, color=C_BLUE)
bullet(sl, [
    "EfficientNet-B4: 1792-dim GAP embedding → MLP head → sigmoid score",
    "Two regimes: linear probe (frozen) vs fine-tuning (last 3 stages)",
    "Also tested: EfficientNet-B0, ResNet-50, Swin-T",
], 0.4, 1.75, 12.0, size=12.5, spacing=0.30)

txt(sl, "Approach 2 — CLIP Pseudo-Label Pretraining",
    0.4, 2.75, 12.5, 0.4, size=14, bold=True, color=C_ORANGE)
bullet(sl, [
    "CLIP (ViT-B/32): compute similarity with 6 positive + 4 negative quality prompts",
    "Generate pseudo-labels for 66,261 RICO screenshots",
    "Pretrain EfficientNet-B4 on RICO, then fine-tune on UICrit",
], 0.4, 3.15, 12.0, size=12.5, spacing=0.30)

txt(sl, "Approach 3 — Multi-Modal Gated Fusion",
    0.4, 4.15, 12.5, 0.4, size=14, bold=True, color=C_GREEN)
bullet(sl, [
    "19 pixel-derived structural features: contrast, symmetry, 3x3 grid density, color stats, lum variance",
    "Structural encoder (MLP + LayerNorm) -> 64-dim projection",
    "Gating: g = sigmoid(Linear([fv; fs])) -> fused = g*proj_v + (1-g)*proj_s",
    "Ablation: visual-only / struct-only / full fusion",
], 0.4, 4.55, 12.0, size=12.5, spacing=0.30)

if (FIG / "fig1_architecture.png").exists():
    add_image(sl, FIG / "fig1_architecture.png", 0.3, 5.55, 12.6, 1.7)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Results Table
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Experimental Results", "Kendall tau with 95% bootstrap CI on UICrit test set (n=100)")
footer(sl, 5)

# Table
col_x = [0.3, 4.8, 6.7, 8.2, 9.5, 10.8, 11.9]
col_w = [4.4, 1.8, 1.4, 1.2, 1.2, 1.0, 1.1]
headers = ["Model", "tau", "95% CI", "rho", "MSE", "MAE", "Sig."]
rows = [
    ("EfficientNet-B4 visual-only (probe) *BEST*", "0.215", "[0.088, 0.344]", "0.303", "0.022", "0.129", "YES"),
    ("Gated Fusion full (finetune)",   "0.183", "[0.046, 0.306]", "0.255", "0.096", "0.298", "YES"),
    ("EfficientNet-B4 probe",          "0.165", "[0.038, 0.284]", "0.247", "0.052", "0.201", "YES"),
    ("Swin-T probe",                   "0.120", "[-0.014, 0.258]","0.164", "0.028", "0.125", "No"),
    ("ResNet-50 probe",                "0.088", "[-0.041, 0.211]","0.126", "0.028", "0.146", "No"),
    ("CLIP->EfficientNet probe",       "0.063", "[-0.070, 0.196]","0.085", "0.076", "0.251", "No"),
    ("CLIP->EfficientNet finetune",    "0.007", "[-0.140, 0.149]","0.017", "0.083", "0.259", "No"),
    ("Gated Fusion full (probe)",      "0.024", "[-0.112, 0.155]","0.037", "0.016", "0.100", "No"),
    ("Structural-only (probe)",        "0.002", "[-0.142, 0.147]","0.003", "0.019", "0.114", "No"),
]

row_h = 0.39
t0 = 1.35

# Header row
for j, (cx, cw, hdr) in enumerate(zip(col_x, col_w, headers)):
    rect(sl, cx, t0, cw, row_h, fill=C_BLUE)
    txt(sl, hdr, cx + 0.03, t0 + 0.05, cw - 0.06, row_h - 0.08,
        size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

for i, row in enumerate(rows):
    is_best = i == 0
    fill = RGBColor(0xD5, 0xF5, 0xE3) if is_best else (
           RGBColor(0xFF, 0xF3, 0xE0) if "finetune" in row[0] and i == 1 else
           RGBColor(0xFF, 0xEB, 0xEB) if row[-1] == "No" and float(row[1]) < 0.05 else C_GRAY)
    for j, (cx, cw, val) in enumerate(zip(col_x, col_w, row)):
        rect(sl, cx, t0 + (i+1)*row_h, cw, row_h, fill=fill,
             line=RGBColor(0xCC, 0xCC, 0xCC), line_w=0.3)
        txt(sl, val, cx + 0.03, t0 + (i+1)*row_h + 0.05,
            cw - 0.06, row_h - 0.08,
            size=10, bold=is_best and j in [0,1],
            color=C_GREEN if (is_best and j == 6) else (C_RED if val == "No" else C_DARK),
            align=PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT)

txt(sl, "* Significant = bootstrap 95% CI excludes zero  |  Green = best  |  Red = collapse",
    0.3, 7.0, 12.5, 0.35, size=10, italic=True, color=C_DARK)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Results Chart
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Results Visualisation", "Model comparison with 95% confidence intervals")
footer(sl, 6)

if (FIG / "fig3_results_comparison.png").exists():
    add_image(sl, FIG / "fig3_results_comparison.png", 0.3, 1.3, 12.7, 5.6)
else:
    txt(sl, "Run: python scripts/make_results_figure.py to generate chart",
        2, 4, 9, 1, size=14, color=C_GRAY, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Analysis & Discussion
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Analysis & Discussion", "Answers to our three research questions")
footer(sl, 7)

rq_answers = [
    (C_GREEN, "RQ1 — ImageNet Visual Features", "SUPPORTED",
     ["EfficientNet-B4 visual-only achieves tau = 0.215 (sig.)",
      "ImageNet features capture semantic UX quality",
      "Frozen probe >= fine-tuned on 686 samples"]),
    (C_RED, "RQ2 — CLIP Pretraining", "REFUTED",
     ["CLIP pseudo-labels: tau = 0.089 vs UICrit (too weak)",
      "Pretraining tau: 0.007 to 0.063 (all not significant)",
      "RICO distribution too different from UICrit"]),
    (C_ORANGE, "RQ3 — Structural Fusion", "REFUTED",
     ["Struct-only: tau = 0.002 (random)",
      "Gated Fusion (probe) COLLAPSES to tau = 0.024",
      "Gating overfits on 686 samples (990K params)"]),
]
for i, (color, title, verdict, points) in enumerate(rq_answers):
    lx = 0.3 + i * 4.4
    rect(sl, lx, 1.35, 4.2, 5.5, fill=C_GRAY, line=color, line_w=2)
    rect(sl, lx, 1.35, 4.2, 0.6, fill=color)
    txt(sl, title, lx + 0.1, 1.38, 4.0, 0.55, size=12, bold=True, color=C_WHITE)
    verdict_col = C_GREEN if verdict == "SUPPORTED" else C_RED
    rect(sl, lx + 0.5, 2.1, 3.2, 0.45, fill=verdict_col)
    txt(sl, verdict, lx + 0.5, 2.12, 3.2, 0.4,
        size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    for j, pt in enumerate(points):
        txt(sl, "• " + pt, lx + 0.15, 2.72 + j * 0.7, 3.9, 0.65,
            size=11.5, color=C_DARK, wrap=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Demo
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "System Demo", "Gradio web application for UX quality assessment")
footer(sl, 8)

rect(sl, 0.3, 1.35, 8.0, 5.5, fill=C_GRAY, line=C_BLUE, line_w=1.5)
txt(sl, "Gradio Demo — http://127.0.0.1:7860", 0.5, 1.45, 7.5, 0.45,
    size=13, bold=True, color=C_BLUE)
txt(sl, "python scripts/app.py", 0.5, 1.95, 7.5, 0.4,
    size=12, color=C_DARK, italic=True)

demo_features = [
    "Upload any mobile screenshot",
    "Quality Score: 0-100% with tier (Poor/Fair/Good/Excellent)",
    "Multi-scale Attention Map (head weights + activation energy)",
    "19 Structural Feature breakdown bar chart",
    "Real-time inference on GPU (< 1 second)",
]
for i, f in enumerate(demo_features):
    rect(sl, 0.5, 2.55 + i * 0.72, 7.5, 0.62,
         fill=RGBColor(0x21, 0x8B, 0xC0) if i % 2 == 0 else C_BLUE)
    txt(sl, "  " + f, 0.55, 2.58 + i * 0.72, 7.4, 0.55,
        size=12.5, color=C_WHITE)

# Checkpoint info
rect(sl, 8.5, 1.35, 4.5, 2.2, fill=C_DARK)
txt(sl, "Best Checkpoint", 8.6, 1.4, 4.2, 0.4, size=13, bold=True, color=C_ACCENT)
txt(sl, "checkpoints/baselines/\nefficientnet_b4/best.pt",
    8.6, 1.82, 4.2, 0.9, size=11.5, color=C_GRAY, italic=True)
txt(sl, "tau = 0.165 (standalone)\ntau = 0.215 (visual-only ablation)",
    8.6, 2.72, 4.2, 0.75, size=11, color=C_GREEN)

rect(sl, 8.5, 3.75, 4.5, 3.1, fill=C_GRAY, line=C_ACCENT)
txt(sl, "Attention Map Method", 8.6, 3.8, 4.2, 0.4, size=13, bold=True, color=C_BLUE)
attn_info = [
    "Multi-scale CAM (3 layers):",
    "50% head-weight CAM (semantic)",
    "30% activation energy features[-2]",
    "20% activation energy features[5]",
    "  (14x14, finer spatial detail)",
]
for i, line in enumerate(attn_info):
    txt(sl, line, 8.6, 4.3 + i * 0.46, 4.2, 0.44,
        size=11, color=C_DARK if i > 0 else C_DARK, bold=(i == 0))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Conclusion
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
header_bar(sl, "Conclusion & Future Work", "")
footer(sl, 9)

rect(sl, 0.3, 1.35, 7.8, 5.6, fill=C_GRAY)
txt(sl, "Key Findings", 0.5, 1.4, 7.4, 0.45, size=15, bold=True, color=C_BLUE)

findings = [
    (C_GREEN, "ImageNet EfficientNet-B4 achieves tau = 0.215 (sig.)",
     "Best model — rich visual semantics sufficient"),
    (C_RED, "CLIP pretraining consistently HURTS (tau 0.007-0.063)",
     "Pseudo-label quality too low (tau = 0.089 vs UICrit)"),
    (C_RED, "Gated Fusion COLLAPSES (tau = 0.024, probe)",
     "990K params overfit on 686 samples"),
    (C_ORANGE, "Structural features alone: tau = 0.002",
     "UX quality is semantic, not measurable by pixels"),
]
for i, (color, finding, detail) in enumerate(findings):
    rect(sl, 0.5, 2.0 + i * 1.15, 0.25, 0.25, fill=color)
    txt(sl, finding, 0.9, 1.97 + i * 1.15, 7.0, 0.38,
        size=12.5, bold=True, color=color)
    txt(sl, "  " + detail, 0.9, 2.32 + i * 1.15, 7.0, 0.35,
        size=11, color=C_DARK)

rect(sl, 8.4, 1.35, 4.6, 5.6, fill=C_BLUE)
txt(sl, "Future Work", 8.55, 1.4, 4.2, 0.45, size=15, bold=True, color=C_WHITE)
future = [
    "Larger annotated datasets\n(active learning / crowdsourcing)",
    "LLM-based quality prediction\n(GPT-4V, Gemini Vision)",
    "View hierarchy structural features\n(element-level, not pixel-level)",
    "Pairwise ranking loss aligned\nwith Kendall tau objective",
    "Cross-app generalization study",
]
for i, f in enumerate(future):
    rect(sl, 8.55, 2.05 + i * 0.9, 4.2, 0.78,
         fill=RGBColor(0x1A, 0x4A, 0x6A))
    txt(sl, f, 8.65, 2.08 + i * 0.9, 4.05, 0.72,
        size=11, color=C_WHITE, wrap=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Thank You
# ════════════════════════════════════════════════════════════════════════════
sl = add_slide()
rect(sl, 0, 0, 13.33, 7.5, fill=C_DARK)
rect(sl, 0, 2.6, 13.33, 2.3, fill=C_BLUE)

txt(sl, "Thank You", 0, 2.72, 13.33, 1.0,
    size=48, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txt(sl, "Questions & Discussion", 0, 3.6, 13.33, 0.6,
    size=20, color=C_ACCENT, align=PP_ALIGN.CENTER)

txt(sl, "Deep Learning-Based UX Quality Prediction from Mobile Screenshots Using Visual and Structural Cues",
    0.5, 5.2, 12.3, 0.7, size=13, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)
txt(sl, "IEEE Access  |  KMUTT  |  2025",
    0.5, 5.85, 12.3, 0.5, size=13, color=C_ACCENT, align=PP_ALIGN.CENTER)

summary_items = [
    "Best model: EfficientNet-B4 visual-only  tau = 0.215",
    "Dataset: UICrit 983 UIs + RICO 66K screens",
    "Code & models: github.com/BankDesu/...",
]
for i, item in enumerate(summary_items):
    txt(sl, "• " + item, 1.5, 6.5 + i * 0.32, 10.3, 0.30,
        size=11, color=C_GRAY, align=PP_ALIGN.CENTER)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "PRESENTATION_UXQA.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
