"""make_script_docx.py — สร้างบทพูด presentation เป็น Word document"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

OUT = Path("presentation_script.docx")

doc = Document()

# ── page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── colors ────────────────────────────────────────────────────────────────────
BLUE   = RGBColor(0x1A, 0x6F, 0xA8)
GREEN  = RGBColor(0x1A, 0x7A, 0x3C)
ORANGE = RGBColor(0xB8, 0x4D, 0x0A)
PURPLE = RGBColor(0x5B, 0x2E, 0x8C)
RED    = RGBColor(0x7B, 0x18, 0x18)
DARK   = RGBColor(0x1A, 0x20, 0x2C)
GRAY   = RGBColor(0x71, 0x80, 0x96)

# ── helpers ───────────────────────────────────────────────────────────────────
def heading1(text, color=BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size  = Pt(16)
    run.font.color.rgb = color
    # bottom border
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '4')
    bottom.set(qn('w:color'), f'{color[0]:02X}{color[1]:02X}{color[2]:02X}')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p

def slide_label(num, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    r1 = p.add_run(f"Slide {num}  ")
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = GRAY
    r2 = p.add_run(title)
    r2.bold = True
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

def note(text, color=GRAY, italic=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.left_indent  = Cm(0.5)
    run = p.add_run(text)
    run.italic = italic
    run.font.size = Pt(9.5)
    run.font.color.rgb = color

def script(text, indent=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    if indent:
        p.paragraph_format.left_indent = Cm(1.0)
    run = p.add_run(text)
    run.font.size = Pt(12)
    run.font.color.rgb = DARK
    return p

def timing(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(0)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(9.5)
    run.font.color.rgb = ORANGE

def spacer():
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)


# ══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Title block
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(24)
run = p.add_run("บทพูดประกอบการนำเสนอ")
run.bold = True; run.font.size = Pt(22); run.font.color.rgb = BLUE

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
run = p.add_run("การทำนายคุณภาพ UX จากภาพหน้าจอแอปพลิเคชันมือถือ\nด้วย Deep Learning")
run.bold = True; run.font.size = Pt(15); run.font.color.rgb = DARK

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Structural Feature Fusion · CLIP Pretraining · Multi-Scale LayerCAM")
run.italic = True; run.font.size = Pt(12); run.font.color.rgb = GRAY

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
run = p.add_run("ปัณชาติ สุวิมลโอภาส  |  มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี (มจธ.)")
run.font.size = Pt(12); run.font.color.rgb = DARK

# timing overview table
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(18)
run = p.add_run("ระยะเวลาโดยประมาณ ~19 นาที")
run.bold = True; run.font.size = Pt(12); run.font.color.rgb = BLUE

table = doc.add_table(rows=6, cols=3)
table.style = 'Table Grid'
table.alignment = WD_ALIGN_PARAGRAPH.CENTER
hdr = [("ส่วน", "Slides", "เวลา"),
       ("Opening + สารบัญ", "1–2", "~2 นาที"),
       ("Background & Dataset", "3–5", "~4 นาที"),
       ("Method & Architecture", "6–8", "~4 นาที"),
       ("ผลลัพธ์ & Findings", "9–11", "~5 นาที"),
       ("LayerCAM + สรุป", "12–15", "~4 นาที")]
colors_row = [BLUE, DARK, DARK, DARK, DARK, DARK]
for r_idx, (row_data, col_r) in enumerate(zip(hdr, colors_row)):
    row = table.rows[r_idx]
    for c_idx, cell_text in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.text = cell_text
        run = cell.paragraphs[0].runs[0]
        run.font.size = Pt(11)
        run.bold = (r_idx == 0)
        run.font.color.rgb = (C_WHITE := RGBColor(0xFF,0xFF,0xFF)) if r_idx == 0 else DARK
        if r_idx == 0:
            tcPr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), '1A6FA8')
            tcPr.append(shd)

doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Slide 1 ──────────────────────────────────────────────────────────────────
heading1("Slide 1 — หน้าปก", BLUE)
timing("⏱ ~1 นาที  |  ยืนหน้าจอ ทักทายผู้ฟัง")
note("(ยืน ทักทาย มองผู้ฟัง)")
script("สวัสดีครับ/ค่ะ วันนี้จะนำเสนองานวิจัยในหัวข้อ "
       "\"การทำนายคุณภาพ UX จากภาพหน้าจอแอปพลิเคชันมือถือด้วย Deep Learning\" ครับ/ค่ะ")
script("โจทย์หลักของงานนี้คือ — ถ้าเรามีแค่ภาพหน้าจอของแอป "
       "โมเดลจะสามารถบอกได้ไหมว่า UX ของมันดีแค่ไหน "
       "โดยไม่ต้องรอให้ผู้เชี่ยวชาญมาประเมิน")
note("(ชี้ที่ stat boxes ด้านล่าง)")
script("เราใช้ UICrit dataset — 1,000 หน้าจอ แต่ละหน้าจอถูก rate โดย UX designer 7 คน "
       "ทดสอบโมเดลทั้งหมด 10 ตัว "
       "และผลที่ดีที่สุดของเราคือ Kendall เทา เท่ากับ 0.2227 ซึ่งมีนัยสำคัญทางสถิติครับ/ค่ะ")
spacer()

# ── Slide 2 ──────────────────────────────────────────────────────────────────
heading1("Slide 2 — สารบัญ", BLUE)
timing("⏱ ~30 วินาที  |  ผ่านไปเร็ว")
script("โครงสร้างการนำเสนอวันนี้มี 8 หัวข้อ "
       "เริ่มจากแรงจูงใจและคำถามวิจัย ผ่าน dataset และ framework ที่เราออกแบบ "
       "ไปถึงผลการทดลอง ข้อจำกัด และแนวทางพัฒนาต่อครับ/ค่ะ")
spacer()

# ── Slide 3 ──────────────────────────────────────────────────────────────────
heading1("Slide 3 — แรงจูงใจ", BLUE)
timing("⏱ ~1.5 นาที")
note("(ชี้ฝั่งซ้าย — ปัญหา)")
script("ก่อนอื่นเลย ทำไมเราถึงต้องการ automate การประเมิน UX?")
script("ปัจจุบันการประเมิน UX คุณภาพต้องอาศัย UX designer ที่มีประสบการณ์ "
       "ซึ่งใช้เวลาและค่าใช้จ่ายสูง "
       "ในขณะที่ App Store มีแอปพลิเคชันนับล้านตัวและอัปเดตตลอดเวลา")
script("เครื่องมือที่มีอยู่ส่วนใหญ่เป็นแบบ rule-based — "
       "เช็ค contrast ratio หรือขนาด touch target — "
       "ซึ่งจับได้แค่ข้อผิดพลาดเฉพาะจุด "
       "ไม่ได้วัดคุณภาพโดยรวมที่ผู้เชี่ยวชาญมอง")
note("(ชี้ฝั่งขวา — approach)")
script("เราจึงต้องการโมเดลที่รับแค่ภาพหน้าจอ แล้วให้คะแนนที่สอดคล้องกับ "
       "การประเมินของผู้เชี่ยวชาญจริงๆ "
       "และยังสามารถอธิบายได้ว่าโมเดลมอง region ไหนของหน้าจอ")
spacer()

# ── Slide 4 ──────────────────────────────────────────────────────────────────
heading1("Slide 4 — คำถามวิจัย", BLUE)
timing("⏱ ~2 นาที")
script("งานนี้ตั้งคำถามวิจัยไว้ 3 ข้อครับ/ค่ะ")
note("(RQ1 — กล่องบน)")
script("RQ1 ถามว่า — feature จาก ImageNet pretraining แค่นั้นพอไหม "
       "โดยไม่ต้อง fine-tune หรือ domain adaptation ใดๆ "
       "คำตอบคือ ได้ — EfficientNet-B4 แบบ frozen probe ให้ Kendall เทา 0.215 "
       "ซึ่งมีนัยสำคัญทางสถิติครับ/ค่ะ")
note("(RQ2 — กล่องกลาง)")
script("RQ2 ถามว่า — ถ้าเราเอา CLIP มา pretrain บน RICO dataset ก่อน "
       "แล้วค่อย fine-tune จะช่วยไหม "
       "คำตอบคือ ไม่ได้ — ประสิทธิภาพตกลงจนเกือบ random "
       "ซึ่งจะอธิบายสาเหตุในสไลด์ผลลัพธ์ครับ/ค่ะ")
note("(RQ3 — กล่องล่าง)")
script("RQ3 ถามว่า — structural feature 19 ตัวที่คำนวณจาก pixel โดยตรง "
       "จะช่วยได้ไหมถ้า fuse เข้ากับ visual feature ผ่าน Learned Gate "
       "คำตอบคือ ได้ — นี่คือผลที่ดีที่สุดของเรา เทา 0.2227")
spacer()

# ── Slide 5 ──────────────────────────────────────────────────────────────────
heading1("Slide 5 — ชุดข้อมูล UICrit", GREEN)
timing("⏱ ~1.5 นาที")
note("(ชี้ที่ stat boxes แถวบน)")
script("Dataset หลักที่ใช้คือ UICrit จาก UIST 2024 ครับ/ค่ะ "
       "มี 1,000 หน้าจอแอป Android แต่ละหน้าจอถูกให้คะแนนโดย UX designer 7 คน "
       "บน scale 1 ถึง 7 ครอบคลุมทั้ง aesthetics, usability, และ clarity "
       "เราปรับ normalize ให้เป็น 0–1 "
       "และแบ่ง 800 train / 100 val / 100 test โดย stratify ตาม score quartile")
note("(ชี้ที่ histogram ซ้าย)")
script("การกระจายตัวของคะแนนเป็น normal ศูนย์กลางอยู่ที่ประมาณ 0.80 "
       "หมายความว่าหน้าจอส่วนใหญ่ใน dataset อยู่ในระดับปานกลางถึงดี "
       "มีหน้าจอคุณภาพต่ำค่อนข้างน้อย")
script("inter-rater agreement อยู่ที่ κ ประมาณ 0.41 ซึ่งถือว่า moderate — "
       "หมายความว่า UX quality มีความ subjective อยู่บ้าง "
       "ซึ่งเป็นหนึ่งในข้อจำกัดที่เราจะพูดถึงทีหลังครับ/ค่ะ")
spacer()

# ── Slide 6 ──────────────────────────────────────────────────────────────────
heading1("Slide 6 — Framework ภาพรวม", GREEN)
timing("⏱ ~1.5 นาที")
note("(ไล่จากซ้ายไปขวา)")
script("นี่คือ framework โดยรวมครับ/ค่ะ เริ่มจากภาพหน้าจอผ่าน preprocessing "
       "resize เป็น 224×224 และ normalize ด้วย ImageNet stats")
script("แบ่งเป็น 2 branch ขนาน — "
       "branch บนคือ visual: ผ่าน EfficientNet-B4 ได้ feature 1,792 มิติ "
       "แล้วผ่าน Visual Head ได้ logit_v "
       "branch ล่างคือ structural: คำนวณ 19 pixel feature โดยตรงจากภาพ "
       "แล้วผ่าน Struct Head ได้ logit_s")
script("ตรงกลางคือ Gate Network — รับ feature จากทั้งสองฝั่ง "
       "คำนวณค่า alpha ที่บอกว่าควรเชื่อ visual หรือ structural มากกว่ากัน "
       "Score ตัวสุดท้ายคือ sigma ของ alpha คูณ logit_v บวก (1 ลบ alpha) คูณ logit_s")
note("(ชี้ที่ LayerCAM ขวาสุด)")
script("ด้านขวาสุดคือ LayerCAM ซึ่งไม่ได้อยู่ใน pipeline การทำนาย "
       "แต่เป็น post-hoc explanation ที่เราสร้างขึ้นเพื่ออธิบาย "
       "ว่าโมเดลมอง region ไหนของหน้าจอ")
spacer()

# ── Slide 7 ──────────────────────────────────────────────────────────────────
heading1("Slide 7 — Architecture รายละเอียด", BLUE)
timing("⏱ ~1.5 นาที")
note("(ชี้ panel ซ้าย — EfficientNet)")
script("EfficientNet-B4 ใช้ compound scaling คือปรับ depth, width, และ resolution "
       "พร้อมกันตาม ratio ที่ optimize ไว้ "
       "ทำให้ได้ผลดีกว่าการ scale แบบ dimension เดียว "
       "มี 19 ล้าน parameters ให้ embedding 1,792 มิติจาก Global Average Pooling")
script("โหมด probe — freeze ทุก stage ใช้แค่ head ที่ train ใหม่ "
       "โหมด finetune — unfreeze Stage 6 และ 7 เพิ่มเติม")
note("(ชี้ panel ขวา — Gate)")
script("Structural Encoder รับ 19-dim feature ที่คำนวณจาก pixel โดยตรง "
       "ไม่มี annotation ไม่มี external detector "
       "แบ่งเป็น 5 กลุ่ม: contrast, whitespace, symmetry, spatial grid density, "
       "และ color stats")
script("ส่วน Gate Network นั้นเพิ่ม parameter แค่ประมาณ 14,000 ตัวเท่านั้น "
       "เทียบกับ backbone 19 ล้านตัว")
spacer()

# ── Slide 8 ──────────────────────────────────────────────────────────────────
heading1("Slide 8 — EfficientNet-B4 ต้นฉบับ vs ที่ใช้", BLUE)
timing("⏱ ~1 นาที  |  ถ้ามีเวลา")
note("(ชี้แถวบน — original)")
script("แถวบนแสดง architecture ต้นฉบับ มี 7 stage ตามด้วย Global Average Pooling "
       "และ Classifier สำหรับ ImageNet 1,000 class")
note("(ชี้แถวล่าง — adapted)")
script("แถวล่างคือ version ที่เราใช้ — "
       "เปลี่ยน classifier ออก ใส่ prediction head ใหม่เข้าไปแทน "
       "Stage 1–5 frozen ทั้งสองโหมด "
       "Stage 6–7 จะ unfreeze เฉพาะ finetune mode เท่านั้น")
spacer()

# ── Slide 9 ──────────────────────────────────────────────────────────────────
heading1("Slide 9 — Setup การทดลอง", ORANGE)
timing("⏱ ~1 นาที")
script("สำหรับ setup การทดลอง ใช้ AdamW optimizer พร้อม cosine learning rate schedule "
       "กับ linear warmup, dropout 0.5 ใน visual head "
       "และทดสอบโมเดลทั้งหมด 10 ตัว บน GPU RTX 4070 Super")
script("Metric หลักคือ Kendall เทา ซึ่งวัด rank correlation กับ expert scores "
       "เราใช้เทา เพราะงานนี้ต้องการรู้ว่าโมเดลเรียงลำดับหน้าจอ "
       "จากแย่ไปดีได้ถูกต้องไหม ไม่ใช่แค่ predict ค่าตัวเลข "
       "ทุกผลลัพธ์รายงาน 95% bootstrap CI จาก 10,000 resamples "
       "ผลที่ significant คือ CI ที่ไม่ครอบ 0 ครับ/ค่ะ")
spacer()

# ── Slide 10 ──────────────────────────────────────────────────────────────────
heading1("Slide 10 — ผลลัพธ์หลัก", ORANGE)
timing("⏱ ~2 นาที")
note("(ชี้กราฟแท่ง)")
script("นี่คือผลลัพธ์ครับ/ค่ะ แกน y คือ Kendall เทา ยิ่งสูงยิ่งดี "
       "ดาวหมายถึงผลมีนัยสำคัญทางสถิติ")
script("แท่งสีเขียวซ้ายสุดคือโมเดลที่ดีที่สุดของเรา — "
       "Learned Gate Fusion probe เทา 0.2227 ดาว "
       "ตามมาด้วย Learned Gate finetune และ EfficientNet-B4 visual-only")
note("(ชี้ DINOv2 และ CLIP)")
script("น่าสังเกตว่า DINOv2 ที่ pretrain บน 142 ล้านภาพ "
       "กลับทำได้แค่ 0.034 ซึ่งไม่ significant "
       "และ CLIP ทั้งสองโหมดก็ให้ผลต่ำมากเช่นกัน")
note("(ชี้ Struct-only ขวาสุด)")
script("Struct-only ติดลบ แสดงว่า structural features อย่างเดียว "
       "ไม่มีประโยชน์เลย ต้องใช้ร่วมกับ visual เท่านั้น")
spacer()

# ── Slide 11 ──────────────────────────────────────────────────────────────────
heading1("Slide 11 — สิ่งที่ค้นพบ", ORANGE)
timing("⏱ ~2 นาที")
note("(คอลัมน์ซ้าย — RQ1)")
script("RQ1: ImageNet feature ใช้ได้ผล "
       "EfficientNet-B4 ที่ pretrain บน ImageNet "
       "ดักจับ visual statistics ที่สอดคล้องกับ UX quality ได้ "
       "โดยเฉพาะ compound scaling ที่ capture หลาย scale พร้อมกัน "
       "น่าสนใจว่า DINOv2 ที่ pretrain บน 142 ล้านภาพทำได้แย่กว่า "
       "เพราะ self-supervised feature อาจไม่เหมาะกับ 800 training samples")
note("(คอลัมน์กลาง — RQ2)")
script("RQ2: CLIP pretraining ให้ผลตรงข้ามกับที่คาดไว้ "
       "สาเหตุสองข้อ — หนึ่ง pseudo-label ที่ CLIP ให้มีคุณภาพต่ำมาก "
       "เพราะ CLIP ไม่ได้ถูก train มาเพื่อ judge UX quality โดยตรง "
       "สอง distribution ของ RICO กว้างมาก "
       "การ pretrain บนมันทำให้ feature เสียหายมากกว่าช่วย")
note("(คอลัมน์ขวา — RQ3)")
script("RQ3: Gate Fusion ช่วยได้แต่ต้องออกแบบถูกต้อง "
       "Key insight คือ gate alpha ประมาณ 1 เมื่อ structural feature noisy "
       "หมายความว่าโมเดลเรียนรู้ที่จะ ignore structural เมื่อไม่มีประโยชน์ "
       "และการ blend ที่ระดับ logit ไม่ใช่ embedding ป้องกัน collapse ของ gate ได้")
spacer()

# ── Slide 12 ──────────────────────────────────────────────────────────────────
heading1("Slide 12 — LayerCAM Visualization", PURPLE)
timing("⏱ ~1.5 นาที")
note("(ชี้แถวบน — high quality)")
script("นอกจาก quantitative results เรายังวิเคราะห์เชิงคุณภาพผ่าน LayerCAM ครับ/ค่ะ")
script("หน้าจอคุณภาพสูง — activation map กระจุกตัวที่ primary content "
       "ทั้ง hero image, CTA button, และหัวข้อหลัก "
       "โมเดลเรียนรู้ implicit visual hierarchy โดยไม่มี annotation ใดๆ")
note("(ชี้แถวล่าง — low quality)")
script("หน้าจอคุณภาพต่ำ — activation กระจายทั่ว ไม่มี focal point ชัดเจน "
       "มักอยู่บริเวณ text ที่หนาแน่นหรือ region ที่ cluttered")
script("สิ่งที่ต้องเน้นคือ LayerCAM นี้เป็น post-hoc explanation เท่านั้น "
       "ไม่ได้เป็น input ของโมเดล แต่ช่วยให้เราเข้าใจว่าโมเดลมองอะไร")
spacer()

# ── Slide 13 ──────────────────────────────────────────────────────────────────
heading1("Slide 13 — ข้อจำกัด", RED)
timing("⏱ ~1.5 นาที")
script("งานนี้มีข้อจำกัดที่ต้องพูดถึงตรงๆ 4 ข้อครับ/ค่ะ")
note("(ชี้แต่ละกล่อง)")
script("ข้อแรก — dataset เล็กมาก 1,000 ตัวอย่าง "
       "ทำให้ CI กว้างถึง 0.12–0.15 และโมเดลที่ซับซ้อนกว่านี้มักจะ overfit")
script("ข้อสอง — test set มีแค่ 100 ตัวอย่าง standard error ของเทา ประมาณ 0.07 "
       "หมายความว่าความแตกต่างเทา ต่ำกว่า 0.05 ระหว่างโมเดล แทบแยกไม่ออกทางสถิติ")
script("ข้อสาม — annotation มี subjectivity สูง κ ประมาณ 0.41 "
       "ทำให้ ground truth เองก็มี noise")
script("ข้อสี่ — ทั้ง dataset และ pretraining มาจาก Android เท่านั้น "
       "ไม่รู้ว่าผลจะ generalize ไป iOS หรือ web ได้ไหม")
spacer()

# ── Slide 14 ──────────────────────────────────────────────────────────────────
heading1("Slide 14 — แนวทางการพัฒนาต่อ", GREEN)
timing("⏱ ~1.5 นาที")
script("สำหรับแนวทางพัฒนาต่อ มี 6 ทิศทางหลักครับ/ค่ะ")
script("ด่วนที่สุดคือข้อ 1 — ต้องได้ dataset ที่ใหญ่กว่านี้ "
       "ไม่ว่าจะผ่าน active learning หรือ crowdsourcing "
       "เพราะ bottleneck หลักของงานนี้คือจำนวนข้อมูล ไม่ใช่ architecture")
script("ข้อ 2 น่าสนใจมาก — ทดสอบ GPT-4V หรือ Gemini Vision "
       "เป็น zero-shot predictor ซึ่งอาจไม่ต้องการ labeled data เลย")
script("ข้อ 3 ใช้ K-fold cross-validation แทน single split "
       "เพื่อ reduce variance ของ เทา estimate")
script("ข้อ 4 ลอง DINOv2 กับ native resolution "
       "ปัจจุบันเราบังคับ resize เป็น 224×224 ซึ่งอาจทำให้ detail เล็กๆ หายไป")
script("ข้อ 5 และ 6 — ขยายไป iOS กับ web และทดลอง personalized model "
       "ที่ตอบสนองต่อความชอบของ user แต่ละคน")
spacer()

# ── Slide 15 ──────────────────────────────────────────────────────────────────
heading1("Slide 15 — สรุป", DARK)
timing("⏱ ~1 นาที")
script("สรุปงานนี้ครับ/ค่ะ")
script("เราเสนอและทดสอบ deep learning approaches "
       "สำหรับ automated UX quality prediction จากภาพหน้าจอมือถือ")
script("ข้อค้นพบหลัก — EfficientNet-B4 frozen ImageNet weights ทำงานได้ดี "
       "โดยไม่ต้อง domain adaptation, "
       "CLIP pretraining กลับเป็น negative เพราะ pseudo-label คุณภาพต่ำ "
       "และ Learned Gate Fusion ที่ lightweight ประมาณ 14,000 parameters "
       "ให้ผลดีที่สุด เทา 0.2227")
script("งานนี้แสดงให้เห็นว่า visual feature บวก structural feature "
       "บวก lightweight gate เป็น combination ที่ work สำหรับ dataset ขนาดเล็ก "
       "และ LayerCAM ช่วยให้เห็นว่าโมเดลจับ visual hierarchy ได้จริง")
spacer()
note("(จบ — รอคำถาม)")
script("ขอบคุณมากครับ/ค่ะ มีคำถามหรือข้อเสนอแนะไหมครับ/ค่ะ")


# ── save ──────────────────────────────────────────────────────────────────────
doc.save(str(OUT))
print(f"Saved: {OUT}")
