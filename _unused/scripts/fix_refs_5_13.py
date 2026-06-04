"""
fix_refs_5_13.py
1. แทน reference entry [5] ด้วย Chen et al. 2020 (paper จริง)
2. ลบ [13] ออกจากเนื้อหาและ reference list
3. Renumber [14]-[35] → [13]-[34] (shift -1)
"""
import sys, zipfile, re, shutil, os
sys.stdout.reconfigure(encoding="utf-8")

PATH = r"E:\Works\KMUTT\CV_Project\Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment\รูปเล่ม.docx"

shutil.copy2(PATH, PATH + ".bak2")

with zipfile.ZipFile(PATH, "r") as z:
    names = z.namelist()
    files = {n: z.read(n) for n in names}

xml = files["word/document.xml"].decode("utf-8")

# ── 1. แทน reference entry [5] ───────────────────────────────────────────────
# หา paragraph ที่มี [5] + "X. Chen" (เนื้อหา ref entry เดิม)
old_5_pattern = (
    r"(<w:p\b[^>]*>"
    r"(?:(?!<w:p\b).)*?"
    r"(?<![a-zA-Z])\[5\]"
    r"(?:(?!<\/w:p>).)*?"
    r"X\. Chen"                     # distinctive text of old entry
    r"(?:(?!<\/w:p>).)*?"
    r"<\/w:p>)"
)

new_ref5_text = (
    "[5]  J. Chen, M. Xie, Z. Xing, C. Chen, X. Xu, L. Zhu, and G. Li, "
    "‘Object Detection for Graphical User Interface: Old Fashioned or "
    "Deep Learning or a Combination?’ in Proc. ACM ESEC/FSE, 2020, "
    "doi: 10.1145/3368089.3409691."
)

# สร้าง paragraph XML ใหม่ที่มี style เดิม (ดึง pPr จาก paragraph เดิมก่อน)
m5 = re.search(old_5_pattern, xml, re.DOTALL)
if m5:
    old_para = m5.group(1)
    # ดึง pPr (properties) จาก paragraph เดิม
    ppr_m = re.search(r"<w:pPr>.*?</w:pPr>", old_para, re.DOTALL)
    ppr = ppr_m.group(0) if ppr_m else ""
    # ดึง rPr จาก run เดิม
    rpr_m = re.search(r"<w:rPr>.*?</w:rPr>", old_para, re.DOTALL)
    rpr = rpr_m.group(0) if rpr_m else ""

    new_para = (
        f'<w:p><w:pPr>{ppr.replace("<w:pPr>","").replace("</w:pPr>","")}</w:pPr>'
        f'<w:r>{rpr}<w:t xml:space="preserve">{new_ref5_text}</w:t></w:r></w:p>'
    )
    xml = xml.replace(old_para, new_para, 1)
    print("✓ [5] reference entry replaced")
else:
    print("✗ [5] entry not found — ลองแทนแบบ fallback")
    # fallback: แทน text ใน paragraph ที่มี [5] ใน ref section
    xml = re.sub(
        r"(?<=\[5\]\s{1,4})X\. Chen.*?(?=<\/w:t>)",
        (new_ref5_text[5:]),   # ข้ามส่วน [5]  ที่มีอยู่แล้ว
        xml, count=1, flags=re.DOTALL
    )
    print("✓ [5] entry replaced (fallback)")

# ── 2. ลบ [13] จากเนื้อหา ────────────────────────────────────────────────────
before = len(re.findall(r"(?<![a-zA-Z])\[13\]", xml))
xml = re.sub(r",?\s*(?<![a-zA-Z])\[13\]", "", xml)
xml = re.sub(r"(?<![a-zA-Z])\[13\],?\s*", "", xml)
after = len(re.findall(r"(?<![a-zA-Z])\[13\]", xml))
print(f"✓ [13] removed from body: {before} → {after}")

# ── 3. ลบ reference entry [13] ───────────────────────────────────────────────
pattern_13 = (
    r"<w:p\b[^>]*>"
    r"(?:(?!<w:p\b).)*?"
    r"(?<![a-zA-Z])\[13\]"
    r"(?:(?!<\/w:p>).)*?"
    r"<\/w:p>"
)
xml, count = re.subn(pattern_13, "", xml, flags=re.DOTALL)
print(f"✓ [13] reference entry removed: {count} paragraph")

# ── 4. Renumber [14]-[35] → [13]-[34] ────────────────────────────────────────
# ใช้ TMP placeholder ป้องกัน collision
for old in range(35, 13, -1):   # ไล่จากสูงลงต่ำ
    new = old - 1
    xml = re.sub(
        rf"(?<![a-zA-Z])\[{old}\]",
        f"[TMP{new}]",
        xml
    )
xml = xml.replace("[TMP", "[")
print("✓ Renumbered [14]-[35] → [13]-[34]")

# ── 5. ตรวจสอบ ────────────────────────────────────────────────────────────────
ref_start = xml.find(">References<")
body = xml[:ref_start]; refs = xml[ref_start:]
body_cites = sorted(set(int(x) for x in re.findall(r"(?<![a-zA-Z])\[(\d+)\]", body)))
ref_entries = sorted(set(int(x) for x in re.findall(r"(?<![a-zA-Z])\[(\d+)\]", refs)))
print(f"\nBody citations : {body_cites}")
print(f"Reference list : {ref_entries}")
mismatch_b = [c for c in body_cites if c not in ref_entries]
mismatch_r = [c for c in ref_entries if c not in body_cites]
print(f"Cited, no entry: {mismatch_b}")
print(f"Entry, not cited: {mismatch_r}")
print(f"Total refs: {len(ref_entries)}")

# ── save ──────────────────────────────────────────────────────────────────────
files["word/document.xml"] = xml.encode("utf-8")
os.remove(PATH)
with zipfile.ZipFile(PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, files[n])
os.remove(PATH + ".bak2")
print(f"\nSaved: {PATH}")
