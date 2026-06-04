"""
download_references_manual.py
หา open-access PDF ผ่าน Semantic Scholar API + alternative URLs
สำหรับ 22 papers ที่ดาวน์โหลดอัตโนมัติไม่ได้
"""
import sys, time, json, re
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

REFS_DIR = Path("references/pdfs")
REFS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; research-downloader/1.0; "
        "mailto:research@university.edu)"
    )
}

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"

# ── helpers ───────────────────────────────────────────────────────────────────
def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def s2_by_doi(doi):
    """ถามข้อมูล paper จาก Semantic Scholar ด้วย DOI"""
    url = f"{S2_BASE}/{urllib.parse.quote('DOI:'+doi)}?fields=openAccessPdf,title,year"
    try:
        data = json.loads(fetch(url))
        return data.get("openAccessPdf", {}).get("url")
    except Exception:
        return None

def s2_by_title(title):
    """ค้นหา paper จาก Semantic Scholar ด้วยชื่อเรื่อง"""
    q = urllib.parse.quote(title)
    url = f"{S2_BASE}/search?query={q}&fields=openAccessPdf,title,year&limit=3"
    try:
        data = json.loads(fetch(url))
        for p in data.get("data", []):
            pdf = (p.get("openAccessPdf") or {}).get("url")
            if pdf:
                return pdf
    except Exception:
        return None

def download_pdf(url, dest, label):
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"    [SKIP] {dest.name}  ({dest.stat().st_size//1024} KB)")
        return True
    try:
        data = fetch(url)
        if len(data) < 8_000:
            return False
        # ตรวจว่าเป็น PDF จริง
        if not data[:4] == b'%PDF' and b'%PDF' not in data[:100]:
            return False
        dest.write_bytes(data)
        print(f"    [OK]   {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        return False

def try_download(num, fname, doi=None, title=None, alt_urls=None, note=""):
    dest = REFS_DIR / f"[{num:02d}]_{fname}"
    print(f"\n[{num:02d}] {note}")

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"    [SKIP] มีไฟล์แล้ว ({dest.stat().st_size//1024} KB)")
        return True

    pdf_url = None

    # 1) ลอง Semantic Scholar ด้วย DOI
    if doi and not pdf_url:
        print(f"    S2 DOI lookup: {doi}")
        pdf_url = s2_by_doi(doi)
        if pdf_url: print(f"    found: {pdf_url[:80]}")
        time.sleep(0.8)

    # 2) ลอง Semantic Scholar ด้วยชื่อ
    if title and not pdf_url:
        print(f"    S2 title search: {title[:60]}")
        pdf_url = s2_by_title(title)
        if pdf_url: print(f"    found: {pdf_url[:80]}")
        time.sleep(0.8)

    # 3) ลอง alternative URLs
    for u in (alt_urls or []):
        if not pdf_url:
            print(f"    trying alt: {u[:80]}")
            dest_test = REFS_DIR / f"[{num:02d}]_{fname}"
            ok = download_pdf(u, dest_test, note)
            if ok:
                return True
        time.sleep(0.5)

    # 4) ดาวน์โหลดจาก S2
    if pdf_url:
        ok = download_pdf(pdf_url, dest, note)
        if ok:
            time.sleep(1.0)
            return True

    print(f"    [FAIL] ไม่พบ open-access PDF")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# รายการ papers ที่ต้องหา
# ═════════════════════════════════════════════════════════════════════════════
results = {}

# [02] Tractinsky 2000
results[2] = try_download(2, "tractinsky_2000_beautiful_usable.pdf",
    doi="10.1016/S0953-5438(00)00031-7",
    title="What is beautiful is usable",
    alt_urls=[
        "https://www.sciencedirect.com/science/article/pii/S0953543800000317/pdfft",
    ])

# [03] Cyr 2010
results[3] = try_download(3, "cyr_2010_colour_appeal.pdf",
    doi="10.1016/j.ijhcs.2009.08.005",
    title="Colour appeal in website design within and across cultures")

# [05] GUI Component Detection 2024
results[5] = try_download(5, "gui_component_yolo_2024.pdf",
    doi="10.1109/ACT57146.2024.10415929",
    title="GUI Component Detection Using YOLO and Faster-RCNN")

# [06] Object Detection GUI Survey 2022
results[6] = try_download(6, "object_detection_gui_survey.pdf",
    doi="10.1145/3503252",
    title="Object Detection in GUI: A Survey")

# [09] CLIP-based UI Understanding 2023
results[9] = try_download(9, "clip_ui_understanding.pdf",
    doi="10.1145/3581641.3584038",
    title="CLIP-based UI Understanding")

# [11] RICO Dataset 2017
results[11] = try_download(11, "rico_dataset.pdf",
    doi="10.1145/3126594.3126651",
    title="Rico: A Mobile App Dataset for Building Data-Driven Design Applications",
    alt_urls=[
        "http://interactionmining.org/papers/deka2017rico.pdf",
    ])

# [14] Accessibility Evaluation 2022
results[14] = try_download(14, "accessibility_mobile_apps.pdf",
    doi="10.1109/ICSEA57313.2022.9956022",
    title="Accessibility Evaluation of Mobile Applications")

# [15] Visual Aesthetics Mobile UIs 2024 (JBCS — open access)
results[15] = try_download(15, "deep_learning_visual_aesthetics.pdf",
    doi="10.5753/jbcs.2024.3255",
    title="A Deep Learning Model for the Assessment of Visual Aesthetics of Mobile UIs",
    alt_urls=[
        "https://journals-sol.sbc.org.br/index.php/jbcs/article/download/3255/2872",
        "https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3255/2872",
    ])

# [16] AI-Driven Aesthetics 2022 (IET open access)
results[16] = try_download(16, "ai_aesthetics_cnn.pdf",
    doi="10.1049/ccs2.12055",
    title="AI-Driven User Aesthetics Preference Prediction for UI Layouts via Deep CNNs",
    alt_urls=[
        "https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/ccs2.12055",
    ])

# [24] Eigen-CAM 2020/2021
results[24] = try_download(24, "eigencam.pdf",
    doi="10.1007/s42979-021-00449-3",
    title="Eigen-CAM: Class Activation Map using Principal Components",
    alt_urls=[
        "https://arxiv.org/pdf/2008.00299",
    ])

# [25] Kendall 1938 (classic — public domain, try multiple sources)
results[25] = try_download(25, "kendall_1938.pdf",
    title="A New Measure of Rank Correlation Kendall 1938 Biometrika",
    alt_urls=[
        "https://www.jstor.org/stable/pdf/2332226.pdf",
        "https://academic.oup.com/biomet/article-pdf/30/1-2/81/553877/30-1-2-81.pdf",
    ])

# [27] Bootstrap book — ไม่มี PDF ฟรี บันทึก note
results[27] = False
print("\n[27] Efron & Tibshirani (1993) Bootstrap — หนังสือ ไม่มี open-access PDF")

# [32] Data Augmentation DICTA 2017
results[32] = try_download(32, "data_augmentation_classification.pdf",
    doi="10.1109/DICTA.2017.8227475",
    title="Understanding Data Augmentation for Classification When to Warp")

# [35] Quantifying Visual Complexity CHI 2014
results[35] = try_download(35, "visual_complexity_web.pdf",
    doi="10.1145/2556288.2557020",
    title="Quantifying Visual Complexity of Web Pages")

# [36] Aesthetic Usability DataViz IEEE IV 2007
results[36] = try_download(36, "aesthetic_usability_data_viz.pdf",
    doi="10.1109/IV.2007.114",
    title="The Effect of Aesthetic on the Usability of Data Visualization")

# [37] Aesthetics Utility HCI 2012
results[37] = try_download(37, "aesthetics_utility_hci.pdf",
    doi="10.1080/10447318.2011.586927",
    title="Aesthetics and Utility in Human Computer Interaction")

# [38] Combinatorial Optimization GUI Proc. IEEE 2020
results[38] = try_download(38, "combinatorial_optimization_gui.pdf",
    doi="10.1109/JPROC.2020.2969687",
    title="Combinatorial Optimization of Graphical User Interface Designs",
    alt_urls=[
        "https://arxiv.org/pdf/1810.02312",
    ])

# [39] Aesthetic Image Harmonization IEEE Trans Cybern 2021
results[39] = try_download(39, "aesthetic_image_harmonization.pdf",
    doi="10.1109/TCYB.2021.3079311",
    title="Aesthetic Image Harmonization with Saliency-Guided Correction")

# [40] Aesthetics-Driven Image Synthesis IEEE TCSVT 2022
results[40] = try_download(40, "aesthetics_image_synthesis.pdf",
    doi="10.1109/TCSVT.2022.3147007",
    title="Aesthetics-Driven Image Synthesis")

# [41] Ko et al. KIISE — Korean journal, likely no open access
results[41] = False
print("\n[41] Ko et al. (2023) UI Aesthetic Score — KIISE (ภาษาเกาหลี) ไม่มี open-access")

# [46] LayerCAM IEEE TIP 2021
results[46] = try_download(46, "layercam.pdf",
    doi="10.1109/TIP.2021.3089943",
    title="LayerCAM: Exploring Hierarchical Class Activation Maps for Localization",
    alt_urls=[
        "https://arxiv.org/pdf/2103.15213",
        "https://arxiv.org/pdf/2103.11351",
    ])

# ── summary ───────────────────────────────────────────────────────────────────
ok   = [k for k,v in results.items() if v]
fail = [k for k,v in results.items() if not v]

print("\n" + "=" * 60)
print(f"สำเร็จ: {len(ok)} ไฟล์  →  {ok}")
print(f"ไม่พบ:  {len(fail)} ไฟล์  →  {fail}")

# อัปเดต manual_download.txt ให้แสดงเฉพาะที่ยังต้องดาวน์โหลด
still_manual = {
    4:  ("[04]", "google_play_statistics_2024.pdf",
         "Statista 2024 — ต้อง screenshot หน้าเว็บ แล้วบันทึกเป็น PDF"),
    27: ("[27]", "bootstrap_efron.pdf",
         "Efron & Tibshirani (1993) Bootstrap — หนังสือ ไม่มี PDF ฟรี ใส่ citation ได้เลย"),
    41: ("[41]", "ui_aesthetic_score_kiise.pdf",
         "Ko et al. (2023) KIISE ภาษาเกาหลี — ต้องการ DBpia access"),
}
for k in fail:
    if k not in still_manual and k != 27 and k != 41 and k != 4:
        still_manual[k] = (f"[{k:02d}]", f"ref_{k:02d}.pdf",
                           "ไม่พบ open-access version — ลอง Google Scholar หรือ ResearchGate")

remaining = Path("references/manual_download_remaining.txt")
with open(remaining, "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("Papers ที่ยังต้องดาวน์โหลดเอง\n")
    f.write("=" * 65 + "\n\n")
    for k in sorted(still_manual):
        tag, fn, note = still_manual[k]
        f.write(f"{tag} {note}\n")
        f.write(f"     ไฟล์: {fn}\n\n")

print(f"\nรายการที่เหลือ: {remaining}")
print("เสร็จสิ้น")
