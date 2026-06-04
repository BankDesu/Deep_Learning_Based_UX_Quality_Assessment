"""Round 2: ลอง Unpaywall API + specific alternative URLs สำหรับ papers ที่เหลือ"""
import sys, time, json, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

REFS_DIR = Path("references/pdfs")
EMAIL    = "research@example.com"   # Unpaywall ต้องการ email

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-downloader/1.0)"}

def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    try:
        d = json.loads(fetch(url))
        # best_oa_location
        loc = d.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None

def save(num, fname, url, label):
    dest = REFS_DIR / f"[{num:02d}]_{fname}"
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest.name}")
        return True
    try:
        data = fetch(url)
        if len(data) < 5_000: return False
        if b'%PDF' not in data[:200]: return False
        dest.write_bytes(data)
        print(f"  [OK]   {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  [ERR]  {e}")
        return False

# ── papers + specific alternative sources ────────────────────────────────────
TARGETS = [
    dict(num=2,  fname="tractinsky_2000_beautiful_usable.pdf",
         doi="10.1016/S0953-5438(00)00031-7",
         alts=["https://www.researchgate.net/profile/Noam-Tractinsky/publication/"
               "220501602_What_is_beautiful_is_usable/links/"
               "09e41510d2a1ebb9e2000000/What-is-beautiful-is-usable.pdf"]),

    dict(num=3,  fname="cyr_2010_colour_appeal.pdf",
         doi="10.1016/j.ijhcs.2009.08.005",
         alts=[]),

    dict(num=5,  fname="gui_component_yolo_2024.pdf",
         doi="10.1109/ACT57146.2024.10415929",
         alts=[]),

    dict(num=6,  fname="object_detection_gui_survey.pdf",
         doi="10.1145/3503252",
         alts=[]),

    dict(num=11, fname="rico_dataset.pdf",
         doi="10.1145/3126594.3126651",
         alts=["https://dl.acm.org/doi/pdf/10.1145/3126594.3126651",
               "https://www.interactionmining.org/papers/rico.pdf",
               "http://ranjithakumar.net/resources/rico.pdf"]),

    dict(num=14, fname="accessibility_mobile_apps.pdf",
         doi="10.1109/ICSEA57313.2022.9956022",
         alts=[]),

    dict(num=25, fname="kendall_1938.pdf",
         doi="10.2307/2332226",
         alts=["https://academic.oup.com/biomet/article-pdf/30/1-2/81/553877/30-1-2-81.pdf",
               "https://www.jstor.org/stable/pdf/2332226.pdf?refreqid=fastly-default"]),

    dict(num=32, fname="data_augmentation_classification.pdf",
         doi="10.1109/DICTA.2017.8227475",
         alts=["https://arxiv.org/pdf/1609.08764"]),

    dict(num=35, fname="visual_complexity_web.pdf",
         doi="10.1145/2556288.2557020",
         alts=["https://hal.science/hal-00957269/file/chi14-classify-hal-v2.pdf",
               "https://hal.archives-ouvertes.fr/hal-00957269/document"]),

    dict(num=36, fname="aesthetic_usability_data_viz.pdf",
         doi="10.1109/IV.2007.114",
         alts=[]),

    dict(num=37, fname="aesthetics_utility_hci.pdf",
         doi="10.1080/10447318.2011.586927",
         alts=[]),

    dict(num=39, fname="aesthetic_image_harmonization.pdf",
         doi="10.1109/TCYB.2021.3079311",
         alts=["https://arxiv.org/pdf/2108.05817",
               "https://arxiv.org/pdf/2004.05683"]),

    dict(num=40, fname="aesthetics_image_synthesis.pdf",
         doi="10.1109/TCSVT.2022.3147007",
         alts=[]),
]

ok, fail = [], []
for t in TARGETS:
    n = t["num"]; fname = t["fname"]; doi = t["doi"]
    print(f"\n[{n:02d}] {fname}")

    dest = REFS_DIR / f"[{n:02d}]_{fname}"
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP]"); ok.append(n); continue

    found = False

    # 1) Unpaywall
    if doi:
        print(f"  Unpaywall: {doi}")
        pdf_url = unpaywall(doi)
        if pdf_url:
            print(f"  → {pdf_url[:90]}")
            found = save(n, fname, pdf_url, "")
        time.sleep(1.0)

    # 2) alt URLs
    for u in t.get("alts", []):
        if found: break
        print(f"  alt: {u[:90]}")
        found = save(n, fname, u, "")
        time.sleep(0.8)

    (ok if found else fail).append(n)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
total_ok  = len(list(REFS_DIR.glob("*.pdf")))
print(f"Round 2 สำเร็จ: {ok}")
print(f"Round 2 ยังไม่ได้: {fail}")
print(f"\nPDF ทั้งหมดใน references/pdfs/: {total_ok} ไฟล์")

# แสดงรายการที่เหลือสุดท้าย
no_free = {
    4:  "Google Play Stats (Statista) — screenshot เอา",
    27: "Bootstrap (Efron 1993) — หนังสือ",
    41: "Ko et al. KIISE — ภาษาเกาหลี, DBpia",
}
truly_failed = [f for f in fail if f not in no_free]
print()
print("="*60)
print("สรุปสถานะ references ทั้งหมด")
print("="*60)
auto_ok = [1,7,8,10,12,13,17,18,19,20,21,22,23,26,28,29,30,31,33,34,42,43,44,45]
round1_ok = [9,15,16,24,38,46]
all_ok = set(auto_ok + round1_ok + ok)
print(f"✓ ดาวน์โหลดได้:   {len(all_ok)} / 46 papers")
print(f"✗ ไม่มี free PDF: {sorted(no_free.keys())}  (3 รายการ)")
if truly_failed:
    print(f"? ยังหาไม่ได้:     {truly_failed}")
