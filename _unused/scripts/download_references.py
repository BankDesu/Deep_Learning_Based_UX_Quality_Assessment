"""
download_references.py
ดาวน์โหลด PDF ของ references ทั้ง 46 ข้อลงใน references/pdfs/
- แหล่งที่ download ได้อัตโนมัติ: arXiv, ACL Anthology, PMLR, JMLR
- แหล่งที่ต้อง download เอง: บันทึกเป็น manual_download.txt
"""
import sys, time, os, re
sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import urllib.error
from pathlib import Path

REFS_DIR = Path("references/pdfs")
REFS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── reference list ────────────────────────────────────────────────────────────
# (ref_num, filename_prefix, url, note)
REFS = [
    # ── arXiv papers ──────────────────────────────────────────────────────────
    ( 1, "uicrit",
      "https://arxiv.org/pdf/2407.08850",
      "UICrit: Enhancing Automated Design Evaluation (UIST 2024)"),
    ( 7, "screen_parsing",
      "https://arxiv.org/pdf/2109.08763",
      "Screen Parsing: Reverse Engineering UI Models (UIST 2021)"),
    ( 8, "ui_tappability",
      "https://arxiv.org/pdf/2204.02448",
      "Predicting Mobile UI Tappability (CHI 2022)"),
    (10, "dinov2",
      "https://arxiv.org/pdf/2304.07193",
      "DINOv2: Learning Robust Visual Features (TMLR 2024)"),
    (12, "clip",
      "https://arxiv.org/pdf/2103.00020",
      "CLIP: Learning Transferable Visual Models (ICML 2021)"),
    (13, "screenai",
      "https://arxiv.org/pdf/2402.04615",
      "ScreenAI: Vision-Language Model for UI (IJCAI 2024)"),
    (17, "uiclip",
      "https://arxiv.org/pdf/2404.12500",
      "UIClip: Data-driven UI Design Quality (UIST 2024)"),
    (19, "swin_transformer",
      "https://arxiv.org/pdf/2103.14030",
      "Swin Transformer: Hierarchical ViT (ICCV 2021)"),
    (20, "resnet",
      "https://arxiv.org/pdf/1512.03385",
      "ResNet: Deep Residual Learning (CVPR 2016)"),
    (21, "vit",
      "https://arxiv.org/pdf/2010.11929",
      "ViT: An Image is Worth 16x16 Words (ICLR 2021)"),
    (22, "gradcam",
      "https://arxiv.org/pdf/1610.02391",
      "Grad-CAM: Visual Explanations from Deep Networks (ICCV 2017)"),
    (23, "gradcam_pp",
      "https://arxiv.org/pdf/1710.11063",
      "Grad-CAM++: Generalized Gradient-Based Explanations (WACV 2018)"),
    (28, "adamw",
      "https://arxiv.org/pdf/1711.05101",
      "AdamW: Decoupled Weight Decay Regularization (ICLR 2019)"),
    (29, "finetuning_distortion",
      "https://arxiv.org/pdf/2202.10054",
      "Fine-Tuning can Distort Pretrained Features (ICLR 2022)"),
    (30, "how_transferable",
      "https://arxiv.org/pdf/1411.1792",
      "How Transferable are Features in Deep NNs? (NeurIPS 2014)"),
    (31, "gan",
      "https://arxiv.org/pdf/1406.2661",
      "GAN: Generative Adversarial Nets (NeurIPS 2014)"),
    (34, "inception",
      "https://arxiv.org/pdf/1409.4842",
      "Going Deeper with Convolutions / Inception (CVPR 2015)"),
    (42, "simclr",
      "https://arxiv.org/pdf/2002.05709",
      "SimCLR: Simple Framework for Contrastive Learning (ICML 2020)"),
    (43, "moco",
      "https://arxiv.org/pdf/1911.05722",
      "MoCo: Momentum Contrast for Unsupervised Learning (CVPR 2020)"),
    (44, "dino",
      "https://arxiv.org/pdf/2104.14294",
      "DINO: Emerging Properties in Self-Supervised ViTs (ICCV 2021)"),
    (45, "sgdr",
      "https://arxiv.org/pdf/1608.03983",
      "SGDR: Stochastic Gradient Descent with Warm Restarts (ICLR 2017)"),

    # ── EfficientNet — PMLR ───────────────────────────────────────────────────
    (18, "efficientnet",
      "https://proceedings.mlr.press/v97/tan19a/tan19a.pdf",
      "EfficientNet: Rethinking Model Scaling (ICML 2019)"),

    # ── ACL Anthology ─────────────────────────────────────────────────────────
    (26, "kendall_tau_nlp",
      "https://aclanthology.org/J06-4002.pdf",
      "Automatic Evaluation of Information Ordering: Kendall Tau (CL 2006)"),

    # ── JMLR ──────────────────────────────────────────────────────────────────
    (33, "tsne",
      "https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf",
      "Visualizing Data using t-SNE (JMLR 2008)"),
]

# ── manual download list ──────────────────────────────────────────────────────
MANUAL = [
    ( 2, "tractinsky_2000_beautiful_usable.pdf",
      "https://doi.org/10.1016/S0953-5438(00)00031-7",
      "Tractinsky et al. (2000) What is beautiful is usable — Interact. Comput."),
    ( 3, "cyr_2010_colour_appeal.pdf",
      "https://doi.org/10.1016/j.ijhcs.2009.08.005",
      "Cyr et al. (2010) Colour appeal in website design — IJHCS"),
    ( 4, "google_play_statistics_2024.pdf",
      "https://www.statista.com/statistics/276623/number-of-apps-available-in-leading-app-stores/",
      "Google Play Store Statistics (Statista 2024) — ต้อง screenshot หรือ export"),
    ( 5, "gui_component_yolo_2024.pdf",
      "https://ieeexplore.ieee.org/document/10415929/",
      "Muhammad & Yeasin (2024) GUI Component Detection — IEEE ACT (ต้องการ access)"),
    ( 6, "object_detection_gui_survey.pdf",
      "https://dl.acm.org/doi/10.1145/3503252",
      "Chen et al. (2022) Object Detection in GUI Survey — ACM CS"),
    ( 9, "clip_ui_understanding.pdf",
      "https://dl.acm.org/doi/10.1145/3581641.3584038",
      "Jiang et al. (2023) CLIP-based UI Understanding — ACM IUI"),
    (11, "rico_dataset.pdf",
      "https://dl.acm.org/doi/10.1145/3126594.3126651",
      "Deka et al. (2017) RICO Dataset — ACM UIST (ต้องการ access)"),
    (14, "accessibility_mobile_apps.pdf",
      "https://ieeexplore.ieee.org/document/9956022",
      "Xiong et al. (2022) Accessibility Evaluation — IEEE ICSEA"),
    (15, "deep_learning_visual_aesthetics.pdf",
      "https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3255",
      "Ramakrishnan et al. (2024) Visual Aesthetics of Mobile UIs — JBCS"),
    (16, "ai_aesthetics_cnn.pdf",
      "https://doi.org/10.1049/ccs2.12055",
      "Kim et al. (2022) AI-Driven Aesthetics Prediction — Cogn. Comput. Syst."),
    (24, "eigencam.pdf",
      "https://link.springer.com/article/10.1007/s42979-021-00449-3",
      "Muhammad & Yeasin (2020) Eigen-CAM — IJCNN / SpringerNature"),
    (25, "kendall_1938.pdf",
      "https://www.jstor.org/stable/2332226",
      "Kendall (1938) A New Measure of Rank Correlation — Biometrika (JSTOR)"),
    (27, "bootstrap_efron.pdf",
      "https://www.amazon.com/Introduction-Bootstrap-Monographs-Statistics-Probability/dp/0412042312",
      "Efron & Tibshirani (1993) An Introduction to the Bootstrap — หนังสือ"),
    (32, "data_augmentation_classification.pdf",
      "https://ieeexplore.ieee.org/document/8227475",
      "Deng et al. (2017) Understanding Data Augmentation — DICTA"),
    (35, "visual_complexity_web.pdf",
      "https://dl.acm.org/doi/10.1145/2556288.2557020",
      "Reinecke & Nachtigall (2014) Quantifying Visual Complexity — CHI"),
    (36, "aesthetic_usability_data_viz.pdf",
      "https://ieeexplore.ieee.org/document/4272015",
      "Cawthon & Moere (2007) Aesthetic on Usability of Data Viz — IEEE IV"),
    (37, "aesthetics_utility_hci.pdf",
      "https://doi.org/10.1080/10447318.2011.586927",
      "Harrison et al. (2012) Aesthetics and Utility in HCI — IJHCI"),
    (38, "combinatorial_optimization_gui.pdf",
      "https://ieeexplore.ieee.org/document/8978428",
      "Oulasvirta et al. (2020) Combinatorial Optimization of GUI — Proc. IEEE"),
    (39, "aesthetic_image_harmonization.pdf",
      "https://ieeexplore.ieee.org/document/9511292",
      "Deng et al. (2021) Aesthetic Image Harmonization — IEEE Trans. Cybern."),
    (40, "aesthetics_image_synthesis.pdf",
      "https://ieeexplore.ieee.org/document/9705562",
      "Jiang et al. (2022) Aesthetics-Driven Image Synthesis — IEEE TCSVT"),
    (41, "ui_aesthetic_score_kiise.pdf",
      "https://www.dbpia.co.kr/",
      "Ko et al. (2023) UI Aesthetic Score Prediction — KIISE (ภาษาเกาหลี)"),
    (46, "layercam.pdf",
      "https://ieeexplore.ieee.org/document/9462463",
      "Jiang et al. (2021) LayerCAM — IEEE TIP (doi:10.1109/TIP.2021.3089943)"),
]


# ── download function ─────────────────────────────────────────────────────────
def download(url, dest, label):
    if dest.exists():
        size = dest.stat().st_size
        if size > 10_000:
            print(f"  [SKIP] {dest.name}  ({size//1024} KB already exists)")
            return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 5_000:
            print(f"  [WARN] {dest.name} — response too small ({len(data)} bytes), likely blocked")
            return False
        dest.write_bytes(data)
        print(f"  [OK]   {dest.name}  ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {dest.name} — {e}")
        return False


# ── main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("downloading references — automatic sources")
print("=" * 65)

ok_list, fail_list = [], []

for (num, prefix, url, note) in REFS:
    fname = f"[{num:02d}]_{prefix}.pdf"
    dest  = REFS_DIR / fname
    print(f"\n[{num:02d}] {note}")
    success = download(url, dest, note)
    (ok_list if success else fail_list).append((num, fname, url, note))
    if success:
        time.sleep(1.2)   # polite delay

# ── write manual download list ────────────────────────────────────────────────
manual_txt = REFS_DIR.parent / "manual_download.txt"
with open(manual_txt, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("ดาวน์โหลดเองจากลิงก์ด้านล่าง (ต้องการ institutional access หรือเปิดในเบราว์เซอร์)\n")
    f.write("บันทึกไฟล์ลงใน:  references/pdfs/\n")
    f.write("=" * 70 + "\n\n")
    for (num, fname, url, note) in MANUAL:
        f.write(f"[{num:02d}] {note}\n")
        f.write(f"     ไฟล์: {fname}\n")
        f.write(f"     URL:  {url}\n\n")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"สำเร็จ:           {len(ok_list):2d} / {len(REFS)} ไฟล์")
print(f"ล้มเหลว:          {len(fail_list):2d} ไฟล์")
print(f"ดาวน์โหลดเองด้วย: {len(MANUAL):2d} ไฟล์  (ดูรายชื่อใน manual_download.txt)")
print(f"\nไฟล์ทั้งหมดอยู่ใน:  references/pdfs/")
print(f"รายการ manual:     references/manual_download.txt")
if fail_list:
    print("\nล้มเหลว:")
    for num, fname, url, note in fail_list:
        print(f"  [{num:02d}] {note}")
        print(f"       ลองเปิด URL เอง: {url}")
