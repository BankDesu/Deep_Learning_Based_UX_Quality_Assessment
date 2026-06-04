"""
remove_refs.py
ลบ [4,31,32,33,34,35,37,38,39,40] ออกจาก รูปเล่ม.docx
และ renumber citations ที่เหลือให้ต่อเนื่อง [1]-[36]
"""
import sys, zipfile, re, shutil, os
sys.stdout.reconfigure(encoding="utf-8")

SRC = r"E:\Works\KMUTT\CV_Project\Multi-Modal-Learning-of-Visual-Structural-and-Attention-Features-for-UX-Quality-Assessment\รูปเล่ม.docx"
DST = SRC  # overwrite

REMOVE = {4, 31, 32, 33, 34, 35, 37, 38, 39, 40}

# renumber map: old → new (after removing the 10 refs)
# remaining old numbers: 1-3, 5-30, 36, 41-46
remaining = [n for n in range(1, 47) if n not in REMOVE]
REMAP = {old: new for new, old in enumerate(remaining, start=1)}
# e.g. 5→4, 6→5, ... 36→30, 41→31, 42→32 ... 46→36
print("Renumber map:")
changed = {k:v for k,v in REMAP.items() if k != v}
for k,v in sorted(changed.items()):
    print(f"  [{k}] → [{v}]")

# ── load ──────────────────────────────────────────────────────────────────────
tmp = SRC + ".bak"
shutil.copy2(SRC, tmp)

with zipfile.ZipFile(SRC, "r") as z:
    names = z.namelist()
    files = {n: z.read(n) for n in names}

xml = files["word/document.xml"].decode("utf-8")

# ── step 1: remove [4] from body text ────────────────────────────────────────
# [4] appears once: "...expert review [4]."
# pattern: space + [4] (not preceded by a letter = not array index)
before = len(re.findall(r"(?<![a-zA-Z])\[4\]", xml))
xml = re.sub(r"\s*(?<![a-zA-Z])\[4\]", "", xml)
after = len(re.findall(r"(?<![a-zA-Z])\[4\]", xml))
print(f"\nStep 1: [4] removed from body: {before} → {after}")

# ── step 2: remove reference list entries for all 10 refs ────────────────────
removed_entries = 0
for n in sorted(REMOVE):
    # reference entry paragraph pattern: starts with [N] followed by author names
    # the paragraph XML contains the text "[N]  Author..."
    pattern = (
        r"<w:p\b[^>]*>"           # paragraph start
        r"(?:(?!<w:p\b).)*?"      # anything (non-greedy, no nested para)
        rf"(?<![a-zA-Z])\[{n}\]"  # citation number NOT preceded by letter
        r"(?:(?!<\/w:p>).)*?"     # rest of paragraph
        r"<\/w:p>"                # paragraph end
    )
    new_xml, count = re.subn(pattern, "", xml, flags=re.DOTALL)
    if count:
        xml = new_xml
        removed_entries += count
        print(f"  Removed ref entry [{n}] ({count} paragraph)")

print(f"\nStep 2: {removed_entries} reference entries removed")

# ── step 3: renumber citations (high→low to avoid collision) ─────────────────
# Use a temporary placeholder approach: replace [N] with [TMPN] first
# then replace [TMPN] with new numbers

# mark all citations with TMP prefix (skip array index like features[3])
def replace_cite(m):
    num = int(m.group(1))
    if num in REMAP:
        return f"[TMP{REMAP[num]}]"
    return m.group(0)

# match [N] not preceded by letter (to skip features[3], features[5])
xml = re.sub(r"(?<![a-zA-Z])\[(\d+)\]", replace_cite, xml)

# remove TMP prefix
xml = xml.replace("[TMP", "[")

# verify no old citation numbers remain (except features[3] etc.)
remaining_old = []
for n in sorted(REMOVE - {4}):  # 4 already removed from body
    hits = re.findall(rf"(?<![a-zA-Z])\[{n}\]", xml)
    if hits:
        remaining_old.append(n)

print(f"\nStep 3: renumbering done")
if remaining_old:
    print(f"  WARNING: old numbers still present: {remaining_old}")
else:
    print(f"  All citations renumbered cleanly")

# verify final citation count
final_cites = sorted(set(int(x) for x in re.findall(r"(?<![a-zA-Z])\[(\d+)\]", xml)))
print(f"\nFinal citations in document: {final_cites}")
print(f"Total unique: {len(final_cites)}")

# ── save ──────────────────────────────────────────────────────────────────────
files["word/document.xml"] = xml.encode("utf-8")
os.remove(SRC)
with zipfile.ZipFile(SRC, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.writestr(n, files[n])

os.remove(tmp)
print(f"\nSaved: {SRC}")
print("Done.")
