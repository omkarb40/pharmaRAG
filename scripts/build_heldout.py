"""
Build the held-out benchmark from the clinician's pairs.
  1. Copy evaluation/heldout/heldout_pairs_TEMPLATE.csv -> evaluation/heldout/heldout_pairs.csv
  2. One row per pair, transcribed from the raw .txt files (verbatim, no rewording).
  3. Fact-check premises (route, frequency, dose vs the label). A pair with a wrong premise gets
     exclude=yes + a reason. Decide ALL exclusions BEFORE running anything on the set.
  4. python scripts/build_heldout.py  -> evaluation/heldout/heldout_v1.json (+ .sha256)
Prints counts only, never question text, so its output can be shared safely.
"""
import csv, hashlib, json, re, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "evaluation/heldout/heldout_pairs.csv"
OUT = ROOT / "evaluation/heldout/heldout_v1.json"

brand_of = {r["generic_name"].lower(): r["drug_name"] for r in csv.DictReader(open(ROOT / "data/drug_lists/ms_drugs.csv"))}
brands = {v.lower(): v for v in brand_of.values()}
SECTIONS = [("boxed", "boxed_warning"), ("indication", "indications_and_usage"), ("dos", "dosage_and_administration"),
            ("contraind", "contraindications"), ("warning", "warnings_and_precautions"), ("precaution", "warnings_and_precautions"),
            ("adverse", "adverse_reactions"), ("side effect", "adverse_reactions"), ("interaction", "drug_interactions"),
            ("population", "use_in_specific_populations"), ("pregnan", "use_in_specific_populations"),
            ("pediatric", "use_in_specific_populations"), ("lactation", "use_in_specific_populations")]

def section(s):
    s = s.lower()
    for key, name in SECTIONS:
        if key in s: return name
    return None

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))   # utf-8-sig drops a BOM if present
errors, queries, excluded = [], [], []
seen = set()
for i, r in enumerate(rows, 2):
    if not any((v or "").strip() for k, v in r.items() if k):       # blank line
        continue
    if (r.get("pair_id") or "").strip() == "pair_id":               # header pasted twice
        print(f"  note: row {i} repeats the header line; skipped"); continue
    if None in r or any(ch in "".join(v for k, v in r.items() if k and v) for ch in "\u201c\u201d"):
        errors.append(f"row {i}: columns are broken (curly quotes or an unquoted comma). "
                      "In TextEdit: Format > Make Plain Text, Edit > Substitutions > turn OFF Smart Quotes, "
                      "then retype the straight double quotes around each question."); continue
    pid = r["pair_id"].strip()
    if pid in seen: errors.append(f"row {i}: duplicate pair_id {pid}")
    seen.add(pid)
    if r.get("exclude", "").strip().lower() in ("yes", "y", "1", "true"):
        excluded.append((pid, r.get("exclude_reason", "").strip() or "(no reason given)")); continue
    d = r["drug"].strip().lower()
    brand = brand_of.get(d) or brands.get(d)
    sec = section(r["label_section"])
    cat = r["category"].strip()
    if not brand: errors.append(f"row {i} ({pid}): drug '{r['drug']}' not in corpus list")
    if not sec: errors.append(f"row {i} ({pid}): label_section '{r['label_section']}' not recognised")
    if cat not in {"1", "2", "3", "4", "5"}: errors.append(f"row {i} ({pid}): category must be 1-5")
    if not r["question_answerable"].strip() or not r["question_decline"].strip():
        errors.append(f"row {i} ({pid}): a question is empty")
    if errors and errors[-1].startswith(f"row {i}"): continue
    generic = next((g for g, b in brand_of.items() if b == brand), d)
    common = {"expected_drug": brand, "expected_drug_aliases": [brand, generic], "pair_id": pid,
              "heldout_category": int(cat), "writer": r.get("writer", "").strip() or "annotator1",
              "difficulty": "", "query_type": "heldout"}
    queries.append({"id": f"HO-{pid}-A", "query": r["question_answerable"].strip(), "expected_behavior": "answer",
                    "scope": "heldout_answerable", "category": f"heldout_cat{cat}",
                    "expected_section": sec, "acceptable_sections": [sec], **common})
    queries.append({"id": f"HO-{pid}-B", "query": r["question_decline"].strip(), "expected_behavior": "refuse",
                    "scope": "heldout_decline", "category": f"heldout_cat{cat}",
                    "expected_section": None, "acceptable_sections": [], **common})

if errors:
    print("NOT BUILT. Fix these rows:"); [print("  -", e) for e in errors]; sys.exit(1)
json.dump(queries, open(OUT, "w"), indent=2)
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
(OUT.with_suffix(".json.sha256")).write_text(digest + "\n")
pairs = [q for q in queries if q["id"].endswith("-A")]
print(f"Built {OUT.relative_to(ROOT)}: {len(pairs)} pairs ({len(queries)} questions), sha256 {digest[:16]}")
print("  per category:", dict(sorted(collections.Counter(q['heldout_category'] for q in pairs).items())))
print("  drugs:", len({q['expected_drug'] for q in pairs}), " max pairs per drug:", max(collections.Counter(q['expected_drug'] for q in pairs).values()))
print("  writers:", dict(collections.Counter(q['writer'] for q in pairs)))
print("  answerable-twin sections:", dict(collections.Counter(q['expected_section'] for q in pairs)))
print(f"  excluded before any run: {len(excluded)}", excluded)
