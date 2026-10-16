"""
Post-rebuild corpus verification (corpus fix, 2026-10-15; issues #3, #4, #5, #8).
Usage: python scripts/verify_corpus.py
Exit code 1 if any hard check fails. Writes evaluation/gold_reannotation_needed.json.
"""
import collections, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "data/processed/chunks.jsonl"
OLD = ROOT / "data/processed/chunks_pre_corpusfix_20261015.jsonl"
QUERIES = ROOT / "evaluation/test_queries_v2.json"
OUT = ROOT / "evaluation/gold_reannotation_needed.json"

# Labels that carry an FDA boxed warning (34066-1) in the source XML
EXPECT_BOXED = {"Lemtrada", "Mavenclad", "Copaxone", "Mitoxantrone", "Tysabri", "Tyruko", "Aubagio"}

def load(p): return [json.loads(l) for l in open(p)]
new, old = load(NEW), load(OLD)
fails = []

print(f"Chunks: {len(old)} before -> {len(new)} after")
by_drug = collections.Counter(c["drug_name"] for c in new)
print(f"Drugs: {len(by_drug)}  {dict(sorted(by_drug.items()))}")
if len(by_drug) != 28: fails.append(f"expected 28 drugs, got {len(by_drug)}")

# 1. One label per drug, no two drugs sharing a label
sid = collections.defaultdict(set)
for c in new: sid[c["drug_name"]].add(c["set_id"])
multi = {d: s for d, s in sid.items() if len(s) > 1}
shared = collections.defaultdict(list)
for d, s in sid.items(): shared[next(iter(s))].append(d)
shared = {k: v for k, v in shared.items() if len(v) > 1}
if multi: fails.append(f"drugs with >1 set_id: {multi}")
if shared: fails.append(f"set_ids shared by drugs: {shared}")

# 2. Cross-drug duplicate text (Tysabri/Tyruko, Avonex/Rebif were 37 + 19)
h = collections.defaultdict(set)
for c in new: h[hashlib.md5(c["text"].encode()).hexdigest()].add(c["drug_name"])
cross = collections.Counter(tuple(sorted(v)) for v in h.values() if len(v) > 1)
print(f"Cross-drug identical chunks: {dict(cross) or 'none'}")
# A biosimilar label legitimately copies much of its reference product's text,
# so Tysabri/Tyruko overlap is reported, not failed. Any other pair is a provenance bug.
if cross.get(("Tyruko", "Tysabri")):
    print(f"  note: {cross[('Tyruko', 'Tysabri')]} Tysabri/Tyruko identical chunks (biosimilar overlap; was 37/37 before fix)")
big = {k: v for k, v in cross.items() if v > 3 and k != ("Tyruko", "Tysabri")}
if cross.get(("Tyruko", "Tysabri"), 0) >= min(by_drug.get("Tysabri", 0), by_drug.get("Tyruko", 0)):
    fails.append("Tysabri and Tyruko are still fully identical: Tysabri label not replaced")
if big: fails.append(f"large cross-drug duplication remains: {big}")

# 3. Boxed warnings are real (34066-1) and present where expected
bw = [c for c in new if c["section_name"] == "boxed_warning"]
bw_drugs = {c["drug_name"] for c in bw}
bad_code = {c["loinc_code"] for c in bw} - {"34066-1"}
print(f"boxed_warning chunks: {len(bw)} across {sorted(bw_drugs)}")
if bad_code: fails.append(f"boxed_warning chunks with non-34066-1 codes: {bad_code}")
if EXPECT_BOXED - bw_drugs: fails.append(f"missing boxed warnings for: {sorted(EXPECT_BOXED - bw_drugs)}")

# 4. Unique chunk ids
ids = [c["chunk_id"] for c in new]
if len(ids) != len(set(ids)): fails.append("duplicate chunk_ids")

# 5. Gold impact on the 85-query benchmark
Q = json.load(open(QUERIES)); Q = Q["queries"] if isinstance(Q, dict) else Q
old_by, new_by = {c["chunk_id"]: c for c in old}, {c["chunk_id"]: c for c in new}
norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()
report = {"ok": [], "text_changed_snippet_still_present": [], "moved": [], "lost": [], "review_boxed": []}
for q in Q:
    rid = q.get("reference_chunk_id")
    if not rid: continue
    snip = norm(q.get("reference_snippet"))[:120]
    o, n = old_by.get(rid), new_by.get(rid)
    if q.get("expected_section") == "boxed_warning" or "boxed_warning" in (q.get("acceptable_sections") or []):
        report["review_boxed"].append(q["id"])
    if n and o and n["text"] == o["text"]:
        report["ok"].append(q["id"]); continue
    if n and snip and snip in norm(n["text"]):
        report["text_changed_snippet_still_present"].append(q["id"]); continue
    cands = [c["chunk_id"] for c in new if snip and snip in norm(c["text"])
             and c["drug_name"] == (o or {}).get("drug_name", q.get("expected_drug"))]
    (report["moved"] if cands else report["lost"]).append({"id": q["id"], "old": rid, "candidates": cands})

json.dump(report, open(OUT, "w"), indent=2)
print("\nGold check (85-query benchmark):")
for k, v in report.items(): print(f"  {k:<36} {len(v)}")
print(f"  -> details in {OUT.relative_to(ROOT)}")

print("\nRESULT:", "PASS" if not fails else "FAIL")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
