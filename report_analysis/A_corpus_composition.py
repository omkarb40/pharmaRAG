"""
Output A: Corpus composition.
Reads data/processed/chunks.jsonl (723 chunks, real schema: chunk_id, drug_name,
generic_name, set_id, section_name, loinc_code, chunk_index, total_chunks, text).
Run from project root: python report_analysis/A_corpus_composition.py
"""
import json, csv, collections, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

chunks = [json.loads(l) for l in open(ROOT / "data/processed/chunks.jsonl")]

# A1: totals
total_chunks = len(chunks)
distinct_drugs = len(set(c["drug_name"] for c in chunks))
with open(OUT / "A1_totals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["total_chunks", total_chunks])
    w.writerow(["distinct_drugs", distinct_drugs])

# A2: chunks per drug, descending
by_drug = collections.Counter(c["drug_name"] for c in chunks)
with open(OUT / "A2_chunks_per_drug.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug_name", "chunk_count"])
    for d, n in by_drug.most_common():
        w.writerow([d, n])

# A3: chunks per SPL section, descending
by_section = collections.Counter(c["section_name"] for c in chunks)
with open(OUT / "A3_chunks_per_section.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["section_name", "chunk_count"])
    for s, n in by_section.most_common():
        w.writerow([s, n])

# A4: core-section gap check (boxed_warning, contraindications, warnings_and_precautions,
# dosage_and_administration, adverse_reactions)
CORE = ["boxed_warning", "contraindications", "warnings_and_precautions",
        "dosage_and_administration", "adverse_reactions"]
by_drug_sections = collections.defaultdict(set)
for c in chunks:
    by_drug_sections[c["drug_name"]].add(c["section_name"])
with open(OUT / "A4_core_section_gaps.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug_name", "missing_core_sections"])
    for d in sorted(by_drug_sections):
        missing = [s for s in CORE if s not in by_drug_sections[d]]
        if missing:
            w.writerow([d, ";".join(missing)])

# A5: full drug x section presence/count matrix (28 x 9)
ALL_SECTIONS = sorted(by_section.keys())
by_drug_counts = collections.defaultdict(collections.Counter)
for c in chunks:
    by_drug_counts[c["drug_name"]][c["section_name"]] += 1
with open(OUT / "A5_drug_section_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug_name"] + ALL_SECTIONS)
    for d in sorted(by_drug_counts):
        w.writerow([d] + [by_drug_counts[d].get(s, 0) for s in ALL_SECTIONS])

print(f"total_chunks={total_chunks} distinct_drugs={distinct_drugs}")
print("Wrote A1-A5 to", OUT)
