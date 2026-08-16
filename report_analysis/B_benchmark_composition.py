"""
Output B: Benchmark composition.
Reads evaluation/test_queries_v2.json (85 records; real fields: id, query, query_type,
scope, expected_drug, expected_drug_aliases, expected_section, acceptable_sections,
reference_snippet, reference_chunk_id, expected_behavior, category, difficulty).
Run from project root: python report_analysis/B_benchmark_composition.py
"""
import json, csv, collections, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

q = json.load(open(ROOT / "evaluation/test_queries_v2.json"))

# B1: category x expected_behavior counts
cat_behavior = collections.Counter((r["category"], r["expected_behavior"]) for r in q)
cats = sorted(set(r["category"] for r in q))
with open(OUT / "B1_category_by_behavior.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "answer", "refuse", "total"])
    for c in cats:
        a = cat_behavior.get((c, "answer"), 0)
        r = cat_behavior.get((c, "refuse"), 0)
        w.writerow([c, a, r, a + r])

# B2: totals
total = len(q)
answerable = sum(1 for r in q if r["expected_behavior"] == "answer")
refuse = sum(1 for r in q if r["expected_behavior"] == "refuse")
with_gold = sum(1 for r in q if r.get("reference_chunk_id"))
with open(OUT / "B2_totals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["total_queries", total])
    w.writerow(["answerable_queries", answerable])
    w.writerow(["refuse_queries", refuse])
    w.writerow(["queries_with_gold_chunk", with_gold])

# B3: refuse split by scope
refuse_scope = collections.Counter(r["scope"] for r in q if r["expected_behavior"] == "refuse")
with open(OUT / "B3_refuse_split_by_scope.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scope", "count"])
    for s, n in refuse_scope.most_common():
        w.writerow([s, n])

print(f"total={total} answerable={answerable} refuse={refuse} with_gold={with_gold}")
print("refuse split:", dict(refuse_scope))
print("Wrote B1-B3 to", OUT)
