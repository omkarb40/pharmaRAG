"""
Output C: Per-category retrieval performance across the 5 pipeline configs.
Canonical per-query files (confirmed via sha256 match against report Config C
values 0.845/0.603/0.978/0.932/0.111 -- see verification script):
  evaluation/results/full_pipeline_results_{baseline,configN_noagents,
  optA_expand,optB_context,optC_rerank}.json
Metrics restricted to answerable queries (expected_behavior == "answer"), n=58.
Run from project root: python report_analysis/C_retrieval_performance.py
"""
import json, csv, collections, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["configN_noagents", "baseline", "optA_expand", "optB_context", "optC_rerank"]
LABELS = {"configN_noagents": "N", "baseline": "Baseline", "optA_expand": "A",
          "optB_context": "B", "optC_rerank": "C"}

data = {}
for cfg in CONFIGS:
    d = json.load(open(ROOT / f"evaluation/results/full_pipeline_results_{cfg}.json"))
    data[cfg] = [r for r in d if r["expected_behavior"] == "answer"]

categories = sorted(set(r["category"] for r in data["optC_rerank"]))


def avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


recall_matrix, gold_matrix, n_matrix = {}, {}, {}
for cat in categories:
    recall_matrix[cat] = {}
    gold_matrix[cat] = {}
    for cfg in CONFIGS:
        rows = [r for r in data[cfg] if r["category"] == cat]
        n_matrix[cat] = len(rows)
        recall_matrix[cat][cfg] = avg([r["recall_at_5"] for r in rows])
        gold_matrix[cat][cfg] = avg([r["gold_chunk_recall_at_5"] for r in rows])

with open(OUT / "C1_recall_by_category.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "n"] + [LABELS[c] for c in CONFIGS])
    for cat in categories:
        w.writerow([cat, n_matrix[cat]] + [recall_matrix[cat][c] for c in CONFIGS])

with open(OUT / "C2_gold_chunk_recall_by_category.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "n"] + [LABELS[c] for c in CONFIGS])
    for cat in categories:
        w.writerow([cat, n_matrix[cat]] + [gold_matrix[cat][c] for c in CONFIGS])

print("Categories:", categories)
print("Wrote C1-C2 to", OUT)
