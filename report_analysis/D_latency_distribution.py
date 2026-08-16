"""
Output D: Latency distribution.
Source: per-config evaluation/results/aggregate_metrics_{config}.json "latency"
blocks, computed over all 85 queries per config (n=85 each; NOT the audit log,
which has only 10 records -- see Phase-1 note).
Each stage block stores avg_ms and p95_ms only -- no median, no max, no stored
end-to-end total. End-to-end MEAN is derived exactly as the sum of the four
per-stage means (E[a+b+c+d] = E[a]+E[b]+E[c]+E[d] always holds). End-to-end
P95 is NOT derived this way (sum of marginal P95s overstates true P95) and is
reported as not derivable. Median and max are not derivable from this source
at all.
Run from project root: python report_analysis/D_latency_distribution.py
"""
import json, csv, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["configN_noagents", "baseline", "optA_expand", "optB_context", "optC_rerank"]
LABELS = {"configN_noagents": "N", "baseline": "Baseline", "optA_expand": "A",
          "optB_context": "B", "optC_rerank": "C"}
STAGES = ["routing", "retrieval", "generation", "validation"]

rows = []
for cfg in CONFIGS:
    d = json.load(open(ROOT / f"evaluation/results/aggregate_metrics_{cfg}.json"))
    lat = d["latency"]
    n = d["total_queries"]
    total_avg = sum(lat[s]["avg_ms"] for s in STAGES)
    row = {"config": LABELS[cfg], "n": n}
    for s in STAGES:
        row[f"{s}_avg_ms"] = lat[s]["avg_ms"]
        row[f"{s}_p95_ms"] = lat[s]["p95_ms"]
    row["end_to_end_avg_ms_derived"] = round(total_avg, 1)
    row["end_to_end_p95_ms"] = "not derivable (sum of marginal P95s overstates true P95; per-query totals not in aggregate file)"
    rows.append(row)

fieldnames = ["config", "n"] + [f"{s}_{m}" for s in STAGES for m in ("avg_ms", "p95_ms")] + \
             ["end_to_end_avg_ms_derived", "end_to_end_p95_ms"]
with open(OUT / "D1_latency_by_config.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow(row)

for row in rows:
    print(row["config"], "n=", row["n"], "end-to-end avg=", row["end_to_end_avg_ms_derived"], "ms")
print("Wrote D1 to", OUT)
print("NOTE: median and max latency are not derivable from aggregate_metrics_*.json (only avg_ms/p95_ms stored per stage).")
