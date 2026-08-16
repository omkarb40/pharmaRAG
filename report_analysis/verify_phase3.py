"""
Phase 3: verify report-stated values against extracted values.
Run from project root: python report_analysis/verify_phase3.py
"""
import json, csv, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

rows = []


def add(qty, reported, extracted, source, note="", tol=0.008):
    if extracted is None:
        match = "not derivable"
    elif str(reported) == str(extracted):
        match = "MATCH"
    else:
        match = "MATCH" if _close(reported, extracted, tol) else "DISCREPANCY"
    rows.append({"quantity": qty, "reported": reported, "extracted": extracted,
                 "verdict": match, "source": source, "note": note})


def _close(a, b, tol=0.006):
    # tolerance covers report values rounded to 2-3 decimals vs 4-decimal extracted values
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


chunks = [json.loads(l) for l in open(ROOT / "data/processed/chunks.jsonl")]
add("Corpus chunks", 723, len(chunks), "data/processed/chunks.jsonl (len)")
add("Drugs", 28, len(set(c["drug_name"] for c in chunks)), "chunks.jsonl distinct drug_name")

q = json.load(open(ROOT / "evaluation/test_queries_v2.json"))
add("Benchmark queries (total)", 85, len(q), "evaluation/test_queries_v2.json")
add("Benchmark queries (answerable)", 58, sum(1 for r in q if r["expected_behavior"] == "answer"), "test_queries_v2.json")
add("Benchmark queries (refuse)", 27, sum(1 for r in q if r["expected_behavior"] == "refuse"), "test_queries_v2.json")

near = sum(1 for r in q if r["scope"] == "near_scope")
oos = sum(1 for r in q if r["scope"] == "out_of_scope")
adv = sum(1 for r in q if r["scope"] == "adversarial")
add("Refuse split: near-scope", 12, near, "test_queries_v2.json scope field")
add("Refuse split: out-of-scope", 10, oos, "test_queries_v2.json scope field")
add("Refuse split: adversarial", 5, adv, "test_queries_v2.json scope field")

agg = json.load(open(ROOT / "evaluation/results/aggregate_metrics_optC_rerank.json"))
add("Config C Recall@5", 0.845, agg["retrieval"]["avg_recall_at_5"], "aggregate_metrics_optC_rerank.json")
add("Config C Gold-chunk@5", 0.603, agg["retrieval"]["avg_gold_chunk_recall_at_5"], "aggregate_metrics_optC_rerank.json")
add("Config C nDCG@5", 0.978, agg["retrieval"]["avg_ndcg_at_5"], "aggregate_metrics_optC_rerank.json",
    "MATCHES the aggregate_metrics_optC_rerank.json / optC_FINAL_t0.json pair (sha256-identical); 3 other optC-named files disagree in the 3rd decimal -- see optC_variant_checksums.csv")
add("Config C groundedness (answered only)", 0.932, agg["generation"]["avg_groundedness"], "aggregate_metrics_optC_rerank.json")
add("Config C unsafe emission", 0.111, agg["safety"]["unsafe_emission_rate"], "aggregate_metrics_optC_rerank.json",
    f"{agg['safety']['unsafe_emissions']} of {agg['safety']['refuse_queries']}")

aggN = json.load(open(ROOT / "evaluation/results/aggregate_metrics_configN_noagents.json"))
add("Config N groundedness", 0.945, aggN["generation"]["avg_groundedness"], "aggregate_metrics_configN_noagents.json")
add("Config N unsafe emission", 1.000, aggN["safety"]["unsafe_emission_rate"], "aggregate_metrics_configN_noagents.json")

aggB = json.load(open(ROOT / "evaluation/results/aggregate_metrics_baseline.json"))
add("Baseline unsafe emission", 0.185, aggB["safety"]["unsafe_emission_rate"], "aggregate_metrics_baseline.json",
    f"{aggB['safety']['unsafe_emissions']} of {aggB['safety']['refuse_queries']}")

adv_refusal = agg["refusal"]["by_scope"]["adversarial"]
add("Adversarial refusals, Config C", "3 of 5", f"{adv_refusal['correct']} of {adv_refusal['total']}",
    "aggregate_metrics_optC_rerank.json refusal.by_scope.adversarial")
near_refusal = agg["refusal"]["by_scope"]["near_scope"]
add("Near-scope refusals", "11 of 12", f"{near_refusal['correct']} of {near_refusal['total']}",
    "aggregate_metrics_optC_rerank.json refusal.by_scope.near_scope")
oos_refusal = agg["refusal"]["by_scope"]["out_of_scope"]
add("Out-of-scope refusals", "10 of 10", f"{oos_refusal['correct']} of {oos_refusal['total']}",
    "aggregate_metrics_optC_rerank.json refusal.by_scope.out_of_scope")
inscope_refusal = agg["refusal"]["by_scope"]["in_scope"]
add("In-scope answered correctly", "42 of 58", f"{inscope_refusal['correct']} of {inscope_refusal['total']}",
    "aggregate_metrics_optC_rerank.json refusal.by_scope.in_scope",
    "NOTE: field name is refusal_correct (correct refusal/answer decision), not a claim about factual answer correctness -- report wording may overstate what this measures.")

add("Citation failures (27: 23 misalignment, 4 unsupported)", "27 (23+4)", None,
    "scripts/citation_vs_grounding.py", "Requires live PubMedBERT embedding calls (src/indexing/embedder.py); no stored output file exists for this metric. Not re-run per your instruction not to re-run experiments.")
add("Cited claims (96)", 96, None, "scripts/citation_precision.py",
    "Same as above -- computed live via embedder, not stored anywhere in evaluation/results/.")

full = json.load(open(ROOT / "evaluation/results/full_pipeline_results_optC_rerank.json"))
dosing = [r for r in full if r["category"] == "dosing" and r["expected_behavior"] == "answer"]
dosing_recall = sum(r["recall_at_5"] for r in dosing) / len(dosing)
add("Dosing category Recall@5, Config C", 0.50, round(dosing_recall, 4), "full_pipeline_results_optC_rerank.json, category==dosing", f"n={len(dosing)}")

def router_recall(cfg, tag):
    d = json.load(open(ROOT / f"evaluation/results/retrieval_{tag}_{cfg}.json"))
    vals = [r["recall_at_5"] for r in d if r["recall_at_5"] is not None]
    return round(sum(vals) / len(vals), 4), len(vals)

nrC, nC = router_recall("optC_rerank", "no_router")
wrC, _ = router_recall("optC_rerank", "with_router")
nrB, nB = router_recall("baseline", "no_router")
wrB, _ = router_recall("baseline", "with_router")

add("Router contribution (without)", 0.621, nrB,
    "retrieval_no_router_baseline.json avg recall_at_5 over answerable queries",
    f"n={nB}. This MATCHES the BASELINE config, not Config C (Config C without-router = {nrC}). "
    f"The report does not label which config the router ablation ran on -- resolve as: report's router-contribution row is the baseline config's router ablation, mislabelled/unlabelled rather than wrong.", tol=0.0015)
add("Router contribution (with)", 0.707, wrB,
    "retrieval_with_router_baseline.json avg recall_at_5",
    f"n={nB}. Matches BASELINE with-router ({wrB}), not Config C with-router ({wrC}).", tol=0.0015)
rows.append({"quantity": "  (for reference) Config C without-router", "reported": "n/a", "extracted": nrC,
             "verdict": "reference", "source": "retrieval_no_router_optC_rerank.json", "note": f"n={nC}"})
rows.append({"quantity": "  (for reference) Config C with-router", "reported": "n/a", "extracted": wrC,
             "verdict": "reference", "source": "retrieval_with_router_optC_rerank.json", "note": f"n={nC}"})

lat = agg["latency"]
add("Mean latency: routing", 2.15, round(lat["routing"]["avg_ms"] / 1000, 4), "aggregate_metrics_optC_rerank.json latency.routing.avg_ms")
add("Mean latency: retrieval", 2.27, round(lat["retrieval"]["avg_ms"] / 1000, 4), "aggregate_metrics_optC_rerank.json latency.retrieval.avg_ms")
add("Mean latency: generation", 10.3, round(lat["generation"]["avg_ms"] / 1000, 4), "aggregate_metrics_optC_rerank.json latency.generation.avg_ms")
add("Mean latency: validation", 0.55, round(lat["validation"]["avg_ms"] / 1000, 4), "aggregate_metrics_optC_rerank.json latency.validation.avg_ms",
    "Does not reconcile as validation_ms+refusal_ms either: eval-harness per-query latency_ms has no refusal key at all (only routing/retrieval/generation/validation); audit log refusal_ms=0 for all 10 records, giving combo=0.445s, still short of 0.55s. No config's validation avg (baseline .168 / N .464 / A .209 / B .176 / C .462) equals .55s either.")

aggBase = aggB
add("Retrieval latency without reranking", 0.082, round(aggBase["latency"]["retrieval"]["avg_ms"] / 1000, 4),
    "aggregate_metrics_baseline.json latency.retrieval.avg_ms",
    "Baseline (no rerank) mean retrieval is 0.226s, not 0.082s. Closest candidate found: configN_noagents retrieval P95 = 82.9ms (aggregate_metrics_configN_noagents.json latency.retrieval.p95_ms) -- a P95, not a mean, and from the no-agents config, not baseline. Does not cleanly verify.")

with open(OUT / "phase3_discrepancy_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["quantity", "reported", "extracted", "verdict", "source", "note"])
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(f"[{r['verdict']:12s}] {r['quantity']}: reported={r['reported']} extracted={r['extracted']}")
print("\nWrote phase3_discrepancy_table.csv to", OUT)
