"""
Output E: Confidence distribution by decision tier.
Primary source: evaluation/results/full_pipeline_results_optC_rerank.json
(Config C, n=85 -- the deployed/reported configuration). The production audit
log (logs/audit_20260422.jsonl) has only 10 records and is reported separately
as a small cross-check, per the Phase-1 finding that it cannot support a
distribution on its own.
Bands (from src/generation/refusal_guard.py: refuse_threshold=0.45,
caution_threshold=0.65): <0.45 -> INSUFFICIENT_EVIDENCE, 0.45-0.65 ->
ANSWER_WITH_CAUTION, >=0.65 -> ANSWER.
Run from project root: python report_analysis/E_confidence_distribution.py
"""
import json, csv, pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "report_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def band(score):
    if score < 0.45:
        return "INSUFFICIENT_EVIDENCE"
    elif score < 0.65:
        return "ANSWER_WITH_CAUTION"
    else:
        return "ANSWER"


def analyze(records, score_key, decision_key, label):
    rows, mismatches, tier_counts = [], [], {}
    for r in records:
        score = r.get(score_key)
        decision = r.get(decision_key)
        if score is None or decision is None:
            continue
        expected = band(score)
        tier_counts.setdefault(decision, 0)
        tier_counts[decision] += 1
        match = (expected == decision)
        if not match:
            mismatches.append({"id": r.get("query_id") or r.get("request_id"),
                                "confidence": score, "recorded_decision": decision,
                                "expected_band": expected})
        rows.append({"id": r.get("query_id") or r.get("request_id"),
                      "confidence": score, "decision": decision,
                      "band_from_confidence": expected, "match": match})
    with open(OUT / f"E_{label}_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "confidence", "decision", "band_from_confidence", "match"])
        w.writeheader()
        w.writerows(rows)
    print(f"--- {label} (n={len(rows)}) ---")
    print("decision tier counts:", tier_counts)
    print("mismatches (band != recorded decision):", len(mismatches))
    for m in mismatches:
        print("  ", m)
    return tier_counts, mismatches


configC = json.load(open(ROOT / "evaluation/results/full_pipeline_results_optC_rerank.json"))
analyze(configC, "confidence_score", "confidence_decision", "configC_n85")

audit = [json.loads(l) for l in open(ROOT / "logs/audit_20260422.jsonl")]
analyze(audit, "confidence_score", "confidence_decision", "auditlog_n10")
