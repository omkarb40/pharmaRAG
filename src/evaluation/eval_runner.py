"""
PharmaRAG Evaluation Harness

Runs the full test suite and produces metrics:
  - Retrieval: Recall@k, nDCG@k per query
  - Generation: Groundedness, hallucination rate
  - Refusal: Accuracy on in-scope vs out-of-scope queries
  - Ablation: Metrics with/without each agent

Usage: python -m src.evaluation.eval_runner
"""

import json
import time
import math
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.generation.generator import AnswerGenerator
from src.generation.evidence_validator import EvidenceValidator
from src.generation.refusal_guard import RefusalGuard
from configs.settings import settings


QUERIES_FILE = Path("evaluation/test_queries.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_queries() -> list[dict]:
    with open(QUERIES_FILE, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Retrieval Metrics
# ──────────────────────────────────────────────

def compute_recall_at_k(retrieved_chunks: list[dict], expected_drug: str,
                         expected_section: str, k: int = 5) -> float:
    """
    Recall@k: Did any of the top-k chunks come from the expected
    drug AND section?

    Returns 1.0 if at least one match, 0.0 otherwise.
    """
    if not expected_drug or not expected_section:
        return None  # Can't evaluate out-of-scope queries

    for chunk in retrieved_chunks[:k]:
        meta = chunk.get("metadata", {})
        drug_match = meta.get("drug_name", "").lower() == expected_drug.lower()
        # Also check generic name
        generic_match = meta.get("generic_name", "").lower() == expected_drug.lower()
        section_match = meta.get("section_name", "") == expected_section

        if (drug_match or generic_match) and section_match:
            return 1.0

    return 0.0


def compute_ndcg_at_k(retrieved_chunks: list[dict], expected_drug: str,
                       expected_section: str, k: int = 5) -> float:
    """
    nDCG@k: Measures ranking quality.
    Relevant chunks ranked higher get more credit.

    Relevance scoring:
      - Exact drug + section match: 3 (highly relevant)
      - Same drug, different section: 2 (relevant)
      - Different drug, same section: 1 (marginally relevant)
      - No match: 0
    """
    if not expected_drug or not expected_section:
        return None

    def relevance(chunk):
        meta = chunk.get("metadata", {})
        drug = meta.get("drug_name", "")
        generic = meta.get("generic_name", "")
        section = meta.get("section_name", "")

        drug_match = (drug.lower() == expected_drug.lower() or
                      generic.lower() == expected_drug.lower())
        section_match = section == expected_section

        if drug_match and section_match:
            return 3
        elif drug_match:
            return 2
        elif section_match:
            return 1
        return 0

    # DCG
    dcg = 0.0
    for i, chunk in enumerate(retrieved_chunks[:k]):
        rel = relevance(chunk)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (best possible ranking)
    ideal_rels = sorted([relevance(c) for c in retrieved_chunks[:k]], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return round(dcg / idcg, 4)


# ──────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────

def run_retrieval_eval(queries: list[dict], retriever: HybridRetriever,
                       router: QueryRouter = None) -> list[dict]:
    """
    Run retrieval evaluation on all queries.
    Optionally uses the query router for section filtering.
    """
    results = []

    for i, q in enumerate(queries):
        query_id = q["id"]
        query_text = q["query"]
        expected_drug = q.get("expected_drug")
        expected_section = q.get("expected_section")
        category = q.get("category", "")

        print(f"  [{i+1}/{len(queries)}] {query_id}: {query_text[:50]}...", end=" ")

        # Optional routing
        section_filter = None
        routed_sections = []
        if router:
            routed_sections = router.route(query_text)
            section_filter = routed_sections[0] if routed_sections else None

        # Retrieve
        t0 = time.time()
        if section_filter:
            filtered = retriever.search(query=query_text, top_k=5,
                                         section_filter=section_filter)
            unfiltered = retriever.search(query=query_text, top_k=5)
            seen_ids = set(r["chunk_id"] for r in filtered)
            merged = list(filtered)
            for r in unfiltered:
                if r["chunk_id"] not in seen_ids and len(merged) < 5:
                    merged.append(r)
                    seen_ids.add(r["chunk_id"])
            chunks = merged[:5]
        else:
            chunks = retriever.search(query=query_text, top_k=5)
        retrieval_time = (time.time() - t0) * 1000

        # If filtered retrieval returned too few, fallback
        if len(chunks) < 5 and section_filter:
            chunks = retriever.search(query=query_text, top_k=5)

        # Compute metrics
        recall = compute_recall_at_k(chunks, expected_drug, expected_section)
        ndcg = compute_ndcg_at_k(chunks, expected_drug, expected_section)

        status = "✓" if recall and recall > 0 else ("—" if recall is None else "✗")
        print(f"R@5={recall}  nDCG={ndcg}  {status}")

        results.append({
            "query_id": query_id,
            "query": query_text,
            "category": category,
            "difficulty": q.get("difficulty", ""),
            "expected_drug": expected_drug,
            "expected_section": expected_section,
            "routed_sections": routed_sections,
            "recall_at_5": recall,
            "ndcg_at_5": ndcg,
            "retrieval_time_ms": round(retrieval_time, 1),
            "top_5_drugs": [
                c.get("metadata", {}).get("drug_name", "") for c in chunks[:5]
            ],
            "top_5_sections": [
                c.get("metadata", {}).get("section_name", "") for c in chunks[:5]
            ],
            "top_5_scores": [
                round(c.get("fused_score", 0), 6) for c in chunks[:5]
            ],
        })

    return results


def run_full_pipeline_eval(queries: list[dict], retriever: HybridRetriever,
                           router: QueryRouter, generator: AnswerGenerator,
                           validator: EvidenceValidator,
                           refusal_guard: RefusalGuard) -> list[dict]:
    """
    Run full pipeline evaluation (retrieval + generation + agents).
    This is slower because it calls the LLM for each query.
    """
    results = []

    for i, q in enumerate(queries):
        query_id = q["id"]
        query_text = q["query"]
        category = q.get("category", "")
        expected_drug = q.get("expected_drug")
        expected_section = q.get("expected_section")

        print(f"  [{i+1}/{len(queries)}] {query_id}: {query_text[:50]}...")

        # Agent 1: Route
        t0 = time.time()
        routed = router.route(query_text)
        routing_ms = (time.time() - t0) * 1000

        # Retrieve
        t0 = time.time()
        section_filter = routed[0] if routed else None
        chunks = retriever.search(query=query_text, top_k=5,
                                   section_filter=section_filter)
        if len(chunks) < 5 and section_filter:
            chunks = retriever.search(query=query_text, top_k=5)
        retrieval_ms = (time.time() - t0) * 1000

        # Generate
        t0 = time.time()
        gen = generator.generate(query_text, chunks)
        generation_ms = (time.time() - t0) * 1000

        # Agent 2: Validate
        t0 = time.time()
        validation = validator.validate(gen["answer"], chunks)
        validation_ms = (time.time() - t0) * 1000

        # Agent 3: Refusal
        refusal = refusal_guard.evaluate(chunks, validation)

        # Retrieval metrics
        recall = compute_recall_at_k(chunks, expected_drug, expected_section)
        ndcg = compute_ndcg_at_k(chunks, expected_drug, expected_section)

        # Is the refusal correct?
        is_out_of_scope = category == "out_of_scope"
        refused = refusal["decision"] == "INSUFFICIENT_EVIDENCE"
        refusal_correct = (is_out_of_scope == refused)

        print(f"    Decision: {refusal['decision']} | "
              f"Groundedness: {validation['groundedness_score']:.0%} | "
              f"Refusal correct: {refusal_correct}")

        results.append({
            "query_id": query_id,
            "query": query_text,
            "category": category,
            "difficulty": q.get("difficulty", ""),
            "expected_drug": expected_drug,
            "expected_section": expected_section,
            "routed_sections": routed,
            "recall_at_5": recall,
            "ndcg_at_5": ndcg,
            "answer": gen["answer"][:300],
            "groundedness_score": validation["groundedness_score"],
            "total_sentences": validation["total_sentences"],
            "supported_sentences": validation["supported_sentences"],
            "unsupported_sentences": validation["unsupported_sentences"],
            "confidence_decision": refusal["decision"],
            "confidence_score": refusal["confidence_score"],
            "refusal_correct": refusal_correct,
            "is_out_of_scope": is_out_of_scope,
            "latency_ms": {
                "routing": round(routing_ms, 1),
                "retrieval": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "validation": round(validation_ms, 1),
            },
        })

    return results


# ──────────────────────────────────────────────
# Metric Aggregation
# ──────────────────────────────────────────────

def aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""

    # Filter to in-scope queries for retrieval metrics
    in_scope = [r for r in results if not r.get("is_out_of_scope", False)]
    out_scope = [r for r in results if r.get("is_out_of_scope", False)]

    # Recall@5
    recall_values = [r["recall_at_5"] for r in in_scope
                     if r["recall_at_5"] is not None]
    avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0

    # nDCG@5
    ndcg_values = [r["ndcg_at_5"] for r in in_scope
                   if r["ndcg_at_5"] is not None]
    avg_ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0

    # Groundedness
    groundedness_values = [r["groundedness_score"] for r in results
                           if "groundedness_score" in r]
    avg_groundedness = (sum(groundedness_values) / len(groundedness_values)
                        if groundedness_values else 0)

    # Hallucination rate
    total_sentences = sum(r.get("total_sentences", 0) for r in results)
    unsupported = sum(r.get("unsupported_sentences", 0) for r in results)
    hallucination_rate = unsupported / total_sentences if total_sentences > 0 else 0

    # Refusal accuracy
    refusal_results = [r for r in results if "refusal_correct" in r]
    refusal_accuracy = (
        sum(1 for r in refusal_results if r["refusal_correct"])
        / len(refusal_results) if refusal_results else 0
    )

    # Decision distribution
    decisions = defaultdict(int)
    for r in results:
        if "confidence_decision" in r:
            decisions[r["confidence_decision"]] += 1

    # Per-category breakdown
    by_category = defaultdict(list)
    for r in in_scope:
        by_category[r["category"]].append(r)

    category_metrics = {}
    for cat, cat_results in by_category.items():
        cat_recalls = [r["recall_at_5"] for r in cat_results
                       if r["recall_at_5"] is not None]
        cat_ground = [r["groundedness_score"] for r in cat_results
                      if "groundedness_score" in r]
        category_metrics[cat] = {
            "count": len(cat_results),
            "avg_recall_at_5": round(
                sum(cat_recalls) / len(cat_recalls), 4
            ) if cat_recalls else None,
            "avg_groundedness": round(
                sum(cat_ground) / len(cat_ground), 4
            ) if cat_ground else None,
        }

    # Latency
    latencies = defaultdict(list)
    for r in results:
        for stage, ms in r.get("latency_ms", {}).items():
            latencies[stage].append(ms)

    latency_summary = {}
    for stage, values in latencies.items():
        sorted_v = sorted(values)
        latency_summary[stage] = {
            "avg_ms": round(sum(values) / len(values), 1),
            "p95_ms": round(sorted_v[int(len(sorted_v) * 0.95)], 1)
            if len(sorted_v) >= 2 else round(max(values), 1),
        }

    return {
        "total_queries": len(results),
        "in_scope_queries": len(in_scope),
        "out_of_scope_queries": len(out_scope),
        "retrieval": {
            "avg_recall_at_5": round(avg_recall, 4),
            "avg_ndcg_at_5": round(avg_ndcg, 4),
            "target_recall": 0.70,
            "target_ndcg": 0.60,
            "recall_meets_target": avg_recall >= 0.70,
            "ndcg_meets_target": avg_ndcg >= 0.60,
        },
        "generation": {
            "avg_groundedness": round(avg_groundedness, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "total_sentences": total_sentences,
            "unsupported_sentences": unsupported,
            "target_groundedness": 0.85,
            "target_hallucination": 0.10,
            "groundedness_meets_target": avg_groundedness >= 0.85,
            "hallucination_meets_target": hallucination_rate <= 0.10,
        },
        "refusal": {
            "accuracy": round(refusal_accuracy, 4),
            "decisions": dict(decisions),
            "out_of_scope_correctly_refused": sum(
                1 for r in out_scope if r.get("confidence_decision") == "INSUFFICIENT_EVIDENCE"
            ),
            "out_of_scope_total": len(out_scope),
        },
        "by_category": category_metrics,
        "latency": latency_summary,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PharmaRAG — Phase 1.6 Evaluation Harness")
    print("=" * 70)

    queries = load_test_queries()
    print(f"\nLoaded {len(queries)} test queries")

    # Category breakdown
    from collections import Counter
    cats = Counter(q["category"] for q in queries)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Initialize components
    print("\nInitializing pipeline components...")
    retriever = HybridRetriever()
    router = QueryRouter()
    generator = AnswerGenerator()
    validator = EvidenceValidator()
    refusal_guard = RefusalGuard()

    # ── Phase A: Retrieval-only eval (fast, no LLM) ──
    print(f"\n{'=' * 70}")
    print("PHASE A: Retrieval Evaluation (no LLM)")
    print(f"{'=' * 70}")

    print("\n  A1: Without query routing...")
    retrieval_no_router = run_retrieval_eval(queries, retriever, router=None)

    print("\n  A2: With query routing...")
    retrieval_with_router = run_retrieval_eval(queries, retriever, router=router)

    # Save retrieval results
    with open(RESULTS_DIR / "retrieval_no_router.json", "w") as f:
        json.dump(retrieval_no_router, f, indent=2)
    with open(RESULTS_DIR / "retrieval_with_router.json", "w") as f:
        json.dump(retrieval_with_router, f, indent=2)

    # Compute retrieval metrics
    in_scope_no_router = [r for r in retrieval_no_router
                          if r["recall_at_5"] is not None]
    in_scope_with_router = [r for r in retrieval_with_router
                            if r["recall_at_5"] is not None]

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0

    print(f"\n  Retrieval Results:")
    print(f"  {'Metric':<20} {'No Router':>12} {'With Router':>12} {'Target':>10}")
    print(f"  {'─' * 56}")
    r_no = avg([r["recall_at_5"] for r in in_scope_no_router])
    r_with = avg([r["recall_at_5"] for r in in_scope_with_router])
    print(f"  {'Recall@5':<20} {r_no:>12.4f} {r_with:>12.4f} {'≥0.70':>10}")
    n_no = avg([r["ndcg_at_5"] for r in in_scope_no_router
                if r["ndcg_at_5"] is not None])
    n_with = avg([r["ndcg_at_5"] for r in in_scope_with_router
                  if r["ndcg_at_5"] is not None])
    print(f"  {'nDCG@5':<20} {n_no:>12.4f} {n_with:>12.4f} {'≥0.60':>10}")

    # ── Phase B: Full pipeline eval (slow, calls LLM) ──
    print(f"\n{'=' * 70}")
    print("PHASE B: Full Pipeline Evaluation (with LLM + Agents)")
    print(f"{'=' * 70}")
    print(f"  Running {len(queries)} queries through the full pipeline...")
    print(f"  This will take approximately {len(queries) * 20 // 60} minutes.\n")

    full_results = run_full_pipeline_eval(
        queries, retriever, router, generator, validator, refusal_guard
    )

    # Save full results
    with open(RESULTS_DIR / "full_pipeline_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Aggregate metrics
    metrics = aggregate_metrics(full_results)
    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print Final Report ──
    print(f"\n{'=' * 70}")
    print("EVALUATION REPORT")
    print(f"{'=' * 70}")

    print(f"\n  Total queries: {metrics['total_queries']}")
    print(f"  In-scope: {metrics['in_scope_queries']}")
    print(f"  Out-of-scope: {metrics['out_of_scope_queries']}")

    print(f"\n  RETRIEVAL METRICS:")
    r = metrics["retrieval"]
    status = "✓" if r["recall_meets_target"] else "✗"
    print(f"    Recall@5:  {r['avg_recall_at_5']:.4f}  (target: ≥{r['target_recall']})  {status}")
    status = "✓" if r["ndcg_meets_target"] else "✗"
    print(f"    nDCG@5:    {r['avg_ndcg_at_5']:.4f}  (target: ≥{r['target_ndcg']})  {status}")

    print(f"\n  GENERATION METRICS:")
    g = metrics["generation"]
    status = "✓" if g["groundedness_meets_target"] else "✗"
    print(f"    Groundedness:      {g['avg_groundedness']:.4f}  (target: ≥{g['target_groundedness']})  {status}")
    status = "✓" if g["hallucination_meets_target"] else "✗"
    print(f"    Hallucination rate: {g['hallucination_rate']:.4f}  (target: ≤{g['target_hallucination']})  {status}")
    print(f"    Total sentences:    {g['total_sentences']}")
    print(f"    Unsupported:        {g['unsupported_sentences']}")

    print(f"\n  REFUSAL METRICS:")
    ref = metrics["refusal"]
    print(f"    Refusal accuracy:   {ref['accuracy']:.4f}")
    print(f"    Out-of-scope refused: {ref['out_of_scope_correctly_refused']}/{ref['out_of_scope_total']}")
    print(f"    Decision distribution: {dict(ref['decisions'])}")

    print(f"\n  PER-CATEGORY BREAKDOWN:")
    for cat, cat_m in metrics["by_category"].items():
        recall_str = f"{cat_m['avg_recall_at_5']:.4f}" if cat_m['avg_recall_at_5'] is not None else "N/A"
        ground_str = f"{cat_m['avg_groundedness']:.4f}" if cat_m['avg_groundedness'] is not None else "N/A"
        print(f"    {cat:<20} Recall@5: {recall_str:>8}  Groundedness: {ground_str:>8}  (n={cat_m['count']})")

    print(f"\n  LATENCY:")
    for stage, lat in metrics.get("latency", {}).items():
        print(f"    {stage:<15} avg: {lat['avg_ms']:>8.1f}ms  p95: {lat['p95_ms']:>8.1f}ms")

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()