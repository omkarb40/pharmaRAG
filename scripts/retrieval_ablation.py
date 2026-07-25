"""
Retrieval component ladder (Phase A only — no generation, no LLM except router).

  R1 BM25-only      lexical signal alone
  R2 Dense-only     semantic signal alone
  R3 Hybrid RRF     fusion            (must reproduce the locked baseline)
  R4 Hybrid+Rerank  cross-encoder     (must reproduce locked Option C)

Router is computed ONCE per query and shared across all rungs, so retrieval
mode is the only variable.
"""
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import CrossEncoderReranker
from src.evaluation.eval_runner import (
    compute_recall_at_k, compute_ndcg_at_k, compute_gold_chunk_recall,
)

QUERIES = json.load(open("evaluation/test_queries_v2.json"))
OUT = Path("evaluation/results/retrieval_ablation.json")


def merged(retriever, qtext, sect, mode, k=5):
    """Filtered + unfiltered merge, mirroring eval_runner._retrieve_merged."""
    if sect:
        filt = retriever.search(query=qtext, top_k=k, section_filter=sect, mode=mode)
        unfilt = retriever.search(query=qtext, top_k=k, mode=mode)
        seen = {r["chunk_id"] for r in filt}
        out = list(filt)
        for r in unfilt:
            if r["chunk_id"] not in seen and len(out) < k:
                out.append(r); seen.add(r["chunk_id"])
        return out[:k]
    return retriever.search(query=qtext, top_k=k, mode=mode)


def main():
    retriever = HybridRetriever()
    router = QueryRouter()
    reranker = CrossEncoderReranker()

    print("\nRouting all queries once (shared across rungs)...")
    routes = {}
    for i, q in enumerate(QUERIES):
        r = router.route(q["query"])
        routes[q["id"]] = r[0] if r else None
        if (i + 1) % 20 == 0:
            print(f"  routed {i+1}/{len(QUERIES)}")

    rungs = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    per_query = {r: {} for r in rungs}
    agg = {}

    for rung in rungs:
        mode = "hybrid" if rung == "hybrid_rerank" else rung
        use_rr = (rung == "hybrid_rerank")
        print(f"\n[{rung}] retrieving...")
        recs, golds, ndcgs = [], [], []
        for q in QUERIES:
            sect = routes[q["id"]]
            if use_rr:
                pool = merged(retriever, q["query"], sect, mode, k=20)
                chunks = reranker.rerank(q["query"], pool, top_k=5)
            else:
                chunks = merged(retriever, q["query"], sect, mode, k=5)

            r = compute_recall_at_k(chunks, query=q)
            g = compute_gold_chunk_recall(chunks, query=q)
            n = compute_ndcg_at_k(chunks, query=q)
            per_query[rung][q["id"]] = {"recall": r, "gold": g, "ndcg": n}
            if r is not None: recs.append(r)
            if g is not None: golds.append(g)
            if n is not None: ndcgs.append(n)

        avg = lambda v: round(sum(v) / len(v), 4) if v else 0.0
        agg[rung] = {
            "n_answer_queries": len(recs),
            "recall_at_5": avg(recs),
            "gold_chunk_recall_at_5": avg(golds),
            "ndcg_at_5": avg(ndcgs),
        }

    print("\n" + "=" * 66)
    print("RETRIEVAL COMPONENT LADDER (answer queries, router held constant)")
    print("=" * 66)
    print(f"{'Configuration':<22}{'R@5':>10}{'Gold@5':>10}{'nDCG@5':>10}")
    print("-" * 66)
    labels = {"bm25": "R1 BM25-only", "dense": "R2 Dense-only",
              "hybrid": "R3 Hybrid RRF", "hybrid_rerank": "R4 Hybrid+Rerank"}
    for r in rungs:
        a = agg[r]
        print(f"{labels[r]:<22}{a['recall_at_5']:>10.4f}{a['gold_chunk_recall_at_5']:>10.4f}{a['ndcg_at_5']:>10.4f}")

    def mcnemar(a, b, key, label):
        ids = [q["id"] for q in QUERIES
               if per_query[a][q["id"]][key] is not None
               and per_query[b][q["id"]][key] is not None]
        b01 = sum(1 for i in ids if per_query[a][i][key] == 0 and per_query[b][i][key] == 1)
        b10 = sum(1 for i in ids if per_query[a][i][key] == 1 and per_query[b][i][key] == 0)
        n = b01 + b10
        if n == 0:
            print(f"  {label}: no discordant pairs"); return
        p = min(sum(comb(n, k) for k in range(min(b01, b10) + 1)) / 2 ** n * 2, 1.0)
        print(f"  {label}: +{b01} / -{b10}, exact two-sided p = {p:.4f}")

    print("\nADJACENT-RUNG SIGNIFICANCE (Recall@5):")
    mcnemar("bm25", "dense", "recall", "BM25 -> Dense")
    mcnemar("bm25", "hybrid", "recall", "BM25 -> Hybrid")
    mcnemar("dense", "hybrid", "recall", "Dense -> Hybrid")
    mcnemar("hybrid", "hybrid_rerank", "recall", "Hybrid -> Hybrid+Rerank")
    print("\nADJACENT-RUNG SIGNIFICANCE (gold-chunk Recall@5):")
    mcnemar("dense", "hybrid", "gold", "Dense -> Hybrid")
    mcnemar("hybrid", "hybrid_rerank", "gold", "Hybrid -> Hybrid+Rerank")

    json.dump({"aggregate": agg, "per_query": per_query}, open(OUT, "w"), indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
