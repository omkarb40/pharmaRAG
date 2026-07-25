"""R5 = Dense-only + cross-encoder rerank, plus a semantic-weight sweep.
Router held constant (reuses routes computed here, same as the ladder)."""
import json, sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.reranker import CrossEncoderReranker
from src.evaluation.eval_runner import (
    compute_recall_at_k, compute_ndcg_at_k, compute_gold_chunk_recall)

QUERIES = json.load(open("evaluation/test_queries_v2.json"))

def merged(r, qt, sect, mode, k):
    if sect:
        f = r.search(query=qt, top_k=k, section_filter=sect, mode=mode)
        u = r.search(query=qt, top_k=k, mode=mode)
        seen = {x["chunk_id"] for x in f}; out = list(f)
        for x in u:
            if x["chunk_id"] not in seen and len(out) < k:
                out.append(x); seen.add(x["chunk_id"])
        return out[:k]
    return r.search(query=qt, top_k=k, mode=mode)

def evaluate(r, routes, mode, rr=None, label=""):
    pq, recs, golds, ndcgs = {}, [], [], []
    for q in QUERIES:
        sect = routes[q["id"]]
        if rr:
            chunks = rr.rerank(q["query"], merged(r, q["query"], sect, mode, 20), top_k=5)
        else:
            chunks = merged(r, q["query"], sect, mode, 5)
        a = compute_recall_at_k(chunks, query=q)
        g = compute_gold_chunk_recall(chunks, query=q)
        n = compute_ndcg_at_k(chunks, query=q)
        pq[q["id"]] = {"recall": a, "gold": g}
        if a is not None: recs.append(a)
        if g is not None: golds.append(g)
        if n is not None: ndcgs.append(n)
    avg = lambda v: round(sum(v)/len(v), 4) if v else 0.0
    print(f"{label:<26}{avg(recs):>10.4f}{avg(golds):>10.4f}{avg(ndcgs):>10.4f}")
    return pq

def mcnemar(a, b, key, label):
    ids = [q["id"] for q in QUERIES
           if a[q["id"]][key] is not None and b[q["id"]][key] is not None]
    b01 = sum(1 for i in ids if a[i][key] == 0 and b[i][key] == 1)
    b10 = sum(1 for i in ids if a[i][key] == 1 and b[i][key] == 0)
    n = b01 + b10
    if n == 0: print(f"  {label}: no discordant pairs"); return
    p = min(sum(comb(n, k) for k in range(min(b01, b10)+1)) / 2**n * 2, 1.0)
    print(f"  {label}: +{b01} / -{b10}, p = {p:.4f}")

def main():
    r = HybridRetriever(); router = QueryRouter(); rr = CrossEncoderReranker()
    print("\nRouting once...")
    routes = {q["id"]: (lambda x: x[0] if x else None)(router.route(q["query"]))
              for q in QUERIES}

    print("\n" + "="*56)
    print(f"{'Configuration':<26}{'R@5':>10}{'Gold@5':>10}{'nDCG@5':>10}")
    print("-"*56)
    hyb_rr = evaluate(r, routes, "hybrid", rr, "R4 Hybrid+Rerank")
    den_rr = evaluate(r, routes, "dense",  rr, "R5 Dense+Rerank")
    print("\nSIGNIFICANCE:")
    mcnemar(hyb_rr, den_rr, "recall", "Hybrid+RR -> Dense+RR (R@5)")
    mcnemar(hyb_rr, den_rr, "gold",   "Hybrid+RR -> Dense+RR (Gold@5)")

    print("\n" + "="*56)
    print("SEMANTIC-WEIGHT SWEEP (no reranker, isolates fusion balance)")
    print(f"{'Configuration':<26}{'R@5':>10}{'Gold@5':>10}{'nDCG@5':>10}")
    print("-"*56)
    orig_s, orig_b = settings.semantic_weight, settings.bm25_weight
    for sw in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        settings.semantic_weight, settings.bm25_weight = sw, round(1 - sw, 2)
        evaluate(r, routes, "hybrid", None, f"  sem={sw:.1f} / bm25={1-sw:.1f}")
    settings.semantic_weight, settings.bm25_weight = orig_s, orig_b

if __name__ == "__main__":
    main()
