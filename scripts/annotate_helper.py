"""
Batch annotation helper.
For each in-scope query missing a reference_snippet, shows the top
retrieved chunks so you can copy the gold span + chunk_id.

Refusal queries (expected_behavior == 'refuse') are skipped — they
correctly have no gold evidence.

Usage:
  python scripts/annotate_helper.py
  python scripts/annotate_helper.py --only CI-002       # one query
  python scripts/annotate_helper.py --unfilled          # only null ones
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from src.retrieval.hybrid_search import HybridRetriever


TEST_FILE = Path("evaluation/test_queries_v2.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, help="Annotate a single query by id")
    parser.add_argument("--unfilled", action="store_true",
                        help="Only show queries with null reference_chunk_id")
    parser.add_argument("--topk", type=int, default=6)
    args = parser.parse_args()

    with open(TEST_FILE, "r") as f:
        queries = json.load(f)

    retriever = HybridRetriever()

    for q in queries:
        if args.only and q["id"] != args.only:
            continue
        # Skip refusal queries — they should stay null
        if q.get("expected_behavior") == "refuse":
            continue
        if args.unfilled and q.get("reference_chunk_id"):
            continue

        print("\n" + "=" * 75)
        print(f"  {q['id']}  [{q['category']}]  expected_drug={q.get('expected_drug')}")
        print(f"  QUERY: {q['query']}")
        print(f"  expected_section: {q.get('expected_section')}")
        already = q.get("reference_chunk_id")
        print(f"  STATUS: {'FILLED — '+already if already else 'NEEDS ANNOTATION'}")
        print("=" * 75)

        # Retrieve using the section filter the query targets, if any
        section = q.get("expected_section")
        results = retriever.search(query=q["query"], top_k=args.topk,
                                    section_filter=section)
        # Fallback to unfiltered if filter gave too few
        if len(results) < args.topk:
            results = retriever.search(query=q["query"], top_k=args.topk)

        for i, c in enumerate(results, 1):
            meta = c.get("metadata", {})
            print(f"\n  [{i}] chunk_id: {c['chunk_id']}  "
                  f"| {meta.get('drug_name')} | {meta.get('section_name')} "
                  f"| fused={c.get('fused_score', 0):.5f}")
            print(f"      {c.get('text', '')[:500]}")

        print("\n  → Copy the chunk_id and the exact answering sentence(s) into the JSON.\n")


if __name__ == "__main__":
    main()