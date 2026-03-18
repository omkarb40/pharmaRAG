"""
PharmaRAG — Keyword Search Demo
BM25 search across chunked drug label data.

"""

import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def load_chunks() -> list[dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Build BM25 index from chunk texts."""
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def search(query: str, chunks: list[dict], bm25: BM25Okapi, top_k: int = 5) -> list[dict]:
    """Search and return top-k results with scores."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices (sorted by score descending)
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        if scores[idx] <= 0:
            continue
        chunk = chunks[idx]
        results.append({
            "rank": rank,
            "score": round(float(scores[idx]), 4),
            "chunk_id": chunk["chunk_id"],
            "drug_name": chunk["drug_name"],
            "generic_name": chunk["generic_name"],
            "section_name": chunk["section_name"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "text_preview": chunk["text"][:300],
        })

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_search.py \"your search query\"")
        print("\nExamples:")
        print('  python demo_search.py "contraindications natalizumab"')
        print('  python demo_search.py "black box warning PML"')
        print('  python demo_search.py "pregnancy"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found.")
        print("Run ingestion first: python demo_ingest.py")
        sys.exit(1)

    # Load and index
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks()
    print(f"Building BM25 index over {len(chunks)} chunks...")
    bm25 = build_bm25_index(chunks)

    # Search
    print(f"\n{'=' * 60}")
    print(f"🔍 Query: \"{query}\"")
    print(f"{'=' * 60}\n")

    results = search(query, chunks, bm25, top_k=5)

    if not results:
        print("No matching chunks found.")
        return

    for r in results:
        print(f"  [{r['rank']}] Score: {r['score']}")
        print(f"      Drug:    {r['drug_name']} ({r['generic_name']})")
        print(f"      Section: {r['section_name']}")
        print(f"      Chunk:   {r['chunk_id']} ({r['chunk_index']+1}/{r['total_chunks']})")
        print(f"      Text:    {r['text_preview'][:200]}...")
        print()

    # Also print as JSON for professor to see structured output
    print(f"\n{'=' * 60}")
    print("📊 Structured JSON output:")
    print(f"{'=' * 60}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()