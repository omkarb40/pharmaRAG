"""
PharmaRAG — Hybrid Retrieval Demo
Combines BM25 keyword search + ChromaDB vector search using
Reciprocal Rank Fusion (RRF).

How Reciprocal Rank Fusion works:
  Each search method returns a ranked list of results.
  For each result, we compute: score = weight × 1/(k + rank)
  where k=60 is a constant that prevents top-ranked items from
  dominating too heavily.
  
  If a chunk ranks high in BOTH methods, its fused score is high.
  If it ranks high in only one, it still appears but lower.

  This is better than raw score fusion because BM25 scores and
  cosine similarity scores are on completely different scales —
  you can't just add them. RRF only uses rank positions, which
  are comparable across any scoring method.

Why this matters for pharma QA:
  Query: "contraindications for natalizumab"
  - BM25 ranks the actual contraindications chunk #1 (exact keyword match)
  - Vector search ranks it #5 (semantic meaning spreads across related chunks)
  - Hybrid fusion: the chunk gets boosted by both signals → ranks #1

  Query: "serious brain infection risks"
  - BM25 might miss it (no exact term "PML" in the query)
  - Vector search finds PML-related chunks (understands "brain infection" ≈ PML)
  - Hybrid fusion: vector search carries it through

Usage:
  hybrid search "contraindications natalizumab"
  hybrid search "serious brain infection risks"
  hybrid search "pregnancy warnings"
  hybrid search "liver problems hepatotoxicity"
"""

import json
import pickle
import sys
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_NAME = "neuml/pubmedbert-base-embeddings"
CHROMA_DIR = Path("data/index/chromadb")
BM25_DIR = Path("data/index/bm25")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
COLLECTION_NAME = "pharma_rag_chunks"

SEMANTIC_WEIGHT = 0.6   # 60% semantic (meaning)
BM25_WEIGHT = 0.4       # 40% keyword (exact terms)
RRF_K = 60              # Standard RRF constant
TOP_K = 5               # Final number of results


# ──────────────────────────────────────────────
# Load everything
# ──────────────────────────────────────────────

def load_resources():
    """Load all search components."""
    print("Loading search components...")

    # 1. Embedding model (for encoding queries)
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Embedding model loaded")

    # 2. ChromaDB (for vector search)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"  ✓ ChromaDB loaded ({collection.count()} chunks)")

    # 3. BM25 (for keyword search)
    with open(BM25_DIR / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DIR / "chunk_ids.json", "r") as f:
        bm25_chunk_ids = json.load(f)
    print(f"  ✓ BM25 index loaded ({len(bm25_chunk_ids)} chunks)")

    # 4. Raw chunks (for BM25 → full document lookup)
    chunks_by_id = {}
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            c = json.loads(line.strip())
            chunks_by_id[c["chunk_id"]] = c
    print(f"  ✓ Chunk lookup table loaded")

    return model, collection, bm25, bm25_chunk_ids, chunks_by_id


# ──────────────────────────────────────────────
# Individual search methods
# ──────────────────────────────────────────────

def semantic_search(
    query: str,
    model: SentenceTransformer,
    collection,
    top_k: int = 20,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Vector similarity search via ChromaDB.
    
    Steps:
    1. Encode query → 768-dim vector
    2. ChromaDB computes cosine distance against all stored vectors
    3. Returns top_k nearest neighbors
    
    Optional section_filter narrows search to a specific label section.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Build ChromaDB where filter (metadata-based filtering)
    where_filter = None
    if section_filter:
        where_filter = {"section_name": section_filter}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            # ChromaDB returns cosine distance; similarity = 1 - distance
            "semantic_score": round(1 - results["distances"][0][i], 4),
        })
    return hits


def keyword_search(
    query: str,
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    chunks_by_id: dict,
    top_k: int = 20,
) -> list[dict]:
    """
    BM25 keyword search.
    
    Steps:
    1. Tokenize query into lowercase words
    2. BM25 scores every document based on term frequency
    3. Return top_k highest scoring documents
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices sorted by score
    top_indices = scores.argsort()[-top_k:][::-1]

    hits = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk_id = bm25_chunk_ids[idx]
        chunk = chunks_by_id.get(chunk_id, {})
        if chunk:
            hits.append({
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "metadata": {
                    "drug_name": chunk.get("drug_name", ""),
                    "generic_name": chunk.get("generic_name", ""),
                    "section_name": chunk.get("section_name", ""),
                    "set_id": chunk.get("set_id", ""),
                    "loinc_code": chunk.get("loinc_code", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 0),
                },
                "bm25_score": round(float(scores[idx]), 4),
            })
    return hits


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────

def hybrid_search(
    query: str,
    model: SentenceTransformer,
    collection,
    bm25: BM25Okapi,
    bm25_chunk_ids: list[str],
    chunks_by_id: dict,
    top_k: int = TOP_K,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Combines semantic + keyword search using Reciprocal Rank Fusion.
    
    RRF formula per chunk:
      fused_score = sem_weight × 1/(k + sem_rank) + bm25_weight × 1/(k + bm25_rank)
    
    Example with k=60:
      Chunk ranked #1 in semantic:  0.6 × 1/61 = 0.00984
      Chunk ranked #1 in BM25:     0.4 × 1/61 = 0.00656
      If it's #1 in both:          total = 0.01639 (strong signal from both)
      If it's #1 in semantic only:  total = 0.00984 (still appears, but lower)
    """
    # Get results from both methods (over-fetch to ensure good fusion)
    fetch_k = top_k * 4

    sem_hits = semantic_search(query, model, collection, fetch_k, section_filter)
    bm25_hits = keyword_search(query, bm25, bm25_chunk_ids, chunks_by_id, fetch_k)

    # Reciprocal Rank Fusion
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score semantic results by rank
    for rank, hit in enumerate(sem_hits):
        cid = hit["chunk_id"]
        rrf_score = SEMANTIC_WEIGHT * (1.0 / (RRF_K + rank + 1))
        fused_scores[cid] = fused_scores.get(cid, 0) + rrf_score
        chunk_data[cid] = {
            "chunk_id": cid,
            "text": hit["text"],
            "metadata": hit["metadata"],
            "semantic_score": hit.get("semantic_score", 0),
            "bm25_score": 0,
        }

    # Score BM25 results by rank
    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        rrf_score = BM25_WEIGHT * (1.0 / (RRF_K + rank + 1))
        fused_scores[cid] = fused_scores.get(cid, 0) + rrf_score
        if cid in chunk_data:
            # Already seen in semantic results — add BM25 score
            chunk_data[cid]["bm25_score"] = hit.get("bm25_score", 0)
        else:
            chunk_data[cid] = {
                "chunk_id": cid,
                "text": hit["text"],
                "metadata": hit["metadata"],
                "semantic_score": 0,
                "bm25_score": hit.get("bm25_score", 0),
            }

    # Sort by fused score, take top_k
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid]
        entry["fused_score"] = round(fused_scores[cid], 6)
        results.append(entry)

    return results


# ──────────────────────────────────────────────
# Comparison demo
# ──────────────────────────────────────────────

def compare_methods(query, model, collection, bm25, bm25_chunk_ids, chunks_by_id):
    """Run all three methods and show side-by-side comparison."""
    print(f"\n{'=' * 70}")
    print(f"🔍 Query: \"{query}\"")
    print(f"{'=' * 70}")

    # BM25 only
    bm25_results = keyword_search(query, bm25, bm25_chunk_ids, chunks_by_id, TOP_K)
    print(f"\n  📝 BM25 (keyword) results:")
    for i, r in enumerate(bm25_results[:3], 1):
        m = r["metadata"]
        print(f"    [{i}] {m['drug_name']:12s} | {m['section_name']:30s} | BM25: {r['bm25_score']:.2f}")
        print(f"        {r['text'][:100]}...")

    # Semantic only
    sem_results = semantic_search(query, model, collection, TOP_K)
    print(f"\n  🧠 Semantic (vector) results:")
    for i, r in enumerate(sem_results[:3], 1):
        m = r["metadata"]
        print(f"    [{i}] {m['drug_name']:12s} | {m['section_name']:30s} | Sim: {r['semantic_score']:.4f}")
        print(f"        {r['text'][:100]}...")

    # Hybrid fusion
    hybrid_results = hybrid_search(
        query, model, collection, bm25, bm25_chunk_ids, chunks_by_id
    )
    print(f"\n  ⚡ Hybrid (BM25 + Semantic) results:")
    for i, r in enumerate(hybrid_results[:5], 1):
        m = r["metadata"]
        sem = r.get("semantic_score", 0)
        bm = r.get("bm25_score", 0)
        source = ""
        if sem > 0 and bm > 0:
            source = "← both"
        elif sem > 0:
            source = "← semantic"
        elif bm > 0:
            source = "← keyword"
        print(f"    [{i}] {m['drug_name']:12s} | {m['section_name']:30s} | Fused: {r['fused_score']:.6f}  {source}")
        print(f"        Semantic: {sem:.4f}  |  BM25: {bm:.2f}")
        print(f"        {r['text'][:120]}...")
        print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PharmaRAG — Hybrid Retrieval Demo")
    print("=" * 70)

    # Load all components
    model, collection, bm25, bm25_chunk_ids, chunks_by_id = load_resources()

    # If user provided a query, run just that
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        compare_methods(query, model, collection, bm25, bm25_chunk_ids, chunks_by_id)
        return

    # Otherwise, run the full comparison demo
    test_queries = [
        # This query has exact keywords — BM25 should shine
        "contraindications for natalizumab",

        # This query uses natural language — vector search should shine
        "serious brain infection risks",

        # This query mixes exact terms + meaning
        "pregnancy warnings teriflunomide",

        # This query is purely conceptual — no drug name
        "can I take this medicine if I have liver problems",

        # This tests section filtering potential
        "what is the recommended dose",
    ]

    for query in test_queries:
        compare_methods(query, model, collection, bm25, bm25_chunk_ids, chunks_by_id)

    # Bonus: demonstrate section-filtered search
    print(f"\n{'=' * 70}")
    print(f"🔍 BONUS: Section-filtered search")
    print(f"   Query: \"side effects\" with filter: adverse_reactions")
    print(f"{'=' * 70}")

    filtered = hybrid_search(
        "side effects", model, collection, bm25, bm25_chunk_ids, chunks_by_id,
        section_filter="adverse_reactions",
    )
    for i, r in enumerate(filtered[:3], 1):
        m = r["metadata"]
        print(f"  [{i}] {m['drug_name']:12s} | {m['section_name']:30s} | Fused: {r['fused_score']:.6f}")
        print(f"      {r['text'][:150]}...")
        print()

    print(f"\n{'=' * 70}")
    print("✅ Hybrid Search Demo Complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()