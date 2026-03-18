"""
PharmaRAG — FastAPI Search Demo
Exposes BM25 keyword search as an API endpoint.

"""

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


# ──────────────────────────────────────────────
# Load data at startup
# ──────────────────────────────────────────────

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


print(f"[Startup] Loading chunks from {CHUNKS_FILE}...")
CHUNKS = load_chunks()
TOKENIZED_CORPUS = [c["text"].lower().split() for c in CHUNKS]
BM25_INDEX = BM25Okapi(TOKENIZED_CORPUS)
print(f"[Startup] BM25 index ready: {len(CHUNKS)} chunks indexed")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="PharmaRAG Demo",
    description="Phase 1.0 — BM25 Keyword Search over FDA Drug Labels",
    version="0.1.0",
)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    section_filter: str | None = None  # Optional: filter by section name

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "contraindications natalizumab",
                    "top_k": 5,
                },
                {
                    "query": "pregnancy warnings",
                    "top_k": 3,
                    "section_filter": "use_in_specific_populations",
                },
            ]
        }


class SearchResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    drug_name: str
    generic_name: str
    section_name: str
    chunk_index: int
    total_chunks: int
    text_preview: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]


class StatsResponse(BaseModel):
    total_chunks: int
    total_drugs: int
    drugs: list[str]
    sections: dict[str, int]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "pharma-rag-demo",
        "chunks_loaded": len(CHUNKS),
    }


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Get database statistics — how many drugs, chunks, sections."""
    drugs = sorted(set(c["drug_name"] for c in CHUNKS))
    section_counts = {}
    for c in CHUNKS:
        sec = c["section_name"]
        section_counts[sec] = section_counts.get(sec, 0) + 1

    return StatsResponse(
        total_chunks=len(CHUNKS),
        total_drugs=len(drugs),
        drugs=drugs,
        sections=section_counts,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    BM25 keyword search across all drug label chunks.
    
    Example queries:
    - "contraindications natalizumab"
    - "black box warning PML"
    - "adverse reactions fingolimod"
    - "pregnancy"
    - "drug interactions"
    """
    tokenized_query = req.query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)

    # Build results
    scored_chunks = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        chunk = CHUNKS[idx]

        # Optional section filter
        if req.section_filter and chunk["section_name"] != req.section_filter:
            continue

        scored_chunks.append((score, idx, chunk))

    # Sort by score descending, take top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[: req.top_k]

    results = []
    for rank, (score, idx, chunk) in enumerate(top, 1):
        results.append(SearchResult(
            rank=rank,
            score=round(score, 4),
            chunk_id=chunk["chunk_id"],
            drug_name=chunk["drug_name"],
            generic_name=chunk["generic_name"],
            section_name=chunk["section_name"],
            chunk_index=chunk["chunk_index"],
            total_chunks=chunk["total_chunks"],
            text_preview=chunk["text"][:500],
        ))

    return SearchResponse(
        query=req.query,
        total_results=len(results),
        results=results,
    )