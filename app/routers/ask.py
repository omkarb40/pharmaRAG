"""
/ask endpoint — the main QA pipeline.
Orchestrates: Retrieval → Generation → Logging → Response

This is where everything comes together:
  1. User sends a question
  2. Hybrid retriever finds the most relevant chunks
  3. LLM generates a cited answer from those chunks
  4. Audit logger records everything
  5. Response includes answer + citations + evidence table + timing
"""

import time
from pydantic import BaseModel

from fastapi import APIRouter

from src.retrieval.hybrid_search import HybridRetriever
from src.generation.generator import AnswerGenerator
from src.monitoring.logger import AuditLogger
from configs.settings import settings


router = APIRouter()

# Initialize components once at startup (singletons)
print("\n[PharmaRAG] Initializing pipeline components...")
retriever = HybridRetriever()
generator = AnswerGenerator()
audit_logger = AuditLogger()
print("[PharmaRAG] Pipeline ready.\n")


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    """What the user sends."""
    query: str
    top_k: int = settings.top_k_final
    section_filter: str | None = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "What are the contraindications for natalizumab?",
                    "top_k": 5,
                },
                {
                    "query": "Can I take fingolimod if I have liver problems?",
                    "top_k": 5,
                    "section_filter": "contraindications",
                },
            ]
        }


class Citation(BaseModel):
    """One citation pointing back to a source chunk."""
    citation_id: int
    chunk_id: str
    drug_name: str
    generic_name: str
    section_name: str
    retrieval_score: float
    text_snippet: str


class EvidenceEntry(BaseModel):
    """One row in the evidence table."""
    rank: int
    chunk_id: str
    drug_name: str
    section: str
    fused_score: float
    semantic_score: float
    bm25_score: float
    snippet: str


class AskResponse(BaseModel):
    """Everything the API returns."""
    request_id: str
    query: str
    answer: str
    citations: list[Citation]
    evidence_table: list[EvidenceEntry]
    latency_ms: dict
    model: str


# ──────────────────────────────────────────────
# Stats endpoint (carried over from demo)
# ──────────────────────────────────────────────

@router.get("/stats")
def stats():
    """Database statistics."""
    collection = retriever.collection
    chunks = retriever.chunks_by_id

    drugs = sorted(set(c["drug_name"] for c in chunks.values()))
    section_counts = {}
    for c in chunks.values():
        sec = c.get("section_name", "unknown")
        section_counts[sec] = section_counts.get(sec, 0) + 1

    return {
        "total_chunks": collection.count(),
        "total_drugs": len(drugs),
        "drugs": drugs,
        "sections": section_counts,
    }


# ──────────────────────────────────────────────
# The main /ask endpoint
# ──────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Ask a drug safety question. Returns a cited answer with evidence table.

    The pipeline:
      1. Hybrid retrieval (BM25 + semantic via RRF)
      2. LLM generation with citation prompt
      3. Audit logging
      4. Structured response
    """
    ctx = audit_logger.create_request_context()

    # ── Step 1: Retrieve ──
    t0 = time.time()
    retrieval_results = retriever.search(
        query=req.query,
        top_k=req.top_k,
        section_filter=req.section_filter,
    )
    ctx["timings"]["retrieval_ms"] = (time.time() - t0) * 1000

    # ── Step 2: Generate answer ──
    t0 = time.time()
    gen_result = generator.generate(req.query, retrieval_results)
    ctx["timings"]["generation_ms"] = (time.time() - t0) * 1000

    # ── Step 3: Audit log ──
    log_entry = audit_logger.log_request(
        context=ctx,
        query=req.query,
        retrieval_results=retrieval_results,
        generation_result=gen_result,
    )

    # ── Step 4: Build response ──

    # Citations (from generator)
    citations = [Citation(**c) for c in gen_result["citations"]]

    # Evidence table (from retriever)
    evidence_table = []
    for i, chunk in enumerate(retrieval_results, 1):
        meta = chunk.get("metadata", {})
        evidence_table.append(EvidenceEntry(
            rank=i,
            chunk_id=chunk.get("chunk_id", ""),
            drug_name=meta.get("drug_name", ""),
            section=meta.get("section_name", ""),
            fused_score=round(chunk.get("fused_score", 0), 6),
            semantic_score=round(chunk.get("semantic_score", 0), 4),
            bm25_score=round(chunk.get("bm25_score", 0), 4),
            snippet=chunk.get("text", "")[:200],
        ))

    return AskResponse(
        request_id=log_entry["request_id"],
        query=req.query,
        answer=gen_result["answer"],
        citations=citations,
        evidence_table=evidence_table,
        latency_ms=log_entry["latency_ms"],
        model=gen_result["model"],
    )