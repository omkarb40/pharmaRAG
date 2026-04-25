"""
/ask endpoint — the full PharmaRAG pipeline with agentic safety.

Pipeline:
  1. Query Router (Agent 1) → classify question → section filter
  2. Hybrid Retrieval → BM25 + semantic via RRF
  3. LLM Generation → cited answer from evidence
  4. Evidence Validator (Agent 2) → check groundedness
  5. Refusal Guard (Agent 3) → decide: answer / caution / refuse
  6. Audit Logging → record everything
  7. Structured Response → answer + citations + evidence + confidence
"""
import time
from pydantic import BaseModel

from fastapi import APIRouter

from src.retrieval.query_router import QueryRouter
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.generator import AnswerGenerator
from src.generation.evidence_validator import EvidenceValidator
from src.generation.refusal_guard import RefusalGuard
from src.monitoring.logger import AuditLogger
from configs.settings import settings


router = APIRouter()

# Initialize all components once at startup
print("\n[PharmaRAG] Initializing pipeline components...")
query_router = QueryRouter()
retriever = HybridRetriever()
generator = AnswerGenerator()
validator = EvidenceValidator()
refusal_guard = RefusalGuard()
audit_logger = AuditLogger()
print("[PharmaRAG] Pipeline ready.\n")


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    top_k: int = settings.top_k_final

    class Config:
        json_schema_extra = {
            "examples": [
                {"query": "What are the contraindications for natalizumab?"},
                {"query": "Can I take fingolimod if I have liver problems?"},
            ]
        }


class Citation(BaseModel):
    citation_id: int
    chunk_id: str
    drug_name: str
    generic_name: str
    section_name: str
    retrieval_score: float
    text_snippet: str


class EvidenceEntry(BaseModel):
    rank: int
    chunk_id: str
    drug_name: str
    section: str
    fused_score: float
    semantic_score: float
    bm25_score: float
    snippet: str


class SentenceValidation(BaseModel):
    sentence: str
    max_similarity: float
    best_supporting_chunk_id: str
    best_supporting_drug: str
    is_supported: bool


class ConfidenceReport(BaseModel):
    decision: str
    confidence_score: float
    signals: dict
    reasons: list[str]


class GroundednessReport(BaseModel):
    is_grounded: bool
    groundedness_score: float
    total_sentences: int
    supported_sentences: int
    unsupported_sentences: int
    sentence_details: list[SentenceValidation]


class AskResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    confidence: ConfidenceReport
    groundedness: GroundednessReport
    citations: list[Citation]
    evidence_table: list[EvidenceEntry]
    routed_sections: list[str]
    latency_ms: dict
    model: str


# ──────────────────────────────────────────────
# Stats endpoint
# ──────────────────────────────────────────────

@router.get("/stats")
def stats():
    """Database statistics."""
    chunks = retriever.chunks_by_id
    drugs = sorted(set(c["drug_name"] for c in chunks.values()))
    section_counts = {}
    for c in chunks.values():
        sec = c.get("section_name", "unknown")
        section_counts[sec] = section_counts.get(sec, 0) + 1

    return {
        "total_chunks": retriever.collection.count(),
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
    Ask a drug safety question.
    Returns a cited answer with confidence scoring and groundedness report.
    """
    ctx = audit_logger.create_request_context()

    # ── Agent 1: Query Router ──
    t0 = time.time()
    routed_sections = query_router.route(req.query)
    ctx["timings"]["routing_ms"] = (time.time() - t0) * 1000
    print(f"  [Router] Query: '{req.query[:50]}...' → {routed_sections}")

    # ── Hybrid Retrieval ──
    t0 = time.time()
    primary_section = routed_sections[0] if routed_sections else None
    retrieval_results = retriever.search(
        query=req.query,
        top_k=req.top_k,
        section_filter=primary_section,
    )
    if len(retrieval_results) < req.top_k:
        retrieval_results = retriever.search(
            query=req.query,
            top_k=req.top_k,
        )
    ctx["timings"]["retrieval_ms"] = (time.time() - t0) * 1000

    # ── LLM Generation ──
    t0 = time.time()
    gen_result = generator.generate(req.query, retrieval_results)
    ctx["timings"]["generation_ms"] = (time.time() - t0) * 1000

    # ── Agent 2: Evidence Validator ──
    t0 = time.time()
    validation = validator.validate(gen_result["answer"], retrieval_results)
    ctx["timings"]["validation_ms"] = (time.time() - t0) * 1000

    # ── Agent 3: Refusal Guard ──
    t0 = time.time()
    refusal = refusal_guard.evaluate(retrieval_results, validation)
    ctx["timings"]["refusal_ms"] = (time.time() - t0) * 1000

    print(f"  [Validator] Groundedness: {validation['groundedness_score']:.1%}")
    print(f"  [RefusalGuard] Decision: {refusal['decision']} "
          f"(confidence: {refusal['confidence_score']:.2f})")

    # ── Override answer if refused ──
    final_answer = gen_result["answer"]
    if refusal["decision"] == "INSUFFICIENT_EVIDENCE":
        final_answer = (
            "⚠️ Insufficient evidence to provide a reliable answer. "
            "The retrieved sources do not adequately support a response "
            "to this query. Please consult authoritative drug labeling "
            "directly or speak with a healthcare professional.\n\n"
            f"Reasons: {'; '.join(refusal['reasons'])}"
        )

    # ── Audit Log (now captures all agent outputs) ──
    log_entry = audit_logger.log_request(
        context=ctx,
        query=req.query,
        routed_sections=routed_sections,
        retrieval_results=retrieval_results,
        generation_result=gen_result,
        validation_report=validation,
        refusal_decision=refusal,
    )

    # ── Build Response ──
    citations = [Citation(**c) for c in gen_result["citations"]]

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

    sentence_validations = []
    for sd in validation.get("sentence_details", []):
        sentence_validations.append(SentenceValidation(
            sentence=sd["sentence"],
            max_similarity=sd["max_similarity"],
            best_supporting_chunk_id=sd["best_supporting_chunk_id"],
            best_supporting_drug=sd["best_supporting_drug"],
            is_supported=sd["is_supported"],
        ))

    return AskResponse(
        request_id=log_entry["request_id"],
        query=req.query,
        answer=final_answer,
        confidence=ConfidenceReport(
            decision=refusal["decision"],
            confidence_score=refusal["confidence_score"],
            signals=refusal["signals"],
            reasons=refusal["reasons"],
        ),
        groundedness=GroundednessReport(
            is_grounded=validation["is_grounded"],
            groundedness_score=validation["groundedness_score"],
            total_sentences=validation["total_sentences"],
            supported_sentences=validation["supported_sentences"],
            unsupported_sentences=validation["unsupported_sentences"],
            sentence_details=sentence_validations,
        ),
        citations=citations,
        evidence_table=evidence_table,
        routed_sections=routed_sections,
        latency_ms=log_entry["latency_ms"],
        model=gen_result["model"],
    )