"""
Enhanced audit logger for PharmaRAG.
Captures the full pipeline trace: retrieval, routing, generation,
validation, and refusal decisions.

Every request gets a complete audit entry that can be analyzed
for drift detection, quality monitoring, and compliance review.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

from configs.settings import settings


class AuditLogger:
    """Append-only JSONL audit logger with full pipeline capture."""

    def __init__(self):
        self.log_dir = settings.logs_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        print(f"[AuditLogger] Logging to {self.log_file}")

    def create_request_context(self) -> dict:
        """Create a new request context with unique ID."""
        return {
            "request_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timings": {},
            "_start": time.time(),
        }

    def log_request(
        self,
        context: dict,
        query: str,
        routed_sections: list[str],
        retrieval_results: list[dict],
        generation_result: dict,
        validation_report: dict,
        refusal_decision: dict,
    ) -> dict:
        """Write a complete audit log entry capturing every pipeline stage."""
        total_time = time.time() - context.pop("_start", time.time())

        entry = {
            # Request identity
            "request_id": context["request_id"],
            "timestamp": context["timestamp"],
            "query": query,

            # Agent 1: Query Router
            "routed_sections": routed_sections,

            # Retrieval
            "retrieved_chunk_ids": [
                r.get("chunk_id") for r in retrieval_results
            ],
            "retrieved_drugs": list(set(
                r.get("metadata", {}).get("drug_name", "")
                for r in retrieval_results
            )),
            "retrieved_sections": list(set(
                r.get("metadata", {}).get("section_name", "")
                for r in retrieval_results
            )),
            "top_k_fused_scores": [
                round(r.get("fused_score", 0), 6) for r in retrieval_results
            ],
            "avg_fused_score": round(
                sum(r.get("fused_score", 0) for r in retrieval_results)
                / max(len(retrieval_results), 1), 6
            ),

            # Generation
            "model": generation_result.get("model", ""),
            "generation_time_ms": generation_result.get("generation_time_ms", 0),
            "answer_length": len(generation_result.get("answer", "")),

            # Agent 2: Evidence Validator
            "groundedness_score": validation_report.get("groundedness_score", 0),
            "is_grounded": validation_report.get("is_grounded", False),
            "total_sentences": validation_report.get("total_sentences", 0),
            "supported_sentences": validation_report.get("supported_sentences", 0),
            "unsupported_sentences": validation_report.get("unsupported_sentences", 0),

            # Agent 3: Refusal Guard
            "confidence_decision": refusal_decision.get("decision", ""),
            "confidence_score": refusal_decision.get("confidence_score", 0),
            "refusal": refusal_decision.get("decision") == "INSUFFICIENT_EVIDENCE",
            "refusal_reasons": refusal_decision.get("reasons", []),

            # Latency breakdown
            "latency_ms": {
                **{k: round(v, 1) for k, v in context.get("timings", {}).items()},
                "total": round(total_time * 1000, 1),
            },
        }

        if settings.enable_audit_logging:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return entry