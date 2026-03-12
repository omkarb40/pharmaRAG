"""
Structured audit logging for every request.
Logs to JSONL files for downstream analysis.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

from configs.settings import settings


class AuditLogger:
    """Append-only JSONL audit logger for pharma RAG requests."""

    def __init__(self):
        self.log_dir = settings.logs_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def create_request_context(self) -> dict:
        """Create a new request context with unique ID and timing."""
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
    ):
        """Write a complete audit log entry for one request."""
        total_time = time.time() - context.pop("_start", time.time())

        entry = {
            "request_id": context["request_id"],
            "timestamp": context["timestamp"],
            "query": query,
            "routed_sections": routed_sections,
            "retrieved_chunk_ids": [r.get("chunk_id") for r in retrieval_results],
            "top_k_scores": [
                round(r.get("fused_score", 0), 6) for r in retrieval_results
            ],
            "latency_ms": {
                **{k: round(v, 1) for k, v in context.get("timings", {}).items()},
                "total": round(total_time * 1000, 1),
            },
            "groundedness_score": validation_report.get("groundedness_score"),
            "confidence_decision": refusal_decision.get("decision"),
            "confidence_score": refusal_decision.get("confidence_score"),
            "refusal": refusal_decision.get("decision") == "INSUFFICIENT_EVIDENCE",
            "refusal_reasons": refusal_decision.get("reasons", []),
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry