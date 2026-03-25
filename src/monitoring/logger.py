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
    """Append-only JSONL audit logger."""

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
        retrieval_results: list[dict],
        generation_result: dict,
    ) -> dict:
        """Write a complete audit log entry."""
        total_time = time.time() - context.pop("_start", time.time())

        entry = {
            "request_id": context["request_id"],
            "timestamp": context["timestamp"],
            "query": query,
            "retrieved_chunk_ids": [
                r.get("chunk_id") for r in retrieval_results
            ],
            "top_k_scores": [
                round(r.get("fused_score", 0), 6) for r in retrieval_results
            ],
            "latency_ms": {
                **{k: round(v, 1) for k, v in context.get("timings", {}).items()},
                "total": round(total_time * 1000, 1),
            },
            "model": generation_result.get("model", ""),
            "generation_time_ms": generation_result.get("generation_time_ms", 0),
            "num_evidence_chunks": generation_result.get("num_evidence_chunks", 0),
        }

        if settings.enable_audit_logging:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return entry