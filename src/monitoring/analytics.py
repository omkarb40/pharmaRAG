"""
Monitoring analytics engine.
Reads audit logs and computes operational metrics.

This is what turns raw log files into actionable insights:
  - Are answers getting less grounded over time? (drift)
  - How often does the system refuse? (safety signal)
  - Which pipeline stage is the bottleneck? (performance)
  - Which drugs get queried most? (usage patterns)
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

from configs.settings import settings


class MonitoringAnalytics:
    """Computes monitoring metrics from audit log files."""

    def __init__(self):
        self.log_dir = settings.logs_dir

    def _load_all_logs(self) -> list[dict]:
        """Load all audit log entries across all log files."""
        entries = []
        if not self.log_dir.exists():
            return entries

        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return entries

    def get_summary(self) -> dict:
        """
        Compute comprehensive monitoring summary.
        This powers the /api/monitoring endpoint and dashboard.
        """
        entries = self._load_all_logs()

        if not entries:
            return {
                "status": "no_data",
                "message": "No audit logs found. Ask some questions first.",
                "total_requests": 0,
            }

        # ── Basic counts ──
        total = len(entries)
        decisions = Counter(e.get("confidence_decision", "") for e in entries)
        refusals = sum(1 for e in entries if e.get("refusal", False))

        # ── Groundedness tracking ──
        groundedness_scores = [
            e.get("groundedness_score", 0) for e in entries
            if e.get("groundedness_score") is not None
        ]
        avg_groundedness = (
            sum(groundedness_scores) / len(groundedness_scores)
            if groundedness_scores else 0
        )

        # Groundedness over time (last 20 requests)
        recent_groundedness = [
            {
                "request_id": e.get("request_id"),
                "timestamp": e.get("timestamp", ""),
                "groundedness": e.get("groundedness_score", 0),
                "decision": e.get("confidence_decision", ""),
            }
            for e in entries[-20:]
        ]

        # ── Confidence score distribution ──
        confidence_scores = [
            e.get("confidence_score", 0) for e in entries
            if e.get("confidence_score") is not None
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0
        )

        # ── Latency analysis ──
        latency_stages = {}
        for e in entries:
            lat = e.get("latency_ms", {})
            for stage, ms in lat.items():
                if isinstance(ms, (int, float)):
                    if stage not in latency_stages:
                        latency_stages[stage] = []
                    latency_stages[stage].append(ms)

        latency_summary = {}
        for stage, values in latency_stages.items():
            sorted_vals = sorted(values)
            latency_summary[stage] = {
                "avg_ms": round(sum(values) / len(values), 1),
                "min_ms": round(min(values), 1),
                "max_ms": round(max(values), 1),
                "p50_ms": round(sorted_vals[len(sorted_vals) // 2], 1),
                "p95_ms": round(
                    sorted_vals[int(len(sorted_vals) * 0.95)], 1
                ) if len(sorted_vals) >= 2 else round(max(values), 1),
                "count": len(values),
            }

        # ── Drug query frequency ──
        drug_counts = Counter()
        for e in entries:
            for drug in e.get("retrieved_drugs", []):
                if drug:
                    drug_counts[drug] += 1

        # ── Section query frequency ──
        section_counts = Counter()
        for e in entries:
            for sec in e.get("routed_sections", []):
                if sec:
                    section_counts[sec] += 1

        # ── Retrieval quality ──
        avg_fused_scores = [
            e.get("avg_fused_score", 0) for e in entries
            if e.get("avg_fused_score") is not None
        ]
        avg_retrieval_quality = (
            sum(avg_fused_scores) / len(avg_fused_scores)
            if avg_fused_scores else 0
        )

        # ── Unsupported sentences (hallucination tracking) ──
        total_sentences = sum(
            e.get("total_sentences", 0) for e in entries
        )
        total_unsupported = sum(
            e.get("unsupported_sentences", 0) for e in entries
        )
        hallucination_rate = (
            total_unsupported / total_sentences
            if total_sentences > 0 else 0
        )

        # ── Recent requests log ──
        recent_requests = []
        for e in entries[-10:]:
            recent_requests.append({
                "request_id": e.get("request_id"),
                "timestamp": e.get("timestamp", ""),
                "query": e.get("query", "")[:80],
                "decision": e.get("confidence_decision", ""),
                "confidence": e.get("confidence_score", 0),
                "groundedness": e.get("groundedness_score", 0),
                "total_ms": e.get("latency_ms", {}).get("total", 0),
            })

        return {
            "status": "ok",
            "total_requests": total,

            # Decision distribution
            "decisions": {
                "ANSWER": decisions.get("ANSWER", 0),
                "ANSWER_WITH_CAUTION": decisions.get("ANSWER_WITH_CAUTION", 0),
                "INSUFFICIENT_EVIDENCE": decisions.get("INSUFFICIENT_EVIDENCE", 0),
            },
            "refusal_rate": round(refusals / total, 4) if total > 0 else 0,

            # Quality metrics
            "avg_groundedness": round(avg_groundedness, 4),
            "avg_confidence": round(avg_confidence, 4),
            "avg_retrieval_quality": round(avg_retrieval_quality, 6),
            "hallucination_rate": round(hallucination_rate, 4),
            "total_sentences_generated": total_sentences,
            "total_unsupported_sentences": total_unsupported,

            # Latency
            "latency": latency_summary,

            # Usage patterns
            "top_drugs_queried": dict(drug_counts.most_common(10)),
            "top_sections_routed": dict(section_counts.most_common()),

            # Time series (for drift detection)
            "recent_groundedness": recent_groundedness,

            # Recent activity
            "recent_requests": recent_requests,
        }