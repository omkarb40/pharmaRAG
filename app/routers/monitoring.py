"""
/monitoring endpoints — operational visibility into PharmaRAG.

Endpoints:
  GET /api/monitoring         → Full monitoring summary (JSON)
  GET /api/monitoring/health  → System health check with quality signals
"""

from fastapi import APIRouter

from src.monitoring.analytics import MonitoringAnalytics


router = APIRouter()
analytics = MonitoringAnalytics()


@router.get("/monitoring")
def monitoring_summary():
    """
    Full monitoring dashboard data.
    Returns metrics on quality, latency, usage patterns, and drift signals.
    """
    return analytics.get_summary()


@router.get("/monitoring/health")
def system_health():
    """
    Quick health check with quality signals.
    Use this for automated monitoring / alerting.
    """
    summary = analytics.get_summary()

    if summary["status"] == "no_data":
        return {
            "status": "no_data",
            "message": "No requests processed yet.",
        }

    # Define health thresholds
    groundedness_ok = summary["avg_groundedness"] >= 0.80
    hallucination_ok = summary["hallucination_rate"] <= 0.15
    latency_ok = (
        summary.get("latency", {}).get("total", {}).get("p95_ms", 0) <= 60000
    )
    refusal_ok = summary["refusal_rate"] <= 0.30

    all_ok = groundedness_ok and hallucination_ok and latency_ok and refusal_ok

    return {
        "status": "healthy" if all_ok else "degraded",
        "total_requests": summary["total_requests"],
        "checks": {
            "groundedness": {
                "ok": groundedness_ok,
                "value": summary["avg_groundedness"],
                "threshold": 0.80,
            },
            "hallucination_rate": {
                "ok": hallucination_ok,
                "value": summary["hallucination_rate"],
                "threshold": 0.15,
            },
            "p95_latency_ms": {
                "ok": latency_ok,
                "value": summary.get("latency", {}).get("total", {}).get("p95_ms", 0),
                "threshold": 60000,
            },
            "refusal_rate": {
                "ok": refusal_ok,
                "value": summary["refusal_rate"],
                "threshold": 0.30,
            },
        },
    }