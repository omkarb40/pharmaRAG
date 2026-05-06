"""
/evaluate endpoint — run evaluation from the API.
Returns aggregate metrics from the most recent evaluation run.
"""

import json
from pathlib import Path

from fastapi import APIRouter


router = APIRouter()

RESULTS_DIR = Path("evaluation/results")


@router.get("/evaluate/results")
def get_evaluation_results():
    """Return the most recent evaluation results."""
    metrics_file = RESULTS_DIR / "aggregate_metrics.json"

    if not metrics_file.exists():
        return {
            "status": "not_run",
            "message": "No evaluation results found. Run: python -m src.evaluation.eval_runner",
        }

    with open(metrics_file, "r") as f:
        return json.load(f)


@router.get("/evaluate/details")
def get_evaluation_details():
    """Return per-query evaluation details."""
    details_file = RESULTS_DIR / "full_pipeline_results.json"

    if not details_file.exists():
        return {
            "status": "not_run",
            "message": "No evaluation details found. Run: python -m src.evaluation.eval_runner",
        }

    with open(details_file, "r") as f:
        results = json.load(f)

    return {
        "total_queries": len(results),
        "results": results,
    }