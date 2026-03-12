"""
Agent 3: Refusal Guard.
Decides whether to answer, answer with caution, or refuse
based on evidence quality signals.
"""

from configs.settings import settings


class RefusalGuard:
    """
    Three-tier confidence decision:
    - ANSWER: strong evidence, high groundedness
    - CAUTION: partial evidence, some gaps
    - REFUSE: weak evidence, high hallucination risk
    """

    def __init__(self):
        self.refusal_threshold = settings.refusal_confidence_threshold
        self.caution_threshold = 0.7  # Between refuse and full answer

    def evaluate(
        self,
        retrieval_results: list[dict],
        validation_report: dict,
    ) -> dict:
        """
        Produce a confidence decision based on retrieval + validation signals.
        """
        # Signal 1: Top retrieval score
        top_scores = [r.get("fused_score", 0) for r in retrieval_results]
        avg_retrieval_score = sum(top_scores) / len(top_scores) if top_scores else 0

        # Signal 2: Groundedness from evidence validator
        groundedness = validation_report.get("groundedness_score", 0)

        # Signal 3: Number of retrieval results
        num_results = len(retrieval_results)

        # Composite confidence
        confidence = (
            0.4 * min(avg_retrieval_score / 0.02, 1.0)  # Normalize RRF scores
            + 0.4 * groundedness
            + 0.2 * min(num_results / 5, 1.0)
        )

        # Decision
        if confidence >= self.caution_threshold:
            decision = "ANSWER"
        elif confidence >= self.refusal_threshold:
            decision = "ANSWER_WITH_CAUTION"
        else:
            decision = "INSUFFICIENT_EVIDENCE"

        reasons = []
        if avg_retrieval_score < 0.005:
            reasons.append("Low retrieval scores — query may not match available labels")
        if groundedness < 0.85:
            reasons.append(f"Groundedness below 85%: {groundedness:.1%}")
        if num_results < 3:
            reasons.append(f"Only {num_results} chunks retrieved")

        return {
            "decision": decision,
            "confidence_score": round(confidence, 4),
            "signals": {
                "avg_retrieval_score": round(avg_retrieval_score, 6),
                "groundedness_score": round(groundedness, 4),
                "num_retrieval_results": num_results,
            },
            "reasons": reasons,
        }