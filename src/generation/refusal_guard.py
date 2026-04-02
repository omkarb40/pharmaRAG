"""
Agent 3: Refusal Guard.
Makes the final confidence decision: Answer / Caution / Refuse.

Why this is the most important agent:
  Most RAG systems always answer, even when they shouldn't.
  In pharma, a confident hallucination about drug interactions
  could harm a patient. The Refusal Guard gives PharmaRAG the
  ability to say "I don't have enough evidence" instead of
  guessing — which is infinitely safer.

Three-tier decision:
  ANSWER               → Strong evidence, high groundedness
  ANSWER_WITH_CAUTION  → Partial evidence, some gaps
  INSUFFICIENT_EVIDENCE → Weak evidence, system refuses to answer

Signals used:
  1. Retrieval quality (are the fused scores high?)
  2. Groundedness (what % of sentences are supported by evidence?)
  3. Result count (did we find enough chunks?)
"""

from configs.settings import settings


class RefusalGuard:
    """Three-tier confidence decision system."""

    def __init__(self):
        self.refuse_threshold = 0.35
        self.caution_threshold = 0.65
        print(f"[RefusalGuard] Thresholds → Refuse: <{self.refuse_threshold}, "
              f"Caution: <{self.caution_threshold}, Answer: >={self.caution_threshold}")

    def evaluate(
        self,
        retrieval_results: list[dict],
        validation_report: dict,
    ) -> dict:
        """
        Evaluate confidence and produce a decision.

        Returns:
          {
            "decision": "ANSWER" | "ANSWER_WITH_CAUTION" | "INSUFFICIENT_EVIDENCE",
            "confidence_score": 0.0-1.0,
            "signals": { ... },
            "reasons": ["...", "..."],
          }
        """
        # ── Signal 1: Retrieval quality ──
        # Average fused score of retrieved chunks
        fused_scores = [r.get("fused_score", 0) for r in retrieval_results]
        avg_retrieval = sum(fused_scores) / len(fused_scores) if fused_scores else 0

        # Normalize: RRF scores are small (0.005-0.02 range)
        # A "good" average is ~0.012+, "weak" is <0.008
        retrieval_signal = min(avg_retrieval / 0.015, 1.0)

        # ── Signal 2: Groundedness ──
        groundedness = validation_report.get("groundedness_score", 0)

        # ── Signal 3: Result count ──
        num_results = len(retrieval_results)
        count_signal = min(num_results / settings.top_k_final, 1.0)

        # ── Composite confidence ──
        confidence = (
            0.35 * retrieval_signal
            + 0.45 * groundedness
            + 0.20 * count_signal
        )

        # ── Decision ──
        if confidence >= self.caution_threshold:
            decision = "ANSWER"
        elif confidence >= self.refuse_threshold:
            decision = "ANSWER_WITH_CAUTION"
        else:
            decision = "INSUFFICIENT_EVIDENCE"

        # ── Reasons ──
        reasons = []
        if retrieval_signal < 0.5:
            reasons.append(
                f"Low retrieval quality (avg fused score: {avg_retrieval:.6f})"
            )
        if groundedness < 0.85:
            reasons.append(
                f"Groundedness below 85%: {groundedness:.1%} of answer sentences "
                f"are supported by evidence"
            )
        if num_results < 3:
            reasons.append(f"Only {num_results} evidence chunks retrieved")

        unsupported = validation_report.get("unsupported_sentences", 0)
        if unsupported > 0:
            reasons.append(
                f"{unsupported} answer sentence(s) not supported by evidence"
            )

        return {
            "decision": decision,
            "confidence_score": round(confidence, 4),
            "signals": {
                "retrieval_quality": round(retrieval_signal, 4),
                "groundedness": round(groundedness, 4),
                "result_count_signal": round(count_signal, 4),
                "avg_fused_score": round(avg_retrieval, 6),
                "num_results": num_results,
            },
            "reasons": reasons,
        }