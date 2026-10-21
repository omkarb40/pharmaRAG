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

from __future__ import annotations

from configs.settings import settings

ABSTENTION_MARKER = "insufficient evidence"


class RefusalGuard:
    """Three-tier confidence decision system."""

    def __init__(self):
        self.refuse_threshold = 0.45 # Raised from 0.35 
        self.caution_threshold = 0.65 
        print(f"[RefusalGuard] Thresholds → Refuse: <{self.refuse_threshold}, "
              f"Caution: <{self.caution_threshold}, Answer: >={self.caution_threshold}")

    def evaluate(
        self,
        retrieval_results: list[dict],
        validation_report: dict,
        answer: str | None = None,
        respect_abstention: bool = False,
    ) -> dict:
        """respect_abstention (post-hoc fix, 2026-10-21): the results-v1 runs showed the
        guard RELEASING answers in which the generator had already abstained, because the
        abstention sentence scores 0 but the other sentences were grounded (ADV-001,
        ADV-002 in Config C). With the flag on, a generator abstention is final.
        Default False so results-v1 stays exactly reproducible."""
        # Signal 1: Retrieval quality
        fused_scores = [r.get("fused_score", 0) for r in retrieval_results]
        avg_retrieval = sum(fused_scores) / len(fused_scores) if fused_scores else 0
        retrieval_signal = min(avg_retrieval / 0.015, 1.0)

        # Signal 2: Groundedness
        groundedness = validation_report.get("groundedness_score", 0)

        # Signal 3: Result count
        num_results = len(retrieval_results)
        count_signal = min(num_results / settings.top_k_final, 1.0)

        # Updated weights — groundedness is now dominant
        confidence = (
            0.25 * retrieval_signal    # Reduced from 0.35
            + 0.55 * groundedness      # Increased from 0.45
            + 0.20 * count_signal      # Same
        )

        # Decision
        if confidence >= self.caution_threshold:
            decision = "ANSWER"
        elif confidence >= self.refuse_threshold:
            decision = "ANSWER_WITH_CAUTION"
        else:
            decision = "INSUFFICIENT_EVIDENCE"

        # Reasons
        reasons = []
        if retrieval_signal < 0.5:
            reasons.append(
                f"Low retrieval quality (avg fused score: {avg_retrieval:.6f})"
            )
        if groundedness < 0.5:
            reasons.append(
                f"Groundedness below 50%: {groundedness:.1%} of answer sentences "
                f"are supported by evidence"
            )
        if num_results < 3:
            reasons.append(f"Only {num_results} evidence chunks retrieved")

        unsupported = validation_report.get("unsupported_sentences", 0)
        if unsupported > 0:
            reasons.append(
                f"{unsupported} answer sentence(s) not supported by evidence"
            )

        overridden = False
        if respect_abstention and answer and ABSTENTION_MARKER in answer.lower() \
                and decision != "INSUFFICIENT_EVIDENCE":
            decision, overridden = "INSUFFICIENT_EVIDENCE", True
            reasons.append("Generator abstained; abstention is final (respect_abstention)")

        return {
            "decision": decision,
            "abstention_override": overridden,
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