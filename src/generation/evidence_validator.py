"""
Agent 2: Evidence Validator.
Checks that each sentence in the generated answer is supported
by at least one retrieved chunk (lightweight NLI approach).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.settings import settings


class EvidenceValidator:
    """
    Validates that answer sentences are grounded in evidence.
    Uses embedding similarity as a lightweight NLI proxy.
    """

    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.threshold = settings.evidence_support_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def validate(self, answer: str, chunks: list[dict]) -> dict:
        """
        Check each answer sentence against evidence chunks.
        Returns validation report with per-sentence support scores.
        """
        sentences = self._split_sentences(answer)
        if not sentences:
            return {
                "is_grounded": False,
                "groundedness_score": 0.0,
                "sentence_scores": [],
            }

        # Embed answer sentences
        sentence_embeddings = self.model.encode(sentences, normalize_embeddings=True)

        # Embed evidence chunks
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = self.model.encode(chunk_texts, normalize_embeddings=True)

        sentence_scores = []
        supported_count = 0

        for i, sent in enumerate(sentences):
            # Max cosine similarity between sentence and any chunk
            similarities = np.dot(sentence_embeddings[i], chunk_embeddings.T)
            max_sim = float(np.max(similarities))
            best_chunk_idx = int(np.argmax(similarities))

            is_supported = max_sim >= self.threshold
            if is_supported:
                supported_count += 1

            sentence_scores.append({
                "sentence": sent,
                "max_similarity": round(max_sim, 4),
                "best_supporting_chunk": chunks[best_chunk_idx].get("chunk_id"),
                "is_supported": is_supported,
            })

        groundedness = supported_count / len(sentences) if sentences else 0

        return {
            "is_grounded": groundedness >= 0.85,
            "groundedness_score": round(groundedness, 4),
            "total_sentences": len(sentences),
            "supported_sentences": supported_count,
            "unsupported_sentences": len(sentences) - supported_count,
            "sentence_scores": sentence_scores,
        }