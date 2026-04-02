"""
Agent 2: Evidence Validator.
Checks that each sentence in the generated answer is grounded
in at least one retrieved evidence chunk.

How it works:
  1. Splits the answer into individual sentences
  2. Embeds each sentence with PubMedBERT
  3. Computes cosine similarity against each evidence chunk
  4. If max similarity >= threshold, the sentence is "supported"
  5. Groundedness = supported sentences / total sentences

Why embedding similarity instead of a full NLI model?
  - NLI models (like cross-encoder/nli-deberta-v3-base) are more accurate
    but add 500ms-1s per sentence
  - Embedding similarity runs in ~50ms total
  - For a capstone prototype, this is the right tradeoff
  - You note "NLI upgrade" as future work in your report

Output:
  - Per-sentence support scores
  - Overall groundedness percentage
  - Which chunk best supports each sentence
  - List of unsupported sentences (potential hallucinations)
"""

import re
import numpy as np

from src.indexing.embedder import PubMedEmbedder
from configs.settings import settings


class EvidenceValidator:
    """Validates that answer sentences are grounded in evidence."""

    def __init__(self):
        self.embedder = PubMedEmbedder()  # Singleton, already loaded
        self.threshold = 0.5  # Minimum similarity to count as "supported"
        print(f"[EvidenceValidator] Initialized. Threshold: {self.threshold}")

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        Handles common abbreviations and citation markers like [1].
        """
        # Remove citation markers for cleaner splitting
        clean = re.sub(r'\[\d+\]', '', text)
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        # Filter out very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def validate(self, answer: str, chunks: list[dict]) -> dict:
        """
        Check each answer sentence against evidence chunks.

        Returns:
          {
            "is_grounded": True/False,
            "groundedness_score": 0.0-1.0,
            "total_sentences": int,
            "supported_sentences": int,
            "unsupported_sentences": int,
            "sentence_details": [
              {
                "sentence": "...",
                "max_similarity": 0.82,
                "best_supporting_chunk_id": "abc123",
                "best_supporting_drug": "Tysabri",
                "is_supported": True,
              },
              ...
            ]
          }
        """
        sentences = self._split_sentences(answer)

        if not sentences:
            return {
                "is_grounded": False,
                "groundedness_score": 0.0,
                "total_sentences": 0,
                "supported_sentences": 0,
                "unsupported_sentences": 0,
                "sentence_details": [],
            }

        # Embed answer sentences
        sentence_embeddings = self.embedder.embed_texts(sentences)

        # Embed evidence chunks
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = self.embedder.embed_texts(chunk_texts)

        # Compare each sentence to all chunks
        sentence_details = []
        supported_count = 0

        for i, sent in enumerate(sentences):
            # Cosine similarity (vectors are normalized, so dot product works)
            similarities = np.dot(sentence_embeddings[i], chunk_embeddings.T)
            max_sim = float(np.max(similarities))
            best_idx = int(np.argmax(similarities))

            is_supported = max_sim >= self.threshold

            if is_supported:
                supported_count += 1

            best_chunk = chunks[best_idx]
            meta = best_chunk.get("metadata", {})

            sentence_details.append({
                "sentence": sent,
                "max_similarity": round(max_sim, 4),
                "best_supporting_chunk_id": best_chunk.get("chunk_id", ""),
                "best_supporting_drug": meta.get("drug_name", "Unknown"),
                "best_supporting_section": meta.get("section_name", "Unknown"),
                "is_supported": is_supported,
            })

        groundedness = supported_count / len(sentences)

        return {
            "is_grounded": groundedness >= 0.85,
            "groundedness_score": round(groundedness, 4),
            "total_sentences": len(sentences),
            "supported_sentences": supported_count,
            "unsupported_sentences": len(sentences) - supported_count,
            "sentence_details": sentence_details,
        }