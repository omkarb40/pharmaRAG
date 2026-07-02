"""
Option C: Cross-encoder reranking.

A bi-encoder (PubMedBERT) retrieves candidates by embedding similarity;
a cross-encoder then re-scores each (query, chunk) PAIR jointly, which
captures query-chunk interaction that cosine similarity cannot. We
retrieve a wide candidate pool and rerank down to top-k.

Design note for the paper: reranking addresses "did we retrieve the
right evidence?" (a retrieval-ordering failure), distinct from the
Evidence Validator, which addresses "did the model use the evidence
correctly?" (a generation failure). The two are complementary layers.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieved chunks by joint (query, chunk) scoring."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print(f"[Reranker] Loading cross-encoder: {_MODEL_NAME}")
        self.model = CrossEncoder(_MODEL_NAME)
        self._initialized = True
        print("[Reranker] Ready.")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-score chunks against the query, return top_k by cross-encoder score.
        Preserves the original chunk dicts; attaches 'rerank_score'.
        """
        if not chunks:
            return chunks
        pairs = [(query, c.get("text", "")) for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]