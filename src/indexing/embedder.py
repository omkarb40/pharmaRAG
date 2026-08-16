"""
Embedding service using PubMedBERT.
Domain-specific embeddings for biomedical/pharmaceutical text.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from configs.settings import settings


class PubMedEmbedder:
    """Generates embeddings using neuml/pubmedbert-base-embeddings."""

    _instance = None  # Singleton — model loads once, reused everywhere

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print(f"[Embedder] Loading model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = settings.embedding_dimension
        self._initialized = True
        print(f"[Embedder] Ready. Dimension: {self.dimension}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts, return (n, 768) numpy array."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, return (768,) numpy array."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
    
    @staticmethod
    def build_context_prefix(chunk: dict) -> str:
        """Option B: context prefix baked into the embedding input only.
        Uses drug + human-readable section so the vector carries identity."""
        drug = chunk.get("drug_name", "")
        section = chunk.get("section_name", "").replace("_", " ")
        return f"Drug: {drug} | Section: {section}. "

    def embed_chunks_with_context(self, chunks: list[dict],
                                  batch_size: int = 32) -> np.ndarray:
        """Embed chunks with a context prefix prepended to each chunk's
        text FOR THE EMBEDDING ONLY. The stored document text is untouched."""
        enriched = [self.build_context_prefix(c) + c["text"] for c in chunks]
        return self.model.encode(
            enriched,
            batch_size=batch_size,
            show_progress_bar=len(enriched) > 50,
            normalize_embeddings=True,
        )