"""
Embedding service using PubMedBERT.
Domain-specific embeddings for biomedical/pharmaceutical text.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from configs.settings import settings


class PubMedEmbedder:
    """
    Generates embeddings using neuml/pubmedbert-base-embeddings.
    768-dim vectors optimized for medical/pharma text.
    """

    def __init__(self, model_name: str = settings.embedding_model):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = settings.embedding_dimension
        print(f"[Embedder] Model loaded. Dimension: {self.dimension}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of texts, return numpy array of shape (n, 768)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]