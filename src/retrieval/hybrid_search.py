"""
Hybrid retrieval: combines BM25 keyword search with ChromaDB vector search.
Weighted reciprocal rank fusion.
"""

import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from configs.settings import settings
from src.indexing.embedder import PubMedEmbedder


class HybridRetriever:
    """
    Retrieves chunks using weighted fusion of BM25 + semantic search.
    Default: 60% semantic / 40% BM25.
    """

    def __init__(self):
        self.embedder = PubMedEmbedder()

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.index_dir / "chromadb")
        )
        self.collection = self.chroma_client.get_collection(
            name=settings.chroma_collection_name
        )

        # Load BM25
        bm25_dir = settings.index_dir / "bm25"
        with open(bm25_dir / "bm25_index.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with open(bm25_dir / "chunk_ids.json", "r") as f:
            self.bm25_chunk_ids: list[str] = json.load(f)

        self.semantic_weight = settings.semantic_weight
        self.bm25_weight = settings.bm25_weight

    def _semantic_search(
        self, query: str, top_k: int, section_filter: str | None = None
    ) -> list[dict]:
        """Vector similarity search via ChromaDB."""
        query_embedding = self.embedder.embed_query(query)

        where_filter = None
        if section_filter:
            where_filter = {"section_name": section_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "semantic_score": 1 - results["distances"][0][i],  # cosine sim
            })
        return hits

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Keyword search via BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.bm25_chunk_ids[idx]
                # Fetch full doc from ChromaDB by ID
                result = self.collection.get(
                    ids=[chunk_id],
                    include=["documents", "metadatas"],
                )
                if result["ids"]:
                    hits.append({
                        "chunk_id": chunk_id,
                        "text": result["documents"][0],
                        "metadata": result["metadatas"][0],
                        "bm25_score": float(scores[idx]),
                    })
        return hits

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_retrieval,
        section_filter: str | None = None,
    ) -> list[dict]:
        """
        Hybrid search with reciprocal rank fusion.
        Returns top_k chunks sorted by fused score.
        """
        semantic_hits = self._semantic_search(query, top_k, section_filter)
        bm25_hits = self._bm25_search(query, top_k)

        # Reciprocal Rank Fusion (k=60 is standard)
        k = 60
        fused_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for rank, hit in enumerate(semantic_hits):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + self.semantic_weight * (1 / (k + rank + 1))
            chunk_data[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + self.bm25_weight * (1 / (k + rank + 1))
            if cid not in chunk_data:
                chunk_data[cid] = hit

        # Sort by fused score
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            entry = chunk_data[cid]
            entry["fused_score"] = fused_scores[cid]
            results.append(entry)

        return results