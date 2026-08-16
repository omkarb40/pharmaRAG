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
    Retrieves chunks using weighted Reciprocal Rank Fusion
    of BM25 (keyword) + ChromaDB (semantic) search.
    """

    def __init__(self):
        # Embedding model (singleton — loads once)
        self.embedder = PubMedEmbedder()

        # ChromaDB vector store
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_dir)
        )
        self.collection = self.chroma_client.get_collection(
            name=settings.chroma_collection_name
        )

        # BM25 keyword index
        with open(settings.bm25_dir / "bm25_index.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with open(settings.bm25_dir / "chunk_ids.json", "r") as f:
            self.bm25_chunk_ids: list[str] = json.load(f)

        # Chunk lookup table (for BM25 → full document)
        self.chunks_by_id: dict[str, dict] = {}
        with open(settings.chunks_file, "r") as f:
            for line in f:
                c = json.loads(line.strip())
                self.chunks_by_id[c["chunk_id"]] = c

        print(f"[Retriever] Ready. {self.collection.count()} chunks indexed.")

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
            cid = results["ids"][0][i]
            meta = dict(results["metadatas"][0][i])
            # Normalize metadata with the BM25 path so alias matching is symmetric
            src = self.chunks_by_id.get(cid, {})
            meta.setdefault("generic_name", src.get("generic_name", ""))
            hits.append({
                "chunk_id": cid,
                "text": results["documents"][0][i],
                "metadata": meta,
                "semantic_score": round(1 - results["distances"][0][i], 4),
            })
        return hits

    def _bm25_search(self, query: str, top_k: int, section_filter: str | None = None) -> list[dict]:
        """Keyword search via BM25, with optional section filter."""
        import re

        def _tokenize(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k * 3:][::-1]  # Over-fetch to filter

        hits = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            if len(hits) >= top_k:
                break
                
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
                
            # Apply section filter
            if section_filter and chunk.get("section_name") != section_filter:
                continue

            hits.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "metadata": {
                    "drug_name": chunk.get("drug_name", ""),
                    "generic_name": chunk.get("generic_name", ""),
                    "section_name": chunk.get("section_name", ""),
                    "set_id": chunk.get("set_id", ""),
                    "loinc_code": chunk.get("loinc_code", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 0),
                },
                "bm25_score": round(float(scores[idx]), 4),
            })
        return hits

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_final,
        section_filter: str | None = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        mode="hybrid" : weighted RRF over BM25 + dense (default, unchanged)
        mode="bm25"   : lexical retrieval only  (ablation)
        mode="dense"  : semantic retrieval only (ablation)
        """
        fetch_k = settings.top_k_retrieval
        k = settings.rrf_k

        if mode == "dense":
            hits = self._semantic_search(query, fetch_k, section_filter)
            out = []
            for rank, h in enumerate(hits[:top_k]):
                h["bm25_score"] = 0
                h["fused_score"] = round(1.0 / (k + rank + 1), 6)
                out.append(h)
            return out

        if mode == "bm25":
            hits = self._bm25_search(query, fetch_k, section_filter)
            out = []
            for rank, h in enumerate(hits[:top_k]):
                h["semantic_score"] = 0
                h["fused_score"] = round(1.0 / (k + rank + 1), 6)
                out.append(h)
            return out

        if mode != "hybrid":
            raise ValueError(f"unknown retrieval mode: {mode}")

        sem_hits = self._semantic_search(query, fetch_k, section_filter)
        bm25_hits = self._bm25_search(query, fetch_k, section_filter)

        fused_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for rank, hit in enumerate(sem_hits):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + settings.semantic_weight * (1 / (k + rank + 1))
            chunk_data[cid] = hit
            chunk_data[cid]["bm25_score"] = 0

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + settings.bm25_weight * (1 / (k + rank + 1))
            if cid in chunk_data:
                chunk_data[cid]["bm25_score"] = hit.get("bm25_score", 0)
            else:
                chunk_data[cid] = hit
                chunk_data[cid]["semantic_score"] = 0

        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            entry = chunk_data[cid]
            entry["fused_score"] = round(fused_scores[cid], 6)
            results.append(entry)
        return results