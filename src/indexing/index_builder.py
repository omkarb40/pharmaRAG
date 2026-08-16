"""
Index builder: creates ChromaDB vector index + BM25 keyword index.
"""

import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from configs.settings import settings
from src.indexing.embedder import PubMedEmbedder
from src.ingestion.chunker import SectionAwareChunker


class IndexBuilder:
    """Builds and persists both vector (ChromaDB) and keyword (BM25) indexes."""

    def __init__(self):
        self.embedder = PubMedEmbedder()

        # ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.index_dir / "chromadb")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_vector_index(self, chunks: list[dict], context_embeddings: bool = False):
        """Embed all chunks and upsert into ChromaDB.
        If context_embeddings=True (Option B), embeddings include a
        drug/section prefix; the STORED document text stays original."""
        texts = [c["text"] for c in chunks]          # stored document (original)
        ids = [c["chunk_id"] for c in chunks]

        metadatas = []
        for c in chunks:
            metadatas.append({
                "drug_name": c["drug_name"],
                "section_name": c["section_name"],
                "set_id": c["set_id"],
                "loinc_code": c["loinc_code"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
            })

        # Embedding input differs by mode; stored documents are always original
        if context_embeddings:
            print(f"[IndexBuilder] Embedding {len(texts)} chunks WITH context prefix (Option B)...")
            embeddings = self.embedder.embed_chunks_with_context(chunks)
        else:
            print(f"[IndexBuilder] Embedding {len(texts)} chunks (baseline)...")
            embeddings = self.embedder.embed_texts(texts)

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=metadatas[i:end],
                documents=texts[i:end],       # original text, not prefixed
            )
        print(f"[IndexBuilder] ChromaDB: {self.collection.count()} chunks indexed")

    def build_bm25_index(self, chunks: list[dict]):
        """Build BM25 index and save to disk."""
        import re

        def _tokenize(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_path = settings.index_dir / "bm25"
        bm25_path.mkdir(parents=True, exist_ok=True)

        with open(bm25_path / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)

        # Save chunk_id mapping for BM25 score lookups
        chunk_ids = [c["chunk_id"] for c in chunks]
        with open(bm25_path / "chunk_ids.json", "w") as f:
            json.dump(chunk_ids, f)

        print(f"[IndexBuilder] BM25: {len(tokenized_corpus)} documents indexed")

    def build_all(self, chunks_path: Path, context_embeddings: bool = False):
        chunks = SectionAwareChunker.load_chunks(chunks_path)
        print(f"[IndexBuilder] Loaded {len(chunks)} chunks from {chunks_path}")
        self.build_vector_index(chunks, context_embeddings=context_embeddings)
        self.build_bm25_index(chunks)
        print("[IndexBuilder] All indexes built successfully.")