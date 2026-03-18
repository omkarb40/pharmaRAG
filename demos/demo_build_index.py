"""
PharmaRAG — Build ChromaDB + BM25 Indexes
Stores embeddings in ChromaDB with metadata for filtered search.
Builds a parallel BM25 index for keyword matching.

Why ChromaDB?
  - Persistent storage (survives restarts)
  - Metadata filtering (search only "contraindications" chunks)
  - Proper vector similarity with cosine distance
  - Industry-standard for production RAG systems

Why BM25 alongside it?
  - Exact keyword matches (drug names, section names, medical codes)
  - Catches things vector search misses (e.g. "PML" → "progressive multifocal leukoencephalopathy")
"""

import json
import pickle
import time
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_NAME = "neuml/pubmedbert-base-embeddings"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
CHROMA_DIR = Path("data/index/chromadb")
BM25_DIR = Path("data/index/bm25")
COLLECTION_NAME = "pharma_rag_chunks"


# ──────────────────────────────────────────────
# Load chunks
# ──────────────────────────────────────────────

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


# ──────────────────────────────────────────────
# STEP 1: Build ChromaDB vector index
# ──────────────────────────────────────────────

def build_chroma_index(chunks: list[dict]):
    """
    Embeds all chunks with PubMedBERT and stores them in ChromaDB.
    
    ChromaDB stores three things per document:
      1. The embedding vector (768 floats)
      2. The document text (for retrieval)
      3. Metadata (drug name, section, etc. — for filtering)
    
    The "cosine" space means similarity = 1 - distance.
    """
    print("\n📦 STEP 1: Building ChromaDB vector index")
    print("─" * 50)

    # Load embedding model
    print(f"  Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")

    # Generate embeddings for all chunks
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  ✓ Embeddings generated in {time.time() - start:.1f}s")

    # Initialize ChromaDB with persistent storage
    # PersistentClient saves to disk — data survives when you stop the script
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists (clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Create collection with cosine similarity
    # "hnsw:space": "cosine" means the distance metric is cosine distance
    # Similarity = 1 - distance (so distance 0 = identical, distance 2 = opposite)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data for ChromaDB
    # ChromaDB requires: ids (unique strings), embeddings, documents, metadatas
    # Metadata must be flat (string/int/float only — no nested dicts or lists)
    ids = [c["chunk_id"] for c in chunks]
    documents = texts
    metadatas = []
    for c in chunks:
        metadatas.append({
            "drug_name": c["drug_name"],
            "generic_name": c["generic_name"],
            "section_name": c["section_name"],
            "set_id": c["set_id"],
            "loinc_code": c["loinc_code"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
        })

    # Upsert in batches (ChromaDB has a batch size limit of ~5000)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"  ✓ ChromaDB: {collection.count()} chunks indexed")
    print(f"  ✓ Stored at: {CHROMA_DIR}")

    return collection


# ──────────────────────────────────────────────
# STEP 2: Build BM25 keyword index
# ──────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]):
    """
    Builds a BM25 index for keyword search.
    
    BM25 (Best Matching 25) is a ranking function that scores documents
    based on query term frequency, document length, and corpus statistics.
    It's the algorithm behind most traditional search engines.
    
    We tokenize by simple whitespace splitting (lowercased).
    For production, you'd use a medical tokenizer, but this works well
    for drug names, section names, and medical terms.
    """
    print("\n🔤 STEP 2: Building BM25 keyword index")
    print("─" * 50)

    # Tokenize: split each chunk's text into lowercase words
    tokenized_corpus = [c["text"].lower().split() for c in chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Save to disk
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    with open(BM25_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Save chunk_id mapping (BM25 returns integer indices,
    # we need to map them back to chunk IDs)
    chunk_ids = [c["chunk_id"] for c in chunks]
    with open(BM25_DIR / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    print(f"  ✓ BM25: {len(tokenized_corpus)} documents indexed")
    print(f"  ✓ Stored at: {BM25_DIR}")

    return bm25, chunk_ids


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PharmaRAG — Phase 1.2 Index Builder")
    print("=" * 60)

    # Load chunks
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found.")
        print("Run ingestion first: python demo_ingest.py")
        return

    chunks = load_chunks()
    print(f"\nLoaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Show what we're indexing
    drugs = sorted(set(c["drug_name"] for c in chunks))
    sections = sorted(set(c["section_name"] for c in chunks))
    print(f"  Drugs:    {len(drugs)} → {', '.join(drugs[:5])}{'...' if len(drugs) > 5 else ''}")
    print(f"  Sections: {len(sections)} → {', '.join(sections[:4])}...")

    # Build both indexes
    build_chroma_index(chunks)
    build_bm25_index(chunks)

    print(f"\n{'=' * 60}")
    print("✅ All indexes built successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()