"""
PharmaRAG — Vector Search Demo
Demonstrates: embed query → embed chunks → cosine similarity → ranked results

This script does the following:
  1. Take a query → convert to vector
  2. Iterate through all chunks → generate a vector for each one
  3. Find chunks that are "close by" using cosine similarity
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_NAME = "neuml/pubmedbert-base-embeddings"
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/chunk_embeddings.npy")
TOP_K = 5


# ──────────────────────────────────────────────
# STEP 0: Load model
# ──────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """Load PubMedBERT sentence embedding model."""
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads ~500MB from HuggingFace, subsequent runs use cache)")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ──────────────────────────────────────────────
# STEP 1: Proof of concept — the "cat" test
# ──────────────────────────────────────────────

def cat_test(model: SentenceTransformer):
    """
    Professor's test: embed 3 sentences, show that semantically
    similar ones are close and dissimilar ones are far.
    """
    print("\n" + "=" * 60)
    print("TEST: Semantic Similarity Proof of Concept")
    print("=" * 60)

    sentences = [
        "The cat is walking",
        "The cat is running",
        "C is my favorite programming language",
    ]

    # Generate embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)

    print(f"\nSentences and their vector shapes:")
    for i, sent in enumerate(sentences):
        print(f"  [{i+1}] \"{sent}\"")
        print(f"       → vector shape: {embeddings[i].shape}")
        print(f"       → first 5 values: {embeddings[i][:5].round(4)}")

    # Compute cosine similarity (for normalized vectors, dot product = cosine sim)
    print(f"\nCosine Similarities:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = np.dot(embeddings[i], embeddings[j])
            label = "✓ SIMILAR" if sim > 0.5 else "✗ DIFFERENT"
            print(f"  [{i+1}] vs [{j+1}]: {sim:.4f}  {label}")
            print(f"    \"{sentences[i]}\"")
            print(f"    \"{sentences[j]}\"")
            print()

    print("Expected: [1] vs [2] should be HIGH (>0.5)")
    print("Expected: [1] vs [3] and [2] vs [3] should be LOW (<0.3)")


# ──────────────────────────────────────────────
# STEP 2: Medical domain similarity test
# ──────────────────────────────────────────────

def medical_test(model: SentenceTransformer):
    """Test with actual medical/pharma sentences."""
    print("\n" + "=" * 60)
    print("TEST: Medical Domain Similarity")
    print("=" * 60)

    sentences = [
        "Natalizumab is contraindicated in patients with PML",
        "Do not use this drug in patients with progressive multifocal leukoencephalopathy",
        "The recommended dosage is 300 mg administered by intravenous infusion",
        "Patients experienced headache, fatigue, and nausea as side effects",
    ]

    embeddings = model.encode(sentences, normalize_embeddings=True)

    print(f"\nSentences:")
    for i, s in enumerate(sentences):
        print(f"  [{i+1}] \"{s}\"")

    print(f"\nCosine Similarity Matrix:")
    print(f"       {'   '.join(f'[{i+1}]' for i in range(len(sentences)))}")
    for i in range(len(sentences)):
        sims = [np.dot(embeddings[i], embeddings[j]) for j in range(len(sentences))]
        row = "  ".join(f"{s:.3f}" for s in sims)
        print(f"  [{i+1}]  {row}")

    print(f"\nExpected:")
    print(f"  [1] vs [2]: HIGH — both about PML contraindication (different wording)")
    print(f"  [1] vs [3]: LOW  — contraindication vs dosing")
    print(f"  [3] vs [4]: LOW  — dosing vs adverse reactions")


# ──────────────────────────────────────────────
# STEP 3: Embed all chunks + vector search
# ──────────────────────────────────────────────

def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_FILE, "r") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def embed_all_chunks(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    """
    Embed every chunk in the database.
    Saves to disk so you only compute once.
    """
    if EMBEDDINGS_FILE.exists():
        print(f"\n  Loading cached embeddings from {EMBEDDINGS_FILE}...")
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"  ✓ Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        return embeddings

    print(f"\n  Embedding {len(chunks)} chunks (this takes 1-2 minutes on CPU)...")
    texts = [c["text"] for c in chunks]

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start

    # Save for reuse
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"  ✓ Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"  ✓ Saved to {EMBEDDINGS_FILE}")
    print(f"  ✓ Embedding matrix shape: {embeddings.shape}")

    return embeddings


def vector_search(
    query: str,
    model: SentenceTransformer,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    The core semantic search:
    1. Embed the query
    2. Compute cosine similarity against ALL chunk embeddings
    3. Return top-k most similar
    """
    # Step 1: Query → vector
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Step 2: Cosine similarity against all chunks
    # (For normalized vectors, dot product = cosine similarity)
    similarities = np.dot(chunk_embeddings, query_embedding)

    # Step 3: Get top-k indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        results.append({
            "rank": rank,
            "cosine_similarity": round(float(similarities[idx]), 4),
            "chunk_id": chunk["chunk_id"],
            "drug_name": chunk["drug_name"],
            "generic_name": chunk["generic_name"],
            "section_name": chunk["section_name"],
            "text_preview": chunk["text"][:200],
        })

    return results


def run_vector_search_demo(model: SentenceTransformer):
    """Full demo: embed all chunks, then search."""
    print("\n" + "=" * 60)
    print("VECTOR SEARCH OVER DRUG LABEL DATABASE")
    print("=" * 60)

    # Load chunks
    chunks = load_chunks()
    print(f"\n  Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Embed all chunks
    chunk_embeddings = embed_all_chunks(model, chunks)

    # Run example queries
    test_queries = [
        "What are the contraindications for natalizumab?",
        "Can I take this drug if I am pregnant?",
        "What are the serious brain infection risks?",
        "recommended dose administration",
        "liver problems hepatotoxicity",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'─' * 60}")

        results = vector_search(query, model, chunks, chunk_embeddings)

        for r in results:
            print(f"  [{r['rank']}] Similarity: {r['cosine_similarity']:.4f}")
            print(f"      Drug:    {r['drug_name']} ({r['generic_name']})")
            print(f"      Section: {r['section_name']}")
            print(f"      Text:    {r['text_preview']}...")
            print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PharmaRAG — Phase 1.1 Vector Search Demo")
    print("=" * 60)

    # Load model (downloads on first run)
    model = load_model()

    # Test 1: Cat test (professor's proof of concept)
    cat_test(model)

    # Test 2: Medical domain test
    medical_test(model)

    # Test 3: Full vector search over your drug label database
    run_vector_search_demo(model)


if __name__ == "__main__":
    main()