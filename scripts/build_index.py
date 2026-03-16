"""
CLI script: Build vector + BM25 indexes from processed chunks.
Usage: python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import settings
from src.indexing.index_builder import IndexBuilder


def main():
    chunks_path = settings.processed_dir / "chunks.jsonl"

    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Run ingestion first: python scripts/ingest_dailymed.py")
        sys.exit(1)

    print("=" * 60)
    print("PharmaRAG — Index Builder")
    print("=" * 60)

    builder = IndexBuilder()
    builder.build_all(chunks_path)

    print(f"\n{'=' * 60}")
    print("Indexes built successfully!")
    print(f"ChromaDB: {settings.index_dir / 'chromadb'}")
    print(f"BM25:     {settings.index_dir / 'bm25'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()