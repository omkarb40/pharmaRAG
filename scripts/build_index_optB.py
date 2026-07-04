"""
Option B: rebuild indexes with context-enriched embeddings.
Overwrites data/index/chromadb IN PLACE. Baseline is preserved in
data/index/chromadb_baseline (back that up FIRST). Restore afterward.

Usage: python scripts/build_index_optB.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from src.indexing.index_builder import IndexBuilder


def main():
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: chunks not found at {chunks_path}")
        sys.exit(1)

    print("=" * 60)
    print("PharmaRAG — Index Builder (OPTION B: context embeddings)")
    print("=" * 60)

    builder = IndexBuilder()
    builder.build_all(chunks_path, context_embeddings=True)

    print("\nOption B index built (in place). Baseline preserved in chromadb_baseline/.")


if __name__ == "__main__":
    main()