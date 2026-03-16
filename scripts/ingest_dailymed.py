"""
CLI script: Ingest drug labels from DailyMed.
Usage: python scripts/ingest_dailymed.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import settings
from src.ingestion.dailymed import DailyMedClient
from src.ingestion.parser import SPLParser
from src.ingestion.chunker import SectionAwareChunker


def main():
    drug_list_path = settings.data_dir / "drug_lists" / "ms_drugs.csv"
    raw_dir = settings.raw_dir
    processed_dir = settings.processed_dir

    print("=" * 60)
    print("PharmaRAG — DailyMed Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Fetch raw SPLs
    print("\n[Step 1] Fetching drug labels from DailyMed API...")
    client = DailyMedClient()
    client.fetch_all_drugs(drug_list_path, raw_dir)

    # Step 2: Parse sections
    print("\n[Step 2] Parsing SPL sections...")
    parser = SPLParser()
    sections = parser.parse_all(raw_dir)

    # Step 3: Chunk
    print("\n[Step 3] Chunking sections...")
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_all(sections)

    # Step 4: Save
    output_path = processed_dir / "chunks.jsonl"
    chunker.save_chunks(chunks, output_path)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(chunks)} chunks saved to {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()