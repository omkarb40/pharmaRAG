"""
Helper to find gold reference snippets from the actual corpus.
Use this while annotating the test set — it shows you the real chunk
text for a given drug + section so you can copy verbatim spans.

Usage:
  python scripts/find_snippet.py --drug Tysabri --section contraindications
  python scripts/find_snippet.py --search "progressive multifocal leukoencephalopathy"
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings


def load_chunks() -> list[dict]:
    chunks = []
    with open(settings.chunks_file, "r") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", type=str, help="Drug name (brand or generic)")
    parser.add_argument("--section", type=str, help="Section name")
    parser.add_argument("--search", type=str, help="Free-text search across all chunks")
    parser.add_argument("--full", action="store_true", help="Show full chunk text, not truncated")
    args = parser.parse_args()

    chunks = load_chunks()

    matches = []
    for c in chunks:
        drug_ok = True
        section_ok = True
        search_ok = True

        if args.drug:
            d = args.drug.lower()
            drug_ok = (d in c.get("drug_name", "").lower() or
                       d in c.get("generic_name", "").lower())
        if args.section:
            section_ok = args.section.lower() in c.get("section_name", "").lower()
        if args.search:
            search_ok = args.search.lower() in c.get("text", "").lower()

        if drug_ok and section_ok and search_ok:
            matches.append(c)

    print(f"\nFound {len(matches)} matching chunk(s)\n" + "=" * 70)
    for c in matches:
        print(f"\nchunk_id:   {c['chunk_id']}")
        print(f"drug:       {c['drug_name']} ({c['generic_name']})")
        print(f"section:    {c['section_name']}")
        print(f"chunk:      {c['chunk_index']+1}/{c['total_chunks']}")
        text = c["text"] if args.full else c["text"][:400]
        print(f"text:       {text}{'...' if not args.full and len(c['text']) > 400 else ''}")
        print("-" * 70)


if __name__ == "__main__":
    main()