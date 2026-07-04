"""
Validate the annotated test set before locking the baseline.

Checks:
  - every 'answer' query has a non-null reference_chunk_id + snippet
  - every 'refuse' query has null reference fields
  - each reference_snippet is an EXACT substring of its referenced chunk
    (whitespace-normalized, so copy-paste line breaks don't cause false fails)
  - referenced chunk_ids actually exist in the corpus

NOTE: This checks that a snippet matches the chunk it points to. It does NOT
verify the chunk is the correct drug's formulation — a snippet can be a valid
substring of a wrong-formulation chunk and still pass here. Ground-truth
correctness (right drug/label) is a separate, manual check.

Usage: python scripts/validate_testset.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

TEST_FILE = Path("evaluation/test_queries_v2.json")


def load_chunks_by_id() -> dict:
    by_id = {}
    with open(settings.chunks_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            by_id[c["chunk_id"]] = c
    return by_id


def norm(s: str) -> str:
    # Normalize whitespace so copy-paste line breaks don't cause false fails
    return " ".join(s.split())


def main():
    with open(TEST_FILE, "r") as f:
        queries = json.load(f)
    chunks = load_chunks_by_id()

    errors = []
    warnings = []
    filled = 0
    refuse_count = 0

    for q in queries:
        qid = q["id"]
        behavior = q.get("expected_behavior", "answer")
        chunk_id = q.get("reference_chunk_id")
        snippet = q.get("reference_snippet")

        if behavior == "refuse":
            refuse_count += 1
            if chunk_id or snippet:
                errors.append(f"{qid}: refusal query should have null reference fields")
            continue

        # answer queries
        if not chunk_id or not snippet:
            warnings.append(f"{qid}: answer query still missing reference (null)")
            continue

        if chunk_id not in chunks:
            errors.append(f"{qid}: reference_chunk_id '{chunk_id}' not found in corpus")
            continue

        chunk_text = norm(chunks[chunk_id]["text"])
        if norm(snippet) not in chunk_text:
            errors.append(f"{qid}: snippet is NOT a verbatim substring of chunk {chunk_id}")
            continue

        filled += 1

    total = len(queries)
    print(f"\nValidation results for {total} queries")
    print("=" * 60)
    print(f"  Correctly annotated answer queries: {filled}")
    print(f"  Refusal queries (null refs, correct): {refuse_count}")
    print(f"  Answer queries still needing annotation: {len(warnings)}")
    print(f"  Errors to fix: {len(errors)}")

    if warnings:
        print(f"\n  ⚠ {len(warnings)} still need annotation:")
        for w in warnings:
            print(f"    {w}")

    if errors:
        print(f"\n  ✗ {len(errors)} ERRORS to fix:")
        for e in errors:
            print(f"    {e}")

    if not errors and not warnings:
        print("\n  ✓ All annotations valid. Ready to lock the baseline.")
    elif not errors:
        print("\n  ✓ No errors. Remaining items are just unannotated answer queries.")


if __name__ == "__main__":
    main()