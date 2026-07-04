#!/usr/bin/env python3
"""
batch_annotation_helper.py
--------------------------
Automates the mechanical retrieval step for PharmaRAG test set annotation.
For each answer-query with null reference_snippet, runs find_snippet.py
across all acceptable sections and outputs a structured worksheet.

YOU still make the judgment call on which chunk + snippet to use.
This script just eliminates 80 terminal commands.

Usage:
    python scripts/batch_annotation_helper.py > annotation_worksheet.txt 2>&1

    Or for just a specific category:
    python scripts/batch_annotation_helper.py --category dosing

Place this in your scripts/ directory alongside find_snippet.py.
"""

import json
import subprocess
import sys
import os
import argparse
from pathlib import Path

# ---------- CONFIG ----------
# Adjust these paths relative to your project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_queries_v2.json"
FIND_SNIPPET_SCRIPT = PROJECT_ROOT / "scripts" / "find_snippet.py"
# -----------------------------


def load_test_set(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)


def get_unfilled_answer_queries(queries: list, category_filter: str = None) -> list:
    """Return answer-queries that still have null reference_snippet."""
    unfilled = []
    for q in queries:
        if q.get("expected_behavior") != "answer":
            continue
        if q.get("reference_snippet") is not None:
            continue
        if category_filter and q.get("category") != category_filter:
            continue
        unfilled.append(q)
    return unfilled


def get_mismatched_annotations(queries: list) -> list:
    """Flag queries where the filled snippet might be wrong
    (snippet exists but doesn't mention expected drug)."""
    issues = []
    for q in queries:
        if q.get("expected_behavior") != "answer":
            continue
        if q.get("reference_snippet") is None:
            continue
        drug = q.get("expected_drug")
        if not drug:
            continue
        aliases = q.get("expected_drug_aliases", [])
        snippet = q.get("reference_snippet", "").lower()
        # Check if any alias appears in the snippet
        found = any(alias.lower() in snippet for alias in aliases)
        if not found:
            issues.append({
                "id": q["id"],
                "query": q["query"],
                "expected_drug": drug,
                "snippet_preview": q["reference_snippet"][:120] + "...",
                "chunk_id": q.get("reference_chunk_id"),
            })
    return issues


def run_find_snippet(drug: str, section: str) -> str:
    """Run find_snippet.py and capture output."""
    cmd = [
        sys.executable,
        str(FIND_SNIPPET_SCRIPT),
        "--drug", drug,
        "--section", section,
        "--full",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"  [TIMEOUT] find_snippet.py --drug {drug} --section {section}"
    except Exception as e:
        return f"  [ERROR] {e}"


def process_query(query: dict) -> str:
    """Process a single query: run find_snippet for all relevant sections."""
    lines = []
    qid = query["id"]
    drug = query.get("expected_drug")
    aliases = query.get("expected_drug_aliases", [])
    sections = query.get("acceptable_sections", [])
    expected_section = query.get("expected_section")

    lines.append("=" * 80)
    lines.append(f"  {qid}  [{query.get('category')}]  difficulty={query.get('difficulty')}")
    lines.append(f"  QUERY: {query['query']}")
    lines.append(f"  expected_drug: {drug}")
    lines.append(f"  expected_section: {expected_section}")
    lines.append(f"  acceptable_sections: {sections}")
    lines.append("=" * 80)

    if not drug:
        # Patient-style queries with no expected drug
        lines.append("")
        lines.append("  [NO EXPECTED DRUG] — This is a drug-agnostic query.")
        lines.append("  You need to decide which drug's label chunk best answers this.")
        lines.append("  Consider running annotate_helper.py for this query manually,")
        lines.append("  or pick a representative drug from your corpus.")
        lines.append("")
        return "\n".join(lines)

    # Deduplicate and order sections: expected first, then others
    ordered_sections = []
    if expected_section and expected_section in sections:
        ordered_sections.append(expected_section)
    for s in sections:
        if s not in ordered_sections:
            ordered_sections.append(s)
    # If no sections listed, try expected_section alone
    if not ordered_sections and expected_section:
        ordered_sections = [expected_section]

    # For multi-drug queries, collect all drugs to search
    drugs_to_search = [drug]  # primary
    if query.get("query_type") == "multi_drug":
        # Extract other drugs from aliases that aren't aliases of the primary
        # Heuristic: brand names start with uppercase
        for alias in aliases:
            if alias != drug and alias[0].isupper() and alias not in drugs_to_search:
                # Check it's likely a different drug (not just another alias)
                # Simple heuristic: if it's not a known generic of the primary
                drugs_to_search.append(alias)
        lines.append(f"  [MULTI-DRUG] Will search: {drugs_to_search}")

    for search_drug in drugs_to_search:
        for section in ordered_sections:
            lines.append("")
            lines.append(f"  --- {search_drug} / {section} ---")
            output = run_find_snippet(search_drug, section)
            if output.strip():
                # Indent the output
                for line in output.strip().split("\n"):
                    lines.append(f"    {line}")
            else:
                lines.append(f"    [NO OUTPUT] — possible corpus gap")

    lines.append("")
    lines.append(f"  >> ACTION: Pick the best chunk_id and verbatim snippet for {qid}")
    lines.append(f"  >> If all sections return 0 chunks, flag as CORPUS GAP")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotation helper for PharmaRAG test set"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to a specific category (e.g., dosing, warnings, indications)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run mismatch audit on already-filled annotations",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Path to test_queries_v2.json (overrides default)",
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set) if args.test_set else TEST_SET_PATH
    if not test_set_path.exists():
        print(f"ERROR: Test set not found at {test_set_path}")
        sys.exit(1)

    if not FIND_SNIPPET_SCRIPT.exists():
        print(f"ERROR: find_snippet.py not found at {FIND_SNIPPET_SCRIPT}")
        sys.exit(1)

    queries = load_test_set(test_set_path)

    # --- Audit mode ---
    if args.audit:
        print("=" * 80)
        print("  MISMATCH AUDIT — filled annotations where snippet may not match drug")
        print("=" * 80)
        issues = get_mismatched_annotations(queries)
        if not issues:
            print("  No mismatches detected. (This only catches missing drug names.)")
        else:
            print(f"  Found {len(issues)} potential mismatches:\n")
            for issue in issues:
                print(f"  {issue['id']}: expected {issue['expected_drug']}")
                print(f"    chunk: {issue['chunk_id']}")
                print(f"    snippet: {issue['snippet_preview']}")
                print()
        return

    # --- Annotation mode ---
    unfilled = get_unfilled_answer_queries(queries, args.category)

    print("=" * 80)
    print(f"  PharmaRAG BATCH ANNOTATION WORKSHEET")
    print(f"  Total queries: {len(queries)}")
    print(f"  Answer queries needing annotation: {len(unfilled)}")
    if args.category:
        print(f"  Filtered to category: {args.category}")
    print(f"  Refuse/adversarial/OOS queries: auto-skipped (correctly null)")
    print("=" * 80)
    print()

    if not unfilled:
        print("  All answer-queries are filled! Run --audit to check for mismatches.")
        return

    # Group by category for readability
    by_category = {}
    for q in unfilled:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    for cat, cat_queries in by_category.items():
        print(f"\n{'#' * 80}")
        print(f"#  CATEGORY: {cat.upper()}  ({len(cat_queries)} queries)")
        print(f"{'#' * 80}")

        for q in cat_queries:
            print(process_query(q))


if __name__ == "__main__":
    main()