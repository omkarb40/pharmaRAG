#!/usr/bin/env python3
"""
make_results_manifest.py — trace every published number back to the file that produced it.

Walks every JSON under evaluation/results/ (including _archive/), flattens it,
and for each number published in the README/report finds which file(s) contain
that value and at which key path.

    python scripts/make_results_manifest.py              # print report
    python scripts/make_results_manifest.py --write      # also write docs/RESULTS.md

Why this exists: the project's central claim is that you can tell which number
came from where. A results directory with PRE_FIX, LOCKED, v2 and FINAL_t0
variants does not support that claim on sight. This script does.

Exit code is 1 if any published value cannot be located, so it can gate a release.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"

# Every number published in the README, as (section, label, value).
# Tolerance is absolute; published figures are rounded to 3 dp.
PUBLISHED = [
    # --- Table: five-configuration ablation -------------------------------
    ("Ablation · N (no agents)",      "Recall@5",        0.586),
    ("Ablation · N (no agents)",      "Gold-chunk@5",    0.328),
    ("Ablation · N (no agents)",      "nDCG@5",          0.894),
    ("Ablation · N (no agents)",      "Groundedness",    0.945),
    ("Ablation · N (no agents)",      "Hallucination",   0.060),
    ("Ablation · N (no agents)",      "Refusal acc.",    0.682),
    ("Ablation · N (no agents)",      "Unsafe emission", 1.000),

    ("Ablation · Baseline (hybrid)",  "Recall@5",        0.707),
    ("Ablation · Baseline (hybrid)",  "Gold-chunk@5",    0.448),
    ("Ablation · Baseline (hybrid)",  "nDCG@5",          0.965),
    ("Ablation · Baseline (hybrid)",  "Groundedness",    0.960),
    ("Ablation · Baseline (hybrid)",  "Hallucination",   0.049),
    ("Ablation · Baseline (hybrid)",  "Refusal acc.",    0.706),
    ("Ablation · Baseline (hybrid)",  "Unsafe emission", 0.185),

    ("Ablation · A (query expansion)", "Recall@5",        0.724),
    ("Ablation · A (query expansion)", "Gold-chunk@5",    0.466),
    ("Ablation · A (query expansion)", "nDCG@5",          0.951),
    ("Ablation · A (query expansion)", "Groundedness",    0.930),
    ("Ablation · A (query expansion)", "Hallucination",   0.070),
    ("Ablation · A (query expansion)", "Refusal acc.",    0.729),
    ("Ablation · A (query expansion)", "Unsafe emission", 0.222),

    ("Ablation · B (context embeddings)", "Recall@5",        0.741),
    ("Ablation · B (context embeddings)", "Gold-chunk@5",    0.431),
    ("Ablation · B (context embeddings)", "nDCG@5",          0.958),
    ("Ablation · B (context embeddings)", "Groundedness",    0.939),
    ("Ablation · B (context embeddings)", "Hallucination",   0.061),
    ("Ablation · B (context embeddings)", "Refusal acc.",    0.741),
    ("Ablation · B (context embeddings)", "Unsafe emission", 0.148),

    ("Ablation · C (reranking, final)", "Recall@5",        0.845),
    ("Ablation · C (reranking, final)", "Gold-chunk@5",    0.603),
    ("Ablation · C (reranking, final)", "nDCG@5",          0.978),
    ("Ablation · C (reranking, final)", "Groundedness",    0.932),
    ("Ablation · C (reranking, final)", "Hallucination",   0.082),
    ("Ablation · C (reranking, final)", "Refusal acc.",    0.777),
    ("Ablation · C (reranking, final)", "Unsafe emission", 0.111),

    # --- Table: retrieval component ladder --------------------------------
    ("Component ladder · BM25 only",      "Recall@5",     0.5345),
    ("Component ladder · BM25 only",      "Gold-chunk@5", 0.2931),
    ("Component ladder · BM25 only",      "nDCG@5",       0.9155),
    ("Component ladder · Dense only",     "Recall@5",     0.7759),
    ("Component ladder · Dense only",     "Gold-chunk@5", 0.5172),
    ("Component ladder · Dense only",     "nDCG@5",       0.9602),
    ("Component ladder · Hybrid RRF",     "Recall@5",     0.7069),
    ("Component ladder · Hybrid RRF",     "Gold-chunk@5", 0.4483),
    ("Component ladder · Hybrid RRF",     "nDCG@5",       0.9626),
    ("Component ladder · Hybrid + rerank", "Recall@5",     0.8448),
    ("Component ladder · Hybrid + rerank", "Gold-chunk@5", 0.6034),
    ("Component ladder · Hybrid + rerank", "nDCG@5",       0.9770),

    # --- Router contribution ----------------------------------------------
    ("Router ablation", "Recall@5 with router",    0.707),
    ("Router ablation", "Recall@5 without router", 0.621),
]

TOLERANCE = 0.0006  # published values are rounded to 3 dp


def flatten(obj, prefix=""):
    """Yield (key_path, value) for every scalar in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def load_all():
    """Return {relative_path: [(key_path, numeric_value), ...]} for every result JSON."""
    if not RESULTS_DIR.is_dir():
        sys.exit(f"ERROR: {RESULTS_DIR} not found. Run this from the repo root.")

    index = {}
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"  ! skipping unreadable {path.name}: {exc}", file=sys.stderr)
            continue
        nums = [
            (kp, float(v))
            for kp, v in flatten(data)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        ]
        index[str(path.relative_to(ROOT))] = nums
    return index


def find(index, target):
    """Return [(file, key_path, value)] for every occurrence of target within tolerance."""
    hits = []
    for rel, nums in index.items():
        for kp, v in nums:
            if abs(v - target) <= TOLERANCE:
                hits.append((rel, kp, v))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write docs/RESULTS.md in addition to printing")
    args = ap.parse_args()

    index = load_all()
    print(f"Scanned {len(index)} result file(s) under {RESULTS_DIR.relative_to(ROOT)}/\n")

    rows, missing, ambiguous = [], [], []
    by_section = defaultdict(list)

    for section, label, value in PUBLISHED:
        hits = find(index, value)
        # Prefer files outside _archive/ — those are the canonical ones.
        live = [h for h in hits if "_archive" not in h[0]]
        chosen = live or hits

        if not chosen:
            missing.append((section, label, value))
            source = "**NOT FOUND**"
        else:
            files = sorted({h[0] for h in chosen})
            if len(files) > 3:
                ambiguous.append((section, label, value, len(files)))
            kp = chosen[0][1] or "(root)"
            source = f"`{files[0]}` → `{kp}`"
            if len(files) > 1:
                source += f" _(+{len(files) - 1} other file(s))_"

        rows.append((section, label, value, source))
        by_section[section].append((label, value, source))

    # ---- console report ------------------------------------------------
    for section, entries in by_section.items():
        print(f"{section}")
        for label, value, source in entries:
            flag = "  " if "NOT FOUND" not in source else "!!"
            print(f"  {flag} {label:<24} {value:<8} {source}")
        print()

    if missing:
        print(f"UNLOCATED: {len(missing)} published value(s) could not be found in any "
              f"result file. Either the value was hand-transcribed, the run output was "
              f"deleted, or the number in the README is wrong:")
        for section, label, value in missing:
            print(f"  - {section} / {label} = {value}")
        print()

    if ambiguous:
        print("AMBIGUOUS: these values appear in more than three files. Prune "
              "evaluation/results/ so each published number has one obvious source:")
        for section, label, value, n in ambiguous:
            print(f"  - {section} / {label} = {value}  ({n} files)")
        print()

    # ---- manifest ------------------------------------------------------
    if args.write:
        out = ROOT / "docs" / "RESULTS.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Results manifest",
            "",
            "Every number published in the README, traced to the result file that",
            "produced it. Regenerate with:",
            "",
            "```bash",
            "python scripts/make_results_manifest.py --write",
            "```",
            "",
            "Runs are deterministic at temperature 0 and reproduce on re-execution.",
            "",
        ]
        for section, entries in by_section.items():
            lines += [f"## {section}", "",
                      "| Metric | Published | Source |", "|---|---|---|"]
            for label, value, source in entries:
                lines.append(f"| {label} | {value} | {source} |")
            lines.append("")
        if missing:
            lines += ["## Unlocated values", "",
                      "These published values were not found in any tracked result file:", ""]
            lines += [f"- {s} / {l} = {v}" for s, l, v in missing]
            lines.append("")
        out.write_text("\n".join(lines))
        print(f"Wrote {out.relative_to(ROOT)}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
