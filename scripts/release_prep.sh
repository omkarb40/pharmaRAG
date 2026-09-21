#!/usr/bin/env bash
# release_prep.sh — one-time cleanup before making pharmaRAG public.
#
# Safe by default: prints what it WOULD do and changes nothing.
# Run with --apply to actually make the changes.
#
#   bash scripts/release_prep.sh            # dry run
#   bash scripts/release_prep.sh --apply    # do it
#
# Nothing is deleted from disk except __pycache__ and .DS_Store. Everything
# else is either untracked from git (file stays on your Mac) or moved into an
# archive folder. Your working copy keeps functioning after this runs.

set -uo pipefail
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
echo "Repo: $ROOT"
[[ $APPLY -eq 0 ]] && echo ">>> DRY RUN — nothing will change. Re-run with --apply to commit these changes." && echo

run() {
  if [[ $APPLY -eq 1 ]]; then
    eval "$@"
  else
    echo "    would run: $*"
  fi
}

say() { echo; echo "── $* ─────────────────────────────────"; }

# ---------------------------------------------------------------- 1. junk
say "1. Removing OS and Python junk from disk"
if [[ $APPLY -eq 1 ]]; then
  find . -name '.DS_Store' -not -path './.git/*' -delete
  find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null
  echo "    removed .DS_Store and __pycache__"
else
  echo "    would delete $(find . -name '.DS_Store' -not -path './.git/*' | wc -l | tr -d ' ') .DS_Store file(s)"
  echo "    would delete $(find . -name '__pycache__' -type d -not -path './.git/*' | wc -l | tr -d ' ') __pycache__ dir(s)"
fi

# --------------------------------------------------- 2. untrack heavy paths
say "2. Untracking heavy / rebuildable paths (files stay on disk)"
UNTRACK=(
  "chroma_db"
  "data/index"
  "data/raw"
  "data/processed/chunk_embeddings.npy"
  "data/processed/chunks_backup_20260630.jsonl"
  "demos/data"
  "external"
  "venv"
  ".venv"
  "annotation_worksheet.txt"
  "worksheet_dosing.txt"
  "worksheet_indications.txt"
  "RAG AI and MLOps for Healthcare and Pharma.docx"
  "RAG Reliability and Governance.docx"
  "PharmaRAG_Project_Report.docx"
)
for p in "${UNTRACK[@]}"; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1 || [[ -n "$(git ls-files "$p" 2>/dev/null)" ]]; then
    echo "  tracked: $p"
    run "git rm -r --cached --quiet '$p'"
  else
    echo "  not tracked (fine): $p"
  fi
done

# ----------------------------------------------- 3. archive stale results
say "3. Archiving non-canonical evaluation results"
CANONICAL=(
  "configN_FINAL.json"
  "configN_full_FINAL.json"
  "baseline_FINAL_t0.json"
  "baseline_full_FINAL_t0.json"
  "optA_FINAL_t0.json"
  "optA_full_FINAL_t0.json"
  "optB_FINAL_t0.json"
  "optB_full_FINAL_t0.json"
  "optC_FINAL_t0.json"
  "optC_full_FINAL_t0.json"
  "optC_metrics_v2_LOCKED.json"
  "optC_LOCKED.json"
  "retrieval_ablation.json"
  "aggregate_metrics_configN_noagents.json"
  "aggregate_metrics_baseline.json"
  "aggregate_metrics_optA_expand_v2.json"
  "aggregate_metrics_optB_context_v2.json"
  "aggregate_metrics_optC_rerank.json"
)
ARCHIVE="evaluation/results/_archive"
run "mkdir -p '$ARCHIVE'"
moved=0; kept=0
for f in evaluation/results/*.json; do
  [[ -e "$f" ]] || continue
  base=$(basename "$f")
  keep=0
  for c in "${CANONICAL[@]}"; do [[ "$base" == "$c" ]] && keep=1 && break; done
  if [[ $keep -eq 1 ]]; then
    kept=$((kept+1))
  else
    moved=$((moved+1))
    run "git mv '$f' '$ARCHIVE/$base' 2>/dev/null || mv '$f' '$ARCHIVE/$base'"
  fi
done
echo "  keeping $kept canonical file(s), archiving $moved"
echo "  NOTE: edit the CANONICAL list at the top of this script if it kept the wrong ones."
echo "  Run scripts/make_results_manifest.py AFTER this to verify which file backs which number."

# ------------------------------------------------------ 4. docs structure
say "4. Creating docs/ structure"
run "mkdir -p docs/images"
echo "  unzip the figures archive into docs/images/ if you have not already"

# --------------------------------------------------------------- 5. report
say "5. Report"
if [[ -f "PharmaRAG_Project_Report.docx" ]]; then
  echo "  Found PharmaRAG_Project_Report.docx at repo root."
  echo "  Export it to PDF and save as docs/PharmaRAG_Report.pdf — the README links there."
fi

# ---------------------------------------------------------------- 6. status
say "6. Result"
if [[ $APPLY -eq 1 ]]; then
  echo "  Done. Review with:  git status"
  echo "  Then:               git add -A && git commit -m 'Prepare repository for public release'"
else
  echo "  Dry run complete. Re-run with --apply when the plan above looks right."
fi
echo
