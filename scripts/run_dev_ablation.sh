#!/bin/bash
# Full development-set ablation on test_queries_v3.json (82 queries), corpus snapshot 2026-10-16.
# Usage (from project root, with `ollama serve` running):
#   caffeinate -i bash scripts/run_dev_ablation.sh
# ~3.5-4 h. Every run writes results + run_manifest_<tag>.json; console output goes to logs/.
# Config C runs on the BASELINE index (C = Baseline + cross-encoder rerank). Config B is run last
# on its own context-embedding index; the baseline index is ALWAYS rebuilt at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
STAMP=$(date +%Y%m%d_%H%M)
LOG=logs/ablation_$STAMP
mkdir -p "$LOG"
restore() { echo ">>> restoring baseline index"; python scripts/build_index.py > "$LOG/zz_restore_baseline_index.log" 2>&1 && echo ">>> baseline index restored"; }
trap restore EXIT

run () {  # name, flags...
  local name=$1; shift
  echo ">>> $(date +%H:%M) starting $name  ($*)"
  python -m src.evaluation.eval_runner "$@" > "$LOG/$name.log" 2>&1
  grep -E "Recall@5 \(alias\)|nDCG@5:|Groundedness:|Overall accuracy|UNSAFE emissions|Answers emitted" "$LOG/$name.log" | sed "s/^/    [$name] /" || true
}

ollama list > /dev/null || { echo "Ollama is not running: start 'ollama serve' first"; exit 1; }
git diff --quiet -- src configs scripts || echo "WARNING: uncommitted code changes; manifests will say code_dirty=True"

run 1_configN          --noagents
run 2_baseline
run 3_configA_expand   --expand
run 4_configC          --optC
run 5_configC_intent   --optC --intent-check
run 6_configC_noGuard  --optC --no-guard
run 7_configC_noSelfRefusal --optC --no-self-refusal

echo ">>> $(date +%H:%M) building Option B context-embedding index"
python scripts/build_index_optB.py > "$LOG/8a_build_optB_index.log" 2>&1
run 8_configB_context  --optB

echo ">>> $(date +%H:%M) ALL RUNS DONE. Logs in $LOG"
