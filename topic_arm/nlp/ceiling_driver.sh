#!/usr/bin/env bash
# Unattended driver for the ideology-recoverability ceiling (Arms 1-3).
#
# Detached on purpose: launched with setsid+nohup so it survives the SSH
# session that started it going away. Every step is IDEMPOTENT -- it skips any
# arm whose report JSON already exists -- so this can be re-run safely after an
# interruption and will only do the work that is still missing.
#
# python -u throughout: block-buffered stdout is why the first attempt's logs
# sat at 0 bytes for ten minutes and looked like a stall when the jobs were in
# fact running fine.
#
#   setsid nohup bash nlp/ceiling_driver.sh > data/output/adspan/ceiling_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=data/output/adspan
mkdir -p "$OUT"

step () {                       # step <marker-file> <label> <cmd...>
  local marker="$1"; local label="$2"; shift 2
  if [[ -s "$marker" ]]; then
    echo "[skip] $label -- $marker already exists"
    return 0
  fi
  echo "[start] $label -- $(date '+%H:%M:%S')"
  # Capture the status BEFORE any echo runs: inside the else-branch $? is the
  # status of the preceding echo, not of "$@", so it always read as 0.
  local rc=0
  "$@" || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[ok] $label -- $(date '+%H:%M:%S')"
  else
    echo "[FAIL] $label (exit $rc) -- continuing to the next arm so one failure"
    echo "       does not cost the others; re-run this driver to retry."
  fi
}

echo "=== ceiling driver started $(date '+%Y-%m-%d %H:%M:%S') ==="

# Arm 2 first: it is the claim-relevant arm (does style add beyond topic).
step "$OUT/ceiling_nested_arm2_k75.json" "Arm 2 representations" \
     $PY -u -m nlp.adspan_ceiling_nested --arm 2 --k 75

step "$OUT/ceiling_nested_arm1_k75.json" "Arm 1 estimators" \
     $PY -u -m nlp.adspan_ceiling_nested --arm 1 --k 75

# Arm 1 again on dense grids. The step above is kept because the dense run
# reports its delta against it, but the DENSE file is what the report and the
# summary CSV consume: on the as-published grids elasticnet's optimum fell in a
# gap between adjacent points and ridge selected the grid floor in 204/204
# folds, so that table ranks tuning resolution as much as estimators.
# Parallelism is on the OUTER loop (inner GridSearchCV stays n_jobs=1, per the
# measured trap in adspan_ceiling_nested); ARM1_DENSE_WORKERS caps it so this
# can share the box with a GPU job.
step "$OUT/ceiling_nested_arm1_k75_dense.json" "Arm 1 estimators (dense grids)" \
     $PY -u -m nlp.adspan_ceiling_arm1_dense \
     --estimators ridge,elasticnet,elasticnet_stdscaler,gbm,rf,mlp \
     --workers "${ARM1_DENSE_WORKERS:-4}" --merge

step "$OUT/ceiling_nested_arm3_wordscores.json" "Arm 3 wordscores" \
     $PY -u -m nlp.adspan_ceiling_supervised --model wordscores

step "$OUT/ceiling_nested_arm3_transformer.json" "Arm 3 transformer fine-tune" \
     $PY -u -m nlp.adspan_ceiling_supervised --model transformer --seeds "${SEEDS:-0,1,2}"

echo "=== assembling tables $(date '+%H:%M:%S') ==="
$PY -u -m nlp.adspan_ceiling_report > "$OUT/CEILING_RESULTS.md" 2>&1
echo "[done] tables -> $OUT/CEILING_RESULTS.md"
echo "=== ceiling driver finished $(date '+%Y-%m-%d %H:%M:%S') ==="
