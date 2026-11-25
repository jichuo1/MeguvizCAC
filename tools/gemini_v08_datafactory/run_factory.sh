#!/usr/bin/env bash
# MeguvizCAC v0.8.0 External Gemini Data Factory — Linux/macOS/Colab 1-Click Launcher (Zero Antigravity Quota!)
set -euo pipefail

MODE="${MODE:-batch}"               # batch | realtime | pack | dry-run
WORK_DIR="${WORK_DIR:-v08-datafactory-work}"
LIMIT_A="${LIMIT_A:-60000}"
LIMIT_B="${LIMIT_B:-120000}"
LIMIT_C="${LIMIT_C:-30000}"
LIMIT_D="${LIMIT_D:-40000}"
BATCH_SIZE="${BATCH_SIZE:-25}"
WORKERS="${WORKERS:-8}"
QPS="${QPS:-6.0}"
TIER1_MODEL="${TIER1_MODEL:-gemini-2.5-flash}"
TIER2_MODEL="${TIER2_MODEL:-gemini-2.5-pro}"
SUBMIT_BATCH="${SUBMIT_BATCH:-0}"
POLL_BATCH="${POLL_BATCH:-0}"

echo "=== [1/4] Exporting Zero-PII Anti-Leakage Queues -> ${WORK_DIR} ==="
python3 tools/gemini_v08_datafactory/01_export_queue.py \
  --out-dir "${WORK_DIR}" \
  --limit-a "${LIMIT_A}" --limit-b "${LIMIT_B}" --limit-c "${LIMIT_C}" --limit-d "${LIMIT_D}" \
  --batch-size "${BATCH_SIZE}"

BATCH_FLAGS=()
[[ "${SUBMIT_BATCH}" == "1" ]] && BATCH_FLAGS+=(--submit-batch)
[[ "${POLL_BATCH}" == "1" ]] && BATCH_FLAGS+=(--poll-batch)

echo "=== [2/4] Running Tier-1 (${TIER1_MODEL}) [mode=${MODE}] ==="
python3 tools/gemini_v08_datafactory/02_run_gemini_factory.py \
  --work-dir "${WORK_DIR}" --mode "${MODE}" --tier tier1 --model "${TIER1_MODEL}" \
  --workers "${WORKERS}" --qps "${QPS}" "${BATCH_FLAGS[@]}"

if [[ "${MODE}" == "realtime" || "${MODE}" == "dry-run" || ( "${MODE}" == "batch" && "${POLL_BATCH}" == "1" ) ]]; then
  echo "=== [3/4] Running Tier-2 Escalation (${TIER2_MODEL}) ==="
  python3 tools/gemini_v08_datafactory/02_run_gemini_factory.py \
    --work-dir "${WORK_DIR}" --mode "${MODE}" --tier tier2 --model "${TIER2_MODEL}" \
    --workers "$(( WORKERS / 2 > 1 ? WORKERS / 2 : 1 ))" --qps "${QPS}" "${BATCH_FLAGS[@]}"

  echo "=== [4/4] Hidden Canary QA Gate + Anti-Leakage Re-Audit + Release Export ==="
  python3 tools/gemini_v08_datafactory/03_qc_and_merge.py --work-dir "${WORK_DIR}"
  echo "[DONE] Clean v0.8.0 outputs saved under ${WORK_DIR}/release/"
else
  echo "[PAUSED] Batch/Pack files generated under ${WORK_DIR}. Finish external run then execute 03_qc_and_merge.py."
fi
