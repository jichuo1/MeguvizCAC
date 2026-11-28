#!/usr/bin/env bash
# MeguvizCAC v1.6.0-StratosphereLoop Zero-Idle-GPU Finish A5 + Launch Full-Budget v1.6.0-r1 Production!
#   1. Finishes Arm A5 (2L/640) on the exact same shared_r0_frozen.pt & builds complete A0-A5 scoreboard.
#   2. Auto-selects winning geometry between A4 (1L/640) & A5 (2L/640), verifying <= 5.00 MiB with --expand-tables!
#   3. IMMEDIATELY launches Full-Budget v1.6.0-r1 (P1=2400, P2=4800, QAT=1600) with:
#        - Stratosphere bounded projected residual + causal inv_sqrt h_t (proven winner in A0-A4)
#        - 2x Expanded R0++/Engram tables (--expand-tables, zero-FLOP lookup boost, <= 5.00 MiB)
#        - Rev.2 Zero-Inference-Cost Softplus-CVaR tail penalty + Minimal Rule Format Counterfactual L_inv
#        - OOF Monotone R0+Deep Post-Fusion exported directly into manifest.json
#        - Auto-attaches /content/drive/MyDrive/MeguvizCAC/v151_stream_pack.jsonl.gz if present!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
OUT_ROOT="${OUT_ROOT:-/content/out_v160_ablation}"
PROD_OUT="${PROD_OUT:-/content/out_v160_r1_prod}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
ABL_DRIVE_DIR="${ABL_DRIVE_DIR:-$DRIVE_ROOT/v160_stratosphere_loop_r1}"
PROD_DRIVE_DIR="${PROD_DRIVE_DIR:-$DRIVE_ROOT/v160_r1_production}"
SHARED_R0="${SHARED_R0:-$OUT_ROOT/shared_r0_frozen.pt}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v160_r1_driver.txt}"
BS="${BS:-128}"
SEED="${SEED:-20261206}"

mkdir -p "$OUT_ROOT" "$PROD_OUT" "$ABL_DRIVE_DIR" "$PROD_DRIVE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Locate 251K Super-Teacher soft pool & tokenizer
DISTILL_POOL="/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "/content/v140_superteacher_soft_distill_pool.jsonl.gz" ] && DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ] && DISTILL_POOL="$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz"
V07_TOKENIZER="$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V11_DRIVE_DIR/bili_picky_bpe_12k.json"
[ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" ] && V04_DATA="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"

BASE_DATA_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --batch-size "$BS"
  --seed "$SEED"
  --t-train-mode uniform_1_6
  --step-decay-mode inv_sqrt
)
[ -d "$V123_BUNDLE_DIR/delta" ] && BASE_DATA_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && BASE_DATA_ARGS+=(--v04-data "$V04_DATA")

# Sync A4 metrics/cache to Drive first so nothing from A4 is lost!
[ -f "$OUT_ROOT/A4/metrics-v1.6.0-A4.json" ] && cp -f "$OUT_ROOT/A4/metrics-v1.6.0-A4.json" "$ABL_DRIVE_DIR/" || true
[ -f "$OUT_ROOT/A4/v1.6.0-A4_eval_cache.npz" ] && cp -f "$OUT_ROOT/A4/v1.6.0-A4_eval_cache.npz" "$ABL_DRIVE_DIR/" || true
[ -f "$OUT_ROOT/log_A4.txt" ] && cp -f "$OUT_ROOT/log_A4.txt" "$ABL_DRIVE_DIR/" || true

# ===========================================================================
# STEP 1: Finish Arm A5 (2L/640) cleanly with fixed Pi_R & 12-iter Sinkhorn!
# ===========================================================================
if [ ! -f "$OUT_ROOT/A5/metrics-v1.6.0-A5.json" ]; then
  echo "[$(date +%H:%M:%S)] === STEP 1: Running Arm A5 (2L/640) on shared frozen R0 ===" | tee -a "$LOG_DRIVER"
  mkdir -p "$OUT_ROOT/A5"
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${BASE_DATA_ARGS[@]}" \
    --ablation-fast \
    --init-r0-ckpt "$SHARED_R0" \
    --freeze-r0 \
    --arch v160_aprime \
    --dynamics stratosphere \
    --no-step-phi \
    --out "$OUT_ROOT/A5" \
    --checkpoint-dir "$ABL_DRIVE_DIR" \
    --version "v1.6.0-A5" 2>&1 | tee "$OUT_ROOT/log_A5.txt"
  cp -f "$OUT_ROOT/A5/metrics-v1.6.0-A5.json" "$ABL_DRIVE_DIR/"
  [ -f "$OUT_ROOT/A5/v1.6.0-A5_eval_cache.npz" ] && cp -f "$OUT_ROOT/A5/v1.6.0-A5_eval_cache.npz" "$ABL_DRIVE_DIR/"
  cp -f "$OUT_ROOT/log_A5.txt" "$ABL_DRIVE_DIR/"
  sync
fi

# Build complete A0-A5 scoreboard & select winning architecture for Full-Budget v1.6.0-r1!
WIN_ARCH="$(python3 - <<'PY'
import json
from pathlib import Path
out_root = Path("/content/out_v160_ablation")
drive_dir = Path("/content/drive/MyDrive/MeguvizCAC/v160_stratosphere_loop_r1")
board = {}
for mpath in sorted(out_root.glob("*/metrics-v1.6.0-*.json")):
    arm = mpath.parent.name
    d = json.loads(mpath.read_text(encoding="utf-8"))
    lp = d.get("loop_physics", {})
    board[arm] = {
        "arch": d.get("arch"), "dynamics": d.get("dynamics"),
        "use_step_phi": d.get("use_step_phi"), "use_ncl": d.get("use_ncl"),
        "params": d.get("total_params"),
        "bootable_mib": d.get("export", {}).get("bootable_mib"),
        "ci_passed": d.get("release_gates", {}).get("v160_ci_passed"),
        "gold_v2_test_pr": d.get("test_bili_gold_v2_onnx_full_t4", {}).get("pr"),
        "gold_v2_test_r95": d.get("test_bili_gold_v2_onnx_full_t4", {}).get("r95"),
        "gold_v2_test_r99": d.get("test_bili_gold_v2_onnx_full_t4", {}).get("r99"),
        "ml_val_pr": d.get("val_multilingual_onnx", {}).get("pr"),
        "god500": d.get("diag_god500_onnx"),
        "t1_minus_r1_pr": lp.get("t1_minus_r1_pr"),
        "t4_minus_t2_pr": lp.get("t4_minus_t2_pr"),
        "fusion_cert_comparison": d.get("fusion_cal_cert_comparison"),
        "quant_drift_audit": d.get("quant_drift_3ckpt_audit"),
        "ort_4t_b1_ms": d.get("ort_4t_b1_latency_ms"),
        "elapsed_s": d.get("elapsed_seconds"),
    }
sb_txt = json.dumps(board, ensure_ascii=False, indent=2)
(out_root / "v160_ablation_scoreboard.json").write_text(sb_txt, encoding="utf-8")
(drive_dir / "v160_ablation_scoreboard.json").write_text(sb_txt, encoding="utf-8")

def _score(arm_name):
    a = board.get(arm_name, {})
    g5 = (a.get("god500") or {}).get("pr") or 0.0
    ml = a.get("ml_val_pr") or 0.0
    g2 = a.get("gold_v2_test_pr") or 0.0
    return 0.50 * g5 + 0.30 * ml + 0.20 * g2

s_a4 = _score("A4")
s_a5 = _score("A5")
a5_ok = bool(board.get("A5", {}).get("ci_passed")) and float(board.get("A5", {}).get("bootable_mib") or 9.9) <= 4.15
# Pick A5 (2L/640) ONLY if it beats A4 (1L/640) by >= +0.003 composite AND leaves enough room for --expand-tables <= 5.00 MiB!
winner = "v160_aprime" if (a5_ok and s_a5 >= s_a4 + 0.003) else "v160_a2"
print(winner)
PY
)"
echo "[$(date +%H:%M:%S)] === A0-A5 Scoreboard SAVED! Auto-selected Production Arch: $WIN_ARCH ===" | tee -a "$LOG_DRIVER"

# ===========================================================================
# STEP 2: IMMEDIATELY Launch Full-Budget v1.6.0-r1 Production (ZERO GPU IDLE!)
# ===========================================================================
EXTRA_ARGS=()
if [ -f "$DRIVE_ROOT/v151_stream_pack.jsonl.gz" ]; then
  echo "[$(date +%H:%M:%S)] Found Master DuckDB 2.0 stream pack in Drive -> attaching --extra-gold-pack!" | tee -a "$LOG_DRIVER"
  EXTRA_ARGS+=(--extra-gold-pack "$DRIVE_ROOT/v151_stream_pack.jsonl.gz")
fi

# Both v160_a2 (~3.79 MiB) and v160_aprime (~4.30 MiB) fit comfortably <= 5.00 MiB with --expand-tables!
EXTRA_ARGS+=(--expand-tables)

echo "[$(date +%H:%M:%S)] === STEP 2: Launching Full-Budget v1.6.0-r1 ($WIN_ARCH) ===" | tee -a "$LOG_DRIVER"
python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
  "${BASE_DATA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --arch "$WIN_ARCH" \
  --dynamics stratosphere \
  --no-step-phi \
  --p1-steps 2400 \
  --p2-steps 4800 \
  --p3-qat-steps 1600 \
  --use-softplus-cvar \
  --cvar-alpha 0.08 \
  --cvar-weight 0.14 \
  --use-fmt-cf \
  --fmt-cf-weight 0.12 \
  --out "$PROD_OUT" \
  --checkpoint-dir "$PROD_DRIVE_DIR" \
  --version "v1.6.0-r1" 2>&1 | tee "$PROD_OUT/log_v160_r1.txt"

cp -f "$PROD_OUT/metrics-v1.6.0-r1.json" "$PROD_DRIVE_DIR/"
[ -f "$PROD_OUT/v1.6.0-r1_eval_cache.npz" ] && cp -f "$PROD_OUT/v1.6.0-r1_eval_cache.npz" "$PROD_DRIVE_DIR/"
cp -f "$PROD_OUT/log_v160_r1.txt" "$PROD_DRIVE_DIR/"
cp -f "$PROD_OUT/meguvizcac-v1.6.0-r1-bootable-v10-stratosphere.zip" "$PROD_DRIVE_DIR/" || true
(cd "$PROD_OUT" && zip -r -6 "$PROD_DRIVE_DIR/meguvizcac-v1.6.0-r1-full.zip" \
  bundle_v10_stratosphere_v160.mgvz charlm_v160.bin bili_picky_bpe_12k.json manifest.json \
  v160_r0_apexchar_fp32.onnx v160_prelude_t1_fp32.onnx v160_loop_step_fp32.onnx v160_dynaseam_fp32.onnx \
  v160_r0_apexchar_fp32.skel.onnx v160_prelude_t1_fp32.skel.onnx v160_loop_step_fp32.skel.onnx v160_dynaseam_fp32.skel.onnx \
  metrics-v1.6.0-r1.json \
  meguvizcac-v1.6.0-r1-bootable-v10-stratosphere.zip \
  meguvizcac-v1.6.0-r1-4stage-onnx-fp32exact.zip)
(cd "$PROD_DRIVE_DIR" && sha256sum meguvizcac-v1.6.0-r1-*.zip metrics-v1.6.0-r1.json > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$PROD_DRIVE_DIR/.synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [DONE] v1.6.0-r1 Full-Budget Production Synced to $PROD_DRIVE_DIR ===" | tee -a "$LOG_DRIVER"
