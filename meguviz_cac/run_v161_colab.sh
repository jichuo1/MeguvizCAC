#!/usr/bin/env bash
# MeguvizCAC v1.6.1 Metamorphosis Dual-Track Runner on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)
#
#   Stage 1 (~5-9 min): Super-Teacher soft-logit rescoring of final clean DuckDB 2.0 pack (cached to Drive!)
#   Stage 2 (~22 min) : Track A (`v1.6.1-A`) — Zero-overhead full-blooded warm-start from v1.6.0-r1 FP16 master
#                       + 4000-step R0++ (shared frozen for Track B) + LARDO W8 core + 3-Cand Auto Ship Gate + 5-fold OOF R0 LTT
#   Stage 3 (~38 min) : Track B (`v1.6.1-B`) — True Metamorphosis Architecture Arm on identical shared frozen R0++
#                       + Review 01 R2 Direct Discrete Contractive Kernel D (kills 100% sat & 0.12 period-2 bounce)
#                       + Review 04 Rev. 2 +0.33ms Discourse Attribution Suite (SelectiveOrder3CumSum + CheapD08PositionMoments + Quote FSM)
#                       + Review 03 Rev. 2 LARDO W8 Core (<= 4.85 MiB pure MGVZ10 W4/W8) + 3-Cand Auto Ship Gate + 5-fold OOF R0 LTT
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/MeguvizCAC}"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
V160_PROD_DIR="${V160_PROD_DIR:-$DRIVE_ROOT/v160_r1_production}"
V161_DATA_DIR="${V161_DATA_DIR:-$DRIVE_ROOT/v161_data}"
V161_DRIVE_OUT="${V161_DRIVE_OUT:-$DRIVE_ROOT/v161_production}"

WORK_ROOT="${WORK_ROOT:-/content/out_v161}"
OUT_A="$WORK_ROOT/v1.6.1-A"
OUT_B="$WORK_ROOT/v1.6.1-B"
SHARED_R0="${SHARED_R0:-$V161_DRIVE_OUT/shared_r0_v161_frozen.pt}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v161_driver.txt}"
BS="${BS:-128}"
SEED="${SEED:-20261215}"
SKIP_RESCORE="${SKIP_RESCORE:-0}"
RUN_TRACK_A="${RUN_TRACK_A:-1}"
RUN_TRACK_B="${RUN_TRACK_B:-1}"

mkdir -p "$WORK_ROOT" "$OUT_A" "$OUT_B" "$V161_DATA_DIR" "$V161_DRIVE_OUT/v1.6.1-A" "$V161_DRIVE_OUT/v1.6.1-B"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Auto-refresh /content/v123_bundle if the latest post-c408 v123_all_gold_sibling_and_delta_bundle.zip is uploaded in v161_data!
FRESH_V123_ZIP="$V161_DATA_DIR/v123_all_gold_sibling_and_delta_bundle.zip"
if [ -f "$FRESH_V123_ZIP" ]; then
  ZIP_SHA="$(sha256sum "$FRESH_V123_ZIP" | awk '{print $1}')"
  CUR_SHA="$(cat "$V123_BUNDLE_DIR/.bundle_sha256" 2>/dev/null || true)"
  if [ "$ZIP_SHA" != "$CUR_SHA" ]; then
    echo "[$(date +%H:%M:%S)] Extracting FRESH v123 bundle ($FRESH_V123_ZIP, sha=${ZIP_SHA:0:12}) -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    rm -rf "$V123_BUNDLE_DIR"
    mkdir -p "$V123_BUNDLE_DIR"
    unzip -q -o "$FRESH_V123_ZIP" -d "$V123_BUNDLE_DIR"
    echo "$ZIP_SHA" > "$V123_BUNDLE_DIR/.bundle_sha256"
  fi
fi

# Locate base 251K Super-Teacher soft pool, tokenizer & v04 corpus
DISTILL_POOL="/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "/content/v140_superteacher_soft_distill_pool.jsonl.gz" ] && DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
if [ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ]; then
  cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "/content/v140_superteacher_soft_distill_pool.jsonl.gz"
  DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
fi
V07_TOKENIZER="$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V11_DRIVE_DIR/bili_picky_bpe_12k.json"
[ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" ] && V04_DATA="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"

# Locate final Master DuckDB 2.0 clean train pack & disjoint calibration holdout
RAW_DUCKDB_PACK="$V161_DATA_DIR/v161_duckdb2_train_pack.jsonl.gz"
[ ! -f "$RAW_DUCKDB_PACK" ] && [ -f "$DRIVE_ROOT/v151_stream_pack.jsonl.gz" ] && RAW_DUCKDB_PACK="$DRIVE_ROOT/v151_stream_pack.jsonl.gz"
CAL_CERT_PACK="$V161_DATA_DIR/v161_duckdb2_cal_cert.jsonl.gz"
RESCORED_PACK="$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz"

echo "[$(date +%H:%M:%S)] === MeguvizCAC v1.6.1 Metamorphosis Launch ===" | tee -a "$LOG_DRIVER"
echo "  DISTILL_POOL=$DISTILL_POOL" | tee -a "$LOG_DRIVER"
echo "  RAW_DUCKDB_PACK=$RAW_DUCKDB_PACK" | tee -a "$LOG_DRIVER"
echo "  CAL_CERT_PACK=$CAL_CERT_PACK" | tee -a "$LOG_DRIVER"

# ===========================================================================
# STAGE 1: Super-Teacher Soft-Logit Rescoring of Final DuckDB 2.0 Clean Pack
# ===========================================================================
EXTRA_PACK_USE="$RAW_DUCKDB_PACK"
if [ -f "$RAW_DUCKDB_PACK" ] && [ "$SKIP_RESCORE" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === STAGE 1: Super-Teacher Soft Rescoring ($RAW_DUCKDB_PACK) ===" | tee -a "$LOG_DRIVER"
  python3 -u -m meguviz_cac.rescore_duckdb2_superteacher \
    --input-pack "$RAW_DUCKDB_PACK" \
    --output-pack "$RESCORED_PACK" \
    --st-ckpt-dir "$ST_DRIVE_DIR" \
    --base-distill-pool "$DISTILL_POOL" \
    --cache-dir "$V161_DATA_DIR/rescore_cache" 2>&1 | tee "$WORK_ROOT/log_stage1_rescore.txt"
  cp -f "$WORK_ROOT/log_stage1_rescore.txt" "$V161_DRIVE_OUT/" || true
  if [ -f "$RESCORED_PACK" ]; then
    cp -f "$RESCORED_PACK" "/content/v161_duckdb2_rescored.jsonl.gz"
    EXTRA_PACK_USE="/content/v161_duckdb2_rescored.jsonl.gz"
  fi
elif [ -f "$RESCORED_PACK" ]; then
  cp -f "$RESCORED_PACK" "/content/v161_duckdb2_rescored.jsonl.gz"
  EXTRA_PACK_USE="/content/v161_duckdb2_rescored.jsonl.gz"
fi

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
  --arch v160_a2
  --no-step-phi
  --expand-tables
  --lardo-w8-core
  --use-softplus-cvar
  --cvar-alpha 0.10
  --cvar-weight 0.05
  --recall-floor-weight 0.12
  --use-fmt-cf
  --fmt-cf-weight 0.10
  --use-promo-flip-cf
  --promo-flip-weight 0.06
  --use-cc-vrex
  --cc-vrex-weight 0.08
  --oof-cal-folds 5
  --auto-ship-gate
)
[ -d "$V123_BUNDLE_DIR/delta" ] && BASE_DATA_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && BASE_DATA_ARGS+=(--v04-data "$V04_DATA")
[ -f "$EXTRA_PACK_USE" ] && BASE_DATA_ARGS+=(--extra-gold-pack "$EXTRA_PACK_USE")
[ -f "$CAL_CERT_PACK" ] && BASE_DATA_ARGS+=(--extra-cal-cert-pack "$CAL_CERT_PACK")

_sync_release() {
  local ver="$1"
  local src="$2"
  local dst="$3"
  mkdir -p "$dst"
  cp -f "$src/metrics-${ver}.json" "$dst/"
  [ -f "$src/${ver}_eval_cache.npz" ] && cp -f "$src/${ver}_eval_cache.npz" "$dst/"
  [ -f "$src/log_${ver}.txt" ] && cp -f "$src/log_${ver}.txt" "$dst/"
  cp -f "$src/meguvizcac-${ver}-bootable-v10-stratosphere.zip" "$dst/" || true
  (cd "$src" && zip -r -6 "$dst/meguvizcac-${ver}-full.zip" \
    bundle_v10_stratosphere_v160.mgvz charlm_v160.bin bili_picky_bpe_12k.json manifest.json \
    v160_r0_apexchar_fp32.onnx v160_prelude_t1_fp32.onnx v160_loop_step_fp32.onnx v160_dynaseam_fp32.onnx \
    v160_r0_apexchar_fp32.skel.onnx v160_prelude_t1_fp32.skel.onnx v160_loop_step_fp32.skel.onnx v160_dynaseam_fp32.skel.onnx \
    "metrics-${ver}.json" \
    "meguvizcac-${ver}-bootable-v10-stratosphere.zip" \
    "meguvizcac-${ver}-4stage-onnx-fp32exact.zip")
  (cd "$dst" && sha256sum "meguvizcac-${ver}"-*.zip "metrics-${ver}.json" > SHA256SUMS.txt && cat SHA256SUMS.txt)
  touch "$dst/.synced_ok" && sync
}

# ===========================================================================
# STAGE 2: Track A (`v1.6.1-A`) — Zero-Overhead Full-Blooded Warm-Start + Shared 4000-Step R0++
# ===========================================================================
if [ "$RUN_TRACK_A" = "1" ] && [ ! -f "$OUT_A/metrics-v1.6.1-A.json" ] && [ ! -f "$V161_DRIVE_OUT/v1.6.1-A/metrics-v1.6.1-A.json" ]; then
  echo "[$(date +%H:%M:%S)] === STAGE 2: Launching Track A (v1.6.1-A Zero-Overhead Warm-Start) ===" | tee -a "$LOG_DRIVER"
  WARM_ARGS=()
  V160_MASTER="$V160_PROD_DIR/v1.6.0-r1_fp16_master.pt"
  if [ -f "$V160_MASTER" ]; then
    WARM_ARGS+=(--init-from-fp16 "$V160_MASTER" --wise-anchor-fp16 "$V160_MASTER")
  fi
  R0_A_ARGS=(--p1-steps 4000 --p1-use-cf --save-r0-ckpt "$SHARED_R0" --freeze-r0)
  if [ -f "$SHARED_R0" ]; then
    R0_A_ARGS=(--init-r0-ckpt "$SHARED_R0" --freeze-r0)
  fi
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${BASE_DATA_ARGS[@]}" \
    "${WARM_ARGS[@]}" \
    "${R0_A_ARGS[@]}" \
    --dynamics stratosphere \
    --p2-steps 4200 \
    --p2-lr 5e-4 \
    --p3-qat-steps 300 \
    --qat-lr 5e-5 \
    --out "$OUT_A" \
    --checkpoint-dir "$V161_DRIVE_OUT/v1.6.1-A" \
    --version "v1.6.1-A" 2>&1 | tee "$OUT_A/log_v1.6.1-A.txt"
  _sync_release "v1.6.1-A" "$OUT_A" "$V161_DRIVE_OUT/v1.6.1-A"
fi

# ===========================================================================
# STAGE 3: Track B (`v1.6.1-B`) — True Metamorphosis (Contractive Kernel D + 0.33ms Discourse Ops!)
# ===========================================================================
if [ "$RUN_TRACK_B" = "1" ] && [ ! -f "$OUT_B/metrics-v1.6.1-B.json" ] && [ ! -f "$V161_DRIVE_OUT/v1.6.1-B/metrics-v1.6.1-B.json" ]; then
  echo "[$(date +%H:%M:%S)] === STAGE 3: Launching Track B (v1.6.1-B Contractive-D + Discourse Ops) ===" | tee -a "$LOG_DRIVER"
  R0_ARGS=()
  if [ -f "$SHARED_R0" ]; then
    R0_ARGS+=(--init-r0-ckpt "$SHARED_R0" --freeze-r0)
  elif [ -f "$V161_DRIVE_OUT/shared_r0_v161_frozen.pt" ]; then
    R0_ARGS+=(--init-r0-ckpt "$V161_DRIVE_OUT/shared_r0_v161_frozen.pt" --freeze-r0)
  else
    R0_ARGS+=(--p1-steps 4000 --p1-use-cf --freeze-r0)
  fi
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${BASE_DATA_ARGS[@]}" \
    "${R0_ARGS[@]}" \
    --dynamics contractive_d \
    --use-discourse-ops \
    --sat-reg-weight 0.02 \
    --p2-steps 5400 \
    --p2-lr 1.1e-3 \
    --p3-qat-steps 300 \
    --qat-lr 5e-5 \
    --out "$OUT_B" \
    --checkpoint-dir "$V161_DRIVE_OUT/v1.6.1-B" \
    --version "v1.6.1-B" 2>&1 | tee "$OUT_B/log_v1.6.1-B.txt"
  _sync_release "v1.6.1-B" "$OUT_B" "$V161_DRIVE_OUT/v1.6.1-B"
fi

# ===========================================================================
# STAGE 4: Build Combined v1.6.1-A vs v1.6.1-B Scoreboard & Final Sync
# ===========================================================================
python3 - <<'PY'
import json
from pathlib import Path

work = Path("/content/out_v161")
drv = Path("/content/drive/MyDrive/MeguvizCAC/v161_production")
board = {}
for tag in ("v1.6.1-A", "v1.6.1-B"):
    mp = work / tag / f"metrics-{tag}.json"
    if not mp.is_file():
        mp = drv / tag / f"metrics-{tag}.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        lp = d.get("loop_physics", {})
        board[tag] = {
            "dynamics": d.get("dynamics"),
            "use_discourse_ops": d.get("use_discourse_ops"),
            "lardo_w8_core": d.get("lardo_w8_core"),
            "bootable_mib": d.get("export", {}).get("bootable_mib"),
            "ci_passed": d.get("release_gates", {}).get("v160_ci_passed"),
            "auto_ship_winner": d.get("quant_drift_3ckpt_audit", {}).get("auto_ship_gate", {}).get("winner"),
            "gold_v2_test_cond": d.get("test_bili_gold_v2_onnx_cond"),
            "gold_v2_test_t4": d.get("test_bili_gold_v2_onnx_full_t4"),
            "val_multilingual": d.get("val_multilingual_onnx"),
            "diag_god500": d.get("diag_god500_onnx"),
            "r0_ltt_policy": d.get("rung0_ai_ltt_policy"),
            "t4_minus_t2_pr": lp.get("t4_minus_t2_pr"),
            "proj_sat_rate": lp.get("proj_sat_rate_t1_t6"),
            "period2_oscillation": lp.get("period2_oscillation"),
            "order_preserving_3seg_probe": d.get("order_preserving_3seg_probe"),
            "ort_4t_b1_ms": d.get("ort_4t_b1_latency_ms"),
            "elapsed_s": d.get("elapsed_seconds"),
        }
txt = json.dumps(board, ensure_ascii=False, indent=2)
(work / "v161_ab_comparison.json").write_text(txt, encoding="utf-8")
(drv / "v161_ab_comparison.json").write_text(txt, encoding="utf-8")
print("=== v1.6.1-A vs v1.6.1-B SCOREBOARD ===")
print(txt)
PY
cp -f "$LOG_DRIVER" "$V161_DRIVE_OUT/" || true
touch "$V161_DRIVE_OUT/.all_v161_synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [ALL DONE] v1.6.1-A & v1.6.1-B Synced to $V161_DRIVE_OUT ===" | tee -a "$LOG_DRIVER"
