#!/usr/bin/env bash
# MeguvizCAC v1.6.2 Dual-Track Runner on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)
#
#   Track S   (`v1.6.2-S`):   Anti-Dilution R0++ (2200 steps) + Scratch Contractive-D + Discourse Ops (3600 steps)
#   Track PUC (`v1.6.2-PUC`): Strategy I Plasticity-Unlocked Warm-Start from v1.6.1-A (400-step PUC warmup + S&P, 3200 steps)
#   Both tracks share: TCBR-R, D02 Paraphrase L_inv, TriSeam-R Rust Kadane FocalRouter (longform-every=3),
#                      Prior-Matched EBA-H (pi_nat=0.05), and Turbo v3 (Rust + BF16 AMP + dynamic compile + FusedAux4).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/MeguvizCAC}"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
V161_DATA_DIR="${V161_DATA_DIR:-$DRIVE_ROOT/v161_data}"
V161_DRIVE_OUT="${V161_DRIVE_OUT:-$DRIVE_ROOT/v161_production}"
V162_DRIVE_OUT="${V162_DRIVE_OUT:-$DRIVE_ROOT/v162_production}"
WHEEL_DIR="${WHEEL_DIR:-$DRIVE_ROOT/wheels}"

WORK_ROOT="${WORK_ROOT:-/content/out_v162}"
OUT_S="$WORK_ROOT/v1.6.2-S"
OUT_PUC="$WORK_ROOT/v1.6.2-PUC"
SHARED_R0_V162="${SHARED_R0_V162:-$V162_DRIVE_OUT/shared_r0_v162_frozen.pt}"
SHARED_CHARLM="${SHARED_CHARLM:-$V161_DRIVE_OUT/shared_oof_charlm_v161.pkl}"
WARM_MASTER_V161A="${WARM_MASTER_V161A:-$V161_DRIVE_OUT/v1.6.1-A/v1.6.1-A_fp16_master.pt}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v162_driver.txt}"

BS="${BS:-256}"
EVAL_BS="${EVAL_BS:-512}"
SEED="${SEED:-20261220}"
RUN_TRACK_S="${RUN_TRACK_S:-1}"
RUN_TRACK_PUC="${RUN_TRACK_PUC:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
AMP_BF16="${AMP_BF16:-1}"
FUSE_AUX="${FUSE_AUX:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"

mkdir -p "$WORK_ROOT" "$OUT_S" "$OUT_PUC" "$V162_DRIVE_OUT/v1.6.2-S" "$V162_DRIVE_OUT/v1.6.2-PUC" "$WHEEL_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Content-hash-keyed Rust `_meguviz_rs` wheel cache (Cargo.toml + src/lib.rs -> sha8)
RUST_SHA8="$(cat "$REPO_DIR/meguviz_native/Cargo.toml" "$REPO_DIR/meguviz_native/src/lib.rs" | sha256sum | cut -c1-8)"
PY_TAG="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
RUST_CACHE_DIR="$WHEEL_DIR/$RUST_SHA8"
CUR_RS_SHA="$(cat /tmp/.meguviz_rs_sha8 2>/dev/null || true)"
if [ "$CUR_RS_SHA" != "$RUST_SHA8" ] || ! python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'extract_post_bpe_all_rs') and hasattr(_meguviz_rs.RustOOFCharLMSet, 'kadane_focal_window')" >/dev/null 2>&1; then
  mkdir -p "$RUST_CACHE_DIR"
  CACHED_WHL="$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl 2>/dev/null | head -n 1 || true)"
  if [ -n "$CACHED_WHL" ]; then
    echo "[$(date +%H:%M:%S)] Installing cached Rust wheel (sha8=$RUST_SHA8): $CACHED_WHL" | tee -a "$LOG_DRIVER"
    pip install -q --no-deps --force-reinstall "$CACHED_WHL"
  else
    echo "[$(date +%H:%M:%S)] Building Rust _meguviz_rs wheel (sha8=$RUST_SHA8) & caching to $RUST_CACHE_DIR..." | tee -a "$LOG_DRIVER"
    if ! command -v cargo >/dev/null 2>&1; then
      curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
      export PATH="$HOME/.cargo/bin:$PATH"
    fi
    pip install -q maturin
    TMP_WHL_DIR="$(mktemp -d)"
    maturin build --release --manifest-path "$REPO_DIR/meguviz_native/Cargo.toml" -i "$(which python3)" -o "$TMP_WHL_DIR"
    BUILT_WHL="$(ls -1t "$TMP_WHL_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl | head -n 1)"
    cp -f "$BUILT_WHL" "$RUST_CACHE_DIR/"
    rm -rf "$TMP_WHL_DIR"
    CACHED_WHL="$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl | head -n 1)"
    pip install -q --no-deps --force-reinstall "$CACHED_WHL"
  fi
  echo "$RUST_SHA8" > /tmp/.meguviz_rs_sha8
fi
python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'extract_post_bpe_all_rs') and hasattr(_meguviz_rs.RustOOFCharLMSet, 'kadane_focal_window'); print('[Rust-OK sha8=$RUST_SHA8]', _meguviz_rs.__file__)"

# 2. Ensure local fast SSD copies of corpus & rescored pack
FRESH_V123_ZIP="$V161_DATA_DIR/v123_all_gold_sibling_and_delta_bundle.zip"
if [ -f "$FRESH_V123_ZIP" ]; then
  ZIP_SHA="$(sha256sum "$FRESH_V123_ZIP" | awk '{print $1}')"
  CUR_SHA="$(cat "$V123_BUNDLE_DIR/.bundle_sha256" 2>/dev/null || true)"
  if [ "$ZIP_SHA" != "$CUR_SHA" ]; then
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    unzip -q -o "$FRESH_V123_ZIP" -d "$V123_BUNDLE_DIR"
    echo "$ZIP_SHA" > "$V123_BUNDLE_DIR/.bundle_sha256"
  fi
fi

DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ] && cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "$DISTILL_POOL"
RESCORED_LOCAL="/content/v161_duckdb2_rescored.jsonl.gz"
[ ! -f "$RESCORED_LOCAL" ] && [ -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" ] && cp -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" "$RESCORED_LOCAL"
V07_TOKENIZER="$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V11_DRIVE_DIR/bili_picky_bpe_12k.json"
[ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" ] && V04_DATA="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"
CAL_CERT_PACK="$V161_DATA_DIR/v161_duckdb2_cal_cert.jsonl.gz"

BASE_V162_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --batch-size "$BS"
  --eval-batch-size "$EVAL_BS"
  --seed "$SEED"
  --t-train-mode uniform_1_6
  --step-decay-mode inv_sqrt
  --arch v160_a2
  --dynamics contractive_d
  --use-discourse-ops
  --sat-reg-weight 0.02
  --no-step-phi
  --expand-tables
  --lardo-w8-core
  --r0-antidilution-mix
  --use-tcbr-r
  --cvar-alpha 0.08
  --cvar-weight 0.07
  --recall-floor-weight 0.14
  --use-fmt-cf
  --fmt-cf-weight 0.12
  --use-d02-paraphrase-cf
  --use-promo-flip-cf
  --promo-flip-weight 0.06
  --use-cc-vrex
  --cc-vrex-weight 0.10
  --longform-every 3
  --use-kadane-focal-router
  --use-eba-h
  --eba-pi-nat 0.05
  --oof-cal-folds 5
  --auto-ship-gate
  --charlm-cache "$SHARED_CHARLM"
  --prefetch-workers "$PREFETCH_WORKERS"
)
[ -d "$V123_BUNDLE_DIR/delta" ] && BASE_V162_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && BASE_V162_ARGS+=(--v04-data "$V04_DATA")
[ -f "$RESCORED_LOCAL" ] && BASE_V162_ARGS+=(--extra-gold-pack "$RESCORED_LOCAL")
[ -f "$CAL_CERT_PACK" ] && BASE_V162_ARGS+=(--extra-cal-cert-pack "$CAL_CERT_PACK")
[ "$USE_COMPILE" = "1" ] && BASE_V162_ARGS+=(--compile --compile-mode "$COMPILE_MODE")
[ "$COMPILE_DYNAMIC" = "1" ] && BASE_V162_ARGS+=(--compile-dynamic) || BASE_V162_ARGS+=(--no-compile-dynamic)
[ "$AMP_BF16" = "1" ] && BASE_V162_ARGS+=(--amp-bf16) || BASE_V162_ARGS+=(--no-amp-bf16)
[ "$FUSE_AUX" = "1" ] && BASE_V162_ARGS+=(--fuse-aux) || BASE_V162_ARGS+=(--no-fuse-aux)

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
# TRACK S (`v1.6.2-S`): Anti-Dilution R0++ (2200 steps) + Scratch Deep (3600 steps)
# ===========================================================================
if [ "$RUN_TRACK_S" = "1" ] && [ ! -f "$OUT_S/metrics-v1.6.2-S.json" ] && [ ! -f "$V162_DRIVE_OUT/v1.6.2-S/metrics-v1.6.2-S.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.6.2-S (Anti-Dilution R0++ + Scratch Contractive-D + Discourse Ops + Turbo v3) ===" | tee -a "$LOG_DRIVER"
  R0_S_ARGS=(--p1-steps 2200 --p1-use-cf --save-r0-ckpt "$SHARED_R0_V162" --freeze-r0)
  if [ -f "$SHARED_R0_V162" ]; then
    R0_S_ARGS=(--init-r0-ckpt "$SHARED_R0_V162" --freeze-r0)
  fi
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${BASE_V162_ARGS[@]}" \
    "${R0_S_ARGS[@]}" \
    --p2-steps 3600 \
    --p2-lr 1.1e-3 \
    --p3-qat-steps 300 \
    --qat-lr 4e-5 \
    --out "$OUT_S" \
    --checkpoint-dir "$V162_DRIVE_OUT/v1.6.2-S" \
    --version "v1.6.2-S" 2>&1 | tee "$OUT_S/log_v1.6.2-S.txt"
  _sync_release "v1.6.2-S" "$OUT_S" "$V162_DRIVE_OUT/v1.6.2-S"
fi

# ===========================================================================
# TRACK PUC (`v1.6.2-PUC`): Strategy I Plasticity-Unlocked Warm-Start from v1.6.1-A
# ===========================================================================
if [ "$RUN_TRACK_PUC" = "1" ] && [ ! -f "$OUT_PUC/metrics-v1.6.2-PUC.json" ] && [ ! -f "$V162_DRIVE_OUT/v1.6.2-PUC/metrics-v1.6.2-PUC.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.6.2-PUC (Strategy I Warm-Start + 400-step PUC + S&P + Turbo v3) ===" | tee -a "$LOG_DRIVER"
  R0_PUC_ARGS=(--p1-steps 2200 --p1-use-cf --save-r0-ckpt "$SHARED_R0_V162" --freeze-r0)
  if [ -f "$SHARED_R0_V162" ]; then
    R0_PUC_ARGS=(--init-r0-ckpt "$SHARED_R0_V162" --freeze-r0)
  fi
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${BASE_V162_ARGS[@]}" \
    "${R0_PUC_ARGS[@]}" \
    --init-from-fp16 "$WARM_MASTER_V161A" \
    --wise-anchor-fp16 "$WARM_MASTER_V161A" \
    --p2-steps 3200 \
    --p2-lr 6.5e-4 \
    --puc-warmup-steps 400 \
    --puc-shrink 0.95 \
    --puc-noise 0.015 \
    --p3-qat-steps 300 \
    --qat-lr 4e-5 \
    --out "$OUT_PUC" \
    --checkpoint-dir "$V162_DRIVE_OUT/v1.6.2-PUC" \
    --version "v1.6.2-PUC" 2>&1 | tee "$OUT_PUC/log_v1.6.2-PUC.txt"
  _sync_release "v1.6.2-PUC" "$OUT_PUC" "$V162_DRIVE_OUT/v1.6.2-PUC"
fi

# ===========================================================================
# SCOREBOARD: Compare v1.6.2-S vs v1.6.2-PUC
# ===========================================================================
python3 - <<'PY'
import json
from pathlib import Path

work = Path("/content/out_v162")
drv = Path("/content/drive/MyDrive/MeguvizCAC/v162_production")
board = {}
for tag in ("v1.6.2-S", "v1.6.2-PUC"):
    mp = work / tag / f"metrics-{tag}.json"
    if not mp.is_file():
        mp = drv / tag / f"metrics-{tag}.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        lp = d.get("loop_physics", {})
        board[tag] = {
            "dynamics": d.get("dynamics"),
            "use_discourse_ops": d.get("use_discourse_ops"),
            "bootable_mib": d.get("export", {}).get("bootable_mib"),
            "ci_passed": d.get("release_gates", {}).get("v160_ci_passed"),
            "auto_ship_winner": d.get("quant_drift_3ckpt_audit", {}).get("auto_ship_gate", {}).get("winner"),
            "gold_v2_test_cond": d.get("test_bili_gold_v2_onnx_cond"),
            "val_multilingual": d.get("val_multilingual_onnx"),
            "diag_god500": d.get("diag_god500_onnx"),
            "r0_ltt_policy": d.get("rung0_ai_ltt_policy"),
            "t4_minus_t2_pr": lp.get("t4_minus_t2_pr"),
            "proj_sat_rate": lp.get("proj_sat_rate_t1_t6"),
            "period2_oscillation": lp.get("period2_oscillation"),
            "order_preserving_3seg_probe": d.get("order_preserving_3seg_probe"),
            "ort_4t_b1_ms": d.get("ort_4t_b1_latency_ms"),
            "elapsed_s": d.get("elapsed_seconds"),
            "phase_timings_sec": d.get("phase_timings_sec"),
        }
txt = json.dumps(board, ensure_ascii=False, indent=2)
(work / "v162_comparison.json").write_text(txt, encoding="utf-8")
(drv / "v162_comparison.json").write_text(txt, encoding="utf-8")
print("=== v1.6.2-S vs v1.6.2-PUC SCOREBOARD ===")
print(txt)
PY
cp -f "$LOG_DRIVER" "$V162_DRIVE_OUT/" || true
touch "$V162_DRIVE_OUT/.all_v162_synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [ALL DONE] v1.6.2 Synced to $V162_DRIVE_OUT ===" | tee -a "$LOG_DRIVER"
