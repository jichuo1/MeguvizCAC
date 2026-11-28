#!/usr/bin/env bash
# MeguvizCAC v1.7.1 Post-Training Suite Runner on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)
#
#   Track DreamCore (`v1.7.1-DreamCore`): Warm-starts from v1.7.0-DreamCore + v1.6.1 4000-step Natural R0++
#   Track M1        (`v1.7.1-M1`):        Warm-starts from v1.7.0-M1 (DualHopfield B2a) + v1.6.1 4000-step Natural R0++
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
V170_DRIVE_OUT="${V170_DRIVE_OUT:-$DRIVE_ROOT/v170_production}"
V171_DRIVE_OUT="${V171_DRIVE_OUT:-$DRIVE_ROOT/v171_production}"
WHEEL_DIR="${WHEEL_DIR:-$DRIVE_ROOT/wheels}"

WORK_ROOT="${WORK_ROOT:-/content/out_v171}"
OUT_DC="$WORK_ROOT/v1.7.1-DreamCore"
OUT_M1="$WORK_ROOT/v1.7.1-M1"
# Use v1.6.1's proven 4000-step Natural-Stream R0++ anchor (52/52 bit-compatible tensors; fallback to v1.7.0 R0)!
SHARED_R0_V171="${SHARED_R0_V171:-$V161_DRIVE_OUT/shared_r0_v161_frozen.pt}"
if [ ! -f "$SHARED_R0_V171" ] && [ -f "$V170_DRIVE_OUT/shared_r0_v170_natural.pt" ]; then
  SHARED_R0_V171="$V170_DRIVE_OUT/shared_r0_v170_natural.pt"
fi
SHARED_CHARLM="${SHARED_CHARLM:-$V161_DRIVE_OUT/shared_oof_charlm_v161.pkl}"
PARENT_DC="${PARENT_DC:-$V170_DRIVE_OUT/v1.7.0-DreamCore/v1.7.0-DreamCore_fp16_master.pt}"
PARENT_M1="${PARENT_M1:-$V170_DRIVE_OUT/v1.7.0-M1/v1.7.0-M1_fp16_master.pt}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v171_driver.txt}"

BS="${BS:-256}"
EVAL_BS="${EVAL_BS:-512}"
SEED="${SEED:-20261230}"
RUN_TRACK_DC="${RUN_TRACK_DC:-1}"
RUN_TRACK_M1="${RUN_TRACK_M1:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
AMP_BF16="${AMP_BF16:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"

mkdir -p "$WORK_ROOT" "$OUT_DC" "$OUT_M1" "$V171_DRIVE_OUT/v1.7.1-DreamCore" "$V171_DRIVE_OUT/v1.7.1-M1" "$WHEEL_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 0. Ensure required Python inference/export packages exist on a fresh Colab VM
if ! python3 -c "import onnx, onnxruntime, onnxscript, tokenizers, scipy, sklearn" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] Installing missing Python deps..." | tee -a "$LOG_DRIVER"
  pip install -q onnx onnxruntime onnxscript tokenizers
fi

# 1. Content-hash-keyed Rust `_meguviz_rs` v4 wheel cache
RUST_SHA8="$(cat "$REPO_DIR/meguviz_native/Cargo.toml" "$REPO_DIR/meguviz_native/src/lib.rs" | sha256sum | cut -c1-8)"
PY_TAG="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
RUST_CACHE_DIR="$WHEEL_DIR/$RUST_SHA8"
CUR_RS_SHA="$(cat /tmp/.meguviz_rs_sha8 2>/dev/null || true)"
if [ "$CUR_RS_SHA" != "$RUST_SHA8" ] || ! python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'extract_v170_uisr_and_surf_batch_rs') and hasattr(_meguviz_rs.RustOOFCharLMSet, 'extract_v170_ph16_batch')" >/dev/null 2>&1; then
  mkdir -p "$RUST_CACHE_DIR"
  CACHED_WHL="$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl 2>/dev/null | head -n 1 || true)"
  if [ -n "$CACHED_WHL" ]; then
    pip install -q --no-deps --force-reinstall "$CACHED_WHL"
  else
    if ! command -v cargo >/dev/null 2>&1; then
      curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
      export PATH="$HOME/.cargo/bin:$PATH"
    fi
    pip install -q maturin
    TMP_WHL_DIR="$(mktemp -d)"
    maturin build --release --manifest-path "$REPO_DIR/meguviz_native/Cargo.toml" -i "$(which python3)" -o "$TMP_WHL_DIR"
    cp -f "$(ls -1t "$TMP_WHL_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl | head -n 1)" "$RUST_CACHE_DIR/"
    rm -rf "$TMP_WHL_DIR"
    pip install -q --no-deps --force-reinstall "$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl | head -n 1)"
  fi
  echo "$RUST_SHA8" > /tmp/.meguviz_rs_sha8
fi

# 2. Ensure local fast SSD copies of all corpora
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave2_longform_gold_1200.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    mkdir -p "$V11_DATA_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  elif [ -d "$V11_DRIVE_DIR/v11_data" ]; then
    mkdir -p "$V11_DATA_DIR"
    cp -rn "$V11_DRIVE_DIR/v11_data/." "$V11_DATA_DIR/" || true
  fi
fi

FRESH_V123_ZIP=""
for zcand in \
  "$V161_DATA_DIR/v123_all_gold_sibling_and_delta_bundle.zip" \
  "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip" \
  "/content/v123_all_gold_sibling_and_delta_bundle.zip"; do
  [ -f "$zcand" ] && FRESH_V123_ZIP="$zcand" && break
done
if [ -n "$FRESH_V123_ZIP" ]; then
  ZIP_SHA="$(sha256sum "$FRESH_V123_ZIP" | awk '{print $1}')"
  CUR_SHA="$(cat "$V123_BUNDLE_DIR/.bundle_sha256" 2>/dev/null || true)"
  if [ "$ZIP_SHA" != "$CUR_SHA" ] || [ ! -d "$V123_BUNDLE_DIR/sibling" ]; then
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$FRESH_V123_ZIP').extractall('$V123_BUNDLE_DIR')"
    echo "$ZIP_SHA" > "$V123_BUNDLE_DIR/.bundle_sha256"
  fi
fi

DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ] && cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "$DISTILL_POOL"
RESCORED_LOCAL="/content/v161_duckdb2_rescored.jsonl.gz"
[ ! -f "$RESCORED_LOCAL" ] && [ -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" ] && cp -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" "$RESCORED_LOCAL"
V07_TOKENIZER="$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V11_DRIVE_DIR/bili_picky_bpe_12k.json"
if [ ! -f "$V04_DATA" ]; then
  for vcand in "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" "$DRIVE_ROOT/data/v04-merged-1m4-r3.jsonl.gz"; do
    if [ -f "$vcand" ]; then
      mkdir -p "$(dirname "$V04_DATA")" && cp -f "$vcand" "$V04_DATA" && break
    fi
  done
fi
CAL_CERT_PACK="$V161_DATA_DIR/v161_duckdb2_cal_cert.jsonl.gz"

echo "[$(date +%H:%M:%S)] === v1.7.1 Pre-flight: R0_ANCHOR=$SHARED_R0_V171 | DC=$PARENT_DC | M1=$PARENT_M1 ===" | tee -a "$LOG_DRIVER"

BASE_V171_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --arch v170_m1
  --init-r0-ckpt "$SHARED_R0_V171"
  --batch-size "$BS"
  --eval-batch-size "$EVAL_BS"
  --stage-a-steps 1600
  --stage-b-steps 800
  --apd-steps 400
  --p3-qat-steps 300
  --lr-a 6.0e-4
  --lr-b 4.5e-4
  --puc-shrink 0.96
  --puc-noise 0.012
  --r-echo-cap 0.50
  --drv-fro-cap 6.0
  --seed "$SEED"
  --fit-titanguard
  --use-eba-h
  --eba-pi-nat 0.05
  --iut-alpha-max 0.04
  --auto-ship-gate
  --charlm-cache "$SHARED_CHARLM"
  --prefetch-workers "$PREFETCH_WORKERS"
)
[ -d "$V123_BUNDLE_DIR/delta" ] && BASE_V171_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && BASE_V171_ARGS+=(--v04-data "$V04_DATA")
[ -f "$RESCORED_LOCAL" ] && BASE_V171_ARGS+=(--extra-gold-pack "$RESCORED_LOCAL")
[ -f "$CAL_CERT_PACK" ] && BASE_V171_ARGS+=(--extra-cal-cert-pack "$CAL_CERT_PACK")
[ "$USE_COMPILE" = "1" ] && BASE_V171_ARGS+=(--compile --compile-mode "$COMPILE_MODE")
[ "$COMPILE_DYNAMIC" = "1" ] && BASE_V171_ARGS+=(--compile-dynamic) || BASE_V171_ARGS+=(--no-compile-dynamic)
[ "$AMP_BF16" = "1" ] && BASE_V171_ARGS+=(--amp-bf16) || BASE_V171_ARGS+=(--no-amp-bf16)

_sync_release() {
  local ver="$1"
  local src="$2"
  local dst="$3"
  mkdir -p "$dst"
  cp -f "$src/metrics-${ver}.json" "$dst/"
  [ -f "$src/${ver}_fp16_master.pt" ] && cp -f "$src/${ver}_fp16_master.pt" "$dst/"
  [ -f "$src/log_${ver}.txt" ] && cp -f "$src/log_${ver}.txt" "$dst/"
  cp -f "$src/meguvizcac-${ver}-bootable-v10-mesosphere.zip" "$dst/" 2>/dev/null || true
  (cd "$src" && zip -r -6 "$dst/meguvizcac-${ver}-full.zip" \
    bundle_v10_mesosphere_v170.mgvz charlm_v170.bin bili_picky_bpe_12k.json manifest.json \
    v170_r0_apexchar_fp32.onnx v170_prelude_t1_fp32.onnx v170_loop_step_fp32.onnx v170_dynaseam_fp32.onnx \
    v170_r0_apexchar_fp32.skel.onnx v170_prelude_t1_fp32.skel.onnx v170_loop_step_fp32.skel.onnx v170_dynaseam_fp32.skel.onnx \
    "metrics-${ver}.json" \
    "meguvizcac-${ver}-bootable-v10-mesosphere.zip" \
    "meguvizcac-${ver}-4stage-onnx-fp32exact.zip")
  (cd "$dst" && sha256sum "meguvizcac-${ver}"-*.zip "metrics-${ver}.json" > SHA256SUMS.txt && cat SHA256SUMS.txt) || true
  touch "$dst/.synced_ok" && sync
}

if [ "$RUN_TRACK_DC" = "1" ] && [ ! -f "$V171_DRIVE_OUT/v1.7.1-DreamCore/.synced_ok" ]; then
  if [ ! -f "$OUT_DC/metrics-v1.7.1-DreamCore.json" ]; then
    echo "[$(date +%H:%M:%S)] === Launching v1.7.1-DreamCore (v1.6.1 R0++ + Protected PNBR + r_echo<=0.50) ===" | tee -a "$LOG_DRIVER"
    python3 -u -m meguviz_cac.posttrain_v171_mesosphere \
      "${BASE_V171_ARGS[@]}" \
      --no-use-dual-hopfield \
      --init-from-fp16 "$PARENT_DC" \
      --out "$OUT_DC" \
      --version "v1.7.1-DreamCore" 2>&1 | tee "$OUT_DC/log_v1.7.1-DreamCore.txt"
  fi
  _sync_release "v1.7.1-DreamCore" "$OUT_DC" "$V171_DRIVE_OUT/v1.7.1-DreamCore"
fi

if [ "$RUN_TRACK_M1" = "1" ] && [ ! -f "$V171_DRIVE_OUT/v1.7.1-M1/.synced_ok" ]; then
  if [ ! -f "$OUT_M1/metrics-v1.7.1-M1.json" ]; then
    echo "[$(date +%H:%M:%S)] === Launching v1.7.1-M1 (v1.6.1 R0++ + DualHopfield B2a + Protected PNBR + r_echo<=0.50) ===" | tee -a "$LOG_DRIVER"
    python3 -u -m meguviz_cac.posttrain_v171_mesosphere \
      "${BASE_V171_ARGS[@]}" \
      --use-dual-hopfield \
      --init-from-fp16 "$PARENT_M1" \
      --out "$OUT_M1" \
      --version "v1.7.1-M1" 2>&1 | tee "$OUT_M1/log_v1.7.1-M1.txt"
  fi
  _sync_release "v1.7.1-M1" "$OUT_M1" "$V171_DRIVE_OUT/v1.7.1-M1"
fi

python3 - <<'PY'
import json
from pathlib import Path
drv = Path("/content/drive/MyDrive/MeguvizCAC/v171_production")
board = {}
for tag in ("v1.7.1-DreamCore", "v1.7.1-M1"):
    mp = drv / tag / f"metrics-{tag}.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        board[tag] = {
            "bootable_mib": d.get("export", {}).get("bootable_mib"),
            "ci_passed": d.get("release_gates", {}).get("v170_ci_passed"),
            "ci_count": f"{d.get('v170_ci_gates', {}).get('passed_count')}/{d.get('v170_ci_gates', {}).get('total_count')}",
            "r_echo": d.get("r_echo"),
            "t1_freeze_diff": d.get("t1_prefix_freeze_max_diff"),
            "b_cert": d.get("b_cert"),
            "hopfield_pr": d.get("dual_hopfield_witness_pr"),
            "iut_exit_rate": d.get("rung0_iut_policy", {}).get("overall_exit_rate"),
            "auto_ship_winner": d.get("auto_ship_gate", {}).get("winner"),
            "b1_r0_t1_ms": d.get("v170_ci_gates", {}).get("V170_B1_ORT_LATENCY_AUDIT", {}).get("b1_r0_plus_t1_ms"),
            "elapsed_s": d.get("elapsed_seconds"),
        }
(drv / "v171_comparison.json").write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
print("=== v1.7.1-DreamCore vs v1.7.1-M1 SCOREBOARD ===")
print(json.dumps(board, ensure_ascii=False, indent=2))
PY
cp -f "$LOG_DRIVER" "$V171_DRIVE_OUT/" || true
touch "$V171_DRIVE_OUT/.all_v171_synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [ALL DONE] v1.7.1 Synced to $V171_DRIVE_OUT ===" | tee -a "$LOG_DRIVER"
