#!/usr/bin/env bash
# MeguvizCAC v1.7.1 Post-Training Suite Runner on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)
#
#   Track PUC       (`v1.7.1-PUC`):       r_echo <= 0.65, Strategy I PUC + Frontier Miner + Protected PNBR + 68B TitanGuard
#   Track TightEcho (`v1.7.1-TightEcho`): r_echo <= 0.50, tighter J-Cert projection + 8-Cand WiSE-FT Auto-Ship-Gate
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
OUT_PUC="$WORK_ROOT/v1.7.1-PUC"
OUT_TE="$WORK_ROOT/v1.7.1-TightEcho"
SHARED_R0_V170="${SHARED_R0_V170:-$V170_DRIVE_OUT/shared_r0_v170_natural.pt}"
SHARED_CHARLM="${SHARED_CHARLM:-$V161_DRIVE_OUT/shared_oof_charlm_v161.pkl}"
PARENT_MASTER="${PARENT_MASTER:-$V170_DRIVE_OUT/v1.7.0-M1/v1.7.0-M1_fp16_master.pt}"
if [ ! -f "$PARENT_MASTER" ] && [ -f "$V170_DRIVE_OUT/v1.7.0-DreamCore/v1.7.0-DreamCore_fp16_master.pt" ]; then
  PARENT_MASTER="$V170_DRIVE_OUT/v1.7.0-DreamCore/v1.7.0-DreamCore_fp16_master.pt"
fi
LOG_DRIVER="${LOG_DRIVER:-/content/log_v171_driver.txt}"

BS="${BS:-256}"
EVAL_BS="${EVAL_BS:-512}"
SEED="${SEED:-20261230}"
RUN_TRACK_PUC="${RUN_TRACK_PUC:-1}"
RUN_TRACK_TE="${RUN_TRACK_TE:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
AMP_BF16="${AMP_BF16:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"

mkdir -p "$WORK_ROOT" "$OUT_PUC" "$OUT_TE" "$V171_DRIVE_OUT/v1.7.1-PUC" "$V171_DRIVE_OUT/v1.7.1-TightEcho" "$WHEEL_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Content-hash-keyed Rust `_meguviz_rs` v4 wheel cache (Cargo.toml + src/lib.rs -> sha8)
RUST_SHA8="$(cat "$REPO_DIR/meguviz_native/Cargo.toml" "$REPO_DIR/meguviz_native/src/lib.rs" | sha256sum | cut -c1-8)"
PY_TAG="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
RUST_CACHE_DIR="$WHEEL_DIR/$RUST_SHA8"
CUR_RS_SHA="$(cat /tmp/.meguviz_rs_sha8 2>/dev/null || true)"
if [ "$CUR_RS_SHA" != "$RUST_SHA8" ] || ! python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'extract_v170_uisr_and_surf_batch_rs') and hasattr(_meguviz_rs.RustOOFCharLMSet, 'extract_v170_ph16_batch')" >/dev/null 2>&1; then
  mkdir -p "$RUST_CACHE_DIR"
  CACHED_WHL="$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl 2>/dev/null | head -n 1 || true)"
  if [ -n "$CACHED_WHL" ]; then
    echo "[$(date +%H:%M:%S)] Installing cached Rust wheel (sha8=$RUST_SHA8): $CACHED_WHL" | tee -a "$LOG_DRIVER"
    pip install -q --no-deps --force-reinstall "$CACHED_WHL"
  else
    echo "[$(date +%H:%M:%S)] Building Rust _meguviz_rs v4 wheel (sha8=$RUST_SHA8) & caching to $RUST_CACHE_DIR..." | tee -a "$LOG_DRIVER"
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

BASE_V171_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --arch v170_m1
  --init-from-fp16 "$PARENT_MASTER"
  --init-r0-ckpt "$SHARED_R0_V170"
  --batch-size "$BS"
  --eval-batch-size "$EVAL_BS"
  --stage-a-steps 1200
  --stage-b-steps 800
  --apd-steps 400
  --p3-qat-steps 300
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
  cp -f "$src/meguvizcac-${ver}-bootable-v10-mesosphere.zip" "$dst/" 2>/dev/null || cp -f "$src/meguvizcac-${ver}-bootable-v10-stratosphere.zip" "$dst/" || true
  (cd "$src" && zip -r -6 "$dst/meguvizcac-${ver}-full.zip" \
    bundle_v10_stratosphere_v160.mgvz charlm_v160.bin bili_picky_bpe_12k.json manifest.json \
    v170_r0_apexchar_fp32.onnx v170_prelude_t1_fp32.onnx v170_loop_step_fp32.onnx v170_dynaseam_fp32.onnx \
    "metrics-${ver}.json") || true
  touch "$dst/.synced_ok" && sync
}

if [ "$RUN_TRACK_PUC" = "1" ] && [ ! -f "$OUT_PUC/metrics-v1.7.1-PUC.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.7.1-PUC (r_echo <= 0.65) ===" | tee -a "$LOG_DRIVER"
  python3 -u -m meguviz_cac.posttrain_v171_mesosphere \
    "${BASE_V171_ARGS[@]}" \
    --r-echo-cap 0.65 \
    --out "$OUT_PUC" \
    --version "v1.7.1-PUC" 2>&1 | tee "$OUT_PUC/log_v1.7.1-PUC.txt"
  _sync_release "v1.7.1-PUC" "$OUT_PUC" "$V171_DRIVE_OUT/v1.7.1-PUC"
fi

if [ "$RUN_TRACK_TE" = "1" ] && [ ! -f "$OUT_TE/metrics-v1.7.1-TightEcho.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.7.1-TightEcho (r_echo <= 0.50) ===" | tee -a "$LOG_DRIVER"
  python3 -u -m meguviz_cac.posttrain_v171_mesosphere \
    "${BASE_V171_ARGS[@]}" \
    --r-echo-cap 0.50 \
    --out "$OUT_TE" \
    --version "v1.7.1-TightEcho" 2>&1 | tee "$OUT_TE/log_v1.7.1-TightEcho.txt"
  _sync_release "v1.7.1-TightEcho" "$OUT_TE" "$V171_DRIVE_OUT/v1.7.1-TightEcho"
fi

python3 - <<'PY'
import json
from pathlib import Path
drv = Path("/content/drive/MyDrive/MeguvizCAC/v171_production")
board = {}
for tag in ("v1.7.1-PUC", "v1.7.1-TightEcho"):
    mp = drv / tag / f"metrics-{tag}.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        board[tag] = {
            "bootable_mib": d.get("export", {}).get("bootable_mib"),
            "ci_passed": d.get("release_gates", {}).get("v170_ci_passed"),
            "r_echo": d.get("r_echo"),
            "t1_freeze_diff": d.get("t1_prefix_freeze_max_diff"),
            "b_cert": d.get("b_cert"),
            "auto_ship_winner": d.get("auto_ship_gate", {}).get("winner"),
            "elapsed_s": d.get("elapsed_seconds"),
        }
(drv / "v171_comparison.json").write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(board, indent=2))
PY
cp -f "$LOG_DRIVER" "$V171_DRIVE_OUT/" || true
touch "$V171_DRIVE_OUT/.all_v171_synced_ok" && sync
