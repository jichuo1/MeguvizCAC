#!/usr/bin/env bash
# MeguvizCAC v1.6.1-BW (Warm Ultimate Edition) — Turbo v2 + Rust `meguviz_native` Runner on Colab G4
#
# Combines:
#   - Track A (`v1.6.1-A`) battle-hardened warm-start backbone (w3_ai_pr=0.843, test_gold_v2 R@P99=87.65%, diag_god500=0.838)
#   - Track B (`v1.6.1-B`) Contractive Kernel D (0% Pi_R sat, 2.50 period2 convergence) + +0.30ms Discourse Attribution Ops (+3.0pp order sensitivity!)
#   - 2.5x Differential LR for newly-initialized discourse ops & contractive alpha vs gentle backbone fine-tuning
#   - Fixed Per-Candidate FP32 Drift Auto-Ship-Gate + WiSE-FT Model Soup (same parent basin!)
#   - Full Turbo v2 stack: Rust Zero-GIL Rayon frontend + GPU EMA + Batched Aux Passes + torch.compile + BS=256!
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
WHEEL_DIR="${WHEEL_DIR:-$DRIVE_ROOT/wheels}"

WORK_ROOT="${WORK_ROOT:-/content/out_v161}"
OUT_BW="$WORK_ROOT/v1.6.1-BW"
SHARED_R0="${SHARED_R0:-$V161_DRIVE_OUT/shared_r0_v161_frozen.pt}"
SHARED_CHARLM="${SHARED_CHARLM:-$V161_DRIVE_OUT/shared_oof_charlm_v161.pkl}"
WARM_MASTER="${WARM_MASTER:-$V161_DRIVE_OUT/v1.6.1-A/v1.6.1-A_fp16_master.pt}"
BS="${BS:-256}"
P2_STEPS="${P2_STEPS:-2800}"
P2_LR="${P2_LR:-5.5e-4}"
SEED="${SEED:-20261215}"
USE_COMPILE="${USE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
AMP_BF16="${AMP_BF16:-1}"
FUSE_AUX="${FUSE_AUX:-1}"
EVAL_BS="${EVAL_BS:-512}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"

mkdir -p "$WORK_ROOT" "$OUT_BW" "$V161_DRIVE_OUT/v1.6.1-BW" "$WHEEL_DIR"
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
if [ "$CUR_RS_SHA" != "$RUST_SHA8" ] || ! python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'RustOOFCharLMSet') and hasattr(_meguviz_rs, 'extract_post_bpe_all_rs')" >/dev/null 2>&1; then
  mkdir -p "$RUST_CACHE_DIR"
  CACHED_WHL="$(ls -1t "$RUST_CACHE_DIR"/meguviz_native-*-"${PY_TAG}"-*x86_64.whl 2>/dev/null | head -n 1 || true)"
  if [ -n "$CACHED_WHL" ]; then
    echo "[$(date +%H:%M:%S)] Installing cached Rust wheel (sha8=$RUST_SHA8): $CACHED_WHL"
    pip install -q --no-deps --force-reinstall "$CACHED_WHL"
  else
    echo "[$(date +%H:%M:%S)] Building Rust _meguviz_rs wheel (sha8=$RUST_SHA8) & caching to $RUST_CACHE_DIR..."
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
python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'RustOOFCharLMSet') and hasattr(_meguviz_rs, 'extract_post_bpe_all_rs'); print('[Rust-OK sha8=$RUST_SHA8]', _meguviz_rs.__file__)"

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

echo "[$(date +%H:%M:%S)] === Launching v1.6.1-BW (Warm Ultimate + Contractive-D + Discourse Ops + Turbo v3 Rust!) ==="
echo "  WARM_MASTER=$WARM_MASTER | SHARED_R0=$SHARED_R0 | BS=$BS | EVAL_BS=$EVAL_BS | P2_STEPS=$P2_STEPS | P2_LR=$P2_LR | COMPILE=$USE_COMPILE (dyn=$COMPILE_DYNAMIC) | AMP_BF16=$AMP_BF16 | FUSE_AUX=$FUSE_AUX"

ARGS=(
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
  --dynamics contractive_d
  --use-discourse-ops
  --sat-reg-weight 0.02
  --no-step-phi
  --expand-tables
  --lardo-w8-core
  --init-from-fp16 "$WARM_MASTER"
  --wise-anchor-fp16 "$WARM_MASTER"
  --init-r0-ckpt "$SHARED_R0"
  --freeze-r0
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
  --charlm-cache "$SHARED_CHARLM"
  --prefetch-workers "$PREFETCH_WORKERS"
  --p2-steps "$P2_STEPS"
  --p2-lr "$P2_LR"
  --p3-qat-steps 250
  --qat-lr 5e-5
  --out "$OUT_BW"
  --checkpoint-dir "$V161_DRIVE_OUT/v1.6.1-BW"
  --version "v1.6.1-BW"
)
[ -d "$V123_BUNDLE_DIR/delta" ] && ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && ARGS+=(--v04-data "$V04_DATA")
[ -f "$RESCORED_LOCAL" ] && ARGS+=(--extra-gold-pack "$RESCORED_LOCAL")
[ -f "$CAL_CERT_PACK" ] && ARGS+=(--extra-cal-cert-pack "$CAL_CERT_PACK")
[ "$USE_COMPILE" = "1" ] && ARGS+=(--compile --compile-mode "$COMPILE_MODE")
[ "$COMPILE_DYNAMIC" = "1" ] && ARGS+=(--compile-dynamic) || ARGS+=(--no-compile-dynamic)
[ "$AMP_BF16" = "1" ] && ARGS+=(--amp-bf16) || ARGS+=(--no-amp-bf16)
[ "$FUSE_AUX" = "1" ] && ARGS+=(--fuse-aux) || ARGS+=(--no-fuse-aux)
ARGS+=(--eval-batch-size "$EVAL_BS")

python3 -u -m meguviz_cac.train_v160_stratosphere_loop "${ARGS[@]}" 2>&1 | tee "$OUT_BW/log_v1.6.1-BW.txt"

# 3. Sync v1.6.1-BW release bundle to Drive & update combined A vs B vs BW scoreboard
DST="$V161_DRIVE_OUT/v1.6.1-BW"
cp -f "$OUT_BW/metrics-v1.6.1-BW.json" "$DST/"
[ -f "$OUT_BW/v1.6.1-BW_eval_cache.npz" ] && cp -f "$OUT_BW/v1.6.1-BW_eval_cache.npz" "$DST/"
[ -f "$OUT_BW/log_v1.6.1-BW.txt" ] && cp -f "$OUT_BW/log_v1.6.1-BW.txt" "$DST/"
cp -f "$OUT_BW/meguvizcac-v1.6.1-BW-bootable-v10-stratosphere.zip" "$DST/" || true
(cd "$OUT_BW" && zip -r -6 "$DST/meguvizcac-v1.6.1-BW-full.zip" \
  bundle_v10_stratosphere_v160.mgvz charlm_v160.bin bili_picky_bpe_12k.json manifest.json \
  v160_r0_apexchar_fp32.onnx v160_prelude_t1_fp32.onnx v160_loop_step_fp32.onnx v160_dynaseam_fp32.onnx \
  v160_r0_apexchar_fp32.skel.onnx v160_prelude_t1_fp32.skel.onnx v160_loop_step_fp32.skel.onnx v160_dynaseam_fp32.skel.onnx \
  "metrics-v1.6.1-BW.json" \
  "meguvizcac-v1.6.1-BW-bootable-v10-stratosphere.zip" \
  "meguvizcac-v1.6.1-BW-4stage-onnx-fp32exact.zip")
(cd "$DST" && sha256sum meguvizcac-v1.6.1-BW-*.zip metrics-v1.6.1-BW.json > SHA256SUMS.txt && cat SHA256SUMS.txt)

python3 - <<'PY'
import json
from pathlib import Path

drv = Path("/content/drive/MyDrive/MeguvizCAC/v161_production")
board = {}
for tag in ("v1.6.1-A", "v1.6.1-B", "v1.6.1-BW"):
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
        }
txt = json.dumps(board, ensure_ascii=False, indent=2)
(drv / "v161_ab_comparison.json").write_text(txt, encoding="utf-8")
print("=== v1.6.1-A vs v1.6.1-B vs v1.6.1-BW TRIPLE SCOREBOARD ===")
print(txt)
PY
touch "$DST/.synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [ALL DONE] v1.6.1-BW Synced to $DST ==="
