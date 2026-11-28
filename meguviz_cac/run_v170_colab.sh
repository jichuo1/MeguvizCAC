#!/usr/bin/env bash
# MeguvizCAC v1.7.0 MesosphereLoop 3-Track Runner on Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)
#
#   Track A0        (`v1.7.0-A0`):        Natural R0++ + H=256 MesosphereLoop baseline (pure FP32 ONNX <= 5.00 MiB)
#   Track DreamCore (`v1.7.0-DreamCore`): Natural R0++ + RI-Meso-PSD-L + AR-U-Loop + Step-LoRA + DeltaRule 16x16 + M6/M7/M8
#   Track M1        (`v1.7.0-M1`):        Full DreamCore + Once-at-T1 DualHopfield (B2a) + M6/M7/M8
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
WHEEL_DIR="${WHEEL_DIR:-$DRIVE_ROOT/wheels}"

WORK_ROOT="${WORK_ROOT:-/content/out_v170}"
OUT_A0="$WORK_ROOT/v1.7.0-A0"
OUT_DC="$WORK_ROOT/v1.7.0-DreamCore"
OUT_M1="$WORK_ROOT/v1.7.0-M1"
SHARED_R0_V170="${SHARED_R0_V170:-$V170_DRIVE_OUT/shared_r0_v170_natural.pt}"
SHARED_CHARLM="${SHARED_CHARLM:-$V161_DRIVE_OUT/shared_oof_charlm_v161.pkl}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v170_driver.txt}"

BS="${BS:-256}"
EVAL_BS="${EVAL_BS:-512}"
P2_STEPS="${P2_STEPS:-3600}"
P2_LR="${P2_LR:-1.1e-3}"
SEED="${SEED:-20261225}"
RUN_TRACK_A0="${RUN_TRACK_A0:-1}"
RUN_TRACK_DREAM="${RUN_TRACK_DREAM:-1}"
RUN_TRACK_M1="${RUN_TRACK_M1:-1}"
USE_COMPILE="${USE_COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_DYNAMIC="${COMPILE_DYNAMIC:-1}"
AMP_BF16="${AMP_BF16:-1}"
FUSE_AUX="${FUSE_AUX:-1}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"

mkdir -p "$WORK_ROOT" "$OUT_A0" "$OUT_DC" "$OUT_M1" "$V170_DRIVE_OUT/v1.7.0-A0" "$V170_DRIVE_OUT/v1.7.0-DreamCore" "$V170_DRIVE_OUT/v1.7.0-M1" "$WHEEL_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 0. Ensure required Python inference/export packages exist on a fresh Colab VM
if ! python3 -c "import onnx, onnxruntime, onnxscript, tokenizers, scipy, sklearn" >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] Installing missing Python deps (onnx, onnxruntime, onnxscript, tokenizers)..." | tee -a "$LOG_DRIVER"
  pip install -q onnx onnxruntime onnxscript tokenizers
fi

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
python3 -c "import _meguviz_rs; assert hasattr(_meguviz_rs, 'extract_v170_uisr_and_surf_batch_rs') and hasattr(_meguviz_rs.RustOOFCharLMSet, 'extract_v170_ph16_batch'); print('[Rust-v4-OK sha8=$RUST_SHA8]', _meguviz_rs.__file__)"

# 2. Ensure local fast SSD copies of ALL required corpora & bundles on a fresh Colab VM
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave2_longform_gold_1200.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Extracting v11_data bundle ($V11_BUNDLE_ZIP) -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
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
    echo "[$(date +%H:%M:%S)] Extracting v123 bundle ($FRESH_V123_ZIP) -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$FRESH_V123_ZIP').extractall('$V123_BUNDLE_DIR')"
    echo "$ZIP_SHA" > "$V123_BUNDLE_DIR/.bundle_sha256"
  fi
fi

DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz" ] && DISTILL_POOL="/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz"
[ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ] && cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "$DISTILL_POOL"
RESCORED_LOCAL="/content/v161_duckdb2_rescored.jsonl.gz"
[ ! -f "$RESCORED_LOCAL" ] && [ -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" ] && cp -f "$V161_DATA_DIR/v161_duckdb2_rescored.jsonl.gz" "$RESCORED_LOCAL"
V07_TOKENIZER="$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V11_DRIVE_DIR/bili_picky_bpe_12k.json"
[ ! -f "$V07_TOKENIZER" ] && V07_TOKENIZER="$V161_DRIVE_OUT/v1.6.1-A/bili_picky_bpe_12k.json"
if [ ! -f "$V04_DATA" ]; then
  for vcand in \
    "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" \
    "$DRIVE_ROOT/data/v04-merged-1m4-r3.jsonl.gz" \
    "$DRIVE_ROOT/v04-merged-1m4-r3.jsonl.gz" \
    "$DRIVE_ROOT/v080_zenith/data/v04-merged-1m4-r3.jsonl.gz" \
    "$DRIVE_ROOT/v070_omniveritas/data/v04-merged-1m4-r3.jsonl.gz"; do
    if [ -f "$vcand" ]; then
      mkdir -p "$(dirname "$V04_DATA")"
      echo "[$(date +%H:%M:%S)] Copying v04 corpus to local NVMe: $vcand -> $V04_DATA" | tee -a "$LOG_DRIVER"
      cp -f "$vcand" "$V04_DATA"
      break
    fi
  done
fi
CAL_CERT_PACK="$V161_DATA_DIR/v161_duckdb2_cal_cert.jsonl.gz"

echo "[$(date +%H:%M:%S)] === Pre-flight Input Check ===" | tee -a "$LOG_DRIVER"
for req in "$DISTILL_POOL" "$V07_TOKENIZER" "$V11_DATA_DIR/v11_wave2_longform_gold_1200.jsonl.gz" "$V123_BUNDLE_DIR/sibling" "$V123_BUNDLE_DIR/gold_v2_w3"; do
  if [ ! -e "$req" ]; then
    echo "[ERROR] Missing required input: $req" | tee -a "$LOG_DRIVER"
    exit 2
  fi
  echo "  [OK] $req" | tee -a "$LOG_DRIVER"
done
echo "  [OPT] V04_DATA=$V04_DATA (exists=$([ -f "$V04_DATA" ] && echo yes || echo no))" | tee -a "$LOG_DRIVER"
echo "  [OPT] RESCORED_LOCAL=$RESCORED_LOCAL (exists=$([ -f "$RESCORED_LOCAL" ] && echo yes || echo no))" | tee -a "$LOG_DRIVER"
echo "  [OPT] CAL_CERT_PACK=$CAL_CERT_PACK (exists=$([ -f "$CAL_CERT_PACK" ] && echo yes || echo no))" | tee -a "$LOG_DRIVER"
echo "  [OPT] SHARED_CHARLM=$SHARED_CHARLM (exists=$([ -f "$SHARED_CHARLM" ] && echo yes || echo no))" | tee -a "$LOG_DRIVER"

BASE_V170_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --p1-steps 2200
  --p1-use-cf
  --apd-steps 350
  --freeze-r0
  --batch-size "$BS"
  --eval-batch-size "$EVAL_BS"
  --p2-steps "$P2_STEPS"
  --p2-lr "$P2_LR"
  --stage2-frac 0.35
  --p3-qat-steps 300
  --qat-lr 4.0e-5
  --seed "$SEED"
  --auto-ship-gate
  --use-m6-ph
  --use-m7-cnma-uisr
  --use-discourse-ops
  --lardo-w8-core
  --use-tcbr-r
  --cvar-alpha 0.08
  --cvar-weight 0.07
  --recall-floor-weight 0.14
  --use-fmt-cf
  --use-d02-paraphrase-cf
  --use-promo-flip-cf
  --use-cc-vrex
  --use-er-dro-r
  --longform-every 4
  --use-kadane-focal-router
  --use-eba-h
  --eba-pi-nat 0.05
  --iut-alpha-max 0.04
  --charlm-cache "$SHARED_CHARLM"
  --prefetch-workers "$PREFETCH_WORKERS"
)
[ -d "$V123_BUNDLE_DIR/delta" ] && BASE_V170_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_DATA" ] && BASE_V170_ARGS+=(--v04-data "$V04_DATA")
[ -f "$RESCORED_LOCAL" ] && BASE_V170_ARGS+=(--extra-gold-pack "$RESCORED_LOCAL")
[ -f "$CAL_CERT_PACK" ] && BASE_V170_ARGS+=(--extra-cal-cert-pack "$CAL_CERT_PACK")
[ "$USE_COMPILE" = "1" ] && BASE_V170_ARGS+=(--compile --compile-mode "$COMPILE_MODE")
[ "$COMPILE_DYNAMIC" = "1" ] && BASE_V170_ARGS+=(--compile-dynamic) || BASE_V170_ARGS+=(--no-compile-dynamic)
[ "$AMP_BF16" = "1" ] && BASE_V170_ARGS+=(--amp-bf16) || BASE_V170_ARGS+=(--no-amp-bf16)
[ "$FUSE_AUX" = "1" ] && BASE_V170_ARGS+=(--fuse-aux) || BASE_V170_ARGS+=(--no-fuse-aux)

_sync_release() {
  local ver="$1"
  local src="$2"
  local dst="$3"
  mkdir -p "$dst"
  [ ! -f "$src/bili_picky_bpe_12k.json" ] && [ -f "$src/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json" ] && cp -f "$src/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json" "$src/bili_picky_bpe_12k.json"
  cp -f "$src/metrics-${ver}.json" "$dst/"
  [ -f "$src/${ver}_eval_cache.npz" ] && cp -f "$src/${ver}_eval_cache.npz" "$dst/"
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

_finalize_existing_export() {
  local ver="$1"
  local src="$2"
  local dst="$3"
  echo "[$(date +%H:%M:%S)] === Fast-Finalizing already-exported $ver (26/26 CI + Drive Sync, skipping redundant training) ===" | tee -a "$LOG_DRIVER"
  python3 - "$ver" "$src" "$V07_TOKENIZER" <<'PY' 2>&1 | tee -a "$src/log_${ver}.txt"
import json, sys, time
from pathlib import Path
import torch
from meguviz_cac.v07_features import BiliPickyBPETokenizer
from meguviz_cac.v150_heaven_loop import CompactCharLMV150
from meguviz_cac.v170_mesosphere_loop import (
    MeguvizV170CascadeEngine,
    MeguvizV170MesosphereLoopModel,
    export_v170_bootable_and_onnx,
    load_bundle_v10_v170_into_model,
    run_v170_ci_gates,
)

t0 = time.time()
ver, out_dir, tok_src = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
tok_dst = out_dir / "bili_picky_bpe_12k.json"
if not tok_dst.is_file():
    tok_dst.write_bytes(tok_src.read_bytes())
tokenizer = BiliPickyBPETokenizer(tok_dst)
charlm_p = out_dir / "charlm_v170.bin"
charlm = None
if charlm_p.is_file():
    charlm = CompactCharLMV150.load(charlm_p)
    charlm.__class__ = CompactCharLMV150

model = MeguvizV170MesosphereLoopModel(
    arch=str(manifest.get("arch", "v170_a0")),
    vocab_size=tokenizer.vocab_size,
    max_len=int(manifest.get("max_len", 288)),
    use_discourse_ops=bool(manifest.get("use_discourse_ops", True)),
    use_ar_uloop=bool(manifest.get("use_ar_uloop", False)),
    use_step_lora=bool(manifest.get("use_step_lora", False)),
    step_lora_r=int(manifest.get("step_lora_r", 0)),
    use_delta_rule=bool(manifest.get("use_delta_rule", False)),
    use_dual_hopfield=bool(manifest.get("use_dual_hopfield", False)),
    use_m6_ph=bool(manifest.get("use_m6_ph", True)),
    use_m7_cnma_uisr=bool(manifest.get("use_m7_cnma_uisr", True)),
    lardo_w8_core=bool(manifest.get("lardo_w8_core", True)),
    use_local_charlm_inject=bool(manifest.get("use_local_charlm_inject", True)),
)
load_bundle_v10_v170_into_model(model, out_dir / "bundle_v10_mesosphere_v170.mgvz")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev).eval()
export_rep = export_v170_bootable_and_onnx(model, tok_dst, manifest, out_dir, ver, charlm=charlm)
eng = MeguvizV170CascadeEngine(out_dir, threads=4, r0_threads=2)
ci_gates = run_v170_ci_gates(model, tokenizer, export_rep, out_dir, dev, charlm=charlm, engine=eng)
report = {
    "version": ver,
    "arch": model.arch,
    "finalized_from_existing_export": True,
    "export": export_rep,
    "v170_ci_gates": ci_gates,
    "release_gates": {
        "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
        "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
        "v170_ci_passed": bool(ci_gates["all_ci_passed"]),
    },
    "finalize_elapsed_seconds": round(time.time() - t0, 2),
}
(out_dir / f"metrics-{ver}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[v1.7.0 Fast-Finalize Done] ver={ver} | bootable={export_rep['bootable_mib']} MiB | CI={ci_gates['passed_count']}/{ci_gates['total_count']} PASS={ci_gates['all_ci_passed']} | Forb={export_rep['forbidden_ops_total']}", flush=True)
if not report["release_gates"]["v170_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
    raise SystemExit(1)
PY
  _sync_release "$ver" "$src" "$dst"
}

_r0_args() {
  if [ -f "$SHARED_R0_V170" ]; then
    echo "--init-r0-ckpt $SHARED_R0_V170"
  else
    echo "--save-r0-ckpt $SHARED_R0_V170"
  fi
}

# Auto-finalize any already-exported v1.7.0-A0 (atomic lock so parallel MPS children never race)
if [ -f "$OUT_A0/bundle_v10_mesosphere_v170.mgvz" ] && [ -f "$OUT_A0/manifest.json" ] && [ ! -f "$V170_DRIVE_OUT/v1.7.0-A0/metrics-v1.7.0-A0.json" ]; then
  if mkdir /tmp/.meguviz_v170_a0_finalize_lock 2>/dev/null; then
    trap 'rmdir /tmp/.meguviz_v170_a0_finalize_lock 2>/dev/null || true' EXIT
    _finalize_existing_export "v1.7.0-A0" "$OUT_A0" "$V170_DRIVE_OUT/v1.7.0-A0"
    rmdir /tmp/.meguviz_v170_a0_finalize_lock 2>/dev/null || true
    trap - EXIT
  fi
fi

# ===========================================================================
# TRACK 1: v1.7.0-A0 (Natural R0++ + H=256 Baseline)
# ===========================================================================
if [ "$RUN_TRACK_A0" = "1" ] && [ ! -f "$V170_DRIVE_OUT/v1.7.0-A0/metrics-v1.7.0-A0.json" ]; then
  if [ -f "$OUT_A0/bundle_v10_mesosphere_v170.mgvz" ] && [ -f "$OUT_A0/manifest.json" ] && [ ! -f "$OUT_A0/metrics-v1.7.0-A0.json" ]; then
    _finalize_existing_export "v1.7.0-A0" "$OUT_A0" "$V170_DRIVE_OUT/v1.7.0-A0"
  elif [ ! -f "$OUT_A0/metrics-v1.7.0-A0.json" ]; then
    echo "[$(date +%H:%M:%S)] === Launching v1.7.0-A0 (Natural R0++ + H=256 Baseline) ===" | tee -a "$LOG_DRIVER"
    read -r -a R0_A0 <<< "$(_r0_args)"
    python3 -u -m meguviz_cac.train_v170_mesosphere \
      "${BASE_V170_ARGS[@]}" \
      "${R0_A0[@]}" \
      --arch v170_a0 \
      --out "$OUT_A0" \
      --checkpoint-dir "$V170_DRIVE_OUT/v1.7.0-A0" \
      --version "v1.7.0-A0" 2>&1 | tee "$OUT_A0/log_v1.7.0-A0.txt"
    _sync_release "v1.7.0-A0" "$OUT_A0" "$V170_DRIVE_OUT/v1.7.0-A0"
  else
    _sync_release "v1.7.0-A0" "$OUT_A0" "$V170_DRIVE_OUT/v1.7.0-A0"
  fi
fi

# ===========================================================================
# TRACK 2: v1.7.0-DreamCore (DreamCore + M6/M7/M8, no DualHopfield)
# ===========================================================================
if [ "$RUN_TRACK_DREAM" = "1" ] && [ ! -f "$OUT_DC/metrics-v1.7.0-DreamCore.json" ] && [ ! -f "$V170_DRIVE_OUT/v1.7.0-DreamCore/metrics-v1.7.0-DreamCore.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.7.0-DreamCore (MesosphereLoop DreamCore + M6/M7/M8) ===" | tee -a "$LOG_DRIVER"
  read -r -a R0_DC <<< "$(_r0_args)"
  python3 -u -m meguviz_cac.train_v170_mesosphere \
    "${BASE_V170_ARGS[@]}" \
    "${R0_DC[@]}" \
    --arch v170_m1 \
    --no-use-dual-hopfield \
    --out "$OUT_DC" \
    --checkpoint-dir "$V170_DRIVE_OUT/v1.7.0-DreamCore" \
    --version "v1.7.0-DreamCore" 2>&1 | tee "$OUT_DC/log_v1.7.0-DreamCore.txt"
  _sync_release "v1.7.0-DreamCore" "$OUT_DC" "$V170_DRIVE_OUT/v1.7.0-DreamCore"
fi

# ===========================================================================
# TRACK 3: v1.7.0-M1 (Full DreamCore + Once-at-T1 DualHopfield B2a)
# ===========================================================================
if [ "$RUN_TRACK_M1" = "1" ] && [ ! -f "$OUT_M1/metrics-v1.7.0-M1.json" ] && [ ! -f "$V170_DRIVE_OUT/v1.7.0-M1/metrics-v1.7.0-M1.json" ]; then
  echo "[$(date +%H:%M:%S)] === Launching v1.7.0-M1 (Full DreamCore + DualHopfield B2a + M6/M7/M8) ===" | tee -a "$LOG_DRIVER"
  read -r -a R0_M1 <<< "$(_r0_args)"
  python3 -u -m meguviz_cac.train_v170_mesosphere \
    "${BASE_V170_ARGS[@]}" \
    "${R0_M1[@]}" \
    --arch v170_m1 \
    --out "$OUT_M1" \
    --checkpoint-dir "$V170_DRIVE_OUT/v1.7.0-M1" \
    --version "v1.7.0-M1" 2>&1 | tee "$OUT_M1/log_v1.7.0-M1.txt"
  _sync_release "v1.7.0-M1" "$OUT_M1" "$V170_DRIVE_OUT/v1.7.0-M1"
fi

# ===========================================================================
# SCOREBOARD: Compare v1.7.0-A0 vs v1.7.0-DreamCore vs v1.7.0-M1
# ===========================================================================
python3 - <<'PY'
import json
from pathlib import Path

work = Path("/content/out_v170")
drv = Path("/content/drive/MyDrive/MeguvizCAC/v170_production")
board = {}
for tag in ("v1.7.0-A0", "v1.7.0-DreamCore", "v1.7.0-M1"):
    mp = work / tag / f"metrics-{tag}.json"
    if not mp.is_file():
        mp = drv / tag / f"metrics-{tag}.json"
    if mp.is_file():
        d = json.loads(mp.read_text(encoding="utf-8"))
        board[tag] = {
            "arch": d.get("arch"),
            "bootable_mib": d.get("export", {}).get("bootable_mib"),
            "ci_passed": d.get("release_gates", {}).get("v170_ci_passed"),
            "ci_count": f"{d.get('v170_ci_gates', {}).get('passed_count')}/{d.get('v170_ci_gates', {}).get('total_count')}",
            "auto_ship_winner": d.get("v170_config", {}).get("auto_ship_winner"),
            "phase2_best_val": d.get("phase2_best_val"),
            "pnbr_B_cert": d.get("pnbr_B_cert_holdout"),
            "hopfield_pr": d.get("dual_hopfield_witness_pr"),
            "b1_r0_t1_ms": d.get("v170_ci_gates", {}).get("V170_B1_ORT_LATENCY_AUDIT", {}).get("b1_r0_plus_t1_ms"),
            "elapsed_s": d.get("elapsed_seconds", d.get("finalize_elapsed_seconds")),
        }
txt = json.dumps(board, ensure_ascii=False, indent=2)
(work / "v170_comparison.json").write_text(txt, encoding="utf-8")
(drv / "v170_comparison.json").write_text(txt, encoding="utf-8")
print("=== v1.7.0-A0 vs v1.7.0-DreamCore vs v1.7.0-M1 SCOREBOARD ===")
print(txt)
PY
cp -f "$LOG_DRIVER" "$V170_DRIVE_OUT/" || true
touch "$V170_DRIVE_OUT/.all_v170_synced_ok" && sync
echo "[$(date +%H:%M:%S)] === [ALL DONE] v1.7.0 Synced to $V170_DRIVE_OUT ===" | tee -a "$LOG_DRIVER"
