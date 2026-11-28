#!/usr/bin/env bash
# MeguvizCAC v1.5.0-HeavenLoop (`HeavenBlock-2L` Dual-Layer Shared Recurrent Core, <=5.00 MiB Bootable!)
# Stage 1: Extract SenseNova-Loop-9B Layer-27 Dual-Visit Process Deltas (H7, cached to Google Drive!)
# Stage 2: Train v1.5.0-r1-heaven-loop (`core_depth=2`, 7 v1.4.0-bench fixes, 17/17 CI gates, <=5.00 MiB bootable)
# Stage 3: Pack full release archive, sync & SHA256-verify to Google Drive!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
OUT_DIR="${OUT_DIR:-/content/out_v150_r1}"
PROC_OUT_DIR="${PROC_OUT_DIR:-/content/out_v150_process}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v150_heaven_loop}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v150_r1_driver.txt}"
LOG_PROC="${LOG_PROC:-/content/log_v150_h7_extract.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v150_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v150_r1_done.json}"
VERSION="${VERSION:-v1.5.0-r1-heaven-loop}"
CORE_DEPTH="${CORE_DEPTH:-2}"
PRELUDE_MODE="${PRELUDE_MODE:-single_0p6m}"

mkdir -p "$OUT_DIR" "$PROC_OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$V123_BUNDLE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Discover v1.4.0 Super-Teacher Ultra 251K Soft Pool & SenseNova-Loop-9B Checkpoint
DISTILL_POOL="${DISTILL_POOL:-/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz}"
if [ ! -f "$DISTILL_POOL" ] && [ -f "/content/v140_superteacher_soft_distill_pool.jsonl.gz" ]; then
  DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
fi
if [ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ]; then
  echo "[$(date +%H:%M:%S)] Copying 251K Super-Teacher soft pool from Drive..." | tee -a "$LOG_DRIVER"
  cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "/content/v140_superteacher_soft_distill_pool.jsonl.gz"
  DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
fi
if [ ! -f "$DISTILL_POOL" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing Super-Teacher distill pool: $DISTILL_POOL" | tee -a "$LOG_DRIVER"
  exit 2
fi

SN9_CKPT="${SN9_CKPT:-$ST_DRIVE_DIR/v140_st_sensenova_loop_9b_best.pt}"
if [ ! -f "$SN9_CKPT" ] && [ -f "/content/out_v140_superteacher/v140_st_sensenova_loop_9b_best.pt" ]; then
  SN9_CKPT="/content/out_v140_superteacher/v140_st_sensenova_loop_9b_best.pt"
fi

# 2. Ensure v11 & unified v123 zero-leak bundles (incl. committee-1470!) are unpacked
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_verified.jsonl.gz" ] && [ -f "$V11_BUNDLE_ZIP" ]; then
  python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
fi
UNIFIED_ZIP=""
for zcand in "/content/v123_all_gold_sibling_and_delta_bundle.zip" "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip"; do
  [ -f "$zcand" ] && UNIFIED_ZIP="$zcand" && break
done
if [ -n "$UNIFIED_ZIP" ]; then
  CUR_SHA="$(sha256sum "$UNIFIED_ZIP" | awk '{print $1}')"
  PREV_SHA="$(cat "$V123_BUNDLE_DIR/.zip_sha" 2>/dev/null || true)"
  if [ "$CUR_SHA" != "$PREV_SHA" ] || [ ! -f "$V123_BUNDLE_DIR/delta/delta_v14_committee_1470_v11v12_overrides.jsonl.gz" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking unified v123 zero-leak bundle $UNIFIED_ZIP -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$UNIFIED_ZIP').extractall('$V123_BUNDLE_DIR')"
    echo "$CUR_SHA" > "$V123_BUNDLE_DIR/.zip_sha"
  fi
fi

# 3. Auto-discover Picky-BPE-12k tokenizer & v04 multilingual anchor source
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" "$V11_DRIVE_DIR/bili_picky_bpe_12k.json" "$DRIVE_ROOT/v122_apexnano/bili_picky_bpe_12k.json"; do
    [ -f "$tc" ] && V07_TOKENIZER="$tc" && break
  done
fi

EXTRA_ARGS=()
[ -d "$V123_BUNDLE_DIR/delta" ] && EXTRA_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
if [ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" ]; then
  V04_DATA="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"
fi
[ -f "$V04_DATA" ] && EXTRA_ARGS+=(--v04-data "$V04_DATA")

# 4. Stage 1: Offline SenseNova-Loop-9B Physical Layer-27 Dual-Visit Process Delta Extraction (H7, cached!)
PROC_NPZ="$PROC_OUT_DIR/v150_sensenova_process_deltas.npz"
if [ ! -f "$PROC_NPZ" ] && [ -f "$DRIVE_DIR/v150_sensenova_process_deltas.npz" ]; then
  echo "[$(date +%H:%M:%S)] Reusing cached H7 process deltas from Drive..." | tee -a "$LOG_DRIVER"
  cp -f "$DRIVE_DIR/v150_sensenova_process_deltas.npz" "$PROC_NPZ"
fi
if [ ! -f "$PROC_NPZ" ] && [ -f "$SN9_CKPT" ] && [ "${SKIP_H7_EXTRACT:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] Stage 1: Extracting SenseNova-Loop-9B Layer-27 Dual-Visit Process Deltas..." | tee -a "$LOG_DRIVER"
  python3 -u -m meguviz_cac.extract_v150_sensenova_process \
    --ckpt "$SN9_CKPT" \
    --distill-pool "$DISTILL_POOL" \
    --sibling-dir "$V123_BUNDLE_DIR/sibling" \
    --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
    --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
    --v11-data-dir "$V11_DATA_DIR" \
    "${EXTRA_ARGS[@]}" \
    --out "$PROC_OUT_DIR" \
    --batch-size "${PROC_BS:-160}" 2>&1 | tee "$LOG_PROC"
  cp -f "$PROC_NPZ" "$DRIVE_DIR/"
  cp -f "$PROC_OUT_DIR/v150_sensenova_process_meta.json" "$DRIVE_DIR/"
  cp -f "$LOG_PROC" "$DRIVE_DIR/"
  sync
  echo "[$(date +%H:%M:%S)] Stage 1 DONE & cached to $DRIVE_DIR/v150_sensenova_process_deltas.npz" | tee -a "$LOG_DRIVER"
fi

PROC_ARGS=()
[ -f "$PROC_NPZ" ] && PROC_ARGS+=(--process-deltas "$PROC_NPZ")

# 5. Stage 2: Train v1.5.0-HeavenLoop (`HeavenBlock-2L`, core_depth=2, <=5.00 MiB Bootable!)
echo "[$(date +%H:%M:%S)] Stage 2: Launching MeguvizCAC $VERSION (core_depth=$CORE_DEPTH, prelude=$PRELUDE_MODE)" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v150_heaven_loop \
  --distill-pool "$DISTILL_POOL" \
  "${PROC_ARGS[@]}" \
  --sibling-dir "$V123_BUNDLE_DIR/sibling" \
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
  --v11-data-dir "$V11_DATA_DIR" \
  --v07-tokenizer "$V07_TOKENIZER" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --prelude-mode "$PRELUDE_MODE" \
  --core-depth "$CORE_DEPTH" \
  --long-char-trigger "${LONG_TRIG:-300}" \
  --p1-steps "${P1_STEPS:-2400}" \
  --p2-steps "${P2_STEPS:-4800}" \
  --p3-qat-steps "${P3_QAT_STEPS:-1800}" \
  --r0-drop-p "${R0_DROP_P:-0.65}" \
  --batch-size "${BS:-128}" 2>&1 | tee "$LOG_TRAIN"

# 6. Stage 3: Sync ALL artifacts + unified release zip + SHA256SUMS + force flush!
cp -f "$OUT_DIR/manifest.json" "$DRIVE_DIR/"
[ -f "$OUT_DIR/charlm_v150.bin" ] && cp -f "$OUT_DIR/charlm_v150.bin" "$DRIVE_DIR/"
cp -f "$OUT_DIR/metrics-${VERSION}.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-bootable-v9-heavenloop.zip" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-5stage-onnx-fp32exact.zip" "$DRIVE_DIR/"
cp -f "$LOG_TRAIN" "$DRIVE_DIR/"

FULL_ZIP="$DRIVE_DIR/meguvizcac-${VERSION}-full.zip"
(cd "$OUT_DIR" && zip -r -6 "$FULL_ZIP" \
  bundle_v9_heavenloop_v150.mgvz charlm_v150.bin bili_picky_bpe_12k.json manifest.json \
  v150_r0_apexchar_fp32.onnx v150_t2_heaven_fp32.onnx v150_t3_heaven_fp32.onnx v150_t4_heaven_fp32.onnx v150_dynaseam_fp32.onnx \
  v150_r0_apexchar_fp32.skel.onnx v150_t2_heaven_fp32.skel.onnx v150_t3_heaven_fp32.skel.onnx v150_t4_heaven_fp32.skel.onnx v150_dynaseam_fp32.skel.onnx \
  "metrics-${VERSION}.json" \
  "meguvizcac-${VERSION}-bootable-v9-heavenloop.zip" \
  "meguvizcac-${VERSION}-5stage-onnx-fp32exact.zip")
(cd "$DRIVE_DIR" && sha256sum "meguvizcac-${VERSION}"-*.zip "metrics-${VERSION}.json" > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'version': '$VERSION', 'core_depth': $CORE_DEPTH, 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.5.0-HeavenLoop DONE] Synced & flushed to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
