#!/usr/bin/env bash
# MeguvizCAC v1.3.1-NestedBoost Full Nested-Loop Student KD + Clean W4/W8 QAT Driver (Colab G4 RTX PRO 6000)
# Automatically consumes `v130_superteacher_soft_distill_pool.jsonl.gz` (200K 3x24L Super-Teacher logits!)
# + EN/JA multilingual anchors from `/content/data/v04-merged-1m4-r3.jsonl.gz`!
# Produces `<= 5.00 MiB` MGVZ8 bundle (ZERO 1.58-bit!) + 4-Stage Modular FP32-Exact ONNX (`R0`, `T2`, `T3`, `DynaSeam`) + 12/12 CI gates!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V122_BUNDLE_DIR="${V122_BUNDLE_DIR:-/content/v122_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
OUT_DIR="${OUT_DIR:-/content/out_v131_r1}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v130_superteacher}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v131_nested_boost}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v131_r1_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v131_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v131_r1_done.json}"
VERSION="${VERSION:-v1.3.1-r1-nested-boost}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$V122_BUNDLE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Discover Super-Teacher 200K Soft-Logit Distillation Pool (prefer fast local NVMe!)
DISTILL_POOL="${DISTILL_POOL:-/content/out_v130_superteacher/v130_superteacher_soft_distill_pool.jsonl.gz}"
if [ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v130_superteacher_soft_distill_pool.jsonl.gz" ]; then
  echo "[$(date +%H:%M:%S)] Copying Super-Teacher 200K soft pool from Drive..." | tee -a "$LOG_DRIVER"
  cp -f "$ST_DRIVE_DIR/v130_superteacher_soft_distill_pool.jsonl.gz" "/content/v130_superteacher_soft_distill_pool.jsonl.gz"
  DISTILL_POOL="/content/v130_superteacher_soft_distill_pool.jsonl.gz"
fi
if [ ! -f "$DISTILL_POOL" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing Super-Teacher distill pool: $DISTILL_POOL" | tee -a "$LOG_DRIVER"
  exit 2
fi

# 2. Ensure v11 & v122/v123 bundles are unpacked
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_verified.jsonl.gz" ] && [ -f "$V11_BUNDLE_ZIP" ]; then
  python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
fi
if [ ! -f "$V122_BUNDLE_DIR/sibling/sibling_v13_dualaxis_gold.jsonl.gz" ]; then
  for zcand in \
    "/content/v123_all_gold_sibling_and_delta_bundle.zip" \
    "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip" \
    "/content/v122_all_gold_and_sibling_bundle.zip" \
    "$DRIVE_ROOT/v122_all_gold_and_sibling_bundle.zip"; do
    if [ -f "$zcand" ]; then
      python3 -c "import zipfile; zipfile.ZipFile('$zcand').extractall('$V122_BUNDLE_DIR')"
      break
    fi
  done
fi
for dcand in "/content/v123_delta_bundle.zip" "$DRIVE_ROOT/v123_delta_bundle.zip"; do
  if [ -f "$dcand" ]; then
    mkdir -p "$V122_BUNDLE_DIR/delta"
    python3 -c "import zipfile; zipfile.ZipFile('$dcand').extractall('$V122_BUNDLE_DIR/delta')"
    break
  fi
done

# 3. Auto-discover Picky-BPE-12k tokenizer & optional v04 multilingual anchor source
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" "$V11_DRIVE_DIR/bili_picky_bpe_12k.json" "$DRIVE_ROOT/v122_apexnano/bili_picky_bpe_12k.json"; do
    [ -f "$tc" ] && V07_TOKENIZER="$tc" && break
  done
fi

EXTRA_ARGS=()
[ -d "$V122_BUNDLE_DIR/delta" ] && EXTRA_ARGS+=(--delta-dir "$V122_BUNDLE_DIR/delta")
if [ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/v04-merged-1m4-r3.jsonl.gz" ]; then
  V04_DATA="$DRIVE_ROOT/v04-merged-1m4-r3.jsonl.gz"
fi
[ -f "$V04_DATA" ] && EXTRA_ARGS+=(--v04-data "$V04_DATA")

echo "[$(date +%H:%M:%S)] Launching MeguvizCAC $VERSION (NestedBoost <=5.00 MiB) | distill=$DISTILL_POOL | v04=$V04_DATA" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v130_xtreme_apex \
  --distill-pool "$DISTILL_POOL" \
  --sibling-dir "$V122_BUNDLE_DIR/sibling" \
  --gold-v2-dir "$V122_BUNDLE_DIR/gold_v2_w3" \
  --wave3-dir "$V122_BUNDLE_DIR/gold_v2_w3" \
  --v11-data-dir "$V11_DATA_DIR" \
  --v07-tokenizer "$V07_TOKENIZER" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --p1-steps "${P1_STEPS:-2200}" \
  --p2-steps "${P2_STEPS:-4000}" \
  --p3-qat-steps "${P3_QAT_STEPS:-1400}" \
  --batch-size "${BS:-128}" 2>&1 | tee "$LOG_TRAIN"

# 4. Sync artifacts + unified release zip + SHA256SUMS + force flush!
cp -f "$OUT_DIR/manifest.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/metrics-${VERSION}.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-bundle-v8-xtreme-apex.zip" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-4stage-onnx-fp32exact.zip" "$DRIVE_DIR/"
cp -f "$LOG_TRAIN" "$DRIVE_DIR/"

FULL_ZIP="$DRIVE_DIR/meguvizcac-${VERSION}-full.zip"
(cd "$OUT_DIR" && zip -r -6 "$FULL_ZIP" \
  bundle_v8_xtreme_apex_v130.mgvz bili_picky_bpe_12k.json manifest.json \
  v130_r0_apexchar_fp32.onnx v130_t2_nested_fp32.onnx v130_t3_correct_fp32.onnx v130_dynaseam_fp32.onnx \
  "metrics-${VERSION}.json" \
  "meguvizcac-${VERSION}-bundle-v8-xtreme-apex.zip" \
  "meguvizcac-${VERSION}-4stage-onnx-fp32exact.zip")
(cd "$DRIVE_DIR" && sha256sum "meguvizcac-${VERSION}"-*.zip "metrics-${VERSION}.json" > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'version': '$VERSION', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.3.1-NestedBoost Student DONE] Synced & flushed to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
