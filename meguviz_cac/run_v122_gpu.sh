#!/usr/bin/env bash
# MeguvizCAC v1.2.2-ApexNano-Ultra Fresh Colab G4 RTX PRO 6000 / A100 80GB Driver
# Dual-Source Re-Base (`v1.0.0-R1` AI + `v1.2.1-R1` Promo/R0/LongStream) + 203k Sibling Gold + Gold-v2 + Wave-3!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V122_BUNDLE_DIR="${V122_BUNDLE_DIR:-/content/v122_bundle}"
OUT_DIR="${OUT_DIR:-/content/out_v122_r1}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V08_DRIVE_DIR="${V08_DRIVE_DIR:-$DRIVE_ROOT/v080_zenith}"
V10_DRIVE_DIR="${V10_DRIVE_DIR:-$DRIVE_ROOT/v100_apexloop}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
V121_DRIVE_DIR="${V121_DRIVE_DIR:-$DRIVE_ROOT/v121_apexnano}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v122_apexnano}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v122_r1_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v122_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v122_r1_done.json}"
VERSION="${VERSION:-v1.2.2-r1-apexnano-b7}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$V122_BUNDLE_DIR" "$DATA_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Auto-discover base v04 corpus (prefer fast local NVMe!)
SHARED_V04_DRIVE="$DRIVE_ROOT/shared_v04_data"
DATA_FILE="${DATA_FILE:-$DATA_DIR/v04-merged-1m4-r3.jsonl.gz}"
if [ ! -f "$DATA_FILE" ] && [ -f "$SHARED_V04_DRIVE/v04-merged-1m4-r3.jsonl.gz" ]; then
  mkdir -p "$DATA_DIR/stage2" "$DATA_DIR/co-teacher-gate"
  echo "[$(date +%H:%M:%S)] Copying shared_v04_data from Drive -> $DATA_DIR..." | tee -a "$LOG_DRIVER"
  cp -rn "$SHARED_V04_DRIVE/." "$DATA_DIR/" 2>/dev/null || cp -f "$SHARED_V04_DRIVE/v04-merged-1m4-r3.jsonl.gz" "$DATA_FILE"
fi
if [ ! -f "$DATA_FILE" ]; then
  FOUND_V04="$(find /content/drive/MyDrive -maxdepth 4 -name 'v04-merged-1m4-r3.jsonl.gz' 2>/dev/null | head -n 1 || true)"
  [ -n "$FOUND_V04" ] && [ -f "$FOUND_V04" ] && DATA_FILE="$FOUND_V04"
fi
if [ ! -f "$DATA_FILE" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing base corpus: $DATA_FILE" | tee -a "$LOG_DRIVER"
  exit 2
fi

# 2. Auto-unpack v11_omniguard_all_data_bundle.zip
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_all5880.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $V11_BUNDLE_ZIP -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  elif [ -f "/content/v11_omniguard_all_data_bundle.zip" ]; then
    python3 -c "import zipfile; zipfile.ZipFile('/content/v11_omniguard_all_data_bundle.zip').extractall('$V11_DATA_DIR')"
  fi
fi

# 3. Auto-unpack `v122_all_gold_and_sibling_bundle.zip` (Sibling 203k + 26k hygiene + Gold-v2 + Wave-3 + Wave-2.5!)
for zcand in \
  "/content/v122_all_gold_and_sibling_bundle.zip" \
  "$DRIVE_ROOT/v122_all_gold_and_sibling_bundle.zip" \
  "$DRIVE_DIR/v122_all_gold_and_sibling_bundle.zip" \
  "/content/drive/MyDrive/v122_all_gold_and_sibling_bundle.zip"; do
  if [ -f "$zcand" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $zcand -> $V122_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$zcand').extractall('$V122_BUNDLE_DIR')"
    break
  fi
done
if [ ! -f "$V122_BUNDLE_DIR/sibling/sibling_v13_dualaxis_gold.jsonl.gz" ]; then
  FOUND_V122="$(find /content/drive/MyDrive -maxdepth 4 -name 'v122_all_gold_and_sibling_bundle.zip' 2>/dev/null | head -n 1 || true)"
  if [ -n "$FOUND_V122" ] && [ -f "$FOUND_V122" ]; then
    echo "[$(date +%H:%M:%S)] Auto-discovered v122 bundle: $FOUND_V122 -> $V122_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$FOUND_V122').extractall('$V122_BUNDLE_DIR')"
  fi
fi

# 4. Auto-discover `v1.0.0-R1` AI Master + `v1.2.1-R1` Promo/R0/LongStream Master + Tokenizer + Gemini Gold + Diamond Pairs
V10_MASTER="${V10_MASTER:-$(ls "$V10_DRIVE_DIR"/*fp16_master.pt 2>/dev/null | head -n 1 || true)}"
V10_MGVZ="${V10_MGVZ:-$(ls "$V10_DRIVE_DIR"/*.mgvz 2>/dev/null | head -n 1 || true)}"
if [ -z "$V10_MASTER" ] && [ -z "$V10_MGVZ" ]; then
  V10_ZIP="$(ls "$V10_DRIVE_DIR"/*bundle-v6*.zip "$V10_DRIVE_DIR"/*full.zip "$V10_DRIVE_DIR"/*.zip 2>/dev/null | head -n 1 || true)"
  if [ -n "$V10_ZIP" ] && [ -f "$V10_ZIP" ]; then
    mkdir -p /content/v10_unpacked
    python3 -c "import zipfile; zipfile.ZipFile('$V10_ZIP').extractall('/content/v10_unpacked')"
    V10_MGVZ="$(find /content/v10_unpacked -name '*.mgvz' | head -n 1 || true)"
  fi
fi
V121_MASTER="${V121_MASTER:-$V121_DRIVE_DIR/v1.2.1-r1-apexnano-b7_fp16_master.pt}"
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" "$V11_DRIVE_DIR/bili_picky_bpe_12k.json" "$V121_DRIVE_DIR/bili_picky_bpe_12k.json"; do
    [ -f "$tc" ] && V07_TOKENIZER="$tc" && break
  done
fi
GEMINI_GOLD="${GEMINI_GOLD:-$V08_DRIVE_DIR/data/v08-gemini-gold-labels.jsonl.gz}"
GEMINI_PAIRS="${GEMINI_PAIRS:-$V08_DRIVE_DIR/data/v08-gemini-counterfactual-pairs.jsonl}"

EXTRA_ARGS=()
[ -n "$V10_MASTER" ] && [ -f "$V10_MASTER" ] && EXTRA_ARGS+=(--v10-fp16-master "$V10_MASTER")
[ -z "$V10_MASTER" ] && [ -n "$V10_MGVZ" ] && [ -f "$V10_MGVZ" ] && EXTRA_ARGS+=(--v10-bundle-mgvz "$V10_MGVZ")
[ -f "$V121_MASTER" ] && EXTRA_ARGS+=(--v121-fp16-master "$V121_MASTER")
[ -f "$V07_TOKENIZER" ] && EXTRA_ARGS+=(--v07-tokenizer "$V07_TOKENIZER")
[ -f "$GEMINI_GOLD" ] && EXTRA_ARGS+=(--v08-gemini-gold "$GEMINI_GOLD")
[ -f "$GEMINI_PAIRS" ] && EXTRA_ARGS+=(--v08-gemini-pairs "$GEMINI_PAIRS")

# Diamond-only extra pairs (qwen38_twins strictly excluded!)
for pf in \
  "$REPO_DIR/meguviz_cac/data/v10-topic-debias-pairs.jsonl" \
  "$DATA_DIR/stage2/stage2-literary-pairs.jsonl" \
  "$DATA_DIR/v04-inkling-wk-pairs.jsonl" \
  "$DATA_DIR/co-teacher-gate/v04-twins-clean-pairs.jsonl"; do
  [ -f "$pf" ] && EXTRA_ARGS+=(--extra-pairs "$pf")
done

GOLD_SUB="$V122_BUNDLE_DIR/gold_v2_w3"
SIB_SUB="$V122_BUNDLE_DIR/sibling"
[ -f "$GOLD_SUB/v11_wave25_dualaxis_gold_verified.jsonl.gz" ] && EXTRA_ARGS+=(--wave25-gold "$GOLD_SUB/v11_wave25_dualaxis_gold_verified.jsonl.gz")
[ -f "$GOLD_SUB/v12_gold_v2_train.jsonl.gz" ] && EXTRA_ARGS+=(--gold-v2-dir "$GOLD_SUB")
[ -f "$GOLD_SUB/v121_wave3_train.jsonl.gz" ] && EXTRA_ARGS+=(--wave3-dir "$GOLD_SUB")
[ -d "$SIB_SUB" ] && [ -f "$SIB_SUB/sibling_v13_dualaxis_gold.jsonl.gz" ] && EXTRA_ARGS+=(--sibling-dir "$SIB_SUB")

echo "[$(date +%H:%M:%S)] Launching MeguvizCAC $VERSION | extra_args=${EXTRA_ARGS[*]}" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v122_apexnano \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  --v11-data-dir "$V11_DATA_DIR" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --pa0-steps "${PA0_STEPS:-250}" \
  --pa1-steps "${PA1_STEPS:-950}" \
  --pb-steps "${PB_STEPS:-2000}" \
  --pc-w4-steps "${PC_W4_STEPS:-500}" \
  --pc-sub2-steps "${PC_SUB2_STEPS:-1000}" \
  --pc-shock-steps "${PC_SHOCK_STEPS:-250}" 2>&1 | tee "$LOG_TRAIN"

# 5. Always sync artifacts + unified release zip + SHA256SUMS + force-flush Drive!
cp -f "$OUT_DIR/manifest.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/metrics-${VERSION}.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-bundle-v7-sub2bit.zip" "$DRIVE_DIR/"
cp -f "$OUT_DIR/meguvizcac-${VERSION}-5stage-onnx-fp32exact.zip" "$DRIVE_DIR/"
cp -f "$LOG_TRAIN" "$DRIVE_DIR/"

FULL_ZIP="$DRIVE_DIR/meguvizcac-${VERSION}-full.zip"
(cd "$OUT_DIR" && zip -r -6 "$FULL_ZIP" \
  bundle_v7_apexnano_v121.mgvz bili_picky_bpe_12k.json manifest.json \
  v121_r0_charhash_fp32.onnx v121_r1_prefix_fp32.onnx v121_r2_t1_fp32.onnx v121_r3_step_fp32.onnx v121_dynaseam_fp32.onnx \
  "metrics-${VERSION}.json" \
  "meguvizcac-${VERSION}-bundle-v7-sub2bit.zip" \
  "meguvizcac-${VERSION}-5stage-onnx-fp32exact.zip")
(cd "$DRIVE_DIR" && sha256sum "meguvizcac-${VERSION}"-*.zip "metrics-${VERSION}.json" > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'version': '$VERSION', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.2.2 DONE] Synced & flushed to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
