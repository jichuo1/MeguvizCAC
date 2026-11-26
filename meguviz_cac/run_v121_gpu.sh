#!/usr/bin/env bash
# MeguvizCAC v1.2.1-ApexNano-Fix Colab G4 RTX PRO 6000 / A100 80GB 3-Phase Driver
# Warm-Start from v1.1.0-R1 FP16 Master + Gold-v2 + Wave-3 (Wild Promo P1-P5 & Frontier A1-A3)!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
GOLD_V2_DIR="${GOLD_V2_DIR:-/content/gold_v2_w3}"
OUT_DIR="${OUT_DIR:-/content/out_v121_r1}"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v070_omniveritas}"
V08_DRIVE_DIR="${V08_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v080_zenith}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v110_omniguard}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v121_apexnano}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v121_r1_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v121_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v121_r1_done.json}"
VERSION="${VERSION:-v1.2.1-r1-apexnano-b7}"

V11_FP16_MASTER="${V11_FP16_MASTER:-$V11_DRIVE_DIR/v1.1.0-r1-omniguard-b6_fp16_master.pt}"
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
GEMINI_GOLD="${GEMINI_GOLD:-$V08_DRIVE_DIR/data/v08-gemini-gold-labels.jsonl.gz}"
GEMINI_PAIRS="${GEMINI_PAIRS:-$V08_DRIVE_DIR/data/v08-gemini-counterfactual-pairs.jsonl}"
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$GOLD_V2_DIR"
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
  if [ -n "$FOUND_V04" ] && [ -f "$FOUND_V04" ]; then
    DATA_FILE="$FOUND_V04"
    echo "[$(date +%H:%M:%S)] Auto-discovered base corpus: $DATA_FILE" | tee -a "$LOG_DRIVER"
  fi
fi

for req in "$DATA_FILE" "$V11_FP16_MASTER"; do
  if [ ! -f "$req" ]; then
    echo "[$(date +%H:%M:%S)] ERROR missing required input: $req" | tee -a "$LOG_DRIVER"
    exit 2
  fi
done

# 2. Auto-unpack v11_omniguard_all_data_bundle.zip
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_all5880.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $V11_BUNDLE_ZIP -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  elif [ -f "/content/v11_omniguard_all_data_bundle.zip" ]; then
    python3 -c "import zipfile; zipfile.ZipFile('/content/v11_omniguard_all_data_bundle.zip').extractall('$V11_DATA_DIR')"
  fi
fi

# 3. Auto-unpack single-file `v121_gold_and_wave3_bundle.zip` (Gold-v2 + Wave-3 + Wave-2.5!)
for w3zip in \
  "$DRIVE_ROOT/v121_gold_and_wave3_bundle.zip" \
  "$DRIVE_DIR/v121_gold_and_wave3_bundle.zip" \
  "/content/v121_gold_and_wave3_bundle.zip" \
  "/content/drive/MyDrive/v121_gold_and_wave3_bundle.zip"; do
  if [ -f "$w3zip" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $w3zip -> $GOLD_V2_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$w3zip').extractall('$GOLD_V2_DIR')"
    break
  fi
done

# Fallback: also search Drive for v121_gold_and_wave3_bundle.zip if not unpacked yet
if [ ! -f "$GOLD_V2_DIR/v121_wave3_train.jsonl.gz" ]; then
  FOUND_W3="$(find /content/drive/MyDrive -maxdepth 4 -name 'v121_gold_and_wave3_bundle.zip' 2>/dev/null | head -n 1 || true)"
  if [ -n "$FOUND_W3" ] && [ -f "$FOUND_W3" ]; then
    echo "[$(date +%H:%M:%S)] Auto-discovered Wave-3 bundle: $FOUND_W3 -> $GOLD_V2_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$FOUND_W3').extractall('$GOLD_V2_DIR')"
  fi
fi

# 4. Auto-discover tokenizer, Wave-2.5, Gold-v2, Wave-3, and DIAMOND-ONLY pair datasets!
if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in \
    "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" \
    "$V08_DRIVE_DIR/bili_picky_bpe_12k.json" \
    "$V11_DRIVE_DIR/bili_picky_bpe_12k.json"; do
    [ -f "$tc" ] && V07_TOKENIZER="$tc" && break
  done
fi

EXTRA_ARGS=()
# Diamond-only extra pairs (`v04-qwen38-twins.jsonl` & `qwen38_27b_twins.jsonl` EXCLUDED by 9/9 pair audit!)
for pf in \
  "$REPO_DIR/meguviz_cac/data/v10-topic-debias-pairs.jsonl" \
  "$DATA_DIR/stage2/stage2-literary-pairs.jsonl" \
  "$DATA_DIR/v04-inkling-wk-pairs.jsonl" \
  "$DATA_DIR/co-teacher-gate/v04-twins-clean-pairs.jsonl"; do
  [ -f "$pf" ] && EXTRA_ARGS+=(--extra-pairs "$pf")
done
[ -f "$V07_TOKENIZER" ] && EXTRA_ARGS+=(--v07-tokenizer "$V07_TOKENIZER")
[ -f "$GEMINI_GOLD" ] && EXTRA_ARGS+=(--v08-gemini-gold "$GEMINI_GOLD")
[ -f "$GEMINI_PAIRS" ] && EXTRA_ARGS+=(--v08-gemini-pairs "$GEMINI_PAIRS")

if [ -f "$GOLD_V2_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz" ]; then
  EXTRA_ARGS+=(--wave25-gold "$GOLD_V2_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz")
elif [ -f "$V11_DATA_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz" ]; then
  EXTRA_ARGS+=(--wave25-gold "$V11_DATA_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz")
fi
[ -f "$GOLD_V2_DIR/v12_gold_v2_train.jsonl.gz" ] && EXTRA_ARGS+=(--gold-v2-dir "$GOLD_V2_DIR")
[ -f "$GOLD_V2_DIR/v121_wave3_train.jsonl.gz" ] && EXTRA_ARGS+=(--wave3-dir "$GOLD_V2_DIR")

echo "[$(date +%H:%M:%S)] Launching MeguvizCAC $VERSION | extra_args=${EXTRA_ARGS[*]}" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v121_apexnano \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  --v11-fp16-master "$V11_FP16_MASTER" \
  --v11-data-dir "$V11_DATA_DIR" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --pa-steps "${PA_STEPS:-1400}" \
  --pb-steps "${PB_STEPS:-2400}" \
  --pc-w4-steps "${PC_W4_STEPS:-600}" \
  --pc-sub2-steps "${PC_SUB2_STEPS:-1000}" \
  --pc-shock-steps "${PC_SHOCK_STEPS:-300}" 2>&1 | tee "$LOG_TRAIN"

# 5. Sync artifacts + unified release zip + SHA256SUMS to Drive
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
python3 -c "import json, time; json.dump({'version': '$VERSION', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.2.1 DONE] Synced to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
