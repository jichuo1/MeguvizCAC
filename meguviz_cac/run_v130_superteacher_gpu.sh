#!/usr/bin/env bash
# MeguvizCAC v1.3.0-ApexKD 3-Model 24L Chinese Super-Teacher Driver (Colab G4 RTX PRO 6000 96GB)
# Trains: chinese-electra-180g-large + chinese-macbert-large + bge-large-zh-v1.5 -> exports 180K Soft-Logit Distill Pool + Top-3000 AL-R4 Queue!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V122_BUNDLE_DIR="${V122_BUNDLE_DIR:-/content/v122_bundle}"
OUT_DIR="${OUT_DIR:-/content/out_v130_superteacher}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v130_superteacher}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v130_st_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v130_st_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v130_st_done.json}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$V122_BUNDLE_DIR" "$DATA_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Ensure v11_omniguard_all_data_bundle.zip is unpacked
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_verified.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $V11_BUNDLE_ZIP -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  fi
fi

# 2. Ensure v122/v123 gold+sibling bundle is unpacked (prefer already-unpacked /content/v122_bundle!)
if [ ! -f "$V122_BUNDLE_DIR/sibling/sibling_v13_dualaxis_gold.jsonl.gz" ]; then
  for zcand in \
    "/content/v123_all_gold_sibling_and_delta_bundle.zip" \
    "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip" \
    "/content/v122_all_gold_and_sibling_bundle.zip" \
    "$DRIVE_ROOT/v122_all_gold_and_sibling_bundle.zip"; do
    if [ -f "$zcand" ]; then
      echo "[$(date +%H:%M:%S)] Unpacking $zcand -> $V122_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
      python3 -c "import zipfile; zipfile.ZipFile('$zcand').extractall('$V122_BUNDLE_DIR')"
      break
    fi
  done
fi

# Optional standalone v123-sibling-delta zip
for dcand in "/content/v123_delta_bundle.zip" "$DRIVE_ROOT/v123_delta_bundle.zip"; do
  if [ -f "$dcand" ]; then
    mkdir -p "$V122_BUNDLE_DIR/delta"
    echo "[$(date +%H:%M:%S)] Unpacking standalone delta $dcand -> $V122_BUNDLE_DIR/delta" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$dcand').extractall('$V122_BUNDLE_DIR/delta')"
    break
  fi
done

# Optional v04 real-wild unlabeled pool (for distillation pool enrichment ONLY — never trained with v04 labels!)
SHARED_V04_DRIVE="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"
V04_FILE="${V04_FILE:-$DATA_DIR/v04-merged-1m4-r3.jsonl.gz}"
if [ ! -f "$V04_FILE" ] && [ -f "$SHARED_V04_DRIVE" ]; then
  V04_FILE="$SHARED_V04_DRIVE"
fi

EXTRA_ARGS=()
[ -d "$V122_BUNDLE_DIR/delta" ] && EXTRA_ARGS+=(--delta-dir "$V122_BUNDLE_DIR/delta")
[ -f "$V04_FILE" ] && EXTRA_ARGS+=(--v04-data "$V04_FILE")

echo "[$(date +%H:%M:%S)] Launching MeguvizCAC v1.3.0 Super-Teacher Ensemble | extra=${EXTRA_ARGS[*]}" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v130_superteacher \
  --sibling-dir "$V122_BUNDLE_DIR/sibling" \
  --gold-v2-dir "$V122_BUNDLE_DIR/gold_v2_w3" \
  --wave3-dir "$V122_BUNDLE_DIR/gold_v2_w3" \
  --v11-data-dir "$V11_DATA_DIR" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --models "${ST_MODELS:-electra_large,macbert_large,bge_large}" \
  --epochs "${ST_EPOCHS:-3}" \
  --batch-size "${ST_BS:-64}" \
  --grad-accum "${ST_ACCUM:-2}" \
  --eval-batch-size "${ST_EVAL_BS:-256}" \
  --resume 2>&1 | tee "$LOG_TRAIN"

# Sync soft distillation pool + AL queue + metrics + log to Drive & force flush!
cp -f "$OUT_DIR/metrics-v1.3.0-superteacher.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/v130_superteacher_soft_distill_pool.jsonl.gz" "$DRIVE_DIR/"
cp -f "$OUT_DIR"/v130_al_disagreement_queue_*.jsonl.gz "$DRIVE_DIR/"
cp -f "$LOG_TRAIN" "$DRIVE_DIR/"
(cd "$DRIVE_DIR" && sha256sum v130_superteacher_soft_distill_pool.jsonl.gz v130_al_disagreement_queue_*.jsonl.gz metrics-v1.3.0-superteacher.json > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'version': 'v1.3.0-superteacher', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.3.0 Super-Teacher DONE] Synced & flushed to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
