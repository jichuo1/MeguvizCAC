#!/usr/bin/env bash
# MeguvizCAC v1.4.0-SuperTeacher-Ultra Driver (Colab G4 RTX PRO 6000 96 GB Blackwell)
# Trains 3-Paradigm Super-Teacher Ensemble anchored by SenseTime SenseNova-Loop-9B (56 effective looped layers + mHC):
#   electra_large (325M RTD) + bge_m3 (568M ML Contrastive) + sensenova_loop_9b (9.0B 56L Looped Hybrid-Attn + mHC LoRA)
# Ingests ALL v122-sibling + newly merged +57,585 v123-sibling-delta + 15,903 CF pairs + EN/JA gold!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
OUT_DIR="${OUT_DIR:-/content/out_v140_superteacher}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v140_st_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v140_st_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v140_st_done.json}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR" "$V123_BUNDLE_DIR" "$DATA_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 0. Ensure modern transformers + accelerate + safetensors + Triton flash-linear-attention (5-10x faster GatedDeltaNet!)
if ! python3 -c "import transformers as t, accelerate, safetensors; from packaging import version; assert version.parse(t.__version__) >= version.parse('4.51.0')" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] Installing/upgrading transformers + accelerate + safetensors for SenseNova-Loop-9B..." | tee -a "$LOG_DRIVER"
  pip install -q -U "transformers>=4.51.0" accelerate safetensors
fi
if ! python3 -c "import fla" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] Installing pure-Triton flash-linear-attention (accelerates SenseNova-Loop-9B GatedDeltaNet 5-10x)..." | tee -a "$LOG_DRIVER"
  pip install -q flash-linear-attention || true
fi

# 1. Ensure v11_omniguard_all_data_bundle.zip is unpacked
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_verified.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $V11_BUNDLE_ZIP -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  fi
fi

# 2. Prefer unified v123_all_gold_sibling_and_delta_bundle.zip (strips all 60 OjinjinBench hits + bundles +57.5K delta!)
UNIFIED_ZIP=""
for zcand in \
  "/content/v123_all_gold_sibling_and_delta_bundle.zip" \
  "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip" \
  "$DRIVE_DIR/v123_all_gold_sibling_and_delta_bundle.zip"; do
  if [ -f "$zcand" ]; then
    UNIFIED_ZIP="$zcand"
    break
  fi
done

if [ -n "$UNIFIED_ZIP" ]; then
  if [ ! -f "$V123_BUNDLE_DIR/delta/delta_ext_v4B_and_v123_real_dualaxis_gold.jsonl.gz" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking unified v123 zero-leak bundle $UNIFIED_ZIP -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$UNIFIED_ZIP').extractall('$V123_BUNDLE_DIR')"
  fi
else
  # Fallback: reuse existing /content/v122_bundle base + overlay standalone v123_delta_bundle.zip
  if [ -d "/content/v122_bundle/sibling" ] && [ ! -d "$V123_BUNDLE_DIR/sibling" ]; then
    ln -sfn /content/v122_bundle/sibling "$V123_BUNDLE_DIR/sibling"
    ln -sfn /content/v122_bundle/gold_v2_w3 "$V123_BUNDLE_DIR/gold_v2_w3"
  elif [ ! -f "$V123_BUNDLE_DIR/sibling/sibling_v13_dualaxis_gold.jsonl.gz" ]; then
    for zcand in "/content/v122_all_gold_and_sibling_bundle.zip" "$DRIVE_ROOT/v122_all_gold_and_sibling_bundle.zip"; do
      if [ -f "$zcand" ]; then
        echo "[$(date +%H:%M:%S)] Unpacking base bundle $zcand -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
        python3 -c "import zipfile; zipfile.ZipFile('$zcand').extractall('$V123_BUNDLE_DIR')"
        break
      fi
    done
  fi
  for dcand in "/content/v123_delta_bundle.zip" "$DRIVE_ROOT/v123_delta_bundle.zip" "$DRIVE_DIR/v123_delta_bundle.zip"; do
    if [ -f "$dcand" ]; then
      if [ ! -f "$V123_BUNDLE_DIR/delta/delta_ext_v4B_and_v123_real_dualaxis_gold.jsonl.gz" ]; then
        mkdir -p "$V123_BUNDLE_DIR/delta"
        echo "[$(date +%H:%M:%S)] Unpacking standalone v123 delta $dcand -> $V123_BUNDLE_DIR/delta" | tee -a "$LOG_DRIVER"
        python3 -c "import zipfile; zipfile.ZipFile('$dcand').extractall('$V123_BUNDLE_DIR/delta')"
      fi
      break
    fi
  done
fi

# Guard: warn loudly if v123-sibling-delta is missing!
if [ ! -f "$V123_BUNDLE_DIR/delta/delta_ext_v4B_and_v123_real_dualaxis_gold.jsonl.gz" ]; then
  echo "[WARNING] v123-sibling-delta NOT found in $V123_BUNDLE_DIR/delta! Upload v123_all_gold_sibling_and_delta_bundle.zip (or v123_delta_bundle.zip) to $DRIVE_ROOT first!" | tee -a "$LOG_DRIVER"
fi

# 3. Wire v04 corpus (provides clean EN/JA multilingual gold anchors + 45K real-wild unlabeled distill pool!)
SHARED_V04_DRIVE="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"
V04_FILE="${V04_FILE:-$DATA_DIR/v04-merged-1m4-r3.jsonl.gz}"
if [ ! -f "$V04_FILE" ] && [ -f "$SHARED_V04_DRIVE" ]; then
  V04_FILE="$SHARED_V04_DRIVE"
fi

EXTRA_ARGS=()
[ -d "$V123_BUNDLE_DIR/delta" ] && EXTRA_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
[ -f "$V04_FILE" ] && EXTRA_ARGS+=(--v04-data "$V04_FILE")
[ -n "${ST_EXTRA_DELTAS:-}" ] && EXTRA_ARGS+=(--extra-delta-dirs "$ST_EXTRA_DELTAS")

echo "[$(date +%H:%M:%S)] Launching MeguvizCAC v1.4.0-SuperTeacher-Ultra | models=${ST_MODELS:-electra_large,bge_m3,sensenova_loop_9b} | extra=${EXTRA_ARGS[*]}" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v140_superteacher \
  --sibling-dir "$V123_BUNDLE_DIR/sibling" \
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3" \
  --v11-data-dir "$V11_DATA_DIR" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --models "${ST_MODELS:-electra_large,bge_m3,sensenova_loop_9b}" \
  --epochs "${ST_EPOCHS:-3}" \
  --max-len "${ST_MAX_LEN:-288}" \
  --head-len "${ST_HEAD_LEN:-192}" \
  --cf-every "${ST_CF_EVERY:-3}" \
  --cf-pairs-per-step "${ST_CF_PAIRS:-8}" \
  --cf-weight "${ST_CF_WEIGHT:-0.30}" \
  --al-top-k "${ST_AL_TOPK:-5000}" \
  --resume 2>&1 | tee "$LOG_TRAIN"

# 4. Sync soft distillation pool + AL-R5 queue + metrics + log to Drive & force flush!
cp -f "$OUT_DIR/metrics-v1.4.0-superteacher.json" "$DRIVE_DIR/"
cp -f "$OUT_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "$DRIVE_DIR/"
cp -f "$OUT_DIR"/v140_al_disagreement_queue_*.jsonl.gz "$DRIVE_DIR/"
cp -f "$LOG_TRAIN" "$DRIVE_DIR/"
(cd "$DRIVE_DIR" && sha256sum v140_superteacher_soft_distill_pool.jsonl.gz v140_al_disagreement_queue_*.jsonl.gz metrics-v1.4.0-superteacher.json > SHA256SUMS.txt && cat SHA256SUMS.txt)
touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'version': 'v1.4.0-superteacher-ultra', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.4.0-SuperTeacher-Ultra DONE] Synced & flushed to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
