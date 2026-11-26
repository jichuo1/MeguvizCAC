#!/usr/bin/env bash
# MeguvizCAC v1.1.0-OmniGuard Colab A100 80GB Deep Post-Training Driver (Warm-Start from v1.0.0-R1 FP16 Master!)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
OUT_DIR="${OUT_DIR:-/content/out_v11_r1}"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v070_omniveritas}"
V08_DRIVE_DIR="${V08_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v080_zenith}"
V10_DRIVE_DIR="${V10_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v100_apexloop}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v110_omniguard}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v11_r1_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v11_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v11_r1_done.json}"
VERSION="${VERSION:-v1.1.0-r1-omniguard-b6}"

V10_FP16_MASTER="${V10_FP16_MASTER:-$V10_DRIVE_DIR/v1.0.0-r1-apexloop-b6_fp16_master.pt}"
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
GEMINI_GOLD="${GEMINI_GOLD:-$V08_DRIVE_DIR/data/v08-gemini-gold-labels.jsonl.gz}"
GEMINI_PAIRS="${GEMINI_PAIRS:-$V08_DRIVE_DIR/data/v08-gemini-counterfactual-pairs.jsonl}"
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR" "$V11_DATA_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Auto-discover base v04 corpus across /content/data and Google Drive (prefer shared_v04_data synced from A100!)
SHARED_V04_DRIVE="/content/drive/MyDrive/MeguvizCAC/shared_v04_data"
DATA_FILE="${DATA_FILE:-$DATA_DIR/v04-merged-1m4-r3.jsonl.gz}"
if [ ! -f "$DATA_FILE" ] && [ -f "$SHARED_V04_DRIVE/v04-merged-1m4-r3.jsonl.gz" ]; then
  mkdir -p "$DATA_DIR/stage2" "$DATA_DIR/co-teacher-gate"
  echo "[$(date +%H:%M:%S)] Copying shared_v04_data from Drive -> $DATA_DIR (fast local NVMe)..." | tee -a "$LOG_DRIVER"
  cp -rn "$SHARED_V04_DRIVE/." "$DATA_DIR/" 2>/dev/null || cp -f "$SHARED_V04_DRIVE/v04-merged-1m4-r3.jsonl.gz" "$DATA_FILE"
fi
if [ ! -f "$DATA_FILE" ]; then
  for cand in \
    "$SHARED_V04_DRIVE/v04-merged-1m4-r3.jsonl.gz" \
    "/content/drive/MyDrive/MeguvizCAC/data/v04-merged-1m4-r3.jsonl.gz" \
    "/content/drive/MyDrive/MeguvizCAC/v04-merged-1m4-r3.jsonl.gz" \
    "/content/drive/MyDrive/MeguvizCAC/v080_zenith/data/v04-merged-1m4-r3.jsonl.gz" \
    "/content/drive/MyDrive/MeguvizCAC/v070_omniveritas/data/v04-merged-1m4-r3.jsonl.gz" \
    "/content/drive/MyDrive/v04-merged-1m4-r3.jsonl.gz"
  do
    if [ -f "$cand" ]; then
      DATA_FILE="$cand"
      echo "[$(date +%H:%M:%S)] Found base corpus on Drive: $DATA_FILE" | tee -a "$LOG_DRIVER"
      break
    fi
  done
fi
if [ ! -f "$DATA_FILE" ]; then
  FOUND_V04="$(find /content/drive/MyDrive -maxdepth 4 -name 'v04-merged-1m4-r3.jsonl.gz' 2>/dev/null | head -n 1 || true)"
  if [ -n "$FOUND_V04" ] && [ -f "$FOUND_V04" ]; then
    DATA_FILE="$FOUND_V04"
    echo "[$(date +%H:%M:%S)] Auto-discovered base corpus: $DATA_FILE" | tee -a "$LOG_DRIVER"
  fi
fi

for req in "$DATA_FILE" "$V10_FP16_MASTER"; do
  if [ ! -f "$req" ]; then
    echo "[$(date +%H:%M:%S)] ERROR missing required input: $req" | tee -a "$LOG_DRIVER"
    exit 2
  fi
done

# Auto-unpack v11_omniguard_all_data_bundle.zip if present on Drive or /content
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_all5880.jsonl.gz" ]; then
  if [ -f "$V11_BUNDLE_ZIP" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking $V11_BUNDLE_ZIP -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
  elif [ -f "/content/v11_omniguard_all_data_bundle.zip" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking /content/v11_omniguard_all_data_bundle.zip -> $V11_DATA_DIR" | tee -a "$LOG_DRIVER"
    python3 -c "import zipfile; zipfile.ZipFile('/content/v11_omniguard_all_data_bundle.zip').extractall('$V11_DATA_DIR')"
  fi
fi

if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in \
    "/content/out_v10_r1/bili_picky_bpe_12k.json" \
    "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" \
    "$V08_DRIVE_DIR/bili_picky_bpe_12k.json" \
    "$V10_DRIVE_DIR/bili_picky_bpe_12k.json"
  do
    if [ -f "$tc" ]; then
      V07_TOKENIZER="$tc"
      break
    fi
  done
fi

EXTRA_ARGS=()
for pf in \
  "$REPO_DIR/meguviz_cac/data/v10-topic-debias-pairs.jsonl" \
  "$DATA_DIR/stage2/stage2-literary-pairs.jsonl" \
  "$DATA_DIR/stage2/qwen38_27b_twins.jsonl" \
  "$DATA_DIR/v04-qwen38-twins.jsonl" \
  "$DATA_DIR/v04-inkling-wk-pairs.jsonl" \
  "$DATA_DIR/co-teacher-gate/v04-twins-clean-pairs.jsonl" \
  "/content/drive/MyDrive/MeguvizCAC/data/stage2/stage2-literary-pairs.jsonl" \
  "/content/drive/MyDrive/MeguvizCAC/data/stage2/qwen38_27b_twins.jsonl" \
  "/content/drive/MyDrive/MeguvizCAC/data/v04-qwen38-twins.jsonl" \
  "/content/drive/MyDrive/MeguvizCAC/data/v04-inkling-wk-pairs.jsonl" \
  "/content/drive/MyDrive/MeguvizCAC/data/co-teacher-gate/v04-twins-clean-pairs.jsonl"
do
  [ -f "$pf" ] && EXTRA_ARGS+=(--extra-pairs "$pf")
done
[ -f "$V07_TOKENIZER" ] && EXTRA_ARGS+=(--v07-tokenizer "$V07_TOKENIZER")
[ -f "$GEMINI_GOLD" ] && EXTRA_ARGS+=(--v08-gemini-gold "$GEMINI_GOLD")
[ -f "$GEMINI_PAIRS" ] && EXTRA_ARGS+=(--v08-gemini-pairs "$GEMINI_PAIRS")
if [ -z "${WAVE25_GOLD:-}" ]; then
  for w25c in \
    "$V11_DATA_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz" \
    "/content/v11_wave25_dualaxis_gold_verified.jsonl.gz" \
    "$DRIVE_DIR/v11_wave25_dualaxis_gold_verified.jsonl.gz"
  do
    if [ -f "$w25c" ]; then
      WAVE25_GOLD="$w25c"
      break
    fi
  done
fi
[ -n "${WAVE25_GOLD:-}" ] && [ -f "$WAVE25_GOLD" ] && EXTRA_ARGS+=(--wave25-gold "$WAVE25_GOLD")

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
BS="${BS_OVERRIDE:-512}"
if [ "${VRAM_MB:-0}" -lt 35000 ]; then
  BS="${BS_OVERRIDE:-192}"
fi

echo "[$(date +%H:%M:%S)] START v1.1.0-OmniGuard gpu=\"$GPU_NAME\" vram_mb=$VRAM_MB bs=$BS version=$VERSION v10_master=$V10_FP16_MASTER" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v11_omniguard \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  --v10-fp16-master "$V10_FP16_MASTER" \
  --v11-data-dir "$V11_DATA_DIR" \
  "${EXTRA_ARGS[@]}" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --max-len 256 \
  --batch-size "$BS" \
  --s1a-steps "${S1A_STEPS_OVERRIDE:-2400}" \
  --s1b-steps "${S1B_STEPS_OVERRIDE:-0}" \
  --s15-steps "${S15_STEPS_OVERRIDE:-600}" \
  --s2a-steps "${S2A_STEPS_OVERRIDE:-600}" \
  --s2b-steps "${S2B_STEPS_OVERRIDE:-1000}" \
  --s3a-steps "${S3A_STEPS_OVERRIDE:-300}" \
  2>&1 | tee "$LOG_TRAIN"

python3 - "$OUT_DIR" "$DRIVE_DIR" "$DONE_JSON" "$VERSION" "$BS" "$LOG_DRIVER" "$LOG_TRAIN" << 'PYEOF' | tee -a "$LOG_DRIVER"
import hashlib, json, shutil, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir, drive_dir, done_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
version, bs_used = sys.argv[4], int(sys.argv[5])
log_drv, log_trn = Path(sys.argv[6]), Path(sys.argv[7])

metrics_p = out_dir / f"metrics-{version}.json"
m = json.loads(metrics_p.read_text(encoding="utf-8"))

full_zip = Path(f"/content/meguvizcac-{version}-full.zip")
with ZipFile(full_zip, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".pt"):
            zf.write(p, p.name)
    for lp in (log_drv, log_trn):
        if lp.is_file():
            zf.write(lp, lp.name)

sha_lines = []
for fn in (f"meguvizcac-{version}-bundle-v6-sub2bit.zip", f"meguvizcac-{version}-4stage-onnx-int8.zip", f"metrics-{version}.json", "manifest.json"):
    src = out_dir / fn
    if src.is_file():
        dst = drive_dir / fn
        shutil.copy2(src, dst)
        sha_lines.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {dst.name}")
dst_full = drive_dir / full_zip.name
shutil.copy2(full_zip, dst_full)
sha_lines.append(f"{hashlib.sha256(dst_full.read_bytes()).hexdigest()}  {dst_full.name}")
(drive_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

done_obj = {
    "status": "DONE",
    "version": version,
    "batch_size": bs_used,
    "bundle_v6_mb": m.get("bundle_v6_total_mb"),
    "test_bili_gold": m.get("test_bili_gold"),
    "test_dsh_gold": m.get("test_dsh_gold"),
    "promobench48": m.get("promobench48"),
    "promo_val_select_and_longform120": m.get("promo_val_select_and_longform120"),
    "release_gates": m.get("release_gates"),
    "drive_dir": str(drive_dir),
}
done_json.write_text(json.dumps(done_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(done_obj, ensure_ascii=False, indent=2))
PYEOF
echo "[$(date +%H:%M:%S)] ALL DONE v1.1.0-OmniGuard! Synced to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
