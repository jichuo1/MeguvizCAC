#!/usr/bin/env bash
# MeguvizCAC v0.7.0-OmniVeritas GPU Driver (Bundle v4 <= 2.20 MB 5-Trit/2-Bit + ONNX INT8 + V7-R0..R4 CI)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out_v07_r2}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v070_omniveritas}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v07_r2_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v07_r2_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v07_r2_done.json}"
VERSION="${VERSION:-v0.7.0-r2-omniveritas-b4}"
TEACHER_V06_CKPT="${TEACHER_V06_CKPT:-/content/drive/MyDrive/MeguvizCAC/v060_hyperloop/v0.6.0-r2-hyperloop-b3_stage1_ckpt.pt}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATA_FILE="$DATA_DIR/v04-merged-1m4-r3.jsonl.gz"
if [ ! -f "$DATA_FILE" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing $DATA_FILE" | tee -a "$LOG_DRIVER"
  exit 2
fi

EXTRA_PAIR_ARGS=()
for pf in \
  "$DATA_DIR/stage2/stage2-literary-pairs.jsonl" \
  "$DATA_DIR/stage2/qwen38_27b_twins.jsonl" \
  "$DATA_DIR/v04-qwen38-twins.jsonl" \
  "$DATA_DIR/v04-inkling-wk-pairs.jsonl" \
  "$DATA_DIR/co-teacher-gate/v04-twins-clean-pairs.jsonl"
do
  if [ -f "$pf" ]; then
    EXTRA_PAIR_ARGS+=(--extra-pairs "$pf")
  fi
done

TEACHER_ARGS=()
if [ -n "$TEACHER_V06_CKPT" ] && [ -f "$TEACHER_V06_CKPT" ]; then
  TEACHER_ARGS+=(--teacher-v06-ckpt "$TEACHER_V06_CKPT")
  echo "[$(date +%H:%M:%S)] STAGE0_TEACHER using unquantized BF16 v0.6 master: $TEACHER_V06_CKPT" | tee -a "$LOG_DRIVER"
else
  echo "[$(date +%H:%M:%S)] STAGE0_TEACHER not found -> training directly from 4-teacher fused gold targets" | tee -a "$LOG_DRIVER"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [ "${VRAM_MB:-0}" -ge 35000 ]; then
  BS="${BS_OVERRIDE:-512}"
else
  BS="${BS_OVERRIDE:-256}"
fi
echo "[$(date +%H:%M:%S)] START v0.7.0-OmniVeritas gpu=\"$GPU_NAME\" vram_mb=$VRAM_MB bs=$BS version=$VERSION" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v07_omniveritas \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  "${EXTRA_PAIR_ARGS[@]}" \
  "${TEACHER_ARGS[@]}" \
  --intent-cache "$DATA_DIR/v04_semantic_intent_cache.npz" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --max-len 256 \
  --stage1-synth-cap 15000 \
  --v04-synth-cap -1 \
  --epochs "${EPOCHS_OVERRIDE:-6}" \
  --stage15-steps "${STAGE15_STEPS_OVERRIDE:-1800}" \
  --stage15-lr "${STAGE15_LR_OVERRIDE:-1.1e-4}" \
  --teacher-v06-polish-steps "${TEACHER_POLISH_STEPS:-900}" \
  --upq-cap 480000 \
  --w4-steps "${W4_STEPS_OVERRIDE:-1200}" \
  --sub2-steps "${SUB2_STEPS_OVERRIDE:-2200}" \
  --batch-size "$BS" \
  --lr "${LR_OVERRIDE:-7.5e-4}" \
  --w4-lr "${W4_LR_OVERRIDE:-8.5e-5}" \
  --sub2-lr "${SUB2_LR_OVERRIDE:-4.5e-5}" \
  2>&1 | tee "$LOG_TRAIN"

echo "[$(date +%H:%M:%S)] DONE v0.7.0-OmniVeritas training bs=$BS" | tee -a "$LOG_DRIVER"

python3 - "$OUT_DIR" "$DRIVE_DIR" "$DONE_JSON" "$VERSION" "$BS" "$LOG_DRIVER" "$LOG_TRAIN" << 'PYEOF' | tee -a "$LOG_DRIVER"
import hashlib, json, shutil, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir = Path(sys.argv[1])
drive_dir = Path(sys.argv[2])
done_json = Path(sys.argv[3])
version = sys.argv[4]
bs_used = int(sys.argv[5])
log_drv = Path(sys.argv[6])
log_trn = Path(sys.argv[7])

metrics_p = out_dir / f"metrics-{version}.json"
m = json.loads(metrics_p.read_text(encoding="utf-8"))
rg = m.get("release_gates", {})

print("=== v0.7.0-OmniVeritas Bundle v4 FULL SCORECARD ===")
print("bundle_v4_total_mb:", m.get("bundle_v4_total_mb"), "mgvz_breakdown_kb:", (m.get("export_info") or {}).get("bundle_v4_mgvz", {}).get("breakdown_kb"))
print("fp16_master_eval:", m.get("fp16_master_eval"))
print("sub2bit_test_bili_gold:", m.get("test_bili_gold"))
print("test_dsh_gold:", m.get("test_dsh_gold"))
print("onnx_int8:", m.get("onnx_int8"))
print("traps_summary:", (m.get("traps") or {}).get("summary"))
print("v07_ci_gates:", m.get("v07_ci_gates"))
print("release_gates:", rg)

full_zip = Path(f"/content/meguvizcac-{version}-full.zip")
with ZipFile(full_zip, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".pt"):
            zf.write(p, p.name)

h = hashlib.sha256(full_zip.read_bytes()).hexdigest()
drive_dir.mkdir(parents=True, exist_ok=True)
drive_zip = drive_dir / full_zip.name
shutil.copy2(full_zip, drive_zip)
shutil.copy2(metrics_p, drive_dir / metrics_p.name)
for fn in (f"meguvizcac-{version}-bundle-v4-sub2bit.zip", f"meguvizcac-{version}-onnx-int8.zip"):
    src_z = out_dir / fn
    if src_z.is_file():
        shutil.copy2(src_z, drive_dir / fn)
for lp in (log_drv, log_trn):
    if lp.is_file():
        shutil.copy2(lp, drive_dir / lp.name)

sums = []
for f in sorted(drive_dir.iterdir()):
    if f.is_file() and f.name != "SHA256SUMS.txt" and not f.name.endswith(".pt"):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
(drive_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
assert hashlib.sha256(drive_zip.read_bytes()).hexdigest() == h, "Drive zip SHA mismatch!"
print(f"DRIVE_VERIFIED dir={drive_dir} files={len(sums)} sha256={h}")

done_json.write_text(
    json.dumps(
        {
            "status": "OK",
            "version": version,
            "batch_size_used": bs_used,
            "elapsed_seconds": m.get("elapsed_seconds"),
            "bundle_v4_total_mb": m.get("bundle_v4_total_mb"),
            "full_zip": str(full_zip),
            "drive_zip": str(drive_zip),
            "sha256": h,
            "bytes": full_zip.stat().st_size,
            "passed_gates": rg.get("passed_count"),
            "total_gates": rg.get("total_gates"),
            "ci_all_passed": (m.get("v07_ci_gates") or {}).get("all_ci_passed"),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PYEOF
echo "[$(date +%H:%M:%S)] ALL_DONE_AND_SYNCED_TO_DRIVE bs=$BS" | tee -a "$LOG_DRIVER"
