#!/usr/bin/env bash
# MeguvizCAC v1.0.0-ApexLoop (`TunnelSymplect x DynaCore-MoLE x 3-Stage Zero-Recompute Cascade`) Colab G4 GPU Driver!
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out_v10_r1}"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v070_omniveritas}"
V08_DRIVE_DIR="${V08_DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v080_zenith}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v100_apexloop}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v10_r1_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v10_r1_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v10_r1_done.json}"
VERSION="${VERSION:-v1.0.0-r1-apexloop-b6}"

V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
TEACHER_V06_CKPT="${TEACHER_V06_CKPT:-$V07_DRIVE_DIR/v0.6.0-r3-polished-teacher_bf16.pt}"
GEMINI_GOLD="${GEMINI_GOLD:-$V08_DRIVE_DIR/data/v08-gemini-gold-labels.jsonl.gz}"
GEMINI_PAIRS="${GEMINI_PAIRS:-$V08_DRIVE_DIR/data/v08-gemini-counterfactual-pairs.jsonl}"

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

if [ ! -f "$V07_TOKENIZER" ] && [ -f "$V07_DRIVE_DIR/meguvizcac-v0.7.0-r2-omniveritas-b4-bundle-v4-sub2bit.zip" ]; then
  mkdir -p /content/_v07_tok_tmp
  python3 -c "import zipfile; zipfile.ZipFile('$V07_DRIVE_DIR/meguvizcac-v0.7.0-r2-omniveritas-b4-bundle-v4-sub2bit.zip').extract('bili_picky_bpe_12k.json', '/content/_v07_tok_tmp')"
  V07_TOKENIZER="/content/_v07_tok_tmp/bili_picky_bpe_12k.json"
fi

EXTRA_ARGS=()
for pf in \
  "$REPO_DIR/meguviz_cac/data/v10-topic-debias-pairs.jsonl" \
  "$DATA_DIR/stage2/stage2-literary-pairs.jsonl" \
  "$DATA_DIR/stage2/qwen38_27b_twins.jsonl" \
  "$DATA_DIR/v04-qwen38-twins.jsonl" \
  "$DATA_DIR/v04-inkling-wk-pairs.jsonl" \
  "$DATA_DIR/co-teacher-gate/v04-twins-clean-pairs.jsonl"
do
  if [ -f "$pf" ]; then
    EXTRA_ARGS+=(--extra-pairs "$pf")
  fi
done

if [ -f "$V07_TOKENIZER" ]; then
  EXTRA_ARGS+=(--v07-tokenizer "$V07_TOKENIZER")
  echo "[$(date +%H:%M:%S)] TOKENIZER BPE12k: $V07_TOKENIZER" | tee -a "$LOG_DRIVER"
fi
if [ -f "$TEACHER_V06_CKPT" ]; then
  EXTRA_ARGS+=(--teacher-v06-ckpt "$TEACHER_V06_CKPT")
  echo "[$(date +%H:%M:%S)] TEACHER_V06 BF16: $TEACHER_V06_CKPT" | tee -a "$LOG_DRIVER"
fi
if [ -f "$GEMINI_GOLD" ]; then
  EXTRA_ARGS+=(--v08-gemini-gold "$GEMINI_GOLD")
  echo "[$(date +%H:%M:%S)] GEMINI_GOLD: $GEMINI_GOLD" | tee -a "$LOG_DRIVER"
fi
if [ -f "$GEMINI_PAIRS" ]; then
  EXTRA_ARGS+=(--v08-gemini-pairs "$GEMINI_PAIRS")
  echo "[$(date +%H:%M:%S)] GEMINI_PAIRS: $GEMINI_PAIRS" | tee -a "$LOG_DRIVER"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [ "${VRAM_MB:-0}" -ge 35000 ]; then
  BS="${BS_OVERRIDE:-512}"
else
  BS="${BS_OVERRIDE:-256}"
fi
echo "[$(date +%H:%M:%S)] START v1.0.0-ApexLoop gpu=\"$GPU_NAME\" vram_mb=$VRAM_MB bs=$BS version=$VERSION" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v10_apexloop \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  "${EXTRA_ARGS[@]}" \
  --intent-cache "$DATA_DIR/v04_semantic_intent_cache.npz" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --max-len 256 \
  --stage1-steps "${STAGE1_STEPS_OVERRIDE:-14000}" \
  --refine-steps "${REFINE_STEPS_OVERRIDE:-2400}" \
  --qat-steps "${QAT_STEPS_OVERRIDE:-2400}" \
  --batch-size "$BS" \
  2>&1 | tee "$LOG_TRAIN"

echo "[$(date +%H:%M:%S)] RUNNING LOFO + Temporal Holdout + Selection-Safe 9-Cell Mondrian Audit (Bundle v6)..." | tee -a "$LOG_DRIVER"
python3 -u -m meguviz_cac.eval_v08_lofo_temporal \
  --bundle-dir "$OUT_DIR" \
  --data "$DATA_FILE" \
  --threads 4 \
  --out "$OUT_DIR/audit-lofo-temporal-$VERSION.json" \
  2>&1 | tee -a "$LOG_TRAIN"

python3 - "$OUT_DIR" "$DRIVE_DIR" "$DONE_JSON" "$VERSION" "$BS" "$LOG_DRIVER" "$LOG_TRAIN" << 'PYEOF' | tee -a "$LOG_DRIVER"
import hashlib, json, shutil, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir, drive_dir, done_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
version, bs_used = sys.argv[4], int(sys.argv[5])
log_drv, log_trn = Path(sys.argv[6]), Path(sys.argv[7])

metrics_p = out_dir / f"metrics-{version}.json"
audit_p = out_dir / f"audit-lofo-temporal-{version}.json"
m = json.loads(metrics_p.read_text(encoding="utf-8"))
rg = m.get("release_gates", {})

full_zip = Path(f"/content/meguvizcac-{version}-full.zip")
with ZipFile(full_zip, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".pt"):
            zf.write(p, p.name)

h = hashlib.sha256(full_zip.read_bytes()).hexdigest()
drive_dir.mkdir(parents=True, exist_ok=True)
drive_zip = drive_dir / full_zip.name
shutil.copy2(full_zip, drive_zip)
for fp in (
    metrics_p,
    audit_p,
    out_dir / f"meguvizcac-{version}-bundle-v6-sub2bit.zip",
    out_dir / f"meguvizcac-{version}-3stage-onnx-int8.zip",
    log_drv,
    log_trn,
):
    if fp.is_file():
        shutil.copy2(fp, drive_dir / fp.name)

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
            "schema_version": "meguviz.bundle.v6",
            "batch_size_used": bs_used,
            "elapsed_seconds": m.get("elapsed_seconds"),
            "bundle_v6_total_mb": m.get("bundle_v6_total_mb"),
            "full_zip": str(full_zip),
            "drive_zip": str(drive_zip),
            "sha256": h,
            "bytes": full_zip.stat().st_size,
            "passed_gates": rg.get("passed_count"),
            "total_gates": rg.get("total_gates"),
            "ci_all_passed": (m.get("v10_ci_gates") or {}).get("all_ci_passed"),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PYEOF
echo "[$(date +%H:%M:%S)] ALL_DONE_AND_SYNCED_TO_DRIVE bs=$BS" | tee -a "$LOG_DRIVER"
