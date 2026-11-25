#!/usr/bin/env bash
# MeguvizCAC v0.6.0-HyperLoop GPU Driver (G4-96G / H100 / A100 / L4 / T4 auto-adaptive)
# Clean BAAI/bge-small-zh-v1.5 2:2 Interleaved Split-Head Hybrid + GroupDRO + Modern-Human Two-Way pAUC
# + 0 KB Surgery + Frozen 22 KB LB-DSM Critic + 0.155 MB Nano-Draft L2D-FP + Clean-Gold Smooth PAVA + Mandap-B.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
DATA_DIR="${DATA_DIR:-/content/data}"
OUT_DIR="${OUT_DIR:-/content/out_v06_r2}"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/MeguvizCAC/v060_hyperloop}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v06_r2_driver.txt}"
LOG_TRAIN="${LOG_TRAIN:-/content/log_v06_r2_full.txt}"
DONE_JSON="${DONE_JSON:-/content/v06_r2_done.json}"
VERSION="${VERSION:-v0.6.0-r2-hyperloop-b3}"

mkdir -p "$OUT_DIR" "$DRIVE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

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

DATA_FILE="$DATA_DIR/v04-merged-1m4-r3.jsonl.gz"
if [ ! -f "$DATA_FILE" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing $DATA_FILE" | tee -a "$LOG_DRIVER"
  exit 2
fi

WARMSTART_ARGS=()
if [ -n "${V05_WARMSTART_CKPT:-}" ] && [ -f "${V05_WARMSTART_CKPT}" ]; then
  WARMSTART_ARGS+=(--warmstart-v05-r3 "${V05_WARMSTART_CKPT}")
  echo "[$(date +%H:%M:%S)] EXPLICIT_WARMSTART using ${V05_WARMSTART_CKPT}" | tee -a "$LOG_DRIVER"
else
  echo "[$(date +%H:%M:%S)] CLEAN_INIT from pristine BAAI/bge-small-zh-v1.5 (zero era-shortcut contamination!)" | tee -a "$LOG_DRIVER"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [ "${VRAM_MB:-0}" -ge 35000 ]; then
  CAND_BS="${CAND_BS_OVERRIDE:-512,384,256}"
  S1_EPOCHS="${S1_EPOCHS_OVERRIDE:-2}"; MOHAWK_STEPS=350; S2_STEPS=2000; QAT_STEPS=2200
elif [ "${VRAM_MB:-0}" -ge 20000 ]; then
  CAND_BS="${CAND_BS_OVERRIDE:-384,256,192}"
  S1_EPOCHS=2; MOHAWK_STEPS=300; S2_STEPS=1800; QAT_STEPS=2000
else
  CAND_BS="${CAND_BS_OVERRIDE:-192,128,96}"
  S1_EPOCHS=2; MOHAWK_STEPS=250; S2_STEPS=1600; QAT_STEPS=1800
fi
echo "[$(date +%H:%M:%S)] GPU_DETECTED name=\"$GPU_NAME\" vram_mb=$VRAM_MB -> cands=$CAND_BS mohawk=$MOHAWK_STEPS epochs=$S1_EPOCHS s2=$S2_STEPS qat=$QAT_STEPS" | tee -a "$LOG_DRIVER"

echo "[$(date +%H:%M:%S)] PROBING worst-case seq=256 HyperLoop QAT+EMA VRAM on candidates=$CAND_BS" | tee -a "$LOG_DRIVER"
PROBE_OUT="$(python3 -m meguviz_cac.train_v06_hyperloop \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  --embed-dim 192 \
  --max-len 256 \
  --out "$OUT_DIR" \
  --probe-vram "$CAND_BS" 2>&1)"
echo "$PROBE_OUT" | tee -a "$LOG_DRIVER"

BS="$(python3 -c '
import json, sys
for line in sys.stdin:
    if line.startswith("VRAM_PROBE_JSON "):
        d = json.loads(line.split(" ", 1)[1])
        print(int(d["chosen_batch_size"]))
        break
else:
    print(256)
' <<< "$PROBE_OUT")"

echo "[$(date +%H:%M:%S)] START v0.6.0-HyperLoop gpu=\"$GPU_NAME\" batch=$BS version=$VERSION mohawk=$MOHAWK_STEPS s1=${S1_EPOCHS}ep s2=${S2_STEPS}st qat=${QAT_STEPS}st" | tee -a "$LOG_DRIVER"

python3 -u -m meguviz_cac.train_v06_hyperloop \
  --data "$DATA_FILE" \
  --gateguard-data "$REPO_DIR/meguviz_cac/data/v05-gateguard-pairs.jsonl" \
  "${EXTRA_PAIR_ARGS[@]}" \
  "${WARMSTART_ARGS[@]}" \
  --intent-cache "$DATA_DIR/v04_semantic_intent_cache.npz" \
  --out "$OUT_DIR" \
  --checkpoint-dir "$DRIVE_DIR" \
  --version "$VERSION" \
  --backbone BAAI/bge-small-zh-v1.5 \
  --embed-dim 192 \
  --max-len 256 \
  --stage1-synth-cap 15000 \
  --v04-synth-cap -1 \
  --mohawk-steps "$MOHAWK_STEPS" \
  --epochs "$S1_EPOCHS" \
  --stage2-steps "$S2_STEPS" \
  --qat-steps "$QAT_STEPS" \
  --stage2-cap 420000 \
  --qat-cap 380000 \
  --batch-size "$BS" \
  --lr 3.2e-5 \
  --refine-lr 1.4e-5 \
  --qat-lr 5.0e-6 \
  2>&1 | tee "$LOG_TRAIN"

echo "[$(date +%H:%M:%S)] DONE v0.6.0-HyperLoop training batch=$BS" | tee -a "$LOG_DRIVER"

python3 - "$OUT_DIR" "$DRIVE_DIR" "$DONE_JSON" "$VERSION" "$BS" << 'PYEOF' | tee -a "$LOG_DRIVER"
import hashlib, json, shutil, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out_dir = Path(sys.argv[1])
drive_dir = Path(sys.argv[2])
done_json = Path(sys.argv[3])
version = sys.argv[4]
bs_used = int(sys.argv[5])

metrics_p = out_dir / f"metrics-{version}.json"
m = json.loads(metrics_p.read_text(encoding="utf-8"))
v = m.get("onnx_verify", {})
tb = m.get("test_bili_gold", {})
dsh = m.get("test_dsh_gold", {})
ox = m.get("onnx_int8", {})
ed = m.get("edge_w4_w8", {})
rg = m.get("release_gates", {})

print("=== v0.6.0-HyperLoop Bundle v3 FULL SCORECARD ===")
print(f"init_mode: {m.get('init_mode')} | batch_size_used: {bs_used} | elapsed_s: {m.get('elapsed_seconds')} | baked_T: {m.get('temperature')} (raw={m.get('raw_fitted_temperature')})")
print("ai_traps:", {x["id"]: round(x["prob"], 3) for x in v.get("ai_stress_traps", [])})
print("human_traps:", {x["id"]: round(x["prob"], 3) for x in v.get("human_traps", [])})
print("ml_traps:", {x["id"]: round(x["prob"], 3) for x in v.get("multilingual_traps", [])})
print("pife_traps:", {x["id"]: round(x["prob"], 3) for x in v.get("pife_adversarial_traps", [])})
print("test_bili_gold:", {k: tb.get(k) for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95", "recall_at_p99", "spearman", "mae")})
print("test_dsh_gold:", {k: dsh.get(k) for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95", "recall_at_p99", "spearman", "mae")})
print("confounding_surgery:", m.get("confounding_surgery"))
print("lb_dsm_critic:", m.get("lb_dsm_critic"))
print("nano_draft:", m.get("nano_draft"))
print("conformal_mandap_b_alpha03:", (m.get("conformal_mandap_b") or {}).get("alpha_03"))
print("onnx_int8:", ox)
print("edge_w4_w8:", ed)
print("release_gates:", rg)

full_zip = Path(f"/content/meguvizcac-{version}-full.zip")
with ZipFile(full_zip, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and not p.name.endswith(".pt"):
            zf.write(p, p.name)

h = hashlib.sha256(full_zip.read_bytes()).hexdigest()
print(f"ZIP_READY name={full_zip.name} bytes={full_zip.stat().st_size} sha256={h}")

drive_dir.mkdir(parents=True, exist_ok=True)
drive_zip = drive_dir / full_zip.name
shutil.copy2(full_zip, drive_zip)
shutil.copy2(metrics_p, drive_dir / metrics_p.name)
for fn in (f"meguvizcac-{version}-int8.zip", f"meguvizcac-{version}-edge-w4.zip"):
    src_z = out_dir / fn
    if src_z.is_file():
        shutil.copy2(src_z, drive_dir / fn)
for lp in (Path("/content/log_v06_r2_driver.txt"), Path("/content/log_v06_r2_full.txt")):
    if lp.is_file():
        shutil.copy2(lp, drive_dir / lp.name)
sums = []
for f in sorted(drive_dir.iterdir()):
    if f.is_file() and f.name != "SHA256SUMS.txt" and not f.name.endswith(".pt"):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
(drive_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
assert hashlib.sha256(drive_zip.read_bytes()).hexdigest() == h, "Drive zip SHA mismatch!"
print(f"DRIVE_VERIFIED dir={drive_dir} files={len(sums)}")

done_json.write_text(
    json.dumps(
        {
            "status": "OK",
            "version": version,
            "batch_size_used": bs_used,
            "elapsed_seconds": m.get("elapsed_seconds"),
            "full_zip": str(full_zip),
            "drive_zip": str(drive_zip),
            "sha256": h,
            "bytes": full_zip.stat().st_size,
            "passed_gates": rg.get("passed_count"),
            "total_gates": rg.get("total_gates"),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PYEOF

echo "[$(date +%H:%M:%S)] ALL_DONE_AND_SYNCED_TO_DRIVE batch=$BS" | tee -a "$LOG_DRIVER"
