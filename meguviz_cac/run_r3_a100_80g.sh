#!/bin/bash
# MeguvizCAC v0.4.0 R3 Full Retrain Driver — A100-SXM4-80GB Saturated Throughput Edition
# - Runs a ~20s worst-case Stage-3 QAT+EMA+Pairs peak-VRAM & torch.compile probe
# - Automatically picks the largest safe batch size fitting <=92% of VRAM (target 768–1024 on 80GB)
# - Pairs with sqrt LR scaling, proportional pair_bs (36–48), 8-worker DataLoader, BF16+TF32+SDPA
# - Uses qat-epochs 5 for bs>=640 (maintaining >=2,000 QAT STE steps & dense SWA snapshots)
# - Immediately packages full archive, verifies SHA256, and syncs to Google Drive upon completion
set -u
cd /content/MeguvizCAC || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/content/MeguvizCAC${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/content/.inductor_cache

DRIVE_OUT="/content/drive/MyDrive/MeguvizCAC/r3_v040"
OUT_DIR="/content/out_v04_r3_2x4_e192"
VERSION="v0.4.0-r3-dualloop-2x4-e192"
FULL_ZIP="/content/meguvizcac-${VERSION}-full.zip"
LOG_FILE="/content/log_r3_full.txt"

COMMON="--data /content/data/v04-merged-1m4-r3.jsonl.gz \
 --qwen-data /content/data/stage2/stage2-fused-train.jsonl.gz \
 --pairs-data /content/data/stage2/stage2-literary-pairs.jsonl \
 --extra-pairs /content/data/stage2/qwen38_27b_twins.jsonl \
 --extra-pairs /content/data/v04-qwen38-twins.jsonl \
 --extra-pairs /content/data/v04-inkling-wk-pairs.jsonl \
 --extra-pairs /content/data/co-teacher-gate/v04-twins-clean-pairs.jsonl \
 --extra-pairs /content/data/co-teacher-gate/v04-gateguard-pairs.jsonl \
 --intent-cache /content/data/v04_semantic_intent_cache.npz"
F_E192="--embed-dim 192 --core-layers 2 --num-loops 4 --normal-max-loops 3 --num-latents 4 --mhc-streams 4 --lora-rank 16 --lora-targets qvo_down"

log() { echo "[$(date +%T)] $*"; }

# 1) Verify R3 data & 92-pair gate-guard integrity
python - <<'PY' || { log "R3_CORPUS_CHECK_FAILED"; exit 1; }
import hashlib, json
from pathlib import Path
rep = json.loads(Path("/content/data/r3_persona_labelfix_report.json").read_text(encoding="utf-8"))
assert rep["modified_rows"] == 196 and rep["non_train_byte_identical"] is True
assert rep["non_train_sha256"] == "6bb3f5bb77d70312d27ef81126583e6d97f9346fdd48ab905bbc462161e87bb5"
h = hashlib.sha256()
with open("/content/data/v04-merged-1m4-r3.jsonl.gz", "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""):
        h.update(c)
assert h.hexdigest() == rep["output_sha256"]
PY
sha256sum /content/data/co-teacher-gate/v04-gateguard-pairs.jsonl | grep -q '^98489d6b' || { log "GATEGUARD_SHA_MISMATCH"; exit 1; }

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
log "PREREQ_VERIFIED commit=$(git -C /content/MeguvizCAC log -1 --format=%h) vram_mb=${VRAM_MB}"

if [ "${VRAM_MB:-0}" -ge 60000 ]; then
  CANDS="1024,896,768,704,640,576,512,384"
  FALLBACK_LADDER=(1024 896 768 704 640 576 512 384 256)
else
  CANDS="384,256,128"
  FALLBACK_LADDER=(384 256 128)
fi

# 2) Fast worst-case Stage-3 QAT+EMA+Pairs VRAM & torch.compile probe
CHOSEN_BS=0
USE_COMPILE=0

log "PROBE_START compile=1 candidates=${CANDS}"
rm -rf /content/_vram_probe
if python -u -m meguviz_cac.train_v04_dualloop --data /content/data/v04-merged-1m4-r3.jsonl.gz $F_E192 \
  --compile --probe-vram "$CANDS" --out /content/_vram_probe > /content/log_vram_probe_compile.txt 2>&1; then
  PROBE_LINE=$(grep 'VRAM_PROBE_JSON ' /content/log_vram_probe_compile.txt | tail -n 1)
  log "$PROBE_LINE"
  CHOSEN_BS=$(python -c "import json,sys; print(json.loads(sys.argv[1].split('VRAM_PROBE_JSON ',1)[1])['chosen_batch_size'])" "$PROBE_LINE" 2>/dev/null || echo 0)
  if [ "${CHOSEN_BS:-0}" -gt 0 ]; then
    USE_COMPILE=1
  fi
else
  log "PROBE_COMPILE_ERROR (falling back to eager SDPA+BF16+TF32)"
  tail -n 10 /content/log_vram_probe_compile.txt
fi

if [ "${CHOSEN_BS:-0}" -le 0 ]; then
  log "PROBE_START compile=0 candidates=${CANDS}"
  rm -rf /content/_vram_probe
  if python -u -m meguviz_cac.train_v04_dualloop --data /content/data/v04-merged-1m4-r3.jsonl.gz $F_E192 \
    --probe-vram "$CANDS" --out /content/_vram_probe > /content/log_vram_probe_eager.txt 2>&1; then
    PROBE_LINE=$(grep 'VRAM_PROBE_JSON ' /content/log_vram_probe_eager.txt | tail -n 1)
    log "$PROBE_LINE"
    CHOSEN_BS=$(python -c "import json,sys; print(json.loads(sys.argv[1].split('VRAM_PROBE_JSON ',1)[1])['chosen_batch_size'])" "$PROBE_LINE" 2>/dev/null || echo 0)
  fi
fi
rm -rf /content/_vram_probe

if [ "${CHOSEN_BS:-0}" -le 0 ]; then
  CHOSEN_BS=512
  [ "${VRAM_MB:-0}" -lt 60000 ] && CHOSEN_BS=384
  log "PROBE_FALLBACK default batch=$CHOSEN_BS compile=$USE_COMPILE"
fi

# Build runtime ladder starting at CHOSEN_BS
LADDER=()
for b in "${FALLBACK_LADDER[@]}"; do
  if [ "$b" -le "$CHOSEN_BS" ]; then
    LADDER+=("$b")
  fi
done
[ "${#LADDER[@]}" -eq 0 ] && LADDER=("$CHOSEN_BS" 512 384 256)

# 3) Launch full R3 training with automatic OOM / compile fallback
TRAIN_OK=0
USED_BS=0
USED_COMPILE=0
for bs in "${LADDER[@]}"; do
  QAT_EP=4
  [ "$bs" -ge 512 ] && QAT_EP=5
  COMP_FLAG=""
  [ "$USE_COMPILE" -eq 1 ] && COMP_FLAG="--compile"

  log "START R3 full retrain batch=$bs qat_epochs=$QAT_EP compile=$USE_COMPILE (epochs 2/1/$QAT_EP + 92 gateguard)"
  rm -rf "$OUT_DIR"
  if python -u -m meguviz_cac.train_v04_dualloop $COMMON $F_E192 $COMP_FLAG \
    --epochs 2 --refine-epochs 1 --qat-epochs "$QAT_EP" \
    --version "$VERSION" --batch-size "$bs" --out "$OUT_DIR" \
    > "$LOG_FILE" 2>&1; then
    log "DONE R3 full retrain batch=$bs qat_epochs=$QAT_EP compile=$USE_COMPILE"
    TRAIN_OK=1
    USED_BS=$bs
    USED_COMPILE=$USE_COMPILE
    break
  fi
  if grep -qi "out of memory" "$LOG_FILE"; then
    cp "$LOG_FILE" "/content/log_r3_oom_bs${bs}.txt"
    log "OOM at batch=$bs (compile=$USE_COMPILE), stepping down"
    continue
  fi
  if [ "$USE_COMPILE" -eq 1 ]; then
    cp "$LOG_FILE" "/content/log_r3_compile_fail_bs${bs}.txt"
    log "Non-OOM failure with --compile; retrying batch=$bs in eager SDPA+BF16+TF32 mode"
    USE_COMPILE=0
    rm -rf "$OUT_DIR"
    if python -u -m meguviz_cac.train_v04_dualloop $COMMON $F_E192 \
      --epochs 2 --refine-epochs 1 --qat-epochs "$QAT_EP" \
      --version "$VERSION" --batch-size "$bs" --out "$OUT_DIR" \
      > "$LOG_FILE" 2>&1; then
      log "DONE R3 full retrain batch=$bs qat_epochs=$QAT_EP compile=0"
      TRAIN_OK=1
      USED_BS=$bs
      USED_COMPILE=0
      break
    fi
    if grep -qi "out of memory" "$LOG_FILE"; then
      cp "$LOG_FILE" "/content/log_r3_oom_eager_bs${bs}.txt"
      log "OOM at batch=$bs (eager), stepping down"
      continue
    fi
  fi
  log "FAILED at batch=$bs (non-OOM)"
  tail -n 30 "$LOG_FILE"
  cp "$LOG_FILE" "$DRIVE_OUT/log_r3_failed.txt" 2>/dev/null || true
  exit 1
done

if [ "$TRAIN_OK" -ne 1 ]; then
  log "ALL_BATCH_SIZES_FAILED"
  exit 1
fi

# 4) Package full artifact archive, verify SHA256, sync to Drive & print R3 scorecard
python - <<'PY'
import hashlib, json, shutil, zipfile
from pathlib import Path

out_dir = Path("/content/out_v04_r3_2x4_e192")
version = "v0.4.0-r3-dualloop-2x4-e192"
full_zip = Path(f"/content/meguvizcac-{version}-full.zip")
drive_out = Path("/content/drive/MyDrive/MeguvizCAC/r3_v040")
drive_out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(full_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(out_dir))

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

zip_sha = sha256_file(full_zip)
zip_size = full_zip.stat().st_size

copied = []
for src in [
    full_zip,
    out_dir / f"metrics-{version}.json",
    out_dir / "manifest.json",
    out_dir / f"meguvizcac-{version}-int8.zip",
    out_dir / f"meguvizcac-{version}-edge-mixed.zip",
    Path("/content/log_r3_full.txt"),
    Path("/content/data/r3_persona_labelfix_report.json"),
]:
    if src.is_file():
        dst = drive_out / src.name
        shutil.copy2(src, dst)
        copied.append((src.name, dst.stat().st_size, sha256_file(dst)))

sha_lines = [f"{s}  {n}" for n, _, s in copied]
(drive_out / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

drive_zip_sha = sha256_file(drive_out / full_zip.name)
assert drive_zip_sha == zip_sha, f"Drive zip SHA mismatch! {drive_zip_sha} != {zip_sha}"

m = json.loads((out_dir / f"metrics-{version}.json").read_text(encoding="utf-8"))
ov = m["onnx_verify"]
print("=== R3 FULL SCORECARD ===")
print("batch_size_used:", m.get("batch_size_used"), "| elapsed_s:", m.get("elapsed_seconds"))
print("ai_traps:", {x["id"]: round(x["prob"], 3) for x in ov["ai_stress_traps"]})
print("human_traps:", {x["id"]: round(x["prob"], 3) for x in ov["human_traps"]})
print("ml_traps:", {x["id"]: round(x["prob"], 3) for x in ov.get("multilingual_traps", [])})
print("hard_trap_gates:", ov["hard_trap_gates"])
print("test_bili_gold:", {k: round(v, 4) for k, v in m["test_bili_gold"].items() if k in ("auroc", "pr_auc", "spearman", "mae", "recall_at_p95", "recall_at_p99")})
print("probe_u001_u017:", {k: round(v, 4) for k, v in m["probe_u001_u017"].items() if k in ("auroc", "ap", "spearman", "mae")})
print("qat_parity_mae:", m["onnx_int8"]["mae_vs_pytorch"])
print("release_gates:", m.get("release_gates"))
print(f"ZIP_READY name={full_zip.name} bytes={zip_size} sha256={zip_sha}")
print(f"DRIVE_VERIFIED dir={drive_out} files={len(copied)}")

done_info = {
    "status": "OK",
    "version": version,
    "batch_size_used": m.get("batch_size_used"),
    "elapsed_seconds": m.get("elapsed_seconds"),
    "full_zip": str(full_zip),
    "drive_zip": str(drive_out / full_zip.name),
    "sha256": zip_sha,
    "bytes": zip_size,
    "passed_gates": (m.get("release_gates") or {}).get("passed_count"),
    "total_gates": (m.get("release_gates") or {}).get("total_gates"),
}
Path("/content/r3_done.json").write_text(json.dumps(done_info, ensure_ascii=False, indent=2), encoding="utf-8")
PY
sync
log "ALL_DONE_AND_SYNCED_TO_DRIVE batch=$USED_BS compile=$USED_COMPILE"
