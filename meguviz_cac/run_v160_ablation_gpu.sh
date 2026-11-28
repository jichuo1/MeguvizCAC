#!/usr/bin/env bash
# MeguvizCAC v1.6.0-StratosphereLoop Remote Colab G4 (RTX PRO 6000 Blackwell 96GB) Locked A0-A5 Ablation Suite
# Implements External GPT Conditional-Approval Unconfounded 6-Arm Matrix:
#   - ALL 6 arms lock IDENTICAL: frozen Phase-1 R0 (shared_r0_frozen.pt), frozen Engram buckets, identical seed/data/teacher,
#     single FFN per sub-block, exact 1/6 uniform Anytime depth weighting (T~U{1..6}), stop-grad L_mono, NO Slot-9 d_t injection,
#     NO survival reweighting, and IDENTICAL Phase-3 LoopQ W4/W8 QAT schedule!
#   - Every arm saves FP16 master + Direct-Packed MGVZ10 + Post-QAT MGVZ10 + offline prediction cache (.npz) + all 6 GPT probes!
#   - Arms:
#       A0: 1L/512 (v160_a1) + legacy v150_anchor (anchor/rho/kappa; step 5/6 reuses step 4), no Fourier, no NCL
#       A1: A0 -> ONLY switch to bounded projected residual (Pi_R + causal h_t = h_0/sqrt(t)), no Fourier, no NCL
#       A2: A1 -> ONLY add continuous Fourier step embedding phi(t)
#       A3: A1 -> ONLY add NCL (identical sample weights, no Slot-9 injection)
#       A4: A1 -> ONLY scale FFN 512 -> 640 (v160_a2)
#       A5: A4 -> ONLY scale Prelude 1L -> 2L (v160_aprime)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/MeguvizCAC}"
V11_DATA_DIR="${V11_DATA_DIR:-/content/v11_data}"
V123_BUNDLE_DIR="${V123_BUNDLE_DIR:-/content/v123_bundle}"
V04_DATA="${V04_DATA:-/content/data/v04-merged-1m4-r3.jsonl.gz}"
OUT_ROOT="${OUT_ROOT:-/content/out_v160_ablation}"
DRIVE_ROOT="/content/drive/MyDrive/MeguvizCAC"
V07_DRIVE_DIR="${V07_DRIVE_DIR:-$DRIVE_ROOT/v070_omniveritas}"
V11_DRIVE_DIR="${V11_DRIVE_DIR:-$DRIVE_ROOT/v110_omniguard}"
ST_DRIVE_DIR="${ST_DRIVE_DIR:-$DRIVE_ROOT/v140_superteacher}"
DRIVE_DIR="${DRIVE_DIR:-$DRIVE_ROOT/v160_stratosphere_loop_r1}"
LOG_DRIVER="${LOG_DRIVER:-/content/log_v160_ablation_driver.txt}"
DONE_JSON="${DONE_JSON:-/content/v160_ablation_done.json}"
ARMS="${ARMS:-A0 A1 A2 A3 A4 A5}"
FULL_BUDGET="${FULL_BUDGET:-0}"
BS="${BS:-128}"
SEED="${SEED:-20261206}"
SHARED_R0="${SHARED_R0:-$OUT_ROOT/shared_r0_frozen.pt}"

mkdir -p "$OUT_ROOT" "$DRIVE_DIR" "$V11_DATA_DIR" "$V123_BUNDLE_DIR"
cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 1. Discover 251K Super-Teacher Soft Distill Pool
DISTILL_POOL="${DISTILL_POOL:-/content/out_v140_superteacher/v140_superteacher_soft_distill_pool.jsonl.gz}"
if [ ! -f "$DISTILL_POOL" ] && [ -f "/content/v140_superteacher_soft_distill_pool.jsonl.gz" ]; then
  DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
fi
if [ ! -f "$DISTILL_POOL" ] && [ -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" ]; then
  echo "[$(date +%H:%M:%S)] Copying 251K Super-Teacher soft pool from Drive..." | tee -a "$LOG_DRIVER"
  cp -f "$ST_DRIVE_DIR/v140_superteacher_soft_distill_pool.jsonl.gz" "/content/v140_superteacher_soft_distill_pool.jsonl.gz"
  DISTILL_POOL="/content/v140_superteacher_soft_distill_pool.jsonl.gz"
fi
if [ ! -f "$DISTILL_POOL" ]; then
  echo "[$(date +%H:%M:%S)] ERROR missing Super-Teacher distill pool: $DISTILL_POOL" | tee -a "$LOG_DRIVER"
  exit 2
fi

# 2. Ensure v11 & unified v123 zero-leak bundles are unpacked
V11_BUNDLE_ZIP="${V11_BUNDLE_ZIP:-$V11_DRIVE_DIR/v11_omniguard_all_data_bundle.zip}"
if [ ! -f "$V11_DATA_DIR/v11_wave1_dualaxis_gold_verified.jsonl.gz" ] && [ -f "$V11_BUNDLE_ZIP" ]; then
  python3 -c "import zipfile; zipfile.ZipFile('$V11_BUNDLE_ZIP').extractall('$V11_DATA_DIR')"
fi
UNIFIED_ZIP=""
for zcand in "/content/v123_all_gold_sibling_and_delta_bundle.zip" "$DRIVE_ROOT/v123_all_gold_sibling_and_delta_bundle.zip"; do
  [ -f "$zcand" ] && UNIFIED_ZIP="$zcand" && break
done
if [ -n "$UNIFIED_ZIP" ]; then
  CUR_SHA="$(sha256sum "$UNIFIED_ZIP" | awk '{print $1}')"
  PREV_SHA="$(cat "$V123_BUNDLE_DIR/.zip_sha" 2>/dev/null || true)"
  if [ "$CUR_SHA" != "$PREV_SHA" ] || [ ! -d "$V123_BUNDLE_DIR/sibling" ]; then
    echo "[$(date +%H:%M:%S)] Unpacking unified v123 zero-leak bundle $UNIFIED_ZIP -> $V123_BUNDLE_DIR" | tee -a "$LOG_DRIVER"
    rm -rf "$V123_BUNDLE_DIR" && mkdir -p "$V123_BUNDLE_DIR"
    python3 -c "import zipfile; zipfile.ZipFile('$UNIFIED_ZIP').extractall('$V123_BUNDLE_DIR')"
    echo "$CUR_SHA" > "$V123_BUNDLE_DIR/.zip_sha"
  fi
fi

# 3. Discover Picky-BPE-12k tokenizer & v04 multilingual anchor
V07_TOKENIZER="${V07_TOKENIZER:-$V07_DRIVE_DIR/v0.7.0-r2-omniveritas-b4_bili_picky_bpe_12k.json}"
if [ ! -f "$V07_TOKENIZER" ]; then
  for tc in "$V07_DRIVE_DIR/bili_picky_bpe_12k.json" "$V11_DRIVE_DIR/bili_picky_bpe_12k.json" "$DRIVE_ROOT/v122_apexnano/bili_picky_bpe_12k.json"; do
    [ -f "$tc" ] && V07_TOKENIZER="$tc" && break
  done
fi

COMMON_ARGS=(
  --distill-pool "$DISTILL_POOL"
  --sibling-dir "$V123_BUNDLE_DIR/sibling"
  --gold-v2-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --wave3-dir "$V123_BUNDLE_DIR/gold_v2_w3"
  --v11-data-dir "$V11_DATA_DIR"
  --v07-tokenizer "$V07_TOKENIZER"
  --batch-size "$BS"
  --seed "$SEED"
  --t-train-mode uniform_1_6
  --step-decay-mode inv_sqrt
  --freeze-r0
)
[ -d "$V123_BUNDLE_DIR/delta" ] && COMMON_ARGS+=(--delta-dir "$V123_BUNDLE_DIR/delta")
if [ ! -f "$V04_DATA" ] && [ -f "$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz" ]; then
  V04_DATA="$DRIVE_ROOT/shared_v04_data/v04-merged-1m4-r3.jsonl.gz"
fi
[ -f "$V04_DATA" ] && COMMON_ARGS+=(--v04-data "$V04_DATA")
if [ "$FULL_BUDGET" != "1" ]; then
  COMMON_ARGS+=(--ablation-fast)
fi

run_arm() {
  local arm="$1"
  shift
  local ver="v1.6.0-${arm}"
  local odir="$OUT_ROOT/${arm}"
  local logf="$OUT_ROOT/log_${arm}.txt"
  mkdir -p "$odir"
  local r0_flags=()
  if [ -f "$SHARED_R0" ]; then
    r0_flags+=(--init-r0-ckpt "$SHARED_R0")
  else
    r0_flags+=(--save-r0-ckpt "$SHARED_R0")
  fi
  echo "[$(date +%H:%M:%S)] === Launching Locked Arm ${arm} (${ver}) ===" | tee -a "$LOG_DRIVER"
  python3 -u -m meguviz_cac.train_v160_stratosphere_loop \
    "${COMMON_ARGS[@]}" \
    "${r0_flags[@]}" \
    --out "$odir" \
    --checkpoint-dir "$DRIVE_DIR" \
    --version "$ver" \
    "$@" 2>&1 | tee "$logf"
  cp -f "$odir/metrics-${ver}.json" "$DRIVE_DIR/"
  [ -f "$odir/${ver}_eval_cache.npz" ] && cp -f "$odir/${ver}_eval_cache.npz" "$DRIVE_DIR/"
  [ -f "$SHARED_R0" ] && cp -f "$SHARED_R0" "$DRIVE_DIR/"
  cp -f "$logf" "$DRIVE_DIR/"
  sync
  echo "[$(date +%H:%M:%S)] === Finished Locked Arm ${arm} -> synced metrics & cache to Drive ===" | tee -a "$LOG_DRIVER"
}

for ARM in $ARMS; do
  case "$ARM" in
    A0)
      # Same 1L/512 geometry + legacy anchor/rho/kappa update (step 5/6 reuses step 4), NO Fourier, NO NCL
      run_arm A0 --arch v160_a1 --dynamics v150_anchor --no-step-phi
      ;;
    A1)
      # Primary candidate: A0 ONLY switches to bounded projected residual (Pi_R + causal h_t), NO Fourier, NO NCL
      run_arm A1 --arch v160_a1 --dynamics stratosphere --no-step-phi
      ;;
    A2)
      # Fourier isolation: A1 ONLY adds continuous Fourier step embedding phi(t)
      run_arm A2 --arch v160_a1 --dynamics stratosphere --use-step-phi
      ;;
    A3)
      # NCL falsification arm: A1 ONLY adds NCL (same weights, no Slot-9 injection)
      run_arm A3 --arch v160_a1 --dynamics stratosphere --no-step-phi --use-ncl
      ;;
    A4)
      # Capacity challenger: A1 ONLY scales FFN 512 -> 640 (1L/640)
      run_arm A4 --arch v160_a2 --dynamics stratosphere --no-step-phi
      ;;
    A5)
      # Long-range attribution challenger: A4 ONLY scales Prelude 1L -> 2L (2L/640)
      run_arm A5 --arch v160_aprime --dynamics stratosphere --no-step-phi
      ;;
    *)
      echo "[WARN] Unknown arm: $ARM" | tee -a "$LOG_DRIVER"
      ;;
  esac
done

# Build unified Pass/Kill scoreboard JSON across all 6 locked arms
python3 - <<'PY'
import json
from pathlib import Path
out_root = Path("/content/out_v160_ablation")
drive_dir = Path("/content/drive/MyDrive/MeguvizCAC/v160_stratosphere_loop_r1")
board = {}
for mpath in sorted(out_root.glob("*/metrics-v1.6.0-*.json")):
    arm = mpath.parent.name
    d = json.loads(mpath.read_text(encoding="utf-8"))
    lp = d.get("loop_physics", {})
    board[arm] = {
        "arch": d.get("arch"),
        "dynamics": d.get("dynamics"),
        "use_step_phi": d.get("use_step_phi"),
        "use_ncl": d.get("use_ncl"),
        "params": d.get("total_params"),
        "bootable_mib": d.get("export", {}).get("bootable_mib"),
        "ci_passed": d.get("release_gates", {}).get("v160_ci_passed"),
        "phase2_sel_obj": d.get("phase2_fp16_val", {}).get("sel_obj"),
        "phase3_qat_obj": d.get("phase3_qat_best_obj"),
        "gold_v2_test_pr": d.get("test_bili_gold_v2_onnx_full_t4", {}).get("pr"),
        "gold_v2_test_r95": d.get("test_bili_gold_v2_onnx_full_t4", {}).get("r95"),
        "ml_val_pr": d.get("val_multilingual_onnx", {}).get("pr"),
        "god500": d.get("diag_god500_onnx"),
        "t1_minus_r1_pr": lp.get("t1_minus_r1_pr"),
        "t3_minus_t2_pr": lp.get("t3_minus_t2_pr"),
        "t4_minus_t2_pr": lp.get("t4_minus_t2_pr"),
        "t3_state_t4_readout_ctrl": lp.get("t3_state_into_t4_readout_control"),
        "period2_oscillation": lp.get("period2_oscillation"),
        "proj_sat_rate_t1_t6": lp.get("proj_sat_rate_t1_t6"),
        "joint_jacobian_t1_t6": lp.get("joint_directional_jacobian_t1_t6"),
        "quant_drift_audit": d.get("quant_drift_3ckpt_audit"),
        "fusion_cert_comparison": d.get("fusion_cal_cert_comparison"),
        "order_3seg_probe": d.get("order_preserving_3seg_probe"),
        "ort_4t_b1_ms": d.get("ort_4t_b1_latency_ms"),
        "elapsed_s": d.get("elapsed_seconds"),
    }
sb_txt = json.dumps(board, ensure_ascii=False, indent=2)
(out_root / "v160_ablation_scoreboard.json").write_text(sb_txt, encoding="utf-8")
(drive_dir / "v160_ablation_scoreboard.json").write_text(sb_txt, encoding="utf-8")
print("=== v1.6.0 STRATOSPHERELOOP LOCKED A0-A5 SCOREBOARD ===")
print(sb_txt)
PY

# Pack full release zips for primary A1 (and A2/A4 if present) & SHA256-verify
for PACK_ARM in A1 A2 A4 A5; do
  if [ -d "$OUT_ROOT/$PACK_ARM" ] && [ -f "$OUT_ROOT/$PACK_ARM/bundle_v10_stratosphere_v160.mgvz" ]; then
    FULL_ZIP="$DRIVE_DIR/meguvizcac-v1.6.0-${PACK_ARM}-full.zip"
    (cd "$OUT_ROOT/$PACK_ARM" && zip -r -6 "$FULL_ZIP" \
      bundle_v10_stratosphere_v160.mgvz charlm_v160.bin bili_picky_bpe_12k.json manifest.json \
      v160_r0_apexchar_fp32.onnx v160_prelude_t1_fp32.onnx v160_loop_step_fp32.onnx v160_dynaseam_fp32.onnx \
      v160_r0_apexchar_fp32.skel.onnx v160_prelude_t1_fp32.skel.onnx v160_loop_step_fp32.skel.onnx v160_dynaseam_fp32.skel.onnx \
      "metrics-v1.6.0-${PACK_ARM}.json" \
      "meguvizcac-v1.6.0-${PACK_ARM}-bootable-v10-stratosphere.zip" \
      "meguvizcac-v1.6.0-${PACK_ARM}-4stage-onnx-fp32exact.zip")
  fi
done
(cd "$DRIVE_DIR" && sha256sum meguvizcac-v1.6.0-*.zip v160_ablation_scoreboard.json > SHA256SUMS.txt 2>/dev/null || true; cat SHA256SUMS.txt 2>/dev/null || true)

touch "$DRIVE_DIR/.synced_ok" && sync
python3 -c "import json, time; json.dump({'arms': '$ARMS', 'done_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'drive_dir': '$DRIVE_DIR'}, open('$DONE_JSON', 'w'), indent=2)"
echo "[$(date +%H:%M:%S)] [v1.6.0-StratosphereLoop Locked A0-A5 DONE] Synced to $DRIVE_DIR" | tee -a "$LOG_DRIVER"
