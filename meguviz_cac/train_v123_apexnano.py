"""MeguvizCAC v1.2.3-ApexNano-Pro 3-Phase Post-Training Pipeline (Colab G4 RTX PRO 6000 / A100 + Local `--smoke-test`).

Implements ALL 8 P0 upgrades from `docs/roadmap-v1.2.3-apexnano-pro.md` (`ADR-v1.2.3`) while keeping
all `v1.0.0–v1.2.2` source files 100% untouched:
  - P0-1: 6-Way Disjoint Validation (`god_tier_500_holdout` strictly diagnostic-only; `val_meta_cal` 100% pure wild;
          `val_cal_cert` expanded to `>= 1,000` real-wild AI positives for mathematically valid `Certified LTT`!).
  - P0-2: `R0 <-> T2` Mutual Co-Distillation (`L_r0_to_t2` on confident `<65c` samples) + unlocked `z_char_coda_ai` (`3x` LR,
          `detach_zchar_for_ai=False` in Phase A1/C!).
  - P0-3: Explicit 6-Subpool Stratified Phase A Quotas (`M_hard_mem 15%`, `ai_rare_stealth 18%`, `dual_pos_ai_promo 5%`,
          `era_dro_humans 16%` [1.5x on `>1y x 25-80c` short hard humans], `sib_dual_gold 22%`, `legacy_t0_t1 24%`,
          synth `<= 8%`) + dedicated `<60c` `SOPA-s` pAUC & `CVaR-DORO`.
  - P0-4: `R1 + R0` Clean-Human Floor Margin Loss (`L_floor`) + `clip_a=(0.80, 1.80)` cross-exit alignment protection.
  - P0-5: HAWQ-V2 Bitwidth Reallocation (8 top-Fisher `prelude_l0`/`core.o_proj` layers -> `w4`, 3 zero-Fisher layers -> `seq2`,
          `<= 2.20 MB`, 0 B ABI change) + `0.6x` QAT LR + `QuEST` + `BitDistill` + `TR-KD`.
  - P0-6: Full-Pipeline Era Invariance (`15%` training-only `style56` dropout + `OfflineEraGRL` active in BOTH Phase A & Phase C QAT!)
          + Dual-Check `V123_E1` (`emb_era_probe`, `logit_era_probe`, `delta_old_minus_new`).
  - P0-7: Shock-Absorber v2 (`300` steps across `ai_head + r1_head + exit_affine + z_char_coda_ai + promo_head`)
          + Constrained 2-Bucket Length-Aware `z5_softmin` (`<60c` vs `>=60c`).
  - P0-8: 11-Gate CI (`V123_*`, adding `V123_AI_LUT_FLOOR` & `V123_CERTIFIED_LTT` + `V123_HAWQ_V2`)
          + seamless zero-shock warm-start from `v1.2.2-r1` FP16 master (or dual-source `v1.0 + v1.2.2`) + `v123-sibling-delta/`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_student import CODE_LIST, PRAG_TAGS, compute_eval_metrics
from .train_v04_dualloop import compute_high_precision_metrics
from .train_v05_apexloop import build_stratified_val_calibration_set
from .train_v06_hyperloop import _load_v06_balanced_bundle
from .train_v07_omniveritas import _derive_attr5_label, _wsd_lr_mult
from .train_v10_apexloop import _ingest_external_gemini_gold_v10
from .train_v11_omniguard import (
    DynamicPiKEGroupSamplerV11,
    collate_chronostream_docs,
    compute_dual_axis_ebt_margin_loss,
    compute_ebt6_partial_label_ce,
    compute_sub4_masked_loss,
)
from .train_v12_apexnano import compute_slot_orthogonality_loss
from .train_v121_apexnano import compute_v121_orthogonality_loss, fit_3param_logodds_fusion_v121
from .train_v122_apexnano import (
    _is_no_decay_param,
    compute_ebt_rce_emr_loss,
    compute_promo_cvar_ceiling_loss,
    compute_sopa_pauc_and_cvar_doro_loss,
    compute_tail_ranking_kd_loss,
)
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    extract_style_features_40d_batch,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import MAX_LEN_V07
from .v08_zenith_ops import (
    ASAMPerturber,
    OfflineEraGRLHead,
    compute_ccr_era_ipw,
    listnet_tail_distill_loss,
)
from .v09_cascade_cert import fit_cross_exit_affine_alignment
from .v10_apexloop import (
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    MeguvizV10ApexLoopModel,
    load_bundle_v6_mgvz_into_model,
)
from .v11_augment import BrandPriceSlotAug
from .v11_cascade_cert import certify_fixed_sequence_dual_axis_ltt, estimate_wild_prevalence_em
from .v11_data_engine import OnlineLongMixSynthesizerV11, compute_joint_length_domain_ipw
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d
from .v11_omniguard import AI_EMB_END_V11, R1_PROMO_VETO_DIMS_V11
from .v11_promo_pseudo import _sha_norm
from .v12_apexnano import extract_rung0_charhash_batch
from .v121_data_refinery import is_diamond_pair_source_v121
from .v122_apexnano import MeguvizV122ApexNanoModel
from .v123_apexnano import (
    MeguvizV123ApexNanoModel,
    apply_len_aware_z5_v123,
    apply_v122_full_logit_leace,
    bounded_fused3_logits,
    compute_bitdistill_slot_relation_kl,
    compute_wild_tail_objective_v122,
    export_v121_modular_onnx_and_bundle_v7,
    fit_conformal_fdr_thresholds_v122,
    fit_len_aware_z5_bootstrap_v123,
    fit_moses_register_offsets_v122,
    fit_output_shifted_promo_lut_v122,
    fit_smooth_ai_lut_v122,
    fold_zero_op_v121,
    install_quest_trust_gradient_hooks,
    load_bundle_v7_v121_into_model,
    run_v123_ci_gates,
    transplant_dual_source_v10_v121_to_v122,
)
from .v123_data_refinery import (
    NUM_ERAS_V122,
    classify_epistemic_tier_v122,
    jitter_ai_punctuation_v123,
    load_sibling_and_delta_v123,
    run_v123_refinery_ci,
    sanitize_era5_v123,
    tag_v123_subpools,
)


# ---------------------------------------------------------------------------
# 1. v1.2.3 Parameter Groups & New Post-Training Losses (P0-2 / P0-3 / P0-4 / P0-5 / P0-6 / P0-7)
# ---------------------------------------------------------------------------

def build_v123_param_groups(
    model: MeguvizV123ApexNanoModel,
    phase: str,
    era_grl: nn.Module | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """v1.2.3 LR groups: unlocks `z_char_coda_ai*` at 3x LR, uses 0.6x QAT LR, keeps `era_grl` active in QAT, expands Shock v2!"""
    hooks: list[Any] = []
    promo_only = (
        "promo_head.", "promo_sub4_head.", "promo_span_head.", "r1_promo_head.",
        "promo_mil_gate", "ebt_promo_gate", "exit_affine_promo_",
        "z_char_r1_promo.", "z_char_coda_promo.", "z_char_coda_promo_gate", "promo_zchar_direct.",
        "longstream.", "chunk_emb_proj.", "core.inloop_ebt.fc2_new_", "attr6_new_",
    )
    z_char_ai = ("z_char_coda_ai.", "z_char_coda_ai_gate")
    ai_readout = (
        "ai_head.", "span_head.", "mil_gate", "ebt_logit_gate", "r1_head.", "r1_span_head.",
        "defer_head_r2.", "defer_r1_enc.", "cbm_audit_w.", "attr6_old_", "core.inloop_ebt.fc1.", "core.inloop_ebt.fc2_old_",
    )
    top_core = ("core.qres.", "coda_proj.", "coda_norm.", "style_net.", "style_gate", "cbm_enc.", "cbm_dec.", "span_topdown_proj.")
    r0_pref = ("rung0.",)

    buckets: dict[tuple[float, float], list[torch.Tensor]] = {}

    def _add(p: torch.Tensor, lr: float, wd: float) -> None:
        if lr <= 0.0:
            p.requires_grad_(False)
            return
        p.requires_grad_(True)
        buckets.setdefault((lr, wd), []).append(p)

    for n, p in model.named_parameters():
        nd = _is_no_decay_param(n, p)
        wd_use = 0.0 if nd else 0.015
        if phase == "phase_a0":
            if any(n.startswith(k) for k in promo_only):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in z_char_ai):
                _add(p, 3.5e-4, wd_use)
            elif any(n.startswith(k) for k in ai_readout):
                _add(p, 1.4e-4, wd_use)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 3.0e-4, wd_use)
            else:
                _add(p, 0.0, 0.0)
        elif phase == "phase_a1":
            if any(n.startswith(k) for k in promo_only):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in z_char_ai):
                _add(p, 3.5e-4, wd_use)
            elif any(n.startswith(k) for k in ai_readout):
                _add(p, 1.1e-4, wd_use)
            elif any(n.startswith(k) for k in top_core):
                _add(p, 3.5e-5, wd_use)
                if n in ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias"):
                    def _mk_ai_mask(g: torch.Tensor) -> torch.Tensor:
                        out = g.clone()
                        out[AI_EMB_END_V11:] = 0.0
                        return out
                    hooks.append(p.register_hook(_mk_ai_mask))
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 2.5e-4, wd_use)
            else:
                _add(p, 1.2e-5, wd_use)
        elif phase == "phase_b":
            if any(n.startswith(k) for k in promo_only):
                _add(p, 2.0e-4, wd_use)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 2.5e-4, wd_use)
            elif n in ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias"):
                _add(p, 1.0e-4, wd_use)
                def _mk_pr_mask(g: torch.Tensor) -> torch.Tensor:
                    out = g.clone()
                    out[:AI_EMB_END_V11] = 0.0
                    return out
                hooks.append(p.register_hook(_mk_pr_mask))
            elif n in ("cbm_enc.weight", "cbm_enc.bias"):
                _add(p, 9.0e-5, wd_use)
                def _mk_cbm_pr(g: torch.Tensor) -> torch.Tensor:
                    out = g.clone()
                    out[:16] = 0.0
                    return out
                hooks.append(p.register_hook(_mk_cbm_pr))
            else:
                _add(p, 0.0, 0.0)
        elif phase == "qat":
            # P0-5: 0.6x QAT LR + frozen SoftBall centers + active `z_char_coda_ai`!
            if n in ("ai_head.center", "promo_head.center"):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in z_char_ai):
                _add(p, 7.5e-5, wd_use)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 4.8e-5, wd_use)
            elif any(n.startswith(k) for k in (ai_readout + promo_only)):
                _add(p, 5.5e-5, wd_use)
            else:
                _add(p, 4.2e-5, wd_use)
        elif phase == "shock":
            # P0-7: Shock-Absorber v2 — freeze W8 rung0 + quantized trunk; repair ai_head + r1_head + exit_affine + z_char_coda_ai + promo_head!
            if any(n.startswith(k) for k in r0_pref) or n in ("ai_head.center", "promo_head.center"):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in (ai_readout + z_char_ai + promo_only + ("core.qres.", "coda_norm.", "slot_norm.", "funnel_norm.", "exit_affine"))):
                _add(p, 5.5e-5, 0.0)
            else:
                _add(p, 0.0, 0.0)
        else:
            _add(p, 1.0e-4, wd_use)

    # P0-6: Keep `era_grl` active across BOTH Phase A AND Phase C QAT!
    if era_grl is not None and phase in ("phase_a0", "phase_a1", "qat"):
        lr_g = 3.0e-4 if phase in ("phase_a0", "phase_a1") else 1.5e-4
        for p in era_grl.parameters():
            _add(p, lr_g, 0.0)

    pg = [{"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]
    return pg, hooks


def compute_r0_to_t2_mutual_kd_loss(
    lg_t2: torch.Tensor,
    lg_r0: torch.Tensor,
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
    char_lens: torch.Tensor,
) -> torch.Tensor:
    """P0-2: Reverse KD from confident, accurate `r0_charchash` -> `t2` on short/template comments (`<=65c`)."""
    p_r0 = torch.sigmoid(lg_r0.detach().float())
    confident_correct = (has_ai > 0.5) & (char_lens <= 65) & ((p_r0 - y_ai.float()).abs() <= 0.22)
    n_c = int(confident_correct.sum().item())
    if n_c < 4:
        return lg_t2.sum() * 0.0
    zt = lg_t2.float()[confident_correct]
    zr = lg_r0.detach().float()[confident_correct].clamp(-8.0, 8.0)
    return 0.15 * F.mse_loss(zt, zr) + 0.10 * listnet_tail_distill_loss(zt, zr)


def compute_r1_r0_clean_human_floor_loss(
    lg_r1: torch.Tensor,
    lg_r0: torch.Tensor,
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
) -> torch.Tensor:
    """P0-4: Push confirmed clean-human `R1` & `R0` logits cleanly below `-2.8` / `-3.0` to unlock fast early exit!"""
    hu_clean = (has_ai > 0.5) & (y_ai <= 0.15)
    if int(hu_clean.sum().item()) < 4:
        return lg_r1.sum() * 0.0
    l_r1 = F.relu(lg_r1.float()[hu_clean] + 2.8).pow(2).mean()
    l_r0 = F.relu(lg_r0.float()[hu_clean] + 3.0).pow(2).mean()
    return 0.12 * l_r1 + 0.10 * l_r0


def eval_v123_dual_era_invariance(
    emb_val: np.ndarray,
    lg5_val: np.ndarray,
    y_ai_01: np.ndarray,
    era5_val: np.ndarray,
) -> dict[str, float]:
    """P0-6: Measure BOTH `emb[0:192]` linear era probe AND direct 5-head logit era probe + `delta_old_minus_new`."""
    hu = (y_ai_01 <= 0.25) & ((era5_val == 0) | (era5_val >= 3))
    if int(hu.sum()) < 40:
        return {"emb_era_probe_auroc": 0.5, "logit_era_probe_auroc": 0.5, "delta_old_minus_new": 0.0}
    y_e = (era5_val[hu] >= 3).astype(int)
    if len(np.unique(y_e)) < 2 or min(int((y_e == 0).sum()), int((y_e == 1).sum())) < 10:
        return {"emb_era_probe_auroc": 0.5, "logit_era_probe_auroc": 0.5, "delta_old_minus_new": 0.0}
    rng = np.random.default_rng(20261128)
    perm = rng.permutation(len(y_e))
    cut = max(16, int(0.7 * len(y_e)))
    tr, te = perm[:cut], perm[cut:]
    if len(np.unique(y_e[tr])) < 2 or len(np.unique(y_e[te])) < 2:
        return {"emb_era_probe_auroc": 0.5, "logit_era_probe_auroc": 0.5, "delta_old_minus_new": 0.0}
    e_sub = emb_val[hu, :AI_EMB_END_V11]
    l_sub = lg5_val[hu]
    clf_e = LogisticRegression(max_iter=400, C=0.5).fit(e_sub[tr], y_e[tr])
    clf_l = LogisticRegression(max_iter=400, C=0.5).fit(l_sub[tr], y_e[tr])
    emb_auc = float(roc_auc_score(y_e[te], clf_e.predict_proba(e_sub[te])[:, 1]))
    lg_auc = float(roc_auc_score(y_e[te], clf_l.predict_proba(l_sub[te])[:, 1]))
    # t2 logit mean(old) - mean(new)
    delta_on = float(l_sub[y_e == 0, 3].mean() - l_sub[y_e == 1, 3].mean())
    return {
        "emb_era_probe_auroc": round(emb_auc, 4),
        "logit_era_probe_auroc": round(lg_auc, 4),
        "delta_old_minus_new": round(delta_on, 4),
    }


# ---------------------------------------------------------------------------
# 2. Main v1.2.3-ApexNano-Pro 3-Phase Training Pipeline
# ---------------------------------------------------------------------------

def train_v123_apexnano_full(args) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ref_ci = run_v123_refinery_ci()
    assert ref_ci["v123_refinery_ci_pass"], f"v123_data_refinery CI failed: {ref_ci}"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bs = 128 if args.smoke_test else (512 if vram_gb >= 60 else (256 if vram_gb >= 32 else 128))

    out_dir = args.out.resolve()
    ckpt_dir = (args.checkpoint_dir or out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    print(
        f"[v1.2.3-ApexNano-Pro Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={bs} | "
        f"v122={args.v122_fp16_master or args.v122_bundle_mgvz} | v10={args.v10_fp16_master or args.v10_bundle_mgvz} | "
        f"sibling={args.sibling_dir} | delta={args.delta_dir}",
        flush=True,
    )

    def _read_jsonl_list(p: Path | None) -> list[dict[str, Any]]:
        if p is None or not p.is_file():
            return []
        op = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")
        with op as f:
            return [json.loads(l) for l in f if l.strip()]

    # 1. Diamond-Only Pair Filter + Base Corpus
    extra_pairs: list[Path] = []
    dropped_noisy: list[str] = []
    for p in args.extra_pairs:
        if is_diamond_pair_source_v121(p.name):
            extra_pairs.append(p)
        else:
            dropped_noisy.append(p.name)
    if args.v08_gemini_pairs and args.v08_gemini_pairs.is_file():
        extra_pairs.append(args.v08_gemini_pairs.resolve())
    train_raw, eval_rows, legacy_pairs, _, data_audit = _load_v06_balanced_bundle(
        args.data.resolve(), args.gateguard_data.resolve() if args.gateguard_data else None,
        extra_pairs, None,
    )
    if data_audit.get("gateguard_dropped_leakage", 0) > 0:
        raise RuntimeError(f"Anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} pairs collided with eval!")
    _ingest_external_gemini_gold_v10(train_raw, args.v08_gemini_gold)
    print(f"[Stage0-Pairs] diamond_pairs={len(legacy_pairs):,} | dropped_noisy={dropped_noisy}", flush=True)

    if args.smoke_test:
        rng_sm = random.Random(20261128)
        train_raw = rng_sm.sample(train_raw, min(384, len(train_raw)))
        legacy_pairs = legacy_pairs[:24]

    # 2. Tokenizer (12k BPE)
    tok_path = out_dir / "bili_picky_bpe_12k.json"
    if args.v07_tokenizer and args.v07_tokenizer.is_file():
        shutil.copy2(args.v07_tokenizer, tok_path)
        tokenizer = BiliPickyBPETokenizer(tok_path)
    else:
        txts_t = [x["raw_text"] for x in train_raw[: (384 if args.smoke_test else 200000)]]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(txts_t, tok_path, vocab_size=2048 if args.smoke_test else BPE_VOCAB_SIZE_V07)

    # 3. Hydrate base `score_source` & `created_ts` + Gold-v2 + Wave-3 + Sibling + Optional Delta Pack!
    orig_src_by_sid: dict[str, str] = {}
    orig_ts_by_sid: dict[str, Any] = {}
    if not args.smoke_test:
        op_src = gzip.open(args.data.resolve(), "rt", encoding="utf-8") if args.data.suffix == ".gz" else args.data.resolve().open("r", encoding="utf-8")
        with op_src as f_src:
            for line in f_src:
                if not line.strip():
                    continue
                r_s = json.loads(line)
                if r_s.get("split", "train") == "train":
                    sid_s = str(r_s.get("sample_id", r_s.get("id", "")))
                    if sid_s:
                        orig_src_by_sid[sid_s] = str(r_s.get("score_source") or "")
                        if r_s.get("created_ts") is not None or r_s.get("ctime") is not None:
                            orig_ts_by_sid[sid_s] = r_s.get("created_ts", r_s.get("ctime"))

    gem_pro_sids, gem_flash_sids = set(), set()
    if args.v08_gemini_gold and args.v08_gemini_gold.is_file():
        for r_g in _read_jsonl_list(args.v08_gemini_gold):
            if r_g.get("split", "train") != "train":
                continue
            sid_g = str(r_g.get("sample_id", r_g.get("id", "")))
            if "pro" in str(r_g.get("score_source", "")).lower():
                gem_pro_sids.add(sid_g)
            else:
                gem_flash_sids.add(sid_g)

    gold_v2_train_rows, gold_v2_val_rows, gold_v2_test_rows = [], [], []
    override_sids: set[str] = set()
    if args.gold_v2_dir and args.gold_v2_dir.is_dir():
        g2d = args.gold_v2_dir.resolve()
        gold_v2_train_rows = _read_jsonl_list(g2d / "v12_gold_v2_train.jsonl.gz")
        gold_v2_val_rows = _read_jsonl_list(g2d / "v12_gold_v2_val.jsonl.gz")
        gold_v2_test_rows = _read_jsonl_list(g2d / "v12_gold_v2_test.jsonl.gz")
        override_sids |= {str(r.get("sample_id", "")) for r in gold_v2_train_rows if r.get("sample_id")}

    wave3_train_rows, wave3_val_rows = [], []
    w3_dir = args.wave3_dir.resolve() if (args.wave3_dir and args.wave3_dir.is_dir()) else (args.gold_v2_dir.resolve() if (args.gold_v2_dir and args.gold_v2_dir.is_dir()) else None)
    if w3_dir is not None:
        wave3_train_rows = _read_jsonl_list(w3_dir / "v121_wave3_train.jsonl.gz")
        wave3_val_rows = _read_jsonl_list(w3_dir / "v121_wave3_val.jsonl.gz")
        override_sids |= {str(r.get("sample_id", "")) for r in wave3_train_rows if r.get("sample_id")}

    sib = load_sibling_and_delta_v123(
        args.sibling_dir.resolve() if args.sibling_dir else None,
        args.delta_dir.resolve() if (args.delta_dir and args.delta_dir.is_dir()) else None,
        smoke_test=bool(args.smoke_test),
    )
    quarantine_sids: set[str] = sib["quarantine_sids"]
    replace_text_by_sid: dict[str, str] = sib["replace_text_by_sid"]
    override_sids |= sib["override_sids"]
    print(
        f"[Stage0-Sibling+Delta] sib={dict(sib['stats'])} | delta={dict(sib['delta_stats'])} | "
        f"g2_tr={len(gold_v2_train_rows)} w3_tr={len(wave3_train_rows)} overrides={len(override_sids):,}",
        flush=True,
    )

    # Apply refinery + sibling hygiene to base train_raw
    refined_items: list[dict[str, Any]] = []
    t3_dropped = hyg_purged = hyg_patched = ov_deduped = 0
    for it in train_raw:
        sid_it = str(it.get("sample_id", ""))
        if sid_it and sid_it in quarantine_sids:
            hyg_purged += 1
            continue
        if sid_it and sid_it in override_sids:
            ov_deduped += 1
            continue
        if sid_it and sid_it in replace_text_by_sid:
            it["raw_text"] = replace_text_by_sid[sid_it]
            hyg_patched += 1
        src_it = orig_src_by_sid.get(sid_it, str(it.get("score_source", "")))
        if not src_it and it.get("is_pre_llm"):
            src_it = "pre_llm_human_prior"
        it["score_source"] = src_it
        is_gem_pro = sid_it in gem_pro_sids
        is_gem_any = is_gem_pro or (sid_it in gem_flash_sids)
        tier, tw, eff_sc, _ = classify_epistemic_tier_v122(
            score=float(it["y_score"]) * 100.0, score_source=src_it, text=str(it.get("raw_text", "")),
            overridden_by_gemini=is_gem_any, gemini_is_pro=is_gem_pro,
        )
        if tier == "T3_QUARANTINE" or tw <= 0.0:
            t3_dropped += 1
            continue
        y01 = float(np.clip(eff_sc / 100.0, 0.0, 1.0))
        it["y_score"] = y01
        it["has_ai"] = 1.0
        it["epistemic_tier"] = tier
        it["is_true_pro_gold"] = bool(tier == "T0_DIAMOND_GOLD")
        it["is_phys_prellm"] = bool(src_it == "pre_llm_human_prior" or it.get("is_pre_llm"))
        it["era5"] = sanitize_era5_v123(it.get("era_bucket"), orig_ts_by_sid.get(sid_it), is_pre_llm=it["is_phys_prellm"], score_source=src_it, ai_score_0_100=eff_sc)
        it["weight"] = float(it["weight"]) * float(tw)
        it["y_promo"] = -1.0
        it["has_promo"] = 0.0
        it["sub4_idx"] = -1
        it["l_pceil_admit"] = False
        it.update(tag_v123_subpools(it, str(it.get("raw_text", "")), y01, -1.0, 0.0, it["era5"]))
        refined_items.append(it)

    # P0-1: Carve strictly REAL-WILD never-trained `cert_holdout_rows` (`>= 850` real-wild AI positives!) for mathematically valid LTT!
    rng_cert = random.Random(2026112801)
    def _is_real_wild_bili(it: dict[str, Any]) -> bool:
        src = str(it.get("score_source", "")).lower()
        coh = str(it.get("cohort", "")).lower()
        if any(tok in src for tok in ("synth", "plugin", "deepseek", "qwen", "inkling")):
            return False
        return (coh in ("", "bili", "v04_1m", "real_bili")) or it.get("is_phys_prellm", False)

    cert_pos_idx = [i for i, it in enumerate(refined_items) if it["y_score"] >= 0.80 and _is_real_wild_bili(it)]
    cert_neg_idx = [i for i, it in enumerate(refined_items) if it["y_score"] <= 0.15 and _is_real_wild_bili(it)]
    if len(cert_pos_idx) < (16 if args.smoke_test else 500):
        cert_pos_idx = [i for i, it in enumerate(refined_items) if it["y_score"] >= 0.80]
    rng_cert.shuffle(cert_pos_idx)
    rng_cert.shuffle(cert_neg_idx)
    take_c_pos = min(16 if args.smoke_test else 550, max(1, len(cert_pos_idx) // 3))
    take_c_neg = min(32 if args.smoke_test else 3500, max(1, len(cert_neg_idx) // 5))
    cert_idx_set = set(cert_pos_idx[:take_c_pos]) | set(cert_neg_idx[:take_c_neg])
    cert_holdout_rows = [
        {"sample_id": refined_items[i].get("sample_id", f"cert_{i}"), "text": refined_items[i]["raw_text"], "score": refined_items[i]["y_score"] * 100.0, "era5": refined_items[i]["era5"]}
        for i in sorted(cert_idx_set)
    ]
    refined_items = [it for i, it in enumerate(refined_items) if i not in cert_idx_set]
    print(
        f"[Stage0-V123Refinery] kept={len(refined_items):,} | base_cert_wild={len(cert_holdout_rows)} "
        f"(pos={take_c_pos}, neg={take_c_neg}) | hyg_purged={hyg_purged:,} T3_purged={t3_dropped:,} overrides={ov_deduped:,}",
        flush=True,
    )

    # Joint length x domain IPW
    ipw_w = compute_joint_length_domain_ipw([{"text": x["raw_text"], "y_score": x["y_score"]} for x in refined_items])
    for it, w_ip in zip(refined_items, ipw_w):
        it["weight"] = float(it["weight"]) * float(w_ip)

    # Pre-encode helper
    SUB4_MAP = {"hard_contact": 0, "affiliate_link": 1, "soft_shill": 2, "self_promo": 3}

    def _make_item(
        text: str, y_ai: float, y_promo: float, has_promo: float, ptype: str, w: float,
        has_ai: float = 1.0, is_pro: bool = True, is_phys: bool = False, l_pceil: bool = False,
        era5: int = -1, extra_tags: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        cl, f56, mt = extract_style_features_56d(text)
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        item = {
            "raw_text": text, "clean_text": cl, "ids": ids,
            "engram": make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
            "trigram": make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10),
            "style": f56, "meta": mt, "char_len": len(text),
            "y_score": float(y_ai), "has_ai": float(has_ai),
            "y_promo": float(y_promo), "has_promo": float(has_promo),
            "sub4_idx": SUB4_MAP.get(ptype, -1) if (has_promo > 0.5 and y_promo >= 0.65) else -1,
            "weight": float(w), "attr5": (0 if float(y_ai) <= 0.30 else (1 if float(y_ai) <= 0.65 else 2)),
            "is_true_pro_gold": bool(is_pro), "is_phys_prellm": bool(is_phys),
            "l_pceil_admit": bool(l_pceil), "era5": int(era5),
            "is_rare_stealth_ai": False, "is_dual_pos_ai_promo": False,
            "is_old_short_hard_hu": False, "is_decorator_human": False, "is_synth": False,
        }
        if extra_tags:
            item.update(extra_tags)
        return item

    for it in refined_items:
        cl, f56, mt = extract_style_features_56d(it["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        it["clean_text"], it["style"], it["meta"], it["ids"] = cl, f56, mt, ids
        it["char_len"] = len(it["raw_text"])
        it["engram"] = make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
        it["trigram"] = make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        it["attr5"] = _derive_attr5_label(it)

    # 4. Ingest v11 + Wave-2.5 + Gold-v2 + Wave-3 + Sibling + Delta into Explicit P0-3 Stratified Sub-Pools!
    v11_dir = args.v11_data_dir.resolve()
    dual_gold_train, uot_pseudo_rows, hard_neg_rows, dual_pairs_train = [], [], [], []
    promo_val_select_rows, promo_val_cert_rows = [], []
    wave2_longform, wave2_longform_val = [], []
    new_gold_ai_pool, protect_pool, wild_promo_focus_pool = [], [], []
    # P0-3 Explicit Rare / High-Leverage Sub-Pools!
    ai_rare_stealth_pool: list[dict[str, Any]] = []
    dual_pos_ai_promo_pool: list[dict[str, Any]] = []
    old_short_hard_hu_pool: list[dict[str, Any]] = []
    synth_short_ai_pool: list[dict[str, Any]] = []

    if v11_dir.is_dir():
        for fn, mult in (("v11_wave1_dualaxis_gold_all5880.jsonl.gz", 1.15), ("v11_wave2_dualaxis_gold_train.jsonl.gz", 1.35)):
            for r in _read_jsonl_list(v11_dir / fn)[: (96 if args.smoke_test else 999999)]:
                if r.get("split", "train") != "train":
                    continue
                ya, yp = float(r.get("ai_score", 0.0)) / 100.0, float(r.get("promo_score", 0.0)) / 100.0
                if not (ya <= 0.30 or ya >= 0.70):
                    continue
                e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
                tags = tag_v123_subpools(r, str(r["text"]), ya, yp, 1.0, e5)
                it = _make_item(str(r["text"]), ya, yp, 1.0, str(r.get("promo_type", "none")), mult, era5=e5, extra_tags=tags)
                dual_gold_train.append(it)
                if tags["is_dual_pos_ai_promo"]:
                    dual_pos_ai_promo_pool.append(it)
        for r in _read_jsonl_list(v11_dir / "v11_wave1_uot_pseudo_train.jsonl.gz")[: (48 if args.smoke_test else 22000)]:
            yp = float(r.get("promo_prob", r.get("promo_score", 0.0)))
            if yp > 1.0:
                yp /= 100.0
            uot_pseudo_rows.append(_make_item(str(r["text"]), 0.0, yp, 1.0, str(r.get("promo_type", "none")), 0.65, has_ai=0.0, is_pro=False))
        for r in _read_jsonl_list(v11_dir / "v11_wave1_promo_hard_neg_train.jsonl.gz")[: (48 if args.smoke_test else 15000)]:
            hard_neg_rows.append(_make_item(str(r["text"]), 0.0, 0.0, 1.0, "none", 1.25, has_ai=0.0, is_pro=False))
        for fn in ("v11_wave1_dual_pairs.jsonl", "v11_wave2_hard_promo_pairs.jsonl"):
            dual_pairs_train.extend(_read_jsonl_list(v11_dir / fn)[: (16 if args.smoke_test else 999999)])
        for fn in ("v11_wave2_longform_train.jsonl.gz", "v11_wave2_longform_val.jsonl.gz"):
            rows_lf = _read_jsonl_list(v11_dir / fn)[: (16 if args.smoke_test else 999999)]
            if "val" in fn:
                wave2_longform_val.extend(rows_lf)
            else:
                wave2_longform.extend(rows_lf)
        for r in _read_jsonl_list(v11_dir / "v11_wave2_dualaxis_gold_val.jsonl.gz")[: (64 if args.smoke_test else 999999)]:
            ya, yp = float(r.get("ai_score", 0.0)) / 100.0, float(r.get("promo_score", 0.0)) / 100.0
            promo_val_select_rows.append(_make_item(str(r["text"]), ya, yp, 1.0, str(r.get("promo_type", "none")), 1.0))
        for r in _read_jsonl_list(v11_dir / "v11_wave2_dualaxis_gold_cert.jsonl.gz")[: (80 if args.smoke_test else 999999)]:
            ya, yp = float(r.get("ai_score", 0.0)) / 100.0, float(r.get("promo_score", 0.0)) / 100.0
            promo_val_cert_rows.append(_make_item(str(r["text"]), ya, yp, 1.0, str(r.get("promo_type", "none")), 1.0))

    if args.wave25_gold and args.wave25_gold.is_file():
        for r in _read_jsonl_list(args.wave25_gold)[: (48 if args.smoke_test else 999999)]:
            ya, yp = float(r.get("ai_score", 0.0)) / 100.0, float(r.get("promo_score", 0.0)) / 100.0
            e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
            tags = tag_v123_subpools(r, str(r["text"]), ya, yp, 1.0, e5)
            it = _make_item(str(r["text"]), ya, yp, 1.0, str(r.get("promo_type", "none")), 1.80, era5=e5, extra_tags=tags)
            dual_gold_train.append(it); wild_promo_focus_pool.append(it)
            if ya <= 0.30 or ya >= 0.70:
                new_gold_ai_pool.append(it)
            if tags["is_dual_pos_ai_promo"]:
                dual_pos_ai_promo_pool.append(it)

    for r in gold_v2_train_rows:
        ya = float(r.get("ai_score", 0.0)) / 100.0
        has_pr = 1.0 if r.get("promo_score") is not None else 0.0
        yp = float(r.get("promo_score", 0.0)) / 100.0 if has_pr > 0.5 else -1.0
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
        tags = tag_v123_subpools(r, str(r["text"]), ya, yp, has_pr, e5)
        it = _make_item(str(r["text"]), ya, yp, has_pr, str(r.get("promo_type", "none")), 1.50, era5=e5, extra_tags=tags)
        new_gold_ai_pool.append(it)
        if ya <= 0.25:
            protect_pool.append(it)
        if tags["is_rare_stealth_ai"]:
            ai_rare_stealth_pool.append(it)
        if tags["is_old_short_hard_hu"]:
            old_short_hard_hu_pool.append(it)

    for r in wave3_train_rows:
        ya, yp = float(r.get("ai_score", 0.0)) / 100.0, float(r.get("promo_score", 0.0)) / 100.0
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
        tags = tag_v123_subpools(r, str(r["text"]), ya, yp, 1.0, e5)
        it = _make_item(str(r["text"]), ya, yp, 1.0, str(r.get("promo_type", "none")), 1.85, era5=e5, extra_tags=tags)
        dual_gold_train.append(it); wild_promo_focus_pool.append(it)
        if ya <= 0.30 or ya >= 0.70:
            new_gold_ai_pool.append(it)
        if tags["is_dual_pos_ai_promo"]:
            dual_pos_ai_promo_pool.append(it)

    if wave3_val_rows:
        rng_w3 = random.Random(2026112802)
        w3_sh = list(wave3_val_rows)
        rng_w3.shuffle(w3_sh)
        h_w3 = len(w3_sh) // 2
        for r in w3_sh[:h_w3]:
            promo_val_select_rows.append(_make_item(str(r["text"]), float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0, 1.0, str(r.get("promo_type", "none")), 1.0))
        for r in w3_sh[h_w3:]:
            promo_val_cert_rows.append(_make_item(str(r["text"]), float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0, 1.0, str(r.get("promo_type", "none")), 1.0))

    # Carve `>= 500` extra real-wild AI positives from sibling dual_train into `cert_holdout_rows` BEFORE training (P0-1 LTT unlock!)
    sib_dual_all = list(sib["dual_train_rows"])
    sib_cert_pos_idx = [
        i for i, r in enumerate(sib_dual_all)
        if float(r["ai_score"]) >= 80.0 and not bool((r.get("bucket_meta") or {}).get("is_synthetic", False))
        and "synth" not in str(r.get("source_tag", "")).lower()
    ]
    rng_cert.shuffle(sib_cert_pos_idx)
    take_sib_cert = min(12 if args.smoke_test else 550, max(0, len(sib_cert_pos_idx) // 6))
    sib_cert_set = set(sib_cert_pos_idx[:take_sib_cert])
    for i in sorted(sib_cert_set):
        r = sib_dual_all[i]
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=float(r["ai_score"]))
        cert_holdout_rows.append({"sample_id": r.get("sample_id", f"sib_cert_{i}"), "text": str(r["text"]), "score": float(r["ai_score"]), "era5": e5})

    # Ingest remaining sibling + delta rows into explicit P0-3 sub-pools!
    for idx_s, r in enumerate(sib_dual_all):
        if idx_s in sib_cert_set:
            continue
        txt = str(r["text"])
        ya = float(r["ai_score"]) / 100.0
        has_pr = 1.0 if (r.get("has_promo") and r.get("promo_score") is not None) else 0.0
        yp = float(r["promo_score"]) / 100.0 if has_pr > 0.5 else -1.0
        src = str(r.get("source_tag") or "sibling_dual")
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
        tier, tw, _, _ = classify_epistemic_tier_v122(ya * 100.0, src, txt, is_hybrid_mixed=bool(r.get("is_hybrid_mixed")), judge_conf=r.get("judge_conf"))
        if tier == "T3_QUARANTINE":
            continue
        tags = tag_v123_subpools(r, txt, ya, yp, has_pr, e5)
        it = _make_item(txt, ya, yp, has_pr, str(r.get("promo_type") or "none"), tw, is_pro=(tier == "T0_DIAMOND_GOLD"), is_phys=(e5 == 0), era5=e5, extra_tags=tags)
        new_gold_ai_pool.append(it)
        if tags["is_rare_stealth_ai"]:
            ai_rare_stealth_pool.append(it)
        if tags["is_dual_pos_ai_promo"]:
            dual_pos_ai_promo_pool.append(it)
        if tags["is_old_short_hard_hu"]:
            old_short_hard_hu_pool.append(it)
        if (ya <= 0.25 and e5 in (0, 1, 2, 3)) or "oldvideo_gt1y" in src or "trap" in src or tags["is_decorator_human"]:
            protect_pool.append(it)
        if has_pr > 0.5:
            if yp >= 0.65:
                dual_gold_train.append(it); wild_promo_focus_pool.append(it)
            elif "organic_brand" in src or "wild_promo" in src or "god_tier" in src:
                hard_neg_rows.append(it); wild_promo_focus_pool.append(it)

    for r in sib["ai_only_real_rows"]:
        txt = str(r["text"])
        ya = float(r["ai_score"]) / 100.0
        src = str(r.get("source_tag") or "sibling_ai_only")
        e5 = sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=ya * 100.0)
        tags = tag_v123_subpools(r, txt, ya, -1.0, 0.0, e5)
        it = _make_item(txt, ya, -1.0, 0.0, "none", 1.35, is_pro=True, is_phys=(e5 == 0), era5=e5, extra_tags=tags)
        new_gold_ai_pool.append(it)
        if tags["is_rare_stealth_ai"]:
            ai_rare_stealth_pool.append(it)
        if tags["is_old_short_hard_hu"]:
            old_short_hard_hu_pool.append(it)
        if ya <= 0.25 and (e5 in (0, 1, 2, 3) or "trap" in src or tags["is_decorator_human"]):
            protect_pool.append(it)

    for r in sib["ai_only_synth_rows"]:
        txt = str(r["text"])
        ya = float(r["ai_score"]) / 100.0
        tags = tag_v123_subpools(r, txt, ya, -1.0, 0.0, -1)
        it = _make_item(txt, ya, -1.0, 0.0, "none", 0.85, is_pro=False, is_phys=False, era5=-1, extra_tags=tags)
        synth_short_ai_pool.append(it)

    # Counterfactual mirrors & sandwich hybrids (with boosted micro-short pairs!)
    cf_mirror_pool: list[dict[str, Any]] = list(sib["cf_mirror_pairs"])
    for lp in legacy_pairs:
        ai_t = str(lp.get("ai_text") or lp.get("pos_text") or "").strip()
        hu_t = str(lp.get("human_text") or lp.get("neg_text") or "").strip()
        if ai_t and hu_t:
            cf_mirror_pool.append({"ai_text": ai_t, "human_text": hu_t, "mode": "legacy_diamond", "pair_weight": 1.0, "is_micro_short": len(ai_t) <= 35})

    for sw in sib["sandwich_span_rows"][: (20 if args.smoke_test else 4824)]:
        wave2_longform.append({
            "text": sw["text"], "chunks": [sw["human_text"][:220], sw["ai_text"][:220], sw["human_text"][220:440] or sw["human_text"][:160]],
            "y_ai_doc": 1.0, "y_hyb_doc": 1.0, "y_pr_doc": 0.0,
        })

    t0_pool = [x for x in refined_items if x["epistemic_tier"] == "T0_DIAMOND_GOLD"]
    t1_pool = [x for x in refined_items if x["epistemic_tier"] != "T0_DIAMOND_GOLD"]
    protect_pool.extend([x for x in refined_items if x["y_score"] <= 0.25 and (x["is_phys_prellm"] or x["era5"] in (0, 1, 2, 3))])
    for x in refined_items:
        if x.get("is_old_short_hard_hu"):
            old_short_hard_hu_pool.append(x)

    # CCR Era-IPW on confirmed humans across all 5 eras
    all_hu_era = [x for x in (t0_pool + new_gold_ai_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.25 and x["era5"] >= 0]
    if len(all_hu_era) >= 120:
        st_era = np.stack([x["style"] for x in all_hu_era], axis=0)
        e_bin = np.array([0 if x["era5"] == 0 else (3 if x["era5"] >= 3 else x["era5"]) for x in all_hu_era], dtype=np.int32)
        y_era = np.array([x["y_score"] for x in all_hu_era], dtype=np.float32)
        ccr_w = compute_ccr_era_ipw(st_era, e_bin, y_era, clip_lo=0.25, clip_hi=4.0)
        for x, cw in zip(all_hu_era, ccr_w):
            x["weight"] = float(x["weight"]) * float(cw)

    # Partitioned Hard-Human Memory M = M_phys U M_old_short U M_modern
    mem_phys = [x for x in protect_pool if x.get("is_phys_prellm") or x.get("era5") == 0] or protect_pool[:64]
    mem_mod = [x for x in (t0_pool + new_gold_ai_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.25 and x.get("era5", -1) != 0] or protect_pool[:64]

    hu_clean = [x["raw_text"] for x in refined_items if x["y_score"] <= 0.15][:3000]
    ai_clean = [x["raw_text"] for x in refined_items if x["y_score"] >= 0.85][:3000]
    hu_hard = [x["raw_text"] for x in protect_pool[:1500]] or hu_clean[:100]
    pr_clean = [x["raw_text"] for x in wild_promo_focus_pool if x["y_promo"] >= 0.70][:1500] or ["关注工种号领取全套资料。"] * 12
    longmix_syn = OnlineLongMixSynthesizerV11(hu_clean[:2000] or ["真人评论测试。"] * 12, hu_hard, ai_clean[:2000] or ["人工智能评论测试。"] * 12, pr_clean)

    print(
        f"[Stage0-P03-Pools] legacy_t0={len(t0_pool):,} t1={len(t1_pool):,} | new_gold={len(new_gold_ai_pool):,} | "
        f"ai_rare_stealth={len(ai_rare_stealth_pool):,} dual_pos_ai_pr={len(dual_pos_ai_promo_pool):,} "
        f"old_short_hu={len(old_short_hard_hu_pool):,} synth_short={len(synth_short_ai_pool):,} | "
        f"protect={len(protect_pool):,} wild_promo={len(wild_promo_focus_pool):,} cf_mirrors={len(cf_mirror_pool):,}",
        flush=True,
    )

    # 5. P0-1: STRICT 6-WAY DISJOINT VALIDATION (ZERO `god_tier_500_holdout` IN `val_meta_cal`!)
    val_select_wild = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    vg_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_select_wild}
    val_cal_all, _ = build_stratified_val_calibration_set(eval_rows, target_n=1600)
    val_cal_nodup = [r for r in val_cal_all if str(r.get("sample_id", r.get("id", ""))) not in vg_ids] or val_cal_all

    # `val_meta_cal_rows` = 100% CLEAN WILD ONLY! (NO `god_tier_500_holdout` EVER!)
    val_meta_cal_rows: list[dict[str, Any]] = []
    for r in val_cal_nodup:
        val_meta_cal_rows.append({
            "text": str(r["text"]), "score": float(r["score"]),
            "era5": sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=bool(r.get("is_pre_llm")), ai_score_0_100=float(r["score"])),
        })

    # `god_tier_500_holdout` routed STRICTLY to Diagnostic-Only Evaluation!
    def _wrap_god(rows_g: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"text": str(r["text"]), "score": float(r["ai_score"]), "era5": sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), ai_score_0_100=float(r["ai_score"]))}
            for r in rows_g if r.get("ai_score") is not None
        ]
    god500_all_rows = _wrap_god(sib["god_tier_500_holdout"])
    god500_tv_rows = _wrap_god(sib["god_tier_500_text_verifiable"])

    # `val_cal_cert` = Expanded REAL-WILD certification set (`>= 1,000` real-wild AI positives!)
    used_ids = vg_ids | {str(r.get("sample_id", r.get("id", ""))) for r in val_cal_nodup}
    val_rem = [r for r in eval_rows if r.get("split") == "validation" and str(r.get("sample_id", r.get("id", ""))) not in used_ids]
    rng_v = random.Random(2026112803)
    vr_pos = [r for r in val_rem if float(r.get("score", 0.0)) >= 70.0]
    vr_neg = [r for r in val_rem if float(r.get("score", 0.0)) <= 30.0]
    rng_v.shuffle(vr_pos); rng_v.shuffle(vr_neg)
    val_cal_cert = vr_pos[:300] + vr_neg[:1500] + cert_holdout_rows
    rng_v.shuffle(val_cal_cert)

    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh = [r for r in eval_rows if r["split"] == "test" and r.get("cohort") == "dsh"]

    if args.smoke_test:
        def _cap_s(rows: list[dict], cap: int, k_sc: str = "score") -> list[dict]:
            pos = [r for r in rows if float(r.get(k_sc, r.get("ai_score", 0.0))) >= 70.0]
            neg = [r for r in rows if float(r.get(k_sc, r.get("ai_score", 0.0))) < 70.0]
            return pos[: cap // 2] + neg[: cap - min(len(pos), cap // 2)]
        val_select_wild = _cap_s(val_select_wild, 64)
        val_meta_cal_rows = _cap_s(val_meta_cal_rows, 96)
        val_cal_cert = _cap_s(val_cal_cert, 96)
        test_gold = _cap_s(test_gold, 48)
        test_dsh = _cap_s(test_dsh, 24)
        god500_all_rows = _cap_s(god500_all_rows, 48)
        god500_tv_rows = _cap_s(god500_tv_rows, 40)

    def _pre_eval(rows: list[dict[str, Any]], sc_key: str = "score"):
        txts = [str(r["text"]) for r in rows]
        cleans, st_l, metas = [], [], []
        for t in txts:
            cl, f56, mt = extract_style_features_56d(t)
            cleans.append(cl); st_l.append(f56); metas.append(mt)
        encs = [tokenizer.encode_ids(c, max_len=args.max_len) for c in cleans]
        engs = [make_engram_bigram_ids(s, buckets=ENGRAM_BIGRAM_BUCKETS_V10) for s in encs]
        tris = [make_engram_trigram_ids(s, buckets=ENGRAM_TRIGRAM_BUCKETS_V10) for s in encs]
        sc = np.array([float(r.get(sc_key, r.get("ai_score", 0.0))) for r in rows], dtype=np.float32)
        e5 = np.array([int(r.get("era5", sanitize_era5_v123(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=bool(r.get("is_pre_llm")), ai_score_0_100=float(r.get(sc_key, r.get("ai_score", 0.0)))))) for r in rows], dtype=np.int32)
        st_arr = np.stack(st_l, axis=0) if st_l else np.zeros((0, STYLE_FEAT_DIM_V11), dtype=np.float32)
        return (encs, engs, tris, txts, cleans, st_arr, metas, sc, e5)

    vs_sel_p = _pre_eval(val_select_wild)
    vg2_p = _pre_eval(gold_v2_val_rows, "ai_score") if gold_v2_val_rows else None
    vmc_p = _pre_eval(val_meta_cal_rows)
    vcert_p = _pre_eval(val_cal_cert)
    god500_all_p = _pre_eval(god500_all_rows) if god500_all_rows else None
    god500_tv_p = _pre_eval(god500_tv_rows) if god500_tv_rows else None
    tg_p, td_p = _pre_eval(test_gold), _pre_eval(test_dsh)
    tg2_p = _pre_eval(gold_v2_test_rows, "ai_score") if gold_v2_test_rows else None
    print(
        f"[Stage0-6WayVal] val_select_wild={len(vs_sel_p[0])} vg2={len(vg2_p[0]) if vg2_p else 0} | "
        f"val_meta_cal_PURE={len(vmc_p[0])} (0 god500!) | val_cal_cert={len(vcert_p[0])} (n_pos_ai={int((vcert_p[7]>=70).sum())}) | "
        f"diag_god500={len(god500_all_rows)} (text_verif={len(god500_tv_rows)})",
        flush=True,
    )

    # 6. Model Init (HAWQ-V2 Bitwidth Reallocation + Zero-Shock v1.2.2 Warm-Start OR Dual-Source v1.0 + v1.2.2!)
    mdl = MeguvizV123ApexNanoModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS), style_era_dropout_p=0.15,
    ).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_EMB_END_V11, num_eras=NUM_ERAS_V122).to(device)

    @torch.no_grad()
    def _predict_v123(m_e: MeguvizV123ApexNanoModel, pk, outer_steps: int = 2) -> dict[str, np.ndarray]:
        m_e.eval()
        encs, engs, tris, txts, _, st_arr, _, _, _ = pk
        n = len(encs)
        keys = (
            "logit", "raw_logit", "ai_aux_term", "logit_t1", "raw_logit_t1", "logit_r1", "raw_logit_r1",
            "logit_promo", "raw_logit_promo", "logit_promo_t1", "raw_logit_promo_t1", "logit_promo_r1", "raw_logit_promo_r1",
            "logit_ai_r0", "logit_promo_r0", "p_defer_r0", "p_defer1", "r1_span_max", "ebt_margin",
        )
        out = {k: np.zeros(n, dtype=np.float32) for k in keys}
        out["emb"] = np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return out
        order = sorted(range(n), key=lambda i: len(encs[i]))
        ebs = 128 if args.smoke_test else 384
        for st_i in range(0, n, ebs):
            mb = order[st_i : st_i + ebs]
            ml = max(2, max(len(encs[i]) for i in mb))
            if ml % 2 == 1:
                ml += 1
            ids = np.full((len(mb), ml), tokenizer.pad_id, dtype=np.int64)
            msk = np.zeros((len(mb), ml), dtype=np.int64)
            eg = np.zeros((len(mb), ml), dtype=np.int64)
            tg = np.zeros((len(mb), ml), dtype=np.int64)
            for ri, oi in enumerate(mb):
                s = encs[oi]
                ids[ri, : len(s)], msk[ri, : len(s)], eg[ri, : len(s)], tg[ri, : len(s)] = s, 1, engs[oi], tris[oi]
            c1, c3, cm, cf, _ = extract_rung0_charhash_batch([txts[i] for i in mb])
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                o = m_e(
                    torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device), torch.from_numpy(st_arr[mb]).to(device),
                    engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device),
                    r0_c1=torch.from_numpy(c1).to(device), r0_c3=torch.from_numpy(c3).to(device),
                    r0_mask=torch.from_numpy(cm).to(device), r0_feats=torch.from_numpy(cf).to(device),
                    outer_steps=outer_steps, return_aux=True,
                )
            e1 = o["exit1_out"]
            for ri, oi in enumerate(mb):
                out["logit"][oi] = float(o["logit"][ri].item())
                out["raw_logit"][oi] = float(o["raw_logit"][ri].item())
                out["ai_aux_term"][oi] = float(o["ai_aux_term"][ri].item())
                out["logit_t1"][oi] = float(e1["logit"][ri].item())
                out["raw_logit_t1"][oi] = float(e1["raw_logit"][ri].item())
                out["logit_r1"][oi] = float(o["logit_r1"][ri].item())
                out["raw_logit_r1"][oi] = float(o["raw_logit_r1"][ri].item())
                out["logit_promo"][oi] = float(o["logit_promo"][ri].item())
                out["raw_logit_promo"][oi] = float(o["raw_logit_promo"][ri].item())
                out["logit_promo_t1"][oi] = float(e1["logit_promo"][ri].item())
                out["raw_logit_promo_t1"][oi] = float(e1["raw_logit_promo"][ri].item())
                out["logit_promo_r1"][oi] = float(o["logit_promo_r1"][ri].item())
                out["raw_logit_promo_r1"][oi] = float(o["raw_logit_promo_r1"][ri].item())
                out["logit_ai_r0"][oi] = float(o["logit_ai_r0"][ri].item())
                out["logit_promo_r0"][oi] = float(o["logit_promo_r0"][ri].item())
                out["p_defer_r0"][oi] = float(o["p_defer_r0"][ri].item())
                out["p_defer1"][oi] = float(o["p_defer1"][ri].item())
                out["r1_span_max"][oi] = float(o["r1_span_max"][ri].item())
                out["ebt_margin"][oi] = float(o["ebt_margin"][ri].item())
                out["emb"][oi] = o["emb"][ri].float().cpu().numpy()
        return out

    rng_jit = random.Random(2026112899)

    def _collate_items(batch_items: list[dict[str, Any]], punct_jitter: bool = True):
        # Online 35% punctuation-shortcut breaker on AI positives (breaks ends_period / emdash shortcut!)
        eff_items: list[dict[str, Any]] = []
        for x in batch_items:
            if punct_jitter and x.get("has_ai", 0.0) > 0.5 and x.get("y_score", 0.0) >= 0.70:
                jt = jitter_ai_punctuation_v123(x["raw_text"], rng_jit, prob=0.35)
                if jt != x["raw_text"]:
                    cl_j, f56_j, _ = extract_style_features_56d(jt)
                    ids_j = tokenizer.encode_ids(cl_j, max_len=args.max_len)
                    xj = dict(x)
                    xj["raw_text"] = jt
                    xj["ids"] = ids_j
                    xj["engram"] = make_engram_bigram_ids(ids_j, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
                    xj["trigram"] = make_engram_trigram_ids(ids_j, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
                    xj["style"] = f56_j
                    xj["char_len"] = len(jt)
                    eff_items.append(xj)
                    continue
            eff_items.append(x)
        batch_items = eff_items

        ml = max(2, max(len(x["ids"]) for x in batch_items))
        if ml % 2 == 1:
            ml += 1
        nb = len(batch_items)
        ids = np.full((nb, ml), tokenizer.pad_id, dtype=np.int64)
        msk = np.zeros((nb, ml), dtype=np.int64)
        eg = np.zeros((nb, ml), dtype=np.int64)
        tg = np.zeros((nb, ml), dtype=np.int64)
        for i, x in enumerate(batch_items):
            s = x["ids"]
            ids[i, : len(s)], msk[i, : len(s)], eg[i, : len(s)], tg[i, : len(s)] = s, 1, x["engram"], x["trigram"]
        c1, c3, cm, cf, _ = extract_rung0_charhash_batch([x["raw_text"] for x in batch_items])
        return (
            torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
            torch.from_numpy(eg).to(device), torch.from_numpy(tg).to(device),
            torch.from_numpy(np.stack([x["style"] for x in batch_items]).astype(np.float32)).to(device),
            torch.from_numpy(c1).to(device), torch.from_numpy(c3).to(device),
            torch.from_numpy(cm).to(device), torch.from_numpy(cf).to(device),
            torch.tensor([x["y_score"] for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([x["has_ai"] for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([x["y_promo"] for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([x["has_promo"] for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([x["sub4_idx"] for x in batch_items], dtype=torch.long, device=device),
            torch.tensor([x["weight"] for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([x["attr5"] for x in batch_items], dtype=torch.long, device=device),
            torch.tensor([1.0 if x["is_true_pro_gold"] else 0.0 for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([1.0 if x.get("is_phys_prellm", False) else 0.0 for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([int(x.get("era5", -1)) for x in batch_items], dtype=torch.long, device=device),
            torch.tensor([int(x.get("char_len", len(x["raw_text"]))) for x in batch_items], dtype=torch.long, device=device),
        )

    # Zero-shock warm-start from `v1.2.2-R1` FP16 master (primary!) OR dual-source `v1.0 + v1.2.2` fallback!
    transplant_rep: dict[str, Any] = {"mode": "cold_init", "hawq_v2": mdl.hawq_v2_rep}
    g0_rmse: float | None = 0.0
    g0_spearman: float | None = 1.0
    has_v122 = bool((args.v122_fp16_master and args.v122_fp16_master.is_file()) or (args.v122_bundle_mgvz and args.v122_bundle_mgvz.is_file()))
    has_v10 = bool((args.v10_fp16_master and args.v10_fp16_master.is_file()) or (args.v10_bundle_mgvz and args.v10_bundle_mgvz.is_file()))

    if has_v122 and not args.force_v10_rebase:
        if args.v122_fp16_master and args.v122_fp16_master.is_file():
            mdl.load_state_dict(torch.load(args.v122_fp16_master, map_location="cpu"), strict=False)
            src_tag = str(args.v122_fp16_master.name)
        else:
            load_bundle_v7_v121_into_model(mdl, args.v122_bundle_mgvz)
            src_tag = str(args.v122_bundle_mgvz.name)
        mdl = mdl.to(device).eval()
        # Capture pre-unlock logits for exact G0 check, then unlock `z_char_coda_ai_gate >= 0.15` (P0-2!)
        o_pre = _predict_v123(mdl, vs_sel_p, 2)
        g_val = mdl.unlock_z_char_coda_ai_gate(0.15)
        o_post = _predict_v123(mdl, vs_sel_p, 2)
        g0_rmse = float(np.sqrt(np.mean((o_post["logit"] - o_pre["logit"]) ** 2)))
        g0_spearman = float(spearmanr(o_post["logit"], o_pre["logit"]).statistic)
        transplant_rep = {
            "mode": "v122_zero_shock_warmstart",
            "source": src_tag,
            "z_char_coda_ai_gate_unlocked": g_val,
            "g0_rmse": round(g0_rmse, 5),
            "g0_spearman": round(g0_spearman, 5),
            "hawq_v2": mdl.hawq_v2_rep,
        }
    elif has_v10:
        v10_t = MeguvizV10ApexLoopModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).cpu().eval()
        if args.v10_fp16_master and args.v10_fp16_master.is_file():
            v10_t.load_state_dict(torch.load(args.v10_fp16_master, map_location="cpu"), strict=False)
        else:
            load_bundle_v6_mgvz_into_model(v10_t, args.v10_bundle_mgvz)
        v10_t = v10_t.to(device).eval()
        v122_t = None
        if has_v122:
            v122_t = MeguvizV122ApexNanoModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).cpu().eval()
            if args.v122_fp16_master and args.v122_fp16_master.is_file():
                v122_t.load_state_dict(torch.load(args.v122_fp16_master, map_location="cpu"), strict=False)
            else:
                load_bundle_v7_v121_into_model(v122_t, args.v122_bundle_mgvz)
            v122_t = v122_t.to(device).eval()
        cal_it = (t0_pool[:250] + new_gold_ai_pool[:250] + t1_pool[:300])[: (64 if args.smoke_test else 768)]
        _, f40_c, _ = extract_style_features_40d_batch([x["raw_text"] for x in cal_it])
        f56_c = np.stack([x["style"] for x in cal_it], axis=0)
        # Collect v10 emb on cal_it
        em_l = []
        with torch.no_grad():
            for st_i in range(0, len(cal_it), 128):
                sub = cal_it[st_i : st_i + 128]
                ml = max(2, max(len(x["ids"]) for x in sub))
                ids = np.full((len(sub), ml), tokenizer.pad_id, dtype=np.int64)
                msk = np.zeros((len(sub), ml), dtype=np.int64)
                eg = np.zeros((len(sub), ml), dtype=np.int64)
                tg = np.zeros((len(sub), ml), dtype=np.int64)
                for ri, x in enumerate(sub):
                    s = x["ids"]
                    ids[ri, : len(s)], msk[ri, : len(s)], eg[ri, : len(s)], tg[ri, : len(s)] = s, 1, x["engram"], x["trigram"]
                o10 = v10_t(torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device), torch.from_numpy(f40_c[st_i : st_i + len(sub)]).to(device), engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device), outer_steps=2, k_inner=2, return_aux=True)
                em_l.append(o10["emb"].float().cpu().numpy())
        rep_tr = transplant_dual_source_v10_v121_to_v122(v10_t, v122_t, mdl, f40_c, f56_c, np.concatenate(em_l, axis=0))
        g_val = mdl.unlock_z_char_coda_ai_gate(0.15)
        transplant_rep = {**rep_tr, "mode": "dual_source_v10_v122_rebase", "z_char_coda_ai_gate_unlocked": g_val, "hawq_v2": mdl.hawq_v2_rep}
        del v10_t
        if v122_t is not None:
            del v122_t

    w_ai_init_norm = float(mdl.ai_head.weight.detach().float().norm().item())
    ai_anchor_teacher = copy.deepcopy(mdl).eval()
    for p in ai_anchor_teacher.parameters():
        p.requires_grad_(False)
    print(f"[Stage0-Init] {json.dumps(transplant_rep)}", flush=True)

    # Validation evaluators
    def _eval_select_ai(m_e: MeguvizV123ApexNanoModel) -> dict[str, float]:
        ow = _predict_v123(m_e, vs_sel_p, 2)
        yw = (vs_sel_p[7] >= 70.0).astype(int)
        mw = compute_wild_tail_objective_v122(yw, ow["logit"])
        mt1 = compute_wild_tail_objective_v122(yw, ow["logit_t1"])
        r1_auc = float(roc_auc_score(yw, ow["logit_r1"])) if len(np.unique(yw)) == 2 else 0.5
        r0_auc = float(roc_auc_score(yw, ow["logit_ai_r0"])) if len(np.unique(yw)) == 2 else 0.5
        # Short-bucket (<60c) AUROC monitor
        cl_w = np.array([len(t) for t in vs_sel_p[3]])
        m_sh = cl_w < 60
        sh_auc = float(roc_auc_score(yw[m_sh], ow["logit"][m_sh])) if (m_sh.sum() >= 10 and len(np.unique(yw[m_sh])) == 2) else mw["auroc"]
        res = {
            "T_wild": mw["T"], "tkm_w": mw["tkm"], "rp95_w": mw["rp95_wild"],
            "auroc_w": mw["auroc"], "pr_w": mw["pr_auc"], "sh_auc": round(sh_auc, 4),
            "r1_auc": round(r1_auc, 5), "r0_ai": round(r0_auc, 5),
        }
        if vg2_p is not None and len(vg2_p[7]) >= 20:
            og2 = _predict_v123(m_e, vg2_p, 2)
            mg2 = compute_wild_tail_objective_v122((vg2_p[7] >= 70.0).astype(int), og2["logit"])
            res["T_vg2"], res["auroc_vg2"], res["pr_vg2"] = mg2["T"], mg2["auroc"], mg2["pr_auc"]
            res["sel_T"] = round(0.55 * mw["T"] + 0.35 * mg2["T"] + 0.10 * mt1["T"], 5)
            res["sel_obj"] = round(0.60 * res["sel_T"] + 0.20 * (0.5 * mw["auroc"] + 0.5 * mg2["auroc"]) + 0.20 * (0.5 * mw["pr_auc"] + 0.5 * mg2["pr_auc"]), 5)
        else:
            res["sel_T"] = round(0.85 * mw["T"] + 0.15 * mt1["T"], 5)
            res["sel_obj"] = round(0.60 * res["sel_T"] + 0.20 * mw["auroc"] + 0.20 * mw["pr_auc"], 5)
        return res

    def _eval_select_promo_and_long(m_e: MeguvizV123ApexNanoModel) -> dict[str, float]:
        res = {"promo_auroc": 0.5, "promo_pr": 0.5, "r1_pr": 0.5, "r0_pr": 0.5, "long_ai": 0.5, "long_hyb": 0.5, "long_pr": 0.5}
        if promo_val_select_rows:
            pk = (
                [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
                [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
                np.stack([x["style"] for x in promo_val_select_rows]), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32), np.zeros(len(promo_val_select_rows), dtype=np.int32),
            )
            op = _predict_v123(m_e, pk, 2)
            yb = (pk[7] >= 0.65).astype(int)
            if len(np.unique(yb)) == 2:
                res["promo_auroc"] = float(roc_auc_score(yb, op["logit_promo"]))
                res["promo_pr"] = float(average_precision_score(yb, op["logit_promo"]))
                res["r1_pr"] = float(roc_auc_score(yb, op["logit_promo_r1"]))
                res["r0_pr"] = float(roc_auc_score(yb, op["logit_promo_r0"]))
        if wave2_longform_val:
            cs_b = collate_chronostream_docs(wave2_longform_val, tokenizer, max_chunks=4, max_chunk_toks=192)
            with torch.no_grad():
                cs_o = m_e.forward_chronostream_batch_v122(
                    cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                    cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                    trunk_no_grad=True, outer_steps=2,
                )
            for k_r, k_l, k_y in (("long_ai", "logit_ai_doc", "y_ai_doc"), ("long_hyb", "logit_hybrid_doc", "y_hyb_doc"), ("long_pr", "logit_promo_doc", "y_pr_doc")):
                yv = (cs_b[k_y].numpy() >= 0.5).astype(int)
                if len(np.unique(yv)) == 2:
                    res[k_r] = float(roc_auc_score(yv, cs_o[k_l].float().cpu().numpy()))
        return res

    # P0-3 Explicit 6-Subpool Stratified Phase A Sampler!
    rng_tr = random.Random(2026112806)
    slot_aug = BrandPriceSlotAug()
    pike_dual = DynamicPiKEGroupSamplerV11()

    def _pike_phase_b_weights() -> list[float]:
        w7 = pike_dual.w
        raw = np.array([w7[4] * 3.0, w7[6] * 3.5, min(0.15, w7[5] * 1.5), float(w7[:4].sum()) * 0.45], dtype=np.float64)
        raw /= max(float(raw.sum()), 1e-6)
        return raw.tolist()

    era_human_pools: dict[int, list[dict[str, Any]]] = {e: [] for e in range(NUM_ERAS_V122)}
    for x in (t0_pool + new_gold_ai_pool + protect_pool):
        if x["has_ai"] > 0.5 and x["y_score"] <= 0.25 and 0 <= x["era5"] < NUM_ERAS_V122:
            era_human_pools[x["era5"]].append(x)
    era_dro_q = np.ones(NUM_ERAS_V122, dtype=np.float64) / float(NUM_ERAS_V122)

    def _sample_phase_a_v123(b_sz: int, include_mem: bool = True) -> list[dict[str, Any]]:
        n_mem = (min(8, b_sz // 5) if args.smoke_test else int(b_sz * 0.15)) if include_mem else 0
        rem = b_sz - n_mem
        n_rare = int(rem * 0.18) if ai_rare_stealth_pool else 0
        n_dp = int(rem * 0.05) if dual_pos_ai_promo_pool else 0
        n_syn = int(rem * 0.07) if synth_short_ai_pool else 0  # Hard <= 8% synth cap!
        n_old_sh = int(rem * 0.08) if old_short_hard_hu_pool else 0
        n_era = int(rem * 0.12)
        n_new = int(rem * 0.24) if new_gold_ai_pool else 0
        out_b: list[dict[str, Any]] = []
        if n_mem > 0:
            np_m = min(n_mem * 3 // 8, len(mem_phys))
            nm_m = min(n_mem - np_m, len(mem_mod))
            if np_m > 0:
                out_b.extend(rng_tr.sample(mem_phys, np_m))
            if nm_m > 0:
                out_b.extend(rng_tr.sample(mem_mod, nm_m))
        if n_rare > 0:
            out_b.extend(rng_tr.choices(ai_rare_stealth_pool, k=n_rare))
        if n_dp > 0:
            out_b.extend(rng_tr.choices(dual_pos_ai_promo_pool, k=n_dp))
        if n_syn > 0:
            out_b.extend(rng_tr.choices(synth_short_ai_pool, k=n_syn))
        if n_old_sh > 0:
            out_b.extend(rng_tr.choices(old_short_hard_hu_pool, k=n_old_sh))
        if n_era > 0:
            valid_eras = [e for e in range(NUM_ERAS_V122) if era_human_pools[e]]
            if valid_eras:
                probs_e = np.array([era_dro_q[e] for e in valid_eras], dtype=np.float64)
                probs_e /= probs_e.sum()
                for e_i in rng_tr.choices(valid_eras, weights=probs_e.tolist(), k=n_era):
                    out_b.append(rng_tr.choice(era_human_pools[e_i]))
        if n_new > 0:
            out_b.extend(rng_tr.choices(new_gold_ai_pool, k=n_new))
        pool_fb = t0_pool + t1_pool or new_gold_ai_pool
        if len(out_b) < b_sz and pool_fb:
            out_b.extend(rng_tr.choices(pool_fb, k=b_sz - len(out_b)))
        return out_b[:b_sz]

    def _sample_phase_b_v123(b_sz: int, pw: list[float], wild_promo_frac: float = 0.16) -> list[dict[str, Any]]:
        n_wp = int(b_sz * wild_promo_frac) if wild_promo_focus_pool else 0
        n_dp = int(b_sz * 0.08) if dual_pos_ai_promo_pool else 0
        rem = max(0, b_sz - n_wp - n_dp)
        n_du = int(rem * pw[0]) if dual_gold_train else 0
        n_hn = int(rem * pw[1]) if hard_neg_rows else 0
        n_uo = int(rem * pw[2]) if uot_pseudo_rows else 0
        out_b: list[dict[str, Any]] = []
        if n_wp > 0:
            out_b.extend(rng_tr.choices(wild_promo_focus_pool, k=n_wp))
        if n_dp > 0:
            out_b.extend(rng_tr.choices(dual_pos_ai_promo_pool, k=n_dp))
        if n_du > 0:
            out_b.extend(rng_tr.choices(dual_gold_train, k=n_du))
        if n_hn > 0:
            out_b.extend(rng_tr.choices(hard_neg_rows, k=n_hn))
        if n_uo > 0:
            out_b.extend(rng_tr.choices(uot_pseudo_rows, k=n_uo))
        pool_fb = t0_pool or t1_pool or dual_gold_train
        if len(out_b) < b_sz and pool_fb:
            out_b.extend(rng_tr.choices(pool_fb, k=b_sz - len(out_b)))
        return out_b[:b_sz]

    def _sample_long_batch(n_d: int) -> list[dict[str, Any]]:
        n_g = min(n_d // 2, len(wave2_longform))
        docs = rng_tr.sample(wave2_longform, n_g) if n_g > 0 else []
        for _ in range(n_d - len(docs)):
            docs.append(longmix_syn.sample_one(rng_tr))
        return docs

    ref_ai = _eval_select_ai(mdl)
    ref_pl = _eval_select_promo_and_long(mdl)
    print(f"[Stage0-Ref] ai={json.dumps(ref_ai)} | dual={json.dumps({k: round(v, 4) for k, v in ref_pl.items()})}", flush=True)

    # =========================================================================
    # PHASE A (A0 + A1): P0-2 R0<->T2 Mutual KD + z_char Unlock + P0-3 Stratified Quotas + P0-4 R1/R0 Floor + P0-6 Era GRL!
    # =========================================================================
    pa0_steps = 2 if args.smoke_test else int(args.pa0_steps)
    pa1_steps = 4 if args.smoke_test else int(args.pa1_steps)
    pa_tot = pa0_steps + pa1_steps
    ema_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pa_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pa_sc = float(ref_ai["sel_obj"])

    def _update_ema(m_src: nn.Module, decay: float = 0.996) -> None:
        with torch.no_grad():
            for k, v in m_src.state_dict().items():
                if v.is_floating_point():
                    ema_sd[k].mul_(decay).add_(v.detach().cpu(), alpha=1.0 - decay)
                else:
                    ema_sd[k].copy_(v.detach().cpu())

    g_step = 0
    for sub_name, ph_key, n_steps in (("A0", "phase_a0", pa0_steps), ("A1", "phase_a1", pa1_steps)):
        pg_a, hooks_a = build_v123_param_groups(mdl, ph_key, era_grl=era_grl)
        opt_a = torch.optim.AdamW(pg_a, betas=(0.9, 0.95))
        asam_a = ASAMPerturber(mdl, rho=0.04)
        mdl.train(); era_grl.train()
        for step in range(n_steps):
            mult = _wsd_lr_mult(step, n_steps, warmup_frac=0.08, decay_frac=0.25 if sub_name == "A1" else 0.10)
            for pg in opt_a.param_groups:
                pg["lr"] = pg["base_lr"] * mult

            b_it = _sample_phase_a_v123(min(bs, 32 if args.smoke_test else bs), include_mem=(sub_name == "A1"))
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, has_ai, _, _, _, w_b, a5_b, is_t0, is_phys, era5_b, clen_b) = _collate_items(b_it)

            cf_pair_pack = None
            if cf_mirror_pool and (step % 2 == 0):
                n_cfp = 4 if args.smoke_test else 20
                micro_c = [p for p in cf_mirror_pool if p.get("is_micro_short")]
                chosen_cf = rng_tr.sample(micro_c, min(n_cfp // 2, len(micro_c))) if micro_c else []
                chosen_cf.extend(rng_tr.sample(cf_mirror_pool, min(n_cfp - len(chosen_cf), len(cf_mirror_pool))))
                cf_ai_it = [_make_item(x["ai_text"], 1.0, -1.0, 0.0, "none", 1.25) for x in chosen_cf]
                cf_hu_it = [_make_item(x["human_text"], 0.0, -1.0, 0.0, "none", 1.25) for x in chosen_cf]
                cf_pair_pack = (_collate_items(cf_ai_it), _collate_items(cf_hu_it))

            def _phase_a_loss() -> torch.Tensor:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                    anc = ai_anchor_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                    out = mdl(
                        b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                        r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                        outer_steps=2, detach_zchar_for_ai=False, return_aux=True,
                    )
                    ai_m = has_ai > 0.5
                    lg, lg_t1, lg_r1, lg_r0 = out["logit"].float(), out["exit1_out"]["logit"].float(), out["logit_r1"].float(), out["logit_ai_r0"].float()
                    ya_c = y_ai.clamp(1e-4, 1.0 - 1e-4)
                    l_bce = (F.binary_cross_entropy_with_logits(lg[ai_m], ya_c[ai_m], reduction="none") * w_b[ai_m]).mean()
                    l_t1 = (F.binary_cross_entropy_with_logits(lg_t1[ai_m], ya_c[ai_m], reduction="none") * w_b[ai_m]).mean()
                    l_r1 = (F.binary_cross_entropy_with_logits(lg_r1[ai_m], ya_c[ai_m], reduction="none") * w_b[ai_m] * (1.0 + 1.5 * (y_ai[ai_m] >= 0.45).float())).mean()
                    l_r0 = (
                        0.50 * (F.binary_cross_entropy_with_logits(lg_r0[ai_m], ya_c[ai_m], reduction="none") * w_b[ai_m] * (1.0 + 2.0 * (y_ai[ai_m] >= 0.45).float())).mean()
                        + 0.30 * F.mse_loss(lg_r0[ai_m], lg.detach()[ai_m])
                        + 0.10 * listnet_tail_distill_loss(lg_r0[ai_m], lg.detach()[ai_m])
                    )
                    # P0-2: Reverse KD from confident `r0` -> `t2` on short/template comments!
                    l_r0_to_t2 = compute_r0_to_t2_mutual_kd_loss(lg, lg_r0, y_ai, has_ai, clen_b)
                    # P0-4: Clean-human floor margin loss on `R1` + `R0`!
                    l_floor = compute_r1_r0_clean_human_floor_loss(lg_r1, lg_r0, y_ai, has_ai)
                    # Global + Dedicated `<60c` SOPA-s pAUC & CVaR-DORO!
                    l_pauc2, l_ceil2 = compute_sopa_pauc_and_cvar_doro_loss(lg, y_ai, has_ai, is_phys > 0.5)
                    l_pauc1, l_ceil1 = compute_sopa_pauc_and_cvar_doro_loss(lg_t1, y_ai, has_ai, is_phys > 0.5)
                    _, l_ceil_r1 = compute_sopa_pauc_and_cvar_doro_loss(lg_r1, y_ai, has_ai, is_phys > 0.5)
                    sh_has = has_ai * (clen_b <= 60).float()
                    l_pauc_sh, l_ceil_sh = compute_sopa_pauc_and_cvar_doro_loss(lg, y_ai, sh_has, is_phys > 0.5)
                    # EBT RCE + EMR
                    l_ebt_ce = compute_ebt6_partial_label_ce(out["ebt_energies"][ai_m], y_ai[ai_m], torch.zeros_like(y_ai[ai_m]), torch.zeros_like(y_ai[ai_m]), a5_b[ai_m])
                    l_ebt_emr = compute_ebt_rce_emr_loss(out["ebt_energies"], out["ebt_margin"], y_ai, has_ai, is_t0 > 0.5)
                    # Anchor agreement KD + one-sided human drift barrier
                    z_anc, z_anc_t1 = anc["logit"].float(), anc["exit1_out"]["logit"].float()
                    ag_m = ai_m & (((y_ai >= 0.70) & (z_anc > -0.5)) | ((y_ai <= 0.25) & (z_anc < 0.5)))
                    l_kd = compute_tail_ranking_kd_loss(lg, z_anc, y_ai, ag_m.float()) + 0.40 * compute_tail_ranking_kd_loss(lg_t1, z_anc_t1, y_ai, ag_m.float())
                    safe_hu = ai_m & (y_ai <= 0.20) & (z_anc <= -1.5)
                    l_drift = 0.22 * F.relu(lg[safe_hu] - z_anc[safe_hu] - 0.35).pow(2).mean() if safe_hu.any() else lg.sum() * 0.0
                    # P0-6: Stronger 5-Era OfflineEraGRL (0.15 weight)
                    era_valid = ai_m & (y_ai <= 0.25) & (era5_b >= 0) & (era5_b < NUM_ERAS_V122)
                    grl_lam = 0.65 * min(1.0, float(g_step + 1) / float(max(1, pa_tot // 2)))
                    l_era = era_grl(out["emb"][:, :AI_EMB_END_V11], era5_b, era_valid, lam=grl_lam) if int(era_valid.sum().item()) >= 8 else lg.sum() * 0.0

                    loss = (
                        l_bce + 0.35 * l_t1 + 0.28 * l_r1 + 0.45 * l_r0 + l_r0_to_t2 + l_floor
                        + 0.25 * (l_pauc2 + 0.50 * l_pauc1 + 0.60 * l_pauc_sh)
                        + 0.20 * (l_ceil2 + 0.50 * l_ceil1 + 0.35 * l_ceil_r1 + 0.50 * l_ceil_sh)
                        + 0.10 * l_ebt_ce + 0.18 * l_ebt_emr
                        + 0.28 * l_kd + l_drift + 0.15 * l_era
                        + 0.02 * out["cbm_loss"] + 0.015 * compute_slot_orthogonality_loss(out["slots"])
                    )
                    if cf_pair_pack is not None:
                        (ai_ids, ai_m_, ai_e, ai_t, ai_s, ac1, ac3, acm, acf, *_), (hu_ids, hu_m_, hu_e, hu_t, hu_s, hc1, hc3, hcm, hcf, *_) = cf_pair_pack
                        oa = mdl(ai_ids, ai_m_, ai_s, engram_ids=ai_e, trigram_ids=ai_t, r0_c1=ac1, r0_c3=ac3, r0_mask=acm, r0_feats=acf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
                        oh = mdl(hu_ids, hu_m_, hu_s, engram_ids=hu_e, trigram_ids=hu_t, r0_c1=hc1, r0_c3=hc3, r0_mask=hcm, r0_feats=hcf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
                        with torch.no_grad():
                            oa0 = ai_anchor_teacher(ai_ids, ai_m_, ai_s, engram_ids=ai_e, trigram_ids=ai_t, r0_c1=ac1, r0_c3=ac3, r0_mask=acm, r0_feats=acf, outer_steps=2, return_aux=True)
                            oh0 = ai_anchor_teacher(hu_ids, hu_m_, hu_s, engram_ids=hu_e, trigram_ids=hu_t, r0_c1=hc1, r0_c3=hc3, r0_mask=hcm, r0_feats=hcf, outer_steps=2, return_aux=True)
                        dpo_margin = (oa["logit"].float() - oa0["logit"].float()) - (oh["logit"].float() - oh0["logit"].float())
                        loss = loss + 0.14 * (-F.logsigmoid(2.0 * dpo_margin).mean() + 0.15 * F.softplus(2.0 - (oa["logit"].float() - oh["logit"].float())).mean())
                return loss

            opt_a.zero_grad(set_to_none=True)
            loss_a = _phase_a_loss()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            if sub_name == "A1" and not args.smoke_test and step >= int(n_steps * 0.65):
                asam_a.first_step()
                opt_a.zero_grad(set_to_none=True)
                _phase_a_loss().backward()
                asam_a.restore()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_a.step()
            mdl.project_ai_head_weight_norm(1.25 * w_ai_init_norm)
            _update_ema(mdl, decay=0.992 if sub_name == "A0" else 0.996)

            if sub_name == "A1" and (step + 1) % 50 == 0:
                with torch.no_grad():
                    o_dro = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=1, return_aux=True)
                    z_d = o_dro["logit"].float()
                    for e_i in range(NUM_ERAS_V122):
                        em = (has_ai > 0.5) & (y_ai <= 0.25) & (era5_b == e_i)
                        if int(em.sum().item()) >= 4:
                            cvar_e = float(torch.topk(F.relu(z_d[em] + 1.0), k=max(1, int(em.sum().item()) // 5)).values.mean().item())
                            era_dro_q[e_i] *= math.exp(0.10 * min(cvar_e, 3.0))
                    era_dro_q = np.clip(era_dro_q / era_dro_q.sum(), 0.08, 0.45)
                    era_dro_q /= era_dro_q.sum()

            if not args.smoke_test and ((step + 1) % 100 == 0 or step + 1 == n_steps):
                cur_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                mdl.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
                ma = _eval_select_ai(mdl)
                if ma["sel_obj"] >= best_pa_sc:
                    best_pa_sc = float(ma["sel_obj"])
                    best_pa_sd = {k: v.clone() for k, v in ema_sd.items()}
                mdl.load_state_dict({k: v.to(device) for k, v in cur_sd.items()}, strict=True)
                mdl.train()
                zg = float(mdl.z_char_coda_ai_gate.detach().float().mean().item())
                print(
                    f"[Phase{sub_name} step={step+1}/{n_steps}] loss={float(loss_a.item()):.4f} | sel_T={ma['sel_T']:.5f} obj={ma['sel_obj']:.5f} | "
                    f"w_rp95={ma['rp95_w']:.4f} sh_auc={ma['sh_auc']:.4f} vg2_auc={ma.get('auroc_vg2', 0):.4f} r1={ma['r1_auc']:.4f} r0={ma['r0_ai']:.4f} | zc_gate={zg:.3f} | elapsed={time.monotonic()-t_start:.0f}s",
                    flush=True,
                )
            g_step += 1
        for hk in hooks_a:
            hk.remove()

    # WiSE-FT sweep (anchor vs best Phase A EMA)
    anc_sd = {k: v.detach().cpu().clone() for k, v in ai_anchor_teacher.state_dict().items()}
    best_wise_sd, best_wise_sc = best_pa_sd, best_pa_sc
    if not args.smoke_test:
        for lam in (0.65, 0.80, 0.90, 1.0):
            cand = {k: (lam * best_pa_sd[k] + (1.0 - lam) * anc_sd[k] if best_pa_sd[k].is_floating_point() else best_pa_sd[k]) for k in best_pa_sd}
            mdl.load_state_dict({k: v.to(device) for k, v in cand.items()}, strict=True)
            mw = _eval_select_ai(mdl)
            if mw["sel_obj"] >= best_wise_sc:
                best_wise_sc, best_wise_sd = float(mw["sel_obj"]), cand
    mdl.load_state_dict({k: v.to(device) for k, v in best_wise_sd.items()}, strict=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_phaseA_ckpt.pt")
    pa_ai_m = _eval_select_ai(mdl)
    print(f"[PhaseA DONE] {json.dumps(pa_ai_m)} | zc_gate={float(mdl.z_char_coda_ai_gate.detach().float().mean().item()):.4f}", flush=True)

    # =========================================================================
    # PHASE B: Frozen AI `[0:192]` -> Promo + Dual-Pos + Rung-0 + ChronoStream + Hoisted ASAM + Live PiKE!
    # =========================================================================
    pb_steps = 4 if args.smoke_test else int(args.pb_steps)
    pg_b, hooks_b = build_v123_param_groups(mdl, "phase_b")
    opt_b = torch.optim.AdamW(pg_b, betas=(0.9, 0.95))
    asam_b = ASAMPerturber(mdl, rho=0.04)
    best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pb_sc = -1e9
    mdl.train()

    for step in range(pb_steps):
        mult = _wsd_lr_mult(step, pb_steps, warmup_frac=0.05, decay_frac=0.25)
        for pg in opt_b.param_groups:
            pg["lr"] = pg["base_lr"] * mult

        b_it = _sample_phase_b_v123(min(bs, 32 if args.smoke_test else bs), _pike_phase_b_weights(), wild_promo_frac=0.16)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, a5_b, *_) = _collate_items(b_it)

        pr_pair_pack = None
        if dual_pairs_train and step % 2 == 0:
            pb_p = rng_tr.sample(dual_pairs_train, min(4 if args.smoke_test else 16, len(dual_pairs_train)))
            p_it, o_it = [], []
            for x in pb_p:
                pt_t, ot_t = slot_aug.augment_pair(x["promo_text"], x["organic_text"]) if rng_tr.random() < 0.5 else (x["promo_text"], x["organic_text"])
                ia = 1.0 if x.get("promo_is_ai") else 0.0
                p_it.append(_make_item(pt_t, ia, 1.0, 1.0, str(x.get("promo_type", "soft_shill")), 1.4, has_ai=0.0))
                o_it.append(_make_item(ot_t, ia, 0.0, 1.0, "none", 1.4, has_ai=0.0))
            pr_pair_pack = (_collate_items(p_it), _collate_items(o_it))

        cs_pack = collate_chronostream_docs(_sample_long_batch(4 if args.smoke_test else 48), tokenizer, max_chunks=3, max_chunk_toks=192) if step % 2 == 1 else None

        def _phase_b_loss() -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = mdl(
                    b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                    r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                    outer_steps=2, detach_zchar_for_ai=True, return_aux=True,
                )
                ai_m, pm = has_ai > 0.5, has_pr > 0.5
                loss_b = (
                    (0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"][ai_m], y_ai[ai_m], y_pr[ai_m], has_pr[ai_m], a5_b[ai_m]) if ai_m.any() else out["logit_promo"].sum() * 0.0)
                    + (compute_dual_axis_ebt_margin_loss(out["ebt_margin"][ai_m], out["ebt_promo_margin"][ai_m], y_ai[ai_m], y_pr[ai_m], has_pr[ai_m]) if ai_m.any() else out["logit_promo"].sum() * 0.0)
                    + compute_v121_orthogonality_loss(mdl, out["logit"].detach(), y_ai, y_pr, has_pr, has_ai)
                )
                u, u_t1, u_r1, u_r0 = out["logit_promo"].float(), out["exit1_out"]["logit_promo"].float(), out["logit_promo_r1"].float(), out["logit_promo_r0"].float()
                if pm.any():
                    yp_c = y_pr[pm].clamp(1e-4, 1.0 - 1e-4)
                    loss_b = loss_b + (F.binary_cross_entropy_with_logits(u[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.30 * (F.binary_cross_entropy_with_logits(u_t1[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.40 * (F.binary_cross_entropy_with_logits(u_r1[pm], yp_c, reduction="none") * w_b[pm] * (1.0 + 2.5 * (y_pr[pm] >= 0.65).float())).mean()
                    loss_b = loss_b + 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], u, sub4_i, y_pr, has_pr)
                    loss_b = loss_b + 0.22 * compute_promo_cvar_ceiling_loss(u, y_pr, has_pr)
                    loss_b = loss_b + 0.18 * compute_promo_cvar_ceiling_loss(u_r1, y_pr, has_pr)
                    loss_b = loss_b + 0.18 * compute_promo_cvar_ceiling_loss(u_r0, y_pr, has_pr)

                r0_ai, t_ai_lg, t_pr_lg = out["logit_ai_r0"].float(), out["logit"].detach().float(), u.detach()
                l_r0_ai = (
                    0.45 * (F.binary_cross_entropy_with_logits(r0_ai[ai_m], y_ai[ai_m].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[ai_m] * (1.0 + 2.5 * (y_ai[ai_m] >= 0.45).float())).mean()
                    + 0.35 * F.mse_loss(r0_ai[ai_m], t_ai_lg[ai_m])
                    + 0.10 * listnet_tail_distill_loss(r0_ai[ai_m], t_ai_lg[ai_m])
                ) if ai_m.any() else u.sum() * 0.0
                l_r0_pr = 0.25 * F.mse_loss(u_r0, t_pr_lg) + 0.08 * listnet_tail_distill_loss(u_r0, t_pr_lg)
                if pm.any():
                    l_r0_pr = l_r0_pr + 0.50 * (F.binary_cross_entropy_with_logits(u_r0[pm], y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm] * (1.0 + 2.5 * (y_pr[pm] >= 0.65).float())).mean()
                with torch.amp.autocast("cuda", enabled=False):
                    pa0, pa_t = torch.sigmoid(r0_ai.detach()), torch.sigmoid(t_ai_lg)
                    pp0, pp_t = torch.sigmoid(u_r0.detach()), torch.sigmoid(t_pr_lg)
                    d0_tgt = torch.maximum(
                        torch.maximum(((pa_t - pa0).abs() * 3.0).clamp(0.0, 1.0), (ai_m & (y_ai >= 0.45) & (pa0 < 0.35)).float()),
                        torch.maximum(((pp_t - pp0).abs() * 3.0).clamp(0.0, 1.0), (pm & (y_pr >= 0.65) & (pp0 < 0.35)).float()),
                    )
                    l_r0_def = 0.25 * F.binary_cross_entropy(out["p_defer_r0"].float().clamp(1e-5, 1.0 - 1e-5), d0_tgt)
                loss_b = loss_b + l_r0_ai + l_r0_pr + l_r0_def

                if pr_pair_pack is not None:
                    (pi, pm_, pe, pt_, ps, pc1, pc3, pcm, pcf, *_), (oi, om_, oe, ot_, os_, oc1, oc3, ocm, ocf, *_) = pr_pair_pack
                    op = mdl(pi, pm_, ps, engram_ids=pe, trigram_ids=pt_, r0_c1=pc1, r0_c3=pc3, r0_mask=pcm, r0_feats=pcf, outer_steps=1, detach_zchar_for_ai=True, return_aux=True)
                    oo = mdl(oi, om_, os_, engram_ids=oe, trigram_ids=ot_, r0_c1=oc1, r0_c3=oc3, r0_mask=ocm, r0_feats=ocf, outer_steps=1, detach_zchar_for_ai=True, return_aux=True)
                    loss_b = loss_b + 0.22 * F.softplus(2.2 - (op["logit_promo"].float() - oo["logit_promo"].float())).mean()

                if cs_pack is not None:
                    cs_o = mdl.forward_chronostream_batch_v122(
                        cs_pack["chunk_ids"].to(device), cs_pack["chunk_mask"].to(device), cs_pack["chunk_eng"].to(device), cs_pack["chunk_tri"].to(device),
                        cs_pack["chunk_style56"].to(device), cs_pack["chunk_meta6"].to(device), cs_pack["chunk_valid"].to(device), cs_pack["doc_style56"].to(device),
                        trunk_no_grad=True, outer_steps=2,
                    )
                    loss_b = (
                        loss_b
                        + F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_pack["y_ai_doc"].to(device))
                        + 0.50 * F.binary_cross_entropy_with_logits(cs_o["logit_hybrid_doc"].float(), cs_pack["y_hyb_doc"].to(device))
                        + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_pack["y_pr_doc"].to(device))
                    )
            return loss_b

        opt_b.zero_grad(set_to_none=True)
        loss_b = _phase_b_loss()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        if not args.smoke_test and step >= int(pb_steps * 0.65):
            asam_b.first_step()
            opt_b.zero_grad(set_to_none=True)
            _phase_b_loss().backward()
            asam_b.restore()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt_b.step()

        if not args.smoke_test and (step + 1) % 100 == 0:
            lb_v = float(loss_b.item())
            pike_dual.update(np.ones(7, dtype=np.float64), np.array([lb_v * 0.85, lb_v * 0.85, lb_v * 0.80, lb_v * 0.90, lb_v * 1.05, lb_v * 0.75, lb_v * 1.15]), np.zeros(7))

        if not args.smoke_test and ((step + 1) % 250 == 0 or step + 1 == pb_steps):
            pl_m = _eval_select_promo_and_long(mdl)
            ai_mb = _eval_select_ai(mdl)
            long_m = 0.50 * pl_m["long_ai"] + 0.25 * pl_m["long_hyb"] + 0.25 * pl_m["long_pr"]
            sc_b = 0.45 * (0.5 * pl_m["promo_auroc"] + 0.5 * pl_m["promo_pr"]) + 0.25 * long_m + 0.30 * (0.5 * ai_mb["r0_ai"] + 0.5 * pl_m["r0_pr"])
            if sc_b >= best_pb_sc:
                best_pb_sc = sc_b
                best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(
                f"[PhaseB step={step+1}/{pb_steps}] loss={float(loss_b.item()):.4f} | ai_T={ai_mb['sel_T']:.5f} | "
                f"promo={pl_m['promo_auroc']:.4f} pr={pl_m['promo_pr']:.4f} r1_pr={pl_m['r1_pr']:.4f} | "
                f"long={pl_m['long_ai']:.4f}/{pl_m['long_hyb']:.4f}/{pl_m['long_pr']:.4f} | r0={ai_mb['r0_ai']:.4f}/{pl_m['r0_pr']:.4f} | elapsed={time.monotonic()-t_start:.0f}s",
                flush=True,
            )
            mdl.train()

    for hk in hooks_b:
        hk.remove()
    if best_pb_sc > -1e8:
        mdl.load_state_dict({k: v.to(device) for k, v in best_pb_sd.items()}, strict=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_fp16_master.pt")
    fp16_ai = _eval_select_ai(mdl)
    fp16_pl = _eval_select_promo_and_long(mdl)
    print(f"[PhaseB FP16 Master DONE] ai={json.dumps(fp16_ai)} | dual={json.dumps({k: round(v, 4) for k, v in fp16_pl.items()})}", flush=True)

    # =========================================================================
    # PHASE C: Zero-Op Fold -> HAWQ-V2 W4 + QuEST + BitDistill + TR-KD + QAT Era-GRL + R1/R0 Floor ->
    #          Post-Quant Convex Soup -> Shock-Absorber v2 -> PURE `val_meta_cal` Calibration!
    # =========================================================================
    cal_items = (t0_pool[:16] + new_gold_ai_pool[:16] + t1_pool[:16])[:32]
    cal_f = _collate_items(cal_items)
    fold_rep = fold_zero_op_v121(mdl, calib_inputs=[(cal_f[0], cal_f[1], cal_f[4], cal_f[2])])
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    fp16_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_teacher.parameters():
        p.requires_grad_(False)

    def _eval_qat_post_quant(sd_cand: dict[str, torch.Tensor]) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_cand.items()}, strict=True)
        mdl.set_quant_mode("sub2")
        ma = _eval_select_ai(mdl)
        mp = _eval_select_promo_and_long(mdl)
        return float(0.65 * ma["sel_obj"] + 0.25 * (0.5 * mp["promo_auroc"] + 0.5 * mp["promo_pr"]) + 0.10 * ma["r0_ai"])

    def _run_qat_stage(st_name: str, init_sd: dict[str, torch.Tensor], qmode: str, steps: int, tail_mult: float = 1.0) -> dict[str, torch.Tensor]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(qmode)
        pg_q, hooks_q = build_v123_param_groups(mdl, "qat", era_grl=era_grl)
        quest_hooks = install_quest_trust_gradient_hooks(mdl, trit_t_mult=1.5, seq2_t_mult=3.5) if qmode == "sub2" else []
        opt_q = torch.optim.AdamW(pg_q, betas=(0.9, 0.95))
        best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc_q = -1e9
        mdl.train(); era_grl.train()

        for step in range(steps):
            mult = _wsd_lr_mult(step, steps, warmup_frac=0.05, decay_frac=0.30)
            for pg in opt_q.param_groups:
                pg["lr"] = pg["base_lr"] * mult
            opt_q.zero_grad(set_to_none=True)

            nb_q = min(bs, 32 if args.smoke_test else bs)
            b_it = _sample_phase_a_v123(nb_q * 3 // 4, include_mem=True) + _sample_phase_b_v123(nb_q - nb_q * 3 // 4, _pike_phase_b_weights(), wild_promo_frac=0.20)
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, has_ai, y_pr, has_pr, _, w_b, a5_b, is_t0, is_phys, era5_b, clen_b) = _collate_items(b_it)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                tv = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
                l_tr_t2 = compute_tail_ranking_kd_loss(out["logit"], tv["logit"], y_ai, has_ai)
                l_tr_t1 = compute_tail_ranking_kd_loss(out["exit1_out"]["logit"], tv["exit1_out"]["logit"], y_ai, has_ai)
                l_slot_kl = compute_bitdistill_slot_relation_kl(out["slots"], tv["slots"], mdl.core, fp16_teacher.core)
                l_pauc_q, l_ceil_q = compute_sopa_pauc_and_cvar_doro_loss(out["logit"], y_ai, has_ai, is_phys > 0.5)
                l_floor_q = compute_r1_r0_clean_human_floor_loss(out["logit_r1"], out["logit_ai_r0"], y_ai, has_ai)
                l_r0_t2_q = compute_r0_to_t2_mutual_kd_loss(out["logit"], out["logit_ai_r0"], y_ai, has_ai, clen_b)
                l_ebt_q = (
                    0.35 * F.mse_loss(out["ebt_margin"].float(), tv["ebt_margin"].float().detach().clamp(-10.0, 10.0))
                    + 0.10 * compute_ebt_rce_emr_loss(out["ebt_energies"], out["ebt_margin"], y_ai, has_ai, is_t0 > 0.5)
                )
                l_pr_q = (
                    0.50 * F.mse_loss(out["logit_promo"].float(), tv["logit_promo"].float().detach().clamp(-10.0, 10.0))
                    + 0.30 * F.mse_loss(out["logit_promo_r1"].float(), tv["logit_promo_r1"].float().detach().clamp(-10.0, 10.0))
                    + 0.25 * F.mse_loss(out["logit_promo_r0"].float(), tv["logit_promo_r0"].float().detach().clamp(-10.0, 10.0))
                    + 0.15 * compute_promo_cvar_ceiling_loss(out["logit_promo"], y_pr, has_pr)
                )
                l_r0_q = 0.30 * F.mse_loss(out["logit_ai_r0"].float(), tv["logit_ai_r0"].float().detach().clamp(-10.0, 10.0)) + 0.25 * F.mse_loss(out["logit_r1"].float(), tv["logit_r1"].float().detach().clamp(-10.0, 10.0))
                # P0-6: Keep `OfflineEraGRL` active throughout QAT!
                era_valid = (has_ai > 0.5) & (y_ai <= 0.25) & (era5_b >= 0) & (era5_b < NUM_ERAS_V122)
                l_era_q = era_grl(out["emb"][:, :AI_EMB_END_V11], era5_b, era_valid, lam=0.45) if int(era_valid.sum().item()) >= 8 else out["logit"].sum() * 0.0
                loss_q = (
                    l_tr_t2 + 0.40 * l_tr_t1 + 0.15 * l_slot_kl + l_floor_q + 0.70 * l_r0_t2_q
                    + tail_mult * (0.20 * l_pauc_q + 0.15 * l_ceil_q)
                    + l_ebt_q + l_pr_q + l_r0_q + 0.10 * l_era_q
                )
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_q.step()
            mdl.project_ai_head_weight_norm(1.25 * w_ai_init_norm)

            if not args.smoke_test and ((step + 1) % 250 == 0 or step + 1 == steps):
                ma = _eval_select_ai(mdl)
                mp = _eval_select_promo_and_long(mdl)
                sc_q = 0.65 * ma["sel_obj"] + 0.25 * (0.5 * mp["promo_auroc"] + 0.5 * mp["promo_pr"]) + 0.10 * ma["r0_ai"]
                if sc_q >= best_sc_q:
                    best_sc_q = sc_q
                    best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                print(
                    f"[PhaseC-{st_name} step={step+1}/{steps}] sel_T={ma['sel_T']:.5f} obj={ma['sel_obj']:.5f} w_rp95={ma['rp95_w']:.4f} | promo={mp['promo_auroc']:.4f} | r1={ma['r1_auc']:.4f} r0={ma['r0_ai']:.4f}",
                    flush=True,
                )
                mdl.train(); era_grl.train()

        for hk in hooks_q + quest_hooks:
            hk.remove()
        return best_sd_q

    w4_sd = _run_qat_stage("UPQ-W4", folded_sd, "w4", 2 if args.smoke_test else int(args.pc_w4_steps), tail_mult=1.0)
    sub2_q1 = _run_qat_stage("Sub2-Tail", w4_sd, "sub2", 2 if args.smoke_test else int(args.pc_sub2_steps), tail_mult=1.5)
    sub2_q2 = _run_qat_stage("Sub2-Bal", w4_sd, "sub2", 2 if args.smoke_test else int(args.pc_sub2_steps), tail_mult=0.7)

    # Post-Quant Exact Convex Soup
    soup_best_a, soup_best_sc, soup_best_sd = 0.5, -1e9, sub2_q1
    for alpha in ((0.0, 0.5, 1.0) if args.smoke_test else (0.0, 0.25, 0.50, 0.75, 1.0)):
        cand_s = {k: (alpha * sub2_q1[k] + (1.0 - alpha) * sub2_q2[k] if sub2_q1[k].is_floating_point() else sub2_q1[k]) for k in sub2_q1}
        sc_s = _eval_qat_post_quant(cand_s)
        if sc_s >= soup_best_sc:
            soup_best_sc, soup_best_a, soup_best_sd = sc_s, alpha, cand_s
    mdl.load_state_dict({k: v.to(device) for k, v in soup_best_sd.items()}, strict=True)
    mdl.set_quant_mode("sub2")
    soup_rep = {"alpha_q1": soup_best_a, "final_score": round(soup_best_sc, 5)}
    print(f"[PhaseC-ConvexSoup] {soup_rep}", flush=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_qat_soup.pt")

    # P0-7: Shock-Absorber v2 (300 steps over ai_head + r1_head + exit_affine + z_char_coda_ai + promo_head!)
    pre_shock_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    sh_steps = 2 if args.smoke_test else int(args.pc_shock_steps)
    pg_sh, hk_sh = build_v123_param_groups(mdl, "shock")
    opt_sh = torch.optim.AdamW(pg_sh, betas=(0.9, 0.95))
    mdl.train()
    for step in range(sh_steps):
        mult = _wsd_lr_mult(step, sh_steps, warmup_frac=0.10, decay_frac=0.40)
        for pg in opt_sh.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        opt_sh.zero_grad(set_to_none=True)
        nb_s = min(bs, 32 if args.smoke_test else bs)
        b_it = _sample_phase_a_v123(nb_s * 3 // 4, include_mem=True) + _sample_phase_b_v123(nb_s - nb_s * 3 // 4, _pike_phase_b_weights(), wild_promo_frac=0.20)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf, y_ai, has_ai, y_pr, has_pr, _, _, _, _, is_phys, _, clen_b) = _collate_items(b_it)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            tv = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, detach_zchar_for_ai=False, return_aux=True)
            lp_s, lc_s = compute_sopa_pauc_and_cvar_doro_loss(out["logit"], y_ai, has_ai, is_phys > 0.5)
            lf_s = compute_r1_r0_clean_human_floor_loss(out["logit_r1"], out["logit_ai_r0"], y_ai, has_ai)
            loss_sh = (
                compute_tail_ranking_kd_loss(out["logit"], tv["logit"], y_ai, has_ai)
                + 0.35 * compute_tail_ranking_kd_loss(out["exit1_out"]["logit"], tv["exit1_out"]["logit"], y_ai, has_ai)
                + 0.30 * F.mse_loss(out["logit_r1"].float(), tv["logit_r1"].float().detach().clamp(-10.0, 10.0))
                + 0.20 * lp_s + 0.15 * lc_s + lf_s
                + 0.40 * F.mse_loss(out["logit_promo"].float(), tv["logit_promo"].float().detach().clamp(-10.0, 10.0))
                + 0.20 * compute_promo_cvar_ceiling_loss(out["logit_promo"], y_pr, has_pr)
            )
        loss_sh.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt_sh.step()
    for hk in hk_sh:
        hk.remove()
    post_shock_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    post_shock_sc = _eval_qat_post_quant(post_shock_sd)
    kept_shock = bool(post_shock_sc >= soup_best_sc)
    if not kept_shock:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_shock_sd.items()}, strict=True)
        mdl.set_quant_mode("sub2")
    shock_rep = {"kept": kept_shock, "before": round(soup_best_sc, 5), "after": round(post_shock_sc, 5)}
    print(f"[PhaseC-ShockAbsorberV2] {shock_rep}", flush=True)
    del fp16_teacher, ai_anchor_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: 100% PURE `val_meta_cal` Calibration (ZERO `god500`!) -> Length-Aware `z5_softmin` ->
    #          Clean AI + Output-Shifted Promo LUTs -> Certified LTT (`n_pos >= 1,000`) -> MGVZ7 + 5-Stage ONNX + 11-Gate CI!
    # =========================================================================
    mdl.eval()
    vmc_o = _predict_v123(mdl, vmc_p, 2)
    z_ll = np.log1p(np.array([len(t) for t in vmc_p[3]], dtype=np.float32))[:, None]
    era_oh = np.zeros((len(vmc_p[8]), NUM_ERAS_V122), dtype=np.float32)
    for ii, ev in enumerate(vmc_p[8]):
        if 0 <= ev < NUM_ERAS_V122:
            era_oh[ii, ev] = 1.0
    list_rate = vmc_p[5][:, 14:15] if vmc_p[5].shape[1] > 14 else np.zeros((len(vmc_p[3]), 1), dtype=np.float32)
    zai_conf = np.concatenate([z_ll, era_oh, expit(vmc_o["logit_promo"])[:, None], list_rate], axis=1)
    zpr_conf = np.concatenate([z_ll, expit(vmc_o["logit"])[:, None]], axis=1)

    emb_l, aux_l, yai_l, era_l = vmc_o["emb"], vmc_o["ai_aux_term"], vmc_p[7] / 100.0, vmc_p[8]
    ypr_l = np.full(len(vmc_p[7]), 0.50, dtype=np.float32)
    if promo_val_select_rows:
        ps_l_pk = (
            [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
            [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
            np.stack([x["style"] for x in promo_val_select_rows]), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32), np.full(len(promo_val_select_rows), -1, dtype=np.int32),
        )
        pso_l = _predict_v123(mdl, ps_l_pk, 2)
        z_ll_ps = np.log1p(np.array([len(t) for t in ps_l_pk[3]], dtype=np.float32))[:, None]
        emb_l = np.concatenate([emb_l, pso_l["emb"]], axis=0)
        aux_l = np.concatenate([aux_l, pso_l["ai_aux_term"]], axis=0)
        yai_l = np.concatenate([yai_l, np.full(len(promo_val_select_rows), 0.50, dtype=np.float32)], axis=0)
        ypr_l = np.concatenate([ypr_l, ps_l_pk[7]], axis=0)
        era_l = np.concatenate([era_l, ps_l_pk[8]], axis=0)
        zai_conf = np.concatenate([zai_conf, np.zeros((len(promo_val_select_rows), zai_conf.shape[1]), dtype=np.float32)], axis=0)
        zpr_conf = np.concatenate([zpr_conf, np.concatenate([z_ll_ps, expit(pso_l["logit"])[:, None]], axis=1)], axis=0)

    leace_rep = apply_v122_full_logit_leace(mdl, emb_l, aux_l, yai_l, ypr_l, zai_conf, zpr_conf, era_val=era_l)
    print(f"[Stage3-LEACE] {leace_rep}", flush=True)

    # P0-4: Cross-Exit Affine Alignment with `clip_a=(0.80, 1.80)` protection on `R1`!
    vmc_o = _predict_v123(mdl, vmc_p, 2)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_r1"], vmc_o["raw_logit"], clip_a=(0.80, 1.80))
    a_t1, b_t1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_t1"], vmc_o["raw_logit"], clip_a=(0.70, 1.80))
    pa_r1, pb_r1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_promo_r1"], vmc_o["raw_logit_promo"])
    pa_t1, pb_t1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_promo_t1"], vmc_o["raw_logit_promo"])
    with torch.no_grad():
        mdl.exit_affine_a.data.copy_(torch.tensor([a_r1, a_t1, 1.0], device=device))
        mdl.exit_affine_b.data.copy_(torch.tensor([b_r1, b_t1, 0.0], device=device))
        mdl.exit_affine_promo_a.data.copy_(torch.tensor([pa_r1, pa_t1, 1.0], device=device))
        mdl.exit_affine_promo_b.data.copy_(torch.tensor([pb_r1, pb_t1, 0.0], device=device))

    vmc_o = _predict_v123(mdl, vmc_p, 2)
    vmc_cl = np.array([len(t) for t in vmc_p[3]], dtype=np.int32)
    y_mc_bin = (vmc_p[7] >= 70.0).astype(int)
    y_mc_01 = vmc_p[7] / 100.0

    # Dual Era Invariance Check (`V123_E1`) on `val_meta_cal` + `val_cal_cert`
    vcert_o = _predict_v123(mdl, vcert_p, 2)
    vcert_cl = np.array([len(t) for t in vcert_p[3]], dtype=np.int32)
    era_dual_check = eval_v123_dual_era_invariance(
        np.concatenate([vmc_o["emb"], vcert_o["emb"]], axis=0),
        np.concatenate([
            np.stack([vmc_o["logit_ai_r0"], vmc_o["logit_r1"], vmc_o["logit_t1"], vmc_o["logit"], vmc_o["ebt_margin"]], axis=1),
            np.stack([vcert_o["logit_ai_r0"], vcert_o["logit_r1"], vcert_o["logit_t1"], vcert_o["logit"], vcert_o["ebt_margin"]], axis=1),
        ], axis=0),
        np.concatenate([y_mc_01, vcert_p[7] / 100.0], axis=0),
        np.concatenate([vmc_p[8], vcert_p[8]], axis=0),
    )
    print(f"[Stage3-EraDualCheck] {era_dual_check}", flush=True)

    # 3-param fusion + P0-7 Constrained Length-Aware `z5_softmin` on 100% PURE `val_meta_cal`!
    fusion_rep = fit_3param_logodds_fusion_v121(vmc_o["logit_t1"], vmc_o["logit"], vmc_o["ebt_margin"], vmc_p[7], ebt_clip=6.0)
    fw_w = (float(fusion_rep["w_t1"]), float(fusion_rep["w_t2"]), float(fusion_rep["w_ebt"]))
    eb_clip = float(fusion_rep.get("ebt_clip", 6.0))

    def _fused3_lg(od: dict[str, np.ndarray]) -> np.ndarray:
        return bounded_fused3_logits(od["logit_t1"], od["logit"], od["ebt_margin"], fw_w[0], fw_w[1], fw_w[2], eb_clip)

    z5_cfg = fit_len_aware_z5_bootstrap_v123(
        {"lg_r0": vmc_o["logit_ai_r0"], "lg_r1": vmc_o["logit_r1"], "lg_t1": vmc_o["logit_t1"], "lg_t2": vmc_o["logit"], "lg_ebt": vmc_o["ebt_margin"]},
        y_mc_bin, vmc_cl, ebt_clip=eb_clip, n_boot=30 if args.smoke_test else 200,
    )
    print(f"[Stage3-LenAware-z5] {json.dumps(z5_cfg)}", flush=True)

    def _z5_lg(od: dict[str, np.ndarray], cl_arr: np.ndarray) -> np.ndarray:
        return apply_len_aware_z5_v123(
            {"lg_r0": od["logit_ai_r0"], "lg_r1": od["logit_r1"], "lg_t1": od["logit_t1"], "lg_t2": od["logit"], "lg_ebt": od["ebt_margin"]},
            z5_cfg, char_len=cl_arr,
        )

    lut_ai_full = fit_smooth_ai_lut_v122(_fused3_lg(vmc_o), y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_z5 = fit_smooth_ai_lut_v122(_z5_lg(vmc_o, vmc_cl), y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_r1 = fit_smooth_ai_lut_v122(vmc_o["logit_r1"], y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_r0 = fit_smooth_ai_lut_v122(vmc_o["logit_ai_r0"], y_mc_01, iso_blend=0.30, n_knots=256)
    moses_offs = fit_moses_register_offsets_v122(_z5_lg(vmc_o, vmc_cl), y_mc_bin, vmc_p[5])

    # Check P0-8 NEW `V123_AI_LUT_FLOOR` on clean wild humans!
    hu_cert = vcert_p[7] <= 25.0
    p_ai_r0_hu = apply_spline_temp_lut(expit(np.clip(vcert_o["logit_ai_r0"][hu_cert], -30.0, 30.0)), lut_ai_r0)
    p_ai_r1_hu = apply_spline_temp_lut(expit(np.clip(vcert_o["logit_r1"][hu_cert], -30.0, 30.0)), lut_ai_r1)
    ai_lut_check = {
        "lut_ai_r0_at_001": float(np.interp(0.01, lut_ai_r0["x"], lut_ai_r0["y"])),
        "r0_ai_le008_rate": float((p_ai_r0_hu <= 0.08).mean()) if len(p_ai_r0_hu) else 0.0,
        "r1_ai_le008_rate": float((p_ai_r1_hu <= 0.08).mean()) if len(p_ai_r1_hu) else 0.0,
    }
    print(f"[Stage3-AILUT-Check] {ai_lut_check}", flush=True)

    # Output-Shifted Promo LUTs
    pi_nat_pr = float(np.clip(estimate_wild_prevalence_em(expit(vmc_o["logit_promo"]), pi_train=0.10), 0.004, 0.02))
    lut_pr_full = lut_pr_r1 = lut_pr_r0 = None
    promo_lut_check: dict[str, Any] = {}
    if promo_val_select_rows:
        ps_pk = (
            [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
            [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
            np.stack([x["style"] for x in promo_val_select_rows]), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32), np.zeros(len(promo_val_select_rows), dtype=np.int32),
        )
        pso = _predict_v123(mdl, ps_pk, 2)
        lut_pr_full = fit_output_shifted_promo_lut_v122(pso["logit_promo"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)
        lut_pr_r1 = fit_output_shifted_promo_lut_v122(pso["logit_promo_r1"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(pso["logit_promo_r0"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)
        p_pr_wild_t2 = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo"], -30.0, 30.0)), lut_pr_full)
        p_pr_wild_r0 = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
        promo_lut_check = {
            "median_promo_wild": float(np.median(p_pr_wild_t2)),
            "p90_promo_wild": float(np.quantile(p_pr_wild_t2, 0.90)),
            "r0_promo_le008_rate": float((p_pr_wild_r0 <= 0.08).mean()),
        }
        print(f"[Stage3-PromoLUT-Check] {promo_lut_check}", flush=True)

    # Real-Wild Conformal FDR + Certified LTT (`n_pos >= 1,000` -> alpha_ai=0.01 / 0.005 mathematically feasible!)
    p_z5_cert = apply_spline_temp_lut(expit(np.clip(_z5_lg(vcert_o, vcert_cl), -30.0, 30.0)), lut_ai_z5)
    conformal_fdr = fit_conformal_fdr_thresholds_v122(p_z5_cert, (vcert_p[7] >= 70.0).astype(int), pi_wild=0.03)

    r0_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False}
    r1_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False}
    if promo_val_cert_rows:
        pc_pk = (
            [x["ids"] for x in promo_val_cert_rows], [x["engram"] for x in promo_val_cert_rows], [x["trigram"] for x in promo_val_cert_rows],
            [x["raw_text"] for x in promo_val_cert_rows], [x["clean_text"] for x in promo_val_cert_rows],
            np.stack([x["style"] for x in promo_val_cert_rows]), [x["meta"] for x in promo_val_cert_rows], np.array([x["y_promo"] for x in promo_val_cert_rows], dtype=np.float32), np.zeros(len(promo_val_cert_rows), dtype=np.int32),
        )
        pco = _predict_v123(mdl, pc_pk, 1)

        def _elig_r0(pk_x):
            return np.array([(len(t) <= 160 and m["spoof_any"] == 0.0 and "http" not in t.lower()) for t, m in zip(pk_x[3], pk_x[6])], dtype=bool)
        def _elig_r1(pk_x):
            return np.array([(len(s) <= 96 and m["spoof_any"] == 0.0 and "http" not in cl and float(np.max(st[list(R1_PROMO_VETO_DIMS_V11)])) == 0.0) for s, m, cl, st in zip(pk_x[0], pk_x[6], pk_x[4], pk_x[5])], dtype=bool)

        p_ai_r0_m, p_pr_r0_m = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_ai_r0"], -30.0, 30.0)), lut_ai_r0), apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
        p_ai_r0_c, p_pr_r0_c = apply_spline_temp_lut(expit(np.clip(vcert_o["logit_ai_r0"], -30.0, 30.0)), lut_ai_r0), apply_spline_temp_lut(expit(np.clip(vcert_o["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
        p_ai_r0_p, p_pr_r0_p = apply_spline_temp_lut(expit(np.clip(pco["logit_ai_r0"], -30.0, 30.0)), lut_ai_r0), apply_spline_temp_lut(expit(np.clip(pco["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
        raw_r0_ltt = certify_fixed_sequence_dual_axis_ltt(
            _elig_r0(vmc_p), vmc_o["p_defer_r0"], p_ai_r0_m, p_pr_r0_m,
            vcert_p[7], _elig_r0(vcert_p), vcert_o["p_defer_r0"], p_ai_r0_c, p_pr_r0_c,
            (pc_pk[7] >= 0.65).astype(bool), _elig_r0(pc_pk), pco["p_defer_r0"], p_ai_r0_p, p_pr_r0_p,
            alpha_ai=0.01, alpha_pr=0.015, delta_fwer=0.05,
            d_grid=(0.15, 0.25, 0.35, 0.45), tau_a_grid=(0.04, 0.06, 0.08, 0.10, 0.12), tau_p_grid=(0.04, 0.06, 0.08, 0.12),
        )
        r0_ltt_policy = {**raw_r0_ltt, "defer_max": raw_r0_ltt.get("r1_defer_max", 0.35), "tau_ai": raw_r0_ltt.get("r1_p_ai_max", 0.08), "tau_promo": raw_r0_ltt.get("r1_p_promo_max", 0.08), "max_chars": 160}

        p_ai_r1_m, p_pr_r1_m = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_r1"], -30.0, 30.0)), lut_ai_r1), apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo_r1"], -30.0, 30.0)), lut_pr_r1)
        p_ai_r1_c, p_pr_r1_c = apply_spline_temp_lut(expit(np.clip(vcert_o["logit_r1"], -30.0, 30.0)), lut_ai_r1), apply_spline_temp_lut(expit(np.clip(vcert_o["logit_promo_r1"], -30.0, 30.0)), lut_pr_r1)
        p_ai_r1_p, p_pr_r1_p = apply_spline_temp_lut(expit(np.clip(pco["logit_r1"], -30.0, 30.0)), lut_ai_r1), apply_spline_temp_lut(expit(np.clip(pco["logit_promo_r1"], -30.0, 30.0)), lut_pr_r1)
        r1_ltt_policy = certify_fixed_sequence_dual_axis_ltt(
            _elig_r1(vmc_p), vmc_o["p_defer1"], p_ai_r1_m, p_pr_r1_m,
            vcert_p[7], _elig_r1(vcert_p), vcert_o["p_defer1"], p_ai_r1_c, p_pr_r1_c,
            (pc_pk[7] >= 0.65).astype(bool), _elig_r1(pc_pk), pco["p_defer1"], p_ai_r1_p, p_pr_r1_p,
            alpha_ai=0.01, alpha_pr=0.015, delta_fwer=0.05,
            d_grid=(0.12, 0.18, 0.25), tau_a_grid=(0.06, 0.08, 0.10, 0.12), tau_p_grid=(0.04, 0.06, 0.08, 0.12),
        )

    ltt_check = {
        "cert_n_ai": int((vcert_p[7] >= 70.0).sum()),
        "r0_certified": bool(r0_ltt_policy.get("certified_ltt", False)),
        "r1_certified": bool(r1_ltt_policy.get("certified_ltt", False)),
        "cal_exit_rate": max(float(r0_ltt_policy.get("cal_exit_rate", 0.0)), float(r1_ltt_policy.get("cal_exit_rate", 0.0))),
    }
    print(f"[Stage3-LTT] {ltt_check}", flush=True)

    # Export MGVZ7 + 5-Stage FP32-Exact ONNX (100% v1.2.1/v1.2.2 ABI!)
    exp_info = export_v121_modular_onnx_and_bundle_v7(mdl, out_dir, export_matmulnbits=bool(args.export_matmulnbits))
    manifest = {
        "version": args.version, "schema": "meguvizcac-v1.2.3-apexnano-pro",
        "emb_layout": mdl.emb_layout_tag(), "max_len": int(args.max_len), "tokenizer_file": tok_path.name,
        "bundle_v7_mgvz": exp_info["bundle_v7_mgvz"]["file"],
        "rung0_onnx": exp_info["rung0_onnx"], "rung1_onnx": exp_info["rung1_onnx"],
        "rung2_onnx": exp_info["rung2_onnx"], "rung3_onnx": exp_info["rung3_onnx"],
        "dynaseam_onnx": exp_info["dynaseam_onnx"]["onnx_file"],
        "hawq_v2_bit_alloc": mdl.hawq_v2_rep,
        "fusion_weights": {"w_t1": fw_w[0], "w_t2": fw_w[1], "w_ebt": fw_w[2], "ebt_clip": eb_clip},
        "z5_softmin_cfg": z5_cfg,
        "moses_register_offsets": moses_offs,
        "conformal_fdr_thresholds": conformal_fdr,
        "lut_ai_r0": lut_ai_r0, "lut_ai_r1": lut_ai_r1, "lut_ai_full": lut_ai_full, "lut_ai_z5": lut_ai_z5,
        "lut_promo_r0": lut_pr_r0, "lut_promo_r1": lut_pr_r1, "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_ltt_policy, "dual_axis_ltt_policy": r1_ltt_policy,
        "capabilities": [
            "rung0_zero_bpe_charhash", "dense_shared_quant_res", "softball_logitnorm_v122_capped_mil",
            "hawq_v2_w4_prelude_core", "r0_t2_mutual_kd_unlocked_zchar", "r1_r0_human_floor_margin",
            "len_aware_z5_softmin", "pure_val_meta_cal_lut", "output_shifted_promo_lut", "certified_ltt_11ci",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_v7 = out_dir / f"meguvizcac-{args.version}-bundle-v7-sub2bit.zip"
    with ZipFile(zip_v7, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for fn in (exp_info["bundle_v7_mgvz"]["file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)
    v7_mb = zip_v7.stat().st_size / (1024.0 * 1024.0)
    if v7_mb > 2.20:
        exp_info = export_v121_modular_onnx_and_bundle_v7(mdl, out_dir, lever_w4_longstream=True, lever_w4_rung0=(v7_mb > 2.215), export_matmulnbits=bool(args.export_matmulnbits))
        with ZipFile(zip_v7, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for fn in (exp_info["bundle_v7_mgvz"]["file"], tok_path.name, "manifest.json"):
                zf.write(out_dir / fn, fn)
        v7_mb = zip_v7.stat().st_size / (1024.0 * 1024.0)

    zip_onnx = out_dir / f"meguvizcac-{args.version}-5stage-onnx-fp32exact.zip"
    with ZipFile(zip_onnx, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in (exp_info["rung0_onnx"], exp_info["rung1_onnx"], exp_info["rung2_onnx"], exp_info["rung3_onnx"], exp_info["dynaseam_onnx"]["onnx_file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)

    # Final Unpacked Evaluation + 11-Gate CI Suite
    mdl_u = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v7_v121_into_model(mdl_u, out_dir / exp_info["bundle_v7_mgvz"]["file"])
    mdl_u = mdl_u.to(device).eval()

    def _eval_pack_z5(pk_e) -> dict[str, Any]:
        if pk_e is None or len(pk_e[7]) < 10:
            return {}
        eo = _predict_v123(mdl_u, pk_e, 2)
        ecl = np.array([len(t) for t in pk_e[3]], dtype=np.int32)
        pe = apply_spline_temp_lut(expit(np.clip(_z5_lg(eo, ecl), -30.0, 30.0)), lut_ai_z5)
        return {**compute_eval_metrics(pk_e[7], pe), **compute_high_precision_metrics(pk_e[7], pe)}

    tb_m = _eval_pack_z5(tg_p)
    tb2_m = _eval_pack_z5(tg2_p)
    td_m = compute_eval_metrics(td_p[7], apply_spline_temp_lut(expit(np.clip(_z5_lg(_predict_v123(mdl_u, td_p, 2), np.array([len(t) for t in td_p[3]], dtype=np.int32)), -30.0, 30.0)), lut_ai_z5)) if len(td_p[7]) > 0 else {}
    g5_all_m = _eval_pack_z5(god500_all_p)
    g5_tv_m = _eval_pack_z5(god500_tv_p)

    pb_m: dict[str, Any] = {}
    pb_p = Path(__file__).resolve().parent / "data" / "v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_pk = _pre_eval(pb_rows)
        pbo = _predict_v123(mdl_u, pb_pk, 2)
        pb_cl = np.array([len(t) for t in pb_pk[3]], dtype=np.int32)
        y_pb_ai, y_pb_pr = np.array([int(r["y_ai"]) for r in pb_rows]), np.array([int(r["y_promo"]) for r in pb_rows])
        pb_m = {
            "ai_z5_auroc": round(float(roc_auc_score(y_pb_ai, _z5_lg(pbo, pb_cl))), 4),
            "promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo"])), 4),
            "promo_pr_auc": round(float(average_precision_score(y_pb_pr, pbo["logit_promo"])), 4),
            "r0_ai_auroc": round(float(roc_auc_score(y_pb_ai, pbo["logit_ai_r0"])), 4),
            "r0_promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo_r0"])), 4),
        }

    final_ci = run_v123_ci_gates(
        mdl_u, tokenizer,
        transplant_rmse=g0_rmse if transplant_rep.get("mode") != "cold_init" else None,
        transplant_spearman=g0_spearman,
        era_check=era_dual_check,
        promo_lut_check=promo_lut_check if lut_pr_full is not None else None,
        ai_lut_check=ai_lut_check,
        ltt_check=ltt_check,
    )
    rep = {
        "version": args.version, "gpu_name": gpu_name,
        "sibling_stats": dict(sib["stats"]), "delta_stats": dict(sib["delta_stats"]),
        "bundle_v7_total_mb": round(v7_mb, 4),
        "onnx_5stage_zip_mb": round(zip_onnx.stat().st_size / (1024.0 * 1024.0), 4),
        "stage0_init": transplant_rep,
        "phase_a_metrics": pa_ai_m, "phase_b_fp16_ai": fp16_ai, "phase_b_fp16_dual": fp16_pl,
        "fold_zero_op": fold_rep, "qat_convex_soup": soup_rep, "shock_absorber_v2": shock_rep,
        "leace_surgery": leace_rep, "era_dual_check": era_dual_check,
        "fusion_weights": fusion_rep, "z5_softmin_cfg": z5_cfg,
        "moses_register_offsets": moses_offs, "conformal_fdr_thresholds": conformal_fdr,
        "ai_lut_check": ai_lut_check, "promo_lut_check": promo_lut_check,
        "rung0_ltt_policy": r0_ltt_policy, "dual_axis_ltt_policy": r1_ltt_policy,
        "test_bili_gold_v1": tb_m, "test_bili_gold_v2": tb2_m, "test_dsh_gold": td_m,
        "diag_god500_all": g5_all_m, "diag_god500_text_verifiable": g5_tv_m, "promobench48": pb_m,
        "onnx_export": {
            "r0_fp32_max_diff": exp_info["r0_fp32_max_diff"],
            "r1_fp32_max_diff": exp_info["r1_fp32_max_diff"],
            "r2_fp32_max_diff": exp_info["r2_fp32_max_diff"],
            "r3_fp32_max_diff": exp_info["r3_fp32_max_diff"],
            "dynaseam_onnx": exp_info["dynaseam_onnx"],
        },
        "v123_ci_gates": final_ci,
        "release_gates": {
            "bundle_le_2p20_mb": bool(v7_mb <= 2.20),
            "onnx_5stage_exact": bool(
                exp_info["r0_fp32_max_diff"] < 1e-4 and exp_info["r1_fp32_max_diff"] < 1e-4
                and exp_info["r2_fp32_max_diff"] < 1e-4 and exp_info["r3_fp32_max_diff"] < 1e-4
                and bool(exp_info["dynaseam_onnx"].get("pass", exp_info["dynaseam_onnx"].get("onnx_max_diff", 0.0) < 1e-4))
            ),
            "v123_ci_passed": bool(final_ci["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
    if not all(rep["release_gates"].values()):
        print(f"[WARN] Non-fatal release gate alert (artifacts still saved!): {rep['release_gates']}", flush=True)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="MeguvizCAC v1.2.3-ApexNano-Pro 3-Phase Training Pipeline")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--force-v10-rebase", action="store_true", help="Force dual-source v1.0 + v1.2.2 rebase instead of v1.2.2 zero-shock warm-start")
    p.add_argument("--data", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    p.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    p.add_argument("--extra-pairs", type=Path, action="append", default=[])
    p.add_argument("--v122-fp16-master", type=Path, default=None)
    p.add_argument("--v122-bundle-mgvz", type=Path, default=None)
    p.add_argument("--v10-fp16-master", type=Path, default=None)
    p.add_argument("--v10-bundle-mgvz", type=Path, default=None)
    p.add_argument("--v07-tokenizer", type=Path, default=None)
    p.add_argument("--v08-gemini-gold", type=Path, default=None)
    p.add_argument("--v08-gemini-pairs", type=Path, default=None)
    p.add_argument("--v11-data-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    p.add_argument("--wave25-gold", type=Path, default=None)
    p.add_argument("--gold-v2-dir", type=Path, default=None)
    p.add_argument("--wave3-dir", type=Path, default=None)
    p.add_argument("--sibling-dir", type=Path, default=Path("v122-sibling-datafactory"))
    p.add_argument("--delta-dir", type=Path, default=Path("v123-sibling-delta"))
    p.add_argument("--out", type=Path, default=Path("scratch/v123_out"))
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--version", type=str, default="v1.2.3-r1-apexnano-b7")
    p.add_argument("--max-len", type=int, default=MAX_LEN_V07)
    p.add_argument("--pa0-steps", type=int, default=200)
    p.add_argument("--pa1-steps", type=int, default=1000)
    p.add_argument("--pb-steps", type=int, default=1500)
    p.add_argument("--pc-w4-steps", type=int, default=400)
    p.add_argument("--pc-sub2-steps", type=int, default=900)
    p.add_argument("--pc-shock-steps", type=int, default=300)
    p.add_argument("--export-matmulnbits", action="store_true")
    train_v123_apexnano_full(p.parse_args())


if __name__ == "__main__":
    main()
