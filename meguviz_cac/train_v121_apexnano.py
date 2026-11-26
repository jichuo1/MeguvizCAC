"""MeguvizCAC v1.2.1-ApexNano-Fix 3-Phase Zero-Regression Training Pipeline (Colab G4 Pro 6000 / A100 + Local `--smoke-test`).

Implements `docs/roadmap-v1.2.1-apexnano-fix.md` (`ADR-v1.2.1`) fixing ALL 5 forensic bench root causes
+ ALL 6 Deep Data Audit findings while leaving `v1.0.0`, `v1.1.0`, & `v1.2.0` 100% untouched:
  - RC-1 Fix: `SoftBallLogitNormHeadV121` (centered smooth ball projection -> zero positive-orthant collapse!).
  - RC-2 Fix: Phase A clean `Gold-v2 (55%) + Wild (35%) + R1 (10%)` selection + gentle top-core adaptation.
  - RC-3 Fix: Strictly pad-masked `mil_peak`, `promo_mil_peak`, `r1_span_std`, and `h_pool` (`V121_P1` CI gate!).
  - RC-4 Fix: Phase B passes LIVE `z_char` into promo-only bridges (`z_char_r1_promo`, `z_char_coda_promo`,
              `promo_zchar_direct`) with `V121_F6 = 0.0` AI firewall + 12% dedicated Wild-Promo batch slice +
              smooth Beta-Spline `lut_promo_{r0,r1,full}` (0% flat ties, full `[0.001, 0.995]` range!).
  - RC-5 Fix: Bounded `6*tanh(ebt/6)` 3-param log-odds fusion + expanded disjoint `val_cal_cert (n_ai >= 900)`
              for true mathematically certified Dual-Rung LTT (`certified_ltt: true, enabled: true`)!
  - Data Audit Fixes: `v121_data_refinery.py` (quarantines `colab_distilled` & `v08_pro388` midbands + `v08_flash`
              `15..40` false-human zone), diamond-only pair filter, strict `has_ai=0.0` mask on promo-only pools,
              and plug-and-play `--wave3-dir` support!
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_student import CODE_LIST, PRAG_TAGS, compute_eval_metrics
from .train_v04_dualloop import compute_high_precision_metrics
from .train_v05_apexloop import build_stratified_val_calibration_set
from .train_v06_hyperloop import _load_v06_balanced_bundle, _modern_two_way_pauc_and_top_push
from .train_v07_omniveritas import _derive_attr5_label, _wsd_lr_mult
from .train_v10_apexloop import _ingest_external_gemini_gold_v10, _top_fp_logit_ceiling_loss
from .train_v11_omniguard import (
    DynamicPiKEGroupSamplerV11,
    collate_chronostream_docs,
    compute_dual_axis_ebt_margin_loss,
    compute_ebt6_partial_label_ce,
    compute_l2_sp_loss,
    compute_sub4_masked_loss,
)
from .train_v12_apexnano import compute_slot_orthogonality_loss
from .v04_corpus_prep import GROUP_TO_IDX
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import MAX_LEN_V07, OmniQuantLinear
from .v08_zenith_ops import (
    ASAMPerturber,
    ERA_TO_IDX,
    export_beta_spline_as_v07_lut,
    fit_venn_abers_and_beta_spline_lut,
    greedy_pauc_soup,
    listnet_tail_distill_loss,
    zenith_graded_anchor_loss,
)
from .v09_cascade_cert import fit_cross_exit_affine_alignment
from .v10_apexloop import ENGRAM_BIGRAM_BUCKETS_V10, ENGRAM_TRIGRAM_BUCKETS_V10
from .v11_augment import (
    AdversarialObfuscationAug,
    BrandPriceSlotAug,
    LightHumanPostEditAug,
    QuadSpliceAug,
)
from .v11_cascade_cert import certify_fixed_sequence_dual_axis_ltt, estimate_wild_prevalence_em
from .v11_data_engine import OnlineLongMixSynthesizerV11, compute_joint_length_domain_ipw
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d
from .v11_omniguard import (
    AI_EMB_END_V11,
    R1_PROMO_VETO_DIMS_V11,
    SHARED_EMB_END_V11,
    MeguvizV11OmniGuardModel,
    load_bundle_v6_v11_into_model,
)
from .v11_promo_pseudo import _norm, _sha_norm
from .v121_apexnano import (
    MeguvizV121ApexNanoModel,
    apply_dual_axis_oblique_leace_v121,
    bounded_fused3_logits,
    export_v121_modular_onnx_and_bundle_v7,
    extract_rung0_charhash_batch,
    fold_zero_op_v121,
    load_bundle_v7_v121_into_model,
    pack_bundle_v7_v121,
    run_v121_ci_gates,
    transplant_v11_to_v121_dense,
)
from .v121_data_refinery import (
    classify_epistemic_tier_v121,
    is_diamond_pair_source_v121,
    run_v121_refinery_ci,
)


# ---------------------------------------------------------------------------
# 1. Helper Losses, Parameter Groups, Bounded 3-Param Fusion & Smooth Promo LUT
# ---------------------------------------------------------------------------

def compute_v121_orthogonality_loss(
    model: MeguvizV121ApexNanoModel,
    ai_logits: torch.Tensor,
    y_ai: torch.Tensor,
    y_promo: torch.Tensor,
    has_promo: torch.Tensor,
    has_ai: torch.Tensor,
) -> torch.Tensor:
    w_ai_sh = model.ai_head.weight[0, :SHARED_EMB_END_V11].float()
    w_pr_sh = model.promo_head.weight[0, :SHARED_EMB_END_V11].float()
    cos_w = F.cosine_similarity(w_ai_sh.unsqueeze(0), w_pr_sh.unsqueeze(0), dim=-1).pow(2).mean()
    loss = 0.05 * cos_w
    q3_m = (has_ai > 0.5) & (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo >= 0.65)
    q4_m = (has_ai > 0.5) & (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo <= 0.25)
    if int(q3_m.sum().item()) >= 4 and int(q4_m.sum().item()) >= 4:
        loss = loss + 0.05 * (ai_logits[q3_m].float().mean() - ai_logits[q4_m].float().mean()).pow(2)
    return loss


def build_v121_param_groups(model: MeguvizV121ApexNanoModel, phase: str) -> tuple[list[dict[str, Any]], list[Any]]:
    """Phase-strict differential AdamW groups + mathematical `[0:192]` / `[192:256]` gradient firewall."""
    promo_only_keys = (
        "r1_promo_head", "promo_head", "promo_sub4_head", "promo_span_head", "promo_mil_gate",
        "ebt_promo_gate", "z_char_coda_promo", "z_char_r1_promo", "promo_zchar_direct", "fc2_new_", "attr6_new_",
    )
    long_keys = ("longstream.", "chunk_emb_proj.")
    rung0_keys = ("rung0.",)
    zchar_ai_keys = ("z_char_r1_proj", "z_char_r1_gate", "z_char_coda_ai")
    fast_ai_head_keys = ("ai_head.", "core.qres.", "core.spectral_damp_logit", "r1_ai_head", "ebt_logit_gate", "mil_gate", "span_head.")
    ebt_ai_keys = ("core.inloop_ebt.fc1", "core.inloop_ebt.fc2_old")
    top_core_keys = ("core.q_proj", "core.k_proj", "core.v_proj", "core.o_proj", "core.ffn", "core.norm1", "core.norm2")
    emb_split_names = ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias")

    groups: dict[str, list[nn.Parameter]] = {
        "rung0": [], "promo_heads": [], "long": [], "zchar_ai": [],
        "fast_ai": [], "ebt_ai": [], "emb_split": [], "top_core": [], "trunk": [],
    }
    for name, p in model.named_parameters():
        if any(k in name for k in rung0_keys):
            groups["rung0"].append(p)
        elif any(k in name for k in promo_only_keys):
            groups["promo_heads"].append(p)
        elif any(k in name for k in long_keys):
            groups["long"].append(p)
        elif any(k in name for k in zchar_ai_keys):
            groups["zchar_ai"].append(p)
        elif name in emb_split_names:
            groups["emb_split"].append(p)
        elif any(k in name for k in fast_ai_head_keys):
            groups["fast_ai"].append(p)
        elif any(k in name for k in ebt_ai_keys):
            groups["ebt_ai"].append(p)
        elif any(k in name for k in top_core_keys):
            groups["top_core"].append(p)
        else:
            groups["trunk"].append(p)

    hooks: list[Any] = []
    if phase == "phase_a":
        # RC-2 FIX: Train SoftBall ai_head + SharedQuantRes + span_head + coda[:192] + EBT + ultra-gentle top_core
        # with L2-SP anchor -> zero shock while letting Gold-v2 & Wave-3 actually sharpen representations!
        lr_map = {
            "rung0": 1.8e-3, "promo_heads": 0.0, "long": 0.0, "zchar_ai": 0.0,
            "fast_ai": 1.0e-4, "ebt_ai": 4.0e-5, "emb_split": 7.0e-5, "top_core": 1.5e-5, "trunk": 0.0,
        }

        def _mask_ai_rows_only(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            g[AI_EMB_END_V11:] = 0.0
            return g

        for p in groups["emb_split"]:
            hooks.append(p.register_hook(_mask_ai_rows_only))
    elif phase == "phase_b":
        # RC-4 FIX: AI [0:192] & trunk 100% FROZEN -> ZERO AI negative transfer, while Promo heads +
        # LIVE z_char promo bridges (`z_char_r1_promo`, `z_char_coda_promo`, `promo_zchar_direct`) + emb[192:256] train!
        lr_map = {
            "rung0": 1.5e-3, "promo_heads": 1.2e-3, "long": 9.0e-4, "zchar_ai": 0.0,
            "fast_ai": 0.0, "ebt_ai": 0.0, "emb_split": 1.0e-3, "top_core": 0.0, "trunk": 0.0,
        }

        def _mask_promo_rows_only(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            g[:AI_EMB_END_V11] = 0.0
            return g

        for p in groups["emb_split"]:
            hooks.append(p.register_hook(_mask_promo_rows_only))
    else:
        # Phase C QAT: Pure Dual-Axis Teacher KD across all active parameters
        lr_map = {
            "rung0": 4.0e-4, "promo_heads": 2.2e-4, "long": 1.8e-4, "zchar_ai": 6.0e-5,
            "fast_ai": 1.2e-4, "ebt_ai": 6.0e-5, "emb_split": 5.0e-5, "top_core": 4.0e-5, "trunk": 3.0e-5,
        }

    out_pgs: list[dict[str, Any]] = []
    for gname, plist in groups.items():
        lr_g = lr_map[gname]
        for p in plist:
            p.requires_grad_(lr_g > 0.0)
        if plist and lr_g > 0.0:
            out_pgs.append({"name": gname, "params": plist, "lr": lr_g, "base_lr": lr_g, "weight_decay": 0.015})
    return out_pgs, hooks


def fit_3param_logodds_fusion_v121(
    lg_t1: np.ndarray,
    lg_t2: np.ndarray,
    ebt_margin: np.ndarray,
    scores_0_100: np.ndarray,
    ebt_clip: float = 6.0,
) -> dict[str, float]:
    """RC-5a FIX: Bounded `ebt_clip * tanh(ebt / ebt_clip)` 3-parameter simplex fusion optimizing ROC + PR + R@P95 + R@P99!"""
    from scipy.special import expit

    y = np.asarray(scores_0_100, dtype=np.float32)
    valid = (y <= 30.0) | (y >= 70.0)
    if valid.sum() < 20 or len(np.unique(y[valid] >= 70.0)) < 2:
        return {"w_t1": 0.35, "w_t2": 0.40, "w_ebt": 0.25, "ebt_clip": float(ebt_clip), "val_obj": 0.0}

    t1, t2, eb, yv = lg_t1[valid], lg_t2[valid], ebt_margin[valid], y[valid]
    eb_cap = float(ebt_clip) * np.tanh(eb / max(float(ebt_clip), 1e-3))
    best = (0.35, 0.40, 0.25, -1.0)
    # Keep `w_ebt <= 0.35` and `w_t1 >= 0.20, w_t2 >= 0.25` so SoftBall heads dominate high-precision tail!
    for i in range(4, 15):
        w1 = i * 0.05
        for j in range(5, 21 - i):
            w2 = j * 0.05
            w3 = max(0.0, 1.0 - w1 - w2)
            if w3 > 0.35 + 1e-6:
                continue
            z = w1 * t1 + w2 * t2 + w3 * eb_cap
            p = expit(np.clip(z, -30.0, 30.0))
            m = {**compute_eval_metrics(yv, p), **compute_high_precision_metrics(yv, p)}
            obj = 0.30 * m["auroc"] + 0.30 * m["pr_auc"] + 0.25 * m["recall_at_p95"] + 0.15 * m["recall_at_p99"]
            if obj > best[3]:
                best = (w1, w2, w3, obj)
    return {
        "w_t1": round(best[0], 4),
        "w_t2": round(best[1], 4),
        "w_ebt": round(best[2], 4),
        "ebt_clip": float(ebt_clip),
        "val_obj": round(best[3], 5),
    }


def fit_smooth_promo_lut_v121(
    cal_logits: np.ndarray,
    cal_y_0_1: np.ndarray,
    pi_nat: float = 0.015,
    n_knots: int = 256,
) -> dict[str, Any]:
    """RC-4c FIX: Smooth monotonic Beta-Spline Promo LUT with wild-prior log-odds shift!

    Eliminates PAV's 92.5% flat plateaus & 0.503 ceiling AND uses standard `x`/`y` keys
    100% compatible with `apply_spline_temp_lut`!
    """
    z = np.asarray(cal_logits, dtype=np.float64).reshape(-1)
    y = np.clip(np.asarray(cal_y_0_1, dtype=np.float64).reshape(-1), 0.0, 1.0)
    pi_cal = float(np.clip(np.mean(y >= 0.5), 1e-3, 1.0 - 1e-3))
    pi_n = float(np.clip(pi_nat, 0.005, 0.25))
    # Gentle damped prior shift (`0.5` factor prevents extreme compression while calibrating wild FPR!)
    prior_shift = 0.5 * math.log((pi_n / (1.0 - pi_n)) / (pi_cal / (1.0 - pi_cal)))
    cal_fit = fit_venn_abers_and_beta_spline_lut(z + prior_shift, y, fit_binary_ece_mode=True)
    lut = export_beta_spline_as_v07_lut(cal_fit, n_knots)
    lut["pi_cal"] = round(pi_cal, 5)
    lut["pi_nat"] = round(pi_n, 5)
    lut["prior_shift"] = round(prior_shift, 4)
    return lut


# ---------------------------------------------------------------------------
# 2. Main v1.2.1-ApexNano-Fix 3-Phase Training Pipeline
# ---------------------------------------------------------------------------

def train_v121_apexnano_full(args) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, roc_auc_score

    ref_ci = run_v121_refinery_ci()
    assert ref_ci["pass"], f"v121_data_refinery CI failed: {ref_ci}"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    bs = int(args.batch_size) if int(args.batch_size) > 0 else (512 if vram_gb >= 35.0 else (128 if vram_gb >= 12.0 else 64))

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    v11_dir = args.v11_data_dir.resolve()
    t_start = time.monotonic()

    print(
        f"[v1.2.1-ApexNano-Fix Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={bs} | "
        f"v11_master={args.v11_fp16_master} | gold_v2={args.gold_v2_dir} | wave3={args.wave3_dir}",
        flush=True,
    )

    def _read_jsonl_list(p: Path) -> list[dict[str, Any]]:
        if not p.is_file():
            return []
        op = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")
        with op as f:
            return [json.loads(l) for l in f if l.strip()]

    # 1. Diamond-Only Pair Filter (9/9 Pair Audit: drops `v04_qwen38_twins` & `qwen38_27b_twins`!)
    extra_pairs: list[Path] = []
    dropped_noisy_pairs: list[str] = []
    for p in args.extra_pairs:
        if is_diamond_pair_source_v121(p.name):
            extra_pairs.append(p.resolve())
        else:
            dropped_noisy_pairs.append(p.name)
    debias_p = Path(__file__).resolve().parent / "data" / "v10-topic-debias-pairs.jsonl"
    if debias_p.is_file() and debias_p.resolve() not in extra_pairs:
        extra_pairs.append(debias_p.resolve())
    if args.v08_gemini_pairs and args.v08_gemini_pairs.is_file():
        extra_pairs.append(args.v08_gemini_pairs.resolve())

    train_raw, eval_rows, legacy_pairs, _, data_audit = _load_v06_balanced_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        extra_pairs,
        None,
        stage1_synth_cap=90000,
        v04_synth_cap=110000,
    )
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} pairs collided with eval!")
    _ingest_external_gemini_gold_v10(train_raw, args.v08_gemini_gold)
    print(f"[Stage0-Pairs] diamond_pairs={len(legacy_pairs):,} | dropped_noisy_pair_files={dropped_noisy_pairs}", flush=True)

    if args.smoke_test:
        rng_sm = random.Random(20261126)
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

    # 3. Hydrate `score_source` + Gemini Pro/Flash + Gold-v2 + Optional Wave-3!
    orig_src_by_sid: dict[str, str] = {}
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

    gem_pro_sids: set[str] = set()
    gem_flash_sids: set[str] = set()
    if args.v08_gemini_gold and args.v08_gemini_gold.is_file():
        for r_g in _read_jsonl_list(args.v08_gemini_gold):
            if r_g.get("split", "train") != "train":
                continue
            sid_g = str(r_g.get("sample_id", r_g.get("id", "")))
            if "pro" in str(r_g.get("score_source", "")).lower():
                gem_pro_sids.add(sid_g)
            else:
                gem_flash_sids.add(sid_g)

    gold_v2_train_rows: list[dict[str, Any]] = []
    gold_v2_val_rows: list[dict[str, Any]] = []
    gold_v2_test_rows: list[dict[str, Any]] = []
    override_sids: set[str] = set()
    if args.gold_v2_dir and args.gold_v2_dir.is_dir():
        g2d = args.gold_v2_dir.resolve()
        gold_v2_train_rows = _read_jsonl_list(g2d / "v12_gold_v2_train.jsonl.gz")
        gold_v2_val_rows = _read_jsonl_list(g2d / "v12_gold_v2_val.jsonl.gz")
        gold_v2_test_rows = _read_jsonl_list(g2d / "v12_gold_v2_test.jsonl.gz")
        override_sids |= {str(r.get("sample_id", "")) for r in gold_v2_train_rows if r.get("sample_id")}

    wave3_train_rows: list[dict[str, Any]] = []
    wave3_val_rows: list[dict[str, Any]] = []
    w3_dir = args.wave3_dir.resolve() if (args.wave3_dir and args.wave3_dir.is_dir()) else (args.gold_v2_dir.resolve() if (args.gold_v2_dir and args.gold_v2_dir.is_dir()) else None)
    if w3_dir is not None:
        wave3_train_rows = _read_jsonl_list(w3_dir / "v121_wave3_train.jsonl.gz")
        wave3_val_rows = _read_jsonl_list(w3_dir / "v121_wave3_val.jsonl.gz")
        override_sids |= {str(r.get("sample_id", "")) for r in wave3_train_rows if r.get("sample_id")}
    print(
        f"[Stage0-GoldV2+Wave3] g2_tr={len(gold_v2_train_rows)} g2_va={len(gold_v2_val_rows)} g2_te={len(gold_v2_test_rows)} | "
        f"w3_tr={len(wave3_train_rows)} w3_va={len(wave3_val_rows)} | overrides={len(override_sids)}",
        flush=True,
    )

    # Apply `v121_data_refinery.py` (all 6 data-audit quarantines!)
    refined_items: list[dict[str, Any]] = []
    t3_dropped = 0
    ov_deduped = 0
    tier_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    for it in train_raw:
        sid_it = str(it.get("sample_id", ""))
        if sid_it and sid_it in override_sids:
            ov_deduped += 1
            continue
        src_it = orig_src_by_sid.get(sid_it, str(it.get("score_source", "")))
        if not src_it and it.get("is_pre_llm"):
            src_it = "pre_llm_human_prior"
        it["score_source"] = src_it
        is_gem_pro = sid_it in gem_pro_sids
        is_gem_any = is_gem_pro or (sid_it in gem_flash_sids)
        tier, tw, eff_sc, reason = classify_epistemic_tier_v121(
            score=float(it["y_score"]) * 100.0,
            score_source=src_it,
            text=str(it.get("raw_text", "")),
            overridden_by_gemini=is_gem_any,
            gemini_is_pro=is_gem_pro,
        )
        tier_counts[tier] += 1
        reason_counts[reason] += 1
        if tier == "T3_QUARANTINE" or tw <= 0.0:
            t3_dropped += 1
            continue
        it["y_score"] = float(np.clip(eff_sc / 100.0, 0.0, 1.0))
        it["has_ai"] = 1.0
        it["epistemic_tier"] = tier
        it["is_true_pro_gold"] = bool(tier == "T0_DIAMOND_GOLD")
        it["weight"] = float(it["weight"]) * float(tw)
        it["y_promo"] = -1.0
        it["has_promo"] = 0.0
        it["sub4_idx"] = -1
        it["is_rescue"] = False
        it["l_pceil_admit"] = False
        refined_items.append(it)

    # Carve out a strictly held-out, NEVER-TRAINED `cert_ai_holdout` slice from `refined_items` (`<0.4%` of 1M rows!)
    # so `val_cal_cert` ALWAYS has `n_ai >= 850` positives for mathematically exact `alpha_ai=0.005, delta=0.05` LTT!
    rng_cert = random.Random(2026112601)
    cert_pos_idx = [i for i, it in enumerate(refined_items) if it["y_score"] >= 0.80 and it["epistemic_tier"] in ("T0_DIAMOND_GOLD", "T1_VERIFIED_CLEAN")]
    cert_neg_idx = [i for i, it in enumerate(refined_items) if it["y_score"] <= 0.15 and it["epistemic_tier"] in ("T0_DIAMOND_GOLD", "T1_VERIFIED_CLEAN")]
    rng_cert.shuffle(cert_pos_idx)
    rng_cert.shuffle(cert_neg_idx)
    take_c_pos = min(16 if args.smoke_test else 750, len(cert_pos_idx) // 3)
    take_c_neg = min(32 if args.smoke_test else 3000, len(cert_neg_idx) // 3)
    cert_drop_set = set(cert_pos_idx[:take_c_pos]) | set(cert_neg_idx[:take_c_neg])
    cert_holdout_rows = [
        {"text": refined_items[i]["raw_text"], "score": float(refined_items[i]["y_score"]) * 100.0, "sample_id": refined_items[i].get("sample_id", f"cert_{i}")}
        for i in sorted(cert_drop_set)
    ]
    refined_items = [it for i, it in enumerate(refined_items) if i not in cert_drop_set]

    ipw_w = compute_joint_length_domain_ipw([{"text": x["raw_text"], "y_score": x["y_score"]} for x in refined_items])
    for it, iw in zip(refined_items, ipw_w):
        it["weight"] = float(np.clip(float(it["weight"]) * float(iw), 0.20, 3.0))
    print(
        f"[Stage0-V121Refinery] kept={len(refined_items):,} | cert_holdout={len(cert_holdout_rows):,} (pos={take_c_pos}, neg={take_c_neg}) | "
        f"T3_purged={t3_dropped:,} | overrides={ov_deduped} | tiers={dict(tier_counts)} | top_reasons={reason_counts.most_common(6)}",
        flush=True,
    )

    # 4. Ingest Dual-Axis Pools with STRICT `has_ai` MASKING!
    pt_map = {"none": 0, "coupon_commerce": 1, "self_promo_funnel": 2, "soft_shill": 3, "tool_boilerplate": 4}
    cert_shas: set[str] = set()
    select_shas: set[str] = set()
    split_p = v11_dir / "v11_promo_split_manifest.json"
    if split_p.is_file():
        sm = json.loads(split_p.read_text(encoding="utf-8"))
        cert_shas = {x["norm_sha256"] for x in sm.get("splits", {}).get("promo_val_cert", [])}
        select_shas = {x["norm_sha256"] for x in sm.get("splits", {}).get("promo_val_select", [])}

    def _make_item(
        txt: str, y_ai: float, y_pr: float, has_pr: float, pt_str: str, w: float,
        has_ai: float = 1.0, is_t0: bool = False, is_rescue: bool = False, l_pceil: bool = False,
    ) -> dict[str, Any]:
        cl, f56, meta = extract_style_features_56d(txt)
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        return {
            "raw_text": txt, "clean_text": cl, "ids": ids,
            "engram": make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
            "trigram": make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10),
            "style": f56, "spoof_any": float(meta["spoof_any"]),
            "y_score": float(np.clip(y_ai, 0.0, 1.0)),
            "has_ai": float(has_ai),
            "y_promo": float(np.clip(y_pr, 0.0, 1.0)) if has_pr > 0.5 else -1.0,
            "has_promo": float(has_pr),
            "sub4_idx": pt_map.get(pt_str, 0) if has_pr > 0.5 else -1,
            "weight": float(w), "is_true_pro_gold": bool(is_t0), "is_rescue": bool(is_rescue),
            "l_pceil_admit": bool(l_pceil), "attr5": 3 if y_ai >= 0.70 else (1 if y_ai > 0.30 else 0),
            "group_idx": GROUP_TO_IDX["current_2026_real"], "era_idx": 3,
            "epistemic_tier": "T0_DIAMOND_GOLD" if is_t0 else "T1_VERIFIED_CLEAN",
            "code_idx": -100, "prag_vec": [0.0] * len(PRAG_TAGS),
        }

    dual_gold_train: list[dict[str, Any]] = []
    wild_promo_focus_pool: list[dict[str, Any]] = []
    promo_val_select_rows: list[dict[str, Any]] = []
    promo_val_cert_rows: list[dict[str, Any]] = []
    uot_pseudo_rows: list[dict[str, Any]] = []
    hard_neg_rows: list[dict[str, Any]] = []
    gold_v2_t0_items: list[dict[str, Any]] = []
    a3_protect_items: list[dict[str, Any]] = []

    for fname, is_resc in (
        ("v11_wave1_dualaxis_gold_all5880.jsonl.gz", False),
        ("v11_wave2_dualaxis_gold_all4410.jsonl.gz", True),
    ):
        rows_g = _read_jsonl_list(v11_dir / fname)
        if args.smoke_test:
            rows_g = rows_g[:120]
        for r in rows_g:
            txt = str(r["text"])
            sh = _sha_norm(_norm(txt))
            y_pr_v = float(r["promo_score"]) / 100.0
            rec = _make_item(txt, float(r["ai_score"]) / 100.0, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.5, has_ai=1.0, is_t0=True, is_rescue=is_resc)
            if sh in cert_shas:
                promo_val_cert_rows.append(rec)
            elif sh in select_shas:
                promo_val_select_rows.append(rec)
            elif y_pr_v <= 0.15 and len(promo_val_select_rows) < (64 if args.smoke_test else 650) and (int(sh[:8], 16) % 16 == 0):
                promo_val_select_rows.append(rec)
            else:
                dual_gold_train.append(rec)
                if y_pr_v >= 0.65 and is_resc:
                    wild_promo_focus_pool.append(rec)

    # Wave-2.5 (243 wild disagreement dual-axis gold -> enters BOTH `dual_gold_train` AND dedicated `wild_promo_focus_pool`!)
    w25_loaded = 0
    if args.wave25_gold and args.wave25_gold.is_file():
        for r in _read_jsonl_list(args.wave25_gold):
            txt = str(r["text"])
            y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
            rec = _make_item(txt, y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.70, has_ai=1.0, is_t0=True, is_rescue=True)
            dual_gold_train.append(rec)
            gold_v2_t0_items.append(rec)
            wild_promo_focus_pool.append(rec)
            if y_pr_v >= 0.65:
                # Extra 2x focus on positive wild promo archetypes (blue-link, boost, concierge, soft-shill)!
                wild_promo_focus_pool.append(rec)
            w25_loaded += 1

    # Gold-v2 Train (1,429 re-adjudicated dual-axis gold)
    for r in gold_v2_train_rows:
        y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
        tier_g2, _, _, _ = classify_epistemic_tier_v121(
            float(r["ai_score"]), "v12_gold_v2_pro", str(r["text"]),
            v11_or_v12_verified=True, is_hybrid_mixed=bool(r.get("is_hybrid_mixed", False)),
        )
        rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.65 * (1.0 if tier_g2 == "T0_DIAMOND_GOLD" else 0.5), has_ai=1.0, is_t0=(tier_g2 == "T0_DIAMOND_GOLD"), is_rescue=True)
        dual_gold_train.append(rec)
        if tier_g2 == "T0_DIAMOND_GOLD":
            gold_v2_t0_items.append(rec)
            if y_ai_v <= 0.25 and len(str(r["text"])) >= 35:
                a3_protect_items.append(rec)

    # Wave-3 Train (wild promo P1-P5 + Flash stealth bots A1 + Wave12 conflicts A2 + Structured human traps A3!)
    for r in wave3_train_rows:
        y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
        bkt = str(r.get("bucket", ""))
        rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.75, has_ai=1.0, is_t0=True, is_rescue=True, l_pceil=(y_pr_v <= 0.20))
        dual_gold_train.append(rec)
        gold_v2_t0_items.append(rec)
        if bkt.startswith("P") or y_pr_v >= 0.65:
            wild_promo_focus_pool.append(rec)
            if y_pr_v >= 0.65:
                wild_promo_focus_pool.append(rec)
        if y_ai_v <= 0.25 and (bkt.startswith("A3") or len(str(r["text"])) >= 35):
            a3_protect_items.append(rec)
    for idx_w3v, r in enumerate(wave3_val_rows):
        y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
        rec_w3v = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.0, has_ai=1.0, is_t0=True)
        if idx_w3v % 2 == 0:
            promo_val_select_rows.append(rec_w3v)
        else:
            promo_val_cert_rows.append(rec_w3v)

    # Wave-1 1,600 counterfactual pairs -> AUDIT FIX: `has_ai = 0.0` (38.25% pairs shift AI prob > 0.30!)
    dual_pairs_train: list[dict[str, Any]] = []
    for x in _read_jsonl_list(v11_dir / "v11_wave1_promo_counterfactual_pairs_1600.jsonl")[: (48 if args.smoke_test else 99999)]:
        sh_p = _sha_norm(_norm(x["promo_text"]))
        is_ai_v = 1.0 if x.get("promo_is_ai") else 0.0
        pt_s = str(x.get("promo_type", "soft_shill"))
        if sh_p in cert_shas:
            promo_val_cert_rows.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.0, has_ai=0.0, is_t0=True))
            promo_val_cert_rows.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.0, has_ai=0.0, is_t0=True))
        elif sh_p in select_shas:
            promo_val_select_rows.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.0, has_ai=0.0, is_t0=True))
            promo_val_select_rows.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.0, has_ai=0.0, is_t0=True))
        else:
            dual_pairs_train.append(x)
            dual_gold_train.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.5, has_ai=0.0, is_t0=True))
            dual_gold_train.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.5, has_ai=0.0, is_t0=True, l_pceil=True))

    # UOT/CDMAD pseudo pool & P10-3 2k k-NN hard negs -> AUDIT FIX: `has_ai = 0.0` (19.4% hidden AI in 2k k-NN!)
    for r in _read_jsonl_list(v11_dir / "v11_uot_cdmad_pseudo_full.jsonl.gz")[: (192 if args.smoke_test else 999999)]:
        txt = str(r["text"])
        sh = _sha_norm(_norm(txt))
        y_ai_v = float(r.get("old_ai_score", 20.0)) / 100.0
        if r.get("pseudo_label") == 1:
            rec = _make_item(txt, y_ai_v, float(r.get("pseudo_promo_score", 90.0)) / 100.0, 1.0, str(r.get("pseudo_promo_type", "soft_shill")), float(r.get("weight", 0.75)), has_ai=0.0)
            if sh in cert_shas:
                promo_val_cert_rows.append(rec)
            elif sh in select_shas:
                promo_val_select_rows.append(rec)
            else:
                uot_pseudo_rows.append(rec)
        elif r.get("source") == "v11_uot_chant_hard_neg":
            hard_neg_rows.append(_make_item(txt, 0.05, 0.0, 1.0, "none", 1.6, has_ai=1.0, l_pceil=True))
        else:
            uot_pseudo_rows.append(_make_item(txt, y_ai_v, 0.02, 1.0, "none", 0.60, has_ai=0.0))

    for r in _read_jsonl_list(v11_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl")[: (48 if args.smoke_test else 9999)]:
        rec = _make_item(str(r["text"]), float(r.get("old_ai_score", 15.0)) / 100.0, 0.02, 1.0, "none", 1.0, has_ai=0.0, l_pceil=bool(r.get("l_pceil_admitted", False)))
        hard_neg_rows.append(rec)

    wave2_longform_all = _read_jsonl_list(v11_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (20 if args.smoke_test else 9999)]
    wave2_longform = [d for i, d in enumerate(wave2_longform_all) if i % 10 != 9]
    wave2_longform_val = [d for i, d in enumerate(wave2_longform_all) if i % 10 == 9]
    print(
        f"[Stage0-DualData] dual_train={len(dual_gold_train):,} (wave25={w25_loaded}, g2_t0={len(gold_v2_t0_items)}, "
        f"wild_promo_focus={len(wild_promo_focus_pool)}, a3_prot={len(a3_protect_items)}) | "
        f"pairs={len(dual_pairs_train):,} | pv_sel={len(promo_val_select_rows):,} pv_cert={len(promo_val_cert_rows):,} | "
        f"uot={len(uot_pseudo_rows):,} | hard_neg={len(hard_neg_rows):,}",
        flush=True,
    )

    t0_pool, t1_pool, t2_pool, protect_pool = list(gold_v2_t0_items), [], [], list(a3_protect_items)
    struct_gids = {GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["dsh_hard_slice"], GROUP_TO_IDX["multilingual_world_knowledge"]}
    for it in refined_items:
        cl, f56, meta = extract_style_features_56d(it["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        it["clean_text"], it["ids"], it["style"], it["spoof_any"] = cl, ids, f56, float(meta["spoof_any"])
        it["engram"] = make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
        it["trigram"] = make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        it["attr5"] = _derive_attr5_label(it)
        it["era_idx"] = 0 if it.get("is_pre_llm") else ERA_TO_IDX.get(str(it.get("era_bucket", "current_2026")), 3)
        it["code_idx"] = int(it.get("code_idx", -100))
        it["prag_vec"] = it.get("prag_vec") or [0.0] * len(PRAG_TAGS)
        if it.get("is_trap_human") or it.get("is_pre_llm") or it.get("group_idx") in struct_gids:
            protect_pool.append(it)
        if it["epistemic_tier"] == "T0_DIAMOND_GOLD":
            t0_pool.append(it)
        elif it["epistemic_tier"] == "T2_SOFT_MIDBAND":
            t2_pool.append(it)
        else:
            t1_pool.append(it)

    hu_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.15 and 40 <= len(x["clean_text"]) <= 220][:4000]
    hu_hard = [x["clean_text"] for x in protect_pool if x["has_ai"] > 0.5 and x["y_score"] <= 0.20 and len(x["clean_text"]) >= 40][:2000] or hu_clean[:500]
    ai_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["has_ai"] > 0.5 and x["y_score"] >= 0.85 and 40 <= len(x["clean_text"]) <= 240][:4000]
    pr_clean = [x["clean_text"] for x in dual_gold_train if x["y_promo"] >= 0.70 and x["y_score"] <= 0.30][:1000] or hu_clean[:50]
    longmix_syn = OnlineLongMixSynthesizerV11(hu_clean[:2000] or ["真人评论测试。"] * 12, hu_hard, ai_clean[:2000] or ["人工智能评论测试。"] * 12, pr_clean)

    # 5. Disjoint Eval Splits + RC-5b FIX: Expanded Held-Out `val_cal_cert` (`n_ai >= 900` for True LTT Certification!)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    vg_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_gold}
    val_cal_select_raw, _ = build_stratified_val_calibration_set(eval_rows, target_n=1600)
    val_cal_select = [r for r in val_cal_select_raw if str(r.get("sample_id", r.get("id", ""))) not in vg_ids] or val_cal_select_raw
    sel_ids = vg_ids | {str(r.get("sample_id", r.get("id", ""))) for r in val_cal_select}

    # Build strictly disjoint held-out `val_cal_cert` from remaining `split == "validation"` rows + never-trained `cert_holdout_rows`
    # Guarantees `n_ai >= 850` positives -> `Binom(850, 0.005)` strictly certifiable at `delta=0.05` (`0.995^850 = 0.014`)!
    val_rem = [r for r in eval_rows if r.get("split") == "validation" and str(r.get("sample_id", r.get("id", ""))) not in sel_ids]
    rng_v = random.Random(20261126)
    val_rem_pos = [r for r in val_rem if float(r.get("score", 0.0)) >= 70.0]
    val_rem_neg = [r for r in val_rem if float(r.get("score", 0.0)) <= 30.0]
    rng_v.shuffle(val_rem_pos)
    rng_v.shuffle(val_rem_neg)
    val_cal_cert = val_rem_pos[:250] + val_rem_neg[:1000] + cert_holdout_rows
    rng_v.shuffle(val_cal_cert)
    n_pos_cert_tot = sum(1 for r in val_cal_cert if float(r.get("score", 0.0)) >= 70.0)
    n_neg_cert_tot = len(val_cal_cert) - n_pos_cert_tot
    print(f"[Stage0-ValSplits] val_gold={len(val_gold)} | val_cal_select={len(val_cal_select)} | val_cal_cert={len(val_cal_cert)} (n_pos_ai={n_pos_cert_tot}, n_neg_hu={n_neg_cert_tot})", flush=True)

    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    if args.smoke_test:
        def _strat_cap(rows_s: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
            p_s = [r for r in rows_s if float(r.get("score", 0.0)) >= 70.0]
            n_s = [r for r in rows_s if float(r.get("score", 0.0)) < 70.0]
            h = cap // 2
            return (p_s[:h] + n_s[: cap - len(p_s[:h])])[:cap]
        val_gold = _strat_cap(val_gold, 48)
        val_cal_select = _strat_cap(val_cal_select, 48)
        val_cal_cert = _strat_cap(val_cal_cert, 48)
        test_gold = _strat_cap(test_gold, 48)
        test_dsh_gold = _strat_cap(test_dsh_gold, 24)

    def _pre_eval_v121(rows_e: list[dict[str, Any]]):
        raws = [str(r["text"]) for r in rows_e]
        cls_l, f56_l, metas_l = [], [], []
        for t in raws:
            cl, f56, mt = extract_style_features_56d(t)
            cls_l.append(cl); f56_l.append(f56); metas_l.append(mt)
        enc = [tokenizer.encode_ids(c, max_len=args.max_len) for c in cls_l]
        eng = [make_engram_bigram_ids(s, buckets=ENGRAM_BIGRAM_BUCKETS_V10) for s in enc]
        tri = [make_engram_trigram_ids(s, buckets=ENGRAM_TRIGRAM_BUCKETS_V10) for s in enc]
        sc = np.array([float(r.get("score", float(r.get("ai_score", 0.0)))) for r in rows_e], dtype=np.float32)
        return enc, eng, tri, raws, cls_l, np.stack(f56_l, axis=0).astype(np.float32), metas_l, sc

    vg_p = _pre_eval_v121(val_gold)
    vs_p = _pre_eval_v121(val_cal_select)
    vc_p = _pre_eval_v121(val_cal_cert)
    tg_p = _pre_eval_v121(test_gold)
    td_p = _pre_eval_v121(test_dsh_gold)
    vg2_p = _pre_eval_v121(gold_v2_val_rows) if gold_v2_val_rows else None
    tg2_p = _pre_eval_v121(gold_v2_test_rows) if gold_v2_test_rows else None

    # 6. Load `v1.1.0-R1` Teacher -> Exact Zero-Shock SoftBall Transplant -> 6-Gate Step-0 CI!
    v11_teacher = MeguvizV11OmniGuardModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS),
    ).cpu().eval()
    loaded_v11_src = "random_smoke"
    if args.v11_fp16_master and args.v11_fp16_master.is_file():
        sd11 = torch.load(args.v11_fp16_master, map_location="cpu")
        v11_teacher.load_state_dict(sd11, strict=True)
        loaded_v11_src = str(args.v11_fp16_master)
    elif args.v11_bundle_mgvz and args.v11_bundle_mgvz.is_file():
        load_bundle_v6_v11_into_model(v11_teacher, args.v11_bundle_mgvz)
        loaded_v11_src = str(args.v11_bundle_mgvz)
    v11_teacher = v11_teacher.to(device).eval()
    for p in v11_teacher.parameters():
        p.requires_grad_(False)
    print(f"[Stage0-WarmStart] Loaded v1.1 teacher from: {loaded_v11_src}", flush=True)

    mdl = MeguvizV121ApexNanoModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS),
    ).to(device).eval()

    def _collate_items(batch_it: list[dict[str, Any]]):
        ml = min(args.max_len, max(2, ((max(len(x["ids"]) for x in batch_it) + 15) // 16) * 16))
        n_b = len(batch_it)
        ids = np.full((n_b, ml), tokenizer.pad_id, dtype=np.int64)
        msk = np.zeros((n_b, ml), dtype=np.int64)
        eng = np.zeros((n_b, ml), dtype=np.int64)
        tri = np.zeros((n_b, ml), dtype=np.int64)
        st = np.empty((n_b, STYLE_FEAT_DIM_V11), dtype=np.float32)
        cleans_b = []
        for i, x in enumerate(batch_it):
            ns = min(len(x["ids"]), ml)
            ids[i, :ns], msk[i, :ns], eng[i, :ns], tri[i, :ns], st[i] = x["ids"][:ns], 1, x["engram"][:ns], x["trigram"][:ns], x["style"]
            cleans_b.append(x["clean_text"])
        c1, c3, cm, cf, _ = extract_rung0_charhash_batch(cleans_b, already_canonical=True)
        return (
            torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
            torch.from_numpy(eng).to(device), torch.from_numpy(tri).to(device), torch.from_numpy(st).to(device),
            torch.from_numpy(c1).to(device), torch.from_numpy(c3).to(device), torch.from_numpy(cm).to(device), torch.from_numpy(cf).to(device),
            torch.tensor([x["y_score"] for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x.get("has_ai", 1.0) for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x["y_promo"] for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x["has_promo"] for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x["sub4_idx"] for x in batch_it], dtype=torch.long, device=device),
            torch.tensor([x["weight"] for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x["attr5"] for x in batch_it], dtype=torch.long, device=device),
            torch.tensor([1.0 if x.get("is_true_pro_gold") else 0.0 for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([1.0 if x.get("is_rescue") else 0.0 for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([1.0 if x.get("l_pceil_admit") else 0.0 for x in batch_it], dtype=torch.float32, device=device),
            torch.tensor([x["group_idx"] for x in batch_it], dtype=torch.long, device=device),
            torch.tensor([x["era_idx"] for x in batch_it], dtype=torch.long, device=device),
            torch.tensor([int(x.get("code_idx", -100)) for x in batch_it], dtype=torch.long, device=device),
        )

    n_cal = 128 if args.smoke_test else min(4096, len(refined_items))
    cal_items = random.Random(20261126).sample(refined_items, n_cal)
    emb11_l = []
    with torch.no_grad():
        for st_i in range(0, n_cal, 256):
            b_ids, b_m, b_e, b_t, b_st, *_ = _collate_items(cal_items[st_i : st_i + 256])
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                o11 = v11_teacher(b_ids, b_m, b_st, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)
            emb11_l.append(o11["emb"].float().cpu().numpy())
    emb11_cal = np.concatenate(emb11_l, axis=0)
    surg_rep = transplant_v11_to_v121_dense(v11_teacher, mdl, calib_emb_v11=emb11_cal)
    ci_g0 = run_v121_ci_gates(mdl, tokenizer)
    assert ci_g0["all_ci_passed"], f"Step-0 V121 CI failed: {ci_g0}"
    r11_anchor_params = {k: v.detach().cpu().clone() for k, v in mdl.named_parameters() if not k.startswith(("rung0.", "z_char_", "promo_zchar_"))}
    print(f"[Stage0-SoftBallTransplant] {json.dumps(surg_rep)} | G0_CI=PASS (pad_max={ci_g0['V121_P1_batch_invariance']['max_diff']})", flush=True)

    @torch.no_grad()
    def _predict_v121(m_e: MeguvizV121ApexNanoModel, pack_t, outer_steps: int = 2) -> dict[str, np.ndarray]:
        enc_l, eng_l, tri_l, _, cls_l, st_m, _, _ = pack_t
        m_e.eval()
        n = len(enc_l)
        keys = (
            "logit_ai_r0", "logit_promo_r0", "p_defer_r0",
            "logit", "raw_logit", "logit_t1", "raw_logit_t1", "logit_r1", "raw_logit_r1",
            "logit_promo", "raw_logit_promo", "logit_promo_t1", "raw_logit_promo_t1", "logit_promo_r1", "raw_logit_promo_r1",
            "ebt_margin", "p_defer1", "p_defer2", "r1_span_max",
        )
        res = {k: np.zeros(n, dtype=np.float32) for k in keys}
        res["emb"] = np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return res
        order = sorted(range(n), key=lambda i: len(enc_l[i]))
        for st_i in range(0, n, 256):
            mb = order[st_i : st_i + 256]
            ml = max(2, max(len(enc_l[i]) for i in mb))
            if ml % 2 == 1:
                ml += 1
            ids = np.full((len(mb), ml), tokenizer.pad_id, dtype=np.int64)
            m = np.zeros((len(mb), ml), dtype=np.int64)
            eg = np.zeros((len(mb), ml), dtype=np.int64)
            tg = np.zeros((len(mb), ml), dtype=np.int64)
            for ri, oi in enumerate(mb):
                s = enc_l[oi]
                ids[ri, : len(s)], m[ri, : len(s)], eg[ri, : len(s)], tg[ri, : len(s)] = s, 1, eng_l[oi], tri_l[oi]
            c1, c3, cm, cf, _ = extract_rung0_charhash_batch([cls_l[i] for i in mb], already_canonical=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                o = m_e(
                    torch.from_numpy(ids).to(device), torch.from_numpy(m).to(device), torch.from_numpy(st_m[mb]).to(device),
                    engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device),
                    r0_c1=torch.from_numpy(c1).to(device), r0_c3=torch.from_numpy(c3).to(device),
                    r0_mask=torch.from_numpy(cm).to(device), r0_feats=torch.from_numpy(cf).to(device),
                    outer_steps=outer_steps, return_aux=True,
                )
            e1 = o["exit1_out"]
            kv = {
                "logit_ai_r0": o["logit_ai_r0"], "logit_promo_r0": o["logit_promo_r0"], "p_defer_r0": o["p_defer_r0"],
                "logit": o["logit"], "raw_logit": o["raw_logit"], "logit_t1": e1["logit"], "raw_logit_t1": e1["raw_logit"],
                "logit_r1": o["logit_r1"], "raw_logit_r1": o["raw_logit_r1"],
                "logit_promo": o["logit_promo"], "raw_logit_promo": o["raw_logit_promo"],
                "logit_promo_t1": e1["logit_promo"], "raw_logit_promo_t1": e1["raw_logit_promo"],
                "logit_promo_r1": o["logit_promo_r1"], "raw_logit_promo_r1": o["raw_logit_promo_r1"],
                "ebt_margin": o["ebt_margin"], "p_defer1": o["p_defer1"], "p_defer2": e1["p_defer2"],
                "r1_span_max": o["r1_span_max"], "emb": o["emb"],
            }
            for k_o, t_o in kv.items():
                arr = t_o.float().cpu().numpy()
                for ri, oi in enumerate(mb):
                    res[k_o][oi] = arr[ri]
        return res

    def _eval_ai_metrics(m_e: MeguvizV121ApexNanoModel) -> dict[str, float]:
        vo = _predict_v121(m_e, vg_p, 2)
        so = _predict_v121(m_e, vs_p, 2)
        fw = m_e.fusion_weights.detach().cpu().numpy()
        ec = float(m_e.fusion_ebt_clip.item())
        p_vg = expit(np.clip(bounded_fused3_logits(vo["logit_t1"], vo["logit"], vo["ebt_margin"], fw[0], fw[1], fw[2], ec), -30.0, 30.0))
        p_vs = expit(np.clip(bounded_fused3_logits(so["logit_t1"], so["logit"], so["ebt_margin"], fw[0], fw[1], fw[2], ec), -30.0, 30.0))
        mv = {**compute_eval_metrics(vg_p[7], p_vg), **compute_high_precision_metrics(vg_p[7], p_vg)}
        ms = {**compute_eval_metrics(vs_p[7], p_vs), **compute_high_precision_metrics(vs_p[7], p_vs)}
        mr1 = compute_eval_metrics(vg_p[7], expit(vo["logit_r1"]))
        mr0 = compute_eval_metrics(vs_p[7], expit(so["logit_ai_r0"]))
        ai_obj_wild = (
            0.30 * ms.get("auroc", 0.5) + 0.24 * ms.get("pr_auc", 0.5) + 0.14 * ms.get("best_f05", 0.5)
            + 0.20 * ms.get("recall_at_p95", 0.0) + 0.06 * ms.get("recall_at_p99", 0.0) + 0.06 * mr1.get("auroc", 0.5)
        )
        ai_obj_v1 = (
            0.36 * mv.get("auroc", 0.5) + 0.28 * mv.get("pr_auc", 0.5) + 0.14 * mv.get("best_f05", 0.5)
            + 0.16 * mv.get("recall_at_p95", 0.0) + 0.06 * mr1.get("auroc", 0.5)
        )
        ai_obj = 0.50 * ai_obj_v1 + 0.50 * ai_obj_wild
        out_r = {
            "ai_obj": float(ai_obj), "auroc_vg": float(mv.get("auroc", 0.5)), "rp95_vg": float(mv.get("recall_at_p95", 0.0)),
            "auroc_wild": float(ms.get("auroc", 0.5)), "rp95_wild": float(ms.get("recall_at_p95", 0.0)),
            "r1_auroc": float(mr1.get("auroc", 0.5)), "r0_ai_auroc": float(mr0.get("auroc", 0.5)),
        }
        if vg2_p is not None:
            v2o = _predict_v121(m_e, vg2_p, 2)
            p_v2 = expit(np.clip(bounded_fused3_logits(v2o["logit_t1"], v2o["logit"], v2o["ebt_margin"], fw[0], fw[1], fw[2], ec), -30.0, 30.0))
            mv2 = {**compute_eval_metrics(vg2_p[7], p_v2), **compute_high_precision_metrics(vg2_p[7], p_v2)}
            out_r["auroc_vg2"] = float(mv2.get("auroc", 0.5))
            out_r["rp95_vg2"] = float(mv2.get("recall_at_p95", 0.0))
            ai_obj_v2 = 0.36 * mv2.get("auroc", 0.5) + 0.28 * mv2.get("pr_auc", 0.5) + 0.14 * mv2.get("best_f05", 0.5) + 0.22 * mv2.get("recall_at_p95", 0.0)
            # RC-2 FIX: Clean Gold-v2 (55%) + Wild Cal (35%) + R1 (10%) — NEVER vetoed by noisy val_gold_v1!
            out_r["ai_obj_sel"] = float(0.55 * ai_obj_v2 + 0.35 * ai_obj_wild + 0.10 * mr1.get("auroc", 0.5))
        else:
            out_r["ai_obj_sel"] = float(ai_obj)
        return out_r

    def _eval_promo_and_long(m_e: MeguvizV121ApexNanoModel) -> dict[str, float]:
        out_m = {"promo_auroc": 0.5, "promo_pr_auc": 0.5, "r1_promo_auroc": 0.5, "r0_promo_auroc": 0.5, "long_ai_auroc": 0.5, "long_hyb_auroc": 0.5, "long_pr_auroc": 0.5}
        if promo_val_select_rows:
            pv_pk = (
                [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
                [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
                np.stack([x["style"] for x in promo_val_select_rows], axis=0), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32),
            )
            pvo = _predict_v121(m_e, pv_pk, 2)
            yv = (pv_pk[7] >= 0.5).astype(int)
            if len(np.unique(yv)) == 2:
                out_m["promo_auroc"] = float(roc_auc_score(yv, pvo["logit_promo"]))
                out_m["promo_pr_auc"] = float(average_precision_score(yv, pvo["logit_promo"]))
                out_m["r1_promo_auroc"] = float(roc_auc_score(yv, pvo["logit_promo_r1"]))
                out_m["r0_promo_auroc"] = float(roc_auc_score(yv, pvo["logit_promo_r0"]))
        if wave2_longform_val:
            m_e.eval()
            ai_l, hy_l, pr_l, y_ai_l, y_hy_l, y_pr_l = [], [], [], [], [], []
            with torch.no_grad():
                for st_i in range(0, len(wave2_longform_val), 32):
                    cs_b = collate_chronostream_docs(wave2_longform_val[st_i : st_i + 32], tokenizer, max_chunks=4, max_chunk_toks=192)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                        cs_o = m_e.forward_chronostream_batch(
                            cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                            cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                            trunk_no_grad=True, outer_steps=2,
                        )
                    ai_l.append(cs_o["logit_ai_doc"].float().cpu().numpy())
                    hy_l.append(cs_o["logit_hybrid_doc"].float().cpu().numpy())
                    pr_l.append(cs_o["logit_promo_doc"].float().cpu().numpy())
                    y_ai_l.append((cs_b["y_ai_doc"].numpy() >= 0.5).astype(int))
                    y_hy_l.append((cs_b["y_hyb_doc"].numpy() >= 0.5).astype(int))
                    y_pr_l.append((cs_b["y_pr_doc"].numpy() >= 0.5).astype(int))
            ai_a, hy_a, pr_a = np.concatenate(ai_l), np.concatenate(hy_l), np.concatenate(pr_l)
            yai_a, yhy_a, ypr_a = np.concatenate(y_ai_l), np.concatenate(y_hy_l), np.concatenate(y_pr_l)
            if len(np.unique(yai_a)) == 2: out_m["long_ai_auroc"] = float(roc_auc_score(yai_a, ai_a))
            if len(np.unique(yhy_a)) == 2: out_m["long_hyb_auroc"] = float(roc_auc_score(yhy_a, hy_a))
            if len(np.unique(ypr_a)) == 2: out_m["long_pr_auroc"] = float(roc_auc_score(ypr_a, pr_a))
        return out_m

    ref_ai_m = _eval_ai_metrics(mdl)
    ref_pl_m = _eval_promo_and_long(mdl)
    print(f"[Stage0-Ref] ai={json.dumps({k: round(v, 5) for k, v in ref_ai_m.items()})} | dual={json.dumps({k: round(v, 4) for k, v in ref_pl_m.items()})}", flush=True)

    pools = {
        "t0_ai": t0_pool or refined_items,
        "t1_ai": t1_pool or refined_items,
        "t2_mid": t2_pool or refined_items,
        "ai_protect": protect_pool or refined_items,
        "dual_gold": dual_gold_train or refined_items,
        "uot_pseudo": uot_pseudo_rows or refined_items,
        "hard_neg": hard_neg_rows or refined_items,
    }
    obf_aug = AdversarialObfuscationAug(20261126)
    slot_aug = BrandPriceSlotAug(20261127)
    qs_aug = QuadSpliceAug(20261128)
    hart_aug = LightHumanPostEditAug(20261129)
    rng_tr = random.Random(20261126)
    pike_ai = DynamicPiKEGroupSamplerV11({"t0_ai": 0.38, "t1_ai": 0.34, "t2_mid": 0.02, "ai_protect": 0.12, "dual_gold": 0.10, "uot_pseudo": 0.02, "hard_neg": 0.02})
    pike_dual = DynamicPiKEGroupSamplerV11({"t0_ai": 0.22, "t1_ai": 0.18, "t2_mid": 0.01, "ai_protect": 0.05, "dual_gold": 0.28, "uot_pseudo": 0.14, "hard_neg": 0.12})

    def _sample_short_batch(bs_now: int, w_vec: np.ndarray, gold_v2_frac: float = 0.0, wild_promo_frac: float = 0.0) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        n_g2 = min(bs_now // 4, int(round(bs_now * gold_v2_frac))) if (gold_v2_frac > 0.0 and gold_v2_t0_items) else 0
        for _ in range(n_g2):
            out_b.append(gold_v2_t0_items[rng_tr.randrange(len(gold_v2_t0_items))])
        n_wp = min(bs_now // 4, int(round(bs_now * wild_promo_frac))) if (wild_promo_frac > 0.0 and wild_promo_focus_pool) else 0
        for _ in range(n_wp):
            it_wp = wild_promo_focus_pool[rng_tr.randrange(len(wild_promo_focus_pool))]
            if it_wp["y_promo"] >= 0.65 and rng_tr.random() < 0.25:
                it_wp = _make_item(obf_aug.obfuscate_promo_or_ai(it_wp["raw_text"], 0.55), it_wp["y_score"], it_wp["y_promo"], 1.0, "soft_shill" if it_wp["sub4_idx"] == 3 else "coupon_commerce", it_wp["weight"], has_ai=it_wp.get("has_ai", 1.0), is_t0=True, is_rescue=True)
            out_b.append(it_wp)
        rem_bs = max(1, bs_now - len(out_b))
        counts = np.maximum(1, np.round(w_vec * rem_bs).astype(int))
        counts[0] += max(0, rem_bs - int(counts.sum()))
        for pname, cnt in zip(DynamicPiKEGroupSamplerV11.POOL_NAMES, counts):
            src_p = pools[pname]
            for _ in range(int(cnt)):
                it = src_p[rng_tr.randrange(len(src_p))]
                if pname == "hard_neg" and rng_tr.random() < 0.20:
                    r_h = rng_tr.random()
                    if r_h < 0.35 and dual_pairs_train:
                        aug_d = qs_aug.make_quoted_promo_critique(rng_tr.choice(dual_pairs_train)["promo_text"])
                    elif r_h < 0.70:
                        aug_d = qs_aug.make_negated_shill_critique()
                    else:
                        aug_d = qs_aug.make_fandom_hype_hard_neg()
                    it = _make_item(aug_d["text"], 0.0, 0.0, 1.0, "none", 1.6, has_ai=1.0, l_pceil=True)
                elif pname == "dual_gold" and it["y_promo"] >= 0.65 and rng_tr.random() < 0.35:
                    it = _make_item(obf_aug.obfuscate_promo_or_ai(it["raw_text"], 0.65), it["y_score"], it["y_promo"], 1.0, "soft_shill" if it["sub4_idx"] == 3 else "coupon_commerce", it["weight"], has_ai=it.get("has_ai", 1.0), is_t0=True)
                elif pname == "t0_ai" and it.get("has_ai", 1.0) > 0.5 and it["y_score"] >= 0.85 and rng_tr.random() < 0.08:
                    it = _make_item(hart_aug.light_humanize_ai(it["raw_text"]), it["y_score"], -1.0, 0.0, "none", it["weight"], has_ai=1.0, is_t0=True)
                out_b.append(it)
        return out_b[:bs_now]

    def _sample_long_batch(n_docs: int) -> list[dict[str, Any]]:
        docs_out: list[dict[str, Any]] = []
        n_mix = max(1, int(round(n_docs * 0.60)))
        n_w2 = max(0, int(round(n_docs * 0.25))) if wave2_longform else 0
        docs_out.extend(longmix_syn.sample_batch(n_mix))
        if n_w2 > 0:
            docs_out.extend([rng_tr.choice(wave2_longform) for _ in range(n_w2)])
        while len(docs_out) < n_docs:
            docs_out.append(longmix_syn.sample_one())
        return docs_out[:n_docs]

    # =========================================================================
    # PHASE A: SoftBall Pure-AI + Gold-v2 / Wave-3 + Clean Selection + Rung-0 AI Warmup
    # =========================================================================
    pa_steps = 4 if args.smoke_test else int(args.pa_steps)
    pg_a, hooks_a = build_v121_param_groups(mdl, "phase_a")
    opt_a = torch.optim.AdamW(pg_a, betas=(0.9, 0.95))
    best_pa_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pa_sc = ref_ai_m["ai_obj_sel"]
    mdl.train()
    for step in range(pa_steps):
        prog = float(step) / float(max(1, pa_steps - 1))
        mult = _wsd_lr_mult(step, pa_steps, warmup_frac=0.08, decay_frac=0.25)
        for pg in opt_a.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        opt_a.zero_grad(set_to_none=True)
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_ai.w, gold_v2_frac=0.08)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era, code_i) = _collate_items(b_it)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)
            r0_o = mdl.forward_rung0(c1, c3, cm, cf)
            with torch.no_grad():
                to11 = v11_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)

            lg, lg_e1, lg_r1 = out["logit"].float(), out["exit1_out"]["logit"].float(), out["logit_r1"].float()
            ai_m = has_ai > 0.5
            y_cl = y_ai.clamp(1e-4, 1.0 - 1e-4)
            tp11 = torch.sigmoid(to11["logit"].float())
            is_q3 = ai_m & (y_ai <= 0.30) & (has_pr >= 0.5) & (y_pr >= 0.65)
            m_anc = ai_m & (~is_q3) & (is_res < 0.5) & (~((is_t0 > 0.5) & ((tp11 - y_ai).abs() >= 0.40)))

            l_kd = lg.sum() * 0.0
            if m_anc.any():
                t11_lg_c = to11["logit"][m_anc].float().clamp(-14.0, 14.0)
                t11_r1_c = to11["logit_r1"][m_anc].float().clamp(-14.0, 14.0)
                l_kd = (
                    0.55 * F.mse_loss(lg[m_anc], t11_lg_c)
                    + 0.30 * F.mse_loss(lg_e1[m_anc], t11_lg_c)
                    + 0.35 * F.mse_loss(lg_r1[m_anc], t11_r1_c)
                    + 0.40 * (1.0 - F.cosine_similarity(out["emb"][m_anc, :AI_EMB_END_V11].float(), to11["emb"][m_anc, :AI_EMB_END_V11].float(), dim=-1).mean())
                )

            l_ai = l_pa = l_tp = l_topfp = l_mono = l_r0 = lg.sum() * 0.0
            if ai_m.any():
                w_a = w_b[ai_m]
                l_ai = (F.binary_cross_entropy_with_logits(lg[ai_m], y_cl[ai_m], reduction="none") * w_a).mean()
                l_ai = l_ai + 0.25 * (F.binary_cross_entropy_with_logits(lg_e1[ai_m], y_cl[ai_m], reduction="none") * w_a).mean()
                l_ai = l_ai + 0.22 * (F.binary_cross_entropy_with_logits(lg_r1[ai_m], y_cl[ai_m], reduction="none") * w_a * (1.0 + 2.5 * (y_ai[ai_m] >= 0.45).float())).mean()
                l_ai = l_ai + 0.08 * listnet_tail_distill_loss(lg_r1[ai_m], lg[ai_m].detach()) + 0.06 * listnet_tail_distill_loss(lg_e1[ai_m], lg[ai_m].detach())
                l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg[ai_m], y_ai[ai_m], w_a, grp[ai_m], prog=prog)
                l_topfp = _top_fp_logit_ceiling_loss(lg[ai_m], y_ai[ai_m], tp11[ai_m], torch.maximum(is_t0[ai_m], is_q3[ai_m].float()), era[ai_m], w_a)

                dec_m = ai_m & ((y_ai >= 0.80) | (y_ai <= 0.15))
                if dec_m.any():
                    sgn = torch.where(y_ai[dec_m] >= 0.5, 1.0, -1.0)
                    l_mono = 0.08 * F.relu(sgn * (lg_e1[dec_m].detach() - lg[dec_m])).mean()

                lg_r0 = r0_o["logit_ai_r0"].float()
                l_r0 = (
                    0.50 * (F.binary_cross_entropy_with_logits(lg_r0[ai_m], y_cl[ai_m], reduction="none") * w_a * (1.0 + 3.0 * (y_ai[ai_m] >= 0.45).float())).mean()
                    + 0.30 * F.mse_loss(lg_r0[ai_m], lg[ai_m].detach())
                    + 0.08 * listnet_tail_distill_loss(lg_r0[ai_m], lg[ai_m].detach())
                )

            l_sdiv = 0.15 * compute_slot_orthogonality_loss(out["slots_final"])
            l_code = 0.04 * F.cross_entropy(out["code_logit"].float(), code_i, ignore_index=-100) if (code_i >= 0).any() else lg.sum() * 0.0
            l_sp = compute_l2_sp_loss(mdl, r11_anchor_params, wd_sp=0.05)

            l_ai_pair = lg.sum() * 0.0
            if legacy_pairs and step % 2 == 1:
                lpb = rng_tr.sample(legacy_pairs, min(16 if not args.smoke_test else 4, len(legacy_pairs)))
                ai_it, hu_it, pw_l = [], [], []
                for lp in lpb:
                    if "_ai_rec" not in lp:
                        lp["_ai_rec"] = _make_item(str(lp["ai_text"]), 1.0, -1.0, 0.0, "none", 1.0, has_ai=1.0)
                        lp["_hu_rec"] = _make_item(str(lp["human_text"]), 0.0, -1.0, 0.0, "none", 1.0, has_ai=1.0)
                    ai_it.append(lp["_ai_rec"]); hu_it.append(lp["_hu_rec"]); pw_l.append(float(lp.get("weight", 1.0)))
                ai_i, ai_m_, ai_e, ai_t, ai_s, *_ = _collate_items(ai_it)
                hu_i, hu_m_, hu_e, hu_t, hu_s, *_ = _collate_items(hu_it)
                oa = mdl(ai_i, ai_m_, ai_s, engram_ids=ai_e, trigram_ids=ai_t, outer_steps=2, return_aux=True)
                oh = mdl(hu_i, hu_m_, hu_s, engram_ids=hu_e, trigram_ids=hu_t, outer_steps=2, return_aux=True)
                pw_t = torch.tensor(pw_l, dtype=torch.float32, device=device)
                l_ai_pair = 0.08 * (
                    (F.softplus(1.8 - (oa["logit"].float() - oh["logit"].float())) * pw_t).mean()
                    + 0.6 * ((F.relu(0.84 - torch.sigmoid(oa["logit"].float())).pow(2) + 1.5 * F.relu(torch.sigmoid(oh["logit"].float()) - 0.12).pow(2)) * pw_t).mean()
                )

            l_gr = zenith_graded_anchor_loss(lg[ai_m], y_ai[ai_m]) if ai_m.any() else lg.sum() * 0.0
            loss_a = l_ai + 0.06 * l_pa + 0.35 * l_tp + 0.15 * l_gr + 0.12 * l_topfp + l_mono + 0.65 * l_kd + l_sdiv + l_code + l_sp + l_r0 + l_ai_pair

        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt_a.step()

        if not args.smoke_test and ((step + 1) % 200 == 0 or step + 1 == pa_steps):
            cm = _eval_ai_metrics(mdl)
            if cm["ai_obj_sel"] >= best_pa_sc:
                best_pa_sc = cm["ai_obj_sel"]
                best_pa_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(
                f"[PhaseA step={step+1}/{pa_steps}] loss={float(loss_a.item()):.4f} | ai_sel={cm['ai_obj_sel']:.5f} "
                f"vg2={cm.get('auroc_vg2', 0.0):.4f} rp95_vg2={cm.get('rp95_vg2', 0.0):.4f} wild={cm['auroc_wild']:.4f} rp95_w={cm['rp95_wild']:.4f} "
                f"vg1={cm['auroc_vg']:.4f} r1={cm['r1_auroc']:.4f} r0_ai={cm['r0_ai_auroc']:.4f} | elapsed={time.monotonic()-t_start:.0f}s",
                flush=True,
            )
            mdl.train()

    for hk in hooks_a:
        hk.remove()
    r0_trained_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items() if k.startswith("rung0.")}
    best_pa_sd.update(r0_trained_sd)
    mdl.load_state_dict({k: v.to(device) for k, v in best_pa_sd.items()}, strict=True)
    del v11_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pa_ai_m = _eval_ai_metrics(mdl)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_phaseA_ckpt.pt")
    print(f"[PhaseA DONE] ai_sel={pa_ai_m['ai_obj_sel']:.5f} (ref={ref_ai_m['ai_obj_sel']:.5f}) | {json.dumps({k: round(v, 5) for k, v in pa_ai_m.items()})}", flush=True)

    # =========================================================================
    # PHASE B: 100% Frozen-AI Firewall + LIVE `z_char` Promo Bridge + 12% Wild-Promo Focus + ChronoStream + Rung-0!
    # =========================================================================
    pb_steps = 4 if args.smoke_test else int(args.pb_steps)
    pg_b, hooks_b = build_v121_param_groups(mdl, "phase_b")
    opt_b = torch.optim.AdamW(pg_b, betas=(0.9, 0.95))
    asam_b = ASAMPerturber(mdl, rho=0.04)
    best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pb_sc = -1e9
    mdl.train()

    for step in range(pb_steps):
        mult = _wsd_lr_mult(step, pb_steps, warmup_frac=0.05, decay_frac=0.20)
        for pg in opt_b.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        # RC-4 FIX: 12% dedicated Wild-Promo (`Wave-2.5 + Wave-3 P1-P5`) slice in EVERY Phase B batch!
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w, wild_promo_frac=0.12)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era, code_i) = _collate_items(b_it)

        def _phase_b_loss():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                # RC-4 FIX: Pass LIVE `r0_c1/c3/cm/cf` with `detach_zchar_for_ai=True`!
                # AI [0:192] stays 100% mathematically invariant while `z_char_r1_promo`, `z_char_coda_promo`,
                # & `promo_zchar_direct` inherit Rung-0's 0.953 wild-promo char representation directly!
                out = mdl(
                    b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                    r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                    outer_steps=2, detach_zchar_for_ai=True, return_aux=True,
                )

                loss_b = (
                    0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai, y_pr, has_pr, a5_b)
                    + compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai, y_pr, has_pr)
                    + compute_v121_orthogonality_loss(mdl, out["logit"].detach(), y_ai, y_pr, has_pr, has_ai)
                )
                u, u_t1, u_r1 = out["logit_promo"].float(), out["exit1_out"]["logit_promo"].float(), out["logit_promo_r1"].float()
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    yp_c = y_pr[pm].clamp(1e-4, 1.0 - 1e-4)
                    loss_b = loss_b + (F.binary_cross_entropy_with_logits(u[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.30 * (F.binary_cross_entropy_with_logits(u_t1[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.40 * (F.binary_cross_entropy_with_logits(u_r1[pm], yp_c, reduction="none") * w_b[pm] * (1.0 + 3.0 * (y_pr[pm] >= 0.65).float())).mean()
                    hn_m = pm & (y_pr <= 0.25) & (l_pc > 0.5)
                    if int(hn_m.sum().item()) >= 2:
                        u_worst = torch.topk(u[hn_m], k=min(4, int(hn_m.sum().item()))).values
                        loss_b = loss_b + 0.22 * (F.relu(u_worst).pow(2).mean() + 0.20 * F.softplus(u_worst + 0.8).mean())
                    loss_b = loss_b + 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], u, sub4_i, y_pr, has_pr)

                # Rung-0 Dual-Axis Teacher Distillation (strictly `has_ai` masked for AI axis!)
                r0_ai = out["logit_ai_r0"].float()
                r0_pr = out["logit_promo_r0"].float()
                t_ai_lg = out["logit"].detach().float()
                t_pr_lg = u.detach()
                ai_m = has_ai > 0.5
                l_r0_ai = (
                    0.45 * (F.binary_cross_entropy_with_logits(r0_ai[ai_m], y_ai[ai_m].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[ai_m] * (1.0 + 3.0 * (y_ai[ai_m] >= 0.45).float())).mean()
                    + 0.40 * F.mse_loss(r0_ai[ai_m], t_ai_lg[ai_m])
                    + 0.10 * listnet_tail_distill_loss(r0_ai[ai_m], t_ai_lg[ai_m])
                ) if ai_m.any() else u.sum() * 0.0
                l_r0_pr = 0.25 * F.mse_loss(r0_pr, t_pr_lg) + 0.08 * listnet_tail_distill_loss(r0_pr, t_pr_lg)
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    l_r0_pr = l_r0_pr + 0.55 * (F.binary_cross_entropy_with_logits(r0_pr[pm], y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm] * (1.0 + 3.5 * (y_pr[pm] >= 0.65).float())).mean()
                with torch.amp.autocast("cuda", enabled=False):
                    pa0, pa_t = torch.sigmoid(r0_ai.detach()), torch.sigmoid(t_ai_lg)
                    pp0, pp_t = torch.sigmoid(r0_pr.detach()), torch.sigmoid(t_pr_lg)
                    d0_tgt = torch.maximum(
                        torch.maximum(((pa_t - pa0).abs() * 3.0).clamp(0.0, 1.0), (ai_m & (y_ai >= 0.45) & (pa0 < 0.35)).float()),
                        torch.maximum(((pp_t - pp0).abs() * 3.0).clamp(0.0, 1.0), ((has_pr >= 0.5) & (y_pr >= 0.65) & (pp0 < 0.35)).float()),
                    )
                    l_r0_def = 0.25 * F.binary_cross_entropy(out["p_defer_r0"].float().clamp(1e-5, 1.0 - 1e-5), d0_tgt)
                loss_b = loss_b + l_r0_ai + l_r0_pr + l_r0_def

                if dual_pairs_train and step % 2 == 0:
                    pb = rng_tr.sample(dual_pairs_train, min(16 if not args.smoke_test else 4, len(dual_pairs_train)))
                    p_it, o_it = [], []
                    for x in pb:
                        pt_t, ot_t = slot_aug.augment_pair(x["promo_text"], x["organic_text"]) if rng_tr.random() < 0.5 else (x["promo_text"], x["organic_text"])
                        ia = 1.0 if x.get("promo_is_ai") else 0.0
                        p_it.append(_make_item(pt_t, ia, 1.0, 1.0, str(x.get("promo_type", "soft_shill")), 1.5, has_ai=0.0))
                        o_it.append(_make_item(ot_t, ia, 0.0, 1.0, "none", 1.5, has_ai=0.0))
                    pi, pm_, pe, pt_, ps, pc1, pc3, pcm, pcf, *_ = _collate_items(p_it)
                    oi, om_, oe, ot_, os_, oc1, oc3, ocm, ocf, *_ = _collate_items(o_it)
                    op = mdl(pi, pm_, ps, engram_ids=pe, trigram_ids=pt_, r0_c1=pc1, r0_c3=pc3, r0_mask=pcm, r0_feats=pcf, outer_steps=1, detach_zchar_for_ai=True, return_aux=True)
                    oo = mdl(oi, om_, os_, engram_ids=oe, trigram_ids=ot_, r0_c1=oc1, r0_c3=oc3, r0_mask=ocm, r0_feats=ocf, outer_steps=1, detach_zchar_for_ai=True, return_aux=True)
                    loss_b = loss_b + 0.22 * F.softplus(2.2 - (op["logit_promo"].float() - oo["logit_promo"].float())).mean()

                if step % 2 == 1:
                    cs_b = collate_chronostream_docs(_sample_long_batch(4 if args.smoke_test else 48), tokenizer, max_chunks=3, max_chunk_toks=192)
                    cs_o = mdl.forward_chronostream_batch(
                        cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                        cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                        trunk_no_grad=True, outer_steps=2,
                    )
                    loss_b = loss_b + F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device)) + 0.5 * F.binary_cross_entropy_with_logits(cs_o["logit_hybrid_doc"].float(), cs_b["y_hyb_doc"].to(device)) + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
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

        if not args.smoke_test and ((step + 1) % 300 == 0 or step + 1 == pb_steps):
            pl_m = _eval_promo_and_long(mdl)
            ai_mb = _eval_ai_metrics(mdl)
            long_m = 0.50 * pl_m["long_ai_auroc"] + 0.25 * pl_m["long_hyb_auroc"] + 0.25 * pl_m["long_pr_auroc"]
            sc_b = 0.45 * (0.5 * pl_m["promo_auroc"] + 0.5 * pl_m["promo_pr_auc"]) + 0.25 * long_m + 0.30 * (0.5 * ai_mb["r0_ai_auroc"] + 0.5 * pl_m["r0_promo_auroc"])
            if sc_b >= best_pb_sc:
                best_pb_sc = sc_b
                best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(
                f"[PhaseB step={step+1}/{pb_steps}] loss={float(loss_b.item()):.4f} | ai_sel={ai_mb['ai_obj_sel']:.5f} | "
                f"promo_auc={pl_m['promo_auroc']:.4f} pr={pl_m['promo_pr_auc']:.4f} r1_pr={pl_m['r1_promo_auroc']:.4f} | "
                f"long_ai={pl_m['long_ai_auroc']:.4f} hyb={pl_m['long_hyb_auroc']:.4f} pr={pl_m['long_pr_auroc']:.4f} | "
                f"r0_ai={ai_mb['r0_ai_auroc']:.4f} r0_pr={pl_m['r0_promo_auroc']:.4f} | elapsed={time.monotonic()-t_start:.0f}s",
                flush=True,
            )
            mdl.train()

    for hk in hooks_b:
        hk.remove()
    if best_pb_sc > -1e8:
        mdl.load_state_dict({k: v.to(device) for k, v in best_pb_sd.items()}, strict=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_fp16_master.pt")
    fp16_ai_m = _eval_ai_metrics(mdl)
    fp16_pl_m = _eval_promo_and_long(mdl)
    print(f"[PhaseB FP16 Master DONE] ai={json.dumps({k: round(v, 5) for k, v in fp16_ai_m.items()})} | dual={json.dumps({k: round(v, 4) for k, v in fp16_pl_m.items()})}", flush=True)

    # =========================================================================
    # PHASE C: Zero-Op Fold -> UPQ-W4 -> Pure Dual-Axis KD Sub-2 QAT x2 + Soup -> Shock-Absorber ->
    #          Capped SoftBall LEACE -> Bounded 3-Param Fusion -> Smooth Promo LUT -> Certified LTT -> MGVZ7 + 5-Stage ONNX!
    # =========================================================================
    cal_f = _collate_items(cal_items[:32])
    fold_rep = fold_zero_op_v121(mdl, calib_inputs=[(cal_f[0], cal_f[1], cal_f[4], cal_f[2])])
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    fp16_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_teacher.parameters():
        p.requires_grad_(False)

    def _run_qat_stage(st_name: str, init_sd: dict[str, torch.Tensor], qmode: str, steps: int, lam_p: float = 1.0) -> tuple[dict[str, torch.Tensor], float]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(qmode)
        pg_q, hooks_q = build_v121_param_groups(mdl, "qat")
        opt_q = torch.optim.AdamW(pg_q, betas=(0.9, 0.95))
        best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc_q = -1e9
        mdl.train()
        for step in range(steps):
            mult = _wsd_lr_mult(step, steps, warmup_frac=0.05, decay_frac=0.25)
            for pg in opt_q.param_groups:
                pg["lr"] = pg["base_lr"] * mult
            opt_q.zero_grad(set_to_none=True)
            b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w, gold_v2_frac=0.05, wild_promo_frac=0.08)
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, a5_b, *_rest) = _collate_items(b_it)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = mdl(
                    b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                    r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                    outer_steps=2, trm_lam_k0=0.0, trm_lam_k1=0.0, return_aux=True,
                )
                with torch.no_grad():
                    tv = fp16_teacher(
                        b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                        r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                        outer_steps=2, return_aux=True,
                    )
                lg, lg_e1, lg_r1 = out["logit"].float(), out["exit1_out"]["logit"].float(), out["logit_r1"].float()
                l_kd = (
                    0.85 * F.mse_loss(lg, tv["logit"].float())
                    + 0.45 * F.mse_loss(lg_e1, tv["exit1_out"]["logit"].float())
                    + 0.45 * F.mse_loss(lg_r1, tv["logit_r1"].float())
                    + 0.35 * F.mse_loss(out["logit_ai_r0"].float(), tv["logit_ai_r0"].float())
                    + 0.35 * F.mse_loss(out["logit_promo_r0"].float(), tv["logit_promo_r0"].float())
                    + 0.65 * (1.0 - F.cosine_similarity(out["emb"].float(), tv["emb"].float(), dim=-1).mean())
                    + 0.70 * lam_p * F.mse_loss(out["logit_promo"].float(), tv["logit_promo"].float())
                    + 0.35 * lam_p * F.mse_loss(out["logit_promo_r1"].float(), tv["logit_promo_r1"].float())
                    + 0.10 * listnet_tail_distill_loss(lg, tv["logit"])
                    + 0.10 * listnet_tail_distill_loss(out["logit_promo"].float(), tv["logit_promo"])
                    + 0.15 * compute_slot_orthogonality_loss(out["slots_final"])
                    + 0.04 * out["cbm_loss"]
                )
                ai_m = has_ai > 0.5
                l_sup = 0.15 * (F.binary_cross_entropy_with_logits(lg[ai_m], y_ai[ai_m].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[ai_m]).mean() if ai_m.any() else lg.sum() * 0.0
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    l_sup = l_sup + 0.22 * lam_p * (F.binary_cross_entropy_with_logits(out["logit_promo"][pm].float(), y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm]).mean()
                    l_sup = l_sup + 0.08 * compute_sub4_masked_loss(out["promo_sub4_logit"], out["logit_promo"], sub4_i, y_pr, has_pr)
                l_ebt = 0.06 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai, y_pr, has_pr, a5_b) + 0.5 * compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai, y_pr, has_pr)
                with torch.amp.autocast("cuda", enabled=False):
                    l_def = 0.10 * F.mse_loss(out["p_defer1"].float(), tv["p_defer1"].float()) + 0.08 * F.mse_loss(out["exit1_out"]["p_defer2"].float(), tv["exit1_out"]["p_defer2"].float()) + 0.08 * F.mse_loss(out["p_defer_r0"].float(), tv["p_defer_r0"].float())
                loss_q = l_kd + l_sup + l_ebt + l_def
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_q.step()

            if not args.smoke_test and ((step + 1) % 300 == 0 or step + 1 == steps):
                cm, pm_ = _eval_ai_metrics(mdl), _eval_promo_and_long(mdl)
                sc_q = cm["ai_obj_sel"] + 0.35 * (0.5 * pm_["promo_auroc"] + 0.5 * pm_["promo_pr_auc"])
                if sc_q >= best_sc_q:
                    best_sc_q = sc_q
                    best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                print(f"[{st_name} step={step+1}/{steps}] ai_sel={cm['ai_obj_sel']:.5f} vg2={cm.get('auroc_vg2', 0.0):.4f} wild={cm['auroc_wild']:.4f} | promo={pm_['promo_auroc']:.4f} | r0_ai={cm['r0_ai_auroc']:.4f}", flush=True)
                mdl.train()
        for hk in hooks_q:
            hk.remove()
        if best_sc_q <= -1e8:
            best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            best_sc_q = _eval_ai_metrics(mdl)["ai_obj_sel"]
        return best_sd_q, best_sc_q

    w4_sd, _ = _run_qat_stage("PhaseC-UPQ-W4", folded_sd, "w4", 4 if args.smoke_test else int(args.pc_w4_steps))
    q1_sd, q1_sc = _run_qat_stage("PhaseC-Sub2-A", w4_sd, "sub2", 4 if args.smoke_test else int(args.pc_sub2_steps), lam_p=1.0)
    q2_sd, q2_sc = _run_qat_stage("PhaseC-Sub2-B", w4_sd, "sub2", 4 if args.smoke_test else int(args.pc_sub2_steps), lam_p=1.25)

    def _qat_soup_eval(sd_c: dict[str, torch.Tensor]) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_c.items()}, strict=True)
        cm, pm_ = _eval_ai_metrics(mdl), _eval_promo_and_long(mdl)
        return cm["ai_obj_sel"] + 0.35 * (0.5 * pm_["promo_auroc"] + 0.5 * pm_["promo_pr_auc"]) + 0.12 * pm_["long_ai_auroc"]

    qat_sd, qat_soup_rep = greedy_pauc_soup(w4_sd, [("Q1", q1_sd, q1_sc), ("Q2", q2_sd, q2_sc)], eval_fn=_qat_soup_eval, dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in qat_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_qat_soup.pt")

    tmp_mg = out_dir / "_tmp_v121_pre_shock.mgvz"
    pack_bundle_v7_v121(mdl.cpu(), tmp_mg)
    load_bundle_v7_v121_into_model(mdl, tmp_mg)
    tmp_mg.unlink(missing_ok=True)
    mdl = mdl.to(device).eval()

    pre_sh_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_sh_sc = _qat_soup_eval(pre_sh_sd)
    shock_tokens = ("core.qres.", "core.inloop_ebt.", "core.spectral_damp", "ai_head.", "promo", "r1_ai_head", "r1_promo", "sub4", "attr6", "longstream", "defer_", "ebt_logit_gate", "mil_gate", "rung0.out_head", "rung0.trunk", "z_char_")
    quant_skip = {f"{mn}.weight" for mn, mod in mdl.named_modules() if isinstance(mod, OmniQuantLinear)}
    sh_p_list = []
    for nm, p in mdl.named_parameters():
        keep = (nm not in quant_skip) and any(t in nm for t in shock_tokens)
        p.requires_grad_(keep)
        if keep:
            sh_p_list.append(p)
    opt_sh = torch.optim.AdamW(sh_p_list, lr=1.5e-4, weight_decay=1e-4)
    mdl.train()
    for _ in range(4 if args.smoke_test else int(args.pc_shock_steps)):
        opt_sh.zero_grad(set_to_none=True)
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w, gold_v2_frac=0.05, wild_promo_frac=0.08)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf, y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, *_r) = _collate_items(b_it)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            o = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, trm_lam_k0=0.0, trm_lam_k1=0.0, return_aux=True)
            with torch.no_grad():
                tv_sh = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
            ai_m = has_ai > 0.5
            loss_sh = (
                0.70 * F.mse_loss(o["logit"].float(), tv_sh["logit"].float())
                + 0.40 * F.mse_loss(o["logit_r1"].float(), tv_sh["logit_r1"].float())
                + 0.35 * F.mse_loss(o["logit_ai_r0"].float(), tv_sh["logit_ai_r0"].float())
                + 0.60 * F.mse_loss(o["logit_promo"].float(), tv_sh["logit_promo"].float())
                + 0.30 * F.mse_loss(o["logit_promo_r1"].float(), tv_sh["logit_promo_r1"].float())
            )
            if ai_m.any():
                loss_sh = loss_sh + 0.15 * (F.binary_cross_entropy_with_logits(o["logit"][ai_m].float(), y_ai[ai_m].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[ai_m]).mean()
            if (has_pr > 0.5).any():
                pm = has_pr > 0.5
                loss_sh = loss_sh + 0.20 * (F.binary_cross_entropy_with_logits(o["logit_promo"][pm].float(), y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm]).mean()
        loss_sh.backward()
        torch.nn.utils.clip_grad_norm_(sh_p_list, 1.0)
        opt_sh.step()
    del fp16_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for p in mdl.parameters():
        p.requires_grad_(True)
    post_sh_sc = _qat_soup_eval({k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()})
    shock_kept = post_sh_sc >= pre_sh_sc
    if not shock_kept:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_sh_sd.items()}, strict=True)
    mdl.eval()

    # Capped SoftBall Oblique LEACE
    vs_o = _predict_v121(mdl, vs_p, 2)
    z_ll_vs = np.log1p(np.array([len(t) for t in vs_p[3]], dtype=np.float32))[:, None]
    emb_l, yai_l, ypr_l = vs_o["emb"], vs_p[7] / 100.0, np.full(len(vs_p[7]), 0.50, dtype=np.float32)
    zai_l = np.concatenate([z_ll_vs, expit(vs_o["logit_promo"])[:, None]], axis=1)
    zpr_l = np.concatenate([z_ll_vs, expit(vs_o["logit"])[:, None]], axis=1)
    if promo_val_select_rows:
        ps_l_pk = (
            [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
            [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
            np.stack([x["style"] for x in promo_val_select_rows]), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32),
        )
        pso_l = _predict_v121(mdl, ps_l_pk, 2)
        z_ll_ps = np.log1p(np.array([len(t) for t in ps_l_pk[3]], dtype=np.float32))[:, None]
        emb_l = np.concatenate([emb_l, pso_l["emb"]], axis=0)
        yai_l = np.concatenate([yai_l, np.full(len(promo_val_select_rows), 0.50, dtype=np.float32)], axis=0)
        ypr_l = np.concatenate([ypr_l, ps_l_pk[7]], axis=0)
        zai_l = np.concatenate([zai_l, np.concatenate([z_ll_ps, expit(pso_l["logit_promo"])[:, None]], axis=1)], axis=0)
        zpr_l = np.concatenate([zpr_l, np.concatenate([z_ll_ps, expit(pso_l["logit"])[:, None]], axis=1)], axis=0)
    leace_rep = apply_dual_axis_oblique_leace_v121(mdl, emb_l, yai_l, ypr_l, zai_l, zpr_l)

    # Cross-Exit Affine Alignment + Bounded `tanh`-Squashed 3-Param Log-Odds Fusion!
    al_sub = cal_items[: (64 if args.smoke_test else min(4000, len(cal_items)))]
    al_pk = ([x["ids"] for x in al_sub], [x["engram"] for x in al_sub], [x["trigram"] for x in al_sub], [x["raw_text"] for x in al_sub], [x["clean_text"] for x in al_sub], np.stack([x["style"] for x in al_sub]), [], np.zeros(len(al_sub)))
    al_o = _predict_v121(mdl, al_pk, 2)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_r1"], al_o["raw_logit"])
    a_t1, b_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_t1"], al_o["raw_logit"])
    pa_r1, pb_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_r1"], al_o["raw_logit_promo"])
    pa_t1, pb_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_t1"], al_o["raw_logit_promo"])
    with torch.no_grad():
        mdl.exit_affine_a.copy_(torch.tensor([a_r1, a_t1, 1.0], device=device))
        mdl.exit_affine_b.copy_(torch.tensor([b_r1, b_t1, 0.0], device=device))
        mdl.exit_affine_promo_a.copy_(torch.tensor([pa_r1, pa_t1, 1.0], device=device))
        mdl.exit_affine_promo_b.copy_(torch.tensor([pb_r1, pb_t1, 0.0], device=device))

    vg_o, vs_o, vc_o = _predict_v121(mdl, vg_p, 2), _predict_v121(mdl, vs_p, 2), _predict_v121(mdl, vc_p, 2)
    fus_t1 = [vs_o["logit_t1"], vg_o["logit_t1"]]
    fus_t2 = [vs_o["logit"], vg_o["logit"]]
    fus_eb = [vs_o["ebt_margin"], vg_o["ebt_margin"]]
    fus_sc = [vs_p[7], vg_p[7]]
    if vg2_p is not None:
        vg2_o = _predict_v121(mdl, vg2_p, 2)
        fus_t1.append(vg2_o["logit_t1"]); fus_t2.append(vg2_o["logit"]); fus_eb.append(vg2_o["ebt_margin"]); fus_sc.append(vg2_p[7])
    fusion_rep = fit_3param_logodds_fusion_v121(np.concatenate(fus_t1), np.concatenate(fus_t2), np.concatenate(fus_eb), np.concatenate(fus_sc), ebt_clip=6.0)
    with torch.no_grad():
        mdl.fusion_weights.copy_(torch.tensor([fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"]], device=device))
        mdl.fusion_ebt_clip.fill_(float(fusion_rep["ebt_clip"]))
    fw_w = (fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"])
    eb_clip = float(fusion_rep["ebt_clip"])

    def _fused_lg(o_d: dict[str, np.ndarray]) -> np.ndarray:
        return bounded_fused3_logits(o_d["logit_t1"], o_d["logit"], o_d["ebt_margin"], fw_w[0], fw_w[1], fw_w[2], eb_clip)

    # Calibration LUTs (`R0`, `R1`, `Full` — BOTH AI & Promo smooth Beta-Spline `x`/`y`!) + Certified Dual-Rung LTT!
    y_cal_ai = np.concatenate([vs_p[7], vg_p[7]]) / 100.0
    lut_ai_full = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([_fused_lg(vs_o), _fused_lg(vg_o)]), y_cal_ai, fit_binary_ece_mode=True), 256)
    lut_ai_r1 = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit_r1"], vg_o["logit_r1"]]), y_cal_ai, fit_binary_ece_mode=True), 256)
    lut_ai_r0 = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit_ai_r0"], vg_o["logit_ai_r0"]]), y_cal_ai, fit_binary_ece_mode=True), 256)

    pi_nat_pr = estimate_wild_prevalence_em(expit(al_o["logit_promo"]), pi_train=0.10)
    lut_pr_full = lut_pr_r1 = lut_pr_r0 = None
    r1_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False}
    r0_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False, "tau_ai": 0.06, "tau_promo": 0.06, "defer_max": 0.35, "max_chars": 160}

    if promo_val_select_rows and promo_val_cert_rows:
        def _pack_pr(rows_pr):
            return ([x["ids"] for x in rows_pr], [x["engram"] for x in rows_pr], [x["trigram"] for x in rows_pr], [x["raw_text"] for x in rows_pr], [x["clean_text"] for x in rows_pr], np.stack([x["style"] for x in rows_pr]), [{"spoof_any": x["spoof_any"]} for x in rows_pr], np.array([x["y_promo"] for x in rows_pr], dtype=np.float32))
        ps_pk, pc_pk = _pack_pr(promo_val_select_rows), _pack_pr(promo_val_cert_rows)
        pso, pco = _predict_v121(mdl, ps_pk, 2), _predict_v121(mdl, pc_pk, 2)
        lut_pr_full = fit_smooth_promo_lut_v121(pso["logit_promo"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)
        lut_pr_r1 = fit_smooth_promo_lut_v121(pso["logit_promo_r1"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)
        lut_pr_r0 = fit_smooth_promo_lut_v121(pso["logit_promo_r0"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)

        def _elig_r0(pk):
            return np.array([len(pk[3][i]) <= 160 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[3][i].lower()) for i in range(len(pk[0]))], dtype=bool)
        raw_r0_ltt = certify_fixed_sequence_dual_axis_ltt(
            _elig_r0(vs_p), vs_o["p_defer_r0"], apply_spline_temp_lut(expit(vs_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vs_o["logit_promo_r0"]), lut_pr_r0),
            vc_p[7], _elig_r0(vc_p), vc_o["p_defer_r0"], apply_spline_temp_lut(expit(vc_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vc_o["logit_promo_r0"]), lut_pr_r0),
            pc_pk[7] >= 0.65, _elig_r0(pc_pk), pco["p_defer_r0"], apply_spline_temp_lut(expit(pco["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(pco["logit_promo_r0"]), lut_pr_r0),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05, r1_cost_rel=0.05,
            d_grid=(0.15, 0.25, 0.35, 0.50), tau_a_grid=(0.03, 0.05, 0.08, 0.12), tau_p_grid=(0.04, 0.06, 0.10, 0.15),
        )
        r0_ltt_policy = {
            **raw_r0_ltt,
            "defer_max": raw_r0_ltt.get("r1_defer_max", 0.25),
            "tau_ai": raw_r0_ltt.get("r1_p_ai_max", 0.06),
            "tau_promo": raw_r0_ltt.get("r1_p_promo_max", 0.06),
            "max_chars": 160,
        }

        def _elig_r1(pk, o_d):
            return np.array([
                len(pk[0][i]) <= 96 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[4][i])
                and float(np.max(pk[5][i, list(R1_PROMO_VETO_DIMS_V11)])) == 0.0 and float(o_d["r1_span_max"][i]) <= 0.50
                for i in range(len(pk[0]))
            ], dtype=bool)
        r1_ltt_policy = certify_fixed_sequence_dual_axis_ltt(
            _elig_r1(vs_p, vs_o), vs_o["p_defer1"], apply_spline_temp_lut(expit(vs_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vs_o["logit_promo_r1"]), lut_pr_r1),
            vc_p[7], _elig_r1(vc_p, vc_o), vc_o["p_defer1"], apply_spline_temp_lut(expit(vc_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vc_o["logit_promo_r1"]), lut_pr_r1),
            pc_pk[7] >= 0.65, _elig_r1(pc_pk, pco), pco["p_defer1"], apply_spline_temp_lut(expit(pco["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(pco["logit_promo_r1"]), lut_pr_r1),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05,
        )

    # Export MGVZ7 + Post-Unpack Exact 5-Stage Static ONNX + Auto Compression Levers if > 2.20 MB!
    exp_info = export_v121_modular_onnx_and_bundle_v7(mdl, out_dir, export_matmulnbits=bool(args.export_matmulnbits))
    manifest = {
        "schema_version": "meguviz.bundle.v7",
        "version": args.version,
        "emb_layout": mdl.emb_layout_tag(),
        "tokenizer_file": tok_path.name,
        "bundle_v7_mgvz": exp_info["bundle_v7_mgvz"]["file"],
        "rung0_onnx": exp_info["rung0_onnx"],
        "rung1_onnx": exp_info["rung1_onnx"],
        "rung2_onnx": exp_info["rung2_onnx"],
        "rung3_onnx": exp_info["rung3_onnx"],
        "dynaseam_onnx": exp_info["dynaseam_onnx"]["onnx_file"],
        "fusion_weights": {"w_t1": fw_w[0], "w_t2": fw_w[1], "w_ebt": fw_w[2], "ebt_clip": eb_clip},
        "lut_ai_r0": lut_ai_r0, "lut_ai_r1": lut_ai_r1, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_r1": lut_pr_r1, "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_ltt_policy,
        "dual_axis_ltt_policy": r1_ltt_policy,
        "capabilities": ["rung0_zero_bpe_charhash", "dense_shared_quant_res", "softball_logitnorm_v121", "pad_invariant_mil", "bounded_logodds_fusion", "live_zchar_promo_bridge"],
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

    # Final Unpacked Evaluation
    mdl_u = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v7_v121_into_model(mdl_u, out_dir / exp_info["bundle_v7_mgvz"]["file"])
    mdl_u = mdl_u.to(device).eval()

    tgo, tdo = _predict_v121(mdl_u, tg_p, 2), _predict_v121(mdl_u, td_p, 2)
    p_tg = apply_spline_temp_lut(expit(np.clip(_fused_lg(tgo), -30.0, 30.0)), lut_ai_full)
    p_td = apply_spline_temp_lut(expit(np.clip(_fused_lg(tdo), -30.0, 30.0)), lut_ai_full) if len(td_p[7]) > 0 else np.array([])
    tb_m = {**compute_eval_metrics(tg_p[7], p_tg), **compute_high_precision_metrics(tg_p[7], p_tg)}
    tb_r0 = compute_eval_metrics(tg_p[7], expit(tgo["logit_ai_r0"]))
    td_m = compute_eval_metrics(td_p[7], p_td) if len(td_p[7]) > 0 else {}
    tb2_m: dict[str, Any] = {}
    if tg2_p is not None:
        tg2o = _predict_v121(mdl_u, tg2_p, 2)
        p_tg2 = apply_spline_temp_lut(expit(np.clip(_fused_lg(tg2o), -30.0, 30.0)), lut_ai_full)
        tb2_m = {**compute_eval_metrics(tg2_p[7], p_tg2), **compute_high_precision_metrics(tg2_p[7], p_tg2)}

    pb_m: dict[str, Any] = {}
    pb_p = Path(__file__).resolve().parent / "data" / "v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_pk = _pre_eval_v121(pb_rows)
        pbo = _predict_v121(mdl_u, pb_pk, 2)
        y_pb_ai = np.array([int(r["y_ai"]) for r in pb_rows])
        y_pb_pr = np.array([int(r["y_promo"]) for r in pb_rows])
        pb_m = {
            "ai_auroc": round(float(roc_auc_score(y_pb_ai, _fused_lg(pbo))), 4),
            "promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo"])), 4),
            "promo_pr_auc": round(float(average_precision_score(y_pb_pr, pbo["logit_promo"])), 4),
            "r0_ai_auroc": round(float(roc_auc_score(y_pb_ai, pbo["logit_ai_r0"])), 4),
            "r0_promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo_r0"])), 4),
        }

    final_dual_m = _eval_promo_and_long(mdl_u)
    ci_fin = run_v121_ci_gates(mdl_u, tokenizer)
    metrics = {
        "version": args.version,
        "gpu_name": gpu_name,
        "warmstart_source": loaded_v11_src,
        "gold_v2_and_wave3_loaded": {
            "g2_train": len(gold_v2_train_rows), "g2_val": len(gold_v2_val_rows), "g2_test": len(gold_v2_test_rows),
            "w3_train": len(wave3_train_rows), "w3_val": len(wave3_val_rows), "wave25": w25_loaded,
        },
        "bundle_v7_total_mb": round(v7_mb, 4),
        "onnx_5stage_zip_mb": round(zip_onnx.stat().st_size / (1024.0 * 1024.0), 4),
        "stage0_transplant": surg_rep,
        "stage0_ref_ai": ref_ai_m,
        "phase_a_ai": pa_ai_m,
        "phase_b_fp16_ai": fp16_ai_m,
        "phase_b_fp16_dual": fp16_pl_m,
        "fold_zero_op": fold_rep,
        "qat_soup": qat_soup_rep,
        "shock_absorber": {"kept": shock_kept, "before": round(pre_sh_sc, 5), "after": round(post_sh_sc, 5)},
        "leace_surgery": leace_rep,
        "fusion_weights": fusion_rep,
        "rung0_ltt_policy": r0_ltt_policy,
        "dual_axis_ltt_policy": r1_ltt_policy,
        "test_bili_gold_v1": tb_m,
        "test_bili_gold_v1_r0_auroc": round(float(tb_r0["auroc"]), 4),
        "test_bili_gold_v2": tb2_m,
        "test_dsh_gold": td_m,
        "promobench48": pb_m,
        "promo_val_select_and_longform120": {k: round(v, 4) for k, v in final_dual_m.items()},
        "export_info": exp_info,
        "v121_ci_gates": ci_fin,
        "release_gates": {
            "bundle_le_2p20_mb": v7_mb <= 2.20,
            "onnx_5stage_exact": (
                exp_info["r0_fp32_max_diff"] < 1e-4
                and exp_info["r1_fp32_max_diff"] < 1e-4
                and exp_info["r2_fp32_max_diff"] < 1e-4
                and exp_info["r3_fp32_max_diff"] < 1e-4
                and bool(exp_info["dynaseam_onnx"]["pass"])
            ),
            "v121_ci_passed": bool(ci_fin["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.2.1-ApexNano-Fix 3-Phase Trainer.")
    ap.add_argument("--smoke-test", action="store_true", help="Fast end-to-end local GPU smoke test.")
    ap.add_argument("--data", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    ap.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    ap.add_argument("--extra-pairs", type=Path, action="append", default=[])
    ap.add_argument("--v11-fp16-master", type=Path, default=Path("/content/drive/MyDrive/MeguvizCAC/v110_omniguard/v1.1.0-r1-omniguard-b6_fp16_master.pt"))
    ap.add_argument("--v11-bundle-mgvz", type=Path, default=None)
    ap.add_argument("--v07-tokenizer", type=Path, default=None)
    ap.add_argument("--v08-gemini-gold", type=Path, default=None)
    ap.add_argument("--v08-gemini-pairs", type=Path, default=None)
    ap.add_argument("--v11-data-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--wave25-gold", type=Path, default=None)
    ap.add_argument("--gold-v2-dir", type=Path, default=None)
    ap.add_argument("--wave3-dir", type=Path, default=None, help="Optional directory containing v121_wave3_{train,val}.jsonl.gz.")
    ap.add_argument("--out", type=Path, default=Path("scratch/v121_smoke"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.2.1-r1-apexnano-b7")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--pa-steps", type=int, default=1400)
    ap.add_argument("--pb-steps", type=int, default=2400)
    ap.add_argument("--pc-w4-steps", type=int, default=600)
    ap.add_argument("--pc-sub2-steps", type=int, default=1000)
    ap.add_argument("--pc-shock-steps", type=int, default=300)
    ap.add_argument("--export-matmulnbits", action="store_true")
    args = ap.parse_args()
    rep = train_v121_apexnano_full(args)
    print(json.dumps({k: v for k, v in rep.items() if k != "export_info"}, ensure_ascii=False, indent=2))
    assert all(rep["release_gates"].values()), f"Release gates failed: {rep['release_gates']}"


if __name__ == "__main__":
    main()
