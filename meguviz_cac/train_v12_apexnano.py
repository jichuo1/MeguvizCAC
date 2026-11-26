"""MeguvizCAC v1.2.0-ApexNano-Omni 3-Phase Zero-Regression Training Pipeline (Colab G4 Pro 6000 / A100 + Local `--smoke-test`).

Implements the finalized Dense-First `ADR-v1.2` blueprint (`docs/roadmap-v1.2.0-apexnano-omni.md`):
  - **Stage 0 (Zero-Shock Transplant & Gold-v2 / v1.2 Refinery)**:
    - Warm-starts from `v1.1.0-R1` (`fp16_master.pt` or `bundle_v6_omniguard.mgvz`) via exact 301-tensor dense copy
      + soft-MoLE -> `SharedQuantResV12` collapse + `CosineLogitNormHeadV12` scale init.
    - Applies `v12_data_refinery.py` (purges ~14.7k ultra-short generic plugin clichés & `v04_qwen_inkling_joint`;
      blocks `30..70` midband from `T0`), ingests `Wave-1 + Wave-2 + Wave-2.5 (243) + UOT + k-NN + Pairs`, and
      natively supports optional `Gold-v2` (`--gold-v2-dir`: overrides train rows into `T0`, evaluates parallel
      `val_gold_v2` & `test_bili_gold_v2` without ever overwriting frozen `v1`!).
  - **Phase A (Dense Pure-AI + `CosineLogitNorm` + Spectral Damping + `Rung-0` AI Warmup)**:
    - Restores full structural regularizers (`l_slot_div=0.15`, `attr6`, `code`, `prag`, `cbm`, `L_mono(T2>=T1)`)
      with selective `v1.1` teacher KD (`M_anc=0` on `Gold-v2` / `Wave-2.5` / `T0` teacher-wrong rows) and frozen
      Promo `[192:256]` slots (`grad[192:] = 0.0`).
  - **Phase B (100% Frozen-AI Firewall + Promo `[192:256]` + `ChronoStream` + `Rung-0` Dual-Axis Distillation)**:
    - Mathematically ZERO AI negative transfer: freezes AI dense trunk, masks `grad[:192] = 0.0` on `coda_proj.0` &
      `style_net.2`, trains Promo + `ChronoStream x DynaSeam-MIL` (with optional ASAM polish tail) AND distills the
      `105 KB` Zero-BPE `Rung0CharHashNanoV12` gatekeeper (Asymmetric FN-Weighted BCE + Dual-Axis Logit-MSE + ListNet!).
  - **Phase C (Pure Dual-Axis Teacher KD `W4 -> Sub-2-Bit` QAT + Shock-Absorber + Capped LEACE + 3-Param Fusion +
    Dual-Rung `R0 & R1` Fixed-Seq LTT + `MGVZ7 <= 2.20 MB` + Post-Unpack 5-Stage `FP32-Exact` ONNX)**:
    - Packs `Bundle v7 (<= 2.20 MB)`, reloads unpacked weights BEFORE exporting 5-Stage `FP32-Exact` ONNX (`<1e-4`),
      certifies strict 1-D Fixed-Sequence Dual-Axis LTT on BOTH `Rung-0` & `Rung-1`, and runs full `V12` CI gates!
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
from .train_v10_apexloop import (
    _ingest_external_gemini_gold_v10,
    _top_fp_logit_ceiling_loss,
)
from .v04_corpus_prep import GROUP_TO_IDX
from .v08_zenith_ops import ERA_TO_IDX
from .train_v11_omniguard import (
    DynamicPiKEGroupSamplerV11,
    collate_chronostream_docs,
    compute_dual_axis_ebt_margin_loss,
    compute_ebt6_partial_label_ce,
    compute_l2_sp_loss,
    compute_sub4_masked_loss,
)
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
from .v11_cascade_cert import (
    certify_fixed_sequence_dual_axis_ltt,
    estimate_wild_prevalence_em,
    fit_weighted_venn_abers_promo_lut,
)
from .v11_data_engine import OnlineLongMixSynthesizerV11, compute_joint_length_domain_ipw
from .v11_promo_pseudo import _norm, _sha_norm
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d
from .v11_omniguard import (
    AI_EMB_END_V11,
    R1_PROMO_VETO_DIMS_V11,
    SHARED_EMB_END_V11,
    MeguvizV11OmniGuardModel,
    load_bundle_v6_v11_into_model,
)
from .v12_apexnano import (
    MeguvizV12ApexNanoModel,
    apply_dual_axis_oblique_leace_v12,
    export_v12_modular_onnx_and_bundle_v7,
    extract_rung0_charhash_batch,
    fold_zero_op_v12,
    load_bundle_v7_v12_into_model,
    pack_bundle_v7_v12,
    run_v12_ci_gates,
    transplant_v11_to_v12_dense,
)
from .v12_data_refinery import classify_epistemic_tier_v12, run_v12_refinery_ci


# ---------------------------------------------------------------------------
# 1. Helper Losses, Parameter Groups & 3-Parameter Log-Odds Fusion Optimizer
# ---------------------------------------------------------------------------

def compute_slot_orthogonality_loss(slots_final: torch.Tensor) -> torch.Tensor:
    """Slow-slot cosine orthogonality across the 4 slow verdict slots (critical for T=2 stability!)."""
    zh = F.normalize(slots_final[:, :4, :].float(), p=2, dim=-1, eps=1e-6)
    gram = torch.bmm(zh, zh.transpose(1, 2))
    eye = torch.eye(4, device=zh.device, dtype=zh.dtype).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def compute_v12_orthogonality_loss(
    model: MeguvizV12ApexNanoModel,
    ai_logits: torch.Tensor,
    y_ai: torch.Tensor,
    y_promo: torch.Tensor,
    has_promo: torch.Tensor,
) -> torch.Tensor:
    """Unit-sphere shared-slice orthogonality between `ai_head` and `promo_head` + Q3 vs Q4 AI alignment."""
    w_ai_sh = model.ai_head.weight[0, :SHARED_EMB_END_V11].float()
    w_pr_sh = model.promo_head.weight[0, :SHARED_EMB_END_V11].float()
    cos_w = F.cosine_similarity(w_ai_sh.unsqueeze(0), w_pr_sh.unsqueeze(0), dim=-1).pow(2).mean()
    loss = 0.05 * cos_w
    q3_m = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo >= 0.65)
    q4_m = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo <= 0.25)
    if int(q3_m.sum().item()) >= 4 and int(q4_m.sum().item()) >= 4:
        loss = loss + 0.05 * (ai_logits[q3_m].float().mean() - ai_logits[q4_m].float().mean()).pow(2)
    return loss


def build_v12_param_groups(model: MeguvizV12ApexNanoModel, phase: str) -> tuple[list[dict[str, Any]], list[Any]]:
    """Phase-strict differential AdamW parameter groups + structural `[0:192]` / `[192:256]` gradient firewall hooks."""
    promo_only_keys = (
        "r1_promo_head", "promo_head", "promo_sub4_head", "promo_span_head", "promo_mil_gate",
        "ebt_promo_gate", "z_char_coda_promo", "fc2_new_", "attr6_new_",
    )
    long_keys = ("longstream.", "chunk_emb_proj.")
    rung0_keys = ("rung0.",)
    zchar_ai_keys = ("z_char_r1_", "z_char_coda_ai")
    fast_ai_head_keys = ("ai_head.", "core.qres.", "core.spectral_damp_logit", "r1_ai_head", "ebt_logit_gate", "mil_gate")
    half_trunk_keys = ("word_emb", "engram_", "trigram_", "pos_emb", "prelude_l0", "r1_rhythm_conv", "r1_attn_pool", "inloop_ebt.fc1", "inloop_ebt.fc2_old")
    emb_split_names = ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias")

    groups: dict[str, list[nn.Parameter]] = {
        "rung0": [], "promo_heads": [], "long": [], "zchar_ai": [],
        "fast_ai": [], "emb_split": [], "half_trunk": [], "trunk": [],
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
        elif any(k in name for k in half_trunk_keys):
            groups["half_trunk"].append(p)
        else:
            groups["trunk"].append(p)

    hooks: list[Any] = []
    if phase == "phase_a":
        # Firewalled Dense-Adapter Phase A (ZERO fresh-AdamW trunk shock on converged v1.1.0-R1!):
        # Freeze deep dense trunk (`word_emb/prelude/core.attn+ffn/slot/ebt_old`); train ONLY
        # `CosineLogitNormHeadV12` + `SharedQuantResV12` + `spectral_damp` + `coda[:192]` + `r1_ai_head` + `rung0`!
        lr_map = {
            "rung0": 1.8e-3, "promo_heads": 0.0, "long": 0.0, "zchar_ai": 0.0,
            "fast_ai": 7.0e-5, "emb_split": 6.0e-5, "half_trunk": 0.0, "trunk": 0.0,
        }

        def _mask_ai_rows_only(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            g[AI_EMB_END_V11:] = 0.0
            return g

        for p in groups["emb_split"]:
            hooks.append(p.register_hook(_mask_ai_rows_only))
    elif phase == "phase_b":
        # Phase B: AI dense trunk 100% FROZEN -> ZERO AI negative transfer by mathematical construction!
        # Trains ONLY Promo heads + emb[192:256] + ChronoStream + Rung-0 Dual-Axis Teacher Distillation!
        lr_map = {
            "rung0": 1.5e-3, "promo_heads": 1.0e-3, "long": 9.0e-4, "zchar_ai": 0.0,
            "fast_ai": 0.0, "emb_split": 1.0e-3, "half_trunk": 0.0, "trunk": 0.0,
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
            "rung0": 4.0e-4, "promo_heads": 2.0e-4, "long": 1.8e-4, "zchar_ai": 8.0e-5,
            "fast_ai": 1.2e-4, "emb_split": 5.0e-5, "half_trunk": 3.0e-5, "trunk": 5.0e-5,
        }

    out_pgs: list[dict[str, Any]] = []
    for gname, plist in groups.items():
        lr_g = lr_map[gname]
        for p in plist:
            p.requires_grad_(lr_g > 0.0)
        if plist and lr_g > 0.0:
            out_pgs.append({"name": gname, "params": plist, "lr": lr_g, "base_lr": lr_g, "weight_decay": 0.015})
    return out_pgs, hooks


def fit_3param_logodds_fusion(
    lg_t1: np.ndarray,
    lg_t2: np.ndarray,
    ebt_margin: np.ndarray,
    scores_0_100: np.ndarray,
) -> dict[str, float]:
    """Fit optimal 3-parameter simplex weights `[w_t1, w_t2, w_ebt]` on validation set (zero extra FLOPs!)."""
    from scipy.special import expit

    y = np.asarray(scores_0_100, dtype=np.float32)
    valid = (y <= 30.0) | (y >= 70.0)
    if valid.sum() < 20 or len(np.unique(y[valid] >= 70.0)) < 2:
        return {"w_t1": 0.50, "w_t2": 0.25, "w_ebt": 0.25, "val_obj": 0.0}

    t1, t2, eb, yv = lg_t1[valid], lg_t2[valid], ebt_margin[valid], y[valid]
    best = (0.50, 0.25, 0.25, -1.0)
    # Coarse-to-fine simplex grid (`step = 0.05`) with healthy 3-view balance (`w_t1>=0.20, w_t2>=0.15, w_ebt<=0.45`)
    for i in range(4, 18):
        w1 = i * 0.05
        for j in range(3, 21 - i):
            w2 = j * 0.05
            w3 = max(0.0, 1.0 - w1 - w2)
            if w3 > 0.45:
                continue
            z = w1 * t1 + w2 * t2 + w3 * eb
            p = expit(np.clip(z, -30.0, 30.0))
            m = {**compute_eval_metrics(yv, p), **compute_high_precision_metrics(yv, p)}
            obj = 0.35 * m["auroc"] + 0.35 * m["pr_auc"] + 0.30 * m["recall_at_p95"]
            if obj > best[3]:
                best = (w1, w2, w3, obj)
    return {
        "w_t1": round(best[0], 4),
        "w_t2": round(best[1], 4),
        "w_ebt": round(best[2], 4),
        "val_obj": round(best[3], 5),
    }


# ---------------------------------------------------------------------------
# 2. Main v1.2.0-ApexNano-Omni 3-Phase Training Pipeline
# ---------------------------------------------------------------------------

def train_v12_apexnano_full(args) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, roc_auc_score

    ref_ci = run_v12_refinery_ci()
    assert ref_ci["pass"], f"v12_data_refinery CI failed: {ref_ci}"

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
        f"[v1.2.0-ApexNano Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={bs} | "
        f"v11_master={args.v11_fp16_master} | v11_mgvz={args.v11_bundle_mgvz} | gold_v2={args.gold_v2_dir}",
        flush=True,
    )

    def _read_jsonl_list(p: Path) -> list[dict[str, Any]]:
        if not p.is_file():
            return []
        op = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")
        with op as f:
            return [json.loads(l) for l in f if l.strip()]

    # 1. Load base corpus + v08 80k Gemini Gold + legacy pairs (strict anti-leakage!)
    extra_pairs = [p.resolve() for p in args.extra_pairs]
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

    # 3. Hydrate true `score_source` + Gemini Pro/Flash provenance + Optional `Gold-v2` Train Override Map
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

    # Optional Gold-v2 Train / Val / Test (strictly parallel — NEVER overwrites frozen test_bili_gold_v1!)
    gold_v2_train_rows: list[dict[str, Any]] = []
    gold_v2_val_rows: list[dict[str, Any]] = []
    gold_v2_test_rows: list[dict[str, Any]] = []
    gold_v2_override_sids: set[str] = set()
    if args.gold_v2_dir and args.gold_v2_dir.is_dir():
        g2d = args.gold_v2_dir.resolve()
        gold_v2_train_rows = _read_jsonl_list(g2d / "v12_gold_v2_train.jsonl.gz")
        gold_v2_val_rows = _read_jsonl_list(g2d / "v12_gold_v2_val.jsonl.gz")
        gold_v2_test_rows = _read_jsonl_list(g2d / "v12_gold_v2_test.jsonl.gz")
        gold_v2_override_sids = {str(r.get("sample_id", "")) for r in gold_v2_train_rows if r.get("sample_id")}
        print(
            f"[Stage0-GoldV2] Loaded Gold-v2: train={len(gold_v2_train_rows)} (override_sids={len(gold_v2_override_sids)}) | "
            f"val_v2={len(gold_v2_val_rows)} | test_v2={len(gold_v2_test_rows)}",
            flush=True,
        )

    # Apply `v12_data_refinery.py` (purges ultra-short plugin clichés + qwen_inkling_joint + blocks midband from T0!)
    refined_items: list[dict[str, Any]] = []
    t3_dropped = 0
    g2_deduped = 0
    tier_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    for it in train_raw:
        sid_it = str(it.get("sample_id", ""))
        if sid_it and sid_it in gold_v2_override_sids:
            g2_deduped += 1
            continue
        src_it = orig_src_by_sid.get(sid_it, str(it.get("score_source", "")))
        if not src_it and it.get("is_pre_llm"):
            src_it = "pre_llm_human_prior"
        it["score_source"] = src_it
        is_gem_pro = sid_it in gem_pro_sids
        is_gem_any = is_gem_pro or (sid_it in gem_flash_sids)
        tier, tw, eff_sc, reason = classify_epistemic_tier_v12(
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
        it["epistemic_tier"] = tier
        it["is_true_pro_gold"] = bool(tier == "T0_DIAMOND_GOLD")
        it["weight"] = float(it["weight"]) * float(tw)
        it["y_promo"] = -1.0
        it["has_promo"] = 0.0
        it["sub4_idx"] = -1
        it["is_rescue"] = False
        it["l_pceil_admit"] = False
        refined_items.append(it)

    ipw_w = compute_joint_length_domain_ipw([{"text": x["raw_text"], "y_score": x["y_score"]} for x in refined_items])
    for it, iw in zip(refined_items, ipw_w):
        it["weight"] = float(np.clip(float(it["weight"]) * float(iw), 0.20, 3.0))
    print(
        f"[Stage0-V12Refinery] kept={len(refined_items):,} | T3_purged={t3_dropped:,} | g2_replaced={g2_deduped} | "
        f"tiers={dict(tier_counts)} | top_reasons={reason_counts.most_common(6)}",
        flush=True,
    )

    # 4. Ingest Wave-1 + Wave-2 + Wave-2.5 + Gold-v2 train + UOT + P10-3 2k k-NN + Pairs + Longform!
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
        is_t0: bool = False, is_rescue: bool = False, l_pceil: bool = False,
    ) -> dict[str, Any]:
        cl, f56, meta = extract_style_features_56d(txt)
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        return {
            "raw_text": txt, "clean_text": cl, "ids": ids,
            "engram": make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
            "trigram": make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10),
            "style": f56, "spoof_any": float(meta["spoof_any"]),
            "y_score": float(np.clip(y_ai, 0.0, 1.0)),
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
    promo_val_select_rows: list[dict[str, Any]] = []
    promo_val_cert_rows: list[dict[str, Any]] = []
    uot_pseudo_rows: list[dict[str, Any]] = []
    hard_neg_rows: list[dict[str, Any]] = []
    gold_v2_t0_items: list[dict[str, Any]] = []

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
            rec = _make_item(txt, float(r["ai_score"]) / 100.0, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.5, is_t0=True, is_rescue=is_resc)
            if sh in cert_shas:
                promo_val_cert_rows.append(rec)
            elif sh in select_shas:
                promo_val_select_rows.append(rec)
            elif y_pr_v <= 0.15 and len(promo_val_select_rows) < (64 if args.smoke_test else 650) and (int(sh[:8], 16) % 16 == 0):
                promo_val_select_rows.append(rec)
            else:
                dual_gold_train.append(rec)

    # Wave-2.5 (243 wild disagreement dual-axis gold -> unlocks unseen stealth homophone promos!)
    w25_loaded = 0
    if args.wave25_gold and args.wave25_gold.is_file():
        for r in _read_jsonl_list(args.wave25_gold):
            txt = str(r["text"])
            rec = _make_item(txt, float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0, 1.0, str(r.get("promo_type", "none")), 1.65, is_t0=True, is_rescue=True)
            dual_gold_train.append(rec)
            gold_v2_t0_items.append(rec)
            w25_loaded += 1

    # Gold-v2 Train (up to 1,429 re-adjudicated dual-axis gold -> T0 + dual_gold_train!)
    for r in gold_v2_train_rows:
        y_ai_v = float(r["ai_score"]) / 100.0
        y_pr_v = float(r["promo_score"]) / 100.0
        tier_g2, tw_g2, _, _ = classify_epistemic_tier_v12(
            float(r["ai_score"]), "v12_gold_v2_pro", str(r["text"]),
            v11_or_v12_verified=True, is_hybrid_mixed=bool(r.get("is_hybrid_mixed", False)),
        )
        rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.60 * (1.0 if tier_g2 == "T0_DIAMOND_GOLD" else 0.5), is_t0=(tier_g2 == "T0_DIAMOND_GOLD"), is_rescue=True)
        dual_gold_train.append(rec)
        if tier_g2 == "T0_DIAMOND_GOLD":
            gold_v2_t0_items.append(rec)

    # Wave-1 1,600 counterfactual pairs
    dual_pairs_train: list[dict[str, Any]] = []
    for x in _read_jsonl_list(v11_dir / "v11_wave1_promo_counterfactual_pairs_1600.jsonl")[: (48 if args.smoke_test else 99999)]:
        sh_p = _sha_norm(_norm(x["promo_text"]))
        is_ai_v = 1.0 if x.get("promo_is_ai") else 0.0
        pt_s = str(x.get("promo_type", "soft_shill"))
        if sh_p in cert_shas:
            promo_val_cert_rows.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.0, is_t0=True))
            promo_val_cert_rows.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.0, is_t0=True))
        elif sh_p in select_shas:
            promo_val_select_rows.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.0, is_t0=True))
            promo_val_select_rows.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.0, is_t0=True))
        else:
            dual_pairs_train.append(x)
            dual_gold_train.append(_make_item(x["promo_text"], is_ai_v, 0.95, 1.0, pt_s, 1.5, is_t0=True))
            dual_gold_train.append(_make_item(x["organic_text"], is_ai_v, 0.02, 1.0, "none", 1.5, is_t0=True, l_pceil=True))

    # UOT/CDMAD pseudo pool + chants + P10-3 2k BGE k-NN hard negs
    for r in _read_jsonl_list(v11_dir / "v11_uot_cdmad_pseudo_full.jsonl.gz")[: (192 if args.smoke_test else 999999)]:
        txt = str(r["text"])
        sh = _sha_norm(_norm(txt))
        y_ai_v = float(r.get("old_ai_score", 20.0)) / 100.0
        if r.get("pseudo_label") == 1:
            rec = _make_item(txt, y_ai_v, float(r.get("pseudo_promo_score", 90.0)) / 100.0, 1.0, str(r.get("pseudo_promo_type", "soft_shill")), float(r.get("weight", 0.75)))
            if sh in cert_shas:
                promo_val_cert_rows.append(rec)
            elif sh in select_shas:
                promo_val_select_rows.append(rec)
            else:
                uot_pseudo_rows.append(rec)
        elif r.get("source") == "v11_uot_chant_hard_neg":
            hard_neg_rows.append(_make_item(txt, y_ai_v, 0.0, 1.0, "none", 1.6, l_pceil=True))
        else:
            uot_pseudo_rows.append(_make_item(txt, y_ai_v, 0.02, 1.0, "none", 0.60))

    for r in _read_jsonl_list(v11_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl")[: (48 if args.smoke_test else 9999)]:
        rec = _make_item(str(r["text"]), float(r.get("old_ai_score", 15.0)) / 100.0, 0.02, 1.0, "none", 1.0, l_pceil=bool(r.get("l_pceil_admitted", False)))
        hard_neg_rows.append(rec)

    wave2_longform_all = _read_jsonl_list(v11_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (20 if args.smoke_test else 9999)]
    wave2_longform = [d for i, d in enumerate(wave2_longform_all) if i % 10 != 9]
    wave2_longform_val = [d for i, d in enumerate(wave2_longform_all) if i % 10 == 9]
    print(
        f"[Stage0-DualData] dual_train={len(dual_gold_train):,} (wave25={w25_loaded}, g2_t0={len(gold_v2_t0_items)}) | "
        f"pairs={len(dual_pairs_train):,} | pv_sel={len(promo_val_select_rows):,} pv_cert={len(promo_val_cert_rows):,} | "
        f"uot={len(uot_pseudo_rows):,} | hard_neg={len(hard_neg_rows):,} | longform_tr={len(wave2_longform):,} val={len(wave2_longform_val):,}",
        flush=True,
    )

    # Pre-encode refined base AI rows with 56-d style features + build pools
    t0_pool, t1_pool, t2_pool, protect_pool = list(gold_v2_t0_items), [], [], []
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

    hu_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["y_score"] <= 0.15 and 40 <= len(x["clean_text"]) <= 220][:4000]
    hu_hard = [x["clean_text"] for x in protect_pool if x["y_score"] <= 0.20 and len(x["clean_text"]) >= 40][:2000] or hu_clean[:500]
    ai_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["y_score"] >= 0.85 and 40 <= len(x["clean_text"]) <= 240][:4000]
    pr_clean = [x["clean_text"] for x in dual_gold_train if x["y_promo"] >= 0.70 and x["y_score"] <= 0.30][:1000] or hu_clean[:50]
    longmix_syn = OnlineLongMixSynthesizerV11(hu_clean[:2000] or ["真人评论测试。"] * 12, hu_hard, ai_clean[:2000] or ["人工智能评论测试。"] * 12, pr_clean)

    # 5. Disjoint Eval Splits (`val_gold_v1`, `val_cal_select`, `val_cal_cert`, `test_bili_gold_v1`, `test_dsh_gold`, optional `val_gold_v2` & `test_bili_gold_v2`)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_all, _ = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    vg_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_gold}
    vc_dis = [r for r in val_cal_all if str(r.get("sample_id", r.get("id", ""))) not in vg_ids] or val_cal_all
    rng_v = random.Random(20261126)
    vc_idx = list(range(len(vc_dis)))
    rng_v.shuffle(vc_idx)
    half_vc = max(200, len(vc_idx) // 2)
    val_cal_select = [vc_dis[i] for i in vc_idx[:half_vc]]
    val_cal_cert = [vc_dis[i] for i in vc_idx[half_vc:]] or val_cal_select
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    if args.smoke_test:
        val_gold, val_cal_select, val_cal_cert, test_gold, test_dsh_gold = val_gold[:48], val_cal_select[:48], val_cal_cert[:48], test_gold[:48], test_dsh_gold[:24]

    def _pre_eval_v12(rows_e: list[dict[str, Any]]):
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

    vg_p = _pre_eval_v12(val_gold)
    vs_p = _pre_eval_v12(val_cal_select)
    vc_p = _pre_eval_v12(val_cal_cert)
    tg_p = _pre_eval_v12(test_gold)
    td_p = _pre_eval_v12(test_dsh_gold)
    vg2_p = _pre_eval_v12(gold_v2_val_rows) if gold_v2_val_rows else None
    tg2_p = _pre_eval_v12(gold_v2_test_rows) if gold_v2_test_rows else None

    # 6. Load `v1.1.0-R1` Teacher -> Exact Dense Transplant -> `V12` Step-0 CI Verification!
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

    mdl = MeguvizV12ApexNanoModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS),
    ).to(device).eval()

    def _collate_items(batch_it: list[dict[str, Any]]):
        ml = min(args.max_len, ((max(len(x["ids"]) for x in batch_it) + 15) // 16) * 16)
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

    # Calibrate CosineLogitNormHeadV12 scale from real v1.1 embeddings
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
    surg_rep = transplant_v11_to_v12_dense(v11_teacher, mdl, calib_emb_v11=emb11_cal)
    ci_g0 = run_v12_ci_gates(mdl, tokenizer)
    assert ci_g0["all_ci_passed"], f"Step-0 V12 CI failed: {ci_g0}"
    r11_anchor_params = {k: v.detach().cpu().clone() for k, v in mdl.named_parameters() if not k.startswith(("rung0.", "z_char_"))}
    print(f"[Stage0-DenseTransplant] {json.dumps(surg_rep)} | G0_CI=PASS", flush=True)

    @torch.no_grad()
    def _predict_v12(m_e: MeguvizV12ApexNanoModel, pack_t, outer_steps: int = 2) -> dict[str, np.ndarray]:
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
            ml = max(len(enc_l[i]) for i in mb)
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

    def _eval_ai_metrics(m_e: MeguvizV12ApexNanoModel) -> dict[str, float]:
        vo = _predict_v12(m_e, vg_p, 2)
        so = _predict_v12(m_e, vs_p, 2)
        fw = m_e.fusion_weights.detach().cpu().numpy()
        p_vg = expit(np.clip(fw[0] * vo["logit_t1"] + fw[1] * vo["logit"] + fw[2] * vo["ebt_margin"], -30.0, 30.0))
        p_vs = expit(np.clip(fw[0] * so["logit_t1"] + fw[1] * so["logit"] + fw[2] * so["ebt_margin"], -30.0, 30.0))
        mv = {**compute_eval_metrics(vg_p[7], p_vg), **compute_high_precision_metrics(vg_p[7], p_vg)}
        ms = {**compute_eval_metrics(vs_p[7], p_vs), **compute_high_precision_metrics(vs_p[7], p_vs)}
        mr1 = compute_eval_metrics(vg_p[7], expit(vo["logit_r1"]))
        mr0 = compute_eval_metrics(vs_p[7], expit(so["logit_ai_r0"]))
        ai_obj = (
            0.50 * (0.36 * mv["auroc"] + 0.28 * mv["pr_auc"] + 0.14 * mv["best_f05"] + 0.16 * mv["recall_at_p95"] + 0.06 * mr1["auroc"])
            + 0.50 * (0.30 * ms["auroc"] + 0.24 * ms["pr_auc"] + 0.14 * ms["best_f05"] + 0.20 * ms["recall_at_p95"] + 0.06 * ms["recall_at_p99"] + 0.06 * mr1["auroc"])
        )
        out_r = {
            "ai_obj": float(ai_obj), "auroc_vg": float(mv["auroc"]), "rp95_vg": float(mv["recall_at_p95"]),
            "rp95_wild": float(ms["recall_at_p95"]), "r1_auroc": float(mr1["auroc"]), "r0_ai_auroc": float(mr0["auroc"]),
        }
        if vg2_p is not None:
            v2o = _predict_v12(m_e, vg2_p, 2)
            p_v2 = expit(np.clip(fw[0] * v2o["logit_t1"] + fw[1] * v2o["logit"] + fw[2] * v2o["ebt_margin"], -30.0, 30.0))
            mv2 = {**compute_eval_metrics(vg2_p[7], p_v2), **compute_high_precision_metrics(vg2_p[7], p_v2)}
            out_r["auroc_vg2"] = float(mv2["auroc"])
            out_r["rp95_vg2"] = float(mv2["recall_at_p95"])
            ai_obj_v2 = 0.36 * mv2["auroc"] + 0.28 * mv2["pr_auc"] + 0.14 * mv2["best_f05"] + 0.16 * mv2["recall_at_p95"] + 0.06 * mr1["auroc"]
            # Blend clean Gold-v2 val (50%) + Wild Cal (35%) + Legacy Gold-v1 (15%) when Gold-v2 is available!
            out_r["ai_obj_sel"] = float(0.50 * ai_obj_v2 + 0.50 * out_r["ai_obj"])
        else:
            out_r["ai_obj_sel"] = float(out_r["ai_obj"])
        return out_r

    def _eval_promo_and_long(m_e: MeguvizV12ApexNanoModel) -> dict[str, float]:
        out_m = {"promo_auroc": 0.5, "promo_pr_auc": 0.5, "r0_promo_auroc": 0.5, "long_ai_auroc": 0.5, "long_hyb_auroc": 0.5, "long_pr_auroc": 0.5}
        if promo_val_select_rows:
            pv_pk = (
                [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
                [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
                np.stack([x["style"] for x in promo_val_select_rows], axis=0), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32),
            )
            pvo = _predict_v12(m_e, pv_pk, 2)
            yv = (pv_pk[7] >= 0.5).astype(int)
            if len(np.unique(yv)) == 2:
                out_m["promo_auroc"] = float(roc_auc_score(yv, pvo["logit_promo"]))
                out_m["promo_pr_auc"] = float(average_precision_score(yv, pvo["logit_promo"]))
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

    # Multi-pool weighted sampler + online forensic-safe augmenters
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
    pike_ai = DynamicPiKEGroupSamplerV11({"t0_ai": 0.38, "t1_ai": 0.34, "t2_mid": 0.02, "ai_protect": 0.10, "dual_gold": 0.12, "uot_pseudo": 0.02, "hard_neg": 0.02})
    pike_dual = DynamicPiKEGroupSamplerV11({"t0_ai": 0.25, "t1_ai": 0.20, "t2_mid": 0.01, "ai_protect": 0.04, "dual_gold": 0.24, "uot_pseudo": 0.14, "hard_neg": 0.12})

    def _sample_short_batch(bs_now: int, w_vec: np.ndarray, gold_v2_frac: float = 0.0) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        n_g2 = min(bs_now // 4, int(round(bs_now * gold_v2_frac))) if (gold_v2_frac > 0.0 and gold_v2_t0_items) else 0
        for _ in range(n_g2):
            out_b.append(gold_v2_t0_items[rng_tr.randrange(len(gold_v2_t0_items))])
        rem_bs = max(1, bs_now - n_g2)
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
                    it = _make_item(aug_d["text"], 0.0, 0.0, 1.0, "none", 1.6, l_pceil=True)
                elif pname == "dual_gold" and it["y_promo"] >= 0.65 and rng_tr.random() < 0.35:
                    it = _make_item(obf_aug.obfuscate_promo_or_ai(it["raw_text"], 0.65), it["y_score"], it["y_promo"], 1.0, "soft_shill" if it["sub4_idx"] == 3 else "coupon_commerce", it["weight"], is_t0=True)
                elif pname == "t0_ai" and it["y_score"] >= 0.85 and rng_tr.random() < 0.08:
                    it = _make_item(hart_aug.light_humanize_ai(it["raw_text"]), it["y_score"], -1.0, 0.0, "none", it["weight"], is_t0=True)
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
    # PHASE A: Firewalled Dense-Adapter Pure-AI (`CosineLogitNorm + SharedQuantRes + coda[:192] + Rung-0`)
    # (Deep dense trunk frozen -> ZERO fresh-AdamW shock while absorbing Gold-v2 + v12_data_refinery!)
    # =========================================================================
    pa_steps = 4 if args.smoke_test else int(args.pa_steps)
    pg_a, hooks_a = build_v12_param_groups(mdl, "phase_a")
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
        # Dedicated 8% Gold-v2 + Wave-2.5 slice per batch (~40 clean re-adjudicated rows/batch at BS=512!)
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_ai.w, gold_v2_frac=0.08)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era, code_i) = _collate_items(b_it)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            # z_char gates stay 0.0 in Phase A so Rung-0 warms up purely on AI with zero coupling shock!
            out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)
            r0_o = mdl.forward_rung0(c1, c3, cm, cf)
            with torch.no_grad():
                to11 = v11_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)

            lg, lg_e1, lg_r1 = out["logit"].float(), out["exit1_out"]["logit"].float(), out["logit_r1"].float()
            y_cl = y_ai.clamp(1e-4, 1.0 - 1e-4)
            tp11 = torch.sigmoid(to11["logit"].float())
            is_q3 = (y_ai <= 0.30) & (has_pr >= 0.5) & (y_pr >= 0.65)
            # Selective v1.1 anchor: excluded on Gold-v2/Wave-2.5 rescue rows, Q3 commercial, or T0 where v1.1 erred!
            m_anc = (~is_q3) & (is_res < 0.5) & (~((is_t0 > 0.5) & ((tp11 - y_ai).abs() >= 0.40)))

            l_kd = lg.sum() * 0.0
            if m_anc.any():
                # Clamp unbounded v1.1 teacher logits to [-12, 12] so CosineLogitNormHeadV12 never fights scale ceiling!
                t11_lg_c = to11["logit"][m_anc].float().clamp(-12.0, 12.0)
                t11_r1_c = to11["logit_r1"][m_anc].float().clamp(-12.0, 12.0)
                l_kd = (
                    0.60 * F.mse_loss(lg[m_anc], t11_lg_c)
                    + 0.30 * F.mse_loss(lg_e1[m_anc], t11_lg_c)
                    + 0.35 * F.mse_loss(lg_r1[m_anc], t11_r1_c)
                    + 0.40 * (1.0 - F.cosine_similarity(out["emb"][m_anc, :AI_EMB_END_V11].float(), to11["emb"][m_anc, :AI_EMB_END_V11].float(), dim=-1).mean())
                )

            l_ai = (F.binary_cross_entropy_with_logits(lg, y_cl, reduction="none") * w_b).mean()
            l_ai = l_ai + 0.25 * (F.binary_cross_entropy_with_logits(lg_e1, y_cl, reduction="none") * w_b).mean()
            l_ai = l_ai + 0.22 * (F.binary_cross_entropy_with_logits(lg_r1, y_cl, reduction="none") * w_b * (1.0 + 2.5 * (y_ai >= 0.45).float())).mean()
            l_ai = l_ai + 0.08 * listnet_tail_distill_loss(lg_r1, lg.detach()) + 0.06 * listnet_tail_distill_loss(lg_e1, lg.detach())
            l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_ai, w_b, grp, prog=prog)
            l_topfp = _top_fp_logit_ceiling_loss(lg, y_ai, tp11, torch.maximum(is_t0, is_q3.float()), era, w_b)

            # Pillar 2 Monotonic T=2 >= T=1 Margin Loss on decisive gold anchors
            dec_m = (y_ai >= 0.80) | (y_ai <= 0.15)
            l_mono = lg.sum() * 0.0
            if dec_m.any():
                sgn = torch.where(y_ai[dec_m] >= 0.5, 1.0, -1.0)
                l_mono = 0.08 * F.relu(sgn * (lg_e1[dec_m].detach() - lg[dec_m])).mean()

            # Structural Regularizers (l_slot_div + valid code_idx + L2-SP -> zero T=2 drift!)
            l_sdiv = 0.15 * compute_slot_orthogonality_loss(out["slots_final"])
            l_code = lg.sum() * 0.0
            if (code_i >= 0).any():
                l_code = 0.04 * F.cross_entropy(out["code_logit"].float(), code_i, ignore_index=-100)
            l_sp = compute_l2_sp_loss(mdl, r11_anchor_params, wd_sp=0.03)

            # Rung-0 AI Warmup (FN-Weighted BCE + Teacher Logit-MSE + ListNet)
            lg_r0 = r0_o["logit_ai_r0"].float()
            l_r0 = (
                0.50 * (F.binary_cross_entropy_with_logits(lg_r0, y_cl, reduction="none") * w_b * (1.0 + 3.0 * (y_ai >= 0.45).float())).mean()
                + 0.30 * F.mse_loss(lg_r0, lg.detach())
                + 0.08 * listnet_tail_distill_loss(lg_r0, lg.detach())
            )

            # Legacy GateGuard / Literary / Inkling-WK pairs (gentle 0.06 weight -> sharpens traps without blowing tail!)
            l_ai_pair = lg.sum() * 0.0
            if legacy_pairs and step % 2 == 1:
                lpb = rng_tr.sample(legacy_pairs, min(16 if not args.smoke_test else 4, len(legacy_pairs)))
                ai_it, hu_it, pw_l = [], [], []
                for lp in lpb:
                    if "_ai_rec" not in lp:
                        lp["_ai_rec"] = _make_item(str(lp["ai_text"]), 1.0, -1.0, 0.0, "none", 1.0)
                        lp["_hu_rec"] = _make_item(str(lp["human_text"]), 0.0, -1.0, 0.0, "none", 1.0)
                    ai_it.append(lp["_ai_rec"]); hu_it.append(lp["_hu_rec"]); pw_l.append(float(lp.get("weight", 1.0)))
                ai_i, ai_m, ai_e, ai_t, ai_s, *_ = _collate_items(ai_it)
                hu_i, hu_m, hu_e, hu_t, hu_s, *_ = _collate_items(hu_it)
                oa = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, trigram_ids=ai_t, outer_steps=2, return_aux=True)
                oh = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, trigram_ids=hu_t, outer_steps=2, return_aux=True)
                pw_t = torch.tensor(pw_l, dtype=torch.float32, device=device)
                l_ai_pair = 0.06 * (
                    (F.softplus(1.8 - (oa["logit"].float() - oh["logit"].float())) * pw_t).mean()
                    + 0.6 * ((F.relu(0.84 - torch.sigmoid(oa["logit"].float())).pow(2) + 1.5 * F.relu(torch.sigmoid(oh["logit"].float()) - 0.12).pow(2)) * pw_t).mean()
                )

            loss_a = (
                l_ai + 0.06 * l_pa + 0.35 * l_tp + 0.15 * zenith_graded_anchor_loss(lg, y_ai) + 0.10 * l_topfp
                + l_mono + 0.70 * l_kd + l_sdiv + l_code + l_sp + l_r0 + l_ai_pair
            )

        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt_a.step()

        if not args.smoke_test and ((step + 1) % 200 == 0 or step + 1 == pa_steps):
            cm = _eval_ai_metrics(mdl)
            if cm["ai_obj_sel"] >= best_pa_sc:
                best_pa_sc = cm["ai_obj_sel"]
                best_pa_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(
                f"[PhaseA step={step+1}/{pa_steps}] loss={float(loss_a.item()):.4f} | ai_sel={cm['ai_obj_sel']:.5f} ai_obj={cm['ai_obj']:.5f} "
                f"auroc_vg={cm['auroc_vg']:.4f} rp95_vg={cm['rp95_vg']:.4f} vg2={cm.get('auroc_vg2', 0.0):.4f} rp95_vg2={cm.get('rp95_vg2', 0.0):.4f} "
                f"r1={cm['r1_auroc']:.4f} r0_ai={cm['r0_ai_auroc']:.4f} | elapsed={time.monotonic()-t_start:.0f}s",
                flush=True,
            )
            mdl.train()

    for hk in hooks_a:
        hk.remove()
    # Preserve trained Rung-0 weights even if AI adapters pick an earlier snapshot!
    r0_trained_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items() if k.startswith("rung0.")}
    best_pa_sd.update(r0_trained_sd)
    mdl.load_state_dict({k: v.to(device) for k, v in best_pa_sd.items()}, strict=True)
    del v11_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pa_ai_m = _eval_ai_metrics(mdl)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_phaseA_ckpt.pt")
    print(f"[PhaseA DONE] best_ai_obj={pa_ai_m['ai_obj']:.5f} (ref={ref_ai_m['ai_obj']:.5f}) | {json.dumps({k: round(v, 5) for k, v in pa_ai_m.items()})}", flush=True)

    # =========================================================================
    # PHASE B: 100% Frozen-AI Firewall + Promo [192:256] + ChronoStream + Rung-0 Dual-Axis Teacher Distillation!
    # =========================================================================
    pb_steps = 4 if args.smoke_test else int(args.pb_steps)
    pg_b, hooks_b = build_v12_param_groups(mdl, "phase_b")
    opt_b = torch.optim.AdamW(pg_b, betas=(0.9, 0.95))
    asam_b = ASAMPerturber(mdl, rho=0.04)
    best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pb_sc = -1e9
    mdl.train()

    for step in range(pb_steps):
        mult = _wsd_lr_mult(step, pb_steps, warmup_frac=0.05, decay_frac=0.20)
        for pg in opt_b.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era, code_i) = _collate_items(b_it)

        def _phase_b_loss():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                r0_o = mdl.forward_rung0(c1, c3, cm, cf)
                # Pass detached z_char during Phase B so Rung-0 updates NEVER alter frozen AI logits!
                out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)

                loss_b = (
                    0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai, y_pr, has_pr, a5_b)
                    + compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai, y_pr, has_pr)
                    + compute_v12_orthogonality_loss(mdl, out["logit"].detach(), y_ai, y_pr, has_pr)
                )
                u, u_t1, u_r1 = out["logit_promo"].float(), out["exit1_out"]["logit_promo"].float(), out["logit_promo_r1"].float()
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    yp_c = y_pr[pm].clamp(1e-4, 1.0 - 1e-4)
                    loss_b = loss_b + (F.binary_cross_entropy_with_logits(u[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.25 * (F.binary_cross_entropy_with_logits(u_t1[pm], yp_c, reduction="none") * w_b[pm]).mean()
                    loss_b = loss_b + 0.35 * (F.binary_cross_entropy_with_logits(u_r1[pm], yp_c, reduction="none") * w_b[pm] * (1.0 + 3.0 * (y_pr[pm] >= 0.65).float())).mean()
                    hn_m = pm & (y_pr <= 0.25) & (l_pc > 0.5)
                    if int(hn_m.sum().item()) >= 2:
                        u_worst = torch.topk(u[hn_m], k=min(4, int(hn_m.sum().item()))).values
                        loss_b = loss_b + 0.22 * (F.relu(u_worst).pow(2).mean() + 0.20 * F.softplus(u_worst + 0.8).mean())
                    loss_b = loss_b + 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], u, sub4_i, y_pr, has_pr)

                # Rung-0 Dual-Axis Teacher Distillation (Asymmetric FN-Weighted BCE + Logit-MSE + ListNet + Deferral!)
                r0_ai = r0_o["logit_ai_r0"].float()
                r0_pr = r0_o["logit_promo_r0"].float()
                t_ai_lg = out["logit"].detach().float()
                t_pr_lg = u.detach()
                l_r0_ai = (
                    0.45 * (F.binary_cross_entropy_with_logits(r0_ai, y_ai.clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b * (1.0 + 3.0 * (y_ai >= 0.45).float())).mean()
                    + 0.40 * F.mse_loss(r0_ai, t_ai_lg)
                    + 0.10 * listnet_tail_distill_loss(r0_ai, t_ai_lg)
                )
                l_r0_pr = 0.35 * F.mse_loss(r0_pr, t_pr_lg) + 0.08 * listnet_tail_distill_loss(r0_pr, t_pr_lg)
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    l_r0_pr = l_r0_pr + 0.50 * (F.binary_cross_entropy_with_logits(r0_pr[pm], y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm] * (1.0 + 3.5 * (y_pr[pm] >= 0.65).float())).mean()
                with torch.amp.autocast("cuda", enabled=False):
                    pa0, pa_t = torch.sigmoid(r0_ai.detach()), torch.sigmoid(t_ai_lg)
                    pp0, pp_t = torch.sigmoid(r0_pr.detach()), torch.sigmoid(t_pr_lg)
                    d0_tgt = torch.maximum(
                        torch.maximum(((pa_t - pa0).abs() * 3.0).clamp(0.0, 1.0), ((y_ai >= 0.45) & (pa0 < 0.35)).float()),
                        torch.maximum(((pp_t - pp0).abs() * 3.0).clamp(0.0, 1.0), ((has_pr >= 0.5) & (y_pr >= 0.65) & (pp0 < 0.35)).float()),
                    )
                    l_r0_def = 0.25 * F.binary_cross_entropy(r0_o["p_defer_r0"].float().clamp(1e-5, 1.0 - 1e-5), d0_tgt)
                loss_b = loss_b + l_r0_ai + l_r0_pr + l_r0_def

                if dual_pairs_train and step % 2 == 0:
                    pb = rng_tr.sample(dual_pairs_train, min(16 if not args.smoke_test else 4, len(dual_pairs_train)))
                    p_it, o_it = [], []
                    for x in pb:
                        pt_t, ot_t = slot_aug.augment_pair(x["promo_text"], x["organic_text"]) if rng_tr.random() < 0.5 else (x["promo_text"], x["organic_text"])
                        ia = 1.0 if x.get("promo_is_ai") else 0.0
                        p_it.append(_make_item(pt_t, ia, 1.0, 1.0, str(x.get("promo_type", "soft_shill")), 1.5))
                        o_it.append(_make_item(ot_t, ia, 0.0, 1.0, "none", 1.5))
                    pi, pm_, pe, pt_, ps, *_ = _collate_items(p_it)
                    oi, om_, oe, ot_, os_, *_ = _collate_items(o_it)
                    op = mdl(pi, pm_, ps, engram_ids=pe, trigram_ids=pt_, outer_steps=1, return_aux=True)
                    oo = mdl(oi, om_, os_, engram_ids=oe, trigram_ids=ot_, outer_steps=1, return_aux=True)
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
        # Firewalled ASAM flatness polish in final 35% of Phase B!
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
                f"[PhaseB step={step+1}/{pb_steps}] loss={float(loss_b.item()):.4f} | ai_obj={ai_mb['ai_obj']:.5f} | "
                f"promo_auc={pl_m['promo_auroc']:.4f} pr={pl_m['promo_pr_auc']:.4f} | long_ai={pl_m['long_ai_auroc']:.4f} "
                f"hyb={pl_m['long_hyb_auroc']:.4f} pr={pl_m['long_pr_auroc']:.4f} | r0_ai={ai_mb['r0_ai_auroc']:.4f} r0_pr={pl_m['r0_promo_auroc']:.4f} | "
                f"elapsed={time.monotonic()-t_start:.0f}s",
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
    #          Capped Unit-Sphere LEACE -> 3-Param Fusion -> Dual-Rung R0 & R1 Fixed-Seq LTT -> MGVZ7 + 5-Stage ONNX!
    # =========================================================================
    cal_f = _collate_items(cal_items[:32])
    fold_rep = fold_zero_op_v12(mdl, calib_inputs=[(cal_f[0], cal_f[1], cal_f[4], cal_f[2])])
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    fp16_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_teacher.parameters():
        p.requires_grad_(False)

    def _run_qat_stage(st_name: str, init_sd: dict[str, torch.Tensor], qmode: str, steps: int, lam_p: float = 1.0) -> tuple[dict[str, torch.Tensor], float]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(qmode)
        pg_q, hooks_q = build_v12_param_groups(mdl, "qat")
        opt_q = torch.optim.AdamW(pg_q, betas=(0.9, 0.95))
        best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc_q = -1e9
        mdl.train()
        for step in range(steps):
            mult = _wsd_lr_mult(step, steps, warmup_frac=0.05, decay_frac=0.25)
            for pg in opt_q.param_groups:
                pg["lr"] = pg["base_lr"] * mult
            opt_q.zero_grad(set_to_none=True)
            b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w)
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, *_rest) = _collate_items(b_it)
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
                l_sup = 0.15 * (F.binary_cross_entropy_with_logits(lg, y_ai.clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b).mean()
                if (has_pr > 0.5).any():
                    pm = has_pr > 0.5
                    l_sup = l_sup + 0.20 * lam_p * (F.binary_cross_entropy_with_logits(out["logit_promo"][pm].float(), y_pr[pm].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[pm]).mean()
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
                sc_q = cm["ai_obj"] + 0.35 * (0.5 * pm_["promo_auroc"] + 0.5 * pm_["promo_pr_auc"])
                if sc_q >= best_sc_q:
                    best_sc_q = sc_q
                    best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                print(f"[{st_name} step={step+1}/{steps}] ai_obj={cm['ai_obj']:.5f} auroc_vg={cm['auroc_vg']:.4f} rp95_vg={cm['rp95_vg']:.4f} | promo={pm_['promo_auroc']:.4f} | r0_ai={cm['r0_ai_auroc']:.4f}", flush=True)
                mdl.train()
        for hk in hooks_q:
            hk.remove()
        if best_sc_q <= -1e8:
            best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            cm = _eval_ai_metrics(mdl)
            best_sc_q = cm["ai_obj"]
        return best_sd_q, best_sc_q

    w4_sd, _ = _run_qat_stage("PhaseC-UPQ-W4", folded_sd, "w4", 4 if args.smoke_test else int(args.pc_w4_steps))
    q1_sd, q1_sc = _run_qat_stage("PhaseC-Sub2-A", w4_sd, "sub2", 4 if args.smoke_test else int(args.pc_sub2_steps), lam_p=1.0)
    q2_sd, q2_sc = _run_qat_stage("PhaseC-Sub2-B", w4_sd, "sub2", 4 if args.smoke_test else int(args.pc_sub2_steps), lam_p=1.25)

    def _qat_soup_eval(sd_c: dict[str, torch.Tensor]) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_c.items()}, strict=True)
        cm, pm_ = _eval_ai_metrics(mdl), _eval_promo_and_long(mdl)
        return cm["ai_obj"] + 0.35 * (0.5 * pm_["promo_auroc"] + 0.5 * pm_["promo_pr_auc"]) + 0.12 * pm_["long_ai_auroc"]

    qat_sd, qat_soup_rep = greedy_pauc_soup(w4_sd, [("Q1", q1_sd, q1_sc), ("Q2", q2_sd, q2_sc)], eval_fn=_qat_soup_eval, dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in qat_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_qat_soup.pt")

    # MGVZ7 Pack/Unpack -> Anchored FP16 Shock-Absorber!
    tmp_mg = out_dir / "_tmp_v12_pre_shock.mgvz"
    pack_bundle_v7_v12(mdl.cpu(), tmp_mg)
    load_bundle_v7_v12_into_model(mdl, tmp_mg)
    tmp_mg.unlink(missing_ok=True)
    mdl = mdl.to(device).eval()

    pre_sh_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_sh_sc = _qat_soup_eval(pre_sh_sd)
    shock_tokens = ("core.qres.", "core.inloop_ebt.", "core.spectral_damp", "ai_head.", "promo", "r1_ai_head", "r1_promo", "sub4", "attr6", "longstream", "defer_", "ebt_logit_gate", "mil_gate", "rung0.out_head", "rung0.trunk")
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
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike_dual.w)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf, y_ai, y_pr, has_pr, sub4_i, w_b, *_r) = _collate_items(b_it)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            o = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, trm_lam_k0=0.0, trm_lam_k1=0.0, return_aux=True)
            with torch.no_grad():
                tv_sh = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
            loss_sh = (
                0.70 * F.mse_loss(o["logit"].float(), tv_sh["logit"].float())
                + 0.40 * F.mse_loss(o["logit_r1"].float(), tv_sh["logit_r1"].float())
                + 0.35 * F.mse_loss(o["logit_ai_r0"].float(), tv_sh["logit_ai_r0"].float())
                + 0.60 * F.mse_loss(o["logit_promo"].float(), tv_sh["logit_promo"].float())
                + 0.30 * F.mse_loss(o["logit_promo_r1"].float(), tv_sh["logit_promo_r1"].float())
                + 0.15 * (F.binary_cross_entropy_with_logits(o["logit"].float(), y_ai.clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b).mean()
            )
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

    # Capped Unit-Sphere Oblique LEACE (max AI strength = 0.35!)
    vs_o = _predict_v12(mdl, vs_p, 2)
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
        pso_l = _predict_v12(mdl, ps_l_pk, 2)
        z_ll_ps = np.log1p(np.array([len(t) for t in ps_l_pk[3]], dtype=np.float32))[:, None]
        emb_l = np.concatenate([emb_l, pso_l["emb"]], axis=0)
        yai_l = np.concatenate([yai_l, np.full(len(promo_val_select_rows), 0.50, dtype=np.float32)], axis=0)
        ypr_l = np.concatenate([ypr_l, ps_l_pk[7]], axis=0)
        zai_l = np.concatenate([zai_l, np.concatenate([z_ll_ps, expit(pso_l["logit_promo"])[:, None]], axis=1)], axis=0)
        zpr_l = np.concatenate([zpr_l, np.concatenate([z_ll_ps, expit(pso_l["logit"])[:, None]], axis=1)], axis=0)
    leace_rep = apply_dual_axis_oblique_leace_v12(mdl, emb_l, yai_l, ypr_l, zai_l, zpr_l)

    # Cross-Exit Affine Alignment + 3-Parameter Log-Odds Fusion!
    al_sub = cal_items[: (64 if args.smoke_test else min(4000, len(cal_items)))]
    al_pk = ([x["ids"] for x in al_sub], [x["engram"] for x in al_sub], [x["trigram"] for x in al_sub], [x["raw_text"] for x in al_sub], [x["clean_text"] for x in al_sub], np.stack([x["style"] for x in al_sub]), [], np.zeros(len(al_sub)))
    al_o = _predict_v12(mdl, al_pk, 2)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_r1"], al_o["raw_logit"])
    a_t1, b_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_t1"], al_o["raw_logit"])
    pa_r1, pb_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_r1"], al_o["raw_logit_promo"])
    pa_t1, pb_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_t1"], al_o["raw_logit_promo"])
    with torch.no_grad():
        mdl.exit_affine_a.copy_(torch.tensor([a_r1, a_t1, 1.0], device=device))
        mdl.exit_affine_b.copy_(torch.tensor([b_r1, b_t1, 0.0], device=device))
        mdl.exit_affine_promo_a.copy_(torch.tensor([pa_r1, pa_t1, 1.0], device=device))
        mdl.exit_affine_promo_b.copy_(torch.tensor([pb_r1, pb_t1, 0.0], device=device))

    vg_o, vs_o, vc_o = _predict_v12(mdl, vg_p, 2), _predict_v12(mdl, vs_p, 2), _predict_v12(mdl, vc_p, 2)
    fus_t1 = [vs_o["logit_t1"], vg_o["logit_t1"]]
    fus_t2 = [vs_o["logit"], vg_o["logit"]]
    fus_eb = [vs_o["ebt_margin"], vg_o["ebt_margin"]]
    fus_sc = [vs_p[7], vg_p[7]]
    if vg2_p is not None:
        vg2_o = _predict_v12(mdl, vg2_p, 2)
        fus_t1.append(vg2_o["logit_t1"]); fus_t2.append(vg2_o["logit"]); fus_eb.append(vg2_o["ebt_margin"]); fus_sc.append(vg2_p[7])
    fusion_rep = fit_3param_logodds_fusion(np.concatenate(fus_t1), np.concatenate(fus_t2), np.concatenate(fus_eb), np.concatenate(fus_sc))
    with torch.no_grad():
        mdl.fusion_weights.copy_(torch.tensor([fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"]], device=device))
    fw_w = (fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"])

    def _fused_lg(o_d: dict[str, np.ndarray]) -> np.ndarray:
        return fw_w[0] * o_d["logit_t1"] + fw_w[1] * o_d["logit"] + fw_w[2] * o_d["ebt_margin"]

    # Calibration LUTs (`R0`, `R1`, `Full`) + Dual-Rung Strict 1-D Fixed-Sequence LTT Certification (`R0` & `R1`)!
    y_cal_ai = np.concatenate([vs_p[7], vg_p[7]]) / 100.0
    lut_ai_full = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([_fused_lg(vs_o), _fused_lg(vg_o)]), y_cal_ai, fit_binary_ece_mode=True), 256)
    lut_ai_r1 = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit_r1"], vg_o["logit_r1"]]), y_cal_ai, fit_binary_ece_mode=True), 256)
    lut_ai_r0 = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit_ai_r0"], vg_o["logit_ai_r0"]]), y_cal_ai, fit_binary_ece_mode=True), 256)

    pi_nat_pr = estimate_wild_prevalence_em(expit(al_o["logit_promo"]), pi_train=0.10)
    lut_pr_full = lut_pr_r1 = lut_pr_r0 = None
    r1_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False}
    r0_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False, "tau_ai": 0.08, "tau_promo": 0.08, "defer_max": 0.35, "max_chars": 160}

    if promo_val_select_rows and promo_val_cert_rows:
        def _pack_pr(rows_pr):
            return ([x["ids"] for x in rows_pr], [x["engram"] for x in rows_pr], [x["trigram"] for x in rows_pr], [x["raw_text"] for x in rows_pr], [x["clean_text"] for x in rows_pr], np.stack([x["style"] for x in rows_pr]), [{"spoof_any": x["spoof_any"]} for x in rows_pr], np.array([x["y_promo"] for x in rows_pr], dtype=np.float32))
        ps_pk, pc_pk = _pack_pr(promo_val_select_rows), _pack_pr(promo_val_cert_rows)
        pso, pco = _predict_v12(mdl, ps_pk, 2), _predict_v12(mdl, pc_pk, 2)
        lut_pr_full = fit_weighted_venn_abers_promo_lut(pso["logit_promo"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)
        lut_pr_r1 = fit_weighted_venn_abers_promo_lut(pso["logit_promo_r1"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)
        lut_pr_r0 = fit_weighted_venn_abers_promo_lut(pso["logit_promo_r0"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)

        # Certify Rung-0 Zero-BPE Fast-Exit LTT!
        def _elig_r0(pk):
            return np.array([len(pk[3][i]) <= 160 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[3][i].lower()) for i in range(len(pk[0]))], dtype=bool)
        raw_r0_ltt = certify_fixed_sequence_dual_axis_ltt(
            _elig_r0(vs_p), vs_o["p_defer_r0"], apply_spline_temp_lut(expit(vs_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vs_o["logit_promo_r0"]), lut_pr_r0),
            vc_p[7], _elig_r0(vc_p), vc_o["p_defer_r0"], apply_spline_temp_lut(expit(vc_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vc_o["logit_promo_r0"]), lut_pr_r0),
            pc_pk[7] >= 0.65, _elig_r0(pc_pk), pco["p_defer_r0"], apply_spline_temp_lut(expit(pco["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(pco["logit_promo_r0"]), lut_pr_r0),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05, r1_cost_rel=0.04,
            d_grid=(0.15, 0.25, 0.35, 0.50), tau_a_grid=(0.04, 0.06, 0.08, 0.12, 0.16), tau_p_grid=(0.04, 0.06, 0.10, 0.15, 0.25),
        )
        r0_ltt_policy = {
            **raw_r0_ltt,
            "defer_max": raw_r0_ltt.get("r1_defer_max", 0.25),
            "tau_ai": raw_r0_ltt.get("r1_p_ai_max", 0.06),
            "tau_promo": raw_r0_ltt.get("r1_p_promo_max", 0.06),
            "max_chars": 160,
        }

        # Certify Rung-1 Prefix Fast-Exit LTT!
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
    exp_info = export_v12_modular_onnx_and_bundle_v7(mdl, out_dir, export_matmulnbits=bool(args.export_matmulnbits))
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
        "fusion_weights": {"w_t1": fw_w[0], "w_t2": fw_w[1], "w_ebt": fw_w[2]},
        "lut_ai_r0": lut_ai_r0, "lut_ai_r1": lut_ai_r1, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_r1": lut_pr_r1, "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_ltt_policy,
        "dual_axis_ltt_policy": r1_ltt_policy,
        "capabilities": ["rung0_zero_bpe_charhash", "dense_shared_quant_res", "cosine_logitnorm_v12", "logodds_3param_fusion"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_v7 = out_dir / f"meguvizcac-{args.version}-bundle-v7-sub2bit.zip"
    with ZipFile(zip_v7, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for fn in (exp_info["bundle_v7_mgvz"]["file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)
    v7_mb = zip_v7.stat().st_size / (1024.0 * 1024.0)
    if v7_mb > 2.20:
        exp_info = export_v12_modular_onnx_and_bundle_v7(mdl, out_dir, lever_w4_longstream=True, lever_w4_rung0=(v7_mb > 2.215), export_matmulnbits=bool(args.export_matmulnbits))
        with ZipFile(zip_v7, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for fn in (exp_info["bundle_v7_mgvz"]["file"], tok_path.name, "manifest.json"):
                zf.write(out_dir / fn, fn)
        v7_mb = zip_v7.stat().st_size / (1024.0 * 1024.0)

    zip_onnx = out_dir / f"meguvizcac-{args.version}-5stage-onnx-fp32exact.zip"
    with ZipFile(zip_onnx, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in (exp_info["rung0_onnx"], exp_info["rung1_onnx"], exp_info["rung2_onnx"], exp_info["rung3_onnx"], exp_info["dynaseam_onnx"]["onnx_file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)

    # Final Unpacked Evaluation (test_bili_gold v1, optional test_bili_gold_v2, test_dsh_gold, PromoBench-48, Longform-120, & V12 CI!)
    mdl_u = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v7_v12_into_model(mdl_u, out_dir / exp_info["bundle_v7_mgvz"]["file"])
    mdl_u = mdl_u.to(device).eval()

    tgo = _predict_v12(mdl_u, tg_p, 2)
    tdo = _predict_v12(mdl_u, td_p, 2)
    p_tg = apply_spline_temp_lut(expit(np.clip(_fused_lg(tgo), -30.0, 30.0)), lut_ai_full)
    p_td = apply_spline_temp_lut(expit(np.clip(_fused_lg(tdo), -30.0, 30.0)), lut_ai_full) if len(td_p[7]) > 0 else np.array([])
    tb_m = {**compute_eval_metrics(tg_p[7], p_tg), **compute_high_precision_metrics(tg_p[7], p_tg)}
    tb_r0 = compute_eval_metrics(tg_p[7], expit(tgo["logit_ai_r0"]))
    td_m = compute_eval_metrics(td_p[7], p_td) if len(td_p[7]) > 0 else {}
    tb2_m: dict[str, Any] = {}
    if tg2_p is not None:
        tg2o = _predict_v12(mdl_u, tg2_p, 2)
        p_tg2 = apply_spline_temp_lut(expit(np.clip(_fused_lg(tg2o), -30.0, 30.0)), lut_ai_full)
        tb2_m = {**compute_eval_metrics(tg2_p[7], p_tg2), **compute_high_precision_metrics(tg2_p[7], p_tg2)}

    pb_m: dict[str, Any] = {}
    pb_p = Path(__file__).resolve().parent / "data" / "v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_pk = _pre_eval_v12(pb_rows)
        pbo = _predict_v12(mdl_u, pb_pk, 2)
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
    ci_fin = run_v12_ci_gates(mdl_u, tokenizer)
    metrics = {
        "version": args.version,
        "gpu_name": gpu_name,
        "warmstart_source": loaded_v11_src,
        "gold_v2_loaded": {"train": len(gold_v2_train_rows), "val": len(gold_v2_val_rows), "test": len(gold_v2_test_rows), "wave25": w25_loaded},
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
        "v12_ci_gates": ci_fin,
        "release_gates": {
            "bundle_le_2p20_mb": v7_mb <= 2.20,
            "onnx_5stage_exact": (
                exp_info["r0_fp32_max_diff"] < 1e-4
                and exp_info["r1_fp32_max_diff"] < 1e-4
                and exp_info["r2_fp32_max_diff"] < 1e-4
                and exp_info["r3_fp32_max_diff"] < 1e-4
                and bool(exp_info["dynaseam_onnx"]["pass"])
            ),
            "v12_ci_passed": bool(ci_fin["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.2.0-ApexNano-Omni 3-Phase Trainer.")
    ap.add_argument("--smoke-test", action="store_true", help="Fast end-to-end local GPU smoke test.")
    ap.add_argument("--data", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    ap.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    ap.add_argument("--extra-pairs", type=Path, action="append", default=[])
    ap.add_argument("--v11-fp16-master", type=Path, default=Path("/content/drive/MyDrive/MeguvizCAC/v110_omniguard/v1.1.0-r1-omniguard-b6_fp16_master.pt"))
    ap.add_argument("--v11-bundle-mgvz", type=Path, default=None, help="Optional fallback MGVZ6 warm-start if FP16 master absent.")
    ap.add_argument("--v07-tokenizer", type=Path, default=None)
    ap.add_argument("--v08-gemini-gold", type=Path, default=None)
    ap.add_argument("--v08-gemini-pairs", type=Path, default=None)
    ap.add_argument("--v11-data-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--wave25-gold", type=Path, default=None)
    ap.add_argument("--gold-v2-dir", type=Path, default=None, help="Optional directory containing v12_gold_v2_{train,val,test}.jsonl.gz.")
    ap.add_argument("--out", type=Path, default=Path("scratch/v12_smoke"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.2.0-r1-apexnano-b7")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--pa-steps", type=int, default=1800)
    ap.add_argument("--pb-steps", type=int, default=2400)
    ap.add_argument("--pc-w4-steps", type=int, default=600)
    ap.add_argument("--pc-sub2-steps", type=int, default=1000)
    ap.add_argument("--pc-shock-steps", type=int, default=300)
    ap.add_argument("--export-matmulnbits", action="store_true", help="Also emit optional MatMulNBits W4 blk=128 ONNX track.")
    args = ap.parse_args()
    rep = train_v12_apexnano_full(args)
    print(json.dumps({k: v for k, v in rep.items() if k != "export_info"}, ensure_ascii=False, indent=2))
    assert all(rep["release_gates"].values()), f"Release gates failed: {rep['release_gates']}"


if __name__ == "__main__":
    main()
