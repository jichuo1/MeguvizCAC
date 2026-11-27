"""MeguvizCAC v1.2.2-ApexNano-Ultra 3-Phase Post-Training Pipeline (Colab G4 RTX PRO 6000 / A100 + Local `--smoke-test`).

Implements ALL 8 P0 upgrades from `docs/roadmap-v1.2.2-apexnano-ultra.md` (`ADR-v1.2.2`) while leaving
`v1.0.0`, `v1.1.0`, `v1.2.0`, and `v1.2.1` 100% untouched:
  - P0-1: Output-shifted Promo Beta-Spline LUTs (exact Bayes fix for `train_v121:L277`!) + multi-rung promo human ceiling.
  - P0-2: Dual-Source Surgical Re-Base (`v1.0.0-R1` AI `[0:192]` + `v1.2.1-R1` Promo/R0/LongStream `[192:256]`) + `G0` gate.
  - P0-3: 5-Era Invariance Stack (`OfflineEraGRLHead(192->5)` + `CCR Era-IPW` + 5-Era `GroupDRO` + Full-Logit Era+Topic `LEACE`).
  - P0-4: Partitioned 16k Hard-Human Memory `M = M_phys (pre_llm) U M_modern (>1y+2026)` + `SOPA-s` pAUC-DRO + `CVaR-DORO`
          ceiling across `t2 + t1 + r1 + QAT` + `v1.0`-Ref DPO on `pack3_C` 14k counterfactual mirrors & `M`.
  - P0-5: EBT `RCE+EMR` anchored energy barrier on `M` (Phase A & C, strictly `has_ai` masked) + `SoftBall` `wd=0` / `||w||_2` cap
          / `5*tanh(mil/5)` + live per-chunk `z_char` & span supervision in `ChronoStream`.
  - P0-6: Anti-Drift Sibling-Data Ingestion Curriculum (`FLOW` + A0 freeze + `<=3.5ep` cap + One-Sided Human Drift Barrier
          + Phase-A `ASAM` + hoisted same-batch Phase-B `ASAM` + live `PiKE.update()` + `EMA(0.999)` + `WiSE-FT`).
  - P0-7: `QuEST` Trust Mask (`ICML'25`) + `BitDistill` `16x16` Slot-KL (`Oct'25`) + `TR-KD` (`LS-KD + tail-pw + RankDistil + CAKLD
          + L_pAUC + L_ceil + T1-KD + 64 M replay`) + Post-Quant Convex Soup + Surgical Shock-Absorber (frozen `W8 rung0`).
  - P0-8: 5-Way Disjoint Validation (`val_select / val_meta_cal [incl. god_tier_500] / val_cal_cert [real-wild only] /
          val_promo_select / val_promo_cert`) + B=200 Bootstrap `z5_softmin` + Conformal FDR + `MoSEs` + 9-Gate CI (`V122_*`).
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
    compute_l2_sp_loss,
    compute_sub4_masked_loss,
)
from .train_v12_apexnano import compute_slot_orthogonality_loss
from .train_v121_apexnano import compute_v121_orthogonality_loss, fit_3param_logodds_fusion_v121
from .v04_corpus_prep import GROUP_TO_IDX
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    extract_style_features_40d_batch,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import MAX_LEN_V07, OmniQuantLinear
from .v08_zenith_ops import (
    ASAMPerturber,
    OfflineEraGRLHead,
    compute_ccr_era_ipw,
    eval_human_era_linear_probe_auroc,
    listnet_tail_distill_loss,
    logit_standardized_dkd_loss,
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
from .v11_omniguard import AI_EMB_END_V11, R1_PROMO_VETO_DIMS_V11, SHARED_EMB_END_V11
from .v11_promo_pseudo import _norm, _sha_norm
from .v12_apexnano import extract_rung0_charhash_batch
from .v121_apexnano import MeguvizV121ApexNanoModel, load_bundle_v7_v121_into_model
from .v121_data_refinery import is_diamond_pair_source_v121
from .v122_apexnano import (
    MeguvizV122ApexNanoModel,
    apply_v122_full_logit_leace,
    apply_z5_softmin_v122,
    bounded_fused3_logits,
    compute_bitdistill_slot_relation_kl,
    compute_wild_tail_objective_v122,
    export_v121_modular_onnx_and_bundle_v7,
    fit_conformal_fdr_thresholds_v122,
    fit_moses_register_offsets_v122,
    fit_output_shifted_promo_lut_v122,
    fit_smooth_ai_lut_v122,
    fit_z5_softmin_bootstrap_v122,
    fold_zero_op_v121,
    install_quest_trust_gradient_hooks,
    run_v122_ci_gates,
    transplant_dual_source_v10_v121_to_v122,
)
from .v122_data_refinery import (
    NUM_ERAS_V122,
    classify_epistemic_tier_v122,
    load_sibling_datafactory_v122,
    map_era_bucket_5tier,
    run_v122_refinery_ci,
)


# ---------------------------------------------------------------------------
# 1. v1.2.2 Core Post-Training Losses (SOPA-s pAUC, CVaR-DORO, RCE+EMR, TR-KD)
# ---------------------------------------------------------------------------

def _is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    if param.ndim <= 1:
        return True
    low = name.lower()
    return any(tok in low for tok in ("center", "bias", "norm", "log_r_cap", "gate", "temperature", "exit_affine", "fusion_"))


def build_v122_param_groups(
    model: MeguvizV122ApexNanoModel,
    phase: str,
    era_grl: nn.Module | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Phase-specific LR groups with strict `weight_decay=0.0` on SoftBall center/norms/biases/gates!"""
    hooks: list[Any] = []
    promo_only = (
        "promo_head.", "promo_sub4_head.", "promo_span_head.", "r1_promo_head.",
        "promo_mil_gate", "ebt_promo_gate", "exit_affine_promo_",
        "z_char_r1_promo.", "z_char_coda_promo.", "z_char_coda_promo_gate", "promo_zchar_direct.",
        "longstream.", "chunk_emb_proj.", "core.inloop_ebt.fc2_new_", "attr6_new_",
    )
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
            # A0: Trunk & top_core 100% FROZEN (`ICLR'25` spurious-forgetting guard!)
            if any(n.startswith(k) for k in promo_only):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in ai_readout):
                _add(p, 1.5e-4, wd_use)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 4.0e-4, wd_use)
            else:
                _add(p, 0.0, 0.0)
        elif phase == "phase_a1":
            if any(n.startswith(k) for k in promo_only):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in ai_readout):
                _add(p, 1.2e-4, wd_use)
            elif any(n.startswith(k) for k in top_core):
                _add(p, 3.5e-5, wd_use)
                if n in ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias"):
                    def _mk_ai_mask(g: torch.Tensor) -> torch.Tensor:
                        out = g.clone()
                        out[AI_EMB_END_V11:] = 0.0
                        return out
                    hooks.append(p.register_hook(_mk_ai_mask))
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 3.0e-4, wd_use)
            else:
                _add(p, 1.2e-5, wd_use)
        elif phase == "phase_b":
            if any(n.startswith(k) for k in promo_only):
                _add(p, 2.2e-4, wd_use)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 3.0e-4, wd_use)
            elif n in ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias"):
                _add(p, 1.2e-4, wd_use)
                def _mk_pr_mask(g: torch.Tensor) -> torch.Tensor:
                    out = g.clone()
                    out[:AI_EMB_END_V11] = 0.0
                    return out
                hooks.append(p.register_hook(_mk_pr_mask))
            elif n in ("cbm_enc.weight", "cbm_enc.bias"):
                _add(p, 1.0e-4, wd_use)
                def _mk_cbm_pr(g: torch.Tensor) -> torch.Tensor:
                    out = g.clone()
                    out[:16] = 0.0
                    return out
                hooks.append(p.register_hook(_mk_cbm_pr))
            else:
                _add(p, 0.0, 0.0)
        elif phase == "qat":
            # Freeze SoftBall.center during QAT so quantized embeddings don't chase a moving origin!
            if n in ("ai_head.center", "promo_head.center"):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in r0_pref):
                _add(p, 8.0e-5, wd_use)
            elif any(n.startswith(k) for k in (ai_readout + promo_only)):
                _add(p, 9.0e-5, wd_use)
            else:
                _add(p, 7.0e-5, wd_use)
        elif phase == "shock":
            # Surgical Shock-Absorber: FREEZE W8 rung0 + quantized weights; tune FP16 heads/norms/qres/ebt at WSD 5e-5!
            if any(n.startswith(k) for k in r0_pref) or n in ("ai_head.center", "promo_head.center"):
                _add(p, 0.0, 0.0)
            elif any(n.startswith(k) for k in (ai_readout + promo_only + ("core.qres.", "coda_norm.", "slot_norm.", "funnel_norm.", "exit_affine"))):
                _add(p, 5.0e-5, 0.0)
            else:
                _add(p, 0.0, 0.0)
        else:
            _add(p, 1.0e-4, wd_use)

    if era_grl is not None and phase in ("phase_a0", "phase_a1"):
        for p in era_grl.parameters():
            _add(p, 3.0e-4, 0.0)

    pg = [{"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]
    return pg, hooks


def compute_sopa_pauc_and_cvar_doro_loss(
    logits: torch.Tensor,
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
    is_phys_or_t0: torch.Tensor,
    sopa_lam: float = 0.5,
    doro_skip_modern: int = 4,
    pos_floor_q: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P0-4: One-way SOPA-s pAUC-DRO + Partitioned CVaR-DORO Quantile Ceiling (`T0/phys` untrimmed, `modern` DORO-trimmed!)."""
    z = logits.float()
    valid = has_ai > 0.5
    pos_m = valid & (y_ai >= 0.70)
    neg_m = valid & (y_ai <= 0.25)
    zero = z.sum() * 0.0
    if int(pos_m.sum().item()) < 2 or int(neg_m.sum().item()) < 6:
        return zero, zero

    s_pos = z[pos_m]
    s_neg = z[neg_m]
    # 1. One-Way SOPA-s pAUC-DRO (cap to top-64 negs for O(1) memory)
    k_neg = min(64, s_neg.numel())
    top_neg = torch.topk(s_neg, k=k_neg).values  # (K,)
    diff = 1.0 - (s_pos.unsqueeze(1) - top_neg.unsqueeze(0))  # (P, K)
    hinge2 = F.relu(diff).pow(2)
    l_pauc = (sopa_lam * (torch.logsumexp(hinge2 / sopa_lam, dim=1) - math.log(float(k_neg)))).mean()

    # 2. Partitioned CVaR-DORO Quantile Ceiling
    q_pos = torch.quantile(s_pos.detach(), pos_floor_q).clamp(min=0.0, max=4.0)
    neg_phys = z[neg_m & is_phys_or_t0]
    neg_mod = z[neg_m & (~is_phys_or_t0)]
    l_ceil_parts: list[torch.Tensor] = []
    if neg_phys.numel() >= 2:
        kp = min(8, neg_phys.numel())
        tp = torch.topk(neg_phys, k=kp).values
        l_ceil_parts.append(F.softplus((tp - q_pos + 1.0) / 0.5).mean())
    if neg_mod.numel() >= 6:
        km = min(28, neg_mod.numel())
        tm = torch.topk(neg_mod, k=km).values
        skip = min(int(doro_skip_modern), max(0, km - 4))
        tm_trim = tm[skip:]
        l_ceil_parts.append(F.softplus((tm_trim - q_pos + 1.0) / 0.5).mean())
    l_ceil = sum(l_ceil_parts) / float(len(l_ceil_parts)) if l_ceil_parts else zero
    return l_pauc, l_ceil


def compute_ebt_rce_emr_loss(
    ebt_energies: torch.Tensor,
    ebt_margin: torch.Tensor,
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
    is_t0: torch.Tensor,
    m_out: float = 2.5,
    m_in: float = 2.5,
) -> torch.Tensor:
    """P0-5: Anchored Energy-Margin Regularization (`RCE + EMR`, ICLR'25) strictly masked by `has_ai > 0.5`!"""
    valid = has_ai > 0.5
    if not valid.any():
        return ebt_margin.sum() * 0.0
    m = ebt_margin.float()
    e = ebt_energies.float()
    hu_m = valid & (y_ai <= 0.25)
    ai_m = valid & (y_ai >= 0.70) & is_t0
    loss = m.sum() * 0.0
    if int(hu_m.sum().item()) >= 4:
        kh = min(12, int(hu_m.sum().item()))
        top_hu = torch.topk(m[hu_m], k=kh).values
        loss = loss + 0.35 * F.relu(top_hu + m_out).pow(2).mean()
    if int(ai_m.sum().item()) >= 2:
        loss = loss + 0.25 * F.relu(m_in - m[ai_m]).pow(2).mean()
    # Explicit LSE energy scale lock prevents unbounded linear drift!
    loss = loss + 0.02 * torch.logsumexp(-e[valid], dim=-1).pow(2).mean() + 0.005 * F.relu(e[valid].abs() - 6.0).pow(2).mean()
    return loss


def compute_promo_cvar_ceiling_loss(
    promo_logits: torch.Tensor,
    y_promo: torch.Tensor,
    has_promo: torch.Tensor,
) -> torch.Tensor:
    """P0-1: Multi-rung confirmed-organic human promo ceiling (keeps human promo baseline cleanly < 0.02!)."""
    org_m = (has_promo > 0.5) & (y_promo <= 0.15)
    n_org = int(org_m.sum().item())
    if n_org < 4:
        return promo_logits.sum() * 0.0
    k = min(8, max(2, n_org // 10))
    u_top = torch.topk(promo_logits.float()[org_m], k=k).values
    return F.relu(u_top + 1.5).pow(2).mean() + 0.20 * F.softplus(u_top + 1.0).mean()


def compute_tail_ranking_kd_loss(
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    y_ai: torch.Tensor,
    has_ai: torch.Tensor,
) -> torch.Tensor:
    """P0-7: Tail-Ranking KD (`LS-KD` + tail-weighted pointwise + `RankDistil` top-32 pairwise + `CAKLD` BKL)."""
    valid = has_ai > 0.5
    if int(valid.sum().item()) < 4:
        return z_s.sum() * 0.0
    zs = z_s.float()[valid]
    zt = z_t.float().detach()[valid].clamp(-10.0, 10.0)
    yv = y_ai.float()[valid]

    # 1. Standardized logit MSE (LS-KD, CVPR'24)
    zs_n = (zs - zs.mean()) / zs.std().clamp(min=0.25)
    zt_n = (zt - zt.mean()) / zt.std().clamp(min=0.25)
    l_std = F.mse_loss(zs_n, zt_n)

    # 2. Tail-weighted pointwise MSE
    pt = torch.sigmoid(zt)
    sig_prime = pt * (1.0 - pt)
    is_hard_neg = (yv <= 0.25) & (zt >= -1.5)
    is_hard_pos = (yv >= 0.70) & (zt <= 2.5)
    w_pw = 1.0 + 4.0 * (is_hard_neg | is_hard_pos).float() + 2.0 * sig_prime
    l_pw = ((zs - zt).pow(2) * w_pw).mean()

    # 3. RankDistil pairwise margin preservation on top-32 region
    k_r = min(32, zs.numel())
    top_idx = torch.topk(zt, k=k_r).indices
    zs_k, zt_k = zs[top_idx], zt[top_idx]
    dt = zt_k.unsqueeze(1) - zt_k.unsqueeze(0)
    ds = zs_k.unsqueeze(1) - zs_k.unsqueeze(0)
    mask_p = dt > 0.25
    l_rank = F.softplus((dt[mask_p] - ds[mask_p]) / 0.5).mean() if mask_p.any() else zs.sum() * 0.0

    # 4. CAKLD confidence-weighted binary KL at T=2
    ps2 = torch.sigmoid(zs / 2.0).clamp(1e-5, 1.0 - 1e-5)
    pt2 = torch.sigmoid(zt / 2.0).clamp(1e-5, 1.0 - 1e-5)
    conf_w = (2.0 * pt2 - 1.0).abs().clamp(min=0.25)
    bkl = (pt2 * torch.log(pt2 / ps2) + (1.0 - pt2) * torch.log((1.0 - pt2) / (1.0 - ps2))) * conf_w
    l_bkl = bkl.mean() * 4.0
    return 0.30 * l_std + 0.35 * l_pw + 0.35 * l_rank + 0.25 * l_bkl


# ---------------------------------------------------------------------------
# 2. Main v1.2.2-ApexNano-Ultra 3-Phase Training Pipeline
# ---------------------------------------------------------------------------

def train_v122_apexnano_full(args) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ref_ci = run_v122_refinery_ci()
    assert ref_ci["pass"], f"v122_data_refinery CI failed: {ref_ci}"

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
        f"[v1.2.2-ApexNano-Ultra Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={bs} | "
        f"v10={args.v10_fp16_master or args.v10_bundle_mgvz} | v121={args.v121_fp16_master or args.v121_bundle_mgvz} | "
        f"sibling={args.sibling_dir} | gold_v2={args.gold_v2_dir}",
        flush=True,
    )

    def _read_jsonl_list(p: Path) -> list[dict[str, Any]]:
        if not p.is_file():
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
        rng_sm = random.Random(20261127)
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

    # 3. Hydrate `score_source` + `created_ts` + Gemini Pro/Flash + Gold-v2 + Wave-3 + Sibling Datafactory!
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

    # Load ALL 6 files from `v122-sibling-datafactory`!
    sib = load_sibling_datafactory_v122(args.sibling_dir.resolve() if args.sibling_dir else None, smoke_test=bool(args.smoke_test))
    quarantine_sids: set[str] = sib["quarantine_sids"]
    replace_text_by_sid: dict[str, str] = sib["replace_text_by_sid"]
    override_sids |= sib["override_sids"]
    print(
        f"[Stage0-Sibling+GoldV2+Wave3] sib={dict(sib['stats'])} | g2_tr={len(gold_v2_train_rows)} g2_va={len(gold_v2_val_rows)} "
        f"g2_te={len(gold_v2_test_rows)} | w3_tr={len(wave3_train_rows)} w3_va={len(wave3_val_rows)} | overrides={len(override_sids):,}",
        flush=True,
    )

    # Apply `v122_data_refinery` + Sibling Hygiene (`quarantine_exclude` & `replace_cleaned_text`) to base train_raw
    refined_items: list[dict[str, Any]] = []
    t3_dropped = hyg_purged = hyg_patched = ov_deduped = 0
    tier_counts: Counter = Counter()
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
        tier_counts[tier] += 1
        if tier == "T3_QUARANTINE" or tw <= 0.0:
            t3_dropped += 1
            continue
        it["y_score"] = float(np.clip(eff_sc / 100.0, 0.0, 1.0))
        it["has_ai"] = 1.0
        it["epistemic_tier"] = tier
        it["is_true_pro_gold"] = bool(tier == "T0_DIAMOND_GOLD")
        it["is_phys_prellm"] = bool(src_it == "pre_llm_human_prior" or it.get("is_pre_llm"))
        it["era5"] = map_era_bucket_5tier(it.get("era_bucket"), orig_ts_by_sid.get(sid_it), is_pre_llm=it["is_phys_prellm"], score_source=src_it)
        it["weight"] = float(it["weight"]) * float(tw)
        it["y_promo"] = -1.0
        it["has_promo"] = 0.0
        it["sub4_idx"] = -1
        it["is_rescue"] = False
        it["l_pceil_admit"] = False
        refined_items.append(it)

    # Carve out strictly REAL-WILD never-trained `cert_holdout_rows` (ZERO synthetic contamination for valid LTT!)
    rng_cert = random.Random(2026112701)
    def _is_real_wild_bili(it: dict[str, Any]) -> bool:
        src = str(it.get("score_source", "")).lower()
        coh = str(it.get("cohort", "")).lower()
        if any(tok in src for tok in ("synth", "plugin", "deepseek", "qwen", "inkling")):
            return False
        return (coh in ("", "bili", "v04_1m", "real_bili")) or it.get("is_phys_prellm", False)

    cert_pos_idx = [i for i, it in enumerate(refined_items) if it["y_score"] >= 0.80 and _is_real_wild_bili(it)]
    cert_neg_idx = [i for i, it in enumerate(refined_items) if it["y_score"] <= 0.15 and _is_real_wild_bili(it)]
    if len(cert_pos_idx) < (16 if args.smoke_test else 400):
        cert_pos_idx = [i for i, it in enumerate(refined_items) if it["y_score"] >= 0.80]
    rng_cert.shuffle(cert_pos_idx)
    rng_cert.shuffle(cert_neg_idx)
    take_c_pos = min(16 if args.smoke_test else 750, max(1, len(cert_pos_idx) // 3))
    take_c_neg = min(32 if args.smoke_test else 3000, max(1, len(cert_neg_idx) // 3))
    cert_drop = set(cert_pos_idx[:take_c_pos]) | set(cert_neg_idx[:take_c_neg])
    cert_holdout_rows = [
        {"text": refined_items[i]["raw_text"], "score": float(refined_items[i]["y_score"]) * 100.0, "sample_id": refined_items[i].get("sample_id", f"cert_{i}")}
        for i in sorted(cert_drop)
    ]
    refined_items = [it for i, it in enumerate(refined_items) if i not in cert_drop]

    ipw_w = compute_joint_length_domain_ipw([{"text": x["raw_text"], "y_score": x["y_score"]} for x in refined_items])
    for it, iw in zip(refined_items, ipw_w):
        it["weight"] = float(np.clip(float(it["weight"]) * float(iw), 0.20, 3.0))
    print(
        f"[Stage0-V122Refinery] kept={len(refined_items):,} | cert_real_wild={len(cert_holdout_rows):,} (pos={take_c_pos}, neg={take_c_neg}) | "
        f"hyg_purged={hyg_purged:,} hyg_patched={hyg_patched:,} T3_purged={t3_dropped:,} overrides={ov_deduped:,}",
        flush=True,
    )

    # 4. Construct Feature-Encoded Items Across ALL Pools (with true 5-Era mapping & strict has_ai/has_promo!)
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
        era5: int = -1, is_phys: bool = False,
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
            "is_phys_prellm": bool(is_phys or era5 == 0),
            "l_pceil_admit": bool(l_pceil), "attr5": 3 if y_ai >= 0.70 else (1 if y_ai > 0.30 else 0),
            "group_idx": GROUP_TO_IDX["current_2026_real"], "era5": int(era5),
            "epistemic_tier": "T0_DIAMOND_GOLD" if is_t0 else "T1_VERIFIED_CLEAN",
            "code_idx": -100, "prag_vec": [0.0] * len(PRAG_TAGS),
        }

    dual_gold_train: list[dict[str, Any]] = []
    wild_promo_focus_pool: list[dict[str, Any]] = []
    promo_val_select_rows: list[dict[str, Any]] = []
    promo_val_cert_rows: list[dict[str, Any]] = []
    uot_pseudo_rows: list[dict[str, Any]] = []
    hard_neg_rows: list[dict[str, Any]] = []
    new_gold_ai_pool: list[dict[str, Any]] = []
    protect_pool: list[dict[str, Any]] = []

    # 4a. Legacy Wave-1 / Wave-2 / Wave-2.5 / Gold-v2 / Wave-3
    for fname, is_resc in (("v11_wave1_dualaxis_gold_all5880.jsonl.gz", False), ("v11_wave2_dualaxis_gold_all4410.jsonl.gz", True)):
        rows_g = _read_jsonl_list(v11_dir / fname)[: (120 if args.smoke_test else 999999)]
        for r in rows_g:
            txt = str(r["text"])
            sh = _sha_norm(_norm(txt))
            y_pr_v = float(r["promo_score"]) / 100.0
            e5 = map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"))
            rec = _make_item(txt, float(r["ai_score"]) / 100.0, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.4, has_ai=1.0, is_t0=True, is_rescue=is_resc, era5=e5)
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

    if args.wave25_gold and args.wave25_gold.is_file():
        for r in _read_jsonl_list(args.wave25_gold):
            y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
            rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.6, has_ai=1.0, is_t0=True, is_rescue=True)
            dual_gold_train.append(rec); new_gold_ai_pool.append(rec); wild_promo_focus_pool.append(rec)
            if y_pr_v >= 0.65:
                wild_promo_focus_pool.append(rec)

    for r in gold_v2_train_rows:
        y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
        tier_g2, _, _, _ = classify_epistemic_tier_v122(float(r["ai_score"]), "v12_gold_v2_pro", str(r["text"]), v11_or_v12_verified=True, is_hybrid_mixed=bool(r.get("is_hybrid_mixed", False)))
        e5 = map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"))
        rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.55 if tier_g2 == "T0_DIAMOND_GOLD" else 0.6, has_ai=1.0, is_t0=(tier_g2 == "T0_DIAMOND_GOLD"), is_rescue=True, era5=e5)
        dual_gold_train.append(rec)
        if tier_g2 == "T0_DIAMOND_GOLD":
            new_gold_ai_pool.append(rec)
            if y_ai_v <= 0.25 and len(str(r["text"])) >= 35:
                protect_pool.append(rec)

    for r in wave3_train_rows:
        y_ai_v, y_pr_v = float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0
        bkt = str(r.get("bucket", ""))
        rec = _make_item(str(r["text"]), y_ai_v, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.65, has_ai=1.0, is_t0=True, is_rescue=True, l_pceil=(y_pr_v <= 0.20))
        dual_gold_train.append(rec); new_gold_ai_pool.append(rec)
        if bkt.startswith("P") or y_pr_v >= 0.65:
            wild_promo_focus_pool.append(rec)
            if y_pr_v >= 0.65:
                wild_promo_focus_pool.append(rec)
        if y_ai_v <= 0.25 and (bkt.startswith("A3") or len(str(r["text"])) >= 35):
            protect_pool.append(rec)
    for idx_w3v, r in enumerate(wave3_val_rows):
        rec_w3v = _make_item(str(r["text"]), float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0, 1.0, str(r.get("promo_type", "none")), 1.0, has_ai=1.0, is_t0=True)
        (promo_val_select_rows if idx_w3v % 2 == 0 else promo_val_cert_rows).append(rec_w3v)

    # 4b. SIBLING DATAFACTORY INGESTION (99k dual-axis gold + 66k real AI-only + 17.7k clean synth + 14.2k counterfactual mirrors!)
    sib_dual_n = sib_old_n = sib_pr_n = sib_ai_only_n = 0
    for r in sib["dual_train_rows"]:
        txt = str(r["text"])
        y_ai_v = float(r["ai_score"]) / 100.0
        y_pr_v = float(r["promo_score"]) / 100.0 if r.get("has_promo") and r.get("promo_score") is not None else -1.0
        has_pr_v = 1.0 if (r.get("has_promo") and r.get("promo_score") is not None) else 0.0
        src_s = str(r.get("source_tag") or "sibling_dual")
        e5 = map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"), score_source=src_s)
        tier_s, tw_s, _, _ = classify_epistemic_tier_v122(float(r["ai_score"]), src_s, txt, is_hybrid_mixed=bool(r.get("is_hybrid_mixed", False)), judge_conf=r.get("judge_conf"))
        if tier_s == "T3_QUARANTINE":
            continue
        is_org_hn = ("organic_brand" in src_s) or (has_pr_v > 0.5 and y_pr_v <= 0.15)
        rec = _make_item(txt, y_ai_v, max(0.0, y_pr_v), has_pr_v, str(r.get("promo_type", "none")), tw_s, has_ai=1.0, is_t0=(tier_s == "T0_DIAMOND_GOLD"), is_rescue=True, l_pceil=is_org_hn, era5=e5, is_phys=(e5 == 0))
        dual_gold_train.append(rec)
        new_gold_ai_pool.append(rec)
        sib_dual_n += 1
        if y_ai_v <= 0.25 and (e5 in (0, 1, 2, 3) or "oldvideo" in src_s or "trap" in src_s or (r.get("bucket_meta") or {}).get("style_tag") in ("scorecard", "literary", "tech_review")):
            protect_pool.append(rec)
            sib_old_n += 1
        if has_pr_v > 0.5 and (y_pr_v >= 0.65 or "organic_brand" in src_s or "wild_promo" in src_s):
            wild_promo_focus_pool.append(rec)
            if y_pr_v >= 0.65:
                wild_promo_focus_pool.append(rec)
                sib_pr_n += 1

    for r in sib["ai_only_real_rows"]:
        txt = str(r["text"])
        y_ai_v = float(r["ai_score"]) / 100.0
        src_s = str(r.get("source_tag") or "sibling_ai_only")
        e5 = map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"), score_source=src_s)
        bm = r.get("bucket_meta") or {}
        rec = _make_item(txt, y_ai_v, -1.0, 0.0, "none", 1.35, has_ai=1.0, is_t0=True, is_rescue=True, era5=e5, is_phys=(e5 == 0))
        new_gold_ai_pool.append(rec)
        sib_ai_only_n += 1
        if y_ai_v <= 0.25 and (e5 in (0, 1, 2, 3) or bool(bm.get("is_trap_human")) or bool(bm.get("is_old_video"))):
            protect_pool.append(rec)

    for r in sib["ai_only_synth_rows"]:
        txt = str(r["text"])
        rec = _make_item(txt, float(r["ai_score"]) / 100.0, -1.0, 0.0, "none", 0.90, has_ai=1.0, is_t0=False, is_rescue=False, era5=-1, is_phys=False)
        new_gold_ai_pool.append(rec)

    # 4c. Promo counterfactual pairs, UOT pseudo, & 2k k-NN hard negs (ALL `has_ai=0.0`!)
    dual_pairs_train: list[dict[str, Any]] = []
    for x in _read_jsonl_list(v11_dir / "v11_wave1_promo_counterfactual_pairs_1600.jsonl")[: (48 if args.smoke_test else 99999)]:
        sh_p = _sha_norm(_norm(x["promo_text"]))
        ia, pt_s = (1.0 if x.get("promo_is_ai") else 0.0), str(x.get("promo_type", "soft_shill"))
        if sh_p in cert_shas:
            promo_val_cert_rows.append(_make_item(x["promo_text"], ia, 0.95, 1.0, pt_s, 1.0, has_ai=0.0, is_t0=True))
            promo_val_cert_rows.append(_make_item(x["organic_text"], ia, 0.02, 1.0, "none", 1.0, has_ai=0.0, is_t0=True))
        elif sh_p in select_shas:
            promo_val_select_rows.append(_make_item(x["promo_text"], ia, 0.95, 1.0, pt_s, 1.0, has_ai=0.0, is_t0=True))
            promo_val_select_rows.append(_make_item(x["organic_text"], ia, 0.02, 1.0, "none", 1.0, has_ai=0.0, is_t0=True))
        else:
            dual_pairs_train.append(x)
            dual_gold_train.append(_make_item(x["promo_text"], ia, 0.95, 1.0, pt_s, 1.4, has_ai=0.0, is_t0=True))
            dual_gold_train.append(_make_item(x["organic_text"], ia, 0.02, 1.0, "none", 1.4, has_ai=0.0, is_t0=True, l_pceil=True))

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
        hard_neg_rows.append(_make_item(str(r["text"]), float(r.get("old_ai_score", 15.0)) / 100.0, 0.02, 1.0, "none", 1.0, has_ai=0.0, l_pceil=bool(r.get("l_pceil_admitted", False))))

    # Encode base `refined_items` + apply CCR 5-Era IPW on humans (`era in {0,1,2,3}` vs `era==4`)!
    t0_pool, t1_pool, t2_pool = [], [], []
    struct_gids = {GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["dsh_hard_slice"], GROUP_TO_IDX["multilingual_world_knowledge"]}
    for it in refined_items:
        cl, f56, meta = extract_style_features_56d(it["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        it["clean_text"], it["ids"], it["style"], it["spoof_any"] = cl, ids, f56, float(meta["spoof_any"])
        it["engram"] = make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
        it["trigram"] = make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        it["attr5"] = _derive_attr5_label(it)
        it["code_idx"] = int(it.get("code_idx", -100))
        it["prag_vec"] = it.get("prag_vec") or [0.0] * len(PRAG_TAGS)
        if it.get("is_trap_human") or it.get("is_phys_prellm") or it.get("group_idx") in struct_gids:
            protect_pool.append(it)
        if it["epistemic_tier"] == "T0_DIAMOND_GOLD":
            t0_pool.append(it)
        elif it["epistemic_tier"] == "T2_SOFT_MIDBAND":
            t2_pool.append(it)
        else:
            t1_pool.append(it)

    # P0-3: Apply CCR Era-IPW (`pre_llm/old` vs `2026` humans) across base + sibling pools!
    all_era_items = t0_pool + t1_pool + new_gold_ai_pool
    if len(all_era_items) >= 400:
        st_era = np.stack([x["style"] for x in all_era_items], axis=0)
        e_bin = np.array([0 if x["era5"] in (0, 1, 2, 3) else (3 if x["era5"] == 4 else -1) for x in all_era_items], dtype=np.int32)
        y_era = np.array([x["y_score"] for x in all_era_items], dtype=np.float32)
        ccr_w = compute_ccr_era_ipw(st_era, e_bin, y_era, clip_lo=0.25, clip_hi=4.0)
        for it, cw in zip(all_era_items, ccr_w):
            it["weight"] = float(np.clip(it["weight"] * float(cw), 0.20, 3.5))

    # Build ChronoStream long docs (Wave-2 1,200 + `pack3_C` 4,824 exact-span sandwich hybrids!)
    wave2_longform_all = _read_jsonl_list(v11_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (20 if args.smoke_test else 9999)]
    wave2_longform = [d for i, d in enumerate(wave2_longform_all) if i % 10 != 9]
    wave2_longform_val = [d for i, d in enumerate(wave2_longform_all) if i % 10 == 9]
    for sw in sib["sandwich_span_rows"]:
        wave2_longform.append({
            "text": sw["text"], "ai_score": sw["ai_score"], "promo_score": 0.0,
            "is_hybrid_mixed": True, "ai_spans": sw["ai_spans"], "promo_spans": [],
        })

    hu_clean = [x["clean_text"] for x in (t0_pool + t1_pool + new_gold_ai_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.15 and 40 <= len(x["clean_text"]) <= 220][:4000]
    hu_hard = [x["clean_text"] for x in protect_pool if x["has_ai"] > 0.5 and x["y_score"] <= 0.20 and len(x["clean_text"]) >= 40][:2000] or hu_clean[:500]
    ai_clean = [x["clean_text"] for x in (t0_pool + t1_pool + new_gold_ai_pool) if x["has_ai"] > 0.5 and x["y_score"] >= 0.85 and 40 <= len(x["clean_text"]) <= 240][:4000]
    pr_clean = [x["clean_text"] for x in dual_gold_train if x["has_promo"] > 0.5 and x["y_promo"] >= 0.70 and x["y_score"] <= 0.30][:1000] or hu_clean[:50]
    longmix_syn = OnlineLongMixSynthesizerV11(hu_clean[:2000] or ["真人评论测试。"] * 12, hu_hard, ai_clean[:2000] or ["人工智能评论测试。"] * 12, pr_clean)

    # Combine legacy diamond pairs + sibling `pack3_C` 14,203 counterfactual mirrors for Anchor-Ref DPO (`L_ref`)!
    cf_mirror_pool = list(sib["cf_mirror_pairs"])
    for lp in legacy_pairs:
        at = str(lp.get("ai_text") or lp.get("pos_text") or "").strip()
        ht = str(lp.get("human_text") or lp.get("neg_text") or "").strip()
        if at and ht:
            cf_mirror_pool.append({"ai_text": at, "human_text": ht, "mode": "legacy_diamond", "pair_weight": 1.0})

    print(
        f"[Stage0-Pools] legacy_t0={len(t0_pool):,} t1={len(t1_pool):,} | new_gold={len(new_gold_ai_pool):,} (sib_dual={sib_dual_n:,}, "
        f"sib_old_prot={sib_old_n:,}, sib_ai_only={sib_ai_only_n:,}) | protect={len(protect_pool):,} | wild_promo={len(wild_promo_focus_pool):,} | "
        f"cf_mirrors={len(cf_mirror_pool):,} | long_tr={len(wave2_longform):,} | pv_sel={len(promo_val_select_rows):,} pv_cert={len(promo_val_cert_rows):,}",
        flush=True,
    )

    # 5. P0-8: Strictly 5-Way Disjoint Validation Partition (`val_select` / `val_meta_cal` / `val_cal_cert` / `pv_sel` / `pv_cert`)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    vg_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_gold}
    val_cal_all, _ = build_stratified_val_calibration_set(eval_rows, target_n=1600)
    val_cal_nodup = [r for r in val_cal_all if str(r.get("sample_id", r.get("id", ""))) not in vg_ids] or val_cal_all

    # Split `val_cal_nodup` cleanly in half: `val_select_wild` (ckpt selection ONLY) vs `val_meta_wild` (LEACE + z5_softmin + LUT ONLY)!
    rng_sp = random.Random(2026112702)
    vc_shuf = list(val_cal_nodup)
    rng_sp.shuffle(vc_shuf)
    half_n = len(vc_shuf) // 2
    val_select_wild = vc_shuf[:half_n] or vc_shuf
    val_meta_wild = vc_shuf[half_n:] or vc_shuf

    # `val_meta_cal` combines `val_meta_wild` + `val_gold` + NEVER-TRAINED `god_tier_500_holdout` (with real 5-era metadata!)
    god500_eval_rows = [
        {
            "text": str(r["text"]),
            "score": float(r["ai_score"]),
            "promo_score": float(r.get("promo_score", 0.0)),
            "era5": map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts")),
            "sample_id": str(r.get("sample_id", "")),
        }
        for r in sib["god_tier_500_holdout"]
        if r.get("has_ai") and r.get("ai_score") is not None
    ]
    val_meta_cal_rows = [
        {"text": str(r["text"]), "score": float(r["score"]), "era5": map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=bool(r.get("is_pre_llm")))}
        for r in (val_meta_wild + val_gold)
    ] + god500_eval_rows

    used_ids = vg_ids | {str(r.get("sample_id", r.get("id", ""))) for r in val_cal_nodup}
    val_rem = [r for r in eval_rows if r.get("split") == "validation" and str(r.get("sample_id", r.get("id", ""))) not in used_ids]
    rng_v = random.Random(2026112703)
    vr_pos = [r for r in val_rem if float(r.get("score", 0.0)) >= 70.0]
    vr_neg = [r for r in val_rem if float(r.get("score", 0.0)) <= 30.0]
    rng_v.shuffle(vr_pos); rng_v.shuffle(vr_neg)
    val_cal_cert = vr_pos[:250] + vr_neg[:1000] + cert_holdout_rows
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
        god500_eval_rows = _cap_s(god500_eval_rows, 48)

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
        e5 = np.array([int(r.get("era5", map_era_bucket_5tier(r.get("era_bucket"), r.get("created_ts"), is_pre_llm=bool(r.get("is_pre_llm"))))) for r in rows], dtype=np.int32)
        st_arr = np.stack(st_l, axis=0) if st_l else np.zeros((0, STYLE_FEAT_DIM_V11), dtype=np.float32)
        return (encs, engs, tris, txts, cleans, st_arr, metas, sc, e5)

    vs_sel_p = _pre_eval(val_select_wild)
    vg2_p = _pre_eval(gold_v2_val_rows, "ai_score") if gold_v2_val_rows else None
    vmc_p = _pre_eval(val_meta_cal_rows)
    vcert_p = _pre_eval(val_cal_cert)
    god500_p = _pre_eval(god500_eval_rows) if god500_eval_rows else None
    tg_p, td_p = _pre_eval(test_gold), _pre_eval(test_dsh)
    tg2_p = _pre_eval(gold_v2_test_rows, "ai_score") if gold_v2_test_rows else None
    print(
        f"[Stage0-5WayVal] val_select_wild={len(vs_sel_p[0])} vg2={len(vg2_p[0]) if vg2_p else 0} | "
        f"val_meta_cal={len(vmc_p[0])} (god500={len(god500_eval_rows)}) | "
        f"val_cal_cert={len(vcert_p[0])} (n_pos_ai={int((vcert_p[7]>=70).sum())})",
        flush=True,
    )

    # 6. P0-2: Dual-Source Surgical Re-Base (`v1.0.0-R1` AI + `v1.2.1-R1` Promo/R0/LongStream) + `G0` Gate!
    mdl = MeguvizV122ApexNanoModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS),
    ).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_EMB_END_V11, num_eras=NUM_ERAS_V122).to(device)

    @torch.no_grad()
    def _predict_v122(m_e: MeguvizV122ApexNanoModel, pk, outer_steps: int = 2) -> dict[str, np.ndarray]:
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
            vals = {
                "logit": o["logit"], "raw_logit": o["raw_logit"], "ai_aux_term": o["ai_aux_term"],
                "logit_t1": o["exit1_out"]["logit"], "raw_logit_t1": o["exit1_out"]["raw_logit"],
                "logit_r1": o["logit_r1"], "raw_logit_r1": o["raw_logit_r1"],
                "logit_promo": o["logit_promo"], "raw_logit_promo": o["raw_logit_promo"],
                "logit_promo_t1": o["exit1_out"]["logit_promo"], "raw_logit_promo_t1": o["exit1_out"]["raw_logit_promo"],
                "logit_promo_r1": o["logit_promo_r1"], "raw_logit_promo_r1": o["raw_logit_promo_r1"],
                "logit_ai_r0": o["logit_ai_r0"], "logit_promo_r0": o["logit_promo_r0"], "p_defer_r0": o["p_defer_r0"],
                "p_defer1": o["p_defer1"], "r1_span_max": o["r1_span_max"], "ebt_margin": o["ebt_margin"], "emb": o["emb"],
            }
            for k, v in vals.items():
                arr = v.float().cpu().numpy()
                for ri, oi in enumerate(mb):
                    out[k][oi] = arr[ri]
        return out

    # Load `v1.0.0-R1` AI Teacher & `v1.2.1-R1` Promo/R0/LongStream Teacher
    v10_teacher: MeguvizV10ApexLoopModel | None = None
    v121_teacher: MeguvizV121ApexNanoModel | None = None

    if (args.v10_fp16_master and args.v10_fp16_master.is_file()) or (args.v10_bundle_mgvz and args.v10_bundle_mgvz.is_file()):
        v10_teacher = MeguvizV10ApexLoopModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).cpu().eval()
        if args.v10_fp16_master and args.v10_fp16_master.is_file():
            v10_teacher.load_state_dict(torch.load(args.v10_fp16_master, map_location="cpu"), strict=False)
        else:
            load_bundle_v6_mgvz_into_model(v10_teacher, args.v10_bundle_mgvz)
        v10_teacher = v10_teacher.to(device).eval()
        for p in v10_teacher.parameters():
            p.requires_grad_(False)

    if (args.v121_fp16_master and args.v121_fp16_master.is_file()) or (args.v121_bundle_mgvz and args.v121_bundle_mgvz.is_file()):
        v121_teacher = MeguvizV121ApexNanoModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).cpu().eval()
        v121_teacher.set_quant_mode("fp16")
        if args.v121_fp16_master and args.v121_fp16_master.is_file():
            v121_teacher.load_state_dict(torch.load(args.v121_fp16_master, map_location="cpu"), strict=False)
        else:
            load_bundle_v7_v121_into_model(v121_teacher, args.v121_bundle_mgvz)
        v121_teacher = v121_teacher.to(device).eval()
        for p in v121_teacher.parameters():
            p.requires_grad_(False)

    # Calibrate 40->56 bridge + OBS perm + SoftBall center on clean items
    cal_items = (t0_pool[:250] + new_gold_ai_pool[:250] + t1_pool[:300])[: (64 if args.smoke_test else 768)]
    cal_txts = [x["raw_text"] for x in cal_items]
    _, f40_cal, _ = extract_style_features_40d_batch(cal_txts)
    f56_cal = np.stack([x["style"] for x in cal_items], axis=0)

    def _collate_items(batch_items: list[dict[str, Any]]):
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
            torch.tensor([1.0 if x["l_pceil_admit"] else 0.0 for x in batch_items], dtype=torch.float32, device=device),
            torch.tensor([int(x.get("era5", -1)) for x in batch_items], dtype=torch.long, device=device),
        )

    @torch.no_grad()
    def _predict_v10_logits_and_emb(txts_l: list[str], encs_l, engs_l, tris_l) -> tuple[np.ndarray, np.ndarray]:
        assert v10_teacher is not None
        _, f40_l, _ = extract_style_features_40d_batch(txts_l)
        n = len(txts_l)
        lg = np.zeros(n, dtype=np.float32)
        em = np.zeros((n, 256), dtype=np.float32)
        for st_i in range(0, n, 128):
            ed = min(n, st_i + 128)
            ml = max(2, max(len(encs_l[i]) for i in range(st_i, ed)))
            ids = np.full((ed - st_i, ml), tokenizer.pad_id, dtype=np.int64)
            msk = np.zeros((ed - st_i, ml), dtype=np.int64)
            eg = np.zeros((ed - st_i, ml), dtype=np.int64)
            tg = np.zeros((ed - st_i, ml), dtype=np.int64)
            for ri, oi in enumerate(range(st_i, ed)):
                s = encs_l[oi]
                ids[ri, : len(s)], msk[ri, : len(s)], eg[ri, : len(s)], tg[ri, : len(s)] = s, 1, engs_l[oi], tris_l[oi]
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                o = v10_teacher(
                    torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
                    torch.from_numpy(f40_l[st_i:ed]).to(device),
                    engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device),
                    outer_steps=2, k_inner=2, return_aux=True,
                )
            lg[st_i:ed] = o["logit"].float().cpu().numpy()
            em[st_i:ed] = o["emb"].float().cpu().numpy()
        return lg, em

    from scipy.stats import spearmanr
    transplant_rep: dict[str, Any] = {"mode": "cold_init"}
    g0_rmse = 0.0
    g0_spearman: float | None = None
    if v10_teacher is not None:
        _, emb_v10_cal = _predict_v10_logits_and_emb(cal_txts, [x["ids"] for x in cal_items], [x["engram"] for x in cal_items], [x["trigram"] for x in cal_items])
        transplant_rep = transplant_dual_source_v10_v121_to_v122(v10_teacher, v121_teacher, mdl, f40_cal, f56_cal, emb_v10_cal)
        lg_v10_vs, _ = _predict_v10_logits_and_emb(vs_sel_p[3], vs_sel_p[0], vs_sel_p[1], vs_sel_p[2])
        o_init_vs = _predict_v122(mdl, vs_sel_p, 2)
        g0_rmse = float(np.sqrt(np.mean((o_init_vs["logit"] - lg_v10_vs) ** 2)))
        g0_spearman = float(spearmanr(o_init_vs["logit"], lg_v10_vs).statistic)
        transplant_rep["g0_rmse_vs_v10"] = round(g0_rmse, 5)
        transplant_rep["g0_spearman_vs_v10"] = round(g0_spearman, 5)
    elif v121_teacher is not None:
        mdl.load_state_dict(v121_teacher.state_dict(), strict=False)
        transplant_rep = {"mode": "v121_only_fallback"}

    w_ai_init_norm = float(mdl.ai_head.weight.detach().float().norm().item())
    g0_ci = run_v122_ci_gates(mdl, tokenizer, transplant_rmse=g0_rmse, transplant_spearman=g0_spearman)
    print(f"[Stage0-DualTransplant] {json.dumps(transplant_rep)} | G0_CI={'PASS' if g0_ci['all_ci_passed'] else 'WARN'} (pad_max={g0_ci['V122_P1_batch_invariance']['max_diff']})", flush=True)

    # Anchor teacher for AI (`v1.0.0-R1` via frozen post-transplant `v122_anchor` so shapes/56d/R0 match 100%!)
    # If `g0_rmse > 0.08` (from MoLE->Dense collapse) and not smoke-test, run fast 200-step collapse-heal first!
    if v10_teacher is not None and g0_rmse > 0.08 and not args.smoke_test:
        print(f"[Stage0-CollapseHeal] g0_rmse={g0_rmse:.4f} > 0.08 -> running 200-step frozen-promo TR-KD heal to v1.0...", flush=True)
        pg_h, hk_h = build_v122_param_groups(mdl, "phase_a1")
        opt_h = torch.optim.AdamW(pg_h, betas=(0.9, 0.95))
        heal_pool = (t0_pool[:20000] + t1_pool[:20000] + protect_pool[:10000])
        rng_h = random.Random(2026112704)
        mdl.train()
        for hs in range(200):
            hb = rng_h.sample(heal_pool, min(bs, len(heal_pool)))
            b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf, y_ai, has_ai, *_ = _collate_items(hb)
            _, f40_b, _ = extract_style_features_40d_batch([x["raw_text"] for x in hb])
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                ov10 = v10_teacher(b_ids, b_m, torch.from_numpy(f40_b).to(device), engram_ids=b_e, trigram_ids=b_t, outer_steps=2, k_inner=2, return_aux=True)
            opt_h.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                om = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
                lh = (
                    compute_tail_ranking_kd_loss(om["logit"], ov10["logit"], y_ai, has_ai)
                    + 0.35 * F.mse_loss(om["logit"].float().clamp(-6.0, 6.0), ov10["logit"].float().detach().clamp(-6.0, 6.0))
                    + 0.50 * compute_tail_ranking_kd_loss(om["exit1_out"]["logit"], ov10["exit1_out"]["logit"], y_ai, has_ai)
                    + 0.30 * F.mse_loss(om["ebt_margin"].float(), ov10["ebt_margin"].float().detach().clamp(-10.0, 10.0))
                )
            lh.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_h.step()
            mdl.project_ai_head_weight_norm(1.20 * w_ai_init_norm)
        for hk in hk_h:
            hk.remove()
        o_healed_vs = _predict_v122(mdl, vs_sel_p, 2)
        g0_rmse = float(np.sqrt(np.mean((o_healed_vs["logit"] - lg_v10_vs) ** 2)))
        g0_spearman = float(spearmanr(o_healed_vs["logit"], lg_v10_vs).statistic)
        transplant_rep["g0_rmse_post_heal"] = round(g0_rmse, 5)
        transplant_rep["g0_spearman_post_heal"] = round(g0_spearman, 5)
        print(f"[Stage0-CollapseHeal DONE] g0_rmse_post_heal={g0_rmse:.5f} spearman={g0_spearman:.5f}", flush=True)

    ai_anchor_teacher = copy.deepcopy(mdl).eval()
    for p in ai_anchor_teacher.parameters():
        p.requires_grad_(False)
    v10_anchor_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    if v10_teacher is not None:
        del v10_teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7. P0-4: Build Partitioned 16k Hard-Human Memory Bank `M = M_phys (pre_llm) U M_modern (>1y+2026 verified)`!
    phys_human_cand = [x for x in (t0_pool + new_gold_ai_pool + protect_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.20 and x.get("is_phys_prellm", False)]
    mod_human_cand = [x for x in (protect_pool + new_gold_ai_pool) if x["has_ai"] > 0.5 and x["y_score"] <= 0.20 and not x.get("is_phys_prellm", False)]
    if not phys_human_cand:
        phys_human_cand = [x for x in (t0_pool + t1_pool + new_gold_ai_pool) if x["y_score"] <= 0.20][:256]
    if not mod_human_cand:
        mod_human_cand = phys_human_cand

    mem_phys_cap = 32 if args.smoke_test else 6144
    mem_mod_cap = 48 if args.smoke_test else 10240
    rng_m = random.Random(2026112705)
    mem_phys: list[dict[str, Any]] = rng_m.sample(phys_human_cand, min(mem_phys_cap, len(phys_human_cand)))
    mem_mod: list[dict[str, Any]] = rng_m.sample(mod_human_cand, min(mem_mod_cap, len(mod_human_cand)))

    @torch.no_grad()
    def _refresh_hard_human_memory(m_cur: MeguvizV122ApexNanoModel, scan_n: int = 12000) -> dict[str, int]:
        """Refresh `M_phys` & `M_modern` keeping top-scoring hard humans (with `v1.0` anchor mislabel screen on `M_modern`!)."""
        nonlocal mem_phys, mem_mod
        if args.smoke_test:
            return {"phys": len(mem_phys), "mod": len(mem_mod)}
        m_cur.eval()
        def _score_subset(cands: list[dict[str, Any]], cap_k: int, screen_anchor: bool) -> list[dict[str, Any]]:
            sub = rng_m.sample(cands, min(scan_n, len(cands)))
            pk_s = (
                [x["ids"] for x in sub], [x["engram"] for x in sub], [x["trigram"] for x in sub],
                [x["raw_text"] for x in sub], [x["clean_text"] for x in sub],
                np.stack([x["style"] for x in sub]), [], np.zeros(len(sub), dtype=np.float32), np.zeros(len(sub), dtype=np.int32),
            )
            sc_cur = _predict_v122(m_cur, pk_s, 2)["logit"]
            if screen_anchor:
                sc_anc = expit(_predict_v122(ai_anchor_teacher, pk_s, 2)["logit"])
                keep_i = [i for i in range(len(sub)) if sc_anc[i] < 0.80]
            else:
                keep_i = list(range(len(sub)))
            keep_i.sort(key=lambda i: float(sc_cur[i]), reverse=True)
            return [sub[i] for i in keep_i[:cap_k]]

        mem_phys = _score_subset(phys_human_cand, mem_phys_cap, screen_anchor=False)
        mem_mod = _score_subset(mod_human_cand, mem_mod_cap, screen_anchor=True)
        return {"phys": len(mem_phys), "mod": len(mem_mod)}

    # Eval helpers (P1 Lexicographic Wild-3% Tail Selection on `val_select` ONLY — NEVER touches `val_meta_cal`!)
    def _eval_select_ai(m_e: MeguvizV122ApexNanoModel) -> dict[str, float]:
        ows = _predict_v122(m_e, vs_sel_p, 2)
        yb_w = (vs_sel_p[7] >= 70.0).astype(int)
        mw = compute_wild_tail_objective_v122(yb_w, ows["logit"])
        mr1 = compute_wild_tail_objective_v122(yb_w, ows["logit_r1"])
        mr0 = compute_wild_tail_objective_v122(yb_w, ows["logit_ai_r0"])
        mt1 = compute_wild_tail_objective_v122(yb_w, ows["logit_t1"])
        res = {
            "T_wild": mw["T"], "pauc_w": mw["pauc_1pct"], "tkm_w": mw["tkm"], "rp95_w": mw["rp95_wild"],
            "auroc_w": mw["auroc"], "pr_w": mw["pr_auc"], "t1_T": mt1["T"], "r1_auc": mr1["auroc"], "r0_ai": mr0["auroc"],
        }
        if vg2_p is not None and len(vg2_p[7]) >= 20:
            og2 = _predict_v122(m_e, vg2_p, 2)
            mg2 = compute_wild_tail_objective_v122((vg2_p[7] >= 70.0).astype(int), og2["logit"])
            res["T_vg2"] = mg2["T"]
            res["auroc_vg2"] = mg2["auroc"]
            res["pr_vg2"] = mg2["pr_auc"]
            res["sel_T"] = round(0.55 * mw["T"] + 0.35 * mg2["T"] + 0.10 * mt1["T"], 5)
            res["sel_obj"] = round(0.60 * res["sel_T"] + 0.20 * (0.5 * mw["auroc"] + 0.5 * mg2["auroc"]) + 0.20 * (0.5 * mw["pr_auc"] + 0.5 * mg2["pr_auc"]), 5)
        else:
            res["sel_T"] = round(0.85 * mw["T"] + 0.15 * mt1["T"], 5)
            res["sel_obj"] = round(0.60 * res["sel_T"] + 0.20 * mw["auroc"] + 0.20 * mw["pr_auc"], 5)
        return res

    def _eval_select_promo_and_long(m_e: MeguvizV122ApexNanoModel) -> dict[str, float]:
        res = {"promo_auroc": 0.5, "promo_pr": 0.5, "r1_pr": 0.5, "r0_pr": 0.5, "long_ai": 0.5, "long_hyb": 0.5, "long_pr": 0.5}
        if promo_val_select_rows:
            pk = (
                [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
                [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows],
                np.stack([x["style"] for x in promo_val_select_rows]), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32), np.zeros(len(promo_val_select_rows), dtype=np.int32),
            )
            op = _predict_v122(m_e, pk, 2)
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

    rng_tr = random.Random(2026112706)
    slot_aug = BrandPriceSlotAug()
    pike_dual = DynamicPiKEGroupSamplerV11()

    def _pike_phase_b_weights() -> list[float]:
        w7 = pike_dual.w
        raw = np.array([w7[4] * 3.0, w7[6] * 3.5, min(0.15, w7[5] * 1.5), float(w7[:4].sum()) * 0.45], dtype=np.float64)
        raw /= max(float(raw.sum()), 1e-6)
        return raw.tolist()

    # Era-stratified human buckets for 5-Era GroupDRO
    era_human_pools: dict[int, list[dict[str, Any]]] = {e: [] for e in range(NUM_ERAS_V122)}
    for x in (t0_pool + new_gold_ai_pool + protect_pool):
        if x["has_ai"] > 0.5 and x["y_score"] <= 0.25 and 0 <= x["era5"] < NUM_ERAS_V122:
            era_human_pools[x["era5"]].append(x)
    era_dro_q = np.ones(NUM_ERAS_V122, dtype=np.float64) / float(NUM_ERAS_V122)

    pa0_steps = 2 if args.smoke_test else int(args.pa0_steps)
    pa1_steps = 4 if args.smoke_test else int(args.pa1_steps)
    pa_tot = pa0_steps + pa1_steps
    # P0-6: Hard effective-epoch cap `<= 3.5` epochs on `new_gold_ai_pool`!
    new_gold_frac = min(0.32, (3.5 * max(1, len(new_gold_ai_pool))) / float(max(1, bs * max(1, pa_tot))))
    new_gold_frac = max(0.08, new_gold_frac) if new_gold_ai_pool else 0.0

    def _sample_phase_a_batch(b_sz: int, include_mem: bool = True) -> list[dict[str, Any]]:
        n_mem = (min(8, b_sz // 4) if args.smoke_test else min(64, b_sz // 6)) if include_mem else 0
        rem = b_sz - n_mem
        n_new = int(rem * new_gold_frac) if new_gold_ai_pool else 0
        n_era = int(rem * 0.14)
        n_prot = int(rem * 0.12) if protect_pool else 0
        n_t0 = int(rem * 0.28) if t0_pool else 0
        n_t1 = max(0, rem - n_new - n_era - n_prot - n_t0)
        out_b: list[dict[str, Any]] = []
        if n_mem > 0:
            np_m = min(n_mem * 3 // 8, len(mem_phys))
            nm_m = min(n_mem - np_m, len(mem_mod))
            if np_m > 0:
                out_b.extend(rng_tr.sample(mem_phys, np_m))
            if nm_m > 0:
                out_b.extend(rng_tr.sample(mem_mod, nm_m))
        if n_new > 0:
            out_b.extend(rng_tr.choices(new_gold_ai_pool, k=n_new))
        # 5-Era GroupDRO human sampling
        if n_era > 0:
            valid_eras = [e for e in range(NUM_ERAS_V122) if era_human_pools[e]]
            if valid_eras:
                probs_e = np.array([era_dro_q[e] for e in valid_eras], dtype=np.float64)
                probs_e /= probs_e.sum()
                chosen_e = rng_tr.choices(valid_eras, weights=probs_e.tolist(), k=n_era)
                for e_i in chosen_e:
                    out_b.append(rng_tr.choice(era_human_pools[e_i]))
        if n_prot > 0:
            out_b.extend(rng_tr.choices(protect_pool, k=n_prot))
        if n_t0 > 0:
            out_b.extend(rng_tr.choices(t0_pool, k=n_t0))
        pool_fb = t1_pool or t0_pool or new_gold_ai_pool
        if len(out_b) < b_sz and pool_fb:
            out_b.extend(rng_tr.choices(pool_fb, k=b_sz - len(out_b)))
        return out_b[:b_sz]

    def _sample_phase_b_batch(b_sz: int, pw: list[float], wild_promo_frac: float = 0.14) -> list[dict[str, Any]]:
        n_wp = int(b_sz * wild_promo_frac) if wild_promo_focus_pool else 0
        rem = b_sz - n_wp
        n_du = int(rem * pw[0]) if dual_gold_train else 0
        n_hn = int(rem * pw[1]) if hard_neg_rows else 0
        n_uo = int(rem * pw[2]) if uot_pseudo_rows else 0
        n_re = max(0, rem - n_du - n_hn - n_uo)
        out_b: list[dict[str, Any]] = []
        if n_wp > 0:
            out_b.extend(rng_tr.choices(wild_promo_focus_pool, k=n_wp))
        if n_du > 0:
            out_b.extend(rng_tr.choices(dual_gold_train, k=n_du))
        if n_hn > 0:
            out_b.extend(rng_tr.choices(hard_neg_rows, k=n_hn))
        if n_uo > 0:
            out_b.extend(rng_tr.choices(uot_pseudo_rows, k=n_uo))
        pool_fb = t0_pool or t1_pool or dual_gold_train
        if n_re > 0 and pool_fb:
            out_b.extend(rng_tr.choices(pool_fb, k=n_re))
        return out_b[:b_sz]

    def _sample_long_batch(n_d: int) -> list[dict[str, Any]]:
        n_g = min(n_d // 2, len(wave2_longform))
        docs = rng_tr.sample(wave2_longform, n_g) if n_g > 0 else []
        for _ in range(n_d - len(docs)):
            docs.append(longmix_syn.sample_one(rng_tr))
        return docs

    ref_ai = _eval_select_ai(mdl)
    ref_pl = _eval_select_promo_and_long(mdl)
    print(f"[Stage0-Ref] ai={json.dumps(ref_ai)} | dual={json.dumps({k: round(v, 4) for k, v in ref_pl.items()})} | new_gold_frac={new_gold_frac:.3f}", flush=True)

    # =========================================================================
    # PHASE A: A0 (250 steps, Trunk+Core Frozen, FLOW) -> A1 (950 steps, Full P0-3..6 + ASAM + EMA + WiSE-FT!)
    # =========================================================================
    ema_sd = {k: v.detach().float().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pa_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pa_obj = ref_ai["sel_obj"]
    best_pa_T = ref_ai["sel_T"]
    top_pa_ckpts: list[tuple[float, dict[str, torch.Tensor]]] = [(ref_ai["sel_obj"], copy.deepcopy(best_pa_sd))]

    def _update_ema(m_cur: nn.Module, decay: float = 0.995) -> None:
        with torch.no_grad():
            for k, v in m_cur.state_dict().items():
                if v.is_floating_point():
                    ema_sd[k].mul_(decay).add_(v.detach().float().cpu(), alpha=1.0 - decay)
                else:
                    ema_sd[k].copy_(v.detach().cpu())

    for sub_name, n_steps, ph_key in (("A0", pa0_steps, "phase_a0"), ("A1", pa1_steps, "phase_a1")):
        pg_a, hooks_a = build_v122_param_groups(mdl, ph_key, era_grl=era_grl)
        opt_a = torch.optim.AdamW(pg_a, betas=(0.9, 0.95))
        asam_a = ASAMPerturber(mdl, rho=0.03)
        mdl.train(); era_grl.train()

        for step in range(n_steps):
            g_step = step if sub_name == "A0" else (pa0_steps + step)
            if sub_name == "A1" and not args.smoke_test and step > 0 and step % 250 == 0:
                m_st = _refresh_hard_human_memory(mdl)
                print(f"[PhaseA1 step={step} MemRefresh] {m_st} | era_dro_q={[round(float(x), 3) for x in era_dro_q]}", flush=True)
                mdl.train()

            mult = _wsd_lr_mult(step, n_steps, warmup_frac=0.08, decay_frac=0.50 if sub_name == "A1" else 0.25)
            for pg in opt_a.param_groups:
                pg["lr"] = pg["base_lr"] * mult

            # P0-7: Light `QuantNoise` (fake-quant mode toggle on 30% of steps in final 30% of A1)
            use_qnoise = (sub_name == "A1" and not args.smoke_test and step >= int(n_steps * 0.70) and (step % 3 == 0))
            if use_qnoise:
                mdl.set_quant_mode("w4")
            else:
                mdl.set_quant_mode("fp16")

            b_it = _sample_phase_a_batch(min(bs, 32 if args.smoke_test else bs), include_mem=(sub_name == "A1"))
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, has_ai, y_pr, has_pr, _, w_b, a5_b, is_t0, is_phys, _, era5_b) = _collate_items(b_it)

            # Hoist counterfactual mirror pair batch OUTSIDE closure so ASAM ascent/descent uses IDENTICAL batch!
            cf_pair_pack = None
            if cf_mirror_pool and sub_name == "A1" and step % 2 == 0:
                pb = rng_tr.sample(cf_mirror_pool, min(4 if args.smoke_test else 16, len(cf_mirror_pool)))
                ai_items = [_make_item(x["ai_text"], 1.0, -1.0, 0.0, "none", 1.0, has_ai=1.0, is_t0=True) for x in pb]
                hu_items = [_make_item(x["human_text"], 0.0, -1.0, 0.0, "none", 1.0, has_ai=1.0, is_t0=True) for x in pb]
                cf_pair_pack = (_collate_items(ai_items), _collate_items(hu_items))

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                anc_o = ai_anchor_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
                z_anc = anc_o["logit"].float().clamp(-10.0, 10.0)
                z_anc_t1 = anc_o["exit1_out"]["logit"].float().clamp(-10.0, 10.0)
                # P0-6: FLOW easy-first sample weighting (`tau_flow: 0.8 -> 4.0`)
                tau_flow = 0.8 + 3.2 * float(g_step) / float(max(1, pa_tot))
                bce_anc = F.binary_cross_entropy_with_logits(z_anc, y_ai.clamp(1e-4, 1.0 - 1e-4), reduction="none")
                w_flow = torch.exp(-bce_anc / tau_flow).clamp(0.25, 1.50)

            def _phase_a_loss() -> torch.Tensor:
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                    out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
                    lg = out["logit"].float()
                    lg_t1 = out["exit1_out"]["logit"].float()
                    lg_r1 = out["logit_r1"].float()
                    lg_r0 = out["logit_ai_r0"].float()
                    ai_m = has_ai > 0.5
                    ya_c = y_ai.clamp(1e-4, 1.0 - 1e-4)
                    w_eff = w_b * w_flow

                    l_bce = (F.binary_cross_entropy_with_logits(lg[ai_m], ya_c[ai_m], reduction="none") * w_eff[ai_m]).mean()
                    l_t1 = (F.binary_cross_entropy_with_logits(lg_t1[ai_m], ya_c[ai_m], reduction="none") * w_eff[ai_m]).mean()
                    l_r1 = (F.binary_cross_entropy_with_logits(lg_r1[ai_m], ya_c[ai_m], reduction="none") * w_eff[ai_m]).mean()
                    l_r0 = (
                        0.45 * (F.binary_cross_entropy_with_logits(lg_r0[ai_m], ya_c[ai_m], reduction="none") * w_b[ai_m] * (1.0 + 3.0 * (y_ai[ai_m] >= 0.45).float())).mean()
                        + 0.35 * F.mse_loss(lg_r0[ai_m], lg.detach()[ai_m])
                        + 0.10 * listnet_tail_distill_loss(lg_r0[ai_m], lg.detach()[ai_m])
                    )

                    # P0-4: Partitioned SOPA-s pAUC-DRO + CVaR-DORO across t2 + t1 + r1!
                    phys_t0_m = (is_phys > 0.5)
                    l_pauc2, l_ceil2 = compute_sopa_pauc_and_cvar_doro_loss(lg, y_ai, has_ai, phys_t0_m)
                    l_pauc1, l_ceil1 = compute_sopa_pauc_and_cvar_doro_loss(lg_t1, y_ai, has_ai, phys_t0_m)
                    _, l_ceil_r1 = compute_sopa_pauc_and_cvar_doro_loss(lg_r1, y_ai, has_ai, phys_t0_m)

                    # P0-5: EBT CE + RCE/EMR anchored barrier strictly `has_ai` masked!
                    l_ebt_ce = compute_ebt6_partial_label_ce(out["ebt_energies"][ai_m], y_ai[ai_m], torch.zeros_like(y_ai[ai_m]), torch.zeros_like(y_ai[ai_m]), a5_b[ai_m])
                    l_ebt_emr = compute_ebt_rce_emr_loss(out["ebt_energies"], out["ebt_margin"], y_ai, has_ai, is_t0 > 0.5)

                    # P0-6: Tail-Ranking Anchor KD (only where anchor agrees with gold OR non-flip!) + One-Sided Human Drift Barrier!
                    ag_m = ai_m & ((torch.sigmoid(z_anc) - y_ai).abs() < 0.25)
                    l_kd = (
                        compute_tail_ranking_kd_loss(lg, z_anc, y_ai, ag_m.float())
                        + 0.40 * compute_tail_ranking_kd_loss(lg_t1, z_anc_t1, y_ai, ag_m.float())
                    )
                    safe_hu = ai_m & (y_ai <= 0.20) & (z_anc <= -1.5)
                    l_drift = 0.25 * F.relu(lg[safe_hu] - z_anc[safe_hu] - 0.35).pow(2).mean() if safe_hu.any() else lg.sum() * 0.0

                    # P0-3: 5-Era OfflineEraGRLHead on confirmed humans (`emb[:, :192]`)!
                    era_valid = ai_m & (y_ai <= 0.25) & (era5_b >= 0) & (era5_b < NUM_ERAS_V122)
                    grl_lam = 0.5 * min(1.0, float(g_step + 1) / float(max(1, pa_tot // 2)))
                    l_era = era_grl(out["emb"][:, :AI_EMB_END_V11], era5_b, era_valid, lam=grl_lam) if int(era_valid.sum().item()) >= 8 else lg.sum() * 0.0

                    loss = (
                        l_bce + 0.35 * l_t1 + 0.25 * l_r1 + 0.45 * l_r0
                        + 0.25 * (l_pauc2 + 0.50 * l_pauc1)
                        + 0.20 * (l_ceil2 + 0.50 * l_ceil1 + 0.30 * l_ceil_r1)
                        + 0.10 * l_ebt_ce + 0.18 * l_ebt_emr
                        + 0.35 * l_kd + l_drift + 0.08 * l_era
                        + 0.02 * out["cbm_loss"] + 0.015 * compute_slot_orthogonality_loss(out["slots"])
                    )

                    # P0-4/P0-8: v1.0-Anchor-Relative DPO (`L_ref`) on `pack3_C` counterfactual mirrors!
                    if cf_pair_pack is not None:
                        (ai_ids, ai_m_, ai_e, ai_t, ai_s, ac1, ac3, acm, acf, *_), (hu_ids, hu_m_, hu_e, hu_t, hu_s, hc1, hc3, hcm, hcf, *_) = cf_pair_pack
                        oa = mdl(ai_ids, ai_m_, ai_s, engram_ids=ai_e, trigram_ids=ai_t, r0_c1=ac1, r0_c3=ac3, r0_mask=acm, r0_feats=acf, outer_steps=2, return_aux=True)
                        oh = mdl(hu_ids, hu_m_, hu_s, engram_ids=hu_e, trigram_ids=hu_t, r0_c1=hc1, r0_c3=hc3, r0_mask=hcm, r0_feats=hcf, outer_steps=2, return_aux=True)
                        with torch.no_grad():
                            oa0 = ai_anchor_teacher(ai_ids, ai_m_, ai_s, engram_ids=ai_e, trigram_ids=ai_t, r0_c1=ac1, r0_c3=ac3, r0_mask=acm, r0_feats=acf, outer_steps=2, return_aux=True)
                            oh0 = ai_anchor_teacher(hu_ids, hu_m_, hu_s, engram_ids=hu_e, trigram_ids=hu_t, r0_c1=hc1, r0_c3=hc3, r0_mask=hcm, r0_feats=hcf, outer_steps=2, return_aux=True)
                        dpo_margin = (oa["logit"].float() - oa0["logit"].float()) - (oh["logit"].float() - oh0["logit"].float())
                        l_ref = -F.logsigmoid(2.0 * dpo_margin).mean() + 0.15 * F.softplus(2.0 - (oa["logit"].float() - oh["logit"].float())).mean()
                        loss = loss + 0.12 * l_ref
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
            mdl.project_ai_head_weight_norm(1.20 * w_ai_init_norm)
            _update_ema(mdl, decay=0.992 if sub_name == "A0" else 0.996)

            # Update 5-Era Human GroupDRO weights every 50 steps
            if sub_name == "A1" and (step + 1) % 50 == 0:
                with torch.no_grad():
                    o_dro = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=1, return_aux=True)
                    z_d = o_dro["logit"].float()
                    for e_i in range(NUM_ERAS_V122):
                        em = (has_ai > 0.5) & (y_ai <= 0.25) & (era5_b == e_i)
                        if int(em.sum().item()) >= 4:
                            k_d = max(1, int(em.sum().item()) // 5)
                            cvar_e = float(torch.topk(F.relu(z_d[em] + 1.0), k=k_d).values.mean().item())
                            era_dro_q[e_i] *= math.exp(0.10 * min(cvar_e, 3.0))
                    era_dro_q = np.clip(era_dro_q / era_dro_q.sum(), 0.08, 0.45)
                    era_dro_q /= era_dro_q.sum()

            if not args.smoke_test and ((step + 1) % 100 == 0 or step + 1 == n_steps):
                mdl.set_quant_mode("fp16")
                cur_sd_bak = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                mdl.load_state_dict({k: v.to(device, dtype=cur_sd_bak[k].dtype) for k, v in ema_sd.items()}, strict=True)
                ma = _eval_select_ai(mdl)
                cand_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                mdl.load_state_dict({k: v.to(device) for k, v in cur_sd_bak.items()}, strict=True)
                # Lexicographic P1 acceptance: protect tail T >= best_T - 0.005!
                if ma["sel_T"] >= best_pa_T - 0.005 and ma["sel_obj"] >= best_pa_obj:
                    best_pa_obj, best_pa_T, best_pa_sd = ma["sel_obj"], max(best_pa_T, ma["sel_T"]), cand_sd
                if ma["sel_T"] >= best_pa_T - 0.015:
                    top_pa_ckpts.append((ma["sel_obj"], cand_sd))
                    top_pa_ckpts.sort(key=lambda x: x[0], reverse=True)
                    top_pa_ckpts = top_pa_ckpts[:4]
                print(
                    f"[Phase{sub_name} step={step+1}/{n_steps}] loss={float(loss_a.item()):.4f} | "
                    f"sel_T={ma['sel_T']:.5f} obj={ma['sel_obj']:.5f} | w_rp95={ma['rp95_w']:.4f} w_tkm={ma['tkm_w']:.4f} "
                    f"vg2_auc={ma.get('auroc_vg2', 0.5):.4f} r0_ai={ma['r0_ai']:.4f} | elapsed={time.monotonic()-t_start:.0f}s",
                    flush=True,
                )
                mdl.train()

        for hk in hooks_a:
            hk.remove()

    mdl.set_quant_mode("fp16")
    # P0-6: 1D WiSE-FT trajectory interpolation between `v10_anchor_sd` and `best_pa_sd` (and top-ckpt uniform soup)
    mdl.load_state_dict({k: v.to(device) for k, v in best_pa_sd.items()}, strict=True)
    if not args.smoke_test:
        best_wise_sc = best_pa_obj
        best_wise_sd = best_pa_sd
        for alpha in (0.35, 0.55, 0.75, 1.0):
            cand = {
                k: ((1.0 - alpha) * v10_anchor_sd[k].float() + alpha * best_pa_sd[k].float()).to(best_pa_sd[k].dtype)
                if best_pa_sd[k].is_floating_point() else best_pa_sd[k].clone()
                for k in best_pa_sd
            }
            mdl.load_state_dict({k: v.to(device) for k, v in cand.items()}, strict=True)
            mw_a = _eval_select_ai(mdl)
            if mw_a["sel_T"] >= best_pa_T - 0.005 and mw_a["sel_obj"] > best_wise_sc:
                best_wise_sc, best_wise_sd = mw_a["sel_obj"], cand
        mdl.load_state_dict({k: v.to(device) for k, v in best_wise_sd.items()}, strict=True)
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_phaseA_ckpt.pt")
    pa_ai_m = _eval_select_ai(mdl)
    print(f"[PhaseA DONE] {json.dumps(pa_ai_m)}", flush=True)

    # =========================================================================
    # PHASE B: Frozen AI [0:192], Live z_char Promo Bridges, Promo CVaR-DORO, Hoisted ASAM, Live PiKE, ChronoStream!
    # =========================================================================
    pb_steps = 4 if args.smoke_test else int(args.pb_steps)
    pg_b, hooks_b = build_v122_param_groups(mdl, "phase_b")
    opt_b = torch.optim.AdamW(pg_b, betas=(0.9, 0.95))
    asam_b = ASAMPerturber(mdl, rho=0.04)
    best_pb_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    best_pb_sc = -1e9
    mdl.train()

    for step in range(pb_steps):
        mult = _wsd_lr_mult(step, pb_steps, warmup_frac=0.05, decay_frac=0.25)
        for pg in opt_b.param_groups:
            pg["lr"] = pg["base_lr"] * mult

        # Hoist ALL random sampling OUTSIDE `_phase_b_loss()` so ASAM ascent/descent uses IDENTICAL minibatches!
        b_it = _sample_phase_b_batch(min(bs, 32 if args.smoke_test else bs), _pike_phase_b_weights(), wild_promo_frac=0.14)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
         y_ai, has_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, _, l_pc, _) = _collate_items(b_it)

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

        cs_pack = None
        if step % 2 == 1:
            cs_docs = _sample_long_batch(4 if args.smoke_test else 48)
            cs_b = collate_chronostream_docs(cs_docs, tokenizer, max_chunks=3, max_chunk_toks=192)
            # Compute live per-chunk `z_char` from frozen/live `rung0` matching serving!
            cs_pack = cs_b

        def _phase_b_loss() -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = mdl(
                    b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t,
                    r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf,
                    outer_steps=2, detach_zchar_for_ai=True, return_aux=True,
                )
                ai_m = has_ai > 0.5
                pm = has_pr > 0.5
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
                    # P0-1: Multi-rung confirmed-organic human promo ceiling (`t2 + r1 + r0`)!
                    loss_b = loss_b + 0.22 * compute_promo_cvar_ceiling_loss(u, y_pr, has_pr)
                    loss_b = loss_b + 0.18 * compute_promo_cvar_ceiling_loss(u_r1, y_pr, has_pr)
                    loss_b = loss_b + 0.18 * compute_promo_cvar_ceiling_loss(u_r0, y_pr, has_pr)

                # Rung-0 Dual-Axis Teacher Distillation (strictly `has_ai` masked!)
                r0_ai = out["logit_ai_r0"].float()
                t_ai_lg = out["logit"].detach().float()
                t_pr_lg = u.detach()
                l_r0_ai = (
                    0.45 * (F.binary_cross_entropy_with_logits(r0_ai[ai_m], y_ai[ai_m].clamp(1e-4, 1.0 - 1e-4), reduction="none") * w_b[ai_m] * (1.0 + 3.0 * (y_ai[ai_m] >= 0.45).float())).mean()
                    + 0.40 * F.mse_loss(r0_ai[ai_m], t_ai_lg[ai_m])
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

        # Live PiKE update every 100 steps using 7-pool Balanced-PiKE + Aioli damper!
        if not args.smoke_test and (step + 1) % 100 == 0:
            lb_v = float(loss_b.item())
            pike_dual.update(
                np.ones(7, dtype=np.float64),
                np.array([lb_v * 0.85, lb_v * 0.85, lb_v * 0.80, lb_v * 0.90, lb_v * 1.05, lb_v * 0.75, lb_v * 1.15], dtype=np.float64),
                np.zeros(7, dtype=np.float64),
            )

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
    # PHASE C: Zero-Op Fold -> UPQ-W4 -> QuEST + BitDistill + TR-KD Sub2-QAT x2 ->
    #          Post-Quant Exact Convex Soup -> Surgical Shock-Absorber -> Disjoint val_meta_cal Calibration!
    # =========================================================================
    cal_f = _collate_items(cal_items[:32])
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
        pg_q, hooks_q = build_v122_param_groups(mdl, "qat")
        quest_hooks = install_quest_trust_gradient_hooks(mdl, trit_t_mult=1.5, seq2_t_mult=3.5) if qmode == "sub2" else []
        opt_q = torch.optim.AdamW(pg_q, betas=(0.9, 0.95))
        best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc_q = -1e9
        mdl.train()

        for step in range(steps):
            mult = _wsd_lr_mult(step, steps, warmup_frac=0.05, decay_frac=0.30)
            for pg in opt_q.param_groups:
                pg["lr"] = pg["base_lr"] * mult
            opt_q.zero_grad(set_to_none=True)

            # Mix Phase A (incl. 64 hard humans from `M`!) + Phase B wild promo in EVERY QAT step!
            nb_q = min(bs, 32 if args.smoke_test else bs)
            b_it = _sample_phase_a_batch(nb_q * 3 // 4, include_mem=True) + _sample_phase_b_batch(nb_q - nb_q * 3 // 4, _pike_phase_b_weights(), wild_promo_frac=0.20)
            (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf,
             y_ai, has_ai, y_pr, has_pr, _, w_b, a5_b, is_t0, is_phys, _, _) = _collate_items(b_it)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                tv = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
                # P0-7: Tail-Ranking KD (`t2 + t1 + r1 + r0`) + BitDistill 16x16 Slot-KL + SOPA-s + CVaR-DORO + EBT EMR!
                l_tr_t2 = compute_tail_ranking_kd_loss(out["logit"], tv["logit"], y_ai, has_ai)
                l_tr_t1 = compute_tail_ranking_kd_loss(out["exit1_out"]["logit"], tv["exit1_out"]["logit"], y_ai, has_ai)
                l_slot_kl = compute_bitdistill_slot_relation_kl(out["slots"], tv["slots"], mdl.core, fp16_teacher.core)
                l_pauc_q, l_ceil_q = compute_sopa_pauc_and_cvar_doro_loss(out["logit"], y_ai, has_ai, is_phys > 0.5)
                l_ebt_q = (
                    0.35 * F.mse_loss(out["ebt_margin"].float(), tv["ebt_margin"].float().detach().clamp(-10.0, 10.0))
                    + 0.10 * compute_ebt_rce_emr_loss(out["ebt_energies"], out["ebt_margin"], y_ai, has_ai, is_t0 > 0.5)
                )
                l_pr_kd = (
                    0.75 * F.mse_loss(out["logit_promo"].float(), tv["logit_promo"].float().detach().clamp(-12.0, 12.0))
                    + 0.35 * F.mse_loss(out["logit_promo_r1"].float(), tv["logit_promo_r1"].float().detach().clamp(-12.0, 12.0))
                    + 0.35 * F.mse_loss(out["logit_promo_r0"].float(), tv["logit_promo_r0"].float().detach().clamp(-12.0, 12.0))
                    + 0.15 * compute_promo_cvar_ceiling_loss(out["logit_promo"], y_pr, has_pr)
                    + 0.10 * compute_promo_cvar_ceiling_loss(out["logit_promo_r0"], y_pr, has_pr)
                )
                l_r0_kd = 0.40 * F.mse_loss(out["logit_ai_r0"].float(), tv["logit_ai_r0"].float().detach().clamp(-12.0, 12.0))
                loss_q = (
                    l_tr_t2 + 0.50 * l_tr_t1 + 0.04 * l_slot_kl
                    + tail_mult * (0.15 * l_pauc_q + 0.15 * l_ceil_q)
                    + l_ebt_q + l_pr_kd + l_r0_kd
                )
                if step % 8 == 7:
                    cs_b = collate_chronostream_docs(_sample_long_batch(4 if args.smoke_test else 24), tokenizer, max_chunks=3, max_chunk_toks=192)
                    cs_o = mdl.forward_chronostream_batch_v122(
                        cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                        cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                        trunk_no_grad=True, outer_steps=2,
                    )
                    loss_q = loss_q + 0.25 * (
                        F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device))
                        + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
                    )
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_q.step()
            mdl.project_ai_head_weight_norm(1.20 * w_ai_init_norm)

            if not args.smoke_test and ((step + 1) % 250 == 0 or step + 1 == steps):
                ma_q = _eval_select_ai(mdl)
                mp_q = _eval_select_promo_and_long(mdl)
                sc_q = 0.65 * ma_q["sel_obj"] + 0.25 * (0.5 * mp_q["promo_auroc"] + 0.5 * mp_q["promo_pr"]) + 0.10 * ma_q["r0_ai"]
                if sc_q >= best_sc_q:
                    best_sc_q = sc_q
                    best_sd_q = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                print(
                    f"[PhaseC-{st_name} step={step+1}/{steps}] sel_T={ma_q['sel_T']:.5f} obj={ma_q['sel_obj']:.5f} "
                    f"w_rp95={ma_q['rp95_w']:.4f} | promo={mp_q['promo_auroc']:.4f} | r0_ai={ma_q['r0_ai']:.4f}",
                    flush=True,
                )
                mdl.train()

        for hk in hooks_q + quest_hooks:
            hk.remove()
        return best_sd_q

    w4_sd = _run_qat_stage("UPQ-W4", folded_sd, "w4", 2 if args.smoke_test else int(args.pc_w4_steps), tail_mult=1.0)
    q1_sd = _run_qat_stage("Sub2-Tail", w4_sd, "sub2", 2 if args.smoke_test else int(args.pc_sub2_steps), tail_mult=1.35)
    q2_sd = _run_qat_stage("Sub2-Bal", w4_sd, "sub2", 2 if args.smoke_test else int(args.pc_sub2_steps), tail_mult=0.75)

    # P0-7: Post-Quant Exact Convex Soup (`α*Q1 + (1-α)*Q2`, replaces pre-quant DARE-TIES!)
    soup_best_sc, soup_best_alpha, soup_best_sd = -1e9, 1.0, q1_sd
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        cand_s = {
            k: (alpha * q1_sd[k].float() + (1.0 - alpha) * q2_sd[k].float()).to(q1_sd[k].dtype)
            if q1_sd[k].is_floating_point() else q1_sd[k].clone()
            for k in q1_sd
        }
        sc_s = _eval_qat_post_quant(cand_s)
        if sc_s > soup_best_sc:
            soup_best_sc, soup_best_alpha, soup_best_sd = sc_s, alpha, cand_s
    mdl.load_state_dict({k: v.to(device) for k, v in soup_best_sd.items()}, strict=True)
    mdl.set_quant_mode("sub2")
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_qat_soup.pt")
    soup_rep = {"alpha_q1": soup_best_alpha, "final_score": round(soup_best_sc, 5)}
    print(f"[PhaseC-ConvexSoup] {soup_rep}", flush=True)

    # P0-7: Surgical Shock-Absorber (FROZEN W8 rung0 + WSD 5e-5 + T1/EBT/pAUC/RankDistil + 50-step ckpt!)
    pre_sh_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_sh_sc = soup_best_sc
    best_sh_sd, best_sh_sc = pre_sh_sd, pre_sh_sc
    pg_sh, hk_sh = build_v122_param_groups(mdl, "shock")
    opt_sh = torch.optim.AdamW(pg_sh, betas=(0.9, 0.95))
    sh_steps = 2 if args.smoke_test else int(args.pc_shock_steps)
    mdl.train()
    for step in range(sh_steps):
        mult = _wsd_lr_mult(step, sh_steps, warmup_frac=0.15, decay_frac=0.40)
        for pg in opt_sh.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        opt_sh.zero_grad(set_to_none=True)
        b_it = _sample_phase_a_batch(min(bs, 32 if args.smoke_test else bs), include_mem=True)
        (b_ids, b_m, b_e, b_t, b_s56, c1, c3, cm, cf, y_ai, has_ai, y_pr, has_pr, _, _, _, is_t0, is_phys, _, _) = _collate_items(b_it)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            tv = fp16_teacher(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, r0_c1=c1, r0_c3=c3, r0_mask=cm, r0_feats=cf, outer_steps=2, return_aux=True)
            lp_s, lc_s = compute_sopa_pauc_and_cvar_doro_loss(out["logit"], y_ai, has_ai, is_phys > 0.5)
            l_sh = (
                compute_tail_ranking_kd_loss(out["logit"], tv["logit"], y_ai, has_ai)
                + 0.50 * compute_tail_ranking_kd_loss(out["exit1_out"]["logit"], tv["exit1_out"]["logit"], y_ai, has_ai)
                + 0.35 * F.mse_loss(out["ebt_margin"].float(), tv["ebt_margin"].float().detach().clamp(-10.0, 10.0))
                + 0.35 * F.mse_loss(out["logit_promo"].float(), tv["logit_promo"].float().detach().clamp(-12.0, 12.0))
                + 0.20 * lp_s + 0.20 * lc_s
            )
        l_sh.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 0.8)
        opt_sh.step()
        if not args.smoke_test and ((step + 1) % 50 == 0 or step + 1 == sh_steps):
            cand_sh = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            sc_sh = _eval_qat_post_quant(cand_sh)
            if sc_sh >= best_sh_sc:
                best_sh_sc, best_sh_sd = sc_sh, cand_sh
            mdl.train()
    for hk in hk_sh:
        hk.remove()
    mdl.load_state_dict({k: v.to(device) for k, v in best_sh_sd.items()}, strict=True)
    shock_rep = {"kept": bool(best_sh_sc > pre_sh_sc), "before": round(pre_sh_sc, 5), "after": round(best_sh_sc, 5)}
    print(f"[PhaseC-ShockAbsorber] {shock_rep}", flush=True)
    del fp16_teacher, ai_anchor_teacher
    if v121_teacher is not None:
        del v121_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3: Disjoint `val_meta_cal` Full-Logit LEACE -> Affine Align -> B=200 Bootstrap z5_softmin ->
    #          Output-Shifted Promo LUTs + Smooth AI LUTs -> Real-Wild Certified LTT + Conformal FDR -> MGVZ7 + 5-Stage ONNX!
    # =========================================================================
    mdl.eval()
    vmc_o = _predict_v122(mdl, vmc_p, 2)
    z_ll = np.log1p(np.array([len(t) for t in vmc_p[3]], dtype=np.float32))[:, None]
    era_oh = np.zeros((len(vmc_p[8]), NUM_ERAS_V122), dtype=np.float32)
    for ii, ev in enumerate(vmc_p[8]):
        if 0 <= ev < NUM_ERAS_V122:
            era_oh[ii, ev] = 1.0
    st_vmc = vmc_p[5]
    list_rate = st_vmc[:, 14:15] if st_vmc.shape[1] > 14 else np.zeros((len(vmc_p[3]), 1), dtype=np.float32)
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
        pso_l = _predict_v122(mdl, ps_l_pk, 2)
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

    # Cross-Exit Affine Alignment strictly on real-wild `val_meta_cal`!
    vmc_o = _predict_v122(mdl, vmc_p, 2)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_r1"], vmc_o["raw_logit"])
    a_t1, b_t1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_t1"], vmc_o["raw_logit"])
    pa_r1, pb_r1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_promo_r1"], vmc_o["raw_logit_promo"])
    pa_t1, pb_t1 = fit_cross_exit_affine_alignment(vmc_o["raw_logit_promo_t1"], vmc_o["raw_logit_promo"])
    with torch.no_grad():
        mdl.exit_affine_a.copy_(torch.tensor([a_r1, a_t1, 1.0], device=device))
        mdl.exit_affine_b.copy_(torch.tensor([b_r1, b_t1, 0.0], device=device))
        mdl.exit_affine_promo_a.copy_(torch.tensor([pa_r1, pa_t1, 1.0], device=device))
        mdl.exit_affine_promo_b.copy_(torch.tensor([pb_r1, pb_t1, 0.0], device=device))

    # Re-predict post-affine on disjoint `val_meta_cal`, `val_select_wild`, and `val_cal_cert`
    vmc_o = _predict_v122(mdl, vmc_p, 2)
    vcert_o = _predict_v122(mdl, vcert_p, 2)
    yb_mc = (vmc_p[7] >= 70.0).astype(int)

    # Legacy 3-param bounded fusion (kept for backwards compatibility) + NEW Native B=200 Bootstrap `z5_softmin`!
    fusion_rep = fit_3param_logodds_fusion_v121(vmc_o["logit_t1"], vmc_o["logit"], vmc_o["ebt_margin"], vmc_p[7], ebt_clip=6.0)
    with torch.no_grad():
        mdl.fusion_weights.copy_(torch.tensor([fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"]], device=device))
        mdl.fusion_ebt_clip.fill_(float(fusion_rep["ebt_clip"]))
    fw_w = (fusion_rep["w_t1"], fusion_rep["w_t2"], fusion_rep["w_ebt"])
    eb_clip = float(fusion_rep["ebt_clip"])

    def _lg_dict(od: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"lg_r0": od["logit_ai_r0"], "lg_r1": od["logit_r1"], "lg_t1": od["logit_t1"], "lg_t2": od["logit"], "lg_ebt": od["ebt_margin"]}

    z5_cfg = fit_z5_softmin_bootstrap_v122(_lg_dict(vmc_o), yb_mc, ebt_clip=eb_clip, n_boot=20 if args.smoke_test else 200)
    print(f"[Stage3-z5_softmin] {json.dumps(z5_cfg)}", flush=True)

    def _fused3_lg(od: dict[str, np.ndarray]) -> np.ndarray:
        return bounded_fused3_logits(od["logit_t1"], od["logit"], od["ebt_margin"], fw_w[0], fw_w[1], fw_w[2], eb_clip)

    def _z5_lg(od: dict[str, np.ndarray]) -> np.ndarray:
        return apply_z5_softmin_v122(_lg_dict(od), z5_cfg)

    # Smooth AI LUTs (`iso_blend=0.30` on `val_meta_cal`) + MoSEs register offsets + Conformal FDR on `val_cal_cert`!
    y_mc_01 = vmc_p[7] / 100.0
    lut_ai_full = fit_smooth_ai_lut_v122(_fused3_lg(vmc_o), y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_z5 = fit_smooth_ai_lut_v122(_z5_lg(vmc_o), y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_r1 = fit_smooth_ai_lut_v122(vmc_o["logit_r1"], y_mc_01, iso_blend=0.30, n_knots=256)
    lut_ai_r0 = fit_smooth_ai_lut_v122(vmc_o["logit_ai_r0"], y_mc_01, iso_blend=0.30, n_knots=256)

    moses_offs = fit_moses_register_offsets_v122(_z5_lg(vmc_o), yb_mc, vmc_p[5])
    p_z5_cert = apply_spline_temp_lut(expit(np.clip(_z5_lg(vcert_o), -30.0, 30.0)), lut_ai_z5)
    conformal_fdr = fit_conformal_fdr_thresholds_v122(p_z5_cert, (vcert_p[7] >= 70.0).astype(int), pi_wild=0.03)

    # P0-1: Output-Shifted Promo Beta-Spline LUTs (exact Bayes fix for L277!) + Real-Wild Certified Dual-Rung LTT!
    pi_nat_pr = float(np.clip(estimate_wild_prevalence_em(expit(vmc_o["logit_promo"]), pi_train=0.10), 0.004, 0.02))
    lut_pr_full = lut_pr_r1 = lut_pr_r0 = None
    r1_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False}
    r0_ltt_policy: dict[str, Any] = {"enabled": False, "certified_ltt": False, "tau_ai": 0.06, "tau_promo": 0.06, "defer_max": 0.35, "max_chars": 160}
    promo_lut_check: dict[str, Any] = {"median_promo_wild": 0.01, "r0_promo_le008_rate": 1.0}

    if promo_val_select_rows and promo_val_cert_rows:
        def _pack_pr(rows_pr):
            return (
                [x["ids"] for x in rows_pr], [x["engram"] for x in rows_pr], [x["trigram"] for x in rows_pr],
                [x["raw_text"] for x in rows_pr], [x["clean_text"] for x in rows_pr],
                np.stack([x["style"] for x in rows_pr]), [{"spoof_any": x["spoof_any"]} for x in rows_pr],
                np.array([x["y_promo"] for x in rows_pr], dtype=np.float32), np.full(len(rows_pr), -1, dtype=np.int32),
            )
        ps_pk, pc_pk = _pack_pr(promo_val_select_rows), _pack_pr(promo_val_cert_rows)
        pso, pco = _predict_v122(mdl, ps_pk, 2), _predict_v122(mdl, pc_pk, 2)
        lut_pr_full = fit_output_shifted_promo_lut_v122(pso["logit_promo"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)
        lut_pr_r1 = fit_output_shifted_promo_lut_v122(pso["logit_promo_r1"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(pso["logit_promo_r0"], ps_pk[7], pi_nat=pi_nat_pr, shift_damp=0.70, iso_blend=0.30)

        p_pr_wild = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo"], -30.0, 30.0)), lut_pr_full)
        p_r0_pr_wild = apply_spline_temp_lut(expit(np.clip(vmc_o["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
        promo_lut_check = {
            "median_promo_wild": float(np.median(p_pr_wild)),
            "p90_promo_wild": float(np.quantile(p_pr_wild, 0.90)),
            "r0_promo_le008_rate": float((p_r0_pr_wild <= 0.08).mean()),
        }
        print(f"[Stage3-PromoLUT-Check] {promo_lut_check}", flush=True)

        def _elig_r0(pk):
            return np.array([len(pk[3][i]) <= 160 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[3][i].lower()) for i in range(len(pk[0]))], dtype=bool)
        raw_r0_ltt = certify_fixed_sequence_dual_axis_ltt(
            _elig_r0(vmc_p), vmc_o["p_defer_r0"], apply_spline_temp_lut(expit(vmc_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vmc_o["logit_promo_r0"]), lut_pr_r0),
            vcert_p[7], _elig_r0(vcert_p), vcert_o["p_defer_r0"], apply_spline_temp_lut(expit(vcert_o["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(vcert_o["logit_promo_r0"]), lut_pr_r0),
            pc_pk[7] >= 0.65, _elig_r0(pc_pk), pco["p_defer_r0"], apply_spline_temp_lut(expit(pco["logit_ai_r0"]), lut_ai_r0), apply_spline_temp_lut(expit(pco["logit_promo_r0"]), lut_pr_r0),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05, r1_cost_rel=0.05,
            d_grid=(0.15, 0.25, 0.35, 0.50), tau_a_grid=(0.03, 0.05, 0.08, 0.12), tau_p_grid=(0.04, 0.06, 0.10, 0.15),
        )
        r0_ltt_policy = {**raw_r0_ltt, "defer_max": raw_r0_ltt.get("r1_defer_max", 0.25), "tau_ai": raw_r0_ltt.get("r1_p_ai_max", 0.06), "tau_promo": raw_r0_ltt.get("r1_p_promo_max", 0.06), "max_chars": 160}

        def _elig_r1(pk, od):
            return np.array([
                len(pk[0][i]) <= 96 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[4][i])
                and float(np.max(pk[5][i, list(R1_PROMO_VETO_DIMS_V11)])) == 0.0 and float(od["r1_span_max"][i]) <= 0.50
                for i in range(len(pk[0]))
            ], dtype=bool)
        r1_ltt_policy = certify_fixed_sequence_dual_axis_ltt(
            _elig_r1(vmc_p, vmc_o), vmc_o["p_defer1"], apply_spline_temp_lut(expit(vmc_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vmc_o["logit_promo_r1"]), lut_pr_r1),
            vcert_p[7], _elig_r1(vcert_p, vcert_o), vcert_o["p_defer1"], apply_spline_temp_lut(expit(vcert_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vcert_o["logit_promo_r1"]), lut_pr_r1),
            pc_pk[7] >= 0.65, _elig_r1(pc_pk, pco), pco["p_defer1"], apply_spline_temp_lut(expit(pco["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(pco["logit_promo_r1"]), lut_pr_r1),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05,
        )

    # Export MGVZ7 + Post-Unpack 5-Stage FP32-Exact ONNX (100% v1.2.1 ABI compatible!)
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
        "z5_softmin_cfg": z5_cfg,
        "moses_register_offsets": moses_offs,
        "conformal_fdr_thresholds": conformal_fdr,
        "lut_ai_r0": lut_ai_r0, "lut_ai_r1": lut_ai_r1, "lut_ai_full": lut_ai_full, "lut_ai_z5": lut_ai_z5,
        "lut_promo_r0": lut_pr_r0, "lut_promo_r1": lut_pr_r1, "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_ltt_policy,
        "dual_axis_ltt_policy": r1_ltt_policy,
        "capabilities": [
            "rung0_zero_bpe_charhash", "dense_shared_quant_res", "softball_logitnorm_v122_capped_mil",
            "dual_source_v10_v121_rebase", "era5_invariance", "z5_softmin_native", "output_shifted_promo_lut",
            "quest_bitdistill_trkd_qat", "conformal_fdr_and_ltt",
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

    # Final Unpacked Evaluation + 9-Gate CI Suite
    mdl_u = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v7_v121_into_model(mdl_u, out_dir / exp_info["bundle_v7_mgvz"]["file"])
    mdl_u = mdl_u.to(device).eval()

    tgo, tdo = _predict_v122(mdl_u, tg_p, 2), _predict_v122(mdl_u, td_p, 2)
    p_tg = apply_spline_temp_lut(expit(np.clip(_z5_lg(tgo), -30.0, 30.0)), lut_ai_z5)
    p_td = apply_spline_temp_lut(expit(np.clip(_z5_lg(tdo), -30.0, 30.0)), lut_ai_z5) if len(td_p[7]) > 0 else np.array([])
    tb_m = {**compute_eval_metrics(tg_p[7], p_tg), **compute_high_precision_metrics(tg_p[7], p_tg)}
    td_m = compute_eval_metrics(td_p[7], p_td) if len(td_p[7]) > 0 else {}
    tb2_m: dict[str, Any] = {}
    if tg2_p is not None:
        tg2o = _predict_v122(mdl_u, tg2_p, 2)
        p_tg2 = apply_spline_temp_lut(expit(np.clip(_z5_lg(tg2o), -30.0, 30.0)), lut_ai_z5)
        tb2_m = {**compute_eval_metrics(tg2_p[7], p_tg2), **compute_high_precision_metrics(tg2_p[7], p_tg2)}
    g500_m: dict[str, Any] = {}
    if god500_p is not None and len(god500_p[7]) >= 20:
        g5o = _predict_v122(mdl_u, god500_p, 2)
        p_g5 = apply_spline_temp_lut(expit(np.clip(_z5_lg(g5o), -30.0, 30.0)), lut_ai_z5)
        g500_m = {**compute_eval_metrics(god500_p[7], p_g5), **compute_high_precision_metrics(god500_p[7], p_g5)}

    pb_m: dict[str, Any] = {}
    pb_p = Path(__file__).resolve().parent / "data" / "v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_pk = _pre_eval(pb_rows)
        pbo = _predict_v122(mdl_u, pb_pk, 2)
        y_pb_ai, y_pb_pr = np.array([int(r["y_ai"]) for r in pb_rows]), np.array([int(r["y_promo"]) for r in pb_rows])
        pb_m = {
            "ai_z5_auroc": round(float(roc_auc_score(y_pb_ai, _z5_lg(pbo))), 4),
            "promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo"])), 4),
            "promo_pr_auc": round(float(average_precision_score(y_pb_pr, pbo["logit_promo"])), 4),
            "r0_ai_auroc": round(float(roc_auc_score(y_pb_ai, pbo["logit_ai_r0"])), 4),
            "r0_promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo_r0"])), 4),
        }

    final_ci = run_v122_ci_gates(
        mdl_u, tokenizer,
        transplant_rmse=g0_rmse if transplant_rep.get("mode") != "cold_init" else None,
        transplant_spearman=g0_spearman,
        era_probe_auroc=float(leace_rep.get("era_probe_auroc", 0.5)),
        promo_lut_check=promo_lut_check if lut_pr_full is not None else None,
    )
    rep = {
        "version": args.version, "gpu_name": gpu_name,
        "sibling_stats": dict(sib["stats"]),
        "bundle_v7_total_mb": round(v7_mb, 4),
        "onnx_5stage_zip_mb": round(zip_onnx.stat().st_size / (1024.0 * 1024.0), 4),
        "stage0_dual_transplant": transplant_rep,
        "phase_a_metrics": pa_ai_m, "phase_b_fp16_ai": fp16_ai, "phase_b_fp16_dual": fp16_pl,
        "fold_zero_op": fold_rep, "qat_convex_soup": soup_rep, "shock_absorber": shock_rep,
        "leace_surgery": leace_rep, "fusion_weights": fusion_rep, "z5_softmin_cfg": z5_cfg,
        "moses_register_offsets": moses_offs, "conformal_fdr_thresholds": conformal_fdr,
        "promo_lut_check": promo_lut_check,
        "rung0_ltt_policy": r0_ltt_policy, "dual_axis_ltt_policy": r1_ltt_policy,
        "test_bili_gold_v1": tb_m, "test_bili_gold_v2": tb2_m, "test_dsh_gold": td_m,
        "god_tier_500_holdout": g500_m, "promobench48": pb_m,
        "onnx_export": {
            "r0_fp32_max_diff": exp_info["r0_fp32_max_diff"],
            "r1_fp32_max_diff": exp_info["r1_fp32_max_diff"],
            "r2_fp32_max_diff": exp_info["r2_fp32_max_diff"],
            "r3_fp32_max_diff": exp_info["r3_fp32_max_diff"],
            "dynaseam_onnx": exp_info["dynaseam_onnx"],
        },
        "v122_ci_gates": final_ci,
        "release_gates": {
            "bundle_le_2p20_mb": bool(v7_mb <= 2.20),
            "onnx_5stage_exact": bool(
                exp_info["r0_fp32_max_diff"] < 1e-4
                and exp_info["r1_fp32_max_diff"] < 1e-4
                and exp_info["r2_fp32_max_diff"] < 1e-4
                and exp_info["r3_fp32_max_diff"] < 1e-4
                and bool(exp_info["dynaseam_onnx"].get("pass", exp_info["dynaseam_onnx"].get("onnx_max_diff", 0.0) < 1e-4))
            ),
            "v122_ci_passed": bool(final_ci["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
    if not all(rep["release_gates"].values()):
        print(f"[WARN] Non-fatal release gate alert (artifacts still saved!): {rep['release_gates']}", flush=True)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="MeguvizCAC v1.2.2-ApexNano-Ultra 3-Phase Training Pipeline")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--data", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    p.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    p.add_argument("--extra-pairs", type=Path, action="append", default=[])
    p.add_argument("--v10-fp16-master", type=Path, default=None)
    p.add_argument("--v10-bundle-mgvz", type=Path, default=None)
    p.add_argument("--v121-fp16-master", type=Path, default=None)
    p.add_argument("--v121-bundle-mgvz", type=Path, default=None)
    p.add_argument("--v07-tokenizer", type=Path, default=None)
    p.add_argument("--v08-gemini-gold", type=Path, default=None)
    p.add_argument("--v08-gemini-pairs", type=Path, default=None)
    p.add_argument("--v11-data-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    p.add_argument("--wave25-gold", type=Path, default=None)
    p.add_argument("--gold-v2-dir", type=Path, default=None)
    p.add_argument("--wave3-dir", type=Path, default=None)
    p.add_argument("--sibling-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("scratch/v122_out"))
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--version", type=str, default="v1.2.2-r1-apexnano-b7")
    p.add_argument("--max-len", type=int, default=MAX_LEN_V07)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--pa0-steps", type=int, default=250)
    p.add_argument("--pa1-steps", type=int, default=950)
    p.add_argument("--pb-steps", type=int, default=2000)
    p.add_argument("--pc-w4-steps", type=int, default=500)
    p.add_argument("--pc-sub2-steps", type=int, default=1000)
    p.add_argument("--pc-shock-steps", type=int, default=250)
    p.add_argument("--export-matmulnbits", action="store_true")
    args = p.parse_args()
    train_v122_apexnano_full(args)


if __name__ == "__main__":
    main()
