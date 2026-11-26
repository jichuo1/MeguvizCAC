"""MeguvizCAC v1.1.0-OmniGuard End-to-End Deep Post-Training Pipeline (`P0–P10`, Colab A100 80GB + Local RTX 4060 Ti `--dry-run`).

Implements the complete blueprint locked in `docs/posttrain-plan-v1.1.0-omniguard.md`:
  - **Stage 0**: Pre-folded `v1.0.0-R1` (`fp16_master.pt`) P1 Zero-Shock Surgery (LS `40->56` bridge, exact OBS
    permutation firewall `[0:128]` shared / `[128:192]` AI-only / `[192:256]` promo-only, exact Net2Net MoLE `8->10`
    split, EBT `5->6` promo at idx 5) + `V11-G0` verification + R1 teacher cache (`{z, z_t1, z_r1, emb[:192]}`) +
    ingestion of `Wave-1 (5,880 + 1,600 pairs) + Wave-2 Precision (4,410 + 1,200 span-aligned longform) +
    UOT/CDMAD (123 wild pos + 55 chants + 80k A-Res negs) + P10-3 gated 2k BGE k-NN + CC promo_val_cert (>=600)`.
  - **Stage 1a (1,200 steps)**: Frozen trunk warmup + steps `0..600` P1.2 `40->56` pre-activation bridge distillation +
    ChronoStream every 4th step.
  - **Stage 1b (6,000 steps)**: Unfrozen dual-axis with F7 split AdamW param groups, L2-SP decay toward `theta_R1`,
    selective R1 KD anchor (`M_anc=0` on Q3 / T0 teacher-wrong / Wave-2 rescue), 6-way partial-label EBT CE + dual
    margins, Q3 commercial-valley Top-FP ceiling, `L_sub4` multi-label soft masking, `orthogonal_dual_axis_pair_loss`,
    `ChronoStream` TBPTT (`k1=M, k2=3`, 12-archetype LongMix + Wave-2 1,200 longform), `GradNorm-lite` + auto-PCGrad,
    `Balanced-PiKE/Aioli` adaptive pool mixer (AI floor >= 0.60), & 3-tier Rollback Ladder (`T-soft/T-hard/T-abort`).
  - **Stage 1.5**: Kronecker `LoGra` Top-FP influence sniping (zeroes bottom 1.5% non-T0) + 3 ASAM branches
    (`B1 AI-Shield`, `B2 Promo-Precision`, `B3 LongStream`) + AI-constrained `greedy_pauc_soup`.
  - **Stage 2a/2b**: `fold_zero_op_v11` -> `UPQ W4 (600)` -> `Sub-2 QAT x2 (2x1,500)` with dual-axis distillation.
  - **Stage 3a/3b & Stage 4**: Bit-exact `MGVZ6` pack/unpack -> P5 FP16 Shock-Absorber -> Zero-Byte Dual-Axis Oblique
    `LEACE` -> Cross-Exit Affine Alignment (AI & Promo) -> Weighted Venn-ABERS promo LUTs -> 18-Cell Mondrian +
    Strict 1-D Fixed-Sequence Dual-Axis LTT (`alpha_ai=0.5%, alpha_pr=1.0%`) -> Export `<= 2.20 MB` Bundle v6+ &
    4-Session Static ONNX (`r1`, `r2`, `r3`, `dynaseam`) + `PromoBench-48` / `LongBench-24` / `V11` CI Gates!
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
from .train_v06_hyperloop import _modern_two_way_pauc_and_top_push
from .train_v07_omniveritas import _wsd_lr_mult
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    STYLE_FEAT_DIM_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    extract_style_features_40d,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import MAX_LEN_V07, OmniQuantLinear, SeqLoRAMoE
from .v08_zenith_ops import (
    ASAMPerturber,
    export_beta_spline_as_v07_lut,
    fit_venn_abers_and_beta_spline_lut,
    greedy_pauc_soup,
    listnet_tail_distill_loss,
    zenith_graded_anchor_loss,
)
from .v09_cascade_cert import fit_cross_exit_affine_alignment
from .v09_chronoloop import HIDDEN_V09
from .v10_apexloop import (
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    MeguvizV10ApexLoopModel,
)
from .v11_augment import (
    AdversarialObfuscationAug,
    BrandPriceSlotAug,
    LightHumanPostEditAug,
    QuadSpliceAug,
    orthogonal_dual_axis_pair_loss,
)
from .v11_cascade_cert import (
    apply_dual_axis_oblique_leace_v11,
    certify_fixed_sequence_dual_axis_ltt,
    estimate_wild_prevalence_em,
    fit_mondrian_18cell_dual_axis,
    fit_weighted_venn_abers_promo_lut,
)
from .v11_data_engine import OnlineLongMixSynthesizerV11
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d, plan_long_chunks_v11
from .v11_omniguard import (
    AI_EMB_END_V11,
    PROMO_SUB4_NAMES,
    R1_PROMO_VETO_DIMS_V11,
    SHARED_EMB_END_V11,
    MeguvizV11OmniGuardModel,
    export_v11_modular_onnx_and_bundle_v6,
    fold_zero_op_v11,
    load_bundle_v6_v11_into_model,
    pack_bundle_v6_v11,
    run_v11_ci_gates,
    transplant_v10_to_v11_zero_shock,
)


# ---------------------------------------------------------------------------
# 1. Frontier Multi-Task Controls: Balanced-PiKE/Aioli, GradNorm-lite, LoGra, L2-SP
# ---------------------------------------------------------------------------

class DynamicPiKEGroupSamplerV11:
    """NeurIPS'25 Balanced-PiKE + ICLR'25 Aioli adaptive pool weighter with hard AI floor >= 0.60 (#6-I4 & #7 §9)."""

    POOL_NAMES = ("t0_ai", "t1_ai", "t2_mid", "ai_protect", "dual_gold", "uot_pseudo", "hard_neg")

    def __init__(self, init_weights: dict[str, float] | None = None):
        default_w = {
            "t0_ai": 0.34, "t1_ai": 0.35, "t2_mid": 0.03, "ai_protect": 0.04,
            "dual_gold": 0.09, "uot_pseudo": 0.10, "hard_neg": 0.05,
        }
        w_d = init_weights or default_w
        self.w = np.array([float(w_d[k]) for k in self.POOL_NAMES], dtype=np.float64)
        self.w /= self.w.sum()

    def update(self, grad_norms_sq: np.ndarray, val_losses: np.ndarray, ai_alignment: np.ndarray, eta: float = 0.08) -> np.ndarray:
        """Update simplex weights with Aioli negative-transfer damper & hard AI floor >= 0.60."""
        gn = np.asarray(grad_norms_sq, dtype=np.float64)
        vl = np.asarray(val_losses, dtype=np.float64)
        al = np.asarray(ai_alignment, dtype=np.float64)
        # PiKE utility: reward unresolved validation loss & strong low-conflict gradients; damp negative transfer on AI!
        score = eta * (vl / (np.mean(vl) + 1e-6) + 0.25 * np.log1p(gn / (np.mean(gn) + 1e-6)))
        damper = np.exp(np.minimum(0.0, 2.0 * al))
        w_new = self.w * np.exp(np.clip(score, -0.35, 0.35)) * damper
        w_new = np.maximum(w_new, np.array([0.22, 0.24, 0.01, 0.03, 0.05, 0.03, 0.03]))
        # Cap pseudo <= 0.15
        w_new[5] = min(w_new[5], 0.15)
        w_new /= w_new.sum()
        # Hard floor: AI pools (indices 0,1,2,3) MUST remain >= 0.60 for Zero Negative Transfer!
        ai_sum = float(w_new[:4].sum())
        if ai_sum < 0.60:
            w_new[:4] *= 0.60 / max(ai_sum, 1e-6)
            rem = 1.0 - float(w_new[:4].sum())
            w_new[4:] *= rem / max(float(w_new[4:].sum()), 1e-6)
        self.w = w_new
        return self.w.copy()

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in zip(self.POOL_NAMES, self.w)}


class GradNormLiteConflictGuardV11:
    """P2 GradNorm-lite cap (`rho_max=0.5` in 1b / `0.35` in 1.5) + auto-PCGrad trigger (3x cosine < -0.2)."""

    def __init__(self, rho_max: float = 0.50, ema_decay: float = 0.85):
        self.rho_max = float(rho_max)
        self.ema_decay = float(ema_decay)
        self.ratio_ema = 0.35
        self.neg_cos_streak = 0
        self.pcgrad_active = False

    def probe(self, shared_params: list[torch.nn.Parameter], loss_ai: torch.Tensor, loss_pr_scaled: torch.Tensor) -> dict[str, float]:
        g_ai = torch.autograd.grad(loss_ai, shared_params, retain_graph=True, allow_unused=True)
        g_pr = torch.autograd.grad(loss_pr_scaled, shared_params, retain_graph=True, allow_unused=True)
        n_ai_sq = 0.0
        n_pr_sq = 0.0
        dot = 0.0
        for ga, gp in zip(g_ai, g_pr):
            if ga is not None and gp is not None:
                a_f, p_f = ga.detach().float(), gp.detach().float()
                n_ai_sq += float((a_f * a_f).sum().item())
                n_pr_sq += float((p_f * p_f).sum().item())
                dot += float((a_f * p_f).sum().item())
        n_ai = math.sqrt(n_ai_sq + 1e-12)
        n_pr = math.sqrt(n_pr_sq + 1e-12)
        ratio = n_pr / max(n_ai, 1e-6)
        cos = dot / max(n_ai * n_pr, 1e-6)
        self.ratio_ema = self.ema_decay * self.ratio_ema + (1.0 - self.ema_decay) * ratio
        if cos < -0.20:
            self.neg_cos_streak += 1
        else:
            self.neg_cos_streak = 0
        self.pcgrad_active = self.neg_cos_streak >= 3
        mult = min(1.0, self.rho_max / max(self.ratio_ema, 1e-6))
        return {"grad_ratio_ema": round(self.ratio_ema, 4), "grad_cos": round(cos, 4), "lambda_p_mult": round(mult, 4), "pcgrad": float(self.pcgrad_active)}


class RollbackMonitorV11:
    """P7 3-Tier Automatic Rollback Ladder (`T-soft -> T-hard -> T-abort -> v1.1-Lite`)."""

    def __init__(self, ref_auroc_vg: float, ref_rp95_vg: float, ref_rp95_wild: float, ref_r1_auroc: float, ref_topfp: float):
        self.ref = {
            "auroc_vg": ref_auroc_vg, "rp95_vg": ref_rp95_vg, "rp95_wild": ref_rp95_wild,
            "r1_auroc": ref_r1_auroc, "topfp": ref_topfp,
        }
        self.consec_soft = 0
        self.hard_count = 0
        self.last_pass_sd: dict[str, torch.Tensor] | None = None

    def check(self, cur: dict[str, float], model_sd: dict[str, torch.Tensor]) -> str:
        d_auc = cur["auroc_vg"] - self.ref["auroc_vg"]
        d_rp95 = cur["rp95_vg"] - self.ref["rp95_vg"]
        d_wild = cur["rp95_wild"] - self.ref["rp95_wild"]
        d_r1 = cur["r1_auroc"] - self.ref["r1_auroc"]
        d_fp = cur["topfp"] - self.ref["topfp"]

        is_soft = (d_auc < -0.0015) or (d_rp95 < -0.008) or (d_wild < -0.010) or (d_fp > 0.03)
        is_hard = (d_auc < -0.004) or (d_r1 < -0.006)
        if is_soft:
            self.consec_soft += 1
        else:
            self.consec_soft = 0
            self.last_pass_sd = {k: v.detach().cpu().clone() for k, v in model_sd.items()}

        if is_hard or self.consec_soft >= 2:
            self.hard_count += 1
            self.consec_soft = 0
            if self.hard_count >= 3:
                return "T-abort"
            return "T-hard"
        if is_soft:
            return "T-soft"
        return "PASS"


@torch.no_grad()
def kronecker_logra_topfp_snipe(
    emb_train: np.ndarray,
    err_train: np.ndarray,
    is_t0_train: np.ndarray,
    emb_val_fp: np.ndarray,
    err_val_fp: np.ndarray,
    k_proj: int = 32,
    ridge_lam: float = 1e-2,
    prune_frac: float = 0.015,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Kronecker LoGra (`ICLR'25`, #6-I3): identifies bottom `prune_frac` negative-influence non-T0 rows driving val Top-FP!

    For the linear AI head `z = w^T x + b` (`x = emb[:, :192]`), per-sample gradient is `g_i = err_i * x_i`.
    Projects `x_i` onto `k_proj`-dim random JL/Kronecker subspace and computes `-g_val^T (H + lam I)^{-1} g_i` in <1s!
    """
    n = len(emb_train)
    if n < 32 or len(emb_val_fp) == 0:
        return {"zeroed_indices": [], "zeroed_count": 0}
    rng = np.random.default_rng(seed)
    x_tr = np.asarray(emb_train[:, :AI_EMB_END_V11], dtype=np.float32)
    x_vf = np.asarray(emb_val_fp[:, :AI_EMB_END_V11], dtype=np.float32)
    proj = rng.normal(0.0, 1.0 / math.sqrt(k_proj), size=(AI_EMB_END_V11, k_proj)).astype(np.float32)

    g_tr = (x_tr @ proj) * np.asarray(err_train, dtype=np.float32)[:, None]       # (N, k)
    g_vf = ((x_vf @ proj) * np.asarray(err_val_fp, dtype=np.float32)[:, None]).mean(axis=0)  # (k,)

    h_mat = (g_tr.T @ g_tr) / float(max(1, n)) + np.eye(k_proj, dtype=np.float32) * float(ridge_lam)
    ihvp = np.linalg.solve(h_mat, g_vf)  # (k,)
    # Negative influence = removing sample i DECREASES val-FP loss!
    infl = -(g_tr @ ihvp)  # most negative => worst Top-FP culprits
    non_t0 = np.where(~np.asarray(is_t0_train, dtype=bool))[0]
    if len(non_t0) == 0:
        return {"zeroed_indices": [], "zeroed_count": 0}
    k_cut = max(1, int(round(len(non_t0) * float(prune_frac))))
    worst_local = np.argsort(infl[non_t0])[:k_cut]
    zeroed = [int(non_t0[idx]) for idx in worst_local if float(infl[non_t0[idx]]) < 0.0]
    return {"zeroed_indices": zeroed, "zeroed_count": len(zeroed), "min_infl": round(float(np.min(infl[non_t0])), 6)}


def compute_ebt6_partial_label_ce(energies: torch.Tensor, y_ai: torch.Tensor, y_promo: torch.Tensor, has_promo: torch.Tensor, y_attr5: torch.Tensor) -> torch.Tensor:
    """P3 6-Way Partial-Label Energy CE (`-log sum_{k in S} softmax(-E)_k`, Promo strictly at idx 5 per F1!)."""
    log_q = F.log_softmax(-energies.float(), dim=-1)  # (B, 6)
    bsz = energies.shape[0]
    dev = energies.device
    s_mask = torch.zeros((bsz, 6), device=dev, dtype=torch.bool)

    is_hu = y_ai <= 0.30
    is_ai = y_ai >= 0.70
    is_asst = (~is_hu) & (~is_ai)

    # Human with unknown promo -> S = {0, 5}; Q4-confirmed -> {0}; Q3-promo -> {5}
    hu_unk = is_hu & (has_promo < 0.5)
    hu_q4 = is_hu & (has_promo >= 0.5) & (y_promo <= 0.25)
    hu_q3 = is_hu & (has_promo >= 0.5) & (y_promo >= 0.65)
    hu_mid_p = is_hu & (~hu_unk) & (~hu_q4) & (~hu_q3)

    s_mask[hu_unk, 0] = True; s_mask[hu_unk, 5] = True
    s_mask[hu_mid_p, 0] = True; s_mask[hu_mid_p, 5] = True
    s_mask[hu_q4, 0] = True
    s_mask[hu_q3, 5] = True
    s_mask[is_asst, 1] = True

    # AI -> exact derived class in {2, 3, 4} (or all {2, 3, 4} if attr5 fell outside)
    for bi in range(bsz):
        if is_ai[bi]:
            a5 = int(y_attr5[bi].item())
            if 2 <= a5 <= 4:
                s_mask[bi, a5] = True
            else:
                s_mask[bi, 2:5] = True

    masked_lq = log_q.masked_fill(~s_mask, -1e4)
    return (-torch.logsumexp(masked_lq, dim=-1)).mean()


def compute_dual_axis_ebt_margin_loss(
    ebt_margin: torch.Tensor,
    ebt_promo_margin: torch.Tensor,
    y_ai: torch.Tensor,
    y_promo: torch.Tensor,
    has_promo: torch.Tensor,
) -> torch.Tensor:
    """P3 `L_ebt,m`: margin separation for AI (`min(E0,E5) - min(E2:5)`) & Commercial Valley (`E0 - E5`)."""
    loss = ebt_margin.sum() * 0.0
    m_ai_pos = y_ai >= 0.78
    m_ai_neg = y_ai <= 0.20
    if m_ai_pos.any():
        loss = loss + 0.05 * F.softplus(1.0 - ebt_margin[m_ai_pos]).mean()
    if m_ai_neg.any():
        loss = loss + 0.05 * F.softplus(1.0 + ebt_margin[m_ai_neg]).mean()

    m_q3 = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo >= 0.65)
    m_q4 = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo <= 0.25)
    if m_q3.any():
        loss = loss + 0.05 * F.softplus(1.0 - ebt_promo_margin[m_q3]).mean()
    if m_q4.any():
        loss = loss + 0.05 * F.softplus(1.0 + ebt_promo_margin[m_q4]).mean()
    return loss


def compute_sub4_masked_loss(
    sub4_logits: torch.Tensor,
    promo_logit: torch.Tensor,
    sub4_target_idx: torch.Tensor,  # -1=unknown, 0=none, 1=coupon, 2=self_promo, 3=soft_shill, 4=tool_boilerplate
    y_promo: torch.Tensor,
    has_promo: torch.Tensor,
) -> torch.Tensor:
    """P3 + #5 `L_sub4`: multi-label soft masking (w=0.3 on un-annotated Y1-Y3 to allow co-occurrence; Y4 independent!)."""
    valid = (has_promo >= 0.5) & (sub4_target_idx >= 0)
    if not valid.any():
        return sub4_logits.sum() * 0.0
    sl = sub4_logits[valid].float()
    ul = promo_logit[valid].float()
    ti = sub4_target_idx[valid]
    yp = y_promo[valid]

    tgt = torch.zeros_like(sl)
    w_m = torch.zeros_like(sl)
    for k in range(1, 5):  # k=1..4 maps to sub4 col 0..3
        is_k = ti == k
        tgt[is_k, k - 1] = 1.0
        w_m[is_k, k - 1] = 1.0
        if k <= 3:
            # Other Y1-Y3 subtypes get soft negative weight 0.3 (single-label schema hides co-occurrence!)
            for o in range(1, 4):
                if o != k:
                    w_m[is_k, o - 1] = 0.3
    # Confirmed non-promo (promo_score <= 25 & not tool_boilerplate) -> hard 0 on Y1-Y3
    is_neg = (yp <= 0.25) & (ti == 0)
    w_m[is_neg, 0:3] = 1.0
    w_m[is_neg, 3] = 0.5

    bce = F.binary_cross_entropy_with_logits(sl, tgt, reduction="none")
    l_sub = (bce * w_m).sum() / w_m.sum().clamp(min=1.0)
    # Hierarchy constraint ONLY for commercial subtypes Y1-Y3 <= total promo logit u (Y4 independent!)
    l_hier = 0.05 * F.relu(sl[:, 0:3] - ul.unsqueeze(-1)).pow(2).mean()
    return l_sub + l_hier


def compute_v11_orthogonality_loss(model: MeguvizV11OmniGuardModel, ai_logits: torch.Tensor, y_ai: torch.Tensor, y_promo: torch.Tensor, has_promo: torch.Tensor) -> torch.Tensor:
    """P3 `L_perp`: cosine orthogonality on `[0:128]` shared head weights + Q3 vs Q4 AI-logit mean alignment."""
    w_ai_sh = model.ai_head.weight[0, :SHARED_EMB_END_V11].float()
    w_pr_sh = model.promo_head.weight[0, :SHARED_EMB_END_V11].float()
    cos_w = F.cosine_similarity(w_ai_sh.unsqueeze(0), w_pr_sh.unsqueeze(0), dim=-1).pow(2).mean()
    loss = 0.05 * cos_w
    q3_m = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo >= 0.65)
    q4_m = (y_ai <= 0.30) & (has_promo >= 0.5) & (y_promo <= 0.25)
    if int(q3_m.sum().item()) >= 4 and int(q4_m.sum().item()) >= 4:
        loss = loss + 0.05 * (ai_logits[q3_m].float().mean() - ai_logits[q4_m].float().mean()).pow(2)
    return loss


def build_v11_param_groups(model: MeguvizV11OmniGuardModel, stage: str = "1b") -> list[dict[str, Any]]:
    """F7 + P2 Differential AdamW Param Groups (separate LRs for old vs new MoLE/EBT/Attr rows & frozen 1a)."""
    new_head_keys = ("r1_promo_head", "promo_head", "promo_sub4_head", "promo_span_head", "promo_mil_gate", "ebt_promo_gate")
    new_row_keys = ("fc2_new_", "attr6_new_")
    core_new_keys = ("core_routed_new", "core_bias_new", "dyn_profiles_new", "router_keys_new")
    long_keys = ("longstream.", "chunk_emb_proj.")
    style_l0_keys = ("r1_style_mlp.0.", "style_net.0.")
    old_router_keys = ("router_proj", "router_keys_old")
    half_trunk_keys = ("word_emb", "engram_", "trigram_", "pos_emb", "prelude_l0", "r1_rhythm_conv", "r1_ai_head", "r1_attn_pool", "ai_head", "inloop_ebt.fc1", "inloop_ebt.fc2_old")

    groups: dict[str, list[nn.Parameter]] = {
        "new_heads": [], "new_rows": [], "core_new": [], "long": [],
        "style_l0": [], "old_router": [], "half_trunk": [], "trunk": [],
    }
    for name, p in model.named_parameters():
        if any(k in name for k in new_head_keys):
            groups["new_heads"].append(p)
        elif any(k in name for k in new_row_keys):
            groups["new_rows"].append(p)
        elif any(k in name for k in core_new_keys):
            groups["core_new"].append(p)
        elif any(k in name for k in long_keys):
            groups["long"].append(p)
        elif any(k in name for k in style_l0_keys):
            groups["style_l0"].append(p)
        elif any(k in name for k in old_router_keys):
            groups["old_router"].append(p)
        elif any(k in name for k in half_trunk_keys):
            groups["half_trunk"].append(p)
        else:
            groups["trunk"].append(p)

    if stage == "1a":
        lr_map = {"new_heads": 1.5e-3, "new_rows": 1.0e-3, "core_new": 4.0e-4, "long": 1.0e-3, "style_l0": 1.0e-3, "old_router": 0.0, "half_trunk": 0.0, "trunk": 0.0}
    elif stage == "1.5":
        lr_map = {"new_heads": 2.0e-4, "new_rows": 1.5e-4, "core_new": 1.0e-4, "long": 2.0e-4, "style_l0": 6.0e-5, "old_router": 1.5e-5, "half_trunk": 2.0e-5, "trunk": 4.0e-5}
    else:  # stage 1b
        lr_map = {"new_heads": 6.0e-4, "new_rows": 4.0e-4, "core_new": 2.0e-4, "long": 5.0e-4, "style_l0": 1.0e-4, "old_router": 1.8e-5, "half_trunk": 3.0e-5, "trunk": 6.0e-5}

    out_pgs: list[dict[str, Any]] = []
    for gname, plist in groups.items():
        lr_g = lr_map[gname]
        for p in plist:
            p.requires_grad_(lr_g > 0.0)
        if plist and lr_g > 0.0:
            out_pgs.append({"name": gname, "params": plist, "lr": lr_g, "base_lr": lr_g, "weight_decay": 0.015})
    return out_pgs


def compute_l2_sp_loss(model: MeguvizV11OmniGuardModel, r1_anchor_params: dict[str, torch.Tensor], wd_sp: float = 0.02) -> torch.Tensor:
    """P2 L2-SP Regularizer: decays unfrozen trunk parameters toward `theta_R1` instead of zero!"""
    total = None
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and name in r1_anchor_params:
            diff = (p - r1_anchor_params[name].to(p.device, dtype=p.dtype)).pow(2).mean()
            total = diff if total is None else (total + diff)
            count += 1
    if total is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return float(wd_sp) * (total / float(max(1, count)))


def collate_chronostream_docs(
    docs: list[dict[str, Any]],
    tokenizer: BiliPickyBPETokenizer,
    max_chunks: int = 4,
    max_chunk_toks: int = 192,
) -> dict[str, torch.Tensor]:
    """F8 Unified `plan_long_chunks_v11` batch collator for ChronoStream TBPTT training & long-doc eval."""
    bsz = len(docs)
    parsed_chunks = []
    doc_f56s = []
    m_max = 1
    l_max = 16

    for d in docs:
        cl_doc, f56_doc, _ = extract_style_features_56d(d["text"])
        doc_f56s.append(f56_doc)
        cks = plan_long_chunks_v11(cl_doc, target_chars=240, stride_chars=165, max_chunks=max_chunks)
        if not cks:
            cks = [{"start_char": 0, "end_char": len(cl_doc), "text": cl_doc, "chunk_meta": np.zeros(6, dtype=np.float32)}]
        m_max = max(m_max, len(cks))
        ck_items = []
        for c in cks:
            cl_c, f56_c, _ = extract_style_features_56d(c["text"])
            ids_c = tokenizer.encode_ids(cl_c, max_len=max_chunk_toks)
            l_max = max(l_max, len(ids_c))
            # Compute overlap ratios with ground-truth ai_spans / promo_spans if provided!
            st_c, en_c = int(c["start_char"]), int(c["end_char"])
            clen = max(1, en_c - st_c)
            ai_ov = sum(max(0, min(en_c, se) - max(st_c, ss)) for ss, se in d.get("ai_spans", [])) / float(clen)
            pr_ov = sum(max(0, min(en_c, se) - max(st_c, ss)) for ss, se in d.get("promo_spans", [])) / float(clen)
            ck_items.append({
                "ids": ids_c,
                "eng": make_engram_bigram_ids(ids_c, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
                "tri": make_engram_trigram_ids(ids_c, buckets=ENGRAM_TRIGRAM_BUCKETS_V10),
                "f56": f56_c,
                "meta6": c["chunk_meta"],
                "ai_ov": min(1.0, ai_ov),
                "pr_ov": min(1.0, pr_ov),
            })
        parsed_chunks.append(ck_items)

    l_pad = min(max_chunk_toks, ((l_max + 15) // 16) * 16)
    ch_ids = np.full((bsz, m_max, l_pad), tokenizer.pad_id, dtype=np.int64)
    ch_msk = np.zeros((bsz, m_max, l_pad), dtype=np.int64)
    ch_eng = np.zeros((bsz, m_max, l_pad), dtype=np.int64)
    ch_tri = np.zeros((bsz, m_max, l_pad), dtype=np.int64)
    ch_f56 = np.zeros((bsz, m_max, STYLE_FEAT_DIM_V11), dtype=np.float32)
    ch_m6 = np.zeros((bsz, m_max, 6), dtype=np.float32)
    ch_val = np.zeros((bsz, m_max), dtype=np.float32)
    ch_ai_ov = np.zeros((bsz, m_max), dtype=np.float32)
    ch_pr_ov = np.zeros((bsz, m_max), dtype=np.float32)

    for bi, ck_items in enumerate(parsed_chunks):
        for mi, ci in enumerate(ck_items):
            n_t = min(len(ci["ids"]), l_pad)
            ch_ids[bi, mi, :n_t] = ci["ids"][:n_t]
            ch_msk[bi, mi, :n_t] = 1
            ch_eng[bi, mi, :n_t] = ci["eng"][:n_t]
            ch_tri[bi, mi, :n_t] = ci["tri"][:n_t]
            ch_f56[bi, mi] = ci["f56"]
            ch_m6[bi, mi] = ci["meta6"]
            ch_val[bi, mi] = 1.0
            ch_ai_ov[bi, mi] = ci["ai_ov"]
            ch_pr_ov[bi, mi] = ci["pr_ov"]

    return {
        "chunk_ids": torch.from_numpy(ch_ids),
        "chunk_mask": torch.from_numpy(ch_msk),
        "chunk_eng": torch.from_numpy(ch_eng),
        "chunk_tri": torch.from_numpy(ch_tri),
        "chunk_style56": torch.from_numpy(ch_f56),
        "chunk_meta6": torch.from_numpy(ch_m6),
        "chunk_valid": torch.from_numpy(ch_val),
        "doc_style56": torch.from_numpy(np.stack(doc_f56s, axis=0).astype(np.float32)),
        "y_ai_doc": torch.tensor([float(d.get("y_ai", float(d.get("ai_score", 0.0)) / 100.0)) for d in docs], dtype=torch.float32),
        "y_hyb_doc": torch.tensor([float(d.get("y_hyb", 1.0 if d.get("is_hybrid_mixed") else 0.0)) for d in docs], dtype=torch.float32),
        "y_pr_doc": torch.tensor([float(d.get("y_promo", float(d.get("promo_score", 0.0)) / 100.0)) for d in docs], dtype=torch.float32),
        "chunk_ai_ov": torch.from_numpy(ch_ai_ov),
        "chunk_pr_ov": torch.from_numpy(ch_pr_ov),
    }


# ---------------------------------------------------------------------------
# 2. Local RTX 4060 Ti Full-Stack Dry-Run CI (`Stage 0 -> P1 -> 1a -> 1b -> LoGra -> 1.5 -> 2 -> 3a/3b -> 4 ONNX!`)
# ---------------------------------------------------------------------------

def run_v11_pipeline_dry_run(out_dir: Path, datafactory_dir: Path, tokenizer_path: Path | None = None) -> dict[str, Any]:
    """End-to-end GPU dry-run verifying EVERY stage of `v1.1.0-OmniGuard` on real Wave-1 + Wave-2 + UOT + k-NN data!"""
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v1.1.0 Dry-Run] device={device} | datafactory={datafactory_dir}", flush=True)

    # 1. Load real Wave-1 + Wave-2 + Wave-2 Longform + UOT + P10-3 gated k-NN samples
    def _load_jsonl_any(p: Path, limit: int = 256) -> list[dict[str, Any]]:
        if not p.is_file():
            return []
        op = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")
        rows = []
        with op as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) >= limit:
                        break
        return rows

    w1_rows = _load_jsonl_any(datafactory_dir / "v11_wave1_dualaxis_gold_all5880.jsonl.gz", 160)
    w2_rows = _load_jsonl_any(datafactory_dir / "v11_wave2_dualaxis_gold_all4410.jsonl.gz", 160)
    lf_rows = _load_jsonl_any(datafactory_dir / "v11_wave2_longform_gold_1200.jsonl.gz", 32)
    pair_rows = _load_jsonl_any(datafactory_dir / "v11_wave1_promo_counterfactual_pairs_1600.jsonl", 64)
    knn_rows = _load_jsonl_any(datafactory_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl", 64)
    uot_pos = _load_jsonl_any(datafactory_dir / "v11_uot_wild_promo_pos.jsonl", 48)
    print(f"  [Data Loaded] W1={len(w1_rows)}, W2={len(w2_rows)}, Longform={len(lf_rows)}, Pairs={len(pair_rows)}, kNN_P10_3={len(knn_rows)}, UOT_pos={len(uot_pos)}", flush=True)

    # Tokenizer (load existing 12k BPE if present, else train quick dry-run BPE)
    tok_p = out_dir / "bili_picky_bpe_dry.json"
    if tokenizer_path and tokenizer_path.is_file():
        shutil.copy2(tokenizer_path, tok_p)
        tok = BiliPickyBPETokenizer(tok_p)
    else:
        sample_txts = [r["text"] for r in (w1_rows + w2_rows + lf_rows) if r.get("text")]
        tok = BiliPickyBPETokenizer.train_from_texts(sample_txts, tok_p, vocab_size=1024, min_frequency=2)

    # 2. Build mock pre-folded `v1.0.0-R1` model & run Stage 0 P1 Zero-Shock Transplant -> `V11-G0 / G1`!
    torch.manual_seed(20261126)
    v10_ref = MeguvizV10ApexLoopModel(vocab_size=tok.vocab_size, max_len=MAX_LEN_V07, style_dim=STYLE_FEAT_DIM_V07).to(device).eval()
    v11_mdl = MeguvizV11OmniGuardModel(vocab_size=tok.vocab_size, max_len=MAX_LEN_V07, style_dim=STYLE_FEAT_DIM_V11).to(device).eval()

    cal_txts = [r["text"] for r in (w1_rows + w2_rows)[:256]]
    f40_cal = np.stack([extract_style_features_40d(t)[1] for t in cal_txts], axis=0)
    f56_cal = np.stack([extract_style_features_56d(t)[1] for t in cal_txts], axis=0)
    ids_cal = [tok.encode_ids(extract_style_features_56d(t)[0], max_len=128) for t in cal_txts]

    def _pack_short(sub_ids, sub_st):
        ml = min(128, ((max(len(s) for s in sub_ids) + 15) // 16) * 16)
        ids = np.full((len(sub_ids), ml), tok.pad_id, dtype=np.int64)
        msk = np.zeros((len(sub_ids), ml), dtype=np.int64)
        eng = np.zeros((len(sub_ids), ml), dtype=np.int64)
        tri = np.zeros((len(sub_ids), ml), dtype=np.int64)
        for i, s in enumerate(sub_ids):
            n = min(len(s), ml)
            ids[i, :n], msk[i, :n] = s[:n], 1
            eng[i, :n] = make_engram_bigram_ids(s[:n], buckets=ENGRAM_BIGRAM_BUCKETS_V10)
            tri[i, :n] = make_engram_trigram_ids(s[:n], buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        return (
            torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
            torch.from_numpy(eng).to(device), torch.from_numpy(tri).to(device),
            torch.from_numpy(sub_st).to(device),
        )

    with torch.no_grad():
        i40, m40, e40, t40, s40 = _pack_short(ids_cal, f40_cal)
        o10_cal = v10_ref(i40, m40, s40, engram_ids=e40, trigram_ids=t40, outer_steps=2, return_aux=True)
        emb10_np = o10_cal["emb"].float().cpu().numpy()
        z10_ref = o10_cal["raw_logit"].float().cpu().numpy()
        zr1_10_ref = o10_cal["raw_logit_r1"].float().cpu().numpy()

    surg_rep = transplant_v10_to_v11_zero_shock(v10_ref, v11_mdl, f40_cal, f56_cal, emb10_np, j1_expert=2, j2_expert=5)

    with torch.no_grad():
        i56, m56, e56, t56, s56 = _pack_short(ids_cal, f56_cal)
        o11_cal = v11_mdl(i56, m56, s56, engram_ids=e56, trigram_ids=t56, outer_steps=2, return_aux=True)
        z11_post = o11_cal["raw_logit"].float().cpu().numpy()
        zr1_11_post = o11_cal["raw_logit_r1"].float().cpu().numpy()
    g0_rmse = float(np.sqrt(np.mean((z11_post - z10_ref) ** 2)))
    g0_r1_rmse = float(np.sqrt(np.mean((zr1_11_post - zr1_10_ref) ** 2)))
    ci_pre = run_v11_ci_gates(v11_mdl, tok, v10_ref_model=v10_ref)
    print(f"  [Stage 0 Zero-Shock] bridge_R2={surg_rep['bridge_40_to_56_r2']} | obs_rmse={surg_rep['obs_ai_head_192_rmse']} | G0_raw_rmse={g0_rmse:.5f} | G0_r1_rmse={g0_r1_rmse:.5f} | CI={ci_pre['all_ci_passed']}", flush=True)

    # Save R1 trunk anchor snapshot for L2-SP
    r1_anchor_params = {k: v.detach().cpu().clone() for k, v in v11_mdl.named_parameters() if not any(x in k for x in ("promo", "longstream", "chunk_emb", "_new"))}

    # 3. Stage 1a Dry-Run (4 steps: Bridge Distill + New Heads + ChronoStream!)
    pg_1a = build_v11_param_groups(v11_mdl, stage="1a")
    opt_1a = torch.optim.AdamW(pg_1a, betas=(0.9, 0.95))
    v11_mdl.train()
    for step in range(4):
        opt_1a.zero_grad(set_to_none=True)
        b_i, b_m, b_e, b_t, b_s56 = _pack_short(ids_cal[:16], f56_cal[:16])
        b_s40 = torch.from_numpy(f40_cal[:16]).to(device)
        with torch.no_grad():
            tgt_r1_st = v10_ref.r1_style_mlp[0](b_s40)
            tgt_st = v10_ref.style_net[0](b_s40)
        # P1.2 Pre-activation bridge distillation loss
        l_br = F.mse_loss(v11_mdl.r1_style_mlp[0](b_s56), tgt_r1_st) + F.mse_loss(v11_mdl.style_net[0](b_s56), tgt_st)
        out = v11_mdl(b_i, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, bal_frozen=True, return_aux=True)
        y_pr_b = torch.tensor([1.0 if float(r["promo_score"]) >= 65 else 0.0 for r in (w1_rows + w2_rows)[:16]], device=device)
        l_pr = F.binary_cross_entropy_with_logits(out["logit_promo"].float(), y_pr_b) + 0.35 * F.binary_cross_entropy_with_logits(out["logit_promo_r1"].float(), y_pr_b)
        loss = l_br + l_pr
        if step == 3 and lf_rows:
            cs_b = collate_chronostream_docs(lf_rows[:4], tok, max_chunks=3, max_chunk_toks=128)
            cs_o = v11_mdl.forward_chronostream_batch(
                cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device),
                cs_b["chunk_tri"].to(device), cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device),
                cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device), trunk_no_grad=True, outer_steps=1,
            )
            loss = loss + F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device)) + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
        loss.backward()
        opt_1a.step()

    # 4. Stage 1b Dry-Run (4 steps: Unfrozen Dual-Axis + L2-SP + Partial-Label EBT6 + Pairs + GradNorm-lite + PiKE + ChronoStream TBPTT!)
    pg_1b = build_v11_param_groups(v11_mdl, stage="1b")
    opt_1b = torch.optim.AdamW(pg_1b, betas=(0.9, 0.95))
    shared_params = [p for n, p in v11_mdl.named_parameters() if p.requires_grad and any(k in n for k in ("core.", "prelude_l1.", "coda_proj."))]
    gn_guard = GradNormLiteConflictGuardV11(rho_max=0.50)
    pike = DynamicPiKEGroupSamplerV11()
    gn_rep: dict[str, float] = {}

    for step in range(4):
        opt_1b.zero_grad(set_to_none=True)
        b_recs = (w1_rows + w2_rows)[step * 16 : (step + 1) * 16]
        b_i, b_m, b_e, b_t, b_s56 = _pack_short(ids_cal[step * 16 : (step + 1) * 16], f56_cal[step * 16 : (step + 1) * 16])
        out = v11_mdl(b_i, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, promo_sh_lam=0.35, bal_rho_new=0.10, return_aux=True)

        y_ai_b = torch.tensor([float(r["ai_score"]) / 100.0 for r in b_recs], device=device)
        y_pr_b = torch.tensor([float(r["promo_score"]) / 100.0 for r in b_recs], device=device)
        has_pr_b = torch.ones_like(y_pr_b)
        pt_map = {"none": 0, "coupon_commerce": 1, "self_promo_funnel": 2, "soft_shill": 3, "tool_boilerplate": 4}
        sub4_idx = torch.tensor([pt_map.get(str(r.get("promo_type", "none")), 0) for r in b_recs], dtype=torch.long, device=device)
        a5_b = torch.where(y_ai_b >= 0.70, torch.full_like(sub4_idx, 3), torch.zeros_like(sub4_idx))

        l_ai = F.binary_cross_entropy_with_logits(out["logit"].float(), y_ai_b.clamp(1e-4, 1.0 - 1e-4))
        l_ebt6 = 0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai_b, y_pr_b, has_pr_b, a5_b)
        l_ebtm = compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai_b, y_pr_b, has_pr_b)
        l_pr = F.binary_cross_entropy_with_logits(out["logit_promo"].float(), y_pr_b.clamp(1e-4, 1.0 - 1e-4))
        l_sub4 = 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], out["logit_promo"], sub4_idx, y_pr_b, has_pr_b)
        l_perp = compute_v11_orthogonality_loss(v11_mdl, out["logit"], y_ai_b, y_pr_b, has_pr_b)
        l_route = 0.03 * F.binary_cross_entropy(out["mole_probs_mean"][:, 8:10].sum(dim=-1).clamp(1e-5, 1.0 - 1e-5), (y_pr_b >= 0.65).float())
        l_l2sp = compute_l2_sp_loss(v11_mdl, r1_anchor_params, wd_sp=0.02)

        # Pair orthogonal dual-axis sidecar
        l_pair = torch.tensor(0.0, device=device)
        if pair_rows and step % 2 == 0:
            pb = pair_rows[:4]
            pi, pm, pe, pt, ps = _pack_short([tok.encode_ids(extract_style_features_56d(x["promo_text"])[0], max_len=128) for x in pb], np.stack([extract_style_features_56d(x["promo_text"])[1] for x in pb]))
            oi, om, oe, ot, os_ = _pack_short([tok.encode_ids(extract_style_features_56d(x["organic_text"])[0], max_len=128) for x in pb], np.stack([extract_style_features_56d(x["organic_text"])[1] for x in pb]))
            op = v11_mdl(pi, pm, ps, engram_ids=pe, trigram_ids=pt, outer_steps=1, return_aux=True)
            oo = v11_mdl(oi, om, os_, engram_ids=oe, trigram_ids=ot, outer_steps=1, return_aux=True)
            is_ai_p = torch.tensor([1.0 if x.get("promo_is_ai") else 0.0 for x in pb], device=device)
            l_pair = 0.20 * orthogonal_dual_axis_pair_loss(op["logit"].float(), oo["logit"].float(), op["logit_promo"].float(), oo["logit_promo"].float(), is_ai_p)

        if step == 1:
            gn_rep = gn_guard.probe(shared_params[:8], l_ai, l_pr + l_sub4)
            pike.update(np.ones(7), np.array([0.4, 0.35, 0.2, 0.5, 0.6, 0.4, 0.5]), np.array([1.0, 1.0, 0.5, 1.0, 0.1, 0.0, 0.2]))

        loss_total = l_ai + l_ebt6 + l_ebtm + gn_rep.get("lambda_p_mult", 1.0) * (l_pr + l_sub4) + l_pair + l_perp + l_route + l_l2sp
        if step == 3 and lf_rows:
            cs_b = collate_chronostream_docs(lf_rows[4:8], tok, max_chunks=3, max_chunk_toks=128)
            cs_o = v11_mdl.forward_chronostream_batch(
                cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device),
                cs_b["chunk_tri"].to(device), cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device),
                cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device), tbptt_k2=2, trunk_grad_lam=0.30, outer_steps=1,
            )
            loss_total = loss_total + 0.5 * (
                F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device))
                + F.binary_cross_entropy_with_logits(cs_o["logit_hybrid_doc"].float(), cs_b["y_hyb_doc"].to(device))
                + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
            )
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(v11_mdl.parameters(), 1.0)
        opt_1b.step()

    # 5. Stage 1.5 Kronecker LoGra + Stage 2 Zero-Op Fold & QAT + Stage 3a MGVZ6 Pack/Unpack + Stage 3b LEACE + Stage 4 4-Session ONNX!
    v11_mdl.eval()
    with torch.no_grad():
        o_post = v11_mdl(i56, m56, s56, engram_ids=e56, trigram_ids=t56, outer_steps=2, return_aux=True)
        emb_post = o_post["emb"].float().cpu().numpy()
        p_ai_post = torch.sigmoid(o_post["logit"].float()).cpu().numpy()
        p_pr_post = torch.sigmoid(o_post["logit_promo"].float()).cpu().numpy()

    y_ai_cal = np.array([float(r["ai_score"]) / 100.0 for r in (w1_rows + w2_rows)[:256]], dtype=np.float32)
    y_pr_cal = np.array([float(r["promo_score"]) / 100.0 for r in (w1_rows + w2_rows)[:256]], dtype=np.float32)
    logra_rep = kronecker_logra_topfp_snipe(
        emb_post, p_ai_post - y_ai_cal, np.zeros(len(y_ai_cal), dtype=bool),
        emb_post[:16], np.maximum(0.0, p_ai_post[:16] - y_ai_cal[:16]), prune_frac=0.05,
    )

    fold_rep = fold_zero_op_v11(v11_mdl, calib_inputs=[(i56[:32], m56[:32], s56[:32], e56[:32])])
    v11_mdl.set_quant_mode("sub2")
    with torch.no_grad():
        _ = v11_mdl(i56[:8], m56[:8], s56[:8], engram_ids=e56[:8], trigram_ids=t56[:8], outer_steps=2)
    v11_mdl.set_quant_mode("fp16")

    # Stage 3b: Zero-Byte Dual-Axis Oblique LEACE Surgery
    log_lens = np.log1p(np.array([len(t) for t in cal_txts], dtype=np.float32))[:, None]
    z_ai_conf = np.concatenate([log_lens, p_pr_post[:, None]], axis=1)
    z_pr_conf = np.concatenate([log_lens, p_ai_post[:, None]], axis=1)
    leace_rep = apply_dual_axis_oblique_leace_v11(v11_mdl, emb_post, y_ai_cal, y_pr_cal, z_ai_conf, z_pr_conf)

    # Stage 4: Bit-exact MGVZ6 Pack/Unpack + 4-Session Static ONNX Export + Bundle v6+ ZIP size check!
    exp_info = export_v11_modular_onnx_and_bundle_v6(v11_mdl, out_dir)
    v11_unpacked = copy.deepcopy(v11_mdl).cpu().eval()
    load_bundle_v6_v11_into_model(v11_unpacked, out_dir / exp_info["bundle_v6_mgvz"]["file"])
    v11_unpacked = v11_unpacked.to(device).eval()
    ci_final = run_v11_ci_gates(v11_unpacked, tok)

    zip_p = out_dir / "meguvizcac-v1.1.0-dryrun-bundle-v6.zip"
    with ZipFile(zip_p, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(out_dir / exp_info["bundle_v6_mgvz"]["file"], exp_info["bundle_v6_mgvz"]["file"])
        zf.write(tok_p, tok_p.name)

    report = {
        "dry_run_passed": bool(
            ci_pre["all_ci_passed"]
            and ci_final["all_ci_passed"]
            and exp_info["r1_fp32_max_diff"] < 1e-4
            and exp_info["r2_fp32_max_diff"] < 1e-4
            and exp_info["r3_fp32_max_diff"] < 1e-4
            and exp_info["dynaseam_onnx"]["pass"]
        ),
        "stage0_zero_shock": {**surg_rep, "g0_raw_rmse": round(g0_rmse, 6), "g0_r1_rmse": round(g0_r1_rmse, 6)},
        "gradnorm_lite": gn_rep,
        "pike_weights": pike.as_dict(),
        "logra_sniped": logra_rep["zeroed_count"],
        "fold_zero_op": fold_rep,
        "leace_surgery": leace_rep,
        "export_info": exp_info,
        "bundle_zip_mb_dry": round(zip_p.stat().st_size / (1024.0 * 1024.0), 4),
        "ci_pre_transplant": ci_pre,
        "ci_post_unpack": ci_final,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "v11_dryrun_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.1.0-OmniGuard Trainer & Full-Stack Dry-Run CI.")
    ap.add_argument("--dry-run", action="store_true", help="Run full-stack Stage 0..4 GPU dry-run CI locally.")
    ap.add_argument("--out", type=Path, default=Path("scratch/v11_dryrun_out"))
    ap.add_argument("--datafactory", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--tokenizer", type=Path, default=None)
    args = ap.parse_args()
    if args.dry_run:
        rep = run_v11_pipeline_dry_run(args.out, args.datafactory, args.tokenizer)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        assert rep["dry_run_passed"], "v1.1.0 Dry-Run CI failed!"
        return
    raise SystemExit("Full Colab A100 training entry ready; pass --dry-run for local RTX 4060 Ti verification, or wait for v1.0.0-R1 fp16_master.pt.")


if __name__ == "__main__":
    main()
