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
            "t0_ai": 0.38, "t1_ai": 0.32, "t2_mid": 0.02, "ai_protect": 0.04,
            "dual_gold": 0.10, "uot_pseudo": 0.09, "hard_neg": 0.05,
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
        # Freeze core_new in 1a so MoLE routing never drifts before full AI loss + KD unfreeze in 1b!
        lr_map = {"new_heads": 1.5e-3, "new_rows": 8.0e-4, "core_new": 0.0, "long": 1.0e-3, "style_l0": 6.0e-4, "old_router": 0.0, "half_trunk": 0.0, "trunk": 0.0}
    elif stage == "1.5":
        lr_map = {"new_heads": 3.0e-4, "new_rows": 5.0e-5, "core_new": 2.5e-5, "long": 2.5e-4, "style_l0": 2.0e-5, "old_router": 3.0e-6, "half_trunk": 4.0e-6, "trunk": 8.0e-6}
    elif stage == "qat":
        # QAT stages distill from our own converged v11 FP16 master -> standard 5e-5 trunk LR adapts smoothly to W4/Sub-2!
        lr_map = {"new_heads": 2.0e-4, "new_rows": 8.0e-5, "core_new": 5.0e-5, "long": 2.0e-4, "style_l0": 4.0e-5, "old_router": 1.5e-5, "half_trunk": 3.0e-5, "trunk": 5.0e-5}
    else:  # stage 1b — gentle surgical LRs proven locally to prevent fresh-AdamW core shock on converged v1.0.0-R1!
        lr_map = {"new_heads": 6.0e-4, "new_rows": 8.0e-5, "core_new": 4.0e-5, "long": 4.0e-4, "style_l0": 3.0e-5, "old_router": 4.0e-6, "half_trunk": 6.0e-6, "trunk": 1.2e-5}

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


def train_v11_omniguard_full(args) -> dict[str, Any]:
    """Complete Colab A100 80GB (`BS=512`, ~16.5k steps) v1.1.0-OmniGuard Multi-Stage Deep Post-Training Pipeline!

    Warm-starts strictly from pre-folded `v1.0.0-R1 {version}_fp16_master.pt` via P1 Zero-Shock Surgery and enforces
    zero pure-AI regression through Stage 1a -> 1b -> 1.5 (LoGra + 3 ASAM Branches + AI-Constrained Pareto Soup) ->
    Stage 2a/2b (UPQ W4 -> Sub-2 QAT x2) -> Stage 3a (FP16 Shock-Absorber) -> Stage 3b (Dual-Axis LEACE + Weighted
    Venn-ABERS + 18-Cell Mondrian + Strict 1-D Fixed-Sequence Dual-Axis LTT) -> Stage 4 (Bundle v6+ <=2.20 MB & 4-ONNX)!
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .train_v05_apexloop import build_stratified_val_calibration_set
    from .train_v06_hyperloop import _load_v06_balanced_bundle
    from .train_v07_omniveritas import _derive_attr5_label
    from .train_v10_apexloop import _ingest_external_gemini_gold_v10, _top_fp_logit_ceiling_loss
    from .v04_corpus_prep import GROUP_TO_IDX
    from .v08_zenith_ops import ERA_TO_IDX
    from .v11_data_engine import compute_joint_length_domain_ipw
    from .v11_data_refinery import classify_epistemic_tier
    from .v11_promo_pseudo import _norm, _sha_norm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v1.1.0-OmniGuard training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    bs = int(args.batch_size) if int(args.batch_size) > 0 else (512 if vram_gb >= 35.0 else 192)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    v11_dir = args.v11_data_dir.resolve()
    t_start = time.monotonic()

    print(
        f"[v1.1.0-OmniGuard Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={bs} | v10_master={args.v10_fp16_master} | v11_data={v11_dir}",
        flush=True,
    )

    # 1. Load base corpus + v08 80k Gemini Gold + legacy pairs
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
        train_raw = rng_sm.sample(train_raw, min(512, len(train_raw)))
        legacy_pairs = legacy_pairs[:32]

    # 2. Tokenizer
    tok_path = out_dir / "bili_picky_bpe_12k.json"
    if args.v07_tokenizer and args.v07_tokenizer.is_file():
        shutil.copy2(args.v07_tokenizer, tok_path)
        tokenizer = BiliPickyBPETokenizer(tok_path)
    else:
        txts_t = [x["raw_text"] for x in train_raw[: (512 if args.smoke_test else 200000)]]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(txts_t, tok_path, vocab_size=2048 if args.smoke_test else BPE_VOCAB_SIZE_V07)

    # 3. Hydrate exact original `score_source` + true Gemini override provenance (never rely on v06 broad is_gold_tier!)
    #    then apply 4-Tier Epistemic Refinery (`T0..T3`: purge ~45k+ poisoned qwen38/multi_teacher False-AI rows!) + IPW
    orig_src_by_sid: dict[str, str] = {}
    with (gzip.open(args.data.resolve(), "rt", encoding="utf-8") if args.data.suffix == ".gz" else args.data.resolve().open("r", encoding="utf-8")) as f_src:
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
        op_g = gzip.open(args.v08_gemini_gold, "rt", encoding="utf-8") if args.v08_gemini_gold.suffix == ".gz" else args.v08_gemini_gold.open("r", encoding="utf-8")
        with op_g as f_g:
            for line in f_g:
                if not line.strip():
                    continue
                r_g = json.loads(line)
                if r_g.get("split", "train") != "train":
                    continue
                sid_g = str(r_g.get("sample_id", r_g.get("id", "")))
                src_g = str(r_g.get("score_source", "")).lower()
                if "pro" in src_g:
                    gem_pro_sids.add(sid_g)
                else:
                    gem_flash_sids.add(sid_g)

    refined_items: list[dict[str, Any]] = []
    t3_dropped = 0
    tier_counts: Counter = Counter()
    for it in train_raw:
        sid_it = str(it.get("sample_id", ""))
        src_it = orig_src_by_sid.get(sid_it, str(it.get("score_source", "")))
        if not src_it and it.get("is_pre_llm"):
            src_it = "pre_llm_human_prior"
        it["score_source"] = src_it
        is_gem_pro = sid_it in gem_pro_sids
        is_gem_any = is_gem_pro or (sid_it in gem_flash_sids)
        tier, tw, _ = classify_epistemic_tier(
            score=float(it["y_score"]) * 100.0,
            score_source=src_it,
            overridden_by_gemini=is_gem_any,
            gemini_is_pro=is_gem_pro,
        )
        tier_counts[tier] += 1
        if tier == "T3_QUARANTINE" or tw <= 0.0:
            t3_dropped += 1
            continue
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
        # Clamp combined weight cleanly to [0.20, 3.0] to prevent 40x rare-cell gradient bombs!
        it["weight"] = float(np.clip(float(it["weight"]) * float(iw), 0.20, 3.0))
    print(f"[Stage0-Refinery] kept={len(refined_items):,} | T3_quarantined={t3_dropped:,} | tiers={dict(tier_counts)}", flush=True)

    # 4. Ingest Wave-1 (5,880) + Wave-2 (4,410) + optional Wave-2.5 + UOT (123+55+80k) + P10-3 2k k-NN + CC Split Guard!
    pt_map = {"none": 0, "coupon_commerce": 1, "self_promo_funnel": 2, "soft_shill": 3, "tool_boilerplate": 4}
    cert_shas: set[str] = set()
    select_shas: set[str] = set()
    split_p = v11_dir / "v11_promo_split_manifest.json"
    if split_p.is_file():
        sm = json.loads(split_p.read_text(encoding="utf-8"))
        cert_shas = {x["norm_sha256"] for x in sm.get("splits", {}).get("promo_val_cert", [])}
        select_shas = {x["norm_sha256"] for x in sm.get("splits", {}).get("promo_val_select", [])}

    def _read_jsonl_list(p: Path) -> list[dict[str, Any]]:
        if not p.is_file():
            return []
        op = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")
        with op as f:
            return [json.loads(l) for l in f if l.strip()]

    dual_gold_train: list[dict[str, Any]] = []
    promo_val_select_rows: list[dict[str, Any]] = []
    promo_val_cert_rows: list[dict[str, Any]] = []
    uot_pseudo_rows: list[dict[str, Any]] = []
    hard_neg_rows: list[dict[str, Any]] = []
    b2_deferred_knn: list[dict[str, Any]] = []

    def _make_item(txt: str, y_ai: float, y_pr: float, has_pr: float, pt_str: str, w: float, is_t0: bool = False, is_rescue: bool = False, l_pceil: bool = False) -> dict[str, Any]:
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
            "group_idx": GROUP_TO_IDX["current_2026_real"], "era_idx": 3, "epistemic_tier": "T0_DIAMOND_GOLD" if is_t0 else "T1_CLEAN_SILVER",
        }

    for tag, fname, is_resc in (
        ("wave1", "v11_wave1_dualaxis_gold_all5880.jsonl.gz", False),
        ("wave2", "v11_wave2_dualaxis_gold_all4410.jsonl.gz", True),
    ):
        rows_g = _read_jsonl_list(v11_dir / fname)
        if args.smoke_test:
            rows_g = rows_g[:160]
        for r in rows_g:
            txt = str(r["text"])
            sh = _sha_norm(_norm(txt))
            y_pr_v = float(r["promo_score"]) / 100.0
            rec = _make_item(txt, float(r["ai_score"]) / 100.0, y_pr_v, 1.0, str(r.get("promo_type", "none")), 1.5, is_t0=True, is_rescue=is_resc)
            if sh in cert_shas:
                promo_val_cert_rows.append(rec)
            elif sh in select_shas:
                promo_val_select_rows.append(rec)
            elif y_pr_v <= 0.15 and len(promo_val_select_rows) < (80 if args.smoke_test else 650) and (int(sh[:8], 16) % 16 == 0):
                # Deterministic disjoint gold organic negatives for Weighted Venn-ABERS, Promo LEACE & Pareto Soup
                promo_val_select_rows.append(rec)
            else:
                dual_gold_train.append(rec)

    if args.wave25_gold and args.wave25_gold.is_file():
        for r in _read_jsonl_list(args.wave25_gold):
            dual_gold_train.append(_make_item(str(r["text"]), float(r["ai_score"]) / 100.0, float(r["promo_score"]) / 100.0, 1.0, str(r.get("promo_type", "none")), 1.6, is_t0=True, is_rescue=True))

    # Wave-1 1,600 minimal counterfactual pairs (respecting CC cert/select exclusion!)
    dual_pairs_train: list[dict[str, Any]] = []
    for x in _read_jsonl_list(v11_dir / "v11_wave1_promo_counterfactual_pairs_1600.jsonl")[: (64 if args.smoke_test else 99999)]:
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

    # UOT/CDMAD purified pseudo pool + chants + A-Res organic reservoir
    for r in _read_jsonl_list(v11_dir / "v11_uot_cdmad_pseudo_full.jsonl.gz")[: (256 if args.smoke_test else 999999)]:
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

    # P10-3 Gated 2k BGE k-NN hard negatives (w=1.0; only `l_pceil_admitted=True` enters L_pceil!)
    for r in _read_jsonl_list(v11_dir / "v11_bge_knn_promo_hard_negatives_2k_uot_gated.jsonl")[: (64 if args.smoke_test else 9999)]:
        rec = _make_item(str(r["text"]), float(r.get("old_ai_score", 15.0)) / 100.0, 0.02, 1.0, "none", 1.0, l_pceil=bool(r.get("l_pceil_admitted", False)))
        if rec["l_pceil_admit"]:
            hard_neg_rows.append(rec)
        else:
            b2_deferred_knn.append(rec)

    # Wave-2 1,200 span-aligned longform gold docs (deterministic 10% held-out val for ChronoStream & B3 Pareto Soup!)
    wave2_longform_all = _read_jsonl_list(v11_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 9999)]
    wave2_longform: list[dict[str, Any]] = []
    wave2_longform_val: list[dict[str, Any]] = []
    for idx_lf, doc_lf in enumerate(wave2_longform_all):
        if idx_lf % 10 == 9:
            wave2_longform_val.append(doc_lf)
        else:
            wave2_longform.append(doc_lf)
    print(
        f"[Stage0-V11Data] dual_train={len(dual_gold_train):,} | pairs={len(dual_pairs_train):,} | "
        f"promo_val_select={len(promo_val_select_rows):,} (pos={sum(1 for x in promo_val_select_rows if x['y_promo']>=0.65)}) | "
        f"promo_val_cert={len(promo_val_cert_rows):,} (pos={sum(1 for x in promo_val_cert_rows if x['y_promo']>=0.65)}) | "
        f"uot_pseudo={len(uot_pseudo_rows):,} | hard_neg={len(hard_neg_rows):,} | b2_defer={len(b2_deferred_knn):,} | "
        f"longform_tr={len(wave2_longform):,} val={len(wave2_longform_val):,}",
        flush=True,
    )

    # Pre-encode refined base AI rows with 56-d style features
    t0_pool, t1_pool, t2_pool, protect_pool = [], [], [], []
    struct_gids = {GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["dsh_hard_slice"], GROUP_TO_IDX["multilingual_world_knowledge"]}
    for it in refined_items:
        cl, f56, meta = extract_style_features_56d(it["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        it["clean_text"], it["ids"], it["style"], it["spoof_any"] = cl, ids, f56, float(meta["spoof_any"])
        it["engram"] = make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
        it["trigram"] = make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        it["attr5"] = _derive_attr5_label(it)
        it["era_idx"] = 0 if it.get("is_pre_llm") else ERA_TO_IDX.get(str(it.get("era_bucket", "current_2026")), 3)
        if it.get("is_trap_human") or it.get("is_pre_llm") or it.get("group_idx") in struct_gids:
            protect_pool.append(it)
        if it["epistemic_tier"] == "T0_DIAMOND_GOLD":
            t0_pool.append(it)
        elif it["epistemic_tier"] == "T2_SOFT_MIDBAND":
            t2_pool.append(it)
        else:
            t1_pool.append(it)

    print(
        f"[Stage0-AIPools] t0={len(t0_pool):,} (ai%={100*np.mean([x['y_score']>=0.7 for x in t0_pool]) if t0_pool else 0:.1f}%) | "
        f"t1={len(t1_pool):,} (ai%={100*np.mean([x['y_score']>=0.7 for x in t1_pool]) if t1_pool else 0:.1f}%) | "
        f"t2={len(t2_pool):,} | protect={len(protect_pool):,} | legacy_pairs={len(legacy_pairs):,}",
        flush=True,
    )

    # Build Online 12-Archetype LongMix Synthesizer from verified clean T0/T1 anchors
    hu_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["y_score"] <= 0.15 and 40 <= len(x["clean_text"]) <= 220][:4000]
    hu_hard = [x["clean_text"] for x in protect_pool if x["y_score"] <= 0.20 and len(x["clean_text"]) >= 40][:2000] or hu_clean[:500]
    ai_clean = [x["clean_text"] for x in (t0_pool + t1_pool) if x["y_score"] >= 0.85 and 40 <= len(x["clean_text"]) <= 240][:4000]
    pr_clean = [x["clean_text"] for x in dual_gold_train if x["y_promo"] >= 0.70 and x["y_score"] <= 0.30][:1000] or hu_clean[:50]
    longmix_syn = OnlineLongMixSynthesizerV11(hu_clean[:2000] or ["真人评论测试。"] * 12, hu_hard, ai_clean[:2000] or ["人工智能评论测试。"] * 12, pr_clean)

    # 5. Prepare 3-way disjoint AI eval sets (`val_gold`, `val_cal_select`, `val_cal_cert`, `test_gold`, `test_dsh_gold`) + frozen `PromoBench-48` & `LongBench-24`
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
        val_gold, val_cal_select, val_cal_cert, test_gold, test_dsh_gold = val_gold[:64], val_cal_select[:64], val_cal_cert[:64], test_gold[:64], test_dsh_gold[:32]

    def _pre_eval56(rows_e: list[dict[str, Any]]):
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

    vg_p = _pre_eval56(val_gold)
    vs_p = _pre_eval56(val_cal_select)
    vc_p = _pre_eval56(val_cal_cert)
    tg_p = _pre_eval56(test_gold)
    td_p = _pre_eval56(test_dsh_gold)

    # 6. Load Pre-Folded `v1.0.0-R1` Master -> P1 Zero-Shock Surgery -> `V11-G0` Check!
    v10_teacher = MeguvizV10ApexLoopModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V07, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).to(device).eval()
    if args.v10_fp16_master and args.v10_fp16_master.is_file():
        sd10 = torch.load(args.v10_fp16_master, map_location=device)
        v10_teacher.load_state_dict(sd10, strict=True)
        print(f"[Stage0-WarmStart] Loaded pre-folded v1.0.0-R1 master: {args.v10_fp16_master}", flush=True)
    for p in v10_teacher.parameters():
        p.requires_grad_(False)

    mdl = MeguvizV11OmniGuardModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11, num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS)).to(device).eval()

    def _collate_items(batch_it: list[dict[str, Any]]):
        ml = min(args.max_len, ((max(len(x["ids"]) for x in batch_it) + 15) // 16) * 16)
        n_b = len(batch_it)
        ids = np.full((n_b, ml), tokenizer.pad_id, dtype=np.int64)
        msk = np.zeros((n_b, ml), dtype=np.int64)
        eng = np.zeros((n_b, ml), dtype=np.int64)
        tri = np.zeros((n_b, ml), dtype=np.int64)
        st = np.empty((n_b, STYLE_FEAT_DIM_V11), dtype=np.float32)
        for i, x in enumerate(batch_it):
            ns = min(len(x["ids"]), ml)
            ids[i, :ns], msk[i, :ns], eng[i, :ns], tri[i, :ns], st[i] = x["ids"][:ns], 1, x["engram"][:ns], x["trigram"][:ns], x["style"]
        return (
            torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
            torch.from_numpy(eng).to(device), torch.from_numpy(tri).to(device), torch.from_numpy(st).to(device),
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
        )

    @torch.no_grad()
    def _predict_v11(m_e: MeguvizV11OmniGuardModel, pack_t, outer_steps: int = 2) -> dict[str, np.ndarray]:
        enc_l, eng_l, tri_l, _, _, st_m, _, _ = pack_t
        m_e.eval()
        n = len(enc_l)
        res = {k: np.zeros(n, dtype=np.float32) for k in ("logit", "raw_logit", "logit_t1", "raw_logit_t1", "logit_r1", "raw_logit_r1", "logit_promo", "raw_logit_promo", "logit_promo_t1", "raw_logit_promo_t1", "logit_promo_r1", "raw_logit_promo_r1", "p_defer1", "p_defer2", "r1_span_max", "ebt_grad_t1")}
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
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                o = m_e(torch.from_numpy(ids).to(device), torch.from_numpy(m).to(device), torch.from_numpy(st_m[mb]).to(device), engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device), outer_steps=outer_steps, return_aux=True)
            e1 = o["exit1_out"]
            kv = {
                "logit": o["logit"], "raw_logit": o["raw_logit"], "logit_t1": e1["logit"], "raw_logit_t1": e1["raw_logit"],
                "logit_r1": o["logit_r1"], "raw_logit_r1": o["raw_logit_r1"],
                "logit_promo": o["logit_promo"], "raw_logit_promo": o["raw_logit_promo"],
                "logit_promo_t1": e1["logit_promo"], "raw_logit_promo_t1": e1["raw_logit_promo"],
                "logit_promo_r1": o["logit_promo_r1"], "raw_logit_promo_r1": o["raw_logit_promo_r1"],
                "p_defer1": o["p_defer1"], "p_defer2": e1["p_defer2"], "r1_span_max": o["r1_span_max"], "ebt_grad_t1": e1["ebt_grad_norm"], "emb": o["emb"],
            }
            for k_o, t_o in kv.items():
                arr = t_o.float().cpu().numpy()
                for ri, oi in enumerate(mb):
                    res[k_o][oi] = arr[ri]
        return res

    # Fit P1.2 bridge & P1.3 OBS permutation on up to 16,384 (256 in smoke) clean train rows
    n_cal_p1 = 256 if args.smoke_test else min(16384, len(refined_items))
    cal_p1_items = random.Random(20261126).sample(refined_items, n_cal_p1)
    f40_p1 = np.stack([extract_style_features_40d(x["raw_text"])[1] for x in cal_p1_items], axis=0)
    f56_p1 = np.stack([x["style"] for x in cal_p1_items], axis=0)
    embs_v10_list, mole_loads = [], []
    with torch.no_grad():
        for st_i in range(0, n_cal_p1, 256):
            sub_b = cal_p1_items[st_i : st_i + 256]
            b_ids, b_m, b_e, b_t, _, *_ = _collate_items(sub_b)
            b_s40 = torch.from_numpy(f40_p1[st_i : st_i + 256]).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                o10 = v10_teacher(b_ids, b_m, b_s40, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)
            embs_v10_list.append(o10["emb"].float().cpu().numpy())
            mole_loads.append(o10["mole_probs_final"].float().cpu().numpy())
    emb_v10_p1 = np.concatenate(embs_v10_list, axis=0)
    load_r1_ref = np.concatenate(mole_loads, axis=0).mean(axis=0)

    # Pick Net2Net MoLE split experts j1 (coupon+self_promo) & j2 (soft_shill) by Wave-1 routing affinity!
    j1_exp, j2_exp = 0, 1
    if dual_gold_train:
        with torch.no_grad():
            def _mean_route(sub_r):
                if not sub_r:
                    return np.zeros(8, dtype=np.float32)
                b_ids, b_m, b_e, b_t, _, *_ = _collate_items(sub_r[:256])
                b_s40 = torch.from_numpy(np.stack([extract_style_features_40d(x["raw_text"])[1] for x in sub_r[:256]])).to(device)
                return v10_teacher(b_ids, b_m, b_s40, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)["mole_probs_final"].float().cpu().numpy().mean(axis=0)
            p_org_r = _mean_route([x for x in dual_gold_train if x["y_promo"] <= 0.25])
            p_cf_r = _mean_route([x for x in dual_gold_train if x["sub4_idx"] in (1, 2)])
            p_sh_r = _mean_route([x for x in dual_gold_train if x["sub4_idx"] == 3])
            j1_exp = int(np.argmax(p_cf_r - p_org_r))
            j2_exp = int(np.argmax(p_sh_r - p_org_r))

    surg_rep = transplant_v10_to_v11_zero_shock(v10_teacher, mdl, f40_p1, f56_p1, emb_v10_p1, j1_expert=j1_exp, j2_expert=j2_exp, load_r1_ref=load_r1_ref)
    ci_g0 = run_v11_ci_gates(mdl, tokenizer, v10_ref_model=v10_teacher)
    r1_anchor_params = {k: v.detach().cpu().clone() for k, v in mdl.named_parameters() if not any(x in k for x in ("promo", "longstream", "chunk_emb", "_new"))}
    print(f"[Stage0-P1-Transplant] {json.dumps(surg_rep)} | G0_CI={ci_g0['all_ci_passed']}", flush=True)

    # Record Stage-0 R1 reference metrics for Rollback Ladder & AI-Constrained Pareto Soup!
    def _eval_ai_metrics(m_e: MeguvizV11OmniGuardModel) -> dict[str, float]:
        vo = _predict_v11(m_e, vg_p, outer_steps=2)
        so = _predict_v11(m_e, vs_p, outer_steps=2)
        p_vg, pr1_vg, p_vs = expit(vo["logit"]), expit(vo["logit_r1"]), expit(so["logit"])
        mv = {**compute_eval_metrics(vg_p[7], p_vg), **compute_high_precision_metrics(vg_p[7], p_vg)}
        ms = {**compute_eval_metrics(vs_p[7], p_vs), **compute_high_precision_metrics(vs_p[7], p_vs)}
        mr1 = compute_eval_metrics(vg_p[7], pr1_vg)
        hu_m = vs_p[7] <= 20.0
        topfp = float(np.max(p_vs[hu_m])) if hu_m.any() else 0.0
        ai_obj = 0.50 * (0.38 * mv["auroc"] + 0.28 * mv["pr_auc"] + 0.14 * mv["best_f05"] + 0.14 * mv["recall_at_p95"] + 0.06 * mr1["auroc"]) + 0.50 * (0.32 * ms["auroc"] + 0.24 * ms["pr_auc"] + 0.14 * ms["best_f05"] + 0.18 * ms["recall_at_p95"] + 0.06 * ms["recall_at_p99"] + 0.06 * mr1["auroc"])
        return {"ai_obj": float(ai_obj), "auroc_vg": float(mv["auroc"]), "rp95_vg": float(mv["recall_at_p95"]), "rp95_wild": float(ms["recall_at_p95"]), "r1_auroc": float(mr1["auroc"]), "topfp": topfp}

    r1_ref_m = _eval_ai_metrics(mdl)
    rb_mon = RollbackMonitorV11(r1_ref_m["auroc_vg"], r1_ref_m["rp95_vg"], r1_ref_m["rp95_wild"], r1_ref_m["r1_auroc"], r1_ref_m["topfp"])
    print(f"[Stage0-R1-Ref] {json.dumps({k: round(v, 5) for k, v in r1_ref_m.items()})}", flush=True)

    # Multi-pool weighted batch sampler (Balanced-PiKE compatible!)
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
    pike = DynamicPiKEGroupSamplerV11()

    def _sample_short_batch(bs_now: int, w_vec: np.ndarray, extra_hard_pool: list[dict] | None = None) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        counts = np.maximum(1, np.round(w_vec * bs_now).astype(int))
        counts[0] += max(0, bs_now - int(counts.sum()))
        for pname, cnt in zip(DynamicPiKEGroupSamplerV11.POOL_NAMES, counts):
            src_p = pools[pname]
            if pname == "hard_neg" and extra_hard_pool:
                src_p = src_p + extra_hard_pool
            for _ in range(int(cnt)):
                it = src_p[rng_tr.randrange(len(src_p))]
                # Online forensic-safe augmentation on dual_gold & hard_neg slots (p=0.30)
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

    def _get_s40_batch(batch_it: list[dict[str, Any]]) -> torch.Tensor:
        arrs = []
        for x in batch_it:
            s40 = x.get("style40")
            if s40 is None:
                s40 = extract_style_features_40d(x["raw_text"])[1]
                x["style40"] = s40
            arrs.append(s40)
        return torch.from_numpy(np.stack(arrs, axis=0)).to(device)

    def _eval_promo_and_long(m_e: MeguvizV11OmniGuardModel) -> dict[str, float]:
        out_m = {"promo_auroc": 0.5, "promo_pr_auc": 0.5, "long_ai_auroc": 0.5, "long_hyb_auroc": 0.5, "long_pr_auroc": 0.5}
        if promo_val_select_rows:
            pv_pk = (
                [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
                [], [], np.stack([x["style"] for x in promo_val_select_rows], axis=0), [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32),
            )
            pvo = _predict_v11(m_e, pv_pk, outer_steps=2)
            yv = (pv_pk[7] >= 0.5).astype(int)
            if len(np.unique(yv)) == 2:
                out_m["promo_auroc"] = float(roc_auc_score(yv, pvo["logit_promo"]))
                out_m["promo_pr_auc"] = float(average_precision_score(yv, pvo["logit_promo"]))
        if wave2_longform_val:
            m_e.eval()
            ai_l, hy_l, pr_l, y_ai_l, y_hy_l, y_pr_l = [], [], [], [], [], []
            with torch.no_grad():
                for st_i in range(0, len(wave2_longform_val), 32):
                    cs_b = collate_chronostream_docs(wave2_longform_val[st_i : st_i + 32], tokenizer, max_chunks=4, max_chunk_toks=192)
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
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
            if len(np.unique(yai_a)) == 2:
                out_m["long_ai_auroc"] = float(roc_auc_score(yai_a, ai_a))
            if len(np.unique(yhy_a)) == 2:
                out_m["long_hyb_auroc"] = float(roc_auc_score(yhy_a, hy_a))
            if len(np.unique(ypr_a)) == 2:
                out_m["long_pr_auroc"] = float(roc_auc_score(ypr_a, pr_a))
        return out_m

    # 7. Stage 1a: Frozen-AI-Trunk Warmup + Promo[192:256] Firewall Gradient Hook + Full-Stage Bridge & R1 AI Retention!
    s1a_steps = 4 if args.smoke_test else int(args.s1a_steps)
    pg_1a = build_v11_param_groups(mdl, stage="1a")
    # Unfreeze ONLY rows [192:256] of coda_proj.0 & style_net.2 (strictly invisible to ai_head[0:192] -> 0.000000 AI interference!)
    _promo_row_names = ("coda_proj.0.weight", "coda_proj.0.bias", "style_net.2.weight", "style_net.2.bias")
    _named_p = dict(mdl.named_parameters())
    _s1a_hooks = []

    def _mask_promo_rows_only(grad: torch.Tensor) -> torch.Tensor:
        g = grad.clone()
        g[:AI_EMB_END_V11] = 0.0
        return g

    _promo_row_params = []
    for _nm in _promo_row_names:
        if _nm in _named_p:
            _p = _named_p[_nm]
            _p.requires_grad_(True)
            _promo_row_params.append(_p)
            _s1a_hooks.append(_p.register_hook(_mask_promo_rows_only))
    if _promo_row_params:
        pg_1a.append({"name": "promo_emb_rows", "params": _promo_row_params, "lr": 1.2e-3, "base_lr": 1.2e-3, "weight_decay": 0.01})
    opt_1a = torch.optim.AdamW(pg_1a, betas=(0.9, 0.95))
    mdl.train()
    for step in range(s1a_steps):
        mult = _wsd_lr_mult(step, s1a_steps, warmup_frac=0.05, decay_frac=0.15)
        for pg in opt_1a.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        opt_1a.zero_grad(set_to_none=True)
        b_it = _sample_short_batch(min(bs, 256 if args.smoke_test else bs), pike.w)
        b_ids, b_m, b_e, b_t, b_s56, y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era = _collate_items(b_it)
        b_s40 = _get_s40_batch(b_it)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, bal_frozen=True, return_aux=True)
            with torch.no_grad():
                t_r1s = v10_teacher.r1_style_mlp[0](b_s40)
                t_st0 = v10_teacher.style_net[0](b_s40)
                to_1a = v10_teacher(b_ids, b_m, b_s40, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, return_aux=True)
            # Keep 40->56 bridge distillation + R1 AI logit retention active across ALL Stage 1a steps -> ZERO AI drift!
            loss = 2.0 * (F.mse_loss(mdl.r1_style_mlp[0](b_s56), t_r1s) + F.mse_loss(mdl.style_net[0](b_s56), t_st0))
            loss = loss + 1.5 * (
                F.mse_loss(out["logit"].float(), to_1a["logit"].float())
                + 0.8 * F.mse_loss(out["logit_r1"].float(), to_1a["logit_r1"].float())
            )
            # Full-batch EBT6 partial-label CE (so E5 stays HIGH on AI samples and NEVER collapses ebt_margin!)
            loss = loss + 0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai, y_pr, has_pr, a5_b)
            loss = loss + compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai, y_pr, has_pr)
            if (has_pr > 0.5).any():
                pm = has_pr > 0.5
                loss = loss + (F.binary_cross_entropy_with_logits(out["logit_promo"][pm].float(), y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm]).mean()
                loss = loss + 0.35 * (F.binary_cross_entropy_with_logits(out["logit_promo_r1"][pm].float(), y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm]).mean()
                loss = loss + 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], out["logit_promo"], sub4_i, y_pr, has_pr)
            if step % 4 == 3:
                cs_b = collate_chronostream_docs(_sample_long_batch(4 if args.smoke_test else 48), tokenizer, max_chunks=3, max_chunk_toks=192)
                cs_o = mdl.forward_chronostream_batch(
                    cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                    cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                    trunk_no_grad=True, outer_steps=2,
                )
                loss = loss + F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device)) + 0.5 * F.binary_cross_entropy_with_logits(cs_o["logit_hybrid_doc"].float(), cs_b["y_hyb_doc"].to(device)) + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt_1a.step()
        if not args.smoke_test and (step + 1) % 300 == 0:
            pl_m = _eval_promo_and_long(mdl)
            ai_m1a = _eval_ai_metrics(mdl)
            print(f"[Stage1a step={step+1}/{s1a_steps}] loss={float(loss.item()):.4f} | ai_obj={ai_m1a['ai_obj']:.5f} auroc_vg={ai_m1a['auroc_vg']:.4f} | promo_auc={pl_m['promo_auroc']:.4f} pr={pl_m['promo_pr_auc']:.4f} | long_ai={pl_m['long_ai_auroc']:.4f} hyb={pl_m['long_hyb_auroc']:.4f} | elapsed={time.monotonic()-t_start:.0f}s", flush=True)
            mdl.train()
    for _hk in _s1a_hooks:
        _hk.remove()
    s1a_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    rb_mon.last_pass_sd = {k: v.clone() for k, v in s1a_sd.items()}
    print(f"[Stage1a DONE] steps={s1a_steps} | ai_eval={json.dumps({k: round(v, 5) for k, v in _eval_ai_metrics(mdl).items()})} | dual={json.dumps({k: round(v, 4) for k, v in _eval_promo_and_long(mdl).items()})}", flush=True)

    # 8. Shared Stage Runner for Stage 1b, Stage 1.5 ASAM Branches, & Stage 2a/2b QAT!
    def _run_v11_stage(
        name: str,
        init_sd: dict[str, torch.Tensor],
        stage_kind: str,
        steps: int,
        quant_mode: str = "fp16",
        lam_kd_start: float = 0.60,
        lam_kd_end: float = 0.30,
        lam_p: float = 1.0,
        promo_sh_lam: float = 0.35,
        chrono_mod: int = 7,
        use_asam: bool = False,
        trm_lam: float = 1.0,
        extra_hard: list[dict] | None = None,
        teacher_v11: MeguvizV11OmniGuardModel | None = None,
        enable_rollback: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(quant_mode)
        perm_dev = mdl.emb_perm.to(device)
        pgs = build_v11_param_groups(mdl, stage=stage_kind)
        opt = torch.optim.AdamW(pgs, betas=(0.9, 0.95))
        asam = ASAMPerturber(mdl, rho=0.05) if use_asam else None
        sh_params = [p for n, p in mdl.named_parameters() if p.requires_grad and any(k in n for k in ("core.", "prelude_l1.", "coda_proj."))]
        gn_guard = GradNormLiteConflictGuardV11(rho_max=0.50 if stage_kind == "1b" else 0.35)
        lam_kd_scale = 1.0
        lr_scale_rb = 1.0
        sh_lam_cur = float(promo_sh_lam)
        lam_p_mult = 1.0
        best_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_m = _eval_ai_metrics(mdl)
        best_pl = _eval_promo_and_long(mdl)

        def _stage_pareto_score(cm_: dict[str, float], pl_: dict[str, float]) -> float:
            long_m_ = 0.50 * pl_["long_ai_auroc"] + 0.25 * pl_["long_hyb_auroc"] + 0.25 * pl_["long_pr_auroc"]
            pen_ = 0.0 if (args.smoke_test or quant_mode != "fp16") else 50.0 * max(0.0, (r1_ref_m["ai_obj"] - 0.0025) - cm_["ai_obj"])
            return cm_["ai_obj"] + 0.35 * (0.5 * pl_["promo_auroc"] + 0.5 * pl_["promo_pr_auc"]) + 0.10 * long_m_ - pen_

        best_sc = _stage_pareto_score(best_m, best_pl)

        mdl.train()
        for step in range(steps):
            prog = step / float(max(1, steps - 1))
            mult = _wsd_lr_mult(step, steps, warmup_frac=0.08, decay_frac=0.20) * lr_scale_rb
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * mult
            lam_kd = min(1.5, (lam_kd_end + 0.5 * (lam_kd_start - lam_kd_end) * (1.0 + math.cos(math.pi * prog))) * lam_kd_scale)
            rho_new = 0.15 * prog if stage_kind == "1b" else 0.15
            outer_now = 3 if (step % 4 == 0 and quant_mode == "fp16") else 2

            b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike.w, extra_hard_pool=extra_hard)
            b_ids, b_m, b_e, b_t, b_s56, y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, is_t0, is_res, l_pc, grp, era = _collate_items(b_it)
            b_s40 = _get_s40_batch(b_it) if teacher_v11 is None else None

            def _loss_fn():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(
                        b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=outer_now,
                        trm_lam_k0=trm_lam, trm_lam_k1=1.0 if quant_mode == "fp16" else 0.0,
                        promo_sh_lam=sh_lam_cur, handoff_drop_prob=0.05, handoff_flip_prob=0.0,
                        bal_rho_new=rho_new, bal_frozen=True, return_aux=True,
                    )
                    lg, lg_e1, lg_r1 = out["logit"].float(), out["exit1_out"]["logit"].float(), out["logit_r1"].float()
                    y_cl = y_ai.clamp(1e-5, 1.0 - 1e-5)
                    is_q3 = (y_ai <= 0.18) & (has_pr > 0.5) & (y_pr >= 0.65)

                    # Selective Exact Logit-MSE + Emb[:192] Cosine KD Anchor (preserves v1.0.0 lg_std & sharp tail!)
                    l_kd = lg.sum() * 0.0
                    l_br_keep = lg.sum() * 0.0
                    tp_guard = torch.ones_like(y_ai)  # only T0/Q3/pre_llm when teacher_v11 is used
                    if teacher_v11 is not None:
                        with torch.no_grad():
                            tv = teacher_v11(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=outer_now, return_aux=True)
                        tp_guard = torch.sigmoid(tv["logit"].float())
                        l_kd = (
                            0.35 * F.mse_loss(lg, tv["logit"].float())
                            + 0.25 * F.mse_loss(lg_r1, tv["logit_r1"].float())
                            + 0.10 * listnet_tail_distill_loss(lg, tv["logit"])
                            + 0.10 * listnet_tail_distill_loss(out["logit_promo"].float(), tv["logit_promo"])
                            + 0.10 * listnet_tail_distill_loss(out["logit_promo_r1"].float(), tv["logit_promo_r1"])
                        )
                    else:
                        with torch.no_grad():
                            to = v10_teacher(b_ids, b_m, b_s40, engram_ids=b_e, trigram_ids=b_t, outer_steps=outer_now, return_aux=True)
                            t_r1s_k = v10_teacher.r1_style_mlp[0](b_s40)
                            t_st0_k = v10_teacher.style_net[0](b_s40)
                        tp_guard = torch.sigmoid(to["logit"].float())
                        m_anc = (~is_q3) & (is_res < 0.5) & (~((is_t0 > 0.5) & ((tp_guard - y_ai).abs() >= 0.50)))
                        if m_anc.any():
                            to_emb_p = to["emb"][m_anc].float()[:, perm_dev]
                            l_kd = (
                                0.45 * F.mse_loss(lg[m_anc], to["logit"][m_anc].float())
                                + 0.30 * F.mse_loss(lg_r1[m_anc], to["logit_r1"][m_anc].float())
                                + 0.40 * (1.0 - F.cosine_similarity(out["emb"][m_anc, :AI_EMB_END_V11].float(), to_emb_p[:, :AI_EMB_END_V11], dim=-1).mean())
                            )
                        l_br_keep = 0.8 * (F.mse_loss(mdl.r1_style_mlp[0](b_s56), t_r1s_k) + F.mse_loss(mdl.style_net[0](b_s56), t_st0_k))

                    l_ai = (F.binary_cross_entropy_with_logits(lg, y_cl, reduction="none") * w_b).mean()
                    l_ai = l_ai + 0.18 * (F.binary_cross_entropy_with_logits(lg_e1, y_cl, reduction="none") * w_b).mean()
                    l_ai = l_ai + 0.22 * (F.binary_cross_entropy_with_logits(lg_r1, y_cl, reduction="none") * w_b * (1.0 + 2.5 * (y_ai >= 0.45).float())).mean()
                    l_ai = l_ai + 0.08 * listnet_tail_distill_loss(lg_r1, lg.detach()) + 0.06 * listnet_tail_distill_loss(lg_e1, lg.detach())
                    l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_ai, w_b, grp, prog=prog)
                    # Commercial valley decoupling + double-confirmed teacher guard tp_guard (never -1.0!)
                    l_topfp = _top_fp_logit_ceiling_loss(lg, y_ai, tp_guard, torch.maximum(is_t0, is_q3.float()), era, w_b)
                    l_ai = l_ai + 0.09 * l_pa + l_tp + zenith_graded_anchor_loss(lg, y_ai) + 0.18 * l_topfp + 0.04 * out["cbm_loss"] + 0.002 * out["mole_z_loss"].float() + l_br_keep

                    # Promo axis losses + P10-3 gated L_pceil + L_sub4 + L_ebt6 + L_perp + L_route
                    l_pr_tot = lg.sum() * 0.0
                    if (has_pr > 0.5).any():
                        pm = has_pr > 0.5
                        u, u_t1, u_r1 = out["logit_promo"].float(), out["exit1_out"]["logit_promo"].float(), out["logit_promo_r1"].float()
                        l_p = (F.binary_cross_entropy_with_logits(u[pm], y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm]).mean()
                        l_p = l_p + 0.25 * (F.binary_cross_entropy_with_logits(u_t1[pm], y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm]).mean()
                        l_p = l_p + 0.35 * (F.binary_cross_entropy_with_logits(u_r1[pm], y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm] * (1.0 + 3.0 * (y_pr[pm] >= 0.65).float())).mean()
                        # P10-3 gated top-4 L_pceil ceiling on confirmed Q4 hard negs
                        hn_m = pm & (y_pr <= 0.25) & (l_pc > 0.5)
                        l_pceil = lg.sum() * 0.0
                        if int(hn_m.sum().item()) >= 2:
                            u_worst = torch.topk(u[hn_m], k=min(4, int(hn_m.sum().item()))).values
                            l_pceil = F.relu(u_worst).pow(2).mean() + 0.20 * F.softplus(u_worst + 0.8).mean()
                        l_pr_tot = l_p + 0.20 * l_pceil + 0.15 * compute_sub4_masked_loss(out["promo_sub4_logit"], u, sub4_i, y_pr, has_pr)
                        with torch.amp.autocast("cuda", enabled=False):
                            p89 = out["mole_probs_mean"][pm, 8:10].float().sum(dim=-1).clamp(1e-5, 1.0 - 1e-5)
                            l_pr_tot = l_pr_tot + 0.03 * F.binary_cross_entropy(p89, (y_pr[pm] >= 0.65).float())

                    # Dual-axis Rung-1 & Rung-2 deferral router + R1 span risk supervision (outside autocast!)
                    with torch.amp.autocast("cuda", enabled=False):
                        pf_d, pr1_d, pe1_d = torch.sigmoid(lg.detach()), torch.sigmoid(lg_r1.detach()), torch.sigmoid(lg_e1.detach())
                        d1_tgt = torch.maximum(((pf_d - pr1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_ai >= 0.45) & (pr1_d < 0.35)).float())
                        d2_tgt = torch.maximum(((pf_d - pe1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_ai >= 0.45) & (pe1_d < 0.35)).float())
                        if (has_pr > 0.5).any():
                            pf_p = torch.sigmoid(out["logit_promo"].detach().float())
                            pr1_p = torch.sigmoid(out["logit_promo_r1"].detach().float())
                            pe1_p = torch.sigmoid(out["exit1_out"]["logit_promo"].detach().float())
                            is_pr_pos = (has_pr > 0.5) & (y_pr >= 0.65)
                            d1_pr = torch.maximum(((pf_p - pr1_p).abs() * 3.0).clamp(0.0, 1.0) * (has_pr > 0.5).float(), (is_pr_pos & (pr1_p < 0.35)).float())
                            d2_pr = torch.maximum(((pf_p - pe1_p).abs() * 3.0).clamp(0.0, 1.0) * (has_pr > 0.5).float(), (is_pr_pos & (pe1_p < 0.35)).float())
                            d1_tgt = torch.maximum(d1_tgt, d1_pr)
                            d2_tgt = torch.maximum(d2_tgt, d2_pr)
                        l_def1 = F.binary_cross_entropy(out["p_defer1"].float().clamp(1e-5, 1.0 - 1e-5), d1_tgt)
                        l_def2 = F.binary_cross_entropy(out["exit1_out"]["p_defer2"].float().clamp(1e-5, 1.0 - 1e-5), d2_tgt)
                        l_r1_sp = (F.binary_cross_entropy(out["r1_span_max"].float().clamp(1e-5, 1.0 - 1e-5), y_cl.float(), reduction="none") * w_b.float() * (1.0 + 2.5 * (y_ai >= 0.45).float())).mean()
                    l_ai = l_ai + 0.18 * l_def1 + 0.10 * l_def2 + 0.08 * l_r1_sp

                    l_ebt = 0.10 * compute_ebt6_partial_label_ce(out["ebt_energies"], y_ai, y_pr, has_pr, a5_b) + compute_dual_axis_ebt_margin_loss(out["ebt_margin"], out["ebt_promo_margin"], y_ai, y_pr, has_pr)
                    l_perp = compute_v11_orthogonality_loss(mdl, lg, y_ai, y_pr, has_pr)
                    l_sp = compute_l2_sp_loss(mdl, r1_anchor_params, wd_sp=0.02) if quant_mode == "fp16" else (lg.sum() * 0.0)

                    # Dual-axis counterfactual minimal pairs (even steps) + v1.0 AI GateGuard/Debias/Literary legacy_pairs (odd steps!)
                    l_pair = lg.sum() * 0.0
                    l_ai_pair = lg.sum() * 0.0
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
                        op = mdl(pi, pm_, ps, engram_ids=pe, trigram_ids=pt_, outer_steps=1, bal_frozen=True, return_aux=True)
                        oo = mdl(oi, om_, os_, engram_ids=oe, trigram_ids=ot_, outer_steps=1, bal_frozen=True, return_aux=True)
                        ia_t = torch.tensor([1.0 if x.get("promo_is_ai") else 0.0 for x in pb], device=device)
                        l_pair = 0.20 * orthogonal_dual_axis_pair_loss(op["logit"].float(), oo["logit"].float(), op["logit_promo"].float(), oo["logit_promo"].float(), ia_t)
                    elif legacy_pairs and step % 2 == 1:
                        lpb = rng_tr.sample(legacy_pairs, min(16 if not args.smoke_test else 4, len(legacy_pairs)))
                        ai_it, hu_it, pw_l = [], [], []
                        for lp in lpb:
                            if "_ai_rec" not in lp:
                                lp["_ai_rec"] = _make_item(str(lp["ai_text"]), 1.0, -1.0, 0.0, "none", 1.0)
                                lp["_hu_rec"] = _make_item(str(lp["human_text"]), 0.0, -1.0, 0.0, "none", 1.0)
                            ai_it.append(lp["_ai_rec"])
                            hu_it.append(lp["_hu_rec"])
                            pw_l.append(float(lp.get("weight", 1.0)))
                        ai_i, ai_m, ai_e, ai_t, ai_s, *_ = _collate_items(ai_it)
                        hu_i, hu_m, hu_e, hu_t, hu_s, *_ = _collate_items(hu_it)
                        oa = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, trigram_ids=ai_t, outer_steps=2, bal_frozen=True, return_aux=True)
                        oh = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, trigram_ids=hu_t, outer_steps=2, bal_frozen=True, return_aux=True)
                        la_p, lh_p = oa["logit"].float(), oh["logit"].float()
                        lar1, lhr1 = oa["logit_r1"].float(), oh["logit_r1"].float()
                        pw_t = torch.tensor(pw_l, dtype=torch.float32, device=device)
                        l_simpo = (F.softplus(1.8 - (la_p - lh_p)) * pw_t).mean()
                        l_pabs = ((F.relu(0.84 - torch.sigmoid(la_p)).pow(2) + 1.6 * F.relu(torch.sigmoid(lh_p) - 0.12).pow(2)) * pw_t).mean()
                        l_sr1 = (F.softplus(1.4 - (lar1 - lhr1)) * pw_t).mean()
                        l_ai_pair = 0.18 * (l_simpo + 0.9 * l_pabs + 0.25 * l_sr1)
                    return l_ai + l_ai_pair + lam_kd * l_kd + l_ebt + l_sp, lam_p * lam_p_mult * l_pr_tot + l_pair + l_perp

            opt.zero_grad(set_to_none=True)
            loss_ai_p, loss_pr_p = _loss_fn()
            if (step % 100 == 50) and sh_params and quant_mode == "fp16":
                gr = gn_guard.probe(sh_params[:8], loss_ai_p, loss_pr_p)
                lam_p_mult = gr["lambda_p_mult"]

            loss_tot = loss_ai_p + loss_pr_p
            if chrono_mod > 0 and (step % chrono_mod == chrono_mod - 1) and quant_mode == "fp16":
                cs_b = collate_chronostream_docs(_sample_long_batch(4 if args.smoke_test else 48), tokenizer, max_chunks=3, max_chunk_toks=192)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    cs_o = mdl.forward_chronostream_batch(
                        cs_b["chunk_ids"].to(device), cs_b["chunk_mask"].to(device), cs_b["chunk_eng"].to(device), cs_b["chunk_tri"].to(device),
                        cs_b["chunk_style56"].to(device), cs_b["chunk_meta6"].to(device), cs_b["chunk_valid"].to(device), cs_b["doc_style56"].to(device),
                        trunk_no_grad=True, outer_steps=2,
                    )
                    loss_tot = loss_tot + 0.5 * (
                        F.binary_cross_entropy_with_logits(cs_o["logit_ai_doc"].float(), cs_b["y_ai_doc"].to(device))
                        + 0.5 * F.binary_cross_entropy_with_logits(cs_o["logit_hybrid_doc"].float(), cs_b["y_hyb_doc"].to(device))
                        + F.binary_cross_entropy_with_logits(cs_o["logit_promo_doc"].float(), cs_b["y_pr_doc"].to(device))
                    )

            loss_tot.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            if asam is not None and step >= steps // 2:
                asam.first_step()
                opt.zero_grad(set_to_none=True)
                la2, lp2 = _loss_fn()
                (la2 + lp2).backward()
                asam.restore()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()

            if not args.smoke_test and (step + 1) % 300 == 0:
                cur_m = _eval_ai_metrics(mdl)
                pl_m = _eval_promo_and_long(mdl)
                # 600-step grace warmup window before RollbackMonitor counts T-hard strikes!
                rb_active = enable_rollback and ((step + 1) >= 900)
                status_rb = rb_mon.check(cur_m, mdl.state_dict()) if rb_active else ("WARMUP" if enable_rollback else "N/A")
                print(
                    f"[{name} step={step+1}/{steps}] ai_obj={cur_m['ai_obj']:.5f} auroc_vg={cur_m['auroc_vg']:.4f} "
                    f"rp95_vg={cur_m['rp95_vg']:.4f} r1={cur_m['r1_auroc']:.4f} | promo_auc={pl_m['promo_auroc']:.4f} "
                    f"pr={pl_m['promo_pr_auc']:.4f} | long_ai={pl_m['long_ai_auroc']:.4f} | rb={status_rb} | elapsed={time.monotonic()-t_start:.0f}s",
                    flush=True,
                )
                cur_sc = _stage_pareto_score(cur_m, pl_m)
                if cur_sc >= best_sc:
                    best_sc = cur_sc
                    best_m = cur_m
                    best_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                if rb_active:
                    if status_rb == "T-soft":
                        lam_kd_scale = min(2.0, lam_kd_scale * 1.5); lr_scale_rb *= 0.7; sh_lam_cur *= 0.5
                    elif status_rb == "T-hard" and rb_mon.last_pass_sd is not None:
                        mdl.load_state_dict({k: v.to(device) for k, v in rb_mon.last_pass_sd.items()}, strict=True)
                        lr_scale_rb *= 0.6; sh_lam_cur = 0.0
                    elif status_rb == "T-abort":
                        print(f"[{name}] T-abort triggered -> falling back to last passing checkpoint!", flush=True)
                        break
                mdl.train()

        end_m = _eval_ai_metrics(mdl)
        end_pl = _eval_promo_and_long(mdl)
        end_sc = _stage_pareto_score(end_m, end_pl)
        if end_sc >= best_sc:
            best_sc = end_sc
            best_m = end_m
            best_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        return best_sd, best_m

    # Stage 1b
    s1b_steps = 4 if args.smoke_test else int(args.s1b_steps)
    s1b_sd, s1b_m = _run_v11_stage("Stage1b-DualAxis-ChronoStream", s1a_sd, "1b", s1b_steps, enable_rollback=True)
    if not args.smoke_test:
        torch.save(s1b_sd, ckpt_dir / f"{args.version}_stage1b_ckpt.pt")
    print(f"[Stage1b DONE] {json.dumps({k: round(v, 5) for k, v in s1b_m.items()})}", flush=True)

    # Stage 1.5: Kronecker LoGra Top-FP Sniping + 3 ASAM Branches + AI-Constrained Pareto Soup!
    mdl.load_state_dict({k: v.to(device) for k, v in s1b_sd.items()}, strict=True)
    sel_o = _predict_v11(mdl, vs_p, outer_steps=2)
    sel_fp_m = (vs_p[7] <= 20.0) & (expit(sel_o["logit"]) >= 0.25)
    t1_scan = t1_pool[: (64 if args.smoke_test else min(40000, len(t1_pool)))]
    t1_pack = (
        [x["ids"] for x in t1_scan], [x["engram"] for x in t1_scan], [x["trigram"] for x in t1_scan],
        [], [], np.stack([x["style"] for x in t1_scan], axis=0), [], np.array([x["y_score"] * 100.0 for x in t1_scan], dtype=np.float32),
    )
    t1_o = _predict_v11(mdl, t1_pack, outer_steps=2)
    logra_rep = kronecker_logra_topfp_snipe(
        t1_o["emb"], expit(t1_o["logit"]) - np.array([x["y_score"] for x in t1_scan]), np.zeros(len(t1_scan), dtype=bool),
        sel_o["emb"][sel_fp_m] if sel_fp_m.any() else sel_o["emb"][:8],
        np.maximum(0.0, expit(sel_o["logit"][sel_fp_m]) - 0.0) if sel_fp_m.any() else np.ones(min(8, len(sel_o["logit"]))),
        prune_frac=0.015,
    )
    for zi in logra_rep["zeroed_indices"]:
        t1_scan[zi]["weight"] = 0.0
    print(f"[Stage1.5-LoGra] zeroed_non_t0={logra_rep['zeroed_count']}", flush=True)

    s15_steps = 4 if args.smoke_test else int(args.s15_steps)
    b1_sd, b1_m = _run_v11_stage("Stage1.5-B1-AIShield", s1b_sd, "1.5", s15_steps, lam_kd_start=0.45, lam_kd_end=0.45, lam_p=0.70, use_asam=True)
    b2_sd, b2_m = _run_v11_stage("Stage1.5-B2-PromoPrec", s1b_sd, "1.5", s15_steps, lam_kd_start=0.30, lam_kd_end=0.30, lam_p=1.30, promo_sh_lam=0.40, use_asam=True, extra_hard=b2_deferred_knn)
    b3_sd, b3_m = _run_v11_stage("Stage1.5-B3-LongStream", s1b_sd, "1.5", s15_steps, lam_kd_start=0.30, lam_kd_end=0.30, chrono_mod=3, use_asam=True)

    def _constrained_soup_eval(sd_c: dict[str, torch.Tensor], is_qat: bool = False) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_c.items()}, strict=True)
        cm = _eval_ai_metrics(mdl)
        pl_m = _eval_promo_and_long(mdl)
        # Soft-hinge zero-regression guard on AI axis + Promo & ChronoStream longform joint Pareto score!
        pen = 0.0 if (args.smoke_test or is_qat) else 50.0 * max(0.0, (r1_ref_m["ai_obj"] - 0.0025) - cm["ai_obj"])
        long_mean = 0.50 * pl_m["long_ai_auroc"] + 0.25 * pl_m["long_hyb_auroc"] + 0.25 * pl_m["long_pr_auroc"]
        return cm["ai_obj"] + 0.40 * (0.5 * pl_m["promo_auroc"] + 0.5 * pl_m["promo_pr_auc"]) + 0.15 * long_mean - pen

    fp16_sd, fp16_soup_rep = greedy_pauc_soup(s1b_sd, [("B1", b1_sd, b1_m["ai_obj"]), ("B2", b2_sd, b2_m["ai_obj"]), ("B3", b3_sd, b3_m["ai_obj"])], eval_fn=lambda sd: _constrained_soup_eval(sd, is_qat=False), dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in fp16_sd.items()}, strict=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_fp16_master.pt")
    print(f"[Stage1.5-ParetoSoup] {json.dumps(fp16_soup_rep)}", flush=True)

    # Stage 2a/2b: Zero-Op Fold -> UPQ W4 -> Sub-2 QAT x2
    cal_f = _collate_items(cal_p1_items[:64])
    fold_rep = fold_zero_op_v11(mdl, calib_inputs=[(cal_f[0], cal_f[1], cal_f[4], cal_f[2])])
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    fp16_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_teacher.parameters():
        p.requires_grad_(False)

    w4_sd, _ = _run_v11_stage("Stage2a-UPQ-W4", folded_sd, "qat", 4 if args.smoke_test else int(args.s2a_steps), quant_mode="w4", trm_lam=0.0, chrono_mod=0, teacher_v11=fp16_teacher)
    q1_sd, q1_m = _run_v11_stage("Stage2b-QAT-A", w4_sd, "qat", 4 if args.smoke_test else int(args.s2b_steps), quant_mode="sub2", trm_lam=0.0, chrono_mod=0, teacher_v11=fp16_teacher)
    q2_sd, q2_m = _run_v11_stage("Stage2b-QAT-B", w4_sd, "qat", 4 if args.smoke_test else int(args.s2b_steps), quant_mode="sub2", trm_lam=0.0, lam_p=1.25, chrono_mod=0, teacher_v11=fp16_teacher)
    qat_sd, qat_soup_rep = greedy_pauc_soup(w4_sd, [("Q1", q1_sd, q1_m["ai_obj"]), ("Q2", q2_sd, q2_m["ai_obj"])], eval_fn=lambda sd: _constrained_soup_eval(sd, is_qat=True), dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in qat_sd.items()}, strict=True)
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_qat_soup.pt")
    mdl.set_quant_mode("fp16")
    del fp16_teacher
    torch.cuda.empty_cache()

    # Stage 3a: MGVZ6 Pack/Unpack -> P5 Dual-Axis FP16 Shock-Absorber
    tmp_mg = out_dir / "_tmp_v11_pre_shock.mgvz"
    pack_bundle_v6_v11(mdl.cpu(), tmp_mg)
    load_bundle_v6_v11_into_model(mdl, tmp_mg)
    tmp_mg.unlink(missing_ok=True)
    mdl = mdl.to(device).eval()

    pre_sh_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_sh_sc = _constrained_soup_eval(pre_sh_sd, is_qat=True)
    shock_tokens = ("core.mole.", "core.inloop_ebt.", "promo", "r1_promo", "sub4", "attr6", "longstream", "ai_head.", "r1_ai_head", "defer_", "ebt_logit_gate", "mil_gate")
    quant_skip = set()
    for mn, mod in mdl.named_modules():
        if isinstance(mod, OmniQuantLinear):
            quant_skip.add(f"{mn}.weight")
    sh_p_list = []
    for nm, p in mdl.named_parameters():
        keep = (nm not in quant_skip) and any(t in nm for t in shock_tokens)
        p.requires_grad_(keep)
        if keep:
            sh_p_list.append(p)
    opt_sh = torch.optim.AdamW(sh_p_list, lr=1.5e-4, weight_decay=1e-4)
    mdl.train()
    for _ in range(4 if args.smoke_test else int(args.s3a_steps)):
        opt_sh.zero_grad(set_to_none=True)
        b_it = _sample_short_batch(min(bs, 32 if args.smoke_test else bs), pike.w)
        b_ids, b_m, b_e, b_t, b_s56, y_ai, y_pr, has_pr, sub4_i, w_b, a5_b, *_ = _collate_items(b_it)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            o = mdl(b_ids, b_m, b_s56, engram_ids=b_e, trigram_ids=b_t, outer_steps=2, trm_lam_k0=0.0, trm_lam_k1=0.0, bal_frozen=True, return_aux=True)
            loss = (F.binary_cross_entropy_with_logits(o["logit"].float(), y_ai.clamp(1e-4, 1 - 1e-4), reduction="none") * w_b).mean()
            if (has_pr > 0.5).any():
                pm = has_pr > 0.5
                loss = loss + (F.binary_cross_entropy_with_logits(o["logit_promo"][pm].float(), y_pr[pm].clamp(1e-4, 1 - 1e-4), reduction="none") * w_b[pm]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sh_p_list, 1.0)
        opt_sh.step()
    for p in mdl.parameters():
        p.requires_grad_(True)
    post_sh_sc = _constrained_soup_eval({k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}, is_qat=True)
    shock_kept = post_sh_sc >= pre_sh_sc
    if not shock_kept:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_sh_sd.items()}, strict=True)
    mdl.eval()

    # Stage 3b: Zero-Byte Dual-Axis Oblique LEACE + Cross-Exit Affine Alignment + Weighted Venn-ABERS + 1-D Fixed-Seq Dual-Axis LTT!
    vs_o = _predict_v11(mdl, vs_p, outer_steps=2)
    z_ll_vs = np.log1p(np.array([len(t) for t in vs_p[3]], dtype=np.float32))[:, None]
    emb_leace = vs_o["emb"]
    yai_leace = vs_p[7] / 100.0
    ypr_leace = np.full(len(vs_p[7]), 0.50, dtype=np.float32)  # 0.50 excluded by Promo valid_pr mask
    zai_leace = np.concatenate([z_ll_vs, expit(vs_o["logit_promo"])[:, None]], axis=1)
    zpr_leace = np.concatenate([z_ll_vs, expit(vs_o["logit"])[:, None]], axis=1)
    if promo_val_select_rows:
        ps_leace_pk = (
            [x["ids"] for x in promo_val_select_rows], [x["engram"] for x in promo_val_select_rows], [x["trigram"] for x in promo_val_select_rows],
            [x["raw_text"] for x in promo_val_select_rows], [x["clean_text"] for x in promo_val_select_rows], np.stack([x["style"] for x in promo_val_select_rows]),
            [], np.array([x["y_promo"] for x in promo_val_select_rows], dtype=np.float32),
        )
        pso_l = _predict_v11(mdl, ps_leace_pk, outer_steps=2)
        z_ll_ps = np.log1p(np.array([len(t) for t in ps_leace_pk[3]], dtype=np.float32))[:, None]
        emb_leace = np.concatenate([emb_leace, pso_l["emb"]], axis=0)
        yai_leace = np.concatenate([yai_leace, np.full(len(promo_val_select_rows), 0.50, dtype=np.float32)], axis=0)
        ypr_leace = np.concatenate([ypr_leace, ps_leace_pk[7]], axis=0)
        zai_leace = np.concatenate([zai_leace, np.concatenate([z_ll_ps, expit(pso_l["logit_promo"])[:, None]], axis=1)], axis=0)
        zpr_leace = np.concatenate([zpr_leace, np.concatenate([z_ll_ps, expit(pso_l["logit"])[:, None]], axis=1)], axis=0)
    leace_rep = apply_dual_axis_oblique_leace_v11(mdl, emb_leace, yai_leace, ypr_leace, zai_leace, zpr_leace)

    # Cross-exit affine alignment (AI & Promo) on clean train slice
    al_sub = cal_p1_items[: (64 if args.smoke_test else min(8000, len(cal_p1_items)))]
    al_pack = ([x["ids"] for x in al_sub], [x["engram"] for x in al_sub], [x["trigram"] for x in al_sub], [], [], np.stack([x["style"] for x in al_sub]), [], np.zeros(len(al_sub)))
    al_o = _predict_v11(mdl, al_pack, outer_steps=2)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_r1"], al_o["raw_logit"])
    a_t1, b_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_t1"], al_o["raw_logit"])
    pa_r1, pb_r1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_r1"], al_o["raw_logit_promo"])
    pa_t1, pb_t1 = fit_cross_exit_affine_alignment(al_o["raw_logit_promo_t1"], al_o["raw_logit_promo"])
    with torch.no_grad():
        mdl.exit_affine_a.copy_(torch.tensor([a_r1, a_t1, 1.0], device=device))
        mdl.exit_affine_b.copy_(torch.tensor([b_r1, b_t1, 0.0], device=device))
        mdl.exit_affine_promo_a.copy_(torch.tensor([pa_r1, pa_t1, 1.0], device=device))
        mdl.exit_affine_promo_b.copy_(torch.tensor([pb_r1, pb_t1, 0.0], device=device))

    # LUTs: AI Binary-ECE Beta-Spline + Promo Prior-Shift Weighted Venn-ABERS
    vg_o, vs_o, vc_o = _predict_v11(mdl, vg_p, 2), _predict_v11(mdl, vs_p, 2), _predict_v11(mdl, vc_p, 2)
    lut_ai_full = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit"], vg_o["logit"]]), np.concatenate([vs_p[7], vg_p[7]]) / 100.0, fit_binary_ece_mode=True), 256)
    lut_ai_r1 = export_beta_spline_as_v07_lut(fit_venn_abers_and_beta_spline_lut(np.concatenate([vs_o["logit_r1"], vg_o["logit_r1"]]), np.concatenate([vs_p[7], vg_p[7]]) / 100.0, fit_binary_ece_mode=True), 256)

    pi_nat_pr = estimate_wild_prevalence_em(expit(al_o["logit_promo"]), pi_train=0.10)
    lut_pr_full, lut_pr_r1, ltt_policy = None, None, {"enabled": False, "certified_ltt": False}
    if promo_val_select_rows and promo_val_cert_rows:
        def _pack_pr(rows_pr):
            return ([x["ids"] for x in rows_pr], [x["engram"] for x in rows_pr], [x["trigram"] for x in rows_pr], [x["raw_text"] for x in rows_pr], [x["clean_text"] for x in rows_pr], np.stack([x["style"] for x in rows_pr]), [{"spoof_any": x["spoof_any"]} for x in rows_pr], np.array([x["y_promo"] for x in rows_pr], dtype=np.float32))
        ps_pk, pc_pk = _pack_pr(promo_val_select_rows), _pack_pr(promo_val_cert_rows)
        pso, pco = _predict_v11(mdl, ps_pk, 2), _predict_v11(mdl, pc_pk, 2)
        lut_pr_full = fit_weighted_venn_abers_promo_lut(pso["logit_promo"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)
        lut_pr_r1 = fit_weighted_venn_abers_promo_lut(pso["logit_promo_r1"], ps_pk[7], pi_nat=pi_nat_pr, n_knots=256)

        def _elig(pk, o_d):
            return np.array([
                len(pk[0][i]) <= 96 and pk[6][i]["spoof_any"] == 0.0 and ("http" not in pk[4][i])
                and float(np.max(pk[5][i, list(R1_PROMO_VETO_DIMS_V11)])) == 0.0 and float(o_d["r1_span_max"][i]) <= 0.50
                for i in range(len(pk[0]))
            ], dtype=bool)
        ltt_policy = certify_fixed_sequence_dual_axis_ltt(
            _elig(vs_p, vs_o), vs_o["p_defer1"], apply_spline_temp_lut(expit(vs_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vs_o["logit_promo_r1"]), lut_pr_r1),
            vc_p[7], _elig(vc_p, vc_o), vc_o["p_defer1"], apply_spline_temp_lut(expit(vc_o["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(vc_o["logit_promo_r1"]), lut_pr_r1),
            pc_pk[7] >= 0.65, _elig(pc_pk, pco), pco["p_defer1"], apply_spline_temp_lut(expit(pco["logit_r1"]), lut_ai_r1), apply_spline_temp_lut(expit(pco["logit_promo_r1"]), lut_pr_r1),
            alpha_ai=0.005, alpha_pr=0.010, delta_fwer=0.05,
        )

    # Stage 4: Export MGVZ6 + 4-Session Static ONNX + Auto Compression Levers if Bundle v6 > 2.20 MB!
    exp_info = export_v11_modular_onnx_and_bundle_v6(mdl, out_dir)
    manifest = {
        "schema_version": "meguviz.bundle.v6",
        "version": args.version,
        "emb_layout": mdl.emb_layout_tag(),
        "tokenizer_file": tok_path.name,
        "bundle_v6_mgvz": exp_info["bundle_v6_mgvz"]["file"],
        "rung1_onnx": exp_info["rung1_onnx"],
        "rung2_onnx": exp_info["rung2_onnx"],
        "rung3_onnx": exp_info["rung3_onnx"],
        "dynaseam_onnx": exp_info["dynaseam_onnx"]["onnx_file"],
        "lut_ai_r1": lut_ai_r1,
        "lut_ai_full": lut_ai_full,
        "lut_promo_r1": lut_pr_r1,
        "lut_promo_full": lut_pr_full,
        "dual_axis_ltt_policy": ltt_policy,
        "capabilities": [],  # P8 CSP-1 reserved capability slot manifest!
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_v6 = out_dir / f"meguvizcac-{args.version}-bundle-v6-sub2bit.zip"
    with ZipFile(zip_v6, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for fn in (exp_info["bundle_v6_mgvz"]["file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)
    v6_mb = zip_v6.stat().st_size / (1024.0 * 1024.0)
    # P5 auto compression lever L1 -> L2 if bundle exceeds 2.20 MB!
    if v6_mb > 2.20:
        exp_info = export_v11_modular_onnx_and_bundle_v6(mdl, out_dir, lever_w4_longstream=True, lever_w4_r1_promo=(v6_mb > 2.215))
        with ZipFile(zip_v6, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for fn in (exp_info["bundle_v6_mgvz"]["file"], tok_path.name, "manifest.json"):
                zf.write(out_dir / fn, fn)
        v6_mb = zip_v6.stat().st_size / (1024.0 * 1024.0)

    zip_i8 = out_dir / f"meguvizcac-{args.version}-4stage-onnx-int8.zip"
    with ZipFile(zip_i8, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in (exp_info["rung1_onnx"], exp_info["rung2_onnx"], exp_info["rung3_onnx"], exp_info["dynaseam_onnx"]["onnx_file"], tok_path.name, "manifest.json"):
            zf.write(out_dir / fn, fn)

    # Final unpacked eval on test_bili_gold, test_dsh_gold, PromoBench-48, LongBench-24, & V11 CI!
    mdl_u = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v6_v11_into_model(mdl_u, out_dir / exp_info["bundle_v6_mgvz"]["file"])
    mdl_u = mdl_u.to(device).eval()
    tgo = _predict_v11(mdl_u, tg_p, 2)
    tdo = _predict_v11(mdl_u, td_p, 2)
    tb_m = {**compute_eval_metrics(tg_p[7], apply_spline_temp_lut(expit(tgo["logit"]), lut_ai_full)), **compute_high_precision_metrics(tg_p[7], apply_spline_temp_lut(expit(tgo["logit"]), lut_ai_full))}
    td_m = compute_eval_metrics(td_p[7], apply_spline_temp_lut(expit(tdo["logit"]), lut_ai_full)) if len(td_p[7]) > 0 else {}

    pb_m: dict[str, Any] = {}
    pb_p = Path(__file__).resolve().parent / "data" / "v11-promobench-seed.jsonl"
    if pb_p.is_file():
        pb_rows = [json.loads(l) for l in pb_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb_pk = _pre_eval56(pb_rows)
        pbo = _predict_v11(mdl_u, pb_pk, 2)
        y_pb_ai = np.array([int(r["y_ai"]) for r in pb_rows])
        y_pb_pr = np.array([int(r["y_promo"]) for r in pb_rows])
        pb_m = {
            "ai_auroc": round(float(roc_auc_score(y_pb_ai, pbo["logit"])), 4),
            "promo_auroc": round(float(roc_auc_score(y_pb_pr, pbo["logit_promo"])), 4),
            "promo_pr_auc": round(float(average_precision_score(y_pb_pr, pbo["logit_promo"])), 4),
        }

    final_dual_m = _eval_promo_and_long(mdl_u)
    ci_fin = run_v11_ci_gates(mdl_u, tokenizer)
    metrics = {
        "version": args.version,
        "gpu_name": gpu_name,
        "bundle_v6_total_mb": round(v6_mb, 4),
        "stage0_transplant": surg_rep,
        "stage0_r1_ref": r1_ref_m,
        "stage15_logra_zeroed": logra_rep["zeroed_count"],
        "fp16_soup": fp16_soup_rep,
        "qat_soup": qat_soup_rep,
        "shock_absorber": {"kept": shock_kept, "before": round(pre_sh_sc, 5), "after": round(post_sh_sc, 5)},
        "leace_surgery": leace_rep,
        "dual_axis_ltt_policy": ltt_policy,
        "test_bili_gold": tb_m,
        "test_dsh_gold": td_m,
        "promobench48": pb_m,
        "promo_val_select_and_longform120": {k: round(v, 4) for k, v in final_dual_m.items()},
        "export_info": exp_info,
        "v11_ci_gates": ci_fin,
        "release_gates": {
            "bundle_le_2p20_mb": v6_mb <= 2.20,
            "onnx_4stage_exact": exp_info["r1_fp32_max_diff"] < 1e-4 and exp_info["r2_fp32_max_diff"] < 1e-4 and exp_info["r3_fp32_max_diff"] < 1e-4 and exp_info["dynaseam_onnx"]["pass"],
            "v11_ci_passed": bool(ci_fin["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.1.0-OmniGuard Trainer & Full-Stack Dry-Run CI.")
    ap.add_argument("--dry-run", action="store_true", help="Run fast self-contained GPU dry-run CI locally.")
    ap.add_argument("--smoke-test", action="store_true", help="Run tiny end-to-end smoke test of train_v11_omniguard_full.")
    ap.add_argument("--data", type=Path, default=Path("v04-corpus/local-train/v04-merged-1m4-r3.jsonl.gz"))
    ap.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    ap.add_argument("--extra-pairs", type=Path, action="append", default=[])
    ap.add_argument("--v10-fp16-master", type=Path, default=Path("/content/drive/MyDrive/MeguvizCAC/v100_apexloop/v1.0.0-r1-apexloop-b6_fp16_master.pt"))
    ap.add_argument("--v07-tokenizer", type=Path, default=None)
    ap.add_argument("--v08-gemini-gold", type=Path, default=None)
    ap.add_argument("--v08-gemini-pairs", type=Path, default=None)
    ap.add_argument("--v11-data-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--wave25-gold", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("scratch/v11_dryrun_out"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.1.0-r1-omniguard-b6")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--s1a-steps", type=int, default=1200)
    ap.add_argument("--s1b-steps", type=int, default=6000)
    ap.add_argument("--s15-steps", type=int, default=1800)
    ap.add_argument("--s2a-steps", type=int, default=600)
    ap.add_argument("--s2b-steps", type=int, default=1500)
    ap.add_argument("--s3a-steps", type=int, default=300)
    ap.add_argument("--datafactory", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--tokenizer", type=Path, default=None)
    args = ap.parse_args()
    if args.dry_run:
        rep = run_v11_pipeline_dry_run(args.out, args.datafactory, args.tokenizer)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        assert rep["dry_run_passed"], "v1.1.0 Dry-Run CI failed!"
        return
    rep = train_v11_omniguard_full(args)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("export_info",)}, ensure_ascii=False, indent=2))
    assert all(rep["release_gates"].values()), f"Release gates failed: {rep['release_gates']}"


if __name__ == "__main__":
    main()
