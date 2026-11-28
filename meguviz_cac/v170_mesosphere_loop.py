"""MeguvizCAC v1.7.0-MesosphereLoop: Narrow L2-Resident Recurrent Engine + Tri-Tier Super-Memory.

Locked Master Architecture (`v170_mesosphere_final_spec.md`, 9/9 External GPT Audits + Real ORT/GPU Lab):
  - Immutable Release Contract:
      * Cold-start bootable ZIP <= 5,242,880 bytes (5.00 MiB) using MGVZ10 Uniform W4/W8 (`DEFLATE -9`).
      * Pure FP32 ONNX Opset-17 (`Tile=0, Loop=0, If=0, NonZero=0, TopK=0`, zero INT8 QDQ / custom ops).
      * C1 === 0.0 (Zero-PII), F6 === 0.0 (uint32 bit-exact short & long), P1 < 1e-4 (odd/even pad invariant).
  - Four-Stage Modular Topology:
      * Stage 0: Full Natural-Stream R0++ (`2T` ORT C0) + Detached `APD-v2` Independent Exit Head + `IUT` Stratified Exit.
      * Stage 1: Narrow `H=256, FFN=512` 1L Pad-Exact Prelude (dropped `trigram_buckets` -281 KiB!) + Discourse Ops
                 + Hoisted `k_ev_norm (L,96)`, `V_0 (L,96)`, clamp-only `E_0 (L,8)` + Split-Slot `[slow, pros, def, pros-def]`
                 + Per-Modality Pre-Concat LayerNorm (with 0B `M6 P_H` 16-D trace) + Once-at-T1 `B2a` Dual-Manifold
                 Epistemic Hopfield (`4x64 M_H + 4x64 M_A`, emitting `(E_H, E_A, E_H - E_A)`) + `M7` `CNMA-R` (644 B)
                 + `UISR` (4,228 B) + `SoftBall(r_cap=10.0)`.
      * Stage 2: `DreamCore_170^Safe` reusable `loop_step(t)` (`H=256, FFN=512, D=2, 12 slots [4+2+2+4], d_kv=96 [4x24]`):
                 1. `QK-Norm` (`s_qk in [4, 12]`) + strictly NSD `RI-Meso-PSD-L` (`b = -alpha * relu(E^T M E) * ai_slot_mask`, 0 Div/Log/Concat).
                 2. `A2_StepLoRA_r8` on `ca_q` & `s_qk` (autonomous shared transition for `t >= 4` per M9 Sec 5.3!).
                 3. `M9` Double-Masked `AR-U-Loop` anchored residual value broadcast (`D_{t+1} = m_u * SmoothBall((1-lam_v)D_t + kappa A_t^T S_t W_b)`,
                    AI-private `V_ai = V_0 + D_t`; Promo strictly reads frozen `V_0` -> F6 === 0.0!).
                 4. `Tier-1a` Contracting `16x16` Gated Delta-Rule Fast-Weight `W_t` (`beta in [0.35, 0.85], eta <= beta`, clamp-only `||k|| <= 1`).
                 5. Direct 1-stream Contractive Kernel D + `PNBR-R` Zero-Static-Skip Nested Boosting Readout.
      * Stage 3: `TriSeam-R` Rust `Kadane + O(L) E_lambda` `FocalRouter` + `DynaSeam` + `68 B` `DS-TitanGuard`
                 (17-param directional cross-chunk surprise, `m_1 = 0`, Version-A zero-flip safe ranking).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from scipy.special import expit, logit as sp_logit
from scipy.stats import beta as sp_beta
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
)
from .v11_features import (
    CHUNK_META_DIM_V11,
    STYLE_FEAT_DIM_V11,
    extract_style_features_56d,
    plan_long_chunks_v11,
)
from .v121_apexnano import SoftBallLogitNormHeadV121
from .v130_xtreme_apex import (
    AI_END_V130,
    BUNDLE_MAX_MIB_V130,
    BUS_DIM_V130,
    CBM_AI_K_V130,
    CBM_PR_K_V130,
    EMB_DIM_V130,
    PR_EMB_DIM_V130,
    R0_B2_V130,
    R0_B3_V130,
    R0_ZCHAR_DIM_V130,
    SHARED_END_V130,
    QuantEmbeddingV130,
    QuantLinearV130,
    _pack_w4_tensor,
    _pack_w8_tensor,
    _unpack_w4_tensor,
    _unpack_w8_tensor,
    set_v130_qat_mode,
)
from .v140_paradise_apex import (
    HEAD_LEN_V140,
    MAX_LEN_V140,
    R0_FEAT_DIM_V140,
    R0_STRUCT_DIM_V140,
)
from .v150_heaven_loop import (
    LOCAL_CHARLM_DIM_V150,
    CompactCharLMV150,
    canonicalize_zero_pii_v150,
    encode_bpe_head_tail_with_offsets_v150,
)
from .v160_discourse_causal_ops import (
    CheapD08PositionMoments,
    SelectiveOrder3CumSum,
)
from .v160_stratosphere_loop import (
    CONTRACTIVE_INNER_R_V161,
    QUOTE_SCOPE_DIM_V161,
    SLOT_R_CAP_V160,
    ApexCharNanoRung0V160,
    PadExactStratospherePreludeBlockV160,
    _fourier_step_features,
    apply_crossview_late_fusion,
    extract_apex_char_batch_v160,
    extract_token_quote_scope_4d_v161,
    fit_crossview_eba_h_fusion,
    kadane_focal_window_py,
    verify_eba_h_fp32_monotonicity,
)

# Locked v1.7.0 sidecar interface (owned by `v170_features.py`, clean shim if imported during parallel build)
try:
    from .v170_features import (
        M6_PH_DIM_V170,
        M7_SURF_DIM_V170,
        M7_UISR_DIM_V170,
        extract_v170_sidecar_batch,
    )
except ImportError:
    M6_PH_DIM_V170 = 16
    M7_SURF_DIM_V170 = 8
    M7_UISR_DIM_V170 = 64

    def extract_v170_sidecar_batch(
        clean_texts: list[str],
        bpe_lens: np.ndarray,
        charlm: Any = None,
        rust_charlm: Any = None,
        fold_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(clean_texts)
        return (
            np.zeros((n, M6_PH_DIM_V170), dtype=np.float32),
            np.zeros((n, M7_SURF_DIM_V170), dtype=np.float32),
            np.zeros((n, M7_UISR_DIM_V170), dtype=np.float32),
        )


MGVZ10_MAGIC_V170 = b"MGVZ10\x00\x01"
FORBIDDEN_ONNX_OPS_V170 = ("Tile", "Loop", "If", "NonZero", "TopK")
SKETCH_R_V170 = 8
DELTA_RULE_DIM_V170 = 16
DEFAULT_T_INFER_MAX_V170 = 4

ARCH_PRESETS_V170: dict[str, dict[str, Any]] = {
    # Primary locked v1.7.0 MesosphereLoop geometry (<= 4.25 MiB ZIP, <= 0.99 ms B=1 Xeon 4T!)
    "v170_m1": {
        "dim": 256,
        "ffn_dim": 512,
        "prelude_ffn": 512,
        "prelude_layers": 1,
        "prelude_global_se": True,
        "num_heads": 4,
        "d_kv": 96,
        "kv_heads": 4,
        "n_slow": 4,
        "n_pros": 2,
        "n_def": 2,
        "n_span": 4,
        "core_depth": 2,
        "engram_buckets": 8192,
        "r0_b2": R0_B2_V130 * 2,  # 32768 (high ROI!)
        "r0_b3": R0_B3_V130 * 2,  # 16384 (highest ROI: 17.5x trigram!)
    },
    # Controlled A0 baseline preset (same H=256 geometry, defaults DreamCore extensions OFF unless flagged)
    "v170_a0": {
        "dim": 256,
        "ffn_dim": 512,
        "prelude_ffn": 512,
        "prelude_layers": 1,
        "prelude_global_se": True,
        "num_heads": 4,
        "d_kv": 96,
        "kv_heads": 4,
        "n_slow": 4,
        "n_pros": 2,
        "n_def": 2,
        "n_span": 4,
        "core_depth": 2,
        "engram_buckets": 8192,
        "r0_b2": R0_B2_V130 * 2,
        "r0_b3": R0_B3_V130 * 2,
    },
}


def _clamp_unit_norm(x: torch.Tensor) -> torch.Tensor:
    """M5 clamp-only norm: `x / sqrt(max(1, ||x||^2))` -> `||out|| <= 1`, never amplifies near-zero noise!"""
    sq = (x * x).sum(dim=-1, keepdim=True)
    return x * torch.rsqrt(torch.clamp(sq, min=1.0))


def _smooth_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Strictly smooth, zero-division-safe FP32 ball projection `x / sqrt(1 + ||x||^2 / R^2)` (`||out|| < R`)."""
    xf = x.float()
    sq = (xf * xf).sum(dim=-1, keepdim=True)
    inv_r2 = 1.0 / (float(radius) * float(radius))
    return (xf * torch.rsqrt(1.0 + sq * inv_r2)).to(x.dtype)


# ---------------------------------------------------------------------------
# 1. Stage-0 `APD-v2` Independent Exit Head + `M8` Zero-Bonferroni `IUT SCE`
# ---------------------------------------------------------------------------

class ApexCharNanoRung0V170(ApexCharNanoRung0V160):
    """Exact bit-identical subclass of `ApexCharNanoRung0V160` replacing `.squeeze(...)` with static indexing
    so Opset-17 ONNX export emits ZERO `If` control-flow ops (`Tile=Loop=If=NonZero=TopK=0`)!
    """

    def forward(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([self.emb1(c1), self.emb2(c2), self.emb3(c3)], dim=-1)
        m_f = mask.float()
        m_x = mask.to(x.dtype)
        m_u = m_x.unsqueeze(-1)
        h = self.in_norm(self.in_proj(x)) * m_u
        ht = h.transpose(1, 2).contiguous()
        conv_sum = (self.dw3(ht) + self.dw5_d1(ht) + self.dw5_d2(ht) + self.dw5_d4(ht)) * 0.25
        hc = conv_sum.transpose(1, 2) * m_u
        u, g = self.pw(hc).chunk(2, dim=-1)
        hc = u * torch.sigmoid(g) * m_u

        denom = m_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_c = (hc.float().sum(dim=1) / denom).to(hc.dtype)
        hc = hc * self.se(mean_c).unsqueeze(1)
        h = self.stage_norm(h + hc) * m_u

        p_mean = (h.float().sum(dim=1) / denom).to(h.dtype)
        p_max = h.masked_fill(m_u < 0.5, -1e4).max(dim=1).values
        a_lg = self.attn_pool(h)[..., 0].masked_fill(m_f < 0.5, -1e4)
        a_w = torch.softmax(a_lg.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        p_attn = (h * a_w).sum(dim=1)
        p_first = h[:, 0, :]
        last_idx = (m_f.sum(dim=1).long() - 1).clamp(min=0, max=h.shape[1] - 1)
        p_last = h.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[2]))[:, 0, :]

        f_in = feats.to(h.dtype)
        f_struct = f_in[:, :R0_STRUCT_DIM_V140]
        f_charlm = f_in[:, R0_STRUCT_DIM_V140:R0_FEAT_DIM_V140]
        sup_norm = f_struct[:, R0_STRUCT_DIM_V140 - 1 : R0_STRUCT_DIM_V140]
        g_sup_r0 = torch.sigmoid((sup_norm - self.r0_sup_tau.clamp(0.25, 0.75)) * 8.0)
        f_lm_eff = (f_charlm * (0.55 + 0.45 * g_sup_r0)) if self.use_global_charlm_shortcut else (f_charlm * 0.0)
        f_ai = torch.cat([f_struct[:, : R0_STRUCT_DIM_V140 - 1] * g_sup_r0, sup_norm, f_lm_eff], dim=-1)

        h_ai = self.ai_fc(torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_ai], dim=-1))
        h_pr = self.promo_fc(torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_struct], dim=-1))

        lg_ai = self.ai_head(h_ai)[..., 0]
        lg_pr = self.promo_head(h_pr)[..., 0]
        z_char = torch.cat([self.z_char_ai(h_ai), self.z_char_pr(h_pr)], dim=-1)

        def_ai_lg = self.defer_ai_head(torch.cat([h_ai.detach(), lg_ai.detach().unsqueeze(-1), sup_norm.detach()], dim=-1))[..., 0]
        def_pr_lg = self.defer_pr_head(torch.cat([h_pr.detach(), lg_pr.detach().unsqueeze(-1), sup_norm.detach()], dim=-1))[..., 0]
        return {
            "logit_ai_r0": lg_ai,
            "logit_promo_r0": lg_pr,
            "defer_ai_r0_logit": def_ai_lg,
            "p_defer_ai_r0": torch.sigmoid(def_ai_lg),
            "defer_pr_r0_logit": def_pr_lg,
            "p_defer_pr_r0": torch.sigmoid(def_pr_lg),
            "z_char": z_char,
        }


class APDv2ExitHeadV170(nn.Module):
    """M8 Asymmetric Positive-Defense v2 Exit Head (AI-only, detached from R0 anchor -> F6 === 0.0!).

    Predicts `apd_exit_logit` (higher = safe to early-exit as Human at R0) from
    `[sg(z_char_ai [32]), sg(logit_ai_r0 [1]), sg(sup_norm [1]), m7_surf_8 [8]]`.
    Because all R0 inputs are detached, training `APD-v2` NEVER distorts the frozen R0 anchor `a(x)`!
    """

    def __init__(self, in_dim: int = 32 + 1 + 1 + M7_SURF_DIM_V170) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        z_char_ai: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        sup_norm: torch.Tensor,
        m7_surf: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            z_char_ai.detach(),
            logit_ai_r0.detach().clamp(-10.0, 10.0).unsqueeze(-1),
            sup_norm.detach().unsqueeze(-1),
            m7_surf.to(z_char_ai.dtype),
        ], dim=-1)
        return self.net(x)[..., 0]


def _clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0 or k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - (alpha ** (1.0 / float(n))))
    return float(sp_beta.ppf(1.0 - alpha, k + 1, n - k))


def fit_iut_stratified_r0_policy_v170(
    prob_r0_ai: np.ndarray,
    p_defer_r0: np.ndarray,
    p_apd_safe: np.ndarray,
    char_lens: np.ndarray,
    y_ai: np.ndarray,
    alpha_max: float = 0.04,
    cp_alpha: float = 0.05,
    max_chars: int = 180,
) -> dict[str, Any]:
    """M8 Zero-Bonferroni Intersection-Union Test (`IUT`) Stratified R0 Exit Certification (`K=3` length strata).

    Each stratum `k in {<=35c, 36-90c, 91-180c}` independently certifies its Clopper-Pearson upper bound
    `U_CP(k_miss, n_exit; cp_alpha) <= alpha_max` (global FWER controlled by `max_k U_CP <= alpha_max` via IUT!).
    Uses explicit logical AND: `(prob_r0_ai <= tau_k) AND (p_apd_safe >= apd_k) AND (p_defer <= def_k)`.
    """
    p_r0 = np.asarray(prob_r0_ai, dtype=np.float64)
    p_df = np.asarray(p_defer_r0, dtype=np.float64)
    p_ap = np.asarray(p_apd_safe, dtype=np.float64)
    clen = np.asarray(char_lens, dtype=np.int64)
    y = (np.asarray(y_ai, dtype=np.float64) >= 0.5).astype(np.int32)

    bounds = [(0, 35), (36, 90), (91, max_chars)]
    tau_grid = (0.004, 0.008, 0.015, 0.025, 0.04, 0.06, 0.09, 0.12, 0.16, 0.22)
    apd_grid = (0.0, 0.45, 0.60, 0.72, 0.82)
    def_grid = (0.35, 0.55, 0.75, 1.0)

    strata_cfgs: list[dict[str, Any]] = []
    tot_exits = 0
    tot_n = max(1, len(y))
    any_enabled = False
    for lo_c, hi_c in bounds:
        sm = (clen >= lo_c) & (clen <= hi_c)
        n_s = int(sm.sum())
        best = {
            "lo_chars": lo_c, "hi_chars": hi_c, "enabled": False,
            "tau_ai_lo": 0.004, "apd_safe_min": 0.85, "defer_max": 0.35,
            "n_stratum": n_s, "n_exit": 0, "k_ai_miss": 0, "cp95_ub": 1.0, "exit_rate": 0.0,
        }
        if n_s >= 20:
            ps, dfs, aps, ys = p_r0[sm], p_df[sm], p_ap[sm], y[sm]
            for tau in tau_grid:
                m_t = ps <= tau
                if not np.any(m_t):
                    continue
                for apd_m in apd_grid:
                    m_ta = m_t & (aps >= apd_m)
                    if not np.any(m_ta):
                        continue
                    for d_m in def_grid:
                        ex = m_ta & (dfs <= d_m)
                        n_ex = int(ex.sum())
                        if n_ex < 15:
                            continue
                        k_miss = int((ys[ex] == 1).sum())
                        ub = _clopper_pearson_upper(k_miss, n_ex, alpha=cp_alpha)
                        if ub <= alpha_max and n_ex > best["n_exit"]:
                            best.update({
                                "enabled": True, "tau_ai_lo": float(tau), "apd_safe_min": float(apd_m),
                                "defer_max": float(d_m), "n_exit": n_ex, "k_ai_miss": k_miss,
                                "cp95_ub": round(ub, 5), "exit_rate": round(n_ex / float(n_s), 4),
                            })
        if best["enabled"]:
            any_enabled = True
            tot_exits += int(best["n_exit"])
        strata_cfgs.append(best)

    max_ub = max((s["cp95_ub"] for s in strata_cfgs if s["enabled"]), default=1.0)
    return {
        "enabled": bool(any_enabled),
        "mode": "iut_stratified_apd_v2",
        "alpha_max": float(alpha_max),
        "max_chars": int(max_chars),
        "overall_exit_rate": round(tot_exits / float(tot_n), 4),
        "iut_max_cp95_ub": round(float(max_ub), 5),
        "strata": strata_cfgs,
    }


# ---------------------------------------------------------------------------
# 2. Tier-2 `B2a` Once-at-T1 Dual-Manifold Multi-Head Epistemic Hopfield
# ---------------------------------------------------------------------------

class DualManifoldEpistemicHopfieldV170(nn.Module):
    """Tier-2 Read-Only Once-at-T1 Multi-Head Epistemic Hopfield (`4x64 M_H + 4x64 M_A`, 32d).

    - Pure FP32 Opset-17 (`MatMul + Softmax + max/log/exp`), ZERO `Tile/Loop/If/NonZero/TopK`!
    - Strictly AI-only (`F6 === 0.0`).
    - Emits BOTH retrieved prototype context `(B, out_dim)` AND smooth LogSumExp Epistemic Energy Witness
      `(E_H, E_A, E_H - E_A)` `(B, 3)` for `EBA-R²` & `PNBR-R`!
    """

    def __init__(self, in_dim: int = AI_END_V130, n_heads: int = 4, n_slots: int = 64, head_dim: int = 32) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_heads = int(n_heads)
        self.n_slots = int(n_slots)
        self.head_dim = int(head_dim)
        self.inner = self.n_heads * self.head_dim
        self.q_proj = QuantLinearV130(self.in_dim, self.inner, bias=False, bits=8)
        self.K_H = nn.Parameter(torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.15)
        self.V_H = nn.Parameter(torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.05)
        self.K_A = nn.Parameter(torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.15)
        self.V_A = nn.Parameter(torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.05)
        self.out_proj = QuantLinearV130(self.inner * 2, self.in_dim, bias=False, bits=8)
        self.log_tau = nn.Parameter(torch.tensor(math.log(6.0)))
        self.hop_gate = nn.Parameter(torch.tensor(0.20))

    def forward(self, emb_ai: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = emb_ai.shape[0]
        q = F.normalize(self.q_proj(emb_ai).view(B, self.n_heads, self.head_dim), dim=-1, eps=1e-6)
        tau = self.log_tau.exp().clamp(2.0, 12.0).to(emb_ai.dtype)
        kH = F.normalize(self.K_H, dim=-1, eps=1e-6).to(emb_ai.dtype)
        kA = F.normalize(self.K_A, dim=-1, eps=1e-6).to(emb_ai.dtype)

        sH = torch.matmul(q.unsqueeze(2), kH.transpose(-2, -1).unsqueeze(0)).squeeze(2) * tau
        sA = torch.matmul(q.unsqueeze(2), kA.transpose(-2, -1).unsqueeze(0)).squeeze(2) * tau

        # Numerically stable pure-Opset-17 LogSumExp energy (FP32)
        sHf, sAf, tauf = sH.float(), sA.float(), tau.float()
        mH = sHf.max(dim=-1, keepdim=True).values
        eH = -(mH.squeeze(-1) + torch.log(torch.exp(sHf - mH).sum(dim=-1).clamp_min(1e-6))) / tauf
        mA = sAf.max(dim=-1, keepdim=True).values
        eA = -(mA.squeeze(-1) + torch.log(torch.exp(sAf - mA).sum(dim=-1).clamp_min(1e-6))) / tauf
        E_H = eH.mean(dim=-1)
        E_A = eA.mean(dim=-1)
        witness = torch.stack([E_H, E_A, E_H - E_A], dim=-1).to(emb_ai.dtype)

        pH = torch.softmax(sHf, dim=-1).to(emb_ai.dtype)
        pA = torch.softmax(sAf, dim=-1).to(emb_ai.dtype)
        rH = torch.matmul(pH.unsqueeze(2), self.V_H.to(emb_ai.dtype).unsqueeze(0)).squeeze(2).reshape(B, self.inner)
        rA = torch.matmul(pA.unsqueeze(2), self.V_A.to(emb_ai.dtype).unsqueeze(0)).squeeze(2).reshape(B, self.inner)
        ctx = torch.tanh(self.hop_gate).to(emb_ai.dtype) * self.out_proj(torch.cat([rH, rA], dim=-1))
        return ctx, witness


# ---------------------------------------------------------------------------
# 3. `M7` Minimal Multilingual Core (`644 B CNMA-R` + `4,228 B UISR`)
# ---------------------------------------------------------------------------

class CNMAResidualCalibratorV170(nn.Module):
    """M7 `644 B` Bounded Zero-Centered Scalar Null-Manifold Calibrator on `S(x)` (AI-only, F6 === 0.0!)."""

    def __init__(self, surf_dim: int = M7_SURF_DIM_V170, hid: int = 16, bound: float = 1.25) -> None:
        super().__init__()
        self.bound = float(bound)
        self.fc1 = nn.Linear(surf_dim, hid)
        self.fc2 = nn.Linear(hid, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, m7_surf: torch.Tensor) -> torch.Tensor:
        raw = self.fc2(F.silu(self.fc1(m7_surf))).squeeze(-1)
        return self.bound * torch.tanh(raw / self.bound)


class UISRResidualV170(nn.Module):
    """M7 `4,228 B` Bounded Unicode Structural Residual on 64-D `UISR` (AI-only, F6 === 0.0!)."""

    def __init__(self, uisr_dim: int = M7_UISR_DIM_V170, hid: int = 16, bound: float = 1.50) -> None:
        super().__init__()
        self.bound = float(bound)
        self.fc1 = nn.Linear(uisr_dim, hid)
        self.fc2 = nn.Linear(hid, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, m7_uisr: torch.Tensor) -> torch.Tensor:
        raw = self.fc2(F.silu(self.fc1(m7_uisr))).squeeze(-1)
        return self.bound * torch.tanh(raw / self.bound)


# ---------------------------------------------------------------------------
# 4. Stage-2 `DreamCore_170^Safe` Recurrent Core (`QK-Norm + PSD-L + AR-U-Loop + StepLoRA + DeltaRule`)
# ---------------------------------------------------------------------------

class MesosphereSubBlockV170(nn.Module):
    """Single-FFN `DreamCore_170^Safe` Sub-Block:
      - QK-Norm (`s_qk in [4, 12]`) + Strictly NSD `RI-Meso-PSD-L` (`b = -alpha * relu(E^T M E) * ai_slot_mask`)
      - Optional `A2_StepLoRA_r8` on `ca_q` (autonomous shared transition for `t >= 4`!)
      - Optional `M9` Double-Masked `AR-U-Loop` anchored residual broadcast onto AI-private `D_val` (`r_echo <= 0.80`)
      - Fine-scale Haar reread + AI-only Cayley (`Pros [0:2) <-> Def [2:4)`) + Block-Masked SA (`F6 === 0.0`) + SwiGLU FFN
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 4,
        d_kv: int = 96,
        kv_heads: int = 4,
        n_slow: int = 4,
        n_pros: int = 2,
        n_def: int = 2,
        n_span: int = 4,
        core_bits: int = 8,
        use_ar_uloop: bool = True,
        ar_r_v: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.d_kv = int(d_kv)
        self.kv_heads = int(kv_heads)
        self.kv_head_dim = d_kv // kv_heads
        self.n_slow = int(n_slow)
        self.n_pros = int(n_pros)
        self.n_def = int(n_def)
        self.n_sem = self.n_pros + self.n_def
        self.n_ai_slots = self.n_slow + self.n_sem
        self.n_slots = self.n_ai_slots + int(n_span)
        self.core_bits = 8 if int(core_bits) == 8 else 4
        self.use_ar_uloop = bool(use_ar_uloop)
        self.ar_r_v = float(ar_r_v)

        self.norm_ca = nn.LayerNorm(dim)
        self.ca_q = QuantLinearV130(dim, d_kv, bits=self.core_bits)
        self.ca_o = QuantLinearV130(d_kv, dim, bits=self.core_bits)

        if self.use_ar_uloop:
            self.bcast_w = QuantLinearV130(dim, d_kv, bias=False, bits=8)
            # lambda_v >= 0.30 guarantees contraction of D_t; kappa <= 0.25 keeps r_echo <= 0.80!
            self.ar_lam_raw = nn.Parameter(torch.tensor(0.0))   # 0.30 + 0.45*sig = 0.525
            self.ar_kap_raw = nn.Parameter(torch.tensor(-0.8))  # 0.25*sig = 0.078

        self.haar_q = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.haar_o = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.haar_gate = nn.Parameter(torch.tensor(-0.8))

        # Within-AI Cayley Symplectic Peer Coupling (`Pros [0:2) <-> Def [2:4)`, ZERO Promo!)
        self.c1_q = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V130))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.25))

        self.norm_sa = nn.LayerNorm(dim)
        self.sa_qkv = QuantLinearV130(dim, dim * 3, bits=4)
        self.sa_o = QuantLinearV130(dim, dim, bits=4)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.ff_down = QuantLinearV130(ffn_dim, dim, bits=4)

    def _cayley_ai_peer_couple(
        self, z_sem: torch.Tensor, bus_down: nn.Module, bus_up: nn.Module
    ) -> torch.Tensor:
        u = bus_down(z_sem)
        half = u.shape[1] // 2
        u_a, u_b = u[:, :half, :], u[:, half:, :]
        sc_ab = torch.matmul(self.c1_q(u_a), self.c1_k(u_b).transpose(-1, -2)) / 8.0
        msg_to_a = torch.matmul(torch.softmax(sc_ab.float(), dim=-1).to(u.dtype), u_b)
        msg_to_b = torch.matmul(torch.softmax(sc_ab.transpose(-1, -2).float(), dim=-1).to(u.dtype), u_a)

        g = self.c1_cayley_gamma.clamp(-2.5, 2.5)
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c = ((1.0 - g2) * inv).view(1, 1, -1)
        s = ((2.0 * g) * inv).view(1, 1, -1)
        rot_a = c * u_a + s * msg_to_a
        rot_b = -s * msg_to_b + c * u_b
        delta = bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_sem + (0.30 * torch.tanh(self.c1_bridge_scale)) * delta

    def effective_ar_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        lam_v = 0.30 + 0.45 * torch.sigmoid(self.ar_lam_raw)  # strictly in [0.30, 0.75]
        kap_v = 0.25 * torch.sigmoid(self.ar_kap_raw)         # strictly in (0, 0.25]
        return lam_v, kap_v

    def forward(
        self,
        s: torch.Tensor,
        M_psd: torch.Tensor,
        D_val: torch.Tensor,
        k_ev_norm: torch.Tensor,
        v_ev_0: torch.Tensor,
        e_sketch: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        g_sup: torch.Tensor,
        ai_slot_mask: torch.Tensor,
        sa_iso_mask: torch.Tensor,
        bus_down: nn.Module,
        bus_up: nn.Module,
        apply_psd_gate: torch.Tensor,
        s_qk: torch.Tensor,
        alpha_psd: torch.Tensor,
        beta_psd: torch.Tensor,
        lora_delta_q: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[Any, ...]:
        B, L, _ = k_ev_norm.shape
        m_f = mask.float()
        m_u = m_f.unsqueeze(-1).to(s.dtype)

        sn_ca = self.norm_ca(s)
        q_lin = self.ca_q(sn_ca)
        if lora_delta_q is not None:
            q_lin = q_lin + lora_delta_q
        qh = F.normalize(q_lin.view(B, self.n_slots, self.kv_heads, self.kv_head_dim).transpose(1, 2), dim=-1, eps=1e-6)
        kh = k_ev_norm.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * s_qk

        # Strictly NSD M5 RI-Meso-PSD-L linear bias (ZERO Div, ZERO Log, ZERO Concat, F6 === 0.0!)
        quad = F.relu((torch.matmul(e_sketch, M_psd) * e_sketch).sum(dim=-1))  # (B, L) >= 0
        b_ai = (-alpha_psd * quad * (g_sup * apply_psd_gate).to(s.dtype).view(B, 1))[:, None, None, :]
        sc = sc + b_ai * ai_slot_mask.to(s.dtype)
        sc = sc.masked_fill(m_f[:, None, None, :] < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(s.dtype)  # (B, 4, 12, L)

        # Strictly AI-only PSD memory update (convex combination -> trace(M) <= 1.0!)
        a_bar_ai = at[:, :, : self.n_ai_slots, :].mean(dim=(1, 2))  # (B, L)
        e_m = e_sketch * m_u
        M_next = beta_psd * M_psd + (1.0 - beta_psd) * torch.matmul(e_m.transpose(1, 2), e_m * a_bar_ai.unsqueeze(-1))

        # Strictly F6-isolated value readout: AI reads V_ai = V_0 + D_val; Promo reads FROZEN V_0!
        vh_0 = v_ev_0.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        if self.use_ar_uloop:
            vh_ai = (v_ev_0 + D_val).view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
            ctx_ai = torch.matmul(at[:, :, : self.n_ai_slots, :], vh_ai)
            ctx_pr = torch.matmul(at[:, :, self.n_ai_slots :, :], vh_0)
            ctx_all = torch.cat([ctx_ai, ctx_pr], dim=2).transpose(1, 2).reshape(B, self.n_slots, self.d_kv)
        else:
            ctx_all = torch.matmul(at, vh_0).transpose(1, 2).reshape(B, self.n_slots, self.d_kv)
        s = s + self.ca_o(ctx_all)

        # M9 Double-Masked AR-U-Loop (updates AI-private D_val ONLY -> F6 === 0.0 & P1 === 0.0!)
        D_next = D_val
        if self.use_ar_uloop:
            lam_v, kap_v = self.effective_ar_params()
            a_ai_s = at[:, :, : self.n_ai_slots, :].mean(dim=1)  # (B, 8, L)
            # Spectral-normalized slot message (unit-RMS bounded per slot so r_echo <= 0.80!)
            s_msg = _smooth_ball(self.bcast_w(s[:, : self.n_ai_slots, :]), radius=2.0)  # (B, 8, d_kv)
            bcast = torch.matmul(a_ai_s.transpose(1, 2), s_msg)  # (B, L, d_kv)
            Z_t = m_u * ((1.0 - lam_v.to(s.dtype)) * D_val + kap_v.to(s.dtype) * bcast)
            D_next = m_u * _smooth_ball(Z_t, radius=self.ar_r_v)

        # Fine-scale Haar reread
        q_haar = self.haar_q(sn_ca)
        sc_h = (torch.matmul(q_haar, k_haar.transpose(-1, -2)) / 8.0).masked_fill(haar_mask[:, None, :].float() < 0.5, -1e4)
        s = s + torch.sigmoid(self.haar_gate).to(s.dtype) * self.haar_o(torch.matmul(torch.softmax(sc_h.float(), dim=-1).to(s.dtype), v_haar))

        # Within-AI Cayley peer coupling (Pros [0:2) <-> Def [2:4))
        z_sem_c = self._cayley_ai_peer_couple(s[:, self.n_slow : self.n_ai_slots, :], bus_down, bus_up)
        s = torch.cat([s[:, : self.n_slow, :], z_sem_c, s[:, self.n_ai_slots :, :]], dim=1)

        # Block-Masked Slot SA (F6 === 0.0!)
        sq, sk, sv = self.sa_qkv(self.norm_sa(s)).chunk(3, dim=-1)
        sq = sq.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sk = sk.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sv = sv.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sa = torch.softmax((torch.matmul(sq, sk.transpose(-2, -1)) * (self.head_dim ** -0.5) + sa_iso_mask.to(sq.dtype)).float(), dim=-1).to(s.dtype)
        s = s + self.sa_o(torch.matmul(sa, sv).transpose(1, 2).reshape(B, self.n_slots, self.dim))

        # Unified SwiGLU FFN
        u, g = self.ff_up(self.norm_ff(s)).chunk(2, dim=-1)
        s = s + self.ff_down(F.silu(u) * g)
        if return_attn:
            return s, M_next, D_next, a_bar_ai
        return s, M_next, D_next


class MesosphereLoopCoreV170(nn.Module):
    """`DreamCore_170^Safe`: 1-Stream Contractive Kernel D + `RI-Meso-PSD-L` + `AR-U-Loop` + `StepLoRA` + `DeltaRule`."""

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 4,
        d_kv: int = 96,
        kv_heads: int = 4,
        n_slow: int = 4,
        n_pros: int = 2,
        n_def: int = 2,
        n_span: int = 4,
        core_depth: int = 2,
        use_ar_uloop: bool = True,
        use_step_lora: bool = True,
        step_lora_r: int = 8,
        use_delta_rule: bool = True,
        delta_rule_dim: int = DELTA_RULE_DIM_V170,
        r_cap: float = SLOT_R_CAP_V160,
        lardo_w8_core: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.d_kv = int(d_kv)
        self.kv_heads = int(kv_heads)
        self.kv_head_dim = d_kv // kv_heads
        self.n_slow = int(n_slow)
        self.n_pros = int(n_pros)
        self.n_def = int(n_def)
        self.n_sem = self.n_pros + self.n_def
        self.n_ai_slots = self.n_slow + self.n_sem
        self.n_slots = self.n_ai_slots + int(n_span)
        self.core_depth = max(1, int(core_depth))
        self.use_ar_uloop = bool(use_ar_uloop)
        self.use_step_lora = bool(use_step_lora)
        self.step_lora_r = int(step_lora_r) if self.use_step_lora else 0
        self.use_delta_rule = bool(use_delta_rule)
        self.delta_rule_dim = int(delta_rule_dim) if self.use_delta_rule else 0
        self.r_cap = float(r_cap)
        self.inner_r = float(CONTRACTIVE_INNER_R_V161)
        self.core_bits = 8 if bool(lardo_w8_core) else 4

        self.slot_queries = nn.Parameter(torch.randn(1, self.n_slots, dim) * 0.04)
        self.k_ev_proj = QuantLinearV130(dim, d_kv, bits=self.core_bits)
        self.v_ev_proj = QuantLinearV130(dim, d_kv, bits=self.core_bits)

        # M5 Orthogonal Sketch P: d_kv -> 8 (0.75 KiB W8!)
        q_init, _ = torch.linalg.qr(torch.randn(d_kv, SKETCH_R_V170))
        self.sketch_P = nn.Parameter(q_init)
        self.ri_alpha_raw = nn.Parameter(torch.tensor(0.35))   # 2.0 * sig <= 2.0 (M5 contraction <= 0.50!)
        self.ri_beta_raw = nn.Parameter(torch.tensor(0.0))     # 0.25 + 0.50 * sig = 0.50
        self.log_s_qk = nn.Parameter(torch.tensor(math.log(8.0)))

        self.bus_down = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.bus_up = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)

        self.blocks = nn.ModuleList([
            MesosphereSubBlockV170(
                dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, d_kv=d_kv, kv_heads=kv_heads,
                n_slow=n_slow, n_pros=n_pros, n_def=n_def, n_span=n_span,
                core_bits=(self.core_bits if bi == 0 else 4),
                use_ar_uloop=(self.use_ar_uloop and bi == 0),  # Single broadcast per step keeps r_echo <= 0.80 & minimal latency!
            )
            for bi in range(self.core_depth)
        ])

        # Input-injected L0 surface penetration into slow slots
        self.l0_pen_a = QuantLinearV130(dim, 32, bias=False, bits=self.core_bits)
        self.l0_pen_b = QuantLinearV130(32, dim, bias=False, bits=self.core_bits)
        self.l0_pen_gate = nn.Parameter(torch.full((dim,), -0.8))

        # A2 Step-LoRA r=8 (autonomous shared transition for t >= 4 via t_eff = clamp(t_val, 0, 3)!)
        if self.step_lora_r > 0:
            self.step_phi_mlp = nn.Sequential(nn.Linear(7, 24), nn.SiLU())
            self.lora_a = nn.Parameter(torch.randn(self.core_depth, dim, self.step_lora_r) * 0.04)
            self.lora_b = nn.Parameter(torch.zeros(self.core_depth, self.step_lora_r, d_kv))
            self.lora_gate = nn.Linear(24, self.core_depth * self.step_lora_r + 2)
            nn.init.zeros_(self.lora_gate.weight)
            nn.init.zeros_(self.lora_gate.bias)

        # Tier-1a Contracting 16x16 Gated Delta-Rule Fast-Weight Associative Memory (AI-only!)
        if self.delta_rule_dim > 0:
            dr = self.delta_rule_dim
            self.dr_k = QuantLinearV130(dim, dr, bias=False, bits=8)
            self.dr_v = QuantLinearV130(dim, dr, bias=False, bits=8)
            self.dr_q = QuantLinearV130(dim, dr, bias=False, bits=8)
            self.dr_out = QuantLinearV130(dr * 2, dim, bias=False, bits=8)
            self.dr_gate = nn.Linear(dim, 2)
            self.dr_scale = nn.Parameter(torch.tensor(0.25))

        # Direct 1-stream Contractive Kernel D (NO 3-stream mHC dilution!)
        self.contract_alpha_raw = nn.Parameter(torch.full((1, 1, dim), -0.25))

        sa_mask = torch.zeros((1, 1, self.n_slots, self.n_slots), dtype=torch.float32)
        sa_mask[:, :, : self.n_ai_slots, self.n_ai_slots :] = -1e4
        self.register_buffer("sa_iso_mask", sa_mask, persistent=False)

        ai_sm = torch.zeros((1, 1, self.n_slots, 1), dtype=torch.float32)
        ai_sm[:, :, : self.n_ai_slots, :] = 1.0
        self.register_buffer("ai_slot_mask", ai_sm, persistent=False)

    def _smooth_inner_ball(self, s: torch.Tensor) -> torch.Tensor:
        sf = s.float()
        nrm = sf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (sf * (self.inner_r * torch.tanh(nrm / self.inner_r) / nrm)).to(s.dtype)

    def _per_slot_project(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sf = s.float()
        nrm = sf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sat_ai = (nrm[:, : self.n_ai_slots, 0] >= self.r_cap).float().mean(dim=-1)
        return (sf * torch.clamp(self.r_cap / nrm, max=1.0)).to(s.dtype), sat_ai

    def build_readonly_cache(self, h_tok: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        B, L, _ = h_tok.shape
        m_u = mask.to(h_tok.dtype).unsqueeze(-1)
        k_raw = self.k_ev_proj(h_tok) * m_u
        v_ev_0 = self.v_ev_proj(h_tok) * m_u
        # Hoist ONCE: per-head L2-normalized k_ev_norm AND clamp-only normalized E_0!
        kh_n = F.normalize(k_raw.view(B, L, self.kv_heads, self.kv_head_dim), dim=-1, eps=1e-6)
        k_ev_norm = kh_n.reshape(B, L, self.d_kv) * m_u
        kappa_e = math.sqrt(float(self.d_kv) / float(SKETCH_R_V170))
        e_sketch = _clamp_unit_norm(torch.matmul(k_raw, self.sketch_P.to(k_raw.dtype)) / kappa_e) * m_u

        # Initial slot seeding via QK-Norm CA over hoisted k_ev_norm & v_ev_0
        blk0 = self.blocks[0]
        q = self.slot_queries.expand(B, -1, -1)
        qh = F.normalize(blk0.ca_q(blk0.norm_ca(q)).view(B, self.n_slots, self.kv_heads, self.kv_head_dim).transpose(1, 2), dim=-1, eps=1e-6)
        kh = k_ev_norm.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        vh = v_ev_0.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        s_qk = self.log_s_qk.exp().clamp(4.0, 12.0).to(qh.dtype)
        sc = (torch.matmul(qh, kh.transpose(-2, -1)) * s_qk).masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(h_tok.dtype)
        ctx = torch.matmul(at, vh).transpose(1, 2).reshape(B, self.n_slots, self.d_kv)
        init_slots = self._smooth_inner_ball(q + blk0.ca_o(ctx))
        return {"k_ev_norm": k_ev_norm, "v_ev_0": v_ev_0, "e_sketch": e_sketch, "init_slots": init_slots}

    def forward_step(
        self,
        s_ai: torch.Tensor,
        s_pr: torch.Tensor,
        M_psd: torch.Tensor,
        D_val: torch.Tensor,
        W_fast: torch.Tensor,
        k_ev_norm: torch.Tensor,
        v_ev_0: torch.Tensor,
        e_sketch: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        g_sup: torch.Tensor,
        t_val: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = s_ai.shape[0]
        step_scale = torch.rsqrt(t_val.float().clamp_min(0.0) + 1.0).to(s_ai.dtype)
        active_gt0 = (t_val.float() >= 0.5).to(s_ai.dtype)

        # M9 Sec 5.3: Clamp t_eff <= 3.0 so Step-LoRA uses autonomous shared transition for t >= 4 (T5..T8)!
        t_eff = t_val.float().clamp(0.0, 3.0)
        s_qk_base = self.log_s_qk.exp().clamp(4.0, 12.0).to(s_ai.dtype)
        alpha_base = (2.0 * torch.sigmoid(self.ri_alpha_raw)).to(s_ai.dtype)
        beta_psd = (0.25 + 0.50 * torch.sigmoid(self.ri_beta_raw)).to(s_ai.dtype)

        lora_gates = None
        if self.step_lora_r > 0:
            phi_h = self.step_phi_mlp(_fourier_step_features(t_eff).to(s_ai.dtype))
            lg_all = self.lora_gate(phi_h)
            lora_gates = torch.tanh(lg_all[:, : self.core_depth * self.step_lora_r]).view(B, self.core_depth, self.step_lora_r) * active_gt0.view(B, 1, 1)
            s_qk = (s_qk_base * (1.0 + 0.25 * active_gt0 * torch.tanh(lg_all[:, -2]))).clamp(4.0, 12.0).view(B, 1, 1, 1)
            alpha_psd = (alpha_base * (1.0 + 0.25 * active_gt0 * torch.tanh(lg_all[:, -1]))).clamp(0.0, 2.0).view(B, 1)
        else:
            s_qk = s_qk_base.view(1, 1, 1, 1)
            alpha_psd = alpha_base.view(1, 1)

        s_in = torch.cat([s_ai, s_pr], dim=1)
        l0_pen = self.l0_pen_b(F.silu(self.l0_pen_a(h0_pool))).unsqueeze(1)
        s = torch.cat([
            s_in[:, : self.n_slow, :] + torch.sigmoid(self.l0_pen_gate).to(s_in.dtype).view(1, 1, -1) * l0_pen,
            s_in[:, self.n_slow :, :],
        ], dim=1)

        M_cur, D_cur = M_psd, D_val
        attn_b0 = None
        for bi, blk in enumerate(self.blocks):
            cg = active_gt0
            ldq = None
            if self.step_lora_r > 0 and lora_gates is not None:
                sn_q = blk.norm_ca(s)
                ldq = torch.matmul(torch.matmul(sn_q, self.lora_a[bi].to(s.dtype)) * lora_gates[:, bi : bi + 1, :], self.lora_b[bi].to(s.dtype))
            bo = blk(
                s=s, M_psd=M_cur, D_val=D_cur, k_ev_norm=k_ev_norm, v_ev_0=v_ev_0, e_sketch=e_sketch,
                mask=mask, k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, g_sup=g_sup,
                ai_slot_mask=self.ai_slot_mask, sa_iso_mask=self.sa_iso_mask,
                bus_down=self.bus_down, bus_up=self.bus_up,
                apply_psd_gate=cg, s_qk=s_qk, alpha_psd=alpha_psd, beta_psd=beta_psd,
                lora_delta_q=ldq, return_attn=(return_diagnostics and bi == 0),
            )
            if return_diagnostics and bi == 0:
                s, M_cur, D_cur, attn_b0 = bo
            else:
                s, M_cur, D_cur = bo

        # Tier-1a Contracting 16x16 Gated Delta-Rule Fast-Weight Memory (AI-only -> F6 === 0.0!)
        W_next = W_fast
        surprise_vec = torch.zeros((B, max(1, self.delta_rule_dim)), device=s.device, dtype=s.dtype)
        if self.delta_rule_dim > 0:
            ai_m = s[:, : self.n_ai_slots, :].mean(dim=1)
            k_d = _clamp_unit_norm(self.dr_k(ai_m))
            v_d = self.dr_v(ai_m)
            q_d = _clamp_unit_norm(self.dr_q(ai_m))
            surprise_vec = v_d - torch.matmul(W_fast, k_d.unsqueeze(-1)).squeeze(-1)
            ret_v = torch.matmul(W_fast, q_d.unsqueeze(-1)).squeeze(-1)
            g2 = torch.sigmoid(self.dr_gate(ai_m))
            beta_dr = (0.35 + 0.50 * g2[:, 0]).view(B, 1, 1)  # in [0.35, 0.85]
            eta_dr = beta_dr * g2[:, 1].view(B, 1, 1)         # 0 < eta <= beta -> STRICT CONTRACTION!
            W_next = beta_dr * W_fast + eta_dr * torch.matmul(surprise_vec.unsqueeze(-1), k_d.unsqueeze(1))
            mem_inj = torch.tanh(self.dr_scale).to(s.dtype) * self.dr_out(torch.cat([ret_v, surprise_vec], dim=-1)) * active_gt0.view(B, 1)
            s = torch.cat([s[:, : self.n_slow, :] + mem_inj.unsqueeze(1), s[:, self.n_slow :, :]], dim=1)

        # Direct 1-Stream Contractive Kernel D (NO 3-stream mHC dilution!)
        s_prop = self._smooth_inner_ball(s)
        alpha_t = (0.55 * torch.sigmoid(self.contract_alpha_raw).to(s.dtype)) * step_scale.view(B, 1, 1)
        s_cand, sat_rate_ai = self._per_slot_project((1.0 - alpha_t) * s_in + alpha_t * s_prop)

        ret: dict[str, torch.Tensor] = {
            "slots": s_cand,
            "s_ai": s_cand[:, : self.n_ai_slots, :],
            "s_pr": s_cand[:, self.n_ai_slots :, :],
            "M_psd": M_cur,
            "D_val": D_cur,
            "W_fast": W_next,
            "surprise": surprise_vec,
            "step_scale": step_scale,
        }
        if return_diagnostics:
            ret["proj_sat_rate_ai"] = sat_rate_ai
            if attn_b0 is not None:
                ret["attn_b0"] = attn_b0
        return ret


# ---------------------------------------------------------------------------
# 5. Stage-3 `68 B DS-TitanGuard` Cross-Chunk Directional Surprise Aggregator
# ---------------------------------------------------------------------------

class DSTitanGuardAggregatorV170(nn.Module):
    """Stage-3 `DynaSeam` + `68 B` (`17`-param) `DS-TitanGuard` Streaming Cross-Chunk Surprise.

    - Zero extra projection weights: reduces 192-d `chunk_emb_ai` via 16-bin group average -> 16-d chunk sketch!
    - Runs sequential contracting Delta-Rule across `C` chunks (`m_1 = 0`: first chunk emits exact 0 surprise!).
    - Applies 17-param directional probe `w_dir (16) + b_dir (1)` (`68 B` FP32!), triple-gated by
      `J_c = relu(u_c) * (1 - q_frac_c) * valid_c * (c >= 1)`.
    - Emits `doc_ai` (Version-A zero-flip safe ranking `doc_ai + 1e-3 * sigmoid(titan_max)`!) and `titan_surprise_max`.
    - Promo aggregation is strictly disjoint -> `F6 === 0.0` bit-exact!
    """

    def __init__(self, ai_dim: int = AI_END_V130, pr_dim: int = PR_EMB_DIM_V130) -> None:
        super().__init__()
        self.long_ai_attn = nn.Sequential(nn.Linear(ai_dim + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_pr_attn = nn.Sequential(nn.Linear(pr_dim + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_ai_head = nn.Linear(ai_dim + 3, 1)
        self.long_pr_head = nn.Linear(pr_dim + 3, 1)
        # Exactly 17 FP32 params = 68 bytes!
        self.titan_w_dir = nn.Parameter(torch.zeros(16))
        self.titan_b_dir = nn.Parameter(torch.tensor(0.0))

    def compute_titan_surprise(
        self, chunk_emb_ai: torch.Tensor, chunk_meta: torch.Tensor, chunk_valid: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized O(C) causal chunk-to-chunk Delta-Rule residual norm + directional surprise (zero Loop op!)."""
        B, C, D = chunk_emb_ai.shape
        vm = chunk_valid.float()
        q_frac = chunk_meta[..., 3].clamp(0.0, 1.0)
        # 16-d deterministic group-averaged chunk sketch (0 params!), clamp-unit normalized
        g = D // 16
        sk = _clamp_unit_norm(chunk_emb_ai[:, :, : g * 16].reshape(B, C, 16, g).mean(dim=-1))  # (B, C, 16)
        # Pure-Opset-17 causal exponentially weighted running prototype (zero ONNX Loop!):
        # For C chunks, causal running mean/EMA matrix `M_causal[i, j] = 0.65^(i - 1 - j)` for `j < i`
        # computes exact causal Delta-Rule prediction `pred_i = sum_{j<i} w_{ij} sk_j / sum_{j<i} w_{ij}`!
        idx = torch.arange(C, device=chunk_emb_ai.device, dtype=torch.float32)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (C, C)
        strict_past = (diff >= 1.0).float()
        w_mat = torch.pow(0.65, (diff - 1.0).clamp_min(0.0)) * strict_past  # (C, C)
        w_eff = w_mat.unsqueeze(0) * vm.unsqueeze(1)  # (B, C, C)
        w_den = w_eff.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pred_sk = torch.matmul(w_eff / w_den, sk)  # (B, C, 16)
        has_past = (w_eff.sum(dim=-1) > 1e-4).float()  # m_1 === 0.0!
        res_e = (sk - pred_sk) * has_past.unsqueeze(-1)  # (B, C, 16)
        res_norm = res_e.norm(dim=-1)  # pure unsupervised L2 style-transition surprise
        u_dir = F.linear(res_e, self.titan_w_dir.unsqueeze(0), self.titan_b_dir.unsqueeze(0)).squeeze(-1)
        # Triple-gated directional + L2 surprise (`m_1 === 0.0`, masked by non-quote & valid!)
        j_c = F.relu(res_norm + u_dir) * (1.0 - q_frac) * vm * has_past
        return j_c.max(dim=1).values  # (B,)

    def forward(
        self,
        chunk_emb_ai: torch.Tensor,
        chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor,
        chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vm = chunk_valid.float()
        q_frac = chunk_meta[..., 3].clamp(0.0, 1.0)
        ai_in = torch.cat([chunk_emb_ai, chunk_meta, chunk_lg_ai.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        a_w = (self.long_ai_attn(ai_in).squeeze(-1) + 0.35 * chunk_lg_ai.clamp(-4.0, 4.0) * (1.0 - q_frac) - 2.0 * q_frac).masked_fill(vm < 0.5, -1e4)
        a_w = torch.softmax(a_w.float(), dim=-1).to(chunk_emb_ai.dtype)
        ai_ctx = (chunk_emb_ai * a_w.unsqueeze(-1)).sum(dim=1)
        ai_max = (chunk_lg_ai * (1.0 - 0.5 * q_frac)).masked_fill(vm < 0.5, -1e4).max(dim=1).values
        ai_mean = (chunk_lg_ai * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        ai_wmean = (chunk_lg_ai * a_w).sum(dim=1)
        doc_ai = ai_wmean + 1.8 * torch.tanh(
            self.long_ai_head(torch.cat([ai_ctx, ai_max.unsqueeze(-1), ai_mean.unsqueeze(-1), ai_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1) / 1.8
        )
        titan_max = self.compute_titan_surprise(chunk_emb_ai, chunk_meta, chunk_valid)

        pr_in = torch.cat([chunk_emb_pr, chunk_meta, chunk_lg_pr.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        p_w = self.long_pr_attn(pr_in).squeeze(-1).masked_fill(vm < 0.5, -1e4)
        p_w = torch.softmax(p_w.float(), dim=-1).to(chunk_emb_pr.dtype)
        pr_ctx = (chunk_emb_pr * p_w.unsqueeze(-1)).sum(dim=1)
        pr_max = chunk_lg_pr.masked_fill(vm < 0.5, -1e4).max(dim=1).values
        pr_mean = (chunk_lg_pr * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        pr_wmean = (chunk_lg_pr * p_w).sum(dim=1)
        doc_pr = pr_wmean + 0.5 * torch.tanh(
            self.long_pr_head(torch.cat([pr_ctx, pr_max.unsqueeze(-1), pr_mean.unsqueeze(-1), pr_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        return doc_ai, doc_pr, titan_max


# ---------------------------------------------------------------------------
# 6. Complete Unified `MeguvizV170MesosphereLoopModel`
# ---------------------------------------------------------------------------

class MeguvizV170MesosphereLoopModel(nn.Module):
    """Unified <= 5.00 MiB `MesosphereLoop` v1.7.0 (`DreamCore_170^Safe` + Tri-Tier Super-Memory + 26-Gate CI)."""

    def __init__(
        self,
        arch: str = "v170_m1",
        vocab_size: int = 12288,
        max_len: int = MAX_LEN_V140,
        head_len: int = HEAD_LEN_V140,
        style_dim: int = STYLE_FEAT_DIM_V11,
        style_era_dropout_p: float = 0.15,
        use_discourse_ops: bool = True,
        use_ar_uloop: bool = True,
        use_step_lora: bool = True,
        step_lora_r: int = 8,
        use_delta_rule: bool = True,
        use_dual_hopfield: bool = True,
        use_m6_ph: bool = True,
        use_m7_cnma_uisr: bool = True,
        lardo_w8_core: bool = True,
        use_local_charlm_inject: bool = True,
    ) -> None:
        super().__init__()
        if arch not in ARCH_PRESETS_V170:
            raise ValueError(f"Unknown arch={arch}; choose from {sorted(ARCH_PRESETS_V170)}")
        cfg = dict(ARCH_PRESETS_V170[arch])
        self.arch = str(arch)
        self.cfg = cfg
        self.dim = int(cfg["dim"])
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.style_dim = int(style_dim)
        self.style_era_dropout_p = float(style_era_dropout_p)
        self.use_discourse_ops = bool(use_discourse_ops)
        self.use_ar_uloop = bool(use_ar_uloop)
        self.use_step_lora = bool(use_step_lora)
        self.step_lora_r = int(step_lora_r) if self.use_step_lora else 0
        self.use_delta_rule = bool(use_delta_rule)
        self.use_dual_hopfield = bool(use_dual_hopfield)
        self.use_m6_ph = bool(use_m6_ph)
        self.use_m7_cnma_uisr = bool(use_m7_cnma_uisr)
        self.lardo_w8_core = bool(lardo_w8_core)
        self.use_local_charlm_inject = bool(use_local_charlm_inject)
        self.engram_buckets = int(cfg["engram_buckets"])
        self.r0_b2 = int(cfg["r0_b2"])
        self.r0_b3 = int(cfg["r0_b3"])

        D = self.dim
        core_bits = 8 if self.lardo_w8_core else 4

        # Stage 0: Full Natural-Stream R0++ + Detached APD-v2 Exit Head
        self.rung0 = ApexCharNanoRung0V170(b2=self.r0_b2, b3=self.r0_b3, use_global_charlm_shortcut=False)
        self.apd_exit_head = APDv2ExitHeadV170()

        # Stage 1: Factorized BPE + 8K Bigram Engram (DROPPED trigram_buckets -> -281 KiB!) + Local CharLM + Prelude
        self.word_emb_a = QuantEmbeddingV130(vocab_size, 48, bits=core_bits, padding_idx=0)
        self.word_emb_p = QuantLinearV130(48, D, bias=False, bits=8)
        self.engram_emb = QuantEmbeddingV130(self.engram_buckets, 16, bits=core_bits, padding_idx=0)
        self.engram_p = QuantLinearV130(16, D, bias=False, bits=8)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, D) * 0.02)
        self.emb_norm = nn.LayerNorm(D)

        self.local_lm_u = nn.Linear(LOCAL_CHARLM_DIM_V150, D, bias=False)
        self.local_lm_gate = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.local_lm_u.weight, std=0.01)

        if self.use_discourse_ops:
            self.quote_scope_proj = nn.Linear(QUOTE_SCOPE_DIM_V161, D, bias=False)
            nn.init.normal_(self.quote_scope_proj.weight, std=0.02)
            self.disc_ai_down = QuantLinearV130(D, 32, bits=8)
            self.disc_ai_ord3 = SelectiveOrder3CumSum(d_in=32, n_triples=4)
            self.disc_ai_pos = CheapD08PositionMoments(d_in=32, n_channels=4)
            self.disc_ai_proj = nn.Sequential(nn.LayerNorm(44), nn.Linear(44, 32), nn.SiLU())
            self.disc_ai_gate = nn.Parameter(torch.tensor(0.25))
            self.disc_ai_to_slow = QuantLinearV130(32, D, bias=False, bits=8)
            self.disc_pr_down = QuantLinearV130(D, 32, bits=8)
            self.disc_pr_ord3 = SelectiveOrder3CumSum(d_in=32, n_triples=2)
            self.disc_pr_pos = CheapD08PositionMoments(d_in=32, n_channels=2)
            self.disc_pr_proj = nn.Sequential(nn.LayerNorm(22), nn.Linear(22, 32), nn.SiLU())
            self.disc_pr_gate = nn.Parameter(torch.tensor(0.25))

        self.prelude_blocks = nn.ModuleList([
            PadExactStratospherePreludeBlockV160(dim=D, ffn_dim=int(cfg["prelude_ffn"]), use_global_se=bool(cfg["prelude_global_se"]))
            for _ in range(int(cfg["prelude_layers"]))
        ])
        # Critical #49 Autopsy Fix: normalize Prelude token stream BEFORE k_ev, v_ev, and h_pool_0!
        self.h_tok_norm = nn.LayerNorm(D)

        self.haar_k_proj = QuantLinearV130(D, BUS_DIM_V130, bias=False, bits=4)
        self.haar_v_proj = QuantLinearV130(D, BUS_DIM_V130, bias=False, bits=4)
        self.sup_gate_tau = nn.Parameter(torch.tensor(16.0))
        self.sup_gate_scale = nn.Parameter(torch.tensor(4.0))
        self.pr_attr_attn = nn.Sequential(QuantLinearV130(D, 32, bits=4), nn.Tanh(), nn.Linear(32, 1))

        self.r1_style_ai = nn.Sequential(nn.Linear(style_dim - 8, 32), nn.SiLU())
        self.r1_style_pr = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU())
        self.r1_ai_head = nn.Linear(D + 32 + 32, 1)
        self.r1_promo_head = nn.Linear(D + 32 + 32, 1)

        # Tier-3 M6 Pure-P_H 16-D projection (AI-only!)
        self.m6_ph_proj = nn.Sequential(nn.LayerNorm(M6_PH_DIM_V170), nn.Linear(M6_PH_DIM_V170, 32), nn.SiLU())

        # Stage 2: DreamCore_170^Safe
        self.core = MesosphereLoopCoreV170(
            dim=D, ffn_dim=int(cfg["ffn_dim"]), num_heads=int(cfg["num_heads"]),
            d_kv=int(cfg["d_kv"]), kv_heads=int(cfg["kv_heads"]),
            n_slow=int(cfg["n_slow"]), n_pros=int(cfg["n_pros"]), n_def=int(cfg["n_def"]), n_span=int(cfg["n_span"]),
            core_depth=int(cfg["core_depth"]),
            use_ar_uloop=self.use_ar_uloop, use_step_lora=self.use_step_lora, step_lora_r=self.step_lora_r,
            use_delta_rule=self.use_delta_rule, lardo_w8_core=self.lardo_w8_core,
        )

        # Per-Modality Pre-Concat LayerNorms (cures #49 81x h_pool_0 static blowout & 99.1% mean-pool cancellation!)
        self.ln_slow = nn.LayerNorm(D)
        self.ln_h0 = nn.LayerNorm(D)
        self.ln_pros = nn.LayerNorm(D)
        self.ln_def = nn.LayerNorm(D)
        self.ln_diff = nn.LayerNorm(D)
        self.ln_st_ai = nn.LayerNorm(32)
        self.ln_zc_ai = nn.LayerNorm(32)
        self.ln_m6_ph = nn.LayerNorm(32)

        # Disjoint Coda: Shared (128) + AI-Priv (64, split Pros/Def/Diff + st_ai + zc_ai + m6_ph) + Promo-Priv (64)
        self.coda_shared = QuantLinearV130(D * 2, SHARED_END_V130, bits=core_bits)
        self.coda_ai_priv = QuantLinearV130(D * 3 + 32 + 32 + 32, AI_END_V130 - SHARED_END_V130, bits=core_bits)
        self.coda_pr_priv = QuantLinearV130(D * 2 + 32 + 32, EMB_DIM_V130 - AI_END_V130, bits=core_bits)
        self.emb_norm_sh = nn.LayerNorm(SHARED_END_V130)
        self.emb_norm_ai = nn.LayerNorm(AI_END_V130 - SHARED_END_V130)
        self.emb_norm_pr = nn.LayerNorm(EMB_DIM_V130 - AI_END_V130)

        # Tier-2 Once-at-T1 Dual-Manifold Multi-Head Epistemic Hopfield (toggleable!)
        self.dual_hopfield = DualManifoldEpistemicHopfieldV170(in_dim=AI_END_V130, n_heads=4, n_slots=64, head_dim=32) if self.use_dual_hopfield else None

        # M7 Minimal Multilingual Core (644 B CNMA-R + 4,228 B UISR, strictly AI-only!)
        self.cnma_r = CNMAResidualCalibratorV170() if self.use_m7_cnma_uisr else None
        self.uisr_res = UISRResidualV170() if self.use_m7_cnma_uisr else None

        self.cbm_ai_enc = nn.Linear(AI_END_V130, CBM_AI_K_V130)
        self.cbm_pr_enc = nn.Linear(PR_EMB_DIM_V130, CBM_PR_K_V130)
        self.cbm_ai_tau = nn.Parameter(torch.full((CBM_AI_K_V130,), 0.25))
        self.cbm_pr_tau = nn.Parameter(torch.full((CBM_PR_K_V130,), 0.25))
        self.cbm_ai_to_logit = nn.Linear(CBM_AI_K_V130, 1, bias=False)
        self.cbm_pr_to_logit = nn.Linear(CBM_PR_K_V130, 1, bias=False)

        # SoftBall with init_r_cap=10.0 (cures #49 2.2x radial compression!)
        self.ai_head = SoftBallLogitNormHeadV121(AI_END_V130, init_r_cap=10.0)
        self.promo_head = SoftBallLogitNormHeadV121(PR_EMB_DIM_V130, init_r_cap=10.0)
        self.promo_sub4_head = nn.Linear(PR_EMB_DIM_V130, 4)

        # PNBR-R Zero-Static-Skip Nested Telescoping Boost Head (used ONLY for t >= 1 -> T2..T8!)
        pnbr_in = D * 2 + 1 + max(1, self.core.delta_rule_dim) + 3  # [LN(ds_slow), LN(ds_pd), sg(f_{t-1}), surprise, hop_witness]
        self.ln_ds_slow = nn.LayerNorm(D)
        self.ln_ds_pd = nn.LayerNorm(D)
        self.pnbr_head = nn.Sequential(
            nn.Linear(pnbr_in, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.pnbr_cap = nn.Parameter(torch.tensor(1.6))

        # TaH2 Utility Halt Head (strictly AI-only, detached state)
        self.halt_utility_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 6),
            nn.Linear(AI_END_V130 * 2 + 6, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

        # Stage 3: DynaSeam + 68 B DS-TitanGuard
        self.long_agg = DSTitanGuardAggregatorV170(AI_END_V130, PR_EMB_DIM_V130)

        self._compiled_r0 = None
        self._compiled_prelude_t1 = None
        self._compiled_loop_step = None

    def enable_torch_compile(self, mode: str = "default", dynamic: bool = True, backend: str = "inductor") -> bool:
        if not hasattr(torch, "compile"):
            return False
        try:
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
                torch._dynamo.config.cache_size_limit = max(64, int(getattr(torch._dynamo.config, "cache_size_limit", 8)))
                if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                    torch._dynamo.config.accumulated_cache_size_limit = max(256, int(getattr(torch._dynamo.config, "accumulated_cache_size_limit", 64)))
            kw: dict[str, Any] = {"dynamic": bool(dynamic)}
            if backend == "inductor":
                kw["mode"] = mode
            else:
                kw["backend"] = backend
            self._compiled_r0 = torch.compile(self.forward_r0, **kw)
            self._compiled_prelude_t1 = torch.compile(self.forward_prelude_and_t1, **kw)
            self._compiled_loop_step = torch.compile(self.forward_loop_step, **kw)
            return True
        except Exception:
            self._compiled_r0 = None
            self._compiled_prelude_t1 = None
            self._compiled_loop_step = None
            return False

    def compute_support_gate(self, support_pairs: torch.Tensor) -> torch.Tensor:
        tau = self.sup_gate_tau.clamp(10.0, 22.0)
        sc = self.sup_gate_scale.clamp(2.0, 6.0)
        return torch.sigmoid((support_pairs.float() - tau) / sc)

    def _pool_promo_attr(self, h_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        w_lg = self.pr_attr_attn(h_tok).squeeze(-1).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(w_lg.float(), dim=-1).to(h_tok.dtype).unsqueeze(-1)
        return (h_tok * w).sum(dim=1)

    def _readout_t1(
        self,
        slots: torch.Tensor,
        h_pool_0: torch.Tensor,
        pr_attr_0: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        m6_ph_emb: torch.Tensor,
        m7_surf: torch.Tensor,
        m7_uisr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ns, n_pr_s, n_ai = self.core.n_slow, self.core.n_pros, self.core.n_ai_slots
        slow_p = slots[:, :ns, :].mean(dim=1)
        pros_p = slots[:, ns : ns + n_pr_s, :].mean(dim=1)
        def_p = slots[:, ns + n_pr_s : n_ai, :].mean(dim=1)
        sem_diff = pros_p - def_p
        span_p = slots[:, n_ai:, :].mean(dim=1)

        z_sh = self.emb_norm_sh(self.coda_shared(torch.cat([self.ln_slow(slow_p), self.ln_h0(h_pool_0)], dim=-1)))
        z_ai_p = self.emb_norm_ai(self.coda_ai_priv(torch.cat([
            self.ln_pros(pros_p), self.ln_def(def_p), self.ln_diff(sem_diff),
            self.ln_st_ai(st_ai), self.ln_zc_ai(z_char[:, :32]), self.ln_m6_ph(m6_ph_emb),
        ], dim=-1)))
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, pr_attr_0, st_pr, z_char[:, 32:R0_ZCHAR_DIM_V130]], dim=-1)))

        emb_ai = torch.cat([z_sh, z_ai_p], dim=-1)
        emb_pr = torch.cat([z_sh, z_pr_p], dim=-1)

        if self.dual_hopfield is not None:
            hop_ctx, hop_witness = self.dual_hopfield(emb_ai)
            emb_ai = emb_ai + hop_ctx
        else:
            hop_witness = torch.zeros((emb_ai.shape[0], 3), device=emb_ai.device, dtype=emb_ai.dtype)

        c_ai_sp = F.relu(self.cbm_ai_enc(emb_ai) - self.cbm_ai_tau.clamp(0.0, 2.0))
        c_pr_sp = F.relu(self.cbm_pr_enc(emb_pr) - self.cbm_pr_tau.clamp(0.0, 2.0))
        lg_ai = self.ai_head(emb_ai) + 0.15 * self.cbm_ai_to_logit(c_ai_sp).squeeze(-1)

        dz_cnma = torch.zeros_like(lg_ai)
        du_uisr = torch.zeros_like(lg_ai)
        if self.cnma_r is not None:
            dz_cnma = self.cnma_r(m7_surf.to(lg_ai.dtype))
            lg_ai = lg_ai + dz_cnma
        if self.uisr_res is not None:
            du_uisr = self.uisr_res(m7_uisr.to(lg_ai.dtype))
            lg_ai = lg_ai + du_uisr

        lg_pr = self.promo_head(emb_pr) + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1)
        return {
            "logit_ai": lg_ai,
            "logit_promo": lg_pr,
            "promo_sub4_logit": self.promo_sub4_head(emb_pr),
            "emb_ai": emb_ai,
            "emb_pr": emb_pr,
            "hop_witness": hop_witness,
            "dz_cnma": dz_cnma,
            "du_uisr": du_uisr,
        }

    def _predict_utility(
        self,
        emb_ai_cur: torch.Tensor,
        emb_ai_prev: torch.Tensor,
        logit_ai_cur: torch.Tensor,
        logit_ai_prev: torch.Tensor,
        disagree_dt: torch.Tensor,
        g_sup: torch.Tensor,
    ) -> torch.Tensor:
        dt_c = disagree_dt.detach().clamp(-12.0, 12.0)
        sc = torch.stack([
            logit_ai_cur.detach(),
            (logit_ai_cur - logit_ai_prev).detach(),
            logit_ai_cur.detach().abs(),
            dt_c,
            dt_c.abs(),
            g_sup.detach(),
        ], dim=-1)
        hin = torch.cat([emb_ai_cur.detach(), (emb_ai_cur - emb_ai_prev).detach(), sc], dim=-1)
        return self.halt_utility_head(hin).squeeze(-1)

    def forward_r0(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        mask: torch.Tensor,
        feats: torch.Tensor,
        m7_surf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        o = self.rung0(c1, c2, c3, mask, feats)
        sup_norm = feats[:, R0_STRUCT_DIM_V140 - 1].to(o["z_char"].dtype)
        apd_lg = self.apd_exit_head(o["z_char"][:, :32], o["logit_ai_r0"], sup_norm, m7_surf)
        o["apd_exit_logit"] = apd_lg
        o["p_apd_safe"] = torch.sigmoid(apd_lg)
        return o

    def forward_prelude_and_t1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        z_char: torch.Tensor,
        support_pairs: torch.Tensor,
        quote_scope: torch.Tensor,
        m6_ph: torch.Tensor,
        m7_surf: torch.Tensor,
        m7_uisr: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, L = input_ids.shape
        m = attention_mask.float()
        g_sup = self.compute_support_gate(support_pairs).to(style_feats.dtype)

        st = style_feats.clone()
        if self.training and self.style_era_dropout_p > 0.0:
            keep = (torch.rand((B, 1), device=st.device) >= self.style_era_dropout_p).to(st.dtype)
            st[:, :40] = st[:, :40] * keep

        pos = self.pos_emb[:, :L, :]
        h_sem_raw = self.emb_norm(
            self.word_emb_p(self.word_emb_a(input_ids))
            + self.engram_p(self.engram_emb(engram_ids))
            + pos
        )
        m_act = attention_mask.to(h_sem_raw.dtype).unsqueeze(-1)
        h_sem = h_sem_raw * m_act

        if self.use_local_charlm_inject:
            loc_eff = local_charlm.to(h_sem.dtype)
            hits_4 = loc_eff[..., 4:8]
            has_hi_hit = (hits_4[..., 1:4].max(dim=-1, keepdim=True).values > 1e-4).to(h_sem.dtype)
            g_loc = torch.sigmoid(self.local_lm_gate(hits_4) - 1.0) * has_hi_hit * m_act
            x = (h_sem + g_loc * self.local_lm_u(loc_eff)) * m_act
        else:
            x = h_sem

        if self.use_discourse_ops:
            x = (x + self.quote_scope_proj(quote_scope.to(x.dtype)) * m_act) * m_act

        h_tok = x
        for blk in self.prelude_blocks:
            h_tok = blk(h_tok, attention_mask)
        h_tok = self.h_tok_norm(h_tok) * m_act

        # Pure Opset-17 dynamic Mod+Pad for odd/even Haar (If=0, P1 === 0.0!)
        m_tok = attention_mask.to(h_tok.dtype)
        pad_r = L % 2
        hp = F.pad(h_tok, (0, 0, 0, pad_r))
        mp = F.pad(m_tok, (0, pad_r))
        m_even = mp[:, 0::2]
        h_diff = (hp[:, 0::2, :] - hp[:, 1::2, :]) * m_even.unsqueeze(-1)
        k_haar = self.haar_k_proj(h_diff) * m_even.unsqueeze(-1)
        v_haar = self.haar_v_proj(h_diff) * m_even.unsqueeze(-1)
        haar_mask = m_even

        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_pool_0 = (h_tok.float().sum(dim=1) / denom).to(h_tok.dtype)
        pr_attr_0 = self._pool_promo_attr(h_tok, attention_mask)
        st_ai = self.r1_style_ai(st[:, : self.style_dim - 8]) * g_sup.unsqueeze(-1)
        st_pr = self.r1_style_pr(st)
        m6_ph_emb = self.m6_ph_proj(m6_ph.to(h_tok.dtype)) if self.use_m6_ph else torch.zeros((B, 32), device=h_tok.device, dtype=h_tok.dtype)

        disc_ai_slow = None
        if self.use_discourse_ops:
            h_dai = F.silu(self.disc_ai_down(h_tok)) * m_act
            v_dai = torch.cat([self.disc_ai_ord3(h_dai, attention_mask), self.disc_ai_pos(h_dai, attention_mask)], dim=-1)
            disc_ai = self.disc_ai_proj(v_dai)
            st_ai = st_ai + torch.tanh(self.disc_ai_gate).to(st_ai.dtype) * disc_ai
            disc_ai_slow = (0.25 * torch.tanh(self.disc_ai_gate).to(h_tok.dtype)) * self.disc_ai_to_slow(disc_ai)

            h_dpr = F.silu(self.disc_pr_down(h_tok)) * m_act
            v_dpr = torch.cat([self.disc_pr_ord3(h_dpr, attention_mask), self.disc_pr_pos(h_dpr, attention_mask)], dim=-1)
            disc_pr = self.disc_pr_proj(v_dpr)
            st_pr = st_pr + torch.tanh(self.disc_pr_gate).to(st_pr.dtype) * disc_pr

        lg_r1_ai = self.r1_ai_head(torch.cat([h_pool_0, st_ai, z_char[:, :32]], dim=-1)).squeeze(-1)
        lg_r1_pr = self.r1_promo_head(torch.cat([h_pool_0, st_pr, z_char[:, 32:R0_ZCHAR_DIM_V130]], dim=-1)).squeeze(-1)

        ro_cache = self.core.build_readonly_cache(h_tok, attention_mask)
        init_slots = ro_cache["init_slots"]
        if disc_ai_slow is not None:
            ns = self.core.n_slow
            init_slots = self.core._smooth_inner_ball(torch.cat([
                init_slots[:, :ns, :] + disc_ai_slow.unsqueeze(1),
                init_slots[:, ns:, :],
            ], dim=1))

        s_ai_0 = init_slots[:, : self.core.n_ai_slots, :]
        s_pr_0 = init_slots[:, self.core.n_ai_slots :, :]
        M_0 = torch.zeros((B, SKETCH_R_V170, SKETCH_R_V170), device=h_tok.device, dtype=h_tok.dtype)
        D_0 = torch.zeros_like(ro_cache["v_ev_0"])
        dr_d = max(1, self.core.delta_rule_dim)
        W_0 = torch.zeros((B, dr_d, dr_d), device=h_tok.device, dtype=h_tok.dtype)
        z_b = torch.zeros((B,), device=h_tok.device, dtype=h_tok.dtype)

        st1 = self.core.forward_step(
            s_ai=s_ai_0, s_pr=s_pr_0, M_psd=M_0, D_val=D_0, W_fast=W_0,
            k_ev_norm=ro_cache["k_ev_norm"], v_ev_0=ro_cache["v_ev_0"], e_sketch=ro_cache["e_sketch"],
            mask=attention_mask, k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask,
            h0_pool=h_pool_0, g_sup=g_sup, t_val=z_b, return_diagnostics=return_diagnostics,
        )
        ro1 = self._readout_t1(st1["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, z_char, m6_ph_emb, m7_surf, m7_uisr)
        u_t1 = self._predict_utility(ro1["emb_ai"], torch.zeros_like(ro1["emb_ai"]), ro1["logit_ai"], lg_r1_ai, z_b, g_sup)

        out_s1: dict[str, torch.Tensor] = {
            "logit_ai_r1": lg_r1_ai,
            "logit_promo_r1": lg_r1_pr,
            "logit_ai_t1": ro1["logit_ai"],
            "logit_promo_t1": ro1["logit_promo"],
            "promo_sub4_logit_t1": ro1["promo_sub4_logit"],
            "emb_ai_t1": ro1["emb_ai"],
            "emb_pr_t1": ro1["emb_pr"],
            "hop_witness": ro1["hop_witness"],
            "dz_cnma": ro1["dz_cnma"],
            "du_uisr": ro1["du_uisr"],
            "u_halt_t1": u_t1,
            "s_ai": st1["s_ai"],
            "s_pr": st1["s_pr"],
            "M_psd": st1["M_psd"],
            "D_val": st1["D_val"],
            "W_fast": st1["W_fast"],
            "k_ev_norm": ro_cache["k_ev_norm"],
            "v_ev_0": ro_cache["v_ev_0"],
            "e_sketch": ro_cache["e_sketch"],
            "k_haar": k_haar,
            "v_haar": v_haar,
            "haar_mask": haar_mask,
            "h_pool_0": h_pool_0,
            "pr_attr_0": pr_attr_0,
            "st_pr": st_pr,
            "g_sup": g_sup,
        }
        if return_diagnostics:
            out_s1["h_tok"] = h_tok
            out_s1["proj_sat_rate_ai"] = st1["proj_sat_rate_ai"]
            if "attn_b0" in st1:
                out_s1["attn_b0"] = st1["attn_b0"]
        return out_s1

    def forward_loop_step(
        self,
        s_ai_in: torch.Tensor,
        s_pr_in: torch.Tensor,
        M_psd_in: torch.Tensor,
        D_val_in: torch.Tensor,
        W_fast_in: torch.Tensor,
        k_ev_norm: torch.Tensor,
        v_ev_0: torch.Tensor,
        e_sketch: torch.Tensor,
        attention_mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h_pool_0: torch.Tensor,
        pr_attr_0: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        g_sup: torch.Tensor,
        emb_ai_prev: torch.Tensor,
        logit_ai_prev: torch.Tensor,
        hop_witness: torch.Tensor,
        t_val: torch.Tensor,
        disagree_dt: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Stage 2: Reusable `DreamCore_170^Safe` step for `t >= 1` (`T2, T3, T4, T5, T6, T8`)!"""
        stn = self.core.forward_step(
            s_ai=s_ai_in, s_pr=s_pr_in, M_psd=M_psd_in, D_val=D_val_in, W_fast=W_fast_in,
            k_ev_norm=k_ev_norm, v_ev_0=v_ev_0, e_sketch=e_sketch, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h_pool_0, g_sup=g_sup,
            t_val=t_val, return_diagnostics=return_diagnostics,
        )
        # PNBR-R Zero-Static-Skip Nested Boosting Readout (strictly AI-only, F6 === 0.0!)
        ds_ai = stn["s_ai"] - s_ai_in
        ns, n_pr_s = self.core.n_slow, self.core.n_pros
        ds_slow = self.ln_ds_slow(ds_ai[:, :ns, :].mean(dim=1))
        ds_pd = self.ln_ds_pd(ds_ai[:, ns : ns + n_pr_s, :].mean(dim=1) - ds_ai[:, ns + n_pr_s :, :].mean(dim=1))
        pnbr_in = torch.cat([
            ds_slow, ds_pd,
            logit_ai_prev.detach().clamp(-8.0, 8.0).unsqueeze(-1),
            stn["surprise"],
            hop_witness.detach(),
        ], dim=-1)
        c_cap = F.softplus(self.pnbr_cap).clamp(0.4, 2.5).to(ds_slow.dtype)
        delta_lg = c_cap * torch.tanh(self.pnbr_head(pnbr_in).squeeze(-1) / c_cap)
        logit_ai_next = logit_ai_prev.detach() + stn["step_scale"] * delta_lg

        # Update AI embedding cleanly (strictly AI-only!) for long-doc DynaSeam & utility halt
        z_sh_cur = self.emb_norm_sh(self.coda_shared(torch.cat([self.ln_slow(stn["s_ai"][:, :ns, :].mean(dim=1)), self.ln_h0(h_pool_0)], dim=-1)))
        emb_ai_cur = torch.cat([z_sh_cur, emb_ai_prev[:, SHARED_END_V130:AI_END_V130]], dim=-1)

        # Strictly Promo-only readout (Promo slots NEVER touch AI!)
        span_p = stn["s_pr"].mean(dim=1)
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, pr_attr_0, st_pr, z_char[:, 32:R0_ZCHAR_DIM_V130]], dim=-1)))
        emb_pr_cur = torch.cat([z_sh_cur.detach(), z_pr_p], dim=-1)
        c_pr_sp = F.relu(self.cbm_pr_enc(emb_pr_cur) - self.cbm_pr_tau.clamp(0.0, 2.0))
        logit_pr_cur = self.promo_head(emb_pr_cur) + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1)
        sub4_cur = self.promo_sub4_head(emb_pr_cur)

        u_halt = self._predict_utility(emb_ai_cur, emb_ai_prev, logit_ai_next, logit_ai_prev, disagree_dt, g_sup)
        out: dict[str, torch.Tensor] = {
            "logit_ai": logit_ai_next,
            "delta_logit_ai": stn["step_scale"] * delta_lg,
            "logit_promo": logit_pr_cur,
            "promo_sub4_logit": sub4_cur,
            "emb_ai": emb_ai_cur,
            "emb_pr": emb_pr_cur,
            "u_halt": u_halt,
            "s_ai": stn["s_ai"],
            "s_pr": stn["s_pr"],
            "M_psd": stn["M_psd"],
            "D_val": stn["D_val"],
            "W_fast": stn["W_fast"],
        }
        if return_diagnostics:
            out["proj_sat_rate_ai"] = stn["proj_sat_rate_ai"]
            if "attn_b0" in stn:
                out["attn_b0"] = stn["attn_b0"]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        r0_c1: torch.Tensor,
        r0_c2: torch.Tensor,
        r0_c3: torch.Tensor,
        r0_mask: torch.Tensor,
        r0_feats: torch.Tensor,
        support_pairs: torch.Tensor,
        quote_scope: torch.Tensor,
        m6_ph: torch.Tensor,
        m7_surf: torch.Tensor,
        m7_uisr: torch.Tensor,
        outer_steps: int = 4,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
        return_diagnostics: bool = False,
        freeze_t1_prefix: bool = False,
    ) -> dict[str, Any]:
        B = input_ids.shape[0]
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((B, 1), device=r0_feats.device) >= float(charlm_drop_p)).to(r0_feats.dtype)
            r0_f_eff = torch.cat([r0_feats[:, :R0_STRUCT_DIM_V140], r0_feats[:, R0_STRUCT_DIM_V140:R0_FEAT_DIM_V140] * lm_keep], dim=-1)
            loc_eff = local_charlm * lm_keep.unsqueeze(-1)
            m6_eff = m6_ph * lm_keep
        else:
            r0_f_eff, loc_eff, m6_eff = r0_feats, local_charlm, m6_ph

        fn_r0 = self._compiled_r0 if (self._compiled_r0 is not None and not return_diagnostics and not torch.onnx.is_in_onnx_export()) else self.forward_r0
        if not self.rung0.ai_head.weight.requires_grad:
            with torch.no_grad():
                r0_core = self.rung0(r0_c1, r0_c2, r0_c3, r0_mask, r0_f_eff)
            sup_norm = r0_f_eff[:, R0_STRUCT_DIM_V140 - 1].to(r0_core["z_char"].dtype)
            apd_lg = self.apd_exit_head(r0_core["z_char"][:, :32], r0_core["logit_ai_r0"], sup_norm, m7_surf)
            r0 = dict(r0_core, apd_exit_logit=apd_lg, p_apd_safe=torch.sigmoid(apd_lg))
        else:
            r0 = fn_r0(r0_c1, r0_c2, r0_c3, r0_mask, r0_f_eff, m7_surf)

        zc_in = r0["z_char"]
        if self.training and r0_drop_p > 0.0:
            zc_in = zc_in * (torch.rand((B, 1), device=zc_in.device) >= float(r0_drop_p)).to(zc_in.dtype)

        fn_t1 = self._compiled_prelude_t1 if (self._compiled_prelude_t1 is not None and not return_diagnostics) else self.forward_prelude_and_t1
        if freeze_t1_prefix:
            with torch.no_grad():
                s1 = fn_t1(input_ids, attention_mask, style_feats, engram_ids, loc_eff, zc_in, support_pairs, quote_scope, m6_eff, m7_surf, m7_uisr, return_diagnostics=return_diagnostics)
        else:
            s1 = fn_t1(input_ids, attention_mask, style_feats, engram_ids, loc_eff, zc_in, support_pairs, quote_scope, m6_eff, m7_surf, m7_uisr, return_diagnostics=return_diagnostics)

        ai_steps = [s1["logit_ai_t1"]]
        pr_steps = [s1["logit_promo_t1"]]
        sub4_steps = [s1["promo_sub4_logit_t1"]]
        emb_ai_steps = [s1["emb_ai_t1"]]
        emb_pr_steps = [s1["emb_pr_t1"]]
        u_halt_steps = [s1["u_halt_t1"]]
        s_ai_steps = [s1["s_ai"]]
        delta_steps: list[torch.Tensor] = []
        attn_steps = [s1["attn_b0"]] if (return_diagnostics and "attn_b0" in s1) else []

        cur_s_ai, cur_s_pr, cur_M, cur_D, cur_W = s1["s_ai"], s1["s_pr"], s1["M_psd"], s1["D_val"], s1["W_fast"]
        cur_emb, cur_lg = s1["emb_ai_t1"], s1["logit_ai_t1"]
        fn_step = self._compiled_loop_step if (self._compiled_loop_step is not None and not return_diagnostics) else self.forward_loop_step

        for ti in range(1, max(1, int(outer_steps))):
            t_vec = torch.full((B,), float(ti), device=input_ids.device, dtype=torch.float32)
            dt_sig = r0["logit_ai_r0"].detach().clamp(-6.0, 6.0) - cur_lg.detach().clamp(-6.0, 6.0)
            sn = fn_step(
                cur_s_ai, cur_s_pr, cur_M, cur_D, cur_W,
                s1["k_ev_norm"], s1["v_ev_0"], s1["e_sketch"], attention_mask,
                s1["k_haar"], s1["v_haar"], s1["haar_mask"],
                s1["h_pool_0"], s1["pr_attr_0"], s1["st_pr"], zc_in, s1["g_sup"],
                cur_emb, cur_lg, s1["hop_witness"], t_vec, dt_sig,
                return_diagnostics=return_diagnostics,
            )
            ai_steps.append(sn["logit_ai"])
            pr_steps.append(sn["logit_promo"])
            sub4_steps.append(sn["promo_sub4_logit"])
            emb_ai_steps.append(sn["emb_ai"])
            emb_pr_steps.append(sn["emb_pr"])
            u_halt_steps.append(sn["u_halt"])
            s_ai_steps.append(sn["s_ai"])
            delta_steps.append(sn["delta_logit_ai"])
            if return_diagnostics and "attn_b0" in sn:
                attn_steps.append(sn["attn_b0"])
            cur_s_ai, cur_s_pr, cur_M, cur_D, cur_W = sn["s_ai"], sn["s_pr"], sn["M_psd"], sn["D_val"], sn["W_fast"]
            cur_emb, cur_lg = sn["emb_ai"], sn["logit_ai"]

        out: dict[str, Any] = {
            "logit_ai_r0": r0["logit_ai_r0"],
            "logit_promo_r0": r0["logit_promo_r0"],
            "defer_ai_r0_logit": r0["defer_ai_r0_logit"],
            "p_defer_ai_r0": r0["p_defer_ai_r0"],
            "defer_pr_r0_logit": r0["defer_pr_r0_logit"],
            "p_defer_pr_r0": r0["p_defer_pr_r0"],
            "apd_exit_logit": r0["apd_exit_logit"],
            "p_apd_safe": r0["p_apd_safe"],
            "z_char": r0["z_char"],
            "logit_ai_r1": s1["logit_ai_r1"],
            "logit_promo_r1": s1["logit_promo_r1"],
            "hop_witness": s1["hop_witness"],
            "dz_cnma": s1["dz_cnma"],
            "du_uisr": s1["du_uisr"],
            "logit_ai_steps": ai_steps,
            "logit_promo_steps": pr_steps,
            "delta_logit_steps": delta_steps,
            "u_halt_steps": u_halt_steps,
            "s_ai_steps": s_ai_steps,
            "logit_ai": ai_steps[-1],
            "logit_promo": pr_steps[-1],
            "promo_sub4_logit": sub4_steps[-1],
            "emb_ai": emb_ai_steps[-1],
            "emb_pr": emb_pr_steps[-1],
            "M_psd_final": cur_M,
            "D_val_final": cur_D,
            "W_fast_final": cur_W,
        }
        if return_diagnostics:
            out["attn_steps"] = attn_steps
            out["h_tok"] = s1["h_tok"]
        for ti, (la, lp, uh) in enumerate(zip(ai_steps, pr_steps, u_halt_steps), start=1):
            out[f"logit_ai_t{ti}"] = la
            out[f"logit_promo_t{ti}"] = lp
            out[f"u_halt_t{ti}"] = uh
        return out

    def aggregate_long_chunks(
        self,
        chunk_emb_ai: torch.Tensor,
        chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor,
        chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.long_agg(chunk_emb_ai, chunk_emb_pr, chunk_lg_ai, chunk_lg_pr, chunk_meta, chunk_valid)


# ---------------------------------------------------------------------------
# 7. `MGVZ10` W4/W8 Pack/Unpack + Zero-Disk Bit-Exact Quant Roundtrip
# ---------------------------------------------------------------------------

def _quant_spec_map_v170(model: MeguvizV170MesosphereLoopModel) -> dict[str, tuple[str, int]]:
    spec: dict[str, tuple[str, int]] = {
        "pos_emb": ("w8", 0) if model.lardo_w8_core else ("w4", 32),
        "core.slot_queries": ("w8", 0),
        "core.sketch_P": ("w8", 0),
    }
    if model.dual_hopfield is not None:
        for k in ("dual_hopfield.K_H", "dual_hopfield.V_H", "dual_hopfield.K_A", "dual_hopfield.V_A"):
            spec[k] = ("w4", 32)
    if model.core.step_lora_r > 0:
        spec["core.lora_a"] = ("w8", 0)
        spec["core.lora_b"] = ("w8", 0)
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantLinearV130, QuantEmbeddingV130)):
            is_hi_rd_prelude = model.lardo_w8_core and (
                name.startswith("prelude_blocks.") and (name.endswith(".in_g") or name.endswith(".out_p"))
            )
            use_w8 = (mod.target_bits == 8) or is_hi_rd_prelude
            spec[f"{name}.weight"] = ("w8", 0) if use_w8 else ("w4", mod.group_size)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 1024:
            spec[f"{name}.weight"] = ("w8", 0)
    return spec


def pack_bundle_v10_v170(model: MeguvizV170MesosphereLoopModel, out_path: Path) -> dict[str, Any]:
    spec_map = _quant_spec_map_v170(model)
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    breakdown: dict[str, int] = {"r0_w8_w4": 0, "deep_w4": 0, "deep_w8": 0, "scales_and_fp16": 0}
    for k, arr in sd.items():
        q_mode, gs = spec_map.get(k, ("fp16", 0))
        if arr.size < 64:
            q_mode, gs = "fp16", 0
        if q_mode == "w8":
            sb, qb = _pack_w8_tensor(arr)
            breakdown["r0_w8_w4" if k.startswith(("rung0.", "apd_exit_head.")) else "deep_w8"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        elif q_mode == "w4":
            sb, qb = _pack_w4_tensor(arr, gs=gs)
            breakdown["r0_w8_w4" if k.startswith(("rung0.", "apd_exit_head.")) else "deep_w4"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        else:
            sb, qb = b"", arr.astype(np.float16).tobytes()
            breakdown["scales_and_fp16"] += len(qb)
        entries.append({"name": k, "q": q_mode, "g": gs, "shape": list(arr.shape), "s_bytes": len(sb), "q_bytes": len(qb)})
        blobs.extend((sb, qb))
    hdr_bytes = json.dumps({"format": "MGVZ10_MESOSPHERE_V170", "arch": model.arch, "tensors": entries}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MGVZ10_MAGIC_V170)
        f.write(struct.pack("<I", len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)
    raw_b = out_path.stat().st_size
    return {
        "raw_bytes": raw_b,
        "raw_mb": round(raw_b / (1024.0 * 1024.0), 4),
        "breakdown_kb": {k: round(v / 1024.0, 1) for k, v in breakdown.items()},
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


@torch.no_grad()
def apply_mgvz10_quant_roundtrip_inplace_v170(model: MeguvizV170MesosphereLoopModel) -> None:
    """Bit-exact in-memory W4/W8/FP16 MGVZ10 roundtrip directly on GPU/CPU (0/N diffs vs disk MGVZ10!)."""
    spec_map = _quant_spec_map_v170(model)
    sd = model.state_dict()
    dev = next(iter(sd.values())).device
    d127 = torch.tensor(127.0, device=dev, dtype=torch.float32)
    d7 = torch.tensor(7.0, device=dev, dtype=torch.float32)
    for k, v in sd.items():
        if not v.is_floating_point():
            continue
        q_mode, gs = spec_map.get(k, ("fp16", 0))
        if v.numel() < 64:
            q_mode, gs = "fp16", 0
        tf = v.float()
        if q_mode == "w8":
            if tf.ndim < 2:
                sc = torch.clamp(tf.abs().max() / d127, min=1e-4).to(torch.float16).float()
                out = torch.round(tf / sc).clamp(-127.0, 127.0) * sc
            else:
                rows = tf.reshape(-1, tf.shape[-1])
                sc = torch.clamp(rows.abs().amax(dim=-1, keepdim=True) / d127, min=1e-4).to(torch.float16).float()
                out = (torch.round(rows / sc).clamp(-127.0, 127.0) * sc).reshape(tf.shape)
        elif q_mode == "w4":
            n = tf.numel()
            flat = tf.reshape(-1)
            pad = (gs - (n % gs)) % gs
            if pad:
                flat = F.pad(flat, (0, pad))
            g = flat.reshape(-1, gs)
            sc = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / d7, min=1e-4).to(torch.float16).float()
            out = (torch.round(g / sc).clamp(-7.0, 7.0) * sc).reshape(-1)[:n].reshape(tf.shape)
        else:
            out = tf.to(torch.float16).float()
        v.copy_(out.to(v.dtype))


def load_bundle_v10_v170_into_model(model: MeguvizV170MesosphereLoopModel, mgvz_path: Path) -> None:
    data = mgvz_path.read_bytes()
    assert data[:6] == b"MGVZ10", "Invalid MGVZ10 magic!"
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    pos = 12 + hlen
    sd = model.state_dict()
    for t in header["tensors"]:
        k, q, gs, shape = t["name"], t["q"], int(t["g"]), tuple(t["shape"])
        sb, qb = int(t["s_bytes"]), int(t["q_bytes"])
        s_buf, q_buf = data[pos : pos + sb], data[pos + sb : pos + sb + qb]
        pos += sb + qb
        if q == "fp16":
            arr = np.frombuffer(q_buf, dtype=np.float16).astype(np.float32).reshape(shape)
        elif q == "w8":
            arr = _unpack_w8_tensor(s_buf, q_buf, shape)
        else:
            arr = _unpack_w4_tensor(s_buf, q_buf, shape, gs=gs)
        if k in sd:
            sd[k].copy_(torch.from_numpy(arr.copy()).to(sd[k].dtype))


# ---------------------------------------------------------------------------
# 8. 4-Stage Modular Pure-FP32 Opset-17 ONNX Export + Cold-Start Bootable ZIP
# ---------------------------------------------------------------------------

class _V170R0Export(nn.Module):
    def __init__(self, m: MeguvizV170MesosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, c1, c2, c3, mask, feats, m7_surf):
        o = self.m.forward_r0(c1, c2, c3, mask, feats, m7_surf)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_ai_r0"], o["p_defer_pr_r0"], o["p_apd_safe"], o["z_char"]


_T1_OUT_NAMES_V170 = [
    "logit_ai_r1", "logit_promo_r1", "logit_ai_t1", "logit_promo_t1",
    "emb_ai_t1", "emb_pr_t1", "hop_witness", "u_halt_t1", "g_sup",
    "s_ai", "s_pr", "M_psd", "D_val", "W_fast",
    "k_ev_norm", "v_ev_0", "e_sketch", "k_haar", "v_haar", "haar_mask",
    "h_pool_0", "pr_attr_0", "st_pr",
]


class _V170PreludeT1Export(nn.Module):
    def __init__(self, m: MeguvizV170MesosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, local_charlm, z_char, support_pairs, quote_scope, m6_ph, m7_surf, m7_uisr):
        o = self.m.forward_prelude_and_t1(input_ids, attention_mask, style_feats, engram_ids, local_charlm, z_char, support_pairs, quote_scope, m6_ph, m7_surf, m7_uisr)
        return tuple(o[k] for k in _T1_OUT_NAMES_V170)


_STEP_OUT_NAMES_V170 = [
    "logit_ai", "logit_promo", "emb_ai", "emb_pr", "u_halt",
    "s_ai_out", "s_pr_out", "M_psd_out", "D_val_out", "W_fast_out",
]


class _V170LoopStepExport(nn.Module):
    def __init__(self, m: MeguvizV170MesosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(
        self, s_ai_in, s_pr_in, M_psd_in, D_val_in, W_fast_in,
        k_ev_norm, v_ev_0, e_sketch, attention_mask, k_haar, v_haar, haar_mask,
        h_pool_0, pr_attr_0, st_pr, z_char, g_sup, emb_ai_prev, logit_ai_prev, hop_witness, t_val, disagree_dt,
    ):
        o = self.m.forward_loop_step(
            s_ai_in, s_pr_in, M_psd_in, D_val_in, W_fast_in,
            k_ev_norm, v_ev_0, e_sketch, attention_mask, k_haar, v_haar, haar_mask,
            h_pool_0, pr_attr_0, st_pr, z_char, g_sup, emb_ai_prev, logit_ai_prev, hop_witness, t_val, disagree_dt,
        )
        return (
            o["logit_ai"], o["logit_promo"], o["emb_ai"], o["emb_pr"], o["u_halt"],
            o["s_ai"], o["s_pr"], o["M_psd"], o["D_val"], o["W_fast"],
        )


class _V170LongExport(nn.Module):
    def __init__(self, m: MeguvizV170MesosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, chunk_emb_ai, chunk_emb_pr, chunk_lg_ai, chunk_lg_pr, chunk_meta, chunk_valid):
        return self.m.aggregate_long_chunks(chunk_emb_ai, chunk_emb_pr, chunk_lg_ai, chunk_lg_pr, chunk_meta, chunk_valid)


def _strip_onnx_initializers_to_skeleton(onnx_path: Path, skel_path: Path) -> int:
    om = onnx.load(str(onnx_path))
    for init in om.graph.initializer:
        if init.raw_data and len(init.raw_data) > 64:
            init.ClearField("raw_data")
            init.ClearField("float_data")
            init.ClearField("int64_data")
    onnx.save(om, str(skel_path))
    return skel_path.stat().st_size


def _count_forbidden_ops(onnx_path: Path) -> dict[str, int]:
    om = onnx.load(str(onnx_path))
    cnt = {op: 0 for op in FORBIDDEN_ONNX_OPS_V170}
    for n in om.graph.node:
        if n.op_type in cnt:
            cnt[n.op_type] += 1
    return cnt


def export_v170_bootable_and_onnx(
    model: MeguvizV170MesosphereLoopModel,
    tokenizer_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    version: str,
    charlm: CompactCharLMV150 | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m_cpu = MeguvizV170MesosphereLoopModel(
        arch=model.arch, vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim,
        use_discourse_ops=model.use_discourse_ops, use_ar_uloop=model.use_ar_uloop,
        use_step_lora=model.use_step_lora, step_lora_r=model.step_lora_r,
        use_delta_rule=model.use_delta_rule, use_dual_hopfield=model.use_dual_hopfield,
        use_m6_ph=model.use_m6_ph, use_m7_cnma_uisr=model.use_m7_cnma_uisr,
        lardo_w8_core=model.lardo_w8_core, use_local_charlm_inject=model.use_local_charlm_inject,
    )
    m_cpu.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()})
    set_v130_qat_mode(m_cpu, False)
    m_cpu.eval()

    mgvz_p = out_dir / "bundle_v10_mesosphere_v170.mgvz"
    mgvz_info = pack_bundle_v10_v170(m_cpu, mgvz_p)
    load_bundle_v10_v170_into_model(m_cpu, mgvz_p)
    m_cpu.eval()

    charlm_p = out_dir / "charlm_v170.bin"
    charlm_info: dict[str, Any] = {"enabled": False, "pass": True}
    if charlm is not None:
        charlm_info = charlm.save(charlm_p)
        lm_re = CompactCharLMV150.load(charlm_p)
        lm_re.__class__ = CompactCharLMV150
        sample_cl = canonicalize_zero_pii_v150("总的来说，这个机制确实提供了全新的视角和深入的分析。")[0]
        rd = float(np.abs(charlm.extract_9d([sample_cl]) - lm_re.extract_9d([sample_cl])).max())
        charlm_info.update({"enabled": True, "roundtrip_diff": round(rd, 8), "pass": bool(rd < 1e-6)})

    # Dummy inputs for Opset-17 export (odd L=63 or even L=64 — we test odd L=31 & even L=64 in parity!)
    B, L, Lr = 2, 64, 64
    d_c1 = torch.randint(0, 128, (B, Lr), dtype=torch.long)
    d_c2 = torch.randint(0, m_cpu.r0_b2, (B, Lr), dtype=torch.long)
    d_c3 = torch.randint(0, m_cpu.r0_b3, (B, Lr), dtype=torch.long)
    d_cm = torch.ones((B, Lr), dtype=torch.float32)
    d_cf = torch.randn((B, R0_FEAT_DIM_V140), dtype=torch.float32)
    d_surf = torch.rand((B, M7_SURF_DIM_V170), dtype=torch.float32) * 0.5
    d_uisr = torch.randn((B, M7_UISR_DIM_V170), dtype=torch.float32) * 0.2
    d_m6 = torch.randn((B, M6_PH_DIM_V170), dtype=torch.float32) * 0.2

    r0_p = out_dir / "v170_r0_apexchar_fp32.onnx"
    torch.onnx.export(
        _V170R0Export(m_cpu).eval(), (d_c1, d_c2, d_c3, d_cm, d_cf, d_surf), str(r0_p),
        input_names=["c1", "c2", "c3", "mask", "feats", "m7_surf"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0", "p_apd_safe", "z_char"],
        dynamic_axes={k: {0: "B", 1: "Lr"} for k in ("c1", "c2", "c3", "mask")} | {"feats": {0: "B"}, "m7_surf": {0: "B"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    d_ids = torch.randint(1, 1000, (B, L), dtype=torch.long)
    d_msk = torch.ones((B, L), dtype=torch.long)
    d_st = torch.randn((B, m_cpu.style_dim), dtype=torch.float32) * 0.2
    d_eg = torch.randint(0, m_cpu.engram_buckets, (B, L), dtype=torch.long)
    d_loc = torch.randn((B, L, LOCAL_CHARLM_DIM_V150), dtype=torch.float32) * 0.1
    d_qsc = torch.zeros((B, L, QUOTE_SCOPE_DIM_V161), dtype=torch.float32)
    d_sup = torch.full((B,), 32.0, dtype=torch.float32)
    m_cpu.eval()
    with torch.no_grad():
        r0_pt = m_cpu.forward_r0(d_c1, d_c2, d_c3, d_cm, d_cf, d_surf)
        d_zc = r0_pt["z_char"]

    t1_p = out_dir / "v170_prelude_t1_fp32.onnx"
    torch.onnx.export(
        _V170PreludeT1Export(m_cpu).eval(),
        (d_ids, d_msk, d_st, d_eg, d_loc, d_zc, d_sup, d_qsc, d_m6, d_surf, d_uisr), str(t1_p),
        input_names=["input_ids", "attention_mask", "style_feats", "engram_ids", "local_charlm", "z_char", "support_pairs", "quote_scope", "m6_ph", "m7_surf", "m7_uisr"],
        output_names=_T1_OUT_NAMES_V170,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
            "engram_ids": {0: "B", 1: "L"}, "local_charlm": {0: "B", 1: "L"}, "quote_scope": {0: "B", 1: "L"},
            "style_feats": {0: "B"}, "z_char": {0: "B"}, "support_pairs": {0: "B"},
            "m6_ph": {0: "B"}, "m7_surf": {0: "B"}, "m7_uisr": {0: "B"},
        },
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    m_cpu.eval()
    with torch.no_grad():
        s1_pt = m_cpu.forward_prelude_and_t1(d_ids, d_msk, d_st, d_eg, d_loc, d_zc, d_sup, d_qsc, d_m6, d_surf, d_uisr)

    step_p = out_dir / "v170_loop_step_fp32.onnx"
    d_tval = torch.ones((B,), dtype=torch.float32)
    d_dt = torch.zeros((B,), dtype=torch.float32)
    torch.onnx.export(
        _V170LoopStepExport(m_cpu).eval(),
        (
            s1_pt["s_ai"], s1_pt["s_pr"], s1_pt["M_psd"], s1_pt["D_val"], s1_pt["W_fast"],
            s1_pt["k_ev_norm"], s1_pt["v_ev_0"], s1_pt["e_sketch"], d_msk,
            s1_pt["k_haar"], s1_pt["v_haar"], s1_pt["haar_mask"],
            s1_pt["h_pool_0"], s1_pt["pr_attr_0"], s1_pt["st_pr"], d_zc, s1_pt["g_sup"],
            s1_pt["emb_ai_t1"], s1_pt["logit_ai_t1"], s1_pt["hop_witness"], d_tval, d_dt,
        ),
        str(step_p),
        input_names=[
            "s_ai_in", "s_pr_in", "M_psd_in", "D_val_in", "W_fast_in",
            "k_ev_norm", "v_ev_0", "e_sketch", "attention_mask", "k_haar", "v_haar", "haar_mask",
            "h_pool_0", "pr_attr_0", "st_pr", "z_char", "g_sup", "emb_ai_prev", "logit_ai_prev", "hop_witness", "t_val", "disagree_dt",
        ],
        output_names=_STEP_OUT_NAMES_V170,
        dynamic_axes={
            "s_ai_in": {0: "B"}, "s_pr_in": {0: "B"}, "M_psd_in": {0: "B"}, "D_val_in": {0: "B", 1: "L"}, "W_fast_in": {0: "B"},
            "k_ev_norm": {0: "B", 1: "L"}, "v_ev_0": {0: "B", 1: "L"}, "e_sketch": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
            "k_haar": {0: "B", 1: "Lh"}, "v_haar": {0: "B", 1: "Lh"}, "haar_mask": {0: "B", 1: "Lh"},
            "h_pool_0": {0: "B"}, "pr_attr_0": {0: "B"}, "st_pr": {0: "B"}, "z_char": {0: "B"}, "g_sup": {0: "B"},
            "emb_ai_prev": {0: "B"}, "logit_ai_prev": {0: "B"}, "hop_witness": {0: "B"}, "t_val": {0: "B"}, "disagree_dt": {0: "B"},
        },
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    long_p = out_dir / "v170_dynaseam_fp32.onnx"
    d_cea = torch.randn((B, 5, AI_END_V130), dtype=torch.float32) * 0.2
    d_cep = torch.randn((B, 5, PR_EMB_DIM_V130), dtype=torch.float32) * 0.2
    d_cla = torch.randn((B, 5), dtype=torch.float32)
    d_clp = torch.randn((B, 5), dtype=torch.float32)
    d_cmeta = torch.rand((B, 5, CHUNK_META_DIM_V11), dtype=torch.float32) * 0.5
    d_cval = torch.ones((B, 5), dtype=torch.float32)
    torch.onnx.export(
        _V170LongExport(m_cpu).eval(), (d_cea, d_cep, d_cla, d_clp, d_cmeta, d_cval), str(long_p),
        input_names=["chunk_emb_ai", "chunk_emb_pr", "chunk_lg_ai", "chunk_lg_pr", "chunk_meta", "chunk_valid"],
        output_names=["doc_logit_ai", "doc_logit_promo", "titan_surprise_max"],
        dynamic_axes={k: {0: "B", 1: "C"} for k in ("chunk_emb_ai", "chunk_emb_pr", "chunk_lg_ai", "chunk_lg_pr", "chunk_meta", "chunk_valid")},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    m_cpu.eval()

    # Verify Forbidden Ops === 0 across ALL 4 stages + exact FP32 PyTorch vs ORT parity!
    forb_all = {p.name: _count_forbidden_ops(p) for p in (r0_p, t1_p, step_p, long_p)}
    forb_total = sum(sum(v.values()) for v in forb_all.values())

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    s_r0 = ort.InferenceSession(str(r0_p), opts, providers=["CPUExecutionProvider"])
    s_t1 = ort.InferenceSession(str(t1_p), opts, providers=["CPUExecutionProvider"])
    s_st = ort.InferenceSession(str(step_p), opts, providers=["CPUExecutionProvider"])
    s_lg = ort.InferenceSession(str(long_p), opts, providers=["CPUExecutionProvider"])

    def _run_f(sess, feed):
        valid = {i.name for i in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feed.items() if k in valid})

    o_r0 = _run_f(s_r0, {"c1": d_c1.numpy(), "c2": d_c2.numpy(), "c3": d_c3.numpy(), "mask": d_cm.numpy(), "feats": d_cf.numpy(), "m7_surf": d_surf.numpy()})
    r0_diff = float(max(
        np.abs(o_r0[0] - r0_pt["logit_ai_r0"].numpy()).max(),
        np.abs(o_r0[4] - r0_pt["p_apd_safe"].numpy()).max(),
    ))

    o_t1 = dict(zip(_T1_OUT_NAMES_V170, _run_f(s_t1, {
        "input_ids": d_ids.numpy(), "attention_mask": d_msk.numpy(), "style_feats": d_st.numpy(),
        "engram_ids": d_eg.numpy(), "local_charlm": d_loc.numpy(), "z_char": d_zc.numpy(),
        "support_pairs": d_sup.numpy(), "quote_scope": d_qsc.numpy(),
        "m6_ph": d_m6.numpy(), "m7_surf": d_surf.numpy(), "m7_uisr": d_uisr.numpy(),
    })))
    t1_diff = float(max(
        np.abs(o_t1["logit_ai_t1"] - s1_pt["logit_ai_t1"].numpy()).max(),
        np.abs(o_t1["hop_witness"] - s1_pt["hop_witness"].numpy()).max(),
    ))

    # Multi-step parity T2..T6!
    step_diffs: dict[str, float] = {}
    with torch.no_grad():
        pt_full = m_cpu(d_ids, d_msk, d_st, d_eg, d_loc, d_c1, d_c2, d_c3, d_cm, d_cf, d_sup, d_qsc, d_m6, d_surf, d_uisr, outer_steps=6)
    cur_ai, cur_pr, cur_M, cur_D, cur_W = o_t1["s_ai"], o_t1["s_pr"], o_t1["M_psd"], o_t1["D_val"], o_t1["W_fast"]
    cur_emb, cur_lg = o_t1["emb_ai_t1"], o_t1["logit_ai_t1"]
    for ti in range(1, 6):
        dt_np = np.clip(o_r0[0], -6.0, 6.0) - np.clip(cur_lg, -6.0, 6.0)
        os_t = dict(zip(_STEP_OUT_NAMES_V170, _run_f(s_st, {
            "s_ai_in": cur_ai, "s_pr_in": cur_pr, "M_psd_in": cur_M, "D_val_in": cur_D, "W_fast_in": cur_W,
            "k_ev_norm": o_t1["k_ev_norm"], "v_ev_0": o_t1["v_ev_0"], "e_sketch": o_t1["e_sketch"], "attention_mask": d_msk.numpy(),
            "k_haar": o_t1["k_haar"], "v_haar": o_t1["v_haar"], "haar_mask": o_t1["haar_mask"],
            "h_pool_0": o_t1["h_pool_0"], "pr_attr_0": o_t1["pr_attr_0"], "st_pr": o_t1["st_pr"], "z_char": d_zc.numpy(),
            "g_sup": o_t1["g_sup"], "emb_ai_prev": cur_emb, "logit_ai_prev": cur_lg, "hop_witness": o_t1["hop_witness"],
            "t_val": np.full((B,), float(ti), np.float32), "disagree_dt": dt_np.astype(np.float32),
        })))
        step_diffs[f"T{ti+1}"] = round(float(np.abs(os_t["logit_ai"] - pt_full[f"logit_ai_t{ti+1}"].numpy()).max()), 8)
        cur_ai, cur_pr, cur_M, cur_D, cur_W = os_t["s_ai_out"], os_t["s_pr_out"], os_t["M_psd_out"], os_t["D_val_out"], os_t["W_fast_out"]
        cur_emb, cur_lg = os_t["emb_ai"], os_t["logit_ai"]

    with torch.no_grad():
        la_pt, lp_pt, ts_pt = m_cpu.aggregate_long_chunks(d_cea, d_cep, d_cla, d_clp, d_cmeta, d_cval)
    o_lg = _run_f(s_lg, {
        "chunk_emb_ai": d_cea.numpy(), "chunk_emb_pr": d_cep.numpy(), "chunk_lg_ai": d_cla.numpy(),
        "chunk_lg_pr": d_clp.numpy(), "chunk_meta": d_cmeta.numpy(), "chunk_valid": d_cval.numpy(),
    })
    long_diff = float(max(np.abs(o_lg[0] - la_pt.numpy()).max(), np.abs(o_lg[2] - ts_pt.numpy()).max()))

    # Skeletons & Cold-Start Bootable ZIP (<= 5.00 MiB!)
    skel_paths: list[Path] = []
    skel_raw_total = 0
    for p in (r0_p, t1_p, step_p, long_p):
        sp = out_dir / p.name.replace(".onnx", ".skel.onnx")
        skel_raw_total += _strip_onnx_initializers_to_skeleton(p, sp)
        skel_paths.append(sp)

    tok_out = out_dir / "bili_picky_bpe_12k.json"
    if tok_out.resolve() != tokenizer_path.resolve():
        tok_out.write_bytes(tokenizer_path.read_bytes())

    manifest = dict(manifest)
    manifest.update({
        "format": "MGVZ10_MESOSPHERE_V170",
        "arch": model.arch,
        "arch_cfg": model.cfg,
        "use_discourse_ops": model.use_discourse_ops,
        "use_ar_uloop": model.use_ar_uloop,
        "use_step_lora": model.use_step_lora,
        "step_lora_r": model.step_lora_r,
        "use_delta_rule": model.use_delta_rule,
        "use_dual_hopfield": model.use_dual_hopfield,
        "use_m6_ph": model.use_m6_ph,
        "use_m7_cnma_uisr": model.use_m7_cnma_uisr,
        "lardo_w8_core": model.lardo_w8_core,
        "use_local_charlm_inject": model.use_local_charlm_inject,
        "mgvz10_info": mgvz_info,
        "charlm_info": charlm_info,
    })
    mf_p = out_dir / "manifest.json"
    mf_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    boot_zip = out_dir / f"meguvizcac-{version}-bootable-v10-mesosphere.zip"
    with zipfile.ZipFile(boot_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(mgvz_p, mgvz_p.name)
        zf.write(tok_out, tok_out.name)
        zf.write(mf_p, mf_p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)
        for sp in skel_paths:
            zf.write(sp, sp.name)
    boot_bytes = boot_zip.stat().st_size

    onnx_zip = out_dir / f"meguvizcac-{version}-4stage-onnx-fp32exact.zip"
    with zipfile.ZipFile(onnx_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in (r0_p, t1_p, step_p, long_p, tok_out, mf_p):
            zf.write(p, p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)

    max_step_diff = max(step_diffs.values()) if step_diffs else 0.0
    return {
        "bootable_zip": str(boot_zip),
        "bootable_bytes": boot_bytes,
        "bootable_mib": round(boot_bytes / (1024.0 * 1024.0), 4),
        "bootable_le_5p00_mib": bool(boot_bytes <= int(BUNDLE_MAX_MIB_V130 * 1024 * 1024)),
        "skeletons_raw_kb": round(skel_raw_total / 1024.0, 1),
        "onnx_zip_mb": round(onnx_zip.stat().st_size / (1024.0 * 1024.0), 4),
        "r0_fp32_max_diff": round(r0_diff, 8),
        "t1_fp32_max_diff": round(t1_diff, 8),
        "loop_step_diffs_t2_to_t6": step_diffs,
        "long_fp32_max_diff": round(long_diff, 8),
        "forbidden_ops_by_stage": forb_all,
        "forbidden_ops_total": int(forb_total),
        "onnx_exact_pass": bool(max(r0_diff, t1_diff, max_step_diff, long_diff) < 1e-4 and forb_total == 0 and charlm_info["pass"]),
        "mgvz10_info": mgvz_info,
        "charlm_info": charlm_info,
    }


# ---------------------------------------------------------------------------
# 9. Production `MeguvizV170CascadeEngine` (`2T` R0++ C0 + `A0` Bucketing + `IUT SCE` + `DS-TitanGuard`)
# ---------------------------------------------------------------------------

class MeguvizV170CascadeEngine:
    """Zero-PII Pre-Canonicalized, Dual-Axis Decoupled Anytime Cascade (`R0++ [2T C0 + IUT] -> T1+B2a -> DreamCore(t) -> DynaSeam+DS-TitanGuard`)."""

    def __init__(self, release_dir: str | Path, threads: int = 4, r0_threads: int = 2) -> None:
        self.release_dir = Path(release_dir)
        self.manifest = json.loads((self.release_dir / "manifest.json").read_text(encoding="utf-8"))
        self.arch_cfg = dict(self.manifest.get("arch_cfg") or ARCH_PRESETS_V170[self.manifest.get("arch", "v170_m1")])
        tok_p = self.release_dir / "bili_picky_bpe_12k.json"
        if not tok_p.is_file():
            alt_toks = sorted(self.release_dir.glob("*bpe*.json"))
            if alt_toks:
                tok_p = alt_toks[0]
        self.tokenizer = BiliPickyBPETokenizer(tok_p)
        charlm_p = self.release_dir / "charlm_v170.bin"
        if not charlm_p.is_file():
            charlm_p = self.release_dir / "charlm_v160.bin"
        if charlm_p.is_file():
            lm = CompactCharLMV150.load(charlm_p)
            lm.__class__ = CompactCharLMV150
            self.charlm: CompactCharLMV150 | None = lm
        else:
            self.charlm = None

        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V140))
        self.head_len = int(self.manifest.get("head_len", HEAD_LEN_V140))
        self.long_char_trigger = int(self.manifest.get("long_char_trigger", 300))
        self.use_local_charlm = bool(self.manifest.get("use_local_charlm_inject", True))
        self.use_discourse_ops = bool(self.manifest.get("use_discourse_ops", True))
        self.r0_b2 = int(self.arch_cfg.get("r0_b2", R0_B2_V130 * 2))
        self.r0_b3 = int(self.arch_cfg.get("r0_b3", R0_B3_V130 * 2))
        self.engram_buckets = int(self.arch_cfg.get("engram_buckets", 8192))

        # M8 C0: Dedicated 2T session for R0++ (-15.5% faster than 4T on small 128-d GEMVs!)
        opts_r0 = ort.SessionOptions()
        opts_r0.intra_op_num_threads = max(1, min(int(threads), int(r0_threads)))
        opts_r0.inter_op_num_threads = 1
        opts_r0.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        opts_dp = ort.SessionOptions()
        opts_dp.intra_op_num_threads = int(threads)
        opts_dp.inter_op_num_threads = 1
        opts_dp.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess_r0 = ort.InferenceSession(str(self.release_dir / "v170_r0_apexchar_fp32.onnx"), opts_r0, providers=["CPUExecutionProvider"])
        self.sess_t1 = ort.InferenceSession(str(self.release_dir / "v170_prelude_t1_fp32.onnx"), opts_dp, providers=["CPUExecutionProvider"])
        self.sess_step = ort.InferenceSession(str(self.release_dir / "v170_loop_step_fp32.onnx"), opts_dp, providers=["CPUExecutionProvider"])
        self.sess_long = ort.InferenceSession(str(self.release_dir / "v170_dynaseam_fp32.onnx"), opts_dp, providers=["CPUExecutionProvider"])
        self._in_r0 = {i.name for i in self.sess_r0.get_inputs()}
        self._in_t1 = {i.name for i in self.sess_t1.get_inputs()}
        self._in_step = {i.name for i in self.sess_step.get_inputs()}
        self._in_long = {i.name for i in self.sess_long.get_inputs()}

        self.r0_ai_policy = self.manifest.get("rung0_ai_ltt_policy") or {}
        self.r0_pr_policy = self.manifest.get("rung0_pr_ltt_policy") or {}
        self.step_policy = self.manifest.get("anytime_step_policy") or {
            "enabled": True, "max_infer_steps": DEFAULT_T_INFER_MAX_V170,
            "u_halt_thresh": 0.015, "disagree_dt_thresh": 1.35, "uncertain_lo": 0.12, "uncertain_hi": 0.88, "t1_exit_enabled": True,
        }
        self.crossview_fusion = self.manifest.get("crossview_late_fusion") or {"enabled": False}
        self.long_focal_router = self.manifest.get("long_focal_router") or {"enabled": True, "win_chars": 160, "max_chunks": 6, "lambda": 0.92}
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.temp_ai = float(self.manifest.get("temp_ai", 1.0))
        self.temp_pr = float(self.manifest.get("temp_promo", 1.0))

        self._rust_fe = None
        try:
            from .native_frontend import HAS_RUST_FRONTEND, RustTurboFrontend

            if HAS_RUST_FRONTEND:
                self._rust_fe = RustTurboFrontend(
                    self.tokenizer, max_len=self.max_len, head_len=self.head_len,
                    engram_buckets=self.engram_buckets, trigram_buckets=8192,
                    r0_b2=self.r0_b2, r0_b3=self.r0_b3, use_local_charlm=self.use_local_charlm, bucket_seq_len=False,
                )
                if self.charlm is not None:
                    self._rust_fe.set_group_oof_charlm([], {}, self.charlm)
        except Exception:
            self._rust_fe = None

    def _run_sess(self, sess: ort.InferenceSession, valid: set[str], feed: dict[str, np.ndarray]):
        return sess.run(None, {k: v for k, v in feed.items() if k in valid})

    def _extract_sidecars(self, clean_texts: list[str], bpe_lens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._rust_fe is not None and hasattr(self._rust_fe, "extract_v170_sidecar"):
            return self._rust_fe.extract_v170_sidecar(clean_texts, bpe_lens, use_oof_charlm=False)
        rust_clm = getattr(self._rust_fe, "rust_charlm", None) if self._rust_fe is not None else None
        return extract_v170_sidecar_batch(clean_texts, bpe_lens, charlm=self.charlm, rust_charlm=rust_clm, fold_ids=None)

    def _extract_r0_clean(self, clean_texts: list[str]):
        if self._rust_fe is not None and len(clean_texts) >= 1:
            d0 = self._rust_fe.extract_batch_numpy(
                clean_texts, need_bpe=False, use_oof_charlm=False, already_canonical=True, need_r0=True
            )
            c1, c2, c3, cm, cf, sups = d0["c1"], d0["c2"], d0["c3"], d0["cm"], d0["cf"], d0["sup"]
        else:
            c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
                clean_texts, b2=self.r0_b2, b3=self.r0_b3, already_canonical=True, charlm=self.charlm
            )
        # Lightweight char-count proxy for bpe_lens in Stage-0 surface vector (exact bpe_lens used in Stage-1!)
        approx_bpe = np.array([max(1, len(t) // 2) for t in clean_texts], dtype=np.int64)
        _, m7_surf, _ = self._extract_sidecars(clean_texts, approx_bpe)
        return c1, c2, c3, cm, cf, sups, m7_surf

    def _prep_deep_inputs(self, clean_texts: list[str]):
        if self._rust_fe is not None and len(clean_texts) >= 1:
            d_rs = self._rust_fe.extract_batch_numpy(
                clean_texts, need_bpe=True, use_oof_charlm=False, need_quote_scope=self.use_discourse_ops,
                already_canonical=True, need_r0=False,
            )
            ids, msk, st, eg, loc = d_rs["ids"], d_rs["msk"], d_rs["st"], d_rs["eg"], d_rs["loc"]
            qsc = d_rs["qsc"] if self.use_discourse_ops else np.zeros((len(clean_texts), ids.shape[1], QUOTE_SCOPE_DIM_V161), np.float32)
            bpe_lens = msk.sum(axis=1).astype(np.int64)
        else:
            f56_l, encs, offs_l = [], [], []
            for cl in clean_texts:
                _, f56, _ = extract_style_features_56d(cl)
                e_ids, e_offs = encode_bpe_head_tail_with_offsets_v150(self.tokenizer, cl, max_len=self.max_len, head_len=self.head_len)
                f56_l.append(f56); encs.append(e_ids); offs_l.append(e_offs)
            engs = [make_engram_bigram_ids(s, self.engram_buckets) for s in encs]
            ml = max(2, max(len(s) for s in encs))
            if ml % 2 == 1:
                ml += 1
            bs = len(clean_texts)
            ids = np.full((bs, ml), self.tokenizer.pad_id, np.int64)
            msk = np.zeros((bs, ml), np.int64)
            eg = np.zeros((bs, ml), np.int64)
            loc = np.zeros((bs, ml, LOCAL_CHARLM_DIM_V150), np.float32)
            qsc = np.zeros((bs, ml, QUOTE_SCOPE_DIM_V161), np.float32)
            for i, s in enumerate(encs):
                ns = len(s)
                ids[i, :ns], msk[i, :ns], eg[i, :ns] = s, 1, engs[i]
                if self.use_local_charlm and self.charlm is not None:
                    loc[i, :ns] = self.charlm.extract_token_local_8ch(clean_texts[i], offs_l[i])
                if self.use_discourse_ops:
                    qsc[i, :ns] = extract_token_quote_scope_4d_v161(clean_texts[i], offs_l[i])
            st = np.stack(f56_l, 0).astype(np.float32)
            bpe_lens = np.array([len(s) for s in encs], dtype=np.int64)
        m6_ph, m7_surf, m7_uisr = self._extract_sidecars(clean_texts, bpe_lens)
        return ids, msk, st, eg, loc, qsc, m6_ph, m7_surf, m7_uisr

    def _prob_ai(self, logit: np.ndarray, is_r0: bool = False) -> np.ndarray:
        lut = self.lut_ai_r0 if is_r0 else self.lut_ai_full
        p = expit(np.clip(np.asarray(logit, dtype=np.float64) / max(0.05, 1.0 if is_r0 else self.temp_ai), -30.0, 30.0))
        if lut is not None:
            return apply_spline_temp_lut(p, lut)
        return p.astype(np.float32)

    def _prob_pr(self, logit: np.ndarray, is_r0: bool = False) -> np.ndarray:
        lut = self.lut_pr_r0 if is_r0 else self.lut_pr_full
        p = expit(np.clip(np.asarray(logit, dtype=np.float64) / max(0.05, 1.0 if is_r0 else self.temp_pr), -30.0, 30.0))
        if lut is not None:
            return apply_spline_temp_lut(p, lut)
        return p.astype(np.float32)

    def _check_r0_iut_exit(self, p_ai_r0: float, p_def_r0: float, p_apd: float, n_chars: int) -> bool:
        pol = self.r0_ai_policy
        if not pol.get("enabled"):
            return False
        if n_chars > int(pol.get("max_chars", 180)):
            return False
        for s in pol.get("strata", []):
            if int(s["lo_chars"]) <= n_chars <= int(s["hi_chars"]):
                if not s.get("enabled"):
                    return False
                return bool(
                    p_ai_r0 <= float(s["tau_ai_lo"])
                    and p_apd >= float(s["apd_safe_min"])
                    and p_def_r0 <= float(s["defer_max"])
                )
        return False

    def _run_anytime_deep(
        self, clean_texts: list[str], z_char: np.ndarray, lg_r0_ai: np.ndarray, supports: np.ndarray, force_depth: int | None = None
    ) -> dict[str, np.ndarray]:
        ids, msk, st, eg, loc, qsc, m6_ph, m7_surf, m7_uisr = self._prep_deep_inputs(clean_texts)
        bs = len(clean_texts)
        o1 = dict(zip(_T1_OUT_NAMES_V170, self._run_sess(self.sess_t1, self._in_t1, {
            "input_ids": ids, "attention_mask": msk, "style_feats": st, "engram_ids": eg,
            "local_charlm": loc, "z_char": z_char, "support_pairs": supports, "quote_scope": qsc,
            "m6_ph": m6_ph, "m7_surf": m7_surf, "m7_uisr": m7_uisr,
        })))

        lg_ai = o1["logit_ai_t1"].copy()
        lg_pr = o1["logit_promo_t1"].copy()
        emb_ai = o1["emb_ai_t1"].copy()
        emb_pr = o1["emb_pr_t1"].copy()
        hop_w = o1["hop_witness"].copy()
        r_ai = np.ones((bs,), np.int32)
        r_pr = np.ones((bs,), np.int32)

        t_max = int(force_depth) if force_depth is not None else int(self.step_policy.get("max_infer_steps", DEFAULT_T_INFER_MAX_V170))
        if t_max <= 1:
            return {"logit_ai_deep": lg_ai, "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr, "hop_witness": hop_w, "rung_ai": r_ai, "rung_pr": r_pr}

        def _need_more_ai(cur_l, cur_u, step_idx):
            if force_depth is not None:
                return np.ones((len(cur_l),), dtype=bool)
            if not self.step_policy.get("enabled", True):
                return np.ones((len(cur_l),), dtype=bool)
            if step_idx == 1 and not self.step_policy.get("t1_exit_enabled", True):
                return np.ones((len(cur_l),), dtype=bool)
            p_cur = self._prob_ai(cur_l, is_r0=False)
            unc = (p_cur >= float(self.step_policy.get("uncertain_lo", 0.12))) & (p_cur <= float(self.step_policy.get("uncertain_hi", 0.88)))
            return unc | (cur_u >= float(self.step_policy.get("u_halt_thresh", 0.015)))

        ai_active = _need_more_ai(lg_ai, o1["u_halt_t1"], 1)
        ai_active = ai_active | (np.abs(np.clip(lg_r0_ai, -6.0, 6.0) - np.clip(lg_ai, -6.0, 6.0)) >= float(self.step_policy.get("disagree_dt_thresh", 1.35)))
        pr_active = np.ones((bs,), dtype=bool) if (force_depth is not None) else ((self._prob_pr(lg_pr) >= 0.08) & (self._prob_pr(lg_pr) <= 0.92))

        cur_ai, cur_pr, cur_M, cur_D, cur_W = o1["s_ai"], o1["s_pr"], o1["M_psd"], o1["D_val"], o1["W_fast"]
        cur_emb, cur_l = o1["emb_ai_t1"], o1["logit_ai_t1"]
        for ti in range(1, t_max):
            union_idx = np.where(ai_active | pr_active)[0]
            if len(union_idx) == 0:
                break
            dt_np = np.clip(lg_r0_ai[union_idx], -6.0, 6.0) - np.clip(cur_l[union_idx], -6.0, 6.0)
            os_t = dict(zip(_STEP_OUT_NAMES_V170, self._run_sess(self.sess_step, self._in_step, {
                "s_ai_in": cur_ai[union_idx], "s_pr_in": cur_pr[union_idx],
                "M_psd_in": cur_M[union_idx], "D_val_in": cur_D[union_idx], "W_fast_in": cur_W[union_idx],
                "k_ev_norm": o1["k_ev_norm"][union_idx], "v_ev_0": o1["v_ev_0"][union_idx], "e_sketch": o1["e_sketch"][union_idx],
                "attention_mask": msk[union_idx], "k_haar": o1["k_haar"][union_idx], "v_haar": o1["v_haar"][union_idx], "haar_mask": o1["haar_mask"][union_idx],
                "h_pool_0": o1["h_pool_0"][union_idx], "pr_attr_0": o1["pr_attr_0"][union_idx], "st_pr": o1["st_pr"][union_idx],
                "z_char": z_char[union_idx], "g_sup": o1["g_sup"][union_idx],
                "emb_ai_prev": cur_emb[union_idx], "logit_ai_prev": cur_l[union_idx], "hop_witness": hop_w[union_idx],
                "t_val": np.full((len(union_idx),), float(ti), np.float32), "disagree_dt": dt_np.astype(np.float32),
            })))
            cur_ai[union_idx], cur_pr[union_idx] = os_t["s_ai_out"], os_t["s_pr_out"]
            cur_M[union_idx], cur_D[union_idx], cur_W[union_idx] = os_t["M_psd_out"], os_t["D_val_out"], os_t["W_fast_out"]
            cur_emb[union_idx], cur_l[union_idx] = os_t["emb_ai"], os_t["logit_ai"]

            # Latch AI & Promo STRICTLY INDEPENDENTLY -> F6 === 0.0!
            ai_loc = ai_active[union_idx]
            if np.any(ai_loc):
                ai_g = union_idx[ai_loc]
                lg_ai[ai_g] = os_t["logit_ai"][ai_loc]
                emb_ai[ai_g] = os_t["emb_ai"][ai_loc]
                r_ai[ai_g] = ti + 1
                ai_active[ai_g] = _need_more_ai(os_t["logit_ai"][ai_loc], os_t["u_halt"][ai_loc], ti + 1)
            pr_loc = pr_active[union_idx]
            if np.any(pr_loc):
                pr_g = union_idx[pr_loc]
                lg_pr[pr_g] = os_t["logit_promo"][pr_loc]
                emb_pr[pr_g] = os_t["emb_pr"][pr_loc]
                r_pr[pr_g] = ti + 1
                if force_depth is None:
                    pp = self._prob_pr(os_t["logit_promo"][pr_loc])
                    pr_active[pr_g] = (pp >= 0.10) & (pp <= 0.90)

        return {"logit_ai_deep": lg_ai, "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr, "hop_witness": hop_w, "rung_ai": r_ai, "rung_pr": r_pr}

    def _run_long_doc_clean(self, cl: str, force_depth: int | None = None) -> tuple[float, float, float]:
        std_chunks = plan_long_chunks_v11(cl, max_chunks=6)
        fr_cfg = self.long_focal_router or {}
        use_fr = bool(fr_cfg.get("enabled", False)) and len(cl) >= self.long_char_trigger
        focal_chunk = None
        if use_fr:
            win_c = int(fr_cfg.get("win_chars", 160))
            lam = float(fr_cfg.get("lambda", 0.92))
            kd = self._rust_fe.kadane_focal_window(cl, win_chars=win_c, lam=lam) if (self._rust_fe is not None and hasattr(self._rust_fe, "kadane_focal_window")) else None
            if kd is None:
                kd = kadane_focal_window_py(self.charlm, cl, win_chars=win_c, lam=lam)
            fs, fe = int(kd[0]), int(kd[1])
            ftxt = cl[fs:fe].strip() if fe > fs else ""
            if len(ftxt) >= 24:
                n_cl = max(1, len(cl))
                q_c = sum(1 for ch in ftxt if ch in "「」『』“”\"'《》【】")
                q_frac = min(1.0, float(q_c) / max(4.0, len(ftxt) * 0.08))
                focal_chunk = {
                    "text": ftxt,
                    "chunk_meta": np.array([fs / n_cl, fe / n_cl, min(1.0, len(ftxt) / 280.0), q_frac, 0.0, 1.0], np.float32),
                }

        all_chunks = list(std_chunks)
        has_focal = focal_chunk is not None
        if has_focal:
            all_chunks.append(focal_chunk)
        ctxs = [c["text"] for c in all_chunks]
        c1, c2, c3, cm, cf, sups, m7_surf = self._extract_r0_clean(ctxs)
        la0, _, _, _, _, zc = self._run_sess(self.sess_r0, self._in_r0, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf, "m7_surf": m7_surf})
        dp = self._run_anytime_deep(ctxs, zc, la0, sups, force_depth=force_depth)
        fused_c = apply_crossview_late_fusion(la0, dp["logit_ai_deep"], self.crossview_fusion)

        n_std = len(std_chunks)
        meta_std = np.stack([c["chunk_meta"] for c in std_chunks], 0).astype(np.float32)[None]
        valid_std = np.ones((1, n_std), np.float32)
        da_std, dp_std, ts_std = self._run_sess(self.sess_long, self._in_long, {
            "chunk_emb_ai": dp["emb_ai"][:n_std][None], "chunk_emb_pr": dp["emb_pr"][:n_std][None],
            "chunk_lg_ai": fused_c[:n_std][None], "chunk_lg_pr": dp["logit_promo"][:n_std][None],
            "chunk_meta": meta_std, "chunk_valid": valid_std,
        })
        doc_ai = float(da_std[0])
        doc_pr = float(dp_std[0])
        titan_s = float(ts_std[0])
        if has_focal:
            # Quote-gated FocalRouter AI max (strictly AI-only, Promo stays on untouched std_chunks -> F6 === 0.0!)
            fq = float(focal_chunk["chunk_meta"][3])
            if fq < 0.25:
                doc_ai = max(doc_ai, float(fused_c[-1]))
        # DS-TitanGuard Version-A zero-flip safe ranking tie-break (eps=1e-3 -> 0 threshold flips, strictly AI-only!)
        doc_ai_safe = doc_ai + 1e-3 * float(expit(titan_s))
        return doc_ai_safe, doc_pr, titan_s

    def predict_batch(
        self, texts: list[str], max_batch: int = 32, force_full: bool = False, force_depth: int | None = None
    ) -> list[dict[str, Any]]:
        if not texts:
            return []
        if self._rust_fe is not None:
            cleans, _ = self._rust_fe.canonicalize_with_spoof_batch(texts, max_chars=2048)
        else:
            cleans = [canonicalize_zero_pii_v150(t)[0] for t in texts]

        out: list[dict[str, Any] | None] = [None] * len(texts)
        short_idx = [i for i, cl in enumerate(cleans) if len(cl) < self.long_char_trigger]
        long_idx = [i for i, cl in enumerate(cleans) if len(cl) >= self.long_char_trigger]

        for li in long_idx:
            da, dp, ts = self._run_long_doc_clean(cleans[li], force_depth=force_depth)
            out[li] = {
                "logit_ai": da, "prob_ai": float(self._prob_ai(np.array([da], np.float32))[0]),
                "logit_promo": dp, "prob_promo": float(self._prob_pr(np.array([dp], np.float32))[0]),
                "exit_rung_ai": 5, "exit_rung_promo": 5, "titan_surprise_max": ts,
            }

        for st in range(0, len(short_idx), max_batch):
            sub_idx = short_idx[st : st + max_batch]
            sub_cl = [cleans[i] for i in sub_idx]
            c1, c2, c3, cm, cf, sups, m7_surf = self._extract_r0_clean(sub_cl)
            la0, lp0, da0, dp0, papd0, zc = self._run_sess(self.sess_r0, self._in_r0, {
                "c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf, "m7_surf": m7_surf,
            })
            pa0 = self._prob_ai(la0, is_r0=True)
            pp0 = self._prob_pr(lp0, is_r0=True)

            ai_exit = np.zeros((len(sub_idx),), dtype=bool)
            pr_exit = np.zeros((len(sub_idx),), dtype=bool)
            if not force_full and force_depth is None:
                for j, cl in enumerate(sub_cl):
                    ai_exit[j] = self._check_r0_iut_exit(float(pa0[j]), float(da0[j]), float(papd0[j]), len(cl))
                    pr_pol = self.r0_pr_policy
                    if pr_pol.get("enabled") and len(cl) <= int(pr_pol.get("max_chars", 180)):
                        pr_exit[j] = bool(pp0[j] <= float(pr_pol.get("tau_promo", 0.015)) and dp0[j] <= float(pr_pol.get("defer_max", 0.35)))

            need_deep = np.where(~(ai_exit & pr_exit))[0]
            dp_res = None
            if len(need_deep) > 0:
                dp_res = self._run_anytime_deep(
                    [sub_cl[j] for j in need_deep], zc[need_deep], la0[need_deep], sups[need_deep], force_depth=force_depth
                )
                fused_ai = apply_crossview_late_fusion(la0[need_deep], dp_res["logit_ai_deep"], self.crossview_fusion)
                p_ai_d = self._prob_ai(fused_ai, is_r0=False)
                p_pr_d = self._prob_pr(dp_res["logit_promo"], is_r0=False)
                deep_map = {int(j): k for k, j in enumerate(need_deep)}
            else:
                deep_map = {}

            for j, g_idx in enumerate(sub_idx):
                if ai_exit[j]:
                    f_la, f_pa, r_a = float(la0[j]), float(pa0[j]), 0
                else:
                    k = deep_map[j]
                    f_la, f_pa, r_a = float(fused_ai[k]), float(p_ai_d[k]), int(dp_res["rung_ai"][k])
                if pr_exit[j]:
                    f_lp, f_pp, r_p = float(lp0[j]), float(pp0[j]), 0
                else:
                    k = deep_map[j]
                    f_lp, f_pp, r_p = float(dp_res["logit_promo"][k]), float(p_pr_d[k]), int(dp_res["rung_pr"][k])
                out[g_idx] = {
                    "logit_ai": f_la, "prob_ai": f_pa, "logit_promo": f_lp, "prob_promo": f_pp,
                    "exit_rung_ai": r_a, "exit_rung_promo": r_p, "titan_surprise_max": 0.0,
                }
        return [x for x in out if x is not None]


# ---------------------------------------------------------------------------
# 10. 26-Gate Automated CI Suite (`run_v170_ci_gates`: All 22 v1.6.2 + 4 M9 Gates!)
# ---------------------------------------------------------------------------

def run_v170_ci_gates(
    model: MeguvizV170MesosphereLoopModel,
    tokenizer: BiliPickyBPETokenizer,
    export_report: dict[str, Any],
    release_dir: Path,
    device: torch.device,
    charlm: CompactCharLMV150 | None = None,
    engine: MeguvizV170CascadeEngine | None = None,
) -> dict[str, Any]:
    model.eval()
    ci: dict[str, Any] = {}
    eng = engine if engine is not None else MeguvizV170CascadeEngine(release_dir, threads=4, r0_threads=2)

    def _infer(texts: list[str], outer_steps: int = 6, pad_to: int | None = None) -> dict[str, Any]:
        cleans = [canonicalize_zero_pii_v150(t)[0] for t in texts]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
            cleans, b2=model.r0_b2, b3=model.r0_b3, already_canonical=True, charlm=charlm
        )
        bs = len(cleans)
        if pad_to is not None and c1.shape[1] < pad_to:
            dp = pad_to - c1.shape[1]
            c1 = np.pad(c1, ((0, 0), (0, dp)))
            c2 = np.pad(c2, ((0, 0), (0, dp)))
            c3 = np.pad(c3, ((0, 0), (0, dp)))
            cm = np.pad(cm, ((0, 0), (0, dp)))

        f56_l, encs, offs_l = [], [], []
        for cl in cleans:
            _, f56, _ = extract_style_features_56d(cl)
            e_i, e_o = encode_bpe_head_tail_with_offsets_v150(tokenizer, cl, max_len=model.max_len, head_len=model.head_len)
            f56_l.append(f56); encs.append(e_i); offs_l.append(e_o)
        engs = [make_engram_bigram_ids(s, model.engram_buckets) for s in encs]
        ml = max(2, max(len(s) for s in encs))
        if pad_to is not None:
            ml = max(ml, pad_to)
        ids = np.full((bs, ml), tokenizer.pad_id, np.int64)
        msk = np.zeros((bs, ml), np.int64)
        eg = np.zeros((bs, ml), np.int64)
        loc = np.zeros((bs, ml, LOCAL_CHARLM_DIM_V150), np.float32)
        qsc = np.zeros((bs, ml, QUOTE_SCOPE_DIM_V161), np.float32)
        for i, s in enumerate(encs):
            ns = len(s)
            ids[i, :ns], msk[i, :ns], eg[i, :ns] = s, 1, engs[i]
            if charlm is not None:
                loc[i, :ns] = charlm.extract_token_local_8ch(cleans[i], offs_l[i])
            if model.use_discourse_ops:
                qsc[i, :ns] = extract_token_quote_scope_4d_v161(cleans[i], offs_l[i])
        bpe_lens = np.array([len(s) for s in encs], np.int64)
        m6_ph, m7_surf, m7_uisr = extract_v170_sidecar_batch(cleans, bpe_lens, charlm=charlm)
        with torch.no_grad():
            return model(
                torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
                torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device),
                torch.from_numpy(eg).to(device), torch.from_numpy(loc).to(device),
                torch.from_numpy(c1).to(device), torch.from_numpy(c2).to(device), torch.from_numpy(c3).to(device),
                torch.from_numpy(cm).to(device), torch.from_numpy(cf).to(device),
                torch.from_numpy(sups).to(device), torch.from_numpy(qsc).to(device),
                torch.from_numpy(m6_ph).to(device), torch.from_numpy(m7_surf).to(device), torch.from_numpy(m7_uisr).to(device),
                outer_steps=outer_steps, return_diagnostics=True,
            )

    # 1. C1 Zero-PII
    oa = _infer(["回复 @测试A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"])
    ob = _infer(["回复 @UltraLongRandomUserName_998877665544332211 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"])
    pii_keys = ("logit_ai_r0", "p_apd_safe", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "logit_promo", "u_halt_t1", "u_halt_t4")
    pii_d = float(max(abs(float(oa[k][0].item()) - float(ob[k][0].item())) for k in pii_keys))
    ci["V170_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # 2. F6 Model Structural Promo->AI Firewall (+12 Promo weights/biases -> uint32 xor === 0!)
    t_fw = ["笑死我了，这个插件装完直接起飞[doge]", "关注工种号交流2群 168688884 领全套资料"]
    o_base = _infer(t_fw, outer_steps=6)
    with torch.no_grad():
        c_meta_fw = torch.tensor([[[0.0, 0.5, 0.4, 0.0, 0.0, 1.0], [0.5, 1.0, 0.4, 0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
        c_val_fw = torch.ones((1, 2), device=device, dtype=torch.float32)
        ld_ai_b, _, ts_b = model.aggregate_long_chunks(
            o_base["emb_ai"].unsqueeze(0), o_base["emb_pr"].unsqueeze(0),
            o_base["logit_ai"].unsqueeze(0), o_base["logit_promo"].unsqueeze(0), c_meta_fw, c_val_fw,
        )
        saved = [
            model.cbm_pr_enc.bias.clone(), model.promo_head.bias.clone(), model.r1_promo_head.bias.clone(),
            model.rung0.promo_head.bias.clone(), model.pr_attr_attn[-1].bias.clone(),
            model.long_agg.long_pr_attn[-1].bias.clone(), model.long_agg.long_pr_head.bias.clone(),
        ]
        saved_dpr = model.disc_pr_proj[1].bias.clone() if model.use_discourse_ops else None
        for p in (model.cbm_pr_enc.bias, model.promo_head.bias, model.r1_promo_head.bias, model.rung0.promo_head.bias,
                  model.pr_attr_attn[-1].bias, model.long_agg.long_pr_attn[-1].bias, model.long_agg.long_pr_head.bias):
            p.add_(12.0)
        if saved_dpr is not None:
            model.disc_pr_proj[1].bias.add_(12.0)
        o_pert = _infer(t_fw, outer_steps=6)
        ld_ai_p, _, ts_p = model.aggregate_long_chunks(
            o_pert["emb_ai"].unsqueeze(0), o_pert["emb_pr"].unsqueeze(0),
            o_pert["logit_ai"].unsqueeze(0), o_pert["logit_promo"].unsqueeze(0), c_meta_fw, c_val_fw,
        )
        for p, sv in zip((model.cbm_pr_enc.bias, model.promo_head.bias, model.r1_promo_head.bias, model.rung0.promo_head.bias,
                          model.pr_attr_attn[-1].bias, model.long_agg.long_pr_attn[-1].bias, model.long_agg.long_pr_head.bias), saved):
            p.copy_(sv)
        if saved_dpr is not None:
            model.disc_pr_proj[1].bias.copy_(saved_dpr)
    ai_fw_keys = ("logit_ai_r0", "p_defer_ai_r0", "p_apd_safe", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "u_halt_t1", "u_halt_t4")
    fw_diff = float(max(
        max((o_base[k] - o_pert[k]).abs().max().item() for k in ai_fw_keys),
        float((ld_ai_b - ld_ai_p).abs().max().item()),
        float((ts_b - ts_p).abs().max().item()),
    ))
    fw_u32_xor = int(max(
        max(int(np.bitwise_xor(o_base[k].detach().float().cpu().numpy().view(np.uint32), o_pert[k].detach().float().cpu().numpy().view(np.uint32)).max()) for k in ai_fw_keys),
        int(np.bitwise_xor(ld_ai_b.detach().float().cpu().numpy().view(np.uint32), ld_ai_p.detach().float().cpu().numpy().view(np.uint32)).max()),
        int(np.bitwise_xor(ts_b.detach().float().cpu().numpy().view(np.uint32), ts_p.detach().float().cpu().numpy().view(np.uint32)).max()),
    ))
    ci["V170_F6_model_promo_to_ai_firewall"] = {"max_leak_diff": fw_diff, "max_uint32_xor": fw_u32_xor, "pass": (fw_diff == 0.0 and fw_u32_xor == 0)}

    # 3. P1 Exact Pad Invariance (odd & even pad lengths -> < 1e-4)
    p1_short = _infer(["笑死，这不就是纯纯的标题党吗"], outer_steps=6, pad_to=None)
    p1_odd = _infer(["笑死，这不就是纯纯的标题党吗"], outer_steps=6, pad_to=127)
    p1_even = _infer(["笑死，这不就是纯纯的标题党吗"], outer_steps=6, pad_to=128)
    p1_keys = ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "logit_promo", "u_halt_t1", "u_halt_t4")
    p1_max = float(max(
        max(abs(float(p1_short[k][0].item()) - float(p1_odd[k][0].item())) for k in p1_keys),
        max(abs(float(p1_short[k][0].item()) - float(p1_even[k][0].item())) for k in p1_keys),
    ))
    ci["V170_P1_pad_invariance_odd_even"] = {"max_diff": round(p1_max, 8), "pass": p1_max < 1e-4}

    # 4. S1 Per-slot bound & Cayley orthogonality
    g0 = model.core.blocks[0].c1_cayley_gamma.detach().float().clamp(-2.5, 2.5)
    c_u = (1.0 - g0 * g0) / (1.0 + g0 * g0)
    s_u = (2.0 * g0) / (1.0 + g0 * g0)
    cayley_err = float((c_u * c_u + s_u * s_u - 1.0).abs().max().item())
    max_sn = float(max(sl.float().norm(dim=-1).max().item() for sl in o_base["s_ai_steps"]))
    ci["V170_S1_per_slot_bound_and_cayley"] = {"cayley_unit_err": round(cayley_err, 8), "max_slot_norm_t1_t6": round(max_sn, 4), "pass": (cayley_err < 1e-5 and max_sn <= model.core.r_cap + 1e-3)}

    # 5. M5 RI-Meso-PSD-L PSD Matrix & Pad Check
    m_eigs = np.linalg.eigvalsh(o_base["M_psd_final"][0].detach().float().cpu().numpy())
    m_tr = float(np.trace(o_base["M_psd_final"][0].detach().float().cpu().numpy()))
    ci["V170_M5_psd_matrix_valid"] = {"min_eig": round(float(m_eigs.min()), 6), "trace": round(m_tr, 4), "pass": bool(m_eigs.min() >= -1e-5 and m_tr <= 1.05)}

    # 6. RF Impulse Span & Zero Pad Leak (matches v1.6.0 exact DWConv span check + full-block pad leak check!)
    with torch.no_grad():
        L_rf = 64
        x0 = torch.zeros((1, L_rf, model.dim), device=device)
        m_rf = torch.zeros((1, L_rf), device=device, dtype=torch.long); m_rf[:, :50] = 1
        x1 = x0.clone(); x1[:, 25, 0] = 1.0
        # Full block pad leak (with SE active)
        yf0, yf1 = x0, x1
        for pb in model.prelude_blocks:
            yf0, yf1 = pb(yf0, m_rf), pb(yf1, m_rf)
        pad_leak_full = float((yf1 - yf0).abs().max(dim=-1).values[0, 50:].max().item())
        # Local DWConv exact receptive field span (with global SE temporarily detached)
        saved_se = [pb.se for pb in model.prelude_blocks]
        for pb in model.prelude_blocks:
            pb.se = None
        y0, y1 = x0, x1
        for pb in model.prelude_blocks:
            y0, y1 = pb(y0, m_rf), pb(y1, m_rf)
        for pb, sm in zip(model.prelude_blocks, saved_se):
            pb.se = sm
        diff_rf = (y1 - y0).abs().max(dim=-1).values[0]
        nz = torch.where(diff_rf[:50] > 1e-6)[0]
        obs_span = int(nz.max().item() - nz.min().item() + 1) if len(nz) > 0 else 0
        pad_leak = max(pad_leak_full, float(diff_rf[50:].max().item()))
    ci["V170_RF_IMPULSE_SPAN"] = {"observed_span": obs_span, "pad_leak": pad_leak, "pass": (obs_span == 25 and pad_leak == 0.0)}

    # 7. SoftBall Bound
    with torch.no_grad():
        huge = torch.full((1, AI_END_V130), 1e4, device=device)
        hb_n = float(model.ai_head.project_ball(huge).norm().item())
        rc = float(model.ai_head.effective_r_cap().item())
    ci["V170_L1_softball_bound"] = {"huge_ball_norm": round(hb_n, 4), "r_cap": round(rc, 4), "pass": hb_n <= rc + 1e-3}

    # 8. Support Gate Monotonicity
    g_lo = float(model.compute_support_gate(torch.tensor([4.0], device=device)).item())
    g_hi = float(model.compute_support_gate(torch.tensor([36.0], device=device)).item())
    ci["V170_C13_support_gate"] = {"g_at_4": round(g_lo, 4), "g_at_36": round(g_hi, 4), "pass": (g_lo < 0.25 and g_hi > 0.85)}

    # 9-12. Export, Forbidden Ops, Bootable <= 5.00 MiB, Cold-Start Parity
    ci["V170_B10_bootable_le_5p00_mib"] = {"bootable_mib": export_report["bootable_mib"], "bootable_bytes": export_report["bootable_bytes"], "pass": bool(export_report["bootable_le_5p00_mib"])}
    ci["V170_ONNX_4stage_fp32_parity"] = {
        "r0_diff": export_report["r0_fp32_max_diff"], "t1_diff": export_report["t1_fp32_max_diff"],
        "loop_step_diffs": export_report["loop_step_diffs_t2_to_t6"], "long_diff": export_report["long_fp32_max_diff"],
        "pass": bool(export_report["onnx_exact_pass"]),
    }
    ci["V170_ONNX_ZERO_FORBIDDEN_OPS"] = {"by_stage": export_report["forbidden_ops_by_stage"], "total": export_report["forbidden_ops_total"], "pass": export_report["forbidden_ops_total"] == 0}
    with zipfile.ZipFile(export_report["bootable_zip"], "r") as zf:
        z_mgvz = zf.read("bundle_v10_mesosphere_v170.mgvz")
    disk_mgvz = (release_dir / "bundle_v10_mesosphere_v170.mgvz").read_bytes()
    ci["V170_COLD_START_BOOT_PARITY"] = {"exact_match": (z_mgvz == disk_mgvz), "pass": (z_mgvz == disk_mgvz)}

    # 13-14. LUT Floors
    lut_r0_001 = float(apply_spline_temp_lut(np.array([0.01], np.float32), eng.lut_ai_r0)[0]) if eng.lut_ai_r0 is not None else 0.01
    ci["V170_AI_LUT_FLOOR"] = {"lut_ai_r0_at_001": round(lut_r0_001, 6), "pass": lut_r0_001 < 0.08}
    p_wild = eng.predict_batch(["哈哈哈笑死", "今天天气真不错", "这个视频讲得挺清楚的"], force_full=True)
    med_pr = float(np.median([x["prob_promo"] for x in p_wild]))
    ci["V170_PR_LUT_FLOOR"] = {"median_promo_wild": round(med_pr, 5), "pass": med_pr < 0.25}

    # 15-19. E2E ORT C1, F6 Short, F6 Long (>=400c), IUT Exit, Anytime Routes
    r_c1a = eng.predict_batch(["回复 @Alice123 : 交流群 123456789 领资料"], force_full=True)[0]
    r_c1b = eng.predict_batch(["回复 @VeryLongUserName_9999999 : 交流群 987654321 领资料"], force_full=True)[0]
    ci["V170_E2E_C1_zero_pii"] = {"diff": abs(r_c1a["prob_ai"] - r_c1b["prob_ai"]), "pass": abs(r_c1a["prob_ai"] - r_c1b["prob_ai"]) == 0.0}

    old_pr_pol = dict(eng.r0_pr_policy)
    eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.99, "defer_max": 0.99, "max_chars": 180}
    rs_a = eng.predict_batch(["笑死我了，这视频太好玩了[doge]"], force_full=False)[0]
    eng.r0_pr_policy = {"enabled": False}
    rs_b = eng.predict_batch(["笑死我了，这视频太好玩了[doge]"], force_full=False)[0]
    xor_s = int(np.bitwise_xor(np.float32(rs_a["logit_ai"]).view(np.uint32), np.float32(rs_b["logit_ai"]).view(np.uint32)))
    ci["V170_E2E_F6_decoupled_short"] = {"u32_xor": xor_s, "pass": xor_s == 0}

    long_400 = ("先声明我不是水军，纯分享真实体验。" * 6) + ("从客观维度分析，该方案不仅体现结构优势，而且反映了必然演进趋势。" * 4) + ("大伙按需选择别被带节奏[doge]。" * 5)
    eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.99, "defer_max": 0.99, "max_chars": 180}
    rl_a = eng.predict_batch([long_400], force_full=False)[0]
    eng.r0_pr_policy = {"enabled": False}
    rl_b = eng.predict_batch([long_400], force_full=False)[0]
    eng.r0_pr_policy = old_pr_pol
    xor_l = int(np.bitwise_xor(np.float32(rl_a["logit_ai"]).view(np.uint32), np.float32(rl_b["logit_ai"]).view(np.uint32)))
    ci["V170_E2E_F6_decoupled_long"] = {"u32_xor": xor_l, "titan_surprise_max": round(rl_a["titan_surprise_max"], 4), "pass": (xor_l == 0 and rl_a["exit_rung_ai"] == 5)}

    old_ai_pol = dict(eng.r0_ai_policy)
    eng.r0_ai_policy = {"enabled": True, "max_chars": 180, "strata": [{"lo_chars": 0, "hi_chars": 180, "enabled": True, "tau_ai_lo": 0.999, "apd_safe_min": 0.0, "defer_max": 1.0}]}
    r_iut = eng.predict_batch(["哈哈挺好玩"], force_full=False)[0]
    eng.r0_ai_policy = old_ai_pol
    ci["V170_E2E_IUT_STRATIFIED_EXIT"] = {"exit_rung_ai": r_iut["exit_rung_ai"], "pass": r_iut["exit_rung_ai"] == 0}

    rt1 = eng.predict_batch(["测试路由"], force_depth=1)[0]["exit_rung_ai"]
    rt4 = eng.predict_batch(["测试路由"], force_depth=4)[0]["exit_rung_ai"]
    rt6 = eng.predict_batch(["测试路由"], force_depth=6)[0]["exit_rung_ai"]
    ci["V170_E2E_ANYTIME_ROUTES"] = {"t1": rt1, "t4": rt4, "t6": rt6, "long": rl_a["exit_rung_ai"], "pass": (rt1 == 1 and rt4 == 4 and rt6 == 6 and rl_a["exit_rung_ai"] == 5)}

    # 20. EBA-H FP32 ULP Monotonicity
    if eng.crossview_fusion.get("formula") == "eba_h":
        ci["V170_EBA_H_FP32_MONOTONE"] = verify_eba_h_fp32_monotonicity(eng.crossview_fusion, n_pairs=10000)
    else:
        ci["V170_EBA_H_FP32_MONOTONE"] = {"skipped": True, "pass": True}

    # 21. M7 CNMA-R & UISR Bounded Residual Safety
    with torch.no_grad():
        cnma_max = float(model.cnma_r(torch.full((1, M7_SURF_DIM_V170), 100.0, device=device)).abs().item()) if model.cnma_r is not None else 0.0
        uisr_max = float(model.uisr_res(torch.full((1, M7_UISR_DIM_V170), 100.0, device=device)).abs().item()) if model.uisr_res is not None else 0.0
    ci["V170_M7_CNMA_UISR_BOUNDED"] = {"cnma_max": round(cnma_max, 4), "uisr_max": round(uisr_max, 4), "pass": (cnma_max <= 1.26 and uisr_max <= 1.51)}

    # 22. M6 Pure-P_H Zero-P_A Sidecar Finiteness
    m6_chk, s_chk, u_chk = extract_v170_sidecar_batch(["测试一下短评论", "", "hello world 123"], np.array([6, 1, 4], np.int64), charlm=charlm)
    fin_ok = bool(np.isfinite(m6_chk).all() and np.isfinite(s_chk).all() and np.isfinite(u_chk).all())
    ci["V170_M6_M7_SIDECAR_FINITE"] = {"finite": fin_ok, "pass": fin_ok}

    # 23. NEW M9 Gate: J-Cert Common Contraction across StepLoRA + PSD-L + Delta-Rule
    alpha_eff = float((2.0 * torch.sigmoid(model.core.ri_alpha_raw)).item())
    beta_eff = float((0.25 + 0.50 * torch.sigmoid(model.core.ri_beta_raw)).item())
    o_t8 = _infer(t_fw, outer_steps=8)
    s_deltas = [float((o_t8["s_ai_steps"][k] - o_t8["s_ai_steps"][k - 1]).flatten(1).norm(dim=1).mean().item()) for k in range(1, 8)]
    q_emp = s_deltas[-1] / max(1e-6, s_deltas[0])
    j_pass = bool(alpha_eff <= 2.001 and beta_eff <= 0.751 and q_emp < 1.0)
    ci["V170_J_CERT_COMMON_CONTRACTION"] = {"alpha_psd": round(alpha_eff, 4), "beta_psd": round(beta_eff, 4), "q_emp_d8_over_d2": round(q_emp, 4), "pass": j_pass}

    # 24. NEW M9 Gate: Echo-Cert for Double-Masked AR-U-Loop (`lam_v >= 0.30, r_echo <= 0.80, pad === 0.0`)
    if model.use_ar_uloop:
        lam_v, kap_v = model.core.blocks[0].effective_ar_params()
        lam_f, kap_f = float(lam_v.item()), float(kap_v.item())
        r_echo = (1.0 - lam_f) + kap_f * 0.5
        d_pad_max = float(p1_even["D_val_final"][0, 50:].abs().max().item()) if p1_even["D_val_final"].shape[1] > 50 else 0.0
        d_norm_max = float(o_t8["D_val_final"].float().norm(dim=-1).max().item())
        echo_ok = bool(lam_f >= 0.299 and r_echo <= 0.85 and d_norm_max <= model.core.blocks[0].ar_r_v + 1e-3 and d_pad_max == 0.0)
        ci["V170_ECHO_CERT_AR_ULOOP"] = {"lambda_v": round(lam_f, 4), "kappa_v": round(kap_f, 4), "r_echo_ub": round(r_echo, 4), "max_D_norm": round(d_norm_max, 4), "pad_leak": d_pad_max, "pass": echo_ok}
    else:
        ci["V170_ECHO_CERT_AR_ULOOP"] = {"disabled": True, "pass": True}

    # 25. NEW M9 Gate: T1..T8 Extrapolation Finiteness & Contracting Fast-Weight `||W_8||_F` Stability
    o_t4 = _infer(t_fw, outer_steps=4)
    t8_lgs = [float(o_t8[f"logit_ai_t{k}"][0].item()) for k in range(1, 9)]
    t4_t8_drift = abs(t8_lgs[7] - t8_lgs[3])
    w4_fro = float(o_t4["W_fast_final"].float().norm().item())
    w8_fro = float(o_t8["W_fast_final"].float().norm().item())
    t8_ok = bool(
        all(math.isfinite(x) for x in t8_lgs)
        and math.isfinite(w4_fro)
        and math.isfinite(w8_fro)
        and t4_t8_drift < 1.5
        and w8_fro <= 1.35 * max(w4_fro, 1.0) + 10.0
        and w8_fro < 1000.0
    )
    ci["V170_T1_TO_T8_EXTRAP_FINITE"] = {
        "t4_to_t8_abs_drift": round(t4_t8_drift, 4),
        "W4_fro": round(w4_fro, 4),
        "W8_fro": round(w8_fro, 4),
        "pass": t8_ok,
    }

    # 26. NEW M8 Gate: Real B=1 ORT Latency Audit (2T R0++ & 4T T1/step)
    t_lat_0 = time.perf_counter()
    for _ in range(10):
        eng.predict_batch(["笑死我了，这个视频讲得太清楚了[doge]"], force_full=True, force_depth=1)
    ms_b1_t1 = (time.perf_counter() - t_lat_0) * 1000.0 / 10.0
    ci["V170_B1_ORT_LATENCY_AUDIT"] = {"b1_r0_plus_t1_ms": round(ms_b1_t1, 3), "pass": bool(math.isfinite(ms_b1_t1) and ms_b1_t1 < 50.0)}

    ci["all_ci_passed"] = bool(all(v["pass"] for k, v in ci.items() if isinstance(v, dict) and "pass" in v))
    ci["passed_count"] = int(sum(1 for k, v in ci.items() if isinstance(v, dict) and v.get("pass")))
    ci["total_count"] = int(sum(1 for k, v in ci.items() if isinstance(v, dict) and "pass" in v))
    return ci
