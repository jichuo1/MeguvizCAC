"""MeguvizCAC v1.6.0-StratosphereLoop: Anytime Input-Injected Recurrent Engine (<=5.00 MiB Bootable, Pure-FP32 Opset-17).

Implements the post-External-GPT-audit locked architectural RFC (`v1.6-A1` primary candidate + unconfounded A0..A5 switches):
  Pillar 1 (StratosphereLoop Core Dynamics — Verifiable 6-Step Contract):
    - Input-injected per-slot norm-bounded residual recurrence:
        s_{t+1} = Pi_R(s_t + h_t * Delta_theta(AdaLN(s_t; phi(t)), e_x))
      * Causal step size `h_t = (0.65 * sigmoid(step_h)) / sqrt(t + 1)` depends ONLY on executed step `t` (never future total `T`).
      * Strictly PER-SLOT L2 norm projection `Pi_R` bounds state norm (empirical 6-step verification of saturation,
        joint `(slots, coverage, bus)` directional gain, period-2 ratio `||s_t - s_{t-2}|| / ||s_t - s_{t-1}||`, and
        `T3-state -> T4-readout` pure-bias control; no unproven global contraction claim).
      * Explicit `use_step_phi` toggle (`False` in A0/A1/A3/A4/A5, `True` in A2) cleanly isolates continuous Fourier `phi(t)`.
      * Single unified SwiGLU FFN per sub-block (drops redundant `_fast_inner_step` FFN), retaining 64-d AI-only slow-bus
        Cayley peer coupling + block-diagonal `diag(M_3x3, 1)` 10-iter Sinkhorn mHC.
      * Single shared depth-conditioned readout head (drops detached T3/T4 band-aid MLPs & v1.5.0 R1/T1/EBT6 blend).
      * Independent F6 dataflow isolation: AI slow/sem slots, AI coverage, norms, routing & calibration NEVER read Promo state.
  Pillar 2 (Zero-Laziness Cross-View Arbitration `R0 x T1..T6`):
    - Deep core NEVER consumes `logit_ai_r0` AND Slot-9 `d_t` injection is REMOVED by default (`use_slot9_dt=False`) so
      `b_{t+1}` has ZERO training or inference dependence on `R0`!
    - Strictly monotone OOF late fusion: `f = bias + w_a * a + w_b * b` (`w_a >= 0, w_b >= 0`) — eliminates `a*b` top-FP trap.
    - Cross-view disagreement lives strictly in the exit gate via clipped signed calibrated logit diff `clip(a,-6,6)-clip(b,-6,6)`
      (never unbounded probit `Phi^-1(p)` whose derivative blows up to 297..2526 in the tails).
  Pillar 3 (Pure-FP32 CPU Speedup — `v1.6-A1` Primary Geometry, Frozen R0/Engram in Round 1):
    - Primary candidate `v160_a1`: 1L `O(L)` Pad-Exact Prelude (3 parallel dilated DWConv branches `k=7, d={1,2,4}` -> exact
      local span `25` tokens for `1L` / `49` for `2L` with edge dilation holes + masked global SE + SwiGLU, `P1 === 0.0`),
      `H=320` (`5x64` SA heads), `FFN=512`, `D=2`, `14` slots (`4 slow + 6 AI + 4 Promo`), compressed `d_kv=128` (`4x32` CA
      heads, verified ZERO `Tile` broadcast ops in exported ONNX!).
    - Round 1 freezes `R0` & BPE Engram table sizes across A0..A5 (`expand_tables=False` default) to isolate dynamics & geometry.
    - 4-Stage Modular Pure-FP32 ONNX (`r0`, `prelude_t1`, ONE reusable `loop_step(t)`, `dynaseam`).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from scipy.special import expit, logit as sp_logit
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
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
    ENGRAM_BUCKETS_V130,
    PR_EMB_DIM_V130,
    R0_B1_V130,
    R0_B2_V130,
    R0_B3_V130,
    R0_D1_V130,
    R0_D2_V130,
    R0_D3_V130,
    R0_HEAD_CHARS_V130,
    R0_HID_V130,
    R0_MAX_CHARS_V130,
    R0_ZCHAR_DIM_V130,
    SHARED_END_V130,
    TRIGRAM_BUCKETS_V130,
    PadExactConvBlockV130,
    QuantEmbeddingV130,
    QuantLinearV130,
    _pack_w4_tensor,
    _pack_w8_tensor,
    _unpack_w4_tensor,
    _unpack_w8_tensor,
    set_v130_qat_mode,
)
from .v140_paradise_apex import (
    CHARLM_FEAT_DIM_V140,
    HEAD_LEN_V140,
    MAX_LEN_V140,
    R0_FEAT_DIM_V140,
    R0_STRUCT_DIM_V140,
    count_distinct_support_pairs_v140,
    extract_apex_char_batch_v140,
)
from .v150_heaven_loop import (
    COV_LOGIT_CAP_V150,
    COV_MAX_V150,
    LOCAL_CHARLM_DIM_V150,
    MHC_AI_STREAMS_V150,
    MHC_SINKHORN_ITERS_V150,
    CompactCharLMV150,
    canonicalize_zero_pii_v150,
    encode_bpe_head_tail_with_offsets_v150,
    fit_group_oof_charlm_v150,
    make_dedup_group_key_v150,
    sinkhorn_nxn_v150,
    strip_onnx_weights_to_skeleton,
)

# ---------------------------------------------------------------------------
# 0. Constants & Architecture Presets (`A0..A5` Ablation Registry)
# ---------------------------------------------------------------------------
MGVZ10_MAGIC = b"MGVZ10\x01\x00"
SLOT_R_CAP_V160 = 12.0
STEP_PHI_DIM_V160 = 32
DEFAULT_T_TRAIN_MAX_V160 = 6
DEFAULT_T_INFER_MAX_V160 = 4

set_v160_qat_mode = set_v130_qat_mode

ARCH_PRESETS_V160: dict[str, dict[str, Any]] = {
    "v160_a1": {
        "dim": 320,
        "num_heads": 5,           # SA heads: 5 x 64 = 320
        "ffn_dim": 512,
        "core_depth": 2,
        "n_slow": 4,
        "n_sem": 6,
        "n_span": 4,
        "d_kv": 128,              # Compressed CA projection dim (NOT 5/2 GQA!)
        "kv_heads": 4,            # CA heads: 4 x 32 = 128 (Q, K, V all 4x32 -> ZERO Tile ops!)
        "prelude_kind": "o_l_dwconv",
        "prelude_layers": 1,      # Parallel k=7 d={1,2,4} -> exact local conv span = 25 tokens
        "prelude_ffn": 512,
        "prelude_global_se": True,
        "r0_b2": R0_B2_V130,      # Frozen in Round 1 (override via expand_tables=True in Round 2)
        "r0_b3": R0_B3_V130,
        "engram_buckets": ENGRAM_BUCKETS_V130,
        "trigram_buckets": TRIGRAM_BUCKETS_V130,
    },
    "v160_a2": {
        "dim": 320,
        "num_heads": 5,
        "ffn_dim": 640,           # Capacity challenger Arm A4 (+0.56ms @ L32; promotes only if >= +2pp hard R@P95)
        "core_depth": 2,
        "n_slow": 4,
        "n_sem": 6,
        "n_span": 4,
        "d_kv": 128,
        "kv_heads": 4,
        "prelude_kind": "o_l_dwconv",
        "prelude_layers": 1,
        "prelude_ffn": 640,
        "prelude_global_se": True,
        "r0_b2": R0_B2_V130,
        "r0_b3": R0_B3_V130,
        "engram_buckets": ENGRAM_BUCKETS_V130,
        "trigram_buckets": TRIGRAM_BUCKETS_V130,
    },
    "v160_aprime": {
        "dim": 320,
        "num_heads": 5,
        "ffn_dim": 640,
        "core_depth": 2,
        "n_slow": 4,
        "n_sem": 6,
        "n_span": 4,
        "d_kv": 128,
        "kv_heads": 4,
        "prelude_kind": "o_l_dwconv",
        "prelude_layers": 2,      # Long-range attribution challenger Arm A5 (parallel conv span = 49 tokens)
        "prelude_ffn": 640,
        "prelude_global_se": True,
        "r0_b2": R0_B2_V130,
        "r0_b3": R0_B3_V130,
        "engram_buckets": ENGRAM_BUCKETS_V130,
        "trigram_buckets": TRIGRAM_BUCKETS_V130,
    },
    "v160_v15geom": {
        "dim": 384,
        "num_heads": 6,
        "ffn_dim": 512,
        "core_depth": 2,
        "n_slow": 4,
        "n_sem": 6,
        "n_span": 6,
        "d_kv": 384,
        "kv_heads": 6,
        "prelude_kind": "v150_attn",
        "prelude_layers": 1,
        "prelude_ffn": 512,
        "prelude_global_se": False,
        "r0_b2": R0_B2_V130,
        "r0_b3": R0_B3_V130,
        "engram_buckets": ENGRAM_BUCKETS_V130,
        "trigram_buckets": TRIGRAM_BUCKETS_V130,
    },
}


# ---------------------------------------------------------------------------
# 1. Zero-FLOP Expanded Char Extractor (`R0++`) & Calibrated Cross-View Math
# ---------------------------------------------------------------------------

def extract_apex_char_batch_v160(
    texts: list[str],
    b2: int = R0_B2_V130,
    b3: int = R0_B3_V130,
    already_canonical: bool = False,
    charlm: CompactCharLMV150 | None = None,
    precomputed_charlm_9d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact drop-in extension of `extract_apex_char_batch_v140` supporting expanded zero-FLOP `(b2, b3)` hash buckets."""
    if int(b2) == R0_B2_V130 and int(b3) == R0_B3_V130:
        return extract_apex_char_batch_v140(
            texts, already_canonical=already_canonical, charlm=charlm, precomputed_charlm_9d=precomputed_charlm_9d
        )

    c1, _, _, cm, cf, spoofs, sups = extract_apex_char_batch_v140(
        texts, already_canonical=already_canonical, charlm=charlm, precomputed_charlm_9d=precomputed_charlm_9d
    )
    n, ml = c1.shape
    c2 = np.zeros((n, ml), dtype=np.int64)
    c3 = np.zeros((n, ml), dtype=np.int64)
    b2_m = max(2, int(b2) - 1)
    b3_m = max(2, int(b3) - 1)
    for i in range(n):
        row = c1[i]
        L_eff = int(cm[i].sum())
        for j in range(L_eff):
            a = int(row[j])
            b = int(row[j + 1]) if j + 1 < L_eff else 0
            c = int(row[j + 2]) if j + 2 < L_eff else 0
            c2[i, j] = ((a * 1315423911 ^ b * 2654435761) & 0x7FFFFFFF) % b2_m + 1
            c3[i, j] = ((a * 73856093 ^ b * 19349663 ^ c * 83492791) & 0x7FFFFFFF) % b3_m + 1
    return c1, c2, c3, cm, cf, spoofs, sups


def clipped_cal_logit(p: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """Bounded calibrated log-odds `clip(logit(p), -clip, clip)` — avoids tail blow-up of `(Phi^-1)'(p)`."""
    p_c = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.clip(sp_logit(p_c), -float(clip), float(clip)).astype(np.float32)


def apply_crossview_late_fusion(
    logit_r0: np.ndarray,
    logit_deep: np.ndarray,
    fusion_cfg: dict[str, Any] | None,
) -> np.ndarray:
    """Strictly monotone OOF late fusion `f = bias + w_a * a + w_b * b` (`w_a >= 0, w_b >= 0`).

    Eliminates the `w2 * a * b` counterexample (`a = b = -5 -> +2.5` top-end FP!) and symmetric
    `-lam * |a - b|` suppression of valid deep AI rescues.
    """
    if not fusion_cfg or not bool(fusion_cfg.get("enabled", False)):
        return np.asarray(logit_deep, dtype=np.float32)
    a = np.clip(np.asarray(logit_r0, dtype=np.float32), -10.0, 10.0)
    b = np.clip(np.asarray(logit_deep, dtype=np.float32), -10.0, 10.0)
    w_a = max(0.0, float(fusion_cfg.get("w_a", fusion_cfg.get("w0", 0.0))))
    w_b = max(0.0, float(fusion_cfg.get("w_b", fusion_cfg.get("w1", 1.0))))
    bias = float(fusion_cfg.get("bias", 0.0))
    return (bias + w_a * a + w_b * b).astype(np.float32)


def _fit_monotone_linear_pair(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, reg_c: float = 1.0
) -> tuple[float, float, float]:
    """Fit `bias + w_a * a + w_b * b` with `w_a >= 0, w_b >= 0` via exact softplus reparameterization."""
    at = torch.from_numpy(a).float()
    bt = torch.from_numpy(b).float()
    yt = torch.from_numpy(y).float()
    raw_wa = torch.tensor(math.log(math.expm1(0.35)), dtype=torch.float32, requires_grad=True)
    raw_wb = torch.tensor(math.log(math.expm1(0.65)), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([raw_wa, raw_wb, bias], lr=0.5, max_iter=80)

    def _closure():
        opt.zero_grad()
        wa = F.softplus(raw_wa)
        wb = F.softplus(raw_wb)
        lg = bias + wa * at + wb * bt
        reg = (0.5 / max(1e-3, reg_c)) * ((wa - 0.30) ** 2 + (wb - 0.70) ** 2 + 0.25 * (bias ** 2))
        loss = F.binary_cross_entropy_with_logits(lg, yt) + reg / float(len(y))
        loss.backward()
        return loss

    opt.step(_closure)
    wa_f = float(F.softplus(raw_wa).detach().item())
    wb_f = float(F.softplus(raw_wb).detach().item())
    bs_f = float(max(-3.0, min(3.0, float(bias.detach().item()))))
    return wa_f, wb_f, bs_f


def fit_crossview_late_fusion(
    logit_r0: np.ndarray,
    logit_deep: np.ndarray,
    y_bin: np.ndarray,
    reg_c: float = 1.0,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Fit strictly monotone OOF late fusion `f = bias + w_a * a + w_b * b` (`w_a, w_b >= 0`).

    Computes 5-fold OOF predictions on the fitting split so we can compare monotone fusion against
    R0 alone, deep alone, and simple 50/50 convex blend without in-sample optimism.
    """
    a = np.clip(np.asarray(logit_r0, dtype=np.float64), -10.0, 10.0)
    b = np.clip(np.asarray(logit_deep, dtype=np.float64), -10.0, 10.0)
    y = np.asarray(y_bin, dtype=np.float64)
    n = len(y)
    if n < 32 or y.sum() < 8 or (n - y.sum()) < 8:
        return {"enabled": False, "formula": "monotone_linear", "w_a": 0.0, "w_b": 1.0, "w0": 0.0, "w1": 1.0, "bias": 0.0}

    # Stratified deterministic folds for clean OOF evaluation
    oof_pred = np.zeros((n,), dtype=np.float32)
    pos_idx = np.where(y >= 0.5)[0]
    neg_idx = np.where(y < 0.5)[0]
    fold_ids = np.zeros((n,), dtype=np.int64)
    for k_i, idx in enumerate(pos_idx):
        fold_ids[idx] = k_i % max(2, n_folds)
    for k_i, idx in enumerate(neg_idx):
        fold_ids[idx] = k_i % max(2, n_folds)
    for fid in range(max(2, n_folds)):
        val_m = fold_ids == fid
        tr_m = ~val_m
        if tr_m.sum() < 16 or y[tr_m].sum() < 4 or (tr_m.sum() - y[tr_m].sum()) < 4:
            oof_pred[val_m] = b[val_m].astype(np.float32)
            continue
        wa_k, wb_k, bs_k = _fit_monotone_linear_pair(a[tr_m], b[tr_m], y[tr_m], reg_c=reg_c)
        oof_pred[val_m] = (bs_k + wa_k * a[val_m] + wb_k * b[val_m]).astype(np.float32)

    wa_f, wb_f, bs_f = _fit_monotone_linear_pair(a, b, y, reg_c=reg_c)
    return {
        "enabled": True,
        "formula": "monotone_linear",
        "w_a": round(wa_f, 5),
        "w_b": round(wb_f, 5),
        "w0": round(wa_f, 5),
        "w1": round(wb_f, 5),
        "bias": round(bs_f, 5),
        "_oof_logits": oof_pred,
    }


# ---------------------------------------------------------------------------
# 2. Stage 0: Expanded Zero-FLOP `ApexCharNanoRung0V160` (`R0++`, Decoupled AI/Promo!)
# ---------------------------------------------------------------------------

class ApexCharNanoRung0V160(nn.Module):
    """W8/W4 Char 1/2/3-gram Nano Engine (`R0++`) with strictly decoupled `defer_ai` / `defer_pr` & expanded zero-FLOP buckets."""

    def __init__(
        self,
        b2: int = R0_B2_V130 * 2,
        b3: int = R0_B3_V130 * 2,
        use_global_charlm_shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.b2 = int(b2)
        self.b3 = int(b3)
        self.use_global_charlm_shortcut = bool(use_global_charlm_shortcut)
        self.emb1 = QuantEmbeddingV130(R0_B1_V130, R0_D1_V130, bits=8, padding_idx=0)
        self.emb2 = QuantEmbeddingV130(self.b2, R0_D2_V130, bits=4, padding_idx=0)
        self.emb3 = QuantEmbeddingV130(self.b3, R0_D3_V130, bits=4 if self.b3 > R0_B3_V130 else 8, padding_idx=0)
        in_d = R0_D1_V130 + R0_D2_V130 + R0_D3_V130  # 112

        self.in_proj = QuantLinearV130(in_d, R0_HID_V130, bits=8)
        self.in_norm = nn.LayerNorm(R0_HID_V130)

        self.dw3 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=3, padding=1, groups=R0_HID_V130)
        self.dw5_d1 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=2, dilation=1, groups=R0_HID_V130)
        self.dw5_d2 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=4, dilation=2, groups=R0_HID_V130)
        self.dw5_d4 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=8, dilation=4, groups=R0_HID_V130)
        self.pw = QuantLinearV130(R0_HID_V130, R0_HID_V130 * 2, bits=8)
        self.se = nn.Sequential(
            QuantLinearV130(R0_HID_V130, 32, bits=8),
            nn.SiLU(),
            QuantLinearV130(32, R0_HID_V130, bits=8),
            nn.Sigmoid(),
        )
        self.stage_norm = nn.LayerNorm(R0_HID_V130)
        self.attn_pool = nn.Sequential(
            QuantLinearV130(R0_HID_V130, 64, bits=8),
            nn.Tanh(),
            QuantLinearV130(64, 1, bits=8),
        )

        self.r0_sup_tau = nn.Parameter(torch.tensor(0.50))
        pool_in_ai = R0_HID_V130 * 5 + R0_FEAT_DIM_V140
        pool_in_pr = R0_HID_V130 * 5 + R0_STRUCT_DIM_V140
        self.ai_fc = nn.Sequential(QuantLinearV130(pool_in_ai, 128, bits=8), nn.LayerNorm(128), nn.SiLU())
        self.promo_fc = nn.Sequential(QuantLinearV130(pool_in_pr, 96, bits=8), nn.LayerNorm(96), nn.SiLU())
        self.ai_head = nn.Linear(128, 1)
        self.promo_head = nn.Linear(96, 1)
        self.z_char_ai = nn.Linear(128, 32)
        self.z_char_pr = nn.Linear(96, 32)
        # Strictly separate AI and Promo defer heads -> ZERO Promo->AI cascade leak (`F6 === 0.0`!)
        self.defer_ai_head = nn.Sequential(nn.Linear(128 + 2, 32), nn.SiLU(), nn.Linear(32, 1))
        self.defer_pr_head = nn.Sequential(nn.Linear(96 + 2, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        m = mask.float()
        m_u = m.unsqueeze(-1)
        x = torch.cat([self.emb1(c1), self.emb2(c2), self.emb3(c3)], dim=-1)
        h = self.in_norm(self.in_proj(x)) * m_u
        ht = h.transpose(1, 2)
        conv_sum = (self.dw3(ht) + self.dw5_d1(ht) + self.dw5_d2(ht) + self.dw5_d4(ht)) * 0.25
        hc = conv_sum.transpose(1, 2) * m_u
        u, g = self.pw(hc).chunk(2, dim=-1)
        hc = u * torch.sigmoid(g) * m_u

        denom = m_u.sum(dim=1).clamp_min(1.0)
        mean_c = hc.sum(dim=1) / denom
        hc = hc * self.se(mean_c).unsqueeze(1)
        h = self.stage_norm(h + hc) * m_u

        p_mean = h.sum(dim=1) / denom
        p_max = h.masked_fill(m_u < 0.5, -1e4).max(dim=1).values
        a_lg = self.attn_pool(h).squeeze(-1).masked_fill(m < 0.5, -1e4)
        a_w = torch.softmax(a_lg.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        p_attn = (h * a_w).sum(dim=1)
        p_first = h[:, 0, :]
        last_idx = (m.sum(dim=1).long() - 1).clamp(min=0, max=h.shape[1] - 1)
        p_last = h.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[2])).squeeze(1)

        f_in = feats.to(h.dtype)
        f_struct = f_in[:, :R0_STRUCT_DIM_V140]
        f_charlm = f_in[:, R0_STRUCT_DIM_V140:]
        sup_norm = f_struct[:, -1:]
        g_sup_r0 = torch.sigmoid((sup_norm - self.r0_sup_tau.clamp(0.25, 0.75)) * 8.0)
        f_lm_eff = (f_charlm * (0.55 + 0.45 * g_sup_r0)) if self.use_global_charlm_shortcut else (f_charlm * 0.0)
        f_ai = torch.cat([f_struct[:, :-1] * g_sup_r0, sup_norm, f_lm_eff], dim=-1)

        h_ai = self.ai_fc(torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_ai], dim=-1))
        h_pr = self.promo_fc(torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_struct], dim=-1))

        lg_ai = self.ai_head(h_ai).squeeze(-1)
        lg_pr = self.promo_head(h_pr).squeeze(-1)
        z_char = torch.cat([self.z_char_ai(h_ai), self.z_char_pr(h_pr)], dim=-1)

        def_ai_lg = self.defer_ai_head(torch.cat([h_ai.detach(), lg_ai.detach().unsqueeze(-1), sup_norm.detach()], dim=-1)).squeeze(-1)
        def_pr_lg = self.defer_pr_head(torch.cat([h_pr.detach(), lg_pr.detach().unsqueeze(-1), sup_norm.detach()], dim=-1)).squeeze(-1)
        return {
            "logit_ai_r0": lg_ai,
            "logit_promo_r0": lg_pr,
            "defer_ai_r0_logit": def_ai_lg,
            "p_defer_ai_r0": torch.sigmoid(def_ai_lg),
            "defer_pr_r0_logit": def_pr_lg,
            "p_defer_pr_r0": torch.sigmoid(def_pr_lg),
            "z_char": z_char,
        }


# ---------------------------------------------------------------------------
# 3. Pad-Exact `O(L)` Multi-Scale Dilated DWConv + Global SE + SwiGLU Prelude
# ---------------------------------------------------------------------------

class PadExactStratospherePreludeBlockV160(nn.Module):
    """`O(L)` Pad-Exact (`P1 === 0.0`) Multi-Scale Dilated DWConv (k=7, d=1,2,4) + Masked Global SE Gate + SwiGLU!"""

    def __init__(self, dim: int = 320, ffn_dim: int = 640, use_global_se: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.use_global_se = bool(use_global_se)
        self.norm1 = nn.LayerNorm(dim)
        self.in_g = QuantLinearV130(dim, dim * 2, bits=4)
        self.dw_d1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim)
        self.dw_d2 = nn.Conv1d(dim, dim, kernel_size=7, padding=6, dilation=2, groups=dim)
        self.dw_d4 = nn.Conv1d(dim, dim, kernel_size=7, padding=12, dilation=4, groups=dim)
        self.out_p = QuantLinearV130(dim, dim, bits=4)
        if self.use_global_se:
            self.se = nn.Sequential(
                nn.Linear(dim, 48),
                nn.SiLU(),
                nn.Linear(48, dim),
                nn.Sigmoid(),
            )
        else:
            self.se = None
        self.norm2 = nn.LayerNorm(dim)
        self.w_ug = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.w_d = QuantLinearV130(ffn_dim, dim, bits=4)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m_u = attention_mask.float().unsqueeze(-1)
        u, g = self.in_g(self.norm1(x) * m_u).chunk(2, dim=-1)
        ut = (u * m_u).transpose(1, 2)
        c = ((self.dw_d1(ut) + self.dw_d2(ut) + self.dw_d4(ut)) * (1.0 / 3.0)).transpose(1, 2) * m_u
        h = c * F.silu(g) * m_u
        if self.se is not None:
            denom = m_u.sum(dim=1).clamp_min(1.0)
            g_ctx = h.sum(dim=1) / denom
            h = h * (0.5 + self.se(g_ctx).unsqueeze(1)) * m_u
        x = (x + self.out_p(h)) * m_u
        a, b = self.w_ug(self.norm2(x) * m_u).chunk(2, dim=-1)
        return (x + self.w_d(F.silu(a) * b)) * m_u


# ---------------------------------------------------------------------------
# 4. Pillar 1: `StratosphereLoopCoreV160` (Fourier phi(t) + Per-Slot Pi_R + 1-FFN + MQA/GQA d_kv=128)
# ---------------------------------------------------------------------------

def _fourier_step_features(t_val: torch.Tensor) -> torch.Tensor:
    """Continuous Fourier step features `(B, 7)` supporting zero-shot extrapolation to T=5, 6+."""
    tf = t_val.float().unsqueeze(-1) * 0.25
    return torch.cat([
        tf,
        torch.sin(tf * math.pi),
        torch.cos(tf * math.pi),
        torch.sin(tf * (2.0 * math.pi)),
        torch.cos(tf * (2.0 * math.pi)),
        torch.sin(tf * (4.0 * math.pi)),
        torch.cos(tf * (4.0 * math.pi)),
    ], dim=-1)


class StratosphereSubBlockV160(nn.Module):
    """Single-FFN Recurrent Sub-Block: Read-Only GQA/MQA Coverage CA + Haar + AI-Only Cayley + Block-Masked SA + SwiGLU + AdaLN-Zero."""

    def __init__(
        self,
        dim: int = 320,
        ffn_dim: int = 640,
        num_heads: int = 5,
        d_kv: int = 128,
        kv_heads: int = 4,
        n_slow: int = 4,
        n_sem: int = 6,
        n_span: int = 4,
        coverage_beta_enabled: bool = True,
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
        self.n_sem = int(n_sem)
        self.n_ai_slots = self.n_slow + self.n_sem
        self.n_slots = self.n_ai_slots + int(n_span)
        self.coverage_beta_enabled = bool(coverage_beta_enabled)

        # Continuous Step-Conditioned AdaLN-Zero modulation (6 gates/shifts from phi(t)!)
        self.adaln_mod = nn.Linear(STEP_PHI_DIM_V160, dim * 4)
        nn.init.zeros_(self.adaln_mod.weight)
        nn.init.zeros_(self.adaln_mod.bias)

        # Compact Read-Only MQA/GQA Cross-Attention (`d_kv=128`) + Decaying Capped Coverage Inhibition
        self.norm_ca = nn.LayerNorm(dim)
        self.ca_q = QuantLinearV130(dim, d_kv, bits=4)
        self.ca_o = QuantLinearV130(d_kv, dim, bits=4)
        self.cov_beta_raw = nn.Parameter(torch.tensor(-0.4))
        self.cov_lam_raw = nn.Parameter(torch.tensor(0.6))

        # Fine-scale Haar Wavelet Reread (64-d, read-only)
        self.haar_q = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.haar_o = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.haar_gate = nn.Parameter(torch.tensor(-0.8))

        # Within-AI Cayley Symplectic Peer Coupling (Prosecutor [0:3) <-> Defense [3:6), ZERO Promo!)
        self.c1_q = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V130))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.20))

        # Block-Masked Slot Self-Attention (`[0:n_ai)` CANNOT see `[n_ai:n_slots)` -> F6 === 0.0!)
        self.norm_sa = nn.LayerNorm(dim)
        self.sa_qkv = QuantLinearV130(dim, dim * 3, bits=4)
        self.sa_o = QuantLinearV130(dim, dim, bits=4)

        # Single Unified SwiGLU FFN (drops redundant `_fast_inner_step` FFN -> -43%..-49% MACs!)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.ff_down = QuantLinearV130(ffn_dim, dim, bits=4)

        # Bounded residual step size `h_t = 0.6 * sigmoid(step_h)` (Huginn/Ouro bounded dynamics)
        self.step_h = nn.Parameter(torch.tensor(0.4))

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

    def forward(
        self,
        s: torch.Tensor,
        cov_prev: torch.Tensor,
        k_ev_0: torch.Tensor,
        v_ev_0: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        g_sup: torch.Tensor,
        step_phi: torch.Tensor,
        sa_iso_mask: torch.Tensor,
        bus_down: nn.Module,
        bus_up: nn.Module,
        apply_cov_gate: torch.Tensor,
        step_scale: torch.Tensor,
        use_bounded_step: bool = True,
        return_adaln_stat: bool = False,
    ) -> tuple[Any, ...]:
        B, L, _ = k_ev_0.shape
        m_f = mask.float()
        g_ca, g_sa, g_ff, b_ff = torch.tanh(self.adaln_mod(step_phi)).chunk(4, dim=-1)

        s_orig = s

        # 1. Read-Only Compressed Cross-Attention (`d_kv=128`, `4x32` heads -> ZERO Tile ops!) + Capped Coverage
        sn_ca = self.norm_ca(s)
        qh = self.ca_q(sn_ca).view(B, self.n_slots, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        kh = k_ev_0.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        vh = v_ev_0.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.kv_head_dim ** -0.5)
        if self.coverage_beta_enabled:
            beta_eff = F.softplus(self.cov_beta_raw) * (g_sup * apply_cov_gate).to(sc.dtype).view(-1, 1, 1, 1)
            cov_pen = (beta_eff * cov_prev.to(sc.dtype)[:, None, None, :]).clamp_max(COV_LOGIT_CAP_V150)
            sc = sc - cov_pen
        sc = sc.masked_fill(m_f[:, None, None, :] < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(s.dtype)
        r_read = self.ca_o(torch.matmul(at, vh).transpose(1, 2).reshape(B, self.n_slots, self.d_kv))
        s = s + (1.0 + 0.5 * g_ca.unsqueeze(1)) * r_read

        # Strictly AI-only decaying coverage update (Promo attention NEVER touches AI coverage!)
        a_bar_ai = at[:, :, : self.n_ai_slots, :].mean(dim=(1, 2))
        cov_lam = 0.85 * torch.sigmoid(self.cov_lam_raw)
        cov_next = (m_f * torch.clamp(cov_lam * cov_prev.float() + a_bar_ai.float(), 0.0, COV_MAX_V150)).to(s.dtype)

        # Fine-scale Haar reread (read-only!)
        q_haar = self.haar_q(sn_ca)
        sc_h = (torch.matmul(q_haar, k_haar.transpose(-1, -2)) / 8.0).masked_fill(haar_mask[:, None, :].float() < 0.5, -1e4)
        at_h = torch.softmax(sc_h.float(), dim=-1).to(s.dtype)
        s = s + torch.sigmoid(self.haar_gate) * self.haar_o(torch.matmul(at_h, v_haar))

        # 2. Within-AI Cayley Peer Coupling (lightweight 64-d bus only, zero extra FFN!)
        z_sem_c = self._cayley_ai_peer_couple(s[:, self.n_slow : self.n_ai_slots, :], bus_down, bus_up)
        s = torch.cat([s[:, : self.n_slow, :], z_sem_c, s[:, self.n_ai_slots :, :]], dim=1)

        # 3. Block-Masked Slot Self-Attention (`F6 === 0.0`)
        sq, sk, sv = self.sa_qkv(self.norm_sa(s)).chunk(3, dim=-1)
        sq = sq.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sk = sk.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sv = sv.view(B, self.n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        sa_sc = torch.matmul(sq, sk.transpose(-2, -1)) * (self.head_dim ** -0.5) + sa_iso_mask.to(sq.dtype)
        sa = torch.softmax(sa_sc.float(), dim=-1).to(s.dtype)
        sa_ctx = torch.matmul(sa, sv).transpose(1, 2).reshape(B, self.n_slots, self.dim)
        s = s + (1.0 + 0.5 * g_sa.unsqueeze(1)) * self.sa_o(sa_ctx)

        # 4. Single Unified SwiGLU FFN with Step-Conditioned AdaLN-Zero
        nf = self.norm_ff(s) * (1.0 + 0.25 * g_ff.unsqueeze(1)) + 0.15 * b_ff.unsqueeze(1)
        u, g = self.ff_up(nf).chunk(2, dim=-1)
        s = s + self.ff_down(F.silu(u) * g)

        if use_bounded_step:
            # Causal step size `h_t = 0.65 * sigmoid(step_h) * step_scale(t)` (depends ONLY on executed step `t`!)
            h_step = (0.65 * torch.sigmoid(self.step_h)) * step_scale.to(s.dtype).view(B, 1, 1)
            s = s_orig + h_step * (s - s_orig)
        if return_adaln_stat:
            g_abs = (g_ca.detach().abs().mean() + g_sa.detach().abs().mean() + g_ff.detach().abs().mean()) / 3.0
            return s, cov_next, g_abs
        return s, cov_next


class StratosphereLoopCoreV160(nn.Module):
    """Input-Injected Per-Slot Norm-Bounded Residual Core with Optional Fourier `phi(t)`, Causal `h_t`, & `diag(M_3x3, 1)` mHC."""

    def __init__(
        self,
        dim: int = 320,
        ffn_dim: int = 512,
        num_heads: int = 5,
        d_kv: int = 128,
        kv_heads: int = 4,
        n_slow: int = 4,
        n_sem: int = 6,
        n_span: int = 4,
        core_depth: int = 2,
        coverage_beta_enabled: bool = True,
        dynamics_mode: str = "stratosphere",
        use_step_phi: bool = False,
        step_decay_mode: str = "inv_sqrt",
        use_slot9_dt: bool = False,
        r_cap: float = SLOT_R_CAP_V160,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.d_kv = int(d_kv)
        self.kv_heads = int(kv_heads)
        self.kv_head_dim = d_kv // kv_heads
        self.n_slow = int(n_slow)
        self.n_sem = int(n_sem)
        self.n_span = int(n_span)
        self.n_ai_slots = self.n_slow + self.n_sem
        self.n_slots = self.n_ai_slots + self.n_span
        self.core_depth = max(1, int(core_depth))
        self.dynamics_mode = str(dynamics_mode)
        self.use_step_phi = bool(use_step_phi)
        self.step_decay_mode = str(step_decay_mode)
        self.use_slot9_dt = bool(use_slot9_dt)
        self.r_cap = float(r_cap)

        self.slot_queries = nn.Parameter(torch.randn(1, self.n_slots, dim) * 0.04)

        # Single-Pass Read-Only Compressed CA Projections (`d_kv=128`, `4x32` heads -> 3x smaller stage transfer, ZERO Tile ops!)
        self.k_ev_proj = QuantLinearV130(dim, d_kv, bits=4)
        self.v_ev_proj = QuantLinearV130(dim, d_kv, bits=4)

        # Shared 64-d slow bus for AI-only Cayley peer coupling & AI slot-9 handoff
        self.bus_down = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.bus_up = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)

        # Continuous Fourier step embedding MLP (active ONLY when `use_step_phi=True`, e.g., Arm A2!)
        self.step_phi = nn.Sequential(
            nn.Linear(7, STEP_PHI_DIM_V160),
            nn.SiLU(),
            nn.Linear(STEP_PHI_DIM_V160, STEP_PHI_DIM_V160),
        )

        self.blocks = nn.ModuleList([
            StratosphereSubBlockV160(
                dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, d_kv=d_kv, kv_heads=kv_heads,
                n_slow=n_slow, n_sem=n_sem, n_span=n_span, coverage_beta_enabled=coverage_beta_enabled,
            )
            for _ in range(self.core_depth)
        ])

        # Input-injected L0 surface penetration into slow slots `[0:n_slow)` (re-injected every step!)
        self.l0_pen_a = QuantLinearV130(dim, 32, bias=False, bits=4)
        self.l0_pen_b = QuantLinearV130(32, dim, bias=False, bits=4)
        self.l0_pen_gate = nn.Parameter(torch.full((dim,), -0.8))

        # AI-only slot-9 self-state handoff: receives ONLY `(prev_bus, 0, |b_t|)` by default (`use_slot9_dt=False` -> ZERO R0 coupling!)
        self.handoff_proj = QuantLinearV130(BUS_DIM_V130 + 2, dim, bits=4)

        # Step-conditioned Block-Diagonal `diag(M_{AI, 3x3}, 1)` 10-Iter Sinkhorn mHC (AI-only!)
        self.mhc_base_log = nn.Parameter(torch.tensor([
            [1.4, -0.5, -0.5],
            [-0.5, 1.4, -0.5],
            [-0.5, -0.5, 1.4],
        ], dtype=torch.float32))
        self.mhc_step_delta = nn.Linear(STEP_PHI_DIM_V160, 9)
        nn.init.zeros_(self.mhc_step_delta.weight)
        nn.init.zeros_(self.mhc_step_delta.bias)
        self.mhc_read_base = nn.Parameter(torch.tensor([1.0, -0.4, -0.4], dtype=torch.float32))
        self.mhc_read_delta = nn.Linear(STEP_PHI_DIM_V160, 3)
        nn.init.zeros_(self.mhc_read_delta.weight)
        nn.init.zeros_(self.mhc_read_delta.bias)

        # Legacy `v150_anchor` parameters (Arm A0 ONLY: faithful `anchor_alpha`, per-step `rho[0..3]`, per-step `kappa[0..3]`;
        # steps 5/6 deterministically reuse index 3 per locked pre-registration!)
        self.anchor_alpha = nn.Parameter(torch.full((1, 1, dim), -1.1))
        self.contract_rho_raw = nn.Parameter(torch.tensor([0.0, 0.04, 0.08, 0.12], dtype=torch.float32))
        self.loop_q_scale = nn.Parameter(torch.ones(4, 1, dim, dtype=torch.float32))

        sa_mask = torch.zeros((1, 1, self.n_slots, self.n_slots), dtype=torch.float32)
        sa_mask[:, :, : self.n_ai_slots, self.n_ai_slots :] = -1e4
        self.register_buffer("sa_iso_mask", sa_mask, persistent=False)

    def build_readonly_cache(self, h_tok: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        m_u = mask.float().unsqueeze(-1)
        k_ev = self.k_ev_proj(h_tok) * m_u
        v_ev = self.v_ev_proj(h_tok) * m_u

        B, L, _ = h_tok.shape
        blk0 = self.blocks[0]
        q = self.slot_queries.expand(B, -1, -1)
        qh = blk0.ca_q(blk0.norm_ca(q)).view(B, self.n_slots, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.kv_heads, self.kv_head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.kv_head_dim ** -0.5)
        sc = sc.masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(h_tok.dtype)
        ctx = torch.matmul(at, vh).transpose(1, 2).reshape(B, self.n_slots, self.d_kv)
        init_slots = q + blk0.ca_o(ctx)
        if self.dynamics_mode != "v150_anchor":
            init_slots, _ = self._per_slot_project(init_slots)
        return {"k_ev": k_ev, "v_ev": v_ev, "init_slots": init_slots}

    def compute_step_phi(self, t_val: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.use_step_phi:
            return self.step_phi(_fourier_step_features(t_val).to(dtype))
        return torch.zeros((t_val.shape[0], STEP_PHI_DIM_V160), device=t_val.device, dtype=dtype)

    def compute_step_scale(self, t_val: torch.Tensor) -> torch.Tensor:
        """Causal step-size multiplier depending ONLY on executed step `t_val >= 0` (NEVER future total depth `T`)."""
        if self.step_decay_mode == "inv_sqrt":
            return torch.rsqrt(t_val.float().clamp_min(0.0) + 1.0)
        return torch.ones_like(t_val.float())

    def _per_slot_project(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Strictly PER-SLOT L2 norm cap `Pi_R` (bounds norm only; does NOT by itself prove contraction or F6)."""
        nrm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sat_ai = (nrm[:, : self.n_ai_slots, 0] >= self.r_cap).float().mean(dim=-1)
        return s * torch.clamp(self.r_cap / nrm, max=1.0), sat_ai

    def forward_step(
        self,
        ai_streams: torch.Tensor,
        pr_stream: torch.Tensor,
        init_slots: torch.Tensor,
        cov_prev: torch.Tensor,
        k_ev_0: torch.Tensor,
        v_ev_0: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        g_sup: torch.Tensor,
        t_val: torch.Tensor,
        disagree_dt: torch.Tensor,
        abs_b_prev: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Execute ONE reusable recurrent step at depth `t_val` (0=T1, 1=T2, 2=T3, 3=T4, 4=T5, 5=T6...)."""
        B = ai_streams.shape[0]
        step_phi = self.compute_step_phi(t_val, ai_streams.dtype)
        step_scale = self.compute_step_scale(t_val)
        s_in = torch.cat([ai_streams[:, 0], pr_stream], dim=1)  # (B, n_slots, D)
        s = s_in

        # AI-only slot-9 handoff for t >= 1 (default `use_slot9_dt=False` -> ZERO R0 signal enters slots!)
        active_gt0 = (t_val.float() >= 0.5).to(s.dtype)
        prev_bus = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        dt_slot9 = disagree_dt.to(s.dtype) if self.use_slot9_dt else torch.zeros_like(abs_b_prev, dtype=s.dtype)
        h_in = 0.22 * self.handoff_proj(torch.cat([prev_bus, dt_slot9.unsqueeze(-1), abs_b_prev.to(s.dtype).unsqueeze(-1)], dim=-1))
        h_in = h_in * active_gt0.unsqueeze(-1)
        s = torch.cat([
            s[:, : self.n_ai_slots - 1, :],
            s[:, self.n_ai_slots - 1 : self.n_ai_slots, :] + h_in.unsqueeze(1),
            s[:, self.n_ai_slots :, :],
        ], dim=1)

        # Input-injected L0 surface penetration into slow slots (every step!)
        l0_pen = self.l0_pen_b(F.silu(self.l0_pen_a(h0_pool))).unsqueeze(1)
        s = torch.cat([
            s[:, : self.n_slow, :] + torch.sigmoid(self.l0_pen_gate).view(1, 1, -1) * l0_pen,
            s[:, self.n_slow :, :],
        ], dim=1)

        cov_cur = cov_prev
        use_strat = self.dynamics_mode != "v150_anchor"
        adaln_acc = torch.zeros((), device=s.device, dtype=torch.float32)
        for b_idx, blk in enumerate(self.blocks):
            cov_gate = torch.ones_like(active_gt0) if b_idx > 0 else active_gt0
            blk_out = blk(
                s=s, cov_prev=cov_cur, k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=mask,
                k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, g_sup=g_sup,
                step_phi=step_phi, sa_iso_mask=self.sa_iso_mask,
                bus_down=self.bus_down, bus_up=self.bus_up,
                apply_cov_gate=cov_gate, step_scale=step_scale,
                use_bounded_step=use_strat, return_adaln_stat=return_diagnostics,
            )
            if return_diagnostics:
                s, cov_cur, g_abs = blk_out
                adaln_acc = adaln_acc + g_abs
            else:
                s, cov_cur = blk_out

        if use_strat:
            # Pillar 1: Strictly per-slot norm projection `Pi_R` (no anchor_alpha lock, no kappa > 1!)
            s_cand, sat_rate_ai = self._per_slot_project(s)
        else:
            # Legacy Arm A0 control: exact `anchor_alpha`, per-step `rho[oi]`, and per-step `kappa[oi]` (`T5/T6` reuse `oi=3`)
            oi = t_val.long().clamp(0, 3)
            a_mix = 0.25 + 0.20 * torch.sigmoid(self.anchor_alpha)
            s_prop = (1.0 - a_mix) * s + a_mix * init_slots
            rho = (0.30 + 0.48 * torch.sigmoid(self.contract_rho_raw[oi])).to(s.dtype).view(B, 1, 1)
            q_sc = self.loop_q_scale[oi].clamp(0.5, 1.8).to(s.dtype)  # (B, 1, D)
            s_cand = ((1.0 - rho) * s_in + rho * s_prop) * q_sc
            sat_rate_ai = torch.zeros((B,), device=s.device, dtype=torch.float32)

        # Block-Diagonal `diag(M_{AI, 3x3}, 1)` 10-Iter Sinkhorn mHC (AI-only!)
        s_ai_cand = s_cand[:, : self.n_ai_slots, :]
        s_pr_next = s_cand[:, self.n_ai_slots :, :]

        m_log = self.mhc_base_log.clamp(-3.0, 3.0).unsqueeze(0) + 0.35 * torch.tanh(self.mhc_step_delta(step_phi)).view(B, 3, 3)
        m3 = sinkhorn_nxn_v150(m_log, iters=max(12, MHC_SINKHORN_ITERS_V150)).to(s_ai_cand.dtype)
        ai_stack = torch.stack([s_ai_cand, ai_streams[:, 1], ai_streams[:, 2]], dim=1)  # (B, 3, n_ai, D)
        ai_streams_next = torch.einsum("bij,bjnd->bind", m3, ai_stack)

        r_lg = self.mhc_read_base.unsqueeze(0) + 0.35 * torch.tanh(self.mhc_read_delta(step_phi))
        r_w = torch.softmax(r_lg.float(), dim=-1).to(s_ai_cand.dtype)
        s_ai_out = (ai_streams_next * r_w.view(B, 3, 1, 1)).sum(dim=1)
        slots_out = torch.cat([s_ai_out, s_pr_next], dim=1)

        ret: dict[str, torch.Tensor] = {
            "slots": slots_out,
            "ai_streams": ai_streams_next,
            "pr_stream": s_pr_next,
            "cov": cov_cur.to(s.dtype),
            "step_phi": step_phi,
        }
        if return_diagnostics:
            ret["proj_sat_rate_ai"] = sat_rate_ai
            ret["adaln_gain_abs"] = adaln_acc / float(max(1, len(self.blocks)))
        return ret


# ---------------------------------------------------------------------------
# 5. Complete Unified `MeguvizV160StratosphereLoopModel`
# ---------------------------------------------------------------------------

class MeguvizV160StratosphereLoopModel(nn.Module):
    """Unified <=5.00 MiB Bootable `StratosphereLoop` Model (`v1.6-A1` primary + full A0..A5 switches & 23-Gate CI)."""

    def __init__(
        self,
        arch: str = "v160_a1",
        vocab_size: int = 12288,
        max_len: int = MAX_LEN_V140,
        style_dim: int = STYLE_FEAT_DIM_V11,
        style_era_dropout_p: float = 0.15,
        dynamics_mode: str = "stratosphere",
        use_step_phi: bool = False,
        step_decay_mode: str = "inv_sqrt",
        use_slot9_dt: bool = False,
        expand_tables: bool = False,
        coverage_beta_enabled: bool = True,
        use_local_charlm_inject: bool = True,
        use_global_charlm_shortcut: bool = False,
        prelude_global_se: bool | None = None,
    ) -> None:
        super().__init__()
        if arch not in ARCH_PRESETS_V160:
            raise ValueError(f"Unknown arch={arch}; choose from {sorted(ARCH_PRESETS_V160)}")
        cfg = dict(ARCH_PRESETS_V160[arch])
        if prelude_global_se is not None:
            cfg["prelude_global_se"] = bool(prelude_global_se)
        if expand_tables:
            cfg["r0_b2"] = R0_B2_V130 * 2
            cfg["r0_b3"] = R0_B3_V130 * 2 if int(cfg["prelude_layers"]) == 1 else R0_B3_V130
            cfg["engram_buckets"] = ENGRAM_BUCKETS_V130 * 2
            cfg["trigram_buckets"] = TRIGRAM_BUCKETS_V130 * 2 if int(cfg["prelude_layers"]) == 1 else TRIGRAM_BUCKETS_V130

        self.arch = str(arch)
        self.cfg = cfg
        self.dim = int(cfg["dim"])
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.style_era_dropout_p = float(style_era_dropout_p)
        self.dynamics_mode = str(dynamics_mode)
        self.use_step_phi = bool(use_step_phi)
        self.step_decay_mode = str(step_decay_mode)
        self.use_slot9_dt = bool(use_slot9_dt)
        self.expand_tables = bool(expand_tables)
        self.coverage_beta_enabled = bool(coverage_beta_enabled)
        self.use_local_charlm_inject = bool(use_local_charlm_inject)
        self.use_global_charlm_shortcut = bool(use_global_charlm_shortcut)
        self.engram_buckets = int(cfg["engram_buckets"])
        self.trigram_buckets = int(cfg["trigram_buckets"])
        self.r0_b2 = int(cfg["r0_b2"])
        self.r0_b3 = int(cfg["r0_b3"])

        D = self.dim

        # Stage 0: `R0++` Expanded Zero-FLOP ApexCharNano
        self.rung0 = ApexCharNanoRung0V160(
            b2=self.r0_b2, b3=self.r0_b3, use_global_charlm_shortcut=self.use_global_charlm_shortcut
        )

        # Stage 1: Factorized BPE + Expanded Zero-FLOP Bigram/Trigram Engrams + Local 8-ch CharLM + Prelude
        self.word_emb_a = QuantEmbeddingV130(vocab_size, 48, bits=4, padding_idx=0)
        self.word_emb_p = QuantLinearV130(48, D, bias=False, bits=8)
        self.engram_emb = QuantEmbeddingV130(self.engram_buckets, 16, bits=4, padding_idx=0)
        self.engram_p = QuantLinearV130(16, D, bias=False, bits=8)
        self.trigram_emb = QuantEmbeddingV130(self.trigram_buckets, 16, bits=4, padding_idx=0)
        self.trigram_p = QuantLinearV130(16, D, bias=False, bits=8)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, D) * 0.02)
        self.emb_norm = nn.LayerNorm(D)

        self.local_lm_u = nn.Linear(LOCAL_CHARLM_DIM_V150, D, bias=False)
        self.local_lm_gate = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.local_lm_u.weight, std=0.01)

        if cfg["prelude_kind"] == "v150_attn":
            self.prelude_blocks = nn.ModuleList([
                PadExactConvBlockV130(D, int(cfg["prelude_ffn"]), int(cfg["num_heads"]), bits=4)
                for _ in range(int(cfg["prelude_layers"]))
            ])
        else:
            self.prelude_blocks = nn.ModuleList([
                PadExactStratospherePreludeBlockV160(
                    dim=D, ffn_dim=int(cfg["prelude_ffn"]), use_global_se=bool(cfg["prelude_global_se"])
                )
                for _ in range(int(cfg["prelude_layers"]))
            ])

        self.haar_k_proj = QuantLinearV130(D, BUS_DIM_V130, bias=False, bits=4)
        self.haar_v_proj = QuantLinearV130(D, BUS_DIM_V130, bias=False, bits=4)

        self.sup_gate_tau = nn.Parameter(torch.tensor(16.0))
        self.sup_gate_scale = nn.Parameter(torch.tensor(4.0))

        self.pr_attr_attn = nn.Sequential(QuantLinearV130(D, 32, bits=4), nn.Tanh(), nn.Linear(32, 1))

        # R1 fast prefix heads (strictly ZERO Promo->AI cross-talk!)
        self.r1_style_ai = nn.Sequential(nn.Linear(style_dim - 8, 32), nn.SiLU())
        self.r1_style_pr = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU())
        self.r1_ai_head = nn.Linear(D + 32 + 32, 1)
        self.r1_promo_head = nn.Linear(D + 32 + 32, 1)

        # Stage 2: StratosphereLoop Core (reusable single step for T=1..6+!)
        self.core = StratosphereLoopCoreV160(
            dim=D, ffn_dim=int(cfg["ffn_dim"]), num_heads=int(cfg["num_heads"]),
            d_kv=int(cfg["d_kv"]), kv_heads=int(cfg["kv_heads"]),
            n_slow=int(cfg["n_slow"]), n_sem=int(cfg["n_sem"]), n_span=int(cfg["n_span"]),
            core_depth=int(cfg["core_depth"]), coverage_beta_enabled=coverage_beta_enabled,
            dynamics_mode=dynamics_mode, use_step_phi=self.use_step_phi,
            step_decay_mode=self.step_decay_mode, use_slot9_dt=self.use_slot9_dt,
        )

        # Strictly Disjoint 256-d Coda (`128 shared + 64 AI-priv + 64 Promo-priv`)
        self.coda_shared = QuantLinearV130(D * 2, SHARED_END_V130, bits=4)
        self.coda_ai_priv = QuantLinearV130(D + 32 + 32, AI_END_V130 - SHARED_END_V130, bits=4)
        self.coda_pr_priv = QuantLinearV130(D * 2 + 32 + 32, EMB_DIM_V130 - AI_END_V130, bits=4)
        self.emb_norm_sh = nn.LayerNorm(SHARED_END_V130)
        self.emb_norm_ai = nn.LayerNorm(AI_END_V130 - SHARED_END_V130)
        self.emb_norm_pr = nn.LayerNorm(EMB_DIM_V130 - AI_END_V130)

        # Single Shared Depth-Conditioned Readout (FiLM on `emb_ai`/`emb_pr` from `phi(t)`!)
        self.read_film_ai = nn.Linear(STEP_PHI_DIM_V160, AI_END_V130 * 2)
        self.read_film_pr = nn.Linear(STEP_PHI_DIM_V160, PR_EMB_DIM_V130 * 2)
        nn.init.zeros_(self.read_film_ai.weight)
        nn.init.zeros_(self.read_film_ai.bias)
        nn.init.zeros_(self.read_film_pr.weight)
        nn.init.zeros_(self.read_film_pr.bias)

        self.cbm_ai_enc = nn.Linear(AI_END_V130, CBM_AI_K_V130)
        self.cbm_pr_enc = nn.Linear(PR_EMB_DIM_V130, CBM_PR_K_V130)
        self.cbm_ai_tau = nn.Parameter(torch.full((CBM_AI_K_V130,), 0.25))
        self.cbm_pr_tau = nn.Parameter(torch.full((CBM_PR_K_V130,), 0.25))
        self.cbm_ai_to_logit = nn.Linear(CBM_AI_K_V130, 1, bias=False)
        self.cbm_pr_to_logit = nn.Linear(CBM_PR_K_V130, 1, bias=False)

        self.ai_head = SoftBallLogitNormHeadV121(AI_END_V130, init_r_cap=7.2)
        self.promo_head = SoftBallLogitNormHeadV121(PR_EMB_DIM_V130, init_r_cap=7.2)
        self.promo_sub4_head = nn.Linear(PR_EMB_DIM_V130, 4)

        # `TaH2` Lookahead Utility Halt Head (strictly AI-only features; receives clipped SIGNED logit diff `(a-b)` & `|a-b|`)
        self.halt_utility_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + STEP_PHI_DIM_V160 + 6),
            nn.Linear(AI_END_V130 * 2 + STEP_PHI_DIM_V160 + 6, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

        # DynaSeam Long-Doc Aggregator
        self.long_ai_attn = nn.Sequential(nn.Linear(AI_END_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_pr_attn = nn.Sequential(nn.Linear(PR_EMB_DIM_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_ai_head = nn.Linear(AI_END_V130 + 3, 1)
        self.long_pr_head = nn.Linear(PR_EMB_DIM_V130 + 3, 1)

    def compute_support_gate(self, support_pairs: torch.Tensor) -> torch.Tensor:
        tau = self.sup_gate_tau.clamp(10.0, 22.0)
        sc = self.sup_gate_scale.clamp(2.0, 6.0)
        return torch.sigmoid((support_pairs.float() - tau) / sc)

    def _pool_promo_attr(self, h_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        w_lg = self.pr_attr_attn(h_tok).squeeze(-1).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(w_lg.float(), dim=-1).to(h_tok.dtype).unsqueeze(-1)
        return (h_tok * w).sum(dim=1)

    def _readout_step(
        self,
        slots: torch.Tensor,
        h_pool_0: torch.Tensor,
        pr_attr_0: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        step_phi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ns, n_ai = self.core.n_slow, self.core.n_ai_slots
        slow_p = slots[:, :ns, :].mean(dim=1)
        sem_p = slots[:, ns:n_ai, :].mean(dim=1)
        span_p = slots[:, n_ai:, :].mean(dim=1)

        z_sh = self.emb_norm_sh(self.coda_shared(torch.cat([slow_p, h_pool_0], dim=-1)))
        z_ai_p = self.emb_norm_ai(self.coda_ai_priv(torch.cat([sem_p, st_ai, z_char[:, :32]], dim=-1)))
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, pr_attr_0, st_pr, z_char[:, 32:]], dim=-1)))

        emb_ai = torch.cat([z_sh, z_ai_p], dim=-1)
        emb_pr = torch.cat([z_sh, z_pr_p], dim=-1)

        # Depth-conditioned FiLM modulation (zero-init -> smooth, shared readout across T=1..6+!)
        ga, ba = self.read_film_ai(step_phi).chunk(2, dim=-1)
        gp, bp = self.read_film_pr(step_phi).chunk(2, dim=-1)
        emb_ai_m = emb_ai * (1.0 + 0.25 * torch.tanh(ga)) + 0.15 * torch.tanh(ba)
        emb_pr_m = emb_pr * (1.0 + 0.25 * torch.tanh(gp)) + 0.15 * torch.tanh(bp)

        # Smooth ISTA soft-threshold sparse CBM (zero discontinuous TopK/Scatter ops -> faster CPU ORT & exact FP32 parity!)
        c_ai_sp = F.relu(self.cbm_ai_enc(emb_ai_m) - self.cbm_ai_tau.clamp(0.0, 2.0))
        c_pr_sp = F.relu(self.cbm_pr_enc(emb_pr_m) - self.cbm_pr_tau.clamp(0.0, 2.0))

        lg_ai = self.ai_head(emb_ai_m) + 0.15 * self.cbm_ai_to_logit(c_ai_sp).squeeze(-1)
        lg_pr = self.promo_head(emb_pr_m) + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1)
        return {
            "logit_ai": lg_ai,
            "logit_promo": lg_pr,
            "promo_sub4_logit": self.promo_sub4_head(emb_pr_m),
            "emb_ai": emb_ai_m,
            "emb_pr": emb_pr_m,
        }

    def _predict_utility(
        self,
        emb_ai_cur: torch.Tensor,
        emb_ai_prev: torch.Tensor,
        logit_ai_cur: torch.Tensor,
        logit_ai_prev: torch.Tensor,
        disagree_dt: torch.Tensor,
        g_sup: torch.Tensor,
        step_phi: torch.Tensor,
        t_val: torch.Tensor,
    ) -> torch.Tensor:
        """`TaH2` Lookahead Utility Head (strictly AI-only, detached state; `disagree_dt` is clipped SIGNED `a - b`)."""
        dt_c = disagree_dt.detach().clamp(-12.0, 12.0)
        sc = torch.stack([
            logit_ai_cur.detach(),
            (logit_ai_cur - logit_ai_prev).detach(),
            logit_ai_cur.detach().abs(),
            dt_c,
            dt_c.abs(),
            g_sup.detach(),
        ], dim=-1)
        hin = torch.cat([
            emb_ai_cur.detach(),
            (emb_ai_cur - emb_ai_prev).detach(),
            step_phi.detach(),
            sc,
        ], dim=-1)
        return self.halt_utility_head(hin).squeeze(-1)

    def readout_at_step(
        self,
        slots: torch.Tensor,
        h_pool_0: torch.Tensor,
        pr_attr_0: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        t_idx: float,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the shared readout head on arbitrary `slots` at depth index `t_idx` (for T3-state -> T4-readout bias control!)."""
        t_vec = torch.full((slots.shape[0],), float(t_idx), device=slots.device, dtype=torch.float32)
        phi = self.core.compute_step_phi(t_vec, slots.dtype)
        return self._readout_step(slots, h_pool_0, pr_attr_0, st_ai, st_pr, z_char, phi)

    def forward_r0(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.rung0(c1, c2, c3, mask, feats)

    def forward_prelude_and_t1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        z_char: torch.Tensor,
        support_pairs: torch.Tensor,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Stage 1: Run `O(L)` Prelude + Single-Pass MQA/GQA `K_0/V_0` + R1 + Loop Step `T1` (`t=0`)!

        Notice: NEVER accepts `logit_ai_r0` -> guarantees zero-laziness by construction!
        """
        B, L = input_ids.shape
        m = attention_mask.float()
        m_u = m.unsqueeze(-1)
        g_sup = self.compute_support_gate(support_pairs).to(style_feats.dtype)

        st = style_feats.clone()
        if self.training and self.style_era_dropout_p > 0.0:
            keep = (torch.rand((B, 1), device=st.device) >= self.style_era_dropout_p).to(st.dtype)
            st[:, :40] = st[:, :40] * keep

        if self.training and r0_drop_p > 0.0:
            r0_keep = (torch.rand((B, 1), device=z_char.device) >= float(r0_drop_p)).to(z_char.dtype)
            zc_eff = z_char * r0_keep
        else:
            zc_eff = z_char

        loc_raw = local_charlm.to(style_feats.dtype)
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((B, 1, 1), device=loc_raw.device) >= float(charlm_drop_p)).to(loc_raw.dtype)
            loc_eff = loc_raw * lm_keep
        else:
            loc_eff = loc_raw

        pos = self.pos_emb[:, :L, :]
        h_sem = self.emb_norm(
            self.word_emb_p(self.word_emb_a(input_ids))
            + self.engram_p(self.engram_emb(engram_ids))
            + self.trigram_p(self.trigram_emb(trigram_ids))
            + pos
        ) * m_u

        if self.use_local_charlm_inject:
            hits_4 = loc_eff[..., 4:8]
            has_hi_hit = (hits_4[..., 1:4].max(dim=-1, keepdim=True).values > 1e-4).to(h_sem.dtype)
            g_loc = torch.sigmoid(self.local_lm_gate(hits_4) - 1.0) * has_hi_hit * m_u
            x = (h_sem + g_loc * self.local_lm_u(loc_eff)) * m_u
        else:
            x = h_sem

        h_tok = x
        for blk in self.prelude_blocks:
            h_tok = blk(h_tok, attention_mask)

        if (not torch.jit.is_tracing()) and (L % 2 == 1):
            hp, mp = F.pad(h_tok, (0, 0, 0, 1)), F.pad(m, (0, 1))
        else:
            hp, mp = h_tok, m
        m_even = mp[:, 0::2]
        h_diff = (hp[:, 0::2, :] - hp[:, 1::2, :]) * m_even.unsqueeze(-1)
        k_haar = self.haar_k_proj(h_diff) * m_even.unsqueeze(-1)
        v_haar = self.haar_v_proj(h_diff) * m_even.unsqueeze(-1)
        haar_mask = m_even

        denom = m_u.sum(dim=1).clamp_min(1.0)
        h_pool_0 = h_tok.sum(dim=1) / denom
        pr_attr_0 = self._pool_promo_attr(h_tok, attention_mask)
        st_ai = self.r1_style_ai(st[:, : self.style_dim - 8]) * g_sup.unsqueeze(-1)
        st_pr = self.r1_style_pr(st)
        lg_r1_ai = self.r1_ai_head(torch.cat([h_pool_0, st_ai, zc_eff[:, :32]], dim=-1)).squeeze(-1)
        lg_r1_pr = self.r1_promo_head(torch.cat([h_pool_0, st_pr, zc_eff[:, 32:]], dim=-1)).squeeze(-1)

        ro_cache = self.core.build_readonly_cache(h_tok, attention_mask)
        init_slots = ro_cache["init_slots"]
        k_ev_0, v_ev_0 = ro_cache["k_ev"], ro_cache["v_ev"]

        ai_init = init_slots[:, : self.core.n_ai_slots, :].unsqueeze(1).expand(-1, MHC_AI_STREAMS_V150, -1, -1).contiguous()
        pr_init = init_slots[:, self.core.n_ai_slots :, :]
        cov_0 = torch.zeros((B, L), device=h_tok.device, dtype=h_tok.dtype)
        z_b = torch.zeros((B,), device=h_tok.device, dtype=h_tok.dtype)

        st1 = self.core.forward_step(
            ai_streams=ai_init, pr_stream=pr_init, init_slots=init_slots, cov_prev=cov_0,
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h_pool_0, g_sup=g_sup,
            t_val=z_b, disagree_dt=z_b, abs_b_prev=z_b, return_diagnostics=return_diagnostics,
        )
        ro1 = self._readout_step(st1["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, zc_eff, st1["step_phi"])
        u_t1 = self._predict_utility(
            ro1["emb_ai"], torch.zeros_like(ro1["emb_ai"]), ro1["logit_ai"], lg_r1_ai, z_b, g_sup, st1["step_phi"], z_b
        )
        out_s1: dict[str, torch.Tensor] = {
            "logit_ai_r1": lg_r1_ai,
            "logit_promo_r1": lg_r1_pr,
            "logit_ai_t1": ro1["logit_ai"],
            "logit_promo_t1": ro1["logit_promo"],
            "promo_sub4_logit_t1": ro1["promo_sub4_logit"],
            "emb_ai_t1": ro1["emb_ai"],
            "emb_pr_t1": ro1["emb_pr"],
            "u_halt_t1": u_t1,
            "ai_streams": st1["ai_streams"],
            "pr_stream": st1["pr_stream"],
            "slots_t1": st1["slots"],
            "init_slots": init_slots,
            "cov": st1["cov"],
            "k_ev_0": k_ev_0,
            "v_ev_0": v_ev_0,
            "k_haar": k_haar,
            "v_haar": v_haar,
            "haar_mask": haar_mask,
            "h_pool_0": h_pool_0,
            "pr_attr_0": pr_attr_0,
            "st_ai": st_ai,
            "st_pr": st_pr,
            "g_sup": g_sup,
            "_zc_eff": zc_eff,
        }
        if return_diagnostics:
            out_s1["h_tok"] = h_tok
            out_s1["proj_sat_rate_ai"] = st1["proj_sat_rate_ai"]
            out_s1["adaln_gain_abs"] = st1["adaln_gain_abs"]
        return out_s1

    def forward_loop_step(
        self,
        ai_streams: torch.Tensor,
        pr_stream: torch.Tensor,
        init_slots: torch.Tensor,
        cov_prev: torch.Tensor,
        k_ev_0: torch.Tensor,
        v_ev_0: torch.Tensor,
        attention_mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h_pool_0: torch.Tensor,
        pr_attr_0: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        g_sup: torch.Tensor,
        emb_ai_prev: torch.Tensor,
        logit_ai_prev: torch.Tensor,
        t_val: torch.Tensor,
        disagree_dt: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Stage 2 (SINGLE REUSABLE GRAPH): Execute one step `t_val >= 1.0` (`T2, T3, T4, T5, T6...`)."""
        st_n = self.core.forward_step(
            ai_streams=ai_streams, pr_stream=pr_stream, init_slots=init_slots, cov_prev=cov_prev,
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h_pool_0, g_sup=g_sup,
            t_val=t_val, disagree_dt=disagree_dt, abs_b_prev=logit_ai_prev.detach().abs(),
            return_diagnostics=return_diagnostics,
        )
        ro = self._readout_step(st_n["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, z_char, st_n["step_phi"])
        u_t = self._predict_utility(
            ro["emb_ai"], emb_ai_prev, ro["logit_ai"], logit_ai_prev, disagree_dt, g_sup, st_n["step_phi"], t_val
        )
        ret: dict[str, torch.Tensor] = {
            "logit_ai": ro["logit_ai"],
            "logit_promo": ro["logit_promo"],
            "promo_sub4_logit": ro["promo_sub4_logit"],
            "emb_ai": ro["emb_ai"],
            "emb_pr": ro["emb_pr"],
            "u_halt": u_t,
            "ai_streams": st_n["ai_streams"],
            "pr_stream": st_n["pr_stream"],
            "slots": st_n["slots"],
            "cov": st_n["cov"],
        }
        if return_diagnostics:
            ret["proj_sat_rate_ai"] = st_n["proj_sat_rate_ai"]
            ret["adaln_gain_abs"] = st_n["adaln_gain_abs"]
        return ret

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        r0_c1: torch.Tensor,
        r0_c2: torch.Tensor,
        r0_c3: torch.Tensor,
        r0_mask: torch.Tensor,
        r0_feats: torch.Tensor,
        support_pairs: torch.Tensor,
        outer_steps: int = 4,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
        dt_drop_p: float = 0.0,
        use_crossview_dt: bool = False,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Full unrolled training/eval pass returning EVERY step trajectory `logit_ai_steps [T]` & `slots_steps [T]`.

        Note: `disagree_dt` is clipped SIGNED `(clip(a,-6,6) - clip(b,-6,6))` and feeds ONLY the exit gate
        (`_predict_utility`) unless `self.use_slot9_dt=True` (disabled by default across A0..A5).
        """
        B = input_ids.shape[0]
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((B, 1), device=r0_feats.device) >= float(charlm_drop_p)).to(r0_feats.dtype)
            r0_f_eff = torch.cat([r0_feats[:, :R0_STRUCT_DIM_V140], r0_feats[:, R0_STRUCT_DIM_V140:] * lm_keep], dim=-1)
            loc_eff = local_charlm * lm_keep.unsqueeze(-1)
        else:
            r0_f_eff, loc_eff = r0_feats, local_charlm

        r0 = self.forward_r0(r0_c1, r0_c2, r0_c3, r0_mask, r0_f_eff)
        s1 = self.forward_prelude_and_t1(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids, loc_eff,
            z_char=r0["z_char"], support_pairs=support_pairs,
            r0_drop_p=r0_drop_p, charlm_drop_p=0.0, return_diagnostics=return_diagnostics,
        )

        ai_steps = [s1["logit_ai_t1"]]
        pr_steps = [s1["logit_promo_t1"]]
        sub4_steps = [s1["promo_sub4_logit_t1"]]
        emb_ai_steps = [s1["emb_ai_t1"]]
        emb_pr_steps = [s1["emb_pr_t1"]]
        u_halt_steps = [s1["u_halt_t1"]]
        slot_steps = [s1["slots_t1"]]
        sat_steps = [s1["proj_sat_rate_ai"]] if return_diagnostics else []
        adaln_steps = [s1["adaln_gain_abs"]] if return_diagnostics else []

        cur_ai_s = s1["ai_streams"]
        cur_pr_s = s1["pr_stream"]
        cur_cov = s1["cov"]
        cur_emb_ai = s1["emb_ai_t1"]
        cur_lg_ai = s1["logit_ai_t1"]
        la0_det = r0["logit_ai_r0"].detach()

        n_steps = max(1, int(outer_steps))
        for t_idx in range(1, n_steps):
            t_vec = torch.full((B,), float(t_idx), device=input_ids.device, dtype=torch.float32)
            # Clipped SIGNED calibrated logit diff `(a - b)` for exit gate (never enters Slot-9 when `use_slot9_dt=False`!)
            dt_raw = la0_det.clamp(-6.0, 6.0) - cur_lg_ai.detach().clamp(-6.0, 6.0)
            if self.training and dt_drop_p > 0.0:
                dt_keep = (torch.rand((B,), device=dt_raw.device) >= float(dt_drop_p)).to(dt_raw.dtype)
                dt_eff = dt_raw * dt_keep
            elif (not use_crossview_dt) and self.use_slot9_dt:
                dt_eff = torch.zeros((B,), device=input_ids.device, dtype=cur_lg_ai.dtype)
            else:
                dt_eff = dt_raw

            st_k = self.forward_loop_step(
                ai_streams=cur_ai_s, pr_stream=cur_pr_s, init_slots=s1["init_slots"], cov_prev=cur_cov,
                k_ev_0=s1["k_ev_0"], v_ev_0=s1["v_ev_0"], attention_mask=attention_mask,
                k_haar=s1["k_haar"], v_haar=s1["v_haar"], haar_mask=s1["haar_mask"],
                h_pool_0=s1["h_pool_0"], pr_attr_0=s1["pr_attr_0"],
                st_ai=s1["st_ai"], st_pr=s1["st_pr"], z_char=s1["_zc_eff"], g_sup=s1["g_sup"],
                emb_ai_prev=cur_emb_ai, logit_ai_prev=cur_lg_ai, t_val=t_vec, disagree_dt=dt_eff,
                return_diagnostics=return_diagnostics,
            )
            cur_ai_s, cur_pr_s, cur_cov = st_k["ai_streams"], st_k["pr_stream"], st_k["cov"]
            cur_emb_ai, cur_lg_ai = st_k["emb_ai"], st_k["logit_ai"]
            ai_steps.append(st_k["logit_ai"])
            pr_steps.append(st_k["logit_promo"])
            sub4_steps.append(st_k["promo_sub4_logit"])
            emb_ai_steps.append(st_k["emb_ai"])
            emb_pr_steps.append(st_k["emb_pr"])
            u_halt_steps.append(st_k["u_halt"])
            slot_steps.append(st_k["slots"])
            if return_diagnostics:
                sat_steps.append(st_k["proj_sat_rate_ai"])
                adaln_steps.append(st_k["adaln_gain_abs"])

        out: dict[str, Any] = {
            **r0,
            "logit_ai_r1": s1["logit_ai_r1"],
            "logit_promo_r1": s1["logit_promo_r1"],
            "logit_ai_steps": ai_steps,
            "logit_promo_steps": pr_steps,
            "promo_sub4_steps": sub4_steps,
            "emb_ai_steps": emb_ai_steps,
            "emb_pr_steps": emb_pr_steps,
            "u_halt_steps": u_halt_steps,
            "slot_steps": slot_steps,
            "cov_last": cur_cov,
            "g_sup": s1["g_sup"],
            "logit_ai": ai_steps[-1],
            "logit_promo": pr_steps[-1],
            "promo_sub4_logit": sub4_steps[-1],
            "emb_ai": emb_ai_steps[-1],
            "emb_pr": emb_pr_steps[-1],
        }
        if return_diagnostics:
            out["proj_sat_steps"] = sat_steps
            out["adaln_gain_steps"] = adaln_steps
            out["h_tok"] = s1["h_tok"]
            # Zero-cost T3-state -> T4-readout pure-bias control!
            if len(slot_steps) >= 3:
                ro_t3_as_t4 = self.readout_at_step(
                    slot_steps[2], s1["h_pool_0"], s1["pr_attr_0"], s1["st_ai"], s1["st_pr"], s1["_zc_eff"], t_idx=3.0
                )
                out["logit_ai_t3_state_t4_readout"] = ro_t3_as_t4["logit_ai"]
            out["_s1_cache"] = s1
        for ti, (la, lp, uh) in enumerate(zip(ai_steps, pr_steps, u_halt_steps), start=1):
            out[f"logit_ai_t{ti}"] = la
            out[f"logit_promo_t{ti}"] = lp
            out[f"u_halt_t{ti}"] = uh
        return out

    def aggregate_long_chunks(
        self,
        chunk_emb_ai: torch.Tensor, chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor, chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor, chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return doc_ai, doc_pr


# ---------------------------------------------------------------------------
# 6. `MGVZ10` W4/W8 Packer & 4-Stage Modular Pure-FP32 ONNX Export (ONE Reusable `loop_step.onnx`!)
# ---------------------------------------------------------------------------

def _quant_spec_map_v160(model: MeguvizV160StratosphereLoopModel) -> dict[str, tuple[str, int]]:
    spec: dict[str, tuple[str, int]] = {
        "pos_emb": ("w4", 32),
        "core.slot_queries": ("w8", 0),
    }
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantLinearV130, QuantEmbeddingV130)):
            spec[f"{name}.weight"] = ("w8" if mod.target_bits == 8 else "w4", mod.group_size)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 1024:
            spec[f"{name}.weight"] = ("w8", 0)
    return spec


def pack_bundle_v10_v160(model: MeguvizV160StratosphereLoopModel, out_path: Path) -> dict[str, Any]:
    spec_map = _quant_spec_map_v160(model)
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
            breakdown["r0_w8_w4" if k.startswith("rung0.") else "deep_w8"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        elif q_mode == "w4":
            sb, qb = _pack_w4_tensor(arr, gs=gs)
            breakdown["r0_w8_w4" if k.startswith("rung0.") else "deep_w4"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        else:
            sb, qb = b"", arr.astype(np.float16).tobytes()
            breakdown["scales_and_fp16"] += len(qb)
        entries.append({"name": k, "q": q_mode, "g": gs, "shape": list(arr.shape), "s_bytes": len(sb), "q_bytes": len(qb)})
        blobs.extend((sb, qb))

    hdr_bytes = json.dumps({"format": "MGVZ10_STRATOSPHERE_V160", "arch": model.arch, "tensors": entries}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MGVZ10_MAGIC)
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


def load_bundle_v10_v160_into_model(model: MeguvizV160StratosphereLoopModel, mgvz_path: Path) -> None:
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


class _V160R0Export(nn.Module):
    def __init__(self, m: MeguvizV160StratosphereLoopModel) -> None:
        super().__init__()
        self.r0 = m.rung0

    def forward(self, c1, c2, c3, mask, feats):
        o = self.r0(c1, c2, c3, mask, feats)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_ai_r0"], o["p_defer_pr_r0"], o["z_char"]


_T1_OUT_NAMES_V160 = [
    "logit_ai_r1", "logit_promo_r1", "logit_ai_t1", "logit_promo_t1",
    "emb_ai_t1", "emb_pr_t1", "u_halt_t1", "g_sup",
    "ai_streams", "pr_stream", "init_slots", "cov",
    "k_ev_0", "v_ev_0", "k_haar", "v_haar", "haar_mask",
    "h_pool_0", "pr_attr_0", "st_ai", "st_pr",
]


class _V160PreludeT1Export(nn.Module):
    def __init__(self, m: MeguvizV160StratosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, local_charlm, z_char, support_pairs):
        o = self.m.forward_prelude_and_t1(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids, local_charlm, z_char, support_pairs
        )
        return tuple(o[k] for k in _T1_OUT_NAMES_V160)


_STEP_IN_NAMES_V160 = [
    "ai_streams_in", "pr_stream_in", "init_slots", "cov_prev", "k_ev_0", "v_ev_0", "attention_mask",
    "k_haar", "v_haar", "haar_mask", "h_pool_0", "pr_attr_0", "st_ai", "st_pr", "z_char", "g_sup",
    "emb_ai_prev", "logit_ai_prev", "t_val", "disagree_dt",
]
_STEP_OUT_NAMES_V160 = [
    "logit_ai", "logit_promo", "emb_ai", "emb_pr", "u_halt", "ai_streams_out", "pr_stream_out", "cov_out",
]


class _V160LoopStepExport(nn.Module):
    def __init__(self, m: MeguvizV160StratosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, *args):
        o = self.m.forward_loop_step(*args)
        return (
            o["logit_ai"], o["logit_promo"], o["emb_ai"], o["emb_pr"],
            o["u_halt"], o["ai_streams"], o["pr_stream"], o["cov"],
        )


class _V160LongAggExport(nn.Module):
    def __init__(self, m: MeguvizV160StratosphereLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, emb_ai, emb_pr, lg_ai, lg_pr, meta, valid):
        return self.m.aggregate_long_chunks(emb_ai, emb_pr, lg_ai, lg_pr, meta, valid)


def export_v160_bootable_and_onnx(
    model: MeguvizV160StratosphereLoopModel,
    tokenizer_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    version: str,
    charlm: CompactCharLMV150 | None = None,
) -> dict[str, Any]:
    """Export `MGVZ10` + 4 Modular FP32-Exact ONNX Graphs (single reusable `loop_step.onnx`!) + `<= 5.00 MiB` Bootable Zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mgvz_p = out_dir / "bundle_v10_stratosphere_v160.mgvz"
    set_v160_qat_mode(model, False)
    mgvz_info = pack_bundle_v10_v160(model, mgvz_p)

    charlm_p = out_dir / "charlm_v160.bin"
    charlm_info: dict[str, Any] = {"enabled": False, "raw_bytes": 0, "roundtrip_diff": 0.0, "pass": True}
    if charlm is not None:
        sv = charlm.save(charlm_p)
        reloaded = CompactCharLMV150.load(charlm_p)
        reloaded.__class__ = CompactCharLMV150
        probe_cl = ["笑死我了，这个插件装完直接起飞", "总的来说，这项技术具有非常显著的综合优势。"]
        rt_diff = float(np.abs(charlm.extract_9d(probe_cl) - reloaded.extract_9d(probe_cl)).max())
        charlm_info = {"enabled": True, **sv, "roundtrip_diff": round(rt_diff, 8), "pass": bool(rt_diff < 1e-5)}

    m_cpu = MeguvizV160StratosphereLoopModel(
        arch=model.arch, vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim,
        style_era_dropout_p=0.0, dynamics_mode=model.dynamics_mode,
        use_step_phi=model.use_step_phi, step_decay_mode=model.step_decay_mode,
        use_slot9_dt=model.use_slot9_dt, expand_tables=model.expand_tables,
        coverage_beta_enabled=model.coverage_beta_enabled,
        use_local_charlm_inject=model.use_local_charlm_inject,
        use_global_charlm_shortcut=model.use_global_charlm_shortcut,
        prelude_global_se=bool(model.cfg["prelude_global_se"]),
    ).cpu().eval()
    load_bundle_v10_v160_into_model(m_cpu, mgvz_p)

    # 1. R0++
    c1 = torch.randint(1, R0_B1_V130, (2, 24), dtype=torch.long)
    c2 = torch.randint(1, model.r0_b2, (2, 24), dtype=torch.long)
    c3 = torch.randint(1, model.r0_b3, (2, 24), dtype=torch.long)
    cm = torch.ones((2, 24), dtype=torch.float32)
    cf = torch.zeros((2, R0_FEAT_DIM_V140), dtype=torch.float32)
    r0_p = out_dir / "v160_r0_apexchar_fp32.onnx"
    torch.onnx.export(
        _V160R0Export(m_cpu).eval(), (c1, c2, c3, cm, cf), str(r0_p),
        input_names=["c1", "c2", "c3", "mask", "feats"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0", "z_char"],
        dynamic_axes={k: {0: "batch", 1: "clen"} for k in ("c1", "c2", "c3", "mask")}
        | {"feats": {0: "batch"}, "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"}, "p_defer_ai_r0": {0: "batch"}, "p_defer_pr_r0": {0: "batch"}, "z_char": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # 2. Prelude + T1
    ids = torch.randint(1, model.vocab_size, (2, 32), dtype=torch.long)
    msk = torch.ones((2, 32), dtype=torch.long)
    st = torch.zeros((2, model.style_dim), dtype=torch.float32)
    eg = torch.randint(1, model.engram_buckets, (2, 32), dtype=torch.long)
    tg = torch.randint(1, model.trigram_buckets, (2, 32), dtype=torch.long)
    loc_lm = torch.zeros((2, 32, LOCAL_CHARLM_DIM_V150), dtype=torch.float32)
    zc = torch.zeros((2, R0_ZCHAR_DIM_V130), dtype=torch.float32)
    sup = torch.tensor([8.0, 28.0], dtype=torch.float32)

    t1_p = out_dir / "v160_prelude_t1_fp32.onnx"
    t1_dyn = {
        "input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
        "engram_ids": {0: "batch", 1: "seq"}, "trigram_ids": {0: "batch", 1: "seq"},
        "local_charlm": {0: "batch", 1: "seq"},
        "style_feats": {0: "batch"}, "z_char": {0: "batch"}, "support_pairs": {0: "batch"},
    }
    for nm in _T1_OUT_NAMES_V160:
        if nm in ("k_ev_0", "v_ev_0", "cov"):
            t1_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t1_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t1_dyn[nm] = {0: "batch"}

    torch.onnx.export(
        _V160PreludeT1Export(m_cpu).eval(), (ids, msk, st, eg, tg, loc_lm, zc, sup), str(t1_p),
        input_names=["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "local_charlm", "z_char", "support_pairs"],
        output_names=_T1_OUT_NAMES_V160,
        dynamic_axes=t1_dyn, opset_version=17, dynamo=False,
    )

    # 3. SINGLE Reusable `loop_step.onnx` (`t_val = 1.0, 2.0, 3.0, 4.0, 5.0...`)
    with torch.no_grad():
        s1_d = m_cpu.forward_prelude_and_t1(ids, msk, st, eg, tg, loc_lm, zc, sup)
    t_vec = torch.tensor([1.0, 2.0], dtype=torch.float32)
    dt_vec = torch.tensor([0.5, 1.8], dtype=torch.float32)
    step_dummy = (
        s1_d["ai_streams"], s1_d["pr_stream"], s1_d["init_slots"], s1_d["cov"],
        s1_d["k_ev_0"], s1_d["v_ev_0"], msk,
        s1_d["k_haar"], s1_d["v_haar"], s1_d["haar_mask"],
        s1_d["h_pool_0"], s1_d["pr_attr_0"], s1_d["st_ai"], s1_d["st_pr"], zc, s1_d["g_sup"],
        s1_d["emb_ai_t1"], s1_d["logit_ai_t1"], t_vec, dt_vec,
    )
    step_dyn = {
        nm: ({0: "batch", 1: "seq"} if nm in ("cov_prev", "cov_out", "k_ev_0", "v_ev_0", "attention_mask")
             else ({0: "batch", 1: "hseq"} if nm in ("k_haar", "v_haar", "haar_mask") else {0: "batch"}))
        for nm in _STEP_IN_NAMES_V160 + _STEP_OUT_NAMES_V160
    }
    step_p = out_dir / "v160_loop_step_fp32.onnx"
    torch.onnx.export(
        _V160LoopStepExport(m_cpu).eval(), step_dummy, str(step_p),
        input_names=_STEP_IN_NAMES_V160, output_names=_STEP_OUT_NAMES_V160,
        dynamic_axes=step_dyn, opset_version=17, dynamo=False,
    )

    # 4. DynaSeam Long Aggregator
    long_p = out_dir / "v160_dynaseam_fp32.onnx"
    ea, ep = torch.randn(2, 4, AI_END_V130), torch.randn(2, 4, PR_EMB_DIM_V130)
    la, lp = torch.randn(2, 4), torch.randn(2, 4)
    mt, vm = torch.zeros(2, 4, CHUNK_META_DIM_V11), torch.ones(2, 4)
    torch.onnx.export(
        _V160LongAggExport(m_cpu).eval(), (ea, ep, la, lp, mt, vm), str(long_p),
        input_names=["emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid"],
        output_names=["doc_ai", "doc_promo"],
        dynamic_axes={k: {0: "batch", 1: "chunks"} for k in ("emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid")}
        | {"doc_ai": {0: "batch"}, "doc_promo": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # Weight-stripped ONNX skeletons (only 4 graphs -> saves ~260 KiB vs v1.5.0!)
    skel_paths: list[Path] = []
    skel_raw_total = 0
    for op in (r0_p, t1_p, step_p, long_p):
        sp = op.with_suffix(".skel.onnx")
        skel_raw_total += strip_onnx_weights_to_skeleton(op, sp)
        skel_paths.append(sp)

    # Exact FP32 Parity across ALL 4 stages + Multi-Step Loop (`T=1 -> T=2 -> T=4 -> T=6`)!
    s0 = ort.InferenceSession(str(r0_p), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(t1_p), providers=["CPUExecutionProvider"])
    ss = ort.InferenceSession(str(step_p), providers=["CPUExecutionProvider"])
    sl = ort.InferenceSession(str(long_p), providers=["CPUExecutionProvider"])

    def _run(sess: ort.InferenceSession, feed: dict[str, np.ndarray]):
        valid = {i.name for i in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feed.items() if k in valid})

    m_cpu.eval()
    with torch.no_grad():
        pt_r0 = _V160R0Export(m_cpu).eval()(c1, c2, c3, cm, cf)
        or_r0 = _run(s0, {"c1": c1.numpy(), "c2": c2.numpy(), "c3": c3.numpy(), "mask": cm.numpy(), "feats": cf.numpy()})
        r0_diff = float(max(np.abs(pt_r0[i].numpy() - or_r0[i]).max() for i in range(5)))

        pt_t1 = _V160PreludeT1Export(m_cpu)(ids, msk, st, eg, tg, loc_lm, pt_r0[4], sup)
        or_t1 = _run(s1, {
            "input_ids": ids.numpy(), "attention_mask": msk.numpy(), "style_feats": st.numpy(),
            "engram_ids": eg.numpy(), "trigram_ids": tg.numpy(), "local_charlm": loc_lm.numpy(),
            "z_char": or_r0[4], "support_pairs": sup.numpy(),
        })
        t1_diff = float(max(np.abs(pt_t1[i].numpy() - or_t1[i]).max() for i in range(8)))

        # Unroll T=2..6 through the SINGLE reusable `loop_step.onnx` and compare against PyTorch `forward(..., outer_steps=6)`!
        m1 = dict(zip(_T1_OUT_NAMES_V160, or_t1))
        cur_ai, cur_pr, cur_cov = m1["ai_streams"], m1["pr_stream"], m1["cov"]
        cur_emb_ai, cur_lg_ai = m1["emb_ai_t1"], m1["logit_ai_t1"]
        la0_np = np.clip(or_r0[0], -6.0, 6.0)
        pt_full6 = m_cpu(ids, msk, st, eg, tg, loc_lm, c1, c2, c3, cm, cf, sup, outer_steps=6, use_crossview_dt=True)

        step_diffs: dict[str, float] = {}
        for ti in range(1, 6):
            dt_np = (la0_np - np.clip(cur_lg_ai, -6.0, 6.0)).astype(np.float32)
            out_s = dict(zip(_STEP_OUT_NAMES_V160, _run(ss, {
                "ai_streams_in": cur_ai, "pr_stream_in": cur_pr, "init_slots": m1["init_slots"], "cov_prev": cur_cov,
                "k_ev_0": m1["k_ev_0"], "v_ev_0": m1["v_ev_0"], "attention_mask": msk.numpy(),
                "k_haar": m1["k_haar"], "v_haar": m1["v_haar"], "haar_mask": m1["haar_mask"],
                "h_pool_0": m1["h_pool_0"], "pr_attr_0": m1["pr_attr_0"], "st_ai": m1["st_ai"], "st_pr": m1["st_pr"],
                "z_char": or_r0[4], "g_sup": m1["g_sup"], "emb_ai_prev": cur_emb_ai, "logit_ai_prev": cur_lg_ai,
                "t_val": np.full((2,), float(ti), dtype=np.float32), "disagree_dt": dt_np,
            })))
            cur_ai, cur_pr, cur_cov = out_s["ai_streams_out"], out_s["pr_stream_out"], out_s["cov_out"]
            cur_emb_ai, cur_lg_ai = out_s["emb_ai"], out_s["logit_ai"]
            d_k = max(
                float(np.abs(pt_full6[f"logit_ai_t{ti+1}"].numpy() - out_s["logit_ai"]).max()),
                float(np.abs(pt_full6[f"logit_promo_t{ti+1}"].numpy() - out_s["logit_promo"]).max()),
                float(np.abs(pt_full6[f"u_halt_t{ti+1}"].numpy() - out_s["u_halt"]).max()),
            )
            step_diffs[f"T{ti+1}"] = round(d_k, 8)

        # Verify ZERO `Tile` broadcast ops in exported ONNX graphs AND verify `t_val` is live (not baked constant!)
        tile_t1 = sum(1 for nd in onnx.load(str(t1_p)).graph.node if nd.op_type == "Tile")
        tile_st = sum(1 for nd in onnx.load(str(step_p)).graph.node if nd.op_type == "Tile")
        t_val_is_graph_input = "t_val" in {i.name for i in ss.get_inputs()}
        probe_feed = {
            "ai_streams_in": m1["ai_streams"], "pr_stream_in": m1["pr_stream"], "init_slots": m1["init_slots"], "cov_prev": m1["cov"],
            "k_ev_0": m1["k_ev_0"], "v_ev_0": m1["v_ev_0"], "attention_mask": msk.numpy(),
            "k_haar": m1["k_haar"], "v_haar": m1["v_haar"], "haar_mask": m1["haar_mask"],
            "h_pool_0": m1["h_pool_0"], "pr_attr_0": m1["pr_attr_0"], "st_ai": m1["st_ai"], "st_pr": m1["st_pr"],
            "z_char": or_r0[4], "g_sup": m1["g_sup"], "emb_ai_prev": m1["emb_ai_t1"], "logit_ai_prev": m1["logit_ai_t1"],
            "disagree_dt": np.zeros((2,), dtype=np.float32),
        }
        out_t1_s = dict(zip(_STEP_OUT_NAMES_V160, _run(ss, {**probe_feed, "t_val": np.full((2,), 1.0, dtype=np.float32)})))
        out_t3_s = dict(zip(_STEP_OUT_NAMES_V160, _run(ss, {**probe_feed, "t_val": np.full((2,), 3.0, dtype=np.float32)})))
        tval_live_diff = float(np.abs(out_t1_s["ai_streams_out"] - out_t3_s["ai_streams_out"]).max())

        pt_l = _V160LongAggExport(m_cpu)(ea, ep, la, lp, mt, vm)
        or_l = _run(sl, {"emb_ai": ea.numpy(), "emb_pr": ep.numpy(), "lg_ai": la.numpy(), "lg_pr": lp.numpy(), "meta": mt.numpy(), "valid": vm.numpy()})
        long_diff = float(max(np.abs(pt_l[i].numpy() - or_l[i]).max() for i in range(2)))

    tok_out = out_dir / "bili_picky_bpe_12k.json"
    if tokenizer_path.resolve() != tok_out.resolve():
        tok_out.write_bytes(tokenizer_path.read_bytes())

    manifest = dict(manifest)
    manifest["format"] = "MGVZ10_STRATOSPHERE_V160"
    manifest["arch"] = model.arch
    manifest["arch_cfg"] = model.cfg
    manifest["dynamics_mode"] = model.dynamics_mode
    manifest["use_step_phi"] = model.use_step_phi
    manifest["step_decay_mode"] = model.step_decay_mode
    manifest["use_slot9_dt"] = model.use_slot9_dt
    manifest["expand_tables"] = model.expand_tables
    manifest["coverage_beta_enabled"] = model.coverage_beta_enabled
    manifest["use_local_charlm_inject"] = model.use_local_charlm_inject
    manifest["use_global_charlm_shortcut"] = model.use_global_charlm_shortcut
    manifest["mgvz10_info"] = mgvz_info
    manifest["charlm_info"] = charlm_info
    mf_p = out_dir / "manifest.json"
    mf_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    boot_zip = out_dir / f"meguvizcac-{version}-bootable-v10-stratosphere.zip"
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
        "onnx_tile_ops": {"prelude_t1": tile_t1, "loop_step": tile_st},
        "t_val_is_graph_input": bool(t_val_is_graph_input),
        "t_val_live_state_diff": round(tval_live_diff, 6),
        "onnx_exact_pass": bool(max(r0_diff, t1_diff, max_step_diff, long_diff) < 1e-4 and charlm_info["pass"]),
        "mgvz10_info": mgvz_info,
        "charlm_info": charlm_info,
    }


# ---------------------------------------------------------------------------
# 7. Anytime 4-Stage `MeguvizV160CascadeEngine` (Two-Sided OR R0 + T1 Exit + Reusable `loop_step.onnx(t)`)
# ---------------------------------------------------------------------------

class MeguvizV160CascadeEngine:
    """Zero-PII Pre-Canonicalized, Dual-Axis Decoupled Anytime Cascade (`R0++ [2-sided OR] -> T1 -> loop_step(t) -> DynaSeam`)."""

    def __init__(self, release_dir: str | Path, threads: int = 4) -> None:
        self.release_dir = Path(release_dir)
        self.manifest = json.loads((self.release_dir / "manifest.json").read_text(encoding="utf-8"))
        self.arch_cfg = dict(self.manifest.get("arch_cfg") or ARCH_PRESETS_V160[self.manifest.get("arch", "v160_a2")])
        self.tokenizer = BiliPickyBPETokenizer(self.release_dir / "bili_picky_bpe_12k.json")
        charlm_p = self.release_dir / "charlm_v160.bin"
        if not charlm_p.is_file():
            charlm_p = self.release_dir / "charlm_v150.bin"
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
        self.r0_b2 = int(self.arch_cfg.get("r0_b2", R0_B2_V130))
        self.r0_b3 = int(self.arch_cfg.get("r0_b3", R0_B3_V130))
        self.engram_buckets = int(self.arch_cfg.get("engram_buckets", ENGRAM_BUCKETS_V130))
        self.trigram_buckets = int(self.arch_cfg.get("trigram_buckets", TRIGRAM_BUCKETS_V130))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_r0 = ort.InferenceSession(str(self.release_dir / "v160_r0_apexchar_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t1 = ort.InferenceSession(str(self.release_dir / "v160_prelude_t1_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_step = ort.InferenceSession(str(self.release_dir / "v160_loop_step_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_long = ort.InferenceSession(str(self.release_dir / "v160_dynaseam_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self._in_r0 = {i.name for i in self.sess_r0.get_inputs()}
        self._in_t1 = {i.name for i in self.sess_t1.get_inputs()}
        self._in_step = {i.name for i in self.sess_step.get_inputs()}
        self._in_long = {i.name for i in self.sess_long.get_inputs()}

        self.r0_ai_policy = self.manifest.get("rung0_ai_ltt_policy") or {}
        self.r0_pr_policy = self.manifest.get("rung0_pr_ltt_policy") or {}
        self.step_policy = self.manifest.get("anytime_step_policy") or {
            "enabled": True,
            "max_infer_steps": DEFAULT_T_INFER_MAX_V160,
            "u_halt_thresh": 0.015,
            "disagree_dt_thresh": 1.35,
            "uncertain_lo": 0.12,
            "uncertain_hi": 0.88,
            "t1_exit_enabled": True,
        }
        self.crossview_fusion = self.manifest.get("crossview_late_fusion") or {"enabled": False}
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.temp_ai = float(self.manifest.get("temp_ai", 1.0))
        self.temp_pr = float(self.manifest.get("temp_promo", 1.0))

    def _run_sess(self, sess: ort.InferenceSession, valid: set[str], feed: dict[str, np.ndarray]):
        return sess.run(None, {k: v for k, v in feed.items() if k in valid})

    def _prep_deep_inputs(self, clean_texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f56_l, encs, offs_l = [], [], []
        for cl in clean_texts:
            _, f56, _ = extract_style_features_56d(cl)
            ids, offs = encode_bpe_head_tail_with_offsets_v150(self.tokenizer, cl, max_len=self.max_len, head_len=self.head_len)
            f56_l.append(f56)
            encs.append(ids)
            offs_l.append(offs)
        engs = [make_engram_bigram_ids(s, self.engram_buckets) for s in encs]
        trigs = [make_engram_trigram_ids(s, self.trigram_buckets) for s in encs]
        ml = max(2, max(len(s) for s in encs))
        if ml % 2 == 1:
            ml += 1
        bs = len(clean_texts)
        ids_a = np.full((bs, ml), self.tokenizer.pad_id, dtype=np.int64)
        msk_a = np.zeros((bs, ml), dtype=np.int64)
        eg_a = np.zeros((bs, ml), dtype=np.int64)
        tg_a = np.zeros((bs, ml), dtype=np.int64)
        loc_a = np.zeros((bs, ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        for i, s in enumerate(encs):
            n_s = len(s)
            ids_a[i, :n_s], msk_a[i, :n_s], eg_a[i, :n_s], tg_a[i, :n_s] = s, 1, engs[i], trigs[i]
            if self.use_local_charlm and self.charlm is not None:
                loc_a[i, :n_s] = self.charlm.extract_token_local_8ch(clean_texts[i], offs_l[i])
        return ids_a, msk_a, np.stack(f56_l, 0).astype(np.float32), eg_a, tg_a, loc_a

    def _run_anytime_deep(
        self,
        clean_texts: list[str],
        z_char: np.ndarray,
        lg_r0_ai: np.ndarray,
        supports: np.ndarray,
        force_depth: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run Stage 1 (`Prelude+T1`) then batch-compacted `loop_step.onnx(t)` for `t = 2..T_max`!

        AI and Promo exit rungs are tracked independently (`r_ai`, `r_pr`), and AI output is strictly
        latched at `r_ai` so Promo escalation NEVER modifies AI outputs (`F6 === 0.0`!).
        """
        ids, mask, st, eg, tg, loc_lm = self._prep_deep_inputs(clean_texts)
        t1_raw = self._run_sess(self.sess_t1, self._in_t1, {
            "input_ids": ids, "attention_mask": mask, "style_feats": st,
            "engram_ids": eg, "trigram_ids": tg, "local_charlm": loc_lm,
            "z_char": z_char.astype(np.float32), "support_pairs": supports.astype(np.float32),
        })
        t1 = dict(zip(_T1_OUT_NAMES_V160, t1_raw))
        bs = len(clean_texts)

        lg_ai_deep = t1["logit_ai_t1"].copy()
        lg_pr = t1["logit_promo_t1"].copy()
        emb_ai = t1["emb_ai_t1"].copy()
        emb_pr = t1["emb_pr_t1"].copy()
        rungs_ai = np.ones((bs,), dtype=np.int64)
        rungs_pr = np.ones((bs,), dtype=np.int64)

        max_steps = int(force_depth) if force_depth is not None else int(self.step_policy.get("max_infer_steps", DEFAULT_T_INFER_MAX_V160))
        step_en = (force_depth is not None and force_depth >= 2) or (force_depth is None and bool(self.step_policy.get("enabled", True)))
        if not step_en or max_steps <= 1:
            return {
                "logit_ai_deep": lg_ai_deep,
                "logit_ai": apply_crossview_late_fusion(lg_r0_ai, lg_ai_deep, self.crossview_fusion),
                "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr,
                "exit_rung_ai": rungs_ai, "exit_rung_pr": rungs_pr,
            }

        u_thr = float(self.step_policy.get("u_halt_thresh", 0.015))
        dt_thr = float(self.step_policy.get("disagree_dt_thresh", 1.35))
        u_lo = float(self.step_policy.get("uncertain_lo", 0.12))
        u_hi = float(self.step_policy.get("uncertain_hi", 0.88))
        t1_exit_ok = bool(self.step_policy.get("t1_exit_enabled", True))

        pa0_cal = expit(np.clip(lg_r0_ai, -30.0, 30.0))
        if self.lut_ai_r0:
            pa0_cal = apply_spline_temp_lut(pa0_cal, self.lut_ai_r0)
        cal_logit_r0 = clipped_cal_logit(pa0_cal, clip=6.0)

        # Maintain compacted active subset across steps `t = 1 .. max_steps-1` (step 1 = T2, step 2 = T3, ...)
        active_orig = np.arange(bs, dtype=np.int64)
        cur_ai_s = t1["ai_streams"]
        cur_pr_s = t1["pr_stream"]
        cur_cov = t1["cov"]
        cur_emb_ai = t1["emb_ai_t1"]
        cur_lg_ai = t1["logit_ai_t1"]
        cur_u = t1["u_halt_t1"]

        for step_idx in range(1, max_steps):
            if len(active_orig) == 0:
                break
            if force_depth is not None:
                need_ai = np.ones((len(active_orig),), dtype=bool)
            else:
                p_cur = expit(np.clip(cur_lg_ai / max(1e-3, self.temp_ai), -30.0, 30.0))
                p_cur_cal = apply_spline_temp_lut(p_cur, self.lut_ai_full) if self.lut_ai_full else p_cur
                signed_diff = cal_logit_r0[active_orig] - clipped_cal_logit(p_cur_cal, clip=6.0)
                unc = (p_cur >= u_lo) & (p_cur <= u_hi)
                if step_idx == 1 and not t1_exit_ok:
                    need_ai = np.ones((len(active_orig),), dtype=bool)
                else:
                    need_ai = (cur_u >= u_thr) | (np.abs(signed_diff) >= dt_thr) | unc

            sel = np.where(need_ai)[0].astype(np.int64)
            if len(sel) == 0:
                break

            active_orig = active_orig[sel]
            cur_ai_s = cur_ai_s[sel]
            cur_pr_s = cur_pr_s[sel]
            cur_cov = cur_cov[sel]
            cur_emb_ai = cur_emb_ai[sel]
            cur_lg_ai = cur_lg_ai[sel]

            dt_feed = (np.clip(lg_r0_ai[active_orig], -6.0, 6.0) - np.clip(cur_lg_ai, -6.0, 6.0)).astype(np.float32)
            st_out = dict(zip(_STEP_OUT_NAMES_V160, self._run_sess(self.sess_step, self._in_step, {
                "ai_streams_in": cur_ai_s, "pr_stream_in": cur_pr_s,
                "init_slots": t1["init_slots"][active_orig], "cov_prev": cur_cov,
                "k_ev_0": t1["k_ev_0"][active_orig], "v_ev_0": t1["v_ev_0"][active_orig],
                "attention_mask": mask[active_orig],
                "k_haar": t1["k_haar"][active_orig], "v_haar": t1["v_haar"][active_orig], "haar_mask": t1["haar_mask"][active_orig],
                "h_pool_0": t1["h_pool_0"][active_orig], "pr_attr_0": t1["pr_attr_0"][active_orig],
                "st_ai": t1["st_ai"][active_orig], "st_pr": t1["st_pr"][active_orig],
                "z_char": z_char[active_orig].astype(np.float32), "g_sup": t1["g_sup"][active_orig],
                "emb_ai_prev": cur_emb_ai, "logit_ai_prev": cur_lg_ai,
                "t_val": np.full((len(active_orig),), float(step_idx), dtype=np.float32),
                "disagree_dt": dt_feed,
            })))

            cur_ai_s, cur_pr_s, cur_cov = st_out["ai_streams_out"], st_out["pr_stream_out"], st_out["cov_out"]
            cur_emb_ai, cur_lg_ai, cur_u = st_out["emb_ai"], st_out["logit_ai"], st_out["u_halt"]

            lg_ai_deep[active_orig] = cur_lg_ai
            lg_pr[active_orig] = st_out["logit_promo"]
            emb_ai[active_orig] = cur_emb_ai
            emb_pr[active_orig] = st_out["emb_pr"]
            rungs_ai[active_orig] = step_idx + 1
            rungs_pr[active_orig] = step_idx + 1

        return {
            "logit_ai_deep": lg_ai_deep,
            "logit_ai": apply_crossview_late_fusion(lg_r0_ai, lg_ai_deep, self.crossview_fusion),
            "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr,
            "exit_rung_ai": rungs_ai, "exit_rung_pr": rungs_pr,
        }

    def _run_long_doc_clean(self, clean_text: str) -> tuple[float, float]:
        chunks = plan_long_chunks_v11(clean_text, max_chunks=8)
        ch_cleans = [c["text"] for c in chunks] if len(chunks) > 1 else [clean_text]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
            ch_cleans, b2=self.r0_b2, b3=self.r0_b3, already_canonical=True, charlm=self.charlm
        )
        la0, lp0, _, _, zc = self._run_sess(self.sess_r0, self._in_r0, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
        od = self._run_anytime_deep(ch_cleans, zc, la0, sups, force_depth=None)
        if len(chunks) <= 1:
            return float(od["logit_ai"][0]), float(od["logit_promo"][0])
        metas = np.stack([c["chunk_meta"] for c in chunks], axis=0).astype(np.float32)[None, :, :]
        valid = np.ones((1, len(chunks)), dtype=np.float32)
        doc_ai, doc_pr = self._run_sess(self.sess_long, self._in_long, {
            "emb_ai": od["emb_ai"][None, :, :], "emb_pr": od["emb_pr"][None, :, :],
            "lg_ai": od["logit_ai"][None, :], "lg_pr": od["logit_promo"][None, :],
            "meta": metas, "valid": valid,
        })
        return float(doc_ai[0]), float(doc_pr[0])

    def predict_batch(
        self,
        texts: list[str],
        max_batch: int = 32,
        force_full: bool = False,
        force_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """P0-Hardened Entry: Strictly pre-canonicalizes ONCE + Two-Sided OR LTT R0 + Independent AI/Promo Latching!"""
        n = len(texts)
        if n == 0:
            return []
        cleans: list[str] = []
        spoofs = np.zeros((n,), dtype=np.float32)
        for i, raw in enumerate(texts):
            cl, sp = canonicalize_zero_pii_v150(str(raw))
            cleans.append(cl)
            if isinstance(sp, dict) and (sp.get("spoof_any", 0.0) or sp.get("zw_count", 0) or sp.get("homoglyph_count", 0)):
                spoofs[i] = 1.0

        results: list[dict[str, Any] | None] = [None] * n
        short_idxs = [i for i, cl in enumerate(cleans) if len(cl) < self.long_char_trigger]
        long_idxs = [i for i, cl in enumerate(cleans) if len(cl) >= self.long_char_trigger]

        for oi in long_idxs:
            lg_a, lg_p = self._run_long_doc_clean(cleans[oi])
            pa = float(expit(np.clip(lg_a / self.temp_ai, -30.0, 30.0)))
            pp = float(expit(np.clip(lg_p / self.temp_pr, -30.0, 30.0)))
            if self.lut_ai_full:
                pa = float(apply_spline_temp_lut(np.array([pa]), self.lut_ai_full)[0])
            if self.lut_pr_full:
                pp = float(apply_spline_temp_lut(np.array([pp]), self.lut_pr_full)[0])
            results[oi] = {"prob_ai": pa, "prob_promo": pp, "exit_rung_ai": 5, "exit_rung_pr": 5, "logit_ai": lg_a, "logit_promo": lg_p}

        order = sorted(short_idxs, key=lambda i: len(cleans[i]))
        ai_en = bool(self.r0_ai_policy.get("enabled", False)) and not force_full and force_depth is None
        pr_en = bool(self.r0_pr_policy.get("enabled", False)) and not force_full and force_depth is None
        ai_tau_lo = float(self.r0_ai_policy.get("tau_ai_lo", self.r0_ai_policy.get("tau_ai", 0.03)))
        ai_tau_hi_en = bool(self.r0_ai_policy.get("hi_exit_enabled", False))
        ai_tau_hi = float(self.r0_ai_policy.get("tau_ai_hi", 0.992))
        ai_def_lo = float(self.r0_ai_policy.get("defer_max", 0.35))
        ai_def_hi = float(self.r0_ai_policy.get("defer_max_hi", 0.15))
        ai_max_ch = int(self.r0_ai_policy.get("max_chars", 180))
        pr_tau = float(self.r0_pr_policy.get("tau_promo", 0.05))
        pr_def = float(self.r0_pr_policy.get("defer_max", 0.35))
        pr_max_ch = int(self.r0_pr_policy.get("max_chars", 180))

        for st in range(0, len(order), max_batch):
            mb = order[st : st + max_batch]
            mb_cleans = [cleans[i] for i in mb]
            c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
                mb_cleans, b2=self.r0_b2, b3=self.r0_b3, already_canonical=True, charlm=self.charlm
            )
            la0, lp0, pdef_ai, pdef_pr, zc = self._run_sess(
                self.sess_r0, self._in_r0, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf}
            )
            pa0 = expit(np.clip(la0, -30.0, 30.0))
            pp0 = expit(np.clip(lp0, -30.0, 30.0))
            if self.lut_ai_r0:
                pa0 = apply_spline_temp_lut(pa0, self.lut_ai_r0)
            if self.lut_pr_r0:
                pp0 = apply_spline_temp_lut(pp0, self.lut_pr_r0)

            ai_latched: dict[int, tuple[float, float]] = {}
            pr_latched: dict[int, tuple[float, float]] = {}
            need_deep: list[int] = []
            for ri, oi in enumerate(mb):
                cl = cleans[oi]
                no_url = ("<URL>" not in cl) and ("http" not in cl.lower())
                lo_ok = float(pdef_ai[ri]) <= ai_def_lo and float(pa0[ri]) <= ai_tau_lo
                hi_ok = ai_tau_hi_en and float(pdef_ai[ri]) <= ai_def_hi and float(pa0[ri]) >= ai_tau_hi
                # Two-Sided OR LTT AI Exit (strictly independent of Promo -> F6 === 0.0!)
                ai_can_exit = ai_en and len(cl) <= ai_max_ch and spoofs[oi] == 0.0 and no_url and (lo_ok or hi_ok)
                pr_can_exit = pr_en and len(cl) <= pr_max_ch and spoofs[oi] == 0.0 and no_url and float(pdef_pr[ri]) <= pr_def and float(pp0[ri]) <= pr_tau
                if ai_can_exit:
                    ai_latched[ri] = (float(pa0[ri]), float(la0[ri]))
                if pr_can_exit:
                    pr_latched[ri] = (float(pp0[ri]), float(lp0[ri]))
                if ai_can_exit and pr_can_exit:
                    results[oi] = {
                        "prob_ai": float(pa0[ri]), "prob_promo": float(pp0[ri]),
                        "exit_rung_ai": 0, "exit_rung_pr": 0,
                        "logit_ai": float(la0[ri]), "logit_promo": float(lp0[ri]),
                    }
                else:
                    need_deep.append(ri)

            if not need_deep:
                continue
            s_idx = np.array(need_deep, dtype=np.int64)
            od = self._run_anytime_deep([mb_cleans[k] for k in need_deep], zc[s_idx], la0[s_idx], sups[s_idx], force_depth=force_depth)
            pa_f = expit(np.clip(od["logit_ai"] / self.temp_ai, -30.0, 30.0))
            pp_f = expit(np.clip(od["logit_promo"] / self.temp_pr, -30.0, 30.0))
            if self.lut_ai_full:
                pa_f = apply_spline_temp_lut(pa_f, self.lut_ai_full)
            if self.lut_pr_full:
                pp_f = apply_spline_temp_lut(pp_f, self.lut_pr_full)
            for j, ri in enumerate(need_deep):
                oi = mb[ri]
                if ri in ai_latched:
                    fin_pa, fin_la, rung_ai = ai_latched[ri][0], ai_latched[ri][1], 0
                else:
                    fin_pa, fin_la, rung_ai = float(pa_f[j]), float(od["logit_ai"][j]), int(od["exit_rung_ai"][j])
                if ri in pr_latched:
                    fin_pp, fin_lp, rung_pr = pr_latched[ri][0], pr_latched[ri][1], 0
                else:
                    fin_pp, fin_lp, rung_pr = float(pp_f[j]), float(od["logit_promo"][j]), int(od["exit_rung_pr"][j])
                results[oi] = {
                    "prob_ai": fin_pa, "prob_promo": fin_pp,
                    "exit_rung_ai": rung_ai, "exit_rung_pr": rung_pr,
                    "logit_ai": fin_la, "logit_promo": fin_lp,
                }
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 8. Automated 20-Gate `v1.6.0 StratosphereLoop` CI Suite
# ---------------------------------------------------------------------------

def run_v160_ci_gates(
    model: MeguvizV160StratosphereLoopModel,
    tokenizer: BiliPickyBPETokenizer,
    export_rep: dict[str, Any] | None = None,
    ai_lut_check: dict[str, Any] | None = None,
    promo_lut_check: dict[str, Any] | None = None,
    charlm: CompactCharLMV150 | None = None,
    release_dir: Path | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    ci: dict[str, Any] = {}

    def _infer(txts: list[str], force_pad_len: int | None = None, outer_steps: int = 6):
        cleans = [canonicalize_zero_pii_v150(t)[0] for t in txts]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
            cleans, b2=model.r0_b2, b3=model.r0_b3, already_canonical=True, charlm=charlm
        )
        f56_l, encs, offs_l = [], [], []
        for cl in cleans:
            _, f56, _ = extract_style_features_56d(cl)
            ids, offs = encode_bpe_head_tail_with_offsets_v150(tokenizer, cl, max_len=model.max_len, head_len=HEAD_LEN_V140)
            f56_l.append(f56)
            encs.append(ids)
            offs_l.append(offs)
        ml = max(2, max(len(s) for s in encs))
        if force_pad_len is not None:
            ml = max(ml, force_pad_len)
        if ml % 2 == 1:
            ml += 1
        ids_a = np.full((len(txts), ml), tokenizer.pad_id, dtype=np.int64)
        msk_a = np.zeros((len(txts), ml), dtype=np.int64)
        eg_a = np.zeros((len(txts), ml), dtype=np.int64)
        tg_a = np.zeros((len(txts), ml), dtype=np.int64)
        loc_a = np.zeros((len(txts), ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        for i, s in enumerate(encs):
            ids_a[i, : len(s)], msk_a[i, : len(s)] = s, 1
            eg_a[i, : len(s)] = make_engram_bigram_ids(s, model.engram_buckets)
            tg_a[i, : len(s)] = make_engram_trigram_ids(s, model.trigram_buckets)
            if charlm is not None:
                loc_a[i, : len(s)] = charlm.extract_token_local_8ch(cleans[i], offs_l[i])
        with torch.no_grad():
            return model(
                torch.from_numpy(ids_a).to(device), torch.from_numpy(msk_a).to(device),
                torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device),
                torch.from_numpy(eg_a).to(device), torch.from_numpy(tg_a).to(device),
                torch.from_numpy(loc_a).to(device),
                torch.from_numpy(c1).to(device), torch.from_numpy(c2).to(device),
                torch.from_numpy(c3).to(device), torch.from_numpy(cm).to(device),
                torch.from_numpy(cf).to(device), torch.from_numpy(sups).to(device),
                outer_steps=outer_steps,
            )

    # 1. V160_C1: Zero-PII across ALL rungs R0..T6 === 0.0!
    oa = _infer(["回复 @测试A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"], outer_steps=6)
    ob = _infer(["回复 @UltraLongRandomUserNameForBoundaryTest_998877665544332211 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"], outer_steps=6)
    pii_keys = ("logit_ai_r0", "p_defer_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "logit_promo", "u_halt_t1", "u_halt_t4")
    pii_d = float(max(abs(float(oa[k][0].item()) - float(ob[k][0].item())) for k in pii_keys))
    ci["V160_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # 2. V160_F6: Structural Promo->AI Firewall under +12 CBM & Promo perturbations across R0..T6 === 0.0!
    t_fw = ["笑死我了，这个插件装完直接起飞[doge]", "关注工种号交流2群 168688884 领全套资料"]
    o_base = _infer(t_fw, outer_steps=6)
    with torch.no_grad():
        saved = (model.cbm_pr_enc.bias.clone(), model.promo_head.bias.clone(), model.r1_promo_head.bias.clone(), model.rung0.promo_head.bias.clone(), model.pr_attr_attn[-1].bias.clone())
        model.cbm_pr_enc.bias.add_(12.0)
        model.promo_head.bias.add_(4.0)
        model.r1_promo_head.bias.add_(4.0)
        model.rung0.promo_head.bias.add_(4.0)
        model.pr_attr_attn[-1].bias.add_(5.0)
        o_pert = _infer(t_fw, outer_steps=6)
        model.cbm_pr_enc.bias.copy_(saved[0])
        model.promo_head.bias.copy_(saved[1])
        model.r1_promo_head.bias.copy_(saved[2])
        model.rung0.promo_head.bias.copy_(saved[3])
        model.pr_attr_attn[-1].bias.copy_(saved[4])
    fw_diff = float(max(
        (o_base[k] - o_pert[k]).abs().max().item()
        for k in ("logit_ai_r0", "p_defer_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "u_halt_t1", "u_halt_t4")
    ))
    ci["V160_F6_model_promo_to_ai_firewall"] = {"max_leak_diff_under_plus12_cbm": fw_diff, "pass": fw_diff == 0.0}

    # 3. V160_P1: Exact Batch-Padding Invariance across R0..T6 (`< 1e-4` at pad=288!)
    t_sh = "笑死我了，这个插件装完直接起飞[doge]"
    o_unp = _infer([t_sh], force_pad_len=None, outer_steps=6)
    o_pad = _infer([t_sh], force_pad_len=288, outer_steps=6)
    p_keys = ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t4", "logit_ai_t6", "logit_promo", "u_halt_t1", "u_halt_t4")
    p_diffs = {k: round(abs(float(o_unp[k][0].item()) - float(o_pad[k][0].item())), 8) for k in p_keys}
    max_pd = max(p_diffs.values())
    ci["V160_P1_pad_invariance_exact"] = {"diffs": p_diffs, "max_diff": max_pd, "pass": bool(max_pd < 1e-4)}

    # 4. V160_S1: Per-Slot Norm Bound `Pi_R` + Cayley Orthogonality
    with torch.no_grad():
        cayley_err = 0.0
        for blk in model.core.blocks:
            g = blk.c1_cayley_gamma
            g2 = g * g
            inv = 1.0 / (1.0 + g2)
            cayley_err = max(cayley_err, float(((((1.0 - g2) * inv) ** 2 + ((2.0 * g) * inv) ** 2) - 1.0).abs().max().item()))
        max_slot_norm = float(max(sl.norm(dim=-1).max().item() for sl in o_base["slot_steps"]))
    ci["V160_S1_per_slot_bound_and_cayley"] = {
        "cayley_unit_err": round(cayley_err, 8),
        "max_slot_norm_t1_t6": round(max_slot_norm, 4),
        "r_cap": model.core.r_cap,
        "pass": bool(cayley_err < 1e-6 and (model.dynamics_mode == "v150_anchor" or max_slot_norm <= model.core.r_cap + 0.5)),
    }

    # 5. V160_S2: Block-Diagonal `3x3` AI-Only Sinkhorn mHC Doubly-Stochastic Invariant across T=1..6
    with torch.no_grad():
        t_all = torch.arange(6, dtype=torch.float32, device=device)
        phi_all = model.core.compute_step_phi(t_all, torch.float32)
        m_log = model.core.mhc_base_log.clamp(-3.0, 3.0).unsqueeze(0) + 0.35 * torch.tanh(model.core.mhc_step_delta(phi_all)).view(6, 3, 3)
        m3 = sinkhorn_nxn_v150(m_log, iters=max(12, MHC_SINKHORN_ITERS_V150))
        r_err = float((m3.sum(dim=-1) - 1.0).abs().max().item())
        c_err = float((m3.sum(dim=-2) - 1.0).abs().max().item())
    ci["V160_S2_mhc_3x3_doubly_stochastic"] = {"row_err": round(r_err, 8), "col_err": round(c_err, 8), "pass": bool(r_err < 1e-4 and c_err < 1e-4)}

    # 5b. V160_RF_IMPULSE: Empirical Impulse Receptive Field Check on Prelude Conv Topology + Zero Pad Leak
    if model.cfg.get("prelude_kind") == "o_l_dwconv":
        with torch.no_grad():
            L_rf, center_rf, valid_rf = 80, 36, 72
            g_rf = torch.Generator(device=device).manual_seed(16005)
            x0 = torch.randn((1, L_rf, model.dim), generator=g_rf, device=device) * 0.5
            x1 = x0.clone()
            x1[:, center_rf, :] = x1[:, center_rf, :] + torch.randn((1, model.dim), generator=g_rf, device=device) * 2.0
            m_rf = torch.ones((1, L_rf), dtype=torch.long, device=device)
            m_rf[:, valid_rf:] = 0  # last 8 tokens are padding!
            saved_se = [b.se for b in model.prelude_blocks]
            for b in model.prelude_blocks:
                b.se = None
            h0_c, h1_c = x0, x1
            for b in model.prelude_blocks:
                h0_c = b(h0_c, m_rf)
                h1_c = b(h1_c, m_rf)
            for b, se_m in zip(model.prelude_blocks, saved_se):
                b.se = se_m
            diff_pos = (h1_c - h0_c).abs().max(dim=-1).values[0]
            nz = torch.where(diff_pos[:valid_rf] > 1e-6)[0].cpu().tolist()
            obs_span = (max(nz) - min(nz) + 1) if nz else 0
            pad_leak = float(diff_pos[valid_rf:].max().item())
            exp_max_span = 1 + 24 * int(model.cfg["prelude_layers"])  # parallel k=7 d={1,2,4}: 25 (1L), 49 (2L)
        ci["V160_RF_IMPULSE_SPAN"] = {
            "prelude_layers": int(model.cfg["prelude_layers"]),
            "observed_conv_span": int(obs_span),
            "expected_max_parallel_span": int(exp_max_span),
            "pad_leak_max": pad_leak,
            "pass": bool(obs_span == exp_max_span and pad_leak == 0.0),
        }

    # 5c. V160_R0_INDEPENDENCE: Verify `b_{t+1}` has EXACT 0.0 dependence on `disagree_dt` when `use_slot9_dt=False`!
    if not model.use_slot9_dt:
        with torch.no_grad():
            s1_ind = model.forward_prelude_and_t1(
                torch.randint(1, 200, (1, 16), device=device), torch.ones((1, 16), dtype=torch.long, device=device),
                torch.zeros((1, model.style_dim), device=device), torch.ones((1, 16), dtype=torch.long, device=device),
                torch.ones((1, 16), dtype=torch.long, device=device), torch.zeros((1, 16, LOCAL_CHARLM_DIM_V150), device=device),
                torch.zeros((1, R0_ZCHAR_DIM_V130), device=device), torch.tensor([24.0], device=device),
            )
            t1_v = torch.tensor([1.0], device=device)
            kw = dict(
                ai_streams=s1_ind["ai_streams"], pr_stream=s1_ind["pr_stream"], init_slots=s1_ind["init_slots"],
                cov_prev=s1_ind["cov"], k_ev_0=s1_ind["k_ev_0"], v_ev_0=s1_ind["v_ev_0"],
                attention_mask=torch.ones((1, 16), dtype=torch.long, device=device),
                k_haar=s1_ind["k_haar"], v_haar=s1_ind["v_haar"], haar_mask=s1_ind["haar_mask"],
                h_pool_0=s1_ind["h_pool_0"], pr_attr_0=s1_ind["pr_attr_0"],
                st_ai=s1_ind["st_ai"], st_pr=s1_ind["st_pr"], z_char=s1_ind["_zc_eff"], g_sup=s1_ind["g_sup"],
                emb_ai_prev=s1_ind["emb_ai_t1"], logit_ai_prev=s1_ind["logit_ai_t1"], t_val=t1_v,
            )
            o_d0 = model.forward_loop_step(**kw, disagree_dt=torch.tensor([0.0], device=device))
            o_d5 = model.forward_loop_step(**kw, disagree_dt=torch.tensor([5.0], device=device))
            r0_ind_diff = float(max(
                (o_d0["logit_ai"] - o_d5["logit_ai"]).abs().max().item(),
                (o_d0["slots"] - o_d5["slots"]).abs().max().item(),
            ))
        ci["V160_R0_INDEPENDENCE_SLOT9_OFF"] = {"max_deep_diff_dt0_vs_dt5": r0_ind_diff, "pass": bool(r0_ind_diff == 0.0)}

    # 6. V160_S3: Coverage PAD Invariant (`pad_tail === 0.0` & `cov <= COV_MAX`)
    cov_pad_tail = float(o_pad["cov_last"][0, 64:].abs().max().item())
    cov_max_obs = float(o_pad["cov_last"].max().item())
    ci["V160_S3_coverage_pad_and_cap"] = {"pad_tail_max": cov_pad_tail, "cov_max": round(cov_max_obs, 4), "pass": bool(cov_pad_tail == 0.0 and cov_max_obs <= COV_MAX_V150 + 1e-5)}

    # 7. V160_S4: Continuous Fourier Extrapolation Stability (`T=6` finite & non-explosive!)
    with torch.no_grad():
        s_t4 = float(o_base["logit_ai_t4"].abs().max().item())
        s_t6 = float(o_base["logit_ai_t6"].abs().max().item())
        finite_ok = all(torch.isfinite(o_base[f"logit_ai_t{k}"]).all().item() for k in range(1, 7))
    ci["V160_S4_fourier_t6_stability"] = {
        "max_abs_t4": round(s_t4, 4), "max_abs_t6": round(s_t6, 4), "finite": bool(finite_ok),
        "pass": bool(finite_ok and s_t6 <= 30.0),
    }

    # 8. V160_L1: SoftBallLogitNorm bound
    with torch.no_grad():
        huge = torch.ones((2, AI_END_V130), device=device) * 100.0
        hb = model.ai_head.project_ball(huge).norm(dim=-1).max().item()
        rc = float(model.ai_head.effective_r_cap().item())
    ci["V160_L1_softball_bound"] = {"huge_ball_norm": round(hb, 4), "r_cap": round(rc, 4), "pass": bool(math.isfinite(hb) and hb <= rc + 1e-3)}

    # 9. V160_C13: Support Gate
    with torch.no_grad():
        g_short = float(model.compute_support_gate(torch.tensor([4.0], device=device)).item())
        g_rich = float(model.compute_support_gate(torch.tensor([36.0], device=device)).item())
    ci["V160_C13_support_gate"] = {"g_at_4": round(g_short, 4), "g_at_36": round(g_rich, 4), "pass": bool(g_short < 0.25 and g_rich > 0.85)}

    # 10 & 11. Bootable `<= 5.00 MiB` + 4-Stage Single-Step ONNX FP32 Exact Parity (`T=1..6`)
    if export_rep is not None:
        ci["V160_B10_bootable_le_5p00_mib"] = {
            "bootable_mib": export_rep["bootable_mib"], "bootable_bytes": export_rep["bootable_bytes"],
            "skeletons_raw_kb": export_rep["skeletons_raw_kb"], "pass": bool(export_rep["bootable_le_5p00_mib"]),
        }
        ci["V160_ONNX_4stage_fp32_parity"] = {
            "r0_diff": export_rep["r0_fp32_max_diff"], "t1_diff": export_rep["t1_fp32_max_diff"],
            "loop_step_diffs": export_rep["loop_step_diffs_t2_to_t6"], "long_diff": export_rep["long_fp32_max_diff"],
            "pass": bool(export_rep["onnx_exact_pass"]),
        }
        tile_ops = export_rep.get("onnx_tile_ops") or {}
        t_live = float(export_rep.get("t_val_live_state_diff", 0.0))
        expect_t_live = (model.dynamics_mode == "v150_anchor") or model.use_step_phi or (model.step_decay_mode == "inv_sqrt")
        ci["V160_ONNX_NO_TILE_AND_LIVE_TVAL"] = {
            "onnx_tile_ops": tile_ops,
            "t_val_is_graph_input": bool(export_rep.get("t_val_is_graph_input", False)),
            "t_val_live_state_diff": t_live,
            "pass": bool(
                tile_ops.get("prelude_t1", 1) == 0
                and tile_ops.get("loop_step", 1) == 0
                and bool(export_rep.get("t_val_is_graph_input", False))
                and ((not expect_t_live) or t_live > 1e-6)
            ),
        }
        # Cold-start bootable zip verification: unpack MGVZ10 straight from bootable <=5.00 MiB zip and verify 0.0 diff!
        boot_z = Path(export_rep["bootable_zip"])
        if boot_z.is_file():
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(boot_z, "r") as zf:
                    zf.extract("bundle_v10_stratosphere_v160.mgvz", td)
                m_boot = MeguvizV160StratosphereLoopModel(
                    arch=model.arch, vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim,
                    style_era_dropout_p=0.0, dynamics_mode=model.dynamics_mode,
                    use_step_phi=model.use_step_phi, step_decay_mode=model.step_decay_mode,
                    use_slot9_dt=model.use_slot9_dt, expand_tables=model.expand_tables,
                    coverage_beta_enabled=model.coverage_beta_enabled,
                    use_local_charlm_inject=model.use_local_charlm_inject,
                    use_global_charlm_shortcut=model.use_global_charlm_shortcut,
                    prelude_global_se=bool(model.cfg["prelude_global_se"]),
                ).cpu().eval()
                load_bundle_v10_v160_into_model(m_boot, Path(td) / "bundle_v10_stratosphere_v160.mgvz")
                with torch.no_grad():
                    c1_b = torch.randint(1, 200, (1, 16), dtype=torch.long)
                    m_b = torch.ones((1, 16), dtype=torch.float32)
                    cf_b = torch.zeros((1, R0_FEAT_DIM_V140), dtype=torch.float32)
                    r0_boot = m_boot.forward_r0(c1_b, c1_b, c1_b, m_b, cf_b)["logit_ai_r0"]
                    m_ref = MeguvizV160StratosphereLoopModel(
                        arch=model.arch, vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim,
                        style_era_dropout_p=0.0, dynamics_mode=model.dynamics_mode,
                        use_step_phi=model.use_step_phi, step_decay_mode=model.step_decay_mode,
                        use_slot9_dt=model.use_slot9_dt, expand_tables=model.expand_tables,
                        coverage_beta_enabled=model.coverage_beta_enabled,
                        use_local_charlm_inject=model.use_local_charlm_inject,
                        use_global_charlm_shortcut=model.use_global_charlm_shortcut,
                        prelude_global_se=bool(model.cfg["prelude_global_se"]),
                    ).cpu().eval()
                    load_bundle_v10_v160_into_model(m_ref, boot_z.parent / "bundle_v10_stratosphere_v160.mgvz")
                    r0_ref = m_ref.forward_r0(c1_b, c1_b, c1_b, m_b, cf_b)["logit_ai_r0"]
                    boot_diff = float((r0_boot - r0_ref).abs().max().item())
            ci["V160_COLD_START_BOOT_PARITY"] = {"boot_zip_mgvz_diff": boot_diff, "pass": bool(boot_diff == 0.0)}

    if ai_lut_check is not None:
        ci["V160_AI_LUT_FLOOR"] = {**ai_lut_check, "pass": bool(ai_lut_check.get("lut_ai_r0_at_001", 1.0) <= 0.025)}
    if promo_lut_check is not None:
        ci["V160_PR_LUT_FLOOR"] = {**promo_lut_check, "pass": bool(promo_lut_check.get("median_promo_wild", 1.0) <= 0.12)}

    # 14..18. End-to-End 4-Stage Cascade CI (C1, F6 decoupled latch, Two-Sided OR R0, T1 Exit, T4/T6 Force Route, LongDoc!)
    if release_dir is not None and (release_dir / "v160_r0_apexchar_fp32.onnx").is_file():
        eng = MeguvizV160CascadeEngine(release_dir, threads=4)

        # E2E C1 across 180 & 360 raw char boundaries
        eng.r0_ai_policy = {"enabled": True, "tau_ai_lo": 0.99, "hi_exit_enabled": False, "defer_max": 0.99, "max_chars": 180}
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.001, "defer_max": 0.001, "max_chars": 180}
        base_body = "这是一条用于验证清洗前置和端到端路由零泄漏的长短边界测试评论。" * 5
        r_a = eng.predict_batch([f"回复 @短名 : {base_body}"], force_full=False)[0]
        r_b = eng.predict_batch([f"回复 @{'超长测试用户名' * 40} : {base_body}"], force_full=False)[0]
        e2e_c1 = max(abs(r_a["prob_ai"] - r_b["prob_ai"]), abs(r_a["prob_promo"] - r_b["prob_promo"]))
        ci["V160_E2E_C1_zero_pii"] = {"max_prob_diff": round(float(e2e_c1), 8), "pass": bool(e2e_c1 == 0.0)}

        # E2E F6 decoupled latch (Promo flips R0 vs Deep; AI stays locked at R0 with 0.0 diff!)
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.99, "defer_max": 0.99, "max_chars": 180}
        r_pe = eng.predict_batch(["笑死我了，这个插件装完直接起飞[doge]"], force_full=False)[0]
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.0, "defer_max": 0.0, "max_chars": 180}
        r_pd = eng.predict_batch(["笑死我了，这个插件装完直接起飞[doge]"], force_full=False)[0]
        e2e_f6 = abs(r_pe["prob_ai"] - r_pd["prob_ai"])
        ci["V160_E2E_F6_decoupled_latch"] = {
            "ai_diff": round(float(e2e_f6), 8),
            "rungs_exit": (r_pe["exit_rung_ai"], r_pe["exit_rung_pr"]),
            "rungs_deep": (r_pd["exit_rung_ai"], r_pd["exit_rung_pr"]),
            "pass": bool(e2e_f6 == 0.0 and r_pd["exit_rung_ai"] == 0 and r_pd["exit_rung_pr"] >= 1),
        }

        # E2E Two-Sided OR R0 Exit check (hi_exit_enabled=True with tau_ai_lo=-1.0, tau_ai_hi=0.0 triggers high-side OR exit!)
        eng.r0_ai_policy = {"enabled": True, "tau_ai_lo": -1.0, "hi_exit_enabled": True, "tau_ai_hi": 0.0, "defer_max": 0.99, "defer_max_hi": 0.99, "max_chars": 180}
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.99, "defer_max": 0.99, "max_chars": 180}
        r_hi = eng.predict_batch(["总的来说，这项技术具有非常显著的综合优势。"], force_full=False)[0]
        ci["V160_E2E_TWO_SIDED_R0_OR"] = {"exit_rung_ai": r_hi["exit_rung_ai"], "pass": bool(r_hi["exit_rung_ai"] == 0)}

        # E2E Force T1, T4, T6 & LongDoc routes
        r_t1 = eng.predict_batch(["测试单轮 T1 极速早退路由。"], force_depth=1)[0]
        r_t4 = eng.predict_batch(["测试单图可重入第四步路由。"], force_depth=4)[0]
        r_t6 = eng.predict_batch(["测试单图可重入第六步外推路由。"], force_depth=6)[0]
        r_ld = eng.predict_batch(["这是一篇用于验证端到端长文分段聚合路由的真实技术长评，核心逻辑清晰且不带模板腔。" * 12], force_full=False)[0]
        ci["V160_E2E_ANYTIME_ROUTES"] = {
            "t1": r_t1["exit_rung_ai"], "t4": r_t4["exit_rung_ai"], "t6": r_t6["exit_rung_ai"], "long": r_ld["exit_rung_ai"],
            "pass": bool(r_t1["exit_rung_ai"] == 1 and r_t4["exit_rung_ai"] == 4 and r_t6["exit_rung_ai"] == 6 and r_ld["exit_rung_ai"] == 5),
        }

    ci["all_ci_passed"] = all(bool(v.get("pass")) for v in ci.values() if isinstance(v, dict))
    return ci
