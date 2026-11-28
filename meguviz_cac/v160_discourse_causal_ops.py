"""MeguvizCAC v1.6.x / v1.7.0 Discourse Attribution, Bounded Path Algebra & Causal Debiasing Ops.

Strict Engineering Implementation of Independent Review 04 Rev. 02 (2026-11-28):
  - P1 (Zero/Ultra-Low Cost FIRST):
      * `CheapD08PositionMoments`: local evidence mass M0, center moment, mid-distance moment
        `sum m_t g_t (r_t - 0.5)^2 / (M0 + eps)` (separates symmetric HMMH vs MHHM!), windowed
        max-pool, and explicit zero-evidence flag. Strictly O(L), pure FP32 Opset-17.
      * `explicit_quote_boundary_fsm`: deterministic bounded-stack / matched-pair quote depth
        extractor excluding quote delimiters themselves from body evidence and tagging
        nested/asymmetric/unclosed quotes as `UNKNOWN` (no fake +/- unbounded bracket claim).
      * `counterfactual_format_and_flip_losses`: zero-inference-cost `L_inv` (format-preserving
        rule/human edits only) + `L_flip` (double-verified stance/intent flip pairs).
  - P2 (Conditional — Division-Free Bounded Scan):
      * `BoundedChunkedAffineScan`: strictly division-free bounded affine recurrence
        `q_t = a_t * q_{t-1} + (1 - a_t) * r_t`, `(A2, B2) o (A1, B1) = (A2*A1, A2*B1 + B2)`
        with fixed chunk size `K=16` local prefix + inter-chunk scan (strict O(L*d) work,
        `torch.where` padding identity `(1, 0)` so padded positions never corrupt valid states).
        Replaces the falsified log-domain `exp(cumsum(log a))` scan (which explodes at step 30!).
  - P3 (Conditional — Bounded MS-EPA & Selective Order-3 Subsequence):
      * `MultiScaleEventPairArea`: reuses axis-private 32-d features (`32->8`, 65K MACs @ L=256)
        with bounded total-variation normalization `F_ij = A_ij / (V^2 + tau^2) in [-1/8, 1/8]`
        computed locally inside overlapping windows `w in {8, 16, 32}` (offsets `0, w/2`).
      * `SelectiveOrder3Subsequence`: exact O(L) scalar tracker `(c_H, c_HM, c_HMH)` without any
        `d^3` tensor (gives `HMMH = 2`, `MHHM = 0`).
  - P4 (Conditional Training-Only Causal Debiasing):
      * `class_conditional_vrex_unbiased`: exact unbiased finite-sample variance correction
        `V_corr = V_hat - sum_e w_e (1 - w_e) s_e^2 / n_e` (NO ReLU clipping!) + explicit
        biased EMA engineering proxy (`EMAGroupRiskTracker`).
      * `linear_cross_covariance_penalty` & `unbiased_u_hsic`: Song et al. (2012) exact 4th-order
        U-statistic HSIC for `m >= 4` with fixed dev-set bandwidth (negative values legal).
  - Full Reproducibility & ONNX Opset-17 ORT Verification:
      * Reproduces ALL 5 independent verification experiments from Review 04 Rev. 02 Page 13,
        AND closes its disclosed gap by exporting pure-FP32 ONNX Opset-17 graphs and measuring
        real ORT CPU latency, standard-domain compliance, `P1` batch/pad drift, and `F6 === 0.0`!
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# P1. Cheap D08 Position Moments & Windowed Max-Pool (Strictly O(L), Pure FP32)
# ---------------------------------------------------------------------------

class CheapD08PositionMoments(nn.Module):
    """Ultra-cheap D08 & sparse-evidence spatial descriptor (Review 04 Section 3.3).

    Given axis-private local features `h: (B, L, d_in)` and boolean/float valid mask `mask: (B, L)`,
    computes non-negative local evidence `g_t >= 0` (C channels) and valid normalized coordinates
    `r_t = (cumsum(m)_t - 0.5) / max(1, n)` strictly on valid tokens:
      - `M0`: total evidence mass `sum m_t g_t`
      - `has_ev`: smooth/explicit zero-evidence flag `M0 / (M0 + tau_m)`
      - `mean_r`: center moment `sum m_t g_t r_t / (M0 + eps)`
      - `mid_dist2`: middle-distance moment `sum m_t g_t (r_t - 0.5)^2 / (M0 + eps)` (separates HMMH vs MHHM!)
      - `wmax_8`, `wmax_16`: fixed-window local mean/peak summaries (strictly masked via `torch.where`).
    Output shape: `(B, 6 * n_channels)`.
    """

    def __init__(self, d_in: int = 32, n_channels: int = 4, eps: float = 1e-4, tau_m: float = 0.25) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.tau_m = float(tau_m)
        self.ev_proj = nn.Linear(self.d_in, self.n_channels)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Strict `torch.where` masking to prevent `0 * NaN = NaN` and padding bias pollution
        m_bool = mask > 0.5
        m = m_bool.to(h.dtype)  # (B, L)
        m_exp = m.unsqueeze(-1)  # (B, L, 1)
        h_clean = torch.where(m_bool.unsqueeze(-1), h, torch.zeros_like(h))

        g_raw = F.softplus(self.ev_proj(h_clean))  # (B, L, C), >= 0
        g = torch.where(m_bool.unsqueeze(-1), g_raw, torch.zeros_like(g_raw))

        # Valid length `n = max(1, sum m_t)` and valid-position coordinate `r_t in (0, 1)`
        n_valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        pos_idx = torch.cumsum(m, dim=1) - 0.5  # (B, L)
        r = torch.where(m_bool, pos_idx / n_valid, torch.zeros_like(pos_idx)).unsqueeze(-1)  # (B, L, 1)

        m0 = g.sum(dim=1)  # (B, C)
        denom = m0 + self.eps
        has_ev = m0 / (m0 + self.tau_m)  # explicit zero-evidence indicator in [0, 1)
        mean_r = (g * r).sum(dim=1) / denom  # (B, C)
        mid_dist2 = (g * (r - 0.5) * (r - 0.5)).sum(dim=1) / denom  # (B, C)

        # Fixed-window local peak (max over valid tokens; empty -> 0.0 explicitly)
        neg_inf = torch.full_like(g, -1e9)
        g_for_max = torch.where(m_bool.unsqueeze(-1), g, neg_inf)
        tok_max = g_for_max.max(dim=1).values
        any_valid = (m.sum(dim=1, keepdim=True) > 0.5)
        tok_max = torch.where(any_valid, tok_max, torch.zeros_like(tok_max))

        # Local 8-token density peak via 1D average pool (zero-padded; valid-rezeroed)
        g_t = g.transpose(1, 2)  # (B, C, L)
        # Pad right to multiple of 8 with zeros so shorter sequences work cleanly
        win_avg = F.avg_pool1d(g_t, kernel_size=8, stride=4, padding=4, count_include_pad=True)
        win_peak = win_avg.max(dim=2).values  # (B, C)
        win_peak = torch.where(any_valid, win_peak, torch.zeros_like(win_peak))

        # Scale-stabilized log-mass `log1p(M0)`
        log_m0 = torch.log1p(m0)
        return torch.cat([log_m0, has_ev, mean_r, mid_dist2, tok_max, win_peak], dim=-1)


# ---------------------------------------------------------------------------
# P1. Explicit Quotation Boundary Finite-State Machine (Bounded Stack + UNKNOWN)
# ---------------------------------------------------------------------------

_QUOTE_PAIRS: dict[str, str] = {
    "「": "」",
    "『": "』",
    "“": "”",
    "【": "】",
    "《": "》",
}
_CLOSE_TO_OPEN: dict[str, str] = {v: k for k, v in _QUOTE_PAIRS.items()}
_SYMMETRIC_QUOTES: set[str] = {'"', "'"}


def explicit_quote_boundary_fsm(
    text: str,
    max_stack_depth: int = 4,
) -> dict[str, Any]:
    """Deterministic bounded-stack quotation scope extractor (Review 04 Section 5.2).

    Produces per-character scope states:
      - `0`: AUTHOR (outside matched quotes, clean stack)
      - `1`: QUOTED (inside matched quotes, excluding quote delimiter characters themselves!)
      - `2`: UNKNOWN (stack overflow, mismatched closing quote, unclosed quote tail, or
             ambiguous asymmetric/unpaired symmetric quotes)
      - `3`: DELIMITER (quote delimiter character itself — excluded from body evidence!)
    """
    n = len(text)
    states = [0] * n
    depths = [0] * n
    stack: list[tuple[str, int]] = []  # (expected_close_char, open_pos)
    unknown_flag = False
    sym_open_pos: dict[str, int | None] = {q: None for q in _SYMMETRIC_QUOTES}

    for i, ch in enumerate(text):
        if ch in _QUOTE_PAIRS:
            states[i] = 3  # delimiter itself excluded from body evidence
            if len(stack) >= max_stack_depth:
                unknown_flag = True
                depths[i] = max_stack_depth
            else:
                stack.append((_QUOTE_PAIRS[ch], i))
                depths[i] = len(stack)
        elif ch in _CLOSE_TO_OPEN:
            states[i] = 3
            if stack and stack[-1][0] == ch:
                stack.pop()
                depths[i] = len(stack)
            else:
                # Mismatched closing delimiter -> mark whole span ambiguous/unknown
                unknown_flag = True
                depths[i] = len(stack)
        elif ch in _SYMMETRIC_QUOTES:
            states[i] = 3
            if sym_open_pos[ch] is None:
                if len(stack) >= max_stack_depth:
                    unknown_flag = True
                else:
                    sym_open_pos[ch] = i
                    stack.append((ch, i))
            else:
                if stack and stack[-1][0] == ch:
                    stack.pop()
                    sym_open_pos[ch] = None
                else:
                    unknown_flag = True
            depths[i] = len(stack)
        else:
            depths[i] = len(stack)
            if unknown_flag:
                states[i] = 2
            elif len(stack) > 0:
                states[i] = 1
            else:
                states[i] = 0

    # Any unclosed quotes at end of string -> mark characters from earliest unclosed opener as UNKNOWN
    if stack:
        unknown_flag = True
        first_unclosed = min(pos for _, pos in stack)
        for i in range(first_unclosed + 1, n):
            if states[i] != 3:
                states[i] = 2

    return {
        "states": states,  # 0=AUTHOR, 1=QUOTED, 2=UNKNOWN, 3=DELIMITER
        "depths": depths,
        "has_unknown_scope": bool(unknown_flag),
        "n_author_chars": sum(1 for s in states if s == 0),
        "n_quoted_chars": sum(1 for s in states if s == 1),
        "n_unknown_chars": sum(1 for s in states if s == 2),
        "n_delim_chars": sum(1 for s in states if s == 3),
    }


# ---------------------------------------------------------------------------
# P2. Division-Free Bounded Chunked Affine Scan (Strict O(L*d), Opset-17 Safe!)
# ---------------------------------------------------------------------------

def bounded_affine_scan_sequential(
    a: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
    q0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference exact sequential bounded affine scan: `q_t = a_t * q_{t-1} + (1 - a_t) * r_t`.

    Padding positions (`mask <= 0.5`) enforce exact identity `(a_t=1, b_t=0)` via `torch.where`.
    """
    bsz, seq_len, dim = a.shape
    m_bool = (mask > 0.5).unsqueeze(-1)
    a_eff = torch.where(m_bool, a.clamp(0.0, 1.0), torch.ones_like(a))
    b_eff = torch.where(m_bool, (1.0 - a_eff) * r.clamp(0.0, 1.0), torch.zeros_like(r))
    q = torch.zeros((bsz, dim), device=a.device, dtype=a.dtype) if q0 is None else q0
    outs: list[torch.Tensor] = []
    for t in range(seq_len):
        q = a_eff[:, t, :] * q + b_eff[:, t, :]
        outs.append(q)
    return torch.stack(outs, dim=1)


@torch.jit.script
def _dynamic_inter_chunk_scan(sum_a: torch.Tensor, sum_b: torch.Tensor) -> torch.Tensor:
    """Dynamic-length O(N) exclusive inter-chunk affine scan (`N = ceil(L / K)`), exports cleanly to Opset-17 Loop!"""
    bsz = sum_a.shape[0]
    n_chunks = sum_a.shape[1]
    dim = sum_a.shape[2]
    q_in = torch.zeros((bsz, n_chunks, dim), device=sum_a.device, dtype=sum_a.dtype)
    q_carry = torch.zeros((bsz, dim), device=sum_a.device, dtype=sum_a.dtype)
    for j in range(n_chunks):
        q_in[:, j, :] = q_carry
        q_carry = sum_a[:, j, :] * q_carry + sum_b[:, j, :]
    return q_in


class BoundedChunkedAffineScan(nn.Module):
    """Two-level division-free bounded affine scan (`K=16` chunk prefix + dynamic inter-chunk scan, Review 04 Sec 4.2-4.3).

    Properties:
      - Zero divisions, zero `exp/log`: mathematically preserves `q_t in [0, 1]^d` and
        `0 <= A <= 1, 0 <= B <= 1 - A` on every interval — NEVER overflows or NaNs!
      - Strictly `O(L * d)` total work (chunk-local vectorized across all `N = ceil(L / K)` chunks,
        followed by dynamic `N`-step inter-chunk scan + vectorized fill-back `q = A * q_in + B`).
      - Exports cleanly to standard pure-FP32 ONNX Opset-17 for ANY dynamic `B` and `L`!
      - Supports bidirectional mode (forward for source scope, reverse for post-hoc rebuttal/stance).
    """

    def __init__(self, chunk_size: int = 16, reverse: bool = False) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.reverse = bool(reverse)

    def forward(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args: `a, r: (B, L, D)` in `[0, 1]`, `mask: (B, L)` -> `q: (B, L, D)` in `[0, 1]`."""
        bsz = a.shape[0]
        seq_len = a.shape[1]
        dim = a.shape[2]
        k = self.chunk_size
        m_bool = (mask > 0.5).unsqueeze(-1)  # (B, L, 1)

        # Clean valid gates and enforce identity `(a=1, b=0)` on padding positions
        a_eff = torch.where(m_bool, a.clamp(0.0, 1.0), torch.ones_like(a))
        b_eff = torch.where(m_bool, (1.0 - a_eff) * r.clamp(0.0, 1.0), torch.zeros_like(r))

        if self.reverse:
            a_eff = torch.flip(a_eff, dims=[1])
            b_eff = torch.flip(b_eff, dims=[1])

        # Dynamic-safe padding to exact multiple of `K` with NO Python `if` branch!
        pad_extra = (k - (seq_len % k)) % k
        a_pad = torch.ones((bsz, pad_extra, dim), device=a.device, dtype=a.dtype)
        b_pad = torch.zeros((bsz, pad_extra, dim), device=a.device, dtype=a.dtype)
        a_work = torch.cat([a_eff, a_pad], dim=1)
        b_work = torch.cat([b_eff, b_pad], dim=1)

        n_chunks = a_work.shape[1] // k
        a_blk = a_work.reshape(bsz, n_chunks, k, dim)
        b_blk = b_work.reshape(bsz, n_chunks, k, dim)

        # Level 1: Intra-chunk local prefix `(A_{j,t}, B_{j,t})` vectorized across all `n_chunks`!
        cur_a = a_blk[:, :, 0, :]
        cur_b = b_blk[:, :, 0, :]
        pref_a = [cur_a]
        pref_b = [cur_b]
        for t in range(1, k):
            at = a_blk[:, :, t, :]
            bt = b_blk[:, :, t, :]
            cur_a = at * cur_a
            cur_b = at * cur_b + bt
            pref_a.append(cur_a)
            pref_b.append(cur_b)
        blk_pref_a = torch.stack(pref_a, dim=2)  # (B, N, K, D)
        blk_pref_b = torch.stack(pref_b, dim=2)  # (B, N, K, D)

        # Chunk summaries are the terminal elements `(A_j, B_j)`
        sum_a = pref_a[-1]  # (B, N, D)
        sum_b = pref_b[-1]  # (B, N, D)

        # Level 2: Dynamic exclusive inter-chunk scan over `N = ceil(L/K)` chunk summaries
        q_in = _dynamic_inter_chunk_scan(sum_a, sum_b).unsqueeze(2)  # (B, N, 1, D)

        # Vectorized fill-back `q_{j,t} = A_{j,t} * q_{j,in} + B_{j,t}`
        q_full = (blk_pref_a * q_in + blk_pref_b).reshape(bsz, n_chunks * k, dim)
        q_out = q_full[:, :seq_len, :]
        if self.reverse:
            q_out = torch.flip(q_out, dims=[1])
        # Re-zero padded positions explicitly so downstream layers never see carry-over padding state
        q_out = torch.where(m_bool, q_out, torch.zeros_like(q_out))
        return q_out


class BoundedBidirectionalDSA(nn.Module):
    """P2 Candidate: Axis-private Source/Stance/Intent Bounded Bidirectional DSA with per-branch mass `M_j`."""

    def __init__(self, d_in: int = 32, d_state: int = 4, chunk_size: int = 16, eps: float = 1e-4) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_state = int(d_state)
        self.eps = float(eps)
        self.gate_fwd = nn.Linear(self.d_in, 2 * self.d_state)
        self.gate_bwd = nn.Linear(self.d_in, 2 * self.d_state)
        self.scan_fwd = BoundedChunkedAffineScan(chunk_size=chunk_size, reverse=False)
        self.scan_bwd = BoundedChunkedAffineScan(chunk_size=chunk_size, reverse=True)
        # Source classifier over `[h_t, q_fwd, q_bwd]` -> 3-way `(author, quoted, unknown)`
        self.src_head = nn.Linear(self.d_in + 2 * self.d_state, 3)
        self.ev_head = nn.Linear(self.d_in, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns concatenated `[M_a, M_q, M_u, v_a, v_q, v_u]` of shape `(B, 3 + 3 * d_in)`."""
        m_bool = mask > 0.5
        m = m_bool.to(h.dtype).unsqueeze(-1)
        h_clean = torch.where(m_bool.unsqueeze(-1), h, torch.zeros_like(h))

        gf = torch.sigmoid(self.gate_fwd(h_clean))
        gb = torch.sigmoid(self.gate_bwd(h_clean))
        q_fwd = self.scan_fwd(gf[..., : self.d_state], gf[..., self.d_state :], mask)
        q_bwd = self.scan_bwd(gb[..., : self.d_state], gb[..., self.d_state :], mask)

        src_logits = self.src_head(torch.cat([h_clean, q_fwd, q_bwd], dim=-1))
        src_w = torch.softmax(src_logits, dim=-1)  # (B, L, 3): author, quoted, unknown
        g = F.softplus(self.ev_head(h_clean))  # (B, L, 1), >= 0
        w_eff = torch.where(m_bool.unsqueeze(-1), src_w * g, torch.zeros_like(src_w))  # (B, L, 3)

        masses = w_eff.sum(dim=1)  # (B, 3)
        # Per-branch conditional pooling `v_j = sum(w_{t,j} h_t) / (M_j + eps)`, empty branch -> 0
        v_list: list[torch.Tensor] = []
        for j in range(3):
            wj = w_eff[:, :, j : j + 1]  # (B, L, 1)
            mj = masses[:, j : j + 1]  # (B, 1)
            vj = (wj * h_clean).sum(dim=1) / (mj + self.eps)
            vj = torch.where(mj > 1e-6, vj, torch.zeros_like(vj))
            v_list.append(vj)
        return torch.cat([torch.log1p(masses)] + v_list, dim=-1)


# ---------------------------------------------------------------------------
# P3. Conditional Bounded MS-EPA & Selective Order-3 Subsequence (No d^3 Tensor!)
# ---------------------------------------------------------------------------

class MultiScaleEventPairArea(nn.Module):
    """P3 Conditional: Bounded Total-Variation Multi-Scale Event-Pair Area (`MS-EPA`, Review 04 Sec 3.1-3.2).

    Reuses axis-private low-dim features (`d_in=32 -> d_ev=8`, only `65K MACs` @ `L=256`) with
    bounded `tanh` event projections and strictly bounded normalized area:
        `V = sum_{t in W} ||u_t||_1`, `|A_ij| <= V^2 / 8`, `F_ij = A_ij / (V^2 + tau^2) in [-1/8, 1/8]`.
    Computed locally inside fixed non-overlapping windows `w in (8, 16, 32)` (offsets `0` and `w/2`)
    plus one global pass — ZERO long-prefix cancellation error, strictly `O(L * K)`!
    """

    def __init__(
        self,
        d_in: int = 32,
        d_ev: int = 8,
        pairs: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)),
        windows: tuple[int, ...] = (8, 16, 32),
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_ev = int(d_ev)
        self.pairs = tuple(pairs)
        self.windows = tuple(windows)
        self.tau2 = float(tau) ** 2
        self.ev_proj = nn.Linear(self.d_in, self.d_ev)
        self.register_buffer("pair_i", torch.tensor([p[0] for p in self.pairs], dtype=torch.long), persistent=False)
        self.register_buffer("pair_j", torch.tensor([p[1] for p in self.pairs], dtype=torch.long), persistent=False)

    def _window_area_bounded(self, u: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bounded area `F: (..., K)` and non-empty flag `non_empty: (..., 1)` along axis `-2`."""
        # u: (..., W, d_ev), m: (..., W)
        p_incl = torch.cumsum(u, dim=-2)
        p_prev = p_incl - u  # exact old-state prefix `p_{t-1}`!
        ui = torch.index_select(u, dim=-1, index=self.pair_i)
        uj = torch.index_select(u, dim=-1, index=self.pair_j)
        pi = torch.index_select(p_prev, dim=-1, index=self.pair_i)
        pj = torch.index_select(p_prev, dim=-1, index=self.pair_j)
        # Exact antisymmetric Lévy area `A_ij = 0.5 * sum_t (p_{t-1,i} u_{t,j} - p_{t-1,j} u_{t,i})`
        area = 0.5 * (pi * uj - pj * ui).sum(dim=-2)  # (..., K)
        v_tv = u.abs().sum(dim=-1).sum(dim=-1, keepdim=True)  # (..., 1)
        f_norm = area / (v_tv * v_tv + self.tau2)  # guaranteed in [-1/8, 1/8]!
        non_empty = (m.sum(dim=-1, keepdim=True) > 0.5)
        f_norm = torch.where(non_empty, f_norm, torch.zeros_like(f_norm))
        return f_norm, non_empty

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns global bounded area + per-scale max-positive & max-negative area + log-TV summary."""
        bsz, seq_len, _ = h.shape
        m_bool = mask > 0.5
        m = m_bool.to(h.dtype)
        h_clean = torch.where(m_bool.unsqueeze(-1), h, torch.zeros_like(h))
        # Bounded event increments `u_t in [-1, 1]^{d_ev}` (prevents V^2 FP32 overflow)
        u = torch.where(m_bool.unsqueeze(-1), torch.tanh(self.ev_proj(h_clean)), torch.zeros((bsz, seq_len, self.d_ev), device=h.device, dtype=h.dtype))

        feats: list[torch.Tensor] = []
        # Global bounded area + global log-TV
        f_glob, _ = self._window_area_bounded(u, m)
        v_glob = torch.log1p(u.abs().sum(dim=-1).sum(dim=-1, keepdim=True))
        feats.extend([f_glob, v_glob])

        # Local fixed-scale windows with dual offsets `0` and `w // 2` (100% dynamic-length ONNX safe — ZERO Python `if`!)
        for w in self.windows:
            pos_peaks: list[torch.Tensor] = []
            neg_peaks: list[torch.Tensor] = []
            # Ensure length >= w so `offset = w // 2` always yields >= 1 window even for tiny sequences!
            pad_min = torch.clamp(torch.tensor(w, device=h.device) - seq_len, min=0)
            u_base = torch.cat([u, torch.zeros((bsz, pad_min, self.d_ev), device=h.device, dtype=h.dtype)], dim=1)
            m_base = torch.cat([m, torch.zeros((bsz, pad_min), device=h.device, dtype=h.dtype)], dim=1)
            for offset in (0, w // 2):
                u_s = u_base[:, offset:, :]
                m_s = m_base[:, offset:]
                pad_w = (w - (u_s.shape[1] % w)) % w
                u_s = torch.cat([u_s, torch.zeros((bsz, pad_w, self.d_ev), device=h.device, dtype=h.dtype)], dim=1)
                m_s = torch.cat([m_s, torch.zeros((bsz, pad_w), device=h.device, dtype=h.dtype)], dim=1)
                nw = u_s.shape[1] // w
                f_w, ne_w = self._window_area_bounded(u_s.reshape(bsz, nw, w, self.d_ev), m_s.reshape(bsz, nw, w))
                # Separate positive and negative pooling, strictly masking out empty windows!
                f_pos = torch.where(ne_w, F.relu(f_w), torch.zeros_like(f_w)).max(dim=1).values
                f_neg = torch.where(ne_w, F.relu(-f_w), torch.zeros_like(f_w)).max(dim=1).values
                pos_peaks.append(f_pos)
                neg_peaks.append(f_neg)
            feats.append(torch.maximum(pos_peaks[0], pos_peaks[1]))
            feats.append(torch.maximum(neg_peaks[0], neg_peaks[1]))

        return torch.cat(feats, dim=-1)


class SelectiveOrder3CumSum(nn.Module):
    """P1/P3 O(L) Zero-Loop Selective 3-Event Order Statistic (Review 04-B Sec 3.2).

    Uses ONLY 2 exclusive `CumSum` ops + elementwise mul + sum (NO ONNX `Loop`!):
      - Symmetric sandwich `(H -> G -> H)`:
        `P_H(t) = sum_{s < t} h_s`, `R_H(t) = sum_{s > t} h_s`, `U_HGH = sum_t g_t P_H(t) R_H(t)`
        Normalized via AM-GM (`P_H * R_H <= C_H^2 / 4`): `J_HGH = 4 * U_HGH / (eps + C_G * C_H^2) in [0, 1]`.
        Gives `U = 2` on `HGGH` and `U = 0` on `GHHG`!
      - Asymmetric discourse triple `(E1 -> E2 -> E3)` (e.g. QuoteOpen -> PromoPitch -> Rebuttal):
        `U_123 = sum_t e2_t * P_{E1}(t) * R_{E3}(t)`, `J_123 = U_123 / (eps + C_1 * C_2 * C_3) in [0, 1]`.
    """

    def __init__(self, d_in: int = 32, n_triples: int = 4, eps: float = 1e-4) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.n_triples = int(n_triples)
        self.eps = float(eps)
        # Project to 3 non-negative event streams per triple: (E1, E2, E3)
        self.ev_proj = nn.Linear(self.d_in, 3 * self.n_triples)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m_bool = (mask > 0.5).unsqueeze(-1)
        h_clean = torch.where(m_bool, h, torch.zeros_like(h))
        # Bounded non-negative events in [0, 1] via sigmoid -> safe from L^3 overflow at L <= 512
        ev = torch.where(m_bool, torch.sigmoid(self.ev_proj(h_clean)), torch.zeros((h.shape[0], h.shape[1], 3 * self.n_triples), device=h.device, dtype=h.dtype))
        e1, e2, e3 = torch.chunk(ev.float(), 3, dim=-1)  # each (B, L, K), FP32 for exact P1!

        # Exclusive prefix sum `P(t) = sum_{s < t}` and exclusive suffix sum `R(t) = sum_{s > t}` (single cumsum per stream!)
        c1_tot = e1.sum(dim=1, keepdim=True)  # (B, 1, K)
        c2_tot = e2.sum(dim=1, keepdim=True)
        c3_tot = e3.sum(dim=1, keepdim=True)

        cs1 = torch.cumsum(e1, dim=1)
        cs3 = torch.cumsum(e3, dim=1)
        p1 = cs1 - e1  # exclusive prefix of E1
        # Exclusive suffix of E1 (for symmetric E1 -> E2 -> E1) & E3 (for asymmetric E1 -> E2 -> E3)
        r1 = (c1_tot - cs1).clamp_min(0.0)
        r3 = (c3_tot - cs3).clamp_min(0.0)

        u_sym = (e2 * p1 * r1).sum(dim=1)  # (B, K)
        j_sym = (4.0 * u_sym) / (self.eps + c2_tot.squeeze(1) * c1_tot.squeeze(1).pow(2))

        u_asym = (e2 * p1 * r3).sum(dim=1)  # (B, K)
        j_asym = u_asym / (self.eps + c1_tot.squeeze(1) * c2_tot.squeeze(1) * c3_tot.squeeze(1))

        # Include log total masses so zero-activity noise cannot masquerade as strong evidence
        return torch.cat([j_sym, j_asym, torch.log1p(c1_tot.squeeze(1)), torch.log1p(c2_tot.squeeze(1)), torch.log1p(c3_tot.squeeze(1))], dim=-1).to(h.dtype)


def softplus_cvar_tail_loss(
    logits: torch.Tensor,
    y_bin: torch.Tensor,
    env_ids: torch.Tensor | None = None,
    alpha: float = 0.05,
    margin: float = 1.5,
    lambda_tail: float = 0.25,
    lambda_pair: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Shift-safe Non-Negative Negative-Class CVaR + Proper BCE + Pairwise Margin Loss (Review 04-B Sec 9.2).

    Replaces raw-logit CVaR (which suffers trivial shift-degeneracy `CVaR(z - c) = CVaR(z) - c`)
    with `l0 = softplus(z)` Rockafellar-Uryasev upper-tail CVaR anchored to proper BCE & pairwise ranking!
    """
    y_f = y_bin.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, y_f)
    neg_mask = y_bin == 0
    pos_mask = y_bin == 1

    def _ru_cvar_upper(vals: torch.Tensor, a: float) -> torch.Tensor:
        if vals.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        # Empirical Rockafellar-Uryasev min_tau [tau + (1/a) mean((v - tau)_+)] attained at (1-a)-quantile!
        q_idx = max(0, min(vals.numel() - 1, int(math.floor((1.0 - a) * vals.numel()))))
        v_sorted, _ = torch.sort(vals)
        tau = v_sorted[q_idx].detach()
        return tau + F.relu(vals - tau).mean() / max(float(a), 1e-6)

    l0 = F.softplus(logits)
    if env_ids is None:
        max_te = _ru_cvar_upper(l0[neg_mask], alpha)
    else:
        env_cvars = []
        for e in torch.unique(env_ids[neg_mask]):
            em = neg_mask & (env_ids == e)
            if int(em.sum().item()) >= 2:
                env_cvars.append(_ru_cvar_upper(l0[em], alpha))
        max_te = torch.stack(env_cvars).max() if env_cvars else torch.zeros((), device=logits.device, dtype=logits.dtype)

    if pos_mask.any() and neg_mask.any():
        z_pos = logits[pos_mask]
        z_neg = logits[neg_mask]
        # Pairwise softplus margin `E[softplus(m + z_- - z_+)]`
        diff = float(margin) + z_neg.unsqueeze(1) - z_pos.unsqueeze(0)
        pair_loss = F.softplus(diff).mean()
    else:
        pair_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    total = bce + float(lambda_tail) * max_te + float(lambda_pair) * pair_loss
    return {"total": total, "bce": bce, "max_env_softplus_cvar": max_te, "pairwise_margin": pair_loss}


def selective_order3_subsequence_hmh(
    h_ev: np.ndarray,
    m_ev: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Exact O(L) ordered 3-tuple subsequence count `(c_H, c_HM, c_HMH)` (Review 04 Sec 3.4).

    Updates strictly using OLD state: `c_HMH += c_HM * h_t`, `c_HM += c_H * m_t`, `c_H += h_t`.
    For `HMMH`: returns `(2, 2, 2)`; for `MHHM`: returns `(2, 2, 0)`!
    """
    h = np.asarray(h_ev, dtype=np.float64)
    m = np.asarray(m_ev, dtype=np.float64)
    if valid_mask is not None:
        vm = (np.asarray(valid_mask) > 0.5).astype(np.float64)
        h = h * vm
        m = m * vm
    c_h = 0.0
    c_hm = 0.0
    c_hmh = 0.0
    for ht, mt in zip(h, m):
        c_hmh = c_hmh + c_hm * ht
        c_hm = c_hm + c_h * mt
        c_h = c_h + ht
    return c_h, c_hm, c_hmh


# ---------------------------------------------------------------------------
# P1/P4. Training-Only Losses: Minimal Counterfactuals, Unbiased CC-V-REx & U-HSIC
# ---------------------------------------------------------------------------

def counterfactual_format_and_flip_losses(
    z_ai_orig: torch.Tensor | None = None,
    z_ai_fmt: torch.Tensor | None = None,
    z_pr_pos: torch.Tensor | None = None,
    z_pr_neg: torch.Tensor | None = None,
    flip_margin: float = 1.5,
) -> dict[str, torch.Tensor]:
    """P1 Zero-Inference-Cost Counterfactual Losses (Review 04 Sec 8.3).

      - `L_inv = mean((z_AI(x) - z_AI(T_fmt x))^2)` for rule/human label-preserving format edits.
      - `L_flip = mean(ReLU(gamma - z_PR(x+) + z_PR(x-)))` for double-verified stance/intent flips.
    """
    out: dict[str, torch.Tensor] = {}
    if z_ai_orig is not None and z_ai_fmt is not None:
        out["loss_inv"] = ((z_ai_orig - z_ai_fmt) ** 2).mean()
    if z_pr_pos is not None and z_pr_neg is not None:
        out["loss_flip"] = F.relu(float(flip_margin) - z_pr_pos + z_pr_neg).mean()
    return out


def class_conditional_vrex_unbiased(
    per_sample_loss: torch.Tensor,
    y_bin: torch.Tensor,
    env_ids: torch.Tensor,
    n_envs: int,
    env_weights: torch.Tensor | None = None,
    beta_0: float = 0.1,
    beta_1: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Class-Conditional V-REx with EXACT unbiased variance correction (Review 04 Sec 7.1-7.2).

    Turbo v3 zero-sync vectorized GPU implementation (ZERO `.item()` / boolean-indexing CUDA syncs!):
    For each class `y in {0, 1}` over supported environments with `n_e >= 2`:
        `V_corr = sum_e w_e (R_hat_e - R_bar)^2 - sum_e w_e (1 - w_e) s_e^2 / n_e`
    where `s_e^2` uses denominator `n_e - 1` and NO ReLU clipping (negative values are legal and
    required for unbiasedness; gradients propagate through `s_e^2` as well!).
    """
    device = per_sample_loss.device
    dtype = per_sample_loss.dtype
    e_cnt = int(n_envs)
    if e_cnt < 2 or per_sample_loss.numel() == 0:
        z0 = torch.zeros((), device=device, dtype=dtype)
        return {"v0_unbiased": z0, "v1_unbiased": z0, "vrex_penalty": z0}

    cls_grid = torch.arange(2, device=device, dtype=y_bin.dtype).view(2, 1, 1)
    env_grid = torch.arange(e_cnt, device=device, dtype=env_ids.dtype).view(1, e_cnt, 1)
    m_ceb = ((y_bin.view(1, 1, -1) == cls_grid) & (env_ids.view(1, 1, -1) == env_grid)).to(dtype)
    n_ce = m_ceb.sum(dim=-1)  # (2, E)
    supp = (n_ce >= 2.0).to(dtype)

    loss_row = per_sample_loss.view(1, 1, -1)
    r_ce = (m_ceb * loss_row).sum(dim=-1) / n_ce.clamp_min(1.0)
    diff = loss_row - r_ce.unsqueeze(-1)
    se2_ce = (m_ceb * diff * diff).sum(dim=-1) / (n_ce - 1.0).clamp_min(1.0)
    s2_over_n = se2_ce / n_ce.clamp_min(1.0)

    if env_weights is None:
        w_raw = supp
    else:
        w_raw = env_weights.to(device=device, dtype=dtype).view(1, e_cnt) * supp
    w_norm = w_raw / w_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    r_bar = (w_norm * r_ce).sum(dim=-1, keepdim=True)
    v_naive = (w_norm * (r_ce - r_bar).pow(2)).sum(dim=-1)
    bias_corr = (w_norm * (1.0 - w_norm) * s2_over_n).sum(dim=-1)
    has_2 = (supp.sum(dim=-1) >= 2.0).to(dtype)
    v_cls = (v_naive - bias_corr) * has_2  # NO ReLU!

    v0 = v_cls[0]
    v1 = v_cls[1]
    total_pen = float(beta_0) * v0 + float(beta_1) * v1
    return {
        "v0_unbiased": v0,
        "v1_unbiased": v1,
        "vrex_penalty": total_pen,
    }


class EMAGroupRiskTracker:
    """Biased engineering EMA surrogate for CC-V-REx (Review 04 Sec 7.3 — explicitly NOT unbiased!)."""

    def __init__(self, n_envs: int, rho: float = 0.98) -> None:
        self.n_envs = int(n_envs)
        self.rho = float(rho)
        self.mu = np.zeros((2, self.n_envs), dtype=np.float64)
        self.counts = np.zeros((2, self.n_envs), dtype=np.int64)

    def surrogate_and_update(
        self,
        per_sample_loss: torch.Tensor,
        y_bin: torch.Tensor,
        env_ids: torch.Tensor,
        beta_0: float = 0.1,
        beta_1: float = 0.05,
    ) -> torch.Tensor:
        """Uses PRIOR-step EMA `sg(mu_e - mu_bar)` to form gradient proxy `2 * sum w_e (mu_e - mu_bar) R_hat_e`."""
        device, dtype = per_sample_loss.device, per_sample_loss.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        for cls_val, beta in ((0, beta_0), (1, beta_1)):
            active = [e for e in range(self.n_envs) if self.counts[cls_val, e] > 0]
            r_cur: dict[int, torch.Tensor] = {}
            for e in range(self.n_envs):
                em = (y_bin == cls_val) & (env_ids == e)
                if int(em.sum().item()) > 0:
                    r_cur[e] = per_sample_loss[em].mean()

            if len(active) >= 2 and r_cur:
                w = 1.0 / float(len(active))
                mu_bar = float(np.mean([self.mu[cls_val, e] for e in active]))
                surr = torch.zeros((), device=device, dtype=dtype)
                for e, re in r_cur.items():
                    if self.counts[cls_val, e] > 0:
                        surr = surr + (2.0 * w * (self.mu[cls_val, e] - mu_bar)) * re
                total = total + float(beta) * surr

            # Update EMA strictly AFTER forming current gradient surrogate
            for e, re in r_cur.items():
                val = float(re.detach().item())
                if self.counts[cls_val, e] == 0:
                    self.mu[cls_val, e] = val
                else:
                    self.mu[cls_val, e] = self.rho * self.mu[cls_val, e] + (1.0 - self.rho) * val
                self.counts[cls_val, e] += 1
        return total


def linear_cross_covariance_penalty(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Cheap P4 baseline: Frobenius squared cross-covariance `||C||_F^2` on low-dim decision subspace."""
    m = z.shape[0]
    if m < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    zc = z - z.mean(dim=0, keepdim=True)
    sc = s - s.mean(dim=0, keepdim=True)
    cov = (zc.T @ sc) / float(m - 1)
    return (cov * cov).sum()


def unbiased_u_hsic(
    z: torch.Tensor,
    s: torch.Tensor,
    sigma_z: float = 1.0,
    sigma_s: float = 1.0,
) -> torch.Tensor:
    """Song et al. (2012) exact 4th-order unbiased U-statistic HSIC (Review 04 Sec 8.2).

    Requires `m >= 4` iid human (`Y = 0`) samples and FIXED dev-set bandwidths `sigma_z, sigma_s`.
    Negative values are mathematically legal and must NOT be clamped!
    """
    m = int(z.shape[0])
    if m < 4:
        raise ValueError(f"Unbiased 4th-order U-HSIC requires m >= 4, got m={m}")
    dz2 = torch.cdist(z, z, p=2.0).pow(2)
    ds2 = torch.cdist(s, s, p=2.0).pow(2)
    k_mat = torch.exp(-dz2 / (2.0 * float(sigma_z) ** 2))
    j_mat = torch.exp(-ds2 / (2.0 * float(sigma_s) ** 2))
    eye = torch.eye(m, device=z.device, dtype=z.dtype)
    k_t = k_mat * (1.0 - eye)
    j_t = j_mat * (1.0 - eye)

    term1 = (k_t * j_t).sum()  # tr(K~ J~) since symmetric
    term2 = (k_t.sum() * j_t.sum()) / float((m - 1) * (m - 2))
    term3 = (2.0 * (k_t @ j_t).sum()) / float(m - 2)
    return (term1 + term2 - term3) / float(m * (m - 3))


# ---------------------------------------------------------------------------
# Full Review 04 Page 13 Reproduction + ONNX Opset-17 ORT P1/F6 Verification!
# ---------------------------------------------------------------------------

def _exact_integer_levy_area(seq: list[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    """Exact integer old-state accumulation `2 * A_HM = sum (p_H * u_M - p_M * u_H)`."""
    ph, pm = 0, 0
    two_a = 0
    for uh, um in seq:
        two_a += ph * um - pm * uh
        ph += uh
        pm += um
    return (ph, pm), two_a


def self_verify_rev04_and_onnx_ort() -> dict[str, Any]:
    """1. Reproduce ALL 5 Page 13 numerical checks; 2. Export pure-FP32 Opset-17 ONNX & audit ORT P1/F6/latency."""
    # --- Check 1: Integer Path Signature & Selective Order-3 ---
    H, M = (1, 0), (0, 1)
    p_hmmh, a2_hmmh = _exact_integer_levy_area([H, M, M, H])
    p_mhhm, a2_mhhm = _exact_integer_levy_area([M, H, H, M])
    _, a2_hm = _exact_integer_levy_area([H, M])
    _, a2_mh = _exact_integer_levy_area([M, H])
    assert p_hmmh == (2, 2) and p_mhhm == (2, 2)
    assert a2_hmmh == 0 and a2_mhhm == 0, "HMMH and MHHM must collide in depth-2 area!"
    assert a2_hm == 1 and a2_mh == -1

    _, _, c3_hmmh = selective_order3_subsequence_hmh([1, 0, 0, 1], [0, 1, 1, 0])
    _, _, c3_mhhm = selective_order3_subsequence_hmh([0, 1, 1, 0], [1, 0, 0, 1])
    assert c3_hmmh == 2.0 and c3_mhhm == 0.0

    # --- Check 2: FP32 Non-Associativity & Log-Domain Scan Blowup vs Chunked Bounded Scan ---
    f1 = np.float32(1.0)
    feps = np.float32(2.0 ** -24)
    fp32_left = float((f1 + feps) + feps)
    fp32_right = float(f1 + (feps + feps))
    assert fp32_left == 1.0 and abs(fp32_right - 1.0000001192092896) < 1e-12

    L_test = 256
    a_c = np.full(L_test, np.float32(0.051), dtype=np.float32)
    b_c = (np.float32(1.0) - a_c) * np.float32(0.7)
    log_g = np.cumsum(np.log(a_c, dtype=np.float32), dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        g_t = np.exp(log_g, dtype=np.float32)
        q_log = g_t * np.cumsum(b_c / g_t, dtype=np.float32)
    nonfin_idx = np.where(~np.isfinite(q_log))[0]
    first_nonfin_1based = int(nonfin_idx[0]) + 1
    n_nonfin = int(nonfin_idx.size)
    assert first_nonfin_1based == 30 and n_nonfin == 227, f"Got first={first_nonfin_1based}, n={n_nonfin}"

    # Bounded chunked scan on the exact same `a=0.051, r=0.7` -> 0 non-finite, converges to 0.7!
    scan_mod = BoundedChunkedAffineScan(chunk_size=16, reverse=False).eval()
    a_t = torch.full((1, L_test, 1), 0.051, dtype=torch.float32)
    r_t = torch.full((1, L_test, 1), 0.7, dtype=torch.float32)
    m_t = torch.ones((1, L_test), dtype=torch.float32)
    with torch.no_grad():
        q_safe = scan_mod(a_t, r_t, m_t)
    assert torch.isfinite(q_safe).all().item() and abs(float(q_safe[0, -1, 0]) - 0.7) < 1e-6

    # Replicate Review 04 Page 5/13 exact seed=427 chunked vs sequential FP32 associativity drift
    rng = np.random.default_rng(427)
    a_427 = rng.uniform(0.05, 1.0, size=(1, 256, 8)).astype(np.float32)
    r_427 = rng.uniform(0.0, 1.0, size=(1, 256, 8)).astype(np.float32)
    m_427 = torch.ones((1, 256), dtype=torch.float32)
    with torch.no_grad():
        q_seq = bounded_affine_scan_sequential(torch.from_numpy(a_427), torch.from_numpy(r_427), m_427)
        q_chk = scan_mod(torch.from_numpy(a_427), torch.from_numpy(r_427), m_427)
    diff_427 = (q_seq - q_chk).abs().numpy()
    n_diff_427 = int((diff_427 > 0.0).sum())
    max_diff_427 = float(diff_427.max())
    assert n_diff_427 == 746 and abs(max_diff_427 - 1.1920929e-7) < 1e-11, f"Got {n_diff_427}, {max_diff_427}"

    # --- Check 3: Exact Enumeration of Unbiased V-REx Variance Correction (2^4 = 16 outcomes) ---
    p0, p1 = Fraction(1, 4), Fraction(3, 4)
    true_var = Fraction(1, 2) * (p0 - Fraction(1, 2)) ** 2 + Fraction(1, 2) * (p1 - Fraction(1, 2)) ** 2
    exp_naive = Fraction(0)
    exp_corr = Fraction(0)
    for x0 in (0, 1):
        for x1 in (0, 1):
            for y0 in (0, 1):
                for y1 in (0, 1):
                    prob = (
                        (p0 if x0 else 1 - p0)
                        * (p0 if x1 else 1 - p0)
                        * (p1 if y0 else 1 - p1)
                        * (p1 if y1 else 1 - p1)
                    )
                    r0 = Fraction(x0 + x1, 2)
                    r1 = Fraction(y0 + y1, 2)
                    s0_sq = (Fraction(x0) - r0) ** 2 + (Fraction(x1) - r0) ** 2  # denom n-1=1
                    s1_sq = (Fraction(y0) - r1) ** 2 + (Fraction(y1) - r1) ** 2
                    rbar = (r0 + r1) / 2
                    vn = Fraction(1, 2) * (r0 - rbar) ** 2 + Fraction(1, 2) * (r1 - rbar) ** 2
                    vc = vn - Fraction(1, 4) * (s0_sq / 2) - Fraction(1, 4) * (s1_sq / 2)
                    exp_naive += prob * vn
                    exp_corr += prob * vc
    assert true_var == Fraction(1, 16) and exp_naive == Fraction(7, 64) and exp_corr == Fraction(1, 16)

    # --- Check 4: Rare-Prior P99 Zero-FP Sample Sizes (Page 11/13) ---
    def _rare_prior_zero_fp_n(pi: float, tpr: float = 0.8, prec: float = 0.99, alpha: float = 0.05) -> tuple[float, int]:
        f_max = (pi * tpr) / (((prec / (1.0 - prec)) * (1.0 - pi)))
        n_req = int(math.ceil(math.log(alpha) / math.log(1.0 - f_max)))
        return f_max, n_req

    f_ai, n_ai = _rare_prior_zero_fp_n(0.03)
    f_pr, n_pr = _rare_prior_zero_fp_n(0.015)
    assert n_ai == 11986 and n_pr == 24343, f"Got {n_ai}, {n_pr}"

    # --- Check 5: Explicit Quote FSM (`CF03` close-quote shift + unclosed UNKNOWN!) ---
    cf03_a = explicit_quote_boundary_fsm("有人说：「这东西还不错。」点击主页领资料，亲测有效。")
    cf03_b = explicit_quote_boundary_fsm("有人说：「这东西还不错。点击主页领资料，亲测有效。」")
    cf03_bad = explicit_quote_boundary_fsm("有人说：「未闭合的引号内容")
    assert cf03_a["n_author_chars"] > cf03_b["n_author_chars"]
    assert cf03_a["n_quoted_chars"] < cf03_b["n_quoted_chars"]
    assert not cf03_a["has_unknown_scope"] and not cf03_b["has_unknown_scope"]
    assert cf03_bad["has_unknown_scope"] and cf03_bad["n_unknown_chars"] > 0

    # --- Check 6: Cheap D08 Mid-Distance Moment separates HMMH vs MHHM! ---
    d08_mod = CheapD08PositionMoments(d_in=4, n_channels=1).eval()
    with torch.no_grad():
        d08_mod.ev_proj.weight.zero_()
        d08_mod.ev_proj.weight[0, 1] = 10.0  # channel 1 = M (AI evidence)
        d08_mod.ev_proj.bias.fill_(-5.0)
        x_hmmh = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]], dtype=torch.float32)
        x_mhhm = torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.float32)
        m4 = torch.ones((1, 4), dtype=torch.float32)
        f_hmmh = d08_mod(x_hmmh, m4)
        f_mhhm = d08_mod(x_mhhm, m4)
        mid2_hmmh = float(f_hmmh[0, 3])
        mid2_mhhm = float(f_mhhm[0, 3])
    assert mid2_mhhm > mid2_hmmh + 0.10, f"Expected MHHM mid_dist2 >> HMMH, got {mid2_mhhm} vs {mid2_hmmh}"

    # --- Check 7: Pure-FP32 ONNX Opset-17 Export + Real ORT CPU Verification (Closes Page 13 Gap!) ---
    import onnx
    import onnxruntime as ort

    class _DualAxisP1P2P3Probe(nn.Module):
        """Axis-private AI/Promo P1+P2+P3 wrapper to test ONNX Opset-17, P1, F6 === 0.0, and CPU latency."""

        def __init__(self) -> None:
            super().__init__()
            self.ai_d08 = CheapD08PositionMoments(d_in=32, n_channels=4)
            self.ai_dsa = BoundedBidirectionalDSA(d_in=32, d_state=4, chunk_size=16)
            self.ai_msepa = MultiScaleEventPairArea(d_in=32, d_ev=8)
            self.pr_d08 = CheapD08PositionMoments(d_in=32, n_channels=4)
            self.pr_dsa = BoundedBidirectionalDSA(d_in=32, d_state=4, chunk_size=16)

        def forward(self, h_ai: torch.Tensor, h_pr: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out_ai = torch.cat([self.ai_d08(h_ai, mask), self.ai_dsa(h_ai, mask), self.ai_msepa(h_ai, mask)], dim=-1)
            out_pr = torch.cat([self.pr_d08(h_pr, mask), self.pr_dsa(h_pr, mask)], dim=-1)
            return out_ai, out_pr

    torch.manual_seed(7)
    probe = _DualAxisP1P2P3Probe().eval()
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "v160_discourse_probe.onnx"
        d_ha = torch.randn(2, 64, 32, dtype=torch.float32)
        d_hp = torch.randn(2, 64, 32, dtype=torch.float32)
        d_m = torch.ones(2, 64, dtype=torch.float32)
        d_m[1, 37:] = 0.0
        torch.onnx.export(
            probe,
            (d_ha, d_hp, d_m),
            str(onnx_path),
            opset_version=17,
            input_names=["h_ai", "h_pr", "mask"],
            output_names=["out_ai", "out_pr"],
            dynamic_axes={
                "h_ai": {0: "B", 1: "L"},
                "h_pr": {0: "B", 1: "L"},
                "mask": {0: "B", 1: "L"},
                "out_ai": {0: "B"},
                "out_pr": {0: "B"},
            },
            dynamo=False,
        )
        om = onnx.load(str(onnx_path))
        onnx.checker.check_model(om)
        domains = sorted({n.domain or "ai.onnx" for n in om.graph.node})
        assert domains == ["ai.onnx"], f"Non-standard ONNX domains: {domains}"

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])

        # Parity check vs PyTorch
        with torch.no_grad():
            pt_ai, pt_pr = probe(d_ha, d_hp, d_m)
        ort_ai, ort_pr = sess.run(None, {"h_ai": d_ha.numpy(), "h_pr": d_hp.numpy(), "mask": d_m.numpy()})
        ort_parity_max = float(max(np.max(np.abs(pt_ai.numpy() - ort_ai)), np.max(np.abs(pt_pr.numpy() - ort_pr))))
        assert ort_parity_max < 1e-5, f"ORT parity failed: {ort_parity_max}"

        # P1 Check: B=1 L=37 (no pad) vs B=32 L=288 (heavy right pad with GARBAGE + NaN in padded slots!)
        rng_p = np.random.default_rng(123)
        h_ai_37 = rng_p.standard_normal((1, 37, 32)).astype(np.float32)
        h_pr_37 = rng_p.standard_normal((1, 37, 32)).astype(np.float32)
        m_37 = np.ones((1, 37), dtype=np.float32)
        ref_ai, ref_pr = sess.run(None, {"h_ai": h_ai_37, "h_pr": h_pr_37, "mask": m_37})

        h_ai_b32 = rng_p.standard_normal((32, 288, 32)).astype(np.float32)
        h_pr_b32 = rng_p.standard_normal((32, 288, 32)).astype(np.float32)
        m_b32 = np.zeros((32, 288), dtype=np.float32)
        h_ai_b32[17, :37, :] = h_ai_37[0]
        h_pr_b32[17, :37, :] = h_pr_37[0]
        h_ai_b32[17, 37:, :] = np.nan  # adversarial NaN in padded region!
        m_b32[17, :37] = 1.0
        for b_idx in range(32):
            if b_idx != 17:
                m_b32[b_idx, : rng_p.integers(10, 200)] = 1.0
        b32_ai, b32_pr = sess.run(None, {"h_ai": h_ai_b32, "h_pr": h_pr_b32, "mask": m_b32})
        assert np.isfinite(b32_ai[17]).all() and np.isfinite(b32_pr[17]).all(), "NaN leaked from padded region!"
        p1_ai_diff = float(np.max(np.abs(ref_ai[0] - b32_ai[17])))
        p1_pr_diff = float(np.max(np.abs(ref_pr[0] - b32_pr[17])))
        assert p1_ai_diff < 1e-5 and p1_pr_diff < 1e-5, f"P1 diff too large: {p1_ai_diff}, {p1_pr_diff}"

        # F6 === 0.0 Check: Perturb Promo private input by +12.0 -> AI output MUST be EXACT 0.0 diff!
        f6_ai_alt, _ = sess.run(None, {"h_ai": h_ai_37, "h_pr": h_pr_37 + np.float32(12.0), "mask": m_37})
        f6_diff = float(np.max(np.abs(ref_ai - f6_ai_alt)))
        assert f6_diff == 0.0, f"F6 violated! diff={f6_diff}"

        # Real CPU single-thread latency benchmark (`B=1, L=256`)
        b1_ha = rng_p.standard_normal((1, 256, 32)).astype(np.float32)
        b1_hp = rng_p.standard_normal((1, 256, 32)).astype(np.float32)
        b1_m = np.ones((1, 256), dtype=np.float32)
        for _ in range(10):
            sess.run(None, {"h_ai": b1_ha, "h_pr": b1_hp, "mask": b1_m})
        t0 = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            sess.run(None, {"h_ai": b1_ha, "h_pr": b1_hp, "mask": b1_m})
        ort_b1_l256_ms = ((time.perf_counter() - t0) * 1000.0) / n_iters

    return {
        "all_rev04_assertions_passed": True,
        "page13_reproductions": {
            "hmmh_vs_mhhm_2nd_order_area": [a2_hmmh, a2_mhhm],
            "hmmh_vs_mhhm_order3_subseq": [c3_hmmh, c3_mhhm],
            "hmmh_vs_mhhm_d08_mid_dist2": [round(mid2_hmmh, 5), round(mid2_mhhm, 5)],
            "hm_vs_mh_area": [a2_hm, a2_mh],
            "fp32_non_assoc": [fp32_left, fp32_right],
            "log_domain_scan_blowup": {"first_nonfin_step": first_nonfin_1based, "n_nonfin_of_256": n_nonfin},
            "chunked_vs_seq_seed427": {"n_diff_of_2048": n_diff_427, "max_abs_diff": max_diff_427},
            "vrex_exact_enum": {"true": str(true_var), "naive": str(exp_naive), "corrected": str(exp_corr)},
            "p99_rare_prior_zero_fp_n": {"ai_3pct": n_ai, "promo_1p5pct": n_pr},
        },
        "onnx_opset17_ort_audit": {
            "domains": domains,
            "ort_pytorch_max_diff": round(ort_parity_max, 8),
            "p1_b1_l37_vs_b32_l288_nanpad_ai_diff": p1_ai_diff,
            "p1_b1_l37_vs_b32_l288_nanpad_pr_diff": p1_pr_diff,
            "f6_promo_plus12_ai_diff": f6_diff,
            "ort_cpu_1t_b1_l256_combined_p1p2p3_ms": round(ort_b1_l256_ms, 4),
        },
    }


if __name__ == "__main__":
    print(json.dumps(self_verify_rev04_and_onnx_ort(), ensure_ascii=False, indent=2))
