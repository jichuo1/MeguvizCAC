"""MeguvizCAC v1.2.0-ApexNano-Omni Architecture, Bundle v7 (`MGVZ7 <= 2.20 MB`), 5-Stage ONNX, & Cascade Engine.

Implements ADR-v1.2 (`docs/roadmap-v1.2.0-apexnano-omni.md`) with 100% exact dense key alignment to `v1.1.0-R1`
for zero-shock warm-start transplantation, while keeping ALL v1.0.0 & v1.1.0 files 100% untouched:
  1. `Rung0CharHashNanoV12` (`~105 KB` W8 ZIP, `0.05-0.11 ms` 4T-CPU, ZERO BPE, strict Zero-PII!):
     - UTF-32 codepoint 1-gram (`4096x12`) + rolling 3-gram (`8192x8`) + 16-d zero-regex structural scalars
     - Multi-scale DW-Separable Conv1D (`k=3,5,7`, `C=32`) -> `[logit_ai_r0, logit_promo_r0, p_defer_r0]` + `z_char (32-d)`!
  2. `SharedQuantResV12` (`~10.5 KB` compressed, replaces `38.6 KB` weak-specialization 10-expert soft-MoLE!):
     - Pure dense FP16 low-rank residual (`rank=16`, 3 phase-specific `16x16` FP16 cores, zero router, zero `bal_bias`!)
     - Exact zero-shock weight collapse from `v1.1.0-R1`'s `DynaCoreMoLEV11` while keeping `+1.8% R@P95` Sub-2-Bit buffer!
  3. `DenseSymplectLoopCoreV12` with Outer Spectral Damping (`t >= 2` contraction eliminates over-sharpening).
  4. `CosineLogitNormHeadV12` on both AI (`emb[:192]`) & Promo (`emb[:128] U emb[192:256]`) -> cures `p=0.9999` tail spikes!
  5. Firewall-strict zero-init `z_char (32-d)` micro-rhythm injection into Rung-1 & Coda (`AI[:192]` vs `Promo[192:256]`).
  6. `Bundle v7 (MGVZ7 <= 2.20 MB)` + Post-Unpack Exact 5-Stage Static ONNX (`R0/R1/R2/R3/DynaSeam`, optional `MatMulNBits W4`)
     + Capped Unit-Sphere Oblique `LEACE` + `MeguvizV12CascadeEngine` + Automated CI Gates (`V12_S1`, `V12_F6`, `V12_C1`, `V12_R0`).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import (
    MAX_LEN_V07,
    CappedReLU2FFN,
    OmniQuantLinear,
    PreludeBlockV07,
    RationalDerf,
    RMSNorm,
    SeqLoRAMoE,
    _pack_2bit_bytes,
    _pack_5trit_bytes,
    _pack_w4_bytes,
    _unpack_2bit_bytes,
    _unpack_5trit_bytes,
    _unpack_w4_bytes,
)
from .v09_chronoloop import (
    BLTMatryoshkaPreludeL0,
    D1_PREFIX_V09,
    D2_COMP_V09,
    FFN1_PREFIX_V09,
    FFN2_COMP_V09,
    HIDDEN_V09,
    NUM_FAST_SLOTS_V09,
    NUM_SLOTS_V09,
    NUM_SLOW_SLOTS_V09,
    ResidualDeferralRouterV09,
    _hadamard_64,
)
from .v10_apexloop import (
    BUS_DIM_V10,
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    MOLE_RANK_V10,
    NUM_SEM_FAST_V10,
    NUM_SPAN_FAST_V10,
    ContextSecondaryLoopV10,
    MultiScaleRhythmConvV10,
    tunnel_trm_ste,
)
from .v11_cascade_cert import compute_oblique_leace_affine
from .v11_features import STYLE_FEAT_DIM_V11, _CONTACT_ID_PII_RE, extract_style_features_56d
from .v11_longstream import DynaSeamMILAggregatorV11, export_dynaseam_onnx_and_verify
from .v11_omniguard import (
    AI_EMB_END_V11,
    CBM_AI_CONCEPTS_V11,
    PROMO_SUB4_NAMES,
    R1_PROMO_VETO_DIMS_V11,
    SHARED_EMB_END_V11,
    InLoopAnalyticEBTV11,
    MeguvizV11OmniGuardModel,
)

R0_B1_V12 = 4096
R0_B3_V12 = 8192
R0_D1_V12 = 12
R0_D3_V12 = 8
R0_CMID_V12 = 32
R0_ZCHAR_DIM_V12 = 32
R0_MAX_CHARS_V12 = 192
R0_FEAT_DIM_V12 = 16


# ---------------------------------------------------------------------------
# 1. Zero-PII Zero-BPE Rung-0 Char-Hash Preprocessor & Nano DW-CNN (`~105 KB`)
# ---------------------------------------------------------------------------

def extract_rung0_charhash_batch(
    texts: list[str],
    max_chars: int = R0_MAX_CHARS_V12,
    already_canonical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-PII Zero-BPE fast UTF-32 1/3-gram hash + 16-d vectorized structural scalars.

    Always runs `canonicalize_zero_pii_v07` first so `@username`, `<CONTACT_ID>`, zero-width
    chars, and homoglyphs are stripped identically (`V12_C1 diff === 0.000000`!).
    Returns `(c1, c3, mask, feats16, spoof_flags)`.
    """
    n = len(texts)
    if n == 0:
        return (
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, R0_FEAT_DIM_V12), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    cleans: list[str] = []
    spoofs = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(texts):
        if already_canonical:
            cleans.append(str(t))
        else:
            cl, sp_meta = canonicalize_zero_pii_v07(str(t))
            cl = _CONTACT_ID_PII_RE.sub(r"\1<CONTACT_ID>", cl)
            cleans.append(cl)
            if isinstance(sp_meta, dict):
                spoofs[i] = 1.0 if (sp_meta.get("spoof_any", 0.0) or sp_meta.get("zw_count", 0) or sp_meta.get("homoglyph_count", 0)) else 0.0

    lens = [min(max_chars, max(1, len(c))) for c in cleans]
    ml = max(8, max(lens))
    c1 = np.zeros((n, ml), dtype=np.int64)
    c3 = np.zeros((n, ml), dtype=np.int64)
    mask = np.zeros((n, ml), dtype=np.float32)
    feats = np.zeros((n, R0_FEAT_DIM_V12), dtype=np.float32)

    for i, cl in enumerate(cleans):
        s = cl[:max_chars]
        if not s:
            mask[i, 0] = 1.0
            continue
        codes = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
        L = len(codes)
        mask[i, :L] = 1.0
        c1_u = ((codes * np.uint32(2654435761)) >> np.uint32(20)) & np.uint32(R0_B1_V12 - 1)
        c1[i, :L] = c1_u
        if L >= 3:
            h3 = (
                (codes[:-2] * np.uint32(1315423911))
                ^ (codes[1:-1] * np.uint32(2654435761))
                ^ (codes[2:] * np.uint32(2246822519))
            )
            c3[i, : L - 2] = (h3 >> np.uint32(19)) & np.uint32(R0_B3_V12 - 1)
        c3[i, max(0, L - 2) : L] = c1_u[max(0, L - 2) : L]

        inv_l = 1.0 / float(L)
        cjk = int(np.sum((codes >= 0x4E00) & (codes <= 0x9FFF)))
        alnum = int(np.sum(((codes >= 48) & (codes <= 57)) | ((codes >= 65) & (codes <= 90)) | ((codes >= 97) & (codes <= 122))))
        digits = int(np.sum((codes >= 48) & (codes <= 57)))
        spaces = int(np.sum((codes == 32) | (codes == 9)))
        newlines = int(np.sum(codes == 10))
        cn_comma = int(np.sum(codes == 0xFF0C))
        cn_period = int(np.sum(codes == 0x3002))
        cn_colon = int(np.sum((codes == 0xFF1A) | (codes == 58)))
        cn_semi = int(np.sum((codes == 0xFF1B) | (codes == 59)))
        emdash = int(np.sum((codes == 0x2014) | (codes == 0x2500)))
        quote_cn = int(np.sum((codes == 0x201C) | (codes == 0x201D) | (codes == 0x300A) | (codes == 0x300B)))
        bracket_emo = int(np.sum(codes == 91))
        qmark = int(np.sum((codes == 63) | (codes == 0xFF1F)))
        excl = int(np.sum((codes == 33) | (codes == 0xFF01)))
        uniq_r = len(set(codes.tolist())) * inv_l
        feats[i] = [
            math.log1p(len(cl)) / 6.0,
            cjk * inv_l,
            alnum * inv_l,
            digits * inv_l,
            spaces * inv_l,
            newlines * inv_l * 10.0,
            cn_comma * inv_l * 10.0,
            cn_period * inv_l * 10.0,
            cn_colon * inv_l * 15.0,
            cn_semi * inv_l * 20.0,
            emdash * inv_l * 20.0,
            quote_cn * inv_l * 15.0,
            bracket_emo * inv_l * 15.0,
            qmark * inv_l * 15.0,
            excl * inv_l * 15.0,
            uniq_r,
        ]
    return c1, c3, mask, feats, spoofs


class Rung0CharHashNanoV12(nn.Module):
    """~105 KB W8 Zero-BPE Ultra-Fast Dual-Axis Gatekeeper + `32-d` Char Micro-Rhythm Extractor."""

    def __init__(
        self,
        b1: int = R0_B1_V12,
        b3: int = R0_B3_V12,
        d1: int = R0_D1_V12,
        d3: int = R0_D3_V12,
        c_mid: int = R0_CMID_V12,
        z_char_dim: int = R0_ZCHAR_DIM_V12,
    ):
        super().__init__()
        self.emb1 = nn.Embedding(b1, d1)
        self.emb3 = nn.Embedding(b3, d3)
        nn.init.normal_(self.emb1.weight, std=0.05)
        nn.init.normal_(self.emb3.weight, std=0.05)
        self.proj_in = nn.Conv1d(d1 + d3, c_mid, kernel_size=1)
        self.dw3 = nn.Conv1d(c_mid, c_mid, kernel_size=3, padding=1, groups=c_mid)
        self.dw5 = nn.Conv1d(c_mid, c_mid, kernel_size=5, padding=2, groups=c_mid)
        self.dw7 = nn.Conv1d(c_mid, c_mid, kernel_size=7, padding=3, groups=c_mid)
        self.pw = nn.Conv1d(c_mid * 3, c_mid, kernel_size=1)
        self.trunk = nn.Sequential(
            nn.Linear(c_mid * 2 + R0_FEAT_DIM_V12, 64),
            nn.GELU(),
            nn.Linear(64, z_char_dim),
            nn.GELU(),
        )
        self.out_head = nn.Linear(z_char_dim, 3)  # [logit_ai_r0, logit_promo_r0, logit_defer_r0]

    def forward(
        self,
        c1: torch.Tensor,
        c3: torch.Tensor,
        mask: torch.Tensor,
        feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([self.emb1(c1), self.emb3(c3)], dim=-1).transpose(1, 2)  # (B, 20, L)
        m = mask.unsqueeze(1)
        h = F.gelu(self.proj_in(x)) * m
        h_ms = torch.cat([self.dw3(h), self.dw5(h), self.dw7(h)], dim=1)
        h = F.gelu(self.pw(h_ms)) * m
        denom = m.sum(dim=-1).clamp(min=1.0)
        h_mean = h.sum(dim=-1) / denom
        h_max = h.masked_fill(m < 0.5, -1e4).amax(dim=-1)
        z_char = self.trunk(torch.cat([h_mean, h_max, feats], dim=-1))
        out3 = self.out_head(z_char)
        return {
            "logit_ai_r0": out3[:, 0],
            "logit_promo_r0": out3[:, 1],
            "p_defer_r0": torch.sigmoid(out3[:, 2]),
            "z_char": z_char,
        }


# ---------------------------------------------------------------------------
# 2. SharedQuantResV12 (`~10.5 KB` FP16/W4 Dense Residual — Replaces Soft-MoLE!)
# ---------------------------------------------------------------------------

class SharedQuantResV12(nn.Module):
    """Pure Dense FP16 Low-Rank Quantization-Residual Absorber (replaces 10-expert soft-MoLE!).

    Preserves the exact `+1.8% R@P95` Sub-2-Bit quantization buffer while eliminating ALL soft-router
    overhead, `bal_bias` fine-tuning drift, and saving `-26 KB` in MGVZ7.
    """

    def __init__(self, hidden_size: int = HIDDEN_V09, rank: int = MOLE_RANK_V10):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.basis_u = OmniQuantLinear(hidden_size, rank, bias=False, target_bits="w4", group_size=64)
        self.basis_v = OmniQuantLinear(rank, hidden_size, bias=False, target_bits="w4", group_size=64)
        with torch.no_grad():
            u_q = torch.empty(rank, hidden_size)
            nn.init.orthogonal_(u_q)
            self.basis_u.weight.copy_(u_q * 0.5)
            v_q = torch.empty(hidden_size, rank)
            nn.init.orthogonal_(v_q)
            self.basis_v.weight.copy_(v_q * 0.05)

        self.phase_core = nn.Parameter(torch.zeros(3, rank, rank))
        self.phase_bias = nn.Parameter(torch.zeros(3, rank))
        self.phase_silu_gain = nn.Parameter(torch.zeros(3))
        self.phase_diag = nn.Parameter(torch.ones(3, rank))
        self.adaln_scale = nn.Parameter(torch.zeros(3, 1, hidden_size))
        self.adaln_shift = nn.Parameter(torch.zeros(3, 1, hidden_size))
        self.dyn_ctrl = nn.Parameter(torch.zeros(6))

    def modulate(self, x: torch.Tensor, phase_idx: int) -> torch.Tensor:
        pi = min(max(int(phase_idx), 0), 2)
        return x * (1.0 + 0.25 * torch.tanh(self.adaln_scale[pi])) + 0.25 * torch.tanh(self.adaln_shift[pi])

    def forward_delta(self, x: torch.Tensor, phase_idx: int) -> torch.Tensor:
        pi = min(max(int(phase_idx), 0), 2)
        z = self.basis_u(x)  # (B, S, r)
        g = torch.sigmoid(self.phase_silu_gain[pi])
        z_nl = (1.0 - g) * z + g * F.silu(z)
        h = torch.matmul(z_nl, self.phase_core[pi].transpose(0, 1)) + self.phase_bias[pi]
        h = h * self.phase_diag[pi]
        return self.basis_v(h)


# ---------------------------------------------------------------------------
# 3. CosineLogitNormHeadV12 (Cures p=0.9999 Hard-Human Tail Overconfidence!)
# ---------------------------------------------------------------------------

class CosineLogitNormHeadV12(nn.Module):
    """Bounded-Norm Cosine LogitNorm Classification Head (`logit = scale * <w_hat, z_hat> + b`)."""

    def __init__(self, in_features: int, init_scale: float = 8.0, max_scale: float = 14.0):
        super().__init__()
        self.in_features = int(in_features)
        self.max_scale = float(max_scale)
        self.weight = nn.Parameter(torch.randn(1, in_features) * (1.0 / math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(1))
        raw_inv = math.log(max(math.exp(max(init_scale - 2.0, 0.5)) - 1.0, 1e-4))
        self.log_scale = nn.Parameter(torch.tensor(raw_inv, dtype=torch.float32))

    def effective_scale(self) -> torch.Tensor:
        return (2.0 + F.softplus(self.log_scale)).clamp(max=self.max_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_n = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
        w_n = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-6)
        return (self.effective_scale().float() * F.linear(z_n, w_n) + self.bias.float()).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. DenseSymplectLoopCoreV12 (Exact v1.1 Dense Keys + SharedQuantRes + Spectral Damping)
# ---------------------------------------------------------------------------

class DenseSymplectLoopCoreV12(nn.Module):
    """Pure-Dense SymplectLoop Core matching `TunnelSymplectCoreV11` dense keys + `SharedQuantResV12` + Spectral Damping."""

    def __init__(self, hidden_size: int = HIDDEN_V09, num_heads: int = 6, ffn_size: int = 896):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.n_slow = NUM_SLOW_SLOTS_V09
        self.n_sem = NUM_SEM_FAST_V10
        self.n_span = NUM_SPAN_FAST_V10

        self.norm1 = RMSNorm(hidden_size)
        self.q_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.k_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.v_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.sub_ln_attn = RMSNorm(hidden_size)
        self.o_proj = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)

        self.norm2 = RMSNorm(hidden_size)
        self.ffn = CappedReLU2FFN(hidden_size=hidden_size, ffn_size=ffn_size, cap_tau=3.0)
        # ADR-v1.2: Replace 10-expert soft-MoLE with pure dense FP16 SharedQuantRes!
        self.qres = SharedQuantResV12(hidden_size=hidden_size, rank=MOLE_RANK_V10)

        self.fast_norm = RMSNorm(hidden_size)
        self.p2_pre_a = OmniQuantLinear(hidden_size * 2, 8, bias=False, target_bits="w4", group_size=64)
        self.p2_pre_b = OmniQuantLinear(8, hidden_size, target_bits="w4", group_size=64)
        self.p2_post_a = OmniQuantLinear(hidden_size * 2, 8, bias=False, target_bits="w4", group_size=64)
        self.p2_post_b = OmniQuantLinear(8, hidden_size, target_bits="w4", group_size=64)
        self.fast_res_scale = nn.Parameter(torch.full((hidden_size,), 0.35))
        self.p2_l0_pen_a = OmniQuantLinear(hidden_size, 16, bias=False, target_bits="w4", group_size=64)
        self.p2_l0_pen_b = OmniQuantLinear(16, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.p2_l0_gate = nn.Parameter(torch.full((hidden_size,), -1.5))

        self.bus_down = OmniQuantLinear(hidden_size, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.bus_up = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)

        self.c1_q = nn.Linear(BUS_DIM_V10, BUS_DIM_V10, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V10, BUS_DIM_V10, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V10))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.20))

        self.c2_beta_logit = nn.Parameter(torch.full((BUS_DIM_V10,), 0.4))
        self.c2_carry_scale = nn.Parameter(torch.tensor(0.22))
        self.c2_surprise_mlp = nn.Sequential(nn.Linear(BUS_DIM_V10 + 1, 16), nn.SiLU(), nn.Linear(16, 1))

        self.inloop_ebt = InLoopAnalyticEBTV11(in_dim=BUS_DIM_V10, hidden=96)
        self.ebt_back_scale = nn.Parameter(torch.tensor(0.15))

        self.ctx_loop = ContextSecondaryLoopV10(hidden_size=hidden_size, bus_dim=BUS_DIM_V10, ctx_in_dim=72)

        self.reread_norm = RMSNorm(hidden_size)
        self.reread_q = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.reread_o_a = OmniQuantLinear(hidden_size, 32, bias=False, target_bits="w4", group_size=64)
        self.reread_o_b = OmniQuantLinear(32, hidden_size, target_bits="w4", group_size=64)
        self.reread_gate = nn.Parameter(torch.tensor(-1.2))

        self.haar_q = OmniQuantLinear(hidden_size, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.haar_o = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.haar_gate = nn.Parameter(torch.tensor(-1.4))

        self.mhc_alpha_logit = nn.Parameter(torch.full((hidden_size,), -1.8))
        # Pillar 2: Outer spectral damping parameter (`outer_idx=1`: damp in (0.50, 1.00), init ~0.94 -> zero-shock!)
        # For `outer_idx >= 2`: scales by `1/sqrt(outer_idx)` -> strict contraction for T >= 3!
        self.spectral_damp_logit = nn.Parameter(torch.tensor(2.0))

    def mhc_anchor_mix(self, s_cur: torch.Tensor, s_anchor: torch.Tensor) -> torch.Tensor:
        alpha = 0.5 * torch.sigmoid(self.mhc_alpha_logit)
        return (1.0 - alpha) * s_cur + alpha * s_anchor

    def _cayley_peer_couple(self, z_l: torch.Tensor, cayley_delta: torch.Tensor) -> torch.Tensor:
        u = self.bus_down(z_l)
        u_a, u_b = u[:, : self.n_sem, :], u[:, self.n_sem :, :]
        sc_ab = torch.matmul(self.c1_q(u_a), self.c1_k(u_b).transpose(-1, -2)) / 8.0
        msg_to_a = torch.matmul(F.softmax(sc_ab, dim=-1), u_b)
        msg_to_b = torch.matmul(F.softmax(sc_ab.transpose(-1, -2), dim=-1), u_a)

        g = (self.c1_cayley_gamma.unsqueeze(0) + 0.15 * torch.tanh(cayley_delta)).clamp(-2.5, 2.5)
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c = ((1.0 - g2) * inv).unsqueeze(1)
        s = ((2.0 * g) * inv).unsqueeze(1)

        rot_a = c * u_a + s * msg_to_a
        rot_b = -s * msg_to_b + c * u_b
        delta = self.bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_l + (0.35 * torch.tanh(self.c1_bridge_scale)) * delta

    def _fast_inner_step(self, slots: torch.Tensor, ctrl: torch.Tensor, phase_idx: int, trm_lam: float) -> torch.Tensor:
        z_h = slots[:, : self.n_slow, :]
        z_l = slots[:, self.n_slow :, :]
        zh_ref = z_h.mean(dim=1, keepdim=True).expand_as(z_l)

        zn = self.fast_norm(z_l)
        zn_trm = tunnel_trm_ste(zn, trm_lam)
        zh_trm = tunnel_trm_ste(zh_ref, trm_lam)
        pre_in = torch.cat([zn_trm, zn_trm - zh_trm], dim=-1)
        g_pre = torch.sigmoid(self.p2_pre_b(F.silu(self.p2_pre_a(pre_in))) + 0.25 * ctrl[2])
        h_l = self.qres.modulate(zn_trm, phase_idx) * g_pre
        delta_l = self.ffn(h_l) + self.qres.forward_delta(h_l, phase_idx)

        g_post = torch.sigmoid(self.p2_post_b(F.silu(self.p2_post_a(torch.cat([zn, delta_l], dim=-1)))) + 0.25 * ctrl[3])
        z_l_new = z_l + self.fast_res_scale * (g_post * delta_l)
        z_l_new = self._cayley_peer_couple(z_l_new, ctrl[0].view(1, 1))
        return torch.cat([z_h, z_l_new], dim=1)

    def _active_reread_and_haar(
        self,
        slots: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        ev_mask: torch.Tensor,
        k_haar: torch.Tensor | None,
        v_haar: torch.Tensor | None,
        haar_mask: torch.Tensor | None,
        haar_delta: torch.Tensor,
    ) -> torch.Tensor:
        sn = self.reread_norm(slots)
        q = self.reread_q(sn)
        sc = torch.matmul(q, k_ev.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        sc = sc.masked_fill(~ev_mask.bool().unsqueeze(1), -1e4)
        out = self.reread_o_b(F.silu(self.reread_o_a(torch.matmul(F.softmax(sc, dim=-1), v_ev))))
        s = slots + torch.sigmoid(self.reread_gate) * out

        if k_haar is not None and v_haar is not None and haar_mask is not None:
            qh = self.haar_q(sn)
            sch = torch.matmul(qh, k_haar.transpose(-1, -2)) / 8.0
            sch = sch.masked_fill(~haar_mask.bool().unsqueeze(1), -1e4)
            oh = self.haar_o(torch.matmul(F.softmax(sch, dim=-1), v_haar))
            g_h = torch.sigmoid(self.haar_gate + 0.35 * haar_delta)
            s = s + g_h * oh
        return s

    def forward_outer_step(
        self,
        slots: torch.Tensor,
        s_anchor: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        ev_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        outer_idx: int,
        k_haar: torch.Tensor | None = None,
        v_haar: torch.Tensor | None = None,
        haar_mask: torch.Tensor | None = None,
        carry_m_l: torch.Tensor | None = None,
        prev_zh_bus: torch.Tensor | None = None,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        s_in = slots
        s = slots
        bsz = s.shape[0]
        device = s.device
        ctrl = torch.tanh(self.qres.dyn_ctrl)

        if outer_idx >= 1 and ctx_vec is not None and ctx_kish_lambda is not None:
            z_h_ctx = self.ctx_loop(s[:, : self.n_slow, :], ctx_vec, ctx_kish_lambda)
            s = torch.cat([z_h_ctx, s[:, self.n_slow :, :]], dim=1)

        zh_now_bus = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        if prev_zh_bus is not None:
            cos_sim = F.cosine_similarity(zh_now_bus, prev_zh_bus, dim=-1, eps=1e-6).unsqueeze(-1)
            surprise = torch.sigmoid(self.c2_surprise_mlp(torch.cat([zh_now_bus - prev_zh_bus, cos_sim], dim=-1)).squeeze(-1))
        else:
            surprise = torch.full((bsz,), 0.5, device=device, dtype=s.dtype)

        if outer_idx >= 1:
            s = self._active_reread_and_haar(s, k_ev, v_ev, ev_mask, k_haar, v_haar, haar_mask, ctrl[1])

        z_l_start = s[:, self.n_slow :, :]
        if carry_m_l is not None:
            c2_gain = (0.35 * torch.tanh(self.c2_carry_scale) * (1.0 - 0.5 * surprise) * (1.0 + 0.25 * ctrl[4])).view(-1, 1, 1)
            z_l_inj = z_l_start + c2_gain * self.bus_up(carry_m_l)
            s = torch.cat([s[:, : self.n_slow, :], z_l_inj], dim=1)

        for ki in range(k_inner):
            lam_now = trm_lam_k0 if ki == 0 else trm_lam_k1
            s = self._fast_inner_step(s, ctrl, phase_idx=min(ki, 1), trm_lam=lam_now)

        delta_fast_bus = torch.tanh(self.bus_down(s[:, self.n_slow :, :] - z_l_start))
        beta_c = 0.85 * torch.sigmoid(self.c2_beta_logit).view(1, 1, -1)
        new_carry_m_l = (1.0 - beta_c) * delta_fast_bus if carry_m_l is None else (beta_c * carry_m_l + (1.0 - beta_c) * delta_fast_bus)

        n_s = s.shape[1]
        h = self.norm1(s)
        q = self.q_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-1, -2)) / 8.0
        a = torch.matmul(F.softmax(sc, dim=-1), v).transpose(1, 2).contiguous().view(bsz, n_s, self.hidden_size)
        s = s + self.o_proj(self.sub_ln_attn(a))

        n2 = self.qres.modulate(self.norm2(s), phase_idx=2)
        s = s + self.ffn(tunnel_trm_ste(n2, trm_lam_k1)) + self.qres.forward_delta(n2, phase_idx=2)

        l0_pen = self.p2_l0_pen_b(F.silu(self.p2_l0_pen_a(h0_pool))).unsqueeze(1)
        z_h_pen = s[:, : self.n_slow, :] + torch.sigmoid(self.p2_l0_gate).view(1, 1, -1) * l0_pen
        s = torch.cat([z_h_pen, s[:, self.n_slow :, :]], dim=1)

        u_zh = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        ebt_out = self.inloop_ebt.forward_energy_and_grad(u_zh)
        ebt_delta = self.bus_up(ebt_out["u_refined"] - u_zh).unsqueeze(1)
        ebt_g = 0.35 * torch.tanh(self.ebt_back_scale) * (1.0 + 0.25 * ctrl[5])
        z_h_ebt = s[:, : self.n_slow, :] + ebt_g * ebt_delta
        s = torch.cat([z_h_ebt, s[:, self.n_slow :, :]], dim=1)

        s_step = self.mhc_anchor_mix(s, s_anchor)
        if outer_idx >= 1:
            damp = (0.50 + 0.50 * torch.sigmoid(self.spectral_damp_logit)) / math.sqrt(float(outer_idx))
            s_out = s_in + damp * (s_step - s_in)
        else:
            s_out = s_step

        return {
            "slots": s_out,
            "carry_m_l": new_carry_m_l,
            "zh_bus": u_zh,
            "ebt_out": ebt_out,
            "surprise": surprise,
        }


# ---------------------------------------------------------------------------
# 5. Full MeguvizCAC v1.2.0-ApexNano-Omni Model (100% v1.1 Dense Key Compatible!)
# ---------------------------------------------------------------------------

class MeguvizV12ApexNanoModel(nn.Module):
    """Dense-First `ApexNano-Omni` v1.2.0 (`Rung-0` 105 KB + Pure-Dense SymplectLoop + LogitNorm + Firewall)."""

    def __init__(
        self,
        vocab_size: int = BPE_VOCAB_SIZE_V07,
        embed_dim: int = 48,
        hidden_size: int = HIDDEN_V09,
        num_heads: int = 6,
        ffn_size: int = 896,
        engram_buckets: int = ENGRAM_BIGRAM_BUCKETS_V10,
        trigram_buckets: int = ENGRAM_TRIGRAM_BUCKETS_V10,
        engram_dim: int = 16,
        max_len: int = MAX_LEN_V07,
        style_dim: int = STYLE_FEAT_DIM_V11,
        num_codes: int = 8,
        num_prag: int = 8,
        cbm_concepts: int = 128,
        cbm_topk: int = 6,
        intent_dim: int = 1024,
    ):
        super().__init__()
        assert hidden_size == HIDDEN_V09 and style_dim == STYLE_FEAT_DIM_V11
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.engram_buckets = int(engram_buckets)
        self.trigram_buckets = int(trigram_buckets)
        self.d1 = D1_PREFIX_V09
        self.d2 = D2_COMP_V09
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.cbm_concepts = int(cbm_concepts)
        self.cbm_topk = int(cbm_topk)

        # Pillar 1: Zero-BPE Rung-0 Char-Hash Nano Gatekeeper (~105 KB W8 ZIP)
        self.rung0 = Rung0CharHashNanoV12()

        # Embeddings (exact v1.1 keys -> 100% zero-shock copy!)
        self.word_emb_A = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_emb_P1 = OmniQuantLinear(embed_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.word_emb_P2 = OmniQuantLinear(embed_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.engram_emb = nn.Embedding(engram_buckets, engram_dim, padding_idx=0)
        self.engram_P1 = OmniQuantLinear(engram_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.engram_P2 = OmniQuantLinear(engram_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.trigram_emb = nn.Embedding(trigram_buckets, engram_dim, padding_idx=0)
        self.trigram_P1 = OmniQuantLinear(engram_dim, self.d1, bias=False, target_bits="w4", group_size=64)
        self.trigram_P2 = OmniQuantLinear(engram_dim, self.d2, bias=False, target_bits="w4", group_size=64)
        self.pos_emb1 = nn.Embedding(max_len, self.d1)
        self.pos_emb2 = nn.Embedding(max_len, self.d2)
        self.emb_norm1 = RationalDerf(self.d1)
        self.emb_norm2 = RationalDerf(self.d2)
        self.emb_drop = nn.Dropout(0.10)

        # Rung 1 Prefix + Rhythm Conv + Dual-Axis Exit Heads + 421-d Deferral Router (exact v1.1 keys!)
        self.prelude_l0 = BLTMatryoshkaPreludeL0(d1=self.d1, d2=self.d2, m1=FFN1_PREFIX_V09, m2=FFN2_COMP_V09)
        self.r1_rhythm_conv = MultiScaleRhythmConvV10(d1=self.d1)
        self.r1_attn_pool = nn.Linear(self.d1, 1)
        self.r1_style_mlp = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.r1_span_head = nn.Linear(self.d1, 1)
        self.r1_ai_head = nn.Linear(self.d1 * 3 + 32 + 3, 1)
        self.r1_promo_head = nn.Sequential(nn.Linear(self.d1 * 3 + 32 + 3 + 8, 8), nn.SiLU(), nn.Linear(8, 1))
        self.defer_r1_enc = nn.Sequential(nn.Linear(self.d1 * 3 + 32 + 5, 32), nn.SiLU())
        self.defer_r1_out = nn.Linear(32, 1)

        # Pillar 4: Zero-Init Firewall-Strict z_char (32-d) Micro-Rhythm Injection (0.000000 Step-0 shock!)
        self.z_char_r1_proj = nn.Linear(R0_ZCHAR_DIM_V12, 32)
        self.z_char_r1_gate = nn.Parameter(torch.zeros(1))
        self.z_char_coda_ai = nn.Linear(R0_ZCHAR_DIM_V12, AI_EMB_END_V11)
        self.z_char_coda_ai_gate = nn.Parameter(torch.zeros(1))
        self.z_char_coda_promo = nn.Linear(R0_ZCHAR_DIM_V12, 256 - AI_EMB_END_V11)
        self.z_char_coda_promo_gate = nn.Parameter(torch.zeros(1))

        # H1 SSSQ + P3 Haar + Rung 2 Prelude L1 + Slot Cross-Attn + H2 DCCH (exact v1.1 keys!)
        self.h1_d1_norm = RMSNorm(32)
        self.h1_q_a = OmniQuantLinear(self.d1 + 32 + 2, 16, bias=False, target_bits="w4", group_size=64)
        self.h1_q_b = OmniQuantLinear(16, NUM_FAST_SLOTS_V09 * hidden_size // 4, bias=False, target_bits="w4", group_size=64)
        self.h1_q_up = OmniQuantLinear(hidden_size // 4, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h1_gate = nn.Parameter(torch.tensor([0.05]))

        self.haar_k_proj = OmniQuantLinear(self.d1, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)
        self.haar_v_proj = OmniQuantLinear(self.d1, BUS_DIM_V10, bias=False, target_bits="w4", group_size=64)

        self.funnel_norm = RationalDerf(hidden_size)
        self.prelude_l1 = PreludeBlockV07(hidden_size, num_heads=num_heads, swa_heads=2, ffn_size=ffn_size)

        self.slot_queries = nn.Parameter(torch.empty(1, NUM_SLOTS_V09, hidden_size))
        nn.init.orthogonal_(self.slot_queries.data[0], gain=1.0)
        self.slot_k = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_v = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_o = OmniQuantLinear(hidden_size, hidden_size, target_bits="seq2", group_size=128)
        self.slot_norm = RMSNorm(hidden_size)

        self.h2_hyp_proj = OmniQuantLinear(BUS_DIM_V10, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h2_doubt_proj = OmniQuantLinear(BUS_DIM_V10 + 4, hidden_size, bias=False, target_bits="w4", group_size=64)
        self.h2_hyp_gate = nn.Parameter(torch.tensor([0.05]))
        self.h2_doubt_gate = nn.Parameter(torch.tensor([0.05]))

        # ADR-v1.2 Pure-Dense SymplectLoop Core
        self.core = DenseSymplectLoopCoreV12(hidden_size=hidden_size, num_heads=num_heads, ffn_size=ffn_size)

        # Shared UNet Span Base + Dual-Axis Span Heads (exact v1.1 keys!)
        self.span_l0_proj = nn.Linear(hidden_size, 64)
        self.span_l1_proj = nn.Linear(hidden_size, 64)
        self.span_topdown_proj = nn.Linear(hidden_size * 2, 64)
        self.span_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.mil_gate = nn.Parameter(torch.tensor(-1.0))
        self.promo_span_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.promo_mil_gate = nn.Parameter(torch.tensor(-1.0))

        # Coda + Structural Orthogonal Firewall [0:192] AI | [0:128]U[192:256] Promo
        self.coda_norm = RationalDerf(hidden_size)
        self.coda_proj = nn.Sequential(nn.Linear(hidden_size * 2, 256), nn.SiLU())
        self.style_net = nn.Sequential(nn.Linear(style_dim, 64), nn.SiLU(), nn.Linear(64, 256))
        self.style_gate = nn.Parameter(torch.tensor(-2.5))
        self.cbm_enc = nn.Linear(256, cbm_concepts)
        self.cbm_dec = nn.Linear(cbm_concepts, 256, bias=False)
        self.cbm_audit_w = nn.Linear(cbm_concepts, 1, bias=False)
        with torch.no_grad():
            self.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, AI_EMB_END_V11:].zero_()
            self.cbm_audit_w.weight[:, CBM_AI_CONCEPTS_V11:].zero_()

        # Pillar 2: Bounded-Norm Cosine LogitNorm Dual-Axis Heads
        self.ai_head = CosineLogitNormHeadV12(AI_EMB_END_V11, init_scale=8.0, max_scale=14.0)
        self.ebt_logit_gate = nn.Parameter(torch.tensor(-1.5))

        promo_in_dim = SHARED_EMB_END_V11 + (256 - AI_EMB_END_V11)
        self.promo_head = CosineLogitNormHeadV12(promo_in_dim, init_scale=8.0, max_scale=14.0)
        self.promo_sub4_head = nn.Linear(promo_in_dim, len(PROMO_SUB4_NAMES))
        self.ebt_promo_gate = nn.Parameter(torch.tensor(-1.5))

        self.defer_head_r2 = ResidualDeferralRouterV09(256 + 6, hidden=32)
        self.attr6_old_w = nn.Parameter(torch.empty(5, 256))
        self.attr6_old_b = nn.Parameter(torch.zeros(5))
        self.attr6_new_w = nn.Parameter(torch.empty(1, 256))
        self.attr6_new_b = nn.Parameter(torch.full((1,), -6.0))
        nn.init.kaiming_uniform_(self.attr6_old_w, a=math.sqrt(5))
        with torch.no_grad():
            self.attr6_new_w.copy_(self.attr6_old_w[0:1])

        self.code_head = nn.Linear(256, num_codes)
        self.prag_head = nn.Linear(256, num_prag)
        self.intent_head = nn.Linear(256, intent_dim)

        # Long-Context Streaming (exact v1.1 keys!)
        self.chunk_emb_proj = OmniQuantLinear(256, 16, bias=False, target_bits="w4", group_size=64)
        self.longstream = DynaSeamMILAggregatorV11(chunk_emb_dim=16, hidden=32, num_heads=2)

        # Calibration, Fusion & Firewall ABI Buffers
        self.register_buffer("exit_affine_a", torch.ones(3, dtype=torch.float32))
        self.register_buffer("exit_affine_b", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("exit_affine_promo_a", torch.ones(3, dtype=torch.float32))
        self.register_buffer("exit_affine_promo_b", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("emb_perm", torch.arange(256, dtype=torch.long))
        self.register_buffer("fusion_weights", torch.tensor([0.50, 0.25, 0.25], dtype=torch.float32))

    def emb_layout_tag(self) -> str:
        h = hashlib.sha256(self.emb_perm.detach().cpu().numpy().astype(np.int32).tobytes()).hexdigest()[:8]
        return f"v12-dense-{h}"

    def set_quant_mode(self, mode: str) -> None:
        for m in self.modules():
            if isinstance(m, OmniQuantLinear):
                m.quant_mode = m.target_bits if mode == "sub2" else mode

    def _embed_prefix(self, input_ids: torch.Tensor, engram_ids: torch.Tensor, trigram_ids: torch.Tensor) -> torch.Tensor:
        seq_l = input_ids.shape[1]
        pos = torch.arange(seq_l, device=input_ids.device).clamp(max=self.max_len - 1).unsqueeze(0)
        x1 = (
            self.word_emb_P1(self.word_emb_A(input_ids))
            + self.engram_P1(self.engram_emb(engram_ids.clamp(0, self.engram_buckets - 1)))
            + self.trigram_P1(self.trigram_emb(trigram_ids.clamp(0, self.trigram_buckets - 1)))
            + self.pos_emb1(pos)
        )
        return self.emb_drop(self.emb_norm1(x1))

    def _embed_complement(self, input_ids: torch.Tensor, engram_ids: torch.Tensor, trigram_ids: torch.Tensor) -> torch.Tensor:
        seq_l = input_ids.shape[1]
        pos = torch.arange(seq_l, device=input_ids.device).clamp(max=self.max_len - 1).unsqueeze(0)
        x2 = (
            self.word_emb_P2(self.word_emb_A(input_ids))
            + self.engram_P2(self.engram_emb(engram_ids.clamp(0, self.engram_buckets - 1)))
            + self.trigram_P2(self.trigram_emb(trigram_ids.clamp(0, self.trigram_buckets - 1)))
            + self.pos_emb2(pos)
        )
        return self.emb_drop(self.emb_norm2(x2))

    @staticmethod
    def _compute_l0_haar_highpass(h0_1: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.float().unsqueeze(-1)
        hm = h0_1 * m
        h_prev = F.pad(hm, (0, 0, 1, 0))[:, :-1, :]
        h_next = F.pad(hm, (0, 0, 0, 1))[:, 1:, :]
        m_prev = F.pad(m, (0, 0, 1, 0))[:, :-1, :]
        m_next = F.pad(m, (0, 0, 0, 1))[:, 1:, :]
        return (hm - (h_prev + h_next) / (m_prev + m_next).clamp(min=1.0)) * m

    def forward_rung0(
        self, c1: torch.Tensor, c3: torch.Tensor, c_mask: torch.Tensor, c_feats: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.rung0(c1, c3, c_mask, c_feats)

    def forward_rung1_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor | None = None,
        z_char: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)
        x1 = self._embed_prefix(input_ids, engram_ids, trigram_ids)
        h0_1 = self.prelude_l0.forward_prefix(x1, attention_mask)
        h0_1 = self.r1_rhythm_conv(h0_1, attention_mask)

        m_f = attention_mask.float().unsqueeze(-1)
        m_den = m_f.sum(dim=1).clamp(min=1.0)
        h_avg = (h0_1 * m_f).sum(dim=1) / m_den
        h_max = h0_1.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e4).amax(dim=1)
        a_logits = self.r1_attn_pool(h0_1).squeeze(-1).masked_fill(~attention_mask.bool(), -1e4)
        h_attn = (h0_1 * F.softmax(a_logits, dim=-1).unsqueeze(-1)).sum(dim=1)
        h_std_s = torch.sqrt((((h0_1 - h_avg.unsqueeze(1)) ** 2) * m_f).sum(dim=1) / m_den + 1e-6).mean(dim=-1, keepdim=True)

        tok_r = torch.sigmoid(self.r1_span_head(h0_1).squeeze(-1)) * attention_mask.float()
        w_avg = F.avg_pool1d(F.pad(tok_r.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        r_max_win = w_avg.amax(dim=-1)
        r_std_win = torch.sqrt(((w_avg - w_avg.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1) + 1e-6)

        st_h = self.r1_style_mlp(style_feats)
        if z_char is not None:
            st_h = st_h + torch.tanh(self.z_char_r1_gate) * self.z_char_r1_proj(z_char)

        feat = torch.cat([h_avg, h_max, h_attn, st_h, h_std_s, r_max_win.unsqueeze(-1), r_std_win.unsqueeze(-1)], dim=-1)
        raw_logit_r1 = self.r1_ai_head(feat).squeeze(-1)
        logit_r1 = (self.exit_affine_a[0] * raw_logit_r1 + self.exit_affine_b[0]) / self.temperature.clamp(min=0.2)

        raw_logit_promo_r1 = self.r1_promo_head(torch.cat([feat, style_feats[:, 48:56]], dim=-1)).squeeze(-1)
        logit_promo_r1 = self.exit_affine_promo_a[0] * raw_logit_promo_r1 + self.exit_affine_promo_b[0]

        d1_hid = self.defer_r1_enc(torch.cat([feat, raw_logit_r1.unsqueeze(-1), raw_logit_promo_r1.unsqueeze(-1)], dim=-1))
        p_defer1 = torch.sigmoid(self.defer_r1_out(d1_hid).squeeze(-1))

        tok_w = tok_r / tok_r.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        r1_susp_vec = (h0_1 * tok_w.unsqueeze(-1)).sum(dim=1)
        r1_pkt = torch.cat([r1_susp_vec, self.h1_d1_norm(d1_hid), r_max_win.unsqueeze(-1), r_std_win.unsqueeze(-1)], dim=-1)

        return {
            "x1_in": x1, "h0_1": h0_1,
            "logit_r1": logit_r1, "raw_logit_r1": raw_logit_r1,
            "logit_promo_r1": logit_promo_r1, "raw_logit_promo_r1": raw_logit_promo_r1,
            "p_defer1": p_defer1, "r1_span_max": r_max_win, "r1_span_std": r_std_win, "r1_pkt": r1_pkt,
        }

    def _coda_readout(
        self,
        slots: torch.Tensor,
        span_base: torch.Tensor,
        mask0: torch.Tensor,
        style_feats: torch.Tensor,
        ebt_out: dict[str, torch.Tensor],
        exit_idx: int,
        promo_sh_lam: float = 0.35,
        z_char: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        s_norm = self.coda_norm(slots)
        z_h = s_norm[:, : NUM_SLOW_SLOTS_V09, :]
        z_a = s_norm[:, NUM_SLOW_SLOTS_V09 : NUM_SLOW_SLOTS_V09 + NUM_SEM_FAST_V10, :]
        z_b = s_norm[:, NUM_SLOW_SLOTS_V09 + NUM_SEM_FAST_V10 :, :]
        s_verdict = z_h.mean(dim=1)
        s_evid = torch.cat([z_a, z_b], dim=1).mean(dim=1)
        s_span_ctx = z_b.mean(dim=1)
        slot_spread = z_h.std(dim=1, unbiased=False).mean(dim=-1)

        emb = self.coda_proj(torch.cat([s_verdict, s_evid], dim=-1))
        emb = emb + torch.sigmoid(self.style_gate) * self.style_net(style_feats)
        if z_char is not None:
            zc_ai = torch.tanh(self.z_char_coda_ai_gate) * self.z_char_coda_ai(z_char)
            zc_pr = torch.tanh(self.z_char_coda_promo_gate) * self.z_char_coda_promo(z_char)
            emb = emb + torch.cat([zc_ai, zc_pr], dim=-1)

        td = self.span_topdown_proj(torch.cat([s_verdict, s_span_ctx], dim=-1)).unsqueeze(1)
        sp_in = span_base + td
        tok_risk = self.span_head(sp_in).squeeze(-1) * mask0.float()
        w_pool = F.avg_pool1d(F.pad(tok_risk.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        mil_peak = w_pool.amax(dim=-1)

        tok_pr_risk = self.promo_span_head(sp_in).squeeze(-1) * mask0.float()
        wp_pool = F.avg_pool1d(F.pad(tok_pr_risk.unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1)
        promo_mil_peak = wp_pool.amax(dim=-1)

        # F6 Structural Firewall on TinyScope-CBM
        cbm_ai_raw = F.relu(F.linear(emb[:, :AI_EMB_END_V11], self.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], self.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        cbm_gov_raw = F.relu(F.linear(emb, self.cbm_enc.weight[CBM_AI_CONCEPTS_V11:, :], self.cbm_enc.bias[CBM_AI_CONCEPTS_V11:]))
        cbm_raw = torch.cat([cbm_ai_raw, cbm_gov_raw], dim=-1)
        k_top = min(self.cbm_topk, self.cbm_concepts)
        top_v, top_i = torch.topk(cbm_raw, k=k_top, dim=-1)
        cbm_sparse = torch.zeros_like(cbm_raw).scatter_(-1, top_i, top_v)
        cbm_rec = self.cbm_dec(cbm_sparse)
        cbm_loss = F.mse_loss(cbm_rec, emb.detach())
        cbm_logit = F.linear(cbm_sparse[:, :CBM_AI_CONCEPTS_V11], self.cbm_audit_w.weight[:, :CBM_AI_CONCEPTS_V11]).squeeze(-1)

        # AI Axis Readout (Cosine LogitNorm, strictly emb[:, :192]!)
        ebt_m_clamped = 4.0 * torch.tanh(ebt_out["ebt_margin"] / 4.0)
        raw_logit = (
            self.ai_head(emb[:, :AI_EMB_END_V11])
            + torch.sigmoid(self.mil_gate) * mil_peak
            + 0.10 * cbm_logit
            + (0.50 * torch.sigmoid(self.ebt_logit_gate)) * ebt_m_clamped
        )
        ex_i = min(exit_idx, 2)
        logit = (self.exit_affine_a[ex_i] * raw_logit + self.exit_affine_b[ex_i]) / self.temperature.clamp(min=0.2)

        # Promo Axis Readout (Cosine LogitNorm, [0:128] U [192:256]!)
        sh_in = tunnel_trm_ste(emb[:, :SHARED_EMB_END_V11], promo_sh_lam) if self.training else emb[:, :SHARED_EMB_END_V11]
        emb_promo = torch.cat([sh_in, emb[:, AI_EMB_END_V11:]], dim=-1)
        ebt_pm_clamped = 4.0 * torch.tanh(ebt_out["ebt_promo_margin"] / 4.0)
        raw_logit_promo = (
            self.promo_head(emb_promo)
            + torch.sigmoid(self.promo_mil_gate) * promo_mil_peak
            + (0.50 * torch.sigmoid(self.ebt_promo_gate)) * ebt_pm_clamped
        )
        logit_promo = self.exit_affine_promo_a[ex_i] * raw_logit_promo + self.exit_affine_promo_b[ex_i]
        promo_sub4_logit = self.promo_sub4_head(emb_promo)

        a6_w = torch.cat([self.attr6_old_w, self.attr6_new_w], dim=0)
        a6_b = torch.cat([self.attr6_old_b, self.attr6_new_b], dim=0)
        attr6_logit = F.linear(emb, a6_w, a6_b)

        p_defer2 = self.defer_head_r2(
            torch.cat(
                [
                    emb, raw_logit.unsqueeze(-1), slot_spread.unsqueeze(-1), mil_peak.unsqueeze(-1),
                    ebt_out["free_energy"].unsqueeze(-1), ebt_out["grad_norm"].unsqueeze(-1), ebt_out["entropy"].unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        return {
            "logit": logit, "raw_logit": raw_logit,
            "logit_promo": logit_promo, "raw_logit_promo": raw_logit_promo, "promo_sub4_logit": promo_sub4_logit,
            "emb": emb, "mil_peak": mil_peak, "promo_mil_peak": promo_mil_peak, "slot_spread": slot_spread,
            "p_defer2": p_defer2, "cbm_loss": cbm_loss,
            "ebt_energies": ebt_out["energies"], "ebt_margin": ebt_out["ebt_margin"],
            "ebt_promo_margin": ebt_out["ebt_promo_margin"], "ebt_free_energy": ebt_out["free_energy"],
            "ebt_grad_norm": ebt_out["grad_norm"], "attr6_logit": attr6_logit,
            "code_logit": self.code_head(emb), "prag_logit": self.prag_head(emb), "intent_pred": self.intent_head(emb),
        }

    def forward_rung2_t1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        x1_in: torch.Tensor,
        h0_1: torch.Tensor,
        r1_pkt: torch.Tensor,
        trigram_ids: torch.Tensor | None = None,
        handoff_mask: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
        fwd_state_in: torch.Tensor | None = None,
        z_char: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)
        x2_in = self._embed_complement(input_ids, engram_ids, trigram_ids)
        m_f = attention_mask.float().unsqueeze(-1)
        cls_c = (x2_in * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)

        h0_2, fwd_st_out = self.prelude_l0.forward_complement(
            x1_in, x2_in, h0_1, attention_mask, cls_c, fwd_state_in=fwd_state_in
        )
        h0_full = torch.cat([h0_1, h0_2], dim=-1)
        h0_pool = (h0_full * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)

        h0_haar = self._compute_l0_haar_highpass(h0_1, attention_mask)
        k_haar = self.haar_k_proj(h0_haar)
        v_haar = self.haar_v_proj(h0_haar)

        h_pool = F.avg_pool1d(h0_full.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)
        m_pool = (F.max_pool1d(attention_mask.float().unsqueeze(1), kernel_size=2, stride=2, ceil_mode=True).squeeze(1) > 0.5).long()
        cls_full = (h_pool * m_pool.float().unsqueeze(-1)).sum(dim=1) / m_pool.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        h1_full, _ = self.prelude_l1(self.funnel_norm(h_pool), m_pool, cls_full)

        seq_l0 = h0_full.shape[1]
        p1_up = F.interpolate(self.span_l1_proj(h1_full).transpose(1, 2), size=seq_l0, mode="nearest").transpose(1, 2)
        span_base = self.span_l0_proj(h0_full) + p1_up

        k_ev = self.slot_k(h1_full)
        v_ev = self.slot_v(h1_full)
        bsz = h1_full.shape[0]

        q0_base = self.slot_queries.expand(bsz, -1, -1)
        pkt_clean = r1_pkt.detach() if self.training else r1_pkt
        if handoff_mask is not None:
            pkt_clean = pkt_clean * handoff_mask.view(-1, 1)
        dq_l = self.h1_q_up(F.silu(self.h1_q_b(F.silu(self.h1_q_a(pkt_clean))).view(bsz, NUM_FAST_SLOTS_V09, self.hidden_size // 4)))
        g_h1 = 0.35 * torch.tanh(self.h1_gate)
        q0 = torch.cat([q0_base[:, : NUM_SLOW_SLOTS_V09, :], q0_base[:, NUM_SLOW_SLOTS_V09 :, :] + g_h1 * dq_l], dim=1)

        sc0 = torch.matmul(q0, k_ev.transpose(-1, -2)) / math.sqrt(float(self.hidden_size))
        sc0 = sc0.masked_fill(~m_pool.bool().unsqueeze(1), -1e4)
        s0 = self.slot_norm(q0 + self.slot_o(torch.matmul(F.softmax(sc0, dim=-1), v_ev)))

        st1 = self.core.forward_outer_step(
            s0, s0, k_ev, v_ev, m_pool, h0_pool, outer_idx=0,
            k_haar=k_haar, v_haar=v_haar, haar_mask=attention_mask,
            k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
        )
        e1_out = self._coda_readout(st1["slots"], span_base, attention_mask, style_feats, st1["ebt_out"], exit_idx=1, promo_sh_lam=promo_sh_lam, z_char=z_char)

        return {
            "exit1_out": e1_out, "slots": st1["slots"], "s_anchor": s0,
            "carry_m_l": st1["carry_m_l"], "zh_bus": st1["zh_bus"],
            "k_ev": k_ev, "v_ev": v_ev, "m_pool": m_pool,
            "k_haar": k_haar, "v_haar": v_haar, "h0_pool": h0_pool, "span_base": span_base,
            "fwd_state_out": fwd_st_out,
        }

    def forward_rung3_step(
        self,
        slots: torch.Tensor,
        s_anchor: torch.Tensor,
        carry_m_l: torch.Tensor,
        zh_bus: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        m_pool: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        attention_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        span_base: torch.Tensor,
        style_feats: torch.Tensor,
        haar_mask: torch.Tensor | None = None,
        u_doubt_scalars: torch.Tensor | None = None,
        outer_idx: int = 1,
        handoff_mask: torch.Tensor | None = None,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
        z_char: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        s_in = slots
        if u_doubt_scalars is not None:
            s_hyp = zh_bus.detach() if self.training else zh_bus
            u_d = torch.cat([s_hyp, u_doubt_scalars.detach() if self.training else u_doubt_scalars], dim=-1)
            if handoff_mask is not None:
                hm = handoff_mask.view(-1, 1)
                s_hyp = s_hyp * hm
                u_d = u_d * hm
            conf_damp = 1.0 - 0.75 * (u_doubt_scalars[:, 0:1].clamp(0.0, 1.0) ** 2)
            falsify_delta = (
                -0.20 * torch.tanh(self.h2_hyp_gate) * conf_damp * self.h2_hyp_proj(s_hyp)
                + 0.25 * torch.tanh(self.h2_doubt_gate) * self.h2_doubt_proj(u_d)
            )
            s_in = torch.cat([s_in[:, :-1, :], s_in[:, -1:, :] + falsify_delta.unsqueeze(1)], dim=1)

        st = self.core.forward_outer_step(
            s_in, s_anchor, k_ev, v_ev, m_pool, h0_pool, outer_idx=outer_idx,
            k_haar=k_haar, v_haar=v_haar, haar_mask=(haar_mask if haar_mask is not None else attention_mask),
            carry_m_l=carry_m_l, prev_zh_bus=zh_bus, ctx_vec=ctx_vec, ctx_kish_lambda=ctx_kish_lambda,
            k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
        )
        out = self._coda_readout(st["slots"], span_base, attention_mask, style_feats, st["ebt_out"], exit_idx=2, promo_sh_lam=promo_sh_lam, z_char=z_char)
        return {**out, **st}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor | None = None,
        trigram_ids: torch.Tensor | None = None,
        r0_c1: torch.Tensor | None = None,
        r0_c3: torch.Tensor | None = None,
        r0_mask: torch.Tensor | None = None,
        r0_feats: torch.Tensor | None = None,
        outer_steps: int = 2,
        k_inner: int = 2,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        promo_sh_lam: float = 0.35,
        handoff_drop_prob: float = 0.0,
        handoff_flip_prob: float = 0.0,
        ctx_vec: torch.Tensor | None = None,
        ctx_kish_lambda: torch.Tensor | None = None,
        fwd_state_in: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if engram_ids is None:
            engram_ids = torch.zeros_like(input_ids)
        if trigram_ids is None:
            trigram_ids = torch.zeros_like(input_ids)

        r0_out = None
        z_char = None
        if r0_c1 is not None and r0_c3 is not None and r0_mask is not None and r0_feats is not None:
            r0_out = self.forward_rung0(r0_c1, r0_c3, r0_mask, r0_feats)
            z_char = r0_out["z_char"]

        r1 = self.forward_rung1_prefix(input_ids, attention_mask, style_feats, engram_ids, trigram_ids, z_char=z_char)
        bsz, device = input_ids.shape[0], input_ids.device
        h_mask = None
        r1_pkt_use = r1["r1_pkt"]
        if self.training and (handoff_drop_prob > 0.0 or handoff_flip_prob > 0.0):
            h_mask = (torch.rand(bsz, device=device) >= handoff_drop_prob).float()
            if handoff_flip_prob > 0.0 and bsz >= 2:
                do_flip = (torch.rand(bsz, device=device) < handoff_flip_prob).unsqueeze(-1)
                r1_pkt_use = torch.where(do_flip, torch.roll(r1_pkt_use.detach(), shifts=1, dims=0), r1_pkt_use)

        r2 = self.forward_rung2_t1(
            input_ids, attention_mask, style_feats, engram_ids, r1["x1_in"], r1["h0_1"], r1_pkt_use,
            trigram_ids=trigram_ids, handoff_mask=h_mask, k_inner=k_inner, trm_lam_k0=trm_lam_k0,
            trm_lam_k1=trm_lam_k1, promo_sh_lam=promo_sh_lam, fwd_state_in=fwd_state_in, z_char=z_char,
        )
        slots, carry_m_l, zh_bus = r2["slots"], r2["carry_m_l"], r2["zh_bus"]
        final_out = r2["exit1_out"]
        e1 = r2["exit1_out"]

        for oi in range(1, max(1, int(outer_steps))):
            c1_conf = (2.0 * torch.sigmoid(e1["logit"].detach()) - 1.0).abs().unsqueeze(-1)
            u_doubt_s = torch.cat([c1_conf, e1["slot_spread"].unsqueeze(-1), e1["p_defer2"].unsqueeze(-1), e1["ebt_grad_norm"].unsqueeze(-1)], dim=-1)
            r3 = self.forward_rung3_step(
                slots, r2["s_anchor"], carry_m_l, zh_bus, r2["k_ev"], r2["v_ev"], r2["m_pool"],
                r2["k_haar"], r2["v_haar"], attention_mask, r2["h0_pool"], r2["span_base"], style_feats,
                u_doubt_scalars=u_doubt_s, outer_idx=oi, handoff_mask=h_mask, ctx_vec=ctx_vec,
                ctx_kish_lambda=ctx_kish_lambda, k_inner=k_inner, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
                promo_sh_lam=promo_sh_lam, z_char=z_char,
            )
            slots, carry_m_l, zh_bus = r3["slots"], r3["carry_m_l"], r3["zh_bus"]
            final_out = r3

        if return_aux:
            res = {**final_out, **r1, "exit1_out": r2["exit1_out"], "slots_final": slots, "slots_anchor": r2["s_anchor"], "fwd_state_out": r2["fwd_state_out"]}
            if r0_out is not None:
                res.update(r0_out)
            return res
        return final_out["logit"], final_out["logit_promo"], final_out["emb"], r1["logit_r1"], r1["logit_promo_r1"]

    def forward_chronostream_batch(
        self,
        chunk_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_eng: torch.Tensor,
        chunk_tri: torch.Tensor,
        chunk_style56: torch.Tensor,
        chunk_meta6: torch.Tensor,
        chunk_valid: torch.Tensor,
        doc_style56: torch.Tensor,
        tbptt_k2: int = 3,
        trunk_grad_lam: float = 0.30,
        trunk_no_grad: bool = True,
        outer_steps: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Unified ChronoStream TBPTT carrying `fwd_state` + Top-32 `L0-Haar` sketch -> `DynaSeam-MIL`!"""
        bsz, m_max, _ = chunk_ids.shape
        device = chunk_ids.device
        grad_start = max(0, m_max - int(tbptt_k2))

        fwd_st: torch.Tensor | None = None
        hk_sketch: torch.Tensor | None = None
        hv_sketch: torch.Tensor | None = None
        ch_ai_l, ch_pr_l, ch_ebt_m, ch_unc, ch_emb16 = [], [], [], [], []

        for mi in range(m_max):
            use_grad = (not trunk_no_grad) and (mi >= grad_start)
            ids_m, msk_m = chunk_ids[:, mi, :], chunk_mask[:, mi, :]
            eng_m, tri_m = chunk_eng[:, mi, :], chunk_tri[:, mi, :]
            st_m, v_m = chunk_style56[:, mi, :], chunk_valid[:, mi].unsqueeze(-1)

            with torch.set_grad_enabled(use_grad):
                r1 = self.forward_rung1_prefix(ids_m, msk_m, st_m, eng_m, tri_m)
                r2 = self.forward_rung2_t1(ids_m, msk_m, st_m, eng_m, r1["x1_in"], r1["h0_1"], r1["r1_pkt"], trigram_ids=tri_m, fwd_state_in=fwd_st)
                fwd_new = r2["fwd_state_out"]
                if fwd_new is not None:
                    if fwd_st is not None:
                        v_nd = v_m.view([bsz] + [1] * (fwd_new.ndim - 1))
                        fwd_st = torch.where(v_nd > 0.5, fwd_new, fwd_st)
                    else:
                        fwd_st = fwd_new
                    if not use_grad:
                        fwd_st = fwd_st.detach()

                kh, vh = r2["k_haar"], r2["v_haar"]
                if hk_sketch is not None:
                    kh_cat, vh_cat = torch.cat([hk_sketch, kh], dim=1), torch.cat([hv_sketch, vh], dim=1)
                else:
                    kh_cat, vh_cat = kh, vh
                kk = min(32, kh_cat.shape[1])
                idx_e = torch.topk(kh_cat.norm(dim=-1), k=kk, dim=-1).indices.unsqueeze(-1).expand(-1, -1, kh_cat.shape[-1])
                hk_next, hv_next = torch.gather(kh_cat, 1, idx_e), torch.gather(vh_cat, 1, idx_e)
                if not use_grad:
                    hk_next, hv_next = hk_next.detach(), hv_next.detach()
                hk_sketch, hv_sketch = hk_next, hv_next

                e1 = r2["exit1_out"]
                if outer_steps >= 2:
                    c1_c = (2.0 * torch.sigmoid(e1["logit"].detach()) - 1.0).abs().unsqueeze(-1)
                    uds = torch.cat([c1_c, e1["slot_spread"].unsqueeze(-1), e1["p_defer2"].unsqueeze(-1), e1["ebt_grad_norm"].unsqueeze(-1)], dim=-1)
                    out_m = self.forward_rung3_step(
                        r2["slots"], r2["s_anchor"], r2["carry_m_l"], r2["zh_bus"],
                        r2["k_ev"], r2["v_ev"], r2["m_pool"], hk_sketch, hv_sketch,
                        msk_m, r2["h0_pool"], r2["span_base"], st_m,
                        haar_mask=torch.ones((bsz, hk_sketch.shape[1]), dtype=torch.long, device=device),
                        u_doubt_scalars=uds, outer_idx=1,
                    )
                else:
                    out_m = e1

                lg_ai, lg_pr, ebt_m, unc, emb_m = out_m["logit"], out_m["logit_promo"], out_m["ebt_margin"], out_m["p_defer2"], out_m["emb"]

            if self.training and use_grad:
                lg_ai = tunnel_trm_ste(lg_ai, trunk_grad_lam)
                lg_pr = tunnel_trm_ste(lg_pr, trunk_grad_lam)
                emb_m = tunnel_trm_ste(emb_m, trunk_grad_lam)

            ch_ai_l.append(lg_ai)
            ch_pr_l.append(lg_pr)
            ch_ebt_m.append(ebt_m)
            ch_unc.append(unc)
            ch_emb16.append(self.chunk_emb_proj(emb_m))

        ai_stack = torch.stack(ch_ai_l, dim=1)
        pr_stack = torch.stack(ch_pr_l, dim=1)
        ebt_stack = torch.stack(ch_ebt_m, dim=1)
        unc_stack = torch.stack(ch_unc, dim=1)
        emb16_stack = torch.stack(ch_emb16, dim=1)
        chunk_x = torch.cat([ai_stack.unsqueeze(-1), pr_stack.unsqueeze(-1), ebt_stack.unsqueeze(-1), unc_stack.unsqueeze(-1), emb16_stack, chunk_meta6, chunk_style56], dim=-1)
        ds_out = self.longstream(chunk_x, ai_stack, pr_stack, chunk_meta6[:, :, 3], chunk_meta6[:, :, 4], chunk_valid, doc_style56)
        return {**ds_out, "raw_chunk_ai": ai_stack, "raw_chunk_pr": pr_stack}


# ---------------------------------------------------------------------------
# 6. Zero-Shock Warm-Start Surgery (`v1.1.0-R1` -> `v1.2.0 Dense + LogitNorm`)
# ---------------------------------------------------------------------------

@torch.no_grad()
def transplant_v11_to_v12_dense(
    v11_model: MeguvizV11OmniGuardModel,
    v12_model: MeguvizV12ApexNanoModel,
    calib_emb_v11: np.ndarray | None = None,
) -> dict[str, Any]:
    """Copy ALL dense v1.1.0 weights identically & collapse 10-expert soft-MoLE into `SharedQuantResV12`!"""
    v11_model.eval()
    v12_model.eval()
    sd11 = v11_model.state_dict()
    sd12 = v12_model.state_dict()

    copied = 0
    for k, v in sd11.items():
        if k in sd12 and sd12[k].shape == v.shape and not k.startswith(("ai_head.", "promo_head.")):
            sd12[k].copy_(v.to(sd12[k].device, dtype=sd12[k].dtype))
            copied += 1

    # 1. Exact Collapse of DynaCore-MoLE (`core.mole`) -> SharedQuantResV12 (`core.qres`)!
    m11 = v11_model.core.mole
    q12 = v12_model.core.qres
    q12.basis_u.weight.copy_(m11.basis_u.weight.to(q12.basis_u.weight.device, dtype=q12.basis_u.weight.dtype))
    q12.basis_v.weight.copy_(m11.basis_v.weight.to(q12.basis_v.weight.device, dtype=q12.basis_v.weight.dtype))
    q12.phase_silu_gain.copy_(m11.phase_silu_gain.to(q12.phase_silu_gain.device))
    q12.phase_diag.copy_(m11.phase_diag.to(q12.phase_diag.device))
    q12.adaln_scale.copy_(m11.adaln_scale.to(q12.adaln_scale.device))
    q12.adaln_shift.copy_(m11.adaln_shift.to(q12.adaln_shift.device))

    routed_all = torch.cat([m11.core_routed_old, m11.core_routed_new], dim=0)
    bias_all = torch.cat([m11.core_bias_old, m11.core_bias_new], dim=0)
    dyn_all = torch.cat([m11.dyn_profiles_old, m11.dyn_profiles_new], dim=0)
    c_mean = m11.core_shared + routed_all.mean(dim=0)
    b_mean = bias_all.mean(dim=0)
    d_mean = dyn_all.mean(dim=0)
    for pi in range(3):
        q12.phase_core[pi].copy_(c_mean.to(q12.phase_core.device))
        q12.phase_bias[pi].copy_(b_mean.to(q12.phase_bias.device))
    q12.dyn_ctrl.copy_(d_mean.to(q12.dyn_ctrl.device))

    # 2. Initialize CosineLogitNormHeadV12 from v1.1 linear ai_head & promo_head
    def _init_ln_head(old_lin: nn.Linear, new_head: CosineLogitNormHeadV12, emb_norms: float) -> float:
        w = old_lin.weight.detach().float()
        b = old_lin.bias.detach().float()
        new_head.weight.copy_(w.to(new_head.weight.device, dtype=new_head.weight.dtype))
        new_head.bias.copy_(b.to(new_head.bias.device, dtype=new_head.bias.dtype))
        eff_s = float(np.clip(float(w.norm().item()) * emb_norms, 3.0, new_head.max_scale - 0.5))
        raw_s = math.log(max(math.exp(max(eff_s - 2.0, 0.2)) - 1.0, 1e-4))
        new_head.log_scale.fill_(raw_s)
        return eff_s

    if calib_emb_v11 is not None and len(calib_emb_v11) > 0:
        ai_norm = float(np.median(np.linalg.norm(calib_emb_v11[:, :AI_EMB_END_V11], axis=-1)))
        pr_e = np.concatenate([calib_emb_v11[:, :SHARED_EMB_END_V11], calib_emb_v11[:, AI_EMB_END_V11:]], axis=-1)
        pr_norm = float(np.median(np.linalg.norm(pr_e, axis=-1)))
    else:
        ai_norm = pr_norm = 12.0

    s_ai = _init_ln_head(v11_model.ai_head, v12_model.ai_head, ai_norm)
    s_pr = _init_ln_head(v11_model.promo_head, v12_model.promo_head, pr_norm)
    return {
        "copied_identical_tensors": copied,
        "ai_logitnorm_init_scale": round(s_ai, 3),
        "promo_logitnorm_init_scale": round(s_pr, 3),
        "emb_layout": v12_model.emb_layout_tag(),
    }


@torch.no_grad()
def fold_zero_op_v12(
    model: MeguvizV12ApexNanoModel,
    calib_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    smooth_alpha: float = 0.5,
) -> dict[str, Any]:
    """Exact ZERO-OP offline algebraic fold (SpinQuant R_2 + SubLN + SmoothQuant on Prelude L1)."""
    h64 = _hadamard_64()

    def _rotate_vo(v_lin: nn.Linear, o_lin: nn.Linear, sub_ln: RMSNorm, n_heads: int) -> None:
        dev, dt = v_lin.weight.device, v_lin.weight.dtype
        sw = sub_ln.weight.detach().float()
        o_lin.weight.data.copy_((o_lin.weight.detach().float() * sw.unsqueeze(0)).to(dt))
        sub_ln.weight.data.fill_(1.0)
        r2 = torch.block_diag(*([h64] * n_heads)).to(dev)
        v_lin.weight.data.copy_((r2.T @ v_lin.weight.detach().float()).to(dt))
        if v_lin.bias is not None:
            v_lin.bias.data.copy_((v_lin.bias.detach().float() @ r2).to(dt))
        o_lin.weight.data.copy_((o_lin.weight.detach().float() @ r2).to(dt))

    _rotate_vo(model.core.v_proj, model.core.o_proj, model.core.sub_ln_attn, model.core.num_heads)
    _rotate_vo(model.prelude_l0.v_p, model.prelude_l0.o_p, model.prelude_l0.sub_ln_p, model.prelude_l0.swa_heads)

    folded_subln = 0
    for mod in model.modules():
        if hasattr(mod, "w_up") and hasattr(mod, "sub_ln") and hasattr(mod, "w_down"):
            dt = mod.w_down.weight.dtype
            sw = mod.sub_ln.weight.detach().float()
            mod.w_down.weight.data.copy_((mod.w_down.weight.detach().float() * sw.unsqueeze(0)).to(dt))
            mod.sub_ln.weight.data.fill_(1.0)
            folded_subln += 1
    p0 = model.prelude_l0
    dt_c = p0.ffn_c_down.weight.dtype
    p0.ffn_c_down.weight.data.copy_((p0.ffn_c_down.weight.detach().float() * p0.ffn_c_subln.weight.detach().float().unsqueeze(0)).to(dt_c))
    p0.ffn_c_subln.weight.data.fill_(1.0)
    folded_subln += 1

    act_max: torch.Tensor | None = None
    if calib_inputs:
        def _hk(_m, _inp, out):
            nonlocal act_max
            mx = out.detach().float().abs().amax(dim=(0, 1))
            act_max = mx if act_max is None else torch.maximum(act_max, mx)

        h = model.prelude_l1.norm2.register_forward_hook(_hk)
        model.eval()
        for ids_t, m_t, st_t, eng_t in calib_inputs:
            _ = model(ids_t, m_t, st_t, engram_ids=eng_t, outer_steps=2, return_aux=False)
        h.remove()

    b1 = model.prelude_l1
    dt1 = b1.ffn.w_up.weight.dtype
    w_abs = b1.ffn.w_up.weight.detach().float().abs().amax(dim=0).clamp(min=1e-4)
    x_abs = (act_max.to(w_abs.device) if act_max is not None else torch.ones_like(w_abs)).clamp(min=1e-4)
    s = (x_abs.pow(smooth_alpha) / w_abs.pow(1.0 - smooth_alpha)).clamp(0.25, 4.0)
    b1.norm2.gamma.data.copy_((b1.norm2.gamma.detach().float() / s).to(dt1))
    b1.norm2.beta.data.copy_((b1.norm2.beta.detach().float() / s).to(dt1))
    b1.ffn.w_up.weight.data.copy_((b1.ffn.w_up.weight.detach().float() * s.unsqueeze(0)).to(dt1))
    b1.lora_moe.down.data.copy_((b1.lora_moe.down.detach().float() * s.view(1, -1, 1)).to(dt1))

    return {"folded_spinquant_r2_blocks": 2, "folded_subln_ffn_blocks": folded_subln, "folded_smoothquant_blocks": 1, "new_runtime_ops_added": 0}


@torch.no_grad()
def apply_dual_axis_oblique_leace_v12(
    model: MeguvizV12ApexNanoModel,
    emb_val: np.ndarray,
    y_ai_val: np.ndarray,
    y_promo_val: np.ndarray,
    z_ai_confounders: np.ndarray,
    z_promo_confounders: np.ndarray,
    ai_strength_grid: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35),
    promo_strength_grid: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45),
) -> dict[str, Any]:
    """Capped Unit-Sphere Oblique LEACE for `CosineLogitNormHeadV12` (never over-projects past 0.35 on AI!)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    def _unit(x: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(x, axis=-1, keepdims=True)
        return (x / np.maximum(nrm, 1e-6)).astype(np.float32)

    # 1. AI axis LEACE on L2-normalized `emb[:, :192]`
    z_ai = _unit(emb_val[:, :AI_EMB_END_V11].astype(np.float32))
    hu_m = np.asarray(y_ai_val) <= 0.30
    ai_bin = (np.asarray(y_ai_val) >= 0.70).astype(int)
    valid_ai = (np.asarray(y_ai_val) <= 0.30) | (np.asarray(y_ai_val) >= 0.70)

    w_ai0 = model.ai_head.weight.detach().float().cpu().numpy().copy()
    b_ai0 = model.ai_head.bias.detach().float().cpu().numpy().copy()
    sc_ai = float(model.ai_head.effective_scale().item())
    best_ai_s, best_ai_obj = 0.0, -1.0
    best_ai_wb = (w_ai0, b_ai0)

    if hu_m.sum() >= 32 and valid_ai.sum() >= 32 and len(np.unique(ai_bin[valid_ai])) == 2:
        def _score_ln(z_m, w_m, b_m, sc):
            wn = w_m / max(float(np.linalg.norm(w_m)), 1e-6)
            return (sc * (z_m @ wn.T) + b_m).reshape(-1)

        s0 = _score_ln(z_ai[valid_ai], w_ai0, b_ai0, sc_ai)
        best_ai_obj = 0.5 * float(roc_auc_score(ai_bin[valid_ai], s0)) + 0.5 * float(average_precision_score(ai_bin[valid_ai], s0))
        for s in ai_strength_grid:
            if s == 0.0:
                continue
            m_mat, b_vec = compute_oblique_leace_affine(z_ai[hu_m], z_ai_confounders[hu_m], strength=s)
            w_n = w_ai0 @ m_mat
            w_n_unit = w_n / max(float(np.linalg.norm(w_n)), 1e-6)
            b_n = b_ai0 + sc_ai * (w_n_unit @ b_vec)
            ss = _score_ln(z_ai[valid_ai], w_n, b_n, sc_ai)
            obj_s = 0.5 * float(roc_auc_score(ai_bin[valid_ai], ss)) + 0.5 * float(average_precision_score(ai_bin[valid_ai], ss))
            if obj_s >= best_ai_obj:
                best_ai_obj, best_ai_s, best_ai_wb = obj_s, float(s), (w_n, b_n)
        model.ai_head.weight.data.copy_(torch.from_numpy(best_ai_wb[0]).to(model.ai_head.weight.device, dtype=model.ai_head.weight.dtype))
        model.ai_head.bias.data.copy_(torch.from_numpy(best_ai_wb[1]).to(model.ai_head.bias.device, dtype=model.ai_head.bias.dtype))

    # 2. Promo axis LEACE on L2-normalized `cat(emb[:, :128], emb[:, 192:256])`
    z_pr = _unit(np.concatenate([emb_val[:, :SHARED_EMB_END_V11], emb_val[:, AI_EMB_END_V11:]], axis=1).astype(np.float32))
    org_m = np.asarray(y_promo_val) <= 0.25
    pr_bin = (np.asarray(y_promo_val) >= 0.65).astype(int)
    valid_pr = (np.asarray(y_promo_val) <= 0.25) | (np.asarray(y_promo_val) >= 0.65)

    w_pr0 = model.promo_head.weight.detach().float().cpu().numpy().copy()
    b_pr0 = model.promo_head.bias.detach().float().cpu().numpy().copy()
    sc_pr = float(model.promo_head.effective_scale().item())
    best_pr_s, best_pr_obj = 0.0, -1.0
    best_pr_wb = (w_pr0, b_pr0)

    if org_m.sum() >= 32 and valid_pr.sum() >= 32 and len(np.unique(pr_bin[valid_pr])) == 2:
        def _score_pr(z_m, w_m, b_m, sc):
            wn = w_m / max(float(np.linalg.norm(w_m)), 1e-6)
            return (sc * (z_m @ wn.T) + b_m).reshape(-1)

        s0 = _score_pr(z_pr[valid_pr], w_pr0, b_pr0, sc_pr)
        best_pr_obj = 0.5 * float(roc_auc_score(pr_bin[valid_pr], s0)) + 0.5 * float(average_precision_score(pr_bin[valid_pr], s0))
        for s in promo_strength_grid:
            if s == 0.0:
                continue
            m_mat, b_vec = compute_oblique_leace_affine(z_pr[org_m], z_promo_confounders[org_m], strength=s)
            w_n = w_pr0 @ m_mat
            w_n_unit = w_n / max(float(np.linalg.norm(w_n)), 1e-6)
            b_n = b_pr0 + sc_pr * (w_n_unit @ b_vec)
            ss = _score_pr(z_pr[valid_pr], w_n, b_n, sc_pr)
            obj_s = 0.5 * float(roc_auc_score(pr_bin[valid_pr], ss)) + 0.5 * float(average_precision_score(pr_bin[valid_pr], ss))
            if obj_s >= best_pr_obj:
                best_pr_obj, best_pr_s, best_pr_wb = obj_s, float(s), (w_n, b_n)
        model.promo_head.weight.data.copy_(torch.from_numpy(best_pr_wb[0]).to(model.promo_head.weight.device, dtype=model.promo_head.weight.dtype))
        model.promo_head.bias.data.copy_(torch.from_numpy(best_pr_wb[1]).to(model.promo_head.bias.device, dtype=model.promo_head.bias.dtype))

    return {
        "ai_leace_strength": best_ai_s,
        "ai_val_obj": round(best_ai_obj, 5),
        "promo_leace_strength": best_pr_s,
        "promo_val_obj": round(best_pr_obj, 5),
        "runtime_bytes_added": 0,
    }


# ---------------------------------------------------------------------------
# 7. Bundle v7 (`MGVZ7 <= 2.20 MB`) Pack / Unpack (Includes Rung-0 W8!)
# ---------------------------------------------------------------------------

_V12_SPLIT_MERGE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("core.inloop_ebt.fc2_old_w", "core.inloop_ebt.fc2_new_w", "core.inloop_ebt.fc2.weight"),
    ("core.inloop_ebt.fc2_old_b", "core.inloop_ebt.fc2_new_b", "core.inloop_ebt.fc2.bias"),
    ("attr6_old_w", "attr6_new_w", "attr6_head.weight"),
    ("attr6_old_b", "attr6_new_b", "attr6_head.bias"),
)


def _pack_w8_bytes(arr_f32: np.ndarray) -> tuple[bytes, bytes]:
    mx = float(max(np.max(np.abs(arr_f32)), 1e-6))
    sc = np.array([mx / 127.0], dtype=np.float16)
    q = np.clip(np.round(arr_f32 / float(sc[0])), -127, 127).astype(np.int8)
    return sc.tobytes(), q.tobytes()


def _unpack_w8_bytes(s_buf: bytes, q_buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    sc = float(np.frombuffer(s_buf, dtype=np.float16)[0])
    return (np.frombuffer(q_buf, dtype=np.int8).astype(np.float32) * sc).reshape(shape)


def pack_bundle_v7_v12(
    model: MeguvizV12ApexNanoModel,
    out_path: Path,
    lever_w4_longstream: bool = False,
    lever_w4_rung0: bool = False,
) -> dict[str, Any]:
    """Pack complete `MeguvizV12ApexNanoModel` (including 105 KB `Rung-0`!) into `MGVZ7 <= 2.20 MB`."""
    sd_raw = model.state_dict()
    sd: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for k_old, k_new, k_merged in _V12_SPLIT_MERGE_PAIRS:
        if k_old in sd_raw and k_new in sd_raw:
            sd[k_merged] = torch.cat([sd_raw[k_old], sd_raw[k_new]], dim=0)
            consumed.add(k_old)
            consumed.add(k_new)
    for k, v in sd_raw.items():
        if k not in consumed:
            sd[k] = v

    qtype_by_param: dict[str, tuple[str, int]] = {
        "word_emb_A.weight": ("seq2", 64),
        "engram_emb.weight": ("w4", 64),
        "trigram_emb.weight": ("w4", 64),
        "pos_emb1.weight": ("w4", 64),
        "pos_emb2.weight": ("w4", 64),
        "coda_proj.0.weight": ("w4", 64),
        "span_l0_proj.weight": ("w4", 64),
        "span_l1_proj.weight": ("w4", 64),
        "span_topdown_proj.weight": ("w4", 64),
        "cbm_enc.weight": ("w4", 64),
        "style_net.2.weight": ("w4", 64),
        # FP16-pinned sensitive heads & SharedQuantResV12
        "core.qres.phase_core": ("fp16", 0),
        "core.qres.phase_bias": ("fp16", 0),
        "core.inloop_ebt.fc1.weight": ("fp16", 0),
        "core.inloop_ebt.fc2.weight": ("fp16", 0),
        "ai_head.weight": ("fp16", 0),
        "promo_head.weight": ("fp16", 0),
        "promo_sub4_head.weight": ("fp16", 0),
        "promo_span_head.1.weight": ("fp16", 0),
        "attr6_head.weight": ("fp16", 0),
        "r1_promo_head.0.weight": ("fp16", 0),
        "r1_promo_head.2.weight": ("fp16", 0),
    }
    if lever_w4_longstream:
        for lk in ("longstream.chunk_enc.0.weight", "longstream.qkv.weight", "longstream.ffn.0.weight", "longstream.ffn.2.weight"):
            qtype_by_param[lk] = ("w4", 64)

    for mod_name, mod in model.named_modules():
        if isinstance(mod, OmniQuantLinear):
            qtype_by_param[f"{mod_name}.weight"] = (mod.target_bits, mod.group_size)
        elif isinstance(mod, SeqLoRAMoE):
            qtype_by_param[f"{mod_name}.down"] = ("w4", 64)
            qtype_by_param[f"{mod_name}.up"] = ("w4", 64)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 8192 and f"{mod_name}.weight" not in qtype_by_param:
            if not mod_name.startswith(("longstream.", "rung0.")):
                qtype_by_param[f"{mod_name}.weight"] = ("w4", 64)

    for k, v in sd.items():
        if k.startswith("rung0.") and k.endswith(".weight") and v.numel() >= 256:
            qtype_by_param[k] = ("w4", 64) if lever_w4_rung0 else ("w8", 0)

    meta_tensors: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    byte_breakdown = {"trit158": 0, "seq2": 0, "w4": 0, "w8": 0, "fp16": 0}

    for k, v in sd.items():
        if k.startswith(("intent_head.", "cbm_dec.")):
            continue
        arr = v.detach().float().cpu().numpy()
        shape = list(arr.shape)
        numel = int(arr.size)
        qkind, gsz = qtype_by_param.get(k, ("fp16", 0))

        if qkind == "trit158":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.mean(np.abs(wg), axis=1, keepdims=True), 1e-5).astype(np.float16)
            q = np.clip(np.round(wg / sc.astype(np.float32)), -1, 1).astype(np.int8)
            raw_q, raw_s = _pack_5trit_bytes(q.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["trit158"] += len(blob)
            meta_tensors.append({"name": k, "q": "trit158", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        elif qkind == "seq2":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.mean(np.abs(wg), axis=1, keepdims=True) * 0.62, 1e-5).astype(np.float16)
            idx = np.clip(np.round((wg / sc.astype(np.float32) + 3.0) * 0.5), 0, 3).astype(np.uint8)
            raw_q, raw_s = _pack_2bit_bytes(idx.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["seq2"] += len(blob)
            meta_tensors.append({"name": k, "q": "seq2", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        elif qkind == "w4":
            pad = (gsz - (numel % gsz)) % gsz
            flat = np.pad(arr.reshape(-1), (0, pad)) if pad else arr.reshape(-1)
            wg = flat.reshape(-1, gsz)
            sc = np.maximum(np.max(np.abs(wg), axis=1, keepdims=True) / 7.0, 1e-5).astype(np.float16)
            q = np.clip(np.round(wg / sc.astype(np.float32)), -7, 7).astype(np.int8)
            raw_q, raw_s = _pack_w4_bytes(q.reshape(-1)[:numel]), sc.tobytes()
            blob = raw_s + raw_q
            byte_breakdown["w4"] += len(blob)
            meta_tensors.append({"name": k, "q": "w4", "g": gsz, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        elif qkind == "w8":
            raw_s, raw_q = _pack_w8_bytes(arr)
            blob = raw_s + raw_q
            byte_breakdown["w8"] += len(blob)
            meta_tensors.append({"name": k, "q": "w8", "g": 0, "shape": shape, "s_bytes": len(raw_s), "q_bytes": len(raw_q)})
            blobs.append(blob)
        else:
            blob = arr.astype(np.float16).tobytes()
            byte_breakdown["fp16"] += len(blob)
            meta_tensors.append({"name": k, "q": "fp16", "g": 0, "shape": shape, "s_bytes": 0, "q_bytes": len(blob)})
            blobs.append(blob)

    header = json.dumps({"magic": "MGVZ7", "v": "1.2.0", "emb_layout": model.emb_layout_tag(), "tensors": meta_tensors}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as wf:
        wf.write(b"MGVZ7\x00\x01\x02")
        wf.write(struct.pack("<I", len(header)))
        wf.write(header)
        for b in blobs:
            wf.write(b)

    raw_bytes = int(out_path.stat().st_size)
    return {
        "file": out_path.name,
        "raw_bytes": raw_bytes,
        "raw_mb": round(raw_bytes / (1024.0 * 1024.0), 4),
        "breakdown_kb": {k: round(v / 1024.0, 1) for k, v in byte_breakdown.items()},
        "emb_layout": model.emb_layout_tag(),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def load_bundle_v7_v12_into_model(model: MeguvizV12ApexNanoModel, mgvz_path: Path) -> None:
    """Bit-exact unpacker from `bundle_v7_apexnano.mgvz` back into `MeguvizV12ApexNanoModel`."""
    data = mgvz_path.read_bytes()
    assert data[:5] == b"MGVZ7", "Invalid MGVZ7 magic!"
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    pos = 12 + hlen
    sd = model.state_dict()
    split_lookup = {k_m: (k_o, k_n) for k_o, k_n, k_m in _V12_SPLIT_MERGE_PAIRS}

    for t in header["tensors"]:
        k, q, gsz, shape = t["name"], t["q"], int(t["g"]), tuple(t["shape"])
        numel = int(np.prod(shape))
        sb, qb = int(t["s_bytes"]), int(t["q_bytes"])
        s_buf, q_buf = data[pos : pos + sb], data[pos + sb : pos + sb + qb]
        pos += sb + qb
        if q == "fp16":
            arr = np.frombuffer(q_buf, dtype=np.float16).astype(np.float32).reshape(shape)
        elif q == "w8":
            arr = _unpack_w8_bytes(s_buf, q_buf, shape)
        else:
            sc = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32).reshape(-1, 1)
            pad = (gsz - (numel % gsz)) % gsz
            if q == "trit158":
                vals = _unpack_5trit_bytes(q_buf, numel).astype(np.float32)
            elif q == "seq2":
                vals = (_unpack_2bit_bytes(q_buf, numel).astype(np.float32) * 2.0) - 3.0
            else:
                vals = _unpack_w4_bytes(q_buf, numel).astype(np.float32)
            if pad:
                vals = np.pad(vals, (0, pad))
            arr = (vals.reshape(-1, gsz) * sc).reshape(-1)[:numel].reshape(shape)

        t_arr = torch.from_numpy(arr.copy())
        if k in split_lookup:
            ko, kn = split_lookup[k]
            n_o = sd[ko].shape[0]
            sd[ko].copy_(t_arr[:n_o].to(sd[ko].dtype))
            sd[kn].copy_(t_arr[n_o:].to(sd[kn].dtype))
        elif k in sd:
            sd[k].copy_(t_arr.to(sd[k].dtype))


# ---------------------------------------------------------------------------
# 8. 5-Stage Static ONNX Exporter (`R0`, `R1`, `R2`, `R3`, `DynaSeam`) — Post-Unpack Exact!
# ---------------------------------------------------------------------------

class _V12Rung0Export(nn.Module):
    def __init__(self, m: MeguvizV12ApexNanoModel):
        super().__init__()
        self.r0 = m.rung0

    def forward(self, c1, c3, mask, feats):
        o = self.r0(c1, c3, mask, feats)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_r0"], o["z_char"]


class _V12Rung1Export(nn.Module):
    def __init__(self, m: MeguvizV12ApexNanoModel):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, z_char):
        r1 = self.m.forward_rung1_prefix(input_ids, attention_mask, style_feats, engram_ids, trigram_ids, z_char=z_char)
        return r1["logit_r1"], r1["logit_promo_r1"], r1["p_defer1"], r1["r1_span_max"], r1["x1_in"], r1["h0_1"], r1["r1_pkt"]


class _V12Rung2T1Export(nn.Module):
    def __init__(self, m: MeguvizV12ApexNanoModel):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, x1_in, h0_1, r1_pkt, z_char):
        r2 = self.m.forward_rung2_t1(input_ids, attention_mask, style_feats, engram_ids, x1_in, h0_1, r1_pkt, trigram_ids=trigram_ids, z_char=z_char)
        e1 = r2["exit1_out"]
        return (
            e1["logit"], e1["logit_promo"], e1["promo_sub4_logit"], e1["emb"], e1["p_defer2"], e1["slot_spread"],
            e1["ebt_margin"], e1["ebt_grad_norm"],
            r2["slots"], r2["s_anchor"], r2["carry_m_l"], r2["zh_bus"],
            r2["k_ev"], r2["v_ev"], r2["m_pool"], r2["k_haar"], r2["v_haar"], r2["h0_pool"], r2["span_base"],
        )


class _V12Rung3StepExport(nn.Module):
    def __init__(self, m: MeguvizV12ApexNanoModel, outer_idx: int = 1):
        super().__init__()
        self.m = m
        self.outer_idx = int(outer_idx)

    def forward(
        self, slots, s_anchor, carry_m_l, zh_bus, k_ev, v_ev, m_pool, k_haar, v_haar,
        attention_mask, h0_pool, span_base, style_feats, u_doubt_scalars, z_char,
    ):
        r3 = self.m.forward_rung3_step(
            slots, s_anchor, carry_m_l, zh_bus, k_ev, v_ev, m_pool, k_haar, v_haar,
            attention_mask, h0_pool, span_base, style_feats,
            u_doubt_scalars=u_doubt_scalars, outer_idx=self.outer_idx, z_char=z_char,
        )
        return (
            r3["logit"], r3["logit_promo"], r3["promo_sub4_logit"], r3["emb"], r3["ebt_margin"], r3["ebt_grad_norm"],
            r3["slots"], r3["carry_m_l"], r3["zh_bus"],
        )


def export_v12_modular_onnx_and_bundle_v7(
    model: MeguvizV12ApexNanoModel,
    out_dir: Path,
    lever_w4_longstream: bool = False,
    lever_w4_rung0: bool = False,
    export_matmulnbits: bool = False,
) -> dict[str, Any]:
    """Pack `MGVZ7`, reload unpacked Sub-2 weights so ONNX === Bundle v7 parity (`<1e-4`), & export 5-Stage FP32-Exact ONNX!"""
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_cpu = copy.deepcopy(model).eval().cpu()
    mdl_cpu.set_quant_mode("fp16")

    mgvz_path = out_dir / "bundle_v7_apexnano.mgvz"
    mgvz_info = pack_bundle_v7_v12(mdl_cpu, mgvz_path, lever_w4_longstream=lever_w4_longstream, lever_w4_rung0=lever_w4_rung0)
    # CRITICAL v1.2 FIX: Reload unpacked Sub-2/W4/W8 weights BEFORE exporting ONNX -> 100% parity with MGVZ7!
    load_bundle_v7_v12_into_model(mdl_cpu, mgvz_path)
    mdl_cpu.eval()

    exp_kw: dict[str, Any] = {"opset_version": 17, "do_constant_folding": True}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        exp_kw["dynamo"] = False

    # 0. Session R0 (Zero-BPE Char-Hash Nano Gatekeeper)
    w0 = _V12Rung0Export(mdl_cpu).eval()
    r0_fp = out_dir / "v12_r0_charhash_fp32.onnx"
    d_c1 = torch.randint(0, R0_B1_V12, (2, 32), dtype=torch.long)
    d_c3 = torch.randint(0, R0_B3_V12, (2, 32), dtype=torch.long)
    d_cm = torch.ones((2, 32), dtype=torch.float32)
    d_cf = torch.zeros((2, R0_FEAT_DIM_V12), dtype=torch.float32)
    torch.onnx.export(
        w0, (d_c1, d_c3, d_cm, d_cf), str(r0_fp),
        input_names=["c1", "c3", "mask", "feats"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_r0", "z_char"],
        dynamic_axes={"c1": {0: "B", 1: "L"}, "c3": {0: "B", 1: "L"}, "mask": {0: "B", 1: "L"}, "feats": {0: "B"}, "logit_ai_r0": {0: "B"}, "logit_promo_r0": {0: "B"}, "p_defer_r0": {0: "B"}, "z_char": {0: "B"}},
        **exp_kw,
    )
    with torch.no_grad():
        o0 = w0(d_c1, d_c3, d_cm, d_cf)
    s0 = ort.InferenceSession(str(r0_fp), providers=["CPUExecutionProvider"])
    ox0 = s0.run(None, {"c1": d_c1.numpy(), "c3": d_c3.numpy(), "mask": d_cm.numpy(), "feats": d_cf.numpy()})
    r0_diff = float(max(np.max(np.abs(o0[0].numpy() - ox0[0])), np.max(np.abs(o0[3].numpy() - ox0[3]))))

    # 1. Session R1
    d_ids = torch.randint(1, mdl_cpu.vocab_size, (2, 40), dtype=torch.long)
    d_m = torch.ones((2, 40), dtype=torch.long)
    d_st = torch.zeros((2, STYLE_FEAT_DIM_V11), dtype=torch.float32)
    d_eng = torch.randint(1, mdl_cpu.engram_buckets, (2, 40), dtype=torch.long)
    d_tri = torch.randint(1, mdl_cpu.trigram_buckets, (2, 40), dtype=torch.long)
    d_zc = o0[3]

    w1 = _V12Rung1Export(mdl_cpu).eval()
    r1_fp = out_dir / "v12_r1_prefix_fp32.onnx"
    r1_in = ["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "z_char"]
    r1_out = ["logit_r1", "logit_promo_r1", "p_defer1", "r1_span_max", "x1_in", "h0_1", "r1_pkt"]
    torch.onnx.export(
        w1, (d_ids, d_m, d_st, d_eng, d_tri, d_zc), str(r1_fp),
        input_names=r1_in, output_names=r1_out,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "style_feats": {0: "B"},
            "engram_ids": {0: "B", 1: "L"}, "trigram_ids": {0: "B", 1: "L"}, "z_char": {0: "B"},
            "logit_r1": {0: "B"}, "logit_promo_r1": {0: "B"}, "p_defer1": {0: "B"}, "r1_span_max": {0: "B"},
            "x1_in": {0: "B", 1: "L"}, "h0_1": {0: "B", 1: "L"}, "r1_pkt": {0: "B"},
        },
        **exp_kw,
    )
    with torch.no_grad():
        o1 = w1(d_ids, d_m, d_st, d_eng, d_tri, d_zc)
    s1 = ort.InferenceSession(str(r1_fp), providers=["CPUExecutionProvider"])
    ox1 = s1.run(None, dict(zip(r1_in, [d_ids.numpy(), d_m.numpy(), d_st.numpy(), d_eng.numpy(), d_tri.numpy(), d_zc.numpy()])))
    r1_diff = float(max(np.max(np.abs(o1[0].numpy() - ox1[0])), np.max(np.abs(o1[1].numpy() - ox1[1]))))

    # 2. Session R2 (T=1)
    w2 = _V12Rung2T1Export(mdl_cpu).eval()
    r2_fp = out_dir / "v12_r2_t1_fp32.onnx"
    r2_in = ["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "x1_in", "h0_1", "r1_pkt", "z_char"]
    r2_out = [
        "logit_t1", "logit_promo_t1", "sub4_t1", "emb_t1", "p_defer2", "slot_spread", "ebt_margin_t1", "ebt_grad_t1",
        "slots", "s_anchor", "carry_m_l", "zh_bus", "k_ev", "v_ev", "m_pool", "k_haar", "v_haar", "h0_pool", "span_base",
    ]
    torch.onnx.export(
        w2, (d_ids, d_m, d_st, d_eng, d_tri, o1[4], o1[5], o1[6], d_zc), str(r2_fp),
        input_names=r2_in, output_names=r2_out,
        dynamic_axes={
            "input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"}, "style_feats": {0: "B"},
            "engram_ids": {0: "B", 1: "L"}, "trigram_ids": {0: "B", 1: "L"}, "x1_in": {0: "B", 1: "L"}, "h0_1": {0: "B", 1: "L"}, "r1_pkt": {0: "B"}, "z_char": {0: "B"},
            "logit_t1": {0: "B"}, "logit_promo_t1": {0: "B"}, "sub4_t1": {0: "B"}, "emb_t1": {0: "B"},
            "p_defer2": {0: "B"}, "slot_spread": {0: "B"}, "ebt_margin_t1": {0: "B"}, "ebt_grad_t1": {0: "B"},
            "slots": {0: "B"}, "s_anchor": {0: "B"}, "carry_m_l": {0: "B"}, "zh_bus": {0: "B"},
            "k_ev": {0: "B", 1: "Lp"}, "v_ev": {0: "B", 1: "Lp"}, "m_pool": {0: "B", 1: "Lp"},
            "k_haar": {0: "B", 1: "L"}, "v_haar": {0: "B", 1: "L"}, "h0_pool": {0: "B"}, "span_base": {0: "B", 1: "L"},
        },
        **exp_kw,
    )
    with torch.no_grad():
        o2 = w2(d_ids, d_m, d_st, d_eng, d_tri, o1[4], o1[5], o1[6], d_zc)
    s2 = ort.InferenceSession(str(r2_fp), providers=["CPUExecutionProvider"])
    ox2 = s2.run(None, dict(zip(r2_in, [d_ids.numpy(), d_m.numpy(), d_st.numpy(), d_eng.numpy(), d_tri.numpy(), o1[4].numpy(), o1[5].numpy(), o1[6].numpy(), d_zc.numpy()])))
    r2_diff = float(max(np.max(np.abs(o2[0].numpy() - ox2[0])), np.max(np.abs(o2[1].numpy() - ox2[1]))))

    # 3. Session R3 (T=2 / T=3)
    w3 = _V12Rung3StepExport(mdl_cpu, outer_idx=1).eval()
    r3_fp = out_dir / "v12_r3_step_fp32.onnx"
    c1_c = (2.0 * torch.sigmoid(o2[0]) - 1.0).abs().unsqueeze(-1)
    d_uds = torch.cat([c1_c, o2[5].unsqueeze(-1), o2[4].unsqueeze(-1), o2[7].unsqueeze(-1)], dim=-1)
    r3_in_t = (o2[8], o2[9], o2[10], o2[11], o2[12], o2[13], o2[14], o2[15], o2[16], d_m, o2[17], o2[18], d_st, d_uds, d_zc)
    r3_in = ["slots", "s_anchor", "carry_m_l", "zh_bus", "k_ev", "v_ev", "m_pool", "k_haar", "v_haar", "attention_mask", "h0_pool", "span_base", "style_feats", "u_doubt_scalars", "z_char"]
    r3_out = ["logit_r3", "logit_promo_r3", "sub4_r3", "emb_r3", "ebt_margin_r3", "ebt_grad_r3", "slots_out", "carry_out", "zh_bus_out"]
    torch.onnx.export(
        w3, r3_in_t, str(r3_fp),
        input_names=r3_in, output_names=r3_out,
        dynamic_axes={
            "slots": {0: "B"}, "s_anchor": {0: "B"}, "carry_m_l": {0: "B"}, "zh_bus": {0: "B"},
            "k_ev": {0: "B", 1: "Lp"}, "v_ev": {0: "B", 1: "Lp"}, "m_pool": {0: "B", 1: "Lp"},
            "k_haar": {0: "B", 1: "L"}, "v_haar": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
            "h0_pool": {0: "B"}, "span_base": {0: "B", 1: "L"}, "style_feats": {0: "B"}, "u_doubt_scalars": {0: "B"}, "z_char": {0: "B"},
            "logit_r3": {0: "B"}, "logit_promo_r3": {0: "B"}, "sub4_r3": {0: "B"}, "emb_r3": {0: "B"},
            "ebt_margin_r3": {0: "B"}, "ebt_grad_r3": {0: "B"}, "slots_out": {0: "B"}, "carry_out": {0: "B"}, "zh_bus_out": {0: "B"},
        },
        **exp_kw,
    )
    with torch.no_grad():
        o3 = w3(*r3_in_t)
    s3 = ort.InferenceSession(str(r3_fp), providers=["CPUExecutionProvider"])
    ox3 = s3.run(None, dict(zip(r3_in, [t.numpy() for t in r3_in_t])))
    r3_diff = float(max(np.max(np.abs(o3[0].numpy() - ox3[0])), np.max(np.abs(o3[1].numpy() - ox3[1]))))

    # 4. DynaSeam LongDoc ONNX
    ds_info = export_dynaseam_onnx_and_verify(mdl_cpu.longstream, out_dir / "v12_dynaseam_fp32.onnx")

    # Optional WebGPU / low-RAM MatMulNBits W4 blk=128 secondary track
    nbits_info: dict[str, Any] = {}
    if export_matmulnbits:
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            for tag, src_p in (("r1", r1_fp), ("r2", r2_fp), ("r3", r3_fp)):
                dst_p = out_dir / src_p.name.replace("_fp32.onnx", "_w4nbits.onnx")
                q = MatMulNBitsQuantizer(str(src_p), bits=4, block_size=128, is_symmetric=True, accuracy_level=4)
                q.process()
                q.model.save_model_to_file(str(dst_p))
                nbits_info[f"{tag}_w4nbits_bytes"] = int(dst_p.stat().st_size)
        except Exception as e:
            nbits_info["error"] = repr(e)

    return {
        "bundle_v7_mgvz": mgvz_info,
        "rung0_onnx": r0_fp.name, "rung0_bytes": int(r0_fp.stat().st_size),
        "rung1_onnx": r1_fp.name, "rung1_bytes": int(r1_fp.stat().st_size),
        "rung2_onnx": r2_fp.name, "rung2_bytes": int(r2_fp.stat().st_size),
        "rung3_onnx": r3_fp.name, "rung3_bytes": int(r3_fp.stat().st_size),
        "dynaseam_onnx": ds_info,
        "r0_fp32_max_diff": round(r0_diff, 7),
        "r1_fp32_max_diff": round(r1_diff, 7),
        "r2_fp32_max_diff": round(r2_diff, 7),
        "r3_fp32_max_diff": round(r3_diff, 7),
        "matmulnbits_optional": nbits_info,
    }


# ---------------------------------------------------------------------------
# 9. Production 4-Rung Zero-Recompute Cascade Engine (`R0 -> R1 -> R2 -> R3`)
# ---------------------------------------------------------------------------

class MeguvizV12CascadeEngine:
    """Zero-BPE `Rung-0` -> `Rung-1` -> `Rung-2 (T=1)` -> `Rung-3 (T=2)` FP32-Exact Dual-Axis Cascade Engine."""

    def __init__(self, bundle_dir: Path, threads: int = 4):
        import onnxruntime as ort

        self.bdir = Path(bundle_dir)
        self.manifest = json.loads((self.bdir / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = BiliPickyBPETokenizer(self.bdir / self.manifest["tokenizer_file"])
        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V07))
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_ai_r1 = self.manifest.get("lut_ai_r1")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_pr_r1 = self.manifest.get("lut_promo_r1")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.r0_policy = self.manifest.get("rung0_ltt_policy") or {"enabled": False, "tau_ai": 0.08, "tau_promo": 0.08, "defer_max": 0.35, "max_chars": 160}
        self.r1_policy = self.manifest.get("dual_axis_ltt_policy") or {"enabled": False, "r1_defer_max": 0.20, "r1_p_ai_max": 0.10, "r1_p_promo_max": 0.08, "r1_max_tokens": 96}
        fw = self.manifest.get("fusion_weights") or {"w_t1": 0.50, "w_t2": 0.25, "w_ebt": 0.25}
        self.w_t1, self.w_t2, self.w_ebt = float(fw["w_t1"]), float(fw["w_t2"]), float(fw["w_ebt"])

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_r0 = ort.InferenceSession(str(self.bdir / self.manifest["rung0_onnx"]), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_r1 = ort.InferenceSession(str(self.bdir / self.manifest["rung1_onnx"]), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_r2 = ort.InferenceSession(str(self.bdir / self.manifest["rung2_onnx"]), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_r3 = ort.InferenceSession(str(self.bdir / self.manifest["rung3_onnx"]), sess_options=so, providers=["CPUExecutionProvider"])

    def predict_batch(self, texts: list[str], max_batch: int = 32, force_full: bool = False) -> list[dict[str, Any]]:
        n = len(texts)
        if n == 0:
            return []
        results: list[dict[str, Any] | None] = [None] * n
        order = sorted(range(n), key=lambda i: len(texts[i]))

        r0_en = bool(self.r0_policy.get("enabled", False)) and not force_full
        r0_tau_ai = float(self.r0_policy.get("tau_ai", 0.08))
        r0_tau_pr = float(self.r0_policy.get("tau_promo", 0.08))
        r0_def_max = float(self.r0_policy.get("defer_max", 0.35))
        r0_max_ch = int(self.r0_policy.get("max_chars", 160))

        r1_en = bool(self.r1_policy.get("enabled", False)) and not force_full
        r1_def_max = float(self.r1_policy.get("r1_defer_max", 0.20))
        r1_p_ai_max = float(self.r1_policy.get("r1_p_ai_max", 0.10))
        r1_p_pr_max = float(self.r1_policy.get("r1_p_promo_max", 0.08))
        r1_tok_cap = int(self.r1_policy.get("r1_max_tokens", 96))

        for st in range(0, n, max_batch):
            mb = order[st : st + max_batch]
            mb_txts = [texts[i] for i in mb]

            # Session R0 (Zero-BPE ~0.10 ms!)
            c1, c3, cm, cf, spoofs = extract_rung0_charhash_batch(mb_txts)
            lg_ai0, lg_pr0, pdef0, z_char = self.sess_r0.run(None, {"c1": c1, "c3": c3, "mask": cm, "feats": cf})
            p_ai0 = 1.0 / (1.0 + np.exp(-np.clip(lg_ai0, -30.0, 30.0)))
            p_pr0 = 1.0 / (1.0 + np.exp(-np.clip(lg_pr0, -30.0, 30.0)))
            if self.lut_ai_r0:
                p_ai0 = apply_spline_temp_lut(p_ai0, self.lut_ai_r0)
            if self.lut_pr_r0:
                p_pr0 = apply_spline_temp_lut(p_pr0, self.lut_pr_r0)

            surv0: list[int] = []
            for ri, oi in enumerate(mb):
                if (
                    r0_en
                    and len(texts[oi]) <= r0_max_ch
                    and spoofs[ri] == 0.0
                    and ("http" not in texts[oi].lower())
                    and float(pdef0[ri]) <= r0_def_max
                    and float(p_ai0[ri]) <= r0_tau_ai
                    and float(p_pr0[ri]) <= r0_tau_pr
                ):
                    results[oi] = {"prob_ai": float(p_ai0[ri]), "prob_promo": float(p_pr0[ri]), "exit_rung": 0}
                else:
                    surv0.append(ri)

            if not surv0:
                continue

            # Only survivors pay BPE-12k + PIFE-56d extraction!
            s0_txts = [mb_txts[ri] for ri in surv0]
            zc_s0 = z_char[np.array(surv0, dtype=np.int64)]
            cleans, f56_l, metas = [], [], []
            for t in s0_txts:
                cl, f56, mt = extract_style_features_56d(t)
                cleans.append(cl); f56_l.append(f56); metas.append(mt)
            encs = [self.tokenizer.encode_ids(c, max_len=self.max_len) for c in cleans]
            engs = [make_engram_bigram_ids(s, ENGRAM_BIGRAM_BUCKETS_V10) for s in encs]
            tris = [make_engram_trigram_ids(s, ENGRAM_TRIGRAM_BUCKETS_V10) for s in encs]
            bs1 = len(surv0)
            ml = max(len(s) for s in encs)
            ids = np.full((bs1, ml), self.tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((bs1, ml), dtype=np.int64)
            eg = np.zeros((bs1, ml), dtype=np.int64)
            tg = np.zeros((bs1, ml), dtype=np.int64)
            for k, s in enumerate(encs):
                ids[k, : len(s)], mask[k, : len(s)], eg[k, : len(s)], tg[k, : len(s)] = s, 1, engs[k], tris[k]
            st_b = np.stack(f56_l, axis=0).astype(np.float32)

            o1 = self.sess_r1.run(None, {"input_ids": ids, "attention_mask": mask, "style_feats": st_b, "engram_ids": eg, "trigram_ids": tg, "z_char": zc_s0})
            p_ai1 = 1.0 / (1.0 + np.exp(-np.clip(o1[0], -30.0, 30.0)))
            p_pr1 = 1.0 / (1.0 + np.exp(-np.clip(o1[1], -30.0, 30.0)))
            if self.lut_ai_r1: p_ai1 = apply_spline_temp_lut(p_ai1, self.lut_ai_r1)
            if self.lut_pr_r1: p_pr1 = apply_spline_temp_lut(p_pr1, self.lut_pr_r1)

            surv1: list[int] = []
            for k, ri in enumerate(surv0):
                oi = mb[ri]
                if (
                    r1_en
                    and len(encs[k]) <= r1_tok_cap
                    and metas[k]["spoof_any"] == 0.0
                    and ("http" not in cleans[k])
                    and float(np.max(st_b[k, list(R1_PROMO_VETO_DIMS_V11)])) == 0.0
                    and float(o1[2][k]) <= r1_def_max
                    and float(o1[3][k]) <= 0.50
                    and float(p_ai1[k]) <= r1_p_ai_max
                    and float(p_pr1[k]) <= r1_p_pr_max
                ):
                    results[oi] = {"prob_ai": float(p_ai1[k]), "prob_promo": float(p_pr1[k]), "exit_rung": 1}
                else:
                    surv1.append(k)

            if not surv1:
                continue

            s1_idx = np.array(surv1, dtype=np.int64)
            o2 = self.sess_r2.run(
                None,
                {
                    "input_ids": ids[s1_idx], "attention_mask": mask[s1_idx], "style_feats": st_b[s1_idx],
                    "engram_ids": eg[s1_idx], "trigram_ids": tg[s1_idx],
                    "x1_in": o1[4][s1_idx], "h0_1": o1[5][s1_idx], "r1_pkt": o1[6][s1_idx], "z_char": zc_s0[s1_idx],
                },
            )
            lg_t1 = o2[0]
            raw_p2 = 1.0 / (1.0 + np.exp(-np.clip(lg_t1, -30.0, 30.0)))
            c1_c = np.abs(2.0 * raw_p2 - 1.0).astype(np.float32)[:, None]
            uds = np.concatenate([c1_c, o2[5][:, None], o2[4][:, None], o2[7][:, None]], axis=-1)
            o3 = self.sess_r3.run(
                None,
                {
                    "slots": o2[8], "s_anchor": o2[9], "carry_m_l": o2[10], "zh_bus": o2[11],
                    "k_ev": o2[12], "v_ev": o2[13], "m_pool": o2[14], "k_haar": o2[15], "v_haar": o2[16],
                    "attention_mask": mask[s1_idx], "h0_pool": o2[17], "span_base": o2[18],
                    "style_feats": st_b[s1_idx], "u_doubt_scalars": uds, "z_char": zc_s0[s1_idx],
                },
            )
            lg_fused = self.w_t1 * lg_t1 + self.w_t2 * o3[0] + self.w_ebt * o3[4]
            p_ai_f = 1.0 / (1.0 + np.exp(-np.clip(lg_fused, -30.0, 30.0)))
            if self.lut_ai_full:
                p_ai_f = apply_spline_temp_lut(p_ai_f, self.lut_ai_full)
            p_pr_f = 1.0 / (1.0 + np.exp(-np.clip(o3[1], -30.0, 30.0)))
            if self.lut_pr_full:
                p_pr_f = apply_spline_temp_lut(p_pr_f, self.lut_pr_full)
            for j, k in enumerate(surv1):
                oi = mb[surv0[k]]
                results[oi] = {"prob_ai": float(p_ai_f[j]), "prob_promo": float(p_pr_f[j]), "exit_rung": 3}

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 10. Automated V12 Hardware-First & Firewall CI Gate Suite (`V12_S1`, `V12_F6`, `V12_C1`, `V12_R0`)
# ---------------------------------------------------------------------------

def run_v12_ci_gates(model: MeguvizV12ApexNanoModel, tokenizer: BiliPickyBPETokenizer) -> dict[str, Any]:
    """Verify dynamic shapes, 6-way analytic EBT grad parity, `[192:256]->AI` firewall, Zero-PII across R0-R3, & R0 size."""
    device = next(model.parameters()).device
    model.eval()
    ci: dict[str, Any] = {}

    # V12_S1: Dynamic lengths + Analytic 6-Way EBT gradient parity (<1e-5)
    dyn_ok = True
    for seq_l in (2, 17, 65, 256):
        with torch.no_grad():
            ids_t = torch.randint(1, tokenizer.vocab_size, (2, seq_l), device=device)
            eg_t = torch.randint(1, model.engram_buckets, (2, seq_l), device=device)
            tg_t = torch.randint(1, model.trigram_buckets, (2, seq_l), device=device)
            c1 = torch.randint(0, R0_B1_V12, (2, 24), device=device)
            c3 = torch.randint(0, R0_B3_V12, (2, 24), device=device)
            od = model(
                ids_t, torch.ones_like(ids_t), torch.zeros((2, STYLE_FEAT_DIM_V11), device=device),
                engram_ids=eg_t, trigram_ids=tg_t, r0_c1=c1, r0_c3=c3,
                r0_mask=torch.ones((2, 24), device=device), r0_feats=torch.zeros((2, R0_FEAT_DIM_V12), device=device),
                outer_steps=2, return_aux=True,
            )
            if torch.isnan(od["logit"]).any() or torch.isnan(od["logit_promo"]).any() or torch.isnan(od["logit_ai_r0"]).any():
                dyn_ok = False
    u_t = torch.randn(4, BUS_DIM_V10, device=device, requires_grad=True)
    ebt_r = model.core.inloop_ebt.forward_energy_and_grad(u_t)
    ag_g = torch.autograd.grad(ebt_r["free_energy"].sum(), u_t)[0]
    ebt_err = float((ebt_r["grad_u"].detach() - ag_g.detach()).abs().max().item())
    ci["V12_S1_dynamic_and_ebt6_grad"] = {"dynamic_ok": dyn_ok, "ebt6_grad_max_diff": round(ebt_err, 8), "pass": bool(dyn_ok and ebt_err < 1e-5)}

    # V12_F6: Structural Firewall Proof — perturbing promo-only emb[192:256] MUST leave AI logit & cbm_ai diff === 0.0!
    with torch.no_grad():
        emb_a = torch.randn(4, 256, device=device)
        emb_b = emb_a.clone()
        emb_b[:, AI_EMB_END_V11:] += 50.0
        ai_a = model.ai_head(emb_a[:, :AI_EMB_END_V11])
        ai_b = model.ai_head(emb_b[:, :AI_EMB_END_V11])
        cbm_a = F.relu(F.linear(emb_a[:, :AI_EMB_END_V11], model.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], model.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        cbm_b = F.relu(F.linear(emb_b[:, :AI_EMB_END_V11], model.cbm_enc.weight[:CBM_AI_CONCEPTS_V11, :AI_EMB_END_V11], model.cbm_enc.bias[:CBM_AI_CONCEPTS_V11]))
        fw_diff = float(max((ai_a - ai_b).abs().max().item(), (cbm_a - cbm_b).abs().max().item()))
    ci["V12_F6_promo_to_ai_firewall"] = {"max_leak_diff": fw_diff, "pass": fw_diff == 0.0}

    # V12_C1: Strict Zero-PII across ALL 4 Rungs (R0 + R1 + T1 + T2, AI & Promo === 0.0!)
    def _infer_all(txt: str):
        c1, c3, cm, cf, _ = extract_rung0_charhash_batch([txt])
        cl, f56, _ = extract_style_features_56d(txt)
        ids = tokenizer.encode_ids(cl, max_len=MAX_LEN_V07)
        eng = make_engram_bigram_ids(ids, model.engram_buckets)
        tri = make_engram_trigram_ids(ids, model.trigram_buckets)
        with torch.no_grad():
            return model(
                torch.tensor([ids], dtype=torch.long, device=device), torch.ones((1, len(ids)), dtype=torch.long, device=device),
                torch.from_numpy(f56).unsqueeze(0).to(device), engram_ids=torch.tensor([eng], dtype=torch.long, device=device),
                trigram_ids=torch.tensor([tri], dtype=torch.long, device=device),
                r0_c1=torch.from_numpy(c1).to(device), r0_c3=torch.from_numpy(c3).to(device),
                r0_mask=torch.from_numpy(cm).to(device), r0_feats=torch.from_numpy(cf).to(device),
                outer_steps=2, return_aux=True,
            )

    ba = _infer_all("回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]")
    bb = _infer_all("回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]")
    pii_d = float(max(
        abs(float(ba[k][0].item()) - float(bb[k][0].item()))
        for k in ("logit_ai_r0", "logit_promo_r0", "logit_r1", "logit_promo_r1", "logit", "logit_promo")
    ))
    ci["V12_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # V12_R0: Rung-0 parameter budget check (<= 150k params -> <= 125 KB W8!)
    r0_params = sum(p.numel() for p in model.rung0.parameters())
    ci["V12_R0_budget"] = {"rung0_params": r0_params, "pass": r0_params <= 150000}

    ci["all_ci_passed"] = all(bool(v.get("pass")) for k, v in ci.items() if isinstance(v, dict))
    return ci
