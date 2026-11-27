"""MeguvizCAC v1.3.0-XtremeApex: Full Nested-Loop <=5.00 MiB On-Device Dual-Axis Detector.

Preserves and FULLY UNLOCKS MeguvizCAC's signature Nested-Loop Symplectic Architecture
(`H=384, FFN=896, 6 heads, 16 slots [4 slow + 6 sem-fast + 6 span-fast], k_inner=2 fast-inner x outer_steps in {2, 3}`)
at clean W4/W8 quantization (ZERO `1.58-bit` ternary loop-error compounding!) fitting comfortably `<= 5.00 MiB`:

  1. `ApexCharNanoRung0V130` (`~1.0 MiB` pure W8/INT8, ZERO sub-2-bit):
     Char Unigram `[4096, 64]` + Char Bigram `[16384, 32]` + Char Trigram `[8192, 16]` + `^/$` boundaries +
     Head-140/Tail-50 + 4-scale Dilated DWConv (`k=3, k=5 d=1/2/4`) + SE + Mask-Safe 5-Stat Pool + `z_char[64]`.
  2. Pad-Exact `H=384` Prelude (`L0` + `L1`) with Bigram+Trigram Engrams + Mask-Before-DWConv (`< 1e-4` pad invariance)
     + Pad-Exact Fine-Scale Haar Wavelet K/V (`64-d`) + Text Evidence K/V (`384-d`).
  3. Signature Full Nested-Loop Core (`NestedSymplectCoreV130`, Clean W4/W8):
     - True `k_inner=2` fast-inner steps per outer step with Fast-Slot Dual-Gate Pre/Post (`z_fast - mean(z_slow)`)
       and Cayley Symplectic Peer Exchange (`c^2 + s^2 === 1`) between 6 `sem_fast` & 6 `span_fast` slots!
     - Surprise-Gated Compressed EMA Fast Memory `carry_m_l (12 x 64)` across outer steps `T1 -> T2 -> T3`.
     - Relaxed Recursive Specialization (`rank=16` LoRA, ICLR 2025): `3 outer x 3 inner` FP16 core mixers + AdaLN
       + per-outer-step `rank=16` LoRA on Active Text Reread `ca_q` & Slot Attention `sa_o`!
     - 3-Role Outer-Step History Delta (`0.1*tanh(g_r)*(mean_r(S_t) - mean_r(S_{t-1}))`) for `slow/sem_fast/span_fast`.
     - Active Text K/V + Fine-Scale Haar Reread + L0 Penetration + Analytic 6-Valley EBT (`< 1e-5` autograd parity).
     - Ouro-Style Conditional `T2 -> T3` Halting / Escalation Gate (`escalate_t3_logit`, `p_escalate_t3`) trained
       explicitly on error-reduction utility so ~85-90% exit at `T2` and hard cases run zero-redundancy `T3`!
  4. Strictly Disjoint `256-d` Coda (`[0:128)` shared, `[128:192)` AI-private, `[192:256)` Promo-private)
     + Split-Axis Top-K CBM Firewall (`+12` Gov/Promo bias perturb -> `0.000000` AI diff!)
     + `T1 -> T_final` Trajectory Readout + Promo Char Bypass + Native `DynaSeam` Long-Doc API.
  5. 4-Stage Modular FP32-Exact ONNX (`r0_apexchar`, `t2_nested`, conditional `t3_correct`, `dynaseam`) + `MGVZ8 <= 5.00 MiB`!
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
import onnxruntime as ort
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v11_features import (
    CHUNK_META_DIM_V11,
    STYLE_FEAT_DIM_V11,
    _CONTACT_ID_PII_RE,
    extract_style_features_56d,
    plan_long_chunks_v11,
)
from .v121_apexnano import SoftBallLogitNormHeadV121

# ---------------------------------------------------------------------------
# Constants & Exact Disjoint Axis Layout (`256-d` final embedding!)
# ---------------------------------------------------------------------------
R0_B1_V130 = 4096
R0_B2_V130 = 16384
R0_B3_V130 = 8192
R0_D1_V130 = 64
R0_D2_V130 = 32
R0_D3_V130 = 16
R0_HID_V130 = 128
R0_ZCHAR_DIM_V130 = 64  # [0:32] shared/AI-char, [32:64] promo-dedicated-char
R0_FEAT_DIM_V130 = 16
R0_MAX_CHARS_V130 = 192
R0_HEAD_CHARS_V130 = 140

H_V130 = 384
FFN_V130 = 896
NUM_HEADS_V130 = 6
NUM_SLOTS_V130 = 16     # 4 slow + 6 semantic-fast + 6 span-fast (exact Meguviz signature!)
NUM_SLOW_V130 = 4
NUM_SEM_FAST_V130 = 6
NUM_SPAN_FAST_V130 = 6
NUM_FAST_V130 = 12
K_INNER_V130 = 2
LORA_RANK_V130 = 16
BUS_DIM_V130 = 64
ENGRAM_BUCKETS_V130 = 8192
TRIGRAM_BUCKETS_V130 = 8192
MAX_LEN_V130 = 256
HEAD_LEN_V130 = 180

# Strictly Disjoint 256-d Embedding Layout:
#   [0:128)   = Shared text/stylometry evidence (128-d)
#   [128:192) = AI-private evidence (64-d)       -> AI reads ONLY [0:192) (192-d)
#   [192:256) = Promo-private evidence (64-d)    -> Promo reads [0:128) + [192:256) (192-d)
SHARED_END_V130 = 128
AI_END_V130 = 192
EMB_DIM_V130 = 256
PR_EMB_DIM_V130 = SHARED_END_V130 + (EMB_DIM_V130 - AI_END_V130)  # 192
CBM_AI_K_V130 = 64
CBM_PR_K_V130 = 64

BUNDLE_MAX_MIB_V130 = 5.00
MGVZ8_MAGIC = b"MGVZ8\x00\x00\x00"


# ---------------------------------------------------------------------------
# 1. Clean W4 / W8 Fake-Quant Linear & Embedding (ZERO 1.58-bit Distortion!)
# ---------------------------------------------------------------------------

class QuantLinearV130(nn.Module):
    """Symmetric group-wise W4 (`gs=64`) or per-channel W8 linear with STE QAT."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 4, group_size: int = 64) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.target_bits = int(bits)
        self.group_size = int(group_size)
        self.qat_active = False
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.target_bits >= 16:
            return w
        if self.target_bits == 8:
            amax = w.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            sc = amax / 127.0
            q = torch.clamp(torch.round(w / sc), -127.0, 127.0) * sc
            return w + (q - w).detach()
        orig_shape = w.shape
        flat = w.reshape(-1)
        gs = self.group_size
        pad = (gs - (flat.numel() % gs)) % gs
        flat_p = F.pad(flat, (0, pad)) if pad else flat
        g = flat_p.reshape(-1, gs)
        amax = g.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        sc = amax / 7.0
        q = torch.clamp(torch.round(g / sc), -7.0, 7.0) * sc
        q_flat = q.reshape(-1)[: flat.numel()].reshape(orig_shape)
        return w + (q_flat - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._quant_weight(self.weight) if self.qat_active else self.weight
        return F.linear(x, w, self.bias)


class QuantEmbeddingV130(nn.Module):
    """Per-row W8 or group-W4 embedding with STE QAT."""

    def __init__(self, num_embeddings: int, embedding_dim: int, bits: int = 8, padding_idx: int | None = None, group_size: int = 64) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.target_bits = int(bits)
        self.group_size = int(group_size)
        self.padding_idx = padding_idx
        self.qat_active = False
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()

    def _quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.target_bits >= 16:
            return w
        if self.target_bits == 8:
            amax = w.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            sc = amax / 127.0
            q = torch.clamp(torch.round(w / sc), -127.0, 127.0) * sc
            return w + (q - w).detach()
        orig_shape = w.shape
        flat = w.reshape(-1)
        gs = self.group_size
        pad = (gs - (flat.numel() % gs)) % gs
        flat_p = F.pad(flat, (0, pad)) if pad else flat
        g = flat_p.reshape(-1, gs)
        amax = g.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        sc = amax / 7.0
        q = torch.clamp(torch.round(g / sc), -7.0, 7.0) * sc
        q_flat = q.reshape(-1)[: flat.numel()].reshape(orig_shape)
        return w + (q_flat - w).detach()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        w = self._quant_weight(self.weight) if self.qat_active else self.weight
        return F.embedding(idx, w, padding_idx=self.padding_idx)


def set_v130_qat_mode(model: nn.Module, active: bool) -> None:
    for m in model.modules():
        if isinstance(m, (QuantLinearV130, QuantEmbeddingV130)):
            m.qat_active = bool(active)


# ---------------------------------------------------------------------------
# 2. Zero-PII Frontend: Head-140/Tail-50 Char-1/2/3 Hash + Head-180/Tail-74 BPE
# ---------------------------------------------------------------------------

def canonicalize_zero_pii_v130(text: str, max_chars: int = 8192) -> tuple[str, dict[str, float]]:
    cl, meta = canonicalize_zero_pii_v07(str(text or ""), max_chars=max_chars)
    return _CONTACT_ID_PII_RE.sub(r"\1<CONTACT_ID>", cl), meta


def encode_bpe_head_tail_v130(
    tokenizer: BiliPickyBPETokenizer,
    clean_text: str,
    max_len: int = MAX_LEN_V130,
    head_len: int = HEAD_LEN_V130,
) -> list[int]:
    """Encode BPE with Head-180 + Tail-74 preservation so both hooks and AI closings survive!"""
    full_ids = tokenizer.encode_ids(clean_text, max_len=max(max_len * 4, 1024))
    if len(full_ids) <= max_len:
        return full_ids
    h_cap = min(int(head_len), max_len - 4)
    t_cap = max_len - h_cap
    return full_ids[:h_cap] + full_ids[-t_cap:]


def extract_apex_char_batch_v130(
    texts: list[str],
    max_chars: int = R0_MAX_CHARS_V130,
    head_chars: int = R0_HEAD_CHARS_V130,
    already_canonical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-PII Head-140/Tail-50 char 1/2/3-gram hash with `^`/`$` boundaries + 16-d structural scalars."""
    n = len(texts)
    if n == 0:
        return (
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, R0_FEAT_DIM_V130), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    cleans: list[str] = []
    spoofs = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(texts):
        if already_canonical:
            cleans.append(str(t))
        else:
            cl, sp_meta = canonicalize_zero_pii_v130(str(t))
            cleans.append(cl)
            if isinstance(sp_meta, dict):
                spoofs[i] = 1.0 if (
                    sp_meta.get("spoof_any", 0.0) or sp_meta.get("zw_count", 0) or sp_meta.get("homoglyph_count", 0)
                ) else 0.0

    body_cap = max(8, int(max_chars) - 2)
    h_cap = min(int(head_chars), body_cap - 4)
    t_cap = body_cap - h_cap

    trimmed: list[str] = []
    for cl in cleans:
        if len(cl) > body_cap:
            trimmed.append(cl[:h_cap] + cl[-t_cap:])
        else:
            trimmed.append(cl)

    ml = max(8, max(len(s) + 2 for s in trimmed))
    c1 = np.zeros((n, ml), dtype=np.int64)
    c2 = np.zeros((n, ml), dtype=np.int64)
    c3 = np.zeros((n, ml), dtype=np.int64)
    mask = np.zeros((n, ml), dtype=np.float32)
    feats = np.zeros((n, R0_FEAT_DIM_V130), dtype=np.float32)

    BOS_CP = np.uint32(2)
    EOS_CP = np.uint32(3)
    for i, s in enumerate(trimmed):
        if s:
            body_codes = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
            codes = np.concatenate([np.array([BOS_CP], dtype=np.uint32), body_codes, np.array([EOS_CP], dtype=np.uint32)])
        else:
            codes = np.array([BOS_CP, EOS_CP], dtype=np.uint32)
        L = len(codes)
        mask[i, :L] = 1.0

        u1 = (((codes * np.uint32(2654435761)) >> np.uint32(20)) % np.uint32(R0_B1_V130 - 1)) + np.uint32(1)
        c1[i, :L] = u1

        if L >= 2:
            h2 = (codes[:-1] * np.uint32(1315423911)) ^ (codes[1:] * np.uint32(2654435761))
            u2 = ((h2 >> np.uint32(18)) % np.uint32(R0_B2_V130 - 1)) + np.uint32(1)
            c2[i, : L - 1] = u2
            c2[i, L - 1] = u2[-1]

        if L >= 3:
            h3 = (
                (codes[:-2] * np.uint32(1315423911))
                ^ (codes[1:-1] * np.uint32(2654435761))
                ^ (codes[2:] * np.uint32(2246822519))
            )
            u3 = ((h3 >> np.uint32(19)) % np.uint32(R0_B3_V130 - 1)) + np.uint32(1)
            c3[i, : L - 2] = u3
            c3[i, L - 2 : L] = u3[-1]
        else:
            c3[i, :L] = u1

        full_cl = cleans[i]
        Lf = max(1, len(full_cl))
        bc = np.frombuffer(full_cl[:512].encode("utf-32-le"), dtype=np.uint32) if full_cl else np.zeros((1,), dtype=np.uint32)
        Lb = max(1, len(bc))
        inv_l = 1.0 / float(Lb)
        cjk = float(np.sum((bc >= 0x4E00) & (bc <= 0x9FFF)))
        alnum = float(np.sum(((bc >= 48) & (bc <= 57)) | ((bc >= 65) & (bc <= 90)) | ((bc >= 97) & (bc <= 122))))
        digits = float(np.sum((bc >= 48) & (bc <= 57)))
        spaces = float(np.sum((bc == 32) | (bc == 9)))
        newlines = float(np.sum(bc == 10))
        cn_comma = float(np.sum((bc == 0xFF0C) | (bc == 44)))
        cn_period = float(np.sum(bc == 0x3002))
        cn_colon = float(np.sum((bc == 0xFF1A) | (bc == 58)))
        cn_semi = float(np.sum((bc == 0xFF1B) | (bc == 59)))
        emdash = float(np.sum((bc == 0x2014) | (bc == 0x2015)))
        quote_cn = float(np.sum((bc == 0x300C) | (bc == 0x300D) | (bc == 0x201C) | (bc == 0x201D)))
        bracket_emo = float(np.sum(bc == 91))
        qmark = float(np.sum((bc == 63) | (bc == 0xFF1F)))
        exclam = float(np.sum((bc == 33) | (bc == 0xFF01)))
        ends_per = 1.0 if full_cl.strip().endswith("。") else 0.0
        uniq_ratio = float(len(np.unique(bc))) * inv_l

        feats[i] = np.array([
            math.log1p(Lf) / 6.0,
            cjk * inv_l,
            alnum * inv_l,
            digits * inv_l,
            spaces * inv_l,
            min(1.0, newlines / 6.0),
            min(1.0, cn_comma * inv_l * 10.0),
            min(1.0, cn_period * inv_l * 10.0),
            min(1.0, cn_colon * 0.25),
            min(1.0, cn_semi * 0.25),
            min(1.0, emdash * 0.25),
            min(1.0, quote_cn * 0.25),
            min(1.0, bracket_emo * 0.25),
            min(1.0, (qmark + exclam) * inv_l * 10.0),
            ends_per,
            uniq_ratio,
        ], dtype=np.float32)

    return c1, c2, c3, mask, feats, spoofs


# ---------------------------------------------------------------------------
# 3. Stage 0: Full-Scale `ApexCharNanoRung0V130` (`~1.0 MiB` Pure W8/INT8)
# ---------------------------------------------------------------------------

class ApexCharNanoRung0V130(nn.Module):
    """High-res Char 1/2/3-gram + 4-Scale Dilated DWConv + SE + Mask-Safe 5-Stat Pool (Pure W8/INT8)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb1 = QuantEmbeddingV130(R0_B1_V130, R0_D1_V130, bits=8, padding_idx=0)
        self.emb2 = QuantEmbeddingV130(R0_B2_V130, R0_D2_V130, bits=8, padding_idx=0)
        self.emb3 = QuantEmbeddingV130(R0_B3_V130, R0_D3_V130, bits=8, padding_idx=0)
        in_d = R0_D1_V130 + R0_D2_V130 + R0_D3_V130  # 64 + 32 + 16 = 112

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
        pool_in = R0_HID_V130 * 5 + R0_FEAT_DIM_V130  # 656
        # Strictly split AI vs Promo trunks inside R0 so Promo never contaminates AI R0!
        self.ai_fc = nn.Sequential(QuantLinearV130(pool_in, 128, bits=8), nn.LayerNorm(128), nn.SiLU())
        self.promo_fc = nn.Sequential(QuantLinearV130(pool_in, 96, bits=8), nn.LayerNorm(96), nn.SiLU())
        self.ai_head = nn.Linear(128, 1)
        self.promo_head = nn.Linear(96, 1)
        self.z_char_ai = nn.Linear(128, 32)
        self.z_char_pr = nn.Linear(96, 32)
        self.defer_head = nn.Sequential(nn.Linear(128 + 96 + 2, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor) -> dict[str, torch.Tensor]:
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

        a_lg = self.attn_pool(h).squeeze(-1)
        a_lg = a_lg.masked_fill(m < 0.5, -1e4)
        a_w = torch.softmax(a_lg.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        p_attn = (h * a_w).sum(dim=1)

        p_first = h[:, 0, :]
        last_idx = (m.sum(dim=1).long() - 1).clamp(min=0, max=h.shape[1] - 1)
        p_last = h.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[2])).squeeze(1)

        pooled = torch.cat([p_mean, p_max, p_attn, p_first, p_last, feats.to(h.dtype)], dim=-1)
        h_ai = self.ai_fc(pooled)
        h_pr = self.promo_fc(pooled)

        lg_ai = self.ai_head(h_ai).squeeze(-1)
        lg_pr = self.promo_head(h_pr).squeeze(-1)
        z_char = torch.cat([self.z_char_ai(h_ai), self.z_char_pr(h_pr)], dim=-1)  # (B, 64)
        def_lg = self.defer_head(
            torch.cat([h_ai.detach(), h_pr.detach(), lg_ai.detach().unsqueeze(-1), lg_pr.detach().unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        return {
            "logit_ai_r0": lg_ai,
            "logit_promo_r0": lg_pr,
            "defer_r0_logit": def_lg,
            "p_defer_r0": torch.sigmoid(def_lg),
            "z_char": z_char,
        }


# ---------------------------------------------------------------------------
# 4. Stage 1 & 2: Pad-Exact Prelude + Full Nested-Loop Symplectic Core + EBT6
# ---------------------------------------------------------------------------

class PadExactConvBlockV130(nn.Module):
    """Mask-before-DWConv local+global token mixer (guarantees <1e-4 batch-padding invariance!)."""

    def __init__(self, dim: int = H_V130, ffn_dim: int = FFN_V130, num_heads: int = NUM_HEADS_V130, bits: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = QuantLinearV130(dim, dim * 3, bits=bits)
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.out_proj = QuantLinearV130(dim, dim, bits=bits)
        self.norm2 = nn.LayerNorm(dim)
        self.w_up = QuantLinearV130(dim, ffn_dim * 2, bits=bits)
        self.w_down = QuantLinearV130(ffn_dim, dim, bits=bits)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        m = mask.float()
        m_u = m.unsqueeze(-1)
        xn = self.norm1(x) * m_u
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        v_clean = v * m_u
        v_loc = self.dw(v_clean.transpose(1, 2)).transpose(1, 2) * m_u

        qh = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = (v_clean + 0.35 * v_loc).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(m[:, None, None, :] < 0.5, -1e4)
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        ctx = torch.matmul(attn, vh).transpose(1, 2).reshape(B, L, D)
        x = (x + self.out_proj(ctx)) * m_u

        h = self.norm2(x) * m_u
        u, g = self.w_up(h).chunk(2, dim=-1)
        x = (x + self.w_down(u * F.silu(g))) * m_u
        return x


class InLoopAnalyticEBTV130(nn.Module):
    """6-Valley Analytic Energy-Based Refiner on 64-d Slow Bus (exact autograd parity <1e-5!)."""

    def __init__(self, bus_dim: int = BUS_DIM_V130, hidden: int = 96, eta_init: float = 0.25) -> None:
        super().__init__()
        self.w1 = nn.Linear(bus_dim, hidden)
        self.w2 = nn.Linear(hidden, 6)
        self.log_eta = nn.Parameter(torch.tensor(math.log(eta_init)))

    def forward_energy_and_grad(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = self.w1(u)
        sig = torch.sigmoid(z1)
        h1 = z1 * sig
        dh1 = sig + z1 * sig * (1.0 - sig)
        e = self.w2(h1)  # (B, 6)
        free_e = -torch.logsumexp(-e, dim=-1)
        p_e = torch.softmax(-e, dim=-1)
        g_h1 = torch.matmul(p_e, self.w2.weight)
        g_z1 = g_h1 * dh1
        grad_u = torch.matmul(g_z1, self.w1.weight)
        gnorm = grad_u.norm(dim=-1, keepdim=True)
        eta = self.log_eta.exp().clamp(0.02, 0.80)
        u_next = u - eta * grad_u / (1.0 + gnorm)
        # Valley layout: e0=Organic Human, e1=Assisted/Quote, e2/e3/e4=AI modes, e5=Human Commercial Promo!
        e_hu = torch.minimum(e[:, 0], e[:, 5])
        e_ai = torch.minimum(torch.minimum(e[:, 2], e[:, 3]), e[:, 4])
        ai_margin = e_hu - e_ai
        pr_margin = e[:, 0] - e[:, 5]
        return {
            "energies": e,
            "free_energy": free_e,
            "grad_u": grad_u,
            "grad_norm": gnorm.squeeze(-1),
            "u_next": u_next,
            "ai_margin": ai_margin,
            "promo_margin": pr_margin,
        }


class RelaxedRecursivePhaseAdapterV130(nn.Module):
    """ICLR-2025 Relaxed Recursive LoRA (`3 outer x 3 inner = 9` exact phase/step specialization states!).

    Shares a single orthogonal `rank=16` basis `(basis_u, basis_v)` while granting every `(outer_idx, phase_idx)`
    its own FP16 `16x16` core mixer, SiLU blend, and AdaLN `(scale, shift)` at tiny byte cost (~22 KB)!
    """

    def __init__(self, dim: int = H_V130, rank: int = LORA_RANK_V130, n_outer: int = 3, n_phases: int = 3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.n_outer = int(n_outer)
        self.n_phases = int(n_phases)
        self.n_states = self.n_outer * self.n_phases

        self.basis_u = QuantLinearV130(dim, rank, bias=False, bits=8)
        self.basis_v = QuantLinearV130(rank, dim, bias=False, bits=8)
        with torch.no_grad():
            uq = torch.empty(rank, dim)
            nn.init.orthogonal_(uq)
            self.basis_u.weight.copy_(uq * 0.5)
            vq = torch.empty(dim, rank)
            nn.init.orthogonal_(vq)
            self.basis_v.weight.copy_(vq * 0.05)

        self.state_core = nn.Parameter(torch.zeros(self.n_states, rank, rank))
        self.state_bias = nn.Parameter(torch.zeros(self.n_states, rank))
        self.state_silu_gain = nn.Parameter(torch.zeros(self.n_states))
        self.state_diag = nn.Parameter(torch.ones(self.n_states, rank))
        self.adaln_scale = nn.Parameter(torch.zeros(self.n_states, 1, dim))
        self.adaln_shift = nn.Parameter(torch.zeros(self.n_states, 1, dim))
        self.dyn_ctrl = nn.Parameter(torch.zeros(self.n_outer, 6))

    def _sid(self, outer_idx: int, phase_idx: int) -> int:
        oi = min(max(int(outer_idx), 0), self.n_outer - 1)
        pi = min(max(int(phase_idx), 0), self.n_phases - 1)
        return oi * self.n_phases + pi

    def modulate(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        sid = self._sid(outer_idx, phase_idx)
        return x * (1.0 + 0.25 * torch.tanh(self.adaln_scale[sid])) + 0.25 * torch.tanh(self.adaln_shift[sid])

    def delta(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        sid = self._sid(outer_idx, phase_idx)
        z = self.basis_u(x)
        g = torch.sigmoid(self.state_silu_gain[sid])
        z_nl = (1.0 - g) * z + g * F.silu(z)
        h = torch.matmul(z_nl, self.state_core[sid].transpose(0, 1)) + self.state_bias[sid]
        h = h * self.state_diag[sid]
        return self.basis_v(h)


class NestedSymplectCoreV130(nn.Module):
    """Full-Scale `H=384, FFN=896, 16-Slot` Nested-Loop Symplectic Core (Clean W4/W8, ZERO 1.58-bit!).

    Inside EVERY outer step `outer_idx in {0 (T1), 1 (T2), 2 (T3)}`:
      1. 3-Role Outer History Delta (`slow / sem_fast / span_fast`) + Slow-Bus Doubt Handoff (`outer_idx >= 1`).
      2. Active Text K/V Reread (with per-outer-step `rank=16` LoRA on `ca_q`) + Fine-Scale Haar Wavelet Reread!
      3. Surprise-Gated EMA Compressed Fast Memory (`carry_m_l`, `[B, 12, 64]`) injection into fast slots.
      4. TRUE NESTED INNER LOOP (`k_inner = 2` fast-inner iterations):
         - Dual-Gated Pre/Post conditioned on `z_fast - mean(z_slow)`
         - Shared W4 SwiGLU FFN + Relaxed Recursive `(outer_idx, phase_idx)` LoRA specialization
         - Exact Cayley Symplectic Peer Coupling (`c^2 + s^2 === 1`) between 6 `sem_fast` & 6 `span_fast` slots!
      5. EMA Fast-Memory Update (`new_carry_m_l`).
      6. All-16-Slot Slow+Fast Self-Attention (with per-outer-step `rank=16` LoRA on `sa_o`) + Shared SwiGLU FFN (`phase=2`).
      7. L0 Surface-Rhythm Penetration into 4 Slow Slots + Analytic 6-Valley EBT refinement on 64-d Slow Bus.
      8. Manifold Anchor Blend + Step-Indexed Spectral Damping (`T1` full, `T2/T3` smoothly damped).
    """

    def __init__(self, dim: int = H_V130, ffn_dim: int = FFN_V130, num_heads: int = NUM_HEADS_V130) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.n_slow = NUM_SLOW_V130
        self.n_sem = NUM_SEM_FAST_V130
        self.n_span = NUM_SPAN_FAST_V130
        self.n_fast = NUM_FAST_V130

        self.slot_queries = nn.Parameter(torch.randn(1, NUM_SLOTS_V130, dim) * 0.04)

        # Active Text K/V Cross-Attention + Per-Outer-Step Relaxed LoRA (`T1/T2/T3` reread specialization!)
        self.norm_ca = nn.LayerNorm(dim)
        self.ca_q = QuantLinearV130(dim, dim, bits=4)
        self.ca_o = QuantLinearV130(dim, dim, bits=4)
        self.ca_q_lora_a = nn.Parameter(torch.randn(3, dim, LORA_RANK_V130) * 0.02)
        self.ca_q_lora_b = nn.Parameter(torch.zeros(3, LORA_RANK_V130, dim))
        self.reread_gate = nn.Parameter(torch.tensor(0.3))

        # Fine-Scale Haar Wavelet Reread (`64-d`)
        self.haar_q = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.haar_o = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.haar_gate = nn.Parameter(torch.tensor(-0.8))

        # Shared All-16-Slot Self-Attention + Per-Outer-Step Relaxed LoRA on `sa_o`
        self.norm_sa = nn.LayerNorm(dim)
        self.sa_qkv = QuantLinearV130(dim, dim * 3, bits=4)
        self.sa_o = QuantLinearV130(dim, dim, bits=4)
        self.sa_o_lora_a = nn.Parameter(torch.randn(3, dim, LORA_RANK_V130) * 0.02)
        self.sa_o_lora_b = nn.Parameter(torch.zeros(3, LORA_RANK_V130, dim))

        # Shared SwiGLU FFN (reused across `k_inner=2` fast steps + outer slow step = up to 9x!) + Relaxed Adapter
        self.norm_ff = nn.LayerNorm(dim)
        self.ff_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.ff_down = QuantLinearV130(ffn_dim, dim, bits=4)
        self.relax = RelaxedRecursivePhaseAdapterV130(dim=dim, rank=LORA_RANK_V130, n_outer=3, n_phases=3)

        # True Nested Fast-Inner Dual-Gate Pre/Post (`z_fast - mean(z_slow)`)
        self.fast_norm = nn.LayerNorm(dim)
        self.p2_pre_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_pre_b = QuantLinearV130(16, dim, bits=4)
        self.p2_post_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_post_b = QuantLinearV130(16, dim, bits=4)
        self.fast_res_scale = nn.Parameter(torch.full((dim,), 0.35))

        # 64-d Bus + Cayley Symplectic Peer Coupling (`c^2 + s^2 === 1` between sem_fast & span_fast!)
        self.bus_down = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.bus_up = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.c1_q = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V130))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.22))

        # Surprise-Gated EMA Compressed Fast Memory (`carry_m_l`)
        self.c2_beta_logit = nn.Parameter(torch.full((BUS_DIM_V130,), 0.4))
        self.c2_carry_scale = nn.Parameter(torch.tensor(0.24))
        self.c2_surprise_mlp = nn.Sequential(nn.Linear(BUS_DIM_V130 + 1, 16), nn.SiLU(), nn.Linear(16, 1))

        # 3-Role Outer-Step History Delta Injection (`slow`, `sem_fast`, `span_fast`) + Doubt Handoff
        self.role_hist_g = nn.Parameter(torch.zeros(3, 1, dim))
        self.handoff_proj = QuantLinearV130(BUS_DIM_V130 + 2, dim, bits=4)

        # L0 Surface Penetration into 4 Slow Slots
        self.l0_pen_a = QuantLinearV130(dim, 24, bias=False, bits=8)
        self.l0_pen_b = QuantLinearV130(24, dim, bias=False, bits=8)
        self.l0_pen_gate = nn.Parameter(torch.full((dim,), -1.2))

        # Analytic 6-Valley EBT on 64-d Slow Bus
        self.ebt = InLoopAnalyticEBTV130(BUS_DIM_V130, hidden=96)
        self.ebt_back_scale = nn.Parameter(torch.tensor(0.20))

        # Manifold Anchor Blend + Step-Indexed Spectral Damping
        self.anchor_alpha = nn.Parameter(torch.full((1, 1, dim), -1.4))
        self.damp_gamma = nn.Parameter(torch.tensor(1.4))

    def _shared_ffn(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        xm = self.relax.modulate(x, outer_idx, phase_idx)
        u, g = self.ff_up(xm).chunk(2, dim=-1)
        return self.ff_down(u * F.silu(g)) + self.relax.delta(xm, outer_idx, phase_idx)

    def _cayley_peer_couple(self, z_l: torch.Tensor, cayley_delta: torch.Tensor) -> torch.Tensor:
        u = self.bus_down(z_l)
        u_a, u_b = u[:, : self.n_sem, :], u[:, self.n_sem :, :]
        sc_ab = torch.matmul(self.c1_q(u_a), self.c1_k(u_b).transpose(-1, -2)) / 8.0
        msg_to_a = torch.matmul(torch.softmax(sc_ab.float(), dim=-1).to(u.dtype), u_b)
        msg_to_b = torch.matmul(torch.softmax(sc_ab.transpose(-1, -2).float(), dim=-1).to(u.dtype), u_a)

        g = (self.c1_cayley_gamma.unsqueeze(0) + 0.15 * torch.tanh(cayley_delta)).clamp(-2.5, 2.5)
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c = ((1.0 - g2) * inv).unsqueeze(1)
        s = ((2.0 * g) * inv).unsqueeze(1)

        rot_a = c * u_a + s * msg_to_a
        rot_b = -s * msg_to_b + c * u_b
        delta = self.bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_l + (0.35 * torch.tanh(self.c1_bridge_scale)) * delta

    def _fast_inner_step(self, slots: torch.Tensor, ctrl: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        z_h = slots[:, : self.n_slow, :]
        z_l = slots[:, self.n_slow :, :]
        zh_ref = z_h.mean(dim=1, keepdim=True).expand_as(z_l)

        zn = self.fast_norm(z_l)
        g_pre = torch.sigmoid(self.p2_pre_b(F.silu(self.p2_pre_a(torch.cat([zn, zn - zh_ref], dim=-1)))) + 0.25 * ctrl[2])
        delta_l = self._shared_ffn(zn * g_pre, outer_idx=outer_idx, phase_idx=phase_idx)
        g_post = torch.sigmoid(self.p2_post_b(F.silu(self.p2_post_a(torch.cat([zn, delta_l], dim=-1)))) + 0.25 * ctrl[3])
        z_l_new = z_l + self.fast_res_scale * (g_post * delta_l)
        z_l_new = self._cayley_peer_couple(z_l_new, ctrl[0].view(1, 1))
        return torch.cat([z_h, z_l_new], dim=1)

    def init_anchor(self, k_ev: torch.Tensor, v_ev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = k_ev.shape
        q = self.slot_queries.expand(B, -1, -1)
        qh = self.ca_q(self.norm_ca(q)).view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        sc = sc.masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(k_ev.dtype)
        ctx = torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        return q + self.ca_o(ctx)

    def forward_outer_step(
        self,
        slots: torch.Tensor,
        anchor: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        outer_idx: int,
        prev_slots: torch.Tensor | None = None,
        carry_m_l: torch.Tensor | None = None,
        prev_bus: torch.Tensor | None = None,
        prev_doubt: torch.Tensor | None = None,
        k_inner: int = K_INNER_V130,
    ) -> dict[str, torch.Tensor]:
        B, L, D = k_ev.shape
        oi = min(max(int(outer_idx), 0), 2)
        ctrl = torch.tanh(self.relax.dyn_ctrl[oi])
        s_in = slots
        s = slots

        # 1. Outer-Step History Delta (3-role!) + Doubt Handoff (strictly AI-safe & ONNX-clean!)
        if outer_idx > 0 and prev_slots is not None:
            d_slow = (s[:, : self.n_slow, :] - prev_slots[:, : self.n_slow, :]).mean(dim=1, keepdim=True)
            d_sem = (s[:, self.n_slow : self.n_slow + self.n_sem, :] - prev_slots[:, self.n_slow : self.n_slow + self.n_sem, :]).mean(dim=1, keepdim=True)
            d_span = (s[:, self.n_slow + self.n_sem :, :] - prev_slots[:, self.n_slow + self.n_sem :, :]).mean(dim=1, keepdim=True)
            g_r = 0.10 * torch.tanh(self.role_hist_g)
            s = torch.cat([
                s[:, : self.n_slow, :] + g_r[0] * d_slow,
                s[:, self.n_slow : self.n_slow + self.n_sem, :] + g_r[1] * d_sem,
                s[:, self.n_slow + self.n_sem :, :] + g_r[2] * d_span,
            ], dim=1)

        if outer_idx > 0 and prev_bus is not None and prev_doubt is not None:
            h_in = 0.25 * self.handoff_proj(torch.cat([prev_bus, prev_doubt], dim=-1))
            s = torch.cat([s[:, :-1, :], s[:, -1:, :] + h_in.unsqueeze(1)], dim=1)

        # Surprise signal from slow-slot bus drift
        zh_now_bus = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        if prev_bus is not None:
            cos_sim = F.cosine_similarity(zh_now_bus, prev_bus, dim=-1, eps=1e-6).unsqueeze(-1)
            surprise = torch.sigmoid(self.c2_surprise_mlp(torch.cat([zh_now_bus - prev_bus, cos_sim], dim=-1)).squeeze(-1))
        else:
            surprise = torch.full((B,), 0.5, device=s.device, dtype=s.dtype)

        # 2. Active Text K/V Reread (with per-outer-step `rank=16` LoRA on `ca_q`!) + Fine-Scale Haar Reread
        sn_ca = self.norm_ca(s)
        q_base = self.ca_q(sn_ca) + torch.matmul(torch.matmul(sn_ca, self.ca_q_lora_a[oi]), self.ca_q_lora_b[oi])
        qh = q_base.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        sc = sc.masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(s.dtype)
        s = s + torch.sigmoid(self.reread_gate) * self.ca_o(torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D))

        # Fine-scale Haar wavelet reread (`64-d`, 100% pad-exact via `haar_mask = m_even`!)
        q_haar = self.haar_q(sn_ca)
        sc_h = torch.matmul(q_haar, k_haar.transpose(-1, -2)) / 8.0
        sc_h = sc_h.masked_fill(haar_mask[:, None, :].float() < 0.5, -1e4)
        at_h = torch.softmax(sc_h.float(), dim=-1).to(s.dtype)
        g_h = torch.sigmoid(self.haar_gate + 0.35 * ctrl[1])
        s = s + g_h * self.haar_o(torch.matmul(at_h, v_haar))

        # 3. Surprise-Gated EMA Compressed Fast Memory Injection
        z_l_start = s[:, self.n_slow :, :]
        if carry_m_l is not None:
            c2_gain = (0.35 * torch.tanh(self.c2_carry_scale) * (1.0 - 0.5 * surprise) * (1.0 + 0.25 * ctrl[4])).view(-1, 1, 1)
            s = torch.cat([s[:, : self.n_slow, :], z_l_start + c2_gain * self.bus_up(carry_m_l)], dim=1)

        # 4. TRUE NESTED INNER LOOP (`k_inner = 2` fast-inner steps with Cayley Symplectic Exchange!)
        for ki in range(int(k_inner)):
            s = self._fast_inner_step(s, ctrl, outer_idx=oi, phase_idx=min(ki, 1))

        # 5. Update Compressed Fast Memory EMA (`new_carry_m_l`)
        delta_fast_bus = torch.tanh(self.bus_down(s[:, self.n_slow :, :] - z_l_start))
        beta_c = 0.85 * torch.sigmoid(self.c2_beta_logit).view(1, 1, -1)
        new_carry_m_l = (1.0 - beta_c) * delta_fast_bus if carry_m_l is None else (beta_c * carry_m_l + (1.0 - beta_c) * delta_fast_bus)

        # 6. All-16-Slot Self-Attention (with per-outer-step `rank=16` LoRA on `sa_o`!) + Slow-Outer Shared FFN
        sq, sk, sv = self.sa_qkv(self.norm_sa(s)).chunk(3, dim=-1)
        sq = sq.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sk = sk.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sv = sv.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sa = torch.softmax((torch.matmul(sq, sk.transpose(-2, -1)) * (self.head_dim ** -0.5)).float(), dim=-1).to(s.dtype)
        sa_ctx = torch.matmul(sa, sv).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        sa_out = self.sa_o(sa_ctx) + torch.matmul(torch.matmul(sa_ctx, self.sa_o_lora_a[oi]), self.sa_o_lora_b[oi])
        s = s + sa_out
        s = s + self._shared_ffn(self.norm_ff(s), outer_idx=oi, phase_idx=2)

        # 7. L0 Surface Penetration + Analytic 6-Valley EBT on 64-d Slow Bus
        l0_pen = self.l0_pen_b(F.silu(self.l0_pen_a(h0_pool))).unsqueeze(1)
        s = torch.cat([s[:, : self.n_slow, :] + torch.sigmoid(self.l0_pen_gate).view(1, 1, -1) * l0_pen, s[:, self.n_slow :, :]], dim=1)

        bus_pre = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        ebt_r = self.ebt.forward_energy_and_grad(bus_pre)
        ebt_delta = self.bus_up(ebt_r["u_next"] - bus_pre).unsqueeze(1)
        ebt_g = 0.35 * torch.tanh(self.ebt_back_scale) * (1.0 + 0.25 * ctrl[5])
        s_prop = torch.cat([s[:, : self.n_slow, :] + ebt_g * ebt_delta, s[:, self.n_slow :, :]], dim=1)

        # 8. Manifold Anchor Blend + Step-Indexed Spectral Damping
        a_mix = 0.35 * torch.sigmoid(self.anchor_alpha)
        s_prop = (1.0 - a_mix) * s_prop + a_mix * anchor
        if outer_idx == 0:
            s_out = s_prop
        else:
            d_i = (0.55 + 0.45 * torch.sigmoid(self.damp_gamma)) / math.sqrt(float(outer_idx))
            s_out = s_in + d_i * (s_prop - s_in)

        return {
            "slots": s_out,
            "carry_m_l": new_carry_m_l,
            "bus": bus_pre,
            "surprise": surprise,
            "ebt_energies": ebt_r["energies"],
            "ebt_margin": ebt_r["ai_margin"],
            "ebt_promo_margin": ebt_r["promo_margin"],
            "ebt_grad_norm": ebt_r["grad_norm"],
        }


# ---------------------------------------------------------------------------
# 5. Complete Unified Model: `MeguvizV130XtremeApexModel`
# ---------------------------------------------------------------------------

class MeguvizV130XtremeApexModel(nn.Module):
    """Unified <=5.00 MiB Full Nested-Loop XtremeApex Model (Clean W8 R0 + Clean W4/W8 H=384 Nested-Loop)."""

    def __init__(
        self,
        vocab_size: int = 12288,
        max_len: int = MAX_LEN_V130,
        style_dim: int = STYLE_FEAT_DIM_V11,
        style_era_dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.engram_buckets = ENGRAM_BUCKETS_V130
        self.trigram_buckets = TRIGRAM_BUCKETS_V130
        self.style_era_dropout_p = float(style_era_dropout_p)

        # Stage 0: Full-Scale W8 Char 1/2/3-gram Nano Engine (`~1.0 MiB`!)
        self.rung0 = ApexCharNanoRung0V130()

        # Stage 1: Full `H=384` Factorized BPE + Bigram & Trigram Engram + Pad-Exact Prelude + Haar Wavelet K/V
        self.word_emb_a = QuantEmbeddingV130(vocab_size, 48, bits=4, padding_idx=0)
        self.word_emb_p = QuantLinearV130(48, H_V130, bias=False, bits=8)
        self.engram_emb = QuantEmbeddingV130(ENGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.engram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.trigram_emb = QuantEmbeddingV130(TRIGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.trigram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, H_V130) * 0.02)
        self.emb_norm = nn.LayerNorm(H_V130)

        self.prelude_l0 = PadExactConvBlockV130(H_V130, FFN_V130, NUM_HEADS_V130, bits=4)
        self.prelude_l1 = PadExactConvBlockV130(H_V130, FFN_V130, NUM_HEADS_V130, bits=4)
        self.k_ev_proj = QuantLinearV130(H_V130, H_V130, bits=4)
        self.v_ev_proj = QuantLinearV130(H_V130, H_V130, bits=4)
        self.haar_k_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)
        self.haar_v_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)

        # R1 fast prefix heads (strictly separate AI vs Promo!)
        self.r1_style_ai = nn.Sequential(nn.Linear(style_dim - 8, 32), nn.SiLU())
        self.r1_style_pr = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU())
        self.r1_ai_head = nn.Linear(H_V130 + 32 + 32, 1)
        self.r1_promo_head = nn.Linear(H_V130 + 32 + 32, 1)
        self.r1_defer_head = nn.Sequential(nn.Linear(H_V130 + 2, 32), nn.SiLU(), nn.Linear(32, 1))

        # Stage 2: Signature Full `H=384, FFN=896, 16-Slot` Nested-Loop Symplectic Core (Clean W4/W8!)
        self.core = NestedSymplectCoreV130(H_V130, FFN_V130, NUM_HEADS_V130)

        # Strictly Disjoint 256-d Coda: [0:128) shared, [128:192) AI-private, [192:256) Promo-private
        self.coda_shared = QuantLinearV130(H_V130 * 2, SHARED_END_V130, bits=4)
        self.coda_ai_priv = QuantLinearV130(H_V130 + 32 + 32, AI_END_V130 - SHARED_END_V130, bits=4)
        self.coda_pr_priv = QuantLinearV130(H_V130 + 32 + 32, EMB_DIM_V130 - AI_END_V130, bits=4)
        self.emb_norm_sh = nn.LayerNorm(SHARED_END_V130)
        self.emb_norm_ai = nn.LayerNorm(AI_END_V130 - SHARED_END_V130)
        self.emb_norm_pr = nn.LayerNorm(EMB_DIM_V130 - AI_END_V130)

        # Split-Axis Top-K CBM Firewall (`+12` Gov/Promo bias perturb -> exact `0.000000` AI diff!)
        self.cbm_ai_enc = nn.Linear(AI_END_V130, CBM_AI_K_V130)
        self.cbm_pr_enc = nn.Linear(PR_EMB_DIM_V130, CBM_PR_K_V130)
        self.cbm_ai_to_logit = nn.Linear(CBM_AI_K_V130, 1, bias=False)
        self.cbm_pr_to_logit = nn.Linear(CBM_PR_K_V130, 1, bias=False)

        # Bounded SoftBallLogitNorm Readouts
        self.ai_head = SoftBallLogitNormHeadV121(AI_END_V130, init_r_cap=7.2)
        self.promo_head = SoftBallLogitNormHeadV121(PR_EMB_DIM_V130, init_r_cap=7.2)
        self.promo_sub4_head = nn.Linear(PR_EMB_DIM_V130, 4)
        self.ebt_ai_gate = nn.Parameter(torch.tensor(0.20))
        self.ebt_pr_gate = nn.Parameter(torch.tensor(0.15))

        # Trajectory History Readout (`[z_final, z_final - z_t1, 5 AI scalars]`) + Promo Char Bypass!
        self.traj_ai_lin = nn.Linear(AI_END_V130 * 2 + 5, 1)
        self.traj_ai_mlp = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 5),
            nn.Linear(AI_END_V130 * 2 + 5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.traj_ai_lin.weight)
        nn.init.zeros_(self.traj_ai_lin.bias)
        nn.init.zeros_(self.traj_ai_mlp[-1].weight)
        nn.init.zeros_(self.traj_ai_mlp[-1].bias)
        self.traj_ai_cap = 2.5

        self.promo_char_rho_raw = nn.Parameter(torch.tensor(0.0))
        self.traj_pr_mlp = nn.Sequential(
            nn.LayerNorm(PR_EMB_DIM_V130 + 32 + 4),
            nn.Linear(PR_EMB_DIM_V130 + 32 + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.traj_pr_mlp[-1].weight)
        nn.init.zeros_(self.traj_pr_mlp[-1].bias)
        self.traj_pr_cap = 2.5

        # Ouro-Style Conditional `T2 -> T3` Escalation Gate (strictly AI-safe inputs -> ZERO promo leak!)
        self.halt_t2_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 7),
            nn.Linear(AI_END_V130 * 2 + 7, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Native Long-Document `DynaSeam` Attention Aggregator
        self.long_ai_attn = nn.Sequential(nn.Linear(AI_END_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_pr_attn = nn.Sequential(nn.Linear(PR_EMB_DIM_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_ai_head = nn.Linear(AI_END_V130 + 3, 1)
        self.long_pr_head = nn.Linear(PR_EMB_DIM_V130 + 3, 1)

    def _readout_step(
        self,
        slots: torch.Tensor,
        h1_pool: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        ebt_ai_m: torch.Tensor,
        ebt_pr_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        slow_p = slots[:, : NUM_SLOW_V130, :].mean(dim=1)
        sem_p = slots[:, NUM_SLOW_V130 : NUM_SLOW_V130 + NUM_SEM_FAST_V130, :].mean(dim=1)
        span_p = slots[:, NUM_SLOW_V130 + NUM_SEM_FAST_V130 :, :].mean(dim=1)

        z_sh = self.emb_norm_sh(self.coda_shared(torch.cat([slow_p, h1_pool], dim=-1)))
        z_ai_p = self.emb_norm_ai(self.coda_ai_priv(torch.cat([sem_p, st_ai, z_char[:, :32]], dim=-1)))
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, st_pr, z_char[:, 32:]], dim=-1)))

        emb_ai = torch.cat([z_sh, z_ai_p], dim=-1)      # [B, 192] — STRICTLY ZERO PROMO PATHWAY!
        emb_pr = torch.cat([z_sh, z_pr_p], dim=-1)      # [B, 192]
        emb_full = torch.cat([z_sh, z_ai_p, z_pr_p], dim=-1)  # [B, 256]

        c_ai_raw = F.relu(self.cbm_ai_enc(emb_ai))
        v_ai, i_ai = torch.topk(c_ai_raw, k=6, dim=-1)
        c_ai_sp = torch.zeros_like(c_ai_raw).scatter(-1, i_ai, v_ai)

        c_pr_raw = F.relu(self.cbm_pr_enc(emb_pr))
        v_pr, i_pr = torch.topk(c_pr_raw, k=6, dim=-1)
        c_pr_sp = torch.zeros_like(c_pr_raw).scatter(-1, i_pr, v_pr)

        lg_ai = (
            self.ai_head(emb_ai)
            + 0.15 * self.cbm_ai_to_logit(c_ai_sp).squeeze(-1)
            + self.ebt_ai_gate * torch.clamp(ebt_ai_m, -6.0, 6.0)
        )
        lg_pr = (
            self.promo_head(emb_pr)
            + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1)
            + self.ebt_pr_gate * torch.clamp(ebt_pr_m, -6.0, 6.0)
        )
        return {
            "logit_ai": lg_ai,
            "logit_promo": lg_pr,
            "promo_sub4_logit": self.promo_sub4_head(emb_pr),
            "emb_ai": emb_ai,
            "emb_pr": emb_pr,
            "emb": emb_full,
            "cbm_ai": c_ai_sp,
            "cbm_pr": c_pr_sp,
        }

    def _compute_trajectory_final(
        self,
        t1_emb_ai: torch.Tensor,
        t1_lg_ai: torch.Tensor,
        t1_lg_pr: torch.Tensor,
        ro_cur: dict[str, torch.Tensor],
        ebt_ai_m: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        lg_r1_ai: torch.Tensor,
        lg_r1_pr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_delta_ai = ro_cur["emb_ai"] - t1_emb_ai
        ai_scalars = torch.stack([
            logit_ai_r0, lg_r1_ai, t1_lg_ai, ro_cur["logit_ai"], torch.clamp(ebt_ai_m, -6.0, 6.0)
        ], dim=-1)
        traj_ai_in = torch.cat([ro_cur["emb_ai"], z_delta_ai, ai_scalars], dim=-1)
        ai_res_raw = self.traj_ai_lin(traj_ai_in).squeeze(-1) + self.traj_ai_mlp(traj_ai_in).squeeze(-1)
        ai_final = ro_cur["logit_ai"] + self.traj_ai_cap * torch.tanh(ai_res_raw / self.traj_ai_cap)

        rho = torch.sigmoid(self.promo_char_rho_raw)
        pr_base = (1.0 - rho) * ro_cur["logit_promo"] + rho * logit_promo_r0
        pr_scalars = torch.stack([logit_promo_r0, lg_r1_pr, t1_lg_pr, ro_cur["logit_promo"]], dim=-1)
        traj_pr_in = torch.cat([ro_cur["emb_pr"], z_char[:, 32:], pr_scalars], dim=-1)
        pr_res_raw = self.traj_pr_mlp(traj_pr_in).squeeze(-1)
        pr_final = pr_base + self.traj_pr_cap * torch.tanh(pr_res_raw / self.traj_pr_cap)
        return ai_final, pr_final

    def forward_r0(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rung0(c1, c2, c3, mask, feats)

    def forward_t2_stage(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run L0 -> R1 -> L1 -> T1 -> T2 (`k_inner=2` x 2 outer steps) + emit exact cache for optional T3!"""
        B, L = input_ids.shape
        m = attention_mask.float()
        m_u = m.unsqueeze(-1)
        st = style_feats.clone()
        if self.training and self.style_era_dropout_p > 0.0:
            keep = (torch.rand((B, 1), device=st.device) >= self.style_era_dropout_p).to(st.dtype)
            st[:, :40] = st[:, :40] * keep

        pos = self.pos_emb[:, :L, :]
        x = self.emb_norm(
            self.word_emb_p(self.word_emb_a(input_ids))
            + self.engram_p(self.engram_emb(engram_ids))
            + self.trigram_p(self.trigram_emb(trigram_ids))
            + pos
        ) * m_u
        h0 = self.prelude_l0(x, attention_mask)

        # Pad-Exact Fine-Scale Haar Wavelet K/V from L0 (`m_even` guarantees 0.0 padding drift for ANY length!)
        if (not torch.jit.is_tracing()) and (L % 2 == 1):
            h0_p = F.pad(h0, (0, 0, 0, 1))
            m_p = F.pad(m, (0, 1))
        else:
            h0_p, m_p = h0, m
        m_even = m_p[:, 0::2]
        h_diff = (h0_p[:, 0::2, :] - h0_p[:, 1::2, :]) * m_even.unsqueeze(-1)
        k_haar = self.haar_k_proj(h_diff) * m_even.unsqueeze(-1)
        v_haar = self.haar_v_proj(h_diff) * m_even.unsqueeze(-1)
        haar_mask = m_even

        # R1 fast prefix readout
        denom = m_u.sum(dim=1).clamp_min(1.0)
        h0_pool = h0.sum(dim=1) / denom
        st_ai = self.r1_style_ai(st[:, : self.style_dim - 8])
        st_pr = self.r1_style_pr(st)
        lg_r1_ai = self.r1_ai_head(torch.cat([h0_pool, st_ai, z_char[:, :32]], dim=-1)).squeeze(-1)
        lg_r1_pr = self.r1_promo_head(torch.cat([h0_pool, st_pr, z_char[:, 32:]], dim=-1)).squeeze(-1)
        def_r1_lg = self.r1_defer_head(
            torch.cat([h0_pool.detach(), lg_r1_ai.detach().unsqueeze(-1), lg_r1_pr.detach().unsqueeze(-1)], dim=-1)
        ).squeeze(-1)

        # L1 Pad-Exact encoding + read-only K/V text evidence
        h1 = self.prelude_l1(h0, attention_mask)
        h1_pool = h1.sum(dim=1) / denom
        k_ev = self.k_ev_proj(h1) * m_u
        v_ev = self.v_ev_proj(h1) * m_u

        # Outer Step T1 (`outer_idx=0`, `k_inner=2`)
        anchor = self.core.init_anchor(k_ev, v_ev, attention_mask)
        st1 = self.core.forward_outer_step(
            slots=anchor, anchor=anchor, k_ev=k_ev, v_ev=v_ev, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=0, prev_slots=None, carry_m_l=None, prev_bus=None, prev_doubt=None, k_inner=K_INNER_V130,
        )
        ro1 = self._readout_step(st1["slots"], h1_pool, st_ai, st_pr, z_char, st1["ebt_margin"], st1["ebt_promo_margin"])
        c_conf1 = (2.0 * torch.sigmoid(ro1["logit_ai"].detach()) - 1.0).abs()
        doubt1 = torch.stack([c_conf1, st1["ebt_grad_norm"].detach()], dim=-1)

        # Outer Step T2 (`outer_idx=1`, `k_inner=2`)
        st2 = self.core.forward_outer_step(
            slots=st1["slots"], anchor=anchor, k_ev=k_ev, v_ev=v_ev, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=1, prev_slots=anchor, carry_m_l=st1["carry_m_l"], prev_bus=st1["bus"], prev_doubt=doubt1, k_inner=K_INNER_V130,
        )
        ro2 = self._readout_step(st2["slots"], h1_pool, st_ai, st_pr, z_char, st2["ebt_margin"], st2["ebt_promo_margin"])
        ai_t2_final, pr_t2_final = self._compute_trajectory_final(
            ro1["emb_ai"], ro1["logit_ai"], ro1["logit_promo"], ro2, st2["ebt_margin"],
            z_char, logit_ai_r0, logit_promo_r0, lg_r1_ai, lg_r1_pr,
        )

        # Ouro Conditional `T2 -> T3` Escalation Gate (reads strictly AI-safe & detached features!)
        halt_in = torch.cat([
            ro2["emb_ai"].detach(),
            (ro2["emb_ai"] - ro1["emb_ai"]).detach(),
            torch.stack([
                logit_ai_r0.detach(),
                lg_r1_ai.detach(),
                ro1["logit_ai"].detach(),
                ai_t2_final.detach(),
                torch.clamp(st2["ebt_margin"].detach(), -6.0, 6.0),
                st2["ebt_grad_norm"].detach(),
                st2["surprise"].detach(),
            ], dim=-1),
        ], dim=-1)
        esc_lg = self.halt_t2_head(halt_in).squeeze(-1)

        return {
            "logit_ai_t2_final": ai_t2_final,
            "logit_promo_t2_final": pr_t2_final,
            "logit_ai_r1": lg_r1_ai,
            "logit_promo_r1": lg_r1_pr,
            "defer_r1_logit": def_r1_lg,
            "p_defer_r1": torch.sigmoid(def_r1_lg),
            "logit_ai_t1": ro1["logit_ai"],
            "logit_promo_t1": ro1["logit_promo"],
            "logit_ai_t2": ro2["logit_ai"],
            "logit_promo_t2": ro2["logit_promo"],
            "promo_sub4_logit_t2": ro2["promo_sub4_logit"],
            "ebt_energies_t2": st2["ebt_energies"],
            "ebt_margin_t2": st2["ebt_margin"],
            "ebt_promo_margin_t2": st2["ebt_promo_margin"],
            "ebt_grad_norm_t2": st2["ebt_grad_norm"],
            "emb_ai_t1": ro1["emb_ai"],
            "emb_ai_t2": ro2["emb_ai"],
            "emb_pr_t2": ro2["emb_pr"],
            "emb_t2": ro2["emb"],
            "escalate_t3_logit": esc_lg,
            "p_escalate_t3": torch.sigmoid(esc_lg),
            # Exact T2 -> T3 zero-redundancy continuation state:
            "slots_t1": st1["slots"],
            "slots_t2": st2["slots"],
            "anchor": anchor,
            "carry_m_l_t2": st2["carry_m_l"],
            "zh_bus_t2": st2["bus"],
            "k_ev": k_ev,
            "v_ev": v_ev,
            "k_haar": k_haar,
            "v_haar": v_haar,
            "haar_mask": haar_mask,
            "h0_pool": h0_pool,
            "h1_pool": h1_pool,
            "st_ai": st_ai,
            "st_pr": st_pr,
        }

    def forward_t3_step(
        self,
        slots_t2: torch.Tensor,
        slots_t1: torch.Tensor,
        anchor: torch.Tensor,
        carry_m_l_t2: torch.Tensor,
        zh_bus_t2: torch.Tensor,
        ebt_grad_norm_t2: torch.Tensor,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        attention_mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        h1_pool: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        logit_ai_r1: torch.Tensor,
        logit_promo_r1: torch.Tensor,
        logit_ai_t1: torch.Tensor,
        logit_promo_t1: torch.Tensor,
        logit_ai_t2_raw: torch.Tensor,
        emb_ai_t1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Conditional 3rd-Outer-Step Active Error-Correction Graph (reuses cached K/V + Haar + T2 state!)."""
        c_conf2 = (2.0 * torch.sigmoid(logit_ai_t2_raw.detach()) - 1.0).abs()
        doubt2 = torch.stack([c_conf2, ebt_grad_norm_t2.detach()], dim=-1)
        st3 = self.core.forward_outer_step(
            slots=slots_t2, anchor=anchor, k_ev=k_ev, v_ev=v_ev, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=2, prev_slots=slots_t1, carry_m_l=carry_m_l_t2, prev_bus=zh_bus_t2, prev_doubt=doubt2, k_inner=K_INNER_V130,
        )
        ro3 = self._readout_step(st3["slots"], h1_pool, st_ai, st_pr, z_char, st3["ebt_margin"], st3["ebt_promo_margin"])
        ai_t3_final, pr_t3_final = self._compute_trajectory_final(
            emb_ai_t1, logit_ai_t1, logit_promo_t1, ro3, st3["ebt_margin"],
            z_char, logit_ai_r0, logit_promo_r0, logit_ai_r1, logit_promo_r1,
        )
        return {
            "logit_ai_t3_final": ai_t3_final,
            "logit_promo_t3_final": pr_t3_final,
            "logit_ai_t3": ro3["logit_ai"],
            "logit_promo_t3": ro3["logit_promo"],
            "promo_sub4_logit_t3": ro3["promo_sub4_logit"],
            "ebt_energies_t3": st3["ebt_energies"],
            "ebt_margin_t3": st3["ebt_margin"],
            "ebt_promo_margin_t3": st3["ebt_promo_margin"],
            "emb_ai_t3": ro3["emb_ai"],
            "emb_pr_t3": ro3["emb_pr"],
            "emb_t3": ro3["emb"],
        }

    def forward_deep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        outer_steps: int = 2,
    ) -> dict[str, torch.Tensor]:
        t2 = self.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
        )
        out = dict(t2)
        if int(outer_steps) >= 3:
            t3 = self.forward_t3_step(
                slots_t2=t2["slots_t2"], slots_t1=t2["slots_t1"], anchor=t2["anchor"],
                carry_m_l_t2=t2["carry_m_l_t2"], zh_bus_t2=t2["zh_bus_t2"], ebt_grad_norm_t2=t2["ebt_grad_norm_t2"],
                k_ev=t2["k_ev"], v_ev=t2["v_ev"], attention_mask=attention_mask,
                k_haar=t2["k_haar"], v_haar=t2["v_haar"], haar_mask=t2["haar_mask"],
                h0_pool=t2["h0_pool"], h1_pool=t2["h1_pool"], st_ai=t2["st_ai"], st_pr=t2["st_pr"],
                z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
                logit_ai_r1=t2["logit_ai_r1"], logit_promo_r1=t2["logit_promo_r1"],
                logit_ai_t1=t2["logit_ai_t1"], logit_promo_t1=t2["logit_promo_t1"],
                logit_ai_t2_raw=t2["logit_ai_t2"], emb_ai_t1=t2["emb_ai_t1"],
            )
            out.update(t3)
            out["logit_ai"] = t3["logit_ai_t3_final"]
            out["logit_promo"] = t3["logit_promo_t3_final"]
            out["promo_sub4_logit"] = t3["promo_sub4_logit_t3"]
            out["ebt_energies"] = t3["ebt_energies_t3"]
            out["ebt_margin"] = t3["ebt_margin_t3"]
            out["ebt_promo_margin"] = t3["ebt_promo_margin_t3"]
            out["emb_ai"] = t3["emb_ai_t3"]
            out["emb_pr"] = t3["emb_pr_t3"]
            out["emb"] = t3["emb_t3"]
        else:
            out["logit_ai"] = t2["logit_ai_t2_final"]
            out["logit_promo"] = t2["logit_promo_t2_final"]
            out["promo_sub4_logit"] = t2["promo_sub4_logit_t2"]
            out["ebt_energies"] = t2["ebt_energies_t2"]
            out["ebt_margin"] = t2["ebt_margin_t2"]
            out["ebt_promo_margin"] = t2["ebt_promo_margin_t2"]
            out["emb_ai"] = t2["emb_ai_t2"]
            out["emb_pr"] = t2["emb_pr_t2"]
            out["emb"] = t2["emb_t2"]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        r0_c1: torch.Tensor,
        r0_c2: torch.Tensor,
        r0_c3: torch.Tensor,
        r0_mask: torch.Tensor,
        r0_feats: torch.Tensor,
        outer_steps: int = 2,
    ) -> dict[str, torch.Tensor]:
        r0 = self.forward_r0(r0_c1, r0_c2, r0_c3, r0_mask, r0_feats)
        deep = self.forward_deep(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=r0["z_char"],
            logit_ai_r0=r0["logit_ai_r0"],
            logit_promo_r0=r0["logit_promo_r0"],
            outer_steps=outer_steps,
        )
        return {**r0, **deep}

    def aggregate_long_chunks(
        self,
        chunk_emb_ai: torch.Tensor,
        chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor,
        chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quote-aware & mask-safe `DynaSeam` document aggregation over `[B, M, ...]` chunks."""
        vm = chunk_valid.float()
        q_frac = chunk_meta[..., 3].clamp(0.0, 1.0)
        ai_in = torch.cat([chunk_emb_ai, chunk_meta, chunk_lg_ai.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        a_w = (self.long_ai_attn(ai_in).squeeze(-1) - 2.0 * q_frac).masked_fill(vm < 0.5, -1e4)
        a_w = torch.softmax(a_w.float(), dim=-1).to(chunk_emb_ai.dtype)
        ai_ctx = (chunk_emb_ai * a_w.unsqueeze(-1)).sum(dim=1)
        ai_max = (chunk_lg_ai * (1.0 - 0.5 * q_frac)).masked_fill(vm < 0.5, -1e4).max(dim=1).values
        ai_mean = (chunk_lg_ai * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        ai_wmean = (chunk_lg_ai * a_w).sum(dim=1)
        doc_ai = ai_wmean + 0.5 * torch.tanh(self.long_ai_head(torch.cat([ai_ctx, ai_max.unsqueeze(-1), ai_mean.unsqueeze(-1), ai_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1))

        pr_in = torch.cat([chunk_emb_pr, chunk_meta, chunk_lg_pr.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        p_w = self.long_pr_attn(pr_in).squeeze(-1).masked_fill(vm < 0.5, -1e4)
        p_w = torch.softmax(p_w.float(), dim=-1).to(chunk_emb_pr.dtype)
        pr_ctx = (chunk_emb_pr * p_w.unsqueeze(-1)).sum(dim=1)
        pr_max = chunk_lg_pr.masked_fill(vm < 0.5, -1e4).max(dim=1).values
        pr_mean = (chunk_lg_pr * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        pr_wmean = (chunk_lg_pr * p_w).sum(dim=1)
        doc_pr = pr_wmean + 0.5 * torch.tanh(self.long_pr_head(torch.cat([pr_ctx, pr_max.unsqueeze(-1), pr_mean.unsqueeze(-1), pr_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1))
        return doc_ai, doc_pr


# ---------------------------------------------------------------------------
# 6. Exact Clean W4/W8 `MGVZ8` Binary Packer & Loader (`<= 5.00 MiB`, ZERO 1.58-bit!)
# ---------------------------------------------------------------------------

def _pack_w8_tensor(arr: np.ndarray) -> tuple[bytes, bytes]:
    arr_f = arr.astype(np.float32)
    if arr_f.ndim < 2:
        sc = np.array([max(1e-4, float(np.abs(arr_f).max()) / 127.0)], dtype=np.float16)
        q = np.clip(np.round(arr_f / sc.astype(np.float32)), -127, 127).astype(np.int8)
        return sc.tobytes(), q.tobytes()
    rows = arr_f.reshape(-1, arr_f.shape[-1])
    amax = np.abs(rows).max(axis=-1, keepdims=True)
    sc = np.maximum(amax / 127.0, 1e-4).astype(np.float16)
    q = np.clip(np.round(rows / sc.astype(np.float32)), -127, 127).astype(np.int8)
    return sc.tobytes(), q.tobytes()


def _unpack_w8_tensor(s_buf: bytes, q_buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    q = np.frombuffer(q_buf, dtype=np.int8).astype(np.float32)
    sc = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32)
    if len(shape) < 2:
        return (q * sc[0]).reshape(shape)
    return (q.reshape(-1, shape[-1]) * sc.reshape(-1, 1)).reshape(shape)


def _pack_w4_tensor(arr: np.ndarray, gs: int = 64) -> tuple[bytes, bytes]:
    flat = arr.astype(np.float32).reshape(-1)
    n = flat.size
    pad = (gs - (n % gs)) % gs
    if pad:
        flat = np.pad(flat, (0, pad))
    g = flat.reshape(-1, gs)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    sc = np.maximum(amax / 7.0, 1e-4).astype(np.float16)
    q = np.clip(np.round(g / sc.astype(np.float32)), -7, 7).astype(np.int8) + 8
    q_flat = q.reshape(-1)[:n].astype(np.uint8)
    if q_flat.size % 2 == 1:
        q_flat = np.pad(q_flat, (0, 1))
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).astype(np.uint8)
    return sc.tobytes(), packed.tobytes()


def _unpack_w4_tensor(s_buf: bytes, q_buf: bytes, shape: tuple[int, ...], gs: int = 64) -> np.ndarray:
    n = int(np.prod(shape))
    packed = np.frombuffer(q_buf, dtype=np.uint8)
    lo = (packed & 0x0F).astype(np.float32) - 8.0
    hi = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0
    vals = np.empty(packed.size * 2, dtype=np.float32)
    vals[0::2] = lo
    vals[1::2] = hi
    vals = vals[:n]
    pad = (gs - (n % gs)) % gs
    if pad:
        vals = np.pad(vals, (0, pad))
    sc = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32).reshape(-1, 1)
    return (vals.reshape(-1, gs) * sc).reshape(-1)[:n].reshape(shape)


def _quant_spec_map_v130(model: MeguvizV130XtremeApexModel) -> dict[str, tuple[str, int]]:
    spec: dict[str, tuple[str, int]] = {
        "pos_emb": ("w8", 0),
        "core.ca_q_lora_a": ("w8", 0),
        "core.ca_q_lora_b": ("w8", 0),
        "core.sa_o_lora_a": ("w8", 0),
        "core.sa_o_lora_b": ("w8", 0),
        "core.slot_queries": ("w8", 0),
    }
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinearV130):
            spec[f"{name}.weight"] = ("w8" if mod.target_bits == 8 else "w4", mod.group_size)
        elif isinstance(mod, QuantEmbeddingV130):
            spec[f"{name}.weight"] = ("w8" if mod.target_bits == 8 else "w4", mod.group_size)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 1024:
            spec[f"{name}.weight"] = ("w8", 0)
    return spec


def pack_bundle_v8_v130(model: MeguvizV130XtremeApexModel, out_path: Path) -> dict[str, Any]:
    """Pack `MeguvizV130XtremeApexModel` into clean W8/W4/FP16 `MGVZ8` format (ZERO 1.58-bit!)."""
    spec_map = _quant_spec_map_v130(model)
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    breakdown: dict[str, int] = {"r0_w8": 0, "deep_w4": 0, "deep_w8": 0, "fp16": 0}

    for k, arr in sd.items():
        q_mode, gs = spec_map.get(k, ("fp16", 0))
        if arr.size < 64:
            q_mode, gs = "fp16", 0
        if q_mode == "w8":
            sb, qb = _pack_w8_tensor(arr)
            bkey = "r0_w8" if k.startswith("rung0.") else "deep_w8"
        elif q_mode == "w4":
            sb, qb = _pack_w4_tensor(arr, gs=gs)
            bkey = "deep_w4"
        else:
            sb, qb = b"", arr.astype(np.float16).tobytes()
            bkey = "fp16"
        entries.append({"name": k, "q": q_mode, "g": gs, "shape": list(arr.shape), "s_bytes": len(sb), "q_bytes": len(qb)})
        blobs.append(sb)
        blobs.append(qb)
        breakdown[bkey] += len(sb) + len(qb)

    hdr_bytes = json.dumps({"format": "MGVZ8_NESTED_V130", "tensors": entries}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MGVZ8_MAGIC)
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


def load_bundle_v8_v130_into_model(model: MeguvizV130XtremeApexModel, mgvz_path: Path) -> None:
    data = mgvz_path.read_bytes()
    assert data[:5] == b"MGVZ8", "Invalid MGVZ8 magic!"
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
# 7. 4-Stage Modular FP32-Exact ONNX Export (`r0`, `t2_nested`, `t3_correct`, `dynaseam`)
# ---------------------------------------------------------------------------

class _V130R0Export(nn.Module):
    def __init__(self, m: MeguvizV130XtremeApexModel) -> None:
        super().__init__()
        self.r0 = m.rung0

    def forward(self, c1, c2, c3, mask, feats):
        o = self.r0(c1, c2, c3, mask, feats)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_r0"], o["z_char"]


_T2_OUT_NAMES = [
    "logit_ai_t2_final", "logit_promo_t2_final", "logit_ai_r1", "logit_promo_r1",
    "logit_ai_t1", "logit_promo_t1", "logit_ai_t2", "ebt_margin_t2", "ebt_grad_norm_t2",
    "emb_ai_t1", "emb_ai_t2", "emb_pr_t2", "p_escalate_t3",
    "slots_t1", "slots_t2", "anchor", "carry_m_l_t2", "zh_bus_t2",
    "k_ev", "v_ev", "k_haar", "v_haar", "haar_mask", "h0_pool", "h1_pool", "st_ai", "st_pr",
]


class _V130T2Export(nn.Module):
    def __init__(self, m: MeguvizV130XtremeApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, z_char, logit_ai_r0, logit_promo_r0):
        o = self.m.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
        )
        return tuple(o[k] for k in _T2_OUT_NAMES)


_T3_IN_NAMES = [
    "slots_t2", "slots_t1", "anchor", "carry_m_l_t2", "zh_bus_t2", "ebt_grad_norm_t2",
    "k_ev", "v_ev", "attention_mask", "k_haar", "v_haar", "haar_mask",
    "h0_pool", "h1_pool", "st_ai", "st_pr", "z_char",
    "logit_ai_r0", "logit_promo_r0", "logit_ai_r1", "logit_promo_r1",
    "logit_ai_t1", "logit_promo_t1", "logit_ai_t2", "emb_ai_t1",
]
_T3_OUT_NAMES = ["logit_ai_t3_final", "logit_promo_t3_final", "logit_ai_t3", "ebt_margin_t3", "emb_ai_t3", "emb_pr_t3"]


class _V130T3Export(nn.Module):
    def __init__(self, m: MeguvizV130XtremeApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(
        self, slots_t2, slots_t1, anchor, carry_m_l_t2, zh_bus_t2, ebt_grad_norm_t2,
        k_ev, v_ev, attention_mask, k_haar, v_haar, haar_mask,
        h0_pool, h1_pool, st_ai, st_pr, z_char,
        logit_ai_r0, logit_promo_r0, logit_ai_r1, logit_promo_r1,
        logit_ai_t1, logit_promo_t1, logit_ai_t2, emb_ai_t1,
    ):
        o = self.m.forward_t3_step(
            slots_t2, slots_t1, anchor, carry_m_l_t2, zh_bus_t2, ebt_grad_norm_t2,
            k_ev, v_ev, attention_mask, k_haar, v_haar, haar_mask,
            h0_pool, h1_pool, st_ai, st_pr, z_char,
            logit_ai_r0, logit_promo_r0, logit_ai_r1, logit_promo_r1,
            logit_ai_t1, logit_promo_t1, logit_ai_t2, emb_ai_t1,
        )
        return tuple(o[k] for k in _T3_OUT_NAMES)


class _V130LongAggExport(nn.Module):
    def __init__(self, m: MeguvizV130XtremeApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, emb_ai, emb_pr, lg_ai, lg_pr, meta, valid):
        return self.m.aggregate_long_chunks(emb_ai, emb_pr, lg_ai, lg_pr, meta, valid)


def export_v130_onnx_and_bundle_v8(
    model: MeguvizV130XtremeApexModel,
    tokenizer_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    version: str,
) -> dict[str, Any]:
    """Pack `MGVZ8`, reload exact dequantized weights, export 4 modular FP32-exact ONNX graphs, verify parity!"""
    out_dir.mkdir(parents=True, exist_ok=True)
    mgvz_p = out_dir / "bundle_v8_xtreme_apex_v130.mgvz"
    set_v130_qat_mode(model, False)
    mgvz_info = pack_bundle_v8_v130(model, mgvz_p)

    m_cpu = MeguvizV130XtremeApexModel(
        vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim, style_era_dropout_p=0.0
    ).cpu().eval()
    load_bundle_v8_v130_into_model(m_cpu, mgvz_p)

    # 1. R0 ApexCharNano
    c1 = torch.randint(1, R0_B1_V130, (2, 24), dtype=torch.long)
    c2 = torch.randint(1, R0_B2_V130, (2, 24), dtype=torch.long)
    c3 = torch.randint(1, R0_B3_V130, (2, 24), dtype=torch.long)
    cm = torch.ones((2, 24), dtype=torch.float32)
    cf = torch.zeros((2, R0_FEAT_DIM_V130), dtype=torch.float32)

    r0_p = out_dir / "v130_r0_apexchar_fp32.onnx"
    torch.onnx.export(
        _V130R0Export(m_cpu), (c1, c2, c3, cm, cf), str(r0_p),
        input_names=["c1", "c2", "c3", "mask", "feats"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_r0", "z_char"],
        dynamic_axes={k: {0: "batch", 1: "clen"} for k in ("c1", "c2", "c3", "mask")} | {"feats": {0: "batch"}, "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"}, "p_defer_r0": {0: "batch"}, "z_char": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # 2. T2 Nested-Loop Stage (`L0 -> R1 -> L1 -> T1 -> T2` + cached state for conditional T3!)
    ids = torch.randint(1, model.vocab_size, (2, 32), dtype=torch.long)
    msk = torch.ones((2, 32), dtype=torch.long)
    st = torch.zeros((2, model.style_dim), dtype=torch.float32)
    eg = torch.randint(1, ENGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    tg = torch.randint(1, TRIGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    zc = torch.zeros((2, R0_ZCHAR_DIM_V130), dtype=torch.float32)
    lr0_a = torch.zeros((2,), dtype=torch.float32)
    lr0_p = torch.zeros((2,), dtype=torch.float32)

    t2_p = out_dir / "v130_t2_nested_fp32.onnx"
    t2_dyn = {
        "input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
        "engram_ids": {0: "batch", 1: "seq"}, "trigram_ids": {0: "batch", 1: "seq"},
        "style_feats": {0: "batch"}, "z_char": {0: "batch"}, "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"},
    }
    for nm in _T2_OUT_NAMES:
        if nm in ("k_ev", "v_ev"):
            t2_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t2_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t2_dyn[nm] = {0: "batch"}

    torch.onnx.export(
        _V130T2Export(m_cpu), (ids, msk, st, eg, tg, zc, lr0_a, lr0_p), str(t2_p),
        input_names=["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "z_char", "logit_ai_r0", "logit_promo_r0"],
        output_names=_T2_OUT_NAMES,
        dynamic_axes=t2_dyn,
        opset_version=17, dynamo=False,
    )

    # 3. Conditional T3 Active Error-Correction Graph (reuses exact T2 cached state!)
    with torch.no_grad():
        t2_dict = m_cpu.forward_t2_stage(ids, msk, st, eg, tg, zc, lr0_a, lr0_p)
    t3_dummy = (
        t2_dict["slots_t2"], t2_dict["slots_t1"], t2_dict["anchor"], t2_dict["carry_m_l_t2"], t2_dict["zh_bus_t2"], t2_dict["ebt_grad_norm_t2"],
        t2_dict["k_ev"], t2_dict["v_ev"], msk, t2_dict["k_haar"], t2_dict["v_haar"], t2_dict["haar_mask"],
        t2_dict["h0_pool"], t2_dict["h1_pool"], t2_dict["st_ai"], t2_dict["st_pr"], zc,
        lr0_a, lr0_p, t2_dict["logit_ai_r1"], t2_dict["logit_promo_r1"],
        t2_dict["logit_ai_t1"], t2_dict["logit_promo_t1"], t2_dict["logit_ai_t2"], t2_dict["emb_ai_t1"],
    )
    t3_dyn = {}
    for nm in _T3_IN_NAMES + _T3_OUT_NAMES:
        if nm in ("k_ev", "v_ev", "attention_mask"):
            t3_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t3_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t3_dyn[nm] = {0: "batch"}

    t3_p = out_dir / "v130_t3_correct_fp32.onnx"
    torch.onnx.export(
        _V130T3Export(m_cpu), t3_dummy, str(t3_p),
        input_names=_T3_IN_NAMES,
        output_names=_T3_OUT_NAMES,
        dynamic_axes=t3_dyn,
        opset_version=17, dynamo=False,
    )

    # 4. DynaSeam Long-Doc Aggregator
    long_p = out_dir / "v130_dynaseam_fp32.onnx"
    ea = torch.randn(2, 4, AI_END_V130)
    ep = torch.randn(2, 4, PR_EMB_DIM_V130)
    la = torch.randn(2, 4)
    lp = torch.randn(2, 4)
    mt = torch.zeros(2, 4, CHUNK_META_DIM_V11)
    vm = torch.ones(2, 4)
    torch.onnx.export(
        _V130LongAggExport(m_cpu), (ea, ep, la, lp, mt, vm), str(long_p),
        input_names=["emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid"],
        output_names=["doc_ai", "doc_promo"],
        dynamic_axes={
            "emb_ai": {0: "batch", 1: "chunks"}, "emb_pr": {0: "batch", 1: "chunks"},
            "lg_ai": {0: "batch", 1: "chunks"}, "lg_pr": {0: "batch", 1: "chunks"},
            "meta": {0: "batch", 1: "chunks"}, "valid": {0: "batch", 1: "chunks"},
            "doc_ai": {0: "batch"}, "doc_promo": {0: "batch"},
        },
        opset_version=17, dynamo=False,
    )

    # Exact Parity Verification: R0, T2, Modular T2->T3 vs PT `outer_steps=3`, and Long-Doc!
    s0 = ort.InferenceSession(str(r0_p), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(t2_p), providers=["CPUExecutionProvider"])
    s3 = ort.InferenceSession(str(t3_p), providers=["CPUExecutionProvider"])
    sl = ort.InferenceSession(str(long_p), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_r0 = _V130R0Export(m_cpu)(c1, c2, c3, cm, cf)
        or_r0 = s0.run(None, {"c1": c1.numpy(), "c2": c2.numpy(), "c3": c3.numpy(), "mask": cm.numpy(), "feats": cf.numpy()})
        r0_diff = float(max(np.abs(pt_r0[i].numpy() - or_r0[i]).max() for i in range(4)))

        pt_t2 = _V130T2Export(m_cpu)(ids, msk, st, eg, tg, pt_r0[3], pt_r0[0], pt_r0[1])
        or_t2 = s2.run(None, {
            "input_ids": ids.numpy(), "attention_mask": msk.numpy(), "style_feats": st.numpy(),
            "engram_ids": eg.numpy(), "trigram_ids": tg.numpy(), "z_char": or_r0[3],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1],
        })
        t2_diff = float(max(np.abs(pt_t2[i].numpy() - or_t2[i]).max() for i in range(len(pt_t2))))

        # Chain ORT T2 -> ORT T3 and compare against PyTorch `forward_deep(..., outer_steps=3)`!
        or_t2_map = dict(zip(_T2_OUT_NAMES, or_t2))
        t3_feed = {
            "slots_t2": or_t2_map["slots_t2"], "slots_t1": or_t2_map["slots_t1"], "anchor": or_t2_map["anchor"],
            "carry_m_l_t2": or_t2_map["carry_m_l_t2"], "zh_bus_t2": or_t2_map["zh_bus_t2"],
            "ebt_grad_norm_t2": or_t2_map["ebt_grad_norm_t2"], "k_ev": or_t2_map["k_ev"], "v_ev": or_t2_map["v_ev"],
            "attention_mask": msk.numpy(), "k_haar": or_t2_map["k_haar"], "v_haar": or_t2_map["v_haar"],
            "haar_mask": or_t2_map["haar_mask"], "h0_pool": or_t2_map["h0_pool"], "h1_pool": or_t2_map["h1_pool"],
            "st_ai": or_t2_map["st_ai"], "st_pr": or_t2_map["st_pr"], "z_char": or_r0[3],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1],
            "logit_ai_r1": or_t2_map["logit_ai_r1"], "logit_promo_r1": or_t2_map["logit_promo_r1"],
            "logit_ai_t1": or_t2_map["logit_ai_t1"], "logit_promo_t1": or_t2_map["logit_promo_t1"],
            "logit_ai_t2": or_t2_map["logit_ai_t2"], "emb_ai_t1": or_t2_map["emb_ai_t1"],
        }
        or_t3 = s3.run(None, t3_feed)
        pt_full3 = m_cpu.forward_deep(ids, msk, st, eg, tg, pt_r0[3], pt_r0[0], pt_r0[1], outer_steps=3)
        pt_t3_ref = [pt_full3[k].numpy() for k in _T3_OUT_NAMES]
        t3_diff = float(max(np.abs(pt_t3_ref[i] - or_t3[i]).max() for i in range(len(_T3_OUT_NAMES))))

        pt_l = _V130LongAggExport(m_cpu)(ea, ep, la, lp, mt, vm)
        or_l = sl.run(None, {"emb_ai": ea.numpy(), "emb_pr": ep.numpy(), "lg_ai": la.numpy(), "lg_pr": lp.numpy(), "meta": mt.numpy(), "valid": vm.numpy()})
        long_diff = float(max(np.abs(pt_l[i].numpy() - or_l[i]).max() for i in range(2)))

    tok_out = out_dir / "bili_picky_bpe_12k.json"
    if tokenizer_path.resolve() != tok_out.resolve():
        tok_out.write_bytes(tokenizer_path.read_bytes())

    manifest = dict(manifest)
    manifest["format"] = "MGVZ8_NESTED_XTREME_APEX_V130"
    manifest["mgvz8_info"] = mgvz_info
    mf_p = out_dir / "manifest.json"
    mf_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle_zip = out_dir / f"meguvizcac-{version}-bundle-v8-xtreme-apex.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(mgvz_p, mgvz_p.name)
        zf.write(tok_out, tok_out.name)
        zf.write(mf_p, mf_p.name)
    bundle_bytes = bundle_zip.stat().st_size

    onnx_zip = out_dir / f"meguvizcac-{version}-4stage-onnx-fp32exact.zip"
    with zipfile.ZipFile(onnx_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in (r0_p, t2_p, t3_p, long_p, tok_out, mf_p):
            zf.write(p, p.name)

    return {
        "bundle_zip": str(bundle_zip),
        "bundle_bytes": bundle_bytes,
        "bundle_mib": round(bundle_bytes / (1024.0 * 1024.0), 4),
        "bundle_le_5p00_mib": bool(bundle_bytes <= int(BUNDLE_MAX_MIB_V130 * 1024 * 1024)),
        "onnx_zip_mb": round(onnx_zip.stat().st_size / (1024.0 * 1024.0), 4),
        "r0_fp32_max_diff": round(r0_diff, 8),
        "t2_fp32_max_diff": round(t2_diff, 8),
        "t3_modular_fp32_max_diff": round(t3_diff, 8),
        "long_fp32_max_diff": round(long_diff, 8),
        "onnx_exact_pass": bool(max(r0_diff, t2_diff, t3_diff, long_diff) < 1e-4),
        "mgvz8_info": mgvz_info,
    }


# ---------------------------------------------------------------------------
# 8. Production 4-Stage CPU ONNX Cascade Engine (`MeguvizV130CascadeEngine`)
#    R0 Early Exit -> T2 Nested Exit -> Conditional T3 Correction -> T4 Long-Doc!
# ---------------------------------------------------------------------------

class MeguvizV130CascadeEngine:
    """Zero-PII 4-Stage Modular ONNX Cascade Engine with Conditional Ouro `T2 -> T3` & Native Long-Doc!"""

    def __init__(self, release_dir: str | Path, threads: int = 4) -> None:
        self.release_dir = Path(release_dir)
        self.manifest = json.loads((self.release_dir / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = BiliPickyBPETokenizer(self.release_dir / "bili_picky_bpe_12k.json")
        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V130))
        self.head_len = int(self.manifest.get("head_len", HEAD_LEN_V130))
        self.long_char_trigger = int(self.manifest.get("long_char_trigger", 340))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_r0 = ort.InferenceSession(str(self.release_dir / "v130_r0_apexchar_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t2 = ort.InferenceSession(str(self.release_dir / "v130_t2_nested_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t3 = ort.InferenceSession(str(self.release_dir / "v130_t3_correct_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_long = ort.InferenceSession(str(self.release_dir / "v130_dynaseam_fp32.onnx"), opts, providers=["CPUExecutionProvider"])

        self.r0_policy = self.manifest.get("rung0_ltt_policy") or {}
        self.t3_policy = self.manifest.get("t3_escalation_policy") or {"enabled": True, "tau_escalate": 0.45, "uncertain_lo": 0.12, "uncertain_hi": 0.88}
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.temp_ai = float(self.manifest.get("temp_ai", 1.0))
        self.temp_pr = float(self.manifest.get("temp_promo", 1.0))

    def _run_t2_and_cond_t3(
        self,
        texts: list[str],
        z_char: np.ndarray,
        lg_r0_ai: np.ndarray,
        lg_r0_pr: np.ndarray,
        force_t3: bool = False,
    ) -> dict[str, np.ndarray]:
        cleans, f56_l = [], []
        for t in texts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56_l.append(f56)
        encs = [encode_bpe_head_tail_v130(self.tokenizer, c, max_len=self.max_len, head_len=self.head_len) for c in cleans]
        engs = [make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130) for s in encs]
        trigs = [make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130) for s in encs]
        ml = max(2, max(len(s) for s in encs))
        if ml % 2 == 1:
            ml += 1
        bs = len(texts)
        ids = np.full((bs, ml), self.tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((bs, ml), dtype=np.int64)
        eg = np.zeros((bs, ml), dtype=np.int64)
        tg = np.zeros((bs, ml), dtype=np.int64)
        for i, s in enumerate(encs):
            ids[i, : len(s)], mask[i, : len(s)], eg[i, : len(s)], tg[i, : len(s)] = s, 1, engs[i], trigs[i]
        st = np.stack(f56_l, axis=0).astype(np.float32)

        t2_raw = self.sess_t2.run(None, {
            "input_ids": ids, "attention_mask": mask, "style_feats": st, "engram_ids": eg, "trigram_ids": tg,
            "z_char": z_char.astype(np.float32), "logit_ai_r0": lg_r0_ai.astype(np.float32), "logit_promo_r0": lg_r0_pr.astype(np.float32),
        })
        t2 = dict(zip(_T2_OUT_NAMES, t2_raw))

        lg_ai = t2["logit_ai_t2_final"].copy()
        lg_pr = t2["logit_promo_t2_final"].copy()
        emb_ai = t2["emb_ai_t2"].copy()
        emb_pr = t2["emb_pr_t2"].copy()
        rungs = np.full((bs,), 2, dtype=np.int64)

        t3_en = bool(self.t3_policy.get("enabled", True)) or force_t3
        if t3_en:
            p_esc = t2["p_escalate_t3"]
            p_t2 = expit(np.clip(lg_ai / self.temp_ai, -30.0, 30.0))
            tau_e = float(self.t3_policy.get("tau_escalate", 0.45))
            u_lo = float(self.t3_policy.get("uncertain_lo", 0.12))
            u_hi = float(self.t3_policy.get("uncertain_hi", 0.88))
            if force_t3:
                esc_idx = np.arange(bs, dtype=np.int64)
            else:
                esc_mask = (p_esc >= tau_e) | ((p_t2 >= u_lo) & (p_t2 <= u_hi))
                esc_idx = np.where(esc_mask)[0].astype(np.int64)

            if len(esc_idx) > 0:
                t3_feed = {
                    "slots_t2": t2["slots_t2"][esc_idx], "slots_t1": t2["slots_t1"][esc_idx],
                    "anchor": t2["anchor"][esc_idx], "carry_m_l_t2": t2["carry_m_l_t2"][esc_idx],
                    "zh_bus_t2": t2["zh_bus_t2"][esc_idx], "ebt_grad_norm_t2": t2["ebt_grad_norm_t2"][esc_idx],
                    "k_ev": t2["k_ev"][esc_idx], "v_ev": t2["v_ev"][esc_idx], "attention_mask": mask[esc_idx],
                    "k_haar": t2["k_haar"][esc_idx], "v_haar": t2["v_haar"][esc_idx], "haar_mask": t2["haar_mask"][esc_idx],
                    "h0_pool": t2["h0_pool"][esc_idx], "h1_pool": t2["h1_pool"][esc_idx],
                    "st_ai": t2["st_ai"][esc_idx], "st_pr": t2["st_pr"][esc_idx], "z_char": z_char[esc_idx].astype(np.float32),
                    "logit_ai_r0": lg_r0_ai[esc_idx].astype(np.float32), "logit_promo_r0": lg_r0_pr[esc_idx].astype(np.float32),
                    "logit_ai_r1": t2["logit_ai_r1"][esc_idx], "logit_promo_r1": t2["logit_promo_r1"][esc_idx],
                    "logit_ai_t1": t2["logit_ai_t1"][esc_idx], "logit_promo_t1": t2["logit_promo_t1"][esc_idx],
                    "logit_ai_t2": t2["logit_ai_t2"][esc_idx], "emb_ai_t1": t2["emb_ai_t1"][esc_idx],
                }
                t3_out = dict(zip(_T3_OUT_NAMES, self.sess_t3.run(None, t3_feed)))
                lg_ai[esc_idx] = t3_out["logit_ai_t3_final"]
                lg_pr[esc_idx] = t3_out["logit_promo_t3_final"]
                emb_ai[esc_idx] = t3_out["emb_ai_t3"]
                emb_pr[esc_idx] = t3_out["emb_pr_t3"]
                rungs[esc_idx] = 3

        return {"logit_ai": lg_ai, "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr, "exit_rung": rungs}

    def _run_long_doc(self, text: str) -> tuple[float, float]:
        chunks = plan_long_chunks_v11(text, max_chunks=8)
        if len(chunks) <= 1:
            c1, c2, c3, cm, cf, _ = extract_apex_char_batch_v130([text])
            la0, lp0, _, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
            od = self._run_t2_and_cond_t3([text], zc, la0, lp0, force_t3=False)
            return float(od["logit_ai"][0]), float(od["logit_promo"][0])
        ch_txts = [c["text"] for c in chunks]
        c1, c2, c3, cm, cf, _ = extract_apex_char_batch_v130(ch_txts)
        la0, lp0, _, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
        od = self._run_t2_and_cond_t3(ch_txts, zc, la0, lp0, force_t3=False)
        metas = np.stack([c["chunk_meta"] for c in chunks], axis=0).astype(np.float32)[None, :, :]
        valid = np.ones((1, len(chunks)), dtype=np.float32)
        doc_ai, doc_pr = self.sess_long.run(None, {
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
        force_t3: bool = False,
    ) -> list[dict[str, Any]]:
        n = len(texts)
        if n == 0:
            return []
        results: list[dict[str, Any] | None] = [None] * n
        short_idxs = [i for i, t in enumerate(texts) if len(t) < self.long_char_trigger]
        long_idxs = [i for i, t in enumerate(texts) if len(t) >= self.long_char_trigger]

        for oi in long_idxs:
            lg_a, lg_p = self._run_long_doc(texts[oi])
            pa = float(expit(np.clip(lg_a / self.temp_ai, -30.0, 30.0)))
            pp = float(expit(np.clip(lg_p / self.temp_pr, -30.0, 30.0)))
            if self.lut_ai_full:
                pa = float(apply_spline_temp_lut(np.array([pa]), self.lut_ai_full)[0])
            if self.lut_pr_full:
                pp = float(apply_spline_temp_lut(np.array([pp]), self.lut_pr_full)[0])
            results[oi] = {"prob_ai": pa, "prob_promo": pp, "exit_rung": 4, "logit_ai": lg_a, "logit_promo": lg_p}

        order = sorted(short_idxs, key=lambda i: len(texts[i]))
        r0_en = bool(self.r0_policy.get("enabled", False)) and not force_full and not force_t3
        r0_tau_ai = float(self.r0_policy.get("tau_ai", 0.06))
        r0_tau_pr = float(self.r0_policy.get("tau_promo", 0.06))
        r0_def_max = float(self.r0_policy.get("defer_max", 0.25))
        r0_max_ch = int(self.r0_policy.get("max_chars", 160))

        for st in range(0, len(order), max_batch):
            mb = order[st : st + max_batch]
            mb_txts = [texts[i] for i in mb]
            c1, c2, c3, cm, cf, spoofs = extract_apex_char_batch_v130(mb_txts)
            la0, lp0, pdef0, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
            pa0 = expit(np.clip(la0, -30.0, 30.0))
            pp0 = expit(np.clip(lp0, -30.0, 30.0))
            if self.lut_ai_r0:
                pa0 = apply_spline_temp_lut(pa0, self.lut_ai_r0)
            if self.lut_pr_r0:
                pp0 = apply_spline_temp_lut(pp0, self.lut_pr_r0)

            surv: list[int] = []
            for ri, oi in enumerate(mb):
                if (
                    r0_en and len(texts[oi]) <= r0_max_ch and spoofs[ri] == 0.0 and ("http" not in texts[oi].lower())
                    and float(pdef0[ri]) <= r0_def_max and float(pa0[ri]) <= r0_tau_ai and float(pp0[ri]) <= r0_tau_pr
                ):
                    results[oi] = {"prob_ai": float(pa0[ri]), "prob_promo": float(pp0[ri]), "exit_rung": 0, "logit_ai": float(la0[ri]), "logit_promo": float(lp0[ri])}
                else:
                    surv.append(ri)
            if not surv:
                continue
            s_idx = np.array(surv, dtype=np.int64)
            od = self._run_t2_and_cond_t3([mb_txts[k] for k in surv], zc[s_idx], la0[s_idx], lp0[s_idx], force_t3=force_t3)
            pa_f = expit(np.clip(od["logit_ai"] / self.temp_ai, -30.0, 30.0))
            pp_f = expit(np.clip(od["logit_promo"] / self.temp_pr, -30.0, 30.0))
            if self.lut_ai_full:
                pa_f = apply_spline_temp_lut(pa_f, self.lut_ai_full)
            if self.lut_pr_full:
                pp_f = apply_spline_temp_lut(pp_f, self.lut_pr_full)
            for j, k in enumerate(surv):
                oi = mb[k]
                results[oi] = {
                    "prob_ai": float(pa_f[j]), "prob_promo": float(pp_f[j]),
                    "exit_rung": int(od["exit_rung"][j]),
                    "logit_ai": float(od["logit_ai"][j]), "logit_promo": float(od["logit_promo"][j]),
                }
        return [r for r in results if r is not None]

    def predict_with_host_meta(self, text: str, host_meta: dict[str, float], meta_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Strictly opt-in, detached host-side metadata adapter (neural core stays 100% Zero-PII!)."""
        base = self.predict_batch([text], force_full=True)[0]
        cfg = meta_cfg or self.manifest.get("optional_host_meta_adapter")
        if not cfg:
            return base
        z = float(cfg.get("bias", 0.0)) + float(cfg.get("w_neural", 1.0)) * float(base["logit_ai"])
        for k, w in (cfg.get("feature_weights") or {}).items():
            z += float(w) * float(host_meta.get(k, 0.0))
        out = dict(base)
        out["prob_ai_host_fused"] = float(expit(np.clip(z, -30.0, 30.0)))
        return out


# ---------------------------------------------------------------------------
# 9. Automated 12-Gate v1.3.0 CI Suite (Nested-Loop + T3 Modular Parity + Structural Probes!)
# ---------------------------------------------------------------------------

def run_v130_ci_gates(
    model: MeguvizV130XtremeApexModel,
    tokenizer: BiliPickyBPETokenizer,
    export_rep: dict[str, Any] | None = None,
    ai_lut_check: dict[str, Any] | None = None,
    promo_lut_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    ci: dict[str, Any] = {}

    def _infer(txts: list[str], force_pad_len: int | None = None, outer_steps: int = 3):
        c1, c2, c3, cm, cf, _ = extract_apex_char_batch_v130(txts)
        cleans, f56_l = [], []
        for t in txts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56_l.append(f56)
        encs = [encode_bpe_head_tail_v130(tokenizer, c, max_len=model.max_len) for c in cleans]
        ml = max(2, max(len(s) for s in encs))
        if force_pad_len is not None:
            ml = max(ml, force_pad_len)
        if ml % 2 == 1:
            ml += 1
        ids = np.full((len(txts), ml), tokenizer.pad_id, dtype=np.int64)
        msk = np.zeros((len(txts), ml), dtype=np.int64)
        eg = np.zeros((len(txts), ml), dtype=np.int64)
        tg = np.zeros((len(txts), ml), dtype=np.int64)
        for i, s in enumerate(encs):
            ids[i, : len(s)], msk[i, : len(s)] = s, 1
            eg[i, : len(s)] = make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130)
            tg[i, : len(s)] = make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130)
        with torch.no_grad():
            return model(
                torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
                torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device),
                torch.from_numpy(eg).to(device), torch.from_numpy(tg).to(device),
                torch.from_numpy(c1).to(device), torch.from_numpy(c2).to(device),
                torch.from_numpy(c3).to(device), torch.from_numpy(cm).to(device),
                torch.from_numpy(cf).to(device), outer_steps=outer_steps,
            )

    # 1. V130_C1: Strict Zero-PII across ALL rungs (R0, R1, T1, T2, T3) === 0.0!
    oa = _infer(["回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"], outer_steps=3)
    ob = _infer(["回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"], outer_steps=3)
    pii_keys = ("logit_ai_r0", "logit_promo_r0", "logit_ai_r1", "logit_promo_r1", "logit_ai_t2_final", "logit_ai", "logit_promo", "p_escalate_t3")
    pii_d = float(max(abs(float(oa[k][0].item()) - float(ob[k][0].item())) for k in pii_keys))
    ci["V130_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # 2. V130_F6: Structural Promo->AI Firewall INCLUDING `+12` CBM Gov/Promo Bias Perturbation across T1/T2/T3/Halt!
    t_fw = ["笑死我了，这个插件装完直接起飞[doge]", "关注工种号交流2群 168688884 领全套资料"]
    o_base = _infer(t_fw, outer_steps=3)
    with torch.no_grad():
        saved_cbm_b = model.cbm_pr_enc.bias.clone()
        saved_pr_b = model.promo_head.bias.clone()
        saved_r1_pb = model.r1_promo_head.bias.clone()
        saved_r0_pb = model.rung0.promo_head.bias.clone()
        model.cbm_pr_enc.bias.add_(12.0)
        model.promo_head.bias.add_(4.0)
        model.r1_promo_head.bias.add_(4.0)
        model.rung0.promo_head.bias.add_(4.0)
        o_pert = _infer(t_fw, outer_steps=3)
        model.cbm_pr_enc.bias.copy_(saved_cbm_b)
        model.promo_head.bias.copy_(saved_pr_b)
        model.r1_promo_head.bias.copy_(saved_r1_pb)
        model.rung0.promo_head.bias.copy_(saved_r0_pb)
    fw_diff = float(max(
        (o_base[k] - o_pert[k]).abs().max().item()
        for k in ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t2_final", "logit_ai_t3", "logit_ai", "p_escalate_t3")
    ))
    ci["V130_F6_promo_to_ai_firewall"] = {"max_leak_diff_under_plus12_cbm": fw_diff, "pass": fw_diff == 0.0}

    # 3. V130_P1: Exact Batch-Padding Invariance across R0, R1, T1, T2, T3, EBT & Haar (`< 1e-4` at pad=256!)
    t_sh = "笑死我了，这个插件装完直接起飞[doge]"
    o_unp = _infer([t_sh], force_pad_len=None, outer_steps=3)
    o_pad = _infer([t_sh], force_pad_len=256, outer_steps=3)
    p_keys = ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2_final", "logit_ai", "logit_promo", "ebt_margin", "p_escalate_t3")
    p_diffs = {k: round(abs(float(o_unp[k][0].item()) - float(o_pad[k][0].item())), 8) for k in p_keys}
    max_pd = max(p_diffs.values())
    ci["V130_P1_pad_invariance_exact"] = {"diffs": p_diffs, "max_diff": max_pd, "pass": bool(max_pd < 1e-4)}

    # 4. V130_S1: Dynamic shapes (`outer_steps in {2, 3}`), Cayley Symplectic Orthogonality (`c^2+s^2===1`), & EBT6 grad parity!
    dyn_ok = True
    for os_n in (2, 3):
        oo = _infer(["短评测试", "这是一段较长的结构化技术点评，用于验证动态长度、嵌套内环与第三步纠错。"], outer_steps=os_n)
        if torch.isnan(oo["logit_ai"]).any() or torch.isnan(oo["logit_promo"]).any():
            dyn_ok = False
    with torch.no_grad():
        g = model.core.c1_cayley_gamma
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c_rot = (1.0 - g2) * inv
        s_rot = (2.0 * g) * inv
        cayley_err = float(((c_rot * c_rot + s_rot * s_rot) - 1.0).abs().max().item())
    u_t = torch.randn(4, BUS_DIM_V130, device=device, requires_grad=True)
    er = model.core.ebt.forward_energy_and_grad(u_t)
    ag = torch.autograd.grad(er["free_energy"].sum(), u_t)[0]
    ebt_err = float((er["grad_u"].detach() - ag.detach()).abs().max().item())
    ci["V130_S1_nested_symplectic_and_ebt6"] = {
        "dynamic_ok": dyn_ok, "cayley_unit_err": round(cayley_err, 8), "ebt6_grad_max_diff": round(ebt_err, 8),
        "pass": bool(dyn_ok and cayley_err < 1e-6 and ebt_err < 1e-5),
    }

    # 5. V130_L1: SoftBallLogitNorm bound
    with torch.no_grad():
        huge = torch.ones((2, AI_END_V130), device=device) * 100.0
        hb = model.ai_head.project_ball(huge).norm(dim=-1).max().item()
        rc = float(model.ai_head.effective_r_cap().item())
    ci["V130_L1_softball_bound"] = {"huge_ball_norm": round(hb, 4), "r_cap": round(rc, 4), "pass": bool(math.isfinite(hb) and hb <= rc + 1e-3)}

    # 6 & 7. Bundle size <= 5.00 MiB (ZERO 1.58-bit!) + 4-Stage Modular ONNX FP32 exact parity (incl. T2->T3!)
    if export_rep is not None:
        ci["V130_B8_bundle_le_5p00_mib"] = {
            "bundle_mib": export_rep["bundle_mib"],
            "bundle_bytes": export_rep["bundle_bytes"],
            "pass": bool(export_rep["bundle_le_5p00_mib"]),
        }
        ci["V130_ONNX_4stage_fp32_parity"] = {
            "r0_diff": export_rep["r0_fp32_max_diff"],
            "t2_diff": export_rep["t2_fp32_max_diff"],
            "t3_modular_diff": export_rep["t3_modular_fp32_max_diff"],
            "long_diff": export_rep["long_fp32_max_diff"],
            "pass": bool(export_rep["onnx_exact_pass"]),
        }

    # 8 & 9. Clean AI & Promo LUT floors
    if ai_lut_check is not None:
        ci["V130_AI_LUT_FLOOR"] = {**ai_lut_check, "pass": bool(ai_lut_check.get("lut_ai_r0_at_001", 1.0) <= 0.025)}
    if promo_lut_check is not None:
        ci["V130_PR_LUT_FLOOR"] = {**promo_lut_check, "pass": bool(promo_lut_check.get("median_promo_wild", 1.0) <= 0.12)}

    ci["all_ci_passed"] = all(bool(v.get("pass")) for v in ci.values() if isinstance(v, dict))
    return ci
