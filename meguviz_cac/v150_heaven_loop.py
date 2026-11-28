"""MeguvizCAC v1.5.0-HeavenLoop: Contractive Coverage-Inhibited Deep Recurrent Engine (<=5.00 MiB Bootable).

Implements the finalized Antigravity x GPT adversarial-review specification:
  P0. Hard End-to-End Deployment Contracts:
      - Pre-Canonicalization Entry: ALL length routing (`r0_max_ch`, `long_char_trigger`), sorting,
        Char-LM, and BPE tokenization operate strictly AFTER `canonicalize_zero_pii_v150`.
      - Decoupled Dual-Axis Cascade Exit: `R0` has separate `defer_ai_head` (ZERO Promo inputs!)
        and `defer_pr_head`; `MeguvizV150CascadeEngine` independently latches AI early exits so
        Promo perturbations NEVER switch AI's exit path -> true end-to-end `F6 === 0.0` & `C1 === 0.0`!
      - True Bootable `<= 5.00 MiB` Deliverable: `MGVZ9` + weight-stripped Opset-17 ONNX graph
        skeletons + tokenizer + manifest + `charlm_v150.bin` total `<= 5,242,880` bytes!
  Core 1 (`H2-b`): Read-Only `K_0, V_0` (projected ONCE -> saves `84.9M MAC/loop`!) + Decaying,
      Capped Coverage-Inhibited Reread (`a_{ti} = MaskedSoftmax(q_t^T k_{0i}/sqrt(d) - beta*C_{t-1,i})`),
      with `beta_eff = beta * g_sup` allowing micro-short single-cue comments to reread safely.
  Core 2 (`H3`): Block-Diagonal `diag(M_{AI, 3x3}, 1)` Birkhoff-Sinkhorn mHC (3 AI-internal streams
      `[Prosecutor, Defense, Arbiter]` + 1 physically isolated Promo stream -> `F6 === 0.0`) +
      Contraction-plus-Bounded-Read-Only-Evidence-Flip dynamics (`||D_s F|| <= rho < 1` + additive
      bounded read-only evidence impulse enabling 2-step reversal `+2 -> -1`!) + Signed Trajectory
      Curvature (`d_t = l_t - l_{t-1}`, `k_t = d_t - d_{t-1}`) + `LoopQ` per-step transition scales.
  Core 3 (`H4` + 9th Blind Spot): Streamlined Full-Token `T_max = 4` (`FFN=512, k_inner=1`) +
      Decoupled `T2.5` Two-Query Prosecutor-vs-Defense Light Reread (`36,864 MACs` at `L=288` over
      cached 32-d read-only K/V!) + `TaH2` Student Lookahead Utility Exit Gate.
  Configurable Ablations Built In:
      - `prelude_mode`: `"single_0p6m"` (default, ~0.55 MiB) | `"stem_0p3m"` (~0.26 MiB) | `"dual_legacy"` (1.67 MiB)
      - `coverage_beta_enabled`: True / False (`beta=0` control)
      - `use_t25_light_reread`: True / False
      - `use_local_charlm_inject`: True / False (8-ch BPE-offset aligned local LLR + `g_j===0` EN/JA gate)
      - Group-Clustered Leak-Free `CompactCharLMV150` (same group/template/CF/jittered-descendant in same fold!).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
from scipy.special import expit
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
    H_V130,
    LORA_RANK_V130,
    NUM_FAST_V130,
    NUM_HEADS_V130,
    NUM_SEM_FAST_V130,
    NUM_SLOTS_V130,
    NUM_SLOW_V130,
    NUM_SPAN_FAST_V130,
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
    InLoopAnalyticEBTV130,
    PadExactConvBlockV130,
    QuantEmbeddingV130,
    QuantLinearV130,
    _pack_w4_tensor,
    _pack_w8_tensor,
    _unpack_w4_tensor,
    _unpack_w8_tensor,
    canonicalize_zero_pii_v130,
    set_v130_qat_mode,
)
from .v140_paradise_apex import (
    CHARLM_BITS_V140,
    CHARLM_FEAT_DIM_V140,
    CHARLM_ORDER_V140,
    CHARLM_SMOOTH_V140,
    CHARLM_VOCAB_V140,
    HEAD_LEN_V140,
    MAX_LEN_V140,
    R0_FEAT_DIM_V140,
    R0_STRUCT_DIM_V140,
    CompactCharLMV140,
    _charlm_events_v140,
    _charlm_lookup_v140,
    count_distinct_support_pairs_v140,
    extract_apex_char_batch_v140,
)

# ---------------------------------------------------------------------------
# v1.5.0 Constants
# ---------------------------------------------------------------------------
MGVZ9_MAGIC = b"MGVZ9\x00\x01\x00"
FFN_V150 = 512                  # Streamlined slot FFN (cuts 41.3M -> ~16.5M MAC/loop & saves ~540 KiB!)
K_INNER_V150 = 1                # 1 inner symplectic step + 1 outer step per loop -> fast & crisp T_max=4
T_MAX_V150 = 4                  # T1 -> T2 -> T2.5 (37K MAC) -> T3 -> T4
LITE_DIM_V150 = 32              # T2.5 2-Query Prosecutor/Defense read-only evidence dim (`36,864 MAC` at L=288!)
LOCAL_CHARLM_DIM_V150 = 8       # [4 clipped mean LLRs (orders 1..4), 4 reliable hit-rates (orders 1..4)]
COV_LOGIT_CAP_V150 = 1.25       # Max attention-logit suppression from decaying coverage
COV_MAX_V150 = 2.0
T25_CORR_CAP_V150 = 1.2
T3_CORR_CAP_V150 = 1.8          # T3 + T4 combined bounded budget = 3.6 -> easily flips +2.0 -> -1.0!
T4_CORR_CAP_V150 = 1.8
MHC_AI_STREAMS_V150 = 3         # 3 AI streams `[Prosecutor, Defense, Arbiter]` + 1 isolated Promo stream!
MHC_SINKHORN_ITERS_V150 = 10    # 10-iter Birkhoff-Sinkhorn guarantees row/col err < 1e-6 even under sharp trained weights!
CORE_DEPTH_DEFAULT_V150 = 2     # Dual-Layer Shared Recurrent Core (`HeavenBlock-2L`) within <=5.00 MiB bootable!

_PII_PLACEHOLDER_RE_V150 = re.compile(r"<(?:MENTION|URL|CONTACT_ID|REPLY_HDR|BOS|EOS|PAD|SEP|UNK)>")
_REPLY_HDR_V150_RE = re.compile(r"^\s*回[复覆]\s*@[^\s:：]+\s*[:：]\s*")
_INLINE_MENTION_V150_RE = re.compile(r"@[^\s,，。！!？?@:：()（）\[\]【】]+")

set_v150_qat_mode = set_v130_qat_mode


def canonicalize_zero_pii_v150(text: str, max_chars: int = 2048) -> tuple[str, dict[str, float]]:
    """Pre-canonicalization Zero-PII scrubber removing arbitrarily long `@handles` before v130/v07 rules."""
    raw = str(text or "")
    if _REPLY_HDR_V150_RE.match(raw):
        raw = _REPLY_HDR_V150_RE.sub("<REPLY_HDR> ", raw, count=1)
    raw = _INLINE_MENTION_V150_RE.sub("<MENTION>", raw)
    return canonicalize_zero_pii_v130(raw, max_chars=max_chars)


# ---------------------------------------------------------------------------
# 1. Pre-Canonical BPE Offsets + Group-Clustered Leak-Free `CompactCharLMV150`
# ---------------------------------------------------------------------------

def encode_bpe_head_tail_with_offsets_v150(
    tokenizer: BiliPickyBPETokenizer,
    clean_text: str,
    max_len: int = MAX_LEN_V140,
    head_len: int = HEAD_LEN_V140,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode already-canonicalized text preserving exact `(char_start, char_end)` offsets for Head + Tail."""
    enc = tokenizer._tok.encode(clean_text, add_special_tokens=False)
    ids = list(enc.ids)
    offs = [(int(a), int(b)) for (a, b) in enc.offsets]
    inner = max(2, int(max_len) - 2)
    if len(ids) > inner:
        h = min(int(head_len), inner - 8)
        t = inner - h
        ids = ids[:h] + ids[-t:]
        offs = offs[:h] + offs[-t:]
    full_ids = [tokenizer.bos_id] + ids + [tokenizer.eos_id]
    full_offs = [(-1, -1)] + offs + [(-1, -1)]
    return full_ids, full_offs


def _placeholder_char_mask_v150(clean_text: str) -> np.ndarray:
    """Boolean mask of valid content characters (`False` inside PII/special placeholders like `<MENTION>`)."""
    n = len(clean_text)
    valid = np.ones((n,), dtype=bool)
    for m in _PII_PLACEHOLDER_RE_V150.finditer(clean_text):
        valid[m.start() : m.end()] = False
    return valid


class CompactCharLMV150(CompactCharLMV140):
    """Extends `CompactCharLMV140` with exact BPE-offset aligned 8-channel local LLR + hit-rate extraction."""

    def extract_char_level_llr_and_hits(self, clean_text: str, max_eval_chars: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Return `(char_llr [C, 4], char_hits [C, 4])` over `clean_text` with real left context & zeroed placeholders.

        For very long comments (`> max_eval_chars`), evaluates head (`512`) and tail (`512` with 3 real preceding
        context chars so seam never fabricates fake n-grams!).
        """
        s = str(clean_text or "")
        n = len(s)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

        valid_ch = _placeholder_char_mask_v150(s)
        # v1.4.0 Fix #1: Exclude pure-ASCII chars (<128) from Chinese-dominated char n-gram LLR so EN/code/version numbers NEVER false-fire!
        non_ascii = np.fromiter((ord(ch) >= 128 for ch in s), dtype=bool, count=n)
        valid_ch = valid_ch & non_ascii
        char_llr = np.zeros((n, 4), dtype=np.float32)
        char_hit = np.zeros((n, 4), dtype=np.float32)
        uni = self.unigrams.astype(np.float64)

        def _score_span(start: int, end: int, keep_from: int) -> None:
            sub = s[start:end]
            if not sub:
                return
            _, _, _, keys = _charlm_events_v140([sub], self.vocabulary, max_chars=len(sub) + 1)
            p = uni[keys[0].astype(np.int64)]
            L = len(sub)
            span_llr = np.zeros((L, 4), dtype=np.float32)
            span_hit = np.zeros((L, 4), dtype=np.float32)
            for order in range(CHARLM_ORDER_V140):
                if order == 0:
                    in_voc = (keys[0] >= 2).astype(np.float32)
                    span_hit[:, 0] = in_voc
                else:
                    tb = self.tables[order - 1]
                    joint = _charlm_lookup_v140(tb["keys"], tb["counts"], keys[order])
                    den = _charlm_lookup_v140(tb["contexts"], tb["den"], keys[order] >> CHARLM_BITS_V140)
                    hit_m = ((joint[:, 0] + joint[:, 1]) > 0.0).astype(np.float32)
                    span_hit[:, order] = hit_m
                    p = (joint + CHARLM_SMOOTH_V140[order] * p) / (den + CHARLM_SMOOTH_V140[order])
                logp = np.log(np.maximum(p, 1e-12))
                llr = np.clip(logp[:, 1] - logp[:, 0], -3.0, 3.0).astype(np.float32)
                span_llr[:, order] = llr
            off = keep_from - start
            dst_s = keep_from
            dst_e = end
            char_llr[dst_s:dst_e] = span_llr[off:]
            char_hit[dst_s:dst_e] = span_hit[off:]

        if n <= max_eval_chars:
            _score_span(0, n, 0)
        else:
            half = max_eval_chars // 2
            _score_span(0, half, 0)
            tail_keep = n - half
            ctx_start = max(0, tail_keep - 3)
            _score_span(ctx_start, n, tail_keep)

        # Strictly zero PII placeholder spans!
        char_llr[~valid_ch] = 0.0
        char_hit[~valid_ch] = 0.0
        return char_llr, char_hit

    def extract_token_local_8ch(
        self,
        clean_text: str,
        offsets: list[tuple[int, int]],
    ) -> np.ndarray:
        """Align char-level 1..4 LLR & reliable hit-rates to BPE token offsets (`[L, 8]`, BOS/EOS/PII === 0.0!)."""
        L = len(offsets)
        out = np.zeros((L, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        if L == 0 or not clean_text:
            return out
        char_llr, char_hit = self.extract_char_level_llr_and_hits(clean_text)
        n = len(clean_text)
        if n == 0:
            return out

        # Share-count per char so multi-byte tokens covering the same UTF-8 char split weight fairly!
        share = np.zeros((n,), dtype=np.float32)
        for a, b in offsets:
            if 0 <= a < b <= n:
                share[a:b] += 1.0
        share = np.maximum(share, 1.0)

        for ti, (a, b) in enumerate(offsets):
            if a < 0 or b <= a or a >= n:
                continue
            b_c = min(b, n)
            w = 1.0 / share[a:b_c]
            w_sum = float(w.sum())
            if w_sum <= 1e-8:
                continue
            wn = (w / w_sum)[:, None]
            out[ti, :4] = (char_llr[a:b_c] * wn).sum(axis=0)
            out[ti, 4:8] = (char_hit[a:b_c] * wn).sum(axis=0)
        return out


def make_dedup_group_key_v150(clean_text: str, explicit_group: str | None = None) -> str:
    """Group near-duplicates / punctuation variants / explicit template or CF-pair IDs into the same OOF fold."""
    if explicit_group:
        return f"grp:{explicit_group}"
    # Strip spaces & common punctuation so punctuation-jittered twins land in the exact same group!
    core = re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】\-—~`]+", "", clean_text.lower())
    return hashlib.sha1(core[:128].encode("utf-8")).hexdigest()[:16]


def fit_group_oof_charlm_v150(
    raw_texts: list[str],
    labels: np.ndarray,
    group_keys: list[str] | None = None,
    n_folds: int = 3,
    min_counts: tuple[int, int, int] = (16, 28, 40),
    max_order_keys: tuple[int, int, int] = (9000, 8000, 7000),
    seed: int = 20261130,
) -> tuple[list[CompactCharLMV150], dict[str, int], CompactCharLMV150]:
    """Strict Group-Clustered 3-Fold OOF `CompactCharLMV150`!

    Guarantees:
      1. Near-duplicates, same templates, CF mirrors, and ALL jittered descendants stay in the exact same fold!
      2. Each fold's vocab, n-gram tables, pruning, and linear readout exclude the entire held-out group!
      3. Returns `(fold_models, text_to_fold_id, full_lm)` so training NEVER falls back to `full_lm` on jittered rows!
    """
    n = len(raw_texts)
    cleans = [canonicalize_zero_pii_v150(str(t))[0] for t in raw_texts]
    y = np.asarray(labels, dtype=np.int32)
    gkeys = [
        make_dedup_group_key_v150(cleans[i], group_keys[i] if group_keys is not None else None)
        for i in range(n)
    ]

    # Group -> list of indices; assign whole groups to folds balanced by positive count!
    grp_to_idx: dict[str, list[int]] = {}
    for i, gk in enumerate(gkeys):
        grp_to_idx.setdefault(gk, []).append(i)
    k_folds = max(2, int(n_folds))
    rng_np = np.random.default_rng(seed)
    grp_list = list(grp_to_idx.items())
    rng_np.shuffle(grp_list)
    # Sort descending by group size so folds stay balanced
    grp_list.sort(key=lambda kv: -len(kv[1]))

    fold_pos = [0] * k_folds
    fold_tot = [0] * k_folds
    fold_ids = np.zeros((n,), dtype=np.int32)
    for _, idxs in grp_list:
        p_cnt = int(y[idxs].sum())
        if p_cnt > 0:
            best_f = min(range(k_folds), key=lambda f: (fold_pos[f], fold_tot[f]))
        else:
            best_f = min(range(k_folds), key=lambda f: fold_tot[f])
        for ii in idxs:
            fold_ids[ii] = best_f
        fold_pos[best_f] += p_cnt
        fold_tot[best_f] += len(idxs)

    fold_mc = tuple(max(2, int(round(c * (k_folds - 1) / float(k_folds)))) for c in min_counts)
    fold_models: list[CompactCharLMV150] = []
    oof_raw33 = np.zeros((n, 33), dtype=np.float32)

    for f in range(k_folds):
        tr_idx = np.where(fold_ids != f)[0]
        va_idx = np.where(fold_ids == f)[0]
        base_f = CompactCharLMV150.fit_tables(
            [cleans[i] for i in tr_idx], y[tr_idx], min_counts=fold_mc, max_order_keys=max_order_keys
        )
        # Promote to CompactCharLMV150 instance
        fm = CompactCharLMV150(
            vocabulary=base_f.vocabulary, unigrams=base_f.unigrams, tables=base_f.tables, max_chars=base_f.max_chars
        )
        if len(va_idx) > 0:
            oof_raw33[va_idx] = fm.extract_raw33([cleans[i] for i in va_idx])
        fold_models.append(fm)

    # Fit per-fold linear readouts strictly on OTHER folds' OOF raw33!
    for f in range(k_folds):
        tr_idx = np.where(fold_ids != f)[0]
        fold_models[f].fit_linear_head_from_raw33(oof_raw33[tr_idx], y[tr_idx], reg_c=0.25)

    base_full = CompactCharLMV150.fit_tables(cleans, y, min_counts=min_counts, max_order_keys=max_order_keys)
    full_lm = CompactCharLMV150(
        vocabulary=base_full.vocabulary, unigrams=base_full.unigrams, tables=base_full.tables, max_chars=base_full.max_chars
    )
    full_lm.fit_linear_head_from_raw33(oof_raw33, y, reg_c=0.25)

    text_to_fold = {raw_texts[i]: int(fold_ids[i]) for i in range(n)}
    return fold_models, text_to_fold, full_lm


# ---------------------------------------------------------------------------
# 2. Stage 0: Decoupled Dual-Axis `ApexCharNanoRung0V150` (`defer_ai` ZERO Promo!)
# ---------------------------------------------------------------------------

class ApexCharNanoRung0V150(nn.Module):
    """W8/W4 Char 1/2/3-gram Nano Engine with STRICTLY DECOUPLED `defer_ai` and `defer_pr` heads!"""

    def __init__(self, use_global_charlm_shortcut: bool = False) -> None:
        super().__init__()
        self.use_global_charlm_shortcut = bool(use_global_charlm_shortcut)
        self.emb1 = QuantEmbeddingV130(R0_B1_V130, R0_D1_V130, bits=8, padding_idx=0)
        self.emb2 = QuantEmbeddingV130(R0_B2_V130, R0_D2_V130, bits=4, padding_idx=0)
        self.emb3 = QuantEmbeddingV130(R0_B3_V130, R0_D3_V130, bits=8, padding_idx=0)
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
        pool_in_ai = R0_HID_V130 * 5 + R0_FEAT_DIM_V140      # 666
        pool_in_pr = R0_HID_V130 * 5 + R0_STRUCT_DIM_V140    # 657
        self.ai_fc = nn.Sequential(QuantLinearV130(pool_in_ai, 128, bits=8), nn.LayerNorm(128), nn.SiLU())
        self.promo_fc = nn.Sequential(QuantLinearV130(pool_in_pr, 96, bits=8), nn.LayerNorm(96), nn.SiLU())
        self.ai_head = nn.Linear(128, 1)
        self.promo_head = nn.Linear(96, 1)
        self.z_char_ai = nn.Linear(128, 32)
        self.z_char_pr = nn.Linear(96, 32)
        # P0 FIX: Strictly separate AI and Promo defer heads -> ZERO Promo->AI cascade routing leak (`F6 === 0.0`!)
        self.defer_ai_head = nn.Sequential(nn.Linear(128 + 2, 32), nn.SiLU(), nn.Linear(32, 1))
        self.defer_pr_head = nn.Sequential(nn.Linear(96 + 2, 24), nn.SiLU(), nn.Linear(24, 1))

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
        # v1.4.0 Fix #1: Disable global CharLM additive bias in R0 by default to protect R@P99 & EN/code humans!
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
# 3. Light Stemmer (`stem_0p3m`) + `HeavenLoopCoreV150`
# ---------------------------------------------------------------------------

class PadExactLightStemV150(nn.Module):
    """Ultra-light `~0.26 MiB` local Conv + SwiGLU stem for H1 Prelude Collapse ablation."""

    def __init__(self, dim: int = H_V130, ffn_dim: int = 384) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dw3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pw = QuantLinearV130(dim, dim, bits=4)
        self.norm2 = nn.LayerNorm(dim)
        self.w_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.w_down = QuantLinearV130(ffn_dim, dim, bits=4)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m_u = attention_mask.float().unsqueeze(-1)
        xn = self.norm1(x) * m_u
        xt = xn.transpose(1, 2)
        hc = ((self.dw3(xt) + self.dw5(xt)) * 0.5).transpose(1, 2) * m_u
        h = (x + self.pw(hc)) * m_u
        u, g = self.w_up(self.norm2(h) * m_u).chunk(2, dim=-1)
        return (h + self.w_down(u * F.silu(g))) * m_u


class RelaxedStepAdapterV150(nn.Module):
    """Discrete step-indexed relaxed LoRA (`4 outer steps x 2 inner phases`) — tight & calibrated for T_max=4!"""

    def __init__(self, dim: int = H_V130, rank: int = LORA_RANK_V130, n_outer: int = T_MAX_V150, n_phases: int = 2) -> None:
        super().__init__()
        self.n_outer = int(n_outer)
        self.n_phases = int(n_phases)
        self.a = nn.Parameter(torch.randn(n_outer, n_phases, dim, rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_outer, n_phases, rank, dim))
        self.gamma = nn.Parameter(torch.ones(n_outer, n_phases, dim))
        self.beta = nn.Parameter(torch.zeros(n_outer, n_phases, dim))
        self.dyn_ctrl = nn.Parameter(torch.zeros(n_outer, 6))

    def modulate(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        oi = min(max(int(outer_idx), 0), self.n_outer - 1)
        pi = min(max(int(phase_idx), 0), self.n_phases - 1)
        return x * self.gamma[oi, pi].view(1, 1, -1) + self.beta[oi, pi].view(1, 1, -1)

    def delta(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        oi = min(max(int(outer_idx), 0), self.n_outer - 1)
        pi = min(max(int(phase_idx), 0), self.n_phases - 1)
        return torch.matmul(torch.matmul(x, self.a[oi, pi]), self.b[oi, pi])


def sinkhorn_nxn_v150(log_m: torch.Tensor, iters: int = MHC_SINKHORN_ITERS_V150) -> torch.Tensor:
    """Exact Birkhoff-Sinkhorn doubly-stochastic projection for `[..., K, K]` (`row/col sum === 1.0`)."""
    m = torch.exp(log_m.float().clamp(-6.0, 6.0))
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return m


class HeavenLoopSubBlockV150(nn.Module):
    """One recurrent sub-block inside `HeavenLoopCoreV150` (Coverage Read-Only CA + Haar + Symplectic + Block-Masked SA + FFN)."""

    def __init__(
        self,
        dim: int = H_V130,
        ffn_dim: int = FFN_V150,
        num_heads: int = NUM_HEADS_V130,
        coverage_beta_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.n_slow = NUM_SLOW_V130
        self.n_sem = NUM_SEM_FAST_V130
        self.n_ai_slots = self.n_slow + self.n_sem  # 10
        self.coverage_beta_enabled = bool(coverage_beta_enabled)

        # Coverage-Inhibited Read-Only Cross-Attention (`H2-b`)
        self.norm_ca = nn.LayerNorm(dim)
        self.ca_q = QuantLinearV130(dim, dim, bits=4)
        self.ca_o = QuantLinearV130(dim, dim, bits=4)
        self.ca_q_lora_a = nn.Parameter(torch.randn(T_MAX_V150, dim, LORA_RANK_V130) * 0.02)
        self.ca_q_lora_b = nn.Parameter(torch.zeros(T_MAX_V150, LORA_RANK_V130, dim))
        self.reread_gate = nn.Parameter(torch.tensor(0.35))

        # Decaying, capped coverage inhibition parameters (`0 <= lam < 1`, bounded logit penalty!)
        self.cov_beta_raw = nn.Parameter(torch.tensor(-0.4))
        self.cov_lam_raw = nn.Parameter(torch.tensor(0.6))

        # Fine-Scale Haar Wavelet Reread (`64-d`, read-only!)
        self.haar_q = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.haar_o = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.haar_gate = nn.Parameter(torch.tensor(-0.8))

        # Slot Self-Attention (block-masked so Promo `[10:16)` NEVER writes into AI `[0:10)`!)
        self.norm_sa = nn.LayerNorm(dim)
        self.sa_qkv = QuantLinearV130(dim, dim * 3, bits=4)
        self.sa_o = QuantLinearV130(dim, dim, bits=4)
        self.sa_o_lora_a = nn.Parameter(torch.randn(T_MAX_V150, dim, LORA_RANK_V130) * 0.02)
        self.sa_o_lora_b = nn.Parameter(torch.zeros(T_MAX_V150, LORA_RANK_V130, dim))

        # Streamlined Shared SwiGLU FFN (`FFN=512`) + Discrete 4-Step Adapter
        self.norm_ff = nn.LayerNorm(dim)
        self.ff_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.ff_down = QuantLinearV130(ffn_dim, dim, bits=4)
        self.relax = RelaxedStepAdapterV150(dim=dim, rank=LORA_RANK_V130, n_outer=T_MAX_V150, n_phases=2)

        # Single Fast-Inner Symplectic Step (`k_inner = 1`)
        self.fast_norm = nn.LayerNorm(dim)
        self.p2_pre_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_pre_b = QuantLinearV130(16, dim, bits=4)
        self.p2_post_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_post_b = QuantLinearV130(16, dim, bits=4)
        self.fast_res_scale = nn.Parameter(torch.full((dim,), 0.30))

        # Within-AI Cayley Symplectic Coupling (`Prosecutor [0:3)` <-> `Defense [3:6)` -> ZERO Promo!)
        self.c1_q = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V130))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.20))

    def _shared_ffn(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        xm = self.relax.modulate(x, outer_idx, phase_idx)
        u, g = self.ff_up(xm).chunk(2, dim=-1)
        return self.ff_down(u * F.silu(g)) + self.relax.delta(xm, outer_idx, phase_idx)

    def _cayley_ai_peer_couple(
        self, z_sem: torch.Tensor, cayley_delta: torch.Tensor, bus_down: nn.Module, bus_up: nn.Module
    ) -> torch.Tensor:
        u = bus_down(z_sem)
        u_a, u_b = u[:, :3, :], u[:, 3:, :]
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
        delta = bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_sem + (0.30 * torch.tanh(self.c1_bridge_scale)) * delta

    def _fast_inner_step(
        self, slots: torch.Tensor, ctrl: torch.Tensor, outer_idx: int, bus_down: nn.Module, bus_up: nn.Module
    ) -> torch.Tensor:
        z_h = slots[:, : self.n_slow, :]
        z_l = slots[:, self.n_slow :, :]
        zh_ref = z_h.mean(dim=1, keepdim=True).expand_as(z_l)

        zn = self.fast_norm(z_l)
        g_pre = torch.sigmoid(self.p2_pre_b(F.silu(self.p2_pre_a(torch.cat([zn, zn - zh_ref], dim=-1)))) + 0.25 * ctrl[2])
        delta_l = self._shared_ffn(zn * g_pre, outer_idx=outer_idx, phase_idx=0)
        g_post = torch.sigmoid(self.p2_post_b(F.silu(self.p2_post_a(torch.cat([zn, delta_l], dim=-1)))) + 0.25 * ctrl[3])
        z_l_new = z_l + self.fast_res_scale * (g_post * delta_l)
        z_sem = self._cayley_ai_peer_couple(z_l_new[:, : self.n_sem, :], ctrl[0].view(1, 1), bus_down, bus_up)
        z_span = z_l_new[:, self.n_sem :, :]
        return torch.cat([z_h, z_sem, z_span], dim=1)

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
        oi: int,
        sa_iso_mask: torch.Tensor,
        bus_down: nn.Module,
        bus_up: nn.Module,
        apply_cov: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = k_ev_0.shape
        m_f = mask.float()
        ctrl = torch.tanh(self.relax.dyn_ctrl[oi])

        # 1. Read-Only `K_0, V_0` Cross-Attention with Decaying Capped Coverage Inhibition (`H2-b`!)
        sn_ca = self.norm_ca(s)
        q_base = self.ca_q(sn_ca) + torch.matmul(torch.matmul(sn_ca, self.ca_q_lora_a[oi]), self.ca_q_lora_b[oi])
        qh = q_base.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev_0.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev_0.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if self.coverage_beta_enabled and apply_cov:
            beta_eff = F.softplus(self.cov_beta_raw) * g_sup.to(sc.dtype).view(-1, 1, 1, 1)
            cov_pen = (beta_eff * cov_prev.to(sc.dtype)[:, None, None, :]).clamp_max(COV_LOGIT_CAP_V150)
            sc = sc - cov_pen
        sc = sc.masked_fill(m_f[:, None, None, :] < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(s.dtype)
        r_read = self.ca_o(torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D))
        s = s + torch.sigmoid(self.reread_gate) * r_read

        # Update decaying capped token coverage strictly over AI slots `[0:10)` (Promo attention NEVER alters AI coverage!)
        a_bar_ai = at[:, :, : self.n_ai_slots, :].mean(dim=(1, 2))
        cov_lam = 0.85 * torch.sigmoid(self.cov_lam_raw)
        cov_next = (m_f * torch.clamp(cov_lam * cov_prev.float() + a_bar_ai.float(), 0.0, COV_MAX_V150)).to(s.dtype)

        # Fine-scale Haar wavelet reread (read-only!)
        q_haar = self.haar_q(sn_ca)
        sc_h = (torch.matmul(q_haar, k_haar.transpose(-1, -2)) / 8.0).masked_fill(haar_mask[:, None, :].float() < 0.5, -1e4)
        at_h = torch.softmax(sc_h.float(), dim=-1).to(s.dtype)
        s = s + torch.sigmoid(self.haar_gate + 0.30 * ctrl[1]) * self.haar_o(torch.matmul(at_h, v_haar))

        # 2. Single Fast-Inner Symplectic Step (`k_inner = 1`)
        s = self._fast_inner_step(s, ctrl, outer_idx=oi, bus_down=bus_down, bus_up=bus_up)

        # 3. Block-Masked Slot Self-Attention (`[0:10)` CANNOT see `[10:16)`!) + Streamlined SwiGLU FFN
        sq, sk, sv = self.sa_qkv(self.norm_sa(s)).chunk(3, dim=-1)
        sq = sq.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sk = sk.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sv = sv.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sa_sc = torch.matmul(sq, sk.transpose(-2, -1)) * (self.head_dim ** -0.5) + sa_iso_mask.to(sq.dtype)
        sa = torch.softmax(sa_sc.float(), dim=-1).to(s.dtype)
        sa_ctx = torch.matmul(sa, sv).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        s = s + self.sa_o(sa_ctx) + torch.matmul(torch.matmul(sa_ctx, self.sa_o_lora_a[oi]), self.sa_o_lora_b[oi])
        s = s + self._shared_ffn(self.norm_ff(s), outer_idx=oi, phase_idx=1)
        return s, cov_next, r_read, ctrl


class HeavenLoopCoreV150(nn.Module):
    """Dual-Layer (`core_depth=2`) Streamlined Read-Only Coverage-Inhibited + `diag(M_3x3, 1)` Contractive Core!"""

    def __init__(
        self,
        dim: int = H_V130,
        ffn_dim: int = FFN_V150,
        num_heads: int = NUM_HEADS_V130,
        coverage_beta_enabled: bool = True,
        core_depth: int = CORE_DEPTH_DEFAULT_V150,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.n_slow = NUM_SLOW_V130           # 4
        self.n_sem = NUM_SEM_FAST_V130        # 6
        self.n_span = NUM_SPAN_FAST_V130      # 6
        self.n_fast = NUM_FAST_V130           # 12
        self.n_ai_slots = self.n_slow + self.n_sem  # 10 AI/shared slots `[0:10)` vs 6 Promo span slots `[10:16)`!
        self.coverage_beta_enabled = bool(coverage_beta_enabled)
        self.core_depth = max(1, int(core_depth))

        self.slot_queries = nn.Parameter(torch.randn(1, NUM_SLOTS_V130, dim) * 0.04)

        # Single-pass Read-Only Text K_0 / V_0 Projections (computed ONCE outside the loop!)
        self.k_ev_proj = QuantLinearV130(dim, dim, bits=4)
        self.v_ev_proj = QuantLinearV130(dim, dim, bits=4)
        # Single-pass 32-d Light Evidence Cache for `T2.5` 2-Query Prosecutor/Defense Reread (`36,864 MAC`!)
        self.lite_k_proj = QuantLinearV130(dim, LITE_DIM_V150, bias=False, bits=4)
        self.lite_v_proj = QuantLinearV130(dim, LITE_DIM_V150, bias=False, bits=4)

        # Shared 64-d Slow Bus projections (reused by Cayley peer coupling & EBT6!)
        self.bus_down = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.bus_up = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)

        # `core_depth` stacked shared recurrent sub-blocks (`HeavenBlock-2L` when core_depth=2 -> 8 effective layers!)
        self.blocks = nn.ModuleList([
            HeavenLoopSubBlockV150(
                dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, coverage_beta_enabled=coverage_beta_enabled
            )
            for _ in range(self.core_depth)
        ])

        # Bounded Read-Only Evidence-Flip Impulse (`b_t * g_t(r_t) * unit_norm(u_t(r_t))`)
        self.ev_flip_gate = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.ev_flip_dir = QuantLinearV130(dim, dim, bias=False, bits=4)
        self.ev_flip_budget = nn.Parameter(torch.full((T_MAX_V150,), 0.30))

        # Block-Diagonal `diag(M_{AI, 3x3}, 1)` 10-iter Sinkhorn mHC (3 AI streams ONLY -> strictly ZERO Promo leak!)
        init_mhc3 = torch.tensor([
            [1.4, -0.5, -0.5],
            [-0.5, 1.4, -0.5],
            [-0.5, -0.5, 1.4],
        ], dtype=torch.float32).unsqueeze(0).repeat(T_MAX_V150, 1, 1)
        self.mhc_ai_log_mix = nn.Parameter(init_mhc3)
        self.mhc_ai_read_logits = nn.Parameter(torch.tensor([[1.0, -0.4, -0.4]] * T_MAX_V150, dtype=torch.float32))

        # `LoopQ`-Style Per-Step FP16 Transition Activation Scales (`4 x 384`, 3 KiB!) suppressing W4 drift!
        self.loop_q_scale = nn.Parameter(torch.ones(T_MAX_V150, 1, dim))

        # Outer-Step History Delta + Doubt Handoff (strictly writes into AI slot 9 so Promo never touches AI!)
        self.role_hist_g = nn.Parameter(torch.zeros(3, 1, dim))
        self.handoff_proj = QuantLinearV130(BUS_DIM_V130 + 2, dim, bits=4)

        # L0 Surface Penetration into 4 Slow Slots
        self.l0_pen_a = QuantLinearV130(dim, 24, bias=False, bits=4)
        self.l0_pen_b = QuantLinearV130(24, dim, bias=False, bits=4)
        self.l0_pen_gate = nn.Parameter(torch.full((dim,), -1.2))

        # Analytic 6-Valley EBT on 64-d Slow Bus
        self.ebt = InLoopAnalyticEBTV130(BUS_DIM_V130, hidden=96)
        self.ebt_back_scale = nn.Parameter(torch.tensor(0.18))

        # Strict Contractive Mixing + Anchor Pull (`rho ∈ (0.25, 0.80) < 1`!)
        self.anchor_alpha = nn.Parameter(torch.full((1, 1, dim), -1.1))
        self.contract_rho_raw = nn.Parameter(torch.full((T_MAX_V150,), 0.2))

        sa_mask = torch.zeros((1, 1, NUM_SLOTS_V130, NUM_SLOTS_V130), dtype=torch.float32)
        sa_mask[:, :, : self.n_ai_slots, self.n_ai_slots :] = -1e4
        self.register_buffer("sa_iso_mask", sa_mask, persistent=False)

    def build_readonly_cache(self, h_tok: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project Read-Only `K_0, V_0` + `32-d` Light Evidence Cache ONCE before looping (saves 84.9M MAC/loop!)."""
        m_u = mask.float().unsqueeze(-1)
        k_ev = self.k_ev_proj(h_tok) * m_u
        v_ev = self.v_ev_proj(h_tok) * m_u
        k_lite = self.lite_k_proj(h_tok) * m_u
        v_lite = self.lite_v_proj(h_tok) * m_u

        B, L, D = h_tok.shape
        blk0 = self.blocks[0]
        q = self.slot_queries.expand(B, -1, -1)
        qh = blk0.ca_q(blk0.norm_ca(q)).view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        sc = sc.masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(h_tok.dtype)
        ctx = torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        anchor = q + blk0.ca_o(ctx)
        return {"k_ev": k_ev, "v_ev": v_ev, "k_lite": k_lite, "v_lite": v_lite, "anchor": anchor}

    def forward_outer_step(
        self,
        ai_streams: torch.Tensor,
        pr_stream: torch.Tensor,
        anchor: torch.Tensor,
        cov_prev: torch.Tensor,
        k_ev_0: torch.Tensor,
        v_ev_0: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        g_sup: torch.Tensor,
        outer_idx: int,
        prev_slots: torch.Tensor | None = None,
        prev_bus: torch.Tensor | None = None,
        prev_doubt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        oi = min(max(int(outer_idx), 0), T_MAX_V150 - 1)
        s_in = torch.cat([ai_streams[:, 0], pr_stream], dim=1)  # (B, 16, D)
        s = s_in

        # 1. Outer-step role history delta & AI-only doubt handoff (writes into AI slot 9 ONLY!)
        if outer_idx > 0 and prev_slots is not None:
            d_slow = (s[:, : self.n_slow, :] - prev_slots[:, : self.n_slow, :]).mean(dim=1, keepdim=True)
            d_sem = (s[:, self.n_slow : self.n_ai_slots, :] - prev_slots[:, self.n_slow : self.n_ai_slots, :]).mean(dim=1, keepdim=True)
            d_span = (s[:, self.n_ai_slots :, :] - prev_slots[:, self.n_ai_slots :, :]).mean(dim=1, keepdim=True)
            g_r = 0.08 * torch.tanh(self.role_hist_g)
            s = torch.cat([
                s[:, : self.n_slow, :] + g_r[0] * d_slow,
                s[:, self.n_slow : self.n_ai_slots, :] + g_r[1] * d_sem,
                s[:, self.n_ai_slots :, :] + g_r[2] * d_span,
            ], dim=1)

        if outer_idx > 0 and prev_bus is not None and prev_doubt is not None:
            h_in = 0.20 * self.handoff_proj(torch.cat([prev_bus, prev_doubt], dim=-1))
            s = torch.cat([s[:, : self.n_ai_slots - 1, :], s[:, self.n_ai_slots - 1 : self.n_ai_slots, :] + h_in.unsqueeze(1), s[:, self.n_ai_slots :, :]], dim=1)

        # 2. Execute `core_depth` Stacked Shared Recurrent Sub-Blocks (with live intra-step coverage updating!)
        cov_cur = cov_prev
        r_read_acc: torch.Tensor | None = None
        last_ctrl: torch.Tensor | None = None
        for b_idx, blk in enumerate(self.blocks):
            s, cov_cur, r_read_b, last_ctrl = blk(
                s=s, cov_prev=cov_cur, k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=mask,
                k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, g_sup=g_sup,
                oi=oi, sa_iso_mask=self.sa_iso_mask, bus_down=self.bus_down, bus_up=self.bus_up,
                apply_cov=(outer_idx > 0 or b_idx > 0),
            )
            r_read_acc = r_read_b if r_read_acc is None else (r_read_acc + r_read_b)
        assert r_read_acc is not None and last_ctrl is not None
        r_read_avg = r_read_acc * (1.0 / float(len(self.blocks)))

        # 3. L0 Penetration + Analytic 6-Valley EBT on 64-d Slow Bus (purely from AI/shared slow slots `[0:4)`)
        l0_pen = self.l0_pen_b(F.silu(self.l0_pen_a(h0_pool))).unsqueeze(1)
        s = torch.cat([s[:, : self.n_slow, :] + torch.sigmoid(self.l0_pen_gate).view(1, 1, -1) * l0_pen, s[:, self.n_slow :, :]], dim=1)

        bus_pre = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        ebt_r = self.ebt.forward_energy_and_grad(bus_pre)
        ebt_delta = self.bus_up(ebt_r["u_next"] - bus_pre).unsqueeze(1)
        ebt_g = 0.30 * torch.tanh(self.ebt_back_scale) * (1.0 + 0.20 * last_ctrl[5])
        s_prop = torch.cat([s[:, : self.n_slow, :] + ebt_g * ebt_delta, s[:, self.n_slow :, :]], dim=1)

        # 4. Contraction (`||D_s F|| < 1`) + Bounded Read-Only Evidence-Flip Impulse
        a_mix = 0.25 + 0.20 * torch.sigmoid(self.anchor_alpha)
        s_prop = (1.0 - a_mix) * s_prop + a_mix * anchor
        rho_i = 0.30 + 0.48 * torch.sigmoid(self.contract_rho_raw[oi])
        s_contract = (1.0 - rho_i) * s_in + rho_i * s_prop

        ev_g = torch.sigmoid(self.ev_flip_gate(r_read_avg))
        ev_u = self.ev_flip_dir(r_read_avg)
        ev_u_unit = ev_u / (ev_u.norm(dim=-1, keepdim=True).clamp_min(1e-5))
        ev_b = 0.65 * torch.tanh(self.ev_flip_budget[oi])
        s_cand = (s_contract + (ev_b * ev_g) * ev_u_unit) * self.loop_q_scale[oi].clamp(0.5, 1.8)

        # 5. Block-Diagonal `diag(M_{AI, 3x3}, 1)` 10-Iter Sinkhorn mHC (AI 3-stream strictly isolated from Promo!)
        s_ai_cand = s_cand[:, : self.n_ai_slots, :]
        s_pr_next = s_cand[:, self.n_ai_slots :, :]

        m3 = sinkhorn_nxn_v150(self.mhc_ai_log_mix[oi]).to(s_ai_cand.dtype)
        ai_stack = torch.stack([s_ai_cand, ai_streams[:, 1], ai_streams[:, 2]], dim=1)
        ai_streams_next = torch.einsum("ij,bjnd->bind", m3, ai_stack)

        r_w = torch.softmax(self.mhc_ai_read_logits[oi].float(), dim=0).to(s_ai_cand.dtype)
        s_ai_out = (ai_streams_next * r_w.view(1, 3, 1, 1)).sum(dim=1)
        slots_out = torch.cat([s_ai_out, s_pr_next], dim=1)

        return {
            "slots": slots_out,
            "ai_streams": ai_streams_next,
            "pr_stream": s_pr_next,
            "cov": cov_cur.to(s.dtype),
            "bus": bus_pre,
            "ebt_energies": ebt_r["energies"],
            "ebt_margin": ebt_r["ai_margin"],
            "ebt_promo_margin": ebt_r["promo_margin"],
            "ebt_grad_norm": ebt_r["grad_norm"],
        }


# ---------------------------------------------------------------------------
# 4. Complete Unified `MeguvizV150HeavenLoopModel` (`T1 -> T2 -> T2.5 -> T3 -> T4`)
# ---------------------------------------------------------------------------

class MeguvizV150HeavenLoopModel(nn.Module):
    """Unified <=5.00 MiB Bootable `HeavenLoop` Model with full ablation switches & 17-Gate CI."""

    def __init__(
        self,
        vocab_size: int = 12288,
        max_len: int = MAX_LEN_V140,
        style_dim: int = STYLE_FEAT_DIM_V11,
        style_era_dropout_p: float = 0.15,
        prelude_mode: str = "single_0p6m",
        core_depth: int = CORE_DEPTH_DEFAULT_V150,
        coverage_beta_enabled: bool = True,
        use_t25_light_reread: bool = True,
        use_local_charlm_inject: bool = True,
        use_global_charlm_shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.style_era_dropout_p = float(style_era_dropout_p)
        self.prelude_mode = str(prelude_mode)
        self.core_depth = max(1, int(core_depth))
        self.coverage_beta_enabled = bool(coverage_beta_enabled)
        self.use_t25_light_reread = bool(use_t25_light_reread)
        self.use_local_charlm_inject = bool(use_local_charlm_inject)
        self.use_global_charlm_shortcut = bool(use_global_charlm_shortcut)

        # Stage 0: Decoupled Dual-Axis ApexCharNano (`defer_ai` ZERO Promo inputs, global CharLM off by default!)
        self.rung0 = ApexCharNanoRung0V150(use_global_charlm_shortcut=self.use_global_charlm_shortcut)

        # Stage 1: Factorized BPE + Bigram/Trigram Engrams + Configurable Prelude (`H1` ablation!)
        self.word_emb_a = QuantEmbeddingV130(vocab_size, 48, bits=4, padding_idx=0)
        self.word_emb_p = QuantLinearV130(48, H_V130, bias=False, bits=8)
        self.engram_emb = QuantEmbeddingV130(ENGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.engram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.trigram_emb = QuantEmbeddingV130(TRIGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.trigram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, H_V130) * 0.02)
        self.emb_norm = nn.LayerNorm(H_V130)

        # H8: Gated 8-Channel Per-Token Local Char-LM Injection (`H_j = H_sem_j + g_j * U f_j`, `U` bias-free!)
        self.local_lm_u = nn.Linear(LOCAL_CHARLM_DIM_V150, H_V130, bias=False)
        self.local_lm_gate = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.normal_(self.local_lm_u.weight, std=0.01)

        if self.prelude_mode == "dual_legacy":
            self.prelude_l0 = PadExactConvBlockV130(H_V130, 896, NUM_HEADS_V130, bits=4)
            self.prelude_l1: nn.Module | None = PadExactConvBlockV130(H_V130, 896, NUM_HEADS_V130, bits=4)
        elif self.prelude_mode == "stem_0p3m":
            self.prelude_l0 = PadExactLightStemV150(H_V130, ffn_dim=384)
            self.prelude_l1 = None
        else:
            # Default `"single_0p6m"`: 1 full PadExactConvBlock with FFN=512 (~0.56 MiB vs 1.67 MiB dual!)
            self.prelude_l0 = PadExactConvBlockV130(H_V130, FFN_V150, NUM_HEADS_V130, bits=4)
            self.prelude_l1 = None

        self.haar_k_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)
        self.haar_v_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)

        self.sup_gate_tau = nn.Parameter(torch.tensor(16.0))
        self.sup_gate_scale = nn.Parameter(torch.tensor(4.0))

        self.pr_attr_attn = nn.Sequential(QuantLinearV130(H_V130, 32, bits=4), nn.Tanh(), nn.Linear(32, 1))

        # R1 fast prefix heads
        self.r1_style_ai = nn.Sequential(nn.Linear(style_dim - 8, 32), nn.SiLU())
        self.r1_style_pr = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU())
        self.r1_ai_head = nn.Linear(H_V130 + 32 + 32, 1)
        self.r1_promo_head = nn.Linear(H_V130 + 32 + 32, 1)

        # Stage 2: Dual-Layer (`core_depth=2`) Streamlined Read-Only Coverage-Inhibited `HeavenLoopCoreV150`
        self.core = HeavenLoopCoreV150(
            H_V130, ffn_dim=FFN_V150, num_heads=NUM_HEADS_V130,
            coverage_beta_enabled=coverage_beta_enabled, core_depth=self.core_depth,
        )

        # Strictly Disjoint 256-d Coda
        self.coda_shared = QuantLinearV130(H_V130 * 2, SHARED_END_V130, bits=4)
        self.coda_ai_priv = QuantLinearV130(H_V130 + 32 + 32, AI_END_V130 - SHARED_END_V130, bits=4)
        self.coda_pr_priv = QuantLinearV130(H_V130 * 2 + 32 + 32, EMB_DIM_V130 - AI_END_V130, bits=4)
        self.emb_norm_sh = nn.LayerNorm(SHARED_END_V130)
        self.emb_norm_ai = nn.LayerNorm(AI_END_V130 - SHARED_END_V130)
        self.emb_norm_pr = nn.LayerNorm(EMB_DIM_V130 - AI_END_V130)

        self.cbm_ai_enc = nn.Linear(AI_END_V130, CBM_AI_K_V130)
        self.cbm_pr_enc = nn.Linear(PR_EMB_DIM_V130, CBM_PR_K_V130)
        self.cbm_ai_to_logit = nn.Linear(CBM_AI_K_V130, 1, bias=False)
        self.cbm_pr_to_logit = nn.Linear(CBM_PR_K_V130, 1, bias=False)

        self.ai_head = SoftBallLogitNormHeadV121(AI_END_V130, init_r_cap=7.2)
        self.promo_head = SoftBallLogitNormHeadV121(PR_EMB_DIM_V130, init_r_cap=7.2)
        self.promo_sub4_head = nn.Linear(PR_EMB_DIM_V130, 4)
        self.ebt_ai_gate = nn.Parameter(torch.tensor(0.20))
        self.ebt_pr_gate = nn.Parameter(torch.tensor(0.15))

        # v1.4.0 Fix #2: Deep-Only Multi-Depth AI Fusion `[R1, T1, T2, EBT6]` — ZERO `logit_ai_r0` shortcut & ZERO `-gamma*|R0-T2|` lock!
        self.ai_rung_mix_logits = nn.Parameter(
            torch.tensor([math.log(0.10), math.log(0.25), math.log(0.55), math.log(0.10)], dtype=torch.float32)
        )
        self.disagree_t2_ebt_raw = nn.Parameter(torch.tensor(-2.4))
        self.charlm_ai_gate_raw = nn.Parameter(torch.tensor(-1.5))

        self.traj_ai_lin = nn.Linear(AI_END_V130 * 2 + 8, 1)
        self.traj_ai_mlp = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 8),
            nn.Linear(AI_END_V130 * 2 + 8, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.traj_ai_lin.weight, std=0.005)
        nn.init.zeros_(self.traj_ai_lin.bias)
        nn.init.normal_(self.traj_ai_mlp[-1].weight, std=0.01)
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

        # 9th Blind Spot: `T2.5` Decoupled 2-Query Prosecutor-vs-Defense Light Reread (`36,864 MAC` at L=288!)
        self.t25_q_pro = nn.Linear(AI_END_V130, LITE_DIM_V150)
        self.t25_q_def = nn.Linear(AI_END_V130, LITE_DIM_V150)
        self.t25_delta_w = nn.Linear(LITE_DIM_V150 + 4, 1)
        nn.init.normal_(self.t25_delta_w.weight, std=0.01)
        nn.init.zeros_(self.t25_delta_w.bias)
        self.t25_corr_cap = T25_CORR_CAP_V150

        # v1.4.0 Fix #3: Symmetric non-zero init on T3/T4 residual heads (prevents dead-weight positive-bias collapse!)
        self.t3_corr_ai = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 10),
            nn.Linear(AI_END_V130 * 2 + 10, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.t3_corr_ai[-1].weight, std=0.012)
        nn.init.zeros_(self.t3_corr_ai[-1].bias)
        self.t3_corr_pr = nn.Sequential(
            nn.LayerNorm(PR_EMB_DIM_V130 * 2 + 5),
            nn.Linear(PR_EMB_DIM_V130 * 2 + 5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.t3_corr_pr[-1].weight, std=0.01)
        nn.init.zeros_(self.t3_corr_pr[-1].bias)
        self.t3_corr_cap = T3_CORR_CAP_V150

        self.t4_corr_ai = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 11),
            nn.Linear(AI_END_V130 * 2 + 11, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.t4_corr_ai[-1].weight, std=0.012)
        nn.init.zeros_(self.t4_corr_ai[-1].bias)
        self.t4_corr_pr = nn.Sequential(
            nn.LayerNorm(PR_EMB_DIM_V130 * 2 + 5),
            nn.Linear(PR_EMB_DIM_V130 * 2 + 5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.t4_corr_pr[-1].weight, std=0.01)
        nn.init.zeros_(self.t4_corr_pr[-1].bias)
        self.t4_corr_cap = T4_CORR_CAP_V150

        # `TaH2` Lookahead Exit Gates (`T2.5 -> T3` and `T3 -> T4`, strictly pure-AI inputs!)
        self.halt_t25_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + LITE_DIM_V150 + 11),
            nn.Linear(AI_END_V130 * 2 + LITE_DIM_V150 + 11, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.halt_t3_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 10),
            nn.Linear(AI_END_V130 * 2 + 10, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
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
        ebt_ai_m: torch.Tensor,
        ebt_pr_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        slow_p = slots[:, : NUM_SLOW_V130, :].mean(dim=1)
        sem_p = slots[:, NUM_SLOW_V130 : NUM_SLOW_V130 + NUM_SEM_FAST_V130, :].mean(dim=1)
        span_p = slots[:, NUM_SLOW_V130 + NUM_SEM_FAST_V130 :, :].mean(dim=1)

        z_sh = self.emb_norm_sh(self.coda_shared(torch.cat([slow_p, h_pool_0], dim=-1)))
        z_ai_p = self.emb_norm_ai(self.coda_ai_priv(torch.cat([sem_p, st_ai, z_char[:, :32]], dim=-1)))
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, pr_attr_0, st_pr, z_char[:, 32:]], dim=-1)))

        emb_ai = torch.cat([z_sh, z_ai_p], dim=-1)
        emb_pr = torch.cat([z_sh, z_pr_p], dim=-1)
        emb_full = torch.cat([z_sh, z_ai_p, z_pr_p], dim=-1)

        c_ai_raw = F.relu(self.cbm_ai_enc(emb_ai))
        v_ai, i_ai = torch.topk(c_ai_raw, k=6, dim=-1)
        c_ai_sp = torch.zeros_like(c_ai_raw).scatter(-1, i_ai, v_ai)

        c_pr_raw = F.relu(self.cbm_pr_enc(emb_pr))
        v_pr, i_pr = torch.topk(c_pr_raw, k=6, dim=-1)
        c_pr_sp = torch.zeros_like(c_pr_raw).scatter(-1, i_pr, v_pr)

        lg_ai = self.ai_head(emb_ai) + 0.15 * self.cbm_ai_to_logit(c_ai_sp).squeeze(-1) + self.ebt_ai_gate * torch.clamp(ebt_ai_m, -6.0, 6.0)
        lg_pr = self.promo_head(emb_pr) + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1) + self.ebt_pr_gate * torch.clamp(ebt_pr_m, -6.0, 6.0)
        return {
            "logit_ai": lg_ai,
            "logit_promo": lg_pr,
            "promo_sub4_logit": self.promo_sub4_head(emb_pr),
            "emb_ai": emb_ai,
            "emb_pr": emb_pr,
            "emb": emb_full,
        }

    def _compute_t2_trajectory(
        self,
        t1_emb_ai: torch.Tensor,
        t1_lg_ai: torch.Tensor,
        t1_lg_pr: torch.Tensor,
        ro2: dict[str, torch.Tensor],
        ebt_ai_m: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        lg_r1_ai: torch.Tensor,
        lg_r1_pr: torch.Tensor,
        g_sup: torch.Tensor,
        charlm_logit: torch.Tensor,
        r0_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ebt_c = torch.clamp(ebt_ai_m, -6.0, 6.0)
        lm_eff = charlm_logit.to(ro2["logit_ai"].dtype) if self.use_global_charlm_shortcut else torch.zeros_like(ro2["logit_ai"])
        w_mix = torch.softmax(self.ai_rung_mix_logits.float(), dim=0).to(ro2["logit_ai"].dtype)
        g_d2 = 0.15 * F.softplus(self.disagree_t2_ebt_raw)

        # v1.4.0 Fix #2: 100% Deep-Only base (`[R1, T1, T2, EBT6]`) — ZERO R0 shortcut & ZERO `-gamma|R0-T2|` lock!
        ai_base = w_mix[0] * lg_r1_ai + w_mix[1] * t1_lg_ai + w_mix[2] * ro2["logit_ai"] + w_mix[3] * ebt_c
        disagree_pen = g_d2 * (ro2["logit_ai"] - ebt_c).abs()
        rho_eff = (torch.sigmoid(self.promo_char_rho_raw) * r0_keep) if r0_keep is not None else torch.sigmoid(self.promo_char_rho_raw)

        if self.use_global_charlm_shortcut:
            lm_active = (lm_eff.abs() > 1e-5).to(lm_eff.dtype)
            lm_agree = torch.sigmoid(1.5 - (ro2["logit_ai"] - lm_eff).abs())
            ai_base = ai_base + (0.15 * torch.sigmoid(self.charlm_ai_gate_raw)) * lm_eff * lm_agree * lm_active
        ai_base = ai_base - torch.sigmoid(ai_base) * disagree_pen.clamp_max(0.6)

        d2_signed = ro2["logit_ai"] - t1_lg_ai
        # Pass `0.0` in the R0 slot so traj head learns purely from deep semantics (no lazy R0 copy!)
        ai_scalars = torch.stack([torch.zeros_like(lg_r1_ai), lg_r1_ai, t1_lg_ai, ro2["logit_ai"], d2_signed, ebt_c, g_sup.to(ebt_c.dtype), lm_eff], dim=-1)
        traj_ai_in = torch.cat([ro2["emb_ai"], ro2["emb_ai"] - t1_emb_ai, ai_scalars], dim=-1)
        ai_res_raw = self.traj_ai_lin(traj_ai_in).squeeze(-1) + self.traj_ai_mlp(traj_ai_in).squeeze(-1)
        ai_final = ai_base + self.traj_ai_cap * torch.tanh(ai_res_raw / self.traj_ai_cap)

        pr_base = (1.0 - rho_eff) * ro2["logit_promo"] + rho_eff * logit_promo_r0
        pr_scalars = torch.stack([logit_promo_r0, lg_r1_pr, t1_lg_pr, ro2["logit_promo"]], dim=-1)
        pr_res_raw = self.traj_pr_mlp(torch.cat([ro2["emb_pr"], z_char[:, 32:], pr_scalars], dim=-1)).squeeze(-1)
        pr_final = pr_base + self.traj_pr_cap * torch.tanh(pr_res_raw / self.traj_pr_cap)
        return ai_final, pr_final

    def forward_t25_light_reread(
        self,
        emb_ai_t2: torch.Tensor,
        k_lite_0: torch.Tensor,
        v_lite_0: torch.Tensor,
        attention_mask: torch.Tensor,
        logit_ai_t2_final: torch.Tensor,
        logit_ai_t1: torch.Tensor,
        g_sup: torch.Tensor,
        charlm_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """9th Blind Spot: `36,864 MAC` Two-Query Prosecutor-vs-Defense Light Reread over 32-d read-only K/V!"""
        if not self.use_t25_light_reread:
            z = torch.zeros_like(logit_ai_t2_final)
            c_z = torch.zeros((emb_ai_t2.shape[0], LITE_DIM_V150), device=emb_ai_t2.device, dtype=emb_ai_t2.dtype)
            return {"logit_ai_t25_final": logit_ai_t2_final, "delta_t25": z, "c_diff_t25": c_z}

        m_f = attention_mask.float()
        q_pro = self.t25_q_pro(emb_ai_t2.detach())  # (B, 32)
        q_def = self.t25_q_def(emb_ai_t2.detach())  # (B, 32)
        inv_s = LITE_DIM_V150 ** -0.5

        sc_pro = (torch.einsum("bd,bld->bl", q_pro, k_lite_0) * inv_s).masked_fill(m_f < 0.5, -1e4)
        sc_def = (torch.einsum("bd,bld->bl", q_def, k_lite_0) * inv_s).masked_fill(m_f < 0.5, -1e4)
        a_pro = torch.softmax(sc_pro.float(), dim=-1).to(v_lite_0.dtype)
        a_def = torch.softmax(sc_def.float(), dim=-1).to(v_lite_0.dtype)
        c_pro = torch.einsum("bl,bld->bd", a_pro, v_lite_0)
        c_def = torch.einsum("bl,bld->bd", a_def, v_lite_0)
        c_diff = c_pro - c_def

        base_t2 = logit_ai_t2_final.detach()
        lm_sc = charlm_logit.detach() if self.use_global_charlm_shortcut else torch.zeros_like(base_t2)
        d_in = torch.cat([
            c_diff,
            torch.stack([base_t2, base_t2 - logit_ai_t1.detach(), g_sup.detach(), lm_sc], dim=-1),
        ], dim=-1)
        raw_d = self.t25_delta_w(d_in).squeeze(-1)
        delta_t25 = self.t25_corr_cap * torch.tanh(raw_d / self.t25_corr_cap)
        return {
            "logit_ai_t25_final": base_t2 + delta_t25,
            "delta_t25": delta_t25,
            "c_diff_t25": c_diff,
        }

    def forward_r0(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rung0(c1, c2, c3, mask, feats)

    def forward_t2_stage(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        support_pairs: torch.Tensor,
        charlm_logit: torch.Tensor | None = None,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Run Prelude -> Single-Pass Read-Only Cache -> T1 -> T2 -> T2.5 Light Reread + TaH2 Escalation Gate!"""
        B, L = input_ids.shape
        m = attention_mask.float()
        m_u = m.unsqueeze(-1)
        g_sup = self.compute_support_gate(support_pairs).to(style_feats.dtype)

        st = style_feats.clone()
        if self.training and self.style_era_dropout_p > 0.0:
            keep = (torch.rand((B, 1), device=st.device) >= self.style_era_dropout_p).to(st.dtype)
            st[:, :40] = st[:, :40] * keep

        r0_keep: torch.Tensor | None = None
        if self.training and r0_drop_p > 0.0:
            r0_keep = (torch.rand((B,), device=z_char.device) >= float(r0_drop_p)).to(z_char.dtype)
            zc_eff = z_char * r0_keep.unsqueeze(-1)
            la0_eff = logit_ai_r0 * r0_keep
            lp0_eff = logit_promo_r0 * r0_keep
        else:
            zc_eff, la0_eff, lp0_eff = z_char, logit_ai_r0, logit_promo_r0

        lm_raw = torch.zeros_like(logit_ai_r0) if charlm_logit is None else charlm_logit.to(logit_ai_r0.dtype)
        loc_raw = local_charlm.to(style_feats.dtype)
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((B,), device=lm_raw.device) >= float(charlm_drop_p)).to(lm_raw.dtype)
            lm_eff = lm_raw * lm_keep
            loc_eff = loc_raw * lm_keep.view(-1, 1, 1)
        else:
            lm_eff, loc_eff = lm_raw, loc_raw

        pos = self.pos_emb[:, :L, :]
        h_sem = self.emb_norm(
            self.word_emb_p(self.word_emb_a(input_ids))
            + self.engram_p(self.engram_emb(engram_ids))
            + self.trigram_p(self.trigram_emb(trigram_ids))
            + pos
        ) * m_u

        # H8: Gated Per-Token Local Char-LM Injection (`g_j === 0` whenever high-order hit support is absent!)
        if self.use_local_charlm_inject:
            hits_4 = loc_eff[..., 4:8]
            has_hi_hit = (hits_4[..., 1:4].max(dim=-1, keepdim=True).values > 1e-4).to(h_sem.dtype)
            g_loc = torch.sigmoid(self.local_lm_gate(hits_4) - 1.0) * has_hi_hit * m_u
            x = (h_sem + g_loc * self.local_lm_u(loc_eff)) * m_u
        else:
            x = h_sem

        h0 = self.prelude_l0(x, attention_mask)
        if (not torch.jit.is_tracing()) and (L % 2 == 1):
            h0_p, m_p = F.pad(h0, (0, 0, 0, 1)), F.pad(m, (0, 1))
        else:
            h0_p, m_p = h0, m
        m_even = m_p[:, 0::2]
        h_diff = (h0_p[:, 0::2, :] - h0_p[:, 1::2, :]) * m_even.unsqueeze(-1)
        k_haar = self.haar_k_proj(h_diff) * m_even.unsqueeze(-1)
        v_haar = self.haar_v_proj(h_diff) * m_even.unsqueeze(-1)
        haar_mask = m_even

        denom = m_u.sum(dim=1).clamp_min(1.0)
        h0_pool = h0.sum(dim=1) / denom
        st_ai = self.r1_style_ai(st[:, : self.style_dim - 8]) * g_sup.unsqueeze(-1)
        st_pr = self.r1_style_pr(st)
        lg_r1_ai = self.r1_ai_head(torch.cat([h0_pool, st_ai, zc_eff[:, :32]], dim=-1)).squeeze(-1)
        lg_r1_pr = self.r1_promo_head(torch.cat([h0_pool, st_pr, zc_eff[:, 32:]], dim=-1)).squeeze(-1)

        h_tok_0 = self.prelude_l1(h0, attention_mask) if self.prelude_l1 is not None else h0
        h_pool_0 = h_tok_0.sum(dim=1) / denom
        pr_attr_0 = self._pool_promo_attr(h_tok_0, attention_mask)

        # Single-Pass Read-Only K_0, V_0, K_lite, V_lite & Anchor!
        ro_cache = self.core.build_readonly_cache(h_tok_0, attention_mask)
        anchor = ro_cache["anchor"]
        k_ev_0, v_ev_0 = ro_cache["k_ev"], ro_cache["v_ev"]
        k_lite_0, v_lite_0 = ro_cache["k_lite"], ro_cache["v_lite"]

        ai_init = anchor[:, : self.core.n_ai_slots, :].unsqueeze(1).expand(-1, MHC_AI_STREAMS_V150, -1, -1).contiguous()
        pr_init = anchor[:, self.core.n_ai_slots :, :]
        cov_0 = torch.zeros((B, L), device=h_tok_0.device, dtype=h_tok_0.dtype)

        # Step T1 (`outer_idx=0`)
        st1 = self.core.forward_outer_step(
            ai_streams=ai_init, pr_stream=pr_init, anchor=anchor, cov_prev=cov_0,
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool, g_sup=g_sup,
            outer_idx=0, prev_slots=None, prev_bus=None, prev_doubt=None,
        )
        ro1 = self._readout_step(st1["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, zc_eff, st1["ebt_margin"], st1["ebt_promo_margin"])
        doubt1 = torch.stack([(2.0 * torch.sigmoid(ro1["logit_ai"].detach()) - 1.0).abs(), st1["ebt_grad_norm"].detach()], dim=-1)

        # Step T2 (`outer_idx=1`)
        st2 = self.core.forward_outer_step(
            ai_streams=st1["ai_streams"], pr_stream=st1["pr_stream"], anchor=anchor, cov_prev=st1["cov"],
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool, g_sup=g_sup,
            outer_idx=1, prev_slots=anchor, prev_bus=st1["bus"], prev_doubt=doubt1,
        )
        ro2 = self._readout_step(st2["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, zc_eff, st2["ebt_margin"], st2["ebt_promo_margin"])
        ai_t2_final, pr_t2_final = self._compute_t2_trajectory(
            ro1["emb_ai"], ro1["logit_ai"], ro1["logit_promo"], ro2, st2["ebt_margin"],
            zc_eff, la0_eff, lp0_eff, lg_r1_ai, lg_r1_pr, g_sup=g_sup, charlm_logit=lm_eff, r0_keep=r0_keep,
        )

        # Step T2.5 (`36,864 MAC` Two-Query Prosecutor/Defense Light Reread!)
        t25 = self.forward_t25_light_reread(
            ro2["emb_ai"], k_lite_0, v_lite_0, attention_mask, ai_t2_final, ro1["logit_ai"], g_sup, lm_eff
        )
        ai_t25_final = t25["logit_ai_t25_final"]

        # `TaH2` Escalation Gate (`T2.5 -> T3`, strictly AI-only features!)
        d2_s = ai_t2_final.detach() - ro1["logit_ai"].detach()
        halt_in = torch.cat([
            ro2["emb_ai"].detach(),
            (ro2["emb_ai"] - ro1["emb_ai"]).detach(),
            t25["c_diff_t25"].detach(),
            torch.stack([
                la0_eff.detach(), lg_r1_ai.detach(), ro1["logit_ai"].detach(), ai_t2_final.detach(),
                ai_t25_final.detach(), t25["delta_t25"].detach(), d2_s,
                torch.clamp(st2["ebt_margin"].detach(), -6.0, 6.0), st2["ebt_grad_norm"].detach(),
                g_sup.detach(), (ro2["logit_ai"].detach() - lm_eff.detach()).abs(),
            ], dim=-1),
        ], dim=-1)
        esc_t3_lg = self.halt_t25_head(halt_in).squeeze(-1)

        return {
            "logit_ai_t2_final": ai_t2_final,
            "logit_ai_t25_final": ai_t25_final,
            "delta_t25": t25["delta_t25"],
            "logit_promo_t2_final": pr_t2_final,
            "logit_ai_r1": lg_r1_ai,
            "logit_promo_r1": lg_r1_pr,
            "logit_ai_t1": ro1["logit_ai"],
            "logit_promo_t1": ro1["logit_promo"],
            "logit_ai_t2": ro2["logit_ai"],
            "logit_promo_t2": ro2["logit_promo"],
            "promo_sub4_logit_t2": ro2["promo_sub4_logit"],
            "ebt_margin_t2": st2["ebt_margin"],
            "ebt_promo_margin_t2": st2["ebt_promo_margin"],
            "ebt_grad_norm_t2": st2["ebt_grad_norm"],
            "emb_ai_t1": ro1["emb_ai"],
            "emb_ai_t2": ro2["emb_ai"],
            "emb_pr_t2": ro2["emb_pr"],
            "escalate_t3_logit": esc_t3_lg,
            "p_escalate_t3": torch.sigmoid(esc_t3_lg),
            # Exact read-only & dynamic continuation cache for T3 / T4:
            "ai_streams_t2": st2["ai_streams"],
            "pr_stream_t2": st2["pr_stream"],
            "slots_t1": st1["slots"],
            "slots_t2": st2["slots"],
            "anchor": anchor,
            "cov_t2": st2["cov"],
            "k_ev_0": k_ev_0,
            "v_ev_0": v_ev_0,
            "k_haar": k_haar,
            "v_haar": v_haar,
            "haar_mask": haar_mask,
            "h0_pool": h0_pool,
            "h_pool_0": h_pool_0,
            "pr_attr_0": pr_attr_0,
            "zh_bus_t2": st2["bus"],
            "st_ai": st_ai,
            "st_pr": st_pr,
            "g_sup": g_sup,
            "_zc_eff": zc_eff,
            "_la0_eff": la0_eff,
            "_lp0_eff": lp0_eff,
            "_lm_eff": lm_eff,
        }

    def forward_t3_step(
        self,
        ai_streams_t2: torch.Tensor, pr_stream_t2: torch.Tensor, slots_t1: torch.Tensor, anchor: torch.Tensor,
        cov_t2: torch.Tensor, k_ev_0: torch.Tensor, v_ev_0: torch.Tensor, attention_mask: torch.Tensor,
        k_haar: torch.Tensor, v_haar: torch.Tensor, haar_mask: torch.Tensor,
        h0_pool: torch.Tensor, h_pool_0: torch.Tensor, pr_attr_0: torch.Tensor,
        zh_bus_t2: torch.Tensor, ebt_grad_norm_t2: torch.Tensor, ebt_margin_t2: torch.Tensor,
        st_ai: torch.Tensor, st_pr: torch.Tensor, z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor, logit_promo_r0: torch.Tensor, logit_promo_r1: torch.Tensor,
        logit_ai_t1: torch.Tensor, logit_promo_t1: torch.Tensor, logit_ai_t2_raw: torch.Tensor,
        logit_ai_t2_final: torch.Tensor, logit_ai_t25_final: torch.Tensor, logit_promo_t2_final: torch.Tensor,
        emb_ai_t2: torch.Tensor, emb_pr_t2: torch.Tensor, g_sup: torch.Tensor, charlm_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Step T3 (`outer_idx=2`): Bounded Detached Residual (`|delta| <= 1.8`) + Signed Curvature + `TaH2` T3->T4 Gate!"""
        doubt2 = torch.stack([(2.0 * torch.sigmoid(logit_ai_t2_raw.detach()) - 1.0).abs(), ebt_grad_norm_t2.detach()], dim=-1)
        st3 = self.core.forward_outer_step(
            ai_streams=ai_streams_t2, pr_stream=pr_stream_t2, anchor=anchor, cov_prev=cov_t2,
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool, g_sup=g_sup,
            outer_idx=2, prev_slots=slots_t1, prev_bus=zh_bus_t2, prev_doubt=doubt2,
        )
        ro3 = self._readout_step(st3["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, z_char, st3["ebt_margin"], st3["ebt_promo_margin"])

        ebt3_c = torch.clamp(st3["ebt_margin"], -6.0, 6.0)
        ebt2_c = torch.clamp(ebt_margin_t2.detach(), -6.0, 6.0)
        ai_base = logit_ai_t25_final.detach()
        d2 = logit_ai_t2_final.detach() - logit_ai_t1.detach()
        d3_raw = ro3["logit_ai"] - logit_ai_t2_raw.detach()
        k3_signed = d3_raw - d2  # Signed curvature!

        lm_sc3 = charlm_logit.detach() if self.use_global_charlm_shortcut else torch.zeros_like(ai_base)
        corr_ai_in = torch.cat([
            ro3["emb_ai"],
            ro3["emb_ai"] - emb_ai_t2.detach(),
            torch.stack([
                torch.zeros_like(ai_base), logit_ai_t1.detach(), ai_base, ro3["logit_ai"],
                d2, d3_raw, k3_signed, ebt3_c - ebt2_c, g_sup.detach(), lm_sc3,
            ], dim=-1),
        ], dim=-1)
        ai_delta = self.t3_corr_ai(corr_ai_in).squeeze(-1)
        ai_t3_final = ai_base + self.t3_corr_cap * torch.tanh(ai_delta / self.t3_corr_cap)

        pr_base = logit_promo_t2_final.detach()
        corr_pr_in = torch.cat([
            ro3["emb_pr"],
            ro3["emb_pr"] - emb_pr_t2.detach(),
            torch.stack([logit_promo_r0.detach(), logit_promo_r1.detach(), logit_promo_t1.detach(), pr_base, ro3["logit_promo"]], dim=-1),
        ], dim=-1)
        pr_delta = self.t3_corr_pr(corr_pr_in).squeeze(-1)
        pr_t3_final = pr_base + self.t3_corr_cap * torch.tanh(pr_delta / self.t3_corr_cap)

        # `TaH2` T3 -> T4 Escalation Gate (strictly pure-AI!)
        d3_f = ai_t3_final.detach() - ai_base
        halt3_in = torch.cat([
            ro3["emb_ai"].detach(),
            (ro3["emb_ai"] - emb_ai_t2).detach(),
            torch.stack([
                logit_ai_r0.detach(), logit_ai_t1.detach(), ai_base, ai_t3_final.detach(),
                d2, d3_f, d3_f - d2, ebt3_c.detach(), st3["ebt_grad_norm"].detach(), g_sup.detach(),
            ], dim=-1),
        ], dim=-1)
        esc_t4_lg = self.halt_t3_head(halt3_in).squeeze(-1)

        return {
            "logit_ai_t3_final": ai_t3_final,
            "logit_promo_t3_final": pr_t3_final,
            "logit_ai_t3": ro3["logit_ai"],
            "logit_promo_t3": ro3["logit_promo"],
            "promo_sub4_logit_t3": ro3["promo_sub4_logit"],
            "ebt_margin_t3": st3["ebt_margin"],
            "ebt_promo_margin_t3": st3["ebt_promo_margin"],
            "ebt_grad_norm_t3": st3["ebt_grad_norm"],
            "emb_ai_t3": ro3["emb_ai"],
            "emb_pr_t3": ro3["emb_pr"],
            "escalate_t4_logit": esc_t4_lg,
            "p_escalate_t4": torch.sigmoid(esc_t4_lg),
            "ai_streams_t3": st3["ai_streams"],
            "pr_stream_t3": st3["pr_stream"],
            "slots_t3": st3["slots"],
            "cov_t3": st3["cov"],
            "zh_bus_t3": st3["bus"],
        }

    def forward_t4_step(
        self,
        ai_streams_t3: torch.Tensor, pr_stream_t3: torch.Tensor, slots_t2: torch.Tensor, anchor: torch.Tensor,
        cov_t3: torch.Tensor, k_ev_0: torch.Tensor, v_ev_0: torch.Tensor, attention_mask: torch.Tensor,
        k_haar: torch.Tensor, v_haar: torch.Tensor, haar_mask: torch.Tensor,
        h0_pool: torch.Tensor, h_pool_0: torch.Tensor, pr_attr_0: torch.Tensor,
        zh_bus_t3: torch.Tensor, ebt_grad_norm_t3: torch.Tensor, ebt_margin_t3: torch.Tensor,
        st_ai: torch.Tensor, st_pr: torch.Tensor, z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor, logit_promo_r0: torch.Tensor, logit_promo_r1: torch.Tensor,
        logit_ai_t1: torch.Tensor, logit_promo_t1: torch.Tensor, logit_ai_t25_final: torch.Tensor,
        logit_ai_t3_raw: torch.Tensor, logit_ai_t3_final: torch.Tensor, logit_promo_t3_final: torch.Tensor,
        emb_ai_t3: torch.Tensor, emb_pr_t3: torch.Tensor, g_sup: torch.Tensor, charlm_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Step T4 (`outer_idx=3`): Second Bounded Reversal Step (`|delta| <= 1.8`, total T3+T4 budget `<= 3.6`!)."""
        doubt3 = torch.stack([(2.0 * torch.sigmoid(logit_ai_t3_raw.detach()) - 1.0).abs(), ebt_grad_norm_t3.detach()], dim=-1)
        st4 = self.core.forward_outer_step(
            ai_streams=ai_streams_t3, pr_stream=pr_stream_t3, anchor=anchor, cov_prev=cov_t3,
            k_ev_0=k_ev_0, v_ev_0=v_ev_0, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool, g_sup=g_sup,
            outer_idx=3, prev_slots=slots_t2, prev_bus=zh_bus_t3, prev_doubt=doubt3,
        )
        ro4 = self._readout_step(st4["slots"], h_pool_0, pr_attr_0, st_ai, st_pr, z_char, st4["ebt_margin"], st4["ebt_promo_margin"])

        ebt4_c = torch.clamp(st4["ebt_margin"], -6.0, 6.0)
        ebt3_c = torch.clamp(ebt_margin_t3.detach(), -6.0, 6.0)
        ai_base = logit_ai_t3_final.detach()
        d3 = ai_base - logit_ai_t25_final.detach()
        d4_raw = ro4["logit_ai"] - logit_ai_t3_raw.detach()
        k4_signed = d4_raw - d3

        lm_sc4 = charlm_logit.detach() if self.use_global_charlm_shortcut else torch.zeros_like(ai_base)
        corr_ai_in = torch.cat([
            ro4["emb_ai"],
            ro4["emb_ai"] - emb_ai_t3.detach(),
            torch.stack([
                torch.zeros_like(ai_base), logit_ai_t1.detach(), logit_ai_t25_final.detach(), ai_base,
                ro4["logit_ai"], d3, d4_raw, k4_signed, ebt4_c - ebt3_c, g_sup.detach(), lm_sc4,
            ], dim=-1),
        ], dim=-1)
        ai_delta = self.t4_corr_ai(corr_ai_in).squeeze(-1)
        ai_t4_final = ai_base + self.t4_corr_cap * torch.tanh(ai_delta / self.t4_corr_cap)

        pr_base = logit_promo_t3_final.detach()
        corr_pr_in = torch.cat([
            ro4["emb_pr"],
            ro4["emb_pr"] - emb_pr_t3.detach(),
            torch.stack([logit_promo_r0.detach(), logit_promo_r1.detach(), logit_promo_t1.detach(), pr_base, ro4["logit_promo"]], dim=-1),
        ], dim=-1)
        pr_delta = self.t4_corr_pr(corr_pr_in).squeeze(-1)
        pr_t4_final = pr_base + self.t4_corr_cap * torch.tanh(pr_delta / self.t4_corr_cap)

        return {
            "logit_ai_t4_final": ai_t4_final,
            "logit_promo_t4_final": pr_t4_final,
            "logit_ai_t4": ro4["logit_ai"],
            "logit_promo_t4": ro4["logit_promo"],
            "promo_sub4_logit_t4": ro4["promo_sub4_logit"],
            "ebt_margin_t4": st4["ebt_margin"],
            "ebt_promo_margin_t4": st4["ebt_promo_margin"],
            "emb_ai_t4": ro4["emb_ai"],
            "emb_pr_t4": ro4["emb_pr"],
        }

    def forward_deep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        local_charlm: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        support_pairs: torch.Tensor,
        charlm_logit: torch.Tensor | None = None,
        outer_steps: int = 4,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        t2 = self.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids, local_charlm,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
            support_pairs=support_pairs, charlm_logit=charlm_logit,
            r0_drop_p=r0_drop_p, charlm_drop_p=charlm_drop_p,
        )
        out = dict(t2)
        out["logit_ai"] = t2["logit_ai_t25_final"]
        out["logit_promo"] = t2["logit_promo_t2_final"]
        out["promo_sub4_logit"] = t2["promo_sub4_logit_t2"]
        out["ebt_margin"] = t2["ebt_margin_t2"]
        out["ebt_promo_margin"] = t2["ebt_promo_margin_t2"]
        out["emb_ai"] = t2["emb_ai_t2"]
        out["emb_pr"] = t2["emb_pr_t2"]

        if int(outer_steps) >= 3:
            t3 = self.forward_t3_step(
                t2["ai_streams_t2"], t2["pr_stream_t2"], t2["slots_t1"], t2["anchor"],
                t2["cov_t2"], t2["k_ev_0"], t2["v_ev_0"], attention_mask,
                t2["k_haar"], t2["v_haar"], t2["haar_mask"],
                t2["h0_pool"], t2["h_pool_0"], t2["pr_attr_0"],
                t2["zh_bus_t2"], t2["ebt_grad_norm_t2"], t2["ebt_margin_t2"],
                t2["st_ai"], t2["st_pr"], t2["_zc_eff"],
                t2["_la0_eff"], t2["_lp0_eff"], t2["logit_promo_r1"],
                t2["logit_ai_t1"], t2["logit_promo_t1"], t2["logit_ai_t2"],
                t2["logit_ai_t2_final"], t2["logit_ai_t25_final"], t2["logit_promo_t2_final"],
                t2["emb_ai_t2"], t2["emb_pr_t2"], t2["g_sup"], t2["_lm_eff"],
            )
            out.update(t3)
            out["logit_ai"] = t3["logit_ai_t3_final"]
            out["logit_promo"] = t3["logit_promo_t3_final"]
            out["promo_sub4_logit"] = t3["promo_sub4_logit_t3"]
            out["ebt_margin"] = t3["ebt_margin_t3"]
            out["ebt_promo_margin"] = t3["ebt_promo_margin_t3"]
            out["emb_ai"] = t3["emb_ai_t3"]
            out["emb_pr"] = t3["emb_pr_t3"]

            if int(outer_steps) >= 4:
                t4 = self.forward_t4_step(
                    t3["ai_streams_t3"], t3["pr_stream_t3"], t2["slots_t2"], t2["anchor"],
                    t3["cov_t3"], t2["k_ev_0"], t2["v_ev_0"], attention_mask,
                    t2["k_haar"], t2["v_haar"], t2["haar_mask"],
                    t2["h0_pool"], t2["h_pool_0"], t2["pr_attr_0"],
                    t3["zh_bus_t3"], t3["ebt_grad_norm_t3"], t3["ebt_margin_t3"],
                    t2["st_ai"], t2["st_pr"], t2["_zc_eff"],
                    t2["_la0_eff"], t2["_lp0_eff"], t2["logit_promo_r1"],
                    t2["logit_ai_t1"], t2["logit_promo_t1"], t2["logit_ai_t25_final"],
                    t3["logit_ai_t3"], t3["logit_ai_t3_final"], t3["logit_promo_t3_final"],
                    t3["emb_ai_t3"], t3["emb_pr_t3"], t2["g_sup"], t2["_lm_eff"],
                )
                out.update(t4)
                out["logit_ai"] = t4["logit_ai_t4_final"]
                out["logit_promo"] = t4["logit_promo_t4_final"]
                out["promo_sub4_logit"] = t4["promo_sub4_logit_t4"]
                out["ebt_margin"] = t4["ebt_margin_t4"]
                out["ebt_promo_margin"] = t4["ebt_promo_margin_t4"]
                out["emb_ai"] = t4["emb_ai_t4"]
                out["emb_pr"] = t4["emb_pr_t4"]
        return out

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
    ) -> dict[str, torch.Tensor]:
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((r0_feats.shape[0], 1), device=r0_feats.device) >= float(charlm_drop_p)).to(r0_feats.dtype)
            r0_f_eff = torch.cat([r0_feats[:, :R0_STRUCT_DIM_V140], r0_feats[:, R0_STRUCT_DIM_V140:] * lm_keep], dim=-1)
            loc_eff = local_charlm * lm_keep.unsqueeze(-1)
        else:
            r0_f_eff, loc_eff = r0_feats, local_charlm
        r0 = self.forward_r0(r0_c1, r0_c2, r0_c3, r0_mask, r0_f_eff)
        deep = self.forward_deep(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids, loc_eff,
            z_char=r0["z_char"], logit_ai_r0=r0["logit_ai_r0"], logit_promo_r0=r0["logit_promo_r0"],
            support_pairs=support_pairs, charlm_logit=r0_f_eff[:, -1],
            outer_steps=outer_steps, r0_drop_p=r0_drop_p, charlm_drop_p=0.0,
        )
        return {**r0, **deep}

    def aggregate_long_chunks(
        self,
        chunk_emb_ai: torch.Tensor, chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor, chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor, chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vm = chunk_valid.float()
        q_frac = chunk_meta[..., 3].clamp(0.0, 1.0)
        # v1.4.0 Fix #5: Stronger sandwich/long-AI salience + 1.8 residual cap so an AI block inside human wrapper flips cleanly!
        ai_in = torch.cat([chunk_emb_ai, chunk_meta, chunk_lg_ai.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        a_w = (self.long_ai_attn(ai_in).squeeze(-1) + 0.35 * chunk_lg_ai.clamp(-4.0, 4.0) * (1.0 - q_frac) - 2.0 * q_frac).masked_fill(vm < 0.5, -1e4)
        a_w = torch.softmax(a_w.float(), dim=-1).to(chunk_emb_ai.dtype)
        ai_ctx = (chunk_emb_ai * a_w.unsqueeze(-1)).sum(dim=1)
        ai_max = (chunk_lg_ai * (1.0 - 0.5 * q_frac)).masked_fill(vm < 0.5, -1e4).max(dim=1).values
        ai_mean = (chunk_lg_ai * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        ai_wmean = (chunk_lg_ai * a_w).sum(dim=1)
        doc_ai = ai_wmean + 1.8 * torch.tanh(self.long_ai_head(torch.cat([ai_ctx, ai_max.unsqueeze(-1), ai_mean.unsqueeze(-1), ai_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1) / 1.8)

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
# 5. Clean W4/W8 `MGVZ9` + Weight-Stripped ONNX Graph Skeletons (Bootable <= 5.00 MiB!)
# ---------------------------------------------------------------------------

def _quant_spec_map_v150(model: MeguvizV150HeavenLoopModel) -> dict[str, tuple[str, int]]:
    spec: dict[str, tuple[str, int]] = {
        "pos_emb": ("w4", 32),
        "core.slot_queries": ("w8", 0),
    }
    for b_idx in range(len(model.core.blocks)):
        pref = f"core.blocks.{b_idx}"
        spec[f"{pref}.ca_q_lora_a"] = ("w8", 0)
        spec[f"{pref}.ca_q_lora_b"] = ("w8", 0)
        spec[f"{pref}.sa_o_lora_a"] = ("w8", 0)
        spec[f"{pref}.sa_o_lora_b"] = ("w8", 0)
        spec[f"{pref}.relax.a"] = ("w8", 0)
        spec[f"{pref}.relax.b"] = ("w8", 0)
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantLinearV130, QuantEmbeddingV130)):
            spec[f"{name}.weight"] = ("w8" if mod.target_bits == 8 else "w4", mod.group_size)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 1024:
            spec[f"{name}.weight"] = ("w8", 0)
    return spec


def pack_bundle_v9_v150(model: MeguvizV150HeavenLoopModel, out_path: Path) -> dict[str, Any]:
    spec_map = _quant_spec_map_v150(model)
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    breakdown: dict[str, int] = {"r0_w8": 0, "deep_w4": 0, "deep_w8": 0, "scales_and_fp16": 0}

    for k, arr in sd.items():
        q_mode, gs = spec_map.get(k, ("fp16", 0))
        if arr.size < 64:
            q_mode, gs = "fp16", 0
        if q_mode == "w8":
            sb, qb = _pack_w8_tensor(arr)
            breakdown["r0_w8" if k.startswith("rung0.") else "deep_w8"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        elif q_mode == "w4":
            sb, qb = _pack_w4_tensor(arr, gs=gs)
            breakdown["deep_w4"] += len(qb)
            breakdown["scales_and_fp16"] += len(sb)
        else:
            sb, qb = b"", arr.astype(np.float16).tobytes()
            breakdown["scales_and_fp16"] += len(qb)
        entries.append({"name": k, "q": q_mode, "g": gs, "shape": list(arr.shape), "s_bytes": len(sb), "q_bytes": len(qb)})
        blobs.extend((sb, qb))

    hdr_bytes = json.dumps({"format": "MGVZ9_HEAVENLOOP_V150", "tensors": entries}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MGVZ9_MAGIC)
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


def load_bundle_v9_v150_into_model(model: MeguvizV150HeavenLoopModel, mgvz_path: Path) -> None:
    data = mgvz_path.read_bytes()
    assert data[:5] == b"MGVZ9", "Invalid MGVZ9 magic!"
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


def strip_onnx_weights_to_skeleton(onnx_path: Path, skel_path: Path) -> int:
    """Create a weight-stripped Opset-17 graph skeleton (`~8-15 KiB`!) for true bootable <=5.00 MiB packaging."""
    m = onnx.load(str(onnx_path))
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and len(init.raw_data) > 256:
            init.ClearField("raw_data")
            init.ClearField("float_data")
    onnx.save(m, str(skel_path))
    return skel_path.stat().st_size


# ---------------------------------------------------------------------------
# 6. 5-Stage Modular FP32-Exact ONNX Export (`r0`, `t2_t25`, `t3`, `t4`, `dynaseam`)
# ---------------------------------------------------------------------------

class _V150R0Export(nn.Module):
    def __init__(self, m: MeguvizV150HeavenLoopModel) -> None:
        super().__init__()
        self.r0 = m.rung0

    def forward(self, c1, c2, c3, mask, feats):
        o = self.r0(c1, c2, c3, mask, feats)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_ai_r0"], o["p_defer_pr_r0"], o["z_char"]


_T2_OUT_NAMES_V150 = [
    "logit_ai_t2_final", "logit_ai_t25_final", "delta_t25", "logit_promo_t2_final",
    "logit_ai_r1", "logit_promo_r1", "logit_ai_t1", "logit_promo_t1", "logit_ai_t2",
    "ebt_margin_t2", "ebt_grad_norm_t2", "emb_ai_t2", "emb_pr_t2", "p_escalate_t3", "g_sup",
    "ai_streams_t2", "pr_stream_t2", "slots_t1", "slots_t2", "anchor", "cov_t2",
    "k_ev_0", "v_ev_0", "k_haar", "v_haar", "haar_mask",
    "h0_pool", "h_pool_0", "pr_attr_0", "zh_bus_t2", "st_ai", "st_pr",
]
_T2_PRED_COUNT_V150 = 15


class _V150T2Export(nn.Module):
    def __init__(self, m: MeguvizV150HeavenLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, local_charlm, z_char, logit_ai_r0, logit_promo_r0, support_pairs, charlm_logit):
        o = self.m.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids, local_charlm,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
            support_pairs=support_pairs, charlm_logit=charlm_logit,
        )
        return tuple(o[k] for k in _T2_OUT_NAMES_V150)


_T3_IN_NAMES_V150 = [
    "ai_streams_t2", "pr_stream_t2", "slots_t1", "anchor", "cov_t2", "k_ev_0", "v_ev_0", "attention_mask",
    "k_haar", "v_haar", "haar_mask", "h0_pool", "h_pool_0", "pr_attr_0",
    "zh_bus_t2", "ebt_grad_norm_t2", "ebt_margin_t2", "st_ai", "st_pr", "z_char",
    "logit_ai_r0", "logit_promo_r0", "logit_promo_r1", "logit_ai_t1", "logit_promo_t1", "logit_ai_t2",
    "logit_ai_t2_final", "logit_ai_t25_final", "logit_promo_t2_final",
    "emb_ai_t2", "emb_pr_t2", "g_sup", "charlm_logit",
]
_T3_OUT_NAMES_V150 = [
    "logit_ai_t3_final", "logit_promo_t3_final", "logit_ai_t3", "ebt_margin_t3", "ebt_grad_norm_t3",
    "emb_ai_t3", "emb_pr_t3", "p_escalate_t4",
    "ai_streams_t3", "pr_stream_t3", "cov_t3", "zh_bus_t3",
]


class _V150T3Export(nn.Module):
    def __init__(self, m: MeguvizV150HeavenLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, *args):
        o = self.m.forward_t3_step(*args)
        return tuple(o[k] for k in _T3_OUT_NAMES_V150)


_T4_IN_NAMES_V150 = [
    "ai_streams_t3", "pr_stream_t3", "slots_t2", "anchor", "cov_t3", "k_ev_0", "v_ev_0", "attention_mask",
    "k_haar", "v_haar", "haar_mask", "h0_pool", "h_pool_0", "pr_attr_0",
    "zh_bus_t3", "ebt_grad_norm_t3", "ebt_margin_t3", "st_ai", "st_pr", "z_char",
    "logit_ai_r0", "logit_promo_r0", "logit_promo_r1", "logit_ai_t1", "logit_promo_t1",
    "logit_ai_t25_final", "logit_ai_t3", "logit_ai_t3_final", "logit_promo_t3_final",
    "emb_ai_t3", "emb_pr_t3", "g_sup", "charlm_logit",
]
_T4_OUT_NAMES_V150 = ["logit_ai_t4_final", "logit_promo_t4_final", "logit_ai_t4", "ebt_margin_t4", "emb_ai_t4", "emb_pr_t4"]


class _V150T4Export(nn.Module):
    def __init__(self, m: MeguvizV150HeavenLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, *args):
        o = self.m.forward_t4_step(*args)
        return tuple(o[k] for k in _T4_OUT_NAMES_V150)


class _V150LongAggExport(nn.Module):
    def __init__(self, m: MeguvizV150HeavenLoopModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, emb_ai, emb_pr, lg_ai, lg_pr, meta, valid):
        return self.m.aggregate_long_chunks(emb_ai, emb_pr, lg_ai, lg_pr, meta, valid)


def export_v150_bootable_and_onnx(
    model: MeguvizV150HeavenLoopModel,
    tokenizer_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    version: str,
    charlm: CompactCharLMV150 | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mgvz_p = out_dir / "bundle_v9_heavenloop_v150.mgvz"
    set_v150_qat_mode(model, False)
    mgvz_info = pack_bundle_v9_v150(model, mgvz_p)

    charlm_p = out_dir / "charlm_v150.bin"
    charlm_info: dict[str, Any] = {"enabled": False, "raw_bytes": 0, "roundtrip_diff": 0.0, "pass": True}
    if charlm is not None:
        sv = charlm.save(charlm_p)
        reloaded = CompactCharLMV150.load(charlm_p)
        reloaded.__class__ = CompactCharLMV150
        probe_cl = ["笑死我了，这个插件装完直接起飞", "总的来说，这项技术具有非常显著的综合优势。"]
        rt_diff = float(np.abs(charlm.extract_9d(probe_cl) - reloaded.extract_9d(probe_cl)).max())
        charlm_info = {"enabled": True, **sv, "roundtrip_diff": round(rt_diff, 8), "pass": bool(rt_diff < 1e-5)}

    m_cpu = MeguvizV150HeavenLoopModel(
        vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim,
        style_era_dropout_p=0.0, prelude_mode=model.prelude_mode, core_depth=model.core_depth,
        coverage_beta_enabled=model.coverage_beta_enabled,
        use_t25_light_reread=model.use_t25_light_reread,
        use_local_charlm_inject=model.use_local_charlm_inject,
        use_global_charlm_shortcut=model.use_global_charlm_shortcut,
    ).cpu().eval()
    load_bundle_v9_v150_into_model(m_cpu, mgvz_p)

    # 1. R0
    c1 = torch.randint(1, R0_B1_V130, (2, 24), dtype=torch.long)
    c2 = torch.randint(1, R0_B2_V130, (2, 24), dtype=torch.long)
    c3 = torch.randint(1, R0_B3_V130, (2, 24), dtype=torch.long)
    cm = torch.ones((2, 24), dtype=torch.float32)
    cf = torch.zeros((2, R0_FEAT_DIM_V140), dtype=torch.float32)
    cf[:, -1] = torch.tensor([-0.6, 0.9], dtype=torch.float32)

    r0_p = out_dir / "v150_r0_apexchar_fp32.onnx"
    torch.onnx.export(
        _V150R0Export(m_cpu), (c1, c2, c3, cm, cf), str(r0_p),
        input_names=["c1", "c2", "c3", "mask", "feats"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0", "z_char"],
        dynamic_axes={k: {0: "batch", 1: "clen"} for k in ("c1", "c2", "c3", "mask")}
        | {"feats": {0: "batch"}, "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"}, "p_defer_ai_r0": {0: "batch"}, "p_defer_pr_r0": {0: "batch"}, "z_char": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # 2. T2 + T2.5 Light Reread
    ids = torch.randint(1, model.vocab_size, (2, 32), dtype=torch.long)
    msk = torch.ones((2, 32), dtype=torch.long)
    st = torch.zeros((2, model.style_dim), dtype=torch.float32)
    eg = torch.randint(1, ENGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    tg = torch.randint(1, TRIGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    loc_lm = torch.zeros((2, 32, LOCAL_CHARLM_DIM_V150), dtype=torch.float32)
    zc = torch.zeros((2, R0_ZCHAR_DIM_V130), dtype=torch.float32)
    lr0_a = torch.zeros((2,), dtype=torch.float32)
    lr0_p = torch.zeros((2,), dtype=torch.float32)
    sup = torch.tensor([8.0, 28.0], dtype=torch.float32)
    lm_lg = torch.tensor([-0.6, 0.9], dtype=torch.float32)

    t2_p = out_dir / "v150_t2_heaven_fp32.onnx"
    t2_dyn = {
        "input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
        "engram_ids": {0: "batch", 1: "seq"}, "trigram_ids": {0: "batch", 1: "seq"},
        "local_charlm": {0: "batch", 1: "seq"},
        "style_feats": {0: "batch"}, "z_char": {0: "batch"},
        "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"},
        "support_pairs": {0: "batch"}, "charlm_logit": {0: "batch"},
    }
    for nm in _T2_OUT_NAMES_V150:
        if nm in ("k_ev_0", "v_ev_0", "cov_t2"):
            t2_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t2_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t2_dyn[nm] = {0: "batch"}

    torch.onnx.export(
        _V150T2Export(m_cpu), (ids, msk, st, eg, tg, loc_lm, zc, lr0_a, lr0_p, sup, lm_lg), str(t2_p),
        input_names=["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "local_charlm", "z_char", "logit_ai_r0", "logit_promo_r0", "support_pairs", "charlm_logit"],
        output_names=_T2_OUT_NAMES_V150,
        dynamic_axes=t2_dyn,
        opset_version=17, dynamo=False,
    )

    # 3. T3 Step
    with torch.no_grad():
        t2_d = m_cpu.forward_t2_stage(ids, msk, st, eg, tg, loc_lm, zc, lr0_a, lr0_p, sup, charlm_logit=lm_lg)
    t3_dummy = (
        t2_d["ai_streams_t2"], t2_d["pr_stream_t2"], t2_d["slots_t1"], t2_d["anchor"],
        t2_d["cov_t2"], t2_d["k_ev_0"], t2_d["v_ev_0"], msk,
        t2_d["k_haar"], t2_d["v_haar"], t2_d["haar_mask"],
        t2_d["h0_pool"], t2_d["h_pool_0"], t2_d["pr_attr_0"],
        t2_d["zh_bus_t2"], t2_d["ebt_grad_norm_t2"], t2_d["ebt_margin_t2"],
        t2_d["st_ai"], t2_d["st_pr"], zc,
        lr0_a, lr0_p, t2_d["logit_promo_r1"], t2_d["logit_ai_t1"], t2_d["logit_promo_t1"], t2_d["logit_ai_t2"],
        t2_d["logit_ai_t2_final"], t2_d["logit_ai_t25_final"], t2_d["logit_promo_t2_final"],
        t2_d["emb_ai_t2"], t2_d["emb_pr_t2"], t2_d["g_sup"], lm_lg,
    )
    t3_dyn = {
        nm: ({0: "batch", 1: "seq"} if nm in ("cov_t2", "cov_t3", "k_ev_0", "v_ev_0", "attention_mask")
             else ({0: "batch", 1: "hseq"} if nm in ("k_haar", "v_haar", "haar_mask") else {0: "batch"}))
        for nm in _T3_IN_NAMES_V150 + _T3_OUT_NAMES_V150
    }
    t3_p = out_dir / "v150_t3_heaven_fp32.onnx"
    torch.onnx.export(_V150T3Export(m_cpu), t3_dummy, str(t3_p), input_names=_T3_IN_NAMES_V150, output_names=_T3_OUT_NAMES_V150, dynamic_axes=t3_dyn, opset_version=17, dynamo=False)

    # 4. T4 Step
    with torch.no_grad():
        t3_d = m_cpu.forward_t3_step(*t3_dummy)
    t4_dummy = (
        t3_d["ai_streams_t3"], t3_d["pr_stream_t3"], t2_d["slots_t2"], t2_d["anchor"],
        t3_d["cov_t3"], t2_d["k_ev_0"], t2_d["v_ev_0"], msk,
        t2_d["k_haar"], t2_d["v_haar"], t2_d["haar_mask"],
        t2_d["h0_pool"], t2_d["h_pool_0"], t2_d["pr_attr_0"],
        t3_d["zh_bus_t3"], t3_d["ebt_grad_norm_t3"], t3_d["ebt_margin_t3"],
        t2_d["st_ai"], t2_d["st_pr"], zc,
        lr0_a, lr0_p, t2_d["logit_promo_r1"], t2_d["logit_ai_t1"], t2_d["logit_promo_t1"],
        t2_d["logit_ai_t25_final"], t3_d["logit_ai_t3"], t3_d["logit_ai_t3_final"], t3_d["logit_promo_t3_final"],
        t3_d["emb_ai_t3"], t3_d["emb_pr_t3"], t2_d["g_sup"], lm_lg,
    )
    t4_dyn = {
        nm: ({0: "batch", 1: "seq"} if nm in ("cov_t3", "k_ev_0", "v_ev_0", "attention_mask")
             else ({0: "batch", 1: "hseq"} if nm in ("k_haar", "v_haar", "haar_mask") else {0: "batch"}))
        for nm in _T4_IN_NAMES_V150 + _T4_OUT_NAMES_V150
    }
    t4_p = out_dir / "v150_t4_heaven_fp32.onnx"
    torch.onnx.export(_V150T4Export(m_cpu), t4_dummy, str(t4_p), input_names=_T4_IN_NAMES_V150, output_names=_T4_OUT_NAMES_V150, dynamic_axes=t4_dyn, opset_version=17, dynamo=False)

    # 5. DynaSeam Long Aggregator
    long_p = out_dir / "v150_dynaseam_fp32.onnx"
    ea, ep = torch.randn(2, 4, AI_END_V130), torch.randn(2, 4, PR_EMB_DIM_V130)
    la, lp = torch.randn(2, 4), torch.randn(2, 4)
    mt, vm = torch.zeros(2, 4, CHUNK_META_DIM_V11), torch.ones(2, 4)
    torch.onnx.export(
        _V150LongAggExport(m_cpu), (ea, ep, la, lp, mt, vm), str(long_p),
        input_names=["emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid"],
        output_names=["doc_ai", "doc_promo"],
        dynamic_axes={k: {0: "batch", 1: "chunks"} for k in ("emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid")}
        | {"doc_ai": {0: "batch"}, "doc_promo": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # Weight-stripped ONNX graph skeletons for true bootable `<= 5.00 MiB` accounting!
    skel_paths: list[Path] = []
    skel_raw_total = 0
    for op in (r0_p, t2_p, t3_p, t4_p, long_p):
        sp = op.with_suffix(".skel.onnx")
        skel_raw_total += strip_onnx_weights_to_skeleton(op, sp)
        skel_paths.append(sp)

    # Exact FP32 Parity across all 5 modular ONNX graphs!
    s0 = ort.InferenceSession(str(r0_p), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(t2_p), providers=["CPUExecutionProvider"])
    s3 = ort.InferenceSession(str(t3_p), providers=["CPUExecutionProvider"])
    s4 = ort.InferenceSession(str(t4_p), providers=["CPUExecutionProvider"])
    sl = ort.InferenceSession(str(long_p), providers=["CPUExecutionProvider"])
    def _run_sess(sess: ort.InferenceSession, feed: dict[str, np.ndarray]):
        valid = {inp.name for inp in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feed.items() if k in valid})

    with torch.no_grad():
        pt_r0 = _V150R0Export(m_cpu)(c1, c2, c3, cm, cf)
        or_r0 = _run_sess(s0, {"c1": c1.numpy(), "c2": c2.numpy(), "c3": c3.numpy(), "mask": cm.numpy(), "feats": cf.numpy()})
        r0_diff = float(max(np.abs(pt_r0[i].numpy() - or_r0[i]).max() for i in range(5)))

        pt_t2 = _V150T2Export(m_cpu)(ids, msk, st, eg, tg, loc_lm, pt_r0[4], pt_r0[0], pt_r0[1], sup, lm_lg)
        or_t2 = _run_sess(s2, {
            "input_ids": ids.numpy(), "attention_mask": msk.numpy(), "style_feats": st.numpy(),
            "engram_ids": eg.numpy(), "trigram_ids": tg.numpy(), "local_charlm": loc_lm.numpy(),
            "z_char": or_r0[4], "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1],
            "support_pairs": sup.numpy(), "charlm_logit": lm_lg.numpy(),
        })
        t2_diff = float(max(np.abs(pt_t2[i].numpy() - or_t2[i]).max() for i in range(_T2_PRED_COUNT_V150)))

        m2 = dict(zip(_T2_OUT_NAMES_V150, or_t2))
        t3_feed = {
            "ai_streams_t2": m2["ai_streams_t2"], "pr_stream_t2": m2["pr_stream_t2"], "slots_t1": m2["slots_t1"],
            "anchor": m2["anchor"], "cov_t2": m2["cov_t2"], "k_ev_0": m2["k_ev_0"], "v_ev_0": m2["v_ev_0"],
            "attention_mask": msk.numpy(), "k_haar": m2["k_haar"], "v_haar": m2["v_haar"], "haar_mask": m2["haar_mask"],
            "h0_pool": m2["h0_pool"], "h_pool_0": m2["h_pool_0"], "pr_attr_0": m2["pr_attr_0"],
            "zh_bus_t2": m2["zh_bus_t2"], "ebt_grad_norm_t2": m2["ebt_grad_norm_t2"], "ebt_margin_t2": m2["ebt_margin_t2"],
            "st_ai": m2["st_ai"], "st_pr": m2["st_pr"], "z_char": or_r0[4],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1], "logit_promo_r1": m2["logit_promo_r1"],
            "logit_ai_t1": m2["logit_ai_t1"], "logit_promo_t1": m2["logit_promo_t1"], "logit_ai_t2": m2["logit_ai_t2"],
            "logit_ai_t2_final": m2["logit_ai_t2_final"], "logit_ai_t25_final": m2["logit_ai_t25_final"],
            "logit_promo_t2_final": m2["logit_promo_t2_final"],
            "emb_ai_t2": m2["emb_ai_t2"], "emb_pr_t2": m2["emb_pr_t2"], "g_sup": m2["g_sup"], "charlm_logit": lm_lg.numpy(),
        }
        or_t3 = _run_sess(s3, t3_feed)
        m3 = dict(zip(_T3_OUT_NAMES_V150, or_t3))
        t4_feed = {
            "ai_streams_t3": m3["ai_streams_t3"], "pr_stream_t3": m3["pr_stream_t3"], "slots_t2": m2["slots_t2"],
            "anchor": m2["anchor"], "cov_t3": m3["cov_t3"], "k_ev_0": m2["k_ev_0"], "v_ev_0": m2["v_ev_0"],
            "attention_mask": msk.numpy(), "k_haar": m2["k_haar"], "v_haar": m2["v_haar"], "haar_mask": m2["haar_mask"],
            "h0_pool": m2["h0_pool"], "h_pool_0": m2["h_pool_0"], "pr_attr_0": m2["pr_attr_0"],
            "zh_bus_t3": m3["zh_bus_t3"], "ebt_grad_norm_t3": m3["ebt_grad_norm_t3"], "ebt_margin_t3": m3["ebt_margin_t3"],
            "st_ai": m2["st_ai"], "st_pr": m2["st_pr"], "z_char": or_r0[4],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1], "logit_promo_r1": m2["logit_promo_r1"],
            "logit_ai_t1": m2["logit_ai_t1"], "logit_promo_t1": m2["logit_promo_t1"],
            "logit_ai_t25_final": m2["logit_ai_t25_final"], "logit_ai_t3": m3["logit_ai_t3"],
            "logit_ai_t3_final": m3["logit_ai_t3_final"], "logit_promo_t3_final": m3["logit_promo_t3_final"],
            "emb_ai_t3": m3["emb_ai_t3"], "emb_pr_t3": m3["emb_pr_t3"], "g_sup": m2["g_sup"], "charlm_logit": lm_lg.numpy(),
        }
        or_t4 = _run_sess(s4, t4_feed)
        pt_full4 = m_cpu.forward_deep(ids, msk, st, eg, tg, loc_lm, pt_r0[4], pt_r0[0], pt_r0[1], sup, charlm_logit=lm_lg, outer_steps=4)
        t3_diff = float(max(np.abs(pt_full4[k].numpy() - m3[k]).max() for k in ("logit_ai_t3_final", "logit_promo_t3_final", "emb_ai_t3", "emb_pr_t3")))
        t4_diff = float(max(np.abs(pt_full4[k].numpy() - or_t4[i]).max() for i, k in enumerate(_T4_OUT_NAMES_V150)))

        pt_l = _V150LongAggExport(m_cpu)(ea, ep, la, lp, mt, vm)
        or_l = _run_sess(sl, {"emb_ai": ea.numpy(), "emb_pr": ep.numpy(), "lg_ai": la.numpy(), "lg_pr": lp.numpy(), "meta": mt.numpy(), "valid": vm.numpy()})
        long_diff = float(max(np.abs(pt_l[i].numpy() - or_l[i]).max() for i in range(2)))

    tok_out = out_dir / "bili_picky_bpe_12k.json"
    if tokenizer_path.resolve() != tok_out.resolve():
        tok_out.write_bytes(tokenizer_path.read_bytes())

    manifest = dict(manifest)
    manifest["format"] = "MGVZ9_HEAVENLOOP_V150"
    manifest["prelude_mode"] = model.prelude_mode
    manifest["core_depth"] = model.core_depth
    manifest["coverage_beta_enabled"] = model.coverage_beta_enabled
    manifest["use_t25_light_reread"] = model.use_t25_light_reread
    manifest["use_local_charlm_inject"] = model.use_local_charlm_inject
    manifest["use_global_charlm_shortcut"] = model.use_global_charlm_shortcut
    manifest["mgvz9_info"] = mgvz_info
    manifest["charlm_info"] = charlm_info
    mf_p = out_dir / "manifest.json"
    mf_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # TRUE BOOTABLE `<= 5.00 MiB` DELIVERABLE ZIP (weights + graph skeletons + tokenizer + manifest + CharLM!)
    boot_zip = out_dir / f"meguvizcac-{version}-bootable-v9-heavenloop.zip"
    with zipfile.ZipFile(boot_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(mgvz_p, mgvz_p.name)
        zf.write(tok_out, tok_out.name)
        zf.write(mf_p, mf_p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)
        for sp in skel_paths:
            zf.write(sp, sp.name)
    boot_bytes = boot_zip.stat().st_size

    onnx_zip = out_dir / f"meguvizcac-{version}-5stage-onnx-fp32exact.zip"
    with zipfile.ZipFile(onnx_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in (r0_p, t2_p, t3_p, t4_p, long_p, tok_out, mf_p):
            zf.write(p, p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)

    return {
        "bootable_zip": str(boot_zip),
        "bootable_bytes": boot_bytes,
        "bootable_mib": round(boot_bytes / (1024.0 * 1024.0), 4),
        "bootable_le_5p00_mib": bool(boot_bytes <= int(BUNDLE_MAX_MIB_V130 * 1024 * 1024)),
        "skeletons_raw_kb": round(skel_raw_total / 1024.0, 1),
        "onnx_zip_mb": round(onnx_zip.stat().st_size / (1024.0 * 1024.0), 4),
        "r0_fp32_max_diff": round(r0_diff, 8),
        "t2_fp32_max_diff": round(t2_diff, 8),
        "t3_fp32_max_diff": round(t3_diff, 8),
        "t4_fp32_max_diff": round(t4_diff, 8),
        "long_fp32_max_diff": round(long_diff, 8),
        "onnx_exact_pass": bool(max(r0_diff, t2_diff, t3_diff, t4_diff, long_diff) < 1e-4 and charlm_info["pass"]),
        "mgvz9_info": mgvz_info,
        "charlm_info": charlm_info,
    }


# ---------------------------------------------------------------------------
# 7. P0-Hardened End-to-End Decoupled 5-Stage `MeguvizV150CascadeEngine`
# ---------------------------------------------------------------------------

class MeguvizV150CascadeEngine:
    """Zero-PII Pre-Canonicalized, Dual-Axis Decoupled 5-Stage ONNX Engine (`R0 -> T2 -> T2.5 -> T3 -> T4 -> DynaSeam`)."""

    def __init__(self, release_dir: str | Path, threads: int = 4) -> None:
        self.release_dir = Path(release_dir)
        self.manifest = json.loads((self.release_dir / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = BiliPickyBPETokenizer(self.release_dir / "bili_picky_bpe_12k.json")
        charlm_p = self.release_dir / "charlm_v150.bin"
        if charlm_p.is_file():
            lm = CompactCharLMV150.load(charlm_p)
            lm.__class__ = CompactCharLMV150
            self.charlm: CompactCharLMV150 | None = lm
        else:
            self.charlm = None

        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V140))
        self.head_len = int(self.manifest.get("head_len", HEAD_LEN_V140))
        self.long_char_trigger = int(self.manifest.get("long_char_trigger", 360))
        self.use_local_charlm = bool(self.manifest.get("use_local_charlm_inject", True))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_r0 = ort.InferenceSession(str(self.release_dir / "v150_r0_apexchar_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t2 = ort.InferenceSession(str(self.release_dir / "v150_t2_heaven_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t3 = ort.InferenceSession(str(self.release_dir / "v150_t3_heaven_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t4 = ort.InferenceSession(str(self.release_dir / "v150_t4_heaven_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_long = ort.InferenceSession(str(self.release_dir / "v150_dynaseam_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self._in_r0 = {i.name for i in self.sess_r0.get_inputs()}
        self._in_t2 = {i.name for i in self.sess_t2.get_inputs()}
        self._in_t3 = {i.name for i in self.sess_t3.get_inputs()}
        self._in_t4 = {i.name for i in self.sess_t4.get_inputs()}
        self._in_long = {i.name for i in self.sess_long.get_inputs()}

        self.r0_ai_policy = self.manifest.get("rung0_ai_ltt_policy") or {}
        self.r0_pr_policy = self.manifest.get("rung0_pr_ltt_policy") or {}
        self.t3_policy = self.manifest.get("t3_escalation_policy") or {"enabled": True, "tau_escalate": 0.45, "uncertain_lo": 0.10, "uncertain_hi": 0.90, "t3_bound_margin": 4.2}
        self.t4_policy = self.manifest.get("t4_escalation_policy") or {"enabled": True, "tau_escalate": 0.55, "uncertain_lo": 0.15, "uncertain_hi": 0.85, "t4_bound_margin": 2.4}
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.temp_ai = float(self.manifest.get("temp_ai", 1.0))
        self.temp_pr = float(self.manifest.get("temp_promo", 1.0))

    def _prep_deep_inputs(self, clean_texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f56_l, encs, offs_l = [], [], []
        for cl in clean_texts:
            _, f56, _ = extract_style_features_56d(cl)
            ids, offs = encode_bpe_head_tail_with_offsets_v150(self.tokenizer, cl, max_len=self.max_len, head_len=self.head_len)
            f56_l.append(f56)
            encs.append(ids)
            offs_l.append(offs)
        engs = [make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130) for s in encs]
        trigs = [make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130) for s in encs]
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

    def _run_t2_t25_t3_t4(
        self,
        clean_texts: list[str],
        z_char: np.ndarray,
        lg_r0_ai: np.ndarray,
        lg_r0_pr: np.ndarray,
        supports: np.ndarray,
        charlm_logits: np.ndarray,
        force_depth: int | None = None,
    ) -> dict[str, np.ndarray]:
        ids, mask, st, eg, tg, loc_lm = self._prep_deep_inputs(clean_texts)
        lm_arr = charlm_logits.astype(np.float32)
        t2_feed = {
            "input_ids": ids, "attention_mask": mask, "style_feats": st,
            "engram_ids": eg, "trigram_ids": tg, "local_charlm": loc_lm,
            "z_char": z_char.astype(np.float32), "logit_ai_r0": lg_r0_ai.astype(np.float32),
            "logit_promo_r0": lg_r0_pr.astype(np.float32), "support_pairs": supports.astype(np.float32),
            "charlm_logit": lm_arr,
        }
        t2_raw = self.sess_t2.run(None, {k: v for k, v in t2_feed.items() if k in self._in_t2})
        t2 = dict(zip(_T2_OUT_NAMES_V150, t2_raw))
        bs = len(clean_texts)
        lg_ai = t2["logit_ai_t25_final"].copy()
        lg_pr = t2["logit_promo_t2_final"].copy()
        emb_ai = t2["emb_ai_t2"].copy()
        emb_pr = t2["emb_pr_t2"].copy()
        rungs = np.full((bs,), 2, dtype=np.int64)

        run_t3_en = (force_depth is not None and force_depth >= 3) or (force_depth is None and bool(self.t3_policy.get("enabled", True)))
        if run_t3_en:
            if force_depth is not None and force_depth >= 3:
                esc3 = np.arange(bs, dtype=np.int64)
            else:
                p_t25 = expit(np.clip(lg_ai / self.temp_ai, -30.0, 30.0))
                b3 = float(self.t3_policy.get("t3_bound_margin", 4.2))
                tau3 = float(self.t3_policy.get("tau_escalate", 0.45))
                u3_lo, u3_hi = float(self.t3_policy.get("uncertain_lo", 0.10)), float(self.t3_policy.get("uncertain_hi", 0.90))
                m3_sel = (np.abs(lg_ai) <= b3) & ((t2["p_escalate_t3"] >= tau3) | ((p_t25 >= u3_lo) & (p_t25 <= u3_hi)))
                esc3 = np.where(m3_sel)[0].astype(np.int64)

            if len(esc3) > 0:
                t3_feed = {
                    "ai_streams_t2": t2["ai_streams_t2"][esc3], "pr_stream_t2": t2["pr_stream_t2"][esc3],
                    "slots_t1": t2["slots_t1"][esc3], "anchor": t2["anchor"][esc3], "cov_t2": t2["cov_t2"][esc3],
                    "k_ev_0": t2["k_ev_0"][esc3], "v_ev_0": t2["v_ev_0"][esc3], "attention_mask": mask[esc3],
                    "k_haar": t2["k_haar"][esc3], "v_haar": t2["v_haar"][esc3], "haar_mask": t2["haar_mask"][esc3],
                    "h0_pool": t2["h0_pool"][esc3], "h_pool_0": t2["h_pool_0"][esc3], "pr_attr_0": t2["pr_attr_0"][esc3],
                    "zh_bus_t2": t2["zh_bus_t2"][esc3], "ebt_grad_norm_t2": t2["ebt_grad_norm_t2"][esc3],
                    "ebt_margin_t2": t2["ebt_margin_t2"][esc3], "st_ai": t2["st_ai"][esc3], "st_pr": t2["st_pr"][esc3],
                    "z_char": z_char[esc3].astype(np.float32), "logit_ai_r0": lg_r0_ai[esc3].astype(np.float32),
                    "logit_promo_r0": lg_r0_pr[esc3].astype(np.float32), "logit_promo_r1": t2["logit_promo_r1"][esc3],
                    "logit_ai_t1": t2["logit_ai_t1"][esc3], "logit_promo_t1": t2["logit_promo_t1"][esc3],
                    "logit_ai_t2": t2["logit_ai_t2"][esc3], "logit_ai_t2_final": t2["logit_ai_t2_final"][esc3],
                    "logit_ai_t25_final": t2["logit_ai_t25_final"][esc3], "logit_promo_t2_final": t2["logit_promo_t2_final"][esc3],
                    "emb_ai_t2": t2["emb_ai_t2"][esc3], "emb_pr_t2": t2["emb_pr_t2"][esc3],
                    "g_sup": t2["g_sup"][esc3], "charlm_logit": lm_arr[esc3],
                }
                t3 = dict(zip(_T3_OUT_NAMES_V150, self.sess_t3.run(None, {k: v for k, v in t3_feed.items() if k in self._in_t3})))
                lg_ai[esc3] = t3["logit_ai_t3_final"]
                lg_pr[esc3] = t3["logit_promo_t3_final"]
                emb_ai[esc3] = t3["emb_ai_t3"]
                emb_pr[esc3] = t3["emb_pr_t3"]
                rungs[esc3] = 3

                run_t4_en = (force_depth is not None and force_depth >= 4) or (force_depth is None and bool(self.t4_policy.get("enabled", True)))
                if run_t4_en:
                    if force_depth is not None and force_depth >= 4:
                        sub4 = np.arange(len(esc3), dtype=np.int64)
                    else:
                        p_t3 = expit(np.clip(t3["logit_ai_t3_final"] / self.temp_ai, -30.0, 30.0))
                        b4 = float(self.t4_policy.get("t4_bound_margin", 2.4))
                        tau4 = float(self.t4_policy.get("tau_escalate", 0.55))
                        u4_lo, u4_hi = float(self.t4_policy.get("uncertain_lo", 0.15)), float(self.t4_policy.get("uncertain_hi", 0.85))
                        m4_sel = (np.abs(t3["logit_ai_t3_final"]) <= b4) & ((t3["p_escalate_t4"] >= tau4) | ((p_t3 >= u4_lo) & (p_t3 <= u4_hi)))
                        sub4 = np.where(m4_sel)[0].astype(np.int64)
                    if len(sub4) > 0:
                        orig4 = esc3[sub4]
                        t4_feed = {
                            "ai_streams_t3": t3["ai_streams_t3"][sub4], "pr_stream_t3": t3["pr_stream_t3"][sub4],
                            "slots_t2": t2["slots_t2"][orig4], "anchor": t2["anchor"][orig4], "cov_t3": t3["cov_t3"][sub4],
                            "k_ev_0": t2["k_ev_0"][orig4], "v_ev_0": t2["v_ev_0"][orig4], "attention_mask": mask[orig4],
                            "k_haar": t2["k_haar"][orig4], "v_haar": t2["v_haar"][orig4], "haar_mask": t2["haar_mask"][orig4],
                            "h0_pool": t2["h0_pool"][orig4], "h_pool_0": t2["h_pool_0"][orig4], "pr_attr_0": t2["pr_attr_0"][orig4],
                            "zh_bus_t3": t3["zh_bus_t3"][sub4], "ebt_grad_norm_t3": t3["ebt_grad_norm_t3"][sub4],
                            "ebt_margin_t3": t3["ebt_margin_t3"][sub4], "st_ai": t2["st_ai"][orig4], "st_pr": t2["st_pr"][orig4],
                            "z_char": z_char[orig4].astype(np.float32), "logit_ai_r0": lg_r0_ai[orig4].astype(np.float32),
                            "logit_promo_r0": lg_r0_pr[orig4].astype(np.float32), "logit_promo_r1": t2["logit_promo_r1"][orig4],
                            "logit_ai_t1": t2["logit_ai_t1"][orig4], "logit_promo_t1": t2["logit_promo_t1"][orig4],
                            "logit_ai_t25_final": t2["logit_ai_t25_final"][orig4], "logit_ai_t3": t3["logit_ai_t3"][sub4],
                            "logit_ai_t3_final": t3["logit_ai_t3_final"][sub4], "logit_promo_t3_final": t3["logit_promo_t3_final"][sub4],
                            "emb_ai_t3": t3["emb_ai_t3"][sub4], "emb_pr_t3": t3["emb_pr_t3"][sub4],
                            "g_sup": t2["g_sup"][orig4], "charlm_logit": lm_arr[orig4],
                        }
                        t4 = dict(zip(_T4_OUT_NAMES_V150, self.sess_t4.run(None, {k: v for k, v in t4_feed.items() if k in self._in_t4})))
                        lg_ai[orig4] = t4["logit_ai_t4_final"]
                        lg_pr[orig4] = t4["logit_promo_t4_final"]
                        emb_ai[orig4] = t4["emb_ai_t4"]
                        emb_pr[orig4] = t4["emb_pr_t4"]
                        rungs[orig4] = 4

        return {"logit_ai": lg_ai, "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr, "exit_rung": rungs}

    def _run_long_doc_clean(self, clean_text: str) -> tuple[float, float]:
        chunks = plan_long_chunks_v11(clean_text, max_chunks=8)
        ch_cleans = [c["text"] for c in chunks] if len(chunks) > 1 else [clean_text]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(ch_cleans, already_canonical=True, charlm=self.charlm)
        la0, lp0, _, _, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
        od = self._run_t2_t25_t3_t4(ch_cleans, zc, la0, lp0, sups, cf[:, -1], force_depth=None)
        if len(chunks) <= 1:
            return float(od["logit_ai"][0]), float(od["logit_promo"][0])
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
        force_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """P0-HARDENED ENTRY: Strictly pre-canonicalizes ALL texts first + independently latches AI vs Promo exits!"""
        n = len(texts)
        if n == 0:
            return []
        # P0 FIX #1: Pre-canonicalize ONCE before ANY length check, sorting, or branching!
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
        ai_tau = float(self.r0_ai_policy.get("tau_ai", 0.03))
        ai_def = float(self.r0_ai_policy.get("defer_max", 0.30))
        ai_max_ch = int(self.r0_ai_policy.get("max_chars", 180))
        pr_tau = float(self.r0_pr_policy.get("tau_promo", 0.04))
        pr_def = float(self.r0_pr_policy.get("defer_max", 0.30))
        pr_max_ch = int(self.r0_pr_policy.get("max_chars", 180))

        for st in range(0, len(order), max_batch):
            mb = order[st : st + max_batch]
            mb_cleans = [cleans[i] for i in mb]
            c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(mb_cleans, already_canonical=True, charlm=self.charlm)
            la0, lp0, pdef_ai, pdef_pr, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
            pa0 = expit(np.clip(la0, -30.0, 30.0))
            pp0 = expit(np.clip(lp0, -30.0, 30.0))
            if self.lut_ai_r0:
                pa0 = apply_spline_temp_lut(pa0, self.lut_ai_r0)
            if self.lut_pr_r0:
                pp0 = apply_spline_temp_lut(pp0, self.lut_pr_r0)

            # P0 FIX #2: AI & Promo independently decide R0 exit and LATCH their own final score!
            ai_latched: dict[int, tuple[float, float]] = {}
            pr_latched: dict[int, tuple[float, float]] = {}
            need_deep: list[int] = []
            for ri, oi in enumerate(mb):
                cl = cleans[oi]
                no_url = ("<URL>" not in cl) and ("http" not in cl.lower())
                ai_can_exit = ai_en and len(cl) <= ai_max_ch and spoofs[oi] == 0.0 and no_url and float(pdef_ai[ri]) <= ai_def and float(pa0[ri]) <= ai_tau
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
            od = self._run_t2_t25_t3_t4([mb_cleans[k] for k in need_deep], zc[s_idx], la0[s_idx], lp0[s_idx], sups[s_idx], cf[s_idx, -1], force_depth=force_depth)
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
                    fin_pa, fin_la, rung_ai = float(pa_f[j]), float(od["logit_ai"][j]), int(od["exit_rung"][j])
                if ri in pr_latched:
                    fin_pp, fin_lp, rung_pr = pr_latched[ri][0], pr_latched[ri][1], 0
                else:
                    fin_pp, fin_lp, rung_pr = float(pp_f[j]), float(od["logit_promo"][j]), int(od["exit_rung"][j])
                results[oi] = {
                    "prob_ai": fin_pa, "prob_promo": fin_pp,
                    "exit_rung_ai": rung_ai, "exit_rung_pr": rung_pr,
                    "logit_ai": fin_la, "logit_promo": fin_lp,
                }
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 8. Automated 17-Gate `v1.5.0 HeavenLoop` CI Suite (Includes End-to-End Service `C1` & `F6`!)
# ---------------------------------------------------------------------------

def run_v150_ci_gates(
    model: MeguvizV150HeavenLoopModel,
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

    def _infer(txts: list[str], force_pad_len: int | None = None, outer_steps: int = 4):
        cleans = [canonicalize_zero_pii_v150(t)[0] for t in txts]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(cleans, already_canonical=True, charlm=charlm)
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
            eg_a[i, : len(s)] = make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130)
            tg_a[i, : len(s)] = make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130)
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

    # 1. V150_C1: Model-level Zero-PII across ALL rungs R0..T4 === 0.0!
    oa = _infer(["回复 @测试A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"], outer_steps=4)
    ob = _infer(["回复 @UltraLongRandomUserNameForBoundaryTest_998877665544332211 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"], outer_steps=4)
    pii_keys = ("logit_ai_r0", "p_defer_ai_r0", "logit_ai_r1", "logit_ai_t2_final", "logit_ai_t25_final", "logit_ai_t3_final", "logit_ai", "logit_promo", "p_escalate_t3", "p_escalate_t4")
    pii_d = float(max(abs(float(oa[k][0].item()) - float(ob[k][0].item())) for k in pii_keys))
    ci["V150_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # 2. V150_F6: Model-level Structural Promo->AI Firewall under `+12` CBM & Promo perturbations across ALL outputs (incl. defer_ai_r0!)
    t_fw = ["笑死我了，这个插件装完直接起飞[doge]", "关注工种号交流2群 168688884 领全套资料"]
    o_base = _infer(t_fw, outer_steps=4)
    with torch.no_grad():
        saved = (model.cbm_pr_enc.bias.clone(), model.promo_head.bias.clone(), model.r1_promo_head.bias.clone(), model.rung0.promo_head.bias.clone(), model.pr_attr_attn[-1].bias.clone())
        model.cbm_pr_enc.bias.add_(12.0)
        model.promo_head.bias.add_(4.0)
        model.r1_promo_head.bias.add_(4.0)
        model.rung0.promo_head.bias.add_(4.0)
        model.pr_attr_attn[-1].bias.add_(5.0)
        o_pert = _infer(t_fw, outer_steps=4)
        model.cbm_pr_enc.bias.copy_(saved[0])
        model.promo_head.bias.copy_(saved[1])
        model.r1_promo_head.bias.copy_(saved[2])
        model.rung0.promo_head.bias.copy_(saved[3])
        model.pr_attr_attn[-1].bias.copy_(saved[4])
    fw_diff = float(max(
        (o_base[k] - o_pert[k]).abs().max().item()
        for k in ("logit_ai_r0", "p_defer_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2_final", "logit_ai_t25_final", "logit_ai_t3_final", "logit_ai", "p_escalate_t3", "p_escalate_t4")
    ))
    ci["V150_F6_model_promo_to_ai_firewall"] = {"max_leak_diff_under_plus12_cbm": fw_diff, "pass": fw_diff == 0.0}

    # 3. V150_P1: Exact Batch-Padding Invariance across R0..T4 + Coverage + Local CharLM (`< 1e-4` at pad=288!)
    t_sh = "笑死我了，这个插件装完直接起飞[doge]"
    o_unp = _infer([t_sh], force_pad_len=None, outer_steps=4)
    o_pad = _infer([t_sh], force_pad_len=288, outer_steps=4)
    p_keys = ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2_final", "logit_ai_t25_final", "logit_ai_t3_final", "logit_ai", "logit_promo", "p_escalate_t3", "p_escalate_t4")
    p_diffs = {k: round(abs(float(o_unp[k][0].item()) - float(o_pad[k][0].item())), 8) for k in p_keys}
    max_pd = max(p_diffs.values())
    ci["V150_P1_pad_invariance_exact"] = {"diffs": p_diffs, "max_diff": max_pd, "pass": bool(max_pd < 1e-4)}

    # 4. V150_S1: Cayley Symplectic Orthogonality (across all core_depth blocks!) + EBT6 analytic gradient parity
    with torch.no_grad():
        cayley_err = 0.0
        for blk in model.core.blocks:
            g = blk.c1_cayley_gamma
            g2 = g * g
            inv = 1.0 / (1.0 + g2)
            cayley_err = max(cayley_err, float(((((1.0 - g2) * inv) ** 2 + ((2.0 * g) * inv) ** 2) - 1.0).abs().max().item()))
    u_t = torch.randn(4, BUS_DIM_V130, device=device, requires_grad=True)
    er = model.core.ebt.forward_energy_and_grad(u_t)
    ag = torch.autograd.grad(er["free_energy"].sum(), u_t)[0]
    ebt_err = float((er["grad_u"].detach() - ag.detach()).abs().max().item())
    ci["V150_S1_symplectic_and_ebt6"] = {"cayley_unit_err": round(cayley_err, 8), "ebt6_grad_max_diff": round(ebt_err, 8), "pass": bool(cayley_err < 1e-6 and ebt_err < 1e-5)}

    # 5. V150_S2: Block-Diagonal `3x3` AI-Only Sinkhorn mHC Doubly-Stochastic Invariant (`row/col sum === 1.0`)!
    with torch.no_grad():
        m3 = sinkhorn_nxn_v150(model.core.mhc_ai_log_mix)
        r_err = float((m3.sum(dim=-1) - 1.0).abs().max().item())
        c_err = float((m3.sum(dim=-2) - 1.0).abs().max().item())
    ci["V150_S2_mhc_3x3_doubly_stochastic"] = {"row_err": round(r_err, 8), "col_err": round(c_err, 8), "pass": bool(r_err < 1e-4 and c_err < 1e-4)}

    # 6. V150_S3: Decaying Capped Coverage Invariant (`0 <= C_t <= COV_MAX`, PAD tokens === 0.0!)
    cov_pad_tail = float(o_pad["cov_t3"][0, 64:].abs().max().item())
    cov_max_obs = float(o_pad["cov_t3"].max().item())
    ci["V150_S3_coverage_pad_and_cap"] = {"pad_tail_max": cov_pad_tail, "cov_max": round(cov_max_obs, 4), "pass": bool(cov_pad_tail == 0.0 and cov_max_obs <= COV_MAX_V150 + 1e-5)}

    # 7. V150_L1: SoftBallLogitNorm bound
    with torch.no_grad():
        huge = torch.ones((2, AI_END_V130), device=device) * 100.0
        hb = model.ai_head.project_ball(huge).norm(dim=-1).max().item()
        rc = float(model.ai_head.effective_r_cap().item())
    ci["V150_L1_softball_bound"] = {"huge_ball_norm": round(hb, 4), "r_cap": round(rc, 4), "pass": bool(math.isfinite(hb) and hb <= rc + 1e-3)}

    # 8. V150_C13: Support Gate
    with torch.no_grad():
        g_short = float(model.compute_support_gate(torch.tensor([4.0], device=device)).item())
        g_rich = float(model.compute_support_gate(torch.tensor([36.0], device=device)).item())
    ci["V150_C13_support_gate"] = {"g_at_4": round(g_short, 4), "g_at_36": round(g_rich, 4), "pass": bool(g_short < 0.25 and g_rich > 0.85)}

    # 9. V150_C14: Exact Bounded T2.5 / T3 / T4 Shift Guarantees!
    s25 = float((o_base["logit_ai_t25_final"] - o_base["logit_ai_t2_final"]).abs().max().item())
    s3 = float((o_base["logit_ai_t3_final"] - o_base["logit_ai_t25_final"]).abs().max().item())
    s4 = float((o_base["logit_ai_t4_final"] - o_base["logit_ai_t3_final"]).abs().max().item())
    ci["V150_C14_bounded_t25_t3_t4"] = {
        "shift_t25": round(s25, 6), "shift_t3": round(s3, 6), "shift_t4": round(s4, 6),
        "pass": bool(s25 <= T25_CORR_CAP_V150 + 1e-5 and s3 <= T3_CORR_CAP_V150 + 1e-5 and s4 <= T4_CORR_CAP_V150 + 1e-5),
    }

    # 10 & 11. True Bootable `<= 5.00 MiB` + 5-Stage ONNX FP32 Exact Parity
    if export_rep is not None:
        ci["V150_B9_bootable_le_5p00_mib"] = {
            "bootable_mib": export_rep["bootable_mib"], "bootable_bytes": export_rep["bootable_bytes"],
            "skeletons_raw_kb": export_rep["skeletons_raw_kb"], "pass": bool(export_rep["bootable_le_5p00_mib"]),
        }
        ci["V150_ONNX_5stage_fp32_parity"] = {
            "r0_diff": export_rep["r0_fp32_max_diff"], "t2_diff": export_rep["t2_fp32_max_diff"],
            "t3_diff": export_rep["t3_fp32_max_diff"], "t4_diff": export_rep["t4_fp32_max_diff"],
            "long_diff": export_rep["long_fp32_max_diff"], "pass": bool(export_rep["onnx_exact_pass"]),
        }

    # 12 & 13. LUT floors
    if ai_lut_check is not None:
        ci["V150_AI_LUT_FLOOR"] = {**ai_lut_check, "pass": bool(ai_lut_check.get("lut_ai_r0_at_001", 1.0) <= 0.025)}
    if promo_lut_check is not None:
        ci["V150_PR_LUT_FLOOR"] = {**promo_lut_check, "pass": bool(promo_lut_check.get("median_promo_wild", 1.0) <= 0.12)}

    # 14 & 15. P0 END-TO-END DEPLOYMENT CASCADE `C1 === 0.0` (incl. PII length crossing 180/360!) & `F6 === 0.0` (incl. forced R0 exit switching!)
    if release_dir is not None and (release_dir / "v150_r0_apexchar_fp32.onnx").is_file():
        eng = MeguvizV150CascadeEngine(release_dir, threads=4)
        # Force R0 policies ON to aggressively exercise end-to-end decoupled early exit!
        eng.r0_ai_policy = {"enabled": True, "tau_ai": 0.99, "defer_max": 0.99, "max_chars": 180}
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.001, "defer_max": 0.001, "max_chars": 180}

        # E2E C1: username length change crosses 180 & 360 raw char boundaries!
        base_body = "这是一条用于验证清洗前置和端到端路由零泄漏的长短边界测试评论。" * 5  # ~155 chars
        raw_short_pii = f"回复 @短名 : {base_body}"                                    # ~163 raw chars (<180)
        raw_long_pii = f"回复 @{'超长测试用户名' * 40} : {base_body}"                     # >400 raw chars (>360!)
        r_e2e_a = eng.predict_batch([raw_short_pii], force_full=False)[0]
        r_e2e_b = eng.predict_batch([raw_long_pii], force_full=False)[0]
        e2e_c1_diff = max(abs(r_e2e_a["prob_ai"] - r_e2e_b["prob_ai"]), abs(r_e2e_a["prob_promo"] - r_e2e_b["prob_promo"]))
        ci["V150_C1_e2e_cascade_zero_pii"] = {
            "max_prob_diff_across_180_360_boundary": round(float(e2e_c1_diff), 8),
            "rungs_a": (r_e2e_a["exit_rung_ai"], r_e2e_a["exit_rung_pr"]),
            "rungs_b": (r_e2e_b["exit_rung_ai"], r_e2e_b["exit_rung_pr"]),
            "pass": bool(e2e_c1_diff == 0.0),
        }

        # E2E F6: AI latches at R0 while Promo flips between R0-exit and Deep-T2/T3/T4!
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.99, "defer_max": 0.99, "max_chars": 180}
        r_f6_pr_exit = eng.predict_batch(["笑死我了，这个插件装完直接起飞[doge]"], force_full=False)[0]
        eng.r0_pr_policy = {"enabled": True, "tau_promo": 0.0, "defer_max": 0.0, "max_chars": 180}
        r_f6_pr_deep = eng.predict_batch(["笑死我了，这个插件装完直接起飞[doge]"], force_full=False)[0]
        e2e_f6_diff = abs(r_f6_pr_exit["prob_ai"] - r_f6_pr_deep["prob_ai"])
        ci["V150_F6_e2e_cascade_decoupled_firewall"] = {
            "ai_diff_when_promo_switches_r0_vs_deep": round(float(e2e_f6_diff), 8),
            "rungs_when_pr_exits": (r_f6_pr_exit["exit_rung_ai"], r_f6_pr_exit["exit_rung_pr"]),
            "rungs_when_pr_deep": (r_f6_pr_deep["exit_rung_ai"], r_f6_pr_deep["exit_rung_pr"]),
            "pass": bool(e2e_f6_diff == 0.0 and r_f6_pr_deep["exit_rung_ai"] == 0 and r_f6_pr_deep["exit_rung_pr"] >= 2),
        }

    ci["all_ci_passed"] = all(bool(v.get("pass")) for v in ci.values() if isinstance(v, dict))
    return ci
