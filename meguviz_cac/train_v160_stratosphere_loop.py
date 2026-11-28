"""MeguvizCAC v1.6.0-StratosphereLoop Unified Trainer & Locked Unconfounded A0-A5 Ablation Harness.

Implements 100% of the post-External-GPT-audit locked experimental contract:
  1. Primary geometry `v160_a1` (`1L O(L) DWConv+SE+GLU Prelude, H=320, FFN=512, D=2, 14 slots, d_kv=128 (4x32)`).
  2. Round-1 freezes `R0` weights (`--init-r0-ckpt` + `--freeze-r0`) & Engram tables (`expand_tables=False`) across A0..A5!
  3. Exact unbiased Horvitz-Thompson Anytime Depth Loss (`T ~ U{1..6}` with step weight `1 / (7 - t)` -> exact expected
     weight `1/6` per depth `t=1..6`!) + stop-grad tolerance monotonic constraint `ReLU(sg(m_t) - m_{t+1} - eps)` on trusted
     labels ONLY + unweighted proper BCE/KD default (NO survival reweighting, NO Slot-9 `d_t` injection!).
  4. Identical Phase-3 LoopQ W4/W8 QAT schedule across ALL 6 arms, with 3-Checkpoint Quantization Drift Audit:
     (i) FP32/FP16 Master, (ii) Direct-Packed `MGVZ10` (pre-QAT), and (iii) Post-QAT `MGVZ10` across `T1..T6`.
  5. All 6 mandatory Loop-Physics probes:
     - Per-step `Pi_R` saturation rate `T1..T6`
     - Per-step AdaLN modulation gain `T1..T6`
     - Joint `(slots, coverage)` directional Jacobian gain `T1..T6`
     - Period-2 oscillation ratio `||s_t - s_{t-2}|| / ||s_t - s_{t-1}||` (`t=3..6`)
     - Zero-cost `T3-state -> T4-readout` pure-bias control (overall + frozen-R0 surviving hard subset)
     - Frozen-Prelude 3-segment `[head/mid/tail]` order-preserving attribution probe vs textless & shuffled controls!
  6. Clean disjoint `cal_fit` / `cal_cert` split: fit temperature, spline LUTs, strictly monotone OOF linear fusion
     `f = c + w_a*a + w_b*b` (`w_a, w_b >= 0`), and candidate thresholds strictly on `cal_fit`; certify strictly on `cal_cert`
     with exact one-sided Clopper-Pearson binomial bounds (high-side exit remains OFF unless natural-stream FDR UB <= 1%!).
  7. Saves full offline prediction cache (`{version}_eval_cache.npz`) across all validation/test splits for offline A5 policy studies.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import pickle
import random
import time
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import beta as sp_beta
from sklearn.metrics import average_precision_score, log_loss
import torch
import torch.nn.functional as F

from .train_v140_paradise_apex import (
    _eval_bin,
    _load_v140_distill_pool,
    _make_adamw,
    compute_progressive_kd_loss_v140,
)
from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v08_zenith_ops import OfflineEraGRLHead, _GradReverseFn
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d, plan_long_chunks_v11
from .v122_apexnano import fit_output_shifted_promo_lut_v122, fit_smooth_ai_lut_v122
from .v123_data_refinery import NUM_ERAS_V122
from .v130_xtreme_apex import AI_END_V130
from .v140_paradise_apex import CHARLM_FEAT_DIM_V140, HEAD_LEN_V140, MAX_LEN_V140, R0_STRUCT_DIM_V140
from .v140_superteacher import (
    build_v140_superteacher_corpus,
    compute_top_tail_pairwise_loss_v140,
    fit_scalar_temperature_v140,
    jitter_ai_text_v140,
    read_jsonl_any_v140,
    sample_cf_mirror_batch_v140,
    text_sha16_v140,
)
from .v150_heaven_loop import (
    LOCAL_CHARLM_DIM_V150,
    CompactCharLMV150,
    canonicalize_zero_pii_v150,
    encode_bpe_head_tail_with_offsets_v150,
    fit_group_oof_charlm_v150,
)
from .native_frontend import HAS_RUST_FRONTEND, RustTurboFrontend
from .v160_discourse_causal_ops import class_conditional_vrex_unbiased
from .v160_stratosphere_loop import (
    ARCH_PRESETS_V160,
    CONTRACTIVE_INNER_R_V161,
    QUOTE_SCOPE_DIM_V161,
    MeguvizV160CascadeEngine,
    MeguvizV160StratosphereLoopModel,
    _STEP_OUT_NAMES_V160,
    _T1_OUT_NAMES_V160,
    apply_crossview_late_fusion,
    apply_mgvz10_quant_roundtrip_inplace,
    export_v160_bootable_and_onnx,
    extract_apex_char_batch_v160,
    extract_token_quote_scope_4d_v161,
    fit_crossview_eba_h_fusion,
    fit_crossview_late_fusion,
    load_bundle_v10_v160_into_model,
    pack_bundle_v10_v160,
    run_v160_ci_gates,
    set_v160_qat_mode,
    verify_eba_h_fp32_monotonicity,
)


class _StrictGroupOOFFeatureCacheV160:
    """Strictly Leak-Free Feature Cache supporting Rust zero-GIL Rayon frontend + Python fallback."""

    def __init__(
        self,
        tokenizer: BiliPickyBPETokenizer,
        max_len: int,
        head_len: int,
        engram_buckets: int,
        trigram_buckets: int,
        use_local_charlm: bool = True,
        max_entries: int = 1000000,
        r0_b2: int = 32768,
        r0_b3: int = 16384,
        use_rust: bool = True,
        bucket_seq_len: bool = False,
    ) -> None:
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.engram_buckets = int(engram_buckets)
        self.trigram_buckets = int(trigram_buckets)
        self.use_local_charlm = bool(use_local_charlm)
        self.max_entries = int(max_entries)
        self.r0_b2 = int(r0_b2)
        self.r0_b3 = int(r0_b3)
        self.bucket_seq_len = bool(bucket_seq_len)

        self.fold_models: list[CompactCharLMV150] = []
        self.text_to_fold: dict[str, int] = {}
        self.full_lm: CompactCharLMV150 | None = None
        self.rust_fe: RustTurboFrontend | None = None
        if use_rust and HAS_RUST_FRONTEND:
            self.rust_fe = RustTurboFrontend(
                tokenizer, max_len=self.max_len, head_len=self.head_len,
                engram_buckets=self.engram_buckets, trigram_buckets=self.trigram_buckets,
                r0_b2=self.r0_b2, r0_b3=self.r0_b3,
                use_local_charlm=self.use_local_charlm, bucket_seq_len=self.bucket_seq_len,
            )

        self._bpe_cache: dict[str, tuple[str, list[int], list[tuple[int, int]], list[int], list[int], np.ndarray]] = {}
        self._lm_global_cache: dict[tuple[int, str], np.ndarray] = {}
        self._lm_local_cache: dict[tuple[int, str], np.ndarray] = {}
        self._quote_scope_cache: dict[str, np.ndarray] = {}

    def get_quote_scope_4d(
        self, clean_text: str, offsets: list[tuple[int, int]], allow_store: bool = True
    ) -> np.ndarray:
        L = len(offsets)
        arr = self._quote_scope_cache.get(clean_text) if allow_store else None
        if arr is None or arr.shape[0] != L:
            arr = extract_token_quote_scope_4d_v161(clean_text, offsets)
            if allow_store and len(self._quote_scope_cache) < self.max_entries:
                self._quote_scope_cache[clean_text] = arr
        return arr

    def set_group_oof_charlm(
        self, fold_models: list[CompactCharLMV150], text_to_fold: dict[str, int], full_lm: CompactCharLMV150
    ) -> None:
        self.fold_models = fold_models
        self.text_to_fold = text_to_fold
        self.full_lm = full_lm
        self._lm_global_cache.clear()
        self._lm_local_cache.clear()
        if self.rust_fe is not None:
            self.rust_fe.set_group_oof_charlm(fold_models, text_to_fold, full_lm)

    def _select_lm(self, parent_raw_text: str, use_oof: bool) -> tuple[int, CompactCharLMV150 | None]:
        if self.full_lm is None:
            return -1, None
        if use_oof and self.fold_models:
            fid = self.text_to_fold.get(parent_raw_text, -1)
            if 0 <= fid < len(self.fold_models):
                return fid, self.fold_models[fid]
            return -1, self.full_lm
        return -1, self.full_lm

    def get_bpe_style_offsets(self, raw_text: str, allow_store: bool = True):
        hit = self._bpe_cache.get(raw_text)
        if hit is not None:
            return hit
        cl, _ = canonicalize_zero_pii_v150(raw_text)
        _, f56, _ = extract_style_features_56d(cl)
        ids, offs = encode_bpe_head_tail_with_offsets_v150(self.tok, cl, max_len=self.max_len, head_len=self.head_len)
        eng = make_engram_bigram_ids(ids, self.engram_buckets)
        trg = make_engram_trigram_ids(ids, self.trigram_buckets)
        val = (cl, ids, offs, eng, trg, f56.astype(np.float32))
        if allow_store and len(self._bpe_cache) < self.max_entries:
            self._bpe_cache[raw_text] = val
        return val

    def get_global_9d(self, clean_text: str, parent_raw_text: str, use_oof: bool, allow_store: bool) -> np.ndarray:
        fid, lm = self._select_lm(parent_raw_text, use_oof=use_oof)
        if lm is None:
            return np.zeros((CHARLM_FEAT_DIM_V140,), dtype=np.float32)
        ck = (fid, clean_text)
        if allow_store:
            hit = self._lm_global_cache.get(ck)
            if hit is not None:
                return hit
        v = lm.extract_9d([clean_text])[0]
        if allow_store and len(self._lm_global_cache) < self.max_entries:
            self._lm_global_cache[ck] = v
        return v

    def get_local_8ch(
        self, clean_text: str, offsets: list[tuple[int, int]], parent_raw_text: str, use_oof: bool, allow_store: bool
    ) -> np.ndarray:
        L = len(offsets)
        if (not self.use_local_charlm) or self.full_lm is None:
            return np.zeros((L, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        fid, lm = self._select_lm(parent_raw_text, use_oof=use_oof)
        if lm is None:
            return np.zeros((L, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        ck = (fid, clean_text)
        arr = self._lm_local_cache.get(ck) if allow_store else None
        if arr is None or arr.shape[0] != L:
            arr = lm.extract_token_local_8ch(clean_text, offsets)
            if allow_store and len(self._lm_local_cache) < self.max_entries:
                self._lm_local_cache[ck] = arr
        return arr


def _prepare_jittered_texts(
    batch: list[dict[str, Any]],
    rng: random.Random | None,
    punct_jitter_prob: float,
) -> tuple[list[str], list[str], list[bool]]:
    """Run deterministic RNG text jitter on the main thread before handing off to zero-GIL Rust worker threads."""
    t1_list: list[str] = []
    t0_list: list[str] = []
    mut_list: list[bool] = []
    for r in batch:
        t0 = str(r["text"])
        is_ai_pos = (r.get("has_ai", 0.0) > 0.5 and r.get("y_ai", 0.0) >= 0.70) or (
            r.get("has_teacher", 0.0) > 0.5 and r.get("t_ai_lg", -5.0) >= 1.2
        )
        if rng is not None and punct_jitter_prob > 0.0 and is_ai_pos:
            t1 = jitter_ai_text_v140(t0, rng, is_ml_en=bool(r.get("is_ml_en")), prob=0.28 if r.get("is_ml_en") else punct_jitter_prob)
            is_m = (t1 != t0)
        else:
            t1, is_m = t0, False
        t1_list.append(t1)
        t0_list.append(t0)
        mut_list.append(is_m)
    return t1_list, t0_list, mut_list


_META_F32_KEYS_V160 = (
    "t_ai_lg", "t_ai_r0", "t_ai_t1", "t_ai_t2", "t_ai_t3",
    "frontier_gap", "t_pr_lg", "has_t", "kd_w", "y_ai", "has_ai",
    "y_pr", "has_pr", "hard_neg", "hard_fn", "trust_neg",
)
_META_I64_KEYS_V160 = ("sub4", "era5")


def _collate_v160_batch_host(
    batch: list[dict[str, Any]],
    fcache: _StrictGroupOOFFeatureCacheV160,
    r0_b2: int,
    r0_b3: int,
    need_bpe: bool = True,
    use_oof_charlm: bool = True,
    need_quote_scope: bool = False,
    pre_jittered: tuple[list[str], list[str], list[bool]] | None = None,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.35,
    pin_memory: bool = False,
    include_meta: bool = True,
) -> dict[str, torch.Tensor]:
    if pre_jittered is not None:
        t1_list, t0_list, mut_list = pre_jittered
    else:
        t1_list, t0_list, mut_list = _prepare_jittered_texts(batch, rng, punct_jitter_prob)

    def _to_host(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr)
        return t.pin_memory() if pin_memory else t

    out: dict[str, torch.Tensor] = {}
    if fcache.rust_fe is not None and fcache.r0_b2 == int(r0_b2) and fcache.r0_b3 == int(r0_b3):
        rs_d = fcache.rust_fe.extract_batch_numpy(
            t1_list, parent_raw_texts=t0_list,
            need_bpe=need_bpe, use_oof_charlm=use_oof_charlm, need_quote_scope=need_quote_scope,
        )
        for k, arr in rs_d.items():
            out[k] = _to_host(arr)
    else:
        cleans: list[str] = []
        g9_list: list[np.ndarray] = []
        bpe_pack = []
        for t1, t0, is_m in zip(t1_list, t0_list, mut_list):
            cl, ids, offs, eng, trg, f56 = fcache.get_bpe_style_offsets(t1, allow_store=not is_m)
            cleans.append(cl)
            g9_list.append(fcache.get_global_9d(cl, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m))
            if need_bpe:
                loc8 = fcache.get_local_8ch(cl, offs, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m)
                qsc4 = fcache.get_quote_scope_4d(cl, offs, allow_store=not is_m) if need_quote_scope else None
                bpe_pack.append((ids, eng, trg, f56, loc8, qsc4))

        g9_arr = np.stack(g9_list, axis=0).astype(np.float32) if g9_list else np.zeros((0, CHARLM_FEAT_DIM_V140), dtype=np.float32)
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
            cleans, b2=r0_b2, b3=r0_b3, already_canonical=True, precomputed_charlm_9d=g9_arr
        )
        out["c1"] = _to_host(c1)
        out["c2"] = _to_host(c2)
        out["c3"] = _to_host(c3)
        out["cm"] = _to_host(cm)
        out["cf"] = _to_host(cf)
        out["sup"] = _to_host(sups)
        if need_bpe:
            ml = max(2, max(len(x[0]) for x in bpe_pack))
            if ml % 2 == 1:
                ml += 1
            nb = len(bpe_pack)
            ids_a = np.full((nb, ml), fcache.tok.pad_id, dtype=np.int64)
            msk_a = np.zeros((nb, ml), dtype=np.int64)
            eg_a = np.zeros((nb, ml), dtype=np.int64)
            tg_a = np.zeros((nb, ml), dtype=np.int64)
            loc_a = np.zeros((nb, ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
            qsc_a = np.zeros((nb, ml, QUOTE_SCOPE_DIM_V161), dtype=np.float32) if need_quote_scope else None
            f56_l = []
            for i, (ids, eng, trg, f56, loc8, qsc4) in enumerate(bpe_pack):
                ns = len(ids)
                ids_a[i, :ns], msk_a[i, :ns], eg_a[i, :ns], tg_a[i, :ns] = ids, 1, eng, trg
                loc_a[i, :ns] = loc8
                if qsc_a is not None and qsc4 is not None:
                    qsc_a[i, :ns] = qsc4
                f56_l.append(f56)
            out["ids"] = _to_host(ids_a)
            out["msk"] = _to_host(msk_a)
            out["eg"] = _to_host(eg_a)
            out["tg"] = _to_host(tg_a)
            out["loc"] = _to_host(loc_a)
            out["st"] = _to_host(np.stack(f56_l, 0).astype(np.float32))
            if qsc_a is not None:
                out["qsc"] = _to_host(qsc_a)

    if include_meta:
        nb = len(batch)
        mf = np.empty((16, nb), dtype=np.float32)
        mi = np.empty((2, nb), dtype=np.int64)
        for i, r in enumerate(batch):
            t_lg = float(r.get("t_ai_lg", 0.0))
            has_a = float(r.get("has_ai", 0.0))
            ya_v = float(r.get("y_ai", 0.0))
            mf[0, i] = t_lg
            mf[1, i] = float(r.get("t_ai_r0", t_lg))
            mf[2, i] = float(r.get("t_ai_t1", t_lg))
            mf[3, i] = float(r.get("t_ai_t2", t_lg))
            mf[4, i] = float(r.get("t_ai_t3", t_lg))
            mf[5, i] = float(r.get("frontier_gap", 0.0))
            mf[6, i] = float(r.get("t_pr_lg", 0.0))
            mf[7, i] = float(r.get("has_teacher", r.get("has_t", 0.0)))
            mf[8, i] = float(r.get("kd_w", 1.0))
            mf[9, i] = ya_v
            mf[10, i] = has_a
            mf[11, i] = float(r.get("y_pr", 0.0))
            mf[12, i] = float(r.get("has_pr", 0.0))
            mf[13, i] = float(r.get("hard_neg", 0.0))
            mf[14, i] = float(r.get("hard_fn", 0.0))
            mf[15, i] = float(r.get("trust_neg", 1.0 if (has_a > 0.5 and ya_v <= 0.25) else 0.0))
            mi[0, i] = int(r.get("sub4", -1))
            mi[1, i] = int(r.get("era5", -1))
        out["_meta_f32"] = _to_host(mf)
        out["_meta_i64"] = _to_host(mi)
    return out


def _h2d_batch(host_d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    nb = device.type == "cuda"
    out: dict[str, torch.Tensor] = {}
    for k, v in host_d.items():
        if k == "_meta_f32":
            mg = v.to(device, non_blocking=nb)
            for idx, mk in enumerate(_META_F32_KEYS_V160):
                out[mk] = mg[idx]
        elif k == "_meta_i64":
            mi = v.to(device, non_blocking=nb)
            for idx, mk in enumerate(_META_I64_KEYS_V160):
                out[mk] = mi[idx]
        else:
            out[k] = v.to(device, non_blocking=nb) if nb else v.to(device)
    return out


def _collate_v160_batch(
    batch: list[dict[str, Any]],
    fcache: _StrictGroupOOFFeatureCacheV160,
    r0_b2: int,
    r0_b3: int,
    device: torch.device,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.35,
    need_bpe: bool = True,
    use_oof_charlm: bool = True,
    need_quote_scope: bool = False,
    include_meta: bool = True,
) -> dict[str, torch.Tensor]:
    host_d = _collate_v160_batch_host(
        batch, fcache, r0_b2, r0_b3,
        need_bpe=need_bpe, use_oof_charlm=use_oof_charlm, need_quote_scope=need_quote_scope,
        rng=rng, punct_jitter_prob=punct_jitter_prob, pin_memory=(device.type == "cuda"),
        include_meta=include_meta,
    )
    return _h2d_batch(host_d, device)


# ---------------------------------------------------------------------------
# Turbo v3 Zero-Sync Vectorized GPU Loss Primitives (0 CUDA sync stalls/step!)
# ---------------------------------------------------------------------------

def _top_tail_pairwise_nosync(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    hard_neg_boost: torch.Tensor | None = None,
    margin: float = 1.8,
    top_neg_frac: float = 0.25,
) -> torch.Tensor:
    """Zero-sync exact equivalent of `compute_top_tail_pairwise_loss_v140` supporting `(B,)` or `(T, B)`."""
    lg_f = logits.float()
    b = lg_f.shape[-1]
    dev = lg_f.device
    m = valid_mask > 0.5
    pos_m = m & (targets >= 0.70)
    neg_m = m & (targets <= 0.25)

    n_val = m.sum()
    n_pos = pos_m.sum()
    n_neg = neg_m.sum()

    max_k = max(1, min(b, int(math.ceil(b * float(top_neg_frac)))))
    neg_fill = torch.full((), -1e9, device=dev, dtype=torch.float32)
    z_neg_masked = torch.where(neg_m, lg_f, neg_fill)
    top_vals, top_idx = torch.topk(z_neg_masked, k=max_k, dim=-1, largest=True)

    k_real = torch.ceil(n_neg.float() * float(top_neg_frac)).long().clamp(min=1)
    k_real = torch.minimum(k_real, n_neg.clamp_min(1))
    rank = torch.arange(max_k, device=dev, dtype=torch.long)
    top_neg_m = (rank < k_real) & (n_neg > 0) & (n_val >= 4) & (n_pos > 0)
    top_vals_safe = torch.where(top_neg_m, top_vals, torch.zeros_like(top_vals))

    diff = lg_f.unsqueeze(-1) - top_vals_safe.unsqueeze(-2)  # (..., B, max_k)
    pair_loss = F.softplus(float(margin) - diff)
    if hard_neg_boost is not None:
        hn_f = hard_neg_boost.float()
        if lg_f.ndim == 2:
            hn_exp = hn_f.unsqueeze(0).expand(lg_f.shape[0], -1)
        else:
            hn_exp = hn_f
        hn_top = torch.gather(hn_exp, -1, top_idx)
        pair_loss = pair_loss * (1.0 + 0.85 * hn_top).unsqueeze(-2)

    pair_m = (pos_m.unsqueeze(-1) & top_neg_m.unsqueeze(-2)).float()
    num = (pair_loss * pair_m).sum(dim=(-2, -1))
    den = pair_m.sum(dim=(-2, -1)).clamp_min(1.0)
    return num / den


def _progressive_kd_loss_nosync(
    lg: torch.Tensor,
    bk: dict[str, torch.Tensor],
    axis: str = "ai",
    teacher_key: str = "t_ai_lg",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-sync exact equivalent of `compute_progressive_kd_loss_v140` supporting `(B,)` or batched `(T, B)`."""
    if axis == "ai":
        t_lg, y_g, has_g, hn = bk[teacher_key], bk["y_ai"], bk["has_ai"], bk["hard_neg"]
    else:
        t_lg, y_g, has_g, hn = bk["t_pr_lg"], bk["y_pr"], bk["has_pr"], bk["hard_neg"]

    has_t, kd_w = bk["has_t"], bk["kd_w"]
    sw = sample_weight.float() if sample_weight is not None else torch.ones_like(kd_w)
    lg_f = lg.float()

    tm = has_t > 0.5
    tm_f = tm.float()
    tm_den = tm_f.sum().clamp_min(1.0)
    t_c = t_lg.clamp(-8.0, 8.0)
    w_t = kd_w * sw * tm_f
    t_c_b = t_c.unsqueeze(0).expand_as(lg_f) if lg_f.ndim == 2 else t_c
    l_mse = (F.smooth_l1_loss(lg_f, t_c_b, reduction="none", beta=1.0) * w_t).sum(dim=-1) / tm_den
    t_p2 = torch.sigmoid(t_c_b / 1.5)
    l_soft = ((F.binary_cross_entropy_with_logits(lg_f / 1.5, t_p2, reduction="none") * w_t).sum(dim=-1) / tm_den) * 2.25
    loss = 0.55 * l_mse + 0.45 * l_soft

    gm = has_g > 0.5
    gm_f = gm.float()
    gm_den = gm_f.sum().clamp_min(1.0)
    yg_s = y_g * 0.96 + 0.02
    yg_s_b = yg_s.unsqueeze(0).expand_as(lg_f) if lg_f.ndim == 2 else yg_s
    w_g = (1.0 + 0.50 * hn) * sw * gm_f
    l_g = (F.binary_cross_entropy_with_logits(lg_f, yg_s_b, reduction="none") * w_g).sum(dim=-1) / gm_den
    loss = loss + 0.65 * l_g

    eff_y = torch.where(gm, y_g, torch.sigmoid(t_lg))
    eff_v = (gm | (tm & ((t_lg >= 1.4) | (t_lg <= -1.4)))).float()
    l_tail = _top_tail_pairwise_nosync(
        lg_f, eff_y, eff_v, hard_neg_boost=hn, margin=1.8 if axis == "ai" else 1.7, top_neg_frac=0.25
    )
    loss = loss + (0.32 if axis == "ai" else 0.28) * l_tail

    hnm_f = ((eff_v > 0.5) & (eff_y <= 0.20) & (hn > 0.5)).float()
    hnm_den = hnm_f.sum().clamp_min(1.0)
    if axis == "ai":
        loss = loss + 0.30 * ((F.relu(lg_f + 0.95).pow(2) * sw * hnm_f).sum(dim=-1) / hnm_den)
    else:
        loss = loss + 0.18 * ((F.relu(lg_f + 0.80).pow(2) * sw * hnm_f).sum(dim=-1) / hnm_den)
    return loss


class _PreCollatedSplitV160:
    """Pre-collated static eval split (length-sorted batches) reused across all evals & Auto-Ship-Gate!"""

    def __init__(
        self,
        texts: list[str],
        fcache: _StrictGroupOOFFeatureCacheV160,
        r0_b2: int,
        r0_b3: int,
        device: torch.device,
        batch_size: int = 512,
        r0_only: bool = False,
        need_quote_scope: bool = False,
        cache_on_device: bool = True,
    ) -> None:
        self.n = len(texts)
        self.r0_only = bool(r0_only)
        self.batches: list[tuple[np.ndarray, dict[str, torch.Tensor]]] = []
        if self.n == 0:
            return
        order = sorted(range(self.n), key=lambda i: len(texts[i]))
        pin_m = (device.type == "cuda") and (not cache_on_device)
        for st in range(0, self.n, batch_size):
            idxs = order[st : st + batch_size]
            sub_b = [{"text": texts[i]} for i in idxs]
            host_d = _collate_v160_batch_host(
                sub_b, fcache, r0_b2, r0_b3,
                need_bpe=not r0_only, use_oof_charlm=False, need_quote_scope=need_quote_scope,
                rng=None, punct_jitter_prob=0.0, pin_memory=pin_m, include_meta=False,
            )
            bk = _h2d_batch(host_d, device) if cache_on_device else host_d
            self.batches.append((np.asarray(idxs, dtype=np.int64), bk))
        self.on_device = bool(cache_on_device)


@torch.inference_mode()
def predict_student_v160(
    model: MeguvizV160StratosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV160,
    texts: list[str],
    device: torch.device,
    batch_size: int = 256,
    r0_only: bool = False,
    outer_steps: int = 6,
    use_crossview_dt: bool = False,
    r0_drop_eval: bool = False,
    record_diagnostics: bool = False,
    precollated: _PreCollatedSplitV160 | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    n = precollated.n if precollated is not None else len(texts)
    keys = (
        ("logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0")
        if r0_only
        else (
            "logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0",
            "logit_ai_r1", "logit_promo_r1", "logit_ai", "logit_promo",
            *[f"logit_ai_t{k}" for k in range(1, outer_steps + 1)],
            *[f"logit_promo_t{k}" for k in range(1, outer_steps + 1)],
            *[f"u_halt_t{k}" for k in range(1, outer_steps + 1)],
            *(("logit_ai_t3_state_t4_readout",) if (record_diagnostics and outer_steps >= 3) else ()),
        )
    )
    out: dict[str, Any] = {k: np.zeros((n,), dtype=np.float32) for k in keys}
    if n == 0:
        return out
    model.eval()

    sat_acc: list[list[float]] = [[] for _ in range(outer_steps)]
    adaln_acc: list[list[float]] = [[] for _ in range(outer_steps)]
    d1_acc: dict[int, list[float]] = {t: [] for t in range(2, outer_steps + 1)}
    d2_acc: dict[int, list[float]] = {t: [] for t in range(3, outer_steps + 1)}

    if precollated is not None:
        batch_iter = precollated.batches
        already_on_dev = precollated.on_device
    else:
        order = sorted(range(n), key=lambda i: len(texts[i]))
        batch_iter = [
            (
                np.asarray(order[st : st + batch_size], dtype=np.int64),
                _collate_v160_batch(
                    [{"text": texts[i]} for i in order[st : st + batch_size]],
                    fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0,
                    need_bpe=not r0_only, use_oof_charlm=False, need_quote_scope=bool(model.use_discourse_ops),
                    include_meta=False,
                ),
            )
            for st in range(0, n, batch_size)
        ]
        already_on_dev = True

    amp_en = bool(use_amp and device.type == "cuda")
    for idxs_np, bk_raw in batch_iter:
        bk = bk_raw if already_on_dev else _h2d_batch(bk_raw, device)
        with torch.amp.autocast("cuda", enabled=amp_en, dtype=amp_dtype):
            if r0_only:
                fn_r0 = model._compiled_r0 if (model._compiled_r0 is not None and not record_diagnostics) else model.forward_r0
                res = fn_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
            else:
                res = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                    outer_steps=outer_steps,
                    r0_drop_p=1.0 if r0_drop_eval else 0.0,
                    dt_drop_p=1.0 if r0_drop_eval else 0.0,
                    use_crossview_dt=use_crossview_dt and (not r0_drop_eval),
                    return_diagnostics=record_diagnostics,
                    quote_scope=bk.get("qsc"),
                )
        if record_diagnostics and not r0_only:
            n_ai = model.core.n_ai_slots
            s_list = res["slot_steps"]
            if res.get("proj_sat_steps"):
                sat_cat = torch.stack(res["proj_sat_steps"], dim=0).float().cpu().numpy()
                ad_cat = torch.stack(res["adaln_gain_steps"], dim=0).float().cpu().numpy()
                for ti in range(sat_cat.shape[0]):
                    sat_acc[ti].extend(sat_cat[ti].tolist())
                    adaln_acc[ti].append(float(ad_cat[ti]))
            for t in range(2, len(s_list) + 1):
                d1 = (s_list[t - 1][:, :n_ai] - s_list[t - 2][:, :n_ai]).flatten(1).norm(dim=1)
                d1_acc[t].extend(d1.float().cpu().tolist())
            for t in range(3, len(s_list) + 1):
                d2 = (s_list[t - 1][:, :n_ai] - s_list[t - 3][:, :n_ai]).flatten(1).norm(dim=1)
                d2_acc[t].extend(d2.float().cpu().tolist())

        # Single stacked D2H transfer + vectorized scatter (replaces 25x Python scatter loops!)
        present_keys = [k for k in keys if k in res]
        if present_keys:
            stacked = torch.stack([res[k].float() for k in present_keys], dim=0).cpu().numpy()
            for ki, k in enumerate(present_keys):
                out[k][idxs_np] = stacked[ki]

    if record_diagnostics and not r0_only:
        out["_proj_sat_rate_t1_t6"] = {f"T{ti+1}": round(float(np.mean(sat_acc[ti])), 5) if sat_acc[ti] else 0.0 for ti in range(outer_steps)}
        out["_adaln_gain_t1_t6"] = {f"T{ti+1}": round(float(np.mean(adaln_acc[ti])), 5) if adaln_acc[ti] else 0.0 for ti in range(outer_steps)}
        p2: dict[str, Any] = {}
        for t in range(2, outer_steps + 1):
            m1 = float(np.mean(d1_acc[t])) if d1_acc[t] else 0.0
            entry: dict[str, Any] = {"norm_s_t_minus_s_tm1": round(m1, 5)}
            if t >= 3 and d2_acc[t]:
                m2 = float(np.mean(d2_acc[t]))
                entry["norm_s_t_minus_s_tm2"] = round(m2, 5)
                entry["period2_ratio_d2_over_d1"] = round(m2 / max(1e-6, m1), 4)
            p2[f"T{t}"] = entry
        if d1_acc.get(3) and d1_acc.get(4):
            out["_state_delta_ratio_s4_over_s3"] = round(float(np.mean(d1_acc[4])) / max(1e-6, float(np.mean(d1_acc[3]))), 4)
        out["_period2_oscillation"] = p2
    return out


@torch.no_grad()
def measure_joint_directional_jacobian_v160(
    model: MeguvizV160StratosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV160,
    texts: list[str],
    device: torch.device,
    outer_steps: int = 6,
    n_dirs: int = 3,
    eps: float = 1e-3,
) -> dict[str, float]:
    """External GPT Section 2 probe: empirical directional Jacobian gain over joint `(ai_streams, cov)` state across T1..T6!"""
    if not texts:
        return {}
    model.eval()
    sub = [{"text": t} for t in texts[:32]]
    bk = _collate_v160_batch(
        sub, fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0,
        need_bpe=True, use_oof_charlm=False, need_quote_scope=bool(model.use_discourse_ops),
    )
    r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
    s1 = model.forward_prelude_and_t1(
        bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
        z_char=r0["z_char"], support_pairs=bk["sup"], quote_scope=bk.get("qsc"),
    )
    B = bk["ids"].shape[0]
    cur_ai, cur_pr, cur_cov = s1["ai_streams"], s1["pr_stream"], s1["cov"]
    cur_lg = s1["logit_ai_t1"]
    cur_emb = s1["emb_ai_t1"]
    z_b = torch.zeros((B,), device=device, dtype=cur_ai.dtype)
    gains: dict[str, float] = {}
    gen = torch.Generator(device=device).manual_seed(16001)

    for ti in range(1, outer_steps):
        t_vec = torch.full((B,), float(ti), device=device, dtype=torch.float32)
        base = model.core.forward_step(
            ai_streams=cur_ai, pr_stream=cur_pr, init_slots=s1["init_slots"], cov_prev=cur_cov,
            k_ev_0=s1["k_ev_0"], v_ev_0=s1["v_ev_0"], mask=bk["msk"],
            k_haar=s1["k_haar"], v_haar=s1["v_haar"], haar_mask=s1["haar_mask"],
            h0_pool=s1["h_pool_0"], g_sup=s1["g_sup"], t_val=t_vec, disagree_dt=z_b, abs_b_prev=cur_lg.abs(),
        )
        dir_gains: list[float] = []
        for _ in range(n_dirs):
            u_ai = torch.randn(cur_ai.shape, generator=gen, device=device, dtype=cur_ai.dtype)
            u_cov = torch.randn(cur_cov.shape, generator=gen, device=device, dtype=cur_cov.dtype) * bk["msk"].to(cur_cov.dtype)
            u_norm = torch.sqrt(u_ai.flatten(1).pow(2).sum(dim=1) + u_cov.flatten(1).pow(2).sum(dim=1)).clamp_min(1e-6)
            u_ai = u_ai / u_norm.view(B, 1, 1, 1)
            u_cov = u_cov / u_norm.view(B, 1)
            pert = model.core.forward_step(
                ai_streams=cur_ai + eps * u_ai, pr_stream=cur_pr, init_slots=s1["init_slots"],
                cov_prev=(cur_cov + eps * u_cov).clamp_min(0.0),
                k_ev_0=s1["k_ev_0"], v_ev_0=s1["v_ev_0"], mask=bk["msk"],
                k_haar=s1["k_haar"], v_haar=s1["v_haar"], haar_mask=s1["haar_mask"],
                h0_pool=s1["h_pool_0"], g_sup=s1["g_sup"], t_val=t_vec, disagree_dt=z_b, abs_b_prev=cur_lg.abs(),
            )
            d_ai = (pert["ai_streams"] - base["ai_streams"]).flatten(1).pow(2).sum(dim=1)
            d_cov = (pert["cov"] - base["cov"]).flatten(1).pow(2).sum(dim=1)
            g_b = (torch.sqrt(d_ai + d_cov) / eps).float().mean().item()
            dir_gains.append(float(g_b))
        gains[f"T{ti+1}"] = round(float(np.mean(dir_gains)), 4)
        cur_ai, cur_pr, cur_cov = base["ai_streams"], base["pr_stream"], base["cov"]
        ro_n = model._readout_step(base["slots"], s1["h_pool_0"], s1["pr_attr_0"], s1["st_ai"], s1["st_pr"], s1["_zc_eff"], base["step_phi"])
        cur_lg, cur_emb = ro_n["logit_ai"], ro_n["emb_ai"]
    return gains


@torch.no_grad()
def run_order_preserving_3seg_probe_v160(
    model: MeguvizV160StratosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV160,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """External GPT Section 1 lightweight probe on frozen Prelude tokens:
    Splits valid tokens into `[head, mid, tail]` 3 segments (order-preserving) and compares 5-fold OOF Ridge/Logistic
    AP against (a) equal-dim textless zero control and (b) randomly segment-order-shuffled control!
    """
    if len(rows) < 32:
        return {"skipped": True, "reason": "too_few_rows"}
    model.eval()
    D = model.dim
    feats_ord = np.zeros((len(rows), D * 3), dtype=np.float32)
    feats_mean = np.zeros((len(rows), D), dtype=np.float32)
    y_ai = np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in rows], dtype=np.int32)
    if y_ai.sum() < 8 or (len(y_ai) - y_ai.sum()) < 8:
        return {"skipped": True, "reason": "single_class"}

    bs = 128
    for st in range(0, len(rows), bs):
        sub = rows[st : st + bs]
        bk = _collate_v160_batch(
            sub, fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0,
            need_bpe=True, use_oof_charlm=False, need_quote_scope=bool(model.use_discourse_ops),
        )
        r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        s1 = model.forward_prelude_and_t1(
            bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
            z_char=r0["z_char"], support_pairs=bk["sup"], return_diagnostics=True, quote_scope=bk.get("qsc"),
        )
        ht = s1["h_tok"].float().cpu().numpy()
        mk = bk["msk"].cpu().numpy()
        for j in range(len(sub)):
            L_v = max(1, int(mk[j].sum()))
            h_v = ht[j, :L_v, :]
            feats_mean[st + j] = h_v.mean(axis=0)
            c1_i = max(1, L_v // 3)
            c2_i = max(c1_i, (2 * L_v) // 3)
            seg0 = h_v[:c1_i].mean(axis=0)
            seg1 = h_v[c1_i:c2_i].mean(axis=0) if c2_i > c1_i else seg0
            seg2 = h_v[c2_i:].mean(axis=0) if L_v > c2_i else seg1
            feats_ord[st + j] = np.concatenate([seg0, seg1, seg2], axis=0)

    # Shuffled segment order control (destroys head/mid/tail order while keeping exact segment content & dim!)
    rng_sh = np.random.default_rng(16002)
    feats_shuf = feats_ord.copy()
    for i in range(len(rows)):
        perm = rng_sh.permutation(3)
        segs = [feats_ord[i, k * D : (k + 1) * D] for k in perm]
        feats_shuf[i] = np.concatenate(segs, axis=0)
    feats_zero = np.zeros_like(feats_ord)

    def _oof_ridge_pr(X: np.ndarray, y: np.ndarray, lam: float = 50.0) -> float:
        n = len(y)
        oof = np.zeros((n,), dtype=np.float32)
        fold = np.arange(n) % 5
        for f in range(5):
            va = fold == f
            tr = ~va
            Xtr = X[tr]
            mu = Xtr.mean(axis=0, keepdims=True)
            Xc = Xtr - mu
            y_c = y[tr].astype(np.float32) - y[tr].mean()
            # Dual kernel ridge for speed when dim > n or primal when n >= dim
            d = Xc.shape[1]
            if Xc.shape[0] >= d:
                w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d, dtype=np.float32), Xc.T @ y_c)
                oof[va] = (X[va] - mu) @ w + float(y[tr].mean())
            else:
                alpha = np.linalg.solve(Xc @ Xc.T + lam * np.eye(Xc.shape[0], dtype=np.float32), y_c)
                oof[va] = ((X[va] - mu) @ Xc.T) @ alpha + float(y[tr].mean())
        return round(float(average_precision_score(y, oof)), 4)

    pr_mean = _oof_ridge_pr(feats_mean, y_ai)
    pr_ord = _oof_ridge_pr(feats_ord, y_ai)
    pr_shuf = _oof_ridge_pr(feats_shuf, y_ai)
    pr_zero = _oof_ridge_pr(feats_zero, y_ai)
    return {
        "n_rows": len(rows),
        "global_mean_oof_pr": pr_mean,
        "ordered_3seg_oof_pr": pr_ord,
        "shuffled_3seg_oof_pr": pr_shuf,
        "textless_zero_oof_pr": pr_zero,
        "ordered_minus_shuffled": round(pr_ord - pr_shuf, 4),
        "ordered_minus_global_mean": round(pr_ord - pr_mean, 4),
    }


def _clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound at level `1 - alpha` (`k=0 -> 1 - alpha^(1/n)`)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - (alpha ** (1.0 / float(n))))
    return float(sp_beta.ppf(1.0 - alpha, k + 1, n - k))


def _clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound at level `1 - alpha`."""
    if n <= 0 or k <= 0:
        return 0.0
    return float(sp_beta.ppf(alpha, k, n - k + 1))


def _safe_bce(y_bin: np.ndarray, logits: np.ndarray) -> float:
    if len(y_bin) == 0:
        return 0.0
    p = np.clip(expit(np.clip(logits, -30.0, 30.0)), 1e-6, 1.0 - 1e-6)
    return round(float(log_loss(y_bin, p, labels=[0, 1])), 5)


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.6.0-StratosphereLoop Trainer & Locked A0-A5 Ablation Harness")
    ap.add_argument("--distill-pool", type=Path, required=True)
    ap.add_argument("--sibling-dir", type=Path, required=True)
    ap.add_argument("--gold-v2-dir", type=Path, required=True)
    ap.add_argument("--wave3-dir", type=Path, required=True)
    ap.add_argument("--v11-data-dir", type=Path, required=True)
    ap.add_argument("--v07-tokenizer", type=Path, required=True)
    ap.add_argument("--delta-dir", type=Path, default=None)
    ap.add_argument("--v04-data", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.6.0-A1")
    ap.add_argument("--arch", type=str, default="v160_a1", choices=sorted(ARCH_PRESETS_V160.keys()))
    ap.add_argument("--dynamics", type=str, default="stratosphere", choices=["stratosphere", "contractive_d", "v150_anchor"])
    ap.add_argument("--use-discourse-ops", action="store_true", help="v1.6.1-B: Enable +0.33ms SelectiveOrder3CumSum + CheapD08PositionMoments + Quote FSM")
    ap.add_argument("--lardo-w8-core", action="store_true", help="v1.6.1 LARDO: Promote core.* + coda + haar from W4 to W8 inside MGVZ10 (<= 5.00 MiB)")
    ap.add_argument("--init-from-fp16", type=Path, default=None, help="v1.6.1 warm-start from FP16 master checkpoint (shape-safe partial load)")
    ap.add_argument("--wise-anchor-fp16", type=Path, default=None, help="Optional FP16 anchor checkpoint included in Cand-3 WiSE-FT Model Soup")
    ap.add_argument("--t-train-mode", type=str, default="uniform_1_6", choices=["uniform_1_6", "fixed_2_4"])
    ap.add_argument("--use-step-phi", action="store_true", help="Enable continuous Fourier step embedding phi(t) (Arm A2)")
    ap.add_argument("--no-step-phi", action="store_true", help="Explicitly disable Fourier step embedding (A0/A1/A3/A4/A5)")
    ap.add_argument("--step-decay-mode", type=str, default="inv_sqrt", choices=["inv_sqrt", "const"])
    ap.add_argument("--use-slot9-dt", action="store_true", help="Legacy opt-in ONLY: inject d_t into AI Slot-9 (OFF in A0-A5)")
    ap.add_argument("--expand-tables", action="store_true", help="Round-2 opt-in ONLY: 2x R0++/Engram lookup tables (OFF in Round-1)")
    ap.add_argument("--init-r0-ckpt", type=Path, default=None, help="Load shared frozen Phase-1 R0 state_dict across A0-A5")
    ap.add_argument("--save-r0-ckpt", type=Path, default=None, help="Save trained Phase-1 R0 state_dict for reuse across A0-A5")
    ap.add_argument("--freeze-r0", action="store_true", help="Freeze R0 weights during Phase-2 & Phase-3 (locked A0-A5 protocol)")
    ap.add_argument("--p1-use-cf", action="store_true", help="v1.6.1: Include CF mirrors, L_inv, and gentle Softplus-CVaR inside Phase-1 R0++")
    ap.add_argument("--no-prelude-global-se", action="store_true")
    ap.add_argument("--no-coverage-beta", action="store_true")
    ap.add_argument("--no-local-charlm", action="store_true")
    ap.add_argument("--no-mono-loss", action="store_true")
    ap.add_argument("--mono-eps", type=float, default=0.05)
    ap.add_argument("--use-ncl", action="store_true", help="Enable NCL falsification arm (Arm A3 ONLY)")
    ap.add_argument("--ncl-gamma", type=float, default=0.12)
    ap.add_argument("--use-softplus-cvar", action="store_true", help="Rev.2 zero-inference-cost negative-class softplus CVaR tail penalty")
    ap.add_argument("--cvar-alpha", type=float, default=0.08)
    ap.add_argument("--cvar-weight", type=float, default=0.14)
    ap.add_argument("--recall-floor-weight", type=float, default=0.0, help="v1.6.1: Positive stealth-AI tail recall-floor pairwise protection weight")
    ap.add_argument("--use-fmt-cf", action="store_true", help="Rev.2 zero-inference-cost rule-based human format counterfactual consistency L_inv")
    ap.add_argument("--fmt-cf-weight", type=float, default=0.12)
    ap.add_argument("--use-promo-flip-cf", action="store_true", help="v1.6.1: Rule-based Promo stance-flip counterfactual L_flip")
    ap.add_argument("--promo-flip-weight", type=float, default=0.06)
    ap.add_argument("--use-cc-vrex", action="store_true", help="v1.6.1: Review 04 P4 exact unbiased Class-Conditional V-REx across era5")
    ap.add_argument("--cc-vrex-weight", type=float, default=0.08)
    ap.add_argument("--sat-reg-weight", type=float, default=0.0, help="v1.6.1-B: Inner-ball saturation regularizer weight for contractive_d")
    ap.add_argument("--extra-gold-pack", type=Path, default=None, help="Optional extra zero-leakage clean gold pack (e.g. rescored DuckDB 2.0 stream pack)")
    ap.add_argument("--extra-cal-cert-pack", type=Path, default=None, help="Optional disjoint zero-leakage DuckDB 2.0 calibration certification holdout pack")
    ap.add_argument("--oof-cal-folds", type=int, default=1, help="1 = legacy 50/50 split; >=2 (e.g. 5) = strict K-fold OOF cross-fitted calibration & R0 LTT")
    ap.add_argument("--auto-ship-gate", action="store_true", help="v1.6.1: 3-candidate auto ship gate (Direct-Pack LARDO-W8 vs Light-QAT vs FP16 WiSE-FT Soup)")
    ap.add_argument("--qat-lr", type=float, default=2.0e-4, help="Phase-3 QAT base LR (v1.6.1 Light-QAT recommends 5e-5)")
    ap.add_argument("--p2-lr", type=float, default=1.1e-3, help="Phase-2 Deep base LR (v1.6.1 warm-start recommends 4e-4)")
    ap.add_argument("--use-cascade-weight", action="store_true", help="Optional survival reweighting (OFF in locked A0-A5)")
    ap.add_argument("--no-crossview-fusion", action="store_true", help="Disable monotone OOF cross-view late fusion")
    ap.add_argument("--two-sided-r0-exit", action="store_true", help="Test exact Clopper-Pearson high-side R0 exit certification")
    ap.add_argument("--skip-qat", action="store_true")
    ap.add_argument("--ablation-fast", action="store_true", help="Controlled 10-12 min/arm budget on Colab G4 RTX PRO 6000")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_V140)
    ap.add_argument("--head-len", type=int, default=HEAD_LEN_V140)
    ap.add_argument("--long-char-trigger", type=int, default=300)
    ap.add_argument("--p1-steps", type=int, default=2400)
    ap.add_argument("--p2-steps", type=int, default=4800)
    ap.add_argument("--p3-qat-steps", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--r0-drop-p", type=float, default=0.55)
    ap.add_argument("--charlm-drop-p", type=float, default=0.30)
    ap.add_argument("--dt-drop-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261206)
    ap.add_argument("--charlm-cache", type=Path, default=None, help="Turbo v2/v3: Load/save 3-fold OOF CharLM pickle to skip repeated 30s CPU fit")
    ap.add_argument("--compile", action="store_true", help="Turbo v3: Enable dynamic torch.compile on R0, prelude_t1 & loop_step")
    ap.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    ap.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True, help="Turbo v3: Use dynamic=True in torch.compile to prevent recompile_limit(8) fallback")
    ap.add_argument("--amp-bf16", action=argparse.BooleanOptionalAction, default=True, help="Turbo v3: Pure BF16 AMP on CUDA (keeps Sinkhorn/L2/F6/audit/export FP32)")
    ap.add_argument("--eval-batch-size", type=int, default=512, help="Turbo v3: Batch size for pre-collated validation/calibration/ship-gate inference")
    ap.add_argument("--fuse-aux", action=argparse.BooleanOptionalAction, default=True, help="Turbo v3: Fuse co-occurring CF/Fmt/QR auxiliary passes into a single GPU forward")
    ap.add_argument("--prefetch-workers", type=int, default=2, help="Turbo v2/v3: Background zero-GIL Rust collate prefetch threads (0 = sync)")
    ap.add_argument("--no-rust-frontend", action="store_true", help="Disable meguviz_native Rust frontend and fall back to Python")
    # v1.6.2 Research Hooks (100% backwards-compatible, all default OFF/legacy for v1.6.1!)
    ap.add_argument("--use-eba-h", action="store_true", help="v1.6.2 M2: Prior-matched EBA-H hard bounded residual arbitration (0 FP32 ULP violations)")
    ap.add_argument("--eba-pi-nat", type=float, default=0.05, help="v1.6.2 M2: Target natural stream AI prior for exact Bayes EBA-H shift")
    ap.add_argument("--eba-m-pos-grid", type=str, default="0.8,1.2,1.6,2.0", help="v1.6.2 M2: Candidate EBA-H upper Deep excursion bounds M_+")
    ap.add_argument("--eba-m-neg-grid", type=str, default="1.0,1.5,2.0,2.5", help="v1.6.2 M2: Candidate EBA-H lower Deep excursion bounds M_-")
    ap.add_argument("--use-tcbr-r", action="store_true", help="v1.6.2 M2: Zero-sync Trust-Conditioned Bounded Tail Ranking (trusted Softplus vs noisy bounded Sigmoid + influence cap)")
    ap.add_argument("--use-d02-paraphrase-cf", action="store_true", help="v1.6.2 M3: 6 quote-free / indirect-speech D02 rebuttal counterfactual templates (every 4 steps)")
    ap.add_argument("--r0-antidilution-mix", action="store_true", help="v1.6.2: Decoupled anti-dilution Phase-1 R0++ & Deep mix + 2x short-AI (D04) & EN/JA human upsampling")
    ap.add_argument("--longform-every", type=int, default=6, help="v1.6.2 M3: DynaSeam longform + sandwich curriculum period (set 3 for v1.6.2 D06+D08!)")
    ap.add_argument("--use-kadane-focal-router", action="store_true", help="v1.6.2 M3: TriSeam-R Kadane 160c + O(L) E_lambda FocalRouter for >=300c long docs (F6===0.0)")
    ap.add_argument("--puc-warmup-steps", type=int, default=0, help="v1.6.2 M4 Strategy I (PUC): Frozen-backbone warmup steps on fresh discourse/contractive modules before Shrink-and-Perturb")
    ap.add_argument("--puc-shrink", type=float, default=0.95, help="v1.6.2 M4 PUC: Selective Shrink factor lambda on low-sensitivity backbone weights")
    ap.add_argument("--puc-noise", type=float, default=0.015, help="v1.6.2 M4 PUC: Selective Perturb noise scale relative to parameter std")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    use_step_phi = bool(args.use_step_phi) and (not bool(args.no_step_phi))
    use_cv_fusion = not bool(args.no_crossview_fusion)

    if args.ablation_fast and not args.smoke_test:
        args.p1_steps = min(args.p1_steps, 1200)
        args.p2_steps = min(args.p2_steps, 2600)
        args.p3_qat_steps = min(args.p3_qat_steps, 800)

    t_all = time.time()
    phase_timings: dict[str, float] = {}
    t_phase = t_all
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    use_amp = bool(args.amp_bf16 and device.type == "cuda")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir or out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Zero-Leakage Corpus + Super-Teacher 251K Soft Pool (skip unused internal distill_pool build!)
    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir, gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, delta_dir=args.delta_dir, v04_data_path=args.v04_data,
        smoke_test=args.smoke_test, build_distill_pool=False,
    )
    blocked = {
        text_sha16_v140(r["text"])
        for k in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "val_multilingual", "diag_god500")
        for r in corpus.get(k, [])
    }
    extra_cal_cert_rows: list[dict[str, Any]] = []
    if args.extra_cal_cert_pack is not None and args.extra_cal_cert_pack.is_file():
        for cr in read_jsonl_any_v140(args.extra_cal_cert_pack):
            ct = str(cr.get("text", "") or "").strip()
            if not ct:
                continue
            sha_c = text_sha16_v140(ct)
            if sha_c in blocked:
                continue
            blocked.add(sha_c)  # Strictly exclude calibration holdout from ALL training pools!
            ya_c = float(cr.get("y_ai", cr.get("label_ai", -1.0)))
            if not (0.0 <= ya_c <= 1.0):
                continue
            extra_cal_cert_rows.append({"text": ct, "y_ai": ya_c, "y_pr": float(cr.get("y_pr", 0.0))})
        print(f"[v1.6.1-ExtraCalCert] Reserved {len(extra_cal_cert_rows):,} strictly zero-leakage holdout rows from {args.extra_cal_cert_pack}", flush=True)

    distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)
    if not distill_pool:
        raise SystemExit(f"Empty distill pool: {args.distill_pool}")

    d_hard = [r for r in distill_pool if r.get("frontier_gap", 0.0) >= 1.3 or r.get("hard_neg", 0.0) > 0.5]
    d_pos = [r for r in distill_pool if r.get("t_ai_lg", -5.0) >= 0.8]
    d_promo = [r for r in distill_pool if r.get("t_pr_lg", -5.0) >= 0.5]
    d_all = distill_pool
    gold_pos, gold_hd, gold_nat, gold_pr = corpus["train_ai_pos"], corpus["train_hu_hard"], corpus["train_hu_natural"], corpus["train_promo_extra"]
    ex_hard_fn: list[dict[str, Any]] = []
    ex_hard_fp: list[dict[str, Any]] = []
    ex_sandwich: list[dict[str, Any]] = []
    ex_q2_dual: list[dict[str, Any]] = []
    ex_override: list[dict[str, Any]] = []
    ex_general: list[dict[str, Any]] = []

    if args.extra_gold_pack is not None and args.extra_gold_pack.is_file():
        extra_rows = read_jsonl_any_v140(args.extra_gold_pack)
        n_ex_add, n_ex_rescored, n_ex_ovr = 0, 0, 0
        for er in extra_rows:
            et = str(er.get("text", "") or "").strip()
            if not et or text_sha16_v140(et) in blocked:
                continue
            ya = float(er.get("y_ai", er.get("label_ai", -1.0)))
            yp = float(er.get("y_pr", er.get("label_promo", -1.0)))
            # FIXED v1.6.1 bug: accept continuous calibrated y_ai in [0, 1] (previously `ya in (0, 1)` dropped 98.45% of stream pack!)
            has_a = float(er.get("has_ai", 1.0 if (0.0 <= ya <= 1.0) else 0.0))
            has_p = float(er.get("has_pr", 1.0 if (0.0 <= yp <= 1.0) else 0.0))
            if has_a < 0.5 and has_p < 0.5:
                continue
            ya_c = max(0.0, min(1.0, ya)) if has_a > 0.5 else 0.0
            yp_c = max(0.0, min(1.0, yp)) if has_p > 0.5 else 0.0
            bkt = str(er.get("bucket", "") or "").lower()
            tier = str(er.get("tier", er.get("epistemic_tier", "")) or "")
            is_ovr = bool(er.get("is_override", False)) or ("override" in bkt)
            has_t_raw = float(er.get("has_teacher", er.get("has_t", 0.0)))
            t_ai_lg = float(er.get("t_ai_lg", er.get("t_ai_logit", 0.0)))
            t_pr_lg = float(er.get("t_pr_lg", er.get("t_pr_logit", 0.0)))
            # Committee Override Shield: if committee explicitly overrode a label, trust committee gold label & mask conflicting teacher!
            use_t = has_t_raw > 0.5 and (not is_ovr) and (tier != "T0_DIAMOND")
            if use_t:
                n_ex_rescored += 1
            if is_ovr:
                n_ex_ovr += 1
            is_hn = bool(has_a > 0.5 and ya_c <= 0.25 and ("hard_fp" in bkt or bool(er.get("is_hard_fp", False)) or (has_p > 0.5 and yp_c >= 0.5)))
            is_hfn = bool(has_a > 0.5 and ya_c >= 0.70 and ("hard_fn" in bkt or "sandwich" in bkt or "q2_dual" in bkt))
            item = {
                "text": et, "y_ai": ya_c, "y_pr": yp_c,
                "has_ai": has_a, "has_pr": has_p,
                "has_teacher": 1.0 if use_t else 0.0,
                "kd_w": float(er.get("kd_w", 1.0)),
                "t_ai_lg": t_ai_lg if use_t else 0.0,
                "t_ai_r0": float(er.get("t_ai_r0", t_ai_lg)) if use_t else 0.0,
                "t_ai_t1": float(er.get("t_ai_t1", t_ai_lg)) if use_t else 0.0,
                "t_ai_t2": float(er.get("t_ai_t2", t_ai_lg)) if use_t else 0.0,
                "t_ai_t3": float(er.get("t_ai_t3", t_ai_lg)) if use_t else 0.0,
                "frontier_gap": float(er.get("frontier_gap", 0.0)) if use_t else 0.0,
                "t_pr_lg": t_pr_lg if use_t else 0.0,
                "hard_neg": 1.0 if is_hn else 0.0,
                "hard_fn": 1.0 if is_hfn else 0.0,
                "trust_neg": 1.0 if (is_ovr or is_hn or (has_a > 0.5 and ya_c <= 0.20 and not use_t)) else 0.0,
                "era5": int(er.get("era5", -1)), "sub4": int(er.get("sub4", -1)),
            }
            if "hard_fn" in bkt:
                ex_hard_fn.append(item)
            elif "sandwich" in bkt:
                ex_sandwich.append(item)
            elif "q2_dual" in bkt:
                ex_q2_dual.append(item)
            elif is_ovr:
                ex_override.append(item)
            elif "hard_fp" in bkt or is_hn:
                ex_hard_fp.append(item)
            else:
                ex_general.append(item)
            if has_a > 0.5 and ya_c >= 0.65:
                gold_pos.append(item)
            elif has_a > 0.5 and is_hn:
                gold_hd.append(item)
            elif has_a > 0.5 and ya_c <= 0.35:
                gold_nat.append(item)
            if has_p > 0.5 and yp_c >= 0.65:
                gold_pr.append(item)
            n_ex_add += 1
        print(
            f"[v1.6.1-ExtraGold] Admitted {n_ex_add:,} zero-leakage rows (rescored={n_ex_rescored:,}, overrides={n_ex_ovr:,} | "
            f"hard_fn={len(ex_hard_fn):,}, hard_fp={len(ex_hard_fp):,}, sandwich={len(ex_sandwich):,}, "
            f"q2_dual={len(ex_q2_dual):,}, ovr_bkt={len(ex_override):,}, gen={len(ex_general):,}) from {args.extra_gold_pack}",
            flush=True,
        )
    longform_train = read_jsonl_any_v140(args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 1200)]

    # 2. Model + Optional Warm-Start + Group-Clustered 3-Fold OOF CharLM
    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    model_kwargs = dict(
        arch=args.arch,
        vocab_size=tokenizer.vocab_size,
        max_len=int(args.max_len),
        style_dim=STYLE_FEAT_DIM_V11,
        dynamics_mode=args.dynamics,
        use_step_phi=use_step_phi,
        step_decay_mode=args.step_decay_mode,
        use_slot9_dt=bool(args.use_slot9_dt),
        expand_tables=bool(args.expand_tables),
        coverage_beta_enabled=not args.no_coverage_beta,
        use_local_charlm_inject=not args.no_local_charlm,
        use_global_charlm_shortcut=False,
        prelude_global_se=None if not args.no_prelude_global_se else False,
        use_discourse_ops=bool(args.use_discourse_ops),
        lardo_w8_core=bool(args.lardo_w8_core),
    )
    model = MeguvizV160StratosphereLoopModel(**model_kwargs).to(device)
    warm_loaded_keys: set[str] = set()
    if args.init_from_fp16 is not None and args.init_from_fp16.is_file():
        ck_init = torch.load(args.init_from_fp16, map_location="cpu", weights_only=False)
        sd_init = ck_init.get("state_dict", ck_init.get("model_state_dict", ck_init))
        cur_sd = model.state_dict()
        loaded_k, skipped_k = 0, 0
        for k, v in sd_init.items():
            if k in cur_sd and cur_sd[k].shape == v.shape:
                cur_sd[k].copy_(v.to(cur_sd[k].dtype))
                warm_loaded_keys.add(k)
                loaded_k += 1
            else:
                skipped_k += 1
        model.load_state_dict(cur_sd)
        n_new_t = len(cur_sd) - loaded_k
        print(f"[v1.6.1-WarmStart] Loaded {loaded_k}/{len(cur_sd)} matching tensors (new_modules={n_new_t}, skipped={skipped_k}) from {args.init_from_fp16}", flush=True)
    era_grl = OfflineEraGRLHead(emb_dim=AI_END_V130, num_eras=NUM_ERAS_V122).to(device)

    use_rust_fe = (not bool(args.no_rust_frontend)) and HAS_RUST_FRONTEND
    fcache = _StrictGroupOOFFeatureCacheV160(
        tokenizer, max_len=int(args.max_len), head_len=int(args.head_len),
        engram_buckets=model.engram_buckets, trigram_buckets=model.trigram_buckets,
        use_local_charlm=not args.no_local_charlm,
        r0_b2=model.r0_b2, r0_b3=model.r0_b3,
        use_rust=use_rust_fe, bucket_seq_len=bool(args.compile),
    )
    phase_timings["startup_corpus_sec"] = round(time.time() - t_phase, 2)
    t_phase = time.time()
    if args.charlm_cache is not None and args.charlm_cache.is_file() and (not args.smoke_test):
        t_clm = time.time()
        with args.charlm_cache.open("rb") as f_clm:
            fold_lms, text_to_fold, full_charlm = pickle.load(f_clm)
        print(f"[Turbo-CharLM] Loaded cached 3-fold OOF CharLM in {time.time()-t_clm:.2f}s from {args.charlm_cache}", flush=True)
    else:
        charlm_rows = gold_pos + gold_hd + gold_nat
        if args.smoke_test:
            charlm_rows = charlm_rows[:240]
        elif len(charlm_rows) > 80000:
            # Stratified deterministic cap for fast <30s OOF CharLM fit (24K top-k n-grams saturate well before 80K rows!)
            charlm_rows = rng.sample(charlm_rows, 80000)
        lm_txts = [r["text"] for r in charlm_rows]
        lm_y = np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in charlm_rows], dtype=np.int32)
        lm_grps = [str(r["cf_group_id"]) if r.get("cf_group_id") else None for r in charlm_rows]
        fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
            lm_txts, lm_y, group_keys=lm_grps, n_folds=3,
            min_counts=(2, 3, 4) if args.smoke_test else (16, 28, 40),
            max_order_keys=(3000, 1500, 1000) if args.smoke_test else (9000, 8000, 7000),
            seed=args.seed,
        )
        if args.charlm_cache is not None and (not args.smoke_test):
            args.charlm_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_clm = args.charlm_cache.with_suffix(".pkl.tmp")
            with tmp_clm.open("wb") as f_clm:
                pickle.dump((fold_lms, text_to_fold, full_charlm), f_clm, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_clm.replace(args.charlm_cache)
            print(f"[Turbo-CharLM] Saved 3-fold OOF CharLM cache to {args.charlm_cache}", flush=True)
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)
    phase_timings["charlm_sec"] = round(time.time() - t_phase, 2)
    t_phase = time.time()

    compiled_ok = False
    if args.compile and device.type == "cuda" and (not args.smoke_test):
        compiled_ok = model.enable_torch_compile(mode=str(args.compile_mode), dynamic=bool(args.compile_dynamic))
        if compiled_ok:
            # Warm up dynamic shapes cleanly with B=16, L=64 in train mode (never pollutes static guards with B=2!)
            try:
                model.train()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    _dm_ids = torch.ones((16, 64), dtype=torch.long, device=device)
                    _dm_msk = torch.ones((16, 64), dtype=torch.long, device=device)
                    _dm_st = torch.zeros((16, STYLE_FEAT_DIM_V11), device=device)
                    _dm_loc = torch.zeros((16, 64, LOCAL_CHARLM_DIM_V150), device=device)
                    _dm_c1 = torch.ones((16, 32), dtype=torch.long, device=device)
                    _dm_cm = torch.ones((16, 32), device=device)
                    _dm_cf = torch.zeros((16, 26), device=device)
                    _dm_sup = torch.full((16,), 16.0, device=device)
                    _dm_qsc = torch.zeros((16, 64, QUOTE_SCOPE_DIM_V161), device=device) if model.use_discourse_ops else None
                    model(_dm_ids, _dm_msk, _dm_st, _dm_ids, _dm_ids, _dm_loc, _dm_c1, _dm_c1, _dm_c1, _dm_cm, _dm_cf, _dm_sup, outer_steps=2, quote_scope=_dm_qsc)
                torch.cuda.synchronize()
            except Exception as _ce:
                print(f"[Turbo-Compile] Runtime fallback to eager (safe): {_ce}", flush=True)
                model._compiled_r0 = None
                model._compiled_prelude_t1 = None
                model._compiled_loop_step = None
                compiled_ok = False
    phase_timings["compile_warmup_sec"] = round(time.time() - t_phase, 2)
    t_phase = time.time()

    tot_p = sum(p.numel() for p in model.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    print(
        f"[v1.6.0-Locked] ver={args.version} arch={args.arch} dyn={args.dynamics} step_phi={use_step_phi} "
        f"step_decay={args.step_decay_mode} ncl={args.use_ncl} slot9_dt={args.use_slot9_dt} "
        f"expand_tbl={args.expand_tables} freeze_r0={args.freeze_r0} "
        f"rust_fe={use_rust_fe} compile={compiled_ok}({args.compile_mode},dyn={bool(args.compile_dynamic)}) "
        f"amp_bf16={use_amp} fuse_aux={bool(args.fuse_aux)} prefetch={args.prefetch_workers} "
        f"params={tot_p:,} (core={core_p:,}) distill={len(distill_pool):,} gpu={gpu_name}",
        flush=True,
    )

    has_extra_buckets = bool(ex_hard_fn or ex_hard_fp or ex_sandwich or ex_q2_dual or ex_override or ex_general)
    gold_pos_short = [r for r in gold_pos if len(str(r.get("text", ""))) <= 64]
    gold_hd_ml = [r for r in gold_hd if r.get("is_multilingual")]

    def _draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rng.choices(pool, k=k) if (pool and k > 0) else []

    def _draw_upsampled(base_pool: list[dict[str, Any]], focus_pool: list[dict[str, Any]], k: int, focus_frac: float = 0.35) -> list[dict[str, Any]]:
        if not base_pool or k <= 0:
            return []
        if args.r0_antidilution_mix and focus_pool:
            kf = min(k, max(1, int(round(k * focus_frac))))
            return _draw(focus_pool, kf) + _draw(base_pool, k - kf)
        return _draw(base_pool, k)

    def _sample_r0_batch(bs: int) -> list[dict[str, Any]]:
        if not (args.r0_antidilution_mix and has_extra_buckets):
            return _sample_batch(bs)
        # v1.6.2 Phase-1 R0++ Anti-Dilution Mix (restores OJ:v3-hard R0 >= 0.835 while guarding D04 & M01-M03!)
        out_b: list[dict[str, Any]] = []
        out_b.extend(_draw(d_all, int(bs * 0.20)))
        out_b.extend(_draw(d_hard, int(bs * 0.20)))
        out_b.extend(_draw(d_pos, int(bs * 0.07)))
        out_b.extend(_draw(d_promo, int(bs * 0.05)))
        out_b.extend(_draw(ex_hard_fn or gold_pos, int(bs * 0.08)))
        out_b.extend(_draw(ex_sandwich or gold_pos, int(bs * 0.03)))
        out_b.extend(_draw(ex_q2_dual or gold_pr, int(bs * 0.04)))
        out_b.extend(_draw(ex_hard_fp or gold_hd, int(bs * 0.09)))
        out_b.extend(_draw(ex_override or gold_hd, int(bs * 0.05)))
        out_b.extend(_draw(ex_general or gold_nat, int(bs * 0.02)))
        out_b.extend(_draw_upsampled(gold_pos, gold_pos_short, int(bs * 0.07), focus_frac=0.35))
        out_b.extend(_draw_upsampled(gold_hd, gold_hd_ml, int(bs * 0.07), focus_frac=0.30))
        out_b.extend(_draw(gold_pr, max(1, bs - len(out_b))))
        return out_b[:bs]

    def _sample_batch(bs: int) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        if has_extra_buckets and args.r0_antidilution_mix:
            # v1.6.2 Phase-2 Deep Anti-Dilution Mix (+4% d_hard, +2% sandwich, +1% hard_fp, -3% easy stream + 2x D04/ML!)
            out_b.extend(_draw(d_all, int(bs * 0.16)))
            out_b.extend(_draw(d_hard, int(bs * 0.16)))
            out_b.extend(_draw(d_pos, int(bs * 0.06)))
            out_b.extend(_draw(d_promo, int(bs * 0.04)))
            out_b.extend(_draw(ex_hard_fn or gold_pos, int(bs * 0.12)))
            out_b.extend(_draw(ex_sandwich or gold_pos, int(bs * 0.07)))
            out_b.extend(_draw(ex_q2_dual or gold_pr, int(bs * 0.04)))
            out_b.extend(_draw(ex_hard_fp or gold_hd, int(bs * 0.11)))
            out_b.extend(_draw(ex_override or gold_hd, int(bs * 0.06)))
            out_b.extend(_draw(ex_general or gold_nat, int(bs * 0.03)))
            out_b.extend(_draw_upsampled(gold_pos, gold_pos_short, int(bs * 0.06), focus_frac=0.35))
            out_b.extend(_draw_upsampled(gold_hd, gold_hd_ml, int(bs * 0.07), focus_frac=0.30))
            out_b.extend(_draw(gold_pr, max(1, bs - len(out_b))))
        elif has_extra_buckets:
            # v1.6.1 Stratified Anti-Dilution Mix: preserves oldvideo/q4 & 251K armor while upsampling stealth hard_fn/sandwich/q2!
            out_b.extend(_draw(d_all, int(bs * 0.16)))
            out_b.extend(_draw(d_hard, int(bs * 0.12)))
            out_b.extend(_draw(d_pos, int(bs * 0.06)))
            out_b.extend(_draw(d_promo, int(bs * 0.05)))
            out_b.extend(_draw(ex_hard_fn or gold_pos, int(bs * 0.12)))
            out_b.extend(_draw(ex_sandwich or gold_pos, int(bs * 0.05)))
            out_b.extend(_draw(ex_q2_dual or gold_pr, int(bs * 0.04)))
            out_b.extend(_draw(ex_hard_fp or gold_hd, int(bs * 0.10)))
            out_b.extend(_draw(ex_override or gold_hd, int(bs * 0.06)))
            out_b.extend(_draw(ex_general or gold_nat, int(bs * 0.06)))
            out_b.extend(_draw(gold_pos, int(bs * 0.06)))
            out_b.extend(_draw(gold_hd, int(bs * 0.07)))
            out_b.extend(_draw(gold_pr, max(1, bs - len(out_b))))
        else:
            out_b.extend(_draw(d_all, int(bs * 0.24)))
            out_b.extend(_draw(d_hard, int(bs * 0.14)))
            out_b.extend(_draw(d_pos, int(bs * 0.08)))
            out_b.extend(_draw(d_promo, int(bs * 0.06)))
            out_b.extend(_draw_upsampled(gold_pos, gold_pos_short, int(bs * 0.15), focus_frac=0.35))
            out_b.extend(_draw_upsampled(gold_hd, gold_hd_ml, int(bs * 0.15), focus_frac=0.30))
            out_b.extend(_draw(gold_nat, int(bs * 0.11)))
            out_b.extend(_draw(gold_pr, max(1, bs - len(out_b))))
        return out_b[:bs]

    # Pre-collate static validation/calibration splits ONCE (~65 MB VRAM) and reuse across ALL evals & Auto-Ship-Gate!
    cal_rows = corpus["val_gold_v2"] + corpus["val_wave3"] + corpus["val_old_video_neg"] + corpus.get("val_multilingual", [])
    cal_txts = [r["text"] for r in cal_rows]
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.float32)
    y_cal_int = (y_cal_ai >= 0.5).astype(np.int32)
    _n_g2 = len(corpus["val_gold_v2"])
    _n_w3 = len(corpus["val_wave3"])
    _n_old = len(corpus["val_old_video_neg"])
    _n_ml = len(corpus.get("val_multilingual", []))
    _yg2 = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
    _yw3_a = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
    _yw3_p = np.array([int(r["y_pr"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
    _q4_m = (_yw3_p == 1) & (_yw3_a == 0)
    _yml = np.array([int(r["y_ai"] >= 0.5) for r in corpus.get("val_multilingual", [])], dtype=np.int32)
    eval_bs = 32 if args.smoke_test else max(64, int(args.eval_batch_size))
    cal_pre_r0 = _PreCollatedSplitV160(
        cal_txts, fcache, model.r0_b2, model.r0_b3, device,
        batch_size=eval_bs, r0_only=True, need_quote_scope=False, cache_on_device=(device.type == "cuda"),
    )
    cal_pre_deep = _PreCollatedSplitV160(
        cal_txts, fcache, model.r0_b2, model.r0_b3, device,
        batch_size=eval_bs, r0_only=False, need_quote_scope=bool(model.use_discourse_ops), cache_on_device=(device.type == "cuda"),
    )

    def _val_obj_from_cal_pred(pr_all: dict[str, Any], k_ai: str = "logit_ai_t4", k_pr: str = "logit_promo") -> float:
        ai_v = pr_all[k_ai]
        pr_v = pr_all[k_pr]
        s0, s1 = 0, _n_g2
        mg2 = _eval_bin(_yg2, ai_v[s0:s1])
        s0, s1 = s1, s1 + _n_w3
        mw3a = _eval_bin(_yw3_a, ai_v[s0:s1])
        mw3p = _eval_bin(_yw3_p, pr_v[s0:s1])
        q4_fpr = float((expit(np.clip(ai_v[s0:s1][_q4_m], -30, 30)) >= 0.5).mean()) if _q4_m.sum() else 0.0
        s0, s1 = s1, s1 + _n_old
        old_fpr = float((expit(np.clip(ai_v[s0:s1], -30, 30)) >= 0.5).mean()) if _n_old else 0.0
        s0, s1 = s1, s1 + _n_ml
        ml_pr = _eval_bin(_yml, ai_v[s0:s1])["pr"] if _n_ml else 0.0
        return round(float(0.30 * mg2["pr"] + 0.24 * mg2["r95"] + 0.16 * mw3a["pr"] + 0.16 * mw3p["pr"] + 0.14 * ml_pr - 1.30 * old_fpr - 0.50 * q4_fpr), 5)

    def _eval_val(m_cur: MeguvizV160StratosphereLoopModel, r0_only: bool = False, outer_eval: int = 4) -> dict[str, Any]:
        p_all = predict_student_v160(
            m_cur, fcache, cal_txts, device, batch_size=eval_bs, r0_only=r0_only, outer_steps=outer_eval,
            use_crossview_dt=args.use_slot9_dt, precollated=(cal_pre_r0 if r0_only else cal_pre_deep),
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        k_ai = "logit_ai_r0" if r0_only else f"logit_ai_t{outer_eval}"
        k_pr = "logit_promo_r0" if r0_only else "logit_promo"
        ai_v = p_all[k_ai]
        pr_v = p_all[k_pr]
        s0, s1 = 0, _n_g2
        mg2 = _eval_bin(_yg2, ai_v[s0:s1])
        s0, s1 = s1, s1 + _n_w3
        mw3a = _eval_bin(_yw3_a, ai_v[s0:s1])
        mw3p = _eval_bin(_yw3_p, pr_v[s0:s1])
        q4_fpr = float((expit(np.clip(ai_v[s0:s1][_q4_m], -30, 30)) >= 0.5).mean()) if _q4_m.sum() else 0.0
        s0, s1 = s1, s1 + _n_old
        old_fpr = float((expit(np.clip(ai_v[s0:s1], -30, 30)) >= 0.5).mean()) if _n_old else 0.0
        s0, s1 = s1, s1 + _n_ml
        ml_pr = _eval_bin(_yml, ai_v[s0:s1])["pr"] if _n_ml else 0.0

        obj = 0.30 * mg2["pr"] + 0.24 * mg2["r95"] + 0.16 * mw3a["pr"] + 0.16 * mw3p["pr"] + 0.14 * ml_pr - 1.30 * old_fpr - 0.50 * q4_fpr
        res_d: dict[str, Any] = {
            "sel_obj": round(float(obj), 5),
            "g2_pr": mg2["pr"], "g2_r95": mg2["r95"], "g2_r99": mg2["r99"],
            "w3_ai_pr": mw3a["pr"], "w3_pr_pr": mw3p["pr"], "ml_pr": ml_pr,
            "old_fpr": round(old_fpr, 4), "q4_fpr": round(q4_fpr, 4),
        }
        if not r0_only:
            res_d["g2_r1_pr"] = _eval_bin(_yg2, p_all["logit_ai_r1"][:_n_g2])["pr"]
            res_d["g2_t1_pr"] = _eval_bin(_yg2, p_all["logit_ai_t1"][:_n_g2])["pr"]
            res_d["g2_t2_pr"] = _eval_bin(_yg2, p_all["logit_ai_t2"][:_n_g2])["pr"]
            res_d["t4_minus_t2_pr"] = round(mg2["pr"] - res_d["g2_t2_pr"], 4)
            res_d["t1_minus_r1_pr"] = round(res_d["g2_t1_pr"] - res_d["g2_r1_pr"], 4)
        return res_d

    def _cosine_mult(step_i: int, total_s: int, warm_frac: float = 0.08, min_f: float = 0.06) -> float:
        warm = max(1, int(total_s * warm_frac))
        w = min(1.0, (step_i + 1) / float(warm))
        c = min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * step_i / float(max(1, total_s))))
        return w * c

    bs = 16 if args.smoke_test else int(args.batch_size)
    p1_steps = 4 if args.smoke_test else int(args.p1_steps)
    p2_steps = (
        max(6, min(24, int(args.puc_warmup_steps) + 4))
        if (args.smoke_test and int(args.puc_warmup_steps) > 0 and warm_loaded_keys)
        else (6 if args.smoke_test else int(args.p2_steps))
    )
    p3_steps = 0 if args.skip_qat else (4 if args.smoke_test else int(args.p3_qat_steps))

    def _minimal_rule_format_edit(txt: str) -> str:
        """Rev.2 P1 deterministic 6-template rule-based label-preserving human format edit (NEVER LLM text rewrite!)."""
        s = str(txt).strip()
        if len(s) < 12:
            return s
        mode = rng.randint(0, 5)
        if mode == 0:
            parts = [p.strip() for p in s.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n") if p.strip()]
            if len(parts) >= 2:
                return "\n".join(f"{idx + 1}. {p}" for idx, p in enumerate(parts[:5]))
            return f"1. {s}"
        if mode == 1:
            lines = [ln.strip().lstrip("0123456789.、-•* ") for ln in s.splitlines() if ln.strip()]
            return " ".join(lines) if lines else s
        if mode == 2:
            return f"【要点】\n{s}"
        if mode == 3:
            parts = [p.strip() for p in s.replace("，", "，\n").replace("。", "。\n").split("\n") if p.strip()]
            if len(parts) >= 2:
                return "\n".join(f"- **{p[:4]}**：{p}" for p in parts[:4])
            return f"- **核心观点**：{s}"
        if mode == 4:
            return f"总结如下：\n{s}"
        # Mode 5: Strip or add casual Bilibili human markers
        if "[doge]" in s:
            return s.replace("[doge]", "").strip() or s
        return f"{s} [doge]"

    # -----------------------------------------------------------------------
    # Phase 1: Shared Frozen `R0` Char-View (trained ONCE and reused bit-for-bit across A0..A5 / v1.6.1!)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    best_r0_obj = -1e9
    has_cf_p1 = corpus["stats"]["admitted_cf_pairs"] > 0
    if args.init_r0_ckpt is not None and args.init_r0_ckpt.is_file():
        ck_r0 = torch.load(args.init_r0_ckpt, map_location="cpu", weights_only=False)
        model.rung0.load_state_dict(ck_r0["rung0_state_dict"])
        best_r0_obj = float(ck_r0.get("best_r0_obj", 0.0))
        ev_r0_loaded = _eval_val(model, r0_only=True)
        print(f"  [Phase1-R0 LOADED SHARED FROZEN CKPT] {args.init_r0_ckpt} {ev_r0_loaded}", flush=True)
    else:
        r0_params = list(model.rung0.parameters())
        opt1 = _make_adamw([{"params": r0_params, "lr": 1.8e-3, "base_lr": 1.8e-3, "weight_decay": 0.01}], device)
        best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
        model.train()
        fn_r0_p1 = model._compiled_r0 if model._compiled_r0 is not None else model.forward_r0
        _hu_pool_p1 = (gold_hd + gold_nat) if (gold_hd or gold_nat) else []
        for step in range(p1_steps):
            mult1 = _cosine_mult(step, p1_steps)
            for pg in opt1.param_groups:
                pg["lr"] = pg["base_lr"] * mult1
            bk = _collate_v160_batch(_sample_r0_batch(bs), fcache, model.r0_b2, model.r0_b3, device, rng=rng, punct_jitter_prob=0.35, need_bpe=False)
            opt1.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                cf_in = bk["cf"]
                if float(args.charlm_drop_p) > 0.0:
                    lm_k = (torch.rand((cf_in.shape[0], 1), device=cf_in.device) >= float(args.charlm_drop_p)).to(cf_in.dtype)
                    cf_in = torch.cat([cf_in[:, :R0_STRUCT_DIM_V140], cf_in[:, R0_STRUCT_DIM_V140:] * lm_k], dim=-1)
                r0 = fn_r0_p1(bk["c1"], bk["c2"], bk["c3"], bk["cm"], cf_in)
                l_ai = _progressive_kd_loss_nosync(r0["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
                l_pr = _progressive_kd_loss_nosync(r0["logit_promo_r0"], bk, axis="promo")
                with torch.no_grad():
                    tp_ai = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                    tp_pr = torch.where(bk["has_pr"] > 0.5, bk["y_pr"], torch.sigmoid(bk["t_pr_lg"]))
                    err_ai = ((torch.sigmoid(r0["logit_ai_r0"].float()) - tp_ai).abs() >= 0.20).float()
                    err_pr = ((torch.sigmoid(r0["logit_promo_r0"].float()) - tp_pr).abs() >= 0.20).float()
                l_def = F.binary_cross_entropy_with_logits(r0["defer_ai_r0_logit"].float(), err_ai) + F.binary_cross_entropy_with_logits(r0["defer_pr_r0_logit"].float(), err_pr)
                loss = l_ai + 0.72 * l_pr + 0.15 * l_def
                if args.p1_use_cf:
                    # Zero-sync balanced CVaR + fused 1-pass CF mirror & rule format L_inv inside Phase-1 R0++!
                    hu_m1 = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
                    n_hu1 = hu_m1.sum()
                    hu_f1 = hu_m1.float()
                    l0_all1 = F.softplus(r0["logit_ai_r0"].float())
                    l0_s1 = torch.sort(torch.where(hu_m1, l0_all1, torch.full_like(l0_all1, 1e9))).values
                    q_i1 = torch.minimum(torch.floor(0.90 * n_hu1.float()).long().clamp(min=0), (n_hu1 - 1).clamp(min=0))
                    t_cv1 = torch.where(n_hu1 >= 8, l0_s1.gather(0, q_i1.view(1)).squeeze(0).detach(), torch.zeros((), device=device))
                    cvar1 = t_cv1 + ((F.relu(l0_all1 - t_cv1) * hu_f1).sum() / hu_f1.sum().clamp_min(1.0)) / 0.10
                    loss = loss + 0.04 * cvar1 * (n_hu1 >= 8).float()
                    if has_cf_p1 and (step % 4 == 0):
                        ch1 = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
                        if ch1:
                            n_c1 = len(ch1)
                            b_cf1 = _collate_v160_batch(
                                [{"text": x["ai_text"]} for x in ch1] + [{"text": x["human_text"]} for x in ch1],
                                fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=False, include_meta=False,
                            )
                            rcf1 = fn_r0_p1(b_cf1["c1"], b_cf1["c2"], b_cf1["c3"], b_cf1["cm"], b_cf1["cf"])["logit_ai_r0"].float()
                            pw1 = torch.tensor([float(x["pair_weight"]) for x in ch1], dtype=torch.float32, device=device)
                            loss = loss + 0.18 * (F.softplus(1.7 - (rcf1[:n_c1] - rcf1[n_c1:])) * pw1).mean()
                    if (step % 5 == 2) and _hu_pool_p1:
                        fc1 = rng.sample(_hu_pool_p1, min(4 if args.smoke_test else 12, len(_hu_pool_p1)))
                        n_f1 = len(fc1)
                        ot1 = [str(r["text"]) for r in fc1]
                        et1 = [_minimal_rule_format_edit(t) for t in ot1]
                        b_fm1 = _collate_v160_batch(
                            [{"text": t} for t in (ot1 + et1)],
                            fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=False, include_meta=False,
                        )
                        rfm1 = fn_r0_p1(b_fm1["c1"], b_fm1["c2"], b_fm1["c3"], b_fm1["cm"], b_fm1["cf"])["logit_ai_r0"].float()
                        loss = loss + 0.08 * ((rfm1[:n_f1] - rfm1[n_f1:]) ** 2).mean().clamp_max(8.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(r0_params, 1.0, foreach=True)
            opt1.step()
            if (step + 1) % max(1, p1_steps // 4) == 0 or step == p1_steps - 1:
                ev1 = _eval_val(model, r0_only=True)
                print(f"  [Phase1-R0 step={step+1}/{p1_steps}] loss={float(loss.item()):.4f} {ev1}", flush=True)
                if ev1["sel_obj"] > best_r0_obj:
                    best_r0_obj, best_r0_sd = ev1["sel_obj"], {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
                model.train()
        model.rung0.load_state_dict(best_r0_sd)
        if args.save_r0_ckpt is not None:
            args.save_r0_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"rung0_state_dict": best_r0_sd, "best_r0_obj": best_r0_obj, "seed": args.seed}, args.save_r0_ckpt)
            print(f"  [Phase1-R0 SAVED SHARED CKPT] {args.save_r0_ckpt}", flush=True)
    phase_timings["phase1_r0_sec"] = round(time.time() - t_phase, 2)

    # Re-seed RNG identically right before Phase 2 so A0..A5 see the EXACT SAME Phase-2/3 sample order regardless of Phase-1!
    rng = random.Random(args.seed + 1000)
    np.random.seed(args.seed + 1000)
    torch.manual_seed(args.seed + 1000)

    if args.freeze_r0:
        for p in model.rung0.parameters():
            p.requires_grad_(False)
        model.rung0.eval()

    # -----------------------------------------------------------------------
    # Turbo v3 Zero-Sync Vectorized Horvitz-Thompson Anytime Depth Loss (0 CUDA sync stalls/step!)
    # -----------------------------------------------------------------------
    _is_u16 = (args.t_train_mode == "uniform_1_6") and (not args.smoke_test)
    _w_depth_cache: dict[int, torch.Tensor] = {
        t_k: torch.tensor(
            [(1.0 / float(max(1, 6 - ti))) if _is_u16 else (1.0 / float(t_k)) for ti in range(t_k)],
            dtype=torch.float32, device=device,
        )
        for t_k in range(1, 7)
    }
    _w_pair_cache: dict[int, torch.Tensor] = {
        t_k: torch.tensor(
            [(6.0 / (5.0 * float(max(1, 5 - ti)))) if _is_u16 else (1.0 / float(t_k - 1)) for ti in range(t_k - 1)],
            dtype=torch.float32, device=device,
        )
        for t_k in range(2, 7)
    }

    def _stratosphere_loss(od: dict[str, Any], bk: dict[str, torch.Tensor]) -> torch.Tensor:
        ai_steps: list[torch.Tensor] = od["logit_ai_steps"]
        pr_steps: list[torch.Tensor] = od["logit_promo_steps"]
        u_steps: list[torch.Tensor] = od["u_halt_steps"]
        T = len(ai_steps)
        ai_stack = torch.stack(ai_steps, dim=0).float()  # (T, B)
        pr_stack = torch.stack(pr_steps, dim=0).float()  # (T, B)
        z_last = ai_stack[-1]
        B_cur = z_last.shape[0]
        w_depth_vec = _w_depth_cache[T]

        with torch.no_grad():
            t_p_sn9 = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_t3"]))
            ai_valid = (bk["has_ai"] > 0.5) | (bk["has_t"] > 0.5)
            trusted_m = ((bk["has_ai"] > 0.5) | ((bk["has_t"] > 0.5) & ((t_p_sn9 - 0.5).abs() >= 0.35))).float()
            y_bin_ai = (t_p_sn9 >= 0.5).float()
            y_sign = 2.0 * y_bin_ai - 1.0
            p_r0 = torch.sigmoid(od["logit_ai_r0"].detach().float())
            err_r0 = (p_r0 - t_p_sn9).abs()
            if args.use_cascade_weight:
                r0_uncertain = ((p_r0 > 0.04) & (p_r0 < 0.96)) | (err_r0 >= 0.15)
                sw = 0.25 + 0.75 * r0_uncertain.float()
                sw = sw / sw.mean().clamp_min(1.0)
            else:
                sw = torch.ones_like(err_r0)

        # 1. Batched Exact Equal-Weight Anytime Depth Loss across all T depths in ONE vectorized pass!
        l_ai_any = (_progressive_kd_loss_nosync(ai_stack, bk, axis="ai", teacher_key="t_ai_t3", sample_weight=sw) * w_depth_vec).sum()
        l_pr_any = (_progressive_kd_loss_nosync(pr_stack, bk, axis="promo") * w_depth_vec).sum()
        ai_ends = torch.stack([ai_stack[0], ai_stack[-1]], dim=0) if T >= 2 else ai_stack[:1]
        l_top_any = _top_tail_pairwise_nosync(ai_ends, y_bin_ai, ai_valid.float()).mean()

        loss = (
            1.65 * l_ai_any
            + 0.70 * l_pr_any
            + 0.24 * l_top_any
            + 0.25 * _progressive_kd_loss_nosync(od["logit_ai_r1"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=sw)
            + 0.16 * _progressive_kd_loss_nosync(od["logit_promo_r1"], bk, axis="promo")
        )
        if not args.freeze_r0:
            loss = (
                loss
                + 0.18 * _progressive_kd_loss_nosync(od["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
                + 0.16 * _progressive_kd_loss_nosync(od["logit_promo_r0"], bk, axis="promo")
            )

        # 2. Vectorized Stop-Grad Tolerance Monotonic Margin: `ReLU(sg(m_t) - m_{t+1} - eps)`
        if (not args.no_mono_loss) and T >= 2:
            w_pair_vec = _w_pair_cache[T]
            eps_m = float(args.mono_eps)
            denom_tr = trusted_m.sum().clamp_min(1.0)
            m_t_sg = y_sign.unsqueeze(0) * ai_stack[:-1].detach().clamp(-6.0, 6.0)
            m_next = y_sign.unsqueeze(0) * ai_stack[1:].clamp(-6.0, 6.0)
            per_pair = (F.relu(m_t_sg - m_next - eps_m) * trusted_m.unsqueeze(0)).sum(dim=-1) / denom_tr
            loss = loss + 0.30 * (per_pair * w_pair_vec).sum()

        # 3. Falsification Arm A3 ONLY: Zero-sync Hard-Example NCL against Frozen Stop-Grad R0
        if args.use_ncl and float(args.ncl_gamma) > 0.0:
            ncl_f = ((err_r0 >= 0.12) & ai_valid).float()
            ncl_den = ncl_f.sum().clamp_min(1.0)
            r0_res = (p_r0 - t_p_sn9).detach().unsqueeze(0)
            b_res = torch.sigmoid(ai_stack) - t_p_sn9.unsqueeze(0)
            l_ncl = (((F.relu(b_res * r0_res) * ncl_f.unsqueeze(0)).sum(dim=-1) / ncl_den) * w_depth_vec).sum()
            loss = loss + float(args.ncl_gamma) * l_ncl

        # 4. Vectorized `TaH2` Lookahead Utility Regression
        if T >= 2:
            w_pair_vec = _w_pair_cache[T]
            with torch.no_grad():
                step_bces = F.binary_cross_entropy_with_logits(ai_stack.detach(), t_p_sn9.unsqueeze(0).expand_as(ai_stack), reduction="none")
                tgt_list: list[torch.Tensor] = []
                for ti in range(T - 1):
                    best_gain = step_bces[ti] - step_bces[ti + 1] - 0.01
                    for j in range(ti + 2, T):
                        best_gain = torch.maximum(best_gain, step_bces[ti] - step_bces[j] - 0.01 * (j - ti))
                    tgt_list.append(best_gain.clamp(-0.5, 1.0))
                tgt_u_stack = torch.stack(tgt_list, dim=0)
            u_stack = torch.stack(u_steps[:-1], dim=0).float()
            l_halt = (F.smooth_l1_loss(u_stack, tgt_u_stack, reduction="none", beta=0.1).mean(dim=-1) * w_pair_vec).sum()
            loss = loss + 0.25 * l_halt

        # 5. Zero-sync Promo 4-way subtype + Era GRL
        s4_f = ((bk["has_pr"] > 0.5) & (bk["sub4"] >= 0) & (bk["sub4"] < 4)).float()
        s4_ce = F.cross_entropy(od["promo_sub4_logit"].float(), bk["sub4"].clamp(0, 3), reduction="none")
        loss = loss + 0.10 * ((s4_ce * s4_f).sum() / s4_f.sum().clamp_min(1.0))

        hu_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
        era_v = hu_m & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
        era_v_cnt = era_v.sum()
        era_v_f = era_v.float()
        era_lg = era_grl.net(_GradReverseFn.apply(od["emb_ai"].float(), 0.50))
        era_ce = F.cross_entropy(era_lg, bk["era5"].clamp(0, NUM_ERAS_V122 - 1), reduction="none")
        loss = loss + 0.12 * ((era_ce * era_v_f).sum() / era_v_f.sum().clamp_min(1.0)) * (era_v_cnt >= 8).float()

        # 6. Zero-sync Negative-Class Tail Protection: v1.6.2 `TCBR-R` (trust-conditioned bounded) OR legacy Softplus-CVaR
        n_hu = hu_m.sum()
        if args.use_tcbr_r and float(args.cvar_weight) > 0.0:
            # M2 TCBR-R: trusted gold/override/hard_fp negatives use `T * softplus(z/T)` with per-sample influence cap `1/K_-`;
            # unverified teacher negatives use bounded `4T * sigmoid((z - tau_u)/T)` (vanishing extreme-outlier gradient!).
            a_cv = max(0.02, min(0.5, float(args.cvar_alpha)))
            trust_m = hu_m & (bk["trust_neg"] > 0.5)
            untrust_m = hu_m & (~trust_m)
            n_tr = trust_m.sum()
            tr_ok = n_tr >= 4
            l0_tr = F.softplus(z_last)
            l0_tr_sorted = torch.sort(torch.where(trust_m, l0_tr, torch.full_like(l0_tr, 1e9))).values
            q_tr = torch.minimum(torch.floor((1.0 - a_cv) * n_tr.float()).long().clamp(min=0), (n_tr - 1).clamp(min=0))
            tau_tr = torch.where(tr_ok, l0_tr_sorted.gather(0, q_tr.view(1)).squeeze(0).detach(), torch.zeros((), device=device))
            tr_f = trust_m.float()
            # Per-sample influence budget: cap individual excess term to avoid 1 mislabeled outlier hijacking CVaR!
            k_tail_tr = (a_cv * n_tr.float()).clamp_min(1.0)
            ex_tr = F.relu(l0_tr - tau_tr)
            # Soft-cap per-sample contribution while preserving exact CVaR slope near the tail boundary
            ex_tr_cap = 3.0 * torch.tanh(ex_tr / 3.0)
            cvar_trust = tau_tr + ((ex_tr_cap * tr_f).sum() / k_tail_tr)
            # Unverified teacher negatives: bounded `4T * sigmoid(u/T)` so extreme residuals have ~0 gradient!
            tu = 1.25
            untr_f = untrust_m.float()
            untr_ok = untrust_m.sum() >= 4
            l_untr = ((4.0 * tu * torch.sigmoid((z_last + 0.25) / tu)) * untr_f).sum() / untr_f.sum().clamp_min(1.0)
            loss = loss + float(args.cvar_weight) * (cvar_trust * tr_ok.float() + 0.35 * l_untr * untr_ok.float())
        elif args.use_softplus_cvar and float(args.cvar_weight) > 0.0:
            l0_all = F.softplus(z_last)
            a_cv = max(0.02, min(0.5, float(args.cvar_alpha)))
            l0_sorted = torch.sort(torch.where(hu_m, l0_all, torch.full_like(l0_all, 1e9))).values
            q_idx = torch.minimum(torch.floor((1.0 - a_cv) * n_hu.float()).long().clamp(min=0), (n_hu - 1).clamp(min=0))
            hu_ok = n_hu >= 8
            tau_cv = torch.where(hu_ok, l0_sorted.gather(0, q_idx.view(1)).squeeze(0).detach(), torch.zeros((), device=device))
            hu_f = hu_m.float()
            cvar_hu = tau_cv + ((F.relu(l0_all - tau_cv) * hu_f).sum() / hu_f.sum().clamp_min(1.0)) / a_cv
            loss = loss + float(args.cvar_weight) * cvar_hu * hu_ok.float()

        # 7. Zero-sync Positive Stealth-AI Tail Recall-Floor Pairwise Protection
        if float(args.recall_floor_weight) > 0.0:
            ai_pos_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] >= 0.70)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] >= 1.4))
            n_p = ai_pos_m.sum()
            rf_ok = (n_p >= 4) & (n_hu >= 8)
            max_kp = max(2, min(B_cur, int(math.ceil(0.50 * B_cur))))
            max_kn = max(2, min(B_cur, int(math.ceil(0.25 * B_cur))))
            zp_raw = torch.topk(torch.where(ai_pos_m, z_last, torch.full_like(z_last, 1e9)), k=max_kp, largest=False).values
            zn_raw = torch.topk(torch.where(hu_m, z_last, torch.full_like(z_last, -1e9)), k=max_kn, largest=True).values
            kp_real = torch.minimum(torch.ceil(0.50 * n_p.float()).long().clamp(min=2), n_p.clamp_min(1))
            kn_real = torch.minimum(torch.ceil(0.25 * n_hu.float()).long().clamp(min=2), n_hu.clamp_min(1))
            pm_k = (torch.arange(max_kp, device=device) < kp_real) & rf_ok
            nm_k = (torch.arange(max_kn, device=device) < kn_real) & rf_ok
            zp_safe = torch.where(pm_k, zp_raw, torch.zeros_like(zp_raw))
            zn_safe = torch.where(nm_k, zn_raw, torch.zeros_like(zn_raw))
            pair_rf_m = (nm_k.unsqueeze(1) & pm_k.unsqueeze(0)).float()
            l_rf = (F.softplus(1.6 + zn_safe.unsqueeze(1) - zp_safe.unsqueeze(0)) * pair_rf_m).sum() / pair_rf_m.sum().clamp_min(1.0)
            hfn_m = ai_pos_m & (bk["hard_fn"] > 0.5)
            hfn_ok = (hfn_m.sum() >= 2) & rf_ok
            pair_hfn_m = (nm_k.unsqueeze(1) & hfn_m.unsqueeze(0)).float()
            l_hfn = (F.softplus(1.8 + zn_safe.unsqueeze(1) - z_last.unsqueeze(0)) * pair_hfn_m).sum() / pair_hfn_m.sum().clamp_min(1.0)
            l_rf = torch.where(hfn_ok, 0.5 * l_rf + 0.5 * l_hfn, l_rf)
            loss = loss + float(args.recall_floor_weight) * l_rf * rf_ok.float()

        # 8. Zero-sync Exact Unbiased Class-Conditional V-REx across era5 environments
        if args.use_cc_vrex and float(args.cc_vrex_weight) > 0.0:
            era_ok = ai_valid & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
            per_bce = F.binary_cross_entropy_with_logits(z_last, y_bin_ai, reduction="none")
            env_eff = torch.where(era_ok, bk["era5"], torch.full_like(bk["era5"], -1))
            vrex_d = class_conditional_vrex_unbiased(
                per_bce, y_bin_ai.long(), env_eff, n_envs=NUM_ERAS_V122, beta_0=1.0, beta_1=0.5,
            )
            loss = loss + float(args.cc_vrex_weight) * vrex_d["vrex_penalty"].clamp(min=-0.15, max=2.0) * (era_ok.sum() >= 16).float()

        # 9. v1.6.1-B Inner-Ball Smooth Saturation Regularizer for Direct Contractive Kernel D
        if float(args.sat_reg_weight) > 0.0 and od.get("slot_steps"):
            r_in = float(CONTRACTIVE_INNER_R_V161)
            sat_pen = torch.stack([
                F.relu(sl[:, : model.core.n_ai_slots, :].float().norm(dim=-1) - r_in).pow(2).mean()
                for sl in od["slot_steps"]
            ]).mean()
            loss = loss + float(args.sat_reg_weight) * sat_pen
        return loss

    # -----------------------------------------------------------------------
    # Phase 2: Full StratosphereLoop Training (`T ~ U{1..6}` + CF Mirrors + Symmetric DynaSeam)
    # Turbo v3: GPU-resident `_foreach_lerp_` EMA + Fused Aux4 Passes + Zero-GIL Rust Prefetcher!
    # -----------------------------------------------------------------------
    t_phase = time.time()
    r0_ids = {id(p) for p in model.rung0.parameters()}
    p2_lr_val = float(args.p2_lr)
    if warm_loaded_keys:
        warm_backbone_p = [p for n, p in model.named_parameters() if id(p) not in r0_ids and n in warm_loaded_keys]
        fresh_new_p = [p for n, p in model.named_parameters() if id(p) not in r0_ids and n not in warm_loaded_keys]
        opt2_groups = []
        if warm_backbone_p:
            opt2_groups.append({"params": warm_backbone_p + list(era_grl.parameters()), "lr": p2_lr_val, "base_lr": p2_lr_val, "weight_decay": 0.01})
        if fresh_new_p:
            new_lr = min(2.0e-3, p2_lr_val * 2.5)
            opt2_groups.append({"params": fresh_new_p, "lr": new_lr, "base_lr": new_lr, "weight_decay": 0.01})
            print(f"  [v1.6.1-DiffLR] warm_backbone={len(warm_backbone_p)} tensors (lr={p2_lr_val:.2e}) | fresh_new_modules={len(fresh_new_p)} tensors (lr={new_lr:.2e})", flush=True)
        if not opt2_groups:
            opt2_groups = [{"params": [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters()), "lr": p2_lr_val, "base_lr": p2_lr_val, "weight_decay": 0.01}]
    else:
        opt2_groups = [{"params": [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters()), "lr": p2_lr_val, "base_lr": p2_lr_val, "weight_decay": 0.01}]
    if not args.freeze_r0:
        opt2_groups.insert(0, {"params": list(model.rung0.parameters()), "lr": min(2.0e-4, p2_lr_val * 0.25), "base_lr": min(2.0e-4, p2_lr_val * 0.25), "weight_decay": 0.01})
    opt2 = _make_adamw(opt2_groups, device)
    p2_grad_params = [p for p in model.parameters() if p.requires_grad] + list(era_grl.parameters())

    # v1.6.2 M4 Strategy I (PUC — Plasticity-Unlocked Warm-Start):
    #   Stage 1 [0..puc_w): freeze old backbone, train ONLY fresh discourse/contractive/long heads (verifying non-zero grad!)
    #   Transition at step == puc_w: selective Shrink-and-Perturb on low-sensitivity old backbone + reset AdamW moments!
    puc_w = min(int(args.puc_warmup_steps), max(0, p2_steps - 1)) if (int(args.puc_warmup_steps) > 0 and warm_loaded_keys) else 0
    puc_grad_log: list[dict[str, Any]] = []
    puc_opt = None
    puc_fresh_params: list[torch.nn.Parameter] = []
    _puc_disc_params: list[tuple[str, torch.nn.Parameter]] = []
    if puc_w > 0:
        puc_fresh_names = {
            n for n, p in model.named_parameters()
            if id(p) not in r0_ids and (
                (n not in warm_loaded_keys)
                or any(tok in n for tok in ("disc_", "quote_scope_proj", "contractive_alpha", "long_ai_"))
            )
        }
        for n, p in model.named_parameters():
            if id(p) in r0_ids:
                p.requires_grad_(False)
            elif n in puc_fresh_names:
                p.requires_grad_(True)
                puc_fresh_params.append(p)
                if "disc_" in n or "contractive_alpha" in n:
                    _puc_disc_params.append((n, p))
            else:
                p.requires_grad_(False)
        if puc_fresh_params:
            puc_lr = min(2.0e-3, p2_lr_val * 2.5)
            puc_opt = _make_adamw([{"params": puc_fresh_params + list(era_grl.parameters()), "lr": puc_lr, "base_lr": puc_lr, "weight_decay": 0.01}], device)
            print(f"  [v1.6.2-PUC] Stage-1 Frozen-Backbone Warmup: steps=0..{puc_w}, fresh_tensors={len(puc_fresh_params)}, lr={puc_lr:.2e}", flush=True)
        else:
            puc_w = 0
            for n, p in model.named_parameters():
                if id(p) not in r0_ids or (not args.freeze_r0):
                    p.requires_grad_(True)

    ema_sd_gpu = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _ema_fp_keys = [k for k, v in ema_sd_gpu.items() if v.is_floating_point()]
    _ema_fp_tensors = [ema_sd_gpu[k] for k in _ema_fp_keys]
    best_p2_sd = {k: v.detach().cpu().clone() for k, v in ema_sd_gpu.items()}
    best_p2_obj = -1e9
    best_p2_ev: dict[str, Any] = {}
    p2_soup_snaps: list[dict[str, torch.Tensor]] = []
    has_cf = corpus["stats"]["admitted_cf_pairs"] > 0
    need_qsc = bool(model.use_discourse_ops)
    _QUOTE_REBUTTAL_TEMPLATES = (
        "楼上这段一看就是AI生成的：「{ai}」大家别被带节奏了。",
        "笑死，把“{ai}”原封不动复制粘贴过来当真人评论呢？",
        "引用楼上机器人话术：【{ai}】——真有人信这是真人写的？",
    )
    # v1.6.2 M3 D02 quote-free / indirect-speech Bilibili human rebuttal templates (fixes D02 where only 9/55 have quotes!)
    _D02_PARAPHRASE_TEMPLATES = (
        "楼上那套分析纯属胡扯：{ai}",
        "刚才看到有人说{ai}，其实根本不对，完全没说到点子上。",
        "别信前面那个长篇大论了，什么{ai}，一听就是瞎编的。",
        "笑死，前面居然有人认真科普说{ai}，现实里压根不是这么回事。",
        "转述一下前面离谱观点，说啥{ai}，大伙图一乐得了别当真。",
        "刚才刷到一条评论吹什么{ai}，我直接原地无语，纯纯废话文学。",
    )
    _PROMO_FLIP_PAIRS = (
        ("强烈推荐大家去搜一下这个宝藏插件，券后只要9.9闭眼冲，主页置顶领福利！", "千万别信楼上推的那个插件，纯智商税广告，大家避雷别买！"),
        ("想学全套干货的私信我或者搜工种号领取免费资料包，名额有限！", "评论区那些让私信领资料加群的全是引流骗子，举报了别理！"),
        ("这款耳机亲测降噪无敌，点击评论区置顶链接立减200，家人们冲！", "这款耳机我买了两天就退了，降噪根本不行，楼上纯水军软广！"),
    )
    # Pre-collate static Promo-Flip pair ONCE onto GPU (zero per-step CPU/H2D cost!)
    _promo_flip_n = len(_PROMO_FLIP_PAIRS)
    _b_promo_flip_gpu: dict[str, torch.Tensor] | None = None
    if args.use_promo_flip_cf and float(args.promo_flip_weight) > 0.0:
        _pf_txts = [p[0] for p in _PROMO_FLIP_PAIRS] + [p[1] for p in _PROMO_FLIP_PAIRS]
        _b_promo_flip_gpu = _collate_v160_batch(
            [{"text": t} for t in _pf_txts], fcache, model.r0_b2, model.r0_b3, device,
            rng=None, punct_jitter_prob=0.0, need_bpe=True, need_quote_scope=need_qsc, include_meta=False,
        )

    pin_cuda = (device.type == "cuda")
    n_prefetch = max(0, int(args.prefetch_workers)) if (not args.smoke_test) else 0
    _hu_pool_cache = (gold_hd + gold_nat) if (gold_hd or gold_nat) else []
    _fuse_aux = bool(args.fuse_aux)
    _lf_every = max(1, int(args.longform_every))
    _d02_para_en = bool(args.use_d02_paraphrase_cf)
    _d02_all_tmpls = _D02_PARAPHRASE_TEMPLATES + _D02_PARAPHRASE_TEMPLATES + _QUOTE_REBUTTAL_TEMPLATES

    def _plan_p2_step_cpu(step_i: int) -> dict[str, Any]:
        """Deterministic main-thread RNG draws for step `step_i` (zero heavy CPU work on main thread)."""
        main_rows = _sample_batch(bs)
        main_jit = _prepare_jittered_texts(main_rows, rng, 0.35)
        if args.t_train_mode == "uniform_1_6":
            outer_n = rng.randint(1, 6) if not args.smoke_test else rng.choice([2, 4, 6])
        else:
            outer_n = 4 if (step_i % 2 == 0) else 2

        cf_pack = None
        if has_cf and (step_i % 3 == 0):
            ch = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
            if ch:
                cf_pack = ([x["ai_text"] for x in ch] + [x["human_text"] for x in ch], [float(x["pair_weight"]) for x in ch], len(ch))

        fmt_pack = None
        if args.use_fmt_cf and float(args.fmt_cf_weight) > 0.0 and (step_i % 4 == 2) and _hu_pool_cache:
            n_fc = 4 if args.smoke_test else 12
            fc_rows = rng.sample(_hu_pool_cache, min(n_fc, len(_hu_pool_cache)))
            orig_txts = [str(r["text"]) for r in fc_rows]
            edit_txts = [_minimal_rule_format_edit(t) for t in orig_txts]
            fmt_pack = (orig_txts + edit_txts, len(orig_txts))

        qr_txts = None
        if _d02_para_en and (step_i % 4 == 0) and gold_pos:
            n_qr = 2 if args.smoke_test else 8
            qr_src = rng.sample(gold_pos, min(n_qr, len(gold_pos)))
            qr_txts = [rng.choice(_d02_all_tmpls).format(ai=str(r["text"])[:110].strip()) for r in qr_src]
        elif args.use_fmt_cf and (step_i % 8 == 4) and gold_pos:
            n_qr = 2 if args.smoke_test else 6
            qr_src = rng.sample(gold_pos, min(n_qr, len(gold_pos)))
            qr_txts = [rng.choice(_QUOTE_REBUTTAL_TEMPLATES).format(ai=str(r["text"])[:110].strip()) for r in qr_src]

        lf_spec = None
        if longform_train and (step_i % _lf_every == 0):
            n_docs_lf = 2 if args.smoke_test else (6 if _lf_every <= 3 else 5)
            docs = rng.sample(longform_train, min(n_docs_lf, len(longform_train)))
            sand_src = None
            if gold_pos and gold_nat:
                sand_src = (
                    str(rng.choice(gold_nat)["text"]),
                    str(rng.choice(gold_pos)["text"]),
                    str(rng.choice(gold_nat)["text"]),
                    str(rng.choice(gold_pos)["text"]),
                    str(rng.choice(gold_nat)["text"]),
                )
            lf_spec = (docs, sand_src)

        return {
            "main_rows": main_rows, "main_jit": main_jit, "outer_n": outer_n,
            "cf_pack": cf_pack, "fmt_pack": fmt_pack, "qr_txts": qr_txts, "lf_spec": lf_spec,
        }

    def _build_p2_step_host(plan: dict[str, Any]) -> dict[str, Any]:
        """Worker-thread zero-GIL Rust collation for main batch + fused auxiliary passes."""
        bk_h = _collate_v160_batch_host(
            plan["main_rows"], fcache, model.r0_b2, model.r0_b3,
            need_bpe=True, need_quote_scope=need_qsc, pre_jittered=plan["main_jit"], pin_memory=pin_cuda,
        )
        aux4_h = None
        cf_h, fmt_h, qr_h = None, None, None
        if _fuse_aux:
            aux_txts: list[str] = []
            r0_dps: list[float] = []
            clm_dps: list[float] = []
            slices: dict[str, Any] = {}
            if plan["cf_pack"] is not None:
                cf_txts, cf_pws, n_cf = plan["cf_pack"]
                s0 = len(aux_txts)
                aux_txts.extend(cf_txts)
                r0_dps.extend([0.50] * len(cf_txts))
                clm_dps.extend([0.35] * len(cf_txts))
                slices["cf"] = (s0, len(aux_txts), np.asarray(cf_pws, dtype=np.float32), n_cf)
            if plan["fmt_pack"] is not None:
                fmt_txts, n_fc = plan["fmt_pack"]
                s0 = len(aux_txts)
                aux_txts.extend(fmt_txts)
                r0_dps.extend([0.50] * len(fmt_txts))
                clm_dps.extend([0.30] * len(fmt_txts))
                slices["fmt"] = (s0, len(aux_txts), n_fc)
            if plan["qr_txts"] is not None:
                qr_t = plan["qr_txts"]
                s0 = len(aux_txts)
                aux_txts.extend(qr_t)
                r0_dps.extend([0.40] * len(qr_t))
                clm_dps.extend([0.25] * len(qr_t))
                slices["qr"] = (s0, len(aux_txts), len(qr_t))
            if aux_txts:
                aux_bk_h = _collate_v160_batch_host(
                    [{"text": t} for t in aux_txts], fcache, model.r0_b2, model.r0_b3,
                    need_bpe=True, need_quote_scope=need_qsc, punct_jitter_prob=0.0,
                    pin_memory=pin_cuda, include_meta=False,
                )
                aux4_h = (aux_bk_h, np.asarray(r0_dps, dtype=np.float32), np.asarray(clm_dps, dtype=np.float32), slices)
        else:
            if plan["cf_pack"] is not None:
                cf_txts, cf_pws, n_cf = plan["cf_pack"]
                cf_h = (
                    _collate_v160_batch_host([{"text": t} for t in cf_txts], fcache, model.r0_b2, model.r0_b3, need_bpe=True, need_quote_scope=need_qsc, punct_jitter_prob=0.0, pin_memory=pin_cuda, include_meta=False),
                    np.asarray(cf_pws, dtype=np.float32),
                    n_cf,
                )
            if plan["fmt_pack"] is not None:
                fmt_txts, n_fc = plan["fmt_pack"]
                fmt_h = (
                    _collate_v160_batch_host([{"text": t} for t in fmt_txts], fcache, model.r0_b2, model.r0_b3, need_bpe=True, need_quote_scope=need_qsc, punct_jitter_prob=0.0, pin_memory=pin_cuda, include_meta=False),
                    n_fc,
                )
            if plan["qr_txts"] is not None:
                qr_h = (
                    _collate_v160_batch_host([{"text": t} for t in plan["qr_txts"]], fcache, model.r0_b2, model.r0_b3, need_bpe=True, need_quote_scope=need_qsc, punct_jitter_prob=0.0, pin_memory=pin_cuda, include_meta=False),
                    len(plan["qr_txts"]),
                )
        lf_h = None
        if plan["lf_spec"] is not None:
            docs, sand_src = plan["lf_spec"]
            all_ch_txts: list[str] = []
            doc_slices: list[tuple[int, int, np.ndarray, float, float, bool]] = []
            for d_it in docs:
                cl_doc, _ = canonicalize_zero_pii_v150(str(d_it.get("text", "")))
                chunks = plan_long_chunks_v11(cl_doc, max_chunks=6)
                if not chunks:
                    continue
                s0 = len(all_ch_txts)
                all_ch_txts.extend(c["text"] for c in chunks)
                cmeta = np.stack([c["chunk_meta"] for c in chunks], 0).astype(np.float32)
                ya_v = 1.0 if float(d_it.get("ai_score", d_it.get("score", 0.0))) >= 65.0 else 0.0
                yp_v = 1.0 if float(d_it.get("promo_score", 0.0)) >= 65.0 else 0.0
                doc_slices.append((s0, len(all_ch_txts), cmeta, ya_v, yp_v, True))
            if sand_src is not None:
                h_wrap = canonicalize_zero_pii_v150(sand_src[0])[0]
                a_core = canonicalize_zero_pii_v150(sand_src[1])[0]
                h_tail = canonicalize_zero_pii_v150(sand_src[2])[0]
                sand_pairs: list[tuple[str, float]] = [
                    (f"{h_wrap}\n\n{a_core}\n\n{h_tail}", 1.0),
                    (f"{h_wrap}\n\n楼上这段评论一看就是AI生成的：「{a_core}」大家别被带节奏。\n\n{h_tail}", 0.0),
                ]
                if _lf_every <= 3 and len(sand_src) >= 5:
                    # v1.6.2 M3 asymmetric off-center sandwiches (D06+D08 localized tail/head AI!)
                    a_core2 = canonicalize_zero_pii_v150(sand_src[3])[0]
                    h_mid = canonicalize_zero_pii_v150(sand_src[4])[0]
                    sand_pairs.append((f"{h_wrap}\n\n{h_mid}\n\n{a_core2}", 1.0))
                    sand_pairs.append((f"{a_core2}\n\n{h_wrap}\n\n{h_tail}", 1.0))
                for sand_doc, tgt_y in sand_pairs:
                    s_chunks = plan_long_chunks_v11(sand_doc, max_chunks=4)
                    if len(s_chunks) >= 2:
                        s0 = len(all_ch_txts)
                        all_ch_txts.extend(c["text"] for c in s_chunks)
                        cmeta = np.stack([c["chunk_meta"] for c in s_chunks], 0).astype(np.float32)
                        doc_slices.append((s0, len(all_ch_txts), cmeta, tgt_y, 0.0, False))
            if all_ch_txts:
                lf_bk_h = _collate_v160_batch_host(
                    [{"text": t} for t in all_ch_txts], fcache, model.r0_b2, model.r0_b3,
                    need_bpe=True, need_quote_scope=need_qsc, punct_jitter_prob=0.0,
                    pin_memory=pin_cuda, include_meta=False,
                )
                lf_h = (lf_bk_h, doc_slices)
        return {"bk": bk_h, "outer_n": plan["outer_n"], "aux4": aux4_h, "cf": cf_h, "fmt": fmt_h, "qr": qr_h, "lf": lf_h}

    model.train()
    if args.freeze_r0:
        model.rung0.eval()
    _p2_pool = ThreadPoolExecutor(max_workers=n_prefetch) if n_prefetch > 0 else None
    _p2_q: deque = deque()
    _p2_next_plan = 0
    _p2_depth = min(4, max(1, n_prefetch * 2))
    try:
        if _p2_pool is not None:
            while _p2_next_plan < min(p2_steps, _p2_depth):
                _p2_q.append(_p2_pool.submit(_build_p2_step_host, _plan_p2_step_cpu(_p2_next_plan)))
                _p2_next_plan += 1

        for step in range(p2_steps):
            # v1.6.2 M4 PUC Transition at step == puc_w: unfreeze, selective Shrink-and-Perturb low-sensitivity weights, reset AdamW!
            if puc_w > 0 and step == puc_w:
                for n, p in model.named_parameters():
                    if id(p) not in r0_ids or (not args.freeze_r0):
                        p.requires_grad_(True)
                n_shrunk = 0
                sh_lam = float(args.puc_shrink)
                sh_noise = float(args.puc_noise)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if (
                            n in warm_loaded_keys
                            and p.ndim >= 2
                            and id(p) not in r0_ids
                            and (n.startswith("prelude_blocks.") or ".ff_" in n or ".sa_" in n)
                            and not any(prot in n for prot in ("ai_head", "promo_head", "contractive", "disc_", "quote_scope", "long_"))
                        ):
                            p_std = p.float().std().clamp_min(1e-3)
                            p.mul_(sh_lam).add_(torch.randn_like(p) * (sh_noise * p_std))
                            n_shrunk += 1
                opt2 = _make_adamw(opt2_groups, device)
                p2_grad_params = [p for p in model.parameters() if p.requires_grad] + list(era_grl.parameters())
                with torch.no_grad():
                    cur_sd_post_puc = model.state_dict()
                    for k_e in _ema_fp_keys:
                        ema_sd_gpu[k_e].copy_(cur_sd_post_puc[k_e])
                print(f"  [v1.6.2-PUC] Transition @ step={step}: shrunk={n_shrunk} low-sens tensors (lam={sh_lam}, noise={sh_noise}), AdamW reset!", flush=True)

            in_puc_warmup = (puc_w > 0) and (step < puc_w)
            active_opt = puc_opt if (in_puc_warmup and puc_opt is not None) else opt2
            active_grad_params = (puc_fresh_params + list(era_grl.parameters())) if in_puc_warmup else p2_grad_params
            if in_puc_warmup:
                mult2 = min(1.0, (step + 1) / float(max(1, int(puc_w * 0.15))))
            else:
                mult2 = _cosine_mult(step - puc_w, max(1, p2_steps - puc_w))
            for pg in active_opt.param_groups:
                pg["lr"] = pg["base_lr"] * mult2
            if _p2_pool is not None:
                step_host = _p2_q.popleft().result()
                if _p2_next_plan < p2_steps:
                    _p2_q.append(_p2_pool.submit(_build_p2_step_host, _plan_p2_step_cpu(_p2_next_plan)))
                    _p2_next_plan += 1
            else:
                step_host = _build_p2_step_host(_plan_p2_step_cpu(step))

            bk = _h2d_batch(step_host["bk"], device)
            outer_n = int(step_host["outer_n"])

            active_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                od = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                    outer_steps=outer_n, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p),
                    dt_drop_p=float(args.dt_drop_p), use_crossview_dt=bool(args.use_slot9_dt),
                    quote_scope=bk.get("qsc"),
                )
                loss = _stratosphere_loss(od, bk)

                # Turbo v3 Fused single-pass Aux4 (CF + Format-CF + Quoted-Rebuttal combined in ONE GPU forward!)
                if step_host["aux4"] is not None:
                    aux_bk_h, r0_dps_np, clm_dps_np, slices = step_host["aux4"]
                    baux = _h2d_batch(aux_bk_h, device)
                    r0_dp_t = torch.from_numpy(r0_dps_np).to(device, non_blocking=pin_cuda)
                    clm_dp_t = torch.from_numpy(clm_dps_np).to(device, non_blocking=pin_cuda)
                    oaux = model(
                        baux["ids"], baux["msk"], baux["st"], baux["eg"], baux["tg"], baux["loc"],
                        baux["c1"], baux["c2"], baux["c3"], baux["cm"], baux["cf"], baux["sup"],
                        outer_steps=4, r0_drop_p=r0_dp_t, charlm_drop_p=clm_dp_t,
                        use_crossview_dt=bool(args.use_slot9_dt), quote_scope=baux.get("qsc"),
                    )
                    if "cf" in slices:
                        s0, s1, cf_pws_np, n_cf = slices["cf"]
                        pw = torch.from_numpy(cf_pws_np).to(device, non_blocking=pin_cuda)
                        t4_c, t2_c, t1_c = oaux["logit_ai_t4"][s0:s1].float(), oaux["logit_ai_t2"][s0:s1].float(), oaux["logit_ai_t1"][s0:s1].float()
                        l_cf = (
                            (F.softplus(1.8 - (t4_c[:n_cf] - t4_c[n_cf:])) * pw).mean()
                            + 0.60 * (F.softplus(1.7 - (t2_c[:n_cf] - t2_c[n_cf:])) * pw).mean()
                            + 0.50 * (F.softplus(1.6 - (t1_c[:n_cf] - t1_c[n_cf:])) * pw).mean()
                        )
                        loss = loss + 0.25 * l_cf
                    if "fmt" in slices:
                        s0, s1, n_fc = slices["fmt"]
                        t4_f, t2_f = oaux["logit_ai_t4"][s0:s1].float(), oaux["logit_ai_t2"][s0:s1].float()
                        l_inv = ((t4_f[:n_fc] - t4_f[n_fc:]) ** 2).mean() + 0.5 * ((t2_f[:n_fc] - t2_f[n_fc:]) ** 2).mean()
                        loss = loss + float(args.fmt_cf_weight) * l_inv.clamp_max(8.0)
                    if "qr" in slices:
                        s0, s1, n_qr_cur = slices["qr"]
                        y_zero = torch.full((n_qr_cur,), 0.02, dtype=torch.float32, device=device)
                        loss = loss + 0.14 * (
                            F.binary_cross_entropy_with_logits(oaux["logit_ai_t4"][s0:s1].float(), y_zero)
                            + 0.5 * F.binary_cross_entropy_with_logits(oaux["logit_ai_t1"][s0:s1].float(), y_zero)
                        )
                else:
                    if step_host["cf"] is not None:
                        cf_bk_h, cf_pws_np, n_cf = step_host["cf"]
                        bcf = _h2d_batch(cf_bk_h, device)
                        ocf = model(
                            bcf["ids"], bcf["msk"], bcf["st"], bcf["eg"], bcf["tg"], bcf["loc"],
                            bcf["c1"], bcf["c2"], bcf["c3"], bcf["cm"], bcf["cf"], bcf["sup"],
                            outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.35,
                            use_crossview_dt=bool(args.use_slot9_dt), quote_scope=bcf.get("qsc"),
                        )
                        pw = torch.from_numpy(cf_pws_np).to(device, non_blocking=pin_cuda)
                        l_cf = (
                            (F.softplus(1.8 - (ocf["logit_ai_t4"][:n_cf].float() - ocf["logit_ai_t4"][n_cf:].float())) * pw).mean()
                            + 0.60 * (F.softplus(1.7 - (ocf["logit_ai_t2"][:n_cf].float() - ocf["logit_ai_t2"][n_cf:].float())) * pw).mean()
                            + 0.50 * (F.softplus(1.6 - (ocf["logit_ai_t1"][:n_cf].float() - ocf["logit_ai_t1"][n_cf:].float())) * pw).mean()
                        )
                        loss = loss + 0.25 * l_cf

                    if step_host["fmt"] is not None:
                        fmt_bk_h, n_fc = step_host["fmt"]
                        bfmt = _h2d_batch(fmt_bk_h, device)
                        ofmt = model(
                            bfmt["ids"], bfmt["msk"], bfmt["st"], bfmt["eg"], bfmt["tg"], bfmt["loc"],
                            bfmt["c1"], bfmt["c2"], bfmt["c3"], bfmt["cm"], bfmt["cf"], bfmt["sup"],
                            outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.30,
                            use_crossview_dt=bool(args.use_slot9_dt), quote_scope=bfmt.get("qsc"),
                        )
                        l_inv = (
                            ((ofmt["logit_ai_t4"][:n_fc].float() - ofmt["logit_ai_t4"][n_fc:].float()) ** 2).mean()
                            + 0.5 * ((ofmt["logit_ai_t2"][:n_fc].float() - ofmt["logit_ai_t2"][n_fc:].float()) ** 2).mean()
                        )
                        loss = loss + float(args.fmt_cf_weight) * l_inv.clamp_max(8.0)

                    if step_host["qr"] is not None:
                        qr_bk_h, n_qr_cur = step_host["qr"]
                        b_qr = _h2d_batch(qr_bk_h, device)
                        o_qr = model(
                            b_qr["ids"], b_qr["msk"], b_qr["st"], b_qr["eg"], b_qr["tg"], b_qr["loc"],
                            b_qr["c1"], b_qr["c2"], b_qr["c3"], b_qr["cm"], b_qr["cf"], b_qr["sup"],
                            outer_steps=4, r0_drop_p=0.40, charlm_drop_p=0.25,
                            use_crossview_dt=bool(args.use_slot9_dt), quote_scope=b_qr.get("qsc"),
                        )
                        y_zero = torch.full((n_qr_cur,), 0.02, dtype=torch.float32, device=device)
                        loss = loss + 0.14 * (
                            F.binary_cross_entropy_with_logits(o_qr["logit_ai_t4"].float(), y_zero)
                            + 0.5 * F.binary_cross_entropy_with_logits(o_qr["logit_ai_t1"].float(), y_zero)
                        )

                # Fused 1-pass Static Promo Stance-Flip Counterfactual L_flip
                if _b_promo_flip_gpu is not None and (step % 8 == 6):
                    bpf = _b_promo_flip_gpu
                    opf = model(
                        bpf["ids"], bpf["msk"], bpf["st"], bpf["eg"], bpf["tg"], bpf["loc"],
                        bpf["c1"], bpf["c2"], bpf["c3"], bpf["cm"], bpf["cf"], bpf["sup"],
                        outer_steps=2, r0_drop_p=0.3, use_crossview_dt=bool(args.use_slot9_dt),
                        quote_scope=bpf.get("qsc"),
                    )
                    l_flip = F.relu(1.8 - opf["logit_promo_t2"][:_promo_flip_n].float() + opf["logit_promo_t2"][_promo_flip_n:].float()).mean()
                    loss = loss + float(args.promo_flip_weight) * l_flip

                # Fused SINGLE-BATCH Longform + Symmetric Sandwich DynaSeam (7.9x faster than 7 serial calls!)
                if step_host["lf"] is not None:
                    lf_bk_h, doc_slices = step_host["lf"]
                    blf = _h2d_batch(lf_bk_h, device)
                    olf = model(
                        blf["ids"], blf["msk"], blf["st"], blf["eg"], blf["tg"], blf["loc"],
                        blf["c1"], blf["c2"], blf["c3"], blf["cm"], blf["cf"], blf["sup"],
                        outer_steps=2, r0_drop_p=0.0, use_crossview_dt=bool(args.use_slot9_dt),
                        quote_scope=blf.get("qsc"),
                    )
                    d_losses = []
                    for s0, s1, cmeta_np, ya_v, yp_v, has_pr_loss in doc_slices:
                        nc = s1 - s0
                        cm_t = torch.from_numpy(cmeta_np).unsqueeze(0).to(device, non_blocking=pin_cuda)
                        vm_t = torch.ones((1, nc), dtype=torch.float32, device=device)
                        da, dp = model.aggregate_long_chunks(
                            olf["emb_ai"][s0:s1].unsqueeze(0), olf["emb_pr"][s0:s1].unsqueeze(0),
                            olf["logit_ai"][s0:s1].unsqueeze(0), olf["logit_promo"][s0:s1].unsqueeze(0),
                            cm_t, vm_t,
                        )
                        y_da = torch.tensor([ya_v], dtype=torch.float32, device=device)
                        if has_pr_loss:
                            y_dp = torch.tensor([yp_v], dtype=torch.float32, device=device)
                            d_losses.append(F.binary_cross_entropy_with_logits(da.float(), y_da) + 0.50 * F.binary_cross_entropy_with_logits(dp.float(), y_dp))
                        else:
                            d_losses.append(F.binary_cross_entropy_with_logits(da.float(), y_da))
                    if d_losses:
                        loss = loss + 0.22 * torch.stack(d_losses).mean()

            loss.backward()
            if in_puc_warmup and (step == max(0, puc_w // 2) or step == puc_w - 1) and _puc_disc_params:
                d_gn = math.sqrt(sum(float(p.grad.float().norm().item() ** 2) for _, p in _puc_disc_params if p.grad is not None))
                puc_grad_log.append({"step": step + 1, "disc_grad_norm": round(d_gn, 6)})
                print(f"  [v1.6.2-PUC Warmup step={step+1}/{puc_w}] disc_grad_norm={d_gn:.5f}", flush=True)
            torch.nn.utils.clip_grad_norm_(active_grad_params, 1.0, foreach=True)
            active_opt.step()
            # GPU-resident fused EMA update (54x faster than CPU sync copy!)
            with torch.no_grad():
                cur_sd_ref = model.state_dict()
                torch._foreach_lerp_(_ema_fp_tensors, [cur_sd_ref[k] for k in _ema_fp_keys], 0.005)
            if (not in_puc_warmup) and ((step + 1) % max(1, p2_steps // 6) == 0 or step == p2_steps - 1):
                cur_sd_gpu = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_sd_gpu)
                ev2 = _eval_val(model, r0_only=False)
                if ev2["sel_obj"] > best_p2_obj:
                    best_p2_obj = ev2["sel_obj"]
                    best_p2_sd = {k: v.detach().cpu().clone() for k, v in ema_sd_gpu.items()}
                    best_p2_ev = ev2
                if step + 1 >= int(p2_steps * 0.5):
                    p2_soup_snaps.append({k: v.detach().cpu().clone() for k, v in ema_sd_gpu.items()})
                    if len(p2_soup_snaps) > 3:
                        p2_soup_snaps.pop(0)
                lr_show = active_opt.param_groups[-1]["lr"]
                print(f"  [Phase2 step={step+1}/{p2_steps}] lr={lr_show:.2e} loss={float(loss.item()):.4f} {ev2}", flush=True)
                model.load_state_dict(cur_sd_gpu)
                model.train()
                if args.freeze_r0:
                    model.rung0.eval()
    finally:
        if _p2_pool is not None:
            _p2_pool.shutdown(wait=True)

    model.load_state_dict(best_p2_sd)
    torch.save({"version": args.version, "arch": args.arch, "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_p2_sd.items()}, "val": best_p2_ev}, ckpt_dir / f"{args.version}_fp16_master.pt")
    phase_timings["phase2_train_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # 3-Checkpoint Quantization Audit Part A: STRICT FP32 Master vs Direct-Packed MGVZ10 (pre-QAT)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    m_dep = MeguvizV160StratosphereLoopModel(**{**model_kwargs, "style_era_dropout_p": 0.0}).to(device).eval()
    m_dep.load_state_dict(best_p2_sd)
    p_fp32_master = predict_student_v160(
        m_dep, fcache, cal_txts, device, batch_size=eval_bs, outer_steps=6,
        use_crossview_dt=args.use_slot9_dt, record_diagnostics=False, precollated=cal_pre_deep, use_amp=False,
    )
    dp_mgvz_path = out_dir / f"{args.version}_direct_packed_preqat.mgvz"
    pack_bundle_v10_v160(m_dep, dp_mgvz_path)
    load_bundle_v10_v160_into_model(m_dep, dp_mgvz_path)
    p_direct_pack = predict_student_v160(
        m_dep, fcache, cal_txts, device, batch_size=eval_bs, outer_steps=6,
        use_crossview_dt=args.use_slot9_dt, record_diagnostics=False, precollated=cal_pre_deep, use_amp=False,
    )
    phase_timings["audit_preqat_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # Phase 3: Identical LoopQ-Aware Clean W4/W8 QAT (`T ~ U{1..6}`) + Prefetching!
    # -----------------------------------------------------------------------
    t_phase = time.time()
    best_q_obj = float(best_p2_obj)
    best_q_sd = {k: v.clone() for k, v in best_p2_sd.items()}
    if p3_steps > 0:
        set_v160_qat_mode(model, True)
        qat_params = [p for p in model.parameters() if p.requires_grad]
        qat_lr_val = float(args.qat_lr)
        opt3 = _make_adamw([{"params": qat_params, "lr": qat_lr_val, "base_lr": qat_lr_val, "weight_decay": 0.005}], device)
        ema_q_gpu = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _emaq_fp_keys = [k for k, v in ema_q_gpu.items() if v.is_floating_point()]
        _emaq_fp_tensors = [ema_q_gpu[k] for k in _emaq_fp_keys]
        best_q_sd, best_q_obj = {k: v.detach().cpu().clone() for k, v in ema_q_gpu.items()}, _eval_val(model, r0_only=False)["sel_obj"]
        saved_cvar_en = args.use_softplus_cvar
        saved_tcbr_en = args.use_tcbr_r
        if args.auto_ship_gate:
            # Light-QAT without CVaR/TCBR shock avoids the v1.6.0-r1 1600-step QAT tail drift!
            args.use_softplus_cvar = False
            args.use_tcbr_r = False
        model.train()
        if args.freeze_r0:
            model.rung0.eval()

        def _plan_p3_step_cpu() -> tuple[list[dict[str, Any]], list[tuple[str, bool]], int]:
            rows_q = _sample_batch(bs)
            jit_q = _prepare_jittered_texts(rows_q, rng, 0.35)
            outer_q = rng.randint(1, 6) if (args.t_train_mode == "uniform_1_6" and not args.smoke_test) else 4
            return rows_q, jit_q, outer_q

        def _build_p3_step_host(plan_q: tuple[list[dict[str, Any]], list[tuple[str, bool]], int]) -> tuple[dict[str, torch.Tensor], int]:
            rows_q, jit_q, outer_q = plan_q
            bk_qh = _collate_v160_batch_host(
                rows_q, fcache, model.r0_b2, model.r0_b3,
                need_bpe=True, need_quote_scope=need_qsc, pre_jittered=jit_q, pin_memory=pin_cuda,
            )
            return bk_qh, outer_q

        _p3_pool = ThreadPoolExecutor(max_workers=n_prefetch) if n_prefetch > 0 else None
        _p3_q: deque = deque()
        _p3_next = 0
        _p3_depth = min(4, max(1, n_prefetch * 2))
        try:
            if _p3_pool is not None:
                while _p3_next < min(p3_steps, _p3_depth):
                    _p3_q.append(_p3_pool.submit(_build_p3_step_host, _plan_p3_step_cpu()))
                    _p3_next += 1
            for step in range(p3_steps):
                mult3 = _cosine_mult(step, p3_steps, warm_frac=0.06, min_f=0.08)
                for pg in opt3.param_groups:
                    pg["lr"] = pg["base_lr"] * mult3
                if _p3_pool is not None:
                    bk_qh, outer_q = _p3_q.popleft().result()
                    if _p3_next < p3_steps:
                        _p3_q.append(_p3_pool.submit(_build_p3_step_host, _plan_p3_step_cpu()))
                        _p3_next += 1
                else:
                    bk_qh, outer_q = _build_p3_step_host(_plan_p3_step_cpu())
                bk = _h2d_batch(bk_qh, device)
                opt3.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    od = model(
                        bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                        bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                        outer_steps=outer_q, r0_drop_p=0.45, charlm_drop_p=0.25,
                        dt_drop_p=float(args.dt_drop_p), use_crossview_dt=bool(args.use_slot9_dt),
                        quote_scope=bk.get("qsc"),
                    )
                    loss = _stratosphere_loss(od, bk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qat_params, 1.0, foreach=True)
                opt3.step()
                with torch.no_grad():
                    cur_q_ref = model.state_dict()
                    torch._foreach_lerp_(_emaq_fp_tensors, [cur_q_ref[k] for k in _emaq_fp_keys], 0.008)
                if (step + 1) % max(1, p3_steps // 3) == 0 or step == p3_steps - 1:
                    cur_sd_gpu = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema_q_gpu)
                    evq = _eval_val(model, r0_only=False)
                    print(f"  [Phase3-QAT step={step+1}/{p3_steps}] loss={float(loss.item()):.4f} {evq}", flush=True)
                    if evq["sel_obj"] >= best_q_obj:
                        best_q_obj = evq["sel_obj"]
                        best_q_sd = {k: v.detach().cpu().clone() for k, v in ema_q_gpu.items()}
                    model.load_state_dict(cur_sd_gpu)
                    model.train()
                    if args.freeze_r0:
                        model.rung0.eval()
        finally:
            if _p3_pool is not None:
                _p3_pool.shutdown(wait=True)
        args.use_softplus_cvar = saved_cvar_en
        args.use_tcbr_r = saved_tcbr_en
        model.load_state_dict(best_q_sd)
        set_v160_qat_mode(model, False)
    # Detach compiled hooks after training so post-training FP32 audit/export is strictly uncompiled FP32!
    model._compiled_r0 = None
    model._compiled_prelude_t1 = None
    model._compiled_loop_step = None
    phase_timings["phase3_qat_sec"] = round(time.time() - t_phase, 2)

    # Evaluate deployed post-QAT MGVZ10 weights & Turbo v3 Zero-Disk In-Memory Auto Ship Gate!
    t_phase = time.time()
    postqat_mgvz_tmp = out_dir / "bundle_v10_stratosphere_v160.mgvz"
    pack_bundle_v10_v160(model, postqat_mgvz_tmp)
    load_bundle_v10_v160_into_model(m_dep, postqat_mgvz_tmp)

    def _quant_drift_summary(p_ref: dict[str, Any], p_q: dict[str, Any]) -> dict[str, Any]:
        out_q: dict[str, Any] = {}
        for k in (1, 2, 3, 4, 5, 6):
            r_l = p_ref[f"logit_ai_t{k}"]
            q_l = p_q[f"logit_ai_t{k}"]
            d = np.abs(q_l - r_l)
            ap_r = _eval_bin(y_cal_int, r_l)["pr"]
            ap_q = _eval_bin(y_cal_int, q_l)["pr"]
            out_q[f"T{k}"] = {
                "max_abs_logit_drift": round(float(d.max()), 5) if len(d) else 0.0,
                "mean_abs_logit_drift": round(float(d.mean()), 5) if len(d) else 0.0,
                "fp32_pr": ap_r,
                "mgvz_pr": ap_q,
                "delta_pr": round(ap_q - ap_r, 5),
            }
        return out_q

    p_postqat_cal = predict_student_v160(
        m_dep, fcache, cal_txts, device, batch_size=eval_bs, r0_only=False, outer_steps=6,
        use_crossview_dt=args.use_slot9_dt, record_diagnostics=False, precollated=cal_pre_deep, use_amp=False,
    )
    ship_gate_report: dict[str, Any] = {"enabled": bool(args.auto_ship_gate), "winner": "post_qat"}
    if args.auto_ship_gate:
        # Turbo v3 Zero-Disk In-Memory Auto-Ship-Gate:
        #   1. Reuses `cal_pre_deep` on GPU in STRICT FP32 (`use_amp=False`)
        #   2. Reuses `p_fp32_master`, `p_direct_pack`, and `p_postqat_cal` (skips 3 redundant passes!)
        #   3. Reuses ONE `m_dep` instance + bit-exact in-memory `apply_mgvz10_quant_roundtrip_inplace` (0 disk I/O!)
        def _score_from_preds(pr_c: dict[str, Any], fp32_ref_t4: np.ndarray, sd_cand: dict[str, torch.Tensor]) -> tuple[float, dict[str, Any], dict[str, torch.Tensor]]:
            k_pr_use = "logit_promo_t4" if "logit_promo_t4" in pr_c else "logit_promo"
            sel_obj = _val_obj_from_cal_pred(pr_c, k_ai="logit_ai_t4", k_pr=k_pr_use)
            drift_t4 = float(np.abs(pr_c["logit_ai_t4"] - fp32_ref_t4).mean())
            cal_m = _eval_bin(y_cal_int, pr_c["logit_ai_t4"])
            sc = float(sel_obj) + 0.25 * float(cal_m["pr"]) + 0.10 * float(cal_m["r99"]) - 0.12 * drift_t4
            info = {"sel_obj": sel_obj, "cal_t4_pr": cal_m["pr"], "cal_t4_r95": cal_m["r95"], "cal_t4_r99": cal_m["r99"], "drift_t4_vs_own_fp32": round(drift_t4, 5), "ship_score": round(sc, 5)}
            return sc, info, sd_cand

        def _score_mgvz_cand(sd_cand: dict[str, torch.Tensor], fp32_ref_t4: np.ndarray | None = None, mgvz_pred: dict[str, Any] | None = None) -> tuple[float, dict[str, Any], dict[str, torch.Tensor]]:
            loaded_sd = False
            if fp32_ref_t4 is None:
                m_dep.load_state_dict(sd_cand)
                loaded_sd = True
                fp32_ref_t4 = predict_student_v160(
                    m_dep, fcache, cal_txts, device, batch_size=eval_bs, outer_steps=4,
                    use_crossview_dt=args.use_slot9_dt, record_diagnostics=False, precollated=cal_pre_deep, use_amp=False,
                )["logit_ai_t4"]
            if mgvz_pred is None:
                if not loaded_sd:
                    m_dep.load_state_dict(sd_cand)
                apply_mgvz10_quant_roundtrip_inplace(m_dep)
                mgvz_pred = predict_student_v160(
                    m_dep, fcache, cal_txts, device, batch_size=eval_bs, outer_steps=4,
                    use_crossview_dt=args.use_slot9_dt, record_diagnostics=False, precollated=cal_pre_deep, use_amp=False,
                )
            return _score_from_preds(mgvz_pred, fp32_ref_t4, sd_cand)

        cands: list[tuple[str, float, dict[str, Any], dict[str, torch.Tensor]]] = []
        # Candidate 1: reuse already-evaluated `p_direct_pack` vs `p_fp32_master`!
        sc1, in1, sd1 = _score_from_preds(p_direct_pack, p_fp32_master["logit_ai_t4"], best_p2_sd)
        cands.append(("direct_pack_preqat", sc1, in1, sd1))
        if p3_steps > 0:
            # Candidate 2: reuse already-evaluated `p_postqat_cal`; only compute own FP32 ref!
            sc2, in2, sd2 = _score_mgvz_cand(best_q_sd, fp32_ref_t4=None, mgvz_pred=p_postqat_cal)
            cands.append(("light_qat", sc2, in2, sd2))

        # Candidate 3: FP16 WiSE-FT Model Soup (averages converging Phase-2 EMA snapshots + optional FP16 anchor grid!)
        soup_best_sd: dict[str, torch.Tensor] | None = None
        if len(p2_soup_snaps) >= 2:
            avg_sd: dict[str, torch.Tensor] = {}
            for k, v in best_p2_sd.items():
                if v.is_floating_point():
                    acc = v.float() * 0.50
                    w_each = 0.50 / float(len(p2_soup_snaps))
                    for sn in p2_soup_snaps:
                        acc = acc + sn[k].float() * w_each
                    avg_sd[k] = acc.to(v.dtype)
                else:
                    avg_sd[k] = v.clone()
            soup_best_sd = avg_sd
        if args.wise_anchor_fp16 is not None and args.wise_anchor_fp16.is_file():
            ck_anc = torch.load(args.wise_anchor_fp16, map_location="cpu", weights_only=False)
            sd_anc = ck_anc.get("state_dict", ck_anc.get("model_state_dict", ck_anc))
            for alpha_w in (0.15, 0.30, 0.50, 0.70, 0.85):
                blend_sd: dict[str, torch.Tensor] = {}
                for k, v in best_p2_sd.items():
                    if k in sd_anc and sd_anc[k].shape == v.shape and v.is_floating_point():
                        blend_sd[k] = ((1.0 - alpha_w) * sd_anc[k].float() + alpha_w * v.float()).to(v.dtype)
                    else:
                        blend_sd[k] = v.clone()
                sc_w, in_w, sd_w = _score_mgvz_cand(blend_sd)
                cands.append((f"wise_ft_a{int(alpha_w*100)}", sc_w, in_w, sd_w))
        if soup_best_sd is not None:
            sc3, in3, sd3 = _score_mgvz_cand(soup_best_sd)
            cands.append(("fp16_ema_soup", sc3, in3, sd3))

        cands.sort(key=lambda x: -x[1])
        win_name, _, win_info, win_sd = cands[0]
        ship_gate_report = {
            "enabled": True,
            "winner": win_name,
            "winner_info": win_info,
            "candidates": {nm: inf for nm, _, inf, _ in cands},
        }
        print(f"  [v1.6.1-AutoShipGate] WINNER = {win_name} | {json.dumps(ship_gate_report['candidates'], ensure_ascii=False)}", flush=True)
        model.load_state_dict(win_sd)
        pack_bundle_v10_v160(model, postqat_mgvz_tmp)
        load_bundle_v10_v160_into_model(m_dep, postqat_mgvz_tmp)
    phase_timings["auto_ship_gate_sec"] = round(time.time() - t_phase, 2)

    t_phase = time.time()
    p_cal = predict_student_v160(
        m_dep, fcache, cal_txts, device, batch_size=eval_bs, r0_only=False, outer_steps=6,
        use_crossview_dt=args.use_slot9_dt, record_diagnostics=True, precollated=cal_pre_deep, use_amp=False,
    )
    joint_jac = measure_joint_directional_jacobian_v160(m_dep, fcache, cal_txts, device, outer_steps=6)
    probe_3seg = run_order_preserving_3seg_probe_v160(m_dep, fcache, corpus["val_gold_v2"] + corpus["val_wave3"], device)

    quant_drift_audit = {
        "direct_packed_preqat_vs_fp32": _quant_drift_summary(p_fp32_master, p_direct_pack),
        "post_qat_raw_vs_fp32": _quant_drift_summary(p_fp32_master, p_postqat_cal),
        "deployed_winner_vs_fp32": _quant_drift_summary(p_fp32_master, p_cal),
        "auto_ship_gate": ship_gate_report,
    }

    # -----------------------------------------------------------------------
    # Phase 4: Disjoint `cal_fit / cal_cert` Split + Monotone OOF Fusion + Exact Clopper-Pearson + Export + 23-Gate CI
    # -----------------------------------------------------------------------
    fit_mask = np.array([(int(text_sha16_v140(r["text"]), 16) % 2) == 0 for r in cal_rows], dtype=bool)
    cert_mask = ~fit_mask
    if fit_mask.sum() < 24 or cert_mask.sum() < 24 or y_cal_int[fit_mask].sum() < 6 or y_cal_int[cert_mask].sum() < 6:
        fit_mask = np.ones((len(cal_rows),), dtype=bool)
        cert_mask = fit_mask.copy()

    # Fit strictly monotone OOF late fusion (`v1.6.2` Prior-Matched `EBA-H` OR legacy `monotone_linear`) strictly on `cal_fit`
    eba_m_pos_grid = tuple(float(x) for x in str(args.eba_m_pos_grid).split(",") if x.strip()) or (0.8, 1.2, 1.6, 2.0)
    eba_m_neg_grid = tuple(float(x) for x in str(args.eba_m_neg_grid).split(",") if x.strip()) or (1.0, 1.5, 2.0, 2.5)
    if use_cv_fusion and args.use_eba_h:
        cv_fit_raw = fit_crossview_eba_h_fusion(
            p_cal["logit_ai_r0"][fit_mask], p_cal["logit_ai_t4"][fit_mask], y_cal_ai[fit_mask],
            pi_nat=float(args.eba_pi_nat), m_pos_grid=eba_m_pos_grid, m_neg_grid=eba_m_neg_grid,
            cert_r0=p_cal["logit_ai_r0"][cert_mask], cert_deep=p_cal["logit_ai_t4"][cert_mask], cert_y=y_cal_ai[cert_mask],
        )
        cv_fusion = {k: v for k, v in cv_fit_raw.items() if not k.startswith("_")}
    elif use_cv_fusion:
        cv_fit_raw = fit_crossview_late_fusion(p_cal["logit_ai_r0"][fit_mask], p_cal["logit_ai_t4"][fit_mask], y_cal_ai[fit_mask])
        cv_fusion = {k: v for k, v in cv_fit_raw.items() if not k.startswith("_")}
    else:
        cv_fusion = {"enabled": False, "formula": "monotone_linear", "w_a": 0.0, "w_b": 1.0, "w0": 0.0, "w1": 1.0, "bias": 0.0}

    cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
    _t_ai_in = (cal_fused_t4 - float(cv_fusion.get("prior_shift", 0.0))) if args.use_eba_h else cal_fused_t4
    t_ai = fit_scalar_temperature_v140(_t_ai_in[fit_mask], y_cal_ai[fit_mask])

    # Unbiased 4-way fusion comparison strictly on held-out `cal_cert` (overall + frozen-R0 surviving subset!)
    p_r0_cert = expit(np.clip(p_cal["logit_ai_r0"][cert_mask], -30.0, 30.0))
    surv_cert = (p_r0_cert > 0.04) & (p_r0_cert < 0.96)

    def _compare_4way(mask_sub: np.ndarray) -> dict[str, Any]:
        ys = y_cal_int[mask_sub]
        if len(ys) < 8 or ys.sum() < 2 or (len(ys) - ys.sum()) < 2:
            return {"n": int(len(ys)), "pos": int(ys.sum())}
        la = p_cal["logit_ai_r0"][mask_sub]
        lb = p_cal["logit_ai_t4"][mask_sub]
        l_50 = 0.5 * la + 0.5 * lb
        l_mf = cal_fused_t4[mask_sub]
        return {
            "n": int(len(ys)), "pos": int(ys.sum()),
            "r0_alone": {**_eval_bin(ys, la), "bce": _safe_bce(ys, la)},
            "deep_t4_alone": {**_eval_bin(ys, lb), "bce": _safe_bce(ys, lb)},
            "convex_50_50": {**_eval_bin(ys, l_50), "bce": _safe_bce(ys, l_50)},
            "monotone_oof_fit": {**_eval_bin(ys, l_mf), "bce": _safe_bce(ys, l_mf)},
        }

    cert_indices = np.where(cert_mask)[0]
    surv_global_mask = np.zeros((len(cal_rows),), dtype=bool)
    surv_global_mask[cert_indices[surv_cert]] = True
    fusion_cert_comp = {
        "cal_cert_all": _compare_4way(cert_mask),
        "cal_cert_r0_surviving": _compare_4way(surv_global_mask),
    }

    # Reuse `val_wave3` T4 promo logits directly from `p_cal`; only run inference on `val_promo_v11`!
    pr_cal_rows = corpus["val_wave3"] + corpus["val_promo_v11"]
    _w3_pr_r0 = p_cal["logit_promo_r0"][_n_g2 : _n_g2 + _n_w3]
    _w3_pr_t4 = p_cal["logit_promo_t4"][_n_g2 : _n_g2 + _n_w3]
    if corpus["val_promo_v11"]:
        _p_v11 = predict_student_v160(
            m_dep, fcache, [r["text"] for r in corpus["val_promo_v11"]], device,
            batch_size=eval_bs, r0_only=False, outer_steps=4, use_crossview_dt=args.use_slot9_dt, use_amp=False,
        )
        p_pr_cal = {
            "logit_promo_r0": np.concatenate([_w3_pr_r0, _p_v11["logit_promo_r0"]], axis=0),
            "logit_promo": np.concatenate([_w3_pr_t4, _p_v11["logit_promo"]], axis=0),
        }
    else:
        p_pr_cal = {"logit_promo_r0": _w3_pr_r0, "logit_promo": _w3_pr_t4}
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in pr_cal_rows], dtype=np.float32)
    t_pr = fit_scalar_temperature_v140(p_pr_cal["logit_promo"], y_cal_pr) if len(pr_cal_rows) else 1.0

    # Fit LUTs strictly on `cal_fit`
    lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"][fit_mask], -30, 30), y_cal_ai[fit_mask])
    if args.use_eba_h:
        # Exact Bayes output-shifted LUT preserves prior-matched EBA-H shift on `prob_ai` (fixes H02/M01/M02 FPs!)
        lut_ai_full = fit_output_shifted_promo_lut_v122(
            np.clip(cal_fused_t4[fit_mask] / max(1e-3, t_ai), -30, 30), y_cal_ai[fit_mask],
            pi_nat=float(args.eba_pi_nat), shift_damp=0.55,
        )
    else:
        lut_ai_full = fit_smooth_ai_lut_v122(np.clip(cal_fused_t4[fit_mask] / max(1e-3, t_ai), -30, 30), y_cal_ai[fit_mask])
    if len(pr_cal_rows) >= 8 and len(np.unique(y_cal_pr)) == 2:
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(np.clip(p_pr_cal["logit_promo_r0"], -30, 30), y_cal_pr, pi_nat=0.015)
        lut_pr_full = fit_output_shifted_promo_lut_v122(np.clip(p_pr_cal["logit_promo"] / max(1e-3, t_pr), -30, 30), y_cal_pr, pi_nat=0.015)
    else:
        dz, dy = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], np.float32), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(dz, dy, pi_nat=0.015)
        lut_pr_full = fit_output_shifted_promo_lut_v122(dz, dy, pi_nat=0.015)

    lut01 = float(apply_spline_temp_lut(np.array([0.01], dtype=np.float32), lut_ai_r0)[0])
    p_pr_wild = apply_spline_temp_lut(expit(np.clip(p_cal["logit_promo_r0"][cert_mask], -30, 30)), lut_pr_r0)
    ai_lut_chk = {"lut_ai_r0_at_001": round(lut01, 5)}
    pr_lut_chk = {"median_promo_wild": round(float(np.median(p_pr_wild)), 5)}

    # Disjoint Fit -> Cert OR Strict K-Fold OOF R0 LTT Selection (v1.6.1 unlocks 75-82% R0 exit with tight CP95 UB!)
    p_r0_all_cal = apply_spline_temp_lut(expit(np.clip(p_cal["logit_ai_r0"], -30, 30)), lut_ai_r0)
    p_r0_fit, p_r0_crt = p_r0_all_cal[fit_mask], p_r0_all_cal[cert_mask]
    def_fit, def_crt = p_cal["p_defer_ai_r0"][fit_mask], p_cal["p_defer_ai_r0"][cert_mask]
    pos_fit, neg_fit = y_cal_int[fit_mask] == 1, y_cal_int[fit_mask] == 0
    pos_crt, neg_crt = y_cal_int[cert_mask] == 1, y_cal_int[cert_mask] == 0

    r0_ai_pol: dict[str, Any] = {
        "enabled": False, "tau_ai_lo": 0.015, "defer_max": 0.35, "hi_exit_enabled": False,
        "tau_ai_hi": 0.992, "defer_max_hi": 0.15, "max_chars": 180,
        "cert_lo_human_exit_rate": 0.0, "cert_lo_ai_miss_cp95_ub": 1.0,
        "requires_frozen_natural_stream_audit": True,
    }
    if int(args.oof_cal_folds) >= 2 and int(y_cal_int.sum()) >= 30:
        # Strict K-fold stratified OOF LUT + full-pool certification (refits deployment LUTs on full cal_rows!)
        k_f = int(args.oof_cal_folds)
        pos_idx_c = np.where(y_cal_int == 1)[0]
        neg_idx_c = np.where(y_cal_int == 0)[0]
        f_ids = np.zeros((len(y_cal_int),), dtype=np.int64)
        for ki, ii in enumerate(pos_idx_c):
            f_ids[ii] = ki % k_f
        for ki, ii in enumerate(neg_idx_c):
            f_ids[ii] = ki % k_f
        oof_p_r0 = np.zeros((len(y_cal_int),), dtype=np.float32)
        for fid in range(k_f):
            va_m = f_ids == fid
            tr_m = ~va_m
            lut_f = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"][tr_m], -30, 30), y_cal_ai[tr_m])
            oof_p_r0[va_m] = apply_spline_temp_lut(expit(np.clip(p_cal["logit_ai_r0"][va_m], -30, 30)), lut_f)
        # Refit final deployment fusion, temperature & LUTs on full cal_rows (eliminating 50% data waste!)
        if use_cv_fusion:
            if args.use_eba_h:
                cv_fit_full = fit_crossview_eba_h_fusion(
                    p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], y_cal_ai,
                    pi_nat=float(args.eba_pi_nat), m_pos_grid=eba_m_pos_grid, m_neg_grid=eba_m_neg_grid,
                    n_folds=k_f,
                )
            else:
                cv_fit_full = fit_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], y_cal_ai)
            cv_fusion = {k: v for k, v in cv_fit_full.items() if not k.startswith("_")}
            cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
            _t_ai_full_in = (cal_fused_t4 - float(cv_fusion.get("prior_shift", 0.0))) if args.use_eba_h else cal_fused_t4
            t_ai = fit_scalar_temperature_v140(_t_ai_full_in, y_cal_ai)
        lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"], -30, 30), y_cal_ai)
        if args.use_eba_h:
            lut_ai_full = fit_output_shifted_promo_lut_v122(
                np.clip(cal_fused_t4 / max(1e-3, t_ai), -30, 30), y_cal_ai,
                pi_nat=float(args.eba_pi_nat), shift_damp=0.55,
            )
        else:
            lut_ai_full = fit_smooth_ai_lut_v122(np.clip(cal_fused_t4 / max(1e-3, t_ai), -30, 30), y_cal_ai)
        lut01 = float(apply_spline_temp_lut(np.array([0.01], dtype=np.float32), lut_ai_r0)[0])
        ai_lut_chk = {"lut_ai_r0_at_001": round(lut01, 5)}

        # Optional independent untouched DuckDB 2.0 certification holdout (NEVER seen in training OR LUT fit!)
        p_ext_r0, def_ext, y_ext = None, None, None
        if extra_cal_cert_rows:
            pr_ext = predict_student_v160(m_dep, fcache, [r["text"] for r in extra_cal_cert_rows], device, batch_size=eval_bs, r0_only=True, use_amp=False)
            p_ext_r0 = apply_spline_temp_lut(expit(np.clip(pr_ext["logit_ai_r0"], -30, 30)), lut_ai_r0)
            def_ext = pr_ext["p_defer_ai_r0"]
            y_ext = np.array([int(r["y_ai"] >= 0.5) for r in extra_cal_cert_rows], dtype=np.int32)

        all_pos = y_cal_int == 1
        all_neg = y_cal_int == 0
        def_all = p_cal["p_defer_ai_r0"]
        best_tau, best_dm, best_cov, best_ub, best_k, best_n = 0.010, 0.35, -1.0, 1.0, 0, int(all_pos.sum())
        best_ext_info: dict[str, Any] = {}
        for tau_a in (0.030, 0.020, 0.015, 0.012, 0.010, 0.008, 0.006, 0.004):
            for dm in (0.35, 0.28, 0.22):
                em_o = (oof_p_r0 <= tau_a) & (def_all <= dm)
                k_o = int((em_o & all_pos).sum())
                n_o = int(all_pos.sum())
                ub_o = _clopper_pearson_upper(k_o, n_o, alpha=0.05)
                ext_ok = True
                ext_d: dict[str, Any] = {}
                if p_ext_r0 is not None and y_ext is not None and def_ext is not None and int((y_ext == 1).sum()) >= 30:
                    em_e = (p_ext_r0 <= tau_a) & (def_ext <= dm)
                    k_e = int((em_e & (y_ext == 1)).sum())
                    n_e = int((y_ext == 1).sum())
                    ub_e = _clopper_pearson_upper(k_e, n_e, alpha=0.05)
                    # Combined pooled exact CP95 across disjoint OOF + independent holdout!
                    ub_comb = _clopper_pearson_upper(k_o + k_e, n_o + n_e, alpha=0.05)
                    ext_ok = (ub_e <= 0.06) and (ub_comb <= 0.05)
                    ext_d = {
                        "ext_cert_ai_miss_k_over_n": f"{k_e}/{n_e}",
                        "ext_cert_ai_miss_cp95_ub": round(ub_e, 5),
                        "ext_cert_human_exit_rate": round(float(em_e[y_ext == 0].mean()), 4) if int((y_ext == 0).sum()) else 0.0,
                        "combined_cp95_ub": round(ub_comb, 5),
                    }
                cov_o = float(em_o[all_neg].mean())
                if ub_o <= 0.058 and ext_ok and cov_o > best_cov:
                    best_cov, best_tau, best_dm, best_ub, best_k, best_n, best_ext_info = cov_o, tau_a, dm, ub_o, k_o, n_o, ext_d
        if best_cov > 0.0:
            r0_ai_pol.update({
                "enabled": True,
                "calibration_mode": f"oof_{k_f}fold" + ("_plus_ext_holdout" if best_ext_info else ""),
                "tau_ai_lo": best_tau, "defer_max": best_dm,
                "cert_lo_human_exit_rate": round(best_cov, 4),
                "cert_lo_ai_miss_k_over_n": f"{best_k}/{best_n}",
                "cert_lo_ai_miss_cp95_ub": round(best_ub, 5),
                **best_ext_info,
            })
            print(f"  [v1.6.1-OOF-R0-LTT UNLOCKED] {json.dumps(r0_ai_pol, ensure_ascii=False)}", flush=True)
    elif pos_fit.sum() >= 30 and pos_crt.sum() >= 30:
        best_tau, best_dm, best_cov = 0.03, 0.30, -1.0
        for tau_a in (0.06, 0.04, 0.03, 0.02, 0.015, 0.010, 0.006):
            for dm in (0.35, 0.28, 0.22):
                em_f = (p_r0_fit <= tau_a) & (def_fit <= dm)
                if float(em_f[pos_fit].mean()) <= 0.020 and float(em_f[neg_fit].mean()) > best_cov:
                    best_cov, best_tau, best_dm = float(em_f[neg_fit].mean()), tau_a, dm
        if best_cov > 0.0:
            em_c = (p_r0_crt <= best_tau) & (def_crt <= best_dm)
            k_miss = int((em_c & pos_crt).sum())
            n_pos_c = int(pos_crt.sum())
            miss_ub = _clopper_pearson_upper(k_miss, n_pos_c, alpha=0.05)
            r0_ai_pol.update({
                "enabled": bool(miss_ub <= 0.06),
                "tau_ai_lo": best_tau, "defer_max": best_dm,
                "cert_lo_human_exit_rate": round(float(em_c[neg_crt].mean()), 4),
                "cert_lo_ai_miss_k_over_n": f"{k_miss}/{n_pos_c}",
                "cert_lo_ai_miss_cp95_ub": round(miss_ub, 5),
            })
        if args.two_sided_r0_exit:
            # Per External GPT Section 4: exact Clopper-Pearson conservative natural-prior (pi=0.03) FDR UB <= 0.01!
            best_ht, best_hd, best_hr = 0.995, 0.15, -1.0
            for tau_h in (0.995, 0.990, 0.985, 0.980):
                for dm_h in (0.15, 0.22):
                    hm_f = (p_r0_fit >= tau_h) & (def_fit <= dm_h)
                    if int((hm_f & neg_fit).sum()) == 0 and float(hm_f[pos_fit].mean()) > best_hr:
                        best_hr, best_ht, best_hd = float(hm_f[pos_fit].mean()), tau_h, dm_h
            hm_c = (p_r0_crt >= best_ht) & (def_crt <= best_hd)
            k_fp = int((hm_c & neg_crt).sum())
            n_neg_c = int(neg_crt.sum())
            k_tp = int((hm_c & pos_crt).sum())
            n_pos_c = int(pos_crt.sum())
            u_f = _clopper_pearson_upper(k_fp, n_neg_c, alpha=0.025)
            l_r = _clopper_pearson_lower(k_tp, n_pos_c, alpha=0.025)
            fdr_ub = (0.97 * u_f) / max(1e-9, 0.03 * l_r + 0.97 * u_f)
            r0_ai_pol.update({
                "hi_exit_enabled": bool(fdr_ub <= 0.01 and k_tp >= 20),
                "tau_ai_hi": best_ht, "defer_max_hi": best_hd,
                "cert_hi_fp_k_over_n": f"{k_fp}/{n_neg_c}",
                "cert_hi_tp_k_over_n": f"{k_tp}/{n_pos_c}",
                "cert_hi_nat3pct_cp95_fdr_ub": round(float(fdr_ub), 5),
            })
    r0_pr_pol = {"enabled": bool(r0_ai_pol["enabled"]), "tau_promo": 0.05, "defer_max": 0.35, "max_chars": 180}
    phase_timings["cal_and_ltt_sec"] = round(time.time() - t_phase, 2)

    t_phase = time.time()
    manifest = {
        "version": args.version, "max_len": int(args.max_len), "head_len": int(args.head_len),
        "long_char_trigger": int(args.long_char_trigger), "temp_ai": t_ai, "temp_promo": t_pr,
        "lut_ai_r0": lut_ai_r0, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_full": lut_pr_full,
        "rung0_ai_ltt_policy": r0_ai_pol, "rung0_pr_ltt_policy": r0_pr_pol,
        "crossview_late_fusion": cv_fusion,
        "long_focal_router": {
            "enabled": bool(args.use_kadane_focal_router),
            "win_chars": 160,
            "max_chunks": 6,
            "lambda": 0.92,
        },
        "anytime_step_policy": {
            "enabled": True, "max_infer_steps": 4, "u_halt_thresh": 0.015,
            "disagree_dt_thresh": 1.60, "uncertain_lo": 0.12, "uncertain_hi": 0.88, "t1_exit_enabled": True,
        },
    }

    export_rep = export_v160_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    eng = MeguvizV160CascadeEngine(out_dir, threads=4)
    ci_gates = run_v160_ci_gates(
        m_dep, tokenizer, export_rep=export_rep, ai_lut_check=ai_lut_chk, promo_lut_check=pr_lut_chk,
        charlm=full_charlm, release_dir=out_dir, engine=eng,
    )
    phase_timings["export_and_ci_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # Loop-Physics Scorecard (All 6 GPT Probes + T3-State -> T4-Readout Pure-Bias Control!)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    cal_traj = {
        "r0": _eval_bin(y_cal_int, p_cal["logit_ai_r0"]),
        "r1": _eval_bin(y_cal_int, p_cal["logit_ai_r1"]),
        **{f"t{k}": _eval_bin(y_cal_int, p_cal[f"logit_ai_t{k}"]) for k in (1, 2, 3, 4, 5, 6)},
        "t3_state_t4_readout": _eval_bin(y_cal_int, p_cal["logit_ai_t3_state_t4_readout"]),
        "t4_monotone_fused": _eval_bin(y_cal_int, cal_fused_t4),
    }

    # T3-state -> T4-readout pure-bias control (overall + frozen-R0 surviving hard subset)
    p_r0_all = expit(np.clip(p_cal["logit_ai_r0"], -30.0, 30.0))
    surv_all = (p_r0_all > 0.04) & (p_r0_all < 0.96)

    def _t4_bias_ctrl(mask_s: np.ndarray) -> dict[str, Any]:
        ys = y_cal_int[mask_s]
        if len(ys) < 8 or ys.sum() < 2 or (len(ys) - ys.sum()) < 2:
            return {"n": int(len(ys))}
        l3 = p_cal["logit_ai_t3"][mask_s]
        l3_ro4 = p_cal["logit_ai_t3_state_t4_readout"][mask_s]
        l4 = p_cal["logit_ai_t4"][mask_s]
        b3, b34, b4 = _safe_bce(ys, l3), _safe_bce(ys, l3_ro4), _safe_bce(ys, l4)
        return {
            "n": int(len(ys)),
            "t3_pr": _eval_bin(ys, l3)["pr"], "t3_bce": b3,
            "t3_state_t4_readout_pr": _eval_bin(ys, l3_ro4)["pr"], "t3_state_t4_readout_bce": b34,
            "t4_real_pr": _eval_bin(ys, l4)["pr"], "t4_real_bce": b4,
            "net_bce_gain_vs_t3": round(b3 - b4, 5),
            "net_bce_gain_vs_t3_state_t4_readout": round(b34 - b4, 5),
            "mean_shift_t4_minus_t3": round(float((l4 - l3).mean()), 5),
            "mean_shift_t3_ro4_minus_t3": round(float((l3_ro4 - l3).mean()), 5),
        }

    t3_t4_bias_ctrl = {"all_cal": _t4_bias_ctrl(np.ones((len(cal_rows),), dtype=bool)), "r0_surviving": _t4_bias_ctrl(surv_all)}

    d34 = p_cal["logit_ai_t4"] - p_cal["logit_ai_t3"]
    pos_imp = int(((y_cal_int == 1) & (d34 > 0.05)).sum())
    pos_deg = int(((y_cal_int == 1) & (d34 < -0.05)).sum())
    neg_exo = int(((y_cal_int == 0) & (d34 < -0.05)).sum())
    neg_deg = int(((y_cal_int == 0) & (d34 > 0.05)).sum())
    t6_over_t4_std = float(np.std(p_cal["logit_ai_t6"]) / max(1e-6, float(np.std(p_cal["logit_ai_t4"]))))

    p_cal_nor0 = predict_student_v160(
        m_dep, fcache, cal_txts, device, batch_size=eval_bs, r0_only=False, outer_steps=4,
        r0_drop_eval=True, precollated=cal_pre_deep, use_amp=False,
    )
    zero_laziness = {
        "deep_t4_with_zchar_pr": cal_traj["t4"]["pr"],
        "deep_t4_zchar_dropped_pr": _eval_bin(y_cal_int, p_cal_nor0["logit_ai_t4"])["pr"],
        "r0_alone_pr": cal_traj["r0"]["pr"],
        "t4_monotone_fused_pr": cal_traj["t4_monotone_fused"]["pr"],
    }

    loop_physics = {
        "cal_trajectory": cal_traj,
        "t1_minus_r1_pr": round(cal_traj["t1"]["pr"] - cal_traj["r1"]["pr"], 4),
        "t3_minus_t2_pr": round(cal_traj["t3"]["pr"] - cal_traj["t2"]["pr"], 4),
        "t4_minus_t2_pr": round(cal_traj["t4"]["pr"] - cal_traj["t2"]["pr"], 4),
        "state_delta_ratio_s4_over_s3": p_cal.get("_state_delta_ratio_s4_over_s3", 0.0),
        "t6_over_t4_logit_std": round(t6_over_t4_std, 4),
        "proj_sat_rate_t1_t6": p_cal.get("_proj_sat_rate_t1_t6", {}),
        "adaln_gain_t1_t6": p_cal.get("_adaln_gain_t1_t6", {}),
        "period2_oscillation": p_cal.get("_period2_oscillation", {}),
        "joint_directional_jacobian_t1_t6": joint_jac,
        "t3_state_into_t4_readout_control": t3_t4_bias_ctrl,
        "bidir_t3_to_t4": {
            "pos_improved": pos_imp, "pos_degraded": pos_deg,
            "neg_exonerated": neg_exo, "neg_degraded": neg_deg,
            "ai_to_human_ratio": round(pos_imp / max(1.0, float(neg_exo)), 3),
        },
        "zero_laziness_audit": zero_laziness,
    }

    # Save Offline Prediction Cache (`.npz`): reuse exact `p_cal` slices for the 4 cal splits!
    cache_splits = ("val_gold_v2", "test_gold_v2", "val_wave3", "val_old_video_neg", "val_multilingual", "diag_god500")
    _cal_slice_map: dict[str, tuple[int, int]] = {
        "val_gold_v2": (0, _n_g2),
        "val_wave3": (_n_g2, _n_g2 + _n_w3),
        "val_old_video_neg": (_n_g2 + _n_w3, _n_g2 + _n_w3 + _n_old),
        "val_multilingual": (_n_g2 + _n_w3 + _n_old, _n_g2 + _n_w3 + _n_old + _n_ml),
    }
    cache_arrays: dict[str, np.ndarray] = {}
    for sk in cache_splits:
        srows = corpus.get(sk, [])
        if not srows:
            continue
        if sk in _cal_slice_map:
            cs0, cs1 = _cal_slice_map[sk]
            ps = {k: v[cs0:cs1] for k, v in p_cal.items() if isinstance(v, np.ndarray)}
        else:
            ps = predict_student_v160(
                m_dep, fcache, [r["text"] for r in srows], device, batch_size=eval_bs, outer_steps=6,
                use_crossview_dt=args.use_slot9_dt, record_diagnostics=True, use_amp=False,
            )
        cache_arrays[f"{sk}__y_ai"] = np.array([float(r.get("y_ai", 0.0)) for r in srows], dtype=np.float32)
        cache_arrays[f"{sk}__y_pr"] = np.array([float(r.get("y_pr", 0.0)) for r in srows], dtype=np.float32)
        for pk in ("logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0", "logit_ai_r1", "logit_promo_r1", "logit_ai_t3_state_t4_readout"):
            if pk in ps:
                cache_arrays[f"{sk}__{pk}"] = ps[pk]
        for k_st in range(1, 7):
            cache_arrays[f"{sk}__logit_ai_t{k_st}"] = ps[f"logit_ai_t{k_st}"]
            cache_arrays[f"{sk}__logit_promo_t{k_st}"] = ps[f"logit_promo_t{k_st}"]
            cache_arrays[f"{sk}__u_halt_t{k_st}"] = ps[f"u_halt_t{k_st}"]
    np.savez_compressed(out_dir / f"{args.version}_eval_cache.npz", **cache_arrays)

    # Final ONNX Test Readouts & Real ORT 4T B=1 Latency (reusing `eng`!)
    def _eval_onnx_set(rows: list[dict[str, Any]], force_depth: int | None = None) -> dict[str, Any]:
        if not rows:
            return {}
        out_r = eng.predict_batch([r["text"] for r in rows], max_batch=64, force_full=(force_depth is not None), force_depth=force_depth)
        y_a = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in rows], dtype=np.int32)
        lg_a = np.array([float(x["logit_ai"]) for x in out_r], dtype=np.float32)
        res_m = _eval_bin(y_a, lg_a)
        res_m["avg_rung_ai"] = round(float(np.mean([x["exit_rung_ai"] for x in out_r])), 3)
        res_m["r0_exit_rate"] = round(float(np.mean([x["exit_rung_ai"] == 0 for x in out_r])), 4)
        res_m["t1_exit_rate"] = round(float(np.mean([x["exit_rung_ai"] == 1 for x in out_r])), 4)
        return res_m

    ort_lat_b1: dict[str, Any] = {}
    for L in (32, 128, 256):
        f_t1 = {
            "input_ids": np.ones((1, L), np.int64), "attention_mask": np.ones((1, L), np.int64),
            "style_feats": np.zeros((1, STYLE_FEAT_DIM_V11), np.float32), "engram_ids": np.ones((1, L), np.int64),
            "trigram_ids": np.ones((1, L), np.int64), "local_charlm": np.zeros((1, L, LOCAL_CHARLM_DIM_V150), np.float32),
            "z_char": np.zeros((1, 64), np.float32), "support_pairs": np.array([24.0], np.float32),
            "quote_scope": np.zeros((1, L, QUOTE_SCOPE_DIM_V161), np.float32),
        }
        o1 = dict(zip(_T1_OUT_NAMES_V160, eng._run_sess(eng.sess_t1, eng._in_t1, f_t1)))
        f_st = {
            "ai_streams_in": o1["ai_streams"], "pr_stream_in": o1["pr_stream"], "init_slots": o1["init_slots"],
            "cov_prev": o1["cov"], "k_ev_0": o1["k_ev_0"], "v_ev_0": o1["v_ev_0"], "attention_mask": f_t1["attention_mask"],
            "k_haar": o1["k_haar"], "v_haar": o1["v_haar"], "haar_mask": o1["haar_mask"],
            "h_pool_0": o1["h_pool_0"], "pr_attr_0": o1["pr_attr_0"], "st_ai": o1["st_ai"], "st_pr": o1["st_pr"],
            "z_char": f_t1["z_char"], "g_sup": o1["g_sup"], "emb_ai_prev": o1["emb_ai_t1"],
            "logit_ai_prev": o1["logit_ai_t1"], "t_val": np.array([1.0], np.float32), "disagree_dt": np.array([0.8], np.float32),
        }
        for _ in range(3):
            eng._run_sess(eng.sess_t1, eng._in_t1, f_t1)
            eng._run_sess(eng.sess_step, eng._in_step, f_st)
        n_it = 12
        t0 = time.perf_counter()
        for _ in range(n_it):
            eng._run_sess(eng.sess_t1, eng._in_t1, f_t1)
        ms_t1 = (time.perf_counter() - t0) * 1000.0 / n_it
        t0 = time.perf_counter()
        for _ in range(n_it):
            eng._run_sess(eng.sess_step, eng._in_step, f_st)
        ms_st = (time.perf_counter() - t0) * 1000.0 / n_it
        ort_lat_b1[f"L={L}"] = {
            "t1_ms": round(ms_t1, 3), "step_ms": round(ms_st, 3),
            "t2_ms": round(ms_t1 + ms_st, 3), "t4_ms": round(ms_t1 + 3.0 * ms_st, 3),
        }

    test_g2_onnx_cond = _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=None)
    test_g2_onnx_t4 = _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=4)
    val_ml_onnx = _eval_onnx_set(corpus.get("val_multilingual", []), force_depth=None)
    diag_god500_onnx = _eval_onnx_set(corpus.get("diag_god500", []), force_depth=None)
    phase_timings["eval_cache_and_onnx_sec"] = round(time.time() - t_phase, 2)

    report = {
        "version": args.version, "arch": args.arch, "dynamics": args.dynamics,
        "use_discourse_ops": bool(args.use_discourse_ops),
        "lardo_w8_core": bool(args.lardo_w8_core),
        "oof_cal_folds": int(args.oof_cal_folds),
        "auto_ship_gate": bool(args.auto_ship_gate),
        "use_step_phi": use_step_phi, "step_decay_mode": args.step_decay_mode,
        "use_ncl": bool(args.use_ncl), "use_slot9_dt": bool(args.use_slot9_dt),
        "expand_tables": bool(args.expand_tables), "freeze_r0": bool(args.freeze_r0),
        "t_train_mode": args.t_train_mode, "gpu_name": gpu_name,
        "turbo_v3": {
            "rust_frontend": use_rust_fe, "compile": compiled_ok, "compile_mode": args.compile_mode,
            "compile_dynamic": bool(args.compile_dynamic), "amp_bf16": use_amp,
            "fuse_aux": bool(args.fuse_aux), "eval_batch_size": eval_bs, "prefetch_workers": int(args.prefetch_workers),
        },
        "v162_hooks": {
            "use_eba_h": bool(args.use_eba_h), "eba_pi_nat": float(args.eba_pi_nat),
            "use_tcbr_r": bool(args.use_tcbr_r), "use_d02_paraphrase_cf": bool(args.use_d02_paraphrase_cf),
            "r0_antidilution_mix": bool(args.r0_antidilution_mix), "longform_every": int(args.longform_every),
            "use_kadane_focal_router": bool(args.use_kadane_focal_router),
            "puc_warmup_steps": int(puc_w), "puc_shrink": float(args.puc_shrink), "puc_noise": float(args.puc_noise),
            "puc_grad_log": puc_grad_log,
        },
        "total_params": tot_p, "core_params": core_p,
        "phase1_r0_best_obj": round(float(best_r0_obj), 5),
        "phase2_fp16_val": best_p2_ev, "phase3_qat_best_obj": round(float(best_q_obj), 5),
        "loop_physics": loop_physics,
        "quant_drift_3ckpt_audit": quant_drift_audit,
        "fusion_cal_cert_comparison": fusion_cert_comp,
        "order_preserving_3seg_probe": probe_3seg,
        "ort_4t_b1_latency_ms": ort_lat_b1,
        "test_bili_gold_v2_onnx_cond": test_g2_onnx_cond,
        "test_bili_gold_v2_onnx_full_t4": test_g2_onnx_t4,
        "val_multilingual_onnx": val_ml_onnx,
        "diag_god500_onnx": diag_god500_onnx,
        "rung0_ai_ltt_policy": r0_ai_pol,
        "crossview_late_fusion": cv_fusion,
        "export": export_rep,
        "v160_ci_gates": ci_gates,
        "release_gates": {
            "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v160_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "phase_timings_sec": phase_timings,
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    print(f"[Turbo-v3 Timings] total={report['elapsed_seconds']}s | {json.dumps(phase_timings)}", flush=True)
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["release_gates"]["v160_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
