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
import json
import math
from pathlib import Path
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
from .v08_zenith_ops import OfflineEraGRLHead
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
from .v160_stratosphere_loop import (
    ARCH_PRESETS_V160,
    MeguvizV160CascadeEngine,
    MeguvizV160StratosphereLoopModel,
    _STEP_OUT_NAMES_V160,
    _T1_OUT_NAMES_V160,
    apply_crossview_late_fusion,
    export_v160_bootable_and_onnx,
    extract_apex_char_batch_v160,
    fit_crossview_late_fusion,
    load_bundle_v10_v160_into_model,
    pack_bundle_v10_v160,
    run_v160_ci_gates,
    set_v160_qat_mode,
)


class _StrictGroupOOFFeatureCacheV160:
    """Strictly Leak-Free Feature Cache supporting frozen or expanded `(engram_buckets, trigram_buckets)`."""

    def __init__(
        self,
        tokenizer: BiliPickyBPETokenizer,
        max_len: int,
        head_len: int,
        engram_buckets: int,
        trigram_buckets: int,
        use_local_charlm: bool = True,
        max_entries: int = 300000,
    ) -> None:
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.engram_buckets = int(engram_buckets)
        self.trigram_buckets = int(trigram_buckets)
        self.use_local_charlm = bool(use_local_charlm)
        self.max_entries = int(max_entries)

        self.fold_models: list[CompactCharLMV150] = []
        self.text_to_fold: dict[str, int] = {}
        self.full_lm: CompactCharLMV150 | None = None

        self._bpe_cache: dict[str, tuple[str, list[int], list[tuple[int, int]], list[int], list[int], np.ndarray]] = {}
        self._lm_global_cache: dict[tuple[int, str], np.ndarray] = {}
        self._lm_local_cache: dict[tuple[int, str], np.ndarray] = {}

    def set_group_oof_charlm(
        self, fold_models: list[CompactCharLMV150], text_to_fold: dict[str, int], full_lm: CompactCharLMV150
    ) -> None:
        self.fold_models = fold_models
        self.text_to_fold = text_to_fold
        self.full_lm = full_lm
        self._lm_global_cache.clear()
        self._lm_local_cache.clear()

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
) -> dict[str, torch.Tensor]:
    cleans: list[str] = []
    g9_list: list[np.ndarray] = []
    bpe_pack = []
    for r in batch:
        t0 = str(r["text"])
        is_ai_pos = (r.get("has_ai", 0.0) > 0.5 and r.get("y_ai", 0.0) >= 0.70) or (r.get("has_teacher", 0.0) > 0.5 and r.get("t_ai_lg", -5.0) >= 1.2)
        if rng is not None and punct_jitter_prob > 0.0 and is_ai_pos:
            t1 = jitter_ai_text_v140(t0, rng, is_ml_en=bool(r.get("is_ml_en")), prob=0.28 if r.get("is_ml_en") else punct_jitter_prob)
            is_m = (t1 != t0)
        else:
            t1, is_m = t0, False
        cl, ids, offs, eng, trg, f56 = fcache.get_bpe_style_offsets(t1, allow_store=not is_m)
        cleans.append(cl)
        g9_list.append(fcache.get_global_9d(cl, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m))
        if need_bpe:
            loc8 = fcache.get_local_8ch(cl, offs, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m)
            bpe_pack.append((ids, eng, trg, f56, loc8))

    g9_arr = np.stack(g9_list, axis=0).astype(np.float32) if g9_list else np.zeros((0, CHARLM_FEAT_DIM_V140), dtype=np.float32)
    c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v160(
        cleans, b2=r0_b2, b3=r0_b3, already_canonical=True, precomputed_charlm_9d=g9_arr
    )

    out: dict[str, torch.Tensor] = {
        "c1": torch.from_numpy(c1).to(device, non_blocking=True),
        "c2": torch.from_numpy(c2).to(device, non_blocking=True),
        "c3": torch.from_numpy(c3).to(device, non_blocking=True),
        "cm": torch.from_numpy(cm).to(device, non_blocking=True),
        "cf": torch.from_numpy(cf).to(device, non_blocking=True),
        "sup": torch.from_numpy(sups).to(device, non_blocking=True),
        "t_ai_lg": torch.tensor([float(r.get("t_ai_lg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "t_ai_r0": torch.tensor([float(r.get("t_ai_r0", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t1": torch.tensor([float(r.get("t_ai_t1", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t2": torch.tensor([float(r.get("t_ai_t2", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t3": torch.tensor([float(r.get("t_ai_t3", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "frontier_gap": torch.tensor([float(r.get("frontier_gap", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "t_pr_lg": torch.tensor([float(r.get("t_pr_lg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_t": torch.tensor([float(r.get("has_teacher", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "kd_w": torch.tensor([float(r.get("kd_w", 1.0)) for r in batch], dtype=torch.float32, device=device),
        "y_ai": torch.tensor([float(r.get("y_ai", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_ai": torch.tensor([float(r.get("has_ai", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "y_pr": torch.tensor([float(r.get("y_pr", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_pr": torch.tensor([float(r.get("has_pr", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "sub4": torch.tensor([int(r.get("sub4", -1)) for r in batch], dtype=torch.long, device=device),
        "hard_neg": torch.tensor([float(r.get("hard_neg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "era5": torch.tensor([int(r.get("era5", -1)) for r in batch], dtype=torch.long, device=device),
    }
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
        f56_l = []
        for i, (ids, eng, trg, f56, loc8) in enumerate(bpe_pack):
            ns = len(ids)
            ids_a[i, :ns], msk_a[i, :ns], eg_a[i, :ns], tg_a[i, :ns] = ids, 1, eng, trg
            loc_a[i, :ns] = loc8
            f56_l.append(f56)
        out["ids"] = torch.from_numpy(ids_a).to(device, non_blocking=True)
        out["msk"] = torch.from_numpy(msk_a).to(device, non_blocking=True)
        out["eg"] = torch.from_numpy(eg_a).to(device, non_blocking=True)
        out["tg"] = torch.from_numpy(tg_a).to(device, non_blocking=True)
        out["loc"] = torch.from_numpy(loc_a).to(device, non_blocking=True)
        out["st"] = torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device, non_blocking=True)
    return out


@torch.no_grad()
def predict_student_v160(
    model: MeguvizV160StratosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV160,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    r0_only: bool = False,
    outer_steps: int = 6,
    use_crossview_dt: bool = False,
    r0_drop_eval: bool = False,
    record_diagnostics: bool = False,
) -> dict[str, Any]:
    n = len(texts)
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
    order = sorted(range(n), key=lambda i: len(texts[i]))

    # Step-wise physics telemetry accumulators (External GPT Section 2!)
    sat_acc: list[list[float]] = [[] for _ in range(outer_steps)]
    adaln_acc: list[list[float]] = [[] for _ in range(outer_steps)]
    d1_acc: dict[int, list[float]] = {t: [] for t in range(2, outer_steps + 1)}  # ||s_t - s_{t-1}||
    d2_acc: dict[int, list[float]] = {t: [] for t in range(3, outer_steps + 1)}  # ||s_t - s_{t-2}||

    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_b = [{"text": texts[i]} for i in idxs]
        bk = _collate_v160_batch(
            sub_b, fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0,
            need_bpe=not r0_only, use_oof_charlm=False,
        )
        if r0_only:
            res = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        else:
            res = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=outer_steps,
                r0_drop_p=1.0 if r0_drop_eval else 0.0,
                dt_drop_p=1.0 if r0_drop_eval else 0.0,
                use_crossview_dt=use_crossview_dt and (not r0_drop_eval),
                return_diagnostics=record_diagnostics,
            )
            if record_diagnostics:
                n_ai = model.core.n_ai_slots
                s_list = res["slot_steps"]
                for ti in range(len(res.get("proj_sat_steps", []))):
                    sat_acc[ti].extend(res["proj_sat_steps"][ti].float().cpu().tolist())
                    adaln_acc[ti].append(float(res["adaln_gain_steps"][ti].float().item()))
                for t in range(2, len(s_list) + 1):
                    d1 = (s_list[t - 1][:, :n_ai] - s_list[t - 2][:, :n_ai]).flatten(1).norm(dim=1)
                    d1_acc[t].extend(d1.float().cpu().tolist())
                for t in range(3, len(s_list) + 1):
                    d2 = (s_list[t - 1][:, :n_ai] - s_list[t - 3][:, :n_ai]).flatten(1).norm(dim=1)
                    d2_acc[t].extend(d2.float().cpu().tolist())
        for k in keys:
            if k in res:
                arr = res[k].float().cpu().numpy()
                for j, oi in enumerate(idxs):
                    out[k][oi] = arr[j]

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
    bk = _collate_v160_batch(sub, fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True, use_oof_charlm=False)
    r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
    s1 = model.forward_prelude_and_t1(
        bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"], z_char=r0["z_char"], support_pairs=bk["sup"]
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
        bk = _collate_v160_batch(sub, fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True, use_oof_charlm=False)
        r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        s1 = model.forward_prelude_and_t1(
            bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
            z_char=r0["z_char"], support_pairs=bk["sup"], return_diagnostics=True,
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
    ap.add_argument("--dynamics", type=str, default="stratosphere", choices=["stratosphere", "v150_anchor"])
    ap.add_argument("--t-train-mode", type=str, default="uniform_1_6", choices=["uniform_1_6", "fixed_2_4"])
    ap.add_argument("--use-step-phi", action="store_true", help="Enable continuous Fourier step embedding phi(t) (Arm A2)")
    ap.add_argument("--no-step-phi", action="store_true", help="Explicitly disable Fourier step embedding (A0/A1/A3/A4/A5)")
    ap.add_argument("--step-decay-mode", type=str, default="inv_sqrt", choices=["inv_sqrt", "const"])
    ap.add_argument("--use-slot9-dt", action="store_true", help="Legacy opt-in ONLY: inject d_t into AI Slot-9 (OFF in A0-A5)")
    ap.add_argument("--expand-tables", action="store_true", help="Round-2 opt-in ONLY: 2x R0++/Engram lookup tables (OFF in Round-1)")
    ap.add_argument("--init-r0-ckpt", type=Path, default=None, help="Load shared frozen Phase-1 R0 state_dict across A0-A5")
    ap.add_argument("--save-r0-ckpt", type=Path, default=None, help="Save trained Phase-1 R0 state_dict for reuse across A0-A5")
    ap.add_argument("--freeze-r0", action="store_true", help="Freeze R0 weights during Phase-2 & Phase-3 (locked A0-A5 protocol)")
    ap.add_argument("--no-prelude-global-se", action="store_true")
    ap.add_argument("--no-coverage-beta", action="store_true")
    ap.add_argument("--no-local-charlm", action="store_true")
    ap.add_argument("--no-mono-loss", action="store_true")
    ap.add_argument("--mono-eps", type=float, default=0.05)
    ap.add_argument("--use-ncl", action="store_true", help="Enable NCL falsification arm (Arm A3 ONLY)")
    ap.add_argument("--ncl-gamma", type=float, default=0.12)
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
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    use_step_phi = bool(args.use_step_phi) and (not bool(args.no_step_phi))
    use_cv_fusion = not bool(args.no_crossview_fusion)

    if args.ablation_fast and not args.smoke_test:
        args.p1_steps = min(args.p1_steps, 1200)
        args.p2_steps = min(args.p2_steps, 2600)
        args.p3_qat_steps = min(args.p3_qat_steps, 800)

    t_all = time.time()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir or out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Zero-Leakage Corpus + Super-Teacher 251K Soft Pool
    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir, gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, delta_dir=args.delta_dir, v04_data_path=args.v04_data,
        smoke_test=args.smoke_test,
    )
    blocked = {
        text_sha16_v140(r["text"])
        for k in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "val_multilingual", "diag_god500")
        for r in corpus.get(k, [])
    }
    distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)
    if not distill_pool:
        raise SystemExit(f"Empty distill pool: {args.distill_pool}")

    d_hard = [r for r in distill_pool if r.get("frontier_gap", 0.0) >= 1.3 or r.get("hard_neg", 0.0) > 0.5]
    d_pos = [r for r in distill_pool if r.get("t_ai_lg", -5.0) >= 0.8]
    d_promo = [r for r in distill_pool if r.get("t_pr_lg", -5.0) >= 0.5]
    d_all = distill_pool
    gold_pos, gold_hd, gold_nat, gold_pr = corpus["train_ai_pos"], corpus["train_hu_hard"], corpus["train_hu_natural"], corpus["train_promo_extra"]
    longform_train = read_jsonl_any_v140(args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 1200)]

    # 2. Model + Group-Clustered 3-Fold OOF CharLM
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
    )
    model = MeguvizV160StratosphereLoopModel(**model_kwargs).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_END_V130, num_eras=NUM_ERAS_V122).to(device)

    fcache = _StrictGroupOOFFeatureCacheV160(
        tokenizer, max_len=int(args.max_len), head_len=int(args.head_len),
        engram_buckets=model.engram_buckets, trigram_buckets=model.trigram_buckets,
        use_local_charlm=not args.no_local_charlm,
    )
    charlm_rows = gold_pos + gold_hd + gold_nat
    if args.smoke_test:
        charlm_rows = charlm_rows[:240]
    lm_txts = [r["text"] for r in charlm_rows]
    lm_y = np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in charlm_rows], dtype=np.int32)
    lm_grps = [str(r["cf_group_id"]) if r.get("cf_group_id") else None for r in charlm_rows]
    fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
        lm_txts, lm_y, group_keys=lm_grps, n_folds=3,
        min_counts=(2, 3, 4) if args.smoke_test else (16, 28, 40),
        max_order_keys=(3000, 1500, 1000) if args.smoke_test else (9000, 8000, 7000),
        seed=args.seed,
    )
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)

    tot_p = sum(p.numel() for p in model.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    print(
        f"[v1.6.0-Locked] ver={args.version} arch={args.arch} dyn={args.dynamics} step_phi={use_step_phi} "
        f"step_decay={args.step_decay_mode} ncl={args.use_ncl} slot9_dt={args.use_slot9_dt} "
        f"expand_tbl={args.expand_tables} freeze_r0={args.freeze_r0} "
        f"params={tot_p:,} (core={core_p:,}) distill={len(distill_pool):,} gpu={gpu_name}",
        flush=True,
    )

    def _draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rng.choices(pool, k=k) if (pool and k > 0) else []

    def _sample_batch(bs: int) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        out_b.extend(_draw(d_all, int(bs * 0.24)))
        out_b.extend(_draw(d_hard, int(bs * 0.14)))
        out_b.extend(_draw(d_pos, int(bs * 0.08)))
        out_b.extend(_draw(d_promo, int(bs * 0.06)))
        out_b.extend(_draw(gold_pos, int(bs * 0.15)))
        out_b.extend(_draw(gold_hd, int(bs * 0.15)))
        out_b.extend(_draw(gold_nat, int(bs * 0.11)))
        out_b.extend(_draw(gold_pr, max(1, bs - len(out_b))))
        return out_b[:bs]

    def _eval_val(m_cur: MeguvizV160StratosphereLoopModel, r0_only: bool = False, outer_eval: int = 4) -> dict[str, Any]:
        p_g2 = predict_student_v160(m_cur, fcache, [r["text"] for r in corpus["val_gold_v2"]], device, r0_only=r0_only, outer_steps=outer_eval, use_crossview_dt=args.use_slot9_dt)
        p_w3 = predict_student_v160(m_cur, fcache, [r["text"] for r in corpus["val_wave3"]], device, r0_only=r0_only, outer_steps=outer_eval, use_crossview_dt=args.use_slot9_dt)
        p_old = predict_student_v160(m_cur, fcache, [r["text"] for r in corpus["val_old_video_neg"]], device, r0_only=r0_only, outer_steps=outer_eval, use_crossview_dt=args.use_slot9_dt)
        k_ai = "logit_ai_r0" if r0_only else f"logit_ai_t{outer_eval}"
        k_pr = "logit_promo_r0" if r0_only else "logit_promo"
        yg2 = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
        yw3_a = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        yw3_p = np.array([int(r["y_pr"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        mg2 = _eval_bin(yg2, p_g2[k_ai])
        mw3a = _eval_bin(yw3_a, p_w3[k_ai])
        mw3p = _eval_bin(yw3_p, p_w3[k_pr])
        old_fpr = float((expit(np.clip(p_old[k_ai], -30, 30)) >= 0.5).mean()) if len(p_old[k_ai]) else 0.0
        q4 = (yw3_p == 1) & (yw3_a == 0)
        q4_fpr = float((expit(np.clip(p_w3[k_ai][q4], -30, 30)) >= 0.5).mean()) if q4.sum() else 0.0

        ml_pr = 0.0
        if corpus.get("val_multilingual"):
            p_ml = predict_student_v160(m_cur, fcache, [r["text"] for r in corpus["val_multilingual"]], device, r0_only=r0_only, outer_steps=outer_eval, use_crossview_dt=args.use_slot9_dt)
            yml = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_multilingual"]], dtype=np.int32)
            ml_pr = _eval_bin(yml, p_ml[k_ai])["pr"]

        obj = 0.30 * mg2["pr"] + 0.24 * mg2["r95"] + 0.16 * mw3a["pr"] + 0.16 * mw3p["pr"] + 0.14 * ml_pr - 1.30 * old_fpr - 0.50 * q4_fpr
        res_d: dict[str, Any] = {
            "sel_obj": round(float(obj), 5),
            "g2_pr": mg2["pr"], "g2_r95": mg2["r95"], "g2_r99": mg2["r99"],
            "w3_ai_pr": mw3a["pr"], "w3_pr_pr": mw3p["pr"], "ml_pr": ml_pr,
            "old_fpr": round(old_fpr, 4), "q4_fpr": round(q4_fpr, 4),
        }
        if not r0_only:
            res_d["g2_r1_pr"] = _eval_bin(yg2, p_g2["logit_ai_r1"])["pr"]
            res_d["g2_t1_pr"] = _eval_bin(yg2, p_g2["logit_ai_t1"])["pr"]
            res_d["g2_t2_pr"] = _eval_bin(yg2, p_g2["logit_ai_t2"])["pr"]
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
    p2_steps = 6 if args.smoke_test else int(args.p2_steps)
    p3_steps = 0 if args.skip_qat else (4 if args.smoke_test else int(args.p3_qat_steps))

    # -----------------------------------------------------------------------
    # Phase 1: Shared Frozen `R0` Char-View (trained ONCE and reused bit-for-bit across A0..A5!)
    # -----------------------------------------------------------------------
    best_r0_obj = -1e9
    if args.init_r0_ckpt is not None and args.init_r0_ckpt.is_file():
        ck_r0 = torch.load(args.init_r0_ckpt, map_location="cpu", weights_only=False)
        model.rung0.load_state_dict(ck_r0["rung0_state_dict"])
        best_r0_obj = float(ck_r0.get("best_r0_obj", 0.0))
        ev_r0_loaded = _eval_val(model, r0_only=True)
        print(f"  [Phase1-R0 LOADED SHARED FROZEN CKPT] {args.init_r0_ckpt} {ev_r0_loaded}", flush=True)
    else:
        opt1 = _make_adamw([{"params": list(model.rung0.parameters()), "lr": 1.8e-3, "base_lr": 1.8e-3, "weight_decay": 0.01}], device)
        best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
        model.train()
        for step in range(p1_steps):
            mult1 = _cosine_mult(step, p1_steps)
            for pg in opt1.param_groups:
                pg["lr"] = pg["base_lr"] * mult1
            bk = _collate_v160_batch(_sample_batch(bs), fcache, model.r0_b2, model.r0_b3, device, rng=rng, punct_jitter_prob=0.35, need_bpe=False)
            opt1.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                cf_in = bk["cf"]
                if float(args.charlm_drop_p) > 0.0:
                    lm_k = (torch.rand((cf_in.shape[0], 1), device=cf_in.device) >= float(args.charlm_drop_p)).to(cf_in.dtype)
                    cf_in = torch.cat([cf_in[:, :R0_STRUCT_DIM_V140], cf_in[:, R0_STRUCT_DIM_V140:] * lm_k], dim=-1)
                r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], cf_in)
                l_ai = compute_progressive_kd_loss_v140(r0["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
                l_pr = compute_progressive_kd_loss_v140(r0["logit_promo_r0"], bk, axis="promo")
                with torch.no_grad():
                    tp_ai = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                    tp_pr = torch.where(bk["has_pr"] > 0.5, bk["y_pr"], torch.sigmoid(bk["t_pr_lg"]))
                    err_ai = ((torch.sigmoid(r0["logit_ai_r0"].float()) - tp_ai).abs() >= 0.20).float()
                    err_pr = ((torch.sigmoid(r0["logit_promo_r0"].float()) - tp_pr).abs() >= 0.20).float()
                l_def = F.binary_cross_entropy_with_logits(r0["defer_ai_r0_logit"].float(), err_ai) + F.binary_cross_entropy_with_logits(r0["defer_pr_r0_logit"].float(), err_pr)
                loss = l_ai + 0.72 * l_pr + 0.15 * l_def
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rung0.parameters(), 1.0)
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

    # Re-seed RNG identically right before Phase 2 so A0..A5 see the EXACT SAME Phase-2/3 sample order regardless of Phase-1!
    rng = random.Random(args.seed + 1000)
    np.random.seed(args.seed + 1000)
    torch.manual_seed(args.seed + 1000)

    if args.freeze_r0:
        for p in model.rung0.parameters():
            p.requires_grad_(False)
        model.rung0.eval()

    # -----------------------------------------------------------------------
    # Exact Unbiased Horvitz-Thompson Anytime Depth Loss + Stop-Grad Tolerance `L_mono` + Optional Arm-A3 NCL
    # -----------------------------------------------------------------------
    def _stratosphere_loss(od: dict[str, Any], bk: dict[str, torch.Tensor]) -> torch.Tensor:
        ai_steps: list[torch.Tensor] = od["logit_ai_steps"]
        pr_steps: list[torch.Tensor] = od["logit_promo_steps"]
        u_steps: list[torch.Tensor] = od["u_halt_steps"]
        T = len(ai_steps)

        with torch.no_grad():
            t_p_sn9 = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_t3"]))
            ai_valid = (bk["has_ai"] > 0.5) | (bk["has_t"] > 0.5)
            # Trusted labels for stop-grad monotonic margin (explicit gold OR confident teacher `|p - 0.5| >= 0.35`)
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
                # Default locked A0-A5: strictly unweighted proper BCE/KD on natural training mix!
                sw = torch.ones_like(err_r0)

        # 1. Exact Equal-Weight Anytime Depth Loss:
        #    When T ~ U{1..6}, step `t = ti + 1` occurs with P(T >= t) = (7 - t) / 6.
        #    Weighting step `ti` by `1 / (6 - ti)` yields EXACT E[sum_{ti=0}^{T-1} w_ti * L_{ti+1}] = (1/6) * sum_{t=1}^6 L_t!
        l_ai_any = torch.zeros((), device=device)
        l_pr_any = torch.zeros((), device=device)
        l_top_any = torch.zeros((), device=device)
        is_u16 = (args.t_train_mode == "uniform_1_6") and (not args.smoke_test)
        for ti in range(T):
            w_depth = (1.0 / float(max(1, 6 - ti))) if is_u16 else (1.0 / float(T))
            l_ai_any = l_ai_any + w_depth * compute_progressive_kd_loss_v140(ai_steps[ti], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=sw)
            l_pr_any = l_pr_any + w_depth * compute_progressive_kd_loss_v140(pr_steps[ti], bk, axis="promo")
            if ti in (0, T - 1):
                l_top_any = l_top_any + compute_top_tail_pairwise_loss_v140(ai_steps[ti], y_bin_ai, ai_valid)
        l_top_any = l_top_any / (2.0 if T >= 2 else 1.0)

        loss = (
            1.65 * l_ai_any
            + 0.70 * l_pr_any
            + 0.24 * l_top_any
            + 0.25 * compute_progressive_kd_loss_v140(od["logit_ai_r1"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=sw)
            + 0.16 * compute_progressive_kd_loss_v140(od["logit_promo_r1"], bk, axis="promo")
        )
        if not args.freeze_r0:
            loss = (
                loss
                + 0.18 * compute_progressive_kd_loss_v140(od["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
                + 0.16 * compute_progressive_kd_loss_v140(od["logit_promo_r0"], bk, axis="promo")
            )

        # 2. Stop-Grad Tolerance Monotonic Margin: `ReLU(sg(m_t) - m_{t+1} - eps)` on trusted labels ONLY!
        #    Pair `(ti, ti+1)` occurs with P(T >= ti+2) = (5 - ti)/6 -> HT weight `6 / (5 * (5 - ti))` gives exact 1/5 per transition!
        if (not args.no_mono_loss) and T >= 2:
            l_mono = torch.zeros((), device=device)
            eps_m = float(args.mono_eps)
            denom_tr = trusted_m.sum().clamp_min(1.0)
            for ti in range(T - 1):
                w_pair = (6.0 / (5.0 * float(max(1, 5 - ti)))) if is_u16 else (1.0 / float(T - 1))
                m_t_sg = y_sign * ai_steps[ti].detach().float().clamp(-6.0, 6.0)
                m_next = y_sign * ai_steps[ti + 1].float().clamp(-6.0, 6.0)
                l_mono = l_mono + w_pair * ((F.relu(m_t_sg - m_next - eps_m) * trusted_m).sum() / denom_tr)
            loss = loss + 0.30 * l_mono

        # 3. Falsification Arm A3 ONLY: Hard-Example NCL against Frozen Stop-Grad R0 (OFF in A0/A1/A2/A4/A5!)
        if args.use_ncl and float(args.ncl_gamma) > 0.0:
            ncl_m = (err_r0 >= 0.12) & ai_valid
            if ncl_m.any():
                r0_res = (p_r0[ncl_m] - t_p_sn9[ncl_m]).detach()
                l_ncl = torch.zeros((), device=device)
                for ti in range(T):
                    w_depth = (1.0 / float(max(1, 6 - ti))) if is_u16 else (1.0 / float(T))
                    b_res = torch.sigmoid(ai_steps[ti].float()[ncl_m]) - t_p_sn9[ncl_m]
                    l_ncl = l_ncl + w_depth * F.relu(b_res * r0_res).mean()
                loss = loss + float(args.ncl_gamma) * l_ncl

        # 4. `TaH2` Lookahead Utility Regression (`u_t -> realized BCE drop minus step cost 0.01`, detached!)
        if T >= 2:
            l_halt = torch.zeros((), device=device)
            with torch.no_grad():
                step_bces = [F.binary_cross_entropy_with_logits(ai_steps[k].float(), t_p_sn9, reduction="none") for k in range(T)]
            for ti in range(T - 1):
                w_pair = (6.0 / (5.0 * float(max(1, 5 - ti)))) if is_u16 else (1.0 / float(T - 1))
                with torch.no_grad():
                    best_gain = step_bces[ti] - step_bces[ti + 1] - 0.01
                    for j in range(ti + 2, T):
                        best_gain = torch.maximum(best_gain, step_bces[ti] - step_bces[j] - 0.01 * (j - ti))
                    tgt_u = best_gain.clamp(-0.5, 1.0)
                l_halt = l_halt + w_pair * F.smooth_l1_loss(u_steps[ti].float(), tgt_u, beta=0.1)
            loss = loss + 0.25 * l_halt

        # 5. Promo 4-way subtype + Era GRL
        s4_m = (bk["has_pr"] > 0.5) & (bk["sub4"] >= 0) & (bk["sub4"] < 4)
        if s4_m.any():
            loss = loss + 0.10 * F.cross_entropy(od["promo_sub4_logit"].float()[s4_m], bk["sub4"][s4_m])

        hu_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
        era_v = hu_m & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
        if int(era_v.sum().item()) >= 8:
            loss = loss + 0.12 * era_grl(od["emb_ai"], bk["era5"], era_v, lam=0.50)
        return loss

    # -----------------------------------------------------------------------
    # Phase 2: Full StratosphereLoop Training (`T ~ U{1..6}` + CF Mirrors + Symmetric DynaSeam)
    # -----------------------------------------------------------------------
    r0_ids = {id(p) for p in model.rung0.parameters()}
    opt2_groups = [{"params": [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters()), "lr": 1.1e-3, "base_lr": 1.1e-3, "weight_decay": 0.01}]
    if not args.freeze_r0:
        opt2_groups.insert(0, {"params": list(model.rung0.parameters()), "lr": 2.0e-4, "base_lr": 2.0e-4, "weight_decay": 0.01})
    opt2 = _make_adamw(opt2_groups, device)
    ema_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_p2_sd, best_p2_obj = {k: v.clone() for k, v in ema_sd.items()}, -1e9
    best_p2_ev: dict[str, Any] = {}
    has_cf = corpus["stats"]["admitted_cf_pairs"] > 0

    model.train()
    if args.freeze_r0:
        model.rung0.eval()
    for step in range(p2_steps):
        mult2 = _cosine_mult(step, p2_steps)
        for pg in opt2.param_groups:
            pg["lr"] = pg["base_lr"] * mult2
        bk = _collate_v160_batch(_sample_batch(bs), fcache, model.r0_b2, model.r0_b3, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        if args.t_train_mode == "uniform_1_6":
            outer_n = rng.randint(1, 6) if not args.smoke_test else rng.choice([2, 4, 6])
        else:
            outer_n = 4 if (step % 2 == 0) else 2

        opt2.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=outer_n, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p),
                dt_drop_p=float(args.dt_drop_p), use_crossview_dt=bool(args.use_slot9_dt),
            )
            loss = _stratosphere_loss(od, bk)

            if has_cf and (step % 3 == 0):
                ch = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
                if ch:
                    ba = _collate_v160_batch([{"text": x["ai_text"]} for x in ch], fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    bh = _collate_v160_batch([{"text": x["human_text"]} for x in ch], fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oa = model(ba["ids"], ba["msk"], ba["st"], ba["eg"], ba["tg"], ba["loc"], ba["c1"], ba["c2"], ba["c3"], ba["cm"], ba["cf"], ba["sup"], outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.35, use_crossview_dt=bool(args.use_slot9_dt))
                    oh = model(bh["ids"], bh["msk"], bh["st"], bh["eg"], bh["tg"], bh["loc"], bh["c1"], bh["c2"], bh["c3"], bh["cm"], bh["cf"], bh["sup"], outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.35, use_crossview_dt=bool(args.use_slot9_dt))
                    pw = torch.tensor([float(x["pair_weight"]) for x in ch], dtype=torch.float32, device=device)
                    l_cf = (
                        (F.softplus(1.8 - (oa["logit_ai_t4"].float() - oh["logit_ai_t4"].float())) * pw).mean()
                        + 0.60 * (F.softplus(1.7 - (oa["logit_ai_t2"].float() - oh["logit_ai_t2"].float())) * pw).mean()
                        + 0.50 * (F.softplus(1.6 - (oa["logit_ai_t1"].float() - oh["logit_ai_t1"].float())) * pw).mean()
                    )
                    loss = loss + 0.25 * l_cf

            if longform_train and (step % 6 == 0):
                docs = rng.sample(longform_train, min(2 if args.smoke_test else 5, len(longform_train)))
                d_losses = []
                for d_it in docs:
                    cl_doc, _ = canonicalize_zero_pii_v150(str(d_it.get("text", "")))
                    chunks = plan_long_chunks_v11(cl_doc, max_chunks=6)
                    if not chunks:
                        continue
                    bk_c = _collate_v160_batch([{"text": c["text"]} for c in chunks], fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oc = model(bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["tg"], bk_c["loc"], bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"], bk_c["sup"], outer_steps=2, r0_drop_p=0.0, use_crossview_dt=bool(args.use_slot9_dt))
                    cm_t = torch.from_numpy(np.stack([c["chunk_meta"] for c in chunks], 0).astype(np.float32)).unsqueeze(0).to(device)
                    vm_t = torch.ones((1, len(chunks)), dtype=torch.float32, device=device)
                    da, dp = model.aggregate_long_chunks(oc["emb_ai"].unsqueeze(0), oc["emb_pr"].unsqueeze(0), oc["logit_ai"].unsqueeze(0), oc["logit_promo"].unsqueeze(0), cm_t, vm_t)
                    y_da = torch.tensor([1.0 if float(d_it.get("ai_score", d_it.get("score", 0.0))) >= 65.0 else 0.0], dtype=torch.float32, device=device)
                    y_dp = torch.tensor([1.0 if float(d_it.get("promo_score", 0.0)) >= 65.0 else 0.0], dtype=torch.float32, device=device)
                    d_losses.append(F.binary_cross_entropy_with_logits(da.float(), y_da) + 0.50 * F.binary_cross_entropy_with_logits(dp.float(), y_dp))
                if gold_pos and gold_nat:
                    h_wrap = canonicalize_zero_pii_v150(str(rng.choice(gold_nat)["text"]))[0]
                    a_core = canonicalize_zero_pii_v150(str(rng.choice(gold_pos)["text"]))[0]
                    h_tail = canonicalize_zero_pii_v150(str(rng.choice(gold_nat)["text"]))[0]
                    for sand_doc, tgt_y in (
                        (f"{h_wrap}\n\n{a_core}\n\n{h_tail}", 1.0),
                        (f"{h_wrap}\n\n楼上这段评论一看就是AI生成的：「{a_core}」大家别被带节奏。\n\n{h_tail}", 0.0),
                    ):
                        s_chunks = plan_long_chunks_v11(sand_doc, max_chunks=4)
                        if len(s_chunks) >= 2:
                            bk_s = _collate_v160_batch([{"text": c["text"]} for c in s_chunks], fcache, model.r0_b2, model.r0_b3, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                            os_c = model(bk_s["ids"], bk_s["msk"], bk_s["st"], bk_s["eg"], bk_s["tg"], bk_s["loc"], bk_s["c1"], bk_s["c2"], bk_s["c3"], bk_s["cm"], bk_s["cf"], bk_s["sup"], outer_steps=2, r0_drop_p=0.0, use_crossview_dt=bool(args.use_slot9_dt))
                            cms = torch.from_numpy(np.stack([c["chunk_meta"] for c in s_chunks], 0).astype(np.float32)).unsqueeze(0).to(device)
                            vms = torch.ones((1, len(s_chunks)), dtype=torch.float32, device=device)
                            da_s, _ = model.aggregate_long_chunks(os_c["emb_ai"].unsqueeze(0), os_c["emb_pr"].unsqueeze(0), os_c["logit_ai"].unsqueeze(0), os_c["logit_promo"].unsqueeze(0), cms, vms)
                            d_losses.append(F.binary_cross_entropy_with_logits(da_s.float(), torch.tensor([tgt_y], dtype=torch.float32, device=device)))
                if d_losses:
                    loss = loss + 0.22 * torch.stack(d_losses).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(era_grl.parameters()), 1.0)
        opt2.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_sd[k].mul_(0.995).add_(v.detach().cpu(), alpha=0.005)
        if (step + 1) % max(1, p2_steps // 6) == 0 or step == p2_steps - 1:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_sd)
            ev2 = _eval_val(model, r0_only=False)
            if ev2["sel_obj"] > best_p2_obj:
                best_p2_obj, best_p2_sd, best_p2_ev = ev2["sel_obj"], {k: v.clone() for k, v in ema_sd.items()}, ev2
            lr_show = opt2.param_groups[-1]["lr"]
            print(f"  [Phase2 step={step+1}/{p2_steps}] lr={lr_show:.2e} loss={float(loss.item()):.4f} {ev2}", flush=True)
            model.load_state_dict(cur_sd)
            model.train()
            if args.freeze_r0:
                model.rung0.eval()

    model.load_state_dict(best_p2_sd)
    torch.save({"version": args.version, "arch": args.arch, "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_p2_sd.items()}, "val": best_p2_ev}, ckpt_dir / f"{args.version}_fp16_master.pt")

    # -----------------------------------------------------------------------
    # 3-Checkpoint Quantization Audit Part A: FP32 Master vs Direct-Packed MGVZ10 (pre-QAT)
    # -----------------------------------------------------------------------
    cal_rows = corpus["val_gold_v2"] + corpus["val_wave3"] + corpus["val_old_video_neg"] + corpus.get("val_multilingual", [])
    cal_txts = [r["text"] for r in cal_rows]
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.float32)
    y_cal_int = (y_cal_ai >= 0.5).astype(np.int32)

    p_fp32_master = predict_student_v160(model, fcache, cal_txts, device, outer_steps=6, use_crossview_dt=args.use_slot9_dt, record_diagnostics=False)
    dp_mgvz_path = out_dir / f"{args.version}_direct_packed_preqat.mgvz"
    pack_bundle_v10_v160(model, dp_mgvz_path)
    m_dp = MeguvizV160StratosphereLoopModel(**{**model_kwargs, "style_era_dropout_p": 0.0}).to(device).eval()
    load_bundle_v10_v160_into_model(m_dp, dp_mgvz_path)
    p_direct_pack = predict_student_v160(m_dp, fcache, cal_txts, device, outer_steps=6, use_crossview_dt=args.use_slot9_dt, record_diagnostics=False)
    del m_dp

    # -----------------------------------------------------------------------
    # Phase 3: Identical LoopQ-Aware Clean W4/W8 QAT (`T ~ U{1..6}`) across ALL 6 Arms
    # -----------------------------------------------------------------------
    best_q_obj = float(best_p2_obj)
    if p3_steps > 0:
        set_v160_qat_mode(model, True)
        qat_params = [p for p in model.parameters() if p.requires_grad]
        opt3 = _make_adamw([{"params": qat_params, "lr": 2.0e-4, "base_lr": 2.0e-4, "weight_decay": 0.005}], device)
        ema_q = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_q_sd, best_q_obj = {k: v.clone() for k, v in ema_q.items()}, _eval_val(model, r0_only=False)["sel_obj"]
        model.train()
        if args.freeze_r0:
            model.rung0.eval()
        for step in range(p3_steps):
            mult3 = _cosine_mult(step, p3_steps, warm_frac=0.06, min_f=0.08)
            for pg in opt3.param_groups:
                pg["lr"] = pg["base_lr"] * mult3
            bk = _collate_v160_batch(_sample_batch(bs), fcache, model.r0_b2, model.r0_b3, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
            outer_q = rng.randint(1, 6) if (args.t_train_mode == "uniform_1_6" and not args.smoke_test) else 4
            opt3.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                od = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                    outer_steps=outer_q, r0_drop_p=0.45, charlm_drop_p=0.25,
                    dt_drop_p=float(args.dt_drop_p), use_crossview_dt=bool(args.use_slot9_dt),
                )
                loss = _stratosphere_loss(od, bk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, 1.0)
            opt3.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        ema_q[k].mul_(0.992).add_(v.detach().cpu(), alpha=0.008)
            if (step + 1) % max(1, p3_steps // 3) == 0 or step == p3_steps - 1:
                cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_q)
                evq = _eval_val(model, r0_only=False)
                print(f"  [Phase3-QAT step={step+1}/{p3_steps}] loss={float(loss.item()):.4f} {evq}", flush=True)
                if evq["sel_obj"] >= best_q_obj:
                    best_q_obj, best_q_sd = evq["sel_obj"], {k: v.clone() for k, v in ema_q.items()}
                model.load_state_dict(cur_sd)
                model.train()
                if args.freeze_r0:
                    model.rung0.eval()
        model.load_state_dict(best_q_sd)
        set_v160_qat_mode(model, False)

    # Evaluate deployed post-QAT MGVZ10 weights (reloaded identically to production!)
    postqat_mgvz_tmp = out_dir / "bundle_v10_stratosphere_v160.mgvz"
    pack_bundle_v10_v160(model, postqat_mgvz_tmp)
    m_dep = MeguvizV160StratosphereLoopModel(**{**model_kwargs, "style_era_dropout_p": 0.0}).to(device).eval()
    load_bundle_v10_v160_into_model(m_dep, postqat_mgvz_tmp)

    p_cal = predict_student_v160(
        m_dep, fcache, cal_txts, device, r0_only=False, outer_steps=6,
        use_crossview_dt=args.use_slot9_dt, record_diagnostics=True,
    )
    joint_jac = measure_joint_directional_jacobian_v160(m_dep, fcache, cal_txts, device, outer_steps=6)
    probe_3seg = run_order_preserving_3seg_probe_v160(m_dep, fcache, corpus["val_gold_v2"] + corpus["val_wave3"], device)

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

    quant_drift_audit = {
        "direct_packed_preqat_vs_fp32": _quant_drift_summary(p_fp32_master, p_direct_pack),
        "post_qat_deployed_vs_fp32": _quant_drift_summary(p_fp32_master, p_cal),
    }

    # -----------------------------------------------------------------------
    # Phase 4: Disjoint `cal_fit / cal_cert` Split + Monotone OOF Fusion + Exact Clopper-Pearson + Export + 23-Gate CI
    # -----------------------------------------------------------------------
    fit_mask = np.array([(int(text_sha16_v140(r["text"]), 16) % 2) == 0 for r in cal_rows], dtype=bool)
    cert_mask = ~fit_mask
    if fit_mask.sum() < 24 or cert_mask.sum() < 24 or y_cal_int[fit_mask].sum() < 6 or y_cal_int[cert_mask].sum() < 6:
        fit_mask = np.ones((len(cal_rows),), dtype=bool)
        cert_mask = fit_mask.copy()

    # Fit strictly monotone OOF late fusion `f = bias + w_a*a + w_b*b` (`w_a >= 0, w_b >= 0`) strictly on `cal_fit`
    if use_cv_fusion:
        cv_fit_raw = fit_crossview_late_fusion(p_cal["logit_ai_r0"][fit_mask], p_cal["logit_ai_t4"][fit_mask], y_cal_ai[fit_mask])
        cv_fusion = {k: v for k, v in cv_fit_raw.items() if not k.startswith("_")}
    else:
        cv_fusion = {"enabled": False, "formula": "monotone_linear", "w_a": 0.0, "w_b": 1.0, "w0": 0.0, "w1": 1.0, "bias": 0.0}

    cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
    t_ai = fit_scalar_temperature_v140(cal_fused_t4[fit_mask], y_cal_ai[fit_mask])

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

    pr_cal_rows = corpus["val_wave3"] + corpus["val_promo_v11"]
    p_pr_cal = predict_student_v160(m_dep, fcache, [r["text"] for r in pr_cal_rows], device, r0_only=False, outer_steps=4, use_crossview_dt=args.use_slot9_dt)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in pr_cal_rows], dtype=np.float32)
    t_pr = fit_scalar_temperature_v140(p_pr_cal["logit_promo"], y_cal_pr) if len(pr_cal_rows) else 1.0

    # Fit LUTs strictly on `cal_fit`
    lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"][fit_mask], -30, 30), y_cal_ai[fit_mask])
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

    # Disjoint Fit -> Cert R0 LTT Selection (fit candidate on `cal_fit`, certify with exact Clopper-Pearson on `cal_cert`!)
    p_r0_all_cal = apply_spline_temp_lut(expit(np.clip(p_cal["logit_ai_r0"], -30, 30)), lut_ai_r0)
    p_r0_fit, p_r0_crt = p_r0_all_cal[fit_mask], p_r0_all_cal[cert_mask]
    def_fit, def_crt = p_cal["p_defer_ai_r0"][fit_mask], p_cal["p_defer_ai_r0"][cert_mask]
    pos_fit, neg_fit = y_cal_int[fit_mask] == 1, y_cal_int[fit_mask] == 0
    pos_crt, neg_crt = y_cal_int[cert_mask] == 1, y_cal_int[cert_mask] == 0

    r0_ai_pol: dict[str, Any] = {
        "enabled": False, "tau_ai_lo": 0.03, "defer_max": 0.35, "hi_exit_enabled": False,
        "tau_ai_hi": 0.992, "defer_max_hi": 0.15, "max_chars": 180,
        "cert_lo_human_exit_rate": 0.0, "cert_lo_ai_miss_cp95_ub": 1.0,
        "requires_frozen_natural_stream_audit": True,
    }
    if pos_fit.sum() >= 30 and pos_crt.sum() >= 30:
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

    manifest = {
        "version": args.version, "max_len": int(args.max_len), "head_len": int(args.head_len),
        "long_char_trigger": int(args.long_char_trigger), "temp_ai": t_ai, "temp_promo": t_pr,
        "lut_ai_r0": lut_ai_r0, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_full": lut_pr_full,
        "rung0_ai_ltt_policy": r0_ai_pol, "rung0_pr_ltt_policy": r0_pr_pol,
        "crossview_late_fusion": cv_fusion,
        "anytime_step_policy": {
            "enabled": True, "max_infer_steps": 4, "u_halt_thresh": 0.015,
            "disagree_dt_thresh": 1.60, "uncertain_lo": 0.12, "uncertain_hi": 0.88, "t1_exit_enabled": True,
        },
    }

    export_rep = export_v160_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    ci_gates = run_v160_ci_gates(
        m_dep, tokenizer, export_rep=export_rep, ai_lut_check=ai_lut_chk, promo_lut_check=pr_lut_chk,
        charlm=full_charlm, release_dir=out_dir,
    )

    # -----------------------------------------------------------------------
    # Loop-Physics Scorecard (All 6 GPT Probes + T3-State -> T4-Readout Pure-Bias Control!)
    # -----------------------------------------------------------------------
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

    p_cal_nor0 = predict_student_v160(m_dep, fcache, cal_txts, device, r0_only=False, outer_steps=4, r0_drop_eval=True)
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

    # Save Offline Prediction Cache (`.npz`) for A5 Offline Calibration / Fusion / Bootstrap Studies!
    cache_splits = ("val_gold_v2", "test_gold_v2", "val_wave3", "val_old_video_neg", "val_multilingual", "diag_god500")
    cache_arrays: dict[str, np.ndarray] = {}
    for sk in cache_splits:
        srows = corpus.get(sk, [])
        if not srows:
            continue
        ps = predict_student_v160(m_dep, fcache, [r["text"] for r in srows], device, outer_steps=6, use_crossview_dt=args.use_slot9_dt, record_diagnostics=True)
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

    # Final ONNX Test Readouts & Real ORT 4T B=1 Latency
    eng = MeguvizV160CascadeEngine(out_dir, threads=4)

    def _eval_onnx_set(rows: list[dict[str, Any]], force_depth: int | None = None) -> dict[str, Any]:
        if not rows:
            return {}
        out_r = eng.predict_batch([r["text"] for r in rows], max_batch=32, force_full=(force_depth is not None), force_depth=force_depth)
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

    report = {
        "version": args.version, "arch": args.arch, "dynamics": args.dynamics,
        "use_step_phi": use_step_phi, "step_decay_mode": args.step_decay_mode,
        "use_ncl": bool(args.use_ncl), "use_slot9_dt": bool(args.use_slot9_dt),
        "expand_tables": bool(args.expand_tables), "freeze_r0": bool(args.freeze_r0),
        "t_train_mode": args.t_train_mode, "gpu_name": gpu_name,
        "total_params": tot_p, "core_params": core_p,
        "phase1_r0_best_obj": round(float(best_r0_obj), 5),
        "phase2_fp16_val": best_p2_ev, "phase3_qat_best_obj": round(float(best_q_obj), 5),
        "loop_physics": loop_physics,
        "quant_drift_3ckpt_audit": quant_drift_audit,
        "fusion_cal_cert_comparison": fusion_cert_comp,
        "order_preserving_3seg_probe": probe_3seg,
        "ort_4t_b1_latency_ms": ort_lat_b1,
        "test_bili_gold_v2_onnx_cond": _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=None),
        "test_bili_gold_v2_onnx_full_t4": _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=4),
        "val_multilingual_onnx": _eval_onnx_set(corpus.get("val_multilingual", []), force_depth=None),
        "diag_god500_onnx": _eval_onnx_set(corpus.get("diag_god500", []), force_depth=None),
        "rung0_ai_ltt_policy": r0_ai_pol,
        "crossview_late_fusion": cv_fusion,
        "export": export_rep,
        "v160_ci_gates": ci_gates,
        "release_gates": {
            "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v160_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["release_gates"]["v160_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
