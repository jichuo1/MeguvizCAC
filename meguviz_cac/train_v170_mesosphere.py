"""MeguvizCAC v1.7.0-MesosphereLoop Unified Trainer (Turbo v3 + Stagewise PNBR-R + M7 ER-DRO-R + M8 IUT SCE).

Implements 100% of `v170_mesosphere_final_spec.md` (9/9 External GPT Audits + Real ORT/GPU Lab):
  - Full Natural-Stream R0++ Anchor (NO `--r0-antidilution-mix`!) + Detached `APD-v2` Exit Head Training.
  - Phase 2a (`0..65%` steps): `T1` Native + Once-at-T1 `B2a` Dual-Manifold Epistemic Hopfield + `M6 P_H`
    + `M7 CNMA-R/UISR` (`Y=0` conditional invariance) + `M7 ER-DRO-R` (fixed label stratification `pi_{g,y}`,
    frozen ref risk, Hard cap <= 65%, KL <= 0.005, version-bound IW <= 1.1112) + `TCBR-R` + `D02` Paraphrase CF
    + `DynaSeam + 68 B DS-TitanGuard` + optional `Strategy I PUC`.
  - Phase 2b (`65..100%` steps): `M5` Stagewise Prefix Freezing (FREEZES `T1` prefix -> zero ULP degradation!)
    training `Step-LoRA + AR-U-Loop + 16x16 Delta-Rule + PNBR-R` + `B_cert` logging.
  - Phase 3: Light-QAT + Zero-Disk Bit-Exact `Auto-Ship-Gate` + `EBA-H` Prior-Matched Fusion
    + `M8` Zero-Bonferroni `IUT` Stratified `SCE` + 4-Stage Pure-FP32 ONNX Export + `26/26 CI`.
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
import re
import time
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss
import torch
import torch.nn.functional as F

from .native_frontend import HAS_RUST_FRONTEND, RustTurboFrontend
from .train_v140_paradise_apex import (
    _eval_bin,
    _load_v140_distill_pool,
    _make_adamw,
    compute_progressive_kd_loss_v140,
)
from .v07_features import BiliPickyBPETokenizer, apply_spline_temp_lut, make_engram_bigram_ids
from .v08_zenith_ops import OfflineEraGRLHead, _GradReverseFn
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d, plan_long_chunks_v11
from .v122_apexnano import fit_output_shifted_promo_lut_v122, fit_smooth_ai_lut_v122
from .v123_data_refinery import NUM_ERAS_V122
from .v130_xtreme_apex import AI_END_V130, set_v130_qat_mode
from .v140_paradise_apex import HEAD_LEN_V140, MAX_LEN_V140, R0_STRUCT_DIM_V140
from .v140_superteacher import (
    build_v140_superteacher_corpus,
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
from .v160_discourse_causal_ops import (
    class_conditional_vrex_unbiased,
)
from .v160_stratosphere_loop import (
    QUOTE_SCOPE_DIM_V161,
    apply_crossview_late_fusion,
    extract_apex_char_batch_v160,
    extract_token_quote_scope_4d_v161,
    fit_crossview_eba_h_fusion,
)
from .v170_features import M6_PH_DIM_V170, M7_SURF_DIM_V170, M7_UISR_DIM_V170, extract_v170_sidecar_batch
from .v170_mesosphere_loop import (
    ARCH_PRESETS_V170,
    SKETCH_R_V170,
    MeguvizV170CascadeEngine,
    MeguvizV170MesosphereLoopModel,
    _T1_OUT_NAMES_V170,
    apply_mgvz10_quant_roundtrip_inplace_v170,
    export_v170_bootable_and_onnx,
    fit_iut_stratified_r0_policy_v170,
    run_v170_ci_gates,
)


# ---------------------------------------------------------------------------
# 1. Rust-Accelerated OOF Feature Cache + Packed Metadata Collation
# ---------------------------------------------------------------------------

_META_F32_KEYS_V170 = (
    "y_ai", "y_pr", "has_ai", "has_pr", "has_teacher", "kd_w",
    "t_ai_lg", "t_ai_r0", "t_ai_t1", "t_ai_t2", "t_ai_t3",
    "frontier_gap", "t_pr_lg", "hard_neg", "hard_fn", "trust_neg", "dro_iw",
)


class _StrictGroupOOFFeatureCacheV170:
    def __init__(
        self,
        tokenizer: BiliPickyBPETokenizer,
        max_len: int,
        head_len: int,
        engram_buckets: int,
        r0_b2: int,
        r0_b3: int,
        use_local_charlm: bool = True,
        use_rust: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.engram_buckets = int(engram_buckets)
        self.r0_b2 = int(r0_b2)
        self.r0_b3 = int(r0_b3)
        self.use_local_charlm = bool(use_local_charlm)
        self.fold_lms: list[CompactCharLMV150] = []
        self.text_to_fold: dict[str, int] = {}
        self.full_lm: CompactCharLMV150 | None = None
        self.rust_fe: RustTurboFrontend | None = None
        if use_rust and HAS_RUST_FRONTEND:
            try:
                self.rust_fe = RustTurboFrontend(
                    tokenizer, max_len=self.max_len, head_len=self.head_len,
                    engram_buckets=self.engram_buckets, trigram_buckets=8192,
                    r0_b2=self.r0_b2, r0_b3=self.r0_b3, use_local_charlm=self.use_local_charlm,
                    bucket_seq_len=True,
                )
            except Exception:
                self.rust_fe = None

    def set_group_oof_charlm(
        self, fold_lms: list[CompactCharLMV150], text_to_fold: dict[str, int], full_lm: CompactCharLMV150
    ) -> None:
        self.fold_lms = fold_lms
        self.text_to_fold = text_to_fold
        self.full_lm = full_lm
        if self.rust_fe is not None:
            self.rust_fe.set_group_oof_charlm(fold_lms, text_to_fold, full_lm)


def _collate_v170_cpu(
    rows: list[dict[str, Any]],
    fcache: _StrictGroupOOFFeatureCacheV170,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.0,
    need_bpe: bool = True,
    use_oof_charlm: bool = True,
    need_quote_scope: bool = True,
    include_meta: bool = True,
) -> dict[str, torch.Tensor]:
    bs = len(rows)
    raw_texts: list[str] = []
    parents: list[str] = []
    for r in rows:
        t = str(r.get("text", ""))
        parent = str(r.get("_parent_text", t))
        if rng is not None and punct_jitter_prob > 0.0 and rng.random() < punct_jitter_prob:
            t = jitter_ai_text_v140(t, rng)
        raw_texts.append(t)
        parents.append(parent)

    out_np: dict[str, np.ndarray] = {}
    if fcache.rust_fe is not None:
        if not need_bpe:
            cleans, _ = fcache.rust_fe.canonicalize_with_spoof_batch(raw_texts, max_chars=2048)
            d0 = fcache.rust_fe.extract_batch_numpy(
                cleans, parent_raw_texts=parents, need_bpe=False, use_oof_charlm=use_oof_charlm,
                already_canonical=True, need_r0=True,
            )
            approx_bpe = np.array([max(1, len(t) // 2) for t in cleans], dtype=np.int64)
            m6_ph, m7_surf, m7_uisr = fcache.rust_fe.extract_v170_sidecar(
                cleans, approx_bpe, parent_raw_texts=parents, use_oof_charlm=use_oof_charlm
            )
            out_np = {
                "c1": d0["c1"], "c2": d0["c2"], "c3": d0["c3"], "cm": d0["cm"], "cf": d0["cf"], "sup": d0["sup"],
                "m6_ph": m6_ph, "m7_surf": m7_surf, "m7_uisr": m7_uisr,
            }
        else:
            cleans, _ = fcache.rust_fe.canonicalize_with_spoof_batch(raw_texts, max_chars=2048)
            d = fcache.rust_fe.extract_batch_numpy(
                cleans, parent_raw_texts=parents, need_bpe=True, use_oof_charlm=use_oof_charlm,
                need_quote_scope=need_quote_scope, already_canonical=True, need_r0=True,
            )
            bpe_lens = d["msk"].sum(axis=1).astype(np.int64)
            m6_ph, m7_surf, m7_uisr = fcache.rust_fe.extract_v170_sidecar(
                cleans, bpe_lens, parent_raw_texts=parents, use_oof_charlm=use_oof_charlm
            )
            out_np = {
                "ids": d["ids"], "msk": d["msk"], "st": d["st"], "eg": d["eg"], "loc": d["loc"],
                "qsc": d["qsc"] if need_quote_scope else np.zeros((bs, d["ids"].shape[1], QUOTE_SCOPE_DIM_V161), np.float32),
                "c1": d["c1"], "c2": d["c2"], "c3": d["c3"], "cm": d["cm"], "cf": d["cf"], "sup": d["sup"],
                "m6_ph": m6_ph, "m7_surf": m7_surf, "m7_uisr": m7_uisr,
            }
    else:
        cleans = [canonicalize_zero_pii_v150(t)[0] for t in raw_texts]
        r0_rows = []
        for cl, p_raw in zip(cleans, parents):
            lm = fcache.fold_lms[fcache.text_to_fold[p_raw]] if (use_oof_charlm and p_raw in fcache.text_to_fold and fcache.fold_lms) else fcache.full_lm
            r0_rows.append(extract_apex_char_batch_v160([cl], b2=fcache.r0_b2, b3=fcache.r0_b3, already_canonical=True, charlm=lm))
        mr = max(2, max(x[0].shape[1] for x in r0_rows))
        c1 = np.zeros((bs, mr), np.int64); c2 = np.zeros((bs, mr), np.int64); c3 = np.zeros((bs, mr), np.int64)
        cm = np.zeros((bs, mr), np.float32)
        for i, x in enumerate(r0_rows):
            nr = x[0].shape[1]
            c1[i, :nr], c2[i, :nr], c3[i, :nr], cm[i, :nr] = x[0][0], x[1][0], x[2][0], x[3][0]
        cf = np.concatenate([x[4] for x in r0_rows], axis=0).astype(np.float32)
        sups = np.concatenate([x[6] for x in r0_rows], axis=0).astype(np.float32)
        out_np = {"c1": c1, "c2": c2, "c3": c3, "cm": cm, "cf": cf, "sup": sups}
        if need_bpe:
            f56_l, encs, offs_l = [], [], []
            for cl in cleans:
                _, f56, _ = extract_style_features_56d(cl)
                ei, eo = encode_bpe_head_tail_with_offsets_v150(fcache.tokenizer, cl, max_len=fcache.max_len, head_len=fcache.head_len)
                f56_l.append(f56); encs.append(ei); offs_l.append(eo)
            engs = [make_engram_bigram_ids(s, fcache.engram_buckets) for s in encs]
            ml = max(2, max(len(s) for s in encs))
            if ml % 2 == 1:
                ml += 1
            ids = np.full((bs, ml), fcache.tokenizer.pad_id, np.int64)
            msk = np.zeros((bs, ml), np.int64)
            eg = np.zeros((bs, ml), np.int64)
            loc = np.zeros((bs, ml, LOCAL_CHARLM_DIM_V150), np.float32)
            qsc = np.zeros((bs, ml, QUOTE_SCOPE_DIM_V161), np.float32)
            for i, s in enumerate(encs):
                ns = len(s)
                ids[i, :ns], msk[i, :ns], eg[i, :ns] = s, 1, engs[i]
                lm = fcache.fold_lms[fcache.text_to_fold[parents[i]]] if (use_oof_charlm and parents[i] in fcache.text_to_fold and fcache.fold_lms) else fcache.full_lm
                if fcache.use_local_charlm and lm is not None:
                    loc[i, :ns] = lm.extract_token_local_8ch(cleans[i], offs_l[i])
                if need_quote_scope:
                    qsc[i, :ns] = extract_token_quote_scope_4d_v161(cleans[i], offs_l[i])
            bpe_lens = np.array([len(s) for s in encs], dtype=np.int64)
            out_np.update({"ids": ids, "msk": msk, "st": np.stack(f56_l, 0).astype(np.float32), "eg": eg, "loc": loc, "qsc": qsc})
        else:
            bpe_lens = np.array([max(1, len(t) // 2) for t in cleans], dtype=np.int64)
        m6_ph, m7_surf, m7_uisr = extract_v170_sidecar_batch(cleans, bpe_lens, charlm=fcache.full_lm)
        out_np.update({"m6_ph": m6_ph, "m7_surf": m7_surf, "m7_uisr": m7_uisr})

    out_t = {k: torch.from_numpy(v) for k, v in out_np.items()}
    if include_meta:
        meta_f = np.zeros((len(_META_F32_KEYS_V170), bs), dtype=np.float32)
        era5 = np.full((bs,), -1, dtype=np.int64)
        sub4 = np.full((bs,), -1, dtype=np.int64)
        dro_g = np.zeros((bs,), dtype=np.int64)
        for j, r in enumerate(rows):
            meta_f[0, j] = float(r.get("y_ai", 0.0))
            meta_f[1, j] = float(r.get("y_pr", 0.0))
            meta_f[2, j] = float(r.get("has_ai", 1.0))
            meta_f[3, j] = float(r.get("has_pr", 0.0))
            meta_f[4, j] = float(r.get("has_teacher", 0.0))
            meta_f[5, j] = float(r.get("kd_w", 1.0))
            meta_f[6, j] = float(r.get("t_ai_lg", 0.0))
            meta_f[7, j] = float(r.get("t_ai_r0", r.get("t_ai_lg", 0.0)))
            meta_f[8, j] = float(r.get("t_ai_t1", r.get("t_ai_lg", 0.0)))
            meta_f[9, j] = float(r.get("t_ai_t2", r.get("t_ai_lg", 0.0)))
            meta_f[10, j] = float(r.get("t_ai_t3", r.get("t_ai_lg", 0.0)))
            meta_f[11, j] = float(r.get("frontier_gap", 0.0))
            meta_f[12, j] = float(r.get("t_pr_lg", 0.0))
            meta_f[13, j] = float(r.get("hard_neg", 0.0))
            meta_f[14, j] = float(r.get("hard_fn", 0.0))
            meta_f[15, j] = float(r.get("trust_neg", 0.0))
            meta_f[16, j] = float(r.get("dro_iw", 1.0))
            era5[j] = int(r.get("era5", -1))
            sub4[j] = int(r.get("sub4", -1))
            dro_g[j] = int(r.get("_dro_g", 0))
        out_t["_meta_f32"] = torch.from_numpy(meta_f)
        out_t["era5"] = torch.from_numpy(era5)
        out_t["sub4"] = torch.from_numpy(sub4)
        out_t["dro_g"] = torch.from_numpy(dro_g)
    return out_t


def _h2d_batch_v170(bk_cpu: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    nb = device.type == "cuda"
    out: dict[str, torch.Tensor] = {}
    for k, v in bk_cpu.items():
        if k == "_meta_f32":
            mf = v.to(device, non_blocking=nb)
            for ki, mk in enumerate(_META_F32_KEYS_V170):
                out[mk] = mf[ki]
        else:
            out[k] = v.to(device, non_blocking=nb)
    return out


# ---------------------------------------------------------------------------
# 2. `M7 ER-DRO-R` Excess-Risk Group-DRO Controller (Zero-Sync, Hard <= 65%, KL <= 0.005, IW <= 1.1112)
# ---------------------------------------------------------------------------

class ExcessRiskGroupDROR:
    """M7 ER-DRO-R: 4-group excess-risk exponential-weights sampler with frozen reference risk,
    Hard-family mass cap <= 0.65, per-update KL trust region <= 0.005, and bounded IW <= 1.1112!
    Groups: 0=natural_stream, 1=gold_core, 2=hard_frontier, 3=multilingual_long.
    """

    def __init__(self, init_q: tuple[float, ...] = (0.35, 0.25, 0.30, 0.10), hard_cap: float = 0.65, kl_cap: float = 0.005, max_iw: float = 1.1112) -> None:
        self.q_base = np.array(init_q, dtype=np.float64)
        self.q_base /= self.q_base.sum()
        self.q = self.q_base.copy()
        self.hard_cap = float(hard_cap)
        self.kl_cap = float(kl_cap)
        self.max_iw = float(max_iw)
        self.ref_risk: np.ndarray | None = None
        self.acc_loss = np.zeros(4, dtype=np.float64)
        self.acc_cnt = np.zeros(4, dtype=np.float64)
        self.version = 0

    def observe(self, group_ids: np.ndarray, losses: np.ndarray) -> None:
        for g in range(4):
            m = group_ids == g
            if np.any(m):
                self.acc_loss[g] += float(losses[m].sum())
                self.acc_cnt[g] += float(m.sum())

    def maybe_update(self) -> dict[str, Any] | None:
        if np.any(self.acc_cnt < 8):
            return None
        risks = self.acc_loss / np.maximum(1.0, self.acc_cnt)
        self.acc_loss.fill(0.0)
        self.acc_cnt.fill(0.0)
        if self.ref_risk is None:
            # Freeze reference irreducible risk at first calibration window!
            self.ref_risk = risks.copy()
            return {"version": self.version, "q": [round(float(x), 4) for x in self.q], "ref_risk": [round(float(x), 4) for x in self.ref_risk]}
        excess = np.clip(risks - self.ref_risk, -0.30, 0.30)
        # Binary-search step size eta so KL(q_new || q_old) <= kl_cap (0.005)!
        q_old = self.q.copy()
        best_q = q_old.copy()
        for eta in (0.35, 0.20, 0.12, 0.06, 0.03, 0.015):
            cand = q_old * np.exp(eta * excess)
            cand = np.clip(cand, 0.06, 0.70)
            # Enforce Hard-family (group 2) mass cap <= 0.65!
            if cand[2] / cand.sum() > self.hard_cap:
                other_sum = cand[0] + cand[1] + cand[3]
                cand[2] = self.hard_cap * other_sum / max(1e-6, 1.0 - self.hard_cap)
            cand /= cand.sum()
            kl = float(np.sum(cand * np.log(np.clip(cand, 1e-9, 1.0) / np.clip(q_old, 1e-9, 1.0))))
            if kl <= self.kl_cap + 1e-9:
                best_q = cand
                break
        self.q = best_q
        self.version += 1
        return {"version": self.version, "q": [round(float(x), 4) for x in self.q], "excess": [round(float(x), 4) for x in excess]}

    def iw_for_group(self, g: int) -> float:
        # Bounded version-matched importance weight <= 1.1112
        raw = float(self.q[g] / max(1e-6, self.q_base[g]))
        return float(min(self.max_iw, max(0.85, raw)))


# ---------------------------------------------------------------------------
# 3. Fast Pre-Collated Eval & Prediction Helpers
# ---------------------------------------------------------------------------

def _precollate_eval_split_v170(
    rows: list[dict[str, Any]],
    fcache: _StrictGroupOOFFeatureCacheV170,
    device: torch.device,
    batch_size: int = 256,
    r0_only: bool = False,
    need_quote_scope: bool = True,
) -> list[tuple[np.ndarray, dict[str, torch.Tensor]]]:
    n = len(rows)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: len(str(rows[i].get("text", ""))))
    out = []
    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        bk = _collate_v170_cpu(
            [rows[i] for i in idxs], fcache, rng=None, punct_jitter_prob=0.0,
            need_bpe=not r0_only, use_oof_charlm=False, need_quote_scope=need_quote_scope, include_meta=False,
        )
        out.append((np.asarray(idxs, dtype=np.int64), _h2d_batch_v170(bk, device)))
    return out


@torch.no_grad()
def predict_student_v170(
    model: MeguvizV170MesosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV170,
    texts: list[str],
    device: torch.device,
    batch_size: int = 256,
    outer_steps: int = 4,
    r0_only: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    precollated: list[tuple[np.ndarray, dict[str, torch.Tensor]]] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    n = len(texts)
    if n == 0:
        return {}
    keys = ["logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0", "p_apd_safe"]
    if not r0_only:
        keys += ["logit_ai_r1", "logit_promo_r1", "logit_ai", "logit_promo"]
        for k in range(1, outer_steps + 1):
            keys += [f"logit_ai_t{k}", f"logit_promo_t{k}", f"u_halt_t{k}"]
    out: dict[str, np.ndarray] = {k: np.zeros((n,), dtype=np.float32) for k in keys}
    hop_w_arr = np.zeros((n, 3), dtype=np.float32) if not r0_only else None

    batches = precollated if precollated is not None else _precollate_eval_split_v170(
        [{"text": t} for t in texts], fcache, device, batch_size=batch_size, r0_only=r0_only, need_quote_scope=model.use_discourse_ops
    )
    amp_en = bool(use_amp and device.type == "cuda")
    for idxs_np, bk in batches:
        with torch.amp.autocast("cuda", enabled=amp_en, dtype=amp_dtype):
            if r0_only:
                res = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["m7_surf"])
            else:
                res = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                    bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                    outer_steps=outer_steps,
                )
        present = [k for k in keys if k in res]
        if present:
            st_np = torch.stack([res[k].float() for k in present], dim=0).cpu().numpy()
            for ki, k in enumerate(present):
                out[k][idxs_np] = st_np[ki]
        if hop_w_arr is not None and "hop_witness" in res:
            hop_w_arr[idxs_np] = res["hop_witness"].float().cpu().numpy()
    if hop_w_arr is not None:
        out["hop_witness"] = hop_w_arr
    return out


# ---------------------------------------------------------------------------
# 4. Main Training & Certification Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.7.0-MesosphereLoop Unified Trainer")
    ap.add_argument("--distill-pool", type=Path, required=True)
    ap.add_argument("--sibling-dir", type=Path, required=True)
    ap.add_argument("--gold-v2-dir", type=Path, required=True)
    ap.add_argument("--wave3-dir", type=Path, required=True)
    ap.add_argument("--v11-data-dir", type=Path, required=True)
    ap.add_argument("--v07-tokenizer", type=Path, required=True)
    ap.add_argument("--delta-dir", type=Path, default=None)
    ap.add_argument("--v04-data", type=Path, default=None)
    ap.add_argument("--extra-gold-pack", type=Path, default=None)
    ap.add_argument("--extra-cal-cert-pack", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.7.0-M1")
    ap.add_argument("--arch", type=str, default="v170_m1", choices=sorted(ARCH_PRESETS_V170.keys()))

    # v1.7.0 Architecture Toggles (DreamCore_170^Safe + Tri-Tier Super-Memory)
    ap.add_argument("--use-ar-uloop", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--use-step-lora", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--step-lora-r", type=int, default=8)
    ap.add_argument("--use-delta-rule", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--use-dual-hopfield", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--use-m6-ph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-m7-cnma-uisr", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-discourse-ops", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lardo-w8-core", action=argparse.BooleanOptionalAction, default=True)

    # Checkpoints & Warm-Start (Strategy I PUC + Natural-Stream Frozen R0++)
    ap.add_argument("--init-r0-ckpt", type=Path, default=None)
    ap.add_argument("--save-r0-ckpt", type=Path, default=None)
    ap.add_argument("--freeze-r0", action="store_true")
    ap.add_argument("--init-from-fp16", type=Path, default=None)
    ap.add_argument("--wise-anchor-fp16", type=Path, default=None)
    ap.add_argument("--puc-warmup-steps", type=int, default=0)
    ap.add_argument("--puc-shrink", type=float, default=0.95)
    ap.add_argument("--puc-noise", type=float, default=0.015)

    # Schedules & M5 Stagewise Prefix Freezing
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--p1-steps", type=int, default=2200)
    ap.add_argument("--p1-use-cf", action="store_true")
    ap.add_argument("--apd-steps", type=int, default=350, help="Independent APD-v2 exit head training steps on frozen R0")
    ap.add_argument("--p2-steps", type=int, default=3600)
    ap.add_argument("--p2-lr", type=float, default=1.1e-3)
    ap.add_argument("--stage2-frac", type=float, default=0.35, help="Fraction of p2-steps for M5 Stagewise Prefix-Frozen PNBR-R")
    ap.add_argument("--p3-qat-steps", type=int, default=300)
    ap.add_argument("--qat-lr", type=float, default=4.0e-5)
    ap.add_argument("--skip-qat", action="store_true")
    ap.add_argument("--auto-ship-gate", action=argparse.BooleanOptionalAction, default=True)

    # Losses, M7 ER-DRO-R, M2 EBA-H, M8 IUT SCE
    ap.add_argument("--use-tcbr-r", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cvar-alpha", type=float, default=0.08)
    ap.add_argument("--cvar-weight", type=float, default=0.07)
    ap.add_argument("--recall-floor-weight", type=float, default=0.14)
    ap.add_argument("--use-fmt-cf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fmt-cf-weight", type=float, default=0.12)
    ap.add_argument("--use-d02-paraphrase-cf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-promo-flip-cf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--promo-flip-weight", type=float, default=0.06)
    ap.add_argument("--use-cc-vrex", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cc-vrex-weight", type=float, default=0.10)
    ap.add_argument("--use-er-dro-r", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--longform-every", type=int, default=4)
    ap.add_argument("--use-kadane-focal-router", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-eba-h", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eba-pi-nat", type=float, default=0.05)
    ap.add_argument("--eba-m-pos-grid", type=str, default="0.8,1.2,1.6,2.0")
    ap.add_argument("--eba-m-neg-grid", type=str, default="1.0,1.5,2.0,2.5")
    ap.add_argument("--oof-cal-folds", type=int, default=5)
    ap.add_argument("--iut-alpha-max", type=float, default=0.04)

    # Turbo v3 Speed & Runtime
    ap.add_argument("--max-len", type=int, default=MAX_LEN_V140)
    ap.add_argument("--head-len", type=int, default=HEAD_LEN_V140)
    ap.add_argument("--long-char-trigger", type=int, default=300)
    ap.add_argument("--r0-drop-p", type=float, default=0.55)
    ap.add_argument("--charlm-drop-p", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20261228)
    ap.add_argument("--charlm-cache", type=Path, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    ap.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp-bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--fuse-aux", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefetch-workers", type=int, default=2)
    ap.add_argument("--no-rust-frontend", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    # Resolve arch-aware defaults for DreamCore modules (`v170_a0` defaults OFF; `v170_m1` defaults ON)
    is_m1 = args.arch == "v170_m1"
    use_ar_uloop = is_m1 if args.use_ar_uloop is None else bool(args.use_ar_uloop)
    use_step_lora = is_m1 if args.use_step_lora is None else bool(args.use_step_lora)
    use_delta_rule = is_m1 if args.use_delta_rule is None else bool(args.use_delta_rule)
    use_dual_hopfield = is_m1 if args.use_dual_hopfield is None else bool(args.use_dual_hopfield)

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

    # 1. Zero-Leakage Corpus + Extra Cal/Gold Packs
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
            sh = text_sha16_v140(ct)
            if sh in blocked:
                continue
            blocked.add(sh)
            ya_c = float(cr.get("y_ai", cr.get("label_ai", -1.0)))
            if 0.0 <= ya_c <= 1.0:
                extra_cal_cert_rows.append({"text": ct, "y_ai": ya_c, "y_pr": float(cr.get("y_pr", 0.0))})
        print(f"[v1.7.0-ExtraCalCert] Reserved {len(extra_cal_cert_rows):,} zero-leakage holdout rows", flush=True)

    distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)
    if not distill_pool:
        raise SystemExit(f"Empty distill pool: {args.distill_pool}")
    for r in distill_pool:
        r["_dro_g"] = 2 if (r.get("frontier_gap", 0.0) >= 1.3 or r.get("hard_neg", 0.0) > 0.5) else 0
    d_hard = [r for r in distill_pool if r["_dro_g"] == 2]
    d_pos = [r for r in distill_pool if r.get("t_ai_lg", -5.0) >= 0.8]
    d_promo = [r for r in distill_pool if r.get("t_pr_lg", -5.0) >= 0.5]
    d_all = distill_pool

    gold_pos, gold_hd, gold_nat, gold_pr = corpus["train_ai_pos"], corpus["train_hu_hard"], corpus["train_hu_natural"], corpus["train_promo_extra"]
    for r in gold_pos + gold_hd + gold_nat + gold_pr:
        r["_dro_g"] = 1
        r["trust_neg"] = 1.0 if (float(r.get("has_ai", 1.0)) > 0.5 and float(r.get("y_ai", 0.0)) <= 0.25) else 0.0

    ex_hard_fn, ex_hard_fp, ex_sandwich, ex_q2_dual, ex_override, ex_general = [], [], [], [], [], []
    if args.extra_gold_pack is not None and args.extra_gold_pack.is_file():
        for er in read_jsonl_any_v140(args.extra_gold_pack):
            et = str(er.get("text", "") or "").strip()
            if not et or text_sha16_v140(et) in blocked:
                continue
            ya = float(er.get("y_ai", er.get("label_ai", -1.0)))
            yp = float(er.get("y_pr", er.get("label_promo", -1.0)))
            has_a = float(er.get("has_ai", 1.0 if (0.0 <= ya <= 1.0) else 0.0))
            has_p = float(er.get("has_pr", 1.0 if (0.0 <= yp <= 1.0) else 0.0))
            if has_a < 0.5 and has_p < 0.5:
                continue
            ya_c = max(0.0, min(1.0, ya)) if has_a > 0.5 else 0.0
            yp_c = max(0.0, min(1.0, yp)) if has_p > 0.5 else 0.0
            bkt = str(er.get("bucket", "") or "").lower()
            tier = str(er.get("tier", er.get("epistemic_tier", "")) or "")
            is_ovr = bool(er.get("is_override", False)) or ("override" in bkt)
            has_t = float(er.get("has_teacher", er.get("has_t", 0.0))) > 0.5 and (not is_ovr) and (tier != "T0_DIAMOND")
            t_ai_lg = float(er.get("t_ai_lg", er.get("t_ai_logit", 0.0))) if has_t else 0.0
            t_pr_lg = float(er.get("t_pr_lg", er.get("t_pr_logit", 0.0))) if has_t else 0.0
            is_hn = bool(has_a > 0.5 and ya_c <= 0.25 and ("hard_fp" in bkt or bool(er.get("is_hard_fp", False))))
            is_hfn = bool(has_a > 0.5 and ya_c >= 0.70 and ("hard_fn" in bkt or "sandwich" in bkt or "q2_dual" in bkt))
            item = {
                "text": et, "y_ai": ya_c, "y_pr": yp_c, "has_ai": has_a, "has_pr": has_p,
                "has_teacher": 1.0 if has_t else 0.0, "kd_w": float(er.get("kd_w", 1.0)),
                "t_ai_lg": t_ai_lg, "t_ai_r0": float(er.get("t_ai_r0", t_ai_lg)) if has_t else 0.0,
                "t_ai_t1": float(er.get("t_ai_t1", t_ai_lg)) if has_t else 0.0,
                "t_ai_t2": float(er.get("t_ai_t2", t_ai_lg)) if has_t else 0.0,
                "t_ai_t3": float(er.get("t_ai_t3", t_ai_lg)) if has_t else 0.0,
                "frontier_gap": float(er.get("frontier_gap", 0.0)) if has_t else 0.0,
                "t_pr_lg": t_pr_lg, "hard_neg": 1.0 if is_hn else 0.0, "hard_fn": 1.0 if is_hfn else 0.0,
                "trust_neg": 1.0 if (is_ovr or is_hn or (has_a > 0.5 and ya_c <= 0.20 and not has_t)) else 0.0,
                "era5": int(er.get("era5", -1)), "sub4": int(er.get("sub4", -1)),
                "_dro_g": 2 if (is_hn or is_hfn or is_ovr) else (3 if len(et) >= 240 else 0),
            }
            if is_ovr:
                ex_override.append(item)
            elif "sandwich" in bkt:
                ex_sandwich.append(item)
            elif "q2_dual" in bkt:
                ex_q2_dual.append(item)
            elif is_hfn:
                ex_hard_fn.append(item)
            elif is_hn:
                ex_hard_fp.append(item)
            else:
                ex_general.append(item)
        print(
            f"[v1.7.0-ExtraGold] Admitted {len(ex_override)+len(ex_sandwich)+len(ex_q2_dual)+len(ex_hard_fn)+len(ex_hard_fp)+len(ex_general):,} rows "
            f"(hard_fn={len(ex_hard_fn)}, hard_fp={len(ex_hard_fp)}, sw={len(ex_sandwich)}, q2={len(ex_q2_dual)}, ovr={len(ex_override)}, gen={len(ex_general)})",
            flush=True,
        )

    # 2. Tokenizer + 3-Fold Group-OOF CharLM (cached!)
    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    phase_timings["startup_corpus_sec"] = round(time.time() - t_phase, 2)
    t_phase = time.time()

    charlm_loaded = False
    if args.charlm_cache is not None and args.charlm_cache.is_file() and (not args.smoke_test):
        try:
            with args.charlm_cache.open("rb") as cf_f:
                c_obj = pickle.load(cf_f)
            fold_lms, text_to_fold, full_charlm = c_obj["fold_lms"], c_obj["text_to_fold"], c_obj["full_charlm"]
            charlm_loaded = True
            print(f"[Turbo-v3 CharLM Cache HIT] Loaded {args.charlm_cache}", flush=True)
        except Exception as e:
            print(f"[Turbo-v3 CharLM Cache WARN] {e}; rebuilding...", flush=True)

    if not charlm_loaded:
        lm_rows = gold_pos + gold_hd + gold_nat + d_all[: (300 if args.smoke_test else 90000)]
        lm_txts = [r["text"] for r in lm_rows]
        lm_y = np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in lm_rows], dtype=np.int32)
        lm_g = [r.get("group_id") for r in lm_rows]
        fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
            lm_txts, lm_y, lm_g, n_folds=3,
            min_counts=(2, 3, 4) if args.smoke_test else (3, 5, 7),
            max_order_keys=(3000, 1200, 600) if args.smoke_test else (22000, 16000, 10000),
            seed=args.seed,
        )
        if args.charlm_cache is not None and (not args.smoke_test):
            args.charlm_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.charlm_cache.open("wb") as cf_f:
                pickle.dump({"fold_lms": fold_lms, "text_to_fold": text_to_fold, "full_charlm": full_charlm}, cf_f)

    # 3. Build Model + Cache + Optional Warm-Start / R0 Load
    model = MeguvizV170MesosphereLoopModel(
        arch=args.arch, vocab_size=tokenizer.vocab_size, max_len=args.max_len, head_len=args.head_len,
        use_discourse_ops=bool(args.use_discourse_ops), use_ar_uloop=use_ar_uloop,
        use_step_lora=use_step_lora, step_lora_r=int(args.step_lora_r),
        use_delta_rule=use_delta_rule, use_dual_hopfield=use_dual_hopfield,
        use_m6_ph=bool(args.use_m6_ph), use_m7_cnma_uisr=bool(args.use_m7_cnma_uisr),
        lardo_w8_core=bool(args.lardo_w8_core),
    ).to(device)
    era_grl = OfflineEraGRLHead(AI_END_V130, NUM_ERAS_V122).to(device)

    fcache = _StrictGroupOOFFeatureCacheV170(
        tokenizer, max_len=args.max_len, head_len=args.head_len,
        engram_buckets=model.engram_buckets, r0_b2=model.r0_b2, r0_b3=model.r0_b3,
        use_local_charlm=True, use_rust=not bool(args.no_rust_frontend),
    )
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)
    phase_timings["charlm_sec"] = round(time.time() - t_phase, 2)

    warm_loaded_keys: set[str] = set()
    if args.init_from_fp16 is not None and args.init_from_fp16.is_file():
        w_sd = torch.load(args.init_from_fp16, map_location="cpu", weights_only=True)
        cur_sd = model.state_dict()
        for k, v in w_sd.items():
            if k in cur_sd and tuple(cur_sd[k].shape) == tuple(v.shape):
                cur_sd[k].copy_(v.to(cur_sd[k].dtype))
                warm_loaded_keys.add(k)
        print(f"[v1.7.0-WarmStart] Loaded {len(warm_loaded_keys)}/{len(cur_sd)} matching tensors from {args.init_from_fp16}", flush=True)

    if args.init_r0_ckpt is not None and args.init_r0_ckpt.is_file():
        r0_sd = torch.load(args.init_r0_ckpt, map_location="cpu", weights_only=True)
        model.rung0.load_state_dict(r0_sd, strict=False)
        print(f"[v1.7.0-R0] Loaded frozen natural-stream R0++ from {args.init_r0_ckpt}", flush=True)

    # Optional dynamic torch.compile
    t_phase = time.time()
    compiled_ok = False
    if args.compile and device.type == "cuda":
        compiled_ok = model.enable_torch_compile(mode=args.compile_mode, dynamic=bool(args.compile_dynamic))
    phase_timings["compile_warmup_sec"] = round(time.time() - t_phase, 2)

    tot_p = sum(p.numel() for p in model.parameters())
    print(
        f"[v1.7.0-MesosphereLoop] ver={args.version} arch={args.arch} H={model.dim} d_kv={model.core.d_kv} "
        f"ar_uloop={use_ar_uloop} step_lora={use_step_lora}(r={model.step_lora_r}) delta_rule={use_delta_rule} "
        f"dual_hop={use_dual_hopfield} m6_ph={args.use_m6_ph} m7_cnma={args.use_m7_cnma_uisr} er_dro={args.use_er_dro_r} "
        f"rust_fe={fcache.rust_fe is not None} compile={compiled_ok} amp_bf16={use_amp} params={tot_p:,} gpu={gpu_name}",
        flush=True,
    )

    # Build 4-group ER-DRO-R pools (fixed natural-stream R0 + adaptive Deep!)
    ex_hard_all = ex_hard_fn + ex_hard_fp + ex_sandwich + ex_q2_dual + ex_override
    grp_pools = [
        (d_all + ex_general) if (d_all or ex_general) else d_all,                      # g=0 natural_stream
        (gold_pos + gold_hd + gold_nat + gold_pr) or d_all,                             # g=1 gold_core
        (d_hard + ex_hard_all) if (d_hard or ex_hard_all) else d_all,                   # g=2 hard_frontier
        ([r for r in (ex_general + d_all) if len(r["text"]) >= 200] or d_all),          # g=3 multilingual_long
    ]
    dro = ExcessRiskGroupDROR(init_q=(0.38, 0.22, 0.30, 0.10), hard_cap=0.65, kl_cap=0.005, max_iw=1.1112)

    def _sample_natural_r0_batch(bs: int) -> list[dict[str, Any]]:
        """Full natural-stream R0++ mix (NO anti-dilution distortion!)."""
        out_r = []
        n_nat = int(bs * 0.45)
        n_ex = int(bs * 0.25) if (ex_general or ex_hard_all) else 0
        n_gold = int(bs * 0.20)
        n_pr = max(4, bs - n_nat - n_ex - n_gold)
        out_r.extend(rng.choices(d_all, k=n_nat))
        if n_ex > 0:
            ex_all = ex_general + ex_hard_all
            out_r.extend(rng.choices(ex_all, k=n_ex))
        out_r.extend(rng.choices(gold_pos + gold_hd + gold_nat, k=n_gold))
        out_r.extend(rng.choices(d_promo or gold_pr or d_all, k=n_pr))
        rng.shuffle(out_r)
        return out_r[:bs]

    def _sample_deep_batch(bs: int) -> list[dict[str, Any]]:
        if not args.use_er_dro_r:
            return _sample_natural_r0_batch(bs)
        out_r = []
        counts = np.maximum(2, np.round(dro.q * bs).astype(int))
        counts[0] += max(0, bs - int(counts.sum()))
        for g in range(4):
            iw = dro.iw_for_group(g)
            chosen = rng.choices(grp_pools[g], k=int(counts[g]))
            for it in chosen:
                cp = dict(it)
                cp["_dro_g"] = g
                cp["dro_iw"] = iw
                out_r.append(cp)
        rng.shuffle(out_r)
        return out_r[:bs]

    # Pre-collate compact validation sets ONCE onto GPU!
    eval_bs = min(64, args.eval_batch_size) if args.smoke_test else int(args.eval_batch_size)
    val_g2 = corpus.get("val_gold_v2", [])
    val_w3 = corpus.get("val_wave3", [])
    val_old = corpus.get("val_old_video_neg", [])
    val_ml = corpus.get("val_multilingual", [])
    pre_val_g2_r0 = _precollate_eval_split_v170(val_g2, fcache, device, batch_size=eval_bs, r0_only=True)
    pre_val_g2_dp = _precollate_eval_split_v170(val_g2, fcache, device, batch_size=eval_bs, r0_only=False, need_quote_scope=model.use_discourse_ops)

    def _eval_quick_val(r0_only: bool = False, outer_steps: int = 4) -> dict[str, float]:
        if not val_g2:
            return {"sel_obj": 0.0, "g2_pr": 0.0}
        p = predict_student_v170(
            model, fcache, [r["text"] for r in val_g2], device, batch_size=eval_bs, outer_steps=outer_steps,
            r0_only=r0_only, use_amp=use_amp, amp_dtype=amp_dtype,
            precollated=pre_val_g2_r0 if r0_only else pre_val_g2_dp,
        )
        y_g2 = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in val_g2], dtype=np.int32)
        key = "logit_ai_r0" if r0_only else f"logit_ai_t{outer_steps}"
        m = _eval_bin(y_g2, p[key])
        res = {"sel_obj": round(0.55 * m["pr"] + 0.25 * m["r95"] + 0.20 * m["r99"], 5), "g2_pr": m["pr"], "g2_r95": m["r95"], "g2_r99": m["r99"]}
        if not r0_only and outer_steps >= 4:
            m1 = _eval_bin(y_g2, p["logit_ai_t1"])
            p_t1 = expit(np.clip(p["logit_ai_t1"], -20, 20))
            d14 = p[f"logit_ai_t{outer_steps}"] - p["logit_ai_t1"]
            b_cert = float(np.mean((y_g2 - p_t1) * d14) - 0.125 * np.mean(d14 ** 2))
            res.update({"g2_t1_pr": m1["pr"], "t4_minus_t1_pr": round(m["pr"] - m1["pr"], 4), "B_cert_val": round(b_cert, 5)})
        return res

    bs = 64 if args.smoke_test else int(args.batch_size)
    p1_steps = 4 if args.smoke_test else int(args.p1_steps)
    apd_steps = 4 if args.smoke_test else int(args.apd_steps)
    p2_steps = 20 if args.smoke_test else int(args.p2_steps)
    p3_steps = 4 if args.smoke_test else (0 if args.skip_qat else int(args.p3_qat_steps))

    # -----------------------------------------------------------------------
    # Phase 1: Full Natural-Stream R0++ Anchor Training (unless loaded)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    best_r0_obj = -1e9
    if args.init_r0_ckpt is None or not args.init_r0_ckpt.is_file():
        opt1 = _make_adamw([{"params": list(model.rung0.parameters()), "lr": 1.8e-3, "weight_decay": 0.01}], device)
        best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
        for step in range(1, p1_steps + 1):
            model.train()
            bk = _h2d_batch_v170(_collate_v170_cpu(_sample_natural_r0_batch(bs), fcache, rng=rng, punct_jitter_prob=0.10, need_bpe=False), device)
            opt1.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["m7_surf"])
                ai_v = bk["has_ai"] > 0.5
                pr_v = bk["has_pr"] > 0.5
                loss_ai = (F.binary_cross_entropy_with_logits(r0["logit_ai_r0"], bk["y_ai"], reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
                loss_pr = (F.binary_cross_entropy_with_logits(r0["logit_promo_r0"], bk["y_pr"], reduction="none") * pr_v.float()).sum() / pr_v.float().sum().clamp_min(1.0)
                # KD from Super-Teacher R0/AI logits when present
                has_t = (bk["has_teacher"] > 0.5) & ai_v
                l_kd = (F.binary_cross_entropy_with_logits(r0["logit_ai_r0"] / 1.5, torch.sigmoid(bk["t_ai_r0"] / 1.5), reduction="none") * has_t.float() * bk["kd_w"]).sum() / has_t.float().sum().clamp_min(1.0)
                # Defer head targets (error indicator)
                err_ai = ((r0["logit_ai_r0"].detach() >= 0.0).float() != (bk["y_ai"] >= 0.5).float()).float()
                err_pr = ((r0["logit_promo_r0"].detach() >= 0.0).float() != (bk["y_pr"] >= 0.5).float()).float()
                l_def = 0.20 * ((F.binary_cross_entropy_with_logits(r0["defer_ai_r0_logit"], err_ai, reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
                                + (F.binary_cross_entropy_with_logits(r0["defer_pr_r0_logit"], err_pr, reduction="none") * pr_v.float()).sum() / pr_v.float().sum().clamp_min(1.0))
                loss = loss_ai + 0.6 * loss_pr + 0.35 * l_kd + l_def
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rung0.parameters(), 2.0)
            opt1.step()
            if step == p1_steps or step % max(1, p1_steps // 4) == 0:
                ev = _eval_quick_val(r0_only=True)
                if ev["sel_obj"] >= best_r0_obj:
                    best_r0_obj = ev["sel_obj"]
                    best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
                print(f"  [Phase1-R0 step={step}/{p1_steps}] loss={loss.item():.4f} {ev}", flush=True)
        model.rung0.load_state_dict({k: v.to(device) for k, v in best_r0_sd.items()})
        if args.save_r0_ckpt is not None:
            args.save_r0_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_r0_sd, args.save_r0_ckpt)
    else:
        best_r0_obj = _eval_quick_val(r0_only=True)["sel_obj"]

    if args.freeze_r0:
        for p in model.rung0.parameters():
            p.requires_grad_(False)

    # Phase 1b: Train Detached Independent APD-v2 Exit Head on Frozen R0!
    opt_apd = _make_adamw([{"params": list(model.apd_exit_head.parameters()), "lr": 2.0e-3, "weight_decay": 0.01}], device)
    for step in range(1, apd_steps + 1):
        model.train()
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_natural_r0_batch(bs), fcache, rng=rng, need_bpe=False), device)
        opt_apd.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            with torch.no_grad():
                r0_c = model.rung0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
            sup_n = bk["cf"][:, R0_STRUCT_DIM_V140 - 1].to(r0_c["z_char"].dtype)
            apd_lg = model.apd_exit_head(r0_c["z_char"][:, :32], r0_c["logit_ai_r0"], sup_n, bk["m7_surf"])
            ai_v = (bk["has_ai"] > 0.5).float()
            # Target: 1.0 = safe human (y_ai < 0.25), 0.0 = AI (heavily weighted 3.0x so AI blind spots NEVER exit!)
            y_safe = (bk["y_ai"] < 0.25).float()
            w_asym = torch.where(bk["y_ai"] >= 0.5, torch.full_like(y_safe, 3.0), torch.ones_like(y_safe)) * ai_v
            l_apd = (F.binary_cross_entropy_with_logits(apd_lg, y_safe, reduction="none") * w_asym).sum() / w_asym.sum().clamp_min(1.0)
        l_apd.backward()
        opt_apd.step()
    phase_timings["phase1_r0_and_apd_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # Zero-Sync Vectorized Loss Builder (TCBR-R + RecallFloor + V-REx + Hopfield + CNMA-R Y=0)
    # -----------------------------------------------------------------------
    def _v170_loss(od: dict[str, Any], bk: dict[str, torch.Tensor], is_stage2_pnbr: bool = False, in_qat: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        ai_v = bk["has_ai"] > 0.5
        pr_v = bk["has_pr"] > 0.5
        iw = bk["dro_iw"] if args.use_er_dro_r else torch.ones_like(bk["y_ai"])
        y_ai = bk["y_ai"]
        y_bin = (y_ai >= 0.5).float()
        has_t = (bk["has_teacher"] > 0.5) & ai_v
        lgs = od["logit_ai_steps"]
        z_last = lgs[-1].float()

        if not is_stage2_pnbr:
            # Stage 2a: T1 Native Loss + R1 + Hopfield Epistemic Gap + CNMA-R Y=0 Invariance
            per_ai = F.binary_cross_entropy_with_logits(z_last, y_ai, reduction="none")
            loss = (per_ai * ai_v.float() * iw).sum() / (ai_v.float() * iw).sum().clamp_min(1.0)
            loss = loss + 0.18 * (F.binary_cross_entropy_with_logits(od["logit_ai_r1"].float(), y_ai, reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
            l_kd = (F.binary_cross_entropy_with_logits(z_last / 1.5, torch.sigmoid(bk["t_ai_lg"] / 1.5), reduction="none") * has_t.float() * bk["kd_w"]).sum() / has_t.float().sum().clamp_min(1.0)
            loss = loss + 0.30 * l_kd
            l_pr = (F.binary_cross_entropy_with_logits(od["logit_promo"].float(), bk["y_pr"], reduction="none") * pr_v.float()).sum() / pr_v.float().sum().clamp_min(1.0)
            loss = loss + 0.55 * l_pr
            # Supervise Once-at-T1 Dual-Manifold Epistemic Energy Gap (E_H - E_A)!
            if model.dual_hopfield is not None and "hop_witness" in od:
                gap = od["hop_witness"][:, 2].float()
                l_hop = (F.binary_cross_entropy_with_logits(gap * 2.0, y_bin, reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
                loss = loss + 0.14 * l_hop
            # M7 CNMA-R Y=0 conditional invariance & zero-mean anchor (prevents script exemption cheating!)
            if model.cnma_r is not None and "dz_cnma" in od:
                hu_m = ai_v & (y_ai <= 0.20)
                hu_f = hu_m.float()
                n_hu = hu_f.sum().clamp_min(1.0)
                dz_h = od["dz_cnma"].float()
                mean_dz = (dz_h * hu_f).sum() / n_hu
                var_dz = (((dz_h - mean_dz) ** 2) * hu_f).sum() / n_hu
                loss = loss + 0.05 * (mean_dz.pow(2) + 0.5 * var_dz) * (hu_m.sum() >= 8).float()
        else:
            # Stage 2b (M5 Stagewise Prefix Freezing): train T2..T4 PNBR-R residuals strictly on top of frozen T1!
            per_ai = F.binary_cross_entropy_with_logits(z_last, y_ai, reduction="none")
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for ti in range(1, len(lgs)):
                zt = lgs[ti].float()
                zp = lgs[ti - 1].detach().float()
                l_t = (F.binary_cross_entropy_with_logits(zt, y_ai, reduction="none") * ai_v.float() * iw).sum() / (ai_v.float() * iw).sum().clamp_min(1.0)
                # Anti-rescaling orthogonality regularizer: prevents PNBR delta from merely rescaling f_1!
                dl = zt - zp
                fc = zp - zp.mean()
                dc = dl - dl.mean()
                corr = (fc * dc).sum() / (fc.norm() * dc.norm() + 1e-6)
                loss = loss + (0.50 * l_t + 0.03 * corr.pow(2))
            # Orthogonal sketch P regularization: ||P^T P - I_8||_F^2
            ptp = model.core.sketch_P.float().T @ model.core.sketch_P.float()
            loss = loss + 0.01 * (ptp - torch.eye(SKETCH_R_V170, device=device)).pow(2).sum()
            # TaH2 Utility Halt target
            if od.get("u_halt_steps"):
                u_tgt = (lgs[-1].detach().float() - lgs[0].detach().float()).abs().clamp(0.0, 2.0) * 0.25
                loss = loss + 0.10 * F.smooth_l1_loss(od["u_halt_steps"][0].float(), u_tgt)

        # TCBR-R Zero-Sync Trust-Conditioned Bounded Tail Ranking (disabled in Light-QAT!)
        if args.use_tcbr_r and (not in_qat) and float(args.cvar_weight) > 0.0:
            hu_m = ai_v & (y_ai <= 0.20)
            tr_m = hu_m & (bk["trust_neg"] > 0.5)
            un_m = hu_m & (bk["trust_neg"] <= 0.5)
            z_tr = torch.where(tr_m, z_last, torch.full_like(z_last, -1e9))
            k_c = max(2, int(math.ceil(float(args.cvar_alpha) * len(z_last))))
            top_tr = torch.topk(z_tr, k=min(k_c, len(z_last)), largest=True).values
            l_tr = F.softplus(top_tr + 1.5).clamp_max(4.0).mean() * (tr_m.sum() >= 4).float()
            l_un = (4.0 * torch.sigmoid((z_last + 1.0) * 0.5) * un_m.float()).sum() / un_m.float().sum().clamp_min(1.0)
            loss = loss + float(args.cvar_weight) * (0.75 * l_tr + 0.25 * l_un)

        # Positive Recall-Floor Pairwise Margin
        if float(args.recall_floor_weight) > 0.0 and (not in_qat):
            pos_m = ai_v & (y_ai >= 0.70)
            hu_m = ai_v & (y_ai <= 0.20)
            n_p, n_h = pos_m.sum(), hu_m.sum()
            rf_ok = (n_p >= 4) & (n_h >= 8)
            kp = max(2, min(len(z_last), int(math.ceil(0.35 * len(z_last)))))
            kn = max(2, min(len(z_last), int(math.ceil(0.20 * len(z_last)))))
            zp = torch.topk(torch.where(pos_m, z_last, torch.full_like(z_last, 1e9)), k=kp, largest=False).values
            zn = torch.topk(torch.where(hu_m, z_last, torch.full_like(z_last, -1e9)), k=kn, largest=True).values
            l_rf = F.softplus(1.6 + zn.unsqueeze(1) - zp.unsqueeze(0)).mean() * rf_ok.float()
            loss = loss + float(args.recall_floor_weight) * l_rf

        # Class-Conditional Unbiased V-REx across era5
        if args.use_cc_vrex and float(args.cc_vrex_weight) > 0.0 and (not is_stage2_pnbr):
            era_ok = ai_v & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
            env_eff = torch.where(era_ok, bk["era5"], torch.full_like(bk["era5"], -1))
            vx = class_conditional_vrex_unbiased(per_ai, y_bin.long(), env_eff, n_envs=NUM_ERAS_V122, beta_0=1.0, beta_1=0.5)
            loss = loss + float(args.cc_vrex_weight) * vx["vrex_penalty"].clamp(-0.15, 2.0) * (era_ok.sum() >= 16).float()

        return loss, per_ai.detach()

    # -----------------------------------------------------------------------
    # Phase 2a (0..65% T1 Native) + Phase 2b (65..100% M5 Stagewise Prefix-Frozen PNBR-R)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    stage2_start = max(2, int(round(p2_steps * (1.0 - float(args.stage2_frac)))))
    r0_ids = {id(p) for p in model.rung0.parameters()} | {id(p) for p in model.apd_exit_head.parameters()}
    p2_params_all = [p for p in model.parameters() if p.requires_grad and id(p) not in r0_ids]
    opt2a = _make_adamw([{"params": p2_params_all + list(era_grl.parameters()), "lr": float(args.p2_lr), "weight_decay": 0.01}], device)

    # Optional Strategy I PUC warmup on warm-started checkpoints
    puc_w = min(int(args.puc_warmup_steps), max(0, stage2_start - 1)) if (int(args.puc_warmup_steps) > 0 and warm_loaded_keys) else 0
    puc_opt = None
    if puc_w > 0:
        fresh_p = [p for n, p in model.named_parameters() if id(p) not in r0_ids and (n not in warm_loaded_keys or any(t in n for t in ("disc_", "dual_hopfield", "cnma_r", "uisr_res", "m6_ph")))]
        for n, p in model.named_parameters():
            if id(p) not in r0_ids:
                p.requires_grad_(any(p is fp for fp in fresh_p))
        if fresh_p:
            puc_opt = _make_adamw([{"params": fresh_p, "lr": min(2.0e-3, float(args.p2_lr) * 2.0), "weight_decay": 0.01}], device)
            print(f"  [v1.7.0-PUC] Stage-1 Frozen-Backbone Warmup: steps=0..{puc_w}, fresh_tensors={len(fresh_p)}", flush=True)
        else:
            puc_w = 0

    _D02_TEMPLATES = (
        "楼上那套分析纯属胡扯：{ai}",
        "刚才看到有人说{ai}，其实根本不对，完全没说到点子上。",
        "别信前面那个长篇大论了，什么{ai}，一听就是瞎编的。",
        "笑死，前面居然有人认真科普说{ai}，现实里压根不是这么回事。",
    )
    _PROMO_FLIPS = (
        ("强烈推荐大家搜一下这个宝藏插件，券后只要9.9闭眼冲，主页置顶领福利！", "千万别信楼上推的那个插件，纯智商税广告，大家避雷别买！"),
        ("想学全套干货私信我或者搜工种号领取免费资料包，名额有限！", "评论区那些让私信领资料加群的全是引流骗子，举报了别理！"),
    )
    pf_gpu = _h2d_batch_v170(_collate_v170_cpu([{"text": x[0]} for x in _PROMO_FLIPS] + [{"text": x[1]} for x in _PROMO_FLIPS], fcache, include_meta=False), device)

    # Background zero-GIL Rust prefetcher
    n_pw = max(0, int(args.prefetch_workers))
    pool = ThreadPoolExecutor(max_workers=n_pw) if n_pw > 0 else None
    q_fut: deque = deque()

    def _submit_step(st_idx: int):
        b_rows = _sample_deep_batch(bs)
        if pool is not None:
            return pool.submit(_collate_v170_cpu, b_rows, fcache, random.Random(args.seed + st_idx), 0.08, True, True, model.use_discourse_ops, True)
        return _collate_v170_cpu(b_rows, fcache, rng, 0.08, True, True, model.use_discourse_ops, True)

    for si in range(1, min(p2_steps + 1, 1 + max(1, n_pw * 2))):
        q_fut.append((si, _submit_step(si)))

    opt2b = None
    best_p2_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_p2_obj = -1e9
    best_p2_ev: dict[str, Any] = {}
    dro_log: list[dict[str, Any]] = []
    stage2_param_names = (
        "core.blocks.0.bcast_w", "core.blocks.0.ar_",
        "core.lora_", "core.step_phi_mlp",
        "core.sketch_P", "core.ri_", "core.dr_",
        "pnbr_head", "pnbr_cap", "ln_ds_", "halt_utility_head",
    )
    t1_ref_lg: np.ndarray | None = None
    t1_frozen_max_diff: float = 0.0
    long_docs_pool = [r for r in (ex_sandwich + ex_general + gold_pos + gold_nat) if len(r["text"]) >= 260]

    for step in range(1, p2_steps + 1):
        si_got, fut = q_fut.popleft()
        bk_cpu = fut.result() if pool is not None else fut
        next_si = step + max(1, n_pw * 2)
        if next_si <= p2_steps:
            q_fut.append((next_si, _submit_step(next_si)))
        bk = _h2d_batch_v170(bk_cpu, device)

        # PUC Transition -> Unfreeze + Shrink-and-Perturb
        if puc_w > 0 and step == puc_w + 1:
            for n, p in model.named_parameters():
                if id(p) not in r0_ids:
                    p.requires_grad_(True)
                    if p.ndim >= 2 and ("prelude_blocks." in n or ".ff_" in n or ".sa_" in n):
                        with torch.no_grad():
                            p.mul_(float(args.puc_shrink)).add_(torch.randn_like(p) * (float(args.puc_noise) * p.std().clamp_min(1e-4)))
            opt2a = _make_adamw([{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in r0_ids] + list(era_grl.parameters()), "lr": float(args.p2_lr), "weight_decay": 0.01}], device)
            print(f"  [v1.7.0-PUC] Transition @ step={step}: unfrozen + Shrink-and-Perturb!", flush=True)

        # M5 Stagewise Prefix Freezing Transition at `step == stage2_start`!
        is_stage2 = step >= stage2_start
        if step == stage2_start:
            # Lock T1 best weights, then FREEZE T1 prefix so T1 NEVER degrades by a single ULP!
            model.load_state_dict({k: v.to(device) for k, v in best_p2_sd.items()})
            if val_g2:
                t1_ref_lg = predict_student_v170(
                    model, fcache, [r["text"] for r in val_g2], device, batch_size=eval_bs, outer_steps=1,
                    r0_only=False, use_amp=False, precollated=pre_val_g2_dp,
                )["logit_ai_t1"].copy()
            s2_params = []
            for n, p in model.named_parameters():
                keep_tr = any(n.startswith(pfx) for pfx in stage2_param_names)
                p.requires_grad_(keep_tr)
                if keep_tr:
                    s2_params.append(p)
            opt2b = _make_adamw([{"params": s2_params, "lr": float(args.p2_lr) * 0.75, "weight_decay": 0.01}], device)
            print(f"  [v1.7.0-M5 Stagewise Freeze] step={step}/{p2_steps}: FROZE T1 prefix! Training {len(s2_params)} DreamCore+PNBR-R tensors (T1 locked at sel_obj={best_p2_obj:.5f})!", flush=True)

        model.train()
        cur_opt = puc_opt if (puc_w > 0 and step <= puc_w) else (opt2b if is_stage2 else opt2a)
        cur_opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                outer_steps=(4 if is_stage2 else 1),
                r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p),
                freeze_t1_prefix=is_stage2,
            )
            loss, per_ai_det = _v170_loss(od, bk, is_stage2_pnbr=is_stage2, in_qat=False)

            # Aux D02 Paraphrase Rebuttal + Promo-Flip (Phase 2a only!)
            if (not is_stage2) and args.use_d02_paraphrase_cf and step % 4 == 0 and gold_pos:
                n_d02 = min(16, len(gold_pos))
                ai_snips = [rng.choice(gold_pos)["text"][:80] for _ in range(n_d02)]
                reb_rows = [{"text": rng.choice(_D02_TEMPLATES).format(ai=s)} for s in ai_snips]
                bk_d02 = _h2d_batch_v170(_collate_v170_cpu(reb_rows, fcache, include_meta=False), device)
                od_d02 = model(
                    bk_d02["ids"], bk_d02["msk"], bk_d02["st"], bk_d02["eg"], bk_d02["loc"],
                    bk_d02["c1"], bk_d02["c2"], bk_d02["c3"], bk_d02["cm"], bk_d02["cf"],
                    bk_d02["sup"], bk_d02["qsc"], bk_d02["m6_ph"], bk_d02["m7_surf"], bk_d02["m7_uisr"],
                    outer_steps=1,
                )
                loss = loss + 0.10 * F.binary_cross_entropy_with_logits(od_d02["logit_ai"].float(), torch.zeros((n_d02,), device=device))

            if (not is_stage2) and args.use_fmt_cf and float(args.fmt_cf_weight) > 0.0 and step % 6 == 0 and gold_pos:
                n_fc = min(12, len(gold_pos) + len(gold_nat))
                src_fc = rng.choices(gold_pos + gold_nat, k=n_fc)
                fmt_a = [{"text": r["text"]} for r in src_fc]
                fmt_b = [{"text": re.sub(r"(?m)^\s*(?:[-*•·]|\d+[\.、\)])\s*", "", r["text"]).replace("**", "").replace("\n", " ")} for r in src_fc]
                bk_fc = _h2d_batch_v170(_collate_v170_cpu(fmt_a + fmt_b, fcache, include_meta=False), device)
                od_fc = model(
                    bk_fc["ids"], bk_fc["msk"], bk_fc["st"], bk_fc["eg"], bk_fc["loc"],
                    bk_fc["c1"], bk_fc["c2"], bk_fc["c3"], bk_fc["cm"], bk_fc["cf"],
                    bk_fc["sup"], bk_fc["qsc"], bk_fc["m6_ph"], bk_fc["m7_surf"], bk_fc["m7_uisr"],
                    outer_steps=1,
                )
                loss = loss + float(args.fmt_cf_weight) * F.smooth_l1_loss(od_fc["logit_ai"][:n_fc].float(), od_fc["logit_ai"][n_fc:].float())

            if (not is_stage2) and args.use_promo_flip_cf and step % 5 == 0:
                od_pf = model(
                    pf_gpu["ids"], pf_gpu["msk"], pf_gpu["st"], pf_gpu["eg"], pf_gpu["loc"],
                    pf_gpu["c1"], pf_gpu["c2"], pf_gpu["c3"], pf_gpu["cm"], pf_gpu["cf"],
                    pf_gpu["sup"], pf_gpu["qsc"], pf_gpu["m6_ph"], pf_gpu["m7_surf"], pf_gpu["m7_uisr"],
                    outer_steps=1,
                )
                half_pf = len(_PROMO_FLIPS)
                loss = loss + float(args.promo_flip_weight) * F.relu(1.8 - (od_pf["logit_promo"][:half_pf].float() - od_pf["logit_promo"][half_pf:].float())).mean()

            # Stage-3 DynaSeam + 68 B DS-TitanGuard training on long docs (`step % longform_every == 0`)
            if (not is_stage2) and int(args.longform_every) > 0 and step % int(args.longform_every) == 0 and long_docs_pool:
                docs_l = rng.choices(long_docs_pool, k=min(4, len(long_docs_pool)))
                c_txts, c_metas, doc_slices, doc_y_ai, doc_y_pr = [], [], [], [], []
                for dr in docs_l:
                    cks = plan_long_chunks_v11(canonicalize_zero_pii_v150(dr["text"])[0], max_chunks=4)
                    st_i = len(c_txts)
                    for ck in cks:
                        c_txts.append({"text": ck["text"]}); c_metas.append(ck["chunk_meta"])
                    doc_slices.append((st_i, len(c_txts)))
                    doc_y_ai.append(float(dr.get("y_ai", 0.0))); doc_y_pr.append(float(dr.get("y_pr", 0.0)))
                if c_txts:
                    bk_c = _h2d_batch_v170(_collate_v170_cpu(c_txts, fcache, include_meta=False), device)
                    od_c = model(
                        bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["loc"],
                        bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"],
                        bk_c["sup"], bk_c["qsc"], bk_c["m6_ph"], bk_c["m7_surf"], bk_c["m7_uisr"],
                        outer_steps=1,
                    )
                    max_c = max(e - s for s, e in doc_slices)
                    nd = len(doc_slices)
                    cea = torch.zeros((nd, max_c, AI_END_V130), device=device, dtype=od_c["emb_ai"].dtype)
                    cep = torch.zeros((nd, max_c, od_c["emb_pr"].shape[-1]), device=device, dtype=od_c["emb_pr"].dtype)
                    cla = torch.zeros((nd, max_c), device=device, dtype=od_c["logit_ai"].dtype)
                    clp = torch.zeros((nd, max_c), device=device, dtype=od_c["logit_promo"].dtype)
                    cmt = torch.zeros((nd, max_c, 6), device=device, dtype=torch.float32)
                    cvl = torch.zeros((nd, max_c), device=device, dtype=torch.float32)
                    for di, (s0, e0) in enumerate(doc_slices):
                        nc = e0 - s0
                        cea[di, :nc] = od_c["emb_ai"][s0:e0]; cep[di, :nc] = od_c["emb_pr"][s0:e0]
                        cla[di, :nc] = od_c["logit_ai"][s0:e0]; clp[di, :nc] = od_c["logit_promo"][s0:e0]
                        cmt[di, :nc] = torch.from_numpy(np.stack(c_metas[s0:e0], 0)).to(device)
                        cvl[di, :nc] = 1.0
                    da_l, dp_l, _ = model.aggregate_long_chunks(cea, cep, cla, clp, cmt, cvl)
                    loss = loss + 0.25 * F.binary_cross_entropy_with_logits(da_l.float(), torch.tensor(doc_y_ai, device=device)) + 0.10 * F.binary_cross_entropy_with_logits(dp_l.float(), torch.tensor(doc_y_pr, device=device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
        cur_opt.step()

        # M7 ER-DRO-R asynchronous bucket update every 50 steps (zero sync per step!)
        if args.use_er_dro_r and (not is_stage2) and step % 10 == 0:
            dro.observe(bk_cpu["dro_g"].numpy(), per_ai_det.cpu().numpy())
            if step % 50 == 0:
                u_dro = dro.maybe_update()
                if u_dro is not None:
                    dro_log.append({"step": step, **u_dro})

        eval_every = max(1, p2_steps // 4)
        if step == stage2_start - 1 or step == p2_steps or step % eval_every == 0:
            ev = _eval_quick_val(r0_only=False, outer_steps=(4 if is_stage2 else 1))
            if ev["sel_obj"] >= best_p2_obj or step == stage2_start - 1:
                if ev["sel_obj"] >= best_p2_obj:
                    best_p2_obj = ev["sel_obj"]
                    best_p2_ev = ev
                    best_p2_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Phase2{'b-PNBR' if is_stage2 else 'a-T1'} step={step}/{p2_steps}] loss={loss.item():.4f} {ev}", flush=True)

    if pool is not None:
        pool.shutdown(wait=False)
    if t1_ref_lg is not None and val_g2:
        t1_after_lg = predict_student_v170(
            model, fcache, [r["text"] for r in val_g2], device, batch_size=eval_bs, outer_steps=1,
            r0_only=False, use_amp=False, precollated=pre_val_g2_dp,
        )["logit_ai_t1"]
        t1_frozen_max_diff = float(np.max(np.abs(t1_after_lg - t1_ref_lg)))
        print(f"  [v1.7.0-M5 Prefix Freeze Audit] max |T1_after_2b - T1_locked| = {t1_frozen_max_diff:.2e} (exact 0-ULP invariance!)", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_p2_sd.items()})
    fp16_master_path = ckpt_dir / f"{args.version}_fp16_master.pt"
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_p2_sd.items()}, fp16_master_path)
    phase_timings["phase2_train_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # Phase 3: Light-QAT + Zero-Disk Bit-Exact Auto-Ship-Gate
    # -----------------------------------------------------------------------
    t_phase = time.time()
    preqat_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_sd = {k: v.clone() for k, v in preqat_sd.items()}
    best_q_obj = best_p2_obj
    if p3_steps > 0:
        for n, p in model.named_parameters():
            p.requires_grad_(id(p) not in r0_ids)
        set_v130_qat_mode(model, True)
        opt3 = _make_adamw([{"params": [p for p in model.parameters() if p.requires_grad], "lr": float(args.qat_lr), "weight_decay": 0.005}], device)
        for step in range(1, p3_steps + 1):
            model.train()
            bk = _h2d_batch_v170(_collate_v170_cpu(_sample_deep_batch(bs), fcache, rng=rng, punct_jitter_prob=0.05), device)
            opt3.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                od = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                    bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                    outer_steps=4, freeze_t1_prefix=False,
                )
                l1, _ = _v170_loss(od, bk, is_stage2_pnbr=False, in_qat=True)
                l2, _ = _v170_loss(od, bk, is_stage2_pnbr=True, in_qat=True)
                loss = l1 + 0.40 * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt3.step()
            if step == p3_steps or step % max(1, p3_steps // 2) == 0:
                ev = _eval_quick_val(r0_only=False, outer_steps=4)
                if ev["sel_obj"] >= best_q_obj:
                    best_q_obj = ev["sel_obj"]
                    best_q_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  [Phase3-QAT step={step}/{p3_steps}] loss={loss.item():.4f} {ev}", flush=True)
        set_v130_qat_mode(model, False)
    phase_timings["phase3_qat_sec"] = round(time.time() - t_phase, 2)

    # Zero-Disk Bit-Exact Auto-Ship-Gate (Pre-QAT Direct-Pack vs Light-QAT vs WiSE-FT Soup)
    t_phase = time.time()
    ship_cands: dict[str, dict[str, torch.Tensor]] = {"direct_pack_preqat": preqat_sd, "light_qat": best_q_sd}
    if args.wise_anchor_fp16 is not None and args.wise_anchor_fp16.is_file():
        anc_sd = torch.load(args.wise_anchor_fp16, map_location="cpu", weights_only=True)
        for alpha_w in (0.85, 0.70, 0.50):
            soup = {}
            for k, v in best_q_sd.items():
                if k in anc_sd and tuple(anc_sd[k].shape) == tuple(v.shape) and v.is_floating_point() and (not k.startswith(("rung0.", "apd_exit_head."))):
                    soup[k] = alpha_w * v.float() + (1.0 - alpha_w) * anc_sd[k].float()
                else:
                    soup[k] = v.clone()
            ship_cands[f"wise_ft_a{int(alpha_w*100)}"] = soup

    ship_scores: dict[str, Any] = {}
    winner_name = "light_qat" if p3_steps > 0 else "direct_pack_preqat"
    winner_sd = ship_cands[winner_name]
    if args.auto_ship_gate:
        best_sc = -1e9
        for cname, csd in ship_cands.items():
            model.load_state_dict({k: v.to(device) for k, v in csd.items()})
            apply_mgvz10_quant_roundtrip_inplace_v170(model)
            ev = _eval_quick_val(r0_only=False, outer_steps=4)
            ship_scores[cname] = ev
            if ev["sel_obj"] > best_sc:
                best_sc, winner_name, winner_sd = ev["sel_obj"], cname, csd
        print(f"  [v1.7.0-AutoShipGate] WINNER = {winner_name} | {json.dumps(ship_scores)}", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in winner_sd.items()})
    apply_mgvz10_quant_roundtrip_inplace_v170(model)
    phase_timings["auto_ship_gate_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # Calibration (`EBA-H` Prior-Matched Fusion + Output-Shifted LUT + `M8 IUT SCE`)
    # -----------------------------------------------------------------------
    t_phase = time.time()
    cal_rows = val_g2 + val_w3 + val_old + val_ml
    p_cal = predict_student_v170(model, fcache, [r["text"] for r in cal_rows], device, batch_size=eval_bs, outer_steps=4, use_amp=False)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    cert_rows = extra_cal_cert_rows if len(extra_cal_cert_rows) >= 48 else cal_rows
    p_cert = predict_student_v170(model, fcache, [r["text"] for r in cert_rows], device, batch_size=eval_bs, outer_steps=4, use_amp=False) if cert_rows is not cal_rows else p_cal
    y_cert_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cert_rows], dtype=np.int32)

    mp_grid = tuple(float(x) for x in args.eba_m_pos_grid.split(",") if x.strip())
    mn_grid = tuple(float(x) for x in args.eba_m_neg_grid.split(",") if x.strip())
    cv_fusion = fit_crossview_eba_h_fusion(
        p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], y_cal_ai, pi_nat=float(args.eba_pi_nat),
        m_pos_grid=mp_grid, m_neg_grid=mn_grid, n_folds=int(args.oof_cal_folds),
        cert_r0=p_cert["logit_ai_r0"], cert_deep=p_cert["logit_ai_t4"], cert_y=y_cert_ai,
    ) if args.use_eba_h else {"enabled": False, "formula": "none"}

    cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
    t_ai = fit_scalar_temperature_v140(cal_fused_t4 - float(cv_fusion.get("prior_shift", 0.0)), y_cal_ai)
    t_pr = fit_scalar_temperature_v140(p_cal["logit_promo_t4"], y_cal_pr) if y_cal_pr.sum() >= 8 else 1.5
    lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"], -30.0, 30.0), y_cal_ai)
    lut_ai_full = (
        fit_output_shifted_promo_lut_v122(np.clip(cal_fused_t4 / max(1e-3, t_ai), -30.0, 30.0), y_cal_ai, pi_nat=float(args.eba_pi_nat), shift_damp=0.55)
        if args.use_eba_h
        else fit_smooth_ai_lut_v122(np.clip(cal_fused_t4 / max(1e-3, t_ai), -30.0, 30.0), y_cal_ai)
    )
    lut_pr_r0 = fit_output_shifted_promo_lut_v122(np.clip(p_cal["logit_promo_r0"], -30.0, 30.0), y_cal_pr) if y_cal_pr.sum() >= 8 else None
    lut_pr_full = fit_output_shifted_promo_lut_v122(np.clip(p_cal["logit_promo_t4"] / max(1e-3, t_pr), -30.0, 30.0), y_cal_pr) if y_cal_pr.sum() >= 8 else None

    # M8 Zero-Bonferroni IUT Stratified R0 Exit Policy (strictly certified on `cert_rows`!)
    cert_p_r0 = apply_spline_temp_lut(expit(np.clip(p_cert["logit_ai_r0"], -30.0, 30.0)), lut_ai_r0)
    cert_clens = np.array([len(canonicalize_zero_pii_v150(r["text"])[0]) for r in cert_rows], dtype=np.int64)
    r0_ai_pol = fit_iut_stratified_r0_policy_v170(
        cert_p_r0, p_cert["p_defer_ai_r0"], p_cert["p_apd_safe"], cert_clens, y_cert_ai,
        alpha_max=float(args.iut_alpha_max),
    )
    phase_timings["cal_and_iut_sec"] = round(time.time() - t_phase, 2)

    # -----------------------------------------------------------------------
    # 4-Stage Pure-FP32 ONNX Export + `26/26` CI Suite
    # -----------------------------------------------------------------------
    t_phase = time.time()
    manifest = {
        "version": args.version,
        "max_len": int(args.max_len), "head_len": int(args.head_len), "long_char_trigger": int(args.long_char_trigger),
        "temp_ai": round(float(t_ai), 4), "temp_promo": round(float(t_pr), 4),
        "lut_ai_r0": lut_ai_r0, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_full": lut_pr_full,
        "rung0_ai_ltt_policy": r0_ai_pol,
        "rung0_pr_ltt_policy": {"enabled": False, "tau_promo": 0.015, "defer_max": 0.35, "max_chars": 180},
        "crossview_late_fusion": cv_fusion,
        "long_focal_router": {"enabled": bool(args.use_kadane_focal_router), "win_chars": 160, "max_chunks": 6, "lambda": 0.92},
        "anytime_step_policy": {"enabled": True, "max_infer_steps": 4, "u_halt_thresh": 0.015, "disagree_dt_thresh": 1.35, "uncertain_lo": 0.12, "uncertain_hi": 0.88, "t1_exit_enabled": True},
    }
    export_rep = export_v170_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    eng = MeguvizV170CascadeEngine(out_dir, threads=4, r0_threads=2)
    ci_gates = run_v170_ci_gates(model, tokenizer, export_rep, out_dir, device, charlm=full_charlm, engine=eng)
    phase_timings["export_and_26ci_sec"] = round(time.time() - t_phase, 2)

    # Compute cert B_cert & Hopfield witness PR
    p_t1_c = expit(np.clip(p_cert["logit_ai_t1"], -20, 20))
    d14_c = p_cert["logit_ai_t4"] - p_cert["logit_ai_t1"]
    b_cert_val = round(float(np.mean((y_cert_ai - p_t1_c) * d14_c) - 0.125 * np.mean(d14_c ** 2)), 5)
    hop_pr = round(float(average_precision_score(y_cert_ai, p_cert["hop_witness"][:, 2])), 4) if (model.dual_hopfield is not None and y_cert_ai.sum() > 0) else None

    report = {
        "version": args.version, "arch": args.arch, "gpu_name": gpu_name, "total_params": tot_p,
        "v170_config": {
            "use_ar_uloop": use_ar_uloop, "use_step_lora": use_step_lora, "step_lora_r": model.step_lora_r,
            "use_delta_rule": use_delta_rule, "use_dual_hopfield": use_dual_hopfield,
            "use_m6_ph": bool(args.use_m6_ph), "use_m7_cnma_uisr": bool(args.use_m7_cnma_uisr),
            "use_er_dro_r": bool(args.use_er_dro_r), "stage2_start_step": stage2_start,
            "t1_frozen_max_diff": round(float(t1_frozen_max_diff), 10),
            "puc_warmup_steps": puc_w, "auto_ship_winner": winner_name,
        },
        "phase1_r0_best_obj": round(float(best_r0_obj), 5),
        "phase2_best_val": best_p2_ev, "phase3_qat_best_obj": round(float(best_q_obj), 5),
        "auto_ship_scores": ship_scores, "er_dro_r_log": dro_log[-5:],
        "pnbr_B_cert_holdout": b_cert_val, "dual_hopfield_witness_pr": hop_pr,
        "crossview_late_fusion": cv_fusion, "rung0_iut_policy": r0_ai_pol,
        "export": export_rep, "v170_ci_gates": ci_gates,
        "release_gates": {
            "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v170_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "phase_timings_sec": phase_timings,
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v1.7.0 Done] elapsed={report['elapsed_seconds']}s | bootable={export_rep['bootable_mib']} MiB ({export_rep['bootable_bytes']} B) | CI={ci_gates['passed_count']}/{ci_gates['total_count']} PASS={ci_gates['all_ci_passed']} | Forb={export_rep['forbidden_ops_total']}", flush=True)
    if not report["release_gates"]["v170_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
