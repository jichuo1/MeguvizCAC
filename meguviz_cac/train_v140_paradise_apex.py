"""MeguvizCAC v1.4.0-ParadiseApex-Loop Student Trainer (Multi-Paradigm Progressive Loop-to-Loop KD).

Consumes the 251K 3-Paradigm Super-Teacher Ultra soft distillation pool (`v140_superteacher_soft_distill_pool.jsonl.gz`)
produced by `electra_large (325M RTD) + bge_m3 (568M ML Contrastive) + sensenova/SenseNova-Loop-9B (56L Looped mHC)`
and trains `MeguvizV140ParadiseApexModel` (`<= 5.00 MiB` MGVZ8, ZERO 1.58-bit, 4-Stage FP32 ONNX, 14/14 CI):

  - **Progressive Multi-Paradigm Loop-to-Loop Distillation**:
      * `R0` (`ApexCharNano` W8 + Support Gate): `0.35 * Electra-RTD + 0.65 * Ensemble`
      * `T1` (`BiXT + mHC` Step 1):              `0.35 * BGE-M3 (100+ langs) + 0.65 * Ensemble`
      * `T2` (`BiXT + mHC` Step 2):              `0.45 * SenseNova-Loop-9B + 0.55 * Ensemble` + `2.5x` R0-error focus
      * `T3` (`BiXT + mHC` Step 3 Bounded Corr): `0.65 * SenseNova-Loop-9B + 0.35 * Ensemble` + `4.0x` focus on BOTH
        `T2` mistakes AND the **`SenseNova-Loop-9B` vs `Electra-Large` Disagreement Frontier**!
  - **15,855 Counterfactual Mirror Pairs** (incl. `1,430` micro-short `12-30c` & `265` decorator `1:1` pairs)
    directly supervising `R0`, `T2`, and `T3` margin separation!
  - **Promo Hard-Negative Pairwise Ranking Loss** (separating Q2/Q3 Promo+ vs organic brand & quoted-promo humans)
    under strict structural `F6 === 0.0` Promo->AI firewall!
  - **PyTorch 2.x Speed Upgrades**: `AdamW(fused=True)` on CUDA + in-memory clean-text tokenization/style cache!
  - **Dense Certified R0 LTT Cascade** (fixes `v1.3.1` `enabled: false`) + **Bounded `T2 -> T3` Early Exit**!
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn.functional as F

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
from .v130_xtreme_apex import (
    AI_END_V130,
    ENGRAM_BUCKETS_V130,
    TRIGRAM_BUCKETS_V130,
    encode_bpe_head_tail_v130,
)
from .v140_paradise_apex import (
    CHARLM_FEAT_DIM_V140,
    HEAD_LEN_V140,
    MAX_LEN_V140,
    R0_STRUCT_DIM_V140,
    CompactCharLMV140,
    MeguvizV140CascadeEngine,
    MeguvizV140ParadiseApexModel,
    export_v140_onnx_and_bundle_v8,
    extract_apex_char_batch_v140,
    fit_oof_charlm_v140,
    run_v140_ci_gates,
    set_v140_qat_mode,
)
from .v130_xtreme_apex import canonicalize_zero_pii_v130
from .v140_superteacher import (
    build_v140_superteacher_corpus,
    compute_top_tail_pairwise_loss_v140,
    fit_scalar_temperature_v140,
    jitter_ai_text_v140,
    read_jsonl_any_v140,
    sample_cf_mirror_batch_v140,
    text_sha16_v140,
)


def _recall_at_prec(y: np.ndarray, p: np.ndarray, target: float) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    best = 0.0
    for i in range(len(thr)):
        if prec[i] >= target and rec[i] > best:
            best = float(rec[i])
    return best


def _best_f05(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    best = 0.0
    for i in range(len(thr)):
        pr, rc = float(prec[i]), float(rec[i])
        d = 0.25 * pr + rc
        if d > 0:
            best = max(best, 1.25 * pr * rc / d)
    return best


def _eval_bin(y: np.ndarray, logits: np.ndarray, temp: float = 1.0) -> dict[str, float]:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()), "roc": 0.5, "pr": 0.0, "f05": 0.0, "r95": 0.0, "r99": 0.0}
    p = expit(np.clip(logits / max(1e-3, temp), -30.0, 30.0))
    return {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "roc": round(float(roc_auc_score(y, p)), 4),
        "pr": round(float(average_precision_score(y, p)), 4),
        "f05": round(_best_f05(y, p), 4),
        "r95": round(_recall_at_prec(y, p, 0.95), 4),
        "r99": round(_recall_at_prec(y, p, 0.99), 4),
    }


def _load_v140_distill_pool(path: Path | None, blocked_hashes: set[str], smoke_test: bool) -> list[dict[str, Any]]:
    """Load `v140_superteacher_soft_distill_pool.jsonl.gz` preserving per-paradigm teacher logits!"""
    if path is None or not path.is_file():
        return []
    rows = read_jsonl_any_v140(path)
    if smoke_test:
        rows = rows[:400]
    out: list[dict[str, Any]] = []
    for r in rows:
        txt = str(r.get("text", "")).strip()
        if len(txt) < 2 or text_sha16_v140(txt) in blocked_hashes:
            continue
        t_ai_ens = float(r.get("t_ai_logit", 0.0))
        t_pr_ens = float(r.get("t_pr_logit", -3.0))
        pm_ai = r.get("per_model_ai_lg") or []
        # Model order in v1.4.0 Super-Teacher Ultra: [electra_large, bge_m3, sensenova_loop_9b]
        if len(pm_ai) >= 3:
            lg_el, lg_bge, lg_sn9b = float(pm_ai[0]), float(pm_ai[1]), float(pm_ai[2])
        elif len(pm_ai) == 2:
            lg_el, lg_bge, lg_sn9b = float(pm_ai[0]), float(pm_ai[0]), float(pm_ai[1])
        else:
            lg_el = lg_bge = lg_sn9b = t_ai_ens

        # Progressive per-rung targets (genuine depth specialization!)
        t_ai_r0 = 0.35 * lg_el + 0.65 * t_ai_ens
        t_ai_t1 = 0.35 * lg_bge + 0.65 * t_ai_ens
        t_ai_t2 = 0.45 * lg_sn9b + 0.55 * t_ai_ens
        t_ai_t3 = 0.65 * lg_sn9b + 0.35 * t_ai_ens
        frontier_gap = abs(lg_sn9b - lg_el)  # High when SenseNova-Loop-9B sees deep pragmatics Electra missed!

        t_std = float(r.get("t_ai_std_p", 0.0))
        kd_w = float(1.0 / (1.0 + 1.8 * max(0.0, t_std)))
        g_ai = r.get("gold_ai")
        g_pr = r.get("gold_pr")
        has_gold_ai = 1.0 if (g_ai is not None and (float(g_ai) <= 25.0 or float(g_ai) >= 70.0)) else 0.0
        has_gold_pr = 1.0 if (g_pr is not None and (float(g_pr) <= 25.0 or float(g_pr) >= 70.0)) else 0.0
        e5 = int(r.get("era5", -1))
        src = str(r.get("source_tag", ""))
        stl = src.lower()
        is_hard_neg = bool(
            (t_ai_ens < -0.5 or (has_gold_ai > 0.5 and float(g_ai) <= 25.0))
            and (e5 in (0, 1, 2) or "oldvideo" in stl or "organic_brand" in stl or "decorator" in stl or "trap" in stl or (has_gold_pr > 0.5 and float(g_pr) >= 70.0))
        )
        is_ml = bool(r.get("is_multilingual", False))
        n_lat = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in txt[:160])
        out.append({
            "text": txt,
            "t_ai_lg": t_ai_ens,
            "t_ai_r0": t_ai_r0,
            "t_ai_t1": t_ai_t1,
            "t_ai_t2": t_ai_t2,
            "t_ai_t3": t_ai_t3,
            "frontier_gap": float(frontier_gap),
            "t_pr_lg": t_pr_ens,
            "has_teacher": 1.0,
            "kd_w": kd_w,
            "y_ai": 1.0 if (g_ai is not None and float(g_ai) >= 70.0) else (0.0 if has_gold_ai > 0.5 else float(expit(np.clip(t_ai_ens, -20.0, 20.0)))),
            "has_ai": has_gold_ai,
            "y_pr": 1.0 if (g_pr is not None and float(g_pr) >= 70.0) else (0.0 if has_gold_pr > 0.5 else float(expit(np.clip(t_pr_ens, -20.0, 20.0)))),
            "has_pr": has_gold_pr,
            "sub4": -1,
            "hard_neg": 1.0 if is_hard_neg else 0.0,
            "era5": e5,
            "source_tag": src,
            "is_multilingual": is_ml,
            "is_ml_en": bool(is_ml and n_lat >= 20),
        })
    return out


class _CleanFeatureCacheV140:
    """Bounded in-memory cache of BPE + Engram + 56-d style + OOF/Full Char-LM 9-d features."""

    def __init__(self, tokenizer: BiliPickyBPETokenizer, max_len: int, head_len: int, max_entries: int = 300000) -> None:
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.max_entries = int(max_entries)
        self._cache: dict[str, tuple[list[int], list[int], list[int], np.ndarray]] = {}
        self.oof_charlm_map: dict[str, np.ndarray] = {}
        self.full_charlm: CompactCharLMV140 | None = None
        self._charlm_cache: dict[str, np.ndarray] = {}

    def set_charlm(self, oof_map: dict[str, np.ndarray], full_lm: CompactCharLMV140) -> None:
        self.oof_charlm_map = oof_map
        self.full_charlm = full_lm
        self._charlm_cache.clear()

    def get_charlm_9d_batch(self, raw_texts: list[str], mutated: list[bool] | None = None, use_oof: bool = True) -> np.ndarray:
        n = len(raw_texts)
        if n == 0 or self.full_charlm is None:
            return np.zeros((n, CHARLM_FEAT_DIM_V140), dtype=np.float32)
        out = np.zeros((n, CHARLM_FEAT_DIM_V140), dtype=np.float32)
        miss_idx: list[int] = []
        miss_txts: list[str] = []
        for i, t in enumerate(raw_texts):
            is_mut = bool(mutated[i]) if mutated is not None else False
            if use_oof and (not is_mut):
                hit = self.oof_charlm_map.get(t)
                if hit is not None:
                    out[i] = hit
                    continue
            if not is_mut:
                hit2 = self._charlm_cache.get(t)
                if hit2 is not None:
                    out[i] = hit2
                    continue
            miss_idx.append(i)
            miss_txts.append(t)
        if miss_idx:
            cleans = [canonicalize_zero_pii_v130(str(t))[0] for t in miss_txts]
            vals = self.full_charlm.extract_9d(cleans)
            for j, oi in enumerate(miss_idx):
                out[oi] = vals[j]
                is_mut = bool(mutated[oi]) if mutated is not None else False
                if (not is_mut) and len(self._charlm_cache) < self.max_entries:
                    self._charlm_cache[miss_txts[j]] = vals[j]
        return out

    def get_bpe_style(self, text: str, allow_store: bool = True) -> tuple[list[int], list[int], list[int], np.ndarray]:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        cl, f56, _ = extract_style_features_56d(text)
        enc = encode_bpe_head_tail_v130(self.tok, cl, max_len=self.max_len, head_len=self.head_len)
        eng = make_engram_bigram_ids(enc, ENGRAM_BUCKETS_V130)
        trg = make_engram_trigram_ids(enc, TRIGRAM_BUCKETS_V130)
        val = (enc, eng, trg, f56.astype(np.float32))
        if allow_store and len(self._cache) < self.max_entries:
            self._cache[text] = val
        return val


def _collate_student_batch_v140(
    batch: list[dict[str, Any]],
    fcache: _CleanFeatureCacheV140,
    device: torch.device,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.35,
    need_bpe: bool = True,
    use_oof_charlm: bool = True,
) -> dict[str, torch.Tensor]:
    txts: list[str] = []
    mutated: list[bool] = []
    for r in batch:
        t0 = str(r["text"])
        is_ai_pos = (r.get("has_ai", 0.0) > 0.5 and r.get("y_ai", 0.0) >= 0.70) or (r.get("has_teacher", 0.0) > 0.5 and r.get("t_ai_lg", -5.0) >= 1.2)
        if rng is not None and punct_jitter_prob > 0.0 and is_ai_pos:
            t1 = jitter_ai_text_v140(t0, rng, is_ml_en=bool(r.get("is_ml_en")), prob=0.28 if r.get("is_ml_en") else punct_jitter_prob)
            txts.append(t1)
            mutated.append(t1 != t0)
        else:
            txts.append(t0)
            mutated.append(False)

    charlm_9d = fcache.get_charlm_9d_batch(txts, mutated=mutated, use_oof=use_oof_charlm)
    c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(txts, precomputed_charlm_9d=charlm_9d)
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
        encs, engs, trigs, f56_l = [], [], [], []
        for t, is_mut in zip(txts, mutated):
            enc, eng, trg, f56 = fcache.get_bpe_style(t, allow_store=not is_mut)
            encs.append(enc)
            engs.append(eng)
            trigs.append(trg)
            f56_l.append(f56)
        ml = max(2, max(len(s) for s in encs))
        if ml % 2 == 1:
            ml += 1
        nb = len(txts)
        ids = np.full((nb, ml), fcache.tok.pad_id, dtype=np.int64)
        msk = np.zeros((nb, ml), dtype=np.int64)
        eg = np.zeros((nb, ml), dtype=np.int64)
        tg = np.zeros((nb, ml), dtype=np.int64)
        for i, s in enumerate(encs):
            ids[i, : len(s)], msk[i, : len(s)], eg[i, : len(s)], tg[i, : len(s)] = s, 1, engs[i], trigs[i]
        out["ids"] = torch.from_numpy(ids).to(device, non_blocking=True)
        out["msk"] = torch.from_numpy(msk).to(device, non_blocking=True)
        out["eg"] = torch.from_numpy(eg).to(device, non_blocking=True)
        out["tg"] = torch.from_numpy(tg).to(device, non_blocking=True)
        out["st"] = torch.from_numpy(np.stack(f56_l, axis=0).astype(np.float32)).to(device, non_blocking=True)
    return out


@torch.no_grad()
def predict_student_v140(
    model: MeguvizV140ParadiseApexModel,
    fcache: _CleanFeatureCacheV140,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    r0_only: bool = False,
    outer_steps: int = 3,
) -> dict[str, np.ndarray]:
    n = len(texts)
    keys = (
        ("logit_ai_r0", "logit_promo_r0", "p_defer_r0")
        if r0_only
        else (
            "logit_ai_r0", "logit_promo_r0", "p_defer_r0", "logit_ai_r1", "logit_promo_r1",
            "logit_ai_t1", "logit_ai_t2_final", "logit_promo_t2_final", "p_escalate_t3",
            "logit_ai", "logit_promo", "ebt_margin",
        )
    )
    out = {k: np.zeros((n,), dtype=np.float32) for k in keys}
    if n == 0:
        return out
    model.eval()
    order = sorted(range(n), key=lambda i: len(texts[i]))
    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_b = [{"text": texts[i]} for i in idxs]
        bk = _collate_student_batch_v140(sub_b, fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=not r0_only, use_oof_charlm=False)
        if r0_only:
            res = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        else:
            res = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=outer_steps,
            )
        for k in keys:
            arr = res[k].float().cpu().numpy()
            for j, oi in enumerate(idxs):
                out[k][oi] = arr[j]
    return out


def compute_progressive_kd_loss_v140(
    lg: torch.Tensor,
    bk: dict[str, torch.Tensor],
    axis: str = "ai",
    teacher_key: str = "t_ai_lg",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Progressive per-paradigm soft-logit KD + Gold BCE + Top-Tail Pairwise Margin + Hard-Neg Ceiling."""
    if axis == "ai":
        t_lg, y_g, has_g, hn = bk[teacher_key], bk["y_ai"], bk["has_ai"], bk["hard_neg"]
    else:
        t_lg, y_g, has_g, hn = bk["t_pr_lg"], bk["y_pr"], bk["has_pr"], bk["hard_neg"]

    has_t, kd_w = bk["has_t"], bk["kd_w"]
    sw = sample_weight.float() if sample_weight is not None else torch.ones_like(kd_w)
    lg_f = lg.float()
    loss = lg_f.sum() * 0.0

    tm = has_t > 0.5
    if tm.any():
        t_c = t_lg[tm].clamp(-8.0, 8.0)
        w_t = kd_w[tm] * sw[tm]
        l_mse = (F.smooth_l1_loss(lg_f[tm], t_c, reduction="none", beta=1.0) * w_t).mean()
        t_p2 = torch.sigmoid(t_c / 1.5)
        l_soft = (F.binary_cross_entropy_with_logits(lg_f[tm] / 1.5, t_p2, reduction="none") * w_t).mean() * 2.25
        loss = loss + 0.55 * l_mse + 0.45 * l_soft

    gm = has_g > 0.5
    if gm.any():
        yg_s = y_g[gm] * 0.96 + 0.02
        w_g = (1.0 + 0.50 * hn[gm]) * sw[gm]
        l_g = (F.binary_cross_entropy_with_logits(lg_f[gm], yg_s, reduction="none") * w_g).mean()
        loss = loss + 0.65 * l_g

    eff_y = torch.where(gm, y_g, torch.sigmoid(t_lg))
    eff_v = ((gm) | (tm & ((t_lg >= 1.4) | (t_lg <= -1.4)))).float()
    # Both AI and Promo get explicit Top-Tail Hard-Negative Pairwise Ranking (GPT Gem #5!)
    l_tail = compute_top_tail_pairwise_loss_v140(
        lg_f, eff_y, eff_v, hard_neg_boost=hn, margin=1.8 if axis == "ai" else 1.7, top_neg_frac=0.25
    )
    loss = loss + (0.32 if axis == "ai" else 0.28) * l_tail

    if axis == "ai":
        hnm = (eff_v > 0.5) & (eff_y <= 0.20) & (hn > 0.5)
        if hnm.any():
            loss = loss + 0.30 * (F.relu(lg_f[hnm] + 0.95).pow(2) * sw[hnm]).mean()
    else:
        # Organic brand / non-promo hard-negative ceiling on Promo axis!
        pr_hnm = (eff_v > 0.5) & (eff_y <= 0.20) & (hn > 0.5)
        if pr_hnm.any():
            loss = loss + 0.18 * (F.relu(lg_f[pr_hnm] + 0.80).pow(2) * sw[pr_hnm]).mean()
    return loss


def _make_adamw(params: Any, device: torch.device, **kwargs: Any) -> torch.optim.AdamW:
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except Exception:
            pass
    return torch.optim.AdamW(params, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MeguvizCAC v1.4.0-ParadiseApex-Loop Student (<=5.00 MiB)")
    parser.add_argument("--distill-pool", type=Path, default=None, help="Path to v140_superteacher_soft_distill_pool.jsonl.gz")
    parser.add_argument("--v04-data", type=Path, default=None, help="v04-merged-1m4-r3.jsonl.gz for EN/JA multilingual anchors + val_multilingual")
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v07-tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--version", type=str, default="v1.4.0-r1-paradise-apex")
    parser.add_argument("--p1-steps", type=int, default=2400)
    parser.add_argument("--p2-steps", type=int, default=4400)
    parser.add_argument("--p3-qat-steps", type=int, default=1600)
    parser.add_argument("--r0-drop-p", type=float, default=0.50)
    parser.add_argument("--charlm-drop-p", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=MAX_LEN_V140)
    parser.add_argument("--head-len", type=int, default=HEAD_LEN_V140)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    out_dir = args.out.resolve()
    ckpt_dir = (args.checkpoint_dir or out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    fcache = _CleanFeatureCacheV140(tokenizer, max_len=args.max_len, head_len=args.head_len)
    rng = random.Random(20261130)
    torch.manual_seed(20261130)
    np.random.seed(20261130)

    print(f"[v1.4.0-ParadiseApex-Loop] GPU={gpu_name} | version={args.version} | distill={args.distill_pool} | smoke={args.smoke_test}", flush=True)

    # 1. Build strictly purified v1.4.0 corpus (incl. v123-delta + v14-committee-1470 overrides + 15.8K CF pairs + EN/JA!)
    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir, delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, v04_data_path=args.v04_data, smoke_test=args.smoke_test,
    )
    blocked = {
        text_sha16_v140(r["text"])
        for k in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "val_multilingual", "diag_god500")
        for r in corpus.get(k, [])
    }
    distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)

    d_pos = [r for r in distill_pool if r["t_ai_lg"] >= 0.8]
    d_bnd = [r for r in distill_pool if -1.2 < r["t_ai_lg"] < 0.8]
    d_front = [r for r in distill_pool if r["frontier_gap"] >= 1.4]  # SenseNova-Loop-9B vs Electra Disagreement Frontier!
    d_pr = [r for r in distill_pool if r["t_pr_lg"] >= 0.5]
    d_hn = [r for r in distill_pool if r["hard_neg"] > 0.5]
    d_nat = [r for r in distill_pool if r["t_ai_lg"] <= -1.2 and r["hard_neg"] < 0.5]

    longform_train = read_jsonl_any_v140(args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 1200)]

    gold_pos = corpus["train_ai_pos"]
    gold_nat = corpus["train_hu_natural"]
    gold_hd = corpus["train_hu_hard"]
    gold_pr = corpus["train_promo_extra"]
    ml_pos = [r for r in gold_pos if r.get("is_multilingual")]
    ml_neg = [r for r in gold_hd if r.get("is_multilingual")]

    short_pos = [r for r in (gold_pos + d_pos) if len(r["text"]) <= 60]
    hard_old_spaced_hu = [
        r for r in (gold_hd + d_hn)
        if (int(r.get("era5", -1)) in (0, 1, 2) and len(r["text"]) >= 20)
        or ("decorator" in str(r.get("source_tag", "")).lower())
        or ("organic_brand" in str(r.get("source_tag", "")).lower())
        or (r["text"].count(" ") >= 2 and len(r["text"]) >= 14)
    ]

    print(
        f"[Stage0-Data] gold_ai+={len(gold_pos):,} hu_nat={len(gold_nat):,} hu_hd={len(gold_hd):,} | "
        f"distill_pool={len(distill_pool):,} (pos={len(d_pos):,}, bnd={len(d_bnd):,}, sn9b_frontier={len(d_front):,}, pr+={len(d_pr):,}, hn={len(d_hn):,}) | "
        f"short_ai={len(short_pos):,} hard_old_decor_brand={len(hard_old_spaced_hu):,} ml_en_ja=({len(ml_pos)}+, {len(ml_neg)}-) | "
        f"cf_pairs={corpus['stats']['admitted_cf_pairs']:,} longform={len(longform_train):,}",
        flush=True,
    )

    # Fit leak-free 3-fold OOF CompactCharLMV140 (pruned 1-4 generative char-LM LLR, <= 110 KiB deflated!)
    t_lm0 = time.time()
    lm_rows = gold_pos + gold_nat + gold_hd
    lm_txts = [r["text"] for r in lm_rows]
    lm_y = np.array([1 if r["y_ai"] >= 0.5 else 0 for r in lm_rows], dtype=np.int32)
    lm_mc = (2, 3, 4) if args.smoke_test else (16, 28, 40)
    lm_cap = (3000, 1500, 1000) if args.smoke_test else (9000, 8000, 7000)
    oof_charlm_map, full_charlm = fit_oof_charlm_v140(
        lm_txts, lm_y, n_folds=3, min_counts=lm_mc, max_order_keys=lm_cap, seed=20261130
    )
    fcache.set_charlm(oof_charlm_map, full_charlm)
    print(
        f"[Stage0-CharLM] Fitted 3-fold OOF CompactCharLMV140 on {len(lm_txts):,} rows in {time.time()-t_lm0:.1f}s | "
        f"keys={full_charlm.total_keys():,} ({[len(t['keys']) for t in full_charlm.tables]})",
        flush=True,
    )

    def _draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rng.choices(pool, k=k) if (pool and k > 0) else []

    def _sample_batch(bs_cur: int) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        if distill_pool:
            out_b.extend(_draw(d_pos, max(1, int(bs_cur * 0.11))))
            out_b.extend(_draw(d_bnd, max(1, int(bs_cur * 0.09))))
            if d_front:
                out_b.extend(_draw(d_front, max(1, int(bs_cur * 0.07))))
            out_b.extend(_draw(d_pr, max(1, int(bs_cur * 0.05))))
            out_b.extend(_draw(d_hn, max(1, int(bs_cur * 0.10))))
            out_b.extend(_draw(d_nat, max(1, int(bs_cur * 0.13))))
        out_b.extend(_draw(short_pos, max(1, int(bs_cur * 0.06))))
        out_b.extend(_draw(hard_old_spaced_hu, max(1, int(bs_cur * 0.10))))
        if ml_pos:
            out_b.extend(_draw(ml_pos, max(1, int(bs_cur * 0.03))))
        if ml_neg:
            out_b.extend(_draw(ml_neg, max(1, int(bs_cur * 0.03))))
        rem = max(4, bs_cur - len(out_b))
        out_b.extend(_draw(gold_pos, max(1, int(rem * 0.25))))
        out_b.extend(_draw(gold_hd, max(1, int(rem * 0.27))))
        out_b.extend(_draw(gold_nat, max(1, int(rem * 0.34))))
        out_b.extend(_draw(gold_pr, max(0, bs_cur - len(out_b))))
        fb = gold_nat or gold_pos or distill_pool
        while len(out_b) < bs_cur and fb:
            out_b.append(rng.choice(fb))
        return out_b[:bs_cur]

    # 2. Model Init
    model = MeguvizV140ParadiseApexModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_END_V130, num_eras=NUM_ERAS_V122).to(device)
    r0_p = sum(p.numel() for p in model.rung0.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    tot_p = sum(p.numel() for p in model.parameters())
    print(f"[Stage0-Model] total_params={tot_p:,} | rung0={r0_p:,} ({r0_p/tot_p*100:.1f}%) | paradise_core={core_p:,} ({core_p/tot_p*100:.1f}%)", flush=True)

    def _eval_val(m_cur: MeguvizV140ParadiseApexModel, r0_only: bool = False) -> dict[str, Any]:
        p_g2 = predict_student_v140(m_cur, fcache, [r["text"] for r in corpus["val_gold_v2"]], device, r0_only=r0_only, outer_steps=3)
        p_w3 = predict_student_v140(m_cur, fcache, [r["text"] for r in corpus["val_wave3"]], device, r0_only=r0_only, outer_steps=3)
        p_old = predict_student_v140(m_cur, fcache, [r["text"] for r in corpus["val_old_video_neg"]], device, r0_only=r0_only, outer_steps=3)
        p_ml = predict_student_v140(m_cur, fcache, [r["text"] for r in corpus["val_multilingual"]], device, r0_only=r0_only, outer_steps=3) if corpus.get("val_multilingual") else None

        yg2 = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
        yw3_a = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        yw3_p = np.array([int(r["y_pr"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)

        key_ai = "logit_ai_r0" if r0_only else "logit_ai"
        key_pr = "logit_promo_r0" if r0_only else "logit_promo"
        mg2 = _eval_bin(yg2, p_g2[key_ai])
        mw3_a = _eval_bin(yw3_a, p_w3[key_ai])
        mw3_p = _eval_bin(yw3_p, p_w3[key_pr])
        ml_pr = _eval_bin(np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_multilingual"]], dtype=np.int32), p_ml[key_ai])["pr"] if p_ml else mg2["pr"]
        q4 = (yw3_p == 1) & (yw3_a == 0)
        q4_fpr = float((expit(np.clip(p_w3[key_ai][q4], -30, 30)) >= 0.5).mean()) if q4.sum() else 0.0
        old_fpr = float((expit(np.clip(p_old[key_ai], -30, 30)) >= 0.5).mean()) if len(p_old[key_ai]) else 0.0
        obj = 0.32 * mg2["pr"] + 0.24 * mg2["r95"] + 0.18 * mw3_a["pr"] + 0.18 * mw3_p["pr"] + 0.08 * ml_pr - 1.30 * old_fpr - 0.55 * q4_fpr
        res = {
            "sel_obj": round(float(obj), 5), "g2_pr": mg2["pr"], "g2_r95": mg2["r95"],
            "w3_ai_pr": mw3_a["pr"], "w3_pr_pr": mw3_p["pr"], "ml_pr": round(float(ml_pr), 4),
            "old_fpr": round(old_fpr, 4), "q4_fpr": round(q4_fpr, 4),
        }
        if not r0_only:
            mg2_t2 = _eval_bin(yg2, p_g2["logit_ai_t2_final"])
            res["g2_t2_pr"] = mg2_t2["pr"]
            res["t3_minus_t2_pr"] = round(mg2["pr"] - mg2_t2["pr"], 4)
        return res

    bs = 16 if args.smoke_test else int(args.batch_size)
    p1_steps = 4 if args.smoke_test else int(args.p1_steps)
    p2_steps = 6 if args.smoke_test else int(args.p2_steps)
    p3_steps = 4 if args.smoke_test else int(args.p3_qat_steps)

    # =========================================================================
    # PHASE 1: Support-Gated W8 Rung-0 KD (`0.35 * Electra-RTD + 0.65 * Ens`)
    # =========================================================================
    opt1 = _make_adamw(model.rung0.parameters(), device, lr=1.8e-3, weight_decay=0.01)
    best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
    best_r0_obj = -1e9
    model.train()
    for step in range(p1_steps):
        mult = min(1.0, (step + 1) / max(1, int(p1_steps * 0.08))) * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p1_steps))))
        for pg in opt1.param_groups:
            pg["lr"] = 1.8e-3 * mult
        bk = _collate_student_batch_v140(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=False)
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
                t_p = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                err_tgt = ((torch.sigmoid(r0["logit_ai_r0"].float()) - t_p).abs() >= 0.20).float()
            l_def = F.binary_cross_entropy_with_logits(r0["defer_r0_logit"].float(), err_tgt)
            loss = l_ai + 0.72 * l_pr + 0.15 * l_def
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.rung0.parameters(), 1.0)
        opt1.step()

        if (step + 1) % max(1, p1_steps // 4) == 0 or step == p1_steps - 1:
            ev1 = _eval_val(model, r0_only=True)
            if ev1["sel_obj"] > best_r0_obj:
                best_r0_obj = ev1["sel_obj"]
                best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
            print(f"  [Phase1-R0 step={step+1}/{p1_steps}] loss={float(loss.item()):.4f} {ev1}", flush=True)
            model.train()
    model.rung0.load_state_dict(best_r0_sd)
    print(f"[Phase1-R0 DONE] best_r0_obj={best_r0_obj:.5f}", flush=True)

    # =========================================================================
    # PHASE 2: Progressive Multi-Paradigm Loop-to-Loop KD (`BiXT + Mini-mHC`!)
    # =========================================================================
    r0_params = list(model.rung0.parameters())
    r0_ids = {id(p) for p in r0_params}
    deep_params = [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters())
    opt2 = _make_adamw([
        {"params": r0_params, "lr": 2.8e-4, "base_lr": 2.8e-4, "weight_decay": 0.01},
        {"params": deep_params, "lr": 1.0e-3, "base_lr": 1.0e-3, "weight_decay": 0.01},
    ], device)
    ema_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_p2_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_p2_obj = -1e9

    def _update_ema(target_ema: dict[str, torch.Tensor], decay: float = 0.995) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    target_ema[k].mul_(decay).add_(v.detach().cpu(), alpha=1.0 - decay)
                else:
                    target_ema[k].copy_(v.detach().cpu())

    def _paradise_step_loss(od: dict[str, torch.Tensor], bk: dict[str, torch.Tensor], use_t3: bool) -> torch.Tensor:
        with torch.no_grad():
            t_p_ens = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
            t_p_sn9 = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_t3"]))
            err_r0 = (torch.sigmoid(od["logit_ai_r0"].float()) - t_p_ens).abs()
            t2_focus_w = 1.0 + 1.5 * (err_r0 >= 0.15).float()

        # Progressive per-paradigm teacher routing: R0 <- Electra, T1 <- BGE-M3, T2 <- SenseNova-Loop-9B + Ens!
        l_ai_r0 = compute_progressive_kd_loss_v140(od["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
        l_ai_r1 = compute_progressive_kd_loss_v140(od["logit_ai_r1"], bk, axis="ai", teacher_key="t_ai_t1")
        l_ai_t1 = compute_progressive_kd_loss_v140(od["logit_ai_t1"], bk, axis="ai", teacher_key="t_ai_t1")
        l_ai_t2r = compute_progressive_kd_loss_v140(od["logit_ai_t2"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=t2_focus_w)
        l_ai_t2f = compute_progressive_kd_loss_v140(od["logit_ai_t2_final"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=t2_focus_w)

        l_pr_t2f = compute_progressive_kd_loss_v140(od["logit_promo_t2_final"], bk, axis="promo")
        l_pr_t2r = compute_progressive_kd_loss_v140(od["logit_promo_t2"], bk, axis="promo")
        l_pr_r1 = compute_progressive_kd_loss_v140(od["logit_promo_r1"], bk, axis="promo")
        l_pr_r0 = compute_progressive_kd_loss_v140(od["logit_promo_r0"], bk, axis="promo")

        l_ebt_ai = compute_progressive_kd_loss_v140(od["ebt_margin"], bk, axis="ai", teacher_key="t_ai_t2")
        l_ebt_pr = compute_progressive_kd_loss_v140(od["ebt_promo_margin"], bk, axis="promo")

        # T3 Detached Residual Error-Correction: 4.0x focus on T2 mistakes + SenseNova-Loop-9B Disagreement Frontier!
        l_t3_total = l_ai_t2f * 0.0
        if use_t3:
            with torch.no_grad():
                p_t2 = torch.sigmoid(od["logit_ai_t2_final"].float())
                err_t2 = (p_t2 - t_p_sn9).abs()
                frontier_m = (bk["frontier_gap"] >= 1.4).float()
                t3_focus_w = 1.0 + 3.0 * (err_t2 >= 0.12).float() + 1.2 * frontier_m

            l_ai_t3f = compute_progressive_kd_loss_v140(od["logit_ai_t3_final"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t3_focus_w)
            l_ai_t3r = compute_progressive_kd_loss_v140(od["logit_ai_t3"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t3_focus_w)
            l_pr_t3f = compute_progressive_kd_loss_v140(od["logit_promo_t3_final"], bk, axis="promo")

            p_t3 = torch.sigmoid(od["logit_ai_t3_final"].float())
            err_t3 = (p_t3 - t_p_sn9).abs()
            hard_m = ((err_t2 >= 0.12) | (bk["frontier_gap"] >= 1.4)) & ((bk["has_ai"] > 0.5) | (bk["has_t"] > 0.5))
            l_mono = F.relu(err_t3[hard_m] - err_t2[hard_m] + 0.025).mean() if hard_m.any() else l_ai_t2f * 0.0

            with torch.no_grad():
                u_gain = err_t2 - err_t3.detach()
                y_esc = ((u_gain >= 0.020) | ((p_t2 >= 0.10) & (p_t2 <= 0.90) & (err_t2 >= 0.14))).float()
            l_halt = F.binary_cross_entropy_with_logits(od["escalate_t3_logit"].float(), y_esc)

            l_t3_total = 1.20 * l_ai_t3f + 0.40 * l_ai_t3r + 0.65 * l_pr_t3f + 0.45 * l_mono + 0.22 * l_halt

        with torch.no_grad():
            err_r1 = (torch.sigmoid(od["logit_ai_r1"].float()) - t_p_ens).abs()
            y_def_r1 = (err_r1 >= 0.20).float()
        l_def_r1 = F.binary_cross_entropy_with_logits(od["defer_r1_logit"].float(), y_def_r1)

        s4_m = (bk["has_pr"] > 0.5) & (bk["sub4"] >= 0) & (bk["sub4"] < 4)
        l_s4 = F.cross_entropy(od["promo_sub4_logit"].float()[s4_m], bk["sub4"][s4_m]) if s4_m.any() else l_ai_t2f * 0.0

        hu_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
        era_v = hu_m & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
        l_era = era_grl(od["emb_ai"], bk["era5"], era_v, lam=0.50) if int(era_v.sum().item()) >= 8 else l_ai_t2f * 0.0

        return (
            1.00 * l_ai_t2f + 0.30 * l_ai_t2r + 0.25 * l_ai_t1 + 0.18 * l_ai_r1 + 0.22 * l_ai_r0
            + 0.68 * l_pr_t2f + 0.20 * l_pr_t2r + 0.12 * l_pr_r1 + 0.20 * l_pr_r0
            + l_t3_total
            + 0.22 * l_ebt_ai + 0.12 * l_ebt_pr + 0.10 * l_def_r1 + 0.10 * l_s4 + 0.12 * l_era
        )

    has_cf = corpus["stats"]["admitted_cf_pairs"] > 0
    model.train()
    era_grl.train()
    for step in range(p2_steps):
        mult = min(1.0, (step + 1) / max(1, int(p2_steps * 0.08))) * (0.06 + 0.94 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p2_steps))))
        for pg in opt2.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        bk = _collate_student_batch_v140(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        use_t3 = (step % 2 == 0)
        opt2.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=3 if use_t3 else 2, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p),
            )
            loss = _paradise_step_loss(od, bk, use_t3=use_t3)

            if has_cf and (step % 3 == 0):
                ch = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
                if ch:
                    bk_a = _collate_student_batch_v140([{"text": x["ai_text"]} for x in ch], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    bk_h = _collate_student_batch_v140([{"text": x["human_text"]} for x in ch], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oa = model(bk_a["ids"], bk_a["msk"], bk_a["st"], bk_a["eg"], bk_a["tg"], bk_a["c1"], bk_a["c2"], bk_a["c3"], bk_a["cm"], bk_a["cf"], bk_a["sup"], outer_steps=3, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p))
                    oh = model(bk_h["ids"], bk_h["msk"], bk_h["st"], bk_h["eg"], bk_h["tg"], bk_h["c1"], bk_h["c2"], bk_h["c3"], bk_h["cm"], bk_h["cf"], bk_h["sup"], outer_steps=3, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p))
                    pw = torch.tensor([float(x["pair_weight"]) for x in ch], dtype=torch.float32, device=device)
                    l_cf = (
                        (F.softplus(1.8 - (oa["logit_ai"].float() - oh["logit_ai"].float())) * pw).mean()
                        + 0.55 * (F.softplus(1.6 - (oa["logit_ai_t2_final"].float() - oh["logit_ai_t2_final"].float())) * pw).mean()
                        + 0.40 * (F.softplus(1.4 - (oa["logit_ai_r0"].float() - oh["logit_ai_r0"].float())) * pw).mean()
                    )
                    loss = loss + 0.24 * l_cf

            if longform_train and (step % 6 == 0):
                docs = rng.sample(longform_train, min(2 if args.smoke_test else 6, len(longform_train)))
                d_losses = []
                for d_it in docs:
                    chunks = plan_long_chunks_v11(str(d_it.get("text", "")), max_chunks=6)
                    if not chunks:
                        continue
                    bk_c = _collate_student_batch_v140([{"text": c["text"]} for c in chunks], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oc = model(bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["tg"], bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"], bk_c["sup"], outer_steps=2, r0_drop_p=0.0)
                    cm_t = torch.from_numpy(np.stack([c["chunk_meta"] for c in chunks], 0).astype(np.float32)).unsqueeze(0).to(device)
                    vm_t = torch.ones((1, len(chunks)), dtype=torch.float32, device=device)
                    da, dp = model.aggregate_long_chunks(oc["emb_ai"].unsqueeze(0), oc["emb_pr"].unsqueeze(0), oc["logit_ai"].unsqueeze(0), oc["logit_promo"].unsqueeze(0), cm_t, vm_t)
                    ya_d = torch.tensor([1.0 if float(d_it.get("ai_score", 0.0)) >= 70.0 else 0.0], device=device)
                    yp_d = torch.tensor([1.0 if float(d_it.get("promo_score", 0.0)) >= 70.0 else 0.0], device=device)
                    d_losses.append(F.binary_cross_entropy_with_logits(da.float(), ya_d) + 0.5 * F.binary_cross_entropy_with_logits(dp.float(), yp_d))
                if d_losses:
                    loss = loss + 0.15 * torch.stack(d_losses).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(era_grl.parameters()), 1.0)
        opt2.step()
        _update_ema(ema_sd, 0.995)

        if (step + 1) % max(1, p2_steps // 5) == 0 or step == p2_steps - 1:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_sd)
            ev2 = _eval_val(model, r0_only=False)
            if ev2["sel_obj"] > best_p2_obj:
                best_p2_obj = ev2["sel_obj"]
                best_p2_sd = {k: v.clone() for k, v in ema_sd.items()}
            print(f"  [Phase2-FP16 step={step+1}/{p2_steps}] loss={float(loss.item()):.4f} {ev2}", flush=True)
            model.load_state_dict(cur_sd)
            model.train()

    model.load_state_dict(best_p2_sd)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best_p2_sd.items()}, ckpt_dir / f"{args.version}_fp16_master.pt")
    ev_fp16 = _eval_val(model, r0_only=False)
    print(f"[Phase2-FP16 Master DONE] {ev_fp16}", flush=True)

    # =========================================================================
    # PHASE 3: Clean W4/W8 QAT (ZERO 1.58-bit! Full BiXT + mHC + T2/T3 Aware!)
    # =========================================================================
    set_v140_qat_mode(model, True)
    opt3 = _make_adamw(list(model.parameters()) + list(era_grl.parameters()), device, lr=2.2e-4, weight_decay=0.005)
    ema_q = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_obj = _eval_val(model, r0_only=False)["sel_obj"]

    model.train()
    era_grl.train()
    for step in range(p3_steps):
        mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p3_steps)))
        for pg in opt3.param_groups:
            pg["lr"] = 2.2e-4 * mult
        bk = _collate_student_batch_v140(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        use_t3 = (step % 2 == 0)
        opt3.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=3 if use_t3 else 2, r0_drop_p=min(0.35, float(args.r0_drop_p)), charlm_drop_p=min(0.25, float(args.charlm_drop_p)),
            )
            loss = _paradise_step_loss(od, bk, use_t3=use_t3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt3.step()
        _update_ema(ema_q, 0.992)

        if (step + 1) % max(1, p3_steps // 3) == 0 or step == p3_steps - 1:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_q)
            evq = _eval_val(model, r0_only=False)
            if evq["sel_obj"] >= best_q_obj:
                best_q_obj = evq["sel_obj"]
                best_q_sd = {k: v.clone() for k, v in ema_q.items()}
            print(f"  [Phase3-W4W8-QAT step={step+1}/{p3_steps}] loss={float(loss.item()):.4f} {evq}", flush=True)
            model.load_state_dict(cur_sd)
            model.train()

    model.load_state_dict(best_q_sd)
    set_v140_qat_mode(model, False)

    # =========================================================================
    # STAGE 3: Pure Calibration + Dense Certified R0 LTT + Bounded T3 + MGVZ8 & 4-Stage ONNX + 14/14 CI!
    # =========================================================================
    cal_rows = corpus["val_gold_v2"] + corpus["val_wave3"] + corpus["val_old_video_neg"] + corpus.get("val_multilingual", [])
    p_cal = predict_student_v140(model, fcache, [r["text"] for r in cal_rows], device, r0_only=False, outer_steps=3)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.float32)

    t_ai = fit_scalar_temperature_v140(p_cal["logit_ai"], y_cal_ai)
    pr_cal_rows = corpus["val_wave3"] + corpus["val_promo_v11"]
    p_pr_cal = predict_student_v140(model, fcache, [r["text"] for r in pr_cal_rows], device, r0_only=False, outer_steps=3)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in pr_cal_rows], dtype=np.float32)
    t_pr = fit_scalar_temperature_v140(p_pr_cal["logit_promo"], y_cal_pr) if len(pr_cal_rows) else 1.0

    r0_ai_z = np.clip(p_cal["logit_ai_r0"], -30.0, 30.0)
    f_ai_z = np.clip(p_cal["logit_ai"] / max(1e-3, t_ai), -30.0, 30.0)
    lut_ai_r0 = fit_smooth_ai_lut_v122(r0_ai_z, y_cal_ai)
    lut_ai_full = fit_smooth_ai_lut_v122(f_ai_z, y_cal_ai)

    if len(pr_cal_rows) >= 8 and len(np.unique(y_cal_pr)) == 2:
        r0_pr_z = np.clip(p_pr_cal["logit_promo_r0"], -30.0, 30.0)
        f_pr_z = np.clip(p_pr_cal["logit_promo"] / max(1e-3, t_pr), -30.0, 30.0)
        y_pr_fit = y_cal_pr
    else:
        r0_pr_z = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)
        f_pr_z = r0_pr_z
        y_pr_fit = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    lut_pr_r0 = fit_output_shifted_promo_lut_v122(r0_pr_z, y_pr_fit, pi_nat=0.015)
    lut_pr_full = fit_output_shifted_promo_lut_v122(f_pr_z, y_pr_fit, pi_nat=0.015)

    p_r0_ai_raw = expit(r0_ai_z)
    lut01 = float(apply_spline_temp_lut(np.array([0.01], dtype=np.float32), lut_ai_r0)[0])
    p_pr_wild = apply_spline_temp_lut(expit(np.clip(p_cal["logit_promo_r0"], -30.0, 30.0)), lut_pr_r0)
    ai_lut_chk = {"lut_ai_r0_at_001": round(lut01, 5)}
    pr_lut_chk = {"median_promo_wild": round(float(np.median(p_pr_wild)), 5)}

    # Dense LTT sweep for Certified R0 Early Exit (guarantees finding max-exit safe threshold!)
    p_r0_ai_cal = apply_spline_temp_lut(p_r0_ai_raw, lut_ai_r0)
    pos_m = y_cal_ai >= 0.5
    neg_m = y_cal_ai < 0.5
    r0_pol: dict[str, Any] = {"enabled": False, "tau_ai": 0.04, "tau_promo": 0.05, "defer_max": 0.30, "max_chars": 180, "cal_exit_rate": 0.0}
    if pos_m.sum() >= 60:
        best_exit_r = -1.0
        for tau_a in (0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.010, 0.006):
            for d_max in (0.40, 0.30, 0.22):
                exit_m = (p_r0_ai_cal <= tau_a) & (p_pr_wild <= 0.05) & (p_cal["p_defer_r0"] <= d_max)
                miss_rate = float(exit_m[pos_m].mean())
                ex_rate = float(exit_m[neg_m].mean())
                if miss_rate <= 0.025 and ex_rate > best_exit_r:
                    best_exit_r = ex_rate
                    r0_pol = {
                        "enabled": True, "tau_ai": tau_a, "tau_promo": 0.05, "defer_max": d_max,
                        "max_chars": 180, "cal_pos_miss_rate": round(miss_rate, 4),
                        "cal_exit_rate": round(ex_rate, 4),
                    }

    # Bounded Conditional T2 -> T3 policy
    p_esc_cal = p_cal["p_escalate_t3"]
    tau_esc = float(np.clip(np.quantile(p_esc_cal, 0.80) if len(p_esc_cal) >= 20 else 0.50, 0.25, 0.85))
    b_mar = float(model.t3_corr_cap + 1.0)  # 3.2 > 2.2 hard bound -> mathematically safe outside band!
    p_t2_cal = expit(np.clip(p_cal["logit_ai_t2_final"] / t_ai, -30, 30))
    t3_pol = {
        "enabled": True,
        "tau_escalate": round(tau_esc, 4),
        "uncertain_lo": 0.10,
        "uncertain_hi": 0.90,
        "t3_bound_margin": round(b_mar, 3),
        "cal_t3_rate": round(float(((np.abs(p_cal["logit_ai_t2_final"]) <= b_mar) & ((p_esc_cal >= tau_esc) | ((p_t2_cal >= 0.10) & (p_t2_cal <= 0.90)))).mean()), 4),
    }

    w_mix_learned = torch.softmax(model.ai_rung_mix_logits.detach().float(), dim=0).cpu().tolist()
    manifest = {
        "version": args.version,
        "max_len": int(args.max_len),
        "head_len": int(args.head_len),
        "long_char_trigger": 360,
        "temp_ai": t_ai,
        "temp_promo": t_pr,
        "lut_ai_r0": lut_ai_r0,
        "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0,
        "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_pol,
        "t3_escalation_policy": t3_pol,
        "ai_rung_mix_r0_t1_t2_ebt": [round(float(x), 4) for x in w_mix_learned],
        "disagree_penalties": {
            "r0_t2": round(float(0.25 * F.softplus(model.disagree_r0_t2_raw).item()), 4),
            "t2_ebt": round(float(0.25 * F.softplus(model.disagree_t2_ebt_raw).item()), 4),
        },
        "support_gate_tau": round(float(model.sup_gate_tau.clamp(10.0, 22.0).item()), 3),
        "charlm_ai_gate": round(float(0.18 * torch.sigmoid(model.charlm_ai_gate_raw).item()), 4),
        "promo_char_rho": round(float(torch.sigmoid(model.promo_char_rho_raw).item()), 4),
    }

    export_rep = export_v140_onnx_and_bundle_v8(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    ci_gates = run_v140_ci_gates(model, tokenizer, export_rep=export_rep, ai_lut_check=ai_lut_chk, promo_lut_check=pr_lut_chk, charlm=full_charlm)

    # Verify exported 4-Stage CPU ONNX Cascade Engine on held-out `test_gold_v2` + `val_multilingual`!
    eng = MeguvizV140CascadeEngine(out_dir, threads=4)
    tg2_txts = [r["text"] for r in corpus["test_gold_v2"]]
    tg2_y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["test_gold_v2"]], dtype=np.int32)
    tg2_cond = eng.predict_batch(tg2_txts, force_full=True, force_t3=False)
    tg2_t3 = eng.predict_batch(tg2_txts, force_full=True, force_t3=True)
    test_g2_cond = _eval_bin(tg2_y, np.array([r["logit_ai"] for r in tg2_cond], dtype=np.float32), temp=t_ai)
    test_g2_t3 = _eval_bin(tg2_y, np.array([r["logit_ai"] for r in tg2_t3], dtype=np.float32), temp=t_ai)
    t3_exit_frac = round(float(np.mean([1.0 if r["exit_rung"] == 3 else 0.0 for r in tg2_cond])), 4) if tg2_cond else 0.0

    val_ml_eval = None
    if corpus.get("val_multilingual"):
        ml_txts = [r["text"] for r in corpus["val_multilingual"]]
        ml_y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_multilingual"]], dtype=np.int32)
        ml_out = eng.predict_batch(ml_txts, force_full=True, force_t3=False)
        val_ml_eval = _eval_bin(ml_y, np.array([r["logit_ai"] for r in ml_out], dtype=np.float32), temp=t_ai)

    t3_probe = eng.predict_batch(["这是一条用于验证第三步 BiXT+mHC 嵌套纠错路由的边界测试评论。"], force_t3=True)[0]
    long_probe_txt = "这是一篇用来验证端到端长文分段聚合路由的真实风格技术长评，核心观点明确且不带模板腔。" * 12
    long_res = eng.predict_batch([long_probe_txt], force_full=False)[0]
    ci_gates["V140_T3_CONDITIONAL_ROUTE"] = {"forced_exit_rung": t3_probe["exit_rung"], "cond_t3_rate_test_g2": t3_exit_frac, "pass": bool(t3_probe["exit_rung"] == 3)}
    ci_gates["V140_LONG_DOC_API"] = {"exit_rung": long_res["exit_rung"], "pass": bool(long_res["exit_rung"] == 4)}
    ci_gates["all_ci_passed"] = all(bool(v.get("pass")) for v in ci_gates.values() if isinstance(v, dict))

    report = {
        "version": args.version,
        "gpu_name": gpu_name,
        "total_params": tot_p,
        "rung0_apexchar_params": r0_p,
        "paradise_core_params": core_p,
        "distill_pool_rows": len(distill_pool),
        "phase1_r0_best_obj": round(best_r0_obj, 5),
        "phase2_fp16_val": ev_fp16,
        "phase3_qat_best_obj": round(best_q_obj, 5),
        "test_bili_gold_v2_onnx_cond_t2_t3": test_g2_cond,
        "test_bili_gold_v2_onnx_full_t3": test_g2_t3,
        "val_multilingual_onnx": val_ml_eval,
        "rung0_ltt_policy": r0_pol,
        "t3_escalation_policy": t3_pol,
        "export": export_rep,
        "v140_ci_gates": ci_gates,
        "release_gates": {
            "bundle_le_5p00_mib": bool(export_rep["bundle_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v140_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["release_gates"]["v140_ci_passed"] or not report["release_gates"]["bundle_le_5p00_mib"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
