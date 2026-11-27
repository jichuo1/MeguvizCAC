"""MeguvizCAC v1.3.0-XtremeApex Student Trainer (Full Nested-Loop T2/T3 Error-Correction + Clean W4/W8 QAT).

Consumes the 3x24L Chinese Super-Teacher 200K soft distillation pool (`v130_superteacher_soft_distill_pool.jsonl.gz`)
produced by `train_v130_superteacher.py` and trains `MeguvizV130XtremeApexModel` (`<= 5.00 MiB` MGVZ8, ZERO 1.58-bit):
  - Phase 1: Full `ApexCharNano` W8 Rung-0 Dedicated Super-Teacher KD (`[4096,64]+[16384,32]+[8192,16]`, 35% punct jitter)
  - Phase 2: Full `H=384, 16-Slot` Nested-Loop (`k_inner=2 x outer_steps in {2,3}`) Joint KD with:
      * Explicit `3.5x` Error-Focused `T3` Correction Loss on `T2` mistakes + Monotonic Error-Shrink Margin
      * Ouro Conditional `T2 -> T3` Halting Gate (`escalate_t3_logit`) Utility Supervision
      * Relaxed Recursive LoRA (`rank=16`) + 3-Role History Delta + Active Text K/V & Haar Reread + EBT6 + `DynaSeam`
  - Phase 3: Clean W4/W8 QAT (ZERO 1.58-bit loop-error compounding!) + Pure Calibration + LTT + `MGVZ8`
             & 4-Stage Modular FP32-Exact ONNX Export (`R0`, `T2`, `T3`, `DynaSeam`) + 12 CI Gates!
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
from .v123_data_refinery import NUM_ERAS_V122, jitter_ai_punctuation_v123, load_sibling_and_delta_v123
from .v130_superteacher import (
    build_v130_superteacher_corpus,
    compute_top_tail_pairwise_loss_v130,
    fit_scalar_temperature_v130,
    read_jsonl_any_v130,
    text_sha16_v130,
)
from .v130_xtreme_apex import (
    AI_END_V130,
    ENGRAM_BUCKETS_V130,
    HEAD_LEN_V130,
    MAX_LEN_V130,
    TRIGRAM_BUCKETS_V130,
    MeguvizV130CascadeEngine,
    MeguvizV130XtremeApexModel,
    encode_bpe_head_tail_v130,
    export_v130_onnx_and_bundle_v8,
    extract_apex_char_batch_v130,
    run_v130_ci_gates,
    set_v130_qat_mode,
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


def _load_distill_pool_v130(path: Path | None, blocked_hashes: set[str], smoke_test: bool) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = read_jsonl_any_v130(path)
    if smoke_test:
        rows = rows[:400]
    out: list[dict[str, Any]] = []
    for r in rows:
        txt = str(r.get("text", "")).strip()
        if len(txt) < 2:
            continue
        if text_sha16_v130(txt) in blocked_hashes:
            continue
        t_ai_lg = float(r.get("t_ai_logit", 0.0))
        t_pr_lg = float(r.get("t_pr_logit", -3.0))
        t_std = float(r.get("t_ai_std_p", 0.0))
        kd_w = float(1.0 / (1.0 + 2.0 * max(0.0, t_std)))
        g_ai = r.get("gold_ai")
        g_pr = r.get("gold_pr")
        has_gold_ai = 1.0 if (g_ai is not None and (float(g_ai) <= 25.0 or float(g_ai) >= 70.0)) else 0.0
        has_gold_pr = 1.0 if (g_pr is not None and (float(g_pr) <= 25.0 or float(g_pr) >= 70.0)) else 0.0
        e5 = int(r.get("era5", -1))
        src = str(r.get("source_tag", ""))
        is_hard_neg = bool(
            (t_ai_lg < -0.5 or (has_gold_ai > 0.5 and float(g_ai) <= 25.0))
            and (e5 in (0, 1, 2) or "oldvideo" in src.lower() or "organic_brand" in src.lower() or "trap" in src.lower() or (has_gold_pr > 0.5 and float(g_pr) >= 70.0))
        )
        out.append({
            "text": txt,
            "t_ai_logit": t_ai_lg,
            "t_pr_logit": t_pr_lg,
            "has_teacher": 1.0,
            "kd_w": kd_w,
            "y_ai": 1.0 if (g_ai is not None and float(g_ai) >= 70.0) else (0.0 if has_gold_ai > 0.5 else float(expit(np.clip(t_ai_lg, -20.0, 20.0)))),
            "has_ai": has_gold_ai,
            "y_pr": 1.0 if (g_pr is not None and float(g_pr) >= 70.0) else (0.0 if has_gold_pr > 0.5 else float(expit(np.clip(t_pr_lg, -20.0, 20.0)))),
            "has_pr": has_gold_pr,
            "sub4": -1,
            "hard_neg": 1.0 if is_hard_neg else 0.0,
            "era5": e5,
            "source_tag": src,
        })
    return out


def _collate_student_batch(
    batch: list[dict[str, Any]],
    tokenizer: BiliPickyBPETokenizer,
    device: torch.device,
    max_len: int,
    head_len: int,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.35,
    need_bpe: bool = True,
) -> dict[str, torch.Tensor]:
    txts: list[str] = []
    for r in batch:
        t = str(r["text"])
        is_ai_pos = (r.get("has_ai", 0.0) > 0.5 and r.get("y_ai", 0.0) >= 0.70) or (r.get("has_teacher", 0.0) > 0.5 and r.get("t_ai_logit", -5.0) >= 1.2)
        if rng is not None and punct_jitter_prob > 0.0 and is_ai_pos:
            t = jitter_ai_punctuation_v123(t, rng, prob=punct_jitter_prob)
        txts.append(t)

    c1, c2, c3, cm, cf, _ = extract_apex_char_batch_v130(txts)
    out: dict[str, torch.Tensor] = {
        "c1": torch.from_numpy(c1).to(device, non_blocking=True),
        "c2": torch.from_numpy(c2).to(device, non_blocking=True),
        "c3": torch.from_numpy(c3).to(device, non_blocking=True),
        "cm": torch.from_numpy(cm).to(device, non_blocking=True),
        "cf": torch.from_numpy(cf).to(device, non_blocking=True),
        "t_ai_lg": torch.tensor([float(r.get("t_ai_logit", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "t_pr_lg": torch.tensor([float(r.get("t_pr_logit", 0.0)) for r in batch], dtype=torch.float32, device=device),
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
        cleans, f56_l = [], []
        for t in txts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56_l.append(f56)
        encs = [encode_bpe_head_tail_v130(tokenizer, c, max_len=max_len, head_len=head_len) for c in cleans]
        engs = [make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130) for s in encs]
        trigs = [make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130) for s in encs]
        ml = max(2, max(len(s) for s in encs))
        if ml % 2 == 1:
            ml += 1
        nb = len(txts)
        ids = np.full((nb, ml), tokenizer.pad_id, dtype=np.int64)
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
def predict_student_v130(
    model: MeguvizV130XtremeApexModel,
    tokenizer: BiliPickyBPETokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    max_len: int = MAX_LEN_V130,
    head_len: int = HEAD_LEN_V130,
    r0_only: bool = False,
    outer_steps: int = 3,
) -> dict[str, np.ndarray]:
    n = len(texts)
    if r0_only:
        keys = ("logit_ai_r0", "logit_promo_r0", "p_defer_r0")
    elif outer_steps >= 3:
        keys = (
            "logit_ai_r0", "logit_promo_r0", "p_defer_r0", "logit_ai_r1", "logit_promo_r1",
            "logit_ai_t1", "logit_ai_t2_final", "logit_promo_t2_final", "p_escalate_t3",
            "logit_ai", "logit_promo", "ebt_margin",
        )
    else:
        keys = (
            "logit_ai_r0", "logit_promo_r0", "p_defer_r0", "logit_ai_r1", "logit_promo_r1",
            "logit_ai_t1", "logit_ai_t2_final", "logit_promo_t2_final", "p_escalate_t3",
            "logit_ai", "logit_promo", "ebt_margin",
        )
    out = {k: np.zeros((n,), dtype=np.float32) for k in keys}
    if n == 0:
        return out
    model.eval()
    order = sorted(range(n), key=lambda i: len(texts[i]))
    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_b = [{"text": texts[i]} for i in idxs]
        bk = _collate_student_batch(sub_b, tokenizer, device, max_len, head_len, rng=None, punct_jitter_prob=0.0, need_bpe=not r0_only)
        if r0_only:
            res = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        else:
            res = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], outer_steps=outer_steps,
            )
        for k in keys:
            arr = res[k].float().cpu().numpy()
            for j, oi in enumerate(idxs):
                out[k][oi] = arr[j]
    return out


def compute_kd_plus_gold_loss_v130(
    lg: torch.Tensor,
    bk: dict[str, torch.Tensor],
    axis: str = "ai",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Super-Teacher soft-logit KD (Huber + soft BCE) + Gold BCE + Top-Tail Pairwise Margin + optional T3 error-focus weights."""
    if axis == "ai":
        t_lg, y_g, has_g, hn = bk["t_ai_lg"], bk["y_ai"], bk["has_ai"], bk["hard_neg"]
    else:
        t_lg, y_g, has_g, hn = bk["t_pr_lg"], bk["y_pr"], bk["has_pr"], torch.zeros_like(bk["hard_neg"])

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
        w_g = (1.0 + 0.45 * hn[gm]) * sw[gm]
        l_g = (F.binary_cross_entropy_with_logits(lg_f[gm], yg_s, reduction="none") * w_g).mean()
        loss = loss + 0.65 * l_g

    eff_y = torch.where(gm, y_g, torch.sigmoid(t_lg))
    eff_v = ((gm) | (tm & ((t_lg >= 1.5) | (t_lg <= -1.5)))).float()
    l_tail = compute_top_tail_pairwise_loss_v130(lg_f, eff_y, eff_v, hard_neg_boost=hn if axis == "ai" else None, margin=1.7, top_neg_frac=0.25)
    loss = loss + 0.30 * l_tail

    if axis == "ai":
        hnm = (eff_v > 0.5) & (eff_y <= 0.20) & (hn > 0.5)
        if hnm.any():
            loss = loss + 0.20 * (F.relu(lg_f[hnm] + 0.6).pow(2) * sw[hnm]).mean()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MeguvizCAC v1.3.0-XtremeApex Full Nested-Loop Student (<=5.00 MiB)")
    parser.add_argument("--distill-pool", type=Path, default=None, help="Path to v130_superteacher_soft_distill_pool.jsonl.gz")
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v07-tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--version", type=str, default="v1.3.0-r1-xtreme-apex")
    parser.add_argument("--p1-steps", type=int, default=2200)
    parser.add_argument("--p2-steps", type=int, default=3200)
    parser.add_argument("--p3-qat-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=MAX_LEN_V130)
    parser.add_argument("--head-len", type=int, default=HEAD_LEN_V130)
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
    rng = random.Random(20261129)
    torch.manual_seed(20261129)
    np.random.seed(20261129)

    print(f"[v1.3.0-NestedXtremeApex] GPU={gpu_name} | version={args.version} | distill={args.distill_pool} | smoke={args.smoke_test}", flush=True)

    # 1. Load Purified Real Gold + Validation + Super-Teacher 200K Soft-Logit Pool + Counterfactual Pairs
    corpus = build_v130_superteacher_corpus(
        sibling_dir=args.sibling_dir, delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, v04_data_path=None, smoke_test=args.smoke_test,
    )
    blocked = {text_sha16_v130(r["text"]) for k in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "diag_god500") for r in corpus.get(k, [])}
    distill_pool = _load_distill_pool_v130(args.distill_pool, blocked, smoke_test=args.smoke_test)

    d_pos = [r for r in distill_pool if r["t_ai_logit"] >= 0.8]
    d_bnd = [r for r in distill_pool if -1.2 < r["t_ai_logit"] < 0.8]
    d_pr = [r for r in distill_pool if r["t_pr_logit"] >= 0.5]
    d_hn = [r for r in distill_pool if r["hard_neg"] > 0.5]
    d_nat = [r for r in distill_pool if r["t_ai_logit"] <= -1.2 and r["hard_neg"] < 0.5]

    sib_pack = load_sibling_and_delta_v123(args.sibling_dir, args.delta_dir, smoke_test=args.smoke_test)
    cf_pairs = sib_pack["cf_mirror_pairs"]
    longform_train = read_jsonl_any_v130(args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 1200)]

    print(
        f"[Stage0-Data] gold_ai+={len(corpus['train_ai_pos']):,} hu_nat={len(corpus['train_hu_natural']):,} hu_hd={len(corpus['train_hu_hard']):,} | "
        f"distill_pool={len(distill_pool):,} (pos={len(d_pos):,}, bnd={len(d_bnd):,}, pr+={len(d_pr):,}, hn={len(d_hn):,}, nat={len(d_nat):,}) | "
        f"cf_pairs={len(cf_pairs):,} longform={len(longform_train):,}",
        flush=True,
    )

    gold_pos = corpus["train_ai_pos"]
    gold_nat = corpus["train_hu_natural"]
    gold_hd = corpus["train_hu_hard"]
    gold_pr = corpus["train_promo_extra"]

    def _draw(pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rng.choices(pool, k=k) if (pool and k > 0) else []

    def _sample_batch(bs_cur: int) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        if distill_pool:
            out_b.extend(_draw(d_pos, max(1, int(bs_cur * 0.14))))
            out_b.extend(_draw(d_bnd, max(1, int(bs_cur * 0.12))))
            out_b.extend(_draw(d_pr, max(1, int(bs_cur * 0.06))))
            out_b.extend(_draw(d_hn, max(1, int(bs_cur * 0.12))))
            out_b.extend(_draw(d_nat, max(1, int(bs_cur * 0.16))))
        rem = max(4, bs_cur - len(out_b))
        out_b.extend(_draw(gold_pos, max(1, int(rem * 0.24))))
        out_b.extend(_draw(gold_hd, max(1, int(rem * 0.28))))
        out_b.extend(_draw(gold_nat, max(1, int(rem * 0.38))))
        out_b.extend(_draw(gold_pr, max(0, rem - len(out_b) + (bs_cur - rem))))
        fb = gold_nat or gold_pos or distill_pool
        while len(out_b) < bs_cur and fb:
            out_b.append(rng.choice(fb))
        return out_b[:bs_cur]

    # 2. Model Init + Parameter Audit
    model = MeguvizV130XtremeApexModel(vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_END_V130, num_eras=NUM_ERAS_V122).to(device)
    r0_p = sum(p.numel() for p in model.rung0.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    tot_p = sum(p.numel() for p in model.parameters())
    print(f"[Stage0-Model] total_params={tot_p:,} | rung0={r0_p:,} ({r0_p/tot_p*100:.1f}%) | nested_core={core_p:,} ({core_p/tot_p*100:.1f}%)", flush=True)

    # Validation helper (reports BOTH T2 and T3 + Delta(T3 - T2) strictly on val_* splits!)
    def _eval_val(m_cur: MeguvizV130XtremeApexModel, r0_only: bool = False) -> dict[str, Any]:
        p_g2 = predict_student_v130(m_cur, tokenizer, [r["text"] for r in corpus["val_gold_v2"]], device, r0_only=r0_only, outer_steps=3)
        p_w3 = predict_student_v130(m_cur, tokenizer, [r["text"] for r in corpus["val_wave3"]], device, r0_only=r0_only, outer_steps=3)
        p_old = predict_student_v130(m_cur, tokenizer, [r["text"] for r in corpus["val_old_video_neg"]], device, r0_only=r0_only, outer_steps=3)

        yg2 = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
        yw3_a = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        yw3_p = np.array([int(r["y_pr"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)

        key_ai = "logit_ai_r0" if r0_only else "logit_ai"
        key_pr = "logit_promo_r0" if r0_only else "logit_promo"
        mg2 = _eval_bin(yg2, p_g2[key_ai])
        mw3_a = _eval_bin(yw3_a, p_w3[key_ai])
        mw3_p = _eval_bin(yw3_p, p_w3[key_pr])
        q4 = (yw3_p == 1) & (yw3_a == 0)
        q4_fpr = float((expit(np.clip(p_w3[key_ai][q4], -30, 30)) >= 0.5).mean()) if q4.sum() else 0.0
        old_fpr = float((expit(np.clip(p_old[key_ai], -30, 30)) >= 0.5).mean()) if len(p_old[key_ai]) else 0.0
        obj = 0.35 * mg2["pr"] + 0.25 * mg2["r95"] + 0.20 * mw3_a["pr"] + 0.20 * mw3_p["pr"] - 1.2 * old_fpr - 0.5 * q4_fpr
        res = {
            "sel_obj": round(float(obj), 5), "g2_pr": mg2["pr"], "g2_r95": mg2["r95"],
            "w3_ai_pr": mw3_a["pr"], "w3_pr_pr": mw3_p["pr"],
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
    # PHASE 1: Dedicated Full `ApexCharNano` W8 Rung-0 Super-Teacher KD!
    # =========================================================================
    opt1 = torch.optim.AdamW(model.rung0.parameters(), lr=1.8e-3, weight_decay=0.01)
    best_r0_sd = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
    best_r0_obj = -1e9
    model.train()
    for step in range(p1_steps):
        mult = min(1.0, (step + 1) / max(1, int(p1_steps * 0.08))) * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p1_steps))))
        for pg in opt1.param_groups:
            pg["lr"] = 1.8e-3 * mult
        bk = _collate_student_batch(_sample_batch(bs), tokenizer, device, args.max_len, args.head_len, rng=rng, punct_jitter_prob=0.35, need_bpe=False)
        opt1.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
            l_ai = compute_kd_plus_gold_loss_v130(r0["logit_ai_r0"], bk, axis="ai")
            l_pr = compute_kd_plus_gold_loss_v130(r0["logit_promo_r0"], bk, axis="promo")
            with torch.no_grad():
                t_p = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                p_r0 = torch.sigmoid(r0["logit_ai_r0"].float())
                err_tgt = ((p_r0 - t_p).abs() >= 0.22).float()
            l_def = F.binary_cross_entropy_with_logits(r0["defer_r0_logit"].float(), err_tgt)
            loss = l_ai + 0.70 * l_pr + 0.15 * l_def
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
    # PHASE 2: Full Nested-Loop (`k_inner=2 x {T2, T3}`) Progressive Error-Correction KD!
    # =========================================================================
    r0_params = list(model.rung0.parameters())
    r0_ids = {id(p) for p in r0_params}
    deep_params = [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters())
    opt2 = torch.optim.AdamW([
        {"params": r0_params, "lr": 3.0e-4, "base_lr": 3.0e-4, "weight_decay": 0.01},
        {"params": deep_params, "lr": 1.0e-3, "base_lr": 1.0e-3, "weight_decay": 0.01},
    ])
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

    def _nested_step_loss(od: dict[str, torch.Tensor], bk: dict[str, torch.Tensor], use_t3: bool) -> torch.Tensor:
        # Base supervision across R0, R1, T1, T2
        l_ai_t2f = compute_kd_plus_gold_loss_v130(od["logit_ai_t2_final"], bk, axis="ai")
        l_ai_t2r = compute_kd_plus_gold_loss_v130(od["logit_ai_t2"], bk, axis="ai")
        l_ai_t1 = compute_kd_plus_gold_loss_v130(od["logit_ai_t1"], bk, axis="ai")
        l_ai_r1 = compute_kd_plus_gold_loss_v130(od["logit_ai_r1"], bk, axis="ai")
        l_ai_r0 = compute_kd_plus_gold_loss_v130(od["logit_ai_r0"], bk, axis="ai")

        l_pr_t2f = compute_kd_plus_gold_loss_v130(od["logit_promo_t2_final"], bk, axis="promo")
        l_pr_t2r = compute_kd_plus_gold_loss_v130(od["logit_promo_t2"], bk, axis="promo")
        l_pr_r1 = compute_kd_plus_gold_loss_v130(od["logit_promo_r1"], bk, axis="promo")
        l_pr_r0 = compute_kd_plus_gold_loss_v130(od["logit_promo_r0"], bk, axis="promo")

        l_ebt_ai = compute_kd_plus_gold_loss_v130(od["ebt_margin"], bk, axis="ai")
        l_ebt_pr = compute_kd_plus_gold_loss_v130(od["ebt_promo_margin"], bk, axis="promo")

        # T3 Error-Focused Progressive Correction + Ouro Conditional Halting Utility Supervision!
        l_t3_total = l_ai_t2f * 0.0
        if use_t3:
            with torch.no_grad():
                t_p_ai = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                p_t2 = torch.sigmoid(od["logit_ai_t2_final"].float())
                err_t2 = (p_t2 - t_p_ai).abs()
                # Focus 3.5x gradient power on T2 mistakes & boundary cases so T3 actively corrects T2!
                t3_focus_w = 1.0 + 2.5 * (err_t2 >= 0.15).float()

            l_ai_t3f = compute_kd_plus_gold_loss_v130(od["logit_ai_t3_final"], bk, axis="ai", sample_weight=t3_focus_w)
            l_ai_t3r = compute_kd_plus_gold_loss_v130(od["logit_ai_t3"], bk, axis="ai", sample_weight=t3_focus_w)
            l_pr_t3f = compute_kd_plus_gold_loss_v130(od["logit_promo_t3_final"], bk, axis="promo")

            # Monotonic error-shrink margin: when T2 has error >= 0.15, T3 must strictly shrink error!
            p_t3 = torch.sigmoid(od["logit_ai_t3_final"].float())
            err_t3 = (p_t3 - t_p_ai).abs()
            hard_m = (err_t2 >= 0.15) & ((bk["has_ai"] > 0.5) | (bk["has_t"] > 0.5))
            l_mono = F.relu(err_t3[hard_m] - err_t2[hard_m] + 0.02).mean() if hard_m.any() else l_ai_t2f * 0.0

            # Ouro Utility Halting Gate (`escalate_t3_logit`): fire when T3 improves over T2 OR sample is uncertain!
            with torch.no_grad():
                u_gain = err_t2 - err_t3.detach()
                y_esc = ((u_gain >= 0.025) | ((p_t2 >= 0.12) & (p_t2 <= 0.88) & (err_t2 >= 0.18))).float()
            l_halt = F.binary_cross_entropy_with_logits(od["escalate_t3_logit"].float(), y_esc)

            l_t3_total = 1.00 * l_ai_t3f + 0.30 * l_ai_t3r + 0.65 * l_pr_t3f + 0.35 * l_mono + 0.20 * l_halt

        # R1 deferral head (autocast-safe BCEWithLogits!)
        with torch.no_grad():
            t_p_ai_r1 = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
            err_r1 = (torch.sigmoid(od["logit_ai_r1"].float()) - t_p_ai_r1).abs()
            y_def_r1 = (err_r1 >= 0.20).float()
        l_def_r1 = F.binary_cross_entropy_with_logits(od["defer_r1_logit"].float(), y_def_r1)

        s4_m = (bk["has_pr"] > 0.5) & (bk["sub4"] >= 0) & (bk["sub4"] < 4)
        l_s4 = F.cross_entropy(od["promo_sub4_logit"].float()[s4_m], bk["sub4"][s4_m]) if s4_m.any() else l_ai_t2f * 0.0

        hu_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
        era_v = hu_m & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
        l_era = era_grl(od["emb_ai"], bk["era5"], era_v, lam=0.50) if int(era_v.sum().item()) >= 8 else l_ai_t2f * 0.0

        return (
            (0.75 if use_t3 else 1.00) * l_ai_t2f + 0.25 * l_ai_t2r + 0.20 * l_ai_t1 + 0.18 * l_ai_r1 + 0.25 * l_ai_r0
            + (0.45 if use_t3 else 0.70) * l_pr_t2f + 0.20 * l_pr_t2r + 0.12 * l_pr_r1 + 0.20 * l_pr_r0
            + l_t3_total
            + 0.20 * l_ebt_ai + 0.12 * l_ebt_pr + 0.10 * l_def_r1 + 0.10 * l_s4 + 0.12 * l_era
        )

    model.train()
    era_grl.train()
    for step in range(p2_steps):
        mult = min(1.0, (step + 1) / max(1, int(p2_steps * 0.08))) * (0.06 + 0.94 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p2_steps))))
        for pg in opt2.param_groups:
            pg["lr"] = pg["base_lr"] * mult
        bk = _collate_student_batch(_sample_batch(bs), tokenizer, device, args.max_len, args.head_len, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        # Stochastic Depth: 50% `outer_steps=3` (true T3 active error correction) & 50% `outer_steps=2`!
        use_t3 = (step % 2 == 0)
        opt2.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], outer_steps=3 if use_t3 else 2,
            )
            loss = _nested_step_loss(od, bk, use_t3=use_t3)

            # Counterfactual Mirror Margin Loss (every 3rd step, with outer_steps=3!)
            if cf_pairs and (step % 3 == 0):
                n_cf = 4 if args.smoke_test else 16
                ch = rng.sample(cf_pairs, min(n_cf, len(cf_pairs)))
                bk_a = _collate_student_batch([{"text": x["ai_text"]} for x in ch], tokenizer, device, args.max_len, args.head_len, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                bk_h = _collate_student_batch([{"text": x["human_text"]} for x in ch], tokenizer, device, args.max_len, args.head_len, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                oa = model(bk_a["ids"], bk_a["msk"], bk_a["st"], bk_a["eg"], bk_a["tg"], bk_a["c1"], bk_a["c2"], bk_a["c3"], bk_a["cm"], bk_a["cf"], outer_steps=3)
                oh = model(bk_h["ids"], bk_h["msk"], bk_h["st"], bk_h["eg"], bk_h["tg"], bk_h["c1"], bk_h["c2"], bk_h["c3"], bk_h["cm"], bk_h["cf"], outer_steps=3)
                l_cf = (
                    F.softplus(1.6 - (oa["logit_ai"].float() - oh["logit_ai"].float())).mean()
                    + 0.5 * F.softplus(1.5 - (oa["logit_ai_t2_final"].float() - oh["logit_ai_t2_final"].float())).mean()
                    + 0.4 * F.softplus(1.4 - (oa["logit_ai_r0"].float() - oh["logit_ai_r0"].float())).mean()
                )
                loss = loss + 0.20 * l_cf

            # Native Long-Document DynaSeam Supervision (every 6th step)
            if longform_train and (step % 6 == 0):
                n_ld = 2 if args.smoke_test else 6
                docs = rng.sample(longform_train, min(n_ld, len(longform_train)))
                d_losses = []
                for d_it in docs:
                    chunks = plan_long_chunks_v11(str(d_it.get("text", "")), max_chunks=6)
                    if not chunks:
                        continue
                    bk_c = _collate_student_batch([{"text": c["text"]} for c in chunks], tokenizer, device, args.max_len, args.head_len, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oc = model(bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["tg"], bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"], outer_steps=2)
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
    # PHASE 3: Clean W4/W8 QAT (ZERO 1.58-bit Collapse! Full T2/T3 Loop-Aware!)
    # =========================================================================
    set_v130_qat_mode(model, True)
    opt3 = torch.optim.AdamW(list(model.parameters()) + list(era_grl.parameters()), lr=2.2e-4, weight_decay=0.005)
    ema_q = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_obj = _eval_val(model, r0_only=False)["sel_obj"]

    model.train()
    era_grl.train()
    for step in range(p3_steps):
        mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, p3_steps)))
        for pg in opt3.param_groups:
            pg["lr"] = 2.2e-4 * mult
        bk = _collate_student_batch(_sample_batch(bs), tokenizer, device, args.max_len, args.head_len, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        use_t3 = (step % 2 == 0)
        opt3.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], outer_steps=3 if use_t3 else 2,
            )
            loss = _nested_step_loss(od, bk, use_t3=use_t3)
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
    set_v130_qat_mode(model, False)

    # =========================================================================
    # STAGE 3: Pure Calibration + Certified R0 & Ouro T3 Policy + MGVZ8 & 4-Stage ONNX + 12 CI Gates!
    # =========================================================================
    cal_rows = corpus["val_gold_v2"] + corpus["val_wave3"] + corpus["val_old_video_neg"]
    p_cal = predict_student_v130(model, tokenizer, [r["text"] for r in cal_rows], device, r0_only=False, outer_steps=3)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.float32)

    t_ai = fit_scalar_temperature_v130(p_cal["logit_ai"], y_cal_ai)
    pr_cal_rows = corpus["val_wave3"] + corpus["val_promo_v11"]
    p_pr_cal = predict_student_v130(model, tokenizer, [r["text"] for r in pr_cal_rows], device, r0_only=False, outer_steps=3)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in pr_cal_rows], dtype=np.float32)
    t_pr = fit_scalar_temperature_v130(p_pr_cal["logit_promo"], y_cal_pr) if len(pr_cal_rows) else 1.0

    # Fit smooth monotone LUTs directly on logits (`x = expit(z)` inside export_beta_spline_as_v07_lut!)
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

    p_r0_ai_cal = apply_spline_temp_lut(p_r0_ai_raw, lut_ai_r0)
    pos_m = y_cal_ai >= 0.5
    neg_m = y_cal_ai < 0.5
    r0_pol = {"enabled": False, "tau_ai": 0.05, "tau_promo": 0.05, "defer_max": 0.25, "max_chars": 160, "cal_exit_rate": 0.0}
    if pos_m.sum() >= 100:
        for tau_a in (0.08, 0.06, 0.04, 0.025):
            exit_m = (p_r0_ai_cal <= tau_a) & (p_pr_wild <= 0.05) & (p_cal["p_defer_r0"] <= 0.30)
            miss_rate = float(exit_m[pos_m].mean())
            if miss_rate <= 0.01:
                r0_pol = {
                    "enabled": True, "tau_ai": tau_a, "tau_promo": 0.05, "defer_max": 0.30,
                    "max_chars": 160, "cal_pos_miss_rate": round(miss_rate, 4),
                    "cal_exit_rate": round(float(exit_m[neg_m].mean()), 4),
                }
                break

    # Ouro Conditional T2 -> T3 escalation policy (target ~15-20% hard-case T3 escalation on pure cal!)
    p_esc_cal = p_cal["p_escalate_t3"]
    tau_esc = float(np.quantile(p_esc_cal, 0.82)) if len(p_esc_cal) >= 20 else 0.50
    tau_esc = float(np.clip(tau_esc, 0.25, 0.85))
    t3_pol = {
        "enabled": True,
        "tau_escalate": round(tau_esc, 4),
        "uncertain_lo": 0.12,
        "uncertain_hi": 0.88,
        "cal_t3_rate": round(float(((p_esc_cal >= tau_esc) | ((expit(np.clip(p_cal["logit_ai_t2_final"] / t_ai, -30, 30)) >= 0.12) & (expit(np.clip(p_cal["logit_ai_t2_final"] / t_ai, -30, 30)) <= 0.88))).mean()), 4),
    }

    manifest = {
        "version": args.version,
        "max_len": int(args.max_len),
        "head_len": int(args.head_len),
        "long_char_trigger": 340,
        "temp_ai": t_ai,
        "temp_promo": t_pr,
        "lut_ai_r0": lut_ai_r0,
        "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0,
        "lut_promo_full": lut_pr_full,
        "rung0_ltt_policy": r0_pol,
        "t3_escalation_policy": t3_pol,
        "promo_char_rho": round(float(torch.sigmoid(model.promo_char_rho_raw).item()), 4),
        "optional_host_meta_adapter": None,
    }

    export_rep = export_v130_onnx_and_bundle_v8(model, args.v07_tokenizer, manifest, out_dir, args.version)
    ci_gates = run_v130_ci_gates(model, tokenizer, export_rep=export_rep, ai_lut_check=ai_lut_chk, promo_lut_check=pr_lut_chk)

    # Evaluate final exported 4-Stage CPU ONNX Cascade Engine on held-out `test_gold_v2` (both conditional & full T3!)
    eng = MeguvizV130CascadeEngine(out_dir, threads=4)
    tg2_txts = [r["text"] for r in corpus["test_gold_v2"]]
    tg2_y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["test_gold_v2"]], dtype=np.int32)
    tg2_cond = eng.predict_batch(tg2_txts, force_full=True, force_t3=False)
    tg2_t3 = eng.predict_batch(tg2_txts, force_full=True, force_t3=True)
    test_g2_cond = _eval_bin(tg2_y, np.array([r["logit_ai"] for r in tg2_cond], dtype=np.float32), temp=t_ai)
    test_g2_t3 = _eval_bin(tg2_y, np.array([r["logit_ai"] for r in tg2_t3], dtype=np.float32), temp=t_ai)
    t3_exit_frac = round(float(np.mean([1.0 if r["exit_rung"] == 3 else 0.0 for r in tg2_cond])), 4) if tg2_cond else 0.0

    # Verify T3 forced route (`exit_rung == 3`) AND native Long-Doc route (`exit_rung == 4`)
    t3_probe = eng.predict_batch(["这是一条用于验证第三步嵌套纠错路由的边界测试评论。"], force_t3=True)[0]
    long_probe_txt = "这是一篇用来验证端到端长文分段聚合路由的真实风格技术长评，核心观点明确且不带模板腔。" * 10
    long_res = eng.predict_batch([long_probe_txt], force_full=False)[0]
    ci_gates["V130_T3_CONDITIONAL_ROUTE"] = {"forced_exit_rung": t3_probe["exit_rung"], "cond_t3_rate_test_g2": t3_exit_frac, "pass": bool(t3_probe["exit_rung"] == 3)}
    ci_gates["V130_LONG_DOC_API"] = {"exit_rung": long_res["exit_rung"], "pass": bool(long_res["exit_rung"] == 4)}
    ci_gates["all_ci_passed"] = all(bool(v.get("pass")) for v in ci_gates.values() if isinstance(v, dict))

    report = {
        "version": args.version,
        "gpu_name": gpu_name,
        "total_params": tot_p,
        "rung0_apexchar_params": r0_p,
        "nested_core_params": core_p,
        "distill_pool_rows": len(distill_pool),
        "phase1_r0_best_obj": round(best_r0_obj, 5),
        "phase2_fp16_val": ev_fp16,
        "phase3_qat_best_obj": round(best_q_obj, 5),
        "test_bili_gold_v2_onnx_cond_t2_t3": test_g2_cond,
        "test_bili_gold_v2_onnx_full_t3": test_g2_t3,
        "rung0_ltt_policy": r0_pol,
        "t3_escalation_policy": t3_pol,
        "export": export_rep,
        "v130_ci_gates": ci_gates,
        "release_gates": {
            "bundle_le_5p00_mib": bool(export_rep["bundle_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v130_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
