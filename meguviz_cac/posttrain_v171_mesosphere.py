"""MeguvizCAC v1.7.1 Post-Training Suite (Frontier Miner -> Stage A PUC+ER-DRO-R -> Stage B Protected PNBR+J-Cert -> Stage C APD-v2 + 68B DS-TitanGuard + 8-Cand WiSE-FT + 26/26 CI).

Strictly non-destructive: imports v1.7.0 modules without modifying v1.7.0 or v1.6.x files.
Zero-Leakage: never accesses OjinjinBench.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .train_v170_mesosphere import (
    ExcessRiskGroupDROR,
    _StrictGroupOOFFeatureCacheV170,
    _collate_v170_cpu,
    _h2d_batch_v170,
    _precollate_eval_split_v170,
    predict_student_v170,
)


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 40, min_ratio: float = 0.05) -> float:
    if step <= warmup:
        return base_lr * max(0.05, step / max(1, warmup))
    t = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, t))))


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _make_adamw(param_groups: list[dict[str, Any]], device: torch.device) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), fused=(device.type == "cuda"))


def _make_synthetic_smoke_corpus(seed: int = 2026) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    hu_base = [
        "笑死我了，这个插件装完直接起飞[doge]",
        "节点参考Blender材质节点这个思路挺对，能白盒点开看子代理在干啥就很难得。",
        "观感评分：画面 9/10，配乐 8.5/10，节奏 7/10，整体算近期很稳的良心作了。",
        "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子",
    ]
    ai_base = [
        "综上所述，该方案在保持系统稳定性的同时显著提升了整体处理效率。",
        "要的不是一次点击，是整只手。给权限容易，收回来难。",
        "渲染省人眼不省token，卡片留在上下文里每轮重读，画得越漂亮账越贵",
        "It is not merely an incremental update — it is a nuanced testament to how modern tooling delves into systemic bottlenecks.",
    ]
    rows = []
    for i in range(64):
        txt = rng.choice(hu_base) if i % 2 == 0 else rng.choice(ai_base)
        y = 0.0 if i % 2 == 0 else 1.0
        rows.append({"text": f"{txt} #{i}", "y_ai": y, "has_ai": 1.0, "y_pr": 0.0, "has_pr": 1.0, "hard_neg": 1.0 if i % 4 == 0 else 0.0, "group_id": f"g{i%6}"})
    return {
        "train_extra": rows[:48],
        "val_gold_v2": rows[48:],
        "val_wave3": rows[48:],
        "val_old_video_neg": rows[48::2],
        "val_multilingual": rows[48:],
    }
from .v07_features import BiliPickyBPETokenizer
from .v122_apexnano import apply_spline_temp_lut, fit_output_shifted_promo_lut_v122, fit_smooth_ai_lut_v122
from .v130_xtreme_apex import (
    R0_B2_V130,
    R0_B3_V130,
    R0_HEAD_CHARS_V130,
    R0_MAX_CHARS_V130,
    set_v130_qat_mode,
)
from .v140_superteacher import (
    build_v140_superteacher_corpus,
    fit_scalar_temperature_v140,
    read_jsonl_any_v140,
    text_sha16_v140,
)
from .v150_heaven_loop import (
    CompactCharLMV150,
    canonicalize_zero_pii_v150,
    fit_group_oof_charlm_v150,
)


def _eval_binary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y_i = (np.asarray(y) >= 0.5).astype(np.int64)
    p_f = np.asarray(p, dtype=np.float64)
    pr = float(average_precision_score(y_i, p_f)) if y_i.sum() > 0 else 0.0
    order = np.argsort(-p_f, kind="mergesort")
    ys = y_i[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    mask = prec >= 0.95
    r95 = float(tp[mask].max() / max(1, int(y_i.sum()))) if np.any(mask) else 0.0
    return {"pr_auc": pr, "r_at_p95": r95}


def _load_v140_distill_pool(path: Path, blocked: set[str], smoke_test: bool = False) -> list[dict[str, Any]]:
    rows = read_jsonl_any_v140(path)
    out = [r for r in rows if r.get("text") and text_sha16_v140(r["text"]) not in blocked]
    return out[:256] if smoke_test else out
from .v170_mesosphere_loop import (
    ARCH_PRESETS_V170,
    SKETCH_R_V170,
    MeguvizV170CascadeEngine,
    MeguvizV170MesosphereLoopModel,
    _clamp_unit_norm,
    apply_crossview_late_fusion,
    apply_mgvz10_quant_roundtrip_inplace_v170,
    export_v170_bootable_and_onnx,
    fit_crossview_eba_h_fusion,
    fit_iut_stratified_r0_policy_v170,
    run_v170_ci_gates,
)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


@torch.no_grad()
def _project_r_echo_cap(model: MeguvizV170MesosphereLoopModel, r_echo_cap: float) -> float:
    """Enforce M9 Tight J-Cert `r_echo <= r_echo_cap` by projecting AR-U-Loop (lam_v, kap_v)."""
    if not getattr(model, "use_ar_uloop", False):
        return 0.0
    blk = model.core.blocks[0]
    lam_v, kap_v = blk.effective_ar_params()
    lam_f, kap_f = float(lam_v.item()), float(kap_v.item())
    r_echo = (1.0 - lam_f) + kap_f * 0.5
    if r_echo > r_echo_cap:
        if (1.0 - lam_f) >= r_echo_cap - 0.01:
            lam_target = min(0.745, max(lam_f, 1.0 - r_echo_cap + 0.02))
            lam_norm = min(0.999, max(0.001, (lam_target - 0.30) / 0.45))
            blk.ar_lam_raw.fill_(_logit(lam_norm))
            lam_f = 0.30 + 0.45 * lam_norm
        kap_target = min(kap_f, max(1e-4, 2.0 * (r_echo_cap - (1.0 - lam_f))))
        kap_norm = min(0.999, max(0.001, kap_target / 0.25))
        blk.ar_kap_raw.fill_(_logit(kap_norm))
        kap_f = 0.25 * kap_norm
        r_echo = (1.0 - lam_f) + kap_f * 0.5
    return float(r_echo)


@torch.no_grad()
def _apply_puc_shrink_and_perturb(
    model: MeguvizV170MesosphereLoopModel,
    shrink: float = 0.95,
    noise: float = 0.015,
    seed: int = 20261230,
) -> int:
    """Apply Strategy I Shrink-and-Perturb ONLY on non-R0 weights (keeps R0++ & APD-v2 untouched)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n_touched = 0
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if name.startswith(("rung0.", "apd_exit_head.")):
            continue
        if p.ndim < 2 or p.numel() < 64:
            continue
        std_p = float(p.float().std().item())
        eps = torch.randn(p.shape, generator=gen, dtype=torch.float32) * (noise * max(1e-4, std_p))
        p.copy_(shrink * p.float() + eps.to(p.device)).to(p.dtype)
        n_touched += 1
    return n_touched


@torch.no_grad()
def _mine_frontier_pools(
    model: MeguvizV170MesosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV170,
    pool_rows: list[dict[str, Any]],
    device: torch.device,
    max_mine: int = 24000,
    batch_size: int = 512,
    rng: random.Random | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Step 0: Zero-Leakage Active Frontier Miner over strictly training rows."""
    r = rng or random.Random(2026)
    if len(pool_rows) > max_mine:
        cand = r.sample(pool_rows, max_mine)
    else:
        cand = list(pool_rows)

    texts = [x["text"] for x in cand]
    preds = predict_student_v170(model, fcache, texts, device, batch_size=batch_size, outer_steps=4, use_amp=False)
    la_r0 = preds["logit_ai_r0"]
    la_t4 = preds["logit_ai_t4"]

    q_d02: list[dict[str, Any]] = []
    q_d04: list[dict[str, Any]] = []
    q_ml: list[dict[str, Any]] = []
    q_long: list[dict[str, Any]] = []

    from .v170_features import extract_v170_sidecar_batch
    cleans = [canonicalize_zero_pii_v150(t)[0] for t in texts]
    _, m7_surf_all, _ = extract_v170_sidecar_batch(cleans, np.array([len(c) for c in cleans], dtype=np.float32))

    for i, row in enumerate(cand):
        y_ai = float(row.get("y_ai", 0.0))
        a, b = float(la_r0[i]), float(la_t4[i])
        clen = len(cleans[i])
        r_h = float(m7_surf_all[i, 0])
        if y_ai <= 0.20 and (max(a, b) >= -0.5 or abs(a - b) >= 2.2):
            q_d02.append(row)
        if y_ai >= 0.80 and clen <= 60 and b <= 2.0:
            q_d04.append(row)
        if r_h < 0.35 or bool(row.get("is_ml_en", False)):
            q_ml.append(row)
        if clen >= 300:
            q_long.append(row)

    return {"Q_D02": q_d02, "Q_D04": q_d04, "Q_ML": q_ml, "Q_Long": q_long}


@torch.no_grad()
def _fit_titanguard_and_placebo(
    model: MeguvizV170MesosphereLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV170,
    long_pool: list[dict[str, Any]],
    short_hu_pool: list[dict[str, Any]],
    short_ai_pool: list[dict[str, Any]],
    device: torch.device,
    rng: random.Random,
) -> dict[str, Any]:
    """Step 3: Closed-form 68 B DS-TitanGuard orthogonal-complement fit + 11-split M9 Placebo Audit."""
    model.eval()
    hu_docs = [r["text"] for r in long_pool if float(r.get("y_ai", 0.0)) <= 0.20]
    ai_docs = [r["text"] for r in long_pool if float(r.get("y_ai", 0.0)) >= 0.80]
    hu_snips = [r["text"] for r in short_hu_pool if 40 <= len(r["text"]) <= 180]
    ai_snips = [r["text"] for r in short_ai_pool if 40 <= len(r["text"]) <= 180]

    # Fallback synthetic snippets if pool is tiny (e.g. --smoke-test)
    if len(hu_snips) < 8:
        hu_snips = ["笑死我了，这个插件装完直接起飞，周末肝了一整天终于搞定了[doge]。"] * 8
    if len(ai_snips) < 8:
        ai_snips = ["综上所述，该方案在保持系统稳定性的同时显著提升了整体处理效率。"] * 8
    if len(hu_docs) < 8:
        hu_docs = [" ".join(hu_snips[:3]) + " 这段主要是记录一下平时打游戏的真实体验。"] * 8
    if len(ai_docs) < 8:
        ai_docs = [" ".join(ai_snips[:3]) + " 总而言之，这一架构具有深远的工程价值。"] * 8

    def _embed_chunks(txts: list[str]) -> torch.Tensor:
        bk = _h2d_batch_v170(_collate_v170_cpu([{"text": t} for t in txts], fcache, include_meta=False), device)
        od = model(
            bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
            bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
            bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
            outer_steps=4, freeze_t1_prefix=False,
        )
        return od["emb_ai"].detach().cpu().float()

    def _chunk_sketches(doc_pairs: list[tuple[str, str]]) -> np.ndarray:
        """Compute causal Delta-Rule residual `res_e` (16-D) on 2-chunk pairs `(chunk0, chunk1)`."""
        e0 = _embed_chunks([p[0] for p in doc_pairs])
        e1 = _embed_chunks([p[1] for p in doc_pairs])
        g = e0.shape[-1] // 16
        sk0 = _clamp_unit_norm(e0[:, : g * 16].reshape(-1, 16, g).mean(dim=-1))
        sk1 = _clamp_unit_norm(e1[:, : g * 16].reshape(-1, 16, g).mean(dim=-1))
        return (sk1 - sk0).numpy()

    n_pair = min(32, max(8, len(hu_snips)))
    # Topic-shift / Quote / Code nuisance splits (H-H Topic, H-Quote, H-Code)
    hh_same = [(hu_docs[i % len(hu_docs)][:140], hu_docs[i % len(hu_docs)][140:280] or hu_snips[i % len(hu_snips)]) for i in range(n_pair)]
    hh_topic = [(hu_snips[i % len(hu_snips)], hu_snips[(i + 3) % len(hu_snips)]) for i in range(n_pair)]
    hh_author = [(hu_snips[i % len(hu_snips)], hu_snips[(i + 5) % len(hu_snips)]) for i in range(n_pair)]
    h_quote = [(hu_snips[i % len(hu_snips)], f"原话是“{ai_snips[i % len(ai_snips)][:80]}”，我觉得纯属扯淡。") for i in range(n_pair)]
    h_code = [(hu_snips[i % len(hu_snips)], "for i in range(64): out[i] = clamp(x[i] * scale[i], -127, 127)") for i in range(n_pair)]
    h_irony = [(hu_snips[i % len(hu_snips)], "真是太厉害了呢，综上所述简直无懈可击[doge]") for i in range(n_pair)]
    h_a = [(hu_snips[i % len(hu_snips)], ai_snips[i % len(ai_snips)]) for i in range(n_pair)]
    a_h = [(ai_snips[i % len(ai_snips)], hu_snips[i % len(hu_snips)]) for i in range(n_pair)]
    a_a = [(ai_docs[i % len(ai_docs)][:140], ai_docs[i % len(ai_docs)][140:280] or ai_snips[i % len(ai_snips)]) for i in range(n_pair)]
    h_sham = [(hu_snips[i % len(hu_snips)], hu_snips[i % len(hu_snips)]) for i in range(n_pair)]
    h_long = [(hu_docs[i % len(hu_docs)][:160], hu_docs[i % len(hu_docs)][160:320] or hu_snips[i % len(hu_snips)]) for i in range(n_pair)]

    res_nuis = np.concatenate([_chunk_sketches(hh_topic), _chunk_sketches(h_quote), _chunk_sketches(h_code)], axis=0)
    res_ha = _chunk_sketches(h_a)
    res_ah = _chunk_sketches(a_h)

    # Orthogonal-complement subspace projection: remove top-3 singular directions of nuisance shifts
    _, _, vt = np.linalg.svd(res_nuis - res_nuis.mean(axis=0, keepdims=True), full_matrices=False)
    u_t = vt[:3].T  # (16, 3)
    p_perp = np.eye(16, dtype=np.float64) - (u_t @ u_t.T)

    all_res = np.concatenate([res_nuis, res_ha, res_ah], axis=0).astype(np.float64)
    cov = np.cov(all_res, rowvar=False) + 0.1 * np.eye(16, dtype=np.float64)
    delta = (res_ha.mean(axis=0) - res_nuis.mean(axis=0)).astype(np.float64)
    w_raw = p_perp @ np.linalg.solve(cov, p_perp @ delta)
    w_norm = float(np.linalg.norm(w_raw))
    w_hat = (w_raw / max(1e-6, w_norm) * 0.5).astype(np.float32)
    b_hat = np.float32(-float(np.dot(w_hat, res_nuis.mean(axis=0))) - 0.15)

    model.long_agg.titan_w_dir.copy_(torch.from_numpy(w_hat).to(device))
    model.long_agg.titan_b_dir.copy_(torch.tensor(b_hat, device=device, dtype=torch.float32))

    # 11-Split M9 Placebo Audit
    splits = {
        "S0_H_H_Same": hh_same,
        "S1_H_H_Topic": hh_topic,
        "S2_H_H_Author": hh_author,
        "S3_H_Quote": h_quote,
        "S4_H_Code": h_code,
        "S5_H_Irony": h_irony,
        "S6_H_A_Splice": h_a,
        "S7_A_H_Splice": a_h,
        "S8_A_A_Long": a_a,
        "S9_H_Sham": h_sham,
        "S10_H_Long": h_long,
    }
    placebo_rep: dict[str, dict[str, float]] = {}
    for sname, spairs in splits.items():
        r_mat = _chunk_sketches(spairs)
        l2_s = np.linalg.norm(r_mat, axis=-1)
        u_dir = r_mat @ w_hat + float(b_hat)
        j_s = np.maximum(0.0, l2_s + u_dir)
        placebo_rep[sname] = {
            "delta_l2_mean": round(float(l2_s.mean()), 4),
            "dir_j_mean": round(float(j_s.mean()), 4),
            "dir_j_p95": round(float(np.percentile(j_s, 95)), 4),
        }
    return {
        "w_norm_raw": round(w_norm, 4),
        "b_hat": round(float(b_hat), 4),
        "placebo_11splits": placebo_rep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.7.1 Post-Training Suite")
    ap.add_argument("--distill-pool", type=Path, required=True)
    ap.add_argument("--sibling-dir", type=Path, required=True)
    ap.add_argument("--gold-v2-dir", type=Path, required=True)
    ap.add_argument("--wave3-dir", type=Path, required=True)
    ap.add_argument("--v11-data-dir", type=Path, required=True)
    ap.add_argument("--delta-dir", type=Path, default=None)
    ap.add_argument("--v04-data", type=Path, default=None)
    ap.add_argument("--extra-gold-pack", type=Path, default=None)
    ap.add_argument("--extra-cal-cert-pack", type=Path, default=None)
    ap.add_argument("--v07-tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", type=str, default="v1.7.1-PUC")
    ap.add_argument("--arch", type=str, default="v170_m1", choices=["v170_a0", "v170_m1"])
    ap.add_argument("--use-dual-hopfield", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--init-from-fp16", type=Path, default=None)
    ap.add_argument("--init-r0-ckpt", type=Path, default=None)
    ap.add_argument("--charlm-cache", type=Path, default=None)

    # Step 1 & 2 Post-Training Hyperparameters
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--stage-a-steps", type=int, default=1200)
    ap.add_argument("--stage-b-steps", type=int, default=800)
    ap.add_argument("--apd-steps", type=int, default=400)
    ap.add_argument("--p3-qat-steps", type=int, default=300)
    ap.add_argument("--lr-a", type=float, default=6.5e-4)
    ap.add_argument("--lr-b", type=float, default=4.5e-4)
    ap.add_argument("--qat-lr", type=float, default=4.0e-5)
    ap.add_argument("--puc-shrink", type=float, default=0.95)
    ap.add_argument("--puc-noise", type=float, default=0.015)
    ap.add_argument("--hop-margin-weight", type=float, default=0.08)
    ap.add_argument("--cnma-inv-weight", type=float, default=0.08)
    ap.add_argument("--protected-pnbr-tau", type=float, default=3.0)
    ap.add_argument("--r-echo-cap", type=float, default=0.50)
    ap.add_argument("--fit-titanguard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-eba-h", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eba-pi-nat", type=float, default=0.05)
    ap.add_argument("--iut-alpha-max", type=float, default=0.04)
    ap.add_argument("--auto-ship-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile-mode", type=str, default="default")
    ap.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp-bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefetch-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261230)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if args.smoke_test:
        args.stage_a_steps = min(args.stage_a_steps, 16)
        args.stage_b_steps = min(args.stage_b_steps, 12)
        args.apd_steps = min(args.apd_steps, 12)
        args.p3_qat_steps = min(args.p3_qat_steps, 8)
        args.batch_size = min(args.batch_size, 32)
        args.eval_batch_size = min(args.eval_batch_size, 64)
        args.prefetch_workers = 0

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir or out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Corpus (Zero-Leakage)
    if args.smoke_test and (not args.distill_pool.is_file()):
        corpus = _make_synthetic_smoke_corpus(args.seed)
        distill_pool = corpus["train_extra"]
        extra_cal_cert_rows = corpus["val_gold_v2"]
        extra_gold_rows = []
    else:
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
        extra_cal_cert_rows = []
        if args.extra_cal_cert_pack is not None and args.extra_cal_cert_pack.is_file():
            import gzip
            with gzip.open(args.extra_cal_cert_pack, "rt", encoding="utf-8") as cf:
                for line in cf:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("text"):
                            extra_cal_cert_rows.append(r)
                            blocked.add(text_sha16_v140(r["text"]))
        distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)
        extra_gold_rows = []
        if args.extra_gold_pack is not None and args.extra_gold_pack.is_file():
            import gzip
            with gzip.open(args.extra_gold_pack, "rt", encoding="utf-8") as gf:
                for line in gf:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("text") and text_sha16_v140(r["text"]) not in blocked:
                            extra_gold_rows.append(r)

    d_all = distill_pool + corpus.get("v04_pool", []) + extra_gold_rows
    gold_pos = [r for r in d_all if float(r.get("y_ai", 0.0)) >= 0.80]
    gold_hd = [r for r in d_all if float(r.get("y_ai", 0.0)) <= 0.20 and float(r.get("hard_neg", 0.0)) > 0.5]
    val_g2 = corpus.get("val_gold_v2", [])
    val_w3 = corpus.get("val_wave3", [])
    val_old = corpus.get("val_old_video_neg", [])
    val_ml = corpus.get("val_multilingual", [])

    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    vocab_size = max(12288, int(getattr(tokenizer, "vocab_size", tokenizer._tok.get_vocab_size())))

    # Load CharLM
    charlm_loaded = False
    if args.charlm_cache is not None and args.charlm_cache.is_file() and (not args.smoke_test):
        with args.charlm_cache.open("rb") as cf_f:
            c_obj = pickle.load(cf_f)
        fold_lms, text_to_fold, full_charlm = c_obj["fold_lms"], c_obj["text_to_fold"], c_obj["full_charlm"]
        charlm_loaded = True
    if not charlm_loaded:
        lm_rows = (gold_pos[:150] + gold_hd[:150] + d_all[: (300 if args.smoke_test else 60000)])
        lm_txts = [r["text"] for r in lm_rows]
        lm_y = np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in lm_rows], dtype=np.int32)
        lm_g = [r.get("group_id") for r in lm_rows]
        fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
            lm_txts, lm_y, lm_g, n_folds=3,
            min_counts=(2, 3, 4) if args.smoke_test else (3, 5, 7),
            max_order_keys=(3000, 1200, 600) if args.smoke_test else (22000, 16000, 10000),
            seed=args.seed,
        )

    fcache = _StrictGroupOOFFeatureCacheV170(
        tokenizer=tokenizer, max_len=288, head_len=192, engram_buckets=8192, r0_b2=R0_B2_V130 * 2, r0_b3=R0_B3_V130,
    )
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)

    # 2. Build & Load Parent v1.7.0 Model
    model = MeguvizV170MesosphereLoopModel(
        arch=args.arch, vocab_size=vocab_size, max_len=288, style_dim=56,
        use_discourse_ops=True, lardo_w8_core=True, use_dual_hopfield=args.use_dual_hopfield,
    ).to(device)

    if args.init_from_fp16 is not None and args.init_from_fp16.is_file():
        parent_sd = torch.load(args.init_from_fp16, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.to(device) for k, v in parent_sd.items() if k in model.state_dict()}, strict=False)
    else:
        parent_sd = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        if args.smoke_test:
            torch.save(parent_sd, out_dir / "_smoke_parent_v170.pt")

    if args.init_r0_ckpt is not None and args.init_r0_ckpt.is_file():
        r0_sd = torch.load(args.init_r0_ckpt, map_location="cpu", weights_only=True)
        model.rung0.load_state_dict({k: v.to(device) for k, v in r0_sd.items() if k in model.rung0.state_dict()}, strict=False)
    # Strictly freeze R0++ weights forever in v1.7.1!
    for p in model.rung0.parameters():
        p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Step 0: Zero-Leakage Active Frontier Miner
    # -----------------------------------------------------------------------
    rng = random.Random(args.seed)
    frontier = _mine_frontier_pools(
        model, fcache, d_all, device,
        max_mine=(256 if args.smoke_test else 24000),
        batch_size=args.eval_batch_size, rng=rng,
    )
    print(f"[v1.7.1 Step 0 Miner] Q_D02={len(frontier['Q_D02'])} | Q_D04={len(frontier['Q_D04'])} | Q_ML={len(frontier['Q_ML'])} | Q_Long={len(frontier['Q_Long'])}", flush=True)

    # Pre-collate quick val
    pre_val_g2 = _precollate_eval_split_v170(val_g2, fcache, device, args.eval_batch_size) if val_g2 else []

    def _eval_quick_val(outer_steps: int = 4) -> dict[str, float]:
        if not val_g2:
            return {"pr": 0.0, "r95": 0.0, "sel_obj": 0.0}
        p = predict_student_v170(model, fcache, [r["text"] for r in val_g2], device, batch_size=args.eval_batch_size, outer_steps=outer_steps, use_amp=False, precollated=pre_val_g2)
        y = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in val_g2], dtype=np.int32)
        m = _eval_binary(y, expit(np.clip(p["logit_ai_t4"], -20, 20)))
        return {"pr": round(m["pr_auc"], 4), "r95": round(m["r_at_p95"], 4), "sel_obj": round(m["pr_auc"] + 0.35 * m["r_at_p95"], 5)}

    def _sample_v171_batch(bs: int, focus_frontier: bool = False) -> list[dict[str, Any]]:
        out_rows: list[dict[str, Any]] = []
        q_pools = [p for p in (frontier["Q_D02"], frontier["Q_D04"], frontier["Q_ML"], frontier["Q_Long"]) if p]
        n_front = int(bs * (0.45 if focus_frontier else 0.30)) if q_pools else 0
        for _ in range(n_front):
            qp = rng.choice(q_pools)
            out_rows.append(rng.choice(qp))
        while len(out_rows) < bs:
            out_rows.append(rng.choice(d_all))
        return out_rows

    # -----------------------------------------------------------------------
    # Step 1: Stage A (Strategy I PUC + Hopfield Margin + CNMA-R Y=0 + ER-DRO-R)
    # -----------------------------------------------------------------------
    n_puc = _apply_puc_shrink_and_perturb(model, shrink=args.puc_shrink, noise=args.puc_noise, seed=args.seed)
    print(f"[v1.7.1 Step 1 Stage A] Applied PUC S&P to {n_puc} non-R0 matrices; steps={args.stage_a_steps}", flush=True)

    stage_a_params = [p for n, p in model.named_parameters() if (not n.startswith(("rung0.", "apd_exit_head."))) and p.requires_grad]
    opt_a = _make_adamw([{"params": stage_a_params, "lr": args.lr_a, "weight_decay": 0.01}], device)
    dro = ExcessRiskGroupDROR()

    if args.compile and device.type == "cuda":
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        fwd_fn = torch.compile(model, mode=args.compile_mode, dynamic=args.compile_dynamic)
    else:
        fwd_fn = model

    for step in range(1, args.stage_a_steps + 1):
        _set_lr(opt_a, _cosine_lr(step, args.stage_a_steps, args.lr_a, warmup=min(80, max(2, args.stage_a_steps // 10))))
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=False), fcache, rng=rng, punct_jitter_prob=0.08), device)
        opt_a.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = fwd_fn(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                outer_steps=1, freeze_t1_prefix=False,
            )
            y_ai = bk["y_ai"]
            ai_v = bk["has_ai"] > 0.5
            iw = bk["dro_iw"]
            z1 = od["logit_ai_steps"][-1].float()
            per_ai = F.binary_cross_entropy_with_logits(z1, y_ai, reduction="none")
            loss = (per_ai * ai_v.float() * iw).sum() / (ai_v.float() * iw).sum().clamp_min(1.0)
            loss = loss + 0.50 * (F.binary_cross_entropy_with_logits(od["logit_promo"].float(), bk["y_pr"], reduction="none") * (bk["has_pr"] > 0.5).float()).sum() / (bk["has_pr"] > 0.5).float().sum().clamp_min(1.0)

            # Hard-Margin Dual-Manifold Energy Loss
            if model.dual_hopfield is not None and "hop_witness" in od:
                e_h = od["hop_witness"][:, 0].float()
                d_gap = od["hop_witness"][:, 2].float()
                is_ai = (ai_v & (y_ai >= 0.5)).float()
                is_hu = (ai_v & (y_ai <= 0.20)).float()
                l_hop = (F.relu(1.2 - d_gap) * is_ai).sum() / is_ai.sum().clamp_min(1.0)
                l_hop = l_hop + (F.relu(1.2 + d_gap) * is_hu).sum() / is_hu.sum().clamp_min(1.0)
                l_hop = l_hop + 0.02 * (F.relu(e_h - 1.5) * is_hu).sum() / is_hu.sum().clamp_min(1.0)
                loss = loss + float(args.hop_margin_weight) * l_hop

            # M7 Y=0 Conditional Zero-Distribution Invariance Loss across strata
            if model.cnma_r is not None:
                hu_m = ai_v & (y_ai <= 0.20)
                if int(hu_m.sum().item()) >= 8:
                    z_hu = z1[hu_m]
                    g_mean = z_hu.mean()
                    # Strata via m7_surf: r_H >= 0.5 & len bucket
                    r_h = bk["m7_surf"][hu_m, 0]
                    log_l = bk["m7_surf"][hu_m, 4] * 6.0  # log1p(L)
                    strata = [
                        (r_h >= 0.5) & (log_l <= math.log1p(60.0)),
                        (r_h >= 0.5) & (log_l > math.log1p(60.0)) & (log_l < math.log1p(300.0)),
                        (r_h >= 0.5) & (log_l >= math.log1p(300.0)),
                        (r_h < 0.5),
                    ]
                    l_inv = torch.zeros((), device=device, dtype=torch.float32)
                    for smask in strata:
                        if int(smask.sum().item()) >= 2:
                            l_inv = l_inv + (z_hu[smask].mean() - g_mean).pow(2)
                    loss = loss + float(args.cnma_inv_weight) * l_inv

        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage_a_params, 1.0, foreach=True)
        opt_a.step()

    # -----------------------------------------------------------------------
    # Step 2: Stage B (M5 Prefix-Frozen Protected PNBR-R + M9 Tight J-Cert Projection)
    # -----------------------------------------------------------------------
    model.eval()
    t1_before = predict_student_v170(model, fcache, [r["text"] for r in val_g2], device, batch_size=args.eval_batch_size, outer_steps=1, use_amp=False, precollated=pre_val_g2)["logit_ai_t1"].copy() if val_g2 else np.zeros((0,))
    stage_b_prefixes = (
        "core.blocks.0.bcast_w", "core.blocks.0.ar_",
        "core.lora_", "core.step_phi_mlp",
        "core.sketch_P", "core.ri_", "core.dr_",
        "pnbr_head", "pnbr_cap", "ln_ds_", "halt_utility_head",
    )
    s2_params = []
    for n, p in model.named_parameters():
        keep_tr = any(n.startswith(pfx) for pfx in stage_b_prefixes)
        p.requires_grad_(keep_tr)
        if keep_tr:
            s2_params.append(p)
    opt_b = _make_adamw([{"params": s2_params, "lr": args.lr_b, "weight_decay": 0.01}], device)

    r_echo_val = 0.0
    for step in range(1, args.stage_b_steps + 1):
        _set_lr(opt_b, _cosine_lr(step, args.stage_b_steps, args.lr_b, warmup=min(40, max(1, args.stage_b_steps // 10))))
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=True), fcache, rng=rng, punct_jitter_prob=0.05), device)
        opt_b.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = fwd_fn(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                outer_steps=4, freeze_t1_prefix=True,
            )
            lgs = od["logit_ai_steps"]
            y_ai = bk["y_ai"]
            ai_v = bk["has_ai"] > 0.5
            iw = bk["dro_iw"]
            z1_det = lgs[0].detach().float()
            # M5 Protected PNBR: freeze residual delta_t on high-confidence |f_1| >= tau!
            prot_mask = (z1_det.abs() < float(args.protected_pnbr_tau)).float()
            loss_b = torch.zeros((), device=device, dtype=torch.float32)
            for ti in range(1, len(lgs)):
                zt_raw = lgs[ti].float()
                zp = lgs[ti - 1].detach().float()
                delta_t = (zt_raw - zp) * prot_mask
                zt = zp + delta_t
                l_t = (F.binary_cross_entropy_with_logits(zt, y_ai, reduction="none") * ai_v.float() * iw).sum() / (ai_v.float() * iw).sum().clamp_min(1.0)
                fc = zp - zp.mean()
                dc = delta_t - delta_t.mean()
                corr = (fc * dc).sum() / (fc.norm() * dc.norm() + 1e-6)
                loss_b = loss_b + (0.50 * l_t + 0.03 * corr.pow(2))
            ptp = model.core.sketch_P.float().T @ model.core.sketch_P.float()
            loss_b = loss_b + 0.01 * (ptp - torch.eye(SKETCH_R_V170, device=device)).pow(2).sum()

        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(s2_params, 1.0, foreach=True)
        opt_b.step()
        r_echo_val = _project_r_echo_cap(model, float(args.r_echo_cap))

    model.eval()
    t1_after = predict_student_v170(model, fcache, [r["text"] for r in val_g2], device, batch_size=args.eval_batch_size, outer_steps=1, use_amp=False, precollated=pre_val_g2)["logit_ai_t1"].copy() if val_g2 else np.zeros((0,))
    t1_freeze_diff = float(np.max(np.abs(t1_after - t1_before))) if len(t1_before) > 0 else 0.0
    print(f"[v1.7.1 Step 2 Stage B] T1 prefix freeze max_diff={t1_freeze_diff:.2e} | r_echo={r_echo_val:.4f} (cap={args.r_echo_cap})", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Stage C (APD-v2 Retrain + 68B DS-TitanGuard + 8-Cand WiSE-FT + Export + 26/26 CI)
    # -----------------------------------------------------------------------
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("apd_exit_head."))
    apd_params = [p for p in model.apd_exit_head.parameters() if p.requires_grad]
    opt_apd = _make_adamw([{"params": apd_params, "lr": 1.0e-3, "weight_decay": 1e-3}], device)
    for _ in range(args.apd_steps):
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=False), fcache, rng=rng, need_bpe=True), device)
        opt_apd.zero_grad(set_to_none=True)
        r0_o = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["m7_surf"])
        p_apd = r0_o["p_apd_safe"]
        y_ai = bk["y_ai"]
        ai_v = bk["has_ai"] > 0.5
        l_apd = (F.binary_cross_entropy(p_apd.clamp(1e-5, 1.0 - 1e-5), (y_ai < 0.5).float(), reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
        l_apd.backward()
        opt_apd.step()

    titan_rep: dict[str, Any] = {"enabled": False}
    if args.fit_titanguard:
        titan_rep = _fit_titanguard_and_placebo(model, fcache, frontier["Q_Long"], [r for r in d_all if float(r.get("y_ai", 0.0)) <= 0.20], gold_pos, device, rng)
        titan_rep["enabled"] = True

    preqat_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Light-QAT
    if args.p3_qat_steps > 0:
        for n, p in model.named_parameters():
            p.requires_grad_(not n.startswith(("rung0.", "apd_exit_head.", "long_agg.titan_")))
        qat_params = [p for p in model.parameters() if p.requires_grad]
        opt_q = _make_adamw([{"params": qat_params, "lr": args.qat_lr, "weight_decay": 0.0}], device)
        set_v130_qat_mode(model, True)
        for _ in range(args.p3_qat_steps):
            bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=True), fcache, rng=rng, punct_jitter_prob=0.05), device)
            opt_q.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                od = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                    bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                    outer_steps=4, freeze_t1_prefix=False,
                )
                loss_q = (F.binary_cross_entropy_with_logits(od["logit_ai_steps"][-1].float(), bk["y_ai"], reduction="none") * (bk["has_ai"] > 0.5).float()).sum() / (bk["has_ai"] > 0.5).float().sum().clamp_min(1.0)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, 1.0, foreach=True)
            opt_q.step()
            _project_r_echo_cap(model, float(args.r_echo_cap))
        set_v130_qat_mode(model, False)

    qat_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 8-Candidate Same-Basin WiSE-FT Auto-Ship-Gate
    ship_cands: dict[str, dict[str, torch.Tensor]] = {
        "direct_pack_preqat": preqat_sd,
        "light_qat": qat_sd,
        "fp16_ema_soup": {k: (0.5 * preqat_sd[k].float() + 0.5 * qat_sd[k].float()) if qat_sd[k].is_floating_point() else qat_sd[k].clone() for k in qat_sd},
    }
    for alpha_w in (0.85, 0.70, 0.50, 0.30, 0.15):
        soup = {}
        for k, v in qat_sd.items():
            if k in parent_sd and tuple(parent_sd[k].shape) == tuple(v.shape) and v.is_floating_point() and (not k.startswith(("rung0.", "apd_exit_head.", "long_agg.titan_"))):
                soup[k] = alpha_w * v.float() + (1.0 - alpha_w) * parent_sd[k].float()
            else:
                soup[k] = v.clone()
        ship_cands[f"wise_ft_a{int(alpha_w * 100)}"] = soup

    winner_name = "light_qat"
    winner_sd = qat_sd
    ship_scores: dict[str, Any] = {}
    if args.auto_ship_gate:
        best_sc = -1e9
        for cname, csd in ship_cands.items():
            model.load_state_dict({k: v.to(device) for k, v in csd.items()})
            apply_mgvz10_quant_roundtrip_inplace_v170(model)
            ev = _eval_quick_val(outer_steps=4)
            ship_scores[cname] = ev
            if ev["sel_obj"] > best_sc:
                best_sc, winner_name, winner_sd = ev["sel_obj"], cname, csd

    model.load_state_dict({k: v.to(device) for k, v in winner_sd.items()})
    apply_mgvz10_quant_roundtrip_inplace_v170(model)
    torch.save({k: v.detach().cpu().half() for k, v in model.state_dict().items()}, out_dir / f"{args.version}_fp16_master.pt")

    # Calibration (EBA-H + M8 IUT SCE)
    cal_rows = val_g2 + val_w3 + val_old + val_ml
    cert_rows = extra_cal_cert_rows if len(extra_cal_cert_rows) >= 48 else cal_rows
    p_cal = predict_student_v170(model, fcache, [r["text"] for r in cal_rows], device, batch_size=args.eval_batch_size, outer_steps=4, use_amp=False)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    p_cert = predict_student_v170(model, fcache, [r["text"] for r in cert_rows], device, batch_size=args.eval_batch_size, outer_steps=4, use_amp=False) if cert_rows is not cal_rows else p_cal
    y_cert_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cert_rows], dtype=np.int32)

    cv_fusion = fit_crossview_eba_h_fusion(
        p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], y_cal_ai, pi_nat=float(args.eba_pi_nat),
        cert_r0=p_cert["logit_ai_r0"], cert_deep=p_cert["logit_ai_t4"], cert_y=y_cert_ai,
    ) if args.use_eba_h else {"enabled": False, "formula": "none"}

    cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
    t_ai = fit_scalar_temperature_v140(cal_fused_t4 - float(cv_fusion.get("prior_shift", 0.0)), y_cal_ai)
    t_pr = fit_scalar_temperature_v140(p_cal["logit_promo_t4"], y_cal_pr) if y_cal_pr.sum() >= 8 else 1.5
    lut_ai_r0 = fit_smooth_ai_lut_v122(p_cal["logit_ai_r0"], y_cal_ai)
    lut_ai_full = fit_output_shifted_promo_lut_v122(cal_fused_t4, y_cal_ai, pi_nat=float(args.eba_pi_nat), shift_damp=0.55) if args.use_eba_h else fit_smooth_ai_lut_v122(cal_fused_t4, y_cal_ai)
    lut_pr_r0 = fit_output_shifted_promo_lut_v122(p_cal["logit_promo_r0"], y_cal_pr) if y_cal_pr.sum() >= 8 else None
    lut_pr_full = fit_output_shifted_promo_lut_v122(p_cal["logit_promo_t4"], y_cal_pr) if y_cal_pr.sum() >= 8 else None

    cert_p_r0 = apply_spline_temp_lut(p_cert["logit_ai_r0"], lut_ai_r0)
    cert_clens = np.array([len(canonicalize_zero_pii_v150(r["text"])[0]) for r in cert_rows], dtype=np.int64)
    r0_ai_pol = fit_iut_stratified_r0_policy_v170(cert_p_r0, p_cert["p_defer_ai_r0"], p_cert["p_apd_safe"], cert_clens, y_cert_ai, alpha_max=float(args.iut_alpha_max))

    manifest = {
        "version": args.version,
        "max_len": 288, "head_len": 192, "long_char_trigger": 300,
        "temp_ai": round(float(t_ai), 4), "temp_promo": round(float(t_pr), 4),
        "lut_ai_r0": lut_ai_r0, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_full": lut_pr_full,
        "rung0_ai_ltt_policy": r0_ai_pol,
        "rung0_pr_ltt_policy": {"enabled": False, "tau_promo": 0.015, "defer_max": 0.35, "max_chars": 180},
        "crossview_late_fusion": cv_fusion,
        "long_focal_router": {"enabled": True, "win_chars": 160, "max_chunks": 6, "lambda": 0.92},
        "anytime_step_policy": {"enabled": True, "max_infer_steps": 4, "u_halt_thresh": 0.015, "disagree_dt_thresh": 1.35, "uncertain_lo": 0.12, "uncertain_hi": 0.88, "t1_exit_enabled": True},
    }
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    export_rep = export_v170_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    eng = MeguvizV170CascadeEngine(out_dir, threads=4, r0_threads=2)
    ci_gates = run_v170_ci_gates(model, tokenizer, export_rep, out_dir, device, charlm=full_charlm, engine=eng)

    p_t1_c = expit(np.clip(p_cert["logit_ai_t1"], -20, 20))
    d14_c = p_cert["logit_ai_t4"] - p_cert["logit_ai_t1"]
    b_cert_val = round(float(np.mean((y_cert_ai - p_t1_c) * d14_c) - 0.125 * np.mean(d14_c ** 2)), 5)

    report = {
        "version": args.version,
        "arch": args.arch,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "frontier_counts": {k: len(v) for k, v in frontier.items()},
        "t1_prefix_freeze_max_diff": t1_freeze_diff,
        "r_echo": round(r_echo_val, 4),
        "r_echo_cap": float(args.r_echo_cap),
        "b_cert": b_cert_val,
        "titanguard": titan_rep,
        "auto_ship_gate": {"winner": winner_name, "candidates": ship_scores},
        "export": export_rep,
        "release_gates": {"v170_ci_passed": bool(ci_gates["all_ci_passed"]), "ci_details": ci_gates},
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v1.7.1 DONE] version={args.version} | bootable={export_rep.get('bootable_mib')} MiB | CI={ci_gates['all_ci_passed']} | r_echo={r_echo_val:.4f} | T1_diff={t1_freeze_diff:.2e}", flush=True)
    if not ci_gates["all_ci_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
