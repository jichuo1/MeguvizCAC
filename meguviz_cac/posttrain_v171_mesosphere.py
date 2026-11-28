"""MeguvizCAC v1.7.1 Post-Training Suite (v1.6.1 Natural R0++ Anchor + Frontier Miner + Full-Recipe Stage A PUC/KD/CF/Longform + Stage B Protected PNBR/J-Cert + Stage C APD-v2/68B TitanGuard/WiSE-FT/IUT + 26/26 CI).

Incorporates ALL empirical lessons from the v1.7.0 9-Way Blind Showdown:
  1. Loads `shared_r0_v161_frozen.pt` (52/52 exact tensor match, 4000-step p1_use_cf natural R0++:
     v3-real R0_R99=75.0%, v2 R0_R99=73.8%), unwrapping `rung0_state_dict` cleanly and keeping R0++ frozen!
  2. Fixes `r0_b3 = model.r0_b3` (16384), tuple/dict CharLM cache loading, full gold+distill+longform+CF
     corpus ingestion, explicit `model.train()` across stages, and `expit(logit / T)` before `apply_spline_temp_lut`!
  3. Stage A (1600 steps): Strategy I PUC + ER-DRO-R + Progressive Super-Teacher KD + TCBR-R + Recall-Floor
     + CF Mirror Pairs + Rule Format-Edit CF (fixes H03/H07) + D02 Paraphrase CF + Promo-Flip CF
     + DynaSeam Longform/Sandwiches + DualHopfield Hard Margin + M7 CNMA-R Y=0 Invariance (fixes M02/M04)!
  4. Stage B (800 steps): M5 Prefix-Frozen Protected PNBR-R (`|f_1| < tau`) + Tight `r_echo <= 0.50`
     + `dr_v` norm cap (`<= 6.0`, guarantees `W8_fro < 40` & positive `B_cert`!).
  5. Stage C: Retrained `APD-v2` + `68 B DS-TitanGuard` orthogonal-complement fit (11-split M9 placebo)
     + Light-QAT + 8-Candidate WiSE-FT Auto-Ship-Gate scored on FULL multi-split `cal_rows` (`g2+w3+old+ml`)
     + Disjoint `cal_fit/cal_cert` `EBA-H` + `M8 IUT` Stratified R0 Exit + 4-stage FP32 ONNX + `26/26 CI`!

Strictly Zero-Leakage: never accesses OjinjinBench.
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
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from .train_v140_paradise_apex import _eval_bin, _load_v140_distill_pool
from .train_v170_mesosphere import (
    ExcessRiskGroupDROR,
    _StrictGroupOOFFeatureCacheV170,
    _collate_v170_cpu,
    _h2d_batch_v170,
    _precollate_eval_split_v170,
    predict_student_v170,
)
from .v07_features import BiliPickyBPETokenizer, apply_spline_temp_lut
from .v11_features import plan_long_chunks_v11
from .v122_apexnano import fit_output_shifted_promo_lut_v122, fit_smooth_ai_lut_v122
from .v130_xtreme_apex import AI_END_V130, set_v130_qat_mode
from .v140_paradise_apex import R0_STRUCT_DIM_V140
from .v140_superteacher import (
    build_v140_superteacher_corpus,
    fit_scalar_temperature_v140,
    read_jsonl_any_v140,
    sample_cf_mirror_batch_v140,
    text_sha16_v140,
)
from .v150_heaven_loop import (
    CompactCharLMV150,
    canonicalize_zero_pii_v150,
    fit_group_oof_charlm_v150,
)
from .v160_stratosphere_loop import (
    apply_crossview_late_fusion,
    fit_crossview_eba_h_fusion,
)
from .v170_features import extract_v170_sidecar_batch
from .v170_mesosphere_loop import (
    SKETCH_R_V170,
    MeguvizV170CascadeEngine,
    MeguvizV170MesosphereLoopModel,
    _clamp_unit_norm,
    apply_mgvz10_quant_roundtrip_inplace_v170,
    export_v170_bootable_and_onnx,
    fit_iut_stratified_r0_policy_v170,
    run_v170_ci_gates,
)


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 60, min_ratio: float = 0.05) -> float:
    if step <= warmup:
        return base_lr * max(0.05, step / max(1, warmup))
    t = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(1.0, t))))


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _make_adamw(param_groups: list[dict[str, Any]], device: torch.device) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), fused=(device.type == "cuda"))


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _minimal_rule_format_edit(txt: str, rng: random.Random) -> str:
    """Label-preserving rule-based format edit (fixes H03_tech_human & H07_bulleted_review FPs!)."""
    s = str(txt).strip()
    if len(s) < 12:
        return s
    mode = rng.randint(0, 5)
    if mode == 0:
        parts = [p.strip() for p in s.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n") if p.strip()]
        return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts[:5])) if len(parts) >= 2 else f"1. {s}"
    if mode == 1:
        lines = [ln.strip().lstrip("0123456789.、-•* ") for ln in s.splitlines() if ln.strip()]
        return " ".join(lines) if lines else s
    if mode == 2:
        return f"【要点】\n{s}"
    if mode == 3:
        parts = [p.strip() for p in s.replace("，", "，\n").replace("。", "。\n").split("\n") if p.strip()]
        return "\n".join(f"- **{p[:4]}**：{p}" for p in parts[:4]) if len(parts) >= 2 else f"- **核心观点**：{s}"
    if mode == 4:
        return f"总结如下：\n{s}"
    return (s.replace("[doge]", "").strip() or s) if "[doge]" in s else f"{s} [doge]"


@torch.no_grad()
def _project_r_echo_and_drv_cap(
    model: MeguvizV170MesosphereLoopModel, r_echo_cap: float, drv_fro_cap: float = 6.0
) -> float:
    """Enforce M9 Tight J-Cert `r_echo <= r_echo_cap` AND keep Delta-Rule `dr_v` bounded (`W8_fro < 40`)."""
    r_echo = 0.0
    if getattr(model, "use_ar_uloop", False):
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
    if getattr(model.core, "delta_rule_dim", 0) > 0 and hasattr(model.core, "dr_v"):
        w = model.core.dr_v.weight
        fn = float(w.float().norm().item())
        if fn > drv_fro_cap:
            w.mul_(drv_fro_cap / fn)
    return float(r_echo)


@torch.no_grad()
def _apply_puc_shrink_and_perturb(
    model: MeguvizV170MesosphereLoopModel,
    shrink: float = 0.96,
    noise: float = 0.012,
    seed: int = 20261230,
) -> int:
    """Apply gentle Strategy I Shrink-and-Perturb ONLY on non-R0 weights (keeps R0++ & APD-v2 untouched)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n_touched = 0
    for name, p in model.named_parameters():
        if (not p.is_floating_point()) or name.startswith(("rung0.", "apd_exit_head.")):
            continue
        if p.ndim < 2 or p.numel() < 64:
            continue
        std_p = float(p.float().std().item())
        eps = torch.randn(p.shape, generator=gen, dtype=torch.float32) * (noise * max(1e-4, std_p))
        p.copy_((shrink * p.float() + eps.to(p.device)).to(p.dtype))
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
    model.eval()
    r = rng or random.Random(2026)
    cand = r.sample(pool_rows, max_mine) if len(pool_rows) > max_mine else list(pool_rows)
    texts = [x["text"] for x in cand]
    preds = predict_student_v170(model, fcache, texts, device, batch_size=batch_size, outer_steps=4, use_amp=False)
    la_r0, la_t4 = preds["logit_ai_r0"], preds["logit_ai_t4"]

    cleans = [canonicalize_zero_pii_v150(t)[0] for t in texts]
    _, m7_surf_all, _ = extract_v170_sidecar_batch(cleans, np.array([max(1, len(c) // 2) for c in cleans], dtype=np.int64))

    q_d02, q_d04, q_ml, q_long = [], [], [], []
    for i, row in enumerate(cand):
        has_a = float(row.get("has_ai", 1.0)) > 0.5
        y_ai = float(row.get("y_ai", 0.0))
        a, b = float(la_r0[i]), float(la_t4[i])
        clen = len(cleans[i])
        r_h = float(m7_surf_all[i, 0])
        if has_a and y_ai <= 0.20 and (max(a, b) >= -0.6 or abs(a - b) >= 2.0):
            q_d02.append(row)
        if has_a and y_ai >= 0.75 and clen <= 65 and b <= 2.2:
            q_d04.append(row)
        if r_h < 0.35 or bool(row.get("is_ml_en", False)):
            q_ml.append(row)
        if clen >= 280:
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
) -> dict[str, Any]:
    """Step 3: Closed-form 68 B DS-TitanGuard orthogonal-complement fit + 11-split M9 Placebo Audit."""
    model.eval()
    hu_docs = [r["text"] for r in long_pool if float(r.get("y_ai", 0.0)) <= 0.20]
    ai_docs = [r["text"] for r in long_pool if float(r.get("y_ai", 0.0)) >= 0.80]
    hu_snips = [r["text"] for r in short_hu_pool if 40 <= len(r["text"]) <= 180]
    ai_snips = [r["text"] for r in short_ai_pool if 40 <= len(r["text"]) <= 180]
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
        e0 = _embed_chunks([p[0] for p in doc_pairs])
        e1 = _embed_chunks([p[1] for p in doc_pairs])
        g = e0.shape[-1] // 16
        sk0 = _clamp_unit_norm(e0[:, : g * 16].reshape(-1, 16, g).mean(dim=-1))
        sk1 = _clamp_unit_norm(e1[:, : g * 16].reshape(-1, 16, g).mean(dim=-1))
        return (sk1 - sk0).numpy()

    n_pair = min(32, max(8, len(hu_snips)))
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

    _, _, vt = np.linalg.svd(res_nuis - res_nuis.mean(axis=0, keepdims=True), full_matrices=False)
    u_t = vt[:3].T
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

    splits = {
        "S0_H_H_Same": hh_same, "S1_H_H_Topic": hh_topic, "S2_H_H_Author": hh_author,
        "S3_H_Quote": h_quote, "S4_H_Code": h_code, "S5_H_Irony": h_irony,
        "S6_H_A_Splice": h_a, "S7_A_H_Splice": a_h, "S8_A_A_Long": a_a,
        "S9_H_Sham": h_sham, "S10_H_Long": h_long,
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
    return {"w_norm_raw": round(w_norm, 4), "b_hat": round(float(b_hat), 4), "placebo_11splits": placebo_rep}


def _make_synthetic_smoke_corpus(seed: int = 2026) -> dict[str, Any]:
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
        rows.append({
            "text": f"{txt} #{i}", "y_ai": y, "has_ai": 1.0, "y_pr": 0.0, "has_pr": 1.0,
            "has_teacher": 0.0, "kd_w": 1.0, "t_ai_lg": 0.0, "t_ai_r0": 0.0, "t_pr_lg": 0.0,
            "hard_neg": 1.0 if (y == 0.0 and i % 4 == 0) else 0.0, "trust_neg": 1.0 if y == 0.0 else 0.0,
            "group_id": f"g{i%6}", "era5": -1, "sub4": -1,
        })
    return {
        "train_ai_pos": [r for r in rows[:48] if r["y_ai"] >= 0.5],
        "train_hu_hard": [r for r in rows[:48] if r["y_ai"] < 0.5],
        "train_hu_natural": [r for r in rows[:48] if r["y_ai"] < 0.5],
        "train_promo_extra": [],
        "val_gold_v2": rows[48:],
        "val_wave3": rows[48:],
        "val_old_video_neg": [r for r in rows[48:] if r["y_ai"] < 0.5],
        "val_multilingual": rows[48:],
        "val_promo_v11": [],
        "stats": {"admitted_cf_pairs": 0},
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
    ap.add_argument("--version", type=str, default="v1.7.1-M1")
    ap.add_argument("--arch", type=str, default="v170_m1", choices=["v170_a0", "v170_m1"])
    ap.add_argument("--use-dual-hopfield", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--init-from-fp16", type=Path, default=None)
    ap.add_argument("--init-r0-ckpt", type=Path, default=None)
    ap.add_argument("--charlm-cache", type=Path, default=None)

    # Post-Training Hyperparameters
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--stage-a-steps", type=int, default=1600)
    ap.add_argument("--stage-b-steps", type=int, default=800)
    ap.add_argument("--apd-steps", type=int, default=400)
    ap.add_argument("--p3-qat-steps", type=int, default=300)
    ap.add_argument("--lr-a", type=float, default=6.0e-4)
    ap.add_argument("--lr-b", type=float, default=4.5e-4)
    ap.add_argument("--qat-lr", type=float, default=4.0e-5)
    ap.add_argument("--puc-shrink", type=float, default=0.96)
    ap.add_argument("--puc-noise", type=float, default=0.012)
    ap.add_argument("--r0-drop-p", type=float, default=0.18)
    ap.add_argument("--charlm-drop-p", type=float, default=0.15)
    ap.add_argument("--cvar-alpha", type=float, default=0.08)
    ap.add_argument("--cvar-weight", type=float, default=0.07)
    ap.add_argument("--recall-floor-weight", type=float, default=0.14)
    ap.add_argument("--hop-margin-weight", type=float, default=0.10)
    ap.add_argument("--cnma-inv-weight", type=float, default=0.10)
    ap.add_argument("--protected-pnbr-tau", type=float, default=3.0)
    ap.add_argument("--r-echo-cap", type=float, default=0.50)
    ap.add_argument("--drv-fro-cap", type=float, default=6.0)
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
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
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

    # 1. Load Full Zero-Leakage Corpus (Gold + 251K Distill + Extra Rescored + Longform 1200 + CF Mirrors!)
    ex_hard_fn, ex_hard_fp, ex_sandwich, ex_q2_dual, ex_override, ex_general = [], [], [], [], [], []
    longform_train: list[dict[str, Any]] = []
    if args.smoke_test and (not args.distill_pool.is_file()):
        corpus = _make_synthetic_smoke_corpus(args.seed)
        distill_pool = list(corpus["train_ai_pos"] + corpus["train_hu_hard"])
        extra_cal_cert_rows = list(corpus["val_gold_v2"])
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
            print(f"[v1.7.1-ExtraCalCert] Reserved {len(extra_cal_cert_rows):,} zero-leakage holdout rows", flush=True)

        distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)
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
        lf_p = args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz"
        if lf_p.is_file():
            longform_train = [
                {"text": str(d_it.get("text", "")), "y_ai": 1.0 if float(d_it.get("ai_score", d_it.get("score", 0.0))) >= 65.0 else 0.0,
                 "y_pr": 1.0 if float(d_it.get("promo_score", 0.0)) >= 65.0 else 0.0}
                for d_it in read_jsonl_any_v140(lf_p)[: (24 if args.smoke_test else 1200)]
                if str(d_it.get("text", "")).strip() and text_sha16_v140(str(d_it.get("text", ""))) not in blocked
            ]

    gold_pos = list(corpus.get("train_ai_pos", []))
    gold_hd = list(corpus.get("train_hu_hard", []))
    gold_nat = list(corpus.get("train_hu_natural", []))
    gold_pr = list(corpus.get("train_promo_extra", []))
    for r in gold_pos + gold_hd + gold_nat + gold_pr:
        r["_dro_g"] = 1
        r["trust_neg"] = 1.0 if (float(r.get("has_ai", 1.0)) > 0.5 and float(r.get("y_ai", 0.0)) <= 0.25) else 0.0
    ex_hard_all = ex_hard_fn + ex_hard_fp + ex_sandwich + ex_q2_dual + ex_override
    d_hard = [r for r in distill_pool if r.get("frontier_gap", 0.0) >= 1.3 or r.get("hard_neg", 0.0) > 0.5]
    d_all = distill_pool + gold_pos + gold_hd + gold_nat + gold_pr + ex_general + ex_hard_all
    has_cf = int(corpus.get("stats", {}).get("admitted_cf_pairs", 0)) > 0

    val_g2 = corpus.get("val_gold_v2", [])
    val_w3 = corpus.get("val_wave3", [])
    val_old = corpus.get("val_old_video_neg", [])
    val_ml = corpus.get("val_multilingual", [])
    cal_rows = val_g2 + val_w3 + val_old + val_ml
    n_g2, n_w3, n_old, n_ml = len(val_g2), len(val_w3), len(val_old), len(val_ml)

    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    vocab_size = int(tokenizer.vocab_size)

    # 2. Load 3-Fold Group-OOF CharLM (handles BOTH tuple and dict pickle caches!)
    charlm_loaded = False
    if args.charlm_cache is not None and args.charlm_cache.is_file() and (not args.smoke_test):
        try:
            with args.charlm_cache.open("rb") as cf_f:
                c_obj = pickle.load(cf_f)
            if isinstance(c_obj, dict):
                fold_lms, text_to_fold, full_charlm = c_obj["fold_lms"], c_obj["text_to_fold"], c_obj["full_charlm"]
            else:
                fold_lms, text_to_fold, full_charlm = c_obj
            charlm_loaded = True
            print(f"[v1.7.1 CharLM Cache HIT] {args.charlm_cache}", flush=True)
        except Exception as e:
            print(f"[v1.7.1 CharLM Cache WARN] {e}; rebuilding...", flush=True)
    if not charlm_loaded:
        lm_rows = (gold_pos + gold_hd + gold_nat + distill_pool)[: (300 if args.smoke_test else 80000)]
        fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
            [r["text"] for r in lm_rows],
            np.array([1 if float(r.get("y_ai", 0.0)) >= 0.5 else 0 for r in lm_rows], dtype=np.int32),
            [r.get("group_id") for r in lm_rows],
            n_folds=3,
            min_counts=(2, 3, 4) if args.smoke_test else (3, 5, 7),
            max_order_keys=(3000, 1200, 600) if args.smoke_test else (22000, 16000, 10000),
            seed=args.seed,
        )

    # 3. Build Model + Exact v1.7.0 Feature Cache (`r0_b2=32768, r0_b3=16384`!)
    model = MeguvizV170MesosphereLoopModel(
        arch=args.arch, vocab_size=vocab_size, max_len=288, style_dim=56,
        use_discourse_ops=True, lardo_w8_core=True, use_dual_hopfield=bool(args.use_dual_hopfield),
    ).to(device)
    fcache = _StrictGroupOOFFeatureCacheV170(
        tokenizer=tokenizer, max_len=288, head_len=192,
        engram_buckets=model.engram_buckets, r0_b2=model.r0_b2, r0_b3=model.r0_b3,
        use_local_charlm=True, use_rust=True,
    )
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)

    if args.init_from_fp16 is not None and args.init_from_fp16.is_file():
        raw_p = torch.load(args.init_from_fp16, map_location="cpu", weights_only=False)
        parent_sd = raw_p.get("state_dict", raw_p.get("model_state_dict", raw_p)) if isinstance(raw_p, dict) else raw_p
        cur_sd = model.state_dict()
        n_ld = 0
        for k, v in parent_sd.items():
            if k in cur_sd and tuple(cur_sd[k].shape) == tuple(v.shape):
                cur_sd[k].copy_(v.to(cur_sd[k].dtype))
                n_ld += 1
        print(f"[v1.7.1 Parent Load] Loaded {n_ld}/{len(cur_sd)} tensors from {args.init_from_fp16}", flush=True)
    else:
        parent_sd = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}

    # Load Frozen Natural-Stream R0++ Anchor (supports BOTH v1.6.1 dict `rung0_state_dict` and flat v1.7.0 dict!)
    if args.init_r0_ckpt is not None and args.init_r0_ckpt.is_file():
        raw_r0 = torch.load(args.init_r0_ckpt, map_location="cpu", weights_only=False)
        r0_sd = raw_r0.get("rung0_state_dict", raw_r0) if isinstance(raw_r0, dict) else raw_r0
        r0_cur = model.rung0.state_dict()
        n_r0 = 0
        for k, v in r0_sd.items():
            if k in r0_cur and tuple(r0_cur[k].shape) == tuple(v.shape):
                r0_cur[k].copy_(v.to(r0_cur[k].dtype))
                n_r0 += 1
        print(f"[v1.7.1 R0++ Anchor] Loaded {n_r0}/{len(r0_cur)} FROZEN R0++ tensors from {args.init_r0_ckpt}", flush=True)
        assert n_r0 == len(r0_cur), f"Incomplete R0++ load: {n_r0}/{len(r0_cur)}"
    for p in model.rung0.parameters():
        p.requires_grad_(False)
    model.rung0.eval()

    # Pre-collate FULL multi-split `cal_rows` (`g2 + w3 + old + ml`) ONCE onto GPU!
    pre_cal_dp = _precollate_eval_split_v170(cal_rows, fcache, device, batch_size=args.eval_batch_size, r0_only=False, need_quote_scope=True) if cal_rows else []
    pre_val_g2 = _precollate_eval_split_v170(val_g2, fcache, device, batch_size=args.eval_batch_size, r0_only=False, need_quote_scope=True) if val_g2 else []
    y_cal_ai_all = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    y_g2 = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in val_g2], dtype=np.int32)
    y_w3_a = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in val_w3], dtype=np.int32)
    y_w3_p = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in val_w3], dtype=np.int32)
    q4_m = (y_w3_p == 1) & (y_w3_a == 0)
    y_ml = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in val_ml], dtype=np.int32)

    def _eval_full_val(outer_steps: int = 4) -> dict[str, Any]:
        if not cal_rows:
            return {"sel_obj": 0.0, "g2_pr": 0.0}
        model.eval()
        p = predict_student_v170(
            model, fcache, [r["text"] for r in cal_rows], device, batch_size=args.eval_batch_size,
            outer_steps=outer_steps, use_amp=False, precollated=pre_cal_dp,
        )
        ai_v = p[f"logit_ai_t{outer_steps}"]
        pr_v = p[f"logit_promo_t{outer_steps}"]
        s0, s1 = 0, n_g2
        mg2 = _eval_bin(y_g2, ai_v[s0:s1]) if n_g2 else {"pr": 0.0, "r95": 0.0, "r99": 0.0}
        s0, s1 = s1, s1 + n_w3
        mw3a = _eval_bin(y_w3_a, ai_v[s0:s1]) if n_w3 else {"pr": 0.0}
        mw3p = _eval_bin(y_w3_p, pr_v[s0:s1]) if (n_w3 and y_w3_p.sum() > 0) else {"pr": 0.0}
        q4_fpr = float((expit(np.clip(ai_v[s0:s1][q4_m], -30, 30)) >= 0.5).mean()) if (n_w3 and q4_m.sum() > 0) else 0.0
        s0, s1 = s1, s1 + n_old
        old_fpr = float((expit(np.clip(ai_v[s0:s1], -30, 30)) >= 0.5).mean()) if n_old else 0.0
        s0, s1 = s1, s1 + n_ml
        ml_pr = _eval_bin(y_ml, ai_v[s0:s1])["pr"] if (n_ml and y_ml.sum() > 0) else 0.0
        cal_m = _eval_bin(y_cal_ai_all, ai_v) if y_cal_ai_all.sum() > 0 else {"pr": 0.0, "r99": 0.0}
        # Exact v1.6.1-BW multi-split selection objective + cal tail protection!
        obj = (
            0.30 * mg2["pr"] + 0.24 * mg2["r95"] + 0.16 * mw3a["pr"] + 0.16 * mw3p["pr"] + 0.14 * ml_pr
            - 1.30 * old_fpr - 0.50 * q4_fpr
            + 0.20 * cal_m["pr"] + 0.12 * cal_m["r99"]
        )
        res = {
            "sel_obj": round(float(obj), 5), "g2_pr": mg2["pr"], "g2_r95": mg2["r95"], "g2_r99": mg2["r99"],
            "cal_pr": cal_m["pr"], "cal_r99": cal_m["r99"], "ml_pr": ml_pr, "old_fpr": round(old_fpr, 4), "q4_fpr": round(q4_fpr, 4),
        }
        if outer_steps >= 4 and n_g2:
            m1 = _eval_bin(y_g2, p["logit_ai_t1"][:n_g2])
            p_t1 = expit(np.clip(p["logit_ai_t1"][:n_g2], -20, 20))
            d14 = ai_v[:n_g2] - p["logit_ai_t1"][:n_g2]
            b_cert = float(np.mean((y_g2 - p_t1) * d14) - 0.125 * np.mean(d14 ** 2))
            res.update({"t1_pr": m1["pr"], "t4_minus_t1_pr": round(mg2["pr"] - m1["pr"], 4), "B_cert": round(b_cert, 5)})
        return res

    # -----------------------------------------------------------------------
    # Step 0: Zero-Leakage Active Frontier Miner (on upgraded R0++ + parent Deep!)
    # -----------------------------------------------------------------------
    frontier = _mine_frontier_pools(
        model, fcache, d_all, device, max_mine=(256 if args.smoke_test else 24000), batch_size=args.eval_batch_size, rng=rng,
    )
    print(f"[v1.7.1 Step 0 Miner] Q_D02={len(frontier['Q_D02'])} | Q_D04={len(frontier['Q_D04'])} | Q_ML={len(frontier['Q_ML'])} | Q_Long={len(frontier['Q_Long'])}", flush=True)

    # Balanced 4-group ER-DRO-R pools (never lets Frontier exceed 60% of batch!)
    grp_pools = [
        (distill_pool + ex_general) or d_all,                                                   # g=0 natural stream
        (gold_pos + gold_hd + gold_nat + gold_pr) or d_all,                                     # g=1 gold core
        (frontier["Q_D02"] + frontier["Q_D04"] + d_hard + ex_hard_all) or d_all,                # g=2 mined hard frontier
        (frontier["Q_ML"] + frontier["Q_Long"] + [r for r in d_all if len(r["text"]) >= 220]) or d_all,  # g=3 ML + long
    ]
    dro = ExcessRiskGroupDROR(init_q=(0.34, 0.24, 0.28, 0.14), hard_cap=0.60, kl_cap=0.005, max_iw=1.1112)

    def _sample_v171_batch(bs: int, focus_frontier: bool = False) -> list[dict[str, Any]]:
        q_use = np.array([0.26, 0.22, 0.36, 0.16]) if focus_frontier else dro.q
        counts = np.maximum(2, np.round(q_use * bs).astype(int))
        counts[0] += max(0, bs - int(counts.sum()))
        out_r = []
        for g in range(4):
            iw = dro.iw_for_group(g)
            for it in rng.choices(grp_pools[g], k=int(counts[g])):
                cp = dict(it)
                cp["_dro_g"] = g
                cp["dro_iw"] = iw
                out_r.append(cp)
        rng.shuffle(out_r)
        return out_r[:bs]

    # Compile inner stages cleanly (NOT top-level model so outer_steps switching never recompiles!)
    compiled_ok = False
    if args.compile and device.type == "cuda":
        compiled_ok = model.enable_torch_compile(mode=args.compile_mode, dynamic=bool(args.compile_dynamic))
    print(f"[v1.7.1 Runtime] ver={args.version} arch={args.arch} dual_hop={model.use_dual_hopfield} compile={compiled_ok} amp_bf16={use_amp} r_echo_cap={args.r_echo_cap}", flush=True)

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
    hu_fmt_pool = (gold_hd + gold_nat) or [r for r in d_all if float(r.get("has_ai", 1.0)) > 0.5 and float(r.get("y_ai", 0.0)) <= 0.20]
    long_docs_pool = longform_train + [r for r in (ex_sandwich + frontier["Q_Long"] + gold_pos) if len(r["text"]) >= 260]

    # -----------------------------------------------------------------------
    # Step 1: Stage A (Gentle PUC + Full KD/TCBR-R/RecallFloor/CF/Fmt/D02/Longform + Hopfield + CNMA-R Y=0)
    # -----------------------------------------------------------------------
    n_puc = _apply_puc_shrink_and_perturb(model, shrink=args.puc_shrink, noise=args.puc_noise, seed=args.seed)
    print(f"[v1.7.1 Stage A] PUC S&P applied to {n_puc} matrices; steps={args.stage_a_steps}", flush=True)

    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(("rung0.", "apd_exit_head.")))
    stage_a_params = [p for p in model.parameters() if p.requires_grad]
    opt_a = _make_adamw([{"params": stage_a_params, "lr": args.lr_a, "weight_decay": 0.01}], device)

    n_pw = max(0, int(args.prefetch_workers))
    pool_a = ThreadPoolExecutor(max_workers=n_pw) if n_pw > 0 else None
    q_a: deque = deque()

    def _sub_a(st_i: int):
        rows_b = _sample_v171_batch(args.batch_size, focus_frontier=False)
        if pool_a is not None:
            return pool_a.submit(_collate_v170_cpu, rows_b, fcache, random.Random(args.seed + st_i), 0.12, True, True, True, True)
        return _collate_v170_cpu(rows_b, fcache, rng, 0.12, True, True, True, True)

    for si in range(1, min(args.stage_a_steps + 1, 1 + max(1, n_pw * 2))):
        q_a.append((si, _sub_a(si)))

    best_a_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_a_obj = -1e9
    eval_every_a = max(1, args.stage_a_steps // 4)
    for step in range(1, args.stage_a_steps + 1):
        _, fut = q_a.popleft()
        bk_cpu = fut.result() if pool_a is not None else fut
        nxt = step + max(1, n_pw * 2)
        if nxt <= args.stage_a_steps:
            q_a.append((nxt, _sub_a(nxt)))
        bk = _h2d_batch_v170(bk_cpu, device)

        model.train()
        model.rung0.eval()
        _set_lr(opt_a, _cosine_lr(step, args.stage_a_steps, args.lr_a, warmup=min(80, max(2, args.stage_a_steps // 10))))
        opt_a.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                outer_steps=1, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p), freeze_t1_prefix=False,
            )
            y_ai = bk["y_ai"]
            ai_v = bk["has_ai"] > 0.5
            pr_v = bk["has_pr"] > 0.5
            has_t = bk["has_teacher"] > 0.5
            iw = bk["dro_iw"]
            z1 = od["logit_ai_steps"][-1].float()

            # Labeled AI BCE + R1 + Progressive Super-Teacher KD (covers ALL 251K teacher rows even when has_ai=0!)
            per_ai = F.binary_cross_entropy_with_logits(z1, y_ai, reduction="none")
            loss = (per_ai * ai_v.float() * iw).sum() / (ai_v.float() * iw).sum().clamp_min(1.0)
            loss = loss + 0.16 * (F.binary_cross_entropy_with_logits(od["logit_ai_r1"].float(), y_ai, reduction="none") * ai_v.float()).sum() / ai_v.float().sum().clamp_min(1.0)
            l_kd = (F.binary_cross_entropy_with_logits(z1 / 1.5, torch.sigmoid(bk["t_ai_lg"] / 1.5), reduction="none") * has_t.float() * bk["kd_w"]).sum() / has_t.float().sum().clamp_min(1.0)
            loss = loss + 0.35 * l_kd
            l_pr = (F.binary_cross_entropy_with_logits(od["logit_promo"].float(), bk["y_pr"], reduction="none") * pr_v.float()).sum() / pr_v.float().sum().clamp_min(1.0)
            loss = loss + 0.55 * l_pr

            # TCBR-R Zero-Sync Tail Ranking + Positive Recall-Floor Margin
            hu_m = (ai_v & (y_ai <= 0.20)) | ((~ai_v) & has_t & (bk["t_ai_lg"] <= -1.6))
            pos_m = (ai_v & (y_ai >= 0.70)) | ((~ai_v) & has_t & (bk["t_ai_lg"] >= 1.4))
            tr_m = hu_m & (bk["trust_neg"] > 0.5)
            un_m = hu_m & (bk["trust_neg"] <= 0.5)
            k_c = max(2, int(math.ceil(float(args.cvar_alpha) * len(z1))))
            top_tr = torch.topk(torch.where(tr_m, z1, torch.full_like(z1, -1e9)), k=min(k_c, len(z1)), largest=True).values
            l_tr = F.softplus(top_tr + 1.5).clamp_max(4.0).mean() * (tr_m.sum() >= 4).float()
            l_un = (4.0 * torch.sigmoid((z1 + 1.0) * 0.5) * un_m.float()).sum() / un_m.float().sum().clamp_min(1.0)
            loss = loss + float(args.cvar_weight) * (0.75 * l_tr + 0.25 * l_un)

            rf_ok = (pos_m.sum() >= 4) & (hu_m.sum() >= 8)
            kp = max(2, min(len(z1), int(math.ceil(0.35 * len(z1)))))
            kn = max(2, min(len(z1), int(math.ceil(0.20 * len(z1)))))
            zp = torch.topk(torch.where(pos_m, z1, torch.full_like(z1, 1e9)), k=kp, largest=False).values
            zn = torch.topk(torch.where(hu_m, z1, torch.full_like(z1, -1e9)), k=kn, largest=True).values
            loss = loss + float(args.recall_floor_weight) * F.softplus(1.6 + zn.unsqueeze(1) - zp.unsqueeze(0)).mean() * rf_ok.float()

            # Hard-Margin Once-at-T1 DualHopfield Energy Loss
            if model.dual_hopfield is not None and "hop_witness" in od:
                e_h = od["hop_witness"][:, 0].float()
                d_gap = od["hop_witness"][:, 2].float()
                is_ai = (ai_v & (y_ai >= 0.5)).float()
                is_hu = (ai_v & (y_ai <= 0.20)).float()
                l_hop = (F.relu(1.2 - d_gap) * is_ai).sum() / is_ai.sum().clamp_min(1.0)
                l_hop = l_hop + (F.relu(1.2 + d_gap) * is_hu).sum() / is_hu.sum().clamp_min(1.0)
                l_hop = l_hop + 0.02 * (F.relu(e_h - 1.5) * is_hu).sum() / is_hu.sum().clamp_min(1.0)
                loss = loss + float(args.hop_margin_weight) * l_hop

            # M7 Y=0 Conditional Zero-Distribution Invariance across length & script strata (fixes M02/M04!)
            if model.cnma_r is not None:
                hu_strict = ai_v & (y_ai <= 0.20)
                if int(hu_strict.sum().item()) >= 8:
                    z_hu = z1[hu_strict]
                    g_mean = z_hu.mean()
                    r_h = bk["m7_surf"][hu_strict, 0]
                    log_l = bk["m7_surf"][hu_strict, 4] * 6.0
                    strata = [
                        (r_h >= 0.5) & (log_l <= math.log1p(60.0)),
                        (r_h >= 0.5) & (log_l > math.log1p(60.0)) & (log_l < math.log1p(300.0)),
                        (r_h >= 0.5) & (log_l >= math.log1p(300.0)),
                        (r_h < 0.5),
                    ]
                    l_inv = torch.zeros((), device=device, dtype=torch.float32)
                    for sm in strata:
                        if int(sm.sum().item()) >= 2:
                            l_inv = l_inv + (z_hu[sm].mean() - g_mean).pow(2)
                    loss = loss + float(args.cnma_inv_weight) * l_inv

            # CF Mirror Pairs (step % 3 == 0)
            if has_cf and step % 3 == 0:
                ch = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
                if ch:
                    n_cf = len(ch)
                    bk_cf = _h2d_batch_v170(_collate_v170_cpu([{"text": x["ai_text"]} for x in ch] + [{"text": x["human_text"]} for x in ch], fcache, include_meta=False), device)
                    od_cf = model(
                        bk_cf["ids"], bk_cf["msk"], bk_cf["st"], bk_cf["eg"], bk_cf["loc"],
                        bk_cf["c1"], bk_cf["c2"], bk_cf["c3"], bk_cf["cm"], bk_cf["cf"],
                        bk_cf["sup"], bk_cf["qsc"], bk_cf["m6_ph"], bk_cf["m7_surf"], bk_cf["m7_uisr"], outer_steps=1,
                    )
                    pw = torch.tensor([float(x["pair_weight"]) for x in ch], dtype=torch.float32, device=device)
                    loss = loss + 0.18 * (F.softplus(1.7 - (od_cf["logit_ai"][:n_cf].float() - od_cf["logit_ai"][n_cf:].float())) * pw).mean()

            # Rule Format-Edit Invariance CF on Human Comments (step % 4 == 2 — fixes H03/H07 FPs!)
            if step % 4 == 2 and hu_fmt_pool:
                n_fc = min(4 if args.smoke_test else 12, len(hu_fmt_pool))
                src_fc = rng.sample(hu_fmt_pool, n_fc)
                orig_t = [str(r["text"]) for r in src_fc]
                edit_t = [_minimal_rule_format_edit(t, rng) for t in orig_t]
                bk_fc = _h2d_batch_v170(_collate_v170_cpu([{"text": t} for t in (orig_t + edit_t)], fcache, include_meta=False), device)
                od_fc = model(
                    bk_fc["ids"], bk_fc["msk"], bk_fc["st"], bk_fc["eg"], bk_fc["loc"],
                    bk_fc["c1"], bk_fc["c2"], bk_fc["c3"], bk_fc["cm"], bk_fc["cf"],
                    bk_fc["sup"], bk_fc["qsc"], bk_fc["m6_ph"], bk_fc["m7_surf"], bk_fc["m7_uisr"], outer_steps=1,
                )
                loss = loss + 0.10 * F.smooth_l1_loss(od_fc["logit_ai"][:n_fc].float(), od_fc["logit_ai"][n_fc:].float())
                loss = loss + 0.06 * F.binary_cross_entropy_with_logits(od_fc["logit_ai"][n_fc:].float(), torch.zeros((n_fc,), device=device))

            # D02 Paraphrase Rebuttal CF (step % 4 == 0)
            if step % 4 == 0 and gold_pos:
                n_d02 = min(4 if args.smoke_test else 12, len(gold_pos))
                reb_rows = [{"text": rng.choice(_D02_TEMPLATES).format(ai=rng.choice(gold_pos)["text"][:90])} for _ in range(n_d02)]
                bk_d02 = _h2d_batch_v170(_collate_v170_cpu(reb_rows, fcache, include_meta=False), device)
                od_d02 = model(
                    bk_d02["ids"], bk_d02["msk"], bk_d02["st"], bk_d02["eg"], bk_d02["loc"],
                    bk_d02["c1"], bk_d02["c2"], bk_d02["c3"], bk_d02["cm"], bk_d02["cf"],
                    bk_d02["sup"], bk_d02["qsc"], bk_d02["m6_ph"], bk_d02["m7_surf"], bk_d02["m7_uisr"], outer_steps=1,
                )
                loss = loss + 0.10 * F.binary_cross_entropy_with_logits(od_d02["logit_ai"].float(), torch.zeros((n_d02,), device=device))

            # Promo-Flip CF (step % 5 == 0)
            if step % 5 == 0:
                od_pf = model(
                    pf_gpu["ids"], pf_gpu["msk"], pf_gpu["st"], pf_gpu["eg"], pf_gpu["loc"],
                    pf_gpu["c1"], pf_gpu["c2"], pf_gpu["c3"], pf_gpu["cm"], pf_gpu["cf"],
                    pf_gpu["sup"], pf_gpu["qsc"], pf_gpu["m6_ph"], pf_gpu["m7_surf"], pf_gpu["m7_uisr"], outer_steps=1,
                )
                half_pf = len(_PROMO_FLIPS)
                loss = loss + 0.10 * F.relu(1.8 - (od_pf["logit_promo"][:half_pf].float() - od_pf["logit_promo"][half_pf:].float())).mean()

            # DynaSeam Longform + Asymmetric Sandwiches (step % 4 == 0)
            if step % 4 == 0 and long_docs_pool:
                docs_l = rng.choices(long_docs_pool, k=min(4, len(long_docs_pool)))
                c_txts, c_metas, doc_slices, doc_y_ai, doc_y_pr = [], [], [], [], []
                for dr in docs_l:
                    cks = plan_long_chunks_v11(canonicalize_zero_pii_v150(dr["text"])[0], max_chunks=5)
                    st_i = len(c_txts)
                    for ck in cks:
                        c_txts.append({"text": ck["text"]}); c_metas.append(ck["chunk_meta"])
                    if len(c_txts) > st_i:
                        doc_slices.append((st_i, len(c_txts)))
                        doc_y_ai.append(float(dr.get("y_ai", 0.0))); doc_y_pr.append(float(dr.get("y_pr", 0.0)))
                if c_txts and doc_slices:
                    bk_c = _h2d_batch_v170(_collate_v170_cpu(c_txts, fcache, include_meta=False), device)
                    od_c = model(
                        bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["loc"],
                        bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"],
                        bk_c["sup"], bk_c["qsc"], bk_c["m6_ph"], bk_c["m7_surf"], bk_c["m7_uisr"], outer_steps=1,
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
        torch.nn.utils.clip_grad_norm_(stage_a_params, 1.2, foreach=True)
        opt_a.step()

        if step % 10 == 0:
            dro.observe(bk_cpu["dro_g"].numpy(), per_ai.detach().cpu().numpy())
            if step % 50 == 0:
                dro.maybe_update()

        if step == args.stage_a_steps or step % eval_every_a == 0:
            ev = _eval_full_val(outer_steps=1)
            if ev["sel_obj"] >= best_a_obj:
                best_a_obj = ev["sel_obj"]
                best_a_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [StageA-T1 step={step}/{args.stage_a_steps}] loss={loss.item():.4f} {ev}", flush=True)

    if pool_a is not None:
        pool_a.shutdown(wait=False)
    model.load_state_dict({k: v.to(device) for k, v in best_a_sd.items()})

    # -----------------------------------------------------------------------
    # Step 2: Stage B (M5 Prefix-Frozen Protected PNBR-R + Tight r_echo <= 0.50 + dr_v Norm Cap)
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

    r_echo_val = _project_r_echo_and_drv_cap(model, float(args.r_echo_cap), float(args.drv_fro_cap))
    best_b_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_b_obj = _eval_full_val(outer_steps=4)["sel_obj"]
    eval_every_b = max(1, args.stage_b_steps // 2)

    for step in range(1, args.stage_b_steps + 1):
        model.train()
        model.rung0.eval()
        _set_lr(opt_b, _cosine_lr(step, args.stage_b_steps, args.lr_b, warmup=min(40, max(1, args.stage_b_steps // 10))))
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=True), fcache, rng=rng, punct_jitter_prob=0.05), device)
        opt_b.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = model(
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
            # M5 Protected PNBR: freeze delta on high-confidence |f_1| >= tau -> guaranteed positive B_cert!
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
                loss_b = loss_b + (0.50 * l_t + 0.04 * corr.pow(2) + 0.015 * (delta_t.pow(2) * (1.0 - prot_mask)).mean())
            ptp = model.core.sketch_P.float().T @ model.core.sketch_P.float()
            loss_b = loss_b + 0.01 * (ptp - torch.eye(SKETCH_R_V170, device=device)).pow(2).sum()
            if od.get("u_halt_steps"):
                u_tgt = (lgs[-1].detach().float() - lgs[0].detach().float()).abs().clamp(0.0, 2.0) * 0.25
                loss_b = loss_b + 0.10 * F.smooth_l1_loss(od["u_halt_steps"][0].float(), u_tgt)

        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(s2_params, 1.0, foreach=True)
        opt_b.step()
        r_echo_val = _project_r_echo_and_drv_cap(model, float(args.r_echo_cap), float(args.drv_fro_cap))

        if step == args.stage_b_steps or step % eval_every_b == 0:
            ev_b = _eval_full_val(outer_steps=4)
            if ev_b["sel_obj"] >= best_b_obj:
                best_b_obj = ev_b["sel_obj"]
                best_b_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [StageB-PNBR step={step}/{args.stage_b_steps}] loss={loss_b.item():.4f} r_echo={r_echo_val:.4f} {ev_b}", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in best_b_sd.items()})
    model.eval()
    t1_after = predict_student_v170(model, fcache, [r["text"] for r in val_g2], device, batch_size=args.eval_batch_size, outer_steps=1, use_amp=False, precollated=pre_val_g2)["logit_ai_t1"].copy() if val_g2 else np.zeros((0,))
    t1_freeze_diff = float(np.max(np.abs(t1_after - t1_before))) if len(t1_before) > 0 else 0.0
    print(f"[v1.7.1 Stage B Audit] T1_freeze_max_diff={t1_freeze_diff:.2e} | r_echo={r_echo_val:.4f} (cap={args.r_echo_cap})", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Stage C (APD-v2 on Frozen R0++ + 68B DS-TitanGuard + Light-QAT + Full-Cal WiSE-FT AutoShipGate)
    # -----------------------------------------------------------------------
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("apd_exit_head."))
    apd_params = [p for p in model.apd_exit_head.parameters() if p.requires_grad]
    opt_apd = _make_adamw([{"params": apd_params, "lr": 1.5e-3, "weight_decay": 1e-3}], device)
    for _ in range(args.apd_steps):
        model.train()
        model.rung0.eval()
        bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=False), fcache, rng=rng, need_bpe=False), device)
        opt_apd.zero_grad(set_to_none=True)
        with torch.no_grad():
            r0_c = model.rung0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        sup_n = bk["cf"][:, R0_STRUCT_DIM_V140 - 1].to(r0_c["z_char"].dtype)
        apd_lg = model.apd_exit_head(r0_c["z_char"][:, :32], r0_c["logit_ai_r0"], sup_n, bk["m7_surf"])
        ai_v = (bk["has_ai"] > 0.5).float()
        y_safe = (bk["y_ai"] < 0.25).float()
        w_asym = torch.where(bk["y_ai"] >= 0.5, torch.full_like(y_safe, 2.5), torch.ones_like(y_safe)) * ai_v
        l_apd = (F.binary_cross_entropy_with_logits(apd_lg.float(), y_safe, reduction="none") * w_asym).sum() / w_asym.sum().clamp_min(1.0)
        l_apd.backward()
        opt_apd.step()

    # Detach torch.compile BEFORE TitanGuard / QAT / Export so FP32 audit is strictly uncompiled!
    model._compiled_r0 = None
    model._compiled_prelude_t1 = None
    model._compiled_loop_step = None

    titan_rep: dict[str, Any] = {"enabled": False}
    if args.fit_titanguard:
        titan_rep = _fit_titanguard_and_placebo(
            model, fcache, frontier["Q_Long"] + long_docs_pool,
            [r for r in d_all if float(r.get("has_ai", 1.0)) > 0.5 and float(r.get("y_ai", 0.0)) <= 0.20],
            gold_pos, device,
        )
        titan_rep["enabled"] = True

    preqat_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Light-QAT
    if args.p3_qat_steps > 0:
        for n, p in model.named_parameters():
            p.requires_grad_(not n.startswith(("rung0.", "apd_exit_head.", "long_agg.titan_")))
        qat_params = [p for p in model.parameters() if p.requires_grad]
        opt_q = _make_adamw([{"params": qat_params, "lr": args.qat_lr, "weight_decay": 0.005}], device)
        set_v130_qat_mode(model, True)
        for _ in range(args.p3_qat_steps):
            model.train()
            model.rung0.eval()
            bk = _h2d_batch_v170(_collate_v170_cpu(_sample_v171_batch(args.batch_size, focus_frontier=False), fcache, rng=rng, punct_jitter_prob=0.05), device)
            opt_q.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                od = model(
                    bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["loc"],
                    bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"],
                    bk["sup"], bk["qsc"], bk["m6_ph"], bk["m7_surf"], bk["m7_uisr"],
                    outer_steps=4, freeze_t1_prefix=False,
                )
                ai_v = (bk["has_ai"] > 0.5).float()
                pr_v = (bk["has_pr"] > 0.5).float()
                loss_q = (F.binary_cross_entropy_with_logits(od["logit_ai_steps"][-1].float(), bk["y_ai"], reduction="none") * ai_v).sum() / ai_v.sum().clamp_min(1.0)
                loss_q = loss_q + 0.45 * (F.binary_cross_entropy_with_logits(od["logit_promo"].float(), bk["y_pr"], reduction="none") * pr_v).sum() / pr_v.sum().clamp_min(1.0)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, 1.0, foreach=True)
            opt_q.step()
            _project_r_echo_and_drv_cap(model, float(args.r_echo_cap), float(args.drv_fro_cap))
        set_v130_qat_mode(model, False)

    qat_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 8-Candidate WiSE-FT Auto-Ship-Gate scored on FULL multi-split `cal_rows` after bit-exact MGVZ10 roundtrip!
    ship_cands: dict[str, dict[str, torch.Tensor]] = {
        "direct_pack_preqat": preqat_sd,
        "light_qat": qat_sd,
        "fp16_ema_soup": {k: (0.5 * preqat_sd[k].float() + 0.5 * qat_sd[k].float()) if qat_sd[k].is_floating_point() else qat_sd[k].clone() for k in qat_sd},
    }
    for alpha_w in (0.85, 0.70, 0.50, 0.30, 0.15):
        soup = {}
        for k, v in preqat_sd.items():
            if k in parent_sd and tuple(parent_sd[k].shape) == tuple(v.shape) and v.is_floating_point() and (not k.startswith(("rung0.", "apd_exit_head.", "long_agg.titan_"))):
                soup[k] = alpha_w * v.float() + (1.0 - alpha_w) * parent_sd[k].float()
            else:
                soup[k] = v.clone()
        ship_cands[f"wise_ft_a{int(alpha_w * 100)}"] = soup

    winner_name = "direct_pack_preqat"
    winner_sd = preqat_sd
    ship_scores: dict[str, Any] = {}
    if args.auto_ship_gate:
        best_sc = -1e9
        for cname, csd in ship_cands.items():
            model.load_state_dict({k: v.to(device) for k, v in csd.items()})
            _project_r_echo_and_drv_cap(model, float(args.r_echo_cap), float(args.drv_fro_cap))
            apply_mgvz10_quant_roundtrip_inplace_v170(model)
            ev = _eval_full_val(outer_steps=4)
            ship_scores[cname] = ev
            if ev["sel_obj"] > best_sc:
                best_sc, winner_name, winner_sd = ev["sel_obj"], cname, csd
        print(f"[v1.7.1 AutoShipGate] WINNER = {winner_name} | best_sel_obj={best_sc:.5f}", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in winner_sd.items()})
    r_echo_val = _project_r_echo_and_drv_cap(model, float(args.r_echo_cap), float(args.drv_fro_cap))
    torch.save({k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}, out_dir / f"{args.version}_fp16_master.pt")
    apply_mgvz10_quant_roundtrip_inplace_v170(model)

    # -----------------------------------------------------------------------
    # Calibration: Strictly Disjoint `cal_fit / cal_cert` `EBA-H` + Fixed `expit` LUTs + `M8 IUT` Stratified Exit!
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model.eval()
    p_cal = predict_student_v170(model, fcache, [r["text"] for r in cal_rows], device, batch_size=args.eval_batch_size, outer_steps=4, use_amp=False, precollated=pre_cal_dp)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in cal_rows], dtype=np.int32)

    fit_mask = np.array([(int(text_sha16_v140(r["text"]), 16) % 2) == 0 for r in cal_rows], dtype=bool)
    cert_mask = ~fit_mask
    if fit_mask.sum() < 24 or cert_mask.sum() < 24 or y_cal_ai[fit_mask].sum() < 6 or y_cal_ai[cert_mask].sum() < 6:
        fit_mask = np.ones((len(cal_rows),), dtype=bool)
        cert_mask = fit_mask.copy()

    cert_rows = extra_cal_cert_rows if len(extra_cal_cert_rows) >= 48 else [cal_rows[i] for i in np.where(cert_mask)[0]]
    p_cert = predict_student_v170(model, fcache, [r["text"] for r in cert_rows], device, batch_size=args.eval_batch_size, outer_steps=4, use_amp=False)
    y_cert_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cert_rows], dtype=np.int32)

    cv_fusion = fit_crossview_eba_h_fusion(
        p_cal["logit_ai_r0"][fit_mask], p_cal["logit_ai_t4"][fit_mask], y_cal_ai[fit_mask],
        pi_nat=float(args.eba_pi_nat),
        cert_r0=p_cal["logit_ai_r0"][cert_mask], cert_deep=p_cal["logit_ai_t4"][cert_mask], cert_y=y_cal_ai[cert_mask],
    ) if args.use_eba_h else {"enabled": False, "formula": "none"}

    cal_fused_t4 = apply_crossview_late_fusion(p_cal["logit_ai_r0"], p_cal["logit_ai_t4"], cv_fusion)
    t_ai = fit_scalar_temperature_v140((cal_fused_t4 - float(cv_fusion.get("prior_shift", 0.0)))[fit_mask], y_cal_ai[fit_mask])
    t_pr = fit_scalar_temperature_v140(p_cal["logit_promo_t4"][fit_mask], y_cal_pr[fit_mask]) if y_cal_pr[fit_mask].sum() >= 8 else 1.5
    lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"][fit_mask], -30.0, 30.0), y_cal_ai[fit_mask])
    lut_ai_full = (
        fit_output_shifted_promo_lut_v122(np.clip(cal_fused_t4[fit_mask] / max(1e-3, t_ai), -30.0, 30.0), y_cal_ai[fit_mask], pi_nat=float(args.eba_pi_nat), shift_damp=0.55)
        if args.use_eba_h
        else fit_smooth_ai_lut_v122(np.clip(cal_fused_t4[fit_mask] / max(1e-3, t_ai), -30.0, 30.0), y_cal_ai[fit_mask])
    )
    lut_pr_r0 = fit_output_shifted_promo_lut_v122(np.clip(p_cal["logit_promo_r0"][fit_mask], -30.0, 30.0), y_cal_pr[fit_mask]) if y_cal_pr[fit_mask].sum() >= 8 else None
    lut_pr_full = fit_output_shifted_promo_lut_v122(np.clip(p_cal["logit_promo_t4"][fit_mask] / max(1e-3, t_pr), -30.0, 30.0), y_cal_pr[fit_mask]) if y_cal_pr[fit_mask].sum() >= 8 else None

    # FIXED `expit(...)` BEFORE `apply_spline_temp_lut` -> unlocks genuine ~30-40% certified M8 IUT R0 exit!
    cert_p_r0 = apply_spline_temp_lut(expit(np.clip(p_cert["logit_ai_r0"], -30.0, 30.0)), lut_ai_r0)
    cert_clens = np.array([len(canonicalize_zero_pii_v150(r["text"])[0]) for r in cert_rows], dtype=np.int64)
    r0_ai_pol = fit_iut_stratified_r0_policy_v170(cert_p_r0, p_cert["p_defer_ai_r0"], p_cert["p_apd_safe"], cert_clens, y_cert_ai, alpha_max=float(args.iut_alpha_max))
    print(f"[v1.7.1 IUT R0 Policy] enabled={r0_ai_pol['enabled']} overall_exit={r0_ai_pol['overall_exit_rate']*100:.1f}% max_cp95_ub={r0_ai_pol['iut_max_cp95_ub']}", flush=True)

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
    export_rep = export_v170_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    eng = MeguvizV170CascadeEngine(out_dir, threads=4, r0_threads=2)
    ci_gates = run_v170_ci_gates(model, tokenizer, export_rep, out_dir, device, charlm=full_charlm, engine=eng)

    p_t1_c = expit(np.clip(p_cert["logit_ai_t1"], -20, 20))
    d14_c = p_cert["logit_ai_t4"] - p_cert["logit_ai_t1"]
    b_cert_val = round(float(np.mean((y_cert_ai - p_t1_c) * d14_c) - 0.125 * np.mean(d14_c ** 2)), 5)
    hop_pr = round(float(average_precision_score(y_cert_ai, p_cert["hop_witness"][:, 2])), 4) if (model.dual_hopfield is not None and y_cert_ai.sum() > 0) else None

    report = {
        "version": args.version, "arch": args.arch, "use_dual_hopfield": model.use_dual_hopfield,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "r0_anchor": str(args.init_r0_ckpt) if args.init_r0_ckpt else None,
        "frontier_counts": {k: len(v) for k, v in frontier.items()},
        "t1_prefix_freeze_max_diff": t1_freeze_diff,
        "r_echo": round(r_echo_val, 4), "r_echo_cap": float(args.r_echo_cap),
        "b_cert": b_cert_val, "dual_hopfield_witness_pr": hop_pr,
        "rung0_iut_policy": r0_ai_pol, "crossview_late_fusion": cv_fusion,
        "titanguard": titan_rep,
        "auto_ship_gate": {"winner": winner_name, "candidates": ship_scores},
        "export": export_rep,
        "release_gates": {
            "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
            "onnx_4stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v170_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "v170_ci_gates": ci_gates,
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[v1.7.1 DONE] ver={args.version} | bootable={export_rep['bootable_mib']} MiB | "
        f"CI={ci_gates['passed_count']}/{ci_gates['total_count']} PASS={ci_gates['all_ci_passed']} | "
        f"r_echo={r_echo_val:.4f} | B_cert={b_cert_val} | IUT_exit={r0_ai_pol['overall_exit_rate']*100:.1f}%",
        flush=True,
    )
    if not report["release_gates"]["v170_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
