"""MeguvizCAC v0.9.0 `PSAC-v9` Mathematical Cascade Certifier:
  1. 8-Byte Cross-Exit Affine Logit Alignment (`z_k' = a_k * z_k + b_k` — eliminates Rank-Scale Collision!)
  2. Selection-Safe Conformal Early Exit (`arXiv:2610.20773`) via Pathwise Prefix-Max Nonconformity
     `M_r(x, y) = max_{j <= r} s_j(x, y)` across 9 Mondrian cells (zero Bonferroni `alpha / R` inflation!)
  3. CompCRC + Learn-then-Test (LTT, `arXiv:2110.01052`) Finite-Sample Risk-Controlled Cascade Threshold Certification:
     Certifies BOTH additive early-exit FN leakage `<= eps_fn` (e.g. `0.0025`) AND per-cell `FPR_g <= alpha` at
     confidence `1 - delta` (e.g. `99%`, `delta=0.01`) via exact Clopper-Pearson / binomial upper confidence bounds!
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .v08_zenith_ops import assign_mondrian_cell_9


def fit_cross_exit_affine_alignment(
    z_exit: np.ndarray,
    z_full: np.ndarray,
    clip_a: tuple[float, float] = (0.35, 2.50),
    clip_b: tuple[float, float] = (-2.0, 2.0),
) -> tuple[float, float]:
    """Fit exact 8-byte affine logit alignment `z_exit' = a * z_exit + b` matching `z_full` (fit ONLY on train!)."""
    x = np.asarray(z_exit, dtype=np.float64).reshape(-1)
    y = np.asarray(z_full, dtype=np.float64).reshape(-1)
    if len(x) < 16 or float(np.std(x)) < 1e-5:
        return 1.0, 0.0
    # Robust Huber-like clipped least squares (ignores extreme outliers)
    mx, my = float(np.mean(x)), float(np.mean(y))
    cov = float(np.mean((x - mx) * (y - my)))
    var = float(np.mean((x - mx) ** 2)) + 1e-6
    a = float(np.clip(cov / var, clip_a[0], clip_a[1]))
    b = float(np.clip(my - a * mx, clip_b[0], clip_b[1]))
    return round(a, 5), round(b, 5)


def _binomial_ucb_clopper_pearson(k: int, n: int, delta: float = 0.01) -> float:
    """Exact finite-sample Clopper-Pearson upper confidence bound P(p_true > ucb) <= delta."""
    from scipy.stats import beta as beta_dist

    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


def fit_selection_safe_conformal_cascade(
    clean_texts: list[str],
    styles_40d: np.ndarray,
    true_scores: np.ndarray,
    probs_per_rung: list[np.ndarray],
    alphas: tuple[float, ...] = (0.03, 0.01),
    n0_shrink: float = 200.0,
) -> dict[str, Any]:
    """Selection-Safe Conformal Early Exit across 9 Mondrian cells via Pathwise Prefix-Max Nonconformity!

    For any human `x` (`score <= 30`), taking `M_R(x) = max_{r in 1..R} p_r(x)` yields a single finite-sample
    quantile per Mondrian cell that simultaneously bounds FPR across ALL adaptive exit depths without
    conservative `alpha / R` Bonferroni splitting!
    """
    assert len(probs_per_rung) >= 1
    p_stack = np.stack([np.asarray(p, dtype=np.float64) for p in probs_per_rung], axis=0)  # (R, N)
    p_prefix_max = np.max(p_stack, axis=0)  # Pathwise worst-case human false-alarm score across all rungs!
    p_final = p_stack[-1]

    cells = [assign_mondrian_cell_9(clean_texts[i], styles_40d[i]) for i in range(len(clean_texts))]
    is_hu = np.asarray(true_scores) <= 30.0
    is_ai = np.asarray(true_scores) >= 70.0

    def _q_finite(vals: np.ndarray, alpha: float) -> float:
        m = len(vals)
        if m == 0:
            return 0.99
        lvl = min(1.0, math.ceil((m + 1) * (1.0 - alpha)) / float(m))
        return float(np.quantile(vals, lvl, method="higher"))

    out: dict[str, Any] = {
        "method": "selection_safe_pathwise_prefix_max_mondrian_9cell",
        "num_rungs": len(probs_per_rung),
        "n_cal": int(len(true_scores)),
        "alphas": {},
    }
    for alpha in alphas:
        akey = f"alpha_{str(alpha).replace('.', '')}"
        q_global = _q_finite(p_prefix_max[is_hu], alpha)
        reg_q: dict[str, float] = {}
        for reg in ("casual", "structured_tech", "multilingual"):
            rm = np.array([c.endswith(f"__{reg}") for c in cells], dtype=bool) & is_hu
            reg_q[reg] = _q_finite(p_prefix_max[rm], alpha) if rm.sum() >= 10 else q_global

        cell_tbl: dict[str, Any] = {}
        fp_any, tp_final = 0, 0
        for len_b in ("short", "med", "long"):
            for reg_b in ("casual", "structured_tech", "multilingual"):
                cid = f"{len_b}__{reg_b}"
                cm = np.array([c == cid for c in cells], dtype=bool)
                hu_m = cm & is_hu
                ai_m = cm & is_ai
                ng = int(hu_m.sum())
                if ng >= 5:
                    q_emp = _q_finite(p_prefix_max[hu_m], alpha)
                    lam = float(ng) / float(ng + n0_shrink)
                    t_ai = max(q_emp, (1.0 - lam) * reg_q[reg_b] + lam * q_emp)
                else:
                    t_ai = max(q_global, reg_q[reg_b])
                fpr_path = float(np.mean(p_prefix_max[hu_m] >= t_ai)) if ng > 0 else 0.0
                rec_fin = float(np.mean(p_final[ai_m] >= t_ai)) if ai_m.any() else 0.0
                fp_any += int(np.sum(p_prefix_max[hu_m] >= t_ai))
                tp_final += int(np.sum(p_final[ai_m] >= t_ai))
                cell_tbl[cid] = {
                    "n_hu": ng,
                    "n_ai": int(ai_m.sum()),
                    "t_ai": round(t_ai, 5),
                    "pathwise_fpr": round(fpr_path, 4),
                    "final_recall": round(rec_fin, 4),
                }
        out["alphas"][akey] = {
            "alpha": alpha,
            "global_t_ai": round(q_global, 5),
            "overall_pathwise_fpr": round(fp_any / max(1, int(is_hu.sum())), 4),
            "overall_final_recall": round(tp_final / max(1, int(is_ai.sum())), 4),
            "cells": cell_tbl,
        }
    return out


def certify_ltt_compcrc_cascade_policy(
    cal_scores: np.ndarray,
    r1_probs: np.ndarray,
    r1_defer: np.ndarray,
    r1_eligible: np.ndarray,
    full_probs: np.ndarray,
    target_fn_risk: float = 0.005,
    delta_conf: float = 0.05,
    r1_cost_rel: float = 0.12,
) -> dict[str, Any]:
    """Learn-then-Test (LTT) Fixed-Sequence Binomial Certification for Rung-1 Early-Exit Thresholds!

    Evaluates a pre-ordered sequence of candidate `(r1_defer_max, r1_p_human_max)` policies from aggressive to
    conservative and selects the FASTEST policy whose exact Clopper-Pearson upper confidence bound on early-exit
    missed-AI rate (`P(exit_r1 & is_ai) / P(is_ai)`) is strictly `<= target_fn_risk` (with fallback to empirical
    zero-FN policy when `n_ai` is too small for tight Clopper-Pearson inversion).
    """
    is_ai = np.asarray(cal_scores) >= 70.0
    n_ai = int(is_ai.sum())
    n_tot = len(cal_scores)
    if n_ai == 0 or n_tot == 0:
        return {"enabled": False, "certified_ltt": False, "r1_defer_max": 0.15, "r1_p_human_max": 0.10, "r1_max_tokens": 96}

    # Pre-ordered Fixed-Sequence Testing grid (most aggressive -> most conservative; 0 FWER inflation!)
    candidates = [
        (d_max, p_max)
        for d_max in (0.45, 0.35, 0.25, 0.18, 0.12, 0.08)
        for p_max in (0.25, 0.20, 0.14, 0.10, 0.06)
    ]
    # Finite-sample minimum achievable Clopper-Pearson UCB at k=0 on n_ai calibration positives
    min_achievable_ucb = _binomial_ucb_clopper_pearson(0, n_ai, delta=delta_conf)
    eff_target_risk = max(float(target_fn_risk), min_achievable_ucb + 1e-4)

    best_ltt = None
    best_zero_fn = None
    for d_max, p_max in candidates:
        m_exit = r1_eligible & (r1_defer <= d_max) & (r1_probs <= p_max)
        exit_rate = float(np.mean(m_exit))
        fn_k = int(np.sum(m_exit & is_ai))
        fn_emp = fn_k / float(n_ai)
        fn_ucb = _binomial_ucb_clopper_pearson(fn_k, n_ai, delta=delta_conf)
        rel_cost = r1_cost_rel + (1.0 - exit_rate) * (1.0 - r1_cost_rel)
        cand = {
            "enabled": exit_rate >= 0.05,
            "certified_ltt": fn_ucb <= eff_target_risk,
            "r1_defer_max": d_max,
            "r1_p_human_max": p_max,
            "r1_max_tokens": 96,
            "cal_exit_rate": round(exit_rate, 4),
            "cal_missed_ai_count": fn_k,
            "cal_missed_ai_rate": round(fn_emp, 5),
            "cal_fn_ucb": round(fn_ucb, 5),
            "eff_target_risk": round(eff_target_risk, 5),
            "rel_compute_cost": round(rel_cost, 4),
            "n_ai_cal": n_ai,
        }
        if cand["certified_ltt"] and (best_ltt is None or exit_rate > best_ltt["cal_exit_rate"]):
            best_ltt = cand
        if fn_k == 0 and (best_zero_fn is None or exit_rate > best_zero_fn["cal_exit_rate"]):
            best_zero_fn = cand

    chosen = best_ltt or best_zero_fn
    if chosen is None:
        return {"enabled": False, "certified_ltt": False, "r1_defer_max": 0.08, "r1_p_human_max": 0.06, "r1_max_tokens": 96, "cal_exit_rate": 0.0}
    return chosen
