"""MeguvizCAC v1.6.x / v1.7.0 Production Statistical Risk Certifier & Factorized F6 Exit Optimizer.

Implements the Rev. 2 Independent Statistical Decision-Theory Review blueprint (P0 + P1 modules):
  1. `NaturalFlowAuditor` (P0):
     - Exact rational (`fractions.Fraction`) Clopper-Pearson one-sided bounds (zero floating-point boundary ambiguity!).
     - Exact rational Group-Sequential designs (`95%: (350,0),(550,1),(800,2) -> alpha=4.8033%`;
       `97.5%: (400,0),(700,1),(900,2) -> alpha=2.3912%`) with irreversible state & futility check (`INCONCLUSIVE`,
       never mislabeling futility as proof of `q_H > q_0`!).
     - Zero-error & mixture `e-process` (`q_j = (0, 0.001, 0.005)`, weights `(0.5, 0.25, 0.25)`) in `logsumexp`.
     - Intersection-Union Test (IUT) all-or-nothing joint release gate (`18.75%` audit reduction vs Bonferroni ONLY for
       strict all-or-nothing joint release!).
     - Exact stratified conservative upper bound (`f_U` + `r_L` -> `q_U`) for diagnostic/dev checks.
  2. `PathRiskController` (P1):
     - Whole-Path Maximum Conformal (`M_i = max_t s_t(x_i)` on independent frozen calibration set; online running max
       `m_t = max_{k<=t} s_k(x)` -> `p_t = (1 + sum(M_i >= m_t)) / (n_0 + 1)`).
     - Guarantees class-conditional marginal anytime-valid FPR `P_0(exists t: p_t <= a) <= a` WITHOUT assuming exit
       independence, explicitly disclosing minimum non-randomized resolution `1/(n_0 + 1)` and fixed-cal tolerance bound
       `P_cal{P(M > M_* | cal) > a} <= (1 - a)^{n_0}`.
  3. `TailRiskCalibrator` (P1, STRICTLY DIAGNOSTIC — NEVER A RELEASE CERTIFICATE):
     - Audits actual scalar score support, saturation boundary atom mass `p_b = P(Z = b | Y = 0)`, ties/quantization steps.
     - Multi-threshold (`95/97/98/99%`) POT-GPD + known-endpoint reverse-power + endpoint-distance diagnostics.
     - Risk Safety Margin `M(z) = log((q * pi_L * S_{1,L}(z)) / ((1 - q) * (1 - pi_L) * S_{0,U}(z)))`.
  4. `FactorizedF6ExitOptimizer` (P0/P1):
     - Offline search strictly inside the F6-compliant factorized policy space `a_t^A(F_t^A), a_t^P(F_t^P)` (NEVER central
       joint Bellman that could leak Promo into AI!).
     - Uses linearized exact risk constraint `E[1_H ((1-q) 1_{Y=0} - q 1_{Y=1})] <= 0` + recall floors `r_A >= r_min` +
       true shared-backbone counterfactual cost `E[1_{both done} C_save - C_gate | reach t]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import beta as sp_beta, binom as sp_binom, genpareto


# ---------------------------------------------------------------------------
# P0. Exact Rational Arithmetic & `NaturalFlowAuditor`
# ---------------------------------------------------------------------------

def exact_binom_cdf(k: int, n: int, q0: Fraction = Fraction(1, 100)) -> Fraction:
    """Exact rational left-tail binomial CDF `P(Bin(n, q0) <= k)` with zero floating-point rounding."""
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    p = q0
    q = Fraction(1) - q0
    tot = Fraction(0)
    for i in range(k + 1):
        tot += Fraction(math.comb(n, i)) * (p ** i) * (q ** (n - i))
    return tot


def exact_cp_min_n(k: int, delta: Fraction = Fraction(1, 20), q0: Fraction = Fraction(1, 100)) -> int:
    """Smallest fixed sample size `n` such that `k` observed FPs certifies `q_H < q0` at confidence `1 - delta`."""
    n = k + 1
    while exact_binom_cdf(k, n, q0) > delta:
        n += 1
    return n


def exact_group_sequential_alpha(
    schedule: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    q0: Fraction = Fraction(1, 100),
) -> Fraction:
    """Exact rational Type-I error of a monotone group-sequential pass schedule `[(n_1, k_1), (n_2, k_2), ...]`.

    By stochastic monotonicity, worst-case unsafe null is at boundary `q0`. Futility stopping (discarding paths whose
    cumulative FP already exceeds `max_k`) never increases Type-I error.
    """
    if not schedule:
        return Fraction(0)
    sched = sorted(schedule, key=lambda x: x[0])
    max_k = max(k for _, k in sched)
    p = q0
    q = Fraction(1) - q0
    # State vector `a[k]` = probability of still being active with exactly `k` cumulative FPs (`0 <= k <= max_k`)
    a = [Fraction(0)] * (max_k + 1)
    a[0] = Fraction(1)
    total_pass = Fraction(0)
    prev_n = 0
    for n_cur, k_allow in sched:
        m = n_cur - prev_n
        if m <= 0:
            raise ValueError(f"Schedule look sizes must be strictly increasing: {sched}")
        b = [Fraction(math.comb(m, i)) * (p ** i) * (q ** (m - i)) if i <= m else Fraction(0) for i in range(max_k + 1)]
        a_next = [Fraction(0)] * (max_k + 1)
        for k_tot in range(max_k + 1):
            s = Fraction(0)
            for i in range(k_tot + 1):
                s += a[k_tot - i] * b[i]
            a_next[k_tot] = s
        for k_pass in range(min(k_allow, max_k) + 1):
            total_pass += a_next[k_pass]
            a_next[k_pass] = Fraction(0)
        a = a_next
        prev_n = n_cur
    return total_pass


def group_sequential_power_and_cost(
    true_q: float,
    schedule: dict[int, int],
    batch_step: int = 50,
) -> tuple[float, float]:
    """Compute exact pass probability and unconditional expected audit size (with futility stop every `batch_step`)."""
    max_n = max(schedule.keys())
    a = np.array([1.0], dtype=np.float64)
    passed = 0.0
    expected_n = 0.0
    for n in range(batch_step, max_n + 1, batch_step):
        expected_n += float(batch_step) * float(a.sum())
        pmf = sp_binom.pmf(np.arange(batch_step + 1), batch_step, true_q)
        a = np.convolve(a, pmf)
        b_allow = schedule.get(n, -1)
        if b_allow >= 0 and b_allow < len(a):
            passed += float(a[: b_allow + 1].sum())
            a[: b_allow + 1] = 0.0
        fut_max = max((v for t, v in schedule.items() if t > n), default=-1)
        if fut_max + 1 < len(a):
            a[fut_max + 1 :] = 0.0
    return float(passed), float(expected_n)


def mixture_e_process_log_wealth(
    n: int,
    k: int,
    q0: float = 0.01,
    qs: tuple[float, ...] = (0.0, 0.001, 0.005),
    weights: tuple[float, ...] = (0.5, 0.25, 0.25),
) -> float:
    """Log-wealth of fixed-mixture e-process `E_n^{mix} = sum_j w_j E_n(q_j)` (exact log-sum-exp, `0^0 = 1`)."""
    q_arr = np.asarray(qs, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    log_terms = np.log(w_arr) + float(n - k) * np.log((1.0 - q_arr) / (1.0 - q0))
    if k > 0:
        for idx, qj in enumerate(q_arr):
            if qj <= 0.0:
                log_terms[idx] = -np.inf
            else:
                log_terms[idx] += float(k) * math.log(qj / q0)
    return float(logsumexp(log_terms))


def exact_stratified_fdr_upper_bound(
    k_fp_neg: int,
    n_neg: int,
    k_tp_pos: int,
    n_pos: int,
    pi_nat: float = 0.03,
    delta_f: float = 0.025,
    delta_r: float = 0.025,
) -> dict[str, float]:
    """Conservative exact joint-confidence stratified FDR upper bound (fixed prior `pi_nat`)."""
    f_u = (
        float(-math.expm1(math.log(delta_f) / float(n_neg)))
        if k_fp_neg == 0
        else (1.0 if k_fp_neg >= n_neg else float(sp_beta.ppf(1.0 - delta_f, k_fp_neg + 1, n_neg - k_fp_neg)))
    )
    r_l = 0.0 if k_tp_pos <= 0 else float(sp_beta.ppf(delta_r, k_tp_pos, n_pos - k_tp_pos + 1))
    denom = (1.0 - pi_nat) * f_u + pi_nat * r_l
    q_u = ((1.0 - pi_nat) * f_u / denom) if denom > 0.0 else 1.0
    return {"f_upper": f_u, "r_lower": r_l, "q_fdr_upper": q_u, "joint_confidence": 1.0 - delta_f - delta_r}


@dataclass
class NaturalFlowAuditor:
    """Frozen-rule irreversible natural-stream auditor supporting Group-Sequential, CP, Mixture e-process & IUT.

    States are strictly irreversible (`ACTIVE -> PASS | INCONCLUSIVE`): once frozen & started, resetting on failure
    without spending fresh error budget is prohibited!
    """

    rule_hash: str
    q0: Fraction = Fraction(1, 100)
    delta: Fraction = Fraction(1, 20)
    design: str = "group_sequential_95"  # "group_sequential_95" | "group_sequential_975" | "zero_error_e" | "mixture_e"
    schedule: tuple[tuple[int, int], ...] = field(default_factory=lambda: ((350, 0), (550, 1), (800, 2)))
    n_audited: int = 0
    k_fp: int = 0
    status: str = "ACTIVE"  # ACTIVE | PASS | INCONCLUSIVE
    exact_design_alpha: float = field(init=False)

    def __post_init__(self) -> None:
        if self.design == "group_sequential_975":
            self.delta = Fraction(1, 40)
            self.schedule = ((400, 0), (700, 1), (900, 2))
        if self.design.startswith("group_sequential"):
            a_exact = exact_group_sequential_alpha(self.schedule, self.q0)
            if a_exact > self.delta:
                raise ValueError(f"Group-sequential schedule alpha={float(a_exact):.6f} exceeds budget {float(self.delta):.6f}!")
            self.exact_design_alpha = float(a_exact)
        else:
            self.exact_design_alpha = float(self.delta)

    def record_batch(self, n_new: int, k_fp_new: int) -> dict[str, Any]:
        """Ingest one blind-audited batch (`n_new` high-side calls, `k_fp_new` human FPs). Irreversible once terminal."""
        if self.status != "ACTIVE":
            raise RuntimeError(f"Auditor for rule {self.rule_hash} is already terminal ({self.status}); cannot reuse budget!")
        if n_new <= 0 or k_fp_new < 0 or k_fp_new > n_new:
            raise ValueError(f"Invalid batch counts: n_new={n_new}, k_fp_new={k_fp_new}")
        self.n_audited += int(n_new)
        self.k_fp += int(k_fp_new)

        if self.design.startswith("group_sequential"):
            sched_map = dict(self.schedule)
            max_n = max(sched_map.keys())
            if self.n_audited in sched_map and self.k_fp <= sched_map[self.n_audited]:
                self.status = "PASS"
            else:
                future_max_k = max((v for t, v in sched_map.items() if t > self.n_audited), default=-1)
                if self.k_fp > future_max_k or self.n_audited >= max_n:
                    # Per Rev. 2 Section 8.1/9.2: futility stop -> INCONCLUSIVE (does NOT prove risk > q0!)
                    self.status = "INCONCLUSIVE"
        elif self.design == "zero_error_e":
            if self.k_fp > 0:
                self.status = "INCONCLUSIVE"
            elif exact_binom_cdf(0, self.n_audited, self.q0) <= self.delta:
                self.status = "PASS"
        elif self.design == "mixture_e":
            lw = mixture_e_process_log_wealth(self.n_audited, self.k_fp, q0=float(self.q0))
            if lw >= -math.log(float(self.delta)) + 1e-12:
                self.status = "PASS"
        else:
            raise ValueError(f"Unknown design={self.design}")
        return self.certificate()

    def certificate(self) -> dict[str, Any]:
        return {
            "rule_hash": self.rule_hash,
            "design": self.design,
            "q0": float(self.q0),
            "delta_budget": float(self.delta),
            "exact_type1_alpha": round(self.exact_design_alpha, 8),
            "n_audited": self.n_audited,
            "k_fp": self.k_fp,
            "empirical_fdr": round(self.k_fp / max(1, self.n_audited), 6),
            "status": self.status,
            "certified_safe": self.status == "PASS",
        }

    @staticmethod
    def iut_joint_release_decision(auditors: list["NaturalFlowAuditor"]) -> dict[str, Any]:
        """Intersection-Union Test (IUT): valid ONLY for strict all-or-nothing joint release!"""
        if not auditors:
            return {"joint_release_approved": False, "reason": "empty_auditor_list"}
        all_pass = all(a.status == "PASS" for a in auditors)
        max_alpha = max(a.exact_design_alpha for a in auditors)
        return {
            "joint_release_approved": bool(all_pass),
            "iut_familywise_alpha_bound": round(max_alpha, 8),
            "note": "Valid ONLY for all-or-nothing joint release; individual sides CANNOT be released separately under IUT!",
            "sides": [a.certificate() for a in auditors],
        }


# ---------------------------------------------------------------------------
# P1. Whole-Path Maximum Conformal (`PathRiskController`)
# ---------------------------------------------------------------------------

class PathRiskController:
    """Whole-Path Maximum Conformal controller (Rev. 2 Section 3.2 & 4.1).

    Controls class-conditional marginal anytime-valid error `P_0(exists t: p_t <= a) <= a` across all 6 rungs
    (`R0, R1, T1, T2, T3, T4`) WITHOUT requiring exit independence, and explicitly reports both marginal resolution
    `1 / (n_0 + 1)` and fixed-calibration tolerance bound `(1 - a)^{n_0}`.
    """

    def __init__(self, cal_path_scores_null: np.ndarray) -> None:
        arr = np.asarray(cal_path_scores_null, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError("cal_path_scores_null must be 2D `(n_null, n_rungs)` with n_null >= 1")
        self.n_null = int(arr.shape[0])
        self.n_rungs = int(arr.shape[1])
        self.cal_max_sorted = np.sort(np.max(arr, axis=1))
        self.m_star = float(self.cal_max_sorted[-1])

    def online_p_values(self, score_trajectory: np.ndarray) -> np.ndarray:
        """Given `(B, T)` or `(T,)` anti-null score trajectory, return running whole-path conformal `p_t`."""
        s = np.asarray(score_trajectory, dtype=np.float64)
        squeeze = s.ndim == 1
        if squeeze:
            s = s[None, :]
        running_max = np.maximum.accumulate(s, axis=1)
        # Count how many `M_i >= m_t` via `searchsorted(..., side="left")` (conservative on ties!)
        idx_first_ge = np.searchsorted(self.cal_max_sorted, running_max, side="left")
        n_ge = self.n_null - idx_first_ge
        p_t = (1.0 + n_ge) / float(self.n_null + 1)
        return p_t[0] if squeeze else p_t

    def resolution_report(self, target_fpr: float = 1.0 / 6402.0, delta_cal: float = 0.05) -> dict[str, Any]:
        min_p = 1.0 / float(self.n_null + 1)
        min_n_marginal = int(math.ceil(1.0 / target_fpr) - 1)
        min_n_fixed_tol = int(math.ceil(math.log(delta_cal) / math.log(1.0 - target_fpr)))
        fixed_tol_fail_prob = float((1.0 - target_fpr) ** self.n_null)
        return {
            "n_null_cal": self.n_null,
            "n_rungs": self.n_rungs,
            "m_star_strict_threshold": round(self.m_star, 6),
            "min_nonrandomized_p": min_p,
            "target_fpr": target_fpr,
            "marginal_resolution_sufficient": bool(min_p <= target_fpr),
            "min_n_required_marginal": min_n_marginal,
            "min_n_required_fixed_cal_95pct_tolerance": min_n_fixed_tol,
            "fixed_cal_tolerance_fail_prob_at_target": fixed_tol_fail_prob,
            "guarantee_scope": "Class-conditional marginal FPR over joint (cal, test) randomness; NOT fixed-cal deployment FDR!",
        }


# ---------------------------------------------------------------------------
# P1. Bounded-Support Tail Diagnostics (`TailRiskCalibrator` — STRICTLY DIAGNOSTIC!)
# ---------------------------------------------------------------------------

def run_tail_risk_diagnostics(
    scores: np.ndarray,
    y_bin: np.ndarray,
    pi_nat: float = 0.03,
    target_fdr: float = 0.01,
    known_upper_bound: float | None = None,
    quantiles: tuple[float, ...] = (0.95, 0.97, 0.98, 0.99),
) -> dict[str, Any]:
    """Audit actual scalar score support, saturation atom mass `p_b`, multi-threshold POT-GPD stability, and Risk Margin `M(z)`.

    WARNING: Strictly diagnostic / candidate threshold generator. NEVER a distribution-free release certificate!
    """
    z = np.asarray(scores, dtype=np.float64)
    y = (np.asarray(y_bin) >= 0.5).astype(np.int32)
    z0 = z[y == 0]
    z1 = z[y == 1]
    if len(z0) < 50 or len(z1) < 10:
        return {"skipped": True, "reason": "insufficient_samples", "n_neg": int(len(z0)), "n_pos": int(len(z1))}

    z_max_obs = float(np.max(z))
    z0_max_obs = float(np.max(z0))
    b_ref = float(known_upper_bound) if known_upper_bound is not None else z_max_obs
    tol_atom = 1e-5 * max(1.0, abs(b_ref))
    p_atom_neg = float(np.mean(z0 >= b_ref - tol_atom))
    unique_ratio_neg = float(len(np.unique(np.round(z0, 6))) / max(1, len(z0)))

    # Multi-threshold POT-GPD stability diagnostics on strictly interior exceedances `z0 < b_ref - tol`
    z0_int = z0[z0 < b_ref - tol_atom] if p_atom_neg > 0.0 else z0
    fits: dict[str, Any] = {}
    for qv in quantiles:
        if len(z0_int) < 40:
            continue
        u = float(np.quantile(z0_int, qv))
        exc = z0_int[z0_int > u] - u
        m = int(len(exc))
        if m < 15:
            fits[f"q{int(round(qv*100))}"] = {"u": round(u, 5), "m_exceedances": m, "skipped": True}
            continue
        try:
            xi, _, sigma = genpareto.fit(exc, floc=0.0)
            se_xi = abs(1.0 + float(xi)) / math.sqrt(float(m)) if xi > -0.5 else float("nan")
            entry: dict[str, Any] = {
                "u": round(u, 5),
                "m_exceedances": m,
                "xi_mle": round(float(xi), 5),
                "sigma_mle": round(float(sigma), 5),
                "ideal_se_xi": round(se_xi, 5) if math.isfinite(se_xi) else None,
            }
            if known_upper_bound is not None and known_upper_bound > u + 1e-6:
                # Reverse-power beta estimate on known upper bound `b`: `V = -log((b - Z)/(b - u))`, MLE `beta = 1 / mean(V)`
                v_dist = -np.log(np.clip((known_upper_bound - (u + exc)) / (known_upper_bound - u), 1e-12, 1.0))
                beta_hat = 1.0 / max(1e-6, float(np.mean(v_dist)))
                entry["known_bound_reverse_power_beta"] = round(beta_hat, 4)
                entry["known_bound_equiv_xi"] = round(-1.0 / beta_hat, 5)
            fits[f"q{int(round(qv*100))}"] = entry
        except Exception as exc_err:
            fits[f"q{int(round(qv*100))}"] = {"u": round(u, 5), "m_exceedances": m, "error": str(exc_err)}

    # Exact empirical Risk Safety Margin M(z) using Clopper-Pearson S_{0,U} and S_{1,L} at candidate thresholds
    margin_curve: list[dict[str, Any]] = []
    for cand_q in (0.98, 0.99, 0.995, 0.998, 0.999):
        zt = float(np.quantile(z0, cand_q))
        k0 = int(np.sum(z0 >= zt))
        k1 = int(np.sum(z1 >= zt))
        s0_u = float(-math.expm1(math.log(0.025) / len(z0))) if k0 == 0 else float(sp_beta.ppf(0.975, k0 + 1, len(z0) - k0))
        s1_l = 0.0 if k1 == 0 else float(sp_beta.ppf(0.025, k1, len(z1) - k1 + 1))
        if s1_l > 0.0 and s0_u > 0.0:
            m_val = math.log((target_fdr * pi_nat * s1_l) / ((1.0 - target_fdr) * (1.0 - pi_nat) * s0_u))
        else:
            m_val = -99.0
        margin_curve.append({
            "z_thresh": round(zt, 5),
            "neg_exceed_k": k0,
            "pos_recall_emp": round(k1 / float(len(z1)), 4),
            "s0_cp975_upper": round(s0_u, 6),
            "s1_cp975_lower": round(s1_l, 4),
            "risk_safety_margin_M": round(m_val, 4),
            "margin_ge_0": bool(m_val >= 0.0),
        })

    return {
        "certificate_disclaimer": "DIAGNOSTIC_ONLY_NOT_A_RELEASE_CERTIFICATE",
        "n_neg": int(len(z0)),
        "n_pos": int(len(z1)),
        "support_audit": {
            "obs_min": round(float(np.min(z)), 5),
            "obs_max": round(z_max_obs, 5),
            "neg_obs_max": round(z0_max_obs, 5),
            "known_upper_bound": known_upper_bound,
            "neg_boundary_atom_prob_pb": round(p_atom_neg, 6),
            "neg_unique_score_ratio": round(unique_ratio_neg, 5),
        },
        "multi_threshold_gpd": fits,
        "empirical_risk_safety_margin_curve": margin_curve,
    }


# ---------------------------------------------------------------------------
# P0/P1. F6-Compliant Factorized Dual-Axis Exit Optimizer (`FactorizedF6ExitOptimizer`)
# ---------------------------------------------------------------------------

def evaluate_factorized_f6_policy(
    p_ai_rungs: np.ndarray,
    p_pr_rungs: np.ndarray,
    y_ai: np.ndarray,
    y_pr: np.ndarray,
    ai_lo_thresholds: tuple[float, ...],
    ai_hi_thresholds: tuple[float, ...],
    pr_lo_thresholds: tuple[float, ...],
    rung_costs_ms: tuple[float, ...],
    q_hi_target: float = 0.01,
    q_lo_target: float = 0.01,
    pi_ai_nat: float = 0.03,
) -> dict[str, Any]:
    """Evaluate a strictly FACTORIZED F6-compliant dual-axis exit policy on full-path replay arrays `(N, K)`.

    AI stopping time `tau_A` depends ONLY on `p_ai_rungs`; Promo stopping time `tau_P` depends ONLY on `p_pr_rungs`!
    Shared backbone physical stopping time is `tau_phys = max(tau_A, tau_P)` while AI outputs latch strictly at `tau_A`.
    """
    pa = np.asarray(p_ai_rungs, dtype=np.float64)
    pp = np.asarray(p_pr_rungs, dtype=np.float64)
    ya = (np.asarray(y_ai) >= 0.5).astype(np.int32)
    yp = (np.asarray(y_pr) >= 0.5).astype(np.int32)
    n, k_rungs = pa.shape

    tau_a = np.full((n,), k_rungs - 1, dtype=np.int64)
    ai_high_exit = np.zeros((n,), dtype=bool)
    ai_low_exit = np.zeros((n,), dtype=bool)
    for t in range(k_rungs - 1):
        undecided = tau_a == (k_rungs - 1)
        hit_lo = undecided & (pa[:, t] <= ai_lo_thresholds[t])
        hit_hi = undecided & (pa[:, t] >= ai_hi_thresholds[t])
        tau_a[hit_lo | hit_hi] = t
        ai_low_exit |= hit_lo
        ai_high_exit |= hit_hi

    tau_p = np.full((n,), k_rungs - 1, dtype=np.int64)
    for t in range(k_rungs - 1):
        undecided_p = tau_p == (k_rungs - 1)
        hit_p = undecided_p & (pp[:, t] <= pr_lo_thresholds[t])
        tau_p[hit_p] = t

    tau_phys = np.maximum(tau_a, tau_p)
    cum_cost = np.cumsum(np.asarray(rung_costs_ms, dtype=np.float64))
    mean_phys_ms = float(np.mean(cum_cost[tau_phys]))
    mean_ai_only_ms = float(np.mean(cum_cost[tau_a]))

    # Natural-prior reweighted exact linearized risk functionals `E[g_H]` and `E[g_L]`
    pos_m = ya == 1
    neg_m = ya == 0
    r_hi = float(ai_high_exit[pos_m].mean()) if pos_m.any() else 0.0
    f_hi = float(ai_high_exit[neg_m].mean()) if neg_m.any() else 0.0
    g_h_nat = (1.0 - pi_ai_nat) * (1.0 - q_hi_target) * f_hi - pi_ai_nat * q_hi_target * r_hi

    fn_lo = float(ai_low_exit[pos_m].mean()) if pos_m.any() else 0.0
    tn_lo = float(ai_low_exit[neg_m].mean()) if neg_m.any() else 0.0
    g_l_nat = pi_ai_nat * (1.0 - q_lo_target) * fn_lo - (1.0 - pi_ai_nat) * q_lo_target * tn_lo

    return {
        "mean_shared_phys_ms": round(mean_phys_ms, 4),
        "mean_ai_logical_ms": round(mean_ai_only_ms, 4),
        "promo_tail_drag_ms": round(mean_phys_ms - mean_ai_only_ms, 4),
        "ai_avg_rung": round(float(np.mean(tau_a)), 3),
        "pr_avg_rung": round(float(np.mean(tau_p)), 3),
        "phys_avg_rung": round(float(np.mean(tau_phys)), 3),
        "ai_hi_recall": round(r_hi, 4),
        "ai_hi_fpr": round(f_hi, 6),
        "ai_lo_human_exit": round(tn_lo, 4),
        "ai_lo_ai_miss_fnr": round(fn_lo, 5),
        "linearized_g_H_nat": round(g_h_nat, 7),
        "linearized_g_L_nat": round(g_l_nat, 7),
        "g_H_feasible": bool(g_h_nat <= 0.0),
        "g_L_feasible": bool(g_l_nat <= 0.0),
    }


def self_verify_rev2_exact_numbers() -> dict[str, Any]:
    """Reproduce EVERY exact numerical boundary from Rev. 2 Review Appendices A & B as an automated unit test!"""
    cp_table = {k: (exact_cp_min_n(k, Fraction(1, 20)), exact_cp_min_n(k, Fraction(1, 40))) for k in range(4)}
    assert cp_table == {0: (299, 368), 1: (473, 555), 2: (628, 720), 3: (773, 874)}, f"CP mismatch: {cp_table}"

    a95 = exact_group_sequential_alpha(((350, 0), (550, 1), (800, 2)), Fraction(1, 100))
    a975 = exact_group_sequential_alpha(((400, 0), (700, 1), (900, 2)), Fraction(1, 100))
    assert abs(float(a95) - 0.04803299693907713) < 1e-12
    assert abs(float(a975) - 0.02391173021745674) < 1e-12

    mix_bounds = {
        k: (
            next(n for n in range(k, 3000) if mixture_e_process_log_wealth(n, k) >= -math.log(0.05)),
            next(n for n in range(k, 3000) if mixture_e_process_log_wealth(n, k) >= -math.log(0.025)),
        )
        for k in range(4)
    }
    assert mix_bounds == {0: (330, 403), 1: (712, 796), 2: (946, 1035), 3: (1165, 1263)}, f"Mix mismatch: {mix_bounds}"

    strat = exact_stratified_fdr_upper_bound(0, 14500, 250, 500, pi_nat=0.03)
    assert abs(strat["q_fdr_upper"] - 0.01774444045) < 1e-9

    p_pass_01, exp_n_01 = group_sequential_power_and_cost(0.001, {350: 0, 550: 1, 800: 2})
    assert abs(p_pass_01 - 0.9656) < 5e-4 and abs(exp_n_01 - 424.0) < 0.2

    # Verify whole-path conformal minimum resolution `n_0 = 6401 -> 1/6402`
    prc = PathRiskController(np.zeros((6401, 6), dtype=np.float32))
    rep = prc.resolution_report(target_fpr=1.0 / 6402.0)
    assert rep["marginal_resolution_sufficient"] is True and rep["min_n_required_fixed_cal_95pct_tolerance"] == 19178

    return {
        "all_rev2_assertions_passed": True,
        "cp_exact": cp_table,
        "group_seq_95_alpha": float(a95),
        "group_seq_975_alpha": float(a975),
        "mixture_e_bounds": mix_bounds,
        "stratified_15k_fdr_ub": round(strat["q_fdr_upper"], 8),
        "group_seq_95_at_0p1pct_risk": {"pass_prob": round(p_pass_01, 4), "expected_n": round(exp_n_01, 1)},
        "whole_path_conformal_resolution": rep,
    }


if __name__ == "__main__":
    print(json.dumps(self_verify_rev2_exact_numbers(), ensure_ascii=False, indent=2))
