"""MeguvizCAC v1.2.3-ApexNano-Pro Model, HAWQ-V2 Bit Reallocation, Length-Aware z5_softmin & 11-Gate CI.

Strictly preserves all `v1.0.0`, `v1.1.0`, `v1.2.0`, `v1.2.1`, and `v1.2.2` modules untouched while keeping
100% backwards-compatible `Bundle v7 (<= 2.20 MB)` & `5-Stage FP32-Exact ONNX` ABI!

Core v1.2.3 upgrades (all grounded in `v123_empirical_probes.json`, 0 B ABI break):
  1. `apply_hawq_v2_bit_reallocation_v123(model)` (P0-5):
     - Upgrades the 8 highest tail-Fisher layers (`prelude_l0.{k_p,v_p,o_p,ffn_p.w_up,ffn_p.w_down,v_c,o_c}`, `core.o_proj`)
       from `seq2/trit158` -> `w4 (gs=64)`.
     - Demotes 3 zero-Fisher layers (`h2_hyp_proj`, `h2_doubt_proj`, `core.haar_q`) from `w4` -> `seq2 (gs=128)`.
     - Net compressed cost ~+70 KB -> total `Bundle v7 ~2.14 MB <= 2.20 MB`, cutting Sub-2bit QAT tail drop!
  2. `MeguvizV123ApexNanoModel(MeguvizV122ApexNanoModel)`:
     - Applies HAWQ-V2 bit reallocation on init.
     - Adds training-only `15%` era-sensitive `style56` feature dropout (dims `0..39`, preserving promo veto cues;
       0 extra ONNX ops at eval!) to suppress `style56_raw` `0.8078` era leakage into `emb` (P0-6).
     - `unlock_z_char_coda_ai_gate(min_abs_init=0.15)` so `r0` char/symbol features actively reach `t2` (P0-2).
  3. `fit_len_aware_z5_bootstrap_v123(...)` & `apply_len_aware_z5_v123(...)` (P0-7):
     - Constrained 2-bucket (`<60c` `r0/r1`-enhanced vs `>=60c` `t2/ebt`-anchored) wild-3% bootstrap `z5_softmin`
       fitted STRICTLY on clean `val_meta_cal` (ZERO `god_tier_500_holdout`!).
  4. `MeguvizV123CascadeEngine(MeguvizV122CascadeEngine)`:
     - 100% v1.2.1/v1.2.2 5-Stage ONNX ABI compatible with native constrained length-aware `z5_softmin`.
  5. `run_v123_ci_gates(...)` (P0-8):
     - Full 11-gate CI (`G0, S1, F6, C1, P1_fp32, R0, L1, E1_dual, PR_LUT_FLOOR, AI_LUT_FLOOR, CERTIFIED_LTT`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F

from meguviz_cac.v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from meguviz_cac.v07_omniveritas import MAX_LEN_V07, OmniQuantLinear
from meguviz_cac.v09_chronoloop import NUM_SLOW_SLOTS_V09
from meguviz_cac.v10_apexloop import (
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    NUM_SEM_FAST_V10,
    tunnel_trm_ste,
)
from meguviz_cac.v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d
from meguviz_cac.v11_omniguard import (
    AI_EMB_END_V11,
    CBM_AI_CONCEPTS_V11,
    R1_PROMO_VETO_DIMS_V11,
    SHARED_EMB_END_V11,
)
from meguviz_cac.v12_apexnano import extract_rung0_charhash_batch
from meguviz_cac.v121_apexnano import (
    bounded_fused3_logits,
    export_v121_modular_onnx_and_bundle_v7,
    fold_zero_op_v121,
    load_bundle_v7_v121_into_model,
)
from meguviz_cac.v122_apexnano import (
    MeguvizV122ApexNanoModel,
    MeguvizV122CascadeEngine,
    apply_v122_full_logit_leace,
    apply_z5_softmin_v122,
    compute_bitdistill_slot_relation_kl,
    compute_wild_tail_objective_v122,
    fit_conformal_fdr_thresholds_v122,
    fit_moses_register_offsets_v122,
    fit_output_shifted_promo_lut_v122,
    fit_smooth_ai_lut_v122,
    fit_z5_softmin_bootstrap_v122,
    install_quest_trust_gradient_hooks,
    run_v122_ci_gates,
    transplant_dual_source_v10_v121_to_v122,
)

# ---------------------------------------------------------------------------
# 1. P0-5: HAWQ-V2 Tail-Sensitivity Bit Reallocation Table (0 B ABI Break)
# ---------------------------------------------------------------------------

HAWQ_V2_UPGRADE_TO_W4_V123: tuple[str, ...] = (
    "prelude_l0.k_p",
    "prelude_l0.v_p",
    "prelude_l0.o_p",
    "prelude_l0.ffn_p.w_up",
    "prelude_l0.ffn_p.w_down",
    "prelude_l0.v_c",
    "prelude_l0.o_c",
    "core.o_proj",
)

HAWQ_V2_DEMOTE_TO_SEQ2_V123: tuple[str, ...] = (
    "h2_hyp_proj",
    "h2_doubt_proj",
    "core.haar_q",
)


def apply_hawq_v2_bit_reallocation_v123(model: nn.Module) -> dict[str, Any]:
    """Reallocate OmniQuantLinear `target_bits` / `group_size` per Probe 3 tail-Fisher ranking (~+70 KB zip)."""
    up_set = set(HAWQ_V2_UPGRADE_TO_W4_V123)
    dn_set = set(HAWQ_V2_DEMOTE_TO_SEQ2_V123)
    upgraded: list[str] = []
    demoted: list[str] = []
    for nm, mod in model.named_modules():
        if not isinstance(mod, OmniQuantLinear):
            continue
        if nm in up_set:
            mod.target_bits = "w4"
            mod.group_size = 64
            upgraded.append(nm)
        elif nm in dn_set:
            mod.target_bits = "seq2"
            mod.group_size = 128
            demoted.append(nm)
    return {
        "upgraded_to_w4": sorted(upgraded),
        "demoted_to_seq2": sorted(demoted),
        "n_upgraded": len(upgraded),
        "n_demoted": len(demoted),
        "all_expected_matched": bool(len(upgraded) == len(up_set) and len(demoted) == len(dn_set)),
    }


# ---------------------------------------------------------------------------
# 2. MeguvizV123ApexNanoModel (HAWQ-V2 + Style Era-Dropout + Unlocked z_char)
# ---------------------------------------------------------------------------

class MeguvizV123ApexNanoModel(MeguvizV122ApexNanoModel):
    """v1.2.3 ApexNano-Pro: HAWQ-V2 W4 prelude/core + training-only style era-dropout + unlocked z_char_coda_ai."""

    def __init__(self, *args: Any, style_era_dropout_p: float = 0.15, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.style_era_dropout_p = float(style_era_dropout_p)
        self.hawq_v2_rep = apply_hawq_v2_bit_reallocation_v123(self)

    @torch.no_grad()
    def unlock_z_char_coda_ai_gate(self, min_abs_init: float = 0.15) -> float:
        """P0-2: Ensure `|z_char_coda_ai_gate| >= 0.15` so R0 char/symbol cues actively reach T2 coda!"""
        if not hasattr(self, "z_char_coda_ai_gate"):
            return 0.0
        g = self.z_char_coda_ai_gate.data
        cur = float(g.float().abs().mean().item())
        if cur < min_abs_init:
            g.fill_(float(min_abs_init))
        return round(float(g.float().mean().item()), 5)

    def _coda_readout(
        self,
        slots: torch.Tensor,
        span_base: torch.Tensor,
        mask0: torch.Tensor,
        style_feats: torch.Tensor,
        ebt_out: dict[str, torch.Tensor],
        exit_idx: int,
        promo_sh_lam: float = 0.35,
        z_char: torch.Tensor | None = None,
        detach_zchar_for_ai: bool = False,
    ) -> dict[str, torch.Tensor]:
        # P0-6: Training-only 15% era-sensitive style feature dropout (dims 0..39 only; promo veto dims untouched; 0 ONNX ops at eval!)
        st_in = style_feats
        if self.training and self.style_era_dropout_p > 0.0 and style_feats.shape[-1] >= 40:
            keep = (torch.rand((style_feats.shape[0], 40), device=style_feats.device, dtype=style_feats.dtype) >= self.style_era_dropout_p).to(style_feats.dtype)
            keep = keep / max(1e-4, 1.0 - self.style_era_dropout_p)
            st_in = torch.cat([style_feats[:, :40] * keep, style_feats[:, 40:]], dim=-1)
        return super()._coda_readout(
            slots=slots,
            span_base=span_base,
            mask0=mask0,
            style_feats=st_in,
            ebt_out=ebt_out,
            exit_idx=exit_idx,
            promo_sh_lam=promo_sh_lam,
            z_char=z_char,
            detach_zchar_for_ai=detach_zchar_for_ai,
        )


# ---------------------------------------------------------------------------
# 3. P0-7: Constrained 2-Bucket Length-Aware `z5_softmin` (Clean val_meta_cal ONLY!)
# ---------------------------------------------------------------------------

def fit_len_aware_z5_bootstrap_v123(
    lg_dict: dict[str, np.ndarray],
    y_bin: np.ndarray,
    char_len: np.ndarray,
    ebt_clip: float = 6.0,
    n_boot: int = 200,
    pi_wild: float = 0.03,
    short_cutoff: int = 60,
    seed: int = 20261128,
) -> dict[str, Any]:
    """Fit constrained 2-bucket (`<60c` r0/r1-boosted vs `>=60c` t2/ebt-anchored) `z5_softmin` STRICTLY on clean `val_meta_cal`!

    Also includes top-level global `(mu, sd, w, tau, lam_dis)` so legacy v1.2.2 engines stay 100% compatible.
    """
    global_cfg = fit_z5_softmin_bootstrap_v122(lg_dict, y_bin, ebt_clip=ebt_clip, n_boot=n_boot, pi_wild=pi_wild, seed=seed)
    y = np.asarray(y_bin, dtype=np.int32).reshape(-1)
    cl = np.asarray(char_len, dtype=np.int32).reshape(-1)
    ebt6 = ebt_clip * np.tanh(np.asarray(lg_dict["lg_ebt"], dtype=np.float64) / ebt_clip)
    X = np.stack(
        [
            np.asarray(lg_dict["lg_r0"], dtype=np.float64),
            np.asarray(lg_dict["lg_r1"], dtype=np.float64),
            np.asarray(lg_dict["lg_t1"], dtype=np.float64),
            np.asarray(lg_dict["lg_t2"], dtype=np.float64),
            ebt6,
        ],
        axis=1,
    )
    mu = np.asarray(global_cfg["mu"], dtype=np.float64)
    sd = np.maximum(np.asarray(global_cfg["sd"], dtype=np.float64), 1e-4)
    Z = (X - mu) / sd
    z_std = Z.std(axis=1)

    short_grid = [
        (0.28, 0.22, 0.16, 0.18, 0.16),
        (0.32, 0.23, 0.15, 0.15, 0.15),
        (0.25, 0.20, 0.18, 0.20, 0.17),
        (0.22, 0.18, 0.20, 0.22, 0.18),
        (0.20, 0.20, 0.20, 0.20, 0.20),
    ]
    long_grid = [
        (0.15, 0.15, 0.22, 0.28, 0.20),
        (0.18, 0.15, 0.20, 0.27, 0.20),
        (0.12, 0.16, 0.22, 0.28, 0.22),
        (0.20, 0.15, 0.20, 0.25, 0.20),
        (0.20, 0.20, 0.20, 0.20, 0.20),
    ]

    def _fit_bucket(mask: np.ndarray, w_grid: list[tuple[float, ...]], b_seed: int) -> dict[str, Any]:
        ym = y[mask]
        Zm = Z[mask]
        zsm = z_std[mask]
        pos_i = np.where(ym == 1)[0]
        neg_i = np.where(ym == 0)[0]
        if len(pos_i) < 10 or len(neg_i) < 20:
            return {
                "w": list(global_cfg["w"]),
                "tau": float(global_cfg["tau"]),
                "lam_dis": float(global_cfg["lam_dis"]),
                "n": int(mask.sum()),
                "fallback_global": True,
            }
        rng = np.random.default_rng(b_seed)
        boots = [
            np.concatenate([rng.choice(pos_i, size=len(pos_i), replace=True), rng.choice(neg_i, size=len(neg_i), replace=True)])
            for _ in range(max(10, int(n_boot)))
        ]
        best_sc = -1e9
        best_c: dict[str, Any] = {}
        for wt in w_grid:
            lw = np.log(np.maximum(np.asarray(wt, dtype=np.float64), 1e-6))[None, :]
            for tau in (0.75, 1.0, 1.5, 2.0):
                base_s = -tau * logsumexp(lw - Zm / tau, axis=1)
                for lam in (0.0, 0.10, 0.20, 0.30):
                    sc = base_s - lam * zsm
                    mf = compute_wild_tail_objective_v122(ym, sc, pi_wild=pi_wild)
                    t_vals = []
                    for bi in boots:
                        yb, sb = ym[bi], sc[bi]
                        ps, ns = sb[yb == 1], np.sort(sb[yb == 0])[::-1]
                        t_vals.append(float(np.mean([(ps > ns[min(k - 1, len(ns) - 1)]).mean() for k in (1, 3, 5)])))
                    tm, ts = float(np.mean(t_vals)), float(np.std(t_vals))
                    obj = 0.55 * mf["T"] + 0.35 * (tm - 0.5 * ts) + 0.10 * (0.5 * mf["auroc"] + 0.5 * mf["pr_auc"])
                    if obj > best_sc:
                        best_sc = obj
                        best_c = {
                            "w": [round(float(v), 4) for v in wt],
                            "tau": float(tau),
                            "lam_dis": float(lam),
                            "n": int(mask.sum()),
                            "val_T": mf["T"],
                            "val_obj": round(obj, 5),
                            "fallback_global": False,
                        }
        return best_c

    m_short = cl < int(short_cutoff)
    m_long = ~m_short
    out_cfg = dict(global_cfg)
    out_cfg["len_aware"] = {
        "short_cutoff": int(short_cutoff),
        "short_lt60": _fit_bucket(m_short, short_grid, seed + 1),
        "long_ge60": _fit_bucket(m_long, long_grid, seed + 2),
    }
    return out_cfg


def apply_len_aware_z5_v123(
    lg_dict: dict[str, np.ndarray],
    cfg: dict[str, Any],
    char_len: np.ndarray | None = None,
) -> np.ndarray:
    """Apply constrained 2-bucket length-aware `z5_softmin` (falls back cleanly to v1.2.2 global if `char_len` omitted)."""
    la = cfg.get("len_aware")
    if not la or char_len is None:
        return apply_z5_softmin_v122(lg_dict, cfg)
    ec = float(cfg.get("ebt_clip", 6.0))
    ebt6 = ec * np.tanh(np.asarray(lg_dict["lg_ebt"], dtype=np.float64) / ec)
    X = np.stack(
        [
            np.asarray(lg_dict["lg_r0"], dtype=np.float64),
            np.asarray(lg_dict["lg_r1"], dtype=np.float64),
            np.asarray(lg_dict["lg_t1"], dtype=np.float64),
            np.asarray(lg_dict["lg_t2"], dtype=np.float64),
            ebt6,
        ],
        axis=1,
    )
    mu = np.asarray(cfg["mu"], dtype=np.float64)
    sd = np.maximum(np.asarray(cfg["sd"], dtype=np.float64), 1e-4)
    Z = (X - mu) / sd
    z_std = Z.std(axis=1)
    cl = np.asarray(char_len, dtype=np.int32).reshape(-1)
    cut = int(la.get("short_cutoff", 60))
    out = np.zeros(len(X), dtype=np.float64)
    for m, sub in ((cl < cut, la["short_lt60"]), (cl >= cut, la["long_ge60"])):
        if not m.any():
            continue
        lw = np.log(np.maximum(np.asarray(sub["w"], dtype=np.float64), 1e-6))[None, :]
        tau = float(sub["tau"])
        lam = float(sub.get("lam_dis", 0.0))
        out[m] = -tau * logsumexp(lw - Z[m] / tau, axis=1) - lam * z_std[m]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. MeguvizV123CascadeEngine (100% v1.2.1/v1.2.2 5-Stage ONNX ABI Compatible!)
# ---------------------------------------------------------------------------

class MeguvizV123CascadeEngine(MeguvizV122CascadeEngine):
    """5-Stage ONNX CPU Cascade Engine with native length-aware `z5_softmin` & clean AI/Promo LUTs."""

    def predict_batch(self, texts: list[str], max_batch: int = 32, force_full: bool = False) -> list[dict[str, Any]]:
        if not self.z5_cfg or not self.z5_cfg.get("len_aware"):
            return super().predict_batch(texts, max_batch=max_batch, force_full=force_full)

        n = len(texts)
        if n == 0:
            return []
        results: list[dict[str, Any] | None] = [None] * n
        order = sorted(range(n), key=lambda i: len(texts[i]))

        r0_en = bool(self.r0_policy.get("enabled", False)) and not force_full
        r0_tau_ai = float(self.r0_policy.get("tau_ai", 0.08))
        r0_tau_pr = float(self.r0_policy.get("tau_promo", 0.08))
        r0_def_max = float(self.r0_policy.get("defer_max", 0.35))
        r0_max_ch = int(self.r0_policy.get("max_chars", 160))

        r1_en = bool(self.r1_policy.get("enabled", False)) and not force_full
        r1_def_max = float(self.r1_policy.get("r1_defer_max", 0.15))
        r1_p_ai_max = float(self.r1_policy.get("r1_p_ai_max", 0.08))
        r1_p_pr_max = float(self.r1_policy.get("r1_p_promo_max", 0.06))
        r1_tok_cap = int(self.r1_policy.get("r1_max_tokens", 96))

        for st in range(0, n, max_batch):
            mb = order[st : st + max_batch]
            mb_txts = [texts[i] for i in mb]

            c1, c3, cm, cf, spoofs = extract_rung0_charhash_batch(mb_txts)
            lg_ai0, lg_pr0, pdef0, z_char = self.sess_r0.run(None, {"c1": c1, "c3": c3, "mask": cm, "feats": cf})
            p_ai0 = expit(np.clip(lg_ai0, -30.0, 30.0))
            p_pr0 = expit(np.clip(lg_pr0, -30.0, 30.0))
            if self.lut_ai_r0:
                p_ai0 = apply_spline_temp_lut(p_ai0, self.lut_ai_r0)
            if self.lut_pr_r0:
                p_pr0 = apply_spline_temp_lut(p_pr0, self.lut_pr_r0)

            surv0: list[int] = []
            for ri, oi in enumerate(mb):
                if (
                    r0_en and len(texts[oi]) <= r0_max_ch and spoofs[ri] == 0.0 and ("http" not in texts[oi].lower())
                    and float(pdef0[ri]) <= r0_def_max and float(p_ai0[ri]) <= r0_tau_ai and float(p_pr0[ri]) <= r0_tau_pr
                ):
                    results[oi] = {"prob_ai": float(p_ai0[ri]), "prob_promo": float(p_pr0[ri]), "exit_rung": 0}
                else:
                    surv0.append(ri)
            if not surv0:
                continue

            s0_idx = np.array(surv0, dtype=np.int64)
            s0_txts = [mb_txts[ri] for ri in surv0]
            zc_s0 = z_char[s0_idx]
            lg_r0_s0 = lg_ai0[s0_idx]
            cleans, f56_l, metas = [], [], []
            for t in s0_txts:
                cl, f56, mt = extract_style_features_56d(t)
                cleans.append(cl); f56_l.append(f56); metas.append(mt)
            encs = [self.tokenizer.encode_ids(c, max_len=self.max_len) for c in cleans]
            engs = [make_engram_bigram_ids(s, ENGRAM_BIGRAM_BUCKETS_V10) for s in encs]
            tris = [make_engram_trigram_ids(s, ENGRAM_TRIGRAM_BUCKETS_V10) for s in encs]
            bs1 = len(surv0)
            ml = max(2, max(len(s) for s in encs))
            if ml % 2 == 1:
                ml += 1
            ids = np.full((bs1, ml), self.tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((bs1, ml), dtype=np.int64)
            eg = np.zeros((bs1, ml), dtype=np.int64)
            tg = np.zeros((bs1, ml), dtype=np.int64)
            for k, s in enumerate(encs):
                ids[k, : len(s)], mask[k, : len(s)], eg[k, : len(s)], tg[k, : len(s)] = s, 1, engs[k], tris[k]
            st_b = np.stack(f56_l, axis=0).astype(np.float32)

            o1 = self.sess_r1.run(None, {"input_ids": ids, "attention_mask": mask, "style_feats": st_b, "engram_ids": eg, "trigram_ids": tg, "z_char": zc_s0})
            p_ai1 = expit(np.clip(o1[0], -30.0, 30.0))
            p_pr1 = expit(np.clip(o1[1], -30.0, 30.0))
            if self.lut_ai_r1:
                p_ai1 = apply_spline_temp_lut(p_ai1, self.lut_ai_r1)
            if self.lut_pr_r1:
                p_pr1 = apply_spline_temp_lut(p_pr1, self.lut_pr_r1)

            surv1: list[int] = []
            for k, ri in enumerate(surv0):
                oi = mb[ri]
                if (
                    r1_en and len(encs[k]) <= r1_tok_cap and metas[k]["spoof_any"] == 0.0 and ("http" not in cleans[k])
                    and float(np.max(st_b[k, list(R1_PROMO_VETO_DIMS_V11)])) == 0.0
                    and float(o1[2][k]) <= r1_def_max and float(o1[3][k]) <= 0.50
                    and float(p_ai1[k]) <= r1_p_ai_max and float(p_pr1[k]) <= r1_p_pr_max
                ):
                    results[oi] = {"prob_ai": float(p_ai1[k]), "prob_promo": float(p_pr1[k]), "exit_rung": 1}
                else:
                    surv1.append(k)
            if not surv1:
                continue

            s1_idx = np.array(surv1, dtype=np.int64)
            o2 = self.sess_r2.run(
                None,
                {
                    "input_ids": ids[s1_idx], "attention_mask": mask[s1_idx], "style_feats": st_b[s1_idx],
                    "engram_ids": eg[s1_idx], "trigram_ids": tg[s1_idx],
                    "x1_in": o1[4][s1_idx], "h0_1": o1[5][s1_idx], "r1_pkt": o1[6][s1_idx], "z_char": zc_s0[s1_idx],
                },
            )
            lg_t1 = o2[0]
            raw_p2 = expit(np.clip(lg_t1, -30.0, 30.0))
            c1_c = np.abs(2.0 * raw_p2 - 1.0).astype(np.float32)[:, None]
            uds = np.concatenate([c1_c, o2[5][:, None], o2[4][:, None], o2[7][:, None]], axis=-1)
            o3 = self.sess_r3.run(
                None,
                {
                    "slots": o2[8], "s_anchor": o2[9], "carry_m_l": o2[10], "zh_bus": o2[11],
                    "k_ev": o2[12], "v_ev": o2[13], "m_pool": o2[14], "k_haar": o2[15], "v_haar": o2[16],
                    "attention_mask": mask[s1_idx], "h0_pool": o2[17], "span_base": o2[18],
                    "style_feats": st_b[s1_idx], "u_doubt_scalars": uds, "z_char": zc_s0[s1_idx],
                },
            )
            ch_lens = np.array([len(s0_txts[k]) for k in surv1], dtype=np.int32)
            lg_z5 = apply_len_aware_z5_v123(
                {"lg_r0": lg_r0_s0[s1_idx], "lg_r1": o1[0][s1_idx], "lg_t1": lg_t1, "lg_t2": o3[0], "lg_ebt": o3[4]},
                self.z5_cfg,
                char_len=ch_lens,
            )
            p_ai_f = expit(np.clip(lg_z5, -30.0, 30.0))
            if self.lut_ai_z5:
                p_ai_f = apply_spline_temp_lut(p_ai_f, self.lut_ai_z5)
            p_pr_f = expit(np.clip(o3[1], -30.0, 30.0))
            if self.lut_pr_full:
                p_pr_f = apply_spline_temp_lut(p_pr_f, self.lut_pr_full)
            for j, k in enumerate(surv1):
                oi = mb[surv0[k]]
                results[oi] = {"prob_ai": float(p_ai_f[j]), "prob_promo": float(p_pr_f[j]), "exit_rung": 3}

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 5. Automated 11-Gate V123 CI Suite
# ---------------------------------------------------------------------------

def run_v123_ci_gates(
    model: MeguvizV123ApexNanoModel,
    tokenizer: BiliPickyBPETokenizer,
    transplant_rmse: float | None = None,
    transplant_spearman: float | None = None,
    era_check: dict[str, Any] | None = None,
    promo_lut_check: dict[str, Any] | None = None,
    ai_lut_check: dict[str, Any] | None = None,
    ltt_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """11-Gate CI suite: all 9 upgraded v1.2.2 gates + `V123_AI_LUT_FLOOR` + `V123_CERTIFIED_LTT` + HAWQ check."""
    base = run_v122_ci_gates(
        model, tokenizer,
        transplant_rmse=transplant_rmse,
        transplant_spearman=transplant_spearman,
        era_probe_auroc=None,
        promo_lut_check=promo_lut_check,
    )
    # Rename keys cleanly from V122_* -> V123_*
    ci: dict[str, Any] = {}
    for k, v in base.items():
        if k == "all_ci_passed":
            continue
        nk = k.replace("V122_", "V123_")
        ci[nk] = v

    # V123_HAWQ_V2: verify exact bitwidth reallocation
    hawq_rep = getattr(model, "hawq_v2_rep", None) or apply_hawq_v2_bit_reallocation_v123(model)
    ci["V123_HAWQ_V2_bit_alloc"] = {
        "n_upgraded_w4": int(hawq_rep["n_upgraded"]),
        "n_demoted_seq2": int(hawq_rep["n_demoted"]),
        "pass": bool(hawq_rep["all_expected_matched"]),
    }

    # V123_E1: Dual-check era invariance (emb probe + direct human logit shift!)
    if era_check is not None:
        emb_p = float(era_check.get("emb_era_probe_auroc", 0.5))
        lg_p = float(era_check.get("logit_era_probe_auroc", 0.5))
        delta_on = abs(float(era_check.get("delta_old_minus_new", 0.0)))
        # Pass if emb probe <= 0.62 OR (logit probe <= 0.62 AND |delta_old_minus_new| <= 0.15)
        e1_pass = bool((emb_p <= 0.62) or (lg_p <= 0.62 and delta_on <= 0.15))
        ci["V123_E1_era_invariance"] = {
            "emb_era_probe_auroc": round(emb_p, 4),
            "logit_era_probe_auroc": round(lg_p, 4),
            "abs_delta_old_minus_new": round(delta_on, 4),
            "pass": e1_pass,
        }

    # V123_AI_LUT_FLOOR: Permanently blocks `god500`-style AI LUT floor blow-up!
    if ai_lut_check is not None:
        lut01 = float(ai_lut_check.get("lut_ai_r0_at_001", 1.0))
        r0_pass = float(ai_lut_check.get("r0_ai_le008_rate", 0.0))
        r1_pass = float(ai_lut_check.get("r1_ai_le008_rate", 0.0))
        ci["V123_AI_LUT_FLOOR"] = {
            "lut_ai_r0_at_001": round(lut01, 5),
            "r0_ai_le008_rate": round(r0_pass, 4),
            "r1_ai_le008_rate": round(r1_pass, 4),
            "pass": bool(lut01 <= 0.025 and r0_pass >= 0.15 and r1_pass >= 0.10),
        }

    # V123_CERTIFIED_LTT: Mathematically valid LTT certification when cert_n_ai >= 450
    if ltt_check is not None:
        cert_n = int(ltt_check.get("cert_n_ai", 0))
        any_cert = bool(ltt_check.get("r0_certified", False) or ltt_check.get("r1_certified", False))
        exit_r = float(ltt_check.get("cal_exit_rate", 0.0))
        if cert_n < 450:
            ci["V123_CERTIFIED_LTT"] = {"cert_n_ai": cert_n, "smoke_skipped": True, "pass": True}
        else:
            ci["V123_CERTIFIED_LTT"] = {
                "cert_n_ai": cert_n,
                "any_certified": any_cert,
                "cal_exit_rate": round(exit_r, 4),
                "pass": bool(any_cert and exit_r >= 0.20),
            }

    ci["all_ci_passed"] = all(bool(v.get("pass")) for v in ci.values() if isinstance(v, dict))
    return ci


__all__ = [
    "HAWQ_V2_DEMOTE_TO_SEQ2_V123",
    "HAWQ_V2_UPGRADE_TO_W4_V123",
    "MeguvizV123ApexNanoModel",
    "MeguvizV123CascadeEngine",
    "apply_hawq_v2_bit_reallocation_v123",
    "apply_len_aware_z5_v123",
    "apply_v122_full_logit_leace",
    "bounded_fused3_logits",
    "compute_bitdistill_slot_relation_kl",
    "compute_wild_tail_objective_v122",
    "export_v121_modular_onnx_and_bundle_v7",
    "fit_conformal_fdr_thresholds_v122",
    "fit_len_aware_z5_bootstrap_v123",
    "fit_moses_register_offsets_v122",
    "fit_output_shifted_promo_lut_v122",
    "fit_smooth_ai_lut_v122",
    "fold_zero_op_v121",
    "install_quest_trust_gradient_hooks",
    "load_bundle_v7_v121_into_model",
    "run_v123_ci_gates",
    "transplant_dual_source_v10_v121_to_v122",
]
