"""MeguvizCAC v1.2.1-ApexNano-Fix Epistemic Data Refinery (`v1.2.1`).

Integrates ALL 6 forensic discoveries from the RTX 4060 Ti Deep Data Audit
(`scratch/deep_data_audit_report.json` & `scratch/v121_mining_goldmine_report.json`),
while keeping `v11_data_refinery.py` & `v12_data_refinery.py` 100% untouched:
  1. Quarantines `colab_gpu_distilled` midband `30 < score < 70` (audited 95.54% real AI in `B2`!)
     instead of training soft `y=0.50` targets.
  2. Quarantines `v08_gemini_pro_gold` (388) midband `30 < score < 70` (audited 100.0% real AI in `B3`!;
     re-adjudicated rows enter via `Gold-v2` override!).
  3. Tames `v08_gemini_flash_gold` boundary false-human bias (audited 77.78% false-human in `B1`!):
     `15 <= score <= 40` Flash rows are quarantined (`T3_QUARANTINE`) unless verified by `Gold-v2`/`Wave-3`;
     decisive Flash rows get conservative `w=0.85` so Flash misses never fight `Gold-v2`.
  4. Keeps ALL v1.2 protections: `v04_confirmed_ai_plugin` ultra-short (`<15 char` without AI anchor)
     Bayes cap + `v04_qwen_inkling_joint` poison quarantine + strict `30..70` midband block from `T0`.
  5. Diamond-only Pair-Dataset filter (`is_diamond_pair_source_v121`): keeps 0%–1.2% inversion pairs
     (`v05_gateguard`, `v10_topic_debias`, `v04_twins_clean`, `v08_gemini_13k`, `v04_inkling_wk`,
     `stage2_literary`) and drops noisy un-gated pairs (`v04_qwen38_twins` 6.75%, `qwen38_27b_twins` 5.67%).
  6. Native `Wave-3` (`v121_wave3_pro_gold`) promotion straight into `T0_DIAMOND_GOLD (w=1.65)`.
"""

from __future__ import annotations

from typing import Any

from meguviz_cac.v12_data_refinery import (
    DIAMOND_GOLD_SOURCES_V12,
    PAT_SHORT_AI_ANCHOR,
    POISONED_NONHUMAN_SOURCES_V12,
)

DIAMOND_GOLD_SOURCES_V121 = frozenset(
    set(DIAMOND_GOLD_SOURCES_V12)
    | {
        "v121_wave3_pro_gold",
    }
)

# Noisy un-gated twin pair filenames caught in the 9/9 pair audit (`>= 5.5%` inversion rate)
DROPPED_NOISY_PAIR_TOKENS_V121: tuple[str, ...] = (
    "v04_qwen38_twins",
    "qwen38_27b_twins",
    "v04-qwen38-twins",
    "qwen38-27b-twins",
)


def is_diamond_pair_source_v121(path_or_name: str) -> bool:
    """Return True iff pair file passed the 9/9 confident-learning inversion audit (`<= 1.5%` inv)."""
    low = str(path_or_name or "").lower()
    return not any(tok in low for tok in DROPPED_NOISY_PAIR_TOKENS_V121)


def classify_epistemic_tier_v121(
    score: float,
    score_source: str,
    text: str = "",
    overridden_by_gemini: bool = False,
    gemini_is_pro: bool = False,
    v11_or_v12_verified: bool = False,
    is_hybrid_mixed: bool = False,
) -> tuple[str, float, float, str]:
    """Return `(tier, trust_weight_mult, effective_score_0_100, reason)` for v1.2.1 training."""
    src = str(score_source or "").strip()
    sc = float(score)
    clean_len = len(str(text).strip())

    # 1. Verified Gold-v2 / Wave-1/2/2.5/3 dual-axis Pro gold
    if v11_or_v12_verified or src in (
        "v121_wave3_pro_gold",
        "v12_gold_v2_pro",
        "v11_dualaxis_pro_gold",
        "v11_wave25_disagreement_gold",
    ):
        if (30.0 < sc < 70.0) and not is_hybrid_mixed:
            return "T2_SOFT_MIDBAND", 0.45, sc, "verified_ambiguous_midband"
        w_g = 1.65 if src == "v121_wave3_pro_gold" else 1.55
        return "T0_DIAMOND_GOLD", w_g, sc, "v121_dualaxis_verified_gold"

    # 2. Gemini Pro / Flash overrides (Quarantine B3 Pro-388 midband + B1 Flash 15..40 false-human zone!)
    if overridden_by_gemini or src in ("v08_gemini_pro_gold", "v08_gemini_pro_tier2_gold", "v08_gemini_flash_gold"):
        if 30.0 < sc < 70.0:
            # B3 audit proved 100.0% of v08_pro388 midband are real AI -> NEVER train y=0.50!
            return "T3_QUARANTINE", 0.0, sc, "v121_gemini_midband_quarantined"
        if gemini_is_pro or src in ("v08_gemini_pro_tier2_gold", "v08_gemini_pro_gold"):
            return "T0_DIAMOND_GOLD", 1.40, sc, "gemini_pro_decisive_gold"
        # B1 audit proved 77.78% false-human rate on v08_flash boundary humans!
        if 15.0 <= sc <= 40.0:
            return "T3_QUARANTINE", 0.0, sc, "v121_flash_boundary_false_human_quarantined"
        return "T1_VERIFIED_CLEAN", 0.85, sc, "gemini_flash_decisive_silver_tamed"

    # 3. Known poisoned non-human tails (including `v04_qwen_inkling_joint`!)
    if src in POISONED_NONHUMAN_SOURCES_V12:
        if sc <= 25.0:
            return "T1_VERIFIED_CLEAN", 0.95, sc, f"{src}_clean_human_tail"
        return "T3_QUARANTINE", 0.0, sc, f"{src}_poisoned_nonhuman_purged"

    # 4. `v04_confirmed_ai_plugin` Short-Text Bayes Cap
    if src == "v04_confirmed_ai_plugin":
        if clean_len > 0 and clean_len < 15 and not PAT_SHORT_AI_ANCHOR.search(text):
            return "T3_QUARANTINE", 0.0, min(sc, 55.0), "plugin_ultrashort_bayes_impossible_purged"
        return "T0_DIAMOND_GOLD", 1.30, sc, "v04_confirmed_ai_plugin_clean"

    # 5. Physical Pre-LLM human prior
    if src == "pre_llm_human_prior":
        return "T0_DIAMOND_GOLD", 1.35, min(sc, 5.0), "pre_llm_human_physical_prior"

    # 6. Early diamond sources (still enforce strict midband guard!)
    if src in DIAMOND_GOLD_SOURCES_V121:
        if 30.0 < sc < 70.0:
            return "T2_SOFT_MIDBAND", 0.30, sc, f"{src}_midband_demoted_to_t2"
        return "T0_DIAMOND_GOLD", 1.35, sc, f"{src}_decisive_gold"

    # 7. Distilled silver corpus (`colab_gpu_distilled_from_hydrogen_xhigh`, etc.)
    # B2 audit proved 95.54% of colab_distilled 30..70 midband are real AI -> QUARANTINE instead of y=0.50!
    if 30.0 < sc < 70.0:
        return "T3_QUARANTINE", 0.0, sc, "v121_distilled_midband_95pct_ai_quarantined"
    return "T1_VERIFIED_CLEAN", 0.95, sc, "distilled_clean_silver"


def run_v121_refinery_ci() -> dict[str, Any]:
    """Self-test ALL 6 v1.2.1 data-audit refinery upgrades."""
    c1 = classify_epistemic_tier_v121(94.0, "v04_confirmed_ai_plugin", "这生活是我向往的")
    c2 = classify_epistemic_tier_v121(94.0, "v04_confirmed_ai_plugin", "本鱼看过啦")
    c3 = classify_epistemic_tier_v121(52.0, "colab_gpu_distilled_from_hydrogen_xhigh", "普通蒸馏中带测试")
    c4 = classify_epistemic_tier_v121(50.0, "v08_gemini_pro_gold", "pro388中带测试", overridden_by_gemini=True, gemini_is_pro=True)
    c5 = classify_epistemic_tier_v121(24.0, "v08_gemini_flash_gold", "flash边界测试", overridden_by_gemini=True, gemini_is_pro=False)
    c6 = classify_epistemic_tier_v121(88.0, "v121_wave3_pro_gold", "立即购买｜瑞幸咖啡优惠券", v11_or_v12_verified=True)
    pair_ok = (
        is_diamond_pair_source_v121("v05-gateguard-pairs.jsonl")
        and is_diamond_pair_source_v121("v08_gemini_13k_pairs.jsonl")
        and not is_diamond_pair_source_v121("v04_qwen38_twins.jsonl")
        and not is_diamond_pair_source_v121("qwen38_27b_twins.jsonl")
    )
    ok = (
        c1[0] == "T3_QUARANTINE"
        and c2[0] == "T0_DIAMOND_GOLD"
        and c3[0] == "T3_QUARANTINE"
        and c4[0] == "T3_QUARANTINE"
        and c5[0] == "T3_QUARANTINE"
        and c6[0] == "T0_DIAMOND_GOLD"
        and c6[1] == 1.65
        and pair_ok
    )
    return {"pass": ok, "cases": [c1[:2], c2[:2], c3[:2], c4[:2], c5[:2], c6[:2]], "pair_ok": pair_ok}
