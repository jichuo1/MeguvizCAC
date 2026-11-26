"""MeguvizCAC v1.2.0-ApexNano-Omni Epistemic Data Refinery (v1.2).

Extends `v11_data_refinery.py` with ALL 3 zero-cost surgical upgrades discovered in our
full-corpus forensic audit (2026-11-26):
  1. `v04_confirmed_ai_plugin` Short-Text Bayes Cap:
     - 5.9% (~14,750 rows) are ultra-short (<15 chars) generic human-like phrases hard-labeled 94
       (e.g., "这生活是我向往的", "这条必须转给同事看"). Without strong AI pragmatic anchors
       (persona self-ref / em-dash / kaomoji / recap marker), they are Bayes-indistinguishable from
       human comments and poison short-text precision (OjinjinBench v2 R@P95).
     - Downgraded from `T0_DIAMOND_GOLD` -> `T2_SOFT_MIDBAND` (`w=0.15`, soft-capped `y<=0.55`) or dropped!
  2. `v04_qwen_inkling_joint` (349 rows, empirically 36.4% False-AI!) added to `POISONED_NONHUMAN_SOURCES`
     so its `score >= 30` tail is cleanly quarantined in `T3_QUARANTINE`.
  3. Midband (`30 < score < 70`) Hard-Block on `T0_DIAMOND_GOLD`:
     - Prevents ambiguous midband rows — especially `v08_gemini_pro_gold` (388 rows where 91.5% are
       in `30..70` with `AUC=0.4399`) and early `hydrogen_xhigh` midband — from receiving `1.35x-1.40x`
       `T0` anchor weight; routes them cleanly to `T2_SOFT_MIDBAND (w=0.30)`.
  4. Native `Gold-v2` (39-shard re-adjudicated dual-axis gold) + `Wave-2.5` (243 wild disagreement rows)
     promotion straight into `T0_DIAMOND_GOLD (w=1.55)`!
"""

from __future__ import annotations

import re
from typing import Any

from meguviz_cac.v11_data_refinery import (
    DIAMOND_GOLD_SOURCES as _V11_DIAMOND,
    POISONED_NONHUMAN_SOURCES as _V11_POISON,
    length_bucket_v11,
    primary_domain_tag_v11,
)

# Pragmatic anchors that genuinely identify ultra-short (<15 char) AI / persona-bot comments
PAT_SHORT_AI_ANCHOR = re.compile(
    r"(本鱼|本鲸|本喵|本机|我的主人|提示词|参数|——|—|\(￣|｡•ω•｡|课代表|AI生成|AI总结)",
    re.I,
)

DIAMOND_GOLD_SOURCES_V12 = frozenset(
    set(_V11_DIAMOND)
    | {
        "v12_gold_v2_pro",
        "v11_wave25_disagreement_gold",
        "v08_gemini_pro_tier2_gold",
    }
)

# Includes `v04_qwen_inkling_joint` (36.4% False-AI caught in forensic audit!)
POISONED_NONHUMAN_SOURCES_V12 = frozenset(
    set(_V11_POISON)
    | {
        "v04_qwen_inkling_joint",
    }
)


def classify_epistemic_tier_v12(
    score: float,
    score_source: str,
    text: str = "",
    overridden_by_gemini: bool = False,
    gemini_is_pro: bool = False,
    v11_or_v12_verified: bool = False,
    is_hybrid_mixed: bool = False,
) -> tuple[str, float, float, str]:
    """Return `(tier, trust_weight_mult, effective_score_0_100, reason)` for v1.2.0 training.

    Unlike v1.1, also returns `effective_score_0_100` so Bayes-impossible ultra-short synthetic
    `v04_confirmed_ai_plugin` clichés are soft-capped instead of hard-forced to 94!
    """
    src = str(score_source or "").strip()
    sc = float(score)
    clean_len = len(str(text).strip())

    # 1. Verified Gold-v2 / Wave-1/2/2.5 dual-axis gold
    if v11_or_v12_verified or src in ("v12_gold_v2_pro", "v11_dualaxis_pro_gold", "v11_wave25_disagreement_gold"):
        if (30.0 < sc < 70.0) and not is_hybrid_mixed:
            return "T2_SOFT_MIDBAND", 0.45, sc, "verified_ambiguous_midband"
        return "T0_DIAMOND_GOLD", 1.55, sc, "v12_dualaxis_verified_gold"

    # 2. Gemini Pro / Flash overrides (strict midband guard: NEVER let 30..70 enter T0!)
    if overridden_by_gemini or src in ("v08_gemini_pro_gold", "v08_gemini_pro_tier2_gold", "v08_gemini_flash_gold"):
        if 30.0 < sc < 70.0:
            return "T2_SOFT_MIDBAND", 0.30, sc, "gemini_midband_blocked_from_t0"
        if gemini_is_pro or src in ("v08_gemini_pro_tier2_gold", "v08_gemini_pro_gold"):
            return "T0_DIAMOND_GOLD", 1.40, sc, "gemini_pro_decisive_gold"
        return "T1_VERIFIED_CLEAN", 1.10, sc, "gemini_flash_decisive_silver"

    # 3. Known poisoned non-human tails (including newly added `v04_qwen_inkling_joint`!)
    if src in POISONED_NONHUMAN_SOURCES_V12:
        if sc <= 25.0:
            return "T1_VERIFIED_CLEAN", 0.95, sc, f"{src}_clean_human_tail"
        return "T3_QUARANTINE", 0.0, sc, f"{src}_poisoned_nonhuman_purged"

    # 4. `v04_confirmed_ai_plugin` Short-Text Bayes Cap (cures ~14,750 short human-like clichés hard-labeled 94!)
    if src == "v04_confirmed_ai_plugin":
        if clean_len > 0 and clean_len < 15 and not PAT_SHORT_AI_ANCHOR.search(text):
            return "T3_QUARANTINE", 0.0, min(sc, 55.0), "plugin_ultrashort_bayes_impossible_purged"
        return "T0_DIAMOND_GOLD", 1.30, sc, "v04_confirmed_ai_plugin_clean"

    # 5. Physical Pre-LLM human prior
    if src == "pre_llm_human_prior":
        return "T0_DIAMOND_GOLD", 1.35, min(sc, 5.0), "pre_llm_human_physical_prior"

    # 6. Early diamond sources (still enforce strict midband guard!)
    if src in DIAMOND_GOLD_SOURCES_V12:
        if 30.0 < sc < 70.0:
            return "T2_SOFT_MIDBAND", 0.35, sc, f"{src}_midband_demoted_to_t2"
        return "T0_DIAMOND_GOLD", 1.35, sc, f"{src}_decisive_gold"

    # 7. Distilled silver corpus (`colab_gpu_distilled_from_hydrogen_xhigh`, etc.)
    if 30.0 < sc < 70.0:
        return "T2_SOFT_MIDBAND", 0.25, sc, "distilled_midband"
    return "T1_VERIFIED_CLEAN", 0.95, sc, "distilled_clean_silver"


def run_v12_refinery_ci() -> dict[str, Any]:
    """Self-test all 4 v1.2 refinery upgrades."""
    # 1. Ultra-short generic plugin cliché purged
    t1, w1, _, _ = classify_epistemic_tier_v12(94.0, "v04_confirmed_ai_plugin", "这生活是我向往的")
    # 2. Ultra-short persona bot KEPT as T0
    t2, w2, _, _ = classify_epistemic_tier_v12(94.0, "v04_confirmed_ai_plugin", "本鱼看过啦")
    # 3. v04_qwen_inkling_joint AI quarantined
    t3, w3, _, _ = classify_epistemic_tier_v12(85.0, "v04_qwen_inkling_joint", "这是一条被误标的普通长评论测试文本")
    # 4. v08_gemini_pro_gold midband blocked from T0
    t4, w4, _, _ = classify_epistemic_tier_v12(50.0, "v08_gemini_pro_gold", "测试中带评论内容", overridden_by_gemini=True, gemini_is_pro=True)
    ok = (t1 == "T3_QUARANTINE" and w1 == 0.0 and t2 == "T0_DIAMOND_GOLD" and t3 == "T3_QUARANTINE" and t4 == "T2_SOFT_MIDBAND")
    return {"pass": ok, "cases": [(t1, w1), (t2, w2), (t3, w3), (t4, w4)]}
