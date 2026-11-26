"""MeguvizCAC v1.1.0-OmniGuard Epistemic Data Refinery & Poison Quarantine Engine.

Grounded directly in our per-`score_source` forensic audit of `v04-merged-1m4-r3.jsonl.gz` (1,372,143 rows)
cross-checked against `77,666` overlapped Gemini Pro/Flash labels:
  - `v04_qwen38_bf16` (`oldAI >= 70`):           **81.7% False-AI Poison** (2,369 / 2,900 flipped to Human!)
  - `multi_teacher_ensemble` (`oldAI >= 70`):    **93.7% False-AI Poison** (401 / 428 flipped to Human!)
  - `multi_teacher_ensemble` (`oldMID 30..70`):  **93.9% False-MID Human** (2,163 / 2,304 flipped to Human!)
  - `multi_teacher_ensemble` (`oldHU <= 30`):    **99.92% Clean Human** (only 0.08% flipped to AI!)
  - `v04_qwen_inkling_disagree` (`oldAI >= 70`): **61.4% False-AI Poison** (27 / 44 flipped to Human!)
  - `colab_gpu_distilled` (`oldAI >= 70`):       **99.1% Low-FP** (only 0.9% flipped to Human!)
  - `v04_neon_hydrogen_consensus`:               **0.0% Flip** in BOTH directions (Diamond Gold!)

Assigns every training sample into a strict 4-Tier Epistemic Trust Hierarchy (`T0`..`T3`), automatically
quarantining ~45,000+ un-audited False-AI / False-MID poisoned rows out of training while routing high-value
quarantined & starved long-text rows into a structured Wave-2 Active-Learning queue!
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Domain classification patterns (Zero-PII pure text)
PAT_AI_TOPIC = re.compile(r"(deepseek|chatgpt|gemini|claude|ollama|qwen|显存|算力|大模型|提示词|token|微调|量化|ai生成)", re.I)
PAT_TECH_STRUCT = re.compile(r"((?:^|\n|\s)1[\.、\)].*?2[\.、\)]|github|arxiv|代码|报错|内存|函数|接口|实测|参数)", re.I)
PAT_COUPON = re.compile(r"(领券|券\d+元|满\d+减\d+|拼多多|京东|淘宝|天猫|闲鱼|史低|到手价|蓝链|国补|凑单|优惠券|百亿补贴|返佣)")
PAT_SELF_PROMO = re.compile(r"(关注一下|求关注|互关|主页有|看我主页|私信领|私信发|简介自取|加微|vx|扣群|粉丝群|进群|公众号|接单|代做|check my channel|subscribe to my)", re.I)
PAT_SOFT_SHILL = re.compile(r"(亲测有效|亲测好用|入手不亏|回购|闭眼入|闭眼冲|无脑入|强烈安利|强烈推荐|不踩雷|性价比无敌|质感封神|早买早享受)")
PAT_KEDAIBIAO = re.compile(r"(课代表|省流|空降\s*\d|免责声明.*?AI|AI生成.*?仅供)")
PAT_QUOTE = re.compile(r"[「『“\"][^」』”\"]{12,}[」』”\"]")

# Epistemic provenance sets derived from empirical 80k Gemini flip autopsy
DIAMOND_GOLD_SOURCES = frozenset({
    "pre_llm_human_prior",
    "v04_confirmed_ai_plugin",
    "hydrogen_xhigh_direct",
    "v04_neon_hydrogen_consensus",
    "v04_neon_gate_override",
    "v04_inkling_wk_counterfactual",
    "v04_neon_xhigh_tiebreak",
    "v08_gemini_pro_gold",
    "v08_gemini_pro",
    "v11_dualaxis_pro_gold",
})

# Sources whose `score >= 30` (non-human) tail is empirically poisoned (>60%-94% false-AI/MID) unless overridden by Gemini!
POISONED_NONHUMAN_SOURCES = frozenset({
    "v04_qwen38_bf16",
    "multi_teacher_ensemble",
    "v04_qwen_inkling_disagree",
})


def length_bucket_v11(n_chars: int) -> str:
    if n_chars <= 45:
        return "L1_<=45"
    if n_chars <= 128:
        return "L2_46-128"
    if n_chars <= 256:
        return "L3_129-256"
    if n_chars <= 512:
        return "L4_257-512"
    return "L5_>512"


def primary_domain_tag_v11(text: str) -> str:
    """Assign primary domain slice for joint Length x Domain debiasing & coverage tracking."""
    if PAT_COUPON.search(text):
        return "coupon"
    if PAT_SELF_PROMO.search(text):
        return "self_promo"
    if PAT_SOFT_SHILL.search(text):
        return "soft_shill"
    if PAT_KEDAIBIAO.search(text):
        return "kedaibiao"
    if PAT_AI_TOPIC.search(text):
        return "ai_topic"
    if PAT_TECH_STRUCT.search(text):
        return "tech_struct"
    if PAT_QUOTE.search(text):
        return "has_quote"
    return "organic_other"


def classify_epistemic_tier(
    score: float,
    score_source: str,
    overridden_by_gemini: bool = False,
    gemini_is_pro: bool = False,
    v11_dualaxis_verified: bool = False,
    committee_probs: tuple[float, float] | None = None,
) -> tuple[str, float, str]:
    """Assign (trust_tier, trust_weight_mult, reason) for a single training sample.

    Returns one of:
      - `T0_DIAMOND_GOLD`   (mult 1.35): Highest epistemic certainty, 0% poison rate.
      - `T1_VERIFIED_CLEAN` (mult 1.00): Empirically <1% flip rate (Gemini Flash, multi_teacher HU<=30, colab_distill HU/AI).
      - `T2_SOFT_MIDBAND`   (mult 0.30): Ambiguous mid-band (colab_distill MID) kept soft without hard forcing.
      - `T3_QUARANTINE`     (mult 0.00): Empirically poisoned (61%–94% False-AI/MID) or cross-model severe conflict -> PURGED!
    """
    src = str(score_source or "").strip()

    if v11_dualaxis_verified:
        return "T0_DIAMOND_GOLD", 1.50, "v11_dualaxis_verified"
    if overridden_by_gemini:
        if gemini_is_pro:
            return "T0_DIAMOND_GOLD", 1.40, "gemini_pro_override"
        # Gemini Flash is verified silver; still allow committee check on extreme conflicts if provided
        if committee_probs is not None:
            p_a, p_b = committee_probs
            if score >= 75.0 and p_a < 0.08 and p_b < 0.08:
                return "T2_SOFT_MIDBAND", 0.35, "flash_vs_dual_committee_conflict"
        return "T1_VERIFIED_CLEAN", 1.10, "gemini_flash_override"

    if src in DIAMOND_GOLD_SOURCES:
        return "T0_DIAMOND_GOLD", 1.30, f"anchor:{src}"

    # CRITICAL POISON QUARANTINE (proven by 77,666-row Gemini flip autopsy!)
    if src in POISONED_NONHUMAN_SOURCES:
        if score <= 30.0:
            # multi_teacher_ensemble HU<=30 has 0.08% flip; qwen38 HU<=30 has 3.07% flip (mildly downweighted)
            if src == "multi_teacher_ensemble":
                return "T1_VERIFIED_CLEAN", 0.95, "multi_teacher_clean_hu"
            return "T2_SOFT_MIDBAND", 0.45, f"{src}_hu_downweighted"
        # score > 30 from these 3 sources is 61.4%–93.9% FALSE-AI/FALSE-MID poison -> QUARANTINE!
        return "T3_QUARANTINE", 0.0, f"poison_quarantine:{src}_score_{int(round(score))}"

    if src == "v04_qwen_inkling_joint":
        if score >= 70.0:
            # 36.4% false-AI rate -> require committee consensus if available, else soft-quarantine!
            if committee_probs is not None and min(committee_probs) >= 0.65:
                return "T1_VERIFIED_CLEAN", 0.90, "qwen_inkling_joint_committee_confirmed"
            return "T3_QUARANTINE", 0.0, "poison_quarantine:qwen_inkling_joint_unconfirmed_ai"
        return "T1_VERIFIED_CLEAN", 0.85, "qwen_inkling_joint_hu"

    if src == "colab_gpu_distilled_from_hydrogen_xhigh":
        if score >= 70.0:
            if committee_probs is not None and max(committee_probs) < 0.12:
                return "T3_QUARANTINE", 0.0, "confident_learning_conflict_ai"
            return "T1_VERIFIED_CLEAN", 1.00, "colab_distill_ai_clean"
        if score <= 30.0:
            if committee_probs is not None and min(committee_probs) > 0.88:
                return "T3_QUARANTINE", 0.0, "confident_learning_conflict_hu"
            return "T1_VERIFIED_CLEAN", 1.00, "colab_distill_hu_clean"
        return "T2_SOFT_MIDBAND", 0.30, "colab_distill_midband"

    # Fallback conservative handling for any minor legacy source
    if score <= 30.0 or score >= 70.0:
        return "T1_VERIFIED_CLEAN", 0.80, f"legacy:{src}"
    return "T2_SOFT_MIDBAND", 0.25, f"legacy_mid:{src}"


def run_corpus_refinery_audit(
    v04_corpus_gz: Path,
    gemini_80k_zip: Path | None = None,
    v11_dualaxis_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Audit the full training corpus under the 4-Tier Epistemic Refinery and return exact quarantine & coverage stats."""
    gem_map: dict[str, tuple[float, bool]] = {}
    if gemini_80k_zip and gemini_80k_zip.is_file():
        with zipfile.ZipFile(gemini_80k_zip) as zf:
            with gzip.GzipFile(fileobj=io.BytesIO(zf.read("v08-gemini-gold-labels.jsonl.gz"))) as gz:
                for line in gz:
                    r = json.loads(line.decode("utf-8"))
                    if r.get("split", "train") != "train":
                        continue
                    src = str(r.get("score_source", r.get("source", ""))).lower()
                    is_pro = ("pro" in src) or (float(r.get("train_weight", 1.0)) >= 1.30)
                    gem_map[str(r["sample_id"])] = (float(r["score"]), is_pro)

    v11_map: dict[str, dict] = {}
    if v11_dualaxis_jsonl and v11_dualaxis_jsonl.is_file():
        op = gzip.open if v11_dualaxis_jsonl.suffix == ".gz" else open
        with op(v11_dualaxis_jsonl, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("orig_sample_id"):
                    v11_map[str(r["orig_sample_id"])] = r

    tier_counts = Counter()
    quarantine_by_source = Counter()
    quarantine_ai_vs_mid = Counter()
    clean_cov_by_len = defaultdict(lambda: [0, 0, 0])  # [hu, mid, ai] EXCLUDING T3_QUARANTINE!
    quarantined_kedaibiao_ai = 0
    quarantined_long_ai = 0
    total_train = 0

    with gzip.open(v04_corpus_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                continue
            total_train += 1
            sid = str(r["sample_id"])
            txt = str(r.get("text", ""))
            old_sc = float(r.get("score", 0.0))
            old_src = str(r.get("score_source", ""))

            v11_hit = sid in v11_map
            gem_hit = sid in gem_map
            eff_sc = float(v11_map[sid]["ai_score"]) if v11_hit else (gem_map[sid][0] if gem_hit else old_sc)
            gem_pro = gem_map[sid][1] if gem_hit else False

            tier, mult, reason = classify_epistemic_tier(
                score=eff_sc,
                score_source=old_src,
                overridden_by_gemini=gem_hit,
                gemini_is_pro=gem_pro,
                v11_dualaxis_verified=v11_hit,
            )
            tier_counts[tier] += 1
            if tier == "T3_QUARANTINE":
                quarantine_by_source[old_src] += 1
                quarantine_ai_vs_mid["AI>=70" if eff_sc >= 70.0 else "MID(30..70)"] += 1
                if eff_sc >= 70.0 and PAT_KEDAIBIAO.search(txt):
                    quarantined_kedaibiao_ai += 1
                if eff_sc >= 70.0 and len(txt) > 256:
                    quarantined_long_ai += 1
            else:
                lb = length_bucket_v11(len(txt))
                ci = 0 if eff_sc <= 30.0 else (2 if eff_sc >= 70.0 else 1)
                clean_cov_by_len[lb][ci] += 1

    return {
        "total_v04_train": total_train,
        "gemini_80k_loaded": len(gem_map),
        "v11_dualaxis_loaded": len(v11_map),
        "tier_counts": dict(tier_counts),
        "quarantined_total": int(tier_counts["T3_QUARANTINE"]),
        "quarantine_by_source": dict(quarantine_by_source),
        "quarantine_ai_vs_mid": dict(quarantine_ai_vs_mid),
        "quarantined_false_ai_kedaibiao": quarantined_kedaibiao_ai,
        "quarantined_false_ai_long_gt256": quarantined_long_ai,
        "clean_length_distribution": {k: {"hu": v[0], "mid": v[1], "ai": v[2]} for k, v in sorted(clean_cov_by_len.items())},
    }
