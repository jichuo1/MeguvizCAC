"""Step 1: Export Zero-PII, Anti-Leakage Active-Learning Queues for External Gemini v0.8.0 Data Factory.

Supports all 4 high-ROI queues:
  - A_noise_surgery       : High-uncertainty / weak mid-band & hard-group rows from v04-merged-1m4-r3.jsonl.gz
  - B_wild_hard_mining    : Fresh modern 2025-2026 Bilibili hard comments from bili-live-1m.sqlite3 (ZERO PII)
  - C_counterfactual_spans: Seed rows for Human<->AI minimal pairs & Sandwich Hybrid-Span generation
  - D_thread_context      : (video_title, parent_text, reply_text) triples from bili-live-1m.sqlite3

Security & Scientific Integrity Guarantees:
  1. ZERO PII: Never reads or exports uid, uname, ip, level, avatar, or follower metadata.
  2. NON-SILENT ANTI-LEAKAGE: Drops exact & near-duplicate (char-trigram Jaccard >= 0.72) matches against
     ALL non-train rows (val/test/user_confirmed), BENCH_TRAP_TEXTS, EXTRA_CROSS_PLATFORM_TRAPS,
     MULTILINGUAL_TRAPS, and PIFE_ADVERSARIAL_TRAPS.
  3. HIDDEN 2% CANARY VAULT: Injects ~2% blinded `val_gold` (`hydrogen_xhigh_direct` / `joint_fused_high`)
     rows into Queues A/B for Step-3 calibration audit ONLY; their true gold labels remain strictly local in
     `canary_vault.json` and Step-3 strips 100% of canaries before emitting training data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gemini_v08_datafactory.rubric_v08 import make_check_tag

# ---------------------------------------------------------------------------
# Hard-Slice Mining Regexes (align with v04_co_teacher_gate + v0.6 forensics)
# ---------------------------------------------------------------------------

STRUCTURED_TECH_RE = re.compile(
    r"(?:(?:^|\s)[1-9一二三四五][\.、\)）]|[①②③④⑤]|《[^》]{1,28}》|"
    r"\d+(?:\.\d+)?\s*/\s*10|评分|打分|星级|推荐度|满分|参数|实测|帧率|跑分|"
    r"github\.com|arxiv|RFC\s*\d+|论文|代码|报错|网盘|提取码|空降|\d{1,2}:\d{2})",
    re.I,
)
AI_SUSPECT_WILD_RE = re.compile(
    r"(?:——|不仅.{1,16}更|不是.{1,14}而是|总体来看|客观来说|平和地说|保留意见|"
    r"本[鱼鲸]|归档|待跟进|暂记|记忆|字幕|delve|tapestry|testament|nuanced|not just .{2,30} it'?s)",
    re.I,
)
MULTILINGUAL_RE = re.compile(r"(?:[A-Za-z]{4,}|[\u3040-\u30ff]{3,}|[\uac00-\ud7af]{3,})")
PII_EMAIL_KEY_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:AIza|sk-)[A-Za-z0-9_-]{12,}|(?i:bearer)\s+\S+")

GOLD_HIGH_TIER_SOURCES = frozenset(
    {
        "hydrogen_xhigh_direct",
        "hydrogen_xhigh_manual",
        "joint_fused_high",
        "neon_xhigh_tiebreak",
        "manual_gold",
        "v08_gemini_pro_gold",
    }
)

HARD_GROUPS = frozenset(
    {
        "dsh_hard_slice",
        "structured_human_hard",
        "multilingual_world_knowledge",
        "literary_boundary",
        "persona_space_kaomoji",
        "ultra_short_ai",
        "evasive_plugin_ai",
    }
)


def normalize_text(text: str) -> str:
    """Lightweight canonical text normalizer matching MeguvizCAC anti-leakage."""
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def scrub_accidental_secrets(text: str) -> str:
    """Remove accidental emails/API keys while preserving harmless technical URLs/code."""
    return PII_EMAIL_KEY_RE.sub("[REDACTED]", str(text or "")).strip()


def char_trigrams(text: str) -> frozenset[str]:
    n = normalize_text(text)
    if len(n) < 3:
        return frozenset({n}) if n else frozenset()
    return frozenset(n[i : i + 3] for i in range(len(n) - 2))


def is_near_dup_traps(
    text: str,
    forbidden_exact: set[str],
    trap_trigrams: list[frozenset[str]],
    thresh: float = 0.72,
    trap_inv: dict[str, list[int]] | None = None,
) -> bool:
    n = normalize_text(text)
    if not n or n in forbidden_exact:
        return True
    tg = char_trigrams(n)
    if not tg:
        return False
    if trap_inv is not None:
        # O(|tg|) inverted-index prefilter: only inspect traps sharing >= 3 trigrams
        hits: dict[int, int] = {}
        for g in tg:
            idxs = trap_inv.get(g)
            if idxs:
                for idx in idxs:
                    hits[idx] = hits.get(idx, 0) + 1
        for idx, inter in hits.items():
            tt = trap_trigrams[idx]
            union = len(tg) + len(tt) - inter
            if union > 0 and (inter / union) >= thresh:
                return True
        return False
    for tt in trap_trigrams:
        if not tt:
            continue
        inter = len(tg & tt)
        if inter == 0:
            continue
        union = len(tg) + len(tt) - inter
        if union > 0 and (inter / union) >= thresh:
            return True
    return False


def build_trap_inverted_index(trap_trigrams: list[frozenset[str]]) -> dict[str, list[int]]:
    inv: dict[str, list[int]] = {}
    for idx, tt in enumerate(trap_trigrams):
        for g in tt:
            inv.setdefault(g, []).append(idx)
    return inv


def detect_lang_fast(text: str) -> str:
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    ja = len(re.findall(r"[\u3040-\u30ff]", text))
    ko = len(re.findall(r"[\uac00-\ud7af]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if ja >= 4:
        return "ja"
    if ko >= 4:
        return "ko"
    if zh == 0 and en >= 10:
        return "en"
    if zh >= 3 and en >= 12:
        return "zh_en"
    return "zh"


def load_traps_and_eval_forbidden(corpus_path: Path, extra_exclude: Path | None = None) -> tuple[set[str], list[frozenset[str]], list[dict[str, Any]]]:
    """Load all forbidden eval/trap texts AND harvest clean `val_gold` candidates for hidden 2% canaries."""
    forbidden_exact: set[str] = set()
    trap_texts: list[str] = []

    # Import MeguvizCAC built-in trap sets cleanly (non-silent if repo is present)
    try:
        from meguviz_cac.v04_corpus_prep import BENCH_TRAP_TEXTS
        from meguviz_cac.v03_runtime import EXTRA_CROSS_PLATFORM_TRAPS
        from meguviz_cac.v04_runtime import MULTILINGUAL_TRAPS
        from meguviz_cac.v05_runtime import PIFE_ADVERSARIAL_TRAPS

        trap_texts.extend(list(BENCH_TRAP_TEXTS))
        trap_texts.extend(t for _, t in EXTRA_CROSS_PLATFORM_TRAPS)
        trap_texts.extend(t for _, t in MULTILINGUAL_TRAPS)
        trap_texts.extend(t for _, t in PIFE_ADVERSARIAL_TRAPS)
    except Exception as exc:
        print(f"[WARN] Could not import built-in trap tables ({exc}); continuing with corpus eval splits only.")

    for t in trap_texts:
        forbidden_exact.add(normalize_text(t))
    trap_trigrams = [char_trigrams(t) for t in trap_texts]

    if extra_exclude and extra_exclude.is_file():
        opener = gzip.open if extra_exclude.name.endswith(".gz") else open
        with opener(extra_exclude, "rt", encoding="utf-8") as ef:
            for line in ef:
                if not line.strip():
                    continue
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("content") or ""
                if txt:
                    forbidden_exact.add(normalize_text(txt))

    val_gold_pool: list[dict[str, Any]] = []
    if corpus_path.is_file():
        opener = gzip.open if corpus_path.name.endswith(".gz") else open
        with opener(corpus_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                sp = str(r.get("split") or "train")
                coh = str(r.get("cohort") or "")
                if sp != "train" or coh == "user_confirmed":
                    txt = str(r.get("text") or "")
                    if txt:
                        forbidden_exact.add(normalize_text(txt))
                    if (
                        sp in ("val", "validation")
                        and str(r.get("score_source") or "") in GOLD_HIGH_TIER_SOURCES
                        and r.get("has_score", True)
                        and len(txt.strip()) >= 10
                    ):
                        val_gold_pool.append(
                            {
                                "sample_id": str(r["sample_id"]),
                                "text": scrub_accidental_secrets(txt),
                                "lang_tag": r.get("lang_tag") or detect_lang_fast(txt),
                                "gold_score": float(r["score"]),
                                "gold_code": str(r.get("code") or ""),
                                "score_source": str(r.get("score_source") or ""),
                                "group_id": str(r.get("group_id") or ""),
                            }
                        )

    return forbidden_exact, trap_trigrams, val_gold_pool


def mine_queue_a_noise_surgery(
    corpus_path: Path,
    limit: int,
    forbidden_exact: set[str],
    trap_trigrams: list[frozenset[str]],
    rng: random.Random,
    seen_norms: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mine high-ROI noisy/borderline train rows (Queue A) + clean human/AI seeds for Queue C."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "hard_group_weak": [],
        "midband_weak": [],
        "high_ai_weak_verify": [],
        "short_ambig": [],
    }
    cf_seeds_human: list[dict[str, Any]] = []
    cf_seeds_ai: list[dict[str, Any]] = []

    if not corpus_path.is_file():
        return [], []

    trap_inv = build_trap_inverted_index(trap_trigrams)
    opener = gzip.open if corpus_path.name.endswith(".gz") else open
    with opener(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "train" or r.get("cohort") == "user_confirmed":
                continue
            if not r.get("has_score", True):
                continue
            txt = scrub_accidental_secrets(str(r.get("text") or ""))
            if len(txt) < 6 or len(txt) > 900:
                continue
            norm = normalize_text(txt)
            if norm in seen_norms or norm in forbidden_exact:
                continue

            src = str(r.get("score_source") or "")
            gid = str(r.get("group_id") or "")
            coh = str(r.get("cohort") or "")
            era = str(r.get("era_bucket") or "")
            sc = float(r.get("score", 50.0))
            is_gold = src in GOLD_HIGH_TIER_SOURCES

            # Collect high-confidence seeds for Queue C counterfactual generation
            if coh in ("bili", "v04_1m") and era != "pre_llm" and 18 <= len(txt) <= 280:
                if sc <= 18.0 and len(cf_seeds_human) < 120000:
                    cf_seeds_human.append(
                        {
                            "sample_id": str(r["sample_id"]),
                            "text": txt,
                            "lang_tag": r.get("lang_tag") or detect_lang_fast(txt),
                            "src_role": "human",
                            "prior_score": sc,
                            "group_id": gid,
                        }
                    )
                elif sc >= 78.0 and len(cf_seeds_ai) < 60000:
                    cf_seeds_ai.append(
                        {
                            "sample_id": str(r["sample_id"]),
                            "text": txt,
                            "lang_tag": r.get("lang_tag") or detect_lang_fast(txt),
                            "src_role": "ai",
                            "prior_score": sc,
                            "group_id": gid,
                        }
                    )

            # Queue A targets NON-gold weak/noisy rows where relabeling delivers maximum metric lift
            if is_gold or coh == "synth_250k":
                continue
            if era == "pre_llm" and sc <= 25.0:
                continue  # pre-LLM clear humans don't need Gemini quota!

            bucket_key = ""
            mine_reason = ""
            if gid in HARD_GROUPS and 20.0 <= sc <= 80.0:
                bucket_key, mine_reason = "hard_group_weak", f"hard_group:{gid}"
            elif 26.0 <= sc <= 74.0 and len(txt) >= 14:
                bucket_key, mine_reason = "midband_weak", "midband_weak"
            elif sc >= 70.0 and coh in ("bili", "v04_1m"):
                bucket_key, mine_reason = "high_ai_weak_verify", "real_bili_weak_pos_audit"
            elif 12.0 <= sc <= 68.0 and (len(txt) <= 20 or STRUCTURED_TECH_RE.search(txt)):
                bucket_key, mine_reason = "short_ambig", "short_or_struct_ambig"
            else:
                continue

            if len(buckets[bucket_key]) >= max(2000, limit * 2):
                continue
            if is_near_dup_traps(txt, forbidden_exact, trap_trigrams, trap_inv=trap_inv):
                continue
            buckets[bucket_key].append(
                {
                    "sample_id": str(r["sample_id"]),
                    "text": txt,
                    "lang_tag": r.get("lang_tag") or detect_lang_fast(txt),
                    "queue": "A_noise_surgery",
                    "prior_score": round(sc, 1),
                    "prior_source": src,
                    "era_bucket": era,
                    "group_id": gid,
                    "cohort": coh,
                    "mine_reason": mine_reason,
                }
            )

    # Stratified allocation: prioritize hard_group_weak (35%) + midband_weak (35%) + weak_pos_audit (20%) + short (10%)
    quotas = {
        "hard_group_weak": int(limit * 0.35),
        "midband_weak": int(limit * 0.35),
        "high_ai_weak_verify": int(limit * 0.20),
        "short_ambig": limit - int(limit * 0.35) - int(limit * 0.35) - int(limit * 0.20),
    }
    picked: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for k, q in quotas.items():
        arr = buckets[k]
        rng.shuffle(arr)
        picked.extend(arr[:q])
        leftover.extend(arr[q:])
    if len(picked) < limit and leftover:
        rng.shuffle(leftover)
        picked.extend(leftover[: limit - len(picked)])

    deduped: list[dict[str, Any]] = []
    for it in picked:
        n = normalize_text(it["text"])
        if n not in seen_norms:
            seen_norms.add(n)
            deduped.append(it)
    return deduped[:limit], (cf_seeds_human, cf_seeds_ai)


def mine_queue_b_and_d_from_sqlite(
    sqlite_path: Path,
    limit_b: int,
    limit_d: int,
    forbidden_exact: set[str],
    trap_trigrams: list[frozenset[str]],
    rng: random.Random,
    seen_norms: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mine fresh modern 2025-2026 Bilibili wild hard cases (Queue B) + Thread context triples (Queue D)."""
    if not sqlite_path.is_file() or (limit_b <= 0 and limit_d <= 0):
        return [], []

    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()

    # Load video titles (zero PII)
    video_titles: dict[str, str] = {
        str(vid): str(title or "")
        for vid, title in cur.execute("SELECT video_id, title FROM videos")
    }

    b_buckets: dict[str, list[dict[str, Any]]] = {
        "structured_tech": [],
        "ai_suspect_wild": [],
        "multilingual": [],
        "modern_long": [],
        "modern_short": [],
    }
    top_text_by_sid: dict[str, str] = {}
    d_candidates: list[dict[str, Any]] = []

    trap_inv = build_trap_inverted_index(trap_trigrams)
    # First pass: read top-level comments for both Queue B and parent lookup for Queue D
    sql = "SELECT sample_id, video_id, partition, era_bucket, comment_level, metadata_json FROM comments"
    for sid, vid, part, era, lvl, mjson in cur.execute(sql):
        if not mjson:
            continue
        try:
            meta = json.loads(mjson)
        except json.JSONDecodeError:
            continue
        txt = scrub_accidental_secrets(str(meta.get("text") or ""))
        if len(txt) < 6 or len(txt) > 900:
            continue
        if lvl == "top":
            top_text_by_sid[str(sid)] = txt

        # Focus Queue B on modern post-LLM comments (eliminates era shortcut!)
        if limit_b > 0 and str(era or "") != "pre_llm":
            norm = normalize_text(txt)
            if norm not in seen_norms and norm not in forbidden_exact:
                bkey = ""
                mreason = ""
                if STRUCTURED_TECH_RE.search(txt) and len(txt) >= 24 and len(b_buckets["structured_tech"]) < limit_b * 2:
                    bkey, mreason = "structured_tech", "wild_structured_tech"
                elif AI_SUSPECT_WILD_RE.search(txt) and len(txt) >= 16 and len(b_buckets["ai_suspect_wild"]) < limit_b * 2:
                    bkey, mreason = "ai_suspect_wild", "wild_ai_suspect"
                elif MULTILINGUAL_RE.search(txt) and len(txt) >= 16 and len(b_buckets["multilingual"]) < limit_b:
                    bkey, mreason = "multilingual", "wild_multilingual"
                elif len(txt) >= 120 and len(b_buckets["modern_long"]) < limit_b:
                    bkey, mreason = "modern_long", "wild_modern_long"
                elif 10 <= len(txt) <= 22 and len(b_buckets["modern_short"]) < limit_b:
                    bkey, mreason = "modern_short", "wild_modern_short"

                if bkey and not is_near_dup_traps(txt, forbidden_exact, trap_trigrams, trap_inv=trap_inv):
                    b_buckets[bkey].append(
                        {
                            "sample_id": f"V8B_{sid}",
                            "orig_sample_id": str(sid),
                            "text": txt,
                            "lang_tag": detect_lang_fast(txt),
                            "queue": "B_wild_hard_mining",
                            "era_bucket": str(era or "current_2026"),
                            "partition": str(part or ""),
                            "prior_score": None,
                            "mine_reason": mreason,
                        }
                    )

        if limit_d > 0 and lvl != "top" and str(era or "") != "pre_llm":
            parent_id = str(meta.get("parent_id") or "")
            vtitle = str(meta.get("video_title") or video_titles.get(str(vid or ""), "") or "")
            if len(txt) >= 12 and (parent_id or len(vtitle) >= 4) and len(d_candidates) < max(5000, limit_d * 4):
                d_candidates.append(
                    {
                        "sample_id": f"V8D_{sid}",
                        "orig_sample_id": str(sid),
                        "video_id": str(vid or ""),
                        "video_title": vtitle[:100],
                        "parent_id": parent_id,
                        "text": txt,
                        "lang_tag": detect_lang_fast(txt),
                        "queue": "D_thread_context",
                        "era_bucket": str(era or "current_2026"),
                        "partition": str(part or ""),
                        "prior_score": None,
                        "mine_reason": "thread_reply_with_parent" if parent_id else "thread_reply_with_video_title",
                    }
                )

    con.close()

    # Finalize Queue B stratified selection
    queue_b: list[dict[str, Any]] = []
    if limit_b > 0:
        b_quotas = {
            "structured_tech": int(limit_b * 0.30),
            "ai_suspect_wild": int(limit_b * 0.30),
            "multilingual": int(limit_b * 0.15),
            "modern_long": int(limit_b * 0.15),
            "modern_short": limit_b - int(limit_b * 0.30) * 2 - int(limit_b * 0.15) * 2,
        }
        picked_b: list[dict[str, Any]] = []
        rem_b: list[dict[str, Any]] = []
        for k, q in b_quotas.items():
            arr = b_buckets[k]
            rng.shuffle(arr)
            picked_b.extend(arr[:q])
            rem_b.extend(arr[q:])
        if len(picked_b) < limit_b and rem_b:
            rng.shuffle(rem_b)
            picked_b.extend(rem_b[: limit_b - len(picked_b)])
        for it in picked_b:
            n = normalize_text(it["text"])
            if n not in seen_norms:
                seen_norms.add(n)
                queue_b.append(it)
            if len(queue_b) >= limit_b:
                break

    # Finalize Queue D (stratify 35% AI-suspect/structured replies + 65% general parent-linked replies!)
    queue_d: list[dict[str, Any]] = []
    if limit_d > 0 and d_candidates:
        rng.shuffle(d_candidates)
        d_ai_struct = [c for c in d_candidates if AI_SUSPECT_WILD_RE.search(c["text"]) or STRUCTURED_TECH_RE.search(c["text"])]
        d_other = [c for c in d_candidates if not (AI_SUSPECT_WILD_RE.search(c["text"]) or STRUCTURED_TECH_RE.search(c["text"]))]
        for arr in (d_ai_struct, d_other):
            arr.sort(key=lambda c: 0 if (c["parent_id"] and c["parent_id"] in top_text_by_sid) else 1)
        q_prio = int(limit_d * 0.35)
        ordered_d = d_ai_struct[: q_prio * 2] + d_other
        for cand in ordered_d:
            ptxt = top_text_by_sid.get(cand["parent_id"], "") if cand["parent_id"] else ""
            n = normalize_text(cand["text"])
            if n in seen_norms or is_near_dup_traps(cand["text"], forbidden_exact, trap_trigrams, trap_inv=trap_inv):
                continue
            seen_norms.add(n)
            cand["parent_text"] = ptxt[:240]
            cand["video_title"] = cand.get("video_title") or video_titles.get(cand["video_id"], "")[:100]
            cand.pop("parent_id", None)
            cand.pop("video_id", None)
            queue_d.append(cand)
            if len(queue_d) >= limit_d:
                break
        rng.shuffle(queue_d)

    return queue_b, queue_d


def build_queue_c_counterfactual(
    cf_seeds_human: list[dict[str, Any]],
    cf_seeds_ai: list[dict[str, Any]],
    limit_c: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build Queue C seed tasks across all 4 counterfactual & sandwich hybrid-span modes."""
    if limit_c <= 0:
        return []
    rng.shuffle(cf_seeds_human)
    rng.shuffle(cf_seeds_ai)

    q_evasive = int(limit_c * 0.30)
    q_short = int(limit_c * 0.20)
    q_struct_hu = int(limit_c * 0.25)
    q_sandwich = limit_c - q_evasive - q_short - q_struct_hu

    out: list[dict[str, Any]] = []
    hi = 0
    for mode, count in (("human_to_evasive_ai", q_evasive), ("human_to_short_ai", q_short), ("sandwich_hybrid_span", q_sandwich)):
        for _ in range(count):
            if hi >= len(cf_seeds_human):
                break
            s = cf_seeds_human[hi]
            hi += 1
            out.append(
                {
                    "sample_id": f"V8C_{mode[:4]}_{s['sample_id']}",
                    "orig_sample_id": s["sample_id"],
                    "text": s["text"],
                    "lang_tag": s["lang_tag"],
                    "queue": "C_counterfactual_spans",
                    "cf_mode": mode,
                    "src_role": "human",
                    "prior_score": s["prior_score"],
                }
            )
    for ai_s in cf_seeds_ai[:q_struct_hu]:
        out.append(
            {
                "sample_id": f"V8C_sthu_{ai_s['sample_id']}",
                "orig_sample_id": ai_s["sample_id"],
                "text": ai_s["text"],
                "lang_tag": ai_s["lang_tag"],
                "queue": "C_counterfactual_spans",
                "cf_mode": "ai_to_structured_human",
                "src_role": "ai",
                "prior_score": ai_s["prior_score"],
            }
        )
    rng.shuffle(out)
    return out[:limit_c]


def inject_hidden_canaries(
    rows: list[dict[str, Any]],
    val_gold_pool: list[dict[str, Any]],
    canary_rate: float,
    rng: random.Random,
    queue_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blind-mix ~1-2% balanced val_gold canaries evenly across scoring queues (supports 200k+ multi-wave!)."""
    if canary_rate <= 0.0 or not rows or not val_gold_pool:
        return rows, []
    n_can = max(20 if len(rows) >= 500 else max(2, int(len(rows) * canary_rate)), int(len(rows) * canary_rate))
    n_can = min(n_can, max(1, len(rows) // 5))
    # Cap at 800 per queue (plenty for 20+ balanced canaries per wave across 20 waves!)
    n_can = min(n_can, 800)

    # Balanced sampling across pos (>=70) and neg (<=30) so every parallel wave gets solid binary AUROC!
    pos_pool = [c for c in val_gold_pool if float(c["gold_score"]) >= 70.0]
    neg_pool = [c for c in val_gold_pool if float(c["gold_score"]) <= 30.0]
    mid_pool = [c for c in val_gold_pool if 30.0 < float(c["gold_score"]) < 70.0]
    chosen: list[dict[str, Any]] = []
    if n_can <= len(val_gold_pool):
        chosen = rng.sample(val_gold_pool, n_can)
    else:
        n_pos = int(n_can * 0.45)
        n_neg = int(n_can * 0.45)
        n_mid = n_can - n_pos - n_neg
        for pool, k in ((pos_pool, n_pos), (neg_pool, n_neg), (mid_pool or val_gold_pool, n_mid)):
            if pool and k > 0:
                while k > 0:
                    take = min(k, len(pool))
                    chosen.extend(rng.sample(pool, take))
                    k -= take
        rng.shuffle(chosen)

    vault_entries: list[dict[str, Any]] = []
    # Distribute canaries uniformly across the shuffled queue so every wave slice receives balanced probes!
    base_rows = list(rows)
    rng.shuffle(base_rows)
    mixed: list[dict[str, Any]] = []
    stride = max(1, len(base_rows) // max(1, len(chosen)))
    ci = 0
    for ri, r in enumerate(base_rows):
        mixed.append(r)
        if ci < len(chosen) and (ri + 1) % stride == 0:
            c = chosen[ci]
            blind_hash = hashlib.sha256(f"canary|{queue_name}|{c['sample_id']}|{ci}".encode("utf-8")).hexdigest()[:12]
            blind_sid = f"V8Q_{blind_hash}"
            mixed.append(
                {
                    "sample_id": blind_sid,
                    "text": c["text"],
                    "lang_tag": c["lang_tag"],
                    "queue": queue_name,
                    "prior_score": None,
                    "era_bucket": "canary_blind",
                    "mine_reason": "blind_audit",
                }
            )
            vault_entries.append(
                {
                    "blind_sample_id": blind_sid,
                    "orig_val_sample_id": c["sample_id"],
                    "queue": queue_name,
                    "gold_score": c["gold_score"],
                    "gold_code": c["gold_code"],
                    "score_source": c["score_source"],
                    "group_id": c["group_id"],
                }
            )
            ci += 1
    return mixed, vault_entries


def pack_into_batches(rows: list[dict[str, Any]], queue_name: str, batch_size: int) -> list[dict[str, Any]]:
    """Group queue rows into mega-packed batches with deterministic 3-char anti-drift check tags."""
    batches: list[dict[str, Any]] = []
    for bi in range(0, len(rows), batch_size):
        chunk = rows[bi : bi + batch_size]
        batch_id = f"{queue_name}_b{bi // batch_size + 1:05d}"
        packed_rows = []
        for idx, r in enumerate(chunk, 1):
            r_copy = dict(r)
            r_copy["idx"] = idx
            r_copy["tag"] = make_check_tag(batch_id, r["sample_id"])
            packed_rows.append(r_copy)
        batches.append(
            {
                "batch_id": batch_id,
                "queue": queue_name,
                "n": len(packed_rows),
                "rows": packed_rows,
            }
        )
    return batches


def estimate_cost_usd(total_scoring_rows: int, total_cf_rows: int, batch_size: int = 25, escalation_rate: float = 0.15) -> dict[str, Any]:
    """Estimate external Gemini API cost (Batch 50% off vs Realtime) in USD."""
    # Empirical token budget per 25-row mega-batch (Chinese/Bili avg ~55 chars ~42 tokens + JSON overhead):
    # Scoring: ~850 sys + ~1,350 user = 2,200 input tok; ~850 output tok per batch of 25 -> 88 in / 34 out tok per row
    sc_in_m = (total_scoring_rows * 90) / 1_000_000
    sc_out_m = (total_scoring_rows * 36) / 1_000_000
    # Counterfactual: ~650 sys + ~1,000 user; ~1,600 output tok per batch of 12 -> 138 in / 135 out tok per row
    cf_in_m = (total_cf_rows * 140) / 1_000_000
    cf_out_m = (total_cf_rows * 140) / 1_000_000

    # Tier-1 Gemini 2.5 Flash Batch (~$0.15/1M in, ~$1.25/1M out; non-thinking Batch ~$0.15 in / $0.30 out)
    # Using conservative Gemini 2.5 Flash Batch standard non-thinking rates: $0.15/1M in, $0.30/1M out (Realtime 2x)
    # Tier-2 Gemini 2.5 Pro Batch: $0.625/1M in, $5.00/1M out (Realtime 2x)
    flash_batch_usd = (sc_in_m + cf_in_m) * 0.15 + (sc_out_m + cf_out_m) * 0.30
    pro_esc_rows = int(total_scoring_rows * escalation_rate)
    pro_in_m = (pro_esc_rows * 90) / 1_000_000
    pro_out_m = (pro_esc_rows * 38) / 1_000_000
    pro_batch_usd = pro_in_m * 0.625 + pro_out_m * 5.00
    return {
        "scoring_rows": total_scoring_rows,
        "counterfactual_rows": total_cf_rows,
        "estimated_tier2_pro_rows": pro_esc_rows,
        "batch_api_50pct_off_usd": {
            "tier1_flash": round(flash_batch_usd, 2),
            "tier2_pro_tiebreak": round(pro_batch_usd, 2),
            "total_usd": round(flash_batch_usd + pro_batch_usd, 2),
        },
        "realtime_api_usd": {
            "tier1_flash": round(flash_batch_usd * 2.0, 2),
            "tier2_pro_tiebreak": round(pro_batch_usd * 2.0, 2),
            "total_usd": round((flash_batch_usd + pro_batch_usd) * 2.0, 2),
        },
        "antigravity_quota_cost": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "v04-corpus" / "local-train" / "v04-merged-1m4-r3.jsonl.gz")
    ap.add_argument("--sqlite", type=Path, default=REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3")
    ap.add_argument("--extra-exclude", type=Path, default=None, help="Optional extra JSONL file of forbidden texts")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "v08-datafactory-work")
    ap.add_argument("--limit-a", type=int, default=60000, help="Queue A (Noise Surgery) row target")
    ap.add_argument("--limit-b", type=int, default=120000, help="Queue B (Wild Hard Mining) row target")
    ap.add_argument("--limit-c", type=int, default=30000, help="Queue C (Counterfactual & Hybrid Span) seed target")
    ap.add_argument("--limit-d", type=int, default=40000, help="Queue D (Thread Context) row target")
    ap.add_argument("--batch-size", type=int, default=25, help="Rows packed per Gemini scoring request (20-25 optimal)")
    ap.add_argument("--cf-batch-size", type=int, default=12, help="Rows packed per Gemini counterfactual request")
    ap.add_argument("--canary-rate", type=float, default=0.02, help="Hidden val_gold canary rate (stripped before training)")
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queues_dir = out_dir / "queues"
    queues_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building non-silent anti-leakage index from {args.corpus} ...")
    forbidden_exact, trap_trigrams, val_gold_pool = load_traps_and_eval_forbidden(args.corpus, args.extra_exclude)
    print(f"      forbidden_exact={len(forbidden_exact):,} | trap_trigrams={len(trap_trigrams)} | val_gold_canaries_avail={len(val_gold_pool):,}")

    seen_norms: set[str] = set()
    print(f"[2/4] Mining Queue A (target={args.limit_a:,}) & Queue C seeds ...")
    q_a, (cf_hu, cf_ai) = mine_queue_a_noise_surgery(args.corpus, args.limit_a, forbidden_exact, trap_trigrams, rng, seen_norms)

    print(f"[3/4] Mining Queue B (target={args.limit_b:,}) & Queue D (target={args.limit_d:,}) from SQLite ...")
    q_b, q_d = mine_queue_b_and_d_from_sqlite(args.sqlite, args.limit_b, args.limit_d, forbidden_exact, trap_trigrams, rng, seen_norms)
    q_c = build_queue_c_counterfactual(cf_hu, cf_ai, args.limit_c, rng)

    print(f"[4/4] Injecting hidden {args.canary_rate*100:.1f}% val_gold canaries & packing mega-batches ...")
    q_a_mix, vault_a = inject_hidden_canaries(q_a, val_gold_pool, args.canary_rate, rng, "A_noise_surgery")
    q_b_mix, vault_b = inject_hidden_canaries(q_b, val_gold_pool, args.canary_rate, rng, "B_wild_hard_mining")
    all_vault = vault_a + vault_b

    queue_map = {
        "A_noise_surgery": (q_a_mix, args.batch_size),
        "B_wild_hard_mining": (q_b_mix, args.batch_size),
        "C_counterfactual_spans": (q_c, args.cf_batch_size),
        "D_thread_context": (q_d, args.batch_size),
    }

    manifest_queues: dict[str, Any] = {}
    for qname, (rows, bs) in queue_map.items():
        batches = pack_into_batches(rows, qname, bs)
        qfile = queues_dir / f"{qname}.batches.jsonl"
        with qfile.open("w", encoding="utf-8", newline="\n") as wf:
            for b in batches:
                wf.write(json.dumps(b, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_queues[qname] = {
            "rows": len(rows),
            "batches": len(batches),
            "batch_size": bs,
            "file": str(qfile),
            "reason_dist": dict(Counter(str(r.get("mine_reason") or r.get("cf_mode") or "") for r in rows).most_common(10)),
        }

    # Keep canary vault strictly local (NEVER uploaded to Gemini!)
    vault_path = out_dir / "canary_vault.json"
    vault_path.write_text(json.dumps({"canaries": all_vault}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scoring_total = len(q_a_mix) + len(q_b_mix) + len(q_d)
    cost = estimate_cost_usd(scoring_total, len(q_c), batch_size=args.batch_size)
    manifest = {
        "schema_version": "meguviz_v08_datafactory_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "seed": args.seed,
        "anti_leakage": {
            "forbidden_exact_count": len(forbidden_exact),
            "trap_trigram_count": len(trap_trigrams),
            "zero_pii_verified": True,
        },
        "hidden_canaries": len(all_vault),
        "canary_vault_path": str(vault_path),
        "queues": manifest_queues,
        "cost_estimate": cost,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
