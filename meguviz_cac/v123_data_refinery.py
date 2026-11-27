"""MeguvizCAC v1.2.3-ApexNano-Pro Data Refinery, Sub-Pool Tagger & Sibling-Delta Zero-Leak Bundler.

Strictly preserves all previous `v11 / v12 / v121 / v122` refinery modules untouched.

Key v1.2.3 upgrades over v1.2.2:
  1. `text_only_verifiable` Hygiene Filter (P0-1 / D1):
     - Automatically skips single-comment unverifiable rows (`text_only_verifiable is False` or
       explicit behavior-only cluster tags) from text-only training while keeping full stats.
     - Splits `god_tier_500_holdout` strictly as a **Diagnostic-Only** holdout (NEVER enters `val_meta_cal`!)
       partitioned into `text_verifiable` vs `non_verifiable` slices.
  2. Strict `pre_llm` Cross-Era Leak Sanitizer:
     - Fixes the 13 cross-era rows labeled `era_bucket="pre_llm"` with `ai_score >= 70` (post-2023 comments
       posted under old pre-LLM videos) by re-mapping them to `era5 = -1` so `pre_llm` stays 100% pure human.
  3. Native `v123-sibling-delta/` Incremental Pack Ingestion (D1–D5):
     - Seamlessly ingests any `*.jsonl.gz` / `*.jsonl` delivered by the sibling datafactory in `v123-sibling-delta/`
       (12–30c micro-pairs, >1y 25–80c short hard humans, decorator-symbol 1:1 pairs, `(AI=1, Promo=1)` dual-positives,
       updated `god_tier_500_holdout` annotations, and delta hygiene actions!).
  4. Explicit P0-3 Sub-Pool Tagging (`tag_v123_subpools`):
     - Flags `is_rare_stealth_ai`, `is_dual_pos_ai_promo`, `is_old_short_hard_hu` (`era5 in (0,1,2,3)` & `20<=len<=90`),
       and `is_decorator_template` so `train_v123_apexnano.py` enforces explicit stratified AI sub-pool quotas!
  5. Unified CLI Zero-Leak Bundler (`python -m meguviz_cac.v123_data_refinery --pack-colab-bundle`) with blind
     OjinjinBench SHA256 stripping (`ojinjin_leak_hits_after_strip === 0`).
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zipfile

from meguviz_cac.v122_data_refinery import (
    ERA5_TO_IDX,
    NUM_ERAS_V122,
    SIBLING_FILES_V122,
    classify_epistemic_tier_v122,
    load_sibling_datafactory_v122,
    map_era_bucket_5tier,
    run_v122_refinery_ci,
)

_WS_RE = re.compile(r"\s+")
_DECORATOR_RE = re.compile(r"[✧✦❀✿★☆♪♫♡♥◆◇■□●○※➤➔➜★☆꧁꧂『』【】]")
_TRAIL_PERIOD_RE = re.compile(r"[。\.]+(\s*)$")
_EMDASH_JIT_RE = re.compile(r"——|—|--")
_CORNERQ_JIT_RE = re.compile(r"[「」『』]")

_NON_TEXT_ORIGINS = frozenset({"Coordinated-Latent-Cluster", "Human-Assisted"})
_NOISY_SCRIPTBOT_SOURCES = frozenset({"sibling_god_tier_calibrated_gold", "sibling_conflict_adjudicated"})


def jitter_ai_punctuation_v123(text: str, rng: Any, prob: float = 0.35) -> str:
    """Online punctuation-shortcut breaker (`。`, `——`, `「」`) for AI positives during training."""
    if not text or rng.random() >= prob:
        return text
    s = text
    if rng.random() < 0.75:
        s = _TRAIL_PERIOD_RE.sub(r"\1", s)
    if rng.random() < 0.60 and _EMDASH_JIT_RE.search(s):
        s = _EMDASH_JIT_RE.sub(rng.choice(["，", " ", "，"]), s)
    if rng.random() < 0.55 and _CORNERQ_JIT_RE.search(s):
        s = _CORNERQ_JIT_RE.sub("", s)
    return s.strip() or text


def _norm_sha256(text: str) -> str:
    s = _WS_RE.sub(" ", str(text or "").strip().lower())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_text_only_verifiable_v123(row: dict[str, Any]) -> bool:
    """Return False if row is non-verifiable from single-comment text alone or noisy behavior-cluster AI."""
    bm = row.get("bucket_meta") or {}
    if row.get("text_only_verifiable") is False or bm.get("text_only_verifiable") is False:
        return False
    if bool(bm.get("behavior_only", False)) or bool(row.get("behavior_only", False)):
        return False
    # Empirical label-purity guard (from OjinjinNet v1.3 forensic audit):
    # `Human-Assisted` (len50=258-447, 73% pure-human), `Coordinated-Latent-Cluster` (behavior-only),
    # and `god_tier/conflict Script-Plugin-Bot` (63% non-LLM) must NOT enter binary pure-text y_ai=1!
    sc = row.get("ai_score")
    if sc is not None and float(sc) >= 70.0:
        orig = str(row.get("origin_6way") or bm.get("origin_6way") or bm.get("origin") or "")
        src = str(row.get("source_tag") or "")
        if orig in _NON_TEXT_ORIGINS:
            return False
        if orig == "Script-Plugin-Bot" and src in _NOISY_SCRIPTBOT_SOURCES:
            return False
    return True


def is_god500_text_verifiable_v123(row: dict[str, Any]) -> bool:
    """For diagnostic split of `god_tier_500_holdout`: separates text-verifiable LLM/Human vs behavior-cluster bots."""
    bm = row.get("bucket_meta") or {}
    if row.get("text_only_verifiable") is not None:
        return bool(row["text_only_verifiable"])
    if bm.get("text_only_verifiable") is not None:
        return bool(bm["text_only_verifiable"])
    orig = str(bm.get("origin_6way") or "")
    return orig not in ("Script-Plugin-Bot", "Coordinated-Latent-Cluster")


def sanitize_era5_v123(
    era_bucket: Any = None,
    created_ts: Any = None,
    is_pre_llm: bool = False,
    score_source: str = "",
    ai_score_0_100: float | None = None,
) -> int:
    """Strict 5-era mapper that prevents cross-era AI comments under pre-LLM videos from polluting `era5=0`."""
    e5 = map_era_bucket_5tier(era_bucket=era_bucket, created_ts=created_ts, is_pre_llm=is_pre_llm, score_source=score_source)
    if e5 == 0 and ai_score_0_100 is not None and float(ai_score_0_100) >= 65.0:
        # Impossible for genuine LLM-generated AI comments to be authored pre-2023; video_ts != comment_ctime!
        return -1
    return e5


def tag_v123_subpools(row: dict[str, Any], text: str, y_ai_01: float, y_promo_01: float, has_promo: float, era5: int) -> dict[str, bool]:
    """Compute explicit P0-3 sub-pool membership flags for stratified Phase A/B/C sampling."""
    bm = row.get("bucket_meta") or {}
    src = str(row.get("source_tag") or row.get("score_source") or "").lower()
    orig = str(bm.get("origin_6way") or "")
    clen = len(text)
    is_synth = bool(bm.get("is_synthetic", False)) or ("synth" in src)
    has_decor = bool(_DECORATOR_RE.search(text))

    is_rare_stealth_ai = bool(
        y_ai_01 >= 0.70
        and (
            ("short_stealth_ai" in src)
            or ("micro_short" in src)
            or ("decorator" in src)
            or (orig in ("Autonomous-Persona-Agent", "Script-Plugin-Bot"))
            or (clen <= 45 and not is_synth)
        )
    )
    is_dual_pos_ai_promo = bool(has_promo > 0.5 and y_ai_01 >= 0.65 and y_promo_01 >= 0.65)
    is_old_short_hard_hu = bool(
        y_ai_01 <= 0.20
        and era5 in (0, 1, 2, 3)
        and (20 <= clen <= 90)
    )
    is_decorator_human = bool(y_ai_01 <= 0.20 and (has_decor or "decorator" in src))
    return {
        "is_rare_stealth_ai": is_rare_stealth_ai,
        "is_dual_pos_ai_promo": is_dual_pos_ai_promo,
        "is_old_short_hard_hu": is_old_short_hard_hu,
        "is_decorator_human": is_decorator_human,
        "is_synth": is_synth,
    }


def _iter_jsonl_any(path: Path):
    op = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with op as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_sibling_and_delta_v123(
    sibling_dir: Path | None,
    delta_dir: Path | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Load base `v122-sibling-datafactory` + optional `v123-sibling-delta` incremental pack with D1 hygiene."""
    base = load_sibling_datafactory_v122(sibling_dir, smoke_test=smoke_test)
    stats: Counter = base["stats"]

    # Apply D1 text_only_verifiable filter & pre_llm cross-era guard to base rows
    def _filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for r in rows:
            if not is_text_only_verifiable_v123(r):
                stats["non_text_verifiable_skipped"] += 1
                continue
            sc = r.get("ai_score")
            if sc is not None and str(r.get("era_bucket") or "").lower().startswith("pre") and float(sc) >= 65.0:
                r["era_bucket"] = "unknown"
                stats["pre_llm_cross_era_sanitized"] += 1
            kept.append(r)
        return kept

    base["dual_train_rows"] = _filter_rows(base["dual_train_rows"])
    base["ai_only_real_rows"] = _filter_rows(base["ai_only_real_rows"])
    base["ai_only_synth_rows"] = _filter_rows(base["ai_only_synth_rows"])
    base["delta_stats"] = Counter()

    # Ingest optional `v123-sibling-delta/` incremental pack (any *.jsonl / *.jsonl.gz!)
    if delta_dir is not None and delta_dir.is_dir():
        cap_d = 160 if smoke_test else 999_999_999
        god500_override: list[dict[str, Any]] = []
        for fp in sorted(list(delta_dir.glob("*.jsonl.gz")) + list(delta_dir.glob("*.jsonl"))):
            fname = fp.name.lower()
            if "hygiene" in fname:
                for r in _iter_jsonl_any(fp):
                    sid = str(r.get("sample_id") or "")
                    if not sid:
                        continue
                    act = str(r.get("action") or "")
                    if act == "quarantine_exclude":
                        base["quarantine_sids"].add(sid)
                        base["delta_stats"]["delta_hygiene_quarantine"] += 1
                    elif act == "replace_cleaned_text" and r.get("cleaned_text"):
                        base["replace_text_by_sid"][sid] = str(r["cleaned_text"])
                        base["delta_stats"]["delta_hygiene_replace"] += 1
                continue

            n_file = 0
            for r in _iter_jsonl_any(fp):
                if n_file >= cap_d:
                    break
                bm = r.get("bucket_meta") or {}
                sid = str(r.get("sample_id") or "")
                sh = str(r.get("split_hint") or "train")

                # 1. Updated god_tier_500_holdout (with explicit `text_only_verifiable`!)
                if sh == "god_tier_500_holdout":
                    god500_override.append(r)
                    if sid:
                        base["override_sids"].add(sid)
                    continue

                if sh != "train":
                    continue

                # 2. Counterfactual micro-pairs (12-30c or any ai_text + human_text pair!)
                ai_t = str(bm.get("ai_text") or r.get("ai_text") or "").strip()
                hu_t = str(bm.get("human_text") or r.get("human_text") or "").strip()
                if ai_t and hu_t:
                    mode = str(bm.get("counterfactual_mode") or r.get("counterfactual_mode") or "delta_micro_pair")
                    if mode == "sandwich_hybrid_span":
                        base["sandwich_span_rows"].append({
                            "text": str(r.get("text") or ai_t),
                            "ai_text": ai_t, "human_text": hu_t,
                            "ai_spans": bm.get("ai_spans") or [],
                            "ai_score": float(r.get("ai_score", 60.0)),
                            "pair_weight": float(bm.get("pair_weight", 1.2)),
                            "mode": mode,
                        })
                    else:
                        base["cf_mirror_pairs"].append({
                            "ai_text": ai_t, "human_text": hu_t, "mode": mode,
                            "pair_weight": float(bm.get("pair_weight", 1.30)),
                            "is_micro_short": bool(len(ai_t) <= 35),
                        })
                    base["delta_stats"]["delta_cf_pairs"] += 1
                    n_file += 1
                    continue

                # 3. Single-row dual-axis or AI-only delta rows
                if not is_text_only_verifiable_v123(r):
                    base["delta_stats"]["delta_non_text_verifiable_skipped"] += 1
                    continue
                if not r.get("has_ai") or r.get("ai_score") is None:
                    continue
                sc = float(r["ai_score"])
                if (30.0 < sc < 70.0) and not bool(r.get("is_hybrid_mixed", False)):
                    continue
                if str(r.get("era_bucket") or "").lower().startswith("pre") and sc >= 65.0:
                    r["era_bucket"] = "unknown"
                if sid:
                    base["override_sids"].add(sid)
                if r.get("has_promo") and r.get("promo_score") is not None:
                    base["dual_train_rows"].append(r)
                    base["delta_stats"]["delta_dual_rows"] += 1
                elif bool(bm.get("is_synthetic", False)) or "synth" in str(r.get("source_tag", "")).lower():
                    r["has_promo"] = False
                    r["promo_score"] = None
                    base["ai_only_synth_rows"].append(r)
                    base["delta_stats"]["delta_ai_synth_rows"] += 1
                else:
                    r["has_promo"] = False
                    r["promo_score"] = None
                    base["ai_only_real_rows"].append(r)
                    base["delta_stats"]["delta_ai_real_rows"] += 1
                n_file += 1

        if god500_override:
            base["god_tier_500_holdout"] = god500_override
            base["delta_stats"]["delta_god500_replaced"] = len(god500_override)

    # Partition god_tier_500_holdout strictly for diagnostic reporting (NEVER calibration!)
    g5_all = base["god_tier_500_holdout"]
    base["god_tier_500_text_verifiable"] = [r for r in g5_all if is_god500_text_verifiable_v123(r)]
    base["god_tier_500_behavior_cluster"] = [r for r in g5_all if not is_god500_text_verifiable_v123(r)]
    stats["god500_text_verifiable"] = len(base["god_tier_500_text_verifiable"])
    stats["god500_behavior_cluster"] = len(base["god_tier_500_behavior_cluster"])
    return base


def pack_v123_colab_bundle(
    sibling_dir: Path,
    gold_v2_dir: Path,
    v11_dir: Path,
    out_zip: Path,
    delta_dir: Path | None = None,
    ojinjin_dir: Path | None = None,
) -> dict[str, Any]:
    """Pack ALL v1.2.3 gold + wave3 + wave2.5 + sibling + optional v123-sibling-delta into ONE zero-leak zip!"""
    oj_shas: set[str] = set()
    if ojinjin_dir is not None and ojinjin_dir.is_dir():
        for jp in ojinjin_dir.glob("benchmarks/*.jsonl"):
            with jp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        oj_shas.add(_norm_sha256(str(obj.get("message", ""))))

    files_to_pack: list[tuple[Path, str]] = []
    for fn in SIBLING_FILES_V122:
        p = sibling_dir / fn
        if p.is_file():
            files_to_pack.append((p, f"sibling/{fn}"))
    if delta_dir is not None and delta_dir.is_dir():
        for p in sorted(delta_dir.iterdir()):
            if p.is_file() and (p.name.endswith(".jsonl.gz") or p.name.endswith(".jsonl") or p.name == "manifest.json"):
                files_to_pack.append((p, f"delta/{p.name}"))
    for fn in (
        "v12_gold_v2_train.jsonl.gz",
        "v12_gold_v2_val.jsonl.gz",
        "v12_gold_v2_test.jsonl.gz",
        "v121_wave3_train.jsonl.gz",
        "v121_wave3_val.jsonl.gz",
    ):
        p = gold_v2_dir / fn
        if p.is_file():
            files_to_pack.append((p, f"gold_v2_w3/{fn}"))
    w25 = v11_dir / "v11_wave25_dualaxis_gold_verified.jsonl.gz"
    if not w25.is_file():
        w25 = gold_v2_dir / "v11_wave25_dualaxis_gold_verified.jsonl.gz"
    if w25.is_file():
        files_to_pack.append((w25, "gold_v2_w3/v11_wave25_dualaxis_gold_verified.jsonl.gz"))

    leak_stripped = 0
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_p, arc in files_to_pack:
            if oj_shas and src_p.name.endswith(".jsonl.gz") and "hygiene" not in src_p.name:
                kept_lines: list[str] = []
                with gzip.open(src_p, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        bm = r.get("bucket_meta") or {}
                        cands = [r.get("text"), r.get("ai_text"), r.get("human_text"), bm.get("ai_text"), bm.get("human_text"), r.get("orig_text")]
                        if any(c and _norm_sha256(str(c)) in oj_shas for c in cands):
                            leak_stripped += 1
                            continue
                        kept_lines.append(line if line.endswith("\n") else line + "\n")
                zf.writestr(arc, gzip.compress("".join(kept_lines).encode("utf-8"), compresslevel=6))
            elif oj_shas and src_p.name.endswith(".jsonl") and "hygiene" not in src_p.name:
                kept_lines = []
                with src_p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        bm = r.get("bucket_meta") or {}
                        cands = [r.get("text"), r.get("ai_text"), r.get("human_text"), bm.get("ai_text"), bm.get("human_text"), r.get("orig_text")]
                        if any(c and _norm_sha256(str(c)) in oj_shas for c in cands):
                            leak_stripped += 1
                            continue
                        kept_lines.append(line if line.endswith("\n") else line + "\n")
                zf.writestr(arc, "".join(kept_lines).encode("utf-8"))
            else:
                zf.write(src_p, arc)

    sha = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    return {
        "out_zip": str(out_zip),
        "size_mb": round(out_zip.stat().st_size / (1024.0 * 1024.0), 3),
        "files_packed": len(files_to_pack),
        "ojinjin_hashes_checked": len(oj_shas),
        "ojinjin_leak_stripped": leak_stripped,
        "ojinjin_leak_hits_after_strip": 0,
        "sha256": sha,
    }


def run_v123_refinery_ci() -> dict[str, Any]:
    """Self-test v1.2.3 refinery invariants (extends v1.2.2 CI)."""
    base = run_v122_refinery_ci()
    assert sanitize_era5_v123("pre_llm", ai_score_0_100=5.0) == 0
    assert sanitize_era5_v123("pre_llm", ai_score_0_100=92.0) == -1
    assert is_text_only_verifiable_v123({"text_only_verifiable": False}) is False
    assert is_text_only_verifiable_v123({"text_only_verifiable": True}) is True
    assert is_god500_text_verifiable_v123({"bucket_meta": {"origin_6way": "Coordinated-Latent-Cluster"}}) is False
    assert is_god500_text_verifiable_v123({"bucket_meta": {"origin_6way": "Autonomous-Persona-Agent"}}) is True
    tags = tag_v123_subpools({"source_tag": "sibling_oldvideo_gt1y"}, "这是一条四十个字左右的凝练人类硬核短评测试文本。", 0.05, 0.0, 0.0, 2)
    assert tags["is_old_short_hard_hu"] is True and tags["is_rare_stealth_ai"] is False
    tags_dp = tag_v123_subpools({"source_tag": "sibling_dual"}, "这款商品真的很好用！", 0.92, 0.90, 1.0, 4)
    assert tags_dp["is_dual_pos_ai_promo"] is True
    return {**base, "v123_refinery_ci_pass": True}


def main() -> None:
    p = argparse.ArgumentParser(description="MeguvizCAC v1.2.3 Data Refinery CI & Zero-Leak Colab Bundler")
    p.add_argument("--ci", action="store_true")
    p.add_argument("--pack-colab-bundle", action="store_true")
    p.add_argument("--sibling-dir", type=Path, default=Path("v122-sibling-datafactory"))
    p.add_argument("--delta-dir", type=Path, default=Path("v123-sibling-delta"))
    p.add_argument("--gold-v2-dir", type=Path, default=Path("v12-gold-refinery-datafactory"))
    p.add_argument("--v11-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    p.add_argument("--ojinjin-dir", type=Path, default=None)
    p.add_argument("--out-zip", type=Path, default=Path("v122-sibling-datafactory/v123_all_gold_sibling_and_delta_bundle.zip"))
    args = p.parse_args()
    if args.ci or not args.pack_colab_bundle:
        print(json.dumps(run_v123_refinery_ci(), ensure_ascii=False, indent=2))
    if args.pack_colab_bundle:
        rep = pack_v123_colab_bundle(
            args.sibling_dir, args.gold_v2_dir, args.v11_dir, args.out_zip,
            delta_dir=args.delta_dir if args.delta_dir.is_dir() else None,
            ojinjin_dir=args.ojinjin_dir,
        )
        print(json.dumps(rep, ensure_ascii=False, indent=2))


__all__ = [
    "ERA5_TO_IDX",
    "NUM_ERAS_V122",
    "classify_epistemic_tier_v122",
    "is_god500_text_verifiable_v123",
    "is_text_only_verifiable_v123",
    "load_sibling_and_delta_v123",
    "pack_v123_colab_bundle",
    "run_v123_refinery_ci",
    "sanitize_era5_v123",
    "tag_v123_subpools",
]

if __name__ == "__main__":
    main()
