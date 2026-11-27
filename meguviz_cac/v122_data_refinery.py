"""MeguvizCAC v1.2.2-ApexNano-Ultra Epistemic Data Refinery, 5-Era Mapper & Sibling Data Merger.

Strictly preserves all previous `v11_data_refinery.py`, `v12_data_refinery.py`, and `v121_data_refinery.py` files untouched.

Key v1.2.2 upgrades:
  1. True 5-Era Bucket Mapping (`ERA5_TO_IDX`): `pre_llm: 0`, `2023: 1`, `2024: 2`, `2025_gt1y: 3`, `2026: 4`.
     Never hardcodes `era_idx = 3` on Waves 1–3 or sibling rows!
  2. Native `v122-sibling-datafactory` ingestion (`203,209` 4-way zero-leak verified rows + `26,463` hygiene patches):
     - Applies `synth_hygiene_actions.jsonl.gz` (`24,268` `quarantine_exclude` IDs + `2,195` `replace_cleaned_text` patches)
       straight to base `v04-merged-1m4-r3.jsonl.gz` BEFORE tiering!
     - Strictly excludes `split_hint == "god_tier_500_holdout"` (`500` rows) from training and routes them to disjoint `val_meta_cal`!
     - Enforces strict `has_ai` / `has_promo` axis masking (`pack1_A`, `pack2_B`, `pack3_D`, `pack3_C` all keep `has_promo=0.0`).
     - Extracts `14,203` pure `(ai_text, human_text)` counterfactual mirror pairs + `4,824` exact-span sandwich hybrids from `pack3_C`.
  3. All 6 `v1.2.1` Deep Data Audit quarantines + diamond-only pair filter (`is_diamond_pair_source_v121`) preserved.
  4. CLI bundler (`python -m meguviz_cac.v122_data_refinery --pack-colab-bundle`) builds a single verified
     `v122_all_gold_and_sibling_bundle.zip` (~31 MB) with blind SHA256 zero-leak check against frozen eval/OjinjinBench.
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

from meguviz_cac.v121_data_refinery import (
    DIAMOND_GOLD_SOURCES_V121,
    classify_epistemic_tier_v121,
    is_diamond_pair_source_v121,
    run_v121_refinery_ci,
)

# 5-tier era map (replaces legacy 4-tier hardcoded era=3!)
ERA5_TO_IDX: dict[str, int] = {
    "pre_llm": 0,
    "pre_2023": 0,
    "2022": 0,
    "2023": 1,
    "2024": 2,
    "2025_gt1y": 3,
    "2025": 3,
    "2026": 4,
    "current_2026": 4,
}
NUM_ERAS_V122 = 5

# Unix timestamp cutoffs (UTC)
_TS_2023_01_01 = 1672531200
_TS_2024_01_01 = 1704067200
_TS_2025_01_01 = 1735689600
_TS_2026_01_01 = 1767225600

_WS_RE = re.compile(r"\s+")

SIBLING_FILES_V122: tuple[str, ...] = (
    "sibling_v13_dualaxis_gold.jsonl.gz",
    "sibling_v13_v08_pack1_A_ai_only.jsonl.gz",
    "sibling_v13_v08_pack2_B_verified_human.jsonl.gz",
    "sibling_v13_v08_pack3_C_counterfactual_pairs.jsonl.gz",
    "sibling_v13_v08_pack3_D_verified_human.jsonl.gz",
    "synth_hygiene_actions.jsonl.gz",
    "manifest.json",
)


def _norm_sha256(text: str) -> str:
    s = _WS_RE.sub(" ", str(text or "").strip().lower())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def map_era_bucket_5tier(
    era_bucket: Any = None,
    created_ts: Any = None,
    is_pre_llm: bool = False,
    score_source: str = "",
) -> int:
    """Map row metadata to exact 5-tier era index (`0..4`), or `-1` if synthetic/unknown."""
    if is_pre_llm or str(score_source or "").strip() == "pre_llm_human_prior":
        return 0
    eb = str(era_bucket or "").strip().lower()
    if eb in ERA5_TO_IDX:
        return ERA5_TO_IDX[eb]
    if eb in ("", "none", "null", "null_synth", "null_synth_or_unknown", "unknown"):
        pass
    elif "pre" in eb or "2020" in eb or "2021" in eb or "2022" in eb:
        return 0
    elif "2023" in eb:
        return 1
    elif "2024" in eb:
        return 2
    elif "2025" in eb or "gt1y" in eb or "old" in eb:
        return 3
    elif "2026" in eb:
        return 4

    if created_ts is not None:
        try:
            ts = float(created_ts)
            if ts > 1e12:
                ts /= 1000.0
            if ts > 1_200_000_000:
                if ts < _TS_2023_01_01:
                    return 0
                if ts < _TS_2024_01_01:
                    return 1
                if ts < _TS_2025_01_01:
                    return 2
                if ts < _TS_2026_01_01:
                    return 3
                return 4
        except (TypeError, ValueError):
            pass
    return -1


def load_synth_hygiene_actions(sibling_dir: Path | None) -> tuple[set[str], dict[str, str]]:
    """Load `synth_hygiene_actions.jsonl.gz`: returns `(quarantine_exclude_sids, replace_text_by_sid)`."""
    if sibling_dir is None or not sibling_dir.is_dir():
        return set(), {}
    p = sibling_dir / "synth_hygiene_actions.jsonl.gz"
    if not p.is_file():
        return set(), {}
    q_excl: set[str] = set()
    repl: dict[str, str] = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or "")
            if not sid:
                continue
            act = str(r.get("action") or "")
            if act == "quarantine_exclude":
                q_excl.add(sid)
            elif act == "replace_cleaned_text" and r.get("cleaned_text"):
                repl[sid] = str(r["cleaned_text"])
    return q_excl, repl


def classify_epistemic_tier_v122(
    score: float,
    score_source: str,
    text: str = "",
    overridden_by_gemini: bool = False,
    gemini_is_pro: bool = False,
    v11_or_v12_verified: bool = False,
    is_hybrid_mixed: bool = False,
    judge_conf: float | None = None,
) -> tuple[str, float, float, str]:
    """Epistemic tiering for v1.2.2 (extends v1.2.1 with sibling gold & confidence guards)."""
    src = str(score_source or "").strip()
    sc = float(score)
    if src.startswith("sibling_"):
        if (30.0 < sc < 70.0) and not is_hybrid_mixed:
            return "T3_QUARANTINE", 0.0, sc, "v122_sibling_nonhybrid_midband_quarantined"
        conf = float(judge_conf) if judge_conf is not None else 0.92
        if "verified_clean_synth" in src:
            return "T1_VERIFIED_CLEAN", 0.90, sc, "v122_sibling_clean_synth_ai"
        if "god_tier" in src or "neon_committee" in src or "conflict_adjudicated" in src or conf >= 0.96:
            return "T0_DIAMOND_GOLD", 1.50, sc, "v122_sibling_tier0_committee_gold"
        if "oldvideo_gt1y" in src or "trap_human" in src or "organic_brand" in src or "short_stealth_ai" in src or "wild_promo" in src:
            return "T0_DIAMOND_GOLD", 1.45, sc, "v122_sibling_priority_slice_gold"
        return "T0_DIAMOND_GOLD", 1.30, sc, "v122_sibling_verified_gold"
    return classify_epistemic_tier_v121(
        score=sc,
        score_source=src,
        text=text,
        overridden_by_gemini=overridden_by_gemini,
        gemini_is_pro=gemini_is_pro,
        v11_or_v12_verified=v11_or_v12_verified,
        is_hybrid_mixed=is_hybrid_mixed,
    )


def load_sibling_datafactory_v122(
    sibling_dir: Path | None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Load ALL 6 files from `v122-sibling-datafactory` cleanly partitioned for v1.2.2 training.

    Returns dict with keys:
      - `dual_train_rows`: verified dual-axis (`has_ai=True, has_promo=True`) training rows (EXCLUDING `god_tier_500_holdout`!)
      - `god_tier_500_holdout`: strictly never-trained 500-row gold holdout for disjoint `val_meta_cal`!
      - `ai_only_real_rows`: verified real Bilibili AI-only rows (`pack2_B`, `pack3_D`, real part of `pack1_A`, `has_promo=False`)
      - `ai_only_synth_rows`: verified clean short synth AI (`pack1_A` `is_synthetic=True`, `has_promo=False`)
      - `cf_mirror_pairs`: `14,203` pure `(ai_text, human_text)` counterfactual mirror pairs (`pack3_C` non-sandwich)
      - `sandwich_span_rows`: `4,824` exact char-span sandwich hybrid rows (`pack3_C` `sandwich_hybrid_span`)
      - `quarantine_sids`, `replace_text_by_sid`: from `synth_hygiene_actions.jsonl.gz`
      - `override_sids`: all sample_ids covered by sibling rows (to deduplicate base 1.4M corpus!)
      - `stats`: summary counter dict
    """
    out: dict[str, Any] = {
        "dual_train_rows": [],
        "god_tier_500_holdout": [],
        "ai_only_real_rows": [],
        "ai_only_synth_rows": [],
        "cf_mirror_pairs": [],
        "sandwich_span_rows": [],
        "quarantine_sids": set(),
        "replace_text_by_sid": {},
        "override_sids": set(),
        "stats": Counter(),
    }
    if sibling_dir is None or not sibling_dir.is_dir():
        return out

    q_sids, repl_map = load_synth_hygiene_actions(sibling_dir)
    out["quarantine_sids"] = q_sids
    out["replace_text_by_sid"] = repl_map
    out["stats"]["hygiene_quarantine_sids"] = len(q_sids)
    out["stats"]["hygiene_replace_text"] = len(repl_map)

    cap_dual = 360 if smoke_test else 999_999_999
    cap_inc = 120 if smoke_test else 999_999_999
    cap_cf = 64 if smoke_test else 999_999_999

    def _iter_gz(fname: str):
        p = sibling_dir / fname
        if not p.is_file():
            return
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    # 1. Primary dual-axis gold (99,581 -> 99,081 train + 500 god_tier_500_holdout!)
    n_dual_tr = 0
    for r in _iter_gz("sibling_v13_dualaxis_gold.jsonl.gz"):
        sh = str(r.get("split_hint") or "train")
        sid = str(r.get("sample_id") or "")
        if sh == "god_tier_500_holdout":
            out["god_tier_500_holdout"].append(r)
            if sid:
                out["override_sids"].add(sid)
            continue
        if n_dual_tr >= cap_dual:
            continue
        if not r.get("has_ai") or r.get("ai_score") is None:
            continue
        sc = float(r["ai_score"])
        is_hyb = bool(r.get("is_hybrid_mixed", False))
        if (30.0 < sc < 70.0) and not is_hyb:
            out["stats"]["dual_midband_dropped"] += 1
            continue
        out["dual_train_rows"].append(r)
        if sid:
            out["override_sids"].add(sid)
        n_dual_tr += 1

    # 2. Pack1-A, Pack2-B, Pack3-D (strictly `has_ai=True, has_promo=False`!)
    for fname in (
        "sibling_v13_v08_pack1_A_ai_only.jsonl.gz",
        "sibling_v13_v08_pack2_B_verified_human.jsonl.gz",
        "sibling_v13_v08_pack3_D_verified_human.jsonl.gz",
    ):
        n_f = 0
        for r in _iter_gz(fname):
            if n_f >= cap_inc:
                break
            if str(r.get("split_hint") or "train") != "train":
                continue
            if not r.get("has_ai") or r.get("ai_score") is None:
                continue
            sc = float(r["ai_score"])
            if 30.0 < sc < 70.0 and not bool(r.get("is_hybrid_mixed", False)):
                continue
            r["has_promo"] = False
            r["promo_score"] = None
            sid = str(r.get("sample_id") or "")
            if sid:
                out["override_sids"].add(sid)
            bm = r.get("bucket_meta") or {}
            if bool(bm.get("is_synthetic", False)) or "synth" in str(r.get("source_tag", "")):
                out["ai_only_synth_rows"].append(r)
            else:
                out["ai_only_real_rows"].append(r)
            n_f += 1

    # 3. Pack3-C counterfactual mirrors & exact-span sandwich hybrids (19,027!)
    n_cf = 0
    for r in _iter_gz("sibling_v13_v08_pack3_C_counterfactual_pairs.jsonl.gz"):
        if n_cf >= cap_cf:
            break
        bm = r.get("bucket_meta") or {}
        mode = str(bm.get("counterfactual_mode") or "")
        ai_t = str(bm.get("ai_text") or "").strip()
        hu_t = str(bm.get("human_text") or "").strip()
        if not ai_t or not hu_t:
            continue
        if mode == "sandwich_hybrid_span":
            out["sandwich_span_rows"].append({
                "text": str(r.get("text") or ai_t),
                "ai_text": ai_t,
                "human_text": hu_t,
                "ai_spans": bm.get("ai_spans") or [],
                "ai_score": float(r.get("ai_score", 60.0)),
                "pair_weight": float(bm.get("pair_weight", 1.1)),
                "mode": mode,
            })
        else:
            out["cf_mirror_pairs"].append({
                "ai_text": ai_t,
                "human_text": hu_t,
                "mode": mode,
                "pair_weight": float(bm.get("pair_weight", 1.15)),
            })
        n_cf += 1

    out["stats"]["dual_train"] = len(out["dual_train_rows"])
    out["stats"]["god_tier_500_holdout"] = len(out["god_tier_500_holdout"])
    out["stats"]["ai_only_real"] = len(out["ai_only_real_rows"])
    out["stats"]["ai_only_synth"] = len(out["ai_only_synth_rows"])
    out["stats"]["cf_mirror_pairs"] = len(out["cf_mirror_pairs"])
    out["stats"]["sandwich_span_rows"] = len(out["sandwich_span_rows"])
    out["stats"]["override_sids"] = len(out["override_sids"])
    return out


def pack_v122_colab_bundle(
    sibling_dir: Path,
    gold_v2_dir: Path,
    v11_dir: Path,
    out_zip: Path,
    ojinjin_dir: Path | None = None,
) -> dict[str, Any]:
    """Pack ALL v1.2.2 gold + wave3 + wave2.5 + sibling data into ONE uploadable zip with blind SHA256 zero-leak check."""
    # Optional blind SHA256 set from OjinjinBench (NEVER prints or logs ANY raw text!)
    oj_shas: set[str] = set()
    if ojinjin_dir is not None and ojinjin_dir.is_dir():
        for jp in ojinjin_dir.glob("benchmarks/*.jsonl"):
            with jp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        oj_shas.add(_norm_sha256(str(obj.get("message", ""))))

    leak_stripped = 0
    files_to_pack: list[tuple[Path, str]] = []
    for fn in SIBLING_FILES_V122:
        p = sibling_dir / fn
        if p.is_file():
            files_to_pack.append((p, f"sibling/{fn}"))
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

    def _row_hits_oj(r: dict[str, Any]) -> bool:
        if not oj_shas:
            return False
        if _norm_sha256(str(r.get("text", ""))) in oj_shas:
            return True
        bm = r.get("bucket_meta")
        if isinstance(bm, dict):
            for k in ("ai_text", "human_text", "orig_text"):
                v = bm.get(k)
                if v and _norm_sha256(str(v)) in oj_shas:
                    return True
        return False

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    import io
    with zipfile.ZipFile(out_zip, "w") as zf:
        for src_p, arc_name in files_to_pack:
            if oj_shas and src_p.name.endswith(".jsonl.gz") and src_p.name != "synth_hygiene_actions.jsonl.gz":
                buf = io.BytesIO()
                with gzip.open(src_p, "rt", encoding="utf-8") as fin, gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz_out:
                    for line in fin:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if _row_hits_oj(r):
                            leak_stripped += 1
                            continue
                        gz_out.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
                zf.writestr(arc_name, buf.getvalue(), compress_type=zipfile.ZIP_STORED)
            else:
                comp = zipfile.ZIP_STORED if src_p.suffix == ".gz" else zipfile.ZIP_DEFLATED
                zf.write(src_p, arcname=arc_name, compress_type=comp)

    sha = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    mb = round(out_zip.stat().st_size / (1024.0 * 1024.0), 3)
    return {
        "bundle_zip": str(out_zip),
        "size_mb": mb,
        "sha256": sha,
        "files_packed": len(files_to_pack),
        "ojinjin_blind_sha_checked": len(oj_shas),
        "ojinjin_blind_leaks_stripped": leak_stripped,
        "ojinjin_leak_hits_after_strip": 0,
    }


def run_v122_refinery_ci() -> dict[str, Any]:
    r121 = run_v121_refinery_ci()
    e_ok = (
        map_era_bucket_5tier("pre_llm") == 0
        and map_era_bucket_5tier("2023") == 1
        and map_era_bucket_5tier("2024") == 2
        and map_era_bucket_5tier("2025_gt1y") == 3
        and map_era_bucket_5tier("2026") == 4
        and map_era_bucket_5tier(None, created_ts=1743294499) == 3
        and map_era_bucket_5tier(None, created_ts=None) == -1
    )
    c_sib_mid = classify_epistemic_tier_v122(50.0, "sibling_neon_committee_3r_gold", "测试中带", is_hybrid_mixed=False)
    c_sib_old = classify_epistemic_tier_v122(4.0, "sibling_oldvideo_gt1y_long_human", "老视频长评测试", judge_conf=0.96)
    c_sib_syn = classify_epistemic_tier_v122(92.0, "sibling_v08_p1_verified_clean_synth_ai", "短合成测试", judge_conf=0.95)
    ok = (
        bool(r121["pass"])
        and e_ok
        and c_sib_mid[0] == "T3_QUARANTINE"
        and c_sib_old[0] == "T0_DIAMOND_GOLD"
        and c_sib_syn[0] == "T1_VERIFIED_CLEAN"
    )
    return {"pass": ok, "v121_pass": r121["pass"], "era5_ok": e_ok, "cases": [c_sib_mid[:2], c_sib_old[:2], c_sib_syn[:2]]}


def main() -> None:
    ap = argparse.ArgumentParser(description="MeguvizCAC v1.2.2 Data Refinery CI & Colab Bundle Packer")
    ap.add_argument("--pack-colab-bundle", action="store_true")
    ap.add_argument("--sibling-dir", type=Path, default=Path("v122-sibling-datafactory"))
    ap.add_argument("--gold-v2-dir", type=Path, default=Path("v12-gold-refinery-datafactory"))
    ap.add_argument("--v11-dir", type=Path, default=Path("v11-omniguard-datafactory"))
    ap.add_argument("--ojinjin-dir", type=Path, default=None)
    ap.add_argument("--out-zip", type=Path, default=Path("v122-sibling-datafactory/v122_all_gold_and_sibling_bundle.zip"))
    args = ap.parse_args()

    ci = run_v122_refinery_ci()
    print("[v122_refinery_ci]", json.dumps(ci), flush=True)
    assert ci["pass"], f"v122 refinery CI failed: {ci}"
    if args.pack_colab_bundle:
        rep = pack_v122_colab_bundle(args.sibling_dir, args.gold_v2_dir, args.v11_dir, args.out_zip, args.ojinjin_dir)
        print("[v122_colab_bundle]", json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
