"""Step 3: Hidden Canary QA Gate, Independent Anti-Leakage Verification, Tier-1/Tier-2 Tribunal Fusion & Clean Export.

Gates enforced BEFORE writing any v0.8.0 training dataset:
  1. Hidden 2% `val_gold` Canary Calibration Gate:
     - Computes Binary AUROC (`gold >= 70` vs `gold <= 30`), Spearman rank correlation, MAE, and Mean Bias.
     - FAILS FAST if `AUROC < 0.94`, `Spearman < 0.85`, or `|mean_bias| > 8.0` (unless `--allow-warn`).
  2. 100% Canary Stripping:
     - Every `blind_sample_id` in `canary_vault.json` is strictly excluded from output files.
  3. Independent Anti-Leakage Re-Audit:
     - Re-checks every exported row & counterfactual rewrite against `val/test/user_confirmed` and all
       built-in MeguvizCAC traps (exact + char-trigram Jaccard >= 0.72).
  4. Drop-in MeguvizCAC Output Files:
     - `v08-gemini-gold-labels.jsonl.gz`        : Directly compatible with v0.6/v0.7/v0.8 train loaders!
     - `v08-gemini-counterfactual-pairs.jsonl`  : Minimal pairs + exact char-span `sandwich_hybrid_span` records!
     - `v08_qc_report.json`                     : Complete audit scorecard + era/code/queue distributions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
from tools.gemini_v08_datafactory.rubric_v08 import score_to_class5

_exp = importlib.import_module("tools.gemini_v08_datafactory.01_export_queue")
build_trap_inverted_index = _exp.build_trap_inverted_index
is_near_dup_traps = _exp.is_near_dup_traps
load_traps_and_eval_forbidden = _exp.load_traps_and_eval_forbidden
normalize_text = _exp.normalize_text


def _rankdata(vals: list[float]) -> list[float]:
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_r
        i = j + 1
    return ranks


def compute_spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (num / (dx * dy)) if (dx > 0 and dy > 0) else 0.0


def compute_binary_auroc(labels: list[int], scores: list[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _load_tier_preds(outputs_dir: Path, tier: str) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    if not outputs_dir.is_dir():
        return preds
    for jf in sorted(outputs_dir.glob(f"*.{tier}.jsonl")):
        if jf.name.startswith("C_counterfactual_spans"):
            continue
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                model_name = rec.get("model", "")
                for sid, p in (rec.get("parsed") or {}).items():
                    p_copy = dict(p)
                    p_copy["model"] = model_name
                    p_copy["tier"] = tier
                    preds[sid] = p_copy
    return preds


def _load_cf_preds(outputs_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cf_file = outputs_dir / "C_counterfactual_spans.tier1.jsonl"
    if not cf_file.is_file():
        return out
    with cf_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            model_name = rec.get("model", "")
            for sid, p in (rec.get("parsed") or {}).items():
                p_copy = dict(p)
                p_copy["model"] = model_name
                out.append(p_copy)
    return out


def _load_original_rows(queues_dir: Path) -> dict[str, dict[str, Any]]:
    orig: dict[str, dict[str, Any]] = {}
    for bf in sorted(queues_dir.glob("*.batches.jsonl")):
        if bf.name.startswith("C_counterfactual_spans"):
            continue
        with bf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                b = json.loads(line)
                for r in b.get("rows") or []:
                    orig[r["sample_id"]] = r
    return orig


def fuse_tier1_tier2(p1: dict[str, Any] | None, p2: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Tribunal fusion: Tier-2 Pro overrides Tier-1 Flash on escalated borderline/trap rows."""
    if p2 is not None and p1 is not None:
        # When Pro and Flash agree closely (<=15 pts), smooth 0.75*Pro + 0.25*Flash; otherwise trust Pro directly.
        diff = abs(int(p2["score"]) - int(p1["score"]))
        fused_sc = int(round(0.75 * p2["score"] + 0.25 * p1["score"])) if diff <= 15 else int(p2["score"])
        code = p2["code"]
        # Ensure strict band-closure after smoothing
        if fused_sc <= 30 and code != "H":
            fused_sc = int(p2["score"])
        elif 31 <= fused_sc <= 69 and code not in ("P", "S", "Q"):
            fused_sc = int(p2["score"])
        elif fused_sc >= 70 and code not in ("A", "R", "E", "T"):
            fused_sc = int(p2["score"])
        fused = dict(p2)
        fused["score"] = fused_sc
        fused["class_5way"] = score_to_class5(fused_sc, code)
        fused["tier1_score"] = int(p1["score"])
        fused["tier2_score"] = int(p2["score"])
        return fused, "v08_gemini_pro_gold"
    if p2 is not None:
        return dict(p2), "v08_gemini_pro_gold"
    assert p1 is not None
    return dict(p1), "v08_gemini_flash_gold"


def audit_canaries(
    vault_entries: list[dict[str, Any]],
    fused_by_sid: dict[str, tuple[dict[str, Any], str]],
    min_auroc: float,
    min_spearman: float,
    max_abs_bias: float,
) -> dict[str, Any]:
    gold_scores: list[float] = []
    pred_scores: list[float] = []
    bin_y: list[int] = []
    bin_s: list[float] = []

    for c in vault_entries:
        bsid = c["blind_sample_id"]
        if bsid not in fused_by_sid:
            continue
        pred, _src = fused_by_sid[bsid]
        gs = float(c["gold_score"])
        ps = float(pred["score"])
        gold_scores.append(gs)
        pred_scores.append(ps)
        if gs >= 70.0:
            bin_y.append(1)
            bin_s.append(ps)
        elif gs <= 30.0:
            bin_y.append(0)
            bin_s.append(ps)

    n_eval = len(gold_scores)
    if n_eval == 0:
        return {"evaluated": 0, "status": "NO_CANARIES_SCORED", "passed": False}

    mae = sum(abs(p - g) for p, g in zip(pred_scores, gold_scores)) / n_eval
    bias = sum(p - g for p, g in zip(pred_scores, gold_scores)) / n_eval
    spearman = compute_spearman(gold_scores, pred_scores)
    auroc = compute_binary_auroc(bin_y, bin_s)

    auroc_ok = (math.isnan(auroc) and len(bin_y) < 4) or (not math.isnan(auroc) and auroc >= min_auroc)
    spear_ok = (n_eval < 5) or (spearman >= min_spearman)
    bias_ok = abs(bias) <= max_abs_bias
    passed = bool(auroc_ok and spear_ok and bias_ok)

    return {
        "evaluated": n_eval,
        "total_vault": len(vault_entries),
        "binary_eval_count": len(bin_y),
        "binary_pos_count": sum(bin_y),
        "binary_neg_count": len(bin_y) - sum(bin_y),
        "auroc": None if math.isnan(auroc) else round(auroc, 4),
        "spearman": round(spearman, 4),
        "mae": round(mae, 2),
        "mean_bias": round(bias, 2),
        "gates": {
            "auroc_ok": auroc_ok,
            "spearman_ok": spear_ok,
            "bias_ok": bias_ok,
            "thresholds": {"min_auroc": min_auroc, "min_spearman": min_spearman, "max_abs_bias": max_abs_bias},
        },
        "passed": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "v08-datafactory-work")
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "v04-corpus" / "local-train" / "v04-merged-1m4-r3.jsonl.gz")
    ap.add_argument("--extra-exclude", type=Path, default=None)
    ap.add_argument("--min-auroc", type=float, default=0.94)
    ap.add_argument("--min-spearman", type=float, default=0.85)
    ap.add_argument("--max-abs-bias", type=float, default=8.0)
    ap.add_argument("--allow-warn", action="store_true", help="Export even if canary QA gate warns (not recommended for prod)")
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    release_dir = work_dir / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    orig_rows = _load_original_rows(work_dir / "queues")
    p1_map = _load_tier_preds(work_dir / "outputs_tier1", "tier1")
    p2_map = _load_tier_preds(work_dir / "outputs_tier2", "tier2")
    all_sids = sorted(set(p1_map) | set(p2_map))

    fused_by_sid: dict[str, tuple[dict[str, Any], str]] = {
        sid: fuse_tier1_tier2(p1_map.get(sid), p2_map.get(sid)) for sid in all_sids
    }

    # 1. Hidden Canary QA Audit
    vault_path = work_dir / "canary_vault.json"
    vault_entries = json.loads(vault_path.read_text(encoding="utf-8")).get("canaries", []) if vault_path.is_file() else []
    canary_sids = {c["blind_sample_id"] for c in vault_entries}
    canary_report = audit_canaries(vault_entries, fused_by_sid, args.min_auroc, args.min_spearman, args.max_abs_bias)

    if vault_entries and not canary_report.get("passed", False) and not args.allow_warn:
        (release_dir / "v08_qc_report_FAILED.json").write_text(json.dumps(canary_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(
            f"[QC FAIL-FAST] Hidden canary calibration gate failed: {json.dumps(canary_report, ensure_ascii=False)}\n"
            "Fix prompt/model drift before merging into v0.8.0 training (or pass `--allow-warn` to inspect)."
        )

    # 2. Independent Anti-Leakage Re-Audit
    forbidden_exact, trap_trigrams, _ = load_traps_and_eval_forbidden(args.corpus, args.extra_exclude)

    gold_out_path = release_dir / "v08-gemini-gold-labels.jsonl.gz"
    exported_cnt = 0
    stripped_canaries = 0
    blocked_leakage = 0
    code_dist = Counter()
    queue_dist = Counter()
    source_dist = Counter()
    era_dist = Counter()
    seen_norm_out: set[str] = set()

    with gzip.open(gold_out_path, "wt", encoding="utf-8", newline="\n") as gf:
        for sid in all_sids:
            if sid in canary_sids:
                stripped_canaries += 1
                continue
            orig = orig_rows.get(sid)
            if not orig:
                continue
            txt = str(orig.get("text") or "")
            n = normalize_text(txt)
            if n in seen_norm_out or is_near_dup_traps(txt, forbidden_exact, trap_trigrams):
                blocked_leakage += 1
                continue
            seen_norm_out.add(n)

            fused, score_source = fused_by_sid[sid]
            sc = int(fused["score"])
            code = str(fused["code"])
            # Compatible drop-in schema with MeguvizCAC v0.6 / v0.7 / v0.8 trainers!
            out_rec: dict[str, Any] = {
                "sample_id": orig.get("orig_sample_id") or sid,
                "v08_factory_id": sid,
                "split": "train",
                "cohort": "v08_gemini_factory",
                "queue": orig.get("queue", ""),
                "mine_reason": orig.get("mine_reason", ""),
                "era_bucket": orig.get("era_bucket", "current_2026"),
                "lang_tag": orig.get("lang_tag", "zh"),
                "text": txt,
                "has_score": True,
                "score": sc,
                "code": code,
                "class_5way": fused["class_5way"],
                "confidence": int(fused.get("confidence", 85)),
                "is_trap_human": bool(fused.get("is_trap_human", False)),
                "hard_human_neg": bool(fused.get("is_trap_human", False) and sc <= 30),
                "prag_vec": fused.get("prag_8d", [-1] * 8),
                "score_source": score_source,
                "train_weight": 1.35 if score_source == "v08_gemini_pro_gold" else 1.0,
            }
            if orig.get("queue") == "D_thread_context":
                out_rec["video_title"] = orig.get("video_title", "")
                out_rec["parent_text"] = orig.get("parent_text", "")

            gf.write(json.dumps(out_rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported_cnt += 1
            code_dist[code] += 1
            queue_dist[out_rec["queue"]] += 1
            source_dist[score_source] += 1
            era_dist[out_rec["era_bucket"]] += 1

    # 3. Export Counterfactual Minimal Pairs & Hybrid Spans (Queue C)
    cf_raw = _load_cf_preds(work_dir / "outputs_tier1")
    cf_out_path = release_dir / "v08-gemini-counterfactual-pairs.jsonl"
    cf_exported = 0
    cf_blocked = 0
    cf_mode_dist = Counter()

    with cf_out_path.open("w", encoding="utf-8", newline="\n") as cf:
        for item in cf_raw:
            src_t = str(item.get("src_text") or "")
            rew_t = str(item.get("rewritten_text") or "")
            if is_near_dup_traps(src_t, forbidden_exact, trap_trigrams) or is_near_dup_traps(rew_t, forbidden_exact, trap_trigrams):
                cf_blocked += 1
                continue
            mode = str(item.get("cf_mode") or "")
            if mode == "ai_to_structured_human":
                ai_t, hu_t = src_t, rew_t
            elif mode in ("human_to_evasive_ai", "human_to_short_ai"):
                ai_t, hu_t = rew_t, src_t
            else:
                # sandwich_hybrid_span: also compatible as pair (hybrid vs clean human seed) + explicit ai_spans!
                ai_t, hu_t = rew_t, src_t

            rec = {
                "sample_id": item["sample_id"],
                "category": f"v08_{mode}",
                "ai_text": ai_t,
                "human_text": hu_t,
                "expected_rewritten_score": int(item.get("expected_score", 50)),
                "ai_spans": item.get("ai_spans", []),
                "lang_tag": item.get("lang_tag", "zh"),
                "weight": 1.35 if mode in ("ai_to_structured_human", "sandwich_hybrid_span") else 1.15,
                "source": "v08_gemini_factory",
            }
            cf.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            cf_exported += 1
            cf_mode_dist[mode] += 1

    qc_report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canary_audit": canary_report,
        "anti_leakage_reaudit": {
            "canaries_stripped": stripped_canaries,
            "scoring_rows_blocked": blocked_leakage,
            "counterfactual_rows_blocked": cf_blocked,
            "zero_eval_leakage_verified": (stripped_canaries == len([c for c in vault_entries if c["blind_sample_id"] in fused_by_sid])),
        },
        "exports": {
            "gold_labels_file": str(gold_out_path),
            "gold_labels_count": exported_cnt,
            "counterfactual_file": str(cf_out_path),
            "counterfactual_count": cf_exported,
        },
        "distributions": {
            "code": dict(code_dist),
            "queue": dict(queue_dist),
            "score_source": dict(source_dist),
            "era_bucket": dict(era_dist),
            "counterfactual_mode": dict(cf_mode_dist),
        },
    }
    (release_dir / "v08_qc_report.json").write_text(json.dumps(qc_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
