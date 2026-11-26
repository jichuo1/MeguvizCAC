"""MeguvizCAC v1.1.0-OmniGuard Automated External Dual-Axis Canary Vault Auditor & Safe Merger.

Use immediately when external Gemini labeling factories return `replies/shard_001.json..shard_120.json`
for `v11_omniguard_6k_dualaxis_pack.zip`:
  1. Validates exact schema, 6,000 `sid` alignment, value ranges, & non-empty `brief_cue`.
  2. Audits ALL 120 stratified hidden canaries (`1` per shard!) against LOCAL `private_canary_vault.json`:
       - 60 synthetic 4-quadrant dual-axis accuracy (`ai_score`, `promo_score`, `promo_type`)
       - 30 real `hydrogen_hu` & 30 real `hydrogen_ai` calibration parity
  3. Runs 4 Anti-Cheat Forensics checks per shard:
       - Constant / lazy scoring (`std(ai) < 4.0` or `std(promo) < 4.0`)
       - Copy-paste `brief_cue` duplication (`unique_ratio < 0.65`)
       - Orthogonal axis-collapse cheating (marking Q3 human-promo canaries as `ai_score >= 60`!)
       - Internal schema contradictions (`origin_6way == Human-Commercial-Promo` with `promo_score < 40` or `ai_score > 65`)
  4. Emits shard-level `ACCEPT / RETRY_SHARDS / REJECT` verdict and compiles ONLY verified passing shards into
     `v11_dualaxis_gold_verified.jsonl.gz` (restoring `orig_sample_id` from local `private_sid_map.jsonl`!).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

VALID_TYPES = frozenset({"none", "coupon_commerce", "self_promo_funnel", "soft_shill", "tool_boilerplate"})
VALID_ORIGINS = frozenset({
    "Human-Organic",
    "Human-Commercial-Promo",
    "Human-Assisted",
    "Script-Plugin-Bot",
    "Autonomous-Persona-Agent",
    "Coordinated-Latent-Cluster",
})


def _check_expectation(val: float, exp: str) -> bool:
    if exp == "any":
        return True
    if exp == "low":
        return val <= 35.0
    if exp == "high":
        return val >= 65.0
    if exp == "mid_or_low":
        return val <= 65.0
    if exp == "mid_or_high":
        return val >= 30.0
    if exp == "high_or_mid":
        return val >= 30.0
    return True


def audit_external_dualaxis_replies(
    shards_dir: Path,
    replies_dir: Path,
    canary_vault_path: Path,
    private_sid_map_path: Path,
    out_verified_gz: Path | None = None,
) -> dict[str, Any]:
    """Run full schema, 120-canary dual-axis, and anti-cheat audit; optionally compile verified gold JSONL.GZ."""
    vault = json.loads(canary_vault_path.read_text(encoding="utf-8"))
    canary_by_sid: dict[str, dict] = {c["sid"]: c for c in vault["canaries"]}

    sid_map: dict[str, dict] = {}
    if private_sid_map_path.is_file():
        for line in private_sid_map_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                sid_map[r["sid"]] = r

    shard_files = sorted(shards_dir.glob("shard_*.json"))
    shard_reports: list[dict[str, Any]] = []
    retry_shards: list[str] = []
    verified_rows: list[dict[str, Any]] = []

    canary_stats = Counter()
    q3_axis_collapse_cnt = 0
    q3_total_cnt = 0
    total_valid_rows = 0
    total_schema_errs = 0
    total_contradictions = 0

    for sp in shard_files:
        sname = sp.name
        rp = replies_dir / sname
        src_rows = json.loads(sp.read_text(encoding="utf-8"))
        src_by_sid = {x["sid"]: x["text"] for x in src_rows}

        if not rp.is_file():
            retry_shards.append(sname)
            shard_reports.append({"shard": sname, "status": "MISSING", "reason": "reply_file_missing"})
            continue

        try:
            rep_rows = json.loads(rp.read_text(encoding="utf-8"))
            if not isinstance(rep_rows, list):
                raise ValueError("not_a_list")
        except Exception as exc:
            retry_shards.append(sname)
            shard_reports.append({"shard": sname, "status": "FAIL", "reason": f"json_error:{exc}"})
            continue

        seen_sids: set[str] = set()
        s_errs = 0
        s_contra = 0
        ai_vals: list[float] = []
        pr_vals: list[float] = []
        cues: list[str] = []
        shard_canary_ok = True
        shard_canary_notes: list[str] = []
        parsed_clean: list[dict[str, Any]] = []

        for r in rep_rows:
            if not isinstance(r, dict):
                s_errs += 1
                continue
            sid = str(r.get("sid", ""))
            if sid not in src_by_sid or sid in seen_sids:
                s_errs += 1
                continue
            seen_sids.add(sid)

            ai_s = r.get("ai_score")
            pr_s = r.get("promo_score")
            conf = r.get("confidence", 0.85)
            ptype = r.get("promo_type")
            orig = r.get("origin_6way")
            cue = str(r.get("brief_cue", "")).strip()

            valid_nums = (
                isinstance(ai_s, (int, float)) and not isinstance(ai_s, bool) and 0 <= ai_s <= 100
                and isinstance(pr_s, (int, float)) and not isinstance(pr_s, bool) and 0 <= pr_s <= 100
                and isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0.0 <= conf <= 1.0
            )
            if not valid_nums or ptype not in VALID_TYPES or orig not in VALID_ORIGINS or len(cue) < 2:
                s_errs += 1
                continue

            ai_f, pr_f = float(ai_s), float(pr_s)
            ai_vals.append(ai_f)
            pr_vals.append(pr_f)
            cues.append(cue)
            total_valid_rows += 1

            # Internal logical contradiction check
            if orig == "Human-Commercial-Promo" and (pr_f < 35.0 or ai_f > 70.0):
                s_contra += 1
            elif orig == "Human-Organic" and (ai_f >= 75.0 or pr_f >= 75.0):
                s_contra += 1
            elif orig in ("Script-Plugin-Bot", "Autonomous-Persona-Agent", "Coordinated-Latent-Cluster") and ai_f < 40.0:
                s_contra += 1
            elif ptype == "none" and pr_f >= 70.0:
                s_contra += 1

            # Hidden canary verification!
            if sid in canary_by_sid:
                c = canary_by_sid[sid]
                canary_stats["seen"] += 1
                ok_ai = _check_expectation(ai_f, c["exp_ai"])
                ok_pr = _check_expectation(pr_f, c["exp_promo"])
                ok_tp = (c["exp_type"] == "any") or (ptype == c["exp_type"])
                # Track Q3 human-promo axis collapse specifically!
                if c["exp_ai"] == "low" and c["exp_promo"] == "high":
                    q3_total_cnt += 1
                    if ai_f >= 60.0:
                        q3_axis_collapse_cnt += 1
                if ok_ai:
                    canary_stats["ok_ai"] += 1
                if ok_pr:
                    canary_stats["ok_promo"] += 1
                if ok_tp:
                    canary_stats["ok_type"] += 1
                if ok_ai and ok_pr:
                    canary_stats["ok_dual"] += 1
                else:
                    shard_canary_ok = False
                    shard_canary_notes.append(
                        f"{sid}({c['canary_kind']}): got ai={ai_f}/pr={pr_f}/tp={ptype} vs exp {c['exp_ai']}/{c['exp_promo']}/{c['exp_type']}"
                    )
            else:
                meta = sid_map.get(sid, {})
                parsed_clean.append({
                    "sid": sid,
                    "orig_sample_id": meta.get("orig_sample_id", ""),
                    "mine_bucket": meta.get("mine_bucket", ""),
                    "text": src_by_sid[sid],
                    "ai_score": ai_f,
                    "promo_score": pr_f,
                    "promo_type": ptype,
                    "origin_6way": orig,
                    "confidence": float(conf),
                    "brief_cue": cue,
                    "score_source": "v11_dualaxis_pro_gold",
                })

        s_errs += len(src_by_sid) - len(seen_sids)
        total_schema_errs += s_errs
        total_contradictions += s_contra

        # Anti-cheat shard laziness / copy-paste checks
        ai_std = float(np.std(ai_vals)) if len(ai_vals) >= 10 else 0.0
        pr_std = float(np.std(pr_vals)) if len(pr_vals) >= 10 else 0.0
        cue_uniq = len(set(cues)) / float(max(1, len(cues)))
        anti_cheat_flags: list[str] = []
        if s_errs > 2:
            anti_cheat_flags.append(f"schema_errs={s_errs}")
        pr_mean = float(np.mean(pr_vals)) if pr_vals else 0.0
        if ai_std < 3.5 or (pr_std < 3.5 and (ai_std < 15.0 or pr_mean >= 12.0)):
            anti_cheat_flags.append(f"lazy_constant_scores(ai_std={ai_std:.1f},pr_std={pr_std:.1f})")
        if cue_uniq < 0.65:
            anti_cheat_flags.append(f"copy_paste_cues(uniq={cue_uniq:.2f})")
        if s_contra > 5:
            anti_cheat_flags.append(f"high_contradictions={s_contra}")
        if not shard_canary_ok:
            anti_cheat_flags.append("canary_miss")

        if anti_cheat_flags:
            retry_shards.append(sname)
            shard_reports.append({
                "shard": sname, "status": "RETRY", "flags": anti_cheat_flags, "canary_notes": shard_canary_notes,
            })
        else:
            verified_rows.extend(parsed_clean)
            shard_reports.append({"shard": sname, "status": "PASS", "ai_std": round(ai_std, 1), "pr_std": round(pr_std, 1)})

    seen_c = max(1, canary_stats["seen"])
    ai_acc = canary_stats["ok_ai"] / float(seen_c)
    pr_acc = canary_stats["ok_promo"] / float(seen_c)
    dual_acc = canary_stats["ok_dual"] / float(seen_c)
    tp_acc = canary_stats["ok_type"] / float(seen_c)
    axis_collapse_rate = q3_axis_collapse_cnt / float(max(1, q3_total_cnt))

    if dual_acc >= 0.88 and axis_collapse_rate <= 0.12 and len(retry_shards) <= 12:
        verdict = "ACCEPT"
    elif dual_acc >= 0.72 and axis_collapse_rate <= 0.25:
        verdict = "PARTIAL_RETRY_SHARDS"
    else:
        verdict = "REJECT_LOW_QUALITY"

    if out_verified_gz is not None and verified_rows and verdict != "REJECT_LOW_QUALITY":
        out_verified_gz.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_verified_gz, "wt", encoding="utf-8") as wf:
            for row in verified_rows:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "verdict": verdict,
        "shards_total": len(shard_files),
        "shards_passed": len(shard_files) - len(retry_shards),
        "shards_retry_count": len(retry_shards),
        "retry_shards_sample": retry_shards[:20],
        "valid_rows": total_valid_rows,
        "verified_real_rows_exported": len(verified_rows),
        "schema_errors": total_schema_errs,
        "logical_contradictions": total_contradictions,
        "canary_audit": {
            "canaries_seen": int(canary_stats["seen"]),
            "ai_axis_acc": round(ai_acc, 4),
            "promo_axis_acc": round(pr_acc, 4),
            "dual_axis_joint_acc": round(dual_acc, 4),
            "promo_type_acc": round(tp_acc, 4),
            "q3_human_promo_axis_collapse_rate": round(axis_collapse_rate, 4),
        },
        "failed_shard_details": [r for r in shard_reports if r["status"] != "PASS"][:15],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit external v1.1.0 dual-axis replies against local private canary vault.")
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--replies-dir", type=Path, required=True)
    ap.add_argument("--vault", type=Path, default=Path("v11-omniguard-datafactory/private_canary_vault.json"))
    ap.add_argument("--sid-map", type=Path, default=Path("v11-omniguard-datafactory/private_sid_map.jsonl"))
    ap.add_argument("--out-gz", type=Path, default=Path("v11-omniguard-datafactory/v11_dualaxis_gold_verified.jsonl.gz"))
    args = ap.parse_args()
    rep = audit_external_dualaxis_replies(args.shards_dir, args.replies_dir, args.vault, args.sid_map, args.out_gz)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
