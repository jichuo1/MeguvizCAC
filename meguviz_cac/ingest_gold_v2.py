"""One-Command Strict QC, 50/50 Canary Audit, & Gold-v2 Ingestor for `v12_gold_relabel_and_boundary_39shard_pack`.

Usage once `replies/` (or `*_replies.zip`) returns from external annotation:
    python -m meguviz_cac.ingest_gold_v2 --replies path/to/replies_or_zip

Produces cleanly separated, zero-leakage datasets in `v12-gold-refinery-datafactory/`:
  - `v12_gold_v2_train.jsonl.gz`       (train_hydrogen 962 + Bucket B boundary 467 = 1,429 dual-axis gold!)
  - `v12_gold_v2_val.jsonl.gz`         (val_gold_v2 = 208 re-adjudicated validation gold!)
  - `v12_gold_v2_test.jsonl.gz`        (test_bili_gold_v2 = 263 parallel test benchmark; NEVER overwrites v1!)
  - `v12_gold_v2_qc_report.json`       (full 50/50 canary audit + drift stats vs Gold-v1!)
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zipfile

VALID_PT = {"none", "coupon_commerce", "self_promo_funnel", "soft_shill", "tool_boilerplate"}
VALID_O6 = {
    "Human-Organic", "Human-Commercial-Promo", "Human-Assisted",
    "Script-Plugin-Bot", "Autonomous-Persona-Agent", "Coordinated-Latent-Cluster",
}


def load_replies(src: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for nm in sorted(z.namelist()):
                if nm.endswith(".json") and "shard_" in nm:
                    for row in json.loads(z.read(nm).decode("utf-8")):
                        row["_shard"] = Path(nm).name
                        out[row["sid"]] = row
    else:
        for p in sorted(src.rglob("shard_*.json")):
            for row in json.loads(p.read_text(encoding="utf-8")):
                row["_shard"] = p.name
                out[row["sid"]] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", type=Path, required=True, help="Path to replies/ folder or replies .zip")
    ap.add_argument("--factory-dir", type=Path, default=Path("v12-gold-refinery-datafactory"))
    args = ap.parse_args()

    fdir = args.factory_dir.resolve()
    vault = json.loads((fdir / "private_gold_v2_canary_vault.json").read_text(encoding="utf-8"))
    canary_map = {c["sid"]: c for c in vault["canaries"]}
    sid_map = {}
    with (fdir / "private_gold_v2_sid_map.jsonl").open("r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                sid_map[r["sid"]] = r

    # Recover exact text from pack zip
    sid_to_text = {}
    with zipfile.ZipFile(fdir / "v12_gold_relabel_and_boundary_39shard_pack.zip") as z:
        for nm in z.namelist():
            if nm.startswith("shards/shard_") and nm.endswith(".json"):
                for it in json.loads(z.read(nm).decode("utf-8")):
                    sid_to_text[it["sid"]] = it["text"]

    replies = load_replies(args.replies.resolve())
    missing = sorted(set(sid_to_text) - set(replies))
    schema_errs = []
    for sid, r in replies.items():
        if not (0 <= float(r.get("ai_score", -1)) <= 100): schema_errs.append(f"bad_ai:{sid}")
        if not (0 <= float(r.get("promo_score", -1)) <= 100): schema_errs.append(f"bad_pr:{sid}")
        if r.get("promo_type") not in VALID_PT: schema_errs.append(f"bad_pt:{sid}")
        if r.get("origin_6way") not in VALID_O6: schema_errs.append(f"bad_o6:{sid}")

    # 50/50 Canary Audit
    ai_ok = pr_ok = pt_ok = 0
    canary_fails = []
    for sid, c in sorted(canary_map.items()):
        r = replies.get(sid)
        if not r:
            canary_fails.append({"sid": sid, "reason": "missing"})
            continue
        a_ok = (r["ai_score"] >= 65) if c["exp_ai"] == "high" else (r["ai_score"] <= 35)
        p_ok = (r["promo_score"] >= 65) if c["exp_promo"] == "high" else (r["promo_score"] <= 35)
        t_ok = (r["promo_type"] == c["exp_type"])
        ai_ok += int(a_ok); pr_ok += int(p_ok); pt_ok += int(t_ok)
        if not (a_ok and p_ok):
            canary_fails.append({"sid": sid, "kind": c["canary_kind"], "exp": (c["exp_ai"], c["exp_promo"]), "got": (r["ai_score"], r["promo_score"], r["promo_type"])})

    # Split cleanly into train / val_gold_v2 / test_bili_gold_v2 (strict split isolation!)
    train_out, val_out, test_out = [], [], []
    drift_counts = Counter()
    for sid, meta in sorted(sid_map.items()):
        r = replies.get(sid)
        if not r:
            continue
        osc = float(meta["orig_score"])
        nsc = float(r["ai_score"])
        if osc >= 70 and nsc <= 30: drift_counts["oldAI_to_newHU"] += 1
        elif osc <= 30 and nsc >= 70: drift_counts["oldHU_to_newAI"] += 1
        elif 30 < osc < 70 and (nsc <= 25 or nsc >= 75): drift_counts["oldMID_resolved"] += 1

        rec = {
            "sid": sid,
            "sample_id": meta["orig_sample_id"],
            "split": meta["orig_split"],
            "bucket": meta["bucket"],
            "orig_source": meta["orig_source"],
            "orig_score": osc,
            "score_source": "v12_gold_v2_pro",
            "dsh": bool(meta.get("dsh", False)),
            "text": sid_to_text[sid],
            "score": int(round(nsc)),
            "ai_score": int(round(nsc)),
            "promo_score": int(round(float(r["promo_score"]))),
            "promo_type": str(r["promo_type"]),
            "origin_6way": str(r["origin_6way"]),
            "is_hybrid_mixed": bool(r.get("is_hybrid_mixed", False)),
            "has_quoted_ai": bool(r.get("has_quoted_ai", False)),
            "confidence": float(r.get("confidence", 0.9)),
            "brief_cue": str(r.get("brief_cue", "")),
        }
        if meta["bucket"] == "A_hydrogen_validation":
            val_out.append(rec)
        elif meta["bucket"] == "A_hydrogen_test":
            test_out.append(rec)
        else:
            train_out.append(rec)

    for fn, rows in (("v12_gold_v2_train.jsonl.gz", train_out), ("v12_gold_v2_val.jsonl.gz", val_out), ("v12_gold_v2_test.jsonl.gz", test_out)):
        with gzip.open(fdir / fn, "wt", encoding="utf-8") as wf:
            for rec in rows:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rep = {
        "total_replies": len(replies),
        "missing": len(missing),
        "schema_errors": len(schema_errs),
        "canary_ai": f"{ai_ok}/{len(canary_map)}",
        "canary_promo": f"{pr_ok}/{len(canary_map)}",
        "canary_subtype": f"{pt_ok}/{len(canary_map)}",
        "canary_pass": (ai_ok >= 47 and pr_ok >= 47 and len(missing) == 0 and len(schema_errs) == 0),
        "canary_fails": canary_fails[:15],
        "drift_vs_gold_v1": dict(drift_counts),
        "written": {"train": len(train_out), "val_gold_v2": len(val_out), "test_bili_gold_v2": len(test_out)},
    }
    (fdir / "v12_gold_v2_qc_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep["canary_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
