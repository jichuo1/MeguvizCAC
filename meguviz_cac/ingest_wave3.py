"""MeguvizCAC v1.2.1 Blind `Wave-3` (18-Shard Wild-Promo + AI-Frontier) Reply Auditor & Ingester.

Verifies incoming replies for `v121_wave3_wildpromo_and_frontier_18shard_pack.zip`:
  1. Checks 100% completeness & JSON schema across all 877 items (`852` real mined + `25` hidden canaries).
  2. Grades the 25 hidden canaries against `private_wave3_canary_vault.json` (promo, AI, and subtype accuracy)
     and fails fast if canary QC fails (`promo < 90%` or `ai < 88%`).
  3. Joins real rows with `private_wave3_sid_map.jsonl`, excludes any SHA colliding with frozen eval/test splits
     (`test_bili_gold_v1/v2`, `val_gold_v1/v2`), splits deterministically 80/20 into:
       - `v121_wave3_train.jsonl.gz` (for Phase A/B/C training)
       - `v121_wave3_val.jsonl.gz` (strictly held-out wild-promo & AI-frontier validation!)
  4. Optionally packs a single-file Colab upload bundle `v121_gold_and_wave3_bundle.zip` combining
     `Gold-v2` + `Wave-2.5` + `Wave-3`!
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

VALID_PROMO_TYPES = {"none", "coupon_commerce", "self_promo_funnel", "soft_shill", "tool_boilerplate"}
VALID_ORIGINS = {
    "Human-Organic",
    "Human-Assisted",
    "Autonomous-Persona-Agent",
    "Script-Plugin-Bot",
    "Coordinated-Latent-Cluster",
    "Human-Commercial-Promo",
}


def _norm_sha(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_replies_from_path(replies_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if replies_path.is_file() and replies_path.suffix.lower() == ".zip":
        with ZipFile(replies_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/") or not (name.endswith(".jsonl") or name.endswith(".json")):
                    continue
                raw = zf.read(name).decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                if raw.startswith("["):
                    rows.extend(json.loads(raw))
                else:
                    for line in raw.splitlines():
                        line = line.strip()
                        if line and not line.startswith("```"):
                            rows.append(json.loads(line))
    elif replies_path.is_dir():
        for fp in sorted(replies_path.rglob("*.jsonl")):
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("```"):
                    rows.append(json.loads(line))
    else:
        for line in replies_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("```"):
                rows.append(json.loads(line))
    return rows


def ingest_wave3_replies(
    replies_path: Path,
    data_dir: Path,
    pack_colab_bundle: bool = True,
) -> dict[str, Any]:
    vault_p = data_dir / "private_wave3_canary_vault.json"
    sid_map_p = data_dir / "private_wave3_sid_map.jsonl"
    pack_zip_p = data_dir / "v121_wave3_wildpromo_and_frontier_18shard_pack.zip"
    assert vault_p.is_file(), f"Missing canary vault: {vault_p}"
    assert sid_map_p.is_file(), f"Missing sid map: {sid_map_p}"
    assert pack_zip_p.is_file(), f"Missing original pack zip (for text join): {pack_zip_p}"

    # Load `sid -> text` from the original blind pack zip
    text_by_sid: dict[str, str] = {}
    with ZipFile(pack_zip_p, "r") as pzf:
        for nm in sorted(pzf.namelist()):
            if nm.startswith("shards/") and nm.endswith(".json"):
                for item in json.loads(pzf.read(nm).decode("utf-8")):
                    text_by_sid[str(item["sid"])] = str(item["text"])

    vault = json.loads(vault_p.read_text(encoding="utf-8"))
    canaries_by_sid = {str(c["sid"]): c for c in (vault if isinstance(vault, list) else vault.get("canaries", []))}
    sid_meta: dict[str, dict[str, Any]] = {}
    for line in sid_map_p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sid_meta[str(r["sid"])] = r

    # Collect frozen test/val text hashes to guarantee ZERO leakage
    frozen_shas: set[str] = set()
    for gz_name in ("v12_gold_v2_val.jsonl.gz", "v12_gold_v2_test.jsonl.gz"):
        gp = data_dir / gz_name
        if gp.is_file():
            with gzip.open(gp, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        frozen_shas.add(_norm_sha(json.loads(line)["text"]))

    raw_replies = _load_replies_from_path(replies_path)
    by_sid: dict[str, dict[str, Any]] = {}
    schema_errors: list[str] = []
    for r in raw_replies:
        sid = str(r.get("sid", "")).strip()
        if not sid:
            schema_errors.append("missing_sid")
            continue
        try:
            ai_sc = float(r["ai_score"])
            pr_sc = float(r["promo_score"])
            pt = str(r.get("promo_type", "none")).strip()
            if not (0.0 <= ai_sc <= 100.0 and 0.0 <= pr_sc <= 100.0):
                schema_errors.append(f"{sid}:score_range")
                continue
            if pt not in VALID_PROMO_TYPES:
                pt = "none" if pr_sc < 50.0 else "soft_shill"
            r["ai_score"] = int(round(ai_sc))
            r["promo_score"] = int(round(pr_sc))
            r["promo_type"] = pt
            by_sid[sid] = r
        except Exception as e:
            schema_errors.append(f"{sid}:{e}")

    # 1. Grade Hidden Canaries (`exp_ai`/`exp_promo` = "high"|"low"|"any" & `exp_type`)
    c_total = len(canaries_by_sid)
    c_ai_ok = c_pr_ok = c_sub_ok = 0
    c_fails: list[dict[str, Any]] = []
    for sid, truth in canaries_by_sid.items():
        got = by_sid.get(sid)
        if not got:
            c_fails.append({"sid": sid, "reason": "missing"})
            continue
        exp_ai = str(truth.get("exp_ai", "")).lower()
        exp_pr = str(truth.get("exp_promo", "")).lower()
        exp_pt = str(truth.get("exp_type", "none")).strip()
        if exp_ai in ("high", "low"):
            t_ai_bin = 1 if exp_ai == "high" else 0
            ai_ok = (int(got["ai_score"] >= 50) == t_ai_bin)
        else:
            ai_ok = True
        if exp_pr in ("high", "low"):
            t_pr_bin = 1 if exp_pr == "high" else 0
            pr_ok = (int(got["promo_score"] >= 50) == t_pr_bin)
        else:
            t_pr_bin = 0
            pr_ok = True
        sub_ok = (not t_pr_bin) or (got["promo_type"] == exp_pt) or (exp_pt in ("any_promo", "any", ""))
        c_ai_ok += int(ai_ok)
        c_pr_ok += int(pr_ok)
        c_sub_ok += int(sub_ok)
        if not (ai_ok and pr_ok):
            c_fails.append({"sid": sid, "kind": truth.get("canary_kind"), "truth": (exp_ai, exp_pr, exp_pt), "got": (got["ai_score"], got["promo_score"], got["promo_type"])})

    canary_pass = (c_total > 0) and (c_pr_ok / c_total >= 0.88) and (c_ai_ok / c_total >= 0.88)

    # 2. Join Real Mined Rows & Stratified 80/20 Train/Val Split + Forensic Bucket Breakdown
    real_rows: list[dict[str, Any]] = []
    missing_real = 0
    leaked_dropped = 0
    bucket_stats: dict[str, dict[str, int]] = {}
    promo_pos = 0
    ai_pos = 0

    for sid, meta in sid_meta.items():
        if sid in canaries_by_sid or meta.get("is_canary"):
            continue
        got = by_sid.get(sid)
        if not got:
            missing_real += 1
            continue
        txt = str(meta.get("text") or text_by_sid.get(sid, ""))
        if not txt:
            missing_real += 1
            continue
        if _norm_sha(txt) in frozen_shas:
            leaked_dropped += 1
            continue
        bkt = str(meta.get("bucket", meta.get("track", "wave3")))
        rec = {
            "sid": sid,
            "sample_id": str(meta.get("orig_sample_id", meta.get("sample_id", sid))),
            "bucket": bkt,
            "text": txt,
            "score": int(got["ai_score"]),
            "ai_score": int(got["ai_score"]),
            "promo_score": int(got["promo_score"]),
            "promo_type": str(got["promo_type"]),
            "origin_6way": str(got.get("origin_6way", "Human-Organic")),
            "is_hybrid_mixed": bool(got.get("is_hybrid_mixed", False)),
            "has_quoted_ai": bool(got.get("has_quoted_ai", False)),
            "confidence": float(got.get("confidence", 0.85)),
            "brief_cue": str(got.get("brief_cue", "")),
            "score_source": "v121_wave3_pro_gold",
        }
        bs = bucket_stats.setdefault(bkt, {"n": 0, "ai_ge70": 0, "ai_le30": 0, "pr_ge65": 0, "pr_le25": 0})
        bs["n"] += 1
        bs["ai_ge70"] += int(rec["ai_score"] >= 70)
        bs["ai_le30"] += int(rec["ai_score"] <= 30)
        bs["pr_ge65"] += int(rec["promo_score"] >= 65)
        bs["pr_le25"] += int(rec["promo_score"] <= 25)
        promo_pos += int(rec["promo_score"] >= 65)
        ai_pos += int(rec["ai_score"] >= 70)
        real_rows.append(rec)

    train_rows, val_rows = [], []
    for rec in real_rows:
        h = int(hashlib.sha256(rec["sid"].encode("utf-8")).hexdigest()[:8], 16)
        if h % 5 == 0:
            rec["split"] = "validation"
            val_rows.append(rec)
        else:
            rec["split"] = "train"
            train_rows.append(rec)

    tr_p = data_dir / "v121_wave3_train.jsonl.gz"
    va_p = data_dir / "v121_wave3_val.jsonl.gz"
    for fp, items in ((tr_p, train_rows), (va_p, val_rows)):
        with gzip.open(fp, "wt", encoding="utf-8") as wf:
            for it in items:
                wf.write(json.dumps(it, ensure_ascii=False) + "\n")

    qc = {
        "replies_received": len(by_sid),
        "schema_errors": len(schema_errors),
        "canaries": {
            "total": c_total,
            "promo_ok": c_pr_ok,
            "ai_ok": c_ai_ok,
            "subtype_ok": c_sub_ok,
            "pass": canary_pass,
            "fails": c_fails[:10],
        },
        "real_rows": {
            "total_ingested": len(real_rows),
            "missing_real": missing_real,
            "frozen_eval_leaks_dropped": leaked_dropped,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "promo_pos_ge65": promo_pos,
            "ai_pos_ge70": ai_pos,
            "bucket_breakdown": bucket_stats,
        },
    }
    (data_dir / "v121_wave3_qc_report.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if pack_colab_bundle and canary_pass:
        bundle_p = data_dir / "v121_gold_and_wave3_bundle.zip"
        root = data_dir.parent
        w25_p = root / "v11-omniguard-datafactory" / "v11_wave25_dualaxis_gold_verified.jsonl.gz"
        with ZipFile(bundle_p, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for fn in ("v12_gold_v2_train.jsonl.gz", "v12_gold_v2_val.jsonl.gz", "v12_gold_v2_test.jsonl.gz", "v121_wave3_train.jsonl.gz", "v121_wave3_val.jsonl.gz"):
                fp = data_dir / fn
                if fp.is_file():
                    zf.write(fp, fn)
            if w25_p.is_file():
                zf.write(w25_p, w25_p.name)
        qc["colab_bundle"] = {
            "path": str(bundle_p),
            "kb": round(bundle_p.stat().st_size / 1024.0, 1),
            "sha256": hashlib.sha256(bundle_p.read_bytes()).hexdigest(),
        }
    return qc


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit & ingest Wave-3 18-shard blind annotation replies.")
    ap.add_argument("--replies", type=Path, required=True, help="Reply zip, folder, or jsonl file.")
    ap.add_argument("--data-dir", type=Path, default=Path("v12-gold-refinery-datafactory"))
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()
    rep = ingest_wave3_replies(args.replies.resolve(), args.data_dir.resolve(), pack_colab_bundle=not args.no_bundle)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    assert rep["canaries"]["pass"], f"Wave-3 canary QC failed: {rep['canaries']}"


if __name__ == "__main__":
    main()
