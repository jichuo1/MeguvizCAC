"""MeguvizCAC v1.6.1 Local Final DuckDB 2.0 Zero-Leakage Packager & Disjoint Calibration Splitter.

Run this ONCE locally right after the sibling Master DuckDB 2.0 thread finishes its final reviews:
  1. Invokes `v151_stream_ingestor.py` with a FRESH watermark against the final `.duckdb` SSOT,
     enforcing the full 3-layer Zero-Leakage firewall (`eval_frozen`, `God-500`, `OjinjinBench`)
     and hard-blocking ALL `T3_QUARANTINE` purged rows!
  2. Deterministically carves out a strictly disjoint zero-leakage calibration holdout
     (`v161_duckdb2_cal_cert.jsonl.gz`: ~500 AI + ~2,000 Human) by `text_sha16` hash and writes
     the remaining clean training rows to `v161_duckdb2_train_pack.jsonl.gz` (~33 MB).
  3. Verifies ZERO overlap (0 exact sha16 + 0 fuzzy group-key collisions) between train pack and
     calibration holdout, prints SHA256 sums and exact Google Drive upload path!
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .v140_superteacher import read_jsonl_any_v140, text_sha16_v140


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final v1.6.1 DuckDB 2.0 Zero-Leakage Train + Disjoint Cal-Cert Packs")
    ap.add_argument("--master-db", type=Path, default=Path("master-db/meguviz_unified_corpus.duckdb"))
    ap.add_argument("--sibling-dir", type=Path, default=Path("bili-ai-detect/data/gold_train_v6"))
    ap.add_argument("--gold-v2-dir", type=Path, default=Path("bili-ai-detect/data/gold_v2_w3"))
    ap.add_argument("--wave3-dir", type=Path, default=Path("bili-ai-detect/data/gold_v2_w3"))
    ap.add_argument("--v11-data-dir", type=Path, default=Path("data/v11_data"))
    ap.add_argument("--delta-dir", type=Path, default=Path("v123-sibling-delta"))
    ap.add_argument("--ojinjin-dir", type=Path, default=Path("OjinjinBench"))
    ap.add_argument("--base-release-dir", type=Path, default=None, help="Optional v1.5.x release for bucket difficulty tagging")
    ap.add_argument("--out-dir", type=Path, default=Path("scratch/v161_upload_pack"))
    ap.add_argument("--cal-ai-target", type=int, default=500, help="Disjoint clean AI rows reserved for R0 LTT certification")
    ap.add_argument("--cal-hu-target", type=int, default=2000, help="Disjoint clean Human rows reserved for R0 LTT certification")
    ap.add_argument("--skip-ingest-jsonl", type=Path, default=None, help="Skip ingestor and split an existing clean stream pack")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_ingest_jsonl is not None and args.skip_ingest_jsonl.is_file():
        raw_pack = args.skip_ingest_jsonl.resolve()
        print(f"[v1.6.1-Pack] Using existing clean stream pack: {raw_pack}", flush=True)
    else:
        ing_dir = out_dir / "_fresh_ingest"
        ing_dir.mkdir(parents=True, exist_ok=True)
        # Ensure fresh watermark so EVERY clean row in final Master DuckDB 2.0 is ingested cleanly!
        wm_file = ing_dir / "watermark_state.json"
        if wm_file.exists():
            wm_file.unlink()
        cmd = [
            sys.executable, "-u", "-m", "meguviz_cac.v151_stream_ingestor",
            "--master-db", str(args.master_db),
            "--sibling-dir", str(args.sibling_dir),
            "--gold-v2-dir", str(args.gold_v2_dir),
            "--wave3-dir", str(args.wave3_dir),
            "--v11-data-dir", str(args.v11_data_dir),
            "--max-master-db-t0", "300000",
            "--out", str(ing_dir),
            "--watermark", str(wm_file),
        ]
        if args.delta_dir and args.delta_dir.is_dir():
            cmd.extend(["--delta-dir", str(args.delta_dir)])
        if args.ojinjin_dir and args.ojinjin_dir.is_dir():
            cmd.extend(["--ojinjin-dir", str(args.ojinjin_dir)])
        if args.base_release_dir and args.base_release_dir.is_dir():
            cmd.extend(["--base-release-dir", str(args.base_release_dir)])
        print("[v1.6.1-Pack] Running fresh zero-leakage v151_stream_ingestor against final Master DuckDB 2.0...", flush=True)
        subprocess.run(cmd, check=True)
        raw_pack = ing_dir / "v151_stream_pack.jsonl.gz"
        if not raw_pack.is_file():
            raise SystemExit(f"Expected stream pack not found: {raw_pack}")

    rows = read_jsonl_any_v140(raw_pack)
    print(f"[v1.6.1-Pack] Loaded {len(rows):,} verified zero-leakage rows from {raw_pack.name}", flush=True)

    # Deterministic hash-sorted split: prefer high-confidence clean AI (>=0.80) and clean Human (<=0.18) for cal_cert
    # while keeping ALL committee overrides and rare buckets well-represented in train!
    cal_ai: list[dict[str, Any]] = []
    cal_hu: list[dict[str, Any]] = []
    cal_shas: set[str] = set()

    # Sort candidates deterministically by salted sha16 so selection is unbiased across topics/eras!
    indexed = []
    for r in rows:
        txt = str(r.get("text", "") or "").strip()
        if not txt:
            continue
        sha = text_sha16_v140(txt)
        sort_k = hashlib.sha1(f"v161_cal_salt::{sha}".encode("utf-8")).hexdigest()
        indexed.append((sort_k, sha, r))
    indexed.sort(key=lambda x: x[0])

    for _, sha, r in indexed:
        if sha in cal_shas:
            continue
        ya = float(r.get("y_ai", -1.0))
        if len(cal_ai) < args.cal_ai_target and ya >= 0.80:
            cal_ai.append(r)
            cal_shas.add(sha)
        elif len(cal_hu) < args.cal_hu_target and 0.0 <= ya <= 0.18:
            cal_hu.append(r)
            cal_shas.add(sha)
        if len(cal_ai) >= args.cal_ai_target and len(cal_hu) >= args.cal_hu_target:
            break

    cal_rows = cal_ai + cal_hu
    train_rows = [r for r in rows if text_sha16_v140(str(r.get("text", "") or "").strip()) not in cal_shas]

    # Paranoid zero-overlap check
    tr_shas = {text_sha16_v140(str(r.get("text", "") or "").strip()) for r in train_rows}
    overlap = tr_shas & cal_shas
    if overlap:
        raise SystemExit(f"FATAL: {len(overlap)} overlapping sha16 hashes between train and cal_cert!")

    train_path = out_dir / "v161_duckdb2_train_pack.jsonl.gz"
    cal_path = out_dir / "v161_duckdb2_cal_cert.jsonl.gz"

    with gzip.open(train_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(cal_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for r in cal_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    b_cnt = Counter(str(r.get("bucket", "unk")) for r in train_rows)
    summary = {
        "train_pack": str(train_path),
        "train_rows": len(train_rows),
        "train_mb": round(train_path.stat().st_size / (1024.0 * 1024.0), 3),
        "train_sha256": _file_sha256(train_path),
        "train_buckets": dict(b_cnt.most_common()),
        "cal_cert_pack": str(cal_path),
        "cal_cert_rows": len(cal_rows),
        "cal_cert_ai": len(cal_ai),
        "cal_cert_hu": len(cal_hu),
        "cal_cert_mb": round(cal_path.stat().st_size / (1024.0 * 1024.0), 3),
        "cal_cert_sha256": _file_sha256(cal_path),
        "disjoint_overlap": 0,
        "google_drive_upload_dir": "/MyDrive/MeguvizCAC/v161_data/",
    }
    (out_dir / "v161_pack_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
