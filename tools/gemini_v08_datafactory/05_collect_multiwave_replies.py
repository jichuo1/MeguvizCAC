"""Step 5: One-Click Multi-Wave External Gemini Reply Collector, Per-Wave Canary Auditor & Unified Merger.

Ingests ANY number of returned external Gemini reply ZIPs or folders (from parallel multi-wave runs!),
optionally fuses prior completed releases (e.g., Wave-1 `v08-datafactory-r1/release`), audits hidden
canaries BOTH per-wave AND globally, strips 100% of canaries, verifies zero eval/trap leakage, and
emits unified drop-in `v08-gemini-gold-labels.jsonl.gz` + `v08-gemini-counterfactual-pairs.jsonl`!
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gemini_v08_datafactory.rubric_v08 import score_to_class5

_exp = importlib.import_module("tools.gemini_v08_datafactory.01_export_queue")
normalize_text = _exp.normalize_text

VALID_QUEUES = {"A_noise_surgery", "B_wild_hard_mining", "C_counterfactual_spans", "D_thread_context"}


def _extract_reply_source(src: Path, work_dir: Path, pro_manifests: list[dict]) -> int:
    """Extract reply shards (`<queue>/<batch_id>.json`) and optional Pro adjudication JSONL from a ZIP or directory."""
    extracted = 0
    if src.is_file() and src.suffix.lower() == ".zip":
        with ZipFile(src, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                fname = parts[-1]
                if fname.endswith("adjudication_manifest.jsonl"):
                    for ln in zf.read(info).decode("utf-8", errors="replace").splitlines():
                        if ln.strip():
                            pro_manifests.append(json.loads(ln))
                    continue
                if fname.endswith((".json", ".txt")) and not fname.startswith("."):
                    qmatch = next((p for p in parts[:-1] if p in VALID_QUEUES), None)
                    if qmatch and fname.startswith(qmatch):
                        dst = work_dir / "offline_pack_tier1" / qmatch / "replies" / fname
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(info))
                        extracted += 1
    elif src.is_dir():
        for fp in sorted(src.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() == ".zip":
                extracted += _extract_reply_source(fp, work_dir, pro_manifests)
            elif fp.name.endswith("adjudication_manifest.jsonl"):
                for ln in fp.read_text(encoding="utf-8", errors="replace").splitlines():
                    if ln.strip():
                        pro_manifests.append(json.loads(ln))
            elif fp.suffix.lower() in (".json", ".txt"):
                qmatch = next((p for p in fp.parts[:-1] if p in VALID_QUEUES), None)
                if qmatch and fp.name.startswith(qmatch):
                    dst = work_dir / "offline_pack_tier1" / qmatch / "replies" / fp.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(fp.read_bytes())
                    extracted += 1
    return extracted


def _register_pro_manifest(work_dir: Path, pro_rows: list[dict]) -> int:
    if not pro_rows:
        return 0
    key_to_sid: dict[tuple[str, int, str], str] = {}
    t1_by_sid: dict[str, dict] = {}
    for qname in ("A_noise_surgery", "B_wild_hard_mining", "D_thread_context"):
        bf = work_dir / "queues" / f"{qname}.batches.jsonl"
        if bf.is_file():
            with bf.open("r", encoding="utf-8") as f:
                for ln in f:
                    b = json.loads(ln)
                    for r in b["rows"]:
                        key_to_sid[(b["batch_id"], int(r["idx"]), str(r["tag"]))] = str(r["sample_id"])
        tf = work_dir / "outputs_tier1" / f"{qname}.tier1.jsonl"
        if tf.is_file():
            with tf.open("r", encoding="utf-8") as f:
                for ln in f:
                    rec = json.loads(ln)
                    for sid, pred in (rec.get("parsed") or {}).items():
                        t1_by_sid[sid] = pred

    by_q: dict[str, dict[str, dict]] = {}
    mapped = 0
    for pr in pro_rows:
        sid = key_to_sid.get((str(pr["batch_id"]), int(pr["idx"]), str(pr["tag"])))
        if not sid:
            continue
        sc = int(pr["tier2_score"])
        cd = str(pr["tier2_code"])
        by_q.setdefault(str(pr["queue"]), {})[sid] = {
            "score": sc,
            "code": cd,
            "confidence": int(pr.get("tier2_confidence", 85)),
            "is_trap_human": bool(pr.get("is_trap_human", False)),
            "prag_8d": (t1_by_sid.get(sid) or {}).get("prag_8d", [0] * 8),
            "class_5way": score_to_class5(sc, cd),
            "rationale": str(pr.get("rationale", "")),
        }
        mapped += 1

    out_t2 = work_dir / "outputs_tier2"
    out_t2.mkdir(parents=True, exist_ok=True)
    for qname, pmap in by_q.items():
        rec = {"batch_id": f"T2_MULTIWAVE_PRO_{qname}", "queue": qname, "tier": "tier2", "model": "gemini-external-tier2-pro", "parsed": pmap}
        (out_t2 / f"{qname}.tier2.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    return mapped


def _merge_prior_releases(work_dir: Path, prior_dirs: list[Path]) -> dict:
    """Optionally fuse earlier clean releases (e.g. Wave-1) with current multi-wave release (deduped by normalized text)."""
    rel_dir = work_dir / "release"
    gold_p = rel_dir / "v08-gemini-gold-labels.jsonl.gz"
    cf_p = rel_dir / "v08-gemini-counterfactual-pairs.jsonl"
    if not prior_dirs:
        return {"merged_prior_gold": 0, "merged_prior_pairs": 0}

    seen_gold: set[str] = set()
    gold_lines: list[str] = []
    if gold_p.is_file():
        with gzip.open(gold_p, "rt", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    obj = json.loads(ln)
                    n = normalize_text(obj.get("text", ""))
                    if n and n not in seen_gold:
                        seen_gold.add(n)
                        gold_lines.append(ln.strip())

    added_gold = 0
    for pdir in prior_dirs:
        pg = pdir / "v08-gemini-gold-labels.jsonl.gz"
        if pg.is_file():
            with gzip.open(pg, "rt", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        obj = json.loads(ln)
                        n = normalize_text(obj.get("text", ""))
                        if n and n not in seen_gold:
                            seen_gold.add(n)
                            gold_lines.append(ln.strip())
                            added_gold += 1

    with gzip.open(gold_p, "wt", encoding="utf-8", newline="\n") as gf:
        for ln in gold_lines:
            gf.write(ln + "\n")

    seen_cf: set[tuple[str, str]] = set()
    cf_lines: list[str] = []
    if cf_p.is_file():
        for ln in cf_p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                obj = json.loads(ln)
                k = (normalize_text(obj.get("ai_text", "")), normalize_text(obj.get("human_text", "")))
                if k not in seen_cf:
                    seen_cf.add(k)
                    cf_lines.append(ln.strip())

    added_cf = 0
    for pdir in prior_dirs:
        pcf = pdir / "v08-gemini-counterfactual-pairs.jsonl"
        if pcf.is_file():
            for ln in pcf.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    obj = json.loads(ln)
                    k = (normalize_text(obj.get("ai_text", "")), normalize_text(obj.get("human_text", "")))
                    if k not in seen_cf:
                        seen_cf.add(k)
                        cf_lines.append(ln.strip())
                        added_cf += 1

    cf_p.write_text("\n".join(cf_lines) + ("\n" if cf_lines else ""), encoding="utf-8")
    return {
        "total_unified_gold": len(gold_lines),
        "merged_prior_gold": added_gold,
        "total_unified_pairs": len(cf_lines),
        "merged_prior_pairs": added_cf,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True, help="Multi-wave workspace directory (e.g. v08-datafactory-200k)")
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="One or more returned Wave ZIPs or reply folders")
    ap.add_argument("--include-prior-release", type=Path, action="append", default=[], help="Optional prior release dirs (e.g. v08-datafactory-r1/release) to fuse")
    ap.add_argument("--allow-warn", action="store_true", help="Pass through to 03_qc_and_merge.py")
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    pro_manifests: list[dict] = []
    total_extracted = 0
    for inp in args.inputs:
        total_extracted += _extract_reply_source(inp.resolve(), work_dir, pro_manifests)
    print(f"[1/4] Extracted {total_extracted:,} reply shards + {len(pro_manifests):,} Pro adjudication rows into {work_dir}")

    print("[2/4] Importing all reply shards via 02_run_gemini_factory.py --mode pack --import-pack ...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "gemini_v08_datafactory" / "02_run_gemini_factory.py"), "--work-dir", str(work_dir), "--mode", "pack", "--tier", "tier1", "--model", "gemini-external-multiwave", "--import-pack"],
        check=True,
    )

    n_pro = _register_pro_manifest(work_dir, pro_manifests)
    if n_pro:
        print(f"      Registered {n_pro:,} Tier-2 Pro adjudicated rows (weight=1.35)")

    print("[3/4] Running Hidden Canary QC + Anti-Leakage Strip via 03_qc_and_merge.py ...")
    qc_cmd = [sys.executable, str(REPO_ROOT / "tools" / "gemini_v08_datafactory" / "03_qc_and_merge.py"), "--work-dir", str(work_dir)]
    if args.allow_warn:
        qc_cmd.append("--allow-warn")
    subprocess.run(qc_cmd, check=True)

    merge_info = _merge_prior_releases(work_dir, [p.resolve() for p in args.include_prior_release])
    print("[4/4] Unified Merge Summary:", json.dumps(merge_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
