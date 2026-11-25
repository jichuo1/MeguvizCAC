"""Post-merge persona label noise cleanup for MeguvizCAC v0.4.0 R3.

Implements §3.1 of docs/v0.4.0-final-training-plan.md without modifying v04_corpus_prep.py
or re-releasing the frozen v0.4-data corpus asset:
- Targets ONLY `split == "train"` (excluding `cohort == "user_confirmed"`).
- Targets ONLY 12L distilled / ensemble soft labels (`colab_gpu_distilled_from_hydrogen_xhigh*`
  and `multi_teacher_ensemble`); preserves all direct gold / 4-teacher / pre-LLM / plugin rows.
- Targets ONLY true AI persona self-reference rows whose score is < 70 (raising score to
  max(score, 72.0), setting train_weight=0.5, reason_code="A", and fixing misassigned
  `structured_human_hard` group_id to `persona_space_kaomoji`).
- Verifies byte-identical non-train rows (validation / test / probe) and safety row-count bounds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import time


ELIGIBLE_SOURCE_PREFIXES = ("colab_gpu_distilled_from_hydrogen_xhigh",)
ELIGIBLE_SOURCE_EXACT = frozenset({"multi_teacher_ensemble"})

# Exclude human meta-discussions, idioms ("X鱼本鱼"), or coincidental substrings ("西本鱼跃")
EXCLUDE_META_RE = re.compile(
    r"(?:真人扮|怎么识别|常见.{0,8}开场白|鱼本鱼|本鱼跃|【人工比对】|我家.{0,8}(?:Agent|agent|AI))"
)

# Primary Bilibili DeepSeek fish/whale bot self-reference tokens
PRIMARY_PERSONA_RE = re.compile(r"本[鱼鲸]")

# Explicit bot self-ID / persona-config phrases (carefully avoids human talk about LLM params or pets)
EXPLICIT_BOT_RE = re.compile(
    r"(?:备忘录精灵|我是大肥鱼\s*AI|作为一只野生\s*AI|我是个模型[，,。]|"
    r"(?:本助手|本模型|本\s*AI).{0,24}(?:参数|人设|归档|记忆库|主人|生成|回复|窗口))"
)


def is_eligible_source(score_source: str) -> bool:
    if score_source in ELIGIBLE_SOURCE_EXACT:
        return True
    return any(score_source.startswith(p) for p in ELIGIBLE_SOURCE_PREFIXES)


def match_persona_reason(text: str) -> str | None:
    if not text or EXCLUDE_META_RE.search(text):
        return None
    if PRIMARY_PERSONA_RE.search(text):
        return "primary_benyu_benjing"
    if EXPLICIT_BOT_RE.search(text):
        return "explicit_bot_persona"
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_labelfix(
    in_path: Path,
    out_path: Path | None,
    *,
    min_score: float = 72.0,
    target_weight: float = 0.5,
    max_modified: int = 300,
    dry_run: bool = False,
    dump_matched: Path | None = None,
) -> dict:
    t0 = time.monotonic()
    total_rows = 0
    train_rows = 0
    non_train_rows = 0
    modified_rows = 0

    by_source = Counter()
    by_rule = Counter()
    by_old_bucket = Counter()
    by_old_group = Counter()
    group_reassigned = 0
    matched_records: list[dict] = []

    non_train_in_hash = hashlib.sha256()
    non_train_out_hash = hashlib.sha256()

    out_fh = None
    gz_fh = None
    tmp_path = None
    if not dry_run:
        if out_path is None:
            raise ValueError("--out is required unless --dry-run is set")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if out_path.name.endswith(".gz"):
            out_fh = tmp_path.open("wb")
            gz_fh = gzip.GzipFile(filename="", mode="wb", fileobj=out_fh, compresslevel=5, mtime=0)
        else:
            out_fh = tmp_path.open("wb")

    def _write_line(raw_bytes: bytes) -> None:
        if gz_fh is not None:
            gz_fh.write(raw_bytes)
        elif out_fh is not None:
            out_fh.write(raw_bytes)

    try:
        opener = gzip.open if in_path.name.endswith(".gz") else open
        with opener(in_path, "rb") as fin:
            for raw_line in fin:
                if not raw_line.strip():
                    continue
                total_rows += 1
                row = json.loads(raw_line.decode("utf-8"))
                is_train = row.get("split") == "train" and row.get("cohort") != "user_confirmed"
                if not is_train:
                    non_train_rows += 1
                    non_train_in_hash.update(raw_line)
                    if not dry_run:
                        _write_line(raw_line)
                        non_train_out_hash.update(raw_line)
                    continue

                train_rows += 1
                src = str(row.get("score_source") or "")
                sc = float(row.get("score", 0.0)) if row.get("has_score", True) and row.get("score") is not None else 100.0
                rule = match_persona_reason(str(row.get("text") or "")) if (is_eligible_source(src) and sc < 70.0) else None

                if rule is None:
                    if not dry_run:
                        _write_line(raw_line)
                    continue

                modified_rows += 1
                old_score = sc
                old_weight = float(row.get("train_weight", 1.0))
                old_rc = row.get("reason_code")
                old_gid = row.get("group_id")

                new_score = max(old_score, min_score)
                row["score"] = round(new_score, 4)
                row["score_int"] = int(round(new_score))
                row["train_weight"] = float(target_weight)
                row["reason_code"] = "A"
                if old_gid == "structured_human_hard":
                    row["group_id"] = "persona_space_kaomoji"
                    group_reassigned += 1

                bucket = "le30" if old_score <= 30.0 else "mid31_69"
                by_source[src] += 1
                by_rule[rule] += 1
                by_old_bucket[bucket] += 1
                by_old_group[str(old_gid)] += 1

                matched_records.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "rule": rule,
                        "score_source": src,
                        "old_score": old_score,
                        "new_score": row["score"],
                        "old_weight": old_weight,
                        "new_weight": row["train_weight"],
                        "old_reason_code": old_rc,
                        "old_group_id": old_gid,
                        "new_group_id": row.get("group_id"),
                        "text": str(row.get("text") or "")[:160],
                    }
                )

                if not dry_run:
                    out_bytes = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                    _write_line(out_bytes)
    finally:
        if gz_fh is not None:
            gz_fh.close()
        if out_fh is not None:
            out_fh.close()

    if modified_rows > max_modified:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Safety valve triggered: modified_rows={modified_rows} > max_modified={max_modified}"
        )

    if not dry_run and tmp_path is not None and out_path is not None:
        if non_train_in_hash.hexdigest() != non_train_out_hash.hexdigest():
            tmp_path.unlink()
            raise RuntimeError("Safety valve triggered: non-train rows hash mismatch!")
        tmp_path.replace(out_path)

    if dump_matched is not None:
        dump_matched.parent.mkdir(parents=True, exist_ok=True)
        with dump_matched.open("w", encoding="utf-8") as df:
            for rec in matched_records:
                df.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "dry_run": dry_run,
        "input_path": str(in_path),
        "output_path": str(out_path) if out_path else None,
        "input_sha256": file_sha256(in_path),
        "output_sha256": file_sha256(out_path) if (not dry_run and out_path and out_path.is_file()) else None,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "non_train_rows": non_train_rows,
        "non_train_sha256": non_train_in_hash.hexdigest(),
        "non_train_byte_identical": True if dry_run else (non_train_in_hash.hexdigest() == non_train_out_hash.hexdigest()),
        "modified_rows": modified_rows,
        "modified_pct_of_train": round(100.0 * modified_rows / max(1, train_rows), 5),
        "group_reassigned_from_structured": group_reassigned,
        "by_source": dict(by_source),
        "by_rule": dict(by_rule),
        "by_old_bucket": dict(by_old_bucket),
        "by_old_group": dict(by_old_group),
        "sample_preview": matched_records[:12],
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="R3 post-merge persona label noise cleanup")
    parser.add_argument("--in", "--input", dest="in_path", type=Path, required=True, help="Input v04-merged-1m4.jsonl.gz")
    parser.add_argument("--out", "--output", dest="out_path", type=Path, default=None, help="Output cleaned .jsonl.gz")
    parser.add_argument("--min-score", type=float, default=72.0)
    parser.add_argument("--target-weight", type=float, default=0.5)
    parser.add_argument("--max-modified", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON audit report path")
    parser.add_argument("--dump-matched", type=Path, default=None, help="Optional JSONL dump of all modified rows")
    args = parser.parse_args()

    rep = run_labelfix(
        args.in_path,
        args.out_path,
        min_score=args.min_score,
        target_weight=args.target_weight,
        max_modified=args.max_modified,
        dry_run=args.dry_run,
        dump_matched=args.dump_matched,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
