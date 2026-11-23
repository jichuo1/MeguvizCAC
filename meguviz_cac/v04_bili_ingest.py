"""Multi-partition x multi-era 1M real Bilibili comment ingest, deduplication & progress manager."""

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bili_round01_data import body_key  # noqa: E402
from bili_round01_generate import CONTACT  # noqa: E402


DEFAULT_DB = REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3"
DEFAULT_LEGACY_DB = REPO_ROOT / "bili-live-corpus.sqlite3"
DEFAULT_PLAN = Path(__file__).resolve().parent / "v04_bili_plan.json"
DEFAULT_STATUS_OUT = REPO_ROOT / "v04-corpus" / "bili-live-1m-status.json"
TZ_CN = timezone(timedelta(hours=8))
YEAR_PREFIX_RE = re.compile(r"\b(20\d{2})[-/年]")


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS comments (
            sample_id TEXT PRIMARY KEY,
            body_hash TEXT UNIQUE NOT NULL,
            video_id TEXT NOT NULL,
            partition TEXT NOT NULL,
            era_bucket TEXT NOT NULL,
            comment_level TEXT NOT NULL,
            posted_ts INTEGER,
            posted_time_display TEXT,
            metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS comments_video ON comments(video_id);
        CREATE INDEX IF NOT EXISTS comments_partition ON comments(partition);
        CREATE INDEX IF NOT EXISTS comments_era ON comments(era_bucket);
        CREATE INDEX IF NOT EXISTS comments_body_hash ON comments(body_hash);

        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            partition TEXT NOT NULL,
            era_bucket TEXT NOT NULL,
            title TEXT,
            pubdate INTEGER,
            fetched_comments INTEGER NOT NULL,
            top_end_reached INTEGER NOT NULL,
            updated_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingested_files (
            file_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            rows_added INTEGER NOT NULL
        );
        """
    )
    return db


def hash_body(text: str) -> str:
    return hashlib.sha256(body_key(text).encode("utf-8")).hexdigest()


def year_to_era(year: int) -> str:
    if year <= 2022:
        return "pre_llm"
    if year <= 2024:
        return "early_llm"
    if year == 2025:
        return "mid_llm"
    return "current_2026"


def derive_era_bucket(posted_ts=None, posted_time_display: str = "", video_pubdate=None, capture_date: str = "") -> str:
    if isinstance(posted_ts, (int, float)) and posted_ts > 1_000_000_000:
        ts = posted_ts / 1000.0 if posted_ts > 1e11 else float(posted_ts)
        yr = datetime.fromtimestamp(ts, tz=TZ_CN).year
        return year_to_era(yr)
    disp = (posted_time_display or "").strip()
    if disp:
        m = YEAR_PREFIX_RE.search(disp)
        if m:
            return year_to_era(int(m[1]))
        if re.search(r"刚刚|秒前|分钟前|小时前|昨天|前天|^\d{1,2}-\d{1,2}", disp):
            cap_yr = int(capture_date[:4]) if re.match(r"^20\d{2}", capture_date or "") else 2026
            return year_to_era(cap_yr)
    if isinstance(video_pubdate, (int, float)) and video_pubdate > 1_000_000_000:
        ts = video_pubdate / 1000.0 if video_pubdate > 1e11 else float(video_pubdate)
        yr = datetime.fromtimestamp(ts, tz=TZ_CN).year
        return year_to_era(yr)
    return "current_2026"


def load_legacy_hashes(legacy_db: Path = DEFAULT_LEGACY_DB) -> tuple[int, set[str]]:
    if not legacy_db.is_file():
        return 0, set()
    hashes: set[str] = set()
    count = 0
    with closing(sqlite3.connect(legacy_db)) as con:
        for (txt,) in con.execute("SELECT json_extract(metadata, '$.text') FROM comments"):
            count += 1
            if txt:
                hashes.add(hash_body(txt))
    return count, hashes


def iter_jsonl_lines(path: Path):
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def sanitize_record(raw: dict, fallback_meta: dict) -> dict | None:
    text = " ".join(str(raw.get("text") or "").split())
    if not text or CONTACT.search(text):
        return None
    bvid = str(raw.get("bvid") or raw.get("video_id") or fallback_meta.get("bvid") or "").strip()
    if not re.fullmatch(r"BV[A-Za-z0-9]{10}", bvid):
        return None
    partition = str(raw.get("partition") or raw.get("sampled_from_channel") or fallback_meta.get("partition") or "未分类").strip() or "未分类"
    level = str(raw.get("comment_level") or "top").strip()
    if level not in ("top", "reply"):
        level = "top"
    posted_ts = raw.get("posted_ts") or raw.get("ctime")
    if isinstance(posted_ts, str) and posted_ts.isdigit():
        posted_ts = int(posted_ts)
    if not isinstance(posted_ts, int):
        posted_ts = None
    disp = str(raw.get("posted_time_display") or raw.get("displayed_time") or "").strip()[:60]
    pubdate = raw.get("pubdate") or fallback_meta.get("pubdate")
    if isinstance(pubdate, str) and pubdate.isdigit():
        pubdate = int(pubdate)
    if not isinstance(pubdate, int):
        pubdate = None
    cap_date = str(raw.get("capture_date") or fallback_meta.get("capture_date") or datetime.now(TZ_CN).date().isoformat())[:16]
    era = str(raw.get("era_bucket") or "").strip()
    if era not in {"pre_llm", "early_llm", "mid_llm", "current_2026"}:
        era = derive_era_bucket(posted_ts, disp, pubdate, cap_date)

    bhash = hash_body(text)
    sid = "V4L" + hashlib.sha256(f"{bvid}\0{level}\0{bhash}".encode("utf-8")).hexdigest()[:22]
    parent_id = str(raw.get("parent_id") or "").strip()[:60] or None
    title = str(raw.get("video_title") or raw.get("title") or fallback_meta.get("title") or "").strip()[:300]
    safe_meta = {
        "sample_id": sid,
        "video_id": bvid,
        "partition": partition,
        "era_bucket": era,
        "comment_level": level,
        "parent_id": parent_id,
        "text": text,
        "posted_ts": posted_ts,
        "posted_time_display": disp,
        "pubdate": pubdate,
        "like_count": int(raw["like_count"]) if isinstance(raw.get("like_count"), int) else None,
        "sort": str(raw.get("sort") or fallback_meta.get("sort") or "api_mixed")[:32],
        "capture_date": cap_date,
        "is_pinned": bool(raw.get("pinned") or raw.get("is_pinned")),
        "video_title": title,
    }
    return {
        "sample_id": sid,
        "body_hash": bhash,
        "video_id": bvid,
        "partition": partition,
        "era_bucket": era,
        "comment_level": level,
        "posted_ts": posted_ts,
        "posted_time_display": disp,
        "metadata_json": json.dumps(safe_meta, ensure_ascii=False, separators=(",", ":")),
        "title": title,
        "pubdate": pubdate,
    }


def parse_pubdate_ts(val) -> int | None:
    if isinstance(val, (int, float)) and val > 1_000_000_000:
        return int(val / 1000.0 if val > 1e11 else val)
    if isinstance(val, str):
        m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", val)
        if m:
            yr, mo, dy = int(m[1]), int(m[2]), int(m[3])
            hr, mn, sc = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
            return int(datetime(yr, mo, dy, hr, mn, sc, tzinfo=TZ_CN).timestamp())
    return None


def strip_ip_suffix(disp: str) -> str:
    return re.split(r"IP\s*[属屬]地", str(disp or ""))[0].strip()


def _ingest_files(
    db: sqlite3.Connection,
    files: list[Path],
    extract_fn,
    legacy_db: Path = DEFAULT_LEGACY_DB,
    preloaded_legacy_hashes: set[str] | None = None,
) -> dict:
    pending: list[tuple[Path, float, int]] = []
    now_epoch = time.time()
    for path in files:
        if not path.is_file():
            continue
        st = path.stat()
        if now_epoch - st.st_mtime < 1.5:
            continue
        prev = db.execute("SELECT mtime, size FROM ingested_files WHERE file_path=?", (str(path.resolve()),)).fetchone()
        if prev and prev[0] == st.st_mtime and prev[1] == st.st_size:
            continue
        pending.append((path, st.st_mtime, st.st_size))

    if not pending:
        return {
            "files_processed": 0,
            "submitted": 0,
            "added": 0,
            "duplicates": 0,
            "filtered": 0,
            **status_report(db, legacy_db=legacy_db),
        }

    if preloaded_legacy_hashes is None:
        _, legacy_hashes = load_legacy_hashes(legacy_db)
    else:
        legacy_hashes = preloaded_legacy_hashes
    existing_hashes = {r[0] for r in db.execute("SELECT body_hash FROM comments")}
    seen_hashes = legacy_hashes | existing_hashes

    files_processed = 0
    total_submitted = 0
    total_added = 0
    total_dups = 0
    total_filtered = 0
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for path, f_mtime, f_size in pending:

        file_added = 0
        video_stats: dict[str, dict] = {}
        with db:
            for item_result in extract_fn(path):
                if item_result is None:
                    total_filtered += 1
                    continue
                raw, fallback = item_result
                total_submitted += 1
                rec = sanitize_record(raw, fallback) if isinstance(raw, dict) else None
                if rec is None:
                    total_filtered += 1
                    continue
                if rec["body_hash"] in seen_hashes:
                    total_dups += 1
                    continue
                seen_hashes.add(rec["body_hash"])
                rc = db.execute(
                    "INSERT OR IGNORE INTO comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec["sample_id"],
                        rec["body_hash"],
                        rec["video_id"],
                        rec["partition"],
                        rec["era_bucket"],
                        rec["comment_level"],
                        rec["posted_ts"],
                        rec["posted_time_display"],
                        rec["metadata_json"],
                    ),
                ).rowcount
                if rc:
                    total_added += 1
                    file_added += 1
                    vs = video_stats.setdefault(
                        rec["video_id"],
                        {"partition": rec["partition"], "era": rec["era_bucket"], "title": rec["title"], "pubdate": rec["pubdate"], "n": 0},
                    )
                    vs["n"] += 1
                else:
                    total_dups += 1

            for bvid, vs in video_stats.items():
                db.execute(
                    """
                    INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(video_id) DO UPDATE SET
                        fetched_comments = videos.fetched_comments + excluded.fetched_comments,
                        updated_utc = excluded.updated_utc
                    """,
                    (bvid, vs["partition"], vs["era"], vs["title"], vs["pubdate"], vs["n"], now_utc),
                )
            db.execute(
                "INSERT INTO ingested_files VALUES (?, ?, ?, ?) ON CONFLICT(file_path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size, rows_added=excluded.rows_added",
                (str(path.resolve()), f_mtime, f_size, file_added),
            )
        files_processed += 1

    return {
        "files_processed": files_processed,
        "submitted": total_submitted,
        "added": total_added,
        "duplicates": total_dups,
        "filtered": total_filtered,
        **status_report(db, legacy_db=legacy_db),
    }


def ingest_dir(
    db: sqlite3.Connection,
    scan_dir: Path,
    legacy_db: Path = DEFAULT_LEGACY_DB,
    pattern: str = "bili_v04_*.jsonl*",
    preloaded_legacy_hashes: set[str] | None = None,
) -> dict:
    def _extract_jsonl(path: Path):
        for item in iter_jsonl_lines(path):
            if isinstance(item, dict) and isinstance(item.get("rows"), list):
                fallback = {
                    "bvid": item.get("bvid"),
                    "partition": item.get("partition") or item.get("sampled_from_channel"),
                    "title": item.get("title"),
                    "pubdate": item.get("pubdate"),
                    "capture_date": item.get("capture_date"),
                    "sort": item.get("sort"),
                }
                for raw in item["rows"]:
                    yield raw, fallback
            elif isinstance(item, dict):
                yield item, {}
            else:
                yield None

    files = sorted(p for p in scan_dir.glob(pattern) if p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
    return _ingest_files(db, files, _extract_jsonl, legacy_db=legacy_db, preloaded_legacy_hashes=preloaded_legacy_hashes)


def ingest_browser_steps(
    db: sqlite3.Connection,
    steps_dir: Path,
    legacy_db: Path = DEFAULT_LEGACY_DB,
    preloaded_legacy_hashes: set[str] | None = None,
) -> dict:
    def _extract_step(path: Path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        val = payload.get("value")
        batches = []
        if isinstance(val, dict):
            if val.get("kind") == "bili_v04_dom_batch":
                batches = [val]
            elif isinstance(val.get("batches"), list):
                batches = [b for b in val["batches"] if isinstance(b, dict) and b.get("kind") == "bili_v04_dom_batch"]
        for b in batches:
            pub_ts = parse_pubdate_ts(b.get("video_pubdate"))
            fallback = {
                "bvid": b.get("bvid"),
                "partition": b.get("partition"),
                "title": b.get("title"),
                "pubdate": pub_ts,
                "capture_date": b.get("capture_date") or "",
                "sort": "dom_scroll",
            }
            top_texts = b.get("top_texts") or []
            top_times = b.get("top_times") or []
            for i, txt in enumerate(top_texts):
                disp = strip_ip_suffix(top_times[i] if i < len(top_times) else "")
                yield {"text": txt, "displayed_time": disp, "comment_level": "top"}, fallback
            reply_texts = b.get("reply_texts") or []
            reply_times = b.get("reply_times") or []
            for i, txt in enumerate(reply_texts):
                disp = strip_ip_suffix(reply_times[i] if i < len(reply_times) else "")
                yield {"text": txt, "displayed_time": disp, "comment_level": "reply"}, fallback

    return _ingest_files(
        db,
        sorted(steps_dir.glob("step_*.json")),
        _extract_step,
        legacy_db=legacy_db,
        preloaded_legacy_hashes=preloaded_legacy_hashes,
    )


def watch_loop(
    db_path: Path,
    watch_dirs: list[Path],
    watch_step_dirs: list[Path],
    legacy_db: Path,
    status_out: Path,
    interval: float = 25.0,
    once: bool = False,
) -> None:
    _, legacy_hashes = load_legacy_hashes(legacy_db)
    status_out.parent.mkdir(parents=True, exist_ok=True)
    cycle = 0
    while True:
        cycle += 1
        added = 0
        files_processed = 0
        with closing(connect(db_path)) as db:
            for d in watch_dirs:
                if d.is_dir():
                    res = ingest_dir(db, d, legacy_db=legacy_db, preloaded_legacy_hashes=legacy_hashes)
                    added += res["added"]
                    files_processed += res["files_processed"]
            for sd in watch_step_dirs:
                if sd.is_dir():
                    res = ingest_browser_steps(db, sd, legacy_db=legacy_db, preloaded_legacy_hashes=legacy_hashes)
                    added += res["added"]
                    files_processed += res["files_processed"]
            st = status_report(db, legacy_db=legacy_db)

        snap = {
            **st,
            "watch_cycle": cycle,
            "last_scan_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_cycle_added": added,
            "last_cycle_files_processed": files_processed,
        }
        pending_status = status_out.with_suffix(".writing")
        pending_status.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending_status.replace(status_out)
        if added > 0 or cycle == 1:
            print(
                json.dumps(
                    {
                        "watch_cycle": cycle,
                        "added": added,
                        "files": files_processed,
                        "combined_total_rows": st["combined_total_rows"],
                        "new_v04_rows": st["new_v04_rows"],
                        "by_partition": st["by_partition"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if once:
            return
        time.sleep(max(2.0, interval))


def status_report(db: sqlite3.Connection, legacy_db: Path = DEFAULT_LEGACY_DB, plan_path: Path = DEFAULT_PLAN) -> dict:
    legacy_rows = 0
    if legacy_db.is_file():
        with closing(sqlite3.connect(legacy_db)) as con:
            legacy_rows = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    new_rows = db.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    combined = legacy_rows + new_rows
    target = 1_000_000
    if plan_path.is_file():
        target = int(json.loads(plan_path.read_text(encoding="utf-8")).get("combined_target", 1_000_000))
    return {
        "combined_target": target,
        "existing_legacy_rows": legacy_rows,
        "new_v04_rows": new_rows,
        "combined_total_rows": combined,
        "remaining_to_target": max(0, target - combined),
        "distinct_v04_videos": db.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        "by_partition": dict(db.execute("SELECT partition, COUNT(*) FROM comments GROUP BY partition ORDER BY COUNT(*) DESC")),
        "by_era_bucket": dict(db.execute("SELECT era_bucket, COUNT(*) FROM comments GROUP BY era_bucket ORDER BY COUNT(*) DESC")),
        "by_comment_level": dict(db.execute("SELECT comment_level, COUNT(*) FROM comments GROUP BY comment_level")),
    }


def export_merged(db: sqlite3.Connection, out_path: Path, legacy_db: Path = DEFAULT_LEGACY_DB) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = out_path.with_suffix(".writing")
    count = 0
    with gzip.open(staging, "wt", encoding="utf-8", newline="\n", compresslevel=5) as f:
        if legacy_db.is_file():
            with closing(sqlite3.connect(legacy_db)) as con:
                for (meta_s,) in con.execute("SELECT metadata FROM comments ORDER BY video_id, sample_id"):
                    r = json.loads(meta_s)
                    clean = {
                        "sample_id": r["sample_id"],
                        "video_id": r.get("video_id"),
                        "partition": r.get("sampled_from_channel") or r.get("partition") or "未分类",
                        "era_bucket": derive_era_bucket(None, r.get("posted_time_display") or "", None, r.get("capture_date") or ""),
                        "comment_level": r.get("comment_level", "top"),
                        "parent_id": r.get("parent_id"),
                        "text": r["text"],
                        "posted_time_display": r.get("posted_time_display"),
                        "source_cohort": "legacy_200k",
                    }
                    f.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
        for (meta_s,) in db.execute("SELECT metadata_json FROM comments ORDER BY partition, video_id, sample_id"):
            r = json.loads(meta_s)
            r["source_cohort"] = "v04_1m"
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    staging.replace(out_path)
    return {"exported_rows": count, "path": str(out_path), "bytes": out_path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--legacy-db", type=Path, default=DEFAULT_LEGACY_DB)
    parser.add_argument("--import-dir", type=Path)
    parser.add_argument("--import-browser-steps", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--watch-dir", type=Path, action="append", default=[])
    parser.add_argument("--watch-browser-steps", type=Path, action="append", default=[])
    parser.add_argument("--interval", type=float, default=25.0)
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    parser.add_argument("--watch-once", action="store_true")
    args = parser.parse_args()

    oneshot = sum(
        bool(x) for x in (args.import_dir is not None, args.import_browser_steps is not None, args.status, args.export is not None)
    )
    has_watch = bool(args.watch_dir or args.watch_browser_steps)
    if oneshot and has_watch:
        parser.error("Cannot mix watcher flags with one-shot actions.")
    if not oneshot and not has_watch:
        parser.error("Specify one action (--import-dir/--import-browser-steps/--status/--export) or --watch-dir/--watch-browser-steps.")
    if oneshot > 1:
        parser.error("Only one one-shot action can be specified at a time.")

    if has_watch:
        watch_loop(
            args.db,
            args.watch_dir,
            args.watch_browser_steps,
            args.legacy_db,
            args.status_out,
            interval=args.interval,
            once=args.watch_once,
        )
        return

    with closing(connect(args.db)) as db:
        if args.import_dir is not None:
            print(json.dumps(ingest_dir(db, args.import_dir, legacy_db=args.legacy_db), ensure_ascii=False, indent=2))
        elif args.import_browser_steps is not None:
            print(json.dumps(ingest_browser_steps(db, args.import_browser_steps, legacy_db=args.legacy_db), ensure_ascii=False, indent=2))
        elif args.status:
            print(json.dumps(status_report(db, legacy_db=args.legacy_db), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(export_merged(db, args.export, legacy_db=args.legacy_db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
