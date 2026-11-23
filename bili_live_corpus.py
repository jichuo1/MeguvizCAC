"""Checkpoint and export Bilibili comments read through the in-app browser."""

import argparse
import hashlib
import html
import json
import os
import re
import secrets
import sqlite3
import unicodedata
from contextlib import closing
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parent
VIDEO_PATH = re.compile(r"/(?:zh-Hans/)?video/(BV[A-Za-z0-9]{10})/?")
CONTACT = re.compile(r"[@＠]|https?://|\bwww\.|\b1[3-9][0-9]{9}\b|手机号|身份证|家庭住址|加我微信|加QQ群", re.I)
MAX_FORM_BYTES = 16 * 1024 * 1024
NICKNAME_HINTS = {"含鱼", "大肥鱼类", "含白饭"}
NICKNAME_RULES = {"大肥鱼", "白饭", "大肥鱼和白饭"}


def connect(path):
    db = sqlite3.connect(path, timeout=15)
    db.execute("PRAGMA journal_mode = WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS comments (
            sample_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            comment_level TEXT NOT NULL,
            nickname_hint TEXT,
            metadata TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS comments_video ON comments(video_id);
        CREATE TABLE IF NOT EXISTS video_progress (
            video_id TEXT NOT NULL,
            sort TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            partition TEXT,
            top_end_reached INTEGER NOT NULL,
            updated_utc TEXT NOT NULL,
            PRIMARY KEY (video_id, sort)
        );
        CREATE TABLE IF NOT EXISTS reply_progress (
            parent_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            last_page INTEGER NOT NULL,
            complete INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS reply_progress_video ON reply_progress(video_id);
        CREATE TABLE IF NOT EXISTS reply_video_progress (
            video_id TEXT PRIMARY KEY,
            sort TEXT NOT NULL,
            complete INTEGER NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS suggestions (
            sample_id TEXT PRIMARY KEY REFERENCES comments(sample_id),
            suggested_label TEXT NOT NULL,
            annotation_source TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS capture_journals (
            journal_path TEXT PRIMARY KEY,
            byte_offset INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS capture_attempts (
            journal_path TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            video_id TEXT NOT NULL,
            capture_date TEXT NOT NULL,
            queue_round INTEGER,
            outcome TEXT NOT NULL,
            rendered INTEGER,
            error TEXT,
            PRIMARY KEY (journal_path, byte_offset)
        );
        CREATE INDEX IF NOT EXISTS capture_attempts_video ON capture_attempts(video_id);
    """)
    return db


def text(value, limit):
    return value.strip()[:limit] if isinstance(value, str) else ""


def body_text(value):
    body = " ".join(text(value, 40000).split())
    return body if body and not CONTACT.search(body) else ""


def sample_id(video, level, parent, body):
    normalized = " ".join(unicodedata.normalize("NFKC", body).casefold().split())
    digest = hashlib.sha256(f"{video}\0{level}\0{parent}\0{normalized}".encode()).hexdigest()
    return f"L{digest[:24]}"


def corpus_status(db):
    return {
        "rows": db.execute("SELECT COUNT(*) FROM comments").fetchone()[0],
        "video_counts": dict(db.execute("SELECT video_id, COUNT(*) FROM comments GROUP BY video_id")),
        "nickname_hints": dict(db.execute("SELECT nickname_hint, COUNT(*) FROM comments WHERE nickname_hint IS NOT NULL GROUP BY nickname_hint")),
        "comment_levels": dict(db.execute("SELECT comment_level, COUNT(*) FROM comments GROUP BY comment_level")),
        "reply_threads_complete": db.execute("SELECT COUNT(*) FROM reply_progress WHERE complete = 1").fetchone()[0],
        "reply_end_reached_videos": [r[0] for r in db.execute("SELECT video_id FROM reply_video_progress WHERE complete=1 ORDER BY video_id")],
        "suggestions": dict(db.execute("SELECT suggested_label, COUNT(*) FROM suggestions GROUP BY suggested_label")),
        "suggestion_sources": dict(db.execute("SELECT annotation_source, COUNT(*) FROM suggestions GROUP BY annotation_source")),
        "top_end_reached_videos": [r[0] for r in db.execute("SELECT DISTINCT video_id FROM video_progress WHERE top_end_reached=1 ORDER BY video_id")],
    }


def ingest(db, batch):
    if not isinstance(batch, dict) or not isinstance(batch.get("rows"), list):
        raise ValueError("批次须包含视频信息与 rows 列表。")
    video = text(batch.get("bvid"), 18)
    url = urlsplit(text(batch.get("url"), 400))
    route = VIDEO_PATH.fullmatch(url.path)
    if url.scheme != "https" or url.netloc != "www.bilibili.com" or not route or route[1] != video:
        raise ValueError("视频链接与 BV 号不一致。")
    page_url = f"{url.scheme}://{url.netloc}{url.path.rstrip('/')}/"
    title = text(batch.get("title"), 500)
    tags = batch.get("tags")
    tags = list(dict.fromkeys(text(tag, 80) for tag in tags if text(tag, 80)))[:20] if isinstance(tags, list) else []
    partition = text(batch.get("partition"), 100) or None
    channel = text(batch.get("sampled_from_channel"), 40) or None
    channel_url = text(batch.get("channel_source_url"), 200) or None
    if channel_url and not re.fullmatch(r"https://www\.bilibili\.com/c/[a-z]+/", channel_url):
        raise ValueError("采样分区链接不一致。")
    order = batch.get("sort")
    if order not in ("latest", "popular", "page_default"):
        raise ValueError("评论排序未确认。")
    capture_date = date.fromisoformat(batch["capture_date"]).isoformat()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    accepted = rejected = added = 0
    with db:
        for item in batch["rows"]:
            if not isinstance(item, dict):
                rejected += 1
                continue
            body = body_text(item.get("text"))
            stamp = text(item.get("displayed_time"), 60)
            if not body or not stamp:
                rejected += 1
                continue
            checked = item.get("checked_name") is True
            hint = text(item.get("name_hint"), 20) if checked else ""
            hint = hint if hint in NICKNAME_HINTS else None
            rule = text(item.get("name_rule"), 20) if checked else ""
            rule = rule if rule in NICKNAME_RULES else None
            level = item.get("comment_level", "top")
            if level not in ("top", "reply"):
                rejected += 1
                continue
            parent = text(item.get("parent_id"), 60) if level == "reply" else ""
            parent_body = body_text(item.get("parent_text")) if level == "reply" else ""
            if parent_body:
                parent = sample_id(video, "top", "", parent_body)
            if level == "reply" and not re.fullmatch(r"L[0-9a-f]{24}", parent):
                rejected += 1
                continue
            if level == "reply" and not db.execute("SELECT 1 FROM comments WHERE sample_id = ? AND video_id = ? AND comment_level = 'top'", (parent, video)).fetchone():
                rejected += 1
                continue
            uid = sample_id(video, level, parent, body)
            row = dict(sample_id=uid, video_id=video, group_id=f"bili:{video}", video_url=page_url,
                       video_title=title, topic_tags=tags, partition=partition, text=body, label="unknown",
                       posted_time_display=stamp, capture_date=capture_date, comment_level=level,
                       parent_id=parent or None, sort=order, source_kind="bili_browser_live",
                       nickname_checked=checked, nickname_hint=hint, nickname_rule=rule,
                       sampled_from_channel=channel, channel_source_url=channel_url,
                       is_pinned=item.get("pinned") is True)
            accepted += 1
            added += db.execute("INSERT OR IGNORE INTO comments VALUES (?, ?, ?, ?, ?)",
                                (uid, video, level, hint, json.dumps(row, ensure_ascii=False))).rowcount
            if rule:
                db.execute("""UPDATE comments SET nickname_hint = ?, metadata = json_set(metadata, '$.nickname_hint', ?,
                           '$.nickname_checked', json('true'), '$.nickname_rule', ?)
                           WHERE sample_id = ? AND json_extract(metadata, '$.nickname_rule') IS NULL""",
                           (hint, hint, rule, uid))
        for progress in batch.get("reply_pages", []):
            parent_body = body_text(progress.get("parent_text"))
            page = progress.get("page")
            if not parent_body or type(page) is not int or page < 1:
                raise ValueError("无效的回复页进度。")
            db.execute("""INSERT INTO reply_progress VALUES (?, ?, ?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET last_page = max(last_page, excluded.last_page),
                complete = max(complete, excluded.complete)""",
                (sample_id(video, "top", "", parent_body), video, page, int(progress.get("complete") is True)))
        db.execute("""INSERT INTO video_progress VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id,sort) DO UPDATE SET
            top_end_reached=max(top_end_reached, excluded.top_end_reached),
            updated_utc=excluded.updated_utc, partition=coalesce(excluded.partition, video_progress.partition)""",
            (video, order, page_url, title, partition, int(batch.get("ended") is True), now))
        if batch.get("replies_ended") is True and batch.get("ended") is True:
            db.execute("INSERT INTO reply_video_progress VALUES (?, ?, 1, ?) ON CONFLICT(video_id) DO UPDATE SET sort=excluded.sort, updated_utc=excluded.updated_utc",
                       (video, order, now))
    return {"bvid": video, "submitted": len(batch["rows"]), "accepted": accepted,
            "added": added, "duplicates": accepted - added, "filtered": rejected,
            "video_rows": db.execute("SELECT COUNT(*) FROM comments WHERE video_id = ?", (video,)).fetchone()[0]}


def read_queue(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def ingest_suggestions(db, rows):
    if not isinstance(rows, list):
        raise ValueError("标注回执须包含 rows 列表。")
    counts = {"stored": 0, "unchanged": 0, "filtered": 0}
    with db:
        for item in rows:
            if not isinstance(item, list) or len(item) != 3 or not all(isinstance(value, str) for value in item):
                counts["filtered"] += 1
                continue
            uid, label, source = item
            comment = db.execute("""SELECT json_extract(metadata, '$.nickname_rule'),
                json_extract(metadata, '$.style_rule') FROM comments WHERE sample_id = ?""", (uid,)).fetchone()
            basis, _, rule = source.partition(":")
            rule_column = {"nickname_rule": 0, "style_rule": 1}.get(basis)
            if (not comment or label not in {"ai_like", "human_like", "uncertain"} or
                (rule_column is not None and (rule != comment[rule_column] or label != "ai_like")) or
                (rule_column is None and source not in {"google/gemini-3.8-flash", "short_text"})):
                counts["filtered"] += 1
                continue
            changed = db.execute("""INSERT INTO suggestions VALUES (?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET suggested_label=excluded.suggested_label,
                annotation_source=excluded.annotation_source
                WHERE (suggestions.suggested_label!=excluded.suggested_label OR suggestions.annotation_source!=excluded.annotation_source)
                AND (CASE WHEN excluded.annotation_source LIKE 'nickname_rule:%' THEN 2
                          WHEN excluded.annotation_source LIKE 'style_rule:%' THEN 1 ELSE 0 END) >=
                    (CASE WHEN suggestions.annotation_source LIKE 'nickname_rule:%' THEN 2
                          WHEN suggestions.annotation_source LIKE 'style_rule:%' THEN 1 ELSE 0 END)""",
                (uid, label, source)).rowcount
            counts["stored" if changed else "unchanged"] += 1
    return {"submitted": len(rows), **counts, "total": db.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0]}


def import_journal(db, journal):
    path = str(journal.resolve())
    cursor = db.execute("SELECT byte_offset FROM capture_journals WHERE journal_path = ?", (path,)).fetchone()
    counts = {"batches": 0, "attempts": 0, "added": 0, "duplicates": 0, "filtered": 0}
    with journal.open("rb") as stream:
        offset = cursor[0] if cursor else 0
        if journal.stat().st_size < offset:
            raise ValueError("采集日志已截断，无法沿用上次位置。")
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event.get("kind") == "comments":
                receipt = ingest(db, event["batch"])
                counts["batches"] += 1
                for key in ("added", "duplicates", "filtered"):
                    counts[key] += receipt[key]
            elif event.get("kind") == "attempt":
                attempt = event["attempt"]
                bvid = text(attempt.get("bvid"), 18)
                outcome = text(attempt.get("status"), 60)
                capture_date = date.fromisoformat(attempt["capture_date"]).isoformat()
                if not re.fullmatch(r"BV[A-Za-z0-9]{10}", bvid) or not outcome:
                    raise ValueError("无效的视频采集回执。")
                with db:
                    counts["attempts"] += db.execute(
                        "INSERT OR IGNORE INTO capture_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (path, offset, bvid, capture_date, attempt.get("queue_round"), outcome,
                         attempt.get("rendered"), text(attempt.get("error"), 200) or None)
                    ).rowcount
            else:
                raise ValueError("无法识别采集日志事件。")
            offset += len(line)
            with db:
                db.execute("""INSERT INTO capture_journals VALUES (?, ?) ON CONFLICT(journal_path)
                    DO UPDATE SET byte_offset=excluded.byte_offset""", (path, offset))
    return {**counts, "rows": db.execute("SELECT COUNT(*) FROM comments").fetchone()[0]}


def reply_checkpoints(db):
    result = {}
    for bvid, parent, page, complete in db.execute("""SELECT p.video_id, json_extract(c.metadata, '$.text'),
            p.last_page, p.complete FROM reply_progress p JOIN comments c ON c.sample_id=p.parent_id"""):
        result.setdefault(bvid, []).append([parent, page, complete])
    return result


def sync_queue(db, path, source):
    queue = read_queue(path)
    videos = queue["videos"]
    known = {video["bvid"] for video in videos}
    added = 0
    if source:
        for item in read_queue(source)["videos"]:
            bvid = item["bvid"]
            url = urlsplit(item["url"])
            route = VIDEO_PATH.fullmatch(url.path)
            if url.scheme != "https" or url.netloc != "www.bilibili.com" or not route or route[1] != bvid or url.query or url.fragment:
                raise ValueError("新队列中存在未核验的 B 站视频链接。")
            if bvid not in known:
                videos.append(item)
                known.add(bvid)
                added += 1

    counts = {bvid: {"top": top, "reply": reply} for bvid, top, reply in db.execute("""SELECT video_id,
        SUM(comment_level='top'), SUM(comment_level='reply') FROM comments GROUP BY video_id""")}
    status = corpus_status(db)
    top_done = set(status["top_end_reached_videos"])
    replies_done = set(status["reply_end_reached_videos"])
    latest = {bvid: (day, outcome) for bvid, day, outcome in db.execute(
        "SELECT video_id,capture_date,outcome FROM capture_attempts ORDER BY rowid"
    )}
    for item in videos:
        bvid = item["bvid"]
        if bvid in latest:
            day, outcome = latest[bvid]
            item["last_attempt"] = {"on": day, "surface": "in_app_browser", "signal": outcome}
        if bvid in counts:
            item["comment_levels"] = counts[bvid]
            item["live_count"] = sum(counts[bvid].values())
            item["status"] = ("top_and_reply_end_observed" if bvid in replies_done else
                              "top_feed_end_observed" if bvid in top_done else "sampling_incomplete")

    rounds = {}
    for number in sorted({video["queue_round"] for video in videos if "queue_round" in video}):
        group = [video for video in videos if video.get("queue_round") == number]
        rounds[str(number)] = {
            "verified": len(group),
            "visited": sum(bool(video.get("last_attempt")) for video in group),
            "with_comments": sum(video["bvid"] in counts for video in group),
            "observed_top_end": sum(video["bvid"] in top_done for video in group),
            "observed_reply_end": sum(video["bvid"] in replies_done for video in group),
        }
    queue["live_comments_collected"] = status["rows"]
    queue["pending_video_count"] = sum(video["status"] == "pending" for video in videos)
    run = queue.setdefault("last_run", {})
    run.update(rounds=rounds, new_suggestions_in_local_db=sum(status["suggestions"].values()), suggestions=status["suggestions"])
    if latest:
        run["date"] = max(day for day, _ in latest.values())
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.resolve().parent, delete=False) as stage:
        json.dump(queue, stage, ensure_ascii=False, indent=2)
        stage.write("\n")
    os.replace(stage.name, path)
    return {"added_videos": added, "rows": status["rows"], "rounds": rounds}


def serve(path, port):
    nonce = secrets.token_hex(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, receipt=None, status=200):
            data = html.escape(json.dumps(receipt, ensure_ascii=False)) if receipt is not None else ""
            page = ("<!doctype html><meta charset=utf-8><title>B 站语料存档</title>"
                    "<h1>B 站语料存档</h1><form action=/ingest method=post>"
                    f"<input type=hidden name=nonce value={nonce}>"
                    "<label>评论批次<textarea name=payload rows=5 cols=45 required></textarea></label>"
                    "<button type=submit>保存批次</button></form><pre id=receipt>" + data + "</pre>")
            payload = page.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; form-action 'self'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def trusted_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.trusted_host():
                self.send_error(404)
                return
            if self.path in ("/", "/ingest"):
                self.reply()
                return
            if self.path == "/queue":
                queue = read_queue(ROOT / "bili-ai-comment-queue.json")
                with closing(connect(path)) as db:
                    status = corpus_status(db)
                    self.reply({"videos": queue["videos"], "status": status})
                return
            match = re.fullmatch(r"/progress/(BV[A-Za-z0-9]{10})", self.path)
            if match:
                with closing(connect(path)) as db:
                    pages = db.execute("""SELECT json_extract(c.metadata, '$.text'), p.last_page, p.complete
                        FROM reply_progress p JOIN comments c ON c.sample_id=p.parent_id WHERE p.video_id=?""", (match[1],))
                    self.reply({"reply_pages": list(pages)})
                return
            self.send_error(404)

        def do_POST(self):
            if not self.trusted_host() or self.path != "/ingest":
                self.send_error(404)
                return
            try:
                amount = int(self.headers["Content-Length"])
                if not 0 < amount <= MAX_FORM_BYTES or self.headers.get_content_type() != "application/x-www-form-urlencoded":
                    raise ValueError("无效的批次体积或表单格式。")
                form = parse_qs(self.rfile.read(amount).decode("utf-8"), strict_parsing=True, max_num_fields=2)
                if form.get("nonce") != [nonce] or len(form.get("payload", [])) != 1:
                    self.send_error(403)
                    return
                batch = json.loads(form["payload"][0])
                with closing(connect(path)) as db:
                    result = ingest_suggestions(db, batch.get("rows")) if isinstance(batch, dict) and batch.get("kind") == "suggestions" else ingest(db, batch)
                self.reply(result)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                self.reply({"error": str(exc)}, 400)
            except sqlite3.Error:
                self.reply({"error": "无法提交到本地 SQLite。"}, 500)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "bili-live-corpus.sqlite3")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve").add_argument("--port", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("import-journal").add_argument("journal", type=Path)
    sub.add_parser("import-suggestions").add_argument("suggestions", type=Path)
    sub.add_parser("checkpoints").add_argument("output", type=Path)
    queue = sub.add_parser("sync-queue")
    queue.add_argument("--path", type=Path, default=ROOT / "bili-ai-comment-queue.json")
    queue.add_argument("--append", type=Path)
    export = sub.add_parser("export")
    export.add_argument("output", type=Path)
    export.add_argument("--exclude", nargs="*", type=Path, default=[])
    export.add_argument("--with-suggestions", action="store_true")
    export.add_argument("--unlabeled", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.db, args.port)
        return
    with closing(connect(args.db)) as db:
        if args.command == "status":
            print(json.dumps(corpus_status(db), ensure_ascii=False))
        elif args.command == "import-journal":
            print(json.dumps(import_journal(db, args.journal), ensure_ascii=False))
        elif args.command == "import-suggestions":
            with args.suggestions.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            print(json.dumps(ingest_suggestions(db, rows), ensure_ascii=False))
        elif args.command == "checkpoints":
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(reply_checkpoints(db), stream, ensure_ascii=False)
        elif args.command == "sync-queue":
            print(json.dumps(sync_queue(db, args.path, args.append), ensure_ascii=False))
        else:
            previous = {}
            for path in args.exclude:
                with path.open(encoding="utf-8") as snapshot:
                    for line in snapshot:
                        row = json.loads(line)
                        previous[row["sample_id"]] = (row.get("nickname_rule"), row.get("style_rule"))
            count = 0
            query = "SELECT c.sample_id, c.metadata, s.suggested_label, s.annotation_source FROM comments c LEFT JOIN suggestions s USING(sample_id)"
            if args.unlabeled:
                query += """ WHERE s.sample_id IS NULL OR
                    (json_extract(c.metadata,'$.nickname_rule') IS NOT NULL AND s.annotation_source NOT LIKE 'nickname_rule:%') OR
                    (json_extract(c.metadata,'$.style_rule') IS NOT NULL AND
                     s.annotation_source NOT LIKE 'nickname_rule:%' AND s.annotation_source NOT LIKE 'style_rule:%')"""
            query += " ORDER BY c.video_id, c.sample_id"
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                for uid, payload, label, source in db.execute(query):
                    row = json.loads(payload) if uid in previous or args.with_suggestions else None
                    if uid in previous and previous[uid] == (row.get("nickname_rule"), row.get("style_rule")):
                        continue
                    if args.with_suggestions and label:
                        row.update(suggested_label=label, annotation_source=source)
                        payload = json.dumps(row, ensure_ascii=False)
                    stream.write(payload + "\n")
                    count += 1
            print(json.dumps({"rows": count, "path": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
