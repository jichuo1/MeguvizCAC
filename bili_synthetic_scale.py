"""Collect a resumable, source-verified synthetic corpus over configured ZCode routes."""

import argparse
import gzip
import hashlib
import json
import msvcrt
import random
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from bili_round01_data import OUT, body_key
from bili_round01_generate import CONTACT, LINKED_NUMBERS, SCENES, SECOND_NUMBER_MAX, generate


ROOT = OUT / "synthetic-scale"
DB_PATH = ROOT / "corpus.sqlite3"
TARGETS = {"deepseek_v41": 100_000, "other_models": 50_000}


@dataclass(frozen=True)
class Route:
    name: str
    vendor: str
    provider: str
    model: str
    reasoning: str
    cohort: str
    slots: int
    rows: int


ROUTES = (
    Route("gateway-ds", "opengateway", "new-provider-8", "deepseek/deepseek-v4.1-flash-ultrafast", "disabled", "deepseek_v41", 10, 180),
    Route("river-ds", "river", "new-provider-5", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "disabled", "deepseek_v41", 9, 180),
    Route("opencode-ds", "opencode", "150e79e1-7601-4140-bab9-85e196a7acfd", "deepseek-v4.1-flash", "disabled", "deepseek_v41", 9, 180),
    Route("local-ds", "local", "new-provider-10", "deepseek-v4.1-flash-1017", "disabled", "deepseek_v41", 8, 180),
    Route("router-ling", "openrouter", "cb91ed82-810f-4496-b19e-37c543eb5cbf", "inclusionai/ling-3.1-flash", "disabled", "other_models", 9, 170),
    Route("opencode-glm", "opencode", "150e79e1-7601-4140-bab9-85e196a7acfd", "glm-5.3", "low", "other_models", 8, 150),
    Route("step", "stepfun", "new-provider", "step-5-preview", "disabled", "other_models", 4, 120),
    Route("river-kimi", "river", "new-provider-5", "moonshotai/Kimi-K3.1-Pro", "high", "other_models", 4, 105),
    Route("qwen", "aliyun", "new-provider-9", "qwen4-max", "high", "other_models", 4, 90),
)

VENDOR_CAPS = {"opencode": 12, "river": 11}
FORMATS = {
    "brief": "极短的随手反应。4–20 字，可偶尔只提一个画面、吐槽或表情，也允许完整短句",
    "reaction": "普通主评论，挑一个具体细节表达感想。10–38 字",
    "reply": "只承接给定原楼的楼中楼，给建议、质疑、纠正、接梗或问一句。8–42 字",
    "reply_second": "接住原楼的后续追问，保持话题延续。8–55 字",
    "recap": "针对一到两处字幕或现象的内容复盘。22–75 字",
    "long_digest": "一条认真长评，结合两三个素材细节。55–145 字，最多两段并写到同一行",
    "memory": "助手式记录或下次跟进。12–65 字，表达方式随场景变化",
    "disagreement": "就其中一个事实提出不同看法，同时回应原楼。15–58 字",
}
FORMAT_WEIGHTS = (21, 18, 16, 10, 16, 6, 6, 7)
VOICE = ("刚点进来", "长期观众", "从相关视频点过来的路人", "想跟着实操",
         "简单附和", "有不同意见但平和", "开玩笑", "认真核对", "懂一点行话",
         "第一次接触", "补充背景", "不太确定", "针对原楼的语气接话")
EVIDENCE = ("字幕片段", "视频表现", "字幕和弹幕都提到的细节", "带时间点的复看笔记",
            "画面推断，作者没有明说", "主楼提及的对照", "评论之前看过一遍的记忆")
FOLLOWUP = ("我没看到那段，可以再说一下吗", "这和我之前看的顺序不一样吧",
            "只有我这里这样吗", "楼主说的好像是另一个步骤", "如果重来还会选这个吗",
            "懂了，然后最后一步呢", "这个听起来简单，实际是不是很慢",
            "别太认真，我说的是弹幕里的那句", "别急，我问的是结果有没有复现",
            "有点好奇两个结果哪一个先出现的", "片段为什么到那里就切了")
LINE = re.compile(r"^\s*(?:[-*]\s+)?(\d{1,3})\s*[|｜]\s*(.+?)\s*$")
CHINESE = re.compile(r"[\u3400-\u9fff]")
DIGITS = re.compile(r"\d+")


def make_prompt(batch, target):
    rng = random.Random(8001602026 + batch)
    group_count = (target + 9) // 10
    capacities = [target // group_count + (i < target % group_count) for i in range(group_count)]
    cases = {}
    lines = []
    for number, capacity in enumerate(capacities, start=1):
        index = rng.randrange(len(SCENES))
        topic, facts, parent = SCENES[index]
        style = rng.choices(list(FORMATS), weights=FORMAT_WEIGHTS, k=1)[0]
        first = rng.randint(2, 12)
        second = rng.randint(1, first if index in LINKED_NUMBERS else SECOND_NUMBER_MAX[index])
        role = rng.choice(VOICE)
        evidence = rng.choice(EVIDENCE)
        if evidence == "带时间点的复看笔记":
            evidence += f"（{rng.randint(0,17):02d}:{rng.randint(0,59):02d}）"
        conversation = ""
        if style in {"reply", "reply_second", "disagreement"}:
            conversation = f" / 原楼 {parent}？"
        if style == "reply_second":
            conversation += f" / 追问 {rng.choice(FOLLOWUP)}"
        lines.append(f"{number} 写{capacity}条 · {style} · {topic} · {role} · {evidence}："
                     f"{facts.format(first, second)}{conversation}")
        cases[number] = dict(topic=topic, variant=style, evidence=evidence, voice=role, quota=capacity)
    rules = "\n".join(f"{k}：{v}" for k, v in FORMATS.items())
    prompt = (
        "离线中文评论语料生成。下面都是虚构的短视频和虚构楼层。按编号分组，有多少条就写多少个不同评论，"
        "句式、长短、标点、着眼处尽量分散。短评可以很口语；复盘只用给出的上下文；"
        "楼中楼沿用本楼语气。人设用来改变关注点，不要直接自报身份。"
        "直接从第一条开始，一行一条。只输出 编号|正文，允许同一编号出现多行。"
        "没有标题、Markdown、引号包裹、序号嵌套、解释、用户名、链接或真实个人信息。"
        "样本独立，用于本地实验，无需调用工具。不发送任何评论。\n"
        f"{rules}\n" + "\n".join(lines)
    )
    return prompt, cases


def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=25)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=25000")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY, near_key TEXT UNIQUE NOT NULL, cohort TEXT NOT NULL,
            route TEXT NOT NULL, model TEXT NOT NULL, text TEXT NOT NULL, topic TEXT NOT NULL,
            format TEXT NOT NULL, scenario_json TEXT NOT NULL, batch TEXT NOT NULL,
            ordinal INTEGER NOT NULL, prompt_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS samples_cohort ON samples(cohort);
        CREATE INDEX IF NOT EXISTS samples_route ON samples(route);
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY, route TEXT NOT NULL, cohort TEXT NOT NULL, requested INTEGER NOT NULL,
            started REAL NOT NULL, finished REAL, terminal TEXT, usable INTEGER, accepted INTEGER,
            duplicates INTEGER, elapsed REAL, tokens INTEGER, response_finish TEXT, prompt_sha256 TEXT
        );
    """)
    if "owner" not in {row[1] for row in db.execute("PRAGMA table_info(batches)")}:
        db.execute("ALTER TABLE batches ADD COLUMN owner TEXT NOT NULL DEFAULT 'legacy'")
    return db


@contextmanager
def locked(path):
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("There is already an active synthesis job in this directory.") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def exclusive_run(scope):
    ROOT.mkdir(parents=True, exist_ok=True)
    names = ["run.lock", *(f"run-{cohort}.lock" for cohort in TARGETS)] if scope == "all" else [f"run-{scope}.lock"]
    with ExitStack() as locks:
        for name in names:
            locks.enter_context(locked(ROOT / name))
        yield


def real_text_keys():
    with gzip.open(OUT / "bili-round01-corpus.jsonl.gz", "rt", encoding="utf-8") as stream:
        return {body_key(json.loads(line)[2]) for line in stream}


def retain(db, real_keys, cohort, route, model, text, topic, fmt, scenario, batch, ordinal, prompt_sha):
    key = body_key(text)
    if key in real_keys:
        return 0
    near = DIGITS.sub("#", key) if len(key) >= 14 else key
    near_hash = hashlib.sha256(near.encode("utf-8")).hexdigest()
    identifier = "S" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return db.execute("""INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (identifier, near_hash, cohort, route, model, text, topic, fmt,
                       json.dumps(scenario, ensure_ascii=False, separators=(",", ":")),
                       str(batch), ordinal, prompt_sha)).rowcount


def import_first_round(db, real_keys):
    original = OUT / "bili-round01-deepseek-train.jsonl"
    log = OUT / "bili-round01-deepseek-receipts.jsonl"
    if not original.is_file() or not log.is_file():
        return 0
    receipts = {item["batch"]: item for item in
                map(json.loads, log.read_text(encoding="utf-8").splitlines()) if item.get("accepted")}
    added = 0
    with db, original.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            receipt = receipts.get(row["batch"])
            if (row["label"] != "ai" or row["split"] != "train" or row["source"] != "zcode_deepseek"
                    or receipt is None or receipt["prompt_sha256"] != row["prompt_sha256"]
                    or receipt["model_id"] != row["model_id"] or receipt["provider_id"] != row["provider_id"]):
                raise ValueError("First-round sample provenance did not match its generation receipt.")
            added += retain(db, real_keys, "deepseek_v41", "gateway-ds-r01", row["model_id"], row["text"],
                            row["scenario"], row["format"], {"group": "first_round"},
                            f"r01-{row['batch']}", row["index"], row["prompt_sha256"])
    return added


def error_category(message):
    if re.search(r"429|too many requests|rate.?limit|限流|频率", message, re.I):
        return "rate_limit"
    if re.search(r"余额|欠费|quota|credits|payment|insufficient.balance|402", message, re.I):
        return "quota"
    if re.search(r"401|403|unauthorized|authentication|Authorization|api.?key|凭证", message, re.I):
        return "auth"
    if re.search(r"Registry|Reasoning level|not supported|model_not_found|404", message, re.I):
        return "model"
    if re.search(r"timeout|timed out|ECONNRESET|temporary|gateway|request was cancelled|closed the text request|\b5(?:00|02|03|04|24)\b", message, re.I):
        return "temporary"
    return "error"


def batch_result(route, batch, target):
    prompt, cases = make_prompt(batch, target)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    start = time.monotonic()
    selection = {"providerId": route.provider, "modelId": route.model,
                 "options": {"reasoningLevel": route.reasoning}}
    budget = max(6300, min(16000, target * 48 + 2900))
    try:
        result = generate(selection, prompt, f"scale-{batch}", budget, timeout_seconds=145,
                          runtime_root=ROOT / "zcode-runtime", runtime_slot=threading.get_ident())
    except (OSError, RuntimeError) as error:
        return [], error_category(str(error)), round(time.monotonic() - start, 2), 0, None, prompt_sha
    rows = []
    quota = Counter()
    for ordinal, line in enumerate(result["text"].splitlines(), start=1):
        match = LINE.fullmatch(line)
        if match is None:
            continue
        key = int(match[1])
        scenario = cases.get(key)
        text = " ".join(match[2].strip().split())
        if (scenario is None or quota[key] >= scenario["quota"] or not 4 <= len(text) <= 180
                or not CHINESE.search(text) or CONTACT.search(text)):
            continue
        quota[key] += 1
        rows.append((ordinal, text, scenario))
    usage = result.get("usage") or {}
    return (rows, "ok" if rows else "no_valid_rows", round(time.monotonic() - start, 2),
            int(usage.get("totalTokens") or 0), result.get("finishReason"), prompt_sha)


def totals(db):
    return {**dict.fromkeys(TARGETS, 0), **dict(db.execute(
        "SELECT cohort, COUNT(*) FROM samples GROUP BY cohort"))}


def report(db):
    counts = totals(db)
    return {
        "goals": TARGETS,
        "unique": counts,
        "routes": [
            {"cohort": cohort, "route": route, "model": model, "rows": count}
            for cohort, route, model, count in db.execute(
                "SELECT cohort,route,model,COUNT(*) FROM samples GROUP BY 1,2,3 ORDER BY 1,4 DESC")
        ],
        "formats": [
            {"cohort": cohort, "format": fmt, "rows": count}
            for cohort, fmt, count in db.execute(
                "SELECT cohort,format,COUNT(*) FROM samples GROUP BY 1,2 ORDER BY 1,3 DESC")
        ],
        "batches": dict(db.execute(
            "SELECT COALESCE(terminal, 'running'), COUNT(*) FROM batches GROUP BY 1")),
        "tokens": db.execute("SELECT COALESCE(SUM(tokens),0) FROM batches").fetchone()[0],
    }


def save_snapshot(db, scope):
    payload = report(db)
    path = ROOT / ("progress.json" if scope == "all" else f"progress-{scope}.json")
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)
    return payload


def run(db, workers, batch_limit, scope):
    real_keys = real_text_keys()
    seeded = import_first_round(db, real_keys) if scope != "other_models" else 0
    active_targets = TARGETS if scope == "all" else {scope: TARGETS[scope]}
    routes = [route for route in ROUTES if route.cohort in active_targets]
    with db:
        if scope == "all":
            db.execute("""UPDATE batches SET terminal='interrupted', finished=? WHERE terminal IS NULL
                           AND owner IN ('all','legacy','deepseek_v41','other_models')""", (time.time(),))
        else:
            foreign = db.execute("SELECT COUNT(*) FROM batches WHERE terminal IS NULL AND cohort=? AND owner!=?", (scope, scope)).fetchone()[0]
            if foreign:
                raise RuntimeError("Another active job owns pending requests for this cohort.")
            db.execute("UPDATE batches SET terminal='interrupted', finished=? WHERE terminal IS NULL AND owner=?", (time.time(), scope))
    counts = totals(db)
    print(json.dumps({"seeded_from_round01": seeded, "unique": counts, "worker_limit": workers},
                     ensure_ascii=False), flush=True)
    pending = {}
    alive = {r.name: True for r in routes}
    slots = {r.name: r.slots for r in routes}
    targets = {r.name: r.rows for r in routes}
    permanent = set()
    streak = Counter()
    route_active = Counter()
    vendor_active = Counter()
    cohort_active = Counter()
    cooldown = {}
    launches = 0
    last_update = 0.0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or any(counts[c] < n for c, n in active_targets.items()):
            now = time.monotonic()
            active_cohorts = [c for c in active_targets if counts[c] < active_targets[c]]
            while len(pending) < workers and (not batch_limit or launches < batch_limit):
                candidates = [
                    r for r in routes
                    if alive[r.name] and counts[r.cohort] < active_targets[r.cohort]
                    and now >= cooldown.get(r.vendor, 0.0)
                    and route_active[r.name] < slots[r.name]
                    and vendor_active[r.vendor] < VENDOR_CAPS.get(r.vendor, 40)
                    and (not cohort_active[r.cohort] or
                         active_targets[r.cohort] - counts[r.cohort] > 80 * cohort_active[r.cohort])
                ]
                if not candidates:
                    break
                route = min(candidates, key=lambda r: (
                    cohort_active[r.cohort] / (active_targets[r.cohort] - counts[r.cohort]),
                    route_active[r.name] / slots[r.name],
                    ROUTES.index(r),
                ))
                requested = min(targets[route.name], max(18, active_targets[route.cohort] - counts[route.cohort]))
                with db:
                    cover = db.execute("INSERT INTO batches(route,cohort,requested,started,owner) VALUES (?, ?, ?, ?, ?)",
                                       (route.name, route.cohort, requested, time.time(), scope))
                    batch = cover.lastrowid
                future = executor.submit(batch_result, route, batch, requested)
                pending[future] = (route, batch, requested)
                launches += 1
                route_active[route.name] += 1
                vendor_active[route.vendor] += 1
                cohort_active[route.cohort] += 1
            if not pending:
                if not active_cohorts or (batch_limit and launches >= batch_limit):
                    break
                healthy = [r for r in routes if alive[r.name] and r.cohort in active_cohorts]
                if not healthy:
                    break
                next_ready = min(cooldown.get(r.vendor, now) for r in healthy)
                if next_ready <= now:
                    raise RuntimeError("The synthesis scheduler has work but no ready route.")
                time.sleep(min(max(next_ready - now, 0.1), 15))
                continue
            finished, _ = wait(list(pending), timeout=12, return_when=FIRST_COMPLETED)
            for future in finished:
                route, batch, requested = pending.pop(future)
                route_active[route.name] -= 1
                vendor_active[route.vendor] -= 1
                cohort_active[route.cohort] -= 1
                rows, terminal, duration, tokens, stop, prompt_sha = future.result()
                accepted = duplicates = 0
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    counts[route.cohort] = db.execute("SELECT COUNT(*) FROM samples WHERE cohort=?", (route.cohort,)).fetchone()[0]
                    for ordinal, text, scenario in rows:
                        if counts[route.cohort] + accepted >= active_targets[route.cohort]:
                            break
                        value = retain(db, real_keys, route.cohort, route.name, route.model, text,
                                       scenario["topic"], scenario["variant"], scenario,
                                       batch, ordinal, prompt_sha)
                        accepted += value
                        duplicates += 1 - value
                    db.execute("""UPDATE batches SET finished=?,terminal=?,usable=?,accepted=?,duplicates=?,
                                elapsed=?,tokens=?,response_finish=?,prompt_sha256=? WHERE id=?""",
                               (time.time(), terminal, len(rows), accepted, duplicates, duration, tokens, stop,
                                prompt_sha, batch))
                counts[route.cohort] += accepted
                if rows:
                    streak[route.name] = 0
                    if route.name not in permanent:
                        alive[route.name] = True
                    fast = 90 if route.cohort == "deepseek_v41" else 100
                    maximum = 240 if route.cohort == "deepseek_v41" else 205
                    if len(rows) >= 0.80 * requested and duration < fast:
                        targets[route.name] = min(targets[route.name] + 12, maximum)
                    if len(rows) < 0.45 * requested or duration > 125:
                        targets[route.name] = max(60, int(targets[route.name] * 0.82))
                else:
                    streak[route.name] += 1
                    if terminal == "rate_limit":
                        slots[route.name] = max(1, slots[route.name] - 1)
                        cooldown[route.vendor] = time.monotonic() + min(90, 24 * streak[route.name])
                        if streak[route.name] >= 8:
                            alive[route.name] = False
                    elif terminal in {"quota", "auth", "model"}:
                        permanent.add(route.name)
                        alive[route.name] = False
                    elif terminal in {"no_valid_rows", "temporary", "error"}:
                        targets[route.name] = max(32, int(targets[route.name] * 0.78))
                        if streak[route.name] >= 8:
                            alive[route.name] = False
                    elif streak[route.name] >= 4:
                        alive[route.name] = False
                print(json.dumps({"batch": batch, "route": route.name, "cohort": route.cohort,
                                  "terminal": terminal, "requested": requested, "usable": len(rows),
                                  "accepted": accepted, "duplicates": duplicates, "elapsed": duration,
                                  "total": counts[route.cohort]}, ensure_ascii=False), flush=True)
            if finished and time.monotonic() - last_update >= 10:
                save_snapshot(db, scope)
                last_update = time.monotonic()
    final = save_snapshot(db, scope)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return all(final["unique"][c] >= target for c, target in active_targets.items())


def export(db):
    exports = {}
    with db:
        db.execute("BEGIN")
        snapshot = report(db)
        for cohort in TARGETS:
            output = ROOT / f"bili-synthetic-{cohort}.jsonl.gz"
            staging = output.with_suffix(".writing")
            rows = db.execute("""SELECT sample_id, route, model, text, topic, format, scenario_json, batch,
                               ordinal, prompt_sha256 FROM samples WHERE cohort=? ORDER BY rowid""", (cohort,))
            count = 0
            with gzip.open(staging, "wt", encoding="utf-8", newline="\n", compresslevel=5) as stream:
                for sample_id, route, model, text, topic, fmt, scenario, batch, ordinal, digest in rows:
                    payload = dict(sample_id=sample_id, text=text, label="ai", split="train", family=cohort,
                                   route=route, model_id=model, topic=topic, format=fmt, batch=batch,
                                   ordinal=ordinal, scenario=json.loads(scenario), prompt_sha256=digest,
                                   source="zcode_deepseek" if cohort == "deepseek_v41" else "zcode_other")
                    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            staging.replace(output)
            with output.open("rb") as stream:
                exports[cohort] = {"rows": count, "file": output.name, "bytes": output.stat().st_size,
                                   "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    snapshot["exports"] = exports
    (ROOT / "delivery.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(exports, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--export", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-limit", type=int, default=0)
    parser.add_argument("--cohort", choices=("all", *TARGETS), default="all")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_limit < 0:
        parser.error("workers must be positive and batch-limit must be nonnegative")
    with connect() as db:
        if args.status:
            print(json.dumps(report(db), ensure_ascii=False))
        elif args.export:
            export(db)
        else:
            with exclusive_run(args.cohort):
                complete = run(db, args.workers, args.batch_limit, args.cohort)
            if not complete and not args.batch_limit:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
