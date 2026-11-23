"""Freeze video-level splits and a minimal Colab input from the live corpus."""

import gzip
import hashlib
import json
import random
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "round01"
LABELS = {"ai_like": "ai", "human_like": "human", "uncertain": "review"}
SPLITS = ("train", "validation", "test")
RATIOS = (0.80, 0.10, 0.10)
RELEASE_ROWS = 200_370


def body_key(text):
    normalized = "".join(unicodedata.normalize("NFKC", text).casefold().split())
    letters = re.sub(r"[^\w]+|_", "", normalized)
    return letters if len(letters) >= 10 else normalized


def video_split(groups, locked_test):
    videos = sorted(groups)
    totals = [sum(group[i] for group in groups.values()) for i in range(6)]
    weights = (1, 2, 2, 2, 3, 1)
    best = None
    for seed in range(512):
        rng = random.Random(20261122 + seed)
        assignment = {}
        counts = [[0] * 6 for _ in SPLITS]
        for video in videos:
            value = rng.random()
            bucket = 2 if video in locked_test or value >= 0.90 else int(value >= 0.80)
            assignment[video] = SPLITS[bucket]
            for index, size in enumerate(groups[video]):
                counts[bucket][index] += size
        loss = sum(weight * ((actual - total * ratio) / max(total * ratio, 1)) ** 2
                   for ratio, observed in zip(RATIOS, counts)
                   for weight, actual, total in zip(weights, observed, totals))
        if best is None or loss < best[0]:
            best = (loss, seed, assignment)
    return 20261122 + best[1], best[2]


def write_jsonl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=5) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    probe_path = OUT / "bili-round01-user-confirmed.jsonl"
    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()]
    if {row["id"] for row in probes} != {f"U{i:03d}" for i in range(1, 18)} or len(probes) != 17:
        raise ValueError("Expected 17 user-confirmed samples.")
    probe_keys = {body_key(row["text"]) for row in probes}
    queue = json.loads((ROOT / "bili-ai-comment-queue.json").read_text(encoding="utf-8-sig"))
    videos = {row["bvid"]: row for row in queue["videos"]}
    db_uri = (ROOT / "bili-live-corpus.sqlite3").as_uri() + "?mode=ro"
    records = []
    groups = {}
    locked_test = set()
    matches = 0
    with sqlite3.connect(db_uri, uri=True) as db:
        rows = db.execute("""SELECT c.sample_id,c.video_id,c.comment_level,c.metadata,
                                   s.suggested_label,s.annotation_source
                             FROM comments c LEFT JOIN suggestions s USING(sample_id)
                             WHERE c.rowid <= ? ORDER BY c.rowid""", (RELEASE_ROWS,))
        for sample_id, video, level, metadata, suggestion, source in rows:
            original = json.loads(metadata)
            label = LABELS[suggestion]
            text = original["text"]
            key = body_key(text)
            scope = videos.get(video, {}).get("topic_scope", "") or ""
            dsh = scope.startswith("DSH ") or scope == "大肥鱼总结与回复插件"
            if key in probe_keys:
                matches += 1
            if key in probe_keys or "k004" in key or "k004" in original.get("video_title", "").casefold():
                locked_test.add(video)
            group = groups.setdefault(video, [0, 0, 0, 0, 0, 1])
            group[0] += 1
            group[1] += label == "ai"
            group[2] += label == "human"
            group[3] += dsh
            group[4] += dsh and label == "ai"
            records.append((sample_id, video, text, label, source, level,
                            bool(original.get("review_priority") or original.get("nickname_hint")),
                            dsh, key))
    if len(records) != RELEASE_ROWS or "" in {record[-1] for record in records}:
        raise ValueError("The input corpus does not match the frozen 200,370-row release.")

    seed, split = video_split(groups, locked_test)
    split_order = {name: order for order, name in enumerate(SPLITS)}
    winner = {}
    first_labels = {}
    conflicts = set()
    for sample_id, video, _, label, source, _, _, dsh, key in records:
        if label == "review":
            continue
        if first_labels.setdefault(key, label) != label:
            conflicts.add(key)
        rank = (split_order[split[video]], dsh, source.startswith("google/"))
        if key not in winner or rank > winner[key][0]:
            winner[key] = (rank, sample_id)

    counts = {name: Counter() for name in SPLITS}
    review = []
    output = []
    used_keys = set()
    for sample_id, video, text, label, source, level, priority, dsh, key in records:
        partition = split[video]
        reason = ""
        if label == "review":
            reason = "uncertain"
        elif key in conflicts:
            reason = "conflicting_suggestions"
        elif winner[key][1] != sample_id:
            reason = "duplicate_body"
        row = [sample_id, video, text, label, source, level, priority, reason]
        output.append(row)
        counts[partition]["all"] += 1
        counts[partition]["dsh_all"] += dsh
        if reason:
            counts[partition][reason] += 1
            if reason != "duplicate_body":
                counts[partition]["dsh_review"] += dsh
                review.append(dict(sample_id=sample_id, video_id=video, text=text,
                                   prior_suggestion=label, annotation_source=source,
                                   partition=partition, dsh=dsh, priority=priority,
                                   review_reason=reason))
        else:
            if key in used_keys:
                raise ValueError("A normalized body was used twice.")
            used_keys.add(key)
            counts[partition][label] += 1
            if dsh:
                counts[partition][f"dsh_{label}"] += 1
                counts[partition]["dsh_gemini_ai"] += label == "ai" and source.startswith("google/")

    dataset = OUT / "bili-round01-corpus.jsonl.gz"
    write_jsonl(dataset, output)
    review.sort(key=lambda row: (not row["priority"], not row["dsh"], row["sample_id"]))
    write_jsonl(OUT / "bili-round01-review.jsonl.gz", review)
    manifest = {
        "schema_version": 1,
        "raw_rows": len(records),
        "seed": seed,
        "ratios": dict(zip(SPLITS, RATIOS)),
        "corpus_sha256": hashlib.file_digest(dataset.open("rb"), "sha256").hexdigest(),
        "corpus_columns": ["sample_id", "video_id", "text", "reference", "annotation_source",
                           "comment_level", "review_priority", "hold_reason"],
        "counts": {name: dict(value, videos=sum(split[v] == name for v in groups))
                   for name, value in counts.items()},
        "probe": {"known_ai": len(probes), "matching_bili_rows": matches,
                  "related_videos_reserved_for_test": sorted(locked_test)},
        "review_rows": len(review),
        "video_assignments": {video: {"split": partition,
                                      "dsh": bool(groups[video][3]),
                                      "scope": videos.get(video, {}).get("topic_scope", "") or ""}
                              for video, partition in sorted(split.items())},
    }
    (OUT / "bili-round01-split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "video_assignments"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
