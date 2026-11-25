"""Step 2: Run External Gemini Data Factory (Zero Antigravity Quota Cost!).

Execution Modes:
  --mode batch    : Gemini Batch API (50% discount! Builds standard Gemini Batch JSONL, submits via
                    `google-genai` SDK or pure-stdlib REST API, polls & downloads results with resume).
  --mode realtime : Concurrent async ThreadPool calling Gemini GenerateContent REST API directly with
                    adaptive QPS pacing, exponential backoff on 429/503, and JSONL per-batch checkpointing.
  --mode pack     : Exports copy-pasteable offline `prompts/*.txt` packs for Gemini Web / AI Studio /
                    external Gemini CLI jobs, plus `--import-pack` to ingest saved JSON replies.
  --mode dry-run  : Zero-network local end-to-end verification using deterministic mock Gemini replies
                    (tests JSON schema parsing, 3-char anti-drift tag checks, Tier-2 escalation routing).

Automatic 2-Tier Tribunal Escalation (`--tier tier1` -> `--tier tier2`):
  - Tier 1 (`gemini-2.5-flash`): Bulk mega-packed scoring (`20-25` rows/call) for all queues.
  - Tier 2 (`gemini-2.5-pro`)  : Automatically harvests ONLY ambiguous / low-confidence / high-disagreement /
                                 `is_trap_human` rows (~12-18%) from Tier-1 outputs and re-scores them with Pro!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gemini_v08_datafactory.rubric_v08 import (
    COUNTERFACTUAL_RESPONSE_SCHEMA,
    COUNTERFACTUAL_SYSTEM_INSTRUCTION_V3,
    SCORING_RESPONSE_SCHEMA,
    SCORING_SYSTEM_INSTRUCTION_V3,
    build_counterfactual_user_prompt,
    build_offline_manual_prompt,
    build_scoring_user_prompt,
    make_check_tag,
    parse_counterfactual_response,
    parse_scoring_response,
)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIER1_MODEL = "gemini-2.5-flash"
DEFAULT_TIER2_MODEL = "gemini-2.5-pro"

ALL_QUEUES = (
    "A_noise_surgery",
    "B_wild_hard_mining",
    "C_counterfactual_spans",
    "D_thread_context",
)


def _get_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key.strip():
        raise SystemExit(
            "[ERROR] Missing Gemini API key! Set environment variable `GEMINI_API_KEY` (or pass `--api-key`), "
            "or use `--mode pack` for offline Gemini Web / CLI, or `--mode dry-run` for local verification."
        )
    return key.strip()


def _load_batches(batch_file: Path) -> list[dict[str, Any]]:
    if not batch_file.is_file():
        return []
    out: list[dict[str, Any]] = []
    with batch_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_done_batch_ids(out_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not out_jsonl.is_file():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("batch_id"):
                    done.add(str(obj["batch_id"]))
            except json.JSONDecodeError:
                pass
    return done


def build_gemini_request_body(
    batch: dict[str, Any],
    temperature: float = 0.0,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    """Construct standard Gemini `GenerateContentRequest` dict (works identically in Realtime & Batch API!)."""
    qname = batch["queue"]
    is_cf = qname == "C_counterfactual_spans"
    sys_inst = COUNTERFACTUAL_SYSTEM_INSTRUCTION_V3 if is_cf else SCORING_SYSTEM_INSTRUCTION_V3
    user_prompt = (
        build_counterfactual_user_prompt(batch["rows"])
        if is_cf
        else build_scoring_user_prompt(batch["rows"], queue_type=qname)
    )
    schema = COUNTERFACTUAL_RESPONSE_SCHEMA if is_cf else SCORING_RESPONSE_SCHEMA

    gen_cfg: dict[str, Any] = {
        "temperature": temperature if not is_cf else max(0.5, temperature),
        "responseMimeType": "application/json",
        "responseSchema": schema,
    }
    if thinking_budget is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

    return {
        "system_instruction": {"parts": [{"text": sys_inst}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": gen_cfg,
    }


def _extract_candidate_text(api_resp: dict[str, Any]) -> str:
    cands = api_resp.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no_candidates:{str(api_resp)[:240]}")
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    texts = [str(p.get("text", "")) for p in parts if not p.get("thought") and "text" in p]
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Pure-Stdlib Gemini REST Client (Realtime)
# ---------------------------------------------------------------------------

def call_gemini_rest(
    model: str,
    body: dict[str, Any],
    api_key: str,
    timeout_s: int = 120,
    max_retries: int = 5,
) -> tuple[str, dict[str, Any]]:
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_err = ""
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                resp_obj = json.loads(resp.read().decode("utf-8"))
                txt = _extract_candidate_text(resp_obj)
                usage = resp_obj.get("usageMetadata") or {}
                return txt, usage
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")[:300]
            last_err = f"HTTP {he.code}: {err_body}"
            if he.code in (429, 500, 502, 503, 504):
                sleep_s = min(60.0, (2.0 ** attempt) + random.uniform(0.3, 1.5))
                time.sleep(sleep_s)
                continue
            raise RuntimeError(last_err) from he
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(30.0, (2.0 ** attempt) + 0.5))
    raise RuntimeError(f"gemini_rest_failed_after_{max_retries}_retries: {last_err}")


# ---------------------------------------------------------------------------
# Dry-Run Deterministic Mock Generator (Zero API Calls, Full Pipeline Test)
# ---------------------------------------------------------------------------

def mock_gemini_reply(batch: dict[str, Any], tier: str, canary_truth: dict[str, dict[str, Any]] | None = None) -> str:
    """Generate realistic mock Structured JSON responses for local end-to-end verification."""
    qname = batch["queue"]
    items = []
    rng = random.Random(hash(batch["batch_id"]) & 0xFFFFFFFF)

    if qname == "C_counterfactual_spans":
        for r in batch["rows"]:
            mode = r["cf_mode"]
            base_t = str(r["text"])[:90]
            if mode == "sandwich_hybrid_span":
                ai_mid = "值得注意的是，这一现象并非单一因素所致——它在效率与体验之间形成了微妙的平衡。"
                rewritten = f"笑死前面确实离谱，{ai_mid} 反正我自己试过两次直接劝退了[doge]"
                items.append(
                    {
                        "idx": r["idx"],
                        "tag": r["tag"],
                        "mode": mode,
                        "rewritten_text": rewritten,
                        "expected_score": 78,
                        "ai_span_texts": [ai_mid],
                        "quality_ok": True,
                    }
                )
            elif mode == "ai_to_structured_human":
                rewritten = f"我自己实测了下（主观评分8.5/10）：1. {base_t[:28]} 这块挺实在；2. 细节还是有点毛糙，懂的都懂哈哈"
                items.append(
                    {
                        "idx": r["idx"],
                        "tag": r["tag"],
                        "mode": mode,
                        "rewritten_text": rewritten,
                        "expected_score": 14,
                        "ai_span_texts": [],
                        "quality_ok": True,
                    }
                )
            else:
                rewritten = f"平心而论，{base_t[:24]}的关键不在于表面形式，而在于背后取舍——看似容易，实则难在边界把控。"
                items.append(
                    {
                        "idx": r["idx"],
                        "tag": r["tag"],
                        "mode": mode,
                        "rewritten_text": rewritten,
                        "expected_score": 84,
                        "ai_span_texts": [],
                        "quality_ok": True,
                    }
                )
        return json.dumps({"items": items}, ensure_ascii=False)

    # Scoring queues A / B / D
    for r in batch["rows"]:
        sid = r["sample_id"]
        if canary_truth and sid in canary_truth:
            # Simulate realistic calibrated Gemini teacher (MAE ~3-5 vs Hydrogen gold)
            g_sc = float(canary_truth[sid]["gold_score"])
            noise = rng.randint(-4, 4)
            sc = max(0, min(100, int(round(g_sc + noise))))
        elif r.get("prior_score") is not None:
            sc = max(0, min(100, int(round(float(r["prior_score"]) + rng.randint(-12, 12)))))
        else:
            txt = str(r.get("text") or "")
            sc = 82 if ("——" in txt or "归档" in txt or "不是" in txt and "而是" in txt) else rng.choice([12, 18, 24, 46, 55, 78, 86])

        if sc <= 30:
            code = "H"
            prag = [0, 0, 0, 1, 1, 0, 0, 1]
            is_trap = ("/" in str(r["text"]) or "1." in str(r["text"]) or "《" in str(r["text"]))
        elif sc <= 69:
            code = "S" if len(str(r["text"])) <= 15 else "P"
            prag = [0, 0, 0, 0, 0, 0, 0, 0]
            is_trap = False
        else:
            code = "T" if qname == "D_thread_context" else ("A" if "归档" in str(r["text"]) else "E")
            prag = [1 if code == "R" else 0, 1 if code == "E" else 0, 1 if code == "A" else 0, 0, 0, 0, 0, 0]
            is_trap = False

        conf = 92 if tier == "tier2" else (65 if 28 <= sc <= 72 else 88)
        items.append(
            {
                "idx": r["idx"],
                "tag": r["tag"],
                "score": sc,
                "code": code,
                "confidence": conf,
                "is_trap_human": is_trap,
                "prag_8d": prag,
            }
        )
    return json.dumps({"items": items}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Automatic Tier-1 -> Tier-2 Escalation Builder
# ---------------------------------------------------------------------------

def build_tier2_escalation_batches(
    work_dir: Path,
    queues: list[str],
    batch_size: int = 20,
    disagreement_thresh: float = 28.0,
    conf_thresh: int = 75,
) -> dict[str, Any]:
    """Scan Tier-1 Flash results and route ambiguous / trap / high-disagreement rows to Tier-2 Pro."""
    tier1_dir = work_dir / "outputs_tier1"
    queues_dir = work_dir / "queues"
    esc_rows: list[dict[str, Any]] = []
    reasons_cnt: dict[str, int] = {}

    for qname in queues:
        if qname == "C_counterfactual_spans":
            continue
        t1_file = tier1_dir / f"{qname}.tier1.jsonl"
        q_file = queues_dir / f"{qname}.batches.jsonl"
        if not t1_file.is_file() or not q_file.is_file():
            continue

        orig_by_sid: dict[str, dict[str, Any]] = {}
        for b in _load_batches(q_file):
            for r in b["rows"]:
                orig_by_sid[r["sample_id"]] = r

        with t1_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for sid, pred in (rec.get("parsed") or {}).items():
                    orig = orig_by_sid.get(sid)
                    if not orig:
                        continue
                    sc = int(pred["score"])
                    conf = int(pred.get("confidence", 80))
                    is_trap = bool(pred.get("is_trap_human", False))
                    prior = orig.get("prior_score")

                    esc_reason = ""
                    if 28 <= sc <= 72:
                        esc_reason = "midband_ambiguous"
                    elif conf < conf_thresh:
                        esc_reason = "low_confidence"
                    elif is_trap and sc >= 20:
                        esc_reason = "trap_human_verify"
                    elif prior is not None and abs(sc - float(prior)) >= disagreement_thresh:
                        esc_reason = "prior_disagreement"

                    if esc_reason:
                        reasons_cnt[esc_reason] = reasons_cnt.get(esc_reason, 0) + 1
                        row_copy = dict(orig)
                        row_copy["tier1_score"] = sc
                        row_copy["tier1_code"] = pred["code"]
                        row_copy["esc_reason"] = esc_reason
                        esc_rows.append(row_copy)

    # Pack Tier-2 escalation batches per original queue (keeps Queue D thread context intact!)
    by_q: dict[str, list[dict[str, Any]]] = {}
    for r in esc_rows:
        by_q.setdefault(r["queue"], []).append(r)

    tier2_q_dir = work_dir / "queues_tier2"
    tier2_q_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"total_escalated": len(esc_rows), "reasons": reasons_cnt, "queues": {}}
    for qname, rows in by_q.items():
        batches = []
        for bi in range(0, len(rows), batch_size):
            chunk = rows[bi : bi + batch_size]
            bid = f"T2_{qname}_b{bi // batch_size + 1:05d}"
            packed = []
            for idx, r in enumerate(chunk, 1):
                rc = dict(r)
                rc["idx"] = idx
                rc["tag"] = make_check_tag(bid, r["sample_id"])
                packed.append(rc)
            batches.append({"batch_id": bid, "queue": qname, "n": len(packed), "rows": packed})
        out_bf = tier2_q_dir / f"{qname}.batches.jsonl"
        with out_bf.open("w", encoding="utf-8", newline="\n") as wf:
            for b in batches:
                wf.write(json.dumps(b, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary["queues"][qname] = {"rows": len(rows), "batches": len(batches), "file": str(out_bf)}

    (work_dir / "tier2_escalation_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Mode Runners: Realtime / Batch API (50% off) / Offline Pack / Dry-Run
# ---------------------------------------------------------------------------

def run_realtime_or_dryrun(
    batches: list[dict[str, Any]],
    out_jsonl: Path,
    model: str,
    tier: str,
    dry_run: bool,
    api_key: str,
    workers: int,
    qps: float,
    thinking_budget: int | None,
    canary_truth: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_batch_ids(out_jsonl)
    todo = [b for b in batches if b["batch_id"] not in done_ids]
    if not todo:
        return {"skipped_done": len(done_ids), "processed": 0, "failed": 0}

    lock = threading.Lock()
    min_interval = (1.0 / max(0.1, qps)) if not dry_run else 0.0
    last_ts = [0.0]

    def _pace():
        if min_interval <= 0:
            return
        with lock:
            now = time.monotonic()
            wait = min_interval - (now - last_ts[0])
            if wait > 0:
                time.sleep(wait)
            last_ts[0] = time.monotonic()

    def _worker(batch: dict[str, Any]) -> dict[str, Any]:
        _pace()
        t0 = time.monotonic()
        if dry_run:
            raw = mock_gemini_reply(batch, tier=tier, canary_truth=canary_truth)
            usage = {"mock": True}
        else:
            body = build_gemini_request_body(batch, temperature=0.0, thinking_budget=thinking_budget)
            raw, usage = call_gemini_rest(model=model, body=body, api_key=api_key)

        if batch["queue"] == "C_counterfactual_spans":
            parsed, stats = parse_counterfactual_response(raw, batch["rows"])
        else:
            parsed, stats = parse_scoring_response(raw, batch["rows"])
        return {
            "batch_id": batch["batch_id"],
            "queue": batch["queue"],
            "tier": tier,
            "model": f"mock-{model}" if dry_run else model,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "usage": usage,
            "stats": stats,
            "parsed": parsed,
        }

    ok_cnt = 0
    fail_cnt = 0
    total_rows_ok = 0
    with out_jsonl.open("a", encoding="utf-8", newline="\n") as af:
        with ThreadPoolExecutor(max_workers=1 if dry_run else max(1, workers)) as ex:
            futs = {ex.submit(_worker, b): b["batch_id"] for b in todo}
            for fut in as_completed(futs):
                bid = futs[fut]
                try:
                    rec = fut.result()
                    with lock:
                        af.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        af.flush()
                    ok_cnt += 1
                    total_rows_ok += int(rec["stats"]["parsed"])
                except Exception as exc:
                    fail_cnt += 1
                    print(f"[ERR] {bid}: {exc}")

    return {"skipped_done": len(done_ids), "processed": ok_cnt, "rows_parsed": total_rows_ok, "failed": fail_cnt}


def run_batch_api_mode(
    work_dir: Path,
    qname: str,
    batches: list[dict[str, Any]],
    out_jsonl: Path,
    model: str,
    tier: str,
    api_key: str | None,
    submit: bool,
    poll: bool,
    thinking_budget: int | None,
    max_requests_per_file: int = 2000,
) -> dict[str, Any]:
    """Generate Gemini Batch API JSONL shards (50% discount!) and optionally submit + poll via google-genai SDK."""
    batch_api_dir = work_dir / f"batch_api_{tier}" / qname
    batch_api_dir.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_batch_ids(out_jsonl)
    todo = [b for b in batches if b["batch_id"] not in done_ids]
    if not todo:
        return {"status": "all_done", "skipped_done": len(done_ids)}

    batch_lookup = {b["batch_id"]: b for b in todo}
    shard_files: list[Path] = []
    for si in range(0, len(todo), max_requests_per_file):
        chunk = todo[si : si + max_requests_per_file]
        sf = batch_api_dir / f"requests_part_{si // max_requests_per_file + 1:03d}.jsonl"
        with sf.open("w", encoding="utf-8", newline="\n") as wf:
            for b in chunk:
                req_line = {
                    "key": b["batch_id"],
                    "request": build_gemini_request_body(b, temperature=0.0, thinking_budget=thinking_budget),
                }
                wf.write(json.dumps(req_line, ensure_ascii=False, separators=(",", ":")) + "\n")
        shard_files.append(sf)

    info: dict[str, Any] = {
        "queue": qname,
        "tier": tier,
        "model": model,
        "pending_batches": len(todo),
        "batch_jsonl_files": [str(p) for p in shard_files],
        "submitted_jobs": [],
    }
    if not submit:
        print(f"[BATCH READY] Generated {len(shard_files)} Gemini Batch API JSONL file(s) under {batch_api_dir}.")
        print("              Pass `--submit-batch` (with `pip install google-genai` + GEMINI_API_KEY) to upload & run at 50% off!")
        return info

    key = _get_api_key(api_key)
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] `--submit-batch` requires the official Google GenAI SDK: run `pip install -U google-genai` "
            "(or upload the generated `requests_part_*.jsonl` files directly in Google AI Studio / Vertex Batch)."
        ) from exc

    client = genai.Client(api_key=key)
    jobs_file = batch_api_dir / "submitted_jobs.json"
    jobs: list[dict[str, Any]] = json.loads(jobs_file.read_text(encoding="utf-8")) if jobs_file.is_file() else []

    for sf in shard_files:
        if any(j.get("source_file") == str(sf) for j in jobs):
            continue
        uploaded = client.files.upload(
            file=str(sf),
            config=types.UploadFileConfig(display_name=f"meguviz_v08_{tier}_{qname}_{sf.stem}", mime_type="jsonl"),
        )
        job = client.batches.create(
            model=f"models/{model}" if not model.startswith("models/") else model,
            src=uploaded.name,
            config={"display_name": f"meguviz-v08-{tier}-{qname}-{sf.stem}"},
        )
        jobs.append({"job_name": job.name, "source_file": str(sf), "state": str(job.state), "imported": False})
        jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[BATCH SUBMITTED] {job.name} ({sf.name})")

    if poll:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        done_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while any(not j.get("imported") for j in jobs):
            pending_any = False
            for j in jobs:
                if j.get("imported"):
                    continue
                cur = client.batches.get(name=j["job_name"])
                st = str(getattr(cur.state, "name", cur.state))
                j["state"] = st
                if st not in done_states:
                    pending_any = True
                    continue
                if st == "JOB_STATE_SUCCEEDED":
                    res_file_name = cur.dest.file_name
                    raw_bytes = client.files.download(file=res_file_name)
                    res_local = batch_api_dir / f"results_{Path(j['source_file']).stem}.jsonl"
                    res_local.write_bytes(raw_bytes)
                    imported_n = _import_batch_api_result_file(res_local, batch_lookup, out_jsonl, model, tier)
                    j["imported"] = True
                    j["imported_batches"] = imported_n
                    print(f"[BATCH IMPORTED] {j['job_name']} -> {imported_n} batches")
                else:
                    j["imported"] = True
                    print(f"[BATCH FAILED] {j['job_name']} state={st}")
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if pending_any:
                print("[BATCH POLL] Waiting 30s for Gemini Batch jobs ...")
                time.sleep(30)

    info["submitted_jobs"] = jobs
    return info


def _import_batch_api_result_file(
    result_jsonl: Path,
    batch_lookup: dict[str, dict[str, Any]],
    out_jsonl: Path,
    model: str,
    tier: str,
) -> int:
    imported = 0
    with result_jsonl.open("r", encoding="utf-8") as rf, out_jsonl.open("a", encoding="utf-8", newline="\n") as af:
        for line in rf:
            if not line.strip():
                continue
            obj = json.loads(line)
            bid = str(obj.get("key") or "")
            batch = batch_lookup.get(bid)
            if not batch or "response" not in obj:
                continue
            try:
                txt = _extract_candidate_text(obj["response"])
                if batch["queue"] == "C_counterfactual_spans":
                    parsed, stats = parse_counterfactual_response(txt, batch["rows"])
                else:
                    parsed, stats = parse_scoring_response(txt, batch["rows"])
                rec = {
                    "batch_id": bid,
                    "queue": batch["queue"],
                    "tier": tier,
                    "model": model,
                    "via": "gemini_batch_api",
                    "stats": stats,
                    "parsed": parsed,
                }
                af.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                imported += 1
            except Exception as exc:
                print(f"[WARN] batch parse failed for {bid}: {exc}")
    return imported


def run_offline_pack_mode(
    work_dir: Path,
    qname: str,
    batches: list[dict[str, Any]],
    out_jsonl: Path,
    tier: str,
    model: str,
    do_import: bool,
) -> dict[str, Any]:
    """Export/import copy-pasteable prompts for Gemini Web / AI Studio / external CLI (zero API key needed)."""
    pack_dir = work_dir / f"offline_pack_{tier}" / qname
    prompts_dir = pack_dir / "prompts"
    replies_dir = pack_dir / "replies"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    replies_dir.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done_batch_ids(out_jsonl)
    if do_import:
        imported = 0
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("a", encoding="utf-8", newline="\n") as af:
            for b in batches:
                bid = b["batch_id"]
                if bid in done_ids:
                    continue
                cand = next((replies_dir / f"{bid}{ext}" for ext in (".json", ".txt", ".out.txt") if (replies_dir / f"{bid}{ext}").is_file()), None)
                if not cand:
                    continue
                raw = cand.read_text(encoding="utf-8", errors="replace")
                try:
                    if qname == "C_counterfactual_spans":
                        parsed, stats = parse_counterfactual_response(raw, b["rows"])
                    else:
                        parsed, stats = parse_scoring_response(raw, b["rows"])
                    af.write(
                        json.dumps(
                            {"batch_id": bid, "queue": qname, "tier": tier, "model": model, "via": "offline_pack", "stats": stats, "parsed": parsed},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    imported += 1
                except Exception as exc:
                    print(f"[WARN] Offline reply parse failed ({cand.name}): {exc}")
        return {"imported_batches": imported, "replies_dir": str(replies_dir)}

    written = 0
    for b in batches:
        if b["batch_id"] in done_ids:
            continue
        txt = build_offline_manual_prompt(b["batch_id"], b["rows"], qname)
        (prompts_dir / f"{b['batch_id']}.txt").write_text(txt, encoding="utf-8", newline="\n")
        written += 1
    return {"exported_prompts": written, "prompts_dir": str(prompts_dir), "replies_dir": str(replies_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "v08-datafactory-work")
    ap.add_argument("--mode", choices=("batch", "realtime", "pack", "dry-run"), default="batch")
    ap.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    ap.add_argument("--queues", nargs="+", default=list(ALL_QUEUES))
    ap.add_argument("--model", type=str, default=None, help="Override Gemini model (default: tier1=gemini-2.5-flash, tier2=gemini-2.5-pro)")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers for --mode realtime")
    ap.add_argument("--qps", type=float, default=6.0, help="Max requests/sec for --mode realtime")
    ap.add_argument("--thinking-budget", type=int, default=None, help="Optional Gemini 2.5 thinkingBudget (0 disables thinking for ultra-low cost)")
    ap.add_argument("--submit-batch", action="store_true", help="Upload & start Gemini Batch API jobs (50% off)")
    ap.add_argument("--poll-batch", action="store_true", help="Wait for Batch API completion & auto-import results")
    ap.add_argument("--import-pack", action="store_true", help="Import saved replies for --mode pack")
    ap.add_argument("--import-batch-results", type=Path, default=None, help="Manually import a downloaded Gemini Batch result JSONL")
    ap.add_argument("--esc-disagreement", type=float, default=28.0, help="Tier-2 escalation |score - prior| threshold")
    ap.add_argument("--esc-conf", type=int, default=75, help="Tier-2 escalation confidence < threshold")
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    model = args.model or (DEFAULT_TIER1_MODEL if args.tier == "tier1" else DEFAULT_TIER2_MODEL)

    if args.tier == "tier2":
        esc_info = build_tier2_escalation_batches(
            work_dir,
            args.queues,
            disagreement_thresh=args.esc_disagreement,
            conf_thresh=args.esc_conf,
        )
        print("[TIER-2 ESCALATION]", json.dumps(esc_info, ensure_ascii=False))
        q_root = work_dir / "queues_tier2"
        out_root = work_dir / "outputs_tier2"
    else:
        q_root = work_dir / "queues"
        out_root = work_dir / "outputs_tier1"

    canary_truth: dict[str, dict[str, Any]] | None = None
    if args.mode == "dry-run":
        vp = work_dir / "canary_vault.json"
        if vp.is_file():
            canary_truth = {c["blind_sample_id"]: c for c in json.loads(vp.read_text(encoding="utf-8")).get("canaries", [])}

    api_key = ""
    if args.mode == "realtime":
        api_key = _get_api_key(args.api_key)

    report: dict[str, Any] = {"mode": args.mode, "tier": args.tier, "model": model, "queues": {}}
    for qname in args.queues:
        bf = q_root / f"{qname}.batches.jsonl"
        batches = _load_batches(bf)
        if not batches:
            continue
        out_jsonl = out_root / f"{qname}.{args.tier}.jsonl"

        if args.import_batch_results and args.import_batch_results.is_file():
            n_imp = _import_batch_api_result_file(
                args.import_batch_results,
                {b["batch_id"]: b for b in batches},
                out_jsonl,
                model,
                args.tier,
            )
            report["queues"][qname] = {"manual_batch_imported": n_imp}
        elif args.mode in ("realtime", "dry-run"):
            report["queues"][qname] = run_realtime_or_dryrun(
                batches=batches,
                out_jsonl=out_jsonl,
                model=model,
                tier=args.tier,
                dry_run=(args.mode == "dry-run"),
                api_key=api_key,
                workers=args.workers,
                qps=args.qps,
                thinking_budget=args.thinking_budget,
                canary_truth=canary_truth,
            )
        elif args.mode == "batch":
            report["queues"][qname] = run_batch_api_mode(
                work_dir=work_dir,
                qname=qname,
                batches=batches,
                out_jsonl=out_jsonl,
                model=model,
                tier=args.tier,
                api_key=args.api_key,
                submit=args.submit_batch,
                poll=args.poll_batch,
                thinking_budget=args.thinking_budget,
            )
        elif args.mode == "pack":
            report["queues"][qname] = run_offline_pack_mode(
                work_dir=work_dir,
                qname=qname,
                batches=batches,
                out_jsonl=out_jsonl,
                tier=args.tier,
                model=model,
                do_import=args.import_pack,
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
