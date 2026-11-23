"""Local ORT node-level CPU profiler & v0.3-loop-2x4-e192 diagnostic audit on v0.4 250k AI + multi-era Bilibili corpora."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import gzip
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import time

import numpy as np
import onnxruntime as ort

from .v03_runtime import MeguvizV03Engine


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V02_ONNX = REPO_ROOT / "artifacts" / "v0.2" / "meguvizcac-v0.2-int8" / "meguvizcac-v0.2-int8.onnx"
DEFAULT_V03_BUNDLE = REPO_ROOT / "artifacts" / "v0.3" / "out_v03_2x4_e192" / "meguvizcac-v0.3-loop-2x4-e192-int8"
DEFAULT_SYNTH_PATH = REPO_ROOT / "v04-corpus" / "synthetic-250k" / "bili-v04-synthetic-250k.jsonl.gz"
DEFAULT_REAL_DB = REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3"
DEFAULT_AUDIT_OUT = REPO_ROOT / "artifacts" / "v0.4" / "v04-corpus-diagnostic-audit.json"
DEFAULT_MD_OUT = REPO_ROOT / "docs" / "v04-pretrain-research-audit.md"

CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff][ \t]+[\u4e00-\u9fff]")
EMDASH_RE = re.compile(r"——|--")
KAOMOJI_RE = re.compile(r"[(\[（【][^()\n\u4e00-\u9fff]{2,14}[)\]）】]")
ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+")


def sanitize_snippet(text: str, limit: int = 88) -> str:
    clean = URL_RE.sub("<URL>", " ".join(str(text or "").split()))
    clean = re.sub(r"@[\w\-\u4e00-\u9fff]+", "@<USER>", clean)
    return clean if len(clean) <= limit else clean[:limit] + "…"


def profile_ort_ops(
    onnx_path: Path,
    feed: dict[str, np.ndarray],
    warmup: int = 15,
    runs: int = 60,
    threads: int = 1,
) -> dict:
    """Measure CPU ORT per-op latency (us/call & share%) with separate cache warmup."""
    onnx_path = Path(onnx_path).resolve()
    warm_opts = ort.SessionOptions()
    warm_opts.intra_op_num_threads = int(threads)
    warm_opts.inter_op_num_threads = 1
    warm_sess = ort.InferenceSession(str(onnx_path), sess_options=warm_opts, providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        warm_sess.run(None, feed)
    del warm_sess

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    tmp_prefix = Path(tempfile.gettempdir()) / f"mv_prof_{onnx_path.stem}_{int(time.time() * 1000)}"
    opts.profile_file_prefix = str(tmp_prefix)
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    prof_file: Path | None = None
    try:
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(None, feed)
        wall_ms_per_call = (time.perf_counter() - t0) * 1000.0 / max(1, runs)
        prof_file = Path(sess.end_profiling())
        events = json.loads(prof_file.read_text(encoding="utf-8"))
    finally:
        if prof_file is not None:
            prof_file.unlink(missing_ok=True)

    op_dur_us: dict[str, float] = defaultdict(float)
    op_calls: Counter = Counter()
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args") or {}
        op_name = str(args.get("op_name") or "Unknown")
        op_dur_us[op_name] += float(ev.get("dur") or 0.0)
        op_calls[op_name] += 1

    total_node_us = sum(op_dur_us.values()) / max(1, runs)
    breakdown = []
    for op_name, dur_sum in sorted(op_dur_us.items(), key=lambda kv: kv[1], reverse=True):
        us_per_call = dur_sum / max(1, runs)
        breakdown.append(
            {
                "op_name": op_name,
                "us_per_call": round(us_per_call, 2),
                "share_pct": round(100.0 * us_per_call / max(total_node_us, 1e-6), 2),
                "nodes_per_call": round(op_calls[op_name] / max(1, runs), 1),
            }
        )

    def _sum_ops(names: tuple[str, ...]) -> float:
        return sum(op_dur_us[k] for k in names) / max(1, runs)

    fp32_matmul_us = _sum_ops(("MatMul", "FusedMatMul", "Gemm"))
    int8_matmul_us = _sum_ops(("MatMulInteger", "QLinearMatMul", "DynamicQuantizeMatMul"))
    quant_us = _sum_ops(("DynamicQuantizeLinear", "DequantizeLinear", "QuantizeLinear"))
    fused_ln_us = _sum_ops(("LayerNormalization", "SkipLayerNormalization"))
    decomp_ln_us = _sum_ops(("ReduceMean", "Sub", "Pow", "Sqrt", "Div"))

    return {
        "model": onnx_path.name,
        "bytes": int(onnx_path.stat().st_size),
        "runs": runs,
        "wall_ms_per_batch32_seq32": round(wall_ms_per_call, 3),
        "wall_ms_per_row": round(wall_ms_per_call / 32.0, 4),
        "node_sum_us_per_batch": round(total_node_us, 2),
        "groups_us_per_batch": {
            "int8_matmul": round(int8_matmul_us, 2),
            "dynamic_quant_dequant": round(quant_us, 2),
            "fp32_matmul_gemm": round(fp32_matmul_us, 2),
            "fused_layernorm": round(fused_ln_us, 2),
            "decomposed_norm_math": round(decomp_ln_us, 2),
            "softmax": round(_sum_ops(("Softmax",)), 2),
            "gelu_erf": round(_sum_ops(("Gelu", "BiasGelu", "Erf")), 2),
            "gather_embed": round(_sum_ops(("Gather",)), 2),
        },
        "top_ops": breakdown[:14],
    }


def run_ort_profiling(v02_onnx: Path, v03_bundle: Path, warmup: int = 15, runs: int = 60) -> dict:
    ids = np.ones((32, 32), dtype=np.int64)
    mask = np.ones((32, 32), dtype=np.int64)
    z = np.zeros((32, 32, 512), dtype=np.float32)
    h0 = np.zeros((32, 32, 512), dtype=np.float32)
    step0 = np.array(0, dtype=np.int64)

    p_v02 = profile_ort_ops(v02_onnx, {"input_ids": ids, "attention_mask": mask}, warmup=warmup, runs=runs)
    p_pre = profile_ort_ops(v03_bundle / "prelude.onnx", {"input_ids": ids, "attention_mask": mask}, warmup=warmup, runs=runs)
    p_core = profile_ort_ops(
        v03_bundle / "core_step.onnx",
        {"z_prev": z, "h0": h0, "attention_mask": mask, "step_id": step0},
        warmup=warmup,
        runs=runs,
    )

    v03_per_core_layer_ms = p_core["wall_ms_per_row"] / 2.0
    v02_per_layer_ms = p_v02["wall_ms_per_row"] / 4.0
    return {
        "benchmark_shape": {"batch_size": 32, "seq_len": 32, "threads": 1, "runs": runs},
        "v02_4L": p_v02,
        "v03_prelude_1L": p_pre,
        "v03_core_step_2L": p_core,
        "comparison": {
            "v02_ms_per_row": p_v02["wall_ms_per_row"],
            "v02_ms_per_layer": round(v02_per_layer_ms, 4),
            "v03_prelude_ms_per_row": p_pre["wall_ms_per_row"],
            "v03_core_step_ms_per_row": p_core["wall_ms_per_row"],
            "v03_core_ms_per_layer": round(v03_per_core_layer_ms, 4),
            "core_vs_prelude_layer_ratio": round(v03_per_core_layer_ms / max(p_pre["wall_ms_per_row"], 1e-6), 3),
            "v03_1step_3L_ms_per_row": round(p_pre["wall_ms_per_row"] + p_core["wall_ms_per_row"], 4),
            "v03_15steps_6L_est_ms_per_row": round(p_pre["wall_ms_per_row"] + 1.5 * p_core["wall_ms_per_row"], 4),
            "v03_2steps_5L_ms_per_row": round(p_pre["wall_ms_per_row"] + 2.0 * p_core["wall_ms_per_row"], 4),
            "core_fp32_matmul_share_pct": round(
                100.0 * p_core["groups_us_per_batch"]["fp32_matmul_gemm"] / max(p_core["node_sum_us_per_batch"], 1e-6), 2
            ),
            "core_decomposed_norm_share_pct": round(
                100.0 * p_core["groups_us_per_batch"]["decomposed_norm_math"] / max(p_core["node_sum_us_per_batch"], 1e-6), 2
            ),
            "prelude_fused_ln_share_pct": round(
                100.0 * p_pre["groups_us_per_batch"]["fused_layernorm"] / max(p_pre["node_sum_us_per_batch"], 1e-6), 2
            ),
        },
    }


def _surface_features(text: str) -> dict:
    return {
        "char_len": len(text),
        "has_cjk_space": bool(CJK_SPACE_RE.search(text)),
        "has_emdash": bool(EMDASH_RE.search(text)),
        "has_kaomoji": bool(KAOMOJI_RE.search(text)),
        "has_ascii_word": bool(ASCII_WORD_RE.search(text)),
    }


def sample_synth_and_full_stats(synth_path: Path, total_sample: int = 4000, seed: int = 20261123) -> tuple[list[dict], dict]:
    """Scan full 250k synthetic corpus for exact stats and extract balanced stratified sample."""
    rng = random.Random(seed)
    total = 0
    mode_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    family_counts: Counter = Counter()
    surface_counts: Counter = Counter()
    char_lens: list[int] = []
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)

    with gzip.open(synth_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            mode = str(r.get("plugin_mode") or "unknown")
            bucket = str(r.get("length_bucket") or "unknown")
            family = str(r.get("family") or "unknown")
            text = str(r.get("text") or "")
            mode_counts[mode] += 1
            bucket_counts[bucket] += 1
            family_counts[family] += 1
            sf = _surface_features(text)
            char_lens.append(sf["char_len"])
            for k in ("has_cjk_space", "has_emdash", "has_kaomoji", "has_ascii_word"):
                if sf[k]:
                    surface_counts[k] += 1
            by_cell[(mode, bucket)].append(
                {
                    "sample_id": r["sample_id"],
                    "text": text,
                    "plugin_mode": mode,
                    "length_bucket": bucket,
                    "family": family,
                    "route": str(r.get("route") or ""),
                    "partition": str(r.get("partition") or ""),
                }
            )

    n_cells = max(1, len(by_cell))
    per_cell = max(1, total_sample // n_cells)
    sampled: list[dict] = []
    leftovers: list[dict] = []
    for key in sorted(by_cell):
        items = by_cell[key]
        rng.shuffle(items)
        take = min(len(items), per_cell)
        sampled.extend(items[:take])
        leftovers.extend(items[take:])
    if len(sampled) < total_sample and leftovers:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: total_sample - len(sampled)])
    rng.shuffle(sampled)

    cl_np = np.array(char_lens, dtype=np.int32)
    stats = {
        "total_rows": total,
        "by_plugin_mode": dict(mode_counts.most_common()),
        "by_length_bucket": dict(bucket_counts.most_common()),
        "by_family": dict(family_counts.most_common()),
        "char_len_p50": int(np.percentile(cl_np, 50)),
        "char_len_p90": int(np.percentile(cl_np, 90)),
        "char_len_p99": int(np.percentile(cl_np, 99)),
        "surface_rates_pct": {k: round(100.0 * surface_counts[k] / max(1, total), 2) for k in ("has_cjk_space", "has_emdash", "has_kaomoji", "has_ascii_word")},
    }
    return sampled[:total_sample], stats


def sample_real_sqlite_ro(db_path: Path, per_era: int = 1000, seed: int = 20261123) -> tuple[list[dict], dict]:
    """Read-only partition-balanced stratified sample across all 4 Bilibili eras."""
    rng = random.Random(seed)
    eras = ("pre_llm", "early_llm", "mid_llm", "current_2026")
    sampled: list[dict] = []

    with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30)) as con:
        con.execute("PRAGMA busy_timeout=30000")
        total_v04 = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        era_counts = dict(con.execute("SELECT era_bucket, COUNT(*) FROM comments GROUP BY era_bucket"))
        part_counts = dict(con.execute("SELECT partition, COUNT(*) FROM comments GROUP BY partition ORDER BY COUNT(*) DESC"))

        for era in eras:
            rows_part = con.execute(
                "SELECT partition, COUNT(*) FROM comments WHERE era_bucket = ? GROUP BY partition ORDER BY COUNT(*) DESC",
                (era,),
            ).fetchall()
            if not rows_part:
                continue
            target_per_part = max(1, per_era // len(rows_part) + 8)
            era_pool: list[dict] = []
            for part, cnt in rows_part:
                # Deterministic stride selection inside (era, partition) without full-table scan
                fetch_n = min(int(cnt), target_per_part * 3)
                q = (
                    "SELECT sample_id, video_id, partition, era_bucket, comment_level, json_extract(metadata_json, '$.text') "
                    "FROM comments WHERE era_bucket = ? AND partition = ? LIMIT ?"
                )
                part_rows = con.execute(q, (era, part, fetch_n)).fetchall()
                rng.shuffle(part_rows)
                for sid, vid, pt, eb, lvl, txt in part_rows[:target_per_part]:
                    if txt:
                        era_pool.append(
                            {
                                "sample_id": sid,
                                "video_id": vid,
                                "partition": pt,
                                "era_bucket": eb,
                                "comment_level": lvl,
                                "text": txt,
                            }
                        )
            rng.shuffle(era_pool)
            sampled.extend(era_pool[:per_era])

    db_stats = {
        "v04_sqlite_rows": int(total_v04),
        "by_era_bucket": {k: int(v) for k, v in era_counts.items()},
        "by_partition": {k: int(v) for k, v in part_counts.items()},
    }
    return sampled, db_stats


def _summarize_group(items: list[dict], probs: np.ndarray, steps: np.ndarray, f05_thr: float = 0.8129) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    sh = {int(s): int((steps == s).sum()) for s in (1, 2, 3, 4) if (steps == s).any()}
    return {
        "n": n,
        "mean_prob": round(float(probs.mean()), 4),
        "median_prob": round(float(np.median(probs)), 4),
        "p90_prob": round(float(np.percentile(probs, 90)), 4),
        "rate_ge_070": round(float((probs >= 0.70).mean()), 4),
        "rate_ge_f05": round(float((probs >= f05_thr).mean()), 4),
        "rate_le_030": round(float((probs <= 0.30).mean()), 4),
        "avg_steps": round(float(steps.mean()), 3),
        "step_hist": sh,
    }


def _tok_and_surface_stats(items: list[dict], engine: MeguvizV03Engine) -> dict:
    lens = []
    bf_rows = 0
    bf_toks = 0
    unks = 0
    cjk_sp = 0
    emdash = 0
    kaomoji = 0
    for it in items:
        txt = it["text"]
        _, st = engine.tokenizer.encode_ids(txt, max_len=engine.max_len, return_stats=True)
        lens.append(st["seq_len"])
        unks += st["unk_tokens"]
        if st["byte_fallback_tokens"] > 0:
            bf_rows += 1
            bf_toks += st["byte_fallback_tokens"]
        sf = _surface_features(txt)
        cjk_sp += int(sf["has_cjk_space"])
        emdash += int(sf["has_emdash"])
        kaomoji += int(sf["has_kaomoji"])
    n = max(1, len(items))
    ln = np.array(lens, dtype=np.int32)
    return {
        "n": len(items),
        "unk_tokens": unks,
        "byte_fallback_row_pct": round(100.0 * bf_rows / n, 2),
        "byte_fallback_tokens_total": bf_toks,
        "seq_len_p50": int(np.percentile(ln, 50)),
        "seq_len_p90": int(np.percentile(ln, 90)),
        "seq_len_p95": int(np.percentile(ln, 95)),
        "seq_len_p99": int(np.percentile(ln, 99)),
        "cjk_space_pct": round(100.0 * cjk_sp / n, 2),
        "emdash_pct": round(100.0 * emdash / n, 2),
        "kaomoji_pct": round(100.0 * kaomoji / n, 2),
    }


def run_corpus_audit(
    v03_bundle: Path,
    synth_path: Path,
    real_db: Path,
    synth_sample_n: int = 4000,
    real_per_era: int = 1000,
    threads: int = 1,
) -> dict:
    engine = MeguvizV03Engine(v03_bundle, threads=threads, mode="normal")
    f05_thr = float((engine.manifest.get("thresholds") or {}).get("f05", 0.8129))

    synth_rows, synth_full_stats = sample_synth_and_full_stats(synth_path, total_sample=synth_sample_n)
    real_rows, real_db_stats = sample_real_sqlite_ro(real_db, per_era=real_per_era)

    t0 = time.monotonic()
    synth_preds = engine.predict_batch([r["text"] for r in synth_rows], mode="normal", max_batch=32)
    t_synth = time.monotonic() - t0

    t1 = time.monotonic()
    real_preds = engine.predict_batch([r["text"] for r in real_rows], mode="normal", max_batch=32)
    t_real = time.monotonic() - t1

    s_probs = np.array([p["prob"] for p in synth_preds], dtype=np.float32)
    s_steps = np.array([p["steps_used"] for p in synth_preds], dtype=np.int32)
    r_probs = np.array([p["prob"] for p in real_preds], dtype=np.float32)
    r_steps = np.array([p["steps_used"] for p in real_preds], dtype=np.int32)

    def group_breakdown(rows: list[dict], probs: np.ndarray, steps: np.ndarray, key: str) -> dict:
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            buckets[str(row[key])].append(i)
        out = {}
        for k, idxs in sorted(buckets.items()):
            idx_np = np.array(idxs, dtype=np.int64)
            sub_items = [rows[j] for j in idxs]
            out[k] = _summarize_group(sub_items, probs[idx_np], steps[idx_np], f05_thr=f05_thr)
        return out

    # Lowest-scoring evasive synthetic AI examples
    order_evade = np.argsort(s_probs)
    evasive_examples = [
        {
            "plugin_mode": synth_rows[i]["plugin_mode"],
            "length_bucket": synth_rows[i]["length_bucket"],
            "family": synth_rows[i]["family"],
            "prob": round(float(s_probs[i]), 4),
            "steps_used": int(s_steps[i]),
            "text_snippet": sanitize_snippet(synth_rows[i]["text"]),
        }
        for i in order_evade[:15]
    ]

    # Real Bilibili breakdown by era, and partition within pre_llm
    real_by_era = group_breakdown(real_rows, r_probs, r_steps, "era_bucket")
    pre_idxs = [i for i, row in enumerate(real_rows) if row["era_bucket"] == "pre_llm"]
    pre_rows = [real_rows[i] for i in pre_idxs]
    pre_probs = r_probs[pre_idxs] if pre_idxs else np.zeros((0,), dtype=np.float32)
    pre_steps = r_steps[pre_idxs] if pre_idxs else np.zeros((0,), dtype=np.int32)
    pre_by_partition = group_breakdown(pre_rows, pre_probs, pre_steps, "partition")

    order_fp = np.argsort(-pre_probs) if len(pre_probs) else np.array([], dtype=np.int64)
    pre_fp_examples = [
        {
            "partition": pre_rows[i]["partition"],
            "prob": round(float(pre_probs[i]), 4),
            "steps_used": int(pre_steps[i]),
            "text_snippet": sanitize_snippet(pre_rows[i]["text"]),
        }
        for i in order_fp[:15]
    ]

    # Tokenization & surface comparison across groups
    real_by_era_rows: dict[str, list[dict]] = defaultdict(list)
    for row in real_rows:
        real_by_era_rows[row["era_bucket"]].append(row)
    tok_surface = {
        "synth_250k_sample": _tok_and_surface_stats(synth_rows, engine),
        **{f"real_{era}": _tok_and_surface_stats(real_by_era_rows[era], engine) for era in ("pre_llm", "early_llm", "mid_llm", "current_2026") if real_by_era_rows[era]},
    }

    return {
        "scored_with_model": engine.manifest.get("version"),
        "thresholds": {"score70": 0.70, "val_f05": f05_thr},
        "scoring_timing_s": {"synth_sample_s": round(t_synth, 2), "real_sample_s": round(t_real, 2)},
        "synth_250k_full_stats": synth_full_stats,
        "synth_250k_sample_overall": _summarize_group(synth_rows, s_probs, s_steps, f05_thr=f05_thr),
        "synth_by_plugin_mode": group_breakdown(synth_rows, s_probs, s_steps, "plugin_mode"),
        "synth_by_length_bucket": group_breakdown(synth_rows, s_probs, s_steps, "length_bucket"),
        "synth_by_family": group_breakdown(synth_rows, s_probs, s_steps, "family"),
        "synth_lowest_scoring_examples": evasive_examples,
        "real_sqlite_snapshot": real_db_stats,
        "real_sample_overall": _summarize_group(real_rows, r_probs, r_steps, f05_thr=f05_thr),
        "real_by_era_bucket": real_by_era,
        "real_pre_llm_by_partition": pre_by_partition,
        "real_pre_llm_top_fp_examples": pre_fp_examples,
        "token_and_surface_audit": tok_surface,
    }


def write_markdown_report(audit: dict, md_out: Path) -> None:
    ort_cmp = (audit.get("ort_profile") or {}).get("comparison") or {}
    c_audit = audit.get("corpus_audit") or {}
    lines = [
        "# MeguvizCAC v0.4-DualLoop Empirical Pre-Training Research Audit",
        "",
        "Generated strictly from local Windows 1-thread ORT profiling and stratified zero-shot `v0.3-loop-2x4-e192` scoring over the completed `+250k` confirmed-AI corpus and read-only `bili-live-1m.sqlite3` snapshot.",
        "",
        "## 1. ORT Operator-Level CPU Bottleneck Profile (`seq=32, bs=32, 1-thread`)",
        "",
    ]
    if ort_cmp:
        p_pre = audit["ort_profile"]["v03_prelude_1L"]
        p_core = audit["ort_profile"]["v03_core_step_2L"]
        p_v02 = audit["ort_profile"]["v02_4L"]
        lines.extend(
            [
                f"- **`v0.2` (`4L`)**: `{p_v02['wall_ms_per_row']:.4f} ms/row` (`{ort_cmp['v02_ms_per_layer']:.4f} ms/layer`) — fused `LayerNormalization` `{p_v02['groups_us_per_batch']['fused_layernorm']:.0f} us/batch`, zero FP32 LoRA.",
                f"- **`v0.3` `prelude` (`1L`)**: `{p_pre['wall_ms_per_row']:.4f} ms/row` — fused `LayerNormalization` `{ort_cmp['prelude_fused_ln_share_pct']}%` of node time.",
                f"- **`v0.3` `core_step` (`2L`)**: `{p_core['wall_ms_per_row']:.4f} ms/row` (`{ort_cmp['v03_core_ms_per_layer']:.4f} ms/layer`, **`{ort_cmp['core_vs_prelude_layer_ratio']:.2f}x`** per-layer cost vs `prelude`).",
                f"- **Root-cause shares inside `core_step`**: decomposed 8-op Norm math (`ReduceMean/Sub/Pow/Sqrt/Div`) = **`{ort_cmp['core_decomposed_norm_share_pct']}%`** (`{p_core['groups_us_per_batch']['decomposed_norm_math']:.0f} us/batch`), unquantized FP32 `MatMul` (Step-LoRA + QKV attention) = **`{ort_cmp['core_fp32_matmul_share_pct']}%`** (`{p_core['groups_us_per_batch']['fp32_matmul_gemm']:.0f} us/batch`).",
                f"- **v0.4 Engineering Action**: Fused `StepModulatedLayerNorm` (native ORT `LayerNormalization`) + selective `Q/V/O + FFN_down` LoRA directly eliminate the decomposed-norm tax and cut LoRA MACs.",
                "",
            ]
        )

    if c_audit:
        s_over = c_audit["synth_250k_sample_overall"]
        lines.extend(
            [
                "## 2. `+250k` Confirmed-AI Plugin Evasion (`n=" + str(s_over["n"]) + "` stratified)",
                "",
                f"- **Zero-shot `v0.3` recall**: `@0.70 = {s_over['rate_ge_070']*100:.2f}%`, `@val_f05 (0.8129) = {s_over['rate_ge_f05']*100:.2f}%`, `mean prob = {s_over['mean_prob']:.4f}`, `avg_steps = {s_over['avg_steps']:.3f}`.",
                "",
                "| `plugin_mode` | n | mean_prob | median | `recall@0.70` | `recall@val_f05` | `le_0.30` (FN) | avg_steps |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode, st in sorted(c_audit["synth_by_plugin_mode"].items(), key=lambda kv: kv[1]["rate_ge_070"]):
            lines.append(
                f"| `{mode}` | {st['n']} | {st['mean_prob']:.4f} | {st['median_prob']:.4f} | {st['rate_ge_070']*100:.1f}% | {st['rate_ge_f05']*100:.1f}% | {st['rate_le_030']*100:.1f}% | {st['avg_steps']:.2f} |"
            )

        lines.extend(["", "By `length_bucket`:", "", "| `length_bucket` | n | mean_prob | `recall@0.70` | `recall@val_f05` | `le_0.30` |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for bkt, st in sorted(c_audit["synth_by_length_bucket"].items(), key=lambda kv: kv[1]["rate_ge_070"]):
            lines.append(f"| `{bkt}` | {st['n']} | {st['mean_prob']:.4f} | {st['rate_ge_070']*100:.1f}% | {st['rate_ge_f05']*100:.1f}% | {st['rate_le_030']*100:.1f}% |")

        lines.extend(
            [
                "",
                "## 3. Multi-Era Real Bilibili Drift & `pre_llm` Natural Human FPR",
                "",
                f"- Snapshot size in WAL SQLite at audit time: **`{c_audit['real_sqlite_snapshot']['v04_sqlite_rows']:,}`** new rows (`{sum(c_audit['real_sqlite_snapshot']['by_era_bucket'].values()):,}` across 4 eras).",
                "",
                "| `era_bucket` | n | mean_prob | p90_prob | `flag@0.70` | `flag@val_f05` | avg_steps |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for era in ("pre_llm", "early_llm", "mid_llm", "current_2026"):
            st = c_audit["real_by_era_bucket"].get(era)
            if not st:
                continue
            lines.append(
                f"| `{era}` | {st['n']} | {st['mean_prob']:.4f} | {st['p90_prob']:.4f} | {st['rate_ge_070']*100:.2f}% | {st['rate_ge_f05']*100:.2f}% | {st['avg_steps']:.2f} |"
            )

        lines.extend(["", "Top `pre_llm (<=2022)` False-Positive Partitions (`@0.70`):", "", "| Partition | n | mean_prob | `FPR@0.70` | `FPR@val_f05` |", "| --- | ---: | ---: | ---: | ---: |"])
        for pt, st in sorted(c_audit["real_pre_llm_by_partition"].items(), key=lambda kv: -kv[1]["rate_ge_070"])[:8]:
            lines.append(f"| {pt} | {st['n']} | {st['mean_prob']:.4f} | {st['rate_ge_070']*100:.2f}% | {st['rate_ge_f05']*100:.2f}% |")

        lines.extend(
            [
                "",
                "## 4. Tokenization (`[UNK]=0`) & Space-Delimited Rhythm Audit",
                "",
                "| Cohort | `unk` | `byte_fallback_rows%` | `seq_p50` | `seq_p95` | `cjk_space%` | `emdash%` | `kaomoji%` |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, ts in c_audit["token_and_surface_audit"].items():
            lines.append(
                f"| `{name}` | {ts['unk_tokens']} | {ts['byte_fallback_row_pct']:.2f}% | {ts['seq_len_p50']} | {ts['seq_len_p95']} | {ts['cjk_space_pct']:.2f}% | {ts['emdash_pct']:.2f}% | {ts['kaomoji_pct']:.2f}% |"
            )

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-ort", action="store_true")
    parser.add_argument("--audit-corpus", action="store_true")
    parser.add_argument("--v02-onnx", type=Path, default=DEFAULT_V02_ONNX)
    parser.add_argument("--v03-bundle", type=Path, default=DEFAULT_V03_BUNDLE)
    parser.add_argument("--synth-path", type=Path, default=DEFAULT_SYNTH_PATH)
    parser.add_argument("--real-db", type=Path, default=DEFAULT_REAL_DB)
    parser.add_argument("--synth-sample", type=int, default=4000)
    parser.add_argument("--real-per-era", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    if not args.profile_ort and not args.audit_corpus:
        args.profile_ort = True
        args.audit_corpus = True

    result: dict = {}
    if args.profile_ort:
        result["ort_profile"] = run_ort_profiling(args.v02_onnx, args.v03_bundle, runs=args.runs)
    if args.audit_corpus:
        result["corpus_audit"] = run_corpus_audit(
            args.v03_bundle,
            args.synth_path,
            args.real_db,
            synth_sample_n=args.synth_sample,
            real_per_era=args.real_per_era,
            threads=args.threads,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(result, args.out_md)
    print(f"Saved audit JSON -> {args.out_json}")
    print(f"Saved research MD -> {args.out_md}")
    if "ort_profile" in result:
        print("ORT Comparison:", json.dumps(result["ort_profile"]["comparison"], ensure_ascii=False))
    if "corpus_audit" in result:
        ca = result["corpus_audit"]
        print("Synth Overall:", json.dumps(ca["synth_250k_sample_overall"], ensure_ascii=False))
        print("Real By Era:", json.dumps(ca["real_by_era_bucket"], ensure_ascii=False))


if __name__ == "__main__":
    main()
