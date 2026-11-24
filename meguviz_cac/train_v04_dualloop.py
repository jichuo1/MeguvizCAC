"""Train, Born-Again EMA refine, 2-epoch QAT+SWA fine-tune, PAVA/Conformal calibrate & export MeguvizCAC v0.4-DualLoop."""

import argparse
from collections import Counter
import copy
import gzip
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .train_student import (
    CODE_LIST,
    CODE_TO_IDX,
    PRAG_TAGS,
    build_train_records,
    compute_eval_metrics,
    find_val_thresholds,
    fit_temperature,
    load_pairs,
    load_qwen_fused,
    load_rescored_corpus,
    sha256_file,
)
from .train_v03_loop import (
    DEFAULT_BACKBONE,
    DEEP_PRAG_IDXS,
    SURFACE_PRAG_IDXS,
    _seed_ascii_bytes_into_a,
    compute_weighted_svd_factors,
)
from .v04_corpus_prep import GROUP_TO_IDX, V04_GROUPS, assign_group_id
from .v04_runtime import (
    CHAR_MAX_BYTES,
    CHAR_PAD_ID,
    CHAR_VOCAB_SIZE,
    MeguvizV04Engine,
    apply_exit_policy_v04,
    apply_isotonic_knots,
    encode_char_bytes,
    encode_char_bytes_batch,
    fit_pava_isotonic,
    fit_split_conformal_3way,
    quantize_prelude_ordered_w4_w8,
    run_verify_v04,
)
from .v03_tokenizer import BASE_VOCAB_SIZE, ByteFallbackWordPieceTokenizer, HEAD_CHARS, MAX_LEN, TAIL_CHARS, prepare_text


def sinkhorn_knopp_np(logits: np.ndarray, iters: int = 10, eps: float = 1e-8) -> np.ndarray:
    """Project an (N, N) unconstrained matrix onto the Birkhoff doubly-stochastic manifold via Sinkhorn-Knopp."""
    a = np.asarray(logits, dtype=np.float64)
    mat = np.exp(a - np.max(a))
    for _ in range(iters):
        mat = mat / (mat.sum(axis=-1, keepdims=True) + eps)
        mat = mat / (mat.sum(axis=-2, keepdims=True) + eps)
    return mat


def compute_skewed_kl_np(p_student: np.ndarray, p_teacher: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Pure-NumPy Bernoulli DistiLLM skewed KL: KL(p_t || (1-alpha)*p_t + alpha*p_s)."""
    ps = np.clip(np.asarray(p_student, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pt = np.clip(np.asarray(p_teacher, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pm = np.clip((1.0 - alpha) * pt + alpha * ps, 1e-6, 1.0 - 1e-6)
    return pt * np.log(pt / pm) + (1.0 - pt) * np.log((1.0 - pt) / (1.0 - pm))


class GroupDROTracker:
    """Exponentiated-gradient GroupDRO subgroup weight tracker with explicit floor protection."""

    def __init__(self, num_groups: int = len(V04_GROUPS), eta: float = 0.05, ema_decay: float = 0.9, floor_share: float = 0.03):
        self.num_groups = int(num_groups)
        self.eta = float(eta)
        self.ema_decay = float(ema_decay)
        self.floor_share = float(floor_share)
        self.q = np.full(self.num_groups, 1.0 / self.num_groups, dtype=np.float64)
        self.ema_losses = np.zeros(self.num_groups, dtype=np.float64)

    def update(self, group_idxs: np.ndarray, sample_losses: np.ndarray) -> np.ndarray:
        """Update EMA loss per present subgroup and return normalized per-group weights >= floor_share."""
        g = np.asarray(group_idxs, dtype=np.int64)
        l = np.asarray(sample_losses, dtype=np.float64)
        for gid in range(self.num_groups):
            mask = g == gid
            if np.any(mask):
                mean_l = float(np.mean(l[mask]))
                if self.ema_losses[gid] == 0.0:
                    self.ema_losses[gid] = mean_l
                else:
                    self.ema_losses[gid] = self.ema_decay * self.ema_losses[gid] + (1.0 - self.ema_decay) * mean_l
                self.q[gid] *= math.exp(min(2.0, self.eta * self.ema_losses[gid]))
        self.q /= max(1e-12, float(self.q.sum()))
        floor_total = min(0.8, self.floor_share * self.num_groups)
        eff = (1.0 - floor_total) * self.q + (floor_total / self.num_groups)
        eff /= eff.sum()
        return eff


def compute_high_precision_metrics(y_true_score: np.ndarray, y_prob: np.ndarray, pos_cutoff: float = 70.0) -> dict:
    """Compute threshold-free PR-AUC, Best F0.5, Recall@P95% and Recall@P99% strictly in NumPy."""
    scores = np.asarray(y_true_score, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_bin = scores >= float(pos_cutoff)
    n_pos = int(y_bin.sum())
    if len(p) == 0 or n_pos == 0 or n_pos == len(p):
        return {"pr_auc": 0.0, "best_f05": 0.0, "recall_at_p95": 0.0, "recall_at_p99": 0.0, "t_p95": 0.95, "t_p99": 0.99}

    order = np.argsort(-p, kind="mergesort")
    p_s = p[order]
    y_s = y_bin[order].astype(np.float64)
    tp_cum = np.cumsum(y_s)
    fp_cum = np.cumsum(1.0 - y_s)

    # Evaluate at distinct score boundaries so ties never artificially inflate precision
    distinct = np.r_[np.flatnonzero(p_s[:-1] > p_s[1:]), len(p_s) - 1]
    tp = tp_cum[distinct]
    fp = fp_cum[distinct]
    thrs = p_s[distinct]
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / float(n_pos)

    rec_prev = np.r_[0.0, rec[:-1]]
    pr_auc = float(np.sum((rec - rec_prev) * prec))
    f05_arr = (1.25 * prec * rec) / np.maximum(0.25 * prec + rec, 1e-12)
    best_f05 = float(np.max(f05_arr))

    def _recall_at_prec(target_p: float) -> tuple[float, float]:
        hits = np.flatnonzero((prec >= target_p) & (tp >= 1.0))
        if len(hits) == 0:
            return 0.0, 0.99
        best_i = int(hits[np.argmax(rec[hits])])
        return float(rec[best_i]), float(thrs[best_i])

    r_p95, t_p95 = _recall_at_prec(0.95)
    r_p99, t_p99 = _recall_at_prec(0.99)
    return {
        "pr_auc": round(pr_auc, 4),
        "best_f05": round(best_f05, 4),
        "recall_at_p95": round(r_p95, 4),
        "recall_at_p99": round(r_p99, 4),
        "t_p95": round(t_p95, 4),
        "t_p99": round(t_p99, 4),
    }


def check_release_gates(metrics: dict) -> dict:
    """Evaluate all quantitative §2.4 + OOD precision release gates for MeguvizCAC v0.4-DualLoop."""
    ox = metrics.get("onnx_int8") or {}
    tb = metrics.get("test_bili_gold") or {}
    traps = (metrics.get("onnx_verify") or {}).get("hard_trap_gates") or {}
    unk = (metrics.get("onnx_verify") or {}).get("unk_tokens_total")
    edge = metrics.get("edge_w4_w8") or {}

    gates = {
        "speed_le_2p8_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 2.8 ms", "pass": (ox.get("compact_batch32_1thread_ms") or 999.0) <= 2.8},
        "spearman_ge_0870": {"value": tb.get("spearman"), "target": ">= 0.870", "pass": (tb.get("spearman") or 0.0) >= 0.870},
        "mae_le_9p5": {"value": tb.get("mae"), "target": "<= 9.5", "pass": (tb.get("mae") or 999.0) <= 9.5},
        "qat_parity_mae_le_0010": {"value": ox.get("mae_vs_pytorch"), "target": "<= 0.010", "pass": (ox.get("mae_vs_pytorch") or 999.0) <= 0.010},
        "ai_traps_ge_078": {"value": traps.get("ai_traps_min_prob"), "target": ">= 0.78", "pass": bool(traps.get("ai_traps_ge_078"))},
        "h03_tech_human_lt_035": {"value": traps.get("h03_tech_human_prob"), "target": "< 0.35", "pass": bool(traps.get("h03_lt_035"))},
        "structured_human_lt_035": {"value": traps.get("structured_human_max_prob"), "target": "< 0.35", "pass": bool(traps.get("structured_human_lt_035"))},
        "u004_ge_065": {"value": traps.get("u004_prob"), "target": ">= 0.65", "pass": bool(traps.get("u004_ge_065"))},
        "edge_w4_w8_ok": {"value": edge.get("ok"), "target": "True", "pass": bool(edge.get("ok"))},
        "zero_unk": {"value": unk, "target": "== 0", "pass": unk == 0},
    }
    return {"gates": gates, "passed_count": sum(1 for v in gates.values() if v["pass"]), "total_gates": len(gates), "all_passed": all(v["pass"] for v in gates.values())}


def _open_read(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _load_intent_cache(cache_path: Path | None) -> tuple[np.ndarray | None, dict[str, int]]:
    if cache_path is None or not cache_path.is_file():
        return None, {}
    z = np.load(cache_path, allow_pickle=True)
    vecs = np.asarray(z["vectors"], dtype=np.float32)
    sids = z["sample_ids"].tolist()
    idxs = z["intent_indices"].tolist()
    sid_map = {str(s): int(i) for s, i in zip(sids, idxs) if int(i) < len(vecs)}
    return vecs, sid_map


def _load_training_bundle(
    data_path: Path,
    qwen_path: Path | None,
    pairs_path: Path | None,
    intent_cache_path: Path | None,
    stage1_synth_cap: int = 15000,
    v04_synth_cap: int = -1,
    extra_pairs_paths: list[Path] | tuple[Path, ...] = (),
    seed: int = 20261124,
) -> tuple[list[dict], list[dict], list[dict], list[dict], np.ndarray | None]:
    """Load either unified v0.4 merged JSONL.GZ or Stage-1 JSONL.GZ without leakage."""
    rows: list[dict] = []
    with _open_read(data_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    is_unified = any(r.get("cohort") in ("synth_250k", "v04_1m") or "group_id" in r for r in rows[:2000])
    qwen_map = load_qwen_fused(qwen_path) if qwen_path and qwen_path.is_file() else {}
    intent_vecs, sid_to_intent = _load_intent_cache(intent_cache_path)

    eval_rows = [r for r in rows if r["split"] != "train" or r.get("cohort") == "user_confirmed"]
    forbidden = {prepare_text(r["text"]) for r in eval_rows}
    pairs = load_pairs(pairs_path, forbidden) if pairs_path and pairs_path.is_file() else []
    for ep_path in extra_pairs_paths:
        if ep_path is None or not ep_path.is_file():
            continue
        with _open_read(ep_path) as pf:
            for line in pf:
                if not line.strip():
                    continue
                p = json.loads(line)
                ai_t = prepare_text(p.get("ai_text") or "")
                hu_t = prepare_text(p.get("human_text") or "")
                if len(ai_t) < 10 or len(hu_t) < 10 or ai_t == hu_t:
                    continue
                if ai_t in forbidden or hu_t in forbidden:
                    continue
                pairs.append(
                    {
                        "ai_text": ai_t,
                        "human_text": hu_t,
                        "weight": float(p.get("weight", 0.5)),
                        "category": p.get("category", "qwen_twin"),
                    }
                )

    rng = random.Random(seed)
    train_items: list[dict] = []

    if is_unified:
        v04_synth_rows: list[dict] = []
        legacy_synth_rows: list[dict] = []
        other_rows: list[dict] = []
        for r in rows:
            if r["split"] != "train" or r.get("cohort") == "user_confirmed":
                continue
            if not r.get("has_score", True):
                continue
            w = float(r.get("train_weight", 1.0))
            if w <= 0.0:
                continue
            c = r.get("cohort", "")
            if c == "synth_250k":
                v04_synth_rows.append(r)
            elif c in ("deepseek", "other_models"):
                if len(r["text"]) >= 26 and float(r["score"]) >= 70.0:
                    legacy_synth_rows.append(r)
            else:
                other_rows.append(r)

        if stage1_synth_cap >= 0 and len(legacy_synth_rows) > stage1_synth_cap:
            legacy_synth_rows = rng.sample(legacy_synth_rows, stage1_synth_cap)
        if v04_synth_cap >= 0 and len(v04_synth_rows) > v04_synth_cap:
            v04_synth_rows = rng.sample(v04_synth_rows, v04_synth_cap)

        for r in (*other_rows, *legacy_synth_rows, *v04_synth_rows):
            sid = r["sample_id"]
            gid_str = r.get("group_id") or assign_group_id(r)
            prag = r.get("prag_vec")
            if not (isinstance(prag, list) and len(prag) == len(PRAG_TAGS)):
                prag = [-1.0] * len(PRAG_TAGS)
            train_items.append(
                {
                    "sample_id": sid,
                    "text": prepare_text(r["text"]),
                    "y_score": float(np.clip(float(r["score"]) / 100.0, 0.0, 1.0)),
                    "code_idx": CODE_TO_IDX.get(r.get("reason_code", ""), -100),
                    "weight": float(r.get("train_weight", 1.0)),
                    "prag_vec": [float(x) for x in prag],
                    "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["general_real"]),
                    "intent_idx": sid_to_intent.get(sid, -1),
                    "is_high_prio": bool(
                        r.get("score_source") == "hydrogen_xhigh_direct"
                        or sid in qwen_map
                        or r.get("hard_human_neg")
                        or gid_str in ("ultra_short_ai", "evasive_plugin_ai", "pre_llm_human", "structured_human_hard", "literary_boundary", "dsh_hard_slice")
                    ),
                }
            )
    else:
        raw_items = build_train_records(rows, synth_cap=max(0, stage1_synth_cap), qwen_map=qwen_map, seed=seed)
        row_by_sid = {r["sample_id"]: r for r in rows}
        for it in raw_items:
            src = row_by_sid.get(it["sample_id"], {})
            gid_str = assign_group_id(src) if src else "general_real"
            train_items.append(
                {
                    **it,
                    "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["general_real"]),
                    "intent_idx": sid_to_intent.get(it["sample_id"], -1),
                    "is_high_prio": bool(src.get("score_source") == "hydrogen_xhigh_direct" or it["sample_id"] in qwen_map or src.get("dsh")),
                }
            )

    return train_items, eval_rows, pairs, rows, intent_vecs


def run_dry(args) -> None:
    """Verify pure-NumPy v0.4 components, zero leakage, 12-group DRO tags, and zero-[UNK] tokenization locally."""
    t0 = time.monotonic()

    # 1. Pure-NumPy Sinkhorn doubly-stochastic invariant check
    rng = np.random.default_rng(20261124)
    logits = rng.normal(0.0, 2.0, size=(4, 4))
    ds = sinkhorn_knopp_np(logits, iters=12)
    row_err = float(np.max(np.abs(ds.sum(axis=-1) - 1.0)))
    col_err = float(np.max(np.abs(ds.sum(axis=-2) - 1.0)))
    assert row_err < 1e-5 and col_err < 1e-5, f"Sinkhorn error too high: {row_err}, {col_err}"

    # 2. GroupDRO invariant check
    dro = GroupDROTracker()
    g_w = dro.update(np.arange(len(V04_GROUPS)), rng.uniform(0.1, 0.8, size=len(V04_GROUPS)))
    assert abs(float(g_w.sum()) - 1.0) < 1e-6 and float(g_w.min()) >= 0.029

    # 3. Skewed KL sanity check
    kl_same = float(np.mean(compute_skewed_kl_np(np.array([0.2, 0.8]), np.array([0.2, 0.8]))))
    assert kl_same < 1e-6

    # 4. Release gate & high-precision metric evaluator sanity check
    hp_test = compute_high_precision_metrics(np.array([10.0, 20.0, 80.0, 90.0]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert hp_test["pr_auc"] == 1.0 and hp_test["recall_at_p95"] == 1.0 and hp_test["recall_at_p99"] == 1.0
    sample_gate_out = check_release_gates(
        {
            "onnx_int8": {"compact_batch32_1thread_ms": 2.4, "mae_vs_pytorch": 0.008},
            "test_bili_gold": {"spearman": 0.875, "mae": 9.1},
            "onnx_verify": {
                "unk_tokens_total": 0,
                "hard_trap_gates": {
                    "ai_traps_min_prob": 0.81,
                    "ai_traps_ge_078": True,
                    "h03_tech_human_prob": 0.29,
                    "h03_lt_035": True,
                    "structured_human_max_prob": 0.29,
                    "structured_human_lt_035": True,
                    "u004_prob": 0.68,
                    "u004_ge_065": True,
                },
            },
            "edge_w4_w8": {"ok": True},
        }
    )
    assert sample_gate_out["all_passed"] is True

    # 5. Dataset & tokenizer check
    train_items, eval_rows, pairs, all_rows, intent_vecs = _load_training_bundle(
        args.data.resolve(),
        args.qwen_data.resolve() if args.qwen_data else None,
        args.pairs_data.resolve() if args.pairs_data else None,
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=min(5000, args.stage1_synth_cap) if args.stage1_synth_cap != 0 else 0,
        v04_synth_cap=5000 if args.v04_synth_cap < 0 else min(5000, args.v04_synth_cap),
        extra_pairs_paths=[p.resolve() for p in args.extra_pairs],
    )
    tok_report: dict | str = "skipped_no_vocab"
    if args.vocab and args.vocab.is_file():
        tok = ByteFallbackWordPieceTokenizer(args.vocab.resolve())
        sample_rows = train_items[:4000]
        unks = 0
        bf = 0
        for item in sample_rows:
            _, st = tok.encode_ids(item["text"], max_len=MAX_LEN, return_stats=True)
            unks += st["unk_tokens"]
            bf += st["byte_fallback_tokens"]
        char_batch, char_lens = encode_char_bytes_batch([x["text"] for x in sample_rows[:32]], max_bytes=CHAR_MAX_BYTES)
        tok_report = {
            "vocab_size": tok.vocab_size,
            "sampled_rows": len(sample_rows),
            "unk_tokens": unks,
            "byte_fallback_tokens": bf,
            "char_rhythm_shape": list(char_batch.shape),
            "char_len_p50": int(np.percentile(char_lens, 50)),
        }

    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "math_invariants": {
                    "sinkhorn_row_err": round(row_err, 9),
                    "sinkhorn_col_err": round(col_err, 9),
                    "dro_min_weight": round(float(g_w.min()), 4),
                    "skewed_kl_identity": round(kl_same, 9),
                    "release_gates_selfcheck": sample_gate_out["all_passed"],
                },
                "total_loaded_rows": len(all_rows),
                "train_items_sampled": len(train_items),
                "eval_rows": len(eval_rows),
                "clean_pairs": len(pairs),
                "intent_cache_shape": list(intent_vecs.shape) if intent_vecs is not None else None,
                "sample_group_counts": dict(Counter(V04_GROUPS[x["group_idx"]] for x in train_items)),
                "tokenizer_and_char_check": tok_report,
                "elapsed_s": round(time.monotonic() - t0, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _export_fp32_and_int8(module, dummy_inputs: tuple, input_names: list[str], output_names: list[str], dynamic_axes: dict, fp32_path: Path, int8_path: Path) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(module, dummy_inputs, str(fp32_path), **export_kwargs)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


def train_and_export(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset, Sampler
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 进行 MeguvizCAC v0.4-DualLoop 训练。")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if int(args.batch_size) > 0:
        batch_size = int(args.batch_size)
        lr_scale = 1.0
    elif vram_gb >= 60.0:
        batch_size = 1024
        lr_scale = 1.0 if args.no_auto_lr_scale else min(2.0, math.sqrt(batch_size / 256.0))
    elif vram_gb >= 35.0:
        batch_size = 512
        lr_scale = 1.0 if args.no_auto_lr_scale else min(2.0, math.sqrt(batch_size / 256.0))
    else:
        batch_size = 256 if vram_gb >= 20.0 else 128
        lr_scale = 1.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Config] GPU={gpu_name} ({vram_gb:.1f} GB) | amp={amp_dtype} | batch_size={batch_size} | lr_scale={lr_scale:.2f}", flush=True)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    # Save and load ByteFallbackWordPieceTokenizer directly from AutoTokenizer assets
    hf_tok = AutoTokenizer.from_pretrained(args.backbone)
    tok_tmp = out_dir / "_tok_tmp"
    hf_tok.save_pretrained(tok_tmp)
    for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src_f = tok_tmp / fn
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / fn)
    if not (out_dir / "vocab.txt").is_file():
        vocab_items = sorted(hf_tok.get_vocab().items(), key=lambda kv: kv[1])
        (out_dir / "vocab.txt").write_text("\n".join(t for t, _ in vocab_items) + "\n", encoding="utf-8")
    shutil.rmtree(tok_tmp, ignore_errors=True)
    tokenizer = ByteFallbackWordPieceTokenizer(out_dir / "vocab.txt")

    # ------------------------------------------------------------------
    # v0.4-DualLoop Neural Architecture
    # ------------------------------------------------------------------
    class QATEmbedding(nn.Module):
        def __init__(self, num_embeddings: int, embedding_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
            self.qat_enabled = False

        def forward(self, input_ids):
            w = self.weight
            if self.qat_enabled:
                scale = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                w_q = torch.clamp(torch.round(w / scale), -128.0, 127.0) * scale
                w = w + (w_q - w).detach()
            return F.embedding(input_ids, w)

    class QATLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            self.qat_enabled = False
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x):
            w = self.weight
            if self.qat_enabled:
                scale_w = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                w_q = torch.clamp(torch.round(w / scale_w), -128.0, 127.0) * scale_w
                w = w + (w_q - w).detach()

                x_min = x.detach().amin().clamp(max=0.0)
                x_max = x.detach().amax().clamp(min=0.0)
                scale_x = ((x_max - x_min) / 255.0).clamp(min=1e-6)
                zp = torch.clamp(torch.round(-x_min / scale_x), 0.0, 255.0)
                x_q = (torch.clamp(torch.round(x / scale_x + zp), 0.0, 255.0) - zp) * scale_x
                x = x + (x_q - x).detach()
            return F.linear(x, w, self.bias)

    def copy_linear_(dst: QATLinear, src: nn.Linear) -> None:
        with torch.no_grad():
            dst.weight.copy_(src.weight)
            if dst.bias is not None and src.bias is not None:
                dst.bias.copy_(src.bias)

    class WeightedSVDEmbedding(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, max_len: int, eps: float):
            super().__init__()
            self.hidden_size = hidden_size
            self.word_emb_A = QATEmbedding(vocab_size, embed_dim)
            self.word_emb_P = QATLinear(embed_dim, hidden_size, bias=False)
            self.position_weight = nn.Parameter(torch.empty(max_len, hidden_size))
            self.token_type_bias = nn.Parameter(torch.zeros(hidden_size))
            self.ln_weight = nn.Parameter(torch.ones(hidden_size))
            self.ln_bias = nn.Parameter(torch.zeros(hidden_size))
            self.eps = float(eps)
            self.register_buffer("position_ids", torch.arange(max_len, dtype=torch.long).unsqueeze(0))

        def load_from_bert(self, bert_emb, a_full: np.ndarray, p_mat: np.ndarray) -> None:
            with torch.no_grad():
                self.word_emb_A.weight.copy_(torch.from_numpy(a_full))
                self.word_emb_P.weight.copy_(torch.from_numpy(p_mat.T))
                self.position_weight.copy_(bert_emb.position_embeddings.weight[:MAX_LEN])
                self.token_type_bias.copy_(bert_emb.token_type_embeddings.weight[0])
                self.ln_weight.copy_(bert_emb.LayerNorm.weight)
                self.ln_bias.copy_(bert_emb.LayerNorm.bias)

        def forward(self, input_ids):
            seq_len = input_ids.shape[1]
            word_vec = self.word_emb_P(self.word_emb_A(input_ids))
            pos_vec = F.embedding(self.position_ids[:, :seq_len], self.position_weight)
            x = word_vec + pos_vec + self.token_type_bias
            return F.layer_norm(x, (self.hidden_size,), self.ln_weight, self.ln_bias, self.eps)

    class CharRhythmStem(nn.Module):
        """Lightweight 1D UTF-8 byte Conv side-stem (~0.10 MB INT8) capturing spaces, kaomoji & code rhythm."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, 32, padding_idx=CHAR_PAD_ID)
            self.conv1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.proj = QATLinear(128, hidden_size)
            self.cls_gate = nn.Parameter(torch.tensor(-2.0))
            self.lat_gate = nn.Parameter(torch.tensor(-2.0))

        def forward(self, char_ids):
            emb = self.char_emb(char_ids)  # (B, L, 32)
            pad_mask = (char_ids != CHAR_PAD_ID).to(emb.dtype).unsqueeze(1)  # (B, 1, L)
            h = emb.transpose(1, 2)
            h = F.gelu(self.conv1(h)) * pad_mask
            h = F.gelu(self.conv2(h)) * pad_mask
            mean_p = h.sum(dim=-1) / pad_mask.sum(dim=-1).clamp(min=1.0)
            max_p = (h + (pad_mask - 1.0) * 10000.0).amax(dim=-1)
            return self.proj(torch.cat([mean_p, max_p], dim=-1))

    class StaticEncoderBlock(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float, drop_p: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)

            self.q_proj = QATLinear(hidden_size, hidden_size)
            self.k_proj = QATLinear(hidden_size, hidden_size)
            self.v_proj = QATLinear(hidden_size, hidden_size)
            self.o_proj = QATLinear(hidden_size, hidden_size)
            self.attn_ln = nn.LayerNorm(hidden_size, eps=eps)
            self.ffn_up = QATLinear(hidden_size, intermediate_size)
            self.ffn_down = QATLinear(intermediate_size, hidden_size)
            self.ffn_ln = nn.LayerNorm(hidden_size, eps=eps)
            self.drop = nn.Dropout(drop_p)
            self.use_sdpa = False

        def load_from_bert_layer(self, layer) -> None:
            copy_linear_(self.q_proj, layer.attention.self.query)
            copy_linear_(self.k_proj, layer.attention.self.key)
            copy_linear_(self.v_proj, layer.attention.self.value)
            copy_linear_(self.o_proj, layer.attention.output.dense)
            with torch.no_grad():
                self.attn_ln.weight.copy_(layer.attention.output.LayerNorm.weight)
                self.attn_ln.bias.copy_(layer.attention.output.LayerNorm.bias)
                self.ffn_ln.weight.copy_(layer.output.LayerNorm.weight)
                self.ffn_ln.bias.copy_(layer.output.LayerNorm.bias)
            copy_linear_(self.ffn_up, layer.intermediate.dense)
            copy_linear_(self.ffn_down, layer.output.dense)

        def forward(self, x, attn_bias_4d):
            bsz, seqlen, _ = x.shape
            q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            if self.use_sdpa:
                ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias_4d.to(q.dtype), dropout_p=self.drop.p if self.training else 0.0)
            else:
                probs = self.drop(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias_4d, dim=-1))
                ctx = torch.matmul(probs, v)
            ctx = ctx.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
            h = self.attn_ln(x + self.drop(self.o_proj(ctx)))
            ff = self.ffn_down(F.gelu(self.ffn_up(h)))
            return self.ffn_ln(h + self.drop(ff))

    class QKNormRMS(nn.Module):
        def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, num_heads, 1, head_dim))
            self.eps = eps

        def forward(self, x):
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * inv_rms * self.weight

    class StepIndexedLoRALinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, num_steps: int, lora_rank: int, lora_drop: float = 0.15):
            super().__init__()
            self.base = QATLinear(in_features, out_features)
            self.lora_A = nn.Parameter(torch.zeros(num_steps, in_features, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(num_steps, lora_rank, out_features))
            self.lora_drop = nn.Dropout(lora_drop)

        def load_base_from(self, src: nn.Linear) -> None:
            copy_linear_(self.base, src)

        def init_lora_from_target(self, step_idx: int, target: nn.Linear, scale: float = 1.0) -> None:
            with torch.no_grad():
                delta = (target.weight.detach().float() - self.base.weight.detach().float()) * float(scale)
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                r = min(self.lora_A.shape[2], len(s))
                sqrt_s = torch.sqrt(s[:r].clamp(min=0.0))
                self.lora_A[step_idx, :, :r].copy_(vh[:r, :].T * sqrt_s.unsqueeze(0))
                self.lora_B[step_idx, :r, :].copy_(sqrt_s.unsqueeze(1) * u[:, :r].T)

        def init_lora_normal(self, step_idx: int, std: float = 0.02) -> None:
            with torch.no_grad():
                nn.init.normal_(self.lora_A[step_idx], mean=0.0, std=std)
                self.lora_B[step_idx].zero_()

        def forward(self, x, step_id):
            out = self.base(x)
            h_lora = self.lora_drop(torch.matmul(x, self.lora_A[step_id]))
            return out + torch.matmul(h_lora, self.lora_B[step_id])

    class StepIndexedPlainLinear(nn.Module):
        """Quantized QATLinear ignoring step_id for selective Step-LoRA pruning."""

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.base = QATLinear(in_features, out_features)

        def load_base_from(self, src: nn.Linear) -> None:
            copy_linear_(self.base, src)

        def init_lora_from_target(self, step_idx: int, target: nn.Linear, scale: float = 1.0) -> None:
            pass

        def init_lora_normal(self, step_idx: int, std: float = 0.02) -> None:
            pass

        def forward(self, x, step_id):
            return self.base(x)

    class StepModulatedLayerNorm(nn.Module):
        """Fused native ONNX LayerNormalization followed by lightweight external step scale/shift."""

        def __init__(self, num_steps: int, hidden_size: int, eps: float):
            super().__init__()
            self.hidden_size = hidden_size
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.step_scale = nn.Parameter(torch.ones(num_steps, hidden_size))
            self.step_shift = nn.Parameter(torch.zeros(num_steps, hidden_size))

        def init_base_from(self, src: nn.LayerNorm) -> None:
            with torch.no_grad():
                self.weight.copy_(src.weight)
                self.bias.copy_(src.bias)
                self.step_scale.fill_(1.0)
                self.step_shift.zero_()

        def init_step_from(self, step_idx: int, src: nn.LayerNorm) -> None:
            with torch.no_grad():
                safe_w = torch.where(self.weight.abs() > 1e-4, self.weight, torch.ones_like(self.weight))
                scale_t = src.weight / safe_w
                self.step_scale[step_idx].copy_(scale_t)
                self.step_shift[step_idx].copy_(src.bias - scale_t * self.bias)

        def forward(self, x, step_id):
            y = F.layer_norm(x, (self.hidden_size,), self.weight, self.bias, self.eps)
            return y * self.step_scale[step_id] + self.step_shift[step_id]

    class CoreStepBlock(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            num_steps: int,
            lora_rank: int,
            eps: float,
            lora_targets: str = "qvo_down",
            use_diff_attn: bool = False,
            drop_p: float = 0.1,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)
            self.use_diff_attn = bool(use_diff_attn)

            def _make_proj(in_f: int, out_f: int, name: str):
                if lora_targets == "all" or name in ("q", "v", "o", "down"):
                    return StepIndexedLoRALinear(in_f, out_f, num_steps, lora_rank)
                return StepIndexedPlainLinear(in_f, out_f)

            self.q_proj = _make_proj(hidden_size, hidden_size, "q")
            self.k_proj = _make_proj(hidden_size, hidden_size, "k")
            self.v_proj = _make_proj(hidden_size, hidden_size, "v")
            self.o_proj = _make_proj(hidden_size, hidden_size, "o")
            self.q_norm = QKNormRMS(num_heads, self.head_dim)
            self.k_norm = QKNormRMS(num_heads, self.head_dim)
            self.attn_ln = StepModulatedLayerNorm(num_steps, hidden_size, eps)
            self.ffn_up = _make_proj(hidden_size, intermediate_size, "up")
            self.ffn_down = _make_proj(intermediate_size, hidden_size, "down")
            self.ffn_ln = StepModulatedLayerNorm(num_steps, hidden_size, eps)
            self.drop = nn.Dropout(drop_p)
            self.use_sdpa = False

            if self.use_diff_attn:
                self.diff_lambda = nn.Parameter(torch.full((num_steps, num_heads // 2, 1, 1), -6.0))

        def load_base_from_bert_layer(self, layer) -> None:
            self.q_proj.load_base_from(layer.attention.self.query)
            self.k_proj.load_base_from(layer.attention.self.key)
            self.v_proj.load_base_from(layer.attention.self.value)
            self.o_proj.load_base_from(layer.attention.output.dense)
            self.attn_ln.init_base_from(layer.attention.output.LayerNorm)
            self.ffn_up.load_base_from(layer.intermediate.dense)
            self.ffn_down.load_base_from(layer.output.dense)
            self.ffn_ln.init_base_from(layer.output.LayerNorm)

        def init_step_lora_from_layer(self, step_idx: int, layer, scale: float = 1.0) -> None:
            self.q_proj.init_lora_from_target(step_idx, layer.attention.self.query, scale=scale)
            self.k_proj.init_lora_from_target(step_idx, layer.attention.self.key, scale=scale)
            self.v_proj.init_lora_from_target(step_idx, layer.attention.self.value, scale=scale)
            self.o_proj.init_lora_from_target(step_idx, layer.attention.output.dense, scale=scale)
            self.ffn_up.init_lora_from_target(step_idx, layer.intermediate.dense, scale=scale)
            self.ffn_down.init_lora_from_target(step_idx, layer.output.dense, scale=scale)
            self.attn_ln.init_step_from(step_idx, layer.attention.output.LayerNorm)
            self.ffn_ln.init_step_from(step_idx, layer.output.LayerNorm)

        def init_step_lora_normal(self, step_idx: int, std: float = 0.02) -> None:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.ffn_up, self.ffn_down):
                mod.init_lora_normal(step_idx, std=std)

        def calibrate_qk_rms(self, x_sample) -> None:
            with torch.no_grad():
                bsz, seqlen, _ = x_sample.shape
                step0 = torch.tensor(0, dtype=torch.long, device=x_sample.device)
                q = self.q_proj(x_sample, step0).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x_sample, step0).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
                q_rms = torch.sqrt(q.pow(2).mean(dim=(0, 2), keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1e-6))
                k_rms = torch.sqrt(k.pow(2).mean(dim=(0, 2), keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1e-6))
                self.q_norm.weight.copy_(q_rms.expand_as(self.q_norm.weight))
                self.k_norm.weight.copy_(k_rms.expand_as(self.k_norm.weight))

        def forward(self, x, attn_bias_4d, step_id):
            bsz, seqlen, _ = x.shape
            q = self.q_norm(self.q_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2))
            k = self.k_norm(self.k_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2))
            v = self.v_proj(x, step_id).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            if self.use_sdpa and not self.use_diff_attn:
                ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias_4d.to(q.dtype), dropout_p=self.drop.p if self.training else 0.0)
            else:
                probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale + attn_bias_4d, dim=-1)
                if self.use_diff_attn:
                    p_pair = probs.view(bsz, self.num_heads // 2, 2, seqlen, seqlen)
                    p0, p1 = p_pair[:, :, 0], p_pair[:, :, 1]
                    lam = torch.sigmoid(self.diff_lambda[step_id]).unsqueeze(0) * 0.4
                    probs = torch.stack([p0 - lam * p1, p1 - lam * p0], dim=2).reshape(bsz, self.num_heads, seqlen, seqlen)
                probs = self.drop(probs)
                ctx = torch.matmul(probs, v)
            ctx = ctx.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
            h = self.attn_ln(x + self.drop(self.o_proj(ctx, step_id)), step_id)
            ff = self.ffn_down(F.gelu(self.ffn_up(h, step_id)), step_id)
            return self.ffn_ln(h + self.drop(ff), step_id)

    def _sinkhorn_torch(logits, iters: int = 10, eps: float = 1e-6):
        mat = torch.exp(logits - logits.amax())
        for _ in range(iters):
            mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
            mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
        return mat

    class MeguvizV04DualLoopModel(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            core_layers: int,
            num_loops: int,
            lora_rank: int,
            eps: float,
            num_latents: int = 4,
            mhc_streams: int = 4,
            use_char_stem: bool = True,
            lora_targets: str = "qvo_down",
            use_diff_attn: bool = False,
            emb_drop_p: float = 0.15,
            intent_dim: int = 512,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.core_layers = core_layers
            self.num_loops = num_loops
            self.num_latents = int(num_latents)
            self.mhc_streams = int(mhc_streams)
            self.use_char_stem = bool(use_char_stem)
            self.mhc_baked = False

            self.embeddings = WeightedSVDEmbedding(vocab_size, embed_dim, hidden_size, MAX_LEN, eps)
            self.emb_drop = nn.Dropout(emb_drop_p)
            self.char_stem = CharRhythmStem(hidden_size) if self.use_char_stem else None
            if self.num_latents > 0:
                self.latent_tokens = nn.Parameter(torch.randn(1, self.num_latents, hidden_size) * 0.02)
            self.prelude = StaticEncoderBlock(hidden_size, num_heads, intermediate_size, eps)

            self.inject_gate = nn.Parameter(torch.full((num_loops, hidden_size), -2.0))
            with torch.no_grad():
                self.inject_gate[0].fill_(-6.0)
            self.inject_norm = StepModulatedLayerNorm(num_loops, hidden_size, eps)

            self.core_blocks = nn.ModuleList(
                [
                    CoreStepBlock(
                        hidden_size,
                        num_heads,
                        intermediate_size,
                        num_loops,
                        lora_rank,
                        eps,
                        lora_targets=lora_targets,
                        use_diff_attn=use_diff_attn,
                    )
                    for _ in range(core_layers)
                ]
            )

            # DeepSeek mHC doubly-stochastic stream mixing (baked for zero-loop ONNX export)
            # Progressively damp deeper step residuals so late loops cannot drift off-manifold.
            damps = [0.0, -0.7, -1.5, -2.0, -2.0]
            if self.mhc_streams > 1:
                n_s = self.mhc_streams
                self.h_pre_logits = nn.Parameter(torch.zeros(num_loops, n_s))
                post_init = torch.zeros(num_loops, n_s)
                for s_i in range(num_loops):
                    post_init[s_i].fill_(damps[min(s_i, len(damps) - 1)])
                self.h_post_logits = nn.Parameter(post_init)
                self.h_out_logits = nn.Parameter(torch.zeros(num_loops, n_s))
                res_init = torch.full((num_loops, n_s, n_s), -3.0)
                for s_i in range(num_loops):
                    res_init[s_i].fill_diagonal_(3.0)
                self.h_res_logits = nn.Parameter(res_init)
                self.stream_bias = nn.Parameter(torch.zeros(1, n_s, 1, hidden_size))

                self.register_buffer("h_pre_baked", torch.full((num_loops, n_s), 1.0 / n_s))
                self.register_buffer("h_post_baked", torch.ones(num_loops, n_s))
                self.register_buffer("h_out_baked", torch.full((num_loops, n_s), 1.0 / n_s))
                self.register_buffer("h_res_baked", torch.eye(n_s).unsqueeze(0).repeat(num_loops, 1, 1))
            else:
                sg_vals = [2.0, 0.0, -1.5, -2.0, -2.0]
                sg_init = [sg_vals[min(s_i, len(sg_vals) - 1)] for s_i in range(num_loops)]
                self.step_delta_gate = nn.Parameter(torch.tensor(sg_init, dtype=torch.float32))

            # Coda inference readout
            self.coda_proj = QATLinear(hidden_size * 2, 256)
            self.coda_ln = nn.LayerNorm(256, eps=eps)
            self.coda_drop = nn.Dropout(0.1)
            self.score_head = QATLinear(256, 1)

            # Exit head conditioned on Coda feat + calibrated |logit| + step bias
            self.exit_feat_proj = nn.Linear(256, 32)
            self.exit_logit_proj = nn.Linear(1, 32, bias=False)
            self.exit_step_bias = nn.Parameter(torch.zeros(num_loops, 32))
            self.exit_out = nn.Linear(32, 1)
            with torch.no_grad():
                self.exit_out.bias.fill_(-1.0)

            # Training-only auxiliary heads (completely omitted from exported ONNX)
            self.code_head = nn.Linear(256, len(CODE_LIST))
            self.prag_head = nn.Linear(256, len(PRAG_TAGS))
            if self.num_latents >= 4:
                self.lat1_surf_prag = nn.Linear(hidden_size, len(SURFACE_PRAG_IDXS))
                self.lat2_intent_proj = nn.Linear(hidden_size, int(intent_dim))
                self.lat3_code = nn.Linear(hidden_size, len(CODE_LIST))
                self.lat3_deep_prag = nn.Linear(hidden_size, len(DEEP_PRAG_IDXS))

            self.register_buffer("step_temperatures", torch.ones(num_loops, dtype=torch.float32))
            self.register_buffer("step_idxs", torch.arange(num_loops, dtype=torch.long))

        def set_qat_enabled(self, enabled: bool) -> None:
            for mod in self.modules():
                if isinstance(mod, (QATLinear, QATEmbedding)):
                    mod.qat_enabled = bool(enabled)

        def set_sdpa(self, enabled: bool) -> None:
            self.prelude.use_sdpa = bool(enabled)
            for blk in self.core_blocks:
                blk.use_sdpa = bool(enabled) and not blk.use_diff_attn

        def bake_mhc_matrices(self) -> None:
            if self.mhc_streams <= 1:
                self.mhc_baked = True
                return
            with torch.no_grad():
                for s in range(self.num_loops):
                    self.h_pre_baked[s].copy_(F.softmax(self.h_pre_logits[s].float(), dim=-1))
                    self.h_post_baked[s].copy_(2.0 * torch.sigmoid(self.h_post_logits[s].float()))
                    self.h_out_baked[s].copy_(F.softmax(self.h_out_logits[s].float(), dim=-1))
                    self.h_res_baked[s].copy_(_sinkhorn_torch(self.h_res_logits[s].float(), iters=12))
            self.mhc_baked = True

        def forward_prelude(self, input_ids, attention_mask, char_ids=None):
            emb = self.emb_drop(self.embeddings(input_ids))
            rhythm = self.char_stem(char_ids) if (self.use_char_stem and char_ids is not None) else None

            if self.num_latents > 0:
                cls_emb = emb[:, :1, :]
                tok_emb = emb[:, 1:, :]
                # Pure broadcast without shape ops keeps dynamic ONNX batch clean
                lat_emb = cls_emb[:, :1, :1] * 0.0 + self.latent_tokens
                if rhythm is not None:
                    cls_emb = cls_emb + torch.sigmoid(self.char_stem.cls_gate) * rhythm.unsqueeze(1)
                    lat_emb = lat_emb + torch.sigmoid(self.char_stem.lat_gate) * rhythm.unsqueeze(1)
                full_emb = torch.cat([cls_emb, lat_emb, tok_emb], dim=1)
                lat_mask = attention_mask[:, :1].expand(-1, self.num_latents)
                core_mask = torch.cat([attention_mask[:, :1], lat_mask, attention_mask[:, 1:]], dim=1)
            else:
                if rhythm is not None:
                    emb = torch.cat([emb[:, :1, :] + torch.sigmoid(self.char_stem.cls_gate) * rhythm.unsqueeze(1), emb[:, 1:, :]], dim=1)
                full_emb = emb
                core_mask = attention_mask

            attn_bias = (1.0 - core_mask[:, None, None, :].to(full_emb.dtype)) * (-10000.0)
            h0 = self.prelude(full_emb, attn_bias)
            if self.mhc_streams > 1:
                z0 = h0.unsqueeze(1) + self.stream_bias.to(h0.dtype)
                return h0, z0, core_mask
            return h0, core_mask

        def forward_core_step(self, z_prev, h0, core_mask, step_id, apply_aux: bool = True):
            if self.mhc_streams > 1:
                if self.mhc_baked:
                    h_pre = self.h_pre_baked[step_id].to(z_prev.dtype)
                    h_post = self.h_post_baked[step_id].to(z_prev.dtype)
                    h_res = self.h_res_baked[step_id].to(z_prev.dtype)
                    h_out = self.h_out_baked[step_id].to(z_prev.dtype)
                else:
                    h_pre = F.softmax(self.h_pre_logits[step_id], dim=-1).to(z_prev.dtype)
                    h_post = (2.0 * torch.sigmoid(self.h_post_logits[step_id])).to(z_prev.dtype)
                    h_res = _sinkhorn_torch(self.h_res_logits[step_id], iters=10).to(z_prev.dtype)
                    h_out = F.softmax(self.h_out_logits[step_id], dim=-1).to(z_prev.dtype)

                z_t = z_prev.permute(0, 2, 3, 1)  # (B, S, H, N)
                z_res = torch.matmul(z_t, h_res.t()).permute(0, 3, 1, 2)  # (B, N, S, H)
                x_in = torch.matmul(z_t, h_pre.unsqueeze(-1)).squeeze(-1)  # (B, S, H)
            else:
                x_in = z_prev

            g_t = torch.sigmoid(self.inject_gate[step_id]).to(x_in.dtype)
            u_t = (1.0 - g_t) * x_in + g_t * self.inject_norm(h0, step_id)
            attn_bias = (1.0 - core_mask[:, None, None, :].to(u_t.dtype)) * (-10000.0)

            z_blk = u_t
            for blk in self.core_blocks:
                z_blk = blk(z_blk, attn_bias, step_id)

            if self.mhc_streams > 1:
                delta = z_blk - x_in
                z_next = z_res + h_post.view(1, self.mhc_streams, 1, 1) * delta.unsqueeze(1)
                z_coda = torch.matmul(z_next.permute(0, 2, 3, 1), h_out.unsqueeze(-1)).squeeze(-1)
            else:
                d_gate = torch.sigmoid(self.step_delta_gate[step_id]).to(z_blk.dtype)
                z_next = x_in + d_gate * (z_blk - x_in)
                z_coda = z_next

            mask_f = core_mask.to(z_coda.dtype).unsqueeze(-1)
            mean_pool = (z_coda * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            cls_vec = z_coda[:, 0, :]
            feat = self.coda_drop(F.gelu(self.coda_ln(self.coda_proj(torch.cat([cls_vec, mean_pool], dim=-1)))))

            raw_logit = self.score_head(feat).reshape(-1)
            calib_logit = raw_logit / self.step_temperatures[step_id]
            prob = torch.sigmoid(calib_logit)

            exit_h = F.gelu(self.exit_feat_proj(feat) + self.exit_logit_proj(calib_logit.abs().unsqueeze(-1)) + self.exit_step_bias[step_id])
            halt_prob = torch.sigmoid(self.exit_out(exit_h)).reshape(-1)

            if apply_aux:
                lat_aux = None
                if self.num_latents >= 4:
                    lat_aux = (
                        self.lat1_surf_prag(z_coda[:, 1, :]),
                        self.lat2_intent_proj(z_coda[:, 2, :]),
                        self.lat3_code(z_coda[:, 3, :]),
                        self.lat3_deep_prag(z_coda[:, 3, :]),
                    )
                return z_next, prob, calib_logit, halt_prob, self.code_head(feat), self.prag_head(feat), lat_aux
            return z_next, prob, calib_logit, halt_prob

    class PreludeExportWrapper(nn.Module):
        def __init__(self, m: MeguvizV04DualLoopModel):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, char_ids=None):
            return self.m.forward_prelude(input_ids, attention_mask, char_ids=char_ids)

    class CoreStepExportWrapper(nn.Module):
        def __init__(self, m: MeguvizV04DualLoopModel):
            super().__init__()
            self.m = m

        def forward(self, z_prev, h0, attention_mask, step_id):
            return self.m.forward_core_step(z_prev, h0, attention_mask, step_id, apply_aux=False)

    # ------------------------------------------------------------------
    # Load corpus & pre-tokenize
    # ------------------------------------------------------------------
    train_items_raw, eval_rows, pairs, _, intent_vecs = _load_training_bundle(
        args.data.resolve(),
        args.qwen_data.resolve() if args.qwen_data else None,
        args.pairs_data.resolve() if args.pairs_data else None,
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
        extra_pairs_paths=[p.resolve() for p in args.extra_pairs],
    )
    intent_vecs_t = torch.from_numpy(intent_vecs).to(device) if intent_vecs is not None and len(intent_vecs) else None

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(480, len(train_items_raw)))
        pairs = pairs[:32]

    # Zero-PII domain-coverage augmentation (~3% reply prefixes + ~3% benign URLs on definite humans)
    if not args.no_domain_aug and not args.smoke_test:
        rng_aug = random.Random(20261128)
        benign_urls = (
            " https://pan.quark.cn/s/demo1234",
            " https://github.com/example/repo",
            " https://b23.tv/BV1demo0001",
        )
        human_grp_set = {GROUP_TO_IDX["pre_llm_human"], GROUP_TO_IDX["structured_human_hard"]}
        for item in train_items_raw:
            txt = item["text"]
            if rng_aug.random() < 0.03 and not txt.startswith(("回复", "回覆")):
                prefix = "回复 @用户 : " if rng_aug.random() < 0.65 else "回复 :"
                txt = prepare_text(prefix + txt)
            if (item["group_idx"] in human_grp_set or item["y_score"] <= 0.12) and rng_aug.random() < 0.03 and "http" not in txt:
                txt = prepare_text(txt + rng_aug.choice(benign_urls))
            item["text"] = txt

    eff_emb_drop = float(args.emb_drop) if float(args.emb_drop) >= 0.0 else (0.15 if args.embed_dim >= 192 else 0.10)

    train_counts = np.zeros(BASE_VOCAB_SIZE, dtype=np.float64)
    for item in train_items_raw:
        ids = tokenizer.encode_ids(item["text"], max_len=MAX_LEN)
        cids = encode_char_bytes(item["text"], max_bytes=CHAR_MAX_BYTES)
        item["ids"] = ids
        item["cids"] = cids
        item["length"] = len(ids)
        for tid in ids:
            if tid < BASE_VOCAB_SIZE:
                train_counts[tid] += 1.0

    # Pre-tokenize ALL minimal & twin pairs once so zero Python tokenization happens inside the GPU loop
    pairs_pre = [
        {
            "ai_ids": tokenizer.encode_ids(p["ai_text"], max_len=MAX_LEN),
            "hu_ids": tokenizer.encode_ids(p["human_text"], max_len=MAX_LEN),
            "ai_cids": encode_char_bytes(p["ai_text"], max_bytes=CHAR_MAX_BYTES),
            "hu_cids": encode_char_bytes(p["human_text"], max_bytes=CHAR_MAX_BYTES),
            "weight": float(p.get("weight", 1.0)),
        }
        for p in pairs
    ]

    bert_src = AutoModel.from_pretrained(args.backbone)
    cfg = bert_src.config
    a_full, p_mat = compute_weighted_svd_factors(
        bert_src.embeddings.word_embeddings.weight.detach().cpu().numpy(),
        train_counts=train_counts,
        embed_dim=args.embed_dim,
        unk_id=tokenizer.unk_id,
    )
    a_full = _seed_ascii_bytes_into_a(a_full, tokenizer, tokenizer.byte_offset)
    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 512

    def build_model() -> MeguvizV04DualLoopModel:
        mdl = MeguvizV04DualLoopModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            core_layers=args.core_layers,
            num_loops=args.num_loops,
            lora_rank=args.lora_rank,
            eps=cfg.layer_norm_eps,
            num_latents=args.num_latents,
            mhc_streams=args.mhc_streams,
            use_char_stem=not args.no_char_stem,
            lora_targets=args.lora_targets,
            use_diff_attn=args.diff_attn,
            emb_drop_p=eff_emb_drop,
            intent_dim=intent_dim,
        )
        mdl.embeddings.load_from_bert(bert_src.embeddings, a_full, p_mat)
        mdl.prelude.load_from_bert_layer(bert_src.encoder.layer[0])
        src_layers = bert_src.encoder.layer
        mdl.inject_norm.init_base_from(bert_src.embeddings.LayerNorm)

        if args.core_layers == 2:
            mdl.core_blocks[0].load_base_from_bert_layer(src_layers[1])
            mdl.core_blocks[1].load_base_from_bert_layer(src_layers[2])
            if args.num_loops >= 2:
                mdl.core_blocks[0].init_step_lora_from_layer(1, src_layers[2], scale=0.25)
                mdl.core_blocks[1].init_step_lora_from_layer(1, src_layers[3], scale=1.0)
            for s in range(2, args.num_loops):
                mdl.core_blocks[0].init_step_lora_normal(s, std=0.02)
                mdl.core_blocks[1].init_step_lora_normal(s, std=0.02)
        elif args.core_layers == 1:
            mdl.core_blocks[0].load_base_from_bert_layer(src_layers[1])
            if args.num_loops >= 2:
                mdl.core_blocks[0].init_step_lora_from_layer(1, src_layers[2], scale=1.0)
            if args.num_loops >= 3:
                mdl.core_blocks[0].init_step_lora_from_layer(2, src_layers[3], scale=1.0)
            for s in range(3, args.num_loops):
                mdl.core_blocks[0].init_step_lora_normal(s, std=0.02)
        else:
            raise ValueError(f"--core-layers must be 1 or 2, got {args.core_layers}")

        mdl = mdl.to(device)
        # Empirical per-head QK RMS calibration
        sample_enc = [x["ids"] for x in train_items_raw[:32]]
        max_s = max(len(x) for x in sample_enc)
        s_ids = torch.full((len(sample_enc), max_s), tokenizer.pad_id, dtype=torch.long, device=device)
        s_mask = torch.zeros((len(sample_enc), max_s), dtype=torch.long, device=device)
        for i, seq in enumerate(sample_enc):
            s_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            s_mask[i, : len(seq)] = 1
        s_char = None
        if not args.no_char_stem:
            c_np, _ = encode_char_bytes_batch([x["text"] for x in train_items_raw[:32]], max_bytes=CHAR_MAX_BYTES)
            s_char = torch.from_numpy(c_np).to(device)
        mdl.eval()
        with torch.no_grad():
            pre = mdl.forward_prelude(s_ids, s_mask, s_char)
            h_cal = pre[0]
            c_mask = pre[-1]
            attn_b = (1.0 - c_mask[:, None, None, :].to(h_cal.dtype)) * (-10000.0)
            step0 = torch.tensor(0, dtype=torch.long, device=device)
            for blk in mdl.core_blocks:
                blk.calibrate_qk_rms(h_cal)
                h_cal = blk(h_cal, attn_b, step0)
        return mdl

    mdl = build_model()
    del bert_src
    torch.cuda.empty_cache()

    class TokenDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    class BucketSampler(Sampler):
        def __init__(self, lengths: list[int], bs: int, seed: int = 20261124):
            self.lengths = lengths
            self.bs = bs
            self.seed = seed
            self.epoch = 0

        def __iter__(self):
            rng_b = random.Random(self.seed + self.epoch)
            self.epoch += 1
            indices = list(range(len(self.lengths)))
            rng_b.shuffle(indices)
            batches = []
            mega = max(self.bs * 16, 2048)
            for st in range(0, len(indices), mega):
                chunk = indices[st : st + mega]
                chunk.sort(key=lambda i: self.lengths[i])
                for bst in range(0, len(chunk), self.bs):
                    batches.append(chunk[bst : bst + self.bs])
            rng_b.shuffle(batches)
            for b in batches:
                yield b

        def __len__(self):
            return (len(self.lengths) + self.bs - 1) // self.bs

    def collate_train(batch):
        max_seq = max(x["length"] for x in batch)
        max_ch = max(8, max(len(x["cids"]) for x in batch))
        ids = np.full((len(batch), max_seq), tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), max_seq), dtype=np.int64)
        cids = np.full((len(batch), max_ch), CHAR_PAD_ID, dtype=np.int64)
        for i, x in enumerate(batch):
            n = x["length"]
            ids[i, :n] = x["ids"]
            mask[i, :n] = 1
            nc = len(x["cids"])
            cids[i, :nc] = x["cids"]
        return (
            torch.from_numpy(ids),
            torch.from_numpy(mask),
            torch.from_numpy(cids),
            torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
            torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
            torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
            torch.tensor([x["prag_vec"] for x in batch], dtype=torch.float32),
            torch.tensor([x["group_idx"] for x in batch], dtype=torch.long),
            torch.tensor([x["intent_idx"] for x in batch], dtype=torch.long),
        )

    def make_loader(items: list[dict]) -> DataLoader:
        nw = 4 if (os.name != "nt" and not args.smoke_test) else 0
        kw = {"batch_sampler": BucketSampler([x["length"] for x in items], batch_size), "collate_fn": collate_train, "num_workers": nw, "pin_memory": True}
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 4
        return DataLoader(TokenDataset(items), **kw)

    def subset_balanced_prio(items: list[dict], cap: int, seed_s: int) -> list[dict]:
        if cap <= 0 or len(items) <= cap:
            return items
        rng_sub = random.Random(seed_s)
        by_g: dict[int, list[dict]] = {g: [] for g in range(len(V04_GROUPS))}
        for x in items:
            by_g[x["group_idx"]].append(x)
        quota = max(1, cap // len(V04_GROUPS))
        selected: list[dict] = []
        leftover: list[dict] = []
        for g in sorted(by_g):
            bucket = by_g[g]
            hi = [x for x in bucket if x["is_high_prio"]]
            lo = [x for x in bucket if not x["is_high_prio"]]
            rng_sub.shuffle(hi)
            rng_sub.shuffle(lo)
            ordered = hi + lo
            take = min(len(ordered), quota)
            selected.extend(ordered[:take])
            leftover.extend(ordered[take:])
        if len(selected) < cap and leftover:
            rng_sub.shuffle(leftover)
            selected.extend(leftover[: cap - len(selected)])
        rng_sub.shuffle(selected)
        return selected[:cap]

    def pre_encode_eval(subset: list[dict]) -> tuple[list[list[int]], list[str], np.ndarray]:
        enc = [tokenizer.encode_ids(r["text"], max_len=MAX_LEN) for r in subset]
        txts = [r["text"] for r in subset]
        scores = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return enc, txts, scores

    val_gold = [r for r in eval_rows if r["split"] == "validation" and r["score_source"] == "hydrogen_xhigh_direct"]
    val_dsh_gold = [r for r in val_gold if r["dsh"]]
    test_gold = [r for r in eval_rows if r["split"] == "test" and r["score_source"] == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r["dsh"]]
    test_bili_all = [r for r in eval_rows if r["cohort"] == "bili" and r["split"] == "test"]
    test_dsh_all = [r for r in test_bili_all if r["dsh"]]
    probe_rows = [r for r in eval_rows if r["cohort"] == "user_confirmed"]

    val_enc, val_txts, val_scores = pre_encode_eval(val_gold)
    val_dsh_enc, val_dsh_txts, val_dsh_scores = pre_encode_eval(val_dsh_gold)
    test_enc, test_txts, test_scores = pre_encode_eval(test_gold)
    test_dsh_enc, test_dsh_txts, test_dsh_scores = pre_encode_eval(test_dsh_gold)
    test_all_enc, test_all_txts, test_all_scores = pre_encode_eval(test_bili_all)
    test_dsh_all_enc, test_dsh_all_txts, test_dsh_all_scores = pre_encode_eval(test_dsh_all)
    probe_enc, probe_txts, probe_scores = pre_encode_eval(probe_rows)

    def predict_all_steps_pt(model_eval: MeguvizV04DualLoopModel, enc_ids: list[list[int]], texts: list[str], bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
        model_eval.eval()
        num_t = model_eval.num_loops
        n_total = len(enc_ids)
        logits_out = np.zeros((num_t, n_total), dtype=np.float32)
        halts_out = np.zeros((num_t, n_total), dtype=np.float32)
        if n_total == 0:
            return logits_out, halts_out
        with torch.no_grad():
            for st in range(0, n_total, bs):
                chunk = enc_ids[st : st + bs]
                max_seq = max(len(s) for s in chunk)
                ids = torch.full((len(chunk), max_seq), tokenizer.pad_id, dtype=torch.long, device=device)
                mask = torch.zeros((len(chunk), max_seq), dtype=torch.long, device=device)
                for i, seq in enumerate(chunk):
                    ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    mask[i, : len(seq)] = 1
                c_t = None
                if model_eval.use_char_stem:
                    c_np, _ = encode_char_bytes_batch(texts[st : st + bs], max_bytes=CHAR_MAX_BYTES)
                    c_t = torch.from_numpy(c_np).to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pre = model_eval.forward_prelude(ids, mask, c_t)
                    if model_eval.mhc_streams > 1:
                        h0, z, c_mask = pre
                    else:
                        h0, c_mask = pre
                        z = h0
                    for s in range(num_t):
                        z, _, c_logit, h_prob, _, _, _ = model_eval.forward_core_step(z, h0, c_mask, model_eval.step_idxs[s], apply_aux=True)
                        logits_out[s, st : st + len(chunk)] = c_logit.float().cpu().numpy()
                        halts_out[s, st : st + len(chunk)] = h_prob.float().cpu().numpy()
        return logits_out, halts_out

    def _skewed_kl_torch(p_s, y_t, alpha: float = 0.1):
        ps = p_s.float().clamp(1e-5, 1.0 - 1e-5)
        pt = y_t.float().clamp(1e-5, 1.0 - 1e-5)
        pm = ((1.0 - alpha) * pt + alpha * ps).clamp(1e-5, 1.0 - 1e-5)
        return pt * torch.log(pt / pm) + (1.0 - pt) * torch.log((1.0 - pt) / (1.0 - pm))

    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_code = nn.CrossEntropyLoss(ignore_index=-100)
    surf_idx_t = torch.tensor(SURFACE_PRAG_IDXS, dtype=torch.long, device=device)
    deep_idx_t = torch.tensor(DEEP_PRAG_IDXS, dtype=torch.long, device=device)

    prag_step_weights = []
    for s in range(args.num_loops):
        alpha_s = s / max(1, args.num_loops - 1)
        wv = np.ones(len(PRAG_TAGS), dtype=np.float32)
        for idx in SURFACE_PRAG_IDXS:
            wv[idx] = 1.3 - 0.6 * alpha_s
        for idx in DEEP_PRAG_IDXS:
            wv[idx] = 0.5 + 1.0 * alpha_s
        prag_step_weights.append(torch.tensor(wv, dtype=torch.float32, device=device))

    class TorchGroupDRO:
        """Zero-sync pure-GPU GroupDRO subgroup weight tracker eliminating per-step host sync stalls."""

        def __init__(self, num_groups: int = len(V04_GROUPS), eta: float = 0.05, ema_decay: float = 0.9, floor_share: float = 0.03):
            self.num_groups = int(num_groups)
            self.eta = float(eta)
            self.ema_decay = float(ema_decay)
            self.floor_share = float(floor_share)
            self.q = torch.full((self.num_groups,), 1.0 / self.num_groups, dtype=torch.float32, device=device)
            self.ema_losses = torch.zeros(self.num_groups, dtype=torch.float32, device=device)
            self.initialized = torch.zeros(self.num_groups, dtype=torch.bool, device=device)

        @torch.no_grad()
        def update(self, group_idxs, sample_losses):
            g = group_idxs.long()
            l = sample_losses.detach().float()
            counts = torch.bincount(g, minlength=self.num_groups).float()
            sums = torch.zeros(self.num_groups, dtype=torch.float32, device=l.device)
            sums.scatter_add_(0, g, l)
            present = counts > 0
            mean_l = torch.where(present, sums / counts.clamp(min=1.0), self.ema_losses)
            new_ema = torch.where(self.initialized, self.ema_decay * self.ema_losses + (1.0 - self.ema_decay) * mean_l, mean_l)
            self.ema_losses = torch.where(present, new_ema, self.ema_losses)
            self.initialized |= present
            self.q = torch.where(present, self.q * torch.exp((self.eta * self.ema_losses).clamp(max=2.0)), self.q)
            self.q = self.q / self.q.sum().clamp(min=1e-12)
            floor_total = min(0.8, self.floor_share * self.num_groups)
            eff = (1.0 - floor_total) * self.q + (floor_total / self.num_groups)
            return eff / eff.sum()

    def _collate_pre_pairs(p_batch: list[dict]):
        bs_p = len(p_batch)
        max_ai = max(len(x["ai_ids"]) for x in p_batch)
        max_hu = max(len(x["hu_ids"]) for x in p_batch)
        ai_ids = np.full((bs_p, max_ai), tokenizer.pad_id, dtype=np.int64)
        ai_mask = np.zeros((bs_p, max_ai), dtype=np.int64)
        hu_ids = np.full((bs_p, max_hu), tokenizer.pad_id, dtype=np.int64)
        hu_mask = np.zeros((bs_p, max_hu), dtype=np.int64)
        ai_c = hu_c = None
        if mdl.use_char_stem:
            mc_ai = max(8, max(len(x["ai_cids"]) for x in p_batch))
            mc_hu = max(8, max(len(x["hu_cids"]) for x in p_batch))
            ai_c = np.full((bs_p, mc_ai), CHAR_PAD_ID, dtype=np.int64)
            hu_c = np.full((bs_p, mc_hu), CHAR_PAD_ID, dtype=np.int64)
        for i, x in enumerate(p_batch):
            na, nh = len(x["ai_ids"]), len(x["hu_ids"])
            ai_ids[i, :na] = x["ai_ids"]
            ai_mask[i, :na] = 1
            hu_ids[i, :nh] = x["hu_ids"]
            hu_mask[i, :nh] = 1
            if ai_c is not None:
                ai_c[i, : len(x["ai_cids"])] = x["ai_cids"]
                hu_c[i, : len(x["hu_cids"])] = x["hu_cids"]
        return (
            torch.from_numpy(ai_ids).to(device, non_blocking=True),
            torch.from_numpy(ai_mask).to(device, non_blocking=True),
            torch.from_numpy(hu_ids).to(device, non_blocking=True),
            torch.from_numpy(hu_mask).to(device, non_blocking=True),
            torch.from_numpy(ai_c).to(device, non_blocking=True) if ai_c is not None else None,
            torch.from_numpy(hu_c).to(device, non_blocking=True) if hu_c is not None else None,
            torch.tensor([x["weight"] for x in p_batch], dtype=torch.float32, device=device),
        )

    dro_tracker = TorchGroupDRO()
    mdl.set_sdpa(torch.cuda.is_available())
    orig_prelude_fn = mdl.forward_prelude
    orig_core_fn = mdl.forward_core_step

    def _set_compiled_hotpaths(enable: bool) -> None:
        if enable and hasattr(torch, "compile"):
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            mdl.forward_prelude = torch.compile(orig_prelude_fn, dynamic=True)
            mdl.forward_core_step = torch.compile(orig_core_fn, dynamic=True)
        else:
            mdl.forward_prelude = orig_prelude_fn
            mdl.forward_core_step = orig_core_fn

    def run_phase(
        items: list[dict],
        num_epochs: int,
        lr: float,
        qat: bool,
        phase_name: str,
        ema_teacher: MeguvizV04DualLoopModel | None = None,
        update_ema: bool = False,
        use_swa: bool = False,
        max_steps_cap: int = 0,
    ) -> dict[str, torch.Tensor] | None:
        if num_epochs <= 0 or not items:
            return None
        mdl.set_qat_enabled(qat)
        _set_compiled_hotpaths(bool(args.compile and not args.smoke_test))
        loader = make_loader(items)
        eff_lr = float(lr) * lr_scale
        try:
            opt = torch.optim.AdamW(mdl.parameters(), lr=eff_lr, weight_decay=0.01, fused=True)
        except TypeError:
            opt = torch.optim.AdamW(mdl.parameters(), lr=eff_lr, weight_decay=0.01)
        total_steps = min(max_steps_cap, len(loader) * num_epochs) if max_steps_cap > 0 else max(1, len(loader) * num_epochs)
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.08)), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        rng_p = random.Random(20261124 + len(phase_name))
        pair_bs = 24 if batch_size >= 512 else 12
        log_every = max(50, len(loader) // 4)
        t_p = time.monotonic()
        global_step = 0
        swa_state: dict[str, torch.Tensor] | None = None
        swa_count = 0

        for ep in range(num_epochs):
            mdl.train()
            running_loss_t = torch.zeros((), dtype=torch.float32, device=device)
            n_seen = 0
            for step, (ids, mask, cids, y_sc, y_cd, w, prag, grp, intent_idx) in enumerate(loader):
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                cids = cids.to(device, non_blocking=True)
                y_sc = y_sc.to(device, non_blocking=True)
                y_cd = y_cd.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                prag = prag.to(device, non_blocking=True)
                grp = grp.to(device, non_blocking=True)
                intent_idx = intent_idx.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                # Optional Born-Again EMA target blending
                ema_last_p = None
                ema_halts = None
                if ema_teacher is not None:
                    ema_teacher.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                        epre = ema_teacher.forward_prelude(ids, mask, cids if mdl.use_char_stem else None)
                        eh0, emask = epre[0], epre[-1]
                        ez = epre[1] if ema_teacher.mhc_streams > 1 else eh0
                        eh_list = []
                        ep_list = []
                        for s in range(args.num_loops):
                            ez, ep_s, _, eh_s, _, _, _ = ema_teacher.forward_core_step(ez, eh0, emask, ema_teacher.step_idxs[s], apply_aux=True)
                            ep_list.append(ep_s.float())
                            eh_list.append(eh_s.float())
                        ema_last_p = ep_list[-1]
                        ema_halts = torch.stack(eh_list, dim=-1)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pre = mdl.forward_prelude(ids, mask, cids if mdl.use_char_stem else None)
                    h0, c_mask = pre[0], pre[-1]
                    z = pre[1] if mdl.mhc_streams > 1 else h0

                    surv = torch.ones_like(y_sc)
                    step_q_list = []
                    step_probs = []
                    step_logits = []
                    step_halts = []
                    step_losses = []
                    loss_cd_total = 0.0
                    loss_pr_total = 0.0
                    loss_lat_total = 0.0
                    prag_mask = (prag >= 0.0).all(dim=-1)

                    y_eff = (0.72 * y_sc + 0.28 * ema_last_p) if ema_last_p is not None else y_sc

                    for s in range(args.num_loops):
                        z, p_s, sc_logit, halt_p, cd_logit, pr_logit, lat_aux = mdl.forward_core_step(
                            z, h0, c_mask, mdl.step_idxs[s], apply_aux=True
                        )
                        q_s = surv * halt_p if s < args.num_loops - 1 else surv
                        surv = surv * (1.0 - halt_p)
                        step_q_list.append(q_s)
                        step_probs.append(p_s)
                        step_logits.append(sc_logit)
                        step_halts.append(halt_p)

                        # Asymmetric DistiLLM Skewed KL + Huber Continuous Regression
                        l_skl = _skewed_kl_torch(p_s, y_eff, alpha=0.1)
                        l_hub = F.smooth_l1_loss(p_s.float(), y_eff.float(), beta=0.05, reduction="none")
                        per_sample = (0.75 * l_skl + 0.25 * l_hub) * w
                        step_losses.append(per_sample)

                        loss_cd_total = loss_cd_total + ce_code(cd_logit, y_cd) * (0.85 if s == 0 else 1.0)
                        if prag_mask.any():
                            pr_mat = bce_none(pr_logit, prag.clamp(min=0.0)) * prag_step_weights[s].unsqueeze(0)
                            loss_pr_total = loss_pr_total + pr_mat[prag_mask].mean()

                        if lat_aux is not None:
                            l1_logit, l2_vec, l3_cd_logit, l3_pr_logit = lat_aux
                            if prag_mask.any():
                                loss_lat_total = loss_lat_total + 0.15 * bce_none(l1_logit[prag_mask], prag[prag_mask][:, surf_idx_t]).mean()
                                loss_lat_total = loss_lat_total + 0.15 * bce_none(l3_pr_logit[prag_mask], prag[prag_mask][:, deep_idx_t]).mean()
                            loss_lat_total = loss_lat_total + 0.12 * ce_code(l3_cd_logit, y_cd)
                            if intent_vecs_t is not None:
                                im = intent_idx >= 0
                                if im.any():
                                    tgt_v = intent_vecs_t[intent_idx[im]].float()
                                    pred_v = F.normalize(l2_vec[im].float(), p=2, dim=-1)
                                    loss_lat_total = loss_lat_total + 0.20 * (1.0 - (pred_v * tgt_v).sum(dim=-1)).mean()

                    # Zero-sync GPU GroupDRO weighting
                    grp_w_t = dro_tracker.update(grp, step_losses[-1]) * len(V04_GROUPS)
                    dro_mult = grp_w_t[grp]

                    q_mat = torch.stack(step_q_list, dim=-1)
                    sc_mat = torch.stack(step_losses, dim=-1) * dro_mult.unsqueeze(-1)
                    loss_ouro = (q_mat * sc_mat).sum(dim=-1).mean()
                    loss_uniform = sc_mat.mean()

                    # Oracle-Margin Exit Supervision eliminating bimodal step-2 collapse
                    loss_oracle_exit = 0.0
                    p_last = step_probs[-1].detach()
                    err_last = (p_last - y_eff).abs()
                    for s in range(args.num_loops - 1):
                        err_s = (step_probs[s].detach() - y_eff).abs()
                        oracle_sufficient = ((err_s <= err_last + 0.035) & (err_s <= 0.18)).float()
                        # F.binary_cross_entropy is rejected inside autocast regions; run it in fp32 with autocast off.
                        with torch.amp.autocast("cuda", enabled=False):
                            loss_oracle_exit = loss_oracle_exit + F.binary_cross_entropy(step_halts[s].float().clamp(1e-4, 1.0 - 1e-4), oracle_sufficient)
                    loss_oracle_exit = loss_oracle_exit / max(1, args.num_loops - 1)

                    # Cross-step logit consistency + monotonic anti-regression step penalty
                    loss_consist = 0.0
                    loss_anti_reg = 0.0
                    for s in range(1, args.num_loops):
                        loss_consist = loss_consist + F.mse_loss(step_logits[s].float(), step_logits[s - 1].detach().float())
                        loss_anti_reg = loss_anti_reg + F.relu(step_losses[s] - step_losses[s - 1].detach()).mean()
                    loss_consist = loss_consist / max(1, args.num_loops - 1)
                    loss_anti_reg = loss_anti_reg / max(1, args.num_loops - 1)

                    # Late-step LoRA weight L2 regularizer (prevents step >= 2 OOD drift)
                    loss_lora_reg = 0.0
                    if args.num_loops > 2:
                        for blk in mdl.core_blocks:
                            for proj in (blk.q_proj, blk.k_proj, blk.v_proj, blk.o_proj, blk.ffn_up, blk.ffn_down):
                                if hasattr(proj, "lora_A"):
                                    loss_lora_reg = loss_lora_reg + proj.lora_A[2:].float().pow(2).mean() + proj.lora_B[2:].float().pow(2).mean()

                    prior = 1.0 / args.num_loops
                    kl_prior = (q_mat * (torch.log(q_mat.clamp(min=1e-6)) - math.log(prior))).sum(dim=-1).mean()

                    loss = 0.45 * loss_ouro + 0.55 * loss_uniform + 0.04 * kl_prior + 0.18 * loss_oracle_exit + 0.05 * loss_consist + 0.15 * loss_anti_reg + 1e-4 * loss_lora_reg
                    loss = loss + 0.14 * (loss_cd_total / args.num_loops) + (loss_lat_total / args.num_loops)
                    if prag_mask.any():
                        loss = loss + 0.22 * (loss_pr_total / args.num_loops)

                    # Asymmetric high-precision tail-push loss (maximizes R@P95%/P99% & suppresses structured/multilingual human FPs)
                    ai_def_mask = (y_sc >= 0.85) & (w > 0.0)
                    struct_grp_idx = GROUP_TO_IDX["structured_human_hard"]
                    pre_grp_idx = GROUP_TO_IDX["pre_llm_human"]
                    ml_grp_idx = GROUP_TO_IDX["multilingual_world_knowledge"]
                    hu_def_mask = (
                        (y_sc <= 0.15)
                        | (((grp == struct_grp_idx) | (grp == pre_grp_idx)) & (y_sc <= 0.22))
                        | ((grp == ml_grp_idx) & (y_sc <= 0.18))
                    ) & (w > 0.0)
                    if ai_def_mask.any() and hu_def_mask.any():
                        l_last_fp = step_logits[-1].float()
                        p_last_fp = step_probs[-1].float()
                        l_ai_m = l_last_fp[ai_def_mask].mean()
                        hu_l = l_last_fp[hu_def_mask]
                        k_top = max(1, hu_l.numel() // 4)
                        l_hu_hard = torch.topk(hu_l, k_top).values.mean()
                        loss_tail_margin = F.softplus(2.4 - (l_ai_m - l_hu_hard))
                        loss_tail_fp = F.relu(p_last_fp[hu_def_mask] - 0.12).pow(2).mean()
                        loss = loss + 0.12 * loss_tail_margin + 0.20 * loss_tail_fp

                    # QAT exit-boundary & probability parity stabilizer
                    if qat and ema_last_p is not None and ema_halts is not None:
                        h_stack = torch.stack(step_halts, dim=-1).float()
                        loss = loss + 0.06 * F.mse_loss(step_probs[-1].float(), ema_last_p) + 0.06 * F.mse_loss(h_stack, ema_halts)

                    # Pre-tokenized Length-normalized SimPO pairwise margin + Prelude topic invariance
                    if pairs_pre:
                        p_batch = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                        ai_ids, ai_mask, hu_ids, hu_mask, ai_c, hu_c, pw_t = _collate_pre_pairs(p_batch)

                        pre_ai = mdl.forward_prelude(ai_ids, ai_mask, ai_c)
                        pre_hu = mdl.forward_prelude(hu_ids, hu_mask, hu_c)
                        h0_ai, cm_ai = pre_ai[0], pre_ai[-1]
                        h0_hu, cm_hu = pre_hu[0], pre_hu[-1]
                        z_ai = pre_ai[1] if mdl.mhc_streams > 1 else h0_ai
                        z_hu = pre_hu[1] if mdl.mhc_streams > 1 else h0_hu

                        loss_topic = (1.0 - F.cosine_similarity(h0_ai[:, 0, :].float(), h0_hu[:, 0, :].float(), dim=-1)).mean()
                        pair_acc = 0.0
                        pair_w = 0.0
                        for s in range(args.num_loops):
                            z_ai, _, l_ai, _, _, _, _ = mdl.forward_core_step(z_ai, h0_ai, cm_ai, mdl.step_idxs[s], apply_aux=True)
                            z_hu, _, l_hu, _, _, _, _ = mdl.forward_core_step(z_hu, h0_hu, cm_hu, mdl.step_idxs[s], apply_aux=True)
                            sw = 0.60 if s == 0 else 1.0
                            simpo_m = (F.softplus(1.6 - (l_ai.float() - l_hu.float())) * pw_t).mean()
                            pair_acc = pair_acc + sw * simpo_m
                            pair_w += sw
                        loss = loss + 0.30 * (pair_acc / max(1e-6, pair_w)) + 0.04 * loss_topic

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                if update_ema and ema_teacher is not None:
                    with torch.no_grad():
                        for p_e, p_s in zip(ema_teacher.parameters(), mdl.parameters()):
                            p_e.data.mul_(0.995).add_(p_s.data, alpha=0.005)

                if use_swa and ep >= num_epochs // 2 and (global_step % 50 == 0 or global_step == total_steps - 1):
                    with torch.no_grad():
                        cur_sd = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
                        if swa_state is None:
                            swa_state = cur_sd
                            swa_count = 1
                        else:
                            for k in swa_state:
                                if swa_state[k].is_floating_point():
                                    swa_state[k].mul_(swa_count / (swa_count + 1)).add_(cur_sd[k], alpha=1.0 / (swa_count + 1))
                            swa_count += 1

                running_loss_t = running_loss_t + loss.detach().float()
                n_seen += 1
                global_step += 1
                if (step + 1) % log_every == 0:
                    avg_l = float(running_loss_t.item()) / max(1, n_seen)
                    print(f"[{phase_name} ep={ep+1}/{num_epochs} step={step+1}/{len(loader)}] loss={avg_l:.4f} elapsed={time.monotonic()-t_p:.1f}s", flush=True)
                if max_steps_cap > 0 and global_step >= max_steps_cap:
                    break

            v_l, _ = predict_all_steps_pt(mdl, val_enc, val_txts)
            m_ep = compute_eval_metrics(val_scores, expit(v_l[min(args.num_loops - 1, 2)]))
            print(f"[{phase_name} ep={ep+1} DONE] val_normal_last_step: {json.dumps(m_ep)}", flush=True)
            if max_steps_cap > 0 and global_step >= max_steps_cap:
                break

        return swa_state

    # Stage 1: Full-corpus warmup
    run_phase(
        train_items_raw,
        num_epochs=1 if args.smoke_test else args.epochs,
        lr=args.lr,
        qat=False,
        phase_name="Stage1-Warmup",
    )

    # Snapshot eager EMA teacher for Stage 2 + QAT boundary parity
    _set_compiled_hotpaths(False)
    ema_teacher = copy.deepcopy(mdl).eval()
    ema_teacher.set_sdpa(torch.cuda.is_available())
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # Stage 2: Born-Again EMA Refinement on stratified 11-group subset
    stage2_items = train_items_raw if args.smoke_test else subset_balanced_prio(train_items_raw, args.stage2_cap, seed_s=20261126)
    run_phase(
        stage2_items,
        num_epochs=1 if args.smoke_test else args.refine_epochs,
        lr=args.refine_lr,
        qat=False,
        phase_name="Stage2-BornAgainEMA",
        ema_teacher=ema_teacher,
        update_ema=True,
    )

    # Stage 3: 2-Epoch INT8 QAT + SWA + exit stability penalty
    if args.qat_epochs > 0:
        qat_items = train_items_raw if args.smoke_test else subset_balanced_prio(train_items_raw, args.qat_cap, seed_s=20261127)
        swa_sd = run_phase(
            qat_items,
            num_epochs=1 if args.smoke_test else args.qat_epochs,
            lr=args.qat_lr,
            qat=True,
            phase_name="Stage3-QAT-SWA",
            ema_teacher=ema_teacher,
            update_ema=False,
            use_swa=not args.smoke_test,
            max_steps_cap=12 if args.smoke_test else 0,
        )
        if swa_sd is not None:
            mdl.load_state_dict(swa_sd, strict=True)
            print("[Stage3-QAT-SWA] Applied SWA weights.", flush=True)

    del ema_teacher
    _set_compiled_hotpaths(False)
    mdl.set_sdpa(False)
    mdl.set_qat_enabled(False)
    mdl.bake_mhc_matrices()
    mdl.eval()

    # ------------------------------------------------------------------
    # Per-Step Temp Calibration + Delta-Convergence Exit Policy + PAVA Isotonic + Split Conformal
    # ------------------------------------------------------------------
    with torch.no_grad():
        mdl.step_temperatures.fill_(1.0)
    raw_val_logits, val_halts = predict_all_steps_pt(mdl, val_enc, val_txts)
    fitted_temps = []
    for s in range(args.num_loops):
        t_val = fit_temperature(raw_val_logits[s], val_scores / 100.0)
        fitted_temps.append(round(float(t_val), 6))
        with torch.no_grad():
            mdl.step_temperatures[s] = float(t_val)
    print(f"[Calibration] step_temperatures = {fitted_temps}", flush=True)

    cal_val_logits, val_halts = predict_all_steps_pt(mdl, val_enc, val_txts)
    val_probs_steps = expit(cal_val_logits)
    normal_max = max(1, min(args.num_loops, args.normal_max_loops))
    audit_max = max(normal_max, args.num_loops)

    from scipy.stats import spearmanr

    val_bin = (val_scores >= 70.0).astype(bool)
    base_norm_auroc = float(roc_auc_score(val_bin, val_probs_steps[normal_max - 1]))
    base_norm_sp = float(spearmanr(val_scores, val_probs_steps[normal_max - 1]).statistic)

    best_policy = {"exit_low": 0.25, "exit_high": 0.85, "halt_threshold": 0.85, "delta_converge": 0.03, "min_converge_step": 1}
    best_avg_steps = float(normal_max) + 1.0
    fallback_policy = best_policy
    fallback_auc = -1.0

    for e_low in (0.15, 0.20, 0.25):
        for e_high in (0.80, 0.85, 0.90):
            for h_thr in (0.75, 0.85, 0.95):
                for d_conv in (0.02, 0.035, 0.05):
                    cand = {"exit_low": e_low, "exit_high": e_high, "halt_threshold": h_thr, "delta_converge": d_conv, "min_converge_step": 1}
                    p_cand, st_cand = apply_exit_policy_v04(val_probs_steps, val_halts, normal_max, cand)
                    auc_cand = float(roc_auc_score(val_bin, p_cand))
                    sp_cand = float(spearmanr(val_scores, p_cand).statistic)
                    avg_s = float(st_cand.mean())
                    mid_frac = float((st_cand == 2).mean()) if normal_max >= 3 else 1.0
                    if auc_cand > fallback_auc:
                        fallback_auc = auc_cand
                        fallback_policy = cand
                    if auc_cand >= base_norm_auroc - 0.0010 and sp_cand >= base_norm_sp - 0.0020 and (mid_frac >= 0.04 or avg_s < 1.6):
                        if avg_s < best_avg_steps:
                            best_avg_steps = avg_s
                            best_policy = cand
    if best_avg_steps > normal_max:
        best_policy = fallback_policy
    print(f"[Calibration] exit_policy={best_policy} (base_auc={base_norm_auroc:.4f}, base_sp={base_norm_sp:.4f})", flush=True)

    # Calibrate PAVA Isotonic + Split Conformal 3-Way + F0.5/FPR05 on val_dsh_gold
    val_dsh_logits, val_dsh_halts = predict_all_steps_pt(mdl, val_dsh_enc, val_dsh_txts)
    val_dsh_raw_p, _ = apply_exit_policy_v04(expit(val_dsh_logits), val_dsh_halts, normal_max, best_policy)
    if not args.smoke_test:
        isotonic_knots = fit_pava_isotonic(val_dsh_raw_p, val_dsh_scores / 100.0, max_knots=64)
        isotonic_knots["raw_blend"] = 0.05
    else:
        isotonic_knots = None
    val_dsh_cal_p = apply_isotonic_knots(val_dsh_raw_p, isotonic_knots)
    t_f05, t_fpr05 = find_val_thresholds(val_dsh_scores, val_dsh_cal_p)
    hp_val = compute_high_precision_metrics(val_dsh_scores, val_dsh_cal_p)
    conformal_cal = fit_split_conformal_3way(val_dsh_scores, val_dsh_cal_p, alphas=(0.03, 0.05))

    def summarize_split(enc: list[list[int]], txts: list[str], scores: np.ndarray) -> tuple[dict, dict, dict, np.ndarray]:
        l_steps, h_steps = predict_all_steps_pt(mdl, enc, txts)
        p_steps = expit(l_steps)
        per_step = {}
        for s in range(args.num_loops):
            ps_cal = apply_isotonic_knots(p_steps[s], isotonic_knots)
            per_step[f"step_{s}"] = {
                **compute_eval_metrics(scores, ps_cal, t_f05, t_fpr05),
                **compute_high_precision_metrics(scores, ps_cal),
            }
        p_norm_raw, st_norm = apply_exit_policy_v04(p_steps, h_steps, normal_max, best_policy)
        p_norm = apply_isotonic_knots(p_norm_raw, isotonic_knots)
        m_norm = {
            **compute_eval_metrics(scores, p_norm, t_f05, t_fpr05),
            **compute_high_precision_metrics(scores, p_norm),
            "avg_steps": round(float(st_norm.mean()), 3),
            "exit_hist": dict(sorted((str(k), int(v)) for k, v in Counter(st_norm.tolist()).items())),
        }
        p_aud_raw, st_aud = apply_exit_policy_v04(p_steps, h_steps, audit_max, best_policy)
        p_aud = apply_isotonic_knots(p_aud_raw, isotonic_knots)
        m_aud = {
            **compute_eval_metrics(scores, p_aud, t_f05, t_fpr05),
            **compute_high_precision_metrics(scores, p_aud),
            "avg_steps": round(float(st_aud.mean()), 3),
            "exit_hist": dict(sorted((str(k), int(v)) for k, v in Counter(st_aud.tolist()).items())),
        }
        return per_step, m_norm, m_aud, np.asarray(p_norm, dtype=np.float32)

    _, val_norm_m, _, _ = summarize_split(val_enc, val_txts, val_scores)
    _, val_dsh_norm_m, _, _ = summarize_split(val_dsh_enc, val_dsh_txts, val_dsh_scores)
    test_step_m, test_norm_m, test_aud_m, _ = summarize_split(test_enc, test_txts, test_scores)
    test_dsh_step_m, test_dsh_norm_m, test_dsh_aud_m, _ = summarize_split(test_dsh_enc, test_dsh_txts, test_dsh_scores)
    _, test_all_norm_m, _, _ = summarize_split(test_all_enc, test_all_txts, test_all_scores)
    _, test_dsh_all_norm_m, _, _ = summarize_split(test_dsh_all_enc, test_dsh_all_txts, test_dsh_all_scores)
    probe_step_m, probe_norm_m, probe_aud_m, probe_norm_p = summarize_split(probe_enc, probe_txts, probe_scores)

    probe_norm_m.update(
        {
            "pred_mean_100": round(float(probe_norm_p.mean() * 100.0), 2),
            "pred_median_100": round(float(np.median(probe_norm_p) * 100.0), 2),
            "ge_060": int((probe_norm_p >= 0.60).sum()),
            "ge_070": int((probe_norm_p >= 0.70).sum()),
            "ge_val_f05": int((probe_norm_p >= t_f05).sum()),
            "ge_val_fpr05": int((probe_norm_p >= t_fpr05).sum()),
            "details": [
                {"id": r["sample_id"], "hx_score": r["score"], "student_prob": round(float(p), 4)}
                for r, p in sorted(zip(probe_rows, probe_norm_p), key=lambda x: x[0]["sample_id"])
            ],
        }
    )

    sample_texts = [prepare_text(r["text"]) for r in test_gold[:256]]
    sample_enc = [tokenizer.encode_ids(t, max_len=MAX_LEN) for t in sample_texts]
    _, _, _, pt_sample_norm_p = summarize_split(sample_enc, sample_texts, test_scores[: len(sample_texts)])

    # ------------------------------------------------------------------
    # Export Dual Static ONNX + Ordered W4-Emb / W8-Body Edge Bundle
    # ------------------------------------------------------------------
    mdl_cpu = mdl.cpu().eval()
    prelude_fp32 = out_dir / "prelude.fp32.tmp.onnx"
    core_fp32 = out_dir / "core_step.fp32.tmp.onnx"
    prelude_onnx = out_dir / "prelude.onnx"
    core_onnx = out_dir / "core_step.onnx"

    dummy_ids = torch.ones((2, 32), dtype=torch.long)
    dummy_mask = torch.ones((2, 32), dtype=torch.long)
    dummy_cids = torch.full((2, 32), 0x61, dtype=torch.long)

    pre_inputs = (dummy_ids, dummy_mask, dummy_cids) if mdl_cpu.use_char_stem else (dummy_ids, dummy_mask)
    pre_in_names = ["input_ids", "attention_mask", "char_ids"] if mdl_cpu.use_char_stem else ["input_ids", "attention_mask"]
    pre_dyn = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "h0": {0: "batch", 1: "core_seq"},
        "core_mask": {0: "batch", 1: "core_seq"},
    }
    if mdl_cpu.use_char_stem:
        pre_dyn["char_ids"] = {0: "batch", 1: "char_seq"}
    if mdl_cpu.mhc_streams > 1:
        pre_out_names = ["h0", "z0", "core_mask"]
        pre_dyn["z0"] = {0: "batch", 2: "core_seq"}
    else:
        pre_out_names = ["h0", "core_mask"]

    _export_fp32_and_int8(
        PreludeExportWrapper(mdl_cpu),
        pre_inputs,
        input_names=pre_in_names,
        output_names=pre_out_names,
        dynamic_axes=pre_dyn,
        fp32_path=prelude_fp32,
        int8_path=prelude_onnx,
    )

    core_seq_d = 32 + mdl_cpu.num_latents
    dummy_h0 = torch.zeros((2, core_seq_d, cfg.hidden_size), dtype=torch.float32)
    dummy_cmask = torch.ones((2, core_seq_d), dtype=torch.long)
    dummy_z = torch.zeros((2, mdl_cpu.mhc_streams, core_seq_d, cfg.hidden_size), dtype=torch.float32) if mdl_cpu.mhc_streams > 1 else dummy_h0
    dummy_step = torch.tensor(0, dtype=torch.long)
    core_dyn = {
        "z_prev": {0: "batch", 2 if mdl_cpu.mhc_streams > 1 else 1: "core_seq"},
        "h0": {0: "batch", 1: "core_seq"},
        "attention_mask": {0: "batch", 1: "core_seq"},
        "z_next": {0: "batch", 2 if mdl_cpu.mhc_streams > 1 else 1: "core_seq"},
        "prob": {0: "batch"},
        "logit": {0: "batch"},
        "halt_prob": {0: "batch"},
    }
    try:
        _export_fp32_and_int8(
            CoreStepExportWrapper(mdl_cpu),
            (dummy_z, dummy_h0, dummy_cmask, dummy_step),
            input_names=["z_prev", "h0", "attention_mask", "step_id"],
            output_names=["z_next", "prob", "logit", "halt_prob"],
            dynamic_axes=core_dyn,
            fp32_path=core_fp32,
            int8_path=core_onnx,
        )
    finally:
        core_fp32.unlink(missing_ok=True)

    # Ordered W4-Emb / W8-Body from CLEAN FP32 prelude (BEFORE deleting prelude_fp32!)
    prelude_w4 = out_dir / "prelude_w4.onnx"
    try:
        w4_res = quantize_prelude_ordered_w4_w8(prelude_fp32, prelude_w4, block_size=32)
    finally:
        prelude_fp32.unlink(missing_ok=True)

    manifest = {
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": (
            f"Prelude(1)+RRT_DualLoop({args.core_layers}x{args.num_loops})+mHC(N={args.mhc_streams})"
            f"+LOTUS(K={args.num_latents})+CharStem({int(not args.no_char_stem)})+DiffAttn({int(args.diff_attn)})+SVDEmb(E={args.embed_dim})"
        ),
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "quant_method": "qat_swa_plus_dynamic_int8",
        "vocab_size": tokenizer.vocab_size,
        "byte_offset": tokenizer.byte_offset,
        "embed_dim": int(args.embed_dim),
        "hidden_size": int(cfg.hidden_size),
        "core_layers": int(args.core_layers),
        "num_loops": int(args.num_loops),
        "num_latents": int(args.num_latents),
        "mhc_streams": int(args.mhc_streams),
        "lora_rank": int(args.lora_rank),
        "lora_targets": args.lora_targets,
        "normal_max_loops": int(normal_max),
        "audit_max_loops": int(audit_max),
        "max_len": MAX_LEN,
        "char_max_bytes": CHAR_MAX_BYTES,
        "head_chars": HEAD_CHARS,
        "tail_chars": TAIL_CHARS,
        "step_temperatures": fitted_temps,
        "exit_policy": best_policy,
        "isotonic_knots": isotonic_knots,
        "conformal": conformal_cal,
        "thresholds": {
            "f05": round(t_f05, 4),
            "p95": hp_val["t_p95"],
            "p99": hp_val["t_p99"],
            "fpr05": round(t_fpr05, 4),
            "score70": 0.70,
        },
        "prelude_file": prelude_onnx.name,
        "core_file": core_onnx.name,
        "prelude_bytes": int(prelude_onnx.stat().st_size),
        "core_bytes": int(core_onnx.stat().st_size),
        "total_onnx_bytes": int(prelude_onnx.stat().st_size + core_onnx.stat().st_size),
        "prelude_sha256": sha256_file(prelude_onnx),
        "core_sha256": sha256_file(core_onnx),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT parity + latency verification on 256 test_gold rows & hard traps
    engine = MeguvizV04Engine(out_dir, threads=1, mode="normal")
    t_single = time.monotonic()
    ort_single = [engine.predict_one(txt, mode="normal") for txt in sample_texts]
    single_ms = (time.monotonic() - t_single) * 1000.0 / max(1, len(sample_texts))

    t_batch = time.monotonic()
    ort_batch = engine.predict_batch(sample_texts, mode="normal", max_batch=32)
    batch_ms = (time.monotonic() - t_batch) * 1000.0 / max(1, len(sample_texts))

    ort_probs = np.array([x["prob"] for x in ort_batch], dtype=np.float32)
    ort_single_probs = np.array([x["prob"] for x in ort_single], dtype=np.float32)
    parity = {
        "mae_vs_pytorch": round(float(np.mean(np.abs(pt_sample_norm_p - ort_probs))), 5),
        "max_abs_diff_vs_pytorch": round(float(np.max(np.abs(pt_sample_norm_p - ort_probs))), 5),
        "batch_vs_single_mae": round(float(np.mean(np.abs(ort_single_probs - ort_probs))), 6),
        "single_1thread_ms": round(single_ms, 2),
        "compact_batch32_1thread_ms": round(batch_ms, 2),
        "ort_avg_steps": round(float(np.mean([x["steps_used"] for x in ort_batch])), 3),
    }

    probe_file = Path("round01/bili-round01-user-confirmed.jsonl")
    verify_res = run_verify_v04(out_dir, probe_file.resolve(), mode="normal", threads=1) if probe_file.is_file() else {}

    last_norm_k = f"step_{normal_max - 1}"
    s0_pr = (test_step_m.get("step_0") or {}).get("pr_auc", 0.0)
    sn_pr = (test_step_m.get(last_norm_k) or {}).get("pr_auc", 0.0)
    loop_monotonicity = {
        "test_bili_gold_pr_auc_by_step": {k: v.get("pr_auc") for k, v in test_step_m.items()},
        "test_bili_gold_rp95_by_step": {k: v.get("recall_at_p95") for k, v in test_step_m.items()},
        "test_dsh_gold_pr_auc_by_step": {k: v.get("pr_auc") for k, v in test_dsh_step_m.items()},
        "non_regressive_normal_last_ge_step0": sn_pr >= s0_pr - 0.003,
    }

    metrics_file = out_dir / f"metrics-{args.version}.json"
    metrics = {
        "version": args.version,
        "backbone": args.backbone,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "batch_size_used": batch_size,
        "arch": {
            "embed_dim": args.embed_dim,
            "core_layers": args.core_layers,
            "num_loops": args.num_loops,
            "num_latents": args.num_latents,
            "mhc_streams": args.mhc_streams,
            "use_char_stem": not args.no_char_stem,
            "diff_attn": args.diff_attn,
            "lora_rank": args.lora_rank,
            "lora_targets": args.lora_targets,
            "emb_drop": eff_emb_drop,
            "domain_aug": not args.no_domain_aug,
            "normal_max_loops": normal_max,
            "audit_max_loops": audit_max,
            "vocab_size": tokenizer.vocab_size,
        },
        "step_temperatures": fitted_temps,
        "exit_policy": best_policy,
        "conformal": conformal_cal,
        "thresholds": {
            "val_f05": round(t_f05, 4),
            "val_p95": hp_val["t_p95"],
            "val_p99": hp_val["t_p99"],
            "val_fpr05": round(t_fpr05, 4),
        },
        "loop_monotonicity": loop_monotonicity,
        "val_bili_gold": val_norm_m,
        "val_dsh_gold": val_dsh_norm_m,
        "test_bili_gold": test_norm_m,
        "test_bili_gold_audit": test_aud_m,
        "test_bili_gold_by_step": test_step_m,
        "test_dsh_gold": test_dsh_norm_m,
        "test_dsh_gold_audit": test_dsh_aud_m,
        "test_dsh_gold_by_step": test_dsh_step_m,
        "test_all_bili_distilled": test_all_norm_m,
        "test_all_dsh_distilled": test_dsh_all_norm_m,
        "probe_u001_u017": probe_norm_m,
        "probe_u001_u017_audit": {k: v for k, v in probe_aud_m.items() if k != "details"},
        "probe_u001_u017_by_step": probe_step_m,
        "onnx_int8": {**parity, "prelude_bytes": manifest["prelude_bytes"], "core_bytes": manifest["core_bytes"], "total_onnx_bytes": manifest["total_onnx_bytes"]},
        "edge_w4_w8": w4_res,
        "onnx_verify": {k: v for k, v in verify_res.items() if k != "probe_u001_u017"},
        "train_items_used": len(train_items_raw),
        "literary_pairs_used": len(pairs),
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics["release_gates"] = check_release_gates(metrics)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_main = out_dir / f"meguvizcac-{args.version}-int8.zip"
    common_assets = ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", metrics_file.name]
    with ZipFile(zip_main, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for fn in (prelude_onnx.name, core_onnx.name, "manifest.json", *common_assets):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    metrics["onnx_int8"]["zip_bytes"] = int(zip_main.stat().st_size)

    if w4_res.get("ok") and prelude_w4.is_file():
        edge_dir = out_dir / "_edge_tmp"
        edge_dir.mkdir(parents=True, exist_ok=True)
        edge_manifest = dict(manifest)
        edge_manifest["quant_method"] = "ordered_w4_emb_opset21_plus_w8_body"
        edge_manifest["prelude_file"] = prelude_w4.name
        edge_manifest["prelude_bytes"] = int(prelude_w4.stat().st_size)
        edge_manifest["total_onnx_bytes"] = int(prelude_w4.stat().st_size + core_onnx.stat().st_size)
        edge_manifest["prelude_sha256"] = sha256_file(prelude_w4)
        (edge_dir / "manifest.json").write_text(json.dumps(edge_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        zip_edge = out_dir / f"meguvizcac-{args.version}-edge-mixed.zip"
        with ZipFile(zip_edge, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
            zf.write(prelude_w4, prelude_w4.name)
            zf.write(core_onnx, core_onnx.name)
            zf.write(edge_dir / "manifest.json", "manifest.json")
            for fn in common_assets:
                fp = out_dir / fn
                if fp.is_file():
                    zf.write(fp, fn)
        shutil.rmtree(edge_dir, ignore_errors=True)
        w4_res["edge_zip_bytes"] = int(zip_edge.stat().st_size)
        w4_res["edge_total_onnx_bytes"] = edge_manifest["total_onnx_bytes"]

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(json.dumps({"test_bili_gold": test_norm_m, "test_dsh_gold": test_dsh_norm_m, "onnx_int8": metrics["onnx_int8"], "release_gates": metrics["release_gates"]}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--qwen-data", type=Path, default=None)
    parser.add_argument("--pairs-data", type=Path, default=None)
    parser.add_argument("--extra-pairs", type=Path, action="append", default=[])
    parser.add_argument("--intent-cache", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("./out_v04"))
    parser.add_argument("--version", default="v0.4-dualloop-2x4-e192")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--core-layers", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--normal-max-loops", type=int, default=3)
    parser.add_argument("--num-latents", type=int, default=4)
    parser.add_argument("--mhc-streams", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", choices=("qvo_down", "all"), default="qvo_down")
    parser.add_argument("--diff-attn", action="store_true")
    parser.add_argument("--no-char-stem", action="store_true")
    parser.add_argument("--emb-drop", type=float, default=-1.0, help="-1 = auto (0.15 for E>=192 else 0.10)")
    parser.add_argument("--no-domain-aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=1)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--stage2-cap", type=int, default=380000)
    parser.add_argument("--qat-cap", type=int, default=350000)
    parser.add_argument("--stage1-synth-cap", type=int, default=15000)
    parser.add_argument("--v04-synth-cap", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-auto-lr-scale", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True) on A100 Linux training hotpaths")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--refine-lr", type=float, default=1.5e-5)
    parser.add_argument("--qat-lr", type=float, default=5e-6)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(args)
        return
    train_and_export(args)


if __name__ == "__main__":
    main()
