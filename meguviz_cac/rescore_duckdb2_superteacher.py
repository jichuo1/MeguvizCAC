"""MeguvizCAC v1.6.1 Super-Teacher Soft-Logit Rescorer for Final Master DuckDB 2.0 Clean Pack.

Runs on Colab G4 (RTX PRO 6000 Blackwell 96 GB) in ~4-9 minutes (and caches every per-teacher
score array atomically to Google Drive for instant zero-waste resume!):
  1. Reuses exact `text_sha16` matches from existing `v140_superteacher_soft_distill_pool.jsonl.gz`
     (instant 0-second hit for overlapping texts!).
  2. Discovers all available cached v1.4.0 Super-Teacher checkpoints (`electra_large`, `bge_m3`,
     `sensenova_loop_9b`) in `--st-ckpt-dir`, loads exact temperatures + validated convex ensemble
     weights from `metrics-v1.4.0-superteacher.json`, merges LoRA in-place for peak throughput,
     and scores ONLY the uncached DuckDB 2.0 rows.
  3. Writes `v161_duckdb2_rescored.jsonl.gz` preserving 100% of DuckDB 2.0 gold/committee fields
     (`y_ai`, `y_pr`, `has_ai`, `has_pr`, `bucket`, `era5`, `sub4`, `is_override`, `tier`) PLUS
     dense progressive per-rung teacher soft targets (`t_ai_lg`, `t_ai_r0`, `t_ai_t1`, `t_ai_t2`,
     `t_ai_t3`, `frontier_gap`, `t_pr_lg`, `kd_w`, `has_teacher=1.0`).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit
import torch
from transformers import AutoTokenizer

from .train_v140_superteacher import (
    _SmokeTinyTokenizerV140,
    _get_cached_prefix_ids,
    _resolve_model_spec,
    predict_dual_logits_v140,
)
from .v140_superteacher import (
    DEFAULT_MODELS_V140,
    UniversalDualAxisSuperTeacherV140,
    read_jsonl_any_v140,
    text_sha16_v140,
)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _load_base_pool_lookup(base_pool: Path | None) -> dict[str, dict[str, Any]]:
    if base_pool is None or not base_pool.is_file():
        return {}
    t0 = time.time()
    lookup: dict[str, dict[str, Any]] = {}
    for r in read_jsonl_any_v140(base_pool):
        txt = str(r.get("text", "") or "").strip()
        if not txt:
            continue
        sha = str(r.get("sha16") or text_sha16_v140(txt))
        pm_ai = r.get("per_model_ai_lg") or []
        t_ai_ens = float(r.get("t_ai_logit", 0.0))
        t_pr_ens = float(r.get("t_pr_logit", 0.0))
        lg_el = float(pm_ai[0]) if len(pm_ai) >= 1 else t_ai_ens
        lg_bge = float(pm_ai[1]) if len(pm_ai) >= 2 else t_ai_ens
        lg_sn9 = float(pm_ai[2]) if len(pm_ai) >= 3 else t_ai_ens
        t_std = float(r.get("t_ai_std_p", 0.0))
        lookup[sha] = {
            "t_ai_lg": t_ai_ens,
            "t_ai_r0": 0.35 * lg_el + 0.65 * t_ai_ens,
            "t_ai_t1": 0.35 * lg_bge + 0.65 * t_ai_ens,
            "t_ai_t2": 0.45 * lg_sn9 + 0.55 * t_ai_ens,
            "t_ai_t3": 0.65 * lg_sn9 + 0.35 * t_ai_ens,
            "frontier_gap": abs(lg_sn9 - lg_el),
            "t_pr_lg": t_pr_ens,
            "kd_w": float(1.0 / (1.0 + 1.8 * max(0.0, t_std))),
            "has_teacher": 1.0,
        }
    print(f"[v1.6.1-Rescore] Indexed {len(lookup):,} cached Super-Teacher texts in {time.time()-t0:.1f}s.", flush=True)
    return lookup


def _load_ensemble_weights(st_ckpt_dir: Path, active_keys: list[str]) -> tuple[list[float], list[float]]:
    mp = st_ckpt_dir / "metrics-v1.4.0-superteacher.json"
    if mp.is_file():
        try:
            d = json.loads(mp.read_text(encoding="utf-8"))
            ew = d.get("ensemble_weights") or {}
            wai_d = ew.get("ai") or {}
            wpr_d = ew.get("promo") or {}
            wa = [float(wai_d.get(k, 0.0)) for k in active_keys]
            wp = [float(wpr_d.get(k, 0.0)) for k in active_keys]
            if sum(wa) > 1e-6 and sum(wp) > 1e-6:
                sa, sp = sum(wa), sum(wp)
                return [x / sa for x in wa], [x / sp for x in wp]
        except Exception:
            pass
    # Fallback weights emphasizing SenseNova-Loop-9B > BGE-M3 > ELECTRA
    prior = {"electra_large": 0.25, "bge_m3": 0.30, "sensenova_loop_9b": 0.45}
    raw = [prior.get(k, 1.0 / max(1, len(active_keys))) for k in active_keys]
    s = sum(raw)
    w = [x / s for x in raw]
    return w, w


def main() -> None:
    ap = argparse.ArgumentParser(description="Rescore DuckDB 2.0 clean pack with cached v1.4.0 Super-Teacher Ensemble")
    ap.add_argument("--input-pack", type=Path, required=True)
    ap.add_argument("--output-pack", type=Path, required=True)
    ap.add_argument("--st-ckpt-dir", type=Path, required=True)
    ap.add_argument("--base-distill-pool", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS_V140)
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--head-len", type=int, default=192)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    in_path = args.input_pack.resolve()
    out_path = args.output_pack.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or (out_path.parent / "rescore_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    in_sha = _file_sha256(in_path)
    meta_path = out_path.with_suffix(".meta.json")
    if (not args.smoke_test) and out_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("input_sha256") == in_sha:
                print(f"[v1.6.1-Rescore] Output ALREADY UP-TO-DATE (input_sha={in_sha[:12]}): {out_path}", flush=True)
                return
        except Exception:
            pass

    rows = read_jsonl_any_v140(in_path)
    if args.smoke_test:
        rows = rows[:48]
    print(f"[v1.6.1-Rescore] Loaded {len(rows):,} DuckDB 2.0 rows from {in_path.name} (sha={in_sha[:12]})", flush=True)

    base_lookup = _load_base_pool_lookup(args.base_distill_pool) if not args.smoke_test else {}

    # Separate instant cache hits from uncached texts needing GPU inference
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    row_shas: list[str] = []
    for i, r in enumerate(rows):
        txt = str(r.get("text", "") or "").strip()
        sha = text_sha16_v140(txt)
        row_shas.append(sha)
        if sha not in base_lookup and txt:
            miss_indices.append(i)
            miss_texts.append(txt)

    n_hit = len(rows) - len(miss_indices)
    print(f"[v1.6.1-Rescore] Cache hits from 251K pool: {n_hit:,} | Need GPU scoring: {len(miss_indices):,}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    req_keys = [k.strip() for k in args.models.split(",") if k.strip()]

    active_keys: list[str] = []
    if args.smoke_test:
        active_keys = ["bge_m3", "sensenova_loop_9b"]
    else:
        for mk in req_keys:
            if (args.st_ckpt_dir / f"v140_st_{mk}_best.pt").is_file():
                active_keys.append(mk)
            else:
                print(f"[v1.6.1-Rescore] WARN checkpoint missing for {mk}: {args.st_ckpt_dir / f'v140_st_{mk}_best.pt'}", flush=True)

    per_model_ai: dict[str, np.ndarray] = {}
    per_model_pr: dict[str, np.ndarray] = {}

    if miss_indices and active_keys:
        miss_order_sha = hashlib.sha256("\n".join(row_shas[i] for i in miss_indices).encode("utf-8")).hexdigest()[:16]
        for mk in active_keys:
            spec = _resolve_model_spec(mk)
            ckpt_p = args.st_ckpt_dir / f"v140_st_{mk}_best.pt"
            c_npz = cache_dir / f"rescore_{mk}_{miss_order_sha}.npz"
            if (not args.smoke_test) and c_npz.is_file():
                z = np.load(c_npz, allow_pickle=False)
                if len(z["ai"]) == len(miss_indices):
                    print(f"[v1.6.1-Rescore:{mk}] Loaded cached scores from {c_npz.name}", flush=True)
                    per_model_ai[mk] = z["ai"].astype(np.float32)
                    per_model_pr[mk] = z["pr"].astype(np.float32)
                    continue

            t_m = time.time()
            hf_name = str(spec["hf_name"])
            arch_type = str(spec["arch_type"])
            ft_mode = str(spec["ft_mode"])
            trust_rc = bool(spec.get("trust_remote_code", False))
            preserve_auto_dtype = bool(spec.get("preserve_auto_dtype", False))

            if args.smoke_test:
                tokenizer: Any = _SmokeTinyTokenizerV140(vocab_size=2048, arch_type=arch_type)
                model = UniversalDualAxisSuperTeacherV140(
                    hf_name, arch_type=arch_type, ft_mode=ft_mode,
                    lora_r=8 if ft_mode == "lora" else 0, lora_alpha=16 if ft_mode == "lora" else 0,
                    dropout=0.10, load_dtype=None, trust_remote_code=False, smoke_tiny=True, vocab_size=2048,
                ).to(device)
                t_ai, t_pr = 1.0, 1.0
                eval_bs = 16
            else:
                sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                t_ai = float(sd.get("temp_ai", 1.0))
                t_pr = float(sd.get("temp_pr", 1.0))
                lora_r = int(sd.get("lora_r", spec["lora_r"]))
                lora_alpha = int(sd.get("lora_alpha", spec["lora_alpha"]))
                tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_rc)
                if arch_type == "causal_lm" and tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                eff_dtype: torch.dtype | str | None = (
                    "auto" if preserve_auto_dtype else (amp_dtype if (device.type == "cuda" and ft_mode == "lora") else None)
                )
                model = UniversalDualAxisSuperTeacherV140(
                    hf_name, arch_type=arch_type, ft_mode=ft_mode,
                    lora_r=lora_r, lora_alpha=lora_alpha, dropout=0.15,
                    load_dtype=eff_dtype, trust_remote_code=trust_rc, smoke_tiny=False,
                ).to(device)
                model.load_trainable_state_dict(sd["trainable_state_dict"])
                if ft_mode == "lora":
                    model.merge_lora_inplace()
                eval_bs = int(spec["eval_batch_size"])

            prefix_ids = _get_cached_prefix_ids(tokenizer, arch_type)
            print(f"[v1.6.1-Rescore:{mk}] Scoring {len(miss_texts):,} rows (bs={eval_bs}, T_ai={t_ai:.3f}, T_pr={t_pr:.3f})...", flush=True)
            la, lp = predict_dual_logits_v140(
                model, tokenizer, miss_texts, device,
                batch_size=eval_bs, max_len=args.max_len, head_len=args.head_len,
                amp_dtype=amp_dtype, prefix_ids=prefix_ids,
            )
            per_model_ai[mk] = (la / max(1e-3, t_ai)).astype(np.float32)
            per_model_pr[mk] = (lp / max(1e-3, t_pr)).astype(np.float32)
            if not args.smoke_test:
                np.savez_compressed(c_npz, ai=per_model_ai[mk], pr=per_model_pr[mk])
            print(f"[v1.6.1-Rescore:{mk}] Done in {time.time()-t_m:.1f}s.", flush=True)
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    w_ai, w_pr = _load_ensemble_weights(args.st_ckpt_dir, active_keys) if active_keys else ([], [])
    miss_scores: dict[int, dict[str, Any]] = {}
    if miss_indices and active_keys:
        mat_ai = np.stack([per_model_ai[mk] for mk in active_keys], axis=1)
        mat_pr = np.stack([per_model_pr[mk] for mk in active_keys], axis=1)
        ens_ai = (mat_ai @ np.asarray(w_ai, dtype=np.float32)).astype(np.float32)
        ens_pr = (mat_pr @ np.asarray(w_pr, dtype=np.float32)).astype(np.float32)
        ai_std_p = expit(np.clip(mat_ai, -30.0, 30.0)).std(axis=1) if len(active_keys) > 1 else np.zeros((len(miss_indices),), dtype=np.float32)

        idx_el = active_keys.index("electra_large") if "electra_large" in active_keys else 0
        idx_bge = active_keys.index("bge_m3") if "bge_m3" in active_keys else min(1, len(active_keys) - 1)
        idx_sn9 = active_keys.index("sensenova_loop_9b") if "sensenova_loop_9b" in active_keys else (len(active_keys) - 1)

        for j, orig_i in enumerate(miss_indices):
            ea = float(ens_ai[j])
            ep = float(ens_pr[j])
            l_el = float(mat_ai[j, idx_el])
            l_bge = float(mat_ai[j, idx_bge])
            l_sn9 = float(mat_ai[j, idx_sn9])
            miss_scores[orig_i] = {
                "t_ai_lg": round(ea, 5),
                "t_ai_r0": round(0.35 * l_el + 0.65 * ea, 5),
                "t_ai_t1": round(0.35 * l_bge + 0.65 * ea, 5),
                "t_ai_t2": round(0.45 * l_sn9 + 0.55 * ea, 5),
                "t_ai_t3": round(0.65 * l_sn9 + 0.35 * ea, 5),
                "frontier_gap": round(abs(l_sn9 - l_el), 5),
                "t_pr_lg": round(ep, 5),
                "kd_w": round(float(1.0 / (1.0 + 1.8 * max(0.0, float(ai_std_p[j])))), 5),
                "has_teacher": 1.0,
            }

    n_with_t = 0
    tmp_out = out_path.with_suffix(".gz.tmp")
    with gzip.open(tmp_out, "wt", encoding="utf-8", compresslevel=6) as f:
        for i, r in enumerate(rows):
            rec = dict(r)
            sha = row_shas[i]
            t_info = base_lookup.get(sha) or miss_scores.get(i)
            if t_info is not None:
                rec.update(t_info)
                n_with_t += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_out.replace(out_path)

    meta_out = {
        "input_pack": str(in_path),
        "input_sha256": in_sha,
        "output_pack": str(out_path),
        "output_mb": round(out_path.stat().st_size / (1024.0 * 1024.0), 3),
        "total_rows": len(rows),
        "cache_hits_251k": n_hit,
        "gpu_rescored_rows": len(miss_scores),
        "rows_with_teacher": n_with_t,
        "active_teachers": active_keys,
        "ensemble_weights_ai": dict(zip(active_keys, [round(x, 4) for x in w_ai])),
        "ensemble_weights_pr": dict(zip(active_keys, [round(x, 4) for x in w_pr])),
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta_out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
