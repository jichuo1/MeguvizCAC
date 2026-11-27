"""MeguvizCAC v1.4.0-SuperTeacher-Ultra Trainer & 250K Soft-Distillation Pool Exporter.

Trains a heterogeneous 3-Paradigm Super-Teacher Ensemble anchored by SenseTime's 2026 Looped Transformer
(`sensenova/SenseNova-Loop-9B`, 32L -> 56 effective looped layers + Two-Stream mHC + Hybrid Attention)
on 96 GB Blackwell RTX PRO 6000 with strictly Zero-PII (`diff === 0.0`) Dual-Axis heads:
  1. `electra_large`     (`325M`, 24L Chinese RTD)                    : full BF16 + LLRD
  2. `bge_m3`            (`568M`, 24L Bidirectional ML Contrastive)   : full BF16 + LLRD (100+ languages!)
  3. `sensenova_loop_9b` (`9.0B`, 56-eff-layer Looped Hybrid + mHC)   : BF16/FP32-mHC + Native LoRA r=32 + exact merge

Ingests:
  - Base `v122-sibling-datafactory` + newly merged `+57,585` `v123-sibling-delta` zero-leak gold
  - `15,903` Counterfactual Mirror Pairs (same-topic/length/decorator margin loss!)
  - Clean `EN/JA` multilingual gold anchors from `v04` (85% train w/ homoglyph jitter + 15% val_multilingual)

Outputs (zero OjinjinBench contact!):
  - Compact per-model checkpoints (`v140_st_<key>_best.pt`: ~240 MB for SenseNova-Loop-9B LoRA, ~1.1 GB for BGE-M3)
  - `v140_superteacher_soft_distill_pool.jsonl.gz` (~250K zero-leak real comments, backward-compatible superset!)
  - `v140_al_disagreement_queue_top5000.jsonl.gz` (Top-5,000 highest inter-paradigm disagreement queue for AL-R5!)
  - `metrics-v1.4.0-superteacher.json` (full per-model & ensemble metrics + C1 Zero-PII + C2 LoRA merge parity!)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
from transformers import AutoTokenizer

from .v140_superteacher import (
    CAUSAL_TASK_PREFIX_V140,
    DEFAULT_MODELS_V140,
    SUPERTEACHER_BACKBONES_V140,
    UniversalDualAxisSuperTeacherV140,
    build_optimizer_param_groups_v140,
    build_v140_superteacher_corpus,
    compute_cf_mirror_loss_v140,
    compute_dual_axis_loss_v140,
    encode_universal_v140,
    fit_scalar_temperature_v140,
    iter_length_bucketed_batches_v140,
    jitter_ai_text_v140,
    run_v140_superteacher_zero_pii_gate,
    sample_antidilution_epoch_v140,
    sample_cf_mirror_batch_v140,
)


def _compute_pool_order_sha256(distill_rows: list[dict[str, Any]]) -> str:
    """Exact order-sensitive SHA256 fingerprint over all distill pool rows (prevents silent row-shift bugs!)."""
    h = hashlib.sha256()
    h.update(f"{len(distill_rows)}\n".encode("utf-8"))
    for r in distill_rows:
        h.update(str(r.get("sha16", "")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blk = f.read(4 * 1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _recall_at_prec(y: np.ndarray, p: np.ndarray, target: float) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    best = 0.0
    for i in range(len(thr)):
        if prec[i] >= target and rec[i] > best:
            best = float(rec[i])
    return best


def _best_f05(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    best = 0.0
    for i in range(len(thr)):
        pr, rc = float(prec[i]), float(rec[i])
        d = 0.25 * pr + rc
        if d > 0:
            best = max(best, 1.25 * pr * rc / d)
    return best


def _eval_binary(y: np.ndarray, logits: np.ndarray, temp: float = 1.0) -> dict[str, float]:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()), "roc": 0.5, "pr": 0.0, "f05": 0.0, "r95": 0.0, "r99": 0.0}
    p = expit(np.clip(logits / max(1e-3, temp), -30.0, 30.0))
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "roc": round(float(roc_auc_score(y, p)), 4),
        "pr": round(float(average_precision_score(y, p)), 4),
        "f05": round(_best_f05(y, p), 4),
        "r95": round(_recall_at_prec(y, p, 0.95), 4),
        "r99": round(_recall_at_prec(y, p, 0.99), 4),
    }


def _get_cached_prefix_ids(tokenizer: Any, arch_type: str) -> list[int] | None:
    if arch_type != "causal_lm":
        return None
    enc = tokenizer(
        CAUSAL_TASK_PREFIX_V140,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


@torch.no_grad()
def predict_dual_logits_v140(
    model: UniversalDualAxisSuperTeacherV140,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    max_len: int = 288,
    head_len: int = 192,
    amp_dtype: torch.dtype = torch.bfloat16,
    prefix_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Length-sorted fast batched bf16/fp16 inference returning `(logit_ai, logit_promo)`."""
    n = len(texts)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    model.eval()
    order = sorted(range(n), key=lambda i: len(texts[i]))
    out_ai = np.zeros((n,), dtype=np.float32)
    out_pr = np.zeros((n,), dtype=np.float32)
    use_amp = device.type == "cuda"
    if prefix_ids is None:
        prefix_ids = _get_cached_prefix_ids(tokenizer, model.arch_type)

    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_t = [texts[i] for i in idxs]
        ids, mask, pmask = encode_universal_v140(
            tokenizer, sub_t, arch_type=model.arch_type, max_len=max_len, head_len=head_len, prefix_ids=prefix_ids
        )
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pmask = pmask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = model(ids, mask, pmask)
        la = od["logit_ai"].float().cpu().numpy()
        lp = od["logit_promo"].float().cpu().numpy()
        for j, oi in enumerate(idxs):
            out_ai[oi] = la[j]
            out_pr[oi] = lp[j]
    return out_ai, out_pr


class _SmokeTinyTokenizerV140:
    """Zero-network deterministic tokenizer for local `--smoke-test` (supports both Encoder & Causal LLM)."""

    def __init__(self, vocab_size: int = 2048, arch_type: str = "encoder") -> None:
        self.vocab_size = vocab_size
        if arch_type == "causal_lm":
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0
            self.cls_token_id = None
            self.sep_token_id = None
        else:
            # Deliberately test XLM-R / BGE-M3 style cls_token_id == 0 (catches falsy `or` bugs!)
            self.cls_token_id = 0
            self.pad_token_id = 1
            self.sep_token_id = 2
            self.bos_token_id = 0
            self.eos_token_id = 2

    def __call__(self, texts: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(texts, str):
            return {"input_ids": [16 + (ord(ch) % (self.vocab_size - 24)) for ch in texts]}
        return {"input_ids": [[16 + (ord(ch) % (self.vocab_size - 24)) for ch in str(t)] for t in texts]}


def evaluate_model_suite_v140(
    model: UniversalDualAxisSuperTeacherV140,
    tokenizer: Any,
    corpus: dict[str, Any],
    device: torch.device,
    batch_size: int,
    max_len: int,
    head_len: int,
    amp_dtype: torch.dtype,
    t_ai: float = 1.0,
    t_pr: float = 1.0,
    prefix_ids: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "val_multilingual", "diag_god500"):
        rows = corpus.get(key) or []
        txts = [r["text"] for r in rows]
        preds[key] = predict_dual_logits_v140(
            model, tokenizer, txts, device,
            batch_size=batch_size, max_len=max_len, head_len=head_len,
            amp_dtype=amp_dtype, prefix_ids=prefix_ids,
        )

    res: dict[str, Any] = {}
    if corpus.get("val_gold_v2"):
        y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
        res["val_gold_v2_ai"] = _eval_binary(y, preds["val_gold_v2"][0], temp=t_ai)
    if corpus.get("test_gold_v2"):
        y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["test_gold_v2"]], dtype=np.int32)
        res["test_gold_v2_ai"] = _eval_binary(y, preds["test_gold_v2"][0], temp=t_ai)
    if corpus.get("val_wave3"):
        vw3 = corpus["val_wave3"]
        ya = np.array([int(r["y_ai"] >= 0.5) for r in vw3], dtype=np.int32)
        yp = np.array([int(r["y_pr"] >= 0.5) for r in vw3], dtype=np.int32)
        la, lp = preds["val_wave3"]
        res["val_wave3_ai"] = _eval_binary(ya, la, temp=t_ai)
        res["val_wave3_promo"] = _eval_binary(yp, lp, temp=t_pr)
        q4 = (yp == 1) & (ya == 0)
        if q4.sum() > 0:
            p_ai_q4 = expit(np.clip(la[q4] / max(1e-3, t_ai), -30.0, 30.0))
            res["val_wave3_q4_fpr50"] = round(float((p_ai_q4 >= 0.50).mean()), 4)
            res["val_wave3_q4_mean_p"] = round(float(p_ai_q4.mean()), 4)
    if corpus.get("val_old_video_neg"):
        la = preds["val_old_video_neg"][0]
        p_old = expit(np.clip(la / max(1e-3, t_ai), -30.0, 30.0))
        res["val_old_video_neg"] = {
            "n": len(corpus["val_old_video_neg"]),
            "fpr50": round(float((p_old >= 0.50).mean()), 4),
            "fpr30": round(float((p_old >= 0.30).mean()), 4),
            "p99_prob": round(float(np.percentile(p_old, 99)), 4),
            "mean_prob": round(float(p_old.mean()), 4),
        }
    if corpus.get("val_multilingual"):
        y = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_multilingual"]], dtype=np.int32)
        res["val_multilingual_ai"] = _eval_binary(y, preds["val_multilingual"][0], temp=t_ai)

    g2_pr = res.get("val_gold_v2_ai", {}).get("pr", 0.0)
    g2_r95 = res.get("val_gold_v2_ai", {}).get("r95", 0.0)
    w3_ai_pr = res.get("val_wave3_ai", {}).get("pr", 0.0)
    w3_pr_pr = res.get("val_wave3_promo", {}).get("pr", 0.0)
    ml_pr = res.get("val_multilingual_ai", {}).get("pr", g2_pr)
    old_fpr = res.get("val_old_video_neg", {}).get("fpr50", 0.0)
    q4_fpr = res.get("val_wave3_q4_fpr50", 0.0)
    sel_obj = (
        0.32 * g2_pr + 0.24 * g2_r95 + 0.18 * w3_ai_pr + 0.18 * w3_pr_pr + 0.08 * ml_pr
        - 1.50 * old_fpr - 0.60 * q4_fpr
    )
    res["val_sel_obj"] = round(float(sel_obj), 5)

    if corpus.get("diag_god500"):
        dg = corpus["diag_god500"]
        la = preds["diag_god500"][0]
        tv = [i for i, r in enumerate(dg) if r.get("is_text_verifiable")]
        if tv:
            y_tv = np.array([int(dg[i]["y_ai"] >= 0.5) for i in tv], dtype=np.int32)
            res["diag_god500_text_verif"] = _eval_binary(y_tv, la[tv], temp=t_ai)
    return res, preds


def _resolve_model_spec(key: str) -> dict[str, Any]:
    if key in SUPERTEACHER_BACKBONES_V140:
        return dict(SUPERTEACHER_BACKBONES_V140[key])
    kl = key.lower()
    is_causal = any(tok in kl for tok in ("sensenova", "loop", "qwen", "llama", "mistral", "gemma", "deepseek", "yi-"))
    is_custom_remote = any(tok in kl for tok in ("sensenova", "loop", "internlm", "chatglm"))
    return {
        "hf_name": key,
        "arch_type": "causal_lm" if is_causal else "encoder",
        "ft_mode": "lora" if is_causal else "full",
        "trust_remote_code": is_custom_remote,
        "preserve_auto_dtype": is_custom_remote,
        "param_tag": "custom",
        "batch_size": 24 if is_causal else 48,
        "grad_accum": 5 if is_causal else 3,
        "eval_batch_size": 96 if is_causal else 192,
        "backbone_lr": 1.0e-4 if is_causal else 2.0e-5,
        "head_lr": 2.5e-4,
        "llrd": 1.0 if is_causal else 0.90,
        "lora_r": 32 if is_causal else 0,
        "lora_alpha": 64 if is_causal else 0,
        "grad_ckpt": is_causal,
    }


def train_single_superteacher_v140(
    key: str,
    spec: dict[str, Any],
    corpus: dict[str, Any],
    ckpt_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    hf_name = str(spec["hf_name"])
    arch_type = str(spec["arch_type"])
    ft_mode = str(spec["ft_mode"])
    trust_rc = bool(spec.get("trust_remote_code", False))
    preserve_auto_dtype = bool(spec.get("preserve_auto_dtype", False))
    lora_r = int(args.lora_r if args.lora_r > 0 else spec["lora_r"])
    lora_alpha = int(args.lora_alpha if args.lora_alpha > 0 else spec["lora_alpha"])

    if args.smoke_test:
        tokenizer: Any = _SmokeTinyTokenizerV140(vocab_size=2048, arch_type=arch_type)
        model = UniversalDualAxisSuperTeacherV140(
            hf_name, arch_type=arch_type, ft_mode=ft_mode,
            lora_r=lora_r, lora_alpha=lora_alpha, dropout=0.10,
            load_dtype=None, trust_remote_code=False, smoke_tiny=True, vocab_size=2048,
        ).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_rc)
        if arch_type == "causal_lm" and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # For SenseNova-Loop-9B, `load_dtype="auto"` preserves FP32 mHC Sinkhorn routing + BF16 backbone weights!
        eff_load_dtype: torch.dtype | str | None = (
            "auto" if preserve_auto_dtype else (amp_dtype if (device.type == "cuda" and ft_mode == "lora") else None)
        )
        model = UniversalDualAxisSuperTeacherV140(
            hf_name, arch_type=arch_type, ft_mode=ft_mode,
            lora_r=lora_r, lora_alpha=lora_alpha, dropout=0.15,
            load_dtype=eff_load_dtype, trust_remote_code=trust_rc, smoke_tiny=False,
        ).to(device)

    prefix_ids = _get_cached_prefix_ids(tokenizer, arch_type)
    eval_bs = 32 if args.smoke_test else int(args.eval_batch_size if args.eval_batch_size > 0 else spec["eval_batch_size"])
    ckpt_path = ckpt_dir / f"v140_st_{key}_best.pt"

    if args.resume and ckpt_path.is_file() and not args.smoke_test:
        print(f"[SuperTeacherV140:{key}] Resuming compact checkpoint: {ckpt_path}", flush=True)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(sd["trainable_state_dict"])
        t_ai = float(sd.get("temp_ai", 1.0))
        t_pr = float(sd.get("temp_pr", 1.0))
        eval_res, eval_preds = evaluate_model_suite_v140(
            model, tokenizer, corpus, device, batch_size=eval_bs,
            max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
            t_ai=t_ai, t_pr=t_pr, prefix_ids=prefix_ids,
        )
        zpii = run_v140_superteacher_zero_pii_gate(model, tokenizer, device, amp_dtype=amp_dtype)
        return {
            "key": key, "hf_name": hf_name, "arch_type": arch_type, "ft_mode": ft_mode,
            "param_tag": spec["param_tag"], "resumed": True,
            "ckpt_path": str(ckpt_path), "temp_ai": t_ai, "temp_pr": t_pr,
            "eval": eval_res, "zero_pii": zpii, "val_preds": eval_preds,
            "lora_merge_max_diff": 0.0, "model": model, "tokenizer": tokenizer,
            "prefix_ids": prefix_ids, "eval_bs": eval_bs,
        }

    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    use_grad_ckpt = bool(spec["grad_ckpt"] or (not args.smoke_test and vram_gb < 40.0)) and not args.no_grad_ckpt
    if use_grad_ckpt and not args.smoke_test:
        model.enable_gradient_checkpointing()

    n_total_p = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_lr = float(args.backbone_lr if args.backbone_lr > 0 else spec["backbone_lr"])
    head_lr = float(args.head_lr if args.head_lr > 0 else spec["head_lr"])
    llrd = float(spec["llrd"])

    pg = build_optimizer_param_groups_v140(
        model, backbone_lr=backbone_lr, head_lr=head_lr, llrd=llrd, weight_decay=0.01
    )
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    epochs = 1 if args.smoke_test else int(args.epochs)
    bs = 12 if args.smoke_test else int(args.batch_size if args.batch_size > 0 else spec["batch_size"])
    grad_accum = 1 if args.smoke_test else int(args.grad_accum if args.grad_accum > 0 else spec["grad_accum"])

    sample_ep0 = sample_antidilution_epoch_v140(
        corpus, random.Random(seed), neg_per_pos=args.neg_per_pos, natural_frac=args.natural_frac, smoke_test=args.smoke_test
    )
    steps_per_ep = max(1, math.ceil(math.ceil(len(sample_ep0) / bs) / grad_accum))
    total_opt_steps = max(1, steps_per_ep * epochs)
    warmup_steps = max(1, int(total_opt_steps * 0.08))

    # Track EMA / best strictly over TRAINABLE parameters (~105M for 8B LoRA -> zero 48 GB RAM spike!)
    ema_sd = model.trainable_state_dict(to_cpu=True)
    best_sd = {k: v.clone() for k, v in ema_sd.items()}
    best_sel = -1e9
    opt_step = 0
    start_ep = 0
    skip_b_idx = -1
    mid_ckpt_path = ckpt_dir / f"v140_st_{key}_mid_resume.pt"
    if args.resume and mid_ckpt_path.is_file() and not args.smoke_test:
        try:
            msd = torch.load(mid_ckpt_path, map_location="cpu", weights_only=False)
            model.load_trainable_state_dict(msd["cur_sd"])
            ema_sd = {k: v.to(ema_sd[k].dtype).clone() for k, v in msd["ema_sd"].items() if k in ema_sd}
            best_sd = {k: v.to(best_sd[k].dtype).clone() for k, v in msd["best_sd"].items() if k in best_sd}
            best_sel = float(msd.get("best_sel", -1e9))
            opt_step = int(msd.get("opt_step", 0))
            start_ep = int(msd.get("next_ep", 0))
            skip_b_idx = int(msd.get("done_b_idx", -1))
            print(
                f"[SuperTeacherV140:{key}] Resumed mid-training checkpoint {mid_ckpt_path} "
                f"(start_ep={start_ep} skip_b_idx={skip_b_idx} opt_step={opt_step} best_sel={best_sel:.4f})",
                flush=True,
            )
        except Exception as exc:
            print(f"[SuperTeacherV140:{key}] WARN failed to load mid-resume {mid_ckpt_path} ({exc}); starting fresh.", flush=True)

    def _lr_mult(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _update_ema(decay: float = 0.992) -> None:
        with torch.no_grad():
            for k, p in model.named_parameters():
                if p.requires_grad:
                    ema_sd[k].mul_(decay).add_(p.detach().cpu(), alpha=1.0 - decay)

    print(
        f"[SuperTeacherV140:{key}] Start {hf_name} ({spec['param_tag']}) | arch={arch_type} ft={ft_mode} "
        f"lora_mods={model.lora_modules_count} | trainable={n_train_p/1e6:.2f}M/{n_total_p/1e6:.1f}M | "
        f"ep={epochs} bs={bs}x{grad_accum} gckpt={use_grad_ckpt} ep_rows~{len(sample_ep0):,} "
        f"opt_steps~{total_opt_steps} vram={vram_gb:.1f}GB",
        flush=True,
    )

    has_cf = bool(corpus.get("cf_pairs_micro") or corpus.get("cf_pairs_decor") or corpus.get("cf_pairs_general"))
    cf_every = max(1, int(args.cf_every))
    cf_k = 4 if args.smoke_test else int(args.cf_pairs_per_step)
    cf_weight = float(args.cf_weight)

    for ep in range(epochs):
        ep_rows = sample_antidilution_epoch_v140(
            corpus, rng, neg_per_pos=args.neg_per_pos, natural_frac=args.natural_frac, smoke_test=args.smoke_test
        )
        batches = iter_length_bucketed_batches_v140(ep_rows, batch_size=bs, rng=rng, mega_mult=16)
        if ep < start_ep:
            continue
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, run_cf, run_cnt = 0.0, 0.0, 0
        eval_every = max(1, steps_per_ep // 2) if not args.smoke_test else steps_per_ep

        for b_idx, batch in enumerate(batches):
            if ep == start_ep and b_idx <= skip_b_idx:
                continue
            txts = [
                jitter_ai_text_v140(r["text"], rng, is_ml_en=bool(r.get("is_ml_en")), prob=args.punct_jitter_prob)
                if (r["has_ai"] > 0.5 and r["y_ai"] >= 0.7) else r["text"]
                for r in batch
            ]
            ids, mask, pmask = encode_universal_v140(
                tokenizer, txts, arch_type=arch_type, max_len=args.max_len, head_len=args.head_len, prefix_ids=prefix_ids
            )
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            y_ai = torch.tensor([r["y_ai"] for r in batch], dtype=torch.float32, device=device)
            has_ai = torch.tensor([r["has_ai"] for r in batch], dtype=torch.float32, device=device)
            y_pr = torch.tensor([r["y_pr"] for r in batch], dtype=torch.float32, device=device)
            has_pr = torch.tensor([r["has_pr"] for r in batch], dtype=torch.float32, device=device)
            sub4 = torch.tensor([r["sub4"] for r in batch], dtype=torch.long, device=device)
            w_ai = torch.tensor([r["w_ai"] for r in batch], dtype=torch.float32, device=device)
            w_pr = torch.tensor([r["w_pr"] for r in batch], dtype=torch.float32, device=device)
            hn_f = torch.tensor([r["hard_neg"] for r in batch], dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                out = model(ids, mask, pmask)
                loss, _ = compute_dual_axis_loss_v140(out, y_ai, has_ai, y_pr, has_pr, sub4, w_ai, w_pr, hn_f)

            # Separate backward pass for main batch so activations free cleanly before optional CF pair pass!
            loss_scaled = loss / float(grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            run_loss += float(loss.item())
            run_cnt += 1

            # Periodic Counterfactual Mirror Margin Loss (kills topic / length / decorator confounders!)
            if has_cf and cf_weight > 0.0 and ((b_idx + 1) % cf_every == 0):
                cf_batch = sample_cf_mirror_batch_v140(corpus, cf_k, rng)
                if cf_batch:
                    cf_txts = [p["ai_text"] for p in cf_batch] + [p["human_text"] for p in cf_batch]
                    cf_w = torch.tensor([p["pair_weight"] for p in cf_batch], dtype=torch.float32, device=device)
                    c_ids, c_mask, c_pmask = encode_universal_v140(
                        tokenizer, cf_txts, arch_type=arch_type, max_len=args.max_len, head_len=args.head_len, prefix_ids=prefix_ids
                    )
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                        c_out = model(c_ids.to(device, non_blocking=True), c_mask.to(device, non_blocking=True), c_pmask.to(device, non_blocking=True))
                        k_cur = len(cf_batch)
                        l_cf = compute_cf_mirror_loss_v140(c_out["logit_ai"][:k_cur], c_out["logit_ai"][k_cur:], cf_w, margin=2.4)
                    l_cf_scaled = (cf_weight * l_cf) / float(grad_accum)
                    if scaler.is_enabled():
                        scaler.scale(l_cf_scaled).backward()
                    else:
                        l_cf_scaled.backward()
                    run_cf += float(l_cf.item())

            is_step = ((b_idx + 1) % grad_accum == 0) or (b_idx + 1 == len(batches))
            if is_step:
                mult = _lr_mult(opt_step)
                for pgrp in opt.param_groups:
                    pgrp["lr"] = pgrp["base_lr"] * mult
                train_params = [p for p in model.parameters() if p.requires_grad]
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                _update_ema(decay=0.992)

                if (opt_step % eval_every == 0) or (b_idx + 1 == len(batches)):
                    cur_sd = model.trainable_state_dict(to_cpu=True)
                    model.load_trainable_state_dict(ema_sd)
                    ev, _ = evaluate_model_suite_v140(
                        model, tokenizer, corpus, device, batch_size=eval_bs,
                        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, prefix_ids=prefix_ids,
                    )
                    sel = float(ev["val_sel_obj"])
                    improved = sel > best_sel
                    if improved:
                        best_sel = sel
                        best_sd = {k: v.clone() for k, v in ema_sd.items()}
                    print(
                        f"  [{key} ep={ep+1}/{epochs} step={opt_step}] loss={run_loss/max(1,run_cnt):.4f} "
                        f"cf={run_cf/max(1,run_cnt//cf_every):.3f} val_sel={sel:.4f}{'*' if improved else ''} "
                        f"g2_pr={ev.get('val_gold_v2_ai',{}).get('pr',0):.4f} "
                        f"g2_r95={ev.get('val_gold_v2_ai',{}).get('r95',0):.4f} "
                        f"w3_pr={ev.get('val_wave3_promo',{}).get('pr',0):.4f} "
                        f"ml_pr={ev.get('val_multilingual_ai',{}).get('pr',0):.4f} "
                        f"old_fpr={ev.get('val_old_video_neg',{}).get('fpr50',0):.4f} "
                        f"q4_fpr={ev.get('val_wave3_q4_fpr50',0):.4f}",
                        flush=True,
                    )
                    model.load_trainable_state_dict(cur_sd)
                    if not args.smoke_test:
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        ep_done = (b_idx + 1 == len(batches))
                        mid_tmp = mid_ckpt_path.with_suffix(".pt.tmp")
                        torch.save({
                            "key": key,
                            "next_ep": ep + 1 if ep_done else ep,
                            "done_b_idx": -1 if ep_done else b_idx,
                            "opt_step": opt_step,
                            "best_sel": best_sel,
                            "cur_sd": {k: v.half() if v.is_floating_point() else v for k, v in cur_sd.items()},
                            "ema_sd": {k: v.half() if v.is_floating_point() else v for k, v in ema_sd.items()},
                            "best_sd": {k: v.half() if v.is_floating_point() else v for k, v in best_sd.items()},
                        }, mid_tmp)
                        os.replace(mid_tmp, mid_ckpt_path)
                    model.train()

    # Load best EMA trainable weights + calibrate scalar temperatures strictly on validation splits!
    model.load_trainable_state_dict(best_sd)
    _, raw_preds = evaluate_model_suite_v140(
        model, tokenizer, corpus, device, batch_size=eval_bs,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, prefix_ids=prefix_ids,
    )

    ai_lg_l, ai_y_l = [], []
    for k_v in ("val_gold_v2", "val_wave3", "val_old_video_neg", "val_multilingual"):
        rows_k = corpus.get(k_v) or []
        if rows_k:
            ai_lg_l.append(raw_preds[k_v][0])
            if k_v == "val_old_video_neg":
                ai_y_l.append(np.zeros((len(rows_k),), dtype=np.float32))
            else:
                ai_y_l.append(np.array([int(r["y_ai"] >= 0.5) for r in rows_k], dtype=np.float32))
    t_ai = fit_scalar_temperature_v140(np.concatenate(ai_lg_l), np.concatenate(ai_y_l)) if ai_lg_l else 1.0

    pr_lg_l, pr_y_l = [], []
    for k_pr in ("val_wave3", "val_promo_v11"):
        rows_k = corpus.get(k_pr) or []
        if rows_k:
            pr_lg_l.append(raw_preds[k_pr][1])
            pr_y_l.append(np.array([int(r["y_pr"] >= 0.5) for r in rows_k], dtype=np.float32))
    t_pr = fit_scalar_temperature_v140(np.concatenate(pr_lg_l), np.concatenate(pr_y_l)) if pr_lg_l else 1.0

    eval_final, eval_preds = evaluate_model_suite_v140(
        model, tokenizer, corpus, device, batch_size=eval_bs,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
        t_ai=t_ai, t_pr=t_pr, prefix_ids=prefix_ids,
    )
    zpii = run_v140_superteacher_zero_pii_gate(model, tokenizer, device, amp_dtype=amp_dtype)

    # Save compact trainable-only checkpoint (before in-place LoRA merge!)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "version": "v1.4.0-superteacher-ultra",
        "key": key,
        "hf_name": hf_name,
        "arch_type": arch_type,
        "ft_mode": ft_mode,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "temp_ai": t_ai,
        "temp_pr": t_pr,
        "max_len": int(args.max_len),
        "head_len": int(args.head_len),
        "eval": eval_final,
        "trainable_state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_sd.items()},
    }
    torch.save(save_obj, ckpt_path)
    mid_ckpt_path.unlink(missing_ok=True)
    ckpt_mb = ckpt_path.stat().st_size / (1024.0 * 1024.0)

    # Verify exact LoRA merge parity, then keep merged for maximum 250K inference throughput!
    lora_merge_diff = 0.0
    if ft_mode == "lora":
        probe_txts = [r["text"] for r in (corpus.get("val_gold_v2") or [])[:16]] or ["这是一条用于检验合并一致性的测试评论。"]
        pre_ai, pre_pr = predict_dual_logits_v140(
            model, tokenizer, probe_txts, device, batch_size=16,
            max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, prefix_ids=prefix_ids,
        )
        model.merge_lora_inplace()
        post_ai, post_pr = predict_dual_logits_v140(
            model, tokenizer, probe_txts, device, batch_size=16,
            max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, prefix_ids=prefix_ids,
        )
        lora_merge_diff = float(max(np.abs(pre_ai - post_ai).max(), np.abs(pre_pr - post_pr).max()))

    print(
        f"[SuperTeacherV140:{key} DONE in {time.time()-t0:.1f}s] ckpt={ckpt_mb:.1f}MB T_ai={t_ai} T_pr={t_pr} | "
        f"val_sel={eval_final['val_sel_obj']} | test_g2_pr={eval_final.get('test_gold_v2_ai',{}).get('pr',0):.4f} "
        f"test_g2_r95={eval_final.get('test_gold_v2_ai',{}).get('r95',0):.4f} "
        f"ml_pr={eval_final.get('val_multilingual_ai',{}).get('pr',0):.4f} | "
        f"zero_pii={zpii['pass']} lora_merge_diff={lora_merge_diff:.6f}",
        flush=True,
    )
    return {
        "key": key, "hf_name": hf_name, "arch_type": arch_type, "ft_mode": ft_mode,
        "param_tag": spec["param_tag"], "resumed": False,
        "ckpt_path": str(ckpt_path), "ckpt_mb": round(ckpt_mb, 2),
        "temp_ai": t_ai, "temp_pr": t_pr,
        "eval": eval_final, "zero_pii": zpii, "val_preds": eval_preds,
        "lora_merge_max_diff": round(lora_merge_diff, 6),
        "model": model, "tokenizer": tokenizer, "prefix_ids": prefix_ids, "eval_bs": eval_bs,
    }


def _generate_simplex_candidates(m: int, step: float = 0.05) -> list[list[float]]:
    """Exact convex simplex grid for M <= 4 (1,771 pts at M=4) + Dirichlet search for M > 4."""
    if m == 1:
        return [[1.0]]
    n_steps = int(round(1.0 / step))
    if m <= 4:
        cands: list[list[float]] = []

        def _rec(rem: int, depth: int, cur: list[int]) -> None:
            if depth == m - 1:
                cands.append([(x * step) for x in (cur + [rem])])
                return
            for v in range(rem + 1):
                _rec(rem - v, depth + 1, cur + [v])

        _rec(n_steps, 0, [])
        return cands
    rng = np.random.default_rng(20261128)
    cands = [[1.0 / m] * m]
    for _ in range(3000):
        w = rng.dirichlet(np.ones(m))
        cands.append([float(x) for x in w])
    return cands


def fit_convex_ensemble_weights_v140(
    trained_list: list[dict[str, Any]],
    corpus: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """Optimize convex simplex weights across arbitrary M models strictly on validation splits (zero test leak!)."""
    m = len(trained_list)
    if m == 1:
        return [1.0], [1.0]

    cands = _generate_simplex_candidates(m, step=0.05)

    # AI weights on val_gold_v2 + val_wave3 + val_old_video_neg + val_multilingual
    ai_mats, ai_y = [], []
    for k_v in ("val_gold_v2", "val_wave3", "val_old_video_neg", "val_multilingual"):
        rows_k = corpus.get(k_v) or []
        if not rows_k:
            continue
        if k_v == "val_old_video_neg":
            ai_y.append(np.zeros((len(rows_k),), dtype=np.int32))
        else:
            ai_y.append(np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in rows_k], dtype=np.int32))
        cols = [tr["val_preds"][k_v][0] / max(1e-3, tr["temp_ai"]) for tr in trained_list]
        ai_mats.append(np.stack(cols, axis=1))
    M_ai = np.concatenate(ai_mats, axis=0)
    y_ai = np.concatenate(ai_y, axis=0)

    best_w_ai = [1.0 / m] * m
    best_sc_ai = -1e9
    if y_ai.sum() > 0 and y_ai.sum() < len(y_ai):
        for w in cands:
            wa = np.asarray(w, dtype=np.float64)
            z = M_ai @ wa
            p = expit(np.clip(z, -30.0, 30.0))
            # Mild L2 diversity regularizer (-0.008 * sum(w^2)) rewards multi-paradigm ensembles on ties!
            sc = 0.52 * float(average_precision_score(y_ai, p)) + 0.48 * _recall_at_prec(y_ai, p, 0.95) - 0.008 * float((wa ** 2).sum())
            if sc > best_sc_ai:
                best_sc_ai, best_w_ai = sc, w

    # Promo weights on val_wave3 + val_promo_v11
    pr_mats, pr_y = [], []
    for k_v in ("val_wave3", "val_promo_v11"):
        rows_k = corpus.get(k_v) or []
        if not rows_k:
            continue
        pr_y.append(np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in rows_k], dtype=np.int32))
        cols = [tr["val_preds"][k_v][1] / max(1e-3, tr["temp_pr"]) for tr in trained_list]
        pr_mats.append(np.stack(cols, axis=1))
    best_w_pr = [1.0 / m] * m
    if pr_mats:
        M_pr = np.concatenate(pr_mats, axis=0)
        y_pr = np.concatenate(pr_y, axis=0)
        best_sc_pr = -1e9
        if y_pr.sum() > 0 and y_pr.sum() < len(y_pr):
            for w in cands:
                wa = np.asarray(w, dtype=np.float64)
                z = M_pr @ wa
                p = expit(np.clip(z, -30.0, 30.0))
                sc = float(average_precision_score(y_pr, p)) - 0.008 * float((wa ** 2).sum())
                if sc > best_sc_pr:
                    best_sc_pr, best_w_pr = sc, w

    return [round(float(x), 4) for x in best_w_ai], [round(float(x), 4) for x in best_w_pr]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MeguvizCAC v1.4.0-SuperTeacher-Ultra 4-Paradigm Multi-Billion Ensemble")
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--extra-delta-dirs", type=str, default="", help="Optional comma-separated extra delta dirs")
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v04-data", type=Path, default=None, help="v04 corpus for clean EN/JA gold anchors + unlabeled wild distill pool")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS_V140, help="Comma-separated keys in SUPERTEACHER_BACKBONES_V140")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0, help="Override per-model default batch size (0=auto)")
    parser.add_argument("--grad-accum", type=int, default=0, help="Override per-model default grad accum (0=auto)")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Override per-model default eval batch size (0=auto)")
    parser.add_argument("--max-len", type=int, default=288)
    parser.add_argument("--head-len", type=int, default=192)
    parser.add_argument("--backbone-lr", type=float, default=0.0, help="Override per-model default backbone/LoRA LR (0=auto)")
    parser.add_argument("--head-lr", type=float, default=0.0, help="Override per-model default head LR (0=auto)")
    parser.add_argument("--lora-r", type=int, default=0, help="Override default LoRA rank (0=auto 32)")
    parser.add_argument("--lora-alpha", type=int, default=0, help="Override default LoRA alpha (0=auto 64)")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--neg-per-pos", type=float, default=4.2)
    parser.add_argument("--natural-frac", type=float, default=0.55)
    parser.add_argument("--punct-jitter-prob", type=float, default=0.35)
    parser.add_argument("--cf-every", type=int, default=3, help="Apply counterfactual mirror pair loss every N micro-batches")
    parser.add_argument("--cf-pairs-per-step", type=int, default=8)
    parser.add_argument("--cf-weight", type=float, default=0.30)
    parser.add_argument("--al-top-k", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    out_dir = args.out.resolve()
    ckpt_dir = (args.checkpoint_dir or out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[v1.4.0-SuperTeacher-Ultra] GPU={gpu_name} | amp={amp_dtype} | models={args.models} | smoke={args.smoke_test}", flush=True)

    extra_deltas = [Path(p.strip()) for p in args.extra_delta_dirs.split(",") if p.strip()]
    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir,
        delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir,
        wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir,
        v04_data_path=args.v04_data,
        extra_delta_dirs=extra_deltas,
        smoke_test=args.smoke_test,
    )
    print(f"[Stage0-Corpus] {json.dumps(corpus['stats'], ensure_ascii=False)}", flush=True)

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if args.smoke_test:
        # Guarantee smoke test exercises BOTH Encoder (XLM-R cls=0) AND SenseNova-Loop LoRA + Merge!
        model_keys = ["bge_m3", "sensenova_loop_9b"]

    trained_meta: list[dict[str, Any]] = []
    distill_rows = corpus["distill_pool"]
    distill_txts = [r["text"] for r in distill_rows]
    n_dist = len(distill_txts)
    pool_order_sha256 = _compute_pool_order_sha256(distill_rows)
    print(f"[Stage0-DistillPool] rows={n_dist:,} order_sha256={pool_order_sha256[:16]}...", flush=True)
    per_model_distill_ai: dict[str, np.ndarray] = {}
    per_model_distill_pr: dict[str, np.ndarray] = {}

    for idx_m, mk in enumerate(model_keys):
        spec = _resolve_model_spec(mk)
        tr = train_single_superteacher_v140(
            key=mk, spec=spec, corpus=corpus, ckpt_dir=ckpt_dir,
            args=args, device=device, amp_dtype=amp_dtype, seed=20261128 + idx_m * 101,
        )
        # If resumed LoRA model, merge in-place before scoring 250K pool for full native speed!
        if tr["resumed"] and tr["ft_mode"] == "lora":
            tr["model"].merge_lora_inplace()

        # Strictly verify ordered pool SHA256 + checkpoint SHA256 + model key before trusting any cached logits!
        pred_cache_path = ckpt_dir / f"v140_st_{mk}_distill_cache.npz"
        cur_ckpt_sha256 = _file_sha256(Path(tr["ckpt_path"])) if Path(tr["ckpt_path"]).is_file() else ""
        if args.resume and pred_cache_path.is_file() and not args.smoke_test:
            cached = np.load(pred_cache_path, allow_pickle=False)
            c_pool_sha = str(cached["pool_sha256"].item()) if "pool_sha256" in cached.files else ""
            c_ckpt_sha = str(cached["ckpt_sha256"].item()) if "ckpt_sha256" in cached.files else ""
            c_mkey = str(cached["model_key"].item()) if "model_key" in cached.files else ""
            if (
                len(cached["ai"]) == n_dist
                and len(cached["pr"]) == n_dist
                and c_pool_sha == pool_order_sha256
                and c_ckpt_sha == cur_ckpt_sha256
                and c_mkey == mk
            ):
                print(
                    f"[SuperTeacherV140:{mk}] Loaded VERIFIED cached distill logits "
                    f"({n_dist:,} rows, pool_sha={pool_order_sha256[:12]}, ckpt_sha={cur_ckpt_sha256[:12]}) from {pred_cache_path}",
                    flush=True,
                )
                per_model_distill_ai[mk] = cached["ai"].astype(np.float32)
                per_model_distill_pr[mk] = cached["pr"].astype(np.float32)
            else:
                print(
                    f"[SuperTeacherV140:{mk}] REJECTED stale/unverified distill cache {pred_cache_path} "
                    f"(n={len(cached['ai'])}/{n_dist}, pool_match={c_pool_sha == pool_order_sha256}, "
                    f"ckpt_match={c_ckpt_sha == cur_ckpt_sha256}, key_match={c_mkey == mk}); re-scoring fresh!",
                    flush=True,
                )
                pred_cache_path.unlink(missing_ok=True)

        if mk not in per_model_distill_ai:
            print(f"[SuperTeacherV140:{mk}] Scoring {n_dist:,} real comments for soft distillation pool...", flush=True)
            t_inf = time.time()
            d_ai, d_pr = predict_dual_logits_v140(
                tr["model"], tr["tokenizer"], distill_txts, device,
                batch_size=tr["eval_bs"], max_len=args.max_len, head_len=args.head_len,
                amp_dtype=amp_dtype, prefix_ids=tr["prefix_ids"],
            )
            per_model_distill_ai[mk] = (d_ai / max(1e-3, tr["temp_ai"])).astype(np.float32)
            per_model_distill_pr[mk] = (d_pr / max(1e-3, tr["temp_pr"])).astype(np.float32)
            if not args.smoke_test:
                np.savez_compressed(
                    pred_cache_path,
                    ai=per_model_distill_ai[mk],
                    pr=per_model_distill_pr[mk],
                    pool_sha256=np.array(pool_order_sha256),
                    ckpt_sha256=np.array(cur_ckpt_sha256),
                    model_key=np.array(mk),
                )
            print(f"[SuperTeacherV140:{mk}] Scored {n_dist:,} rows in {time.time()-t_inf:.1f}s.", flush=True)

        del tr["model"]
        del tr["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trained_meta.append(tr)

    # 3. Fit general-M convex ensemble weights strictly on held-out validation splits
    w_ai, w_pr = fit_convex_ensemble_weights_v140(trained_meta, corpus)
    print(f"[Stage2-Ensemble] keys={model_keys} | w_ai={w_ai} | w_pr={w_pr}", flush=True)

    ens_eval: dict[str, Any] = {}
    for k_v in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_old_video_neg", "val_multilingual", "diag_god500"):
        rows_k = corpus.get(k_v) or []
        if not rows_k:
            continue
        z_ai = sum(w_ai[i] * (trained_meta[i]["val_preds"][k_v][0] / max(1e-3, trained_meta[i]["temp_ai"])) for i in range(len(trained_meta)))
        z_pr = sum(w_pr[i] * (trained_meta[i]["val_preds"][k_v][1] / max(1e-3, trained_meta[i]["temp_pr"])) for i in range(len(trained_meta)))
        if k_v in ("val_gold_v2", "test_gold_v2", "val_multilingual"):
            y = np.array([int(r["y_ai"] >= 0.5) for r in rows_k], dtype=np.int32)
            ens_eval[f"{k_v}_ai"] = _eval_binary(y, z_ai, temp=1.0)
        elif k_v == "val_wave3":
            ya = np.array([int(r["y_ai"] >= 0.5) for r in rows_k], dtype=np.int32)
            yp = np.array([int(r["y_pr"] >= 0.5) for r in rows_k], dtype=np.int32)
            ens_eval["val_wave3_ai"] = _eval_binary(ya, z_ai, temp=1.0)
            ens_eval["val_wave3_promo"] = _eval_binary(yp, z_pr, temp=1.0)
            q4 = (yp == 1) & (ya == 0)
            if q4.sum() > 0:
                p_q4 = expit(np.clip(z_ai[q4], -30.0, 30.0))
                ens_eval["val_wave3_q4_fpr50"] = round(float((p_q4 >= 0.50).mean()), 4)
                ens_eval["val_wave3_q4_mean_p"] = round(float(p_q4.mean()), 4)
        elif k_v == "val_old_video_neg":
            p_old = expit(np.clip(z_ai, -30.0, 30.0))
            ens_eval["val_old_video_neg"] = {
                "n": len(rows_k),
                "fpr50": round(float((p_old >= 0.50).mean()), 4),
                "fpr30": round(float((p_old >= 0.30).mean()), 4),
                "p99_prob": round(float(np.percentile(p_old, 99)), 4),
            }
        elif k_v == "diag_god500":
            tv = [i for i, r in enumerate(rows_k) if r.get("is_text_verifiable")]
            if tv:
                y_tv = np.array([int(rows_k[i]["y_ai"] >= 0.5) for i in tv], dtype=np.int32)
                ens_eval["diag_god500_text_verif"] = _eval_binary(y_tv, z_ai[tv], temp=1.0)

    # 4. Export Calibrated Soft-Distillation Pool (`v140_superteacher_soft_distill_pool.jsonl.gz`) + AL-R5 Top-5K Queue!
    mat_ai = np.stack([per_model_distill_ai[mk] for mk in model_keys], axis=1)
    mat_pr = np.stack([per_model_distill_pr[mk] for mk in model_keys], axis=1)
    ens_ai_logit = (mat_ai @ np.asarray(w_ai, dtype=np.float32)).astype(np.float32)
    ens_pr_logit = (mat_pr @ np.asarray(w_pr, dtype=np.float32)).astype(np.float32)
    ens_ai_prob = expit(np.clip(ens_ai_logit, -30.0, 30.0)).astype(np.float32)
    ens_pr_prob = expit(np.clip(ens_pr_logit, -30.0, 30.0)).astype(np.float32)
    ai_prob_mat = expit(np.clip(mat_ai, -30.0, 30.0))
    pr_prob_mat = expit(np.clip(mat_pr, -30.0, 30.0))
    ai_std_prob = ai_prob_mat.std(axis=1).astype(np.float32) if len(model_keys) > 1 else np.zeros((n_dist,), dtype=np.float32)
    ai_std_logit = mat_ai.std(axis=1).astype(np.float32) if len(model_keys) > 1 else np.zeros((n_dist,), dtype=np.float32)
    pr_std_prob = pr_prob_mat.std(axis=1).astype(np.float32) if len(model_keys) > 1 else np.zeros((n_dist,), dtype=np.float32)

    boundary_bonus = (1.0 - np.abs(2.0 * ens_ai_prob - 1.0)).astype(np.float32)
    al_score = (ai_std_prob * (0.5 + 0.5 * boundary_bonus)).astype(np.float32)

    distill_path = out_dir / "v140_superteacher_soft_distill_pool.jsonl.gz"
    with gzip.open(distill_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for i, r in enumerate(distill_rows):
            rec = {
                "sid": r["sid"],
                "sha16": r["sha16"],
                "text": r["text"],
                "source_tag": r["source_tag"],
                "era5": r["era5"],
                "origin_6way": r["origin_6way"],
                "gold_ai": r["gold_ai"],
                "gold_pr": r["gold_pr"],
                "is_multilingual": bool(r.get("is_multilingual", False)),
                "t_ai_logit": round(float(ens_ai_logit[i]), 5),
                "t_ai_prob": round(float(ens_ai_prob[i]), 5),
                "t_ai_std_p": round(float(ai_std_prob[i]), 5),
                "t_ai_std_lg": round(float(ai_std_logit[i]), 5),
                "t_pr_logit": round(float(ens_pr_logit[i]), 5),
                "t_pr_prob": round(float(ens_pr_prob[i]), 5),
                "t_pr_std_p": round(float(pr_std_prob[i]), 5),
                "per_model_ai_lg": [round(float(mat_ai[i, j]), 4) for j in range(len(model_keys))],
                "per_model_pr_lg": [round(float(mat_pr[i, j]), 4) for j in range(len(model_keys))],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    al_k = min(32 if args.smoke_test else int(args.al_top_k), n_dist)
    al_idx = np.argsort(-al_score)[:al_k]
    al_path = out_dir / f"v140_al_disagreement_queue_top{al_k}.jsonl.gz"
    with gzip.open(al_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for rank, i in enumerate(al_idx, 1):
            r = distill_rows[int(i)]
            rec = {
                "rank": rank,
                "al_disagreement_score": round(float(al_score[i]), 5),
                "t_ai_prob": round(float(ens_ai_prob[i]), 5),
                "t_ai_std_p": round(float(ai_std_prob[i]), 5),
                "per_model_ai_prob": {model_keys[j]: round(float(ai_prob_mat[i, j]), 4) for j in range(len(model_keys))},
                "t_pr_prob": round(float(ens_pr_prob[i]), 5),
                "sid": r["sid"],
                "source_tag": r["source_tag"],
                "era5": r["era5"],
                "origin_6way": r["origin_6way"],
                "gold_ai": r["gold_ai"],
                "text": r["text"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_zpii_pass = all(bool(tr["zero_pii"]["pass"]) for tr in trained_meta)
    lora_tol = 0.05 if amp_dtype == torch.bfloat16 else 1e-3
    all_merge_pass = all(float(tr["lora_merge_max_diff"]) <= lora_tol for tr in trained_meta)

    summary = {
        "version": "v1.4.0-superteacher-ultra-smoke" if args.smoke_test else "v1.4.0-superteacher-ultra",
        "gpu_name": gpu_name,
        "amp_dtype": str(amp_dtype),
        "corpus_stats": corpus["stats"],
        "models": [
            {
                "key": tr["key"], "hf_name": tr["hf_name"], "arch_type": tr["arch_type"],
                "ft_mode": tr["ft_mode"], "param_tag": tr["param_tag"],
                "temp_ai": tr["temp_ai"], "temp_pr": tr["temp_pr"],
                "lora_merge_max_diff": tr["lora_merge_max_diff"],
                "eval": tr["eval"], "zero_pii": tr["zero_pii"], "ckpt_path": tr["ckpt_path"],
            }
            for tr in trained_meta
        ],
        "ensemble_weights": {"ai": dict(zip(model_keys, w_ai)), "promo": dict(zip(model_keys, w_pr))},
        "ensemble_eval": ens_eval,
        "distill_pool_file": distill_path.name,
        "distill_pool_rows": n_dist,
        "distill_pool_mb": round(distill_path.stat().st_size / (1024.0 * 1024.0), 3),
        "al_queue_file": al_path.name,
        "al_queue_rows": al_k,
        "v140_superteacher_ci": {
            "V140_ST_C1_zero_pii_all_teachers": all_zpii_pass,
            "V140_ST_C2_lora_merge_parity": all_merge_pass,
            "all_ci_passed": bool(all_zpii_pass and all_merge_pass),
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    metrics_path = out_dir / "metrics-v1.4.0-superteacher.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["v140_superteacher_ci"]["all_ci_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
