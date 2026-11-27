"""MeguvizCAC v1.3.0-ApexKD 3-Model 24L Chinese Super-Teacher Trainer & Soft-Distillation Pool Exporter.

Trains 3 heterogeneous 24-layer Chinese pretrained encoders (`bf16` + LLRD + 35% online
punctuation-shortcut jitter + Top-Tail Pairwise Margin Loss + Anti-Dilution 65/35 real sampling):
  1. `electra_large` : `hfl/chinese-electra-180g-large-discriminator`
  2. `macbert_large` : `hfl/chinese-macbert-large`
  3. `bge_large`     : `BAAI/bge-large-zh-v1.5`

Outputs (zero OjinjinBench contact!):
  - Per-model calibrated checkpoints (`best_<key>.pt`) + `superteacher_ensemble_manifest.json`
  - `v130_superteacher_soft_distill_pool.jsonl.gz` (~180K real Bilibili comments with ensemble
    calibrated AI + Promo logits, per-model logits, committee std, and gold labels for v1.3.0 KD!)
  - `v130_al_disagreement_queue_top3000.jsonl.gz` (Top-3000 highest-disagreement real comments
    for AL-R4 sibling gold adjudication!)
  - `metrics-v1.3.0-superteacher.json` (val_gold_v2, test_gold_v2, val_wave3 Q4 promo firewall,
    val_old_video_neg FPR, diag_god500, and V130_ST_C1 Zero-PII === 0.0!)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer

from .v130_superteacher import (
    SUPERTEACHER_BACKBONES_V130,
    DualAxisSuperTeacherNet,
    build_llrd_param_groups_v130,
    build_v130_superteacher_corpus,
    compute_dual_axis_superteacher_loss_v130,
    encode_head_tail_hf_v130,
    fit_scalar_temperature_v130,
    jitter_ai_punctuation_v130,
    run_v130_superteacher_zero_pii_gate,
    sample_antidilution_epoch_v130,
)


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


@torch.no_grad()
def predict_dual_logits_v130(
    model: DualAxisSuperTeacherNet,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    max_len: int = 256,
    head_len: int = 180,
    amp_dtype: torch.dtype = torch.bfloat16,
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

    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_t = [texts[i] for i in idxs]
        ids, mask = encode_head_tail_hf_v130(tokenizer, sub_t, max_len=max_len, head_len=head_len)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            od = model(ids, mask)
        la = od["logit_ai"].float().cpu().numpy()
        lp = od["logit_promo"].float().cpu().numpy()
        for j, oi in enumerate(idxs):
            out_ai[oi] = la[j]
            out_pr[oi] = lp[j]
    return out_ai, out_pr


class _SmokeTinyTokenizer:
    """Zero-network deterministic UTF-16 char tokenizer strictly for local `--smoke-test`."""

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        out = []
        for t in texts:
            ids = [104 + (ord(ch) % (self.vocab_size - 110)) for ch in str(t)]
            out.append(ids)
        return {"input_ids": out}


def evaluate_model_suite_v130(
    model: DualAxisSuperTeacherNet,
    tokenizer: Any,
    corpus: dict[str, Any],
    device: torch.device,
    batch_size: int,
    max_len: int,
    head_len: int,
    amp_dtype: torch.dtype,
    t_ai: float = 1.0,
    t_pr: float = 1.0,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "diag_god500"):
        rows = corpus.get(key) or []
        txts = [r["text"] for r in rows]
        preds[key] = predict_dual_logits_v130(model, tokenizer, txts, device, batch_size=batch_size, max_len=max_len, head_len=head_len, amp_dtype=amp_dtype)

    res: dict[str, Any] = {}
    # 1. val_gold_v2 AI
    vg2 = corpus["val_gold_v2"]
    if vg2:
        y = np.array([int(r["y_ai"] >= 0.5) for r in vg2], dtype=np.int32)
        res["val_gold_v2_ai"] = _eval_binary(y, preds["val_gold_v2"][0], temp=t_ai)
    # 2. test_gold_v2 AI
    tg2 = corpus["test_gold_v2"]
    if tg2:
        y = np.array([int(r["y_ai"] >= 0.5) for r in tg2], dtype=np.int32)
        res["test_gold_v2_ai"] = _eval_binary(y, preds["test_gold_v2"][0], temp=t_ai)
    # 3. val_wave3 AI + Promo + Q4 Human-Promo Firewall FPR!
    vw3 = corpus["val_wave3"]
    if vw3:
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
    # 4. val_old_video_neg (>1y & pre_llm human FPR guard!)
    vold = corpus["val_old_video_neg"]
    if vold:
        la = preds["val_old_video_neg"][0]
        p_old = expit(np.clip(la / max(1e-3, t_ai), -30.0, 30.0))
        res["val_old_video_neg"] = {
            "n": len(vold),
            "fpr50": round(float((p_old >= 0.50).mean()), 4),
            "fpr30": round(float((p_old >= 0.30).mean()), 4),
            "p99_prob": round(float(np.percentile(p_old, 99)), 4),
            "mean_prob": round(float(p_old.mean()), 4),
        }
    # 5. Composite selection objective (STRICTLY uses val_* ONLY — never test_gold_v2!)
    g2_pr = res.get("val_gold_v2_ai", {}).get("pr", 0.0)
    g2_r95 = res.get("val_gold_v2_ai", {}).get("r95", 0.0)
    w3_ai_pr = res.get("val_wave3_ai", {}).get("pr", 0.0)
    w3_pr_pr = res.get("val_wave3_promo", {}).get("pr", 0.0)
    old_fpr = res.get("val_old_video_neg", {}).get("fpr50", 0.0)
    q4_fpr = res.get("val_wave3_q4_fpr50", 0.0)
    sel_obj = (
        0.35 * g2_pr + 0.25 * g2_r95 + 0.20 * w3_ai_pr + 0.20 * w3_pr_pr
        - 1.50 * old_fpr - 0.60 * q4_fpr
    )
    res["val_sel_obj"] = round(float(sel_obj), 5)

    # 6. Diagnostic god500 text-verifiable (read-only diagnostic!)
    dg = corpus["diag_god500"]
    if dg:
        la = preds["diag_god500"][0]
        tv = [i for i, r in enumerate(dg) if r.get("is_text_verifiable")]
        if tv:
            y_tv = np.array([int(dg[i]["y_ai"] >= 0.5) for i in tv], dtype=np.int32)
            res["diag_god500_text_verif"] = _eval_binary(y_tv, la[tv], temp=t_ai)
    return res, preds


def train_single_superteacher_v130(
    key: str,
    hf_name: str,
    corpus: dict[str, Any],
    out_dir: Path,
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

    if args.smoke_test:
        tokenizer: Any = _SmokeTinyTokenizer(vocab_size=2048)
        model = DualAxisSuperTeacherNet(hf_name, dropout=0.10, smoke_tiny=True, vocab_size=2048).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = DualAxisSuperTeacherNet(hf_name, dropout=0.15, smoke_tiny=False).to(device)

    ckpt_path = ckpt_dir / f"v130_superteacher_{key}_best.pt"
    if args.resume and ckpt_path.is_file() and not args.smoke_test:
        print(f"[SuperTeacher:{key}] Resuming existing trained checkpoint: {ckpt_path}", flush=True)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["state_dict"])
        t_ai = float(sd.get("temp_ai", 1.0))
        t_pr = float(sd.get("temp_pr", 1.0))
        eval_res, eval_preds = evaluate_model_suite_v130(
            model, tokenizer, corpus, device, batch_size=args.eval_batch_size,
            max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, t_ai=t_ai, t_pr=t_pr,
        )
        zpii = run_v130_superteacher_zero_pii_gate(model, tokenizer, device)
        return {
            "key": key, "hf_name": hf_name, "resumed": True,
            "ckpt_path": str(ckpt_path), "temp_ai": t_ai, "temp_pr": t_pr,
            "eval": eval_res, "zero_pii": zpii, "val_preds": eval_preds,
            "model": model, "tokenizer": tokenizer,
        }

    # Enable gradient checkpointing if VRAM < 40 GB (RTX 4060 Ti); keep off on 96 GB RTX PRO 6000 for max speed!
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    if not args.smoke_test and vram_gb < 40.0 and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    pg = build_llrd_param_groups_v130(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        llrd=args.llrd,
        weight_decay=0.01,
    )
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    epochs = 1 if args.smoke_test else int(args.epochs)
    bs = 16 if args.smoke_test else int(args.batch_size)
    grad_accum = 1 if args.smoke_test else int(args.grad_accum)

    # Estimate total optimizer steps for cosine schedule
    sample_ep0 = sample_antidilution_epoch_v130(corpus, random.Random(seed), neg_per_pos=args.neg_per_pos, natural_frac=args.natural_frac, smoke_test=args.smoke_test)
    steps_per_ep = max(1, math.ceil(len(sample_ep0) / (bs * grad_accum)))
    total_opt_steps = max(1, steps_per_ep * epochs)
    warmup_steps = max(1, int(total_opt_steps * 0.08))

    ema_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_sel = -1e9
    opt_step = 0

    def _lr_mult(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _update_ema(decay: float = 0.992) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_sd[k].mul_(decay).add_(v.detach().cpu(), alpha=1.0 - decay)
                else:
                    ema_sd[k].copy_(v.detach().cpu())

    print(
        f"[SuperTeacher:{key}] Start {hf_name} | epochs={epochs} bs={bs}x{grad_accum} "
        f"ep_rows~{len(sample_ep0):,} opt_steps~{total_opt_steps} vram={vram_gb:.1f}GB",
        flush=True,
    )

    for ep in range(epochs):
        ep_rows = sample_antidilution_epoch_v130(
            corpus, rng, neg_per_pos=args.neg_per_pos, natural_frac=args.natural_frac, smoke_test=args.smoke_test
        )
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, run_cnt = 0.0, 0
        eval_every = max(1, steps_per_ep // 2) if not args.smoke_test else steps_per_ep

        for b_idx, st in enumerate(range(0, len(ep_rows), bs)):
            batch = ep_rows[st : st + bs]
            # Online 35% punctuation-shortcut jitter strictly on AI positives!
            txts = [
                jitter_ai_punctuation_v130(r["text"], rng, prob=args.punct_jitter_prob)
                if (r["has_ai"] > 0.5 and r["y_ai"] >= 0.7) else r["text"]
                for r in batch
            ]
            ids, mask = encode_head_tail_hf_v130(tokenizer, txts, max_len=args.max_len, head_len=args.head_len)
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y_ai = torch.tensor([r["y_ai"] for r in batch], dtype=torch.float32, device=device)
            has_ai = torch.tensor([r["has_ai"] for r in batch], dtype=torch.float32, device=device)
            y_pr = torch.tensor([r["y_pr"] for r in batch], dtype=torch.float32, device=device)
            has_pr = torch.tensor([r["has_pr"] for r in batch], dtype=torch.float32, device=device)
            sub4 = torch.tensor([r["sub4"] for r in batch], dtype=torch.long, device=device)
            w_ai = torch.tensor([r["w_ai"] for r in batch], dtype=torch.float32, device=device)
            w_pr = torch.tensor([r["w_pr"] for r in batch], dtype=torch.float32, device=device)
            hn_f = torch.tensor([r["hard_neg"] for r in batch], dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                out = model(ids, mask)
                loss, linfo = compute_dual_axis_superteacher_loss_v130(
                    out, y_ai, has_ai, y_pr, has_pr, sub4, w_ai, w_pr, hn_f
                )
                loss_scaled = loss / float(grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            run_loss += float(loss.item())
            run_cnt += 1

            is_step = ((b_idx + 1) % grad_accum == 0) or (st + bs >= len(ep_rows))
            if is_step:
                mult = _lr_mult(opt_step)
                for pgrp in opt.param_groups:
                    pgrp["lr"] = pgrp["base_lr"] * mult
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                _update_ema(decay=0.992)

                if (opt_step % eval_every == 0) or (st + bs >= len(ep_rows)):
                    # Evaluate EMA weights for rock-solid generalization
                    cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(ema_sd)
                    ev, _ = evaluate_model_suite_v130(
                        model, tokenizer, corpus, device, batch_size=args.eval_batch_size,
                        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
                    )
                    sel = float(ev["val_sel_obj"])
                    improved = sel > best_sel
                    if improved:
                        best_sel = sel
                        best_sd = {k: v.clone() for k, v in ema_sd.items()}
                    print(
                        f"  [{key} ep={ep+1}/{epochs} step={opt_step}] loss={run_loss/max(1,run_cnt):.4f} "
                        f"val_sel={sel:.4f}{'*' if improved else ''} "
                        f"g2_val_pr={ev.get('val_gold_v2_ai',{}).get('pr',0):.4f} "
                        f"g2_val_r95={ev.get('val_gold_v2_ai',{}).get('r95',0):.4f} "
                        f"w3_pr={ev.get('val_wave3_promo',{}).get('pr',0):.4f} "
                        f"old_fpr={ev.get('val_old_video_neg',{}).get('fpr50',0):.4f} "
                        f"q4_fpr={ev.get('val_wave3_q4_fpr50',0):.4f}",
                        flush=True,
                    )
                    model.load_state_dict(cur_sd)
                    model.train()

    # Load best EMA checkpoint + fit scalar temperatures strictly on validation splits!
    model.load_state_dict(best_sd)
    _, raw_preds = evaluate_model_suite_v130(
        model, tokenizer, corpus, device, batch_size=args.eval_batch_size,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
    )

    # Fit AI calibration temperature on (val_gold_v2 + val_wave3 + val_old_video_neg)
    ai_lg_l, ai_y_l = [], []
    if corpus["val_gold_v2"]:
        ai_lg_l.append(raw_preds["val_gold_v2"][0])
        ai_y_l.append(np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.float32))
    if corpus["val_wave3"]:
        ai_lg_l.append(raw_preds["val_wave3"][0])
        ai_y_l.append(np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.float32))
    if corpus["val_old_video_neg"]:
        ai_lg_l.append(raw_preds["val_old_video_neg"][0])
        ai_y_l.append(np.zeros((len(corpus["val_old_video_neg"]),), dtype=np.float32))
    t_ai = fit_scalar_temperature_v130(np.concatenate(ai_lg_l), np.concatenate(ai_y_l)) if ai_lg_l else 1.0

    # Fit Promo calibration temperature on (val_wave3 + val_promo_v11)
    pr_lg_l, pr_y_l = [], []
    for k_pr in ("val_wave3", "val_promo_v11"):
        rows_k = corpus.get(k_pr) or []
        if rows_k:
            pr_lg_l.append(raw_preds[k_pr][1])
            pr_y_l.append(np.array([int(r["y_pr"] >= 0.5) for r in rows_k], dtype=np.float32))
    t_pr = fit_scalar_temperature_v130(np.concatenate(pr_lg_l), np.concatenate(pr_y_l)) if pr_lg_l else 1.0

    eval_final, eval_preds = evaluate_model_suite_v130(
        model, tokenizer, corpus, device, batch_size=args.eval_batch_size,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype, t_ai=t_ai, t_pr=t_pr,
    )
    zpii = run_v130_superteacher_zero_pii_gate(model, tokenizer, device)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "key": key,
        "hf_name": hf_name,
        "temp_ai": t_ai,
        "temp_pr": t_pr,
        "max_len": int(args.max_len),
        "head_len": int(args.head_len),
        "eval": eval_final,
        "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_sd.items()},
    }
    torch.save(save_obj, ckpt_path)
    print(
        f"[SuperTeacher:{key} DONE in {time.time()-t0:.1f}s] T_ai={t_ai} T_pr={t_pr} | "
        f"val_sel={eval_final['val_sel_obj']} | test_g2_pr={eval_final.get('test_gold_v2_ai',{}).get('pr',0):.4f} "
        f"test_g2_r95={eval_final.get('test_gold_v2_ai',{}).get('r95',0):.4f} | zero_pii={zpii['pass']}",
        flush=True,
    )
    return {
        "key": key, "hf_name": hf_name, "resumed": False,
        "ckpt_path": str(ckpt_path), "temp_ai": t_ai, "temp_pr": t_pr,
        "eval": eval_final, "zero_pii": zpii, "val_preds": eval_preds,
        "model": model, "tokenizer": tokenizer,
    }


def fit_convex_ensemble_weights_v130(
    trained_list: list[dict[str, Any]],
    corpus: dict[str, Any],
) -> tuple[list[float], list[float]]:
    """Optimize convex simplex weights on held-out validation splits (strictly zero test leakage)."""
    m = len(trained_list)
    if m == 1:
        return [1.0], [1.0]

    # Build AI validation logits (calibrated by each model's T_ai)
    ai_mats, ai_y = [], []
    for k_v in ("val_gold_v2", "val_wave3", "val_old_video_neg"):
        rows_k = corpus.get(k_v) or []
        if not rows_k:
            continue
        ai_y.append(np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in rows_k], dtype=np.int32))
        cols = [
            tr["val_preds"][k_v][0] / max(1e-3, tr["temp_ai"])
            for tr in trained_list
        ]
        ai_mats.append(np.stack(cols, axis=1))
    M_ai = np.concatenate(ai_mats, axis=0)
    y_ai = np.concatenate(ai_y, axis=0)

    # Grid search over convex simplex (step 0.05) maximizing PR-AUC + R@P95!
    best_w_ai = [1.0 / m] * m
    best_sc_ai = -1e9
    grid_vals = [round(i * 0.05, 2) for i in range(21)]
    cands: list[list[float]] = []
    if m == 2:
        cands = [[a, round(1.0 - a, 2)] for a in grid_vals]
    elif m == 3:
        for a in grid_vals:
            for b in grid_vals:
                c = round(1.0 - a - b, 2)
                if c >= -1e-6:
                    cands.append([a, b, max(0.0, c)])
    else:
        cands = [[1.0 / m] * m]

    for w in cands:
        z = M_ai @ np.asarray(w, dtype=np.float64)
        p = expit(np.clip(z, -30.0, 30.0))
        if y_ai.sum() > 0 and y_ai.sum() < len(y_ai):
            sc = 0.55 * float(average_precision_score(y_ai, p)) + 0.45 * _recall_at_prec(y_ai, p, 0.95)
            if sc > best_sc_ai:
                best_sc_ai, best_w_ai = sc, w

    # Promo ensemble weights on (val_wave3 + val_promo_v11)
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
                z = M_pr @ np.asarray(w, dtype=np.float64)
                p = expit(np.clip(z, -30.0, 30.0))
                sc = float(average_precision_score(y_pr, p))
                if sc > best_sc_pr:
                    best_sc_pr, best_w_pr = sc, w
    return [round(float(x), 4) for x in best_w_ai], [round(float(x), 4) for x in best_w_pr]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MeguvizCAC v1.3.0-ApexKD 3-Model 24L Chinese Super-Teacher Ensemble")
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v04-data", type=Path, default=None, help="Optional v04 corpus strictly for unlabeled real-wild distillation pool")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--models", type=str, default="electra_large,macbert_large,bge_large", help="Comma-separated keys in SUPERTEACHER_BACKBONES_V130")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--head-len", type=int, default=180)
    parser.add_argument("--backbone-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=2.5e-4)
    parser.add_argument("--llrd", type=float, default=0.90)
    parser.add_argument("--neg-per-pos", type=float, default=4.5)
    parser.add_argument("--natural-frac", type=float, default=0.65)
    parser.add_argument("--punct-jitter-prob", type=float, default=0.35)
    parser.add_argument("--al-top-k", type=int, default=3000)
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
    print(f"[v1.3.0-SuperTeacher] GPU={gpu_name} | amp={amp_dtype} | models={args.models} | smoke={args.smoke_test}", flush=True)

    # 1. Assemble strictly purified, zero-leakage real corpus + full distillation pool
    corpus = build_v130_superteacher_corpus(
        sibling_dir=args.sibling_dir,
        delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir,
        wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir,
        v04_data_path=args.v04_data,
        smoke_test=args.smoke_test,
    )
    print(f"[Stage0-Corpus] {json.dumps(corpus['stats'], ensure_ascii=False)}", flush=True)

    # 2. Train each heterogeneous Super-Teacher backbone sequentially (maximizing VRAM headroom)
    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if args.smoke_test:
        model_keys = model_keys[:2]  # Verify multi-teacher ensembling + disagreement std in smoke-test!

    trained_meta: list[dict[str, Any]] = []
    distill_rows = corpus["distill_pool"]
    distill_txts = [r["text"] for r in distill_rows]
    n_dist = len(distill_txts)
    per_model_distill_ai: dict[str, np.ndarray] = {}
    per_model_distill_pr: dict[str, np.ndarray] = {}

    for idx_m, mk in enumerate(model_keys):
        hf_name = SUPERTEACHER_BACKBONES_V130.get(mk, mk)
        tr = train_single_superteacher_v130(
            key=mk, hf_name=hf_name, corpus=corpus, out_dir=out_dir, ckpt_dir=ckpt_dir,
            args=args, device=device, amp_dtype=amp_dtype, seed=20261128 + idx_m * 101,
        )
        # Immediately score the full ~180K distillation pool while this model is warm on GPU, then free VRAM!
        print(f"[SuperTeacher:{mk}] Scoring {n_dist:,} real comments for soft distillation pool...", flush=True)
        t_inf = time.time()
        d_ai, d_pr = predict_dual_logits_v130(
            tr["model"], tr["tokenizer"], distill_txts, device,
            batch_size=args.eval_batch_size, max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
        )
        per_model_distill_ai[mk] = (d_ai / max(1e-3, tr["temp_ai"])).astype(np.float32)
        per_model_distill_pr[mk] = (d_pr / max(1e-3, tr["temp_pr"])).astype(np.float32)
        print(f"[SuperTeacher:{mk}] Scored {n_dist:,} rows in {time.time()-t_inf:.1f}s.", flush=True)

        # Release GPU weights before loading next 24L Large backbone
        del tr["model"]
        del tr["tokenizer"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trained_meta.append(tr)

    # 3. Fit optimal convex ensemble weights strictly on held-out validation splits
    w_ai, w_pr = fit_convex_ensemble_weights_v130(trained_meta, corpus)
    print(f"[Stage2-Ensemble] keys={model_keys} | w_ai={w_ai} | w_pr={w_pr}", flush=True)

    # Evaluate Super-Teacher Ensemble across all validation & held-out test suites
    ens_eval: dict[str, Any] = {}
    for k_v in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_old_video_neg", "diag_god500"):
        rows_k = corpus.get(k_v) or []
        if not rows_k:
            continue
        z_ai = sum(w_ai[i] * (trained_meta[i]["val_preds"][k_v][0] / max(1e-3, trained_meta[i]["temp_ai"])) for i in range(len(trained_meta)))
        z_pr = sum(w_pr[i] * (trained_meta[i]["val_preds"][k_v][1] / max(1e-3, trained_meta[i]["temp_pr"])) for i in range(len(trained_meta)))
        if k_v in ("val_gold_v2", "test_gold_v2"):
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

    # 4. Export Full-Corpus Calibrated Soft-Logit Distillation Pool (`v130_superteacher_soft_distill_pool.jsonl.gz`)
    #    + Top-K Highest Committee Disagreement Active Learning Queue (`v130_al_disagreement_queue_top3000.jsonl.gz`)!
    mat_ai = np.stack([per_model_distill_ai[mk] for mk in model_keys], axis=1)  # (N, M)
    mat_pr = np.stack([per_model_distill_pr[mk] for mk in model_keys], axis=1)  # (N, M)
    ens_ai_logit = (mat_ai @ np.asarray(w_ai, dtype=np.float32)).astype(np.float32)
    ens_pr_logit = (mat_pr @ np.asarray(w_pr, dtype=np.float32)).astype(np.float32)
    ens_ai_prob = expit(np.clip(ens_ai_logit, -30.0, 30.0)).astype(np.float32)
    ens_pr_prob = expit(np.clip(ens_pr_logit, -30.0, 30.0)).astype(np.float32)
    ai_prob_mat = expit(np.clip(mat_ai, -30.0, 30.0))
    ai_std_prob = ai_prob_mat.std(axis=1).astype(np.float32) if len(model_keys) > 1 else np.zeros((n_dist,), dtype=np.float32)
    ai_std_logit = mat_ai.std(axis=1).astype(np.float32) if len(model_keys) > 1 else np.zeros((n_dist,), dtype=np.float32)

    # Active-learning disagreement score peaks near boundary (0.15..0.85) with max inter-model disagreement!
    boundary_bonus = (1.0 - np.abs(2.0 * ens_ai_prob - 1.0)).astype(np.float32)
    al_score = (ai_std_prob * (0.5 + 0.5 * boundary_bonus)).astype(np.float32)

    distill_path = out_dir / "v130_superteacher_soft_distill_pool.jsonl.gz"
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
                "t_ai_logit": round(float(ens_ai_logit[i]), 5),
                "t_ai_prob": round(float(ens_ai_prob[i]), 5),
                "t_ai_std_p": round(float(ai_std_prob[i]), 5),
                "t_ai_std_lg": round(float(ai_std_logit[i]), 5),
                "t_pr_logit": round(float(ens_pr_logit[i]), 5),
                "t_pr_prob": round(float(ens_pr_prob[i]), 5),
                "per_model_ai_lg": [round(float(mat_ai[i, j]), 4) for j in range(len(model_keys))],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    al_k = min(32 if args.smoke_test else int(args.al_top_k), n_dist)
    al_idx = np.argsort(-al_score)[:al_k]
    al_path = out_dir / f"v130_al_disagreement_queue_top{al_k}.jsonl.gz"
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
    summary = {
        "version": "v1.3.0-superteacher-smoke" if args.smoke_test else "v1.3.0-superteacher",
        "gpu_name": gpu_name,
        "corpus_stats": corpus["stats"],
        "models": [
            {
                "key": tr["key"], "hf_name": tr["hf_name"], "temp_ai": tr["temp_ai"], "temp_pr": tr["temp_pr"],
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
        "v130_superteacher_ci": {
            "V130_ST_C1_zero_pii_all_teachers": all_zpii_pass,
            "all_ci_passed": all_zpii_pass,
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    metrics_path = out_dir / "metrics-v1.3.0-superteacher.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
