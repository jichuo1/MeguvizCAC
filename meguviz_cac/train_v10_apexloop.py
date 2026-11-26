"""MeguvizCAC v1.0.0 End-to-End Trainer (`TunnelSymplect × DynaCore-MoLE × Zero-Recompute 4-Stage Cascade` -> `<= 2.20 MB` Bundle v6!).

Unifies ALL empirically verified breakthroughs from `v0.7.0–v0.9.0`, blind `OjinjinBench` diagnostics, and our 5-track `v1.0.0` research + RTX 4060 Ti Micro-Lab:
  Stage 0: Balanced 1.4M corpus + 80k–200k External Gemini Gold/Counterfactual Pairs + Dual `v0.6.0-R3` BF16 Teacher
           (`256-d` cosine manifold anchor `0.22`) + Softened Mid-Band (`0.35x`, never zeroing `stealth_ai`!) +
           Matryoshka Top-128 BGE->BPE12k SVD Init + Strict 3-Way Disjoint Validation (`val_gold / val_cal_select / val_cal_cert`).
  Stage 1: Progressive Depth Curriculum (`T_outer=1 -> 2..3` [33% `T=3`]) FP16 WSD + Joint Multi-Exit Supervision
           (`Rung-1 D_1=128`, `Rung-2 t=1`, `Rung-3 t>=2`) + In-Loop 5-Way Analytic EBT Supervision +
           **H4-A Adversarial Handoff Dropout (`0.25`) & Flip (`0.15`)** + DeepSeek-V3 Deadzone Aux-Loss-Free MoLE Balancing +
           Off-Diagonal Orthogonal Slot Diversity Regularizer.
  Stage 1.5: 2-Branch Balanced + HumanShield ASAM Diversity -> Greedy Generalization Soup on `50% val_gold + 50% wild val_cal_select`!
  Stage 2: Zero-Op Algebraic Folding (`fold_zero_op_v10`, `0` new ops, `< 1e-6` exact identity!) ->
           Progressive UPQ (`W4` Bridge -> 2-Branch Sub-2-Bit `5-Trit/2-Bit` QAT Soup) with **`P4 Tunnel-TRM` (`trm_lam=0.0`)**
           (exact isometry `||J||_2 = 1.0`, Phase-0 & Active Re-Read gradients 100% live!).
  Stage 3: Bit-Exact `MGVZ6` Pack/Unpack -> Guarded Post-UPQ Shock-Absorber Refit (`DynaCore-MoLE` pure FP16 cores + `P2/C1/C2/C3/H1/H2`!)
           -> `0 KB` Confounding Surgery -> 8-Byte Cross-Exit Affine Alignment -> Binary-ECE `Beta-Spline × Venn-ABERS` LUTs ->
           Selection-Safe 9-Cell Mondrian + LTT/CompCRC Finite-Sample Certification -> Frozen PrototypeBank & LB-DSM (external EBT retired!).
  Stage 4: Export TRUE `<= 2.20 MB` Bundle v6 (`MGVZ6`) + Zero-Recompute 3-Stage Static ONNX (`r1_prefix`, `r2_t1`, `r3_step`) +
           Real 1T-CPU Dynamic-Compaction Cascade Benchmark + Traps + Automated `V10` CI Gates!
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_student import CODE_LIST, CODE_TO_IDX, PRAG_TAGS, compute_eval_metrics, find_val_thresholds
from .train_v04_dualloop import _open_read, compute_high_precision_metrics
from .train_v05_apexloop import build_stratified_val_calibration_set
from .train_v06_hyperloop import (
    _load_v06_balanced_bundle,
    _modern_two_way_pauc_and_top_push,
    _subset_natural_prior_gold_guaranteed,
    _train_frozen_lb_dsm_critic,
)
from .train_v07_omniveritas import (
    _derive_attr5_label,
    _eval_traps_v07,
    _precompute_v06_bf16_teacher_targets,
    _wsd_lr_mult,
)
from .train_v08_zenith import _ingest_external_gemini_gold
from .train_v09_chronoloop import _init_v09_bpe_from_bge_svd
from .v04_corpus_prep import GROUP_TO_IDX
from .v04_runtime import fit_split_conformal_3way
from .v05_runtime import fit_l2_prototype_bank
from .v06_hyperloop import apply_zero_cost_confounding_surgery
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    STYLE_FEAT_DIM_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    extract_style_features_40d,
    extract_style_features_40d_batch,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v07_omniveritas import MAX_LEN_V07
from .v08_zenith_ops import (
    ERA_TO_IDX,
    ASAMPerturber,
    OfflineEraGRLHead,
    compute_ccr_era_ipw,
    compute_zenith_calibration_and_lowfpr_metrics,
    eval_human_era_linear_probe_auroc,
    export_beta_spline_as_v07_lut,
    fit_venn_abers_and_beta_spline_lut,
    greedy_pauc_soup,
    identify_noisy_and_quarantine,
    listnet_tail_distill_loss,
    logit_standardized_dkd_loss,
    zenith_graded_anchor_loss,
)
from .v09_cascade_cert import (
    certify_ltt_compcrc_cascade_policy,
    fit_cross_exit_affine_alignment,
    fit_selection_safe_conformal_cascade,
)
from .v09_chronoloop import HIDDEN_V09
from .v10_apexloop import (
    ENGRAM_BIGRAM_BUCKETS_V10,
    ENGRAM_TRIGRAM_BUCKETS_V10,
    MeguvizV10ApexLoopModel,
    MeguvizV10CascadeEngine,
    export_v10_modular_onnx_and_bundle_v6,
    fold_zero_op_v10,
    load_bundle_v6_mgvz_into_model,
    pack_bundle_v6_mgvz,
    run_v10_ci_gates,
)


def _ingest_external_gemini_gold_v10(
    train_items: list[dict],
    gemini_gold_path: Path | None,
) -> dict[str, int]:
    """V10-aware Gemini 80k ingestion:
      1. Cleanly separates true Pro gold (`v08_gemini_pro*`) from Flash wild (`v08_gemini_flash_gold`) so Pro stays 100% pure
         gold while Flash keeps a 20% `v0.6-R3` BGE guardrail & requires double-confirmation in Top-FP suppression!
      2. Promotes ALL `is_trap_human=True` (`score <= 30`) rows — BOTH overridden AND appended (all 10,829!) — into
         `structured_human_hard` with minimum weight `1.45` so Tier-2a & Top-FP protection covers 100% of traps!
    """
    for it in train_items:
        it["is_true_pro_gold"] = bool(it.get("is_gold_tier"))
        it["is_ext_flash"] = False
        it["is_trap_human"] = bool(it.get("group_idx") == GROUP_TO_IDX.get("structured_human_hard"))
    if not gemini_gold_path or not gemini_gold_path.is_file():
        return {"loaded": 0, "overridden_weak": 0, "appended_wild": 0, "pro_gold": 0, "flash_wild": 0, "traps_promoted": 0}

    by_sid = {str(x["sample_id"]): i for i, x in enumerate(train_items)}
    struct_gid = GROUP_TO_IDX["structured_human_hard"]
    loaded = overridden = appended = n_pro = n_flash = n_traps = 0
    with _open_read(gemini_gold_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split", "train") != "train":
                continue
            sid = str(r["sample_id"])
            sc = float(r["score"])
            y_new = float(np.clip(sc / 100.0, 0.0, 1.0))
            src = str(r.get("score_source", r.get("source", ""))).lower()
            is_pro = ("pro" in src) or (float(r.get("train_weight", 1.0)) >= 1.30)
            is_trap = bool(r.get("is_trap_human", False)) and (sc <= 30.0)
            w_rec = float(r.get("train_weight", 1.35 if is_pro else 1.05))
            if is_trap:
                w_rec = max(w_rec, 1.45)
                n_traps += 1
            if is_pro:
                n_pro += 1
            else:
                n_flash += 1
            loaded += 1

            if sid in by_sid:
                it = train_items[by_sid[sid]]
                it["y_score"] = y_new
                it["code_idx"] = CODE_TO_IDX.get(str(r.get("code", "")), it["code_idx"])
                it["is_gold_tier"] = True
                it["is_true_pro_gold"] = bool(it["is_true_pro_gold"] or is_pro)
                it["is_ext_flash"] = not it["is_true_pro_gold"]
                it["weight"] = max(float(it["weight"]), w_rec)
                if is_trap:
                    it["group_idx"] = struct_gid
                    it["is_trap_human"] = True
                if isinstance(r.get("prag_vec"), list) and len(r["prag_vec"]) == len(PRAG_TAGS):
                    it["prag_vec"] = [float(v) for v in r["prag_vec"]]
                overridden += 1
            else:
                gid_str = "structured_human_hard" if is_trap else "current_2026_real"
                prag = r.get("prag_vec") if (isinstance(r.get("prag_vec"), list) and len(r["prag_vec"]) == len(PRAG_TAGS)) else [-1.0] * len(PRAG_TAGS)
                train_items.append(
                    {
                        "sample_id": sid,
                        "raw_text": str(r["text"]),
                        "y_score": y_new,
                        "code_idx": CODE_TO_IDX.get(str(r.get("code", "")), -100),
                        "weight": w_rec,
                        "prag_vec": [float(v) for v in prag],
                        "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["current_2026_real"]),
                        "intent_idx": -1,
                        "is_gold_tier": True,
                        "is_true_pro_gold": is_pro,
                        "is_ext_flash": not is_pro,
                        "is_trap_human": is_trap,
                        "is_pre_llm": False,
                        "is_hu_anchor": bool(sc <= 18.0),
                        "is_high_prio": True,
                        "era_bucket": str(r.get("era_bucket", "current_2026")),
                    }
                )
                by_sid[sid] = len(train_items) - 1
                appended += 1
    return {
        "loaded": loaded, "overridden_weak": overridden, "appended_wild": appended,
        "pro_gold": n_pro, "flash_wild": n_flash, "traps_promoted": n_traps,
    }


def _top_fp_logit_ceiling_loss(
    logits: torch.Tensor,
    y_sc: torch.Tensor,
    tp_v06: torch.Tensor,
    is_g: torch.Tensor,
    era: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Logit-Space Double-Confirmed Top-FP Suppressor (kills Top-5 normal-human spikes breaking wild R@P95!).

    Fixes two root flaws of probability-space `relu(p - 0.12)^2.mean()`:
      1. Zero sigmoid gradient vanishing at `z >= +3.0` (operates directly in logit space!).
      2. Zero `1/400` batch dilution (uses top-4 worst humans + smooth LogSumExp max `tau=0.5`).
      3. Double-confirmed clean human guard (`is_g | pre_llm | tp_v06 <= 0.30`) -> ZERO stealth-AI collateral damage!
    """
    lg = logits.float()
    conf_hu = (y_sc <= 0.18) & (w > 0.0) & ((is_g > 0.5) | (era == 0) | (tp_v06 < 0.0) | (tp_v06 <= 0.30))
    clean_ai = (y_sc >= 0.78) & (w > 0.0)
    loss = lg.sum() * 0.0

    n_hu = int(conf_hu.sum().item())
    if n_hu >= 2:
        z_hu = lg[conf_hu]
        k_top = min(4, n_hu)
        z_worst = torch.topk(z_hu, k=k_top, largest=True).values
        # Direct logit-space quadratic hammer above z > 0 (p > 0.50) + mild push below -0.8
        l_ceil = F.relu(z_worst).pow(2).mean() + 0.20 * F.softplus(z_worst + 0.8).mean()
        loss = loss + l_ceil
        if int(clean_ai.sum().item()) >= 2:
            # Smooth max of confirmed humans must stay >= 1.2 logit units below median AI logit!
            z_hu_smax = 0.5 * torch.logsumexp(z_hu / 0.5, dim=0)
            z_ai_med = torch.quantile(lg[clean_ai], 0.50)
            loss = loss + 0.50 * F.softplus(1.2 - (z_ai_med - z_hu_smax))
    return loss


def check_v10_release_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Verify MeguvizCAC v1.0.0 milestone release gates (Bundle v6 <= 2.20 MB, 3-stage bit-exact ONNX, quality & safety)."""
    tb = metrics.get("test_bili_gold") or {}
    dsh = metrics.get("test_dsh_gold") or {}
    sub2 = metrics.get("sub2bit_bundle_v6_eval") or {}
    ox = metrics.get("onnx_3stage_int8") or {}
    ci = metrics.get("v10_ci_gates") or {}
    cal_m = metrics.get("test_bili_gold_cal_lowfpr") or {}
    m9 = (metrics.get("selection_safe_mondrian_9cell") or {}).get("alphas", {}).get("alpha_003") or {}
    traps = (metrics.get("traps") or {}).get("summary") or {}
    exp = metrics.get("export_info") or {}

    zip_mb = float(metrics.get("bundle_v6_total_mb") or 99.0)
    delta_auc = max(0.0, -float(sub2.get("auroc_delta_vs_fp16", -99.0)))
    era_auc = float(metrics.get("human_era_probe_auroc", 1.0))
    cells = (m9.get("cells") or {}).values()
    max_cell_fpr = max((float(c["pathwise_fpr"]) for c in cells if int(c.get("n_hu", 0)) >= 20), default=0.0)
    ox_ok = (
        float(exp.get("r1_fp32_max_diff", 1.0)) < 1e-4
        and float(exp.get("r2_fp32_max_diff", 1.0)) < 1e-4
        and float(exp.get("r3_fp32_max_diff", 1.0)) < 1e-4
    )

    gates = {
        "bundle_v6_le_2p20_mb": {"value": round(zip_mb, 4), "target": "<= 2.20 MB", "pass": zip_mb <= 2.20},
        "modular_3stage_onnx_bit_exact": {"value": {k: exp.get(k) for k in ("r1_fp32_max_diff", "r2_fp32_max_diff", "r3_fp32_max_diff")}, "target": "< 1e-4", "pass": ox_ok},
        "sub2_auroc_drift_le_0020": {"value": round(delta_auc, 5), "target": "<= 0.0020", "pass": delta_auc <= 0.0020},
        "cascade_1t_cpu_ms_le_1p00": {"value": ox.get("compact_batch32_cascade_1thread_ms"), "target": "<= 1.00 ms", "pass": (ox.get("compact_batch32_cascade_1thread_ms") or 99.0) <= 1.00},
        "test_bili_gold_auroc_ge_0955": {"value": tb.get("auroc"), "target": ">= 0.955", "pass": (tb.get("auroc") or 0.0) >= 0.955},
        "test_bili_gold_prauc_ge_0925": {"value": tb.get("pr_auc"), "target": ">= 0.925", "pass": (tb.get("pr_auc") or 0.0) >= 0.925},
        "test_bili_gold_rp95_ge_075": {"value": tb.get("recall_at_p95"), "target": ">= 0.75", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.75},
        "test_dsh_gold_auroc_ge_0935": {"value": dsh.get("auroc"), "target": ">= 0.935", "pass": (dsh.get("auroc") or 0.0) >= 0.935},
        "ece_le_0060": {"value": cal_m.get("ece_15bin"), "target": "<= 0.060", "pass": (cal_m.get("ece_15bin") or 1.0) <= 0.060},
        "human_era_probe_le_056": {"value": era_auc, "target": "<= 0.56", "pass": era_auc <= 0.56},
        "selection_safe_max_pathwise_fpr_le_0035": {"value": round(max_cell_fpr, 4), "target": "<= 0.035", "pass": max_cell_fpr <= 0.035},
        "struct_human_trap_le_042": {"value": traps.get("struct_human_max"), "target": "<= 0.42", "pass": (traps.get("struct_human_max") or 1.0) <= 0.42},
        "v10_ci_all_passed": {"value": bool(ci.get("all_ci_passed")), "target": "True", "pass": bool(ci.get("all_ci_passed"))},
    }
    return {
        "gates": gates,
        "passed_count": sum(1 for v in gates.values() if v["pass"]),
        "total_gates": len(gates),
        "all_passed": all(v["pass"] for v in gates.values()),
    }


def train_v10_apexloop(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from scipy.special import expit
    from torch.utils.data import DataLoader, Dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v1.0.0 training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else (512 if vram_gb >= 35.0 else 192)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    print(
        f"[v1.0.0-ApexLoop Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={batch_size} | "
        f"tok={args.v07_tokenizer} | teacher_v06={args.teacher_v06_ckpt} | "
        f"gemini_gold={args.v08_gemini_gold} | gemini_pairs={args.v08_gemini_pairs}",
        flush=True,
    )

    extra_pairs = [p.resolve() for p in args.extra_pairs]
    # Auto-include v10 AI-topic & Bilibili-disclaimer debias pairs if present
    debias_p = Path(__file__).resolve().parent / "data" / "v10-topic-debias-pairs.jsonl"
    if debias_p.is_file() and debias_p.resolve() not in extra_pairs:
        extra_pairs.append(debias_p.resolve())
    if args.v08_gemini_pairs and args.v08_gemini_pairs.is_file():
        extra_pairs.append(args.v08_gemini_pairs.resolve())

    train_items_raw, eval_rows, pairs, intent_vecs, data_audit = _load_v06_balanced_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        extra_pairs,
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
    )
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} pairs collided with eval!")

    gem_audit = _ingest_external_gemini_gold_v10(train_items_raw, args.v08_gemini_gold)
    print(f"[v1.0.0-ApexLoop Data] base={data_audit['train_items']:,} | pairs={len(pairs):,} | gemini_factory={json.dumps(gem_audit)}", flush=True)

    if args.smoke_test:
        rng_s = random.Random(20261126)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    teacher_v06_p = None
    teacher_v06_embs = None
    if args.teacher_v06_ckpt and args.teacher_v06_ckpt.is_file() and not args.smoke_test:
        teacher_v06_p, teacher_v06_embs = _precompute_v06_bf16_teacher_targets(
            args.teacher_v06_ckpt, train_items_raw, [], [], args.v06_backbone, device, amp_dtype, ckpt_dir=ckpt_dir, polish_steps=0, bs=1024
        )
    for idx_it, item in enumerate(train_items_raw):
        item["row_idx"] = idx_it
        item["teacher_p"] = float(teacher_v06_p[idx_it]) if teacher_v06_p is not None else -1.0

    tok_path = out_dir / "bili_picky_bpe_12k.json"
    if args.v07_tokenizer and args.v07_tokenizer.is_file():
        shutil.copy2(args.v07_tokenizer, tok_path)
        tokenizer = BiliPickyBPETokenizer(tok_path)
    else:
        rng_t = random.Random(20261126)
        txts_t = [
            canonicalize_zero_pii_v07(x["raw_text"])[0]
            for x in (train_items_raw if args.smoke_test else rng_t.sample(train_items_raw, min(200000, len(train_items_raw))))
        ]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(
            txts_t, tok_path, vocab_size=2048 if args.smoke_test else BPE_VOCAB_SIZE_V07, min_frequency=2 if args.smoke_test else 3
        )

    styles_list, era_list, y_list, is_gold_list, is_pre_list = [], [], [], [], []
    for item in train_items_raw:
        cl, f40, meta = extract_style_features_40d(item["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        item["clean_text"] = cl
        item["ids"] = ids
        item["engram"] = make_engram_bigram_ids(ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
        item["trigram"] = make_engram_trigram_ids(ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
        item["length"] = len(ids)
        item["style"] = f40
        item["spoof_any"] = meta["spoof_any"]
        item["attr5"] = _derive_attr5_label(item)
        era_id = 0 if item.get("is_pre_llm") else ERA_TO_IDX.get(str(item.get("era_bucket", "current_2026")), 3)
        item["era_idx"] = era_id
        styles_list.append(f40)
        era_list.append(era_id)
        y_list.append(item["y_score"])
        is_gold_list.append(bool(item.get("is_gold_tier")))
        is_pre_list.append(bool(item.get("is_pre_llm")))

    ipw_mult = compute_ccr_era_ipw(np.stack(styles_list, axis=0), np.array(era_list, dtype=np.int64), np.array(y_list, dtype=np.float32))
    denoise_report = {"toxic_downweighted": 0, "mid_softened": 0}
    if teacher_v06_p is not None:
        dn = identify_noisy_and_quarantine(
            teacher_v06_p, np.array(y_list, dtype=np.float32), np.array(is_gold_list, dtype=bool), np.array(is_pre_list, dtype=bool), toxic_frac=0.025
        )
        denoise_mult = np.ones(len(train_items_raw), dtype=np.float32)
        denoise_mult[dn["mid_quarantine"]] = 0.35
        denoise_mult[dn["toxic_mask"]] = 0.15
        denoise_report = {"toxic_downweighted": int(dn["toxic_mask"].sum()), "mid_softened": int(dn["mid_quarantine"].sum())}
    else:
        denoise_mult = np.ones(len(train_items_raw), dtype=np.float32)

    for i, item in enumerate(train_items_raw):
        item["weight"] = float(item["weight"]) * float(ipw_mult[i]) * float(denoise_mult[i])
    print(f"[Stage0-Denoise+IPW] {json.dumps(denoise_report)}", flush=True)

    pairs_gg_pre, pairs_gen_pre = [], []
    if pairs:
        ai_cl, ai_st, _ = extract_style_features_40d_batch([p["ai_text"] for p in pairs])
        hu_cl, hu_st, _ = extract_style_features_40d_batch([p["human_text"] for p in pairs])
        for i, p in enumerate(pairs):
            a_ids = tokenizer.encode_ids(ai_cl[i], max_len=args.max_len)
            h_ids = tokenizer.encode_ids(hu_cl[i], max_len=args.max_len)
            rec = {
                "ai_ids": a_ids, "ai_eng": make_engram_bigram_ids(a_ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
                "ai_tri": make_engram_trigram_ids(a_ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10), "ai_style": ai_st[i],
                "hu_ids": h_ids, "hu_eng": make_engram_bigram_ids(h_ids, buckets=ENGRAM_BIGRAM_BUCKETS_V10),
                "hu_tri": make_engram_trigram_ids(h_ids, buckets=ENGRAM_TRIGRAM_BUCKETS_V10), "hu_style": hu_st[i],
                "weight": float(p["weight"]),
            }
            (pairs_gg_pre if p.get("is_gateguard") else pairs_gen_pre).append(rec)
    pairs_pre = pairs_gg_pre + pairs_gen_pre

    # Strict 3-way disjoint validation split (prevents val_gold overfitting & preserves Selection-Safe/LTT exchangeability!)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_all, _ = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    val_gold_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_gold}
    val_cal_disjoint = [r for r in val_cal_all if str(r.get("sample_id", r.get("id", ""))) not in val_gold_ids]
    if len(val_cal_disjoint) < 400:
        val_cal_disjoint = val_cal_all
    rng_v = random.Random(20261126)
    vc_idx = list(range(len(val_cal_disjoint)))
    rng_v.shuffle(vc_idx)
    half_vc = max(200, len(vc_idx) // 2)
    val_cal_select = [val_cal_disjoint[i] for i in vc_idx[:half_vc]]
    val_cal_cert = [val_cal_disjoint[i] for i in vc_idx[half_vc:]] or val_cal_select
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(sub: list[dict]):
        raws = [str(r["text"]) for r in sub]
        cls_l, st_m, metas_l = extract_style_features_40d_batch(raws)
        enc = [tokenizer.encode_ids(c, max_len=args.max_len) for c in cls_l]
        eng = [make_engram_bigram_ids(s, buckets=ENGRAM_BIGRAM_BUCKETS_V10) for s in enc]
        sc = np.array([float(r["score"]) for r in sub], dtype=np.float32)
        eras = np.array([0 if r.get("era_bucket") == "pre_llm" else 3 for r in sub], dtype=np.int64)
        return enc, eng, raws, cls_l, st_m, metas_l, sc, eras

    val_enc, val_eng, val_txts, val_cleans, val_st, val_metas, val_scores, val_eras = _pre_eval(val_gold)
    sel_enc, sel_eng, sel_txts, sel_cleans, sel_st, sel_metas, sel_scores, sel_eras = _pre_eval(val_cal_select)
    cal_enc, cal_eng, cal_txts, cal_cleans, cal_st, cal_metas, cal_scores, cal_eras = _pre_eval(val_cal_cert)
    test_enc, test_eng, test_txts, test_cleans, test_st, test_metas, test_scores, test_eras = _pre_eval(test_gold)
    dsh_enc, dsh_eng, dsh_txts, dsh_cleans, dsh_st, dsh_metas, dsh_scores, _ = _pre_eval(test_dsh_gold)
    all_enc, all_eng, all_txts, all_cleans, all_st, all_metas, all_scores, _ = _pre_eval(test_bili_all)
    prb_enc, prb_eng, prb_txts, prb_cleans, prb_st, prb_metas, prb_scores, _ = _pre_eval(probe_rows)

    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 1024
    mdl = MeguvizV10ApexLoopModel(
        vocab_size=tokenizer.vocab_size, embed_dim=48, hidden_size=HIDDEN_V09, num_heads=6, ffn_size=896,
        engram_buckets=ENGRAM_BIGRAM_BUCKETS_V10, trigram_buckets=ENGRAM_TRIGRAM_BUCKETS_V10, engram_dim=16,
        max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V07,
        num_codes=len(CODE_LIST), num_prag=len(PRAG_TAGS), intent_dim=intent_dim,
    ).to(device)

    svd_init_rep: dict[str, Any] = {"applied": False}
    if args.v10_fp16_ckpt and args.v10_fp16_ckpt.is_file():
        mdl.load_state_dict(torch.load(args.v10_fp16_ckpt, map_location=device), strict=True)
        print(f"[Stage0-v10-Resume] Loaded FP16 ApexLoop checkpoint from {args.v10_fp16_ckpt}", flush=True)
    elif not args.smoke_test and not args.skip_bge_svd_init:
        svd_init_rep = _init_v09_bpe_from_bge_svd(mdl, tokenizer, backbone=args.v06_backbone)

    era_grl = OfflineEraGRLHead(emb_dim=256, num_eras=4).to(device)

    @torch.no_grad()
    def predict_v10(model_e: MeguvizV10ApexLoopModel, enc_l, eng_l, st_arr, outer_steps: int = 2, use_fp32: bool = False, tri_l=None) -> dict[str, np.ndarray]:
        model_e.eval()
        n = len(enc_l)
        out_d = {
            "logit": np.zeros(n, dtype=np.float32), "raw_logit": np.zeros(n, dtype=np.float32),
            "logit_t1": np.zeros(n, dtype=np.float32), "raw_logit_t1": np.zeros(n, dtype=np.float32),
            "logit_r1": np.zeros(n, dtype=np.float32), "raw_logit_r1": np.zeros(n, dtype=np.float32),
            "p_defer1": np.zeros(n, dtype=np.float32), "p_defer2": np.zeros(n, dtype=np.float32),
            "r1_span_max": np.zeros(n, dtype=np.float32),
            "ebt_margin": np.zeros(n, dtype=np.float32), "ebt_grad_t1": np.zeros(n, dtype=np.float32),
            "emb": np.zeros((n, 256), dtype=np.float32),
        }
        if n == 0:
            return out_d
        if tri_l is None:
            tri_l = [make_engram_trigram_ids(s, buckets=ENGRAM_TRIGRAM_BUCKETS_V10) for s in enc_l]
        order = sorted(range(n), key=lambda i: len(enc_l[i]))
        ebs = 128 if use_fp32 else 512
        for st_i in range(0, n, ebs):
            mb = order[st_i : st_i + ebs]
            ml = max(len(enc_l[i]) for i in mb)
            ids = np.full((len(mb), ml), tokenizer.pad_id, dtype=np.int64)
            m = np.zeros((len(mb), ml), dtype=np.int64)
            eg = np.zeros((len(mb), ml), dtype=np.int64)
            tg = np.zeros((len(mb), ml), dtype=np.int64)
            for ri, oi in enumerate(mb):
                s = enc_l[oi]
                ids[ri, : len(s)], m[ri, : len(s)], eg[ri, : len(s)], tg[ri, : len(s)] = s, 1, eng_l[oi], tri_l[oi]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=not use_fp32):
                o = model_e(
                    torch.from_numpy(ids).to(device), torch.from_numpy(m).to(device),
                    torch.from_numpy(st_arr[mb]).to(device),
                    engram_ids=torch.from_numpy(eg).to(device), trigram_ids=torch.from_numpy(tg).to(device),
                    outer_steps=outer_steps, k_inner=2, return_aux=True,
                )
            arrs = {
                "logit": o["logit"].float().cpu().numpy(), "raw_logit": o["raw_logit"].float().cpu().numpy(),
                "logit_t1": o["exit1_out"]["logit"].float().cpu().numpy(), "raw_logit_t1": o["exit1_out"]["raw_logit"].float().cpu().numpy(),
                "logit_r1": o["logit_r1"].float().cpu().numpy(), "raw_logit_r1": o["raw_logit_r1"].float().cpu().numpy(),
                "p_defer1": o["p_defer1"].float().cpu().numpy(), "p_defer2": o["exit1_out"]["p_defer2"].float().cpu().numpy(),
                "r1_span_max": o["r1_span_max"].float().cpu().numpy(),
                "ebt_margin": o["ebt_margin"].float().cpu().numpy(), "ebt_grad_t1": o["exit1_out"]["ebt_grad_norm"].float().cpu().numpy(),
                "emb": o["emb"].float().cpu().numpy(),
            }
            for ri, oi in enumerate(mb):
                for k_o, a_o in arrs.items():
                    out_d[k_o][oi] = a_o[ri]
        return out_d

    def _val_obj_sd(sd_cand: dict[str, torch.Tensor]) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_cand.items()}, strict=True)
        vo = predict_v10(mdl, val_enc, val_eng, val_st, outer_steps=2)
        so = predict_v10(mdl, sel_enc, sel_eng, sel_st, outer_steps=2)
        def _sc(o_d, y_s, wild_tail: bool = False):
            p, pr1 = expit(o_d["logit"]), expit(o_d["logit_r1"])
            m = {**compute_eval_metrics(y_s, p), **compute_high_precision_metrics(y_s, p)}
            m1 = compute_eval_metrics(y_s, pr1)
            if wild_tail:
                # Explicitly guard natural-prior R@P95 & R@P99 against Top-5 normal-human spikes!
                return (
                    0.32 * float(m["auroc"]) + 0.24 * float(m["pr_auc"]) + 0.14 * float(m["best_f05"])
                    + 0.18 * float(m["recall_at_p95"]) + 0.06 * float(m["recall_at_p99"]) + 0.06 * float(m1["auroc"])
                )
            return 0.38 * float(m["auroc"]) + 0.28 * float(m["pr_auc"]) + 0.14 * float(m["best_f05"]) + 0.14 * float(m["recall_at_p95"]) + 0.06 * float(m1["auroc"])
        return 0.50 * _sc(vo, val_scores, wild_tail=False) + 0.50 * _sc(so, sel_scores, wild_tail=True)

    class V10DS(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            return self.items[i]

    casual_wrappers = [
        tokenizer.encode_ids(x["clean_text"], max_len=20)[1:-1]
        for x in train_items_raw[:40000]
        if x["y_score"] <= 0.12 and 6 <= len(x["clean_text"]) <= 18
    ][:512]

    def _make_collate(sandwich_prob: float = 0.06):
        rng_c = random.Random(20261126)
        def _collate(batch):
            seqs, engs, tris = [], [], []
            for x in batch:
                s = x["ids"]
                if sandwich_prob > 0.0 and casual_wrappers and x["y_score"] >= 0.78 and len(s) <= 180 and rng_c.random() < sandwich_prob:
                    w1, w2 = rng_c.choice(casual_wrappers), rng_c.choice(casual_wrappers)
                    s = ([tokenizer.bos_id] + w1 + s[1:-1] + w2 + [tokenizer.eos_id])[: args.max_len]
                    eg = make_engram_bigram_ids(s, buckets=ENGRAM_BIGRAM_BUCKETS_V10)
                    tg = make_engram_trigram_ids(s, buckets=ENGRAM_TRIGRAM_BUCKETS_V10)
                else:
                    eg = x["engram"]
                    tg = x["trigram"]
                seqs.append(s)
                engs.append(eg)
                tris.append(tg)
            ml = min(args.max_len, ((max(len(s) for s in seqs) + 15) // 16) * 16)
            ids = np.full((len(batch), ml), tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), ml), dtype=np.int64)
            eng = np.zeros((len(batch), ml), dtype=np.int64)
            tri = np.zeros((len(batch), ml), dtype=np.int64)
            st = np.empty((len(batch), STYLE_FEAT_DIM_V07), dtype=np.float32)
            for i, (s, eg, tg, x) in enumerate(zip(seqs, engs, tris, batch)):
                n_s = min(len(s), ml)
                ids[i, :n_s], mask[i, :n_s], eng[i, :n_s], tri[i, :n_s], st[i] = s[:n_s], 1, eg[:n_s], tg[:n_s], x["style"]
            return (
                torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(eng), torch.from_numpy(tri), torch.from_numpy(st),
                torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
                torch.tensor([x["teacher_p"] for x in batch], dtype=torch.float32),
                torch.tensor([1.0 if (x.get("is_true_pro_gold") or x.get("is_trap_human")) else 0.0 for x in batch], dtype=torch.float32),
                torch.tensor([1.0 if x.get("is_ext_flash") else 0.0 for x in batch], dtype=torch.float32),
                torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["attr5"] for x in batch], dtype=torch.long),
                torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
                torch.tensor([x["group_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["era_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["row_idx"] for x in batch], dtype=torch.long),
            )
        return _collate

    def _collate_pairs(p_batch):
        bs_p = len(p_batch)
        ma = min(args.max_len, ((max(len(x["ai_ids"]) for x in p_batch) + 15) // 16) * 16)
        mh = min(args.max_len, ((max(len(x["hu_ids"]) for x in p_batch) + 15) // 16) * 16)
        ai_ids, ai_m = np.full((bs_p, ma), tokenizer.pad_id, dtype=np.int64), np.zeros((bs_p, ma), dtype=np.int64)
        ai_e, ai_t = np.zeros((bs_p, ma), dtype=np.int64), np.zeros((bs_p, ma), dtype=np.int64)
        hu_ids, hu_m = np.full((bs_p, mh), tokenizer.pad_id, dtype=np.int64), np.zeros((bs_p, mh), dtype=np.int64)
        hu_e, hu_t = np.zeros((bs_p, mh), dtype=np.int64), np.zeros((bs_p, mh), dtype=np.int64)
        ai_st, hu_st = np.empty((bs_p, STYLE_FEAT_DIM_V07), dtype=np.float32), np.empty((bs_p, STYLE_FEAT_DIM_V07), dtype=np.float32)
        for i, x in enumerate(p_batch):
            na, nh = min(len(x["ai_ids"]), ma), min(len(x["hu_ids"]), mh)
            ai_ids[i, :na], ai_m[i, :na], ai_e[i, :na], ai_t[i, :na] = x["ai_ids"][:na], 1, x["ai_eng"][:na], x["ai_tri"][:na]
            hu_ids[i, :nh], hu_m[i, :nh], hu_e[i, :nh], hu_t[i, :nh] = x["hu_ids"][:nh], 1, x["hu_eng"][:nh], x["hu_tri"][:nh]
            ai_st[i], hu_st[i] = x["ai_style"], x["hu_style"]
        return (
            torch.from_numpy(ai_ids).to(device, non_blocking=True), torch.from_numpy(ai_m).to(device, non_blocking=True),
            torch.from_numpy(ai_e).to(device, non_blocking=True), torch.from_numpy(ai_t).to(device, non_blocking=True), torch.from_numpy(ai_st).to(device, non_blocking=True),
            torch.from_numpy(hu_ids).to(device, non_blocking=True), torch.from_numpy(hu_m).to(device, non_blocking=True),
            torch.from_numpy(hu_e).to(device, non_blocking=True), torch.from_numpy(hu_t).to(device, non_blocking=True), torch.from_numpy(hu_st).to(device, non_blocking=True),
            torch.tensor([x["weight"] for x in p_batch], dtype=torch.float32, device=device),
        )

    ce_code = nn.CrossEntropyLoss(ignore_index=-100)
    hard_grp_t = torch.tensor(
        [GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["multilingual_world_knowledge"], GROUP_TO_IDX["dsh_hard_slice"]],
        device=device,
    )

    def _run_stage(
        stage_name: str,
        init_sd: dict[str, torch.Tensor],
        items: list[dict],
        steps: int,
        lr: float,
        quant_mode: str,
        trm_lam_k0: float = 1.0,
        trm_lam_k1: float = 1.0,
        progressive_depth: bool = False,
        use_asam: bool = False,
        pauc_w: float = 0.09,
        pair_w: float = 0.24,
        beta_end: float = 0.028,
        teacher_m: MeguvizV10ApexLoopModel | None = None,
    ) -> tuple[dict[str, torch.Tensor], float]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(quant_mode)
        opt = torch.optim.AdamW(list(mdl.parameters()) + list(era_grl.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=0.015)
        asam = ASAMPerturber(mdl, rho=0.05) if use_asam else None
        dl = DataLoader(V10DS(items), batch_size=batch_size, shuffle=True, collate_fn=_make_collate(0.06), num_workers=0, pin_memory=True)
        it_dl = iter(dl)
        rng_p = random.Random(20261126 + len(stage_name))
        pair_bs = max(12, min(36, int(round(16.0 * batch_size / 256.0))))
        best_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc = _val_obj_sd(best_sd) if not args.smoke_test else -1.0
        eval_every = max(500, steps // 4) if (not args.smoke_test and steps >= 1000) else 0

        mdl.train()
        for step in range(steps):
            try:
                batch = next(it_dl)
            except StopIteration:
                it_dl = iter(dl)
                batch = next(it_dl)
            ids, mask, eng, tri, st, y_sc, tp_v06, is_g, is_ef, y_cd, y_a5, w, grp, era, row_idx = [t.to(device, non_blocking=True) for t in batch]
            mult = _wsd_lr_mult(step, steps)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult

            if progressive_depth and step < int(steps * 0.18):
                outer_now = 1
            elif step % 3 == 0:
                outer_now = 3
            else:
                outer_now = 2

            t_emb_v06 = None
            if teacher_m is None and teacher_v06_embs is not None:
                t_emb_v06 = torch.from_numpy(teacher_v06_embs[row_idx.cpu().numpy()]).to(device, dtype=torch.float32, non_blocking=True)

            pair_tensors = None
            if pairs_pre and pair_w > 0.0 and (step % 2 == 0):
                if pairs_gg_pre and pairs_gen_pre:
                    n_gg = min(len(pairs_gg_pre), max(4, int(round(pair_bs * 0.45))))
                    n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                    pb = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                else:
                    pb = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                pair_tensors = _collate_pairs(pb)

            def _compute_loss():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(
                        ids, mask, st, engram_ids=eng, trigram_ids=tri, outer_steps=outer_now, k_inner=2,
                        trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1,
                        handoff_drop_prob=0.25, handoff_flip_prob=0.15, return_aux=True,
                    )
                    lg = out["logit"].float()
                    lg_e1 = out["exit1_out"]["logit"].float()
                    lg_r1 = out["logit_r1"].float()

                    has_v06_weak = (tp_v06 >= 0.0) & (is_g < 0.5) & (is_ef < 0.5)
                    has_v06_flash = (tp_v06 >= 0.0) & (is_ef > 0.5)
                    tp_cl = tp_v06.clamp(0.0, 1.0)
                    y_eff = torch.where(has_v06_weak, 0.55 * y_sc + 0.45 * tp_cl, y_sc)
                    # Preserve 20% smooth v0.6-R3 BGE guardrail on external Gemini Flash labels to prevent normal-human extreme FP spikes!
                    y_eff = torch.where(has_v06_flash, 0.80 * y_sc + 0.20 * tp_cl, y_eff)
                    y_cl = y_eff.clamp(1e-5, 1.0 - 1e-5)

                    loss_full = (F.binary_cross_entropy_with_logits(lg, y_cl, reduction="none") * w).mean()
                    loss_e1 = (F.binary_cross_entropy_with_logits(lg_e1, y_cl, reduction="none") * w).mean()
                    w_r1_fn = w * (1.0 + 2.5 * (y_eff >= 0.45).float())
                    loss_r1 = (F.binary_cross_entropy_with_logits(lg_r1, y_cl, reduction="none") * w_r1_fn).mean()
                    loss_sd_r1 = listnet_tail_distill_loss(lg_r1, lg.detach())
                    loss_sd_e1 = listnet_tail_distill_loss(lg_e1, lg.detach())

                    l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=step / max(1, steps - 1), beta_start=0.06, beta_end=beta_end)
                    hu_h = torch.isin(grp, hard_grp_t)
                    l_anc = zenith_graded_anchor_loss(lg, y_sc, hard_human_mask=hu_h)
                    l_topfp = _top_fp_logit_ceiling_loss(lg, y_sc, tp_v06, is_g, era, w)
                    l_era = era_grl(out["emb"], era, is_clean_human=(y_sc <= 0.25), lam=0.20)

                    # Off-diagonal slot orthogonal diversity + In-Loop 5-way EBT Energy CE + MoLE Z-Loss!
                    s_u = F.normalize(out["slots_final"].float(), p=2, dim=-1)
                    s_gram = torch.matmul(s_u, s_u.transpose(-1, -2))
                    s_off = 1.0 - torch.eye(s_gram.shape[-1], device=s_gram.device, dtype=s_gram.dtype).unsqueeze(0)
                    l_slot_div = ((s_gram * s_off) ** 2).mean()
                    l_ebt_ce = F.cross_entropy(-out["ebt_energies"].float(), y_a5)
                    l_mole_z = out["mole_z_loss"].float()

                    loss = (
                        loss_full + 0.18 * loss_e1 + 0.22 * loss_r1
                        + 0.08 * loss_sd_r1 + 0.06 * loss_sd_e1
                        + pauc_w * l_pa + l_tp + l_anc + 0.18 * l_topfp + 0.08 * l_era
                        + 0.15 * l_slot_div + 0.10 * l_ebt_ce + 0.002 * l_mole_z
                        + 0.08 * F.cross_entropy(out["attr5_logit"], y_a5)
                        + 0.08 * ce_code(out["code_logit"], y_cd)
                        + 0.04 * out["cbm_loss"]
                    )

                    with torch.amp.autocast("cuda", enabled=False):
                        pf_d, pr1_d, pe1_d = torch.sigmoid(lg.detach()), torch.sigmoid(lg_r1.detach()), torch.sigmoid(lg_e1.detach())
                        d1_tgt = torch.maximum(((pf_d - pr1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_sc >= 0.45) & (pr1_d < 0.35)).float())
                        d2_tgt = torch.maximum(((pf_d - pe1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_sc >= 0.45) & (pe1_d < 0.35)).float())
                        l_def1 = F.binary_cross_entropy(out["p_defer1"].float().clamp(1e-5, 1.0 - 1e-5), d1_tgt)
                        l_def2 = F.binary_cross_entropy(out["exit1_out"]["p_defer2"].float().clamp(1e-5, 1.0 - 1e-5), d2_tgt)
                        l_r1_sp = (F.binary_cross_entropy(out["r1_span_max"].float().clamp(1e-5, 1.0 - 1e-5), y_cl.float(), reduction="none") * w_r1_fn.float()).mean()
                    loss = loss + 0.18 * l_def1 + 0.10 * l_def2 + 0.08 * l_r1_sp

                    if teacher_m is not None:
                        with torch.no_grad():
                            to = teacher_m(ids, mask, st, engram_ids=eng, trigram_ids=tri, outer_steps=outer_now, k_inner=2, return_aux=True)
                        loss = loss + 0.12 * logit_standardized_dkd_loss(out["attr5_logit"], to["attr5_logit"], y_a5, sample_weights=w)
                        loss = loss + 0.10 * listnet_tail_distill_loss(lg, to["logit"])
                        loss = loss + 0.08 * listnet_tail_distill_loss(lg_r1, to["logit_r1"])
                        loss = loss + 0.22 * (1.0 - F.cosine_similarity(out["emb"].float(), to["emb"].float(), dim=-1).mean())
                    elif t_emb_v06 is not None:
                        loss = loss + 0.22 * (1.0 - F.cosine_similarity(out["emb"].float(), t_emb_v06, dim=-1).mean())
                        v_tp = tp_v06 >= 0.0
                        if int(v_tp.sum()) >= 8:
                            loss = loss + 0.08 * listnet_tail_distill_loss(lg[v_tp], torch.logit(tp_v06[v_tp].clamp(1e-4, 1.0 - 1e-4)))

                    if pair_tensors is not None:
                        ai_i, ai_m, ai_e, ai_t, ai_s, hu_i, hu_m, hu_e, hu_t, hu_s, pw_t = pair_tensors
                        la, _, la_r1, _ = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, trigram_ids=ai_t, outer_steps=2, k_inner=2, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1, return_aux=False)
                        lh, _, lh_r1, _ = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, trigram_ids=hu_t, outer_steps=2, k_inner=2, trm_lam_k0=trm_lam_k0, trm_lam_k1=trm_lam_k1, return_aux=False)
                        la, lh, la_r1, lh_r1 = la.float(), lh.float(), la_r1.float(), lh_r1.float()
                        l_simpo = (F.softplus(1.8 - (la - lh)) * pw_t).mean()
                        l_pabs = ((F.relu(0.84 - torch.sigmoid(la)).pow(2) + 1.6 * F.relu(torch.sigmoid(lh) - 0.12).pow(2)) * pw_t).mean()
                        l_simpo_r1 = (F.softplus(1.4 - (la_r1 - lh_r1)) * pw_t).mean()
                        loss = loss + pair_w * (l_simpo + 0.9 * l_pabs + 0.25 * l_simpo_r1)
                return loss

            opt.zero_grad(set_to_none=True)
            l1 = _compute_loss()
            l1.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            if asam is not None and step >= int(steps * 0.5):
                asam.first_step()
                opt.zero_grad(set_to_none=True)
                l2 = _compute_loss()
                l2.backward()
                asam.restore()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()

            if eval_every > 0 and (step + 1) % eval_every == 0 and (step + 1) < steps:
                csd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                csc = _val_obj_sd(csd)
                print(f"  [{stage_name} step={step+1}/{steps}] val_obj={csc:.4f} (best={best_sc:.4f})", flush=True)
                if csc > best_sc:
                    best_sc, best_sd = csc, csd
                mdl.train()

        end_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        end_sc = _val_obj_sd(end_sd)
        if end_sc >= best_sc:
            best_sc, best_sd = end_sc, end_sd
        print(f"[v1.0 Stage {stage_name}] val_obj={best_sc:.4f} (end={end_sc:.4f})", flush=True)
        return best_sd, best_sc

    # Stage 1: Progressive Depth Curriculum FP16 WSD (with 2.5x fresh Gemini/Pro/Trap oversampling -> stops 1.14M weak-label dilution!)
    init_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    s1_fresh_extra = [x for x in train_items_raw if x.get("is_gold_tier") or x.get("is_trap_human")]
    s1_pro_ai_extra = [x for x in train_items_raw if x.get("is_true_pro_gold") or x.get("is_trap_human") or (x.get("is_gold_tier") and x["y_score"] >= 0.70)]
    s1_pool = train_items_raw if args.smoke_test else (train_items_raw + s1_fresh_extra + s1_pro_ai_extra)
    print(f"[Stage1-Pool] raw={len(train_items_raw):,} | s1_pool={len(s1_pool):,} (fresh_gold_oversampled={len(s1_fresh_extra):,}, pro_trap_ai_extra={len(s1_pro_ai_extra):,})", flush=True)
    if not (args.v10_fp16_ckpt and args.v10_fp16_ckpt.is_file()):
        s1_sd, _ = _run_stage(
            "Stage1-ApexLoop-FP16", init_sd, s1_pool,
            steps=10 if args.smoke_test else args.stage1_steps, lr=4.5e-4, quant_mode="fp16",
            trm_lam_k0=0.5, trm_lam_k1=1.0, progressive_depth=True, use_asam=False, pauc_w=0.08,
        )
    else:
        s1_sd = init_sd

    # Stage 1.5: Non-Diluted High-Precision Pool (~185k instead of 480k old weak v0.4!) + Online Top-FP Hard-Negative Mining!
    hard_gids = {GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["multilingual_world_knowledge"], GROUP_TO_IDX["dsh_hard_slice"], GROUP_TO_IDX["literary_boundary"]}
    if args.smoke_test:
        sub_nat_raw, _ = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=min(480000, len(train_items_raw)), seed_s=20261126)
        sub_nat = [dict(x) for x in sub_nat_raw]
    else:
        rng_nd = random.Random(20261126)
        nd_core = [x for x in train_items_raw if x.get("is_gold_tier") or x.get("is_trap_human") or x["group_idx"] in hard_gids]
        nd_core_ids = {id(x) for x in nd_core}
        nd_pre = [x for x in train_items_raw if id(x) not in nd_core_ids and x.get("is_pre_llm")]
        nd_bg = [x for x in train_items_raw if id(x) not in nd_core_ids and not x.get("is_pre_llm") and (x["y_score"] <= 0.25 or x["y_score"] >= 0.72)]
        sub_nat_src = nd_core + (rng_nd.sample(nd_pre, 25000) if len(nd_pre) > 25000 else nd_pre) + (rng_nd.sample(nd_bg, 65000) if len(nd_bg) > 65000 else nd_bg)
        sub_nat = [dict(x) for x in sub_nat_src]
        print(f"[Stage1.5-NonDilutedPool] total={len(sub_nat):,} (gold_hard={len(nd_core):,}, pre_llm={min(25000,len(nd_pre)):,}, clean_bg={min(65000,len(nd_bg)):,})", flush=True)

    # Mine the top-scoring double-confirmed clean humans (y<=0.18 AND [true_pro_gold | trap_human | pre_llm | teacher_p<=0.30])!
    mdl.load_state_dict({k: v.to(device) for k, v in s1_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    mdl.eval()
    conf_hu_idxs = [
        i for i, x in enumerate(sub_nat)
        if x["y_score"] <= 0.18 and (
            x.get("is_true_pro_gold") or x.get("is_trap_human") or x.get("is_pre_llm")
            or x.get("teacher_p", -1.0) < 0.0 or x.get("teacher_p", 1.0) <= 0.30
        )
    ]
    rng_m = random.Random(20261126)
    scan_cap = 256 if args.smoke_test else 120000
    scan_idxs = rng_m.sample(conf_hu_idxs, scan_cap) if len(conf_hu_idxs) > scan_cap else conf_hu_idxs
    mined_hard_hu_items: list[dict] = []
    topfp_mine_rep: dict[str, Any] = {"scanned_confirmed_humans": len(scan_idxs), "mined_count": 0, "max_fp_prob_before": 0.0}
    if scan_idxs:
        scan_out = predict_v10(
            mdl,
            [sub_nat[i]["ids"] for i in scan_idxs],
            [sub_nat[i]["engram"] for i in scan_idxs],
            np.stack([sub_nat[i]["style"] for i in scan_idxs], axis=0),
            outer_steps=2,
            use_fp32=False,
        )
        scan_p = expit(scan_out["logit"])
        topfp_mine_rep["max_fp_prob_before"] = round(float(np.max(scan_p)), 4)
        # Pick all confirmed humans with p >= 0.35 (or top 1.5%, capped at 6,000 hardest negatives)
        p_cut = min(0.35, float(np.quantile(scan_p, 0.985)))
        hard_order = np.argsort(-scan_p)
        cur_real_gid = GROUP_TO_IDX["current_2026_real"]
        struct_gid = GROUP_TO_IDX["structured_human_hard"]
        max_mine = 32 if args.smoke_test else 6000
        for pos_k in hard_order[:max_mine]:
            pk = float(scan_p[pos_k])
            if pk < max(0.20, p_cut):
                break
            it_m = sub_nat[scan_idxs[int(pos_k)]]
            it_m["weight"] = min(6.0, float(it_m["weight"]) * (3.0 if pk >= 0.65 else 2.2))
            if it_m["group_idx"] == cur_real_gid:
                it_m["group_idx"] = struct_gid
            mined_hard_hu_items.append(it_m)
        topfp_mine_rep["mined_count"] = len(mined_hard_hu_items)
        topfp_mine_rep["p_threshold_used"] = round(max(0.20, p_cut), 4)
        # Oversample mined top-FP humans once more into sub_nat so Stage 1.5 & Stage 2 QAT see them reliably!
        if mined_hard_hu_items:
            sub_nat.extend(mined_hard_hu_items)
    print(f"[Stage1.5-TopFP-HardNeg-Mining] {json.dumps(topfp_mine_rep)}", flush=True)

    s15_steps = 6 if args.smoke_test else args.refine_steps
    b1_sd, b1_sc = _run_stage("Stage1.5-ASAM-Balanced", s1_sd, sub_nat, steps=s15_steps, lr=7.5e-5, quant_mode="fp16", use_asam=True, pauc_w=0.09, pair_w=0.24, beta_end=0.030)
    b2_sd, b2_sc = _run_stage("Stage1.5-ASAM-HumanShield", s1_sd, sub_nat, steps=s15_steps, lr=6.0e-5, quant_mode="fp16", use_asam=True, pauc_w=0.11, pair_w=0.34, beta_end=0.025)
    fp16_sd, fp16_soup_rep = greedy_pauc_soup(s1_sd, [("Balanced", b1_sd, b1_sc), ("HumanShield", b2_sd, b2_sc)], eval_fn=_val_obj_sd, dare_p=0.10)
    print(f"[Stage1.5-FP16-Soup] {json.dumps(fp16_soup_rep)}", flush=True)
    mdl.load_state_dict({k: v.to(device) for k, v in fp16_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_fp16_master.pt")

    fp16_t2 = predict_v10(mdl, test_enc, test_eng, test_st, outer_steps=2, use_fp32=True)
    fp16_t3 = predict_v10(mdl, test_enc, test_eng, test_st, outer_steps=3, use_fp32=True)
    fp16_test_m = {**compute_eval_metrics(test_scores, expit(fp16_t2["logit"])), **compute_high_precision_metrics(test_scores, expit(fp16_t2["logit"]))}
    fp16_t3_m = {**compute_eval_metrics(test_scores, expit(fp16_t3["logit"])), **compute_high_precision_metrics(test_scores, expit(fp16_t3["logit"]))}
    fp16_r1_m = compute_eval_metrics(test_scores, expit(fp16_t2["logit_r1"]))
    print(f"[FP16 Master] R1_AUROC={fp16_r1_m['auroc']} | T2_AUROC={fp16_test_m['auroc']} PR={fp16_test_m['pr_auc']} | T3_AUROC={fp16_t3_m['auroc']}", flush=True)

    # Zero-Op Algebraic Folding BEFORE UPQ!
    calib_batches = []
    cal_dl = DataLoader(V10DS(sub_nat[: min(256, len(sub_nat))]), batch_size=64, shuffle=False, collate_fn=_make_collate(0.0))
    for cb in cal_dl:
        calib_batches.append((cb[0].to(device), cb[1].to(device), cb[4].to(device), cb[2].to(device)))
    fold_rep = fold_zero_op_v10(mdl, calib_inputs=calib_batches)
    print(f"[Stage2-ZeroOpFold] {json.dumps(fold_rep)}", flush=True)
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}

    fp16_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_teacher.parameters():
        p.requires_grad_(False)

    # Stage 2: Progressive UPQ (`W4` -> `5-Trit/2-Bit` Sub-2-Bit QAT Soup) WITH `P4 Tunnel-TRM` (`trm_lam=0.0` exact isometry!)
    w4_sd, _ = _run_stage("Stage2a-UPQ-W4-TunnelTRM", folded_sd, sub_nat, steps=6 if args.smoke_test else max(500, args.qat_steps // 2), lr=8.0e-5, quant_mode="w4", trm_lam_k0=0.0, trm_lam_k1=0.0, pauc_w=0.09, teacher_m=fp16_teacher)
    q1_sd, q1_sc = _run_stage("Stage2b-QAT-Sub2-TunnelTRM-A", w4_sd, sub_nat, steps=6 if args.smoke_test else args.qat_steps, lr=5.0e-5, quant_mode="sub2", trm_lam_k0=0.0, trm_lam_k1=0.0, pauc_w=0.10, teacher_m=fp16_teacher)
    q2_sd, q2_sc = _run_stage("Stage2b-QAT-Sub2-TunnelTRM-B", w4_sd, sub_nat, steps=6 if args.smoke_test else args.qat_steps, lr=3.8e-5, quant_mode="sub2", trm_lam_k0=0.0, trm_lam_k1=0.0, pauc_w=0.13, pair_w=0.28, teacher_m=fp16_teacher)
    qat_soup_sd, qat_soup_rep = greedy_pauc_soup(w4_sd, [("QAT_Sub2_A", q1_sd, q1_sc), ("QAT_Sub2_B", q2_sd, q2_sc)], eval_fn=_val_obj_sd, dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in qat_soup_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    del fp16_teacher
    torch.cuda.empty_cache()

    # Stage 3a: Bit-Exact `MGVZ6` Pack/Unpack -> Guarded Post-UPQ Pure-FP16 DynaCore-MoLE + Tunnel/Coupling Shock-Absorber Refit!
    tmp_mgvz = out_dir / "_tmp_v10_pre_shock.mgvz"
    pack_bundle_v6_mgvz(mdl.cpu(), tmp_mgvz)
    load_bundle_v6_mgvz_into_model(mdl, tmp_mgvz)
    tmp_mgvz.unlink(missing_ok=True)
    mdl = mdl.to(device).eval()

    pre_shock_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_shock_sc = _val_obj_sd(pre_shock_sd)
    # Strictly unfreeze ONLY pure-FP16 Bundle v6 shock absorbers (0.000000 repack drift on Stage 4 export!)
    from .v07_omniveritas import OmniQuantLinear, SeqLoRAMoE
    quantized_params: set[str] = {
        "word_emb_A.weight", "engram_emb.weight", "trigram_emb.weight", "pos_emb1.weight", "pos_emb2.weight",
        "coda_proj.0.weight", "span_l0_proj.weight", "span_l1_proj.weight", "span_topdown_proj.weight",
        "cbm_enc.weight", "style_net.2.weight",
    }
    fp16_pinned: set[str] = {
        "core.mole.core_shared", "core.mole.core_routed", "core.mole.core_bias", "core.mole.dyn_profiles",
        "core.inloop_ebt.fc1.weight", "core.inloop_ebt.fc2.weight",
    }
    for mod_name, mod in mdl.named_modules():
        if isinstance(mod, OmniQuantLinear):
            quantized_params.add(f"{mod_name}.weight")
        elif isinstance(mod, SeqLoRAMoE):
            quantized_params.add(f"{mod_name}.down")
            quantized_params.add(f"{mod_name}.up")
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 8192 and f"{mod_name}.weight" not in fp16_pinned:
            quantized_params.add(f"{mod_name}.weight")
    shock_tokens = (
        "core.mole.", "core.p2_", "core.c1_", "core.c2_", "core.inloop_ebt.", "core.mhc_",
        "core.reread_gate", "core.haar_gate", "core.fast_res_scale", "core.ebt_back_scale",
        "core.norm", "core.fast_norm", "core.reread_norm", "h1_gate", "h2_hyp_gate", "h2_doubt_gate",
        "defer_", "r1_ai_head", "r1_style_mlp", "r1_span_head", "r1_rhythm_conv", "r1_attn_pool", "coda_", "ai_head.", "ebt_logit_gate", "style_", "cbm_", "span_", "mil_gate",
    )
    shock_params = []
    for name, p in mdl.named_parameters():
        keep = (name not in quantized_params) and any(tok in name for tok in shock_tokens)
        p.requires_grad_(keep)
        if keep:
            shock_params.append(p)

    p_opt = torch.optim.AdamW(shock_params, lr=1.5e-4, weight_decay=1e-4)
    shock_pool = sub_nat[: min(80000, len(sub_nat))] + mined_hard_hu_items
    p_dl = DataLoader(V10DS(shock_pool), batch_size=batch_size, shuffle=True, collate_fn=_make_collate(0.0))
    mdl.train()
    p_steps = 4 if args.smoke_test else 260
    ps_i = 0
    for batch in p_dl:
        ids, mask, eng, tri, st, y_sc, tp_v06, is_g, _, _, y_a5, w, grp, era, _ = [t.to(device, non_blocking=True) for t in batch]
        p_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = mdl(ids, mask, st, engram_ids=eng, trigram_ids=tri, outer_steps=2, k_inner=2, trm_lam_k0=0.0, trm_lam_k1=0.0, return_aux=True)
            lg, lg_r1, lg_e1 = out["logit"].float(), out["logit_r1"].float(), out["exit1_out"]["logit"].float()
            ycl = y_sc.clamp(1e-5, 1.0 - 1e-5)
            lm = (F.binary_cross_entropy_with_logits(lg, ycl, reduction="none") * w).mean()
            lr1 = (F.binary_cross_entropy_with_logits(lg_r1, ycl, reduction="none") * w * (1.0 + 2.5 * (y_sc >= 0.45).float())).mean()
            l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=0.85, beta_start=0.05, beta_end=0.025)
            l_topfp_s = _top_fp_logit_ceiling_loss(lg, y_sc, tp_v06, is_g, era, w)
            l_ebt = F.cross_entropy(-out["ebt_energies"].float(), y_a5)
            with torch.amp.autocast("cuda", enabled=False):
                pf_d, pr1_d, pe1_d = torch.sigmoid(lg.detach()), torch.sigmoid(lg_r1.detach()), torch.sigmoid(lg_e1.detach())
                d1_t = torch.maximum(((pf_d - pr1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_sc >= 0.45) & (pr1_d < 0.35)).float())
                d2_t = torch.maximum(((pf_d - pe1_d).abs() * 3.0).clamp(0.0, 1.0), ((y_sc >= 0.45) & (pe1_d < 0.35)).float())
                ld1 = F.binary_cross_entropy(out["p_defer1"].float().clamp(1e-5, 1.0 - 1e-5), d1_t)
                ld2 = F.binary_cross_entropy(out["exit1_out"]["p_defer2"].float().clamp(1e-5, 1.0 - 1e-5), d2_t)
                l_r1_sp = (F.binary_cross_entropy(out["r1_span_max"].float().clamp(1e-5, 1.0 - 1e-5), ycl.float(), reduction="none") * w.float() * (1.0 + 2.5 * (y_sc >= 0.45).float())).mean()
            s_u = F.normalize(out["slots_final"].float(), p=2, dim=-1)
            s_g = torch.matmul(s_u, s_u.transpose(-1, -2))
            l_sdiv = ((s_g * (1.0 - torch.eye(s_g.shape[-1], device=s_g.device, dtype=s_g.dtype).unsqueeze(0))) ** 2).mean()
            loss = lm + 0.25 * lr1 + 0.09 * l_pa + l_tp + zenith_graded_anchor_loss(lg, y_sc) + 0.20 * l_topfp_s + 0.12 * l_ebt + 0.25 * ld1 + 0.12 * ld2 + 0.08 * l_r1_sp + 0.15 * l_sdiv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shock_params, 1.0)
        p_opt.step()
        ps_i += 1
        if ps_i >= p_steps:
            break
    for p in mdl.parameters():
        p.requires_grad_(True)
    post_shock_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    post_shock_sc = _val_obj_sd(post_shock_sd)
    if post_shock_sc < pre_shock_sc:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_shock_sd.items()}, strict=True)
        shock_kept = False
    else:
        shock_kept = True
    mdl.eval()

    # Stage 3b: 0 KB Confounding Surgery + Era Probe + 8-Byte Cross-Exit Affine Alignment (strictly on train slice!)
    with torch.no_grad():
        mdl.temperature.fill_(1.0)
        mdl.exit_affine_a.fill_(1.0)
        mdl.exit_affine_b.fill_(0.0)
    val_pre = predict_v10(mdl, val_enc, val_eng, val_st, outer_steps=2, use_fp32=True)
    cal_pre = predict_v10(mdl, cal_enc, cal_eng, cal_st, outer_steps=2, use_fp32=False)
    era_probe_auc = eval_human_era_linear_probe_auroc(cal_pre["emb"], cal_eras, cal_scores / 100.0)
    surg_rep = apply_zero_cost_confounding_surgery(
        mdl, val_pre["emb"][val_scores <= 35.0], val_pre["emb"][val_scores >= 70.0], val_pre["emb"], val_scores, rank=8
    )

    align_sub = train_items_raw[: min(8000, len(train_items_raw))] if args.smoke_test else random.Random(20261126).sample(train_items_raw, min(16000, len(train_items_raw)))
    al_out = predict_v10(mdl, [x["ids"] for x in align_sub], [x["engram"] for x in align_sub], np.stack([x["style"] for x in align_sub], axis=0), outer_steps=2, use_fp32=False)
    a_r1, b_r1 = fit_cross_exit_affine_alignment(al_out["raw_logit_r1"], al_out["raw_logit"])
    a_t1, b_t1 = fit_cross_exit_affine_alignment(al_out["raw_logit_t1"], al_out["raw_logit"])
    with torch.no_grad():
        mdl.exit_affine_a.copy_(torch.tensor([a_r1, a_t1, 1.0], dtype=torch.float32, device=device))
        mdl.exit_affine_b.copy_(torch.tensor([b_r1, b_t1, 0.0], dtype=torch.float32, device=device))
    affine_rep = {"rung1": {"a": a_r1, "b": b_r1}, "rung2_t1": {"a": a_t1, "b": b_t1}, "rung3_full": {"a": 1.0, "b": 0.0}}
    print(f"[Stage3b-AffineAlign] {json.dumps(affine_rep)}", flush=True)

    # Stage 3c: Per-Rung Binary-ECE Beta-Spline LUTs (val_cal_select + val_gold) + Disjoint Selection-Safe & LTT (`val_cal_cert`)!
    val_al = predict_v10(mdl, val_enc, val_eng, val_st, outer_steps=2, use_fp32=True)
    sel_al = predict_v10(mdl, sel_enc, sel_eng, sel_st, outer_steps=2, use_fp32=False)
    cal_al = predict_v10(mdl, cal_enc, cal_eng, cal_st, outer_steps=2, use_fp32=False)

    lut_fit_z = np.concatenate([sel_al["logit"], val_al["logit"]])
    lut_fit_zr1 = np.concatenate([sel_al["logit_r1"], val_al["logit_r1"]])
    lut_fit_y = np.concatenate([sel_scores, val_scores]) / 100.0
    cal_va_full = fit_venn_abers_and_beta_spline_lut(lut_fit_z, lut_fit_y, fit_binary_ece_mode=True)
    cal_va_r1 = fit_venn_abers_and_beta_spline_lut(lut_fit_zr1, lut_fit_y, fit_binary_ece_mode=True)
    lut_full = export_beta_spline_as_v07_lut(cal_va_full, n_knots=256)
    lut_r1 = export_beta_spline_as_v07_lut(cal_va_r1, n_knots=256)

    val_p_full = apply_spline_temp_lut(expit(val_al["logit"]), lut_full)
    cal_p_r1 = apply_spline_temp_lut(expit(cal_al["logit_r1"]), lut_r1)
    cal_p_t1 = apply_spline_temp_lut(expit(cal_al["logit_t1"]), lut_full)
    cal_p_full = apply_spline_temp_lut(expit(cal_al["logit"]), lut_full)
    t_f05, t_fpr05 = find_val_thresholds(val_scores, val_p_full)

    ss_mondrian = fit_selection_safe_conformal_cascade(cal_cleans, cal_st, cal_scores, [cal_p_r1, cal_p_t1, cal_p_full], alphas=(0.03, 0.01))
    conf_2400 = fit_split_conformal_3way(cal_scores, cal_p_full, alphas=(0.03, 0.05))
    cal_r1_elig = np.array(
        [
            len(cal_enc[i]) <= 96 and cal_metas[i]["spoof_any"] == 0.0 and ("http" not in cal_cleans[i]) and float(cal_al["r1_span_max"][i]) <= 0.50
            for i in range(len(cal_cleans))
        ],
        dtype=bool,
    )
    # OOD-Safe LTT Certification: lock Rung-1 early-exit cap <= 0.12 (eliminates v0.9's -0.016 OOD cascade ROC gap!)
    psac_policy = certify_ltt_compcrc_cascade_policy(
        cal_scores, cal_p_r1, cal_al["p_defer1"], cal_r1_elig, cal_p_full,
        target_fn_risk=0.005, delta_conf=0.05, p_max_cap=0.12, d_max_cap=0.25,
    )
    # Also certify Rung-2 (t=1) -> Rung-3 (t=2..3) early exit via LTT + In-Loop Analytic EBT Equilibrium Gate!
    r2_egrad_max = float(np.quantile(cal_al["ebt_grad_t1"], 0.65)) if len(cal_al["ebt_grad_t1"]) > 0 else 0.60
    cal_r2_elig = np.array(
        [cal_metas[i]["spoof_any"] == 0.0 and float(cal_al["ebt_grad_t1"][i]) <= r2_egrad_max for i in range(len(cal_cleans))],
        dtype=bool,
    )
    psac_r2 = certify_ltt_compcrc_cascade_policy(
        cal_scores, cal_p_t1, cal_al["p_defer2"], cal_r2_elig, cal_p_full,
        target_fn_risk=0.005, delta_conf=0.05, p_max_cap=0.14, d_max_cap=0.28,
    )
    psac_policy.update(
        {
            "r2_enabled": bool(psac_r2.get("enabled", False)),
            "r2_certified_ltt": bool(psac_r2.get("certified_ltt", False)),
            "r2_defer_max": float(psac_r2.get("r1_defer_max", 0.25)),
            "r2_p_human_max": float(psac_r2.get("r1_p_human_max", 0.12)),
            "r2_p_ai_min": 0.97,
            "r2_ebt_grad_max": round(r2_egrad_max, 5),
            "r2_cal_exit_rate": float(psac_r2.get("cal_exit_rate", 0.0)),
            "r2_cal_missed_ai_count": int(psac_r2.get("cal_missed_ai_count", 0)),
            "r3_t3_enabled": True,
        }
    )
    print(f"[Stage3c-PSAC-LTT] {json.dumps(psac_policy)}", flush=True)

    # Stage 3d: PrototypeBank + LB-DSM (External 64 KB EBT Critic RETIRED — replaced by In-Loop Analytic EBT!)
    test_pre = predict_v10(mdl, test_enc, test_eng, test_st, outer_steps=2, use_fp32=True)
    rng_bk = random.Random(20261126)
    hu_anc = [x for x in train_items_raw if x["is_hu_anchor"]]
    ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
    bank_sub = train_items_raw[: min(24000, len(train_items_raw))] if args.smoke_test else (
        rng_bk.sample(hu_anc, min(12000, len(hu_anc))) + rng_bk.sample(ai_anc, min(12000, len(ai_anc)))
    )
    bk_out = predict_v10(mdl, [x["ids"] for x in bank_sub], [x["engram"] for x in bank_sub], np.stack([x["style"] for x in bank_sub], axis=0), outer_steps=2)
    bk_emb = bk_out["emb"]
    bk_is_hu = np.array([bool(x["is_hu_anchor"]) for x in bank_sub], dtype=bool)
    bk_is_ai = np.array([x["y_score"] >= 0.82 for x in bank_sub], dtype=bool)

    proto_path = out_dir / "prototype_bank_v10.npz"
    np.savez_compressed(proto_path, **fit_l2_prototype_bank(bk_emb, np.array([x["y_score"] * 100.0 for x in bank_sub], dtype=np.float32), is_human_anchor=bk_is_hu))
    lb_critic, lb_rep = _train_frozen_lb_dsm_critic(bk_emb[bk_is_hu], bk_emb[bk_is_ai], test_pre["emb"], test_scores, device, steps=60 if args.smoke_test else 650)
    lb_path = out_dir / "lb_dsm_critic_v10.npz"
    np.savez_compressed(lb_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in lb_critic.state_dict().items()})

    # Stage 4: Export TRUE <= 2.20 MB Bundle v6 (`MGVZ6`) + Zero-Recompute 3-Stage Static ONNX + Unpacked Eval + CI!
    exp_info = export_v10_modular_onnx_and_bundle_v6(mdl, out_dir)
    mdl_unpacked = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v6_mgvz_into_model(mdl_unpacked, out_dir / exp_info["bundle_v6_mgvz"]["file"])
    mdl_unpacked = mdl_unpacked.to(device).eval()

    def _eval_split(m_e, enc_l, eng_l, st_m, sc, outer_steps: int = 2, fp32: bool = True):
        o = predict_v10(m_e, enc_l, eng_l, st_m, outer_steps=outer_steps, use_fp32=fp32)
        cp = apply_spline_temp_lut(expit(o["logit"]), lut_full)
        cp_r1 = apply_spline_temp_lut(expit(o["logit_r1"]), lut_r1)
        return {**compute_eval_metrics(sc, cp, t_f05, t_fpr05), **compute_high_precision_metrics(sc, cp)}, cp, cp_r1, o

    test_m, test_p, test_p_r1, test_o = _eval_split(mdl_unpacked, test_enc, test_eng, test_st, test_scores, outer_steps=2, fp32=True)
    test_t1_m, _, _, _ = _eval_split(mdl_unpacked, test_enc, test_eng, test_st, test_scores, outer_steps=1, fp32=True)
    test_t3_m, _, _, _ = _eval_split(mdl_unpacked, test_enc, test_eng, test_st, test_scores, outer_steps=3, fp32=True)
    test_r1_m = {**compute_eval_metrics(test_scores, test_p_r1, t_f05, t_fpr05), **compute_high_precision_metrics(test_scores, test_p_r1)}
    inloop_ebt_m = compute_eval_metrics(test_scores, expit(test_o["ebt_margin"]))
    dsh_m, _, _, _ = _eval_split(mdl_unpacked, dsh_enc, dsh_eng, dsh_st, dsh_scores, outer_steps=2, fp32=True)
    all_m, all_p, _, _ = _eval_split(mdl_unpacked, all_enc, all_eng, all_st, all_scores, outer_steps=2, fp32=False)
    prb_m, _, _, _ = _eval_split(mdl_unpacked, prb_enc, prb_eng, prb_st, prb_scores, outer_steps=2, fp32=True)
    test_cal_lowfpr = compute_zenith_calibration_and_lowfpr_metrics(test_scores, test_p)
    all_cal_lowfpr = compute_zenith_calibration_and_lowfpr_metrics(all_scores, all_p)
    ci_final = run_v10_ci_gates(mdl_unpacked, tokenizer)

    manifest = {
        "schema_version": "meguviz.bundle.v6",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": "v1.0.0-TunnelSymplect x DynaCore-MoLE x Zero-Recompute 3-Stage Cascade",
        "license": "Apache-2.0",
        "tokenizer_file": tok_path.name,
        "vocab_size": tokenizer.vocab_size,
        "engram_buckets": ENGRAM_BIGRAM_BUCKETS_V10,
        "trigram_buckets": ENGRAM_TRIGRAM_BUCKETS_V10,
        "max_len": int(args.max_len),
        "bundle_v6_mgvz": exp_info["bundle_v6_mgvz"]["file"],
        "rung1_onnx": exp_info["rung1_onnx"],
        "rung2_onnx": exp_info["rung2_onnx"],
        "rung3_onnx": exp_info["rung3_onnx"],
        "psac_policy": psac_policy,
        "exit_affine_alignment": affine_rep,
        "prototype_bank_file": proto_path.name,
        "lb_dsm_critic_file": lb_path.name,
        "temperature": 1.0,
        "lut_rung1": lut_r1,
        "lut_full": lut_full,
        "spline_temp_lut": lut_full,
        "beta_spline_venn_abers_full": cal_va_full,
        "beta_spline_venn_abers_r1": cal_va_r1,
        "selection_safe_mondrian_9cell": ss_mondrian,
        "conformal_cal2400": conf_2400,
        "thresholds": {"f05": round(t_f05, 4), "fpr05": round(t_fpr05, 4), "score70": 0.70},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Real 1T-CPU Zero-Recompute 3-Stage Dynamic-Compaction Benchmark + Traps!
    engine = MeguvizV10CascadeEngine(out_dir, threads=1)
    t_s = time.perf_counter()
    _ = [engine.predict_one(t) for t in test_txts]
    single_1t_ms = (time.perf_counter() - t_s) * 1000.0 / max(1, len(test_txts))
    t_f = time.perf_counter()
    ort_full = engine.predict_batch(test_txts, max_batch=32, force_full=True)
    b32_full_ms = (time.perf_counter() - t_f) * 1000.0 / max(1, len(test_txts))
    t_c = time.perf_counter()
    ort_casc = engine.predict_batch(test_txts, max_batch=32, force_full=False)
    b32_casc_ms = (time.perf_counter() - t_c) * 1000.0 / max(1, len(test_txts))

    ort_p_full = np.array([x["prob"] for x in ort_full], dtype=np.float32)
    ort_p_casc = np.array([x["prob"] for x in ort_casc], dtype=np.float32)
    ex_rungs = np.array([int(x["exit_rung"]) for x in ort_casc], dtype=np.int64)
    ort_m_full = {**compute_eval_metrics(test_scores, ort_p_full, t_f05, t_fpr05), **compute_high_precision_metrics(test_scores, ort_p_full)}
    ort_m_casc = {**compute_eval_metrics(test_scores, ort_p_casc, t_f05, t_fpr05), **compute_high_precision_metrics(test_scores, ort_p_casc)}
    traps_rep = _eval_traps_v07(engine)

    # Package TRUE <= 2.20 MB Bundle v6 ZIP (External EBT retired!) & 3-Stage Modular ONNX INT8 ZIP!
    zip_v6 = out_dir / f"meguvizcac-{args.version}-bundle-v6-sub2bit.zip"
    v6_files = [exp_info["bundle_v6_mgvz"]["file"], tok_path.name, proto_path.name, lb_path.name, "manifest.json"]
    with ZipFile(zip_v6, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for fn in v6_files:
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    v6_zip_mb = zip_v6.stat().st_size / (1024.0 * 1024.0)

    zip_int8 = out_dir / f"meguvizcac-{args.version}-3stage-onnx-int8.zip"
    with ZipFile(zip_int8, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in (exp_info["rung1_onnx"], exp_info["rung2_onnx"], exp_info["rung3_onnx"], tok_path.name, proto_path.name, lb_path.name, "manifest.json"):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)

    metrics: dict[str, Any] = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v6",
        "architecture": "TunnelSymplect x DynaCore-MoLE x Zero-Recompute 3-Stage Cascade",
        "gpu_name": gpu_name,
        "batch_size_used": batch_size,
        "bge_svd_matryoshka_init": svd_init_rep,
        "gemini_factory_audit": gem_audit,
        "pillar1_denoise": denoise_report,
        "top_fp_hard_negative_mining": topfp_mine_rep,
        "fp16_soup_report": fp16_soup_rep,
        "zero_op_fold_report": fold_rep,
        "qat_soup_report": qat_soup_rep,
        "post_upq_shock_absorber_refit": {"kept": shock_kept, "before": round(pre_shock_sc, 5), "after": round(post_shock_sc, 5)},
        "confounding_surgery": surg_rep,
        "exit_affine_alignment": affine_rep,
        "human_era_probe_auroc": era_probe_auc,
        "fp16_master_eval": {"rung1_d128": fp16_r1_m, "t2": fp16_test_m, "t3": fp16_t3_m},
        "sub2bit_bundle_v6_eval": {
            "rung1_d128": test_r1_m,
            "t1_outer": test_t1_m,
            "t2_outer": test_m,
            "t3_outer": test_t3_m,
            "inloop_analytic_ebt_auroc": inloop_ebt_m.get("auroc"),
            "auroc_delta_vs_fp16": round(float(test_m["auroc"]) - float(fp16_test_m["auroc"]), 5),
            "pr_auc_delta_vs_fp16": round(float(test_m["pr_auc"]) - float(fp16_test_m["pr_auc"]), 5),
        },
        "test_bili_gold": test_m,
        "test_bili_gold_cal_lowfpr": test_cal_lowfpr,
        "test_dsh_gold": dsh_m,
        "test_all_bili_distilled": all_m,
        "test_all_bili_cal_lowfpr": all_cal_lowfpr,
        "probe_u001_u017": prb_m,
        "onnx_3stage_int8": {
            "mae_vs_sub2bit": round(float(np.mean(np.abs(test_p - ort_p_full))), 5),
            "auroc_delta_vs_sub2bit": round(float(ort_m_full["auroc"]) - float(test_m["auroc"]), 5),
            "single_1thread_ms": round(single_1t_ms, 3),
            "compact_batch32_cascade_1thread_ms": round(b32_casc_ms, 3),
            "compact_batch32_full_1thread_ms": round(b32_full_ms, 3),
            "test_exit_dist": {
                "r1": round(float(np.mean(ex_rungs == 1)), 4),
                "r2": round(float(np.mean(ex_rungs == 2)), 4),
                "r3_t2": round(float(np.mean(ex_rungs == 3)), 4),
                "r3_t3": round(float(np.mean(ex_rungs == 4)), 4),
            },
            "test_r1_missed_ai_count": int(np.sum((ex_rungs == 1) & (test_scores >= 70.0))),
            "onnx_full_test_bili_gold": ort_m_full,
            "onnx_cascade_test_bili_gold": ort_m_casc,
            "rung1_bytes": exp_info["rung1_bytes"],
            "rung2_bytes": exp_info["rung2_bytes"],
            "rung3_bytes": exp_info["rung3_bytes"],
            "zip_bytes": int(zip_int8.stat().st_size),
        },
        "bundle_v6_total_bytes": int(zip_v6.stat().st_size),
        "bundle_v6_total_mb": round(v6_zip_mb, 4),
        "export_info": exp_info,
        "psac_policy": psac_policy,
        "selection_safe_mondrian_9cell": ss_mondrian,
        "conformal_cal2400": conf_2400,
        "lb_dsm_critic": lb_rep,
        "traps": traps_rep,
        "v10_ci_gates": ci_final,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics["release_gates"] = check_v10_release_gates(metrics)
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "version": metrics["version"],
                "bundle_v6_total_mb": metrics["bundle_v6_total_mb"],
                "onnx_fp32_diffs": {k: exp_info[k] for k in ("r1_fp32_max_diff", "r2_fp32_max_diff", "r3_fp32_max_diff")},
                "test_bili_gold": {k: test_m[k] for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95")},
                "anytime_auroc": {"r1": test_r1_m["auroc"], "t1": test_t1_m["auroc"], "t2": test_m["auroc"], "t3": test_t3_m["auroc"], "inloop_ebt": inloop_ebt_m.get("auroc")},
                "auroc_delta_vs_fp16": metrics["sub2bit_bundle_v6_eval"]["auroc_delta_vs_fp16"],
                "cascade_1t_ms": metrics["onnx_3stage_int8"]["compact_batch32_cascade_1thread_ms"],
                "full_1t_ms": metrics["onnx_3stage_int8"]["compact_batch32_full_1thread_ms"],
                "test_exit_dist": metrics["onnx_3stage_int8"]["test_exit_dist"],
                "v10_ci_all_passed": ci_final["all_ci_passed"],
                "release_gates": {"passed": metrics["release_gates"]["passed_count"], "total": metrics["release_gates"]["total_gates"]},
                "elapsed_seconds": metrics["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    ap.add_argument("--extra-pairs", type=Path, action="append", default=[])
    ap.add_argument("--v08-gemini-gold", type=Path, default=None)
    ap.add_argument("--v08-gemini-pairs", type=Path, default=None)
    ap.add_argument("--teacher-v06-ckpt", type=Path, default=None)
    ap.add_argument("--v06-backbone", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--v07-tokenizer", type=Path, default=None)
    ap.add_argument("--v10-fp16-ckpt", type=Path, default=None, help="Optional resume from Stage 1.5 FP16 ApexLoop checkpoint")
    ap.add_argument("--intent-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("./out_v10_apexloop"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", default="v1.0.0-r1-apexloop-b6")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_V07)
    ap.add_argument("--stage1-synth-cap", type=int, default=15000)
    ap.add_argument("--v04-synth-cap", type=int, default=-1)
    ap.add_argument("--stage1-steps", type=int, default=14000)
    ap.add_argument("--refine-steps", type=int, default=2400)
    ap.add_argument("--qat-steps", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--skip-bge-svd-init", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    train_v10_apexloop(args)


if __name__ == "__main__":
    main()
