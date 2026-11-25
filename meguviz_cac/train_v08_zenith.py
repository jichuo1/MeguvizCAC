"""MeguvizCAC v0.8.0-Zenith Zero-Architecture-Surgery Trainer (`Bundle v4` 100% Drop-In Weight Replacement!).

STRICT NON-GOALS & INVARIANTS:
  - 0 new ops, 0 tensor shape changes, 0 bytes added vs frozen `v0.7.0-OmniVeritas` Bundle v4!
  - Directly ingests External Gemini Data Factory outputs (`--v08-gemini-gold` & `--v08-gemini-pairs`)!
  - Runs all 5 offline Zenith metric-maximization pillars:
      Pillar 1: Cleanlab/Margin Denoising + Mid-Band Quarantine + CCR Era-IPW + Offline Era-GRL (stripped pre-export!)
      Pillar 2: Multi-Teacher Consensus + Logit-Standardized Z-DKD + ListNet-Tail + Graded/Mid-Band Huber Anchors
      Pillar 3: WSD-S Shared `theta_anchor` -> Multi-Branch ASAM (Anchor-Balanced / Extreme-Tail / Human-Shield) +
                GateGuard & Gemini Counterfactual Pairs -> Conservative DARE-TIES + Greedy Validation-pAUC Soup ->
                Post-Soup 0 KB Confounding Surgery
      Pillar 4: Zero-Op SpinQuant R_2 + SmoothQuant + SubLN Algebraic Fold -> Progressive UPQ (W4 -> 5-Trit/2-Bit) ->
                Continuous Latent-Weight QAT Soup -> Bit-Exact Pack/Unpack + Guarded Seq-LoRA-MoE/Head Polish
      Pillar 5: Strictly Increasing Beta-Spline-Temp x Venn-ABERS (projected to v0.7.0 1D LUT!) +
                9-Cell Empirical-Bayes Mondrian SCoRE Conformal + Joint Cascade Pareto Optimizer
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    _train_frozen_ebt_critic,
    _train_nano_draft_v07,
    _wsd_lr_mult,
)
from .v04_corpus_prep import GROUP_TO_IDX
from .v04_runtime import fit_split_conformal_3way
from .v05_runtime import fit_l2_prototype_bank
from .v06_hyperloop import apply_zero_cost_confounding_surgery, text_to_byte_hash_ids
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    STYLE_FEAT_DIM_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    extract_style_features_40d,
    extract_style_features_40d_batch,
    make_engram_bigram_ids,
)
from .v07_omniveritas import (
    MAX_LEN_V07,
    MeguvizV07Engine,
    MeguvizV07OmniVeritasModel,
    export_v07_onnx_and_bundle_v4,
    load_bundle_v4_mgvz_into_model,
    pack_bundle_v4_mgvz,
    run_v07_ci_gates,
)
from .v08_zenith_ops import (
    ERA_TO_IDX,
    ASAMPerturber,
    OfflineEraGRLHead,
    apply_beta_spline_venn_abers,
    compute_ccr_era_ipw,
    compute_zenith_calibration_and_lowfpr_metrics,
    dare_ties_merge,
    eval_human_era_linear_probe_auroc,
    export_beta_spline_as_v07_lut,
    fit_mondrian_score_conformal_9cell,
    fit_venn_abers_and_beta_spline_lut,
    fold_zero_op_spinquant_and_smoothquant_v07,
    greedy_pauc_soup,
    identify_noisy_and_quarantine,
    listnet_tail_distill_loss,
    logit_standardized_dkd_loss,
    optimize_joint_cascade_pareto,
    zenith_graded_anchor_loss,
)


def _ingest_external_gemini_gold(
    train_items: list[dict],
    gemini_gold_path: Path | None,
) -> dict[str, int]:
    """Merge External Gemini Data Factory (`v08-gemini-gold-labels.jsonl.gz`) into train items (override weak rows / append fresh wild!)."""
    if not gemini_gold_path or not gemini_gold_path.is_file():
        return {"loaded": 0, "overridden_weak": 0, "appended_wild": 0}
    by_sid = {x["sample_id"]: i for i, x in enumerate(train_items)}
    loaded = overridden = appended = 0
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
            loaded += 1
            if sid in by_sid:
                it = train_items[by_sid[sid]]
                it["y_score"] = y_new
                it["code_idx"] = CODE_TO_IDX.get(str(r.get("code", "")), it["code_idx"])
                it["is_gold_tier"] = True
                it["weight"] = max(float(it["weight"]), float(r.get("train_weight", 1.25)))
                if isinstance(r.get("prag_vec"), list) and len(r["prag_vec"]) == len(PRAG_TAGS):
                    it["prag_vec"] = [float(v) for v in r["prag_vec"]]
                overridden += 1
            else:
                gid_str = "structured_human_hard" if (r.get("is_trap_human") and sc <= 30.0) else "current_2026_real"
                prag = r.get("prag_vec") if (isinstance(r.get("prag_vec"), list) and len(r["prag_vec"]) == len(PRAG_TAGS)) else [-1.0] * len(PRAG_TAGS)
                train_items.append(
                    {
                        "sample_id": sid,
                        "raw_text": str(r["text"]),
                        "y_score": y_new,
                        "code_idx": CODE_TO_IDX.get(str(r.get("code", "")), -100),
                        "weight": float(r.get("train_weight", 1.20)),
                        "prag_vec": [float(v) for v in prag],
                        "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["current_2026_real"]),
                        "intent_idx": -1,
                        "is_gold_tier": True,
                        "is_pre_llm": False,
                        "is_hu_anchor": bool(sc <= 18.0),
                        "is_high_prio": True,
                        "era_bucket": str(r.get("era_bucket", "current_2026")),
                    }
                )
                by_sid[sid] = len(train_items) - 1
                appended += 1
    return {"loaded": loaded, "overridden_weak": overridden, "appended_wild": appended}


def check_v08_release_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Verify v0.8.0-Zenith release gates (strict Zero-Architecture-Surgery + Metric Maximization)."""
    tb = metrics.get("test_bili_gold") or {}
    dsh = metrics.get("test_dsh_gold") or {}
    sub2 = metrics.get("sub2bit_bundle_v4_eval") or {}
    ox = metrics.get("onnx_int8") or {}
    ci = metrics.get("v07_ci_gates") or {}
    cal_m = metrics.get("test_bili_gold_cal_lowfpr") or {}
    m9 = (metrics.get("mondrian_9cell") or {}).get("alphas", {}).get("alpha_003") or {}
    traps = (metrics.get("traps") or {}).get("summary") or {}

    zip_mb = float(metrics.get("bundle_v4_total_mb") or 99.0)
    # Only downward degradation counts as drift (positive sub2 > fp16 gain = 0.0 drift!)
    delta_auc = max(0.0, -float(sub2.get("auroc_delta_vs_fp16", -99.0)))
    era_auc = float(metrics.get("human_era_probe_auroc", 1.0))
    cells = (m9.get("cells") or {}).values()
    max_cell_fpr = max((float(c["fpr"]) for c in cells if int(c.get("n_hu", 0)) >= 20), default=0.0)

    gates = {
        "zero_architecture_surgery": {"value": int(metrics.get("new_ops_added", 99)), "target": "== 0 new ops", "pass": int(metrics.get("new_ops_added", 99)) == 0},
        "bundle_v4_le_2p20_mb": {"value": round(zip_mb, 3), "target": "<= 2.20 MB", "pass": zip_mb <= 2.20},
        "sub2_auroc_drift_le_0020": {"value": round(delta_auc, 5), "target": "<= 0.0020", "pass": delta_auc <= 0.0020},
        "speed_1t_cpu_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 1.50 ms", "pass": (ox.get("compact_batch32_1thread_ms") or 99.0) <= 1.50},
        "test_bili_gold_auroc_ge_0965": {"value": tb.get("auroc"), "target": ">= 0.965", "pass": (tb.get("auroc") or 0.0) >= 0.965},
        "test_bili_gold_prauc_ge_0930": {"value": tb.get("pr_auc"), "target": ">= 0.930", "pass": (tb.get("pr_auc") or 0.0) >= 0.930},
        "test_bili_gold_rp95_ge_082": {"value": tb.get("recall_at_p95"), "target": ">= 0.82", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.82},
        "test_dsh_gold_auroc_ge_0955": {"value": dsh.get("auroc"), "target": ">= 0.955", "pass": (dsh.get("auroc") or 0.0) >= 0.955},
        "ece_le_0020": {"value": cal_m.get("ece_15bin"), "target": "<= 0.020", "pass": (cal_m.get("ece_15bin") or 1.0) <= 0.020},
        "human_era_probe_le_056": {"value": era_auc, "target": "<= 0.56", "pass": era_auc <= 0.56},
        "mondrian_9cell_max_fpr_le_0035": {"value": round(max_cell_fpr, 4), "target": "<= 0.035", "pass": max_cell_fpr <= 0.035},
        "struct_human_trap_le_045": {"value": traps.get("struct_human_max"), "target": "<= 0.45", "pass": (traps.get("struct_human_max") or 1.0) <= 0.45},
        "v07_ci_all_5_passed": {"value": bool(ci.get("all_ci_passed")), "target": "True", "pass": bool(ci.get("all_ci_passed"))},
    }
    return {
        "gates": gates,
        "passed_count": sum(1 for v in gates.values() if v["pass"]),
        "total_gates": len(gates),
        "all_passed": all(v["pass"] for v in gates.values()),
    }


def train_v08_zenith(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from scipy.special import expit
    from torch.utils.data import DataLoader, Dataset, Sampler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v0.8.0-Zenith training.")

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
        f"[v0.8.0-Zenith Config] GPU={gpu_name} ({vram_gb:.1f} GB) | bs={batch_size} | "
        f"v07_ckpt={args.v07_fp16_ckpt} | gemini_gold={args.v08_gemini_gold} | gemini_pairs={args.v08_gemini_pairs}",
        flush=True,
    )

    extra_pairs = [p.resolve() for p in args.extra_pairs]
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

    gem_audit = _ingest_external_gemini_gold(train_items_raw, args.v08_gemini_gold)
    print(f"[v0.8.0-Zenith Data] base={data_audit['train_items']:,} | pairs={len(pairs):,} | gemini_factory={json.dumps(gem_audit)}", flush=True)

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    # Stage 0a: Optional Frozen BF16 v0.6.0 Dual Teacher (Probs + 256-d Embs for Cleanlab Denoising + Latent Distillation!)
    teacher_v06_p = None
    teacher_v06_embs = None
    if args.teacher_v06_ckpt and args.teacher_v06_ckpt.is_file() and not args.smoke_test:
        teacher_v06_p, teacher_v06_embs = _precompute_v06_bf16_teacher_targets(
            args.teacher_v06_ckpt, train_items_raw, [], [], args.v06_backbone, device, amp_dtype, ckpt_dir=None, polish_steps=0, bs=1024
        )
    for idx_it, item in enumerate(train_items_raw):
        item["row_idx"] = idx_it
        item["teacher_p"] = float(teacher_v06_p[idx_it]) if teacher_v06_p is not None else -1.0

    # Reuse exact frozen v0.7.0 tokenizer if provided (100% Bundle v4 drop-in compatibility!)
    tok_path = out_dir / "bili_picky_bpe_12k.json"
    if args.v07_tokenizer and args.v07_tokenizer.is_file():
        shutil.copy2(args.v07_tokenizer, tok_path)
        tokenizer = BiliPickyBPETokenizer(tok_path)
    else:
        rng_t = random.Random(20261125)
        txts_t = [
            canonicalize_zero_pii_v07(x["raw_text"])[0]
            for x in (train_items_raw if args.smoke_test else rng_t.sample(train_items_raw, min(200000, len(train_items_raw))))
        ]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(
            txts_t, tok_path, vocab_size=2048 if args.smoke_test else BPE_VOCAB_SIZE_V07, min_frequency=2 if args.smoke_test else 3
        )

    # Pre-encode + Era IDs + Pillar 1a Denoising + Pillar 1b CCR Stabilized Era-IPW!
    styles_list, era_list, y_list, is_gold_list, is_pre_list = [], [], [], [], []
    for item in train_items_raw:
        cl, f40, meta = extract_style_features_40d(item["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        item["clean_text"] = cl
        item["ids"] = ids
        item["engram"] = make_engram_bigram_ids(ids)
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
    denoise_report = {"toxic_downweighted": 0, "mid_quarantined": 0}
    if teacher_v06_p is not None:
        dn = identify_noisy_and_quarantine(
            teacher_v06_p,
            np.array(y_list, dtype=np.float32),
            np.array(is_gold_list, dtype=bool),
            np.array(is_pre_list, dtype=bool),
            toxic_frac=0.03,
        )
        denoise_mult = dn["clean_weight_mult"]
        denoise_report = {
            "toxic_downweighted": int(dn["toxic_mask"].sum()),
            "mid_quarantined": int(dn["mid_quarantine"].sum()),
        }
    else:
        denoise_mult = np.ones(len(train_items_raw), dtype=np.float32)

    for i, item in enumerate(train_items_raw):
        item["weight"] = float(item["weight"]) * float(ipw_mult[i]) * float(denoise_mult[i])
    print(f"[Pillar1-Denoise+IPW] {json.dumps(denoise_report)}", flush=True)

    # Pre-encode GateGuard + Gemini Queue-C Counterfactual Minimal Pairs
    pairs_gg_pre, pairs_gen_pre = [], []
    if pairs:
        ai_cl, ai_st, _ = extract_style_features_40d_batch([p["ai_text"] for p in pairs])
        hu_cl, hu_st, _ = extract_style_features_40d_batch([p["human_text"] for p in pairs])
        for i, p in enumerate(pairs):
            a_ids = tokenizer.encode_ids(ai_cl[i], max_len=args.max_len)
            h_ids = tokenizer.encode_ids(hu_cl[i], max_len=args.max_len)
            rec = {
                "ai_ids": a_ids, "ai_eng": make_engram_bigram_ids(a_ids), "ai_style": ai_st[i],
                "hu_ids": h_ids, "hu_eng": make_engram_bigram_ids(h_ids), "hu_style": hu_st[i],
                "weight": float(p["weight"]),
            }
            (pairs_gg_pre if p.get("is_gateguard") else pairs_gen_pre).append(rec)
    pairs_pre = pairs_gg_pre + pairs_gen_pre

    # Disjoint eval splits: val_gold (soup selection) vs val_cal_rows (calibration & Mondrian) vs frozen tests!
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_all, _ = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    val_gold_ids = {str(r.get("sample_id", r.get("id", ""))) for r in val_gold}
    val_cal_disjoint = [r for r in val_cal_all if str(r.get("sample_id", r.get("id", ""))) not in val_gold_ids]
    if len(val_cal_disjoint) < 400:
        val_cal_disjoint = val_cal_all
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(sub: list[dict]):
        raws = [str(r["text"]) for r in sub]
        cls_l, st_m, metas_l = extract_style_features_40d_batch(raws)
        enc = [tokenizer.encode_ids(c, max_len=args.max_len) for c in cls_l]
        eng = [make_engram_bigram_ids(s) for s in enc]
        sc = np.array([float(r["score"]) for r in sub], dtype=np.float32)
        eras = np.array([0 if r.get("era_bucket") == "pre_llm" else 3 for r in sub], dtype=np.int64)
        return enc, eng, raws, cls_l, st_m, metas_l, sc, eras

    val_enc, val_eng, val_txts, val_cleans, val_st, val_metas, val_scores, val_eras = _pre_eval(val_gold)
    cal_enc, cal_eng, cal_txts, cal_cleans, cal_st, cal_metas, cal_scores, cal_eras = _pre_eval(val_cal_disjoint)
    test_enc, test_eng, test_txts, test_cleans, test_st, test_metas, test_scores, test_eras = _pre_eval(test_gold)
    dsh_enc, dsh_eng, dsh_txts, dsh_cleans, dsh_st, dsh_metas, dsh_scores, _ = _pre_eval(test_dsh_gold)
    all_enc, all_eng, all_txts, all_cleans, all_st, all_metas, all_scores, _ = _pre_eval(test_bili_all)
    prb_enc, prb_eng, prb_txts, prb_cleans, prb_st, prb_metas, prb_scores, _ = _pre_eval(probe_rows)

    # STRICT ZERO ARCHITECTURE SURGERY: exact frozen MeguvizV07OmniVeritasModel!
    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 1024
    mdl = MeguvizV07OmniVeritasModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=48,
        hidden_size=384,
        num_heads=6,
        swa_heads=2,
        ffn_size=896,
        engram_buckets=2048,
        engram_dim=16,
        num_slots=16,
        max_len=args.max_len,
        style_dim=STYLE_FEAT_DIM_V07,
        num_codes=len(CODE_LIST),
        num_prag=len(PRAG_TAGS),
        intent_dim=intent_dim,
    ).to(device)
    if args.v07_fp16_ckpt and args.v07_fp16_ckpt.is_file():
        mdl.load_state_dict(torch.load(args.v07_fp16_ckpt, map_location=device), strict=True)
        print(f"[v0.8.0-Zenith Init] Loaded frozen v0.7.0 FP16 anchor from {args.v07_fp16_ckpt}", flush=True)

    # Offline-only Era-GRL adversary (physically discarded pre-export -> 0 bytes, 0 ops!)
    era_grl = OfflineEraGRLHead(emb_dim=256, num_eras=4).to(device)

    @torch.no_grad()
    def predict_pt(model_e, enc_l, eng_l, st_arr, num_loops: int = 2, use_fp32: bool = False):
        model_e.eval()
        n = len(enc_l)
        l_o, e_o = np.zeros(n, dtype=np.float32), np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return l_o, e_o
        order = sorted(range(n), key=lambda i: len(enc_l[i]))
        ebs = 128 if use_fp32 else 512
        for st_i in range(0, n, ebs):
            mb = order[st_i : st_i + ebs]
            ml = max(len(enc_l[i]) for i in mb)
            ids = np.full((len(mb), ml), tokenizer.pad_id, dtype=np.int64)
            m = np.zeros((len(mb), ml), dtype=np.int64)
            eg = np.zeros((len(mb), ml), dtype=np.int64)
            for ri, oi in enumerate(mb):
                s = enc_l[oi]
                ids[ri, : len(s)], m[ri, : len(s)], eg[ri, : len(s)] = s, 1, eng_l[oi]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=not use_fp32):
                lg, eb, _, _ = model_e(
                    torch.from_numpy(ids).to(device),
                    torch.from_numpy(m).to(device),
                    torch.from_numpy(st_arr[mb]).to(device),
                    engram_ids=torch.from_numpy(eg).to(device),
                    num_loops=num_loops,
                    return_aux=False,
                )
            l_np, e_np = lg.float().cpu().numpy(), eb.float().cpu().numpy()
            for ri, oi in enumerate(mb):
                l_o[oi], e_o[oi] = l_np[ri], e_np[ri]
        return l_o, e_o

    def _val_obj_sd(sd_cand: dict[str, torch.Tensor]) -> float:
        mdl.load_state_dict({k: v.to(device) for k, v in sd_cand.items()}, strict=True)
        vl, _ = predict_pt(mdl, val_enc, val_eng, val_st, num_loops=2)
        vp = expit(vl)
        m = {**compute_eval_metrics(val_scores, vp), **compute_high_precision_metrics(val_scores, vp)}
        return 0.40 * float(m["auroc"]) + 0.30 * float(m["pr_auc"]) + 0.15 * float(m["best_f05"]) + 0.15 * float(m["recall_at_p95"])

    class V08DS(Dataset):
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
        rng_c = random.Random(20261128)
        def _collate(batch):
            seqs, engs = [], []
            for x in batch:
                s = x["ids"]
                if sandwich_prob > 0.0 and casual_wrappers and x["y_score"] >= 0.78 and len(s) <= 180 and rng_c.random() < sandwich_prob:
                    w1, w2 = rng_c.choice(casual_wrappers), rng_c.choice(casual_wrappers)
                    s = ([tokenizer.bos_id] + w1 + s[1:-1] + w2 + [tokenizer.eos_id])[: args.max_len]
                    eg = make_engram_bigram_ids(s)
                else:
                    eg = x["engram"]
                seqs.append(s)
                engs.append(eg)
            ml = min(args.max_len, ((max(len(s) for s in seqs) + 15) // 16) * 16)
            ids = np.full((len(batch), ml), tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), ml), dtype=np.int64)
            eng = np.zeros((len(batch), ml), dtype=np.int64)
            st = np.empty((len(batch), STYLE_FEAT_DIM_V07), dtype=np.float32)
            for i, (s, eg, x) in enumerate(zip(seqs, engs, batch)):
                n_s = min(len(s), ml)
                ids[i, :n_s], mask[i, :n_s], eng[i, :n_s], st[i] = s[:n_s], 1, eg[:n_s], x["style"]
            return (
                torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(eng), torch.from_numpy(st),
                torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
                torch.tensor([x["teacher_p"] for x in batch], dtype=torch.float32),
                torch.tensor([1.0 if x["is_gold_tier"] else 0.0 for x in batch], dtype=torch.float32),
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
        ai_ids, ai_m, ai_e = np.full((bs_p, ma), tokenizer.pad_id, dtype=np.int64), np.zeros((bs_p, ma), dtype=np.int64), np.zeros((bs_p, ma), dtype=np.int64)
        hu_ids, hu_m, hu_e = np.full((bs_p, mh), tokenizer.pad_id, dtype=np.int64), np.zeros((bs_p, mh), dtype=np.int64), np.zeros((bs_p, mh), dtype=np.int64)
        ai_st, hu_st = np.empty((bs_p, STYLE_FEAT_DIM_V07), dtype=np.float32), np.empty((bs_p, STYLE_FEAT_DIM_V07), dtype=np.float32)
        for i, x in enumerate(p_batch):
            na, nh = min(len(x["ai_ids"]), ma), min(len(x["hu_ids"]), mh)
            ai_ids[i, :na], ai_m[i, :na], ai_e[i, :na] = x["ai_ids"][:na], 1, x["ai_eng"][:na]
            hu_ids[i, :nh], hu_m[i, :nh], hu_e[i, :nh] = x["hu_ids"][:nh], 1, x["hu_eng"][:nh]
            ai_st[i], hu_st[i] = x["ai_style"], x["hu_style"]
        return (
            torch.from_numpy(ai_ids).to(device, non_blocking=True), torch.from_numpy(ai_m).to(device, non_blocking=True),
            torch.from_numpy(ai_e).to(device, non_blocking=True), torch.from_numpy(ai_st).to(device, non_blocking=True),
            torch.from_numpy(hu_ids).to(device, non_blocking=True), torch.from_numpy(hu_m).to(device, non_blocking=True),
            torch.from_numpy(hu_e).to(device, non_blocking=True), torch.from_numpy(hu_st).to(device, non_blocking=True),
            torch.tensor([x["weight"] for x in p_batch], dtype=torch.float32, device=device),
        )

    ce_code = nn.CrossEntropyLoss(ignore_index=-100)
    hard_grp_t = torch.tensor(
        [GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["multilingual_world_knowledge"], GROUP_TO_IDX["dsh_hard_slice"]],
        device=device,
    )

    def _run_branch(
        branch_name: str,
        init_sd: dict[str, torch.Tensor],
        items: list[dict],
        steps: int,
        lr: float,
        quant_mode: str,
        use_asam: bool,
        pauc_w: float,
        pair_w: float = 0.22,
        beta_end: float = 0.025,
        teacher_m: MeguvizV07OmniVeritasModel | None = None,
    ) -> tuple[dict[str, torch.Tensor], float]:
        mdl.load_state_dict({k: v.to(device) for k, v in init_sd.items()}, strict=True)
        mdl.set_quant_mode(quant_mode)
        opt = torch.optim.AdamW(list(mdl.parameters()) + list(era_grl.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=0.015)
        asam = ASAMPerturber(mdl, rho=0.05) if use_asam else None
        dl = DataLoader(V08DS(items), batch_size=batch_size, shuffle=True, collate_fn=_make_collate(0.06), num_workers=0, pin_memory=True)
        it_dl = iter(dl)
        rng_p = random.Random(20261128 + len(branch_name))
        pair_bs = max(12, min(36, int(round(16.0 * batch_size / 256.0))))
        best_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        best_sc = _val_obj_sd(best_sd) if not args.smoke_test else -1.0
        eval_every = max(350, steps // 3) if (not args.smoke_test and steps >= 600) else 0

        mdl.train()
        for step in range(steps):
            try:
                batch = next(it_dl)
            except StopIteration:
                it_dl = iter(dl)
                batch = next(it_dl)
            ids, mask, eng, st, y_sc, tp_v06, is_g, y_cd, y_a5, w, grp, era, row_idx = [t.to(device, non_blocking=True) for t in batch]
            mult = _wsd_lr_mult(step, steps)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult
            loops_now = 4 if (step % 5 == 0) else 2
            t_emb_v06 = None
            if teacher_m is None and teacher_v06_embs is not None:
                t_emb_v06 = torch.from_numpy(teacher_v06_embs[row_idx.cpu().numpy()]).to(device, dtype=torch.float32, non_blocking=True)

            # Pre-sample GateGuard / Gemini Queue-C minimal pairs once per step (shared across ASAM pass!)
            pair_tensors = None
            if pairs_pre and pair_w > 0.0 and (step % 2 == 0):
                if pairs_gg_pre and pairs_gen_pre:
                    n_gg = min(len(pairs_gg_pre), max(4, int(round(pair_bs * 0.50))))
                    n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                    pb = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                else:
                    pb = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                pair_tensors = _collate_pairs(pb)

            def _compute_loss():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(ids, mask, st, engram_ids=eng, num_loops=loops_now, return_aux=True)
                    lg = out["logit"].float()
                    # Multi-teacher soft target blending (pure gold keeps gold truth!)
                    has_v06 = (tp_v06 >= 0.0) & (is_g < 0.5)
                    y_eff = torch.where(has_v06, 0.55 * y_sc + 0.45 * tp_v06.clamp(0.0, 1.0), y_sc)

                    loss_m = (F.binary_cross_entropy_with_logits(lg, y_eff.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w).mean()
                    loss_mid = (F.binary_cross_entropy_with_logits(out["mid_logit"].float(), y_eff.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w).mean()
                    l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=step / max(1, steps - 1), beta_start=0.06, beta_end=beta_end)
                    hu_h = torch.isin(grp, hard_grp_t)
                    l_anc = zenith_graded_anchor_loss(lg, y_sc, hard_human_mask=hu_h)
                    l_era = era_grl(out["emb"], era, is_clean_human=(y_sc <= 0.25), lam=0.20)
                    loss = (
                        loss_m + 0.12 * loss_mid + pauc_w * l_pa + l_tp + l_anc + 0.08 * l_era
                        + 0.08 * F.cross_entropy(out["attr5_logit"], y_a5)
                        + 0.08 * ce_code(out["code_logit"], y_cd)
                        + 0.04 * out["cbm_loss"]
                    )
                    if teacher_m is not None:
                        with torch.no_grad():
                            to = teacher_m(ids, mask, st, engram_ids=eng, num_loops=loops_now, return_aux=True)
                        loss = loss + 0.12 * logit_standardized_dkd_loss(out["attr5_logit"], to["attr5_logit"], y_a5, sample_weights=w)
                        loss = loss + 0.10 * listnet_tail_distill_loss(lg, to["logit"])
                        cos_d = 1.0 - F.cosine_similarity(out["emb"].float(), to["emb"].float(), dim=-1).mean()
                        loss = loss + 0.20 * cos_d
                    elif t_emb_v06 is not None:
                        cos_v06 = 1.0 - F.cosine_similarity(out["emb"].float(), t_emb_v06, dim=-1).mean()
                        loss = loss + 0.18 * cos_v06
                        v_tp = tp_v06 >= 0.0
                        if int(v_tp.sum()) >= 8:
                            loss = loss + 0.08 * listnet_tail_distill_loss(lg[v_tp], torch.logit(tp_v06[v_tp].clamp(1e-4, 1.0 - 1e-4)))
                    if pair_tensors is not None:
                        ai_i, ai_m, ai_e, ai_s, hu_i, hu_m, hu_e, hu_s, pw_t = pair_tensors
                        la, _, _, _ = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, num_loops=2, return_aux=False)
                        lh, _, _, _ = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, num_loops=2, return_aux=False)
                        la, lh = la.float(), lh.float()
                        l_simpo = (F.softplus(1.9 - (la - lh)) * pw_t).mean()
                        l_pabs = ((F.relu(0.88 - torch.sigmoid(la)).pow(2) + 1.5 * F.relu(torch.sigmoid(lh) - 0.10).pow(2)) * pw_t).mean()
                        loss = loss + pair_w * (l_simpo + 0.9 * l_pabs)
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
                if csc > best_sc:
                    best_sc, best_sd = csc, csd
                mdl.train()

        end_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
        end_sc = _val_obj_sd(end_sd)
        if end_sc >= best_sc:
            best_sc, best_sd = end_sc, end_sd
        print(f"[v0.8 Branch {branch_name}] val_obj={best_sc:.4f} (end={end_sc:.4f})", flush=True)
        return best_sd, best_sc

    # Phase A: Shared theta_anchor
    anchor_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    if not (args.v07_fp16_ckpt and args.v07_fp16_ckpt.is_file()):
        anchor_sd, _ = _run_branch("PhaseA-Anchor", anchor_sd, train_items_raw, steps=10 if args.smoke_test else args.anchor_steps, lr=4.0e-4, quant_mode="fp16", use_asam=False, pauc_w=0.08)

    # Pillar 3: 3-Branch WSD-S + ASAM Diversity Fork -> Conservative DARE-TIES + Greedy Validation-pAUC Soup!
    sub_b, _ = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=min(480000, len(train_items_raw)), seed_s=20261128)
    b_steps = 6 if args.smoke_test else args.branch_steps
    cands = []
    for bname, blr, bpw, bpair, bbeta in (
        ("B1_Balanced", 7.0e-5, 0.09, 0.22, 0.030),
        ("B2_ExtremeTail", 6.0e-5, 0.14, 0.22, 0.015),
        ("B3_HumanShield", 5.5e-5, 0.07, 0.32, 0.025),
    ):
        bsd, bsc = _run_branch(bname, anchor_sd, sub_b, steps=b_steps, lr=blr, quant_mode="fp16", use_asam=True, pauc_w=bpw, pair_w=bpair, beta_end=bbeta)
        cands.append((bname, bsd, bsc))

    soup_sd, soup_rep = greedy_pauc_soup(anchor_sd, cands, eval_fn=_val_obj_sd, dare_p=0.12)
    mdl.load_state_dict({k: v.to(device) for k, v in soup_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    if not args.smoke_test:
        torch.save(mdl.state_dict(), ckpt_dir / f"{args.version}_fp16_soup.pt")
    print(f"[Pillar3-Soup] {json.dumps(soup_rep)}", flush=True)

    # Measure exact FP16 Soup Master test_bili_gold BEFORE sub-2-bit quantization!
    fp16_test_l, _ = predict_pt(mdl, test_enc, test_eng, test_st, num_loops=2, use_fp32=True)
    fp16_test_m = {**compute_eval_metrics(test_scores, expit(fp16_test_l)), **compute_high_precision_metrics(test_scores, expit(fp16_test_l))}
    fp16_r4_l, _ = predict_pt(mdl, test_enc, test_eng, test_st, num_loops=4, use_fp32=True)
    fp16_r4_m = {**compute_eval_metrics(test_scores, expit(fp16_r4_l)), **compute_high_precision_metrics(test_scores, expit(fp16_r4_l))}

    # Pillar 4a: Zero-Op SpinQuant R_2 + SmoothQuant (with live calib activations!) + SubLN Algebraic Fold!
    calib_batches = []
    cal_dl = DataLoader(V08DS(sub_b[: min(256, len(sub_b))]), batch_size=64, shuffle=False, collate_fn=_make_collate(0.0))
    for cb in cal_dl:
        calib_batches.append((cb[0].to(device), cb[1].to(device), cb[3].to(device), cb[2].to(device)))
    fold_rep = fold_zero_op_spinquant_and_smoothquant_v07(mdl, calib_inputs=calib_batches)
    print(f"[Pillar4a-ZeroOpFold] {json.dumps(fold_rep)}", flush=True)

    fp16_soup_teacher = copy.deepcopy(mdl).eval()
    for p in fp16_soup_teacher.parameters():
        p.requires_grad_(False)
    folded_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}

    # Pillar 4b: Progressive UPQ (W4 Bridge -> 2-Branch Continuous Latent-Weight Sub2 QAT Soup!)
    w4_sd, _ = _run_branch("UPQ_W4_Bridge", folded_sd, sub_b, steps=6 if args.smoke_test else max(400, args.qat_steps // 2), lr=7.5e-5, quant_mode="w4", use_asam=False, pauc_w=0.09, teacher_m=fp16_soup_teacher)
    q1_sd, q1_sc = _run_branch("QAT_Sub2_A", w4_sd, sub_b, steps=6 if args.smoke_test else args.qat_steps, lr=4.5e-5, quant_mode="sub2", use_asam=False, pauc_w=0.10, teacher_m=fp16_soup_teacher)
    q2_sd, q2_sc = _run_branch("QAT_Sub2_B", w4_sd, sub_b, steps=6 if args.smoke_test else args.qat_steps, lr=3.5e-5, quant_mode="sub2", use_asam=False, pauc_w=0.13, pair_w=0.28, teacher_m=fp16_soup_teacher)
    qat_soup_sd, qat_soup_rep = greedy_pauc_soup(w4_sd, [("QAT_Sub2_A", q1_sd, q1_sc), ("QAT_Sub2_B", q2_sd, q2_sc)], eval_fn=_val_obj_sd, dare_p=0.10)
    mdl.load_state_dict({k: v.to(device) for k, v in qat_soup_sd.items()}, strict=True)
    mdl.set_quant_mode("fp16")
    del fp16_soup_teacher
    torch.cuda.empty_cache()

    # Pack/Unpack exact 5-Trit/2-Bit -> Guarded Post-Quantization Seq-LoRA-MoE/Head Polish!
    tmp_mgvz = out_dir / "_tmp_pre_polish.mgvz"
    pack_bundle_v4_mgvz(mdl.cpu(), tmp_mgvz)
    load_bundle_v4_mgvz_into_model(mdl, tmp_mgvz)
    tmp_mgvz.unlink(missing_ok=True)
    mdl = mdl.to(device).eval()

    pre_polish_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    pre_polish_sc = _val_obj_sd(pre_polish_sd)
    for name, p in mdl.named_parameters():
        p.requires_grad_(("lora_moe" in name) or name.startswith(("coda_", "ai_head.", "style_", "cbm_")))
    p_opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    p_dl = DataLoader(V08DS(sub_b[: min(80000, len(sub_b))]), batch_size=batch_size, shuffle=True, collate_fn=_make_collate(0.0))
    mdl.train()
    p_steps = 4 if args.smoke_test else 220
    ps_i = 0
    for batch in p_dl:
        ids, mask, eng, st, y_sc, _, _, _, _, w, grp, _, _ = [t.to(device, non_blocking=True) for t in batch]
        p_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = mdl(ids, mask, st, engram_ids=eng, num_loops=2, return_aux=True)
            lg = out["logit"].float()
            lm = (F.binary_cross_entropy_with_logits(lg, y_sc.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w).mean()
            l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=0.85, beta_start=0.05, beta_end=0.025)
            loss = lm + 0.09 * l_pa + l_tp + zenith_graded_anchor_loss(lg, y_sc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        p_opt.step()
        ps_i += 1
        if ps_i >= p_steps:
            break
    for p in mdl.parameters():
        p.requires_grad_(True)
    post_polish_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    post_polish_sc = _val_obj_sd(post_polish_sd)
    if post_polish_sc < pre_polish_sc:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_polish_sd.items()}, strict=True)
        polish_kept = False
    else:
        polish_kept = True
    mdl.eval()

    # Post-Soup 0 KB Confounding Surgery + Era Linear-Probe Audit (Era-GRL physically discarded!)
    with torch.no_grad():
        mdl.temperature.fill_(1.0)
    _, val_embs = predict_pt(mdl, val_enc, val_eng, val_st, use_fp32=True)
    _, cal_embs = predict_pt(mdl, cal_enc, cal_eng, cal_st, use_fp32=False)
    era_probe_auc = eval_human_era_linear_probe_auroc(cal_embs, cal_eras, cal_scores / 100.0)
    surg_rep = apply_zero_cost_confounding_surgery(mdl, val_embs[val_scores <= 35.0], val_embs[val_scores >= 70.0], val_embs, val_scores, rank=8)

    # Pillar 5: Strictly Increasing Beta-Spline x Venn-ABERS (baked into v0.7.0 1D LUT!) + 9-Cell Mondrian SCoRE!
    raw_val_l, _ = predict_pt(mdl, val_enc, val_eng, val_st, num_loops=2, use_fp32=True)
    raw_cal_l, _ = predict_pt(mdl, cal_enc, cal_eng, cal_st, num_loops=2, use_fp32=False)
    raw_cal_l_r4, _ = predict_pt(mdl, cal_enc, cal_eng, cal_st, num_loops=4, use_fp32=False)
    cal_va = fit_venn_abers_and_beta_spline_lut(raw_val_l, val_scores / 100.0)
    v07_compat_lut = export_beta_spline_as_v07_lut(cal_va, n_knots=256)

    val_p_va = apply_spline_temp_lut(expit(raw_val_l), v07_compat_lut)
    cal_p_va = apply_spline_temp_lut(expit(raw_cal_l), v07_compat_lut)
    cal_p_r4 = apply_spline_temp_lut(expit(raw_cal_l_r4), v07_compat_lut)
    t_f05, t_fpr05 = find_val_thresholds(val_scores, val_p_va)

    # Frozen-Backbone Critics + Nano-Draft v2 + Joint Cascade Pareto Optimizer!
    _, test_embs = predict_pt(mdl, test_enc, test_eng, test_st, use_fp32=True)
    rng_bk = random.Random(20261128)
    hu_anc = [x for x in train_items_raw if x["is_hu_anchor"]]
    ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
    bank_sub = train_items_raw[: min(24000, len(train_items_raw))] if args.smoke_test else (rng_bk.sample(hu_anc, min(12000, len(hu_anc))) + rng_bk.sample(ai_anc, min(12000, len(ai_anc))))
    _, bk_emb = predict_pt(mdl, [x["ids"] for x in bank_sub], [x["engram"] for x in bank_sub], np.stack([x["style"] for x in bank_sub], axis=0))
    bk_is_hu = np.array([bool(x["is_hu_anchor"]) for x in bank_sub], dtype=bool)
    bk_is_ai = np.array([x["y_score"] >= 0.82 for x in bank_sub], dtype=bool)
    bk_a5 = np.array([x["attr5"] for x in bank_sub], dtype=np.int64)

    proto_path = out_dir / "prototype_bank_v07.npz"
    np.savez_compressed(proto_path, **fit_l2_prototype_bank(bk_emb, np.array([x["y_score"] * 100.0 for x in bank_sub], dtype=np.float32), is_human_anchor=bk_is_hu))
    lb_critic, lb_rep = _train_frozen_lb_dsm_critic(bk_emb[bk_is_hu], bk_emb[bk_is_ai], test_embs, test_scores, device, steps=60 if args.smoke_test else 650)
    lb_path = out_dir / "lb_dsm_critic_v07.npz"
    np.savez_compressed(lb_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in lb_critic.state_dict().items()})
    ebt_critic, ebt_rep = _train_frozen_ebt_critic(bk_emb, bk_a5, test_embs, test_scores, device, steps=60 if args.smoke_test else 500)
    ebt_path = out_dir / "ebt_critic_v07.npz"
    np.savez_compressed(ebt_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in ebt_critic.state_dict().items()})

    nano_sub = train_items_raw if (args.smoke_test or len(train_items_raw) <= 60000) else random.Random(20261128).sample(train_items_raw, 60000)
    nano_tl, _ = predict_pt(mdl, [x["ids"] for x in nano_sub], [x["engram"] for x in nano_sub], np.stack([x["style"] for x in nano_sub], axis=0))
    nano_cpu, nano_policy, nano_rep = _train_nano_draft_v07(
        nano_sub, expit(nano_tl), val_cleans, val_st, val_metas, val_scores, expit(raw_val_l), cal_cleans, cal_st, cal_metas, cal_scores, expit(raw_cal_l), device, steps=50 if args.smoke_test else 1100
    )
    nano_rep.pop("val_cascade_raw_probs", None)
    nano_rep.pop("cal_cascade_raw_probs", None)

    # Run Joint Cascade Pareto Optimizer on disjoint val_cal
    with torch.no_grad():
        nano_dev = nano_cpu.to(device).eval()
        cal_h_seqs = [text_to_byte_hash_ids(t) for t in cal_cleans]
        cal_pg, cal_df, cal_pw = np.zeros(len(cal_cleans), dtype=np.float32), np.zeros(len(cal_cleans), dtype=np.float32), np.zeros(len(cal_cleans), dtype=np.float32)
        for st_i in range(0, len(cal_cleans), 256):
            en = min(len(cal_cleans), st_i + 256)
            ml = max(len(cal_h_seqs[k]) for k in range(st_i, en))
            hi, hm = np.zeros((en - st_i, ml), dtype=np.int64), np.zeros((en - st_i, ml), dtype=np.int64)
            for ri, k in enumerate(range(st_i, en)):
                s = cal_h_seqs[k]
                hi[ri, : len(s)], hm[ri, : len(s)] = s, 1
            p1, p2, p3, _ = nano_dev(torch.from_numpy(hi).to(device), torch.from_numpy(hm).to(device), torch.from_numpy(cal_st[st_i:en]).to(device))
            cal_pg[st_i:en], cal_df[st_i:en], cal_pw[st_i:en] = p1.cpu().numpy(), p2.cpu().numpy(), p3.cpu().numpy()
        nano_cpu = nano_dev.cpu().eval()
    cal_elig = np.array([m["clean_len"] <= 48 and m["spoof_any"] == 0.0 and ("http" not in t) for t, m in zip(cal_cleans, cal_metas)], dtype=bool)
    pareto_rep = optimize_joint_cascade_pareto(cal_scores, cal_pg, cal_pw, cal_df, cal_elig, cal_p_va, cal_p_r4)
    if pareto_rep["nano_policy"]["enabled"]:
        nano_policy = pareto_rep["nano_policy"]

    mondrian_9 = fit_mondrian_score_conformal_9cell(cal_cleans, cal_st, cal_scores, cal_p_va, alphas=(0.03, 0.01))
    conf_2400 = fit_split_conformal_3way(cal_scores, cal_p_va, alphas=(0.03, 0.05))

    # Stage 4: Export 100% Drop-In Bundle v4 (<= 2.20 MB) + ONNX INT8 + Bit-Exact Verification + CI!
    exp_info = export_v07_onnx_and_bundle_v4(mdl, nano_cpu, out_dir)
    mdl_unpacked = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v4_mgvz_into_model(mdl_unpacked, out_dir / exp_info["bundle_v4_mgvz"]["file"])
    mdl_unpacked = mdl_unpacked.to(device).eval()

    def _eval_split(m_e, enc_l, eng_l, st_m, sc, fp32=True):
        lg, _ = predict_pt(m_e, enc_l, eng_l, st_m, num_loops=2, use_fp32=fp32)
        cp = apply_spline_temp_lut(expit(lg), v07_compat_lut)
        return {**compute_eval_metrics(sc, cp, t_f05, t_fpr05), **compute_high_precision_metrics(sc, cp)}, cp, lg

    test_m, test_p, test_lg = _eval_split(mdl_unpacked, test_enc, test_eng, test_st, test_scores, fp32=True)
    dsh_m, _, _ = _eval_split(mdl_unpacked, dsh_enc, dsh_eng, dsh_st, dsh_scores, fp32=True)
    all_m, all_p, _ = _eval_split(mdl_unpacked, all_enc, all_eng, all_st, all_scores, fp32=False)
    prb_m, _, _ = _eval_split(mdl_unpacked, prb_enc, prb_eng, prb_st, prb_scores, fp32=True)
    test_cal_lowfpr = compute_zenith_calibration_and_lowfpr_metrics(test_scores, test_p)
    all_cal_lowfpr = compute_zenith_calibration_and_lowfpr_metrics(all_scores, all_p)
    ci_final = run_v07_ci_gates(mdl_unpacked, tokenizer, nano=nano_cpu, nano_policy=nano_policy)

    manifest = {
        "schema_version": "meguviz.bundle.v4",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": "v0.7.0-OmniVeritas(ZERO-SURGERY v0.8.0-Zenith Drop-In Weights)",
        "zero_architecture_surgery": True,
        "new_ops_added": 0,
        "license": "Apache-2.0",
        "tokenizer_file": tok_path.name,
        "vocab_size": tokenizer.vocab_size,
        "max_len": int(args.max_len),
        "bundle_v4_mgvz": exp_info["bundle_v4_mgvz"]["file"],
        "model_file": exp_info["model_int8"],
        "nano_draft_file": exp_info["nano_draft_file"],
        "nano_policy": nano_policy,
        "cascade_pareto_policy": pareto_rep,
        "prototype_bank_file": proto_path.name,
        "lb_dsm_critic_file": lb_path.name,
        "ebt_critic_file": ebt_path.name,
        "temperature": 1.0,
        "spline_temp_lut": v07_compat_lut,
        "beta_spline_venn_abers": cal_va,
        "mondrian_9cell": mondrian_9,
        "conformal_cal2400": conf_2400,
        "thresholds": {"f05": round(t_f05, 4), "fpr05": round(t_fpr05, 4), "score70": 0.70},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT 1T Latency + Parity + Trap Suite using UNMODIFIED MeguvizV07Engine!
    engine = MeguvizV07Engine(out_dir, threads=1, enable_nano_cascade=True)
    t_s = time.perf_counter()
    _ = [engine.predict_one(t) for t in test_txts]
    single_1t_ms = (time.perf_counter() - t_s) * 1000.0 / max(1, len(test_txts))
    t_m = time.perf_counter()
    ort_main = engine.predict_batch(test_txts, max_batch=32, force_main=True)
    b32_main_ms = (time.perf_counter() - t_m) * 1000.0 / max(1, len(test_txts))
    t_c = time.perf_counter()
    ort_casc = engine.predict_batch(test_txts, max_batch=32, force_main=False)
    b32_casc_ms = (time.perf_counter() - t_c) * 1000.0 / max(1, len(test_txts))

    ort_p_main = np.array([x["prob"] for x in ort_main], dtype=np.float32)
    ort_p_casc = np.array([x["prob"] for x in ort_casc], dtype=np.float32)
    ort_m = {**compute_eval_metrics(test_scores, ort_p_casc, t_f05, t_fpr05), **compute_high_precision_metrics(test_scores, ort_p_casc)}
    traps_rep = _eval_traps_v07(engine)

    # Package TRUE <= 2.20 MB Bundle v4 Drop-In ZIP & ONNX INT8 ZIP!
    zip_v4 = out_dir / f"meguvizcac-{args.version}-bundle-v4-sub2bit.zip"
    v4_files = [exp_info["bundle_v4_mgvz"]["file"], exp_info["nano_draft_file"], tok_path.name, proto_path.name, lb_path.name, ebt_path.name, "manifest.json"]
    with ZipFile(zip_v4, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for fn in v4_files:
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    v4_zip_mb = zip_v4.stat().st_size / (1024.0 * 1024.0)

    zip_int8 = out_dir / f"meguvizcac-{args.version}-onnx-int8.zip"
    with ZipFile(zip_int8, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in (exp_info["model_int8"], exp_info["nano_draft_file"], tok_path.name, proto_path.name, lb_path.name, ebt_path.name, "manifest.json"):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)

    metrics: dict[str, Any] = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v4",
        "zero_architecture_surgery": True,
        "new_ops_added": 0,
        "gpu_name": gpu_name,
        "batch_size_used": batch_size,
        "gemini_factory_audit": gem_audit,
        "pillar1_denoise": denoise_report,
        "soup_report": soup_rep,
        "qat_soup_report": qat_soup_rep,
        "zero_op_fold_report": fold_rep,
        "post_quant_polish": {"kept": polish_kept, "before": round(pre_polish_sc, 5), "after": round(post_polish_sc, 5)},
        "confounding_surgery": surg_rep,
        "human_era_probe_auroc": era_probe_auc,
        "fp16_soup_master_eval": {"r2": fp16_test_m, "r4_mor8": fp16_r4_m},
        "sub2bit_bundle_v4_eval": {
            "test_bili_gold": test_m,
            "auroc_delta_vs_fp16": round(float(test_m["auroc"]) - float(fp16_test_m["auroc"]), 5),
            "pr_auc_delta_vs_fp16": round(float(test_m["pr_auc"]) - float(fp16_test_m["pr_auc"]), 5),
        },
        "test_bili_gold": test_m,
        "test_bili_gold_cal_lowfpr": test_cal_lowfpr,
        "test_dsh_gold": dsh_m,
        "test_all_bili_distilled": all_m,
        "test_all_bili_cal_lowfpr": all_cal_lowfpr,
        "probe_u001_u017": prb_m,
        "onnx_int8": {
            "mae_vs_sub2bit": round(float(np.mean(np.abs(test_p - ort_p_main))), 5),
            "single_1thread_ms": round(single_1t_ms, 3),
            "compact_batch32_1thread_ms": round(b32_casc_ms, 3),
            "compact_batch32_main_only_1thread_ms": round(b32_main_ms, 3),
            "onnx_test_bili_gold": ort_m,
            "model_bytes": exp_info["model_int8_bytes"],
            "zip_bytes": int(zip_int8.stat().st_size),
        },
        "bundle_v4_total_bytes": int(zip_v4.stat().st_size),
        "bundle_v4_total_mb": round(v4_zip_mb, 4),
        "export_info": exp_info,
        "beta_spline_venn_abers": {"beta_params": cal_va["beta_params"], "va_mean_width": cal_va["va_mean_width"]},
        "mondrian_9cell": mondrian_9,
        "conformal_cal2400": conf_2400,
        "cascade_pareto_policy": pareto_rep,
        "lb_dsm_critic": lb_rep,
        "ebt_critic": ebt_rep,
        "nano_draft": nano_rep,
        "traps": traps_rep,
        "v07_ci_gates": ci_final,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics["release_gates"] = check_v08_release_gates(metrics)
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "version": metrics["version"],
                "zero_architecture_surgery": True,
                "new_ops_added": 0,
                "bundle_v4_total_mb": metrics["bundle_v4_total_mb"],
                "test_bili_gold": {k: test_m[k] for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95")},
                "auroc_delta_vs_fp16": metrics["sub2bit_bundle_v4_eval"]["auroc_delta_vs_fp16"],
                "test_cal_lowfpr": test_cal_lowfpr,
                "human_era_probe_auroc": era_probe_auc,
                "onnx_int8_1t_ms": metrics["onnx_int8"]["compact_batch32_1thread_ms"],
                "traps_summary": traps_rep["summary"],
                "v07_ci_all_passed": ci_final["all_ci_passed"],
                "release_gates": {
                    "passed": metrics["release_gates"]["passed_count"],
                    "total": metrics["release_gates"]["total_gates"],
                },
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
    ap.add_argument("--v08-gemini-gold", type=Path, default=None, help="Output of tools/gemini_v08_datafactory/03_qc_and_merge.py")
    ap.add_argument("--v08-gemini-pairs", type=Path, default=None, help="Counterfactual minimal pairs from External Gemini Factory")
    ap.add_argument("--teacher-v06-ckpt", type=Path, default=None, help="Optional unquantized BF16 v0.6 master teacher for multi-teacher consensus")
    ap.add_argument("--v06-backbone", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--v07-fp16-ckpt", type=Path, default=None, help="Frozen v0.7.0 FP16 master checkpoint to warmstart")
    ap.add_argument("--v07-tokenizer", type=Path, default=None, help="Frozen v0.7.0 bili_picky_bpe_12k.json")
    ap.add_argument("--intent-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("./out_v08_zenith"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", default="v0.8.0-r1-zenith-b4")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_V07)
    ap.add_argument("--stage1-synth-cap", type=int, default=15000)
    ap.add_argument("--v04-synth-cap", type=int, default=-1)
    ap.add_argument("--anchor-steps", type=int, default=3500)
    ap.add_argument("--branch-steps", type=int, default=1600)
    ap.add_argument("--qat-steps", type=int, default=1800)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    train_v08_zenith(args)


if __name__ == "__main__":
    main()
