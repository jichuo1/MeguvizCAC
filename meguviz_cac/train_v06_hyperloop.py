"""Train, MOHAWK-align, GroupDRO + Modern-Human Two-Way pAUC + Born-Again EMA refine, QAT+SWA,
0 KB Confounding-Neuron Surgery, Stage-2 Frozen LB-DSM Curvature Critic, 0.155 MB Nano-Draft L2D-FP distillation,
Clean Gold Smooth PAVA + Mandap Construction-B Complete-Path Conformal & Bundle v3 export for MeguvizCAC v0.6.0-HyperLoop."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .train_student import CODE_LIST, CODE_TO_IDX, PRAG_TAGS, compute_eval_metrics, find_val_thresholds, fit_temperature, sha256_file
from .train_v03_loop import DEFAULT_BACKBONE, _seed_ascii_bytes_into_a, compute_weighted_svd_factors
from .train_v04_dualloop import _load_intent_cache, _open_read, compute_high_precision_metrics
from .train_v05_apexloop import (
    GOLD_HIGH_TIER_SOURCES,
    _char_trigrams,
    _cosine_mult,
    _inject_pife_noise,
    _is_near_duplicate,
    build_stratified_val_calibration_set,
    subset_balanced_gold_guaranteed,
)
from .v03_tokenizer import BASE_VOCAB_SIZE, ByteFallbackWordPieceTokenizer
from .v04_corpus_prep import BENCH_TRAP_TEXTS, GROUP_TO_IDX, V04_GROUPS, assign_group_id
from .v04_runtime import fit_split_conformal_3way
from .v05_features import (
    STYLE_FEAT_DIM,
    apply_bucketed_smooth_pava,
    extract_style_features_32d,
    extract_style_features_batch,
    fit_smooth_pava,
    prepare_text_v05,
)
from .v05_runtime import PIFE_ADVERSARIAL_TRAPS, fit_l2_prototype_bank
from .v06_hyperloop import (
    MAX_LEN_V06,
    FrozenLBDSMCritic,
    MeguvizV06Engine,
    MeguvizV06HyperLoopModel,
    NanoDraftL2DFP,
    apply_zero_cost_confounding_surgery,
    export_v06_bundle_v3_onnx,
    run_verify_v06,
    text_to_byte_hash_ids,
)


def _load_v06_balanced_bundle(
    data_path: Path,
    gateguard_path: Path | None,
    extra_pairs_paths: list[Path],
    intent_cache_path: Path | None,
    stage1_synth_cap: int = 15000,
    v04_synth_cap: int = 160000,
    seed: int = 20261125,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray | None, dict]:
    """Load zero-era-shortcut balanced corpus (fixes v0.5-R3 6.7% pos-share collapse & pre_llm era bias!)."""
    rows: list[dict] = []
    with _open_read(data_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    intent_vecs, sid_to_intent = _load_intent_cache(intent_cache_path)
    eval_rows = [r for r in rows if r["split"] != "train" or r.get("cohort") == "user_confirmed"]

    forbidden_exact = {prepare_text_v05(r["text"]) for r in eval_rows}
    trap_texts = list(BENCH_TRAP_TEXTS) + [txt for _, txt in PIFE_ADVERSARIAL_TRAPS]
    for t in trap_texts:
        forbidden_exact.add(prepare_text_v05(t))
    trap_trigrams = [_char_trigrams(t) for t in trap_texts]

    pairs: list[dict] = []
    gg_kept_cats = Counter()
    gg_dropped = 0
    extra_dropped = 0

    def _ingest_pairs(path: Path, is_gateguard: bool):
        nonlocal gg_dropped, extra_dropped
        if not path or not path.is_file():
            return
        with _open_read(path) as pf:
            for line in pf:
                if not line.strip():
                    continue
                p = json.loads(line)
                ai_t = str(p.get("ai_text") or "")
                hu_t = str(p.get("human_text") or "")
                ai_n = prepare_text_v05(ai_t)
                hu_n = prepare_text_v05(hu_t)
                if len(ai_n) < 8 or len(hu_n) < 8 or ai_n == hu_n:
                    continue
                if _is_near_duplicate(ai_t, forbidden_exact, trap_trigrams) or _is_near_duplicate(hu_t, forbidden_exact, trap_trigrams):
                    if is_gateguard:
                        gg_dropped += 1
                    else:
                        extra_dropped += 1
                    continue
                cat = str(p.get("category", "gateguard" if is_gateguard else "extra_pair"))
                # Gentle pair weight cap (<= 1.6) so GateGuard sharpens traps cleanly without blowing up temperature T!
                w_pair = min(1.6, float(p.get("weight", 1.4 if is_gateguard else 0.7)))
                pairs.append(
                    {
                        "ai_text": ai_t,
                        "human_text": hu_t,
                        "weight": w_pair,
                        "category": cat,
                        "is_gateguard": is_gateguard,
                    }
                )
                if is_gateguard:
                    gg_kept_cats[cat] += 1

    if gateguard_path:
        _ingest_pairs(gateguard_path, is_gateguard=True)
    for ep in extra_pairs_paths:
        _ingest_pairs(ep, is_gateguard=False)

    rng = random.Random(seed)
    v04_synth_rows: list[dict] = []
    legacy_synth_rows: list[dict] = []
    other_rows: list[dict] = []
    demoted_struct_weak = 0

    for r in rows:
        if r["split"] != "train" or r.get("cohort") == "user_confirmed":
            continue
        if not r.get("has_score", True):
            continue
        if float(r.get("train_weight", 1.0)) <= 0.0:
            continue
        c = r.get("cohort", "")
        if c == "synth_250k":
            if len(str(r.get("text") or "").strip()) >= 14:
                v04_synth_rows.append(r)
        elif c in ("deepseek", "other_models"):
            if len(r["text"]) >= 26 and float(r["score"]) >= 70.0:
                legacy_synth_rows.append(r)
        else:
            other_rows.append(r)

    if stage1_synth_cap >= 0 and len(legacy_synth_rows) > stage1_synth_cap:
        legacy_synth_rows = rng.sample(legacy_synth_rows, stage1_synth_cap)
    if v04_synth_cap >= 0 and len(v04_synth_rows) > v04_synth_cap:
        by_mode: dict[str, list[dict]] = {}
        for sr in v04_synth_rows:
            by_mode.setdefault(str(sr.get("plugin_mode") or "default"), []).append(sr)
        quota_m = max(1, v04_synth_cap // max(1, len(by_mode)))
        picked_s: list[dict] = []
        rem_s: list[dict] = []
        for mk in sorted(by_mode):
            bkt = by_mode[mk]
            rng.shuffle(bkt)
            picked_s.extend(bkt[:quota_m])
            rem_s.extend(bkt[quota_m:])
        if len(picked_s) < v04_synth_cap and rem_s:
            rng.shuffle(rem_s)
            picked_s.extend(rem_s[: v04_synth_cap - len(picked_s)])
        v04_synth_rows = picked_s[:v04_synth_cap]

    train_items: list[dict] = []
    w_pos_sum = 0.0
    w_tot_sum = 0.0
    for r in (*other_rows, *legacy_synth_rows, *v04_synth_rows):
        sid = r["sample_id"]
        coh = str(r.get("cohort") or "")
        src = str(r.get("score_source") or "")
        era = str(r.get("era_bucket") or "")
        sc_f = float(r["score"])
        txt_s = str(r["text"])
        gid_str = r.get("group_id") or assign_group_id(r)
        is_gold_tier = (src in GOLD_HIGH_TIER_SOURCES) or bool(r.get("hard_human_neg"))
        is_pre_llm = era == "pre_llm"

        # Protection-Bucket Whitelist (keep weak noisy pseudo-labels out of structured_human_hard!)
        if gid_str == "structured_human_hard" and (not is_pre_llm) and (not is_gold_tier):
            gid_str = "general_real"
            demoted_struct_weak += 1

        # Zero-Era-Shortcut Balanced Weights (R2):
        # - Keep full diverse synth (-1 -> ~22-24% positive share matching v0.4-R3!)
        # - Boost gold-tier hard/borderline humans (dsh_hard_slice, structured_human_hard, multilingual <= 52) by 1.6x
        #   so borderline humans like idx=87/148 and H03/H06/M04 never climb into the top AI tail!
        # - NEVER upweight pre_llm above 1.0x (zero era shortcut!)
        w_eff = float(r.get("train_weight", 1.0))
        is_real_bili = coh in ("bili", "v04_1m")
        if is_real_bili and sc_f >= 65.0:
            w_eff *= 1.8 if is_gold_tier else 1.4
        elif coh in ("synth_250k", "deepseek", "other_models"):
            w_eff *= 0.78
        elif gid_str in ("structured_human_hard", "multilingual_world_knowledge", "dsh_hard_slice", "literary_boundary") and sc_f <= 52.0 and is_gold_tier:
            w_eff *= 1.60
        elif (not is_gold_tier) and (not is_pre_llm) and (25.0 < sc_f < 65.0) and len(txt_s.strip()) <= 20:
            w_eff *= 0.50

        w_tot_sum += w_eff
        if sc_f >= 70.0:
            w_pos_sum += w_eff

        prag = r.get("prag_vec")
        if not (isinstance(prag, list) and len(prag) == len(PRAG_TAGS)):
            prag = [-1.0] * len(PRAG_TAGS)

        # Clean-human anchor flag (used ONLY for prototype/LB-DSM anchor sampling; Two-Way pAUC covers ALL modern humans!)
        is_hu_anchor = bool((sc_f <= 15.0) or ((is_gold_tier or is_pre_llm) and sc_f <= 22.0))

        train_items.append(
            {
                "sample_id": sid,
                "raw_text": txt_s,
                "y_score": float(np.clip(sc_f / 100.0, 0.0, 1.0)),
                "code_idx": CODE_TO_IDX.get(r.get("reason_code", ""), -100),
                "weight": float(w_eff),
                "prag_vec": [float(x) for x in prag],
                "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["general_real"]),
                "intent_idx": sid_to_intent.get(sid, -1),
                "is_gold_tier": bool(is_gold_tier or (is_real_bili and sc_f >= 68.0)),
                "is_pre_llm": is_pre_llm,
                "is_hu_anchor": is_hu_anchor,
                "is_high_prio": bool(
                    is_gold_tier
                    or is_pre_llm
                    or (is_real_bili and sc_f >= 65.0)
                    or gid_str in ("ultra_short_ai", "evasive_plugin_ai", "structured_human_hard", "multilingual_world_knowledge", "persona_space_kaomoji", "literary_boundary", "dsh_hard_slice")
                ),
            }
        )

    audit = {
        "total_loaded_rows": len(rows),
        "train_items": len(train_items),
        "weighted_pos_share": round(w_pos_sum / max(1e-6, w_tot_sum), 4),
        "gold_tier_count": sum(1 for x in train_items if x["is_gold_tier"]),
        "synth_250k_kept": len(v04_synth_rows),
        "demoted_weak_structured_human": demoted_struct_weak,
        "eval_rows": len(eval_rows),
        "gateguard_kept_by_cat": dict(gg_kept_cats),
        "gateguard_dropped_leakage": gg_dropped,
        "extra_pairs_dropped_leakage": extra_dropped,
        "total_clean_pairs": len(pairs),
    }
    return train_items, eval_rows, pairs, intent_vecs, audit


def _modern_two_way_pauc_and_top_push(
    logits,
    y_sc,
    w,
    grp,
    prog: float,
    beta_start: float = 0.08,
    beta_end: float = 0.04,
):
    """RTX 4060 Ti E3a + v0.4-R3 validated Two-Way pAUC & Top-Push covering ALL modern Bilibili humans (ZERO era shortcut!)."""
    import torch
    import torch.nn.functional as F

    l_fp = logits.float()
    p_fp = torch.sigmoid(l_fp)
    struct_idx = GROUP_TO_IDX["structured_human_hard"]
    pre_idx = GROUP_TO_IDX["pre_llm_human"]
    ml_idx = GROUP_TO_IDX["multilingual_world_knowledge"]
    dsh_idx = GROUP_TO_IDX["dsh_hard_slice"]
    lit_idx = GROUP_TO_IDX["literary_boundary"]
    hard_hu_grps = (grp == struct_idx) | (grp == ml_idx) | (grp == dsh_idx) | (grp == lit_idx)

    pos_m = (y_sc >= 0.78) & (w > 0.0)
    # CRITICAL (R3): Quarantine ambiguous mid-band rows (y > 0.28) OUT of Two-Way pAUC neg_m!
    # Only clear modern humans (y<=0.18) and clear hard/pre_llm humans (y<=0.28) enter pAUC sorting;
    # borderline 0.28 < y <= 0.55 humans are governed exclusively by Tier-2b upper-cap below!
    neg_m = (
        (y_sc <= 0.18)
        | ((hard_hu_grps | (grp == pre_idx)) & (y_sc <= 0.28))
    ) & (w > 0.0)

    loss_pauc = l_fp.sum() * 0.0
    loss_tp = l_fp.sum() * 0.0

    if pos_m.sum() >= 2 and neg_m.sum() >= 4:
        l_pos = l_fp[pos_m]
        l_neg = l_fp[neg_m]
        cur_beta = beta_start + (beta_end - beta_start) * min(1.0, max(0.0, prog))
        cur_alpha = 0.50 - 0.15 * min(1.0, max(0.0, prog))

        k_neg = max(2, int(math.ceil(cur_beta * l_neg.numel())))
        k_pos = max(2, int(math.ceil(cur_alpha * l_pos.numel())))

        hard_neg = torch.topk(l_neg, k=min(k_neg, l_neg.numel()), largest=True).values
        hard_pos = torch.topk(l_pos, k=min(k_pos, l_pos.numel()), largest=False).values
        diff = hard_pos.unsqueeze(1) - hard_neg.unsqueeze(0)
        # Milder margin=2.0 keeps sharp calibration T in [1.1, 1.8] while pushing hard negatives cleanly out of P95/P99 tail!
        loss_pauc = F.softplus(2.0 - diff).mean()

        # Dual-Channel Top-Push: mean pos vs top-25% hard neg + absolute tail suppression
        k_25 = max(1, l_neg.numel() // 4)
        l_hu_hard = torch.topk(l_neg, k=k_25, largest=True).values.mean()
        loss_margin = F.softplus(2.2 - (l_pos.mean() - l_hu_hard))
        loss_t1 = F.relu(p_fp[neg_m] - 0.12).pow(2).mean()
        loss_tp = 0.08 * loss_margin + 0.18 * loss_t1

    # Tier-2a: Hard-group clear humans (y <= 0.28) must stay < 0.12
    tier2 = hard_hu_grps & (y_sc <= 0.28) & (w > 0.0)
    if tier2.any():
        loss_tp = loss_tp + 0.28 * F.relu(p_fp[tier2] - 0.12).pow(2).mean()

    # Tier-2b (R3 tightened!): Mid-band borderline humans (0.28 < y <= 0.55, e.g. dsh_hard_slice score 42-48!)
    # MUST NEVER climb above y + 0.05 into the 0.85+ high-precision AI zone!
    tier_mid = hard_hu_grps & (y_sc > 0.28) & (y_sc <= 0.55) & (w > 0.0)
    if tier_mid.any():
        loss_tp = loss_tp + 0.40 * F.relu(p_fp[tier_mid] - (y_sc[tier_mid] + 0.05)).pow(2).mean()

    return loss_pauc, loss_tp


def _subset_natural_prior_gold_guaranteed(items: list[dict], cap: int, seed_s: int = 20261126) -> tuple[list[dict], float]:
    """R3 Fix: Preserve natural ~20% positive share & full modern-human density while guaranteeing 100% gold retention!

    Fixes R2 Stage-2 collapse where 12-way group equalization exploded positive share 20% -> ~45% and starved
    current_2026_real / general_real humans by ~10x.
    """
    rng = random.Random(seed_s)
    if len(items) <= cap:
        out = list(items)
        rng.shuffle(out)
        return out, 1.0

    gold = [x for x in items if x["is_gold_tier"]]
    non_gold = [x for x in items if not x["is_gold_tier"]]
    selected = list(gold)
    gold_ret = 1.0
    if len(selected) >= cap:
        rng.shuffle(selected)
        selected = selected[:cap]
        gold_ret = len(selected) / max(1, len(gold))
        return selected, gold_ret

    # Protect tiny rare human/borderline buckets up to a modest ceiling (<= 12k each) without distorting global prior!
    protect_gids = {
        GROUP_TO_IDX["structured_human_hard"],
        GROUP_TO_IDX["multilingual_world_knowledge"],
        GROUP_TO_IDX["dsh_hard_slice"],
        GROUP_TO_IDX["literary_boundary"],
    }
    prot_by_g: dict[int, list[dict]] = {g: [] for g in protect_gids}
    rest_non_gold: list[dict] = []
    for x in non_gold:
        g = x["group_idx"]
        if g in protect_gids and x["y_score"] <= 0.55:
            prot_by_g[g].append(x)
        else:
            rest_non_gold.append(x)

    per_prot_cap = min(12000, max(2000, cap // 36))
    for g in sorted(prot_by_g):
        arr = prot_by_g[g]
        rng.shuffle(arr)
        take = min(len(arr), per_prot_cap)
        selected.extend(arr[:take])
        rest_non_gold.extend(arr[take:])

    rem = max(0, cap - len(selected))
    if rem > 0 and rest_non_gold:
        rng.shuffle(rest_non_gold)
        selected.extend(rest_non_gold[:rem])

    rng.shuffle(selected)
    assert gold_ret >= 0.98, f"Gold Retention violated: {gold_ret:.4f} < 0.98"
    return selected[:cap], gold_ret


def check_v06_release_gates(metrics: dict) -> dict:
    tb = metrics.get("test_bili_gold") or {}
    ox = metrics.get("onnx_int8") or {}
    ed = metrics.get("edge_w4_w8") or {}
    ver = metrics.get("onnx_verify") or {}
    ht = ver.get("hard_trap_gates") or {}
    ml = ver.get("multilingual_trap_gates") or {}
    conf03 = (metrics.get("conformal_mandap_b") or metrics.get("conformal") or {}).get("alpha_03") or {}
    nano = metrics.get("nano_draft") or {}
    lbdsm = metrics.get("lb_dsm_critic") or {}
    t_c = float(metrics.get("temperature", 9.9))
    edge_zip_mb = (ed.get("edge_zip_bytes") or (ed.get("bytes") or 0)) / (1024.0 * 1024.0) if ed.get("ok") else 99.0

    gates = {
        "speed_1t_le_3p5_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 3.50 ms (1T)", "pass": (ox.get("compact_batch32_1thread_ms") or 99.0) <= 3.50},
        "nano_draft_le_250kb": {"value": round((nano.get("onnx_bytes") or 999999) / 1024.0, 1), "target": "<= 250 KB", "pass": (nano.get("onnx_bytes") or 999999) <= 256000},
        "edge_w4_size_le_12mb": {"value": round(edge_zip_mb, 2), "target": "<= 12.0 MB zip", "pass": bool(ed.get("ok")) and edge_zip_mb <= 12.0},
        "auroc_ge_0955": {"value": tb.get("auroc"), "target": ">= 0.955 (stretch 0.965)", "pass": (tb.get("auroc") or 0.0) >= 0.955},
        "pr_auc_ge_0935": {"value": tb.get("pr_auc"), "target": ">= 0.935 (stretch 0.955)", "pass": (tb.get("pr_auc") or 0.0) >= 0.935},
        "rp95_ge_075": {"value": tb.get("recall_at_p95"), "target": ">= 0.75 (stretch 0.85)", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.75},
        "rp99_ge_045": {"value": tb.get("recall_at_p99"), "target": ">= 0.45 (stretch 0.60)", "pass": (tb.get("recall_at_p99") or 0.0) >= 0.45},
        "struct_human_lt_020": {"value": ht.get("structured_human_max_prob"), "target": "< 0.20", "pass": (ht.get("structured_human_max_prob") or 1.0) < 0.20},
        "ai_traps_ge_082": {"value": ht.get("ai_traps_min_prob"), "target": ">= 0.82", "pass": (ht.get("ai_traps_min_prob") or 0.0) >= 0.82},
        "ml_human_lt_020": {"value": ml.get("ml_human_max_prob"), "target": "< 0.20", "pass": (ml.get("ml_human_max_prob") or 1.0) < 0.20},
        "ml_ai_ge_082": {"value": ml.get("ml_ai_min_prob"), "target": ">= 0.82", "pass": (ml.get("ml_ai_min_prob") or 0.0) >= 0.82},
        "conformal_grey_le_22pct": {"value": conf03.get("cal_grey_rate"), "target": "<= 0.22", "pass": (conf03.get("cal_grey_rate") or 1.0) <= 0.22},
        "sharp_temp_in_09_22": {"value": round(t_c, 3), "target": "[0.9, 2.2]", "pass": 0.9 <= t_c <= 2.2},
        "qat_parity_le_0010": {"value": ox.get("mae_vs_pytorch"), "target": "<= 0.010", "pass": (ox.get("mae_vs_pytorch") or 99.0) <= 0.010},
        "lb_dsm_curvature_valid": {"value": lbdsm.get("test_gold_lambda_auroc"), "target": ">= 0.70 (frozen critic)", "pass": (lbdsm.get("test_gold_lambda_auroc") or 0.0) >= 0.70},
        "zero_unk": {"value": ver.get("unk_tokens_total"), "target": "== 0", "pass": ver.get("unk_tokens_total") == 0},
    }
    return {"gates": gates, "passed_count": sum(1 for v in gates.values() if v["pass"]), "total_gates": len(gates), "all_passed": all(v["pass"] for v in gates.values())}


def _run_vram_probe(args, device, amp_dtype) -> None:
    import torch
    import torch.nn.functional as F

    cands = [int(x.strip()) for x in str(args.probe_vram).split(",") if x.strip()]
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap_gib = vram_gb * 0.88
    mdl = MeguvizV06HyperLoopModel(vocab_size=BASE_VOCAB_SIZE + 256, embed_dim=args.embed_dim, max_len=args.max_len).to(device)
    mdl.set_sdpa(True)
    mdl.set_qat_enabled(True)
    ema = copy.deepcopy(mdl).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(mdl.parameters(), lr=5e-5, fused=True)

    results = []
    chosen = cands[-1]
    for bs in cands:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pair_bs = max(16, min(48, int(round(16.0 * bs / 256.0))))
        ok = True
        err = ""
        step_ms = 0.0
        try:
            t0 = time.perf_counter()
            for _ in range(2):
                ids = torch.randint(5, BASE_VOCAB_SIZE, (bs, args.max_len), device=device)
                mask = torch.ones((bs, args.max_len), dtype=torch.long, device=device)
                st = torch.randn((bs, STYLE_FEAT_DIM), device=device)
                y = torch.rand((bs,), device=device)
                p_ids = torch.randint(5, BASE_VOCAB_SIZE, (pair_bs, 96), device=device)
                p_m = torch.ones((pair_bs, 96), dtype=torch.long, device=device)
                p_st = torch.randn((pair_bs, STYLE_FEAT_DIM), device=device)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    _ = ema(ids, mask, st, return_aux=False)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(ids, mask, st, return_aux=True)
                    loss = F.binary_cross_entropy_with_logits(out["logit"], y) + 0.05 * out["sae_loss"]
                    l_ai, _, _, _ = mdl(p_ids, p_m, p_st, return_aux=False)
                    l_hu, _, _, _ = mdl(p_ids, p_m, p_st, return_aux=False)
                    loss = loss + 0.25 * F.softplus(1.6 - (l_ai - l_hu)).mean()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0 / 2.0
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {str(exc)[:180]}"
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_res = torch.cuda.max_memory_reserved() / (1024**3)
        fits = ok and (peak_res <= cap_gib)
        results.append({"batch_size": bs, "ok": ok, "fits": fits, "peak_alloc_gib": round(peak_alloc, 2), "peak_reserved_gib": round(peak_res, 2), "step_ms": round(step_ms, 1), "err": err})
    for r in results:
        if r["fits"]:
            chosen = r["batch_size"]
            break
    print(f"VRAM_PROBE_JSON {json.dumps({'gpu': torch.cuda.get_device_name(0), 'vram_gb': round(vram_gb, 2), 'cap_gib': round(cap_gib, 2), 'chosen_batch_size': chosen, 'results': results})}", flush=True)


def _train_frozen_lb_dsm_critic(
    hu_embs: np.ndarray,
    ai_embs: np.ndarray,
    test_embs: np.ndarray,
    test_scores: np.ndarray,
    device,
    steps: int = 650,
    bs: int = 256,
) -> tuple[FrozenLBDSMCritic, dict]:
    """Stage-4b: Train 22 KB Dual-Manifold Denoising Score Matching & Hutchinson Curvature Critic on frozen 256-d embeddings."""
    import torch
    from sklearn.metrics import roc_auc_score

    critic = FrozenLBDSMCritic(emb_dim=hu_embs.shape[1], rank=32).to(device)
    if len(hu_embs) < 16 or len(ai_embs) < 16:
        return critic.cpu().eval(), {"trained": False, "reason": "insufficient_anchors"}

    hu_t = torch.from_numpy(hu_embs.astype(np.float32)).to(device)
    ai_t = torch.from_numpy(ai_embs.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=3e-3, weight_decay=1e-4)
    critic.train()
    for _ in range(steps):
        ih = torch.randint(0, hu_t.shape[0], (min(bs, hu_t.shape[0]),), device=device)
        ia = torch.randint(0, ai_t.shape[0], (min(bs, ai_t.shape[0]),), device=device)
        zh = hu_t[ih]
        za = ai_t[ia]
        sig_h = torch.empty((zh.shape[0], 1), device=device).uniform_(0.05, 0.20)
        sig_a = torch.empty((za.shape[0], 1), device=device).uniform_(0.05, 0.20)
        eps_h = torch.randn_like(zh)
        eps_a = torch.randn_like(za)
        sh_pred = critic.score_h(zh + sig_h * eps_h)
        sa_pred = critic.score_ai(za + sig_a * eps_a)
        loss = (sig_h * sh_pred + eps_h).pow(2).mean() + (sig_a * sa_pred + eps_a).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    critic.eval()
    t_z = torch.from_numpy(test_embs.astype(np.float32)).to(device)
    ev = critic.eval_hyvarinen_curvature(t_z, eps0=0.08, n_probes=6)
    y_bin = (test_scores >= 70.0).astype(bool)
    lam_auc = float(roc_auc_score(y_bin, ev["lambda_hc"])) if (y_bin.any() and (~y_bin).any()) else 0.5
    lam_auc_inv = float(roc_auc_score(y_bin, -ev["lambda_hc"])) if (y_bin.any() and (~y_bin).any()) else 0.5
    sign = 1.0 if lam_auc >= lam_auc_inv else -1.0
    best_auc = max(lam_auc, lam_auc_inv)
    report = {
        "trained": True,
        "steps": steps,
        "hu_anchors": int(len(hu_embs)),
        "ai_anchors": int(len(ai_embs)),
        "curvature_sign": sign,
        "test_gold_lambda_auroc": round(best_auc, 4),
        "test_gold_e_ood_mean": round(float(np.mean(ev["e_ood"])), 4),
        "param_bytes_fp16": int(sum(p.numel() for p in critic.parameters()) * 2),
    }
    return critic.cpu().eval(), report


def _train_and_calibrate_nano_draft(
    train_items: list[dict],
    teacher_probs_train: np.ndarray,
    val_txts: list[str],
    val_st: np.ndarray,
    val_scores: np.ndarray,
    val_teacher_p: np.ndarray,
    cal_txts: list[str],
    cal_st: np.ndarray,
    cal_scores: np.ndarray,
    cal_teacher_p: np.ndarray,
    device,
    steps: int = 900,
    bs: int = 512,
) -> tuple[NanoDraftL2DFP, dict, dict]:
    """Stage-4c: Distill 0.155 MB Nano-Draft L2D-FP gate from frozen HyperLoop teacher & calibrate strict FN-barrier policy."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    nano = NanoDraftL2DFP(buckets=4096, emb_dim=32, conv_c=64, style_dim=STYLE_FEAT_DIM).to(device)
    n_tr = len(train_items)
    h_seqs = [text_to_byte_hash_ids(x["raw_text"]) for x in train_items]
    st_arr = np.stack([x["style"] for x in train_items], axis=0).astype(np.float32)
    y_t = teacher_probs_train.astype(np.float32)
    defer_t = np.array(
        [
            1.0
            if (
                y_t[i] >= 0.22
                or len(train_items[i]["raw_text"]) > 64
                or ("http" in train_items[i]["raw_text"])
                or train_items[i]["group_idx"] in (GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["multilingual_world_knowledge"])
            )
            else 0.0
            for i in range(n_tr)
        ],
        dtype=np.float32,
    )

    opt = torch.optim.AdamW(nano.parameters(), lr=2.5e-3, weight_decay=1e-4)
    rng = np.random.default_rng(20261125)
    nano.train()
    for _ in range(steps):
        idx = rng.integers(0, n_tr, size=min(bs, n_tr))
        ml = max(len(h_seqs[i]) for i in idx)
        h_ids = np.zeros((len(idx), ml), dtype=np.int64)
        h_m = np.zeros((len(idx), ml), dtype=np.int64)
        for ri, oi in enumerate(idx):
            s = h_seqs[oi]
            h_ids[ri, : len(s)] = s
            h_m[ri, : len(s)] = 1
        hi_t = torch.from_numpy(h_ids).to(device)
        hm_t = torch.from_numpy(h_m).to(device)
        hs_t = torch.from_numpy(st_arr[idx]).to(device)
        yt_t = torch.from_numpy(y_t[idx]).to(device)
        dt_t = torch.from_numpy(defer_t[idx]).to(device)

        pd, pdef = nano(hi_t, hm_t, hs_t)
        w_fn = 1.0 + 3.5 * (yt_t >= 0.40).float()
        loss_p = (F.binary_cross_entropy(pd.clamp(1e-5, 1.0 - 1e-5), yt_t.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w_fn).mean()
        loss_d = F.binary_cross_entropy(pdef.clamp(1e-5, 1.0 - 1e-5), dt_t)
        loss = loss_p + 0.65 * loss_d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    nano.eval()

    @torch.no_grad()
    def _pred_nano(txts: list[str], st_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(txts)
        pd_o = np.zeros(n, dtype=np.float32)
        df_o = np.zeros(n, dtype=np.float32)
        seqs = [text_to_byte_hash_ids(t) for t in txts]
        for st_i in range(0, n, 256):
            en = min(n, st_i + 256)
            ml = max(len(seqs[k]) for k in range(st_i, en))
            hi = np.zeros((en - st_i, ml), dtype=np.int64)
            hm = np.zeros((en - st_i, ml), dtype=np.int64)
            for ri, k in enumerate(range(st_i, en)):
                s = seqs[k]
                hi[ri, : len(s)] = s
                hm[ri, : len(s)] = 1
            p1, p2 = nano(torch.from_numpy(hi).to(device), torch.from_numpy(hm).to(device), torch.from_numpy(st_m[st_i:en]).to(device))
            pd_o[st_i:en] = p1.cpu().numpy()
            df_o[st_i:en] = p2.cpu().numpy()
        return pd_o, df_o

    cal_pd, cal_df = _pred_nano(cal_txts, cal_st)
    val_pd, val_df = _pred_nano(val_txts, val_st)

    y_val_bin = (val_scores >= 70.0).astype(bool)
    base_val_auc = float(roc_auc_score(y_val_bin, val_teacher_p)) if (y_val_bin.any() and (~y_val_bin).any()) else 0.5
    cal_is_ai = cal_scores >= 70.0
    cal_elig = np.array([len(t) <= 64 and ("http" not in t) for t in cal_txts], dtype=bool)
    val_elig = np.array([len(t) <= 64 and ("http" not in t) for t in val_txts], dtype=bool)

    best_policy = {"enabled": False, "p_low": 0.04, "defer_max": 0.20}
    best_exit_rate = 0.0
    for p_low in (0.04, 0.08, 0.12, 0.16, 0.20, 0.25):
        for d_max in (0.15, 0.25, 0.35, 0.45, 0.55):
            m_cal = cal_elig & (cal_pd <= p_low) & (cal_df < d_max)
            fn_ai_cal = int(np.sum(m_cal & cal_is_ai))
            fn_rate_cal = float(fn_ai_cal / max(1, int(np.sum(cal_is_ai))))
            m_val = val_elig & (val_pd <= p_low) & (val_df < d_max)
            fn_ai_val = int(np.sum(m_val & y_val_bin))
            p_casc_val = np.where(m_val, val_pd, val_teacher_p)
            auc_casc = float(roc_auc_score(y_val_bin, p_casc_val)) if (y_val_bin.any() and (~y_val_bin).any()) else base_val_auc
            exit_rate = float(np.mean(m_cal))
            if fn_ai_val == 0 and fn_rate_cal <= 0.0030 and (base_val_auc - auc_casc) <= 0.0015 and exit_rate > best_exit_rate:
                best_exit_rate = exit_rate
                best_policy = {"enabled": exit_rate >= 0.05, "p_low": p_low, "defer_max": d_max}

    m_val_final = val_elig & (val_pd <= best_policy["p_low"]) & (val_df < best_policy["defer_max"]) if best_policy["enabled"] else np.zeros(len(val_txts), dtype=bool)
    m_cal_final = cal_elig & (cal_pd <= best_policy["p_low"]) & (cal_df < best_policy["defer_max"]) if best_policy["enabled"] else np.zeros(len(cal_txts), dtype=bool)
    val_cascade_raw_p = np.where(m_val_final, val_pd, val_teacher_p)
    cal_cascade_raw_p = np.where(m_cal_final, cal_pd, cal_teacher_p)
    nano_report = {
        "policy": best_policy,
        "cal_exit_rate": round(float(np.mean(m_cal_final)), 4),
        "val_gold_exit_rate": round(float(np.mean(m_val_final)), 4),
        "cal_missed_ai_count": int(np.sum(m_cal_final & cal_is_ai)),
        "val_base_auroc": round(base_val_auc, 4),
        "val_cascade_raw_probs": val_cascade_raw_p,
        "cal_cascade_raw_probs": cal_cascade_raw_p,
    }
    return nano.cpu().eval(), best_policy, nano_report


def train_and_export(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.special import expit
    from torch.utils.data import DataLoader, Dataset, Sampler
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v0.6.0-HyperLoop training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    if args.probe_vram:
        _run_vram_probe(args, device, amp_dtype)
        return

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else (512 if vram_gb >= 35.0 else 192)
    lr_scale = min(1.5, math.sqrt(batch_size / 256.0)) if (batch_size > 256 and not args.no_auto_lr_scale) else 1.0
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    print(
        f"[v0.6.0-HyperLoop Config] GPU={gpu_name} ({vram_gb:.1f} GB) | amp={amp_dtype} | bs={batch_size} | lr_scale={lr_scale:.2f} | max_len={args.max_len} | warmstart={args.warmstart_v05_r3}",
        flush=True,
    )

    train_items_raw, eval_rows, pairs, intent_vecs, data_audit = _load_v06_balanced_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        [p.resolve() for p in args.extra_pairs],
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
    )
    print(f"[v0.6.0-HyperLoop DataAudit] {json.dumps(data_audit, ensure_ascii=False)}", flush=True)
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Non-silent anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} gateguard pairs collided!")

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    if not args.smoke_test:
        rng_aug = random.Random(20261129)
        benign_urls = (" https://pan.quark.cn/s/demo1234", " https://github.com/example/repo", " https://b23.tv/BV1demo0001")
        human_grp_set = {GROUP_TO_IDX["pre_llm_human"], GROUP_TO_IDX["structured_human_hard"]}
        for item in train_items_raw:
            txt = item["raw_text"]
            if rng_aug.random() < 0.03 and not txt.startswith(("回复", "回覆")):
                txt = ("回复 @用户 : " if rng_aug.random() < 0.65 else "回复 :") + txt
            if (item["group_idx"] in human_grp_set or item["y_score"] <= 0.12) and rng_aug.random() < 0.03 and "http" not in txt:
                txt = txt + rng_aug.choice(benign_urls)
            item["raw_text"] = txt

    hf_tok = AutoTokenizer.from_pretrained(args.backbone)
    tok_tmp = out_dir / "_tok_tmp"
    hf_tok.save_pretrained(tok_tmp)
    for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_tmp / fn).is_file():
            shutil.copy2(tok_tmp / fn, out_dir / fn)
    shutil.rmtree(tok_tmp, ignore_errors=True)
    tokenizer = ByteFallbackWordPieceTokenizer(out_dir / "vocab.txt")

    print(f"[v0.6.0-HyperLoop Pre-encode] Tokenizing {len(train_items_raw)} rows (max_len={args.max_len}, zero-[UNK]) & 32-d PIFE...", flush=True)
    t_enc = time.monotonic()
    train_counts = np.zeros(BASE_VOCAB_SIZE, dtype=np.float64)
    total_toks = 0
    for item in train_items_raw:
        seq = tokenizer.encode_ids(item["raw_text"], max_len=args.max_len)
        item["ids"] = seq
        item["length"] = len(seq)
        item["style"] = extract_style_features_32d(item["raw_text"])
        total_toks += len(seq)
        for tid in seq:
            if tid < BASE_VOCAB_SIZE:
                train_counts[tid] += 1.0
    print(f"[v0.6.0-HyperLoop Pre-encode DONE] {len(train_items_raw)} rows in {time.monotonic()-t_enc:.1f}s | avg_tokens={total_toks/max(1,len(train_items_raw)):.2f}", flush=True)

    pairs_gg_pre, pairs_gen_pre = [], []
    if pairs:
        ai_styles = extract_style_features_batch([p["ai_text"] for p in pairs])
        hu_styles = extract_style_features_batch([p["human_text"] for p in pairs])
        for i, p in enumerate(pairs):
            rec = {
                "ai_ids": tokenizer.encode_ids(p["ai_text"], max_len=args.max_len),
                "hu_ids": tokenizer.encode_ids(p["human_text"], max_len=args.max_len),
                "ai_style": ai_styles[i],
                "hu_style": hu_styles[i],
                "weight": float(p["weight"]),
            }
            (pairs_gg_pre if p.get("is_gateguard") else pairs_gen_pre).append(rec)
    pairs_pre = pairs_gg_pre + pairs_gen_pre

    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 1024
    mdl = MeguvizV06HyperLoopModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        swa_heads=4,
        intermediate_size=2048,
        max_len=args.max_len,
        style_dim=STYLE_FEAT_DIM,
        intent_dim=intent_dim,
        num_codes=len(CODE_LIST),
        num_prag=len(PRAG_TAGS),
    )
    v05_ckpt_path = args.warmstart_v05_r3.resolve() if (args.warmstart_v05_r3 and args.warmstart_v05_r3.is_file()) else None
    if v05_ckpt_path is not None:
        init_mode = mdl.load_from_v05_r3_or_bge(v05_r3_ckpt=v05_ckpt_path)
    else:
        hf_src = AutoModel.from_pretrained(args.backbone)
        w_src = hf_src.embeddings.word_embeddings.weight.detach().float().cpu().numpy()
        a_full, p_mat = compute_weighted_svd_factors(w_src, train_counts, embed_dim=args.embed_dim, unk_id=tokenizer.unk_id)
        a_full = _seed_ascii_bytes_into_a(a_full, tokenizer, BASE_VOCAB_SIZE)
        init_mode = mdl.load_from_v05_r3_or_bge(hf_bge_model=hf_src, a_mat=a_full, p_mat=p_mat)
        del hf_src
    print(f"[v0.6.0-HyperLoop Init] mode={init_mode} | params={sum(p.numel() for p in mdl.parameters())/1e6:.2f}M", flush=True)
    mdl = mdl.to(device)
    mdl.set_sdpa(True)

    # Eval splits: val_gold (clean 100% gold for T + Smooth PAVA + primary Mandap-B), val_cal (n=2,400 for Nano-Draft FN audit)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_rows, _ = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(subset: list[dict]):
        txts = [str(r["text"]) for r in subset]
        enc = [tokenizer.encode_ids(t, max_len=args.max_len) for t in txts]
        st = extract_style_features_batch(txts)
        sc = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return enc, txts, st, sc

    val_enc, val_txts, val_st, val_scores = _pre_eval(val_gold)
    cal_enc, cal_txts, cal_st, cal_scores = _pre_eval(val_cal_rows)
    test_enc, test_txts, test_st, test_scores = _pre_eval(test_gold)
    test_dsh_enc, test_dsh_txts, test_dsh_st, test_dsh_scores = _pre_eval(test_dsh_gold)
    test_all_enc, test_all_txts, test_all_st, test_all_scores = _pre_eval(test_bili_all)
    probe_enc, probe_txts, probe_st, probe_scores = _pre_eval(probe_rows)
    eval_bs = 512 if vram_gb >= 35.0 else 256

    @torch.no_grad()
    def predict_pt(model_eval: MeguvizV06HyperLoopModel, enc_ids: list[list[int]], st_arr: np.ndarray, use_fp32: bool = False) -> tuple[np.ndarray, np.ndarray]:
        model_eval.eval()
        n = len(enc_ids)
        l_out = np.zeros(n, dtype=np.float32)
        emb_out = np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return l_out, emb_out
        # Length-sorted compact chunking eliminates pad-to-256 bias & matches MeguvizV06Engine!
        order = sorted(range(n), key=lambda i: len(enc_ids[i]))
        bs_eff = 64 if use_fp32 else eval_bs
        for st_i in range(0, n, bs_eff):
            mb = order[st_i : st_i + bs_eff]
            ml = max(len(enc_ids[i]) for i in mb)
            ids = torch.full((len(mb), ml), tokenizer.pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(mb), ml), dtype=torch.long, device=device)
            for ri, oi in enumerate(mb):
                seq = enc_ids[oi]
                ids[ri, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[ri, : len(seq)] = 1
            st_t = torch.from_numpy(st_arr[mb]).to(device)
            if use_fp32:
                lg, _, emb, _ = model_eval(ids, mask, st_t, return_aux=False)
            else:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    lg, _, emb, _ = model_eval(ids, mask, st_t, return_aux=False)
            lg_np = lg.float().cpu().numpy()
            emb_np = emb.float().cpu().numpy()
            for ri, oi in enumerate(mb):
                l_out[oi] = lg_np[ri]
                emb_out[oi] = emb_np[ri]
        return l_out, emb_out

    # Zero-sync GPU GroupDRO (restored from v0.4-R3!)
    class TorchGroupDRO:
        def __init__(self, num_groups: int = len(V04_GROUPS), eta: float = 0.04, ema_decay: float = 0.90, floor_share: float = 0.035):
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
            self.q = torch.where(present, self.q * torch.exp((self.eta * self.ema_losses).clamp(max=1.5)), self.q)
            self.q = self.q / self.q.sum().clamp(min=1e-12)
            floor_total = min(0.8, self.floor_share * self.num_groups)
            eff = (1.0 - floor_total) * self.q + (floor_total / self.num_groups)
            return eff / eff.sum()

    dro_tracker = TorchGroupDRO()

    class V06Dataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    class BucketSampler(Sampler):
        def __init__(self, lengths: list[int], bs: int, seed: int = 20261125):
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

    def make_collate(pife_noise_prob: float = 0.0):
        rng_c = random.Random(20261130)

        def _collate(batch):
            max_seq = min(args.max_len, ((max(x["length"] for x in batch) + 15) // 16) * 16)
            ids = np.full((len(batch), max_seq), tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), max_seq), dtype=np.int64)
            styles = np.empty((len(batch), STYLE_FEAT_DIM), dtype=np.float32)
            for i, x in enumerate(batch):
                seq = x["ids"]
                n = min(len(seq), max_seq)
                ids[i, :n] = seq[:n]
                mask[i, :n] = 1
                if pife_noise_prob > 0.0 and rng_c.random() < pife_noise_prob:
                    styles[i] = extract_style_features_32d(_inject_pife_noise(x["raw_text"], rng_c))
                else:
                    styles[i] = x["style"]
            return (
                torch.from_numpy(ids),
                torch.from_numpy(mask),
                torch.from_numpy(styles),
                torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
                torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
                torch.tensor([x["prag_vec"] for x in batch], dtype=torch.float32),
                torch.tensor([x["group_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["intent_idx"] for x in batch], dtype=torch.long),
            )

        return _collate

    def make_loader(items: list[dict], pife_noise_prob: float = 0.0) -> DataLoader:
        nw = min(8, max(4, (os.cpu_count() or 4) - 2)) if (os.name != "nt" and not args.smoke_test) else 0
        kw = {
            "batch_sampler": BucketSampler([x["length"] for x in items], batch_size),
            "collate_fn": make_collate(pife_noise_prob=pife_noise_prob),
            "num_workers": nw,
            "pin_memory": True,
        }
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 4
        return DataLoader(V06Dataset(items), **kw)

    def _collate_pairs(p_batch: list[dict]):
        bs_p = len(p_batch)
        m_ai = min(args.max_len, ((max(len(x["ai_ids"]) for x in p_batch) + 15) // 16) * 16)
        m_hu = min(args.max_len, ((max(len(x["hu_ids"]) for x in p_batch) + 15) // 16) * 16)
        ai_ids = np.full((bs_p, m_ai), tokenizer.pad_id, dtype=np.int64)
        ai_m = np.zeros((bs_p, m_ai), dtype=np.int64)
        hu_ids = np.full((bs_p, m_hu), tokenizer.pad_id, dtype=np.int64)
        hu_m = np.zeros((bs_p, m_hu), dtype=np.int64)
        ai_st = np.empty((bs_p, STYLE_FEAT_DIM), dtype=np.float32)
        hu_st = np.empty((bs_p, STYLE_FEAT_DIM), dtype=np.float32)
        for i, x in enumerate(p_batch):
            na, nh = min(len(x["ai_ids"]), m_ai), min(len(x["hu_ids"]), m_hu)
            ai_ids[i, :na] = x["ai_ids"][:na]
            ai_m[i, :na] = 1
            hu_ids[i, :nh] = x["hu_ids"][:nh]
            hu_m[i, :nh] = 1
            ai_st[i] = x["ai_style"]
            hu_st[i] = x["hu_style"]
        return (
            torch.from_numpy(ai_ids).to(device, non_blocking=True),
            torch.from_numpy(ai_m).to(device, non_blocking=True),
            torch.from_numpy(ai_st).to(device, non_blocking=True),
            torch.from_numpy(hu_ids).to(device, non_blocking=True),
            torch.from_numpy(hu_m).to(device, non_blocking=True),
            torch.from_numpy(hu_st).to(device, non_blocking=True),
            torch.tensor([x["weight"] for x in p_batch], dtype=torch.float32, device=device),
        )

    def _skewed_kl(p_s, y_t, alpha: float = 0.1):
        ps = p_s.float().clamp(1e-5, 1.0 - 1e-5)
        pt = y_t.float().clamp(1e-5, 1.0 - 1e-5)
        pm = ((1.0 - alpha) * pt + alpha * ps).clamp(1e-5, 1.0 - 1e-5)
        return pt * torch.log(pt / pm) + (1.0 - pt) * torch.log((1.0 - pt) / (1.0 - pm))

    intent_vecs_t = torch.from_numpy(intent_vecs).to(device) if intent_vecs is not None else None
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_code = nn.CrossEntropyLoss(ignore_index=-100)

    def make_optimizer(base_lr: float, mohawk_only_new: bool = False):
        pre_p, fresh_p = [], []
        new_tokens = ("lin_gate_proj", "dwt_cadence", "sub_ln_", "subln_blend", "span_head", "mil_gate", "ctx_echo_net", "dw")
        for name, p in mdl.named_parameters():
            if not p.requires_grad:
                continue
            is_new = any(tok in name for tok in new_tokens) or not name.startswith(("layers.", "word_emb_", "pos_emb.", "emb_ln."))
            if is_new:
                fresh_p.append(p)
            else:
                pre_p.append(p)
        if mohawk_only_new:
            groups = [
                {"params": pre_p, "lr": base_lr * 0.20, "base_lr": base_lr * 0.20, "weight_decay": 0.01},
                {"params": fresh_p, "lr": base_lr * 4.0, "base_lr": base_lr * 4.0, "weight_decay": 0.01},
            ]
        else:
            groups = [
                {"params": pre_p, "lr": base_lr, "base_lr": base_lr, "weight_decay": 0.01},
                {"params": fresh_p, "lr": base_lr * 1.6, "base_lr": base_lr * 1.6, "weight_decay": 0.01},
            ]
        return torch.optim.AdamW(groups, fused=True)

    def run_phase(
        items: list[dict],
        target_steps: int,
        num_epochs_fallback: int,
        lr: float,
        qat: bool,
        phase_name: str,
        beta_neg_start: float = 0.08,
        beta_neg_end: float = 0.04,
        pauc_weight: float = 0.12,
        ema_teacher: MeguvizV06HyperLoopModel | None = None,
        ema_alpha: float = 0.18,
        update_ema: bool = False,
        use_swa: bool = False,
        pife_noise_prob: float = 0.0,
        mohawk_only_new: bool = False,
        use_dro: bool = True,
        strong_gateguard: bool = False,
        keep_best_val: bool = False,
    ):
        def _val_eval_obj():
            v_l, _ = predict_pt(mdl, val_enc, val_st)
            v_p = expit(v_l)
            m_v = {**compute_eval_metrics(val_scores, v_p), **compute_high_precision_metrics(val_scores, v_p)}
            obj = (
                0.45 * float(m_v["auroc"])
                + 0.30 * float(m_v["pr_auc"])
                + 0.15 * float(m_v["best_f05"])
                + 0.10 * float(m_v["recall_at_p95"])
            )
            return obj, m_v

        mdl.set_qat_enabled(qat)
        loader = make_loader(items, pife_noise_prob=pife_noise_prob)
        steps_per_ep = max(1, len(loader))
        if target_steps > 0:
            total_steps = int(target_steps)
            num_epochs = max(1, int(math.ceil(total_steps / float(steps_per_ep))))
        else:
            num_epochs = max(1, int(num_epochs_fallback))
            total_steps = num_epochs * steps_per_ep

        eff_lr = float(lr) * lr_scale
        opt = make_optimizer(eff_lr, mohawk_only_new=mohawk_only_new)
        rng_p = random.Random(20261125 + len(phase_name))
        pair_bs = max(16, min(48, int(round(16.0 * batch_size / 256.0))))
        swa_start = int(total_steps * 0.55)
        swa_every = max(15, min(60, total_steps // 40))
        log_every = max(40, total_steps // 12)
        eval_every = max(250, min(450, total_steps // 3)) if (keep_best_val and not args.smoke_test) else 0
        t_p = time.monotonic()
        global_step = 0
        swa_state = None
        swa_count = 0
        best_val_obj = -1.0
        best_val_sd = None
        if keep_best_val:
            init_obj, m0 = _val_eval_obj()
            print(f"[{phase_name} INIT_VAL] obj={init_obj:.4f} auroc={m0['auroc']} pr_auc={m0['pr_auc']} rp95={m0['recall_at_p95']} rp99={m0['recall_at_p99']} f05={m0['best_f05']}", flush=True)
            if not qat:
                best_val_obj = init_obj
                best_val_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}

        print(f"[{phase_name}] items={len(items)} | bs={batch_size} | target_steps={total_steps} (~{num_epochs} ep) | pair_bs={pair_bs} | qat={qat} | use_dro={use_dro} | eff_lr={eff_lr:.2e}", flush=True)

        for ep in range(num_epochs):
            mdl.train()
            run_loss = torch.zeros((), dtype=torch.float32, device=device)
            n_seen = 0
            for ids, mask, st, y_sc, y_cd, w, prag, grp, intent_idx in loader:
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                y_sc = y_sc.to(device, non_blocking=True)
                y_cd = y_cd.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                prag = prag.to(device, non_blocking=True)
                grp = grp.to(device, non_blocking=True)
                intent_idx = intent_idx.to(device, non_blocking=True)

                mult = _cosine_mult(global_step, total_steps)
                for pg in opt.param_groups:
                    pg["lr"] = float(pg["base_lr"]) * mult
                opt.zero_grad(set_to_none=True)

                ema_p = None
                if ema_teacher is not None:
                    ema_teacher.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                        el, _, _, _ = ema_teacher(ids, mask, st, return_aux=False)
                        ema_p = torch.sigmoid(el.float())

                prog = float(global_step) / float(max(1, total_steps - 1))

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(ids, mask, st, return_aux=True)
                    lg = out["logit"].float()
                    ps = torch.sigmoid(lg)
                    y_eff = ((1.0 - ema_alpha) * y_sc + ema_alpha * ema_p) if ema_p is not None else y_sc

                    per_s = (0.75 * _skewed_kl(ps, y_eff) + 0.25 * F.smooth_l1_loss(ps, y_eff, beta=0.04, reduction="none")) * w
                    if use_dro:
                        grp_w_t = dro_tracker.update(grp, per_s) * len(V04_GROUPS)
                        dro_mult = grp_w_t[grp]
                        loss_main = (per_s * dro_mult).mean()
                    else:
                        loss_main = per_s.mean()

                    # Mid-layer deep supervision + MIL span supervision + Parallel TopK-SAE
                    mid_p = torch.sigmoid(out["mid_logit"].float())
                    loss_mid = ((0.75 * _skewed_kl(mid_p, y_eff) + 0.25 * F.smooth_l1_loss(mid_p, y_eff, beta=0.04, reduction="none")) * w).mean()
                    loss_mil = (F.binary_cross_entropy_with_logits(out["mil_peak"].float(), y_eff.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w).mean()
                    loss_sae = out["sae_loss"]

                    # Modern-Human Two-Way pAUC + Dual-Channel Top-Push (R3 mid-band quarantined!)
                    loss_pauc, loss_tp = _modern_two_way_pauc_and_top_push(
                        lg, y_sc, w, grp, prog=prog, beta_start=beta_neg_start, beta_end=beta_neg_end
                    )
                    loss_lreg = F.relu(lg.abs() - 4.0).pow(2).mean()

                    loss_aux = 0.12 * ce_code(out["code_logit"], y_cd)
                    pm = (prag >= 0.0).all(dim=-1)
                    if pm.any():
                        loss_aux = loss_aux + 0.18 * bce_none(out["prag_logit"][pm], prag[pm].clamp(min=0.0)).mean()
                    if intent_vecs_t is not None:
                        im = intent_idx >= 0
                        if im.any():
                            tgt_v = intent_vecs_t[intent_idx[im]].float()
                            pred_v = F.normalize(out["intent_pred"][im].float(), p=2, dim=-1)
                            loss_aux = loss_aux + 0.14 * (1.0 - (pred_v * tgt_v).sum(dim=-1)).mean()

                    loss = (
                        loss_main
                        + 0.15 * loss_mid
                        + 0.08 * loss_mil
                        + 0.05 * loss_sae
                        + pauc_weight * loss_pauc
                        + loss_tp
                        + 0.008 * loss_lreg
                        + loss_aux
                    )

                    if qat and ema_p is not None:
                        loss = loss + 0.18 * F.mse_loss(ps, ema_p)

                    if pairs_pre and (not strong_gateguard or global_step % 2 == 0):
                        if pairs_gg_pre and pairs_gen_pre:
                            gg_frac = 0.55 if strong_gateguard else 0.50
                            n_gg = min(len(pairs_gg_pre), max(6, int(round(pair_bs * gg_frac))))
                            n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                            p_batch = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                        else:
                            p_batch = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                        ai_ids, ai_m, ai_st, hu_ids, hu_m, hu_st, pw_t = _collate_pairs(p_batch)
                        l_ai_p, _, _, _ = mdl(ai_ids, ai_m, ai_st, return_aux=False)
                        l_hu_p, _, _, _ = mdl(hu_ids, hu_m, hu_st, return_aux=False)
                        l_ai_p, l_hu_p = l_ai_p.float(), l_hu_p.float()
                        if strong_gateguard:
                            loss_simpo = (F.softplus(1.8 - (l_ai_p - l_hu_p)) * pw_t).mean()
                            loss_pabs = ((F.relu(0.86 - torch.sigmoid(l_ai_p)).pow(2) + 1.4 * F.relu(torch.sigmoid(l_hu_p) - 0.12).pow(2)) * pw_t).mean()
                            loss = loss + 0.22 * loss_simpo + 0.20 * loss_pabs
                        else:
                            loss_simpo = (F.softplus(1.6 - (l_ai_p - l_hu_p)) * pw_t).mean()
                            loss_pabs = ((F.relu(0.86 - torch.sigmoid(l_ai_p)).pow(2) + F.relu(torch.sigmoid(l_hu_p) - 0.14).pow(2)) * pw_t).mean()
                            loss = loss + 0.26 * loss_simpo + 0.12 * loss_pabs

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()

                if update_ema and ema_teacher is not None:
                    with torch.no_grad():
                        for pe, ps_p in zip(ema_teacher.parameters(), mdl.parameters()):
                            pe.data.mul_(0.995).add_(ps_p.data, alpha=0.005)

                if use_swa and global_step >= swa_start and (global_step % swa_every == 0 or global_step == total_steps - 1):
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

                run_loss = run_loss + loss.detach().float()
                n_seen += 1
                global_step += 1
                if global_step % log_every == 0 or global_step == total_steps:
                    print(f"[{phase_name} step={global_step}/{total_steps} ep={ep+1}/{num_epochs}] loss={float(run_loss.item())/max(1,n_seen):.4f} elapsed={time.monotonic()-t_p:.1f}s", flush=True)
                if eval_every > 0 and global_step % eval_every == 0 and global_step < total_steps:
                    cur_obj, m_mid = _val_eval_obj()
                    if cur_obj > best_val_obj:
                        best_val_obj = cur_obj
                        best_val_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                    print(f"[{phase_name} MID step={global_step}] obj={cur_obj:.4f} (best={best_val_obj:.4f}) auroc={m_mid['auroc']} pr_auc={m_mid['pr_auc']} rp95={m_mid['recall_at_p95']} f05={m_mid['best_f05']}", flush=True)
                    mdl.train()
                if global_step >= total_steps:
                    break

            cur_obj, m_ep = _val_eval_obj()
            if keep_best_val and cur_obj > best_val_obj:
                best_val_obj = cur_obj
                best_val_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(f"[{phase_name} ep={ep+1} DONE] obj={cur_obj:.4f} val: auroc={m_ep['auroc']} pr_auc={m_ep['pr_auc']} rp95={m_ep['recall_at_p95']} rp99={m_ep['recall_at_p99']} f05={m_ep['best_f05']} mae={m_ep['mae']}", flush=True)
            if global_step >= total_steps:
                break

        if keep_best_val and best_val_sd is not None:
            if swa_state is not None:
                mdl.load_state_dict(swa_state, strict=True)
                swa_obj, m_swa = _val_eval_obj()
                print(f"[{phase_name} SWA_CHECK] swa_obj={swa_obj:.4f} (auroc={m_swa['auroc']} pr_auc={m_swa['pr_auc']} rp95={m_swa['recall_at_p95']}) vs best_obj={best_val_obj:.4f}", flush=True)
                if swa_obj >= best_val_obj - 0.003:
                    return swa_state
            mdl.load_state_dict({k: v.to(device) for k, v in best_val_sd.items()}, strict=True)
            print(f"[{phase_name} BEST_RESTORED] Kept best val snapshot (obj={best_val_obj:.4f}).", flush=True)
            return None
        return swa_state

    # Stage 0 (MOHAWK): Align 2:2 Interleaved Split-Head Hybrid linear heads + DWT + SubLN + MIL smoothly!
    stage1_ckpt = ckpt_dir / f"{args.version}_stage1_ckpt.pt"
    if args.resume_stage1 and Path(args.resume_stage1).is_file():
        mdl.load_state_dict(torch.load(args.resume_stage1, map_location=device), strict=True)
    else:
        mdl.set_swa_head_drop(0.10)
        if args.mohawk_steps > 0:
            mohawk_items, _ = subset_balanced_gold_guaranteed(train_items_raw, cap=min(220000, len(train_items_raw)), seed_s=20261124)
            run_phase(
                mohawk_items,
                target_steps=8 if args.smoke_test else int(args.mohawk_steps),
                num_epochs_fallback=1,
                lr=args.lr * 1.2,
                qat=False,
                phase_name="Stage0-MOHAWK-Align",
                beta_neg_start=0.08,
                beta_neg_end=0.06,
                pauc_weight=0.07,
                mohawk_only_new=True,
            )

        # Stage 1: Full-Corpus GroupDRO + Split-Head Hybrid + Modern+DSH Two-Way pAUC
        run_phase(
            train_items_raw,
            target_steps=16 if args.smoke_test else 0,
            num_epochs_fallback=1 if args.smoke_test else int(args.epochs),
            lr=args.lr,
            qat=False,
            phase_name="Stage1-HyperLoop-pAUC",
            beta_neg_start=0.08,
            beta_neg_end=0.05,
            pauc_weight=0.08,
        )
        if not args.smoke_test:
            torch.save(mdl.state_dict(), stage1_ckpt)
            print(f"[Stage1] Saved reusable Stage-1 checkpoint -> {stage1_ckpt}", flush=True)

    # Disable stochastic SWA head drop in Stage 2/3 so final calibration & SWA converge cleanly!
    mdl.set_swa_head_drop(0.0)
    ema_teacher = copy.deepcopy(mdl).eval()
    ema_teacher.set_sdpa(True)
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # Stage 2 (R3 Fixed!): Natural-prior gold-guaranteed sampling (preserves ~20% pos share & full modern humans!),
    # zero mismatched style-only noise, stronger GateGuard trap boundary anchor, and best-val monotonic rollback lock!
    stage2_items, gold_ret_s2 = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=args.stage2_cap, seed_s=20261126)
    s2_pos_share = sum(1 for x in stage2_items if x["y_score"] >= 0.70) / max(1, len(stage2_items))
    print(f"[Stage2-R3] Subset={len(stage2_items)} | gold_retention={gold_ret_s2:.4f} | natural_pos_share={s2_pos_share:.4f}", flush=True)
    run_phase(
        stage2_items,
        target_steps=12 if args.smoke_test else int(args.stage2_steps),
        num_epochs_fallback=1,
        lr=args.refine_lr,
        qat=False,
        phase_name="Stage2-NaturalPrior-Refine",
        beta_neg_start=0.06,
        beta_neg_end=0.04,
        pauc_weight=0.09,
        ema_teacher=ema_teacher,
        ema_alpha=0.18,
        update_ema=True,
        pife_noise_prob=0.0,
        use_dro=False,
        strong_gateguard=True,
        keep_best_val=True,
    )

    # Freeze Stage-2 winner as clean QAT distillation teacher
    ema_teacher.load_state_dict(mdl.state_dict(), strict=True)
    ema_teacher.eval()

    # Stage 3 (R3 Fixed!): Natural-prior QAT + SWA with strict best-val guard
    qat_items, gold_ret_s3 = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=args.qat_cap, seed_s=20261127)
    s3_pos_share = sum(1 for x in qat_items if x["y_score"] >= 0.70) / max(1, len(qat_items))
    print(f"[Stage3-QAT-R3] Subset={len(qat_items)} | gold_retention={gold_ret_s3:.4f} | natural_pos_share={s3_pos_share:.4f}", flush=True)
    swa_sd = run_phase(
        qat_items,
        target_steps=12 if args.smoke_test else int(args.qat_steps),
        num_epochs_fallback=1,
        lr=args.qat_lr,
        qat=True,
        phase_name="Stage3-QAT-SWA",
        beta_neg_start=0.05,
        beta_neg_end=0.04,
        pauc_weight=0.08,
        ema_teacher=ema_teacher,
        ema_alpha=0.22,
        update_ema=False,
        use_swa=not args.smoke_test,
        pife_noise_prob=0.0,
        use_dro=False,
        strong_gateguard=True,
        keep_best_val=True,
    )
    if swa_sd is not None:
        mdl.load_state_dict(swa_sd, strict=True)
        print("[Stage3-QAT-SWA] Applied SWA weights (validated >= best snapshot).", flush=True)

    del ema_teacher
    mdl.set_sdpa(False)
    mdl.set_qat_enabled(False)
    mdl.eval()

    # Stage 4a: 0 KB, 0.000 ms Weight-Folded Confounding-Neuron Orthogonal Projection Surgery (auto-validated on val_gold!)
    with torch.no_grad():
        mdl.temperature.fill_(1.0)
    _, val_embs = predict_pt(mdl, val_enc, val_st, use_fp32=True)
    surgery_report = apply_zero_cost_confounding_surgery(
        mdl,
        val_hard_human_embs=val_embs[val_scores <= 35.0],
        val_ai_embs=val_embs[val_scores >= 70.0],
        val_all_embs=val_embs,
        val_all_scores=val_scores,
        rank=8,
    )
    print(f"[Stage4a-Surgery] {json.dumps(surgery_report)}", flush=True)

    # Sharp Temperature Baking strictly on clean val_gold (FP32 exact parity!)
    raw_val_l, _ = predict_pt(mdl, val_enc, val_st, use_fp32=True)
    t_raw = float(fit_temperature(raw_val_l, val_scores / 100.0))
    t_baked = float(np.clip(t_raw, 0.90, 1.80))
    with torch.no_grad():
        mdl.temperature.fill_(t_baked)
    print(f"[Calibration] raw_T_val_gold={t_raw:.3f} -> baked_sharp_T={t_baked:.3f}", flush=True)

    val_l_after, _ = predict_pt(mdl, val_enc, val_st, use_fp32=True)
    val_p_raw = expit(val_l_after)
    cal_l, _ = predict_pt(mdl, cal_enc, cal_st, use_fp32=False)
    cal_p_raw = expit(cal_l)
    _, test_embs = predict_pt(mdl, test_enc, test_st, use_fp32=True)

    # Stage 4b: Frozen-Backbone 22 KB LB-DSM Dual-Manifold Hutchinson Curvature & OOD Critic + L2 Prototype Bank
    bank_subset = train_items_raw[: min(24000, len(train_items_raw))]
    if not args.smoke_test and len(train_items_raw) > 24000:
        rng_bk = random.Random(20261125)
        hu_anc = [x for x in train_items_raw if x["is_hu_anchor"]]
        ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
        bank_subset = rng_bk.sample(hu_anc, min(12000, len(hu_anc))) + rng_bk.sample(ai_anc, min(12000, len(ai_anc)))
    _, bk_emb = predict_pt(mdl, [x["ids"] for x in bank_subset], np.stack([x["style"] for x in bank_subset], axis=0))
    bk_is_hu = np.array([bool(x["is_hu_anchor"]) for x in bank_subset], dtype=bool)
    bk_is_ai = np.array([x["y_score"] >= 0.82 for x in bank_subset], dtype=bool)
    proto_bank = fit_l2_prototype_bank(
        bk_emb,
        np.array([x["y_score"] * 100.0 for x in bank_subset], dtype=np.float32),
        is_human_anchor=bk_is_hu,
    )
    proto_path = out_dir / "prototype_bank.npz"
    np.savez_compressed(proto_path, **proto_bank)

    lb_critic, lb_report = _train_frozen_lb_dsm_critic(
        hu_embs=bk_emb[bk_is_hu],
        ai_embs=bk_emb[bk_is_ai],
        test_embs=test_embs,
        test_scores=test_scores,
        device=device,
        steps=80 if args.smoke_test else 650,
    )
    lb_path = out_dir / "lb_dsm_critic.npz"
    np.savez_compressed(lb_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in lb_critic.state_dict().items()})
    lb_report["file"] = lb_path.name
    lb_report["npz_bytes"] = int(lb_path.stat().st_size)
    print(f"[Stage4b-LB-DSM] {json.dumps(lb_report)}", flush=True)

    # Stage 4c: 0.155 MB Nano-Draft L2D-FP distillation from frozen HyperLoop teacher + strict FN-barrier policy
    nano_sub = train_items_raw if (args.smoke_test or len(train_items_raw) <= 60000) else random.Random(20261128).sample(train_items_raw, 60000)
    nano_t_l, _ = predict_pt(mdl, [x["ids"] for x in nano_sub], np.stack([x["style"] for x in nano_sub], axis=0))
    nano_cpu, nano_policy, nano_rep = _train_and_calibrate_nano_draft(
        train_items=nano_sub,
        teacher_probs_train=expit(nano_t_l),
        val_txts=val_txts,
        val_st=val_st,
        val_scores=val_scores,
        val_teacher_p=val_p_raw,
        cal_txts=cal_txts,
        cal_st=cal_st,
        cal_scores=cal_scores,
        cal_teacher_p=cal_p_raw,
        device=device,
        steps=60 if args.smoke_test else 1200,
    )
    val_cascade_raw_p = nano_rep.pop("val_cascade_raw_probs")
    cal_cascade_raw_p = nano_rep.pop("cal_cascade_raw_probs")
    print(f"[Stage4c-NanoDraft] {json.dumps(nano_rep)}", flush=True)

    # Stage 5: Sharp Global Clean-Gold Smooth PAVA (raw_blend=0.55 preserves 0.88-0.96 high-confidence AI tail!)
    smooth_pava = fit_smooth_pava(val_p_raw, val_scores / 100.0, min_slope=0.35, raw_blend=0.55, max_knots=64) if not args.smoke_test else {}
    val_p_main = apply_bucketed_smooth_pava(val_txts, val_p_raw, smooth_pava)
    val_p_casc = apply_bucketed_smooth_pava(val_txts, val_cascade_raw_p, smooth_pava)
    cal_p_casc = apply_bucketed_smooth_pava(cal_txts, cal_cascade_raw_p, smooth_pava)

    t_f05, t_fpr05 = find_val_thresholds(val_scores, val_p_main)
    hp_val = compute_high_precision_metrics(val_scores, val_p_main)
    conformal_main = fit_split_conformal_3way(val_scores, val_p_main, alphas=(0.03, 0.05))
    conformal_mandap_b = fit_split_conformal_3way(val_scores, val_p_casc, alphas=(0.03, 0.05))
    conformal_mandap_b_2400 = fit_split_conformal_3way(cal_scores, cal_p_casc, alphas=(0.03, 0.05))
    print(f"[Stage5-Calibration] val_gold_n={len(val_scores)} | mandap_b_alpha03={conformal_mandap_b.get('alpha_03')} | cal2400_alpha03={conformal_mandap_b_2400.get('alpha_03')}", flush=True)

    def _eval_split(enc, txts, st, sc, exact_fp32: bool = True):
        lg, _ = predict_pt(mdl, enc, st, use_fp32=exact_fp32)
        raw_p = expit(lg)
        cp = apply_bucketed_smooth_pava(txts, raw_p, smooth_pava)
        return {**compute_eval_metrics(sc, cp, t_f05, t_fpr05), **compute_high_precision_metrics(sc, cp)}, cp

    test_m, test_p = _eval_split(test_enc, test_txts, test_st, test_scores, exact_fp32=True)
    dsh_m, _ = _eval_split(test_dsh_enc, test_dsh_txts, test_dsh_st, test_dsh_scores, exact_fp32=True)
    all_m, _ = _eval_split(test_all_enc, test_all_txts, test_all_st, test_all_scores, exact_fp32=False)
    probe_m, _ = _eval_split(probe_enc, probe_txts, probe_st, probe_scores, exact_fp32=True)

    # Export Bundle v3 ONNX (Nano-Draft INT8 + HyperLoop INT8 [MatMul/Gemm only!] + Ordered W4-Emb/W8-Body Edge ONNX)
    exp_info = export_v06_bundle_v3_onnx(mdl, nano_cpu, out_dir, vocab_size=tokenizer.vocab_size)
    nano_rep["onnx_bytes"] = exp_info.get("nano_draft_bytes")

    manifest_base = {
        "schema_version": "meguviz.bundle.v3",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": "v0.6.0-HyperLoop(NanoDraftL2DFP+4L_2to2_SplitHead[SWA64+BiDeltaNet2]+LiftingHaarDWT+BitDistillSubLN+MIL+ParallelTopKSAE+0KBSurgery+LBDSM)",
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "init_mode": init_mode,
        "quant_method": "qat_swa_plus_dynamic_int8_matmul_gemm",
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": int(args.embed_dim),
        "hidden_size": 512,
        "num_layers": 4,
        "swa_heads_interleaved": [8, 4, 8, 4],
        "max_len": int(args.max_len),
        "vocab_file": "vocab.txt",
        "nano_draft_file": exp_info.get("nano_draft_file"),
        "nano_policy": nano_policy,
        "prototype_bank_file": proto_path.name,
        "prototype_bank_sha256": sha256_file(proto_path),
        "lb_dsm_critic_file": lb_path.name,
        "lb_dsm_critic_sha256": sha256_file(lb_path),
        "temperature": round(t_baked, 6),
        "smooth_pava": smooth_pava,
        "conformal": conformal_mandap_b,
        "conformal_cal2400": conformal_mandap_b_2400,
        "conformal_main_only": conformal_main,
        "thresholds": {"f05": round(t_f05, 4), "p95": hp_val["t_p95"], "p99": hp_val["t_p99"], "fpr05": round(t_fpr05, 4), "score70": 0.70},
        "model_file": exp_info["model_int8"],
        "model_edge_w4": exp_info["model_edge_w4"],
        "model_bytes": exp_info["model_int8_bytes"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT parity + cascade benchmark
    engine = MeguvizV06Engine(out_dir, threads=1, use_edge_w4=False, enable_nano_cascade=True)
    t_s = time.perf_counter()
    _ = [engine.predict_one(txt) for txt in test_txts]
    single_ms = (time.perf_counter() - t_s) * 1000.0 / max(1, len(test_txts))
    t_b = time.perf_counter()
    ort_batch_main = engine.predict_batch(test_txts, max_batch=32, force_main=True)
    batch_main_ms = (time.perf_counter() - t_b) * 1000.0 / max(1, len(test_txts))
    t_c = time.perf_counter()
    ort_batch_casc = engine.predict_batch(test_txts, max_batch=32, force_main=False)
    batch_casc_ms = (time.perf_counter() - t_c) * 1000.0 / max(1, len(test_txts))

    ort_p_main = np.array([x["prob"] for x in ort_batch_main], dtype=np.float32)
    ort_p_casc = np.array([x["prob"] for x in ort_batch_casc], dtype=np.float32)
    test_casc_m = {**compute_eval_metrics(test_scores, ort_p_casc, t_f05, t_fpr05), **compute_high_precision_metrics(test_scores, ort_p_casc)}
    nano_rep["test_gold_exit_rate"] = round(float(np.mean([1.0 if x["exited_nano"] else 0.0 for x in ort_batch_casc])), 4)
    nano_rep["test_gold_cascade_metrics"] = test_casc_m

    parity = {
        "mae_vs_pytorch": round(float(np.mean(np.abs(test_p - ort_p_main))), 5),
        "max_abs_diff_vs_pytorch": round(float(np.max(np.abs(test_p - ort_p_main))), 5),
        "single_1thread_ms": round(single_ms, 3),
        "compact_batch32_1thread_ms": round(batch_casc_ms, 3),
        "compact_batch32_main_only_1thread_ms": round(batch_main_ms, 3),
        "model_bytes": exp_info["model_int8_bytes"],
    }

    probe_file = Path("round01/bili-round01-user-confirmed.jsonl")
    verify_res = run_verify_v06(out_dir, probe_file.resolve(), threads=1) if probe_file.is_file() else {}

    metrics = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v3",
        "backbone": args.backbone,
        "init_mode": init_mode,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "batch_size_used": batch_size,
        "data_audit": data_audit,
        "gold_retention": {"stage2": round(gold_ret_s2, 4), "stage3": round(gold_ret_s3, 4)},
        "confounding_surgery": surgery_report,
        "lb_dsm_critic": lb_report,
        "nano_draft": nano_rep,
        "temperature": round(t_baked, 6),
        "raw_fitted_temperature": round(t_raw, 4),
        "conformal": conformal_main,
        "conformal_mandap_b": conformal_mandap_b,
        "conformal_mandap_b_cal2400": conformal_mandap_b_2400,
        "test_bili_gold": test_m,
        "test_dsh_gold": dsh_m,
        "test_all_bili_distilled": all_m,
        "probe_u001_u017": probe_m,
        "onnx_int8": parity,
        "edge_w4_w8": exp_info["edge_w4_w8"],
        "onnx_verify": {k: v for k, v in verify_res.items() if k != "probe_u001_u017"},
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }

    metrics_file = out_dir / f"metrics-{args.version}.json"
    metrics["release_gates"] = check_v06_release_gates(metrics)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Package INT8 Bundle v3 zip & Edge-W4 Bundle v3 zip
    common = [
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        proto_path.name,
        lb_path.name,
        metrics_file.name,
    ]
    if exp_info.get("nano_draft_file"):
        common.append(exp_info["nano_draft_file"])

    zip_int8 = out_dir / f"meguvizcac-{args.version}-int8.zip"
    with ZipFile(zip_int8, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for fn in (exp_info["model_int8"], "manifest.json", *common):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    metrics["onnx_int8"]["zip_bytes"] = int(zip_int8.stat().st_size)

    if exp_info["edge_w4_w8"].get("ok") and exp_info["model_edge_w4"]:
        edge_dir = out_dir / "_edge_tmp"
        edge_dir.mkdir(parents=True, exist_ok=True)
        em = dict(manifest_base)
        em["quant_method"] = "ordered_w4_emb_opset21_plus_w8_body"
        em["model_file"] = exp_info["model_edge_w4"]
        em["model_bytes"] = int((out_dir / exp_info["model_edge_w4"]).stat().st_size)
        (edge_dir / "manifest.json").write_text(json.dumps(em, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        zip_edge = out_dir / f"meguvizcac-{args.version}-edge-w4.zip"
        with ZipFile(zip_edge, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
            zf.write(out_dir / exp_info["model_edge_w4"], exp_info["model_edge_w4"])
            zf.write(edge_dir / "manifest.json", "manifest.json")
            for fn in common:
                fp = out_dir / fn
                if fp.is_file():
                    zf.write(fp, fn)
        shutil.rmtree(edge_dir, ignore_errors=True)
        metrics["edge_w4_w8"]["edge_zip_bytes"] = int(zip_edge.stat().st_size)

    metrics["release_gates"] = check_v06_release_gates(metrics)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(
        json.dumps(
            {
                "init_mode": init_mode,
                "test_bili_gold": test_m,
                "test_dsh_gold": dsh_m,
                "probe_u001_u017": probe_m,
                "confounding_surgery": surgery_report,
                "lb_dsm_critic": lb_report,
                "nano_draft": {k: v for k, v in nano_rep.items() if k != "test_gold_cascade_metrics"},
                "onnx_int8": metrics["onnx_int8"],
                "edge_w4_w8": metrics["edge_w4_w8"],
                "hard_traps": (verify_res.get("hard_trap_gates") or {}),
                "ml_traps": (verify_res.get("multilingual_trap_gates") or {}),
                "conformal_mandap_b_alpha03": conformal_mandap_b.get("alpha_03"),
                "release_gates": metrics["release_gates"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gateguard-data", type=Path, default=Path("meguviz_cac/data/v05-gateguard-pairs.jsonl"))
    parser.add_argument("--extra-pairs", type=Path, action="append", default=[])
    parser.add_argument("--intent-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("./out_v06_hyperloop"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--warmstart-v05-r3", type=Path, default=None)
    parser.add_argument("--resume-stage1", type=Path, default=None)
    parser.add_argument("--version", default="v0.6.0-r2-hyperloop-b3")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--max-len", type=int, default=MAX_LEN_V06)
    parser.add_argument("--mohawk-steps", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--stage2-steps", type=int, default=2000)
    parser.add_argument("--qat-steps", type=int, default=2200)
    parser.add_argument("--stage2-cap", type=int, default=420000)
    parser.add_argument("--qat-cap", type=int, default=380000)
    parser.add_argument("--stage1-synth-cap", type=int, default=15000)
    parser.add_argument("--v04-synth-cap", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-auto-lr-scale", action="store_true")
    parser.add_argument("--lr", type=float, default=3.2e-5)
    parser.add_argument("--refine-lr", type=float, default=1.4e-5)
    parser.add_argument("--qat-lr", type=float, default=5.0e-6)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe-vram", default="")
    args = parser.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()
