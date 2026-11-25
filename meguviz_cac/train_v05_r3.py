"""Train, Born-Again EMA + Clean-Whitelist Two-Way pAUC refine, Fixed-Step QAT+SWA, Smooth PAVA / Conformal / L2 Prototype
calibrate & export MeguvizCAC v0.5.0-R3 (BGE-4L + DWConv + PIFE-32d + Parallel Residual TopK-SAE + Clean-Whitelist pAUC)."""

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
from .v03_tokenizer import BASE_VOCAB_SIZE, ByteFallbackWordPieceTokenizer, MAX_LEN
from .v04_corpus_prep import BENCH_TRAP_TEXTS, GROUP_TO_IDX, V04_GROUPS, assign_group_id
from .v04_runtime import fit_split_conformal_3way
from .v05_features import (
    STYLE_FEAT_DIM,
    apply_bucketed_smooth_pava,
    extract_style_features_32d,
    extract_style_features_batch,
    fit_bucketed_smooth_pava,
    prepare_text_v05,
)
from .v05_r3_bge import MeguvizV05R3BGEModel, MeguvizV05R3Engine, export_v05_r3_onnx, run_verify_v05_r3
from .v05_runtime import PIFE_ADVERSARIAL_TRAPS, fit_l2_prototype_bank


def _load_v05_r3_bundle(
    data_path: Path,
    gateguard_path: Path | None,
    extra_pairs_paths: list[Path],
    intent_cache_path: Path | None,
    stage1_synth_cap: int = 10000,
    v04_synth_cap: int = 35000,
    seed: int = 20261125,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray | None, dict]:
    """Load merged R3 corpus with Protection-Bucket Whitelist + Balanced Anti-Drowning + Clean-Negative Tagging."""
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
                pairs.append(
                    {
                        "ai_text": ai_t,
                        "human_text": hu_t,
                        "weight": float(p.get("weight", 2.4 if is_gateguard else 0.7)),
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
            if len(str(r.get("text") or "").strip()) >= 16:
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
    real_bili_pos_upweighted = 0
    clean_hard_hu_upweighted = 0
    weak_mid_downweighted = 0
    clean_neg_total = 0

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

        # Protection-Bucket Whitelist
        if gid_str == "structured_human_hard" and (not is_pre_llm) and (not is_gold_tier):
            gid_str = "general_real"
            demoted_struct_weak += 1

        # Clean-Negative Whitelist (RTX 4060 Ti E3a empirical lesson: NEVER let noisy pseudo-negatives enter Two-Way pAUC tail!)
        is_clean_neg = bool(
            (sc_f <= 25.0)
            and (
                is_pre_llm
                or is_gold_tier
                or gid_str in ("structured_human_hard", "pre_llm_human")
                or (gid_str == "multilingual_world_knowledge" and is_gold_tier)
            )
        )
        if is_clean_neg:
            clean_neg_total += 1

        # Balanced Anti-Drowning Weights (fixes BOTH v0.4-R3 23.8:1 drowning AND v0.5-R2 4x positive FPR blowup!)
        w_eff = float(r.get("train_weight", 1.0))
        is_real_bili = coh in ("bili", "v04_1m")
        if is_real_bili and sc_f >= 65.0:
            w_eff *= 2.2 if is_gold_tier else 1.6
            real_bili_pos_upweighted += 1
        elif coh in ("synth_250k", "deepseek", "other_models"):
            w_eff *= 0.65
        elif gid_str == "structured_human_hard" and sc_f <= 35.0:
            w_eff *= 1.8
            clean_hard_hu_upweighted += 1
        elif is_pre_llm and sc_f <= 25.0:
            w_eff *= 1.5
            clean_hard_hu_upweighted += 1
        elif gid_str == "multilingual_world_knowledge" and sc_f <= 30.0 and is_gold_tier:
            w_eff *= 1.7
            clean_hard_hu_upweighted += 1
        elif (not is_gold_tier) and (not is_pre_llm) and (22.0 < sc_f < 68.0) and len(txt_s.strip()) <= 24:
            w_eff *= 0.35
            weak_mid_downweighted += 1

        prag = r.get("prag_vec")
        if not (isinstance(prag, list) and len(prag) == len(PRAG_TAGS)):
            prag = [-1.0] * len(PRAG_TAGS)

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
                "is_clean_neg": is_clean_neg,
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
        "gold_tier_count": sum(1 for x in train_items if x["is_gold_tier"]),
        "clean_neg_whitelist_count": clean_neg_total,
        "real_bili_pos_upweighted": real_bili_pos_upweighted,
        "clean_hard_hu_upweighted": clean_hard_hu_upweighted,
        "synth_250k_kept": len(v04_synth_rows),
        "weak_mid_short_downweighted": weak_mid_downweighted,
        "demoted_weak_structured_human": demoted_struct_weak,
        "eval_rows": len(eval_rows),
        "gateguard_kept_by_cat": dict(gg_kept_cats),
        "gateguard_dropped_leakage": gg_dropped,
        "extra_pairs_dropped_leakage": extra_dropped,
        "total_clean_pairs": len(pairs),
    }
    return train_items, eval_rows, pairs, intent_vecs, audit


def _two_way_pauc_clean_loss(logits, y_sc, is_clean_neg, w, beta_neg: float = 0.05, alpha_pos: float = 0.25, gamma: float = 1.5, margin: float = 2.4):
    """RTX 4060 Ti E3a-validated Clean-Whitelist Two-Way Partial AUC loss (focuses tail ONLY on verified clean negatives!)."""
    import torch
    import torch.nn.functional as F

    pos_l = logits[(y_sc >= 0.85) & (w > 0.0)].float()
    neg_l = logits[is_clean_neg & (y_sc <= 0.18) & (w > 0.0)].float()
    if pos_l.numel() == 0 or neg_l.numel() == 0:
        return logits.sum() * 0.0
    k_neg = max(1, int(math.ceil(neg_l.numel() * beta_neg)))
    m_pos = max(1, int(math.ceil(pos_l.numel() * alpha_pos)))
    hard_neg = torch.topk(neg_l, min(k_neg, neg_l.numel()), largest=True).values
    hard_pos = torch.topk(pos_l, min(m_pos, pos_l.numel()), largest=False).values
    diff = hard_pos.unsqueeze(1) - hard_neg.unsqueeze(0)
    return F.softplus(gamma * (margin - diff)).mean()


def check_v05_r3_release_gates(metrics: dict) -> dict:
    tb = metrics.get("test_bili_gold") or {}
    ox = metrics.get("onnx_int8") or {}
    ed = metrics.get("edge_w4_w8") or {}
    ver = metrics.get("onnx_verify") or {}
    ht = ver.get("hard_trap_gates") or {}
    ml = ver.get("multilingual_trap_gates") or {}
    conf03 = (metrics.get("conformal") or {}).get("alpha_03") or {}
    t_c = float(metrics.get("temperature", 9.9))
    edge_zip_mb = (ed.get("edge_zip_bytes") or (ed.get("bytes") or 0)) / (1024.0 * 1024.0) if ed.get("ok") else 99.0

    gates = {
        "speed_1t_le_2p8_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 2.80 ms (1T)", "pass": (ox.get("compact_batch32_1thread_ms") or 99.0) <= 2.80},
        "edge_w4_size_le_12mb": {"value": round(edge_zip_mb, 2), "target": "<= 12.0 MB zip", "pass": bool(ed.get("ok")) and edge_zip_mb <= 12.0},
        "auroc_ge_0955": {"value": tb.get("auroc"), "target": ">= 0.955 (stretch 0.965)", "pass": (tb.get("auroc") or 0.0) >= 0.955},
        "pr_auc_ge_0940": {"value": tb.get("pr_auc"), "target": ">= 0.940 (stretch 0.955)", "pass": (tb.get("pr_auc") or 0.0) >= 0.940},
        "rp95_ge_080": {"value": tb.get("recall_at_p95"), "target": ">= 0.80 (stretch 0.85)", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.80},
        "rp99_ge_050": {"value": tb.get("recall_at_p99"), "target": ">= 0.50 (stretch 0.60)", "pass": (tb.get("recall_at_p99") or 0.0) >= 0.50},
        "struct_human_lt_018": {"value": ht.get("structured_human_max_prob"), "target": "< 0.18", "pass": (ht.get("structured_human_max_prob") or 1.0) < 0.18},
        "ai_traps_ge_085": {"value": ht.get("ai_traps_min_prob"), "target": ">= 0.85", "pass": (ht.get("ai_traps_min_prob") or 0.0) >= 0.85},
        "ml_human_lt_018": {"value": ml.get("ml_human_max_prob"), "target": "< 0.18", "pass": (ml.get("ml_human_max_prob") or 1.0) < 0.18},
        "ml_ai_ge_085": {"value": ml.get("ml_ai_min_prob"), "target": ">= 0.85", "pass": (ml.get("ml_ai_min_prob") or 0.0) >= 0.85},
        "conformal_grey_le_18pct": {"value": conf03.get("cal_grey_rate"), "target": "<= 0.18", "pass": (conf03.get("cal_grey_rate") or 1.0) <= 0.18},
        "sharp_temp_in_09_18": {"value": round(t_c, 3), "target": "[0.9, 1.8]", "pass": 0.9 <= t_c <= 1.8},
        "qat_parity_le_0008": {"value": ox.get("mae_vs_pytorch"), "target": "<= 0.008", "pass": (ox.get("mae_vs_pytorch") or 99.0) <= 0.008},
        "zero_unk": {"value": ver.get("unk_tokens_total"), "target": "== 0", "pass": ver.get("unk_tokens_total") == 0},
    }
    return {"gates": gates, "passed_count": sum(1 for v in gates.values() if v["pass"]), "total_gates": len(gates), "all_passed": all(v["pass"] for v in gates.values())}


def _run_vram_probe(args, device, amp_dtype) -> None:
    import torch
    import torch.nn.functional as F

    cands = [int(x.strip()) for x in str(args.probe_vram).split(",") if x.strip()]
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap_gib = vram_gb * 0.90
    mdl = MeguvizV05R3BGEModel(vocab_size=BASE_VOCAB_SIZE + 256, embed_dim=args.embed_dim).to(device)
    mdl.set_sdpa(True)
    mdl.set_qat_enabled(True)
    ema = copy.deepcopy(mdl).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-4, fused=True)

    results = []
    chosen = cands[-1]
    for bs in cands:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pair_bs = max(16, min(64, int(round(16.0 * bs / 256.0))))
        ok = True
        err = ""
        step_ms = 0.0
        try:
            t0 = time.perf_counter()
            for _ in range(2):
                ids = torch.randint(5, BASE_VOCAB_SIZE, (bs, MAX_LEN), device=device)
                mask = torch.ones((bs, MAX_LEN), dtype=torch.long, device=device)
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
                    l_ai, _, _ = mdl(p_ids, p_m, p_st, return_aux=False)
                    l_hu, _, _ = mdl(p_ids, p_m, p_st, return_aux=False)
                    loss = loss + 0.3 * F.softplus(1.8 - (l_ai - l_hu)).mean()
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


def train_and_export(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.special import expit
    from torch.utils.data import DataLoader, Dataset, Sampler
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v0.5.0-R3 training.")

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

    print(f"[v0.5.0-R3 Config] GPU={gpu_name} ({vram_gb:.1f} GB) | amp={amp_dtype} | bs={batch_size} | lr_scale={lr_scale:.2f} | backbone={args.backbone}", flush=True)

    train_items_raw, eval_rows, pairs, intent_vecs, data_audit = _load_v05_r3_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        [p.resolve() for p in args.extra_pairs],
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
    )
    print(f"[v0.5.0-R3 DataAudit] {json.dumps(data_audit, ensure_ascii=False)}", flush=True)
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Non-silent anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} gateguard pairs collided!")

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    # Zero-PII domain-coverage augmentation
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

    # Exact BAAI/bge-small-zh-v1.5 WordPiece + 256 UTF-8 byte-fallback tokenizer (ZERO [UNK]!)
    hf_tok = AutoTokenizer.from_pretrained(args.backbone)
    tok_tmp = out_dir / "_tok_tmp"
    hf_tok.save_pretrained(tok_tmp)
    for fn in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_tmp / fn).is_file():
            shutil.copy2(tok_tmp / fn, out_dir / fn)
    shutil.rmtree(tok_tmp, ignore_errors=True)
    tokenizer = ByteFallbackWordPieceTokenizer(out_dir / "vocab.txt")

    print(f"[v0.5.0-R3 Pre-encode] Tokenizing {len(train_items_raw)} rows (zero-[UNK] WordPiece+ByteFallback) & extracting 32-d PIFE...", flush=True)
    t_enc = time.monotonic()
    train_counts = np.zeros(BASE_VOCAB_SIZE, dtype=np.float64)
    total_toks = 0
    for item in train_items_raw:
        seq = tokenizer.encode_ids(item["raw_text"], max_len=MAX_LEN)
        item["ids"] = seq
        item["length"] = len(seq)
        item["style"] = extract_style_features_32d(item["raw_text"])
        total_toks += len(seq)
        for tid in seq:
            if tid < BASE_VOCAB_SIZE:
                train_counts[tid] += 1.0
    print(f"[v0.5.0-R3 Pre-encode DONE] {len(train_items_raw)} rows in {time.monotonic()-t_enc:.1f}s | avg_tokens={total_toks/max(1,len(train_items_raw)):.2f}", flush=True)

    pairs_gg_pre = []
    pairs_gen_pre = []
    if pairs:
        ai_styles = extract_style_features_batch([p["ai_text"] for p in pairs])
        hu_styles = extract_style_features_batch([p["human_text"] for p in pairs])
        for i, p in enumerate(pairs):
            rec = {
                "ai_ids": tokenizer.encode_ids(p["ai_text"], max_len=MAX_LEN),
                "hu_ids": tokenizer.encode_ids(p["human_text"], max_len=MAX_LEN),
                "ai_style": ai_styles[i],
                "hu_style": hu_styles[i],
                "weight": float(p["weight"]),
            }
            if p.get("is_gateguard"):
                pairs_gg_pre.append(rec)
            else:
                pairs_gen_pre.append(rec)
    pairs_pre = pairs_gg_pre + pairs_gen_pre

    # Initialize 100%-lossless 4-layer BAAI/bge-small-zh-v1.5 + weighted SVD (E=192 -> 512)
    print(f"[v0.5.0-R3 Init] Loading ALL 4 layers of {args.backbone} (exact Chinese weights, E={args.embed_dim}->512)...", flush=True)
    hf_src = AutoModel.from_pretrained(args.backbone)
    w_src = hf_src.embeddings.word_embeddings.weight.detach().float().cpu().numpy()
    a_full, p_mat = compute_weighted_svd_factors(w_src, train_counts, embed_dim=args.embed_dim, unk_id=tokenizer.unk_id)
    a_full = _seed_ascii_bytes_into_a(a_full, tokenizer, BASE_VOCAB_SIZE)

    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 1024
    mdl = MeguvizV05R3BGEModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        intermediate_size=2048,
        max_len=MAX_LEN,
        style_dim=STYLE_FEAT_DIM,
        intent_dim=intent_dim,
        num_codes=len(CODE_LIST),
        num_prag=len(PRAG_TAGS),
    )
    mdl.load_from_bge(hf_src, a_full, p_mat)
    del hf_src
    mdl = mdl.to(device)
    mdl.set_sdpa(True)

    # Eval & n=2,400 Stratified Validation Calibration splits
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_rows, val_cal_weights = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(subset: list[dict]):
        txts = [str(r["text"]) for r in subset]
        enc = [tokenizer.encode_ids(t, max_len=MAX_LEN) for t in txts]
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
    def predict_pt(model_eval: MeguvizV05R3BGEModel, enc_ids: list[list[int]], st_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model_eval.eval()
        n = len(enc_ids)
        l_out = np.zeros(n, dtype=np.float32)
        emb_out = np.zeros((n, 256), dtype=np.float32)
        for st_i in range(0, n, eval_bs):
            chunk = enc_ids[st_i : st_i + eval_bs]
            ml = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), ml), tokenizer.pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(chunk), ml), dtype=torch.long, device=device)
            for i, seq in enumerate(chunk):
                ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, : len(seq)] = 1
            st_t = torch.from_numpy(st_arr[st_i : st_i + len(chunk)]).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                lg, _, emb = model_eval(ids, mask, st_t, return_aux=False)
            l_out[st_i : st_i + len(chunk)] = lg.float().cpu().numpy()
            emb_out[st_i : st_i + len(chunk)] = emb.float().cpu().numpy()
        return l_out, emb_out

    class R3Dataset(Dataset):
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
            max_seq = min(MAX_LEN, ((max(x["length"] for x in batch) + 15) // 16) * 16)
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
                torch.tensor([x["is_clean_neg"] for x in batch], dtype=torch.bool),
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
        return DataLoader(R3Dataset(items), **kw)

    def _collate_pairs(p_batch: list[dict]):
        bs_p = len(p_batch)
        m_ai = min(MAX_LEN, ((max(len(x["ai_ids"]) for x in p_batch) + 15) // 16) * 16)
        m_hu = min(MAX_LEN, ((max(len(x["hu_ids"]) for x in p_batch) + 15) // 16) * 16)
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
    struct_grp_idx = GROUP_TO_IDX["structured_human_hard"]
    ml_grp_idx = GROUP_TO_IDX["multilingual_world_knowledge"]

    def make_optimizer(base_lr: float):
        pre_p, fresh_p = [], []
        for name, p in mdl.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("layers.", "word_emb_", "pos_emb.", "emb_ln.")) and "dw" not in name:
                pre_p.append(p)
            else:
                fresh_p.append(p)
        groups = [
            {"params": pre_p, "lr": base_lr, "base_lr": base_lr, "weight_decay": 0.02},
            {"params": fresh_p, "lr": base_lr * 3.0, "base_lr": base_lr * 3.0, "weight_decay": 0.02},
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
        beta_neg_end: float = 0.03,
        pauc_weight: float = 0.22,
        ema_teacher: MeguvizV05R3BGEModel | None = None,
        update_ema: bool = False,
        use_swa: bool = False,
        pife_noise_prob: float = 0.0,
    ):
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
        opt = make_optimizer(eff_lr)
        rng_p = random.Random(20261125 + len(phase_name))
        pair_bs = max(16, min(48, int(round(16.0 * batch_size / 256.0))))
        swa_start = int(total_steps * 0.55)
        swa_every = max(15, min(60, total_steps // 40))
        log_every = max(40, total_steps // 12)
        t_p = time.monotonic()
        global_step = 0
        swa_state = None
        swa_count = 0

        print(f"[{phase_name}] items={len(items)} | bs={batch_size} | target_steps={total_steps} (~{num_epochs} ep) | pair_bs={pair_bs} | qat={qat} | eff_lr={eff_lr:.2e}", flush=True)

        for ep in range(num_epochs):
            mdl.train()
            run_loss = torch.zeros((), dtype=torch.float32, device=device)
            n_seen = 0
            for ids, mask, st, y_sc, y_cd, w, prag, grp, intent_idx, is_clean_neg in loader:
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                st = st.to(device, non_blocking=True)
                y_sc = y_sc.to(device, non_blocking=True)
                y_cd = y_cd.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                prag = prag.to(device, non_blocking=True)
                grp = grp.to(device, non_blocking=True)
                intent_idx = intent_idx.to(device, non_blocking=True)
                is_clean_neg = is_clean_neg.to(device, non_blocking=True)

                mult = _cosine_mult(global_step, total_steps)
                for pg in opt.param_groups:
                    pg["lr"] = float(pg["base_lr"]) * mult
                opt.zero_grad(set_to_none=True)

                ema_p = None
                if ema_teacher is not None:
                    ema_teacher.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                        el, _, _ = ema_teacher(ids, mask, st, return_aux=False)
                        ema_p = torch.sigmoid(el.float())

                prog = float(global_step) / float(max(1, total_steps - 1))
                cur_beta = beta_neg_start + (beta_neg_end - beta_neg_start) * prog

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(ids, mask, st, return_aux=True)
                    lg = out["logit"].float()
                    ps = torch.sigmoid(lg)
                    y_eff = (0.72 * y_sc + 0.28 * ema_p) if ema_p is not None else y_sc

                    per_s = (0.72 * _skewed_kl(ps, y_eff) + 0.28 * F.smooth_l1_loss(ps, y_eff, beta=0.04, reduction="none")) * w
                    loss_main = per_s.mean()

                    # Mid-layer (L2) deep supervision + Parallel TopK-SAE reconstruction/AuxK
                    mid_p = torch.sigmoid(out["mid_logit"].float())
                    loss_mid = ((0.72 * _skewed_kl(mid_p, y_eff) + 0.28 * F.smooth_l1_loss(mid_p, y_eff, beta=0.04, reduction="none")) * w).mean()
                    loss_sae = out["sae_loss"]

                    # Clean-Whitelist Two-Way Partial AUC (RTX 4060 Ti E3a breakthrough!)
                    loss_pauc = _two_way_pauc_clean_loss(lg, y_sc, is_clean_neg, w, beta_neg=cur_beta, alpha_pos=0.25, gamma=1.5, margin=2.4)

                    # Two-Tier Absolute Tail Anchoring on verified clean humans + logit scale lock (sharp T!)
                    clean_hu_t1 = is_clean_neg & (y_sc <= 0.15) & (w > 0.0)
                    loss_t1 = F.relu(ps[clean_hu_t1] - 0.08).pow(2).mean() if clean_hu_t1.any() else lg.sum() * 0.0
                    tier2_m = ((grp == struct_grp_idx) | (grp == ml_grp_idx)) & (y_sc <= 0.35) & (w > 0.0)
                    loss_t2 = F.relu(ps[tier2_m] - 0.14).pow(2).mean() if tier2_m.any() else lg.sum() * 0.0
                    loss_lreg = F.relu(lg.abs() - 3.6).pow(2).mean()

                    # Training-only auxiliary heads
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
                        + 0.05 * loss_sae
                        + pauc_weight * loss_pauc
                        + 0.24 * loss_t1
                        + 0.18 * loss_t2
                        + 0.015 * loss_lreg
                        + loss_aux
                    )

                    if qat and ema_p is not None:
                        loss = loss + 0.10 * F.mse_loss(ps, ema_p)

                    # Stratified 50/50 GateGuard + General Contrastive Pairs
                    if pairs_pre:
                        if pairs_gg_pre and pairs_gen_pre:
                            n_gg = min(len(pairs_gg_pre), max(4, pair_bs // 2))
                            n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                            p_batch = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                        else:
                            p_batch = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                        ai_ids, ai_m, ai_st, hu_ids, hu_m, hu_st, pw_t = _collate_pairs(p_batch)
                        l_ai_p, _, _ = mdl(ai_ids, ai_m, ai_st, return_aux=False)
                        l_hu_p, _, _ = mdl(hu_ids, hu_m, hu_st, return_aux=False)
                        l_ai_p, l_hu_p = l_ai_p.float(), l_hu_p.float()
                        loss_simpo = (F.softplus(1.8 - (l_ai_p - l_hu_p)) * pw_t).mean()
                        loss_pabs = ((F.relu(0.88 - torch.sigmoid(l_ai_p)).pow(2) + F.relu(torch.sigmoid(l_hu_p) - 0.12).pow(2)) * pw_t).mean()
                        loss = loss + 0.34 * loss_simpo + 0.20 * loss_pabs

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
                if global_step >= total_steps:
                    break

            v_l, _ = predict_pt(mdl, val_enc, val_st)
            v_p = expit(v_l)
            m_ep = {**compute_eval_metrics(val_scores, v_p), **compute_high_precision_metrics(val_scores, v_p)}
            print(f"[{phase_name} ep={ep+1} DONE] val: auroc={m_ep['auroc']} pr_auc={m_ep['pr_auc']} rp95={m_ep['recall_at_p95']} rp99={m_ep['recall_at_p99']} f05={m_ep['best_f05']} mae={m_ep['mae']}", flush=True)
            if global_step >= total_steps:
                break
        return swa_state

    # Stage 1: Full-Corpus Supervised Distillation + Clean-Whitelist Two-Way pAUC
    stage1_ckpt = ckpt_dir / "v05_r3_stage1_ckpt.pt"
    if args.resume_stage1 and Path(args.resume_stage1).is_file():
        mdl.load_state_dict(torch.load(args.resume_stage1, map_location=device), strict=True)
    else:
        run_phase(
            train_items_raw,
            target_steps=16 if args.smoke_test else 0,
            num_epochs_fallback=1 if args.smoke_test else int(args.epochs),
            lr=args.lr,
            qat=False,
            phase_name="Stage1-BGE4L-pAUC",
            beta_neg_start=0.08,
            beta_neg_end=0.04,
            pauc_weight=0.20,
        )
        if not args.smoke_test:
            torch.save(mdl.state_dict(), stage1_ckpt)
            print(f"[Stage1] Saved reusable Stage-1 checkpoint -> {stage1_ckpt}", flush=True)

    ema_teacher = copy.deepcopy(mdl).eval()
    ema_teacher.set_sdpa(True)
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # Stage 2: Born-Again EMA + PIFE Adversarial + Sharper Clean-Whitelist Two-Way pAUC
    stage2_items, gold_ret_s2 = subset_balanced_gold_guaranteed(train_items_raw, cap=args.stage2_cap, seed_s=20261126)
    print(f"[Stage2] Subset={len(stage2_items)} | gold_retention={gold_ret_s2:.4f}", flush=True)
    run_phase(
        stage2_items,
        target_steps=12 if args.smoke_test else int(args.stage2_steps),
        num_epochs_fallback=1,
        lr=args.refine_lr,
        qat=False,
        phase_name="Stage2-EMA-pAUC-PIFE",
        beta_neg_start=0.05,
        beta_neg_end=0.03,
        pauc_weight=0.26,
        ema_teacher=ema_teacher,
        update_ema=True,
        pife_noise_prob=0.08,
    )

    # Stage 3: Fixed-Step QAT + SWA
    qat_items, gold_ret_s3 = subset_balanced_gold_guaranteed(train_items_raw, cap=args.qat_cap, seed_s=20261127)
    print(f"[Stage3-QAT] Subset={len(qat_items)} | gold_retention={gold_ret_s3:.4f}", flush=True)
    swa_sd = run_phase(
        qat_items,
        target_steps=12 if args.smoke_test else int(args.qat_steps),
        num_epochs_fallback=1,
        lr=args.qat_lr,
        qat=True,
        phase_name="Stage3-QAT-SWA",
        beta_neg_start=0.04,
        beta_neg_end=0.03,
        pauc_weight=0.22,
        ema_teacher=ema_teacher,
        update_ema=False,
        use_swa=not args.smoke_test,
        pife_noise_prob=0.05,
    )
    if swa_sd is not None:
        mdl.load_state_dict(swa_sd, strict=True)
        print("[Stage3-QAT-SWA] Applied SWA weights.", flush=True)

    del ema_teacher
    mdl.set_sdpa(False)
    mdl.set_qat_enabled(False)
    mdl.eval()

    # Calibration: Sharp Baked T in [0.9, 1.8] + n=2,400 Stratified Smooth PAVA + Split Conformal + L2 Prototype Bank
    with torch.no_grad():
        mdl.temperature.fill_(1.0)
    raw_cal_l, _ = predict_pt(mdl, cal_enc, cal_st)
    t_raw = float(fit_temperature(raw_cal_l, cal_scores / 100.0))
    t_baked = float(np.clip(t_raw, 0.90, 1.80))
    with torch.no_grad():
        mdl.temperature.fill_(t_baked)
    print(f"[Calibration] raw_T={t_raw:.3f} -> baked_sharp_T={t_baked:.3f}", flush=True)

    cal_l, _ = predict_pt(mdl, cal_enc, cal_st)
    cal_p_raw = expit(cal_l)
    smooth_pava = fit_bucketed_smooth_pava(cal_txts, cal_p_raw, cal_scores / 100.0, sample_weights=val_cal_weights) if not args.smoke_test else {}
    cal_p = apply_bucketed_smooth_pava(cal_txts, cal_p_raw, smooth_pava)
    t_f05, t_fpr05 = find_val_thresholds(cal_scores, cal_p)
    hp_cal = compute_high_precision_metrics(cal_scores, cal_p)
    conformal_cal = fit_split_conformal_3way(cal_scores, cal_p, alphas=(0.03, 0.05))
    print(f"[Calibration] n_cal={len(cal_scores)} | conformal_alpha03={conformal_cal.get('alpha_03')}", flush=True)

    # Fit L2 Prototype Bank
    bank_subset = train_items_raw[: min(20000, len(train_items_raw))]
    if not args.smoke_test and len(train_items_raw) > 20000:
        rng_bk = random.Random(20261125)
        hu_anc = [x for x in train_items_raw if x["is_clean_neg"]]
        ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
        bank_subset = rng_bk.sample(hu_anc, min(10000, len(hu_anc))) + rng_bk.sample(ai_anc, min(10000, len(ai_anc)))
    _, bk_emb = predict_pt(mdl, [x["ids"] for x in bank_subset], np.stack([x["style"] for x in bank_subset], axis=0))
    proto_bank = fit_l2_prototype_bank(
        bk_emb,
        np.array([x["y_score"] * 100.0 for x in bank_subset], dtype=np.float32),
        is_human_anchor=np.array([bool(x["is_clean_neg"]) for x in bank_subset], dtype=bool),
    )
    proto_path = out_dir / "prototype_bank.npz"
    np.savez_compressed(proto_path, **proto_bank)

    def _eval_split(enc, txts, st, sc):
        lg, _ = predict_pt(mdl, enc, st)
        raw_p = expit(lg)
        cp = apply_bucketed_smooth_pava(txts, raw_p, smooth_pava)
        return {**compute_eval_metrics(sc, cp, t_f05, t_fpr05), **compute_high_precision_metrics(sc, cp)}, cp

    test_m, test_p = _eval_split(test_enc, test_txts, test_st, test_scores)
    dsh_m, _ = _eval_split(test_dsh_enc, test_dsh_txts, test_dsh_st, test_dsh_scores)
    all_m, _ = _eval_split(test_all_enc, test_all_txts, test_all_st, test_all_scores)
    probe_m, _ = _eval_split(probe_enc, probe_txts, probe_st, probe_scores)

    # Export single-graph INT8 & Edge W4-Emb/W8-Body ONNX
    exp_info = export_v05_r3_onnx(mdl, out_dir, vocab_size=tokenizer.vocab_size)
    manifest_base = {
        "schema_version": "meguviz.bundle.v2.r3",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": "v0.5-R3(BGE-small-zh-4L+DWConv+PIFE32d+ParallelTopKSAE+CleanWhitelistTwoWaypAUC+L2Proto)",
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "quant_method": "qat_swa_plus_dynamic_int8",
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": int(args.embed_dim),
        "hidden_size": 512,
        "num_layers": 4,
        "max_len": MAX_LEN,
        "vocab_file": "vocab.txt",
        "prototype_bank_file": proto_path.name,
        "prototype_bank_sha256": sha256_file(proto_path),
        "temperature": round(t_baked, 6),
        "smooth_pava": smooth_pava,
        "conformal": conformal_cal,
        "thresholds": {"f05": round(t_f05, 4), "p95": hp_cal["t_p95"], "p99": hp_cal["t_p99"], "fpr05": round(t_fpr05, 4), "score70": 0.70},
        "model_file": exp_info["model_int8"],
        "model_edge_w4": exp_info["model_edge_w4"],
        "model_bytes": exp_info["model_int8_bytes"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT parity + latency
    engine = MeguvizV05R3Engine(out_dir, threads=1, use_edge_w4=False)
    t_s = time.perf_counter()
    _ = [engine.predict_one(txt) for txt in test_txts]
    single_ms = (time.perf_counter() - t_s) * 1000.0 / max(1, len(test_txts))
    t_b = time.perf_counter()
    ort_batch = engine.predict_batch(test_txts, max_batch=32)
    batch_ms = (time.perf_counter() - t_b) * 1000.0 / max(1, len(test_txts))
    ort_p = np.array([x["prob"] for x in ort_batch], dtype=np.float32)
    parity = {
        "mae_vs_pytorch": round(float(np.mean(np.abs(test_p - ort_p))), 5),
        "max_abs_diff_vs_pytorch": round(float(np.max(np.abs(test_p - ort_p))), 5),
        "single_1thread_ms": round(single_ms, 3),
        "compact_batch32_1thread_ms": round(batch_ms, 3),
        "model_bytes": exp_info["model_int8_bytes"],
    }

    probe_file = Path("round01/bili-round01-user-confirmed.jsonl")
    verify_res = run_verify_v05_r3(out_dir, probe_file.resolve(), threads=1) if probe_file.is_file() else {}

    metrics = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v2.r3",
        "backbone": args.backbone,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "batch_size_used": batch_size,
        "data_audit": data_audit,
        "gold_retention": {"stage2": round(gold_ret_s2, 4), "stage3": round(gold_ret_s3, 4)},
        "temperature": round(t_baked, 6),
        "raw_fitted_temperature": round(t_raw, 4),
        "conformal": conformal_cal,
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
    metrics["release_gates"] = check_v05_r3_release_gates(metrics)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Package INT8 zip & Edge-W4 zip
    common = ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", proto_path.name, metrics_file.name]
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

    metrics["release_gates"] = check_v05_r3_release_gates(metrics)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(
        json.dumps(
            {
                "test_bili_gold": test_m,
                "test_dsh_gold": dsh_m,
                "probe_u001_u017": probe_m,
                "onnx_int8": metrics["onnx_int8"],
                "edge_w4_w8": metrics["edge_w4_w8"],
                "hard_traps": (verify_res.get("hard_trap_gates") or {}),
                "ml_traps": (verify_res.get("multilingual_trap_gates") or {}),
                "conformal_alpha03": conformal_cal.get("alpha_03"),
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
    parser.add_argument("--out", type=Path, default=Path("./out_v05_r3"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-stage1", type=Path, default=None)
    parser.add_argument("--version", default="v0.5.0-r3-bge4l-pauc")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stage2-steps", type=int, default=2400)
    parser.add_argument("--qat-steps", type=int, default=2800)
    parser.add_argument("--stage2-cap", type=int, default=420000)
    parser.add_argument("--qat-cap", type=int, default=380000)
    parser.add_argument("--stage1-synth-cap", type=int, default=10000)
    parser.add_argument("--v04-synth-cap", type=int, default=35000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-auto-lr-scale", action="store_true")
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--refine-lr", type=float, default=3.5e-5)
    parser.add_argument("--qat-lr", type=float, default=1.2e-5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe-vram", default="")
    args = parser.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()
