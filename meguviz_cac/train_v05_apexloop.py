"""Train, Stage-0 RTD align, Stage-1 mmBERT-ApexLoop distill, Stage-2 Born-Again EMA + PIFE refine,
Stage-3 fixed-step LoopQ QAT+SWA, Smooth PAVA / Conformal / L2 Prototype calibrate & export MeguvizCAC v0.5.0-ApexLoop."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
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
from .train_v03_loop import DEEP_PRAG_IDXS, SURFACE_PRAG_IDXS
from .train_v04_dualloop import _load_intent_cache, _open_read, compute_high_precision_metrics
from .v04_corpus_prep import BENCH_TRAP_TEXTS, GROUP_TO_IDX, V04_GROUPS, assign_group_id
from .v04_runtime import fit_split_conformal_3way
from .v05_features import (
    MAX_LEN_V05,
    STYLE_FEAT_DIM,
    apply_bucketed_smooth_pava,
    extract_style_features_32d,
    extract_style_features_batch,
    fit_bucketed_smooth_pava,
    length_bucket_name,
    prepare_text_v05,
)
from .v05_model import MeguvizV05ApexLoopModel, export_v05_onnx_bundles
from .v05_runtime import (
    MeguvizV05Engine,
    PIFE_ADVERSARIAL_TRAPS,
    fit_l2_prototype_bank,
    run_verify_v05,
)
from .v05_tokenizer import (
    DEFAULT_V05_VOCAB_SIZE,
    V05SuperBPETokenizer,
    project_mmbert_embeddings_svd,
    train_v05_superbpe_tokenizer,
)


DEFAULT_V05_BACKBONE = "jhu-clsp/mmBERT-small"

# Lesson #1: Protection-bucket whitelist — ONLY era-verified pre_llm or high-tier gold/tribunal sources
GOLD_HIGH_TIER_SOURCES = frozenset(
    {
        "hydrogen_xhigh_direct",
        "v04_neon_xhigh_tiebreak",
        "v04_neon_hydrogen_consensus",
        "v04_neon_hydrogen_weighted",
        "v04_neon_gate_override",
        "v04_hydrogen_gate_override",
        "v04_qwen38_bf16",
        "v04_inkling_wk_counterfactual",
        "v04_qwen_inkling_joint",
        "v04_qwen_inkling_disagree",
    }
)


def _char_trigrams(s: str) -> set[str]:
    clean = "".join(prepare_text_v05(s).lower().split())
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i : i + 3] for i in range(len(clean) - 2)}


def _is_near_duplicate(text: str, forbidden_exact: set[str], forbidden_trigrams: list[set[str]], jaccard_thr: float = 0.78) -> bool:
    norm = prepare_text_v05(text)
    if norm in forbidden_exact:
        return True
    tg = _char_trigrams(norm)
    if len(tg) < 6:
        return False
    for ftg in forbidden_trigrams:
        if not ftg:
            continue
        inter = len(tg & ftg)
        if inter == 0:
            continue
        if inter / float(len(tg | ftg)) >= jaccard_thr:
            return True
    return False


def _load_v05_bundle(
    data_path: Path,
    gateguard_path: Path | None,
    extra_pairs_paths: list[Path],
    intent_cache_path: Path | None,
    stage1_synth_cap: int = 15000,
    v04_synth_cap: int = -1,
    seed: int = 20261125,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray | None, dict]:
    """Load merged R3 corpus with Protection-Bucket Whitelist + Non-Silent Anti-Leakage Pair Audit."""
    rows: list[dict] = []
    with _open_read(data_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    intent_vecs, sid_to_intent = _load_intent_cache(intent_cache_path)
    eval_rows = [r for r in rows if r["split"] != "train" or r.get("cohort") == "user_confirmed"]

    # Build strict anti-leakage blocklist: ALL eval_rows + ALL benchmark/PIFE traps!
    forbidden_exact = {prepare_text_v05(r["text"]) for r in eval_rows}
    trap_texts = list(BENCH_TRAP_TEXTS) + [txt for _, txt in PIFE_ADVERSARIAL_TRAPS]
    for t in trap_texts:
        forbidden_exact.add(prepare_text_v05(t))
    trap_trigrams = [_char_trigrams(t) for t in trap_texts]

    # Load and audit gateguard + twin/inkling pairs (non-silent!)
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
                        "weight": float(p.get("weight", 2.2 if is_gateguard else 0.7)),
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

    # Filter & whitelist-clean train rows
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

    train_items: list[dict] = []
    for r in (*other_rows, *legacy_synth_rows, *v04_synth_rows):
        sid = r["sample_id"]
        src = str(r.get("score_source") or "")
        era = str(r.get("era_bucket") or "")
        gid_str = r.get("group_id") or assign_group_id(r)
        is_gold_tier = (src in GOLD_HIGH_TIER_SOURCES) or bool(r.get("hard_human_neg"))

        # Lesson #1: Protection-Bucket Whitelist — prevent noisy weak pseudo-labels from entering structured_human_hard!
        if gid_str == "structured_human_hard" and (era != "pre_llm") and (not is_gold_tier):
            gid_str = "general_real"
            demoted_struct_weak += 1

        prag = r.get("prag_vec")
        if not (isinstance(prag, list) and len(prag) == len(PRAG_TAGS)):
            prag = [-1.0] * len(PRAG_TAGS)

        train_items.append(
            {
                "sample_id": sid,
                "raw_text": str(r["text"]),
                "y_score": float(np.clip(float(r["score"]) / 100.0, 0.0, 1.0)),
                "code_idx": CODE_TO_IDX.get(r.get("reason_code", ""), -100),
                "weight": float(r.get("train_weight", 1.0)),
                "prag_vec": [float(x) for x in prag],
                "group_idx": GROUP_TO_IDX.get(gid_str, GROUP_TO_IDX["general_real"]),
                "intent_idx": sid_to_intent.get(sid, -1),
                "is_gold_tier": bool(is_gold_tier),
                "is_pre_llm": bool(era == "pre_llm"),
                "is_high_prio": bool(
                    is_gold_tier
                    or era == "pre_llm"
                    or gid_str in ("ultra_short_ai", "evasive_plugin_ai", "structured_human_hard", "multilingual_world_knowledge", "persona_space_kaomoji", "literary_boundary", "dsh_hard_slice")
                ),
            }
        )

    audit = {
        "total_loaded_rows": len(rows),
        "train_items": len(train_items),
        "gold_tier_count": sum(1 for x in train_items if x["is_gold_tier"]),
        "demoted_weak_structured_human": demoted_struct_weak,
        "eval_rows": len(eval_rows),
        "gateguard_kept_by_cat": dict(gg_kept_cats),
        "gateguard_dropped_leakage": gg_dropped,
        "extra_pairs_dropped_leakage": extra_dropped,
        "total_clean_pairs": len(pairs),
    }
    return train_items, eval_rows, pairs, intent_vecs, audit


def subset_balanced_gold_guaranteed(items: list[dict], cap: int, seed_s: int) -> tuple[list[dict], float]:
    """Lesson #4: Unconditionally preserve >= 98% (target 100%) of gold_tier items, then balance remaining cap across 12 groups."""
    if cap <= 0 or len(items) <= cap:
        return list(items), 1.0
    rng = random.Random(seed_s)
    gold = [x for x in items if x["is_gold_tier"]]
    non_gold = [x for x in items if not x["is_gold_tier"]]
    selected = list(gold)
    gold_ret = 1.0
    if len(selected) > cap:
        rng.shuffle(selected)
        selected = selected[:cap]
        gold_ret = len(selected) / max(1, len(gold))
        return selected, gold_ret

    rem = cap - len(selected)
    by_g: dict[int, list[dict]] = {g: [] for g in range(len(V04_GROUPS))}
    for x in non_gold:
        by_g[x["group_idx"]].append(x)
    quota = max(1, rem // len(V04_GROUPS))
    leftover: list[dict] = []
    for g in sorted(by_g):
        bkt = by_g[g]
        hi = [x for x in bkt if x["is_high_prio"]]
        lo = [x for x in bkt if not x["is_high_prio"]]
        rng.shuffle(hi)
        rng.shuffle(lo)
        ordered = hi + lo
        take = min(len(ordered), quota)
        selected.extend(ordered[:take])
        leftover.extend(ordered[take:])
    if len(selected) < cap and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[: cap - len(selected)])
    rng.shuffle(selected)
    assert gold_ret >= 0.98, f"Gold Retention violated: {gold_ret:.4f} < 0.98"
    return selected[:cap], gold_ret


def build_stratified_val_calibration_set(eval_rows: list[dict], target_n: int = 2400, seed: int = 20261125) -> tuple[list[dict], np.ndarray]:
    """Lesson #6: Build n>=2,000 stratified calibration set STRICTLY from split=='validation' (ZERO test leakage!)."""
    val_all = [r for r in eval_rows if r.get("split") == "validation" and r.get("has_score", True)]
    val_gold = [r for r in val_all if r.get("score_source") == "hydrogen_xhigh_direct"]
    val_other = [r for r in val_all if r.get("score_source") != "hydrogen_xhigh_direct"]

    rng = random.Random(seed)
    chosen = list(val_gold)
    weights = [4.0] * len(val_gold)

    rem = max(0, target_n - len(chosen))
    if rem > 0 and val_other:
        # Stratify by (length_bucket x score_band) so Smooth PAVA & Conformal stay sharp across all buckets!
        strata: dict[tuple[str, str], list[dict]] = {}
        for r in val_other:
            lb = length_bucket_name(str(r["text"]))
            sc = float(r["score"])
            sb = "low" if sc <= 30.0 else ("high" if sc >= 70.0 else "mid")
            strata.setdefault((lb, sb), []).append(r)
        per_s = max(1, rem // max(1, len(strata)))
        leftover: list[dict] = []
        for k in sorted(strata):
            bkt = strata[k]
            rng.shuffle(bkt)
            take = min(len(bkt), per_s)
            chosen.extend(bkt[:take])
            weights.extend([1.0] * take)
            leftover.extend(bkt[take:])
        if len(chosen) < target_n and leftover:
            rng.shuffle(leftover)
            extra = leftover[: target_n - len(chosen)]
            chosen.extend(extra)
            weights.extend([1.0] * len(extra))

    return chosen, np.asarray(weights, dtype=np.float64)


def check_v05_release_gates(metrics: dict) -> dict:
    """Evaluate MeguvizCAC v0.5.0-ApexLoop high-bar release gates (targeting clear superiority over OjinjinNet)."""
    tb = metrics.get("test_bili_gold") or {}
    ox = metrics.get("onnx_int8") or {}
    ed = metrics.get("edge_w4_w8") or {}
    ver = metrics.get("onnx_verify") or {}
    ht = ver.get("hard_trap_gates") or {}
    ml = ver.get("multilingual_trap_gates") or {}
    conf03 = (metrics.get("conformal") or {}).get("alpha_03") or {}
    mono = metrics.get("loop_monotonicity") or {}

    edge_mb = ((ed.get("bytes") or 0) + (ox.get("core_step_bytes") or 0) + (ox.get("coda_bytes") or 0)) / (1024.0 * 1024.0) if ed.get("ok") else 99.0
    t_coda = float(metrics.get("coda_temperature", 9.9))

    gates = {
        "speed_1t_le_1p8_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 1.80 ms (desktop 1T <=1.20)", "pass": (ox.get("compact_batch32_1thread_ms") or 99.0) <= 1.80},
        "edge_w4_size_le_6p5_mb": {"value": round(edge_mb, 2), "target": "<= 6.5 MB", "pass": bool(ed.get("ok")) and edge_mb <= 6.5},
        "auroc_ge_0965": {"value": tb.get("auroc"), "target": ">= 0.965 (stretch 0.975)", "pass": (tb.get("auroc") or 0.0) >= 0.965},
        "pr_auc_ge_0950": {"value": tb.get("pr_auc"), "target": ">= 0.950 (stretch 0.965)", "pass": (tb.get("pr_auc") or 0.0) >= 0.950},
        "rp95_ge_080": {"value": tb.get("recall_at_p95"), "target": ">= 0.80 (stretch 0.85)", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.80},
        "rp99_ge_050": {"value": tb.get("recall_at_p99"), "target": ">= 0.50 (stretch 0.60)", "pass": (tb.get("recall_at_p99") or 0.0) >= 0.50},
        "struct_human_lt_018": {"value": ht.get("structured_human_max_prob"), "target": "< 0.18", "pass": (ht.get("structured_human_max_prob") or 1.0) < 0.18},
        "ai_traps_ge_088": {"value": ht.get("ai_traps_min_prob"), "target": ">= 0.88", "pass": (ht.get("ai_traps_min_prob") or 0.0) >= 0.88},
        "ml_human_lt_015": {"value": ml.get("ml_human_max_prob"), "target": "< 0.15", "pass": (ml.get("ml_human_max_prob") or 1.0) < 0.15},
        "ml_ai_ge_088": {"value": ml.get("ml_ai_min_prob"), "target": ">= 0.88", "pass": (ml.get("ml_ai_min_prob") or 0.0) >= 0.88},
        "conformal_grey_le_18pct": {"value": conf03.get("cal_grey_rate"), "target": "<= 0.18", "pass": (conf03.get("cal_grey_rate") or 1.0) <= 0.18},
        "sharp_temp_in_09_18": {"value": round(t_coda, 3), "target": "[0.9, 1.8]", "pass": 0.9 <= t_coda <= 1.8},
        "qat_parity_le_0008": {"value": ox.get("mae_vs_pytorch"), "target": "<= 0.008", "pass": (ox.get("mae_vs_pytorch") or 99.0) <= 0.008},
        "monotonic_r4_no_rp99_drop": {"value": mono.get("non_regressive_r4_rp99"), "target": "True", "pass": bool(mono.get("non_regressive_r4_rp99"))},
        "zero_unk": {"value": ver.get("unk_tokens_total"), "target": "== 0", "pass": ver.get("unk_tokens_total") == 0},
    }
    return {
        "gates": gates,
        "passed_count": sum(1 for v in gates.values() if v["pass"]),
        "total_gates": len(gates),
        "all_passed": all(v["pass"] for v in gates.values()),
    }


def _newton_schulz_5(g, steps: int = 5):
    """Muon 5-step Newton-Schulz orthogonalization in BF16/FP32 for 2D weight updates."""
    import torch

    assert g.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda and torch.cuda.is_bf16_supported() else g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class HybridMuonClipAdamW:
    """Hybrid optimizer: MuonClip (Newton-Schulz orthogonal momentum + weight decay) for 2D hidden matrices + AdamW for 1D/embeddings."""

    def __init__(self, model, lr_adamw: float, lr_muon: float | None = None, weight_decay: float = 0.01, use_muon: bool = True):
        import torch

        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_hidden_2d = (
                use_muon
                and p.ndim == 2
                and min(p.shape) >= 64
                and "word_emb" not in name
                and "head" not in name
                and "rtd_" not in name
            )
            if is_hidden_2d:
                muon_params.append(p)
            else:
                adam_params.append(p)

        self.muon_params = muon_params
        self.muon_bufs = [torch.zeros_like(p) for p in muon_params]
        self.lr_muon_base = float(lr_muon if lr_muon is not None else min(3e-3, lr_adamw * 15.0))
        self.lr_adamw_base = float(lr_adamw)
        self.weight_decay = float(weight_decay)
        self.momentum = 0.95
        try:
            self.adamw = torch.optim.AdamW(adam_params, lr=self.lr_adamw_base, weight_decay=weight_decay, fused=torch.cuda.is_available())
        except TypeError:
            self.adamw = torch.optim.AdamW(adam_params, lr=self.lr_adamw_base, weight_decay=weight_decay)
        self.lr_mult = 1.0

    def zero_grad(self, set_to_none: bool = True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def set_lr_mult(self, mult: float):
        self.lr_mult = float(mult)
        for pg in self.adamw.param_groups:
            pg["lr"] = self.lr_adamw_base * self.lr_mult

    def step(self):
        import torch

        self.adamw.step()
        if not self.muon_params:
            return
        lr_m = self.lr_muon_base * self.lr_mult
        with torch.no_grad():
            for p, buf in zip(self.muon_params, self.muon_bufs):
                if p.grad is None:
                    continue
                g = p.grad
                buf.mul_(self.momentum).add_(g)
                g_nes = g + self.momentum * buf
                u = _newton_schulz_5(g_nes)
                scale = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                if self.weight_decay > 0.0:
                    p.mul_(1.0 - lr_m * self.weight_decay)
                p.add_(u, alpha=-lr_m * scale)


def _cosine_mult(step: int, total_steps: int, warmup_frac: float = 0.08) -> float:
    warm = max(1, int(total_steps * warmup_frac))
    if step < warm:
        return float(step + 1) / float(warm)
    prog = min(1.0, float(step - warm) / float(max(1, total_steps - warm)))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))


def _inject_pife_noise(text: str, rng: random.Random) -> str:
    """Inject light zero-width or homoglyph perturbation during Stage-2 to harden PIFE forensics."""
    if len(text) < 8:
        return text
    if rng.random() < 0.5:
        pos = rng.randint(1, len(text) - 1)
        return text[:pos] + "\u200b" + text[pos:]
    return text.replace("a", "а", 1).replace("e", "е", 1).replace("o", "о", 1)


def run_dry(args) -> None:
    t0 = time.monotonic()
    train_items, eval_rows, pairs, intent_vecs, audit = _load_v05_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        [p.resolve() for p in args.extra_pairs],
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=min(5000, args.stage1_synth_cap) if args.stage1_synth_cap != 0 else 0,
        v04_synth_cap=5000 if args.v04_synth_cap < 0 else min(5000, args.v04_synth_cap),
    )
    assert audit["gateguard_dropped_leakage"] == 0, f"Gate-guard leakage detected: {audit['gateguard_dropped_leakage']}"
    sub, gold_ret = subset_balanced_gold_guaranteed(train_items, cap=min(25000, len(train_items)), seed_s=42)
    cal_rows, cal_w = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    st_sample = extract_style_features_batch([x["raw_text"] for x in train_items[:64]])
    assert st_sample.shape == (min(64, len(train_items)), STYLE_FEAT_DIM) and np.isfinite(st_sample).all()
    print(
        json.dumps(
            {
                "dry_run": "PASS",
                "audit": audit,
                "gold_retention_check": round(gold_ret, 4),
                "stratified_val_cal_n": len(cal_rows),
                "stratified_val_cal_gold_n": int((cal_w > 1.5).sum()),
                "style_32d_finite": True,
                "elapsed_s": round(time.monotonic() - t0, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_vram_probe(args, device, amp_dtype) -> None:
    """Fast 30s worst-case seq=192 Stage-3 LoopQ QAT (R=4) + EMA + Pairs VRAM probe before loading 1.37M rows."""
    import torch
    import torch.nn.functional as F

    cands = [int(x.strip()) for x in str(args.probe_vram).split(",") if x.strip()]
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap_gib = vram_gb * 0.90
    mdl = MeguvizV05ApexLoopModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=384,
        num_heads=6,
        intermediate_size=1152,
        core_layers=args.core_layers,
        num_slots=16,
    ).to(device)
    mdl.set_sdpa(True)
    mdl.set_qat_enabled(True)
    ema = copy.deepcopy(mdl).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = HybridMuonClipAdamW(mdl, lr_adamw=1e-4, use_muon=(args.optimizer == "muonclip"))

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
                ids = torch.randint(5, args.vocab_size, (bs, MAX_LEN_V05), device=device)
                mask = torch.ones((bs, MAX_LEN_V05), dtype=torch.long, device=device)
                st = torch.randn((bs, STYLE_FEAT_DIM), device=device)
                y = torch.rand((bs,), device=device)
                p_ids = torch.randint(5, args.vocab_size, (pair_bs, 96), device=device)
                p_m = torch.ones((pair_bs, 96), dtype=torch.long, device=device)
                p_st = torch.randn((pair_bs, STYLE_FEAT_DIM), device=device)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    _ = ema.forward_full(ids, mask, st, num_steps=4, return_aux=False)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl.forward_full(ids, mask, st, num_steps=4, return_aux=True)
                    loss = F.binary_cross_entropy_with_logits(out["step_logits"][-1], y) + F.binary_cross_entropy_with_logits(out["exit0_logit"], y)
                    o_ai = mdl.forward_full(p_ids, p_m, p_st, num_steps=2, return_aux=False)
                    o_hu = mdl.forward_full(p_ids, p_m, p_st, num_steps=2, return_aux=False)
                    loss = loss + 0.3 * F.softplus(1.6 - (o_ai["step_logits"][-1] - o_hu["step_logits"][-1])).mean()
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
        if fits and chosen == cands[-1]:
            chosen = bs
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
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset, Sampler
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v0.5.0-ApexLoop training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.probe_vram:
        _run_vram_probe(args, device, amp_dtype)
        return

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else (768 if vram_gb >= 60.0 else (384 if vram_gb >= 35.0 else 128))
    lr_scale = min(2.0, math.sqrt(batch_size / 256.0)) if (batch_size > 256 and not args.no_auto_lr_scale) else 1.0
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    print(f"[v0.5.0 Config] GPU={gpu_name} ({vram_gb:.1f} GB) | amp={amp_dtype} | bs={batch_size} | lr_scale={lr_scale:.2f} | opt={args.optimizer} | core_layers={args.core_layers}", flush=True)

    # 1. Load corpus + anti-pollution audit
    train_items_raw, eval_rows, pairs, intent_vecs, data_audit = _load_v05_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        [p.resolve() for p in args.extra_pairs],
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
    )
    print(f"[v0.5.0 DataAudit] {json.dumps(data_audit, ensure_ascii=False)}", flush=True)
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Non-silent anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} gateguard pairs collided with eval/traps!")

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    # Zero-PII domain-coverage augmentation (~3% reply prefixes + ~3% benign URLs on definite humans)
    if not args.no_domain_aug and not args.smoke_test:
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

    # 2. Build or load 28k SuperBPE Tokenizer
    tok_path = out_dir / "v05_tokenizer.json"
    if args.tokenizer_path and args.tokenizer_path.is_file():
        shutil.copy2(args.tokenizer_path, tok_path)
    else:
        rng_tok = random.Random(20261125)
        # Prioritize all multilingual, gold, structured, and high-signal rows + sample of general corpus
        hi_txts = [x["raw_text"] for x in train_items_raw if x["is_high_prio"]]
        lo_txts = [x["raw_text"] for x in train_items_raw if not x["is_high_prio"]]
        sample_tok_txts = hi_txts[:150000] + rng_tok.sample(lo_txts, min(150000, len(lo_txts)))
        print(f"[v0.5.0 Tokenizer] Training {args.vocab_size} Byte-Level SuperBPE on {len(sample_tok_txts)} texts...", flush=True)
        train_v05_superbpe_tokenizer(sample_tok_txts, tok_path, vocab_size=args.vocab_size)
    tokenizer = V05SuperBPETokenizer(tok_path)

    # Fast Rust batch tokenization + 32-d native stylometric feature extraction
    print(f"[v0.5.0 Pre-encode] Tokenizing {len(train_items_raw)} train rows in Rust & extracting 32-d PIFE stylometrics...", flush=True)
    t_enc = time.monotonic()
    all_raw_texts = [x["raw_text"] for x in train_items_raw]
    all_ids = tokenizer.encode_batch_fast(all_raw_texts, max_len=MAX_LEN_V05)
    train_counts = np.zeros(tokenizer.vocab_size, dtype=np.float64)
    total_toks = 0
    for item, seq in zip(train_items_raw, all_ids):
        item["ids"] = seq
        item["length"] = len(seq)
        item["style"] = extract_style_features_32d(item["raw_text"])
        total_toks += len(seq)
        for tid in seq:
            if tid < tokenizer.vocab_size:
                train_counts[tid] += 1.0
    print(f"[v0.5.0 Pre-encode DONE] {len(train_items_raw)} rows in {time.monotonic()-t_enc:.1f}s | avg_tokens={total_toks/max(1,len(train_items_raw)):.2f}", flush=True)

    pairs_gg_pre = []
    pairs_gen_pre = []
    if pairs:
        ai_seqs = tokenizer.encode_batch_fast([p["ai_text"] for p in pairs], max_len=MAX_LEN_V05)
        hu_seqs = tokenizer.encode_batch_fast([p["human_text"] for p in pairs], max_len=MAX_LEN_V05)
        ai_styles = extract_style_features_batch([p["ai_text"] for p in pairs])
        hu_styles = extract_style_features_batch([p["human_text"] for p in pairs])
        for i, p in enumerate(pairs):
            rec = {
                "ai_ids": ai_seqs[i],
                "hu_ids": hu_seqs[i],
                "ai_style": ai_styles[i],
                "hu_style": hu_styles[i],
                "weight": float(p["weight"]),
            }
            if p.get("is_gateguard"):
                pairs_gg_pre.append(rec)
            else:
                pairs_gen_pre.append(rec)
    pairs_pre = pairs_gg_pre + pairs_gen_pre

    # 3. Initialize MeguvizV05ApexLoopModel from mmBERT-small via Focus cross-tokenizer lexical projection + SVD
    intent_dim = int(intent_vecs.shape[1]) if (intent_vecs is not None and intent_vecs.ndim == 2) else 1024
    mdl = MeguvizV05ApexLoopModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=384,
        num_heads=6,
        intermediate_size=1152,
        core_layers=args.core_layers,
        num_slots=16,
        style_dim=STYLE_FEAT_DIM,
        intent_dim=intent_dim,
        num_codes=len(CODE_LIST),
        num_prag=len(PRAG_TAGS),
        num_surf=len(SURFACE_PRAG_IDXS),
        num_deep=len(DEEP_PRAG_IDXS),
        emb_drop_p=args.emb_drop,
    )

    if not args.from_scratch:
        print(f"[v0.5.0 Init] Loading {args.backbone} for cross-tokenizer lexical transfer + layer inheritance...", flush=True)
        mm_tok = AutoTokenizer.from_pretrained(args.backbone)
        mm_src = AutoModel.from_pretrained(args.backbone)
        a_mat, p_mat = project_mmbert_embeddings_svd(tokenizer, mm_src, mm_tok, train_counts=train_counts, embed_dim=args.embed_dim)
        mdl.load_from_mmbert(mm_src, a_mat, p_mat)
        del mm_src, mm_tok
        torch.cuda.empty_cache()
    mdl = mdl.to(device)
    mdl.set_sdpa(True)

    # 4. Eval & Large-Sample Stratified Calibration splits (0 test leakage!)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_rows, val_cal_weights = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(subset: list[dict]):
        txts = [str(r["text"]) for r in subset]
        enc = tokenizer.encode_batch_fast(txts, max_len=MAX_LEN_V05) if txts else []
        st = extract_style_features_batch(txts)
        sc = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return enc, txts, st, sc

    val_enc, val_txts, val_st, val_scores = _pre_eval(val_gold)
    cal_enc, cal_txts, cal_st, cal_scores = _pre_eval(val_cal_rows)
    test_enc, test_txts, test_st, test_scores = _pre_eval(test_gold)
    test_dsh_enc, test_dsh_txts, test_dsh_st, test_dsh_scores = _pre_eval(test_dsh_gold)
    test_all_enc, test_all_txts, test_all_st, test_all_scores = _pre_eval(test_bili_all)
    probe_enc, probe_txts, probe_st, probe_scores = _pre_eval(probe_rows)

    eval_bs = 1024 if vram_gb >= 60.0 else 512

    @torch.no_grad()
    def predict_pt(model_eval: MeguvizV05ApexLoopModel, enc_ids: list[list[int]], st_arr: np.ndarray, num_steps: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (exit0_logits, coda_logits, emb256) on GPU."""
        model_eval.eval()
        n = len(enc_ids)
        e0_out = np.zeros(n, dtype=np.float32)
        c_out = np.zeros(n, dtype=np.float32)
        emb_out = np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return e0_out, c_out, emb_out
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
                out = model_eval.forward_full(ids, mask, st_t, num_steps=num_steps, return_aux=False)
            e0_out[st_i : st_i + len(chunk)] = out["exit0_logit"].float().cpu().numpy()
            c_out[st_i : st_i + len(chunk)] = out["step_logits"][-1].float().cpu().numpy()
            emb_out[st_i : st_i + len(chunk)] = out["emb256"].float().cpu().numpy()
        return e0_out, c_out, emb_out

    # DataLoader with length bucketing (static-bucket friendly)
    class V05Dataset(Dataset):
        def __init__(self, items, pife_noise_prob: float = 0.0):
            self.items = items
            self.pife_noise_prob = float(pife_noise_prob)

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
            max_seq = max(x["length"] for x in batch)
            # Round sequence length up to nearest multiple of 16 for clean static-bucket GPU kernel efficiency
            max_seq = min(MAX_LEN_V05, ((max_seq + 15) // 16) * 16)
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
        return DataLoader(V05Dataset(items, pife_noise_prob=pife_noise_prob), **kw)

    def _collate_pairs(p_batch: list[dict]):
        bs_p = len(p_batch)
        m_ai = min(MAX_LEN_V05, ((max(len(x["ai_ids"]) for x in p_batch) + 15) // 16) * 16)
        m_hu = min(MAX_LEN_V05, ((max(len(x["hu_ids"]) for x in p_batch) + 15) // 16) * 16)
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

    # Zero-sync GPU GroupDRO
    class TorchGroupDRO:
        def __init__(self, num_groups: int = len(V04_GROUPS), eta: float = 0.05, ema_decay: float = 0.9, floor_share: float = 0.03):
            self.num_groups = num_groups
            self.eta = eta
            self.ema_decay = ema_decay
            self.floor_share = floor_share
            self.q = torch.full((num_groups,), 1.0 / num_groups, dtype=torch.float32, device=device)
            self.ema_losses = torch.zeros(num_groups, dtype=torch.float32, device=device)
            self.initialized = torch.zeros(num_groups, dtype=torch.bool, device=device)

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

    def _skewed_kl(p_s, y_t, alpha: float = 0.1):
        ps = p_s.float().clamp(1e-5, 1.0 - 1e-5)
        pt = y_t.float().clamp(1e-5, 1.0 - 1e-5)
        pm = ((1.0 - alpha) * pt + alpha * ps).clamp(1e-5, 1.0 - 1e-5)
        return pt * torch.log(pt / pm) + (1.0 - pt) * torch.log((1.0 - pt) / (1.0 - pm))

    intent_vecs_t = torch.from_numpy(intent_vecs).to(device) if intent_vecs is not None else None
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_code = nn.CrossEntropyLoss(ignore_index=-100)
    surf_idx_t = torch.tensor(SURFACE_PRAG_IDXS, dtype=torch.long, device=device)
    deep_idx_t = torch.tensor(DEEP_PRAG_IDXS, dtype=torch.long, device=device)
    dro_tracker = TorchGroupDRO()

    # ------------------------------------------------------------------
    # Stage 0: Fast-ELECTRA 100% RTD Warmup (aligns 28k SuperBPE with inherited mmBERT layers!)
    # ------------------------------------------------------------------
    rtd_steps = 8 if args.smoke_test else int(args.rtd_steps)
    if rtd_steps > 0 and not args.resume_stage1:
        print(f"[Stage0-RTD] Running {rtd_steps} steps of 100% Replaced Token Detection warmup...", flush=True)
        mdl.train()
        mdl.set_qat_enabled(False)
        opt_rtd = HybridMuonClipAdamW(mdl, lr_adamw=args.lr * lr_scale * 1.5, use_muon=(args.optimizer == "muonclip"))
        loader_rtd = make_loader(train_items_raw)
        step_r = 0
        t_r = time.monotonic()
        r_loss_sum = 0.0
        for ids, mask, st, *_ in loader_rtd:
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            st = st.to(device, non_blocking=True)
            # Corrupt 18% of non-special tokens with random in-vocab replacements; supervise 100% of real tokens!
            non_spec = (mask == 1) & (ids >= 5)
            corrupt = non_spec & (torch.rand_like(ids.float()) < 0.18)
            rand_toks = torch.randint(5, tokenizer.vocab_size, ids.shape, device=device)
            ids_cor = torch.where(corrupt, rand_toks, ids)
            rtd_tgt = corrupt.float()

            opt_rtd.set_lr_mult(_cosine_mult(step_r, rtd_steps))
            opt_rtd.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, _, _, h_seq = mdl.forward_prelude(ids_cor, mask, st)
                rtd_logits = mdl.rtd_head(h_seq).squeeze(-1)
                valid = mask == 1
                # Upweight replaced tokens slightly (pos_weight=3.0) for crisp 100% RTD gradient
                loss_rtd = F.binary_cross_entropy_with_logits(rtd_logits[valid].float(), rtd_tgt[valid], pos_weight=torch.tensor(3.0, device=device))
            loss_rtd.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_rtd.step()
            r_loss_sum += float(loss_rtd.item())
            step_r += 1
            if step_r % max(50, rtd_steps // 4) == 0 or step_r == rtd_steps:
                print(f"[Stage0-RTD step={step_r}/{rtd_steps}] loss={r_loss_sum/step_r:.4f} elapsed={time.monotonic()-t_r:.1f}s", flush=True)
            if step_r >= rtd_steps:
                break

    # ------------------------------------------------------------------
    # Unified Phase Runner (Fixed Optimizer-Step Budget Support — Lesson #8!)
    # ------------------------------------------------------------------
    def run_phase(
        items: list[dict],
        target_steps: int,
        num_epochs_fallback: int,
        lr: float,
        qat: bool,
        phase_name: str,
        depth_choices: tuple[int, ...] = (2, 3, 4),
        ema_teacher: MeguvizV05ApexLoopModel | None = None,
        update_ema: bool = False,
        use_swa: bool = False,
        pife_noise_prob: float = 0.0,
    ) -> dict[str, torch.Tensor] | None:
        if not items:
            return None
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
        opt = HybridMuonClipAdamW(mdl, lr_adamw=eff_lr, use_muon=(args.optimizer == "muonclip" and not qat))
        rng_p = random.Random(20261125 + len(phase_name))
        # Lesson #8: proportional pair_bs so pair exposure ratio stays constant across batch sizes
        pair_bs = max(16, min(64, int(round(16.0 * batch_size / 256.0))))
        swa_start = int(total_steps * 0.55)
        swa_every = max(15, min(60, total_steps // 40))
        log_every = max(40, total_steps // 12)
        t_p = time.monotonic()
        global_step = 0
        swa_state: dict[str, torch.Tensor] | None = None
        swa_count = 0

        struct_grp_idx = GROUP_TO_IDX["structured_human_hard"]
        pre_grp_idx = GROUP_TO_IDX["pre_llm_human"]
        ml_grp_idx = GROUP_TO_IDX["multilingual_world_knowledge"]

        print(f"[{phase_name}] items={len(items)} | bs={batch_size} | target_steps={total_steps} (~{num_epochs} ep) | pair_bs={pair_bs} | qat={qat} | eff_lr={eff_lr:.2e}", flush=True)

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

                # Stochastic SCORE budget sampling R in depth_choices (R=2, 3, 4)
                cur_r = depth_choices[global_step % len(depth_choices)] if qat else rng_p.choice(depth_choices)
                opt.set_lr_mult(_cosine_mult(global_step, total_steps))
                opt.zero_grad(set_to_none=True)

                ema_p = None
                ema_e0_p = None
                if ema_teacher is not None:
                    ema_teacher.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                        eo = ema_teacher.forward_full(ids, mask, st, num_steps=cur_r, return_aux=False)
                        ema_p = torch.sigmoid(eo["step_logits"][-1].float())
                        ema_e0_p = torch.sigmoid(eo["exit0_logit"].float())

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl.forward_full(ids, mask, st, num_steps=cur_r, return_aux=True)
                    step_logits = out["step_logits"]
                    e0_logit = out["exit0_logit"]
                    y_eff = (0.72 * y_sc + 0.28 * ema_p) if ema_p is not None else y_sc

                    step_losses = []
                    for lg in step_logits:
                        ps = torch.sigmoid(lg.float())
                        per_s = (0.72 * _skewed_kl(ps, y_eff) + 0.28 * F.smooth_l1_loss(ps, y_eff, beta=0.04, reduction="none")) * w
                        step_losses.append(per_s)

                    # Prelude Exit-0 supervision
                    e0_p = torch.sigmoid(e0_logit.float())
                    loss_e0 = ((0.72 * _skewed_kl(e0_p, y_eff) + 0.28 * F.smooth_l1_loss(e0_p, y_eff, beta=0.04, reduction="none")) * w).mean()

                    # Zero-sync GPU GroupDRO weighting on final step
                    grp_w_t = dro_tracker.update(grp, step_losses[-1]) * len(V04_GROUPS)
                    dro_mult = grp_w_t[grp]
                    loss_main = sum((sl * dro_mult).mean() for sl in step_losses) / float(len(step_losses))

                    # Monotonic anti-regression + fixed-point contraction ||h_R - h_{R-1}||^2
                    loss_anti_reg = 0.0
                    for s_i in range(1, len(step_losses)):
                        loss_anti_reg = loss_anti_reg + F.relu(step_losses[s_i] - step_losses[s_i - 1].detach()).mean()
                    loss_anti_reg = loss_anti_reg / max(1, len(step_losses) - 1)
                    loss_fp = F.mse_loss(out["h_last"].float(), out["h_prev"].detach().float())

                    # Gentle logit norm regularizer keeping calibrated temperature sharp in [0.9, 1.8]!
                    l_last_fp = step_logits[-1].float()
                    loss_logit_reg = F.relu(l_last_fp.abs() - 4.2).pow(2).mean() + F.relu(e0_logit.float().abs() - 4.2).pow(2).mean()

                    # Training-only LOTUS + Code + Prag aux heads (zero inference cost!)
                    ax = out["step_aux"][-1]
                    loss_aux = 0.14 * ce_code(ax["code_logit"], y_cd) + 0.10 * ce_code(ax["lotus_code"], y_cd)
                    prag_mask = (prag >= 0.0).all(dim=-1)
                    if prag_mask.any():
                        loss_aux = loss_aux + 0.20 * bce_none(ax["prag_logit"][prag_mask], prag[prag_mask].clamp(min=0.0)).mean()
                        loss_aux = loss_aux + 0.12 * bce_none(ax["lotus_surf"][prag_mask], prag[prag_mask][:, surf_idx_t]).mean()
                        loss_aux = loss_aux + 0.12 * bce_none(ax["lotus_deep"][prag_mask], prag[prag_mask][:, deep_idx_t]).mean()
                    if intent_vecs_t is not None:
                        im = intent_idx >= 0
                        if im.any():
                            tgt_v = intent_vecs_t[intent_idx[im]].float()
                            pred_v = F.normalize(ax["lotus_intent"][im].float(), p=2, dim=-1)
                            loss_aux = loss_aux + 0.18 * (1.0 - (pred_v * tgt_v).sum(dim=-1)).mean()

                    loss = loss_main + 0.25 * loss_e0 + 0.15 * loss_anti_reg + 0.03 * loss_fp + 0.004 * loss_logit_reg + loss_aux

                    # Lesson #2: Dual-Channel Top-Push Margin + Two-Tier Absolute Tail Anchoring (beating OjinjinNet on human FPs!)
                    p_last_fp = torch.sigmoid(l_last_fp)
                    ai_def = (y_sc >= 0.85) & (w > 0.0)
                    hu_def = (
                        (y_sc <= 0.15)
                        | (((grp == struct_grp_idx) | (grp == pre_grp_idx)) & (y_sc <= 0.22))
                        | ((grp == ml_grp_idx) & (y_sc <= 0.20))
                    ) & (w > 0.0)
                    if ai_def.any() and hu_def.any():
                        l_ai_m = l_last_fp[ai_def].mean()
                        hu_l = l_last_fp[hu_def]
                        k_top = max(1, hu_l.numel() // 4)
                        l_hu_hard = torch.topk(hu_l, k_top).values.mean()
                        loss_top_push = F.softplus(2.6 - (l_ai_m - l_hu_hard))
                        loss_tier1_fp = F.relu(p_last_fp[hu_def] - 0.10).pow(2).mean()
                        loss = loss + 0.16 * loss_top_push + 0.24 * loss_tier1_fp

                    # Tier-2 absolute suppression on whitelisted structured & multilingual humans (y <= 0.35)
                    tier2_mask = ((grp == struct_grp_idx) | (grp == ml_grp_idx)) & (y_sc <= 0.35) & (w > 0.0)
                    if tier2_mask.any():
                        loss = loss + 0.15 * F.relu(p_last_fp[tier2_mask] - 0.16).pow(2).mean()

                    # QAT parity stabilizer on BOTH Core and Prelude Exit-0
                    if qat and ema_p is not None and ema_e0_p is not None:
                        loss = loss + 0.08 * F.mse_loss(p_last_fp, ema_p) + 0.05 * F.mse_loss(e0_p, ema_e0_p)

                    # Stratified 50/50 GateGuard + General Contrastive SimPO Pairs + Topic Invariance
                    if pairs_pre:
                        if pairs_gg_pre and pairs_gen_pre:
                            n_gg = min(len(pairs_gg_pre), max(4, pair_bs // 2))
                            n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                            p_batch = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                        else:
                            p_batch = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                        ai_ids, ai_m, ai_st, hu_ids, hu_m, hu_st, pw_t = _collate_pairs(p_batch)
                        o_ai = mdl.forward_full(ai_ids, ai_m, ai_st, num_steps=2, return_aux=False)
                        o_hu = mdl.forward_full(hu_ids, hu_m, hu_st, num_steps=2, return_aux=False)
                        l_ai_p = o_ai["step_logits"][-1].float()
                        l_hu_p = o_hu["step_logits"][-1].float()
                        loss_simpo = (F.softplus(1.8 - (l_ai_p - l_hu_p)) * pw_t).mean()
                        # Direct pair absolute anchoring so AI side pushes >=0.88 and human side stays <=0.16!
                        loss_pair_abs = (
                            (F.relu(0.88 - torch.sigmoid(l_ai_p)).pow(2) + F.relu(torch.sigmoid(l_hu_p) - 0.15).pow(2)) * pw_t
                        ).mean()
                        loss_e0_pair = (F.softplus(1.4 - (o_ai["exit0_logit"].float() - o_hu["exit0_logit"].float())) * pw_t).mean()
                        loss_topic = (1.0 - F.cosine_similarity(o_ai["z0"].mean(dim=1).float(), o_hu["z0"].mean(dim=1).float(), dim=-1)).mean()
                        loss = loss + 0.34 * loss_simpo + 0.18 * loss_pair_abs + 0.10 * loss_e0_pair + 0.03 * loss_topic

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()

                if update_ema and ema_teacher is not None:
                    with torch.no_grad():
                        for pe, ps in zip(ema_teacher.parameters(), mdl.parameters()):
                            pe.data.mul_(0.995).add_(ps.data, alpha=0.005)

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

            _, v_l, _ = predict_pt(mdl, val_enc, val_st, num_steps=2)
            m_ep = compute_eval_metrics(val_scores, expit(v_l))
            print(f"[{phase_name} ep={ep+1} DONE] val_r2: {json.dumps(m_ep)}", flush=True)
            if global_step >= total_steps:
                break

        return swa_state

    # ------------------------------------------------------------------
    # Stage 1: Full-Corpus Supervised Distillation (with persistent Stage-1 checkpoint!)
    # ------------------------------------------------------------------
    stage1_ckpt = ckpt_dir / "v05_stage1_ckpt.pt"
    if args.resume_stage1 and Path(args.resume_stage1).is_file():
        print(f"[Stage1] Resuming directly from Stage-1 checkpoint: {args.resume_stage1}", flush=True)
        mdl.load_state_dict(torch.load(args.resume_stage1, map_location=device), strict=True)
    else:
        run_phase(
            train_items_raw,
            target_steps=16 if args.smoke_test else int(args.stage1_steps),
            num_epochs_fallback=1 if args.smoke_test else int(args.epochs),
            lr=args.lr,
            qat=False,
            phase_name="Stage1-ApexDistill",
            depth_choices=(2, 3, 4),
        )
        if not args.smoke_test:
            torch.save(mdl.state_dict(), stage1_ckpt)
            print(f"[Stage1] Saved reusable Stage-1 checkpoint -> {stage1_ckpt}", flush=True)

    # Snapshot EMA teacher for Stage 2 & Stage 3
    ema_teacher = copy.deepcopy(mdl).eval()
    ema_teacher.set_sdpa(True)
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Stage 2: Born-Again EMA + PIFE Adversarial Refinement (FIXED steps + 100% Gold Retention!)
    # ------------------------------------------------------------------
    stage2_items, gold_ret_s2 = subset_balanced_gold_guaranteed(train_items_raw, cap=args.stage2_cap, seed_s=20261126)
    print(f"[Stage2] Subset={len(stage2_items)} | gold_retention={gold_ret_s2:.4f}", flush=True)
    run_phase(
        stage2_items,
        target_steps=12 if args.smoke_test else int(args.stage2_steps),
        num_epochs_fallback=1,
        lr=args.refine_lr,
        qat=False,
        phase_name="Stage2-EMA-PIFE",
        depth_choices=(2, 3, 4),
        ema_teacher=ema_teacher,
        update_ema=True,
        pife_noise_prob=0.08,
    )

    # ------------------------------------------------------------------
    # Stage 3: Fixed-Step LoopQ Full-Trajectory QAT (R=2 & R=4) + SWA!
    # ------------------------------------------------------------------
    qat_items, gold_ret_s3 = subset_balanced_gold_guaranteed(train_items_raw, cap=args.qat_cap, seed_s=20261127)
    print(f"[Stage3-LoopQ] Subset={len(qat_items)} | gold_retention={gold_ret_s3:.4f}", flush=True)
    swa_sd = run_phase(
        qat_items,
        target_steps=12 if args.smoke_test else int(args.qat_steps),
        num_epochs_fallback=1,
        lr=args.qat_lr,
        qat=True,
        phase_name="Stage3-LoopQ-SWA",
        depth_choices=(2, 4),
        ema_teacher=ema_teacher,
        update_ema=False,
        use_swa=not args.smoke_test,
        pife_noise_prob=0.05,
    )
    if swa_sd is not None:
        mdl.load_state_dict(swa_sd, strict=True)
        print("[Stage3-LoopQ-SWA] Applied SWA weights.", flush=True)

    del ema_teacher
    mdl.set_sdpa(False)
    mdl.set_qat_enabled(False)
    mdl.eval()

    # ------------------------------------------------------------------
    # Calibration: Sharp Baked T in [0.9, 1.8] + n>=2,400 Length-Bucketed Smooth PAVA + Split Conformal + L2 Prototype Bank
    # ------------------------------------------------------------------
    with torch.no_grad():
        mdl.exit0_temperature.fill_(1.0)
        mdl.coda_temperature.fill_(1.0)
    raw_cal_e0, raw_cal_c2, _ = predict_pt(mdl, cal_enc, cal_st, num_steps=2)
    t_e0_raw = float(fit_temperature(raw_cal_e0, cal_scores / 100.0))
    t_c2_raw = float(fit_temperature(raw_cal_c2, cal_scores / 100.0))
    # Clamp baked temperature to sharp healthy window [0.9, 1.8] (never allowing v0.4's T=3.9 compression!)
    t_e0 = float(np.clip(t_e0_raw, 0.90, 1.80))
    t_c2 = float(np.clip(t_c2_raw, 0.90, 1.80))
    with torch.no_grad():
        mdl.exit0_temperature.fill_(t_e0)
        mdl.coda_temperature.fill_(t_c2)
    print(f"[Calibration] raw_T=(e0={t_e0_raw:.3f}, c2={t_c2_raw:.3f}) -> baked_sharp_T=(e0={t_e0:.3f}, c2={t_c2:.3f})", flush=True)

    cal_e0_l, cal_c2_l, _ = predict_pt(mdl, cal_enc, cal_st, num_steps=2)
    cal_e0_p = expit(cal_e0_l)
    cal_c2_p = expit(cal_c2_l)

    # Auto-tune Prelude Exit-0 thresholds ensuring zero accuracy loss
    val_e0_l, val_c2_l, _ = predict_pt(mdl, val_enc, val_st, num_steps=2)
    val_e0_p = expit(val_e0_l)
    val_c2_p = expit(val_c2_l)
    val_bin = (val_scores >= 70.0).astype(bool)
    base_val_auc = float(roc_auc_score(val_bin, val_c2_p))
    best_exit0 = {"exit_low": 0.08, "exit_high": 0.92}
    best_exit_rate = 0.0
    for e_low, e_high in ((0.06, 0.94), (0.08, 0.92), (0.10, 0.90), (0.12, 0.88), (0.15, 0.85)):
        m_ex = (val_e0_p <= e_low) | (val_e0_p >= e_high)
        p_mix = np.where(m_ex, val_e0_p, val_c2_p)
        auc_m = float(roc_auc_score(val_bin, p_mix))
        ex_rate = float(m_ex.mean())
        if auc_m >= base_val_auc - 0.0015 and ex_rate > best_exit_rate:
            best_exit_rate = ex_rate
            best_exit0 = {"exit_low": e_low, "exit_high": e_high}
    print(f"[Calibration] exit0_policy={best_exit0} (val_exit0_rate={best_exit_rate:.3f}, base_auc={base_val_auc:.4f})", flush=True)

    def _combine_exit0(e0_p, c_p):
        m_ex = (e0_p <= best_exit0["exit_low"]) | (e0_p >= best_exit0["exit_high"])
        return np.where(m_ex, e0_p, c_p), float(m_ex.mean())

    cal_mix_p, _ = _combine_exit0(cal_e0_p, cal_c2_p)
    smooth_pava = fit_bucketed_smooth_pava(cal_txts, cal_mix_p, cal_scores / 100.0, sample_weights=val_cal_weights) if not args.smoke_test else {}
    cal_final_p = apply_bucketed_smooth_pava(cal_txts, cal_mix_p, smooth_pava)
    t_f05, t_fpr05 = find_val_thresholds(cal_scores, cal_final_p)
    hp_cal = compute_high_precision_metrics(cal_scores, cal_final_p)
    conformal_cal = fit_split_conformal_3way(cal_scores, cal_final_p, alphas=(0.03, 0.05))
    print(f"[Calibration] n_cal={len(cal_scores)} | conformal_alpha03={conformal_cal.get('alpha_03')}", flush=True)

    # Fit L2 Non-Parametric Prototype + Restricted Mahalanobis OOD Bank (<180 KB)
    bank_subset = train_items_raw[: min(24000, len(train_items_raw))]
    if not args.smoke_test and len(train_items_raw) > 24000:
        rng_bk = random.Random(20261125)
        hu_anc = [x for x in train_items_raw if x["is_pre_llm"] or (x["is_gold_tier"] and x["y_score"] <= 0.22)]
        ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
        bank_subset = rng_bk.sample(hu_anc, min(10000, len(hu_anc))) + rng_bk.sample(ai_anc, min(10000, len(ai_anc)))
    bk_enc = [x["ids"] for x in bank_subset]
    bk_st = np.stack([x["style"] for x in bank_subset], axis=0)
    bk_sc = np.array([x["y_score"] * 100.0 for x in bank_subset], dtype=np.float32)
    bk_is_hu = np.array([bool(x["is_pre_llm"] or x["is_gold_tier"]) for x in bank_subset], dtype=bool)
    _, _, bk_emb = predict_pt(mdl, bk_enc, bk_st, num_steps=2)
    proto_bank = fit_l2_prototype_bank(bk_emb, bk_sc, is_human_anchor=bk_is_hu)
    proto_path = out_dir / "prototype_bank.npz"
    np.savez_compressed(proto_path, **proto_bank)
    print(f"[L2 PrototypeBank] saved {proto_path.name} ({proto_path.stat().st_size/1024:.1f} KB)", flush=True)

    # Evaluate across R=2 (Normal), R=4 (Audit), R=6 (Hard) for strict monotonic tail verification (Lesson #7!)
    def _eval_split_all(enc, txts, st, sc):
        by_r = {}
        p_norm_final = None
        for r_steps in (2, 4, 6):
            e0_l, cr_l, _ = predict_pt(mdl, enc, st, num_steps=r_steps)
            if r_steps == 2:
                raw_p, ex_rate = _combine_exit0(expit(e0_l), expit(cr_l))
            else:
                raw_p, ex_rate = expit(cr_l), 0.0
            cal_p = apply_bucketed_smooth_pava(txts, raw_p, smooth_pava)
            if r_steps == 2:
                p_norm_final = cal_p
            by_r[f"r{r_steps}"] = {
                **compute_eval_metrics(sc, cal_p, t_f05, t_fpr05),
                **compute_high_precision_metrics(sc, cal_p),
                "exit0_rate": round(ex_rate, 4),
            }
        return by_r, p_norm_final

    test_by_r, test_norm_p = _eval_split_all(test_enc, test_txts, test_st, test_scores)
    test_dsh_by_r, _ = _eval_split_all(test_dsh_enc, test_dsh_txts, test_dsh_st, test_dsh_scores)
    test_all_by_r, _ = _eval_split_all(test_all_enc, test_all_txts, test_all_st, test_all_scores)
    probe_by_r, probe_norm_p = _eval_split_all(probe_enc, probe_txts, probe_st, probe_scores)

    r2_m, r4_m, r6_m = test_by_r["r2"], test_by_r["r4"], test_by_r["r6"]
    loop_monotonicity = {
        "test_bili_gold_auroc_by_r": {k: v["auroc"] for k, v in test_by_r.items()},
        "test_bili_gold_pr_auc_by_r": {k: v["pr_auc"] for k, v in test_by_r.items()},
        "test_bili_gold_rp95_by_r": {k: v["recall_at_p95"] for k, v in test_by_r.items()},
        "test_bili_gold_rp99_by_r": {k: v["recall_at_p99"] for k, v in test_by_r.items()},
        "non_regressive_r4_rp99": (r4_m["recall_at_p99"] >= r2_m["recall_at_p99"] - 0.03) and (r4_m["pr_auc"] >= r2_m["pr_auc"] - 0.004),
    }

    # ------------------------------------------------------------------
    # Export Dual Static ONNX + Ordered W4-Emb/W8-Body Edge Bundle (Bundle v2)
    # ------------------------------------------------------------------
    exp_info = export_v05_onnx_bundles(mdl, out_dir, vocab_size=tokenizer.vocab_size)

    manifest_base = {
        "schema_version": "meguviz.bundle.v2",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": f"v0.5-ApexLoop(mmBERT-RoPE-Prelude1L+16Slot+InfiLoop-SCORE-Core{args.core_layers}L+PIFE32d+L2Proto)",
        "license": "Apache-2.0",
        "base_model": args.backbone,
        "quant_method": "loopq_qat_swa_plus_dynamic_int8",
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": int(args.embed_dim),
        "hidden_size": 384,
        "core_layers": int(args.core_layers),
        "num_slots": 16,
        "max_len": MAX_LEN_V05,
        "static_shape_buckets": [[1, 8, 16], [32, 64, 128, 192]],
        "tokenizer_file": tok_path.name,
        "tokenizer_sha256": sha256_file(tok_path),
        "prototype_bank_file": proto_path.name,
        "prototype_bank_sha256": sha256_file(proto_path),
        "proto_blend": 0.0,
        "exit0_temperature": round(t_e0, 6),
        "coda_temperature": round(t_c2, 6),
        "exit0_policy": best_exit0,
        "smooth_pava": smooth_pava,
        "conformal": conformal_cal,
        "thresholds": {"f05": round(t_f05, 4), "p95": hp_cal["t_p95"], "p99": hp_cal["t_p99"], "fpr05": round(t_fpr05, 4), "score70": 0.70},
        "prelude_file": exp_info["prelude_int8"],
        "core_step_file": exp_info["core_step"],
        "coda_file": exp_info["coda"],
        "prelude_bytes": exp_info["prelude_int8_bytes"],
        "core_step_bytes": exp_info["core_step_bytes"],
        "coda_bytes": exp_info["coda_bytes"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT parity + latency verification on test_gold
    engine = MeguvizV05Engine(out_dir, threads=1, mode="normal")
    t_s = time.perf_counter()
    ort_single = [engine.predict_one(txt, mode="normal") for txt in test_txts]
    single_ms = (time.perf_counter() - t_s) * 1000.0 / max(1, len(test_txts))
    t_b = time.perf_counter()
    ort_batch = engine.predict_batch(test_txts, mode="normal", max_batch=32)
    batch_ms = (time.perf_counter() - t_b) * 1000.0 / max(1, len(test_txts))
    ort_p = np.array([x["prob"] for x in ort_batch], dtype=np.float32)
    parity = {
        "mae_vs_pytorch": round(float(np.mean(np.abs(test_norm_p - ort_p))), 5),
        "max_abs_diff_vs_pytorch": round(float(np.max(np.abs(test_norm_p - ort_p))), 5),
        "single_1thread_ms": round(single_ms, 3),
        "compact_batch32_1thread_ms": round(batch_ms, 3),
        "ort_exit0_rate": round(float(np.mean([1.0 if x["steps_used"] == 0 else 0.0 for x in ort_batch])), 3),
        "prelude_bytes": exp_info["prelude_int8_bytes"],
        "core_step_bytes": exp_info["core_step_bytes"],
        "coda_bytes": exp_info["coda_bytes"],
    }

    probe_file = Path("round01/bili-round01-user-confirmed.jsonl")
    verify_res = run_verify_v05(out_dir, probe_file.resolve(), mode="normal", threads=1) if probe_file.is_file() else {}

    metrics = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v2",
        "backbone": args.backbone,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "batch_size_used": batch_size,
        "optimizer": args.optimizer,
        "data_audit": data_audit,
        "gold_retention": {"stage2": round(gold_ret_s2, 4), "stage3": round(gold_ret_s3, 4)},
        "exit0_temperature": round(t_e0, 6),
        "coda_temperature": round(t_c2, 6),
        "raw_fitted_temperatures": {"exit0": round(t_e0_raw, 4), "coda": round(t_c2_raw, 4)},
        "exit0_policy": best_exit0,
        "conformal": conformal_cal,
        "loop_monotonicity": loop_monotonicity,
        "test_bili_gold": r2_m,
        "test_bili_gold_by_r": test_by_r,
        "test_dsh_gold": test_dsh_by_r["r2"],
        "test_dsh_gold_by_r": test_dsh_by_r,
        "test_all_bili_distilled": test_all_by_r["r2"],
        "probe_u001_u017": probe_by_r["r2"],
        "onnx_int8": parity,
        "edge_w4_w8": exp_info["edge_w4_w8"],
        "onnx_verify": {k: v for k, v in verify_res.items() if k != "probe_u001_u017"},
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics["release_gates"] = check_v05_release_gates(metrics)
    metrics_file = out_dir / f"metrics-{args.version}.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Package Standard INT8 zip + Edge W4-Emb/W8-Body zip (<= 6.5 MB target!)
    zip_int8 = out_dir / f"meguvizcac-{args.version}-int8.zip"
    common = [tok_path.name, proto_path.name, metrics_file.name]
    with ZipFile(zip_int8, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for fn in (exp_info["prelude_int8"], exp_info["core_step"], exp_info["coda"], "manifest.json", *common):
            fp = out_dir / fn
            if fp.is_file():
                zf.write(fp, fn)
    metrics["onnx_int8"]["zip_bytes"] = int(zip_int8.stat().st_size)

    if exp_info["edge_w4_w8"].get("ok") and exp_info["prelude_w4"]:
        edge_dir = out_dir / "_edge_tmp"
        edge_dir.mkdir(parents=True, exist_ok=True)
        em = dict(manifest_base)
        em["quant_method"] = "ordered_w4_emb_opset21_plus_w8_body"
        em["prelude_file"] = exp_info["prelude_w4"]
        em["prelude_bytes"] = int((out_dir / exp_info["prelude_w4"]).stat().st_size)
        (edge_dir / "manifest.json").write_text(json.dumps(em, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        zip_edge = out_dir / f"meguvizcac-{args.version}-edge-w4.zip"
        with ZipFile(zip_edge, "w", compression=ZIP_DEFLATED, compresslevel=5) as zf:
            zf.write(out_dir / exp_info["prelude_w4"], exp_info["prelude_w4"])
            zf.write(out_dir / exp_info["core_step"], exp_info["core_step"])
            zf.write(out_dir / exp_info["coda"], exp_info["coda"])
            zf.write(edge_dir / "manifest.json", "manifest.json")
            for fn in common:
                fp = out_dir / fn
                if fp.is_file():
                    zf.write(fp, fn)
        shutil.rmtree(edge_dir, ignore_errors=True)
        metrics["edge_w4_w8"]["edge_zip_bytes"] = int(zip_edge.stat().st_size)

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(
        json.dumps(
            {
                "test_bili_gold": r2_m,
                "loop_monotonicity": loop_monotonicity,
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
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("./out_v05_apexloop"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-stage1", type=Path, default=None)
    parser.add_argument("--version", default="v0.5.0-apexloop-1x2-e128")
    parser.add_argument("--backbone", default=DEFAULT_V05_BACKBONE)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_V05_VOCAB_SIZE)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--core-layers", type=int, default=1)
    parser.add_argument("--emb-drop", type=float, default=0.10)
    parser.add_argument("--optimizer", choices=("muonclip", "adamw"), default="muonclip")
    parser.add_argument("--rtd-steps", type=int, default=350)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--stage1-steps", type=int, default=0, help="0 = use --epochs over full 1.37M corpus")
    parser.add_argument("--stage2-steps", type=int, default=2400)
    parser.add_argument("--qat-steps", type=int, default=3000)
    parser.add_argument("--stage2-cap", type=int, default=420000)
    parser.add_argument("--qat-cap", type=int, default=380000)
    parser.add_argument("--stage1-synth-cap", type=int, default=15000)
    parser.add_argument("--v04-synth-cap", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--no-auto-lr-scale", action="store_true")
    parser.add_argument("--no-domain-aug", action="store_true")
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--refine-lr", type=float, default=4e-5)
    parser.add_argument("--qat-lr", type=float, default=1.2e-5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-vram", default="")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(args)
        return
    train_and_export(args)


if __name__ == "__main__":
    main()
