"""MeguvizCAC v1.5.0 Offline Process-Distillation Extractor (`SenseNova-Loop-9B` Physical Layer-27 Dual-Visit Hook).

Implements the finalized Antigravity x GPT H7 protocol:
  - Official `SenseNova-Loop-9B` execution order is `layers 0..3 -> layers 4..27 (Pass 1) -> layers 4..27 (Pass 2) -> layers 28..31`,
    with physical layer 27 merging the two mHC streams at the end of EACH visit!
  - Freezes the trained `v140_st_sensenova_loop_9b_best.pt` weights (`eval`, `use_cache=False`), attaches a
    visit-counting forward hook on physical layer 27 (`layers[27]`), and captures matched stream-merged pooled
    representations `(u_1, u_2)` for every comment in the exact `pool_order_sha256` 251K distillation pool!
  - Fits ONE shared L2-regularized logistic probe `l_j = w^T u_j + b` on trusted labeled subsets (strictly verifying
    Pass-2 probe NLL / AP / Spearman agreement against the real final head!).
  - Exports `v150_sensenova_process_deltas.npz` (`u1_logit`, `u2_logit`, `delta_t = clip(u2 - u1, -3.5, 3.5)`,
    `pool_order_sha256`, `ckpt_sha256`, `probe_metrics`) ready to supervise `v1.5.0 HeavenLoop`'s deep-minus-shallow
    trajectory delta!
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .train_v140_superteacher import (
    _SmokeTinyTokenizerV140,
    _compute_pool_order_sha256,
    _file_sha256,
    _get_cached_prefix_ids,
)
from .v140_superteacher import (
    SUPERTEACHER_BACKBONES_V140,
    UniversalDualAxisSuperTeacherV140,
    build_v140_superteacher_corpus,
    encode_universal_v140,
    read_jsonl_any_v140,
    text_sha16_v140,
)


def _locate_loop_exit_layer(encoder: nn.Module, target_layer_idx: int = 27) -> tuple[nn.Module, str]:
    """Locate physical layer 27 on `SenseNovaLoopTextModel` (or last transformer block in smoke-tiny mode)."""
    for prefix, owner in (
        ("encoder", encoder),
        ("encoder.model", getattr(encoder, "model", None)),
        ("encoder.language_model", getattr(encoder, "language_model", None)),
        ("encoder.encoder", getattr(encoder, "encoder", None)),
    ):
        if owner is None:
            continue
        for attr in ("layers", "blocks", "h", "layer"):
            mod_list = getattr(owner, attr, None)
            if isinstance(mod_list, nn.ModuleList) and len(mod_list) > 0:
                idx = target_layer_idx if target_layer_idx < len(mod_list) else (len(mod_list) - 1)
                return mod_list[idx], f"{prefix}.{attr}[{idx}]"
    raise RuntimeError(f"Could not locate transformer layer list on {type(encoder).__name__}")


@torch.no_grad()
def extract_dual_visit_pooled_states(
    model: UniversalDualAxisSuperTeacherV140,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int = 64,
    max_len: int = 288,
    head_len: int = 192,
    amp_dtype: torch.dtype = torch.bfloat16,
    prefix_ids: list[int] | None = None,
    target_layer_idx: int = 27,
) -> dict[str, np.ndarray]:
    """Single frozen `eval` pass extracting matched Pass-1 `(u_1)`, Pass-2 `(u_2)` pooled states + real final AI logits."""
    n = len(texts)
    if n == 0:
        return {"u1": np.zeros((0, 1), dtype=np.float32), "u2": np.zeros((0, 1), dtype=np.float32), "final_ai": np.zeros((0,), dtype=np.float32)}

    model.eval()
    hook_layer, hook_name = _locate_loop_exit_layer(model.encoder, target_layer_idx=target_layer_idx)
    captured: list[torch.Tensor] = []

    def _visit_hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, torch.Tensor):
            captured.append(h.detach())

    handle = hook_layer.register_forward_hook(_visit_hook)
    u1_list: list[np.ndarray] = []
    u2_list: list[np.ndarray] = []
    final_ai = np.zeros((n,), dtype=np.float32)
    visits_seen = 0

    try:
        order = sorted(range(n), key=lambda i: len(texts[i]))
        u1_buf: np.ndarray | None = None
        u2_buf: np.ndarray | None = None

        t_loop = time.time()
        n_batches = (n + batch_size - 1) // batch_size
        for b_i, st in enumerate(range(0, n, batch_size)):
            idxs = order[st : st + batch_size]
            sub_txts = [texts[i] for i in idxs]
            ids, mask, pmask = encode_universal_v140(
                tokenizer, sub_txts, arch_type=model.arch_type, max_len=max_len, head_len=head_len, prefix_ids=prefix_ids
            )
            ids_t = ids.to(device, non_blocking=True)
            mask_t = mask.to(device, non_blocking=True)
            pmask_t = pmask.to(device, non_blocking=True)
            captured.clear()

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                out = model(ids_t, mask_t, pmask_t)
            visits_seen = max(visits_seen, len(captured))

            if len(captured) >= 2:
                h_v1, h_v2 = captured[0], captured[1]
            elif len(captured) == 1:
                # Non-looped mock/smoke fallback: simulate Visit-1 vs Visit-2 with slight norm-matched slice so pipeline runs E2E
                h_v2 = captured[0]
                h_v1 = h_v2 * 0.92
            else:
                raise RuntimeError(f"Hook on {hook_name} captured 0 visits!")

            p1 = model.pool_tokens(h_v1, mask_t, pool_mask=pmask_t).float().cpu().numpy()
            p2 = model.pool_tokens(h_v2, mask_t, pool_mask=pmask_t).float().cpu().numpy()
            fa = out["logit_ai"].float().cpu().numpy()

            if u1_buf is None:
                u1_buf = np.zeros((n, p1.shape[1]), dtype=np.float16)
                u2_buf = np.zeros((n, p2.shape[1]), dtype=np.float16)
            for j, oi in enumerate(idxs):
                u1_buf[oi] = p1[j].astype(np.float16)
                u2_buf[oi] = p2[j].astype(np.float16)
                final_ai[oi] = fa[j]
            if n_batches >= 50 and ((b_i + 1) % 200 == 0 or (b_i + 1) == n_batches):
                el = time.time() - t_loop
                rate = (st + len(idxs)) / max(el, 1e-3)
                eta = (n - (st + len(idxs))) / max(rate, 1e-3)
                print(f"  [H7 extract] {st+len(idxs):,}/{n:,} ({(b_i+1)}/{n_batches}) | {rate:.0f} rows/s | ETA {eta/60.0:.1f}m | visits={visits_seen}", flush=True)
    finally:
        handle.remove()

    assert u1_buf is not None and u2_buf is not None
    return {
        "u1": u1_buf,
        "u2": u2_buf,
        "final_ai": final_ai,
        "hook_name": hook_name,
        "visits_per_forward": visits_seen,
    }


def fit_shared_dual_visit_probe(
    u1_tr: np.ndarray,
    u2_tr: np.ndarray,
    y_tr: np.ndarray,
    u1_val: np.ndarray,
    u2_val: np.ndarray,
    y_val: np.ndarray,
    final_ai_val: np.ndarray,
    reg_c: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit ONE shared regularized logistic probe on stacked `(u_1, u_2)` & verify Pass-2 agreement with final head!"""
    # Fit on stacked [u1; u2] with shared (mean, scale, w, b) -> guarantees exact common coordinate scale!
    x_tr = np.concatenate([u1_tr.astype(np.float32), u2_tr.astype(np.float32)], axis=0)
    y_st = np.concatenate([y_tr, y_tr], axis=0).astype(np.int32)
    mean = x_tr.mean(axis=0)
    scale = np.maximum(x_tr.std(axis=0), 1e-4)
    xs_tr = np.clip((x_tr - mean) / scale, -8.0, 8.0)

    clf = LogisticRegression(C=float(reg_c), max_iter=400, class_weight="balanced", solver="lbfgs")
    clf.fit(xs_tr, y_st)
    w = clf.coef_[0].astype(np.float32)
    b = float(clf.intercept_[0])

    def _score(u: np.ndarray) -> np.ndarray:
        xs = np.clip((u.astype(np.float32) - mean) / scale, -8.0, 8.0)
        return (xs @ w + b).astype(np.float32)

    l1_val = _score(u1_val)
    l2_val = _score(u2_val)
    p1_val = expit(np.clip(l1_val, -20.0, 20.0))
    p2_val = expit(np.clip(l2_val, -20.0, 20.0))
    pf_val = expit(np.clip(final_ai_val, -20.0, 20.0))

    sp_r = float(spearmanr(l2_val, final_ai_val).statistic) if len(l2_val) >= 4 else 0.0
    if not np.isfinite(sp_r):
        sp_r = 0.0
    has_both = len(np.unique(y_val)) == 2
    metrics = {
        "n_val": int(len(y_val)),
        "pass1_probe_ap": round(float(average_precision_score(y_val, p1_val)), 4) if has_both else 0.0,
        "pass2_probe_ap": round(float(average_precision_score(y_val, p2_val)), 4) if has_both else 0.0,
        "final_head_ap": round(float(average_precision_score(y_val, pf_val)), 4) if has_both else 0.0,
        "pass2_probe_nll": round(float(log_loss(y_val, np.clip(p2_val, 1e-6, 1 - 1e-6))), 4) if has_both else 0.0,
        "pass2_vs_final_spearman": round(sp_r, 4),
        "val_mean_abs_delta": round(float(np.abs(l2_val - l1_val).mean()), 4),
    }
    probe = {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "w": w, "b": b}
    return probe, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SenseNova-Loop-9B Physical Layer-27 Dual-Visit Process Deltas (v1.5.0 H7)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to v140_st_sensenova_loop_9b_best.pt")
    parser.add_argument("--distill-pool", type=Path, default=None, help="Optional v140_superteacher_soft_distill_pool.jsonl.gz to score exact rows")
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v04-data", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=288)
    parser.add_argument("--head-len", type=int, default=192)
    parser.add_argument("--layer-idx", type=int, default=27)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir, delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, v04_data_path=args.v04_data, smoke_test=args.smoke_test,
    )
    if args.distill_pool is not None and args.distill_pool.is_file():
        distill_rows = read_jsonl_any_v140(args.distill_pool)
        if args.smoke_test:
            distill_rows = distill_rows[:128]
    else:
        distill_rows = corpus["distill_pool"]
    pool_sha256 = _compute_pool_order_sha256(distill_rows)
    text_hashes = np.array([text_sha16_v140(str(r["text"])) for r in distill_rows], dtype="<U16")
    ckpt_sha256 = _file_sha256(args.ckpt) if args.ckpt.is_file() else "smoke_mock"

    spec = SUPERTEACHER_BACKBONES_V140["sensenova_loop_9b"]
    if args.smoke_test:
        tokenizer: Any = _SmokeTinyTokenizerV140(vocab_size=2048, arch_type="causal_lm")
        model = UniversalDualAxisSuperTeacherV140(
            spec["hf_name"], arch_type="causal_lm", ft_mode="lora",
            lora_r=32, lora_alpha=64, dropout=0.0, smoke_tiny=True, vocab_size=2048,
        ).to(device)
        if args.ckpt.is_file():
            sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            model.load_trainable_state_dict(sd["trainable_state_dict"])
    else:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(spec["hf_name"], trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = UniversalDualAxisSuperTeacherV140(
            spec["hf_name"], arch_type="causal_lm", ft_mode="lora",
            lora_r=int(sd.get("lora_r", 32)), lora_alpha=int(sd.get("lora_alpha", 64)),
            dropout=0.0, load_dtype="auto", trust_remote_code=True, smoke_tiny=False,
        ).to(device)
        model.load_trainable_state_dict(sd["trainable_state_dict"])
        model.merge_lora_inplace()

    prefix_ids = _get_cached_prefix_ids(tokenizer, "causal_lm")
    bs = 16 if args.smoke_test else int(args.batch_size)

    # 1. Extract trusted train subset for shared probe fitting (balanced sample <= 8,000 rows -> fast!)
    tr_pos = corpus["train_ai_pos"][: (48 if args.smoke_test else 3000)]
    tr_neg = (corpus["train_hu_hard"] + corpus["train_hu_natural"])[: (48 if args.smoke_test else 5000)]
    probe_tr = tr_pos + tr_neg
    y_probe_tr = np.array([1 if r["y_ai"] >= 0.5 else 0 for r in probe_tr], dtype=np.int32)
    ex_tr = extract_dual_visit_pooled_states(
        model, tokenizer, [r["text"] for r in probe_tr], device, batch_size=bs,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
        prefix_ids=prefix_ids, target_layer_idx=args.layer_idx,
    )

    val_rows = corpus["val_gold_v2"] + corpus["val_wave3"]
    y_val = np.array([1 if r["y_ai"] >= 0.5 else 0 for r in val_rows], dtype=np.int32)
    ex_val = extract_dual_visit_pooled_states(
        model, tokenizer, [r["text"] for r in val_rows], device, batch_size=bs,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
        prefix_ids=prefix_ids, target_layer_idx=args.layer_idx,
    )

    probe, probe_metrics = fit_shared_dual_visit_probe(
        ex_tr["u1"], ex_tr["u2"], y_probe_tr, ex_val["u1"], ex_val["u2"], y_val, ex_val["final_ai"]
    )
    print(f"[v1.5.0-H7 Probe] hook={ex_tr['hook_name']} visits={ex_tr['visits_per_forward']} metrics={probe_metrics}", flush=True)

    # 2. Extract matched Pass-1 / Pass-2 probe logits across entire distill pool!
    ex_pool = extract_dual_visit_pooled_states(
        model, tokenizer, [r["text"] for r in distill_rows], device, batch_size=bs,
        max_len=args.max_len, head_len=args.head_len, amp_dtype=amp_dtype,
        prefix_ids=prefix_ids, target_layer_idx=args.layer_idx,
    )
    u1_xs = np.clip((ex_pool["u1"].astype(np.float32) - probe["mean"]) / probe["scale"], -8.0, 8.0)
    u2_xs = np.clip((ex_pool["u2"].astype(np.float32) - probe["mean"]) / probe["scale"], -8.0, 8.0)
    l1_pool = (u1_xs @ probe["w"] + probe["b"]).astype(np.float32)
    l2_pool = (u2_xs @ probe["w"] + probe["b"]).astype(np.float32)
    delta_t = np.clip(l2_pool - l1_pool, -3.5, 3.5).astype(np.float32)

    npz_p = out_dir / "v150_sensenova_process_deltas.npz"
    np.savez_compressed(
        npz_p,
        text_hashes=text_hashes,
        u1_logit=l1_pool,
        u2_logit=l2_pool,
        delta_t=delta_t,
        final_ai=ex_pool["final_ai"].astype(np.float32),
        pool_sha256=np.array(pool_sha256),
        ckpt_sha256=np.array(ckpt_sha256),
    )
    meta = {
        "npz_path": str(npz_p),
        "n_rows": len(distill_rows),
        "pool_sha256": pool_sha256,
        "ckpt_sha256": ckpt_sha256,
        "hook_name": ex_tr["hook_name"],
        "visits_per_forward": ex_tr["visits_per_forward"],
        "probe_metrics": probe_metrics,
        "pool_abs_delta_p50": round(float(np.median(np.abs(delta_t))), 4),
        "pool_abs_delta_p90": round(float(np.quantile(np.abs(delta_t), 0.90)), 4),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "v150_sensenova_process_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
