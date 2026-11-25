"""Train, Distill (from frozen BF16 v0.6 master), Progressive UPQ Sub-2-Bit Compress (FP16 -> W4 -> 5-Trit/2-Bit SEQ),
Post-Quantization Refit (0 KB Surgery + Frozen EBT + LB-DSM + Sliding-Window Nano-Draft v2 + Spline-Temp + Mandap-B),
Run V7-R0..R4 Automated CI Gates & Export <= 2.20 MB Bundle v4 (`bundle_v4_sub2bit.mgvz` + ONNX INT8) for MeguvizCAC v0.7.0-OmniVeritas."""

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
import torch

from .train_student import CODE_LIST, CODE_TO_IDX, PRAG_TAGS, compute_eval_metrics, find_val_thresholds, sha256_file
from .train_v04_dualloop import _open_read, compute_high_precision_metrics
from .train_v05_apexloop import build_stratified_val_calibration_set
from .train_v06_hyperloop import (
    _load_v06_balanced_bundle,
    _modern_two_way_pauc_and_top_push,
    _subset_natural_prior_gold_guaranteed,
    _train_frozen_lb_dsm_critic,
)
from .v03_runtime import EXTRA_CROSS_PLATFORM_TRAPS
from .v04_corpus_prep import BENCH_TRAP_TEXTS, GROUP_TO_IDX, V04_GROUPS
from .v04_runtime import MULTILINGUAL_TRAPS, fit_split_conformal_3way
from .v05_runtime import PIFE_ADVERSARIAL_TRAPS, fit_l2_prototype_bank
from .v06_hyperloop import apply_zero_cost_confounding_surgery, text_to_byte_hash_ids
from .v07_features import (
    BPE_VOCAB_SIZE_V07,
    STYLE_FEAT_DIM_V07,
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    extract_style_features_40d,
    extract_style_features_40d_batch,
    fit_spline_temp_lut,
    make_engram_bigram_ids,
)
from .v07_omniveritas import (
    ATTRIBUTION_5WAY,
    MAX_LEN_V07,
    FrozenEBTCriticV07,
    MeguvizV07Engine,
    MeguvizV07OmniVeritasModel,
    NanoDraftV07,
    export_v07_onnx_and_bundle_v4,
    load_bundle_v4_mgvz_into_model,
    pack_bundle_v4_mgvz,
    run_v07_ci_gates,
)
from .v08_zenith_ops import (
    fold_zero_op_spinquant_and_smoothquant_v07,
    listnet_tail_distill_loss,
    zenith_graded_anchor_loss,
)


def _wsd_lr_mult(step: int, total_steps: int, warmup_frac: float = 0.05, decay_frac: float = 0.22, min_ratio: float = 0.04) -> float:
    """WSD (Warmup -> Stable -> 1-sqrt Decay) River-Valley LR schedule."""
    if total_steps <= 1:
        return 1.0
    w_st = max(1, int(total_steps * warmup_frac))
    d_st = max(1, int(total_steps * decay_frac))
    s_end = max(w_st, total_steps - d_st)
    if step < w_st:
        return float(step + 1) / float(w_st)
    if step < s_end:
        return 1.0
    prog = min(1.0, max(0.0, float(step - s_end) / float(max(1, total_steps - s_end))))
    return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(prog))


def _derive_attr5_label(item: dict) -> int:
    """Derive 5-way attribution label from existing MeguvizCAC score/code/group metadata."""
    y = float(item["y_score"])
    cd = int(item["code_idx"])
    gid = int(item["group_idx"])
    if y <= 0.30:
        return 0  # Human-Organic
    if y <= 0.65:
        return 1  # Human-Assisted / Grey boundary
    if cd == CODE_TO_IDX.get("A", 0) or gid == GROUP_TO_IDX.get("persona_space_kaomoji", -1):
        return 3  # Autonomous-Persona-Agent
    if cd == CODE_TO_IDX.get("T", 3):
        return 4  # Coordinated-Latent-Cluster / misaligned reply bot
    return 2  # Script-Plugin-Bot


@torch.no_grad()
def _init_bpe_embeddings_from_bge_svd(
    mdl: MeguvizV07OmniVeritasModel,
    tokenizer: BiliPickyBPETokenizer,
    backbone: str = "BAAI/bge-small-zh-v1.5",
) -> dict[str, Any]:
    """Cross-tokenizer BGE -> Picky-BPE12k SVD semantic initialization (0 extra params, 0 extra bytes!)."""
    try:
        from transformers import AutoModel, AutoTokenizer

        t0 = time.monotonic()
        hf_tok = AutoTokenizer.from_pretrained(backbone)
        hf_mdl = AutoModel.from_pretrained(backbone)
        bge_w = hf_mdl.embeddings.word_embeddings.weight.detach().float().cpu().numpy()  # (V_bge, 512)
        del hf_mdl
        bge_mean = bge_w.mean(axis=0)

        v_sz = int(mdl.vocab_size)
        d_bge = int(bge_w.shape[1])
        e_mat = np.zeros((v_sz, d_bge), dtype=np.float32)
        mapped = 0
        for tid in range(1, v_sz):
            s = tokenizer._tok.decode([tid], skip_special_tokens=True).strip()
            if s and "\ufffd" not in s:
                b_ids = hf_tok.encode(s, add_special_tokens=False)
                if b_ids:
                    e_mat[tid] = bge_w[b_ids].mean(axis=0)
                    mapped += 1
                    continue
            e_mat[tid] = bge_mean

        # Centered truncated SVD: E ≈ A @ P^T where A is (V, 48) and P is (384, 48)
        e_c = e_mat[1:] - e_mat[1:].mean(axis=0, keepdims=True)
        u, s_vals, vt = np.linalg.svd(e_c, full_matrices=False)
        k = int(mdl.embed_dim)
        d_h = int(mdl.hidden_size)
        a_init = u[:, :k] * np.sqrt(np.maximum(s_vals[:k], 1e-6))[None, :]
        a_init = a_init * (0.06 / max(1e-5, float(a_init.std())))

        # Right projection (384, 48): preserve top-k singular axes cleanly inside 384-d space
        p_raw = vt[:k, :d_h] * np.sqrt(np.maximum(s_vals[:k], 1e-6))[:, None]  # (48, 384)
        p_init = p_raw.T * (0.06 / max(1e-5, float(p_raw.std())))  # (384, 48)

        mdl.word_emb_A.weight.data[0].zero_()
        mdl.word_emb_A.weight.data[1:].copy_(torch.from_numpy(a_init).to(mdl.word_emb_A.weight.device, mdl.word_emb_A.weight.dtype))
        mdl.word_emb_P.weight.data.copy_(torch.from_numpy(p_init).to(mdl.word_emb_P.weight.device, mdl.word_emb_P.weight.dtype))
        rep = {"applied": True, "mapped_tokens": mapped, "vocab_size": v_sz, "elapsed_s": round(time.monotonic() - t0, 2)}
        print(f"[Stage0-BGE-SVD-Init] {json.dumps(rep)}", flush=True)
        return rep
    except Exception as exc:
        print(f"[Stage0-BGE-SVD-Init WARN] Skipped ({exc})", flush=True)
        return {"applied": False, "error": str(exc)}


def _precompute_v06_bf16_teacher_targets(
    teacher_ckpt: Path,
    train_items: list[dict],
    val_gold: list[dict],
    pairs: list[dict],
    backbone: str,
    device,
    amp_dtype,
    ckpt_dir: Path | None = None,
    polish_steps: int = 900,
    bs: int = 1024,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stage-0 Dual Teacher: Optionally fast-polish v0.6 BF16 Stage-1 into R3-grade teacher (~45s on G4, cached!)
    and precompute BOTH probs (N,) and L2-normalized 256-d embeddings (N, 256) fp16 (~80s on G4!)."""
    if not teacher_ckpt or not teacher_ckpt.is_file():
        return None, None
    import torch
    import torch.nn.functional as F
    from scipy.special import expit
    from transformers import AutoTokenizer
    from .v03_tokenizer import ByteFallbackWordPieceTokenizer
    from .v05_features import extract_style_features_batch
    from .v06_hyperloop import MeguvizV06HyperLoopModel

    t0 = time.monotonic()
    tmp_tok = (ckpt_dir or teacher_ckpt.parent) / "_tmp_v06_tok"
    tmp_tok.mkdir(parents=True, exist_ok=True)
    hf_tok = AutoTokenizer.from_pretrained(backbone)
    hf_tok.save_pretrained(tmp_tok)
    tok_v06 = ByteFallbackWordPieceTokenizer(tmp_tok / "vocab.txt")
    shutil.rmtree(tmp_tok, ignore_errors=True)

    cached_r3 = (ckpt_dir / "v0.6.0-r3-polished-teacher_bf16.pt") if ckpt_dir else None
    t_mdl = MeguvizV06HyperLoopModel(vocab_size=tok_v06.vocab_size, embed_dim=192, max_len=256)
    if cached_r3 and cached_r3.is_file():
        print(f"[Stage0-v06-Teacher] Loading cached polished R3 BF16 teacher from {cached_r3} ...", flush=True)
        t_mdl.load_state_dict(torch.load(cached_r3, map_location="cpu"), strict=True)
        t_mdl = t_mdl.to(device).eval()
        t_mdl.set_sdpa(True)
    else:
        print(f"[Stage0-v06-Teacher] Loading BF16 17.65M teacher from {teacher_ckpt} ...", flush=True)
        t_mdl.load_state_dict(torch.load(teacher_ckpt, map_location="cpu"), strict=True)
        t_mdl = t_mdl.to(device)
        t_mdl.set_sdpa(True)

        # Fast R3-style natural-prior + GateGuard polish (~45s on G4!) so teacher hits ~0.95+ AUROC / ~72% R@P95!
        if polish_steps > 0 and val_gold:
            t_pol = time.monotonic()
            v_txts = [str(r["text"]) for r in val_gold]
            v_enc = [tok_v06.encode_ids(t, max_len=256) for t in v_txts]
            v_st = extract_style_features_batch(v_txts)
            v_sc = np.array([float(r["score"]) for r in val_gold], dtype=np.float32)

            @torch.no_grad()
            def _eval_t():
                t_mdl.eval()
                ml = max(len(s) for s in v_enc)
                ids = np.full((len(v_enc), ml), tok_v06.pad_id, dtype=np.int64)
                msk = np.zeros((len(v_enc), ml), dtype=np.int64)
                for ri, s in enumerate(v_enc):
                    ids[ri, : len(s)], msk[ri, : len(s)] = s, 1
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    lg, _, _, _ = t_mdl(torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device), torch.from_numpy(v_st).to(device), return_aux=False)
                vp = expit(lg.float().cpu().numpy())
                mv = {**compute_eval_metrics(v_sc, vp), **compute_high_precision_metrics(v_sc, vp)}
                obj = 0.45 * float(mv["auroc"]) + 0.30 * float(mv["pr_auc"]) + 0.15 * float(mv["best_f05"]) + 0.10 * float(mv["recall_at_p95"])
                return obj, mv

            best_obj, mv0 = _eval_t()
            best_sd = {k: v.detach().cpu().clone() for k, v in t_mdl.state_dict().items()}
            print(f"[Stage0-v06-Polish START] val auroc={mv0['auroc']} rp95={mv0['recall_at_p95']} obj={best_obj:.4f}", flush=True)

            sub_p, _ = _subset_natural_prior_gold_guaranteed(train_items, cap=min(360000, len(train_items)), seed_s=20261125)
            p_encs = [tok_v06.encode_ids(x["raw_text"], max_len=256) for x in sub_p]
            p_sts = extract_style_features_batch([x["raw_text"] for x in sub_p])
            p_ys = np.array([x["y_score"] for x in sub_p], dtype=np.float32)
            p_ws = np.array([x["weight"] for x in sub_p], dtype=np.float32)
            p_gs = np.array([x["group_idx"] for x in sub_p], dtype=np.int64)

            pair_v06 = []
            if pairs:
                pa_s = extract_style_features_batch([p["ai_text"] for p in pairs])
                ph_s = extract_style_features_batch([p["human_text"] for p in pairs])
                for i, p in enumerate(pairs):
                    pair_v06.append((tok_v06.encode_ids(p["ai_text"], max_len=256), pa_s[i], tok_v06.encode_ids(p["human_text"], max_len=256), ph_s[i], float(p["weight"])))

            opt_t = torch.optim.AdamW(t_mdl.parameters(), lr=2.8e-5, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
            rng_pol = np.random.default_rng(20261125)
            pbs = 384
            t_mdl.train()
            for st_p in range(polish_steps):
                idx = rng_pol.integers(0, len(sub_p), size=min(pbs, len(sub_p)))
                ml = min(256, ((max(len(p_encs[i]) for i in idx) + 15) // 16) * 16)
                ids = np.full((len(idx), ml), tok_v06.pad_id, dtype=np.int64)
                msk = np.zeros((len(idx), ml), dtype=np.int64)
                for ri, oi in enumerate(idx):
                    s = p_encs[oi][:ml]
                    ids[ri, : len(s)], msk[ri, : len(s)] = s, 1
                lr_now = 2.8e-5 * _wsd_lr_mult(st_p, polish_steps)
                for pg in opt_t.param_groups:
                    pg["lr"] = lr_now
                opt_t.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    lg, _, _, _ = t_mdl(torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device), torch.from_numpy(p_sts[idx]).to(device), return_aux=False)
                    lg = lg.float()
                    yt = torch.from_numpy(p_ys[idx]).to(device)
                    wt = torch.from_numpy(p_ws[idx]).to(device)
                    gt = torch.from_numpy(p_gs[idx]).to(device)
                    loss = (F.binary_cross_entropy_with_logits(lg, yt.clamp(1e-5, 1.0 - 1e-5), reduction="none") * wt).mean()
                    l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, yt, wt, gt, prog=st_p / max(1, polish_steps - 1), beta_start=0.06, beta_end=0.03)
                    loss = loss + 0.10 * l_pa + l_tp + 0.6 * zenith_graded_anchor_loss(lg, yt)
                    if pair_v06 and (st_p % 2 == 0):
                        p_idx = rng_pol.integers(0, len(pair_v06), size=min(24, len(pair_v06)))
                        ma = max(len(pair_v06[k][0]) for k in p_idx)
                        mh = max(len(pair_v06[k][2]) for k in p_idx)
                        ai_i, ai_m = np.full((len(p_idx), ma), tok_v06.pad_id, dtype=np.int64), np.zeros((len(p_idx), ma), dtype=np.int64)
                        hu_i, hu_m = np.full((len(p_idx), mh), tok_v06.pad_id, dtype=np.int64), np.zeros((len(p_idx), mh), dtype=np.int64)
                        ai_st = np.stack([pair_v06[k][1] for k in p_idx], axis=0)
                        hu_st = np.stack([pair_v06[k][3] for k in p_idx], axis=0)
                        pw = torch.tensor([pair_v06[k][4] for k in p_idx], dtype=torch.float32, device=device)
                        for ri, k in enumerate(p_idx):
                            sa, sh = pair_v06[k][0], pair_v06[k][2]
                            ai_i[ri, : len(sa)], ai_m[ri, : len(sa)] = sa, 1
                            hu_i[ri, : len(sh)], hu_m[ri, : len(sh)] = sh, 1
                        la, _, _, _ = t_mdl(torch.from_numpy(ai_i).to(device), torch.from_numpy(ai_m).to(device), torch.from_numpy(ai_st).to(device), return_aux=False)
                        lh, _, _, _ = t_mdl(torch.from_numpy(hu_i).to(device), torch.from_numpy(hu_m).to(device), torch.from_numpy(hu_st).to(device), return_aux=False)
                        la, lh = la.float(), lh.float()
                        loss = loss + 0.25 * (F.softplus(1.9 - (la - lh)) * pw).mean() + 0.22 * ((F.relu(0.88 - torch.sigmoid(la)).pow(2) + 1.5 * F.relu(torch.sigmoid(lh) - 0.10).pow(2)) * pw).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(t_mdl.parameters(), 1.0)
                opt_t.step()
                if (st_p + 1) % 300 == 0 or (st_p + 1) == polish_steps:
                    co, mvc = _eval_t()
                    if co > best_obj:
                        best_obj = co
                        best_sd = {k: v.detach().cpu().clone() for k, v in t_mdl.state_dict().items()}
                    print(f"[Stage0-v06-Polish step={st_p+1}/{polish_steps}] val auroc={mvc['auroc']} rp95={mvc['recall_at_p95']} obj={co:.4f} (best={best_obj:.4f})", flush=True)
                    t_mdl.train()

            t_mdl.load_state_dict({k: v.to(device) for k, v in best_sd.items()}, strict=True)
            t_mdl.eval()
            if cached_r3:
                torch.save(best_sd, cached_r3)
            print(f"[Stage0-v06-Polish DONE] best_obj={best_obj:.4f} in {time.monotonic()-t_pol:.1f}s (cached={cached_r3})", flush=True)

    # Precompute BOTH probs (N,) float32 AND L2-normalized 256-d embeddings (N, 256) float16!
    t_mdl.eval()
    n = len(train_items)
    probs = np.zeros(n, dtype=np.float32)
    embs_fp16 = np.zeros((n, 256), dtype=np.float16)
    txts = [x["raw_text"] for x in train_items]
    encs = [tok_v06.encode_ids(t, max_len=256) for t in txts]
    order = sorted(range(n), key=lambda i: len(encs[i]))

    with torch.no_grad():
        for st_i in range(0, n, bs):
            mb = order[st_i : st_i + bs]
            ml = max(len(encs[i]) for i in mb)
            ids = np.full((len(mb), ml), tok_v06.pad_id, dtype=np.int64)
            mask = np.zeros((len(mb), ml), dtype=np.int64)
            for ri, oi in enumerate(mb):
                s = encs[oi]
                ids[ri, : len(s)] = s
                mask[ri, : len(s)] = 1
            st_arr = extract_style_features_batch([txts[oi] for oi in mb])
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                lg, eb, _, _ = t_mdl(
                    torch.from_numpy(ids).to(device, non_blocking=True),
                    torch.from_numpy(mask).to(device, non_blocking=True),
                    torch.from_numpy(st_arr).to(device, non_blocking=True),
                    return_aux=False,
                )
            p_np = torch.sigmoid(lg.float()).cpu().numpy()
            e_np = F.normalize(eb.float(), p=2, dim=-1).cpu().numpy().astype(np.float16)
            for ri, oi in enumerate(mb):
                probs[oi] = p_np[ri]
                embs_fp16[oi] = e_np[ri]

    del t_mdl
    torch.cuda.empty_cache()
    print(f"[Stage0-v06-Teacher DONE] Pre-computed {n:,} BF16 probs + 256-d fp16 embs in {time.monotonic()-t0:.1f}s!", flush=True)
    return probs, embs_fp16


@torch.no_grad()
def _precompute_v06_bf16_teacher_probs(
    teacher_ckpt: Path,
    train_items: list[dict],
    backbone: str,
    device,
    amp_dtype,
    bs: int = 1024,
) -> np.ndarray | None:
    """Backwards-compatible wrapper used by train_v08_zenith."""
    probs, _ = _precompute_v06_bf16_teacher_targets(
        teacher_ckpt, train_items, [], [], backbone, device, amp_dtype, ckpt_dir=None, polish_steps=0, bs=bs
    )
    return probs


def _train_frozen_ebt_critic(
    embs_tr: np.ndarray,
    attr5_tr: np.ndarray,
    embs_test: np.ndarray,
    test_scores: np.ndarray,
    device,
    steps: int = 500,
    bs: int = 256,
) -> tuple[FrozenEBTCriticV07, dict]:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    ebt = FrozenEBTCriticV07(emb_dim=embs_tr.shape[1], num_classes=len(ATTRIBUTION_5WAY), hidden=96).to(device)
    x_t = torch.from_numpy(embs_tr.astype(np.float32)).to(device)
    y_t = torch.from_numpy(attr5_tr.astype(np.int64)).to(device)
    opt = torch.optim.AdamW(ebt.parameters(), lr=2.5e-3, weight_decay=1e-4)
    ebt.train()
    for _ in range(steps):
        idx = torch.randint(0, x_t.shape[0], (min(bs, x_t.shape[0]),), device=device)
        loss = F.cross_entropy(ebt(x_t[idx]), y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ebt.eval()
    with torch.no_grad():
        margin = ebt.energy_margin(torch.from_numpy(embs_test.astype(np.float32)).to(device)).cpu().numpy()
    yb = (test_scores >= 70.0).astype(bool)
    auc = float(roc_auc_score(yb, margin)) if (yb.any() and (~yb).any()) else 0.5
    return ebt.cpu().eval(), {
        "trained": True,
        "test_gold_ebt_margin_auroc": round(auc, 4),
        "param_bytes_fp16": int(sum(p.numel() for p in ebt.parameters()) * 2),
    }


def _train_nano_draft_v07(
    train_items: list[dict],
    teacher_probs_train: np.ndarray,
    val_cleans: list[str],
    val_st: np.ndarray,
    val_metas: list[dict],
    val_scores: np.ndarray,
    val_teacher_p: np.ndarray,
    cal_cleans: list[str],
    cal_st: np.ndarray,
    cal_metas: list[dict],
    cal_scores: np.ndarray,
    cal_teacher_p: np.ndarray,
    device,
    steps: int = 1100,
    bs: int = 512,
) -> tuple[NanoDraftV07, dict, dict]:
    """Train Sliding-Window Max-Risk Nano-Draft v2 (strict L<=48 & spoof==0 early-exit policy!)."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    nano = NanoDraftV07(buckets=4096, emb_dim=32, conv_c=64, style_dim=STYLE_FEAT_DIM_V07).to(device)
    n_tr = len(train_items)
    h_seqs = [text_to_byte_hash_ids(x["clean_text"]) for x in train_items]
    st_arr = np.stack([x["style"] for x in train_items], axis=0).astype(np.float32)
    y_t = teacher_probs_train.astype(np.float32)
    defer_t = np.array(
        [
            1.0
            if (
                y_t[i] >= 0.22
                or len(train_items[i]["clean_text"]) > 48
                or train_items[i]["spoof_any"] > 0.0
                or ("http" in train_items[i]["clean_text"])
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
        pg, pdef, pwin, _ = nano(
            torch.from_numpy(h_ids).to(device),
            torch.from_numpy(h_m).to(device),
            torch.from_numpy(st_arr[idx]).to(device),
        )
        yt_t = torch.from_numpy(y_t[idx]).to(device)
        dt_t = torch.from_numpy(defer_t[idx]).to(device)
        w_fn = 1.0 + 3.5 * (yt_t >= 0.40).float()
        loss_p = (F.binary_cross_entropy(pg.clamp(1e-5, 1.0 - 1e-5), yt_t.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w_fn).mean()
        loss_w = (F.binary_cross_entropy(pwin.clamp(1e-5, 1.0 - 1e-5), yt_t.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w_fn).mean()
        loss_d = F.binary_cross_entropy(pdef.clamp(1e-5, 1.0 - 1e-5), dt_t)
        loss = loss_p + 0.45 * loss_w + 0.65 * loss_d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    nano.eval()

    @torch.no_grad()
    def _pred(cleans: list[str], st_m: np.ndarray):
        n = len(cleans)
        pg_o, pd_o, pw_o = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        seqs = [text_to_byte_hash_ids(t) for t in cleans]
        for st_i in range(0, n, 256):
            en = min(n, st_i + 256)
            ml = max(len(seqs[k]) for k in range(st_i, en))
            hi = np.zeros((en - st_i, ml), dtype=np.int64)
            hm = np.zeros((en - st_i, ml), dtype=np.int64)
            for ri, k in enumerate(range(st_i, en)):
                s = seqs[k]
                hi[ri, : len(s)] = s
                hm[ri, : len(s)] = 1
            p1, p2, p3, _ = nano(torch.from_numpy(hi).to(device), torch.from_numpy(hm).to(device), torch.from_numpy(st_m[st_i:en]).to(device))
            pg_o[st_i:en], pd_o[st_i:en], pw_o[st_i:en] = p1.cpu().numpy(), p2.cpu().numpy(), p3.cpu().numpy()
        return pg_o, pd_o, pw_o

    cal_pg, cal_df, cal_pw = _pred(cal_cleans, cal_st)
    val_pg, val_df, val_pw = _pred(val_cleans, val_st)
    y_val_bin = (val_scores >= 70.0).astype(bool)
    base_val_auc = float(roc_auc_score(y_val_bin, val_teacher_p)) if (y_val_bin.any() and (~y_val_bin).any()) else 0.5
    cal_is_ai = cal_scores >= 70.0
    cal_elig = np.array([m["clean_len"] <= 48 and m["spoof_any"] == 0.0 and ("http" not in t) for t, m in zip(cal_cleans, cal_metas)], dtype=bool)
    val_elig = np.array([m["clean_len"] <= 48 and m["spoof_any"] == 0.0 and ("http" not in t) for t, m in zip(val_cleans, val_metas)], dtype=bool)

    best_policy = {"enabled": False, "p_low": 0.06, "p_win_max": 0.12, "defer_max": 0.30}
    best_exit = 0.0
    for p_low in (0.04, 0.08, 0.12, 0.16, 0.22):
        for d_max in (0.20, 0.30, 0.45, 0.55):
            p_win = min(0.32, p_low + 0.08)
            m_cal = cal_elig & (cal_pg <= p_low) & (cal_pw <= p_win) & (cal_df < d_max)
            fn_cal = float(np.sum(m_cal & cal_is_ai)) / max(1, int(np.sum(cal_is_ai)))
            m_val = val_elig & (val_pg <= p_low) & (val_pw <= p_win) & (val_df < d_max)
            fn_val = int(np.sum(m_val & y_val_bin))
            p_cv = np.where(m_val, val_pg, val_teacher_p)
            auc_cv = float(roc_auc_score(y_val_bin, p_cv)) if (y_val_bin.any() and (~y_val_bin).any()) else base_val_auc
            ex_r = float(np.mean(m_cal))
            if fn_val == 0 and fn_cal <= 0.0030 and (base_val_auc - auc_cv) <= 0.0015 and ex_r > best_exit:
                best_exit = ex_r
                best_policy = {"enabled": ex_r >= 0.05, "p_low": p_low, "p_win_max": round(p_win, 3), "defer_max": d_max}

    m_vf = val_elig & (val_pg <= best_policy["p_low"]) & (val_pw <= best_policy["p_win_max"]) & (val_df < best_policy["defer_max"]) if best_policy["enabled"] else np.zeros(len(val_cleans), dtype=bool)
    m_cf = cal_elig & (cal_pg <= best_policy["p_low"]) & (cal_pw <= best_policy["p_win_max"]) & (cal_df < best_policy["defer_max"]) if best_policy["enabled"] else np.zeros(len(cal_cleans), dtype=bool)
    return nano.cpu().eval(), best_policy, {
        "policy": best_policy,
        "cal_exit_rate": round(float(np.mean(m_cf)), 4),
        "val_gold_exit_rate": round(float(np.mean(m_vf)), 4),
        "cal_missed_ai_count": int(np.sum(m_cf & cal_is_ai)),
        "val_base_auroc": round(base_val_auc, 4),
        "val_cascade_raw_probs": np.where(m_vf, val_pg, val_teacher_p),
        "cal_cascade_raw_probs": np.where(m_cf, cal_pg, cal_teacher_p),
    }


def _eval_traps_v07(engine: MeguvizV07Engine) -> dict[str, Any]:
    """Run all built-in AI, Structured-Human, Multilingual, and PIFE Adversarial traps."""
    human_traps = [
        ("H01_casual", "笑死我了，这个插件装完直接起飞[doge]"),
        ("H02_space", "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子"),
        ("H03_tech_human", "节点参考Blender材质节点这个思路挺对，接口丑不丑先不说，能白盒点开看子代理在干啥就很难得。"),
        ("H04_meme_banter", "大肥鱼在吗？我今天闲得发慌，想找条同类聊聊天，你平时都在干嘛呀？"),
        ("H05_emoji", "表情包已经出现了[妙啊][妙啊]"),
        ("H06_scorecard", "观感评分：画面 9/10，配乐 8.5/10，节奏 7/10，整体算近期很稳的良心作了。"),
        ("H07_bulleted_review", "两点真实感受：1. 前半段科普节奏很顺，没废话；2. 后面实操演示字体有点小，手机看费眼。"),
        ("H08_dict_quote", "查了下《现代汉语词典》，“敷衍”本来就可以做形容词用，弹幕没必要抓着这个字眼吵半天。"),
        ("H09_resource_link", "指路原版开源仓库 https://github.com/example/demo ，配环境卡住可以看置顶评论里的报错对照。"),
    ]
    ai_traps = [
        ("A01_persona_space", "回复 :本鱼看了 是不是那条说 让AI刷B站学人文关怀 本鱼承认 本鱼今天已经有点学坏了"),
        ("A02_param_kaomoji", "回复 :她挺可爱的，不过本鱼就是这个参数，换不动啦 (￣▽￣)"),
        ("A03_concise_emdash", "渲染省人眼不省token，卡片留在上下文里每轮重读，画得越漂亮账越贵"),
        ("A04_aphorism", "要的不是一次点击，是整只手。给权限容易，收回来难。"),
        ("A05_recap", "373行字幕读完：前两步重在定位边界，最后一步才是真正决定通过率的关键——慢一点反而不容易返工。"),
    ]

    ai_res = [{"id": tid, "prob": round(engine.predict_one(txt)["prob"], 4)} for tid, txt in ai_traps]
    hu_res = [{"id": tid, "prob": round(engine.predict_one(txt)["prob"], 4)} for tid, txt in human_traps]
    ml_res = [{"id": tid, "prob": round(engine.predict_one(txt)["prob"], 4)} for tid, txt in MULTILINGUAL_TRAPS]
    pf_res = [{"id": tid, "prob": round(engine.predict_one(txt)["prob"], 4)} for tid, txt in PIFE_ADVERSARIAL_TRAPS]

    struct_ids = {"H03_tech_human", "H06_scorecard", "H07_bulleted_review", "H08_dict_quote", "H09_resource_link"}
    ml_hu_ids = {"M01_en_human_casual", "M02_en_human_structured", "M03_ja_human_casual", "M04_zhen_tech_human"}
    ml_ai_ids = {"M05_en_ai_classic", "M06_en_ai_recap"}

    return {
        "ai_stress_traps": ai_res,
        "human_traps": hu_res,
        "multilingual_traps": ml_res,
        "pife_adversarial_traps": pf_res,
        "summary": {
            "ai_traps_min": min(x["prob"] for x in ai_res),
            "struct_human_max": max(x["prob"] for x in hu_res if x["id"] in struct_ids),
            "ml_human_max": max(x["prob"] for x in ml_res if x["id"] in ml_hu_ids),
            "ml_ai_min": min(x["prob"] for x in ml_res if x["id"] in ml_ai_ids),
        },
    }


def check_v07_release_gates(metrics: dict) -> dict:
    tb = metrics.get("test_bili_gold") or {}
    sub2 = metrics.get("sub2bit_bundle_v4_eval") or {}
    ox = metrics.get("onnx_int8") or {}
    traps = (metrics.get("traps") or {}).get("summary") or {}
    ci = metrics.get("v07_ci_gates") or {}
    conf = (metrics.get("conformal_cal2400") or {}).get("alpha_03") or {}
    mgvz = (metrics.get("export_info") or {}).get("bundle_v4_mgvz") or {}
    zip_mb = float(metrics.get("bundle_v4_total_mb") or 99.0)
    delta_auc = abs(float(sub2.get("auroc_delta_vs_fp16", 99.0)))

    gates = {
        "bundle_v4_le_2p20_mb": {"value": round(zip_mb, 3), "target": "<= 2.20 MB", "pass": zip_mb <= 2.20},
        "sub2_auroc_drift_le_0025": {"value": round(delta_auc, 5), "target": "<= 0.0025", "pass": delta_auc <= 0.0025},
        "speed_1t_cpu_ms": {"value": ox.get("compact_batch32_1thread_ms"), "target": "<= 1.50 ms (stretch <=0.40)", "pass": (ox.get("compact_batch32_1thread_ms") or 99.0) <= 1.50},
        "auroc_ge_0950": {"value": tb.get("auroc"), "target": ">= 0.950", "pass": (tb.get("auroc") or 0.0) >= 0.950},
        "pr_auc_ge_0900": {"value": tb.get("pr_auc"), "target": ">= 0.900", "pass": (tb.get("pr_auc") or 0.0) >= 0.900},
        "rp95_ge_072": {"value": tb.get("recall_at_p95"), "target": ">= 0.72", "pass": (tb.get("recall_at_p95") or 0.0) >= 0.72},
        "conformal_grey_le_12pct": {"value": conf.get("cal_grey_rate"), "target": "<= 0.12", "pass": (conf.get("cal_grey_rate") or 1.0) <= 0.12},
        "ci_zero_pii_diff_eq_0": {"value": (ci.get("CI_PII_04_zero_pii_username_fuzz") or {}).get("max_abs_diff"), "target": "== 0.0", "pass": bool((ci.get("CI_PII_04_zero_pii_username_fuzz") or {}).get("pass"))},
        "ci_astral_silverspeak": {"value": (ci.get("CI_Sec_01_astral_silverspeak") or {}).get("pass"), "target": "True", "pass": bool((ci.get("CI_Sec_01_astral_silverspeak") or {}).get("pass"))},
        "ci_sandwich_guard": {"value": (ci.get("CI_Sec_02_sandwich_no_early_exit") or {}).get("pass"), "target": "True", "pass": bool((ci.get("CI_Sec_02_sandwich_no_early_exit") or {}).get("pass"))},
        "ci_streaming_2048": {"value": (ci.get("CI_Long_03_streaming_2048") or {}).get("pass"), "target": "True", "pass": bool((ci.get("CI_Long_03_streaming_2048") or {}).get("pass"))},
        "ci_coldstart_lorentz100": {"value": (ci.get("CI_MV3_05_coldstart_and_lorentz100") or {}).get("pass"), "target": "True", "pass": bool((ci.get("CI_MV3_05_coldstart_and_lorentz100") or {}).get("pass"))},
    }
    return {"gates": gates, "passed_count": sum(1 for v in gates.values() if v["pass"]), "total_gates": len(gates), "all_passed": all(v["pass"] for v in gates.values())}


def train_and_export(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.special import expit
    from torch.utils.data import DataLoader, Dataset, Sampler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MeguvizCAC v0.7.0-OmniVeritas training.")

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

    print(f"[v0.7.0-OmniVeritas Config] GPU={gpu_name} ({vram_gb:.1f} GB) | amp={amp_dtype} | bs={batch_size} | teacher_v06={args.teacher_v06_ckpt}", flush=True)

    train_items_raw, eval_rows, pairs, intent_vecs, data_audit = _load_v06_balanced_bundle(
        args.data.resolve(),
        args.gateguard_data.resolve() if args.gateguard_data else None,
        [p.resolve() for p in args.extra_pairs],
        args.intent_cache.resolve() if args.intent_cache else None,
        stage1_synth_cap=args.stage1_synth_cap,
        v04_synth_cap=args.v04_synth_cap,
    )
    print(f"[v0.7.0 DataAudit] {json.dumps(data_audit, ensure_ascii=False)}", flush=True)
    if data_audit["gateguard_dropped_leakage"] > 0:
        raise RuntimeError(f"Non-silent anti-leakage fail-fast: {data_audit['gateguard_dropped_leakage']} pairs collided!")

    if args.smoke_test:
        rng_s = random.Random(20261125)
        train_items_raw = rng_s.sample(train_items_raw, min(640, len(train_items_raw)))
        pairs = pairs[:48]

    # Pre-filter val_gold early so Stage-0 teacher polish has strict val lock!
    val_gold_early = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]

    # Stage 0a: Optional Frozen BF16 v0.6.0 Dual Teacher (Probs + 256-d Embs, auto-polished to R3 quality & cached!)
    teacher_v06_p = None
    teacher_v06_embs = None
    if args.teacher_v06_ckpt and Path(args.teacher_v06_ckpt).is_file() and not args.smoke_test:
        teacher_v06_p, teacher_v06_embs = _precompute_v06_bf16_teacher_targets(
            Path(args.teacher_v06_ckpt),
            train_items_raw,
            val_gold_early,
            pairs,
            args.v06_backbone,
            device,
            amp_dtype,
            ckpt_dir=ckpt_dir,
            polish_steps=int(args.teacher_v06_polish_steps),
            bs=1024,
        )
    for idx_it, item in enumerate(train_items_raw):
        item["row_idx"] = idx_it
        item["teacher_p"] = float(teacher_v06_p[idx_it]) if teacher_v06_p is not None else -1.0
        item["attr5"] = _derive_attr5_label(item)

    # Stage 0b: Train 12,288 Picky-CJK-BPE Tokenizer strictly on train texts (with ~10% AI mix for 1-token clichés!)
    tok_path = out_dir / "bili_picky_bpe_12k.json"
    if args.smoke_test:
        tok_texts = [canonicalize_zero_pii_v07(x["raw_text"])[0] for x in train_items_raw]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(tok_texts, tok_path, vocab_size=2048, min_frequency=2)
    else:
        t_tok = time.monotonic()
        rng_t = random.Random(20261125)
        hu_pool = [x["raw_text"] for x in train_items_raw if x["y_score"] < 0.65]
        ai_pool = [x["raw_text"] for x in train_items_raw if x["y_score"] >= 0.70]
        sample_tok = rng_t.sample(hu_pool, min(180000, len(hu_pool))) + rng_t.sample(ai_pool, min(25000, len(ai_pool)))
        clean_tok = [canonicalize_zero_pii_v07(t)[0] for t in sample_tok]
        tokenizer = BiliPickyBPETokenizer.train_from_texts(clean_tok, tok_path, vocab_size=BPE_VOCAB_SIZE_V07, min_frequency=3)
        print(f"[Stage0-BPE12k DONE] vocab={tokenizer.vocab_size} trained on {len(clean_tok):,} train texts in {time.monotonic()-t_tok:.1f}s", flush=True)

    # Pre-encode train items (Zero-PII canonicalized + 40-d PIFE + BPE-12k + Engram!)
    t_enc = time.monotonic()
    total_toks = 0
    for item in train_items_raw:
        cl, f40, meta = extract_style_features_40d(item["raw_text"])
        ids = tokenizer.encode_ids(cl, max_len=args.max_len)
        item["clean_text"] = cl
        item["ids"] = ids
        item["engram"] = make_engram_bigram_ids(ids)
        item["length"] = len(ids)
        item["style"] = f40
        item["spoof_any"] = meta["spoof_any"]
        total_toks += len(ids)
    avg_tok = total_toks / max(1, len(train_items_raw))
    print(f"[v0.7.0 Pre-encode DONE] {len(train_items_raw):,} rows in {time.monotonic()-t_enc:.1f}s | avg_tokens={avg_tok:.2f} (vs ~37 WordPiece!)", flush=True)

    # Pre-encode GateGuard & Counterfactual Pairs
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

    # Pre-encode eval splits (ZERO test leakage!)
    val_gold = [r for r in eval_rows if r["split"] == "validation" and r.get("score_source") == "hydrogen_xhigh_direct"]
    val_cal_rows, _ = build_stratified_val_calibration_set(eval_rows, target_n=2400)
    test_gold = [r for r in eval_rows if r["split"] == "test" and r.get("score_source") == "hydrogen_xhigh_direct"]
    test_dsh_gold = [r for r in test_gold if r.get("dsh")]
    test_bili_all = [r for r in eval_rows if r.get("cohort") == "bili" and r["split"] == "test"]
    probe_rows = [r for r in eval_rows if r.get("cohort") == "user_confirmed"]

    def _pre_eval(subset: list[dict]):
        raws = [str(r["text"]) for r in subset]
        cls_l, st_m, metas_l = extract_style_features_40d_batch(raws)
        enc = [tokenizer.encode_ids(c, max_len=args.max_len) for c in cls_l]
        eng = [make_engram_bigram_ids(s) for s in enc]
        sc = np.array([float(r["score"]) for r in subset], dtype=np.float32)
        return enc, eng, raws, cls_l, st_m, metas_l, sc

    val_enc, val_eng, val_txts, val_cleans, val_st, val_metas, val_scores = _pre_eval(val_gold)
    cal_enc, cal_eng, cal_txts, cal_cleans, cal_st, cal_metas, cal_scores = _pre_eval(val_cal_rows)
    test_enc, test_eng, test_txts, test_cleans, test_st, test_metas, test_scores = _pre_eval(test_gold)
    dsh_enc, dsh_eng, dsh_txts, dsh_cleans, dsh_st, dsh_metas, dsh_scores = _pre_eval(test_dsh_gold)
    all_enc, all_eng, all_txts, all_cleans, all_st, all_metas, all_scores = _pre_eval(test_bili_all)
    prb_enc, prb_eng, prb_txts, prb_cleans, prb_st, prb_metas, prb_scores = _pre_eval(probe_rows)

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
    total_params = sum(p.numel() for p in mdl.parameters())
    edge_params = sum(p.numel() for n, p in mdl.named_parameters() if not n.startswith(("intent_head.", "cbm_dec.")))
    print(f"[v0.7.0-OmniVeritas Init] total_params={total_params/1e6:.2f}M | edge_params={edge_params/1e6:.2f}M", flush=True)

    # Cross-tokenizer BGE -> Picky-BPE12k SVD semantic initialization (0 extra bytes, instant Chinese semantics at Step 0!)
    bge_init_rep = {"applied": False}
    if not args.smoke_test:
        bge_init_rep = _init_bpe_embeddings_from_bge_svd(mdl, tokenizer, args.v06_backbone)

    # Pre-training V7-R0 Sanity Check!
    ci_pre = run_v07_ci_gates(mdl, tokenizer)
    print(f"[V7-R0 Pre-Train CI] all_passed={ci_pre['all_ci_passed']} | {json.dumps(ci_pre, ensure_ascii=False)}", flush=True)
    if not ci_pre["all_ci_passed"]:
        raise RuntimeError(f"V7-R0 CI gate failed before training: {ci_pre}")

    @torch.no_grad()
    def predict_pt(model_e: MeguvizV07OmniVeritasModel, enc_l, eng_l, st_arr: np.ndarray, num_loops: int = 2, use_fp32: bool = False):
        model_e.eval()
        n = len(enc_l)
        l_out = np.zeros(n, dtype=np.float32)
        emb_out = np.zeros((n, 256), dtype=np.float32)
        if n == 0:
            return l_out, emb_out
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
                ids[ri, : len(s)] = s
                m[ri, : len(s)] = 1
                eg[ri, : len(s)] = eng_l[oi]
            ids_t, m_t, eg_t = torch.from_numpy(ids).to(device), torch.from_numpy(m).to(device), torch.from_numpy(eg).to(device)
            st_t = torch.from_numpy(st_arr[mb]).to(device)
            if use_fp32:
                lg, eb, _, _ = model_e(ids_t, m_t, st_t, engram_ids=eg_t, num_loops=num_loops, return_aux=False)
            else:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    lg, eb, _, _ = model_e(ids_t, m_t, st_t, engram_ids=eg_t, num_loops=num_loops, return_aux=False)
            lg_np, eb_np = lg.float().cpu().numpy(), eb.float().cpu().numpy()
            for ri, oi in enumerate(mb):
                l_out[oi] = lg_np[ri]
                emb_out[oi] = eb_np[ri]
        return l_out, emb_out

    def _val_eval_obj(model_e: MeguvizV07OmniVeritasModel, num_loops: int = 2):
        vl, _ = predict_pt(model_e, val_enc, val_eng, val_st, num_loops=num_loops)
        vp = expit(vl)
        mv = {**compute_eval_metrics(val_scores, vp), **compute_high_precision_metrics(val_scores, vp)}
        obj = 0.45 * float(mv["auroc"]) + 0.30 * float(mv["pr_auc"]) + 0.15 * float(mv["best_f05"]) + 0.10 * float(mv["recall_at_p95"])
        return obj, mv

    class V07Dataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    class BucketSampler(Sampler):
        def __init__(self, lengths, bs, seed=20261125):
            self.lengths, self.bs, self.seed, self.epoch = lengths, bs, seed, 0
        def __iter__(self):
            rng_b = random.Random(self.seed + self.epoch)
            self.epoch += 1
            idxs = list(range(len(self.lengths)))
            rng_b.shuffle(idxs)
            batches = []
            mega = max(self.bs * 16, 2048)
            for st in range(0, len(idxs), mega):
                ch = idxs[st : st + mega]
                ch.sort(key=lambda i: self.lengths[i])
                for bst in range(0, len(ch), self.bs):
                    batches.append(ch[bst : bst + self.bs])
            rng_b.shuffle(batches)
            for b in batches:
                yield b
        def __len__(self):
            return (len(self.lengths) + self.bs - 1) // self.bs

    # Pre-collect short casual human wrapper phrases for 10% on-the-fly Sandwich Adversarial Augmentation!
    casual_wrappers = [
        tokenizer.encode_ids(x["clean_text"], max_len=20)[1:-1]
        for x in train_items_raw[:40000]
        if x["y_score"] <= 0.12 and 6 <= len(x["clean_text"]) <= 18
    ][:512]

    def make_collate(sandwich_prob: float = 0.08):
        rng_c = random.Random(20261125)
        def _collate(batch):
            seqs, engs = [], []
            for x in batch:
                s = x["ids"]
                if sandwich_prob > 0.0 and casual_wrappers and x["y_score"] >= 0.78 and len(s) <= 180 and rng_c.random() < sandwich_prob:
                    w1 = rng_c.choice(casual_wrappers)
                    w2 = rng_c.choice(casual_wrappers)
                    s = [tokenizer.bos_id] + w1 + s[1:-1] + w2 + [tokenizer.eos_id]
                    s = s[: args.max_len]
                    eg = make_engram_bigram_ids(s)
                else:
                    eg = x["engram"]
                seqs.append(s)
                engs.append(eg)
            max_seq = min(args.max_len, ((max(len(s) for s in seqs) + 15) // 16) * 16)
            ids = np.full((len(batch), max_seq), tokenizer.pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), max_seq), dtype=np.int64)
            eng = np.zeros((len(batch), max_seq), dtype=np.int64)
            st = np.empty((len(batch), STYLE_FEAT_DIM_V07), dtype=np.float32)
            for i, (s, eg, x) in enumerate(zip(seqs, engs, batch)):
                n_s = min(len(s), max_seq)
                ids[i, :n_s] = s[:n_s]
                mask[i, :n_s] = 1
                eng[i, :n_s] = eg[:n_s]
                st[i] = x["style"]
            return (
                torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(eng), torch.from_numpy(st),
                torch.tensor([x["y_score"] for x in batch], dtype=torch.float32),
                torch.tensor([x["teacher_p"] for x in batch], dtype=torch.float32),
                torch.tensor([1.0 if x["is_gold_tier"] else 0.0 for x in batch], dtype=torch.float32),
                torch.tensor([x["code_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["attr5"] for x in batch], dtype=torch.long),
                torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
                torch.tensor([x["prag_vec"] for x in batch], dtype=torch.float32),
                torch.tensor([x["group_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["intent_idx"] for x in batch], dtype=torch.long),
                torch.tensor([x["row_idx"] for x in batch], dtype=torch.long),
            )
        return _collate

    def make_loader(items, sandwich_prob=0.08):
        nw = min(8, max(4, (os.cpu_count() or 4) - 2)) if (os.name != "nt" and not args.smoke_test) else 0
        kw = {"batch_sampler": BucketSampler([x["length"] for x in items], batch_size), "collate_fn": make_collate(sandwich_prob), "num_workers": nw, "pin_memory": True}
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 4
        return DataLoader(V07Dataset(items), **kw)

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
            torch.from_numpy(ai_ids).to(device, non_blocking=True), torch.from_numpy(ai_m).to(device, non_blocking=True), torch.from_numpy(ai_e).to(device, non_blocking=True), torch.from_numpy(ai_st).to(device, non_blocking=True),
            torch.from_numpy(hu_ids).to(device, non_blocking=True), torch.from_numpy(hu_m).to(device, non_blocking=True), torch.from_numpy(hu_e).to(device, non_blocking=True), torch.from_numpy(hu_st).to(device, non_blocking=True),
            torch.tensor([x["weight"] for x in p_batch], dtype=torch.float32, device=device),
        )

    def _skewed_kl(p_s, y_t, alpha=0.1):
        ps, pt = p_s.float().clamp(1e-5, 1.0 - 1e-5), y_t.float().clamp(1e-5, 1.0 - 1e-5)
        pm = ((1.0 - alpha) * pt + alpha * ps).clamp(1e-5, 1.0 - 1e-5)
        return pt * torch.log(pt / pm) + (1.0 - pt) * torch.log((1.0 - pt) / (1.0 - pm))

    intent_vecs_t = torch.from_numpy(intent_vecs).to(device) if intent_vecs is not None else None
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_code = nn.CrossEntropyLoss(ignore_index=-100)

    hard_grp_t = torch.tensor(
        [GROUP_TO_IDX["structured_human_hard"], GROUP_TO_IDX["multilingual_world_knowledge"], GROUP_TO_IDX["dsh_hard_slice"]],
        device=device,
    )

    def run_phase_v07(
        items: list[dict],
        target_steps: int,
        num_epochs_fallback: int,
        lr: float,
        quant_mode: str,
        phase_name: str,
        fp16_teacher: MeguvizV07OmniVeritasModel | None = None,
        use_swa: bool = False,
        keep_best_val: bool = True,
        sandwich_prob: float = 0.08,
        pauc_w: float = 0.09,
        pair_w: float = 0.24,
        beta_end: float = 0.030,
    ):
        mdl.set_quant_mode(quant_mode)
        loader = make_loader(items, sandwich_prob=sandwich_prob)
        steps_per_ep = max(1, len(loader))
        if target_steps > 0:
            total_steps = int(target_steps)
            num_epochs = max(1, int(math.ceil(total_steps / float(steps_per_ep))))
        else:
            num_epochs = max(1, int(num_epochs_fallback))
            total_steps = num_epochs * steps_per_ep

        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.015, fused=True)
        rng_p = random.Random(20261125 + len(phase_name))
        pair_bs = max(16, min(40, int(round(16.0 * batch_size / 256.0))))
        log_every = max(40, total_steps // 12)
        eval_every = max(350, min(600, total_steps // 3)) if (keep_best_val and not args.smoke_test) else 0
        swa_start = int(total_steps * 0.60)
        swa_every = max(15, min(60, total_steps // 30))
        swa_state, swa_count = None, 0
        best_obj, best_sd = -1.0, None
        global_step = 0
        t_p = time.monotonic()

        print(f"[{phase_name}] items={len(items):,} | bs={batch_size} | steps={total_steps} (~{num_epochs} ep) | quant={quant_mode} | lr={lr:.2e}", flush=True)

        for ep in range(num_epochs):
            mdl.train()
            run_l, n_seen = torch.zeros((), device=device), 0
            for batch in loader:
                ids, mask, eng, st, y_sc, tp_v06, is_g, y_cd, y_a5, w, prag, grp, intent_idx, row_idx = [t.to(device, non_blocking=True) for t in batch]
                mult = _wsd_lr_mult(global_step, total_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr * mult
                opt.zero_grad(set_to_none=True)

                # Multi-step training: 20% of steps run R=4 w/ Static MoR 16->8 so deep gray-zone loops stay non-regressive!
                loops_now = 4 if (global_step % 5 == 0) else 2

                t_p_master, t_emb_master = None, None
                if fp16_teacher is not None:
                    fp16_teacher.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                        tl, te, _, _ = fp16_teacher(ids, mask, st, engram_ids=eng, num_loops=loops_now, return_aux=False)
                        t_p_master = torch.sigmoid(tl.float())
                        t_emb_master = te.float()

                t_emb_v06 = None
                if fp16_teacher is None and teacher_v06_embs is not None:
                    t_emb_v06 = torch.from_numpy(teacher_v06_embs[row_idx.cpu().numpy()]).to(device, dtype=torch.float32, non_blocking=True)

                prog = float(global_step) / float(max(1, total_steps - 1))
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = mdl(ids, mask, st, engram_ids=eng, num_loops=loops_now, return_aux=True)
                    lg = out["logit"].float()
                    ps = torch.sigmoid(lg)

                    # Blend targets cleanly: pure gold keeps strong gold truth; weak rows absorb R3 BF16 v0.6 / FP16 v0.7 teacher!
                    if t_p_master is not None:
                        y_eff = torch.where(is_g > 0.5, 0.70 * y_sc + 0.30 * t_p_master, 0.40 * y_sc + 0.60 * t_p_master)
                    else:
                        has_v06 = (tp_v06 >= 0.0) & (is_g < 0.5)
                        y_eff = torch.where(has_v06, 0.40 * y_sc + 0.60 * tp_v06.clamp(0.0, 1.0), y_sc)

                    loss_main = ((0.75 * _skewed_kl(ps, y_eff) + 0.25 * F.smooth_l1_loss(ps, y_eff, beta=0.04, reduction="none")) * w).mean()
                    mid_p = torch.sigmoid(out["mid_logit"].float())
                    loss_mid = ((0.75 * _skewed_kl(mid_p, y_eff) + 0.25 * F.smooth_l1_loss(mid_p, y_eff, beta=0.04, reduction="none")) * w).mean()
                    loss_mil = (F.binary_cross_entropy_with_logits(out["mil_peak"].float(), y_eff.clamp(1e-5, 1.0 - 1e-5), reduction="none") * w).mean()
                    loss_pauc, loss_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=prog, beta_start=0.07, beta_end=beta_end)
                    hu_h = torch.isin(grp, hard_grp_t)
                    loss_anc = 0.65 * zenith_graded_anchor_loss(lg, y_sc, hard_human_mask=hu_h)

                    loss_aux = 0.12 * ce_code(out["code_logit"], y_cd) + 0.10 * F.cross_entropy(out["attr5_logit"], y_a5) + 0.04 * out["cbm_loss"]
                    pm = (prag >= 0.0).all(dim=-1)
                    if pm.any():
                        loss_aux = loss_aux + 0.15 * bce_none(out["prag_logit"][pm], prag[pm].clamp(min=0.0)).mean()
                    if intent_vecs_t is not None:
                        im = intent_idx >= 0
                        if im.any():
                            tgt_v = intent_vecs_t[intent_idx[im]].float()
                            pred_v = F.normalize(out["intent_pred"][im].float(), p=2, dim=-1)
                            loss_aux = loss_aux + 0.12 * (1.0 - (pred_v * tgt_v).sum(dim=-1)).mean()

                    loss = loss_main + 0.14 * loss_mid + 0.10 * loss_mil + pauc_w * loss_pauc + loss_tp + loss_anc + loss_aux
                    if t_emb_master is not None:
                        # BitDistill representation + probability + tail-ranking alignment during progressive UPQ!
                        cos_d = 1.0 - F.cosine_similarity(out["emb"].float(), t_emb_master, dim=-1).mean()
                        loss = loss + 0.28 * cos_d + 0.22 * F.mse_loss(ps, t_p_master) + 0.08 * listnet_tail_distill_loss(lg, torch.logit(t_p_master.clamp(1e-4, 1.0 - 1e-4)))
                    elif t_emb_v06 is not None:
                        # R3 v0.6 dual distillation: 256-d latent geometry + tail-ranking alignment!
                        cos_v06 = 1.0 - F.cosine_similarity(out["emb"].float(), t_emb_v06, dim=-1).mean()
                        loss = loss + 0.22 * cos_v06
                        valid_tp = tp_v06 >= 0.0
                        if int(valid_tp.sum()) >= 8:
                            loss = loss + 0.08 * listnet_tail_distill_loss(lg[valid_tp], torch.logit(tp_v06[valid_tp].clamp(1e-4, 1.0 - 1e-4)))

                    if pairs_pre and pair_w > 0.0 and (global_step % 2 == 0):
                        if pairs_gg_pre and pairs_gen_pre:
                            n_gg = min(len(pairs_gg_pre), max(6, int(round(pair_bs * 0.55))))
                            n_gen = min(len(pairs_gen_pre), max(1, pair_bs - n_gg))
                            pb = rng_p.sample(pairs_gg_pre, n_gg) + rng_p.sample(pairs_gen_pre, n_gen)
                        else:
                            pb = rng_p.sample(pairs_pre, min(pair_bs, len(pairs_pre)))
                        ai_i, ai_m, ai_e, ai_s, hu_i, hu_m, hu_e, hu_s, pw_t = _collate_pairs(pb)
                        la, _, _, _ = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, num_loops=2, return_aux=False)
                        lh, _, _, _ = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, num_loops=2, return_aux=False)
                        la, lh = la.float(), lh.float()
                        l_simpo = (F.softplus(1.9 - (la - lh)) * pw_t).mean()
                        l_pabs = ((F.relu(0.88 - torch.sigmoid(la)).pow(2) + 1.6 * F.relu(torch.sigmoid(lh) - 0.10).pow(2)) * pw_t).mean()
                        loss = loss + pair_w * (l_simpo + 0.90 * l_pabs)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()

                if use_swa and global_step >= swa_start and (global_step % swa_every == 0 or global_step == total_steps - 1):
                    with torch.no_grad():
                        csd = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
                        if swa_state is None:
                            swa_state, swa_count = csd, 1
                        else:
                            for k in swa_state:
                                if swa_state[k].is_floating_point():
                                    swa_state[k].mul_(swa_count / (swa_count + 1)).add_(csd[k], alpha=1.0 / (swa_count + 1))
                            swa_count += 1

                run_l = run_l + loss.detach().float()
                n_seen += 1
                global_step += 1
                if global_step % log_every == 0 or global_step == total_steps:
                    print(f"[{phase_name} step={global_step}/{total_steps} ep={ep+1}/{num_epochs}] loss={float(run_l.item())/max(1,n_seen):.4f} elapsed={time.monotonic()-t_p:.1f}s", flush=True)
                if eval_every > 0 and global_step % eval_every == 0 and global_step < total_steps:
                    co, mm = _val_eval_obj(mdl)
                    if co > best_obj:
                        best_obj, best_sd = co, {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                    print(f"[{phase_name} MID step={global_step}] obj={co:.4f} (best={best_obj:.4f}) auroc={mm['auroc']} pr_auc={mm['pr_auc']} rp95={mm['recall_at_p95']} f05={mm['best_f05']}", flush=True)
                    mdl.train()
                if global_step >= total_steps:
                    break

            co, me = _val_eval_obj(mdl)
            if keep_best_val and co > best_obj:
                best_obj, best_sd = co, {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
            print(f"[{phase_name} ep={ep+1} DONE] obj={co:.4f} val: auroc={me['auroc']} pr_auc={me['pr_auc']} rp95={me['recall_at_p95']} rp99={me['recall_at_p99']} f05={me['best_f05']} mae={me['mae']}", flush=True)
            if global_step >= total_steps:
                break

        if keep_best_val and best_sd is not None:
            if swa_state is not None:
                mdl.load_state_dict(swa_state, strict=True)
                so, ms = _val_eval_obj(mdl)
                print(f"[{phase_name} SWA_CHECK] swa_obj={so:.4f} (auroc={ms['auroc']} rp95={ms['recall_at_p95']}) vs best={best_obj:.4f}", flush=True)
                if so >= best_obj - 0.0025:
                    return swa_state
            mdl.load_state_dict({k: v.to(device) for k, v in best_sd.items()}, strict=True)
            print(f"[{phase_name} BEST_RESTORED] obj={best_obj:.4f}", flush=True)
        return None

    # =========================================================================
    # Stage 1: Full-Corpus Natural-Prior WSD + R3 BF16 v0.6 Dual Distillation -> FP16 Base
    # =========================================================================
    run_phase_v07(
        train_items_raw,
        target_steps=14 if args.smoke_test else 0,
        num_epochs_fallback=1 if args.smoke_test else int(args.epochs),
        lr=args.lr,
        quant_mode="fp16",
        phase_name="Stage1-FP16-Master-WSD",
        fp16_teacher=None,
        use_swa=not args.smoke_test,
        keep_best_val=True,
        sandwich_prob=0.08,
        pauc_w=0.08,
        pair_w=0.24,
        beta_end=0.032,
    )

    # =========================================================================
    # Stage 1.5 (NEW in R2!): FP16 Natural-Prior Gold-Guaranteed Hard-Human Refinement!
    # Mirrors v0.6.0-R3's exact jump (+17% R@P95, H03 0.91 -> 0.05) BEFORE touching quantization!
    # =========================================================================
    upq_items, gold_ret_s2 = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=args.upq_cap, seed_s=20261126)
    if args.stage15_steps > 0:
        run_phase_v07(
            upq_items,
            target_steps=10 if args.smoke_test else int(args.stage15_steps),
            num_epochs_fallback=1,
            lr=args.stage15_lr,
            quant_mode="fp16",
            phase_name="Stage1.5-FP16-NaturalPrior-Refine",
            fp16_teacher=None,
            use_swa=not args.smoke_test,
            keep_best_val=True,
            sandwich_prob=0.06,
            pauc_w=0.11,
            pair_w=0.34,
            beta_end=0.025,
        )

    # Zero-Op SpinQuant R_2 + SmoothQuant + SubLN Algebraic Fold BEFORE UPQ (0 new ops, slashes sub2bit drift!)
    mdl.set_quant_mode("fp16")
    calib_batches = []
    cal_dl = DataLoader(V07Dataset(upq_items[: min(256, len(upq_items))]), batch_size=64, shuffle=False, collate_fn=make_collate(0.0))
    for cb in cal_dl:
        calib_batches.append((cb[0].to(device), cb[1].to(device), cb[3].to(device), cb[2].to(device)))
    fold_rep = fold_zero_op_spinquant_and_smoothquant_v07(mdl, calib_inputs=calib_batches)
    print(f"[Stage2-Pre ZeroOpFold] {json.dumps(fold_rep)}", flush=True)

    fp16_master = copy.deepcopy(mdl).eval()
    for p in fp16_master.parameters():
        p.requires_grad_(False)
    if not args.smoke_test:
        torch.save(fp16_master.state_dict(), ckpt_dir / f"{args.version}_fp16_master.pt")
        shutil.copy2(tok_path, ckpt_dir / f"{args.version}_bili_picky_bpe_12k.json")

    # Measure exact FP16 Master test_bili_gold baseline BEFORE sub-2-bit quantization!
    fp16_test_l, _ = predict_pt(fp16_master, test_enc, test_eng, test_st, num_loops=2, use_fp32=True)
    fp16_test_m = {**compute_eval_metrics(test_scores, expit(fp16_test_l)), **compute_high_precision_metrics(test_scores, expit(fp16_test_l))}
    fp16_r4_l, _ = predict_pt(fp16_master, test_enc, test_eng, test_st, num_loops=4, use_fp32=True)
    fp16_r4_m = {**compute_eval_metrics(test_scores, expit(fp16_r4_l)), **compute_high_precision_metrics(test_scores, expit(fp16_r4_l))}
    print(f"[Stage1.5-FP16 Master] test_gold R=2 auroc={fp16_test_m['auroc']} pr_auc={fp16_test_m['pr_auc']} rp95={fp16_test_m['recall_at_p95']} | R=4 auroc={fp16_r4_m['auroc']} rp95={fp16_r4_m['recall_at_p95']}", flush=True)

    # =========================================================================
    # Stage 2: Progressive UPQ Compression Bridge (FP16 -> W4 -> 5-Trit/2-Bit SEQ g=128)
    # =========================================================================
    run_phase_v07(
        upq_items,
        target_steps=10 if args.smoke_test else int(args.w4_steps),
        num_epochs_fallback=1,
        lr=args.w4_lr,
        quant_mode="w4",
        phase_name="Stage2a-UPQ-W4-Bridge",
        fp16_teacher=fp16_master,
        use_swa=False,
        keep_best_val=True,
        sandwich_prob=0.05,
        pauc_w=0.09,
        pair_w=0.26,
    )
    run_phase_v07(
        upq_items,
        target_steps=12 if args.smoke_test else int(args.sub2_steps),
        num_epochs_fallback=1,
        lr=args.sub2_lr,
        quant_mode="sub2",
        phase_name="Stage2b-UPQ-5Trit-2BitSEQ",
        fp16_teacher=fp16_master,
        use_swa=not args.smoke_test,
        keep_best_val=True,
        sandwich_prob=0.04,
        pauc_w=0.10,
        pair_w=0.28,
    )
    del fp16_master
    torch.cuda.empty_cache()

    # =========================================================================
    # Stage 3: Bit-Exact 5-Trit/2-Bit Packing -> Guarded Seq-LoRA-MoE & Head Refit -> Surgery -> Critics!
    # =========================================================================
    mdl.set_quant_mode("fp16")
    mdl.eval()
    tmp_mgvz = out_dir / "_tmp_pre_refit.mgvz"
    pack_bundle_v4_mgvz(mdl.cpu(), tmp_mgvz)
    load_bundle_v4_mgvz_into_model(mdl, tmp_mgvz)
    tmp_mgvz.unlink(missing_ok=True)
    mdl = mdl.to(device)

    # Stage 3a: Strictly Guarded Frozen-Backbone Post-Quantization Refit (rolls back if val regresses!)
    pre_refit_obj, _ = _val_eval_obj(mdl)
    pre_refit_sd = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
    for name, p in mdl.named_parameters():
        p.requires_grad_(("lora_moe" in name) or name.startswith(("coda_", "ai_head.", "style_", "cbm_")))
    refit_opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    refit_items, _ = _subset_natural_prior_gold_guaranteed(train_items_raw, cap=min(120000, len(train_items_raw)), seed_s=20261129)
    refit_loader = make_loader(refit_items, sandwich_prob=0.0)
    rng_rf = random.Random(20261129)
    mdl.train()
    refit_steps = 8 if args.smoke_test else 260
    rs = 0
    for batch in refit_loader:
        ids, mask, eng, st, y_sc, _, _, _, _, w, _, grp, _, _ = [t.to(device, non_blocking=True) for t in batch]
        refit_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = mdl(ids, mask, st, engram_ids=eng, num_loops=2, return_aux=True)
            lg = out["logit"].float()
            ps = torch.sigmoid(lg)
            lp = ((0.75 * _skewed_kl(ps, y_sc) + 0.25 * F.smooth_l1_loss(ps, y_sc, beta=0.04, reduction="none")) * w).mean()
            l_pa, l_tp = _modern_two_way_pauc_and_top_push(lg, y_sc, w, grp, prog=0.85, beta_start=0.05, beta_end=0.025)
            hu_h = torch.isin(grp, hard_grp_t)
            loss = lp + 0.09 * l_pa + l_tp + 0.65 * zenith_graded_anchor_loss(lg, y_sc, hard_human_mask=hu_h)
            if pairs_pre and (rs % 2 == 0):
                pb = rng_rf.sample(pairs_pre, min(20, len(pairs_pre)))
                ai_i, ai_m, ai_e, ai_s, hu_i, hu_m, hu_e, hu_s, pw_t = _collate_pairs(pb)
                la, _, _, _ = mdl(ai_i, ai_m, ai_s, engram_ids=ai_e, num_loops=2, return_aux=False)
                lh, _, _, _ = mdl(hu_i, hu_m, hu_s, engram_ids=hu_e, num_loops=2, return_aux=False)
                la, lh = la.float(), lh.float()
                loss = loss + 0.24 * (F.softplus(1.9 - (la - lh)) * pw_t).mean() + 0.22 * ((F.relu(0.88 - torch.sigmoid(la)).pow(2) + 1.6 * F.relu(torch.sigmoid(lh) - 0.10).pow(2)) * pw_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        refit_opt.step()
        rs += 1
        if rs >= refit_steps:
            break
    for p in mdl.parameters():
        p.requires_grad_(True)
    post_refit_obj, _ = _val_eval_obj(mdl)
    if post_refit_obj < pre_refit_obj:
        mdl.load_state_dict({k: v.to(device) for k, v in pre_refit_sd.items()}, strict=True)
        print(f"[Stage3a-Refit ROLLED_BACK] post={post_refit_obj:.4f} < pre={pre_refit_obj:.4f}", flush=True)
    else:
        print(f"[Stage3a-Refit KEPT] pre={pre_refit_obj:.4f} -> post={post_refit_obj:.4f}", flush=True)
    mdl.eval()

    # Stage 3b: 0 KB Confounding-Neuron Orthogonal Projection Surgery (auto-validated on val_gold!)
    with torch.no_grad():
        mdl.temperature.fill_(1.0)
    _, val_embs = predict_pt(mdl, val_enc, val_eng, val_st, use_fp32=True)
    surgery_rep = apply_zero_cost_confounding_surgery(
        mdl,
        val_hard_human_embs=val_embs[val_scores <= 35.0],
        val_ai_embs=val_embs[val_scores >= 70.0],
        val_all_embs=val_embs,
        val_all_scores=val_scores,
        rank=8,
    )
    print(f"[Stage3b-Surgery] {json.dumps(surgery_rep)}", flush=True)

    # Stage 3c: Tail-Preserving Strictly Increasing Spline-Temp Calibration (0 top-tail ties!)
    raw_val_l, _ = predict_pt(mdl, val_enc, val_eng, val_st, use_fp32=True)
    spline_lut = fit_spline_temp_lut(raw_val_l, val_scores / 100.0, t_min=0.85, t_max=1.65)
    t_baked = float(spline_lut["T_baked"])
    with torch.no_grad():
        mdl.temperature.fill_(t_baked)
    val_l_t, _ = predict_pt(mdl, val_enc, val_eng, val_st, use_fp32=True)
    val_p_raw = expit(val_l_t)
    cal_l_t, _ = predict_pt(mdl, cal_enc, cal_eng, cal_st, use_fp32=False)
    cal_p_raw = expit(cal_l_t)
    _, test_embs = predict_pt(mdl, test_enc, test_eng, test_st, use_fp32=True)

    # Stage 3d: Frozen-Backbone Critics (LB-DSM Curvature + Forward EBT + Prototype Bank + Sliding-Window Nano-Draft v2)
    rng_bk = random.Random(20261125)
    hu_anc = [x for x in train_items_raw if x["is_hu_anchor"]]
    ai_anc = [x for x in train_items_raw if x["y_score"] >= 0.82]
    bank_sub = train_items_raw[: min(24000, len(train_items_raw))] if args.smoke_test else (rng_bk.sample(hu_anc, min(12000, len(hu_anc))) + rng_bk.sample(ai_anc, min(12000, len(ai_anc))))
    _, bk_emb = predict_pt(mdl, [x["ids"] for x in bank_sub], [x["engram"] for x in bank_sub], np.stack([x["style"] for x in bank_sub], axis=0))
    bk_is_hu = np.array([bool(x["is_hu_anchor"]) for x in bank_sub], dtype=bool)
    bk_is_ai = np.array([x["y_score"] >= 0.82 for x in bank_sub], dtype=bool)
    bk_a5 = np.array([x["attr5"] for x in bank_sub], dtype=np.int64)

    proto_path = out_dir / "prototype_bank_v07.npz"
    np.savez_compressed(proto_path, **fit_l2_prototype_bank(bk_emb, np.array([x["y_score"] * 100.0 for x in bank_sub], dtype=np.float32), is_human_anchor=bk_is_hu))

    lb_critic, lb_rep = _train_frozen_lb_dsm_critic(bk_emb[bk_is_hu], bk_emb[bk_is_ai], test_embs, test_scores, device, steps=80 if args.smoke_test else 650)
    lb_path = out_dir / "lb_dsm_critic_v07.npz"
    np.savez_compressed(lb_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in lb_critic.state_dict().items()})
    lb_rep["npz_bytes"] = int(lb_path.stat().st_size)

    ebt_critic, ebt_rep = _train_frozen_ebt_critic(bk_emb, bk_a5, test_embs, test_scores, device, steps=80 if args.smoke_test else 500)
    ebt_path = out_dir / "ebt_critic_v07.npz"
    np.savez_compressed(ebt_path, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in ebt_critic.state_dict().items()})
    ebt_rep["npz_bytes"] = int(ebt_path.stat().st_size)

    nano_sub = train_items_raw if (args.smoke_test or len(train_items_raw) <= 60000) else random.Random(20261128).sample(train_items_raw, 60000)
    nano_tl, _ = predict_pt(mdl, [x["ids"] for x in nano_sub], [x["engram"] for x in nano_sub], np.stack([x["style"] for x in nano_sub], axis=0))
    nano_cpu, nano_policy, nano_rep = _train_nano_draft_v07(
        nano_sub, expit(nano_tl), val_cleans, val_st, val_metas, val_scores, val_p_raw, cal_cleans, cal_st, cal_metas, cal_scores, cal_p_raw, device, steps=60 if args.smoke_test else 1100
    )
    val_casc_raw = nano_rep.pop("val_cascade_raw_probs")
    cal_casc_raw = nano_rep.pop("cal_cascade_raw_probs")

    # Conformal Mandap-B on Spline-Temp calibrated probs
    val_p_cal = apply_spline_temp_lut(val_p_raw, spline_lut)
    cal_p_cal = apply_spline_temp_lut(cal_casc_raw, spline_lut)
    t_f05, t_fpr05 = find_val_thresholds(val_scores, val_p_cal)
    conf_val = fit_split_conformal_3way(val_scores, apply_spline_temp_lut(val_casc_raw, spline_lut), alphas=(0.03, 0.05))
    conf_2400 = fit_split_conformal_3way(cal_scores, cal_p_cal, alphas=(0.03, 0.05))

    # =========================================================================
    # Stage 4: Final Bit-Exact Bundle v4 Packing (<= 2.20 MB!), Fidelity Verification, ONNX & CI!
    # =========================================================================
    exp_info = export_v07_onnx_and_bundle_v4(mdl, nano_cpu, out_dir)
    # Bit-exact unpack verification of final `bundle_v4_sub2bit.mgvz`!
    mdl_unpacked = copy.deepcopy(mdl).cpu().eval()
    load_bundle_v4_mgvz_into_model(mdl_unpacked, out_dir / exp_info["bundle_v4_mgvz"]["file"])
    mdl_unpacked = mdl_unpacked.to(device).eval()

    def _eval_split(m_e, enc_l, eng_l, st_m, sc, fp32=True):
        lg, _ = predict_pt(m_e, enc_l, eng_l, st_m, use_fp32=fp32)
        cp = apply_spline_temp_lut(expit(lg), spline_lut)
        return {**compute_eval_metrics(sc, cp, t_f05, t_fpr05), **compute_high_precision_metrics(sc, cp)}, cp

    test_m, test_p = _eval_split(mdl_unpacked, test_enc, test_eng, test_st, test_scores, fp32=True)
    dsh_m, _ = _eval_split(mdl_unpacked, dsh_enc, dsh_eng, dsh_st, dsh_scores, fp32=True)
    all_m, _ = _eval_split(mdl_unpacked, all_enc, all_eng, all_st, all_scores, fp32=False)
    prb_m, _ = _eval_split(mdl_unpacked, prb_enc, prb_eng, prb_st, prb_scores, fp32=True)

    # Run all 6 automated V7-R0..R4 CI gates on the FINAL bit-exact unpacked model!
    ci_final = run_v07_ci_gates(mdl_unpacked, tokenizer, nano=nano_cpu, nano_policy=nano_policy)

    # Write Bundle v4 manifest
    manifest = {
        "schema_version": "meguviz.bundle.v4",
        "name": "MeguvizCAC",
        "version": args.version,
        "architecture": "v0.7.0-OmniVeritas(PickyBPE12k+Engram+NanoDraftV2+2L_Prelude[2SWA32+4BiDeltaProduct2+CappedReLU2+Funnel2to1+SeqLoRAMoE]+16Slot+2L_InfiLoopCore[MoR16to8]+UNetSpanMIL+TinyScopeCBM+EBT+LBDSM)",
        "license": "Apache-2.0",
        "tokenizer_file": tok_path.name,
        "vocab_size": tokenizer.vocab_size,
        "max_len": int(args.max_len),
        "bundle_v4_mgvz": exp_info["bundle_v4_mgvz"]["file"],
        "model_file": exp_info["model_int8"],
        "nano_draft_file": exp_info["nano_draft_file"],
        "nano_policy": nano_policy,
        "prototype_bank_file": proto_path.name,
        "lb_dsm_critic_file": lb_path.name,
        "ebt_critic_file": ebt_path.name,
        "temperature": round(t_baked, 6),
        "spline_temp_lut": spline_lut,
        "conformal": conf_val,
        "conformal_cal2400": conf_2400,
        "thresholds": {"f05": round(t_f05, 4), "fpr05": round(t_fpr05, 4), "score70": 0.70},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CPU ORT 1T Latency + Parity + Trap Suite
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

    # Package TRUE <= 2.20 MB Bundle v4 ZIP (bundle_v4_sub2bit.mgvz + compressed BPE12k + NanoDraft + Critics + manifest!)
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

    metrics = {
        "version": args.version,
        "schema_version": "meguviz.bundle.v4",
        "gpu_name": gpu_name,
        "batch_size_used": batch_size,
        "avg_tokens_bpe12k": round(avg_tok, 2),
        "total_params": total_params,
        "edge_params": edge_params,
        "fp16_master_eval": {"r2": fp16_test_m, "r4_mor8": fp16_r4_m},
        "sub2bit_bundle_v4_eval": {
            "test_bili_gold": test_m,
            "auroc_delta_vs_fp16": round(float(test_m["auroc"]) - float(fp16_test_m["auroc"]), 5),
            "pr_auc_delta_vs_fp16": round(float(test_m["pr_auc"]) - float(fp16_test_m["pr_auc"]), 5),
        },
        "test_bili_gold": test_m,
        "test_dsh_gold": dsh_m,
        "test_all_bili_distilled": all_m,
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
        "confounding_surgery": surgery_rep,
        "lb_dsm_critic": lb_rep,
        "ebt_critic": ebt_rep,
        "nano_draft": nano_rep,
        "spline_temp_lut": {"T_raw": spline_lut["T_raw"], "T_baked": spline_lut["T_baked"]},
        "conformal": conf_val,
        "conformal_cal2400": conf_2400,
        "traps": traps_rep,
        "v07_ci_gates": ci_final,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    metrics["release_gates"] = check_v07_release_gates(metrics)
    m_path = out_dir / f"metrics-{args.version}.json"
    m_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"=== MeguvizCAC {args.version} COMPLETE ({metrics['elapsed_seconds']}s) ===", flush=True)
    print(
        json.dumps(
            {
                "bundle_v4_total_mb": metrics["bundle_v4_total_mb"],
                "mgvz_breakdown_kb": exp_info["bundle_v4_mgvz"]["breakdown_kb"],
                "avg_tokens_bpe12k": metrics["avg_tokens_bpe12k"],
                "fp16_master_r2": {k: fp16_test_m[k] for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95")},
                "sub2bit_test_bili_gold": {k: test_m[k] for k in ("auroc", "pr_auc", "best_f05", "recall_at_p95")},
                "auroc_delta_vs_fp16": metrics["sub2bit_bundle_v4_eval"]["auroc_delta_vs_fp16"],
                "onnx_int8_1t_ms": metrics["onnx_int8"]["compact_batch32_1thread_ms"],
                "traps_summary": traps_rep["summary"],
                "v07_ci_all_passed": ci_final["all_ci_passed"],
                "release_gates": metrics["release_gates"],
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
    ap.add_argument("--intent-cache", type=Path, default=None)
    ap.add_argument("--teacher-v06-ckpt", type=Path, default=None, help="Unquantized BF16 17.65M v0.6 master teacher")
    ap.add_argument("--v06-backbone", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--out", type=Path, default=Path("./out_v07_omniveritas"))
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--version", default="v0.7.0-r2-omniveritas-b4")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_V07)
    ap.add_argument("--stage1-synth-cap", type=int, default=15000)
    ap.add_argument("--v04-synth-cap", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--stage15-steps", type=int, default=1800)
    ap.add_argument("--stage15-lr", type=float, default=1.1e-4)
    ap.add_argument("--teacher-v06-polish-steps", type=int, default=900)
    ap.add_argument("--upq-cap", type=int, default=480000)
    ap.add_argument("--w4-steps", type=int, default=1200)
    ap.add_argument("--sub2-steps", type=int, default=2200)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=7.5e-4)
    ap.add_argument("--w4-lr", type=float, default=8.5e-5)
    ap.add_argument("--sub2-lr", type=float, default=4.5e-5)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()
