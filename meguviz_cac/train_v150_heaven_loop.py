"""MeguvizCAC v1.5.0-HeavenLoop Student Trainer (Full Ablation Matrix + Group-OOF CharLM + TaH2 + 17/17 CI).

Supports every locked Antigravity x GPT ablation switch from a single unified CLI:
  - `--prelude-mode`: `single_0p6m` (default) | `stem_0p3m` | `dual_legacy` (H1 ablation)
  - `--no-coverage-beta`: disable decaying coverage inhibition (`beta=0` H2 control)
  - `--no-t25`: disable 37K-MAC 2-query `T2.5` Prosecutor/Defense light reread (9th blind spot control)
  - `--no-local-charlm` / `--shuffle-local-charlm`: H8 local 8-ch token LLR ablation + position-shuffle negative control
  - `--process-deltas` / `--shuffle-process-delta`: H7 SenseNova-Loop-9B Layer-27 process distillation + shuffled control
  - Strict Group-Clustered 3-Fold OOF `CompactCharLMV150` (jittered descendants ALWAYS score on parent's fold model!)
  - `TaH2` Student Lookahead Utility Supervision (`U_t = max_{j>t} {L(y, p_t) - L(y, p_j) - lam*(c_j - c_t)}`)
  - True Bootable `<= 5.00 MiB` Export (`MGVZ9 + ONNX skeletons + tokenizer + manifest + CharLM`) + 17/17 CI!
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v08_zenith_ops import OfflineEraGRLHead
from .v11_features import STYLE_FEAT_DIM_V11, extract_style_features_56d, plan_long_chunks_v11
from .v122_apexnano import fit_output_shifted_promo_lut_v122, fit_smooth_ai_lut_v122
from .v123_data_refinery import NUM_ERAS_V122
from .v130_xtreme_apex import AI_END_V130, ENGRAM_BUCKETS_V130, TRIGRAM_BUCKETS_V130
from .v140_paradise_apex import (
    CHARLM_FEAT_DIM_V140,
    HEAD_LEN_V140,
    MAX_LEN_V140,
    R0_STRUCT_DIM_V140,
    extract_apex_char_batch_v140,
)
from .v140_superteacher import (
    build_v140_superteacher_corpus,
    compute_top_tail_pairwise_loss_v140,
    fit_scalar_temperature_v140,
    jitter_ai_text_v140,
    read_jsonl_any_v140,
    sample_cf_mirror_batch_v140,
    text_sha16_v140,
)
from .train_v140_paradise_apex import (
    _eval_bin,
    _load_v140_distill_pool,
    _make_adamw,
    compute_progressive_kd_loss_v140,
)
from .v150_heaven_loop import (
    CORE_DEPTH_DEFAULT_V150,
    LOCAL_CHARLM_DIM_V150,
    CompactCharLMV150,
    MeguvizV150CascadeEngine,
    MeguvizV150HeavenLoopModel,
    canonicalize_zero_pii_v150,
    encode_bpe_head_tail_with_offsets_v150,
    export_v150_bootable_and_onnx,
    fit_group_oof_charlm_v150,
    run_v150_ci_gates,
    set_v150_qat_mode,
)


class _StrictGroupOOFFeatureCacheV150:
    """Strictly Leak-Free Feature Cache: jittered training rows ALWAYS score on their parent's exact OOF fold model!"""

    def __init__(
        self,
        tokenizer: BiliPickyBPETokenizer,
        max_len: int,
        head_len: int,
        use_local_charlm: bool = True,
        shuffle_local_charlm: bool = False,
        max_entries: int = 300000,
    ) -> None:
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.use_local_charlm = bool(use_local_charlm)
        self.shuffle_local_charlm = bool(shuffle_local_charlm)
        self.max_entries = int(max_entries)

        self.fold_models: list[CompactCharLMV150] = []
        self.text_to_fold: dict[str, int] = {}
        self.full_lm: CompactCharLMV150 | None = None

        self._bpe_cache: dict[str, tuple[str, list[int], list[tuple[int, int]], list[int], list[int], np.ndarray]] = {}
        self._lm_global_cache: dict[tuple[int, str], np.ndarray] = {}
        self._lm_local_cache: dict[tuple[int, str], np.ndarray] = {}

    def set_group_oof_charlm(
        self,
        fold_models: list[CompactCharLMV150],
        text_to_fold: dict[str, int],
        full_lm: CompactCharLMV150,
    ) -> None:
        self.fold_models = fold_models
        self.text_to_fold = text_to_fold
        self.full_lm = full_lm
        self._lm_global_cache.clear()
        self._lm_local_cache.clear()

    def _select_lm(self, parent_raw_text: str, use_oof: bool) -> tuple[int, CompactCharLMV150 | None]:
        if self.full_lm is None:
            return -1, None
        if use_oof and self.fold_models:
            fid = self.text_to_fold.get(parent_raw_text, -1)
            if 0 <= fid < len(self.fold_models):
                return fid, self.fold_models[fid]
            # Unseen distill-pool text: deterministic fold assignment (never seen by any fold model!)
            return -1, self.full_lm
        return -1, self.full_lm

    def get_bpe_style_offsets(self, raw_text: str, allow_store: bool = True):
        hit = self._bpe_cache.get(raw_text)
        if hit is not None:
            return hit
        cl, _ = canonicalize_zero_pii_v150(raw_text)
        _, f56, _ = extract_style_features_56d(cl)
        ids, offs = encode_bpe_head_tail_with_offsets_v150(self.tok, cl, max_len=self.max_len, head_len=self.head_len)
        eng = make_engram_bigram_ids(ids, ENGRAM_BUCKETS_V130)
        trg = make_engram_trigram_ids(ids, TRIGRAM_BUCKETS_V130)
        val = (cl, ids, offs, eng, trg, f56.astype(np.float32))
        if allow_store and len(self._bpe_cache) < self.max_entries:
            self._bpe_cache[raw_text] = val
        return val

    def get_global_9d(self, clean_text: str, parent_raw_text: str, use_oof: bool, allow_store: bool) -> np.ndarray:
        fid, lm = self._select_lm(parent_raw_text, use_oof=use_oof)
        if lm is None:
            return np.zeros((CHARLM_FEAT_DIM_V140,), dtype=np.float32)
        ck = (fid, clean_text)
        if allow_store:
            hit = self._lm_global_cache.get(ck)
            if hit is not None:
                return hit
        v = lm.extract_9d([clean_text])[0]
        if allow_store and len(self._lm_global_cache) < self.max_entries:
            self._lm_global_cache[ck] = v
        return v

    def get_local_8ch(
        self,
        clean_text: str,
        offsets: list[tuple[int, int]],
        parent_raw_text: str,
        use_oof: bool,
        allow_store: bool,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        L = len(offsets)
        if (not self.use_local_charlm) or self.full_lm is None:
            return np.zeros((L, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        fid, lm = self._select_lm(parent_raw_text, use_oof=use_oof)
        if lm is None:
            return np.zeros((L, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        ck = (fid, clean_text)
        arr = self._lm_local_cache.get(ck) if allow_store else None
        if arr is None or arr.shape[0] != L:
            arr = lm.extract_token_local_8ch(clean_text, offsets)
            if allow_store and len(self._lm_local_cache) < self.max_entries:
                self._lm_local_cache[ck] = arr
        if self.shuffle_local_charlm and L > 3:
            # H8 Negative Control: permute inner-token local LLR positions while preserving global distribution!
            out = arr.copy()
            perm = list(range(1, L - 1))
            if rng is not None:
                rng.shuffle(perm)
            else:
                random.Random(42).shuffle(perm)
            out[1 : L - 1] = arr[perm]
            return out
        return arr


def _collate_v150_batch(
    batch: list[dict[str, Any]],
    fcache: _StrictGroupOOFFeatureCacheV150,
    device: torch.device,
    rng: random.Random | None = None,
    punct_jitter_prob: float = 0.35,
    need_bpe: bool = True,
    use_oof_charlm: bool = True,
) -> dict[str, torch.Tensor]:
    cleans: list[str] = []
    parents: list[str] = []
    mutated: list[bool] = []
    g9_list: list[np.ndarray] = []

    bpe_pack = []
    for r in batch:
        t0 = str(r["text"])
        is_ai_pos = (r.get("has_ai", 0.0) > 0.5 and r.get("y_ai", 0.0) >= 0.70) or (r.get("has_teacher", 0.0) > 0.5 and r.get("t_ai_lg", -5.0) >= 1.2)
        if rng is not None and punct_jitter_prob > 0.0 and is_ai_pos:
            t1 = jitter_ai_text_v140(t0, rng, is_ml_en=bool(r.get("is_ml_en")), prob=0.28 if r.get("is_ml_en") else punct_jitter_prob)
            is_m = (t1 != t0)
        else:
            t1, is_m = t0, False
        cl, ids, offs, eng, trg, f56 = fcache.get_bpe_style_offsets(t1, allow_store=not is_m)
        cleans.append(cl)
        parents.append(t0)
        mutated.append(is_m)
        g9_list.append(fcache.get_global_9d(cl, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m))
        if need_bpe:
            loc8 = fcache.get_local_8ch(cl, offs, parent_raw_text=t0, use_oof=use_oof_charlm, allow_store=not is_m, rng=rng)
            bpe_pack.append((ids, eng, trg, f56, loc8))

    g9_arr = np.stack(g9_list, axis=0).astype(np.float32) if g9_list else np.zeros((0, CHARLM_FEAT_DIM_V140), dtype=np.float32)
    c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(cleans, already_canonical=True, precomputed_charlm_9d=g9_arr)

    out: dict[str, torch.Tensor] = {
        "c1": torch.from_numpy(c1).to(device, non_blocking=True),
        "c2": torch.from_numpy(c2).to(device, non_blocking=True),
        "c3": torch.from_numpy(c3).to(device, non_blocking=True),
        "cm": torch.from_numpy(cm).to(device, non_blocking=True),
        "cf": torch.from_numpy(cf).to(device, non_blocking=True),
        "sup": torch.from_numpy(sups).to(device, non_blocking=True),
        "t_ai_lg": torch.tensor([float(r.get("t_ai_lg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "t_ai_r0": torch.tensor([float(r.get("t_ai_r0", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t1": torch.tensor([float(r.get("t_ai_t1", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t2": torch.tensor([float(r.get("t_ai_t2", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "t_ai_t3": torch.tensor([float(r.get("t_ai_t3", r.get("t_ai_lg", 0.0))) for r in batch], dtype=torch.float32, device=device),
        "frontier_gap": torch.tensor([float(r.get("frontier_gap", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "proc_delta": torch.tensor([float(r.get("proc_delta", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_proc": torch.tensor([float(r.get("has_proc", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "t_pr_lg": torch.tensor([float(r.get("t_pr_lg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_t": torch.tensor([float(r.get("has_teacher", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "kd_w": torch.tensor([float(r.get("kd_w", 1.0)) for r in batch], dtype=torch.float32, device=device),
        "y_ai": torch.tensor([float(r.get("y_ai", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_ai": torch.tensor([float(r.get("has_ai", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "y_pr": torch.tensor([float(r.get("y_pr", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "has_pr": torch.tensor([float(r.get("has_pr", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "sub4": torch.tensor([int(r.get("sub4", -1)) for r in batch], dtype=torch.long, device=device),
        "hard_neg": torch.tensor([float(r.get("hard_neg", 0.0)) for r in batch], dtype=torch.float32, device=device),
        "era5": torch.tensor([int(r.get("era5", -1)) for r in batch], dtype=torch.long, device=device),
    }
    if need_bpe:
        ml = max(2, max(len(x[0]) for x in bpe_pack))
        if ml % 2 == 1:
            ml += 1
        nb = len(bpe_pack)
        ids_a = np.full((nb, ml), fcache.tok.pad_id, dtype=np.int64)
        msk_a = np.zeros((nb, ml), dtype=np.int64)
        eg_a = np.zeros((nb, ml), dtype=np.int64)
        tg_a = np.zeros((nb, ml), dtype=np.int64)
        loc_a = np.zeros((nb, ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
        f56_l = []
        for i, (ids, eng, trg, f56, loc8) in enumerate(bpe_pack):
            ns = len(ids)
            ids_a[i, :ns], msk_a[i, :ns], eg_a[i, :ns], tg_a[i, :ns] = ids, 1, eng, trg
            loc_a[i, :ns] = loc8
            f56_l.append(f56)
        out["ids"] = torch.from_numpy(ids_a).to(device, non_blocking=True)
        out["msk"] = torch.from_numpy(msk_a).to(device, non_blocking=True)
        out["eg"] = torch.from_numpy(eg_a).to(device, non_blocking=True)
        out["tg"] = torch.from_numpy(tg_a).to(device, non_blocking=True)
        out["loc"] = torch.from_numpy(loc_a).to(device, non_blocking=True)
        out["st"] = torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device, non_blocking=True)
    return out


@torch.no_grad()
def predict_student_v150(
    model: MeguvizV150HeavenLoopModel,
    fcache: _StrictGroupOOFFeatureCacheV150,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
    r0_only: bool = False,
    outer_steps: int = 4,
) -> dict[str, np.ndarray]:
    n = len(texts)
    keys = (
        ("logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0")
        if r0_only
        else (
            "logit_ai_r0", "logit_promo_r0", "p_defer_ai_r0", "p_defer_pr_r0",
            "logit_ai_t1", "logit_ai_t2_final", "logit_ai_t25_final", "logit_ai_t3_final",
            "logit_ai", "logit_promo", "p_escalate_t3", "p_escalate_t4",
        )
    )
    out = {k: np.zeros((n,), dtype=np.float32) for k in keys}
    if n == 0:
        return out
    model.eval()
    order = sorted(range(n), key=lambda i: len(texts[i]))
    for st in range(0, n, batch_size):
        idxs = order[st : st + batch_size]
        sub_b = [{"text": texts[i]} for i in idxs]
        bk = _collate_v150_batch(sub_b, fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=not r0_only, use_oof_charlm=False)
        if r0_only:
            res = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"])
        else:
            res = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"], outer_steps=outer_steps,
            )
        for k in keys:
            if k in res:
                arr = res[k].float().cpu().numpy()
                for j, oi in enumerate(idxs):
                    out[k][oi] = arr[j]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MeguvizCAC v1.5.0-HeavenLoop Student (<=5.00 MiB Bootable)")
    parser.add_argument("--distill-pool", type=Path, default=None)
    parser.add_argument("--process-deltas", type=Path, default=None, help="Optional v150_sensenova_process_deltas.npz (H7)")
    parser.add_argument("--shuffle-process-delta", action="store_true", help="H7 negative control: shuffle teacher delta")
    parser.add_argument("--v04-data", type=Path, default=None)
    parser.add_argument("--sibling-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None)
    parser.add_argument("--gold-v2-dir", type=Path, required=True)
    parser.add_argument("--wave3-dir", type=Path, required=True)
    parser.add_argument("--v11-data-dir", type=Path, required=True)
    parser.add_argument("--v07-tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--version", type=str, default="v1.5.0-r1-heaven-loop")
    parser.add_argument("--prelude-mode", type=str, default="single_0p6m", choices=("single_0p6m", "stem_0p3m", "dual_legacy"))
    parser.add_argument("--core-depth", type=int, default=CORE_DEPTH_DEFAULT_V150, help="Shared recurrent layers inside HeavenLoopCoreV150 (default 2)")
    parser.add_argument("--long-char-trigger", type=int, default=300, help="Post-canonical char length trigger for DynaSeam LongDoc")
    parser.add_argument("--no-coverage-beta", action="store_true")
    parser.add_argument("--no-t25", action="store_true")
    parser.add_argument("--no-local-charlm", action="store_true")
    parser.add_argument("--use-global-charlm", action="store_true", help="Ablation: re-enable global CharLM shortcut (off by default after v1.4.0 R@P99 audit)")
    parser.add_argument("--shuffle-local-charlm", action="store_true")
    parser.add_argument("--p1-steps", type=int, default=2400)
    parser.add_argument("--p2-steps", type=int, default=4800)
    parser.add_argument("--p3-qat-steps", type=int, default=1800)
    parser.add_argument("--r0-drop-p", type=float, default=0.65)
    parser.add_argument("--charlm-drop-p", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=MAX_LEN_V140)
    parser.add_argument("--head-len", type=int, default=HEAD_LEN_V140)
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
    tokenizer = BiliPickyBPETokenizer(args.v07_tokenizer)
    fcache = _StrictGroupOOFFeatureCacheV150(
        tokenizer, max_len=args.max_len, head_len=args.head_len,
        use_local_charlm=not args.no_local_charlm, shuffle_local_charlm=args.shuffle_local_charlm,
    )
    rng = random.Random(20261201)
    torch.manual_seed(20261201)
    np.random.seed(20261201)

    corpus = build_v140_superteacher_corpus(
        sibling_dir=args.sibling_dir, delta_dir=args.delta_dir,
        gold_v2_dir=args.gold_v2_dir, wave3_dir=args.wave3_dir,
        v11_dir=args.v11_data_dir, v04_data_path=args.v04_data, smoke_test=args.smoke_test,
    )
    blocked = {
        text_sha16_v140(r["text"])
        for k in ("val_gold_v2", "test_gold_v2", "val_wave3", "val_promo_v11", "val_old_video_neg", "val_multilingual", "diag_god500")
        for r in corpus.get(k, [])
    }
    distill_pool = _load_v140_distill_pool(args.distill_pool, blocked, smoke_test=args.smoke_test)

    # H7 SenseNova-Loop-9B Layer-27 Process-Distillation Deltas joined strictly by exact text_sha16 hash!
    proc_attached = 0
    if args.process_deltas is not None and args.process_deltas.is_file() and distill_pool:
        pd = np.load(args.process_deltas, allow_pickle=False)
        deltas = pd["delta_t"].astype(np.float32)
        if "text_hashes" in pd.files:
            th_arr = pd["text_hashes"]
            if args.shuffle_process_delta:
                deltas = deltas.copy()
                np.random.default_rng(20261201).shuffle(deltas)
            h_to_delta = {str(th_arr[i]): float(deltas[i]) for i in range(min(len(th_arr), len(deltas)))}
            for r in distill_pool:
                h = text_sha16_v140(str(r["text"]))
                if h in h_to_delta:
                    r["proc_delta"] = h_to_delta[h]
                    r["has_proc"] = 1.0
                    proc_attached += 1
        elif len(deltas) == len(distill_pool):
            d_use = deltas.copy()
            if args.shuffle_process_delta:
                np.random.default_rng(20261201).shuffle(d_use)
            for i, r in enumerate(distill_pool):
                r["proc_delta"] = float(d_use[i])
                r["has_proc"] = 1.0
            proc_attached = len(distill_pool)
        else:
            raise RuntimeError(
                f"Process-delta file {args.process_deltas} lacks text_hashes and length {len(deltas)} != {len(distill_pool)}!"
            )

    # Stage 0: Fit Strict Group-Clustered 3-Fold OOF `CompactCharLMV150`
    gold_pos, gold_nat, gold_hd, gold_pr = corpus["train_ai_pos"], corpus["train_hu_natural"], corpus["train_hu_hard"], corpus["train_promo_extra"]
    lm_rows = gold_pos + gold_nat + gold_hd
    lm_txts = [r["text"] for r in lm_rows]
    lm_y = np.array([1 if r["y_ai"] >= 0.5 else 0 for r in lm_rows], dtype=np.int32)
    lm_groups = [str(r.get("group_id") or r.get("template_id") or "") or None for r in lm_rows]
    t_lm = time.time()
    fold_lms, text_to_fold, full_charlm = fit_group_oof_charlm_v150(
        lm_txts, lm_y, group_keys=lm_groups, n_folds=3,
        min_counts=(2, 3, 4) if args.smoke_test else (16, 28, 40),
        max_order_keys=(3000, 1500, 1000) if args.smoke_test else (9000, 8000, 7000),
        seed=20261201,
    )
    fcache.set_group_oof_charlm(fold_lms, text_to_fold, full_charlm)
    print(
        f"[v1.5.0-HeavenLoop] prelude={args.prelude_mode} core_depth={args.core_depth} cov_beta={not args.no_coverage_beta} "
        f"t25={not args.no_t25} local_lm={not args.no_local_charlm} global_lm={args.use_global_charlm} "
        f"proc_deltas={proc_attached:,}/{len(distill_pool):,} | GroupOOF-CharLM fitted in {time.time()-t_lm:.1f}s (keys={full_charlm.total_keys():,})",
        flush=True,
    )

    d_pos = [r for r in distill_pool if r["t_ai_lg"] >= 0.8]
    d_bnd = [r for r in distill_pool if -1.2 < r["t_ai_lg"] < 0.8]
    d_front = [r for r in distill_pool if r["frontier_gap"] >= 1.4 or abs(r.get("proc_delta", 0.0)) >= 0.7]
    d_pr = [r for r in distill_pool if r["t_pr_lg"] >= 0.5]
    # v1.4.0 Fix #5: Dedicated high-leverage buckets targeting God-500 D12 (Q2 dual-high), D06/D08 (Long), D04 (Short Stealth)!
    d_q2 = [r for r in distill_pool if r["t_ai_lg"] >= 0.4 and r["t_pr_lg"] >= 0.4]
    d_long = [r for r in distill_pool if len(str(r["text"])) >= 160]
    d_short_pos = [r for r in distill_pool if len(str(r["text"])) <= 45 and r["t_ai_lg"] >= 0.6]
    d_hn = [r for r in distill_pool if r["hard_neg"] > 0.5]
    d_nat = [r for r in distill_pool if r["t_ai_lg"] <= -1.2 and r["hard_neg"] < 0.5]
    longform_train = read_jsonl_any_v140(args.v11_data_dir / "v11_wave2_longform_gold_1200.jsonl.gz")[: (24 if args.smoke_test else 1200)]

    def _draw(pool, k):
        return rng.choices(pool, k=k) if (pool and k > 0) else []

    def _sample_batch(bs_cur: int) -> list[dict[str, Any]]:
        out_b: list[dict[str, Any]] = []
        if distill_pool:
            out_b.extend(_draw(d_pos, max(1, int(bs_cur * 0.09))))
            out_b.extend(_draw(d_bnd, max(1, int(bs_cur * 0.09))))
            if d_front:
                out_b.extend(_draw(d_front, max(1, int(bs_cur * 0.09))))
            if d_q2:
                out_b.extend(_draw(d_q2, max(1, int(bs_cur * 0.04))))
            if d_long:
                out_b.extend(_draw(d_long, max(1, int(bs_cur * 0.05))))
            if d_short_pos:
                out_b.extend(_draw(d_short_pos, max(1, int(bs_cur * 0.05))))
            out_b.extend(_draw(d_pr, max(1, int(bs_cur * 0.04))))
            out_b.extend(_draw(d_hn, max(1, int(bs_cur * 0.09))))
            out_b.extend(_draw(d_nat, max(1, int(bs_cur * 0.09))))
        rem = max(4, bs_cur - len(out_b))
        out_b.extend(_draw(gold_pos, max(1, int(rem * 0.28))))
        out_b.extend(_draw(gold_hd, max(1, int(rem * 0.30))))
        out_b.extend(_draw(gold_nat, max(1, int(rem * 0.34))))
        out_b.extend(_draw(gold_pr, max(0, bs_cur - len(out_b))))
        fb = gold_nat or gold_pos or distill_pool
        while len(out_b) < bs_cur and fb:
            out_b.append(rng.choice(fb))
        return out_b[:bs_cur]

    model = MeguvizV150HeavenLoopModel(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len, style_dim=STYLE_FEAT_DIM_V11,
        prelude_mode=args.prelude_mode, core_depth=int(args.core_depth),
        coverage_beta_enabled=not args.no_coverage_beta,
        use_t25_light_reread=not args.no_t25,
        use_local_charlm_inject=not args.no_local_charlm,
        use_global_charlm_shortcut=bool(args.use_global_charlm),
    ).to(device)
    era_grl = OfflineEraGRLHead(emb_dim=AI_END_V130, num_eras=NUM_ERAS_V122).to(device)
    tot_p = sum(p.numel() for p in model.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    print(f"[Stage0-Model] total_params={tot_p:,} | heaven_core={core_p:,} ({core_p/tot_p*100:.1f}%) | core_depth={args.core_depth}", flush=True)

    def _eval_val(m_cur: MeguvizV150HeavenLoopModel, r0_only: bool = False) -> dict[str, Any]:
        p_g2 = predict_student_v150(m_cur, fcache, [r["text"] for r in corpus["val_gold_v2"]], device, r0_only=r0_only, outer_steps=4)
        p_w3 = predict_student_v150(m_cur, fcache, [r["text"] for r in corpus["val_wave3"]], device, r0_only=r0_only, outer_steps=4)
        p_old = predict_student_v150(m_cur, fcache, [r["text"] for r in corpus["val_old_video_neg"]], device, r0_only=r0_only, outer_steps=4)
        yg2 = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_gold_v2"]], dtype=np.int32)
        yw3_a = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        yw3_p = np.array([int(r["y_pr"] >= 0.5) for r in corpus["val_wave3"]], dtype=np.int32)
        ka = "logit_ai_r0" if r0_only else "logit_ai"
        kp = "logit_promo_r0" if r0_only else "logit_promo"
        mg2, mw3a, mw3p = _eval_bin(yg2, p_g2[ka]), _eval_bin(yw3_a, p_w3[ka]), _eval_bin(yw3_p, p_w3[kp])
        old_fpr = float((expit(np.clip(p_old[ka], -30, 30)) >= 0.5).mean()) if len(p_old[ka]) else 0.0
        q4 = (yw3_p == 1) & (yw3_a == 0)
        q4_fpr = float((expit(np.clip(p_w3[ka][q4], -30, 30)) >= 0.5).mean()) if q4.sum() else 0.0

        ml_pr = 0.0
        if corpus.get("val_multilingual"):
            p_ml = predict_student_v150(m_cur, fcache, [r["text"] for r in corpus["val_multilingual"]], device, r0_only=r0_only, outer_steps=4)
            yml = np.array([int(r["y_ai"] >= 0.5) for r in corpus["val_multilingual"]], dtype=np.int32)
            ml_pr = _eval_bin(yml, p_ml[ka])["pr"]

        obj = 0.30 * mg2["pr"] + 0.24 * mg2["r95"] + 0.18 * mw3a["pr"] + 0.18 * mw3p["pr"] + 0.10 * ml_pr - 1.30 * old_fpr - 0.55 * q4_fpr
        res_d: dict[str, Any] = {
            "sel_obj": round(float(obj), 5), "g2_pr": mg2["pr"], "g2_r95": mg2["r95"],
            "w3_ai_pr": mw3a["pr"], "w3_pr_pr": mw3p["pr"], "ml_pr": ml_pr,
            "old_fpr": round(old_fpr, 4), "q4_fpr": round(q4_fpr, 4),
        }
        if not r0_only:
            mg2_t2 = _eval_bin(yg2, p_g2["logit_ai_t2_final"])
            mg2_t3 = _eval_bin(yg2, p_g2["logit_ai_t3_final"])
            res_d["g2_t2_pr"] = mg2_t2["pr"]
            res_d["t4_minus_t2_pr"] = round(mg2["pr"] - mg2_t2["pr"], 4)
            res_d["t3_minus_t2_pr"] = round(mg2_t3["pr"] - mg2_t2["pr"], 4)
        return res_d

    def _cosine_mult(step_i: int, total_s: int, warm_frac: float = 0.08, min_f: float = 0.06) -> float:
        warm = max(1, int(total_s * warm_frac))
        w = min(1.0, (step_i + 1) / float(warm))
        c = min_f + (1.0 - min_f) * 0.5 * (1.0 + math.cos(math.pi * step_i / float(max(1, total_s))))
        return w * c

    bs = 16 if args.smoke_test else int(args.batch_size)
    p1_steps = 4 if args.smoke_test else int(args.p1_steps)
    p2_steps = 6 if args.smoke_test else int(args.p2_steps)
    p3_steps = 4 if args.smoke_test else int(args.p3_qat_steps)

    # Phase 1: Decoupled R0 KD (separate pure-AI & pure-Promo defer heads, cosine schedule!)
    opt1 = _make_adamw([{"params": list(model.rung0.parameters()), "lr": 1.8e-3, "base_lr": 1.8e-3, "weight_decay": 0.01}], device)
    best_r0_sd, best_r0_obj = {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}, -1e9
    model.train()
    for step in range(p1_steps):
        mult1 = _cosine_mult(step, p1_steps)
        for pg in opt1.param_groups:
            pg["lr"] = pg["base_lr"] * mult1
        bk = _collate_v150_batch(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=False)
        opt1.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            cf_in = bk["cf"]
            if float(args.charlm_drop_p) > 0.0:
                lm_k = (torch.rand((cf_in.shape[0], 1), device=cf_in.device) >= float(args.charlm_drop_p)).to(cf_in.dtype)
                cf_in = torch.cat([cf_in[:, :R0_STRUCT_DIM_V140], cf_in[:, R0_STRUCT_DIM_V140:] * lm_k], dim=-1)
            r0 = model.forward_r0(bk["c1"], bk["c2"], bk["c3"], bk["cm"], cf_in)
            l_ai = compute_progressive_kd_loss_v140(r0["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
            l_pr = compute_progressive_kd_loss_v140(r0["logit_promo_r0"], bk, axis="promo")
            with torch.no_grad():
                tp_ai = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
                tp_pr = torch.where(bk["has_pr"] > 0.5, bk["y_pr"], torch.sigmoid(bk["t_pr_lg"]))
                err_ai = ((torch.sigmoid(r0["logit_ai_r0"].float()) - tp_ai).abs() >= 0.20).float()
                err_pr = ((torch.sigmoid(r0["logit_promo_r0"].float()) - tp_pr).abs() >= 0.20).float()
            l_def = F.binary_cross_entropy_with_logits(r0["defer_ai_r0_logit"].float(), err_ai) + F.binary_cross_entropy_with_logits(r0["defer_pr_r0_logit"].float(), err_pr)
            loss = l_ai + 0.72 * l_pr + 0.15 * l_def
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.rung0.parameters(), 1.0)
        opt1.step()
        if (step + 1) % max(1, p1_steps // 4) == 0 or step == p1_steps - 1:
            ev1 = _eval_val(model, r0_only=True)
            print(f"  [Phase1-R0 step={step+1}/{p1_steps}] loss={float(loss.item()):.4f} {ev1}", flush=True)
            if ev1["sel_obj"] > best_r0_obj:
                best_r0_obj, best_r0_sd = ev1["sel_obj"], {k: v.detach().cpu().clone() for k, v in model.rung0.state_dict().items()}
            model.train()
    model.rung0.load_state_dict(best_r0_sd)

    def _heaven_loss(od: dict[str, torch.Tensor], bk: dict[str, torch.Tensor], outer_n: int) -> torch.Tensor:
        with torch.no_grad():
            t_p_ens = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_lg"]))
            t_p_sn9 = torch.where(bk["has_ai"] > 0.5, bk["y_ai"], torch.sigmoid(bk["t_ai_t3"]))
            err_r0 = (torch.sigmoid(od["logit_ai_r0"].float()) - t_p_ens).abs()
            t2_w = 1.0 + 1.8 * (err_r0 >= 0.15).float()
            t2_w = t2_w / t2_w.mean().clamp_min(1.0)
            ai_valid = (bk["has_ai"] > 0.5) | (bk["has_t"] > 0.5)
            y_bin_ai = (t_p_sn9 >= 0.5).float()

        # v1.4.0 Fix #2: Strong standalone T1 & T2_raw supervision against SenseNova-Loop-9B -> eliminates R0 gradient starvation!
        loss = (
            1.00 * compute_progressive_kd_loss_v140(od["logit_ai_t2_final"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=t2_w)
            + 0.55 * compute_progressive_kd_loss_v140(od["logit_ai_t2"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=t2_w)
            + 0.85 * compute_progressive_kd_loss_v140(od["logit_ai_t25_final"], bk, axis="ai", teacher_key="t_ai_t2", sample_weight=t2_w)
            + 0.35 * compute_progressive_kd_loss_v140(od["logit_ai_t1"], bk, axis="ai", teacher_key="t_ai_t1")
            + 0.20 * compute_progressive_kd_loss_v140(od["logit_ai_r1"], bk, axis="ai", teacher_key="t_ai_t1")
            + 0.18 * compute_progressive_kd_loss_v140(od["logit_ai_r0"], bk, axis="ai", teacher_key="t_ai_r0")
            + 0.68 * compute_progressive_kd_loss_v140(od["logit_promo_t2_final"], bk, axis="promo")
            + 0.20 * compute_progressive_kd_loss_v140(od["logit_promo_r0"], bk, axis="promo")
            + 0.18 * compute_progressive_kd_loss_v140(od["ebt_margin"], bk, axis="ai", teacher_key="t_ai_t2")
            + 0.20 * compute_top_tail_pairwise_loss_v140(od["logit_ai_t25_final"], y_bin_ai, ai_valid)
        )

        s4_m = (bk["has_pr"] > 0.5) & (bk["sub4"] >= 0) & (bk["sub4"] < 4)
        if s4_m.any():
            loss = loss + 0.10 * F.cross_entropy(od["promo_sub4_logit"].float()[s4_m], bk["sub4"][s4_m])

        # H7 SenseNova-Loop-9B Layer-27 Bidirectional Process-Distillation Delta (`deep - shallow` aligns to `Delta_T`!)
        pm = bk["has_proc"] > 0.5
        if pm.any():
            stu_delta_fin = (od["logit_ai"].float() - od["logit_ai_t1"].float()).clamp(-3.5, 3.5)
            loss = loss + 0.25 * F.smooth_l1_loss(stu_delta_fin[pm], bk["proc_delta"][pm], beta=0.5)

        if outer_n >= 3:
            with torch.no_grad():
                p_t25 = torch.sigmoid(od["logit_ai_t25_final"].float())
                err_t25 = (p_t25 - t_p_sn9).abs()
                l_t25 = F.binary_cross_entropy_with_logits(od["logit_ai_t25_final"].float(), t_p_sn9, reduction="none")
                # v1.4.0 Fix #3: Explicit +5.0x Hard-Human Top-Tail FP Reversal weight so T3 actively EXONERATES humans!
                hard_hu_fp_25 = ((t_p_sn9 <= 0.15) & (p_t25 >= 0.12)).float()
                hard_ai_fn_25 = ((t_p_sn9 >= 0.75) & (p_t25 <= 0.70)).float()
                t3_w = 1.0 + 2.5 * (err_t25 >= 0.12).float() + 5.0 * hard_hu_fp_25 + 2.5 * hard_ai_fn_25 + 1.2 * (bk["frontier_gap"] >= 1.4).float()
                t3_w = t3_w / t3_w.mean().clamp_min(1.0)

            loss = loss + 1.10 * compute_progressive_kd_loss_v140(od["logit_ai_t3_final"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t3_w)
            loss = loss + 0.40 * compute_progressive_kd_loss_v140(od["logit_ai_t3"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t3_w)
            loss = loss + 0.55 * compute_progressive_kd_loss_v140(od["logit_promo_t3_final"], bk, axis="promo")
            loss = loss + 0.25 * compute_top_tail_pairwise_loss_v140(od["logit_ai_t3_final"], y_bin_ai, ai_valid)

            err_t3 = (torch.sigmoid(od["logit_ai_t3_final"].float()) - t_p_sn9).abs()
            hm3 = ((err_t25 >= 0.12) | (hard_hu_fp_25 > 0.5) | (bk["frontier_gap"] >= 1.4)) & ai_valid
            if hm3.any():
                loss = loss + 0.45 * F.relu(err_t3[hm3] - err_t25[hm3] + 0.025).mean()
            if pm.any():
                stu_delta_3 = (od["logit_ai_t3_final"].float() - od["logit_ai_t1"].float()).clamp(-3.5, 3.5)
                loss = loss + 0.15 * F.smooth_l1_loss(stu_delta_3[pm], bk["proc_delta"][pm], beta=0.5)

            if outer_n >= 4:
                with torch.no_grad():
                    p_t3 = torch.sigmoid(od["logit_ai_t3_final"].float())
                    err_t3_d = (p_t3 - t_p_sn9).abs()
                    l_t3 = F.binary_cross_entropy_with_logits(od["logit_ai_t3_final"].float(), t_p_sn9, reduction="none")
                    hard_hu_fp_3 = ((t_p_sn9 <= 0.15) & (p_t3 >= 0.12)).float()
                    hard_ai_fn_3 = ((t_p_sn9 >= 0.75) & (p_t3 <= 0.70)).float()
                    t4_w = 1.0 + 3.0 * (err_t3_d >= 0.10).float() + 5.0 * hard_hu_fp_3 + 2.5 * hard_ai_fn_3
                    t4_w = t4_w / t4_w.mean().clamp_min(1.0)

                loss = loss + 1.20 * compute_progressive_kd_loss_v140(od["logit_ai_t4_final"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t4_w)
                loss = loss + 0.45 * compute_progressive_kd_loss_v140(od["logit_ai_t4"], bk, axis="ai", teacher_key="t_ai_t3", sample_weight=t4_w)
                loss = loss + 0.50 * compute_progressive_kd_loss_v140(od["logit_promo_t4_final"], bk, axis="promo")
                loss = loss + 0.25 * compute_top_tail_pairwise_loss_v140(od["logit_ai_t4_final"], y_bin_ai, ai_valid)

                err_t4 = (torch.sigmoid(od["logit_ai_t4_final"].float()) - t_p_sn9).abs()
                hm4 = ((err_t3_d >= 0.10) | (hard_hu_fp_3 > 0.5)) & ai_valid
                if hm4.any():
                    loss = loss + 0.45 * F.relu(err_t4[hm4] - err_t3_d[hm4] + 0.025).mean()

                # `TaH2` Student Lookahead Utility Supervision (`U_t = max_{j>t} {L_t - L_j - lam*(c_j - c_t)} > 0`!)
                with torch.no_grad():
                    l_t4 = F.binary_cross_entropy_with_logits(od["logit_ai_t4_final"].float(), t_p_sn9, reduction="none")
                    u_t25 = torch.maximum(l_t25 - l_t3 - 0.015, l_t25 - l_t4 - 0.030)
                    u_t3 = l_t3 - l_t4 - 0.018
                    y_esc3 = (u_t25 > 0.0).float()
                    y_esc4 = (u_t3 > 0.0).float()
                loss = loss + 0.22 * F.binary_cross_entropy_with_logits(od["escalate_t3_logit"].float(), y_esc3)
                loss = loss + 0.22 * F.binary_cross_entropy_with_logits(od["escalate_t4_logit"].float(), y_esc4)

        hu_m = ((bk["has_ai"] > 0.5) & (bk["y_ai"] <= 0.20)) | ((bk["has_t"] > 0.5) & (bk["t_ai_lg"] <= -1.5))
        era_v = hu_m & (bk["era5"] >= 0) & (bk["era5"] < NUM_ERAS_V122)
        if int(era_v.sum().item()) >= 8:
            loss = loss + 0.12 * era_grl(od["emb_ai"], bk["era5"], era_v, lam=0.50)
        return loss

    # Phase 2: Full `T_max = 4` FP16/BF16 Training + CF Mirrors + Inline DynaSeam LongDoc & Synthetic Sandwich!
    r0_ids = {id(p) for p in model.rung0.parameters()}
    opt2 = _make_adamw([
        {"params": list(model.rung0.parameters()), "lr": 2.4e-4, "base_lr": 2.4e-4, "weight_decay": 0.01},
        {"params": [p for p in model.parameters() if id(p) not in r0_ids] + list(era_grl.parameters()), "lr": 1.0e-3, "base_lr": 1.0e-3, "weight_decay": 0.01},
    ], device)
    ema_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_p2_sd, best_p2_obj = {k: v.clone() for k, v in ema_sd.items()}, -1e9
    best_p2_ev: dict[str, Any] = {}
    has_cf = corpus["stats"]["admitted_cf_pairs"] > 0

    model.train()
    for step in range(p2_steps):
        mult2 = _cosine_mult(step, p2_steps)
        for pg in opt2.param_groups:
            pg["lr"] = pg["base_lr"] * mult2
        bk = _collate_v150_batch(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        outer_n = 4 if (step % 2 == 0) else 2
        opt2.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=outer_n, r0_drop_p=float(args.r0_drop_p), charlm_drop_p=float(args.charlm_drop_p),
            )
            loss = _heaven_loss(od, bk, outer_n=outer_n)
            if has_cf and (step % 3 == 0):
                ch = sample_cf_mirror_batch_v140(corpus, 4 if args.smoke_test else 16, rng)
                if ch:
                    ba = _collate_v150_batch([{"text": x["ai_text"]} for x in ch], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    bh = _collate_v150_batch([{"text": x["human_text"]} for x in ch], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oa = model(ba["ids"], ba["msk"], ba["st"], ba["eg"], ba["tg"], ba["loc"], ba["c1"], ba["c2"], ba["c3"], ba["cm"], ba["cf"], ba["sup"], outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.35)
                    oh = model(bh["ids"], bh["msk"], bh["st"], bh["eg"], bh["tg"], bh["loc"], bh["c1"], bh["c2"], bh["c3"], bh["cm"], bh["cf"], bh["sup"], outer_steps=4, r0_drop_p=0.50, charlm_drop_p=0.35)
                    pw = torch.tensor([float(x["pair_weight"]) for x in ch], dtype=torch.float32, device=device)
                    l_cf = (
                        (F.softplus(1.8 - (oa["logit_ai"].float() - oh["logit_ai"].float())) * pw).mean()
                        + 0.60 * (F.softplus(1.7 - (oa["logit_ai_t3_final"].float() - oh["logit_ai_t3_final"].float())) * pw).mean()
                        + 0.50 * (F.softplus(1.6 - (oa["logit_ai_t25_final"].float() - oh["logit_ai_t25_final"].float())) * pw).mean()
                    )
                    loss = loss + 0.26 * l_cf

            # v1.4.0 Fix #5: Inline DynaSeam LongDoc + Synthetic Human/AI Sandwich Training (fixes D06 & D08!)
            if longform_train and (step % 6 == 0):
                docs = rng.sample(longform_train, min(2 if args.smoke_test else 6, len(longform_train)))
                d_losses = []
                for d_it in docs:
                    cl_doc, _ = canonicalize_zero_pii_v150(str(d_it.get("text", "")))
                    chunks = plan_long_chunks_v11(cl_doc, max_chunks=6)
                    if not chunks:
                        continue
                    bk_c = _collate_v150_batch([{"text": c["text"]} for c in chunks], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                    oc = model(bk_c["ids"], bk_c["msk"], bk_c["st"], bk_c["eg"], bk_c["tg"], bk_c["loc"], bk_c["c1"], bk_c["c2"], bk_c["c3"], bk_c["cm"], bk_c["cf"], bk_c["sup"], outer_steps=2, r0_drop_p=0.0)
                    cm_t = torch.from_numpy(np.stack([c["chunk_meta"] for c in chunks], 0).astype(np.float32)).unsqueeze(0).to(device)
                    vm_t = torch.ones((1, len(chunks)), dtype=torch.float32, device=device)
                    da, dp = model.aggregate_long_chunks(oc["emb_ai"].unsqueeze(0), oc["emb_pr"].unsqueeze(0), oc["logit_ai"].unsqueeze(0), oc["logit_promo"].unsqueeze(0), cm_t, vm_t)
                    y_da = torch.tensor([1.0 if float(d_it.get("ai_score", d_it.get("score", 0.0))) >= 65.0 else 0.0], dtype=torch.float32, device=device)
                    y_dp = torch.tensor([1.0 if float(d_it.get("promo_score", 0.0)) >= 65.0 else 0.0], dtype=torch.float32, device=device)
                    d_losses.append(F.binary_cross_entropy_with_logits(da.float(), y_da) + 0.50 * F.binary_cross_entropy_with_logits(dp.float(), y_dp))
                # Synthetic Human-Wrapper + AI-Core Sandwich Doc (`D08_HYB_sandwich` direct supervision!)
                if gold_pos and gold_nat and len(d_losses) > 0:
                    h_wrap = canonicalize_zero_pii_v150(str(rng.choice(gold_nat)["text"]))[0]
                    a_core = canonicalize_zero_pii_v150(str(rng.choice(gold_pos)["text"]))[0]
                    h_tail = canonicalize_zero_pii_v150(str(rng.choice(gold_nat)["text"]))[0]
                    sand_doc = f"{h_wrap}\n{a_core}\n{h_tail}"
                    s_chunks = plan_long_chunks_v11(sand_doc, max_chunks=4)
                    if len(s_chunks) >= 2:
                        bk_s = _collate_v150_batch([{"text": c["text"]} for c in s_chunks], fcache, device, rng=None, punct_jitter_prob=0.0, need_bpe=True)
                        os_c = model(bk_s["ids"], bk_s["msk"], bk_s["st"], bk_s["eg"], bk_s["tg"], bk_s["loc"], bk_s["c1"], bk_s["c2"], bk_s["c3"], bk_s["cm"], bk_s["cf"], bk_s["sup"], outer_steps=2, r0_drop_p=0.0)
                        cms = torch.from_numpy(np.stack([c["chunk_meta"] for c in s_chunks], 0).astype(np.float32)).unsqueeze(0).to(device)
                        vms = torch.ones((1, len(s_chunks)), dtype=torch.float32, device=device)
                        da_s, _ = model.aggregate_long_chunks(os_c["emb_ai"].unsqueeze(0), os_c["emb_pr"].unsqueeze(0), os_c["logit_ai"].unsqueeze(0), os_c["logit_promo"].unsqueeze(0), cms, vms)
                        d_losses.append(F.binary_cross_entropy_with_logits(da_s.float(), torch.ones((1,), dtype=torch.float32, device=device)))
                if d_losses:
                    loss = loss + 0.22 * torch.stack(d_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(era_grl.parameters()), 1.0)
        opt2.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_sd[k].mul_(0.995).add_(v.detach().cpu(), alpha=0.005)
        if (step + 1) % max(1, p2_steps // 6) == 0 or step == p2_steps - 1:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_sd)
            ev2 = _eval_val(model, r0_only=False)
            if ev2["sel_obj"] > best_p2_obj:
                best_p2_obj, best_p2_sd, best_p2_ev = ev2["sel_obj"], {k: v.clone() for k, v in ema_sd.items()}, ev2
            print(f"  [Phase2 step={step+1}/{p2_steps}] lr={opt2.param_groups[1]['lr']:.2e} loss={float(loss.item()):.4f} {ev2}", flush=True)
            model.load_state_dict(cur_sd)
            model.train()
    model.load_state_dict(best_p2_sd)
    torch.save({"version": args.version, "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_p2_sd.items()}, "val": best_p2_ev}, ckpt_dir / f"{args.version}_fp16_master.pt")

    # Phase 3: LoopQ-Aware Clean W4/W8 QAT (`T=4` unrolled, cosine decay!)
    set_v150_qat_mode(model, True)
    opt3 = _make_adamw([{"params": list(model.parameters()) + list(era_grl.parameters()), "lr": 2.2e-4, "base_lr": 2.2e-4, "weight_decay": 0.005}], device)
    ema_q = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_q_sd, best_q_obj = {k: v.clone() for k, v in ema_q.items()}, _eval_val(model, r0_only=False)["sel_obj"]
    model.train()
    for step in range(p3_steps):
        mult3 = _cosine_mult(step, p3_steps, warm_frac=0.06, min_f=0.08)
        for pg in opt3.param_groups:
            pg["lr"] = pg["base_lr"] * mult3
        bk = _collate_v150_batch(_sample_batch(bs), fcache, device, rng=rng, punct_jitter_prob=0.35, need_bpe=True)
        opt3.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
            od = model(
                bk["ids"], bk["msk"], bk["st"], bk["eg"], bk["tg"], bk["loc"],
                bk["c1"], bk["c2"], bk["c3"], bk["cm"], bk["cf"], bk["sup"],
                outer_steps=4, r0_drop_p=0.45, charlm_drop_p=0.25,
            )
            loss = _heaven_loss(od, bk, outer_n=4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt3.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_q[k].mul_(0.992).add_(v.detach().cpu(), alpha=0.008)
        if (step + 1) % max(1, p3_steps // 3) == 0 or step == p3_steps - 1:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_q)
            evq = _eval_val(model, r0_only=False)
            print(f"  [Phase3-QAT step={step+1}/{p3_steps}] loss={float(loss.item()):.4f} {evq}", flush=True)
            if evq["sel_obj"] >= best_q_obj:
                best_q_obj, best_q_sd = evq["sel_obj"], {k: v.clone() for k, v in ema_q.items()}
            model.load_state_dict(cur_sd)
            model.train()
    model.load_state_dict(best_q_sd)
    set_v150_qat_mode(model, False)

    # Calibration + Decoupled AI/Promo R0 LTT + Export Bootable `<=5.00 MiB` + 17/17 CI!
    cal_rows = corpus["val_gold_v2"] + corpus["val_wave3"] + corpus["val_old_video_neg"] + corpus.get("val_multilingual", [])
    p_cal = predict_student_v150(model, fcache, [r["text"] for r in cal_rows], device, r0_only=False, outer_steps=4)
    y_cal_ai = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in cal_rows], dtype=np.float32)
    t_ai = fit_scalar_temperature_v140(p_cal["logit_ai"], y_cal_ai)

    pr_cal_rows = corpus["val_wave3"] + corpus["val_promo_v11"]
    p_pr_cal = predict_student_v150(model, fcache, [r["text"] for r in pr_cal_rows], device, r0_only=False, outer_steps=4)
    y_cal_pr = np.array([int(r.get("y_pr", 0.0) >= 0.5) for r in pr_cal_rows], dtype=np.float32)
    t_pr = fit_scalar_temperature_v140(p_pr_cal["logit_promo"], y_cal_pr) if len(pr_cal_rows) else 1.0

    lut_ai_r0 = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai_r0"], -30, 30), y_cal_ai)
    lut_ai_full = fit_smooth_ai_lut_v122(np.clip(p_cal["logit_ai"] / max(1e-3, t_ai), -30, 30), y_cal_ai)
    if len(pr_cal_rows) >= 8 and len(np.unique(y_cal_pr)) == 2:
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(np.clip(p_pr_cal["logit_promo_r0"], -30, 30), y_cal_pr, pi_nat=0.015)
        lut_pr_full = fit_output_shifted_promo_lut_v122(np.clip(p_pr_cal["logit_promo"] / max(1e-3, t_pr), -30, 30), y_cal_pr, pi_nat=0.015)
    else:
        dummy_z = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)
        dummy_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        lut_pr_r0 = fit_output_shifted_promo_lut_v122(dummy_z, dummy_y, pi_nat=0.015)
        lut_pr_full = fit_output_shifted_promo_lut_v122(dummy_z, dummy_y, pi_nat=0.015)

    lut01 = float(apply_spline_temp_lut(np.array([0.01], dtype=np.float32), lut_ai_r0)[0])
    p_pr_wild = apply_spline_temp_lut(expit(np.clip(p_cal["logit_promo_r0"], -30, 30)), lut_pr_r0)
    ai_lut_chk = {"lut_ai_r0_at_001": round(lut01, 5)}
    pr_lut_chk = {"median_promo_wild": round(float(np.median(p_pr_wild)), 5)}

    # Decoupled AI-only R0 LTT sweep (NEVER touches Promo!)
    p_r0_ai_cal = apply_spline_temp_lut(expit(np.clip(p_cal["logit_ai_r0"], -30, 30)), lut_ai_r0)
    pos_m, neg_m = y_cal_ai >= 0.5, y_cal_ai < 0.5
    r0_ai_pol: dict[str, Any] = {"enabled": False, "tau_ai": 0.03, "defer_max": 0.30, "max_chars": 180, "cal_exit_rate": 0.0}
    if pos_m.sum() >= 60:
        best_ex = -1.0
        for tau_a in (0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.010, 0.006):
            for dm in (0.40, 0.30, 0.22):
                em = (p_r0_ai_cal <= tau_a) & (p_cal["p_defer_ai_r0"] <= dm)
                if float(em[pos_m].mean()) <= 0.025 and float(em[neg_m].mean()) > best_ex:
                    best_ex = float(em[neg_m].mean())
                    r0_ai_pol = {"enabled": True, "tau_ai": tau_a, "defer_max": dm, "max_chars": 180, "cal_exit_rate": round(best_ex, 4)}
    r0_pr_pol = {"enabled": bool(r0_ai_pol["enabled"]), "tau_promo": 0.05, "defer_max": 0.35, "max_chars": 180}

    manifest = {
        "version": args.version, "max_len": int(args.max_len), "head_len": int(args.head_len),
        "long_char_trigger": int(args.long_char_trigger), "temp_ai": t_ai, "temp_promo": t_pr,
        "lut_ai_r0": lut_ai_r0, "lut_ai_full": lut_ai_full,
        "lut_promo_r0": lut_pr_r0, "lut_promo_full": lut_pr_full,
        "rung0_ai_ltt_policy": r0_ai_pol, "rung0_pr_ltt_policy": r0_pr_pol,
        "t3_escalation_policy": {"enabled": True, "tau_escalate": 0.45, "uncertain_lo": 0.10, "uncertain_hi": 0.90, "t3_bound_margin": 4.2},
        "t4_escalation_policy": {"enabled": True, "tau_escalate": 0.55, "uncertain_lo": 0.15, "uncertain_hi": 0.85, "t4_bound_margin": 2.4},
    }

    export_rep = export_v150_bootable_and_onnx(model, args.v07_tokenizer, manifest, out_dir, args.version, charlm=full_charlm)
    ci_gates = run_v150_ci_gates(
        model, tokenizer, export_rep=export_rep, ai_lut_check=ai_lut_chk, promo_lut_check=pr_lut_chk,
        charlm=full_charlm, release_dir=out_dir,
    )

    eng = MeguvizV150CascadeEngine(out_dir, threads=4)
    t4_p = eng.predict_batch(["这是一条用于验证 HeavenLoop 第四步双界翻转路由的边界测试评论。"], force_depth=4)[0]
    long_p = eng.predict_batch(["这是一篇用于验证端到端长文分段聚合路由的真实技术长评，核心逻辑清晰且不带模板腔。" * 12], force_full=False)[0]
    ci_gates["V150_T4_CONDITIONAL_ROUTE"] = {"forced_exit_rung_ai": t4_p["exit_rung_ai"], "pass": bool(t4_p["exit_rung_ai"] == 4)}
    ci_gates["V150_LONG_DOC_API"] = {"exit_rung_ai": long_p["exit_rung_ai"], "pass": bool(long_p["exit_rung_ai"] == 5)}
    ci_gates["all_ci_passed"] = all(bool(v.get("pass")) for v in ci_gates.values() if isinstance(v, dict))

    # Final ONNX Test Readouts (Gold-v2 Test, Multilingual Val, Read-Only God-500 Diag)
    def _eval_onnx_set(rows: list[dict[str, Any]], force_depth: int | None = None) -> dict[str, Any]:
        if not rows:
            return {}
        out_r = eng.predict_batch([r["text"] for r in rows], max_batch=32, force_full=True, force_depth=force_depth)
        y_a = np.array([int(r.get("y_ai", 0.0) >= 0.5) for r in rows], dtype=np.int32)
        lg_a = np.array([float(x["logit_ai"]) for x in out_r], dtype=np.float32)
        res_m = _eval_bin(y_a, lg_a)
        res_m["t3_or_t4_rate"] = round(float(np.mean([x["exit_rung_ai"] >= 3 for x in out_r])), 4)
        res_m["t4_rate"] = round(float(np.mean([x["exit_rung_ai"] == 4 for x in out_r])), 4)
        return res_m

    test_g2_cond = _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=None)
    test_g2_t4 = _eval_onnx_set(corpus.get("test_gold_v2", []), force_depth=4)
    val_ml_onnx = _eval_onnx_set(corpus.get("val_multilingual", []), force_depth=None)
    diag_g5_onnx = _eval_onnx_set(corpus.get("diag_god500", []), force_depth=None)

    report = {
        "version": args.version, "gpu_name": gpu_name, "total_params": tot_p, "heaven_core_params": core_p,
        "distill_pool_rows": len(distill_pool), "phase1_r0_best_obj": round(float(best_r0_obj), 5),
        "phase2_fp16_val": best_p2_ev, "phase3_qat_best_obj": round(float(best_q_obj), 5),
        "test_bili_gold_v2_onnx_cond": test_g2_cond,
        "test_bili_gold_v2_onnx_full_t4": test_g2_t4,
        "val_multilingual_onnx": val_ml_onnx,
        "diag_god500_onnx": diag_g5_onnx,
        "rung0_ai_ltt_policy": r0_ai_pol,
        "ablations": {
            "prelude_mode": args.prelude_mode, "core_depth": int(args.core_depth),
            "long_char_trigger": int(args.long_char_trigger),
            "coverage_beta": not args.no_coverage_beta,
            "t25_light_reread": not args.no_t25, "local_charlm": not args.no_local_charlm,
            "global_charlm_shortcut": bool(args.use_global_charlm),
            "shuffle_local_charlm": args.shuffle_local_charlm, "process_deltas": proc_attached,
        },
        "export": export_rep, "v150_ci_gates": ci_gates,
        "release_gates": {
            "bootable_le_5p00_mib": bool(export_rep["bootable_le_5p00_mib"]),
            "onnx_5stage_exact": bool(export_rep["onnx_exact_pass"]),
            "v150_ci_passed": bool(ci_gates["all_ci_passed"]),
        },
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    (out_dir / f"metrics-{args.version}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["release_gates"]["v150_ci_passed"] or not report["release_gates"]["bootable_le_5p00_mib"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
