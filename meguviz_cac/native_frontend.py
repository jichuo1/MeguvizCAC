"""High-Speed Rust (`meguviz_native`) Frontend Bridge for MeguvizCAC v1.6.x+ Turbo Trainer & Inference.

Provides 100% bit-exact, zero-GIL Rayon-parallel batch extraction for:
  - `canonicalize_zero_pii_v150` + `extract_style_features_56d`
  - HuggingFace Rust `encode_batch` + `build_bpe_engram_tensors_rs`
  - `RustOOFCharLMSet`: per-row OOF fold-routed global 9-d + token-aligned local 8-ch CharLM LLR
  - `extract_quote_scope_4d_batch_rs`: bounded-stack 4-ch Quote Scope FSM
  - `extract_apex_char_batch_rs`: R0++ char 1/2/3-gram hashes + 26-d struct+CharLM + support_pairs

Automatically falls back gracefully when `meguviz_native` is not yet compiled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import _meguviz_rs as _mn

    HAS_RUST_FRONTEND = hasattr(_mn, "RustOOFCharLMSet") and hasattr(_mn, "extract_apex_char_batch_rs")
    if not HAS_RUST_FRONTEND:
        _mn = None
except Exception:  # pragma: no cover
    _mn = None
    HAS_RUST_FRONTEND = False

from .v07_features import BiliPickyBPETokenizer
from .v130_xtreme_apex import R0_HEAD_CHARS_V130, R0_MAX_CHARS_V130
from .v140_paradise_apex import CHARLM_FEAT_DIM_V140
from .v150_heaven_loop import LOCAL_CHARLM_DIM_V150, CompactCharLMV150


_SEQ_BUCKETS = (32, 64, 96, 128, 160, 192, 224, 256, 288)


def snap_seq_len_bucket(ml: int, max_len: int = 288) -> int:
    """Snap padded sequence length to fixed even buckets for 100% `torch.compile` CUDA Graph reuse."""
    target = max(2, int(ml))
    if target % 2 == 1:
        target += 1
    for b in _SEQ_BUCKETS:
        if target <= b:
            return min(b, max_len if max_len % 2 == 0 else max_len + 1)
    return target


def _extract_charlm_py_arrays(lm: CompactCharLMV150):
    items = list(lm.vocabulary.items())
    vocab_chars = "".join(ch for ch, _ in items)
    vocab_ids = [int(i) for _, i in items]
    unigrams = np.ascontiguousarray(lm.unigrams, dtype=np.float32)
    t_keys = [np.ascontiguousarray(tb["keys"], dtype=np.uint64) for tb in lm.tables]
    t_counts = [np.ascontiguousarray(tb["counts"], dtype=np.uint32) for tb in lm.tables]
    t_ctxs = [np.ascontiguousarray(tb["contexts"], dtype=np.uint64) for tb in lm.tables]
    t_den = [np.ascontiguousarray(tb["den"], dtype=np.uint32) for tb in lm.tables]
    f_mean = np.ascontiguousarray(lm.feat_mean, dtype=np.float32)
    f_scale = np.ascontiguousarray(lm.feat_scale, dtype=np.float32)
    coef = np.ascontiguousarray(lm.coef, dtype=np.float32)
    return (
        vocab_chars, vocab_ids, unigrams,
        t_keys, t_counts, t_ctxs, t_den,
        f_mean, f_scale, coef, float(lm.bias), int(lm.max_chars),
    )


def build_rust_oof_charlm(
    full_lm: CompactCharLMV150 | None,
    fold_models: list[CompactCharLMV150] | None = None,
) -> Any | None:
    if (not HAS_RUST_FRONTEND) or full_lm is None:
        return None
    rs_set = _mn.RustOOFCharLMSet(*_extract_charlm_py_arrays(full_lm))
    if fold_models:
        for fm in fold_models:
            rs_set.add_fold(*_extract_charlm_py_arrays(fm))
    return rs_set


def encode_bpe_head_tail_batch_flat(
    tokenizer: BiliPickyBPETokenizer,
    clean_texts: list[str],
    max_len: int = 288,
    head_len: int = 192,
    bucket_len: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Batch-encode via Rust HuggingFace `tokenizers.encode_batch` and return flat `(ids, starts, ends, splits, ml)`."""
    n = len(clean_texts)
    if n == 0:
        z_i64 = np.zeros((0,), dtype=np.int64)
        z_i32 = np.zeros((0,), dtype=np.int32)
        return z_i64, z_i32, z_i32, np.zeros((1,), dtype=np.int32), 2

    encs = tokenizer._tok.encode_batch(clean_texts, add_special_tokens=False)
    bos_id = int(tokenizer.bos_id)
    eos_id = int(tokenizer.eos_id)
    inner = max(2, int(max_len) - 2)
    h_cap = min(int(head_len), inner - 8)
    t_cap = inner - h_cap

    splits = np.empty((n + 1,), dtype=np.int32)
    splits[0] = 0
    lens = [min(len(e.ids), inner) + 2 for e in encs]
    tot = int(sum(lens))
    flat_ids = np.empty((tot,), dtype=np.int64)
    flat_st = np.empty((tot,), dtype=np.int32)
    flat_en = np.empty((tot,), dtype=np.int32)

    ml_raw = 2
    pos = 0
    for i, e in enumerate(encs):
        ids_e = e.ids
        offs_e = e.offsets
        if len(ids_e) > inner:
            ids_e = ids_e[:h_cap] + ids_e[-t_cap:]
            offs_e = offs_e[:h_cap] + offs_e[-t_cap:]
        l_i = len(ids_e) + 2
        if l_i > ml_raw:
            ml_raw = l_i
        flat_ids[pos] = bos_id
        flat_st[pos] = -1
        flat_en[pos] = -1
        if ids_e:
            flat_ids[pos + 1 : pos + l_i - 1] = ids_e
            # offs_e is list[tuple[int, int]]
            for j, (a, b) in enumerate(offs_e):
                flat_st[pos + 1 + j] = a
                flat_en[pos + 1 + j] = b
        flat_ids[pos + l_i - 1] = eos_id
        flat_st[pos + l_i - 1] = -1
        flat_en[pos + l_i - 1] = -1
        pos += l_i
        splits[i + 1] = pos

    if bucket_len:
        ml = snap_seq_len_bucket(ml_raw, max_len=max_len)
    else:
        ml = ml_raw + (ml_raw & 1)
    return flat_ids, flat_st, flat_en, splits, ml


class RustTurboFrontend:
    """Zero-GIL Rayon-parallel batch frontend matching `_StrictGroupOOFFeatureCacheV160` bit-for-bit!"""

    def __init__(
        self,
        tokenizer: BiliPickyBPETokenizer,
        max_len: int,
        head_len: int,
        engram_buckets: int,
        trigram_buckets: int,
        r0_b2: int,
        r0_b3: int,
        use_local_charlm: bool = True,
        bucket_seq_len: bool = False,
    ) -> None:
        if not HAS_RUST_FRONTEND:
            raise RuntimeError("meguviz_native Rust extension is not installed.")
        self.tok = tokenizer
        self.max_len = int(max_len)
        self.head_len = int(head_len)
        self.engram_buckets = int(engram_buckets)
        self.trigram_buckets = int(trigram_buckets)
        self.r0_b2 = int(r0_b2)
        self.r0_b3 = int(r0_b3)
        self.use_local_charlm = bool(use_local_charlm)
        self.bucket_seq_len = bool(bucket_seq_len)
        self.text_to_fold: dict[str, int] = {}
        self.rust_charlm: Any | None = None

    def set_group_oof_charlm(
        self,
        fold_models: list[CompactCharLMV150],
        text_to_fold: dict[str, int],
        full_lm: CompactCharLMV150,
    ) -> None:
        self.text_to_fold = text_to_fold
        self.rust_charlm = build_rust_oof_charlm(full_lm, fold_models)

    def extract_batch_numpy(
        self,
        raw_texts: list[str],
        parent_raw_texts: list[str] | None = None,
        need_bpe: bool = True,
        use_oof_charlm: bool = True,
        need_quote_scope: bool = False,
        already_canonical: bool = False,
        need_r0: bool = True,
    ) -> dict[str, np.ndarray]:
        n = len(raw_texts)
        if use_oof_charlm and self.rust_charlm is not None and self.text_to_fold:
            parents = parent_raw_texts if parent_raw_texts is not None else raw_texts
            t2f = self.text_to_fold
            fold_ids = np.fromiter((t2f.get(p, -1) for p in parents), dtype=np.int32, count=n)
        else:
            fold_ids = None

        out: dict[str, np.ndarray] = {}
        # Turbo v3 fused single-pass R0-only path (eliminates extra PyO3 copies & duplicate UTF-8 scans!)
        if (not need_bpe) and hasattr(_mn, "extract_r0_only_batch_rs"):
            c1, c2, c3, cm, cf, sups = _mn.extract_r0_only_batch_rs(
                raw_texts, bool(already_canonical), 2048,
                self.rust_charlm, fold_ids,
                self.r0_b2, self.r0_b3, R0_MAX_CHARS_V130, R0_HEAD_CHARS_V130,
            )
            out["c1"] = c1
            out["c2"] = c2
            out["c3"] = c3
            out["cm"] = cm
            out["cf"] = cf
            out["sup"] = sups
            return out

        if already_canonical:
            cleans = raw_texts
            st56 = None
            if need_bpe:
                if hasattr(_mn, "style56_from_clean_batch_rs"):
                    st56 = _mn.style56_from_clean_batch_rs(raw_texts, 8192)
                else:
                    _, st56 = _mn.canonicalize_and_style56_batch_rs(raw_texts, 2048, 8192)
        elif need_bpe:
            cleans, st56 = _mn.canonicalize_and_style56_batch_rs(raw_texts, 2048, 8192)
        else:
            cleans = _mn.canonicalize_v150_batch_rs(raw_texts, 2048)
            st56 = None

        # Turbo v3 fused single-pass Post-BPE + CharLM(9d+8ch) + QuoteScope + R0 ApexChar!
        if need_bpe and hasattr(_mn, "extract_post_bpe_all_rs"):
            encs = self.tok._tok.encode_batch(cleans, add_special_tokens=False)
            enc_ids = [e.ids for e in encs]
            enc_offs = [e.offsets for e in encs]
            ids_a, msk_a, eg_a, tg_a, loc_a, qsc_a, c1, c2, c3, cm, cf, sups = _mn.extract_post_bpe_all_rs(
                cleans, enc_ids, enc_offs,
                int(self.tok.bos_id), int(self.tok.eos_id), int(self.tok.pad_id),
                self.max_len, self.head_len, self.bucket_seq_len,
                self.engram_buckets, self.trigram_buckets,
                self.rust_charlm, fold_ids,
                self.use_local_charlm, bool(need_quote_scope), bool(need_r0),
                self.r0_b2, self.r0_b3, R0_MAX_CHARS_V130, R0_HEAD_CHARS_V130,
            )
            out["ids"] = ids_a
            out["msk"] = msk_a
            out["eg"] = eg_a
            out["tg"] = tg_a
            out["loc"] = loc_a
            out["st"] = st56
            if need_quote_scope:
                out["qsc"] = qsc_a
            if need_r0:
                out["c1"] = c1
                out["c2"] = c2
                out["c3"] = c3
                out["cm"] = cm
                out["cf"] = cf
                out["sup"] = sups
            return out

        if fold_ids is None:
            fold_ids = np.full((n,), -1, dtype=np.int32)

        if need_bpe:
            flat_ids, flat_st, flat_en, splits, ml = encode_bpe_head_tail_batch_flat(
                self.tok, cleans, max_len=self.max_len, head_len=self.head_len, bucket_len=self.bucket_seq_len
            )
            ids_a, msk_a, eg_a, tg_a = _mn.build_bpe_engram_tensors_rs(
                flat_ids, splits, ml, int(self.tok.pad_id), self.engram_buckets, self.trigram_buckets
            )
            if self.rust_charlm is not None:
                g9, loc_a = self.rust_charlm.extract_9d_and_local_batch(
                    cleans, fold_ids, flat_st, flat_en, splits, ml, self.use_local_charlm
                )
                if not self.use_local_charlm:
                    loc_a = np.zeros((n, ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)
            else:
                g9 = np.zeros((n, CHARLM_FEAT_DIM_V140), dtype=np.float32)
                loc_a = np.zeros((n, ml, LOCAL_CHARLM_DIM_V150), dtype=np.float32)

            out["ids"] = ids_a
            out["msk"] = msk_a
            out["eg"] = eg_a
            out["tg"] = tg_a
            out["loc"] = loc_a
            out["st"] = st56
            if need_quote_scope:
                out["qsc"] = _mn.extract_quote_scope_4d_batch_rs(cleans, flat_st, flat_en, splits, ml)
        else:
            if self.rust_charlm is not None:
                g9 = self.rust_charlm.extract_9d_batch(cleans, fold_ids)
            else:
                g9 = np.zeros((n, CHARLM_FEAT_DIM_V140), dtype=np.float32)

        if need_r0:
            c1, c2, c3, cm, cf, sups = _mn.extract_apex_char_batch_rs(
                cleans, self.r0_b2, self.r0_b3, R0_MAX_CHARS_V130, R0_HEAD_CHARS_V130, g9
            )
            out["c1"] = c1
            out["c2"] = c2
            out["c3"] = c3
            out["cm"] = cm
            out["cf"] = cf
            out["sup"] = sups
        return out

    def canonicalize_with_spoof_batch(self, raw_texts: list[str], max_chars: int = 2048) -> tuple[list[str], np.ndarray]:
        if hasattr(_mn, "canonicalize_v150_with_spoof_batch_rs"):
            return _mn.canonicalize_v150_with_spoof_batch_rs(raw_texts, int(max_chars))
        cleans = _mn.canonicalize_v150_batch_rs(raw_texts, int(max_chars))
        return cleans, np.zeros((len(cleans),), dtype=np.float32)

    def kadane_focal_window(
        self, clean_text: str, win_chars: int = 160, lam: float = 0.92
    ) -> tuple[int, int, float, float] | None:
        if self.rust_charlm is not None and hasattr(self.rust_charlm, "kadane_focal_window"):
            return self.rust_charlm.kadane_focal_window(clean_text, int(win_chars), float(lam))
        return None
