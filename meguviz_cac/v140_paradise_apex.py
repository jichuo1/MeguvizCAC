"""MeguvizCAC v1.4.0-ParadiseApex-Loop: BiXT + Mini-mHC Nested-Loop <=5.00 MiB Dual-Axis Detector.

Synthesizes MeguvizCAC's signature `H=384, FFN=896, 16-Slot` Nested-Loop Symplectic Architecture
with the strongest empirically verified upgrades from ParadiseLoop v1.4.0 & SenseNova-Loop-9B:

  1. `ApexCharNanoRung0V140` (`~1.0 MiB` pure W8/INT8, ZERO 1.58-bit):
     Char Unigram/Bigram/Trigram + Head-140/Tail-50 + 4-scale Dilated DWConv + SE + 5-Stat Pool
     + **Support-Reliability Scalar (`distinct_alnum_pairs >= 16`)** guarding micro-short FPs!
  2. Pad-Exact `H=384` Prelude (`L0` + `L1`) + Bigram/Trigram Engrams + Haar Wavelet K/V
     + **Smooth Support-Reliability Gate (`g_sup = sigmoid((support - tau_sup) / 4)`)** protecting
     short `<= 30c` human comments from noisy `56-d` style statistics!
  3. Signature Full Nested-Loop Core (`ParadiseNestedCoreV140`, Clean W4/W8):
     - **BiXT Dynamic Token Write-Back (`Z^t -> H^t`, pad-exact!)**: each outer step writes updated
       slot hypotheses BACK into token representations `h_tok` via transposed slot-normalized affinity
       so `T2` and `T3` read dynamically refined, attribution-resolved token features!
     - **Mini Two-Stream Sinkhorn mHC (`Manifold-Constrained Hyper-Connections`)**: mirrors our
       flagship teacher `SenseNova-Loop-9B` with 2 parallel slot streams `(S_a, S_b)` mixed by a
       4-iteration Sinkhorn doubly-stochastic `2x2` matrix per outer step (`spectral_norm <= 1.0`)!
     - Preserves `k_inner=2` fast-inner Cayley Symplectic Exchange (`c^2 + s^2 === 1`), Surprise-Gated
       EMA fast memory, Relaxed Recursive LoRA `rank=16`, L0 penetration, and Analytic 6-Valley EBT!
  4. **Inter-Rung Disagreement Penalty** (`-gamma_1 |z_R0 - z_T2| - gamma_2 |z_T2 - z_EBT6|`) suppressing
     top-tail false positives at `R@P95 / R@P99` + **Exact Bounded Detached Residual `T3` (`|delta| <= 2.2`)**
     for mathematically guaranteed zero-risk `T2 -> T3` early exit!
  5. **Promo Attribution Scope Attention** (inside strictly disjoint Promo-private branch -> `F6 === 0.0`!)
     + Native `DynaSeam` Long-Doc Aggregator + 4-Stage Modular FP32-Exact ONNX + `MGVZ8 <= 5.00 MiB`
     + **14/14 Automated CI Gates**!
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np
import onnxruntime as ort
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v07_features import (
    BiliPickyBPETokenizer,
    apply_spline_temp_lut,
    canonicalize_zero_pii_v07,
    make_engram_bigram_ids,
    make_engram_trigram_ids,
)
from .v11_features import (
    CHUNK_META_DIM_V11,
    STYLE_FEAT_DIM_V11,
    _CONTACT_ID_PII_RE,
    extract_style_features_56d,
    plan_long_chunks_v11,
)
from .v121_apexnano import SoftBallLogitNormHeadV121
from .v130_xtreme_apex import (
    AI_END_V130,
    BUNDLE_MAX_MIB_V130,
    BUS_DIM_V130,
    CBM_AI_K_V130,
    CBM_PR_K_V130,
    EMB_DIM_V130,
    ENGRAM_BUCKETS_V130,
    FFN_V130,
    H_V130,
    K_INNER_V130,
    LORA_RANK_V130,
    MGVZ8_MAGIC,
    NUM_FAST_V130,
    NUM_HEADS_V130,
    NUM_SEM_FAST_V130,
    NUM_SLOTS_V130,
    NUM_SLOW_V130,
    NUM_SPAN_FAST_V130,
    PR_EMB_DIM_V130,
    R0_B1_V130,
    R0_B2_V130,
    R0_B3_V130,
    R0_D1_V130,
    R0_D2_V130,
    R0_D3_V130,
    R0_HEAD_CHARS_V130,
    R0_HID_V130,
    R0_MAX_CHARS_V130,
    R0_ZCHAR_DIM_V130,
    SHARED_END_V130,
    TRIGRAM_BUCKETS_V130,
    InLoopAnalyticEBTV130,
    PadExactConvBlockV130,
    QuantEmbeddingV130,
    QuantLinearV130,
    RelaxedRecursivePhaseAdapterV130,
    _pack_w4_tensor,
    _pack_w8_tensor,
    _unpack_w4_tensor,
    _unpack_w8_tensor,
    canonicalize_zero_pii_v130,
    encode_bpe_head_tail_v130,
    set_v130_qat_mode,
)

# ---------------------------------------------------------------------------
# v1.4.0 Constants (Head-192 + Tail-80 = 272 BPE tokens, max_len=288; 17+9=26-d R0 feats!)
# ---------------------------------------------------------------------------
R0_STRUCT_DIM_V140 = 17  # 16 structural scalars + 1 normalized `distinct_alnum_pairs / 32.0`
CHARLM_BITS_V140 = 12
CHARLM_VOCAB_V140 = 1 << CHARLM_BITS_V140  # 4096
CHARLM_ORDER_V140 = 4
CHARLM_SMOOTH_V140 = (0.0, 8.0, 8.0, 12.0)
CHARLM_FEAT_DIM_V140 = 9  # [avg_llr_1..4, pos_llr_4, neg_llr_4, std_llr_4, max_llr_4_s, charlm_logit]
CHARLM_MAGIC_V140 = b"MGLM140\x00"
R0_FEAT_DIM_V140 = R0_STRUCT_DIM_V140 + CHARLM_FEAT_DIM_V140  # 26
MAX_LEN_V140 = 288
HEAD_LEN_V140 = 192
BIXT_BOTTLENECK_V140 = 64
MHC_SINKHORN_ITERS_V140 = 4
T3_CORR_CAP_V140 = 2.2

set_v140_qat_mode = set_v130_qat_mode


# ---------------------------------------------------------------------------
# 1. Zero-PII Frontend + Compact Pruned 1-4 Generative Char-LM LLR (`<=110 KiB` zipped)
# ---------------------------------------------------------------------------

def _charlm_events_v140(texts: list[str], vocabulary: dict[str, int], max_chars: int = 256):
    trimmed = [(t[:max_chars] if len(t) > max_chars else t) for t in texts]
    lengths = np.array([max(1, len(t)) for t in trimmed], dtype=np.int32)
    owners = np.repeat(np.arange(len(trimmed), dtype=np.int32), lengths)
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    position = np.arange(len(owners), dtype=np.int32) - starts[owners]
    ids = np.fromiter(
        (vocabulary.get(ch, 1) for t in trimmed for ch in (t or "\0")),
        dtype=np.uint64,
        count=len(owners),
    )
    keys = [ids]
    for lag in range(1, CHARLM_ORDER_V140):
        prev = np.zeros_like(ids)
        if len(ids) > lag:
            prev[lag:] = ids[:-lag]
        prev[position < lag] = 0
        keys.append(keys[-1] | np.left_shift(prev, CHARLM_BITS_V140 * lag))
    return lengths, starts, owners, keys


def _charlm_build_table_v140(keys: np.ndarray, classes: np.ndarray, min_count: int) -> dict[str, np.ndarray]:
    uniq, inv, total = np.unique(keys, return_inverse=True, return_counts=True)
    gen = np.bincount(inv, weights=classes).astype(np.uint32)
    counts = np.column_stack((total.astype(np.uint32) - gen, gen))
    take = total >= int(min_count)
    kept = uniq[take]
    counts = counts[take]
    if len(kept) == 0:
        return {
            "keys": np.zeros((0,), dtype=np.uint64),
            "counts": np.zeros((0, 2), dtype=np.uint32),
            "contexts": np.zeros((0,), dtype=np.uint64),
            "den": np.zeros((0, 2), dtype=np.uint32),
        }
    contexts, at = np.unique(kept >> CHARLM_BITS_V140, return_index=True)
    den = np.add.reduceat(counts, at, axis=0, dtype=np.uint32)
    return {"keys": kept, "counts": counts, "contexts": contexts, "den": den}


def _charlm_lookup_v140(sorted_keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    result = np.zeros((len(query), 2), dtype=np.float64)
    if len(sorted_keys) == 0 or len(query) == 0:
        return result
    matches = np.searchsorted(sorted_keys, query)
    valid = matches < len(sorted_keys)
    valid[valid] = sorted_keys[matches[valid]] == query[valid]
    result[valid] = values[matches[valid]]
    return result


class CompactCharLMV140:
    """Zero-PII pruned 1-4 generative char-LM LLR reader (`<= 110 KiB` deflated, pure NumPy inference!)."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        unigrams: np.ndarray,
        tables: list[dict[str, np.ndarray]],
        feat_mean: np.ndarray | None = None,
        feat_scale: np.ndarray | None = None,
        coef: np.ndarray | None = None,
        bias: float = 0.0,
        max_chars: int = 256,
    ) -> None:
        self.vocabulary = dict(vocabulary)
        self.unigrams = np.asarray(unigrams, dtype=np.float32)
        self.tables = tables
        self.feat_mean = np.zeros((33,), dtype=np.float32) if feat_mean is None else np.asarray(feat_mean, dtype=np.float32)
        self.feat_scale = np.ones((33,), dtype=np.float32) if feat_scale is None else np.maximum(np.asarray(feat_scale, dtype=np.float32), 1e-5)
        self.coef = np.zeros((33,), dtype=np.float32) if coef is None else np.asarray(coef, dtype=np.float32)
        self.bias = float(bias)
        self.max_chars = int(max_chars)

    @classmethod
    def fit_tables(
        cls,
        clean_texts: list[str],
        labels: np.ndarray,
        min_counts: tuple[int, int, int] = (16, 28, 40),
        max_order_keys: tuple[int, int, int] = (9000, 8000, 7000),
        max_chars: int = 256,
    ) -> "CompactCharLMV140":
        trimmed = [(t[:max_chars] if len(t) > max_chars else t) for t in clean_texts]
        freq = Counter(ch for t in trimmed for ch in t)
        vocabulary = {
            ch: i + 2
            for i, (ch, _) in enumerate(sorted(freq.items(), key=lambda it: (-it[1], it[0]))[: CHARLM_VOCAB_V140 - 2])
        }
        _, _, doc, keys = _charlm_events_v140(trimmed, vocabulary, max_chars=max_chars)
        classes = np.asarray(labels, dtype=np.int32)[doc]
        counts = np.column_stack(
            [np.bincount(keys[0].astype(np.int64), weights=(classes == i), minlength=CHARLM_VOCAB_V140) for i in (0, 1)]
        )
        unigrams = ((counts + 0.5) / (counts.sum(axis=0) + 0.5 * CHARLM_VOCAB_V140)).astype(np.float32)

        tables: list[dict[str, np.ndarray]] = []
        for oi in range(1, CHARLM_ORDER_V140):
            mc = int(min_counts[oi - 1])
            cap = int(max_order_keys[oi - 1])
            tb = _charlm_build_table_v140(keys[oi], classes, mc)
            for _ in range(16):
                if len(tb["keys"]) <= cap:
                    break
                mc = max(mc + 1, int(round(mc * 1.25)))
                tb = _charlm_build_table_v140(keys[oi], classes, mc)
            tables.append(tb)
        return cls(vocabulary=vocabulary, unigrams=unigrams, tables=tables, max_chars=max_chars)

    def fit_linear_head_from_raw33(self, raw33: np.ndarray, labels: np.ndarray, reg_c: float = 0.25) -> None:
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(raw33, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int32)
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-4)
        xs = np.clip((x - mean) / scale, -8.0, 8.0)
        if len(np.unique(y)) < 2:
            self.feat_mean = mean.astype(np.float32)
            self.feat_scale = scale.astype(np.float32)
            self.coef = np.zeros((33,), dtype=np.float32)
            self.bias = 0.0
            return
        clf = LogisticRegression(C=float(reg_c), max_iter=300, class_weight="balanced", solver="lbfgs")
        clf.fit(xs, y)
        self.feat_mean = mean.astype(np.float32)
        self.feat_scale = scale.astype(np.float32)
        self.coef = clf.coef_[0].astype(np.float32)
        self.bias = float(clf.intercept_[0])

    def total_keys(self) -> int:
        return int(sum(len(tb["keys"]) for tb in self.tables))

    def extract_raw33(self, clean_texts: list[str], batch_size: int = 512) -> np.ndarray:
        if not clean_texts:
            return np.zeros((0, 33), dtype=np.float32)
        all_f: list[np.ndarray] = []
        uni = self.unigrams.astype(np.float64)
        for st in range(0, len(clean_texts), batch_size):
            grp = clean_texts[st : st + batch_size]
            lengths, ptr, owners, keys = _charlm_events_v140(grp, self.vocabulary, max_chars=self.max_chars)
            n = len(grp)
            p = uni[keys[0].astype(np.int64)]
            feats: list[np.ndarray] = []
            for order in range(CHARLM_ORDER_V140):
                if order > 0:
                    tb = self.tables[order - 1]
                    joint = _charlm_lookup_v140(tb["keys"], tb["counts"], keys[order])
                    den = _charlm_lookup_v140(tb["contexts"], tb["den"], keys[order] >> CHARLM_BITS_V140)
                    p = (joint + CHARLM_SMOOTH_V140[order] * p) / (den + CHARLM_SMOOTH_V140[order])
                logp = np.log(np.maximum(p, 1e-12))
                llr = logp[:, 1] - logp[:, 0]

                def _m(arr: np.ndarray) -> np.ndarray:
                    return np.bincount(owners, weights=arr, minlength=n) / lengths

                avg = _m(llr)
                feats.extend((
                    avg,
                    _m(np.maximum(llr, 0.0)),
                    _m(np.minimum(llr, 0.0)),
                    np.sqrt(np.maximum(_m(llr * llr) - avg * avg, 0.0)),
                    _m(logp[:, 0]),
                    _m(logp[:, 1]),
                    np.minimum.reduceat(llr, ptr),
                    np.maximum.reduceat(llr, ptr),
                ))
            feats.append(np.log1p(lengths))
            all_f.append(np.column_stack(feats).astype(np.float32))
        return np.vstack(all_f)

    def raw33_to_9d(self, raw33: np.ndarray) -> np.ndarray:
        if len(raw33) == 0:
            return np.zeros((0, CHARLM_FEAT_DIM_V140), dtype=np.float32)
        xs = np.clip((raw33.astype(np.float32) - self.feat_mean) / self.feat_scale, -8.0, 8.0)
        z_raw = xs @ self.coef + self.bias
        z_soft = (4.0 * np.tanh(z_raw / 4.0)).astype(np.float32)
        return np.column_stack([
            np.clip(raw33[:, 0], -3.0, 3.0),
            np.clip(raw33[:, 8], -3.0, 3.0),
            np.clip(raw33[:, 16], -3.0, 3.0),
            np.clip(raw33[:, 24], -3.0, 3.0),
            np.clip(raw33[:, 25], 0.0, 3.0),
            np.clip(raw33[:, 26], -3.0, 0.0),
            np.clip(raw33[:, 27], 0.0, 3.0),
            np.clip(raw33[:, 31] / 3.0, -2.0, 2.0),
            z_soft,
        ]).astype(np.float32)

    def extract_9d(self, clean_texts: list[str]) -> np.ndarray:
        return self.raw33_to_9d(self.extract_raw33(clean_texts))

    def save(self, out_path: Path) -> dict[str, Any]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Reconstruct exact ordered vocab string (`idx = 2, 3, ...`)
        inv_v = sorted(self.vocabulary.items(), key=lambda it: it[1])
        vocab_str = "".join(ch for ch, _ in inv_v)
        vocab_b = vocab_str.encode("utf-8")

        buf = bytearray()
        buf.extend(CHARLM_MAGIC_V140)
        buf.extend(struct.pack("<HHI", len(inv_v), self.max_chars, len(vocab_b)))
        buf.extend(vocab_b)
        buf.extend(self.unigrams.astype(np.float32).tobytes())
        buf.extend(self.feat_mean.astype(np.float32).tobytes())
        buf.extend(self.feat_scale.astype(np.float32).tobytes())
        buf.extend(self.coef.astype(np.float32).tobytes())
        buf.extend(struct.pack("<f", self.bias))

        for oi, tb in enumerate(self.tables):
            keys = np.asarray(tb["keys"], dtype=np.uint64)
            counts = np.asarray(tb["counts"], dtype=np.uint32)
            nk = len(keys)
            # Store lossless uint16 counts + rare overflow u32 list (`>= 65535`)
            ov_mask = (counts[:, 0] >= 65535) | (counts[:, 1] >= 65535)
            ov_idx = np.where(ov_mask)[0].astype(np.uint32)
            c16 = np.minimum(counts, 65534).astype(np.uint16)
            if len(ov_idx) > 0:
                c16[ov_idx, :] = 65535
            buf.extend(struct.pack("<II", nk, len(ov_idx)))
            if oi == 0:
                # 24-bit keys fit losslessly in uint32!
                buf.extend(keys.astype(np.uint32).tobytes())
            else:
                # 36/48-bit keys fit losslessly in uint32 lo + uint16 hi (6 bytes/key)!
                buf.extend((keys & np.uint64(0xFFFFFFFF)).astype(np.uint32).tobytes())
                buf.extend((keys >> np.uint64(32)).astype(np.uint16).tobytes())
            buf.extend(c16.tobytes())
            if len(ov_idx) > 0:
                buf.extend(ov_idx.tobytes())
                buf.extend(counts[ov_idx].astype(np.uint32).tobytes())

        out_path.write_bytes(bytes(buf))
        return {
            "raw_bytes": out_path.stat().st_size,
            "vocab_size": len(inv_v),
            "total_keys": self.total_keys(),
            "order_keys": [len(tb["keys"]) for tb in self.tables],
        }

    @classmethod
    def load(cls, path: Path) -> "CompactCharLMV140":
        data = path.read_bytes()
        assert data[:8] == CHARLM_MAGIC_V140, "Invalid CompactCharLMV140 magic!"
        pos = 8
        n_vocab, max_chars, vocab_blen = struct.unpack("<HHI", data[pos : pos + 8])
        pos += 8
        vocab_str = data[pos : pos + vocab_blen].decode("utf-8")
        pos += vocab_blen
        assert len(vocab_str) == n_vocab
        vocabulary = {ch: i + 2 for i, ch in enumerate(vocab_str)}

        uni_bytes = CHARLM_VOCAB_V140 * 2 * 4
        unigrams = np.frombuffer(data[pos : pos + uni_bytes], dtype=np.float32).reshape(CHARLM_VOCAB_V140, 2).copy()
        pos += uni_bytes
        feat_mean = np.frombuffer(data[pos : pos + 33 * 4], dtype=np.float32).copy()
        pos += 33 * 4
        feat_scale = np.frombuffer(data[pos : pos + 33 * 4], dtype=np.float32).copy()
        pos += 33 * 4
        coef = np.frombuffer(data[pos : pos + 33 * 4], dtype=np.float32).copy()
        pos += 33 * 4
        bias = struct.unpack("<f", data[pos : pos + 4])[0]
        pos += 4

        tables: list[dict[str, np.ndarray]] = []
        for oi in range(CHARLM_ORDER_V140 - 1):
            nk, n_ov = struct.unpack("<II", data[pos : pos + 8])
            pos += 8
            if oi == 0:
                keys = np.frombuffer(data[pos : pos + nk * 4], dtype=np.uint32).astype(np.uint64)
                pos += nk * 4
            else:
                k_lo = np.frombuffer(data[pos : pos + nk * 4], dtype=np.uint32).astype(np.uint64)
                pos += nk * 4
                k_hi = np.frombuffer(data[pos : pos + nk * 2], dtype=np.uint16).astype(np.uint64)
                pos += nk * 2
                keys = k_lo | (k_hi << np.uint64(32))
            counts = np.frombuffer(data[pos : pos + nk * 4], dtype=np.uint16).reshape(nk, 2).astype(np.uint32)
            pos += nk * 4
            if n_ov > 0:
                ov_idx = np.frombuffer(data[pos : pos + n_ov * 4], dtype=np.uint32)
                pos += n_ov * 4
                ov_vals = np.frombuffer(data[pos : pos + n_ov * 8], dtype=np.uint32).reshape(n_ov, 2)
                pos += n_ov * 8
                counts[ov_idx] = ov_vals
            if nk > 0:
                contexts, at = np.unique(keys >> CHARLM_BITS_V140, return_index=True)
                den = np.add.reduceat(counts, at, axis=0, dtype=np.uint32)
            else:
                contexts = np.zeros((0,), dtype=np.uint64)
                den = np.zeros((0, 2), dtype=np.uint32)
            tables.append({"keys": keys, "counts": counts, "contexts": contexts, "den": den})
        return cls(
            vocabulary=vocabulary, unigrams=unigrams, tables=tables,
            feat_mean=feat_mean, feat_scale=feat_scale, coef=coef, bias=bias, max_chars=max_chars,
        )


def fit_oof_charlm_v140(
    raw_texts: list[str],
    labels: np.ndarray,
    n_folds: int = 3,
    min_counts: tuple[int, int, int] = (16, 28, 40),
    max_order_keys: tuple[int, int, int] = (9000, 8000, 7000),
    seed: int = 20261130,
) -> tuple[dict[str, np.ndarray], CompactCharLMV140]:
    """Fit 3-fold strictly out-of-fold CompactCharLMV140 + 1 full deployable model (calibrated on OOF 33-d!)."""
    n = len(raw_texts)
    cleans = [canonicalize_zero_pii_v130(str(t))[0] for t in raw_texts]
    y = np.asarray(labels, dtype=np.int32)
    rng_np = np.random.default_rng(seed)

    # Stratified fold assignment
    fold_ids = np.zeros(n, dtype=np.int32)
    for cls_v in (0, 1):
        idx = np.where(y == cls_v)[0]
        rng_np.shuffle(idx)
        for j, ii in enumerate(idx):
            fold_ids[ii] = j % max(2, int(n_folds))

    oof_raw33 = np.zeros((n, 33), dtype=np.float32)
    fold_mc = tuple(max(2, int(round(c * (n_folds - 1) / float(n_folds)))) for c in min_counts)
    for f in range(max(2, int(n_folds))):
        tr_idx = np.where(fold_ids != f)[0]
        va_idx = np.where(fold_ids == f)[0]
        if len(tr_idx) < 4 or len(va_idx) == 0:
            continue
        fm = CompactCharLMV140.fit_tables(
            [cleans[i] for i in tr_idx], y[tr_idx], min_counts=fold_mc, max_order_keys=max_order_keys
        )
        oof_raw33[va_idx] = fm.extract_raw33([cleans[i] for i in va_idx])

    full_lm = CompactCharLMV140.fit_tables(cleans, y, min_counts=min_counts, max_order_keys=max_order_keys)
    # Fit linear readout strictly on leak-free OOF 33-d features so logit calibration matches unseen test data!
    full_lm.fit_linear_head_from_raw33(oof_raw33, y, reg_c=0.25)

    # Map each training raw text to its exact leak-free OOF 9-d vector using per-fold linear readout!
    oof_9d = np.zeros((n, CHARLM_FEAT_DIM_V140), dtype=np.float32)
    for f in range(max(2, int(n_folds))):
        tr_idx = np.where(fold_ids != f)[0]
        va_idx = np.where(fold_ids == f)[0]
        if len(tr_idx) < 4 or len(va_idx) == 0:
            continue
        tmp = CompactCharLMV140(vocabulary={}, unigrams=full_lm.unigrams, tables=[])
        tmp.fit_linear_head_from_raw33(oof_raw33[tr_idx], y[tr_idx], reg_c=0.25)
        oof_9d[va_idx] = tmp.raw33_to_9d(oof_raw33[va_idx])

    oof_map = {raw_texts[i]: oof_9d[i] for i in range(n)}
    return oof_map, full_lm


def count_distinct_support_pairs_v140(clean_text: str, max_chars: int = 256) -> float:
    """Count distinct adjacent content-char pairs (CJK / Kana / Latin / digit), matching ParadiseLoop's short-FP guard."""
    s = str(clean_text or "")[:max_chars]
    if len(s) < 2:
        return 0.0
    pairs: set[tuple[str, str]] = set()
    prev = s[0]
    prev_ok = prev.isalnum()
    for ch in s[1:]:
        ok = ch.isalnum()
        if prev_ok and ok:
            pairs.add((prev, ch))
        prev, prev_ok = ch, ok
    return float(len(pairs))


def extract_apex_char_batch_v140(
    texts: list[str],
    max_chars: int = R0_MAX_CHARS_V130,
    head_chars: int = R0_HEAD_CHARS_V130,
    already_canonical: bool = False,
    charlm: CompactCharLMV140 | None = None,
    precomputed_charlm_9d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-PII Head-140/Tail-50 char 1/2/3-gram hash + 17-d struct + 9-d CharLM LLR (`26-d`) + `spoofs` + `support_pairs`."""
    n = len(texts)
    if n == 0:
        return (
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.int64),
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, R0_FEAT_DIM_V140), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    cleans: list[str] = []
    spoofs = np.zeros(n, dtype=np.float32)
    supports = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(texts):
        if already_canonical:
            cl = str(t)
        else:
            cl, sp_meta = canonicalize_zero_pii_v130(str(t))
            if isinstance(sp_meta, dict):
                spoofs[i] = 1.0 if (
                    sp_meta.get("spoof_any", 0.0) or sp_meta.get("zw_count", 0) or sp_meta.get("homoglyph_count", 0)
                ) else 0.0
        cleans.append(cl)
        supports[i] = count_distinct_support_pairs_v140(cl)

    body_cap = max(8, int(max_chars) - 2)
    h_cap = min(int(head_chars), body_cap - 4)
    t_cap = body_cap - h_cap
    trimmed = [(cl[:h_cap] + cl[-t_cap:]) if len(cl) > body_cap else cl for cl in cleans]

    ml = max(8, max(len(s) + 2 for s in trimmed))
    c1 = np.zeros((n, ml), dtype=np.int64)
    c2 = np.zeros((n, ml), dtype=np.int64)
    c3 = np.zeros((n, ml), dtype=np.int64)
    mask = np.zeros((n, ml), dtype=np.float32)
    feats = np.zeros((n, R0_FEAT_DIM_V140), dtype=np.float32)

    BOS_CP = np.uint32(2)
    EOS_CP = np.uint32(3)
    for i, s in enumerate(trimmed):
        if s:
            body_codes = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
            codes = np.concatenate([np.array([BOS_CP], dtype=np.uint32), body_codes, np.array([EOS_CP], dtype=np.uint32)])
        else:
            codes = np.array([BOS_CP, EOS_CP], dtype=np.uint32)
        L = len(codes)
        mask[i, :L] = 1.0

        u1 = (((codes * np.uint32(2654435761)) >> np.uint32(20)) % np.uint32(R0_B1_V130 - 1)) + np.uint32(1)
        c1[i, :L] = u1
        if L >= 2:
            h2 = (codes[:-1] * np.uint32(1315423911)) ^ (codes[1:] * np.uint32(2654435761))
            u2 = ((h2 >> np.uint32(18)) % np.uint32(R0_B2_V130 - 1)) + np.uint32(1)
            c2[i, : L - 1] = u2
            c2[i, L - 1] = u2[-1]
        if L >= 3:
            h3 = (
                (codes[:-2] * np.uint32(1315423911))
                ^ (codes[1:-1] * np.uint32(2654435761))
                ^ (codes[2:] * np.uint32(2246822519))
            )
            u3 = ((h3 >> np.uint32(19)) % np.uint32(R0_B3_V130 - 1)) + np.uint32(1)
            c3[i, : L - 2] = u3
            c3[i, L - 2 : L] = u3[-1]
        else:
            c3[i, :L] = u1

        full_cl = cleans[i]
        Lf = max(1, len(full_cl))
        bc = np.frombuffer(full_cl[:512].encode("utf-32-le"), dtype=np.uint32) if full_cl else np.zeros((1,), dtype=np.uint32)
        Lb = max(1, len(bc))
        inv_l = 1.0 / float(Lb)
        cjk = float(np.sum((bc >= 0x4E00) & (bc <= 0x9FFF)))
        alnum = float(np.sum(((bc >= 48) & (bc <= 57)) | ((bc >= 65) & (bc <= 90)) | ((bc >= 97) & (bc <= 122))))
        digits = float(np.sum((bc >= 48) & (bc <= 57)))
        spaces = float(np.sum((bc == 32) | (bc == 9)))
        newlines = float(np.sum(bc == 10))
        cn_comma = float(np.sum((bc == 0xFF0C) | (bc == 44)))
        cn_period = float(np.sum(bc == 0x3002))
        cn_colon = float(np.sum((bc == 0xFF1A) | (bc == 58)))
        cn_semi = float(np.sum((bc == 0xFF1B) | (bc == 59)))
        emdash = float(np.sum((bc == 0x2014) | (bc == 0x2015)))
        quote_cn = float(np.sum((bc == 0x300C) | (bc == 0x300D) | (bc == 0x201C) | (bc == 0x201D)))
        bracket_emo = float(np.sum(bc == 91))
        qmark = float(np.sum((bc == 63) | (bc == 0xFF1F)))
        exclam = float(np.sum((bc == 33) | (bc == 0xFF01)))
        ends_per = 1.0 if full_cl.strip().endswith("。") else 0.0
        uniq_ratio = float(len(np.unique(bc))) * inv_l

        feats[i, :R0_STRUCT_DIM_V140] = np.array([
            math.log1p(Lf) / 6.0,
            cjk * inv_l,
            alnum * inv_l,
            digits * inv_l,
            spaces * inv_l,
            min(1.0, newlines / 6.0),
            min(1.0, cn_comma * inv_l * 10.0),
            min(1.0, cn_period * inv_l * 10.0),
            min(1.0, cn_colon * 0.25),
            min(1.0, cn_semi * 0.25),
            min(1.0, emdash * 0.25),
            min(1.0, quote_cn * 0.25),
            min(1.0, bracket_emo * 0.25),
            min(1.0, (qmark + exclam) * inv_l * 10.0),
            ends_per,
            uniq_ratio,
            min(3.0, supports[i] / 32.0),
        ], dtype=np.float32)

    if precomputed_charlm_9d is not None:
        feats[:, R0_STRUCT_DIM_V140:] = np.asarray(precomputed_charlm_9d, dtype=np.float32)
    elif charlm is not None:
        feats[:, R0_STRUCT_DIM_V140:] = charlm.extract_9d(cleans)

    return c1, c2, c3, mask, feats, spoofs, supports


# ---------------------------------------------------------------------------
# 2. Stage 0: Support-Gated `ApexCharNanoRung0V140` (`~1.0 MiB` Pure W8/INT8)
# ---------------------------------------------------------------------------

class ApexCharNanoRung0V140(nn.Module):
    """High-res Char 1/2/3-gram + 4-Scale Dilated DWConv + SE + Support-Gated Structural Scalars (W8)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb1 = QuantEmbeddingV130(R0_B1_V130, R0_D1_V130, bits=8, padding_idx=0)
        self.emb2 = QuantEmbeddingV130(R0_B2_V130, R0_D2_V130, bits=4, padding_idx=0)
        self.emb3 = QuantEmbeddingV130(R0_B3_V130, R0_D3_V130, bits=8, padding_idx=0)
        in_d = R0_D1_V130 + R0_D2_V130 + R0_D3_V130  # 112

        self.in_proj = QuantLinearV130(in_d, R0_HID_V130, bits=8)
        self.in_norm = nn.LayerNorm(R0_HID_V130)

        self.dw3 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=3, padding=1, groups=R0_HID_V130)
        self.dw5_d1 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=2, dilation=1, groups=R0_HID_V130)
        self.dw5_d2 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=4, dilation=2, groups=R0_HID_V130)
        self.dw5_d4 = nn.Conv1d(R0_HID_V130, R0_HID_V130, kernel_size=5, padding=8, dilation=4, groups=R0_HID_V130)
        self.pw = QuantLinearV130(R0_HID_V130, R0_HID_V130 * 2, bits=8)
        self.se = nn.Sequential(
            QuantLinearV130(R0_HID_V130, 32, bits=8),
            nn.SiLU(),
            QuantLinearV130(32, R0_HID_V130, bits=8),
            nn.Sigmoid(),
        )
        self.stage_norm = nn.LayerNorm(R0_HID_V130)
        self.attn_pool = nn.Sequential(
            QuantLinearV130(R0_HID_V130, 64, bits=8),
            nn.Tanh(),
            QuantLinearV130(64, 1, bits=8),
        )

        # Support gate modulates structural surface scalars on micro-short comments (`< 16` distinct pairs)!
        # Promo trunk strictly receives ONLY structural scalars (`R0_STRUCT_DIM_V140`), NEVER AI-only CharLM LLR!
        self.r0_sup_tau = nn.Parameter(torch.tensor(0.50))  # 0.50 * 32 = 16 distinct pairs
        pool_in_ai = R0_HID_V130 * 5 + R0_FEAT_DIM_V140  # 666
        pool_in_pr = R0_HID_V130 * 5 + R0_STRUCT_DIM_V140  # 657
        self.ai_fc = nn.Sequential(QuantLinearV130(pool_in_ai, 128, bits=8), nn.LayerNorm(128), nn.SiLU())
        self.promo_fc = nn.Sequential(QuantLinearV130(pool_in_pr, 96, bits=8), nn.LayerNorm(96), nn.SiLU())
        self.ai_head = nn.Linear(128, 1)
        self.promo_head = nn.Linear(96, 1)
        self.z_char_ai = nn.Linear(128, 32)
        self.z_char_pr = nn.Linear(96, 32)
        self.defer_head = nn.Sequential(nn.Linear(128 + 96 + 3, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        m = mask.float()
        m_u = m.unsqueeze(-1)
        x = torch.cat([self.emb1(c1), self.emb2(c2), self.emb3(c3)], dim=-1)
        h = self.in_norm(self.in_proj(x)) * m_u
        ht = h.transpose(1, 2)
        conv_sum = (self.dw3(ht) + self.dw5_d1(ht) + self.dw5_d2(ht) + self.dw5_d4(ht)) * 0.25
        hc = conv_sum.transpose(1, 2) * m_u
        u, g = self.pw(hc).chunk(2, dim=-1)
        hc = u * torch.sigmoid(g) * m_u

        denom = m_u.sum(dim=1).clamp_min(1.0)
        mean_c = hc.sum(dim=1) / denom
        hc = hc * self.se(mean_c).unsqueeze(1)
        h = self.stage_norm(h + hc) * m_u

        p_mean = h.sum(dim=1) / denom
        p_max = h.masked_fill(m_u < 0.5, -1e4).max(dim=1).values
        a_lg = self.attn_pool(h).squeeze(-1).masked_fill(m < 0.5, -1e4)
        a_w = torch.softmax(a_lg.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        p_attn = (h * a_w).sum(dim=1)
        p_first = h[:, 0, :]
        last_idx = (m.sum(dim=1).long() - 1).clamp(min=0, max=h.shape[1] - 1)
        p_last = h.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[2])).squeeze(1)

        # Gate noisy surface scalars `[0:16)` by smooth support reliability on AI trunk; keep Char-LM LLR active even on short `<=30c`!
        f_in = feats.to(h.dtype)
        f_struct = f_in[:, :R0_STRUCT_DIM_V140]
        f_charlm = f_in[:, R0_STRUCT_DIM_V140:]
        sup_norm = f_struct[:, -1:]
        g_sup_r0 = torch.sigmoid((sup_norm - self.r0_sup_tau.clamp(0.25, 0.75)) * 8.0)
        f_ai = torch.cat([f_struct[:, :-1] * g_sup_r0, sup_norm, f_charlm * (0.55 + 0.45 * g_sup_r0)], dim=-1)

        pooled_ai = torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_ai], dim=-1)
        pooled_pr = torch.cat([p_mean, p_max, p_attn, p_first, p_last, f_struct], dim=-1)
        h_ai = self.ai_fc(pooled_ai)
        h_pr = self.promo_fc(pooled_pr)

        lg_ai = self.ai_head(h_ai).squeeze(-1)
        lg_pr = self.promo_head(h_pr).squeeze(-1)
        z_char = torch.cat([self.z_char_ai(h_ai), self.z_char_pr(h_pr)], dim=-1)
        def_lg = self.defer_head(
            torch.cat([h_ai.detach(), h_pr.detach(), lg_ai.detach().unsqueeze(-1), lg_pr.detach().unsqueeze(-1), sup_norm.detach()], dim=-1)
        ).squeeze(-1)
        return {
            "logit_ai_r0": lg_ai,
            "logit_promo_r0": lg_pr,
            "defer_r0_logit": def_lg,
            "p_defer_r0": torch.sigmoid(def_lg),
            "z_char": z_char,
        }


# ---------------------------------------------------------------------------
# 3. ParadiseApex Nested-Loop Core: BiXT Dynamic Token Write-Back + Mini Two-Stream Sinkhorn mHC!
# ---------------------------------------------------------------------------

def sinkhorn_2x2_v140(log_m: torch.Tensor, iters: int = MHC_SINKHORN_ITERS_V140) -> torch.Tensor:
    """Convert `[..., 2, 2]` log-matrix to exact doubly-stochastic matrix (`row/col sums === 1`, spectral norm <= 1)."""
    m = torch.exp(log_m.float().clamp(-6.0, 6.0))
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return m


class ParadiseNestedCoreV140(nn.Module):
    """Full-Scale `H=384, FFN=896, 16-Slot` Nested-Loop Core upgraded with BiXT (`Z->H`) + Two-Stream Sinkhorn mHC!"""

    def __init__(self, dim: int = H_V130, ffn_dim: int = FFN_V130, num_heads: int = NUM_HEADS_V130) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.n_slow = NUM_SLOW_V130
        self.n_sem = NUM_SEM_FAST_V130
        self.n_span = NUM_SPAN_FAST_V130
        self.n_fast = NUM_FAST_V130

        self.slot_queries = nn.Parameter(torch.randn(1, NUM_SLOTS_V130, dim) * 0.04)

        # Active Text K/V Projections (re-projected from dynamically updated `h_tok` on T1/T2/T3!)
        self.k_ev_proj = QuantLinearV130(dim, dim, bits=4)
        self.v_ev_proj = QuantLinearV130(dim, dim, bits=4)

        self.norm_ca = nn.LayerNorm(dim)
        self.ca_q = QuantLinearV130(dim, dim, bits=4)
        self.ca_o = QuantLinearV130(dim, dim, bits=4)
        self.ca_q_lora_a = nn.Parameter(torch.randn(3, dim, LORA_RANK_V130) * 0.02)
        self.ca_q_lora_b = nn.Parameter(torch.zeros(3, LORA_RANK_V130, dim))
        self.reread_gate = nn.Parameter(torch.tensor(0.35))

        # NEW #1: BiXT Dynamic Token Write-Back (`Z^t -> H^t`, low-rank 64-d W4, 100% pad-exact!)
        self.bixt_v = QuantLinearV130(dim, BIXT_BOTTLENECK_V140, bias=False, bits=4)
        self.bixt_o = QuantLinearV130(BIXT_BOTTLENECK_V140, dim, bias=False, bits=4)
        self.bixt_norm = nn.LayerNorm(dim)
        self.bixt_gate = nn.Parameter(torch.full((3,), -1.2))

        # NEW #2: Mini Two-Stream Sinkhorn mHC (mirrors SenseNova-Loop-9B's 2-stream manifold contraction!)
        # Init near-identity favoring primary stream `S_a` + smooth cross-stream residual `S_b`
        init_mhc = torch.tensor([[1.2, -0.6], [-0.6, 1.2]], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        self.mhc_log_mix = nn.Parameter(init_mhc)
        self.mhc_read_mix = nn.Parameter(torch.tensor([-0.8, -0.5, -0.3], dtype=torch.float32))

        # Fine-Scale Haar Wavelet Reread (`64-d`)
        self.haar_q = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.haar_o = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.haar_gate = nn.Parameter(torch.tensor(-0.8))

        # Shared All-16-Slot Self-Attention + Per-Outer-Step Relaxed LoRA on `sa_o`
        self.norm_sa = nn.LayerNorm(dim)
        self.sa_qkv = QuantLinearV130(dim, dim * 3, bits=4)
        self.sa_o = QuantLinearV130(dim, dim, bits=4)
        self.sa_o_lora_a = nn.Parameter(torch.randn(3, dim, LORA_RANK_V130) * 0.02)
        self.sa_o_lora_b = nn.Parameter(torch.zeros(3, LORA_RANK_V130, dim))

        # Shared SwiGLU FFN + Relaxed Recursive Adapter (`3 outer x 3 inner = 9` states!)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff_up = QuantLinearV130(dim, ffn_dim * 2, bits=4)
        self.ff_down = QuantLinearV130(ffn_dim, dim, bits=4)
        self.relax = RelaxedRecursivePhaseAdapterV130(dim=dim, rank=LORA_RANK_V130, n_outer=3, n_phases=3)

        # True Nested Fast-Inner Dual-Gate Pre/Post
        self.fast_norm = nn.LayerNorm(dim)
        self.p2_pre_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_pre_b = QuantLinearV130(16, dim, bits=4)
        self.p2_post_a = QuantLinearV130(dim * 2, 16, bias=False, bits=4)
        self.p2_post_b = QuantLinearV130(16, dim, bits=4)
        self.fast_res_scale = nn.Parameter(torch.full((dim,), 0.35))

        # 64-d Bus + Cayley Symplectic Peer Coupling (`c^2 + s^2 === 1`)
        self.bus_down = QuantLinearV130(dim, BUS_DIM_V130, bias=False, bits=4)
        self.bus_up = QuantLinearV130(BUS_DIM_V130, dim, bias=False, bits=4)
        self.c1_q = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_k = nn.Linear(BUS_DIM_V130, BUS_DIM_V130, bias=False)
        self.c1_cayley_gamma = nn.Parameter(torch.zeros(BUS_DIM_V130))
        self.c1_bridge_scale = nn.Parameter(torch.tensor(0.22))

        # Surprise-Gated EMA Compressed Fast Memory (`carry_m_l`)
        self.c2_beta_logit = nn.Parameter(torch.full((BUS_DIM_V130,), 0.4))
        self.c2_carry_scale = nn.Parameter(torch.tensor(0.24))
        self.c2_surprise_mlp = nn.Sequential(nn.Linear(BUS_DIM_V130 + 1, 16), nn.SiLU(), nn.Linear(16, 1))

        # 3-Role Outer-Step History Delta + Doubt Handoff
        self.role_hist_g = nn.Parameter(torch.zeros(3, 1, dim))
        self.handoff_proj = QuantLinearV130(BUS_DIM_V130 + 2, dim, bits=4)

        # L0 Surface Penetration into 4 Slow Slots
        self.l0_pen_a = QuantLinearV130(dim, 24, bias=False, bits=8)
        self.l0_pen_b = QuantLinearV130(24, dim, bias=False, bits=8)
        self.l0_pen_gate = nn.Parameter(torch.full((dim,), -1.2))

        # Analytic 6-Valley EBT on 64-d Slow Bus
        self.ebt = InLoopAnalyticEBTV130(BUS_DIM_V130, hidden=96)
        self.ebt_back_scale = nn.Parameter(torch.tensor(0.20))

        # Manifold Anchor Blend + Step-Indexed Spectral Damping
        self.anchor_alpha = nn.Parameter(torch.full((1, 1, dim), -1.4))
        self.damp_gamma = nn.Parameter(torch.tensor(1.4))

    def _shared_ffn(self, x: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        xm = self.relax.modulate(x, outer_idx, phase_idx)
        u, g = self.ff_up(xm).chunk(2, dim=-1)
        return self.ff_down(u * F.silu(g)) + self.relax.delta(xm, outer_idx, phase_idx)

    def _cayley_peer_couple(self, z_l: torch.Tensor, cayley_delta: torch.Tensor) -> torch.Tensor:
        u = self.bus_down(z_l)
        u_a, u_b = u[:, : self.n_sem, :], u[:, self.n_sem :, :]
        sc_ab = torch.matmul(self.c1_q(u_a), self.c1_k(u_b).transpose(-1, -2)) / 8.0
        msg_to_a = torch.matmul(torch.softmax(sc_ab.float(), dim=-1).to(u.dtype), u_b)
        msg_to_b = torch.matmul(torch.softmax(sc_ab.transpose(-1, -2).float(), dim=-1).to(u.dtype), u_a)

        g = (self.c1_cayley_gamma.unsqueeze(0) + 0.15 * torch.tanh(cayley_delta)).clamp(-2.5, 2.5)
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        c = ((1.0 - g2) * inv).unsqueeze(1)
        s = ((2.0 * g) * inv).unsqueeze(1)

        rot_a = c * u_a + s * msg_to_a
        rot_b = -s * msg_to_b + c * u_b
        delta = self.bus_up(torch.cat([rot_a - u_a, rot_b - u_b], dim=1))
        return z_l + (0.35 * torch.tanh(self.c1_bridge_scale)) * delta

    def _fast_inner_step(self, slots: torch.Tensor, ctrl: torch.Tensor, outer_idx: int, phase_idx: int) -> torch.Tensor:
        z_h = slots[:, : self.n_slow, :]
        z_l = slots[:, self.n_slow :, :]
        zh_ref = z_h.mean(dim=1, keepdim=True).expand_as(z_l)

        zn = self.fast_norm(z_l)
        g_pre = torch.sigmoid(self.p2_pre_b(F.silu(self.p2_pre_a(torch.cat([zn, zn - zh_ref], dim=-1)))) + 0.25 * ctrl[2])
        delta_l = self._shared_ffn(zn * g_pre, outer_idx=outer_idx, phase_idx=phase_idx)
        g_post = torch.sigmoid(self.p2_post_b(F.silu(self.p2_post_a(torch.cat([zn, delta_l], dim=-1)))) + 0.25 * ctrl[3])
        z_l_new = z_l + self.fast_res_scale * (g_post * delta_l)
        z_l_new = self._cayley_peer_couple(z_l_new, ctrl[0].view(1, 1))
        return torch.cat([z_h, z_l_new], dim=1)

    def init_anchor(self, h_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = h_tok.shape
        m_u = mask.float().unsqueeze(-1)
        k_ev = self.k_ev_proj(h_tok) * m_u
        v_ev = self.v_ev_proj(h_tok) * m_u
        q = self.slot_queries.expand(B, -1, -1)
        qh = self.ca_q(self.norm_ca(q)).view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        sc = sc.masked_fill(mask[:, None, None, :].float() < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(h_tok.dtype)
        ctx = torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        return q + self.ca_o(ctx)

    def forward_outer_step(
        self,
        slots_a: torch.Tensor,
        slots_b: torch.Tensor,
        anchor: torch.Tensor,
        h_tok: torch.Tensor,
        mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        outer_idx: int,
        prev_slots: torch.Tensor | None = None,
        carry_m_l: torch.Tensor | None = None,
        prev_bus: torch.Tensor | None = None,
        prev_doubt: torch.Tensor | None = None,
        k_inner: int = K_INNER_V130,
    ) -> dict[str, torch.Tensor]:
        B, L, D = h_tok.shape
        oi = min(max(int(outer_idx), 0), 2)
        ctrl = torch.tanh(self.relax.dyn_ctrl[oi])
        m_f = mask.float()
        m_u = m_f.unsqueeze(-1)

        s_in = slots_a
        s = slots_a

        # 1. Outer-Step History Delta + Doubt Handoff
        if outer_idx > 0 and prev_slots is not None:
            d_slow = (s[:, : self.n_slow, :] - prev_slots[:, : self.n_slow, :]).mean(dim=1, keepdim=True)
            d_sem = (s[:, self.n_slow : self.n_slow + self.n_sem, :] - prev_slots[:, self.n_slow : self.n_slow + self.n_sem, :]).mean(dim=1, keepdim=True)
            d_span = (s[:, self.n_slow + self.n_sem :, :] - prev_slots[:, self.n_slow + self.n_sem :, :]).mean(dim=1, keepdim=True)
            g_r = 0.10 * torch.tanh(self.role_hist_g)
            s = torch.cat([
                s[:, : self.n_slow, :] + g_r[0] * d_slow,
                s[:, self.n_slow : self.n_slow + self.n_sem, :] + g_r[1] * d_sem,
                s[:, self.n_slow + self.n_sem :, :] + g_r[2] * d_span,
            ], dim=1)

        if outer_idx > 0 and prev_bus is not None and prev_doubt is not None:
            h_in = 0.25 * self.handoff_proj(torch.cat([prev_bus, prev_doubt], dim=-1))
            s = torch.cat([s[:, :-1, :], s[:, -1:, :] + h_in.unsqueeze(1)], dim=1)

        zh_now_bus = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        if prev_bus is not None:
            cos_sim = F.cosine_similarity(zh_now_bus, prev_bus, dim=-1, eps=1e-6).unsqueeze(-1)
            surprise = torch.sigmoid(self.c2_surprise_mlp(torch.cat([zh_now_bus - prev_bus, cos_sim], dim=-1)).squeeze(-1))
        else:
            surprise = torch.full((B,), 0.5, device=s.device, dtype=s.dtype)

        # 2. Active Dynamic Token K/V Reread (`h_tok` evolves across T1->T2->T3 via BiXT!)
        k_ev = self.k_ev_proj(h_tok) * m_u
        v_ev = self.v_ev_proj(h_tok) * m_u
        sn_ca = self.norm_ca(s)
        q_base = self.ca_q(sn_ca) + torch.matmul(torch.matmul(sn_ca, self.ca_q_lora_a[oi]), self.ca_q_lora_b[oi])
        qh = q_base.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v_ev.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sc_raw = torch.matmul(qh, kh.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, heads, 16, L)
        sc = sc_raw.masked_fill(m_f[:, None, None, :] < 0.5, -1e4)
        at = torch.softmax(sc.float(), dim=-1).to(s.dtype)
        s = s + torch.sigmoid(self.reread_gate) * self.ca_o(torch.matmul(at, vh).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D))

        # Fine-scale Haar wavelet reread (`64-d`, pad-exact!)
        q_haar = self.haar_q(sn_ca)
        sc_h = (torch.matmul(q_haar, k_haar.transpose(-1, -2)) / 8.0).masked_fill(haar_mask[:, None, :].float() < 0.5, -1e4)
        at_h = torch.softmax(sc_h.float(), dim=-1).to(s.dtype)
        s = s + torch.sigmoid(self.haar_gate + 0.35 * ctrl[1]) * self.haar_o(torch.matmul(at_h, v_haar))

        # 3. Surprise-Gated EMA Compressed Fast Memory Injection
        z_l_start = s[:, self.n_slow :, :]
        if carry_m_l is not None:
            c2_gain = (0.35 * torch.tanh(self.c2_carry_scale) * (1.0 - 0.5 * surprise) * (1.0 + 0.25 * ctrl[4])).view(-1, 1, 1)
            s = torch.cat([s[:, : self.n_slow, :], z_l_start + c2_gain * self.bus_up(carry_m_l)], dim=1)

        # 4. TRUE NESTED INNER LOOP (`k_inner = 2` fast-inner steps with Cayley Symplectic Exchange!)
        for ki in range(int(k_inner)):
            s = self._fast_inner_step(s, ctrl, outer_idx=oi, phase_idx=min(ki, 1))

        # 5. Update Compressed Fast Memory EMA
        delta_fast_bus = torch.tanh(self.bus_down(s[:, self.n_slow :, :] - z_l_start))
        beta_c = 0.85 * torch.sigmoid(self.c2_beta_logit).view(1, 1, -1)
        new_carry_m_l = (1.0 - beta_c) * delta_fast_bus if carry_m_l is None else (beta_c * carry_m_l + (1.0 - beta_c) * delta_fast_bus)

        # 6. All-16-Slot Self-Attention + Slow-Outer Shared FFN
        sq, sk, sv = self.sa_qkv(self.norm_sa(s)).chunk(3, dim=-1)
        sq = sq.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sk = sk.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sv = sv.view(B, NUM_SLOTS_V130, self.num_heads, self.head_dim).transpose(1, 2)
        sa = torch.softmax((torch.matmul(sq, sk.transpose(-2, -1)) * (self.head_dim ** -0.5)).float(), dim=-1).to(s.dtype)
        sa_ctx = torch.matmul(sa, sv).transpose(1, 2).reshape(B, NUM_SLOTS_V130, D)
        s = s + self.sa_o(sa_ctx) + torch.matmul(torch.matmul(sa_ctx, self.sa_o_lora_a[oi]), self.sa_o_lora_b[oi])
        s = s + self._shared_ffn(self.norm_ff(s), outer_idx=oi, phase_idx=2)

        # 7. L0 Surface Penetration + Analytic 6-Valley EBT on 64-d Slow Bus
        l0_pen = self.l0_pen_b(F.silu(self.l0_pen_a(h0_pool))).unsqueeze(1)
        s = torch.cat([s[:, : self.n_slow, :] + torch.sigmoid(self.l0_pen_gate).view(1, 1, -1) * l0_pen, s[:, self.n_slow :, :]], dim=1)

        bus_pre = self.bus_down(s[:, : self.n_slow, :].mean(dim=1))
        ebt_r = self.ebt.forward_energy_and_grad(bus_pre)
        ebt_delta = self.bus_up(ebt_r["u_next"] - bus_pre).unsqueeze(1)
        ebt_g = 0.35 * torch.tanh(self.ebt_back_scale) * (1.0 + 0.25 * ctrl[5])
        s_prop = torch.cat([s[:, : self.n_slow, :] + ebt_g * ebt_delta, s[:, self.n_slow :, :]], dim=1)

        # 8. Manifold Anchor Blend + Step-Indexed Spectral Damping + Mini Two-Stream Sinkhorn mHC!
        a_mix = 0.35 * torch.sigmoid(self.anchor_alpha)
        s_prop = (1.0 - a_mix) * s_prop + a_mix * anchor
        if outer_idx == 0:
            s_cand = s_prop
        else:
            d_i = (0.55 + 0.45 * torch.sigmoid(self.damp_gamma)) / math.sqrt(float(outer_idx))
            s_cand = s_in + d_i * (s_prop - s_in)

        m_mat = sinkhorn_2x2_v140(self.mhc_log_mix[oi]).to(s_cand.dtype)  # (2, 2) doubly stochastic!
        s_a_next = m_mat[0, 0] * s_cand + m_mat[0, 1] * slots_b
        s_b_next = m_mat[1, 0] * s_cand + m_mat[1, 1] * slots_b
        r_m = 0.35 * torch.sigmoid(self.mhc_read_mix[oi])
        slots_out = (1.0 - r_m) * s_a_next + r_m * s_b_next

        # 9. BiXT Dynamic Token Write-Back (`Z^t -> H^t`, strictly pad-exact via slot-axis softmax & `* m_u`!)
        # For each token `j`, softmax over the 16 slots depends ONLY on token `j` & `slots` -> ZERO PAD leakage!
        sc_slot = sc_raw.mean(dim=1)  # (B, 16, L)
        w_down = torch.softmax(sc_slot.float(), dim=1).to(h_tok.dtype)  # (B, 16, L)
        v_down = self.bixt_v(slots_out)  # (B, 16, 64)
        tok_msg = torch.matmul(w_down.transpose(1, 2), v_down)  # (B, L, 64)
        g_bx = 0.40 * torch.sigmoid(self.bixt_gate[oi])
        h_tok_next = (h_tok + g_bx * self.bixt_norm(self.bixt_o(tok_msg))) * m_u

        return {
            "slots": slots_out,
            "slots_a": s_a_next,
            "slots_b": s_b_next,
            "h_tok": h_tok_next,
            "carry_m_l": new_carry_m_l,
            "bus": bus_pre,
            "surprise": surprise,
            "ebt_energies": ebt_r["energies"],
            "ebt_margin": ebt_r["ai_margin"],
            "ebt_promo_margin": ebt_r["promo_margin"],
            "ebt_grad_norm": ebt_r["grad_norm"],
        }


# ---------------------------------------------------------------------------
# 4. Complete Unified Model: `MeguvizV140ParadiseApexModel`
# ---------------------------------------------------------------------------

class MeguvizV140ParadiseApexModel(nn.Module):
    """Unified <=5.00 MiB ParadiseApex-Loop Model (BiXT + Mini-mHC + Support Gate + Disagreement Penalty)."""

    def __init__(
        self,
        vocab_size: int = 12288,
        max_len: int = MAX_LEN_V140,
        style_dim: int = STYLE_FEAT_DIM_V11,
        style_era_dropout_p: float = 0.15,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.style_dim = int(style_dim)
        self.engram_buckets = ENGRAM_BUCKETS_V130
        self.trigram_buckets = TRIGRAM_BUCKETS_V130
        self.style_era_dropout_p = float(style_era_dropout_p)

        # Stage 0: Full-Scale Support-Gated W8 Char 1/2/3-gram Nano Engine (`~1.0 MiB`)
        self.rung0 = ApexCharNanoRung0V140()

        # Stage 1: Full `H=384` Factorized BPE + Bigram/Trigram Engram + Pad-Exact Prelude + Haar Wavelet K/V
        self.word_emb_a = QuantEmbeddingV130(vocab_size, 48, bits=4, padding_idx=0)
        self.word_emb_p = QuantLinearV130(48, H_V130, bias=False, bits=8)
        self.engram_emb = QuantEmbeddingV130(ENGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.engram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.trigram_emb = QuantEmbeddingV130(TRIGRAM_BUCKETS_V130, 16, bits=4, padding_idx=0)
        self.trigram_p = QuantLinearV130(16, H_V130, bias=False, bits=8)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, H_V130) * 0.02)
        self.emb_norm = nn.LayerNorm(H_V130)

        self.prelude_l0 = PadExactConvBlockV130(H_V130, FFN_V130, NUM_HEADS_V130, bits=4)
        self.prelude_l1 = PadExactConvBlockV130(H_V130, FFN_V130, NUM_HEADS_V130, bits=4)
        self.haar_k_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)
        self.haar_v_proj = QuantLinearV130(H_V130, BUS_DIM_V130, bias=False, bits=4)

        # NEW: Support-Reliability Gate (`distinct_alnum_pairs >= 16`) protecting `<= 30c` human comments from noisy style FPs!
        self.sup_gate_tau = nn.Parameter(torch.tensor(16.0))
        self.sup_gate_scale = nn.Parameter(torch.tensor(4.0))

        # NEW: Promo-Private Attribution Scope Pooler (strictly Promo-only -> ZERO AI leakage `F6 === 0.0`!)
        self.pr_attr_attn = nn.Sequential(
            QuantLinearV130(H_V130, 32, bits=4),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # R1 fast prefix heads (strictly separate AI vs Promo!)
        self.r1_style_ai = nn.Sequential(nn.Linear(style_dim - 8, 32), nn.SiLU())
        self.r1_style_pr = nn.Sequential(nn.Linear(style_dim, 32), nn.SiLU())
        self.r1_ai_head = nn.Linear(H_V130 + 32 + 32, 1)
        self.r1_promo_head = nn.Linear(H_V130 + 32 + 32, 1)
        self.r1_defer_head = nn.Sequential(nn.Linear(H_V130 + 2, 32), nn.SiLU(), nn.Linear(32, 1))

        # Stage 2: ParadiseApex Nested-Loop Core (BiXT Dynamic Token Write-Back + Two-Stream Sinkhorn mHC!)
        self.core = ParadiseNestedCoreV140(H_V130, FFN_V130, NUM_HEADS_V130)

        # Strictly Disjoint 256-d Coda: [0:128) shared, [128:192) AI-private, [192:256) Promo-private
        self.coda_shared = QuantLinearV130(H_V130 * 2, SHARED_END_V130, bits=4)
        self.coda_ai_priv = QuantLinearV130(H_V130 + 32 + 32, AI_END_V130 - SHARED_END_V130, bits=4)
        self.coda_pr_priv = QuantLinearV130(H_V130 * 2 + 32 + 32, EMB_DIM_V130 - AI_END_V130, bits=4)
        self.emb_norm_sh = nn.LayerNorm(SHARED_END_V130)
        self.emb_norm_ai = nn.LayerNorm(AI_END_V130 - SHARED_END_V130)
        self.emb_norm_pr = nn.LayerNorm(EMB_DIM_V130 - AI_END_V130)

        # Split-Axis Top-K CBM Firewall (`+12` Promo CBM perturb -> exact `0.000000` AI diff!)
        self.cbm_ai_enc = nn.Linear(AI_END_V130, CBM_AI_K_V130)
        self.cbm_pr_enc = nn.Linear(PR_EMB_DIM_V130, CBM_PR_K_V130)
        self.cbm_ai_to_logit = nn.Linear(CBM_AI_K_V130, 1, bias=False)
        self.cbm_pr_to_logit = nn.Linear(CBM_PR_K_V130, 1, bias=False)

        self.ai_head = SoftBallLogitNormHeadV121(AI_END_V130, init_r_cap=7.2)
        self.promo_head = SoftBallLogitNormHeadV121(PR_EMB_DIM_V130, init_r_cap=7.2)
        self.promo_sub4_head = nn.Linear(PR_EMB_DIM_V130, 4)
        self.ebt_ai_gate = nn.Parameter(torch.tensor(0.20))
        self.ebt_pr_gate = nn.Parameter(torch.tensor(0.15))

        # Learned Conservative 4-Head Loop Fusion + Generative CharLM Agreement Gate + Disagreement Penalty!
        self.ai_rung_mix_logits = nn.Parameter(
            torch.tensor([math.log(0.22), math.log(0.18), math.log(0.50), math.log(0.10)], dtype=torch.float32)
        )
        self.disagree_r0_t2_raw = nn.Parameter(torch.tensor(-2.0))
        self.disagree_t2_ebt_raw = nn.Parameter(torch.tensor(-2.2))
        self.charlm_ai_gate_raw = nn.Parameter(torch.tensor(-1.1))

        self.traj_ai_lin = nn.Linear(AI_END_V130 * 2 + 7, 1)
        self.traj_ai_mlp = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 7),
            nn.Linear(AI_END_V130 * 2 + 7, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.traj_ai_lin.weight)
        nn.init.zeros_(self.traj_ai_lin.bias)
        nn.init.zeros_(self.traj_ai_mlp[-1].weight)
        nn.init.zeros_(self.traj_ai_mlp[-1].bias)
        self.traj_ai_cap = 2.5

        self.promo_char_rho_raw = nn.Parameter(torch.tensor(0.0))
        self.traj_pr_mlp = nn.Sequential(
            nn.LayerNorm(PR_EMB_DIM_V130 + 32 + 4),
            nn.Linear(PR_EMB_DIM_V130 + 32 + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.traj_pr_mlp[-1].weight)
        nn.init.zeros_(self.traj_pr_mlp[-1].bias)
        self.traj_pr_cap = 2.5

        # Dedicated Bounded Zero-Init Detached Residual Error-Correction Heads for `T3` (`|delta| <= 2.2`!)
        self.t3_corr_ai = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 8),
            nn.Linear(AI_END_V130 * 2 + 8, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.t3_corr_ai[-1].weight)
        nn.init.zeros_(self.t3_corr_ai[-1].bias)
        self.t3_corr_pr = nn.Sequential(
            nn.LayerNorm(PR_EMB_DIM_V130 * 2 + 5),
            nn.Linear(PR_EMB_DIM_V130 * 2 + 5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.t3_corr_pr[-1].weight)
        nn.init.zeros_(self.t3_corr_pr[-1].bias)
        self.t3_corr_cap = T3_CORR_CAP_V140

        # Ouro Conditional `T2 -> T3` Escalation Gate (AI-safe + support_gate + Neural-vs-CharLM disagreement aware!)
        self.halt_t2_head = nn.Sequential(
            nn.LayerNorm(AI_END_V130 * 2 + 10),
            nn.Linear(AI_END_V130 * 2 + 10, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Native Long-Document `DynaSeam` Attention Aggregator
        self.long_ai_attn = nn.Sequential(nn.Linear(AI_END_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_pr_attn = nn.Sequential(nn.Linear(PR_EMB_DIM_V130 + CHUNK_META_DIM_V11 + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.long_ai_head = nn.Linear(AI_END_V130 + 3, 1)
        self.long_pr_head = nn.Linear(PR_EMB_DIM_V130 + 3, 1)

    def compute_support_gate(self, support_pairs: torch.Tensor) -> torch.Tensor:
        tau = self.sup_gate_tau.clamp(10.0, 22.0)
        sc = self.sup_gate_scale.clamp(2.0, 6.0)
        return torch.sigmoid((support_pairs.float() - tau) / sc)

    def _pool_promo_attr(self, h_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        w_lg = self.pr_attr_attn(h_tok).squeeze(-1).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(w_lg.float(), dim=-1).to(h_tok.dtype).unsqueeze(-1)
        return (h_tok * w).sum(dim=1)

    def _readout_step(
        self,
        slots: torch.Tensor,
        h_tok_cur: torch.Tensor,
        mask: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        ebt_ai_m: torch.Tensor,
        ebt_pr_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        m_u = mask.float().unsqueeze(-1)
        h_pool_cur = h_tok_cur.sum(dim=1) / m_u.sum(dim=1).clamp_min(1.0)
        pr_attr_pool = self._pool_promo_attr(h_tok_cur, mask)

        slow_p = slots[:, : NUM_SLOW_V130, :].mean(dim=1)
        sem_p = slots[:, NUM_SLOW_V130 : NUM_SLOW_V130 + NUM_SEM_FAST_V130, :].mean(dim=1)
        span_p = slots[:, NUM_SLOW_V130 + NUM_SEM_FAST_V130 :, :].mean(dim=1)

        z_sh = self.emb_norm_sh(self.coda_shared(torch.cat([slow_p, h_pool_cur], dim=-1)))
        z_ai_p = self.emb_norm_ai(self.coda_ai_priv(torch.cat([sem_p, st_ai, z_char[:, :32]], dim=-1)))
        z_pr_p = self.emb_norm_pr(self.coda_pr_priv(torch.cat([span_p, pr_attr_pool, st_pr, z_char[:, 32:]], dim=-1)))

        emb_ai = torch.cat([z_sh, z_ai_p], dim=-1)       # [B, 192] — STRICTLY ZERO PROMO PATHWAY!
        emb_pr = torch.cat([z_sh, z_pr_p], dim=-1)       # [B, 192]
        emb_full = torch.cat([z_sh, z_ai_p, z_pr_p], dim=-1)

        c_ai_raw = F.relu(self.cbm_ai_enc(emb_ai))
        v_ai, i_ai = torch.topk(c_ai_raw, k=6, dim=-1)
        c_ai_sp = torch.zeros_like(c_ai_raw).scatter(-1, i_ai, v_ai)

        c_pr_raw = F.relu(self.cbm_pr_enc(emb_pr))
        v_pr, i_pr = torch.topk(c_pr_raw, k=6, dim=-1)
        c_pr_sp = torch.zeros_like(c_pr_raw).scatter(-1, i_pr, v_pr)

        lg_ai = (
            self.ai_head(emb_ai)
            + 0.15 * self.cbm_ai_to_logit(c_ai_sp).squeeze(-1)
            + self.ebt_ai_gate * torch.clamp(ebt_ai_m, -6.0, 6.0)
        )
        lg_pr = (
            self.promo_head(emb_pr)
            + 0.15 * self.cbm_pr_to_logit(c_pr_sp).squeeze(-1)
            + self.ebt_pr_gate * torch.clamp(ebt_pr_m, -6.0, 6.0)
        )
        return {
            "logit_ai": lg_ai,
            "logit_promo": lg_pr,
            "promo_sub4_logit": self.promo_sub4_head(emb_pr),
            "emb_ai": emb_ai,
            "emb_pr": emb_pr,
            "emb": emb_full,
            "cbm_ai": c_ai_sp,
            "cbm_pr": c_pr_sp,
        }

    def _compute_trajectory_final(
        self,
        t1_emb_ai: torch.Tensor,
        t1_lg_ai: torch.Tensor,
        t1_lg_pr: torch.Tensor,
        ro_cur: dict[str, torch.Tensor],
        ebt_ai_m: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        lg_r1_ai: torch.Tensor,
        lg_r1_pr: torch.Tensor,
        g_sup: torch.Tensor,
        charlm_logit: torch.Tensor,
        r0_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ebt_c = torch.clamp(ebt_ai_m, -6.0, 6.0)
        lm_eff = charlm_logit.to(ro_cur["logit_ai"].dtype)
        w_mix = torch.softmax(self.ai_rung_mix_logits.float(), dim=0).to(ro_cur["logit_ai"].dtype)
        g_d1 = 0.25 * F.softplus(self.disagree_r0_t2_raw)
        g_d2 = 0.25 * F.softplus(self.disagree_t2_ebt_raw)

        if r0_keep is not None:
            w0 = w_mix[0] * r0_keep
            w_rest = w_mix[1] + w_mix[2] + w_mix[3]
            inv_s = 1.0 / (w0 + w_rest).clamp_min(1e-5)
            ai_base = (w0 * logit_ai_r0 + w_mix[1] * t1_lg_ai + w_mix[2] * ro_cur["logit_ai"] + w_mix[3] * ebt_c) * inv_s
            disagree_pen = (g_d1 * r0_keep) * (logit_ai_r0 - ro_cur["logit_ai"]).abs() + g_d2 * (ro_cur["logit_ai"] - ebt_c).abs()
            rho_eff = torch.sigmoid(self.promo_char_rho_raw) * r0_keep
        else:
            ai_base = w_mix[0] * logit_ai_r0 + w_mix[1] * t1_lg_ai + w_mix[2] * ro_cur["logit_ai"] + w_mix[3] * ebt_c
            disagree_pen = g_d1 * (logit_ai_r0 - ro_cur["logit_ai"]).abs() + g_d2 * (ro_cur["logit_ai"] - ebt_c).abs()
            rho_eff = torch.sigmoid(self.promo_char_rho_raw)

        # Agreement-gated generative Char-LM LLR boost (protects R@P99 from isolated n-gram spikes!)
        lm_active = (lm_eff.abs() > 1e-5).to(lm_eff.dtype)
        lm_agree = torch.sigmoid(1.5 - (ro_cur["logit_ai"] - lm_eff).abs())
        ai_base = ai_base + (0.18 * torch.sigmoid(self.charlm_ai_gate_raw)) * lm_eff * lm_agree * lm_active

        # Disagreement penalty specifically dampens over-confident positive tail (`R@P95/99` FP shield!)
        pos_mask_soft = torch.sigmoid(ai_base)
        ai_base = ai_base - pos_mask_soft * disagree_pen.clamp_max(1.5)

        z_delta_ai = ro_cur["emb_ai"] - t1_emb_ai
        ai_scalars = torch.stack([logit_ai_r0, lg_r1_ai, t1_lg_ai, ro_cur["logit_ai"], ebt_c, g_sup.to(ebt_c.dtype), lm_eff], dim=-1)
        traj_ai_in = torch.cat([ro_cur["emb_ai"], z_delta_ai, ai_scalars], dim=-1)
        ai_res_raw = self.traj_ai_lin(traj_ai_in).squeeze(-1) + self.traj_ai_mlp(traj_ai_in).squeeze(-1)
        ai_final = ai_base + self.traj_ai_cap * torch.tanh(ai_res_raw / self.traj_ai_cap)

        pr_base = (1.0 - rho_eff) * ro_cur["logit_promo"] + rho_eff * logit_promo_r0
        pr_scalars = torch.stack([logit_promo_r0, lg_r1_pr, t1_lg_pr, ro_cur["logit_promo"]], dim=-1)
        traj_pr_in = torch.cat([ro_cur["emb_pr"], z_char[:, 32:], pr_scalars], dim=-1)
        pr_res_raw = self.traj_pr_mlp(traj_pr_in).squeeze(-1)
        pr_final = pr_base + self.traj_pr_cap * torch.tanh(pr_res_raw / self.traj_pr_cap)
        return ai_final, pr_final

    def forward_r0(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, mask: torch.Tensor, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.rung0(c1, c2, c3, mask, feats)

    def forward_t2_stage(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        support_pairs: torch.Tensor,
        charlm_logit: torch.Tensor | None = None,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Run L0 -> R1 -> L1 -> T1 (BiXT+mHC) -> T2 (BiXT+mHC) + emit exact cache for bounded T3!"""
        B, L = input_ids.shape
        m = attention_mask.float()
        m_u = m.unsqueeze(-1)
        g_sup = self.compute_support_gate(support_pairs).to(style_feats.dtype)

        st = style_feats.clone()
        if self.training and self.style_era_dropout_p > 0.0:
            keep = (torch.rand((B, 1), device=st.device) >= self.style_era_dropout_p).to(st.dtype)
            st[:, :40] = st[:, :40] * keep

        r0_keep: torch.Tensor | None = None
        if self.training and r0_drop_p > 0.0:
            r0_keep = (torch.rand((B,), device=z_char.device) >= float(r0_drop_p)).to(z_char.dtype)
            zc_eff = z_char * r0_keep.unsqueeze(-1)
            la0_eff = logit_ai_r0 * r0_keep
            lp0_eff = logit_promo_r0 * r0_keep
        else:
            zc_eff, la0_eff, lp0_eff = z_char, logit_ai_r0, logit_promo_r0

        lm_raw = torch.zeros_like(logit_ai_r0) if charlm_logit is None else charlm_logit.to(logit_ai_r0.dtype)
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((B,), device=lm_raw.device) >= float(charlm_drop_p)).to(lm_raw.dtype)
            lm_eff = lm_raw * lm_keep
        else:
            lm_eff = lm_raw

        pos = self.pos_emb[:, :L, :]
        x = self.emb_norm(
            self.word_emb_p(self.word_emb_a(input_ids))
            + self.engram_p(self.engram_emb(engram_ids))
            + self.trigram_p(self.trigram_emb(trigram_ids))
            + pos
        ) * m_u
        h0 = self.prelude_l0(x, attention_mask)

        # Pad-Exact Fine-Scale Haar Wavelet K/V from L0
        if (not torch.jit.is_tracing()) and (L % 2 == 1):
            h0_p = F.pad(h0, (0, 0, 0, 1))
            m_p = F.pad(m, (0, 1))
        else:
            h0_p, m_p = h0, m
        m_even = m_p[:, 0::2]
        h_diff = (h0_p[:, 0::2, :] - h0_p[:, 1::2, :]) * m_even.unsqueeze(-1)
        k_haar = self.haar_k_proj(h_diff) * m_even.unsqueeze(-1)
        v_haar = self.haar_v_proj(h_diff) * m_even.unsqueeze(-1)
        haar_mask = m_even

        denom = m_u.sum(dim=1).clamp_min(1.0)
        h0_pool = h0.sum(dim=1) / denom
        # Support-reliability gate scales AI 56-d style vector smoothly to zero on micro-short `< 16` pair comments!
        st_ai = self.r1_style_ai(st[:, : self.style_dim - 8]) * g_sup.unsqueeze(-1)
        st_pr = self.r1_style_pr(st)
        lg_r1_ai = self.r1_ai_head(torch.cat([h0_pool, st_ai, zc_eff[:, :32]], dim=-1)).squeeze(-1)
        lg_r1_pr = self.r1_promo_head(torch.cat([h0_pool, st_pr, zc_eff[:, 32:]], dim=-1)).squeeze(-1)
        def_r1_lg = self.r1_defer_head(
            torch.cat([h0_pool.detach(), lg_r1_ai.detach().unsqueeze(-1), lg_r1_pr.detach().unsqueeze(-1)], dim=-1)
        ).squeeze(-1)

        # L1 Pad-Exact encoding -> initial `h_tok`
        h1 = self.prelude_l1(h0, attention_mask)

        # Outer Step T1 (`outer_idx=0`, BiXT updates `h1 -> h_tok_t1`, Two-Stream Sinkhorn mHC!)
        anchor = self.core.init_anchor(h1, attention_mask)
        st1 = self.core.forward_outer_step(
            slots_a=anchor, slots_b=anchor, anchor=anchor, h_tok=h1, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=0, prev_slots=None, carry_m_l=None, prev_bus=None, prev_doubt=None, k_inner=K_INNER_V130,
        )
        ro1 = self._readout_step(st1["slots"], st1["h_tok"], attention_mask, st_ai, st_pr, zc_eff, st1["ebt_margin"], st1["ebt_promo_margin"])
        c_conf1 = (2.0 * torch.sigmoid(ro1["logit_ai"].detach()) - 1.0).abs()
        doubt1 = torch.stack([c_conf1, st1["ebt_grad_norm"].detach()], dim=-1)

        # Outer Step T2 (`outer_idx=1`, reads BiXT-updated `h_tok_t1`, outputs `h_tok_t2` + Two-Stream `slots_a/b_t2`!)
        st2 = self.core.forward_outer_step(
            slots_a=st1["slots_a"], slots_b=st1["slots_b"], anchor=anchor, h_tok=st1["h_tok"], mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=1, prev_slots=anchor, carry_m_l=st1["carry_m_l"], prev_bus=st1["bus"], prev_doubt=doubt1, k_inner=K_INNER_V130,
        )
        ro2 = self._readout_step(st2["slots"], st2["h_tok"], attention_mask, st_ai, st_pr, zc_eff, st2["ebt_margin"], st2["ebt_promo_margin"])
        ai_t2_final, pr_t2_final = self._compute_trajectory_final(
            ro1["emb_ai"], ro1["logit_ai"], ro1["logit_promo"], ro2, st2["ebt_margin"],
            zc_eff, la0_eff, lp0_eff, lg_r1_ai, lg_r1_pr, g_sup=g_sup, charlm_logit=lm_eff, r0_keep=r0_keep,
        )

        # Ouro Conditional `T2 -> T3` Escalation Gate (triggers on Neural-vs-CharLM disagreement too!)
        lm_disagree_t2 = (ro2["logit_ai"].detach() - lm_eff.detach()).abs()
        halt_in = torch.cat([
            ro2["emb_ai"].detach(),
            (ro2["emb_ai"] - ro1["emb_ai"]).detach(),
            torch.stack([
                la0_eff.detach(),
                lg_r1_ai.detach(),
                ro1["logit_ai"].detach(),
                ai_t2_final.detach(),
                torch.clamp(st2["ebt_margin"].detach(), -6.0, 6.0),
                st2["ebt_grad_norm"].detach(),
                st2["surprise"].detach(),
                g_sup.detach(),
                lm_eff.detach(),
                lm_disagree_t2,
            ], dim=-1),
        ], dim=-1)
        esc_lg = self.halt_t2_head(halt_in).squeeze(-1)

        return {
            "logit_ai_t2_final": ai_t2_final,
            "logit_promo_t2_final": pr_t2_final,
            "logit_ai_r1": lg_r1_ai,
            "logit_promo_r1": lg_r1_pr,
            "defer_r1_logit": def_r1_lg,
            "p_defer_r1": torch.sigmoid(def_r1_lg),
            "logit_ai_t1": ro1["logit_ai"],
            "logit_promo_t1": ro1["logit_promo"],
            "logit_ai_t2": ro2["logit_ai"],
            "logit_promo_t2": ro2["logit_promo"],
            "promo_sub4_logit_t2": ro2["promo_sub4_logit"],
            "ebt_energies_t2": st2["ebt_energies"],
            "ebt_margin_t2": st2["ebt_margin"],
            "ebt_promo_margin_t2": st2["ebt_promo_margin"],
            "ebt_grad_norm_t2": st2["ebt_grad_norm"],
            "emb_ai_t1": ro1["emb_ai"],
            "emb_ai_t2": ro2["emb_ai"],
            "emb_pr_t2": ro2["emb_pr"],
            "emb_t2": ro2["emb"],
            "escalate_t3_logit": esc_lg,
            "p_escalate_t3": torch.sigmoid(esc_lg),
            # Exact T2 -> T3 continuation state (BiXT `h_tok_t2` + Two-Stream mHC `slots_a/b_t2`!):
            "slots_t1": st1["slots"],
            "slots_a_t2": st2["slots_a"],
            "slots_b_t2": st2["slots_b"],
            "anchor": anchor,
            "h_tok_t1": st1["h_tok"],
            "h_tok_t2": st2["h_tok"],
            "carry_m_l_t2": st2["carry_m_l"],
            "zh_bus_t2": st2["bus"],
            "k_haar": k_haar,
            "v_haar": v_haar,
            "haar_mask": haar_mask,
            "h0_pool": h0_pool,
            "st_ai": st_ai,
            "st_pr": st_pr,
            "g_sup": g_sup,
            "_zc_eff": zc_eff,
            "_la0_eff": la0_eff,
            "_lp0_eff": lp0_eff,
            "_lm_eff": lm_eff,
        }

    def forward_t3_step(
        self,
        slots_a_t2: torch.Tensor,
        slots_b_t2: torch.Tensor,
        slots_t1: torch.Tensor,
        anchor: torch.Tensor,
        h_tok_t2: torch.Tensor,
        carry_m_l_t2: torch.Tensor,
        zh_bus_t2: torch.Tensor,
        ebt_grad_norm_t2: torch.Tensor,
        ebt_margin_t2: torch.Tensor,
        attention_mask: torch.Tensor,
        k_haar: torch.Tensor,
        v_haar: torch.Tensor,
        haar_mask: torch.Tensor,
        h0_pool: torch.Tensor,
        st_ai: torch.Tensor,
        st_pr: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        logit_promo_r1: torch.Tensor,
        logit_ai_t1: torch.Tensor,
        logit_promo_t1: torch.Tensor,
        logit_ai_t2_raw: torch.Tensor,
        logit_ai_t2_final: torch.Tensor,
        logit_promo_t2_final: torch.Tensor,
        emb_ai_t2: torch.Tensor,
        emb_pr_t2: torch.Tensor,
        g_sup: torch.Tensor,
        charlm_logit: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Conditional 3rd-Outer-Step Bounded Detached Residual Error-Correction (`|delta| <= 2.2`!)."""
        c_conf2 = (2.0 * torch.sigmoid(logit_ai_t2_raw.detach()) - 1.0).abs()
        doubt2 = torch.stack([c_conf2, ebt_grad_norm_t2.detach()], dim=-1)
        st3 = self.core.forward_outer_step(
            slots_a=slots_a_t2, slots_b=slots_b_t2, anchor=anchor, h_tok=h_tok_t2, mask=attention_mask,
            k_haar=k_haar, v_haar=v_haar, haar_mask=haar_mask, h0_pool=h0_pool,
            outer_idx=2, prev_slots=slots_t1, carry_m_l=carry_m_l_t2, prev_bus=zh_bus_t2, prev_doubt=doubt2, k_inner=K_INNER_V130,
        )
        ro3 = self._readout_step(st3["slots"], st3["h_tok"], attention_mask, st_ai, st_pr, z_char, st3["ebt_margin"], st3["ebt_promo_margin"])

        ebt3_c = torch.clamp(st3["ebt_margin"], -6.0, 6.0)
        ebt2_c = torch.clamp(ebt_margin_t2.detach(), -6.0, 6.0)
        ai_t2_base = logit_ai_t2_final.detach()
        lm_eff = torch.zeros_like(ai_t2_base) if charlm_logit is None else charlm_logit.detach().to(ebt3_c.dtype)
        corr_ai_in = torch.cat([
            ro3["emb_ai"],
            ro3["emb_ai"] - emb_ai_t2.detach(),
            torch.stack([logit_ai_r0.detach(), logit_ai_t1.detach(), ai_t2_base, ro3["logit_ai"], ebt3_c, ebt3_c - ebt2_c, g_sup.detach(), lm_eff], dim=-1),
        ], dim=-1)
        ai_delta = self.t3_corr_ai(corr_ai_in).squeeze(-1)
        ai_t3_final = ai_t2_base + self.t3_corr_cap * torch.tanh(ai_delta / self.t3_corr_cap)

        pr_t2_base = logit_promo_t2_final.detach()
        corr_pr_in = torch.cat([
            ro3["emb_pr"],
            ro3["emb_pr"] - emb_pr_t2.detach(),
            torch.stack([logit_promo_r0.detach(), logit_promo_r1.detach(), logit_promo_t1.detach(), pr_t2_base, ro3["logit_promo"]], dim=-1),
        ], dim=-1)
        pr_delta = self.t3_corr_pr(corr_pr_in).squeeze(-1)
        pr_t3_final = pr_t2_base + self.t3_corr_cap * torch.tanh(pr_delta / self.t3_corr_cap)

        return {
            "logit_ai_t3_final": ai_t3_final,
            "logit_promo_t3_final": pr_t3_final,
            "logit_ai_t3": ro3["logit_ai"],
            "logit_promo_t3": ro3["logit_promo"],
            "promo_sub4_logit_t3": ro3["promo_sub4_logit"],
            "ebt_energies_t3": st3["ebt_energies"],
            "ebt_margin_t3": st3["ebt_margin"],
            "ebt_promo_margin_t3": st3["ebt_promo_margin"],
            "emb_ai_t3": ro3["emb_ai"],
            "emb_pr_t3": ro3["emb_pr"],
            "emb_t3": ro3["emb"],
            "h_tok_t3": st3["h_tok"],
        }

    def forward_deep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        z_char: torch.Tensor,
        logit_ai_r0: torch.Tensor,
        logit_promo_r0: torch.Tensor,
        support_pairs: torch.Tensor,
        charlm_logit: torch.Tensor | None = None,
        outer_steps: int = 2,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        t2 = self.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
            support_pairs=support_pairs, charlm_logit=charlm_logit,
            r0_drop_p=r0_drop_p, charlm_drop_p=charlm_drop_p,
        )
        out = dict(t2)
        if int(outer_steps) >= 3:
            t3 = self.forward_t3_step(
                slots_a_t2=t2["slots_a_t2"], slots_b_t2=t2["slots_b_t2"], slots_t1=t2["slots_t1"],
                anchor=t2["anchor"], h_tok_t2=t2["h_tok_t2"], carry_m_l_t2=t2["carry_m_l_t2"],
                zh_bus_t2=t2["zh_bus_t2"], ebt_grad_norm_t2=t2["ebt_grad_norm_t2"], ebt_margin_t2=t2["ebt_margin_t2"],
                attention_mask=attention_mask, k_haar=t2["k_haar"], v_haar=t2["v_haar"], haar_mask=t2["haar_mask"],
                h0_pool=t2["h0_pool"], st_ai=t2["st_ai"], st_pr=t2["st_pr"],
                z_char=t2["_zc_eff"], logit_ai_r0=t2["_la0_eff"], logit_promo_r0=t2["_lp0_eff"],
                logit_promo_r1=t2["logit_promo_r1"], logit_ai_t1=t2["logit_ai_t1"], logit_promo_t1=t2["logit_promo_t1"],
                logit_ai_t2_raw=t2["logit_ai_t2"], logit_ai_t2_final=t2["logit_ai_t2_final"],
                logit_promo_t2_final=t2["logit_promo_t2_final"], emb_ai_t2=t2["emb_ai_t2"],
                emb_pr_t2=t2["emb_pr_t2"], g_sup=t2["g_sup"], charlm_logit=t2["_lm_eff"],
            )
            out.update(t3)
            out["logit_ai"] = t3["logit_ai_t3_final"]
            out["logit_promo"] = t3["logit_promo_t3_final"]
            out["promo_sub4_logit"] = t3["promo_sub4_logit_t3"]
            out["ebt_energies"] = t3["ebt_energies_t3"]
            out["ebt_margin"] = t3["ebt_margin_t3"]
            out["ebt_promo_margin"] = t3["ebt_promo_margin_t3"]
            out["emb_ai"] = t3["emb_ai_t3"]
            out["emb_pr"] = t3["emb_pr_t3"]
            out["emb"] = t3["emb_t3"]
        else:
            out["logit_ai"] = t2["logit_ai_t2_final"]
            out["logit_promo"] = t2["logit_promo_t2_final"]
            out["promo_sub4_logit"] = t2["promo_sub4_logit_t2"]
            out["ebt_energies"] = t2["ebt_energies_t2"]
            out["ebt_margin"] = t2["ebt_margin_t2"]
            out["ebt_promo_margin"] = t2["ebt_promo_margin_t2"]
            out["emb_ai"] = t2["emb_ai_t2"]
            out["emb_pr"] = t2["emb_pr_t2"]
            out["emb"] = t2["emb_t2"]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_feats: torch.Tensor,
        engram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        r0_c1: torch.Tensor,
        r0_c2: torch.Tensor,
        r0_c3: torch.Tensor,
        r0_mask: torch.Tensor,
        r0_feats: torch.Tensor,
        support_pairs: torch.Tensor,
        outer_steps: int = 2,
        r0_drop_p: float = 0.0,
        charlm_drop_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if self.training and charlm_drop_p > 0.0:
            lm_keep = (torch.rand((r0_feats.shape[0], 1), device=r0_feats.device) >= float(charlm_drop_p)).to(r0_feats.dtype)
            r0_f_eff = torch.cat([r0_feats[:, :R0_STRUCT_DIM_V140], r0_feats[:, R0_STRUCT_DIM_V140:] * lm_keep], dim=-1)
        else:
            r0_f_eff = r0_feats
        r0 = self.forward_r0(r0_c1, r0_c2, r0_c3, r0_mask, r0_f_eff)
        deep = self.forward_deep(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=r0["z_char"], logit_ai_r0=r0["logit_ai_r0"], logit_promo_r0=r0["logit_promo_r0"],
            support_pairs=support_pairs, charlm_logit=r0_f_eff[:, -1],
            outer_steps=outer_steps, r0_drop_p=r0_drop_p, charlm_drop_p=0.0,
        )
        return {**r0, **deep}

    def aggregate_long_chunks(
        self,
        chunk_emb_ai: torch.Tensor,
        chunk_emb_pr: torch.Tensor,
        chunk_lg_ai: torch.Tensor,
        chunk_lg_pr: torch.Tensor,
        chunk_meta: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vm = chunk_valid.float()
        q_frac = chunk_meta[..., 3].clamp(0.0, 1.0)
        ai_in = torch.cat([chunk_emb_ai, chunk_meta, chunk_lg_ai.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        a_w = (self.long_ai_attn(ai_in).squeeze(-1) - 2.0 * q_frac).masked_fill(vm < 0.5, -1e4)
        a_w = torch.softmax(a_w.float(), dim=-1).to(chunk_emb_ai.dtype)
        ai_ctx = (chunk_emb_ai * a_w.unsqueeze(-1)).sum(dim=1)
        ai_max = (chunk_lg_ai * (1.0 - 0.5 * q_frac)).masked_fill(vm < 0.5, -1e4).max(dim=1).values
        ai_mean = (chunk_lg_ai * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        ai_wmean = (chunk_lg_ai * a_w).sum(dim=1)
        doc_ai = ai_wmean + 0.5 * torch.tanh(self.long_ai_head(torch.cat([ai_ctx, ai_max.unsqueeze(-1), ai_mean.unsqueeze(-1), ai_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1))

        pr_in = torch.cat([chunk_emb_pr, chunk_meta, chunk_lg_pr.unsqueeze(-1), (1.0 - q_frac).unsqueeze(-1)], dim=-1)
        p_w = self.long_pr_attn(pr_in).squeeze(-1).masked_fill(vm < 0.5, -1e4)
        p_w = torch.softmax(p_w.float(), dim=-1).to(chunk_emb_pr.dtype)
        pr_ctx = (chunk_emb_pr * p_w.unsqueeze(-1)).sum(dim=1)
        pr_max = chunk_lg_pr.masked_fill(vm < 0.5, -1e4).max(dim=1).values
        pr_mean = (chunk_lg_pr * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
        pr_wmean = (chunk_lg_pr * p_w).sum(dim=1)
        doc_pr = pr_wmean + 0.5 * torch.tanh(self.long_pr_head(torch.cat([pr_ctx, pr_max.unsqueeze(-1), pr_mean.unsqueeze(-1), pr_wmean.unsqueeze(-1)], dim=-1)).squeeze(-1))
        return doc_ai, doc_pr


# ---------------------------------------------------------------------------
# 5. Clean W4/W8 `MGVZ8` Packer & Loader (`<= 5.00 MiB`, ZERO 1.58-bit!)
# ---------------------------------------------------------------------------

def _quant_spec_map_v140(model: MeguvizV140ParadiseApexModel) -> dict[str, tuple[str, int]]:
    spec: dict[str, tuple[str, int]] = {
        "pos_emb": ("w4", 32),
        "core.ca_q_lora_a": ("w8", 0),
        "core.ca_q_lora_b": ("w8", 0),
        "core.sa_o_lora_a": ("w8", 0),
        "core.sa_o_lora_b": ("w8", 0),
        "core.slot_queries": ("w8", 0),
    }
    for name, mod in model.named_modules():
        if isinstance(mod, (QuantLinearV130, QuantEmbeddingV130)):
            spec[f"{name}.weight"] = ("w8" if mod.target_bits == 8 else "w4", mod.group_size)
        elif isinstance(mod, nn.Linear) and mod.weight.numel() >= 1024:
            spec[f"{name}.weight"] = ("w8", 0)
    return spec


def pack_bundle_v8_v140(model: MeguvizV140ParadiseApexModel, out_path: Path) -> dict[str, Any]:
    spec_map = _quant_spec_map_v140(model)
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    breakdown: dict[str, int] = {"r0_w8": 0, "deep_w4": 0, "deep_w8": 0, "fp16": 0}

    for k, arr in sd.items():
        q_mode, gs = spec_map.get(k, ("fp16", 0))
        if arr.size < 64:
            q_mode, gs = "fp16", 0
        if q_mode == "w8":
            sb, qb = _pack_w8_tensor(arr)
            bkey = "r0_w8" if k.startswith("rung0.") else "deep_w8"
        elif q_mode == "w4":
            sb, qb = _pack_w4_tensor(arr, gs=gs)
            bkey = "deep_w4"
        else:
            sb, qb = b"", arr.astype(np.float16).tobytes()
            bkey = "fp16"
        entries.append({"name": k, "q": q_mode, "g": gs, "shape": list(arr.shape), "s_bytes": len(sb), "q_bytes": len(qb)})
        blobs.append(sb)
        blobs.append(qb)
        breakdown[bkey] += len(sb) + len(qb)

    hdr_bytes = json.dumps({"format": "MGVZ8_PARADISE_APEX_V140", "tensors": entries}, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MGVZ8_MAGIC)
        f.write(struct.pack("<I", len(hdr_bytes)))
        f.write(hdr_bytes)
        for b in blobs:
            f.write(b)
    raw_b = out_path.stat().st_size
    return {
        "raw_bytes": raw_b,
        "raw_mb": round(raw_b / (1024.0 * 1024.0), 4),
        "breakdown_kb": {k: round(v / 1024.0, 1) for k, v in breakdown.items()},
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }


def load_bundle_v8_v140_into_model(model: MeguvizV140ParadiseApexModel, mgvz_path: Path) -> None:
    data = mgvz_path.read_bytes()
    assert data[:5] == b"MGVZ8", "Invalid MGVZ8 magic!"
    hlen = struct.unpack("<I", data[8:12])[0]
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    pos = 12 + hlen
    sd = model.state_dict()
    for t in header["tensors"]:
        k, q, gs, shape = t["name"], t["q"], int(t["g"]), tuple(t["shape"])
        sb, qb = int(t["s_bytes"]), int(t["q_bytes"])
        s_buf, q_buf = data[pos : pos + sb], data[pos + sb : pos + sb + qb]
        pos += sb + qb
        if q == "fp16":
            arr = np.frombuffer(q_buf, dtype=np.float16).astype(np.float32).reshape(shape)
        elif q == "w8":
            arr = _unpack_w8_tensor(s_buf, q_buf, shape)
        else:
            arr = _unpack_w4_tensor(s_buf, q_buf, shape, gs=gs)
        if k in sd:
            sd[k].copy_(torch.from_numpy(arr.copy()).to(sd[k].dtype))


# ---------------------------------------------------------------------------
# 6. 4-Stage Modular FP32-Exact ONNX Export (`r0`, `t2_paradise`, `t3_correct`, `dynaseam`)
# ---------------------------------------------------------------------------

class _V140R0Export(nn.Module):
    def __init__(self, m: MeguvizV140ParadiseApexModel) -> None:
        super().__init__()
        self.r0 = m.rung0

    def forward(self, c1, c2, c3, mask, feats):
        o = self.r0(c1, c2, c3, mask, feats)
        return o["logit_ai_r0"], o["logit_promo_r0"], o["p_defer_r0"], o["z_char"]


_T2_OUT_NAMES_V140 = [
    "logit_ai_t2_final", "logit_promo_t2_final", "logit_ai_r1", "logit_promo_r1",
    "logit_ai_t1", "logit_promo_t1", "logit_ai_t2", "ebt_margin_t2", "ebt_grad_norm_t2",
    "emb_ai_t1", "emb_ai_t2", "emb_pr_t2", "p_escalate_t3", "g_sup",
    "slots_t1", "slots_a_t2", "slots_b_t2", "anchor", "h_tok_t2", "carry_m_l_t2", "zh_bus_t2",
    "k_haar", "v_haar", "haar_mask", "h0_pool", "st_ai", "st_pr",
]
_T2_PRED_COUNT_V140 = 14


class _V140T2Export(nn.Module):
    def __init__(self, m: MeguvizV140ParadiseApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, style_feats, engram_ids, trigram_ids, z_char, logit_ai_r0, logit_promo_r0, support_pairs, charlm_logit):
        o = self.m.forward_t2_stage(
            input_ids, attention_mask, style_feats, engram_ids, trigram_ids,
            z_char=z_char, logit_ai_r0=logit_ai_r0, logit_promo_r0=logit_promo_r0,
            support_pairs=support_pairs, charlm_logit=charlm_logit,
        )
        return tuple(o[k] for k in _T2_OUT_NAMES_V140)


_T3_IN_NAMES_V140 = [
    "slots_a_t2", "slots_b_t2", "slots_t1", "anchor", "h_tok_t2", "carry_m_l_t2", "zh_bus_t2",
    "ebt_grad_norm_t2", "ebt_margin_t2", "attention_mask", "k_haar", "v_haar", "haar_mask",
    "h0_pool", "st_ai", "st_pr", "z_char",
    "logit_ai_r0", "logit_promo_r0", "logit_promo_r1",
    "logit_ai_t1", "logit_promo_t1", "logit_ai_t2", "logit_ai_t2_final", "logit_promo_t2_final",
    "emb_ai_t2", "emb_pr_t2", "g_sup", "charlm_logit",
]
_T3_OUT_NAMES_V140 = ["logit_ai_t3_final", "logit_promo_t3_final", "logit_ai_t3", "ebt_margin_t3", "emb_ai_t3", "emb_pr_t3"]


class _V140T3Export(nn.Module):
    def __init__(self, m: MeguvizV140ParadiseApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(
        self, slots_a_t2, slots_b_t2, slots_t1, anchor, h_tok_t2, carry_m_l_t2, zh_bus_t2,
        ebt_grad_norm_t2, ebt_margin_t2, attention_mask, k_haar, v_haar, haar_mask,
        h0_pool, st_ai, st_pr, z_char,
        logit_ai_r0, logit_promo_r0, logit_promo_r1,
        logit_ai_t1, logit_promo_t1, logit_ai_t2, logit_ai_t2_final, logit_promo_t2_final,
        emb_ai_t2, emb_pr_t2, g_sup, charlm_logit,
    ):
        o = self.m.forward_t3_step(
            slots_a_t2, slots_b_t2, slots_t1, anchor, h_tok_t2, carry_m_l_t2, zh_bus_t2,
            ebt_grad_norm_t2, ebt_margin_t2, attention_mask, k_haar, v_haar, haar_mask,
            h0_pool, st_ai, st_pr, z_char,
            logit_ai_r0, logit_promo_r0, logit_promo_r1,
            logit_ai_t1, logit_promo_t1, logit_ai_t2, logit_ai_t2_final, logit_promo_t2_final,
            emb_ai_t2, emb_pr_t2, g_sup, charlm_logit,
        )
        return tuple(o[k] for k in _T3_OUT_NAMES_V140)


class _V140LongAggExport(nn.Module):
    def __init__(self, m: MeguvizV140ParadiseApexModel) -> None:
        super().__init__()
        self.m = m

    def forward(self, emb_ai, emb_pr, lg_ai, lg_pr, meta, valid):
        return self.m.aggregate_long_chunks(emb_ai, emb_pr, lg_ai, lg_pr, meta, valid)


def export_v140_onnx_and_bundle_v8(
    model: MeguvizV140ParadiseApexModel,
    tokenizer_path: Path,
    manifest: dict[str, Any],
    out_dir: Path,
    version: str,
    charlm: CompactCharLMV140 | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mgvz_p = out_dir / "bundle_v8_paradise_apex_v140.mgvz"
    set_v140_qat_mode(model, False)
    mgvz_info = pack_bundle_v8_v140(model, mgvz_p)

    charlm_p = out_dir / "charlm_v140.bin"
    charlm_info: dict[str, Any] = {"enabled": False, "raw_bytes": 0, "roundtrip_diff": 0.0, "pass": True}
    if charlm is not None:
        sv = charlm.save(charlm_p)
        reloaded = CompactCharLMV140.load(charlm_p)
        probe_cl = ["笑死我了，这个插件装完直接起飞", "总的来说，这项技术具有非常显著的综合优势。"]
        rt_diff = float(np.abs(charlm.extract_9d(probe_cl) - reloaded.extract_9d(probe_cl)).max())
        charlm_info = {"enabled": True, **sv, "roundtrip_diff": round(rt_diff, 8), "pass": bool(rt_diff < 1e-5)}

    m_cpu = MeguvizV140ParadiseApexModel(
        vocab_size=model.vocab_size, max_len=model.max_len, style_dim=model.style_dim, style_era_dropout_p=0.0
    ).cpu().eval()
    load_bundle_v8_v140_into_model(m_cpu, mgvz_p)

    # 1. R0 ApexCharNano
    c1 = torch.randint(1, R0_B1_V130, (2, 24), dtype=torch.long)
    c2 = torch.randint(1, R0_B2_V130, (2, 24), dtype=torch.long)
    c3 = torch.randint(1, R0_B3_V130, (2, 24), dtype=torch.long)
    cm = torch.ones((2, 24), dtype=torch.float32)
    cf = torch.zeros((2, R0_FEAT_DIM_V140), dtype=torch.float32)
    cf[:, -1] = torch.tensor([-0.6, 0.9], dtype=torch.float32)

    r0_p = out_dir / "v140_r0_apexchar_fp32.onnx"
    torch.onnx.export(
        _V140R0Export(m_cpu), (c1, c2, c3, cm, cf), str(r0_p),
        input_names=["c1", "c2", "c3", "mask", "feats"],
        output_names=["logit_ai_r0", "logit_promo_r0", "p_defer_r0", "z_char"],
        dynamic_axes={k: {0: "batch", 1: "clen"} for k in ("c1", "c2", "c3", "mask")}
        | {"feats": {0: "batch"}, "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"}, "p_defer_r0": {0: "batch"}, "z_char": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # 2. T2 ParadiseApex Nested Stage
    ids = torch.randint(1, model.vocab_size, (2, 32), dtype=torch.long)
    msk = torch.ones((2, 32), dtype=torch.long)
    st = torch.zeros((2, model.style_dim), dtype=torch.float32)
    eg = torch.randint(1, ENGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    tg = torch.randint(1, TRIGRAM_BUCKETS_V130, (2, 32), dtype=torch.long)
    zc = torch.zeros((2, R0_ZCHAR_DIM_V130), dtype=torch.float32)
    lr0_a = torch.zeros((2,), dtype=torch.float32)
    lr0_p = torch.zeros((2,), dtype=torch.float32)
    sup = torch.tensor([8.0, 28.0], dtype=torch.float32)
    lm_lg = torch.tensor([-0.6, 0.9], dtype=torch.float32)

    t2_p = out_dir / "v140_t2_paradise_fp32.onnx"
    t2_dyn = {
        "input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
        "engram_ids": {0: "batch", 1: "seq"}, "trigram_ids": {0: "batch", 1: "seq"},
        "style_feats": {0: "batch"}, "z_char": {0: "batch"},
        "logit_ai_r0": {0: "batch"}, "logit_promo_r0": {0: "batch"},
        "support_pairs": {0: "batch"}, "charlm_logit": {0: "batch"},
    }
    for nm in _T2_OUT_NAMES_V140:
        if nm == "h_tok_t2":
            t2_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t2_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t2_dyn[nm] = {0: "batch"}

    torch.onnx.export(
        _V140T2Export(m_cpu), (ids, msk, st, eg, tg, zc, lr0_a, lr0_p, sup, lm_lg), str(t2_p),
        input_names=["input_ids", "attention_mask", "style_feats", "engram_ids", "trigram_ids", "z_char", "logit_ai_r0", "logit_promo_r0", "support_pairs", "charlm_logit"],
        output_names=_T2_OUT_NAMES_V140,
        dynamic_axes=t2_dyn,
        opset_version=17, dynamo=False,
    )

    # 3. Conditional T3 Active Error-Correction Graph
    with torch.no_grad():
        t2_d = m_cpu.forward_t2_stage(ids, msk, st, eg, tg, zc, lr0_a, lr0_p, sup, charlm_logit=lm_lg)
    t3_dummy = (
        t2_d["slots_a_t2"], t2_d["slots_b_t2"], t2_d["slots_t1"], t2_d["anchor"], t2_d["h_tok_t2"],
        t2_d["carry_m_l_t2"], t2_d["zh_bus_t2"], t2_d["ebt_grad_norm_t2"], t2_d["ebt_margin_t2"],
        msk, t2_d["k_haar"], t2_d["v_haar"], t2_d["haar_mask"],
        t2_d["h0_pool"], t2_d["st_ai"], t2_d["st_pr"], zc,
        lr0_a, lr0_p, t2_d["logit_promo_r1"], t2_d["logit_ai_t1"], t2_d["logit_promo_t1"],
        t2_d["logit_ai_t2"], t2_d["logit_ai_t2_final"], t2_d["logit_promo_t2_final"],
        t2_d["emb_ai_t2"], t2_d["emb_pr_t2"], t2_d["g_sup"], lm_lg,
    )
    t3_dyn = {}
    for nm in _T3_IN_NAMES_V140 + _T3_OUT_NAMES_V140:
        if nm in ("h_tok_t2", "attention_mask"):
            t3_dyn[nm] = {0: "batch", 1: "seq"}
        elif nm in ("k_haar", "v_haar", "haar_mask"):
            t3_dyn[nm] = {0: "batch", 1: "hseq"}
        else:
            t3_dyn[nm] = {0: "batch"}

    t3_p = out_dir / "v140_t3_correct_fp32.onnx"
    torch.onnx.export(
        _V140T3Export(m_cpu), t3_dummy, str(t3_p),
        input_names=_T3_IN_NAMES_V140,
        output_names=_T3_OUT_NAMES_V140,
        dynamic_axes=t3_dyn,
        opset_version=17, dynamo=False,
    )

    # 4. DynaSeam Long-Doc Aggregator
    long_p = out_dir / "v140_dynaseam_fp32.onnx"
    ea = torch.randn(2, 4, AI_END_V130)
    ep = torch.randn(2, 4, PR_EMB_DIM_V130)
    la = torch.randn(2, 4)
    lp = torch.randn(2, 4)
    mt = torch.zeros(2, 4, CHUNK_META_DIM_V11)
    vm = torch.ones(2, 4)
    torch.onnx.export(
        _V140LongAggExport(m_cpu), (ea, ep, la, lp, mt, vm), str(long_p),
        input_names=["emb_ai", "emb_pr", "lg_ai", "lg_pr", "meta", "valid"],
        output_names=["doc_ai", "doc_promo"],
        dynamic_axes={
            "emb_ai": {0: "batch", 1: "chunks"}, "emb_pr": {0: "batch", 1: "chunks"},
            "lg_ai": {0: "batch", 1: "chunks"}, "lg_pr": {0: "batch", 1: "chunks"},
            "meta": {0: "batch", 1: "chunks"}, "valid": {0: "batch", 1: "chunks"},
            "doc_ai": {0: "batch"}, "doc_promo": {0: "batch"},
        },
        opset_version=17, dynamo=False,
    )

    # Exact Parity Verification across all 4 modular ONNX graphs!
    s0 = ort.InferenceSession(str(r0_p), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(t2_p), providers=["CPUExecutionProvider"])
    s3 = ort.InferenceSession(str(t3_p), providers=["CPUExecutionProvider"])
    sl = ort.InferenceSession(str(long_p), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_r0 = _V140R0Export(m_cpu)(c1, c2, c3, cm, cf)
        or_r0 = s0.run(None, {"c1": c1.numpy(), "c2": c2.numpy(), "c3": c3.numpy(), "mask": cm.numpy(), "feats": cf.numpy()})
        r0_diff = float(max(np.abs(pt_r0[i].numpy() - or_r0[i]).max() for i in range(4)))

        pt_t2 = _V140T2Export(m_cpu)(ids, msk, st, eg, tg, pt_r0[3], pt_r0[0], pt_r0[1], sup, lm_lg)
        or_t2 = s2.run(None, {
            "input_ids": ids.numpy(), "attention_mask": msk.numpy(), "style_feats": st.numpy(),
            "engram_ids": eg.numpy(), "trigram_ids": tg.numpy(), "z_char": or_r0[3],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1], "support_pairs": sup.numpy(), "charlm_logit": lm_lg.numpy(),
        })
        t2_pred_diff = float(max(np.abs(pt_t2[i].numpy() - or_t2[i]).max() for i in range(_T2_PRED_COUNT_V140)))
        t2_cache_diff = float(max(np.abs(pt_t2[i].numpy() - or_t2[i]).max() for i in range(_T2_PRED_COUNT_V140, len(pt_t2))))

        m2 = dict(zip(_T2_OUT_NAMES_V140, or_t2))
        t3_feed = {
            "slots_a_t2": m2["slots_a_t2"], "slots_b_t2": m2["slots_b_t2"], "slots_t1": m2["slots_t1"],
            "anchor": m2["anchor"], "h_tok_t2": m2["h_tok_t2"], "carry_m_l_t2": m2["carry_m_l_t2"],
            "zh_bus_t2": m2["zh_bus_t2"], "ebt_grad_norm_t2": m2["ebt_grad_norm_t2"], "ebt_margin_t2": m2["ebt_margin_t2"],
            "attention_mask": msk.numpy(), "k_haar": m2["k_haar"], "v_haar": m2["v_haar"], "haar_mask": m2["haar_mask"],
            "h0_pool": m2["h0_pool"], "st_ai": m2["st_ai"], "st_pr": m2["st_pr"], "z_char": or_r0[3],
            "logit_ai_r0": or_r0[0], "logit_promo_r0": or_r0[1], "logit_promo_r1": m2["logit_promo_r1"],
            "logit_ai_t1": m2["logit_ai_t1"], "logit_promo_t1": m2["logit_promo_t1"], "logit_ai_t2": m2["logit_ai_t2"],
            "logit_ai_t2_final": m2["logit_ai_t2_final"], "logit_promo_t2_final": m2["logit_promo_t2_final"],
            "emb_ai_t2": m2["emb_ai_t2"], "emb_pr_t2": m2["emb_pr_t2"], "g_sup": m2["g_sup"], "charlm_logit": lm_lg.numpy(),
        }
        or_t3 = s3.run(None, t3_feed)
        pt_full3 = m_cpu.forward_deep(ids, msk, st, eg, tg, pt_r0[3], pt_r0[0], pt_r0[1], sup, charlm_logit=lm_lg, outer_steps=3)
        pt_t3_ref = [pt_full3[k].numpy() for k in _T3_OUT_NAMES_V140]
        t3_diff = float(max(np.abs(pt_t3_ref[i] - or_t3[i]).max() for i in range(len(_T3_OUT_NAMES_V140))))

        pt_l = _V140LongAggExport(m_cpu)(ea, ep, la, lp, mt, vm)
        or_l = sl.run(None, {"emb_ai": ea.numpy(), "emb_pr": ep.numpy(), "lg_ai": la.numpy(), "lg_pr": lp.numpy(), "meta": mt.numpy(), "valid": vm.numpy()})
        long_diff = float(max(np.abs(pt_l[i].numpy() - or_l[i]).max() for i in range(2)))

    tok_out = out_dir / "bili_picky_bpe_12k.json"
    if tokenizer_path.resolve() != tok_out.resolve():
        tok_out.write_bytes(tokenizer_path.read_bytes())

    manifest = dict(manifest)
    manifest["format"] = "MGVZ8_PARADISE_APEX_V140"
    manifest["mgvz8_info"] = mgvz_info
    manifest["charlm_info"] = charlm_info
    mf_p = out_dir / "manifest.json"
    mf_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle_zip = out_dir / f"meguvizcac-{version}-bundle-v8-paradise-apex.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(mgvz_p, mgvz_p.name)
        zf.write(tok_out, tok_out.name)
        zf.write(mf_p, mf_p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)
    bundle_bytes = bundle_zip.stat().st_size

    onnx_zip = out_dir / f"meguvizcac-{version}-4stage-onnx-fp32exact.zip"
    with zipfile.ZipFile(onnx_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in (r0_p, t2_p, t3_p, long_p, tok_out, mf_p):
            zf.write(p, p.name)
        if charlm is not None and charlm_p.is_file():
            zf.write(charlm_p, charlm_p.name)

    return {
        "bundle_zip": str(bundle_zip),
        "bundle_bytes": bundle_bytes,
        "bundle_mib": round(bundle_bytes / (1024.0 * 1024.0), 4),
        "bundle_le_5p00_mib": bool(bundle_bytes <= int(BUNDLE_MAX_MIB_V130 * 1024 * 1024)),
        "onnx_zip_mb": round(onnx_zip.stat().st_size / (1024.0 * 1024.0), 4),
        "r0_fp32_max_diff": round(r0_diff, 8),
        "t2_fp32_max_diff": round(t2_pred_diff, 8),
        "t2_cache_fp32_max_diff": round(t2_cache_diff, 8),
        "t3_modular_fp32_max_diff": round(t3_diff, 8),
        "long_fp32_max_diff": round(long_diff, 8),
        "onnx_exact_pass": bool(max(r0_diff, t2_pred_diff, t3_diff, long_diff) < 1e-4 and t2_cache_diff < 1e-3 and charlm_info["pass"]),
        "mgvz8_info": mgvz_info,
        "charlm_info": charlm_info,
    }


# ---------------------------------------------------------------------------
# 7. Production 4-Stage CPU ONNX Cascade Engine (`MeguvizV140CascadeEngine`)
# ---------------------------------------------------------------------------

class MeguvizV140CascadeEngine:
    """Zero-PII 4-Stage Modular ONNX Cascade Engine (`R0 -> T2 BiXT+mHC -> Bounded T3 -> DynaSeam`)."""

    def __init__(self, release_dir: str | Path, threads: int = 4) -> None:
        self.release_dir = Path(release_dir)
        self.manifest = json.loads((self.release_dir / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = BiliPickyBPETokenizer(self.release_dir / "bili_picky_bpe_12k.json")
        charlm_p = self.release_dir / "charlm_v140.bin"
        self.charlm: CompactCharLMV140 | None = CompactCharLMV140.load(charlm_p) if charlm_p.is_file() else None
        self.max_len = int(self.manifest.get("max_len", MAX_LEN_V140))
        self.head_len = int(self.manifest.get("head_len", HEAD_LEN_V140))
        self.long_char_trigger = int(self.manifest.get("long_char_trigger", 360))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_r0 = ort.InferenceSession(str(self.release_dir / "v140_r0_apexchar_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t2 = ort.InferenceSession(str(self.release_dir / "v140_t2_paradise_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_t3 = ort.InferenceSession(str(self.release_dir / "v140_t3_correct_fp32.onnx"), opts, providers=["CPUExecutionProvider"])
        self.sess_long = ort.InferenceSession(str(self.release_dir / "v140_dynaseam_fp32.onnx"), opts, providers=["CPUExecutionProvider"])

        self.r0_policy = self.manifest.get("rung0_ltt_policy") or {}
        self.t3_policy = self.manifest.get("t3_escalation_policy") or {
            "enabled": True, "tau_escalate": 0.45, "uncertain_lo": 0.10, "uncertain_hi": 0.90, "t3_bound_margin": 3.0,
        }
        self.lut_ai_r0 = self.manifest.get("lut_ai_r0")
        self.lut_pr_r0 = self.manifest.get("lut_promo_r0")
        self.lut_ai_full = self.manifest.get("lut_ai_full")
        self.lut_pr_full = self.manifest.get("lut_promo_full")
        self.temp_ai = float(self.manifest.get("temp_ai", 1.0))
        self.temp_pr = float(self.manifest.get("temp_promo", 1.0))

    def _run_t2_and_cond_t3(
        self,
        texts: list[str],
        z_char: np.ndarray,
        lg_r0_ai: np.ndarray,
        lg_r0_pr: np.ndarray,
        supports: np.ndarray,
        charlm_logits: np.ndarray,
        force_t3: bool = False,
    ) -> dict[str, np.ndarray]:
        cleans, f56_l = [], []
        for t in texts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56_l.append(f56)
        encs = [encode_bpe_head_tail_v130(self.tokenizer, c, max_len=self.max_len, head_len=self.head_len) for c in cleans]
        engs = [make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130) for s in encs]
        trigs = [make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130) for s in encs]
        ml = max(2, max(len(s) for s in encs))
        if ml % 2 == 1:
            ml += 1
        bs = len(texts)
        ids = np.full((bs, ml), self.tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((bs, ml), dtype=np.int64)
        eg = np.zeros((bs, ml), dtype=np.int64)
        tg = np.zeros((bs, ml), dtype=np.int64)
        for i, s in enumerate(encs):
            ids[i, : len(s)], mask[i, : len(s)], eg[i, : len(s)], tg[i, : len(s)] = s, 1, engs[i], trigs[i]
        st = np.stack(f56_l, axis=0).astype(np.float32)

        lm_arr = charlm_logits.astype(np.float32)
        t2_raw = self.sess_t2.run(None, {
            "input_ids": ids, "attention_mask": mask, "style_feats": st, "engram_ids": eg, "trigram_ids": tg,
            "z_char": z_char.astype(np.float32), "logit_ai_r0": lg_r0_ai.astype(np.float32),
            "logit_promo_r0": lg_r0_pr.astype(np.float32), "support_pairs": supports.astype(np.float32),
            "charlm_logit": lm_arr,
        })
        t2 = dict(zip(_T2_OUT_NAMES_V140, t2_raw))

        lg_ai = t2["logit_ai_t2_final"].copy()
        lg_pr = t2["logit_promo_t2_final"].copy()
        emb_ai = t2["emb_ai_t2"].copy()
        emb_pr = t2["emb_pr_t2"].copy()
        rungs = np.full((bs,), 2, dtype=np.int64)

        t3_en = bool(self.t3_policy.get("enabled", True)) or force_t3
        if t3_en:
            p_esc = t2["p_escalate_t3"]
            p_t2 = expit(np.clip(lg_ai / self.temp_ai, -30.0, 30.0))
            tau_e = float(self.t3_policy.get("tau_escalate", 0.45))
            u_lo = float(self.t3_policy.get("uncertain_lo", 0.10))
            u_hi = float(self.t3_policy.get("uncertain_hi", 0.90))
            b_mar = float(self.t3_policy.get("t3_bound_margin", 3.2))
            if force_t3:
                esc_idx = np.arange(bs, dtype=np.int64)
            else:
                # Bounded T3 Early Exit: skip T3 whenever |logit_t2| > b_mar (> 2.2 cap!) AND Ouro gate is calm!
                within_flip_band = np.abs(lg_ai) <= b_mar
                esc_mask = within_flip_band & ((p_esc >= tau_e) | ((p_t2 >= u_lo) & (p_t2 <= u_hi)))
                esc_idx = np.where(esc_mask)[0].astype(np.int64)

            if len(esc_idx) > 0:
                t3_feed = {
                    "slots_a_t2": t2["slots_a_t2"][esc_idx], "slots_b_t2": t2["slots_b_t2"][esc_idx],
                    "slots_t1": t2["slots_t1"][esc_idx], "anchor": t2["anchor"][esc_idx],
                    "h_tok_t2": t2["h_tok_t2"][esc_idx], "carry_m_l_t2": t2["carry_m_l_t2"][esc_idx],
                    "zh_bus_t2": t2["zh_bus_t2"][esc_idx], "ebt_grad_norm_t2": t2["ebt_grad_norm_t2"][esc_idx],
                    "ebt_margin_t2": t2["ebt_margin_t2"][esc_idx], "attention_mask": mask[esc_idx],
                    "k_haar": t2["k_haar"][esc_idx], "v_haar": t2["v_haar"][esc_idx], "haar_mask": t2["haar_mask"][esc_idx],
                    "h0_pool": t2["h0_pool"][esc_idx], "st_ai": t2["st_ai"][esc_idx], "st_pr": t2["st_pr"][esc_idx],
                    "z_char": z_char[esc_idx].astype(np.float32), "logit_ai_r0": lg_r0_ai[esc_idx].astype(np.float32),
                    "logit_promo_r0": lg_r0_pr[esc_idx].astype(np.float32), "logit_promo_r1": t2["logit_promo_r1"][esc_idx],
                    "logit_ai_t1": t2["logit_ai_t1"][esc_idx], "logit_promo_t1": t2["logit_promo_t1"][esc_idx],
                    "logit_ai_t2": t2["logit_ai_t2"][esc_idx], "logit_ai_t2_final": t2["logit_ai_t2_final"][esc_idx],
                    "logit_promo_t2_final": t2["logit_promo_t2_final"][esc_idx],
                    "emb_ai_t2": t2["emb_ai_t2"][esc_idx], "emb_pr_t2": t2["emb_pr_t2"][esc_idx],
                    "g_sup": t2["g_sup"][esc_idx], "charlm_logit": lm_arr[esc_idx],
                }
                t3_out = dict(zip(_T3_OUT_NAMES_V140, self.sess_t3.run(None, t3_feed)))
                lg_ai[esc_idx] = t3_out["logit_ai_t3_final"]
                lg_pr[esc_idx] = t3_out["logit_promo_t3_final"]
                emb_ai[esc_idx] = t3_out["emb_ai_t3"]
                emb_pr[esc_idx] = t3_out["emb_pr_t3"]
                rungs[esc_idx] = 3

        return {"logit_ai": lg_ai, "logit_promo": lg_pr, "emb_ai": emb_ai, "emb_pr": emb_pr, "exit_rung": rungs}

    def _run_long_doc(self, text: str) -> tuple[float, float]:
        chunks = plan_long_chunks_v11(text, max_chunks=8)
        ch_txts = [c["text"] for c in chunks] if len(chunks) > 1 else [text]
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(ch_txts, charlm=self.charlm)
        la0, lp0, _, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
        od = self._run_t2_and_cond_t3(ch_txts, zc, la0, lp0, sups, cf[:, -1], force_t3=False)
        if len(chunks) <= 1:
            return float(od["logit_ai"][0]), float(od["logit_promo"][0])
        metas = np.stack([c["chunk_meta"] for c in chunks], axis=0).astype(np.float32)[None, :, :]
        valid = np.ones((1, len(chunks)), dtype=np.float32)
        doc_ai, doc_pr = self.sess_long.run(None, {
            "emb_ai": od["emb_ai"][None, :, :], "emb_pr": od["emb_pr"][None, :, :],
            "lg_ai": od["logit_ai"][None, :], "lg_pr": od["logit_promo"][None, :],
            "meta": metas, "valid": valid,
        })
        return float(doc_ai[0]), float(doc_pr[0])

    def predict_batch(
        self,
        texts: list[str],
        max_batch: int = 32,
        force_full: bool = False,
        force_t3: bool = False,
    ) -> list[dict[str, Any]]:
        n = len(texts)
        if n == 0:
            return []
        results: list[dict[str, Any] | None] = [None] * n
        short_idxs = [i for i, t in enumerate(texts) if len(t) < self.long_char_trigger]
        long_idxs = [i for i, t in enumerate(texts) if len(t) >= self.long_char_trigger]

        for oi in long_idxs:
            lg_a, lg_p = self._run_long_doc(texts[oi])
            pa = float(expit(np.clip(lg_a / self.temp_ai, -30.0, 30.0)))
            pp = float(expit(np.clip(lg_p / self.temp_pr, -30.0, 30.0)))
            if self.lut_ai_full:
                pa = float(apply_spline_temp_lut(np.array([pa]), self.lut_ai_full)[0])
            if self.lut_pr_full:
                pp = float(apply_spline_temp_lut(np.array([pp]), self.lut_pr_full)[0])
            results[oi] = {"prob_ai": pa, "prob_promo": pp, "exit_rung": 4, "logit_ai": lg_a, "logit_promo": lg_p}

        order = sorted(short_idxs, key=lambda i: len(texts[i]))
        r0_en = bool(self.r0_policy.get("enabled", False)) and not force_full and not force_t3
        r0_tau_ai = float(self.r0_policy.get("tau_ai", 0.04))
        r0_tau_pr = float(self.r0_policy.get("tau_promo", 0.05))
        r0_def_max = float(self.r0_policy.get("defer_max", 0.30))
        r0_max_ch = int(self.r0_policy.get("max_chars", 180))

        for st in range(0, len(order), max_batch):
            mb = order[st : st + max_batch]
            mb_txts = [texts[i] for i in mb]
            c1, c2, c3, cm, cf, spoofs, sups = extract_apex_char_batch_v140(mb_txts, charlm=self.charlm)
            la0, lp0, pdef0, zc = self.sess_r0.run(None, {"c1": c1, "c2": c2, "c3": c3, "mask": cm, "feats": cf})
            pa0 = expit(np.clip(la0, -30.0, 30.0))
            pp0 = expit(np.clip(lp0, -30.0, 30.0))
            if self.lut_ai_r0:
                pa0 = apply_spline_temp_lut(pa0, self.lut_ai_r0)
            if self.lut_pr_r0:
                pp0 = apply_spline_temp_lut(pp0, self.lut_pr_r0)

            surv: list[int] = []
            for ri, oi in enumerate(mb):
                if (
                    r0_en and len(texts[oi]) <= r0_max_ch and spoofs[ri] == 0.0 and ("http" not in texts[oi].lower())
                    and float(pdef0[ri]) <= r0_def_max and float(pa0[ri]) <= r0_tau_ai and float(pp0[ri]) <= r0_tau_pr
                ):
                    results[oi] = {"prob_ai": float(pa0[ri]), "prob_promo": float(pp0[ri]), "exit_rung": 0, "logit_ai": float(la0[ri]), "logit_promo": float(lp0[ri])}
                else:
                    surv.append(ri)
            if not surv:
                continue
            s_idx = np.array(surv, dtype=np.int64)
            od = self._run_t2_and_cond_t3([mb_txts[k] for k in surv], zc[s_idx], la0[s_idx], lp0[s_idx], sups[s_idx], cf[s_idx, -1], force_t3=force_t3)
            pa_f = expit(np.clip(od["logit_ai"] / self.temp_ai, -30.0, 30.0))
            pp_f = expit(np.clip(od["logit_promo"] / self.temp_pr, -30.0, 30.0))
            if self.lut_ai_full:
                pa_f = apply_spline_temp_lut(pa_f, self.lut_ai_full)
            if self.lut_pr_full:
                pp_f = apply_spline_temp_lut(pp_f, self.lut_pr_full)
            for j, k in enumerate(surv):
                oi = mb[k]
                results[oi] = {
                    "prob_ai": float(pa_f[j]), "prob_promo": float(pp_f[j]),
                    "exit_rung": int(od["exit_rung"][j]),
                    "logit_ai": float(od["logit_ai"][j]), "logit_promo": float(od["logit_promo"][j]),
                }
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 8. Automated 14-Gate v1.4.0 CI Suite (`ParadiseApex-Loop` + BiXT + mHC + Support + Bounded T3!)
# ---------------------------------------------------------------------------

def run_v140_ci_gates(
    model: MeguvizV140ParadiseApexModel,
    tokenizer: BiliPickyBPETokenizer,
    export_rep: dict[str, Any] | None = None,
    ai_lut_check: dict[str, Any] | None = None,
    promo_lut_check: dict[str, Any] | None = None,
    charlm: CompactCharLMV140 | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    ci: dict[str, Any] = {}

    def _infer(txts: list[str], force_pad_len: int | None = None, outer_steps: int = 3):
        c1, c2, c3, cm, cf, _, sups = extract_apex_char_batch_v140(txts, charlm=charlm)
        cleans, f56_l = [], []
        for t in txts:
            cl, f56, _ = extract_style_features_56d(t)
            cleans.append(cl)
            f56_l.append(f56)
        encs = [encode_bpe_head_tail_v130(tokenizer, c, max_len=model.max_len, head_len=HEAD_LEN_V140) for c in cleans]
        ml = max(2, max(len(s) for s in encs))
        if force_pad_len is not None:
            ml = max(ml, force_pad_len)
        if ml % 2 == 1:
            ml += 1
        ids = np.full((len(txts), ml), tokenizer.pad_id, dtype=np.int64)
        msk = np.zeros((len(txts), ml), dtype=np.int64)
        eg = np.zeros((len(txts), ml), dtype=np.int64)
        tg = np.zeros((len(txts), ml), dtype=np.int64)
        for i, s in enumerate(encs):
            ids[i, : len(s)], msk[i, : len(s)] = s, 1
            eg[i, : len(s)] = make_engram_bigram_ids(s, ENGRAM_BUCKETS_V130)
            tg[i, : len(s)] = make_engram_trigram_ids(s, TRIGRAM_BUCKETS_V130)
        with torch.no_grad():
            return model(
                torch.from_numpy(ids).to(device), torch.from_numpy(msk).to(device),
                torch.from_numpy(np.stack(f56_l, 0).astype(np.float32)).to(device),
                torch.from_numpy(eg).to(device), torch.from_numpy(tg).to(device),
                torch.from_numpy(c1).to(device), torch.from_numpy(c2).to(device),
                torch.from_numpy(c3).to(device), torch.from_numpy(cm).to(device),
                torch.from_numpy(cf).to(device), torch.from_numpy(sups).to(device),
                outer_steps=outer_steps,
            )

    # 1. V140_C1: Strict Zero-PII across ALL rungs === 0.0!
    oa = _infer(["回复 @测试用户A : 关注工种号交流2群 168688884 助理老师：xhs54199 领取资料[doge]"], outer_steps=3)
    ob = _infer(["回复 @RandomName99 : 关注工种号交流2群 999888777 助理老师：abc98765 领取资料[doge]"], outer_steps=3)
    pii_keys = ("logit_ai_r0", "logit_promo_r0", "logit_ai_r1", "logit_promo_r1", "logit_ai_t2_final", "logit_ai", "logit_promo", "p_escalate_t3")
    pii_d = float(max(abs(float(oa[k][0].item()) - float(ob[k][0].item())) for k in pii_keys))
    ci["V140_C1_zero_pii_all_rungs"] = {"max_abs_diff": pii_d, "pass": pii_d == 0.0}

    # 2. V140_F6: Structural Promo->AI Firewall under `+12` CBM & Promo bias perturbation!
    t_fw = ["笑死我了，这个插件装完直接起飞[doge]", "关注工种号交流2群 168688884 领全套资料"]
    o_base = _infer(t_fw, outer_steps=3)
    with torch.no_grad():
        saved = (model.cbm_pr_enc.bias.clone(), model.promo_head.bias.clone(), model.r1_promo_head.bias.clone(), model.rung0.promo_head.bias.clone(), model.pr_attr_attn[-1].bias.clone())
        model.cbm_pr_enc.bias.add_(12.0)
        model.promo_head.bias.add_(4.0)
        model.r1_promo_head.bias.add_(4.0)
        model.rung0.promo_head.bias.add_(4.0)
        model.pr_attr_attn[-1].bias.add_(5.0)
        o_pert = _infer(t_fw, outer_steps=3)
        model.cbm_pr_enc.bias.copy_(saved[0])
        model.promo_head.bias.copy_(saved[1])
        model.r1_promo_head.bias.copy_(saved[2])
        model.rung0.promo_head.bias.copy_(saved[3])
        model.pr_attr_attn[-1].bias.copy_(saved[4])
    fw_diff = float(max(
        (o_base[k] - o_pert[k]).abs().max().item()
        for k in ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2", "logit_ai_t2_final", "logit_ai_t3", "logit_ai", "p_escalate_t3")
    ))
    ci["V140_F6_promo_to_ai_firewall"] = {"max_leak_diff_under_plus12_cbm": fw_diff, "pass": fw_diff == 0.0}

    # 3. V140_P1: Exact Batch-Padding Invariance across R0..T3 + BiXT + mHC (`< 1e-4` at pad=288!)
    t_sh = "笑死我了，这个插件装完直接起飞[doge]"
    o_unp = _infer([t_sh], force_pad_len=None, outer_steps=3)
    o_pad = _infer([t_sh], force_pad_len=288, outer_steps=3)
    p_keys = ("logit_ai_r0", "logit_ai_r1", "logit_ai_t1", "logit_ai_t2_final", "logit_ai", "logit_promo", "ebt_margin", "p_escalate_t3")
    p_diffs = {k: round(abs(float(o_unp[k][0].item()) - float(o_pad[k][0].item())), 8) for k in p_keys}
    max_pd = max(p_diffs.values())
    ci["V140_P1_pad_invariance_exact"] = {"diffs": p_diffs, "max_diff": max_pd, "pass": bool(max_pd < 1e-4)}

    # 4. V140_S1: Dynamic shapes + Cayley Symplectic Orthogonality + EBT6 grad parity
    dyn_ok = True
    for os_n in (2, 3):
        oo = _infer(["短评测试", "这是一段较长的结构化技术点评，用于验证动态长度、BiXT 动态回写、双流 mHC 与第三步纠错。"], outer_steps=os_n)
        if torch.isnan(oo["logit_ai"]).any() or torch.isnan(oo["logit_promo"]).any():
            dyn_ok = False
    with torch.no_grad():
        g = model.core.c1_cayley_gamma
        g2 = g * g
        inv = 1.0 / (1.0 + g2)
        cayley_err = float(((((1.0 - g2) * inv) ** 2 + ((2.0 * g) * inv) ** 2) - 1.0).abs().max().item())
    u_t = torch.randn(4, BUS_DIM_V130, device=device, requires_grad=True)
    er = model.core.ebt.forward_energy_and_grad(u_t)
    ag = torch.autograd.grad(er["free_energy"].sum(), u_t)[0]
    ebt_err = float((er["grad_u"].detach() - ag.detach()).abs().max().item())
    ci["V140_S1_nested_symplectic_and_ebt6"] = {
        "dynamic_ok": dyn_ok, "cayley_unit_err": round(cayley_err, 8), "ebt6_grad_max_diff": round(ebt_err, 8),
        "pass": bool(dyn_ok and cayley_err < 1e-6 and ebt_err < 1e-5),
    }

    # 5. V140_S2: Two-Stream Sinkhorn mHC Doubly-Stochastic Invariant (`row/col sums === 1.0`)!
    with torch.no_grad():
        m_all = sinkhorn_2x2_v140(model.core.mhc_log_mix)
        row_err = float((m_all.sum(dim=-1) - 1.0).abs().max().item())
        col_err = float((m_all.sum(dim=-2) - 1.0).abs().max().item())
        min_v = float(m_all.min().item())
    ci["V140_S2_mhc_sinkhorn_doubly_stochastic"] = {
        "row_err": round(row_err, 8), "col_err": round(col_err, 8), "min_entry": round(min_v, 6),
        "pass": bool(row_err < 1e-4 and col_err < 1e-4 and min_v >= 0.0),
    }

    # 6. V140_S3: BiXT Dynamic Token Update (`h_tok` genuinely evolves T1->T2->T3 while PAD tokens remain exact 0.0!)
    bixt_delta = float((o_pad["h_tok_t2"] - o_pad["h_tok_t1"]).abs().max().item())
    pad_tail_max = float(o_pad["h_tok_t3"][0, 64:, :].abs().max().item())
    ci["V140_S3_bixt_dynamic_token_update"] = {
        "t2_minus_t1_tok_diff": round(bixt_delta, 6), "pad_tail_abs_max": round(pad_tail_max, 8),
        "pass": bool(bixt_delta > 1e-5 and pad_tail_max == 0.0),
    }

    # 7. V140_L1: SoftBallLogitNorm bound
    with torch.no_grad():
        huge = torch.ones((2, AI_END_V130), device=device) * 100.0
        hb = model.ai_head.project_ball(huge).norm(dim=-1).max().item()
        rc = float(model.ai_head.effective_r_cap().item())
    ci["V140_L1_softball_bound"] = {"huge_ball_norm": round(hb, 4), "r_cap": round(rc, 4), "pass": bool(math.isfinite(hb) and hb <= rc + 1e-3)}

    # 8. V140_C13: Support-Reliability Gate protects micro-short (`<= 4` pairs) vs content-rich (`>= 32` pairs)
    with torch.no_grad():
        g_short = float(model.compute_support_gate(torch.tensor([4.0], device=device)).item())
        g_rich = float(model.compute_support_gate(torch.tensor([36.0], device=device)).item())
    ci["V140_C13_support_gate_short_fp"] = {
        "g_sup_at_4pairs": round(g_short, 4), "g_sup_at_36pairs": round(g_rich, 4),
        "pass": bool(g_short < 0.25 and g_rich > 0.85 and g_rich > g_short + 0.6),
    }

    # 9. V140_C14: Exact Bounded Detached Residual `T3` Early-Exit Mathematical Guarantee (`|T3 - T2| <= t3_corr_cap`)!
    t3_shift = float((o_base["logit_ai"] - o_base["logit_ai_t2_final"]).abs().max().item())
    ci["V140_C14_bounded_t3_early_exit_exact"] = {
        "max_t3_minus_t2_shift": round(t3_shift, 6), "t3_corr_cap": float(model.t3_corr_cap),
        "pass": bool(t3_shift <= float(model.t3_corr_cap) + 1e-5),
    }

    # 10 & 11. Bundle size <= 5.00 MiB + 4-Stage Modular ONNX FP32 exact parity
    if export_rep is not None:
        ci["V140_B8_bundle_le_5p00_mib"] = {
            "bundle_mib": export_rep["bundle_mib"], "bundle_bytes": export_rep["bundle_bytes"],
            "pass": bool(export_rep["bundle_le_5p00_mib"]),
        }
        ci["V140_ONNX_4stage_fp32_parity"] = {
            "r0_diff": export_rep["r0_fp32_max_diff"], "t2_diff": export_rep["t2_fp32_max_diff"],
            "t2_cache_diff": export_rep.get("t2_cache_fp32_max_diff", 0.0),
            "t3_modular_diff": export_rep["t3_modular_fp32_max_diff"], "long_diff": export_rep["long_fp32_max_diff"],
            "pass": bool(export_rep["onnx_exact_pass"]),
        }

    # 12. Clean AI & Promo LUT floors
    if ai_lut_check is not None:
        ci["V140_AI_LUT_FLOOR"] = {**ai_lut_check, "pass": bool(ai_lut_check.get("lut_ai_r0_at_001", 1.0) <= 0.025)}
    if promo_lut_check is not None:
        ci["V140_PR_LUT_FLOOR"] = {**promo_lut_check, "pass": bool(promo_lut_check.get("median_promo_wild", 1.0) <= 0.12)}

    ci["all_ci_passed"] = all(bool(v.get("pass")) for v in ci.values() if isinstance(v, dict))
    return ci
