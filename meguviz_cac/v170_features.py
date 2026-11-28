"""MeguvizCAC v1.7.0 Sidecar Features (`M6 P_H^(1..4)` + `M7` Surface & Unicode Structural Residual).

Locked interface contract:
  - `M6_PH_DIM_V170 = 16`
  - `M7_SURF_DIM_V170 = 8`
  - `M7_UISR_DIM_V170 = 64`
  - `extract_v170_sidecar_batch(clean_texts, bpe_lens, charlm=None, rust_charlm=None, fold_ids=None)`
    -> `(m6_ph_16 [N, 16], m7_surf_8 [N, 8], m7_uisr_64 [N, 64])` (all `np.float32`, finite, `C1 === 0.0`).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import _meguviz_rs as _mn
except Exception:  # pragma: no cover
    _mn = None

from .v140_paradise_apex import CHARLM_BITS_V140, CHARLM_SMOOTH_V140

M6_PH_DIM_V170 = 16
M7_SURF_DIM_V170 = 8
M7_UISR_DIM_V170 = 64

_PH_CLIP_LO = 0.05
_PH_CLIP_HI = 18.0
_PH_Q90_THRESH = 14.0


def _script_bucket(ch: str) -> int:
    """0=CJK(Han), 1=Latin, 2=Kana, 3=Other."""
    cp = ord(ch)
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF):
        return 0
    if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0x00C0 <= cp <= 0x024F):
        return 1
    if (0x3040 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF):
        return 2
    return 3


def _uisr_and_surf_one_py(text: str, bpe_len: float) -> tuple[np.ndarray, np.ndarray]:
    surf = np.zeros((M7_SURF_DIM_V170,), dtype=np.float32)
    uisr = np.zeros((M7_UISR_DIM_V170,), dtype=np.float32)
    chars = list(text[:2048])
    n = len(chars)
    if n == 0:
        return surf, uisr

    inv_n = 1.0 / n
    counts = [0, 0, 0, 0]
    first_pos = [1.0, 1.0, 1.0, 1.0]
    last_pos = [0.0, 0.0, 0.0, 0.0]
    head_c = [0, 0, 0, 0]
    mid_c = [0, 0, 0, 0]
    tail_c = [0, 0, 0, 0]
    runs: list[list[int]] = [[], [], [], []]
    trans_out = [0, 0, 0, 0]

    n_digit = 0
    n_space = 0
    n_punct = 0
    n_upper = 0
    n_latin_ascii = 0
    n_nl = 0
    punct_runs = 0
    in_punct_run = False
    uniq_mod64 = 0

    prev_b = -1
    cur_run = 0
    n_switches = 0
    all_runs: list[int] = []
    t1 = n // 3
    t2 = (2 * n) // 3

    for i, ch in enumerate(chars):
        b = _script_bucket(ch)
        counts[b] += 1
        pos_f = i * inv_n
        if pos_f < first_pos[b]:
            first_pos[b] = pos_f
        if pos_f > last_pos[b]:
            last_pos[b] = pos_f
        if i < t1:
            head_c[b] += 1
        elif i < t2:
            mid_c[b] += 1
        else:
            tail_c[b] += 1

        if b == prev_b:
            cur_run += 1
        else:
            if prev_b >= 0:
                runs[prev_b].append(cur_run)
                all_runs.append(cur_run)
                trans_out[prev_b] += 1
                n_switches += 1
            prev_b = b
            cur_run = 1

        cp = ord(ch)
        uniq_mod64 |= 1 << (cp & 63)
        if 0x30 <= cp <= 0x39:
            n_digit += 1
            in_punct_run = False
        elif ch.isspace():
            n_space += 1
            if ch == "\n":
                n_nl += 1
            in_punct_run = False
        elif (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            n_latin_ascii += 1
            if 0x41 <= cp <= 0x5A:
                n_upper += 1
            in_punct_run = False
        elif b == 3:
            n_punct += 1
            if not in_punct_run:
                punct_runs += 1
                in_punct_run = True
        else:
            in_punct_run = False

    if prev_b >= 0:
        runs[prev_b].append(cur_run)
        all_runs.append(cur_run)

    r_h = counts[0] * inv_n
    r_l = counts[1] * inv_n
    r_k = counts[2] * inv_n
    r_o = counts[3] * inv_n
    log_l = math.log1p(n) / 6.0
    fert = float(bpe_len) / max(1.0, float(n))

    surf[0] = r_h
    surf[1] = r_l
    surf[2] = r_k
    surf[3] = r_o
    surf[4] = log_l
    surf[5] = min(4.0, max(0.0, fert))

    n_bytes = len(text[:2048].encode("utf-8", errors="replace"))
    mean_run = (sum(all_runs) / max(1, len(all_runs))) / max(1.0, float(n))
    max_run = (max(all_runs) if all_runs else 0) / max(1.0, float(n))

    uisr[0] = r_h
    uisr[1] = r_l
    uisr[2] = r_k
    uisr[3] = r_o
    uisr[4] = n_digit * inv_n
    uisr[5] = n_space * inv_n
    uisr[6] = n_punct * inv_n
    uisr[7] = n_upper / max(1.0, float(n_latin_ascii))
    uisr[8] = min(1.5, (n_bytes * inv_n) / 4.0)
    uisr[9] = bin(uniq_mod64).count("1") / 64.0
    uisr[10] = log_l
    uisr[11] = min(1.0, mean_run)
    uisr[12] = min(1.0, max_run)
    uisr[13] = n_switches / max(1.0, float(n - 1))
    uisr[14] = punct_runs * inv_n
    uisr[15] = n_nl * inv_n

    n_h = max(1, t1)
    n_m = max(1, t2 - t1)
    n_t = max(1, n - t2)
    for b in range(4):
        off = 16 + b * 12
        cnt = counts[b]
        if cnt == 0:
            continue
        rb = runs[b]
        mean_r = (sum(rb) / max(1, len(rb))) / max(1.0, float(n))
        max_r = (max(rb) if rb else 0) / max(1.0, float(n))
        hf = head_c[b] / n_h
        mf = mid_c[b] / n_m
        tf = tail_c[b] / n_t
        mu3 = (hf + mf + tf) / 3.0
        var3 = ((hf - mu3) ** 2 + (mf - mu3) ** 2 + (tf - mu3) ** 2) / 3.0
        uisr[off + 0] = cnt * inv_n
        uisr[off + 1] = math.log1p(cnt) / 6.0
        uisr[off + 2] = min(1.0, mean_r)
        uisr[off + 3] = min(1.0, max_r)
        uisr[off + 4] = first_pos[b]
        uisr[off + 5] = last_pos[b]
        uisr[off + 6] = max(0.0, last_pos[b] - first_pos[b])
        uisr[off + 7] = trans_out[b] / max(1.0, float(cnt))
        uisr[off + 8] = hf
        uisr[off + 9] = mf
        uisr[off + 10] = tf
        uisr[off + 11] = var3

    return surf, uisr


def _ph16_one_py(lm: Any, clean_text: str) -> np.ndarray:
    out = np.zeros((M6_PH_DIM_V170,), dtype=np.float32)
    out[3] = 1.0  # Q default
    out[14] = 1.0  # R42 default
    if lm is None:
        return out
    chars = list(clean_text[: lm.max_chars])
    l = len(chars)
    if l == 0:
        return out

    vocab = lm.vocabulary
    ids = [vocab.get(c, 1) for c in chars]
    unigrams = lm.unigrams
    p0 = [float(unigrams[cid, 0]) for cid in ids]
    keys_prev = list(ids)

    mean_h = [0.0, 0.0, 0.0, 0.0]
    h4 = [0.0] * l
    cov_voc_cnt = sum(1 for cid in ids if cid >= 2)
    cov_ge3_cnt = 0

    for order in range(4):
        if order == 0:
            s = 0.0
            for i in range(l):
                hv = min(_PH_CLIP_HI, max(_PH_CLIP_LO, -math.log(max(1e-12, p0[i]))))
                s += hv
            mean_h[0] = s / l
        else:
            tb = lm.tables[order - 1]
            keys_arr = tb["keys"]
            cnt_arr = tb["counts"]
            ctx_arr = tb["contexts"]
            den_arr = tb["den"]
            sm = float(CHARLM_SMOOTH_V140[order])
            shift = CHARLM_BITS_V140 * order
            s = 0.0
            for i in range(l):
                lag_id = ids[i - order] if i >= order else 0
                k = keys_prev[i] | (lag_id << shift)
                keys_prev[i] = k

                j_idx = int(np.searchsorted(keys_arr, k))
                j0 = float(cnt_arr[j_idx, 0]) if (j_idx < len(keys_arr) and keys_arr[j_idx] == k) else 0.0

                ctx_k = k >> CHARLM_BITS_V140
                d_idx = int(np.searchsorted(ctx_arr, ctx_k))
                d0 = float(den_arr[d_idx, 0]) if (d_idx < len(ctx_arr) and ctx_arr[d_idx] == ctx_k) else 0.0

                p0[i] = (j0 + sm * p0[i]) / (d0 + sm)
                hv = min(_PH_CLIP_HI, max(_PH_CLIP_LO, -math.log(max(1e-12, p0[i]))))
                s += hv
                if order == 2 and j0 > 0.0:
                    cov_ge3_cnt += 1
                elif order == 3:
                    h4[i] = hv
            mean_h[order] = s / l

    # Welford FP64 mean & variance on h4
    w_mean = 0.0
    w_m2 = 0.0
    j_tail_sum = 0.0
    for idx, x in enumerate(h4, start=1):
        delta = x - w_mean
        w_mean += delta / idx
        w_m2 += delta * (x - w_mean)
        if x > _PH_Q90_THRESH:
            j_tail_sum += x - _PH_Q90_THRESH
    var = (w_m2 / l) if l > 0 else 0.0

    d_sum = 0.0
    for i in range(l - 1):
        diff = h4[i + 1] - h4[i]
        d_sum += diff * diff
    d_val = (d_sum / (l - 1)) if l >= 2 else 0.0

    valid_q = 1.0 if (l >= 6 and var > 1e-6) else 0.0
    q_val = (d_val / (var + 1e-6)) if valid_q > 0.5 else 1.0

    # Haar s=2, s=4 prefix sums (non-overlapping windows, >=3 for valid_haar)
    ps = [0.0] * (l + 1)
    for i in range(l):
        ps[i + 1] = ps[i] + h4[i]

    n2 = l // 2
    e2 = 0.0
    if n2 >= 3:
        s2 = 0.0
        for j in range(n2):
            diff2 = ((ps[2 * j + 2] - ps[2 * j + 1]) - (ps[2 * j + 1] - ps[2 * j])) * 0.5
            s2 += diff2 * diff2
        e2 = s2 / n2

    n4 = l // 4
    e4 = 0.0
    if n4 >= 3:
        s4 = 0.0
        for j in range(n4):
            diff4 = ((ps[4 * j + 4] - ps[4 * j + 2]) - (ps[4 * j + 2] - ps[4 * j])) * 0.25
            s4 += diff4 * diff4
        e4 = s4 / n4

    valid_haar = 1.0 if n4 >= 3 else 0.0
    r42 = (e4 / (e2 + 1e-4)) if valid_haar > 0.5 else 1.0

    out[0] = w_mean
    out[1] = math.log(var + 1e-4)
    out[2] = math.log(d_val + 1e-4)
    out[3] = q_val
    out[4] = valid_q
    out[5] = mean_h[0] - mean_h[3]
    out[6] = mean_h[1] - mean_h[2]
    out[7] = mean_h[2] - mean_h[3]
    out[8] = cov_ge3_cnt / l
    out[9] = cov_voc_cnt / l
    out[10] = math.log1p(l)
    out[11] = j_tail_sum / l
    out[12] = math.log(e2 + 1e-4)
    out[13] = math.log(e4 + 1e-4)
    out[14] = r42
    out[15] = valid_haar
    return out


def extract_v170_sidecar_batch(
    clean_texts: list[str],
    bpe_lens: np.ndarray,
    charlm: Any = None,
    rust_charlm: Any = None,
    fold_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(m6_ph_16 [N,16], m7_surf_8 [N,8], m7_uisr_64 [N,64])`, all `np.float32` finite."""
    n = len(clean_texts)
    if n == 0:
        return (
            np.zeros((0, M6_PH_DIM_V170), dtype=np.float32),
            np.zeros((0, M7_SURF_DIM_V170), dtype=np.float32),
            np.zeros((0, M7_UISR_DIM_V170), dtype=np.float32),
        )

    bpe_arr = np.asarray(bpe_lens, dtype=np.float32).reshape(-1)
    if _mn is not None and hasattr(_mn, "extract_v170_uisr_and_surf_batch_rs"):
        m7_surf_8, m7_uisr_64 = _mn.extract_v170_uisr_and_surf_batch_rs(clean_texts, bpe_arr)
    else:
        m7_surf_8 = np.zeros((n, M7_SURF_DIM_V170), dtype=np.float32)
        m7_uisr_64 = np.zeros((n, M7_UISR_DIM_V170), dtype=np.float32)
        for i, txt in enumerate(clean_texts):
            s_i, u_i = _uisr_and_surf_one_py(txt, float(bpe_arr[i]))
            m7_surf_8[i] = s_i
            m7_uisr_64[i] = u_i

    if rust_charlm is not None and hasattr(rust_charlm, "extract_v170_ph16_batch"):
        fids = np.asarray(fold_ids, dtype=np.int32) if fold_ids is not None else None
        m6_ph_16 = rust_charlm.extract_v170_ph16_batch(clean_texts, fids)
    else:
        m6_ph_16 = np.zeros((n, M6_PH_DIM_V170), dtype=np.float32)
        for i, txt in enumerate(clean_texts):
            m6_ph_16[i] = _ph16_one_py(charlm, txt)

    # Mirror cov_ge3 and cov_voc into M7 surface vector slots [6, 7]
    m7_surf_8[:, 6] = m6_ph_16[:, 8]
    m7_surf_8[:, 7] = m6_ph_16[:, 9]

    np.nan_to_num(m6_ph_16, copy=False, nan=0.0, posinf=18.0, neginf=-18.0)
    np.nan_to_num(m7_surf_8, copy=False, nan=0.0, posinf=4.0, neginf=0.0)
    np.nan_to_num(m7_uisr_64, copy=False, nan=0.0, posinf=4.0, neginf=0.0)
    return m6_ph_16, m7_surf_8, m7_uisr_64
