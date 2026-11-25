"""Dual-channel PIFE normalization, 32-d zero-PII native stylometrics, Smooth PAVA & length-bucketed Conformal calibration for MeguvizCAC v0.5.0-ApexLoop."""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata

import numpy as np


MAX_LEN_V05 = 192
HEAD_CHARS_V05 = 140
TAIL_CHARS_V05 = 50
STYLE_FEAT_DIM = 32

# Zero-width & invisible control characters commonly abused in obfuscation attacks
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad\u034f\u180e]")

# Common Cyrillic / Greek Latin-lookalike homoglyphs (PIFE dual-channel forensics)
_HOMOGLYPH_MAP = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
        "α": "a", "β": "b", "ε": "e", "ο": "o", "ρ": "p", "τ": "t", "χ": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
    }
)
_HOMOGLYPH_KEYS = frozenset(_HOMOGLYPH_MAP.keys())

_CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff][ \t]+[\u4e00-\u9fff]")
_EMDASH_RE = re.compile(r"——|—|–|--")
_BOOK_QUOTE_RE = re.compile(r"《[^》]{1,32}》")
_SCORECARD_RE = re.compile(r"\d+(?:\.\d+)?\s*/\s*(?:10|5|100)|(?:评分|打分|星级|推荐度)")
_LIST_MARK_RE = re.compile(r"(?:^|[\s，。；;])(?:[1-9][\.、\)）]|[①②③④⑤]|[-•\*]\s+)")
_BENIGN_URL_RE = re.compile(r"https?://|github\.com|arxiv\.org|b23\.tv|pan\.quark\.cn|BV[A-Za-z0-9]{10}", re.IGNORECASE)
_BILI_EMOTE_RE = re.compile(r"\[[\u4e00-\u9fffA-Za-z0-9_]{1,8}\]")
_KAOMOJI_RE = re.compile(r"[(\[（【][^()\n\u4e00-\u9fffA-Za-z0-9]{2,12}[)\]）】]")
_REPLY_PREFIX_RE = re.compile(r"^\s*回[复覆]\s*(?:@[^\s:：]{1,24}\s*)?[:：]")
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_UPPER_TECH_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,10}\b")
_FULLWIDTH_ASCII_RE = re.compile(r"[\uff01-\uff5e]")


def pife_normalize_and_forensics(raw_text: str, max_len: int = MAX_LEN_V05) -> tuple[str, tuple[float, float, float, float]]:
    """Dual-channel PIFE preprocessor: returns sanitized text + 4-d obfuscation forensics tuple."""
    raw = str(raw_text or "")
    raw_len = max(1, len(raw))

    zw_matches = _ZERO_WIDTH_RE.findall(raw)
    zw_cnt = len(zw_matches)
    no_zw = _ZERO_WIDTH_RE.sub("", raw) if zw_cnt else raw

    fw_cnt = len(_FULLWIDTH_ASCII_RE.findall(no_zw))
    homo_cnt = sum(1 for ch in no_zw if ch in _HOMOGLYPH_KEYS)
    mapped = no_zw.translate(_HOMOGLYPH_MAP) if homo_cnt else no_zw

    norm = " ".join(unicodedata.normalize("NFKC", mapped).split())
    edit_delta = abs(len(raw) - len(norm)) / float(raw_len)
    if len(norm) > max_len:
        norm = norm[:HEAD_CHARS_V05] + " " + norm[-TAIL_CHARS_V05:]

    forensics = (
        min(1.0, math.log1p(zw_cnt) / 3.0),
        min(1.0, math.log1p(homo_cnt) / 3.0),
        min(1.0, fw_cnt / float(raw_len) * 4.0),
        min(1.0, float(edit_delta)),
    )
    return norm, forensics


def prepare_text_v05(raw_text: str, max_len: int = MAX_LEN_V05) -> str:
    """Return PIFE-sanitized text ready for v0.5 SuperBPE tokenization."""
    norm, _ = pife_normalize_and_forensics(raw_text, max_len=max_len)
    return norm


def _char_script_bucket(cp: int) -> int:
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return 1  # CJK
    if (65 <= cp <= 90) or (97 <= cp <= 122):
        return 2  # ASCII Latin
    if 48 <= cp <= 57:
        return 3  # Digit
    if 0x3040 <= cp <= 0x30FF:
        return 4  # Japanese Kana
    if 0xAC00 <= cp <= 0xD7AF:
        return 5  # Korean Hangul
    return 0


def extract_style_features_32d(raw_text: str) -> np.ndarray:
    """Compute 32-d zero-PII, topic-agnostic stylometric + PIFE forensics vector in <0.01ms."""
    raw = str(raw_text or "")
    norm, (f_zw, f_homo, f_fw, f_delta) = pife_normalize_and_forensics(raw, max_len=512)
    n_ch = max(1, len(norm))
    raw_bytes = norm.encode("utf-8", errors="replace")
    n_b = max(1, len(raw_bytes))

    # Byte Shannon entropy
    b_counts = Counter(raw_bytes)
    entropy = 0.0
    inv_nb = 1.0 / float(n_b)
    for cnt in b_counts.values():
        p = cnt * inv_nb
        entropy -= p * math.log2(p)

    # Script distributions & transitions
    cjk = ascii_a = digits = kana = hangul = switches = 0
    prev_b = 0
    for ch in norm:
        b = _char_script_bucket(ord(ch))
        if b == 1:
            cjk += 1
        elif b == 2:
            ascii_a += 1
        elif b == 3:
            digits += 1
        elif b == 4:
            kana += 1
        elif b == 5:
            hangul += 1
        if b in (1, 2, 4, 5) and prev_b in (1, 2, 4, 5) and b != prev_b:
            switches += 1
        if b != 0:
            prev_b = b

    ttr = len(set(norm)) / float(n_ch)
    cjk_spaces = len(_CJK_SPACE_RE.findall(raw))
    emdashes = len(_EMDASH_RE.findall(norm))
    colons = norm.count(":") + norm.count("：")
    book_quotes = len(_BOOK_QUOTE_RE.findall(norm))
    scorecards = len(_SCORECARD_RE.findall(norm))
    list_marks = len(_LIST_MARK_RE.findall(norm))
    has_url = 1.0 if _BENIGN_URL_RE.search(norm) else 0.0
    bili_emotes = len(_BILI_EMOTE_RE.findall(norm))
    kaomojis = len(_KAOMOJI_RE.findall(raw))
    has_reply = 1.0 if _REPLY_PREFIX_RE.search(norm) else 0.0
    upper_tech = len(_UPPER_TECH_RE.findall(norm))

    # Bracket / quote closure balance
    pairs_chk = (("(", ")"), ("（", "）"), ("《", "》"), ("“", "”"), ("[", "]"))
    tot_br = 0
    matched_br = 0
    for lch, rch in pairs_chk:
        lc = norm.count(lch)
        rc = norm.count(rch)
        tot_br += lc + rc
        matched_br += 2 * min(lc, rc)
    punct_closure = (matched_br / float(tot_br)) if tot_br > 0 else 0.0

    # Sentence rhythm stats
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(norm) if s.strip()]
    n_sents = max(1, len(sents))
    s_lens = [len(s) for s in sents] if sents else [n_ch]
    s_mean = sum(s_lens) / float(len(s_lens))
    s_var = sum((x - s_mean) ** 2 for x in s_lens) / float(len(s_lens))
    s_std = math.sqrt(s_var)

    commas = norm.count(",") + norm.count("，") + norm.count("、")
    periods = norm.count(".") + norm.count("。")
    q_exc = norm.count("?") + norm.count("？") + norm.count("!") + norm.count("！")
    casual_tone = norm.count("~") + norm.count("～") + norm.count("…") + sum(norm.count(c) for c in ("啦", "呢", "呀", "哇", "咯", "嘛", "哈哈", "草"))

    feat = np.array(
        [
            min(1.5, math.log1p(n_ch) / 6.0),
            min(1.5, math.log1p(n_b) / 6.5),
            min(1.0, entropy / 8.0),
            ttr,
            cjk / float(n_ch),
            ascii_a / float(n_ch),
            digits / float(n_ch),
            kana / float(n_ch),
            hangul / float(n_ch),
            min(1.0, switches / float(n_ch) * 6.0),
            min(1.0, cjk_spaces / max(1.0, n_ch / 10.0)),
            min(1.0, emdashes / 3.0),
            min(1.0, colons / 4.0),
            punct_closure,
            min(1.0, book_quotes / 2.0),
            min(1.0, scorecards / 3.0),
            min(1.0, list_marks / 4.0),
            has_url,
            min(1.0, bili_emotes / 3.0),
            min(1.0, kaomojis / 2.0),
            has_reply,
            min(1.0, math.log1p(n_sents) / 3.0),
            min(1.5, math.log1p(s_mean) / 5.0),
            min(1.5, math.log1p(s_std) / 4.5),
            min(1.0, commas / float(commas + periods + 1)),
            min(1.0, q_exc / 4.0),
            min(1.0, casual_tone / 5.0),
            min(1.0, upper_tech / 4.0),
            f_zw,
            f_homo,
            f_fw,
            f_delta,
        ],
        dtype=np.float32,
    )
    return feat


def extract_style_features_batch(texts: list[str]) -> np.ndarray:
    """Batch-compute (B, 32) float32 stylometric features."""
    if not texts:
        return np.zeros((0, STYLE_FEAT_DIM), dtype=np.float32)
    out = np.empty((len(texts), STYLE_FEAT_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = extract_style_features_32d(t)
    return out


def extract_context_echo_4d(text: str, video_title: str = "", root_text: str = "") -> np.ndarray:
    """Optional zero-PII content-echo features (video_echo & root_echo, zero username/level PII)."""
    if not video_title and not root_text:
        return np.zeros(4, dtype=np.float32)

    def _bigrams(s: str) -> set[str]:
        clean = re.sub(r"\s+", "", prepare_text_v05(s, max_len=256).lower())
        if len(clean) < 2:
            return {clean} if clean else set()
        return {clean[i : i + 2] for i in range(len(clean) - 1)}

    bg_t = _bigrams(text)
    v_ov = v_jac = r_ov = 0.0
    if video_title and bg_t:
        bg_v = _bigrams(video_title)
        if bg_v:
            inter = len(bg_t & bg_v)
            v_ov = inter / float(max(1, min(len(bg_t), len(bg_v))))
            v_jac = inter / float(max(1, len(bg_t | bg_v)))
    if root_text and bg_t:
        bg_r = _bigrams(root_text)
        if bg_r:
            r_ov = len(bg_t & bg_r) / float(max(1, min(len(bg_t), len(bg_r))))
    return np.array([v_ov, v_jac, r_ov, 1.0], dtype=np.float32)


def length_bucket_name(text: str) -> str:
    n = len(prepare_text_v05(text))
    if n <= 15:
        return "short"
    if n <= 64:
        return "medium"
    return "long"


def fit_smooth_pava(
    probs: np.ndarray,
    y_soft: np.ndarray,
    sample_weights: np.ndarray | None = None,
    min_slope: float = 0.12,
    raw_blend: float = 0.15,
    max_knots: int = 96,
) -> dict:
    """Weighted Smooth PAVA: collapses flat step plateaus via block centroids + minimum strictly-increasing slope floor."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(y_soft, dtype=np.float64).reshape(-1)
    w_in = np.ones_like(p) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0], "raw_blend": float(raw_blend)}

    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    y_s = y[order]
    w_s = np.maximum(w_in[order], 1e-6)

    # PAVA blocks storing [sum_wx, sum_wy, sum_w] so each plateau resolves to ONE smooth centroid knot
    blocks: list[list[float]] = []
    for xi, yi, wi in zip(p_s.tolist(), y_s.tolist(), w_s.tolist()):
        blocks.append([wi * xi, wi * yi, wi])
        while len(blocks) >= 2:
            prev_val = blocks[-2][1] / blocks[-2][2]
            curr_val = blocks[-1][1] / blocks[-1][2]
            if prev_val >= curr_val - 1e-12:
                last = blocks.pop()
                blocks[-1][0] += last[0]
                blocks[-1][1] += last[1]
                blocks[-1][2] += last[2]
            else:
                break

    xs = [float(b[0] / b[2]) for b in blocks]
    ys = [float(np.clip(b[1] / b[2], 0.0, 1.0)) for b in blocks]

    # Anchor endpoints smoothly so boundary extrapolation never clips flat prematurely
    if xs[0] > 1e-6:
        y0 = max(0.0, ys[0] - min_slope * xs[0])
        xs.insert(0, 0.0)
        ys.insert(0, y0)
    if xs[-1] < 1.0 - 1e-6:
        y1 = min(1.0, ys[-1] + min_slope * (1.0 - xs[-1]))
        xs.append(1.0)
        ys.append(y1)

    # Enforce strict positive monotonicity with gentle blend against identity to prevent plateau snapping
    alpha_id = float(np.clip(raw_blend, 0.05, 0.40))
    ys_smooth = [(1.0 - alpha_id) * yi + alpha_id * xi for xi, yi in zip(xs, ys)]
    for i in range(1, len(ys_smooth)):
        min_next = ys_smooth[i - 1] + 1e-4 * max(1e-4, xs[i] - xs[i - 1])
        if ys_smooth[i] < min_next:
            ys_smooth[i] = min(1.0, min_next)

    if len(xs) > max_knots:
        keep_idx = np.unique(np.round(np.linspace(0, len(xs) - 1, max_knots)).astype(np.int64))
        xs = [xs[int(i)] for i in keep_idx]
        ys_smooth = [ys_smooth[int(i)] for i in keep_idx]

    return {
        "x": [round(float(v), 6) for v in xs],
        "y": [round(float(np.clip(v, 0.0, 1.0)), 6) for v in ys_smooth],
        "raw_blend": 0.0,  # already smoothly baked into ys_smooth
        "smooth_pava": True,
        "n_blocks": len(blocks),
    }


def apply_smooth_pava(probs: np.ndarray | float, knots: dict | None):
    if not knots or not knots.get("x") or not knots.get("y"):
        return probs
    return np.interp(probs, knots["x"], knots["y"])


def fit_bucketed_smooth_pava(
    texts: list[str],
    probs: np.ndarray,
    y_soft: np.ndarray,
    sample_weights: np.ndarray | None = None,
    min_bucket_n: int = 180,
) -> dict:
    """Fit global + length-bucketed (short/medium/long) Smooth PAVA calibration tables."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_soft, dtype=np.float64)
    w = np.ones_like(p) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    global_cal = fit_smooth_pava(p, y, sample_weights=w)
    buckets: dict[str, dict] = {"global": global_cal}
    b_names = [length_bucket_name(t) for t in texts]
    for bn in ("short", "medium", "long"):
        idx = [i for i, b in enumerate(b_names) if b == bn]
        if len(idx) >= min_bucket_n:
            buckets[bn] = fit_smooth_pava(p[idx], y[idx], sample_weights=w[idx])
        else:
            buckets[bn] = global_cal
    return buckets


def apply_bucketed_smooth_pava(texts: list[str], probs: np.ndarray, cal_tables: dict | None) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if not cal_tables:
        return p.astype(np.float32)
    if "x" in cal_tables and "y" in cal_tables:
        return np.asarray(apply_smooth_pava(p, cal_tables), dtype=np.float32)
    out = np.empty_like(p)
    glob = cal_tables.get("global")
    for i, t in enumerate(texts):
        bn = length_bucket_name(t)
        tbl = cal_tables.get(bn) or glob
        out[i] = float(apply_smooth_pava(p[i], tbl))
    return out.astype(np.float32)
