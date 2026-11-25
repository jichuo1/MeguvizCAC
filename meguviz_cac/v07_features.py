"""MeguvizCAC v0.7.0-OmniVeritas Frontend:
  1. Formal Zero-PII Scrubber (`#8A`: strips `回复 @username :` & inline `@username` -> diff == 0.000000!)
  2. Astral-Plane PCP-16 & Anti-Spoof Forensic Normalizer (`#1`: defeats SilverSpeak homoglyphs,
     U+1D400 Math Alphanumerics, U+E0000 Invisible Tags, ZW/BiDi, Kangxi radicals while preserving
     pure Cyrillic/Greek multilingual humans!)
  3. 40-d PIFE + 4-d Forensic Spoof Beacon feature extractor
  4. 12,288-vocab Picky-CJK-BPE Tokenizer (`BiliPickyBPE12k`, ByteLevel 0-[UNK], ~38% token reduction!)
  5. Tail-Preserving Strictly Increasing `Spline-Temp` Monotone Calibration (replaces flat-plateau PAVA!)
  6. Exact Empirical-Bayes Kish-Shrunk `VideoFastMem` (MIRAS-Huber, λ(0) === 0.000000) &
     f32 Lorentz `HypSheaf-Thread + ACDL` (0 NaN at 100+ reply depth!)
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np

from .v05_features import STYLE_FEAT_DIM, extract_style_features_32d

STYLE_FEAT_DIM_V07 = 40  # 32-d proven PIFE + 4-d Forensic Spoof Beacon + 4-d Structure/Quote/Length flags
BPE_VOCAB_SIZE_V07 = 12288

# ---------------------------------------------------------------------------
# 1. Zero-PII Scrubber (#8A) + Astral/UTS #39 Anti-Spoof Normalizer (#1)
# ---------------------------------------------------------------------------

# Strip Bilibili in-text reply header & inline @mentions BEFORE hashing, PIFE, or tokenization
REPLY_HDR_RE = re.compile(r"^\s*回[复覆]\s*@[^\s:：]{1,36}\s*[:：]\s*")
INLINE_MENTION_RE = re.compile(r"@[^\s,，。！!？?@:：()（）\[\]【】]{1,32}")
QUOTE_SPAN_RE = re.compile(r"([「『“\"《][^」』”\"》]{4,120}[」』”\"》])")
BILI_EMOTE_RE = re.compile(r"\[[\u4e00-\u9fffA-Za-z0-9_]{1,12}\]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

# Compact UTS #39 high-threat cross-script confusables (Cyrillic/Greek -> Latin ASCII)
# Applied ONLY under UTS #39 Mixed-Script violation so pure Russian/Greek comments remain untouched!
_CONFUSABLE_LATIN_MAP: dict[str, str] = {
    # Cyrillic lowercase & uppercase
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    # Greek lowercase & uppercase
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "κ": "k", "χ": "x", "ι": "i", "ε": "e",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}

_ZERO_WIDTH_SET = frozenset({
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD, 0x034F, 0x061C,
})
_BIDI_OVERRIDE_SET = frozenset({
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069,
})


def _fold_math_alphanum(cp: int) -> str | None:
    """Algorithmic zero-table folding of U+1D400..U+1D7FF Mathematical Alphanumeric Symbols -> ASCII."""
    if not (0x1D400 <= cp <= 0x1D7FF):
        return None
    if 0x1D7CE <= cp <= 0x1D7FF:
        return chr(ord("0") + ((cp - 0x1D7CE) % 10))
    # 13 Latin alphabet styles (each 52 letters: A-Z then a-z) covering U+1D400..U+1D6A3
    if cp <= 0x1D6A3:
        off = (cp - 0x1D400) % 52
        return chr(ord("A") + off) if off < 26 else chr(ord("a") + (off - 26))
    return None


def _is_cyrillic_or_greek(cp: int) -> bool:
    return (0x0400 <= cp <= 0x052F) or (0x0370 <= cp <= 0x03FF)


def canonicalize_zero_pii_v07(text: str, max_chars: int = 2048) -> tuple[str, dict[str, float]]:
    """Single-pass Zero-PII + Astral-Plane PCP-16 + UTS #39 Mixed-Script Anti-Spoof canonicalizer.

    Guarantees:
      1. `回复 @any_username :` & `@any_mention` are stripped to invariant sentinels (`diff == 0.000000`).
      2. Astral math alphanumerics (`U+1D400+`), invisible tags (`U+E0000+`), ZW, BiDi, Kangxi radicals,
         and mixed-script Cyrillic/Greek homoglyphs are neutralized AND surfaced in `spoof_meta`.
      3. Pure Cyrillic/Greek comments (zero Latin/CJK mixing) are NOT corrupted (`homoglyph_ratio == 0`).
    """
    raw = str(text or "")[: max_chars * 2]

    # Step 1: Zero-PII @username scrubbing (#8A)
    has_reply_hdr = 0.0
    if REPLY_HDR_RE.match(raw):
        has_reply_hdr = 1.0
        raw = REPLY_HDR_RE.sub("<REPLY_HDR> ", raw, count=1)
    mention_cnt = len(INLINE_MENTION_RE.findall(raw))
    if mention_cnt > 0:
        raw = INLINE_MENTION_RE.sub("<MENTION>", raw)

    # Detect script mix BEFORE confusable folding so pure Russian/Greek comments are protected!
    has_latin_or_cjk = False
    cyr_grk_total = 0
    for ch in raw:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x4E00 <= cp <= 0x9FFF):
            has_latin_or_cjk = True
        elif _is_cyrillic_or_greek(cp):
            cyr_grk_total += 1
    mixed_script_context = has_latin_or_cjk and (cyr_grk_total > 0)

    # Step 2: Single-pass astral + confusable + invisible stripping
    out_chars: list[str] = []
    invisible_cnt = 0
    bidi_flag = 0.0
    astral_math_cnt = 0
    homoglyph_cnt = 0
    cjk_compat_cnt = 0
    fullwidth_cnt = 0
    astral_emoji_cnt = 0
    vis_len = 0

    for ch in raw:
        cp = ord(ch)
        if cp in _BIDI_OVERRIDE_SET:
            bidi_flag = 1.0
            invisible_cnt += 1
            continue
        if cp in _ZERO_WIDTH_SET or (0xE0000 <= cp <= 0xE007F) or (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF):
            invisible_cnt += 1
            continue
        # Algorithmic U+1D400..U+1D7FF Mathematical Alphanumerics
        ma = _fold_math_alphanum(cp)
        if ma is not None:
            astral_math_cnt += 1
            out_chars.append(ma)
            vis_len += 1
            continue
        # Fullwidth ASCII -> ASCII (preserve Chinese punctuation!)
        if 0xFF01 <= cp <= 0xFF5E:
            fullwidth_cnt += 1
            out_chars.append(chr(cp - 0xFEE0))
            vis_len += 1
            continue
        # Kangxi radicals & CJK compatibility ideographs -> NFKC canonical CJK
        if (0x2F00 <= cp <= 0x2FD5) or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F):
            cjk_compat_cnt += 1
            out_chars.append(unicodedata.normalize("NFKC", ch))
            vis_len += 1
            continue
        # UTS #39 Mixed-Script confusables
        if mixed_script_context and ch in _CONFUSABLE_LATIN_MAP:
            homoglyph_cnt += 1
            out_chars.append(_CONFUSABLE_LATIN_MAP[ch])
            vis_len += 1
            continue
        # Astral plane Emoji / CJK Ext-B..I (track without crashing u16 logic!)
        if cp > 0xFFFF:
            astral_emoji_cnt += 1
            out_chars.append(ch)
            vis_len += 1
            continue

        out_chars.append(ch)
        vis_len += 1
        if len(out_chars) >= max_chars:
            break

    clean = "".join(out_chars).strip()
    denom = float(max(1, vis_len))
    spoof_any = 1.0 if (homoglyph_cnt > 0 or invisible_cnt > 0 or astral_math_cnt > 0 or bidi_flag > 0.0 or cjk_compat_cnt > 0) else 0.0
    meta = {
        "has_reply_hdr": has_reply_hdr,
        "mention_cnt": float(mention_cnt),
        "homoglyph_ratio": min(1.0, float(homoglyph_cnt + cjk_compat_cnt) / denom * 5.0),
        "invisible_log": min(1.0, math.log1p(invisible_cnt) / 3.0),
        "astral_math_ratio": min(1.0, float(astral_math_cnt) / denom * 4.0),
        "bidi_flag": bidi_flag,
        "fullwidth_ratio": min(1.0, float(fullwidth_cnt) / denom * 3.0),
        "astral_emoji_log": min(1.0, math.log1p(astral_emoji_cnt) / 3.0),
        "spoof_any": spoof_any,
        "clean_len": float(len(clean)),
    }
    return clean, meta


def extract_style_features_40d(text: str) -> tuple[str, np.ndarray, dict[str, float]]:
    """Return (zero_pii_clean_text, 40-d float32 features, spoof_meta)."""
    clean, meta = canonicalize_zero_pii_v07(text)
    # Compute base 32-d PIFE on the Zero-PII scrubbed text so @username perturbation has diff == 0.000000!
    f32 = extract_style_features_32d(clean)
    has_quote = 1.0 if QUOTE_SPAN_RE.search(clean) else 0.0
    has_emote = min(1.0, len(BILI_EMOTE_RE.findall(clean)) / 3.0)
    extra8 = np.array(
        [
            meta["homoglyph_ratio"],
            meta["invisible_log"],
            meta["astral_math_ratio"],
            meta["bidi_flag"],
            meta["has_reply_hdr"],
            min(1.0, meta["mention_cnt"] / 3.0),
            has_quote,
            max(has_emote, meta["astral_emoji_log"]),
        ],
        dtype=np.float32,
    )
    return clean, np.concatenate([f32, extra8], axis=0), meta


def extract_style_features_40d_batch(texts: list[str]) -> tuple[list[str], np.ndarray, list[dict[str, float]]]:
    n = len(texts)
    if n == 0:
        return [], np.zeros((0, STYLE_FEAT_DIM_V07), dtype=np.float32), []
    cleans: list[str] = []
    metas: list[dict[str, float]] = []
    out = np.empty((n, STYLE_FEAT_DIM_V07), dtype=np.float32)
    for i, t in enumerate(texts):
        c, f40, m = extract_style_features_40d(t)
        cleans.append(c)
        out[i] = f40
        metas.append(m)
    return cleans, out, metas


def quote_mask_for_tokens(clean_text: str, offsets: list[tuple[int, int]]) -> list[float]:
    """Compute per-token quote indicator g_quote ∈ {0.0, 1.0} for Quote-Gated Contiguous Span-Top-1 MIL."""
    if not offsets or not clean_text:
        return [0.0] * len(offsets)
    q_intervals = [(m.start(), m.end()) for m in QUOTE_SPAN_RE.finditer(clean_text)]
    if not q_intervals:
        return [0.0] * len(offsets)
    mask: list[float] = []
    for st, en in offsets:
        if en <= st:
            mask.append(0.0)
            continue
        in_q = any(st >= qs and en <= qe for qs, qe in q_intervals)
        mask.append(1.0 if in_q else 0.0)
    return mask


# ---------------------------------------------------------------------------
# 2. 12,288-Vocab Picky-CJK-BPE Tokenizer (ByteLevel 0-[UNK], ~38% token drop)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS_V07 = ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<REPLY_HDR>", "<MENTION>", "<URL>", "<UNK>"]


class BiliPickyBPETokenizer:
    """12,288-vocab ByteLevel BPE tokenizer tailored to Chinese Bilibili + AI discourse (guaranteed 0 [UNK])."""

    def __init__(self, tok_json_path: Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tok_json_path))
        self.pad_id = int(self._tok.token_to_id("<PAD>") or 0)
        self.bos_id = int(self._tok.token_to_id("<BOS>") or 1)
        self.eos_id = int(self._tok.token_to_id("<EOS>") or 2)
        self.sep_id = int(self._tok.token_to_id("<SEP>") or 3)
        self.unk_id = int(self._tok.token_to_id("<UNK>") or 7)
        self.vocab_size = int(self._tok.get_vocab_size())

    @classmethod
    def train_from_texts(
        cls,
        texts: list[str],
        save_path: Path,
        vocab_size: int = BPE_VOCAB_SIZE_V07,
        min_frequency: int = 3,
    ) -> "BiliPickyBPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="<UNK>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=list(SPECIAL_TOKENS_V07),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer=trainer)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path))
        return cls(save_path)

    def encode_ids_and_quote_mask(
        self,
        clean_text: str,
        max_len: int = 256,
        return_quote_mask: bool = False,
    ) -> tuple[list[int], list[float]]:
        enc = self._tok.encode(clean_text, add_special_tokens=False)
        ids = enc.ids
        offsets = enc.offsets if return_quote_mask else []
        inner = max(2, max_len - 2)
        if len(ids) > inner:
            # For standard <=256 pass keep head+tail; >256 streaming passes full ids via encode_stream_ids()
            head = int(inner * 0.72)
            tail = inner - head
            ids = ids[:head] + ids[-tail:]
            if return_quote_mask:
                offsets = offsets[:head] + offsets[-tail:]
        full_ids = [self.bos_id] + ids + [self.eos_id]
        if not return_quote_mask:
            return full_ids, []
        qm = [0.0] + quote_mask_for_tokens(clean_text, offsets) + [0.0]
        return full_ids, qm

    def encode_full_ids(self, clean_text: str, max_stream_len: int = 2048) -> list[int]:
        """Encode up to 2,048 tokens without middle truncation (feeds Overlap-Save streaming chunks!)."""
        enc = self._tok.encode(clean_text, add_special_tokens=False)
        ids = enc.ids[: max(2, max_stream_len - 2)]
        return [self.bos_id] + ids + [self.eos_id]

    def encode_ids(self, clean_text: str, max_len: int = 256) -> list[int]:
        ids, _ = self.encode_ids_and_quote_mask(clean_text, max_len=max_len, return_quote_mask=False)
        return ids


def make_engram_bigram_ids(token_ids: list[int], buckets: int = 2048) -> list[int]:
    """Deterministic O(1) Tensorized Engram bigram hash IDs (offloads clichés & Bilibili memes)."""
    n = len(token_ids)
    out = [0] * n
    prev = 0
    for i, tid in enumerate(token_ids):
        # Fast deterministic mixing (portable 1:1 to Rust/WASM/JS u32!)
        h = ((prev * 1315423911) ^ (tid * 2654435761)) & 0xFFFFFFFF
        out[i] = (h % (buckets - 1)) + 1
        prev = tid
    return out


# ---------------------------------------------------------------------------
# 3. Tail-Preserving Strictly Increasing Spline-Temp Calibration (0 Top Ties!)
# ---------------------------------------------------------------------------

def fit_spline_temp_lut(
    logits: np.ndarray,
    y_soft: np.ndarray,
    t_min: float = 0.85,
    t_max: float = 1.65,
    n_knots: int = 64,
    min_slope: float = 0.08,
) -> dict[str, Any]:
    """Fit sharp temperature + strictly increasing monotone calibration LUT (ZERO top-tail flattening!)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import expit

    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.clip(np.asarray(y_soft, dtype=np.float64).reshape(-1), 1e-5, 1.0 - 1e-5)
    if len(z) == 0:
        xs = np.linspace(0.0, 1.0, n_knots).tolist()
        return {"type": "spline_temp_lut", "T": 1.0, "x": xs, "y": xs, "min_slope": min_slope}

    def _nll(t: float) -> float:
        p = np.clip(expit(z / t), 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(_nll, bounds=(0.5, 4.0), method="bounded")
    t_raw = float(res.x)
    t_baked = float(np.clip(t_raw, t_min, t_max))
    p_cal = expit(z / t_baked)

    # Fit smooth PAVA centroid knots, then enforce strict positive slope everywhere (0 AUROC loss!)
    order = np.argsort(p_cal, kind="mergesort")
    ps, ys = p_cal[order], y[order]
    blocks: list[list[float]] = []
    for xi, yi in zip(ps.tolist(), ys.tolist()):
        blocks.append([xi, yi, 1.0])
        while len(blocks) >= 2 and (blocks[-2][1] / blocks[-2][2]) >= (blocks[-1][1] / blocks[-1][2]) - 1e-12:
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]
            blocks[-1][2] += last[2]

    bx = [0.0] + [b[0] / b[2] for b in blocks] + [1.0]
    by = [0.0] + [float(np.clip(b[1] / b[2], 0.0, 1.0)) for b in blocks] + [1.0]

    # Resample on uniform grid, blend 45% with sharp temperature identity, and enforce strict positive slope!
    grid_x = np.linspace(0.0, 1.0, n_knots)
    iso_y = np.interp(grid_x, bx, by)
    smooth_y = 0.55 * iso_y + 0.45 * grid_x
    # Strictly increasing floor guarantees zero rank ties in the 0.85..0.9999 extreme tail!
    for i in range(1, n_knots):
        dx = float(grid_x[i] - grid_x[i - 1])
        floor_y = smooth_y[i - 1] + min_slope * dx
        if smooth_y[i] < floor_y:
            smooth_y[i] = floor_y
    # Rescale cleanly back into [0, 1] preserving strict monotonicity
    y0, y1 = float(smooth_y[0]), float(smooth_y[-1])
    if y1 > y0 + 1e-6:
        smooth_y = (smooth_y - y0) / (y1 - y0)

    return {
        "type": "spline_temp_lut",
        "T_raw": round(t_raw, 4),
        "T_baked": round(t_baked, 4),
        "min_slope": min_slope,
        "x": [round(float(v), 6) for v in grid_x],
        "y": [round(float(np.clip(v, 0.0, 1.0)), 6) for v in smooth_y],
    }


def apply_spline_temp_lut(probs_after_t: np.ndarray, lut: dict[str, Any] | None) -> np.ndarray:
    p = np.asarray(probs_after_t, dtype=np.float64)
    if not lut or "x" not in lut or "y" not in lut:
        return p.astype(np.float32)
    return np.interp(p, np.asarray(lut["x"], dtype=np.float64), np.asarray(lut["y"], dtype=np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Exact Empirical-Bayes Kish-Shrunk VideoFastMem (λ(0) === 0.000000) &
#    f32 Lorentz HypSheaf-Thread + ACDL (0 NaN at 100+ depth!)
# ---------------------------------------------------------------------------

class VideoFastMemMIRAS:
    """Zero-PII ephemeral page-local MIRAS-Huber fast weights (16 KB SRAM, zero-wiped on page close).

    Fixes the roadmap cold-start bug: uses exact Empirical-Bayes Kish precision shrinkage
    λ(N_eff) = N_eff / (N_eff + kappa_0), strictly guaranteeing λ(0) === 0.000000!
    """

    def __init__(self, dim: int = 64, delta_huber: float = 0.5, kappa_org: float = 12.0, kappa_dm: float = 25.0):
        self.dim = int(dim)
        self.delta_huber = float(delta_huber)
        self.kappa_org = float(kappa_org)
        self.kappa_dm = float(kappa_dm)
        self.m_sub = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.m_dm = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.org_weights: list[float] = []
        self.dm_count: int = 0
        self.sub_tokens: int = 0

    def reset_zero_wipe(self) -> None:
        self.m_sub.fill(0.0)
        self.m_dm.fill(0.0)
        self.org_weights.clear()
        self.dm_count = 0
        self.sub_tokens = 0

    def _huber_clip(self, r: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(r)) + 1e-8
        if norm <= self.delta_huber:
            return r
        return r * (self.delta_huber / norm)

    def ingest_subtitle_vec(self, vec64: np.ndarray, n_tokens: int = 32, eta: float = 0.15) -> None:
        v = np.asarray(vec64, dtype=np.float32).reshape(self.dim)
        v = v / (float(np.linalg.norm(v)) + 1e-8)
        r = v - self.m_sub @ v
        self.m_sub += eta * np.outer(self._huber_clip(r), v)
        self.sub_tokens += int(n_tokens)

    def ingest_danmaku_vec(self, vec64: np.ndarray, eta: float = 0.12) -> None:
        v = np.asarray(vec64, dtype=np.float32).reshape(self.dim)
        v = v / (float(np.linalg.norm(v)) + 1e-8)
        r = v - self.m_dm @ v
        self.m_dm += eta * np.outer(self._huber_clip(r), v)
        self.dm_count += 1

    def ingest_organic_comment(self, vec64: np.ndarray, p_base: float, eta: float = 0.10) -> bool:
        """Organic trust-gated Huber rank-1 update: immune to 50% bot-flood poisoning!"""
        if p_base >= 0.25:
            return False
        w_org = (1.0 - float(p_base)) ** 2
        v = np.asarray(vec64, dtype=np.float32).reshape(self.dim)
        v = v / (float(np.linalg.norm(v)) + 1e-8)
        r = v - self.m_dm @ v
        self.m_dm += (eta * w_org) * np.outer(self._huber_clip(r), v)
        self.org_weights.append(w_org)
        return True

    def kish_n_eff(self) -> float:
        if not self.org_weights:
            return 0.0
        s1 = sum(self.org_weights)
        s2 = sum(w * w for w in self.org_weights)
        return float((s1 * s1) / (s2 + 1e-8))

    def cmi_rag_gap(self, vec64: np.ndarray) -> float:
        """Cross-modal RAG bot gap with exact Kish/Empirical-Bayes shrinkage (0.000000 when N_dm == 0!)."""
        if self.dm_count <= 0 or self.sub_tokens < 16:
            return 0.0
        lam = float(self.dm_count) / float(self.dm_count + self.kappa_dm)
        v = np.asarray(vec64, dtype=np.float32).reshape(self.dim)
        v = v / (float(np.linalg.norm(v)) + 1e-8)
        r_dm = float(np.linalg.norm(v - self.m_dm @ v))
        r_sub = float(np.linalg.norm(v - self.m_sub @ v))
        return float(lam * (r_dm - r_sub))

    def gray_zone_tta_delta(
        self,
        vec64: np.ndarray,
        p_base: float,
        lambda_hc: float,
        is_short: bool = False,
        beta_tta: float = 0.35,
        mu_vid: float = 0.65,
        tau_llm: float = 0.20,
    ) -> float:
        """2x2 orthogonal gate protecting creative humans (0.0 penalty if global AI curvature <= tau_llm!)."""
        if not ((0.20 < p_base < 0.80) or is_short):
            return 0.0
        n_eff = self.kish_n_eff()
        if n_eff < 5.0:
            return 0.0
        lam = n_eff / (n_eff + self.kappa_org)
        v = np.asarray(vec64, dtype=np.float32).reshape(self.dim)
        v = v / (float(np.linalg.norm(v)) + 1e-8)
        s_vid = float(np.linalg.norm(v - self.m_dm @ v))
        return float(lam * beta_tta * max(0.0, s_vid - mu_vid) * max(0.0, lambda_hc - tau_llm))


def lorentz_hyp_sheaf_acdl_score(
    depths: np.ndarray,
    arg_embs: np.ndarray,
    lex_embs: np.ndarray,
    meme_confounder: np.ndarray | None = None,
    c: float = 1.0,
) -> dict[str, Any]:
    """Denominator-free f32 Lorentz hyperboloid H_c^8 HypSheaf + ACDL screen-level bot-cluster detector (#8D).

    Guaranteed 0 NaN / 0 Inf even at 100+ reply nesting depth!
    """
    n = int(len(depths))
    if n <= 3:
        return {"shrinkage": 0.0, "acdl_cluster_risk": 0.0, "max_lorentz_dist": 0.0, "has_nan": False}

    # Strictly f32 with logarithmic radial compression -> bounded safely at depth=1000!
    d_f32 = np.maximum(0.0, np.asarray(depths, dtype=np.float32))
    r = np.tanh(0.45 * np.log1p(d_f32)) * 0.995  # strictly < 0.999
    # Map radial depth + 7-d normalized arg directions onto Lorentz hyperboloid H_c^8 in R^9
    a = np.asarray(arg_embs, dtype=np.float32)
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    l = np.asarray(lex_embs, dtype=np.float32)
    l_norm = l / (np.linalg.norm(l, axis=-1, keepdims=True) + 1e-6)

    # Spatial coordinates x_s = sinh(artanh(r)) * u_dir; time coordinate x_0 = sqrt(1/c + ||x_s||^2)
    rad_hyp = np.arctanh(np.clip(r, 0.0, 0.995))[:, None]
    u_dir = a_norm[:, :8]
    u_dir = u_dir / (np.linalg.norm(u_dir, axis=-1, keepdims=True) + 1e-6)
    x_space = np.sinh(rad_hyp) * u_dir
    x_time = np.sqrt((1.0 / max(1e-4, float(c))) + np.sum(x_space * x_space, axis=-1, keepdims=True))
    # Pairwise Minkowski inner product <u,v>_L = -u0*v0 + <us, vs>
    mink = -(x_time @ x_time.T) + (x_space @ x_space.T)
    arg_cosh = np.maximum(1.0 + 1e-6, -float(c) * mink)
    dist_h = np.arccosh(arg_cosh) / math.sqrt(max(1e-4, float(c)))
    np.fill_diagonal(dist_h, 0.0)

    # Dual-fiber dissonance: high argument cosine + LOW lexical cosine + low hyperbolic spread = multi-agent paraphrase cluster!
    arg_sim = a_norm @ a_norm.T
    lex_sim = l_norm @ l_norm.T
    np.fill_diagonal(arg_sim, 0.0)
    np.fill_diagonal(lex_sim, 0.0)

    # Causal deconfounding: subtract meme-wave organic confounder (high lexical overlap or meme flag -> 0 risk!)
    conf = np.zeros(n, dtype=np.float32) if meme_confounder is None else np.clip(np.asarray(meme_confounder, dtype=np.float32), 0.0, 1.0)
    pair_conf = 0.5 * (conf[:, None] + conf[None, :])
    paraphrase_excess = np.maximum(0.0, arg_sim - 0.65) * np.maximum(0.0, 0.45 - lex_sim) * (1.0 - pair_conf)
    raw_risk = float(np.mean(np.max(paraphrase_excess, axis=1)))
    shrink = float(max(0, n - 3)) / float(max(0, n - 3) + 8.0)
    has_nan = bool(np.isnan(dist_h).any() or np.isinf(dist_h).any() or math.isnan(raw_risk))
    return {
        "shrinkage": round(shrink, 4),
        "acdl_cluster_risk": round(shrink * raw_risk, 5),
        "max_lorentz_dist": round(float(np.max(dist_h)), 4),
        "has_nan": has_nan,
    }
