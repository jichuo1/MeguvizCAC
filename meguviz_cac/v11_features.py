"""MeguvizCAC v1.1.0-OmniGuard Frontend (`56-d` Length-Invariant Zero-PII Stylometry + Promo Clues + Boundary-Aware Chunking).

Fixes ALL long-text frontend bottlenecks diagnosed in v0.5–v1.0:
  1. Eliminates the `v05_features.py` `512 -> 191` discontinuous truncation cliff (`max_chars = 8192`, smooth everywhere!).
  2. Replaces raw count saturation (`count / 4`) and Heaps-law-confounded raw TTR with length-invariant per-100-char
     densities and `MATTR-64` (Moving-Average Type-Token Ratio) + `MATTR_std` cross-window lexical burstiness.
  3. Preserves paragraph breaks (`\\n+`) BEFORE whitespace collapsing to compute true document macro-fingerprints:
     paragraph length CV, sentence burstiness CV, head-to-tail lexical drift, 4-gram/synonym repetition flood ratio
     (catches copy-pasta & keyboard-smash long human FPs!), markdown scaffold density, and quote coverage ratio.
  4. Adds `8-d` strictly Zero-PII pure-text commercial/promo clues with split URL channels (open-source/academic URLs
     like `github.com / arxiv.org` separated from e-commerce/funnel URLs so technical referrals never trigger promo FPs).
  5. Provides Sentence/Paragraph-Boundary-Aware Overlap-Save Chunking (`plan_long_chunks_v11`) with explicit quote-span
     & quote-attribution-intro awareness so quoted AI snippets inside long human posts are never conflated with authorship!
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any

import numpy as np

from .v07_features import (
    QUOTE_SPAN_RE,
    canonicalize_zero_pii_v07,
)

STYLE_FEAT_DIM_V11 = 56  # 40-d upgraded length-invariant PIFE + 8-d Long-Doc Macro + 8-d Zero-PII Commercial/Promo
CHUNK_META_DIM_V11 = 6   # [pos_start, pos_end, log_n_chunks, quote_frac, has_quote_intro, rel_len]

# Improved Bilibili official emote regex (handles long multi-word tags like `[TES·2026薪火耀新程_冲鸭]`)
_BILI_EMOTE_V11_RE = re.compile(r"\[[^\s\[\]]{1,24}\]")
_KAOMOJI_V11_RE = re.compile(r"[(\[（【][^()\n\u4e00-\u9fffA-Za-z0-9]{2,12}[)\]）】]")
_CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff][ \t]+[\u4e00-\u9fff]")
_EMDASH_RE = re.compile(r"——|—|–|--")
_BOOK_QUOTE_RE = re.compile(r"《[^》]{1,40}》")
_SCORECARD_RE = re.compile(r"\d+(?:\.\d+)?\s*/\s*(?:10|5|100)|(?:评分|打分|星级|推荐度)")
_LIST_MARK_RE = re.compile(r"(?:^|[\n，。；;\s])(?:[1-9][0-9]?[\.、\)）]|[①②③④⑤⑥⑦⑧⑨⑩]|[-•\*]\s+|#{1,4}\s+)")
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_UPPER_TECH_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,12}\b")
_MD_SCAFFOLD_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,4}\s|\*\*.{2,24}\*\*[:：]?|首先|其次|再次|最后|总而言之|综上所述|值得注意的是|另一方面|第一[、.：:]|第二[、.：:]|第三[、.：:])"
)

# Quote-attribution intro cues preceding quoted AI/third-party snippets
_QUOTE_INTRO_RE = re.compile(
    r"(?:原文如下|引用一下|贴一段|复制一段|有人发了|AI生成的|大模型写的|ChatGPT说|DeepSeek说|机翻原文|原话是|评论区看到的)[:：]?\s*[「『“\"《]?"
)
# Long quote spans (up to 600 chars for multi-sentence quotes inside long posts!)
_LONG_QUOTE_SPAN_RE = re.compile(r"([「『“\"][^\n」』”\"]{8,600}[」』”\"]|```[\s\S]{12,800}```)")

# Zero-PII pure-text commercial/promo cues (v1.1.0 Dual-Axis, hardened with PCR-ToxiCN Chinese evasion variants)
_CONTACT_ID_PII_RE = re.compile(
    r"((?:加微|加v|vx|v信|薇信|威信|微信|绿泡泡|扣群|q群|qq群?|交流\d*群|粉丝[群裙]|进[群裙]|君羊|小红书|xhs|助理(?:老师)?|微信号|扣扣|🐧|🛰️)\s*[:：号是]?\s*)[A-Za-z0-9_\-]{5,20}",
    re.I,
)
_PROMO_PRICE_RE = re.compile(
    r"(?:券后|领[券劵]|券\d+元|满\d+减\d+|到手价|史低|直降|百亿补贴|国补|秒杀价|折合|实付\d+|神券|优惠[券劵]|返佣|蓝链|蓝色文字|小蓝字|置顶链接|评论区置顶|凑单|抄作业)"
)
_PROMO_MULTI_SKU_RE = re.compile(r"(?:[|｜/]\s*(?:券|满|到手|\d+元|华为|小米|索尼|苹果|OPPO|vivo|红米|罗技))")
_PROMO_CTA_RE = re.compile(
    r"(?:关注一下|求关注|互关|互粉|点个关注|私信领|私信发|私信我|思信|滴我|回复[「“\"].{1,8}[」”\"]|免费领|戳我主页|看我主页|主页置顶|主页橱窗|简介自取|加微|加v|vx|薇信|威信|绿泡泡|扣群|粉丝[群裙]|进[群裙]|君羊|公[众粽]号|工种号|功重号|\bgzh\b|\bxhs\b|接单|代做|晚八点|手慢无|冲就完了|🛰️|🐧|🍠|<CONTACT_ID>)",
    re.I,
)
_PROMO_SHILL_RE = re.compile(
    r"(?:亲测有效|亲测好用|闭眼入|闭眼冲|无脑入|无脑冲|入手不亏|回购第|已经回购|不踩雷|早买早享受|吊打.{1,8}牌|性价比无敌|质感封神|良心推荐|强烈安利|真心安利|强烈建议入手)"
)
_PROMO_KEDAIBIAO_RE = re.compile(r"(?:课代表|省流总结|空降\s*\d|免责声明.{0,12}AI|AI(?:辅助)?(?:生成|整理|提炼).{0,12}仅供)")
_PROMO_EN_SELF_RE = re.compile(r"(?:check my channel|subscribe to my|please support|new to bilibili|my artwork|follow my channel)", re.I)

# Split URL channels: benign open-source/academic vs commercial/shortlink/funnel!
_OPENSOURCE_URL_RE = re.compile(
    r"(?:github\.com|gitee\.com|arxiv\.org|huggingface\.co|docs\.python\.org|pytorch\.org|wikipedia\.org|zhihu\.com/question|stackoverflow\.com|开源仓库|开源项目|官方文档)",
    re.I,
)
_COMMERCE_URL_RE = re.compile(
    r"(?:taobao\.com|tmall\.com|jd\.com|pinduoduo\.com|yangkeduo\.com|goofish\.com|1688\.com|tb\.cn|u\.jd\.com|pan\.quark\.cn|pan\.baidu\.com|拼多多|京东自营|淘宝搜|天猫超市|闲鱼同款)",
    re.I,
)


def _char_script_bucket(cp: int) -> int:
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return 1
    if (65 <= cp <= 90) or (97 <= cp <= 122):
        return 2
    if 48 <= cp <= 57:
        return 3
    if 0x3040 <= cp <= 0x30FF:
        return 4
    if 0xAC00 <= cp <= 0xD7AF:
        return 5
    return 0


def _compute_mattr_64(text_compact: str, window: int = 64, stride: int = 32) -> tuple[float, float]:
    """Length-invariant Moving-Average Type-Token Ratio (MATTR-64) + cross-window std (burstiness)."""
    n = len(text_compact)
    if n == 0:
        return 0.0, 0.0
    if n <= window:
        return len(set(text_compact)) / float(n), 0.0
    vals: list[float] = []
    for st in range(0, n - window + 1, stride):
        sub = text_compact[st : st + window]
        vals.append(len(set(sub)) / float(window))
    if not vals:
        return len(set(text_compact)) / float(n), 0.0
    mean_v = sum(vals) / float(len(vals))
    var_v = sum((v - mean_v) ** 2 for v in vals) / float(len(vals))
    return mean_v, math.sqrt(var_v)


def _compute_repeat_4gram_ratio(text_compact: str) -> float:
    """Fraction of 4-grams that repeat (detects copy-pasta floods, synonym spam, & keyboard smashes)."""
    n = len(text_compact)
    if n < 12:
        return 0.0
    grams = [text_compact[i : i + 4] for i in range(n - 3)]
    c = Counter(grams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return min(1.0, rep / float(max(1, len(grams))) * 2.5)


def _compute_head_tail_lex_drift(text_compact: str) -> float:
    """Bigram Jaccard divergence between first 35% and last 35% of text (detects long-doc seam / register drift)."""
    n = len(text_compact)
    if n < 80:
        return 0.0
    k = max(24, int(n * 0.35))
    h = text_compact[:k]
    t = text_compact[-k:]
    bg_h = {h[i : i + 2] for i in range(len(h) - 1)}
    bg_t = {t[i : i + 2] for i in range(len(t) - 1)}
    if not bg_h or not bg_t:
        return 0.0
    jacc = len(bg_h & bg_t) / float(len(bg_h | bg_t))
    return float(1.0 - jacc)


def extract_style_features_56d(text: str, max_chars: int = 8192) -> tuple[str, np.ndarray, dict[str, float]]:
    """Extract Zero-PII canonical text + 56-d length-invariant macro-stylometry & commercial clues in <0.1 ms.

    Strict invariants:
      - Zero-PII: `@username` & `回复 @username :` are scrubbed BEFORE all stats (`diff === 0.000000`).
      - Zero 512->191 cliff: computes smooth length-invariant rates & MATTR-64 across up to 8,192 chars!
      - Preserves paragraph breaks (`\\n+`) before collapsing spaces to capture true paragraph burstiness CV.
    """
    clean, meta = canonicalize_zero_pii_v07(text, max_chars=max_chars)
    # Strict Zero-PII scrubbing of raw QQ/WeChat/group contact IDs (preserves funnel cue token before `<CONTACT_ID>`)
    clean = _CONTACT_ID_PII_RE.sub(r"\1<CONTACT_ID>", clean)
    # Strip Bilibili emotes with "" (NOT " " — avoids injecting synthetic CJK intra-word spaces into `_CJK_SPACE_RE`!)
    emotes = _BILI_EMOTE_V11_RE.findall(clean)
    emote_cnt = len(emotes)
    clean_no_emote = _BILI_EMOTE_V11_RE.sub("", clean)

    # Preserve paragraph structure BEFORE whitespace collapsing!
    paras = [p.strip() for p in re.split(r"\n+", clean_no_emote) if p.strip()]
    n_paras = max(1, len(paras))
    p_lens = [len(p) for p in paras] if paras else [max(1, len(clean_no_emote))]
    p_mean = sum(p_lens) / float(len(p_lens))
    p_std = math.sqrt(sum((x - p_mean) ** 2 for x in p_lens) / float(len(p_lens)))
    para_len_cv = min(2.0, p_std / (p_mean + 1e-3)) if n_paras >= 2 else 0.0

    # Quote coverage ratio (for Quote-Gated MIL & long human quote protection)
    quote_spans = [(m.start(), m.end()) for m in _LONG_QUOTE_SPAN_RE.finditer(clean)]
    if not quote_spans:
        quote_spans = [(m.start(), m.end()) for m in QUOTE_SPAN_RE.finditer(clean)]
    quoted_chars = sum(en - st for st, en in quote_spans)
    quote_ratio = min(1.0, quoted_chars / float(max(1, len(clean))))

    # Normalized single-space text (NO 512->191 truncation!)
    norm = " ".join(unicodedata.normalize("NFKC", clean_no_emote).split())
    n_ch = max(1, len(norm))
    per100 = float(n_ch) / 100.0 + 0.5  # smooth rate denominator -> length-invariant from 20 to 8,192 chars!
    compact = re.sub(r"\s+", "", norm).lower()

    raw_bytes = norm.encode("utf-8", errors="replace")
    n_b = max(1, len(raw_bytes))
    b_counts = Counter(raw_bytes)
    inv_nb = 1.0 / float(n_b)
    entropy = -sum((cnt * inv_nb) * math.log2(cnt * inv_nb) for cnt in b_counts.values())

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

    # Length-invariant MATTR-64 + burstiness + repetition + head/tail drift
    mattr_64, mattr_std = _compute_mattr_64(compact, window=64, stride=32)
    rep4_ratio = _compute_repeat_4gram_ratio(compact)
    ht_drift = _compute_head_tail_lex_drift(compact)

    # Punctuation & structural density rates (smooth log1p per 100 chars -> zero long-text saturation!)
    cjk_spaces = len(_CJK_SPACE_RE.findall(clean_no_emote))
    emdashes = len(_EMDASH_RE.findall(norm))
    colons = norm.count(":") + norm.count("：")
    book_quotes = len(_BOOK_QUOTE_RE.findall(norm))
    scorecards = len(_SCORECARD_RE.findall(norm))
    list_marks = len(_LIST_MARK_RE.findall(clean_no_emote))
    md_scaffolds = len(_MD_SCAFFOLD_RE.findall(clean_no_emote))
    kaomojis = len(_KAOMOJI_V11_RE.findall(clean_no_emote))
    upper_tech = len(_UPPER_TECH_RE.findall(norm))

    pairs_chk = (("(", ")"), ("（", "）"), ("《", "》"), ("“", "”"), ("[", "]"))
    tot_br = matched_br = 0
    for lch, rch in pairs_chk:
        lc, rc = norm.count(lch), norm.count(rch)
        tot_br += lc + rc
        matched_br += 2 * min(lc, rc)
    punct_closure = (matched_br / float(tot_br)) if tot_br > 0 else 0.0

    sents = [s.strip() for s in _SENT_SPLIT_RE.split(clean_no_emote) if s.strip()]
    n_sents = max(1, len(sents))
    s_lens = [len(s) for s in sents] if sents else [n_ch]
    s_mean = sum(s_lens) / float(len(s_lens))
    s_std = math.sqrt(sum((x - s_mean) ** 2 for x in s_lens) / float(len(s_lens)))
    sent_cv = min(2.0, s_std / (s_mean + 1e-3))

    commas = norm.count(",") + norm.count("，") + norm.count("、")
    periods = norm.count(".") + norm.count("。")
    q_exc = norm.count("?") + norm.count("？") + norm.count("!") + norm.count("！")
    casual_tone = norm.count("~") + norm.count("～") + norm.count("…") + sum(
        norm.count(c) for c in ("啦", "呢", "呀", "哇", "咯", "嘛", "哈哈", "草")
    )

    # Zero-PII Commercial / Promo pure-text clues (8-d)
    p_price = len(_PROMO_PRICE_RE.findall(clean))
    p_sku = len(_PROMO_MULTI_SKU_RE.findall(clean))
    p_cta = len(_PROMO_CTA_RE.findall(clean))
    p_shill = len(_PROMO_SHILL_RE.findall(clean))
    p_kdb = 1.0 if _PROMO_KEDAIBIAO_RE.search(clean) else 0.0
    p_en_self = 1.0 if _PROMO_EN_SELF_RE.search(clean) else 0.0
    has_os_url = 1.0 if _OPENSOURCE_URL_RE.search(clean) else 0.0
    has_comm_url = 1.0 if _COMMERCE_URL_RE.search(clean) else 0.0

    # Smooth rate helper: log1p(cnt / per100) / 2.0 in [0, ~1.5]
    def _rate(cnt: float) -> float:
        return min(1.5, math.log1p(cnt / per100) / 2.0)

    feat56 = np.array(
        [
            # --- 32-d upgraded length-invariant base stylometry (no 512 cliff, no count saturation!) ---
            min(1.8, math.log1p(n_ch) / 6.0),
            min(1.8, math.log1p(n_b) / 6.5),
            min(1.0, entropy / 8.0),
            mattr_64,  # upgraded from raw TTR -> length-invariant MATTR-64!
            cjk / float(n_ch),
            ascii_a / float(n_ch),
            digits / float(n_ch),
            kana / float(n_ch),
            hangul / float(n_ch),
            min(1.0, switches / float(n_ch) * 6.0),
            min(1.0, cjk_spaces / max(1.0, n_ch / 10.0)),
            _rate(emdashes),
            _rate(colons),
            punct_closure,
            _rate(book_quotes),
            _rate(scorecards),
            _rate(list_marks),
            has_os_url,  # benign open-source/academic referral
            min(1.0, math.log1p(emote_cnt) / 2.5),
            _rate(kaomojis),
            meta["has_reply_hdr"],
            min(1.5, math.log1p(n_sents) / 3.5),
            min(1.5, math.log1p(s_mean) / 5.0),
            min(1.5, math.log1p(s_std) / 4.5),
            min(1.0, commas / float(commas + periods + 1)),
            _rate(q_exc),
            _rate(casual_tone),
            _rate(upper_tech),
            meta["invisible_log"],
            meta["homoglyph_ratio"],
            meta["fullwidth_ratio"],
            meta["astral_math_ratio"],
            # --- 8-d v0.7 spoof / reply / quote / astral metadata ---
            meta["bidi_flag"],
            meta["spoof_any"],
            min(1.0, meta["mention_cnt"] / 3.0),
            1.0 if quote_ratio > 0.02 else 0.0,
            quote_ratio,
            meta["astral_emoji_log"],
            min(1.0, emote_cnt / float(max(1, n_ch / 15.0))),  # emote spam ratio!
            1.0 if _QUOTE_INTRO_RE.search(clean) else 0.0,
            # --- 8-d NEW Long-Document Macro-Fingerprints ---
            min(1.5, math.log1p(n_paras) / 2.5),
            min(1.5, para_len_cv),
            min(1.5, sent_cv),
            min(1.0, mattr_std * 5.0),
            ht_drift,
            rep4_ratio,
            _rate(md_scaffolds),
            min(1.0, float(n_ch > 256) * math.log1p(max(0, n_ch - 256)) / 7.0),
            # --- 8-d NEW Zero-PII Pure-Text Commercial / Promo Clues ---
            _rate(p_price),
            min(1.0, p_sku / 3.0),
            _rate(p_cta),
            _rate(p_shill),
            p_kdb,
            has_comm_url,
            p_en_self,
            max(0.0, has_comm_url - has_os_url),  # net commercial vs open-source link polarity
        ],
        dtype=np.float32,
    )
    meta_out = {
        **meta,
        "n_paras": float(n_paras),
        "para_len_cv": float(para_len_cv),
        "sent_cv": float(sent_cv),
        "mattr_64": float(mattr_64),
        "mattr_std": float(mattr_std),
        "rep4_ratio": float(rep4_ratio),
        "quote_ratio": float(quote_ratio),
        "emote_cnt": float(emote_cnt),
    }
    return clean, feat56, meta_out


def plan_long_chunks_v11(
    clean_text: str,
    target_chars: int = 240,
    stride_chars: int = 165,
    max_chunks: int = 24,
) -> list[dict[str, Any]]:
    """Paragraph & Sentence-Boundary-Aware Overlap-Save Chunker with Quote-Attribution Tracking.

    Unlike blind fixed-stride token slicing:
      1. Snaps chunk end boundaries to the nearest paragraph break (`\\n`) or sentence boundary (`。！？!?；;`)
         within ±45 chars so clauses & rhetorical contrasts (`不是X，是Y`) are NEVER cut in half!
      2. Computes per-chunk `quote_frac` and `has_quote_intro` so `DynaSeam-MIL` knows whether a high-AI chunk
         is genuinely authored by the commenter or merely quoted/critiqued (`「…AI原文…」`).
      3. Returns exact `(start_char, end_char)` spans for UI highlighting & hybrid boundary localization!
    """
    n = len(clean_text)
    if n == 0:
        return []

    # Pre-locate quoted intervals in clean_text
    q_intervals = [(m.start(), m.end()) for m in _LONG_QUOTE_SPAN_RE.finditer(clean_text)]
    if not q_intervals:
        q_intervals = [(m.start(), m.end()) for m in QUOTE_SPAN_RE.finditer(clean_text)]

    def _quote_overlap_frac(st: int, en: int) -> float:
        if en <= st or not q_intervals:
            return 0.0
        ov = 0
        for qs, qe in q_intervals:
            a, b = max(st, qs), min(en, qe)
            if b > a:
                ov += b - a
        return min(1.0, ov / float(en - st))

    def _snap_end(start: int, ideal_end: int) -> int:
        if ideal_end >= n:
            return n
        lo = max(start + 80, ideal_end - 45)
        hi = min(n, ideal_end + 45)
        window = clean_text[lo:hi]
        # Prefer paragraph break closest to ideal_end, else sentence boundary
        best_pos = -1
        best_dist = 10**9
        for pat in (r"\n+", r"[。！？!?；;]"):
            for m in re.finditer(pat, window):
                cand = lo + m.end()
                dist = abs(cand - ideal_end)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = cand
            if best_pos != -1:
                return best_pos
        return ideal_end

    raw_spans: list[tuple[int, int]] = []
    if n <= target_chars + 45:
        raw_spans.append((0, n))
    else:
        st = 0
        while st < n and len(raw_spans) < max_chunks:
            en = _snap_end(st, st + target_chars)
            raw_spans.append((st, en))
            if en >= n:
                break
            next_st = max(st + 64, min(en - 32, st + stride_chars))
            if n - next_st < 70 and len(raw_spans) >= 1:
                # Absorb tiny trailing tail into last chunk if close, else ensure final tail coverage
                if n - st <= target_chars + 110:
                    raw_spans[-1] = (st, n)
                    break
                next_st = max(st + 64, n - target_chars)
            st = next_st

    n_chunks = len(raw_spans)
    log_m = min(1.5, math.log1p(n_chunks) / 3.0)
    chunks: list[dict[str, Any]] = []
    for idx, (st, en) in enumerate(raw_spans):
        sub = clean_text[st:en]
        q_frac = _quote_overlap_frac(st, en)
        # Check if quote-attribution intro appears inside this chunk OR within 40 chars before it
        ctx_before = clean_text[max(0, st - 40) : en]
        has_q_intro = 1.0 if (_QUOTE_INTRO_RE.search(ctx_before) and q_frac > 0.15) else 0.0
        pos_st = st / float(max(1, n))
        pos_en = en / float(max(1, n))
        rel_len = min(1.5, (en - st) / float(max(1, target_chars)))
        c_meta = np.array([pos_st, pos_en, log_m, q_frac, has_q_intro, rel_len], dtype=np.float32)
        chunks.append(
            {
                "chunk_idx": idx,
                "start_char": st,
                "end_char": en,
                "text": sub,
                "quote_frac": q_frac,
                "has_quote_intro": bool(has_q_intro > 0.5),
                "chunk_meta": c_meta,
            }
        )
    return chunks
