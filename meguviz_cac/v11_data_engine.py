"""MeguvizCAC v1.1.0-OmniGuard Unified Online Data Engine (`Length×Domain IPW`, `8-Archetype LongMix`, `Format-Invariant Aug`).

Directly targets all 3 structural data bottlenecks proven by our corpus X-ray audit:
  1. `compute_joint_length_domain_ipw()`: Smooth joint Inverse-Propensity Weighting across `5 Length × 8 Domain`
     cells, neutralizing spurious shortcuts (`L4/L5 = 90% Human`, `kedaibiao = 99% AI`, `tech_struct = 85% AI`)
     while capping weights in `[0.35, 4.0]` so sparse cells never explode gradient variance!
  2. `OnlineLongMixSynthesizerV11`: Zero-cost online multi-paragraph synthesizer generating balanced `400–2,048`-char
     documents across ALL 8 long-text archetypes (`pure_hu`, `hard_hu`, `pure_ai`, `sandw_2ai`, `tiny_1ai`,
     `tail_ai`, `bare_quote_ai`, `zero_marker_hidden_promo`) with exact char-level `ai_spans` & `promo_spans`
     strictly from verified `T0/T1` clean rows!
  3. `FormatShortcutAugmentorV11`: Label-preserving stochastic perturbations (`@mention` Zero-PII fuzz, Bilibili
     long/short emote parity injection on BOTH human & AI, list-marker jitter, paragraph spacing jitter) so the
     model cannot overfit superficial formatting cues!
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any

from .v11_data_refinery import length_bucket_v11, primary_domain_tag_v11

_BILI_AUG_EMOTES = (
    "[doge]",
    "[笑哭]",
    "[滑稽]",
    "[脱单doge]",
    "[妙啊]",
    "[TES·2026薪火耀新程_冲鸭]",
    "[Cat2_EatDonut]",
    "[星星眼]",
)
_FAKE_USERS = ("路人甲_9527", "Bili_Dev_2026", "夜航星辰", "Kiro_4060Ti", "阿宅同学")
_LIST_ALT = ("1. ", "① ", "- ", "1、")


def compute_joint_length_domain_ipw(
    records: list[dict[str, Any]],
    prior_smooth: float = 25.0,
    min_w: float = 0.35,
    max_w: float = 4.0,
) -> list[float]:
    """Compute smooth joint `Length × Domain` Inverse-Propensity Weights balancing AI vs Human inside every slice.

    Each record needs `text` (str) and `y_score` (float in `[0, 1]` or `0..100`).
    Mid-band samples (`0.30 < y < 0.70`) receive neutral `1.0` IPW so they are neither inflated nor crushed.
    """
    cell_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [n_hu, n_ai]
    cell_keys: list[tuple[str, str]] = []
    cls_flags: list[int] = []  # 0=hu, 1=ai, -1=mid

    for r in records:
        txt = str(r.get("text") or r.get("raw_text") or "")
        y = float(r.get("y_score", r.get("score", 0.5)))
        if y > 1.0:
            y = y / 100.0
        key = (length_bucket_v11(len(txt)), primary_domain_tag_v11(txt))
        cell_keys.append(key)
        if y <= 0.30:
            cell_counts[key][0] += 1
            cls_flags.append(0)
        elif y >= 0.70:
            cell_counts[key][1] += 1
            cls_flags.append(1)
        else:
            cls_flags.append(-1)

    # Empirical-Bayes smoothed balance target (50/50 inside each Length x Domain cell!)
    weights: list[float] = []
    for key, flg in zip(cell_keys, cls_flags):
        if flg == -1:
            weights.append(1.0)
            continue
        n_hu, n_ai = cell_counts[key]
        # Smoothed propensity p(class | cell)
        p_cls = ( (n_hu if flg == 0 else n_ai) + prior_smooth ) / float(n_hu + n_ai + 2.0 * prior_smooth)
        raw_ipw = 0.5 / max(1e-4, p_cls)
        # Mild extra boost on starved long buckets (`L4/L5` > 256 chars) so long texts get healthy epoch share!
        if key[0] in ("L4_257-512", "L5_>512"):
            raw_ipw *= 1.35
        weights.append(round(float(min(max_w, max(min_w, raw_ipw))), 4))

    # Mean-normalize around 1.0 so global learning rate schedule remains invariant
    if weights:
        mean_w = sum(weights) / float(len(weights))
        if mean_w > 1e-6:
            weights = [round(float(min(max_w, max(min_w, w / mean_w))), 4) for w in weights]
    return weights


class FormatShortcutAugmentorV11:
    """Label-preserving stochastic perturbations that destroy superficial format shortcuts & enforce Zero-PII."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def augment(self, text: str, p_mention: float = 0.12, p_emote: float = 0.15, p_list_jitter: float = 0.15) -> str:
        out = str(text or "")
        if not out:
            return out
        # 1. Zero-PII reply header / @mention fuzz (must leave model output strictly invariant!)
        if self.rng.random() < p_mention and not out.startswith("回复 @"):
            u = self.rng.choice(_FAKE_USERS)
            if self.rng.random() < 0.6:
                out = f"回复 @{u} : {out}"
            else:
                out = f"@{u} {out}"
        # 2. Bilibili emote injection on BOTH human & AI (breaks lazy "emote == human" or "long-emote == AI" shortcuts!)
        if self.rng.random() < p_emote:
            em = self.rng.choice(_BILI_AUG_EMOTES)
            out = f"{out}{em}"
        # 3. List-numbering format jitter (`1. ` <-> `① ` <-> `- `) so numbered structure isn't a brittle trigger
        if self.rng.random() < p_list_jitter and re.search(r"(?:^|\n)1\.\s", out):
            if self.rng.random() < 0.5:
                out = re.sub(r"(^|\n)([1-9])\.\s", r"\1\2、", out)
            else:
                out = re.sub(r"(^|\n)[1-9]\.\s", r"\1- ", out)
        return out


class OnlineLongMixSynthesizerV11:
    """Zero-cost 12-Archetype Online Long-Document & Hybrid-Boundary Synthesizer (strictly from clean T0/T1 rows!)."""

    ARCHETYPES = (
        "1_pure_hu",
        "2_hard_hu",
        "3_pure_ai",
        "4_sandw_2ai",
        "5_tiny_1ai",
        "6_tail_ai",
        "7_bare_quote_ai",
        "8_hidden_promo",
        "9_seamless_no_newline_hybrid",
        "10_interleaved_sent_hybrid",
        "11_quote_wrapped_ai_bypass",
        "12_long_quoted_promo_critique",
    )

    def __init__(
        self,
        clean_hu: list[str],
        clean_hu_hard: list[str],
        clean_ai: list[str],
        clean_promo_hu: list[str],
        seed: int = 20261126,
    ):
        assert len(clean_hu) >= 10 and len(clean_ai) >= 10, "Need at least 10 clean human & AI anchors!"
        self.hu = list(clean_hu)
        self.hu_hard = list(clean_hu_hard) if clean_hu_hard else list(clean_hu)
        self.ai = list(clean_ai)
        self.promo_hu = list(clean_promo_hu) if clean_promo_hu else list(clean_hu)
        self.rng = random.Random(seed)

    def _pick(self, arr: list[str], k: int) -> list[str]:
        return [arr[self.rng.randrange(len(arr))] for _ in range(k)]

    def sample_one(self, archetype: str | None = None) -> dict[str, Any]:
        kind = archetype or self.rng.choice(self.ARCHETYPES)
        if kind == "1_pure_hu":
            txt = "\n\n".join(self._pick(self.hu, self.rng.randint(4, 7)))
            return {"kind": kind, "text": txt, "y_ai": 0.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [], "promo_spans": []}

        if kind == "2_hard_hu":
            if self.rng.random() < 0.35:
                b = self._pick(self.hu, 2)
                flood = "太稳了，太绝了，太顶了，太干脆，太直白，太通透，" * self.rng.randint(5, 8)
                em = self.rng.choice(_BILI_AUG_EMOTES) * self.rng.randint(3, 6)
                txt = f"{b[0]}\n\n{flood}{em}\n\n{b[1]}"
            else:
                segs = self._pick(self.hu_hard, self.rng.randint(3, 5)) + self._pick(self.hu, 2)
                self.rng.shuffle(segs)
                txt = "\n\n".join(segs)
            return {"kind": kind, "text": txt, "y_ai": 0.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [], "promo_spans": []}

        if kind == "3_pure_ai":
            txt = "\n\n".join(self._pick(self.ai, self.rng.randint(4, 7)))
            return {"kind": kind, "text": txt, "y_ai": 1.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [(0, len(txt))], "promo_spans": []}

        if kind == "4_sandw_2ai":
            h1 = "\n\n".join(self._pick(self.hu, self.rng.randint(2, 3)))
            am = "\n\n".join(self._pick(self.ai, 2))
            h2 = "\n\n".join(self._pick(self.hu, self.rng.randint(2, 3)))
            st = len(h1) + 2
            return {"kind": kind, "text": f"{h1}\n\n{am}\n\n{h2}", "y_ai": 1.0, "y_hyb": 1.0, "y_promo": 0.0, "ai_spans": [(st, st + len(am))], "promo_spans": []}

        if kind == "5_tiny_1ai":
            h1 = "\n\n".join(self._pick(self.hu, 3))
            am = self._pick(self.ai, 1)[0]
            h2 = "\n\n".join(self._pick(self.hu, 2))
            st = len(h1) + 2
            return {"kind": kind, "text": f"{h1}\n\n{am}\n\n{h2}", "y_ai": 1.0, "y_hyb": 1.0, "y_promo": 0.0, "ai_spans": [(st, st + len(am))], "promo_spans": []}

        if kind == "6_tail_ai":
            h1 = "\n\n".join(self._pick(self.hu, self.rng.randint(3, 5)))
            at = "\n\n".join(self._pick(self.ai, 2))
            st = len(h1) + 2
            return {"kind": kind, "text": f"{h1}\n\n{at}", "y_ai": 1.0, "y_hyb": 1.0, "y_promo": 0.0, "ai_spans": [(st, st + len(at))], "promo_spans": []}

        if kind == "7_bare_quote_ai":
            h1 = "\n\n".join(self._pick(self.hu, 2))
            q_ai = self._pick(self.ai, 1)[0]
            h2 = "\n\n".join(self._pick(self.hu, 2))
            if self.rng.random() < 0.5:
                mid = f"前面看到有人贴了段AI评论，原文如下：「{q_ai}」看得人脚趾抠地。"
            else:
                mid = f"“{q_ai}”——我就问这种话说出来谁信啊。"
            return {"kind": kind, "text": f"{h1}\n\n{mid}\n\n{h2}", "y_ai": 0.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [], "promo_spans": []}

        if kind == "8_hidden_promo":
            # 8_hidden_promo (ZERO artificial transition cue!)
            h1 = "\n\n".join(self._pick(self.hu, self.rng.randint(3, 5)))
            pr = self._pick(self.promo_hu, 1)[0]
            if self.rng.random() < 0.5:
                st = len(h1) + 2
                txt = f"{h1}\n\n{pr}"
            else:
                h2 = "\n\n".join(self._pick(self.hu, 2))
                st = len(h1) + 2
                txt = f"{h1}\n\n{pr}\n\n{h2}"
            return {"kind": "8_hidden_promo", "text": txt, "y_ai": 0.0, "y_hyb": 0.0, "y_promo": 1.0, "ai_spans": [], "promo_spans": [(st, st + len(pr))]}

        if kind == "9_seamless_no_newline_hybrid":
            # No `\n\n` boundary jump — inline continuation destroying paragraph-break seam shortcuts (#5!)
            h1 = "".join(self._pick(self.hu, 3))
            am = "".join(self._pick(self.ai, 2))
            h2 = "".join(self._pick(self.hu, 2))
            st = len(h1)
            txt = f"{h1}{am}{h2}"
            return {"kind": kind, "text": txt, "y_ai": 1.0, "y_hyb": 1.0, "y_promo": 0.0, "ai_spans": [(st, st + len(am))], "promo_spans": []}

        if kind == "10_interleaved_sent_hybrid":
            # Alternating H-A-H-A within one continuous block
            hs = self._pick(self.hu, 3)
            ais = self._pick(self.ai, 2)
            parts = [hs[0], ais[0], hs[1], ais[1], hs[2]]
            spans: list[tuple[int, int]] = []
            cur = 0
            for idx, p in enumerate(parts):
                if idx in (1, 3):
                    spans.append((cur, cur + len(p)))
                cur += len(p) + 1
            txt = " ".join(parts)
            return {"kind": kind, "text": txt, "y_ai": 1.0, "y_hyb": 1.0, "y_promo": 0.0, "ai_spans": spans, "promo_spans": []}

        if kind == "11_quote_wrapped_ai_bypass":
            # Adversarial AI wrapping its entire multi-paragraph output in quotation marks -> MUST remain `y_ai=1.0`!
            body = "\n\n".join(self._pick(self.ai, self.rng.randint(3, 5)))
            ql, qr = self.rng.choice((("“", "”"), ("「", "」")))
            txt = f"{ql}{body}{qr}"
            return {"kind": kind, "text": txt, "y_ai": 1.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [(0, len(txt))], "promo_spans": []}

        # 12_long_quoted_promo_critique: long human post quoting & mocking a promo ad -> `y_ai=0.0, y_promo=0.0`!
        h1 = "\n\n".join(self._pick(self.hu, 2))
        pr = self._pick(self.promo_hu, 1)[0]
        h2 = "\n\n".join(self._pick(self.hu, 2))
        mid = f"笑死评论区又刷这种硬广了，原文如下：「{pr}」大伙千万别信这种割韭菜的话术。"
        return {"kind": "12_long_quoted_promo_critique", "text": f"{h1}\n\n{mid}\n\n{h2}", "y_ai": 0.0, "y_hyb": 0.0, "y_promo": 0.0, "ai_spans": [], "promo_spans": []}

    def sample_batch(self, n: int) -> list[dict[str, Any]]:
        return [self.sample_one(self.ARCHETYPES[i % len(self.ARCHETYPES)]) for i in range(n)]
