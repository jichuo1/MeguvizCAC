"""MeguvizCAC v1.1.0-OmniGuard Forensic-Safe Dataset Augmentation Suite (`v11_augment.py`).

Unlike naive NLP augmentations (random word deletion, synonym swap, back-translation) which DESTROY
micro-cadence & self-poison AI-detection labels, every transform here is mathematically **Label-Preserving**
for BOTH orthogonal axes (`AI Origin × Commercial/Soft-Shill Intent`):

  1. `AdversarialObfuscationAug`: Simulates real-world spammer & AI evasion attacks (`ZWSP` zero-width insertion,
     CJK intra-word spacing on CTA/coupon keywords like `"领 券"` / `"私 信"`, fullwidth ASCII, Cyrillic/Latin
     homoglyphs, mathematical alphanumerics `U+1D400`) while strictly preserving true labels!
  2. `BrandPriceSlotAug`: Lockstep randomization of numbers/prices (`299 -> 459`, `满300-50 -> 满500-80`), reply
     codewords (`「666」 -> 「干货」`), & platform tokens across BOTH `promo_text` and `organic_text` of minimal
     counterfactual pairs — eliminates brand/number memorization!
  3. `QuadSpliceAug`: Cross-quadrant structural composition:
       - `Q3_Hidden_Tail_Promo`: neutral human paragraphs + verified human promo (`y_ai=0, y_promo=1`)
       - `Q2_Hidden_Tail_AI_Shill`: pure AI paragraphs + verified AI soft-shill (`y_ai=1, y_promo=1`)
       - `Quoted_Promo_Critique` (CRITICAL HARD NEGATIVE!): human mocking/quoting an ad or shill (`y_ai=0, y_promo=0`)
       - `Negated_Shill_Critique` (CRITICAL HARD NEGATIVE!): human warning against hype keywords (`"别信闭眼入全是托"` -> `y_promo=0`)!
  4. `ContiguousCadenceCropAug`: Extracts contiguous 1–2 sentence spans (`35–110` chars) from verified long AI/human
     comments to enrich short stealth-AI coverage WITHOUT corrupting intra-sentence LLM syntax!
  5. `orthogonal_dual_axis_pair_loss()`: PyTorch pairwise ranking + axis-orthogonality loss for minimal pairs
     (`promo` separates `promo_text > organic_text` while `ai` stays strictly pinned to `promo_is_ai` on BOTH!).
"""

from __future__ import annotations

import random
import re
from typing import Any

import torch
import torch.nn.functional as F

_ZW_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060")
_HOMO_LATIN = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у"}
_CTA_SPLIT_PAT = re.compile(r"(领券|私信|关注|加微|网盘|闭眼入|闭眼冲|百亿补贴|蓝链|置顶|回购|进群|公众号)")
# Real-world Bilibili PCR-ToxiCN Chinese evasion substitutions (homophones, pinyin initials, emoji cloaking, char-split)
_ZH_HOMOPHONE_AD_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("公众号", ("工种号", "功重号", "公粽号", "gzh")),
    ("微信", ("薇信", "威信", "V信", "绿泡泡", "vx", "🛰️")),
    ("加微", ("加v", "➕微", "加绿泡泡", "加🛰️")),
    ("蓝链", ("蓝色文字", "蓝色字", "小蓝字", "👇蓝字")),
    ("私信", ("思信", "滴我", "私聊", "sx")),
    ("网盘", ("网盼", "夸克盘", "度盘", "wp", "qk盘")),
    ("优惠券", ("隐藏福利", "专属优惠", "福利劵", "隐藏劵")),
    ("领券", ("领劵", "拿劵", "薅劵")),
    ("粉丝群", ("粉丝裙", "交流裙", "粉丝君羊")),
    ("进群", ("进裙", "进君羊", "加🐧群")),
    ("小红书", ("🍠", "xhs", "某书")),
    ("拼多多", ("pdd", "拼夕夕", "并夕夕")),
    ("淘宝", ("tb", "某宝", "桃宝")),
    ("京东", ("jd", "东哥家", "某东")),
    ("置顶", ("评论区👆", "楼顶", "首评")),
)
_FANDOM_HYPE_HARD_NEGS: tuple[str, ...] = (
    "黑神话这美术和配乐直接封神！国产单机真的站起来了，趁周末闭眼冲绝不后悔！",
    "这季番剧作画质量质感封神，强烈安利给全宿舍一起追，每周蹲更新太爽了！",
    "UP主这期开源架构拆解太顶了，真良心推荐，比我之前看的一堆水视频强十倍，已一键三连！",
    "空洞骑士丝之歌要是真发了我直接无脑入首发，老玩家早买早享受不踩雷哈哈！",
    "这个开源免费字体排版效果亲测好用，没有任何收费套路，强烈建议做PPT的同学都去GitHub下！",
    "老头环DLC这地图立体设计性价比无敌，几十块钱玩一百多小时真的入股不亏，魂游粉必冲！",
)
_HART_FORMAL_TO_CASUAL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("然而，", ("但是吧，", "不过说实话，")),
    ("此外，", ("还有就是，", "另外")),
    ("综上所述，", ("说白了，", "反正总结下来，")),
    ("总而言之，", ("归根结底说白了，", "讲真，")),
    ("值得注意的是，", ("有个点要注意，", "其实")),
    ("不仅如此，", ("而且更绝的是，", "不止这样，")),
)
_PRICE_NUM_PAT = re.compile(r"(?<![A-Za-z0-9])([1-9]\d{1,3})(?=(?:元|块|减|\-|折|\s|$|，|。|！))")
_CODEWORD_PAT = re.compile(r"[「“\"]([0-9A-Za-z\u4e00-\u9fff]{1,6})[」”\"]")
_PLATFORMS = ("京东", "淘宝", "拼多多", "天猫")
_CODEWORDS = ("666", "888", "资料", "干货", "上岸", "进阶", "111", "领取")
_SENT_BREAK_PAT = re.compile(r"(?<=[。！？!?])\s*")

_QUOTE_PROMO_INTROS = (
    "笑死评论区又来水军刷托儿了，原话是：",
    "别信楼上那个带货广告，人家直接复制粘贴的：",
    "刚看到一条离谱软广，贴出来给大伙避避雷：",
    "这年头托儿演都不演了，评论区清一色写着：",
)
_QUOTE_PROMO_OUTROS = (
    "这种一看就是收钱刷的，真有人信吗？",
    "翻主页全是同款带货话术，大伙别被割韭菜了[笑哭]。",
    "但凡真买过的都知道根本没吹得那么神，纯纯智商税。",
    "建议直接点举报，评论区乌烟瘴气的。",
)
_NEGATED_HYPE_TEMPLATES = (
    "千万别听评论区那些吹「{kw}」的，我上个月自己掏钱买回来踩大坑了，退货还得自付运费。",
    "凡是张口就让你「{kw}」还不提一点缺点的，十有八九是收钱的水军托儿，大伙买之前多看差评。",
    "以前我也信了网上说的「{kw}」，结果到手质感稀烂客服还装死，以后再也不跟风买网红款了。",
    "笑死，底下好几个号复读「{kw}」，点进主页连一条正常动态都没有，品牌方买水军能不能走点心？",
)
_HYPE_KWS = ("闭眼入不踩雷", "亲测有效回购三瓶", "性价比无敌吊打大牌", "早买早享受直接冲", "私信666免费领全套资料", "领券史低手慢无")


class AdversarialObfuscationAug:
    """Forensic-safe evasion perturbation simulating real-world spammer & AI obfuscation attacks."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def obfuscate_promo_or_ai(self, text: str, intensity: float = 0.5) -> str:
        """Apply realistic zero-width / CTA-spacing / fullwidth / homoglyph obfuscation (label-preserving!)."""
        s = str(text or "")
        if len(s) < 8:
            return s
        # Reallocate 80% budget to attacks that SURVIVE `canonicalize_zero_pii_v07()` (PCR Chinese homophones, emoji/pinyin
        # cloaking, CTA intra-word spacing) & 20% to canonicalizer-handled Unicode tricks (zw/fullwidth/homoglyph/math)!
        roll = self.rng.random()
        if roll < 0.55:
            mode = "zh_homophone"
        elif roll < 0.80:
            mode = "space_cta"
        else:
            mode = self.rng.choice(("zw_cta", "fullwidth_ascii", "homoglyph_mix", "math_ascii"))
        if mode == "zh_homophone":
            out_s = s
            changed = False
            for src_k, alts in _ZH_HOMOPHONE_AD_MAP:
                if src_k in out_s:
                    out_s = out_s.replace(src_k, self.rng.choice(alts), 1)
                    changed = True
            if changed:
                return out_s
            mode = "space_cta"
        if mode == "zw_cta":
            # Insert zero-width chars inside CTA/promo keywords or random CJK boundaries
            zw = self.rng.choice(_ZW_CHARS)
            if _CTA_SPLIT_PAT.search(s):
                return _CTA_SPLIT_PAT.sub(lambda m: zw.join(list(m.group(0))), s, count=2)
            pos = self.rng.randint(2, max(2, len(s) - 2))
            return s[:pos] + zw + s[pos:]
        if mode == "space_cta":
            if _CTA_SPLIT_PAT.search(s):
                return _CTA_SPLIT_PAT.sub(lambda m: " ".join(list(m.group(0))), s, count=2)
            # Subtle CJK spacing around middle characters
            mid = len(s) // 2
            return s[:mid] + " " + s[mid:]
        if mode == "fullwidth_ascii":
            out = []
            for ch in s:
                cp = ord(ch)
                if 0x21 <= cp <= 0x7E and self.rng.random() < intensity:
                    out.append(chr(cp + 0xFEE0))
                else:
                    out.append(ch)
            return "".join(out)
        if mode == "homoglyph_mix":
            out = []
            for ch in s:
                if ch in _HOMO_LATIN and self.rng.random() < intensity:
                    out.append(_HOMO_LATIN[ch])
                else:
                    out.append(ch)
            return "".join(out)
        # math_ascii: map ASCII letters to Mathematical Sans-Serif U+1D5A0..U+1D5D3 (common WeChat/VX bypass!)
        out = []
        for ch in s:
            if "A" <= ch <= "Z" and self.rng.random() < intensity:
                out.append(chr(0x1D5A0 + (ord(ch) - ord("A"))))
            elif "a" <= ch <= "z" and self.rng.random() < intensity:
                out.append(chr(0x1D5BA + (ord(ch) - ord("a"))))
            else:
                out.append(ch)
        return "".join(out)


class BrandPriceSlotAug:
    """Lockstep randomization of prices, platforms, & codewords across counterfactual pairs (prevents entity memorization!)."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def augment_pair(self, promo_text: str, organic_text: str) -> tuple[str, str]:
        p_out, o_out = str(promo_text), str(organic_text)
        # 1. Lockstep platform swap (e.g. 京东 <-> 淘宝 across BOTH texts!)
        present = [pl for pl in _PLATFORMS if pl in p_out or pl in o_out]
        if present and self.rng.random() < 0.65:
            src_pl = present[0]
            dst_pl = self.rng.choice([pl for pl in _PLATFORMS if pl != src_pl])
            p_out = p_out.replace(src_pl, dst_pl)
            o_out = o_out.replace(src_pl, dst_pl)
        # 2. Lockstep price/number perturbation (consistent mapping so organic price-talk matches coupon price!)
        nums = sorted(set(_PRICE_NUM_PAT.findall(p_out) + _PRICE_NUM_PAT.findall(o_out)), key=len, reverse=True)
        if nums and self.rng.random() < 0.65:
            scale = self.rng.choice((0.8, 0.9, 1.15, 1.25))
            repl: dict[str, str] = {}
            for num in nums[:4]:
                v = int(num)
                nv = max(10, int(round(v * scale / 10.0) * 10))
                if nv != v:
                    repl[num] = str(nv)
            for old_n, new_n in repl.items():
                p_out = re.sub(rf"(?<![A-Za-z0-9]){old_n}(?![A-Za-z0-9])", new_n, p_out)
                o_out = re.sub(rf"(?<![A-Za-z0-9]){old_n}(?![A-Za-z0-9])", new_n, o_out)
        # 3. Codeword randomization in CTA funnel (`回复「666」` -> `回复「资料」`)
        if _CODEWORD_PAT.search(p_out) and self.rng.random() < 0.50:
            new_cw = self.rng.choice(_CODEWORDS)
            p_out = _CODEWORD_PAT.sub(f"「{new_cw}」", p_out, count=1)
        return p_out, o_out


class QuadSpliceAug:
    """Cross-quadrant structural splicer + Anti-Keyword-Shortcut Hard-Negative Generator."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def make_quoted_promo_critique(self, promo_text: str) -> dict[str, Any]:
        """Turn any real/counterfactual ad into a Q4 hard negative: human quoting & mocking the ad (`y_ai=0, y_promo=0`)."""
        intro = self.rng.choice(_QUOTE_PROMO_INTROS)
        outro = self.rng.choice(_QUOTE_PROMO_OUTROS)
        q_l, q_r = self.rng.choice((("「", "」"), ("“", "”")))
        snippet = promo_text.strip()[:140]
        return {
            "text": f"{intro}{q_l}{snippet}{q_r}{outro}",
            "y_ai": 0.0,
            "y_promo": 0.0,
            "promo_type": "none",
            "aug_kind": "quoted_promo_critique_hard_neg",
        }

    def make_negated_shill_critique(self) -> dict[str, Any]:
        """Generate negated-hype human warnings (`"别信闭眼入全是托"` -> `y_ai=0, y_promo=0`) to break keyword shortcuts."""
        tpl = self.rng.choice(_NEGATED_HYPE_TEMPLATES)
        kw = self.rng.choice(_HYPE_KWS)
        return {
            "text": tpl.format(kw=kw),
            "y_ai": 0.0,
            "y_promo": 0.0,
            "promo_type": "none",
            "aug_kind": "negated_hype_hard_neg",
        }

    def make_fandom_hype_hard_neg(self) -> dict[str, Any]:
        """Generate genuine non-commercial fandom/game/anime/OSS hype (`"闭眼冲/封神/强烈安利"` -> `y_ai=0, y_promo=0`)."""
        return {
            "text": self.rng.choice(_FANDOM_HYPE_HARD_NEGS),
            "y_ai": 0.0,
            "y_promo": 0.0,
            "promo_type": "none",
            "aug_kind": "fandom_hype_hard_neg",
        }

    def make_q2_long_ai_soft_shill(self, pure_ai_paras: list[str], q2_shill_snippet: str) -> dict[str, Any]:
        """Synthesize rare long-form Q2 Dual-High (75% neutral AI discourse + 25% hidden tail AI soft-shill)."""
        body = "\n\n".join(p.strip() for p in pure_ai_paras)
        st = len(body) + 2
        txt = f"{body}\n\n{q2_shill_snippet.strip()}"
        return {
            "text": txt,
            "y_ai": 1.0,
            "y_promo": 1.0,
            "promo_type": "soft_shill",
            "ai_spans": [(0, len(txt))],
            "promo_spans": [(st, len(txt))],
            "aug_kind": "q2_long_ai_soft_shill",
        }


class LightHumanPostEditAug:
    """HART Type-2 Light Human Post-Edit Simulator (AI text lightly humanized: em-dash strip, casual connector/tail; preserves `y_ai=1.0`)."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def light_humanize_ai(self, ai_text: str) -> str:
        out = str(ai_text or "").strip()
        if len(out) < 20:
            return out
        # 1. Strip telltale em-dash (`——`) -> comma
        if "——" in out and self.rng.random() < 0.85:
            out = out.replace("——", "，")
        # 2. Swap 1 formal AI connector with colloquial Chinese discourse marker
        for formal, casuals in _HART_FORMAL_TO_CASUAL:
            if formal in out:
                out = out.replace(formal, self.rng.choice(casuals), 1)
                break
        # 3. Optional light humanizer tail (`[doge]` / `哈哈` / strip final period)
        r = self.rng.random()
        if r < 0.35 and out.endswith("。"):
            out = out[:-1]
        elif r < 0.70:
            out = out + self.rng.choice(("[doge]", "[笑哭]", " 哈哈", " 确实"))
        return out


class ContiguousCadenceCropAug:
    """Extract contiguous 1–2 sentence spans (35–115 chars) from long verified texts without breaking LLM syntax."""

    def __init__(self, seed: int = 20261126):
        self.rng = random.Random(seed)

    def crop(self, text: str, min_chars: int = 32, max_chars: int = 120) -> str | None:
        sents = [s.strip() for s in _SENT_BREAK_PAT.split(str(text or "").strip()) if len(s.strip()) >= 14]
        if len(sents) < 2:
            return None
        cands: list[str] = []
        for i in range(len(sents)):
            if min_chars <= len(sents[i]) <= max_chars:
                cands.append(sents[i])
            if i + 1 < len(sents):
                pair = sents[i] + sents[i + 1]
                if min_chars <= len(pair) <= max_chars:
                    cands.append(pair)
        return self.rng.choice(cands) if cands else None


def orthogonal_dual_axis_pair_loss(
    ai_logit_promo: torch.Tensor,
    ai_logit_organic: torch.Tensor,
    promo_logit_promo: torch.Tensor,
    promo_logit_organic: torch.Tensor,
    promo_is_ai: torch.Tensor,  # float in {0.0, 1.0} (1.0 for Q2 vs Q1 AI pairs, 0.0 for Q3 vs Q4 Human pairs!)
    margin_promo: float = 2.2,
    ai_pin_margin: float = 1.4,
) -> torch.Tensor:
    """Enforce sharp promo-axis separation WHILE locking AI-axis orthogonality (blocks Q3 axis collapse!).

    - Promo axis: `promo_logit(promo_text) - promo_logit(organic_text) >= margin_promo` on ALL pairs.
    - AI axis:
        * When `promo_is_ai == 0` (Q3 Human Promo vs Q4 Organic Human): BOTH must stay strongly negative (`<= -ai_pin_margin`)!
        * When `promo_is_ai == 1` (Q2 AI Shill vs Q1 Pure AI): BOTH must stay strongly positive (`>= +ai_pin_margin`)!
    """
    loss_rank_promo = F.softplus(margin_promo - (promo_logit_promo - promo_logit_organic)).mean()
    # Pin sign: +1 for AI pairs (Q2/Q1), -1 for Human pairs (Q3/Q4)
    sign_ai = 2.0 * promo_is_ai.float() - 1.0
    loss_pin_ai = 0.5 * (
        F.softplus(ai_pin_margin - sign_ai * ai_logit_promo).mean()
        + F.softplus(ai_pin_margin - sign_ai * ai_logit_organic).mean()
    )
    # Invariance penalty: adding/removing promo persuasion inside same-origin text should NOT swing AI logit wildly!
    loss_orth_invar = F.smooth_l1_loss(ai_logit_promo, ai_logit_organic)
    return loss_rank_promo + 0.65 * loss_pin_ai + 0.15 * loss_orth_invar


def run_v11_augment_ci() -> dict[str, Any]:
    """Verify all forensic-safe v1.1.0 augmentations & orthogonal dual-axis pair loss."""
    obf = AdversarialObfuscationAug(seed=1)
    slot = BrandPriceSlotAug(seed=2)
    qs = QuadSpliceAug(seed=3)
    cropper = ContiguousCadenceCropAug(seed=4)
    hart = LightHumanPostEditAug(seed=5)

    p_ex = "【券后299】京东满300-50，关注公众号回复「666」领网盘资料，进群领券！"
    o_ex = "去年京东卖299今年涨到399，真贵。"
    p2, o2 = slot.augment_pair(p_ex, o_ex)
    obf_samples = {obf.obfuscate_promo_or_ai(p_ex, intensity=0.8) for _ in range(16)}
    qpc = qs.make_quoted_promo_critique(p_ex)
    nhc = qs.make_negated_shill_critique()
    fhn = qs.make_fandom_hype_hard_neg()
    long_ai = "第一句是铺垫背景，说明整个系统设计的初衷与约束。然而，真正决定韧性的从来不是纸面参数——而是可追溯的底层边界。综上所述，工程克制最为稀缺。"
    cropped = cropper.crop(long_ai)
    hart_ai = hart.light_humanize_ai(long_ai)

    # Check orthogonal_dual_axis_pair_loss gradient flows cleanly
    ai_p = torch.tensor([-2.0, 2.0], requires_grad=True)
    ai_o = torch.tensor([-2.2, 1.8], requires_grad=True)
    pr_p = torch.tensor([2.5, 2.4], requires_grad=True)
    pr_o = torch.tensor([-2.0, -1.9], requires_grad=True)
    is_ai = torch.tensor([0.0, 1.0])
    loss = orthogonal_dual_axis_pair_loss(ai_p, ai_o, pr_p, pr_o, is_ai)
    loss.backward()
    grad_ok = all(t.grad is not None and torch.isfinite(t.grad).all().item() for t in (ai_p, ai_o, pr_p, pr_o))
    with torch.no_grad():
        loss_bad = float(orthogonal_dual_axis_pair_loss(-ai_p, -ai_o, -pr_p, -pr_o, is_ai).item())

    ok = bool(
        len(obf_samples) >= 4
        and len(p2) >= 10 and len(o2) >= 8
        and qpc["y_ai"] == 0.0 and qpc["y_promo"] == 0.0
        and nhc["y_ai"] == 0.0 and nhc["y_promo"] == 0.0
        and fhn["y_ai"] == 0.0 and fhn["y_promo"] == 0.0
        and cropped is not None and 32 <= len(cropped) <= 120
        and hart_ai != long_ai and len(hart_ai) >= 30
        and grad_ok and float(loss.item()) < 0.50 and loss_bad > 4.0
    )
    return {
        "obf_variants": len(obf_samples),
        "cropped_len": len(cropped) if cropped else 0,
        "hart_changed": hart_ai != long_ai,
        "orth_pair_loss_aligned": round(float(loss.item()), 5),
        "orth_pair_loss_inverted": round(loss_bad, 5),
        "grad_ok": grad_ok,
        "pass": ok,
    }
