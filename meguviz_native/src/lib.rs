use ahash::{AHashMap, AHashSet};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Constants (bit-exact with MeguvizCAC v0.7 / v1.1 / v1.3 / v1.4 / v1.5 / v1.6)
// ---------------------------------------------------------------------------
const R0_B1_V130: usize = 4096;
const R0_B2_V130: usize = 16384;
const R0_B3_V130: usize = 8192;
const R0_STRUCT_DIM_V140: usize = 17;
const CHARLM_FEAT_DIM_V140: usize = 9;
const R0_FEAT_DIM_V140: usize = R0_STRUCT_DIM_V140 + CHARLM_FEAT_DIM_V140; // 26
const CHARLM_BITS_V140: u64 = 12;
const CHARLM_ORDER_V140: usize = 4;
const CHARLM_SMOOTH_V140: [f64; 4] = [0.0, 8.0, 8.0, 12.0];
const LOCAL_CHARLM_DIM_V150: usize = 8;
const QUOTE_SCOPE_DIM_V161: usize = 4;
const STYLE_FEAT_DIM_V11: usize = 56;

// ---------------------------------------------------------------------------
// Compiled Static Regexes (zero-lock after first init)
// ---------------------------------------------------------------------------
struct RegexBank {
    reply_hdr_v150: Regex,
    inline_mention_v150: Regex,
    reply_hdr_v07: Regex,
    inline_mention_v07: Regex,
    contact_id_pii: Regex,
    pii_placeholder: Regex,
    quote_span: Regex,
    long_quote_span: Regex,
    bili_emote_v11: Regex,
    kaomoji_v11: Regex,
    cjk_space: Regex,
    emdash: Regex,
    book_quote: Regex,
    scorecard: Regex,
    list_mark: Regex,
    sent_split: Regex,
    upper_tech: Regex,
    md_scaffold: Regex,
    quote_intro: Regex,
    promo_price: Regex,
    promo_multi_sku: Regex,
    promo_cta: Regex,
    promo_shill: Regex,
    promo_kedaibiao: Regex,
    promo_en_self: Regex,
    opensource_url: Regex,
    commerce_url: Regex,
}

fn rx() -> &'static RegexBank {
    static BANK: OnceLock<RegexBank> = OnceLock::new();
    BANK.get_or_init(|| RegexBank {
        reply_hdr_v150: Regex::new(r"^\s*回[复覆]\s*@[^\s:：]+\s*[:：]\s*").unwrap(),
        inline_mention_v150: Regex::new(r"@[^\s,，。！!？?@:：()（）\[\]【】]+").unwrap(),
        reply_hdr_v07: Regex::new(r"^\s*回[复覆]\s*@[^\s:：]{1,36}\s*[:：]\s*").unwrap(),
        inline_mention_v07: Regex::new(r"@[^\s,，。！!？?@:：()（）\[\]【】]{1,32}").unwrap(),
        contact_id_pii: Regex::new(r"(?i)((?:加微|加v|vx|v信|薇信|威信|微信|绿泡泡|扣群|q群|qq群?|交流\d*群|粉丝[群裙]|进[群裙]|君羊|小红书|xhs|助理(?:老师)?|微信号|扣扣|🐧|🛰️)\s*[:：号是]?\s*)[A-Za-z0-9_\-]{5,20}").unwrap(),
        pii_placeholder: Regex::new(r"<(?:MENTION|URL|CONTACT_ID|REPLY_HDR|BOS|EOS|PAD|SEP|UNK)>").unwrap(),
        quote_span: Regex::new(r#"[「『“"《][^」』”"》]{4,120}[」』”"》]"#).unwrap(),
        long_quote_span: Regex::new(r#"(?:[「『“"][^\n」』”"]{8,600}[」』”"]|```[\s\S]{12,800}```)"#).unwrap(),
        bili_emote_v11: Regex::new(r"\[[^\s\[\]]{1,24}\]").unwrap(),
        kaomoji_v11: Regex::new(r"[(\[（【][^()\n\u{4e00}-\u{9fff}A-Za-z0-9]{2,12}[)\]）】]").unwrap(),
        cjk_space: Regex::new(r"[\u{4e00}-\u{9fff}][ \t]+[\u{4e00}-\u{9fff}]").unwrap(),
        emdash: Regex::new(r"——|—|–|--").unwrap(),
        book_quote: Regex::new(r"《[^》]{1,40}》").unwrap(),
        scorecard: Regex::new(r"\d+(?:\.\d+)?\s*/\s*(?:10|5|100)|(?:评分|打分|星级|推荐度)").unwrap(),
        list_mark: Regex::new(r"(?:^|[\n，。；;\s])(?:[1-9][0-9]?[\.、\)）]|[①②③④⑤⑥⑦⑧⑨⑩]|[-•\*]\s+|#{1,4}\s+)").unwrap(),
        sent_split: Regex::new(r"[。！？!?；;\n]+").unwrap(),
        upper_tech: Regex::new(r"\b[A-Z][A-Z0-9_]{1,12}\b").unwrap(),
        md_scaffold: Regex::new(r"(?:^|\n)\s*(?:#{1,4}\s|\*\*.{2,24}\*\*[:：]?|首先|其次|再次|最后|总而言之|综上所述|值得注意的是|另一方面|第一[、.：:]|第二[、.：:]|第三[、.：:])").unwrap(),
        quote_intro: Regex::new(r#"(?:原文如下|引用一下|贴一段|复制一段|有人发了|AI生成的|大模型写的|ChatGPT说|DeepSeek说|机翻原文|原话是|评论区看到的)[:：]?\s*[「『“"《]?"#).unwrap(),
        promo_price: Regex::new(r"(?:券后|领[券劵]|券\d+元|满\d+减\d+|到手价|史低|直降|百亿补贴|国补|秒杀价|折合|实付\d+|神券|优惠[券劵]|返佣|蓝链|蓝色文字|小蓝字|置顶链接|评论区置顶|凑单|抄作业)").unwrap(),
        promo_multi_sku: Regex::new(r"(?:[|｜/]\s*(?:券|满|到手|\d+元|华为|小米|索尼|苹果|OPPO|vivo|红米|罗技))").unwrap(),
        promo_cta: Regex::new(r#"(?i)(?:关注一下|求关注|互关|互粉|点个关注|私信领|私信发|私信我|思信|滴我|回复[「“"].{1,8}[」”"]|免费领|戳我主页|看我主页|主页置顶|主页橱窗|简介自取|加微|加v|vx|薇信|威信|绿泡泡|扣群|粉丝[群裙]|进[群裙]|君羊|公[众粽]号|工种号|功重号|\bgzh\b|\bxhs\b|接单|代做|晚八点|手慢无|冲就完了|🛰️|🐧|🍠|<CONTACT_ID>)"#).unwrap(),
        promo_shill: Regex::new(r"(?:亲测有效|亲测好用|闭眼入|闭眼冲|无脑入|无脑冲|入手不亏|回购第|已经回购|不踩雷|早买早享受|吊打.{1,8}牌|性价比无敌|质感封神|良心推荐|强烈安利|真心安利|强烈建议入手)").unwrap(),
        promo_kedaibiao: Regex::new(r"(?:课代表|省流总结|空降\s*\d|免责声明.{0,12}AI|AI(?:辅助)?(?:生成|整理|提炼).{0,12}仅供)").unwrap(),
        promo_en_self: Regex::new(r"(?i)(?:check my channel|subscribe to my|please support|new to bilibili|my artwork|follow my channel)").unwrap(),
        opensource_url: Regex::new(r"(?i)(?:github\.com|gitee\.com|arxiv\.org|huggingface\.co|docs\.python\.org|pytorch\.org|wikipedia\.org|zhihu\.com/question|stackoverflow\.com|开源仓库|开源项目|官方文档)").unwrap(),
        commerce_url: Regex::new(r"(?i)(?:taobao\.com|tmall\.com|jd\.com|pinduoduo\.com|yangkeduo\.com|goofish\.com|1688\.com|tb\.cn|u\.jd\.com|pan\.quark\.cn|pan\.baidu\.com|拼多多|京东自营|淘宝搜|天猫超市|闲鱼同款)").unwrap(),
    })
}

// ---------------------------------------------------------------------------
// 1. Zero-PII Canonicalizer (bit-exact with v07 + v11 + v130 + v150!)
// ---------------------------------------------------------------------------
#[derive(Default, Clone, Copy)]
struct CanonMeta {
    has_reply_hdr: f32,
    mention_cnt: f32,
    homoglyph_ratio: f32,
    invisible_log: f32,
    astral_math_ratio: f32,
    bidi_flag: f32,
    fullwidth_ratio: f32,
    astral_emoji_log: f32,
    spoof_any: f32,
}

#[inline]
fn is_zero_width(cp: u32) -> bool {
    matches!(
        cp,
        0x200B | 0x200C | 0x200D | 0x200E | 0x200F | 0x2060 | 0xFEFF | 0x00AD | 0x034F | 0x061C
    ) || (0xE0000..=0xE007F).contains(&cp)
        || (0xFE00..=0xFE0F).contains(&cp)
        || (0xE0100..=0xE01EF).contains(&cp)
}

#[inline]
fn is_bidi_override(cp: u32) -> bool {
    matches!(
        cp,
        0x202A | 0x202B | 0x202C | 0x202D | 0x202E | 0x2066 | 0x2067 | 0x2068 | 0x2069
    )
}

#[inline]
fn fold_math_alphanum(cp: u32) -> Option<char> {
    if !(0x1D400..=0x1D7FF).contains(&cp) {
        return None;
    }
    if (0x1D7CE..=0x1D7FF).contains(&cp) {
        return char::from_u32(('0' as u32) + ((cp - 0x1D7CE) % 10));
    }
    if cp <= 0x1D6A3 {
        let off = (cp - 0x1D400) % 52;
        return if off < 26 {
            char::from_u32(('A' as u32) + off)
        } else {
            char::from_u32(('a' as u32) + (off - 26))
        };
    }
    None
}

#[inline]
fn confusable_latin(ch: char) -> Option<char> {
    match ch {
        'а' => Some('a'), 'е' => Some('e'), 'о' => Some('o'), 'р' => Some('p'),
        'с' => Some('c'), 'у' => Some('y'), 'х' => Some('x'), 'і' => Some('i'),
        'ј' => Some('j'), 'ѕ' => Some('s'), 'ԁ' => Some('d'),
        'А' => Some('A'), 'В' => Some('B'), 'Е' => Some('E'), 'К' => Some('K'),
        'М' => Some('M'), 'Н' => Some('H'), 'О' => Some('O'), 'Р' => Some('P'),
        'С' => Some('C'), 'Т' => Some('T'), 'Х' => Some('X'), 'І' => Some('I'),
        'Ј' => Some('J'), 'Ѕ' => Some('S'),
        'α' => Some('a'), 'ο' => Some('o'), 'ρ' => Some('p'), 'ν' => Some('v'),
        'κ' => Some('k'), 'χ' => Some('x'), 'ι' => Some('i'), 'ε' => Some('e'),
        'Α' => Some('A'), 'Β' => Some('B'), 'Ε' => Some('E'), 'Ζ' => Some('Z'),
        'Η' => Some('H'), 'Ι' => Some('I'), 'Κ' => Some('K'), 'Μ' => Some('M'),
        'Ν' => Some('N'), 'Ο' => Some('O'), 'Ρ' => Some('P'), 'Τ' => Some('T'),
        'Υ' => Some('Y'), 'Χ' => Some('X'),
        _ => None,
    }
}

fn canonicalize_v07_single(text: &str, max_chars: usize) -> (String, CanonMeta) {
    let bank = rx();
    let lim = max_chars.saturating_mul(2);
    let truncated: String = text.chars().take(lim).collect();
    let mut raw = truncated;

    let mut has_reply_hdr = 0.0f32;
    if bank.reply_hdr_v07.is_match(&raw) {
        has_reply_hdr = 1.0;
        raw = bank.reply_hdr_v07.replace(&raw, "<REPLY_HDR> ").into_owned();
    }
    let mention_cnt = bank.inline_mention_v07.find_iter(&raw).count();
    if mention_cnt > 0 {
        raw = bank.inline_mention_v07.replace_all(&raw, "<MENTION>").into_owned();
    }

    let mut has_latin_or_cjk = false;
    let mut cyr_grk_total = 0usize;
    for ch in raw.chars() {
        let cp = ch as u32;
        if (0x0041..=0x005A).contains(&cp)
            || (0x0061..=0x007A).contains(&cp)
            || (0x4E00..=0x9FFF).contains(&cp)
        {
            has_latin_or_cjk = true;
        } else if (0x0400..=0x052F).contains(&cp) || (0x0370..=0x03FF).contains(&cp) {
            cyr_grk_total += 1;
        }
    }
    let mixed_script_context = has_latin_or_cjk && cyr_grk_total > 0;

    let mut out = String::with_capacity(raw.len());
    let mut out_items = 0usize;
    let mut invisible_cnt = 0usize;
    let mut bidi_flag = 0.0f32;
    let mut astral_math_cnt = 0usize;
    let mut homoglyph_cnt = 0usize;
    let mut cjk_compat_cnt = 0usize;
    let mut fullwidth_cnt = 0usize;
    let mut astral_emoji_cnt = 0usize;
    let mut vis_len = 0usize;

    for ch in raw.chars() {
        let cp = ch as u32;
        if is_bidi_override(cp) {
            bidi_flag = 1.0;
            invisible_cnt += 1;
            continue;
        }
        if is_zero_width(cp) {
            invisible_cnt += 1;
            continue;
        }
        if let Some(ma) = fold_math_alphanum(cp) {
            astral_math_cnt += 1;
            out.push(ma);
            out_items += 1;
            vis_len += 1;
            continue;
        }
        if (0xFF01..=0xFF5E).contains(&cp) {
            fullwidth_cnt += 1;
            if let Some(c2) = char::from_u32(cp - 0xFEE0) {
                out.push(c2);
                out_items += 1;
            }
            vis_len += 1;
            continue;
        }
        if (0x2F00..=0x2FD5).contains(&cp)
            || (0xF900..=0xFAFF).contains(&cp)
            || (0x2F800..=0x2FA1F).contains(&cp)
        {
            cjk_compat_cnt += 1;
            for nc in ch.nfkc() {
                out.push(nc);
            }
            out_items += 1;
            vis_len += 1;
            continue;
        }
        if mixed_script_context {
            if let Some(mapped) = confusable_latin(ch) {
                homoglyph_cnt += 1;
                out.push(mapped);
                out_items += 1;
                vis_len += 1;
                continue;
            }
        }
        if cp > 0xFFFF {
            astral_emoji_cnt += 1;
            out.push(ch);
            out_items += 1;
            vis_len += 1;
            continue;
        }
        out.push(ch);
        out_items += 1;
        vis_len += 1;
        if out_items >= max_chars {
            break;
        }
    }

    let clean = out.trim().to_string();
    let denom = (vis_len.max(1)) as f64;
    let spoof_any = if homoglyph_cnt > 0
        || invisible_cnt > 0
        || astral_math_cnt > 0
        || bidi_flag > 0.0
        || cjk_compat_cnt > 0
    {
        1.0f32
    } else {
        0.0f32
    };
    let meta = CanonMeta {
        has_reply_hdr,
        mention_cnt: mention_cnt as f32,
        homoglyph_ratio: (((homoglyph_cnt + cjk_compat_cnt) as f64 / denom) * 5.0).min(1.0) as f32,
        invisible_log: ((invisible_cnt as f64).ln_1p() / 3.0).min(1.0) as f32,
        astral_math_ratio: ((astral_math_cnt as f64 / denom) * 4.0).min(1.0) as f32,
        bidi_flag,
        fullwidth_ratio: ((fullwidth_cnt as f64 / denom) * 3.0).min(1.0) as f32,
        astral_emoji_log: ((astral_emoji_cnt as f64).ln_1p() / 3.0).min(1.0) as f32,
        spoof_any,
    };
    (clean, meta)
}

fn canonicalize_v150_single(text: &str, max_chars: usize) -> (String, CanonMeta) {
    let bank = rx();
    let mut s = text.to_string();
    if bank.reply_hdr_v150.is_match(&s) {
        s = bank.reply_hdr_v150.replace(&s, "<REPLY_HDR> ").into_owned();
    }
    if s.contains('@') {
        s = bank.inline_mention_v150.replace_all(&s, "<MENTION>").into_owned();
    }
    let (cl, meta) = canonicalize_v07_single(&s, max_chars);
    let cl2 = bank.contact_id_pii.replace_all(&cl, "${1}<CONTACT_ID>").into_owned();
    (cl2, meta)
}

#[inline]
fn char_script_bucket(cp: u32) -> u8 {
    if (0x4E00..=0x9FFF).contains(&cp) || (0x3400..=0x4DBF).contains(&cp) || (0xF900..=0xFAFF).contains(&cp) {
        1
    } else if (65..=90).contains(&cp) || (97..=122).contains(&cp) {
        2
    } else if (48..=57).contains(&cp) {
        3
    } else if (0x3040..=0x30FF).contains(&cp) {
        4
    } else if (0xAC00..=0xD7AF).contains(&cp) {
        5
    } else {
        0
    }
}

fn compute_mattr_64(compact: &[char]) -> (f64, f64) {
    let n = compact.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let window = 64usize;
    let stride = 32usize;
    let mut seen: AHashSet<char> = AHashSet::with_capacity(64);
    if n <= window {
        for &c in compact {
            seen.insert(c);
        }
        return (seen.len() as f64 / n as f64, 0.0);
    }
    let mut vals: Vec<f64> = Vec::with_capacity((n - window) / stride + 1);
    let mut st = 0usize;
    while st + window <= n {
        seen.clear();
        for &c in &compact[st..st + window] {
            seen.insert(c);
        }
        vals.push(seen.len() as f64 / window as f64);
        st += stride;
    }
    if vals.is_empty() {
        seen.clear();
        for &c in compact {
            seen.insert(c);
        }
        return (seen.len() as f64 / n as f64, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
    (mean, var.sqrt())
}

fn compute_repeat_4gram_ratio(compact: &[char]) -> f64 {
    let n = compact.len();
    if n < 12 {
        return 0.0;
    }
    let total = n - 3;
    let mut counts: AHashMap<[char; 4], u32> = AHashMap::with_capacity(total);
    for i in 0..total {
        let key = [compact[i], compact[i + 1], compact[i + 2], compact[i + 3]];
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut rep = 0u64;
    for &v in counts.values() {
        if v > 1 {
            rep += (v - 1) as u64;
        }
    }
    ((rep as f64 / total.max(1) as f64) * 2.5).min(1.0)
}

fn compute_head_tail_lex_drift(compact: &[char]) -> f64 {
    let n = compact.len();
    if n < 80 {
        return 0.0;
    }
    let k = 24usize.max((n as f64 * 0.35) as usize);
    let h = &compact[..k];
    let t = &compact[n - k..];
    let mut bg_h: AHashSet<(char, char)> = AHashSet::with_capacity(k);
    let mut bg_t: AHashSet<(char, char)> = AHashSet::with_capacity(k);
    for i in 0..h.len().saturating_sub(1) {
        bg_h.insert((h[i], h[i + 1]));
    }
    for i in 0..t.len().saturating_sub(1) {
        bg_t.insert((t[i], t[i + 1]));
    }
    if bg_h.is_empty() || bg_t.is_empty() {
        return 0.0;
    }
    let inter = bg_h.intersection(&bg_t).count();
    let uni = bg_h.len() + bg_t.len() - inter;
    1.0 - (inter as f64 / uni as f64)
}

fn extract_style_56d_from_text(text: &str, max_chars: usize) -> [f32; STYLE_FEAT_DIM_V11] {
    let bank = rx();
    let (cl0, meta) = canonicalize_v07_single(text, max_chars);
    let clean = bank.contact_id_pii.replace_all(&cl0, "${1}<CONTACT_ID>").into_owned();
    let clean_char_len = clean.chars().count();

    let emote_cnt = bank.bili_emote_v11.find_iter(&clean).count();
    let clean_no_emote = if emote_cnt > 0 {
        bank.bili_emote_v11.replace_all(&clean, "").into_owned()
    } else {
        clean.clone()
    };

    // Paragraph structure
    let mut p_lens: Vec<usize> = Vec::new();
    for seg in clean_no_emote.split('\n') {
        let tr = seg.trim();
        if !tr.is_empty() {
            p_lens.push(tr.chars().count());
        }
    }
    let n_paras = p_lens.len().max(1);
    if p_lens.is_empty() {
        p_lens.push(clean_no_emote.chars().count().max(1));
    }
    let p_mean = p_lens.iter().sum::<usize>() as f64 / p_lens.len() as f64;
    let p_var = p_lens.iter().map(|&x| {
        let d = x as f64 - p_mean;
        d * d
    }).sum::<f64>() / p_lens.len() as f64;
    let p_std = p_var.sqrt();
    let para_len_cv = if n_paras >= 2 { (p_std / (p_mean + 1e-3)).min(2.0) } else { 0.0 };

    // Quote coverage ratio (note: Python m.end() - m.start() is CHAR count!)
    let mut quoted_chars = 0usize;
    let mut found_long = false;
    for m in bank.long_quote_span.find_iter(&clean) {
        found_long = true;
        quoted_chars += m.as_str().chars().count();
    }
    if !found_long {
        for m in bank.quote_span.find_iter(&clean) {
            quoted_chars += m.as_str().chars().count();
        }
    }
    let quote_ratio = (quoted_chars as f64 / clean_char_len.max(1) as f64).min(1.0);

    // NFKC + whitespace collapse
    let nfkc: String = clean_no_emote.nfkc().collect();
    let mut norm = String::with_capacity(nfkc.len());
    for (idx, tok) in nfkc.split_whitespace().enumerate() {
        if idx > 0 {
            norm.push(' ');
        }
        norm.push_str(tok);
    }
    let n_ch = norm.chars().count().max(1);
    let per100 = (n_ch as f64) / 100.0 + 0.5;

    let mut compact: Vec<char> = Vec::with_capacity(n_ch);
    for ch in norm.chars() {
        if !ch.is_whitespace() {
            for lc in ch.to_lowercase() {
                compact.push(lc);
            }
        }
    }

    // Byte Shannon entropy
    let raw_bytes = norm.as_bytes();
    let n_b = raw_bytes.len().max(1);
    let mut b_counts = [0u32; 256];
    for &b in raw_bytes {
        b_counts[b as usize] += 1;
    }
    let inv_nb = 1.0 / (n_b as f64);
    let mut entropy = 0.0f64;
    for &cnt in &b_counts {
        if cnt > 0 {
            let p = (cnt as f64) * inv_nb;
            entropy -= p * p.log2();
        }
    }

    // Single-pass script stats + punctuation counts on norm
    let mut cjk = 0usize;
    let mut ascii_a = 0usize;
    let mut digits = 0usize;
    let mut kana = 0usize;
    let mut hangul = 0usize;
    let mut switches = 0usize;
    let mut prev_b = 0u8;
    let mut colons = 0usize;
    let mut commas = 0usize;
    let mut periods = 0usize;
    let mut q_exc = 0usize;
    let mut casual_ch = 0usize;
    let mut br_l = [0usize; 5];
    let mut br_r = [0usize; 5];

    for ch in norm.chars() {
        let b = char_script_bucket(ch as u32);
        match b {
            1 => cjk += 1,
            2 => ascii_a += 1,
            3 => digits += 1,
            4 => kana += 1,
            5 => hangul += 1,
            _ => {}
        }
        if matches!(b, 1 | 2 | 4 | 5) && matches!(prev_b, 1 | 2 | 4 | 5) && b != prev_b {
            switches += 1;
        }
        if b != 0 {
            prev_b = b;
        }
        match ch {
            ':' | '：' => colons += 1,
            ',' | '，' | '、' => commas += 1,
            '.' | '。' => periods += 1,
            '?' | '？' | '!' | '！' => q_exc += 1,
            '~' | '～' | '…' | '啦' | '呢' | '呀' | '哇' | '咯' | '嘛' | '草' => casual_ch += 1,
            '(' => br_l[0] += 1, ')' => br_r[0] += 1,
            '（' => br_l[1] += 1, '）' => br_r[1] += 1,
            '《' => br_l[2] += 1, '》' => br_r[2] += 1,
            '“' => br_l[3] += 1, '”' => br_r[3] += 1,
            '[' => br_l[4] += 1, ']' => br_r[4] += 1,
            _ => {}
        }
    }
    let casual_tone = casual_ch + norm.matches("哈哈").count();

    let mut tot_br = 0usize;
    let mut matched_br = 0usize;
    for k in 0..5 {
        tot_br += br_l[k] + br_r[k];
        matched_br += 2 * br_l[k].min(br_r[k]);
    }
    let punct_closure = if tot_br > 0 { matched_br as f64 / tot_br as f64 } else { 0.0 };

    let (mattr_64, mattr_std) = compute_mattr_64(&compact);
    let rep4_ratio = compute_repeat_4gram_ratio(&compact);
    let ht_drift = compute_head_tail_lex_drift(&compact);

    let cjk_spaces = bank.cjk_space.find_iter(&clean_no_emote).count();
    let emdashes = bank.emdash.find_iter(&norm).count();
    let book_quotes = bank.book_quote.find_iter(&norm).count();
    let scorecards = bank.scorecard.find_iter(&norm).count();
    let list_marks = bank.list_mark.find_iter(&clean_no_emote).count();
    let md_scaffolds = bank.md_scaffold.find_iter(&clean_no_emote).count();
    let kaomojis = bank.kaomoji_v11.find_iter(&clean_no_emote).count();
    let upper_tech = bank.upper_tech.find_iter(&norm).count();

    let mut s_lens: Vec<usize> = Vec::new();
    for seg in bank.sent_split.split(&clean_no_emote) {
        let tr = seg.trim();
        if !tr.is_empty() {
            s_lens.push(tr.chars().count());
        }
    }
    let n_sents = s_lens.len().max(1);
    if s_lens.is_empty() {
        s_lens.push(n_ch);
    }
    let s_mean = s_lens.iter().sum::<usize>() as f64 / s_lens.len() as f64;
    let s_var = s_lens.iter().map(|&x| {
        let d = x as f64 - s_mean;
        d * d
    }).sum::<f64>() / s_lens.len() as f64;
    let s_std = s_var.sqrt();
    let sent_cv = (s_std / (s_mean + 1e-3)).min(2.0);

    let p_price = bank.promo_price.find_iter(&clean).count();
    let p_sku = bank.promo_multi_sku.find_iter(&clean).count();
    let p_cta = bank.promo_cta.find_iter(&clean).count();
    let p_shill = bank.promo_shill.find_iter(&clean).count();
    let p_kdb = if bank.promo_kedaibiao.is_match(&clean) { 1.0f32 } else { 0.0f32 };
    let p_en_self = if bank.promo_en_self.is_match(&clean) { 1.0f32 } else { 0.0f32 };
    let has_os_url = if bank.opensource_url.is_match(&clean) { 1.0f32 } else { 0.0f32 };
    let has_comm_url = if bank.commerce_url.is_match(&clean) { 1.0f32 } else { 0.0f32 };
    let has_quote_intro = if bank.quote_intro.is_match(&clean) { 1.0f32 } else { 0.0f32 };

    let rate = |cnt: usize| -> f32 { (((cnt as f64) / per100).ln_1p() / 2.0).min(1.5) as f32 };
    let n_ch_f = n_ch as f64;

    [
        ((n_ch_f).ln_1p() / 6.0).min(1.8) as f32,
        ((n_b as f64).ln_1p() / 6.5).min(1.8) as f32,
        (entropy / 8.0).min(1.0) as f32,
        mattr_64 as f32,
        (cjk as f64 / n_ch_f) as f32,
        (ascii_a as f64 / n_ch_f) as f32,
        (digits as f64 / n_ch_f) as f32,
        (kana as f64 / n_ch_f) as f32,
        (hangul as f64 / n_ch_f) as f32,
        ((switches as f64 / n_ch_f) * 6.0).min(1.0) as f32,
        (cjk_spaces as f64 / (n_ch_f / 10.0).max(1.0)).min(1.0) as f32,
        rate(emdashes),
        rate(colons),
        punct_closure as f32,
        rate(book_quotes),
        rate(scorecards),
        rate(list_marks),
        has_os_url,
        ((emote_cnt as f64).ln_1p() / 2.5).min(1.0) as f32,
        rate(kaomojis),
        meta.has_reply_hdr,
        ((n_sents as f64).ln_1p() / 3.5).min(1.5) as f32,
        (s_mean.ln_1p() / 5.0).min(1.5) as f32,
        (s_std.ln_1p() / 4.5).min(1.5) as f32,
        (commas as f64 / (commas + periods + 1) as f64).min(1.0) as f32,
        rate(q_exc),
        rate(casual_tone),
        rate(upper_tech),
        meta.invisible_log,
        meta.homoglyph_ratio,
        meta.fullwidth_ratio,
        meta.astral_math_ratio,
        // 8-d spoof/reply/quote/astral
        meta.bidi_flag,
        meta.spoof_any,
        (meta.mention_cnt / 3.0).min(1.0),
        if quote_ratio > 0.02 { 1.0 } else { 0.0 },
        quote_ratio as f32,
        meta.astral_emoji_log,
        (emote_cnt as f64 / (n_ch_f / 15.0).max(1.0)).min(1.0) as f32,
        has_quote_intro,
        // 8-d Long-doc macro
        ((n_paras as f64).ln_1p() / 2.5).min(1.5) as f32,
        para_len_cv.min(1.5) as f32,
        sent_cv.min(1.5) as f32,
        (mattr_std * 5.0).min(1.0) as f32,
        ht_drift as f32,
        rep4_ratio as f32,
        rate(md_scaffolds),
        (if n_ch > 256 { ((n_ch - 256) as f64).ln_1p() / 7.0 } else { 0.0 }).min(1.0) as f32,
        // 8-d Promo
        rate(p_price),
        (p_sku as f64 / 3.0).min(1.0) as f32,
        rate(p_cta),
        rate(p_shill),
        p_kdb,
        has_comm_url,
        p_en_self,
        (has_comm_url - has_os_url).max(0.0),
    ]
}

// ---------------------------------------------------------------------------
// 2. CompactCharLMV150 Native Engine (Single + Multi-Fold OOF Set!)
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct CharLMTable {
    keys: Vec<u64>,
    counts: Vec<[u32; 2]>,
    contexts: Vec<u64>,
    den: Vec<[u32; 2]>,
}

impl CharLMTable {
    #[inline]
    fn lookup_joint(&self, q: u64) -> [f64; 2] {
        match self.keys.binary_search(&q) {
            Ok(idx) => {
                let c = self.counts[idx];
                [c[0] as f64, c[1] as f64]
            }
            Err(_) => [0.0, 0.0],
        }
    }

    #[inline]
    fn lookup_den(&self, ctx: u64) -> [f64; 2] {
        match self.contexts.binary_search(&ctx) {
            Ok(idx) => {
                let d = self.den[idx];
                [d[0] as f64, d[1] as f64]
            }
            Err(_) => [0.0, 0.0],
        }
    }
}

#[derive(Clone)]
struct CharLMCore {
    ascii_vocab: [u16; 128],
    vocab_map: AHashMap<char, u16>,
    unigrams: Vec<[f64; 2]>, // [4096, 2]
    tables: Vec<CharLMTable>, // 3 tables (orders 2,3,4)
    feat_mean: [f32; 33],
    feat_scale: [f32; 33],
    coef: [f32; 33],
    bias: f32,
    max_chars: usize,
}

impl CharLMCore {
    #[inline]
    fn char_id(&self, ch: char) -> u64 {
        let cp = ch as u32;
        if cp < 128 {
            self.ascii_vocab[cp as usize] as u64
        } else {
            *self.vocab_map.get(&ch).unwrap_or(&1) as u64
        }
    }

    fn extract_9d_one(&self, clean_text: &str) -> [f32; CHARLM_FEAT_DIM_V140] {
        let mut ids: Vec<u64> = clean_text.chars().take(self.max_chars).map(|c| self.char_id(c)).collect();
        if ids.is_empty() {
            ids.push(self.char_id('\0'));
        }
        let n = ids.len();
        let inv_n = 1.0 / (n as f64);

        let mut p: Vec<[f64; 2]> = ids.iter().map(|&id| self.unigrams[id as usize]).collect();
        let mut keys_prev: Vec<u64> = ids.clone();
        let mut raw33 = [0.0f32; 33];

        for order in 0..CHARLM_ORDER_V140 {
            if order > 0 {
                let tb = &self.tables[order - 1];
                let sm = CHARLM_SMOOTH_V140[order];
                let shift = CHARLM_BITS_V140 * (order as u64);
                for i in 0..n {
                    let lag_id = if i >= order { ids[i - order] } else { 0 };
                    let k = keys_prev[i] | (lag_id << shift);
                    keys_prev[i] = k;
                    let joint = tb.lookup_joint(k);
                    let den = tb.lookup_den(k >> CHARLM_BITS_V140);
                    p[i][0] = (joint[0] + sm * p[i][0]) / (den[0] + sm);
                    p[i][1] = (joint[1] + sm * p[i][1]) / (den[1] + sm);
                }
            }
            let mut s_llr = 0.0f64;
            let mut s_pos = 0.0f64;
            let mut s_neg = 0.0f64;
            let mut s_llr2 = 0.0f64;
            let mut s_lp0 = 0.0f64;
            let mut s_lp1 = 0.0f64;
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for pi in &p {
                let lp0 = pi[0].max(1e-12).ln();
                let lp1 = pi[1].max(1e-12).ln();
                let llr = lp1 - lp0;
                s_llr += llr;
                if llr > 0.0 {
                    s_pos += llr;
                } else {
                    s_neg += llr;
                }
                s_llr2 += llr * llr;
                s_lp0 += lp0;
                s_lp1 += lp1;
                if llr < mn {
                    mn = llr;
                }
                if llr > mx {
                    mx = llr;
                }
            }
            let avg = s_llr * inv_n;
            let std = ((s_llr2 * inv_n) - avg * avg).max(0.0).sqrt();
            let base = order * 8;
            raw33[base] = avg as f32;
            raw33[base + 1] = (s_pos * inv_n) as f32;
            raw33[base + 2] = (s_neg * inv_n) as f32;
            raw33[base + 3] = std as f32;
            raw33[base + 4] = (s_lp0 * inv_n) as f32;
            raw33[base + 5] = (s_lp1 * inv_n) as f32;
            raw33[base + 6] = mn as f32;
            raw33[base + 7] = mx as f32;
        }
        raw33[32] = (n as f64).ln_1p() as f32;

        let mut z_raw = self.bias;
        for j in 0..33 {
            let xs = ((raw33[j] - self.feat_mean[j]) / self.feat_scale[j]).clamp(-8.0, 8.0);
            z_raw += xs * self.coef[j];
        }
        let z_soft = 4.0f32 * (z_raw / 4.0f32).tanh();
        [
            raw33[0].clamp(-3.0, 3.0),
            raw33[8].clamp(-3.0, 3.0),
            raw33[16].clamp(-3.0, 3.0),
            raw33[24].clamp(-3.0, 3.0),
            raw33[25].clamp(0.0, 3.0),
            raw33[26].clamp(-3.0, 0.0),
            raw33[27].clamp(0.0, 3.0),
            (raw33[31] / 3.0).clamp(-2.0, 2.0),
            z_soft,
        ]
    }

    fn score_char_span(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        keep_from: usize,
        char_llr: &mut [[f32; 4]],
        char_hit: &mut [[f32; 4]],
    ) {
        if end <= start {
            return;
        }
        let sub = &chars[start..end];
        let l = sub.len();
        let ids: Vec<u64> = sub.iter().map(|&c| self.char_id(c)).collect();
        let mut p: Vec<[f64; 2]> = ids.iter().map(|&id| self.unigrams[id as usize]).collect();
        let mut keys_prev: Vec<u64> = ids.clone();
        let off = keep_from - start;

        for order in 0..CHARLM_ORDER_V140 {
            if order == 0 {
                for i in off..l {
                    let dst = keep_from + (i - off);
                    char_hit[dst][0] = if ids[i] >= 2 { 1.0 } else { 0.0 };
                    let lp0 = p[i][0].max(1e-12).ln();
                    let lp1 = p[i][1].max(1e-12).ln();
                    char_llr[dst][0] = (lp1 - lp0).clamp(-3.0, 3.0) as f32;
                }
            } else {
                let tb = &self.tables[order - 1];
                let sm = CHARLM_SMOOTH_V140[order];
                let shift = CHARLM_BITS_V140 * (order as u64);
                for i in 0..l {
                    let lag_id = if i >= order { ids[i - order] } else { 0 };
                    let k = keys_prev[i] | (lag_id << shift);
                    keys_prev[i] = k;
                    let joint = tb.lookup_joint(k);
                    let den = tb.lookup_den(k >> CHARLM_BITS_V140);
                    p[i][0] = (joint[0] + sm * p[i][0]) / (den[0] + sm);
                    p[i][1] = (joint[1] + sm * p[i][1]) / (den[1] + sm);
                    if i >= off {
                        let dst = keep_from + (i - off);
                        char_hit[dst][order] = if (joint[0] + joint[1]) > 0.0 { 1.0 } else { 0.0 };
                        let lp0 = p[i][0].max(1e-12).ln();
                        let lp1 = p[i][1].max(1e-12).ln();
                        char_llr[dst][order] = (lp1 - lp0).clamp(-3.0, 3.0) as f32;
                    }
                }
            }
        }
    }

    fn extract_local_8ch_one(
        &self,
        clean_text: &str,
        starts: &[i32],
        ends: &[i32],
        out_row: &mut [f32], // len == max_len * 8
    ) {
        let n_tok = starts.len();
        if n_tok == 0 || clean_text.is_empty() {
            return;
        }
        let chars: Vec<char> = clean_text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return;
        }

        // Valid mask: non-ASCII AND outside <PII> placeholders
        let mut valid = vec![false; n];
        for (i, &ch) in chars.iter().enumerate() {
            if (ch as u32) >= 128 {
                valid[i] = true;
            }
        }
        // Only run placeholder regex if '<' is present (fast-path!)
        if clean_text.contains('<') {
            let bank = rx();
            // Map byte offset -> char index once when '<' is present
            for m in bank.pii_placeholder.find_iter(clean_text) {
                let cs = clean_text[..m.start()].chars().count();
                let ce = cs + m.as_str().chars().count();
                for v in valid.iter_mut().take(ce.min(n)).skip(cs) {
                    *v = false;
                }
            }
        }

        let mut char_llr = vec![[0.0f32; 4]; n];
        let mut char_hit = vec![[0.0f32; 4]; n];
        let max_eval = 1024usize;
        if n <= max_eval {
            self.score_char_span(&chars, 0, n, 0, &mut char_llr, &mut char_hit);
        } else {
            let half = max_eval / 2;
            self.score_char_span(&chars, 0, half, 0, &mut char_llr, &mut char_hit);
            let tail_keep = n - half;
            let ctx_start = tail_keep.saturating_sub(3);
            self.score_char_span(&chars, ctx_start, n, tail_keep, &mut char_llr, &mut char_hit);
        }
        for i in 0..n {
            if !valid[i] {
                char_llr[i] = [0.0; 4];
                char_hit[i] = [0.0; 4];
            }
        }

        let mut share = vec![0.0f32; n];
        for ti in 0..n_tok {
            let a = starts[ti];
            let b = ends[ti];
            if a >= 0 && b > a && (b as usize) <= n {
                for s in &mut share[(a as usize)..(b as usize)] {
                    *s += 1.0;
                }
            }
        }
        for s in &mut share {
            if *s < 1.0 {
                *s = 1.0;
            }
        }

        let max_tok = out_row.len() / LOCAL_CHARLM_DIM_V150;
        let lim = n_tok.min(max_tok);
        for ti in 0..lim {
            let a = starts[ti];
            let b = ends[ti];
            if a < 0 || b <= a || (a as usize) >= n {
                continue;
            }
            let au = a as usize;
            let bu = (b as usize).min(n);
            let mut w_sum = 0.0f32;
            for ci in au..bu {
                w_sum += 1.0 / share[ci];
            }
            if w_sum <= 1e-8 {
                continue;
            }
            let inv_w = 1.0 / w_sum;
            let base = ti * LOCAL_CHARLM_DIM_V150;
            let mut acc_l = [0.0f32; 4];
            let mut acc_h = [0.0f32; 4];
            for ci in au..bu {
                let wn = (1.0 / share[ci]) * inv_w;
                for k in 0..4 {
                    acc_l[k] += char_llr[ci][k] * wn;
                    acc_h[k] += char_hit[ci][k] * wn;
                }
            }
            for k in 0..4 {
                out_row[base + k] = acc_l[k];
                out_row[base + 4 + k] = acc_h[k];
            }
        }
    }
}

fn build_charlm_core_from_py(
    vocab_chars: &str,
    vocab_ids: &[u16],
    unigrams: PyReadonlyArray2<f32>,
    table_keys: Vec<PyReadonlyArray1<u64>>,
    table_counts: Vec<PyReadonlyArray2<u32>>,
    table_contexts: Vec<PyReadonlyArray1<u64>>,
    table_den: Vec<PyReadonlyArray2<u32>>,
    feat_mean: PyReadonlyArray1<f32>,
    feat_scale: PyReadonlyArray1<f32>,
    coef: PyReadonlyArray1<f32>,
    bias: f32,
    max_chars: usize,
) -> PyResult<CharLMCore> {
    let mut ascii_vocab = [1u16; 128];
    let mut vocab_map: AHashMap<char, u16> = AHashMap::with_capacity(vocab_ids.len());
    for (ch, &id) in vocab_chars.chars().zip(vocab_ids.iter()) {
        let cp = ch as u32;
        if cp < 128 {
            ascii_vocab[cp as usize] = id;
        } else {
            vocab_map.insert(ch, id);
        }
    }
    let u_view = unigrams.as_array();
    let mut uni_vec = Vec::with_capacity(u_view.nrows());
    for i in 0..u_view.nrows() {
        uni_vec.push([u_view[[i, 0]] as f64, u_view[[i, 1]] as f64]);
    }
    let mut tables = Vec::with_capacity(table_keys.len());
    for oi in 0..table_keys.len() {
        let ks = table_keys[oi].as_slice()?.to_vec();
        let cv = table_counts[oi].as_array();
        let mut cts = Vec::with_capacity(cv.nrows());
        for i in 0..cv.nrows() {
            cts.push([cv[[i, 0]], cv[[i, 1]]]);
        }
        let ctxs = table_contexts[oi].as_slice()?.to_vec();
        let dv = table_den[oi].as_array();
        let mut dens = Vec::with_capacity(dv.nrows());
        for i in 0..dv.nrows() {
            dens.push([dv[[i, 0]], dv[[i, 1]]]);
        }
        tables.push(CharLMTable { keys: ks, counts: cts, contexts: ctxs, den: dens });
    }
    let fm_s = feat_mean.as_slice()?;
    let fs_s = feat_scale.as_slice()?;
    let cf_s = coef.as_slice()?;
    let mut fm = [0.0f32; 33];
    let mut fs = [1.0f32; 33];
    let mut cf = [0.0f32; 33];
    fm.copy_from_slice(&fm_s[..33]);
    fs.copy_from_slice(&fs_s[..33]);
    cf.copy_from_slice(&cf_s[..33]);
    Ok(CharLMCore {
        ascii_vocab,
        vocab_map,
        unigrams: uni_vec,
        tables,
        feat_mean: fm,
        feat_scale: fs,
        coef: cf,
        bias,
        max_chars,
    })
}

#[pyclass]
pub struct RustOOFCharLMSet {
    full: CharLMCore,
    folds: Vec<CharLMCore>,
}

#[pymethods]
impl RustOOFCharLMSet {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_chars: &str,
        vocab_ids: Vec<u16>,
        unigrams: PyReadonlyArray2<f32>,
        table_keys: Vec<PyReadonlyArray1<u64>>,
        table_counts: Vec<PyReadonlyArray2<u32>>,
        table_contexts: Vec<PyReadonlyArray1<u64>>,
        table_den: Vec<PyReadonlyArray2<u32>>,
        feat_mean: PyReadonlyArray1<f32>,
        feat_scale: PyReadonlyArray1<f32>,
        coef: PyReadonlyArray1<f32>,
        bias: f32,
        max_chars: usize,
    ) -> PyResult<Self> {
        let full = build_charlm_core_from_py(
            vocab_chars, &vocab_ids, unigrams, table_keys, table_counts, table_contexts, table_den,
            feat_mean, feat_scale, coef, bias, max_chars,
        )?;
        Ok(Self { full, folds: Vec::new() })
    }

    #[allow(clippy::too_many_arguments)]
    fn add_fold(
        &mut self,
        vocab_chars: &str,
        vocab_ids: Vec<u16>,
        unigrams: PyReadonlyArray2<f32>,
        table_keys: Vec<PyReadonlyArray1<u64>>,
        table_counts: Vec<PyReadonlyArray2<u32>>,
        table_contexts: Vec<PyReadonlyArray1<u64>>,
        table_den: Vec<PyReadonlyArray2<u32>>,
        feat_mean: PyReadonlyArray1<f32>,
        feat_scale: PyReadonlyArray1<f32>,
        coef: PyReadonlyArray1<f32>,
        bias: f32,
        max_chars: usize,
    ) -> PyResult<()> {
        let fold = build_charlm_core_from_py(
            vocab_chars, &vocab_ids, unigrams, table_keys, table_counts, table_contexts, table_den,
            feat_mean, feat_scale, coef, bias, max_chars,
        )?;
        self.folds.push(fold);
        Ok(())
    }

    /// Parallel batch extraction of BOTH global 9d AND (optional) token-aligned local 8ch with per-row OOF fold routing!
    fn extract_9d_and_local_batch<'py>(
        &self,
        py: Python<'py>,
        clean_texts: Vec<String>,
        fold_ids: PyReadonlyArray1<'py, i32>,
        flat_starts: PyReadonlyArray1<'py, i32>,
        flat_ends: PyReadonlyArray1<'py, i32>,
        row_splits: PyReadonlyArray1<'py, i32>,
        max_len: usize,
        need_local: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<f32>>)> {
        let n = clean_texts.len();
        let fids = fold_ids.as_slice()?.to_vec();
        let f_st = flat_starts.as_slice()?.to_vec();
        let f_en = flat_ends.as_slice()?.to_vec();
        let splits = row_splits.as_slice()?.to_vec();

        let (g9_flat, loc_flat) = py.allow_threads(|| {
            let mut g9 = vec![0.0f32; n * CHARLM_FEAT_DIM_V140];
            let mut loc = if need_local {
                vec![0.0f32; n * max_len * LOCAL_CHARLM_DIM_V150]
            } else {
                Vec::new()
            };

            if need_local {
                g9.par_chunks_mut(CHARLM_FEAT_DIM_V140)
                    .zip(loc.par_chunks_mut(max_len * LOCAL_CHARLM_DIM_V150))
                    .enumerate()
                    .for_each(|(i, (g_row, l_row))| {
                        let fid = fids[i];
                        let lm = if fid >= 0 && (fid as usize) < self.folds.len() {
                            &self.folds[fid as usize]
                        } else {
                            &self.full
                        };
                        let v9 = lm.extract_9d_one(&clean_texts[i]);
                        g_row.copy_from_slice(&v9);
                        let s0 = splits[i] as usize;
                        let s1 = splits[i + 1] as usize;
                        lm.extract_local_8ch_one(&clean_texts[i], &f_st[s0..s1], &f_en[s0..s1], l_row);
                    });
            } else {
                g9.par_chunks_mut(CHARLM_FEAT_DIM_V140)
                    .enumerate()
                    .for_each(|(i, g_row)| {
                        let fid = fids[i];
                        let lm = if fid >= 0 && (fid as usize) < self.folds.len() {
                            &self.folds[fid as usize]
                        } else {
                            &self.full
                        };
                        let v9 = lm.extract_9d_one(&clean_texts[i]);
                        g_row.copy_from_slice(&v9);
                    });
            }
            (g9, loc)
        });

        let g9_arr = Array2::from_shape_vec((n, CHARLM_FEAT_DIM_V140), g9_flat)
            .unwrap()
            .into_pyarray(py);
        let loc_arr = if need_local {
            Array3::from_shape_vec((n, max_len, LOCAL_CHARLM_DIM_V150), loc_flat)
                .unwrap()
                .into_pyarray(py)
        } else {
            Array3::<f32>::zeros((0, 0, LOCAL_CHARLM_DIM_V150)).into_pyarray(py)
        };
        Ok((g9_arr, loc_arr))
    }

    fn extract_9d_batch<'py>(
        &self,
        py: Python<'py>,
        clean_texts: Vec<String>,
        fold_ids: Option<PyReadonlyArray1<'py, i32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = clean_texts.len();
        let fids: Vec<i32> = match fold_ids {
            Some(arr) => arr.as_slice()?.to_vec(),
            None => vec![-1; n],
        };
        let g9_flat = py.allow_threads(|| {
            let mut g9 = vec![0.0f32; n * CHARLM_FEAT_DIM_V140];
            g9.par_chunks_mut(CHARLM_FEAT_DIM_V140)
                .enumerate()
                .for_each(|(i, g_row)| {
                    let fid = fids[i];
                    let lm = if fid >= 0 && (fid as usize) < self.folds.len() {
                        &self.folds[fid as usize]
                    } else {
                        &self.full
                    };
                    g_row.copy_from_slice(&lm.extract_9d_one(&clean_texts[i]));
                });
            g9
        });
        Ok(Array2::from_shape_vec((n, CHARLM_FEAT_DIM_V140), g9_flat).unwrap().into_pyarray(py))
    }
}

// ---------------------------------------------------------------------------
// 3. Quote Scope 4D Bounded-Stack FSM (bit-exact with v160_discourse_causal_ops!)
// ---------------------------------------------------------------------------
fn quote_scope_one(clean_text: &str, starts: &[i32], ends: &[i32], out_row: &mut [f32]) {
    let n_tok = starts.len();
    if n_tok == 0 || clean_text.is_empty() {
        return;
    }
    let chars: Vec<char> = clean_text.chars().collect();
    let n = chars.len();
    let mut states = vec![0u8; n];
    let mut stack: Vec<(char, usize)> = Vec::with_capacity(4);
    let mut unknown_flag = false;
    let mut sym_dq_open = false;
    let mut sym_sq_open = false;
    let max_depth = 4usize;

    for (i, &ch) in chars.iter().enumerate() {
        let open_pair = match ch {
            '「' => Some('」'),
            '『' => Some('』'),
            '“' => Some('”'),
            '【' => Some('】'),
            '《' => Some('》'),
            _ => None,
        };
        if let Some(close_c) = open_pair {
            states[i] = 3;
            if stack.len() >= max_depth {
                unknown_flag = true;
            } else {
                stack.push((close_c, i));
            }
        } else if matches!(ch, '」' | '』' | '”' | '】' | '》') {
            states[i] = 3;
            if stack.last().map(|x| x.0) == Some(ch) {
                stack.pop();
            } else {
                unknown_flag = true;
            }
        } else if ch == '"' || ch == '\'' {
            states[i] = 3;
            let is_open = if ch == '"' { &mut sym_dq_open } else { &mut sym_sq_open };
            if !*is_open {
                if stack.len() >= max_depth {
                    unknown_flag = true;
                } else {
                    *is_open = true;
                    stack.push((ch, i));
                }
            } else if stack.last().map(|x| x.0) == Some(ch) {
                stack.pop();
                *is_open = false;
            } else {
                unknown_flag = true;
            }
        } else if unknown_flag {
            states[i] = 2;
        } else if !stack.is_empty() {
            states[i] = 1;
        } else {
            states[i] = 0;
        }
    }

    if !stack.is_empty() {
        let first_unclosed = stack.iter().map(|x| x.1).min().unwrap_or(0);
        for s in states.iter_mut().skip(first_unclosed + 1) {
            if *s != 3 {
                *s = 2;
            }
        }
    }

    let max_tok = out_row.len() / QUOTE_SCOPE_DIM_V161;
    let lim = n_tok.min(max_tok);
    for ti in 0..lim {
        let a = starts[ti].clamp(0, n as i32) as usize;
        let b = ends[ti].clamp(a as i32, n as i32) as usize;
        let base = ti * QUOTE_SCOPE_DIM_V161;
        if b <= a {
            out_row[base] = 1.0;
            continue;
        }
        let inv_len = 1.0f32 / ((b - a) as f32);
        for &st in &states[a..b] {
            if (st as usize) < QUOTE_SCOPE_DIM_V161 {
                out_row[base + (st as usize)] += inv_len;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. ApexChar R0++ Batch Extractor (bit-exact with v140 + v160 expanded tables!)
// ---------------------------------------------------------------------------
fn count_distinct_support_pairs(chars: &[char]) -> f32 {
    let lim = chars.len().min(256);
    if lim < 2 {
        return 0.0;
    }
    let mut pairs: AHashSet<(char, char)> = AHashSet::with_capacity(lim);
    let mut prev = chars[0];
    let mut prev_ok = prev.is_alphanumeric();
    for &ch in &chars[1..lim] {
        let ok = ch.is_alphanumeric();
        if prev_ok && ok {
            pairs.insert((prev, ch));
        }
        prev = ch;
        prev_ok = ok;
    }
    pairs.len() as f32
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn extract_apex_char_batch_rs<'py>(
    py: Python<'py>,
    clean_texts: Vec<String>,
    b2: usize,
    b3: usize,
    max_chars: usize,
    head_chars: usize,
    precomputed_9d: PyReadonlyArray2<'py, f32>,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let n = clean_texts.len();
    if n == 0 {
        return Ok((
            Array2::<i64>::zeros((0, 8)).into_pyarray(py),
            Array2::<i64>::zeros((0, 8)).into_pyarray(py),
            Array2::<i64>::zeros((0, 8)).into_pyarray(py),
            Array2::<f32>::zeros((0, 8)).into_pyarray(py),
            Array2::<f32>::zeros((0, R0_FEAT_DIM_V140)).into_pyarray(py),
            Array1::<f32>::zeros(0).into_pyarray(py),
        ));
    }
    let g9_slice = precomputed_9d.as_slice()?.to_vec();
    let is_expanded = b2 != R0_B2_V130 || b3 != R0_B3_V130;
    let body_cap = max_chars.saturating_sub(2).max(8);
    let h_cap = head_chars.min(body_cap.saturating_sub(4));
    let t_cap = body_cap - h_cap;

    let (ml, c1_v, c2_v, c3_v, cm_v, cf_v, sup_v) = py.allow_threads(|| {
        // Compute max length across batch
        let char_vecs: Vec<Vec<char>> = clean_texts.par_iter().map(|s| s.chars().collect()).collect();
        let ml = char_vecs
            .par_iter()
            .map(|cv| cv.len().min(body_cap) + 2)
            .max()
            .unwrap_or(8)
            .max(8);

        let mut c1_v = vec![0i64; n * ml];
        let mut c2_v = vec![0i64; n * ml];
        let mut c3_v = vec![0i64; n * ml];
        let mut cm_v = vec![0.0f32; n * ml];
        let mut cf_v = vec![0.0f32; n * R0_FEAT_DIM_V140];
        let mut sup_v = vec![0.0f32; n];

        let b1_m = (R0_B1_V130 - 1) as u32;
        let b2_m_v140 = (R0_B2_V130 - 1) as u32;
        let b3_m_v140 = (R0_B3_V130 - 1) as u32;
        let b2_m_exp = b2.saturating_sub(1).max(2) as u64;
        let b3_m_exp = b3.saturating_sub(1).max(2) as u64;

        c1_v.par_chunks_mut(ml)
            .zip(c2_v.par_chunks_mut(ml))
            .zip(c3_v.par_chunks_mut(ml))
            .zip(cm_v.par_chunks_mut(ml))
            .zip(cf_v.par_chunks_mut(R0_FEAT_DIM_V140))
            .zip(sup_v.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((((r1, r2), r3), rm), rf), rsup))| {
                let cv = &char_vecs[i];
                let sup = count_distinct_support_pairs(cv);
                *rsup = sup;

                let l_body = cv.len().min(body_cap);
                let l_tot = l_body + 2;
                let mut codes = Vec::with_capacity(l_tot);
                codes.push(2u32);
                if cv.len() > body_cap {
                    for &ch in &cv[..h_cap] {
                        codes.push(ch as u32);
                    }
                    for &ch in &cv[cv.len() - t_cap..] {
                        codes.push(ch as u32);
                    }
                } else {
                    for &ch in cv {
                        codes.push(ch as u32);
                    }
                }
                codes.push(3u32);

                for j in 0..l_tot {
                    rm[j] = 1.0;
                    let u1 = ((codes[j].wrapping_mul(2654435761) >> 20) % b1_m) + 1;
                    r1[j] = u1 as i64;
                }

                if is_expanded {
                    for j in 0..l_tot {
                        let a = r1[j] as u64;
                        let b = if j + 1 < l_tot { r1[j + 1] as u64 } else { 0 };
                        let c = if j + 2 < l_tot { r1[j + 2] as u64 } else { 0 };
                        r2[j] = ((((a.wrapping_mul(1315423911) ^ b.wrapping_mul(2654435761)) & 0x7FFF_FFFF) % b2_m_exp) + 1) as i64;
                        r3[j] = ((((a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ c.wrapping_mul(83492791)) & 0x7FFF_FFFF) % b3_m_exp) + 1) as i64;
                    }
                } else {
                    if l_tot >= 2 {
                        let mut last_u2 = 1i64;
                        for j in 0..(l_tot - 1) {
                            let h2 = codes[j].wrapping_mul(1315423911) ^ codes[j + 1].wrapping_mul(2654435761);
                            let u2 = (((h2 >> 18) % b2_m_v140) + 1) as i64;
                            r2[j] = u2;
                            last_u2 = u2;
                        }
                        r2[l_tot - 1] = last_u2;
                    }
                    if l_tot >= 3 {
                        let mut last_u3 = 1i64;
                        for j in 0..(l_tot - 2) {
                            let h3 = codes[j].wrapping_mul(1315423911)
                                ^ codes[j + 1].wrapping_mul(2654435761)
                                ^ codes[j + 2].wrapping_mul(2246822519);
                            let u3 = (((h3 >> 19) % b3_m_v140) + 1) as i64;
                            r3[j] = u3;
                            last_u3 = u3;
                        }
                        r3[l_tot - 2] = last_u3;
                        r3[l_tot - 1] = last_u3;
                    } else {
                        for j in 0..l_tot {
                            r3[j] = r1[j];
                        }
                    }
                }

                // 17-d structural scalars on first 512 chars
                let lf = cv.len().max(1) as f64;
                let lb = cv.len().min(512).max(1);
                let inv_l = 1.0 / (lb as f64);
                let mut cjk = 0u32;
                let mut alnum = 0u32;
                let mut digits = 0u32;
                let mut spaces = 0u32;
                let mut newlines = 0u32;
                let mut cn_comma = 0u32;
                let mut cn_period = 0u32;
                let mut cn_colon = 0u32;
                let mut cn_semi = 0u32;
                let mut emdash = 0u32;
                let mut quote_cn = 0u32;
                let mut bracket_emo = 0u32;
                let mut qmark = 0u32;
                let mut exclam = 0u32;
                let mut uniq: AHashSet<u32> = AHashSet::with_capacity(lb.min(128));

                if cv.is_empty() {
                    uniq.insert(0);
                } else {
                    for &ch in cv.iter().take(512) {
                        let cp = ch as u32;
                        uniq.insert(cp);
                        if (0x4E00..=0x9FFF).contains(&cp) {
                            cjk += 1;
                        } else if (48..=57).contains(&cp) {
                            alnum += 1;
                            digits += 1;
                        } else if (65..=90).contains(&cp) || (97..=122).contains(&cp) {
                            alnum += 1;
                        } else {
                            match cp {
                                32 | 9 => spaces += 1,
                                10 => newlines += 1,
                                0xFF0C | 44 => cn_comma += 1,
                                0x3002 => cn_period += 1,
                                0xFF1A | 58 => cn_colon += 1,
                                0xFF1B | 59 => cn_semi += 1,
                                0x2014 | 0x2015 => emdash += 1,
                                0x300C | 0x300D | 0x201C | 0x201D => quote_cn += 1,
                                91 => bracket_emo += 1,
                                63 | 0xFF1F => qmark += 1,
                                33 | 0xFF01 => exclam += 1,
                                _ => {}
                            }
                        }
                    }
                }
                let ends_per = if clean_texts[i].trim().ends_with('。') { 1.0f32 } else { 0.0f32 };
                rf[0] = (lf.ln_1p() / 6.0) as f32;
                rf[1] = (cjk as f64 * inv_l) as f32;
                rf[2] = (alnum as f64 * inv_l) as f32;
                rf[3] = (digits as f64 * inv_l) as f32;
                rf[4] = (spaces as f64 * inv_l) as f32;
                rf[5] = ((newlines as f64) / 6.0).min(1.0) as f32;
                rf[6] = ((cn_comma as f64) * inv_l * 10.0).min(1.0) as f32;
                rf[7] = ((cn_period as f64) * inv_l * 10.0).min(1.0) as f32;
                rf[8] = ((cn_colon as f64) * 0.25).min(1.0) as f32;
                rf[9] = ((cn_semi as f64) * 0.25).min(1.0) as f32;
                rf[10] = ((emdash as f64) * 0.25).min(1.0) as f32;
                rf[11] = ((quote_cn as f64) * 0.25).min(1.0) as f32;
                rf[12] = ((bracket_emo as f64) * 0.25).min(1.0) as f32;
                rf[13] = (((qmark + exclam) as f64) * inv_l * 10.0).min(1.0) as f32;
                rf[14] = ends_per;
                rf[15] = (uniq.len() as f64 * inv_l) as f32;
                rf[16] = ((sup as f64) / 32.0).min(3.0) as f32;
                // Copy precomputed 9d CharLM features
                let g0 = i * CHARLM_FEAT_DIM_V140;
                rf[R0_STRUCT_DIM_V140..R0_FEAT_DIM_V140].copy_from_slice(&g9_slice[g0..g0 + CHARLM_FEAT_DIM_V140]);
            });

        (ml, c1_v, c2_v, c3_v, cm_v, cf_v, sup_v)
    });

    Ok((
        Array2::from_shape_vec((n, ml), c1_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, ml), c2_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, ml), c3_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, ml), cm_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, R0_FEAT_DIM_V140), cf_v).unwrap().into_pyarray(py),
        Array1::from_vec(sup_v).into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// 5. Parallel Batch Canonicalize + Style-56D + Quote-Scope + Engram Builder
// ---------------------------------------------------------------------------
#[pyfunction]
fn canonicalize_and_style56_batch_rs<'py>(
    py: Python<'py>,
    raw_texts: Vec<String>,
    max_chars_canon: usize,
    max_chars_style: usize,
) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f32>>)> {
    let n = raw_texts.len();
    let (cleans, f56_flat) = py.allow_threads(|| {
        let mut cleans = vec![String::new(); n];
        let mut f56 = vec![0.0f32; n * STYLE_FEAT_DIM_V11];
        cleans
            .par_iter_mut()
            .zip(f56.par_chunks_mut(STYLE_FEAT_DIM_V11))
            .enumerate()
            .for_each(|(i, (cl_out, st_out))| {
                let (cl, _) = canonicalize_v150_single(&raw_texts[i], max_chars_canon);
                let feat = extract_style_56d_from_text(&cl, max_chars_style);
                st_out.copy_from_slice(&feat);
                *cl_out = cl;
            });
        (cleans, f56)
    });
    let arr = Array2::from_shape_vec((n, STYLE_FEAT_DIM_V11), f56_flat).unwrap().into_pyarray(py);
    Ok((cleans, arr))
}

#[pyfunction]
fn canonicalize_v150_batch_rs(py: Python<'_>, raw_texts: Vec<String>, max_chars: usize) -> Vec<String> {
    py.allow_threads(|| {
        raw_texts
            .par_iter()
            .map(|t| canonicalize_v150_single(t, max_chars).0)
            .collect()
    })
}

#[pyfunction]
fn extract_quote_scope_4d_batch_rs<'py>(
    py: Python<'py>,
    clean_texts: Vec<String>,
    flat_starts: PyReadonlyArray1<'py, i32>,
    flat_ends: PyReadonlyArray1<'py, i32>,
    row_splits: PyReadonlyArray1<'py, i32>,
    max_len: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let n = clean_texts.len();
    let f_st = flat_starts.as_slice()?.to_vec();
    let f_en = flat_ends.as_slice()?.to_vec();
    let splits = row_splits.as_slice()?.to_vec();
    let qsc_flat = py.allow_threads(|| {
        let mut qsc = vec![0.0f32; n * max_len * QUOTE_SCOPE_DIM_V161];
        qsc.par_chunks_mut(max_len * QUOTE_SCOPE_DIM_V161)
            .enumerate()
            .for_each(|(i, q_row)| {
                let s0 = splits[i] as usize;
                let s1 = splits[i + 1] as usize;
                quote_scope_one(&clean_texts[i], &f_st[s0..s1], &f_en[s0..s1], q_row);
            });
        qsc
    });
    Ok(Array3::from_shape_vec((n, max_len, QUOTE_SCOPE_DIM_V161), qsc_flat).unwrap().into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_bpe_engram_tensors_rs<'py>(
    py: Python<'py>,
    flat_ids: PyReadonlyArray1<'py, i64>,
    row_splits: PyReadonlyArray1<'py, i32>,
    max_len: usize,
    pad_id: i64,
    engram_buckets: usize,
    trigram_buckets: usize,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let f_ids = flat_ids.as_slice()?.to_vec();
    let splits = row_splits.as_slice()?.to_vec();
    let n = splits.len().saturating_sub(1);
    let eg_m = (engram_buckets.saturating_sub(1)).max(1) as u64;
    let tg_m = (trigram_buckets.saturating_sub(1)).max(1) as u64;

    let (ids_v, msk_v, eg_v, tg_v) = py.allow_threads(|| {
        let mut ids_v = vec![pad_id; n * max_len];
        let mut msk_v = vec![0i64; n * max_len];
        let mut eg_v = vec![0i64; n * max_len];
        let mut tg_v = vec![0i64; n * max_len];

        ids_v
            .par_chunks_mut(max_len)
            .zip(msk_v.par_chunks_mut(max_len))
            .zip(eg_v.par_chunks_mut(max_len))
            .zip(tg_v.par_chunks_mut(max_len))
            .enumerate()
            .for_each(|(i, (((ri, rm), re), rt))| {
                let s0 = splits[i] as usize;
                let s1 = splits[i + 1] as usize;
                let ns = (s1 - s0).min(max_len);
                let mut p2 = 0u64;
                let mut p1 = 0u64;
                for j in 0..ns {
                    let tid = f_ids[s0 + j];
                    let tu = tid as u64;
                    ri[j] = tid;
                    rm[j] = 1;
                    let h2 = ((p1.wrapping_mul(1315423911)) ^ (tu.wrapping_mul(2654435761))) & 0xFFFF_FFFF;
                    re[j] = ((h2 % eg_m) + 1) as i64;
                    let h3 = ((p2.wrapping_mul(2246822519)) ^ (p1.wrapping_mul(1315423911)) ^ (tu.wrapping_mul(2654435761))) & 0xFFFF_FFFF;
                    rt[j] = ((h3 % tg_m) + 1) as i64;
                    p2 = p1;
                    p1 = tu;
                }
            });
        (ids_v, msk_v, eg_v, tg_v)
    });

    Ok((
        Array2::from_shape_vec((n, max_len), ids_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, max_len), msk_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, max_len), eg_v).unwrap().into_pyarray(py),
        Array2::from_shape_vec((n, max_len), tg_v).unwrap().into_pyarray(py),
    ))
}

#[pymodule]
fn _meguviz_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustOOFCharLMSet>()?;
    m.add_function(wrap_pyfunction!(extract_apex_char_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_and_style56_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_v150_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(extract_quote_scope_4d_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(build_bpe_engram_tensors_rs, m)?)?;
    Ok(())
}
