"""Colab Pro+ GPU topic-debiased distillation pipeline for Hydrogen XHigh corpus relabeling."""

import argparse
import gzip
import hashlib
import html
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_INPUT_ORDER_SHA256 = "63b0b036b2dee7ded4a2bfedabd0bc5f82589fd40e426ef8e223cb0a2a5a6f66"
FILES = {
    "bili-round01-corpus.jsonl.gz": "e64762a497309612d88a299e12b58a89c6406429de379b8811a0866ae438197c",
    "bili-round01-split.json": "293265436d2e4ab2dc5e4966ffadfb8fafd46b73a5675c4c84cee28c4a2276ce",
    "bili-round01-user-confirmed.jsonl": "49739d460fc48c8fd22d85c732f2d81069e780ea90657f8e2f7a4b826e96f372",
    "bili-synthetic-deepseek_v41.jsonl.gz": "5697051044f46803384f73b38b64feb8675915586b94b28e25c6c88bf452348d",
    "bili-synthetic-other_models.jsonl.gz": "030cf9f956675961d0c18868f266e6258a5db494d4142abbec7ae45284692283",
}
COUNTS = {"bili": 200_370, "deepseek": 100_000, "other_models": 50_000, "user_confirmed": 17}
CODE_LIST = ("A", "R", "E", "T", "P", "H", "S", "Q")
CODE_TO_IDX = {c: i for i, c in enumerate(CODE_LIST)}
CODES_V2 = {
    "A": "助手/记忆/流程/任务自述或身份暴露",
    "R": "结构化视频细节/多事实/定量信息压缩复盘",
    "E": "典型 AI 评述结构与排版特征",
    "T": "与评论区语境错位的模板化附和、追问或异议",
    "P": "轻度书面化/工整句号化等弱 AI 倾向",
    "H": "明显真人自然互动证据",
    "S": "过短或泛化短句，缺乏字面可辨证据",
    "Q": "引用、玩梗反串、戏仿拟态或信号显著冲突",
}

# Precompiled purely topic-free regex patterns (ZERO 24-scene topic nouns)
BILI_EMOJI_RE = re.compile(r"\[[^\s\[\]]{1,16}\]")
CLAUSE_SPLIT_RE = re.compile(r"[,，。.!！?？;；\n]+")
TIME_RE = re.compile(r"(?<![0-9])\d{1,2}:[0-5]\d(?::[0-5]\d)?(?![0-9])")
UNIT_NUM_RE = re.compile(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百两]+)\s*(?:%|倍|成|轮|次|组|条|段|档|帧|格|张|份|层|个|分钟|秒|小时|毫米|厘米|公里|元|度|行|处)")
SPACE_CJK_RE = re.compile(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]")
REPEAT_PUNCT_RE = re.compile(r"([!！?？~～.。])\1+")
CONTRAST_RE = re.compile(r"不是.{1,15}(?:而是|是)|不在.{1,15}在|不等于|比.{1,12}更能|关键不在|难的不在|容易.{1,10}难")
RECAP_OPEN_RE = re.compile(r"^(?:复盘|记一下|记录|待跟进|暂记|归档|总结|先说|作为|本[鱼鲸]|收到|补一句)")
METHOD_CUE_RE = re.compile(r"控制变量|对照|基准|样本|顺序效应|变量|权衡|取舍|两码事|打个折|站不住|分开看|本质上|核心在于|关键在于|值得注意|换句话说|说白了|大概率|确实")
POLITE_REPLY_RE = re.compile(r"平和地说|保留意见|你的疑惑|是这样的|我理解为|客观来说|换个角度|先肯定|质疑一下")
ASSISTANT_CUE_RE = re.compile(r"记一下|记进|归档|待核|仅作素材|不是指令|工作流|提示词|子代理|读完.{0,4}字幕|转写")
SLANG_RE = re.compile(r"哈哈|笑死|绷不住|逆天|牛逼|牛批|卧槽|我靠|啊这|离谱|绝了|好耶|乐死|草|寄|nb|yyds|tql", re.I)
MODAL_END_RE = re.compile(r"[啥咋嘛呀呗哈喵捏呐啦哇哦噢吧呢]$")
MARKDOWN_RE = re.compile(r"\*\*[^*]+\*\*|^\s*(?:[-*]|\d+\.)\s+", re.M)

# Closed-class Chinese grammatical/rhetorical whitelist for topic-blind skeleton TF-IDF
FUNC_CHARS = set(
    "的了是在不这那有也就都而和与及或但可却并因所如果虽只还才再更最很太挺偏先总复盘记下留归对比"
    "实本核心关键质问说明致影响值得适合正常我你他她它们把被让给对从向于以把为之其者呢吧吗呀啊哦哈嘛"
    "呗啥咋一二三四五六七八九十两多几每各全另其非无没未"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_corpus(data_dir: Path):
    """Load all 350,387 frozen comments in canonical order and verify SHA256."""
    paths = {}
    for name, expected in FILES.items():
        p = data_dir / name
        if not p.is_file():
            p = data_dir / "synthetic-scale" / name
        if not p.is_file():
            raise FileNotFoundError(f"缺少语料文件: {name}")
        if sha256_file(p) != expected:
            raise ValueError(f"SHA256 校验失败: {name}")
        paths[name] = p

    manifest = json.loads(paths["bili-round01-split.json"].read_text(encoding="utf-8"))
    groups = manifest["video_assignments"]
    records = []
    with gzip.open(paths["bili-round01-corpus.jsonl.gz"], "rt", encoding="utf-8") as f:
        for line in f:
            uid, video, text, prior, origin, level, _, hold = json.loads(line)
            entry = groups[video]
            records.append({
                "sample_id": uid,
                "text": text,
                "cohort": "bili",
                "split": entry["split"],
                "video_id": video,
                "dsh": bool(entry["dsh"]),
                "level": level,
                "previous_label": prior,
                "previous_source": origin,
                "previous_hold_reason": hold,
                "generator_model": "",
                "format": "",
                "origin_status": "unverified",
            })

    for cohort, fname in (
        ("deepseek", "bili-synthetic-deepseek_v41.jsonl.gz"),
        ("other_models", "bili-synthetic-other_models.jsonl.gz"),
    ):
        with gzip.open(paths[fname], "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["label"] != "ai" or row["split"] != "train":
                    raise ValueError(f"合成样本标记异常: {cohort}")
                records.append({
                    "sample_id": row["sample_id"],
                    "text": row["text"],
                    "cohort": cohort,
                    "split": "train",
                    "video_id": f"synth_{row.get('topic', cohort)}_{row.get('format', '')}",
                    "dsh": False,
                    "level": "",
                    "previous_label": "ai",
                    "previous_source": row["source"],
                    "previous_hold_reason": "",
                    "generator_model": row["model_id"],
                    "format": row["format"],
                    "origin_status": "confirmed_ai",
                })

    for line in paths["bili-round01-user-confirmed.jsonl"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append({
            "sample_id": row["id"],
            "text": row["text"],
            "cohort": "user_confirmed",
            "split": "probe",
            "video_id": f"probe_{row['id']}",
            "dsh": True,
            "level": "",
            "previous_label": "ai",
            "previous_source": row["source"],
            "previous_hold_reason": "",
            "generator_model": "",
            "format": "",
            "origin_status": "confirmed_ai",
        })

    if Counter(r["cohort"] for r in records) != COUNTS:
        raise ValueError("语料计数不匹配")
    digest = hashlib.sha256()
    for r in records:
        digest.update((r["sample_id"] + "\n").encode("utf-8"))
    order_sha = digest.hexdigest()
    if order_sha != EXPECTED_INPUT_ORDER_SHA256:
        raise ValueError(f"输入顺序哈希失配: {order_sha}")
    return records, order_sha


def load_gold(gold_path: Path) -> dict[str, dict]:
    gold = {}
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[row["sample_id"]] = row
    return gold


def mask_topic_skeleton(text: str) -> str:
    """Replace content CJK nouns/verbs with 'C', digits with 'D', ASCII with 'A', keeping punctuation & closed-class chars."""
    t = unicodedata.normalize("NFKC", text)
    t = BILI_EMOJI_RE.sub(" \x01 ", t)
    out = []
    for ch in t:
        if ch == "\x01":
            out.append("[EMOJI]")
        elif ch in FUNC_CHARS or ch in " \t\n,，。.!！?？;；:：“”\"'‘’「」『』《》（）()[]—-…~～*/":
            out.append(ch)
        elif "0" <= ch <= "9":
            out.append("D")
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            out.append("A")
        elif "\u3400" <= ch <= "\u9fff":
            out.append("C")
        else:
            out.append(ch)
    return "".join(out)


STYLE_FEATURE_NAMES = [
    "len_chars", "log_len", "n_clauses", "log_clauses", "mean_clause_len", "max_clause_len",
    "std_clause_len", "ends_period", "ends_exclam", "ends_quest", "ends_ellipsis", "ends_bare",
    "ends_emoji", "cnt_emdash", "rate_emdash", "cnt_dash2", "cnt_corner_l", "cnt_corner_r",
    "rate_corner", "cnt_quote_cn", "cnt_quote_ascii", "cnt_colon", "rate_colon", "cnt_semi",
    "rate_semi", "cnt_comma", "rate_comma", "cnt_period", "rate_period", "cnt_ellip",
    "cnt_tilde", "cnt_exclam", "cnt_quest", "cnt_emoji", "rate_emoji", "cnt_space_cjk",
    "rate_space_cjk", "space_ratio", "ascii_ratio", "digit_ratio", "cjk_ratio",
    "cnt_time", "cnt_unit_num", "cnt_newlines", "has_markdown", "unclosed_paren",
    "cnt_repeat_punct", "has_contrast", "has_recap_open", "cnt_method_cue",
    "has_polite_reply", "has_assistant_cue", "cnt_slang", "ends_modal",
    "comma_to_period_ratio", "emdash_plus_corner", "emdash_and_period",
]


def extract_style_features(texts: list[str]) -> np.ndarray:
    """Compute 57 purely domain-invariant stylistic/structural features."""
    mat = np.zeros((len(texts), len(STYLE_FEATURE_NAMES)), dtype=np.float32)
    for idx, raw in enumerate(texts):
        text = unicodedata.normalize("NFKC", raw)
        stripped = text.rstrip()
        n = max(1, len(text))
        clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]
        c_lens = [len(c) for c in clauses] or [0]

        end_ch = stripped[-1] if stripped else ""
        ends_emoji = 1.0 if BILI_EMOJI_RE.search(stripped[-18:]) and stripped.endswith("]") else 0.0
        cnt_emdash = text.count("——")
        cnt_dash2 = text.count("--")
        cnt_cl = text.count("「") + text.count("『")
        cnt_cr = text.count("」") + text.count("』")
        cnt_qcn = text.count("“") + text.count("”") + text.count("‘") + text.count("’")
        cnt_qasc = text.count('"') + text.count("'")
        cnt_colon = text.count("：") + text.count(":")
        cnt_semi = text.count("；") + text.count(";")
        cnt_comma = text.count("，") + text.count(",")
        cnt_period = text.count("。")
        cnt_ellip = text.count("…") + text.count("...")
        cnt_tilde = text.count("~") + text.count("～")
        cnt_exclam = text.count("！") + text.count("!")
        cnt_quest = text.count("？") + text.count("?")
        cnt_emoji = len(BILI_EMOJI_RE.findall(text))
        cnt_space_cjk = len(SPACE_CJK_RE.findall(text))
        spaces = sum(c.isspace() for c in text)
        asciis = sum(c.isascii() and c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        cjks = sum("\u3400" <= c <= "\u9fff" for c in text)
        unclosed = abs(text.count("(") - text.count(")")) + abs(text.count("（") - text.count("）"))

        scale = 100.0 / n
        mat[idx] = [
            float(len(text)),
            math.log1p(len(text)),
            float(len(clauses)),
            math.log1p(len(clauses)),
            float(sum(c_lens) / len(c_lens)),
            float(max(c_lens)),
            float(np.std(c_lens)),
            1.0 if end_ch == "。" else 0.0,
            1.0 if end_ch in "!！" else 0.0,
            1.0 if end_ch in "?？" else 0.0,
            1.0 if stripped.endswith(("...", "……", "…")) else 0.0,
            1.0 if end_ch not in "。!！?？…~～]" else 0.0,
            ends_emoji,
            float(cnt_emdash),
            cnt_emdash * scale,
            float(cnt_dash2),
            float(cnt_cl),
            float(cnt_cr),
            (cnt_cl + cnt_cr) * scale,
            float(cnt_qcn),
            float(cnt_qasc),
            float(cnt_colon),
            cnt_colon * scale,
            float(cnt_semi),
            cnt_semi * scale,
            float(cnt_comma),
            cnt_comma * scale,
            float(cnt_period),
            cnt_period * scale,
            float(cnt_ellip),
            float(cnt_tilde),
            float(cnt_exclam),
            float(cnt_quest),
            float(cnt_emoji),
            cnt_emoji * scale,
            float(cnt_space_cjk),
            cnt_space_cjk * scale,
            spaces / n,
            asciis / n,
            digits / n,
            cjks / n,
            float(len(TIME_RE.findall(text))),
            float(len(UNIT_NUM_RE.findall(text))),
            float(text.count("\n")),
            1.0 if MARKDOWN_RE.search(text) else 0.0,
            float(unclosed),
            float(len(REPEAT_PUNCT_RE.findall(text))),
            1.0 if CONTRAST_RE.search(text) else 0.0,
            1.0 if RECAP_OPEN_RE.search(text) else 0.0,
            float(len(METHOD_CUE_RE.findall(text))),
            1.0 if POLITE_REPLY_RE.search(text) else 0.0,
            1.0 if ASSISTANT_CUE_RE.search(text) else 0.0,
            float(len(SLANG_RE.findall(text))),
            1.0 if MODAL_END_RE.search(stripped) else 0.0,
            (cnt_comma + 1.0) / (cnt_period + 1.0),
            float(cnt_emdash + cnt_cl),
            1.0 if (cnt_emdash > 0 and end_ch == "。") else 0.0,
        ]
    return mat


def judgment_v2(score: float, code: str, threshold: float = 70.0) -> str:
    if score >= threshold and code in "ARET":
        return "ai_like"
    if score <= 30.0 and code == "H":
        return "human_like"
    return "uncertain"


def run_dry_test(data_dir: Path, gold_path: Path | None, rubric_path: Path | None) -> None:
    """Fast CPU local verification of corpus loading, masking, feature extraction, and schema."""
    t0 = time.monotonic()
    records, order_sha = load_frozen_corpus(data_dir)
    sample = records[:80] + records[200370:200450] + records[300370:300410] + records[-17:]
    texts = [r["text"] for r in sample]
    feats = extract_style_features(texts)
    skeletons = [mask_topic_skeleton(t) for t in texts[:5]]
    assert feats.shape == (len(sample), len(STYLE_FEATURE_NAMES))
    assert np.isfinite(feats).all()
    gold_count = len(load_gold(gold_path)) if (gold_path and gold_path.is_file()) else 0
    rubric_len = len(rubric_path.read_text(encoding="utf-8")) if (rubric_path and rubric_path.is_file()) else 0
    print(json.dumps({
        "dry_run": "PASS",
        "total_records": len(records),
        "input_order_sha256": order_sha,
        "tested_slice": len(sample),
        "feature_shape": list(feats.shape),
        "gold_rows_loaded": gold_count,
        "rubric_chars": rubric_len,
        "sample_skeleton": skeletons[0][:80],
        "elapsed_s": round(time.monotonic() - t0, 2),
    }, ensure_ascii=False, indent=2))


def make_group_folds(groups: list[str], n_splits: int = 5, seed: int = 20261123) -> list[int]:
    """Deterministic video-level GroupKFold assignment balanced by group size."""
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    ordered = sorted(by_g.items(), key=lambda kv: (-len(kv[1]), hashlib.md5(kv[0].encode()).hexdigest()))
    rng = random.Random(seed)
    # Slight deterministic shuffle among size-1 groups
    singletons = [kv for kv in ordered if len(kv[1]) == 1]
    larger = [kv for kv in ordered if len(kv[1]) > 1]
    rng.shuffle(singletons)
    fold_sizes = [0] * n_splits
    folds = [0] * len(groups)
    for _, idxs in larger + singletons:
        f = min(range(n_splits), key=lambda k: fold_sizes[k])
        fold_sizes[f] += len(idxs)
        for i in idxs:
            folds[i] = f
    return folds


def safe_metrics(y_true_score: np.ndarray, y_pred_score: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_bin = (y_true_score >= 70.0).astype(int)
    mae = float(np.mean(np.abs(y_true_score - y_pred_score)))
    rmse = float(np.sqrt(np.mean((y_true_score - y_pred_score) ** 2)))
    sp = float(spearmanr(y_true_score, y_pred_score).statistic) if len(np.unique(y_true_score)) > 1 else 0.0
    pe = float(pearsonr(y_true_score, y_pred_score).statistic) if len(np.unique(y_true_score)) > 1 else 0.0
    out = {"n": int(len(y_true_score)), "positives_ge70": int(y_bin.sum()), "mae": round(mae, 3), "rmse": round(rmse, 3), "spearman": round(sp, 4), "pearson": round(pe, 4)}
    if 0 < y_bin.sum() < len(y_bin):
        out["auroc"] = round(float(roc_auc_score(y_bin, y_pred_score)), 4)
        out["ap"] = round(float(average_precision_score(y_bin, y_pred_score)), 4)
    return out


def train_model_a(records, gold_map: dict[str, dict], style_mat: np.ndarray):
    """Model A: A1 topic-free 150k style classifier + A2 Gold style regressor."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    n_all = len(records)
    print(f"[Model A] 构建掩码内容骨架 TF-IDF (Topic-Free, has_lgb={has_lgb})...", flush=True)
    # Fit TF-IDF on a balanced sample of real Bili + synthetic (content nouns masked to 'C')
    rng = random.Random(20261123)
    bili_indices = [i for i, r in enumerate(records) if r["cohort"] == "bili"]
    synth_indices = [i for i, r in enumerate(records) if r["cohort"] in ("deepseek", "other_models")]
    fit_idx = rng.sample(bili_indices, min(40_000, len(bili_indices))) + rng.sample(synth_indices, min(40_000, len(synth_indices)))
    skeletons_fit = [mask_topic_skeleton(records[i]["text"]) for i in fit_idx]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=5, max_features=12_000, sublinear_tf=True, dtype=np.float32)
    x_fit_sk = vec.fit_transform(skeletons_fit)
    svd = TruncatedSVD(n_components=16, random_state=20261123)
    svd.fit(x_fit_sk)

    # Transform all 350,387 in chunks
    svd_feats = np.zeros((n_all, 16), dtype=np.float32)
    sk_logit = np.zeros(n_all, dtype=np.float32)

    # Fit topic-blind skeleton LogisticRegression for A1 base signal
    y_fit_bin = np.array([1 if records[i]["origin_status"] == "confirmed_ai" else 0 for i in fit_idx], dtype=np.int32)
    sk_clf = LogisticRegression(C=1.0, max_iter=300)
    sk_clf.fit(x_fit_sk, y_fit_bin)

    for start in range(0, n_all, 50_000):
        end = min(n_all, start + 50_000)
        sk_batch = [mask_topic_skeleton(records[i]["text"]) for i in range(start, end)]
        xb = vec.transform(sk_batch)
        svd_feats[start:end] = svd.transform(xb).astype(np.float32)
        sk_logit[start:end] = sk_clf.decision_function(xb).astype(np.float32)

    x_full = np.hstack([style_mat, svd_feats, sk_logit[:, None]])

    # A1_synth_style: 150,000 confirmed synthetic AI vs clean non-DSH Bili human + Gold human (<=30)
    neg_pool = [
        i for i, r in enumerate(records)
        if (r["cohort"] == "bili" and not r["dsh"] and r["previous_label"] == "human" and style_mat[i, 13] == 0 and style_mat[i, 16] == 0)
        or (r["sample_id"] in gold_map and gold_map[r["sample_id"]]["score"] <= 30)
    ]
    if len(neg_pool) > 150_000:
        neg_pool = rng.sample(neg_pool, 150_000)
    a1_idx = synth_indices + neg_pool
    a1_y = np.array([1] * len(synth_indices) + [0] * len(neg_pool), dtype=np.int32)
    a1_groups = [records[i]["video_id"] for i in a1_idx]
    a1_folds = make_group_folds(a1_groups, n_splits=5, seed=20261123)

    a1_prob_all = np.zeros(n_all, dtype=np.float32)
    a1_oof = np.zeros(len(a1_idx), dtype=np.float32)
    x_a1 = x_full[a1_idx]
    for f in range(5):
        tr = np.array([k for k, fk in enumerate(a1_folds) if fk != f])
        va = np.array([k for k, fk in enumerate(a1_folds) if fk == f])
        if has_lgb:
            clf = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.06, num_leaves=47, subsample_for_bin=80000, colsample_bytree=0.8, random_state=2026 + f, n_jobs=-1, verbose=-1)
        else:
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=47, random_state=2026 + f)
        clf.fit(x_a1[tr], a1_y[tr])
        a1_oof[va] = clf.predict_proba(x_a1[va])[:, 1]
        a1_prob_all += clf.predict_proba(x_full)[:, 1] / 5.0
    # Put strict OOF for a1_idx items
    for k, orig_i in enumerate(a1_idx):
        a1_prob_all[orig_i] = a1_oof[k]

    # A2_gold_style: 5-fold GroupKFold regressor directly on Hydrogen XHigh Gold
    gold_indices = [i for i, r in enumerate(records) if r["sample_id"] in gold_map]
    gold_y = np.array([float(gold_map[records[i]["sample_id"]]["score"]) for i in gold_indices], dtype=np.float32)
    gold_groups = [records[i]["video_id"] for i in gold_indices]
    gold_folds = make_group_folds(gold_groups, n_splits=5, seed=20261124)
    x_with_a1 = np.hstack([x_full, a1_prob_all[:, None]])
    x_gold = x_with_a1[gold_indices]

    a2_oof = np.zeros(len(gold_indices), dtype=np.float32)
    a2_all = np.zeros(n_all, dtype=np.float32)
    for f in range(5):
        tr = np.array([k for k, fk in enumerate(gold_folds) if fk != f])
        va = np.array([k for k, fk in enumerate(gold_folds) if fk == f])
        if has_lgb:
            reg = lgb.LGBMRegressor(n_estimators=180, learning_rate=0.04, num_leaves=23, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, random_state=3026 + f, n_jobs=-1, verbose=-1)
        else:
            reg = HistGradientBoostingRegressor(max_iter=140, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=12, random_state=3026 + f)
        reg.fit(x_gold[tr], gold_y[tr])
        # Blend with Ridge to smooth extrapolation
        ridge = Ridge(alpha=50.0)
        ridge.fit(x_gold[tr], gold_y[tr])
        a2_oof[va] = 0.75 * reg.predict(x_gold[va]) + 0.25 * ridge.predict(x_gold[va])
        a2_all += (0.75 * reg.predict(x_with_a1) + 0.25 * ridge.predict(x_with_a1)) / 5.0

    a2_oof = np.clip(a2_oof, 0.0, 100.0)
    a2_all = np.clip(a2_all, 0.0, 100.0)
    for k, orig_i in enumerate(gold_indices):
        a2_all[orig_i] = a2_oof[k]

    bili_mask = np.array([records[i]["cohort"] == "bili" for i in gold_indices])
    print("[Model A] 真实 B 站 Gold OOF 指标:", json.dumps(safe_metrics(gold_y[bili_mask], a2_oof[bili_mask]), ensure_ascii=False), flush=True)
    return a1_prob_all, a2_all, a2_oof, gold_indices, gold_folds


def train_model_b_transformer(
    records,
    gold_map: dict[str, dict],
    gold_indices: list[int],
    gold_folds: list[int],
    a1_prob_all: np.ndarray,
    backbone: str = "hfl/chinese-macbert-base",
    max_len: int = 256,
    epochs: int = 4,
    batch_size: int = 32,
    synth_ablation: tuple[float, ...] = (0.0, 0.08),
):
    """5-Fold GroupKFold Transformer with BCE soft-score + 8-class code head + controlled synth ablation."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Colab 运行时未检测到 CUDA GPU，无法执行 Transformer 蒸馏训练。")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[Model B] GPU={gpu_name}, amp={amp_dtype}, backbone={backbone}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    class CommentDataset(Dataset):
        def __init__(self, texts, scores, codes, weights):
            self.texts = texts
            self.scores = scores
            self.codes = codes
            self.weights = weights

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], self.scores[idx], self.codes[idx], self.weights[idx]

    def collate_fn(batch):
        texts, scores, codes, weights = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(codes, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )

    class DualHeadClassifier(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.drop = nn.Dropout(0.15)
            self.score_head = nn.Linear(hidden, 1)
            self.code_head = nn.Linear(hidden, len(CODE_LIST))

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            h = self.drop(pooled)
            return self.score_head(h).squeeze(-1), self.code_head(h)

    gold_texts = [records[i]["text"] for i in gold_indices]
    gold_scores = [float(gold_map[records[i]["sample_id"]]["score"]) / 100.0 for i in gold_indices]
    gold_codes = [CODE_TO_IDX[gold_map[records[i]["sample_id"]]["code"]] for i in gold_indices]
    gold_weights = [1.0 if records[i]["cohort"] == "bili" else 0.9 if records[i]["cohort"] == "user_confirmed" else 0.7 for i in gold_indices]
    gold_y_100 = np.array([s * 100.0 for s in gold_scores], dtype=np.float32)
    bili_mask = np.array([records[i]["cohort"] == "bili" for i in gold_indices])

    # Style-informative synthetic pool for controlled ablation (A1_synth_style >= 0.55)
    gold_id_set = set(gold_map)
    rng = random.Random(20261125)
    aux_pool = [
        i for i, r in enumerate(records)
        if r["cohort"] in ("deepseek", "other_models") and r["sample_id"] not in gold_id_set and a1_prob_all[i] >= 0.55 and len(r["text"]) >= 16
    ]
    aux_sample = rng.sample(aux_pool, min(len(gold_indices) * 2, len(aux_pool)))

    def train_one_fold(fold_id: int, synth_weight: float, custom_tr_va=None):
        if custom_tr_va is None:
            tr_idx = [k for k, fk in enumerate(gold_folds) if fk != fold_id]
            va_idx = [k for k, fk in enumerate(gold_folds) if fk == fold_id]
        else:
            tr_idx, va_idx = custom_tr_va

        tr_texts = [gold_texts[k] for k in tr_idx]
        tr_scores = [gold_scores[k] for k in tr_idx]
        tr_codes = [gold_codes[k] for k in tr_idx]
        tr_weights = [gold_weights[k] for k in tr_idx]
        if synth_weight > 0.0 and aux_sample:
            take_aux = aux_sample[: min(len(aux_sample), int(len(tr_idx) * 1.5))]
            for i in take_aux:
                tr_texts.append(records[i]["text"])
                # Smooth target scaled by A1 confidence (0.76..0.92), no reason code (-100)
                tr_scores.append(float(np.clip(0.72 + 0.22 * a1_prob_all[i], 0.72, 0.92)))
                tr_codes.append(-100)
                tr_weights.append(synth_weight)

        train_loader = DataLoader(CommentDataset(tr_texts, tr_scores, tr_codes, tr_weights), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(
            CommentDataset([gold_texts[k] for k in va_idx], [gold_scores[k] for k in va_idx], [gold_codes[k] for k in va_idx], [1.0] * len(va_idx)),
            batch_size=64, shuffle=False, collate_fn=collate_fn,
        )

        model = DualHeadClassifier(backbone).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
        total_steps = max(1, len(train_loader) * epochs)
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.1)), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        bce = nn.BCEWithLogitsLoss(reduction="none")
        ce = nn.CrossEntropyLoss(ignore_index=-100)

        for _ in range(epochs):
            model.train()
            for input_ids, attn_mask, y_sc, y_cd, w in train_loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                y_sc, y_cd, w = y_sc.to(device), y_cd.to(device), w.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits_sc, logits_cd = model(input_ids, attn_mask)
                    loss_sc = (bce(logits_sc, y_sc) * w).mean()
                    loss_cd = ce(logits_cd, y_cd)
                    loss = loss_sc + 0.25 * loss_cd
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

        model.eval()
        va_preds = []
        va_code_probs = []
        with torch.no_grad():
            for input_ids, attn_mask, _, _, _ in val_loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits_sc, logits_cd = model(input_ids, attn_mask)
                    va_preds.append((torch.sigmoid(logits_sc) * 100.0).float().cpu().numpy())
                    va_code_probs.append(torch.softmax(logits_cd, dim=-1).float().cpu().numpy())
        return model, va_idx, np.concatenate(va_preds), np.concatenate(va_code_probs)

    # 1. Ablation across synth_ablation weights
    best_weight = synth_ablation[0]
    best_score = -1.0
    best_models = None
    best_oof_scores = None
    best_oof_code_probs = None
    ablation_report = {}

    for sw in synth_ablation:
        t_sw = time.monotonic()
        oof_sc = np.zeros(len(gold_indices), dtype=np.float32)
        oof_cp = np.zeros((len(gold_indices), len(CODE_LIST)), dtype=np.float32)
        fold_models = []
        for f in range(5):
            mdl, va_idx, preds, cprobs = train_one_fold(f, sw)
            oof_sc[va_idx] = preds
            oof_cp[va_idx] = cprobs
            fold_models.append(mdl)
        m_bili = safe_metrics(gold_y_100[bili_mask], oof_sc[bili_mask])
        ablation_report[f"synth_weight_{sw}"] = {**m_bili, "seconds": round(time.monotonic() - t_sw, 1)}
        print(f"[Model B Ablation synth_weight={sw}] Real Bili OOF:", json.dumps(m_bili, ensure_ascii=False), flush=True)
        rank_key = m_bili["spearman"] + m_bili.get("auroc", 0.0)
        if rank_key > best_score:
            best_score = rank_key
            best_weight = sw
            best_models = fold_models
            best_oof_scores = oof_sc
            best_oof_code_probs = oof_cp
        else:
            for mdl in fold_models:
                del mdl
            torch.cuda.empty_cache()

    # 2. Strict Train-Only Holdout Check (train split -> val/test/probe, ZERO leakage!)
    tr_only_idx = [k for k, i in enumerate(gold_indices) if records[i]["split"] == "train"]
    holdout_idx = [k for k, i in enumerate(gold_indices) if records[i]["split"] in ("validation", "test", "probe")]
    strict_metrics = {}
    if len(tr_only_idx) >= 50 and len(holdout_idx) >= 20:
        mdl_strict, _, h_preds, _ = train_one_fold(99, best_weight, custom_tr_va=(tr_only_idx, holdout_idx))
        for sp in ("validation", "test", "probe"):
            sub_mask = np.array([records[gold_indices[k]]["split"] == sp for k in holdout_idx])
            if sub_mask.any():
                y_sub = gold_y_100[holdout_idx][sub_mask]
                p_sub = h_preds[sub_mask]
                if sp == "probe":
                    strict_metrics["probe_u001_u017"] = {
                        "n": int(len(y_sub)),
                        "true_mean": round(float(y_sub.mean()), 2),
                        "pred_mean": round(float(p_sub.mean()), 2),
                        "pred_median": round(float(np.median(p_sub)), 2),
                        "ge_60": int((p_sub >= 60).sum()),
                        "ge_70": int((p_sub >= 70).sum()),
                    }
                else:
                    strict_metrics[f"strict_{sp}"] = safe_metrics(y_sub, p_sub)
        del mdl_strict
        torch.cuda.empty_cache()

    # 3. Full 350,387-row batched inference sorted by length for fast dynamic padding
    print(f"[Model B] 开始全库 350,387 条 5-Fold 推理 (chosen synth_weight={best_weight})...", flush=True)
    t_inf = time.monotonic()
    n_all = len(records)
    order = sorted(range(n_all), key=lambda i: len(records[i]["text"]))
    fold_preds = np.zeros((5, n_all), dtype=np.float32)
    code_probs_all = np.zeros((n_all, len(CODE_LIST)), dtype=np.float32)

    inf_bs = 512
    for mdl in best_models:
        mdl.eval()
    with torch.no_grad():
        for start in range(0, n_all, inf_bs):
            batch_pos = order[start:start + inf_bs]
            enc = tokenizer([records[p]["text"] for p in batch_pos], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            for f, mdl in enumerate(best_models):
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits_sc, logits_cd = mdl(input_ids, attn_mask)
                    fold_preds[f, batch_pos] = (torch.sigmoid(logits_sc) * 100.0).float().cpu().numpy()
                    code_probs_all[batch_pos] += torch.softmax(logits_cd, dim=-1).float().cpu().numpy() / 5.0
            if (start // inf_bs) % 150 == 0 and start > 0:
                print(f"  推理进度 {start:,} / {n_all:,} ({time.monotonic() - t_inf:.1f}s)", flush=True)

    mean_preds = fold_preds.mean(axis=0)
    std_preds = fold_preds.std(axis=0)
    # Restore strict OOF predictions for Gold rows
    for k, orig_i in enumerate(gold_indices):
        mean_preds[orig_i] = best_oof_scores[k]
        code_probs_all[orig_i] = best_oof_code_probs[k]

    for mdl in best_models:
        del mdl
    torch.cuda.empty_cache()
    print(f"[Model B] 全库推理完成，耗时 {time.monotonic() - t_inf:.1f}s", flush=True)

    return {
        "oof_scores": best_oof_scores,
        "all_scores": mean_preds,
        "fold_std": std_preds,
        "code_probs_all": code_probs_all,
        "best_synth_weight": best_weight,
        "ablation": ablation_report,
        "strict_train_holdout": strict_metrics,
        "gpu_name": gpu_name,
    }


def calibrate_ensemble(records, gold_map: dict[str, dict], gold_indices: list[int], gold_folds: list[int], a1_prob, a2_all, a2_oof, model_b_res, model_b2_res=None):
    """Model C: 5-Fold GroupKFold Ridge blending + Isotonic calibration onto 0-100 scale."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge

    gold_y = np.array([float(gold_map[records[i]["sample_id"]]["score"]) for i in gold_indices], dtype=np.float32)
    b_oof = model_b_res["oof_scores"]
    b_all = model_b_res["all_scores"]

    if model_b2_res is not None:
        x_meta_gold = np.column_stack([a1_prob[gold_indices] * 100.0, a2_oof, b_oof, model_b2_res["oof_scores"]])
        x_meta_all = np.column_stack([a1_prob * 100.0, a2_all, b_all, model_b2_res["all_scores"]])
        b_blend_all = 0.5 * b_all + 0.5 * model_b2_res["all_scores"]
        b_std = 0.5 * (model_b_res["fold_std"] + model_b2_res["fold_std"]) + 0.5 * np.abs(b_all - model_b2_res["all_scores"])
    else:
        x_meta_gold = np.column_stack([a1_prob[gold_indices] * 100.0, a2_oof, b_oof])
        x_meta_all = np.column_stack([a1_prob * 100.0, a2_all, b_all])
        b_blend_all = b_all
        b_std = model_b_res["fold_std"]

    meta_oof = np.zeros(len(gold_indices), dtype=np.float32)
    meta_all = np.zeros(len(records), dtype=np.float32)
    for f in range(5):
        tr = np.array([k for k, fk in enumerate(gold_folds) if fk != f])
        va = np.array([k for k, fk in enumerate(gold_folds) if fk == f])
        ridge = Ridge(alpha=10.0)
        ridge.fit(x_meta_gold[tr], gold_y[tr])
        raw_va = np.clip(ridge.predict(x_meta_gold[va]), 0.0, 100.0)
        raw_tr = np.clip(ridge.predict(x_meta_gold[tr]), 0.0, 100.0)
        iso = IsotonicRegression(y_min=0.0, y_max=100.0, out_of_bounds="clip")
        iso.fit(raw_tr, gold_y[tr])
        meta_oof[va] = iso.predict(raw_va)
        meta_all += iso.predict(np.clip(ridge.predict(x_meta_all), 0.0, 100.0)) / 5.0

    for k, orig_i in enumerate(gold_indices):
        meta_all[orig_i] = meta_oof[k]

    uncertainty = b_std + 0.5 * np.abs(a2_all - b_blend_all)
    bili_mask = np.array([records[i]["cohort"] == "bili" for i in gold_indices])
    uc_mask = np.array([records[i]["cohort"] == "user_confirmed" for i in gold_indices])

    metrics = {
        "ensemble_all_gold_oof": safe_metrics(gold_y, meta_oof),
        "ensemble_bili_gold_oof": safe_metrics(gold_y[bili_mask], meta_oof[bili_mask]),
        "model_b_bili_gold_oof": safe_metrics(gold_y[bili_mask], b_oof[bili_mask]),
        "model_a_bili_gold_oof": safe_metrics(gold_y[bili_mask], a2_oof[bili_mask]),
    }
    if model_b2_res is not None:
        metrics["model_b2_bili_gold_oof"] = safe_metrics(gold_y[bili_mask], model_b2_res["oof_scores"][bili_mask])
    if uc_mask.any():
        metrics["user_confirmed_oof"] = safe_metrics(gold_y[uc_mask], meta_oof[uc_mask])
    return meta_oof, meta_all, uncertainty, metrics


def select_active_learning_candidates(records, gold_map: dict[str, dict], meta_all: np.ndarray, uncertainty: np.ndarray, a2_all: np.ndarray, b_all: np.ndarray, out_path: Path):
    """Select ~650 highest-leverage unscored comments for Round 2 Hydrogen (XHigh) adjudication."""
    scored_ids = set(gold_map)
    unscored_bili = [i for i, r in enumerate(records) if r["cohort"] == "bili" and r["sample_id"] not in scored_ids]
    unscored_synth = [i for i, r in enumerate(records) if r["cohort"] in ("deepseek", "other_models") and r["sample_id"] not in scored_ids]

    selected = []
    seen = set()

    def take(pool, limit, stratum):
        added = 0
        for i in pool:
            if added >= limit:
                break
            uid = records[i]["sample_id"]
            if uid in seen:
                continue
            seen.add(uid)
            selected.append({
                "index": i,
                "sample_id": uid,
                "stratum": stratum,
                "distilled_score": round(float(meta_all[i]), 2),
                "uncertainty": round(float(uncertainty[i]), 2),
                "previous_label": records[i]["previous_label"],
                "split": records[i]["split"],
                "dsh": records[i]["dsh"],
            })
            added += 1

    # 1. 250 biggest disagreements in DSH (old_human scoring >=60 or old_ai scoring <=35)
    dsh_fn = sorted(
        [i for i in unscored_bili if records[i]["dsh"] and records[i]["previous_label"] == "human" and meta_all[i] >= 60.0],
        key=lambda i: -meta_all[i],
    )
    dsh_fp = sorted(
        [i for i in unscored_bili if records[i]["dsh"] and records[i]["previous_label"] == "ai" and meta_all[i] <= 35.0],
        key=lambda i: meta_all[i],
    )
    take(dsh_fn, 150, "r2_dsh_old_human_high_ai")
    take(dsh_fp, 100, "r2_dsh_old_ai_low_score")

    # 2. 200 highest-uncertainty DSH boundary rows (35-70, prioritizing validation & test splits first)
    split_prio = {"validation": 0, "test": 0, "train": 1}
    dsh_unc = sorted(
        [i for i in unscored_bili if records[i]["dsh"] and 35.0 <= meta_all[i] <= 70.0],
        key=lambda i: (split_prio.get(records[i]["split"], 2), -uncertainty[i]),
    )
    take(dsh_unc, 200, "r2_dsh_high_uncertainty_boundary")

    # 3. 120 hard non-DSH high-scoring rows (>= 65)
    nondsh_high = sorted(
        [i for i in unscored_bili if not records[i]["dsh"] and meta_all[i] >= 65.0],
        key=lambda i: (-meta_all[i], -uncertainty[i]),
    )
    take(nondsh_high, 120, "r2_nondsh_high_score")

    # 4. 80 synthetic boundary/surprise rows (style vs transformer disagreement or unexpectedly low scores)
    synth_surprise = sorted(
        unscored_synth,
        key=lambda i: -(abs(a2_all[i] - b_all[i]) + (50.0 - min(meta_all[i], 50.0))),
    )
    take(synth_surprise, 80, "r2_synth_surprise")

    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"active_learning_candidates": len(selected), "strata": dict(Counter(x["stratum"] for x in selected))}, ensure_ascii=False))
    return selected


def pick_consistent_code(score: float, code_probs: np.ndarray) -> str:
    """Select primary reason code consistent with score band."""
    if score >= 70.0:
        allowed = [CODE_TO_IDX[c] for c in ("A", "R", "E", "T")]
    elif score <= 30.0:
        return "H"
    else:
        allowed = [CODE_TO_IDX[c] for c in ("P", "S", "Q")]
    best = max(allowed, key=lambda idx: float(code_probs[idx]))
    return CODE_LIST[best]


def render_dashboard_html(report: dict, out_path: Path) -> None:
    rows_html = []
    for cohort, vals in report["cohorts"].items():
        rows_html.append(
            f"<tr><td>{html.escape(cohort)}</td><td>{vals.get('ai_like', 0):,}</td>"
            f"<td>{vals.get('uncertain', 0):,}</td><td>{vals.get('human_like', 0):,}</td></tr>"
        )
    dsh_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>"
        for k, v in sorted(report["dsh_label_transitions"].items())
    )
    synth_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v.get('count', 0):,}</td><td>{v.get('mean', 0):.1f}</td>"
        f"<td>{v.get('ge_70', 0):,}</td><td>{v.get('le_30', 0):,}</td></tr>"
        for k, v in sorted(report["confirmed_ai_detectability_by_length"].items())
    )
    m = report["metrics"]["ensemble_bili_gold_oof"]
    content = f"""<div style="font-family:system-ui;padding:14px;border:1px solid #94a3b8;border-radius:10px;max-width:920px">
<h2 style="margin:0 0 6px">MeguvizModel 06 · Hydrogen XHigh 全库重标已完成</h2>
<div>全量核验 <b>{report['total_rows']:,}</b> 条 · Hydrogen 直接盲评 <b>{report['gold_rows']:,}</b> 条 · GPU 蒸馏补齐 <b>{report['total_rows'] - report['gold_rows']:,}</b> 条</div>
<div style="margin:6px 0">真实 B 站 OOF Spearman=<b>{m['spearman']:.4f}</b> · AUROC=<b>{m.get('auroc', 0):.4f}</b> · MAE=<b>{m['mae']:.2f}</b></div>
<h3>一、各队列重标分布（150,017 确认 AI 保持 origin_status=confirmed_ai）</h3>
<table border="1" cellpadding="4" cellspacing="0"><tr><th>来源队列</th><th>明确 AI (ai_like)</th><th>待复核 (uncertain)</th><th>偏真人 (human_like)</th></tr>{''.join(rows_html)}</table>
<h3>二、DSH 评论新旧标签迁移</h3>
<table border="1" cellpadding="4" cellspacing="0"><tr><th>旧标签 → 新标签</th><th>条数</th></tr>{dsh_rows}</table>
<h3>三、150,017 条确认 AI 字面可辨识度分布（按字数）</h3>
<table border="1" cellpadding="4" cellspacing="0"><tr><th>长度段</th><th>总数</th><th>均分</th><th>&gt;=70 (可辨)</th><th>&lt;=30 (不可辨短噪)</th></tr>{synth_rows}</table>
<div>导出校验 SHA256: <code>{report['export_sha256']}</code></div>
</div>"""
    out_path.write_text(content, encoding="utf-8")


def export_final_outputs(records, gold_map: dict[str, dict], meta_oof: np.ndarray, meta_all: np.ndarray, a2_all: np.ndarray, b_all: np.ndarray, uncertainty: np.ndarray, code_probs_all: np.ndarray, rubric_text: str, order_sha: str, metrics_bundle: dict, out_dir: Path):
    rubric_sha = hashlib.sha256(rubric_text.encode("utf-8")).hexdigest()

    export_gz = out_dir / "bili-hydrogen-rescored-350387.jsonl.gz"
    cohorts = defaultdict(Counter)
    dsh_transitions = Counter()
    synth_by_len = defaultdict(list)

    with gzip.open(export_gz, "wt", encoding="utf-8", newline="\n", compresslevel=5) as gz:
        for i, r in enumerate(records):
            uid = r["sample_id"]
            oof_val = float(meta_all[i])
            if uid in gold_map:
                g = gold_map[uid]
                score = float(g["score"])
                score_int = int(g["score"])
                code = g["code"]
                judg = judgment_v2(score_int, code)
                source = "hydrogen_xhigh_direct"
                hx_score = score_int
                hx_code = code
            else:
                score = round(oof_val, 2)
                score_int = int(round(oof_val))
                code = pick_consistent_code(score, code_probs_all[i])
                judg = judgment_v2(score, code)
                source = "colab_gpu_distilled_from_hydrogen_xhigh"
                hx_score = None
                hx_code = None

            row = {
                "sample_id": uid,
                "text": r["text"],
                "cohort": r["cohort"],
                "split": r["split"],
                "dsh": r["dsh"],
                "video_id": r["video_id"],
                "origin_status": r["origin_status"],
                "score": score,
                "score_int": score_int,
                "reason_code": code,
                "reason": CODES_V2[code],
                "judgment": judg,
                "score_source": source,
                "hydrogen_xhigh_score": hx_score,
                "hydrogen_xhigh_code": hx_code,
                "distilled_oof_score": round(oof_val, 2),
                "style_gbdt_score": round(float(a2_all[i]), 2),
                "transformer_oof_score": round(float(b_all[i]), 2),
                "uncertainty": round(float(uncertainty[i]), 2),
                "previous_label": r["previous_label"],
                "previous_source": r["previous_source"],
                "generator_model": r["generator_model"],
                "format": r["format"],
                "rubric_version": "hydrogen-xhigh-v2",
                "rubric_sha256": rubric_sha,
            }
            gz.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            cohorts[r["cohort"]][judg] += 1
            if r["cohort"] == "bili" and r["dsh"]:
                dsh_transitions[f"{r['previous_label']}->{judg}"] += 1
            if r["origin_status"] == "confirmed_ai":
                band = "1-8字" if len(r["text"]) <= 8 else "9-25字" if len(r["text"]) <= 25 else "26字+"
                synth_by_len[band].append(score)

    export_sha = sha256_file(export_gz)
    synth_summary = {
        band: {
            "count": len(vals),
            "mean": round(sum(vals) / len(vals), 2),
            "ge_70": sum(v >= 70.0 for v in vals),
            "le_30": sum(v <= 30.0 for v in vals),
        }
        for band, vals in synth_by_len.items()
    }
    report = {
        "total_rows": len(records),
        "gold_rows": len(gold_map),
        "input_order_sha256": order_sha,
        "rubric_sha256": rubric_sha,
        "export_file": export_gz.name,
        "export_sha256": export_sha,
        "cohorts": {k: dict(v) for k, v in cohorts.items()},
        "dsh_label_transitions": dict(dsh_transitions),
        "confirmed_ai_detectability_by_length": synth_summary,
        "metrics": metrics_bundle,
    }
    (out_dir / "hydrogen-relabel-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "hydrogen-relabel-metrics.json").write_text(json.dumps(metrics_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_dashboard_html(report, out_dir / "hydrogen-dashboard.html")
    print(json.dumps({"exported": export_gz.name, "sha256": export_sha, "gold_rows": len(gold_map)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--rubric", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument("--stage", choices=["round1", "final"], default="round1")
    parser.add_argument("--backbone", default="hfl/chinese-macbert-base")
    parser.add_argument("--second-backbone", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_test(args.data.resolve(), args.gold.resolve() if args.gold else None, args.rubric.resolve() if args.rubric else None)
        return

    if not args.gold or not args.rubric:
        raise ValueError("正式执行需要 --gold 和 --rubric")
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    records, order_sha = load_frozen_corpus(args.data.resolve())
    gold_map = load_gold(args.gold.resolve())
    rubric_text = args.rubric.resolve().read_text(encoding="utf-8")
    print(f"已加载冻结语料 {len(records):,} 条，Hydrogen XHigh Gold {len(gold_map):,} 条。", flush=True)

    style_mat = extract_style_features([r["text"] for r in records])
    a1_prob, a2_all, a2_oof, gold_indices, gold_folds = train_model_a(records, gold_map, style_mat)
    model_b_res = train_model_b_transformer(
        records, gold_map, gold_indices, gold_folds, a1_prob,
        backbone=args.backbone, epochs=args.epochs,
    )
    model_b2_res = None
    best_sw = model_b_res["best_synth_weight"]
    ab_secs = model_b_res["ablation"].get(f"synth_weight_{best_sw}", {}).get("seconds", 999)
    sec_name = "hfl/chinese-roberta-wwm-ext" if args.second_backbone == "auto" else args.second_backbone
    if args.stage == "final" and sec_name.lower() not in ("none", "") and ab_secs < 450:
        print(f"[Model B2] 首选骨干训练迅速 ({ab_secs}s < 450s)，启用第二骨干 {sec_name} 增强集成...", flush=True)
        model_b2_res = train_model_b_transformer(
            records, gold_map, gold_indices, gold_folds, a1_prob,
            backbone=sec_name, epochs=args.epochs, synth_ablation=(best_sw,),
        )
        b_blend = 0.5 * model_b_res["all_scores"] + 0.5 * model_b2_res["all_scores"]
        code_blend = 0.5 * model_b_res["code_probs_all"] + 0.5 * model_b2_res["code_probs_all"]
    else:
        b_blend = model_b_res["all_scores"]
        code_blend = model_b_res["code_probs_all"]

    meta_oof, meta_all, uncertainty, ens_metrics = calibrate_ensemble(
        records, gold_map, gold_indices, gold_folds, a1_prob, a2_all, a2_oof, model_b_res, model_b2_res
    )
    metrics_bundle = {
        **ens_metrics,
        "model_b_best_synth_weight": best_sw,
        "model_b_ablation": model_b_res["ablation"],
        "strict_train_holdout": model_b_res["strict_train_holdout"],
        "second_backbone": sec_name if model_b2_res is not None else None,
        "gpu_name": model_b_res["gpu_name"],
        "gold_rows": len(gold_map),
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
    }
    print("[Model C Ensemble Metrics]", json.dumps(metrics_bundle, ensure_ascii=False, indent=2), flush=True)

    if args.stage == "round1":
        (out_dir / "r1-metrics.json").write_text(json.dumps(metrics_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        select_active_learning_candidates(
            records, gold_map, meta_all, uncertainty, a2_all, b_blend,
            out_dir / "active-learning-r2-candidates.json",
        )
    else:
        export_final_outputs(
            records, gold_map, meta_oof, meta_all, a2_all, b_blend,
            uncertainty, code_blend, rubric_text, order_sha, metrics_bundle, out_dir,
        )


if __name__ == "__main__":
    main()
