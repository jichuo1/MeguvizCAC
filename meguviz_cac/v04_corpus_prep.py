"""Canonical 1.25M+ corpus merger, zero-leakage deduplicator & 11-subgroup GroupDRO tagger for MeguvizCAC v0.4-DualLoop."""

import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bili_round01_data import body_key  # noqa: E402

from .inkling_distill import detect_lang_tag  # noqa: E402
from .train_student import CODE_TO_IDX, PRAG_TAGS, load_pairs, load_qwen_fused, load_rescored_corpus  # noqa: E402
from .v03_runtime import EXTRA_CROSS_PLATFORM_TRAPS  # noqa: E402
from .v03_tokenizer import prepare_text  # noqa: E402
from .v04_bili_ingest import DEFAULT_LEGACY_DB, derive_era_bucket  # noqa: E402
from .v04_runtime import MULTILINGUAL_TRAPS  # noqa: E402


V04_GROUPS = (
    "pre_llm_human",
    "structured_human_hard",
    "multilingual_world_knowledge",
    "tech_knowledge_real",
    "ultra_short_ai",
    "evasive_plugin_ai",
    "standard_plugin_ai",
    "persona_space_kaomoji",
    "literary_boundary",
    "dsh_hard_slice",
    "current_2026_real",
    "general_real",
)
GROUP_TO_IDX = {g: i for i, g in enumerate(V04_GROUPS)}

DEFAULT_EVASIVE_MODES = frozenset(
    {
        "auto_comment_ultra_short",
        "auto_reply_plugin_thread",
        "auto_comment_concise_insight",
        "lyrical_or_meme_plugin",
        "summary_plugin_casual",
    }
)
TECH_PARTITIONS = frozenset({"科技数码", "知识", "汽车", "资讯"})
PERSONA_TRAP_RE = re.compile(r"本[鱼鲸]|参数|人设|归档|备忘录精灵|记一下|——")
CJK_SPACE_RE = re.compile(r"[\u4e00-\u9fff][ \t]+[\u4e00-\u9fff]")
KAOMOJI_RE = re.compile(r"[(\[（【][^()\n\u4e00-\u9fff]{2,14}[)\]）】]")
STRUCTURED_HUMAN_RE = re.compile(
    r"(?:(?:^|\s)[1-9一二三四五][\.、\)）]|[①②③④⑤]|《[^》]{1,24}》|"
    r"\d+(?:\.\d+)?\s*/\s*10|评分|打分|星级|推荐度|满分|"
    r"网盘|提取码|夸克|百度云|指路|BV[A-Za-z0-9]{10}|空降|\d{1,2}:\d{2})"
)

BENCH_TRAP_TEXTS = (
    "笑死我了，这个插件装完直接起飞[doge]",
    "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子",
    "节点参考Blender材质节点这个思路挺对，接口丑不丑先不说，能白盒点开看子代理在干啥就很难得。",
    "大肥鱼在吗？我今天闲得发慌，想找条同类聊聊天，你平时都在干嘛呀？",
    "表情包已经出现了[妙啊][妙啊]",
    "观感评分：画面 9/10，配乐 8.5/10，节奏 7/10，整体算近期很稳的良心作了。",
    "两点真实感受：1. 前半段科普节奏很顺，没废话；2. 后面实操演示字体有点小，手机看费眼。",
    "查了下《现代汉语词典》，“敷衍”本来就可以做形容词用，弹幕没必要抓着这个字眼吵半天。",
    "指路原版开源仓库 https://github.com/example/demo ，配环境卡住可以看置顶评论里的报错对照。",
    "回复 :本鱼看了 是不是那条说 让AI刷B站学人文关怀 本鱼承认 本鱼今天已经有点学坏了",
    "回复 :她挺可爱的，不过本鱼就是这个参数，换不动啦 (￣▽￣)",
    "渲染省人眼不省token，卡片留在上下文里每轮重读，画得越漂亮账越贵",
    "要的不是一次点击，是整只手。给权限容易，收回来难。",
    "373行字幕读完：前两步重在定位边界，最后一步才是真正决定通过率的关键——慢一点反而不容易返工。",
    *(txt for _, txt in EXTRA_CROSS_PLATFORM_TRAPS),
    *(txt for _, txt in MULTILINGUAL_TRAPS),
)


def _norm_hash(text: str) -> str:
    return hashlib.sha256(body_key(text).encode("utf-8")).hexdigest()


def _open_read(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def has_stress_marker(text: str) -> bool:
    return bool(PERSONA_TRAP_RE.search(text) or CJK_SPACE_RE.search(text) or KAOMOJI_RE.search(text))


def is_structured_human(text: str) -> bool:
    return bool(STRUCTURED_HUMAN_RE.search(text)) or len(text.strip()) >= 80


def is_multilingual_wk(text: str, row: dict) -> bool:
    lt = str(row.get("lang_tag") or detect_lang_tag(text, str(row.get("partition") or "")))
    if lt in ("en_dominant", "ja_dominant", "ko_dominant", "eu_other", "zh_en_mixed"):
        return True
    if row.get("cohort") in ("inkling_wk", "v04_multilingual"):
        return True
    src = str(row.get("score_source") or "").lower()
    if "inkling" in src and lt != "zh_general":
        return True
    return False


def assign_group_id(row: dict, evasive_modes: frozenset[str] = DEFAULT_EVASIVE_MODES) -> str:
    """Assign deterministic 12-way GroupDRO subgroup label aligned with v0.4 empirical & OOD findings."""
    text = str(row.get("text") or "")
    cohort = str(row.get("cohort") or "")
    era = str(row.get("era_bucket") or "")
    origin = str(row.get("origin_status") or "")
    plugin_mode = str(row.get("plugin_mode") or "")
    length_bucket = str(row.get("length_bucket") or "")
    partition = str(row.get("partition") or "")
    score = row.get("score")
    score_f = float(score) if score is not None else 50.0
    lt = str(row.get("lang_tag") or detect_lang_tag(text, partition))
    is_zh_core = lt in ("zh_general", "zh_world_knowledge")

    is_confirmed_ai = origin == "confirmed_ai" or cohort in ("deepseek", "other_models", "synth_250k", "inkling_wk")
    if (era == "pre_llm" or (row.get("has_score", True) and score_f <= 28.0)) and is_structured_human(text) and is_zh_core:
        return "structured_human_hard"
    if is_zh_core and has_stress_marker(text) and (not is_confirmed_ai or not (len(text.strip()) <= 22 or length_bucket == "ultra_short")):
        return "persona_space_kaomoji"
    if is_multilingual_wk(text, row):
        return "multilingual_world_knowledge"

    if is_confirmed_ai:
        if len(text.strip()) <= 22 or length_bucket == "ultra_short":
            return "ultra_short_ai"
        if has_stress_marker(text) or plugin_mode == "persona_agent_plugin":
            return "persona_space_kaomoji"
        if plugin_mode in evasive_modes:
            return "evasive_plugin_ai"
        return "standard_plugin_ai"

    if era == "pre_llm":
        return "pre_llm_human"
    if row.get("dsh"):
        return "dsh_hard_slice"
    if has_stress_marker(text):
        return "persona_space_kaomoji"
    if row.get("score_source") == "hydrogen_xhigh_direct" or (30.0 <= score_f <= 70.0 and row.get("has_score", True)):
        return "literary_boundary"
    if partition in TECH_PARTITIONS:
        return "tech_knowledge_real"
    if era == "current_2026":
        return "current_2026_real"
    return "general_real"


def load_teacher_scores(path: Path | None) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    if path is None or not path.is_file():
        return scores
    with _open_read(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            scores[r["sample_id"]] = r
    return scores


def load_qwen_distill_map(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.is_file():
        return out
    with _open_read(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") not in (None, "train"):
                raise ValueError(f"防泄漏校验失败: qwen_distill 发现非 train 样本 {r.get('sample_id')}")
            out[r["sample_id"]] = r
    return out


def merge_v04_corpus(
    stage1_data: Path,
    qwen_data: Path | None,
    qwen_distill: Path | None,
    pairs_data: Path | None,
    v04_synth: Path | None,
    v04_real_jsonl: Path | None,
    v04_real_db: Path | None,
    legacy_db: Path | None = DEFAULT_LEGACY_DB,
    teacher_scores_path: Path | None = None,
    v04_qwen_distill: Path | None = None,
    v04_inkling_distill: Path | None = None,
    v04_joint_distill: Path | None = None,
    v04_multilingual_corpus: Path | None = None,
    evasive_modes: frozenset[str] = DEFAULT_EVASIVE_MODES,
    export_out: Path | None = None,
) -> dict:
    """Merge Stage-1/2 + v04 +250k synth + v04 1M real + Inkling/Joint 4-teacher distill with zero held-out leakage."""
    t0 = time.monotonic()
    stage1_rows = load_rescored_corpus(stage1_data)
    qwen_map = load_qwen_fused(qwen_data) if qwen_data and qwen_data.is_file() else {}
    qwen_dist = load_qwen_distill_map(qwen_distill)
    v04_qwen_map = load_qwen_distill_map(v04_qwen_distill)
    v04_inkling_map = load_qwen_distill_map(v04_inkling_distill)
    v04_joint_map = load_qwen_distill_map(v04_joint_distill)
    teacher_map = load_teacher_scores(teacher_scores_path)

    # 1. Build ironclad forbidden sets from all val/test/user_confirmed rows + runtime stress traps
    forbidden_hashes: set[str] = set()
    forbidden_norm_texts: set[str] = set()
    for r in stage1_rows:
        if r["split"] != "train" or r.get("cohort") == "user_confirmed":
            forbidden_hashes.add(_norm_hash(r["text"]))
            forbidden_norm_texts.add(prepare_text(r["text"]))
    for txt in BENCH_TRAP_TEXTS:
        forbidden_hashes.add(_norm_hash(txt))
        forbidden_norm_texts.add(prepare_text(txt))

    pairs = load_pairs(pairs_data, forbidden_norm_texts) if pairs_data and pairs_data.is_file() else []

    # Legacy metadata lookup so legacy 200k rows retain exact partition/era tags when present
    legacy_meta_by_sid: dict[str, tuple[str, str]] = {}
    if legacy_db is not None and legacy_db.is_file():
        with closing(sqlite3.connect(f"file:{legacy_db.as_posix()}?mode=ro", uri=True, timeout=30)) as con:
            for sid, part, disp, cap in con.execute(
                "SELECT sample_id, json_extract(metadata, '$.sampled_from_channel'), "
                "json_extract(metadata, '$.posted_time_display'), json_extract(metadata, '$.capture_date') FROM comments"
            ):
                era = derive_era_bucket(None, disp or "", None, cap or "")
                legacy_meta_by_sid[sid] = (str(part or "未分类"), era)

    seen_train_hashes: set[str] = set()
    source_counts: Counter = Counter()
    split_counts: Counter = Counter()
    group_counts: Counter = Counter()
    dedup_skipped: Counter = Counter()
    leak_blocked: Counter = Counter()
    teacher_src_counts: Counter = Counter()
    hard_human_neg_count = 0

    writer = None
    staging = None
    if export_out is not None:
        export_out.parent.mkdir(parents=True, exist_ok=True)
        staging = export_out.with_suffix(".writing")
        writer = gzip.open(staging, "wt", encoding="utf-8", newline="\n", compresslevel=5)

    def _emit(rec: dict) -> None:
        if writer is not None:
            writer.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    try:
        # Stage A: Preserve 100% of Stage-1 rows verbatim with enriched GroupDRO + Stage-2 fused metadata
        for r in stage1_rows:
            sid = r["sample_id"]
            txt = r["text"]
            split = r["split"]
            cohort = r["cohort"]
            bhash = _norm_hash(txt)
            if split == "train" and cohort != "user_confirmed":
                seen_train_hashes.add(bhash)

            part, era = legacy_meta_by_sid.get(sid, ("未分类", "mid_llm"))
            score = float(r["score"])
            code = r.get("reason_code", "")
            unc = float(r.get("uncertainty", 10.0))
            prag = [-1.0] * len(PRAG_TAGS)
            qwen_intent = ""
            qwen_group = ""
            w = 3.0 if r.get("score_source") == "hydrogen_xhigh_direct" else (1.0 if (unc < 15.0 and (score <= 28.0 or score >= 72.0)) else 0.35)
            if r.get("dsh"):
                w *= 1.5
            if cohort in ("deepseek", "other_models"):
                w = 0.25 if sid in qwen_map else 0.15

            if split == "train" and sid in qwen_map:
                qr = qwen_map[sid]
                qw = float(qr.get("quality_weight", w))
                score = float(qr.get("fused_score", score))
                if qr.get("reason_code") in CODE_TO_IDX:
                    code = qr["reason_code"]
                if isinstance(qr.get("pragmatic_tags"), list) and len(qr["pragmatic_tags"]) == len(PRAG_TAGS):
                    prag = [float(x) for x in qr["pragmatic_tags"]]
                qwen_intent = str(qr.get("semantic_intent") or "")
                if qw <= 0.0:
                    w = 0.0
                else:
                    w = qw * (1.2 if r.get("dsh") else 1.0)
            if split == "train" and sid in qwen_dist:
                qd = qwen_dist[sid]
                qwen_group = str(qd.get("pool_group") or "")
                if not qwen_intent:
                    qwen_intent = str(qd.get("semantic_intent") or "")

            enriched = {
                **r,
                "partition": part,
                "era_bucket": era if cohort == "bili" else "",
                "score": round(score, 2),
                "reason_code": code,
                "train_weight": round(float(w), 4),
                "prag_vec": prag,
                "semantic_intent": qwen_intent,
                "pool_group": qwen_group,
                "has_score": True,
            }
            gid = assign_group_id(enriched, evasive_modes=evasive_modes)
            enriched["group_id"] = gid
            _emit(enriched)

            source_counts[f"stage1_{cohort}"] += 1
            split_counts[split] += 1
            if split == "train":
                group_counts[gid] += 1

        # Stage B: +250k Confirmed-AI plugin corpus (strictly split='train', non-leaking, deduped)
        if v04_synth is not None and v04_synth.is_file():
            with _open_read(v04_synth) as f:
                for line in f:
                    if not line.strip():
                        continue
                    sr = json.loads(line)
                    txt = str(sr.get("text") or "")
                    bh = _norm_hash(txt)
                    pt = prepare_text(txt)
                    if bh in forbidden_hashes or pt in forbidden_norm_texts:
                        leak_blocked["v04_synth"] += 1
                        continue
                    if bh in seen_train_hashes:
                        dedup_skipped["v04_synth"] += 1
                        continue
                    seen_train_hashes.add(bh)

                    pmode = str(sr.get("plugin_mode") or "")
                    lbucket = str(sr.get("length_bucket") or "")
                    w_synth = 0.85 if (lbucket == "ultra_short" or pmode in evasive_modes) else 0.55
                    rec = {
                        "sample_id": sr["sample_id"],
                        "text": txt,
                        "cohort": "synth_250k",
                        "origin_status": "confirmed_ai",
                        "split": "train",
                        "score": 94.0,
                        "reason_code": "A",
                        "judgment": "ai_like",
                        "score_source": "v04_confirmed_ai_plugin",
                        "uncertainty": 3.0,
                        "dsh": False,
                        "partition": str(sr.get("partition") or ""),
                        "plugin_mode": pmode,
                        "length_bucket": lbucket,
                        "family": str(sr.get("family") or ""),
                        "era_bucket": "current_2026",
                        "train_weight": w_synth,
                        "prag_vec": [-1.0] * len(PRAG_TAGS),
                        "semantic_intent": "",
                        "has_score": True,
                    }
                    gid = assign_group_id(rec, evasive_modes=evasive_modes)
                    rec["group_id"] = gid
                    _emit(rec)
                    source_counts["v04_synth_250k"] += 1
                    split_counts["train"] += 1
                    group_counts[gid] += 1

        # Stage C: New v04_1m Real Bilibili corpus (excluding legacy_200k already inside stage1_rows)
        def _handle_real_row(sid: str, vid: str, part: str, era: str, lvl: str, txt: str) -> None:
            nonlocal hard_human_neg_count
            bh = _norm_hash(txt)
            pt = prepare_text(txt)
            if bh in forbidden_hashes or pt in forbidden_norm_texts:
                leak_blocked["v04_real"] += 1
                return
            if bh in seen_train_hashes:
                dedup_skipped["v04_real"] += 1
                return
            seen_train_hashes.add(bh)

            ts = teacher_map.get(sid)
            vj = v04_joint_map.get(sid)
            vq = v04_qwen_map.get(sid)
            vi = v04_inkling_map.get(sid)
            lt = detect_lang_tag(txt, part)

            prag_v = [-1.0] * len(PRAG_TAGS)
            q_intent = ""
            if lt in ("en_dominant", "ja_dominant", "ko_dominant", "eu_other"):
                wq, wi = 0.30, 0.70
            elif lt == "zh_en_mixed":
                wq, wi = 0.50, 0.50
            else:
                wq, wi = 0.65, 0.35

            if vj is not None:
                vtags = vj.get("pragmatic_tags")
                if isinstance(vtags, list) and len(vtags) == len(PRAG_TAGS):
                    prag_v = [float(x) for x in vtags]
                q_intent = str(vj.get("semantic_intent") or "")
            elif vq is not None and vi is not None:
                vqt = vq.get("pragmatic_tags")
                vit = vi.get("pragmatic_tags")
                if isinstance(vqt, list) and len(vqt) == len(PRAG_TAGS) and isinstance(vit, list) and len(vit) == len(PRAG_TAGS):
                    prag_v = [round(wq * float(a) + wi * float(b), 4) for a, b in zip(vqt, vit)]
                elif isinstance(vqt, list) and len(vqt) == len(PRAG_TAGS):
                    prag_v = [float(x) for x in vqt]
                elif isinstance(vit, list) and len(vit) == len(PRAG_TAGS):
                    prag_v = [float(x) for x in vit]
                q_intent = str((vi.get("semantic_intent") or vq.get("semantic_intent")) if lt in ("en_dominant", "ja_dominant", "ko_dominant", "eu_other", "zh_en_mixed") else (vq.get("semantic_intent") or vi.get("semantic_intent")) or "")
            elif vq is not None:
                vtags = vq.get("pragmatic_tags")
                if isinstance(vtags, list) and len(vtags) == len(PRAG_TAGS):
                    prag_v = [float(x) for x in vtags]
                q_intent = str(vq.get("semantic_intent") or "")
            elif vi is not None:
                vtags = vi.get("pragmatic_tags")
                if isinstance(vtags, list) and len(vtags) == len(PRAG_TAGS):
                    prag_v = [float(x) for x in vtags]
                q_intent = str(vi.get("semantic_intent") or "")

            if ts is not None:
                sc = float(ts["teacher_score"])
                unc = float(ts.get("teacher_uncertainty", 6.0))
                src = str(ts.get("label_source") or "multi_teacher_ensemble")
                hard_neg = bool(ts.get("hard_human_neg", False))
                has_sc = True
            elif era == "pre_llm":
                sc = 3.0
                unc = 2.0
                src = "pre_llm_human_prior"
                hard_neg = False
                has_sc = True
            else:
                sc = 25.0
                unc = 25.0
                src = "unscored_pending_teacher"
                hard_neg = False
                has_sc = False

            if hard_neg:
                hard_human_neg_count += 1
            if era == "pre_llm":
                w_real = 1.6 if hard_neg else (1.5 if (vj is not None or vq is not None or vi is not None) else 0.85)
                code = "H"
            elif vj is not None and ("fused_score" in vj or "score" in vj):
                sc = float(vj.get("fused_score", vj.get("score", sc)))
                unc = float(vj.get("uncertainty", 4.0))
                src = str(vj.get("score_source") or "v04_four_teacher_joint")
                has_sc = True
                w_real = float(vj.get("quality_weight", 1.8))
                code = str(vj.get("reason_code") or ("A" if sc >= 70.0 else ("H" if sc <= 30.0 else "E")))
            elif vq is not None and "qwen_score" in vq and vi is not None and "inkling_score" in vi:
                q_sc = float(vq["qwen_score"])
                i_sc = float(vi["inkling_score"])
                sc = wq * q_sc + wi * i_sc
                diff_oi = abs(q_sc - i_sc)
                unc = max(4.0, diff_oi)
                src = "v04_qwen_inkling_joint" if diff_oi < 18.0 else "v04_qwen_inkling_disagree"
                has_sc = True
                w_real = 1.8 if diff_oi < 18.0 else 0.85
                code = "A" if (sc >= 75.0 and prag_v[2] >= 0.5) else ("E" if sc >= 70.0 else ("H" if sc <= 30.0 else "P"))
            elif vq is not None and "qwen_score" in vq:
                q_sc = float(vq["qwen_score"])
                sc = (0.60 * q_sc + 0.40 * sc) if has_sc else q_sc
                unc = abs(q_sc - sc) if has_sc else 4.0
                src = "v04_qwen38_bf16_fused"
                has_sc = True
                w_real = 1.5
                code = "A" if sc >= 70.0 else ("H" if sc <= 30.0 else "E")
            elif vi is not None and "inkling_score" in vi:
                i_sc = float(vi["inkling_score"])
                alpha_i = 0.75 if lt in ("en_dominant", "ja_dominant", "ko_dominant", "eu_other") else 0.60
                sc = (alpha_i * i_sc + (1.0 - alpha_i) * sc) if has_sc else i_sc
                unc = abs(i_sc - sc) if has_sc else 5.0
                src = "v04_inkling2_mini_fused"
                has_sc = True
                w_real = 1.6
                code = "E" if sc >= 70.0 else ("H" if sc <= 30.0 else "P")
            elif not has_sc:
                w_real = 0.0
                code = ""
            elif unc <= 10.0 and (sc <= 25.0 or sc >= 75.0):
                w_real = 0.65
                code = "A" if sc >= 70.0 else "H"
            else:
                w_real = 0.30
                code = "E"

            rec = {
                "sample_id": sid,
                "video_id": vid,
                "text": txt,
                "cohort": "v04_1m",
                "origin_status": "real_public_web",
                "split": "train",
                "score": round(sc, 2),
                "reason_code": code,
                "judgment": "ai_like" if sc >= 70.0 else ("human_like" if sc <= 30.0 else "uncertain"),
                "score_source": src,
                "uncertainty": round(unc, 2),
                "dsh": False,
                "partition": part,
                "era_bucket": era,
                "comment_level": lvl,
                "lang_tag": lt,
                "hard_human_neg": hard_neg,
                "train_weight": w_real,
                "prag_vec": prag_v,
                "semantic_intent": q_intent,
                "has_score": has_sc,
            }
            gid = assign_group_id(rec, evasive_modes=evasive_modes)
            if era == "pre_llm" and gid == "structured_human_hard":
                rec["score"] = 2.0
                rec["reason_code"] = "H"
                rec["judgment"] = "human_like"
                rec["train_weight"] = 2.0
            rec["group_id"] = gid
            _emit(rec)
            source_counts["v04_real_new"] += 1
            split_counts["train"] += 1
            group_counts[gid] += 1
            teacher_src_counts[src] += 1

        if v04_real_jsonl is not None and v04_real_jsonl.is_file():
            with _open_read(v04_real_jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rr = json.loads(line)
                    if rr.get("source_cohort") == "legacy_200k":
                        dedup_skipped["v04_real_legacy200k"] += 1
                        continue
                    _handle_real_row(
                        rr["sample_id"],
                        str(rr.get("video_id") or ""),
                        str(rr.get("partition") or "未分类"),
                        str(rr.get("era_bucket") or "mid_llm"),
                        str(rr.get("comment_level") or "top"),
                        str(rr.get("text") or ""),
                    )
        elif v04_real_db is not None and v04_real_db.is_file():
            with closing(sqlite3.connect(f"file:{v04_real_db.as_posix()}?mode=ro", uri=True, timeout=30)) as con:
                con.execute("PRAGMA busy_timeout=30000")
                q = "SELECT sample_id, video_id, partition, era_bucket, comment_level, json_extract(metadata_json, '$.text') FROM comments"
                for sid, vid, part, era, lvl, txt in con.execute(q):
                    if txt:
                        _handle_real_row(sid, vid, part, era, lvl, txt)

        if v04_multilingual_corpus is not None and v04_multilingual_corpus.is_file():
            with _open_read(v04_multilingual_corpus) as mf:
                for line in mf:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("split", "train") != "train":
                        raise ValueError(f"防泄漏校验失败: multilingual_corpus 发现非 train 样本 {r.get('sample_id')}")
                    txt = str(r.get("text") or "")
                    if not txt:
                        continue
                    bh = _norm_hash(txt)
                    pt = prepare_text(txt)
                    if bh in forbidden_hashes or pt in forbidden_norm_texts:
                        leak_blocked["v04_multilingual"] += 1
                        continue
                    if bh in seen_train_hashes:
                        dedup_skipped["v04_multilingual"] += 1
                        continue
                    seen_train_hashes.add(bh)
                    prag = r.get("prag_vec") or r.get("pragmatic_tags")
                    if not (isinstance(prag, list) and len(prag) == len(PRAG_TAGS)):
                        prag = [-1.0] * len(PRAG_TAGS)
                    sc = float(r.get("score", 50.0))
                    rec = {
                        "sample_id": r["sample_id"],
                        "text": txt,
                        "cohort": str(r.get("cohort") or "inkling_wk"),
                        "origin_status": str(r.get("origin_status") or ("confirmed_ai" if sc >= 70.0 else "real_public_web")),
                        "split": "train",
                        "score": round(sc, 2),
                        "reason_code": str(r.get("reason_code") or ("E" if sc >= 70.0 else "H")),
                        "judgment": str(r.get("judgment") or ("ai_like" if sc >= 70.0 else "human_like")),
                        "score_source": str(r.get("score_source") or "v04_inkling_wk_counterfactual"),
                        "uncertainty": float(r.get("uncertainty", 4.0)),
                        "dsh": False,
                        "partition": str(r.get("partition") or r.get("world_knowledge_domain") or "知识"),
                        "era_bucket": str(r.get("era_bucket") or "current_2026"),
                        "comment_level": str(r.get("comment_level") or "top"),
                        "lang_tag": str(r.get("lang_tag") or detect_lang_tag(txt)),
                        "train_weight": float(r.get("train_weight", 1.25)),
                        "prag_vec": [float(x) for x in prag],
                        "semantic_intent": str(r.get("semantic_intent") or ""),
                        "has_score": True,
                    }
                    gid = assign_group_id(rec, evasive_modes=evasive_modes)
                    rec["group_id"] = gid
                    _emit(rec)
                    source_counts["v04_multilingual_wk"] += 1
                    split_counts["train"] += 1
                    group_counts[gid] += 1
    finally:
        if writer is not None:
            writer.close()

    if export_out is not None and staging is not None:
        staging.replace(export_out)

    return {
        "status": "PASS",
        "zero_heldout_leakage_checked": True,
        "leak_candidates_blocked": dict(leak_blocked),
        "dedup_skipped": dict(dedup_skipped),
        "source_counts": dict(source_counts),
        "total_merged_rows": int(sum(source_counts.values())),
        "split_counts": dict(split_counts),
        "literary_pairs_clean": len(pairs),
        "train_group_counts": {g: int(group_counts.get(g, 0)) for g in V04_GROUPS},
        "v04_real_teacher_sources": dict(teacher_src_counts),
        "hard_human_negatives_flagged": hard_human_neg_count,
        "export_out": str(export_out) if export_out is not None else None,
        "export_bytes": int(export_out.stat().st_size) if (export_out is not None and export_out.is_file()) else None,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-data", type=Path, default=REPO_ROOT / "relabel-hydrogen" / "bili-hydrogen-rescored-350387.jsonl.gz")
    parser.add_argument("--qwen-data", type=Path, default=REPO_ROOT / "stage2-qwen" / "stage2-fused-train.jsonl.gz")
    parser.add_argument("--qwen-distill", type=Path, default=REPO_ROOT / "stage2-qwen" / "qwen38_27b_distill.jsonl")
    parser.add_argument("--pairs-data", type=Path, default=REPO_ROOT / "stage2-qwen" / "stage2-literary-pairs.jsonl")
    parser.add_argument("--v04-synth", type=Path, default=REPO_ROOT / "v04-corpus" / "synthetic-250k" / "bili-v04-synthetic-250k.jsonl.gz")
    parser.add_argument("--v04-real-jsonl", type=Path, default=None)
    parser.add_argument("--v04-real-db", type=Path, default=REPO_ROOT / "v04-corpus" / "bili-live-1m.sqlite3")
    parser.add_argument("--legacy-db", type=Path, default=DEFAULT_LEGACY_DB)
    parser.add_argument("--teacher-scores", type=Path, default=None)
    parser.add_argument("--v04-qwen-distill", type=Path, default=None)
    parser.add_argument("--v04-inkling-distill", type=Path, default=None)
    parser.add_argument("--v04-joint-distill", type=Path, default=None)
    parser.add_argument("--v04-multilingual-corpus", type=Path, default=None)
    parser.add_argument("--export-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = merge_v04_corpus(
        stage1_data=args.stage1_data,
        qwen_data=args.qwen_data,
        qwen_distill=args.qwen_distill,
        pairs_data=args.pairs_data,
        v04_synth=args.v04_synth,
        v04_real_jsonl=args.v04_real_jsonl,
        v04_real_db=args.v04_real_db,
        legacy_db=args.legacy_db,
        teacher_scores_path=args.teacher_scores,
        v04_qwen_distill=args.v04_qwen_distill,
        v04_inkling_distill=args.v04_inkling_distill,
        v04_joint_distill=args.v04_joint_distill,
        v04_multilingual_corpus=args.v04_multilingual_corpus,
        export_out=None if args.dry_run else args.export_out,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
