"""Suggest AI authorship from corroborating features in the live comment corpus."""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from itertools import combinations
from pathlib import Path

from bili_live_corpus import ROOT, connect, ingest_suggestions


QUOTED = re.compile(r'「[^」]{1,800}」|『[^』]{1,800}』|“[^”]{1,800}”|‘[^’]{1,800}’|《[^》]{1,800}》|"[^"\n]{1,800}"')
QUOTE_RECAP = re.compile(r"这(?:个|一段|段|条|张|一幕)[^，。；:：]{0,11}(?:对比|对照|剪辑点|镜头|最高赞|总结)"
                         r"|(?:把|将)[^，。]{0,27}(?:浓缩|说完|概括|压成)"
                         r"|\d+\s*(?:个)?赞|最高赞")
MEMORY = re.compile(r"我(?:就|先|已经|暂且)?(?:记下(?:了|来)?|记着|记住了|记在(?:笔记|备忘录|小本本))")
RECAP = re.compile(r"(?:这|本)(?:期|次|个视频|段)(?:的)?(?:结果|亮点|结论|要点|重点|最[^，。]{0,9}的是)"
                   r"|(?:复盘|要点|总结|结论|结果|划重点|片尾预告)\s*[:：]"
                   r"|先[^，。]{2,28}再[^，。]{2,28}")
OUTCOME = re.compile(r"通过|通关|拿[下到]|收掉|完成|输在|没过|改了|只(?:改|动)|跑完|打出")
FOLLOW_UP = re.compile(r"(?:下次|回头|之后|等(?:会|下一期))[^。！？]{0,18}(?:整理|复盘|补(?:上|充|笔记)?|翻出来|再试|写到)")
ADDRESSING = re.compile(r"收到[，,！!\s]|我(?:把|给你)(?:这|先)|(?:这段|这个点|这条)我(?:先)?记")
AGENT_ADDRESS = re.compile(r"^(?:回复\s*[:：]\s*)?(?:收到(?:了)?|遵命|明白)[，,!\s]*(?:老板|主人|鱼主|饲主)")
AGENT_PLAN = re.compile(r"(?:我|我们)(?:都|会|就|下次|先)?(?:照(?:这个|这套|你的要求)来|记下|整理|补上|归档|存入)")
REPLY_RECAP = re.compile(r"这(?:条|段|期)[^，。]{0,14}(?:一招|亮点|方法|重点)")
AGENT_SELF_REFERENCE = re.compile(r"我(?:这|那|也是一|就是一)条(?:大肥)?[鱼魚]|我的(?:主人|鱼主|饲主)")
ONE_OF_FISH = re.compile(r"我(?:是|就是)[^，。!！]{1,14}的一条")
WORKFLOW_DETAIL = re.compile(r"压缩前|另开(?:一轮|一个会话)|token|人设|工作目录|转写|存档|归档|会话|提示词|子代理|自动评论|自动回复")
ASSISTANT = re.compile(r"^(?:作为|身为)(?:一[个名位])?(?:ai(?:助手)?|语言模型|人工智能助手)[\s,，。:：]"
                       r"|^(?:我(?:是|作为))(?:一[个名位])?(?:ai助手|语言模型|人工智能助手)[\s,，。:：]")
ASSISTANT_CAPABILITY = re.compile(r"我?(?:无法|不能|没有(?:权限|能力|个人)|不会|不能直接|只能|可以为你|不具备)")
FACTS = re.compile(
    r"(?P<time>(?<![0-9])\d{1,2}:[0-5]\d(?::[0-5]\d)?(?![0-9]))"
    r"|(?P<id>(?<![a-z0-9])[a-z]{1,5}[-_]?\d{3,5}(?![a-z0-9]))"
    r"|(?P<model>(?<![a-z0-9])(?:gemini|claude|deepseek|chatgpt|gpt|qwen|astra|fable|carbon|豆包|千问|通义|混元|飞豹|群星)(?![a-z0-9]))"
    r"|(?P<quantity>(?:\d+(?:\.\d+)?|[零〇一二三四五六七八九十百两]{1,5})\s*"
    r"(?:个文件|行代码|行字幕|轮(?:修改|修复)?|次尝试|次通过|分钟|秒钟|个模型|%))"
)
DIGITS = dict(zip("零一二三四五六七八九", range(10)))


def normalize_quantity(number):
    number = number.replace("两", "二").replace("〇", "零")
    if number.isascii():
        return number
    if "十" in number and "百" not in number and number.count("十") == 1:
        before, after = number.split("十")
        return str(10 * DIGITS.get(before, 1) + DIGITS.get(after, 0))
    return str(DIGITS.get(number, number))


def ordered_facts(body):
    found = {}
    for match in FACTS.finditer(body):
        category = match.lastgroup
        fact = re.sub(r"\s+", "", match[0])
        if category == "quantity":
            number, unit = re.fullmatch(r"([0-9.零〇一二三四五六七八九十百两]+)(.+)", fact).groups()
            fact = normalize_quantity(number) + re.sub(r"轮(?:修改|修复)", "轮", unit)
        found.setdefault((category, fact), None)
        if len(found) >= 10:
            break
    return list(found)


def style_suggestions(rows):
    """Group only distinct comment threads in the same video."""
    signatures = defaultdict(set)
    by_sample = {}
    hints = {}

    for row in rows:
        uid = row["sample_id"]
        normalized = unicodedata.normalize("NFKC", row["text"]).casefold()
        body = QUOTED.sub(" ", normalized).strip()
        if len(body) < 20:
            continue
        facts = ordered_facts(body)
        recap = (RECAP.search(body) is not None or
                 (len(facts) >= 3 and OUTCOME.search(body) is not None and len(re.findall(r"[，,。；;]", body)) >= 2))
        workflow = list(dict.fromkeys(WORKFLOW_DETAIL.findall(body)))
        agent_self = bool(AGENT_SELF_REFERENCE.search(body) or (re.search(r"[鱼魚]", normalized) and ONE_OF_FISH.search(body)))
        if ASSISTANT.search(body) and ASSISTANT_CAPABILITY.search(body):
            hints[uid] = ("assistant_self_description", {"role_and_capability": True})
        elif (row.get("comment_level") == "reply" and AGENT_ADDRESS.search(body) and AGENT_PLAN.search(body)
              and (recap or REPLY_RECAP.search(body)) and QUOTED.search(normalized)):
            hints[uid] = ("assistant_commitment", {
                "address_and_follow_up": True, "recap_with_quoted_reference": True
            })
        elif MEMORY.search(body) and ((recap and any(kind != "model" for kind, _ in facts))
                                      or (FOLLOW_UP.search(body) and ADDRESSING.search(body))
                                      or (agent_self and len(workflow) >= 2)):
            hints[uid] = ("memory_and_recap", {"memory_phrase": True, "recap": recap,
                                                     "follow_up": bool(FOLLOW_UP.search(body)),
                                                     "agent_self_reference": agent_self, "workflow": workflow,
                                                     "facts": [fact for _, fact in facts]})

        video = row["video_id"]
        thread = row.get("parent_id") or uid
        selected = []
        if recap:
            for triple in combinations(facts, 3):
                types = {kind for kind, _ in triple}
                if len(types) < 2 or "quantity" not in types and "time" not in types:
                    continue
                selected.append((video, triple))
        if QUOTE_RECAP.search(body):
            for match in QUOTED.finditer(normalized):
                quote = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", match[0])
                if 8 <= len(quote) <= 54:
                    selected.append((video, (("quote", quote),)))
        if selected:
            by_sample[uid] = selected
            for key in selected:
                signatures[key].add(thread)

    for uid, candidates in by_sample.items():
        if uid in hints:
            continue
        key = max(candidates, key=lambda item: len(signatures[item]))
        frequency = len(signatures[key])
        if frequency >= 3:
            evidence = {"distinct_threads": frequency}
            if key[1][0][0] == "quote":
                evidence["shared_quote"] = key[1][0][1]
            else:
                evidence["shared_ordered_facts"] = [fact for _, fact in key[1]]
            hints[uid] = ("shared_structured_recap", evidence)
    return hints


def apply_style_suggestions(db):
    rows = [json.loads(row[0]) for row in db.execute("""SELECT metadata FROM comments
        WHERE json_extract(metadata,'$.source_kind')='bili_browser_live' AND json_extract(metadata,'$.label')='unknown'""")]
    found = style_suggestions(rows)
    by_rule = Counter()
    suggestions = []
    with db:
        for row in rows:
            uid = row["sample_id"]
            if uid not in found:
                continue
            rule, evidence = found[uid]
            by_rule[rule] += 1
            db.execute("""UPDATE comments SET metadata=json_set(metadata, '$.style_rule', ?, '$.style_evidence', json(?))
                          WHERE sample_id=?""", (rule, json.dumps(evidence, ensure_ascii=False), uid))
            if not row.get("nickname_rule"):
                suggestions.append([uid, "ai_like", "style_rule:" + rule])
    return {"corpus_rows": len(rows), "feature_matches": dict(by_rule), **ingest_suggestions(db, suggestions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "bili-live-corpus.sqlite3")
    args = parser.parse_args()
    with closing(connect(args.db)) as db:
        print(json.dumps(apply_style_suggestions(db), ensure_ascii=False))


if __name__ == "__main__":
    main()
