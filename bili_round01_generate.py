"""Generate verified DeepSeek positives using the installed ZCode text RPC."""

import argparse
import hashlib
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bili_round01_data import OUT, body_key


CONTACT = re.compile(r"[@＠]|https?://|\bwww\.|\b1[3-9][0-9]{9}\b|身份证|加我微信|加QQ群", re.I)
FORMATS = {
    "reply": "写给原楼的一条楼中楼回复，接住具体问题或上文，通常 8–48 字，少数可到 85 字。",
    "reaction": "写一条对视频细节的主评论，可以表达保留、好奇、吐槽或实际疑问，通常 8–55 字。",
    "recap": "写一条主评论，挑视频的一两个事实做简短复盘，通常 35–120 字，少数可到 175 字。",
    "memory": "以虚构的助手视角，记下一个细节或给出下次再看时的提醒，通常 14–85 字。",
}
SCENES = (
    ("模型测评", "测{}组输入；同一轮只换提示顺序；一个模型快{}秒；长输入保留得更完整", "平均值是不是被那一轮拖偏了"),
    ("模型蒸馏", "保留{}类长回答；只比较同一测试集；换成{}位量化后有退化", "完整推理和只学答案差别大吗"),
    ("代码教程", "先读{}处调用；错误日志少过{}行；一项回归测的是空输入", "为什么先改入口会多一条分支"),
    ("桌面工具", "切换了{}种搜索方向；索引占{}MB；旧缓存会延迟刷新", "历史记录打开后会更慢吗"),
    ("插件演示", "先读{}段字幕；抽取{}类主题；切到回复模式才处理楼中楼", "它会不会把整期视频复述一遍"),
    ("浏览器扩展", "对照{}种拦截阈值；抽{}条长评复核；只把高分折叠", "正常影评会不会跟着被折叠"),
    ("安卓续航", "后台挂了{}小时；刷新率从{}档来回切；最省电的一组信号最好", "通勤那段是不是信号断了"),
    ("显卡实测", "比较{}款风扇曲线；温差{}度；帧率低点出现在进图那下", "为什么不把功耗也拉到一条线"),
    ("游戏速通", "重试{}次才过；最后一跳多等{}拍；旧路线绕一圈拿护符", "第二段为什么要故意停一下"),
    ("独立游戏试玩", "收了{}个碎片；第{}张地图可走暗门；配乐在进门前变调", "错过碎片还可以回头补吗"),
    ("野外观察", "沿河停了{}次；第{}次换了长焦；阴影里能见到两只不同的鸟", "后面叫的那只是同一种吗"),
    ("骑行记录", "路线有{}段缓坡；水站间隔{}公里；第三段遇到侧风", "不用过桥也能绕过去吗"),
    ("电影镜头分析", "用了{}次回切；原声安静了{}秒；中间那次倒影没对焦", "你们怎么理解灯突然熄掉那下"),
    ("动画分镜", "同一动作画了{}张关键帧；第三段减了{}格；线稿没连笔", "为什么背景也跟着错开一拍"),
    ("录音对比", "做了{}个盲听片段；隔开{}厘米录人声；第二段有风噪", "耳机听不出差别是正常的吗"),
    ("键盘测评", "试了{}种轴；静音垫厚{}毫米；同一段谱用两种力度打", "声音大的是空格键还是大键位"),
    ("烘焙", "同一锅分{}份；第二次多烤{}分钟；冷却后边上才变脆", "厚一点要不要再延长时间"),
    ("厨房实测", "做了{}个对照；焖{}分钟再开盖；最后才放酱油的保水好", "我家锅底薄能照那个火力吗"),
    ("机器人制作", "测试了{}个弯道；校准后误差少了{}毫米；急转时轮胎滑动", "会不会问题是传感器装偏了"),
    ("科普实验", "抽了{}组材料；第{}次改用同容量杯；对照只有光照不同", "样本小的时候是不是看误差更合适"),
    ("音乐制作", "试了{}个录音位置；有{}段只保留环境声；重低音在近场更明显", "为什么最远那个反而最清楚"),
    ("省钱改造", "只换{}件旧件；一共花了{}元；问题主要出在接缝晃动", "原来的螺丝和孔位还能接着用吗"),
    ("摄影教程", "做了{}组曝光包围；上调了{}档；逆光时测光点在树干上", "我怎么感觉偏暗那张反而更耐看"),
    ("轨道交通", "录了{}个站台；间隔大约{}分钟；末段换乘只走一层楼", "周末跟工作日一样的走法吗"),
)
SECOND_NUMBER_MAX = (14, 8, 25, 96, 8, 40, 4, 18, 5, 11, 10, 30,
                     30, 7, 60, 5, 12, 13, 30, 10, 8, 240, 3, 35)
LINKED_NUMBERS = {9, 10, 19}


def batch_prompt(batch, rows_per_batch):
    rng = random.Random(918013 + batch)
    styles = ("reply", "reaction", "recap", "memory", "reply", "recap")
    cards = []
    for number in range(1, rows_per_batch + 1):
        index = rng.randrange(len(SCENES))
        title, facts, parent = SCENES[index]
        mode = styles[(number + batch) % len(styles)]
        first = rng.randint(2, 12)
        second = rng.randint(1, first if index in LINKED_NUMBERS else SECOND_NUMBER_MAX[index])
        cards.append((number, index, mode,
                      f"{number}. {title} / {mode} / 视频片段：{facts.format(first, second)}。"
                      + (f" 原楼：{parent}？" if mode == "reply" else "")))
    rules = "\n".join(f"{key}：{value}" for key, value in FORMATS.items())
    prompt = (
        "这是离线的文本分类研究。以下视频、片段和原楼都是虚构素材。每个编号写一条独立的简体中文视频评论。"
        "参考字幕摘要插件的口吻：挑读到的具体细节；参考自动回复插件的口吻：优先接续上下文和语气。"
        "不要提到生成任务、身份设定、插件配置或模型名字，素材确实谈到插件或模型时可正常谈论。"
        "避免全用一种句式。长短、断句、标点、闲聊和事实密度混合。保留少量常见 B 站表情，无需每条都用。"
        "避免重复其他编号文字。使用给定内容即可，不访问网站，不发帖，不使用工具。"
        f"\n{rules}\n一行一个 JSON 数组 [编号,\"评论正文\"]，每个编号只出现一次。"
        "除此之外不要输出标题、代码围栏、解释或空行。\n" + "\n".join(card[3] for card in cards)
    )
    return prompt, {number: (scene, mode) for number, scene, mode, _ in cards}


def generate(selection, prompt, batch, output_tokens, *, timeout_seconds=105, runtime_root=None, runtime_slot=None):
    resource = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "ZCode" / "resources"
    providers = Path(os.environ["USERPROFILE"]) / ".zcode" / "v2" / "provider_config.json"
    work = (runtime_root or OUT / "zcode-runtime") / "prompt-only"
    work.mkdir(parents=True, exist_ok=True)
    data = work.parent / f"session-{runtime_slot if runtime_slot is not None else batch}"
    env = {key: os.environ[key] for key in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "APPDATA", "LOCALAPPDATA",
        "USERPROFILE", "PROGRAMFILES", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    ) if key in os.environ}
    env.update(ZCODE_DATA_BASE_DIR=str(data), ZCODE_PERSONAL_PROVIDER_CONFIG_FILE=str(providers),
               ZCODE_BUILTIN_PROVIDER_CONFIG_FILE=str(resource / "config" / "provider" / "zcode-builtin.json"),
               ZCODE_MODEL_TELEMETRY_ENABLED="0", ZCODE_MODEL_RETRY_MAX_RETRIES="0")
    operation = f"meguviz-train-{batch}"
    params = dict(workspace={"workspacePath": str(work), "workspaceKey": operation},
                  selection=selection, prompt=prompt, tools=[], querySource="corpus_synthesis",
                  maxOutputTokens=output_tokens, operationId=operation)
    command = ["node", str(resource / "glm" / "zcode.cjs"), "app-server", "--no-color"]
    with subprocess.Popen(command, cwd=work, env=env, encoding="utf-8", bufsize=1,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          creationflags=subprocess.CREATE_NO_WINDOW) as proc:
        messages = queue.Queue()

        def read_messages():
            for line in proc.stdout:
                try:
                    messages.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
            messages.put(None)

        def send(message):
            print(json.dumps(message, ensure_ascii=False), file=proc.stdin, flush=True)

        threading.Thread(target=read_messages, daemon=True).start()
        expires = time.monotonic() + timeout_seconds
        try:
            send({"id": "generate", "method": "workspace/generateText", "params": params})
            while time.monotonic() < expires:
                response = messages.get(timeout=max(expires - time.monotonic(), 0.1))
                if response is None:
                    raise RuntimeError("ZCode closed the text request without a result.")
                if response.get("id") == "generate":
                    if "error" in response:
                        error = response["error"]
                        message = str(error.get("message", ""))
                        safe = re.sub(r"https?://\S+|(?i:Bearer)\s+\S+|\bsk-[\w-]+", "[redacted]", message)
                        raise RuntimeError(f"ZCode RPC {error.get('code')}: {safe[:200]}")
                    result = response["result"]
                    returned = result.get("selection", {})
                    if any(returned.get(key) != selection[key] for key in ("providerId", "modelId")):
                        raise RuntimeError("ZCode returned a different model selection.")
                    if result.get("toolCalls"):
                        raise RuntimeError("The text-only request unexpectedly returned tool calls.")
                    return result
                if "id" in response and "method" in response:
                    send({"id": response["id"], "error": {"code": -32601,
                          "message": "This text-only request has no host or interactive tools."}})
        except queue.Empty as error:
            try:
                send({"id": "cancel", "method": "workspace/cancelGenerateText",
                      "params": {"operationId": operation}})
            except OSError:
                pass
            raise RuntimeError("ZCode text request timed out.") from error
        finally:
            proc.terminate()
    raise RuntimeError("ZCode text request timed out.")


def parse_rows(result, cards, batch, prompt_sha):
    seen_numbers = set()
    model = result["selection"]["modelId"]
    for line in result["text"].splitlines():
        try:
            pair = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        number, text = pair
        if type(number) is not int or number not in cards or number in seen_numbers or not isinstance(text, str):
            continue
        text = " ".join(text.split())
        if not 6 <= len(text) <= 220 or not re.search(r"[\u3400-\u9fff]", text) or CONTACT.search(text):
            continue
        seen_numbers.add(number)
        scene, mode = cards[number]
        key = body_key(text)
        uid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        yield {"sample_id": f"DS{uid}", "text": text, "label": "ai", "split": "train",
               "source": "zcode_deepseek", "model_id": model, "provider_id": result["selection"]["providerId"],
               "scenario": SCENES[scene][0], "format": mode, "batch": batch, "index": number,
               "prompt_sha256": prompt_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--limit-batches", type=int, default=72)
    parser.add_argument("--first-batch", type=int, default=1)
    parser.add_argument("--rows-per-batch", type=int, default=56)
    parser.add_argument("--output-tokens", type=int, default=7200)
    parser.add_argument("--provider", default="new-provider-8")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-ultrafast")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    selection = {"providerId": args.provider, "modelId": args.model}
    if args.reasoning:
        selection["options"] = {"reasoningLevel": args.reasoning}
    if args.probe:
        prompt, cards = batch_prompt(0, 4)
        result = generate(selection, prompt, 0, 4500)
        parsed = list(parse_rows(result, cards, 0, hashlib.sha256(prompt.encode()).hexdigest()))
        print(json.dumps({"accepted": len(parsed), "model": result["selection"]["modelId"],
                          "usage": result.get("usage", {}), "finish": result.get("finishReason")},
                         ensure_ascii=False))
        if not parsed:
            raise RuntimeError("The DeepSeek probe did not return usable comments.")
        return

    data_path = OUT / "bili-round01-deepseek-train.jsonl"
    receipt_path = OUT / "bili-round01-deepseek-receipts.jsonl"
    prior = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()] if data_path.exists() else []
    receipts = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()] if receipt_path.exists() else []
    seen = {row["sample_id"] for row in prior}
    done = {row["batch"] for row in receipts if row.get("accepted", 0)}
    if any(row["model_id"] != args.model or row["provider_id"] != args.provider for row in prior):
        raise RuntimeError("The existing synthetic file uses another model route.")

    def run_batch(batch):
        prompt, cards = batch_prompt(batch, args.rows_per_batch)
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        try:
            generated = generate(selection, prompt, batch, args.output_tokens)
        except (OSError, RuntimeError) as error:
            return [], {"batch": batch, "error": str(error)}
        rows = list(parse_rows(generated, cards, batch, prompt_sha))
        return rows, dict(batch=batch, requested=len(cards), usable=len(rows), model_id=args.model,
                          provider_id=args.provider, usage=generated.get("usage"),
                          finish=generated.get("finishReason"), prompt_sha256=prompt_sha)

    failures = 0
    batches = [batch for batch in range(args.first_batch, args.limit_batches + 1) if batch not in done]
    with (ThreadPoolExecutor(max_workers=4) as pool, data_path.open("a", encoding="utf-8") as data,
          receipt_path.open("a", encoding="utf-8") as log):
        for offset in range(0, len(batches), 4):
            if len(seen) >= args.samples or failures >= 8:
                break
            for rows, receipt in pool.map(run_batch, batches[offset:offset + 4]):
                accepted = 0
                for row in rows:
                    if row["sample_id"] in seen:
                        continue
                    seen.add(row["sample_id"])
                    prior.append(row)
                    data.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted += 1
                data.flush()
                receipt.update(accepted=accepted, total=len(seen))
                line = json.dumps(receipt, ensure_ascii=False)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
                failures = 0 if accepted else failures + 1
    print(json.dumps({"accepted_total": len(seen), "formats": dict(Counter(row["format"] for row in prior)),
                      "model_id": args.model}, ensure_ascii=False), flush=True)
    if len(seen) < args.samples:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
