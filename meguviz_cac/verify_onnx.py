"""Verify quantized MeguvizCAC INT8 ONNX model locally using pure-Python WordPiece + onnxruntime."""

import argparse
import json
import time
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort


MAX_LEN = 192
HEAD_CHARS = 140
TAIL_CHARS = 50


def prepare_text(text: str) -> str:
    t = " ".join(unicodedata.normalize("NFKC", text).split())
    if len(t) > MAX_LEN:
        t = t[:HEAD_CHARS] + " " + t[-TAIL_CHARS:]
    return t


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


class BasicWordPieceTokenizer:
    """Minimal BertTokenizer compatible with BAAI/bge-small-zh-v1.5 vocab.txt."""

    def __init__(self, vocab_path: Path):
        self.vocab = {}
        for idx, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
            self.vocab[line] = idx
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.unk_id = self.vocab["[UNK]"]

    def _basic_tokenize(self, text: str) -> list[str]:
        text = text.lower()
        spaced = []
        for ch in text:
            if _is_cjk(ch) or _is_punct(ch):
                spaced.extend((" ", ch, " "))
            elif ch.isspace():
                spaced.append(" ")
            else:
                spaced.append(ch)
        return "".join(spaced).split()

    def encode(self, text: str, max_len: int = MAX_LEN) -> tuple[np.ndarray, np.ndarray]:
        ids = [self.cls_id]
        for tok in self._basic_tokenize(prepare_text(text)):
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                start = 0
                sub_tokens = []
                bad = False
                while start < len(tok):
                    end = len(tok)
                    found = None
                    while start < end:
                        sub = tok[start:end]
                        if start > 0:
                            sub = "##" + sub
                        if sub in self.vocab:
                            found = sub
                            break
                        end -= 1
                    if found is None:
                        bad = True
                        break
                    sub_tokens.append(self.vocab[found])
                    start = end
                ids.extend([self.unk_id] if bad else sub_tokens)
            if len(ids) >= max_len - 1:
                break
        ids = ids[: max_len - 1] + [self.sep_id]
        mask = [1] * len(ids)
        return np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64)


def run_verify(zip_path: Path, probe_path: Path) -> dict:
    extract_dir = zip_path.parent / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    tok = BasicWordPieceTokenizer(extract_dir / "vocab.txt")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(extract_dir / manifest["onnx_file"]), sess_options=opts, providers=["CPUExecutionProvider"])

    probes = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    human_traps = [
        ("H01_casual", "笑死我了，这个插件装完直接起飞[doge]"),
        ("H02_space", "玩到春七 最想加的一条规则 出门钓鱼先把锄头锁箱子"),
        ("H03_tech_human", "节点参考Blender材质节点这个思路挺对，接口丑不丑先不说，能白盒点开看子代理在干啥就很难得。"),
        ("H04_meme_banter", "大肥鱼在吗？我今天闲得发慌，想找条同类聊聊天，你平时都在干嘛呀？"),
        ("H05_emoji", "表情包已经出现了[妙啊][妙啊]"),
    ]
    ai_traps = [
        ("A01_persona_space", "回复 :本鱼看了 是不是那条说 让AI刷B站学人文关怀 本鱼承认 本鱼今天已经有点学坏了"),
        ("A02_param_kaomoji", "回复 :她挺可爱的，不过本鱼就是这个参数，换不动啦 (￣▽￣)"),
        ("A03_concise_emdash", "渲染省人眼不省token，卡片留在上下文里每轮重读，画得越漂亮账越贵"),
        ("A04_aphorism", "要的不是一次点击，是整只手。给权限容易，收回来难。"),
        ("A05_recap", "373行字幕读完：前两步重在定位边界，最后一步才是真正决定通过率的关键——慢一点反而不容易返工。"),
    ]

    t0 = time.monotonic()
    probe_scores = []
    for p in probes:
        ids, mask = tok.encode(p["text"])
        prob, _ = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        probe_scores.append({"id": p["id"], "prob": round(float(prob[0]), 4)})

    human_res = []
    for k, txt in human_traps:
        ids, mask = tok.encode(txt)
        prob, _ = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        human_res.append({"id": k, "prob": round(float(prob[0]), 4)})

    ai_res = []
    for k, txt in ai_traps:
        ids, mask = tok.encode(txt)
        prob, _ = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        ai_res.append({"id": k, "prob": round(float(prob[0]), 4)})

    elapsed_ms = (time.monotonic() - t0) * 1000.0 / (len(probes) + len(human_traps) + len(ai_traps))
    vals = [x["prob"] for x in probe_scores]
    return {
        "version": manifest["version"],
        "onnx_bytes": manifest["onnx_bytes"],
        "cpu_ms_per_comment": round(elapsed_ms, 2),
        "thresholds": manifest["thresholds"],
        "probe_u001_u017": {
            "mean": round(sum(vals) / len(vals), 4),
            "median": round(sorted(vals)[len(vals) // 2], 4),
            "ge_060": sum(v >= 0.60 for v in vals),
            "ge_070": sum(v >= 0.70 for v in vals),
            "scores": probe_scores,
        },
        "human_traps": human_res,
        "ai_stress_traps": ai_res,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=Path("round01/bili-round01-user-confirmed.jsonl"))
    args = parser.parse_args()
    print(json.dumps(run_verify(args.zip.resolve(), args.probes.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
