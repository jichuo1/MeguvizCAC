"""Step 4 (Optional): Build Sanitized Portable ZIP Packages for External Gemini Tasks.

STRICT SECURITY INVARIANT:
  - NEVER includes `canary_vault.json` (hidden 2% val_gold ground truth stays 100% local on the owner's machine!).
  - Produces two ready-to-drop external packages:
    1) `meguviz-v08-gemini-batch-api-<tag>.zip` : Full pre-packed Gemini Batch API JSONL shards (50% off)
       + standalone runner scripts + README (for external Gemini API / Batch / Colab / Vertex jobs).
    2) `meguviz-v08-gemini-agent-pack-<tag>.zip`: Ready-to-drop `prompts/*.txt` + `replies/` + copy-paste
       Agent instruction (`README_外部Gemini一键指令.md`) for external Gemini Agent / CLI / Web jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AGENT_README_TEMPLATE = """# MeguvizCAC v0.8.0 外部 Gemini 一键标注工程包（批次 {stamp}）

本包专为 **外部 Gemini 任务（Gemini Agent / Gemini CLI / AI Studio / Batch API / 网页端）** 打造，**零 Antigravity 额度消耗**，严格零 PII、严格防评估泄漏，并与 MeguvizCAC Hydrogen Rubric v2.1 尺度 100% 同源闭环。

## 本批任务清单

| 队列 | 作用 | 本包分片数 | 样本条数 | 建议模型 |
| :--- | :--- | ---: | ---: | :--- |
{queue_rows}

> 注：本包内含 **2% 匿名隐藏校准探针**，回收后将自动执行 `AUROC / Spearman / Bias` 质检门禁并在入库前 **100% 剥离**，严禁编写正则/关键词脚本批量蒙混过关！

---

## 用法一（最推荐）：交给可读写文件的外部 Gemini Agent / Gemini CLI

将本目录解压后，直接把下面这段指令原样发给外部 Gemini（推荐 `gemini-2.5-flash` 跑全量 Tier-1，或 `gemini-2.5-pro` 跑难例）：

```text
你是 MeguvizCAC v0.8.0 独立法医盲评裁判。请依次处理当前目录 prompts/ 下所有子队列目录里、且 replies/ 下尚无对应 <batch_id>.json 的分片文件。
对每个分片：
1. 完整阅读 prompts/<队列>/<batch_id>.txt，严格遵守其中的 Rubric v3.0 尺度闭环（0-30必选H、31-69必选P/S/Q、70-100必选A/R/E/T；硬核结构化/评分表/代码/书目真人保护为H且 is_trap_human=true）。
2. 必须逐条独立阅读正文判断，绝对禁止编写或运行任何 Python/正则/关键词脚本批量打分，禁止联网。
3. 将且仅将严格合法的纯 JSON 对象 `{{"items": [...]}}` 写入 replies/<队列>/<batch_id>.json（每条严格照抄 idx 和 3 位校验码 tag，不得遗漏或跳号）。
每完成 10 个分片汇报一次进度，全部完成后汇报总写入分片数与总条数。
```

---

## 用法二：外部 Gemini Batch API（官方 5 折 · 适合 10 万 – 50 万条大规模一键跑）

若你使用 Gemini API Key（无需消耗对话界面）：
```powershell
pip install -U google-genai
$env:GEMINI_API_KEY="你的_GEMINI_API_KEY"
# 直接提交本包已预打包好的 Batch JSONL（或运行根目录 scripts/02_run_gemini_factory.py）：
python scripts/02_run_gemini_factory.py --work-dir . --mode batch --tier tier1 --submit-batch --poll-batch
```
或直接把 `batch_api_tier1/<队列>/requests_part_*.jsonl` 上传至 Google AI Studio / Vertex AI Batch 控制台执行。

---

## 用法三：做完后回收与一键质检合库

把外部跑完的 `replies/` 目录（或 Batch API 下载的结果 JSONL）放回本机工作目录 `{work_dir_hint}`，告诉我一声（或自己运行）：
```powershell
python tools/gemini_v08_datafactory/02_run_gemini_factory.py --work-dir {work_dir_hint} --mode pack --tier tier1 --import-pack
python tools/gemini_v08_datafactory/03_qc_and_merge.py --work-dir {work_dir_hint}
```
系统会自动校验全部 3 位校验码与隐藏金丝雀 `AUROC / Bias`，通过后自动筛选 ~15% 灰区/分歧难例生成 **Tier-2 (`gemini-2.5-pro`) 终审包**，并产出即插即用的 `v08-gemini-gold-labels.jsonl.gz` 与 `v08-gemini-counterfactual-pairs.jsonl`！
"""


def _assert_not_vault(p: Path) -> None:
    if "canary_vault" in p.name.lower():
        raise AssertionError(f"SECURITY VIOLATION: Refusing to pack local secret canary vault: {p}")


def build_agent_pack_zip(
    work_dir: Path,
    out_zip: Path,
    stamp: str,
    max_batches_per_queue: dict[str, int] | None = None,
) -> dict:
    """Pack sanitized offline_pack_tier1 prompts + empty replies/ dirs + README into a portable ZIP."""
    pack_root = work_dir / "offline_pack_tier1"
    queues_dir = work_dir / "queues"
    if not pack_root.is_dir():
        raise FileNotFoundError(f"Missing {pack_root}; run 02_run_gemini_factory.py --mode pack first.")

    q_meta = {
        "A_noise_surgery": ("噪声手术（灰区/硬组清洗）", "gemini-2.5-flash -> pro"),
        "B_wild_hard_mining": ("2026现代野生难例（破年代捷径）", "gemini-2.5-flash -> pro"),
        "C_counterfactual_spans": ("反事实最小对 & 三明治混写Span", "gemini-2.5-pro 推荐"),
        "D_thread_context": ("楼中楼与视频标题上下文裁决", "gemini-2.5-flash -> pro"),
    }

    root_name = out_zip.stem
    table_lines: list[str] = []
    stats: dict[str, dict] = {}
    total_files = 0
    total_rows = 0

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for qname in sorted(q_meta):
            pdir = pack_root / qname / "prompts"
            if not pdir.is_dir():
                continue
            pfiles = sorted(pdir.glob("*.txt"))
            if max_batches_per_queue and qname in max_batches_per_queue:
                pfiles = pfiles[: max_batches_per_queue[qname]]
            if not pfiles:
                continue

            # Compute exact row count from batch manifest
            row_cnt_map: dict[str, int] = {}
            bf = queues_dir / f"{qname}.batches.jsonl"
            if bf.is_file():
                with bf.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            row_cnt_map[obj["batch_id"]] = int(obj["n"])

            q_rows = 0
            zf.writestr(f"{root_name}/replies/{qname}/.keep", "")
            for pf in pfiles:
                _assert_not_vault(pf)
                zf.write(pf, f"{root_name}/prompts/{qname}/{pf.name}")
                q_rows += row_cnt_map.get(pf.stem, 25)
                total_files += 1
            total_rows += q_rows
            desc, mdl = q_meta[qname]
            table_lines.append(f"| `{qname}` | {desc} | {len(pfiles)} | {q_rows:,} | {mdl} |")
            stats[qname] = {"batches": len(pfiles), "rows": q_rows}

        readme = AGENT_README_TEMPLATE.format(
            stamp=stamp,
            queue_rows="\n".join(table_lines),
            work_dir_hint=str(work_dir),
        )
        zf.writestr(f"{root_name}/README_外部Gemini一键指令.md", readme)

    return {
        "zip": str(out_zip),
        "zip_bytes": out_zip.stat().st_size,
        "total_prompt_files": total_files,
        "total_rows": total_rows,
        "queues": stats,
        "canary_vault_excluded": True,
    }


def build_batch_api_zip(work_dir: Path, out_zip: Path, stamp: str) -> dict:
    """Pack full Batch API JSONL shards + standalone factory scripts (ZERO canary_vault.json!)."""
    batch_root = work_dir / "batch_api_tier1"
    factory_src = REPO_ROOT / "tools" / "gemini_v08_datafactory"
    root_name = out_zip.stem
    files_added = 0

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        # Include runner scripts (standalone executable)
        for fn in ("rubric_v08.py", "02_run_gemini_factory.py", "run_factory.ps1", "run_factory.sh", "README.md"):
            fp = factory_src / fn
            if fp.is_file():
                zf.write(fp, f"{root_name}/scripts/{fn}")
                files_added += 1
        # Include sanitized export manifest (strip canary_vault_path string just for tidiness)
        em_path = work_dir / "export_manifest.json"
        if em_path.is_file():
            em = json.loads(em_path.read_text(encoding="utf-8"))
            em.pop("canary_vault_path", None)
            zf.writestr(f"{root_name}/export_manifest.json", json.dumps(em, ensure_ascii=False, indent=2) + "\n")
            files_added += 1
        # Include all pre-packed Batch API JSONL shards
        if batch_root.is_dir():
            for jf in sorted(batch_root.rglob("requests_part_*.jsonl")):
                _assert_not_vault(jf)
                rel = jf.relative_to(work_dir).as_posix()
                zf.write(jf, f"{root_name}/{rel}")
                files_added += 1
        # Include queues/*.batches.jsonl (no ground truth inside — only blind texts & 3-char tags!)
        q_dir = work_dir / "queues"
        if q_dir.is_dir():
            for bf in sorted(q_dir.glob("*.batches.jsonl")):
                _assert_not_vault(bf)
                zf.write(bf, f"{root_name}/queues/{bf.name}")
                files_added += 1

    return {
        "zip": str(out_zip),
        "zip_bytes": out_zip.stat().st_size,
        "files_added": files_added,
        "canary_vault_excluded": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--stamp", type=str, default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--agent-wave-batches", type=int, default=50, help="Max prompt files per queue in the Agent Wave-1 ZIP (0 = all)")
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    out_summary: dict = {"stamp": args.stamp, "work_dir": str(work_dir)}

    if (work_dir / "offline_pack_tier1").is_dir():
        caps = None
        if args.agent_wave_batches > 0:
            caps = {
                "A_noise_surgery": args.agent_wave_batches,
                "B_wild_hard_mining": args.agent_wave_batches,
                "C_counterfactual_spans": max(15, args.agent_wave_batches // 2),
                "D_thread_context": max(20, args.agent_wave_batches // 2),
            }
        agent_zip = work_dir / f"meguviz-v08-gemini-agent-wave1-{args.stamp}.zip"
        out_summary["agent_pack_zip"] = build_agent_pack_zip(work_dir, agent_zip, args.stamp, max_batches_per_queue=caps)

    if (work_dir / "batch_api_tier1").is_dir():
        batch_zip = work_dir / f"meguviz-v08-gemini-batch-api-100k-{args.stamp}.zip"
        out_summary["batch_api_zip"] = build_batch_api_zip(work_dir, batch_zip, args.stamp)

    print(json.dumps(out_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
