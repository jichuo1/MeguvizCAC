# MeguvizCAC v0.8.0 External Gemini One-Click Data Factory（零反重力额度消耗）

本工程包专为 **MeguvizCAC v0.8.0-Zenith（零架构动刀 · 纯指标登顶版）** 设计，允许你在**外部直接调用 Gemini Batch API（5 折）/ 实时 API / AI Studio / Gemini CLI / 网页端**，将高质量金标（Gold-Tier）与反事实最小对规模从当前的 **~2.3 万条** 一键拉升至 **25 万 – 50 万+ 条**，**完全不消耗 Antigravity 对话额度**。

---

## 核心特性与安全铁律

1. **100% 尺度同源对齐**：内置 `Rubric v3.0`（[rubric_v08.py](rubric_v08.py)），逐字继承 `Hydrogen Rubric v2.1` 尺度闭环（`0–30 => H`，`31–69 => P/S/Q`，`70–100 => A/R/E/T`）与 `PRAG_TAGS` 8 维语用标签，并加入 `is_trap_human` 硬核真人保护标记与 `ai_spans` 混合片段精确字符下标。
2. **四大高 ROI 主动学习队列**：
   - **Queue A（噪声手术 `A_noise_surgery`）**：优先清洗 `v04-merged-1m4-r3.jsonl.gz` 中非金标、中段灰区（`26–74`）及 `dsh_hard_slice / structured_human_hard / multilingual / literary` 易错样本。
   - **Queue B（现代野生难例挖掘 `B_wild_hard_mining`）**：从 `bili-live-1m.sqlite3` 的 2025–2026 现代真实 B 站评论中按桶抽取硬核技术长评、可疑 AI 特征、多语种、长评与超短评，彻底消灭 `pre_llm` 年代捷径。
   - **Queue C（反事实最小对 + 三明治混合片段 `C_counterfactual_spans`）**：同主题、同实体、同长度改写四类最强对抗样本（`human_to_evasive_ai`、`human_to_short_ai`、`ai_to_structured_human`、`sandwich_hybrid_span` 带精确字符级 `ai_spans`）。
   - **Queue D（楼中楼上下文裁决 `D_thread_context`）**：自动拼装 `(video_title, parent_text, reply_text)` 三元组，专打楼中楼错位复读（`T` 类）与接梗真人误杀。
3. **两级自动提级裁判（Tier-1 Flash → Tier-2 Pro，省 80%+ 费用）**：
   - **Tier-1（`gemini-2.5-flash`）** 以每请求 **25 条超级拼包（Mega-Pack）** + 3 位防错位校验码（`tag`）全量盲评。
   - **Tier-2（`gemini-2.5-pro`）** 自动筛出 Tier-1 中 `28–72` 灰区、低置信度（`<75`）、潜在硬核真人陷阱（`is_trap_human`）或与旧分数分歧 `>=28` 的 ~15% 难例进行 Pro 级终审裁决。
4. **严格零 PII + 不可静默防泄漏 + 2% 隐藏金丝雀自检**：
   - 全程不读取/不导出 `UID/昵称/IP/等级/粉丝数`。
   - 导出前与合并后**双重校验**，剔除所有 `val / test / user_confirmed` 及全部内置陷阱（精确匹配 + char-trigram Jaccard `>= 0.72`）。
   - 自动在队列 A/B 中盲混 **2% `val_gold` 隐藏金丝雀**（真值仅留本地 `canary_vault.json`，绝不发给 Gemini）；合并前自动核算 `AUROC >= 0.94 / Spearman >= 0.85 / |Bias| <= 8.0`，不达标直接 Fail-Fast 阻断，通过后 **100% 剥离金丝雀**，确保零评估泄漏！

---

## 外部费用预估（按每批 25 条超级拼包 + 15% Tier-2 Pro 复核）

| 规模档位 | 评分队列 (A+B+D) | 反事实队列 (C) | Gemini Batch API（5 折）预估 | 实时并发 API 预估 | Antigravity 额度消耗 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **轻量冲刺（5 万）** | 40,000 | 10,000 | **~$3.8** | ~$7.6 | **0** |
| **黄金标准（15 万）** | 125,000 | 25,000 | **~$11.2** | ~$22.4 | **0** |
| **满血推荐（25 万）** | 220,000 | 30,000 | **~$18.6** | ~$37.2 | **0** |
| **极限拉满（50 万）** | 440,000 | 60,000 | **~$37.1** | ~$74.2 | **0** |

> 提示：若在 `02_run_gemini_factory.py` 加 `--thinking-budget 0`（对 Tier-1 Flash 关闭长思考链），输出 token 还可进一步压缩，费用再降 ~20–30%。

---

## 一键运行指南

### 0. 先做零成本本地自检（无需 API Key，10 秒验通全链路）
```powershell
.\tools\gemini_v08_datafactory\run_factory.ps1 -Mode dry-run -LimitA 100 -LimitB 100 -LimitC 40 -LimitD 60 -WorkDir v08-datafactory-dryrun
```

### 方式一：Gemini Batch API（最省钱 · 官方 5 折 · 推荐 15 万–50 万条）
```powershell
pip install -U google-genai
$env:GEMINI_API_KEY="你的_GEMINI_API_KEY"

# 一键导出 -> 提交 Tier-1 Batch -> 等待完成 -> 自动提级 Tier-2 Pro Batch -> 金丝雀 QC -> 输出清洁训练包
.\tools\gemini_v08_datafactory\run_factory.ps1 -Mode batch -SubmitBatch -PollBatch
```
> 若只想先生成标准 Batch JSONL 文件自己上传到 AI Studio / Vertex AI，不加 `-SubmitBatch` 即可，文件位于 `v08-datafactory-work/batch_api_tier1/<队列>/requests_part_*.jsonl`。

### 方式二：实时并发 API（速度最快 · 纯 Python 标准库零第三方依赖 · 断点续跑）
```powershell
$env:GEMINI_API_KEY="你的_GEMINI_API_KEY"
.\tools\gemini_v08_datafactory\run_factory.ps1 -Mode realtime -LimitA 30000 -LimitB 50000 -LimitC 12000 -LimitD 18000 -Workers 12 -Qps 8
```
中断后重新执行同一条命令会自动按 `batch_id` 跳过已完成分片，无缝断点续跑。

### 方式三：离线 Prompt 包模式（零 API Key · 适合 Gemini 网页 / 外部 CLI / 外部 Agent）
```powershell
# 1. 生成独立提示词文件
python tools/gemini_v08_datafactory/01_export_queue.py --out-dir v08-datafactory-pack --limit-a 5000 --limit-b 5000 --limit-c 1000 --limit-d 1000
python tools/gemini_v08_datafactory/02_run_gemini_factory.py --work-dir v08-datafactory-pack --mode pack --tier tier1

# 2. 将外部 Gemini 的纯 JSON 回复存为 offline_pack_tier1/<队列>/replies/<batch_id>.json 后导入并 QC：
python tools/gemini_v08_datafactory/02_run_gemini_factory.py --work-dir v08-datafactory-pack --mode pack --tier tier1 --import-pack
python tools/gemini_v08_datafactory/03_qc_and_merge.py --work-dir v08-datafactory-pack
```

---

## 最终产物（直接对接 v0.8.0 训练脚本）

全部通过隐藏金丝雀门控与双重防泄漏校验后，产物自动输出至 `<work-dir>/release/`：
- `v08-gemini-gold-labels.jsonl.gz`：包含 `score / code / class_5way / confidence / is_trap_human / hard_human_neg / prag_vec(8d) / score_source / train_weight`（含 Queue D 的 `video_title` & `parent_text`）。
- `v08-gemini-counterfactual-pairs.jsonl`：同主题反事实最小对 + 带精确字符区间 `ai_spans` 的三明治混合评论。
- `v08_qc_report.json`：金丝雀校准 AUROC / Spearman / MAE / Bias 与全队列分布审计报告。
