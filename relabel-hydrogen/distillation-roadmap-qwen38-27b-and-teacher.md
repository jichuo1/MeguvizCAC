# Stage-2 双教师互补蒸馏路线图：Qwen 3.8 27B（评论理解）+ Gemini 4 文学/语感教师与高质量筛滤

> **适用阶段**：在 Stage-1 完成 `Hydrogen (XHigh)` Rubric v2 冻结、1,400+ 金标准盲评、Colab GPU 去偏蒸馏与 350,387 条全库重标之后启动，面向终端部署型小模型（如 `OjinjinNet-Standard`：`BAAI/bge-small-zh-v1.5` / `MiniRBT` / `RBT6`，~17M 参数，INT8 ONNX 量化，`max_len=192`，`head_len=140`）。

---

## 一、双教师互补分工设计

| 维度 | Teacher 1：`Qwen 3.8 27B`（广度评论理解教师） | Teacher 2：Gemini 4 Neon / Hydrogen XHigh（文学语感裁判 + 高质筛滤官） |
| :--- | :--- | :--- |
| **核心优势** | 中文社区语境吞吐高、对长短评论事实关系/逻辑承接/多轮回复意图理解稳定 | 对文学修辞、抒情张力、反讽戏仿、凝练金句与“空洞 AI 润色 vs 真诚表达”极其敏感 |
| **已知短板** | 容易对“工整抒情长评”“戏仿玩梗”“凝练但去情绪的文学点评”产生机械误判或模板幻觉 | 单条深度思考耗时长、配额昂贵，不适合对 5w+ 条低风险样本逐条全扫描 |
| **承担角色** | 1. 扫描 30,000–50,000 条高信息量样本（全部 20,324 条 DSH + 高难非 DSH + 风格筛净合成子集）<br>2. 抽取深层评论理解标签、语用意图与语义摘要<br>3. 构造同主题“语义双生”难负例对 | 1. **高分歧与文学样本拦截质检**：裁决 Qwen 与 Stage-1 模型分歧样本、抒情长评、引用反串与模糊边界<br>2. **1,000–2,000 对文学语感精标最小对**：专门教小模型辨别“真情实感 vs AI 工整套壳”<br>3. **高质量样本加权与幻觉剔除**：核验通过样本高权重、幻觉样本清零剔除 |

---

## 二、Stage-1 全库重标导出接口保障

Stage-1 最终导出的 `bili-hydrogen-rescored-350387.jsonl.gz` 直接为 Stage-2 `Qwen 3.8 27B` 路由与质检提供以下无损字段（零重整直接消费）：
- `score`（`0–100` 连续概率/强度分）与 `score_int`、`judgment`（`human_like` / `uncertain` / `ai_like`）
- `reason_code`（`A/R/E/T/P/H/S/Q`）与 `reason`
- `score_source`（区分 `hydrogen_xhigh_direct` 与 `colab_gpu_distilled_from_hydrogen_xhigh`）
- `origin_status`（保持 `150,017` 条确认 AI 永远为 `confirmed_ai`，平台未知为 `unverified`）
- `distilled_oof_score`、`style_gbdt_score`、`transformer_oof_score`、`uncertainty`
- `split`（`train`/`validation`/`test`/`probe`）与 `dsh`（严格隔离，确保 `Qwen 3.8 27B` 蒸馏仅在 `train` 上筛选/生成，绝不污染 `validation`/`test` 或 `U001–U017`）

---

## 三、Stage-2 流水线四步执行架构

### Step A：`Qwen 3.8 27B` 广度评论理解扫描（Colab A100 4-bit/vLLM 或配置推理端）
对 **`split == "train"` 中约 35,000–50,000 条高价值样本**（全量 `train` DSH ~16,111 条 + 非 DSH 高难样本 ~15,000 条 + `style_gbdt_score >= 55` 的合成正样本 ~12,000 条）执行结构化标注，输出：
1. `qwen_ai_prob`（连续概率与温度缩放 logits）
2. **8 项评论理解多任务软标签**（`pragmatic_tags`）：
   - `recap`（多事实压缩复盘）
   - `contrast_thesis`（`不是A而是B` / 破折号抽象升华）
   - `assistant_leak`（助手/记忆/工作流自述）
   - `contextual_ellipsis`（强依赖楼层省略语境——典型真人互动特征）
   - `external_personal_exp`（引入视频外个人经历/社区真实吐槽——典型真人特征）
   - `irony_parody`（玩梗反串/戏仿拟态）
   - `lyrical_emotion`（真情实感抒情长文）
   - `community_slang`（B 站社区黑话与原生情绪词）
3. `semantic_intent`（一句话提炼评论真实说话意图与情感走向）
4. **同主题反事实双生改写对**：对精选 3,000 条真实评论，令 Qwen 改写为“保留相同主题名词但切换真人/AI 话语方式”的对照句，彻底剪断 17M 小模型的主题词捷径。

### Step B：我方（Gemini 4 教师）文学质检筛滤与精标最小对构建
1. **自动路由拦截规则**（从 Qwen 扫描结果中筛出高风险池约 2,500–4,000 条）：
   - 路由条件 1：`abs(qwen_ai_prob * 100 - stage1_score) >= 25`（双师严重分歧）
   - 路由条件 2：Qwen 标注了 `lyrical_emotion=True` 或 `irony_parody=True` 且 `qwen_ai_prob >= 0.45`（防 Qwen 文学误杀）
   - 路由条件 3：`0.30 <= qwen_ai_prob <= 0.75` 且 `uncertainty` 位于前 15%
2. **我方精标筛滤与样本权重（`quality_weight`）**：
   - Qwen 结论与我方复核一致的高质量样本：`quality_weight = 1.0–1.5`
   - 我方纠偏覆写的高价值文学/反讽/凝练金句样本：`quality_weight = 2.0–3.0`，以我方软分与 `reason_code` 为准
   - 确认由 Qwen 主题幻想或机械套规则导致的低质噪声/幻觉样本：`quality_weight = 0.0`（彻底剔除）
3. **构建 1,000–2,000 对文学与真实感最小对**：
   - 真诚抒情排比 / 热泪致敬长评（`H`） vs 空洞工整 AI 润色小作文（`E`）
   - B 站反串玩梗 / 钓鱼拟态（`Q/H`） vs 冷静去情绪 AI 评述（`E/T`）
   - 人类一针见血短评（`H`） vs AI 模板化 `——`/`不是A而是B` 凝练收束（`E`）

### Step C：面向 17M 小模型的多任务三合一蒸馏训练（Colab GPU）
面向 `BAAI/bge-small-zh-v1.5` / `MiniRBT` / `RBT6` 学生网络（保持 `OjinjinNet-Standard` 192/140 token 截断窗口与轻量化表征结构）：
$$\mathcal{L}_{\text{total}} = w_i \cdot \Big( \mathcal{L}_{\text{soft}}(\sigma(z_s / T),\, y_{\text{fused}}) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}(\text{code}_8, \text{pragmatic}_8) + \lambda_{\text{rep}} \mathcal{L}_{\text{cos}}(h_{\text{student}}, P h_{\text{teacher}}) + \lambda_{\text{pair}} \mathcal{L}_{\text{margin}}(p_{\text{AI}} - p_{\text{Human}}) \Big)$$
- **主蒸馏损失 $\mathcal{L}_{\text{soft}}$**：融合软目标（我方覆写优先 + Qwen 质检通过样本与 Stage-1 `Hydrogen` 蒸馏分加权）。
- **多任务理解辅助头 $\mathcal{L}_{\text{aux}}$**：同时预测 Rubric v2 的 8 个证据代码（`A/R/E/T/P/H/S/Q`）与 8 项评论理解语用标签；**导出 INT8 ONNX 时直接裁剪所有辅助头**，既在训练期迫使小模型真正读懂语用意图与修辞边界，又保持推理期 **0 额外延迟与体积**。
- **表征空间对齐与最小对排序损失 $\mathcal{L}_{\text{rep}} + \mathcal{L}_{\text{pair}}$**：对齐中间层 Pooled 表征并拉大文学最小对与同主题双生对的 Margin。

### Step D：温度校准、INT8 ONNX 导出与留出集闭环验证
1. 在严格的 `validation` 切分上拟合逻辑回归融合层 + 温度系数 $T$（对齐 `OjinjinNet` 融合权重接口），锁定 `F0.5` 与低误判（`FPR <= 5%`）工作点。
2. 导出动态 `INT8 ONNX`（核验本地 `onnxruntime` 与 PyTorch 推理 MAE < `0.01`）。
3. 严禁泄漏评估：在未参与 Stage-2 蒸馏的 **`test` 留出集（特别是 DSH 留出测试集）** 与 **17 条截图真值 `U001–U017`** 上对比“Stage-1 基线模型” vs “Stage-2 Qwen 27B + Gemini 文学教师蒸馏小模型”的 Spearman 相关、AUROC、召回率与误杀率，确认显著提升后发布模型包。
