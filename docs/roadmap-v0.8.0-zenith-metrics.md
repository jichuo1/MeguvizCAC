# MeguvizCAC v0.8.0-Zenith 路线图：零架构手术 × 纯指标登顶规划

> **战略定位（Tick-Tock 架构冻结期）**：承接 [roadmap-v0.7.0-omni-authenticity.md](roadmap-v0.7.0-omni-authenticity.md)（`v0.7.0-OmniVeritas`）与 [roadmap-v0.6.0-self-evolution.md](roadmap-v0.6.0-self-evolution.md)（`v0.6.0-HyperLoop`）。`v0.7.0` 已终结全部神经网络结构、多模态因果、双曲胞腔层、安全对抗与端侧硬件编译创新，并**永久冻结 `Bundle v4` 二进制执行图**（总包 `1.998–2.163 MB <= 2.20 MB`；真端到端延迟：NPU `<= 0.04 ms/row`、WebGPU `<= 0.085 ms/row`、1T CPU `<= 0.40 ms/row`）。
>
> **`v0.8.0-Zenith` 的唯一铁律**：**零架构手术（Zero Architecture Surgery：`0` 新增算子、`0` 张量形状变化、`0 B` 体积增加、`+0.000 ms` 延迟代价）** × **形式化零 PII 不变性（`diff == 0.000000`）**。一切工作全部投入六大非架构指标杠杆（数据物理去噪与外部一键 Gemini 250k–500k+ 数据工厂、5 家族异构教师委员会、WSD+ASAM 课程与 7 路权重汤、零算子代数折叠亚 2-bit QAT 汤、尾保秩校准与 9 格 Mondrian 保形 + `qNEHVI` 级联寻优、LOFO/时序压测），把冻结架构的实际检测指标压榨至极限。

---

## 一、零架构手术契约与 MeguvizCAC 历史六大失分根因精确映射

### 1.1 `Bundle v4` 100% 原位热替换不变量
| 维度 | 冻结资产（严禁修改） | `v0.8.0-Zenith` 合法优化资产 | CI 硬门禁 |
| :--- | :--- | :--- | :--- |
| **执行图与形状** | `Nano-Draft`、`WaveByte`/`12k Picky-BPE+Engram`、`1D Packed [1,256]`、`2L` 分头 Prelude（`2 SWA-32 + 4 Bi-DeltaProduct-2`, `ReLU²`, `2:1 Funnel`, `Rational-Derf`, `24 KB Seq-LoRA-MoE`）、`16-Slot` 工作台、`2L` 共享 `InfiLoop+SCORE` Core、`EBT`/`HypSheaf+ACDL`/`TinyScope-CBM+Span-Refiner`/`VideoFastMem` | 全网权重数值（FP16 底座 → 7 路权重汤 → 零算子代数折叠 → `5-Trit / 2-Bit SEQ (g=128)` 与 `W4/INT8` 权重）+ 汤后 `0 KB` 混淆神经元正交投影 $W \leftarrow W P_\perp$ | ONNX / WGSL / `T-MAN` 计算图哈希 **`100%` 字节级一致** |
| **体积与延迟** | `1.998–2.163 MB <= 2.20 MB`；NPU `<= 0.04 ms` / WebGPU `<= 0.085 ms` / 1T CPU `<= 0.40 ms` | **零字节增加、零延迟增加**（所有教师、去噪、离线 `Era-GRL`、ASAM、权重汤、SpinQuant/AWQ 折叠全部离线执行并在导出前剥离） | `ΔSize == 0 B`；三端延迟偏移 `<= ±1%` |
| **校准与阈值槽** | `Bundle v4` 原有 1D 单调 LUT 槽与标量级联阈值槽 | 原位替换为 **严格递增 `Spline-Temp × Venn-ABERS` 1D LUT** + **9 格 Mondrian `SCoRE`** + **`qNEHVI` 12 维联合阈值** | 推理额外开销 `< 0.001 ms` |
| **零 PII 隐私公理** | 全栈禁止账号级 PII（`UID/昵称/Lv/IP/粉丝`） | 训练、外部数据工厂、校准与评估全链零账号特征 | CI 账号/昵称扰动偏移 **`== 0.000000`** |

### 1.2 历史六大失分根因 × 零架构手术打击矩阵
| 历史实证失分根因 | 量化病灶 | `v0.8.0-Zenith` 非架构根治方案 |
| :--- | :--- | :--- |
| **1. `pre_llm` 年代捷径** | `pre_llm` FPR `0.10%` vs `current_2026` 同类真人 `4.90%` | **Pillar 1**：`CCR` 逆倾向加权 (`2411.01045`) + **离线 `Era-GRL` 梯度反转头（仅 `Y=0`，导出前剥离，压真人年代线性探针 AUROC `<= 0.56`）** + `>=60%` 同年代现代白名单负例 `Two-Way pAUC` + 层次 `Group DRO` (`2510.02818`) |
| **2. `dsh_hard_slice`（`42–48` 真人飙至 `0.91`）** | 灰区中段真人被排序损失推入高分 AI 尾，打崩 `R@P95% / R@P99%` | **Pillar 1+2+3**：`Cleanlab + AUM + Cartography + LESS` 区分毒标与真灰区；**硬锁 `y ∈ [0.35, 0.65]` 禁入 `Two-Way pAUC` 极值矿池**；`CUD` 不确定性蒸馏 + 中段 Huber 锚定 + Phase-B `CeGDRO` |
| **3. `H03 / H06 / H08 / M04` 结构化/多语真人误杀** | 分点/评分表/词典/中英代码被误判为 AI（历史最高 `0.68–0.86`） | **Pillar 1+2+3+5**：`HART` (`2503.00258`) 同格式反事实孪生 + `ModernBERT-Large` / `Qwen3-32B/72B` 教师 + **汤后二次 `0 KB` 混淆神经元投影 $P_\perp$** + **9 格 Mondrian 保形**单独锁死 `Structured-Tech / Multilingual` FPR |
| **4. 合成淹没野生分布 & 子集重采样崩塌** | `v0.4-R3` 曾现 `23.8:1` 合成正样本淹没；`v0.6-R2` Stage-2 等量 12 组子集拉爆正样本占比致 `R@P95%` 断崖 | **Pillar 1+3**：确立 **数据物理守恒律**——全库合成:野生真实正样本 **`<= 1.5:1`**、反事实步数占比 **`<= 12%`**、Stage-2/3 严守自然生态先验（禁止破坏性等量重采样）且金标留存 **`>= 99.5%`** |
| **5. `Smooth PAVA` 顶部平台压缩** | 保序回归在 `p ∈ [0.85, 0.999]` 合并排序阶梯，压平 `TPR@0.1% / 0.01% FPR` | **Pillar 5**：彻底替换为 **严格递增 `Spline-Temp` (`2006.12800`) × `Venn-ABERS` (`1211.0025`, `2601.19944`) 1D LUT**（导数下界 $f'(z) \ge 10^{-3}$，**`0` 顶部并列、`0` AUROC 损失**） |
| **6. 亚 2-bit 量化税放大尾损** | `|ΔAUROC| <= 0.0025` 在 `0.01% FPR` 极端尾仍损失 `2%–4%` 召回 | **Pillar 3+4**：`ASAM` 平坦盆地 + **零算子代数折叠**（Core `SpinQuant R_1` + 全头 `R_2` + Prelude `SmoothQuant` + **`ReLU²` 二次正齐次缩放**）+ Hessian-AWQ/GPTQ 初始化 + `UPQ/BitDistill` + **3 路潜权重 QAT-EMA 汤**，压至 **`|ΔAUROC| <= 0.0010`** |

---

## 二、Pillar 1：数据物理去噪 × 外部一键 Gemini `250k–500k+` 数据工厂

### 2.1 四信号自动化标签去噪（精准清洗全库 `~2–4%` 高毒误标）
在 `1.4M–5.0M+` 训练池上利用 5 折折外（OOF）FP16 探针提取四项互补信号：
1. **Confident Learning (`Cleanlab`, `arXiv:1911.00068`)**：估计联合噪声矩阵 $\hat{Q}_{\tilde{y}, y^*}$；
2. **Area Under the Margin (`AUM`, `arXiv:2001.10528`)**：结合伪类阈值样本自动标定 99% 截止线 $\tau_{\text{AUM}}$；
3. **Dataset Cartography (`arXiv:2009.10795`)**：精准切分 `Easy`（Phase A 底座）、`Ambiguous`（**严禁删除！**送入 Phase B 边界强化）、`Mislabeled`（送教师委员会重裁）；
4. **低秩梯度影响归因 (`LESS`, `arXiv:2402.04333`; `TracIn`, `arXiv:2002.08484`)**：直接锁定对 `val_tail`（`0.1% FPR` 尾区与 `dsh_hard_slice`）具有最大负梯度影响的 Top-3% 毒标。

### 2.2 外部独立一键 Gemini 金标与反事实数据工厂（已落地可运行工程包：`tools/gemini_v08_datafactory/`）
> **零反重力额度消耗**：用户直接在外部运行 [tools/gemini_v08_datafactory/](../tools/gemini_v08_datafactory/README.md)（支持 **Gemini Batch API 官方 5 折**、**纯标准库并发 Realtime API** 与 **离线 Prompt 包**），将金标与反事实最小对池从当前 **`~2.3 万条`** 暴力扩容 **`10×–20×`** 至 **`25 万 – 50 万+ 条`**！

| 工程组件 | 文件路径 | 核心设计 |
| :--- | :--- | :--- |
| **Rubric v3.0 与强约束 Schema** | [rubric_v08.py](../tools/gemini_v08_datafactory/rubric_v08.py) | 100% 继承 `Hydrogen Rubric v2.1` 尺度闭环（`0–30=>H`, `31–69=>P/S/Q`, `70–100=>A/R/E/T`）与 `PRAG_TAGS` 8 维标签，扩展 `class_5way`、`is_trap_human`、`confidence`、`ai_span_texts -> ai_spans` 精确字符下标与 3 位防错位校验码 `tag` |
| **四队列零 PII 导出与隐藏金丝雀** | [01_export_queue.py](../tools/gemini_v08_datafactory/01_export_queue.py) | 导出 **Q-A（噪声手术 `20%–24%`）**、**Q-B（2026 现代野生难例 `40%–48%`）**、**Q-C（反事实最小对 + 三明治混写 Span `12%–25%`）**、**Q-D（`video_title + parent_text + reply_text` 楼中楼 `15%`）**；双重防泄漏过滤全部 `val/test/user_confirmed` 与内置陷阱；向 Q-A/B 盲插 **2% `val_gold` 隐藏金丝雀**（真值仅留本地 `canary_vault.json`） |
| **两级自动提级调度引擎** | [02_run_gemini_factory.py](../tools/gemini_v08_datafactory/02_run_gemini_factory.py) | 每请求 **25 条超级拼包** + Batch API 5 折 / 断点续跑并发 REST / 离线包 / 本地 `--mode dry-run`；**Tier-1 (`gemini-2.5-flash`) 全量初审 → 自动路由 ~15% 灰区/低置信/陷阱/分歧行至 Tier-2 (`gemini-2.5-pro`) 终审**，25 万条 Batch API 总费用仅约 **`~$18.6`** |
| **隐藏金丝雀熔断 QC 与清洁合库** | [03_qc_and_merge.py](../tools/gemini_v08_datafactory/03_qc_and_merge.py) | 核验隐藏金丝雀 `AUROC >= 0.94 / Spearman >= 0.85 / |Bias| <= 8.0`（不达标自动 Fail-Fast 阻断）；**100% 剥离金丝雀**；独立重跑防泄漏三查；直接输出即插即用 `v08-gemini-gold-labels.jsonl.gz` 与 `v08-gemini-counterfactual-pairs.jsonl` |
| **一键启动脚本** | [run_factory.ps1](../tools/gemini_v08_datafactory/run_factory.ps1) / [run_factory.sh](../tools/gemini_v08_datafactory/run_factory.sh) | Windows / Linux / Colab 单命令完成 Export → Tier-1 → Tier-2 → Canary QC → Release 导出 |

### 2.3 年代正交去偏四道防线
1. **人类侧 `CCR` 稳定截断逆倾向加权（`arXiv:2411.01045`）**：$w_{\text{IPW}}(x) = \operatorname{clip}(\hat{\pi}(e_{2026}|x)/(\hat{\pi}(e|x)+\varepsilon), 0.25, 4.0)$，降权明显旧年代老梗，提权句式现代化的 `pre_llm` 真人长评；
2. **离线 `Era-GRL` 对抗梯度反转头（仅 `Y=0`，导出前物理剥离 `0 B`）**：强迫 `Coda` 表征对真人年代盲视（线性探针 AUROC `<= 0.56`）；
3. **`Two-Way pAUC` 同年代白名单硬锁**：`>= 60%` 难负例来自 5 教师全票确证的 `2025–2026` 现代真人；
4. **层次 `Group DRO`（`arXiv:2510.02818`）**：按 `(4 年代 × 4 语体)` 动态压低最差子群风险。

---

## 三、Pillar 2：离线五家族异构前沿教师委员会 × 尾部聚焦复合蒸馏

| 教师家族 | 底座模型 | 独特归纳偏置与专项补强 |
| :--- | :--- | :--- |
| **T1** | **`DeBERTa-v3-Large`** (`2111.09543`, 435M) | RTD 内容-位置解耦注意力，极敏锐捕捉短句替换、混写接缝与隐蔽润色 |
| **T2** | **`ModernBERT-Large`** (`2412.13663`, 395M, 8k RoPE) | 2T 文本+代码预训练，直接根除 `H03 / H06 / M04` 硬核技术与代码真人误杀 |
| **T3** | **`BGE-M3` 判别微调** (`2402.03216`, 568M) | 100+ 语种稠密+稀疏表征，清零 `M01–M06` 多语种误报与漏检 |
| **T4** | **`Qwen3-32B / 72B-LoRA`** (`2505.09388`) | 深层中文网络梗、反讽/玩梗语义常识与特权语境 `OPSD` 蒸馏靶 |
| **T5** | **3× 同构 FP16 `v0.7.0` 教师群** | 零架构表征鸿沟，提供 `16-Slot` 隐态对齐与 $R \in \{2,4,6\}$ 深度单调轨迹靶 |

- **不确定性加权共识（`CUD` `2602.12687` + `MT-BKD` `2605.27967`）**：按每教师验证集最优温度校准与香农熵加权，教师高分歧样本自动降权 $\omega_{\text{cons}} = \exp(-\kappa_U U_i^{\mathcal{T}})$，不强迫学生拟合真不可分噪声。
- **四合一 Zenith 蒸馏复合损失（`0` 推理开销）**：
  1. **Z-Score Logit 标准化解耦蒸馏（`Logit-Standardized DKD`, `arXiv:2403.01427` + `2203.08679`）**：消除 72B→9M logit 幅值失配，强化 5 类非目标关系结构；
  2. **`ListNet-Tail` × 同年代白名单 `Two-Way pAUC`（`2203.00176`, `2206.11655`）**：排除 `y ∈ [0.35, 0.65]` 灰区，聚焦 `0.1% / 0.01% FPR` 极端顶部排序；
  3. **分级绝对锚定 + 中段 Huber 保真**：确证真人钉在 `< 0.06`，硬核结构化真人钉在 `< 0.14`，中段 `42–48` 锁在客观灰区，稳定锐利温度 $T \in [0.85, 1.35]$；
  4. **特权语境 `OPSD` + `16-Slot` 隐态与多步单调对齐**。

---

## 四、Pillar 3：`WSD × ASAM` 平坦盆地、3 阶段反遗忘课程与 7 路权重汤

1. **`WSD` 河谷课表（`2404.06395`, `2410.05192`, `2412.19437`）× 末 30% `ASAM`（`2102.11600`, `SAFT 2504.04257`）**：
   - `Warmup (5%) -> Stable (75%) -> Decay (20%)`，在 Phase A 末尾冻结共享祖先检查点 **`θ_anchor`**（保证多路线性模式连通性并节省 `~60%` 算力）；
   - 后 30% 步数开启尺度自适应 `ASAM`（`ρ: 0.04 -> 0.08`），一石三鸟打通 **LOFO 泛化 × 零壁垒权重合并 × 抗 2-bit 量化抖动**。
2. **三阶段反遗忘锚定课程（`CeGDRO 2411.15272` + `RDS-MOPD 2611.12012`）**：
   - **Phase A（洁净正交底座）**：去噪野生/金标池 + `CCR-IPW + Offline Era-GRL + Z-DKD`，导出 `θ_anchor`；
   - **Phase B（难边界攻坚）**：`CeGDRO` 强化 `dsh_hard_slice / H03/H06/M04 / ultra_short`，强制 **`>= 45%` Phase A 底座重放**（严禁重演 `v0.6-R2` Stage-2 分布偏移！）；
   - **Phase C（对抗与 2027 推理外推冷却）**：注入洗稿/同形字/三明治后缀欺骗/`READER` 短评，强制 **`>= 50%` 锚定重放**。
3. **7 路多样性权重汤（`B1..B7`）+ 汤后二次 `0 KB` 手术**：
   - 从 `θ_anchor` 分出 7 个互补专长分支（Anchor-Balanced / Extreme-Tail / Human-Shield / UltraShort-CMI / Wild-2026-LOFO / Adversarial-Span / Ultra-Flat-SWA）；
   - 针对 `~9M` 紧凑模型采用 **保守 `DARE`（$p \in [0.10, 0.25]$）+ `TIES`（`2306.01708`）+ 独立 `val_soup` 贪心 pAUC 入汤（`2203.05482`）**；
   - 在最终冻结的 `θ_soup^FP16` 上重新执行一次 **`0 KB` 混淆神经元正交投影 $P_\perp$** 折叠进 `coda_proj / ai_head`。

---

## 五、Pillar 4：零算子代数折叠 × 亚 2-bit 零税 QAT 汤（`|ΔAUROC| <= 0.0010`）

### 5.1 严守零架构手术的 4 类精确离线代数等价折叠（`0` 在线 Hadamard 算子！）
1. **`InfiLoop Core` 残差流（`RMSNorm SubLN`）**：离线折叠 Cayley 学习正交旋转 **`SpinQuant R_1 ∈ O(384)`**（`2405.16406`）；
2. **全部注意力头（Prelude + Core）**：离线折叠每头正交旋转 **`R_{2,h} ∈ O(64)`** 至 $W_{V,h}', W_{O,h}'$；
3. **全部 `ReLU²` FFN 层（`1.58-bit` 最大参数池）**：利用 **`ReLU²` 严格二次正齐次性** $\text{ReLU}^2(s_j u_j) \equiv s_j^2 \text{ReLU}^2(u_j)$，离线精确折叠：
   $$W_{\text{down}}\,\text{ReLU}^2(W_{\text{up}} x) \equiv \big(W_{\text{down}}\operatorname{Diag}(s^{-2})\big)\,\text{ReLU}^2\big(\operatorname{Diag}(s) W_{\text{up}} x\big)$$
4. **`Prelude / Coda` 非递归层（`Rational-Derf`）**：严禁跨逐元素非线性乱折稠密 $R_1$，改用 **`SmoothQuant (2211.10438)` + `AWQ`** 正对角缩放折入既有 $\gamma$ 与下游线性矩阵。

### 5.2 二阶 Hessian 初始化 → 渐进退火 → 3 路连续潜权重 QAT-EMA 汤
1. 在 `n=4,096` 分层尾部校准集上执行 **Hessian-AWQ/GPTQ (`2210.17323`, `2306.00978`) + `EfficientQAT (2407.11062)`** 块级标尺初始化（QAT 步 `0` 前 PTQ 漂移已 `< 0.006`）；
2. **`UPQ (2506.09104)` 两段渐进退火**：`FP16 -> W4A8 (1,500 steps) -> 5-Trit FFN + 2-Bit SEQ Attn (4,000 steps)`，全程由 `θ_soup^FP16` 以 **`BitDistill (2510.13998)` + `QuEST (2502.05003)` + `WinQ (2605.17471)` + 白名单 `Two-Way pAUC`** 监督；
3. **3 路连续潜权重 QAT-EMA 汤（`HDRQ 2505.23651`）**：在连续 Shadow Master Weights $W_{\text{latent}}$ 空间加权融合 3 路微差异 QAT 分支，再执行 `500` 步极低 LR 贴网精修（Grid-Snap Polish），将最终量化税死锁在 **`|ΔAUROC| <= 0.0010`**、`ΔTPR@0.1%FPR >= -0.010`。

---

## 六、Pillar 5：保秩尾部校准、9 格 Mondrian 保形与 `qNEHVI` 12 维级联寻优

1. **彻底取代 `Smooth PAVA`：`Spline-Temp (2006.12800)` × `Venn-ABERS (1211.0025, 2601.19944)` 严格递增 1D LUT**：
   - 在独立 `val_cal` (`n >= 4,000`) 上先拟合三参数 Beta-Logit 偏态温度映射（$T \in [0.85, 1.35]$），再拟合带正导数下界 $f'(\tilde{z}) \ge 10^{-3}$ 的 Fritsch-Carlson 严格递增三次样条，编译入 `Bundle v4` 原有 `256`-节 LUT——**`0` 顶部并列、`ΔAUROC_cal ≡ 0.0000`、`ECE < 0.015`**；
   - 同步预存 `<= 2 KB` Venn-ABERS 双边界宽度 $\delta_{\text{VA}}(z) = p_1 - p_0$，高认知不确定度自动转 `Tier-3` 弃权。
2. **9 格零 PII Mondrian 分组保形（`2305.12616`）+ `SCoRE` E-Value（`2503.13050`, `2603.24704`）+ `Mandap-B`（`2609.31547`）**：
   - 按 `({Short, Medium, Long} × {Casual, Structured-Tech, Multilingual})` 9 格做经验贝叶斯层级收缩分位数标定，保证**每一子群独立满足 `FPR_g <= α`**，彻底消灭结构化与多语子群隐藏 FPR 尖峰，并将 `@α=0.01` 灰区率压至 **`<= 5.5%`**。
3. **12 维 5 级级联阈值四目标联合寻优（`qNEHVI`, `arXiv:2105.08195`）**：
   - 严格三切分互不相交 **`val_soup` / `val_opt` / `val_cal`**（保障 Conformal 有限样本定理 100% 严格成立）；
   - 在 `val_opt` (`n >= 5,000`) 上联合优化 `Nano-Draft / VideoFastMem / EBT-MoR / HypSheaf-ACDL` 12 维阈值，同时最大化 `(TPR@0.1%FPR, TPR@0.01%FPR, -GrayZoneRate, -Latency_1T)`。

---

## 七、Pillar 6：全维指标跃迁靶标与 `V8-M0 ~ V8-M4` 可证伪门控

### 7.1 `v0.7.0-OmniVeritas` → `v0.8.0-Zenith` 零架构改动指标目标总表
| 核心指标 | `v0.7.0` 冻结基线门槛 | **`v0.8.0-Zenith` 封板靶标** | 主驱动支柱 |
| :--- | :---: | :---: | :--- |
| **架构 / 体积 / 延迟** | `1.998–2.163 MB <= 2.20 MB`；NPU `<=0.04` / WebGPU `<=0.085` / 1T `<=0.40 ms` | **100% 冻结原位不变（`0` 新算子、`0 B`、`+0.000 ms`）** | 零架构手术铁律 |
| **`test_bili_gold` AUROC / PR-AUC** | `>= 0.985 / >= 0.978` | **`>= 0.993 / >= 0.990`** | P1 外部金标工厂+去噪 + P2 5教师 + P3 7路汤 |
| **`test_dsh_gold` AUROC / PR-AUC** | `>= 0.980 / >= 0.972` | **`>= 0.991 / >= 0.987`** | P1 反事实 + P2 CUD + P3 CeGDRO |
| **`dsh_hard_slice`（原 `42–48` 真人最高分）** | 历史飙至 `0.91` | **`p95 < 0.25`、`max < 0.42`（0 例超 `0.50`）** | P1 灰区隔离 + P2 中段锚定 |
| **`R@P95%` / `R@P99%`** | `>= 92% / >= 78%` | **`>= 96.5% / >= 88.0%`** | P1 毒标清洗 + P2 尾排序 + P3 权重汤 |
| **`TPR@0.1%FPR` / `TPR@0.01%FPR`** | `>= 78.0% / >= 45.0%` | **`>= 86.0% / >= 60.0%`** | P1+P2 + P4 零税 + P5 去 PAVA 顶压 + `qNEHVI` |
| **野生真实 AI `Recall@0.1%FPR`** | `>= 70.0%` | **`>= 80.0%`** | P1 合成:野生 `<= 1.5:1` + `LESS` 对齐 |
| **外部盲测 `OjinjinBench`（纯文本 / 零 PII 语境）** | `>= 0.94 / >= 0.96` | **`>= 0.965 / >= 0.980`（硬负误杀 `0%`）** | P1 年代去偏 + P2 5教师 + P3 ASAM 汤 |
| **6 大家族 `LOFO` 最差家族 AUROC** | `>= 0.965` | **`>= 0.982`**（2027 推理 `TPR@0.1% >= 82%`） | P2 5家族异构 + P3 ASAM 汤 + `HART` |
| **超短评（`<=20` 字）`R@P95%`（零 PII 语境）** | `>= 80.0%` | **`>= 87.0%`**（纯文本 `>= 78.0%`） | P1 短评反事实 + P2 `OPSD` + P3 `B4` |
| **真人陷阱 `H01–H09 / M01–M04 / P01–P03`** | `< 0.10` | **全部 `< 0.06`（`H03/H06/H08/M04 < 0.05`）** | P1 孪生 + P2 ModernBERT + P3 汤后手术 + P5 Mondrian |
| **AI 陷阱 `A01–A05 / M05–M06` 最小值** | `>= 0.92` | **`>= 0.955`** | P2 DeBERTa-v3/BGE-M3 + P3 `B2/B6` |
| **年代偏移 `|FPR(2026) - FPR(pre_llm)|`** | 早期差 `4.8%` | **`<= 0.25%`（`Era` 线性探针 AUROC `<= 0.56`）** | P1 `CCR-IPW + Offline Era-GRL` |
| **校准与保形（`ECE` / `T` / `@α=0.01` 灰区）** | `ECE ~0.025` / 灰区 `<= 8%` | **`ECE < 0.015` / `T ∈ [0.85, 1.35]` / 灰区 `<= 5.5%`** | P5 `Spline-Temp × Venn-ABERS` + 9格 Mondrian |
| **量化税 `|ΔAUROC(FP16 -> 5-Trit/2-Bit)|`** | `<= 0.0025` | **`<= 0.0010`**（`|ΔTPR@0.1%FPR| <= 0.010`） | P3 ASAM + P4 零算子折叠 + 潜权重 QAT 汤 |

### 7.2 `V8-M0 ~ V8-M4` 分阶段 Go / No-Go 里程碑
- **`V8-M0`（数据物理去噪 × 外部 Gemini `250k–500k+` 金标工厂 × 5 家族共识库）**：全分片隐藏金丝雀通过 QC、毒标精确率 `>= 85%`、合成:野生 `<= 1.5:1`、零 PII & 三查泄漏 `== 0`、金标留存 `>= 99.5%`、5 教师离线集成 `test_bili_gold AUROC >= 0.995`。
- **`V8-M1`（FP16 单模型 `WSD + ASAM + 3 阶段课程`）**：较 `v0.7.0` FP16 单模型 `R@P99% >= +4%`、`TPR@0.1%FPR >= +4%`、`dsh_hard_slice max < 0.45`、真人年代探针 AUROC `<= 0.56`。
- **`V8-M2`（7 路权重汤 + 汤后 `0 KB` 手术）**：`θ_soup^FP16` 严格碾压 7 路中每一单模型，`TPR@0.1%FPR` 再涨 `>= +2%`、`TPR@0.01%FPR` 再涨 `>= +4%`、`Worst-Family LOFO AUROC >= 0.982`、全真人陷阱 `< 0.06`。
- **`V8-M3`（零算子代数折叠 + 渐进 `UPQ/BitDistill` + 3 路潜权重 QAT 汤）**：`|ΔAUROC| <= 0.0010`、`|ΔTPR@0.1%FPR| <= 0.010`、`Bundle v4` 计算图哈希 `100%` 一致、`ΔSize == 0 B`。
- **`V8-M4`（尾保秩校准 + 9 格 Mondrian `SCoRE` + `qNEHVI` 12 维寻优 + 终审盲测）**：9 格每格独立满足 `FPR_g <= α`、`ECE < 0.015`、`@α=0.01` 灰区 `<= 5.5%`、一次性启封 `OjinjinBench` 与全套 `MeguvizBench-v4` 达标封板。

> 完整学术推导与 45+ 篇经核验 arXiv 索引账本详见本地研究底稿 `scratch/arch_research/07_v08_metric_maximization.md`；外部一键数据工厂运行说明详见 [tools/gemini_v08_datafactory/README.md](../tools/gemini_v08_datafactory/README.md)。
