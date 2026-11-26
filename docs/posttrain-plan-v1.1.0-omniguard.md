# MeguvizCAC v1.1.0-OmniGuard — 深度后训练总体蓝图与数据进化实证汇编

> **编制依据**：综合四代理 Swarm 协同产出——`opus-posttrain`（Claude Opus 5.5 High，代码级架构审计与后训练蓝图 `P0–P10`）、`gpu-data-evo`（本地 RTX 4060 Ti 三项严格对照实验 Exp 1–3）、`lit-adversarial`（2025–2026 对抗检测与中文绕过文献审计）、`lit-datacentric`（2025–2026 零字节数据中心前沿方法审计）。  
> **严格代码锚定**：所有模块名与行号直接对应 `meguviz_cac/v10_apexloop.py`（简称 `V10`）、`meguviz_cac/train_v10_apexloop.py`（简称 `TR`）、`meguviz_cac/v09_cascade_cert.py`、`meguviz_cac/v06_hyperloop.py` 及 `meguviz_cac/v11_*.py`。  
> **核心铁律**：纯 AI 检测 **绝对零退化**（`test_bili / test_dsh / OjinjinBench v1 & v2 >= v1.0.0-R1`），`Bundle v6 <= 2.20 MB`，端侧 1T-CPU 级联延迟增幅 `<= +0.03 ms`，绝对 `Zero-PII`（`diff === 0.000000`）。

---

## 0. 代码级 MUST-FIX 阻断项（启动 Stage 1 前必改）

经 `opus-posttrain` 与 `lit-adversarial` 逐行核查 `v10_apexloop.py` / `v09_cascade_cert.py` / `v11_*.py`，以下 **12 项硬性发现（F1–F12）** 及 **2 项 `v11` 前端/流式漏洞** 必须先于训练落地（其中标记 ✅ 者已在本轮同步修复）：

| # | 位置与根因 | 后果 | 落地修正方案 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| **F1** | `ATTRIBUTION_5WAY` = `(Organic0, Assisted1, Bot2, Agent3, Cluster4)`；EBT `e_ai_min = energies[:, 2:].amin()`（`V10 L279`） | 若按初版路线图将 Promo 插在 idx 2 会导致全表错位并被 `2:` 误吞为 AI 谷 | 将 `Human-Commercial-Promo` **追加在 idx 5**；`e_ai_min = energies[:, 2:5].amin()`，`e_hu_min = min(E0, E5)` | 已更新路线图，待 `v11_omniguard.py` 实装 |
| **F2** | `DynaSeamMILAggregatorV11` 默认 `(32, 48)` = `31,942` 参数 = **`62.4 KB`** | 超出路线图 `33.7 KB` 预算近一倍 | 默认实例化改为 **`(chunk_emb_dim=16, hidden=32, num_heads=2)`** = `~17,191` 参数 = **`33.6 KB`** | ✅ 已修复 `v11_longstream.py` |
| **F3** | `DynaSeam` `M=1` 时 `q_keep` 会折减单块引号评论，且 `logit_promo_doc` 未受 `multi_gate` 门控；另存在全篇引号包裹绕过（`Quote-Wrap AI Bypass`）与长文引用推广吐槽无保护漏洞 | 短评带引号或 `M=1` 推广头非恒等；全篇套引号可压低 AI 分 `85–92%` | 推理引擎在 `M=1` 旁路 DynaSeam；模块内 `q_keep / q_pen / head_promo_delta` 均乘 `multi_gate`；引入 `g_q = σ(12·((1-quote_ratio)-0.35))` 封堵全篇套引号绕过并保护长文引用推广吐槽 | ✅ 已修复 `v11_longstream.py` + 扩展 `V11-L1` CI |
| **F4** | `certify_ltt_compcrc_cascade_policy`（`v09_cascade_cert L156–195`）在 48 点 2D 网格上取所有通过点的最大退出率，并用 `eff_target_risk` 放宽 $\alpha$，且失败时回退到未认证 zero-FN | 非严格固定序列 LTT（FWER 可达 $48\delta$） | 改为**预排序 1-D 路径严格固定序列 LTT**，禁止放宽 $\alpha$，认证失败回退为禁用 R1 退出 | 待 `v11_cascade_cert.py` 实装 |
| **F5** | `apply_zero_cost_confounding_surgery`（`v06 L650`）在全 256-d 上改写 `ai_head` | 重新向 `[192:256]` 推广专有区泄漏 AI 权重 | 手术严格限制在 `[0:192]`，并由 Stage 3b **双轴斜交 LEACE** 取代（见 §9） | 待 Stage 3b 实装 |
| **F6** | `0.10 * cbm_logit` 进入 AI 原始 logit 且 `cbm_enc` 读取全 256 维（`V10 L849–862`） | 防火墙旁路泄漏（promo → AI） | 置零 `cbm_enc.weight[0:112, 192:256]`（恰好对齐 1 个 W4 `g64` 量化组），冻结 `cbm_audit_w[112:128] = 0` 预留 16 个治理概念槽 | 待 `v11_omniguard.py` 实装 |
| **F7** | Adam 按逐元素梯度归一化，对共享张量注册 grad-scale hook **无法**实现按行差异化 LR | 新增专家/EBT 行无法与老权重分离学习率 | 将新增行拆为独立 `nn.Parameter`（`fc2_new`, `core_routed_new`, `router_keys_new`），前向 `torch.cat`，打包时合并回 `v1.0` 同名键 | 待 `v11_omniguard.py` 实装 |
| **F8** | `V10 forward_streaming_long` 用 `256-tok/192-stride`，`v11` 用 `plan_long_chunks_v11` `240-char`；且 `V10` 未实现跨块 `z_H` 传递（仅有 `fwd_state` + `Top-32 Haar`） | 训练与服务分块错配 | 训练与服务统一采用 `plan_long_chunks_v11`；v1.1 仅延续 `fwd_state + Haar sketch`，不引入 `z_H` 跨块传递 | 已同步更新路线图 |
| **F9** | MoLE `8 → 10` 专家增量实为 `2×(24×24 + 24 + 64 + 6) = 1,340` 参数 = **`+2.6 KB`** FP16 | 旧文档误估为 `+4.6 / 11.5 KB` | 统一采用 **`+2.6 KB`** 精准预算核算 | 已同步更新路线图 |
| **F10** | `v11_features.py` `56-d` 前 40 维已重造为长度无关速率与 `MATTR-64`，不等价于 `v07` `40-d` | 直接列复制导致 Rung-1 严重冷启动震荡 | 最小二乘桥接矩阵 $W'=WA, b'=b+Wc$ + Stage 1a 前 600 步预激活蒸馏（P1.2） | 纳入 Stage 0 / 1a |
| **F11** | `update_deepseek_bal_bias` 目标为 `1/E`；扩为 `E=10` 后 8 个老专家每步被压低 `0.004` | 隐性 AI 路由漂移 | 改为**分组目标掩码平衡**（老专家维持 `(1-ρ_new)·load_R1(e)`，新专家渐进 `0 → 0.15`）（P1.4） | 纳入 Stage 1b |
| **F12** | 延迟耦合：`r1_cost_rel = 0.12`，在 `~1 ms` 级联上每损失 1 个百分点 R1 退出率 ≈ `+0.0088 ms` | `+0.03 ms` 预算硬约束 **`ΔExit_R1 <= 3 pts`**（含 `f56` 提取开销） | `r1_promo_head` 在认证阈值 $\tau_p$ 下有机假阳 `<= 3%`，显式硬广由零成本正则门控拦截 | 纳入 P4 门禁 |
| **B-F1** | `v11_features.py:152` `_BILI_EMOTE_V11_RE.sub(" ", clean)` 注入人工汉字空格；野生群号/微信号未脱敏 | 误抬 `_CJK_SPACE_RE`；群号构成潜在 PII | 改为 `sub("", clean)`，并增加上下文触发的 `<CONTACT_ID>` Zero-PII 脱敏 + PCR-ToxiCN 中文绕过词表 | ✅ 已修复 `v11_features.py` |

---

## 1. RTX 4060 Ti 本地数据进化实证（`gpu-data-evo` Exp 1–3）

所有实验严格在 `train / validation` 执行（零 `test`、零 `OjinjinBench` 读取），产出物已落盘至 `v11-omniguard-datafactory/v11_bge_knn_promo_hard_negatives_2k.jsonl`（`2,000` 条，`1.33 MB`）。

### 1.1 Exp 1：`SemDeDup` 语义冗余与流形坍缩审计（`100,000` 条洁净 `T0/T1`，`bge-small-zh-v1.5`）

| 数据源（全库净池规模） | 抽样 `N` | 精确重复 | `nn_cos` 中位 / P95 | `HasNN@0.92` | `Prune@0.92 / 0.95` | 熵 `H` | 诊断结论 |
| :--- | ---: | ---: | :---: | ---: | :---: | ---: | :--- |
| **`v04_confirmed_ai_plugin`** (`250k`) | 24,000 | 0.00% | **`0.8809`** / 0.9409 | **`17.87%`** | **`12.06%`** / 1.26% | 0.970 | **重度槽位模板坍缩**（`L2_46-128` 剪枝率达 `16.59%`，与真实 B 站源 `cos>=0.95` 跨源重合 **0.00%**） |
| **`colab_gpu_distilled`** (`163k`) | 20,000 | **`1.92%`** | 0.7197 / 0.9360 | **`7.07%`** | **`5.01%` / `2.79%`** | 0.979 | 短评 `L1<=45` 存在隐蔽精确/硬件模板克隆（最大度 `56`） |
| `multi_teacher_ensemble` (`622k`) | 20,000 | 0.00% | 0.6924 / 0.8174 | 0.64% | 0.40% / 0.27% | **0.984** | 高度多样，无坍缩 |
| `pre_llm_human_prior` (`129k`) | 15,120 | 0.00% | 0.7065 / 0.8511 | 1.22% | 0.71% / 0.40% | 0.974 | 洁净多样 |
| `v08_gemini_80k` (`79.8k`) | 15,000 | 0.00% | 0.7021 / 0.8867 | 1.69% | 1.02% / 0.26% | 0.979 | 最洁净（同 `N=5,880` 匹配 `Prune@0.95` 仅 **`0.07%`**） |
| **`v11_wave1_gold`** (`5,880`) | 5,880 | 0.00% | 0.7598 / **`0.9458`** | **`13.96%`** | **`10.29%` / `2.45%`** | 0.957 | **抓出隐蔽隐患**：含 **`192` 条** `v04_confirmed_ai_plugin` 记账/凑单模板克隆（因匹配 `"凑单"` 进入抽样） |

> **关键发现**：在 `80k` 蓄水池上运行 `SemDeDup @ 0.92` 共剪枝 `8,545` 条（`10.68%`），其中 **`99.4%`** 精准落于 `v04_confirmed_ai_plugin`（`6,939`）与 `colab_gpu_distilled`（`1,556`）两源，而对真实 B 站源几乎零误伤！

### 1.2 Exp 2：BGE k-NN 边界硬负样本挖掘（`2,013` 确认推广 × `150,000` 洁净有机池）
- 施加 `SemDeDup < 0.90`（去重 127）、单查询上限 `<= 12`（防刷屏 2,099）、`v04_confirmed_ai_plugin <= 150`（防合成淹没 2,248）与显式导流/复读护栏后，精准收获 **`Top-2,000`** 边界硬负样本（`cos ∈ [0.7041, 0.8682]`，`182` 条 `>= 0.78`）。
- **构成**：**`92.5%`** 为真实 B 站有机评论（`self_promo` 边界 `893` / `soft_shill` 边界 `575` / `coupon` 边界 `532`；其中真人 `Q4 = 78.9%`，纯 AI `Q1 = 18.7%`）；抽检 `10/10` 为 **100% 精准边界有机样本**（如预算 5000 配电脑求助、iPhone 吐槽、一加 Ace 刷机真实体验、SQL 索引干货等）。

### 1.3 Exp 3：8 组 × 3 种子双轴严格对照实证（`BGE-512 + 56-d` 探针，零泄漏）

| 实验组（3 种子均值） | `PB48` AI ROC | `PB47` Promo ROC | `PB47` 4Q 联合精度 | `Q3` AI 误杀 | `LB24` Promo ROC / FP | **`Val2k` AI ROC** | **`Val2k` AI FPR@0.5** | **`Val2k` AI R@P95** | `Val2k` 野生推广误报 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E_AI_CTRL`（单轴纯 AI 对照） | 0.9317 | 0.6057 | 47.5% | 0.0% | 0.3439 / 27.0% | **0.9896** | 0.93% | 89.0% | — |
| `E_AI_CTRL_DeDup`（+`SemDeDup`） | 0.9294 | 0.4940 | 43.3% | 0.0% | 0.4550 / 7.9% | **0.9895** | **`0.82%`** | **`90.9% (+1.9)`** | — |
| `E0_AIMatched`（无防火墙共享基线） | 0.8889 | 0.9257 | 77.3% | 0.0% | 0.9735 / 15.9% | 0.9786 | 1.10% | 80.6% *(-8.4!)* | 2.65% |
| `E1_+80kPurif`（+80k 蓄水池+51+75） | 0.9005 | 0.9342 | 77.3% | 0.0% | 0.9735 / **14.3%** | 0.9792 | 1.01% | 81.5% | 1.12% |
| **`E2_+KNN+DeDup`（共享主干全集）** | 0.8953 | **`0.9541 (+0.020)`** | **`80.1% (+2.8)`** | **0.0%** | 0.9735 / **14.3%** | 0.9863 | 1.06% | 85.4% *(+3.9)* | **`1.03%`** |
| **`E2_OrthoFirewl`（隔离 AI 主干）** | **`0.9259`** | 0.9269 | 78.7% | **0.0%** | **`0.9894`** / 19.0% | **`0.9878`** | **`0.84%`** | **`88.3%`** | 1.08% |

**四大实证铁律**：
1. **单用 `SemDeDup@0.92` 即可白嫖纯 AI 精度**：纯 AI `R@P95` **`89.0% → 90.9% (+1.9 pts)`**，`FPR@0.5` **`0.93% → 0.82%`**；
2. **`2k k-NN + SemDeDup` 是严格帕累托支配解**：推广 `PB47 ROC +0.0199`、4Q 联合精度 `+2.8 pts`、野生推广误报下降 **2.8×**（`2.85% → 0.82–1.03%`），且即使在共享主干内也反向回收 `+3.9 pts` AI `R@P95`；
3. **直接实证 `opus-posttrain` 混合三区防火墙（P1.3）的不可替代性**：无保护共享主干令 AI `R@P95` 掉点 `-8.4 pts`，而纯硬隔离小容量推广主干使 `PB47 Promo ROC` 从 `0.9541` 跌至 `0.9269`；唯有 **`[0:128]` 受控共享（`D_λ = 0.35`）+ `[128:192]` AI 专有 + `[192:256]` Promo 专有 + R1 KL 锚定** 可双赢！

---

## 2. 深度后训练八大支柱蓝图（`P1–P8` + 文献与实证融合 `P9–P10`）

### P1 · 零震荡外科手术热启动（从 `v1.0.0-R1 {version}_fp16_master.pt` 预折叠主权重起步）
> 严禁从已 SpinQuant/SmoothQuant 折叠后的 QAT 权重起步，避免二次折叠失真；Stage 2 重新折叠。

1. **P1.1 精确张量移植映射**：
   - `r1_style_mlp.0` `(32, 40→56)` 与 `style_net.0` `(64, 40→56)`：采用 P1.2 最小二乘特征桥 $W'=WA, b'=b+Wc$。
   - `core.inloop_ebt.fc2`：拆分 `fc2_old (5,96)` + `fc2_new (1,96)`，初始化 $W_5 = W_0 + \mathcal{N}(0,(0.01\sigma)^2)$、$b_5 = b_0 + 6.0$（使 $q_5/q_0 = e^{-6} = 0.25\%$ 且 $\min(E_0,E_5)=E_0$，**`ebt_ai_margin` 逐比特恒等**）。
   - `attr5_head → attr6_head`：第 5 行拷贝第 0 行，$b_5 = b_0 - 6.0$（idx 5 = `Human-Commercial-Promo`）。
   - **MoLE `8 → 10` 精确 Net2Net 分裂**：在 Wave-1 上选取对 `coupon+self_promo` 与 `soft_shill` 亲和度最高的专家 $j_1, j_2$，复制 `core_routed/core_bias/dyn_profiles/router_keys` 至 `new[0..1]`（key 加 $10^{-3}u_\perp$ 破对称），并令 $\forall t,\ \beta[t,j] \mathrel{-}= \ln 2,\ \beta[t,8/9] = \beta_{\text{old}}[t,j_{1/2}] - \ln 2$（若 $j_1=j_2$ 则减 $\ln 3$）。因 $2 e^{l+\beta-\ln 2} = e^{l+\beta}$，**初始混合核 `c_mix / b_mix / ctrl / momentum` 数学完全恒等（`diff <= 1e-5`）**！
   - `defer_r1_enc.0` `(32, 420→421)`：新增列（对应 `raw_logit_promo_r1`）零初始化。
   - **新增任务头与流式模块**：`r1_promo_head = Linear(427→8)·SiLU·Linear(8→1)`（输入 `cat(feat419, style56[48:56])`，输出层 `W=0, b=-2.0`）；`promo_head Linear(192,1)`、`promo_sub4_head Linear(192,4)`、`promo_span_head Linear(64,1)` 全零权初始化；`promo_mil_gate = -1.0`、`ebt_promo_gate = -1.5`；`chunk_emb_proj = OmniQuantLinear(256, 16, w4, g64)`；`longstream = DynaSeamMILAggregatorV11(16, 32, 2)`。
2. **P1.2 `40 → 56` 特征桥接与蒸馏**：在 20 万条精炼文本上岭回归拟合 $f_{40} \approx A f_{56} + c$（$\lambda=10^{-3}$）并折叠入两层首矩阵；Stage 1a 前 600 步仅训练两层首矩阵做预激活蒸馏 $\|L_0'(f_{56}) - L_{0,\text{R1}}(f_{40})\|^2$。**门禁 G1**：$R^2 \ge 0.97$，`Δlogit_r1 RMSE <= 0.03`。
3. **P1.3 `256-d` 精确置换（Permutation）正交防火墙**：因 `coda_proj.0` 后紧接逐元素 `SiLU`，一般正交旋转不可交换，而**坐标置换矩阵 $P$ 与 `SiLU` 严格可交换（零误差）**！
   - 在 6.4 万验证样本上计算 OBS 显著度 $s_c = w_{\text{ai},c}^2 / (\Sigma_{\text{emb}}^{-1})_{cc}$：最低 64 维移至 `[192:256]`（Promo/未来治理专有区），最高 64 维移至 `[128:192]`（AI 专有区），其余 128 维归 `[0:128]`（受控共享区）。
   - 将 $P$ 精确折叠进 `coda_proj.0`（`768 = 12×64` 量化组精确对齐）、`style_net.2`（`64 = 1×64`）的行及所有下游头列；置零 `ai_head.weight[192:256]` 并岭回归微调 `[0:192]` 补偿 OBS 残差；落实 F6 置零 `cbm_enc[0:112, 192:256]`。
   - **读取隔离与梯控**：`ai_head` 读 `[0:192]`；`promo/sub4` 读 `cat(D_{λ_sh}(emb[0:128]), emb[192:256])`（`D_λ` 复用 `tunnel_trm_ste` 零 ONNX 开销，由 Exp 3 实证校准 **$\lambda_{\text{sh}} = 0.35$（1b）/ `0.40`（B2）**）。**门禁 G0**：`Δraw_logit RMSE <= 0.02`，`max <= 0.08`，`ΔAUROC(val_gold) >= -0.0003`。
4. **P1.4 扩核后分组掩码专家平衡**：1a 冻结 `bal_bias`；1b+ 目标由 `1/E` 改为老专家 `(1-ρ_new)·load_R1(e)`、新专家 `ρ_new/2`（`ρ_new: 0 → 0.15`），仅更新 `{j1, j2, 8, 9}`；叠加路由引导损失 $L_{\text{route}} = 0.03\cdot \text{BCE}(\overline{p_8+p_9}, y_{\text{promo}})$；`router_proj` 与老 `router_keys[0:8]` 使用 `0.3×` LR。

---

### P2 · 分阶段课程与显存/算力排程（Colab A100 80GB，`BS=512`，总计 `~16.5k` 步）

| 阶段 | 步数 | 可训练参数与峰值 LR | 调度与关键配置 |
| :--- | ---: | :--- | :--- |
| **Stage 0** 外科手术与验证 | 0 | — | 执行 P1 + G0/G1；按源执行硬 `SemDeDup@0.92`（仅 `v04_confirmed_ai_plugin` & `colab_gpu_distilled`）+ 全局 `SoftDedup` $w=1/\sqrt{1+\text{deg}_{0.92}}$；缓存未增强样本的 R1 教师 `{z, z_t1, z_r1, emb[0:192]}`（`~0.5 GB`）；构建 `cos>=0.92` 连通分量防泄漏验证/认证切分（**P10-2**） |
| **Stage 1a** 冻主干冷启预热 | 1,200 | 新头 `1.5e-3`；`fc2_new/attr6_new` `1e-3`；`core_new/keys_new` `4e-4`；`longstream+chunk_proj` `1e-3`；`style L0` 前 600 步桥接 `1e-3` | WSD（60 步热身 / 尾 15% 衰减）；每 4 步插 1 步 `ChronoStream`（主干 `no_grad` 极低成本） |
| **Stage 1b** 全解冻双轴 + `ChronoStream` TBPTT | 6,000 | 主干 `6e-5`；Embedding/R1 主干/AI 头/老 EBT `0.5×`；老路由 `0.3×`；新头 `6e-4`；`core_new` `2e-4`；`longstream` `5e-4` | WSD（300 热身 / 80% 平台 / 衰减至 `0.1×`）；**L2-SP 正则向 $\theta_{\text{R1}}$ 衰减**（`wd_sp=0.02`）；每 7 步第 3 步跑 `ChronoStream`（`14.3%`，`160 docs × <=3 chunks`，`k1=M, k2=3` TBPTT）；每 200 步运行 `Balanced-PiKE/Aioli` 在线混合配比（硬下限：AI 池 `>= 0.60`，金标 `>= 0.05`，伪标 `<= 0.15`） |
| **Stage 1.5** LoGra 狙毒 + ASAM + 3 分支帕累托 Soup | 3 × 1,800 | 先跑 `~90s` **Kronecker LoGra** Top-FP 影响力狙毒（置零后 1–2% 非 T0 毒样本）；主干 `4e-5`，头 `2e-4`；900 步后开 ASAM $\rho=0.05$ | **B1 AI-Shield**（$\lambda_{\text{KD}}=0.45, \lambda_P=0.7$）；**B2 Promo-Precision**（$\lambda_P=1.3, \lambda_{\text{sh}}=0.40$，硬负 `2×`）；**B3 LongStream**（`ChronoStream 30%`，12 原型 `LongMix`）；严格受约束 `greedy_pauc_soup`（AI 退化即 `-inf` 否决） |
| **Stage 2a/2b** UPQ W4 + Sub-2 QAT ×2 | 600 + 2×1,500 | `8e-5` / `5e-5` / `3.8e-5` | 以 v1.1 FP16 Soup 为教师，新增推广双轴 ListNet + BKL 蒸馏（P5） |
| **Stage 3a/3b** FP16 减震器 + LEACE + 校准认证 | 300 + 0 | `1.5e-4` | FP16 恢复 + 双轴斜交 LEACE 零字节折叠 + 18-Cell Mondrian 与严格固定序列双轴 LTT 认证 |

- **1b 短评批配比（`BS=512`，即 `PiKE` 初始权重 $w^{(0)}$）**：`T0 钻石 32%` · `T1 洁净 34%` · `T2 软灰区 3% (×0.30)` · `Wave-1+2 双轴金标 8%`（在线 `v11_augment` $p=0.5$）· `UOT+CDMAD 伪标池 10%`（正例占比 `<= 35%`）· `LongMix 短评 5%` · `硬负强化池 4%`（含引述/否定吐槽、非商业安利硬负及 **Exp 2 `2k BGE k-NN` 硬负以 `w=1.0` 入池**，且仅 UOT 委员会确认 `p_promo <= 0.10` 者进入 `L_pceil` top-4 重锤，其余进 B2 防止误压隐蔽无词软广）· `AI 防护池 4%`（结构化真人陷阱 / `pre_llm` / `dsh`）。
- **防遗忘与梯度冲突守卫**：(1) 冻结 R1 锚定 KL；(2) 三区置换防火墙 + $D_\lambda$；(3) L2-SP 向 $\theta_{\text{R1}}$ 衰减；(4) `GradNorm-lite` 每 100 步监控共享主干梯度比 $\hat r$，硬顶 $\rho_{\max} = 0.5$（1b）/ `0.35`（1.5）；(5) 仅当连续 3 次探针余弦 `< -0.2` 时才自动触发共享主干 `PCGrad`。

---

### P3 · 联合双轴损失与长文本损失
$$\mathcal{L} = \mathcal{L}_{\text{AI}} + \lambda_{\text{KD}} \mathcal{L}_{\text{KD}} + \lambda_{P,\text{eff}}(\mathcal{L}_p + 0.06\,\mathcal{L}_{\text{pauc},p} + 0.20\,\mathcal{L}_{\text{pceil}}) + 0.15\,\mathcal{L}_{\text{sub4}} + \lambda_{\text{op}}\,\mathcal{L}_{\text{pair}\perp} + 0.10\,\mathcal{L}_{\text{ebt6}} + \mathcal{L}_{\text{ebt},m} + \mathcal{L}_\perp + \mathcal{L}_{\text{route}}$$

- **`L_AI` 增量改造**：剔除 `T3`、降权 `T2 ×0.30`；未双轴精标的商业域 `T1` 高分样本降权 `0.5×` 并移出 `pAUC` 正集；将 **`Q3` 金标真人推广纳入 `_top_fp_logit_ceiling_loss` 真人天花板**（实现商业低能谷解耦）；R1 锚定取代旧 `v0.6` 嵌入余弦项。
- **`L_KD` 选择性教师锚定**：$\tau=2$ 覆盖 `{full, t1, r1}` 与 `0.10·(1 - cos(emb[0:192]))`，对 `(a) Q3 样本`、`(b) T0 中 R1 自信出错样本 (|p_T - y| >= 0.5)`、`(c) Wave-2 救毒样本` 自动置掩码 $M_{\text{anc}}=0$，防止把 R1 的历史误杀锁死进 v1.1！
- **`L_sub4` 多标签软掩蔽**（吸纳 #5）：标注子类型给正标 `w=1.0`，其余 `Y1–Y3` 仅给软负标 `w=0.3`（兼容单标签 Schema 下的复合营销共现），`promo_score <= 25` 时 `Y1–Y3` 硬负；`Y4 tool_boilerplate` 独立建模不嵌套在商业推广下；附加层级项 $0.05\sum_{k\le 3}\text{ReLU}(s_k - u)^2$。
- **`L_ebt6` 与能量融合**：6 路部分标签 CE（未知推广真人 $S=\{0,5\}$，确认 Q4 $S=\{0\}$，Q3 $S=\{5\}$，Assisted $S=\{1\}$，AI $S=\{2,3,4\}$）；$m_{\text{ai}}=\min(E_0,E_5)-\min(E_{2:5})$、$m_{\text{pr}}=E_0-E_5$；推广原始 logit 融合 `promo_head + σ(promo_mil_gate)·promo_mil_peak + 0.5σ(ebt_promo_gate)·4tanh(m_pr/4)`。
- **`L_long`（`ChronoStream` 步）**：12 原型 `LongMix` 60% + Wave-2 长文/救毒 25% + 真实单作者长评 `>220ch` 15%；联合优化 `logit_ai_doc / hybrid / promo_doc / chunk_ai / chunk_promo / seam[1:]` + 双轴长文 Top-FP 天花板，通过 $D_\lambda=0.3$ 反传主干。

---

### P4 · 双轴安全级联与严格固定序列联合 LTT 认证
- **Rung-1 安全退出条件**：
  $$\text{exit}_1 = (\text{len}\le 96) \land (\text{spoof}=0) \land (\text{"http"}\notin x) \land (\mathbf{\text{veto}_{\text{pr}}=0}) \land (p_{\text{defer1}}\le d) \land (\hat p_{\text{ai,r1}}\le \tau_a) \land (\mathbf{\hat p_{\text{pr,r1}}\le \tau_p}) \land (\text{r1\_span\_max}\le 0.5)$$
  其中 $\text{veto}_{\text{pr}} = \mathbb{1}[\max f_{56}[\{48,49,50,51,53,54,55\}] > 0]$（零额外算力，排除 52 课代表通道）。退出即输出 `{p_ai, p_promo, quadrant=Q4}`。
- **有效联合 LTT 认证**：在独立 `path_split` 上将 `(d, τ_a, τ_p)` 网格按退出率由保守到激进排成 1-D 固定路径；在独立连通分量切分的 `cert_split` 上用交并检验 $p_j = \max(p_{\text{ai},j}, p_{\text{pr},j})$（精确二项/Bentkus，不放宽 $\alpha$），首次未拒绝即停。硬风险：$R_{\text{ai}} \le 0.5\%$ 且 $R_{\text{pr}} \le 1.0\%$（$\delta=0.05$，需在 `promo_val_cert` 留出 **`>= 600`** 条短推广正样本，Clopper-Pearson 允许最多 `1/600` 漏放，上界 `0.79%`）。
- **门禁 G8**：自然先验 R1 退出率 `>= 55%`，`ΔExit_R1 <= 3 pts`，认证漏放 Q3/Q2 `< 1%`，`Δms <= +0.03 ms`。

---

### P5 · QAT 混合精度分配、减震器与 `<= 2.20 MB` 体积杠杆
- **严禁将任何新增敏感头压至 `<= 2-bit`**（`ParetoQ NeurIPS'25` 证明极低比特会重组表示，且省不足 30 KB）：新增头、6-Way EBT、10-核 MoLE、`DynaSeam(16,32,2)` 全保留 **FP16**；仅 `chunk_emb_proj` 与 `defer_r1_enc.0` 用 **W4**；`DynaSeam ONNX` 与推广/属性头永久排除在 INT8 量化外。
- **Stage 2 QAT 增项**：`+ 0.10·ListNet(u, u_T) + 0.10·ListNet(u_r1, u_T,r1) + 0.08·BKL(u || u_T)`；Stage 3a FP16 减震器纳入 `("promo", "r1_promo", "sub4", "attr6", "longstream", "core.mole.router_keys")`，仅当 $J_{\text{ai}}$ 上升且 $J_{\text{pr}} \ge \text{pre}-0.001$ 接收。
- **体积账本**：核心新增 `~15 KB` + `DynaSeam` `33.6 KB` + `chunk_proj` `2.2 KB` + LUT/Affine `~8 KB` + 头 `~1 KB` ≈ **`+55 KB` 压缩增量**（要求 `v1.0.0-R1` zip `<= 2.145 MB`）。若吃紧按序启用四级杠杆：**L1** `longstream` 线性层转 W4 (`-14.5 KB`) → **L2** `r1_promo_head.0` 转 W4 (`-5 KB`) → **L3** 推广/长文 LUT 节点 `256→128` (`-4 KB`) → **L4** 申请放宽至 `2.25 MB`。

---

### P6 · 校准、先验漂移修正与 18-Cell Mondrian 决策层（Stage 3b）
- **先验漂移修正**：训练期推广占比高而自然评论 `< 1%`，使用加权 **Venn-ABERS**（$w_+ = \pi_{\text{nat}}/\pi_{\text{cal}}$，与 `CDMAD` 逻辑一致），区间宽度 `> 0.15` 自动标 `uncertain`；自然先验 $\pi_{\text{nat}}$ 通过 EM 在洁净 T1 野生流上估计。
- **双轴零字节斜交 LEACE 手术**（取代 F5）：在 `emb[0:192]` 上对 `ai_head` 折叠去混杂投影 $Z_{\text{ai}} = [\log\text{len}, \text{era}, \text{topic}_{32}, \hat p_{\text{pr}}]$，对 `promo_head` 折叠 $Z_{\text{pr}} = [\log\text{len}, \text{topic}_{32}, \hat p_{\text{ai}}]$（**0 B / 0.000 ms** 运行时开销）。
- **18-Cell Mondrian 决策**：原有 `9` 单元格 × `{\hat p_pr < 0.5, >= 0.5}`，同步控 `FPR_ai <= 3.5%/cell`、`Q4 硬负 promo FP <= 2%`、`P(pred AI | Q3) <= 3%`；长文（$M\ge 2$）按 $M \in \{2\text{--}3, 4\text{--}8, 9\text{--}24\}$ 分桶拟合 LUT。
- **切分卫生**：所有调参/选 Soup 仅用 `*_select`；认证仅用 `*_cert`（且按 `cos>=0.92` 连通分量整块切分防副本泄漏）；`PromoBench / LongBench` 种子卷为终验专用；`OjinjinBench` 永远由作者单次盲跑。

---

### P7 · 三级自动回滚梯队与 `V11-G0..G10` 门禁矩阵
每 500 步（1b）/ 300 步（1.5）对标 Stage 0 R1 基准（`val_gold`、`val_cal_select`、`R1 AUROC`、12 万确认真人 `top-FP max p`）：
- **`T-soft`**：`ΔAUROC_vg < -0.0015` 或 `ΔR@P95_vg < -0.008` 或 `ΔR@P95_wild < -0.010` 或 `Δtop-FP > +0.03` → 自动 $\lambda_{\text{KD}}\times 1.5$（顶 `1.2`）、$\rho_{\max}\times 0.7$、主干 LR $\times 0.7$、$\lambda_{\text{sh}}\times 0.5$。
- **`T-hard`**：连续 2 次 soft 或 `ΔAUROC < -0.004` 或 `R1 AUROC -0.006` → 回滚至最近通过的 checkpoint，主干 LR $\times 0.5$，冻结 R1 主干与 Embedding，$\lambda_{\text{sh}}=0$。
- **`T-abort`**：3 次 hard 或 1b 结束 `R@P95_vg < R1 - 0.02` → 一键切换 **`v1.1-Lite`** 兜底方案（AI 路径完全锁定 R1 原值，仅在冻结特征上训练推广头与 DynaSeam 后再做 QAT）。

| 门禁 | 检验阶段 | 硬性通过标准 |
| :--- | :--- | :--- |
| **G0 / G1** | 外科手术 / 桥接 | `Δlogit RMSE <= 0.02`，`MoLE diff <= 1e-5`，`Zero-PII = 0`，`M=1` 逐比特恒等；桥接 $R^2 \ge 0.97$ |
| **G2 / G5** | 1a 末 / 专家健康 | AI `val_obj Δ >= -0.0005`，`promo AUROC >= 0.90`；E9/E10 负载 `∈ [3%, 25%]`，老专家 L1 漂移 `<= 0.10`，推广亲和度 `>= 2×` |
| **G6 / G7** | Soup / QAT | 严格受约束 Soup（AI 不退化）；FP16→Sub2 `ΔAUROC <= 0.002`，`ΔR@P95_promo >= -0.01`，R1 推广召回掉点 `<= 0.5 pt` |
| **G8 / G9** | 级联 / 长文 | R1 退出 `>= 55%`，`ΔExit <= 3 pts`，认证漏放 Q3/Q2 `< 1%`，`Δms <= +0.03`；夹心/隐藏尾广/全篇套引号拦截均 `>= 90%`，裸引用 FP `<= 2%` |
| **G10** | 最终发布 | `test_bili / test_dsh / OjinjinBench v1&v2 >= R1`；`PromoBench AUROC >= 0.96 / R@P95 >= 75% / FP <= 2% / 4Q >= 90%`；`<= 2.20 MB`；`PII = 0` |

---

### P8 · `v1.2.0+` 零遗忘能力插槽协议（`CSP-1`）
- **Tier A（`< 20` 分钟，结构级绝对零遗忘）**：一次性缓存全语料 FP16 `{emb_t2(256), r1_pkt(162), style56}`（`~1.25 GB`）；每扩展一项新能力 $k$（反诈、恶意引战、刷屏复读等），在缓存上数分钟训练 `Adapter A_k (256→16, W4 2.1 KB)` + 任务头 + R1 门控 `Linear(218→1, 0.4 KB)`，重做联合 LTT 并将 `cap.<id>.*` 追加进 MGVZ6 与 manifest。运行时仅对现有 ONNX 输出 `emb_t1 / emb_r3 / r1_pkt` 做后置 GEMV，**无需重新导出主干 ONNX，AI 与推广分逐比特差 `0.000000`**！
- **当前即刻预留**：CBM 概念槽 `112..127`、ABI 标签 `emb_layout = v11-perm-<sha8>`、单能力硬预算 `<= 8 KB` & `<= 1 pt` 退出率损失。
- **Tier B（深度新能力）**：按 P1.1 Net2Net 分裂挂载 `E11` 专家 + 双轴 KL 锚定 + 完整 Stage 2 重做 QAT。

---

## 3. 待决策事项（3 项轻量拍板）

1. **`v1.0.0-R1` 成品 zip 体积余量**：待 Colab A100 `v1.0.0-R1` 产出后确认其是否 `<= 2.145 MB`（若略超则按 P5 顺次启用 L1–L3 零损压缩杠杆，共可再省 `23.5 KB`）。
2. **Rung-1 推广漏放证书 $\alpha_{\text{pr}} = 1\%$**：认证严格要求 `promo_val_cert` 独立留出 **`>= 600`** 条短推广正样本（需从 Wave-1 + Wave-2 + UOT 收割池按连通分量整块划出）。
3. **6-Way 属性顺序确认**：采纳 **F1** 建议，将 `Human-Commercial-Promo` 追加至 **idx 5**（而非插在 idx 2），100% 保护 `v1.0.0` `ATTRIBUTION_5WAY` 与 EBT 索引兼容。
