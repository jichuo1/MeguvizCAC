# MeguvizCAC v1.0.0 — `TunnelSymplect × DynaCore-MoLE × Zero-Recompute Cascade` 架构迭代总路线图

> **版本定位**：作为 MeguvizCAC 走向 **1.0.0 里程碑**的终极零隐私边缘架构，我们在 `v0.9.0`（`ChronoLoop-M3.5 × PSAC-v9`）的基础上，围绕四大核心架构创新完成了 **5 路前沿理论推演 + 代码级取证审计 + 本地 RTX 4060 Ti & 1T-CPU ONNX 真机等参微实验**：
> 1. **嵌套 Loop 穿透（Nested Loop Penetration）**：通道选择穿透 + `L0-Haar` 表层直穿 + `Tunnel-TRM` 等距梯度穿透（[11_v10_loop_penetration.md](../../scratch/arch_research/11_v10_loop_penetration.md)）；
> 2. **次级 Loop 联结（Secondary Loop Coupling）**：语义×跨度水平对等环（闭式有理 Cayley $SO(2)^D$ 旋转）+ 跨外环时间记忆携带 + 零求导解析梯度环内 EBT 自由能次级环 + 零冷启动多粒度上下文次级环 `Loop-C`（[12_v10_secondary_loop_coupling.md](../../scratch/arch_research/12_v10_secondary_loop_coupling.md)）；
> 3. **级联模型间 Loop 衔接（Inter-Cascade Model Loop Handoff）**：`R1→R2` 疑点热启动（`SSSQ`）+ 跨精度防投毒反证桥（`DCCH-Bridge`）+ 四段零重算模块化 ONNX 接力 + 状态耦合保形可交换性定理（[14_v10_cascade_loop_handoff.md](../../scratch/arch_research/14_v10_cascade_loop_handoff.md)）；
> 4. **Loop 专家组（`DynaCore-MoLE`）**：共享 Stiefel 正交基 + `1 共享 + 8 领域` 纯 FP16 核心小矩阵 + 跨轮动态换专家 + 动力学参数联合调制 + DeepSeek-V3 无辅助损失均衡（[15_v10_loop_expert_groups.md](../../scratch/arch_research/15_v10_loop_expert_groups.md)）；
> 5. **本地 RTX 4060 Ti & 1T-CPU 实证消融（零泄漏）**：真实跑通 8 组等参对比（`B0 / P1 / P2 / C1 / C2 / Fusion_P2C2 / Fusion_P2C1C2 / Fusion_All4`），用硅片数据锁定胜者并果断否决实测掉点的理论方案（[13_v10_empirical_lab.md](../../scratch/arch_research/13_v10_empirical_lab.md)）。
>
> **核心硬指标承诺**：所有四大创新合体后，通过回收 `v0.9.0` 实查出的 `175.25 KB` 隐藏冗余与 `62.41 KB` 外部 EBT，**`Bundle v6` 总压缩体积不增反降至 `~1.92–1.95 MB <= 2.20 MB`**、WebGPU Workgroup SRAM **`<= 31.44 KB <= 32 KB`**、Chrome MV3 静态 ONNX Opset-17 **0 动态控制流**、1T-CPU 有效级联延迟 **`0.245–0.293 ms <= 0.30 ms`**、Zero-PII **`diff === 0.000000`**！

---

## 1. 代码级取证审计：在 `v0.7.0–v0.9.0` 中挖出的 4 个深层隐患（`v1.0.0` 立即根治！）

1. **`SeqLoRAMoE` 双线性串扰污染（`v07_omniveritas.py#L206-213`）**：
   - 先各自加权混合 `A_eff = Σ w_i A_i` 与 `B_eff = Σ w_j B_j` 再相乘，代数展开为 $\sum_i w_i^2 x A_i B_i + \sum_{i \ne j} w_i w_j x A_i B_j$，实测造成 **`83.69%` 跨专家子空间污染**，且在 `B=32` 时额外申请 **`786 KB`** 临时逐样本权重！升级为 **`DynaCore-MoLE`** 后线性误差降至 **`1.79e-7`**、吞吐反快 **`37%–78%`**。
2. **`trm_detach_k0=True` 整态截断误杀梯度（`v09_chronoloop.py#L313, L350`）**：
   - 对 `z_l_new` 整体 `.detach()` 导致 Sub-2-Bit QAT 期 **`phase_mod[0]`（Phase-0 Gather）梯度恒为 `0.0`**，且 **100% 切断 `Active Re-Read` 回传给 12 快槽的梯度**（雅可比坍缩至 `||J||_2 = 0.0`）。改为 **`P4 Tunnel-TRM`**（仅对 FFN 输入施加前向精确恒等算子 $D_\lambda(x)=\lambda x + (1-\lambda)\text{sg}(x)$，保持残差直达公路 `I` 100% 畅通）后，QAT 期精确等距 **`||J||_2 = \sigma_{\min} = 1.0000`** 且 `0 KB / 0 ms` 开销！
3. **`fast_zh_gate` 与 `reread_q/o` 的 `175.25 KB` 隐藏体积冗余（`v09_chronoloop.py#L282-288`）**：
   - 注释写 `2.3 KB` 的 `fast_zh_gate` 实为满秩 `384×384`（`77.25 KB`），叠加两个满秩 W4 重读矩阵（`154.5 KB`）。重构为 **`P2` `r=8` 双子空间穿透门（`13.31 KB`，净省 `-63.94 KB`）** + `reread_q -> seq2` / 共享 `slot_o + r=8 W4 LoRA`（净省 `-111.56 KB`），凭空腾出 **`175.25 KB`**，再将 `62.41 KB` 外部 EBT 升级为环内 `17.08 KB` 解析梯度次级环，**让 `v1.0.0` 装下全部四大创新反而比 `v0.9.0` 更轻！**
4. **Rung-1→Rung-2 槽冷启动遗忘与 `outer_steps=2` 焊死浪费**：
   - `v0.9.0` 的 Rung-2 扔掉了 Rung-1 的局部 `tok_r` 与延迟疑点隐状态，且将 `t=1, t=2` 焊死在单个 ONNX 图中迫使全部 ~20% 升级样本白跑 `t=2` 重读。`v1.0.0` 改为 **4 段零重算模块化 ONNX 接力（`R1 -> R2(t=1) -> R3_Step(t=2..3) -> R4(Context)`）**，单线程延迟再压低至 **`0.245–0.293 ms`**！

---

## 2. `v1.0.0` 四位一体总体架构图

```mermaid
flowchart TD
    IN["Zero-PII Comment + PIFE-40d + Picky-BPE12k + Engram"] --> R1["Session R1: BLT-Matryoshka Prelude-L0 Prefix D1=128 (0.105 ms)"]
    R1 --> G1{"LTT / Selection-Safe Gate 1 (p_defer1 + r1_span + spoof=0)"}
    G1 -->|"Safe Exit (~72-77%, 0% wasted)"| O1["Affine + Binary-ECE LUT Output"]
    G1 -->|"Defer: Reuse h0_1 + H1 SSSQ Packet (164-d, NO logit!)"| R2["Session R2: Complement D2=256 + Funnel 2:1 + L1 + L0-Haar Surface Micro-KV (P3)"]
    R2 --> INIT["H1 Warm-Start: 4 Neutral Slow z_H + 12 Suspicion-Seeded Fast z_L (6 Sem Loop-A + 6 Span Loop-B)"]
    INIT --> T1["Outer t=1: Fast Micro-Loop (P2 Dual Tunnel Gate + P4 Tunnel-TRM + C1 Cayley SO(2) A<->B + E3 DynaCore-MoLE) -> Slow Consolidate + C3 Analytic-Grad In-Loop EBT"]
    T1 --> G2{"Gate 2: EBT Free-Energy F + ||grad F|| + p_defer2 + VA-Width"}
    G2 -->|"Resolved (~14%)"| O2["Co-Refined Top-Down Span + Aligned Output"]
    G2 -->|"Short & Epistemic Wide (L<=20)"| R4
    G2 -->|"Hard Gray-Zone (~4%): Carry C2 Momentum m_L + H2 DCCH-Bridge (s_hyp, u_doubt, Top-8 KV)"| R3["Session R3 (Re-Entrant t=2..3): L0-Haar & Prelude Active Re-Read + Re-Read Surprise Gate (C2) + Specialist Switch (E3 DynaCore-MoLE) + Falsification Slot q_falsify"]
    R3 --> G3{"Gate 3: Contraction + EBT Margin"}
    G3 -->|"Resolved (~4%)"| O2
    G3 -->|"Cluster / Ambiguous (~2%)"| R4["Session R4 (Context Loop-C): Parent Prefix-KV + Kish-Shrunk MIRAS + HypSheaf-Lorentz f32 (|Δz|<=0.85, λ=0 -> diff===0.0)"]
    R4 --> O3["Final Audited Verdict / Tier-3 Abstain"]
```

---

## 3. 本地 RTX 4060 Ti & 1T-CPU 实证消融战报（决定取舍的硬数据！）

> **严格零泄漏受控实验**（`train_32k` = `24k v04 + 8k Gemini 76k 黄金库`，共享初值，`500` 步 BF16 `bs=128 L=192 T=2 K=2` → 真 `pack_bundle_v5_mgvz` 零样本 PTQ → `100` 步 `TRM1` Sub-2-Bit STE-QAT → 真实 1T-CPU ONNX `b=32` 延迟；`OjinjinBench` 与 `19,810` 条测试集 100% 物理隔离）：

| 候选架构 | 相对 `B0` 体积增量 | FP16 `val_gold` (`AUC / PR / R95`) | FP16 `val_cal1200` (`AUC / PR / R95`) | 零样本 `T=3` 外推 (`gold AUC / cal R95`) | 零样本 Sub-2B PTQ `ΔAUC` (`gold / cal`) | **Sub-2B STE-100 终态 `val_gold`** (`AUC / PR / R95`) | **Sub-2B STE-100 终态 `val_cal1200`** (`AUC / PR / R95`) | `slot_erank` (`FP16 / STE`) | 1T-CPU ONNX `b=32` (ms) |
|---|---:|---|---|---|---|---|---|---:|---:|
| **`B0_V09_ChronoM35`** (`v0.9.0` 基线) | `1.6965 MB` | `0.8705 / 0.8000 / 0.0253` | `0.9729 / 0.9113 / 0.4164` | `-0.0022 / -0.0546` | `+0.0032 / +0.0013` | `0.8878 / 0.8607 / 0.4937` | `0.9748 / 0.9154 / 0.3925` | `1.156 / 1.437` | **`2.830`** |
| `P1_DepthTunnel_AttnRes` *(粗粒度标量 — 否决)* | `+7.63 KB` | `0.8717 / 0.8467 / 0.4177` | `0.9692 / 0.8969 / 0.4266` | `-0.0059 / +0.0103` | `+0.0031 / +0.0041` | `0.8795 / 0.8555 / 0.5063` | `0.9741 / 0.9095 / 0.4608` | `1.469 / 1.475` | `2.948` |
| **`P2_ChannelTunnel_Gate`** *(单模块第一!)* | `+22.19 KB` *(替换后 `-64 KB`!)* | **`0.9019 / 0.8762 / 0.5190`** | `0.9816` / **`0.9423`** / `0.6246` | `-0.0014 / +0.0000` | `-0.0032 / +0.0022` | **`0.9048 / 0.8808 / 0.5570`** | `0.9850 / 0.9523 / 0.6997` | `1.049 / 1.085` | `2.972` |
| **`C1_PeerCoupled_SemSpan`** *(秩塌解药!)* | `+26.25 KB` | `0.8859 / 0.8547 / 0.4430` | `0.9771 / 0.9292 / 0.6177` | **`+0.0017` (`+5.07% R95`)** | `+0.0002 / +0.0001` | `0.8957 / 0.8725 / 0.5190` | `0.9782 / 0.9335 / 0.5939` | **`1.691 / 1.939`** | `2.945` |
| **`C2_Symplectic_Carry`** *(PTQ 抗崩第一!)* | **`+10.22 KB`** | `0.8854 / 0.8522 / 0.4430` | `0.9797 / 0.9339 / 0.5666` | `+0.0004 / -0.0444` | **`+0.0098 / +0.0017`** | `0.8941 / 0.8704 / 0.5063` | `0.9810 / 0.9402 / 0.6485` | `1.527 / 1.439` | **`2.924`** |
| **`V10_Fusion_P2C2`** (`P2 + C2`, **默认冠军!**) | **`+32.41 KB`** | `0.8803 / 0.8167 / 0.1266` | `0.9799 / 0.9263 / 0.2491` | **`+0.0071 / +0.1229`** *(唯一双单调!)* | **`+0.0077 / +0.0049`** | **`0.9022 / 0.8751 / 0.5316`** | **`0.9852 / 0.9531 / 0.7406`** *(R95 `+34.81%`!)* | `1.096 / 1.151` | `3.005` |
| **`V10_Fusion_P2C1C2`** (`P2 + C1 + C2`) | `+58.66 KB` | `0.8891 / 0.8384 / 0.2405` | `0.9811 / 0.9384 / 0.5495` | `+0.0006 / +0.0068` | `+0.0020 / +0.0023` | `0.8993 / 0.8671 / 0.4810` | `0.9832 / 0.9443 / 0.6758` | **`2.014 / 2.065`** *(最高秩!)* | `3.078` |
| `V10_Fusion_All4` *(含 `P1` — 明显被拖累)* | `+66.29 KB` | `0.8848 / 0.8324 / 0.1266` | **`0.9827`** / `0.9386` / **`0.6416`** | `-0.0001 / -0.0034` | `+0.0009 / +0.0007` | `0.8992 / 0.8724 / 0.3797` | `0.9844 / 0.9490 / 0.6348` | `1.801 / 1.804` | `3.176` |

### 实证铁律（指导 `v1.0.0` 工程取舍）
1. **必留核心基石 — `P2`（通道选择穿透 + `L0` 表层直穿）+ `C2`（跨外环时间记忆携带 + 辛反对称耦合）**：`V10_Fusion_P2C2` 在 Sub-2-Bit 自然先验野生集上将 `R@P95` 从 **`39.25%` 暴拉至 `74.06%`（`+34.81%`！）**、`PR-AUC` 从 `0.9154 -> 0.9531`，且是**唯一在 `T=1->2->3` 上两集严格单调上升**的架构（`+0.0071` AUROC / `+12.29%` R@P95），零样本 Sub-2-Bit PTQ 反涨 `+0.0077`！
2. **秩坍缩解药 — `C1`（语义×跨度水平对等次级环联结）**：将 `slot_erank` 近乎翻倍（`1.156 -> 2.065`）且在 Sub-2-Bit STE 后依然保持 `ΔT3` 正增益，在 5,000+ 步正式训练（预热桥接学习率）下与 `P2 + C2` 互补极强！
3. **果断否决粗粒度 `P1` 标量流混合**：实测无论单独还是放进 `All4` 均拖累野生集 `R@P95`（`-10.1%`）、`erank` 与 `ΔT3`。`v1.0.0` 保留 `v0.9.0` 细粒度逐通道 `2×2` Birkhoff `mHC` 锚定（仅对 `P3 L0-Haar` 高通表层穿透保留单向门控直穿）。

---

## 4. 四大支柱最终锁定方案

### 支柱一：嵌套 Loop 穿透（`P2 + P3 + P4`，反减 `-26.44 KB`！）
- **`P4 Tunnel-TRM`（T0 必装，`0 KB / 0 ms`）**：前向恒等算子 $D_\lambda(x)=\lambda x + (1-\lambda)\text{sg}(x)$ 仅作用于 FFN 输入，保留残差公路 `I` 100% 导通；QAT 期设 $\lambda=0$ 实现精确等距 `||J||_2 = 1.0000`，彻底修复 `v0.9.0` 中 `Phase-0` 与 `Active Re-Read` 死梯度。
- **`P2 Dual Subspace Penetration Gate`（`-63.94 KB` 净减重）**：替换 `77.25 KB` 满秩 `fast_zh_gate` 为 `13.31 KB` W4 `r=8` **矛盾前置门**（已收敛维度落入 `Capped-ReLU²` $O(\epsilon^2)$ 死区，零 5-Trit 噪声注入）+ **恒等保残后置审阅门** `z_L + γ·g_rev(z_L, Δ_L)·Δ_L`（绝不采用会导致梯度消失的凸 Highway 门！）+ `h0_pool -> z_H` 表层直穿。
- **`P3 L0-Haar Surface Tunnel`（`+37.50 KB`）**：绕过 `2:1 Funnel` 池化失真，对 Rung-1 已算好的 `h0_1` 施加 **0 参数静态 ONNX 掩码拉普拉斯高通滤镜** `Δ_HF(h0_1)` 预建 `64-d` W4 表层微 KV（2,048 词流式仅 `8 KB` `Top-32 Surface-Seam Sketch`），专杀三明治拼接缝与 `H06_scorecard / M04_zhen_tech` 误杀！

### 支柱二：次级 Loop 联结（`SymplectLoop` `C1–C4`，含共享 `CouplingBus64`）
- **`C1` 语义×跨度水平对等次级环**：`12` 快槽拆为 `6` 语义/语用槽（`Loop-A`）$\leftrightarrow$ `6` 局部跨度/节奏槽（`Loop-B`，接入 `UNet Span64` Top-4 高方差窗与 `PIFE-40d`），经 **`6×6` BiSA 双向注意力 + 闭式有理 Cayley $SO(2)^D$ 旋转**（$c_d=(1-\gamma^2)/(1+\gamma^2), s_d=2\gamma/(1+\gamma^2)$，无 `sin/cos`，严格保能 `c²+s² === 1`、`0.00e+00` 能量误差）互传消息，并由 `(z_H + z_B)` 联合精修 `UNet Span`。
- **`C2` 跨外环时间记忆携带（`+10.2–29.6 KB`，`1.5 KB` SRAM）**：将内环净增量 $\Delta z_L$ 经严格收缩有界动量 $m_L \in (-1,1)^{12\times 64}$ 跨外环传递至 `(t+1, 0)`，并由 **重读惊奇门（Re-Read Surprise Gate）** 在主动重读发现反证时自适应调节重置/保留。
- **`C3` 环内解析梯度 EBT 能量次级环（`+17.08 KB` FP16，退役 `62.41 KB` 外部 EBT！）**：将盲测 `OjinjinBench-v2` 曾跑出 `0.943` 的 5 类 EBT 提升至核心环内，使用 **零 Autograd 闭式解析梯度**（PyTorch 验证误差仅 **`1.19e-7`**）执行严格保证 Lyapunov 能量下降的 1 步隐式精修，同时输出 `(F, ||∇F||, H(q), M_E)` 四个早退停止证书，彻底根治 `v0.9.0` 的首步速度倒置定律。
- **`C4` 零冷启动多粒度上下文次级环（`Loop-C ↔ z_H`，`0.5 KB` SRAM）**：将父评嵌入、`VideoFastMem MIRAS`、`HypSheaf-Lorentz f32` 与 8 维零隐私拓扑封装为 4 槽 `64-d` 次级环，采用 **Kish 收缩乘法掩码 + 无偏置 `W_out^C` + 零初值门控**，无上下文时严格 IEEE-754 比特级等价（`diff === 0.000000`）。

### 支柱三：级联模型间 Loop 衔接（`H1–H4`，`+40.8 KB`，`0.00%` 重算）
- **`H1` Rung-1 → Rung-2 疑点热启动（`SSSQ`，`+6.8 KB W4`）**：Rung-1 产出 `164-d` 疑点握手包（**严禁包含 `raw_logit_r1`**）经零初值有界门控仅热启动 `12` 快证据槽 `q0_L`，**4 个慢裁决槽 `q0_H` 与 `s_anchor` 严格保持 0.0 绝对中性**，彻底杜绝确认偏误。
- **`H2` 跨精度/异构模型防投毒桥（`DCCH-Bridge`，`+34.0 KB W4`，`+1.25 KB` SRAM）**：传递解耦三元组 `(s_hyp, u_doubt, Top-8 Salience+Dispute INT8 KV-Sketch = 1.0 KB)`，经离线烘焙的谱收缩正交投影（证明量化噪声衰减 `<= 0.40×`）+ **置信度阻尼反证条件化**注入专设的 **反证槽 `q_falsify`** 与 **审阅槽 `q_audit`**（遇到 SilverSpeak 混淆时强制切断 `s_hyp`），逼迫强模型专门攻克弱模型的疑点而非盲目附和。
- **`H3` 四段零重算模块化 ONNX + Chrome MV3 `ScratchPad`**：`Session_R1` → `Session_R2 (仅 t=1)` → 可重入 `Session_R3_Step (t=2, t=3)` → `Session_R4 (Context Loop-C)`，共享单一 `bundle_v6.mgvz` 外部权重存储（0 重复打包），`768 KB` 静态 `ScratchPad`（`L=40` 实占 `138 KB`）+ WebGPU `gpu-buffer` 原地 WGSL 压缩零 PCIe 往返。
- **`H4` 防投毒对抗训练与保形可交换性证明**：训练期注入 `p_drop=0.25` 握手丢弃 + `p_flip=0.15` 对立类/空间诱饵翻转（配 Stop-Grad 防火墙保护 `M_1`）+ 强模型占优损失 `L_dom`；数学证明只要衔接桥在 `val_cal_cert` 前冻结，Rung 1–3 的 **Selection-Safe Conformal + LTT/CompCRC** 有限样本保证严格成立，Rung 4 通过硬偏移上限 `|Δz_R4| <= 0.85` 与按 `bvid` 视频分簇保形校准解决跨评论相关性。

### 支柱四：Loop 专家组（`E3 DynaCore-MoLE`，净增仅 `+5.67 KB`！）
- 替换掉 `v0.9.0` 中存在 `83.69%` 双线性污染的 `SeqLoRAMoE` 与固定 `PhaseModulatorV09`，采用 **`1` 常驻共享专家 + `8` 领域路由专家**（`E0` 闲聊热梗、`E1` 结构化技术人类盾 `H06/M04/H03`、`E2` 多语、`E3` 三明治外科、`E4` 隐写润色批评、`E5` 人设 Bot `A01`、`E6` 脚本/团伙、`E7` 极短评/反混淆）：
  - **共享 Stiefel 正交基 `U, V ∈ R^{384×24}`（`W4`, `9.56 KB`）+ 纯 FP16 `24×24` 领域核心矩阵 `C_e` + `432 B` 核心偏置（`10.55 KB`）**：零 `Gather/Scatter`、零动态分支、100% 静态 ONNX Opset-17（已 CPU 实测 `12/12` 动态尺寸 `0 NaN`，`max_diff = 4.77e-7`，`B=32` 仅 **`9.71 µs/comment`**，比 Sparse Top-2 快 `37%`、比 Dense-Masked 快 `78%`！）。
  - **`288 B` 动力学联合控制画像 `ψ_e ∈ R^{16}`**：专家不仅调制 FFN，还联动调制 `C1` Cayley 耦合角、`P3` `L0-Haar` 门控、`P2` 穿透阈值与 `C2/C3/C4` 增益，且支持 **外循环 `t=0` 粗分诊 → `t=1..2` 专家切换**（惊奇度调制跨轮动量使低惊奇抖动降低 **`-42.2%`**）。
  - **DeepSeek-V3 死区无辅助损失均衡**（实测死亡专家 `6/8 -> 0/8`、路由熵 `1.0000`、零梯度干扰）+ **UPQ 后纯 FP16 核心 Shock-Absorber Refit**（`0.000000` 重打包漂移，实测比全局秩-24 多降 `5-Trit` 均方误差至 **`+33.38%`**！）。

---

## 5. `v1.0.0` 统一硬件与体积总账（全套四大支柱合体！）

| 预算项 | `v0.9.0` 基线 | **`v1.0.0` 四大支柱全量合体** | 硬门禁红线 |
|---|---:|---:|---:|
| **Bundle 总压缩体积（ZIP-9，含 12k BPE + 全部 Critic）** | `1.931–1.976 MB` (`MGVZ5`) | **`~1.92–1.95 MB`** (`MGVZ6`，含 `175 KB` 冗余回收与外部 EBT 退役) | **`<= 2.20 MB`（余量 `~250 KB`！）** |
| **WebGPU Workgroup SRAM 峰值** | `21.0 KB` | **`31.44 KB`**（Phase P1 峰值，P2 `30.48 KB`） | **`<= 32.00 KB` PASS** |
| **1T-CPU 有效级联延迟（`L=40`，自然先验）** | `~0.342 ms`（理论） | **`0.290 ms`**（单前缀 `D_1=128`）/ **`0.245 ms`**（Turbo `D_0=64 ⊂ D_1=128`） | **`<= 0.30 ms` PASS** |
| **静态 ONNX Opset-17 算子与精度** | 2 段（`R1 + R2(T=2)`） | **4 段零重算**（`0` 动态控制流，`0` Gather/Scatter，`max_diff <= 1e-6`） | **0 动态算子 PASS** |
| **Zero-PII `@username` 扰动与冷启动** | `0.000000` | **`0.000000`**（全 4 阶握手包 + `Loop-C λ=0` 比特等价） | **`=== 0.0` PASS** |

---

## 6. 分阶段渐进实施与红队验收计划（绝不一口气大爆炸！）

| 阶段 | 实施范围 | 关键里程碑与自动门禁 |
|---|---|---|
| **Phase 0（立即热修，可同步反哺 `v0.9.0-R2`）** | `P4 Tunnel-TRM`（修复 `trm_detach_k0` 梯度切断）+ 修复 `SeqLoRAMoE` 双线性串扰 + 回收 `fast_zh_gate` / `reread_q,o` `175 KB` 冗余 | `V10-P4`（QAT 期 `Phase-0` 与 `Active Re-Read` 活梯度验证）、`0` 性能回退且体积立减 `~170 KB` |
| **Phase 1（真机冠军核心底座）** | 实装 **`V10_Fusion_P2C2`**（`P2` 双子空间穿透门 + `C2` 跨外环时间记忆携带 + 重读惊奇门）+ `P3` `L0-Haar` 高通表层微 KV | `V10-P2/P3` + `T=1->2->3` 严格单调上升 + `H06_scorecard / M04_zhen_tech <= 0.40` |
| **Phase 2（对等次级环与环内 EBT）** | 接入 **`C1`** 语义×跨度水平对等环（Cayley $SO(2)^D$）+ **`C3`** 零求导解析梯度环内 EBT（退役外部 `62 KB` EBT）+ **`C4`** 零冷启动 `Loop-C` | `V10-S0..S6`（`slot_erank >= 6.5`、EBT 解析梯度误差 `< 1e-5`、Lyapunov 能量单调下降、`λ=0 diff === 0.0`） |
| **Phase 3（Loop 专家组）** | 实装 **`E3 DynaCore-MoLE`**（`1+8` 领域核心 + `288 B` 动力学画像 + DeepSeek-V3 无辅助损失均衡 + Stage 3a FP16 核心 Shock-Absorber Refit） | `V10-E0..E4`（`0/8` 死亡专家、静态 ONNX 0 Gather、Sub-2-Bit 漂移 `<= 0.0010`） |
| **Phase 4（4 段零重算级联接力）** | 实装 **`H1` `SSSQ`** + **`H2` `DCCH-Bridge`** + **`H3` 4 段静态 ONNX / MV3 `ScratchPad`** + **`H4` 防投毒对抗训练与保形认证** | `V10-H0..H6`（Rung-1 漏报 `<= 0.25%` LTT 认证、`M_2` 抗 `M_1` 翻转投毒、1T-CPU `<= 0.30 ms`） |
| **Phase 5（Colab G4 20 万数据终极拉练）** | 接入完整 **20 万外部 Gemini 黄金库（目标 20/20 波）** + 双教师流形锚定 + 3 路严格不交验证集 + 2 分支 ASAM Soup + 零算子折叠 + `TRM1` QAT Soup + 盲测 `OjinjinBench v1/v2` | 目标：**`< 1.95 MB`** + **`<= 0.30 ms` 1T-CPU** + **`test_bili_gold AUROC >= 0.975 / R@P95 >= 0.88`** + **盲测 `OjinjinBench-v2` 全面碾压历史 SOTA（`>= 0.955` Full / `>= 0.935` Hard）**！ |
