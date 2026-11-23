# MeguvizCAC Architecture & Data Evolution Roadmap (v0.3-Loop & v0.4-DualLoop)

This document locks the architectural, training, and corpus-scaling specifications for **MeguvizCAC v0.3-Loop** and **MeguvizCAC v0.4-DualLoop**, following the baseline `v0.1` release and Stage-2 `Qwen3.8-27B` + `Gemini 4 Hydrogen (XHigh)` distillation (`v0.2`).

---

## 1. Locked Specification: `MeguvizCAC v0.3-Loop`

Design goal: shrink ONNX footprint by ~45%–55%, eliminate `[UNK]` out-of-vocabulary degradation across all social platforms, and replace fixed shallow 4-layer inference with adaptive depth (`3–9` effective layers) at equal or lower CPU latency.

### 1.1 Extensible Zero-`[UNK]` Compact Embedding
- **Keep 100% of the full pretrained vocabulary (`21,128` WordPiece tokens)** rather than hard-pruning to Bilibili-only tokens, preserving cross-platform generalization (Zhihu, Xiaohongshu, Weibo, Tieba, code/English-mixed tech discussions).
- **Multi-domain weighted Truncated SVD factorization (`V × E + E × H`)**:
  - Factorize the pretrained `21,128 × 512` matrix (`10.82M` params) into `(21,128 + 256) × E` + `E × 512`, using multi-domain general frequency + uniform floor weights during initialization so rare technical/literary tokens stay well-conditioned.
  - Primary candidates: `E=128` (~`2.80 MB` INT8) and `E=192` (~`4.18 MB` INT8).
- **`+256` UTF-8 Byte-Fallback slots**:
  - Append 256 byte-fallback embeddings (~`32–48 KB`) so rare CJK characters, kaomoji `(˘ω˘)`, new emojis, dialect slang, and symbols never collapse to `[UNK]`.

### 1.2 Backbone: `Prelude + RRT Core Loop + Ouro Early Exit`
- **Huginn-style three-stage topology `(Prelude, Core, Coda)`**:
  - **Prelude (`1` untied layer, ~`3.15 MB`)**: runs once per sequence to extract surface/lexicon/kaomoji anchors $h_0$.
  - **Recurrent Core (`Cycle` weight sharing)**:
    - Primary target **`1 + 2×T`**: `2` shared physical Transformer layers cycled $T \in \{1, 2, 3\}$ times in normal mode (**`3 / 5 / 7` effective layers**) and $T=4$ in audit mode (**`9` effective layers**).
    - Ultralight ablation **`1 + 1×T`**: `1` shared physical layer cycled $T \in \{1..5\}$ times (**`2..6` effective layers**).
  - **Coda**: lightweight `[CLS] + MeanPool` classification, uncertainty estimation, 8-tag pragmatic head, and exit gate.
- **Loop Specialization & Stability Mechanisms**:
  - **Depth-wise LoRA + Truncated SVD initialization** (Relaxed Recursive Transformers, ICLR 2025): each loop step $t$ has lightweight step-specific LoRA (`r=16/32`, ~`0.88 MB` total) + independent LayerNorm/RMSNorm, initialized via Truncated SVD of residual differences $(W_\ell - W')$ from `v0.2` / Stage-1 12L teachers.
  - **Gated Input Injection**: $z_t = \text{Core}_t(\text{LN}(z_{t-1} + \gamma_t \odot h_0))$ to prevent forgetting surface clues (`本鱼`, em-dashes, version tags) over deep iterations.
  - **QK-Norm**: normalize Query and Key vectors before dot-product attention inside the shared Core to prevent softmax collapse across loops.
  - **Ouro entropy-regularized adaptive early exit + progressive pragmatic supervision**: supervise all loop exits with an Ouro uniform-prior entropy regularizer; weight surface signals (`assistant_leak`, `community_slang`) stronger on early loops and deep pragmatic signals (`irony_parody`, `lyrical_emotion`, `external_personal_exp`) + minimal pairs stronger on deeper loops.

### 1.3 Deployment: Zero-Duplication Dual-Graph ONNX + INT8 QAT
- **Dual static ONNX graphs** (zero duplicated core weights on disk; no dynamic `If`/`Loop` control-flow inside ORT graphs):
  1. `prelude.onnx` (full-vocab SVD embedding + byte fallback + Prelude layer).
  2. `core_step.onnx` (single shared Core block parameterized by `step_id` LoRA/Norm index + exit head).
- **Host-side compact-batch engine**:
  - Length-sorted micro-batching (`<=32 / <=64 / <=128 / <=192`) to eliminate padding waste.
  - Run `prelude` + `core_step(0)` on full batch -> pop ~65%–70% high-confidence easy comments -> pack remaining ~30% hard samples into compact contiguous tensors for `core_step(1..2)` (optional `3` in audit mode).
- **Quantization**:
  - **Main package (`v0.3-CPU-INT8`)**: INT8 QAT (`~11.5–14.5 MB` ONNX / `~7–8.5 MB` zip, estimated `~1.5–2.3 ms/comment` avg on CPU).
  - **Edge package (`v0.3-Edge-Mixed`, opset 21)**: `W4` (`GatherBlockQuantized`) for memory-bound embeddings + `INT8` for compute-bound Core linear layers (`~9.5–11.5 MB` ONNX / `~5.5–6.5 MB` zip).

### 1.4 Empirical Post-Mortem from `v0.3.0` (`A100-80GB` + Local Windows CPU)
Observed gaps between `v0.3.0` (`2x4-e192`, `2x4-e128`, `1x5-e128`) and ideal targets — directly feeding `v0.4-DualLoop`:
1. **Core Per-Layer CPU Overhead (`+40%` vs `v0.2` Linear Layers)**: On unloaded Windows 1-thread CPU (`seq=32`), `v0.2` (`4L`) runs `2.97 ms/row` (`bs=32`) / `3.36 ms/row` (`bs=1`), whereas `v0.3-2x4-e192` runs `prelude (1L)` at `0.73 ms/row` vs `1× core_step (2L)` at `2.05 ms/row` (`1.03 ms/layer`), summing to `3.80 ms/row` at `1.5×` average steps. Cause: decomposed 8-op manual `StepIndexedLayerNorm` (instead of fused ONNX `LayerNormalization`) plus 6 dynamic FP32 LoRA branches executed every Core layer.
2. **Bimodal Early-Exit Collapse**: On `test_all_bili_distilled (n=19,793)`, `75.4%` (`14,933`) exited at step 1 and `23.1%` (`4,567`) ran to step 3, with only `1.5%` (`293`) halting at step 2 — even though `v0.3-2x4-e192` `step_1` (`5L`) hit `test_bili_gold` AUROC `0.9564` / Spearman `0.8555` and `probe_u001_u017` AUROC `0.9333` (`AP=0.9764`), slightly beating `step_2`.
3. **Continuous Rank (`Spearman`) Dip & Over-Conservative `val_f05` Threshold**: `v0.3-2x4-e192` reached record AUROC (`0.9565`), AP (`0.9406`), and F0.5 (`0.9080`, Precision `97.47%`, FPR `1.29%`), but `Spearman` slipped (`0.8598 -> 0.8487`, `MAE 10.57 -> 11.10`) and `val_f05` drifted up to `0.8129` (leaving ~8–10% Recall unused under a `3%–5%` FPR budget) due to score jumps across early-exit steps and hard `(0.85, 0.15)` pair endpoints.
4. **Looped INT8 Drift & `W4-Emb` Export Order**: FP32-vs-INT8 `MAE` increased from `0.01256` (`v0.2`) to `0.01872–0.02218` (`max_abs_diff 0.121–0.143`) as quantization noise accumulated across Core iterations and shifted threshold exits; `edge_w4_w8` skipped on Colab due to missing `onnx_ir`/`ml_dtypes` (and should quantize `Gather` on FP32 Prelude prior to INT8 body quantization).
5. **Vocab Dimension Winner (`E=192`) & Remaining Hard Subgroups**: `E=192` beat `E=128` on Gold AUROC (`0.9565` vs `0.9546`) and all 5 AI adversarial traps (`A01 +0.061`, `A02 +0.069`), locking `E=192` for primary `v0.4`. Weakest individual probes remain ultra-short/casual AI (`U004`: `0.276–0.462` vs HX `73.0`) and eloquent technical human reviews (`H03_tech_human`: `0.571–0.661`).
6. **Wave-1 Real Bilibili Era/Partition Skew**: At `501,888 / 1,000,000` real rows (`301,518` new), `mid_llm (2025)` accounts for `86.4%` of new comments (`260,439`), while `pre_llm (<=2022)` is `0.45%` (`1,351`) and `资讯` is `2,169`, requiring a targeted Wave-2 era collector for historical human calibration.

---

## 2. Roadmap: `MeguvizCAC v0.4-DualLoop` & `1.25M` Corpus Expansion

Design goal: exploit the complementary strengths of the dual teachers (**`Qwen3.8-27B`** high-throughput pragmatic parsing & counterfactual rewrites + **`Gemini 4 Hydrogen (XHigh)`** literary sensibility, authentic human voice calibration, & minimal pairs + **Stage-1 12L `MacBERT`/`RoBERTa`** representation geometry) over a vastly broader `1,000,000` real Bilibili + `+250,000` new confirmed-AI dataset, while resolving every empirical bottleneck found in `v0.3.0`.

### 2.1 Corpus Expansion Plan (`+1,050,000` New Samples)
- **Real Bilibili Multi-Partition × Multi-Era Expansion (`200,370 -> 1,000,000`, Wave-1 >50% + Wave-2 Era-Targeted)**:
  - Stored in isolated WAL SQLite `v04-corpus/bili-live-1m.sqlite3` (deduplicated against `bili-live-corpus.sqlite3`, zero nicknames/UIDs/PII).
  - Wave-1 (`meguviz_cac/v04_bili_wave1.js`) crossed `501,888` rows across 16 partitions; **Wave-2 (`v04_bili_wave2_era.js`)** explicitly filters/targets **`pre_llm` (`<=2022`, goal `>=80k–100k`)**, **`early_llm` (`2023–2024`, goal `>=100k`)**, **`current_2026` (`goal >=150k`)**, and **`资讯` (`goal >=20k`)** so pre-ChatGPT human priors and latest 2026 LLM styles are well-balanced.
- **Virtual Video-Summary & Auto-Reply Plugin Confirmed-AI Expansion (`+250,000` unique rows — COMPLETED & VERIFIED)**:
  - Exported to `v04-corpus/synthetic-250k/bili-v04-synthetic-250k.jsonl.gz` (`33,651,295` B, SHA256 `b4a0437f5f5a4db415f15ea3f0b35f3b47384943d0fd39cd31fc9e0075dad8c4`): **`140,000` DeepSeek** (`local-ds 75,124` + `opencode-ds 64,876`) + **`110,000` multi-model** (`router-ling 62,706`, `opencode-glm 42,753`, `step 2,935`, `qwen 1,606`), `16` partitions (`15,079–15,971`/each), `8` plugin modes, `4` length buckets (including `29,328` ultra-short AI samples).
  - Audited **`0` duplicates** and **`0` held-out leaks** against all `350,387` frozen rows and `validation`/`test`/`U001–U017` splits (`origin_status = "confirmed_ai"`, `split = "train"`).

### 2.2 Dual-Teacher Distillation & Rationale Internalization
- **`LOTUS` Parallel Latent Reasoning Slots (`K=4`)**: append `[LATENT_1..4]` slots (zero text generation at inference, `<0.15 ms` overhead) aligned across loops to:
  1. `[LATENT_1]` ← `Qwen3.8-27B` surface cues (`assistant_leak`, `community_slang`, `recap`).
  2. `[LATENT_2]` ← `Qwen3.8-27B` frozen `semantic_intent` vector cosine distillation.
  3. `[LATENT_3]` ← `Gemini 4` literary/voice diagnosis (`reason_code`, `lyrical_emotion`, `irony_parody`, `external_personal_exp` + `judgment` rationale vector).
  4. `[LATENT_4]` ← Stage-1 `12L` (`MacBERT` + `RoBERTa`) attention-relation / CKA representation geometry.
- **Competence-Aware Disagreement Routing**: when $|s_{\text{Qwen}} - s_{\text{Gemini}}| \ge \Delta$, route primary authenticity supervision toward `Gemini 4` while preserving `Qwen3.8-27B` pragmatic/intent structure as auxiliary heads, modulated by student competence gating.
- **Skewed KL (`DistiLLM`)**: blend teacher and student distributions for boundary (`31–69`) and soft teacher supervision to prevent both over-smoothing and mode collapse.

### 2.3 Debiased Training, Calibration & Micro-Architecture
- **Worst-Subgroup Robustness (`GroupDRO`)**: dynamically reweight hard subgroups (`pool_group`, 6 minimal-pair categories, length buckets, persona/em-dash/kaomoji stress slices) so minority blind spots cannot hide behind average AUROC.
- **Length-Normalized `SimPO`-Style Minimal-Pair Ranking + Topic/Voice Decoupling**: enforce margin separation on `(x_ai, x_human)` minimal pairs while encouraging Prelude topic invariance and Coda voice separability.
- **Three-Stage Curriculum + Born-Again EMA Self-Distillation + SWA + QAT**: warm up on 1.25M+ soft labels & 12L alignment -> multi-teacher LOTUS + GroupDRO + minimal pairs + Ouro exit -> Born-Again EMA self-distillation + INT8 QAT + SWA weight averaging.
- **Split Conformal Prediction (3-Way Decision)**: zero-cost validation calibration outputting finite-sample FPR-controlled decisions (`human`, `ai` with `FPR <= alpha`, or `grey-zone escalate`).
- **Micro-Architectural Upgrades on `v0.3-Loop` (`~12–15 MB` ONNX)**:
  - **DeepSeek `mHC` (`N=4` Manifold-Constrained Hyper-Connections)** via Sinkhorn-Knopp doubly-stochastic residual mixing across loops.
  - **Differential Attention** in the shared Core to suppress comment noise / spam tokens.
  - **Lightweight 1D Byte/Char Conv Side-Stem** (`<0.3 MB`) fused before Prelude for intra-sentence spacing rhythm, kaomoji geometry, and code/CJK mixing.
  - **Optional Token-Level `MoR-Light`**: prune low-salience tokens during deeper loops `2–3` while keeping `[CLS]`, `[LATENT_1..4]`, and high-salience spans active.

### 2.4 `v0.3 -> v0.4` Concrete Upgrades & Quantitative Release Gates
Directly addresses the 6 empirical issues logged in Section 1.4:
1. **CPU Kernel Fusion & Selective Step-LoRA (Speed Gate: `<= 2.8 ms/row` unloaded `bs=32` at `seq=32`, beating `v0.2`'s `2.97 ms/row`)**:
   - Replace decomposed 8-op `StepIndexedLayerNorm` inside Core with fused `F.layer_norm` (`eps=1e-12`, lowering to a single native ORT `LayerNormalization` node) followed by cheap external step-indexed scale/shift modulation (`y * gamma_t + beta_t`).
   - Apply selective Step-LoRA (`Q/V/O` + `FFN_down`, pruning `K` & `FFN_up` dynamic LoRA overhead by ~45% MACs), while shipping both `2x4-e192` Primary and `1x5-e128` Ultralight variants.
2. **Oracle-Margin Exit Supervision & Delta-Convergence Halting (Eliminate Bimodal `75%/1.5%/23%` Collapse)**:
   - Train `exit_gate` against an online oracle sufficiency target (whether stopping at step $t$ already achieves the same correctness margin as deeper steps) + runtime delta-probability convergence check (`|p_t - p_{t-1}| < delta` on `t >= 1`) + depth-wise dropout/EMA so intermediate steps (`step_1`, `step_2`) steadily refine boundary cases.
3. **Continuous Rank & Calibrated Threshold Recovery (`Spearman >= 0.870`, `MAE <= 9.5`, calibrated `FPR 3% / 5%` with higher Recall)**:
   - Add cross-step logit consistency regularizer + Huber continuous regression (`weight=0.25`) + Skewed KL, replace hard `(0.85, 0.15)` pair endpoint clamping with `SimPO`-style margin + teacher soft targets, and apply post-exit Isotonic / Split Conformal calibration per target FPR (`3%` & `5%`).
4. **Low-Drift Looped INT8 QAT & Ordered `W4-Emb / W8-Body` Pipeline (Parity Gate: `MAE <= 0.010`, valid Edge zip)**:
   - Run 2 QAT epochs + SWA with an explicit exit-boundary stability penalty so INT8 quantization noise cannot flip early-exit routing.
   - Install `onnx_ir` & `ml_dtypes` and quantize `Gather` on **FP32 `prelude`** (`GatherBlockQuantized` bits=4, block_size=32) *before* applying dynamic INT8 quantization to dense Core/Prelude linear layers.
5. **Lock `E=192` Primary + Targeted Hard-Subgroup `GroupDRO` (`A01..A05 >= 0.78`, `H03_tech_human < 0.35`, `U004 >= 0.65`)**:
   - Lock `E=192` SVD embeddings (`+256` UTF-8 Byte-Fallback) for primary `v0.4-DualLoop`; leverage the new `29,328` ultra-short (`4–20` char) confirmed-AI samples + technical human Subgroup slices inside `GroupDRO` so concise casual AI and technical human discussions stop colliding.
6. **Wave-2 Era-Balanced Real Bilibili Harvest (`1,000,000` Total Real Corpus)**:
   - Launch `v04_bili_wave2_era.js` explicitly targeting `pre_llm` (`<=2022 >= 80k–100k`), `early_llm` (`2023–2024 >= 100k`), `current_2026` (`>= 150k`), and `资讯` (`>= 20k`) over the remaining `498k` real-corpus budget.
