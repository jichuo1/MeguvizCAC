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

---

## 2. Roadmap: `MeguvizCAC v0.4-DualLoop` & `1.25M` Corpus Expansion

Design goal: exploit the complementary strengths of the dual teachers (**`Qwen3.8-27B`** high-throughput pragmatic parsing & counterfactual rewrites + **`Gemini 4 Hydrogen (XHigh)`** literary sensibility, authentic human voice calibration, & minimal pairs + **Stage-1 12L `MacBERT`/`RoBERTa`** representation geometry) over a vastly broader `1,000,000` real Bilibili + `+250,000` new confirmed-AI dataset.

### 2.1 Corpus Expansion Plan (`+1,050,000` New Samples)
- **Real Bilibili Multi-Partition × Multi-Era Expansion (`200,370 -> 1,000,000`)**:
  - Stored in isolated WAL SQLite `v04-corpus/bili-live-1m.sqlite3` (deduplicated against `bili-live-corpus.sqlite3`, zero nicknames/UIDs/PII).
  - Coverage: **16 major Bilibili partitions** × **4 time eras** (`<=2022` `15%`, `2023-2024` `20%`, `2025` `25%`, `2026` `40%`) via authorized logged-in browser collection (`meguviz_cac/v04_bili_ingest.py` & `meguviz_cac/v04_bili_plan.json`).
- **Virtual Video-Summary & Auto-Reply Plugin Confirmed-AI Expansion (`+250,000` unique rows)**:
  - Generated via local ZCode CLI (`meguviz_cac/v04_synth_250k.py`: `140,000` DeepSeek routes + `110,000` multi-model routes across GLM, Kimi, Qwen, Step, Ling).
  - `16` partitions × `6–8` rich sub-topics (**100+ topics**, eliminating legacy 24-scene memorization) × **8 realistic AI plugin workflows** (structured subtitle summary, casual/de-AI summary, auto thread reply, ultra-short auto-comment, concise insight with/without em-dash, diverse non-hardcoded persona agent plugins, deep multi-detail review, lyrical/parody plugin) × **4 explicit length buckets** (`4–20`, `21–50`, `51–120`, `121–320` chars).
  - Strict exclusion against all existing `350k` rows, legacy SQLite corpora, and `validation`/`test`/`U001–U017` splits (`origin_status = "confirmed_ai"`, `split = "train"`).

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
