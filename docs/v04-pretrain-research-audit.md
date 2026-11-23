# MeguvizCAC v0.4-DualLoop Empirical Pre-Training Research Audit

Generated strictly from local Windows 1-thread ORT profiling and stratified zero-shot `v0.3-loop-2x4-e192` scoring over the completed `+250k` confirmed-AI corpus and read-only `bili-live-1m.sqlite3` snapshot.

## 1. ORT Operator-Level CPU Bottleneck Profile (`seq=32, bs=32, 1-thread`)

- **`v0.2` (`4L`)**: `4.7686 ms/row` (`1.1922 ms/layer`) — fused `LayerNormalization` `12509 us/batch`, zero FP32 LoRA.
- **`v0.3` `prelude` (`1L`)**: `1.2938 ms/row` — fused `LayerNormalization` `8.35%` of node time.
- **`v0.3` `core_step` (`2L`)**: `3.1435 ms/row` (`1.5717 ms/layer`, **`1.22x`** per-layer cost vs `prelude`).
- **Root-cause shares inside `core_step`**: decomposed 8-op Norm math (`ReduceMean/Sub/Pow/Sqrt/Div`) = **`3.55%`** (`3517 us/batch`), unquantized FP32 `MatMul` (26 nodes/call for Step-LoRA + attention) = **`13.14%`** (`13006 us/batch`), plus elementwise `Add` overhead surging to **`11.45%`** (`11338 us/batch`, 24 nodes/call vs only `1.96%` / `2969 us` in `v0.2`).
- **v0.4 Engineering Action**: Fused `StepModulatedLayerNorm` (native ORT `LayerNormalization`) + selective `Q/V/O + FFN_down` LoRA directly target ~26% of `core_step` node time.

## 2. `+250k` Confirmed-AI Plugin Evasion (`n=4000` stratified)

- **Zero-shot `v0.3` recall**: `@0.70 = 74.10%`, `@val_f05 (0.8129) = 43.03%`, `mean prob = 0.7150`, `avg_steps = 1.886`.

| `plugin_mode` | n | mean_prob | median | `recall@0.70` | `recall@val_f05` | `le_0.30` (FN) | avg_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_comment_ultra_short` | 500 | 0.6422 | 0.7626 | 56.8% | 27.4% | 9.6% | 2.23 |
| `auto_reply_plugin_thread` | 589 | 0.6785 | 0.7927 | 68.6% | 33.5% | 12.0% | 1.98 |
| `auto_comment_concise_insight` | 473 | 0.7079 | 0.8032 | 71.0% | 37.0% | 5.9% | 1.96 |
| `lyrical_or_meme_plugin` | 486 | 0.7091 | 0.8077 | 73.0% | 42.4% | 8.4% | 1.87 |
| `summary_plugin_casual` | 578 | 0.7073 | 0.8029 | 73.5% | 37.7% | 8.5% | 1.95 |
| `persona_agent_plugin` | 384 | 0.7382 | 0.8053 | 77.1% | 42.4% | 3.4% | 2.04 |
| `summary_plugin_structured` | 519 | 0.7643 | 0.8237 | 85.2% | 62.2% | 4.6% | 1.53 |
| `deep_review_long` | 471 | 0.7873 | 0.8264 | 89.6% | 64.1% | 2.8% | 1.54 |

By `length_bucket`:

| `length_bucket` | n | mean_prob | `recall@0.70` | `recall@val_f05` | `le_0.30` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ultra_short` | 748 | 0.5001 | 25.0% | 4.0% | 22.3% |
| `short` | 1191 | 0.6986 | 72.1% | 26.3% | 7.9% |
| `medium` | 1272 | 0.7913 | 91.3% | 60.1% | 1.7% |
| `long` | 789 | 0.8205 | 95.8% | 77.8% | 0.5% |

## 3. Multi-Era Real Bilibili Drift & `pre_llm` Natural Human FPR

- Snapshot size in WAL SQLite at audit time: **`501,094`** new rows (`501,094` across 4 eras).

| `era_bucket` | n | mean_prob | p90_prob | `flag@0.70` | `flag@val_f05` | avg_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre_llm` | 1000 | 0.2378 | 0.4031 | 0.10% | 0.00% | 1.45 |
| `early_llm` | 1000 | 0.2299 | 0.3924 | 0.30% | 0.00% | 1.41 |
| `mid_llm` | 1000 | 0.2350 | 0.4021 | 0.20% | 0.00% | 1.43 |
| `current_2026` | 1000 | 0.2763 | 0.4355 | 4.90% | 2.50% | 1.49 |

Top `pre_llm (<=2022)` False-Positive Partitions (`@0.70`):

| Partition | n | mean_prob | `FPR@0.70` | `FPR@val_f05` |
| --- | ---: | ---: | ---: | ---: |
| 鬼畜 | 63 | 0.2299 | 1.59% | 0.00% |
| 动物 | 60 | 0.2436 | 0.00% | 0.00% |
| 动画 | 65 | 0.2107 | 0.00% | 0.00% |
| 娱乐 | 62 | 0.2286 | 0.00% | 0.00% |
| 影视 | 63 | 0.2415 | 0.00% | 0.00% |
| 时尚舞蹈 | 64 | 0.2622 | 0.00% | 0.00% |
| 汽车 | 63 | 0.2559 | 0.00% | 0.00% |
| 游戏 | 60 | 0.2160 | 0.00% | 0.00% |

## 4. Tokenization (`[UNK]=0`) & Space-Delimited Rhythm Audit

| Cohort | `unk` | `byte_fallback_rows%` | `seq_p50` | `seq_p95` | `cjk_space%` | `emdash%` | `kaomoji%` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `synth_250k_sample` | 0 | 25.52% | 52 | 162 | 7.90% | 2.92% | 4.72% |
| `real_pre_llm` | 0 | 10.90% | 21 | 146 | 10.00% | 0.30% | 9.90% |
| `real_early_llm` | 0 | 8.80% | 22 | 93 | 7.20% | 0.10% | 11.10% |
| `real_mid_llm` | 0 | 9.00% | 19 | 97 | 7.10% | 0.10% | 6.30% |
| `real_current_2026` | 0 | 13.00% | 21 | 103 | 6.40% | 4.10% | 8.40% |
