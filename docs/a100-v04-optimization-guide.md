# A100-SXM4-80GB (`sm_80`) Optimization & 4-Teacher Joint Distillation Guide (`v0.4-DualLoop`)

This guide documents the hardware-aware optimizations built for **NVIDIA A100-SXM4-80GB (`cc = 8.0 / sm_80`)** and the **4-Teacher Joint Distillation** pipeline combining **Qwen3.8-27B BF16**, **Inkling-2-Mini**, and homologous co-teachers **Gemini 4 Neon** + **Hydrogen XHigh**.

---

## 1. Why `sm_80` Precision Selection Matters

| Model / Engine | Hardware Reality on A100 (`sm_80`) | Configured Strategy |
| :--- | :--- | :--- |
| **`Qwen/Qwen3.8-27B`** | A100 lacks native FP8 Tensor Cores (`cc < 8.9`); FP8 falls back to dequant/Marlin overhead while leaving `80 GB` VRAM under-used | Prefer **unquantized `bfloat16`** (`~54 GB` weights + `~19 GB` KV cache at `gpu_memory_utilization=0.92`, `max_model_len=2560`) with `enable_prefix_caching=True`, `max_num_seqs=128`, `max_num_batched_tokens=16384`, auto `gpu_bs=128`; HF `attn_implementation="sdpa"` fallback |
| **`thinkingmachines/Inkling-2-Mini`** | Looped 24-layer MoE (`~26.4B` total / `~3.4B` active; upstream NVFP4/MXFP8 targets Hopper/Blackwell `sm_90+` and may hit kernel incompatibility on `sm_80`) | **4-tier `InklingRunner`**: `--api-base` OpenAI endpoint -> `vllm` (with `sm_80` fallback check) -> `llama_cpp` `unsloth/Inkling-2-Mini-GGUF` (`*UD-Q8_K_XL*.gguf` on >= 36 GB) -> BF16 `transformers` |
| **12L Pseudo-Teacher (`v04_teacher_score.py`)** | Sequence padding wastes Tensor Core FLOPs | BF16 + TF32 + `TeacherBucketSampler` + `bs=512` train + length-sorted `torch.inference_mode()` `bs=1024` scoring + `AdamW(..., fused=True)` |
| **Student Dual-Loop Trainer (`train_v04_dualloop.py`)** | Per-step `.item()` CPU/GPU sync and Python collate stalled training | **Zero-sync GPU `TorchGroupDRO`**, `bs=1024` + `min(2.0, sqrt(bs/256))` LR scaling, `SDPA` during training (disabled cleanly via `set_sdpa(False)` before calibration/ONNX export), pre-tokenized `pairs_pre`, multi-worker NumPy collate, fused `AdamW`, optional `--compile` |

---

## 2. 4-Teacher Division of Labor & Priority Hierarchy

### Open-Weight Joint Teachers
1. **`Qwen/Qwen3.8-27B` (Full BF16)**: scans the `20k` active audit pool for deep Chinese social pragmatics, 8-bit `pragmatic_tags`, `semantic_intent`, and Chinese counterfactual minimal pairs.
2. **`thinkingmachines/Inkling-2-Mini`**: scans up to `4,000` non-Chinese (`EN/JA/KO/EU`) + `4,000` `zh_en_mixed` + `4,000` **Qwen-overlap** rows from `v04-active-audit-pool.jsonl`, and synthesizes `4,000` counterfactual **World-Knowledge (`WK_DOMAINS` x 12)** pairs (`EN` 50%, `ZH-EN` 22%, `JA` 12%, `KO` 8%, `EU` 8%) where both AI and human sides share identical entities/numbers/technical terms.

### Homologous Co-Teacher Tribunal (`v04_co_teacher_gate.py`)
Strengthens boundary calibration using **`gemini-4-neon-high`** (with optional **`gemini-4-neon-xhigh`** tiebreak on >25-pt splits) paired with frozen **`gemini-4-hydrogen-xhigh`** under bilingual **Rubric v2.1** (`0–100`, `A/R/E/T/P/H/S/Q`):
- Routes 6 priority buckets strictly on `split == "train"`: `qwen_vs_inkling_disagreement`, `structured_human_fp_trap`, `persona_space_disguise`, `literary_or_irony_risk`, `multilingual_wk_boundary`, and `open_vs_teacher12l_drift`.
- Uses small safe batches (`neon_batch_size=8`, `hydrogen_batch_size=10`, `tiebreak_batch_size=4`) with binary-split retries to avoid `xhigh` print timeouts.
- Audits Qwen + Inkling minimal pairs (`PASS/FAIL | AI | HU`) and exports verified pairs (`ai - hu >= 25`) for `--extra-pairs`.

### Exact Fusion Precedence (`fuse_four_teachers` & `merge_v04_corpus`)
1. **`pre_llm` Ground Truth First**: `pre_llm` human rows always retain `score = 2.0..3.0`, `reason_code = "H"`, while inheriting open-teacher `prag_vec` & `semantic_intent`.
2. **Tier 1 (`Neon + Hydrogen` Dual-Vote + Optional `Neon-XHigh` Tiebreak)**:
   - Tiebroken by `neon-xhigh` (`weight=3.0`) > Consensus `|N - H| <= 18` (`0.65*N + 0.35*H`, `weight=2.8`) > Weighted `<= 28` (`weight=2.0`) > High conflict (`code="Q"`, `weight=0.65`).
3. **Tier 2 (Single Homologous Gate)**: `Neon` alone (`weight=2.5`) or `Hydrogen` alone (`weight=2.2`).
4. **Tier 3 (Joint Open-Weight Consensus `Qwen + Inkling`)**:
   - Non-ZH: `0.70 * Inkling + 0.30 * Qwen` | `zh_en_mixed`: `0.50 * Inkling + 0.50 * Qwen` | ZH: `0.65 * Qwen + 0.35 * Inkling`.
   - Agreement `< 18` (`weight=1.8`) vs disagreement (`weight=0.85`).
5. **Tier 4 (Single Open-Weight Teacher)** > **12L Ensemble Teacher** > **`unscored_pending_teacher`** (`weight=0.0`).

---

## 3. Multilingual Semantic Cache (`bge-m3`), 12-GroupDRO & Edge Traps

- **Dynamic `intent_dim`**: `v04_teacher_score.py` encodes merged multilingual `semantic_intent` strings with **`BAAI/bge-m3` (`1024-d`)** by default (falling back to `BAAI/bge-small-zh-v1.5` `512-d` if unavailable), and `MeguvizV04DualLoopModel` dynamically sizes its training-only `lat2_intent_proj` head with **zero change** to exported INT8 ONNX weights.
- **12-Subgroup `GroupDRO`**: Adds `multilingual_world_knowledge` alongside the 11 Chinese subgroups (incl. `structured_human_hard`), plus asymmetric tail-push suppression on definite humans (`<= 0.15` general, `<= 0.22` structured/pre-LLM, `<= 0.18` multilingual).
- **Runtime Verification Traps**: `v04_runtime.py` validates both `H01..H09` / `A01..A05` and **`M01..M06`** (casual/structured EN, casual JA, ZH-EN tech benchmarks, and EN AI template/recap clichés) with strict zero-`[UNK]` UTF-8 byte fallback.

---

## 4. Estimated A100-SXM4-80GB Wall-Clock Budget (Full End-to-End Run)

| Stage | Module | Estimated A100 Time | Notes |
| :--- | :--- | :--- | :--- |
| 12L Teacher Train + 1M Score | `v04_teacher_score.py` | ~4–6 min | `bs=512` bucketed + `bs=1024` length-sorted score |
| Qwen3.8-27B BF16 20k + 800 Twins | `qwen_distill.py` | ~18–26 min | vLLM Prefix Caching `max_num_seqs=128` |
| Inkling-2-Mini 12k + 4k WK Pairs | `inkling_distill.py` | ~18–30 min | Depends on `vllm` vs `llama_cpp` `UD-Q8_K_XL` backend |
| Neon+Hydrogen Tribunal (Local CLI) | `v04_co_teacher_gate.py` | ~10–20 min offline | Run locally in parallel via `agy.exe`, then `--fuse-joint` |
| `bge-m3` Cache + 1.4M Corpus Merge | `v04_teacher_score.py` + `v04_corpus_prep.py` | ~2–3 min | Zero held-out leakage verification |
| Student Train (per variant `2x4-e192`) | `train_v04_dualloop.py` | ~10–14 min | Zero-sync `TorchGroupDRO`, `bs=1024`, `SDPA` -> INT8/W4-W8 ONNX |
