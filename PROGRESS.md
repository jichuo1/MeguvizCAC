# MeguvizCAC Progress Log

## Phase 0 — Frozen Corpus & Legacy Failure Diagnosis [DONE]
- Frozen corpus verified: `350,387` rows (`200,370` Bilibili + `100,000` DeepSeek-V4.1 + `50,000` Other Models + `17` blind screenshot probes `U001–U017`), canonical order SHA256 `63b0b036b2dee7ded4a2bfedabd0bc5f82589fd40e426ef8e223cb0a2a5a6f66`.
- Diagnosed legacy failure modes: ~73% of DSH `ai` labels relied on nickname rules; `dsh_old_human` contained 6.5x `——` and 9.7x `「」` leakage; old Neon rubric over-penalized 15–45 char concise AI comments.

## Phase 1 — Hydrogen (XHigh) Rubric v2 & Full 350,387-Row Relabeling [DONE]
- Froze topic-invariant `hydrogen-xhigh-rubric-v2.txt` (SHA256 `729bd98c3695320fc1feebf45e8ee385bfebe529d4dc42e9d19a85d9b77f2ec4`).
- Passed 36-row blind pilot gate (`U001–U017` median `81`, `12/17 >= 70`; `10/10` Bili human traps `<= 30`).
- Completed Round 1 (`800`) + Round 2 active-learning (`650`) = **1,450** direct `Gemini 4 Hydrogen (XHigh)` Gold rows.
- Trained Colab L4 dual-backbone distillation (`Model A` 150k topic-free style GBDT + 5-fold `hfl/chinese-macbert-base` + 5-fold `hfl/chinese-roberta-wwm-ext`, automated CV chose `synth_weight=0.0`):
  - `ensemble_bili_gold_oof` (`n=1,193`): **AUROC `0.9703`**, **AP `0.9683`**, **Spearman `0.8527`**, **MAE `10.362`**.
  - `ensemble_all_gold_oof` (`n=1,450`): **AUROC `0.9661`**, **Spearman `0.8622`**.
  - Export `bili-hydrogen-rescored-350387.jsonl.gz` verified bit-for-bit (SHA256 `e6714373e3968c4a21fd36e377e29bedccb52b937ed8c79c156e5bf3a413378c`).
- Independent 60-sample flip audit (`hydrogen-final-audit.md`): **52/60 (86.7%)** strict pass rate (**93.3%** direct blind), identified `本鱼+空格/颜文字` & `uncertainty >= 15` border filters for Stage-2.

## Phase 2 — `MeguvizCAC` Repo Init & `v0.1` INT8 ONNX Baseline [DONE]
- [x] Initialize `jichuo1/MeguvizCAC` public repository & `v0.1-data` Release
- [x] Train `MeguvizCAC v0.1` (`BAAI/bge-small-zh-v1.5`) on Colab L4 GPU with zero val/test/probe leakage (`best_synth_cap=0`, `T=1.0223`)
  - `test_bili_gold` (`n=263`): **AUROC `0.9552`**, **AP `0.9349`**, **Spearman `0.8537`**, **MAE `10.525`** (`@val_fpr05`: Precision `92.71%`, Recall `82.41%`, FPR `4.52%`, F0.5 `0.9045`)
  - `test_dsh_gold` (`n=218`): **AUROC `0.9476`**, **Spearman `0.8377`**
  - `probe_u001_u017` (`n=17`): **AUROC `0.8667`**, **AP `0.9516`**, `@0.70` Precision `100.0%` / Recall `75.0%`
- [x] Export `meguvizcac-v0.1-int8.onnx` (`23,935,914` bytes), verify Windows CPU latency (`4.60 ms/row`) & parity (`MAE=0.0131`), publish Release `v0.1.0`

## Phase 3 — `Qwen3.8-27B` Comment-Understanding Distillation [DONE]
- [x] Selected `2:5:2:1` bucket-interleaved `train`-only high-leverage pool (`meguviz_cac/qwen_distill.py`): `bili_persona_trap: 682`, `bili_dsh_train: 1,705`, `bili_nondsh_hard: 681`, `synth_style_verified: 340` (`3,408` total, zero val/test/probe leakage)
- [x] Executed `greghavens/Qwen3.8-27B-bnb-4bit` (`enable_thinking=False`) + `flash-linear-attention` on Colab L4 GPU (`1.30 rows/s`, `100.0%` parse, zero OOM)
- [x] Extracted continuous probabilities, 8 pragmatic tags, semantic intents & `388` counterfactual twins; published Release `v0.2-qwen-raw` & closed Issue `#2`

## Phase 4 — Gemini 4 (`Hydrogen XHigh`) Literary Gate & Minimal Pairs [DONE]
- [x] Routed `530` high-risk Qwen comments through 5 native `Gemini 4 Hydrogen (XHigh)` Swarm judges: caught **`59/150` (39.3%)** Qwen lyrical/ironic false positives on real human text, **`30`** `本鱼+空格` persona-space disguises, and adjudicated `200` dual-teacher disagreements
- [x] Audited `150` Qwen counterfactual twins (`81/150 = 54.0%` PASS -> `<75%` quality gate dropped unaudited twins) and curated **`500`** unique Hydrogen XHigh literary minimal pairs (`581` total Stage-2 pairs across 6 categories)
- [x] Built `stage2-qwen/stage2_distill_bundle.zip`, saved `stage2-qwen/qwen-vs-hydrogen-audit.json`, published Release `v0.2-data` & closed Issue `#3`

## Phase 5 — `MeguvizCAC v0.2` Dual-Teacher INT8 ONNX Release [DONE]
- [x] Trained `MeguvizCAC v0.2` on Colab L4 (`1524.4s`, `best_synth_cap=5000`, `T=1.0632`, `val_f05=0.7908`, `val_fpr05=0.7797`, `qwen_rows=3408`, `pairs_rows=581`)
- [x] Benchmarked `v0.2` vs `v0.1`:
  - `val_bili_gold` (`n=208`): AUROC `0.9536 -> 0.9550`, Spearman `0.8701 -> 0.8760`, MAE `10.744 -> 10.311`
  - `test_bili_gold` (`n=263`): AUROC `0.9552 -> 0.9530` (`-0.0022`), Spearman `0.8537 -> 0.8598` (`+0.0061`), Recall@0.70 `85.19% -> 86.11%`, `@val_f05` Precision `92.39%` / Recall `78.70%` / FPR `4.52%` / F0.5 `0.8929`
  - `test_dsh_gold` (`n=218`): Spearman `0.8377 -> 0.8452` (`+0.0075`), AUROC `0.9452`
  - `probe_u001_u017` (`n=17`): **AUROC `0.8667 -> 0.9167` (`+0.0500`)**, **AP `0.9516 -> 0.9705`**, Recall@0.70 `75.0% -> 83.3%`, local INT8 `ge_070` `8/17 -> 11/17`, median `0.6774 -> 0.7763`
  - Local INT8 adversarial traps: `A01_persona_space` **`0.2103 -> 0.7935`**, `A02_param_kaomoji` **`0.2774 -> 0.7389`**, `A03_concise_emdash` **`0.6529 -> 0.8382`**, `A04_aphorism` **`0.7681 -> 0.8218`**, `H02_space` FP **`0.2721 -> 0.1920`**
- [x] Exported `meguvizcac-v0.2-int8.onnx` (`23,935,914` bytes, SHA256 `4662e3e6ace6ff01c5dc340d66cd0bb78bfcf8c1b50b5a5feecc5254a101b0c6`, parity `MAE=0.01256`, Colab 1-thread `9.86 ms/row`), published Release `v0.2.0` & closed Issue `#4`

## Phase 6 — `MeguvizCAC v0.3-Loop` Extensible Looped Transformer [DONE — Issue #5]
- [x] Locked & implemented `docs/roadmap-v0.3-v0.4.md`: Full `21,128` WordPiece weighted SVD (`E=128/192`) + `+256` UTF-8 Byte-Fallback (`21,384` total, `0% [UNK]`, rescuing uppercase `AI`/`B站`/`Blender` and `——` previously collapsed to `[UNK]`), `Prelude(1) + RRT Cycle Core(1/2×T)`, SVD LoRA init, empirical QK-Norm, Gated Input Injection, Ouro entropy-regularized adaptive exit, Stage-B INT8 QAT, and dual static ONNX (`prelude.onnx` + `core_step.onnx`) via `v03_tokenizer.py`, `v03_runtime.py`, `train_v03_loop.py`, and `notebooks/MeguvizCAC_08_v03_Loop.ipynb` (`cd6dd76`)
- [x] Completed 3-way **A100-80GB** ablation & local 1-thread Windows CPU 500-comment benchmark (`v0.2` baseline on same loaded machine: `15.12 ms` single / `15.25 ms` bucket bs32 / `82.85 ms` naive bs32):
  - **`v0.3-loop-2x4-e192` [Primary]** (`15.42 MB` ONNX / `9.93 MB` zip, `-35.6%`): new record `test_bili_gold` **AUROC `0.9565`**, **AP `0.9406`**, `@val_f05` Precision **`97.47%`** / FPR **`1.29%`** / F0.5 **`0.9080`**, `test_dsh_gold` **AUROC `0.9516`** / F0.5 **`0.8982`**, `probe_u001_u017` AUROC `0.9000` (`11/17 >= 0.70`), 5/5 local AI traps `>= 0.724` (`A01=0.7452, A02=0.7690, A03=0.8262, A04=0.7238, A05=0.8207`), real-500 `avg_steps=1.500` (`74.2%` early-exit at step 1), Windows CPU `20.25 ms` / `21.76 ms`
  - **`v0.3-loop-1x5-e128` [Ultralight]** (`10.36 MB` ONNX / `6.84 MB` zip, `-56.7%`): **`14.48 ms`** compact-batch32 Windows CPU (faster than `v0.2`, `avg_steps=1.690` = `2.69L`), `test_bili_gold` AUROC `0.9524` (`@0.70` Acc `89.35%`, R `88.89%`), `probe_u001_u017` **AUROC `0.9333`**, Recall@70 **`91.67%`**, local INT8 **`13/17 >= 0.70`**
  - **`v0.3-loop-2x4-e128` [High-Probe]** (`14.02 MB` ONNX / `8.92 MB` zip, `-41.4%`): `test_bili_gold` AUROC `0.9546` (`AP=0.9338`, `R@70=89.81%`), `test_dsh_gold` AUROC `0.9500`, `probe_u001_u017` **AUROC `0.9333`**, **AP `0.9736`**, Recall@70 **`91.67%`**, local INT8 **`13/17 >= 0.70`** (`med 0.8154`), real-500 `avg_steps=1.478` (`75.6%` exit step 1)
- [x] Published GitHub Release `v0.3.0` & closed Issue `#5`

## Phase 7 — `MeguvizCAC v0.4-DualLoop` & `1.25M` Corpus Expansion [IN PROGRESS — Issue #6]
- [x] Locked 16-partition × 4-era real Bilibili (`200k -> 1,000,000`) + virtual video-summary/reply plugin (`+250,000` confirmed AI) schema and collectors (`v04_synth_250k.py`, `v04_bili_ingest.py`, `v04_bili_wave1.js`)
- [ ] Complete `1,000,000` real Bilibili comment ingestion into `v04-corpus/bili-live-1m.sqlite3` & `+250,000` local ZCode CLI confirmed-AI generation
- [ ] Distill `v0.4-DualLoop` over the expanded `1.25M` corpus
