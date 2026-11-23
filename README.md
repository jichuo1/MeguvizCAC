# MeguvizCAC

**MeguvizCAC** (Meguviz Comment AI Classifier) is a topic-debiased, extensible Chinese Bilibili comment AI-generation detector (`INT8 ONNX`, `10.4–23.9 MB`) trained via a multi-stage multi-teacher distillation pipeline:

1. **Stage-1: Gemini 4 Hydrogen (XHigh) Blind Rubric v2 + Colab L4 Dual-Backbone Relabeling (350,387 rows)**
   - Eliminates nickname-rule artifacts (~73% of legacy AI labels) and 24-scene synthetic topic leakage.
   - Directly blind-scored **1,450** stratified Gold rows with `Gemini 4 Hydrogen (XHigh)` across two rounds (800 Round-1 + 650 Round-2 active learning).
   - Distilled over all **350,387** frozen comments (`200,370` Bilibili + `150,000` confirmed synthetic AI + `17` blind screenshot probes `U001–U017`) using topic-free 150k style GBDT + 5-fold GroupKFold `hfl/chinese-macbert-base` & `hfl/chinese-roberta-wwm-ext` + Isotonic calibration (`OOF AUROC = 0.9703`, `Spearman = 0.8527` on real Bilibili Gold).
2. **Stage-2: `MeguvizCAC v0.1` -> `Qwen3.8-27B` Comment Understanding + Gemini 4 Literary Quality Gate -> `MeguvizCAC v0.2`**
   - **`MeguvizCAC v0.1`**: `BAAI/bge-small-zh-v1.5` (`max_len=192`, `head_len=140`) trained on Stage-1 relabeled `train` split with uncertainty-weighted soft targets, temperature-calibrated & exported to dynamic `INT8 ONNX`.
   - **`Qwen/Qwen3.8-27B` Teacher**: Broad comment-understanding pass (`enable_thinking=False`) extracting 8 pragmatic tags (`recap`, `contrast_thesis`, `assistant_leak`, `contextual_ellipsis`, `external_personal_exp`, `irony_parody`, `lyrical_emotion`, `community_slang`), semantic intent, and topic-matched counterfactual twins.
   - **Gemini 4 (`Hydrogen XHigh`) Literary Gate & Minimal-Pair Teacher**: Audits high-disagreement / lyrical / ironic Qwen outputs, blocks space-disguised persona leaks (`本鱼` + spaces/kaomoji), curates literary minimal pairs, and trains **`MeguvizCAC v0.2`** with multi-task auxiliary heads (pruned at ONNX export for zero runtime overhead).

---

## Stage-1 Relabeling Summary (`350,387` Frozen Rows)

| Cohort | Total | `human_like` (0–30) | `uncertain` (31–69) | `ai_like` (70–100) | Provenance |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `bili` (DSH + non-DSH) | 200,370 | 140,943 | 55,332 | 4,095 | `unverified` |
| `deepseek` (confirmed synth) | 100,000 | 6,786 | 79,594 | 13,620 | `confirmed_ai` |
| `other_models` (confirmed synth) | 50,000 | 2,960 | 39,215 | 7,825 | `confirmed_ai` |
| `user_confirmed` (`U001–U017`) | 17 | 2 | 3 | 12 | `confirmed_ai` |

- Independent 60-sample flip audit ([hydrogen-final-audit.md](relabel-hydrogen/hydrogen-final-audit.md)): **52/60 (86.7%)** strict pass rate (**93.3%** direct blind, **95.0%** directional accuracy).

---

## Student Benchmarks (`MeguvizCAC v0.1` vs `v0.2` vs `v0.3-Loop` on Held-Out `test` & Blind `U001–U017`)

| Version | Backbone & Format | `test_bili_gold` AUROC / AP | `test_bili_gold` Spearman | `probe_u001_u017` AUROC / Recall@70 | ONNX / Zip | Local Real-500 Win CPU (1-thr) | Status |
| :--- | :--- | :---: | ---: | :---: | :---: | :---: | :--- |
| **`v0.1`** | `BAAI/bge-small-zh-v1.5` INT8 ONNX (`4L`) | `0.9552` / `0.9349` | `0.8537` | `0.8667` / `75.0%` (`P=1.00`) | `23.9 MB` | — (`9.73 ms` Colab) | Released (`v0.1.0`) |
| **`v0.2`** | Dual-Teacher (`Qwen3.8-27B` + Gemini 4) INT8 (`4L`) | `0.9530` / `0.9111` | **`0.8598`** | `0.9167` / `83.3%` (`AP=0.9705`) | `23.9 / 14.6 MB` | `15.12 ms` (`15.25 ms` b32) | Released (`v0.2.0`) |
| **`v0.3 (2x4-e192)` [Primary]** | `Prelude(1)+RRT(2×4) E=192 + ByteFallback` INT8 QAT | **`0.9565` / `0.9406`** | `0.8487` | `0.9000` / `83.3%` (`11/17 >= 0.70`) | `15.4 / 9.9 MB` | `20.25 ms` (`1.50` steps / `4.0L` avg) | Released (`v0.3.0`) |
| **`v0.3 (2x4-e128)` [High-Probe]** | `Prelude(1)+RRT(2×4) E=128 + ByteFallback` INT8 QAT | `0.9546` / `0.9338` | `0.8407` | **`0.9333`** / **`91.7%`** (`13/17 >= 0.70`) | `14.0 / 8.9 MB` | `21.06 ms` (`1.48` steps / `3.96L` avg) | Released (`v0.3.0`) |
| **`v0.3 (1x5-e128)` [Ultralight]** | `Prelude(1)+RRT(1×5) E=128 + ByteFallback` INT8 QAT | `0.9524` / `0.9166` | `0.8412` | **`0.9333`** / **`91.7%`** (`13/17 >= 0.70`) | **`10.4 / 6.8 MB`** | **`14.48 ms`** (`1.69` steps / `2.69L` avg) | Released (`v0.3.0`) |

- **`v0.2` Additional Splits**: `val_bili_gold` (`n=208`) AUROC `0.9536 -> 0.9550`, Spearman `0.8701 -> 0.8760`, MAE `10.744 -> 10.311`; `test_dsh_gold` (`n=218`) Spearman `0.8377 -> 0.8452`; `test_bili_gold @val_f05 (0.7908)` Precision `92.39%`, Recall `78.70%`, FPR `4.52%`, F0.5 `0.8929`.
- **`v0.3-Loop` Additional Highlights (Trained on A100-SXM4-80GB)**:
  - **`v0.3 (2x4-e192)` [Primary]**: `-35.6%` smaller ONNX, sets new high on `test_bili_gold` AUROC (`0.9565`), AP (`0.9406`), and `@val_f05 (0.8129)` Precision **`97.47%`** / FPR **`1.29%`** / F0.5 **`0.9080`**, plus `test_dsh_gold` AUROC **`0.9516`** (`@val_f05` P=`96.77%`, FPR=`1.52%`, F0.5=`0.8982`). `74.2%–75.4%` of real comments early-exit at step 1.
  - **`v0.3 (1x5-e128)` [Ultralight]**: `-56.7%` smaller ONNX (`10.36 MB`, `6.84 MB` zip) and **faster than `v0.2` on Windows 1-thread CPU** (`14.48 ms` compact-batch32 vs `15.25 ms` bucketed / `82.85 ms` naive `v0.2`), while raising blind `probe_u001_u017` AUROC to **`0.9333`** (`13/17 >= 0.70`).
  - **Zero-`[UNK]` Vocabulary Discovery**: Original `BAAI/bge-small-zh-v1.5` (`do_lower_case=false`) collapsed uppercase ASCII (`AI`, `B站`, `Blender`) and Chinese em-dashes (`——`) directly to `[UNK]`. `v0.3`'s `+256` UTF-8 Byte-Fallback slots rescued **`5,793`** unknown spans in 5,000 sampled comments with **`0` `[UNK]`** everywhere.

### Adversarial & Literary Stress Comparison (Local CPU INT8 ONNX)

| Stress Case (`verify_onnx.py` / `v03_runtime.py`) | Expected | `v0.1` | `v0.2` | `v0.3 (2x4-e192)` | `v0.3 (2x4-e128)` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `A01_persona_space` (`本鱼+空格` disguise) | High AI (`>=0.70`) | `0.2103` | **`0.7935`** | `0.7452` | `0.6839` |
| `A02_param_kaomoji` (`参数+颜文字` AI reply) | High AI (`>=0.70`) | `0.2774` | `0.7389` | **`0.7690`** | `0.7004` |
| `A03_concise_emdash` (concise sharp metaphor) | High AI (`>=0.70`) | `0.6529` | `0.8382` | `0.8262` | **`0.8680`** |
| `A04_aphorism` (antithesis aphorism) | High AI (`>=0.70`) | `0.7681` | **`0.8218`** | `0.7238` | `0.8026` |
| `A05_recap` (structured subtitle summary) | High AI (`>=0.70`) | **`0.8545`** | `0.8054` | `0.8207` | `0.8099` |
| `H02_space` (human gaming comment w/ spaces) | Low Human (`<0.30`) | `0.2721` | **`0.1920`** | `0.2439` | `0.2437` |
| `H03_tech_human` (human tech opinion w/ `Blender`) | Low/Mid (`<0.70`) | `0.6023` | `0.6078` | `0.6608` | **`0.5710`** |
| `probe_u001_u017` (`ge_070` / median, local INT8) | `12` / high | `8/17` / `0.6774` | `11/17` / `0.7763` | `11/17` (`13>=0.6`) / `0.8089` | **`13/17` / `0.8154`** |
| Cross-platform zero-`[UNK]` (`kaomoji`/`rare CJK`/`code`/`math`) | `0` `[UNK]` | `[UNK]` leaks | `[UNK]` leaks | **`0 [UNK]`** | **`0 [UNK]`** |

---

## Repository Layout

- `bili_hydrogen_relabel.py` — Hydrogen (XHigh) rubric pack, blind batch scorer, audit, and export SHA256 verifier
- `bili_hydrogen_distill.py` — Stage-1 Colab GPU topic-debiased distillation (`Model A` + `MacBERT` + `RoBERTa` + `Isotonic`)
- `relabel-hydrogen/` — Frozen Rubric v2 (`hydrogen-xhigh-rubric-v2.txt`), boundary notes, Stage-1 metrics, final audit, and Stage-2 roadmap
- `meguviz_cac/` — Student trainers (`train_student.py`, `train_v03_loop.py`), zero-`[UNK]` tokenizer (`v03_tokenizer.py`), dual-ONNX compact-batch runtime (`v03_runtime.py`), Qwen3.8-27B distillation (`qwen_distill.py`), literary gate (`literary_gate.py`), ONNX verifier (`verify_onnx.py`), and `v0.4` corpus collectors (`v04_synth_250k.py`, `v04_bili_ingest.py`, `v04_bili_wave1.js`)
- `notebooks/` — A100/L4 training & ablation notebook (`MeguvizCAC_08_v03_Loop.ipynb`)
- `docs/roadmap-v0.3-v0.4.md` — Locked `MeguvizCAC v0.3-Loop` extensible Looped Transformer architecture & `v0.4-DualLoop` 1.25M corpus expansion roadmap
- Large datasets & quantized ONNX release packages are hosted on [GitHub Releases](https://github.com/jichuo1/MeguvizCAC/releases) (`v0.1-data`, `v0.1.0`, `v0.2-qwen-raw`, `v0.2-data`, `v0.2.0`, `v0.3.0`).

## Colab Notebooks

- **Stage-1 Relabeling (`MeguvizModel 06`)**: [Colab Link](https://colab.research.google.com/drive/1_YT29gFwdCKwFaF7gV_QnzdkFYAAVQWO)
- **Stage-2 Student & Qwen3.8-27B (`MeguvizCAC 07`)**: [Colab Link](https://colab.research.google.com/drive/1lshjncyEzJ1yWw98kxoasEnvDFHwj-ru)
- **Stage-3 `v0.3-Loop` A100/L4 Ablations (`MeguvizCAC 08`)**: [Colab Link](https://colab.research.google.com/github/jichuo1/MeguvizCAC/blob/main/notebooks/MeguvizCAC_08_v03_Loop.ipynb)
