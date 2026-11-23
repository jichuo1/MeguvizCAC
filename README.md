# MeguvizCAC

**MeguvizCAC** (Meguviz Comment AI Classifier) is a topic-debiased, compact Chinese Bilibili comment AI-generation detector (`INT8 ONNX`, ~24 MB) trained via a two-stage multi-teacher distillation pipeline:

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

## Student Benchmarks (`MeguvizCAC v0.1` vs `v0.2` on Held-Out `test` & Blind `U001–U017`)

| Version | Backbone & Format | `test_bili_gold` AUROC | `test_bili_gold` Spearman | `probe_u001_u017` AUROC / Recall@70 | CPU Latency (ms/row) | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **`v0.1`** | `BAAI/bge-small-zh-v1.5` INT8 ONNX (`23.9 MB`) | `0.9552` | `0.8537` | `0.8667` / `75.0%` (`P=1.00`) | `4.60 ms` (`1-thread Win`) / `9.73 ms` (`Colab`) | Released (`v0.1.0`) |
| **`v0.2`** | Dual-Teacher (`Qwen3.8-27B` + Gemini 4) INT8 ONNX | *Pending* | *Pending* | *Pending* | *Pending* | In Progress |

---

## Repository Layout

- `bili_hydrogen_relabel.py` — Hydrogen (XHigh) rubric pack, blind batch scorer, audit, and export SHA256 verifier
- `bili_hydrogen_distill.py` — Stage-1 Colab GPU topic-debiased distillation (`Model A` + `MacBERT` + `RoBERTa` + `Isotonic`)
- `relabel-hydrogen/` — Frozen Rubric v2 (`hydrogen-xhigh-rubric-v2.txt`), boundary notes, Stage-1 metrics, final audit, and Stage-2 roadmap
- `meguviz_cac/` — Student trainer (`train_student.py`), Qwen3.8-27B distillation (`qwen_distill.py`), literary gate (`literary_gate.py`), and ONNX verifier
- Large datasets & quantized ONNX release packages are hosted on [GitHub Releases](https://github.com/jichuo1/MeguvizCAC/releases) (`v0.1-data`, `v0.1.0`, `v0.2-data`, `v0.2.0`).

## Colab Notebooks

- **Stage-1 Relabeling (`MeguvizModel 06`)**: [Colab Link](https://colab.research.google.com/drive/1_YT29gFwdCKwFaF7gV_QnzdkFYAAVQWO)
- **Stage-2 Student & Qwen3.8-27B (`MeguvizCAC 07`)**: [Colab Link](https://colab.research.google.com/drive/1lshjncyEzJ1yWw98kxoasEnvDFHwj-ru)
