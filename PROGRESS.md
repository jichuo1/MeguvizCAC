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

## Phase 2 — `MeguvizCAC` Repo Init & `v0.1` INT8 ONNX Baseline [IN PROGRESS]
- [x] Initialize `jichuo1/MeguvizCAC` public repository & `v0.1-data` Release
- [ ] Train `MeguvizCAC v0.1` (`BAAI/bge-small-zh-v1.5`) on Colab L4 GPU with zero val/test/probe leakage
- [ ] Export `meguvizcac-v0.1-int8.onnx`, verify CPU parity & latency, publish Release `v0.1.0`

## Phase 3 — `Qwen3.8-27B` Comment-Understanding Distillation [PENDING]
- [ ] Select ~12k–15k `train`-only high-leverage pool (`meguviz_cac/qwen_distill.py`)
- [ ] Run 100-row throughput pilot (`enable_thinking=False`) on Colab L4/A100
- [ ] Extract continuous probabilities, 8 pragmatic tags, semantic intents & counterfactual twins

## Phase 4 — Gemini 4 (`Hydrogen XHigh`) Literary Gate & Minimal Pairs [PENDING]
- [ ] Route high-disagreement, lyrical/ironic, high-uncertainty & `本鱼+空格` samples through `Gemini 4 Hydrogen (XHigh)`
- [ ] Generate ~500 literary & voice-authenticity minimal pairs via Hydrogen XHigh swarm
- [ ] Publish fused `v0.2-data` Release

## Phase 5 — `MeguvizCAC v0.2` Dual-Teacher INT8 ONNX Release [PENDING]
- [ ] Train `MeguvizCAC v0.2` with soft score + Rubric code + 8 pragmatic tag heads + pairwise margin loss
- [ ] Benchmark `v0.2` vs `v0.1` on held-out `test_bili_gold`, `U001–U017`, and literary stress cases
- [ ] Publish Release `v0.2.0`
