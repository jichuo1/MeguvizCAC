# MeguvizCAC v1.3.0-ApexKD Roadmap

> **Goal**: Close and surpass the pure-text recall gap vs `OjinjinNet v1.3` (`Nano 1.22 MB` / `Lite 7.05 MB` / `Standard 16.86 MB`) while strictly preserving ALL MeguvizCAC production invariants: **`Bundle <= 2.20 MB`**, **`Zero-PII === 0.0`**, **`F6 Promo→AI Firewall === 0.0`**, **Anti-Spoof PIFE**, and **`<= 1.6 ms` CPU cascade**.

## Forensic Drivers (from `OjinjinNet v1.3` Head-to-Head Audit)
1. **Super-Teacher Pretrained Chinese Encoder Distillation**: Replacing random-init `v0.4` synthetic supervision with a 3-model heterogeneous 24L Chinese pretrained Super-Teacher (`chinese-electra-180g-large` + `chinese-macbert-large` + `bge-large-zh-v1.5`) distilled onto `180,000+` real Bilibili comments.
2. **Label Purification**: Excluding behavior/cluster-only positives (`Coordinated-Latent-Cluster`, `Human-Assisted`, `god_tier Script-Plugin-Bot`) and `v0.4` synth from binary `y_ai=1` supervision, moving them to the unlabeled committee distillation pool.
3. **Anti-Dilution 65/35 Real Sampling + Old-Video (`>1y` / `pre_llm`) Anchoring**: Directly eliminates the top-3 old-video human FPs that collapsed `R@P95`.
4. **35% Online Punctuation-Shortcut Jitter + Head-180/Tail-74 Truncation**: Breaks the `85% ends_period` & `——/「」` shortcut on `30–60c` colloquial AI and preserves closing AI sign-offs in `>120c` long posts.

## Two-Stage Execution Plan
- **Stage 1 (`meguviz_cac/v130_superteacher.py`, `train_v130_superteacher.py`, `run_v130_superteacher_gpu.sh`)**:
  - Train the 3×24L Dual-Axis Zero-PII Super-Teacher Ensemble on Colab G4 RTX PRO 6000 (`~30 min`).
  - Export `v130_superteacher_soft_distill_pool.jsonl.gz` (`~180K` real comments with calibrated AI + Promo soft logits + committee uncertainty) AND `v130_al_disagreement_queue_top3000.jsonl.gz` for `AL-R4` sibling adjudication.
- **Stage 2 (`v1.3.0-ApexKD` On-Device Student `<= 2.20 MB`)**:
  - Rebalance the `2.20 MB` parameter budget: **`~0.95 MB` INT8 `ApexCharNano` Rung-0** (Char Unigram `4096×64` + Bigram `16384×24` + Trigram `8192×16` + boundary markers + 4-scale Dilated DWConv + SE + 5-Stat Pool) + **`~1.05 MB` W4 Compact Semantic/EBT/Promo-Firewall Verifier**, distilled directly from Stage 1's 180K Super-Teacher soft logits!
