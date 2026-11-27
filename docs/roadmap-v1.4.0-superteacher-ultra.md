# MeguvizCAC v1.4.0-SuperTeacher-Ultra Blueprint (SenseNova-Loop-9B Flagship)

## 1. Motivation: Loop-to-Loop Architectural Resonance + Newly Merged `+57,585` Zero-Leak Gold

`v1.3.0` Super-Teacher (3 × 300M 24L Chinese BERT encoders) lifted our Nested-Loop student from `v1.2.2` `OjinjinBench v3-hard R@P95 = 1.9%` to **`40.0%+`**, beating `OjinjinNet-Nano v1.3`. To unlock the next tier for `v1.4.0`, we upgrade BOTH the teacher backbone and the corpus:
1. **2026 Flagship Looped Backbone (`sensenova/SenseNova-Loop-9B`)**: SenseTime's 9B Looped Transformer executes 32 physical decoder layers with **Layers 4–27 looped 2×** -> **56 effective decoder layers**, **Two-Stream mHC (Manifold-Constrained Hyper-Connections)** with Sinkhorn normalization, and **Hybrid 3:1 Linear/Full Attention**! Distilling from a **56-effective-layer Looped Teacher** into our **`<= 5 MiB` Nested-Loop Student** creates true **Loop-to-Loop distillation synergy** while solving `v1.3.0`'s Chinese-only & pragmatic-reasoning blind spots natively!
2. **Newly Merged Zero-Leak Delta (`+57,585` rows in `v123-sibling-delta/`)**: `+28,991` `>1y` old-video hard humans, `1,422` R1c-rescued text-verifiable real AI (`273` short stealth AI + `87` real Q2), `200` no-period short stealth AI, `1,200` Q2 `(AI=1, Promo=1)` dual-pos, `813` real decorator humans, and `1,700` new micro-short (`12–30c`) & decorator (`1:1`) counterfactual pairs (`15,903` mirror pairs total!).

## 2. 3-Paradigm Super-Teacher Ultra Ensemble (`96 GB` RTX PRO 6000 Blackwell)

| Model Key | Public HuggingFace Backbone | Scale / Effective Depth | Paradigm & Tokenizer | Fine-Tuning Strategy | Est. Train + Score 250K |
|---|---|---|---|---|---|
| `electra_large` | `hfl/chinese-electra-180g-large-discriminator` | `325M` (24L) | **Chinese RTD** (WordPiece 21k) | Full BF16 + LLRD (`0.90`) | ~6 min |
| `bge_m3` | `BAAI/bge-m3` | `568M` (24L) | **Multilingual Bidirectional Contrastive** (100+ langs, SPM 250k) | Full BF16 + LLRD (`0.90`) | ~9 min |
| `sensenova_loop_9b` | **`sensenova/SenseNova-Loop-9B`** | **`9.0B` (32L → 56 eff. Looped L)** | **2026 Looped Hybrid-Attn + Two-Stream mHC** (Multilingual BPE) | `dtype="auto"` (BF16 backbone + **FP32 mHC Sinkhorn**) + Native **LoRA `r=32, α=64`** on Full-Attn + Linear-Attn + FFN + Exact Merge | ~18–25 min |

*Additional optional backbones in registry*: `bge_reranker_m3`, `sensenova_loop_9b_preview`, or any custom HF repo via `ST_MODELS=...`.

## 3. SenseNova-Loop-9B Specific Engineering (`v140_superteacher.py`)

- **Pure Text-Decoder Unwrapping (`_unwrap_text_backbone_v140`)**: Automatically extracts `.model.language_model` / `.language_model` from `SenseNova-Loop-9B`'s VLM wrapper and drops the unused vision tower & `lm_head` immediately to maximize VRAM headroom!
- **FP32 mHC Stability Guard (`preserve_auto_dtype=True`)**: Loads with `dtype="auto"` so SenseNova-Loop-9B's Sinkhorn-normalized dual-stream mHC routing stays strictly in **FP32** while linear weights remain **BF16**; LoRA targets ONLY Full-Attn + Linear-Attn (`in_proj*`, `out_proj`) + FFN projections with `min(in, out) >= 64` (never wraps mHC routing!).
- **Strict Zero-PII (`V140_ST_C1 === 0.0`) & Exact LoRA Merge Parity (`V140_ST_C2`)**: Canonicalizes `@user` and `<CONTACT_ID>` BEFORE tokenization (`canonicalize_zero_pii_v07` + `_CONTACT_ID_PII_RE`), tracks EMA & saves checkpoints strictly over trainable weights (`~240 MB` instead of `18 GB`!), and merges LoRA in-place for full-speed 250K pool inference.
- **Architecture-Matched Normalized 3-Stat Poolers**:
  - Bidirectional Encoders (`electra_large`, `bge_m3`): `[LN(max), LN(mean), LN(attn)]` (fixes XLM-R `cls_token_id=0` bug!).
  - Looped Causal Backbone (`sensenova_loop_9b`): Fixed short instruction prefix (`判断评论是否为AI生成或商业推广：\n`) + Head-192 / Tail-80 comment + final EOS, pooling `[LN(last_valid_eos), LN(comment_mean), LN(comment_attn)]`.
- **Counterfactual Mirror Margin Loss (`15,903` Pairs)**: Direct same-topic/length/decorator margin loss (`margin = 2.4`) prioritizing `delta_d2` micro-short (`12–30c`, 35%) & `delta_d4` decorator `1:1` (20%) pairs.
- **Native Multilingual EN/JA Training + Eval (`val_multilingual`)**: Clean EN/JA gold from `v04` with 28% online Cyrillic homoglyph / zero-width jitter on EN AI+.
- **Mega-Batch Length Bucketing (`16×`)**: Shrinks average padded length from `~240` -> `~65` tokens for `~3×` faster 56-layer looped training with zero batch-order bias.

## 4. Outputs & Colab G4 Launch

Outputs synced to `/content/drive/MyDrive/MeguvizCAC/v140_superteacher/`:
- `v140_superteacher_soft_distill_pool.jsonl.gz` (`~245K–260K` strictly zero-leak real comments; backward-compatible superset of `v130` schema!)
- `v140_al_disagreement_queue_top5000.jsonl.gz` (Top-5,000 highest inter-paradigm disagreement queue for AL-R5)
- Compact per-model checkpoints `v140_st_<key>_best.pt` + cached per-model distill logits (`--resume` friendly!)
- `metrics-v1.4.0-superteacher.json` + `SHA256SUMS.txt` + `.synced_ok`

**Colab Launch**:
1. Upload `v122-sibling-datafactory/v123_all_gold_sibling_and_delta_bundle.zip` (`39.16 MB`, `0` OjinjinBench hits — recommended!) or `v123-sibling-delta/v123_delta_bundle.zip` (`7.87 MB`) to `MyDrive/MeguvizCAC/`.
2. Run:
```bash
cd /content/MeguvizCAC && git pull origin main && bash meguviz_cac/run_v140_superteacher_gpu.sh
```
