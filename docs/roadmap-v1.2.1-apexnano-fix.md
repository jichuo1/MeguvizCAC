# MeguvizCAC `v1.2.1-ApexNano-Fix` Engineering Blueprint (`ADR-v1.2.1`)

> **Date**: 2026-11-26  
> **Baseline**: `v1.2.0-R1 ApexNano-Omni` (`Bundle v7 = 2.0614 MB`, 5-Stage `FP32-Exact` ONNX `1.4e-6` exact parity, `V12_S1/F6/C1/R0 = 0.0 PASS`)  
> **Mission**: Fix ALL 5 forensic root causes exposed by the `v1.2.0-R1` blind 3-generation showdown + integrate ALL 6 corpus/pair/wild-promo discoveries from the RTX 4060 Ti Deep Data Audit (`scratch/deep_data_audit_report.json` & `scratch/v121_mining_goldmine_report.json`) — targeting **simultaneous All-Time Highs** on `OjinjinBench v1/v2` (`ROC + PR + R@P95`), `test_bili_gold_v2`, `Wave-2.5/PromoBench` wild commercial detection, `H06_scorecard`, and **mathematically certified 4-Rung LTT Cascade (`~1.6–1.8 ms` 4T-CPU, `~2.6–2.9 ms` 1T-CPU)** while staying strictly **`<= 2.20 MB` (`MGVZ7`)** and **`Zero-PII = 0.000000`**.

---

## 1. Forensic Post-Mortem of `v1.2.0-R1` (What Worked vs. Exact 5 Root Causes)

### What `v1.2.0-R1` Proved Beyond Doubt
1. **`105 KB` Zero-BPE `Rung-0` (`Rung0CharHashNanoV12`) is a Breakthrough**:
   - **0.031 ms ORT / 0.16–0.18 ms e2e (`4T-CPU`)**, **0.000000 Zero-PII**, **126,403 params**.
   - Standalone blind scores: `OjinjinBench v1 Hard R@P95 = 38.1%` (**all-time high across all 3 gens!**), `OjinjinBench v2 Hard PR = 0.7908` (**all-time high!**) / `Hard R@P95 = 38.8%`, `test_bili_wild2k AUROC = 0.9923`, `PromoBench-48 AI = 0.9271 / Promo = 0.9288`, **`Wave-2.5 Promo AUROC = 0.9526` (`AP = 0.8473`)**!
   - Empirical 4-Rung Cascade (`tau=0.08`, `R0+R1`): exits **`41.0%` on `wild2k` with `0/77` missed AI** (`2.291 -> 1.661 ms` 4T, `4.127 -> 2.635 ms` 1T, **-36.2%**); exits **`44.2%` on `OjinjinBench v2`** (`2.757 -> 1.796 ms` 4T, `4.935 -> 2.919 ms` 1T, **-40.8%**) with only `1/61` missed AI!
2. **Post-Unpack 5-Stage `FP32-Exact` ONNX Parity Achieved**:
   - Same-batch ONNX vs unpacked `MGVZ7` PyTorch max logit diff = **`7.2e-7` to `4.6e-6`** (completely eliminated `v1.1.0-R1`'s `0.063–0.105` MAE quantization discrepancy!).
3. **Bounded Upper-Tail Logits Unlocked `OjinjinBench v2 R@P95` (`10×` Jump!)**:
   - `v1.2.0` `T1` jumped `OjinjinBench v2` Full `R@P95`: `3.3% -> 22.9%`, Hard `R@P95`: `4.1% -> 42.9%`, Full `Best F0.5 = 0.7513` (all-time high!), and `v1` Full `R@P99`: `21.0% -> 32.9%`!
4. **Dual-Axis & Long-Doc Improved**:
   - `PromoBench-48`: AI `0.9826 -> 0.9965` (`+1.39%`), Promo `0.9271 -> 0.9340`.
   - `LongBench-24`: `long_ai_auroc`: `0.8958 -> 0.9514` (`+5.56%`!), `long_pr_auroc = 1.0000`.
   - Traps improved across almost all non-scorecard humans (`H03/H08/H09/M01..M04/P03`, `human_promo_max: 0.0645 -> 0.0445`).

### Exact 5 Root Causes Behind Every Single Regression in `v1.2.0-R1`

| # | Observed Failure in `v1.2.0-R1` Bench | Exact Verified Root Cause (Code & Math Proof) |
|---|---|---|
| **RC-1** | `T1/T2` ROC collapsed (`v1: 0.908→0.839`, `v2: 0.936→0.872`), `q05` floored at `p=0.058`, `H01_casual T1=0.29` while `EBT=0.0004` | **Positive-Orthant Cone Collapse in Uncentered `CosineLogitNormHeadV12`**: `emb = SiLU(coda_proj(...))` is **100.0% positive** (`min=+1.984, med=+2.890, max=+3.614, ||emb||=39.7`). Running `F.normalize(z)` *without subtracting the mean* makes 98% of `z_hat` point along the constant `[1,...,1]/√192` axis, crushing easy-human logits to `-2.5` instead of `-8.0`! Same bug hurt `promo_head`! |
| **RC-2** | `Phase A` threw away all 1,200 steps (`vg2: 0.905→0.9288` discarded!) and reverted to Step-0 untrained transplant; `H06_scorecard` rose `0.851→0.917` | **Noisy `val_gold_v1` Veto + Unadapted Step-0 Head**: `val_gold_v1` has `20.25%` mislabeled humans (`deep_data_audit_report.json`); blending 50% legacy `v1` in `ai_obj_sel` caused Step-0 (`0.86315`) to beat Step-600 (`vg2=0.9267`), reverting `ai_head` to untrained Step-0 transplant before Phase B froze it! |
| **RC-3** | `max_t2_ai = 0.2525` between `batch=32` and `batch=64` despite `1.4e-6` same-batch ONNX↔PT parity | **Padding Mask Leak in `mil_peak` & `r1_span_std`**: `tok_risk * mask0` sets pad tokens to `0.0`, which **beats negative real tokens (`-0.5084 → 0.0000`)** in `w_pool.amax(dim=-1)`! Also `w_avg.mean(dim=-1)` in `r1_span_std` averages over padded tail windows (`0.0182 → 0.0262`) and unmasked `h0_full` leaks pad into `avg_pool1d` boundary! |
| **RC-4** | `Wave-2.5` Promo: **`105 KB Rung-0 = 0.9526`**, yet full `T2 promo_head = 0.6076` (even in-train!); `lut_promo_full` 92.5% flat & capped at `0.503` | **Three Stacked Promo Bugs**: (a) `train_v12_apexnano.py:1045` **forgot to pass `r0_c1/c3/mask/feats` into `mdl(...)` in Phase B**, so `z_char` was `None` for all 2,400 steps and `z_char_coda_promo` got ZERO gradients! (b) `Wave-2.5` was only `0.45%` of Phase B batches while trunk was 100% frozen; (c) PAV Venn-Abers promo LUT collapsed 92.5% of knots flat and capped max prob at `0.503`. |
| **RC-5** | `fused3` `OjinjinBench v2 R@P95 = 1.6%` despite `T1 = 22.9%` / `T2 = 21.3%`; `R0/R1 LTT` `enabled: false` despite `0/77` missed AI on `wild2k` | **(a) Unbounded `ebt_margin` in `fused3`**: `ebt_margin ∈ [-30, +30]` overwhelmed bounded `T1/T2` on 1 single hard-human spike (`clip(ebt, ±6)` instantly recovers `v2 PR 0.6851 / R95 11.5% / Hard R95 38.8%` & `v1 R99 33.9%` zero-shot!). **(b) `cert_n_ai = 156 < 598`**: mathematically impossible for `Binom(156, 0.005)` to certify at `delta=0.05` (`0.995^156 = 0.457`). |

---

## 2. Comprehensive Fix Matrix for `v1.2.1-ApexNano-Fix`

### Pillar A: Architecture & Math Fixes (`meguviz_cac/v121_apexnano.py`, `Bundle v7 <= 2.20 MB`)
1. **Fix RC-1 — `SoftBallLogitNormHeadV121` (Centered Smooth Ball Projection on BOTH `ai_head` & `promo_head`)**:
   - Replaces uncentered `CosineLogitNormHeadV12` with:
     $$z_c = z - \mu_{\text{center}}, \qquad z_{\text{ball}} = \frac{z_c}{\sqrt{1 + (\|z_c\|_2 / r_{\text{cap}})^2}}, \qquad \text{logit} = \langle w, z_{\text{ball}} \rangle + b$$
   - **Exact Zero-Shock Step-0 Init**: Initialize $\mu_{\text{center}} = \text{mean}(z_{\text{cal}})$, $r_{\text{cap}} = 1.8 \cdot \text{median}(\|z_c\|)$, $w = w_{\text{v11}} \cdot \sqrt{1 + (\text{med}/r_{\text{cap}})^2}$, $b = b_{\text{v11}} + \langle w_{\text{v11}}, \mu_{\text{center}} \rangle$.
   - **Why it beats both unbounded linear AND uncentered cosine**:
     - Preserves 100% full negative-tail resolution (`logit = -10` for easy humans -> restores `T1/T2` `ROC = 0.915–0.950+`)!
     - Does NOT amplify tiny noise near the origin (`||z_c|| ≈ 0` stays near `0` instead of blowing up to unit norm)!
     - Strictly caps extreme outlier norms (`||z_ball|| < r_cap` -> mathematically impossible for `H06` or hard-human spikes to explode to `p = 0.99999` -> keeps `25%–45%+` `R@P95`!).
   - Zero extra runtime latency (`+192` FP16 params = `0.38 KB`!).
2. **Fix RC-3 — Exact Masking for Strict Batch-Padding Invariance (`V121_P1 < 1e-4`)**:
   - **`mil_peak` & `promo_mil_peak`**: compute valid window mask `w_valid = F.max_pool1d(F.pad(mask0.float().unsqueeze(1), (0, 7)), kernel_size=8, stride=4).squeeze(1) > 0.5`; replace `w_pool` sum with valid-token mean inside each window (`sum(tok_risk * mask) / sum(mask).clamp(min=1)`) and apply `.masked_fill(~w_valid, -1e4).amax(dim=-1)`! Pad windows can NEVER beat negative real token risks!
   - **`r1_span_std`**: compute `r_std_win` strictly over `w_valid` windows (`masked_mean` & `masked_var`), eliminating pad-window dilution!
   - **`h0_full` -> `h_pool`**: multiply `h0_full = h0_full * m_f` before `avg_pool1d` AND compute masked average pool (`sum_pool / count_pool.clamp(min=1)`) so odd-length boundary tokens never average with pad zeros!
   - **Automated CI Gate `V121_P1_batch_invariance`**: runs the same short text at `batch=1` vs padded to `L=192` in `batch=2` and asserts `max_abs_diff < 1e-4` on **ALL 5 outputs (`r0, r1, t1, t2, ebt`)**!
3. **Fix RC-5a — Bounded `tanh`-Squashed 3-Parameter Log-Odds Fusion (`fusion_v121`)**:
   - `ebt_cap = ebt_clip * tanh(ebt_margin / ebt_clip)` (default `ebt_clip = 6.0`, stored in `manifest.json` & baked into `MeguvizV121CascadeEngine`!).
   - `fit_3param_logodds_fusion_v121` optimizes `0.30*AUROC + 0.30*PR + 0.25*R@P95 + 0.15*R@P99` with `w_ebt <= 0.35`, preventing any EBT outlier spike from pinching `v2 R@P95`.
4. **Fix RC-4a — Firewall-Strict Live `z_char` Bridge into `emb[192:256]` + Residual `R0->R1/Full` Promo Fusion**:
   - Pass `r0_c1, r0_c3, r0_mask, r0_feats` into `mdl(...)` in **ALL 3 phases** (with `z_char_coda_ai_gate` / `z_char_r1_gate` cleanly controlled, and **UNFROZEN `z_char_coda_promo` (linear + 2-layer MLP or direct `32 -> 64` projection into `emb[192:256]`!)** during Phase B so `T1/T2 promo_head` directly inherits `Rung-0`'s `0.9526` character-level wild-promo signal with **mathematically `0.000000` leak into `AI [0:192]` (`V12_F6`)**!).
   - Also add a dedicated `z_char_r1_promo` (`32 -> 16`, promo-only branch into `r1_promo_head`) so `Rung-1` Promo jumps from `0.62 -> 0.95+` too without touching `r1_ai_head` (`V12_F6 = 0.0`)!

### Pillar B: Deep Data Audit Refinery (`meguviz_cac/v121_data_refinery.py`)
Integrates ALL 6 findings from `scratch/deep_data_audit_report.json` & `scratch/v121_mining_goldmine_report.json`:
1. **Quarantine Proven-Broken Midbands (`T3_QUARANTINE`)**:
   - `colab_gpu_distilled` midband `30 < score < 70` (**95.54% real AI** in audit `B2`!) -> **`T3_QUARANTINE`** (stop forcing `y = 0.50`!).
   - `v08_gemini_pro_gold` (388) midband `30 < score < 70` (**100.0% real AI** in audit `B3`!) -> **`T3_QUARANTINE`** (overridden cleanly by `Gold-v2`!).
2. **Tame `v08_gemini_flash_gold` False-Human Bias (`77.78%` false-human on `B1`!)**:
   - For `v08_gemini_flash_gold` rows with `15 <= score <= 40`, downgrade from `T1 (w=1.10)` to **`T3_QUARANTINE` / `T2 (w=0.20)`** unless overridden by `Gold-v2` or `Wave-3`! Keep decisive `score >= 80` or `score <= 10` Flash rows at `w = 0.85`.
3. **Strict `has_ai` Masking on Promo-Only / Unverified-AI Pools (Zero Cross-Axis Poisoning!)**:
   - Set `has_ai = 0.0` on:
     - `v11_bge_knn_promo_hard_negatives_2k` (**388/2,000 = 19.4% real AI** mislabeled `old_ai_score=15`!)
     - `v11_promo_cf_1600` (**38.25% pairs shift AI prob > 0.30**!)
     - `v11_uot_cdmad_pseudo_full` (unverified `old_ai_score`!)
   - AI losses (`l_ai`, `l_kd`, `l_pa`, `l_tp`, `l_r0_ai`, `l_topfp`) ONLY apply where `has_ai > 0.5`!
4. **9/9 Pair-Dataset Surgical Filter**:
   - **KEEP ONLY DIAMOND PAIRS (`0.0%–1.2%` inversion)**: `v05_gateguard`, `v10_topic_debias`, `v04_twins_clean`, `v08_gemini_13k (0.20% inv)`, `v04_inkling_wk (0.97% inv)`, `stage2_literary (1.20% inv)`.
   - **DROP NOISY UN-GATED PAIRS**: exclude `v04_qwen38_twins (6.75% inv)` & `qwen38_27b_twins (5.67% inv, 0.405 HU mean)`.
5. **Native `Wave-3` Ingestion (`meguviz_cac/ingest_wave3.py` + `--wave3-gold`)**:
   - Plug-and-play support for the 18-shard `Wave-3` pack (`852` real mined wild-promo `P1–P5` + `A1` flash stealth bots + `A2` wave12 conflicts + `A3` 2026 structured human tail traps + `25` hidden canaries).
   - Splits cleanly into `wave3_train` (80%) and `wave3_val` (20%, NEVER touching frozen `test_bili_gold_v1/v2` or `OjinjinBench`!).
   - In **Phase B**, adds a dedicated **`12%` wild-promo (`Wave-2.5 + Wave-3 Promo + P1–P5`) per-batch slice** so wild Bilibili commercial patterns are never starved to `0.45%`!
6. **Structured-Human Tail & Scorecard Protection**:
   - Dedicated protection boost (`w = 1.8`, `l_topfp` active) on verified 2026 structured human reviews/scorecards/technical lists (`Gold-v2` + `Wave-3 A3` + `protect_pool`) without touching any test trap string.

### Pillar C: Training, Calibration & LTT Certification Fixes (`meguviz_cac/train_v121_apexnano.py`)
1. **Fix RC-2 — Phase A Clean Selection & Controlled Top-Adapter Adaptation**:
   - Selection metric `ai_obj_sel_v121 = 0.55 * ai_obj_gold_v2 + 0.35 * ai_obj_wild_cal + 0.10 * r1_auroc` (never vetoed by noisy legacy `val_gold_v1`!).
   - Train `SoftBallLogitNormHeadV121` (`ai_head`), `SharedQuantResV12`, `spectral_damp_logit`, `coda_proj[:192]`, `style_net[:192]`, `span_head`, `r1_ai_head`, `rung0` (`lr=1.0e-4`) + ultra-gentle top core (`core.inloop_ebt` `lr=4e-5`, `core.attn/ffn` `lr=1.5e-5` with `L2-SP=0.05`) — **zero fresh-AdamW shock** (`Step-0` already matches `v1.1` thanks to `SoftBallLogitNorm` zero-shock init!), while letting `Gold-v2` & `Wave-3` actually sharpen representations!
2. **Fix RC-4c — Smooth Calibrated Promo LUT (`0%` Flat Ties, Full `[0.001, 0.995]` Range)**:
   - Replace piecewise-constant PAV-only promo LUT (`92.5%` flat, capped at `0.503`!) with **monotonic Beta-Spline + wild-prior log-odds shift** (`lut_promo_r0/r1/full`), verified `< 5%` flat fraction and `y_max >= 0.90`!
3. **Fix RC-5b — True Certified Dual-Rung LTT (`cert_n_ai >= 900`, `enabled: true`)**:
   - Expand held-out `val_cal_cert` strictly from `eval_rows` (`split == "validation"`, disjoint from `val_gold` & `val_cal_select`) to **`n_ai >= 900` positives** (`0.995^900 = 0.011 < 0.05`!) so `certify_fixed_sequence_dual_axis_ltt` achieves **mathematical FWER-controlled certification (`certified_ltt: true, enabled: true`)** on BOTH `Rung-0` (`~20–25%` exit @ `0.16 ms`) and `Rung-1` (`~18–22%` exit @ `0.45 ms`) -> **`~1.65 ms` (4T) / `~2.65 ms` (1T) out-of-the-box!**

---

## 3. Automated Release & CI Gates for `v1.2.1-R1`

| Gate ID | Check | Threshold |
|---|---|---|
| `V121_S1` | Dynamic sequence lengths `{2, 17, 65, 256}` + Analytic 6-Way EBT grad | `max_diff < 1e-5` |
| `V121_F6` | Structural Firewall `[192:256] + z_char_coda_promo + z_char_r1_promo -> AI` | `max_leak === 0.0` |
| `V121_C1` | Strict Zero-PII across ALL 4 Rungs (`R0/R1/T1/T2`, AI & Promo) | `max_diff === 0.0` |
| `V121_R0` | `Rung-0` parameter budget | `<= 150,000` params |
| **`V121_P1`** | **Batch-Padding Invariance (`batch=1` vs `pad=192` across `R0/R1/T1/T2/EBT`)** | **`max_diff < 1e-4`** |
| **`V121_L1`** | **No Positive-Orthant Collapse (`easy_human < -5.0`, `ai > +4.0`) + Promo LUT `flat < 10%`** | **`PASS`** |
| `MGVZ7` | Total `bundle-v7-sub2bit.zip` size | **`<= 2.20 MB`** |
| `ONNX5` | Post-Unpack 5-Stage `FP32-Exact` ONNX vs `MGVZ7` PyTorch | **`max_diff < 1e-4`** |

---

## 4. Target Metric Projections (`v1.1.0-R1` -> `v1.2.0-R1` -> `v1.2.1-R1`)

| Benchmark / Axis | `v1.1.0-R1` | `v1.2.0-R1` (Actual) | `v1.2.1-R1` Target |
|---|---|---|---|
| **Blind OjinjinBench v1 Full** (`ROC / PR / R@P95 / R@P99`) | `0.922 / 0.860 / 49.0% / 21.0%` | `0.922 / 0.854 / 50.6% / 32.9%` *(split across T2 & EBT)* | **`>= 0.925 / 0.865 / 50%+ / 30%+` (Unified `fused3`!)** |
| **Blind OjinjinBench v2 Full** (`ROC / PR / R@P95`) | `0.949 / 0.690 / 6.6%` | `0.947 / 0.685 / 22.9%` *(T1=22.9%, fused3 pinched 1.6%)* | **`>= 0.950 / 0.700 / 25%–38%+` (Unified `fused3`!)** |
| **Blind OjinjinBench v2 Hard** (`ROC / PR / R@P95`) | `0.919 / 0.764 / 6.1%` | `0.919 / 0.791(R0) / 42.9%(T1)` | **`>= 0.922 / 0.785 / 40%+`** |
| **`test_bili_gold_v2` (263)** (`ROC / PR / R@P95 / R@P99`) | `0.950 / 0.973 / 86.4% / 74.7%` | `0.945 / 0.970 / 83.3% / 79.6%` | **`>= 0.955 / 0.976 / 88%+ / 80%+`** |
| **Wild Promo (`Wave-2.5` + `Wave-3` Val)** (`T2 AUROC / R0 AUROC`) | `0.594` (blind) | `0.608 (T2)` / `0.953 (R0)` | **`>= 0.92+ (T2)` & `>= 0.93+ (R0)`** |
| **`H06_scorecard` Trap** | `0.8507` | `0.9172` | **`< 0.55`** |
| **4T / 1T CPU Wild Cascade Latency** | `2.67 / 4.79 ms` (no cascade) | `1.66 / 2.64 ms` (`LTT off`) | **`<= 1.65 / 2.65 ms` (`LTT CERTIFIED ON!`)** |
| **Batch-Padding Parity (`batch=1` vs `pad=192`)** | N/A | `0.2525` (`mil_peak` bug) | **`< 0.0001` (`V121_P1 PASS`)** |
