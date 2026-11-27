# MeguvizCAC v1.2.3-ApexNano-Pro — Implementation-Grade Architecture & Post-Training Spec (`ADR-v1.2.3`)

## 1. Empirical Baseline (`v1.2.2-r1` 5-Gen Blind Showdown + 6 Local RTX 4060 Ti Probes)

- **`v1.2.2-r1-apexnano-b7` Breakthrough**: All-time 5-generation highs across **all 4 blind OjinjinBench suites** (`v3-real` `PR 0.8027 / F0.5 0.8676 / R@P95 64.1% / Hard R@P95 71.7%`; `v1` `PR 0.8672 / R@P95 58.7% / R@P99 34.5%`; `v2` `ROC 0.9570 / PR 0.7598`; `v3-hard` `ROC 0.9183 / PR 0.6873 / F0.5 0.7143`), `Wave-3` Promo (`AUROC 0.9185 / AP 0.8531 / Q4 FPR 2.47%`), and `Wild2k` (`AUROC 0.9970 / PR 0.9237 / R@P95 55.8%`), while reducing non-promo false positives by **17×–36×**.
- **6 Root Causes Identified by Local Empirical Probes (`v123_empirical_probes.json`)**:
  1. **`god_tier_500_holdout` Contamination of `val_meta_cal`**: Mixing 500 extreme/mixed-verifiability holdout rows (`35.6%` of `val_meta_cal`, `45.6%` pos, `41%` cluster/script bots) inflated `lut_ai_r0` floor `4.4×` (`0.0075 → 0.0333`), throttled R0 early-exit to `2.0%`, and cost `-13.0pp` `v3-real R@P95`.
  2. **Closed `z_char_coda_ai_gate` (`0.0318` vs `promo 0.4387`)**: Left `t2` blind to `r0_charchash`'s `+8–14pp` gains on short text, decorator templates (`✧/❀`), and persona agents.
  3. **Over-Compressed `R1` Negative Logits (`exit_affine_a[0] = 0.5557`)**: Compressed `r1` clean-human logits to `[-1.6, -1.0]`, holding `exit_r1 = 0%` even after clean LUT calibration.
  4. **Phase C Sub-2bit QAT Tail Loss (`-6.3pp w_rp95`)**: HAWQ-V2 empirical tail-Fisher ranking pinpointed 8 high-sensitivity `prelude_l0` + `core.o_proj` layers (`seq2/trit158`) and 3 zero-Fisher `w4` layers (`h2_hyp_proj`, `h2_doubt_proj`, `core.haar_q`). Reallocating these 11 layers fits inside `~2.146 MB <= 2.20 MB` with **zero Bundle v7 / 5-Stage ONNX ABI change**.
  5. **Era Subspace Leakage & LTT Finite-Sample Math**: `style56_raw` alone carries `0.8078` era AUROC (while final logits already reversed old-comment bias: `delta_old_minus_new = -0.076`), and `Phase C` disabled `OfflineEraGRL`; `val_cal_cert` had `n_pos=320`, below the exact Clopper-Pearson minimum (`960 @ alpha=0.005`, `690 @ alpha=0.01`).
  6. **Sibling 203k Corpus Gaps**: `<30c` real HU:AI = `25.6:1` (while `pack1_A` synth `<30c` is `5,612:1` AI), `AI_PR` dual-positive quadrant has only `265` rows (`0.27%`), `>1y × 25–80c` short epigram hard humans are under-covered vs `>120c`, and `text_only_verifiable` needs explicit separation.

## 2. Eight P0 Upgrades in `v1.2.3-ApexNano-Pro`

| ID | Component | Specification |
|---|---|---|
| **P0-1** | **6-Way Disjoint Splits + Certified LTT** | `god_tier_500_holdout` strictly diagnostic-only (`text_verifiable` vs `all`); `val_meta_cal` 100% pure wild; `val_cal_cert` expanded to `>= 1,000` real-wild AI positives so `certified_ltt = True`. |
| **P0-2** | **`R0 ↔ T2` Mutual Co-Distillation + `z_char` Unlock** | Unlock `z_char_coda_ai_gate` (`>= 0.15`), `3×` LR on `z_char_coda_ai*`, `detach_zchar_for_ai=False` in Phase A1/C, plus `L_r0_to_t2` reverse KD on confident `<60c` & template samples. |
| **P0-3** | **6-Subpool Stratified Phase A Quotas + `<60c` SOPA-s** | Explicit quotas (`M_hard_mem 15%`, `ai_rare_stealth 18%`, `dual_pos_ai_promo 5%`, `era_dro_humans 16%` [1.5× `>1y × 25–80c`], `sib_dual_gold 22%`, `legacy_t0_t1 24%`, synth `<= 8%`) + `<60c` dedicated `SOPA-s` pAUC. |
| **P0-4** | **`R1` Clean-Human Floor Margin + Affine Clip** | Add `L_floor = 0.12*ReLU(z_r1+2.8)^2 + 0.10*ReLU(z_r0+3.0)^2` on clean humans (`y<=0.15`) + tighten `fit_cross_exit_affine_alignment` `clip_a=(0.80, 1.80)` to unlock `R0+R1 >= 45–60%` fast exit. |
| **P0-5** | **HAWQ-V2 Bit Reallocation (`~2.14 MB <= 2.20 MB`)** | Upgrade 8 top-Fisher layers (`prelude_l0.{k_p,v_p,o_p,ffn_p.w_up,ffn_p.w_down,v_c,o_c}`, `core.o_proj`) -> `w4` (`gs=64`), demote 3 zero-Fisher layers (`h2_hyp_proj`, `h2_doubt_proj`, `core.haar_q`) -> `seq2` (`gs=128`); `0.6×` QAT LR + `QuEST` + `BitDistill`. |
| **P0-6** | **Full-Pipeline Era Invariance (`Phase A + C`)** | `15%` era-sensitive `style56` feature dropout during training + keep `OfflineEraGRL` active in BOTH Phase A (`0.15`) and Phase C QAT (`0.10`); dual-check `V123_E1` (`emb <= 0.62`, `logit <= 0.58`, `|delta| <= 0.12`). |
| **P0-7** | **Shock-Absorber v2 + Constrained Length-Aware `z5_softmin`** | 300-step post-quant Shock-Absorber over `ai_head + r1_head + exit_affine + z_char_coda_ai + promo_head`; 2-bucket `<60c` (`r0/r1`-boosted) vs `>=60c` (`t2/ebt`-anchored) `z5_softmin` (`<120 B` manifest). |
| **P0-8** | **11-Gate CI (`V123_*`) + `v123-sibling-delta` Ready** | Adds `V123_AI_LUT_FLOOR` and `V123_CERTIFIED_LTT` gates on top of all 9 `v1.2.2` gates; supports seamless warm-start from `v1.2.2-r1` FP16 master (`g0_rmse=0.0`) or dual-source `v1.0 + v1.2.2`, plus zero-leak `v123-sibling-delta/` auto-ingestion. |

## 3. Target Release Acceptance Criteria

- **Deployment Hard Gates**: `bundle_v7_total_mb <= 2.20 MB`, `onnx_5stage_exact == True` (`< 1e-4`), `Zero-PII === 0.0`, `F6 === 0.0`, `11/11 V123 CI PASS`.
- **Blind Showdown Targets**: `v3-real` `PR >= 0.825 / R@P95 >= 70%`, `v3-hard` `PR >= 0.720 / R@P95 >= 25%`, `v2` `PR >= 0.790 / R@P95 >= 42%`, `v1` `PR >= 0.880 / R@P95 >= 63%`, `Wave-3` Promo `AUROC >= 0.935 / AP >= 0.885`, `v3-real` Cascade `exit >= 45%` (`<= 1.6 ms` 4T, `certified_ltt == True`).
