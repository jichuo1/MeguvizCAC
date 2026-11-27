# MeguvizCAC v1.2.2-ApexNano-Ultra — Post-Training Capability Leap (`ADR-v1.2.2`)

- **Version**: `v1.2.2-r1-apexnano-b7`
- **Hard Deployment Contract**: `<= 2.20 MB` Bundle v7 (`Sub-2bit QAT` + `100%` ABI-compatible 5-Stage `FP32-Exact` ONNX), `Zero-PII diff === 0.000000`, `F6 Promo→AI Firewall === 0.0`, real-time CPU cascade `<= 1.6 ms (4T) / <= 2.5 ms (1T)`.
- **Data Merge**: Integrates **`v122-sibling-datafactory`** (`203,209` 4-way zero-leak verified rows: `99,581` dual-axis gold incl. `500` `god_tier_500_holdout` never trained + `40,994` Pack2-B verified humans + `26,955` Pack1-A AI-only + `16,652` Pack3-D thread humans + `19,027` Pack3-C counterfactual mirrors & 4,824 exact-span sandwich hybrids + `26,463` `synth_hygiene_actions`) alongside `Gold-v2`, `Wave-3`, `Wave-2.5`, and `v1.1` dual-axis pools.

## Eight P0 Post-Training Upgrades (All `0 B` Extra Runtime Size)

1. **P0-1 · Output-Shifted Promo Beta-Spline LUT + Multi-Rung Promo Human Ceiling**:
   - Fixes `train_v121_apexnano.py:277` sign-inversion bug (fits unshifted `z`, applies `prior_shift < 0` to LUT **output `y`**) + `CVaR-DORO` human promo ceiling on `r0_promo / r1_promo / t2_promo`, restoring `>= 45%` early exit (`~1.45 ms`) & `human_promo_max <= 0.05`.
2. **P0-2 · Dual-Source Surgical Re-Base (`v1.0.0-R1` AI `[0:192]` + `v1.2.1-R1` Promo/R0/LongStream `[192:256]`)**:
   - Exact `OBS-perm + Net2Net` chain (`v1.0 → v1.1 → v1.2.2-dense`) + graft of `v1.2.1` `rung0 / z_char_* / promo_* / longstream / emb[192:256] / ebt_row5`, guarded by `V122_G0` (`RMSE <= 0.02`, automatic 300-step `TR-KD` heal if `> 0.05`).
3. **P0-3 · 5-Era Invariance Stack (`OfflineEraGRLHead(192→5)` + `CCR Era-IPW` + 5-Era `GroupDRO` + Full-Confounder Wild-3% `LEACE`)**:
   - Maps all rows to real 5 era buckets (`pre_llm, 2023, 2024, 2025_gt1y, 2026`), eliminates `v3-hard` `>1y` old-video top-FP dominance (`75%` of `v1.2.1` top-8 FPs were `1.6–4.3 yr` old!), enforced by `V122_E1_era_probe <= 0.56`.
4. **P0-4 · Partitioned 16k Hard-Human Memory `𝓜 = 𝓜_phys ∪ 𝓜_modern` + `SOPA-s` pAUC-DRO + `CVaR-DORO` + `v1.0`-Ref DPO**:
   - Untrimmed `top-8` ceiling on `𝓜_phys` (`pre_llm` 0% AI contamination), DORO-trimmed (`skip top-4`) ceiling on `𝓜_modern` (`>1y` & `2026` verified humans), `SOPA-s` one-way pAUC-DRO (`λ=0.5`), `v1.0`-referenced DPO on `pack3_C` `14,203` counterfactual mirrors & `𝓜`, applied across `t2 + t1 + r1 + QAT`.
5. **P0-5 · EBT Anchored Energy-Barrier (`RCE+EMR` on `𝓜`) + `SoftBall` `wd=0` / `||w||_2` Cap / `5·tanh(mil/5)` + `ChronoStream` Span Supervision**:
   - Wires EBT CE + `RCE/EMR` (`m_out=+2.5, m_in=-2.5`, `0.02·LSE(-E)^2`) into Phase A & C strictly `has_ai > 0.5` masked; splits `weight_decay=0.0` on `SoftBall.center/log_r_cap/norms/biases`; caps `mil_peak`; feeds live per-chunk `z_char` & exact spans into `ChronoStream`.
6. **P0-6 · Anti-Drift Sibling-Data Ingestion Curriculum (`FLOW` + A0 Freeze + `<=3.5ep` Cap + One-Sided Human Drift Barrier + `ASAM` + `EMA`/`WiSE-FT`)**:
   - Subphase A0 (`250 steps`, trunk+core frozen, `FLOW` `τ: 0.8→4.0`) → Subphase A1 (`950 steps`, `<=3.5ep` cap, `35–45%` replay floor, One-Sided Human Drift Guard, Phase-A `ASAM`, hoisted same-batch Phase-B `ASAM`, live `PiKE.update()`, 100-step `EMA(0.999)` + `WiSE-FT`).
7. **P0-7 · `QuEST` + `BitDistill` + `TR-KD` + Post-Quant Convex Soup + Surgical Shock-Absorber + `AdaRound-Lite` (`Phase C`)**:
   - `QuEST` Trust Mask (`T=1.5·scale` `trit158`, `T=3.5·scale` `seq2`), `BitDistill` `16×16` core slot-attention relation KL (`<=0.05`), `TR-KD` (`LS-KD + tail-pw + RankDistil + CAKLD + L_pAUC + L_ceil + T1-KD + 64 𝓜 rows/step`), post-quant convex soup `α·Q1+(1-α)·Q2`, surgical Shock-Absorber (frozen `W8 rung0`, WSD `5e-5`), constant-byte tail-Fisher HAWQ + bit-exact `AdaRound-lite`.
8. **P0-8 · Native `5-Head Conservative SoftMin` (`z5_softmin`) on Disjoint `val_meta_cal` + `5-Way` Validation Partition + 9-Gate CI (`V122_*`)**:
   - Strictly disjoint `val_select / val_meta_cal (incl. god_tier_500_holdout) / val_cal_cert (real-wild only!) / val_promo_select / val_promo_cert`; `B=200` wild-3% bootstrap for `z5_softmin`; `iso_blend=0.30`; `V122_G0/S1/F6/C1/P1/R0/L1/E1/PR_LUT_FLOOR`.
