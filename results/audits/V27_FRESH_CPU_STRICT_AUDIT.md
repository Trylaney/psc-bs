# ContextBench v2.7 Fresh CPU Confirmation — Strict Audit

**结论：PASS。可以正式标记 `v2.7 fresh CPU confirmation PASSED`，并进入 frozen Fresh GPU confirmation。**

## 1. 硬完整性

- task JSON: 5760/5760；status=ok: 5760；parse error=0；missing/duplicate/unexpected combinations = 0/0/0。
- datasets: 9/9。
- frozen method SHA256: `f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d`，与预注册值完全一致。
- context cells: 180/180；逐格 certificate/fallback/索引完整性问题 = 0。
- PSC-BS certified contexts: 29/180 = 16.11%。

## 2. Aggregate effect

| Comparison | Δsmooth-EO² | ΔAUC | Δlogloss | severe AUC tail | fairness win | guarded win |
|---|---:|---:|---:|---:|---:|---:|
| vs safety anchor | -0.001200 | +0.000780 | -0.003554 | 0.42% (3/720) | — | — |
| vs random5 mean | -0.000535 | +0.011063 | -0.013533 | 0.56% | 64.44% | 52.36% |

## 3. Frozen gates

- gate1_complete_zero_error: PASS
- gate2_acceptance_10_60: PASS
- gate3_anchor_mean_deo2_lt0: PASS
- gate4_anchor_mean_dauc_ge_m0005: PASS
- gate5_anchor_mean_dll_le_p001: PASS
- gate6_tail_le_1pct: PASS
- gate7_vs_random5: PASS
- gate8_not_single_dataset_or_model: PASS

All 8 gates: **PASS**.

## 4. Bootstrap 95% CI

10,000 次 deterministic bootstrap，seed=20260927。`cell bootstrap` 重采样 720 paired cells；`dataset-cluster bootstrap` 以 9 个 dataset 为完整 cluster 重采样。

| Metric | Point | cell bootstrap 95% CI | dataset-cluster 95% CI |
|---|---:|---:|---:|
| anchor_dEO2 | -0.001200 | [-0.002400, -0.000345] | [-0.002581, -0.000332] |
| anchor_dAUC | +0.000780 | [-0.000122, +0.001731] | [-0.000367, +0.001797] |
| anchor_dLogLoss | -0.003554 | [-0.005851, -0.001286] | [-0.006429, -0.000949] |
| anchor_tail | +0.004167 | [+0.000000, +0.009722] | [+0.000000, +0.009722] |
| random5_fair_win | +0.644444 | [+0.609722, +0.679167] | [+0.587500, +0.708333] |
| random5_guarded_win | +0.523611 | [+0.487500, +0.561111] | [+0.455556, +0.594444] |
| random5_dEO2 | -0.000535 | [-0.001940, +0.001058] | [-0.000852, -0.000233] |
| random5_dAUC | +0.011063 | [+0.009383, +0.012780] | [+0.009175, +0.013229] |
| random5_dLogLoss | -0.013533 | [-0.016379, -0.010791] | [-0.017655, -0.009264] |

Acceptance rate 95% CI: cell [11.11%, 21.67%]；dataset-cluster [8.89%, 23.33%]。

**解释边界：** 预注册 gate 是 point-estimate gate，因此 PASS 不受 bootstrap CI 是否跨阈值影响。dataset-cluster CI 中 random5 fairness-win 和 guarded-win 的下界分别低于 60%/50%，说明跨数据集不确定性仍应如实报告，不能把 aggregate gate PASS 写成“所有数据集都稳定超过阈值”。相对 safety anchor 的 mean ΔEO² 的 dataset-cluster CI 全部位于 0 以下。

## 5. Stratification — dataset

| dataset | accept | anchor ΔEO² | anchor ΔAUC | anchor ΔLL | anchor tail | random fair | random guarded | random ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acs23_income | 25.00% | -0.000802 | +0.002740 | -0.006319 | 0.00% | 65.00% | 62.50% | +0.011252 |
| acs23_employment | 15.00% | -0.000064 | +0.002204 | -0.002747 | 0.00% | 81.25% | 70.00% | +0.014425 |
| acs23_public_coverage | 5.00% | -0.000056 | -0.000059 | -0.006719 | 0.00% | 71.25% | 62.50% | +0.012162 |
| acs23_mobility | 30.00% | -0.000348 | -0.002625 | -0.003893 | 2.50% | 75.00% | 53.75% | +0.009698 |
| acs23_travel_time | 10.00% | -0.000888 | +0.000610 | -0.012366 | 0.00% | 62.50% | 50.00% | +0.008632 |
| acs23_health_insurance | 10.00% | -0.006304 | +0.002143 | +0.001856 | 0.00% | 56.25% | 50.00% | +0.017436 |
| acs23_income_poverty | 15.00% | -0.001335 | -0.000055 | +0.001526 | 0.00% | 53.75% | 35.00% | +0.010413 |
| law_school | 35.00% | -0.001022 | +0.002376 | -0.002992 | 1.25% | 62.50% | 48.75% | +0.009092 |
| diabetes_hospital | 0.00% | +0.000016 | -0.000317 | -0.000329 | 0.00% | 52.50% | 38.75% | +0.006461 |

## 6. Stratification — budget

| n | accept | anchor ΔEO² | anchor ΔAUC | anchor ΔLL | anchor tail | random fair | random guarded | random ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32.0 | 13.33% | -0.000593 | +0.001583 | -0.004970 | 0.56% | 63.89% | 55.00% | +0.022045 |
| 64.0 | 13.33% | -0.003072 | +0.000697 | -0.003821 | 0.56% | 64.44% | 51.11% | +0.010771 |
| 128.0 | 15.56% | -0.000356 | +0.000901 | -0.003580 | 0.00% | 70.00% | 58.33% | +0.009035 |
| 256.0 | 22.22% | -0.000779 | -0.000063 | -0.001843 | 0.56% | 59.44% | 45.00% | +0.002403 |

## 7. Stratification — held-out model

| model | accept | anchor ΔEO² | anchor ΔAUC | anchor ΔLL | anchor tail | random fair | random guarded | random ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extra_trees | — | -0.000848 | +0.000552 | -0.001009 | 0.56% | 69.44% | 56.67% | +0.009595 |
| xgboost | — | -0.000269 | +0.001277 | -0.001515 | 0.00% | 58.89% | 48.89% | +0.011654 |
| lightgbm | — | -0.000374 | +0.001653 | -0.003535 | 0.00% | 61.67% | 53.33% | +0.013037 |
| mlp | — | -0.003310 | -0.000363 | -0.008156 | 1.11% | 67.78% | 50.56% | +0.009968 |

## 8. Severe tail cells vs safety anchor

| dataset | outer | n | model | ΔEO² | ΔAUC | Δlogloss |
|---|---:|---:|---|---:|---:|---:|
| acs23_mobility | 1 | 64 | extra_trees | +0.001153 | -0.059790 | +0.015132 |
| acs23_mobility | 2 | 256 | mlp | -0.004796 | -0.051536 | +0.067444 |
| law_school | 0 | 32 | mlp | +0.000195 | -0.072319 | +0.023457 |

There are exactly 3/720 severe tails. Two occur on ACS mobility and one on Law School; none are hidden by averaging.

## 9. Audit notes

- `diabetes_hospital` has 0% PSC-BS acceptance and a tiny positive mean ΔEO² vs anchor (+0.000016), while 8/9 datasets have negative mean ΔEO² vs anchor. This still satisfies the frozen anti-single-dataset gate but should remain visible in the paper.
- Budget 256 is the weakest random5 stratum (fairness win 59.44%, guarded win 45.00%), despite aggregate gate PASS. This is a reviewer-relevant reason to keep the preplanned candidate-count / certification-sample-size sensitivity analysis, not a reason to retune v2.7.
- All four held-out CPU model families have negative mean ΔEO² vs anchor; MLP has mean ΔAUC -0.000363 vs anchor, still comfortably above the frozen -0.005 gate.

## 10. Freshness / package identity

The currently uploaded `V27_FRESH_CPU_RESULTS(3).zip` is byte-for-byte identical to the previously frozen `(2)` copy used to construct the GPU package. Both SHA256:

`4d61dbf5f4eac28930cbdf7bed3e1bfa98b79853ae6597ced344c6a03f6ebdfe`

