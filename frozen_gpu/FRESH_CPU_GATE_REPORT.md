# v2.7 Fresh CPU Confirmation — Automatic Gate Report

Tasks: 5760/5760 OK, expected=5760, errors=0
Datasets available: 9/9; bad dataset/split statuses: 0
Context cells: 180, acceptance rate: 16.111%

## Aggregate vs safety anchor
- mean Δsmooth-EO²: -0.001200
- mean ΔAUC: +0.000780
- mean Δlogloss: -0.003554
- P(ΔAUC < -0.05): 0.417%

## Aggregate vs mean of five prevalence-random contexts
- fairness win: 64.444%
- guarded win: 52.361%
- mean ΔAUC: +0.011063

## Frozen gates
- gate1_complete_zero_error: PASS
- gate2_acceptance_10_60: PASS
- gate3_anchor_mean_deo2_lt0: PASS
- gate4_anchor_mean_dauc_ge_m0005: PASS
- gate5_anchor_mean_dll_le_p001: PASS
- gate6_tail_le_1pct: PASS
- gate7_vs_random5: PASS
- gate8_not_single_dataset_or_model: PASS

Overall automatic verdict: PASS

Note: gate 8 is operationalized conservatively as improvement direction appearing in at least two datasets and two held-out model families; inspect by_dataset.csv/by_model.csv as well.
