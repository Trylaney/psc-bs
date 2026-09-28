# ContextBench v2.7 Freeze Report — PSC-BS

## Decision

Freeze **Post-Selection Benefit-Safety Certification (PSC-BS)** before any fresh v2.7 confirmation data are inspected.

The method is now a selector-agnostic wrapper rather than a claim that one heterogeneous proposal bank always dominates random search.

## Five-way protocol

For each dataset/split:

1. **P — context pool** (3000)
2. **C — proposal calibration** (500; optional for the base selector)
3. **V — selection** (500)
4. **W — independent post-selection certification** (1000)
5. **T — untouched test** (2000)

The base selector and safety anchor are fully fixed before W is revealed.

## Frozen certificate

Certifier ensemble: Logistic Regression, HistGradientBoosting, Random Forest.

For selected context S* and safety anchor A, accept S* only if:

- every certifier has ΔAUC(S*, A) >= -0.01;
- every certifier has ΔLogLoss(S*, A) <= +0.05;
- the mean certifier Δsmooth-EO²(S*, A) is strictly negative.

Otherwise deploy A exactly.

No coverage gate and no bootstrap confidence-bound gate are part of the frozen main method.

## Why this rule was selected

On the old ACS-2022 development suite, using an independent W split:

| Development rule | Accept rate | CPU held-out mean ΔEO² vs anchor | GPU-old dev fair win vs random5 mean | GPU-old dev guarded win | severe AUC tail |
|---|---:|---:|---:|---:|---:|
| Utility-only worst-model point guard | 34.3% | +0.000193 | 70.0% | 59.6% | 0.71% |
| Utility + coverage | 14.3% | +0.000019 | 66.8% | 56.4% | 0.36% |
| Simultaneous bootstrap LCB (90% FWER prototype) | **3.6%** | ~anchor | ~anchor | ~anchor | ~anchor |
| **Utility + independent fairness-benefit gate (PSC-BS)** | **22.1%** | **-0.000102** | **70.0%** | **60.0%** | **1.07%** |

Interpretation:

- the point utility guard has useful acceptance but can accept candidates whose fairness gain fails to transfer to held-out CPU models on average;
- coverage is too conservative and contributes little beyond falling back to the anchor;
- the simultaneous bootstrap confidence rule is currently pathological at W=1000 (5/140 accepted) and effectively collapses to the anchor;
- the zero-margin fairness-benefit gate is threshold-free, keeps acceptance just above 20%, reverses held-out mean fairness to the intended direction, and preserves strong old-GPU development behavior.

All GPU numbers above are **development replay only** because W was carved from the old GPU test population after v2.6. They are not v2.7 confirmation evidence.

## Novelty boundary

Do not claim first demonstration validation, first cross-model validation, first fairness-aware ICL selection, or first coverage-based selection. Closest prior lines include D.Va (ACL 2025) demonstration validation and Fair-TabICL (TMLR 2026) fairness interventions for tabular foundation-model ICL.

The defensible contribution is the problem/method combination:

> fairness-aware context search under an unavailable/unknown downstream learner, followed by an independent post-selection benefit-and-utility safety check against a predeclared anchor, with fallback and explicit evaluation of left-tail utility transfer to unseen downstream learners.

## Fresh confirmation (locked)

Do not use ACS-2022 AZ/GA/MA/NC/WA again for v2.7 confirmation.

Fresh CPU confirmation:

- ACS 2023, states CO/MI/MN/NJ/OR;
- seven frozen ACS prediction tasks;
- Law School bar-passage dataset;
- UCI Diabetes 130-US Hospitals;
- 5 outer splits;
- context budgets 32/64/128/256;
- certifiers fixed to Logistic/HistGB/RF;
- held-out downstream CPU learners fixed to ExtraTrees/XGBoost/LightGBM/MLP.

Only after CPU confirmation passes should a new TabPFN 3.5 / TabICL v2 GPU confirmation be rented.

## Confirmatory gates

Primary gates (predeclared):

1. complete run, zero task errors;
2. acceptance rate 10–60% (guards against a trivial always-fallback method);
3. on held-out CPU learners, mean incremental Δsmooth-EO² vs safety anchor < 0;
4. mean incremental ΔAUC vs safety anchor >= -0.005;
5. mean incremental Δlogloss vs safety anchor <= +0.01;
6. P(incremental ΔAUC < -0.05) <= 1%;
7. against the mean of five prevalence-random contexts: fairness win >= 60%, guarded win >= 50%, mean ΔAUC >= -0.005;
8. gains must not be attributable to only one dataset or one held-out model family.

The gates are not to be changed after fresh results are inspected.
