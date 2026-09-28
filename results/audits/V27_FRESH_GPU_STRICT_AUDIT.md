# ContextBench v2.7 Fresh GPU — Strict Independent Audit

## Verdict

**PASS. All 9 frozen preregistered GPU gates pass under independent recomputation from the 2880 task results.**

- 2880/2880 tasks OK; 0 errors
- 360 paired cells
- 9 datasets × 5 outer splits × 4 budgets × 2 TFMs × 8 selectors
- PSC-BS acceptance: 29/180 context cells = 16.111%
- Frozen PSC-BS SHA256: `f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d`

## Aggregate PSC-BS vs safety anchor

- mean Δsmooth-EO² = -0.0005128490
- mean ΔAUC = +0.0012081202
- mean Δlogloss = -0.0024098224
- severe tail P(ΔAUC < -0.05) = 0.278% (1/360)

Raw candidate (`valfair_random_search16`) severe-tail rate relative to the same anchor:
- 7.778% (28/360)

PSC-BS therefore reduces the observed severe-tail rate by:
- 7.500 percentage points
- 96.43% relative at the point estimate
- paired-cell bootstrap 95% CI for risk difference: [5.000, 10.278] pp
- dataset-cluster bootstrap 95% CI: [5.278, 9.444] pp

## Against mean of five prevalence-random contexts

- fairness win = 65.000%
- guarded win = 51.667%
- mean Δsmooth-EO² = -0.0003604042
- mean ΔAUC = +0.0100305007
- mean Δlogloss = -0.0098918508

## Bootstrap

Primary safety-anchor fairness effect:
- paired-cell 95% CI for ΔEO²: [-0.0008366312, -0.0002307194]
- dataset-cluster 95% CI: [-0.0008769358, -0.0002021766]

Safety-anchor ΔAUC:
- paired-cell 95% CI: [+0.0001019938, +0.0024091847]
- dataset-cluster 95% CI: [-0.0001372005, +0.0025401328]

Random5 point gates pass, but their dataset-cluster CIs cross the preregistered 60% / 50% thresholds:
- fairness win cluster CI: [57.78%, 72.78%]
- guarded win cluster CI: [43.06%, 60.56%]

This does not invalidate the preregistered gate, which was defined on the point estimate, but it means the strongest paper claim should emphasize anchor-relative safety rather than thresholded random5 win rates.

## Single deployed severe-tail miss

There is exactly one deployed PSC-BS cell with ΔAUC < -0.05:

- dataset: law_school
- outer split: 4
- budget: 32
- downstream model: tabiclv2
- accepted candidate: True
- ΔAUC = -0.066069
- Δsmooth-EO² = +0.009451
- Δlogloss = +0.081882

This is a genuine unseen-model certification miss and should be disclosed. PSC-BS strongly controls, but does not eliminate, severe utility tails.

## Integrity checks

- Exact 2880-task lattice present; no missing or duplicate task combinations.
- All task statuses are `ok`.
- 2880 task JSON records agree with `GPU_RESULTS.csv` on scalar result fields.
- Independent paired reconstruction agrees with packaged paired-cell analysis to floating-point precision.
- Frozen PSC-BS source SHA256 recomputed from the preregistered runner matches the declared hash.
- Frozen context-manifest SHA256 matches the preregistered protocol.
- Certification metadata uses Logistic + HistGB + RandomForest, AUC tolerance 0.01, logloss tolerance 0.05, fairness margin 0.0; no certification-rule inconsistency found.

## Paper-facing conclusion

The evidence supports:

> Independent post-selection benefit-safety certification with a safety-anchor fallback substantially reduces unseen-model severe utility-tail incidence while retaining aggregate fairness improvements and near-neutral-to-positive utility across TabPFN 3.5 and TabICL v2.

Do not claim complete elimination of catastrophic tails.
