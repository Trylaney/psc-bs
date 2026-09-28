# v2.7 PSC-BS — Certification Sample-Size Sensitivity

This is a **post-confirmatory sensitivity analysis**. The frozen v2.7 method and its fresh CPU/GPU confirmation remain unchanged. We rerun only the cheap Logistic + HistGB + RandomForest certification stage on deterministic stratified subsets of the independent certification split W, then reuse the already-computed frozen fresh-GPU candidate/anchor outcomes. No new TabPFN/TabICL inference and no threshold tuning are used.

## Main result

| Certification n | Acceptance | ΔEO² vs anchor | ΔAUC vs anchor | ΔLogLoss vs anchor | Severe AUC tail | Fairness win vs random5 | Guarded win |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 11.67% | -0.000431 | +0.000496 | -0.001658 | 0.556% | 63.89% | 51.11% |
| 250 | 15.56% | -0.000727 | +0.001353 | -0.003006 | 0.278% | 65.28% | 51.11% |
| 500 | 13.33% | -0.000378 | +0.000718 | -0.001739 | 0.000% | 62.78% | 49.17% |
| 1000 | 16.11% | -0.000513 | +0.001208 | -0.002410 | 0.278% | 65.00% | 51.67% |

## Interpretation

- The full-W setting (`n=1000`) reproduces the frozen PSC-BS certification decisions **180/180 exactly**.
- Downside control is not dependent on using the maximum certification split: severe-tail incidence remains at or below 0.556% across `n=125/250/500/1000`.
- `n=125` is visibly noisier: acceptance falls to 11.67% and the aggregate fairness gain weakens.
- `n=250–1000` is broadly stable. There is no monotonic gain from increasing `n`; the certification decision is discrete and depends on both utility and fairness gates.
- `n=500` happened to produce zero severe GPU tails in this finite sample, but this is a sensitivity observation, **not** a reason to retune the frozen method away from `n=1000`.
- The paper should therefore present `n=1000` as the preregistered confirmation setting and the smaller values only as robustness/sensitivity evidence.

## Recommended paper wording

> PSC-BS remains qualitatively stable when the independent certification sample is reduced from 1,000 to 250 examples. At 125 examples the gate becomes more conservative and aggregate fairness benefit weakens, consistent with increased certification noise. We retain the preregistered 1,000-example certification split and do not tune the method to the post-hoc sensitivity results.
