# v2.7 PSC-BS — Zero-Cost GPU Ablation Audit

These ablations are reconstructed from the frozen certification metadata and the already-completed fresh GPU evaluations. No new GPU inference and no threshold tuning are used.

## Main table

| Policy | Accept | ΔEO² vs anchor | ΔAUC vs anchor | ΔLogLoss vs anchor | Severe tail | Fair win vs random5 | Guarded win |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full PSC-BS | 16.1% | -0.000513 | +0.001208 | -0.002410 | 0.278% | 65.00% | 51.67% |
| Uncertified candidate / no fallback | 100.0% | -0.002451 | -0.004297 | -0.008518 | 7.778% | 78.06% | 55.00% |
| Utility-only certificate | 31.7% | -0.000426 | +0.003150 | -0.005188 | 0.278% | 64.17% | 51.67% |
| Benefit-only certificate | 55.6% | -0.002508 | -0.003313 | -0.005157 | 5.000% | 77.78% | 57.22% |
| Single Logistic certifier | 31.7% | -0.000765 | +0.001794 | -0.004286 | 1.389% | 67.22% | 53.61% |
| Single HistGB certifier | 31.1% | -0.001525 | +0.002203 | -0.004777 | 0.833% | 69.72% | 55.56% |
| Single RandomForest certifier | 34.4% | -0.001056 | +0.003062 | -0.004737 | 1.111% | 68.06% | 53.61% |

## Interpretation

- **Utility guard is the key tail-control component.** Removing it (benefit-only certification) raises severe-tail incidence from 0.278% to 5.000%.
- **Independent certification/fallback is essential.** Always deploying the V-selected candidate raises severe-tail incidence to 7.778%.
- **Heterogeneous certification matters.** Single-certifier variants have severe-tail incidence between 0.833% and 1.389%, versus 0.278% for the three-certifier PSC-BS.
- **The fairness-benefit gate is not the main safety mechanism.** Utility-only certification retains the same observed 0.278% severe-tail rate, but yields a smaller aggregate fairness improvement than full PSC-BS.

## Important equivalence

In the frozen v2.7 design there is exactly one selected candidate per context cell. Therefore **'no independent certification / always deploy the selected candidate'** and **'no safety-anchor fallback after rejection'** collapse to the same deployed policy. Reporting them as two separate numeric rows would duplicate the same experiment; this should be stated explicitly rather than padded as two ablations.

## What still requires new computation

- Candidate-count sensitivity (the unselected draws were not persisted as contexts).
- Certification-sample-size sensitivity (requires rerunning cheap certifiers on deterministic subsets of W; no new TabPFN/TabICL inference is necessary if candidate/anchor contexts stay fixed).
- A selector-swap / heterogeneous-proposal experiment, if desired to demonstrate PSC-BS wrapper generality. This is conceptually separate from the frozen fresh-confirmation result.