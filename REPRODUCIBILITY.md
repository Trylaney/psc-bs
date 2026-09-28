# Reproducibility notes

## What is confirmatory

The v2.7 CPU and GPU fresh-confirmation protocols were frozen before their corresponding fresh results were produced. Their method implementation, contexts, gate definitions, and result archives are preserved under `frozen_*` and `results/`.

## What is post-confirmatory

The following are mechanism/sensitivity analyses and should not be described as preregistered confirmation:

- utility-only / benefit-only / single-certifier ablations;
- certification-sample-size sensitivity;
- candidate-count sensitivity;
- matched-acceptance random-gate diagnostic.

## Expected primary summaries

Fresh CPU:
- 5760 task records, 0 errors;
- 180 context cells / 720 held-out model cells;
- acceptance = 29/180 = 16.111%;
- mean delta smooth-EO^2 vs anchor = -0.0012002923;
- mean delta AUC vs anchor = +0.0007796321;
- mean delta logloss vs anchor = -0.0035537568.

Fresh GPU:
- 2880 task records, 0 errors;
- 360 paired model cells;
- mean delta smooth-EO^2 vs anchor = -0.0005128489603;
- mean delta AUC vs anchor = +0.0012081202105;
- mean delta logloss vs anchor = -0.0024098223726;
- deployed severe AUC tail = 1/360 = 0.2778%;
- raw-candidate severe AUC tail = 28/360 = 7.7778%.

## Exact model hashes used in fresh GPU confirmation

- TabPFN 3.5 checkpoint SHA256: `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`
- TabICLv2 checkpoint SHA256: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`

Model weights are not included.
