# v2.7 PSC-BS — Candidate-Count Sensitivity (Audited)

This is a **post-confirmatory sensitivity analysis**. The frozen v2.7 method, the preregistered M=16 fresh CPU confirmation, and the fresh GPU confirmation remain unchanged.

We regenerate deterministic prefixes of the same frozen 16-draw proposal stream at $M\in\{1,2,4,8,16\}$, rerun the **unchanged** PSC-BS certification rule, and evaluate the resulting deployed context with ExtraTrees, XGBoost, LightGBM, and MLP. No threshold tuning is performed.

## Integrity

- 900 decision rows = 180 context cells × 5 candidate counts.
- 3,600 held-out evaluation rows = 180 context cells × 4 downstream CPU models × 5 candidate counts.
- At M=16, proposal selection reproduces the frozen candidate in **180/180** context cells.
- At M=16, certification decisions reproduce the frozen v2.7 decision in **180/180** context cells.
- Held-out model evaluation uses one common **sensitivity-analysis seed per dataset/split/budget/model across all M**. Therefore the M=16 downstream metrics are an internally controlled sensitivity replicate, not a byte-for-byte reproduction of the preregistered fresh-CPU metric table. This is intentional: using a common seed prevents model stochasticity from confounding comparisons across M.

## Main result

| Candidate count M | Acceptance | Same selected candidate as M=16 | ΔEO² vs anchor | ΔAUC vs anchor | ΔLogLoss vs anchor | Severe AUC tail | Datasets ΔEO²<0 | Models ΔEO²<0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.11% | 6.67% | -0.001730 | +0.000879 | -0.002136 | 0.278% | 5/9 | 4/4 |
| 2 | 15.00% | 11.67% | -0.001175 | +0.002442 | -0.003200 | 0.139% | 6/9 | 4/4 |
| 4 | 15.56% | 18.33% | -0.001451 | +0.001719 | -0.002244 | 0.278% | 6/9 | 4/4 |
| 8 | 22.22% | 45.00% | -0.001687 | +0.002243 | -0.003546 | 0.417% | 8/9 | 4/4 |
| 16 | 16.11% | 100.00% | -0.001517 | +0.000743 | -0.002745 | 0.556% | 8/9 | 4/4 |

## Interpretation

- Proposal count materially changes the upstream search result: with M=1 only **6.67%** of selected candidates match the M=16 choice; with M=8 the match rate is **45.0%**.
- Despite materially different selected candidates, PSC-BS keeps the observed severe-tail rate below **0.6%** for every tested M on the four held-out classical model families.
- Acceptance is not monotone in M (11.1%, 15.0%, 15.6%, 22.2%, 16.1%). This is expected because selection quality and the independent benefit/safety certificate interact discretely.
- There is no empirical evidence that increasing search capacity monotonically improves fairness or utility. The correct interpretation is that **M controls proposal/search capacity**, while the independent certification/fallback layer controls deployment risk.
- The frozen M=16 value is retained exactly as preregistered. These post-hoc sensitivity curves are robustness evidence, not a basis for retuning M.

## Scope

- This candidate-count sensitivity is evaluated on the four held-out classical downstream families. Alternative prefix-selected candidates were not part of the preregistered TabPFN/TabICL GPU lattice.
- The frozen M=16 TabPFN 3.5 + TabICL v2 fresh-GPU experiment remains the main unseen-TFM confirmation.
- New TFM inference for every post-hoc M is not needed for the current paper unless a reviewer specifically requests candidate-count curves on foundation models.