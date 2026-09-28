# Matched-Acceptance Diagnostic

This is an **exploratory post-confirmatory diagnostic**, not a frozen gate and not a new confirmatory experiment.

PSC-BS accepts 29/180 context cells (16.11%). Because any low-acceptance policy mechanically reduces deployed tail events by falling back to the anchor, this diagnostic asks whether **which** contexts PSC-BS accepts is informative beyond the acceptance rate itself.

We repeatedly choose 29 of the 180 context cells uniformly at random, deploy the raw selected candidate on those cells, and deploy the same safety anchor elsewhere. The already-computed TabPFN-3.5 and TabICLv2 outcomes are reused; no new model inference is performed.

- PSC-BS severe-tail rate: **0.278%** (1/360).
- Matched-random gate mean severe-tail rate over 100,000 draws: **1.254%**.
- Matched-random central 95% Monte Carlo range: **[0.278%, 2.500%]**.
- Fraction of matched-random draws with a tail count no larger than PSC-BS: **0.086**.

Mean effects versus anchor:
- PSC-BS: ΔEO²=-0.000513, ΔAUC=+0.001208, Δlogloss=-0.002410.
- Matched-random gate: ΔEO²=-0.000396, ΔAUC=-0.000694, Δlogloss=-0.001371.

Interpretation: low coverage explains part of the tail reduction mechanically, but a same-coverage random gate has a substantially larger expected tail rate and worse mean AUC. This supports the certificate being informative rather than merely selective. Because this diagnostic was added after seeing the confirmation results, it should be presented as **mechanistic evidence**, not as a confirmatory hypothesis test.
