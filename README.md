# PSC-BS: Post-Selection Certification for Safe Context Selection under Downstream-Model Uncertainty

This repository is the reproducibility artifact for the manuscript **Post-Selection Certification for Safe Context Selection under Downstream-Model Uncertainty**.

**Author:** Tao Ran, Anhui University  
**ORCID:** 0009-0002-4449-8982

## Frozen method

PSC-BS v2.7 was frozen before fresh confirmation. The canonical method source has SHA256:

`f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d`

Both `frozen_cpu/v27_psc_method.py` and `frozen_gpu/v27_psc_method.py` must match this digest.

## Confirmatory evidence

- Fresh CPU: 5760/5760 tasks, 0 errors, all frozen CPU gates passed.
- Fresh GPU: 2880/2880 tasks, 0 errors, all frozen GPU gates passed.
- GPU severe AUC tail relative to the safety anchor:
  - uncertified selected candidate: 28/360 = 7.778%
  - PSC-BS deployed policy: 1/360 = 0.278%
- The remaining unseen-model miss is retained in the artifact and manuscript.

The compact exact result archives are in `results/`. The strict independent audits and post-confirmatory sensitivity analyses are in `results/audits/`.

## Repository layout

- `frozen_cpu/` - exact CPU fresh-confirmation source and frozen metadata.
- `frozen_gpu/` - exact GPU fresh-confirmation source, frozen contexts, protocol, and analysis code. Runtime caches and model weights are intentionally excluded.
- `results/` - exact compressed CPU/GPU confirmation results.
- `results/audits/` - strict audits, ablations, sensitivity summaries, and matched-acceptance diagnostic.
- `environment/` - dependency lists used by the frozen runners.
- `paper/` - manuscript snapshot.
- `scripts/verify_artifact.py` - offline integrity checker.
- `MANIFEST_SHA256.txt` - SHA256 manifest for released files.

## Quick integrity check

```bash
python scripts/verify_artifact.py
```

Expected headline output:

```text
PSC-BS frozen source hash: PASS
Frozen GPU contexts: 180
Fresh CPU summary: PASS
Fresh GPU summary: PASS
ARTIFACT VERIFICATION PASS
```

## CPU reproduction

The exact CPU runner is preserved in `frozen_cpu/`.

```bash
cd frozen_cpu
bash RUN_ONE_CLICK.sh
```

This creates a virtual environment, verifies the freeze, retrieves/prepares the public datasets, runs the fresh CPU lattice, analyzes it, and packages the results. Network availability and upstream dataset hosting can affect automated retrieval.

## GPU reproduction

Install a CUDA-enabled PyTorch environment compatible with the local GPU, then install:

```bash
pip install -r environment/requirements-gpu.txt
```

TabPFN 3.5 model access may require accepting the upstream model provider's terms. Model checkpoints are **not redistributed** in this artifact. The paper records the exact checkpoint hashes used in confirmation.

After data preparation and model availability:

```bash
cd frozen_gpu
bash RUN_GPU_SMOKE.sh
bash RUN_GPU_FINAL_AUTO.sh
```

The frozen GPU runner replays the 180 CPU-selected context cells. It does not reselect or recertify contexts.

## Data and model-weight policy

Raw benchmark data and foundation-model weights are not redistributed here. The repository includes download/preparation code, frozen context indices, provenance metadata, and exact hashes needed to audit the run. Users remain responsible for the original datasets' and model providers' terms.

## Post-confirmatory analyses

Ablations, candidate-count sensitivity, certification-size sensitivity, and the matched-acceptance diagnostic are clearly labeled as post-confirmatory where appropriate. They do not modify the frozen v2.7 method or its preregistered confirmation gates.

## License

A software license has not yet been selected for public release. See `LICENSE_PENDING.md` before publishing the repository.
