#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /etc/network_turbo 2>/dev/null || true
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache}"
python DOWNLOAD_FRESH_DATA.py
python PREPARE_FRESH_DATA.py
python PRECHECK_V27_GPU.py
rm -rf runs/v2_7_gpu_smoke
CUDA_VISIBLE_DEVICES=0 python v27_gpu_fresh.py --mode smoke --output runs/v2_7_gpu_smoke
python MERGE_GPU_RESULTS.py --input runs/v2_7_gpu_smoke
cat runs/v2_7_gpu_smoke/GPU_MERGED_MANIFEST.json
