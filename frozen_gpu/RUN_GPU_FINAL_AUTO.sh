#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /etc/network_turbo 2>/dev/null || true
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache}"
python DOWNLOAD_FRESH_DATA.py
python PREPARE_FRESH_DATA.py
python PRECHECK_V27_GPU.py
N="${V27_GPU_COUNT:-$(nvidia-smi -L | wc -l)}"
if [ "$N" -lt 1 ]; then echo 'No GPU detected'; exit 2; fi
mkdir -p runs/v2_7_gpu_fresh logs
rm -f logs/gpu_shard_*.pid
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup python v27_gpu_fresh.py --mode final --output runs/v2_7_gpu_fresh --num-shards "$N" --shard-id "$i" > "logs/gpu_shard_${i}.log" 2>&1 &
  echo $! > "logs/gpu_shard_${i}.pid"
  echo "started shard $i/$N pid $(cat logs/gpu_shard_${i}.pid)"
done
wait
python MERGE_GPU_RESULTS.py --input runs/v2_7_gpu_fresh
python ANALYZE_GPU_RESULTS.py --input runs/v2_7_gpu_fresh
python PACKAGE_RESULTS.py
cat runs/v2_7_gpu_fresh/GPU_MERGED_MANIFEST.json
cat runs/v2_7_gpu_fresh/analysis/GPU_FRESH_GATE_REPORT.md
