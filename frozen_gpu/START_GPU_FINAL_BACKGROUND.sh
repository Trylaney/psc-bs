#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p logs
nohup bash RUN_GPU_FINAL_AUTO.sh > logs/gpu_master.log 2>&1 &
echo $! > logs/gpu_master.pid
echo "GPU final started in background. PID $(cat logs/gpu_master.pid)"
echo "Progress: bash CHECK_PROGRESS.sh"
echo "Master log: tail -f logs/gpu_master.log"
