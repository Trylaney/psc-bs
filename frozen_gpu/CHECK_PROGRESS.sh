#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
N=$(find runs/v2_7_gpu_fresh/tasks -name '*.json' 2>/dev/null | wc -l || true)
E=$(grep -R '"status": "error"' runs/v2_7_gpu_fresh/tasks 2>/dev/null | wc -l || true)
echo "$N / 2880 task files; errors seen: $E"
for f in logs/gpu_shard_*.log; do [ -f "$f" ] && echo "--- $f" && tail -n 2 "$f"; done
