#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "Detected GPU: ${GPU_NAME:-unknown}"

check_torch() {
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA-enabled PyTorch is required'
name=torch.cuda.get_device_name(0)
cap=torch.cuda.get_device_capability(0)
arches=list(torch.cuda.get_arch_list())
print('torch',torch.__version__,'CUDA build',torch.version.cuda,'GPU',name,'capability',cap)
print('compiled CUDA arches:',arches)
# RTX 50-series / Blackwell must have a binary with sm_120 support.
if cap >= (12,0) and 'sm_120' not in arches:
    raise RuntimeError('Blackwell GPU detected but this PyTorch wheel lacks sm_120 support')
# torch.cuda.is_available() alone is not enough; execute a real CUDA kernel.
x=torch.randn((512,512),device='cuda')
y=x @ x
torch.cuda.synchronize()
print('CUDA kernel smoke PASS:',float(y[0,0]))
PY
}

if ! check_torch; then
  echo "Existing PyTorch is not usable on this GPU. Installing an official CUDA 12.8 Blackwell-capable wheel..."
  python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  # PyTorch CUDA 12.8+ supports RTX 50-series (sm_120). 2.11 is retained here for a stable, reproducible benchmark env.
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0'
  check_torch
else
  echo "Keeping rental image PyTorch because it passed the real CUDA/sm_120 check."
fi

python -m pip install -r requirements-gpu.txt
python CHECK_GPU_ENV.py
