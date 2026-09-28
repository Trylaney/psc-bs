import sys
print('Python:',sys.version)
try:
 import torch
 print('torch:',torch.__version__,'CUDA build:',torch.version.cuda,'available:',torch.cuda.is_available())
 if not torch.cuda.is_available():
  raise RuntimeError('CUDA unavailable')
 print('GPU count:',torch.cuda.device_count())
 for i in range(torch.cuda.device_count()):
  p=torch.cuda.get_device_properties(i)
  print(i,torch.cuda.get_device_name(i),round(p.total_memory/2**30,2),'GiB','capability',torch.cuda.get_device_capability(i))
 print('compiled arches:',torch.cuda.get_arch_list())
 cap=torch.cuda.get_device_capability(0)
 if cap >= (12,0) and 'sm_120' not in torch.cuda.get_arch_list():
  raise RuntimeError('Blackwell/RTX 50-series detected but PyTorch does not contain sm_120 kernels')
 a=torch.randn((256,256),device='cuda'); b=a@a; torch.cuda.synchronize()
 print('real CUDA kernel: PASS')
except Exception as e:
 print('torch/CUDA ERROR:',repr(e))
 raise

import importlib.metadata as md
for pkg in ('tabpfn','tabicl'):
 try: print(pkg,md.version(pkg))
 except Exception as e: print(pkg,'NOT INSTALLED',repr(e))
