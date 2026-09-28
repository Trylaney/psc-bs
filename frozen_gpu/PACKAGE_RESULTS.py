from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
out=root/'V27_FRESH_GPU_RESULTS.zip'
include=[root/'runs'/'v2_7_gpu_fresh',root/'GPU_CONFIRM_PROTOCOL.json',root/'FROZEN_CONTEXT_MANIFEST.json',root/'FRESH_CPU_SUMMARY.json',root/'FRESH_CPU_GATE_REPORT.md',root/'v27_psc_method.py',root/'v27_gpu_fresh.py']
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in include:
        if p.is_file(): z.write(p,p.relative_to(root))
        elif p.is_dir():
            for q in p.rglob('*'):
                if q.is_file(): z.write(q,q.relative_to(root))
print(out, out.stat().st_size)
