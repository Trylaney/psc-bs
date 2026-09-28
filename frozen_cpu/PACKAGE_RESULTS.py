from pathlib import Path
import zipfile,hashlib,json
root=Path(__file__).resolve().parent
run=root/'runs'/'v2_7_fresh_cpu'
out=root/'V27_FRESH_CPU_RESULTS.zip'
if not (run/'TASK_MANIFEST.json').exists(): raise SystemExit('No TASK_MANIFEST.json; run did not finish.')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(run.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(root))
    for rel in ['frozen/V27_FREEZE_REPORT.md','frozen/FREEZE_MANIFEST.json','frozen/FROZEN_V27_METHOD.json','incoming/SOURCE_MANIFEST.json','v27_psc_method.py','v27_fresh_cpu.py','ANALYZE_FRESH_RESULTS.py']:
        p=root/rel
        if p.exists(): z.write(p,rel)
h=hashlib.sha256(out.read_bytes()).hexdigest()
print('RESULT PACKAGE:',out)
print('SIZE:',out.stat().st_size,'bytes')
print('SHA256:',h)
