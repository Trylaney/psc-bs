from pathlib import Path
import argparse,json,csv
ap=argparse.ArgumentParser(); ap.add_argument('--input',default='runs/v2_7_gpu_fresh'); a=ap.parse_args(); root=Path(a.input)
rows=[]
for p in sorted((root/'tasks').glob('*.json')):
    try: rows.append(json.loads(p.read_text(encoding='utf-8')))
    except Exception: pass
cols=sorted(set().union(*(r.keys() for r in rows))) if rows else []
with open(root/'GPU_RESULTS.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
manifest={'task_files':len(rows),'ok':sum(r.get('status')=='ok' for r in rows),'error':sum(r.get('status')=='error' for r in rows)}
(root/'GPU_MERGED_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(manifest); print(root/'GPU_RESULTS.csv')
