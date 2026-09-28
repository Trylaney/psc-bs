"""Deterministic identities, atomic writes, and integrity checks."""
import hashlib, json, os, platform, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def now(): return datetime.now(timezone.utc).isoformat()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x): return hashlib.sha256(x).hexdigest()
def file_hash(p): return digest(Path(p).read_bytes())
def seed(c,*parts): return int.from_bytes(hashlib.sha256(canon([c['seed'],*parts])).digest()[:4],'little')
def rng(c,*parts): return np.random.default_rng(seed(c,*parts))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def verify_task(r):
 if r.get('record_sha256')!=digest(canon({k:v for k,v in r.items() if k!='record_sha256'})):raise RuntimeError('Task metadata integrity failure')
 return r
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def save_npz(p,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_name(p.name+'.tmp').open('wb') as f: np.savez_compressed(f,**arrays)
 os.replace(p.with_name(p.name+'.tmp'),p)
def source_hashes():
 return {p.name:file_hash(p) for p in sorted(ROOT.iterdir()) if (p.suffix in ('.py','.txt','.md','.cmd') or p.name in ('config.json','confirmation_plan.template.json')) and p.name not in ('VALIDATION.md',)}
def environment(with_tabicl=False):
 from importlib.metadata import version,PackageNotFoundError
 names=['numpy','scipy','scikit-learn','matplotlib','threadpoolctl']+(['torch','tabicl'] if with_tabicl else [])
 out={'python':sys.version,'platform':platform.platform(),'packages':{}}
 for n in names:
  try:out['packages'][n]=version(n)
  except PackageNotFoundError:out['packages'][n]='MISSING'
 return out
def locked_record(path,identity,make):
 p=Path(path)
 if p.exists():
  d=read(p)
  if d['identity']!=identity or d['payload_hash']!=digest(canon(d['payload'])):raise RuntimeError('Frozen record changed: '+str(p))
  return d['payload']
 v=make();write(p,{'identity':identity,'payload_hash':digest(canon(v)),'payload':v});return v
