from __future__ import annotations
import concurrent.futures as cf
import hashlib, json, os, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCOMING = ROOT / "incoming"
INCOMING.mkdir(exist_ok=True)

SOURCES = [
    ("csv_pco.zip", "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pco.zip", "zip", "psam_p08.csv"),
    ("csv_pmi.zip", "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pmi.zip", "zip", "psam_p26.csv"),
    ("csv_pmn.zip", "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pmn.zip", "zip", "psam_p27.csv"),
    ("csv_pnj.zip", "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pnj.zip", "zip", "psam_p34.csv"),
    ("csv_por.zip", "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_por.zip", "zip", "psam_p41.csv"),
    ("law_dataset.csv", "https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv", "csv", "pass_bar"),
    ("diabetes_130.zip", "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip", "zip", "diabetic_data.csv"),
]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def validate(path: Path, kind: str, needle: str) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 1000: return False
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                names=[Path(n).name.lower() for n in z.namelist()]
                return needle.lower() in names
        head=path.open('r',encoding='utf-8',errors='ignore').readline().lower()
        return needle.lower() in head
    except Exception:
        return False

def download_one(spec):
    name,url,kind,needle=spec
    dst=INCOMING/name
    if validate(dst,kind,needle):
        return {"file":name,"status":"cached","bytes":dst.stat().st_size,"sha256":sha256(dst),"url":url}
    part=dst.with_suffix(dst.suffix+'.part')
    part.unlink(missing_ok=True)
    err=None
    for attempt in range(1,4):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"ContextBench-v2.7/1.0"})
            with urllib.request.urlopen(req,timeout=120) as r, part.open('wb') as out:
                total=int(r.headers.get('Content-Length','0') or 0); done=0; last=time.time()
                while True:
                    b=r.read(1<<20)
                    if not b: break
                    out.write(b); done += len(b)
                    if time.time()-last>2:
                        pct=f"{done*100/total:5.1f}%" if total else f"{done/1e6:.1f}MB"
                        print(f"[{name}] {pct}",flush=True); last=time.time()
            part.replace(dst)
            if not validate(dst,kind,needle): raise RuntimeError("downloaded file failed validation")
            return {"file":name,"status":"downloaded","bytes":dst.stat().st_size,"sha256":sha256(dst),"url":url}
        except Exception as e:
            err=repr(e); print(f"[{name}] attempt {attempt}/3 failed: {e}",flush=True)
            part.unlink(missing_ok=True); time.sleep(2*attempt)
    raise RuntimeError(f"{name} download failed after 3 attempts: {err}")

def main():
    print("Downloading 7 locked fresh-data artifacts (4 parallel workers)...",flush=True)
    rows=[]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs={ex.submit(download_one,s):s[0] for s in SOURCES}
        for f in cf.as_completed(futs):
            row=f.result(); rows.append(row); print("OK",row['file'],row['status'],row['bytes'],flush=True)
    rows=sorted(rows,key=lambda x:x['file'])
    (INCOMING/'SOURCE_MANIFEST.json').write_text(json.dumps({"locked_protocol":"v2.7-psc-bs-fresh-20260927","files":rows},indent=2),encoding='utf-8')
    print("ALL DOWNLOADS OK")

if __name__=='__main__': main()
