from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
expected='f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d'
p=root/'v27_psc_method.py'
h=hashlib.sha256(p.read_bytes()).hexdigest()
print('v27_psc_method.py sha256 =',h)
if h!=expected:
    raise SystemExit('FREEZE CHECK FAILED: frozen method file was modified')
print('FREEZE CHECK OK: v2.7 PSC-BS method matches frozen manifest')
