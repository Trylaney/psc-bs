from pathlib import Path
import json, hashlib
import numpy as np
from v27_gpu_fresh import DATASETS,SIZES,CFG,load_v27,split5,read_context,SELECTORS,EXPECTED_METHOD_SHA256
root=Path(__file__).resolve().parent
method_hash=hashlib.sha256((root/'v27_psc_method.py').read_bytes()).hexdigest()
assert method_hash==EXPECTED_METHOD_SHA256,(method_hash,EXPECTED_METHOD_SHA256)
ctx=list((root/'frozen_contexts').glob('*.json'))
assert len(ctx)==180,f'expected 180 contexts, got {len(ctx)}'
rows=[]
for d in DATASETS:
    X,y,s,orig,meta=load_v27(d,str(root/'data_cache'))
    for outer in range(5):
        p,cal,v,w,t=split5(CFG,y,s,d,outer)
        for n in SIZES:
            for sel in SELECTORS:
                ix,_=read_context(root,d,outer,n,sel)
                assert len(ix)==n and len(np.unique(ix))==n and ix.min()>=0 and ix.max()<len(p)
        rows.append({'dataset':d,'outer':outer,'rows':len(y),'pool':len(p),'test':len(t)})
print('PRECHECK PASS: 9/9 datasets, 180/180 context cells, frozen method hash OK')
print('Expected final GPU tasks: 2880')
