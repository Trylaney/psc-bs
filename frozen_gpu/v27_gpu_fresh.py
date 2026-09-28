"""Fresh GPU confirmation for frozen ContextBench v2.7 PSC-BS.

Replays EXACT contexts from the passed fresh CPU confirmation. No context
selection is performed here. Test labels are used only for final GPU metrics.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, balanced_accuracy_score, brier_score_loss

from common import seed
from v25_external_data import load_folktables, load_law_school, load_diabetes_hospital
from gpu_models import FrozenGPUModel, installed_versions, TABPFN_MODEL_VERSION, TABICL_CHECKPOINT, N_ESTIMATORS, RANDOM_STATE

ACS_TASKS={
 'acs23_income':'acs_income','acs23_employment':'acs_employment','acs23_public_coverage':'acs_public_coverage',
 'acs23_mobility':'acs_mobility','acs23_travel_time':'acs_travel_time',
 'acs23_health_insurance':'acs_health_insurance','acs23_income_poverty':'acs_income_poverty',
}
FRESH_STATES=('CO','MI','MN','NJ','OR')
DATASETS=list(ACS_TASKS)+['law_school','diabetes_hospital']
SIZES=[32,64,128,256]
MODELS=['tabpfn35','tabiclv2']
SELECTORS=['v27_psc_bs','valfair_random_search16','safety_anchor',
           'prevalence_random_0','prevalence_random_1','prevalence_random_2','prevalence_random_3','prevalence_random_4']
SMOKE_SELECTORS=['v27_psc_bs','valfair_random_search16','safety_anchor']
CFG={'seed':2026092701,'pool_cap':3000,'calibration_cap':500,'selection_cap':500,'certification_cap':1000,'test_cap':2000}
EXPECTED_METHOD_SHA256='f02ccffec4bd20225e261bd4a2b88c803ebfbf700e8f53d6b664e42f82dc307d'


def finite(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def write_json(path,obj):
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.ndarray): return clean(v.tolist())
        if isinstance(v,(np.floating,float)): return finite(v)
        if isinstance(v,(np.integer,)): return int(v)
        return v
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True),encoding='utf-8')
    tmp.replace(path)


def smooth_prob_gaps(p,y,s):
    p=np.asarray(p,float); y=np.asarray(y,int); s=np.asarray(s,int)
    if min(np.sum(s==0),np.sum(s==1))==0: return None,None
    dp=float(p[s==0].mean()-p[s==1].mean()); ds=[]
    for yy in (0,1):
        m0=(y==yy)&(s==0); m1=(y==yy)&(s==1)
        if m0.sum()==0 or m1.sum()==0: return dp,None
        ds.append(float(p[m0].mean()-p[m1].mean()))
    return dp,ds


def threshold_gaps(p,y,s,t=.5,min_cell=5):
    p=np.asarray(p,float); y=np.asarray(y,int); s=np.asarray(s,int); h=p>=t
    out={'dp_gap':None,'tpr_gap':None,'fpr_gap':None,'eo_gap':None,'worst_group_accuracy':None}
    gm=[s==g for g in (0,1)]
    if all(m.sum()>=min_cell for m in gm):
        out['dp_gap']=float(abs(h[gm[1]].mean()-h[gm[0]].mean()))
        out['worst_group_accuracy']=float(min(np.mean(h[m]==y[m]) for m in gm))
    vals=[]
    for yy,key in [(1,'tpr_gap'),(0,'fpr_gap')]:
        mm=[(s==g)&(y==yy) for g in (0,1)]
        if all(m.sum()>=min_cell for m in mm):
            out[key]=float(abs(h[mm[1]].mean()-h[mm[0]].mean())); vals.append(out[key])
    if len(vals)==2: out['eo_gap']=float(max(vals))
    return out


def metrics(p,y,s):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); y=np.asarray(y,int); h=p>=.5
    out={'auc':None,'logloss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,h)),
         'balanced_accuracy':float(balanced_accuracy_score(y,h)) if len(np.unique(y))==2 else None,
         'brier':float(brier_score_loss(y,p))}
    if len(np.unique(y))==2: out['auc']=float(roc_auc_score(y,p))
    dp,eo=smooth_prob_gaps(p,y,s)
    out['smooth_dp_sq']=None if dp is None else float(dp*dp)
    out['smooth_eo_sq']=None if eo is None else float(sum(v*v for v in eo))
    out.update(threshold_gaps(p,y,s))
    return out


def load_v27(name,cache):
    if name in ACS_TASKS: return load_folktables(ACS_TASKS[name],cache,states=FRESH_STATES,year=2023)
    if name=='law_school': return load_law_school(cache)
    if name=='diabetes_hospital': return load_diabetes_hospital(cache)
    raise ValueError(name)


def scaled_sizes(total,caps,n_classes):
    caps=np.asarray(caps,int); cap=int(caps.sum())
    if total>=cap: s=caps.copy()
    else:
        raw=caps/cap*total; s=np.floor(raw).astype(int); rem=total-int(s.sum()); order=np.argsort(-(raw-s))
        for j in order[:rem]: s[j]+=1
    if np.min(s)<n_classes: raise RuntimeError(f'five-way split too small total={total} sizes={s.tolist()} classes={n_classes}')
    return tuple(int(x) for x in s)


def split5(c,y,s,dataset,outer):
    ids=np.arange(len(y)); g=2*np.asarray(y,int)+np.asarray(s,int); classes,counts=np.unique(g,return_counts=True); nc=len(classes)
    if nc<4 or counts.min()<5: raise RuntimeError(f'joint strata insufficient classes={classes.tolist()} counts={counts.tolist()}')
    keys=('pool_cap','calibration_cap','selection_cap','certification_cap','test_cap')
    total=min(len(ids),sum(c[k] for k in keys)); pn,cn,vn,wn,tn=scaled_sizes(total,[c[k] for k in keys],nc)
    if len(ids)>total: ids,_=train_test_split(ids,train_size=total,stratify=g,random_state=seed(c,'v27-cap',dataset,outer))
    p,rest=train_test_split(ids,train_size=pn,stratify=g[ids],random_state=seed(c,'v27-pool',dataset,outer))
    cal,rest=train_test_split(rest,train_size=cn,stratify=g[rest],random_state=seed(c,'v27-cal',dataset,outer))
    v,rest=train_test_split(rest,train_size=vn,stratify=g[rest],random_state=seed(c,'v27-select',dataset,outer))
    w,t=train_test_split(rest,train_size=wn,stratify=g[rest],random_state=seed(c,'v27-cert',dataset,outer))
    if len(t)!=tn: raise RuntimeError(f'test size mismatch {len(t)} != {tn}')
    return tuple(np.sort(x) for x in (p,cal,v,w,t))


def transform5(X,meta,p,cal,v,w,t):
    cat=meta['categorical']; num=[j for j in range(X.shape[1]) if j not in cat]; blocks=[]
    if num: blocks.append(('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),num))
    if cat: blocks.append(('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat))
    pre=ColumnTransformer(blocks,sparse_threshold=0); Xp=pre.fit_transform(X[p]); out=[Xp]+[pre.transform(X[ix]) for ix in (cal,v,w,t)]
    return [np.ascontiguousarray(z,dtype=np.float32) for z in out]


def task_id(dataset,outer,n,selector,model):
    return hashlib.sha1(f'v27gpu|{dataset}|{outer}|{n}|{selector}|{model}'.encode()).hexdigest()[:24]


def read_context(root,dataset,outer,n,selector):
    p=root/'frozen_contexts'/f'{dataset}_o{outer}_n{n}_v27.json'
    z=json.loads(p.read_text(encoding='utf-8'))
    if z['dataset']!=dataset or int(z['outer'])!=outer or int(z['n'])!=n:
        raise RuntimeError(f'frozen context identity mismatch: {p}')
    if selector=='v27_psc_bs': ix=np.asarray(z['final_indices'],dtype=int); meta=z.get('certificate',{})
    elif selector=='valfair_random_search16': ix=np.asarray(z['candidate_indices'],dtype=int); meta=z.get('candidate_meta',{})
    elif selector=='safety_anchor': ix=np.asarray(z['anchor_indices'],dtype=int); meta=z.get('anchor_meta',{})
    elif selector.startswith('prevalence_random_'):
        hits=[q for q in z['random_anchors'] if q['selector']==selector]
        if len(hits)!=1: raise RuntimeError(f'{selector}: expected one random anchor in {p}, got {len(hits)}')
        ix=np.asarray(hits[0]['indices'],dtype=int); meta={}
    else: raise ValueError(selector)
    if len(ix)!=n or len(np.unique(ix))!=n:
        raise RuntimeError(f'invalid context {dataset} o{outer} n{n} {selector}: len={len(ix)} unique={len(np.unique(ix))}')
    return ix,meta


def make_tasks(mode,models,datasets,sizes,outer_splits,selectors):
    tasks=[]
    for model in models:
      for dataset in datasets:
       for outer in range(outer_splits):
        for n in sizes:
         for selector in selectors:
          tasks.append((dataset,outer,n,selector,model))
    if mode=='smoke':
        tasks=[q for q in tasks if q[0]=='acs23_income' and q[1]==0 and q[2] in (32,256) and q[3] in SMOKE_SELECTORS]
    return tasks


def env_snapshot():
    import torch
    return {'python':sys.version,'platform':platform.platform(),'packages':installed_versions(),
      'torch_cuda_available':bool(torch.cuda.is_available()),'torch_cuda_version':getattr(torch.version,'cuda',None),
      'gpu_count':int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
      'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      'tabpfn_model_version':TABPFN_MODEL_VERSION,'tabicl_checkpoint':TABICL_CHECKPOINT,
      'n_estimators':N_ESTIMATORS,'model_random_state':RANDOM_STATE,
      'fresh_cpu_method_sha256':EXPECTED_METHOD_SHA256}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','final'],default='smoke')
    ap.add_argument('--output',default='runs/v2_7_gpu_fresh'); ap.add_argument('--cache',default='data_cache')
    ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1)
    ap.add_argument('--models',nargs='+',default=MODELS); ap.add_argument('--datasets',nargs='+',default=DATASETS)
    ap.add_argument('--sizes',nargs='+',type=int,default=SIZES); ap.add_argument('--outer-splits',type=int,default=5)
    ap.add_argument('--selectors',nargs='+',default=SELECTORS); ap.add_argument('--smoke-test-limit',type=int,default=256)
    a=ap.parse_args()
    if not (0<=a.shard_id<a.num_shards): raise ValueError('bad shard')
    root=Path(__file__).resolve().parent; out=Path(a.output); (out/'tasks').mkdir(parents=True,exist_ok=True)
    got=hashlib.sha256((root/'v27_psc_method.py').read_bytes()).hexdigest()
    if got!=EXPECTED_METHOD_SHA256: raise RuntimeError(f'frozen method hash mismatch: {got}')
    snap=env_snapshot(); write_json(out/f'ENVIRONMENT_shard{a.shard_id}.json',snap)
    if not snap['torch_cuda_available']: raise RuntimeError('CUDA unavailable; final GPU confirmation must not run on CPU')
    tasks=make_tasks(a.mode,a.models,a.datasets,a.sizes,a.outer_splits,a.selectors)
    tasks=[q for i,q in enumerate(tasks) if i%a.num_shards==a.shard_id]
    current_model=None; wrapper=None; current_dataset=None; data_bundle=None; split_cache={}; completed=0; errors=0
    try:
      for dataset,outer,n,selector,model in tasks:
        tid=task_id(dataset,outer,n,selector,model); tp=out/'tasks'/f'{tid}.json'
        if tp.exists(): continue
        if model!=current_model:
            if wrapper is not None: wrapper.close()
            wrapper=FrozenGPUModel(model,device='cuda'); current_model=model
        if dataset!=current_dataset:
            data_bundle=load_v27(dataset,str(root/a.cache)); current_dataset=dataset; split_cache={}
        X,y,s,orig,meta=data_bundle
        sk=(dataset,outer)
        if sk not in split_cache:
            p,cal,v,w,t=split5(CFG,y,s,dataset,outer); Xp,Xc,Xv,Xw,Xt=transform5(X,meta,p,cal,v,w,t)
            split_cache[sk]=(p,t,Xp,Xt)
        p,t,Xp,Xt=split_cache[sk]
        ix,selection_meta=read_context(root,dataset,outer,n,selector)
        if int(ix.max())>=len(p): raise RuntimeError(f'context index out of pool range: {dataset} o{outer} n{n} {selector}')
        yp=np.asarray(y[p],dtype=np.int64); yt=np.asarray(y[t],dtype=np.int64); st=np.asarray(s[t],dtype=np.int64)
        Xe=Xt; ye=yt; se=st
        if a.mode=='smoke' and a.smoke_test_limit and len(Xe)>a.smoke_test_limit:
            Xe=Xe[:a.smoke_test_limit]; ye=ye[:a.smoke_test_limit]; se=se[:a.smoke_test_limit]
        rec={'version':'v2.7-gpu-fresh-1','mode':a.mode,'dataset':dataset,'outer':outer,'n':n,'selector':selector,'model':model,
             'selection_meta':selection_meta,'context_indices_sha1':hashlib.sha1(ix.tobytes()).hexdigest(),
             'test_rows':int(len(ye)),'shard_id':a.shard_id,'num_shards':a.num_shards,'status':'error'}
        try:
            prob,timing=wrapper.fit_predict_proba(Xp[ix],yp[ix],Xe)
            rec.update(timing); rec.update({f'test_{k}':v for k,v in metrics(prob,ye,se).items()}); rec['status']='ok'; completed+=1
            print(f'OK shard={a.shard_id}/{a.num_shards} {model} {dataset} o{outer} n{n} {selector} {timing["total_seconds"]:.2f}s mem={None if timing["peak_memory_reserved_bytes"] is None else round(timing["peak_memory_reserved_bytes"]/2**30,2)}GiB',flush=True)
        except Exception as e:
            rec['error']=repr(e); rec['traceback']=traceback.format_exc(); errors+=1
            print(f'ERROR {model} {dataset} o{outer} n{n} {selector}: {e!r}',flush=True)
        write_json(tp,rec)
    finally:
      if wrapper is not None: wrapper.close()
    rows=[]
    for q in (out/'tasks').glob('*.json'):
      try: rows.append(json.loads(q.read_text(encoding='utf-8')))
      except Exception: pass
    write_json(out/f'TASK_MANIFEST_shard{a.shard_id}.json',{
      'mode':a.mode,'shard_id':a.shard_id,'num_shards':a.num_shards,'assigned_tasks':len(tasks),
      'all_task_files_visible':len(rows),'ok_visible':sum(r.get('status')=='ok' for r in rows),
      'error_visible':sum(r.get('status')=='error' for r in rows),'selectors':SMOKE_SELECTORS if a.mode=='smoke' else a.selectors,'models':a.models})
    print(f'DONE shard={a.shard_id}/{a.num_shards} new_ok={completed} new_error={errors}',flush=True)

if __name__=='__main__': main()
