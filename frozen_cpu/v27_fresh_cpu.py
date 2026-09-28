"""Fresh CPU confirmation runner for frozen ContextBench v2.7 PSC-BS."""
from __future__ import annotations
import argparse,csv,hashlib,json,traceback,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from common import seed,rng
from v25_external_data import load_folktables,load_law_school,load_diabetes_hospital
from v2_falsification import prevalence_random
from v26_method import verify_and_select
from v25_strong_baselines import validation_random_search
from v24_benchmark import evaluate_model,write_json,write_csv
from v27_psc_method import deploy_after_certification,V27_VERSION

ACS_TASKS={
 'acs23_income':'acs_income','acs23_employment':'acs_employment','acs23_public_coverage':'acs_public_coverage',
 'acs23_mobility':'acs_mobility','acs23_travel_time':'acs_travel_time',
 'acs23_health_insurance':'acs_health_insurance','acs23_income_poverty':'acs_income_poverty',
}
FRESH_STATES=('CO','MI','MN','NJ','OR')
DEFAULT_DATASETS=list(ACS_TASKS)+['law_school','diabetes_hospital']
HELDOUT_MODELS=('extra_trees','xgboost','lightgbm','mlp')

def load_v27(name,cache):
    if name in ACS_TASKS:return load_folktables(ACS_TASKS[name],cache,states=FRESH_STATES,year=2023)
    if name=='law_school':return load_law_school(cache)
    if name=='diabetes_hospital':return load_diabetes_hospital(cache)
    raise ValueError(name)

def scaled_sizes(total,caps,n_classes):
    caps=np.asarray(caps,int); cap=int(caps.sum())
    if total>=cap:s=caps.copy()
    else:
        raw=caps/cap*total;s=np.floor(raw).astype(int);rem=total-int(s.sum());order=np.argsort(-(raw-s))
        for j in order[:rem]:s[j]+=1
    if np.min(s)<n_classes:raise RuntimeError(f'five-way split too small total={total} sizes={s.tolist()} classes={n_classes}')
    return tuple(int(x) for x in s)

def split5(c,y,s,dataset,outer):
    ids=np.arange(len(y));g=2*np.asarray(y,int)+np.asarray(s,int);classes,counts=np.unique(g,return_counts=True);nc=len(classes)
    if nc<4 or counts.min()<5:raise RuntimeError(f'joint strata insufficient classes={classes.tolist()} counts={counts.tolist()}')
    keys=('pool_cap','calibration_cap','selection_cap','certification_cap','test_cap')
    total=min(len(ids),sum(c[k] for k in keys)); pn,cn,vn,wn,tn=scaled_sizes(total,[c[k] for k in keys],nc)
    if len(ids)>total:ids,_=train_test_split(ids,train_size=total,stratify=g,random_state=seed(c,'v27-cap',dataset,outer))
    p,rest=train_test_split(ids,train_size=pn,stratify=g[ids],random_state=seed(c,'v27-pool',dataset,outer))
    cal,rest=train_test_split(rest,train_size=cn,stratify=g[rest],random_state=seed(c,'v27-cal',dataset,outer))
    v,rest=train_test_split(rest,train_size=vn,stratify=g[rest],random_state=seed(c,'v27-select',dataset,outer))
    w,t=train_test_split(rest,train_size=wn,stratify=g[rest],random_state=seed(c,'v27-cert',dataset,outer))
    if len(t)!=tn:raise RuntimeError(f'test size mismatch {len(t)} != {tn}')
    return tuple(np.sort(x) for x in (p,cal,v,w,t))

def transform5(X,meta,p,cal,v,w,t):
    cat=meta['categorical'];num=[j for j in range(X.shape[1]) if j not in cat];blocks=[]
    if num:blocks.append(('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),num))
    if cat:blocks.append(('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat))
    pre=ColumnTransformer(blocks,sparse_threshold=0);Xp=pre.fit_transform(X[p]);out=[Xp]+[pre.transform(X[ix]) for ix in (cal,v,w,t)]
    return [np.ascontiguousarray(z,dtype=np.float32) for z in out]

def random_anchor_proposals(yp,k,c,dataset,outer):
    rows=[]
    for d in range(5):
        ix=prevalence_random(yp,k,rng(c,dataset,outer,k,'pr',d));rows.append((f'prevalence_random_{d}',np.sort(ix),{}))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='runs/v2_7_fresh_cpu');ap.add_argument('--cache',default='data_cache')
    ap.add_argument('--datasets',nargs='+',default=DEFAULT_DATASETS);ap.add_argument('--outer-splits',type=int,default=5);ap.add_argument('--sizes',nargs='+',type=int,default=[32,64,128,256]);ap.add_argument('--models',nargs='+',default=list(HELDOUT_MODELS));ap.add_argument('--seed',type=int,default=2026092701)
    a=ap.parse_args();out=Path(a.output);(out/'tasks').mkdir(parents=True,exist_ok=True);(out/'contexts').mkdir(exist_ok=True)
    c={'seed':a.seed,'pool_cap':3000,'calibration_cap':500,'selection_cap':500,'certification_cap':1000,'test_cap':2000};status=[]
    for dataset in a.datasets:
        try:X,y,s,orig,meta=load_v27(dataset,a.cache)
        except Exception as e:status.append({'dataset':dataset,'status':'unavailable','error':repr(e)});print(dataset,'UNAVAILABLE',repr(e),flush=True);continue
        status.append({'dataset':dataset,'status':'available','rows':len(y),'source':meta.get('source','')});print(dataset,'LOADED',len(y),flush=True)
        for outer in range(a.outer_splits):
            try:p,cal,v,w,t=split5(c,y,s,dataset,outer);Xp,Xc,Xv,Xw,Xt=transform5(X,meta,p,cal,v,w,t)
            except Exception as e:status.append({'dataset':dataset,'outer':outer,'status':'split_error','error':repr(e)});print(dataset,outer,'SPLIT_ERROR',repr(e),flush=True);continue
            yp,yv,yw,yt=y[p],y[v],y[w],y[t];sv,sw,st=s[v],s[w],s[t]
            print(dataset,'outer',outer,'split',len(p),len(cal),len(v),len(w),len(t),flush=True)
            for k in a.sizes:
                if k>len(yp):continue
                cp=out/'contexts'/f'{dataset}_o{outer}_n{k}_v27.json'
                if cp.exists():
                    z=json.loads(cp.read_text());cand=np.asarray(z['candidate_indices'],int);anchor=np.asarray(z['anchor_indices'],int);final=np.asarray(z['final_indices'],int);cert=z['certificate'];anchors=[(q['selector'],np.asarray(q['indices'],int),{}) for q in z['random_anchors']]
                else:
                    cand,cmeta=validation_random_search(Xp,yp,Xv,yv,sv,k,c,dataset,outer,draws=16,models=('logistic','histgb'))
                    anchors=random_anchor_proposals(yp,k,c,dataset,outer);anchor,ameta=verify_and_select(anchors,Xp,yp,Xv,yv,sv,c,dataset,outer,k)
                    final,cert=deploy_after_certification(cand,anchor,Xp,yp,Xw,yw,sw,c,dataset,outer,k)
                    write_json(cp,{'version':V27_VERSION,'dataset':dataset,'outer':outer,'n':k,'candidate_indices':cand.tolist(),'candidate_meta':cmeta,'anchor_indices':anchor.tolist(),'anchor_meta':ameta,'final_indices':final.tolist(),'certificate':cert,'random_anchors':[{'selector':n,'indices':ix.tolist()} for n,ix,_ in anchors]})
                evals=[('v27_psc_bs',final,cert),('valfair_random_search16',cand,{}),('safety_anchor',anchor,{})]+[(n,ix,{}) for n,ix,_ in anchors]
                for selector,ix,sm in evals:
                    for model in a.models:
                        tid=hashlib.sha1(f'v27|{dataset}|{outer}|{k}|{selector}|{model}'.encode()).hexdigest()[:20];tp=out/'tasks'/f'{tid}.json'
                        if tp.exists():continue
                        rec={'version':V27_VERSION,'dataset':dataset,'outer':outer,'n':k,'selector':selector,'model':model,'selection_meta':sm}
                        try:rec.update(evaluate_model(model,Xp[ix],yp[ix],Xw,yw,sw,Xt,yt,st,seed(c,'v27-eval',dataset,outer,k,selector,model)));rec['status']='ok'
                        except Exception as e:rec['status']='error';rec['error']=repr(e);traceback.print_exc()
                        write_json(tp,rec)
                print(dataset,'outer',outer,'n',k,'certified',bool(cert.get('certified')),flush=True)
    write_csv(out/'dataset_status.csv',status);tasks=[json.loads(p.read_text()) for p in (out/'tasks').glob('*.json')]
    write_json(out/'TASK_MANIFEST.json',{'version':V27_VERSION,'task_files':len(tasks),'ok':sum(q.get('status')=='ok' for q in tasks),'error':sum(q.get('status')=='error' for q in tasks),'datasets':a.datasets,'outer_splits':a.outer_splits,'sizes':a.sizes,'models':a.models,'fresh_data_locked':True})
if __name__=='__main__':main()
