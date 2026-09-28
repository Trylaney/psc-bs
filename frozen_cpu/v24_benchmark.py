"""ContextBench v2.4 resumable CPU benchmark harness.

Development runner only: it intentionally operates on datasets designated for
method development. Planned external datasets remain unavailable until their
adapters/data are explicitly supplied and validated.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,traceback,warnings
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score,balanced_accuracy_score,brier_score_loss

from common import seed,rng,canon,digest
from data import load,split
from design import transform,balanced
from v2_method import build_proxy,label_targets
from v2_falsification import farthest,prevalence_random
from v24_method import (build_bootstrap_moment_proxies,risk_greedy,risk_predict,constrained_risk_select,constrained_gradient_risk_select)
from v24_models import fit_predict,make_model,MODEL_NAMES


def finite(x):
    try:
        x=float(x)
        return x if np.isfinite(x) else None
    except Exception:return None

def write_json(path,obj):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,(np.floating,float)):return finite(v)
        if isinstance(v,(np.integer,)):return int(v)
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),indent=2,sort_keys=True),encoding='utf-8');tmp.replace(path)

def write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def smooth_prob_gaps(p,y,s):
    p=np.asarray(p,float);y=np.asarray(y,int);s=np.asarray(s,int)
    if min(np.sum(s==0),np.sum(s==1))==0:return None,None
    dp=float(p[s==0].mean()-p[s==1].mean())
    ds=[]
    for yy in (0,1):
        m0=(y==yy)&(s==0);m1=(y==yy)&(s==1)
        if m0.sum()==0 or m1.sum()==0:return dp,None
        ds.append(float(p[m0].mean()-p[m1].mean()))
    return dp,ds

def threshold_gaps(p,y,s,t=.5,min_cell=5):
    p=np.asarray(p,float);y=np.asarray(y,int);s=np.asarray(s,int);h=p>=t
    out={'dp_gap':None,'tpr_gap':None,'fpr_gap':None,'eo_gap':None,'worst_group_accuracy':None}
    gm=[s==g for g in (0,1)]
    if all(m.sum()>=min_cell for m in gm):
        out['dp_gap']=float(abs(h[gm[1]].mean()-h[gm[0]].mean()))
        out['worst_group_accuracy']=float(min(np.mean(h[m]==y[m]) for m in gm))
    vals=[]
    for yy,key in [(1,'tpr_gap'),(0,'fpr_gap')]:
        mm=[(s==g)&(y==yy) for g in (0,1)]
        if all(m.sum()>=min_cell for m in mm):
            out[key]=float(abs(h[mm[1]].mean()-h[mm[0]].mean()));vals.append(out[key])
    if len(vals)==2:out['eo_gap']=float(max(vals))
    return out

def metrics(p,y,s):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);y=np.asarray(y,int);h=p>=.5
    out={'auc':None,'logloss':float(log_loss(y,p,labels=[0,1])),'accuracy':float(accuracy_score(y,h)),
         'balanced_accuracy':float(balanced_accuracy_score(y,h)) if len(np.unique(y))==2 else None,
         'brier':float(brier_score_loss(y,p))}
    if len(np.unique(y))==2:out['auc']=float(roc_auc_score(y,p))
    dp,eo=smooth_prob_gaps(p,y,s)
    out['smooth_dp_sq']=None if dp is None else float(dp*dp)
    out['smooth_eo_sq']=None if eo is None else float(sum(v*v for v in eo))
    out.update(threshold_gaps(p,y,s))
    return out

def evaluate_model(name,Xs,ys,Xv,yv,sv,Xt,yt,st,sd):
    tic=time.perf_counter();pv,pt,ww=fit_predict(name,Xs,ys,Xv,Xt,sd);elapsed=time.perf_counter()-tic
    out={'fit_predict_seconds':elapsed,'warnings_count':len(ww)}
    out.update({f'val_{k}':v for k,v in metrics(pv,yv,sv).items()})
    out.update({f'test_{k}':v for k,v in metrics(pt,yt,st).items()})
    return out

def group_balance(s,k,r):return balanced(s,k,r)

def baseline_context(name,X,y,s,k,r,proxy_prob):
    if name=='random':return np.sort(r.choice(len(y),k,replace=False))
    if name=='prevalence_random':return prevalence_random(y,k,r)
    if name=='label_balance':return balanced(y,k,r)
    if name=='group_balance':return group_balance(s,k,r)
    if name=='joint_balance':return balanced(2*y+s,k,r)
    if name=='uncertainty':return np.argsort(np.abs(proxy_prob-.5),kind='stable')[:k]
    if name=='diversity':return np.sort(farthest(X,k,r))
    raise ValueError(name)

def context_key(dataset,outer,k,selector):
    return hashlib.sha1(f'{dataset}|{outer}|{k}|{selector}'.encode()).hexdigest()[:16]

def task_key(dataset,outer,k,selector,model):
    return hashlib.sha1(f'{dataset}|{outer}|{k}|{selector}|{model}'.encode()).hexdigest()[:20]


def stratified_bootstrap_envs(y,s,count,r):
    y=np.asarray(y,int);s=np.asarray(s,int);out=[np.arange(len(y),dtype=int)]
    for _ in range(max(0,count-1)):
        parts=[]
        for yy in (0,1):
          for gg in (0,1):
            ids=np.flatnonzero((y==yy)&(s==gg))
            if len(ids)==0:raise ValueError('utility bootstrap missing label/group cell')
            parts.append(r.choice(ids,len(ids),replace=True))
        out.append(np.concatenate(parts))
    return out


def cvar(values,frac=.5):
    v=np.asarray(values,float);q=max(1,int(np.ceil(len(v)*frac)))
    return float(np.sort(v)[-q:].mean())


def proxy_validation_risk(model_name,Xp,yp,ix,Xv,yv,env_ids,sd):
    m=make_model(model_name,sd)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');m.fit(Xp[ix],yp[ix])
    p=np.asarray(m.predict_proba(Xv)[:,list(m.classes_).index(1)],float)
    losses=[float(log_loss(yv[e],p[e],labels=[0,1])) for e in env_ids]
    return cvar(losses,.5)


def proxy_utility_guard_select(gps,Xp,yp,Xv,yv,sv,k,c,dataset,outer,quantile=.75,
                               utility_models=('logistic','histgb')):
    """Select minimum fairness-risk path point satisfying robust proxy utility budgets.

    Utility budgets are fixed quantiles of prevalence-matched random-context
    validation CVaR log-loss, separately for every cheap utility proxy.
    """
    env_ids=stratified_bootstrap_envs(yv,sv,6,rng(c,dataset,outer,k,'utility-envs'))
    refs=[]
    for draw in range(10):
        ix=prevalence_random(yp,k,rng(c,dataset,outer,k,'utility-ref',draw))
        vals={m:proxy_validation_risk(m,Xp,yp,ix,Xv,yv,env_ids,seed(c,dataset,outer,k,'utility-ref',draw,m)) for m in utility_models}
        refs.append(vals)
    tau={m:float(np.quantile([r[m] for r in refs],quantile)) for m in utility_models}
    path=[]
    for eta in (0.,.02,.05,.1,.25,.5,1.,2.,5.,10.):
        ix,_=risk_greedy(gps,Xp,yp,k,mode='cvar50',representation_weight=eta,scale_rng=rng(c,dataset,outer,k,'proxyguard-path',eta))
        fair=risk_predict(gps,ix,'cvar50')['risk']
        util={m:proxy_validation_risk(m,Xp,yp,ix,Xv,yv,env_ids,seed(c,dataset,outer,k,'proxyguard-path',eta,m)) for m in utility_models}
        # positive normalized violation; 0 is feasible
        violation=max([max(0.0,(util[m]-tau[m])/max(tau[m],1e-8)) for m in utility_models])
        path.append((np.sort(ix),fair,util,violation,float(eta)))
    feasible=[z for z in path if z[3]<=1e-15]
    if feasible:chosen=min(feasible,key=lambda z:(z[1],max(z[2].values()),z[4]));fallback=False
    else:chosen=min(path,key=lambda z:(z[3],z[1],z[4]));fallback=True
    ix,fair,util,viol,eta=chosen
    return ix,{'utility_quantile':float(quantile),'utility_models':list(utility_models),'utility_tau':tau,
               'utility_risk':util,'utility_violation':float(viol),'chosen_eta':eta,'fairness_risk':float(fair),
               'fallback_no_feasible':fallback,'path_size':len(path)}


def proxy_utility_guard_select_multi(gps,Xp,yp,Xv,yv,sv,k,c,dataset,outer,quantiles=(.50,.75),
                                     utility_models=('logistic','histgb')):
    env_ids=stratified_bootstrap_envs(yv,sv,6,rng(c,dataset,outer,k,'utility-envs'))
    refs=[]
    for draw in range(10):
        ix=prevalence_random(yp,k,rng(c,dataset,outer,k,'utility-ref',draw))
        refs.append({m:proxy_validation_risk(m,Xp,yp,ix,Xv,yv,env_ids,seed(c,dataset,outer,k,'utility-ref',draw,m)) for m in utility_models})
    path=[]
    for eta in (0.,.02,.05,.1,.25,.5,1.,2.,5.,10.):
        ix,_=risk_greedy(gps,Xp,yp,k,mode='cvar50',representation_weight=eta,scale_rng=rng(c,dataset,outer,k,'proxyguard-path',eta))
        fair=risk_predict(gps,ix,'cvar50')['risk']
        util={m:proxy_validation_risk(m,Xp,yp,ix,Xv,yv,env_ids,seed(c,dataset,outer,k,'proxyguard-path',eta,m)) for m in utility_models}
        path.append((np.sort(ix),fair,util,float(eta)))
    out={}
    for quantile in quantiles:
        tau={m:float(np.quantile([r[m] for r in refs],quantile)) for m in utility_models}
        scored=[]
        for ix,fair,util,eta in path:
            violation=max(max(0.0,(util[m]-tau[m])/max(tau[m],1e-8)) for m in utility_models)
            scored.append((ix,fair,util,violation,eta))
        feasible=[z for z in scored if z[3]<=1e-15]
        if feasible:chosen=min(feasible,key=lambda z:(z[1],max(z[2].values()),z[4]));fallback=False
        else:chosen=min(scored,key=lambda z:(z[3],z[1],z[4]));fallback=True
        ix,fair,util,viol,eta=chosen
        out[float(quantile)]=(ix,{'utility_quantile':float(quantile),'utility_models':list(utility_models),'utility_tau':tau,
              'utility_risk':util,'utility_violation':float(viol),'chosen_eta':eta,'fairness_risk':float(fair),
              'fallback_no_feasible':fallback,'path_size':len(path)})
    return out

def build_contexts(out,c,dataset,outer,k,Xp,yp,sp,Xv,yv,sv,envs,random_baselines,include_eodp):
    cp=out/'contexts'/f'{context_key(dataset,outer,k,f"cellv4-e{envs}-r{random_baselines}-edp{int(include_eodp)}")}.json'
    if cp.exists():
        z=json.loads(cp.read_text());return [(x['selector'],np.asarray(x['indices'],int),x.get('selection_meta',{})) for x in z['contexts']]
    regs=(5e-4,5e-3)
    gps=build_bootstrap_moment_proxies(Xp,yp,Xv,yv,sv,regs=regs,environments=envs,
                                       random_state=seed(c,dataset,outer,k,'eo-envs'),objective='eo')
    gps_eodp=None
    if include_eodp:
        gps_eodp=build_bootstrap_moment_proxies(Xp,yp,Xv,yv,sv,regs=regs,environments=envs,
                                                random_state=seed(c,dataset,outer,k,'eodp-envs'),objective='eo+dp')
    base=build_proxy(Xp,yp,Xv,yv,sv,regs[0]);p_unc=expit(np.column_stack([Xp,np.ones(len(Xp))])@base.w)
    rows=[]
    for d in range(random_baselines):
        ix=baseline_context('prevalence_random',Xp,yp,sp,k,rng(c,dataset,outer,k,'pr',d),p_unc)
        rows.append((f'prevalence_random_{d}',ix,{}))
    ix=baseline_context('random',Xp,yp,sp,k,rng(c,dataset,outer,k,'random'),p_unc);rows.append(('random',ix,{}))
    for b in ('label_balance','group_balance','joint_balance','uncertainty','diversity'):
        ix=baseline_context(b,Xp,yp,sp,k,rng(c,dataset,outer,k,b),p_unc);rows.append((b,ix,{}))
    # Method/ablation family. Hyperparameters are fixed globally, not selected by downstream test metrics.
    ix,sm=risk_greedy([gps[0]],Xp,yp,k,mode='mean',representation_weight=.5,scale_rng=rng(c,dataset,outer,k,'single'));rows.append(('single_proxy_gn',np.sort(ix),sm))
    ix,sm=risk_greedy(gps,Xp,yp,k,mode='mean',representation_weight=.5,scale_rng=rng(c,dataset,outer,k,'mean'));rows.append(('mean_env_gn',np.sort(ix),sm))
    ix,sm=risk_greedy(gps,Xp,yp,k,mode='cvar50',representation_weight=.5,scale_rng=rng(c,dataset,outer,k,'cvarw'));rows.append(('cvar50_weighted',np.sort(ix),sm))
    ix,sm=constrained_risk_select(gps,Xp,yp,k,mode='cvar50',fidelity_quantile=.10,scale_rng=rng(c,dataset,outer,k,'cvarc'));rows.append(('cvar50_constrained',np.sort(ix),sm))
    pg=proxy_utility_guard_select_multi(gps,Xp,yp,Xv,yv,sv,k,c,dataset,outer,quantiles=(.50,.75))
    ix,sm=pg[.50];rows.append(('cvar50_proxyguard_q50',np.sort(ix),sm))
    ix,sm=pg[.75];rows.append(('cvar50_proxyguard_q75',np.sort(ix),sm))
    ix,sm=constrained_gradient_risk_select(gps,Xp,yp,k,mode='cvar50',fidelity_quantile=.10,scale_rng=rng(c,dataset,outer,k,'cvargrad'));rows.append(('cvar50_gradconstrained',np.sort(ix),sm))
    ix,sm=risk_greedy(gps,Xp,yp,k,mode='worst',representation_weight=.5,scale_rng=rng(c,dataset,outer,k,'worst'));rows.append(('worst_env_gn',np.sort(ix),sm))
    if gps_eodp is not None:
        ix,sm=constrained_risk_select(gps_eodp,Xp,yp,k,mode='cvar50',fidelity_quantile=.10,scale_rng=rng(c,dataset,outer,k,'eodpc'));rows.append(('cvar50_constrained_eodp',np.sort(ix),sm))
    payload={'dataset':dataset,'outer':outer,'n':k,'contexts':[{'selector':n,'indices':np.asarray(ix,int).tolist(),'selection_meta':sm} for n,ix,sm in rows]}
    write_json(cp,payload)
    return rows

def summarize(out,task_rows):
    ok=[r for r in task_rows if r.get('status')=='ok']
    # paired deltas against prevalence-matched random mean in each dataset/outer/n/model cell
    agg=[]
    cells=sorted(set((r['dataset'],r['outer'],r['n'],r['model']) for r in ok))
    for dataset,outer,n,model in cells:
        cell=[r for r in ok if (r['dataset'],r['outer'],r['n'],r['model'])==(dataset,outer,n,model)]
        base=[r for r in cell if r['selector'].startswith('prevalence_random_')]
        if not base:continue
        def avg(k):
            z=[r.get(k) for r in base if r.get(k) is not None];return float(np.mean(z)) if z else None
        bf,bdp,ba,bll=avg('test_smooth_eo_sq'),avg('test_smooth_dp_sq'),avg('test_auc'),avg('test_logloss')
        for r in cell:
            if r['selector'].startswith('prevalence_random_'):continue
            df=None if bf is None or r.get('test_smooth_eo_sq') is None else r['test_smooth_eo_sq']-bf
            ddp=None if bdp is None or r.get('test_smooth_dp_sq') is None else r['test_smooth_dp_sq']-bdp
            da=None if ba is None or r.get('test_auc') is None else r['test_auc']-ba
            dll=None if bll is None or r.get('test_logloss') is None else r['test_logloss']-bll
            agg.append({'dataset':dataset,'outer':outer,'n':n,'model':model,'selector':r['selector'],
                        'delta_smooth_eo_sq':df,'delta_smooth_dp_sq':ddp,'delta_auc':da,'delta_logloss':dll,
                        'fair_win':None if df is None else int(df<0),
                        'guarded_win':None if df is None or da is None or dll is None else int(df<0 and da>=-.01 and dll<=.05)})
    write_csv(out/'aggregate_vs_prevalence_random.csv',agg)
    summary=[]
    for selector in sorted(set(r['selector'] for r in agg)):
        z=[r for r in agg if r['selector']==selector]
        for scope,pred in [('all',lambda r:True),('real',lambda r:not r['dataset'].startswith('synth_')),('synthetic',lambda r:r['dataset'].startswith('synth_'))]:
            q=[r for r in z if pred(r)]
            if not q:continue
            def mean(k):
                v=[r[k] for r in q if r.get(k) is not None];return float(np.mean(v)) if v else None
            summary.append({'selector':selector,'scope':scope,'cells':len(q),'fair_win_rate':mean('fair_win'),'guarded_win_rate':mean('guarded_win'),
                            'mean_delta_smooth_eo_sq':mean('delta_smooth_eo_sq'),'mean_delta_smooth_dp_sq':mean('delta_smooth_dp_sq'),
                            'mean_delta_auc':mean('delta_auc'),'mean_delta_logloss':mean('delta_logloss')})
    write_csv(out/'summary.csv',summary)
    return agg,summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='runs/v2_4_cpu')
    ap.add_argument('--datasets',nargs='+',default=['adult','german','bank','synth_linear','synth_nonlinear','synth_rare_group'])
    ap.add_argument('--sizes',nargs='+',type=int,default=[32,64])
    ap.add_argument('--outer-splits',type=int,default=3)
    ap.add_argument('--models',nargs='+',default=list(MODEL_NAMES))
    ap.add_argument('--envs',type=int,default=6);ap.add_argument('--random-baselines',type=int,default=3)
    ap.add_argument('--include-eodp',action='store_true');ap.add_argument('--retry-errors',action='store_true')
    ap.add_argument('--pool-cap',type=int,default=1000);ap.add_argument('--validation-cap',type=int,default=300);ap.add_argument('--test-cap',type=int,default=600)
    ap.add_argument('--seed',type=int,default=2026092605)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'tasks').mkdir(exist_ok=True)
    c={'seed':a.seed,'explore_fraction':.6,'pool_fraction':.6,'validation_fraction':.2,'pool_cap':a.pool_cap,'validation_cap':a.validation_cap,'test_cap':a.test_cap}
    t0=time.time();status=[];completed=0;failed=0
    log=open(out/'run.log','a',encoding='utf-8')
    def say(s):print(s,flush=True);log.write(s+'\n');log.flush()
    say('ContextBench v2.4 development benchmark start/resume')
    for dataset in a.datasets:
      try:X,y,s,orig,meta=load(dataset)
      except Exception as e:
        status.append({'dataset':dataset,'status':'unavailable','error':repr(e)});say(f'{dataset}: UNAVAILABLE {e!r}');continue
      if not np.all(np.isin(np.unique(s),[0,1])):
        status.append({'dataset':dataset,'status':'no_binary_group'});continue
      status.append({'dataset':dataset,'status':'available','rows':len(y),'source':meta.get('source','')})
      for outer in range(a.outer_splits):
        pool,val,test=split(c,y,s,dataset,'explore',outer,'v24dev')
        Xp,Xv,Xt=transform(X,meta,pool,val,test);yp,yv,yt=y[pool],y[val],y[test];sp,sv,st=s[pool],s[val],s[test]
        say(f'{dataset} outer={outer}: pool={len(yp)} val={len(yv)} test={len(yt)} d={Xp.shape[1]}')
        for k in a.sizes:
          if k>len(yp) or min(np.sum(yp==0),np.sum(yp==1))<2:continue
          try:contexts=build_contexts(out,c,dataset,outer,k,Xp,yp,sp,Xv,yv,sv,a.envs,a.random_baselines,a.include_eodp)
          except Exception as e:
            say(f'{dataset} outer={outer} n={k} selection ERROR {e!r}');traceback.print_exc();continue
          for selector,ix,sm in contexts:
            if len(np.unique(yp[ix]))<2:continue
            for model in a.models:
              tid=task_key(dataset,outer,k,selector,model);tp=out/'tasks'/f'{tid}.json'
              if tp.exists():
                old=json.loads(tp.read_text())
                if old.get('status')=='ok' or (old.get('status')=='error' and not a.retry_errors):continue
              base={'task_id':tid,'dataset':dataset,'outer':outer,'n':k,'selector':selector,'model':model,
                    'positive_fraction':float(yp[ix].mean()),'group_fraction':float(sp[ix].mean()),'context_hash':digest(canon(np.asarray(ix,int).tolist()))}
              try:
                ev=evaluate_model(model,Xp[ix],yp[ix],Xv,yv,sv,Xt,yt,st,seed(c,dataset,outer,k,selector,model))
                rec={**base,'status':'ok',**ev};completed+=1
              except Exception as e:
                rec={**base,'status':'error','error':repr(e),'traceback':traceback.format_exc()};failed+=1
              write_json(tp,rec)
          say(f'{dataset} outer={outer} n={k}: contexts={len(contexts)} model tasks checkpointed')
    task_rows=[]
    for p in sorted((out/'tasks').glob('*.json')):
        try:task_rows.append(json.loads(p.read_text()))
        except Exception:pass
    agg,summary=summarize(out,task_rows);write_csv(out/'dataset_status.csv',status)
    meta={'elapsed_seconds_this_invocation':time.time()-t0,'task_files':len(task_rows),'ok_tasks':sum(r.get('status')=='ok' for r in task_rows),
          'error_tasks':sum(r.get('status')=='error' for r in task_rows),'datasets':a.datasets,'models':a.models,'sizes':a.sizes,'outer_splits':a.outer_splits,
          'include_eodp':a.include_eodp,'development_only':True}
    write_json(out/'SUMMARY.json',{'meta':meta,'summary':summary})
    # Compact markdown report focusing on main method and ablations.
    md=['# ContextBench v2.4 CPU development benchmark','',
        '**Development only. Existing Adult/German/Bank data have already informed method design; they are not final confirmation datasets.**','',
        f"Completed task files: **{meta['ok_tasks']} OK / {meta['error_tasks']} error**.",'',
        'Main method: CVaR50 over proxy×bootstrap fairness environments, exact label-prevalence quota, and a fixed 10th-percentile representation-fidelity constraint via a predeclared Lagrangian path.','',
        '|selector|scope|cells|fair win|guarded win|mean Δsmooth EO²|mean ΔAUC|mean Δlogloss|','|---|---|---:|---:|---:|---:|---:|---:|']
    keep={'cvar50_constrained','cvar50_proxyguard_q50','cvar50_proxyguard_q75','cvar50_gradconstrained','cvar50_constrained_eodp','cvar50_weighted','mean_env_gn','worst_env_gn','single_proxy_gn','diversity','uncertainty','joint_balance','group_balance','label_balance','random'}
    for z in summary:
        if z['selector'] not in keep:continue
        def f(k,fmt='.3f'):
            v=z.get(k);return 'NA' if v is None else format(v,fmt)
        md.append(f"|{z['selector']}|{z['scope']}|{z['cells']}|{f('fair_win_rate')}|{f('guarded_win_rate')}|{f('mean_delta_smooth_eo_sq','+.5f')}|{f('mean_delta_auc','+.4f')}|{f('mean_delta_logloss','+.4f')}|")
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    log.close();print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
