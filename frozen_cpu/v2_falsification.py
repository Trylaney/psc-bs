"""CPU falsification run for ContextBench v2.1 robust Gauss-Newton selector."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score

from common import seed,rng
from data import load,split
from design import transform,balanced
from v2_method import (build_proxy,build_gap_proxy,build_crossfit_gap_proxies,robust_gn_greedy,robust_gn_predict,
                       label_targets,representation_penalty,estimate_scales)


def smooth_eo_sq_prob(p,y,s):
    vals=[]
    for yy in (0,1):
        m0=(y==yy)&(s==0);m1=(y==yy)&(s==1)
        if m0.sum()==0 or m1.sum()==0:return float('nan')
        vals.append(float(p[m0].mean()-p[m1].mean()))
    return float(sum(v*v for v in vals))

def threshold_gaps(p,y,s,t=.5):
    q=(p>=t).astype(int);dp=abs(float(q[s==0].mean()-q[s==1].mean()));ds=[]
    for yy in (0,1):
        m0=(y==yy)&(s==0);m1=(y==yy)&(s==1)
        if m0.sum()==0 or m1.sum()==0:return dp,float('nan')
        ds.append(abs(float(q[m0].mean()-q[m1].mean())))
    return dp,max(ds)

def metrics(p,y,s):
    dp,eo=threshold_gaps(p,y,s)
    return {'auc':float(roc_auc_score(y,p)),'logloss':float(log_loss(y,p,labels=[0,1])),
            'accuracy':float(accuracy_score(y,p>=.5)),'smooth_eo_sq':smooth_eo_sq_prob(p,y,s),
            'dp_gap':dp,'eo_gap':eo}

def evaluator(name,Xs,ys,Xv,yv,sv,Xt,yt,st,seed_value):
    if name=='logistic':m=LogisticRegression(C=1,max_iter=1000,random_state=seed_value)
    elif name=='histgb':m=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,min_samples_leaf=5,
          l2_regularization=1,early_stopping=False,random_state=seed_value)
    else:raise ValueError(name)
    m.fit(Xs,ys);pv=m.predict_proba(Xv)[:,list(m.classes_).index(1)];pt=m.predict_proba(Xt)[:,list(m.classes_).index(1)]
    out={f'val_{k}':v for k,v in metrics(pv,yv,sv).items()};out.update({f'test_{k}':v for k,v in metrics(pt,yt,st).items()});return out

def farthest(X,k,r):
    first=int(r.integers(len(X)));sel=[first];dist=np.sum((X-X[first])**2,axis=1);dist[first]=-1
    while len(sel)<k:
        j=int(np.argmax(dist));sel.append(j);dist=np.minimum(dist,np.sum((X-X[j])**2,axis=1));dist[sel]=-1
    return np.asarray(sel,int)

def prevalence_random(y,k,r):
    t=label_targets(y,k);return np.sort(np.r_[r.choice(np.flatnonzero(y==0),t[0],replace=False),r.choice(np.flatnonzero(y==1),t[1],replace=False)])

def baseline(name,X,y,s,k,r,proxy_prob):
    if name=='random':return np.sort(r.choice(len(y),k,replace=False))
    if name=='prevalence_random':return prevalence_random(y,k,r)
    if name=='label_balance':return balanced(y,k,r)
    if name=='joint_balance':return balanced(2*y+s,k,r)
    if name=='uncertainty':return np.argsort(np.abs(proxy_prob-.5),kind='stable')[:k]
    if name=='diversity':return np.sort(farthest(X,k,r))
    raise ValueError(name)

def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<5 or np.std(a[m])<1e-15 or np.std(b[m])<1e-15:return float('nan')
    return float(spearmanr(a[m],b[m]).statistic)

def write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='runs/v2_1_falsification');ap.add_argument('--datasets',nargs='+',default=['adult','german'])
    ap.add_argument('--sizes',nargs='+',type=int,default=[32,64]);ap.add_argument('--random-contexts',type=int,default=32);ap.add_argument('--calibration-folds',type=int,default=1)
    ap.add_argument('--pool-cap',type=int,default=1000);ap.add_argument('--validation-cap',type=int,default=400);ap.add_argument('--test-cap',type=int,default=800);ap.add_argument('--seed',type=int,default=2026092602)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    c={'seed':a.seed,'explore_fraction':.6,'pool_fraction':.6,'validation_fraction':.2,'pool_cap':a.pool_cap,'validation_cap':a.validation_cap,'test_cap':a.test_cap}
    rows=[];probe=[];summaries=[];start=time.time();log=open(out/'run.log','w',encoding='utf-8')
    def say(x):print(x,flush=True);log.write(x+'\n');log.flush()
    say('ContextBench v2.1 robust-GN falsification start')
    for dataset in a.datasets:
        X,y,s,orig,meta=load(dataset);pool,val,test=split(c,y,s,dataset,'explore',0,'v2f')
        Xp,Xv,Xt=transform(X,meta,pool,val,test);yp,yv,yt=y[pool],y[val],y[test];sp,sv,st=s[pool],s[val],s[test]
        regs=[5e-4,5e-3];gps=build_crossfit_gap_proxies(Xp,yp,Xv,yv,sv,regs=regs,folds=a.calibration_folds,random_state=seed(c,dataset,'calibration-folds'));basep=build_proxy(Xp,yp,Xv,yv,sv,regs[0])
        p_unc=expit(np.column_stack([Xp,np.ones(len(Xp))])@basep.w);say(f'{dataset}: pool={len(yp)} val={len(yv)} test={len(yt)} d={Xp.shape[1]}')
        for k in a.sizes:
            if k>len(yp):continue
            pred=[];actual={m:{'val':[],'test':[]} for m in ('logistic','histgb')}
            for draw in range(a.random_contexts):
                ix=np.sort(rng(c,dataset,k,'gn-probe',draw).choice(len(yp),k,replace=False));pr=robust_gn_predict(gps,ix);pred.append(pr['worst_pred_smooth_eo_sq'])
                for m in ('logistic','histgb'):
                    ev=evaluator(m,Xp[ix],yp[ix],Xv,yv,sv,Xt,yt,st,seed(c,dataset,k,'probe',draw,m));actual[m]['val'].append(ev['val_smooth_eo_sq']);actual[m]['test'].append(ev['test_smooth_eo_sq'])
                    probe.append({'dataset':dataset,'n':k,'draw':draw,'model':m,**pr,**ev})
            ss={'dataset':dataset,'n':k,'random_contexts':a.random_contexts,
                'logistic_rho_predF_val_fair':corr(pred,actual['logistic']['val']),
                'histgb_rho_predF_val_fair':corr(pred,actual['histgb']['val']),
                'logistic_rho_predF_test_fair':corr(pred,actual['logistic']['test']),
                'histgb_rho_predF_test_fair':corr(pred,actual['histgb']['test'])}
            summaries.append(ss);say('CORR '+json.dumps(ss))
            fscale,rscale=estimate_scales(gps,Xp,yp,k,rng(c,dataset,k,'scales'),draws=32)
            methods=[]
            for b in ('label_balance','joint_balance','uncertainty','diversity'):
                methods.append((b,baseline(b,Xp,yp,sp,k,rng(c,dataset,k,b),p_unc),None))
            for eta in (0.0,0.1,0.5,1.0,2.0):
                ix,meta_sel=robust_gn_greedy(gps,Xp,yp,k,representation_weight=eta,fairness_scale=fscale,representation_scale=rscale)
                methods.append((f'robust_gn_eta{eta:g}',np.sort(ix),meta_sel))
            ix,meta_sel=robust_gn_greedy([gps[0]],Xp,yp,k,representation_weight=.5,fairness_scale=fscale,representation_scale=rscale)
            methods.append(('single_proxy_gn_eta0.5',np.sort(ix),meta_sel))
            for draw in range(8):
                methods.append((f'random_{draw}',baseline('random',Xp,yp,sp,k,rng(c,dataset,k,'random',draw),p_unc),None))
                methods.append((f'prevalence_random_{draw}',baseline('prevalence_random',Xp,yp,sp,k,rng(c,dataset,k,'prandom',draw),p_unc),None))
            for name,ix,sm in methods:
                pr=robust_gn_predict(gps,ix);rep=representation_penalty(Xp,yp,ix,label_targets(yp,k))
                common={'dataset':dataset,'n':k,'selector':name,'positive_fraction':float(yp[ix].mean()),'group_fraction':float(sp[ix].mean()),
                        'pred_smooth_eo_sq':pr['worst_pred_smooth_eo_sq'],'representation_penalty':rep}
                for m in ('logistic','histgb'):
                    rows.append({**common,'model':m,**evaluator(m,Xp[ix],yp[ix],Xv,yv,sv,Xt,yt,st,seed(c,dataset,k,name,m))})
            say(f'{dataset} n={k}: selector comparison done')
    write_csv(out/'selector_results.csv',rows);write_csv(out/'random_probe_results.csv',probe);write_csv(out/'surrogate_correlations.csv',summaries)
    primary=[z['logistic_rho_predF_val_fair'] for z in summaries if np.isfinite(z['logistic_rho_predF_val_fair'])];median=float(np.median(primary)) if primary else float('nan')
    transfer=[z['histgb_rho_predF_val_fair'] for z in summaries if np.isfinite(z['histgb_rho_predF_val_fair'])];median_transfer=float(np.median(transfer)) if transfer else float('nan')
    gate={'primary_median_proxy_retrain_spearman':median,'primary_gate_pass':bool(np.isfinite(median) and median>=.35),
          'primary_gate':'median Spearman(GN fairness surrogate, exact logistic retrain fairness on calibration) >= 0.35',
          'histgb_transfer_median_spearman':median_transfer,'elapsed_seconds':time.time()-start}
    (out/'SUMMARY.json').write_text(json.dumps({'gate':gate,'correlations':summaries},indent=2),encoding='utf-8')
    md=['# ContextBench v2.1 — CPU Falsification Report','',
        'Method: cross-fitted robust Gauss–Newton set-level fairness surrogate + exact label-prevalence constraint + class-conditional mean-matching utility guard.','',
        f"**Primary surrogate gate:** {median:.3f} — **{'PASS' if gate['primary_gate_pass'] else 'FAIL'}** (threshold 0.35).",
        f"**Unseen HistGB correlation (diagnostic, not gate):** median {median_transfer:.3f}.",'','## Cell-level correlations','',
        '|dataset|n|Logistic val ρ|HistGB val ρ|Logistic test ρ|HistGB test ρ|','|---|---:|---:|---:|---:|---:|']
    for z in summaries:md.append(f"|{z['dataset']}|{z['n']}|{z['logistic_rho_predF_val_fair']:.3f}|{z['histgb_rho_predF_val_fair']:.3f}|{z['logistic_rho_predF_test_fair']:.3f}|{z['histgb_rho_predF_test_fair']:.3f}|")
    md += ['','`selector_results.csv` contains actual retraining results for baselines, raw random, prevalence-matched random, the robust-GN eta frontier, and a single-proxy ablation.','',
           'Passing this gate justifies a larger CPU confirmation of the method. It does not justify a GPU/TabICL run by itself.']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8');log.close();print(json.dumps(gate,indent=2),flush=True)
if __name__=='__main__':main()
