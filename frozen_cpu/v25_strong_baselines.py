"""Predeclared stronger context-selection baselines for the external v2.5 run.

All selectors use pool/validation data only; none inspect downstream test metrics.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss
from common import rng,seed
from v2_method import build_proxy,label_targets
from v2_falsification import prevalence_random,farthest
from v24_models import make_model
from v24_benchmark import smooth_prob_gaps


def _quota_pick(scores,y,k,smallest=True):
    y=np.asarray(y,int);targets=label_targets(y,k);parts=[]
    for yy in (0,1):
        ids=np.flatnonzero(y==yy);ord_=np.argsort(scores[ids],kind='stable')
        if not smallest:ord_=ord_[::-1]
        parts.append(ids[ord_[:targets[yy]]])
    return np.sort(np.concatenate(parts))


def uncertainty_quota(Xp,yp,Xv,yv,sv,k,c,dataset,outer):
    p=build_proxy(Xp,yp,Xv,yv,sv,5e-4)
    Xa=np.column_stack([Xp,np.ones(len(Xp))]);prob=expit(Xa@p.w)
    return _quota_pick(np.abs(prob-.5),yp,k,True),{'exact_label_prevalence':True}


def diversity_quota(Xp,yp,k,c,dataset,outer):
    targets=label_targets(yp,k);parts=[]
    for yy in (0,1):
        ids=np.flatnonzero(yp==yy);sub=farthest(Xp[ids],targets[yy],rng(c,dataset,outer,k,'v25-divquota',yy));parts.append(ids[sub])
    return np.sort(np.concatenate(parts)),{'exact_label_prevalence':True,'method':'class-conditional farthest-point'}


def gradmatch_quota(Xp,yp,Xv,yv,sv,k,c,dataset,outer):
    """Greedy pool-gradient mean matching with exact label-prevalence quota."""
    p=build_proxy(Xp,yp,Xv,yv,sv,5e-4);G=np.asarray(p.train_grad_samples,float);target=np.asarray(p.train_grad_mean,float)
    targets=label_targets(yp,k);counts={0:0,1:0};avail=np.ones(len(yp),bool);sumg=np.zeros(G.shape[1]);chosen=[]
    for t in range(k):
        a=1.0/(t+1);cand=a*(sumg[None,:]+G)-target[None,:];score=np.einsum('ij,ij->i',cand,cand)
        for yy in (0,1):
            if counts[yy]>=targets[yy]:avail[yp==yy]=False
        score[~avail]=np.inf;j=int(np.argmin(score));chosen.append(j);avail[j]=False;counts[int(yp[j])]+=1;sumg+=G[j]
    return np.sort(np.asarray(chosen,int)),{'exact_label_prevalence':True,'method':'greedy gradient-mean matching'}


def _fit_val(model_name,Xp,yp,ix,Xv,yv,sv,sd):
    m=make_model(model_name,sd);m.fit(Xp[ix],yp[ix]);p=np.asarray(m.predict_proba(Xv)[:,list(m.classes_).index(1)],float)
    dp,eo=smooth_prob_gaps(p,yv,sv);eo2=np.inf if eo is None else float(sum(v*v for v in eo));ll=float(log_loss(yv,np.clip(p,1e-8,1-1e-8),labels=[0,1]))
    return eo2,ll


def validation_random_search(Xp,yp,Xv,yv,sv,k,c,dataset,outer,draws=16,models=('logistic','histgb')):
    """Strong search baseline: choose among fixed prevalence-matched random contexts.

    Feasibility is the per-proxy median validation log-loss across the same fixed
    candidate pool; among feasible candidates choose the smallest mean validation
    smooth-EO^2. This is deliberately strong but search-based, not a set surrogate.
    """
    rows=[]
    for d in range(draws):
        ix=prevalence_random(yp,k,rng(c,dataset,outer,k,'v25-valsearch',d));vals={}
        for m in models:vals[m]=_fit_val(m,Xp,yp,ix,Xv,yv,sv,seed(c,dataset,outer,k,'v25-valsearch',d,m))
        rows.append((np.sort(ix),vals,d))
    tau={m:float(np.median([z[1][m][1] for z in rows])) for m in models}
    scored=[]
    for ix,vals,d in rows:
        fair=float(np.mean([vals[m][0] for m in models]));viol=max(max(0.0,(vals[m][1]-tau[m])/max(tau[m],1e-12)) for m in models)
        scored.append((ix,fair,viol,d,vals))
    feasible=[z for z in scored if z[2]<=1e-15]
    if feasible:z=min(feasible,key=lambda q:(q[1],q[3]));fallback=False
    else:z=min(scored,key=lambda q:(q[2],q[1],q[3]));fallback=True
    return z[0],{'draws':draws,'models':list(models),'utility_tau':tau,'validation_fairness':z[1],'utility_violation':z[2],'chosen_draw':z[3],'fallback_no_feasible':fallback}


def all_strong_baselines(Xp,yp,Xv,yv,sv,k,c,dataset,outer):
    out=[]
    for name,fn in [
        ('uncertainty_quota',lambda:uncertainty_quota(Xp,yp,Xv,yv,sv,k,c,dataset,outer)),
        ('diversity_quota',lambda:diversity_quota(Xp,yp,k,c,dataset,outer)),
        ('gradmatch_quota',lambda:gradmatch_quota(Xp,yp,Xv,yv,sv,k,c,dataset,outer)),
        ('valfair_random_search16',lambda:validation_random_search(Xp,yp,Xv,yv,sv,k,c,dataset,outer,16)),
    ]:
        ix,meta=fn();out.append((name,ix,meta))
    return out
