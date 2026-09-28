"""ContextBench v2.4: generic moment-risk context selection.

The method treats fairness as a vector of differentiable probability moments.
A cheap ridge-logistic proxy supplies an influence map from a selected set to
those moments. Bootstrap calibration environments x proxy regularizations form
an uncertainty set. Selection minimizes a risk functional (mean/CVaR/worst)
subject to exact label-prevalence and a data-fidelity constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from v2_method import (_augment, build_proxy, label_targets, representation_penalty,
                       aggregate_environment_risk, envrisk_gn_greedy)

@dataclass
class MomentProxy:
    reg: float
    w: np.ndarray
    grad_samples: np.ndarray
    grad_mean: np.ndarray
    A: np.ndarray          # moments x gradient-dim
    d0: np.ndarray         # signed base moment gaps
    names: tuple[str,...]
    weights: np.ndarray
    Hinv: np.ndarray

    def moment_prediction(self, idx: np.ndarray) -> np.ndarray:
        u=self.grad_samples[np.asarray(idx,int)].mean(axis=0)-self.grad_mean
        return self.d0+self.A@u

    def fairness(self, idx: np.ndarray) -> float:
        z=self.moment_prediction(idx)
        return float(np.sum(self.weights*z*z))

    def gradient_fidelity(self, idx: np.ndarray) -> float:
        u=self.grad_samples[np.asarray(idx,int)].mean(axis=0)-self.grad_mean
        return float(u@self.Hinv@u)


def _diff_mean_prob_jac(Xa,p,v,m0,m1):
    if int(m0.sum())==0 or int(m1.sum())==0:
        raise ValueError('fairness moment undefined: empty cell')
    d=float(p[m0].mean()-p[m1].mean())
    J=(Xa[m0]*v[m0,None]).mean(axis=0)-(Xa[m1]*v[m1,None]).mean(axis=0)
    return d,J


def moment_value_jacobian(X,y,s,w,objective='eo'):
    """Return a generic vector of signed group probability moments.

    objective:
      eo      -> class-conditional gaps for y=0 and y=1
      dp      -> unconditional demographic-parity probability gap
      eo+dp   -> all three moments
    """
    Xa=_augment(X); y=np.asarray(y,int); s=np.asarray(s,int)
    if not np.all(np.isin(np.unique(s),[0,1])):
        raise ValueError('binary group required')
    p=expit(Xa@w); v=p*(1-p)
    names=[]; ds=[]; Js=[]
    if objective in ('dp','eo+dp'):
        d,J=_diff_mean_prob_jac(Xa,p,v,s==0,s==1)
        names.append('dp'); ds.append(d); Js.append(J)
    if objective in ('eo','eo+dp'):
        for yy in (0,1):
            d,J=_diff_mean_prob_jac(Xa,p,v,(y==yy)&(s==0),(y==yy)&(s==1))
            names.append(f'eo_y{yy}'); ds.append(d); Js.append(J)
    if not names: raise ValueError('unknown objective '+str(objective))
    # Equal weight per moment keeps the formulation transparent. Reporting still
    # separates threshold DP/EO metrics downstream.
    return tuple(names),np.asarray(ds,float),np.stack(Js),np.ones(len(names),float)


def build_moment_proxy(Xp,yp,Xc,yc,sc,reg,objective='eo') -> MomentProxy:
    base=build_proxy(Xp,yp,Xc,yc,sc,reg)
    names,d0,J,weights=moment_value_jacobian(Xc,yc,sc,base.w,objective)
    Hinv=np.linalg.inv(base.train_hessian)
    return MomentProxy(reg=float(reg),w=base.w,grad_samples=base.train_grad_samples,
                       grad_mean=base.train_grad_mean,A=-J@Hinv,d0=d0,
                       names=names,weights=weights,Hinv=Hinv)


def _stratified_bootstrap_indices(y,s,r):
    out=[]; y=np.asarray(y,int); s=np.asarray(s,int)
    for yy in (0,1):
      for gg in (0,1):
        ix=np.flatnonzero((y==yy)&(s==gg))
        if len(ix)==0: raise ValueError('missing label/group cell')
        out.append(r.choice(ix,len(ix),replace=True))
    return np.concatenate(out)


def build_bootstrap_moment_proxies(Xp,yp,Xc,yc,sc,regs=(5e-4,5e-3),environments=6,
                                   random_state=0,objective='eo'):
    r=np.random.default_rng(random_state)
    env=[np.arange(len(yc),dtype=int)]
    for _ in range(max(0,environments-1)):
        env.append(_stratified_bootstrap_indices(yc,sc,r))
    return [build_moment_proxy(Xp,yp,Xc[ix],yc[ix],sc[ix],reg,objective)
            for reg in regs for ix in env]


def environment_candidate_matrix(proxies,sum_g,t):
    alpha=1.0/(t+1.0); vals=[]
    for p,sg in zip(proxies,sum_g):
        base_u=alpha*sg-p.grad_mean
        base=p.d0+p.A@base_u
        cand=base[None,:]+alpha*(p.grad_samples@p.A.T)
        vals.append(np.sum((cand*cand)*p.weights[None,:],axis=1))
    return np.stack(vals,axis=0)


def risk_predict(proxies,idx,mode='cvar50',rho=2.0):
    vals=np.asarray([p.fairness(idx) for p in proxies],float)
    risk=float(aggregate_environment_risk(vals[:,None],mode=mode,rho=rho)[0])
    return {'risk':risk,'env_mean':float(vals.mean()),'env_std':float(vals.std()),
            'env_worst':float(vals.max()),'env_values':vals.tolist(),
            'moment_names':list(proxies[0].names)}


def risk_greedy(proxies,X,y,k,mode='cvar50',rho=2.0,representation_weight=.5,
                fairness_scale=None,representation_scale=None,scale_rng=None):
    """Generic version of v2.3 greedy for arbitrary moment vectors."""
    X=np.asarray(X,float); y=np.asarray(y,int); n,d=X.shape; targets=label_targets(y,k)
    if scale_rng is None: scale_rng=np.random.default_rng(0)
    ids={yy:np.flatnonzero(y==yy) for yy in (0,1)}
    if fairness_scale is None or representation_scale is None:
        fs=[]; rs=[]
        for _ in range(40):
            ix=np.sort(np.r_[scale_rng.choice(ids[0],targets[0],replace=False),
                             scale_rng.choice(ids[1],targets[1],replace=False)])
            vv=np.asarray([p.fairness(ix) for p in proxies],float)
            fs.append(float(aggregate_environment_risk(vv[:,None],mode=mode,rho=rho)[0]))
            rs.append(representation_penalty(X,y,ix,targets))
        fairness_scale=max(float(np.median(fs)),1e-6)
        representation_scale=max(float(np.median(rs)),1e-9)
    selected=[]; available=np.ones(n,bool); counts={0:0,1:0}
    sums_g=[np.zeros(p.grad_samples.shape[1],float) for p in proxies]
    sums_x={0:np.zeros(d,float),1:np.zeros(d,float)}
    mus={yy:X[y==yy].mean(axis=0) for yy in (0,1)}
    for t in range(k):
        env=environment_candidate_matrix(proxies,sums_g,t)
        fair=aggregate_environment_risk(env,mode=mode,rho=rho)/fairness_scale
        rep=np.empty(n,float); const={}
        for yy in (0,1):
            target_sum=targets[yy]*mus[yy]
            const[yy]=float(np.mean(((sums_x[yy]-target_sum)/max(targets[yy],1))**2))
        for yy in (0,1):
            mask=y==yy; target_sum=targets[yy]*mus[yy]
            rr=(sums_x[yy][None,:]+X[mask]-target_sum[None,:])/max(targets[yy],1)
            rep[mask]=(np.mean(rr*rr,axis=1)+const[1-yy])/representation_scale
        score=fair+float(representation_weight)*rep
        for yy in (0,1):
            if counts[yy]>=targets[yy]: available[y==yy]=False
        score[~available]=np.inf
        j=int(np.argmin(score))
        if not np.isfinite(score[j]): raise RuntimeError('moment-risk greedy infeasible')
        selected.append(j); available[j]=False; counts[int(y[j])]+=1; sums_x[int(y[j])]+=X[j]
        for q,p in enumerate(proxies): sums_g[q]+=p.grad_samples[j]
    ix=np.asarray(selected,int)
    return ix,{'fairness_scale':float(fairness_scale),'representation_scale':float(representation_scale),
               'target_counts':targets,'mode':mode,'rho':float(rho),'representation_weight':float(representation_weight)}


def constrained_risk_select(proxies,X,y,k,mode='cvar50',rho=2.0,fidelity_quantile=.10,
                            weight_grid=(0.,.05,.1,.25,.5,1.,2.,5.),scale_rng=None):
    """Approximate constrained optimization without dataset-specific eta tuning.

    1) Estimate a fixed fidelity threshold tau from prevalence-matched random sets.
    2) Trace a pre-specified Lagrangian path across representation weights.
    3) Among feasible path solutions, choose the one with minimum fairness risk.
       If no path point is feasible, choose minimum violation then risk.

    The weight grid and fidelity quantile are fixed before downstream evaluation.
    """
    X=np.asarray(X,float); y=np.asarray(y,int); targets=label_targets(y,k)
    if scale_rng is None: scale_rng=np.random.default_rng(0)
    ids={yy:np.flatnonzero(y==yy) for yy in (0,1)}
    fair_samples=[]; rep_samples=[]
    for _ in range(64):
        ix=np.sort(np.r_[scale_rng.choice(ids[0],targets[0],replace=False),
                         scale_rng.choice(ids[1],targets[1],replace=False)])
        vv=np.asarray([p.fairness(ix) for p in proxies],float)
        fair_samples.append(float(aggregate_environment_risk(vv[:,None],mode=mode,rho=rho)[0]))
        rep_samples.append(representation_penalty(X,y,ix,targets))
    fair_scale=max(float(np.median(fair_samples)),1e-6)
    rep_scale=max(float(np.median(rep_samples)),1e-9)
    tau=float(np.quantile(rep_samples,float(fidelity_quantile)))
    candidates=[]
    for eta in weight_grid:
        ix,_=risk_greedy(proxies,X,y,k,mode=mode,rho=rho,representation_weight=float(eta),
                         fairness_scale=fair_scale,representation_scale=rep_scale,scale_rng=scale_rng)
        rep=float(representation_penalty(X,y,ix,targets)); pr=risk_predict(proxies,ix,mode,rho)
        candidates.append((ix,rep,pr['risk'],float(eta)))
    feasible=[z for z in candidates if z[1] <= tau+1e-15]
    if feasible:
        chosen=min(feasible,key=lambda z:(z[2],z[1],z[3])); fallback=False
    else:
        chosen=min(candidates,key=lambda z:(z[1]-tau,z[2],z[3])); fallback=True
    ix,rep,risk,eta=chosen
    return np.sort(ix),{'mode':mode,'rho':float(rho),'fidelity_quantile':float(fidelity_quantile),
                        'fidelity_tau':tau,'representation_penalty':rep,'chosen_eta':eta,
                        'risk':risk,'fallback_no_feasible':fallback,'path_size':len(candidates)}


# ---- v2.4 predictive-fidelity constraint -----------------------------------

def robust_gradient_fidelity(proxies,idx):
    """Worst Hessian-preconditioned pool-gradient mismatch across proxy family."""
    return float(max(p.gradient_fidelity(idx) for p in proxies))


def gradient_fidelity_candidate(proxies,sums_g,t):
    """Worst proxy gradient-fidelity for every candidate at the next greedy step."""
    alpha=1.0/(t+1.0); vals=[]
    for p,sg in zip(proxies,sums_g):
        b=alpha*sg-p.grad_mean
        # q_i=(b+alpha*g_i)^T Hinv (b+alpha*g_i)
        Hb=p.Hinv@b
        base=float(b@Hb)
        cross=2.0*alpha*(p.grad_samples@Hb)
        GH=p.grad_samples@p.Hinv
        diag=np.einsum('ij,ij->i',GH,p.grad_samples)
        vals.append(base+cross+(alpha*alpha)*diag)
    return np.maximum.reduce(vals)


def gradient_fidelity_risk_greedy(proxies,X,y,k,mode='cvar50',rho=2.0,fidelity_weight=.5,
                                  fair_scale=None,grad_scale=None,scale_rng=None):
    """Fairness-risk greedy with a predictive (gradient/Hessian) fidelity penalty."""
    X=np.asarray(X,float);y=np.asarray(y,int);n=len(y);targets=label_targets(y,k)
    if scale_rng is None:scale_rng=np.random.default_rng(0)
    ids={yy:np.flatnonzero(y==yy) for yy in (0,1)}
    if fair_scale is None or grad_scale is None:
        fs=[];gs=[]
        for _ in range(48):
            ix=np.sort(np.r_[scale_rng.choice(ids[0],targets[0],replace=False),scale_rng.choice(ids[1],targets[1],replace=False)])
            vv=np.asarray([p.fairness(ix) for p in proxies],float)
            fs.append(float(aggregate_environment_risk(vv[:,None],mode=mode,rho=rho)[0]))
            gs.append(robust_gradient_fidelity(proxies,ix))
        fair_scale=max(float(np.median(fs)),1e-7);grad_scale=max(float(np.median(gs)),1e-9)
    selected=[];available=np.ones(n,bool);counts={0:0,1:0};sums=[np.zeros(p.grad_samples.shape[1],float) for p in proxies]
    for t in range(k):
        env=environment_candidate_matrix(proxies,sums,t)
        fair=aggregate_environment_risk(env,mode=mode,rho=rho)/fair_scale
        gf=gradient_fidelity_candidate(proxies,sums,t)/grad_scale
        score=fair+float(fidelity_weight)*gf
        for yy in (0,1):
            if counts[yy]>=targets[yy]:available[y==yy]=False
        score[~available]=np.inf;j=int(np.argmin(score))
        if not np.isfinite(score[j]):raise RuntimeError('gradient-fidelity greedy infeasible')
        selected.append(j);available[j]=False;counts[int(y[j])]+=1
        for q,p in enumerate(proxies):sums[q]+=p.grad_samples[j]
    return np.asarray(selected,int),{'fair_scale':float(fair_scale),'grad_scale':float(grad_scale),'fidelity_weight':float(fidelity_weight)}


def constrained_gradient_risk_select(proxies,X,y,k,mode='cvar50',rho=2.0,fidelity_quantile=.10,
                                     weight_grid=(0.,.02,.05,.1,.25,.5,1.,2.,5.),scale_rng=None):
    """CVaR fairness minimization subject to robust predictive-fidelity budget.

    The fidelity statistic is max_m u_m^T H_m^{-1} u_m where u_m is the
    selected-set minus full-pool mean per-example gradient under proxy m.
    The threshold is frozen as a quantile of prevalence-matched random contexts.
    """
    X=np.asarray(X,float);y=np.asarray(y,int);targets=label_targets(y,k)
    if scale_rng is None:scale_rng=np.random.default_rng(0)
    ids={yy:np.flatnonzero(y==yy) for yy in (0,1)};fs=[];gs=[]
    for _ in range(64):
        ix=np.sort(np.r_[scale_rng.choice(ids[0],targets[0],replace=False),scale_rng.choice(ids[1],targets[1],replace=False)])
        vv=np.asarray([p.fairness(ix) for p in proxies],float)
        fs.append(float(aggregate_environment_risk(vv[:,None],mode=mode,rho=rho)[0]));gs.append(robust_gradient_fidelity(proxies,ix))
    fair_scale=max(float(np.median(fs)),1e-7);grad_scale=max(float(np.median(gs)),1e-9);tau=float(np.quantile(gs,float(fidelity_quantile)))
    path=[]
    for eta in weight_grid:
        ix,_=gradient_fidelity_risk_greedy(proxies,X,y,k,mode=mode,rho=rho,fidelity_weight=float(eta),fair_scale=fair_scale,grad_scale=grad_scale,scale_rng=scale_rng)
        gf=robust_gradient_fidelity(proxies,ix);pr=risk_predict(proxies,ix,mode,rho)
        path.append((ix,gf,pr['risk'],float(eta)))
    feasible=[z for z in path if z[1]<=tau+1e-15]
    if feasible:chosen=min(feasible,key=lambda z:(z[2],z[1],z[3]));fallback=False
    else:chosen=min(path,key=lambda z:(z[1]-tau,z[2],z[3]));fallback=True
    ix,gf,risk,eta=chosen
    return np.sort(ix),{'mode':mode,'rho':float(rho),'fidelity_quantile':float(fidelity_quantile),'gradient_fidelity_tau':tau,
                        'gradient_fidelity':gf,'chosen_eta':eta,'risk':risk,'fallback_no_feasible':fallback,'path_size':len(path)}
