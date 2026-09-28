"""ContextBench v2.6: proposal-verified context selection.

Core principle
--------------
Surrogates/heuristics are *proposal mechanisms*, not final decision rules.
A finite heterogeneous bank of candidate contexts is generated using the pool
and a proposal-calibration split only. A disjoint verifier split then evaluates
all candidates with a small family of cheap learners. The final context is the
fairness-best candidate among those satisfying cross-verifier utility guards.

This avoids the v2.5 failure mode where optimizing an approximate surrogate
all the way to the final context improved fairness but transferred poorly in
utility to unseen downstream learners.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from v24_benchmark import baseline_context, evaluate_model
from v24_method import (build_bootstrap_moment_proxies, risk_greedy,
                        constrained_risk_select, constrained_gradient_risk_select)
from v2_method import build_proxy
from v25_strong_baselines import uncertainty_quota, diversity_quota, gradmatch_quota
from common import rng, seed

FROZEN_PROPOSAL_NAMES = (
    'prevalence_random_0','prevalence_random_1','prevalence_random_2','prevalence_random_3','prevalence_random_4',
    'random','label_balance','group_balance','joint_balance','uncertainty','diversity',
    'uncertainty_quota','diversity_quota','gradmatch_quota',
    'single_proxy_gn','mean_env_gn','cvar50_weighted','cvar50_constrained','cvar50_gradconstrained',
)
FROZEN_VERIFIER_MODELS = ('logistic','histgb','random_forest')


def build_proposal_bank(Xp,yp,sp,Xc,yc,sc,k,c,dataset,outer,*,envs=8,random_refs=5):
    """Generate the frozen v2.6 19-context proposal bank.

    Candidate generation may use only pool P and proposal-calibration C.
    It must never access verifier V or test T.
    """
    if int(random_refs) != 5:
        raise ValueError('v2.6 frozen protocol requires exactly 5 prevalence-random anchors')
    regs=(5e-4,5e-3)
    gps=build_bootstrap_moment_proxies(
        Xp,yp,Xc,yc,sc,regs=regs,environments=int(envs),
        random_state=seed(c,dataset,outer,k,'v26-proposal-eo-envs'),objective='eo')
    base=build_proxy(Xp,yp,Xc,yc,sc,regs[0])
    p_unc=expit(np.column_stack([Xp,np.ones(len(Xp))])@base.w)
    rows=[]
    for d in range(5):
        ix=baseline_context('prevalence_random',Xp,yp,sp,k,rng(c,dataset,outer,k,'v26-pr',d),p_unc)
        rows.append((f'prevalence_random_{d}',np.sort(ix),{'family':'prevalence_random','draw':d}))
    ix=baseline_context('random',Xp,yp,sp,k,rng(c,dataset,outer,k,'v26-random'),p_unc);rows.append(('random',np.sort(ix),{'family':'random'}))
    for b in ('label_balance','group_balance','joint_balance','uncertainty','diversity'):
        ix=baseline_context(b,Xp,yp,sp,k,rng(c,dataset,outer,k,'v26-'+b),p_unc);rows.append((b,np.sort(ix),{'family':b}))
    # Strong quota/gradient proposals use proposal-calibration only.
    ix,sm=uncertainty_quota(Xp,yp,Xc,yc,sc,k,c,dataset,outer);rows.append(('uncertainty_quota',np.sort(ix),sm))
    ix,sm=diversity_quota(Xp,yp,k,c,dataset,outer);rows.append(('diversity_quota',np.sort(ix),sm))
    ix,sm=gradmatch_quota(Xp,yp,Xc,yc,sc,k,c,dataset,outer);rows.append(('gradmatch_quota',np.sort(ix),sm))
    # Surrogate proposals. The surrogate only proposes; verifier chooses final context.
    ix,sm=risk_greedy([gps[0]],Xp,yp,k,mode='mean',representation_weight=.5,
                      scale_rng=rng(c,dataset,outer,k,'v26-single'));rows.append(('single_proxy_gn',np.sort(ix),sm))
    ix,sm=risk_greedy(gps,Xp,yp,k,mode='mean',representation_weight=.5,
                      scale_rng=rng(c,dataset,outer,k,'v26-mean'));rows.append(('mean_env_gn',np.sort(ix),sm))
    ix,sm=risk_greedy(gps,Xp,yp,k,mode='cvar50',representation_weight=.5,
                      scale_rng=rng(c,dataset,outer,k,'v26-cvarw'));rows.append(('cvar50_weighted',np.sort(ix),sm))
    ix,sm=constrained_risk_select(gps,Xp,yp,k,mode='cvar50',fidelity_quantile=.10,
                                  scale_rng=rng(c,dataset,outer,k,'v26-cvarc'));rows.append(('cvar50_constrained',np.sort(ix),sm))
    ix,sm=constrained_gradient_risk_select(gps,Xp,yp,k,mode='cvar50',fidelity_quantile=.10,
                                           scale_rng=rng(c,dataset,outer,k,'v26-cvargrad'));rows.append(('cvar50_gradconstrained',np.sort(ix),sm))
    names=[z[0] for z in rows]
    if tuple(names)!=FROZEN_PROPOSAL_NAMES:
        raise AssertionError(f'proposal bank drift: {names}')
    return rows


def _finite(v, bad):
    try:
        x=float(v)
        return x if np.isfinite(x) else bad
    except Exception:
        return bad


def verify_and_select(proposals,Xp,yp,Xv,yv,sv,c,dataset,outer,k,*,
                      verifier_models=FROZEN_VERIFIER_MODELS,
                      auc_slack=0.005,logloss_reference_quantile=0.75):
    """Exact finite-candidate verifier for the frozen v2.6 rule.

    Utility guard, independently for each verifier model m:
      val_AUC(S,m) >= mean_random_ref_AUC(m) - 0.005
      val_logloss(S,m) <= q75_random_ref_logloss(m)

    Among feasible candidates, minimize mean normalized smooth-EO^2 across
    verifier models. If no candidate is feasible, minimize max normalized
    utility violation first (then sum violation, then fairness). Random anchors
    themselves are eligible proposals, making fallback rare by construction.
    """
    verifier_models=tuple(verifier_models)
    rows=[]
    for name,ix,meta in proposals:
        vals={}
        for m in verifier_models:
            ev=evaluate_model(m,Xp[ix],yp[ix],Xv,yv,sv,Xv,yv,sv,
                              seed(c,'v26-verify',dataset,outer,k,name,m))
            vals[m]={
                'auc':_finite(ev.get('val_auc'),-np.inf),
                'logloss':_finite(ev.get('val_logloss'),np.inf),
                'smooth_eo_sq':_finite(ev.get('val_smooth_eo_sq'),np.inf),
                'smooth_dp_sq':_finite(ev.get('val_smooth_dp_sq'),np.inf),
            }
        rows.append({'selector':name,'indices':np.asarray(ix,int),'proposal_meta':meta,'vals':vals})
    refs=[r for r in rows if r['selector'].startswith('prevalence_random_')]
    if len(refs)!=5:raise RuntimeError('v2.6 verifier requires five prevalence-random anchors')
    auc_floor={m:float(np.mean([r['vals'][m]['auc'] for r in refs])-auc_slack) for m in verifier_models}
    ll_ceil={m:float(np.quantile([r['vals'][m]['logloss'] for r in refs],logloss_reference_quantile)) for m in verifier_models}
    eo_scale={m:max(float(np.mean([r['vals'][m]['smooth_eo_sq'] for r in refs])),1e-6) for m in verifier_models}
    scored=[]
    for r in rows:
        violations=[]
        for m in verifier_models:
            a=r['vals'][m]['auc'];ll=r['vals'][m]['logloss']
            violations.append(max(0.0,(auc_floor[m]-a)/max(abs(auc_floor[m]),1e-6)))
            violations.append(max(0.0,(ll-ll_ceil[m])/max(abs(ll_ceil[m]),1e-6)))
        max_v=float(max(violations));sum_v=float(sum(violations))
        fair=float(np.mean([r['vals'][m]['smooth_eo_sq']/eo_scale[m] for m in verifier_models]))
        scored.append({**r,'max_violation':max_v,'sum_violation':sum_v,'fairness_score':fair})
    feasible=[r for r in scored if r['max_violation']<=1e-15]
    if feasible:
        chosen=min(feasible,key=lambda r:(r['fairness_score'],r['sum_violation'],r['selector']));fallback=False
    else:
        chosen=min(scored,key=lambda r:(r['max_violation'],r['sum_violation'],r['fairness_score'],r['selector']));fallback=True
    meta={
        'method':'v26_proposal_verifier','proposal_count':len(scored),'verifier_models':list(verifier_models),
        'auc_slack':float(auc_slack),'logloss_reference_quantile':float(logloss_reference_quantile),
        'auc_floor':auc_floor,'logloss_ceiling':ll_ceil,'eo_scale':eo_scale,
        'chosen_proposal':chosen['selector'],'chosen_fairness_score':float(chosen['fairness_score']),
        'max_normalized_utility_violation':float(chosen['max_violation']),
        'sum_normalized_utility_violation':float(chosen['sum_violation']),
        'fallback_no_feasible':bool(fallback),
        'finite_candidate_verification':True,
    }
    compact=[]
    for r in scored:
        compact.append({'selector':r['selector'],'fairness_score':float(r['fairness_score']),
                        'max_violation':float(r['max_violation']),'sum_violation':float(r['sum_violation']),
                        'verifier_metrics':r['vals']})
    meta['candidate_scores']=compact
    return np.sort(chosen['indices']),meta
