"""ContextBench v2 prototype: robust second-order set-level context selection.

The method uses a small family of differentiable ridge-logistic proxy learners.
For each proxy, it linearizes the retraining parameter shift induced by a selected
context and uses second-order Taylor surrogates for validation log-loss and a
smooth equalized-odds objective. A greedy selector minimizes the worst normalized
surrogate objective across proxies.

This is deliberately a falsifiable CPU prototype, not a claim that the same
approximation theorem applies to black-box downstream models.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit

EPS = 1e-12


def _augment(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([X, np.ones(len(X), dtype=np.float64)])


def _logloss_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _ridge_mask(d: int) -> np.ndarray:
    m = np.ones(d, dtype=np.float64)
    m[-1] = 0.0  # do not penalize intercept
    return m


def fit_ridge_logistic_newton(X: np.ndarray, y: np.ndarray, reg: float,
                              max_iter: int = 80, tol: float = 1e-9) -> np.ndarray:
    """Deterministic Newton solver for mean logistic loss + 0.5*reg*||w||^2."""
    Xa = _augment(X)
    y = np.asarray(y, dtype=np.float64)
    n, d = Xa.shape
    mask = _ridge_mask(d)
    w = np.zeros(d, dtype=np.float64)
    prev = np.inf
    for _ in range(max_iter):
        z = Xa @ w
        p = expit(z)
        v = np.maximum(p * (1.0 - p), 1e-8)
        grad = Xa.T @ (p - y) / n + reg * mask * w
        H = (Xa.T * v) @ Xa / n + np.diag(reg * mask + 1e-8)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=1e-10)[0]
        # Damped Newton for numerical robustness.
        base = _logloss_from_logits(z, y) + 0.5 * reg * float(np.sum((mask*w)**2))
        alpha = 1.0
        while alpha > 1e-5:
            wn = w - alpha * step
            zn = Xa @ wn
            loss = _logloss_from_logits(zn, y) + 0.5 * reg * float(np.sum((mask*wn)**2))
            if loss <= base + 1e-12:
                w = wn
                prev = loss
                break
            alpha *= 0.5
        if np.linalg.norm(alpha * step) <= tol * (1.0 + np.linalg.norm(w)):
            break
    return w


def loss_grad_hess(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    Xa = _augment(X)
    y = np.asarray(y, dtype=np.float64)
    z = Xa @ w
    p = expit(z)
    v = p * (1.0 - p)
    loss = _logloss_from_logits(z, y)
    grad = Xa.T @ (p - y) / len(y)
    H = (Xa.T * v) @ Xa / len(y)
    return loss, grad, H, p


def smooth_eo_value_grad_hess(X: np.ndarray, y: np.ndarray, s: np.ndarray,
                              w: np.ndarray):
    """Squared smooth EO: sum_y (E[p|y,s=0]-E[p|y,s=1])^2."""
    Xa = _augment(X)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=int)
    p = expit(Xa @ w)
    v = p * (1.0 - p)
    curvature = v * (1.0 - 2.0 * p)
    d = Xa.shape[1]
    phi = 0.0
    grad = np.zeros(d, dtype=np.float64)
    H = np.zeros((d, d), dtype=np.float64)
    valid = 0
    for yy in (0, 1):
        masks = [(y == yy) & (s == gg) for gg in (0, 1)]
        if min(m.sum() for m in masks) == 0:
            continue
        m0, m1 = masks
        diff = float(p[m0].mean() - p[m1].mean())
        gd = (Xa[m0] * v[m0, None]).mean(axis=0) - (Xa[m1] * v[m1, None]).mean(axis=0)
        h0 = np.einsum('n,ni,nj->ij', curvature[m0], Xa[m0], Xa[m0]) / m0.sum()
        h1 = np.einsum('n,ni,nj->ij', curvature[m1], Xa[m1], Xa[m1]) / m1.sum()
        hd = h0 - h1
        phi += diff * diff
        grad += 2.0 * diff * gd
        H += 2.0 * np.outer(gd, gd) + 2.0 * diff * hd
        valid += 1
    if valid == 0:
        raise ValueError('smooth EO undefined: missing label/group cells')
    return float(phi), grad, H


@dataclass
class QuadraticComponent:
    const: float
    linear_u: np.ndarray
    quad_u: np.ndarray
    g_linear: np.ndarray
    g_quad: np.ndarray
    # g_quad_rows[i] = g_i^T Q as a row vector, precomputed for O(Nd) greedy steps.
    g_quad_rows: np.ndarray

    def candidates(self, g: np.ndarray, gbar: np.ndarray, sum_g: np.ndarray, t: int) -> np.ndarray:
        alpha = 1.0 / (t + 1.0)
        b = sum_g * alpha - gbar
        base = self.const + float(b @ self.linear_u) + 0.5 * float(b @ self.quad_u @ b)
        return (base
                + alpha * self.g_linear
                + alpha * (self.g_quad_rows @ b)
                + 0.5 * alpha * alpha * self.g_quad)

    def value_for_indices(self, g: np.ndarray, gbar: np.ndarray, idx: np.ndarray) -> float:
        u = g[idx].mean(axis=0) - gbar
        return float(self.const + u @ self.linear_u + 0.5 * u @ self.quad_u @ u)


@dataclass
class ProxySurrogate:
    reg: float
    w: np.ndarray
    train_hessian: np.ndarray
    train_grad_samples: np.ndarray
    train_grad_mean: np.ndarray
    base_val_loss: float
    base_fairness: float
    utility: QuadraticComponent
    fairness: QuadraticComponent

    def score_candidates(self, sum_g: np.ndarray, t: int, fairness_weight: float,
                         fair_floor: float = 1e-3) -> np.ndarray:
        uhat = self.utility.candidates(self.train_grad_samples, self.train_grad_mean, sum_g, t)
        fhat = self.fairness.candidates(self.train_grad_samples, self.train_grad_mean, sum_g, t)
        # Taylor approximations can become slightly negative away from expansion point.
        uhat = np.maximum(uhat, 1e-8)
        fhat = np.maximum(fhat, 0.0)
        return uhat / max(self.base_val_loss, 1e-6) + fairness_weight * fhat / max(self.base_fairness, fair_floor)

    def predict_set(self, idx: np.ndarray, fairness_weight: float, fair_floor: float = 1e-3):
        u = max(self.utility.value_for_indices(self.train_grad_samples, self.train_grad_mean, idx), 1e-8)
        f = max(self.fairness.value_for_indices(self.train_grad_samples, self.train_grad_mean, idx), 0.0)
        score = u / max(self.base_val_loss, 1e-6) + fairness_weight * f / max(self.base_fairness, fair_floor)
        return {'pred_val_logloss': float(u), 'pred_smooth_eo_sq': float(f), 'normalized_score': float(score)}


def build_proxy(Xp: np.ndarray, yp: np.ndarray, Xc: np.ndarray, yc: np.ndarray, sc: np.ndarray,
                reg: float) -> ProxySurrogate:
    Xpa = _augment(Xp)
    ypf = np.asarray(yp, dtype=np.float64)
    w = fit_ridge_logistic_newton(Xp, yp, reg=reg)
    pp = expit(Xpa @ w)
    v = np.maximum(pp * (1.0 - pp), 1e-8)
    mask = _ridge_mask(Xpa.shape[1])
    Htrain = (Xpa.T * v) @ Xpa / len(yp) + np.diag(reg * mask + 1e-8)
    # Per-sample *data* gradients. Regularizer cancels when comparing subset vs full pool.
    gs = Xpa * (pp - ypf)[:, None]
    gbar = gs.mean(axis=0)
    Hinv = np.linalg.inv(Htrain)

    L0, gL, HL, _ = loss_grad_hess(Xc, yc, w)
    F0, gF, HF = smooth_eo_value_grad_hess(Xc, yc, sc, w)

    def component(const, a, B):
        # delta_theta = -H^{-1} u
        linear = -(Hinv @ a)
        Q = Hinv @ B @ Hinv
        Q = 0.5 * (Q + Q.T)
        gl = gs @ linear
        gq_rows = gs @ Q
        gq = np.einsum('ij,ij->i', gq_rows, gs)
        return QuadraticComponent(float(const), linear, Q, gl, gq, gq_rows)

    return ProxySurrogate(reg=reg, w=w, train_hessian=Htrain,
                          train_grad_samples=gs, train_grad_mean=gbar,
                          base_val_loss=float(L0), base_fairness=float(F0),
                          utility=component(L0, gL, HL), fairness=component(F0, gF, HF))


def robust_greedy(proxies: list[ProxySurrogate], k: int, fairness_weight: float,
                  candidate_mask: np.ndarray | None = None) -> np.ndarray:
    """Greedy minimization of worst normalized quadratic proxy score."""
    n = len(proxies[0].train_grad_samples)
    if k > n:
        raise ValueError('k exceeds pool')
    if candidate_mask is None:
        candidate_mask = np.ones(n, dtype=bool)
    available = candidate_mask.copy()
    selected = []
    sums = [np.zeros(p.train_grad_samples.shape[1], dtype=np.float64) for p in proxies]
    for t in range(k):
        scores = []
        for p, sg in zip(proxies, sums):
            scores.append(p.score_candidates(sg, t, fairness_weight))
        worst = np.maximum.reduce(scores)
        worst[~available] = np.inf
        j = int(np.argmin(worst))
        if not np.isfinite(worst[j]):
            raise RuntimeError('no feasible candidate remains')
        selected.append(j)
        available[j] = False
        for z, p in enumerate(proxies):
            sums[z] += p.train_grad_samples[j]
    return np.asarray(selected, dtype=int)


def robust_predict_set(proxies: list[ProxySurrogate], idx: np.ndarray, fairness_weight: float):
    rows = [p.predict_set(idx, fairness_weight) for p in proxies]
    return {
        'proxy_rows': rows,
        'worst_pred_val_logloss': float(max(r['pred_val_logloss'] for r in rows)),
        'worst_pred_smooth_eo_sq': float(max(r['pred_smooth_eo_sq'] for r in rows)),
        'worst_normalized_score': float(max(r['normalized_score'] for r in rows)),
    }

# ---- v2.1: PSD Gauss-Newton fairness surrogate + prevalence/representation guard ----

@dataclass
class GapProxy:
    reg: float
    w: np.ndarray
    grad_samples: np.ndarray
    grad_mean: np.ndarray
    A_gap: np.ndarray   # shape [2,d], maps gradient perturbation u to gap perturbation
    d0: np.ndarray      # baseline smooth EO signed gaps, shape [2]

    def gap_prediction(self, idx: np.ndarray):
        u = self.grad_samples[idx].mean(axis=0) - self.grad_mean
        return self.d0 + self.A_gap @ u

    def fairness(self, idx: np.ndarray):
        z = self.gap_prediction(idx)
        return float(z @ z)


def _gap_value_jacobian(X: np.ndarray, y: np.ndarray, s: np.ndarray, w: np.ndarray):
    Xa = _augment(X)
    y = np.asarray(y, int);s=np.asarray(s,int)
    p = expit(Xa @ w);v=p*(1-p)
    ds=[];js=[]
    for yy in (0,1):
        m0=(y==yy)&(s==0);m1=(y==yy)&(s==1)
        if m0.sum()==0 or m1.sum()==0:
            raise ValueError('EO gap undefined: missing label/group cell')
        ds.append(float(p[m0].mean()-p[m1].mean()))
        js.append((Xa[m0]*v[m0,None]).mean(axis=0)-(Xa[m1]*v[m1,None]).mean(axis=0))
    return np.asarray(ds,float),np.stack(js)


def build_gap_proxy(Xp: np.ndarray, yp: np.ndarray, Xc: np.ndarray, yc: np.ndarray, sc: np.ndarray,
                    reg: float) -> GapProxy:
    base=build_proxy(Xp,yp,Xc,yc,sc,reg)
    d0,J=_gap_value_jacobian(Xc,yc,sc,base.w)
    Hinv=np.linalg.inv(base.train_hessian)
    # delta_theta=-H^{-1}u, hence delta_gap=J*delta_theta = -J H^{-1}u
    A=-J@Hinv
    return GapProxy(reg=reg,w=base.w,grad_samples=base.train_grad_samples,
                    grad_mean=base.train_grad_mean,A_gap=A,d0=d0)


def label_targets(y: np.ndarray, k: int):
    """Preserve pool label prevalence as closely as integer k permits."""
    p=float(np.mean(y))
    n1=int(np.clip(round(k*p),1,k-1))
    return {0:k-n1,1:n1}


def representation_penalty(X: np.ndarray, y: np.ndarray, idx: np.ndarray, targets: dict[int,int]):
    X=np.asarray(X,float);y=np.asarray(y,int)
    val=0.0
    for yy in (0,1):
        mu=X[y==yy].mean(axis=0);j=idx[y[idx]==yy]
        if len(j)==0:return float('inf')
        val += float(np.mean((X[j].mean(axis=0)-mu)**2))
    return val


def estimate_scales(gap_proxies: list[GapProxy], X: np.ndarray, y: np.ndarray, k: int,
                    random_state: np.random.Generator, draws: int=32):
    targets=label_targets(y,k);fs=[];rs=[]
    ids0=np.flatnonzero(y==0);ids1=np.flatnonzero(y==1)
    for _ in range(draws):
        ix=np.sort(np.r_[random_state.choice(ids0,targets[0],replace=False),
                         random_state.choice(ids1,targets[1],replace=False)])
        fs.append(max(p.fairness(ix) for p in gap_proxies));rs.append(representation_penalty(X,y,ix,targets))
    return max(float(np.median(fs)),1e-5),max(float(np.median(rs)),1e-8)


def robust_gn_greedy(gap_proxies: list[GapProxy], X: np.ndarray, y: np.ndarray, k: int,
                     representation_weight: float=0.5, fairness_scale: float|None=None,
                     representation_scale: float|None=None, scale_rng: np.random.Generator|None=None):
    """Greedy robust Gauss-Newton fairness selection with label-prevalence and class-mean guards.

    Final objective (up to greedy approximation):
      max_proxy sum_y gap_hat_y(S)^2 / fair_scale
      + eta * sum_y ||mean_X(S_y)-mean_X(D_y)||^2 / rep_scale
    subject to |S_y| matching the pool label prevalence at budget k.
    """
    X=np.asarray(X,float);y=np.asarray(y,int);n,d=X.shape
    targets=label_targets(y,k)
    if min(np.sum(y==0),np.sum(y==1)) < min(targets.values()):
        raise ValueError('infeasible label quota')
    if fairness_scale is None or representation_scale is None:
        if scale_rng is None:scale_rng=np.random.default_rng(0)
        fairness_scale,representation_scale=estimate_scales(gap_proxies,X,y,k,scale_rng)

    # Precompute candidate effect on each proxy gap. For a provisional set of size t+1,
    # u=(sum_g+g_i)/(t+1)-gbar. Each proxy gap therefore admits O(N) candidate evaluation.
    selected=[];available=np.ones(n,bool);counts={0:0,1:0}
    sums_g=[np.zeros(p.grad_samples.shape[1],float) for p in gap_proxies]
    sums_x={0:np.zeros(d,float),1:np.zeros(d,float)}
    mus={yy:X[y==yy].mean(axis=0) for yy in (0,1)}
    for t in range(k):
        alpha=1.0/(t+1.0)
        fair_by_proxy=[]
        for p,sg in zip(gap_proxies,sums_g):
            base_u=alpha*sg-p.grad_mean
            base_gap=p.d0+p.A_gap@base_u
            cand_gap=base_gap[None,:] + alpha*(p.grad_samples@p.A_gap.T)
            fair_by_proxy.append(np.einsum('ij,ij->i',cand_gap,cand_gap))
        fair=np.maximum.reduce(fair_by_proxy)/fairness_scale

        # Quadratic class-conditional mean matching evaluated against final target counts.
        rep=np.empty(n,float)
        const_other={}
        for yy in (0,1):
            target_sum=targets[yy]*mus[yy]
            const_other[yy]=float(np.mean(((sums_x[yy]-target_sum)/max(targets[yy],1))**2))
        for yy in (0,1):
            mask=(y==yy)
            target_sum=targets[yy]*mus[yy]
            rr=(sums_x[yy][None,:]+X[mask]-target_sum[None,:])/max(targets[yy],1)
            vals=np.mean(rr*rr,axis=1)+const_other[1-yy]
            rep[mask]=vals/representation_scale
        score=fair+representation_weight*rep

        # Feasibility under exact final label quotas.
        for yy in (0,1):
            if counts[yy]>=targets[yy]:available[y==yy]=False
        score[~available]=np.inf
        j=int(np.argmin(score))
        if not np.isfinite(score[j]):raise RuntimeError('quota-constrained greedy became infeasible')
        selected.append(j);available[j]=False;counts[int(y[j])]+=1;sums_x[int(y[j])]+=X[j]
        for q,p in enumerate(gap_proxies):sums_g[q]+=p.grad_samples[j]
    ix=np.asarray(selected,int)
    assert counts[0]==targets[0] and counts[1]==targets[1]
    return ix,{'target_counts':targets,'fairness_scale':float(fairness_scale),'representation_scale':float(representation_scale)}


def robust_gn_predict(gap_proxies: list[GapProxy], idx: np.ndarray):
    vals=[p.fairness(idx) for p in gap_proxies]
    return {'proxy_fairness':vals,'worst_pred_smooth_eo_sq':float(max(vals)),'mean_pred_smooth_eo_sq':float(np.mean(vals))}


def build_crossfit_gap_proxies(Xp: np.ndarray, yp: np.ndarray, Xc: np.ndarray, yc: np.ndarray, sc: np.ndarray,
                               regs=(5e-4,5e-3), folds: int=2, random_state: int=2026):
    """Build proxy family across regularization regimes and stratified calibration folds."""
    from sklearn.model_selection import StratifiedKFold
    strat=2*np.asarray(yc,int)+np.asarray(sc,int)
    if folds<=1:
        ids=[np.arange(len(yc))]
    else:
        skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=random_state)
        ids=[te for _,te in skf.split(np.zeros(len(yc)),strat)]
    out=[]
    for reg in regs:
        for j in ids:
            out.append(build_gap_proxy(Xp,yp,Xc[j],yc[j],sc[j],reg))
    return out

# ---- v2.3: distributionally robust Gauss-Newton across calibration environments ----

def _stratified_bootstrap_indices(y: np.ndarray, s: np.ndarray, r: np.random.Generator) -> np.ndarray:
    y=np.asarray(y,int); s=np.asarray(s,int); parts=[]
    for yy in (0,1):
        for gg in (0,1):
            ids=np.flatnonzero((y==yy)&(s==gg))
            if len(ids)==0:
                raise ValueError('bootstrap environment undefined: missing label/group cell')
            parts.append(r.choice(ids,size=len(ids),replace=True))
    return np.concatenate(parts)


def build_bootstrap_gap_proxies(Xp: np.ndarray, yp: np.ndarray,
                                Xc: np.ndarray, yc: np.ndarray, sc: np.ndarray,
                                regs=(5e-4,5e-3), environments: int=6,
                                random_state: int=20260926) -> list[GapProxy]:
    """Proxy family = regularization regimes x calibration environments.

    Environment 0 is the full calibration set. Remaining environments are
    stratified nonparametric bootstraps that preserve every (label, group) cell.
    The pool-trained proxy weights stay fixed; only the fairness moment/Jacobian
    environment changes. This isolates calibration-distribution uncertainty.
    """
    r=np.random.default_rng(random_state)
    env_ids=[np.arange(len(yc),dtype=int)]
    for _ in range(max(0,environments-1)):
        env_ids.append(_stratified_bootstrap_indices(yc,sc,r))
    out=[]
    for reg in regs:
        # Fit once per regularization regime; build_gap_proxy currently refits, but
        # this keeps the public API simple and deterministic for the CPU prototype.
        for ids in env_ids:
            out.append(build_gap_proxy(Xp,yp,Xc[ids],yc[ids],sc[ids],reg))
    return out


def _environment_risk_matrix(gap_proxies: list[GapProxy], sums_g: list[np.ndarray], t: int) -> np.ndarray:
    """Return predicted fairness for every environment x candidate."""
    alpha=1.0/(t+1.0)
    vals=[]
    for p,sg in zip(gap_proxies,sums_g):
        base_u=alpha*sg-p.grad_mean
        base_gap=p.d0+p.A_gap@base_u
        cand_gap=base_gap[None,:] + alpha*(p.grad_samples@p.A_gap.T)
        vals.append(np.einsum('ij,ij->i',cand_gap,cand_gap))
    return np.stack(vals,axis=0)


def dr_gn_predict(gap_proxies: list[GapProxy], idx: np.ndarray, rho: float=1.0):
    vals=np.asarray([p.fairness(idx) for p in gap_proxies],float)
    mu=float(vals.mean()); sd=float(vals.std(ddof=0))
    return {'env_fairness':vals.tolist(), 'mean_pred_smooth_eo_sq':mu,
            'std_pred_smooth_eo_sq':sd, 'dr_pred_smooth_eo_sq':mu+float(rho)*sd,
            'worst_pred_smooth_eo_sq':float(vals.max())}


def dr_gn_greedy(gap_proxies: list[GapProxy], X: np.ndarray, y: np.ndarray, k: int,
                 rho: float=1.0, representation_weight: float=0.5,
                 fairness_scale: float|None=None, representation_scale: float|None=None,
                 scale_rng: np.random.Generator|None=None):
    """Distributionally robust set selection.

    Minimize mean(environment fairness) + rho * std(environment fairness), plus
    class-conditional feature-mean fidelity, under an exact label-prevalence quota.
    Environment uncertainty spans both proxy regularization and bootstrap
    calibration distributions.
    """
    X=np.asarray(X,float); y=np.asarray(y,int); n,d=X.shape
    targets=label_targets(y,k)
    if fairness_scale is None or representation_scale is None:
        if scale_rng is None: scale_rng=np.random.default_rng(0)
        # Scale the DR risk itself on prevalence-matched random contexts.
        fs=[]; rs=[]; ids0=np.flatnonzero(y==0); ids1=np.flatnonzero(y==1)
        for _ in range(32):
            ix=np.sort(np.r_[scale_rng.choice(ids0,targets[0],replace=False),
                             scale_rng.choice(ids1,targets[1],replace=False)])
            vv=np.asarray([p.fairness(ix) for p in gap_proxies],float)
            fs.append(float(vv.mean()+rho*vv.std(ddof=0)))
            rs.append(representation_penalty(X,y,ix,targets))
        fairness_scale=max(float(np.median(fs)),1e-5)
        representation_scale=max(float(np.median(rs)),1e-8)

    selected=[]; available=np.ones(n,bool); counts={0:0,1:0}
    sums_g=[np.zeros(p.grad_samples.shape[1],float) for p in gap_proxies]
    sums_x={0:np.zeros(d,float),1:np.zeros(d,float)}
    mus={yy:X[y==yy].mean(axis=0) for yy in (0,1)}
    for t in range(k):
        env=_environment_risk_matrix(gap_proxies,sums_g,t)
        fair=(env.mean(axis=0)+float(rho)*env.std(axis=0,ddof=0))/fairness_scale
        rep=np.empty(n,float); const_other={}
        for yy in (0,1):
            target_sum=targets[yy]*mus[yy]
            const_other[yy]=float(np.mean(((sums_x[yy]-target_sum)/max(targets[yy],1))**2))
        for yy in (0,1):
            mask=(y==yy); target_sum=targets[yy]*mus[yy]
            rr=(sums_x[yy][None,:]+X[mask]-target_sum[None,:])/max(targets[yy],1)
            rep[mask]=(np.mean(rr*rr,axis=1)+const_other[1-yy])/representation_scale
        score=fair+float(representation_weight)*rep
        for yy in (0,1):
            if counts[yy]>=targets[yy]: available[y==yy]=False
        score[~available]=np.inf
        j=int(np.argmin(score))
        if not np.isfinite(score[j]): raise RuntimeError('DR-GN quota-constrained greedy infeasible')
        selected.append(j); available[j]=False; counts[int(y[j])]+=1; sums_x[int(y[j])]+=X[j]
        for q,p in enumerate(gap_proxies): sums_g[q]+=p.grad_samples[j]
    ix=np.asarray(selected,int)
    return ix,{'rho':float(rho),'environments':len(gap_proxies),'target_counts':targets,
               'fairness_scale':float(fairness_scale),'representation_scale':float(representation_scale)}

def aggregate_environment_risk(values: np.ndarray, mode: str='mean', rho: float=2.0) -> np.ndarray:
    """Aggregate environment losses along axis 0; supports vectors or env x candidate matrices."""
    v=np.asarray(values,float)
    if mode=='mean': return v.mean(axis=0)
    if mode=='ucb': return v.mean(axis=0)+float(rho)*v.std(axis=0,ddof=0)
    if mode=='worst': return v.max(axis=0)
    if mode.startswith('cvar'):
        # cvar25 = average worst 25% environments; cvar50 = worst 50%.
        frac=float(mode[4:])/100.0
        q=max(1,int(np.ceil(v.shape[0]*frac)))
        part=np.partition(v,v.shape[0]-q,axis=0)[v.shape[0]-q:]
        return part.mean(axis=0)
    raise ValueError('unknown environment risk mode: '+str(mode))


def envrisk_gn_predict(gap_proxies: list[GapProxy], idx: np.ndarray, mode: str='mean', rho: float=2.0):
    vals=np.asarray([p.fairness(idx) for p in gap_proxies],float)
    risk=float(aggregate_environment_risk(vals[:,None],mode=mode,rho=rho)[0])
    return {'env_fairness':vals.tolist(),'risk':risk,'mean':float(vals.mean()),'std':float(vals.std(ddof=0)),'worst':float(vals.max())}


def envrisk_gn_greedy(gap_proxies: list[GapProxy], X: np.ndarray, y: np.ndarray, k: int,
                      mode: str='mean', rho: float=2.0, representation_weight: float=0.5,
                      fairness_scale: float|None=None, representation_scale: float|None=None,
                      scale_rng: np.random.Generator|None=None):
    X=np.asarray(X,float); y=np.asarray(y,int); n,d=X.shape; targets=label_targets(y,k)
    if scale_rng is None: scale_rng=np.random.default_rng(0)
    if fairness_scale is None or representation_scale is None:
        fs=[];rs=[];ids0=np.flatnonzero(y==0);ids1=np.flatnonzero(y==1)
        for _ in range(32):
            ix=np.sort(np.r_[scale_rng.choice(ids0,targets[0],replace=False),scale_rng.choice(ids1,targets[1],replace=False)])
            vv=np.asarray([p.fairness(ix) for p in gap_proxies],float)
            fs.append(float(aggregate_environment_risk(vv[:,None],mode=mode,rho=rho)[0]));rs.append(representation_penalty(X,y,ix,targets))
        fairness_scale=max(float(np.median(fs)),1e-5);representation_scale=max(float(np.median(rs)),1e-8)
    selected=[];available=np.ones(n,bool);counts={0:0,1:0}
    sums_g=[np.zeros(p.grad_samples.shape[1],float) for p in gap_proxies];sums_x={0:np.zeros(d,float),1:np.zeros(d,float)}
    mus={yy:X[y==yy].mean(axis=0) for yy in (0,1)}
    for t in range(k):
        env=_environment_risk_matrix(gap_proxies,sums_g,t)
        fair=aggregate_environment_risk(env,mode=mode,rho=rho)/fairness_scale
        rep=np.empty(n,float);const_other={}
        for yy in (0,1):
            target_sum=targets[yy]*mus[yy];const_other[yy]=float(np.mean(((sums_x[yy]-target_sum)/max(targets[yy],1))**2))
        for yy in (0,1):
            mask=(y==yy);target_sum=targets[yy]*mus[yy]
            rr=(sums_x[yy][None,:]+X[mask]-target_sum[None,:])/max(targets[yy],1)
            rep[mask]=(np.mean(rr*rr,axis=1)+const_other[1-yy])/representation_scale
        score=fair+float(representation_weight)*rep
        for yy in (0,1):
            if counts[yy]>=targets[yy]:available[y==yy]=False
        score[~available]=np.inf;j=int(np.argmin(score))
        if not np.isfinite(score[j]):raise RuntimeError('env-risk GN infeasible')
        selected.append(j);available[j]=False;counts[int(y[j])]+=1;sums_x[int(y[j])]+=X[j]
        for q,p in enumerate(gap_proxies):sums_g[q]+=p.grad_samples[j]
    ix=np.asarray(selected,int)
    return ix,{'mode':mode,'rho':float(rho),'environments':len(gap_proxies),'target_counts':targets,'fairness_scale':float(fairness_scale),'representation_scale':float(representation_scale)}
