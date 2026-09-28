"""ContextBench v2.7 frozen method: Post-Selection Benefit-Safety Certification (PSC-BS).

Status
------
Frozen after development on the ACS-2022 AZ/GA/MA/NC/WA development suite and
before any v2.7 fresh confirmation data are inspected.

Design
------
P: context pool
C: proposal calibration (optional, selector-specific)
V: candidate/anchor selection
W: independent post-selection certification
T: untouched final test

A base selector commits to exactly one candidate S* using P/C/V only.  A safety
anchor A is also fixed using P/C/V only.  W is revealed only after both are
fixed.  Cheap heterogeneous certifiers compare S* with A on W.

S* is deployed iff:
  1) every certifier has delta AUC >= -0.01;
  2) every certifier has delta log-loss <= +0.05;
  3) mean delta smooth-EO^2 across certifiers is < 0.
Otherwise return A exactly.

The default base selector for fresh confirmation is the frozen 16-draw
validation fairness random search.  The wrapper itself is selector-agnostic;
v2.6 heterogeneous proposal selection and direct-surrogate selectors are
predeclared wrapper ablations.
"""
from __future__ import annotations
import warnings
import numpy as np
from v24_models import make_model
from v24_benchmark import metrics
from common import seed

V27_CERTIFIER_MODELS = ('logistic', 'histgb', 'random_forest')
V27_AUC_TOLERANCE = 0.01
V27_LOGLOSS_TOLERANCE = 0.05
V27_FAIRNESS_MARGIN = 0.0
V27_VERSION = 'v2.7-psc-bs-frozen-20260927'


def _predict(model_name, X_train, y_train, X_eval, random_state):
    m = make_model(model_name, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(X_train, y_train)
    j = list(m.classes_).index(1)
    return np.asarray(m.predict_proba(X_eval)[:, j], dtype=float)


def certify_selected_context(candidate_ix, anchor_ix, Xp, yp, Xw, yw, sw,
                             c, dataset, outer, k,
                             certifier_models=V27_CERTIFIER_MODELS,
                             auc_tolerance=V27_AUC_TOLERANCE,
                             logloss_tolerance=V27_LOGLOSS_TOLERANCE,
                             fairness_margin=V27_FAIRNESS_MARGIN):
    """Certify one already-selected context against one predeclared anchor.

    Important: candidate_ix and anchor_ix must be fixed without using W.
    W is used only once here, after selection has committed.
    """
    candidate_ix = np.asarray(candidate_ix, dtype=int)
    anchor_ix = np.asarray(anchor_ix, dtype=int)
    yp = np.asarray(yp, dtype=int)
    yw = np.asarray(yw, dtype=int)
    sw = np.asarray(sw, dtype=int)

    by_model = {}
    for m in certifier_models:
        pa = _predict(m, Xp[anchor_ix], yp[anchor_ix], Xw,
                      seed(c, 'v27psc', dataset, outer, k, 'cert_anchor', m))
        pc = _predict(m, Xp[candidate_ix], yp[candidate_ix], Xw,
                      seed(c, 'v27psc', dataset, outer, k, 'cert_cand', m))
        ma = metrics(pa, yw, sw)
        mc = metrics(pc, yw, sw)
        by_model[m] = {
            'anchor_auc': float(ma['auc']),
            'candidate_auc': float(mc['auc']),
            'delta_auc': float(mc['auc'] - ma['auc']),
            'anchor_logloss': float(ma['logloss']),
            'candidate_logloss': float(mc['logloss']),
            'delta_logloss': float(mc['logloss'] - ma['logloss']),
            'anchor_smooth_eo_sq': float(ma['smooth_eo_sq']),
            'candidate_smooth_eo_sq': float(mc['smooth_eo_sq']),
            'delta_smooth_eo_sq': float(mc['smooth_eo_sq'] - ma['smooth_eo_sq']),
        }

    min_delta_auc = min(v['delta_auc'] for v in by_model.values())
    max_delta_logloss = max(v['delta_logloss'] for v in by_model.values())
    mean_delta_fairness = float(np.mean([v['delta_smooth_eo_sq'] for v in by_model.values()]))

    utility_pass = (min_delta_auc >= -float(auc_tolerance)) and \
                   (max_delta_logloss <= float(logloss_tolerance))
    benefit_pass = mean_delta_fairness < -float(fairness_margin)
    certified = bool(utility_pass and benefit_pass)

    return {
        'version': V27_VERSION,
        'certified': certified,
        'fallback_to_anchor': bool(not certified),
        'utility_pass': bool(utility_pass),
        'benefit_pass': bool(benefit_pass),
        'min_delta_auc': float(min_delta_auc),
        'max_delta_logloss': float(max_delta_logloss),
        'mean_delta_smooth_eo_sq': mean_delta_fairness,
        'auc_tolerance': float(auc_tolerance),
        'logloss_tolerance': float(logloss_tolerance),
        'fairness_margin': float(fairness_margin),
        'certifier_models': list(certifier_models),
        'by_model': by_model,
    }


def deploy_after_certification(candidate_ix, anchor_ix, *args, **kwargs):
    meta = certify_selected_context(candidate_ix, anchor_ix, *args, **kwargs)
    final_ix = np.asarray(candidate_ix if meta['certified'] else anchor_ix, dtype=int)
    return np.sort(final_ix), meta
