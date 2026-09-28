"""Classical downstream model registry for the v2.4 CPU benchmark."""
from __future__ import annotations
import warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

MODEL_NAMES=('logistic','histgb','random_forest','extra_trees','xgboost','lightgbm','mlp')

def make_model(name,seed):
    if name=='logistic':
        return LogisticRegression(C=1,max_iter=1200,random_state=seed)
    if name=='histgb':
        return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,min_samples_leaf=4,l2_regularization=1,early_stopping=False,random_state=seed)
    if name=='random_forest':
        return RandomForestClassifier(n_estimators=120,min_samples_leaf=2,max_features='sqrt',n_jobs=1,random_state=seed)
    if name=='extra_trees':
        return ExtraTreesClassifier(n_estimators=120,min_samples_leaf=2,max_features='sqrt',n_jobs=1,random_state=seed)
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,
                             min_child_weight=2,reg_lambda=1,n_jobs=1,random_state=seed,eval_metric='logloss',verbosity=0)
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.05,min_child_samples=5,
                              reg_lambda=1,n_jobs=1,random_state=seed,verbosity=-1)
    if name=='mlp':
        return MLPClassifier(hidden_layer_sizes=(64,),activation='relu',solver='adam',alpha=1e-3,
                             max_iter=250,early_stopping=False,random_state=seed)
    raise ValueError(name)

def fit_predict(name,Xs,ys,Xv,Xt,seed):
    m=make_model(name,seed)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always');m.fit(Xs,ys)
        classes=list(m.classes_);j=classes.index(1)
        pv=np.asarray(m.predict_proba(Xv)[:,j],float);pt=np.asarray(m.predict_proba(Xt)[:,j],float)
    return pv,pt,[str(w.message) for w in ww]
