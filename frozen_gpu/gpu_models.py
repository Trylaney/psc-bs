"""Frozen GPU model wrappers for ContextBench v2.6.

Models are intentionally fixed before the GPU test results are observed.
"""
from __future__ import annotations
import gc, importlib.metadata, time
import numpy as np

TABPFN_PACKAGE_VERSION = "9.0.0"
TABPFN_MODEL_VERSION = "v3.5"
TABICL_PACKAGE_VERSION = "2.2.0"
TABICL_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
N_ESTIMATORS = 8
RANDOM_STATE = 20260926


def installed_versions():
    out={}
    for p in ("torch","tabpfn","tabicl","numpy","scikit-learn","pandas"):
        try: out[p]=importlib.metadata.version(p)
        except Exception: out[p]=None
    return out


def _positive_probability(clf, X):
    p=np.asarray(clf.predict_proba(X),dtype=float)
    classes=np.asarray(clf.classes_)
    pos=np.flatnonzero(classes==1)
    if len(pos)!=1:
        raise RuntimeError(f"positive class 1 missing; classes={classes.tolist()}")
    return np.asarray(p[:,int(pos[0])],dtype=float)


class FrozenGPUModel:
    def __init__(self,name,device="cuda"):
        self.name=name; self.device=device; self.clf=None
        self._build()

    def _build(self):
        if self.name=="tabpfn35":
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            self.clf=TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5,
                n_estimators=N_ESTIMATORS,
                device=self.device,
                inference_precision="auto",
                fit_mode="fit_with_cache",
                memory_saving_mode="auto",
                random_state=RANDOM_STATE,
                show_progress_bar=False,
            )
        elif self.name=="tabiclv2":
            from tabicl import TabICLClassifier
            self.clf=TabICLClassifier(
                n_estimators=N_ESTIMATORS,
                batch_size=8,
                kv_cache=True,
                checkpoint_version=TABICL_CHECKPOINT,
                allow_auto_download=True,
                device=self.device,
                use_amp="auto",
                use_fa3="auto",
                offload_mode="auto",
                random_state=RANDOM_STATE,
                verbose=False,
            )
        else:
            raise ValueError(self.name)

    def fit_predict_proba(self,Xtrain,ytrain,Xtest):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0=time.perf_counter()
        self.clf.fit(Xtrain,ytrain)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        tfit=time.perf_counter()-t0
        t1=time.perf_counter()
        p=_positive_probability(self.clf,Xtest)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        tpredict=time.perf_counter()-t1
        peak_alloc=peak_reserved=None
        if torch.cuda.is_available():
            peak_alloc=int(torch.cuda.max_memory_allocated())
            peak_reserved=int(torch.cuda.max_memory_reserved())
        return p,{"fit_seconds":tfit,"predict_seconds":tpredict,"total_seconds":tfit+tpredict,
                  "peak_memory_allocated_bytes":peak_alloc,"peak_memory_reserved_bytes":peak_reserved}

    def close(self):
        import torch
        self.clf=None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
