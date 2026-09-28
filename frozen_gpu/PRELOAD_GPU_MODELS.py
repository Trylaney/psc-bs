from gpu_models import FrozenGPUModel, installed_versions
import numpy as np
print('versions',installed_versions())
X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[.2,.8],[.8,.2]],dtype=np.float32)
y=np.array([0,0,1,1,0,1],dtype=np.int64)
T=np.array([[.1,.2],[.9,.7]],dtype=np.float32)
for name in ('tabpfn35','tabiclv2'):
 print('PRELOAD',name,flush=True)
 m=FrozenGPUModel(name)
 p,t=m.fit_predict_proba(X,y,T)
 print(name,'OK',p.tolist(),t,flush=True)
 m.close()
print('PRELOAD COMPLETE')
