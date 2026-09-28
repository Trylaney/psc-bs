"""Shared preprocessing and contexts, independent of evaluated model."""
import time,warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from common import rng,seed,digest,canon

def transform(X,meta,pool,val,test):
 cat=meta['categorical'];num=[j for j in range(X.shape[1]) if j not in cat]
 blocks=[]
 if num:blocks.append(('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),num))
 if cat:blocks.append(('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat))
 pre=ColumnTransformer(blocks,sparse_threshold=0)
 a=pre.fit_transform(X[pool]);b=pre.transform(X[val]);d=pre.transform(X[test])
 return [np.ascontiguousarray(v,dtype=np.float32) for v in (a,b,d)]

def balanced(labels,n,r):
 cells=[np.flatnonzero(labels==v) for v in np.unique(labels)]
 order=r.permutation(len(cells));counts=np.zeros(len(cells),int)
 while counts.sum()<n:
  changed=False
  for k in order:
   if counts[k]<len(cells[k]) and counts.sum()<n:counts[k]+=1;changed=True
  if not changed:raise ValueError('Infeasible balanced allocation')
 return np.sort(np.concatenate([r.choice(a,int(k),replace=False) for a,k in zip(cells,counts)]))

def matched(labels,target,r):
 parts=[]
 for cell,count in zip(*np.unique(labels[target],return_counts=True)):
  available=np.flatnonzero(labels==cell)
  if len(available)<count:raise ValueError('INFEASIBLE_EXACT_MATCH')
  parts.append(r.choice(available,int(count),replace=False))
 return np.sort(np.concatenate(parts))

def contexts(c,X,y,s,identity):
 start=time.perf_counter();has_s=np.all(s>=0);warnings_out=[]
 target=s if has_s else y
 with warnings.catch_warnings(record=True) as ww:
  proxy=LogisticRegression(C=1,max_iter=1000,random_state=seed(c,*identity,'proxy')).fit(X,target)
  q=proxy.predict_proba(X)[:,1]
  warnings_out=[str(w.message) for w in ww]
 proxy_seconds=time.perf_counter()-start
 w=np.exp(-abs(q-.5)/c['proxy_temperature']);w/=w.sum()
 rows=[];skips=[]
 for draw in range(c['draws']):
  for n in c['context_sizes']:
   if n>len(y):
    skips.append({'draw':draw,'n':n,'reason':'context exceeds pool'});continue
   for selector in c['selectors']:
    if selector=='joint_balance' and not has_s:
     skips.append({'draw':draw,'n':n,'selector':selector,'reason':'no sensitive/group attribute'});continue
    r=rng(c,*identity,draw,n,selector);tic=time.perf_counter()
    if selector=='random':ix=np.sort(r.choice(len(y),n,replace=False))
    elif selector=='label_balance':ix=balanced(y,n,r)
    elif selector=='joint_balance':ix=balanced(2*y+s,n,r)
    elif selector=='uncertainty':ix=np.sort(r.choice(len(y),n,replace=False,p=w))
    elif selector=='diversity':
     first=int(r.integers(len(y)));chosen=[first];dist=np.sum((X-X[first])**2,axis=1);dist[first]=-1
     while len(chosen)<n:
      j=int(np.argmax(dist));chosen.append(j);dist=np.minimum(dist,np.sum((X-X[j])**2,axis=1));dist[chosen]=-1
     ix=np.sort(chosen)
    else:raise ValueError(selector)
    elapsed=time.perf_counter()-tic
    key=f'd{draw}_n{n}_{selector}'
    row={'context_key':key,'draw':draw,'n':n,'selector':selector,'parent':'','level':'','k':-1,
         'indices':ix.tolist(),'selection_seconds':elapsed,'proxy_seconds':proxy_seconds if selector=='uncertainty' else 0.,
         'selection_seconds_cold':elapsed+(proxy_seconds if selector=='uncertainty' else 0.),
         'positive_fraction':float(y[ix].mean()),'group_fraction':float(s[ix].mean()) if has_s else None,
         'proxy_uncertainty':float((.5-abs(q[ix]-.5)).mean()),
         'feature_mean_shift':float(np.linalg.norm(X[ix].mean(axis=0)-X.mean(axis=0))),
         'context_hash':digest(canon(ix.tolist()))}
    rows.append(row)
    if selector not in c['matched_targets']:continue
    for level in c['matching_levels']:
     if level=='joint' and not has_s:continue
     labels=y if level=='label' else 2*y+s
     for k in range(c['controls_per_target']):
      tic=time.perf_counter();ci=matched(labels,ix,rng(c,*identity,draw,n,selector,level,k,'control'));sec=time.perf_counter()-tic
      assert np.array_equal(np.bincount(labels[ci],minlength=4),np.bincount(labels[ix],minlength=4))
      rows.append({**row,'context_key':f'{key}_control_{level}_{k}','selector':'matched_control','parent':key,'level':level,'k':k,
                   'indices':ci.tolist(),'selection_seconds':sec,'proxy_seconds':0.,'selection_seconds_cold':sec,
                   'positive_fraction':float(y[ci].mean()),'group_fraction':float(s[ci].mean()) if has_s else None,
                   'proxy_uncertainty':float((.5-abs(q[ci]-.5)).mean()),
                   'feature_mean_shift':float(np.linalg.norm(X[ci].mean(axis=0)-X.mean(axis=0))),
                   'context_hash':digest(canon(ci.tolist()))})
 bykey={v['context_key']:v for v in rows}
 for v in rows:
  ref=bykey.get(f"d{v['draw']}_n{v['n']}_random")
  v['overlap_random']=len(set(v['indices'])&set(ref['indices']))/v['n'] if ref else None
  v['overlap_parent']=len(set(v['indices'])&set(bykey[v['parent']]['indices']))/v['n'] if v['parent'] else None
 return {'contexts':rows,'skips':skips,'proxy_target':'group' if has_s else 'label','proxy_seconds':proxy_seconds,'warnings':warnings_out}
