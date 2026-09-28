"""Public dataset adapters. No silent substitution on download failure."""
import csv,io,zipfile,urllib.request,time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from common import ROOT,write,read,digest,file_hash,seed

URLS={
 'adult':'https://archive.ics.uci.edu/static/public/2/adult.zip',
 'german':'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip',
 'bank':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'}
ARCHIVE_HASHES={'adult':'7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb',
 'german':'e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b',
 'bank':'e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4'}

def download(name,cache):
 p=Path(cache)/(name+'.zip');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  meta=read(p.with_suffix('.json'))
  if file_hash(p)!=meta['sha256'] or meta['sha256']!=ARCHIVE_HASHES[name]:raise RuntimeError('Cached data differs from release-pinned archive: '+str(p))
  return p
 errors=[]
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(URLS[name],headers={'User-Agent':'ContextBench/1.0'}),timeout=30) as r:b=r.read(20_000_000)
   with zipfile.ZipFile(io.BytesIO(b)) as z:
    if z.testzip():raise RuntimeError('ZIP integrity failure')
   if digest(b)!=ARCHIVE_HASHES[name]:raise RuntimeError('Archive differs from release; do not silently replace dataset')
   p.write_bytes(b);write(p.with_suffix('.json'),{'url':URLS[name],'sha256':digest(b),'bytes':len(b)})
   return p
  except Exception as e:errors.append(repr(e))
 raise RuntimeError('DATA_DOWNLOAD_FAILED '+name+'; cache may be populated manually; '+str(errors))

def load(name,cache=None):
 cache=cache or ROOT/'data_cache'
 if name in ('compas','default_credit','acs_income','acs_employment','acs_public_coverage','acs_mobility','acs_travel_time','law_school','diabetes_hospital'):
  from v25_external_data import load_external
  return load_external(name,cache)
 if name in ('synth_linear','synth_nonlinear','synth_rare_group'):
  seeds={'synth_linear':2401,'synth_nonlinear':2402,'synth_rare_group':2403};r=np.random.default_rng(seeds[name]);n=5000
  if name=='synth_rare_group': s=(r.random(n)<0.18).astype(int)
  else: s=(r.random(n)<0.5).astype(int)
  Z=r.normal(size=(n,8));
  # Features contain group-correlated proxies but never the sensitive bit itself.
  Z[:,0]+=0.8*s; Z[:,3]-=0.45*s
  if name=='synth_linear':
   logit=1.15*Z[:,0]-0.9*Z[:,1]+0.55*Z[:,2]-0.35*Z[:,4]+0.65*s
  elif name=='synth_nonlinear':
   logit=1.45*np.sin(Z[:,0])+0.85*Z[:,1]*Z[:,2]-0.55*(Z[:,4]**2)+0.6*Z[:,5]+0.75*s
  else:
   logit=1.0*Z[:,0]-0.75*Z[:,1]+0.5*Z[:,2]+1.05*s-0.6*s*Z[:,1]
  prob=1/(1+np.exp(-logit));y=(r.random(n)<prob).astype(int);X=Z.astype(object)
  meta={'source':'ContextBench controlled synthetic generator '+name,'target':'Bernoulli structural binary outcome','group':'generated binary group, excluded from X','categorical':[],
        'generator_seed':seeds[name],'development_only':True}
 elif name=='breast_cancer':
  d=load_breast_cancer();X=d.data.astype(object);y=(d.target==1).astype(int);s=np.full(len(y),-1)
  meta={'source':'sklearn.datasets.load_breast_cancer / UCI 17','target':'benign=1','group':None,'categorical':[]}
 elif name=='adult':
  p=download(name,cache)
  with zipfile.ZipFile(p) as z:
   lines=(z.read('adult.data').decode()+ '\n'+z.read('adult.test').decode()).splitlines()
  rows=[[v.strip() for v in row] for row in csv.reader(l for l in lines if l.strip() and not l.startswith('|'))]
  a=np.asarray(rows,dtype=object)
  if a.shape!=(48842,15):raise RuntimeError('Adult schema changed: '+str(a.shape))
  y=np.array([v.rstrip('.')=='>50K' for v in a[:,-1]],int);s=(a[:,9]=='Male').astype(int)
  keep=[i for i in range(14) if i!=9];X=a[:,keep]
  numeric={0,2,4,10,11,12};cat=[j for j,i in enumerate(keep) if i not in numeric]
  meta={'source':URLS[name],'raw_sha256':file_hash(p),'target':'>50K=1','group':'recorded sex: Male=1, Female=0; sex column removed','categorical':cat}
 elif name=='german':
  p=download(name,cache)
  with zipfile.ZipFile(p) as z:a=np.asarray([l.split() for l in z.read('german.data').decode().splitlines() if l.strip()],object)
  if a.shape!=(1000,21):raise RuntimeError('German schema changed')
  y=(a[:,20]=='1').astype(int);s=(a[:,12].astype(float)>=25).astype(int)
  keep=[i for i in range(20) if i!=12];X=a[:,keep];nums={1,4,7,10,12,15,17}
  meta={'source':URLS[name],'raw_sha256':file_hash(p),'target':'good_credit=1; not cost-sensitive credit evaluation','group':'age>=25=1; descriptive age subgroup, not legal fairness definition; age removed','categorical':[j for j,i in enumerate(keep) if i not in nums]}
 elif name=='bank':
  p=download(name,cache)
  with zipfile.ZipFile(p) as outer:
   with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as z:rows=list(csv.reader(io.StringIO(z.read('bank-full.csv').decode()),delimiter=';'))
  names=rows[0];a=np.asarray(rows[1:],object)
  if a.shape!=(45211,17):raise RuntimeError('Bank schema changed')
  y=(a[:,-1]=='yes').astype(int);s=(a[:,0].astype(float)>=25).astype(int)
  keep=[i for i in range(16) if names[i] not in ('age','duration')];X=a[:,keep]
  numeric={'age','balance','day','duration','campaign','pdays','previous'}
  meta={'source':URLS[name],'raw_sha256':file_hash(p),'target':'subscription=yes=1','group':'age>=25=1; descriptive subgroup; age removed; call duration excluded to avoid post-call information','categorical':[j for j,i in enumerate(keep) if names[i] not in numeric]}
 else:raise ValueError(name)
 # Replace missing numeric values with NaN; categorical '?' remains an explicit category.
 for j in range(X.shape[1]):
  if j not in meta['categorical']:X[:,j]=[float(v) if str(v) not in ('?','nan','') else np.nan for v in X[:,j]]
 # Exact duplicate full records stay in one stage and one role: collapse to one unit.
 keys={};idx=[]
 for i in range(len(y)):
  k=tuple(str(v) for v in X[i])+(str(y[i]),str(s[i]))
  if k not in keys:keys[k]=i;idx.append(i)
 idx=np.asarray(idx);meta['duplicates_removed']=len(y)-len(idx)
 X,y,s=X[idx],y[idx],s[idx];meta['rows']=len(y)
 meta['data_hash']=digest(('|'.join(','.join(map(str,r)) for r in X)+'|'+str(y.tolist())+'|'+str(s.tolist())).encode())
 return X,y,s,idx,meta

def strata(y,s):return y*2+s if np.all(s>=0) else y
def split(c,y,s,dataset,stage,outer,profile):
 ids=np.arange(len(y));g=strata(y,s)
 # The stage boundary does not change between smoke, pilot and discovery.
 ex,co=train_test_split(ids,train_size=c['explore_fraction'],stratify=g,random_state=seed(c,'stage',dataset))
 stage_ids=ex if stage=='explore' else co
 pool,rest=train_test_split(stage_ids,train_size=c['pool_fraction'],stratify=g[stage_ids],random_state=seed(c,'roles',profile,stage,dataset,outer))
 val,test=train_test_split(rest,test_size=.5,stratify=g[rest],random_state=seed(c,'validation',profile,stage,dataset,outer))
 def cap(a,n,part):
  if len(a)<=n:return np.sort(a)
  aa,_=train_test_split(a,train_size=n,stratify=g[a],random_state=seed(c,'cap',profile,stage,dataset,outer,part));return np.sort(aa)
 return [cap(a,c[k],k) for a,k in [(pool,'pool_cap'),(val,'validation_cap'),(test,'test_cap')]]
