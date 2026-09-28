"""Validated external real-data adapters for ContextBench v2.5.

Adapters intentionally fail closed when a required package/file/download is
unavailable.  They never substitute a different dataset.  Every successful
load records the source URL/version and a SHA-256 of the local raw artifact.
"""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,os
from pathlib import Path
import numpy as np

COMPAS_URL='https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
DEFAULT_URL='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
LAW_SCHOOL_URL='https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv'

FOLK_TASKS={
 'acs_income':'ACSIncome',
 'acs_employment':'ACSEmployment',
 'acs_public_coverage':'ACSPublicCoverage',
 'acs_mobility':'ACSMobility',
 'acs_travel_time':'ACSTravelTime',
 'acs_health_insurance':'ACSHealthInsurance',
 'acs_income_poverty':'ACSIncomePovertyRatio',
}

def _sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def _download(url,path,max_bytes=80_000_000):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():return path
 if os.environ.get('CONTEXTBENCH_OFFLINE','0') == '1':
  raise RuntimeError(f'OFFLINE_DATA_MISSING cache={path}; place the exact raw dataset file there before running')
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ContextBench/2.5'})
  with urllib.request.urlopen(req,timeout=60) as r:
   b=r.read(max_bytes+1)
  if len(b)>max_bytes:raise RuntimeError('external artifact exceeds byte cap')
  tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b);tmp.replace(path)
  return path
 except Exception as e:
  raise RuntimeError(f'EXTERNAL_DATA_UNAVAILABLE url={url} cache={path}: {e!r}')

def _finalize(X,y,s,meta):
 X=np.asarray(X,dtype=object);y=np.asarray(y,int);s=np.asarray(s,int)
 if len(X)!=len(y) or len(y)!=len(s):raise ValueError('length mismatch')
 if not set(np.unique(y)).issubset({0,1}):raise ValueError('binary target required')
 if not set(np.unique(s)).issubset({0,1}):raise ValueError('binary group required')
 # numeric conversion only for predeclared numeric columns
 cats=set(meta.get('categorical',[]))
 for j in range(X.shape[1]):
  if j not in cats:
   X[:,j]=[float(v) if str(v) not in ('?','nan','','None') else np.nan for v in X[:,j]]
 # collapse exact duplicate (X,y,s) rows to prevent split leakage
 seen={};keep=[]
 for i in range(len(y)):
  key=tuple(str(v) for v in X[i])+(str(int(y[i])),str(int(s[i])))
  if key not in seen:seen[key]=i;keep.append(i)
 keep=np.asarray(keep,int);meta=dict(meta);meta['duplicates_removed']=int(len(y)-len(keep));meta['rows']=int(len(keep))
 return X[keep],y[keep],s[keep],keep,meta

def load_compas(cache):
 import pandas as pd
 p=_download(COMPAS_URL,Path(cache)/'compas-scores-two-years.csv',20_000_000)
 df=pd.read_csv(p)
 required=['days_b_screening_arrest','is_recid','c_charge_degree','score_text','race','sex','age','age_cat',
           'juv_fel_count','juv_misd_count','juv_other_count','priors_count','two_year_recid']
 miss=[c for c in required if c not in df.columns]
 if miss:raise RuntimeError('COMPAS schema changed: '+str(miss))
 # ProPublica analysis filters; then use the two largest race groups so the
 # protected comparison is explicit rather than heterogeneous "other".
 q=df[(df.days_b_screening_arrest>=-30)&(df.days_b_screening_arrest<=30)&
      (df.is_recid!=-1)&(df.c_charge_degree!='O')&(df.score_text!='N/A')&
      (df.race.isin(['African-American','Caucasian']))].copy()
 feats=['sex','age','age_cat','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree']
 X=q[feats].to_numpy(object);y=q['two_year_recid'].astype(int).to_numpy();s=(q['race']=='Caucasian').astype(int).to_numpy()
 cats=[feats.index(x) for x in ['sex','age_cat','c_charge_degree']]
 meta={'source':COMPAS_URL,'raw_sha256':_sha(p),'target':'two_year_recid=1',
       'group':'race restricted to African-American/Caucasian; Caucasian=1; race removed from X',
       'filter':'ProPublica two-year analysis quality filters; races restricted to African-American/Caucasian',
       'categorical':cats,'external_confirmation_candidate':True}
 return _finalize(X,y,s,meta)

def load_default_credit(cache):
 import pandas as pd
 p=_download(DEFAULT_URL,Path(cache)/'default_credit.zip',10_000_000)
 with zipfile.ZipFile(p) as z:
  names=z.namelist();xls=[n for n in names if n.lower().endswith(('.xls','.xlsx'))]
  if len(xls)!=1:raise RuntimeError('Default Credit archive schema changed: '+str(names))
  b=z.read(xls[0])
 df=pd.read_excel(io.BytesIO(b),header=1)
 # UCI release has an ID column, SEX, and target "default payment next month".
 target=[c for c in df.columns if str(c).strip().lower() in ('default payment next month','default.payment.next.month')]
 if len(target)!=1:raise RuntimeError('Default Credit target schema changed: '+str(list(df.columns)))
 t=target[0]
 if 'SEX' not in df.columns:raise RuntimeError('Default Credit SEX column missing')
 y=df[t].astype(int).to_numpy();s=(df['SEX'].astype(int)==1).astype(int).to_numpy()  # male=1, female=0
 drop=[t,'SEX']+([c for c in df.columns if str(c).strip().upper()=='ID'])
 feat=[c for c in df.columns if c not in drop]
 X=df[feat].to_numpy(object)
 # EDUCATION/MARRIAGE/PAY_* are coded categories/ordinals; leave PAY_* numeric
 cats=[i for i,c in enumerate(feat) if str(c) in ('EDUCATION','MARRIAGE')]
 meta={'source':DEFAULT_URL,'raw_sha256':_sha(p),'target':'default payment next month=1',
       'group':'SEX: male=1, female=0; SEX removed from X','categorical':cats,
       'external_confirmation_candidate':True}
 return _finalize(X,y,s,meta)

# Self-contained ACS task definitions matching the official Folktables tasks.
# This removes runtime dependence on the folktables package and makes the
# OfflineFull confirmation reproducible with only pandas/numpy + the bundled
# Census PUMS files.
ACS_TASK_SPECS={
 'acs_income':{
  'features':['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P'],
  'target':'PINCP','target_fn':lambda x:x>50000,
  'filter':lambda d:d[(d.AGEP>16)&(d.PINCP>100)&(d.WKHP>0)&(d.PWGTP>=1)],
  'official_name':'ACSIncome'},
 'acs_employment':{
  'features':['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P'],
  'target':'ESR','target_fn':lambda x:x==1,'filter':lambda d:d,'official_name':'ACSEmployment'},
 'acs_public_coverage':{
  'features':['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P'],
  'target':'PUBCOV','target_fn':lambda x:x==1,
  'filter':lambda d:d[(d.AGEP<65)&(d.PINCP<=30000)],'official_name':'ACSPublicCoverage'},
 'acs_mobility':{
  'features':['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','COW','ESR','WKHP','JWMNP','PINCP'],
  'target':'MIG','target_fn':lambda x:x==1,
  'filter':lambda d:d[(d.AGEP>18)&(d.AGEP<35)],'official_name':'ACSMobility'},
 'acs_travel_time':{
  'features':['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','JWTR','POWPUMA','POVPIP'],
  'target':'JWMNP','target_fn':lambda x:x>20,
  'filter':lambda d:d[(d.AGEP>16)&(d.PWGTP>=1)&(d.ESR==1)],'official_name':'ACSTravelTime'},
 'acs_health_insurance':{
  'features':['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RACAIAN','RACASN','RACBLK','RACNH','RACPI','RACSOR','RACWHT','PINCP','ESR','ST','FER'],
  'target':'HINS2','target_fn':lambda x:x==1,'filter':lambda d:d,'official_name':'ACSHealthInsurance'},
 'acs_income_poverty':{
  'features':['AGEP','SCHL','MAR','SEX','DIS','ESP','MIG','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','ESR','OCCP','WKHP'],
  'target':'POVPIP','target_fn':lambda x:x<250,'filter':lambda d:d,'official_name':'ACSIncomePovertyRatio'},
}
ACS_STATE_CODES={'AZ':'04','GA':'13','MA':'25','NC':'37','WA':'53','CA':'06','TX':'48','NY':'36','FL':'12','IL':'17'}

def _acs_compat_col(c,year):
 if int(year)>=2019:
  if c=='RELP':return 'RELSHIPP'
  if c=='JWTR':return 'JWTRNS'
 return c

def load_folktables(name,cache,states=('CA','TX','NY','FL','IL'),year=2018):
 import pandas as pd
 if name not in ACS_TASK_SPECS:raise ValueError(name)
 spec=ACS_TASK_SPECS[name]
 root=Path(cache)/'folktables'/str(year)/'1-Year'
 # Read only columns needed for the frozen task plus filter/group columns.
 logical=set(spec['features'])|{spec['target'],'RAC1P'}
 # Filter dependencies from official task definitions.
 if name=='acs_income': logical|={'PWGTP','PINCP','WKHP','AGEP'}
 if name=='acs_public_coverage': logical|={'AGEP','PINCP'}
 if name=='acs_mobility': logical|={'AGEP'}
 if name=='acs_travel_time': logical|={'AGEP','PWGTP','ESR'}
 physical={_acs_compat_col(c,year) for c in logical}
 dfs=[];raw=[]
 for st in states:
  code=ACS_STATE_CODES[st]
  fn=(f'psam_p{code}.csv' if int(year)>=2017 else f'ss{str(year)[-2:]}p{st.lower()}.csv')
  p=root/fn
  if not p.exists():
   raise RuntimeError(f'OFFLINE_FOLKTABLES_DATA_MISSING: expected {p}')
  d=pd.read_csv(p,usecols=lambda c:c in physical)
  # Compatibility aliases for official Folktables task definitions.
  if int(year)>=2019:
   if 'RELP' in logical and 'RELSHIPP' in d.columns:d['RELP']=d['RELSHIPP']
   if 'JWTR' in logical and 'JWTRNS' in d.columns:d['JWTR']=d['JWTRNS']
  dfs.append(d);raw.append({'file':str(p),'sha256':_sha(p)})
 df=pd.concat(dfs,ignore_index=True)
 d=spec['filter'](df.copy()).copy()
 feats=list(spec['features'])
 # Protected RAC1P must never be in X.
 feats=[c for c in feats if c!='RAC1P']
 g=d['RAC1P'].to_numpy()
 mask=np.isin(g,[1,2]);d=d.loc[mask].reset_index(drop=True);g=g[mask]
 y=np.asarray(spec['target_fn'](d[spec['target']]),dtype=int)
 s=(g==1).astype(int)
 X=d[feats].to_numpy(object)
 # Match Folktables postprocess: replace NaN/inf with finite values where numeric.
 # Keep coded categorical ACS columns as numeric, consistent with v2.5 development.
 X=np.nan_to_num(np.asarray(X,dtype=float),nan=-1.0,posinf=0.0,neginf=0.0).astype(object)
 meta={'source':'U.S. Census Bureau ACS PUMS via self-contained official Folktables task definitions',
   'folktables_task':spec['official_name'],'survey_year':int(year),'states':list(states),
   'raw_files':raw,'target':spec['official_name']+' official binary task label',
   'group':'RAC1P restricted to White alone(1) and Black alone(2); White=1; RAC1P excluded from X',
   'categorical':[],'external_confirmation_candidate':True,
   'adapter_fix':'self-contained Folktables-compatible task spec; 2019+ RELP<-RELSHIPP and JWTR<-JWTRNS aliases'}
 return _finalize(X,y,s,meta)


def load_law_school(cache):
 import pandas as pd
 p=_download(LAW_SCHOOL_URL,Path(cache)/'law_dataset.csv',8_000_000)
 df=pd.read_csv(p)
 required=['decile1b','decile3','lsat','ugpa','zfygpa','zgpa','fulltime','fam_inc','male','racetxt','tier','pass_bar']
 miss=[c for c in required if c not in df.columns]
 if miss:raise RuntimeError('Law School schema changed: '+str(miss))
 # The paper benchmark uses bar passage as the binary prediction target and
 # sex as the protected comparison. The protected column itself is excluded.
 q=df.copy();q['pass_bar']=pd.to_numeric(q['pass_bar'],errors='coerce');q=q[q.pass_bar.isin([0,1])].copy()
 y=q['pass_bar'].astype(int).to_numpy();s=q['male'].astype(int).to_numpy()
 feats=[c for c in required if c not in ('pass_bar','male')]
 X=q[feats].to_numpy(object)
 cats=[feats.index(c) for c in ('fam_inc','racetxt','tier')]
 meta={'source':LAW_SCHOOL_URL,'raw_sha256':_sha(p),'target':'pass_bar=1 (passed bar exam on first try)',
       'group':'male=1, female=0; male removed from X','categorical':cats,
       'construction':'Wightman LSAC longitudinal bar-passage data; rows with missing/non-binary pass_bar removed',
       'external_confirmation_candidate':True}
 return _finalize(X,y,s,meta)

def load_diabetes_hospital(cache):
 import pandas as pd
 # Prefer a directly cached UCI archive so fully offline confirmation is possible.
 p=Path(cache)/'diabetes_130.zip'
 if p.exists():
  with zipfile.ZipFile(p) as z:
   names=z.namelist();csvs=[n for n in names if n.lower().endswith('diabetic_data.csv')]
   if len(csvs)!=1:raise RuntimeError('Diabetes 130 archive schema changed: '+str(names[:20]))
   df=pd.read_csv(io.BytesIO(z.read(csvs[0])))
  required=['race','gender','readmitted']
  miss=[c for c in required if c not in df.columns]
  if miss:raise RuntimeError('Diabetes 130 schema changed: '+str(miss))
  rr=df['race'].astype(str).str.replace(' ','',regex=False).str.replace('-','',regex=False)
  mask=rr.isin(['Caucasian','AfricanAmerican']).to_numpy();df=df.loc[mask].reset_index(drop=True);rr=rr[mask].reset_index(drop=True)
  y=(df['readmitted'].astype(str)=='<30').astype(int).to_numpy();s=(rr=='Caucasian').astype(int).to_numpy()
  drop=[c for c in ('race','readmitted','encounter_id','patient_nbr') if c in df.columns]
  feat=[c for c in df.columns if c not in drop];X=df[feat].to_numpy(object)
  cats=[]
  for i,c in enumerate(feat):
   dt=df[c].dtype
   if dt==object or str(dt).startswith('string') or str(dt)=='category' or str(dt)=='bool':cats.append(i)
  meta={'source':'UCI Diabetes 130-US Hospitals dataset id=296; offline cached archive',
        'raw_sha256':_sha(p),'target':'readmitted within 30 days (<30)=1',
        'group':'race restricted to Caucasian/AfricanAmerican; Caucasian=1; race removed from X',
        'categorical':cats,'construction':'IDs and target/protected columns removed; explicit two-race comparison',
        'external_confirmation_candidate':True}
  return _finalize(X,y,s,meta)
 try:
  from fairlearn.datasets import fetch_diabetes_hospital
 except Exception as e:
  raise RuntimeError('OFFLINE_DIABETES_DATA_MISSING: place UCI dataset 296 ZIP at data_cache/diabetes_130.zip') from e
 data=fetch_diabetes_hospital(as_frame=True,cache=True,data_home=str(Path(cache)/'fairlearn'))
 df=data.data.copy();y=np.asarray(data.target).astype(int)
 if 'race' not in df.columns:raise RuntimeError('Diabetes hospital race column missing')
 rr=df['race'].astype(str).str.replace(' ','',regex=False).str.replace('-','',regex=False)
 mask=rr.isin(['Caucasian','AfricanAmerican']).to_numpy();df=df.loc[mask].reset_index(drop=True);y=y[mask];rr=rr[mask].reset_index(drop=True)
 s=(rr=='Caucasian').astype(int).to_numpy()
 drop=[c for c in ('race','readmitted','readmit_binary') if c in df.columns]
 feat=[c for c in df.columns if c not in drop];X=df[feat].to_numpy(object)
 cats=[]
 for i,c in enumerate(feat):
  dt=df[c].dtype
  if str(dt)=='category' or dt==object or str(dt).startswith('string') or str(dt)=='bool':cats.append(i)
 meta={'source':'fairlearn.datasets.fetch_diabetes_hospital / UCI Diabetes 130-Hospitals',
       'target':'Fairlearn readmit_30_days binary target','group':'race restricted to Caucasian/AfricanAmerican; Caucasian=1; race removed from X',
       'categorical':cats,'construction':'Fairlearn preprocessed release; alternate readmission columns removed; explicit two-race comparison',
       'external_confirmation_candidate':True}
 return _finalize(X,y,s,meta)

def load_external(name,cache):
 if name=='compas':return load_compas(cache)
 if name=='default_credit':return load_default_credit(cache)
 if name=='law_school':return load_law_school(cache)
 if name=='diabetes_hospital':return load_diabetes_hospital(cache)
 if name in FOLK_TASKS:return load_folktables(name,cache)
 raise ValueError(name)
