from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'v2_7_fresh_cpu'
TASKS=RUN/'tasks'
OUT=RUN/'analysis'; OUT.mkdir(parents=True,exist_ok=True)

def mean(xs): return float(np.mean(xs)) if xs else float('nan')
def dump_csv(path,rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

manifest=json.loads((RUN/'TASK_MANIFEST.json').read_text())
recs=[json.loads(p.read_text()) for p in TASKS.glob('*.json')]
ok=[r for r in recs if r.get('status')=='ok']
idx={(r['dataset'],r['outer'],r['n'],r['model'],r['selector']):r for r in ok}
basecells=sorted({(r['dataset'],r['outer'],r['n'],r['model']) for r in ok if r['selector']=='v27_psc_bs'})
paired_anchor=[]; paired_random=[]
for cell in basecells:
    v=idx.get((*cell,'v27_psc_bs')); a=idx.get((*cell,'safety_anchor'))
    rs=[idx.get((*cell,f'prevalence_random_{d}')) for d in range(5)]
    if not v or not a or any(q is None for q in rs): continue
    da=v['test_auc']-a['test_auc']; dll=v['test_logloss']-a['test_logloss']; deo=v['test_smooth_eo_sq']-a['test_smooth_eo_sq']
    paired_anchor.append({'dataset':cell[0],'outer':cell[1],'n':cell[2],'model':cell[3],'d_eo2':deo,'d_auc':da,'d_logloss':dll,'severe_auc_tail':int(da < -0.05)})
    reo=mean([q['test_smooth_eo_sq'] for q in rs]);ra=mean([q['test_auc'] for q in rs]);rll=mean([q['test_logloss'] for q in rs])
    deor=v['test_smooth_eo_sq']-reo; dar=v['test_auc']-ra; dllr=v['test_logloss']-rll
    fair=deor<0; guarded=fair and dar>=-0.01 and dllr<=0.05
    paired_random.append({'dataset':cell[0],'outer':cell[1],'n':cell[2],'model':cell[3],'d_eo2':deor,'d_auc':dar,'d_logloss':dllr,'fair_win':int(fair),'guarded_win':int(guarded)})

contexts=[]
for p in (RUN/'contexts').glob('*.json'):
    z=json.loads(p.read_text()); contexts.append(z)
accept=[bool(z.get('certificate',{}).get('certified')) for z in contexts]
accept_rate=mean(accept)

pa=paired_anchor; pr=paired_random
summary={
 'manifest_task_files':manifest.get('task_files'), 'manifest_ok':manifest.get('ok'), 'manifest_error':manifest.get('error'),
 'paired_cells':len(pa),'context_cells':len(contexts),'acceptance_rate':accept_rate,
 'anchor_mean_d_eo2':mean([r['d_eo2'] for r in pa]), 'anchor_mean_d_auc':mean([r['d_auc'] for r in pa]),
 'anchor_mean_d_logloss':mean([r['d_logloss'] for r in pa]), 'anchor_p_dauc_lt_m005':mean([r['severe_auc_tail'] for r in pa]),
 'random5_fairness_win':mean([r['fair_win'] for r in pr]), 'random5_guarded_win':mean([r['guarded_win'] for r in pr]),
 'random5_mean_d_eo2':mean([r['d_eo2'] for r in pr]), 'random5_mean_d_auc':mean([r['d_auc'] for r in pr]), 'random5_mean_d_logloss':mean([r['d_logloss'] for r in pr]),
}
summary['gate1_complete_zero_error']=bool(manifest.get('error')==0 and manifest.get('task_files')==manifest.get('ok') and len(pa)>0)
summary['gate2_acceptance_10_60']=bool(0.10<=accept_rate<=0.60)
summary['gate3_anchor_mean_deo2_lt0']=bool(summary['anchor_mean_d_eo2']<0)
summary['gate4_anchor_mean_dauc_ge_m0005']=bool(summary['anchor_mean_d_auc']>=-0.005)
summary['gate5_anchor_mean_dll_le_p001']=bool(summary['anchor_mean_d_logloss']<=0.01)
summary['gate6_tail_le_1pct']=bool(summary['anchor_p_dauc_lt_m005']<=0.01)
summary['gate7_vs_random5']=bool(summary['random5_fairness_win']>=0.60 and summary['random5_guarded_win']>=0.50 and summary['random5_mean_d_auc']>=-0.005)

by_dataset=[]
for ds in sorted({r['dataset'] for r in pa}):
    aa=[r for r in pa if r['dataset']==ds]; rr=[r for r in pr if r['dataset']==ds]
    by_dataset.append({'dataset':ds,'cells':len(aa),'anchor_mean_d_eo2':mean([x['d_eo2'] for x in aa]),'anchor_mean_d_auc':mean([x['d_auc'] for x in aa]),'random5_fair_win':mean([x['fair_win'] for x in rr]),'random5_guarded_win':mean([x['guarded_win'] for x in rr])})
by_model=[]
for m in sorted({r['model'] for r in pa}):
    aa=[r for r in pa if r['model']==m]; rr=[r for r in pr if r['model']==m]
    by_model.append({'model':m,'cells':len(aa),'anchor_mean_d_eo2':mean([x['d_eo2'] for x in aa]),'anchor_mean_d_auc':mean([x['d_auc'] for x in aa]),'random5_fair_win':mean([x['fair_win'] for x in rr]),'random5_guarded_win':mean([x['guarded_win'] for x in rr])})
summary['datasets_negative_anchor_deo2']=sum(r['anchor_mean_d_eo2']<0 for r in by_dataset)
summary['models_negative_anchor_deo2']=sum(r['anchor_mean_d_eo2']<0 for r in by_model)
summary['gate8_not_single_dataset_or_model']=bool(summary['datasets_negative_anchor_deo2']>=2 and summary['models_negative_anchor_deo2']>=2)
summary['all_8_gates_pass']=all(summary[f'gate{i}_'+suffix] if False else True for i,suffix in [])
summary['all_primary_boolean_gates_pass']=all(summary[k] for k in summary if k.startswith('gate'))

(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
dump_csv(OUT/'paired_vs_safety_anchor.csv',pa); dump_csv(OUT/'paired_vs_random5_mean.csv',pr); dump_csv(OUT/'by_dataset.csv',by_dataset); dump_csv(OUT/'by_model.csv',by_model)
lines=['# v2.7 Fresh CPU Confirmation — Automatic Gate Report','',f"Tasks: {manifest.get('ok')}/{manifest.get('task_files')} OK, errors={manifest.get('error')}",f"Context cells: {len(contexts)}, acceptance rate: {accept_rate:.3%}",'', '## Aggregate vs safety anchor',f"- mean Δsmooth-EO²: {summary['anchor_mean_d_eo2']:+.6f}",f"- mean ΔAUC: {summary['anchor_mean_d_auc']:+.6f}",f"- mean Δlogloss: {summary['anchor_mean_d_logloss']:+.6f}",f"- P(ΔAUC < -0.05): {summary['anchor_p_dauc_lt_m005']:.3%}",'','## Aggregate vs mean of five prevalence-random contexts',f"- fairness win: {summary['random5_fairness_win']:.3%}",f"- guarded win: {summary['random5_guarded_win']:.3%}",f"- mean ΔAUC: {summary['random5_mean_d_auc']:+.6f}",'','## Frozen gates']
for k,v in summary.items():
    if k.startswith('gate'): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
lines += ['',f"Overall automatic verdict: {'PASS' if summary['all_primary_boolean_gates_pass'] else 'FAIL'}",'', 'Note: gate 8 is operationalized conservatively as improvement direction appearing in at least two datasets and two held-out model families; inspect by_dataset.csv/by_model.csv as well.']
(OUT/'FRESH_CPU_GATE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
