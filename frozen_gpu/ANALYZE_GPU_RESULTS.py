from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--input',default='runs/v2_7_gpu_fresh'); a=ap.parse_args(); root=Path(a.input)
rows=[]
for p in sorted((root/'tasks').glob('*.json')):
    try: rows.append(json.loads(p.read_text(encoding='utf-8')))
    except Exception: pass
if not rows: raise RuntimeError('no task results')
df=pd.DataFrame(rows)
keys=['dataset','outer','n','model']
metrics=['test_smooth_eo_sq','test_auc','test_logloss']

def one(sel):
    q=df[(df.selector==sel)&(df.status=='ok')][keys+metrics].copy()
    return q.rename(columns={m:f'{sel}__{m}' for m in metrics})
base=one('v27_psc_bs')
for sel in ['safety_anchor','valfair_random_search16']:
    base=base.merge(one(sel),on=keys,how='inner')
randoms=[]
for i in range(5):
    q=one(f'prevalence_random_{i}')
    q=q.rename(columns={f'prevalence_random_{i}__{m}':m for m in metrics})
    q['draw']=i; randoms.append(q)
r=pd.concat(randoms,ignore_index=True)
rmean=r.groupby(keys,as_index=False)[metrics].mean().rename(columns={m:f'random5__{m}' for m in metrics})
base=base.merge(rmean,on=keys,how='inner')
for ref in ['safety_anchor','valfair_random_search16','random5']:
    base[f'deo_vs_{ref}']=base['v27_psc_bs__test_smooth_eo_sq']-base[f'{ref}__test_smooth_eo_sq']
    base[f'dauc_vs_{ref}']=base['v27_psc_bs__test_auc']-base[f'{ref}__test_auc']
    base[f'dll_vs_{ref}']=base['v27_psc_bs__test_logloss']-base[f'{ref}__test_logloss']
base['fair_win_vs_random5']=base['deo_vs_random5']<0
base['guarded_win_vs_random5']=base['fair_win_vs_random5']&(base['dauc_vs_random5']>=-0.01)&(base['dll_vs_random5']<=0.05)
# Candidate tail is measured relative to the same safety anchor.
base['cand_dauc_vs_anchor']=base['valfair_random_search16__test_auc']-base['safety_anchor__test_auc']
summary={
 'task_files':int(len(df)),'ok':int((df.status=='ok').sum()),'error':int((df.status=='error').sum()),
 'paired_cells':int(len(base)),
 'anchor_mean_d_eo2':float(base.deo_vs_safety_anchor.mean()),
 'anchor_mean_d_auc':float(base.dauc_vs_safety_anchor.mean()),
 'anchor_mean_d_logloss':float(base.dll_vs_safety_anchor.mean()),
 'anchor_p_dauc_lt_m005':float((base.dauc_vs_safety_anchor<-0.05).mean()),
 'candidate_p_dauc_lt_m005':float((base.cand_dauc_vs_anchor<-0.05).mean()),
 'random5_fairness_win':float(base.fair_win_vs_random5.mean()),
 'random5_guarded_win':float(base.guarded_win_vs_random5.mean()),
 'random5_mean_d_eo2':float(base.deo_vs_random5.mean()),
 'random5_mean_d_auc':float(base.dauc_vs_random5.mean()),
 'random5_mean_d_logloss':float(base.dll_vs_random5.mean()),
}
by_model=[]
for m,g in base.groupby('model'):
    by_model.append({'model':m,'cells':len(g),'mean_d_eo2_vs_anchor':g.deo_vs_safety_anchor.mean(),'mean_d_auc_vs_anchor':g.dauc_vs_safety_anchor.mean(),'tail_rate':(g.dauc_vs_safety_anchor<-0.05).mean(),'fair_win_vs_random5':g.fair_win_vs_random5.mean(),'guarded_win_vs_random5':g.guarded_win_vs_random5.mean()})
by_dataset=[]
for m,g in base.groupby('dataset'):
    by_dataset.append({'dataset':m,'cells':len(g),'mean_d_eo2_vs_anchor':g.deo_vs_safety_anchor.mean(),'mean_d_auc_vs_anchor':g.dauc_vs_safety_anchor.mean(),'tail_rate':(g.dauc_vs_safety_anchor<-0.05).mean(),'fair_win_vs_random5':g.fair_win_vs_random5.mean(),'guarded_win_vs_random5':g.guarded_win_vs_random5.mean()})
bm=pd.DataFrame(by_model); bd=pd.DataFrame(by_dataset)
summary['datasets_negative_anchor_deo2']=int((bd.mean_d_eo2_vs_anchor<0).sum())
summary['models_negative_anchor_deo2']=int((bm.mean_d_eo2_vs_anchor<0).sum())
# GPU gates are frozen in GPU_CONFIRM_PROTOCOL.json before any fresh GPU result is observed.
gates={
 'gate1_complete_zero_error': len(df)==2880 and int((df.status=='ok').sum())==2880 and int((df.status=='error').sum())==0 and len(base)==360,
 'gate2_anchor_mean_deo2_lt0': summary['anchor_mean_d_eo2']<0,
 'gate3_anchor_mean_dauc_ge_m0005': summary['anchor_mean_d_auc']>=-0.005,
 'gate4_anchor_mean_dll_le_p001': summary['anchor_mean_d_logloss']<=0.01,
 'gate5_anchor_tail_le_1pct': summary['anchor_p_dauc_lt_m005']<=0.01,
 'gate6_vs_random5': summary['random5_fairness_win']>=0.60 and summary['random5_guarded_win']>=0.50 and summary['random5_mean_d_auc']>=-0.005,
 'gate7_both_tfms_transfer': len(bm)==2 and bool(((bm.mean_d_eo2_vs_anchor<0)&(bm.mean_d_auc_vs_anchor>=-0.01)).all()),
 'gate8_not_single_dataset': summary['datasets_negative_anchor_deo2']>=5,
 'gate9_tail_no_worse_than_candidate': summary['anchor_p_dauc_lt_m005']<=summary['candidate_p_dauc_lt_m005']+1e-12,
}
summary.update(gates); summary['all_9_gates_pass']=all(gates.values())
(root/'analysis').mkdir(exist_ok=True)
base.to_csv(root/'analysis'/'paired_cells.csv',index=False)
bm.to_csv(root/'analysis'/'by_model.csv',index=False)
bd.to_csv(root/'analysis'/'by_dataset.csv',index=False)
(root/'analysis'/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# v2.7 Fresh GPU Confirmation — Automatic Gate Report','',f"Tasks: {summary['ok']}/2880 OK; errors={summary['error']}",f"Paired cells: {summary['paired_cells']}",'','## Aggregate vs safety anchor',f"- mean Δsmooth-EO²: {summary['anchor_mean_d_eo2']:+.6f}",f"- mean ΔAUC: {summary['anchor_mean_d_auc']:+.6f}",f"- mean Δlogloss: {summary['anchor_mean_d_logloss']:+.6f}",f"- P(ΔAUC < -0.05): {summary['anchor_p_dauc_lt_m005']:.3%}",f"- candidate tail rate: {summary['candidate_p_dauc_lt_m005']:.3%}",'','## Against mean of five prevalence-random contexts',f"- fairness win: {summary['random5_fairness_win']:.3%}",f"- guarded win: {summary['random5_guarded_win']:.3%}",f"- mean ΔAUC: {summary['random5_mean_d_auc']:+.6f}",'','## Frozen GPU gates']
for k,v in gates.items(): lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
lines += ['',f"Overall automatic verdict: {'PASS' if summary['all_9_gates_pass'] else 'FAIL'}"]
(root/'analysis'/'GPU_FRESH_GATE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2)); print(root/'analysis'/'GPU_FRESH_GATE_REPORT.md')
