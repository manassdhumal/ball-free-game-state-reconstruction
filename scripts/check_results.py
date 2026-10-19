import json, os
r=json.load(open('results/runs/robustness_expansion/full_run_report.json','r'))
runs=r.get('runs',[])
valid=0;invalid=0;affected=0;details=[]
for run in runs:
    paths=run.get('paths',{})
    sj=paths.get('summary_json')
    cfgp=paths.get('config_json')
    tc=paths.get('tactical_csv')
    ok=True
    if not (sj and os.path.exists(sj)):
        ok=False
    else:
        try:
            j=json.load(open(sj,'r'))
            keys=set(j.keys())
            if not {'CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED'}.issubset(keys):
                ok=False
        except Exception as e:
            ok=False
    if ok:
        valid+=1
    else:
        invalid+=1
    anypath=' '.join([p for p in (sj or '', cfgp or '', tc or '')])
    if anypath.count('robustness_expansion')>1:
        affected+=1
        details.append((run.get('run_id'), anypath.count('robustness_expansion')))
# Specific check for exp_0_60s_seed_72_sev_mild
spec_id='exp_0_60s_seed_72_sev_mild'
status='missing'
for run in runs:
    if run.get('run_id')==spec_id:
        p=run.get('paths',{})
        sj=p.get('summary_json')
        tc=p.get('tactical_csv')
        cfgp=p.get('config_json')
        status='ok' if (sj and os.path.exists(sj) and tc and os.path.exists(tc) and cfgp and os.path.exists(cfgp)) else 'invalid'
        break
print('counts_valid,invalid,affected=',valid,invalid,affected)
print('spec_status',spec_id,status)
print('affected_sample=', details[:20])
