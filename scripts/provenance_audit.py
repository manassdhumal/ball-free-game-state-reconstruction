"""
Provenance audit for Step 74 (read-only).

Produces a JSON and human-readable report to stdout. Does not modify existing results.
"""
import re
import os
import json
import hashlib
from pathlib import Path

ROOT = Path('.').resolve()
METRICS_ROOT = ROOT / 'results' / 'metrics'
RUNS_ROOT = ROOT / 'results' / 'runs'
FULL_RUN_REPORT = RUNS_ROOT / 'robustness_expansion' / 'full_run_report.json'

EXPECTED_WINDOWS = ['0_60s','300_360s','600_660s','900_960s']
EXPECTED_SEEDS = ['72','73','74','75','76']
EXPECTED_SEVS = ['mild','moderate','severe']

EXPECTED_RUN_IDS = [f"exp_{w}_seed_{s}_sev_{v}" for w in EXPECTED_WINDOWS for s in EXPECTED_SEEDS for v in EXPECTED_SEVS]
EXPECTED_SET = set(EXPECTED_RUN_IDS)

# Patterns
SUMMARY_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_summary\.csv$')
TACTICAL_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_tactical_scores\.csv$')
CONFIG_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_config\.json$')

# collect candidates
candidates = {}
for root,dirs,files in os.walk(str(ROOT / 'results')):
    for fn in files:
        p = Path(root) / fn
        m1 = SUMMARY_RE.search(fn)
        m2 = TACTICAL_RE.search(fn)
        m3 = CONFIG_RE.search(fn)
        if m1:
            runid = m1.group(1)
            candidates.setdefault(runid,{}).setdefault('summary',[]).append(str(p))
        if m2:
            runid = m2.group(1)
            candidates.setdefault(runid,{}).setdefault('tactical',[]).append(str(p))
        if m3:
            runid = m3.group(1)
            candidates.setdefault(runid,{}).setdefault('config',[]).append(str(p))

# path_score and canonical choice
def path_score(path_str):
    parts = Path(path_str).parts
    return parts.count('robustness_expansion')

def choose_canonical(paths):
    # return one canonical path and list of noncanonical extras
    if not paths:
        return None,[]
    scored = [(path_score(p), p) for p in paths]
    scored.sort(key=lambda x:(x[0], x[1]))
    canonical = scored[0][1]
    extras = [p for _,p in scored[1:]]
    return canonical, extras

# For each expected run id, pick canonicals
report = {
    'expected_ids': sorted(EXPECTED_RUN_IDS),
    'found_run_ids': sorted(list(candidates.keys())),
    'per_run': {},
    'noncanonical_paths': [],
}

import pandas as pd

for runid in EXPECTED_RUN_IDS:
    info = {'runid':runid,'summary':None,'summary_extras':[],'tactical':None,'tactical_extras':[],'config':None,'config_extras':[]}
    c = candidates.get(runid, {})
    summary_can, summary_ex = choose_canonical(c.get('summary',[]))
    tact_can, tact_ex = choose_canonical(c.get('tactical',[]))
    conf_can, conf_ex = choose_canonical(c.get('config',[]))
    info['summary'] = summary_can
    info['summary_extras'] = summary_ex
    info['tactical'] = tact_can
    info['tactical_extras'] = tact_ex
    info['config'] = conf_can
    info['config_extras'] = conf_ex
    # record noncanonical paths with classification
    for p in summary_ex + tact_ex + conf_ex:
        cls = 'LEGACY_DUPLICATE' if path_score(p)>1 else 'NONCANONICAL'
        report['noncanonical_paths'].append({'path':p,'class':cls,'runid':runid})
    # validate contents if present
    details = {}
    # summary checks
    if summary_can:
        try:
            df = pd.read_csv(summary_can)
            details['summary_rows'] = len(df)
            conditions = list(df['condition'].astype(str)) if 'condition' in df.columns else None
            details['summary_conditions'] = conditions
            # compute hash
            with open(summary_can,'rb') as fh:
                hsum = hashlib.sha256(fh.read()).hexdigest()
            details['summary_hash'] = hsum
            details['summary_mtime'] = os.path.getmtime(summary_can)
            details['summary_size'] = os.path.getsize(summary_can)
        except Exception as e:
            details['summary_error'] = str(e)
    if tact_can:
        try:
            with open(tact_can,'rb') as fh:
                ht = hashlib.sha256(fh.read()).hexdigest()
            details['tactical_hash'] = ht
            details['tactical_mtime'] = os.path.getmtime(tact_can)
            details['tactical_size'] = os.path.getsize(tact_can)
        except Exception as e:
            details['tactical_error'] = str(e)
    if conf_can:
        try:
            import json as _json
            with open(conf_can,'r',encoding='utf8') as fh:
                cfg = _json.load(fh)
            details['config'] = cfg
            with open(conf_can,'rb') as fh:
                hc = hashlib.sha256(fh.read()).hexdigest()
            details['config_hash'] = hc
            details['config_mtime'] = os.path.getmtime(conf_can)
            details['config_size'] = os.path.getsize(conf_can)
        except Exception as e:
            details['config_error'] = str(e)
    report['per_run'][runid] = {'paths':info, 'details':details}

# compute expected vs valid
valid_ids = []
missing_ids = []
extra_ids = []
for runid in EXPECTED_RUN_IDS:
    p = report['per_run'][runid]['paths']
    d = report['per_run'][runid]['details']
    if p['summary'] and p['tactical'] and p['config'] and 'summary_rows' in d and d.get('summary_rows',0)==3 and d.get('summary_conditions'):
        conds = set([c.strip() for c in d['summary_conditions']])
        if conds==set(['CLEAN','SYNTHETIC-DEGRADED','NOISE-MITIGATED']):
            # verify config fields
            cfg = d.get('config')
            if isinstance(cfg,dict):
                # check fields
                w = str(cfg.get('window') or cfg.get('window_label') or cfg.get('window_name') or (cfg.get('params') or {}).get('window') or '')
                s = str(cfg.get('seed') or (cfg.get('params') or {}).get('seed') or '')
                sev = str(cfg.get('severity') or (cfg.get('params') or {}).get('severity') or '')
                if (w in runid) and (s in runid) and (sev in runid):
                    valid_ids.append(runid)
                else:
                    missing_ids.append(runid)
            else:
                missing_ids.append(runid)
        else:
            missing_ids.append(runid)
    else:
        missing_ids.append(runid)

# extra ids: ones found on filesystem but not expected
for fid in sorted(candidates.keys()):
    if fid not in EXPECTED_SET:
        extra_ids.append(fid)

# fingerprints across runs for duplication
fingerprints = {}
for runid, v in report['per_run'].items():
    d = v['details']
    if all(k in d for k in ('summary_hash','tactical_hash','config_hash')):
        fp = hashlib.sha256((d['summary_hash']+d['tactical_hash']+d['config_hash']).encode()).hexdigest()
        fingerprints.setdefault(fp,[]).append(runid)

duplicate_content_groups = [g for g in fingerprints.values() if len(g)>1]

# full_run_report.json comparison
full_report = None
if FULL_RUN_REPORT.exists():
    try:
        full_report = json.loads(FULL_RUN_REPORT.read_text(encoding='utf8'))
    except Exception as e:
        full_report = {'error':str(e)}

report_summary = {
    'expected_count': len(EXPECTED_RUN_IDS),
    'valid_unique_runspecs': len(valid_ids),
    'valid_ids': sorted(valid_ids),
    'missing_ids': sorted(set(missing_ids)),
    'extra_ids': sorted(extra_ids),
    'duplicate_runs_count': len([1 for item in report['noncanonical_paths'] if item['class']=='LEGACY_DUPLICATE']),
    'duplicate_artifacts_count': sum(len(v.get('summary_extras',[]))+len(v.get('tactical_extras',[]))+len(v.get('config_extras',[])) for v in [report['per_run'][rid]['paths'] for rid in EXPECTED_RUN_IDS]),
    'noncanonical_paths_count': len(report['noncanonical_paths']),
    'noncanonical_paths': report['noncanonical_paths'],
    'duplicate_content_groups': duplicate_content_groups,
    'full_run_report': full_report,
    'per_run': report['per_run'],
}

print(json.dumps(report_summary, indent=2, ensure_ascii=False))

# write to scripts/provenance_audit_output.json (workspace-local)
outp = ROOT / 'scripts' / 'provenance_audit_output.json'
outp.write_text(json.dumps(report_summary, indent=2, ensure_ascii=False))
print('\nWrote audit JSON to', outp)
