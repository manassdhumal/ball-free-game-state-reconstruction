import re, os, json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path('.').resolve()

CASES = [
    'exp_0_60s_seed_72_sev_mild',
    'exp_0_60s_seed_72_sev_moderate',
    'exp_0_60s_seed_72_sev_severe',
    'exp_300_360s_seed_73_sev_mild',
    'exp_600_660s_seed_74_sev_moderate',
    'exp_900_960s_seed_76_sev_severe',
]

SUMMARY_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_summary\.csv$')
TACTICAL_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_tactical_scores\.csv$')
CONFIG_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_config\.json$')

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

from pathlib import Path

def path_score(path_str):
    return Path(path_str).parts.count('robustness_expansion')

def choose(paths):
    if not paths: return None,[]
    scored = [(path_score(p), p) for p in paths]
    scored.sort(key=lambda x:(x[0], x[1]))
    return scored[0][1], [p for _,p in scored[1:]]

out = {}
for runid in CASES:
    c = candidates.get(runid, {})
    s_can, s_ex = choose(c.get('summary', []))
    t_can, t_ex = choose(c.get('tactical', []))
    c_can, c_ex = choose(c.get('config', []))
    d = {'summary':s_can,'summary_extras':s_ex,'tactical':t_can,'tactical_extras':t_ex,'config':c_can,'config_extras':c_ex}
    details = {}
    if s_can:
        try:
            df = pd.read_csv(s_can)
            details['summary_row_count'] = len(df)
            details['summary_conditions'] = list(df['condition']) if 'condition' in df.columns else None
            with open(s_can,'rb') as fh:
                details['summary_hash'] = hashlib.sha256(fh.read()).hexdigest()
            details['summary_mtime'] = os.path.getmtime(s_can)
        except Exception as e:
            details['summary_error']=str(e)
    if t_can:
        try:
            with open(t_can,'rb') as fh:
                details['tactical_hash'] = hashlib.sha256(fh.read()).hexdigest()
            details['tactical_mtime'] = os.path.getmtime(t_can)
        except Exception as e:
            details['tactical_error']=str(e)
    if c_can:
        try:
            with open(c_can,'r',encoding='utf8') as fh:
                cfg = json.load(fh)
            details['config'] = cfg
            with open(c_can,'rb') as fh:
                details['config_hash'] = hashlib.sha256(fh.read()).hexdigest()
            details['config_mtime'] = os.path.getmtime(c_can)
        except Exception as e:
            details['config_error']=str(e)
    out[runid] = {'paths':d,'details':details}

print(json.dumps(out, indent=2))
