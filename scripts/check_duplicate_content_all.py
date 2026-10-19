import json, os, hashlib
from pathlib import Path

ROOT = Path('.').resolve()
PA = ROOT / 'scripts' / 'provenance_audit_output.json'
if not PA.exists():
    print('missing provenance_audit_output.json')
    raise SystemExit(1)

data = json.loads(PA.read_text(encoding='utf8'))
valid_ids = data.get('valid_ids', [])

import re
SUMMARY_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_summary\.csv$')
TACTICAL_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_tactical_scores\.csv$')
CONFIG_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_config\.json$')

candidates = {}
for root,dirs,files in os.walk(str(ROOT / 'results')):
    for fn in files:
        p = Path(root)/fn
        m1 = SUMMARY_RE.search(fn)
        m2 = TACTICAL_RE.search(fn)
        m3 = CONFIG_RE.search(fn)
        if m1:
            candidates.setdefault(m1.group(1),{}).setdefault('summary',[]).append(str(p))
        if m2:
            candidates.setdefault(m2.group(1),{}).setdefault('tactical',[]).append(str(p))
        if m3:
            candidates.setdefault(m3.group(1),{}).setdefault('config',[]).append(str(p))

from pathlib import Path

def path_score(path_str):
    return Path(path_str).parts.count('robustness_expansion')

def choose(paths):
    if not paths: return None
    scored = [(path_score(p), p) for p in paths]
    scored.sort(key=lambda x:(x[0], x[1]))
    return scored[0][1]

fingerprints = {}
for rid in valid_ids:
    cand = candidates.get(rid, {})
    s = choose(cand.get('summary',[]))
    t = choose(cand.get('tactical',[]))
    c = choose(cand.get('config',[]))
    if s and t and c:
        with open(s,'rb') as fh:
            sh = hashlib.sha256(fh.read()).hexdigest()
        with open(t,'rb') as fh:
            th = hashlib.sha256(fh.read()).hexdigest()
        with open(c,'rb') as fh:
            ch = hashlib.sha256(fh.read()).hexdigest()
        fp = hashlib.sha256((sh+th+ch).encode()).hexdigest()
        fingerprints.setdefault(fp,[]).append(rid)

dups = [v for v in fingerprints.values() if len(v)>1]
print(json.dumps({'duplicate_content_groups':dups, 'count':len(dups)}, indent=2))
