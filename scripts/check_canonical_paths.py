import json, os
from pathlib import Path
import re

ROOT = Path('.').resolve()
PA = ROOT / 'scripts' / 'provenance_audit_output.json'
if not PA.exists():
    print('missing provenance_audit_output.json')
    raise SystemExit(1)

data = json.loads(PA.read_text(encoding='utf8'))
valid_ids = data.get('valid_ids', [])

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

bad = []
for rid in valid_ids:
    cand = candidates.get(rid, {})
    def choose(paths):
        if not paths: return None
        scored = [(Path(p).parts.count('robustness_expansion'), p) for p in paths]
        scored.sort(key=lambda x:(x[0], x[1]))
        return scored[0][1]
    s = choose(cand.get('summary',[]))
    t = choose(cand.get('tactical',[]))
    c = choose(cand.get('config',[]))
    for p in (s,t,c):
        if p and Path(p).parts.count('robustness_expansion')>1:
            bad.append({'runid':rid,'path':p,'count':Path(p).parts.count('robustness_expansion')})

print({'bad':bad,'bad_count':len(bad)})
