import json, os, hashlib
from pathlib import Path

ROOT = Path('.').resolve()
PA = ROOT / 'scripts' / 'provenance_audit_output.json'
if not PA.exists():
    print('provenance_audit_output.json missing')
    raise SystemExit(1)

data = json.loads(PA.read_text(encoding='utf8'))
valid_ids = data.get('valid_ids', [])

# find canonical summary for each valid id
import re
SUMMARY_RE = re.compile(r'robustness_benchmark_(' + r'[^_].*?)_summary\.csv$')

candidates = {}
for root,dirs,files in os.walk(str(ROOT / 'results')):
    for fn in files:
        m = SUMMARY_RE.search(fn)
        if m:
            runid = m.group(1)
            candidates.setdefault(runid,[]).append(str(Path(root)/fn))

from pathlib import Path

def path_score(path_str):
    return Path(path_str).parts.count('robustness_expansion')

def choose(paths):
    if not paths: return None
    scored = [(path_score(p), p) for p in paths]
    scored.sort(key=lambda x:(x[0], x[1]))
    return scored[0][1]

mtimes = {}
for rid in valid_ids:
    paths = candidates.get(rid, [])
    can = choose(paths)
    if can and Path(can).exists():
        mtimes[rid] = os.path.getmtime(can)

if not mtimes:
    print('No mtimes found')
    raise SystemExit(1)

items = sorted(mtimes.items(), key=lambda x:x[1])
first = items[0]
last = items[-1]

# distribution by window/seed/severity
from collections import Counter, defaultdict
win_ctr = Counter()
seed_ctr = Counter()
sev_ctr = Counter()

for rid, m in mtimes.items():
    import re
    mobj = re.match(r'^exp_(.+?)_seed_(\d+)_sev_(.+)$', rid)
    if mobj:
        window = mobj.group(1)
        seed = mobj.group(2)
        sev = mobj.group(3)
    else:
        window,seed,sev = '??','??','??'
    win_ctr[window]+=1
    seed_ctr[seed]+=1
    sev_ctr[sev]+=1

# identical timestamp clusters
clusters = defaultdict(list)
for rid,m in mtimes.items():
    clusters[m].append(rid)

identical = {k:v for k,v in clusters.items() if len(v)>1}

out = {
    'earliest': {'runid':first[0],'mtime':first[1]},
    'latest': {'runid':last[0],'mtime':last[1]},
    'counts_by_window': dict(win_ctr),
    'counts_by_seed': dict(seed_ctr),
    'counts_by_severity': dict(sev_ctr),
    'identical_timestamp_clusters': {str(k):v for k,v in identical.items()},
}
print(json.dumps(out, indent=2))
