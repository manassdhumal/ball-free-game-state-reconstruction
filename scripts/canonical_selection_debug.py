import json,os,re
from pathlib import Path
ROOT=Path('.').resolve()
PA=ROOT/'scripts'/'provenance_audit_output.json'
data=json.loads(PA.read_text(encoding='utf8'))
valid_ids=data.get('valid_ids',[])

SUMMARY_RE=re.compile(r'robustness_benchmark_(' + r'[^_].*?)_summary\\.csv$')

candidates={}
for root,dirs,files in os.walk(str(ROOT/'results')):
    for fn in files:
        m=SUMMARY_RE.search(fn)
        if m:
            candidates.setdefault(m.group(1),[]).append(str(Path(root)/fn))

from pathlib import Path

def path_score(path_str):
    return Path(path_str).parts.count('robustness_expansion')


def choose(paths):
    if not paths: return None
    scored=[(path_score(p),p) for p in paths]
    scored.sort(key=lambda x:(x[0],x[1]))
    return scored[0], scored

for rid in valid_ids:
    paths=candidates.get(rid, [])
    if not paths:
        print(rid,'NO CANDIDATES')
        continue
    chosen, scored = choose(paths)
    min_score = scored[0][0]
    chosen_score = chosen[0]
    if chosen_score>1:
        print('OFFENDING', rid, 'chosen_score', chosen_score, 'min_score', min_score)
        for sc,p in scored[:10]:
            print('  ',sc,p)
        print('---')
