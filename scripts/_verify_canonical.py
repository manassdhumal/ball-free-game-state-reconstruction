from scripts import aggregate_robustness_summary as agg
import pandas as pd
import json

art = agg.find_artifacts()
print('canonical_runs:', len(art))
rows = 0
invalid = 0
conds = {}
seeds = {}
windows = {}
severities = {}
cols_sample = None
for run_id, a in sorted(art.items()):
    s = agg.choose_canonical(a.get('summary_csv', []))
    if not s:
        invalid += 1
        continue
    df = pd.read_csv(s)
    if cols_sample is None:
        cols_sample = list(df.columns)
    rows += len(df)
    # condition distribution
    for c in df['condition'].unique():
        conds[c] = conds.get(c, 0) + df[df['condition']==c].shape[0]
    # parse run_id
    m = agg.re.match(r'exp_([0-9]+_[0-9]+s)_seed_([0-9]+)_sev_([A-Z-]+)', run_id)
    if m:
        w, sd, sev = m.groups()
        windows[w] = windows.get(w, 0) + 1
        seeds[sd] = seeds.get(sd, 0) + 1
        severities[sev] = severities.get(sev, 0) + 1

print('invalid_summaries:', invalid)
print('total_condition_rows:', rows)
print('condition_distribution:', conds)
print('seed_counts:', seeds)
print('severity_counts:', severities)
print('window_counts:', windows)
print('sample_summary_columns:', cols_sample)

# quick check of expected counts
print('expected_runs=60 expected_rows=180')
print('runs_ok=', len(art)==60, 'rows_ok=', rows==180)
