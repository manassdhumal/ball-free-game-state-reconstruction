"""Generate final report CSVs, figures, and an extended JSON report (analysis-only).

This script reads canonical summary CSVs via the existing aggregator and writes
report-ready CSVs and PNG figures under results/metrics/final_analysis and
results/figures/final_analysis.

Do NOT modify experimental outputs; this script only reads them and writes
derived summaries/figures.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts import aggregate_robustness_summary as agg

OUT = Path('results') / 'metrics' / 'final_analysis'
FIG = Path('results') / 'figures' / 'final_analysis'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# gather canonical summaries
art = agg.find_artifacts()
records = []
meta = []
for run_id, a in sorted(art.items()):
    s = agg.choose_canonical(a.get('summary_csv', []))
    cfg = agg.choose_canonical(a.get('config_json', []))
    if not s:
        continue
    df = pd.read_csv(s)
    # attach run_id
    for _, r in df.iterrows():
        rec = dict(r)
        rec['run_id'] = run_id
        records.append(rec)
    # parse run_id components if possible
    m = agg.re.match(r'exp_([0-9]+_[0-9]+s)_seed_([0-9]+)_sev_([A-Z-]+)', run_id)
    if m:
        w, sd, sev = m.groups()
    else:
        # fallback: attempt split
        parts = run_id.split('_')
        w = parts[1] if len(parts) > 1 else 'unknown'
        sd = parts[-2] if len(parts) > 2 else 'na'
        sev = parts[-1] if len(parts) > 0 else 'na'
    meta.append({'run_id': run_id, 'window': w, 'seed': sd, 'severity': sev})

all_df = pd.DataFrame(records)
meta_df = pd.DataFrame(meta).drop_duplicates(subset=['run_id']).set_index('run_id')

# basic verification
n_runs = all_df['run_id'].nunique()
n_rows = len(all_df)
print('n_runs', n_runs, 'n_rows', n_rows)

# experimental_design.csv
exp_design = meta_df.reset_index()
exp_design.to_csv(OUT / 'experimental_design.csv', index=False)

# condition distribution
cond_dist = all_df.groupby('condition').size().reset_index(name='count')
cond_dist.to_csv(OUT / 'condition_distribution.csv', index=False)

# tracking quality by condition
tracking_metrics = [c for c in all_df.columns if c.startswith('tracking_')]
tracking_summary = []
for cond, g in all_df.groupby('condition'):
    row = {'condition': cond, 'n_runs': g['run_id'].nunique()}
    for mcol in tracking_metrics:
        arr = g[mcol].astype(float).dropna().values
        if len(arr)>0:
            row.update({f'{mcol}_mean': np.mean(arr), f'{mcol}_median': np.median(arr), f'{mcol}_std': np.std(arr, ddof=1), f'{mcol}_min': np.min(arr), f'{mcol}_max': np.max(arr)})
        else:
            row.update({f'{mcol}_mean': np.nan, f'{mcol}_median': np.nan, f'{mcol}_std': np.nan, f'{mcol}_min': np.nan, f'{mcol}_max': np.nan})
    tracking_summary.append(row)
pd.DataFrame(tracking_summary).to_csv(OUT / 'tracking_quality_by_condition.csv', index=False)

# possession by condition
poss_metrics = ['possession_precision','possession_recall','possession_f1']
poss_summary = []
for cond, g in all_df.groupby('condition'):
    row = {'condition': cond, 'n_runs': g['run_id'].nunique()}
    for mcol in poss_metrics:
        arr = g[mcol].astype(float).dropna().values
        if len(arr)>0:
            # bootstrap CI for mean
            rng = np.random.RandomState(42)
            bmeans = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(2000)]
            lo, hi = np.percentile(bmeans, [2.5,97.5])
            row.update({f'{mcol}_mean': np.mean(arr), f'{mcol}_median': np.median(arr), f'{mcol}_std': np.std(arr, ddof=1), f'{mcol}_min': np.min(arr), f'{mcol}_max': np.max(arr), f'{mcol}_ci95_lo': lo, f'{mcol}_ci95_hi': hi})
        else:
            row.update({f'{mcol}_mean': np.nan})
    poss_summary.append(row)
pd.DataFrame(poss_summary).to_csv(OUT / 'possession_by_condition.csv', index=False)

# tactical by condition
tac_metrics = ['tactical_mean_absolute_score_error','tactical_median_absolute_score_error','tactical_correlation_with_clean','tactical_top_k_agreement','tactical_possession_level_score_stability']
tac_summary=[]
for cond,g in all_df.groupby('condition'):
    row={'condition':cond,'n_runs':g['run_id'].nunique()}
    for mcol in tac_metrics:
        if mcol in g.columns:
            arr = g[mcol].astype(float).dropna().values
            if len(arr)>0:
                rng = np.random.RandomState(42)
                bmeans=[np.mean(rng.choice(arr,size=len(arr),replace=True)) for _ in range(2000)]
                lo,hi=np.percentile(bmeans,[2.5,97.5])
                row.update({f'{mcol}_mean':np.mean(arr),f'{mcol}_median':np.median(arr),f'{mcol}_std':np.std(arr,ddof=1),f'{mcol}_min':np.min(arr),f'{mcol}_max':np.max(arr),f'{mcol}_ci95_lo':lo,f'{mcol}_ci95_hi':hi})
            else:
                row.update({f'{mcol}_mean':np.nan})
        else:
            row.update({f'{mcol}_mean':np.nan})
    tac_summary.append(row)
pd.DataFrame(tac_summary).to_csv(OUT / 'tactical_robustness_by_condition.csv', index=False)

# possession_by_severity and tactical_by_severity
all_with_meta = all_df.join(meta_df, on='run_id')
poss_by_sev=[]
for sev,g in all_with_meta.groupby('severity'):
    row={'severity':sev,'n_runs':g['run_id'].nunique()}
    for mcol in poss_metrics:
        arr=g[mcol].astype(float).dropna().values
        if len(arr)>0:
            row.update({f'{mcol}_mean':np.mean(arr),f'{mcol}_std':np.std(arr,ddof=1),f'{mcol}_median':np.median(arr),f'{mcol}_min':np.min(arr),f'{mcol}_max':np.max(arr)})
        else:
            row.update({f'{mcol}_mean':np.nan})
    poss_by_sev.append(row)
pd.DataFrame(poss_by_sev).to_csv(OUT / 'possession_by_severity.csv', index=False)

# tactical by severity
tac_by_sev=[]
for sev,g in all_with_meta.groupby('severity'):
    row={'severity':sev,'n_runs':g['run_id'].nunique()}
    for mcol in tac_metrics:
        if mcol in g.columns:
            arr=g[mcol].astype(float).dropna().values
            if len(arr)>0:
                row.update({f'{mcol}_mean':np.mean(arr),f'{mcol}_std':np.std(arr,ddof=1),f'{mcol}_median':np.median(arr),f'{mcol}_min':np.min(arr),f'{mcol}_max':np.max(arr)})
            else:
                row.update({f'{mcol}_mean':np.nan})
        else:
            row.update({f'{mcol}_mean':np.nan})
    tac_by_sev.append(row)
pd.DataFrame(tac_by_sev).to_csv(OUT / 'tactical_by_severity.csv', index=False)

# mitigation_recovery per metric (possession_f1 and tactical_mae)
# build wide per-run
poss_wide = all_df.pivot_table(index='run_id', columns='condition', values='possession_f1')
tac_wide = all_df.pivot_table(index='run_id', columns='condition', values='tactical_mean_absolute_score_error')
rec_rows=[]
for idx,row in poss_wide.iterrows():
    clean=row.get('CLEAN',np.nan); deg=row.get('SYNTHETIC-DEGRADED',np.nan); mit=row.get('NOISE-MITIGATED',np.nan)
    if np.isnan(clean) or np.isnan(deg) or np.isnan(mit):
        continue
    abs_imp = mit - deg
    denom = clean - deg
    frac = abs_imp/denom if denom!=0 else np.nan
    rec_rows.append({'run_id':idx,'metric':'possession_f1','clean':clean,'degraded':deg,'mitigated':mit,'abs_improvement':abs_imp,'recovery_fraction':frac})
for idx,row in tac_wide.iterrows():
    clean=row.get('CLEAN',np.nan); deg=row.get('SYNTHETIC-DEGRADED',np.nan); mit=row.get('NOISE-MITIGATED',np.nan)
    if np.isnan(clean) or np.isnan(deg) or np.isnan(mit):
        continue
    abs_imp = deg - mit
    denom = deg - clean
    frac = abs_imp/denom if denom!=0 else np.nan
    rec_rows.append({'run_id':idx,'metric':'tactical_mae','clean':clean,'degraded':deg,'mitigated':mit,'abs_improvement':abs_imp,'recovery_fraction':frac})

pd.DataFrame(rec_rows).to_csv(OUT / 'mitigation_recovery.csv', index=False)

# robust_by_window and by_seed
win_summary=[]
for w,g in all_with_meta.groupby('window'):
    row={'window':w,'n_runs':g['run_id'].nunique()}
    # possession f1 mean
    arr=g['possession_f1'].astype(float).dropna().values
    row.update({'possession_f1_mean':np.mean(arr) if len(arr)>0 else np.nan})
    win_summary.append(row)
pd.DataFrame(win_summary).to_csv(OUT / 'robustness_by_window.csv', index=False)

seed_summary=[]
for sd,g in all_with_meta.groupby('seed'):
    row={'seed':sd,'n_runs':g['run_id'].nunique()}
    arr=g['possession_f1'].astype(float).dropna().values
    row.update({'possession_f1_mean':np.mean(arr) if len(arr)>0 else np.nan,'possession_f1_std':np.std(arr,ddof=1) if len(arr)>1 else np.nan})
    seed_summary.append(row)
pd.DataFrame(seed_summary).to_csv(OUT / 'robustness_by_seed.csv', index=False)

# simple figures
# 1. possession_f1_vs_severity
sev_order = list(all_with_meta['severity'].unique())
poss_by_sev_df = pd.read_csv(OUT / 'possession_by_severity.csv')
plt.figure()
plt.bar(poss_by_sev_df['severity'], poss_by_sev_df['possession_f1_mean'])
plt.title('Possession F1 vs Severity')
plt.ylabel('Possession F1 (mean)')
plt.xlabel('Severity')
plt.savefig(FIG / 'possession_f1_vs_severity.png')
plt.close()

# 2. tactical_mae_vs_severity
tac_by_sev_df = pd.read_csv(OUT / 'tactical_by_severity.csv')
plt.figure()
if 'tactical_mean_absolute_score_error_mean' in tac_by_sev_df.columns:
    plt.bar(tac_by_sev_df['severity'], tac_by_sev_df['tactical_mean_absolute_score_error_mean'])
plt.title('Tactical MAE vs Severity')
plt.ylabel('Tactical MAE (mean)')
plt.xlabel('Severity')
plt.savefig(FIG / 'tactical_mae_vs_severity.png')
plt.close()

# 3. tracking_quality_vs_severity (use tracking_missingness_rate if present)
track_cols = [c for c in all_df.columns if c.startswith('tracking_')]
plt.figure()
if 'tracking_missingness_rate' in all_df.columns:
    grp = all_with_meta.groupby('severity')['tracking_missingness_rate'].mean().reset_index()
    plt.bar(grp['severity'], grp['tracking_missingness_rate'])
plt.title('Tracking Missingness vs Severity')
plt.ylabel('Missingness rate')
plt.savefig(FIG / 'tracking_quality_vs_severity.png')
plt.close()

# 4. possession_degraded_vs_mitigated (scatter)
if 'SYNTHETIC-DEGRADED' in poss_wide.columns and 'NOISE-MITIGATED' in poss_wide.columns:
    plt.figure()
    plt.scatter(poss_wide['SYNTHETIC-DEGRADED'], poss_wide['NOISE-MITIGATED'])
    plt.xlabel('Degraded possession_f1')
    plt.ylabel('Mitigated possession_f1')
    plt.title('Possession: Degraded vs Mitigated')
    plt.savefig(FIG / 'possession_degraded_vs_mitigated.png')
    plt.close()

# 5. tactical_degraded_vs_mitigated
if 'SYNTHETIC-DEGRADED' in tac_wide.columns and 'NOISE-MITIGATED' in tac_wide.columns:
    plt.figure()
    plt.scatter(tac_wide['SYNTHETIC-DEGRADED'], tac_wide['NOISE-MITIGATED'])
    plt.xlabel('Degraded tactical MAE')
    plt.ylabel('Mitigated tactical MAE')
    plt.title('Tactical: Degraded vs Mitigated')
    plt.savefig(FIG / 'tactical_degraded_vs_mitigated.png')
    plt.close()

# 6. mitigation_by_window (mean recovery fraction)
rec_df = pd.read_csv(OUT / 'mitigation_recovery.csv')
# focus on possession_f1 rows
rec_poss = rec_df[rec_df['metric']=='possession_f1'].merge(meta_df.reset_index(), on='run_id')
rec_win = rec_poss.groupby('window')['recovery_fraction'].mean().reset_index()
plt.figure()
plt.bar(rec_win['window'], rec_win['recovery_fraction'])
plt.title('Possession Recovery by Window')
plt.ylabel('Recovery fraction')
plt.savefig(FIG / 'mitigation_by_window.png')
plt.close()

# 7. mitigation_by_seed
rec_seed = rec_poss.groupby('seed')['recovery_fraction'].mean().reset_index()
plt.figure()
plt.bar(rec_seed['seed'].astype(str), rec_seed['recovery_fraction'])
plt.title('Possession Recovery by Seed')
plt.ylabel('Recovery fraction')
plt.xlabel('Seed')
plt.savefig(FIG / 'mitigation_by_seed.png')
plt.close()

# 8. tracking_vs_possession scatter
plt.figure()
plt.scatter(all_df['tracking_missingness_rate'].astype(float), all_df['possession_f1'].astype(float))
plt.xlabel('tracking_missingness_rate')
plt.ylabel('possession_f1')
plt.title('Tracking Missingness vs Possession F1')
plt.savefig(FIG / 'tracking_vs_possession.png')
plt.close()

# 9. tracking_vs_tactical_instability (use tactical_median_absolute_score_error)
plt.figure()
plt.scatter(all_df['tracking_missingness_rate'].astype(float), all_df['tactical_median_absolute_score_error'].astype(float))
plt.xlabel('tracking_missingness_rate')
plt.ylabel('tactical_median_absolute_score_error')
plt.title('Tracking Missingness vs Tactical Median AE')
plt.savefig(FIG / 'tracking_vs_tactical_instability.png')
plt.close()

# final JSON summary
summary = {
    'n_runs': int(n_runs),
    'n_condition_rows': int(n_rows),
}
Path(OUT / 'final_statistical_report_extended.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Generated CSVs and figures under', OUT, 'and', FIG)
