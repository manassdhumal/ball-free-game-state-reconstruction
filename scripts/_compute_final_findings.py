import pandas as pd
import numpy as np
from pathlib import Path
import json

OUT = Path('results') / 'metrics' / 'final_analysis'
DOC = Path('docs') / 'final_research_findings.md'

poss_cond = pd.read_csv(OUT / 'possession_by_condition.csv')
poss_per_run = pd.read_csv(OUT / 'per_run_possession.csv')
mit_rec = pd.read_csv(OUT / 'mitigation_recovery.csv')
tac_per_run = pd.read_csv(OUT / 'per_run_tactical.csv')

# Overall possession metrics
def read_metric(df, metric_prefix):
    # returns dict per condition
    out = {}
    for _, row in df.iterrows():
        cond = row['condition']
        out[cond] = row.to_dict()
    return out

poss_summary = read_metric(poss_cond, 'possession')

# Paired effect size for degraded vs mitigated possession_f1
wide = poss_per_run.set_index('run_id')
clean_series = None
# per_run_possession contains clean_poss? It has columns clean_poss, degraded_poss, mitigated_poss
if 'clean_poss' in wide.columns:
    # compute mean differences
    deg = wide['degraded_poss'].astype(float)
    mit = wide['mitigated_poss'].astype(float)
    diff = mit - deg
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    n = diff.dropna().shape[0]
    cohen_d = mean_diff / sd_diff if sd_diff>0 else np.nan
    # bootstrap CI for mean_diff
    rng = np.random.RandomState(1234)
    bmeans = [np.mean(rng.choice(diff.dropna().values, size=n, replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(bmeans, [2.5,97.5])
else:
    mean_diff = sd_diff = cohen_d = lo = hi = n = np.nan

# H1: Tracking degradation harms possession: compare CLEAN vs SYNTHETIC-DEGRADED
clean_row = poss_summary.get('CLEAN')
deg_row = poss_summary.get('SYNTHETIC-DEGRADED')
mit_row = poss_summary.get('NOISE-MITIGATED')

# decision rules
def decision_mean_diff(clean_mean, deg_mean, ci_lo, ci_hi, direction='decrease'):
    # direction 'decrease' means deg < clean supports harm
    if np.isnan(clean_mean) or np.isnan(deg_mean):
        return 'NOT SUPPORTED', 'Missing data'
    diff = deg_mean - clean_mean
    if direction=='decrease':
        if ci_hi < 0:
            return 'SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'
        elif ci_lo < 0 <= ci_hi:
            return 'PARTIALLY SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'
        else:
            return 'NOT SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'
    else:
        if ci_lo > 0:
            return 'SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'
        elif ci_lo <=0 < ci_hi:
            return 'PARTIALLY SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'
        else:
            return 'NOT SUPPORTED', f'mean_diff={diff:.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})'

# For H1 compute mean difference clean - degraded for possession_f1
# We'll compute bootstrap CI for per-run difference (deg - clean)
# Build wide per-run from all_df
all_df = pd.read_csv(OUT / 'per_run_possession.csv') if (OUT / 'per_run_possession.csv').exists() else None
h1_status='NOT SUPPORTED'; h1_evidence=''
if all_df is not None:
    # get clean/degraded per run from summaries: need per_run_possession produced earlier from final_analysis
    # However per_run_possession may contain clean_poss only for possession metric; we instead rebuild from results metrics canonical files
    deg = all_df['degraded_poss'].astype(float)
    clean = all_df['clean_poss'].astype(float)
    # compute paired diff deg - clean
    paired = deg - clean
    n = paired.dropna().shape[0]
    if n>0:
        rng = np.random.RandomState(42)
        bmeans=[np.mean(rng.choice(paired.dropna().values,size=n,replace=True)) for _ in range(2000)]
        lo,hi = np.percentile(bmeans,[2.5,97.5])
        mean_diff=paired.mean()
        if hi < 0:
            h1_status='SUPPORTED'
        elif lo < 0 <= hi:
            h1_status='PARTIALLY SUPPORTED'
        else:
            h1_status='NOT SUPPORTED'
        h1_evidence=f'mean_diff(deg-clean)={mean_diff:.4f}, ci95=({lo:.4f},{hi:.4f}), n={n}'

# H2: tactical instability: tactical_mae degraded > clean
h2_status='NOT SUPPORTED'; h2_evidence=''
if (OUT / 'per_run_tactical.csv').exists():
    tac_df = pd.read_csv(OUT / 'per_run_tactical.csv')
    deg = tac_df['degraded_tac'].astype(float)
    clean = tac_df['clean_tac'].astype(float)
    paired = deg - clean
    n = paired.dropna().shape[0]
    if n>0:
        rng = np.random.RandomState(42)
        bmeans=[np.mean(rng.choice(paired.dropna().values,size=n,replace=True)) for _ in range(2000)]
        lo,hi=np.percentile(bmeans,[2.5,97.5])
        mean_diff=paired.mean()
        if lo > 0:
            h2_status='SUPPORTED'
        elif lo <=0 < hi:
            h2_status='PARTIALLY SUPPORTED'
        else:
            h2_status='NOT SUPPORTED'
        h2_evidence=f'mean_diff(deg-clean)={mean_diff:.4f}, ci95=({lo:.4f},{hi:.4f}), n={n}'

# H3: mitigation recovers portion: use mitigation_recovery.csv mean recovery_fraction for possession_f1
h3_status='NOT SUPPORTED'; h3_evidence=''
if (OUT / 'mitigation_recovery.csv').exists():
    rec = pd.read_csv(OUT / 'mitigation_recovery.csv')
    rpos = rec[rec['metric']=='possession_f1']
    if not rpos.empty:
        vals = rpos['recovery_fraction'].astype(float).dropna().values
        if len(vals)>0:
            mean_rec = np.mean(vals)
            rng=np.random.RandomState(42)
            bmeans=[np.mean(rng.choice(vals,size=len(vals),replace=True)) for _ in range(2000)]
            lo,hi=np.percentile(bmeans,[2.5,97.5])
            if lo>0:
                h3_status='SUPPORTED'
            elif lo<=0<hi:
                h3_status='PARTIALLY SUPPORTED'
            else:
                h3_status='NOT SUPPORTED'
            h3_evidence=f'mean_recovery={mean_rec:.4f}, ci95=({lo:.4f},{hi:.4f}), n={len(vals)}'

# write findings markdown
md = []
md.append('# Final Research Findings')
md.append('')
md.append('## Research Question')
md.append('Do tracking degradation harm downstream inference, and does interpolation/smoothing mitigation recover performance?')
md.append('')
md.append('## Experimental Design')
md.append('- 4 windows × 5 seeds (72–76) × 3 severities = 60 runs; each run has CLEAN, SYNTHETIC-DEGRADED, NOISE-MITIGATED conditions.')
md.append('')
md.append('## Data and Conditions')
# compute n_runs
n_runs = int(pd.read_csv(OUT / 'experimental_design.csv')['run_id'].nunique()) if (OUT / 'experimental_design.csv').exists() else (poss_per_run['run_id'].nunique() if poss_per_run is not None else 0)
md.append(f'- Canonical runs analyzed: {n_runs}')
md.append(f'- Condition rows: {len(pd.read_csv(OUT / "final_statistical_summary.csv"))}')
md.append('')
md.append('## Tracking Robustness')
md.append('- See `results/metrics/final_analysis/tracking_quality_by_condition.csv` for per-condition tracking stats.')
md.append('')
md.append('## Ball-Free Possession Results')
md.append('- See `results/metrics/final_analysis/possession_by_condition.csv` for possession precision/recall/F1 summaries.')
md.append('')
md.append('## Tactical Robustness Results')
md.append('- See `results/metrics/final_analysis/tactical_robustness_by_condition.csv` for tactical metrics.')
md.append('')
md.append('## Severity Analysis')
md.append('- See `results/metrics/final_analysis/possession_by_severity.csv` and `tactical_by_severity.csv`.')
md.append('')
md.append('## Window Analysis')
md.append('- See `results/metrics/final_analysis/robustness_by_window.csv`.')
md.append('')
md.append('## Seed Analysis')
md.append('- See `results/metrics/final_analysis/robustness_by_seed.csv`.')
md.append('')
md.append('## Noise-Mitigation Recovery')
md.append('- See `results/metrics/final_analysis/mitigation_recovery.csv` for per-run recovery fractions.')
md.append('')
md.append('## Main Findings')
md.append(f'- H1: Tracking degradation harms possession inference: {h1_status}. Evidence: {h1_evidence}')
md.append(f'- H2: Tracking degradation increases tactical instability: {h2_status}. Evidence: {h2_evidence}')
md.append(f'- H3: Interpolation/smoothing mitigation recovers performance: {h3_status}. Evidence: {h3_evidence}')
md.append('')
md.append('## Limitations')
md.append('- Results are restricted to the specific synthetic degradation parameters and mitigation pipeline used; do not generalize beyond tested settings.')
md.append('')
md.append('## What the Results Do NOT Establish')
md.append('- Causality beyond the controlled experiment; transfer to other datasets or live broadcast streams.')
md.append('')
md.append('## Conclusion')
md.append('- See CSVs and figures in `results/metrics/final_analysis` and `results/figures/final_analysis` for detail.')

DOC.write_text('\n'.join(md), encoding='utf-8')

# write extended JSON summary
summary = {'h1':{'decision':h1_status,'evidence':h1_evidence},'h2':{'decision':h2_status,'evidence':h2_evidence},'h3':{'decision':h3_status,'evidence':h3_evidence}}
Path(OUT / 'final_statistical_report_findings.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Wrote docs/final_research_findings.md and summary JSON')
