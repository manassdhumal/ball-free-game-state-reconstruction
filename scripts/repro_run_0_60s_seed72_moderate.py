import time
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.validation.robustness_experiment import load_experiment_config, make_grid, run_single_runspec


def find_runspec(grid):
    for r in grid:
        if r.window_label == '0_60s' and r.seed == 72 and r.severity == 'moderate':
            return r
    raise SystemExit('Runspec not found')


def locate_previous_summary():
    p = Path('results') / 'metrics' / 'robustness_expansion'
    matches = list(p.rglob('*_seed_72_sev_moderate_summary.csv'))
    if not matches:
        raise SystemExit('Previous summary CSV not found')
    # pick the first match
    return matches[0]


def df_to_metric_dict(df):
    # index by condition
    d = {}
    for _, row in df.iterrows():
        cond = row.get('condition')
        d[cond] = {}
        for k, v in row.items():
            # convert numpy types
            try:
                if pd.isna(v):
                    d[cond][k] = None
                elif isinstance(v, (np.floating, float)):
                    d[cond][k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    d[cond][k] = int(v)
                else:
                    d[cond][k] = v
            except Exception:
                d[cond][k] = str(v)
    return d


def compare_dfs(df_old, df_new):
    # align by condition and columns
    df_old = df_old.set_index('condition')
    df_new = df_new.set_index('condition')
    common_conditions = sorted(set(df_old.index) & set(df_new.index))
    cols = sorted(set(df_old.columns) & set(df_new.columns))
    diffs = {}
    all_close = True
    for cond in common_conditions:
        diffs[cond] = {}
        for col in cols:
            v_old = df_old.at[cond, col]
            v_new = df_new.at[cond, col]
            try:
                if pd.api.types.is_numeric_dtype(type(v_old)) or pd.api.types.is_numeric_dtype(type(v_new)):
                    ok = np.allclose([v_old], [v_new], rtol=1e-12, atol=1e-12, equal_nan=True)
                    diffs[cond][col] = {'old': float(v_old) if not pd.isna(v_old) else None, 'new': float(v_new) if not pd.isna(v_new) else None, 'equal': bool(ok)}
                    if not ok:
                        all_close = False
                else:
                    eq = (v_old == v_new)
                    diffs[cond][col] = {'old': v_old, 'new': v_new, 'equal': bool(eq)}
                    if not eq:
                        all_close = False
            except Exception:
                diffs[cond][col] = {'old': str(v_old), 'new': str(v_new), 'equal': v_old == v_new}
                if v_old != v_new:
                    all_close = False
    return all_close, diffs


def human_table(df):
    # return small dict of key metrics required
    out = {}
    for _, row in df.iterrows():
        cond = row.get('condition')
        out.setdefault(cond, {})
        out[cond]['possession_f1'] = float(row.get('possession_f1')) if 'possession_f1' in row and not pd.isna(row.get('possession_f1')) else None
        out[cond]['tactical_mean_absolute_score_error'] = float(row.get('tactical_mean_absolute_score_error')) if 'tactical_mean_absolute_score_error' in row and not pd.isna(row.get('tactical_mean_absolute_score_error')) else None
    return out


def main():
    cfg = load_experiment_config('configs/robustness_experiment.yaml')
    grid = make_grid(cfg)
    runspec = find_runspec(grid)

    prev_summary_path = locate_previous_summary()
    df_prev = pd.read_csv(prev_summary_path)

    start = time.perf_counter()
    out = run_single_runspec(runspec, cfg, root=ROOT)
    dur = time.perf_counter() - start

    df_new = out['summary_table']

    # compare
    reproducible, diffs = compare_dfs(df_prev, df_new)

    report = {
        'runspec': {'window': '0_60s', 'seed': 72, 'severity': 'moderate'},
        'previous_summary_path': str(prev_summary_path),
        'duration_s': dur,
        'reproducible': bool(reproducible),
        'metrics_prev': df_to_metric_dict(df_prev),
        'metrics_new': df_to_metric_dict(df_new),
        'diffs': diffs,
    }

    outpath = Path('results') / 'runs' / 'robustness_expansion' / 'repro_report_seed72_moderate.json'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, default=str))
    print('WROTE', outpath)
    print('REPRODUCIBLE:', reproducible)


if __name__ == '__main__':
    main()
