"""Final statistical analysis for Step 75 (analysis-only).

This module is analysis-only: it reads canonical summary CSVs selected by the
existing aggregator, computes run-level aggregates, group summaries, effect
sizes, bootstrap CIs, recovery fractions, and writes machine-readable outputs
under `results/metrics/final_analysis` and figures under
`results/figures/final_analysis`.

DO NOT modify any existing result files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts import aggregate_robustness_summary as agg


OUT_DIR = Path("results") / "metrics" / "final_analysis"
FIG_DIR = Path("results") / "figures" / "final_analysis"


def load_canonical_summaries(require_n_runs: int = 60) -> pd.DataFrame:
    """Load validated canonical summary CSVs into a single DataFrame.

    Returns a DataFrame with one row per (run_id, condition) and columns from
    the summary CSVs plus `run_id`, `window`, `seed`, `severity`.
    """
    artifacts = agg.find_artifacts()
    records = []
    valid_run_ids = []

    for run_id in sorted(artifacts.keys()):
        art = artifacts[run_id]
        summary = agg.choose_canonical(art.get("summary_csv", []))
        tactical = agg.choose_canonical(art.get("tactical_csv", []))
        config = agg.choose_canonical(art.get("config_json", []))
        if not (summary and tactical and config):
            continue
        v = agg.validate_summary_csv(summary)
        if not v.get("ok"):
            continue
        # try to parse config to ensure readable
        try:
            _ = json.loads(Path(config).read_text(encoding="utf-8"))
        except Exception:
            continue
        df = pd.read_csv(summary)
        for _, row in df.iterrows():
            rec = dict(row)
            rec["run_id"] = run_id
            # parse run_id
            m = agg.re.match(r"exp_([0-9]+_[0-9]+s)_seed_([0-9]+)_sev_([a-zA-Z]+)", run_id)
            if m:
                rec["window"], rec["seed"], rec["severity"] = m.groups()
                rec["seed"] = int(rec["seed"])
            records.append(rec)
        valid_run_ids.append(run_id)

    df_all = pd.DataFrame(records)
    # sanity checks
    unique_runs = df_all["run_id"].nunique()
    if unique_runs != require_n_runs:
        raise RuntimeError(f"Expected {require_n_runs} canonical runs, found {unique_runs}")
    # ensure 3 conditions per run
    cond_counts = df_all.groupby("run_id").size()
    if not (cond_counts == 3).all():
        raise RuntimeError("Not every run contains exactly 3 condition rows")
    return df_all


def describe_series(series: pd.Series, n_boot: int = 2000, seed: int = 42) -> Dict:
    arr = series.dropna().astype(float).values
    n = len(arr)
    if n == 0:
        return {"count": 0}
    mean = float(np.mean(arr))
    med = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    # bootstrap CI for the mean (percentile)
    rng = np.random.RandomState(seed)
    bmeans = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        bmeans.append(np.mean(sample))
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    return {
        "count": int(n),
        "mean": mean,
        "median": med,
        "std": std,
        "min": mn,
        "max": mx,
        "ci95_mean_lo": float(lo),
        "ci95_mean_hi": float(hi),
    }


def paired_effect_size(a: np.ndarray, b: np.ndarray) -> Dict:
    """Compute paired Cohen's d and mean diff with bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n == 0:
        return {"n": 0}
    dif = a - b
    md = float(np.mean(dif))
    sd = float(np.std(dif, ddof=1)) if n > 1 else 0.0
    cohen_d = float(md / sd) if sd > 0 else math.copysign(float('inf'), md) if md != 0 else 0.0
    # bootstrap CI for mean diff
    rng = np.random.RandomState(1234)
    bmeans = []
    for _ in range(2000):
        idx = rng.randint(0, n, size=n)
        bmeans.append(np.mean(dif[idx]))
    lo, hi = np.percentile(bmeans, [2.5, 97.5])
    return {"n": n, "mean_diff": md, "sd_diff": sd, "cohen_d": cohen_d, "ci95_mean_diff": [float(lo), float(hi)]}


def compute_recovery(clean_vals: np.ndarray, degraded_vals: np.ndarray, mitigated_vals: np.ndarray, higher_is_better: bool = True) -> Dict:
    """Compute recovery fraction and absolute improvement per paired run-level arrays."""
    clean = np.asarray(clean_vals, dtype=float)
    deg = np.asarray(degraded_vals, dtype=float)
    mit = np.asarray(mitigated_vals, dtype=float)
    mask = ~np.isnan(clean) & ~np.isnan(deg) & ~np.isnan(mit)
    clean = clean[mask]; deg = deg[mask]; mit = mit[mask]
    n = len(clean)
    if n == 0:
        return {"n": 0}
    # absolute improvement (mitigated - degraded) for higher-is-better, else (degraded - mitigated)
    if higher_is_better:
        abs_imp = np.mean(mit - deg)
        denom = np.mean(clean - deg)
        frac = float((np.mean(mit - deg) / denom) if denom != 0 else float('nan'))
    else:
        abs_imp = np.mean(deg - mit)
        denom = np.mean(deg - clean)
        frac = float((np.mean(deg - mit) / denom) if denom != 0 else float('nan'))
    return {"n": int(n), "abs_improvement": float(abs_imp), "recovery_fraction": float(frac), "denominator": float(denom)}


def run_analysis(out_dir: Path = OUT_DIR, fig_dir: Path = FIG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_canonical_summaries(require_n_runs=60)

    # identify metric columns
    possession_cols = [c for c in df.columns if c.startswith('possession_')]
    tactical_cols = [c for c in df.columns if c.startswith('tactical_')]
    tracking_cols = [c for c in df.columns if c.startswith('tracking_')]

    # Choose primary metrics
    primary_poss = 'possession_f1' if 'possession_f1' in possession_cols else (possession_cols[0] if possession_cols else None)
    primary_tactical = 'tactical_mean_absolute_score_error' if 'tactical_mean_absolute_score_error' in tactical_cols else (tactical_cols[0] if tactical_cols else None)
    primary_tracking = 'tracking_missingness' if 'tracking_missingness' in tracking_cols else (tracking_cols[0] if tracking_cols else None)

    summary = {}
    # Overall by condition
    for cond in ['CLEAN', 'SYNTHETIC-DEGRADED', 'NOISE-MITIGATED']:
        sub = df[df['condition'] == cond]
        summary[cond] = {}
        if primary_poss:
            summary[cond]['possession'] = describe_series(sub[primary_poss])
        if primary_tactical:
            summary[cond]['tactical'] = describe_series(sub[primary_tactical])
        if primary_tracking:
            summary[cond]['tracking'] = describe_series(sub[primary_tracking])

    # Recovery and paired effect sizes (degraded vs mitigated)
    # build wide tables per-run for numeric metrics
    poss_wide = df.pivot(index='run_id', columns='condition', values=primary_poss) if primary_poss else None
    tac_wide = df.pivot(index='run_id', columns='condition', values=primary_tactical) if primary_tactical else None

    def get_metric_array_from_wide(wide_df):
        if wide_df is None:
            return None, None, None
        try:
            return wide_df['CLEAN'].values, wide_df['SYNTHETIC-DEGRADED'].values, wide_df['NOISE-MITIGATED'].values
        except Exception:
            return None, None, None

    clean_arr, deg_arr, mit_arr = get_metric_array_from_wide(poss_wide) if primary_poss else (None, None, None)
    poss_recovery = None
    poss_effect = None
    if clean_arr is not None:
        poss_recovery = compute_recovery(clean_arr, deg_arr, mit_arr, higher_is_better=True)
        poss_effect = paired_effect_size(mit_arr, deg_arr)

    tac_clean, tac_deg, tac_mit = get_metric_array_from_wide(tac_wide) if primary_tactical else (None, None, None)
    tac_recovery = None
    tac_effect = None
    if tac_clean is not None:
        # tactical error: lower is better
        tac_recovery = compute_recovery(tac_clean, tac_deg, tac_mit, higher_is_better=False)
        tac_effect = paired_effect_size(tac_deg, tac_mit)

    report = {
        'n_runs': int(df['run_id'].nunique()),
        'n_conditions': int(df.shape[0]),
        'primary_metrics': {'possession_metric': primary_poss, 'tactical_metric': primary_tactical, 'tracking_metric': primary_tracking},
        'summary_by_condition': summary,
        'possession_recovery': poss_recovery,
        'possession_paired_effect': poss_effect,
        'tactical_recovery': tac_recovery,
        'tactical_paired_effect': tac_effect,
    }

    # Write outputs
    out_dir.joinpath('final_statistical_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # simple CSV summary table per condition
    rows = []
    for cond, d in summary.items():
        row = {'condition': cond}
        if 'possession' in d:
            row.update({f'possession_{k}': v for k, v in d['possession'].items()})
        if 'tactical' in d:
            row.update({f'tactical_{k}': v for k, v in d['tactical'].items()})
        if 'tracking' in d:
            row.update({f'tracking_{k}': v for k, v in d['tracking'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'final_statistical_summary.csv', index=False)

    # write an overview CSV of per-run primary metrics
    if primary_poss and poss_wide is not None:
        poss_out = pd.DataFrame({
            'run_id': poss_wide.index,
            'clean_poss': poss_wide['CLEAN'].values,
            'degraded_poss': poss_wide['SYNTHETIC-DEGRADED'].values,
            'mitigated_poss': poss_wide['NOISE-MITIGATED'].values,
        })
        poss_out.to_csv(out_dir / 'per_run_possession.csv', index=False)
    if primary_tactical and tac_wide is not None:
        tac_out = pd.DataFrame({
            'run_id': tac_wide.index,
            'clean_tac': tac_wide['CLEAN'].values,
            'degraded_tac': tac_wide['SYNTHETIC-DEGRADED'].values,
            'mitigated_tac': tac_wide['NOISE-MITIGATED'].values,
        })
        tac_out.to_csv(out_dir / 'per_run_tactical.csv', index=False)

    print('Final analysis written to', str(out_dir))
    return report


if __name__ == '__main__':
    run_analysis()
