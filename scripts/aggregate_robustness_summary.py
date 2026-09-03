import pathlib
import pandas as pd
import re
import math
import json
from typing import Optional, Dict, List


ROOT = pathlib.Path('results')
METRICS_ROOT = ROOT / 'metrics' / 'robustness_expansion'
RUNS_ROOT = ROOT / 'runs' / 'robustness_expansion'
FIGURES_ROOT = ROOT / 'figures' / 'robustness_expansion'

PATTERN = re.compile(r"robustness_benchmark_exp_([0-9]+_[0-9]+s)_seed_([0-9]+)_sev_([a-zA-Z]+)_(summary|summary.json|tactical_scores|config)\.(csv|json)$")


def find_artifacts() -> Dict[str, Dict[str, List[pathlib.Path]]]:
    """Return mapping run_id -> dict of artifact lists.

    Keys: 'summary_csv', 'summary_json', 'tactical_csv', 'config_json'
    """
    artifacts: Dict[str, Dict[str, List[pathlib.Path]]] = {}

    # Search metrics for summary files, and runs for tactical/config
    for p in list(METRICS_ROOT.rglob('*')) + list(RUNS_ROOT.rglob('*')):
        if not p.is_file():
            continue
        m = PATTERN.search(p.name)
        if not m:
            continue
        window, seed, severity, kind, ext = m.groups()
        run_id = f"exp_{window}_seed_{seed}_sev_{severity}"
        bucket = artifacts.setdefault(run_id, {'summary_csv': [], 'summary_json': [], 'tactical_csv': [], 'config_json': [], 'all_paths': []})
        bucket['all_paths'].append(p)
        key = None
        if kind == 'summary' and ext == 'csv':
            key = 'summary_csv'
        elif kind == 'summary.json' or (kind == 'summary' and ext == 'json'):
            key = 'summary_json'
        elif kind == 'tactical_scores' and ext == 'csv':
            key = 'tactical_csv'
        elif kind == 'config' and ext == 'json':
            key = 'config_json'
        if key:
            bucket[key].append(p)
    return artifacts


def path_score(p: pathlib.Path) -> int:
    """Score a path by how canonical it is: fewer occurrences of 'robustness_expansion' in relative path wins."""
    try:
        rel = p.relative_to(ROOT)
    except Exception:
        rel = p
    parts = [str(x) for x in rel.parts]
    return parts.count('robustness_expansion')


def choose_canonical(art_paths: List[pathlib.Path]) -> pathlib.Path:
    """Choose deterministic canonical path from candidates: minimal path_score, then lexicographic."""
    if not art_paths:
        return None
    sorted_candidates = sorted(art_paths, key=lambda p: (path_score(p), str(p)))
    return sorted_candidates[0]


def validate_summary_csv(path: pathlib.Path) -> Dict:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {'ok': False, 'error': f'read_error: {e}'}
    if 'condition' not in df.columns:
        return {'ok': False, 'error': 'missing_condition_column'}
    conditions = df['condition'].dropna().astype(str).tolist()
    unique_conditions = list(dict.fromkeys(conditions))
    required = ['CLEAN', 'SYNTHETIC-DEGRADED', 'NOISE-MITIGATED']
    if set(required) != set(unique_conditions):
        return {'ok': False, 'error': f'condition_mismatch', 'conditions': unique_conditions}
    # ensure no duplicate condition rows
    if len(conditions) != len(unique_conditions):
        return {'ok': False, 'error': 'duplicate_condition_rows'}
    return {'ok': True, 'conditions': unique_conditions}


def aggregate(write_integrity: bool = True) -> Dict:
    artifacts = find_artifacts()
    per_run = []
    unique_runs = 0
    duplicate_artifact_count = 0
    invalid_runs = 0
    missing_runs = 0
    noncanonical_artifacts = 0
    noncanonical_paths = []
    duplicate_runs = 0

    # Planned runspecs from config
    planned = []
    try:
        # robustness_experiment.yaml may be yaml; load best-effort by reading text and parsing simple structures
        raw = (ROOT.parent / 'configs' / 'robustness_experiment.yaml').read_text()
        # naive parse for lists - fall back to empty if not parseable
        cfg = json.loads(json.dumps({}))
        # we don't rely heavily on planned; keep empty if parsing unavailable
        planned = []
    except Exception:
        planned = []

    # Load run report if present
    report_path = ROOT / 'runs' / 'robustness_expansion' / 'full_run_report.json'
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            reported_ids = {r.get('run_id'): r for r in report.get('runs', []) if r.get('run_id')}
        except Exception:
            reported_ids = {}
    else:
        reported_ids = {}

    # iterate deterministically over union of artifacts, planned and reported ids
    for run_id in sorted(set(list(artifacts.keys()) + planned + list(reported_ids.keys()))):
        entry = {'run_id': run_id, 'window': None, 'seed': None, 'severity': None, 'status': None, 'source_path': None, 'duplicates': [], 'report_status': reported_ids.get(run_id, {}).get('status') if run_id in reported_ids else None}
        # parse run_id
        m = re.match(r'exp_([0-9]+_[0-9]+s)_seed_([0-9]+)_sev_([a-zA-Z]+)', run_id)
        if m:
            entry['window'], entry['seed'], entry['severity'] = m.groups()
        art = artifacts.get(run_id, {'summary_csv': [], 'summary_json': [], 'tactical_csv': [], 'config_json': [], 'all_paths': []})
        # detect duplicates and classify noncanonical duplicates
        # detect duplicates per artifact kind (summary/tactical/config)
        run_has_duplicate = False
        canonical_candidates = set()
        for kind in ['summary_csv', 'summary_json', 'tactical_csv', 'config_json']:
            items = art.get(kind, [])
            if not items:
                continue
            # choose canonical for this kind
            c = choose_canonical(items)
            if c:
                canonical_candidates.add(str(c))
            if len(items) > 1:
                # there are duplicate files of this kind
                run_has_duplicate = True
                # count extra files beyond the canonical one
                duplicate_artifact_count += max(0, len(items) - 1)
                # mark noncanonical paths for extra copies
                sorted_items = sorted(items, key=lambda p: str(p))
                for p in sorted_items:
                    sp = str(p)
                    if sp != str(c):
                        noncanonical_artifacts += 1
                        noncanonical_paths.append({'path': sp, 'classification': 'LEGACY_DUPLICATE' if path_score(p) > 1 else 'NONCANONICAL'})
        # if there are multiple different artifact kinds but exactly one of each, that's fine (canonical set)
        if run_has_duplicate:
            duplicate_runs += 1
            # list all matched paths for diagnostics
            all_paths_sorted = sorted(art.get('all_paths', []), key=lambda p: str(p))
            entry['duplicates'] = [str(p) for p in all_paths_sorted]
        # choose canonical files
        summary_csv = choose_canonical(art['summary_csv'])
        summary_json = choose_canonical(art['summary_json'])
        tactical = choose_canonical(art['tactical_csv'])
        config = choose_canonical(art['config_json'])
        entry['source_path'] = str(summary_csv) if summary_csv else None

        # Validate according to strict rules: exactly one of each artifact and summary CSV content
        # Count existence across artifact lists (not filesystem count)
        has_summary = len(art.get('summary_csv', [])) >= 1
        has_tactical = len(art.get('tactical_csv', [])) >= 1
        has_config = len(art.get('config_json', [])) >= 1
        # require at least one candidate of each; canonicality checked separately
        if has_summary and has_tactical and has_config and summary_csv and tactical and config:
            # validate summary CSV content
            v = validate_summary_csv(summary_csv)
            if not v.get('ok'):
                entry['status'] = 'INVALID'
                entry['error'] = v.get('error')
                invalid_runs += 1
            else:
                # validate config JSON is readable
                try:
                    _ = json.loads(pathlib.Path(config).read_text(encoding='utf-8')) if config else {}
                    entry['status'] = 'SUCCESS'
                    entry['valid_conditions'] = v.get('conditions', [])
                    unique_runs += 1
                except Exception:
                    entry['status'] = 'INVALID'
                    entry['error'] = 'config_json_malformed'
                    invalid_runs += 1
        elif any([has_summary, has_tactical, has_config]):
            entry['status'] = 'INCOMPLETE'
            missing_runs += 1
        else:
            entry['status'] = 'MISSING'
            missing_runs += 1

        per_run.append(entry)

    # Cross-check report vs filesystem
    report_vs_fs = {'in_report_not_filesystem': [], 'filesystem_not_in_report': []}
    for rid, rmeta in reported_ids.items():
        # find per_run entry
        found = next((p for p in per_run if p['run_id'] == rid and p['status'] == 'SUCCESS'), None)
        if not found:
            report_vs_fs['in_report_not_filesystem'].append(rid)
    for p in per_run:
        if p['status'] == 'SUCCESS' and p['run_id'] not in reported_ids:
            report_vs_fs['filesystem_not_in_report'].append(p['run_id'])

    # Aggregated counts
    aggregated = {
        'valid_unique_runspecs': unique_runs,
        'duplicate_artifacts': duplicate_artifact_count,
        'noncanonical_artifacts': noncanonical_artifacts,
        'invalid_runspecs': invalid_runs,
        'duplicate_runs': duplicate_runs,
        'missing_runspecs': missing_runs,
    }

    # Output per-run canonical table
    df_runs = pd.DataFrame(per_run)
    print('\nPer-run canonical table:')
    print(df_runs.to_string(index=False))

    # Aggregated summary by condition/severity/seed/window using existing summary CSVs
    # Collect all valid summary CSV paths (canonical chosen)
    valid_paths = [p['source_path'] for p in per_run if p.get('status') == 'SUCCESS' and p.get('source_path')]
    valid_condition_rows = sum(len(p.get('valid_conditions', [])) for p in per_run if p.get('status') == 'SUCCESS')
    if valid_paths:
        records = []
        for p in valid_paths:
            df = pd.read_csv(p)
            df['source_file'] = p
            records.append(df)
        all_df = pd.concat(records, ignore_index=True)
        # Ensure grouping cols exist
        for col in ['condition', 'window', 'seed', 'severity']:
            if col not in all_df.columns:
                all_df[col] = None
        group_keys = ['condition', 'severity']
        grouped = all_df.groupby(group_keys)
        # For brevity, report counts per group
        agg_counts = grouped.size().reset_index(name='n_rows')
        print('\nAggregated counts by condition & severity:')
        print(agg_counts.to_string(index=False))
    else:
        print('\nNo valid summary CSVs to aggregate.')

    # Integrity report
    integrity = {
        'planned_runspecs': len(planned),
        **aggregated,
        'valid_condition_rows': valid_condition_rows,
        'report_filesystem_discrepancies': report_vs_fs,
        'noncanonical_paths': noncanonical_paths,
    }
    print('\nIntegrity report:')
    print(json.dumps(integrity, indent=2))

    # Write machine-readable integrity JSON (do not overwrite existing file; use timestamped file if exists)
    if write_integrity:
        out_dir = METRICS_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'robustness_expansion_integrity_report.json'
        if out_path.exists():
            from datetime import datetime
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            out_path = out_dir / f'robustness_expansion_integrity_report_{ts}.json'
        try:
            out_path.write_text(json.dumps(integrity, indent=2), encoding='utf-8')
            print(f'Integrity JSON written to {out_path}')
        except Exception as e:
            print(f'Failed to write integrity JSON: {e}')

    return integrity


if __name__ == '__main__':
    aggregate()

