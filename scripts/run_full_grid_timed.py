import time
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src.validation import robustness_experiment as rexp
import argparse


def main():
    parser = argparse.ArgumentParser(description='Timed full-grid runner for Step 74')
    parser.add_argument('--config', default='configs/robustness_experiment.yaml')
    parser.add_argument('--smoke', action='store_true', help='Run only one smoke runspec')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results and skip completed runs')
    parser.add_argument('--force-rerun', action='store_true', help='Rerun runs even if already completed when --resume is used')
    args = parser.parse_args()

    cfg = rexp.load_experiment_config(args.config)
    grid = rexp.make_grid(cfg)

    out_report = {'start_time': time.time(), 'runs': []}
    report_path = Path('results') / 'runs' / 'robustness_expansion' / 'full_run_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # If resuming, load previously recorded runs to detect completed work
    completed_run_ids = set()
    if args.resume and report_path.exists():
        try:
            prev = json.loads(report_path.read_text(encoding='utf-8'))
            for r in prev.get('runs', []):
                if r.get('status') == 'ok' and r.get('run_id'):
                    completed_run_ids.add(r['run_id'])
            out_report = prev
        except Exception:
            # If report is malformed, continue from scratch but keep a backup
            backup = report_path.with_suffix('.bak.json')
            report_path.replace(backup)

    if args.smoke:
        grid = grid[:1]

    for runspec in grid:
        run_id = f"exp_{runspec.window_label}_seed_{runspec.seed}_sev_{runspec.severity}"
        meta = {'window': runspec.window_label, 'seed': runspec.seed, 'severity': runspec.severity, 'run_id': run_id}

        # Skip when resuming and run already completed, unless forced.
        if args.resume and not args.force_rerun and run_id in completed_run_ids:
            meta.update({'duration_s': 0.0, 'status': 'skipped', 'paths': {}})
            out_report['runs'].append(meta)
            report_path.write_text(json.dumps(out_report, indent=2, default=str), encoding='utf-8')
            print(f"Skipping already-completed runspec: {run_id}")
            continue

        start = time.perf_counter()
        try:
            res = rexp.run_single_runspec(runspec, cfg, root=ROOT)
            duration = time.perf_counter() - start
            meta.update({'duration_s': duration, 'status': 'ok', 'paths': {k: str(v) for k, v in (res.get('paths') or {}).items()}})
        except Exception as e:
            duration = time.perf_counter() - start
            meta.update({'duration_s': duration, 'status': 'failed', 'error': repr(e)})
        out_report['runs'].append(meta)
        # flush intermediate report
        report_path.write_text(json.dumps(out_report, indent=2, default=str), encoding='utf-8')
        print(f"Completed runspec: {meta['run_id']} status={meta['status']} duration_s={meta['duration_s']:.2f}")

    out_report['end_time'] = time.time()
    out_report['total_duration_s'] = out_report['end_time'] - out_report['start_time']
    report_path.write_text(json.dumps(out_report, indent=2, default=str), encoding='utf-8')
    print('Full grid run complete. Report written to', report_path)


if __name__ == '__main__':
    main()
