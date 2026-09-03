from pathlib import Path
import json
import copy

from src.validation import robustness_experiment as rexp

cfg = rexp.load_experiment_config('configs/robustness_experiment.yaml')
grid = rexp.make_grid(cfg)
missing = None
for r in grid:
    if not rexp.runspec_is_complete(r, cfg):
        missing = r
        break
if missing is None:
    print('No missing runspec found')
    raise SystemExit(0)
print('Selected missing runspec:', missing)
# Run only that runspec
res = rexp.run_single_runspec(missing, cfg, root=Path.cwd())
print('Completed missing runspec, status paths:')
for k,v in (res.get('paths') or {}).items():
    print(k, v)
# Append to full_run_report.json safely
report_path = Path('results') / 'runs' / 'robustness_expansion' / 'full_run_report.json'
if report_path.exists():
    rep = json.loads(report_path.read_text(encoding='utf-8'))
else:
    rep = {'start_time': None, 'runs': []}
meta = {'window': missing.window_label, 'seed': missing.seed, 'severity': missing.severity, 'run_id': res['config']['output']['run_id'], 'duration_s': 0.0, 'status': 'ok', 'paths': {k: str(v) for k, v in (res.get('paths') or {}).items()}}
rep['runs'].append(meta)
report_path.write_text(json.dumps(rep, indent=2, default=str), encoding='utf-8')
print('Appended run to report')
