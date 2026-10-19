from pathlib import Path
import copy
import json

from src.validation import robustness_experiment as rexp

# Load base config
cfg = rexp.load_experiment_config('configs/robustness_experiment.yaml')
# Deepcopy and redirect outputs to a safe temporary folder
test_root = Path('results_tmp_smoke')
cfg2 = copy.deepcopy(cfg)
cfg2.setdefault('output', {})
cfg2['output']['metrics_dir'] = str(test_root / 'metrics')
cfg2['output']['runs_dir'] = str(test_root / 'runs')
cfg2['output']['figures_dir'] = str(test_root / 'figures')

# Pick the first runspec (0-60s, seed 72, mild)
grid = rexp.make_grid(cfg)
runspec = grid[0]
print('Running smoke safe for', runspec)
res = rexp.run_single_runspec(runspec, cfg2, root=Path.cwd())
print('Summary conditions:', set(res['summary_table']['condition'].tolist()))
print('Paths:')
for k,v in (res.get('paths') or {}).items():
    print(k, v)
# Write a small local report
report = {'runs': [{'run_id': res['config']['output']['run_id'], 'paths': {k: str(v) for k, v in (res.get('paths') or {}).items()}}], 'root': str(test_root)}
Path('results_tmp_smoke').mkdir(parents=True, exist_ok=True)
with open('results_tmp_smoke/smoke_report.json','w',encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print('Smoke-safe report written to results_tmp_smoke/smoke_report.json')
