import json
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts import aggregate_robustness_summary as agg
artifacts = agg.find_artifacts()
valid = []
for run_id, art in sorted(artifacts.items()):
    summary = agg.choose_canonical(art.get('summary_csv', []))
    tactical = agg.choose_canonical(art.get('tactical_csv', []))
    config = agg.choose_canonical(art.get('config_json', []))
    if summary and tactical and config:
        v = agg.validate_summary_csv(summary)
        try:
            _ = json.loads(open(config, 'r', encoding='utf-8').read())
        except Exception:
            v = {'ok': False}
        if v.get('ok'):
            valid.append(run_id)
print(json.dumps({'valid_count': len(valid), 'valid_runs': valid}, indent=2))
