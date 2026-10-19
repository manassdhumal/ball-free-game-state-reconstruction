import json
from pathlib import Path
ROOT = Path('.').resolve()
PA = ROOT / 'scripts' / 'provenance_audit_output.json'
if not PA.exists():
    print('missing provenance_audit_output.json')
    raise SystemExit(1)
data = json.loads(PA.read_text(encoding='utf8'))
valid_ids = set(data.get('valid_ids', []))
full = data.get('full_run_report')
report_ids = []
failed = []
skipped = []
if isinstance(full, dict) and 'runs' in full:
    for r in full['runs']:
        rid = r.get('run_id')
        report_ids.append(rid)
        status = r.get('status')
        if status=='failed' or status=='error':
            failed.append(rid)
        if status=='skipped':
            skipped.append(rid)

report_set = set([x for x in report_ids if x])
intersection = sorted(list(report_set & valid_ids))
report_only = sorted(list(report_set - valid_ids))
filesystem_only = sorted(list(valid_ids - report_set))

out = {
    'report_run_count': len(report_ids),
    'report_only_ids': report_only,
    'filesystem_only_ids': filesystem_only,
    'intersection_count': len(intersection),
    'failed_ids': failed,
    'skipped_ids': skipped,
}
print(json.dumps(out, indent=2))
