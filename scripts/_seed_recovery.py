import pandas as pd
import re
from pathlib import Path
rec = pd.read_csv('results/metrics/final_analysis/mitigation_recovery.csv')
rec['seed'] = rec['run_id'].apply(lambda x: re.search(r'_seed_([0-9]+)_', x).group(1) if re.search(r'_seed_([0-9]+)_', x) else 'na')
seeds = rec.groupby('seed')['recovery_fraction'].agg(['mean','std','count']).reset_index()
print(seeds.to_string(index=False))
Path('results/metrics/final_analysis/mitigation_recovery_by_seed.csv').write_text(seeds.to_csv(index=False),encoding='utf-8')
print('wrote mitigation_recovery_by_seed.csv')
