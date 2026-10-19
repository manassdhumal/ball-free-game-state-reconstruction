import pandas as pd, numpy as np
OUT='results/metrics/final_analysis'
rec=pd.read_csv(OUT+'/mitigation_recovery.csv')
rpos=rec[rec['metric']=='possession_f1']
mean_rec=rpos['recovery_fraction'].astype(float).mean()
per=pd.read_csv(OUT+'/per_run_possession.csv').set_index('run_id')
deg=per['degraded_poss'].astype(float)
mit=per['mitigated_poss'].astype(float)
diff=(mit-deg).dropna()
mean_diff=diff.mean()
sd_diff=diff.std(ddof=1)
cohend=mean_diff/sd_diff if sd_diff>0 else float('nan')
import numpy as np
rng=np.random.RandomState(1234)
bs=[np.mean(rng.choice(diff.values,size=len(diff),replace=True)) for _ in range(2000)]
lo,hi=np.percentile(bs,[2.5,97.5])
print('mean_recovery_possession_f1=',mean_rec)
print('possession_mean_diff_mit_deg=',mean_diff,'sd_diff=',sd_diff,'cohen_d=',cohend,'ci95_mean_diff=(',lo,hi,')')
