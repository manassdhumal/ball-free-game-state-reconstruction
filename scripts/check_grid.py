from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.validation import robustness_experiment as rexp
cfg = rexp.load_experiment_config('configs/robustness_experiment.yaml')
grid = rexp.make_grid(cfg)
print('grid_len=', len(grid))
for i,g in enumerate(grid[:8]):
    print(i, g.window_label, g.seed, g.severity)
