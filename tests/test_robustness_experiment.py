import unittest
import tempfile
from pathlib import Path

from src.validation.robustness_experiment import load_experiment_config, make_grid, run_single_runspec

class TestRobustnessExperiment(unittest.TestCase):
    def test_load_and_grid(self):
        cfg = load_experiment_config("configs/robustness_experiment.yaml")
        grid = make_grid(cfg)
        self.assertGreater(len(grid), 0)

    def test_runspec_smoke(self):
        cfg = load_experiment_config("configs/robustness_experiment.yaml")
        grid = make_grid(cfg)
        spec = grid[0]
        # Run the single runspec smoke invocation (writes compact outputs)
        out = run_single_runspec(spec, cfg)
        self.assertIn("summary_table", out)
        # summary table should include rows for the three conditions
        st = out["summary_table"]
        conditions = set(st["condition"].tolist())
        self.assertTrue({"CLEAN", "SYNTHETIC-DEGRADED", "NOISE-MITIGATED"}.issubset(conditions))

if __name__ == '__main__':
    unittest.main()
