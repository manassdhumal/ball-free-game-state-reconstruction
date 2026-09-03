import unittest
from pathlib import Path
import copy

from src.validation import robustness_experiment as rexp


class TestRunResumption(unittest.TestCase):
    def setUp(self):
        self.cfg = rexp.load_experiment_config('configs/robustness_experiment.yaml')
        self.grid = rexp.make_grid(self.cfg)

    def test_make_grid_length(self):
        self.assertEqual(len(self.grid), 60)

    def test_build_runspec_config_does_not_mutate_base(self):
        base = copy.deepcopy(self.cfg)
        runspec = self.grid[0]
        _ = rexp.build_runspec_config(runspec, base)
        # base['output'] should remain the same as original default
        self.assertEqual(base.get('output', {}).get('metrics_dir'), self.cfg.get('output', {}).get('metrics_dir'))

    def test_runspec_is_complete_detects_existing_run(self):
        # This repo already contains outputs for exp_0_60s_seed_72_sev_mild from pilot.
        runspec = next(r for r in self.grid if r.window_label == '0_60s' and r.seed == 72 and r.severity == 'mild')
        ok = rexp.runspec_is_complete(runspec, self.cfg)
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
