import unittest
import pandas as pd
import numpy as np
from src.validation.robustness_experiment import load_experiment_config, make_grid, run_single_runspec
from src.validation.robustness_benchmark import load_metrica_subset, build_conditions
from src.noise.degradation import degrade_tracking, SEVERITY_CONFIGS
from src.noise.interpolation import interpolate_trajectory
from src.noise.smoothing import smooth_trajectory
from src.validation.robustness_benchmark import _TRACKING_COLUMNS


class TestRobustnessAudit(unittest.TestCase):
    def setUp(self):
        self.cfg = load_experiment_config("configs/robustness_experiment.yaml")
        self.grid = make_grid(self.cfg)

    def test_grid_size_and_parameters(self):
        # Expect 4 windows x 5 seeds x 3 severities = 60
        self.assertEqual(len(self.grid), 60)
        seeds = sorted({r.seed for r in self.grid})
        self.assertEqual(seeds, [72, 73, 74, 75, 76])
        severities = sorted({r.severity for r in self.grid})
        self.assertEqual(severities, ["mild", "moderate", "severe"])

    def test_severity_and_seed_effects(self):
        # Load a single tracking subset for the first runspec
        spec = self.grid[0]
        base_cfg = dict(self.cfg)
        base_cfg['match'] = dict(base_cfg['match'])
        base_cfg['match']['start_frame'] = spec.start_frame
        base_cfg['match']['end_frame'] = spec.end_frame
        tracking, events = load_metrica_subset(base_cfg)

        # Degrade with same seed twice -> identical
        d1, m1 = degrade_tracking(tracking, severity='moderate', seed=123)
        d2, m2 = degrade_tracking(tracking, severity='moderate', seed=123)
        pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))

        # Different seeds -> likely different
        d3, m3 = degrade_tracking(tracking, severity='moderate', seed=124)
        # It's acceptable for very low-prob events to coincide; assert at least one value differs
        self.assertFalse(d1.equals(d3))

        # Different severities should produce different parameters
        self.assertIn('mild', SEVERITY_CONFIGS)
        self.assertIn('severe', SEVERITY_CONFIGS)
        self.assertNotEqual(SEVERITY_CONFIGS['mild']['missing_prob'], SEVERITY_CONFIGS['severe']['missing_prob'])

    def test_clean_invariance_and_mitigation_pipeline(self):
        spec = self.grid[0]
        base_cfg = dict(self.cfg)
        base_cfg['match'] = dict(base_cfg['match'])
        base_cfg['match']['start_frame'] = spec.start_frame
        base_cfg['match']['end_frame'] = spec.end_frame
        tracking, events = load_metrica_subset(base_cfg)

        conditions = build_conditions(tracking, base_cfg)
        # CLEAN must equal the subset tracking (no mutation)
        pd.testing.assert_frame_equal(conditions['CLEAN'].reset_index(drop=True), tracking.reset_index(drop=True))

        # NOISE-MITIGATED must equal smoothing(interpolate(degraded))
        degraded = conditions['SYNTHETIC-DEGRADED']
        interp = interpolate_trajectory(degraded, max_gap=int(base_cfg['interpolation']['max_gap']), confidence_decay=bool(base_cfg['interpolation'].get('confidence_decay', False)), imputed_confidence=float(base_cfg['interpolation'].get('imputed_confidence', 0.5)))
        smooth = smooth_trajectory(interp, method=base_cfg['smoothing']['method'], **dict(base_cfg['smoothing'].get('parameters', {})))
        pd.testing.assert_frame_equal(smooth.reset_index(drop=True), conditions['NOISE-MITIGATED'].reset_index(drop=True))

    def test_window_isolation_and_no_leakage(self):
        # Ensure windows do not include frames outside their specified bounds
        spec = self.grid[1]  # second window
        base_cfg = dict(self.cfg)
        base_cfg['match'] = dict(base_cfg['match'])
        base_cfg['match']['start_frame'] = spec.start_frame
        base_cfg['match']['end_frame'] = spec.end_frame
        tracking, events = load_metrica_subset(base_cfg)
        self.assertGreaterEqual(tracking['frame'].min(), spec.start_frame)
        self.assertLessEqual(tracking['frame'].max(), spec.end_frame)

        # Anti-leakage: adding ball columns should not change inference outputs
        conditions = build_conditions(tracking, base_cfg)
        clean = conditions['CLEAN']
        clean_with_ball = clean.copy()
        clean_with_ball['Ball_X'] = 0.5
        clean_with_ball['Ball_Y'] = 0.5

        from src.validation.robustness_benchmark import run_inference_condition
        out1 = run_inference_condition(clean, base_cfg)
        out2 = run_inference_condition(clean_with_ball, base_cfg)
        # Tactical scores should be identical
        pd.testing.assert_frame_equal(out1['tactical'].reset_index(drop=True), out2['tactical'].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
