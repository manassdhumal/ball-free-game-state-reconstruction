"""Tests for the task-driven downstream robustness benchmark."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.robustness_benchmark import (
    CONDITIONS,
    _summary_row,
    build_conditions,
    possession_change_metrics,
    run_inference_condition,
    save_outputs,
    tactical_stability_metrics,
    tracking_quality_metrics,
)


def make_tracking(n_frames=24):
    rows = []
    for frame in range(1, n_frames + 1):
        for team, offset in (("home", 0.25), ("away", 0.65)):
            for player in range(3):
                rows.append({
                    "match_id": "synthetic", "frame": frame, "timestamp": frame / 25,
                    "player_id": f"{team}_{player}", "team": team,
                    "x": offset + player * 0.035 + frame * 0.0008,
                    "y": 0.35 + player * 0.1 + (frame % 3) * 0.002,
                    "confidence": 1.0, "visible": True,
                })
    return pd.DataFrame(rows)


def benchmark_config():
    return {
        "orientation": {"home_attacks_x1": True},
        "degradation": {
            "severity": "moderate", "seed": 72,
            "enabled_degradations": ["random_missing", "coordinate_jitter"],
        },
        "interpolation": {"max_gap": 10, "confidence_decay": False, "imputed_confidence": 0.5},
        "smoothing": {"method": "savgol", "parameters": {"window_length": 7, "polyorder": 2}},
        "possession": {
            "density_radius": 0.05, "knn_k": 3, "switch_margin": 0.15,
            "persistence_window": 3, "min_score_threshold": 0.10, "dt": 0.04,
            "pitch_center": (0.5, 0.5),
        },
        "tracking_quality": {"large_motion_threshold": 0.05},
        "event_matching": {"tolerance_sec": 1.0},
        "tactical": {"top_k": 5},
        "output": {"run_id": "unit_test"},
    }


class TestBenchmarkConditions(unittest.TestCase):
    def setUp(self):
        self.clean = make_tracking()
        self.config = benchmark_config()

    def test_conditions_are_reproducible_and_labelled(self):
        first = build_conditions(self.clean, self.config)
        second = build_conditions(self.clean, self.config)
        self.assertEqual(tuple(first), CONDITIONS)
        for condition in CONDITIONS:
            pd.testing.assert_frame_equal(first[condition], second[condition])

    def test_clean_condition_remains_unchanged(self):
        conditions = build_conditions(self.clean, self.config)
        pd.testing.assert_frame_equal(self.clean, conditions["CLEAN"])

    def test_degradation_changes_tracking_and_mitigation_runs(self):
        conditions = build_conditions(self.clean, self.config)
        self.assertFalse(conditions["SYNTHETIC-DEGRADED"].equals(self.clean))
        self.assertEqual(len(conditions["NOISE-MITIGATED"]), len(self.clean))
        self.assertIn("imputed", conditions["NOISE-MITIGATED"].columns)

    def test_inference_is_ball_and_event_free(self):
        baseline = run_inference_condition(self.clean, self.config)
        decorated = self.clean.assign(Ball_X=0.2, Ball_Y=0.7, event_label="PASS")
        decorated_output = run_inference_condition(decorated, self.config)
        pd.testing.assert_frame_equal(baseline["possession"], decorated_output["possession"])
        pd.testing.assert_frame_equal(baseline["tactical"], decorated_output["tactical"])

    def test_end_to_end_outputs_and_sparse_tracking_are_valid(self):
        conditions = build_conditions(self.clean, self.config)
        output = run_inference_condition(conditions["NOISE-MITIGATED"], self.config)
        self.assertEqual(len(output["possession"]), self.clean["frame"].nunique())
        self.assertEqual(len(output["tactical"]), self.clean["frame"].nunique())
        sparse = self.clean.copy()
        sparse.loc[sparse["frame"].between(5, 15), ["visible", "confidence"]] = [False, 0.0]
        sparse.loc[sparse["frame"].between(5, 15), ["x", "y"]] = np.nan
        sparse_output = run_inference_condition(sparse, self.config)
        self.assertEqual(len(sparse_output["possession"]), self.clean["frame"].nunique())


class TestBenchmarkMetrics(unittest.TestCase):
    def test_tracking_metrics_are_deterministic(self):
        tracking = make_tracking()
        first = tracking_quality_metrics(tracking, expected_players=6)
        second = tracking_quality_metrics(tracking, expected_players=6)
        self.assertEqual(first, second)
        self.assertEqual(first["missingness_rate"], 0.0)
        self.assertEqual(first["coordinate_validity_rate"], 1.0)

    def test_tactical_score_differences_are_correct(self):
        clean = pd.DataFrame({
            "frame": [1, 2, 3], "timestamp": [0.04, 0.08, 0.12],
            "possessor_id": ["h", "h", "a"], "team": ["home", "home", "away"],
            "tactical_score": [0.1, 0.4, 0.8],
        })
        changed = clean.copy()
        changed["tactical_score"] = [0.2, 0.2, 0.8]
        metrics = tactical_stability_metrics(clean, changed, top_k=2)
        self.assertAlmostEqual(metrics["mean_absolute_score_error"], 0.1)
        self.assertAlmostEqual(metrics["median_absolute_score_error"], 0.1)
        self.assertEqual(metrics["top_k"], 2)

    def test_possession_matching_is_deterministic(self):
        candidates = pd.DataFrame({"frame": [100, 150], "event_type": ["pass_candidate", "turnover_candidate"]})
        events = pd.DataFrame({"start_frame": [101, 149], "event_type": ["PASS", "BALL LOST"]})
        first = possession_change_metrics(candidates, events, fps=25.0, tolerance_sec=0.5)
        second = possession_change_metrics(candidates, events, fps=25.0, tolerance_sec=0.5)
        self.assertEqual(first, second)
        self.assertEqual(first["f1"], 1.0)

    def test_condition_summary_and_machine_outputs(self):
        quality = tracking_quality_metrics(make_tracking(), expected_players=6)
        possession = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        tactical = {"mean_absolute_score_error": 0.0, "median_absolute_score_error": 0.0}
        summary = pd.DataFrame([_summary_row("CLEAN", quality, possession, tactical)])
        scores = pd.DataFrame({"frame": [1], "timestamp": [0.04], "possessor_id": ["h"], "team": ["home"], "tactical_score": [0.5]})
        result = {"summary_table": summary, "condition_summaries": {"CLEAN": {"condition": "CLEAN"}}, "tactical_scores": {condition: scores for condition in CONDITIONS}}
        with tempfile.TemporaryDirectory() as directory:
            config = benchmark_config()
            config["output"].update({"metrics_dir": "metrics", "runs_dir": "runs", "figures_dir": "figures"})
            files = save_outputs(result, config, Path(directory))
            self.assertTrue(files["summary_csv"].exists())
            self.assertTrue(files["summary_json"].exists())
            self.assertTrue(files["config_json"].exists())
            self.assertTrue(files["tactical_csv"].exists())
            self.assertTrue(files["figure"].exists())


if __name__ == "__main__":
    unittest.main()


