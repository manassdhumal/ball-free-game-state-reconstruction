"""Focused tests for the isolated Step 80 broadcast-noise analysis."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.step80_analysis import (
    METHODS,
    apply_mitigation,
    causal_kalman_filter,
    detect_candidate_cuts,
    step80_tracking_metrics,
)
from src.possession.possession_baseline import predict_possession
from src.validation.robustness_benchmark import run_inference_condition, tactical_scores


def make_tracking(frames=8, players=4):
    rows = []
    identities = [("home", "h1"), ("home", "h2"), ("away", "a1"), ("away", "a2")][:players]
    for frame in range(1, frames + 1):
        for index, (team, player_id) in enumerate(identities):
            rows.append({
                "match_id": "test", "frame": frame, "timestamp": frame / 25,
                "team": team, "player_id": player_id, "x": 0.1 + index * 0.1 + frame * 0.001,
                "y": 0.2 + index * 0.05, "confidence": 1.0, "visible": True,
            })
    return pd.DataFrame(rows)


class TestStep80Detector(unittest.TestCase):
    def test_simultaneous_discontinuity_is_candidate(self):
        tracking = make_tracking()
        tracking.loc[tracking["frame"] == 5, ["x", "y"]] += 0.2
        signal = detect_candidate_cuts(tracking)
        self.assertTrue(bool(signal.loc[signal["frame"] == 5, "candidate_cut"].iloc[0]))

    def test_isolated_anomaly_is_not_candidate(self):
        tracking = make_tracking()
        mask = (tracking["frame"] == 5) & (tracking["player_id"] == "h1")
        tracking.loc[mask, ["x", "y"]] += 0.2
        signal = detect_candidate_cuts(tracking)
        self.assertFalse(bool(signal.loc[signal["frame"] == 5, "candidate_cut"].iloc[0]))

    def test_no_discontinuity_and_first_frame_boundary(self):
        signal = detect_candidate_cuts(make_tracking())
        self.assertFalse(signal["candidate_cut"].any())
        self.assertNotIn(1, signal.loc[signal["candidate_cut"], "frame"].tolist())

    def test_non_contiguous_frames_are_not_compared_as_adjacent(self):
        tracking = make_tracking(frames=4)
        tracking = tracking[tracking["frame"] != 3].copy()
        signal = detect_candidate_cuts(tracking)
        row = signal.loc[signal["frame"] == 4].iloc[0]
        self.assertTrue(pd.isna(row["frame_gap"]))
        self.assertFalse(bool(row["candidate_cut"]))


class TestStep80Mitigation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "interpolation": {"max_gap": 2, "imputed_confidence": 0.5},
            "smoothing": {"moving_average_window": 3, "savgol_window": 5, "savgol_polyorder": 2},
        }

    def test_all_methods_preserve_schema_and_row_count(self):
        tracking = make_tracking()
        for method in METHODS:
            result = apply_mitigation(tracking, method, self.config)
            self.assertEqual(len(result), len(tracking))
            self.assertTrue({"frame", "player_id", "x", "y", "visible"}.issubset(result.columns))

    def test_long_gap_remains_unresolved(self):
        tracking = make_tracking(frames=10, players=1)
        mask = tracking["frame"].isin([4, 5, 6, 7])
        tracking.loc[mask, ["x", "y"]] = np.nan
        tracking.loc[mask, "visible"] = False
        result = apply_mitigation(tracking, "interpolation", self.config)
        self.assertFalse(result.loc[mask, "visible"].any())

    def test_kalman_is_causal_and_does_not_fill_gaps(self):
        tracking = make_tracking(frames=6, players=1)
        tracking.loc[tracking["frame"] == 3, ["x", "y"]] = np.nan
        tracking.loc[tracking["frame"] == 3, "visible"] = False
        filtered = causal_kalman_filter(tracking)
        self.assertTrue(pd.isna(filtered.loc[filtered["frame"] == 3, "x"]).all())
        altered = tracking.copy()
        altered.loc[altered["frame"] > 3, "x"] += 0.5
        first = causal_kalman_filter(tracking)
        second = causal_kalman_filter(altered)
        self.assertAlmostEqual(first.loc[first["frame"] == 2, "x"].iloc[0], second.loc[second["frame"] == 2, "x"].iloc[0])


class TestStep80Metrics(unittest.TestCase):
    def test_metrics_include_required_tracking_fields(self):
        tracking = make_tracking()
        signal = detect_candidate_cuts(tracking)
        metrics = step80_tracking_metrics(tracking, signal, expected_players=4)
        for key in ("missingness_rate", "unresolved_gap_count", "mean_gap_length", "max_gap_length", "trajectory_continuity", "velocity_stability", "candidate_cut_count"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["candidate_cut_count"], 0)

    def test_cached_inference_matches_default_outputs(self):
        tracking = make_tracking(frames=6)
        config = {
            "possession": {"density_radius": 0.05, "knn_k": 2, "switch_margin": 0.15,
                           "persistence_window": 5, "min_score_threshold": 0.10,
                           "dt": 0.04, "pitch_center": (0.5, 0.5)},
            "orientation": {"home_attacks_x1": True},
        }
        uncached_possession = predict_possession(tracking, **config["possession"])
        uncached_tactical = tactical_scores(
            tracking, uncached_possession, True,
            config["possession"]["density_radius"], config["possession"]["knn_k"],
        )
        cached = run_inference_condition(tracking, config)
        pd.testing.assert_frame_equal(uncached_possession, cached["possession"])
        pd.testing.assert_frame_equal(uncached_tactical, cached["tactical"])


if __name__ == "__main__":
    unittest.main()