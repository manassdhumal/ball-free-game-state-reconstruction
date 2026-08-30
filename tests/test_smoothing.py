"""
Unit tests for Trajectory Smoothing Module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noise.smoothing import (
    smooth_moving_average,
    smooth_savitzky_golay,
    smooth_trajectory,
)


def _make_noisy_line_df(n_frames: int = 50, n_players: int = 2) -> pd.DataFrame:
    """Generate canonical tracking DataFrame with linear motion + Gaussian noise."""
    rng = np.random.default_rng(42)
    rows = []
    teams = ["home", "away"][:n_players]
    pids = [str(i + 1) for i in range(n_players)]

    for f in range(1, n_frames + 1):
        ts = f * 0.04
        for team, pid in zip(teams, pids):
            true_x = 0.1 + 0.01 * f
            true_y = 0.2 + 0.005 * f
            noise_x = rng.normal(0, 0.02)
            noise_y = rng.normal(0, 0.02)
            rows.append({
                "match_id": "test_match",
                "frame": f,
                "timestamp": ts,
                "player_id": pid,
                "team": team,
                "x": float(true_x + noise_x),
                "y": float(true_y + noise_y),
                "confidence": 1.0,
                "visible": True,
            })
    return pd.DataFrame(rows)


class TestSmoothing(unittest.TestCase):
    def test_moving_average_noise_reduction(self):
        df = _make_noisy_line_df(n_frames=60, n_players=1)
        res = smooth_moving_average(df, window_size=5)

        # Variance of second differences (jitter) should decrease
        raw_diff2 = np.diff(np.diff(df["x"].values))
        smooth_diff2 = np.diff(np.diff(res["x"].values))
        self.assertLess(np.var(smooth_diff2), np.var(raw_diff2))

    def test_savitzky_golay_noise_reduction(self):
        df = _make_noisy_line_df(n_frames=60, n_players=1)
        res = smooth_savitzky_golay(df, window_length=7, polyorder=2)

        raw_diff2 = np.diff(np.diff(df["x"].values))
        smooth_diff2 = np.diff(np.diff(res["x"].values))
        self.assertLess(np.var(smooth_diff2), np.var(raw_diff2))

    def test_missing_values_preserved_not_smoothed_across_gaps(self):
        df = _make_noisy_line_df(n_frames=40, n_players=1)
        # Create gap at frames 15-25
        gap_mask = df["frame"].isin(range(15, 26))
        df.loc[gap_mask, "visible"] = False
        df.loc[gap_mask, ["x", "y"]] = np.nan
        df.loc[gap_mask, "confidence"] = 0.0

        res = smooth_savitzky_golay(df, window_length=7, polyorder=2)

        # Gap rows must remain NaN and invisible (no fabrication)
        self.assertTrue(res.loc[gap_mask, "x"].isna().all())
        self.assertTrue(res.loc[gap_mask, "y"].isna().all())
        self.assertFalse(res.loc[gap_mask, "visible"].any())

        # Visible segments before and after should be smoothed
        vis_mask = df["visible"]
        self.assertTrue(res.loc[vis_mask, "x"].notna().all())

    def test_per_player_independence(self):
        df = _make_noisy_line_df(n_frames=40, n_players=2)
        res = smooth_trajectory(df, method="moving_average", window_size=5)

        # Check that player 1 smoothing does not depend on player 2
        p1_df = df[df["player_id"] == "1"].copy()
        p1_res = smooth_trajectory(p1_df, method="moving_average", window_size=5)

        pd.testing.assert_series_equal(
            res[res["player_id"] == "1"]["x"],
            p1_res["x"],
        )

    def test_source_immutability(self):
        df = _make_noisy_line_df(n_frames=30, n_players=1)
        copy_df = df.copy()
        _ = smooth_savitzky_golay(df, window_length=7, polyorder=2)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_parameter_validation_raises(self):
        df = _make_noisy_line_df(n_frames=30, n_players=1)

        # Even window_length
        with self.assertRaises(ValueError):
            smooth_savitzky_golay(df, window_length=6, polyorder=2)

        # Window_length < 3
        with self.assertRaises(ValueError):
            smooth_savitzky_golay(df, window_length=1, polyorder=2)

        # Polyorder >= window_length
        with self.assertRaises(ValueError):
            smooth_savitzky_golay(df, window_length=5, polyorder=5)

        # Moving average window < 1
        with self.assertRaises(ValueError):
            smooth_moving_average(df, window_size=0)

        # Unsupported method
        with self.assertRaises(ValueError):
            smooth_trajectory(df, method="kalman_unsupported")


if __name__ == "__main__":
    unittest.main()
