"""
Unit tests for Missing-Track Interpolation Module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noise.interpolation import interpolate_trajectory


def _make_test_tracking_df(n_frames: int = 50, n_players: int = 2) -> pd.DataFrame:
    """Generate a clean synthetic canonical tracking DataFrame."""
    rows = []
    teams = ["home", "away"][:n_players]
    pids = [str(i + 1) for i in range(n_players)]

    for f in range(1, n_frames + 1):
        ts = f * 0.04
        for team, pid in zip(teams, pids):
            rows.append({
                "match_id": "test_match",
                "frame": f,
                "timestamp": ts,
                "player_id": pid,
                "team": team,
                "x": float(0.1 + 0.01 * f),
                "y": float(0.2 + 0.01 * f),
                "confidence": 1.0,
                "visible": True,
            })
    return pd.DataFrame(rows)


class TestInterpolation(unittest.TestCase):
    def test_linear_interpolation_correctness(self):
        df = _make_test_tracking_df(n_frames=20, n_players=1)
        # Create a gap of length 3 at frames 5, 6, 7 (indices for frames 1..20)
        # Frame 4 has x=0.14, Frame 8 has x=0.18
        mask_gap = df["frame"].isin([5, 6, 7])
        df.loc[mask_gap, "visible"] = False
        df.loc[mask_gap, ["x", "y"]] = np.nan
        df.loc[mask_gap, "confidence"] = 0.0

        res = interpolate_trajectory(df, max_gap=5)

        # Check filled values
        f5 = res[res["frame"] == 5].iloc[0]
        f6 = res[res["frame"] == 6].iloc[0]
        f7 = res[res["frame"] == 7].iloc[0]

        self.assertTrue(f5["visible"])
        self.assertTrue(f6["visible"])
        self.assertTrue(f7["visible"])

        self.assertAlmostEqual(f5["x"], 0.15, places=5)
        self.assertAlmostEqual(f6["x"], 0.16, places=5)
        self.assertAlmostEqual(f7["x"], 0.17, places=5)

        self.assertTrue(f5["imputed"])
        self.assertTrue(f6["imputed"])
        self.assertTrue(f7["imputed"])

    def test_max_gap_enforcement(self):
        df = _make_test_tracking_df(n_frames=30, n_players=1)
        # Create a short gap (length 3, frames 4,5,6) and a long gap (length 8, frames 12-19)
        df.loc[df["frame"].isin([4, 5, 6]), "visible"] = False
        df.loc[df["frame"].isin([4, 5, 6]), ["x", "y"]] = np.nan
        df.loc[df["frame"].isin(range(12, 20)), "visible"] = False
        df.loc[df["frame"].isin(range(12, 20)), ["x", "y"]] = np.nan

        res = interpolate_trajectory(df, max_gap=5)

        # Short gap must be interpolated
        short_filled = res[res["frame"].isin([4, 5, 6])]
        self.assertTrue(short_filled["visible"].all())
        self.assertTrue(short_filled["x"].notna().all())
        self.assertTrue(short_filled["imputed"].all())

        # Long gap must remain missing
        long_missing = res[res["frame"].isin(range(12, 20))]
        self.assertFalse(long_missing["visible"].any())
        self.assertTrue(long_missing["x"].isna().all())
        self.assertFalse(long_missing["imputed"].any())

    def test_boundary_gaps_not_extrapolated(self):
        df = _make_test_tracking_df(n_frames=20, n_players=1)
        # Leading gap (frames 1, 2) and trailing gap (frames 19, 20)
        df.loc[df["frame"].isin([1, 2, 19, 20]), "visible"] = False
        df.loc[df["frame"].isin([1, 2, 19, 20]), ["x", "y"]] = np.nan

        res = interpolate_trajectory(df, max_gap=5)

        leading = res[res["frame"].isin([1, 2])]
        trailing = res[res["frame"].isin([19, 20])]

        self.assertFalse(leading["visible"].any())
        self.assertTrue(leading["x"].isna().all())
        self.assertFalse(trailing["visible"].any())
        self.assertTrue(trailing["x"].isna().all())

    def test_observed_points_unchanged(self):
        df = _make_test_tracking_df(n_frames=20, n_players=1)
        df.loc[df["frame"].isin([5, 6]), "visible"] = False
        df.loc[df["frame"].isin([5, 6]), ["x", "y"]] = np.nan

        res = interpolate_trajectory(df, max_gap=5)
        obs_mask = df["visible"]

        pd.testing.assert_series_equal(res.loc[obs_mask, "x"], df.loc[obs_mask, "x"])
        pd.testing.assert_series_equal(res.loc[obs_mask, "y"], df.loc[obs_mask, "y"])
        self.assertFalse(res.loc[obs_mask, "imputed"].any())

    def test_per_player_independence(self):
        df = _make_test_tracking_df(n_frames=20, n_players=2)
        # Player 1 has gap at frames 4-6; Player 2 is fully visible
        p1_mask = (df["player_id"] == "1") & df["frame"].isin([4, 5, 6])
        df.loc[p1_mask, "visible"] = False
        df.loc[p1_mask, ["x", "y"]] = np.nan

        res = interpolate_trajectory(df, max_gap=5)

        # Player 1 gap interpolated
        self.assertTrue(res[(res["player_id"] == "1") & res["frame"].isin([4, 5, 6])]["visible"].all())
        # Player 2 points unaffected and unimputed
        p2_df = res[res["player_id"] == "2"]
        self.assertFalse(p2_df["imputed"].any())

    def test_source_immutability(self):
        df = _make_test_tracking_df(n_frames=20, n_players=1)
        copy_df = df.copy()
        _ = interpolate_trajectory(df, max_gap=5)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_invalid_parameters_raise(self):
        df = _make_test_tracking_df()
        with self.assertRaises(ValueError):
            interpolate_trajectory(df, max_gap=0)
        with self.assertRaises(ValueError):
            interpolate_trajectory(df.drop(columns=["frame"]), max_gap=5)


if __name__ == "__main__":
    unittest.main()
