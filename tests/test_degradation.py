"""
Unit tests for the Synthetic Tracking Degradation Framework.

Tests all 12 requirements using small synthetic DataFrames for deterministic,
fast execution without requiring raw dataset files.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noise.degradation import (
    ALL_DEGRADATION_NAMES,
    DEGRADATION_HIERARCHY,
    SEVERITY_CONFIGS,
    apply_contiguous_gaps,
    apply_coordinate_jitter,
    apply_identity_switch,
    apply_isolated_jumps,
    apply_random_missing,
    apply_track_fragmentation,
    degrade_tracking,
)


def _make_test_df(n_frames: int = 50, n_players: int = 4) -> pd.DataFrame:
    """Generate a clean synthetic canonical tracking DataFrame."""
    rng = np.random.default_rng(100)
    rows = []
    teams = ["home"] * (n_players // 2) + ["away"] * (n_players - n_players // 2)
    player_ids = [str(i + 1) for i in range(n_players)]

    for f in range(1, n_frames + 1):
        ts = f * 0.04  # 25 FPS
        for team, pid in zip(teams, player_ids):
            rows.append({
                "match_id": "test_match_01",
                "frame": f,
                "timestamp": ts,
                "player_id": pid,
                "team": team,
                "x": float(rng.uniform(0.1, 0.9)),
                "y": float(rng.uniform(0.1, 0.9)),
                "confidence": 1.0,
                "visible": True,
            })

    return pd.DataFrame(rows)


class TestInputImmutability(unittest.TestCase):
    """1. Input data is never modified in-place."""

    def test_random_missing_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_random_missing(df, missing_prob=0.5, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_contiguous_gaps_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_contiguous_gaps(df, gap_start_prob=0.1, gap_p_continue=0.05, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_jitter_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_coordinate_jitter(df, scale=0.05, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_jumps_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_isolated_jumps(df, jump_prob=0.5, magnitude_min=0.1, magnitude_max=0.3, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_fragmentation_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_track_fragmentation(df, frag_prob=1.0, num_splits=2, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_identity_switch_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _ = apply_identity_switch(df, switch_prob=1.0, duration_min=5, duration_max=10, seed=42)
        pd.testing.assert_frame_equal(df, copy_df)

    def test_composition_immutability(self):
        df = _make_test_df()
        copy_df = df.copy()
        _, _ = degrade_tracking(df, severity="moderate", seed=42)
        pd.testing.assert_frame_equal(df, copy_df)


class TestDeterminism(unittest.TestCase):
    """2. Same input + same seed + same config -> identical output."""

    def test_degrade_tracking_determinism(self):
        df = _make_test_df()
        out1, meta1 = degrade_tracking(df, severity="moderate", seed=42)
        out2, meta2 = degrade_tracking(df, severity="moderate", seed=42)
        pd.testing.assert_frame_equal(out1, out2)
        self.assertEqual(meta1["severity"], meta2["severity"])
        self.assertEqual(meta1["random_seed"], meta2["random_seed"])
        self.assertEqual(meta1["actual_parameters"], meta2["actual_parameters"])

    def test_functions_determinism(self):
        df = _make_test_df()
        res1 = apply_coordinate_jitter(df, scale=0.01, seed=123)
        res2 = apply_coordinate_jitter(df, scale=0.01, seed=123)
        pd.testing.assert_frame_equal(res1, res2)


class TestSeedDivergence(unittest.TestCase):
    """3. Different seeds produce distinct outputs."""

    def test_different_seeds_diverge(self):
        df = _make_test_df(n_frames=100)
        out1, _ = degrade_tracking(df, severity="moderate", seed=1)
        out2, _ = degrade_tracking(df, severity="moderate", seed=2)
        self.assertFalse(out1.equals(out2))


class TestCleanSeverity(unittest.TestCase):
    """4. Clean severity produces an identity transformation."""

    def test_clean_severity_identity(self):
        df = _make_test_df()
        out, meta = degrade_tracking(df, severity="clean", seed=42)
        pd.testing.assert_frame_equal(df, out)
        self.assertEqual(meta["severity"], "clean")


class TestRandomMissing(unittest.TestCase):
    """5. Random missing detections."""

    def test_missingness_semantics(self):
        df = _make_test_df(n_frames=100, n_players=4)
        out = apply_random_missing(df, missing_prob=0.4, seed=42)

        # Confirm some observations were dropped
        invisible = out[~out["visible"]]
        self.assertGreater(len(invisible), 0)

        # Confirm coordinates are NaN and confidence is 0.0
        self.assertTrue(invisible["x"].isna().all())
        self.assertTrue(invisible["y"].isna().all())
        self.assertTrue((invisible["confidence"] == 0.0).all())

        # Confirm visible rows remain intact
        visible = out[out["visible"]]
        self.assertTrue(visible["x"].notna().all())
        self.assertTrue(visible["y"].notna().all())
        self.assertTrue((visible["confidence"] == 1.0).all())

        # Confirm row count preserved
        self.assertEqual(len(df), len(out))


class TestContiguousGaps(unittest.TestCase):
    """6. Contiguous track gaps."""

    def test_gaps_contiguity_and_bounds(self):
        df = _make_test_df(n_frames=200, n_players=2)
        out = apply_contiguous_gaps(
            df, gap_start_prob=0.05, gap_p_continue=0.04, gap_min_length=6, gap_max_length=20, seed=42
        )

        for (_, _), grp in out.groupby(["team", "player_id"]):
            grp_sorted = grp.sort_values("frame")
            vis = grp_sorted["visible"].values

            gap_lens = []
            cur_len = 0
            in_gap = False
            for v in vis:
                if not v:
                    in_gap = True
                    cur_len += 1
                else:
                    if in_gap:
                        gap_lens.append(cur_len)
                        cur_len = 0
                        in_gap = False
            if in_gap:
                gap_lens.append(cur_len)

            for glen in gap_lens:
                self.assertGreaterEqual(glen, 6)

        self.assertEqual(len(df), len(out))


class TestCoordinateJitter(unittest.TestCase):
    """7. Coordinate jitter."""

    def test_jitter_perturbation_and_distributions(self):
        df = _make_test_df(n_frames=50, n_players=4)
        df_dropped = apply_random_missing(df, missing_prob=0.3, seed=1)

        # Test Gaussian
        out_gauss = apply_coordinate_jitter(df_dropped, scale=0.02, distribution="gaussian", seed=42)
        vis_mask = df_dropped["visible"]

        self.assertTrue((out_gauss.loc[vis_mask, "x"] != df_dropped.loc[vis_mask, "x"]).any())
        self.assertTrue(out_gauss.loc[~vis_mask, "x"].isna().all())

        # Test Uniform
        out_unif = apply_coordinate_jitter(df_dropped, scale=0.02, distribution="uniform", seed=42)
        self.assertTrue((out_unif.loc[vis_mask, "x"] != df_dropped.loc[vis_mask, "x"]).any())

        # Test Laplace
        out_lap = apply_coordinate_jitter(df_dropped, scale=0.02, distribution="laplace", seed=42)
        self.assertTrue((out_lap.loc[vis_mask, "x"] != df_dropped.loc[vis_mask, "x"]).any())


class TestIsolatedJumps(unittest.TestCase):
    """8. Isolated coordinate jumps."""

    def test_isolated_jump_magnitudes(self):
        df = _make_test_df(n_frames=100, n_players=4)
        out = apply_isolated_jumps(df, jump_prob=0.05, magnitude_min=0.10, magnitude_max=0.30, seed=42)

        dx = out["x"].values - df["x"].values
        dy = out["y"].values - df["y"].values
        disp = np.sqrt(dx**2 + dy**2)

        jumped = disp > 0.001
        self.assertTrue(jumped.any())

        jump_mags = disp[jumped]
        # Check within bounds with small numerical tolerance
        self.assertTrue(np.all(jump_mags >= 0.099))
        self.assertTrue(np.all(jump_mags <= 0.301))


class TestTrackFragmentation(unittest.TestCase):
    """9. Track fragmentation."""

    def test_fragmentation_id_syntax_and_keys(self):
        df = _make_test_df(n_frames=60, n_players=4)
        out = apply_track_fragmentation(df, frag_prob=1.0, num_splits=2, seed=42)

        # Check deterministic naming syntax: <id>_frag_<seg>
        frag_rows = out[out["player_id"].str.contains("_frag_")]
        self.assertGreater(len(frag_rows), 0)

        # Validate no duplicate keys introduced
        dups = out.duplicated(subset=["match_id", "frame", "team", "player_id"])
        self.assertFalse(dups.any())
        self.assertEqual(len(df), len(out))


class TestIdentitySwitches(unittest.TestCase):
    """10. Identity switches."""

    def test_identity_switch_same_team_and_keys(self):
        df = _make_test_df(n_frames=100, n_players=4)
        out = apply_identity_switch(df, switch_prob=1.0, duration_min=10, duration_max=20, seed=42)

        # Confirm identity swap occurred
        self.assertTrue((out["player_id"].values != df["player_id"].values).any())

        # Confirm team identity was strictly preserved
        pd.testing.assert_series_equal(out["team"], df["team"], check_names=False)

        # Confirm no duplicate (match_id, frame, team, player_id) keys
        dups = out.duplicated(subset=["match_id", "frame", "team", "player_id"])
        self.assertFalse(dups.any())


class TestCompositeDegradation(unittest.TestCase):
    """11. Composite degradation and metadata hierarchy."""

    def test_composition_metadata(self):
        df = _make_test_df(n_frames=80, n_players=4)
        out, meta = degrade_tracking(df, severity="moderate", seed=42)

        self.assertFalse(out.equals(df))
        self.assertIn("degradation_classes", meta)
        self.assertIn("observation_level", meta["degradation_classes"])
        self.assertIn("identity_level", meta["degradation_classes"])
        self.assertEqual(len(meta["enabled_degradations"]), 6)


class TestInvalidConfig(unittest.TestCase):
    """12. Invalid severity / configuration validation."""

    def test_invalid_severity_raises(self):
        df = _make_test_df()
        with self.assertRaises(ValueError):
            degrade_tracking(df, severity="ultra_severe", seed=42)

    def test_custom_without_config_raises(self):
        df = _make_test_df()
        with self.assertRaises(ValueError):
            degrade_tracking(df, severity="custom", seed=42)

    def test_unknown_distribution_raises(self):
        df = _make_test_df()
        with self.assertRaises(ValueError):
            apply_coordinate_jitter(df, scale=0.01, distribution="cauchy", seed=42)


if __name__ == "__main__":
    unittest.main()
