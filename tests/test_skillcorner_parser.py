"""
Unit Tests for SkillCorner Tracking Parser
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.skillcorner_parser import (
    load_skillcorner_match,
    parse_skillcorner_tracking_file,
    parse_timestamp_str,
    validate_canonical_tracking,
)


class TestSkillCornerParserRealData(unittest.TestCase):
    """Tests using real SkillCorner Open Data match 1886347."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path("data/raw/skillcorner/data/matches")
        cls.match_id = 1886347
        if (cls.data_dir / str(cls.match_id)).exists():
            cls.players_df, cls.ball_df = load_skillcorner_match(
                match_dir=cls.data_dir,
                match_id=cls.match_id,
                normalize_coords=True,
            )
        else:
            cls.players_df = None
            cls.ball_df = None

    def setUp(self):
        if self.players_df is None:
            self.skipTest(f"Real SkillCorner match {self.match_id} not available locally.")

    def test_non_empty_canonical_dataframe(self):
        self.assertFalse(self.players_df.empty, "Canonical players DataFrame must not be empty.")
        self.assertGreater(len(self.players_df), 100000)

    def test_expected_canonical_columns(self):
        expected_cols = [
            "match_id",
            "frame",
            "timestamp",
            "player_id",
            "team",
            "x",
            "y",
            "confidence",
            "visible",
        ]
        self.assertEqual(list(self.players_df.columns), expected_cols)

    def test_match_id_populated(self):
        self.assertTrue((self.players_df["match_id"] == f"skillcorner_{self.match_id}").all())

    def test_deterministic_player_id(self):
        # player_id must be stringified source ID (e.g. '51009'), NOT jersey number
        self.assertFalse(self.players_df["player_id"].isna().any())
        pids = self.players_df["player_id"].unique()
        self.assertIn("51009", pids)
        self.assertIn("38673", pids)
        # Verify all are numeric strings
        self.assertTrue(all(pid.isdigit() for pid in pids))

    def test_frame_and_timestamp_validity(self):
        self.assertTrue(pd.api.types.is_integer_dtype(self.players_df["frame"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.players_df["timestamp"]))
        self.assertFalse(self.players_df["frame"].isna().any())
        self.assertFalse(self.players_df["timestamp"].isna().any())
        self.assertGreaterEqual(self.players_df["timestamp"].min(), 0.0)

    def test_teams_representation(self):
        teams = set(self.players_df["team"].unique())
        self.assertEqual(teams, {"home", "away"})
        home_count = (self.players_df["team"] == "home").sum()
        away_count = (self.players_df["team"] == "away").sum()
        self.assertGreater(home_count, 0)
        self.assertGreater(away_count, 0)

    def test_normalized_coordinate_ranges(self):
        self.assertTrue(pd.api.types.is_float_dtype(self.players_df["x"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.players_df["y"]))
        # Most coordinates should fall within standard pitch [0.0, 1.0]
        on_pitch_x = (self.players_df["x"] >= 0.0) & (self.players_df["x"] <= 1.0)
        on_pitch_y = (self.players_df["y"] >= 0.0) & (self.players_df["y"] <= 1.0)
        self.assertGreater(on_pitch_x.mean(), 0.95, "Over 95% of coordinates should be on pitch")
        self.assertGreater(on_pitch_y.mean(), 0.95, "Over 95% of coordinates should be on pitch")

    def test_metric_coordinate_loading(self):
        players_metric, _ = load_skillcorner_match(
            match_dir=self.data_dir,
            match_id=self.match_id,
            normalize_coords=False,
        )
        # Metric coordinates centered at 0, with pitch 104x68
        self.assertLess(players_metric["x"].min(), -40.0)
        self.assertGreater(players_metric["x"].max(), 40.0)
        self.assertLess(players_metric["y"].min(), -25.0)
        self.assertGreater(players_metric["y"].max(), 25.0)

    def test_confidence_and_visibility_fields(self):
        self.assertTrue(pd.api.types.is_bool_dtype(self.players_df["visible"]))
        self.assertTrue(pd.api.types.is_float_dtype(self.players_df["confidence"]))
        self.assertTrue(((self.players_df["confidence"] == 1.0) | (self.players_df["confidence"] == 0.0)).all())
        # Visible True should have confidence 1.0
        vis_rows = self.players_df[self.players_df["visible"]]
        self.assertTrue((vis_rows["confidence"] == 1.0).all())
        # Visible False (extrapolated) should have confidence 0.0
        non_vis_rows = self.players_df[~self.players_df["visible"]]
        self.assertTrue((non_vis_rows["confidence"] == 0.0).all())
        # Both visible (detected) and extrapolated (non-detected) observations should exist
        self.assertGreater(len(vis_rows), 0)
        self.assertGreater(len(non_vis_rows), 0)

    def test_no_duplicate_player_frames(self):
        dups = self.players_df.duplicated(subset=["match_id", "frame", "team", "player_id"])
        self.assertFalse(dups.any(), f"Found {dups.sum()} duplicate player frames")

    def test_ball_dataframe_separation(self):
        self.assertFalse(self.ball_df.empty)
        ball_cols = ["match_id", "frame", "timestamp", "x", "y", "z", "confidence", "visible"]
        self.assertEqual(list(self.ball_df.columns), ball_cols)
        # No ball rows inside players_df
        self.assertFalse(self.players_df["player_id"].str.contains("ball", case=False).any())


class TestSkillCornerParserMocked(unittest.TestCase):
    """Focused unit tests using mocked inputs for validation, normalization, and edge cases."""

    def test_parse_timestamp_str(self):
        self.assertEqual(parse_timestamp_str("00:00:00.00"), 0.0)
        self.assertEqual(parse_timestamp_str("00:01:30.50"), 90.50)
        self.assertEqual(parse_timestamp_str("01:15:00.00"), 4500.0)
        self.assertEqual(parse_timestamp_str("12:34.50"), 754.50)
        self.assertIsNone(parse_timestamp_str(None))
        self.assertIsNone(parse_timestamp_str(""))
        self.assertIsNone(parse_timestamp_str("invalid_time"))

    def test_unknown_player_id_raises_value_error(self):
        # Create temporary match.json and tracking.jsonl where tracking has unknown player 99999
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            match_file = tmp_path / "1001_match.json"
            tracking_file = tmp_path / "1001_tracking_extrapolated.jsonl"

            match_data = {
                "home_team": {"id": 10},
                "away_team": {"id": 20},
                "pitch_length": 105,
                "pitch_width": 68,
                "players": [{"id": 1, "team_id": 10, "number": 7}],
            }
            with open(match_file, "w") as mf:
                json.dump(match_data, mf)

            tracking_line = {
                "frame": 100,
                "timestamp": "00:00:10.00",
                "player_data": [{"player_id": 99999, "x": 0.0, "y": 0.0, "is_detected": True}],
            }
            with open(tracking_file, "w") as tf:
                tf.write(json.dumps(tracking_line) + "\n")

            with self.assertRaisesRegex(ValueError, "Unknown player_id '99999' in match 'skillcorner_1001'"):
                parse_skillcorner_tracking_file(tracking_file, match_file, "skillcorner_1001")

    def test_missing_coordinates_preserved_as_nan(self):
        # Tracking record with None for x and y
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            match_file = tmp_path / "1002_match.json"
            tracking_file = tmp_path / "1002_tracking_extrapolated.jsonl"

            match_data = {
                "home_team": {"id": 10},
                "away_team": {"id": 20},
                "pitch_length": 105,
                "pitch_width": 68,
                "players": [{"id": 1, "team_id": 10, "number": 7}],
            }
            with open(match_file, "w") as mf:
                json.dump(match_data, mf)

            tracking_line = {
                "frame": 100,
                "timestamp": "00:00:10.00",
                "player_data": [{"player_id": 1, "x": None, "y": None, "is_detected": False}],
            }
            with open(tracking_file, "w") as tf:
                tf.write(json.dumps(tracking_line) + "\n")

            players_df, _ = parse_skillcorner_tracking_file(
                tracking_file, match_file, "skillcorner_1002", normalize_coords=True
            )
            self.assertEqual(len(players_df), 1)
            self.assertTrue(np.isnan(players_df.iloc[0]["x"]))
            self.assertTrue(np.isnan(players_df.iloc[0]["y"]))
            self.assertEqual(players_df.iloc[0]["player_id"], "1")
            self.assertFalse(players_df.iloc[0]["visible"])
            self.assertEqual(players_df.iloc[0]["confidence"], 0.0)

    def test_coordinate_normalization_exact_bounds(self):
        # Test exact mapping of center spot, edges, and out of bounds
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            match_file = tmp_path / "1003_match.json"
            tracking_file = tmp_path / "1003_tracking_extrapolated.jsonl"

            pitch_l = 100.0
            pitch_w = 60.0
            match_data = {
                "home_team": {"id": 10},
                "away_team": {"id": 20},
                "pitch_length": pitch_l,
                "pitch_width": pitch_w,
                "players": [
                    {"id": 1, "team_id": 10, "number": 1},
                    {"id": 2, "team_id": 10, "number": 2},
                    {"id": 3, "team_id": 10, "number": 3},
                    {"id": 4, "team_id": 10, "number": 4},
                    {"id": 5, "team_id": 10, "number": 5},
                ],
            }
            with open(match_file, "w") as mf:
                json.dump(match_data, mf)

            tracking_line = {
                "frame": 1,
                "timestamp": "00:00:00.10",
                "player_data": [
                    {"player_id": 1, "x": 0.0, "y": 0.0, "is_detected": True},  # Center -> (0.5, 0.5)
                    {"player_id": 2, "x": -50.0, "y": -30.0, "is_detected": True},  # Bottom-Left -> (0.0, 0.0)
                    {"player_id": 3, "x": 50.0, "y": 30.0, "is_detected": True},  # Top-Right -> (1.0, 1.0)
                    {"player_id": 4, "x": -50.0, "y": 30.0, "is_detected": True},  # Top-Left -> (0.0, 1.0)
                    {"player_id": 5, "x": 60.0, "y": 36.0, "is_detected": True},  # Out of bounds -> (1.1, 1.1)
                ],
            }
            with open(tracking_file, "w") as tf:
                tf.write(json.dumps(tracking_line) + "\n")

            players_df, _ = parse_skillcorner_tracking_file(
                tracking_file, match_file, "skillcorner_1003", normalize_coords=True
            )

            p1 = players_df[players_df["player_id"] == "1"].iloc[0]
            self.assertAlmostEqual(p1["x"], 0.5, places=5)
            self.assertAlmostEqual(p1["y"], 0.5, places=5)

            p2 = players_df[players_df["player_id"] == "2"].iloc[0]
            self.assertAlmostEqual(p2["x"], 0.0, places=5)
            self.assertAlmostEqual(p2["y"], 0.0, places=5)

            p3 = players_df[players_df["player_id"] == "3"].iloc[0]
            self.assertAlmostEqual(p3["x"], 1.0, places=5)
            self.assertAlmostEqual(p3["y"], 1.0, places=5)

            p4 = players_df[players_df["player_id"] == "4"].iloc[0]
            self.assertAlmostEqual(p4["x"], 0.0, places=5)
            self.assertAlmostEqual(p4["y"], 1.0, places=5)

            # Not clipped
            p5 = players_df[players_df["player_id"] == "5"].iloc[0]
            self.assertAlmostEqual(p5["x"], 1.1, places=5)
            self.assertAlmostEqual(p5["y"], 1.1, places=5)

    def test_validate_missing_columns(self):
        df = pd.DataFrame({"match_id": ["m1"], "frame": [1]})
        with self.assertRaisesRegex(ValueError, "Missing required canonical tracking columns"):
            validate_canonical_tracking(df)

    def test_validate_invalid_team(self):
        df = pd.DataFrame({
            "match_id": ["m1"],
            "frame": [1],
            "timestamp": [0.1],
            "player_id": ["1"],
            "team": ["invalid_team"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [1.0],
            "visible": [True],
        })
        with self.assertRaisesRegex(ValueError, "Invalid team values found"):
            validate_canonical_tracking(df)

    def test_validate_confidence_out_of_bounds(self):
        df = pd.DataFrame({
            "match_id": ["m1"],
            "frame": [1],
            "timestamp": [0.1],
            "player_id": ["1"],
            "team": ["home"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [1.5],
            "visible": [True],
        })
        with self.assertRaisesRegex(ValueError, "confidence values must be bounded within"):
            validate_canonical_tracking(df)

    def test_validate_duplicate_rows(self):
        df = pd.DataFrame({
            "match_id": ["m1", "m1"],
            "frame": [1, 1],
            "timestamp": [0.1, 0.1],
            "player_id": ["1", "1"],
            "team": ["home", "home"],
            "x": [0.5, 0.6],
            "y": [0.5, 0.6],
            "confidence": [1.0, 1.0],
            "visible": [True, True],
        })
        with self.assertRaisesRegex(ValueError, "duplicate player rows per frame"):
            validate_canonical_tracking(df)


if __name__ == "__main__":
    unittest.main()
