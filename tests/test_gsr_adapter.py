"""
Unit Tests for SoccerNet-GSR Canonical Tracking Adapter.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.gsr_adapter import load_canonical_gsr_tracking, validate_gsr_canonical_tracking


class TestGSRAdapter(unittest.TestCase):
    """Test suite for validating canonical GSR tracking loading and schema rules."""

    def test_valid_canonical_gsr_dataframe(self):
        df = pd.DataFrame({
            "match_id": ["gsr_sngs04", "gsr_sngs04"],
            "frame": [1, 1],
            "timestamp": [0.04, 0.04],
            "player_id": ["1", "2"],
            "team": ["home", "away"],
            "x": [0.45, 0.55],
            "y": [0.50, 0.50],
            "confidence": [0.95, 0.88],
            "visible": [True, True],
        })
        # Should validate without error
        validate_gsr_canonical_tracking(df)

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({
            "match_id": ["gsr_sngs04"],
            "frame": [1],
            "player_id": ["1"],
        })
        with self.assertRaisesRegex(ValueError, "GSR tracking data missing canonical columns"):
            validate_gsr_canonical_tracking(df)

    def test_invalid_team_raises_error(self):
        df = pd.DataFrame({
            "match_id": ["gsr_sngs04"],
            "frame": [1],
            "timestamp": [0.04],
            "player_id": ["1"],
            "team": ["invalid_team_name"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [1.0],
            "visible": [True],
        })
        with self.assertRaisesRegex(ValueError, "Invalid team values found in GSR tracking"):
            validate_gsr_canonical_tracking(df)

    def test_confidence_out_of_bounds_raises_error(self):
        df = pd.DataFrame({
            "match_id": ["gsr_sngs04"],
            "frame": [1],
            "timestamp": [0.04],
            "player_id": ["1"],
            "team": ["home"],
            "x": [0.5],
            "y": [0.5],
            "confidence": [1.5],
            "visible": [True],
        })
        with self.assertRaisesRegex(ValueError, "confidence values must be bounded within"):
            validate_gsr_canonical_tracking(df)

    def test_duplicate_player_per_frame_raises_error(self):
        df = pd.DataFrame({
            "match_id": ["gsr_sngs04", "gsr_sngs04"],
            "frame": [1, 1],
            "timestamp": [0.04, 0.04],
            "player_id": ["1", "1"],
            "team": ["home", "home"],
            "x": [0.5, 0.6],
            "y": [0.5, 0.6],
            "confidence": [1.0, 1.0],
            "visible": [True, True],
        })
        with self.assertRaisesRegex(ValueError, "duplicate player entries per frame"):
            validate_gsr_canonical_tracking(df)

    def test_load_canonical_gsr_tracking_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_gsr.csv"
            df = pd.DataFrame({
                "match_id": ["gsr_sngs04", "gsr_sngs04"],
                "frame": [1, 1],
                "timestamp": [0.04, 0.04],
                "player_id": ["10", "20"],
                "team": ["home", "away"],
                "x": [0.4, 0.6],
                "y": [0.5, 0.5],
                "confidence": [0.99, 0.95],
                "visible": [True, True],
            })
            df.to_csv(csv_path, index=False)

            loaded_df = load_canonical_gsr_tracking(csv_path, expected_match_id="gsr_sngs04")
            self.assertEqual(len(loaded_df), 2)
            self.assertEqual(list(loaded_df["player_id"]), ["10", "20"])
            self.assertTrue(pd.api.types.is_float_dtype(loaded_df["timestamp"]))


if __name__ == "__main__":
    unittest.main()
