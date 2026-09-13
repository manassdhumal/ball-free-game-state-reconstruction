"""
Unit Tests for kaggle/export_gsr.py module.
"""

import unittest
import numpy as np
import pandas as pd

from kaggle.export_gsr import (
    convert_raw_tracklab_to_canonical,
    stitch_tracklets,
    compute_tracking_quality_metrics,
    validate_canonical_dataframe,
)


class TestExportGSR(unittest.TestCase):
    """Test suite for SoccerNet-GSR canonical export processing."""

    def test_unpack_nested_attributes_and_team_mapping(self):
        """Verify that nested attributes (team, role, jersey) are unpacked and mapped correctly."""
        raw_preds = [
            # Home player (left)
            {
                "image_id": 2040000001,
                "track_id": 10,
                "supercategory": "object",
                "confidence": 0.95,
                "attributes": {"role": "player", "jersey": "7", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": -20.0, "y_bottom_middle": 10.0},
            },
            # Away player (right)
            {
                "image_id": 2040000001,
                "track_id": 20,
                "supercategory": "object",
                "confidence": 0.90,
                "attributes": {"role": "player", "jersey": "9", "team": "right"},
                "bbox_pitch": {"x_bottom_middle": 25.0, "y_bottom_middle": -5.0},
            },
            # Referee (role: referee, team: None)
            {
                "image_id": 2040000001,
                "track_id": 99,
                "supercategory": "object",
                "confidence": 0.85,
                "attributes": {"role": "referee", "jersey": None, "team": None},
                "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
            },
        ]
        df = pd.DataFrame(raw_preds)
        canon = convert_raw_tracklab_to_canonical(df, match_id="test_match", stitch=False)

        self.assertEqual(len(canon), 3)
        teams = dict(zip(canon["player_id"], canon["team"]))
        self.assertEqual(teams["10"], "home")
        self.assertEqual(teams["20"], "away")
        self.assertEqual(teams["99"], "referee")

    def test_three_cluster_balanced_team_mapping(self):
        """Verify that TrackLab 3-cluster predictions (0, 1, 2) map clusters 0 & 1 to home, cluster 2 to away, and center/sideline to referee."""
        raw_preds = [
            {"image_id": 1, "track_id": 1, "supercategory": "object", "attributes": {"role": "player", "team": "0"}, "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 10.0}},
            {"image_id": 1, "track_id": 2, "supercategory": "object", "attributes": {"role": "player", "team": "1"}, "bbox_pitch": {"x_bottom_middle": 5.0, "y_bottom_middle": -10.0}},
            {"image_id": 1, "track_id": 3, "supercategory": "object", "attributes": {"role": "player", "team": "2"}, "bbox_pitch": {"x_bottom_middle": 15.0, "y_bottom_middle": 5.0}},
            {"image_id": 1, "track_id": 16, "supercategory": "object", "attributes": {"role": "player", "team": "2"}, "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}},
            {"image_id": 1, "track_id": 99, "supercategory": "object", "attributes": {"role": "player", "team": "0"}, "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 35.0}},
        ]
        df = pd.DataFrame(raw_preds)
        canon = convert_raw_tracklab_to_canonical(df, match_id="test_3cluster", stitch=False)
        teams = dict(zip(canon["player_id"], canon["team"]))
        self.assertEqual(teams["1"], "home")
        self.assertEqual(teams["2"], "home")
        self.assertEqual(teams["3"], "away")
        self.assertEqual(teams["16"], "referee")
        self.assertEqual(teams["99"], "referee")

    def test_out_of_bounds_coordinate_preservation(self):
        """Verify coordinates slightly beyond [0, 1] are NOT clamped to 0.0 or 1.0."""
        raw_preds = [
            # Sideline position: y = 35.0m on a 68m-wide pitch -> norm_y = (35 + 34)/68 = 1.014706
            {
                "image_id": 1,
                "track_id": 1,
                "supercategory": "object",
                "confidence": 0.95,
                "attributes": {"role": "player", "jersey": "2", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 35.0},
            },
            # Endline position: x = -53.5m on 105m pitch -> norm_x = (-53.5 + 52.5)/105 = -0.009524
            {
                "image_id": 2,
                "track_id": 1,
                "supercategory": "object",
                "confidence": 0.95,
                "attributes": {"role": "player", "jersey": "2", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": -53.5, "y_bottom_middle": 0.0},
            },
        ]
        df = pd.DataFrame(raw_preds)
        canon = convert_raw_tracklab_to_canonical(df, match_id="test_match", stitch=False)

        p1_row1 = canon[canon["frame"] == 1].iloc[0]
        p1_row2 = canon[canon["frame"] == 2].iloc[0]

        # Must not be clamped to exactly 1.0
        self.assertGreater(p1_row1["y"], 1.0)
        self.assertAlmostEqual(p1_row1["y"], (35.0 + 34.0) / 68.0, places=5)

        # Must not be clamped to exactly 0.0
        self.assertLess(p1_row2["x"], 0.0)
        self.assertAlmostEqual(p1_row2["x"], (-53.5 + 52.5) / 105.0, places=5)

    def test_homography_divergence_rejected_as_nan(self):
        """Verify extreme calibration explosions (> 1.20 or < -0.20) are marked NaN and visible=False."""
        raw_preds = [
            {
                "image_id": 1,
                "track_id": 1,
                "supercategory": "object",
                "confidence": 0.95,
                "attributes": {"role": "player", "jersey": "2", "team": "left"},
                "bbox_pitch": {"x_bottom_middle": 200.0, "y_bottom_middle": -150.0},
            },
        ]
        df = pd.DataFrame(raw_preds)
        canon = convert_raw_tracklab_to_canonical(df, match_id="test_match", stitch=False)

        row = canon.iloc[0]
        self.assertTrue(np.isnan(row["x"]))
        self.assertTrue(np.isnan(row["y"]))
        self.assertFalse(row["visible"])

    def test_tracklet_stitching(self):
        """Verify that tracklets fragmented by occlusion are merged if physically consistent."""
        raw_preds = [
            # Tracklet A (frames 1-5, jersey 10)
            {"frame": 1, "player_id": "3", "team": "home", "jersey": "10", "x": 0.50, "y": 0.50, "confidence": 0.9, "visible": True},
            {"frame": 2, "player_id": "3", "team": "home", "jersey": "10", "x": 0.51, "y": 0.50, "confidence": 0.9, "visible": True},
            {"frame": 3, "player_id": "3", "team": "home", "jersey": "10", "x": 0.52, "y": 0.51, "confidence": 0.9, "visible": True},
            {"frame": 4, "player_id": "3", "team": "home", "jersey": "10", "x": 0.53, "y": 0.51, "confidence": 0.9, "visible": True},
            # Gap of 2 frames (occlusion)
            # Tracklet B (frames 7-10, jersey 10, nearby position)
            {"frame": 7, "player_id": "16", "team": "home", "jersey": "10", "x": 0.54, "y": 0.51, "confidence": 0.9, "visible": True},
            {"frame": 8, "player_id": "16", "team": "home", "jersey": "10", "x": 0.55, "y": 0.52, "confidence": 0.9, "visible": True},
            {"frame": 9, "player_id": "16", "team": "home", "jersey": "10", "x": 0.56, "y": 0.52, "confidence": 0.9, "visible": True},
            # Transient flicker (only 1 frame)
            {"frame": 5, "player_id": "99", "team": "away", "jersey": None, "x": 0.80, "y": 0.20, "confidence": 0.4, "visible": True},
        ]
        df = pd.DataFrame(raw_preds)
        stitched = stitch_tracklets(df, fps=25.0, max_gap_frames=25, min_track_len=3)

        # Track 16 should be merged into Track 3
        self.assertIn("3", stitched["player_id"].values)
        self.assertNotIn("16", stitched["player_id"].values)
        # Transient flicker 99 should be dropped
        self.assertNotIn("99", stitched["player_id"].values)
        self.assertEqual(len(stitched), 7)

    def test_quality_metrics_computation(self):
        """Verify summary metrics compute team and coordinate metrics properly."""
        df = pd.DataFrame({
            "match_id": ["m1"] * 4,
            "frame": [1, 1, 2, 2],
            "timestamp": [0.04, 0.04, 0.08, 0.08],
            "player_id": ["1", "2", "1", "2"],
            "team": ["home", "away", "home", "away"],
            "x": [0.45, 0.55, 0.46, 0.56],
            "y": [0.50, 0.50, 0.51, 0.51],
            "confidence": [0.95, 0.88, 0.94, 0.89],
            "visible": [True, True, True, True],
        })
        summary = compute_tracking_quality_metrics(df)
        row = summary.iloc[0]
        self.assertEqual(row["home_players"], 1)
        self.assertEqual(row["away_players"], 1)
        self.assertEqual(row["referees"], 0)
        self.assertAlmostEqual(row["x_min"], 0.45)
        self.assertAlmostEqual(row["x_max"], 0.56)


if __name__ == "__main__":
    unittest.main()
