"""
Unit tests for src.possession.possession_baseline

Verifies:
- Possession score shape and columns
- Anti-leakage (no ball columns, no event labels consumed)
- Temporal persistence mechanism
- No-possession state representation
- Score reproducibility (deterministic)
- Event candidate generation
- Pass vs turnover distinction
- Recovery detection
- Evaluation precision/recall
- Invalid input handling
"""

import unittest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.possession.possession_baseline import (
    predict_possession,
    infer_events,
    evaluate_predictions,
    possession_summary,
    score_frame,
    DEFAULT_WEIGHTS,
)
from src.possession.spatial_graph import build_frame_graph


def _make_tracking_df(n_frames=10, n_home=3, n_away=3, match_id="test"):
    """Create a minimal canonical tracking DataFrame for testing.

    Home players at x ∈ [0.2, 0.4], away players at x ∈ [0.6, 0.8].
    Players have slight frame-to-frame movement.
    """
    rows = []
    rng = np.random.RandomState(42)
    dt = 0.04

    for f in range(1, n_frames + 1):
        ts = f * dt
        for i in range(n_home):
            rows.append({
                "match_id": match_id,
                "frame": f,
                "timestamp": ts,
                "player_id": f"H{i+1}",
                "team": "home",
                "x": 0.2 + i * 0.1 + rng.normal(0, 0.005),
                "y": 0.4 + rng.normal(0, 0.005),
                "confidence": 1.0,
                "visible": True,
            })
        for i in range(n_away):
            rows.append({
                "match_id": match_id,
                "frame": f,
                "timestamp": ts,
                "player_id": f"A{i+1}",
                "team": "away",
                "x": 0.6 + i * 0.1 + rng.normal(0, 0.005),
                "y": 0.6 + rng.normal(0, 0.005),
                "confidence": 1.0,
                "visible": True,
            })
    return pd.DataFrame(rows)


class TestPossessionScoreShape(unittest.TestCase):
    """Test that the possession output has the expected schema."""

    def test_output_columns(self):
        df = _make_tracking_df(n_frames=5)
        poss = predict_possession(df)
        expected_cols = [
            "match_id", "frame", "timestamp",
            "possessor_id", "team",
            "possession_score", "confidence",
        ]
        self.assertListEqual(list(poss.columns), expected_cols)

    def test_output_row_count(self):
        df = _make_tracking_df(n_frames=10)
        poss = predict_possession(df)
        self.assertEqual(len(poss), 10)

    def test_match_id_consistent(self):
        df = _make_tracking_df(n_frames=5, match_id="game_1")
        poss = predict_possession(df)
        self.assertTrue((poss["match_id"] == "game_1").all())


class TestAntiLeakage(unittest.TestCase):
    """Verify the inference pipeline does not consume ball or event data."""

    def test_no_ball_columns_consumed(self):
        """Adding Ball_X/Ball_Y columns should not change predictions."""
        df = _make_tracking_df(n_frames=10)
        poss_clean = predict_possession(df)

        df_ball = df.copy()
        df_ball["Ball_X"] = 0.5
        df_ball["Ball_Y"] = 0.5
        poss_ball = predict_possession(df_ball)

        pd.testing.assert_frame_equal(poss_clean, poss_ball)

    def test_no_event_labels_consumed(self):
        """predict_possession does not accept event data as input."""
        import inspect
        sig = inspect.signature(predict_possession)
        param_names = set(sig.parameters.keys())
        # Should not have event-related parameters
        self.assertNotIn("events", param_names)
        self.assertNotIn("event_df", param_names)
        self.assertNotIn("reference_events", param_names)


class TestTemporalPersistence(unittest.TestCase):
    """Test that temporal continuity prevents random switching."""

    def test_possessor_persistent(self):
        """With high persistence_window, possessor should not switch rapidly."""
        df = _make_tracking_df(n_frames=20)
        poss = predict_possession(df, persistence_window=10, switch_margin=0.5)
        # Count switches
        switches = 0
        for i in range(1, len(poss)):
            if poss.iloc[i]["possessor_id"] != poss.iloc[i-1]["possessor_id"]:
                switches += 1
        # With high persistence & margin, very few switches expected
        self.assertLess(switches, 5)

    def test_low_persistence_more_switches(self):
        """With low persistence, more switches should occur."""
        df = _make_tracking_df(n_frames=20)
        poss_high = predict_possession(df, persistence_window=15, switch_margin=0.5)
        poss_low = predict_possession(df, persistence_window=1, switch_margin=0.01)

        switches_high = sum(
            1 for i in range(1, len(poss_high))
            if poss_high.iloc[i]["possessor_id"] != poss_high.iloc[i-1]["possessor_id"]
        )
        switches_low = sum(
            1 for i in range(1, len(poss_low))
            if poss_low.iloc[i]["possessor_id"] != poss_low.iloc[i-1]["possessor_id"]
        )
        self.assertGreaterEqual(switches_low, switches_high)


class TestNoPossessionState(unittest.TestCase):
    """Test explicit no-possession representation."""

    def test_high_threshold_produces_none(self):
        """When min_score_threshold is very high, no player qualifies."""
        df = _make_tracking_df(n_frames=5)
        poss = predict_possession(df, min_score_threshold=999.0)
        self.assertTrue(poss["possessor_id"].isna().all())
        self.assertTrue(poss["team"].isna().all())


class TestReproducibility(unittest.TestCase):
    """Test deterministic output."""

    def test_same_input_same_output(self):
        """Running predict_possession twice on the same data gives identical results."""
        df = _make_tracking_df(n_frames=10)
        poss_1 = predict_possession(df)
        poss_2 = predict_possession(df)
        pd.testing.assert_frame_equal(poss_1, poss_2)


class TestEventInference(unittest.TestCase):
    """Test event candidate generation from possession changes."""

    def _make_possession_df(self, sequence):
        """Build a possession DataFrame from a list of (player_id, team) tuples."""
        rows = []
        for i, (pid, team) in enumerate(sequence):
            rows.append({
                "match_id": "test",
                "frame": i + 1,
                "timestamp": (i + 1) * 0.04,
                "possessor_id": pid,
                "team": team,
                "possession_score": 0.5,
                "confidence": 0.5,
            })
        return pd.DataFrame(rows)

    def test_event_candidates_generated(self):
        """Possession changes produce candidate events."""
        poss = self._make_possession_df([
            ("H1", "home"), ("H1", "home"), ("A1", "away"), ("A1", "away"),
        ])
        events = infer_events(poss)
        self.assertGreater(len(events), 0)

    def test_pass_vs_turnover(self):
        """Same-team switch → pass_candidate; cross-team → turnover_candidate."""
        poss = self._make_possession_df([
            ("H1", "home"),
            ("H2", "home"),  # same team → pass
            ("A1", "away"),  # different team → turnover
        ])
        events = infer_events(poss)
        types = events["event_type"].tolist()
        self.assertIn("pass_candidate", types)
        self.assertIn("turnover_candidate", types)

    def test_recovery_detection(self):
        """A→B→A pattern yields recovery_candidate."""
        poss = self._make_possession_df([
            ("H1", "home"),
            ("A1", "away"),  # turnover
            ("H2", "home"),  # recovery (home regains)
        ])
        events = infer_events(poss)
        types = events["event_type"].tolist()
        self.assertIn("recovery_candidate", types)

    def test_no_events_from_stable_possession(self):
        """No events when possession doesn't change."""
        poss = self._make_possession_df([
            ("H1", "home"), ("H1", "home"), ("H1", "home"),
        ])
        events = infer_events(poss)
        self.assertEqual(len(events), 0)


class TestEvaluation(unittest.TestCase):
    """Test evaluation against reference events."""

    def test_perfect_match(self):
        """When predicted and reference events are identical, P=R=F1=1."""
        pred = pd.DataFrame({
            "match_id": ["test"] * 2,
            "frame": [100, 200],
            "timestamp": [4.0, 8.0],
            "event_type": ["pass_candidate", "pass_candidate"],
            "from_player_id": ["H1", "H2"],
            "to_player_id": ["H2", "H3"],
            "from_team": ["home", "home"],
            "to_team": ["home", "home"],
        })
        ref = pd.DataFrame({
            "match_id": ["test"] * 2,
            "event_id": [1, 2],
            "period": [1, 1],
            "start_frame": [100, 200],
            "start_timestamp": [4.0, 8.0],
            "event_type": ["PASS", "PASS"],
            "team": ["home", "home"],
            "from_player_id": ["HOME_1", "HOME_2"],
        })
        results = evaluate_predictions(pred, ref, tolerances_sec=[0.5])
        pass_results = results[results["event_type"] == "pass_candidate"]
        self.assertEqual(pass_results["precision"].values[0], 1.0)
        self.assertEqual(pass_results["recall"].values[0], 1.0)
        self.assertEqual(pass_results["f1"].values[0], 1.0)

    def test_no_match(self):
        """When predicted and reference events are far apart, P=R=F1=0."""
        pred = pd.DataFrame({
            "match_id": ["test"],
            "frame": [100],
            "timestamp": [4.0],
            "event_type": ["pass_candidate"],
            "from_player_id": ["H1"],
            "to_player_id": ["H2"],
            "from_team": ["home"],
            "to_team": ["home"],
        })
        ref = pd.DataFrame({
            "match_id": ["test"],
            "event_id": [1],
            "period": [1],
            "start_frame": [10000],
            "start_timestamp": [400.0],
            "event_type": ["PASS"],
            "team": ["home"],
            "from_player_id": ["HOME_1"],
        })
        results = evaluate_predictions(pred, ref, tolerances_sec=[0.5])
        pass_results = results[results["event_type"] == "pass_candidate"]
        self.assertEqual(pass_results["precision"].values[0], 0.0)
        self.assertEqual(pass_results["recall"].values[0], 0.0)


class TestInvalidInput(unittest.TestCase):
    """Test error handling for bad inputs."""

    def test_missing_columns(self):
        """Missing required columns raise ValueError."""
        bad_df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        with self.assertRaises(ValueError):
            predict_possession(bad_df)

    def test_empty_dataframe(self):
        """Empty tracking DataFrame produces empty possession."""
        empty_df = pd.DataFrame(columns=[
            "match_id", "frame", "timestamp", "player_id",
            "team", "x", "y", "visible",
        ])
        # Should not crash — returns empty or minimal output
        poss = predict_possession(empty_df)
        self.assertEqual(len(poss), 0)


class TestPossessionSummary(unittest.TestCase):
    """Test the summary statistics function."""

    def test_summary_keys(self):
        df = _make_tracking_df(n_frames=5)
        poss = predict_possession(df)
        summary = possession_summary(poss)
        expected_keys = {
            "total_frames", "possessing_frames", "no_possession_frames",
            "possession_pct", "n_switches", "home_pct", "away_pct",
        }
        self.assertEqual(set(summary.keys()), expected_keys)
        self.assertEqual(summary["total_frames"], 5)


if __name__ == "__main__":
    unittest.main()
