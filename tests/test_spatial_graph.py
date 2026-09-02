"""
Unit tests for src.possession.spatial_graph

Verifies:
- Graph construction from canonical tracking data
- Pairwise distance computation
- No self-edges
- Team separation (nearest teammate vs opponent)
- Invisible player exclusion
- KNN edge construction
- Edge cases (single player, empty frame)
- Anti-leakage (ball columns ignored)
- Invalid input handling
"""

import unittest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.possession.spatial_graph import (
    FrameGraph,
    build_frame_graph,
    build_spatial_features,
    _compute_pairwise_distances,
)


def _make_frame_df(players, frame=1, timestamp=0.04, match_id="test"):
    """Helper to create a canonical tracking subset for one frame.

    Parameters
    ----------
    players : list of dict
        Each dict has keys: player_id, team, x, y, visible
    """
    rows = []
    for p in players:
        rows.append({
            "match_id": match_id,
            "frame": frame,
            "timestamp": timestamp,
            "player_id": p["player_id"],
            "team": p["team"],
            "x": p["x"],
            "y": p["y"],
            "confidence": 1.0 if p["visible"] else 0.0,
            "visible": p["visible"],
        })
    return pd.DataFrame(rows)


class TestPairwiseDistances(unittest.TestCase):
    """Test the internal distance matrix computation."""

    def test_distance_matrix_shape(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        D = _compute_pairwise_distances(coords)
        self.assertEqual(D.shape, (3, 3))

    def test_distance_matrix_symmetry(self):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        D = _compute_pairwise_distances(coords)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_diagonal_is_inf(self):
        coords = np.array([[0, 0], [1, 0]])
        D = _compute_pairwise_distances(coords)
        self.assertTrue(np.isinf(D[0, 0]))
        self.assertTrue(np.isinf(D[1, 1]))

    def test_known_distances(self):
        coords = np.array([[0, 0], [3, 4]])
        D = _compute_pairwise_distances(coords)
        self.assertAlmostEqual(D[0, 1], 5.0)
        self.assertAlmostEqual(D[1, 0], 5.0)


class TestBuildFrameGraph(unittest.TestCase):
    """Test the public build_frame_graph function."""

    def setUp(self):
        self.players = [
            {"player_id": "1", "team": "home", "x": 0.2, "y": 0.3, "visible": True},
            {"player_id": "2", "team": "home", "x": 0.25, "y": 0.35, "visible": True},
            {"player_id": "3", "team": "away", "x": 0.7, "y": 0.6, "visible": True},
            {"player_id": "4", "team": "away", "x": 0.75, "y": 0.65, "visible": True},
        ]
        self.frame_df = _make_frame_df(self.players)

    def test_graph_construction_basic(self):
        """Node attributes are populated correctly."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04)
        self.assertIsInstance(g, FrameGraph)
        self.assertEqual(g.frame, 1)
        self.assertAlmostEqual(g.timestamp, 0.04)
        self.assertEqual(g.n_visible, 4)
        self.assertEqual(len(g.node_df), 4)
        self.assertIn("player_id", g.node_df.columns)
        self.assertIn("nearest_teammate_dist", g.node_df.columns)
        self.assertIn("nearest_opponent_dist", g.node_df.columns)
        self.assertIn("local_density", g.node_df.columns)

    def test_pairwise_distances(self):
        """Distance matrix has correct shape, symmetry, and positive values."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04)
        self.assertEqual(g.distance_matrix.shape, (4, 4))
        # Symmetry
        np.testing.assert_array_almost_equal(
            g.distance_matrix, g.distance_matrix.T
        )
        # Off-diagonal positive
        mask = ~np.eye(4, dtype=bool)
        self.assertTrue(np.all(g.distance_matrix[mask] > 0))

    def test_no_self_edges(self):
        """Diagonal is inf, no player appears as its own nearest neighbour."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04)
        for i in range(g.n_visible):
            self.assertTrue(np.isinf(g.distance_matrix[i, i]))
        # No player is its own nearest teammate or opponent
        for _, row in g.node_df.iterrows():
            self.assertNotEqual(row["player_id"], row["nearest_teammate_id"])
            self.assertNotEqual(row["player_id"], row["nearest_opponent_id"])

    def test_team_separation(self):
        """Nearest teammate is same team; nearest opponent is different team."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04)
        for _, row in g.node_df.iterrows():
            if row["nearest_teammate_id"] is not None:
                tm_team = g.node_df[
                    g.node_df["player_id"] == row["nearest_teammate_id"]
                ]["team"].values[0]
                self.assertEqual(row["team"], tm_team)
            if row["nearest_opponent_id"] is not None:
                opp_team = g.node_df[
                    g.node_df["player_id"] == row["nearest_opponent_id"]
                ]["team"].values[0]
                self.assertNotEqual(row["team"], opp_team)

    def test_invisible_players_excluded(self):
        """Players with visible=False are not in the graph."""
        players = self.players + [
            {"player_id": "99", "team": "home", "x": np.nan, "y": np.nan, "visible": False},
        ]
        frame_df = _make_frame_df(players)
        g = build_frame_graph(frame_df, frame=1, timestamp=0.04)
        self.assertEqual(g.n_visible, 4)
        self.assertNotIn("99", g.player_ids)

    def test_knn_edges(self):
        """KNN edge list has correct structure."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04, knn_k=2)
        # 4 players × 2 neighbours = 8 edges
        self.assertEqual(len(g.edges), 8)
        for src, tgt, dist in g.edges:
            self.assertIn(src, g.player_ids)
            self.assertIn(tgt, g.player_ids)
            self.assertNotEqual(src, tgt)
            self.assertGreater(dist, 0)

    def test_knn_disabled(self):
        """Setting knn_k=None produces no edges."""
        g = build_frame_graph(self.frame_df, frame=1, timestamp=0.04, knn_k=None)
        self.assertEqual(len(g.edges), 0)

    def test_single_player_frame(self):
        """Gracefully handles a frame with only one visible player."""
        players = [
            {"player_id": "1", "team": "home", "x": 0.5, "y": 0.5, "visible": True},
        ]
        frame_df = _make_frame_df(players)
        g = build_frame_graph(frame_df, frame=1, timestamp=0.04)
        self.assertEqual(g.n_visible, 1)
        self.assertEqual(g.distance_matrix.shape, (1, 1))
        self.assertTrue(np.isinf(g.distance_matrix[0, 0]))
        row = g.node_df.iloc[0]
        self.assertIsNone(row["nearest_teammate_id"])
        self.assertIsNone(row["nearest_opponent_id"])
        self.assertEqual(len(g.edges), 0)

    def test_empty_frame(self):
        """Returns empty graph for a frame with no visible players."""
        players = [
            {"player_id": "1", "team": "home", "x": np.nan, "y": np.nan, "visible": False},
        ]
        frame_df = _make_frame_df(players)
        g = build_frame_graph(frame_df, frame=1, timestamp=0.04)
        self.assertEqual(g.n_visible, 0)
        self.assertEqual(len(g.node_df), 0)
        self.assertEqual(g.distance_matrix.shape, (0, 0))
        self.assertEqual(len(g.edges), 0)

    def test_no_ball_columns_consumed(self):
        """Ball columns in the input do not affect graph construction."""
        frame_df = self.frame_df.copy()
        frame_df["Ball_X"] = 0.5
        frame_df["Ball_Y"] = 0.5

        g_with_ball = build_frame_graph(frame_df, frame=1, timestamp=0.04)
        g_without_ball = build_frame_graph(self.frame_df, frame=1, timestamp=0.04)

        pd.testing.assert_frame_equal(
            g_with_ball.node_df.reset_index(drop=True),
            g_without_ball.node_df.reset_index(drop=True),
        )

    def test_invalid_input(self):
        """Missing required columns raises ValueError."""
        bad_df = pd.DataFrame({"player_id": ["1"], "team": ["home"]})
        with self.assertRaises(ValueError):
            build_frame_graph(bad_df, frame=1, timestamp=0.04)


class TestBuildSpatialFeatures(unittest.TestCase):
    """Test batch graph construction."""

    def test_multi_frame(self):
        """build_spatial_features returns one graph per frame."""
        rows = []
        for f in [1, 2, 3]:
            for pid, team, x, y in [("1", "home", 0.2, 0.3), ("2", "away", 0.7, 0.6)]:
                rows.append({
                    "match_id": "test",
                    "frame": f,
                    "timestamp": f * 0.04,
                    "player_id": pid,
                    "team": team,
                    "x": x,
                    "y": y,
                    "confidence": 1.0,
                    "visible": True,
                })
        df = pd.DataFrame(rows)
        graphs = build_spatial_features(df)
        self.assertEqual(len(graphs), 3)
        for g in graphs:
            self.assertEqual(g.n_visible, 2)


if __name__ == "__main__":
    unittest.main()
