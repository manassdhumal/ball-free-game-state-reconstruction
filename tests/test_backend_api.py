import unittest
from unittest.mock import patch

import pandas as pd

from fastapi.testclient import TestClient

from backend.app import app
from backend.services import data_service


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_matches(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        response = self.client.get("/api/matches")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["data"])

    def test_game_state_and_players(self):
        response = self.client.get("/api/game-state/metrica_game1/1250")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["players"]), 28)
        self.assertEqual(self.client.get("/api/players/metrica_game1").status_code, 200)

    def test_game_state_includes_observed_jersey_color(self):
        frame = pd.DataFrame([{
            "frame": 7,
            "timestamp": 0.28,
            "player_id": "HOME_1",
            "team": "home",
            "x": 0.4,
            "y": 0.6,
            "jersey_color": "#d73741",
            "confidence": 0.9,
            "visible": True,
        }])
        with patch("backend.services.data_service.tracking", return_value=frame):
            result = data_service.frame("uploaded_test", 7)

        self.assertEqual(result["players"][0]["jersey_color"], "#d73741")

    def test_game_state_uses_exactly_two_team_palette_colors(self):
        frame = pd.DataFrame([
            {"frame": 7, "timestamp": 0.28, "player_id": "1", "team": "home", "x": 0.4, "y": 0.6, "jersey_color": "#d73741", "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "2", "team": "home", "x": 0.42, "y": 0.6, "jersey_color": "#c92f39", "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "3", "team": "away", "x": 0.6, "y": 0.6, "jersey_color": "#2748ad", "visible": True},
        ])
        with patch("backend.services.data_service.tracking", return_value=frame):
            result = data_service.frame("uploaded_test", 7)

        colors_by_team = {
            team: {player["jersey_color"] for player in result["players"] if player["team"] == team}
            for team in ("home", "away")
        }
        self.assertEqual(colors_by_team["home"], {"#c92f39"})
        self.assertEqual(colors_by_team["away"], {"#2748ad"})
        self.assertEqual(result["team_colors"], {"home": "#c92f39", "away": "#2748ad"})

    def test_full_clip_frames_include_palette_colors_for_legacy_tracks(self):
        tracking = pd.DataFrame([
            {"frame": 0, "timestamp": 0.0, "player_id": "1", "team": "home", "x": 0.2, "y": 0.3, "visible": True},
            {"frame": 0, "timestamp": 0.0, "player_id": "2", "team": "away", "x": 0.8, "y": 0.7, "visible": True},
        ])
        with patch("backend.services.data_service.tracking", return_value=tracking):
            response = self.client.get("/api/game-state/uploaded_test/all")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        colors = {
            player["jersey_color"]
            for frame_rows in result["frames"].values()
            for player in frame_rows
        }
        self.assertEqual(colors, {"#c83232", "#3232c8"})
        self.assertEqual(result["team_colors"], {"home": "#c83232", "away": "#3232c8"})

    def test_tracking_quality_is_measured_from_active_tracking_rows(self):
        tracking = pd.DataFrame([
            {"frame": 0, "player_id": "1", "team": "home", "x": 0.2, "y": 0.3},
            {"frame": 1, "player_id": "1", "team": "home", "x": 0.3, "y": 0.4},
            {"frame": 3, "player_id": "1", "team": "home", "x": 1.1, "y": 0.4},
            {"frame": 0, "player_id": "2", "team": "away", "x": 0.8, "y": 0.4},
            {"frame": 1, "player_id": "2", "team": "away", "x": 0.7, "y": 0.4},
            {"frame": 3, "player_id": "2", "team": "away", "x": 0.6, "y": 0.4},
        ])
        with patch("backend.services.data_service.tracking", return_value=tracking):
            result = data_service.tracking_quality("uploaded_test")

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["observed_rows"], 6)
        self.assertEqual(result["coordinate_validity"], 0.8333)
        self.assertEqual(result["gap_count"], 2)
        self.assertTrue(all(track["active_frame_coverage"] == 0.75 for track in result["tracks"]))

    def test_video_pass_ranking_excludes_unknown_forward_direction(self):
        tracking = pd.DataFrame([
            {"frame": 7, "timestamp": 0.28, "player_id": "1", "team": "home", "x": 0.2, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "2", "team": "home", "x": 0.4, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "3", "team": "home", "x": 0.3, "y": 0.7, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "4", "team": "away", "x": 0.6, "y": 0.5, "visible": True},
        ])
        with patch("backend.services.data_service.tracking", return_value=tracking):
            result = data_service.pass_ranking("uploaded_test", 7, "HOME_1")

        self.assertTrue(result)
        self.assertTrue(all(row["forward_progress"] is None for row in result))

    def test_tactical_endpoint_returns_frame_specific_tas(self):
        frame = pd.DataFrame([
            {"frame": 7, "timestamp": 0.28, "player_id": "1", "team": "home", "x": 0.4, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "2", "team": "home", "x": 0.45, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "3", "team": "away", "x": 0.6, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "4", "team": "away", "x": 0.65, "y": 0.5, "visible": True},
        ])

        with patch("backend.services.data_service.tracking", return_value=frame):
            result = data_service.tactical("uploaded_test", 7)

        self.assertEqual(result["status"], "available")
        self.assertIn(result["possessor_team"], ("home", "away"))
        self.assertGreaterEqual(result["tas"], 0.0)
        self.assertLessEqual(result["tas"], 1.0)
        self.assertIsNotNone(result["features"]["possessor_pressure"])
        self.assertIsNone(result["features"]["forward_options"])
        self.assertIn("attack_direction", result["provenance"])

    def test_tactical_score_is_unavailable_without_both_teams(self):
        frame = pd.DataFrame([
            {"frame": 7, "timestamp": 0.28, "player_id": "1", "team": "home", "x": 0.4, "y": 0.5, "visible": True},
            {"frame": 7, "timestamp": 0.28, "player_id": "2", "team": "home", "x": 0.45, "y": 0.5, "visible": True},
        ])
        with patch("backend.services.data_service.tracking", return_value=frame):
            result = data_service.tactical("uploaded_test", 7)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["tas"])

    def test_possession_timeline_returns_sampled_inferences(self):
        frame = pd.DataFrame([
            {"frame": n, "timestamp": n / 25, "player_id": str(i), "team": team, "x": x, "y": 0.5, "visible": True}
            for n in (0, 1, 2, 3, 4)
            for i, (team, x) in enumerate((("home", 0.4), ("home", 0.45), ("away", 0.6), ("away", 0.65)), start=1)
        ])
        with patch("backend.services.data_service.tracking", return_value=frame):
            response = self.client.get(
                "/api/possession/uploaded_test/timeline?start_frame=0&end_frame=4&stride=2"
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["availability"]["status"], "available")
        self.assertEqual([item["frame"] for item in result["data"]], [0, 2, 4])
        self.assertIn(result["data"][0]["team"], ("home", "away"))

    def test_feature_endpoints(self):
        for path in (
            "/api/possession/metrica_game1/1250",
            "/api/events/metrica_game1",
            "/api/tactical/metrica_game1/1250",
            "/api/pass-candidates/metrica_game1/1250/HOME_10",
            "/api/pass-ranking/metrica_game1/1250/HOME_10",
            "/api/robustness/step83",
            "/api/tracking-quality/metrica_game1",
            "/api/evaluation/summary",
            "/api/gsr/status",
        ):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_unavailable_frame(self):
        response = self.client.get("/api/game-state/metrica_game1/999999999")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()