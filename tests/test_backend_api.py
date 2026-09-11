import unittest

from fastapi.testclient import TestClient

from backend.app import app


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

    def test_feature_endpoints(self):
        for path in (
            "/api/possession/metrica_game1/1250",
            "/api/events/metrica_game1",
            "/api/tactical/metrica_game1/1250",
            "/api/pass-candidates/metrica_game1/1250/HOME_10",
            "/api/pass-ranking/metrica_game1/1250/HOME_10",
            "/api/robustness/step83",
            "/api/gsr/status",
        ):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_unavailable_frame(self):
        response = self.client.get("/api/game-state/metrica_game1/999999999")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()