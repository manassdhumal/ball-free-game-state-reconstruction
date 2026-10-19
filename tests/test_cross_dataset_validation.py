"""Unit tests for Step 85: Cross-Dataset Validation."""

import unittest
import pandas as pd
from src.validation.cross_dataset_validation import run_cross_dataset_validation


class TestCrossDatasetValidation(unittest.TestCase):
    """Verify cross-dataset comparison across tracking sources."""

    def setUp(self):
        # Synthetic Metrica sample
        m_rows = []
        for f in range(20):
            ts = f * 0.04
            m_rows.append({"match_id": "metrica", "frame": f, "timestamp": ts, "player_id": "H1", "team": "home", "x": 0.3 + f*0.005, "y": 0.4, "confidence": 1.0, "visible": True})
            m_rows.append({"match_id": "metrica", "frame": f, "timestamp": ts, "player_id": "H2", "team": "home", "x": 0.4, "y": 0.5 + f*0.005, "confidence": 1.0, "visible": True})
            m_rows.append({"match_id": "metrica", "frame": f, "timestamp": ts, "player_id": "A1", "team": "away", "x": 0.6, "y": 0.5, "confidence": 1.0, "visible": True})
            m_rows.append({"match_id": "metrica", "frame": f, "timestamp": ts, "player_id": "A2", "team": "away", "x": 0.7, "y": 0.6, "confidence": 1.0, "visible": True})
        self.metrica_df = pd.DataFrame(m_rows)

        # Synthetic Video Perception sample
        v_rows = []
        for f in range(20):
            ts = f * 0.04
            v_rows.append({"match_id": "video", "frame": f, "timestamp": ts, "player_id": "HOME_1", "team": "home", "x": 0.32 + f*0.005, "y": 0.41, "confidence": 0.9, "visible": True})
            v_rows.append({"match_id": "video", "frame": f, "timestamp": ts, "player_id": "HOME_2", "team": "home", "x": 0.41, "y": 0.52 + f*0.005, "confidence": 0.88, "visible": True})
            v_rows.append({"match_id": "video", "frame": f, "timestamp": ts, "player_id": "AWAY_1", "team": "away", "x": 0.61, "y": 0.49, "confidence": 0.92, "visible": True})
            v_rows.append({"match_id": "video", "frame": f, "timestamp": ts, "player_id": "AWAY_2", "team": "away", "x": 0.69, "y": 0.59, "confidence": 0.85, "visible": True})
        self.video_df = pd.DataFrame(v_rows)

    def test_cross_dataset_execution(self):
        summary = run_cross_dataset_validation(
            metrica_df=self.metrica_df,
            video_df=self.video_df,
            output_dir="results/step85",
        )
        self.assertIn("results", summary)
        self.assertEqual(len(summary["results"]), 2)
        datasets = [r["dataset"] for r in summary["results"]]
        self.assertTrue(any("Metrica" in d for d in datasets))
        self.assertTrue(any("Video" in d for d in datasets))


if __name__ == "__main__":
    unittest.main()
