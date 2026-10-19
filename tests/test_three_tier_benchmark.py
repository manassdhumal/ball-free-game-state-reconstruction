"""Unit tests for Step 84: Three-Tier Robustness Benchmark."""

import unittest
import numpy as np
import pandas as pd

from src.validation.three_tier_benchmark import run_three_tier_benchmark


class TestThreeTierBenchmark(unittest.TestCase):
    """Verify that the Three-Tier benchmark runs cleanly across all tiers."""

    def setUp(self):
        # Create a synthetic 10-frame 4-player clean tracking DataFrame
        rows = []
        for f in range(10):
            ts = f * 0.04
            # Team home
            rows.append({"match_id": "test", "frame": f, "timestamp": ts, "player_id": "H1", "team": "home", "x": 0.2 + f * 0.01, "y": 0.3, "confidence": 1.0, "visible": True})
            rows.append({"match_id": "test", "frame": f, "timestamp": ts, "player_id": "H2", "team": "home", "x": 0.3, "y": 0.4 + f * 0.01, "confidence": 1.0, "visible": True})
            # Team away
            rows.append({"match_id": "test", "frame": f, "timestamp": ts, "player_id": "A1", "team": "away", "x": 0.6, "y": 0.5 - f * 0.01, "confidence": 1.0, "visible": True})
            rows.append({"match_id": "test", "frame": f, "timestamp": ts, "player_id": "A2", "team": "away", "x": 0.7 - f * 0.01, "y": 0.6, "confidence": 1.0, "visible": True})
        self.clean_df = pd.DataFrame(rows)

    def test_three_tier_benchmark_execution(self):
        summary = run_three_tier_benchmark(
            clean_tracking_df=self.clean_df,
            degradation_severity="mild",
            output_dir="results/step84",
        )
        self.assertIn("results", summary)
        self.assertEqual(len(summary["results"]), 4)

        tiers = [r["tier"] for r in summary["results"]]
        self.assertIn("CLEAN", tiers)
        self.assertIn("SYNTHETIC-DEGRADED", tiers)
        self.assertIn("REAL-PERCEPTION", tiers)
        self.assertIn("REAL-PERCEPTION-MITIGATED", tiers)

        # Clean tier should have 0 TAS error
        clean_res = next(r for r in summary["results"] if r["tier"] == "CLEAN")
        self.assertEqual(clean_res["tas_mae_relative_to_clean"], 0.0)
        self.assertEqual(clean_res["possessor_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
