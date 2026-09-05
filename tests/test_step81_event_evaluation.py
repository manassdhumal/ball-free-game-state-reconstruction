"""Tests for controlled Step 81 ball-free event evaluation."""

import os
import sys
import inspect
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.step81_event_evaluation import evaluate_event_candidates, extract_candidate_events


class TestStep81EventEvaluation(unittest.TestCase):
    def test_candidate_extraction_has_no_event_label_argument(self):
        parameters = set(inspect.signature(extract_candidate_events).parameters)
        self.assertNotIn("reference_events", parameters)
        self.assertNotIn("events", parameters)

    def setUp(self):
        self.reference = pd.DataFrame({
            "event_type": ["PASS", "BALL LOST", "RECOVERY", "CHALLENGE"],
            "start_frame": [10, 20, 30, 40],
        })
        self.candidates = pd.DataFrame({
            "event_type": ["pass_candidate", "turnover_candidate", "recovery_candidate"],
            "frame": [11, 26, 30],
        })

    def test_tolerance_changes_matching_and_accounts_tp_fp_fn(self):
        result = evaluate_event_candidates(self.candidates, self.reference, 25.0, [0.2, 0.5])
        aggregate = result[result["scope"] == "aggregate"].sort_values("tolerance_sec")
        self.assertEqual(aggregate.iloc[0]["true_positive"], 2)
        self.assertEqual(aggregate.iloc[0]["false_positive"], 1)
        self.assertEqual(aggregate.iloc[0]["false_negative"], 1)
        self.assertEqual(aggregate.iloc[1]["true_positive"], 3)

    def test_duplicate_predictions_are_not_double_counted(self):
        candidates = pd.concat([self.candidates, self.candidates.iloc[[0]]], ignore_index=True)
        result = evaluate_event_candidates(candidates, self.reference, 25.0, [0.5])
        row = result[(result["scope"] == "aggregate")].iloc[0]
        self.assertEqual(row["n_predicted"], 3)
        self.assertEqual(row["true_positive"], 3)

    def test_zero_prediction_and_zero_reference_edges(self):
        empty = pd.DataFrame(columns=["event_type", "frame"])
        result = evaluate_event_candidates(empty, self.reference, 25.0, [1.0])
        aggregate = result[result["scope"] == "aggregate"].iloc[0]
        self.assertEqual(aggregate["true_positive"], 0)
        self.assertEqual(aggregate["false_positive"], 0)
        self.assertEqual(aggregate["false_negative"], 3)
        self.assertEqual(aggregate["f1"], 0.0)
        no_reference = pd.DataFrame(columns=["event_type", "start_frame"])
        result = evaluate_event_candidates(self.candidates, no_reference, 25.0, [1.0])
        self.assertEqual(result[result["scope"] == "aggregate"].iloc[0]["recall"], 0.0)

    def test_event_type_aggregation_and_determinism(self):
        first = evaluate_event_candidates(self.candidates, self.reference, 25.0, [0.2, 0.5, 1.0])
        second = evaluate_event_candidates(self.candidates, self.reference, 25.0, [0.2, 0.5, 1.0])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first[first["scope"] == "event_type"]["event_type"]), {
            "pass_candidate", "turnover_candidate", "recovery_candidate",
        })
        self.assertTrue(np.isfinite(first[["precision", "recall", "f1"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
