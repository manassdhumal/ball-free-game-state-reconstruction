import unittest

import numpy as np
import pandas as pd

from src.tactics.counterfactual_analysis import (
    actual_target_rank,
    compare_rankings,
    perturb_snapshot,
    rank_counterfactual_actions,
    ranking_metrics,
    summarize_rankings,
)
from src.tactics.pass_candidates import generate_pass_candidates
from src.tactics.pass_probability import fit_pass_model


def snapshot():
    return pd.DataFrame([
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_1", "team": "home", "x": 0.20, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_2", "team": "home", "x": 0.50, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_3", "team": "home", "x": 0.32, "y": 0.70, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "AWAY_1", "team": "away", "x": 0.42, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "AWAY_2", "team": "away", "x": 0.75, "y": 0.20, "visible": True},
    ])


class TestStep83CounterfactualAnalysis(unittest.TestCase):
    def setUp(self):
        self.state = snapshot()
        candidates = generate_pass_candidates(self.state, "HOME_1")
        self.model = fit_pass_model(candidates, [1, 0], {})
        self.config = {"home_attacks_x1": True, "ranking_weights": {"success": 0.55, "progress": 0.20, "space": 0.15, "support": 0.10, "pressure": 0.10, "distance": 0.05}}

    def test_deterministic_rank_and_actual_target_metrics(self):
        first = rank_counterfactual_actions(self.state, self.model, "HOME_1", self.config)
        second = rank_counterfactual_actions(self.state, self.model, "HOME_1", self.config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["rank"].tolist(), list(range(1, len(first) + 1)))
        self.assertEqual(actual_target_rank(first, first.iloc[0]["target_player_id"])["rank"], 1)
        metrics = ranking_metrics(first, first.iloc[0]["target_player_id"])
        self.assertEqual(metrics["top_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_stability_and_flip_metrics(self):
        clean = rank_counterfactual_actions(self.state, self.model, "HOME_1", self.config)
        perturbed = rank_counterfactual_actions(perturb_snapshot(self.state, "coordinate_jitter", "mild", 83), self.model, "HOME_1", self.config)
        result = compare_rankings(clean, perturbed)
        self.assertEqual(result["common_candidates"], len(clean))
        self.assertTrue(np.isfinite([result["spearman"], result["kendall"], result["mean_rank_displacement"]]).all())
        self.assertIn(result["decision_flip"], (0.0, 1.0))

    def test_perturbations_are_deterministic_and_snapshot_only(self):
        altered = self.state.assign(Ball_X=0.9, event="PASS", outcome=1, future_x=123.0)
        first = perturb_snapshot(altered, "coordinate_jitter", "moderate", 83)
        second = perturb_snapshot(self.state, "coordinate_jitter", "moderate", 83)
        pd.testing.assert_frame_equal(first.drop(columns=["Ball_X", "event", "outcome", "future_x"]), second)
        self.assertTrue((first.loc[first.visible, ["x", "y"]].to_numpy() >= 0).all())
        self.assertTrue((first.loc[first.visible, ["x", "y"]].to_numpy() <= 1).all())

    def test_dropout_and_edge_cases(self):
        dropped = perturb_snapshot(self.state, "player_dropout", "severe", 83)
        self.assertLessEqual(int(dropped["visible"].sum()), int(self.state["visible"].sum()))
        empty = rank_counterfactual_actions(self.state, self.model, "MISSING", self.config)
        self.assertEqual(len(empty), 0)
        self.assertEqual(summarize_rankings([empty], ["MISSING"], "empty")["n_states"], 0)

    def test_future_rows_do_not_affect_current_ranking(self):
        current = rank_counterfactual_actions(self.state, self.model, "HOME_1", self.config)
        future = pd.concat([self.state, self.state.assign(frame=11, timestamp=0.44, x=0.99, y=0.01)], ignore_index=True)
        ranking = rank_counterfactual_actions(future[future["frame"] == 10], self.model, "HOME_1", self.config)
        pd.testing.assert_frame_equal(current, ranking)

    def test_event_and_ball_annotations_do_not_affect_ranking(self):
        current = rank_counterfactual_actions(self.state, self.model, "HOME_1", self.config)
        annotated = self.state.assign(Ball_X=0.99, Ball_Y=0.01, event="PASS", outcome=0)
        ranking = rank_counterfactual_actions(annotated, self.model, "HOME_1", self.config)
        pd.testing.assert_frame_equal(current, ranking)


if __name__ == "__main__":
    unittest.main()