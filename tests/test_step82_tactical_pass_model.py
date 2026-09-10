import inspect
import unittest

import numpy as np
import pandas as pd

from src.tactics.pass_candidates import FEATURE_NAMES, build_pass_features, generate_pass_candidates
from src.tactics.pass_probability import fit_pass_model, predict_pass_probability
from src.tactics.pass_ranking import evaluate_rankings, rank_counterfactual_passes


def snapshot():
    return pd.DataFrame([
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_1", "team": "home", "x": 0.20, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_2", "team": "home", "x": 0.50, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "HOME_3", "team": "home", "x": 0.32, "y": 0.70, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "AWAY_1", "team": "away", "x": 0.42, "y": 0.50, "visible": True},
        {"frame": 10, "timestamp": 0.4, "player_id": "AWAY_2", "team": "away", "x": 0.75, "y": 0.20, "visible": True},
    ])


class TestStep82TacticalPassModel(unittest.TestCase):
    def test_inference_entry_points_do_not_accept_labels_or_ball(self):
        self.assertNotIn("events", inspect.signature(build_pass_features).parameters)
        self.assertNotIn("labels", inspect.signature(generate_pass_candidates).parameters)
        base = snapshot()
        with_ball = base.assign(Ball_X=0.01, Ball_Y=0.99, event="PASS", outcome=0)
        pd.testing.assert_frame_equal(
            generate_pass_candidates(base, "HOME_1").drop(columns=["frame", "timestamp"]),
            generate_pass_candidates(with_ball, "HOME_1").drop(columns=["frame", "timestamp"]),
            check_dtype=False,
        )

    def test_features_are_current_snapshot_only(self):
        state = snapshot()
        altered = state.copy()
        altered["future_x"] = 999.0
        altered["future_frame"] = 9999
        self.assertEqual(build_pass_features(state, "HOME_1", "HOME_2"), build_pass_features(altered, "HOME_1", "HOME_2"))

    def test_candidates_are_same_team_and_configurable(self):
        candidates = generate_pass_candidates(snapshot(), "HOME_1", {"max_distance": 0.35})
        self.assertEqual(set(candidates["target_player_id"]), {"HOME_2", "HOME_3"})
        self.assertTrue((candidates["pass_distance"] <= 0.35).all())

    def test_fallback_and_logistic_model(self):
        state = snapshot()
        candidates = generate_pass_candidates(state, "HOME_1")
        fallback = fit_pass_model(candidates, [1, 0], {"minimum_examples_per_class": 10})
        self.assertEqual(fallback.model_type, "heuristic_fallback")
        self.assertTrue(np.isfinite(predict_pass_probability(fallback, candidates)).all())
        data = pd.concat([candidates] * 12, ignore_index=True)
        labels = [0, 1] * 12
        model = fit_pass_model(data, labels, {"minimum_examples_per_class": 5, "iterations": 40})
        self.assertEqual(model.model_type, "logistic_regression")
        self.assertTrue(np.isfinite(predict_pass_probability(model, data)).all())

    def test_ranking_is_deterministic_and_metrics_use_identifiable_target(self):
        state = snapshot()
        model = fit_pass_model(generate_pass_candidates(state, "HOME_1"), [1, 0], {})
        first = rank_counterfactual_passes(state, model, "HOME_1")
        second = rank_counterfactual_passes(state, model, "HOME_1")
        pd.testing.assert_frame_equal(first, second)
        metrics = evaluate_rankings([first], [first.iloc[0]["target_player_id"]])
        self.assertEqual(metrics.iloc[0]["n_ranked_states"], 1)
        self.assertTrue(np.isfinite(metrics.to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()