import unittest
import numpy as np
import pandas as pd
from src.tactics.tactical_score import calculate_tactical_score

class TestTacticalScore(unittest.TestCase):
    def test_ideal_conditions(self):
        # Maximum score scenario: max support, max forward options, low pressure, terrible opposing defense
        features = {
            'possessor_pressure': 0.3, # cap
            'possessor_support': 3.0, # cap
            'forward_options': 4.0, # cap
            'away_compactness_area': 0.4 # very loose defense
        }
        score = calculate_tactical_score(features, possessor_team='home')
        
        # All clipped to 1.0, weights sum to 1.0, so score should be 1.0
        self.assertAlmostEqual(score, 1.0)

    def test_worst_conditions(self):
        # Minimum score scenario: no support, no forward options, high pressure, compact opposing defense
        features = {
            'possessor_pressure': 0.0,
            'possessor_support': 0.0,
            'forward_options': 0.0,
            'away_compactness_area': 0.0
        }
        score = calculate_tactical_score(features, possessor_team='home')
        self.assertAlmostEqual(score, 0.0)

    def test_missing_pressure_returns_zero(self):
        # If we don't know pressure (no possessor), score is 0
        features = {
            'possessor_support': 3.0,
            'forward_options': 4.0,
            'away_compactness_area': 0.4
        }
        score = calculate_tactical_score(features, possessor_team='home')
        self.assertAlmostEqual(score, 0.0)
        
    def test_nan_pressure_returns_zero(self):
        features = {
            'possessor_pressure': np.nan,
            'possessor_support': 3.0
        }
        score = calculate_tactical_score(features, possessor_team='home')
        self.assertAlmostEqual(score, 0.0)

    def test_missing_defensive_compactness(self):
        features = {
            'possessor_pressure': 0.15, # 0.5 clipped
            'possessor_support': 1.5, # 0.5 clipped
            'forward_options': 2.0, # 0.5 clipped
        }
        score = calculate_tactical_score(features, possessor_team='home')
        # Without away_compactness_area, it defaults to 1.0 (clipped to 1.0)
        # Weights: 0.35 * 0.5 + 0.35 * 0.5 + 0.15 * 0.5 + 0.15 * 1.0 = 0.175 + 0.175 + 0.075 + 0.15 = 0.575
        self.assertAlmostEqual(score, 0.575)

    def test_away_team_possession(self):
        # When away has possession, we look at home's compactness
        features = {
            'possessor_pressure': 0.3,
            'possessor_support': 3.0,
            'forward_options': 4.0,
            'home_compactness_area': 0.0, # home is compact -> away score goes down
            'away_compactness_area': 0.4  # should be ignored
        }
        score = calculate_tactical_score(features, possessor_team='away')
        # Weights: 0.35(1) + 0.35(1) + 0.15(1) + 0.15(0) = 0.85
        self.assertAlmostEqual(score, 0.85)

    def test_deterministic(self):
        features = {
            'possessor_pressure': 0.1,
            'possessor_support': 1.0,
            'forward_options': 1.0,
            'home_compactness_area': 0.1
        }
        score1 = calculate_tactical_score(features, possessor_team='away')
        score2 = calculate_tactical_score(features, possessor_team='away')
        self.assertEqual(score1, score2)
