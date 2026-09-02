import unittest
import pandas as pd
import numpy as np
from src.possession.spatial_graph import FrameGraph, build_frame_graph
from src.tactics.tactical_features import extract_tactical_features, _empty_features

class TestTacticalFeatures(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic tracking dataframe
        self.df = pd.DataFrame([
            {'player_id': 'HOME_1', 'team': 'home', 'x': 0.1, 'y': 0.5, 'visible': True},
            {'player_id': 'HOME_2', 'team': 'home', 'x': 0.2, 'y': 0.4, 'visible': True},
            {'player_id': 'HOME_3', 'team': 'home', 'x': 0.3, 'y': 0.6, 'visible': True},
            {'player_id': 'AWAY_1', 'team': 'away', 'x': 0.8, 'y': 0.5, 'visible': True},
            {'player_id': 'AWAY_2', 'team': 'away', 'x': 0.9, 'y': 0.2, 'visible': True},
        ])
        self.fg = build_frame_graph(self.df, frame=1, timestamp=0.0)

    def test_extract_features_no_possessor(self):
        features = extract_tactical_features(self.fg, possessor_id=None)
        
        # Team centroids
        self.assertAlmostEqual(features['home_centroid_x'], 0.2)
        self.assertAlmostEqual(features['home_centroid_y'], 0.5)
        self.assertAlmostEqual(features['away_centroid_x'], 0.85)
        self.assertAlmostEqual(features['away_centroid_y'], 0.35)
        
        # Width/Length
        self.assertAlmostEqual(features['home_width'], 0.2) # 0.6 - 0.4
        self.assertAlmostEqual(features['home_length'], 0.2) # 0.3 - 0.1
        self.assertAlmostEqual(features['home_compactness_area'], 0.04)
        
        # Possessor features should be nan/0
        self.assertTrue(pd.isna(features['possessor_pressure']))
        self.assertTrue(pd.isna(features['possessor_support']))
        self.assertEqual(features['forward_options'], 0)

    def test_extract_features_with_possessor(self):
        # HOME_1 has possession
        features = extract_tactical_features(self.fg, possessor_id='HOME_1', home_attacks_x1=True)
        
        # Nearest opponent to HOME_1 is AWAY_1 (dist = sqrt(0.7^2 + 0) = 0.7)
        self.assertAlmostEqual(features['possessor_pressure'], 0.7)
        
        # Support: HOME_2 is dist sqrt(0.1^2 + 0.1^2) = 0.1414 <= 0.15. HOME_3 is sqrt(0.2^2 + 0.1^2) = 0.22 > 0.15
        self.assertEqual(features['possessor_support'], 1.0)
        
        # Forward options: Home attacks X=1. HOME_1 is at x=0.1.
        # HOME_2 and HOME_3 are > 0.1
        self.assertEqual(features['forward_options'], 2)

    def test_reverse_attacking_direction(self):
        # HOME_1 has possession, but home attacks X=0
        features = extract_tactical_features(self.fg, possessor_id='HOME_1', home_attacks_x1=False)
        # No teammates have x < 0.1
        self.assertEqual(features['forward_options'], 0)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['player_id', 'team', 'x', 'y', 'visible'])
        empty_fg = build_frame_graph(empty_df, 1, 0.0)
        
        features = extract_tactical_features(empty_fg)
        self.assertTrue(pd.isna(features['home_centroid_x']))
        self.assertTrue(pd.isna(features['possessor_pressure']))

    def test_missing_team(self):
        # Only home players
        home_only = self.df[self.df['team'] == 'home']
        fg = build_frame_graph(home_only, 1, 0.0)
        features = extract_tactical_features(fg)
        
        self.assertTrue(pd.isna(features['away_centroid_x']))
        self.assertFalse(pd.isna(features['home_centroid_x']))

    def test_no_leakage(self):
        # Add ball and event cols, ensure they are ignored
        df_leak = self.df.copy()
        df_leak['Ball_X'] = 0.99
        df_leak['event'] = 'PASS'
        
        fg = build_frame_graph(df_leak, 1, 0.0)
        features = extract_tactical_features(fg, possessor_id='HOME_1')
        
        # Ensure pressure is still computed relative to players, not ball
        self.assertAlmostEqual(features['possessor_pressure'], 0.7)
