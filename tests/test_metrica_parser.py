import unittest
import pandas as pd
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.metrica_parser import load_metrica_match

class TestMetricaParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.home_path = os.path.join(base_dir, 'data', 'raw', 'metrica', 'data', 'Sample_Game_1', 'Sample_Game_1_RawTrackingData_Home_Team.csv')
        cls.away_path = os.path.join(base_dir, 'data', 'raw', 'metrica', 'data', 'Sample_Game_1', 'Sample_Game_1_RawTrackingData_Away_Team.csv')
        
        # Load the data once for all tests
        cls.players_df, cls.ball_df = load_metrica_match(cls.home_path, cls.away_path, 'sample_game_1')

    def test_player_df_not_empty(self):
        self.assertFalse(self.players_df.empty, "Player dataframe should not be empty")

    def test_expected_canonical_columns(self):
        expected_cols = ['match_id', 'frame', 'timestamp', 'player_id', 'team', 'x', 'y', 'confidence', 'visible']
        self.assertListEqual(list(self.players_df.columns), expected_cols, "Player dataframe columns do not match canonical schema")

    def test_match_id_populated(self):
        self.assertTrue(self.players_df['match_id'].notna().all(), "Match IDs should not have NaN")
        self.assertEqual(self.players_df['match_id'].unique()[0], 'sample_game_1')

    def test_teams_exist(self):
        teams = self.players_df['team'].unique()
        self.assertIn('home', teams, "Home team missing")
        self.assertIn('away', teams, "Away team missing")

    def test_player_ids_populated(self):
        self.assertTrue(self.players_df['player_id'].notna().all(), "Player IDs should not have NaN")
        self.assertNotIn('Ball', self.players_df['player_id'].unique(), "Ball should not be in the player dataframe")

    def test_frame_and_timestamp_populated(self):
        self.assertTrue(self.players_df['frame'].notna().all(), "Frames should not have NaN")
        self.assertTrue(self.players_df['timestamp'].notna().all(), "Timestamps should not have NaN")

    def test_visible_confidence_consistency(self):
        # visible -> x/y not NaN, confidence == 1.0
        visible_mask = self.players_df['visible']
        self.assertTrue(self.players_df.loc[visible_mask, 'x'].notna().all())
        self.assertTrue(self.players_df.loc[visible_mask, 'y'].notna().all())
        self.assertTrue((self.players_df.loc[visible_mask, 'confidence'] == 1.0).all())
        
        # invisible -> x/y is NaN, confidence == 0.0
        invisible_mask = ~self.players_df['visible']
        self.assertTrue(self.players_df.loc[invisible_mask, 'x'].isna().all())
        self.assertTrue(self.players_df.loc[invisible_mask, 'y'].isna().all())
        self.assertTrue((self.players_df.loc[invisible_mask, 'confidence'] == 0.0).all())

    def test_ball_dataframe_separate(self):
        self.assertFalse(self.ball_df.empty, "Ball dataframe should not be empty")
        expected_ball_cols = ['match_id', 'frame', 'timestamp', 'x', 'y', 'visible']
        self.assertListEqual(list(self.ball_df.columns), expected_ball_cols, "Ball dataframe columns do not match expected")

    def test_validation_frame_mismatch(self):
        import unittest.mock as mock
        # Mock parse_metrica_tracking_file to return mismatched frames
        home_players = pd.DataFrame()
        home_ball = pd.DataFrame({'frame': [1, 2], 'timestamp': [0.1, 0.2], 'x': [0,0], 'y': [0,0]})
        away_players = pd.DataFrame()
        away_ball = pd.DataFrame({'frame': [1, 3], 'timestamp': [0.1, 0.2], 'x': [0,0], 'y': [0,0]})
        
        with mock.patch('src.data.metrica_parser.parse_metrica_tracking_file', side_effect=[(home_players, home_ball), (away_players, away_ball)]):
            with self.assertRaisesRegex(ValueError, "different Frame sequences"):
                load_metrica_match('dummy_home', 'dummy_away', 'dummy_match')

    def test_validation_time_mismatch(self):
        import unittest.mock as mock
        home_players = pd.DataFrame()
        home_ball = pd.DataFrame({'frame': [1, 2], 'timestamp': [0.1, 0.2], 'x': [0,0], 'y': [0,0]})
        away_players = pd.DataFrame()
        away_ball = pd.DataFrame({'frame': [1, 2], 'timestamp': [0.1, 0.3], 'x': [0,0], 'y': [0,0]})
        
        with mock.patch('src.data.metrica_parser.parse_metrica_tracking_file', side_effect=[(home_players, home_ball), (away_players, away_ball)]):
            with self.assertRaisesRegex(ValueError, "different Time \[s\] sequences"):
                load_metrica_match('dummy_home', 'dummy_away', 'dummy_match')

    def test_validation_ball_coord_mismatch(self):
        import unittest.mock as mock
        home_players = pd.DataFrame()
        home_ball = pd.DataFrame({'frame': [1, 2], 'timestamp': [0.1, 0.2], 'x': [0.5, 0.6], 'y': [0.5, 0.6]})
        away_players = pd.DataFrame()
        away_ball = pd.DataFrame({'frame': [1, 2], 'timestamp': [0.1, 0.2], 'x': [0.5, 0.9], 'y': [0.5, 0.6]})
        
        with mock.patch('src.data.metrica_parser.parse_metrica_tracking_file', side_effect=[(home_players, home_ball), (away_players, away_ball)]):
            with self.assertRaisesRegex(ValueError, "Ball X coordinates do not match"):
                load_metrica_match('dummy_home', 'dummy_away', 'dummy_match')

if __name__ == '__main__':
    unittest.main()
