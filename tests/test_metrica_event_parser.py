import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.metrica_event_parser import load_metrica_events, validate_canonical_events

class TestMetricaEventParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.events_path = os.path.join(base_dir, 'data', 'raw', 'metrica', 'data', 'Sample_Game_1', 'Sample_Game_1_RawEventsData.csv')
        cls.df = load_metrica_events(cls.events_path, 'sample_game_1')

    def test_dataframe_is_non_empty(self):
        self.assertFalse(self.df.empty, "DataFrame should not be empty")

    def test_expected_canonical_columns_in_order(self):
        expected_cols = [
            'match_id', 'event_id', 'period', 'start_frame', 'end_frame',
            'start_timestamp', 'end_timestamp', 'event_type', 'event_subtype',
            'team', 'from_player_id', 'to_player_id',
            'start_x', 'start_y', 'end_x', 'end_y'
        ]
        self.assertListEqual(list(self.df.columns), expected_cols)

    def test_row_count(self):
        self.assertEqual(len(self.df), 1745, "Should load exactly 1745 events")

    def test_event_ids_unique_and_sequential(self):
        self.assertTrue(self.df['event_id'].is_unique)
        self.assertEqual(self.df['event_id'].min(), 1)
        self.assertEqual(self.df['event_id'].max(), 1745)
        # Check sequential order starting from 1
        expected_seq = list(range(1, 1746))
        self.assertListEqual(list(self.df['event_id']), expected_seq)

    def test_both_teams_exist(self):
        teams = self.df['team'].unique()
        self.assertIn('home', teams)
        self.assertIn('away', teams)
        self.assertEqual(len(teams), 2)

    def test_expected_event_types(self):
        types = self.df['event_type'].unique()
        self.assertIn('PASS', types)
        self.assertIn('SHOT', types)
        self.assertIn('RECOVERY', types)

    def test_missing_to_values_preserved(self):
        missing_to_rows = self.df[self.df['to_player_id'].isna()]
        self.assertFalse(missing_to_rows.empty, "There should be rows with missing to_player_id")
        
        # Verify first row (SET PIECE KICK OFF) to_player_id is None
        first_row = self.df.iloc[0]
        self.assertIsNone(first_row['to_player_id'], "First row 'To' value should be None")

    def test_missing_coordinates_preserved_as_nan(self):
        # Coordinates for first row should be NaN
        first_row = self.df.iloc[0]
        self.assertTrue(np.isnan(first_row['start_x']))
        self.assertTrue(np.isnan(first_row['start_y']))
        self.assertTrue(np.isnan(first_row['end_x']))
        self.assertTrue(np.isnan(first_row['end_y']))

    def test_match_id_populated(self):
        self.assertTrue((self.df['match_id'] == 'sample_game_1').all())

    def test_player_ids_correctly_extracted(self):
        # Row 2 (index 1) is a PASS from Player19 to Player21 (Away team)
        # So it should be AWAY_19 to AWAY_21
        row_2 = self.df.iloc[1]
        self.assertEqual(row_2['from_player_id'], 'AWAY_19')
        self.assertEqual(row_2['to_player_id'], 'AWAY_21')
        self.assertEqual(row_2['team'], 'away')

    def test_event_timing_fields(self):
        self.assertTrue(self.df['start_frame'].notna().all())
        self.assertTrue(self.df['start_timestamp'].notna().all())


class TestMetricaEventValidation(unittest.TestCase):
    def setUp(self):
        # A valid baseline dataframe to mutate for validation tests
        self.base_df = pd.DataFrame({
            'match_id': ['test_match', 'test_match'],
            'event_id': [1, 2],
            'period': [1, 1],
            'start_frame': [100, 105],
            'end_frame': [104, 110],
            'start_timestamp': [4.0, 4.2],
            'end_timestamp': [4.16, 4.4],
            'event_type': ['PASS', 'PASS'],
            'event_subtype': [None, None],
            'team': ['home', 'away'],
            'from_player_id': ['HOME_1', 'AWAY_2'],
            'to_player_id': ['HOME_2', 'AWAY_3'],
            'start_x': [0.5, 0.6],
            'start_y': [0.5, 0.6],
            'end_x': [0.6, 0.7],
            'end_y': [0.6, 0.7]
        })

    def test_valid_baseline(self):
        # Should validate without error
        validate_canonical_events(self.base_df)

    def test_invalid_team(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'team'] = 'invalid_team'
        with self.assertRaisesRegex(ValueError, "Invalid team values found"):
            validate_canonical_events(invalid_df)

    def test_duplicate_event_ids(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[1, 'event_id'] = 1
        with self.assertRaisesRegex(ValueError, "event_id must be unique"):
            validate_canonical_events(invalid_df)

    def test_non_sequential_event_ids(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[1, 'event_id'] = 3
        with self.assertRaisesRegex(ValueError, "event_id must be sequential"):
            validate_canonical_events(invalid_df)

    def test_invalid_frame_ordering(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'start_frame'] = 200
        with self.assertRaisesRegex(ValueError, "start_frame must be less than or equal to end_frame"):
            validate_canonical_events(invalid_df)

    def test_invalid_coordinate_consistency_start(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'start_x'] = np.nan
        with self.assertRaisesRegex(ValueError, "Coordinate inconsistency: start_x and start_y"):
            validate_canonical_events(invalid_df)

    def test_invalid_coordinate_consistency_end(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'end_y'] = np.nan
        with self.assertRaisesRegex(ValueError, "Coordinate inconsistency: end_x and end_y"):
            validate_canonical_events(invalid_df)

    def test_coordinate_out_of_bounds(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'start_x'] = 2.0
        with self.assertRaisesRegex(ValueError, "values significantly out of normalized range"):
            validate_canonical_events(invalid_df)

    def test_invalid_player_id_format(self):
        invalid_df = self.base_df.copy()
        invalid_df.loc[0, 'from_player_id'] = 'Player1'
        with self.assertRaisesRegex(ValueError, "Invalid from_player_id format"):
            validate_canonical_events(invalid_df)

if __name__ == '__main__':
    unittest.main()
