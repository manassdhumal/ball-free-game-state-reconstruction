"""
Metrica Sports Event Parser

Converts raw Metrica event CSVs into the canonical event format (v0.1.0).
"""

import pandas as pd
import numpy as np
import re
from typing import Optional

def validate_canonical_events(df: pd.DataFrame) -> None:
    """
    Validates a canonical event DataFrame against schema v0.1.0 rules.

    Args:
        df: The canonical event DataFrame to validate.

    Raises:
        ValueError: If the DataFrame does not comply with the schema.
    """
    # 1. Required columns
    required_cols = {
        'match_id', 'event_id', 'period', 'start_frame', 'start_timestamp',
        'event_type', 'team', 'from_player_id'
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # 2. event_id uniqueness and sequential order
    if not df['event_id'].is_unique:
        raise ValueError("event_id must be unique across all rows.")
    
    if len(df) > 0:
        sorted_ids = df['event_id'].sort_values().values
        expected_ids = np.arange(1, len(df) + 1)
        if not np.array_equal(sorted_ids, expected_ids):
            raise ValueError("event_id must be sequential starting from 1.")

    # 3. match_id populated
    if df['match_id'].isna().any() or (df['match_id'] == '').any():
        raise ValueError("match_id must be populated and non-null for all rows.")

    # 4. valid team values
    invalid_teams = df[~df['team'].isin(['home', 'away'])]
    if not invalid_teams.empty:
        raise ValueError(f"Invalid team values found: {invalid_teams['team'].unique()}")

    # 5. frame/timestamp numeric validity
    if df['start_frame'].isna().any() or df['start_timestamp'].isna().any():
        raise ValueError("start_frame and start_timestamp cannot be NaN.")

    if not pd.api.types.is_numeric_dtype(df['start_frame']) or not pd.api.types.is_numeric_dtype(df['start_timestamp']):
        raise ValueError("start_frame and start_timestamp must be numeric.")

    if not pd.api.types.is_numeric_dtype(df['end_frame']) or not pd.api.types.is_numeric_dtype(df['end_timestamp']):
        raise ValueError("end_frame and end_timestamp must be numeric.")

    # 6. start_frame <= end_frame where both are present
    valid_end = df['end_frame'].notna()
    if (df.loc[valid_end, 'start_frame'] > df.loc[valid_end, 'end_frame']).any():
        raise ValueError("start_frame must be less than or equal to end_frame.")

    # 7. coordinate consistency
    start_x_nan = df['start_x'].isna()
    start_y_nan = df['start_y'].isna()
    if (start_x_nan != start_y_nan).any():
        raise ValueError("Coordinate inconsistency: start_x and start_y must be either both present or both missing.")

    end_x_nan = df['end_x'].isna()
    end_y_nan = df['end_y'].isna()
    if (end_x_nan != end_y_nan).any():
        raise ValueError("Coordinate inconsistency: end_x and end_y must be either both present or both missing.")

    for col in ['start_x', 'start_y', 'end_x', 'end_y']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Coordinate column {col} must be numeric.")
        # Check coordinate bounds
        valid_vals = df[col].dropna()
        if not ((valid_vals >= -0.5) & (valid_vals <= 1.5)).all():
            raise ValueError(f"Coordinate column {col} has values significantly out of normalized range [0.0, 1.0].")

    # 8. player ID formatting where present
    player_pattern = re.compile(r'^(HOME|AWAY)_\d+$')
    
    # from_player_id is required
    for pid in df['from_player_id'].dropna():
        if not player_pattern.match(pid):
            raise ValueError(f"Invalid from_player_id format: '{pid}'")

    for pid in df['to_player_id'].dropna():
        if not player_pattern.match(pid):
            raise ValueError(f"Invalid to_player_id format: '{pid}'")


def load_metrica_events(filepath: str, match_id: str) -> pd.DataFrame:
    """
    Loads raw Metrica event data from a CSV file and converts it into the
    canonical event schema (v0.1.0).

    Args:
        filepath: Path to the raw Metrica events CSV file.
        match_id: Unique identifier for the match.

    Returns:
        A pandas DataFrame formatted to the canonical event schema.
    """
    # 1. Load the CSV file
    # Treat "NaN" strings in coordinates as proper pandas NaN values
    df = pd.read_csv(filepath, na_values=['NaN', 'nan', 'NaN '])

    # 2. Generate sequential event_id (1-indexed) based on original chronological order
    df['event_id'] = range(1, len(df) + 1)
    df['match_id'] = match_id

    # 3. Map team
    # Home -> home, Away -> away
    team_mapping = {'Home': 'home', 'Away': 'away'}
    df['team'] = df['Team'].map(team_mapping)

    # 4. Map event types and subtypes
    df['event_type'] = df['Type']
    df['event_subtype'] = df['Subtype'].apply(lambda x: None if pd.isna(x) or str(x).strip() == "" else x)

    # 5. Map frame/time fields
    df['period'] = df['Period'].astype(int)
    df['start_frame'] = df['Start Frame'].astype(int)
    df['start_timestamp'] = df['Start Time [s]'].astype(float)

    # Parse end_frame and end_timestamp
    df['end_frame'] = df['End Frame'].astype(float)
    df['end_timestamp'] = df['End Time [s]'].astype(float)

    # Map 0 (or values less than start_frame) to NaN in end_frame and end_timestamp
    # since Metrica uses 0 as a placeholder for missing/invalid end frame
    missing_end = (df['end_frame'] == 0) | (df['end_frame'] < df['start_frame'])
    df.loc[missing_end, 'end_frame'] = np.nan
    df.loc[missing_end, 'end_timestamp'] = np.nan

    # 6. Parse player IDs (From / To)
    # Convert PlayerXX to the canonical format e.g. HOME_XX or AWAY_XX
    def to_canonical_player_id(player_val, team_val):
        if pd.isna(player_val) or str(player_val).strip() in ['', 'NaN', 'nan']:
            return None
        player_str = str(player_val).strip()
        jersey = player_str.replace('Player', '')
        if team_val == 'home':
            return f"HOME_{jersey}"
        elif team_val == 'away':
            return f"AWAY_{jersey}"
        else:
            return f"UNKNOWN_{jersey}"

    df['from_player_id'] = df.apply(lambda row: to_canonical_player_id(row['From'], row['team']), axis=1)
    df['from_player_id'] = df['from_player_id'].replace({np.nan: None})
    df['to_player_id'] = df.apply(lambda row: to_canonical_player_id(row['To'], row['team']), axis=1)
    df['to_player_id'] = df['to_player_id'].replace({np.nan: None})

    # 7. Map coordinates (Start X/Y, End X/Y)
    df['start_x'] = df['Start X'].astype(float)
    df['start_y'] = df['Start Y'].astype(float)
    df['end_x'] = df['End X'].astype(float)
    df['end_y'] = df['End Y'].astype(float)

    # Order columns as defined in canonical schema
    canonical_cols = [
        'match_id', 'event_id', 'period', 'start_frame', 'end_frame',
        'start_timestamp', 'end_timestamp', 'event_type', 'event_subtype',
        'team', 'from_player_id', 'to_player_id',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]
    df_canonical = df[canonical_cols].copy()

    # Validate the canonical DataFrame
    validate_canonical_events(df_canonical)

    return df_canonical
