"""
Metrica Sports Tracking Parser

Converts raw Metrica tracking CSVs into the canonical tall tracking format (v0.1.0).
"""

import pandas as pd
import numpy as np
from typing import Tuple

def parse_metrica_tracking_file(filepath: str, team_name: str, match_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses a single Metrica tracking CSV file.

    Args:
        filepath: Path to the Metrica CSV tracking file.
        team_name: The team name to assign to players ('home' or 'away').
        match_id: Unique identifier for the match.

    Returns:
        A tuple of (player_tracking_df, ball_tracking_df) in intermediate formats.
    """
    # The Metrica format has 3 header rows. We skip the first 2.
    # The 3rd row has headers like: Period, Frame, Time [s], Player11, Unnamed, Player1, Unnamed, ..., Ball, Unnamed
    df = pd.read_csv(filepath, skiprows=2)
    cols = list(df.columns)
    
    player_dfs = []
    ball_dfs = []
    
    # Coordinates start at index 3 and come in X, Y pairs
    for i in range(3, len(cols), 2):
        x_col = cols[i]
        y_col = cols[i+1] if i + 1 < len(cols) else None
        
        if not y_col:
            continue
            
        if x_col.startswith('Player'):
            player_id = x_col.replace('Player', '')
            temp_df = df[['Frame', 'Time [s]', x_col, y_col]].copy()
            temp_df.columns = ['frame', 'timestamp', 'x', 'y']
            temp_df['match_id'] = match_id
            temp_df['player_id'] = player_id
            temp_df['team'] = team_name
            player_dfs.append(temp_df)
            
        elif x_col == 'Ball':
            temp_df = df[['Frame', 'Time [s]', x_col, y_col]].copy()
            temp_df.columns = ['frame', 'timestamp', 'x', 'y']
            temp_df['match_id'] = match_id
            ball_dfs.append(temp_df)
            
    players_concat = pd.concat(player_dfs, ignore_index=True) if player_dfs else pd.DataFrame()
    ball_concat = pd.concat(ball_dfs, ignore_index=True) if ball_dfs else pd.DataFrame()
    
    return players_concat, ball_concat

def validate_canonical_tracking(df: pd.DataFrame) -> None:
    """
    Validates a canonical player tracking DataFrame against schema v0.2.0 rules.
    """
    required_cols = {'match_id', 'frame', 'timestamp', 'player_id', 'team', 'x', 'y', 'confidence', 'visible'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
        
    if df['match_id'].isna().any():
        raise ValueError("match_id must be populated for all rows.")
        
    # Check for duplicate player-frame rows
    dups = df.duplicated(subset=['match_id', 'frame', 'team', 'player_id'])
    if dups.any():
        raise ValueError(f"Found {dups.sum()} duplicate player rows per frame.")
        
    # Check confidence bounds
    if not df['confidence'].between(0.0, 1.0).all():
        raise ValueError("Confidence scores must be between 0.0 and 1.0.")
        
    # Check visible vs NaN consistency
    inconsistent_visible = df[df['visible'] & (df['x'].isna() | df['y'].isna())]
    if not inconsistent_visible.empty:
        raise ValueError("Found rows with visible=True but NaN coordinates.")
        
    inconsistent_invisible = df[(~df['visible']) & (df['x'].notna() | df['y'].notna())]
    if not inconsistent_invisible.empty:
        raise ValueError("Found rows with visible=False but valid coordinates.")

def load_metrica_match(home_filepath: str, away_filepath: str, match_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads both Home and Away Metrica tracking files and returns canonical dataframes.

    Args:
        home_filepath: Path to the Home team tracking CSV.
        away_filepath: Path to the Away team tracking CSV.
        match_id: Unique identifier for the match.

    Returns:
        A tuple of (canonical_player_tracking, canonical_ball_tracking).
    """
    home_players, home_ball = parse_metrica_tracking_file(home_filepath, 'home', match_id)
    away_players, away_ball = parse_metrica_tracking_file(away_filepath, 'away', match_id)
    
    # Synchronization Checks
    if not home_ball['frame'].equals(away_ball['frame']):
        raise ValueError("Home and Away tracking files have different Frame sequences.")
        
    if not np.allclose(home_ball['timestamp'], away_ball['timestamp'], atol=1e-5, equal_nan=True):
        raise ValueError("Home and Away tracking files have different Time [s] sequences.")
        
    both_visible = home_ball['x'].notna() & away_ball['x'].notna()
    if not np.allclose(home_ball.loc[both_visible, 'x'], away_ball.loc[both_visible, 'x'], atol=1e-5):
        raise ValueError("Ball X coordinates do not match between Home and Away files.")
    if not np.allclose(home_ball.loc[both_visible, 'y'], away_ball.loc[both_visible, 'y'], atol=1e-5):
        raise ValueError("Ball Y coordinates do not match between Home and Away files.")
    
    # 1. Player Tracking
    players_df = pd.concat([home_players, away_players], ignore_index=True)
    
    # Calculate visibility and confidence
    players_df['visible'] = players_df['x'].notna() & players_df['y'].notna()
    players_df['confidence'] = np.where(players_df['visible'], 1.0, 0.0)
    
    # Order columns
    canonical_cols = ['match_id', 'frame', 'timestamp', 'player_id', 'team', 'x', 'y', 'confidence', 'visible']
    players_df = players_df[canonical_cols]
    
    # Sort
    players_df = players_df.sort_values(by=['match_id', 'frame', 'team', 'player_id']).reset_index(drop=True)
    
    # Validate
    validate_canonical_tracking(players_df)
    
    # 2. Ball Tracking
    # Home and away contain the same ball data. We'll use the home file's ball data.
    # In a full implementation we might verify they match or average them.
    ball_df = home_ball.copy()
    ball_df['visible'] = ball_df['x'].notna() & ball_df['y'].notna()
    ball_df = ball_df[['match_id', 'frame', 'timestamp', 'x', 'y', 'visible']]
    ball_df = ball_df.sort_values(by=['match_id', 'frame']).reset_index(drop=True)
    
    return players_df, ball_df
