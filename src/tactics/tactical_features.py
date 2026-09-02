import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.possession.spatial_graph import FrameGraph

def extract_tactical_features(
    frame_graph: FrameGraph, 
    possessor_id: Optional[str] = None,
    home_attacks_x1: bool = True
) -> Dict[str, Any]:
    """
    Extracts geometric and tactical features from a single frame using the
    reconstructed spatial graph. 
    
    Anti-leakage guarantee: Does NOT use ball coordinates or event labels.

    Parameters
    ----------
    frame_graph : FrameGraph
        The spatial graph representation of the current frame.
    possessor_id : str, optional
        The player_id of the inferred possessor from the possession baseline.
    home_attacks_x1 : bool
        If True, the home team attacks toward X=1. If False, toward X=0.
        (Typically True in 1st half, False in 2nd half).

    Returns
    -------
    dict
        Dictionary containing tactical features.
    """
    df = frame_graph.node_df
    if len(df) == 0:
        return _empty_features()

    features = {}

    # 1. Team-level features
    for team in ['home', 'away']:
        team_df = df[df['team'] == team]
        if len(team_df) > 0:
            features[f'{team}_centroid_x'] = team_df['x'].mean()
            features[f'{team}_centroid_y'] = team_df['y'].mean()
            features[f'{team}_width'] = team_df['y'].max() - team_df['y'].min()
            features[f'{team}_length'] = team_df['x'].max() - team_df['x'].min()
            features[f'{team}_compactness_area'] = features[f'{team}_width'] * features[f'{team}_length']
        else:
            features[f'{team}_centroid_x'] = np.nan
            features[f'{team}_centroid_y'] = np.nan
            features[f'{team}_width'] = np.nan
            features[f'{team}_length'] = np.nan
            features[f'{team}_compactness_area'] = np.nan

    # 2. Possessor-level features
    features['possessor_pressure'] = np.nan
    features['possessor_support'] = np.nan
    features['forward_options'] = 0

    if possessor_id and possessor_id in df['player_id'].values:
        possessor_row = df[df['player_id'] == possessor_id].iloc[0]
        team = possessor_row['team']
        px, py = possessor_row['x'], possessor_row['y']

        features['possessor_pressure'] = possessor_row['nearest_opponent_dist']
        
        # Count teammates within a support radius (e.g. 15% of pitch width)
        # Using the distance matrix from frame_graph
        try:
            p_idx = frame_graph.player_ids.index(possessor_id)
            dists = frame_graph.distance_matrix[p_idx, :]
            
            # support = teammates (excluding self) within distance 0.15
            teammates = df['team'] == team
            teammate_indices = np.where(teammates)[0]
            support_count = np.sum((dists[teammate_indices] <= 0.15) & (dists[teammate_indices] > 0))
            features['possessor_support'] = float(support_count)

            # forward options = teammates positioned ahead of possessor
            # "Ahead" depends on attacking direction
            if team == 'home':
                attacks_x1 = home_attacks_x1
            else:
                attacks_x1 = not home_attacks_x1

            if attacks_x1:
                forward_teammates = team_df[team_df['x'] > px]
            else:
                forward_teammates = team_df[team_df['x'] < px]
                
            features['forward_options'] = len(forward_teammates)
            
        except ValueError:
            # possessor_id not in distance matrix (shouldn't happen if in df, but safe fallback)
            pass

    return features

def _empty_features() -> Dict[str, Any]:
    features = {}
    for team in ['home', 'away']:
        features[f'{team}_centroid_x'] = np.nan
        features[f'{team}_centroid_y'] = np.nan
        features[f'{team}_width'] = np.nan
        features[f'{team}_length'] = np.nan
        features[f'{team}_compactness_area'] = np.nan
    
    features['possessor_pressure'] = np.nan
    features['possessor_support'] = np.nan
    features['forward_options'] = 0
    return features
