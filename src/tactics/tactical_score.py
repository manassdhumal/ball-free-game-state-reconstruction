import numpy as np
from typing import Dict, Any

def calculate_tactical_score(features: Dict[str, Any], possessor_team: str) -> float:
    """
    Calculates a heuristic Tactical Advantage Score (TAS) in the range [0, 1] 
    for the team currently in possession.
    
    A higher score indicates a better tactical situation (more support, 
    more forward options, less pressure, disorganized defense).
    
    Parameters
    ----------
    features : dict
        Tactical features extracted for the current frame.
    possessor_team : str
        'home' or 'away' indicating which team has possession.
        
    Returns
    -------
    float
        The Tactical Advantage Score, normalized to [0, 1].
        Returns 0.0 if features are missing or invalid.
    """
    if pd.isna(features.get('possessor_pressure', np.nan)):
        return 0.0
        
    opposing_team = 'away' if possessor_team == 'home' else 'home'
    
    # Extract features with defaults if missing
    pressure = features.get('possessor_pressure', 0.0)
    support = features.get('possessor_support', 0.0)
    forward_options = features.get('forward_options', 0.0)
    defensive_compactness = features.get(f'{opposing_team}_compactness_area', 1.0)
    
    if pd.isna(defensive_compactness):
        defensive_compactness = 1.0
        
    # Heuristic scoring components
    
    # 1. Pressure component: 
    # High pressure (small distance) -> bad. Low pressure (large distance) -> good.
    # Cap distance at 0.3 for normalization (approx 20-30 meters)
    score_pressure = np.clip(pressure / 0.3, 0.0, 1.0)
    
    # 2. Support component:
    # Cap support at 3 players
    score_support = np.clip(support / 3.0, 0.0, 1.0)
    
    # 3. Penetration / Options:
    # Cap forward options at 4 players
    score_options = np.clip(forward_options / 4.0, 0.0, 1.0)
    
    # 4. Defensive Disorganization:
    # High compactness (small area) -> bad. Low compactness (large area) -> good.
    # Typical pitch area in normalized coords is 1.0. 
    # Let's say a highly compact block is 0.1, dispersed is > 0.4.
    score_defense_spread = np.clip(defensive_compactness / 0.4, 0.0, 1.0)
    
    # Combine with weights (must sum to 1.0 for normalized output)
    w_pressure = 0.35
    w_options = 0.35
    w_support = 0.15
    w_defense = 0.15
    
    tas = (
        w_pressure * score_pressure +
        w_options * score_options +
        w_support * score_support +
        w_defense * score_defense_spread
    )
    
    return float(np.clip(tas, 0.0, 1.0))

import pandas as pd
