# Tactical Scoring Baseline

This document outlines the **Tactical Advantage Score (TAS)** and the underlying spatial features extracted from the tracking data. This module is part of Step 71 of the Ball-Free Game State Reconstruction project.

## Overview

The purpose of this module is to quantify the tactical advantage of the possessing team using *only* player positioning and geometry. It strictly adheres to the anti-leakage guarantee: **no ball coordinates and no reference event labels are used during inference.**

This is a **heuristic baseline**. It is intended to serve as a lightweight, interpretable proxy for spatial control and passing options. It does *not* claim to be a ground-truth machine learning model of expected threat (xT) or expected goals (xG).

## Extracted Features

Features are extracted per-frame via `src.tactics.tactical_features.extract_tactical_features`.

### Team-Level Features
- **Centroids (`team_centroid_x`, `team_centroid_y`)**: The mean X and Y position of all visible players on a team.
- **Width & Length (`team_width`, `team_length`)**: The spread of the team across the Y-axis (width) and X-axis (length).
- **Defensive Compactness (`team_compactness_area`)**: The bounding box area (`width * length`) of the defending team. A smaller area indicates a more organized, compact defensive block.

### Possessor-Level Features
- **Possessor Pressure (`possessor_pressure`)**: The Euclidean distance to the nearest opponent from the inferred possessor. Short distances indicate high pressure.
- **Possessor Support (`possessor_support`)**: The count of teammates within a `0.15` normalized radius (approx. 15-20 meters) of the possessor.
- **Forward Options (`forward_options`)**: The count of teammates positioned strictly closer to the opponent's goal line than the possessor. 
  - *Note on Attacking Direction*: This relies on knowing which direction the team is attacking (`home_attacks_x1` boolean flag).

## Tactical Advantage Score (TAS)

The TAS is calculated via `src.tactics.tactical_score.calculate_tactical_score`. It normalizes and combines the features into a single value in the range `[0, 1]`.

A score of **1.0** represents an ideal attacking state: low pressure, many forward options, high support, and a disorganized defense.
A score of **0.0** represents severe danger or isolation: high pressure, no support, no forward options, and a highly compact defense.

### Scoring Formula

The score is a weighted sum of four clipped components:

1. **Pressure Score (Weight: 35%)**: 
   `clip(possessor_pressure / 0.3, 0.0, 1.0)`
   Rewards having space. Caps out when the nearest opponent is > 0.3 away.

2. **Penetration / Forward Options (Weight: 35%)**: 
   `clip(forward_options / 4.0, 0.0, 1.0)`
   Rewards having teammates ahead of the ball. Caps out at 4 forward players.

3. **Support Score (Weight: 15%)**: 
   `clip(possessor_support / 3.0, 0.0, 1.0)`
   Rewards having passing options nearby. Caps out at 3 supporting players.

4. **Defensive Disorganization (Weight: 15%)**: 
   `clip(defending_team_compactness / 0.4, 0.0, 1.0)`
   Rewards stretching the defending team. A highly compact defense yields a low score component.

## Limitations & Assumptions

1. **Static Attacking Direction**: The `forward_options` feature assumes a static attacking direction for the duration of the analysis window. If applied across a half-time break, the flag must be flipped.
2. **Missing Players**: The features are sensitive to players moving out of the camera frame. If 3 attacking players are out of frame, the `forward_options` and `team_compactness_area` will artificially shrink.
3. **No Velocity Vectoring**: The baseline relies purely on static geometry (distance) rather than dynamic geometry (time-to-intercept). Future CRF/GNN iterations will incorporate player momentum.
