"""
SkillCorner Open Data Tracking Parser

Converts raw SkillCorner tracking JSONL and match metadata into the
canonical tracking schema (v0.2.0).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def parse_timestamp_str(ts_str: Optional[str]) -> Optional[float]:
    """
    Parses a SkillCorner timestamp string (e.g. '00:00:00.00' or '00:15.50') into total seconds.

    Args:
        ts_str: Timestamp string.

    Returns:
        Total elapsed seconds as float, or None if input is invalid/null.
    """
    if ts_str is None or pd.isna(ts_str) or not str(ts_str).strip():
        return None
    parts = str(ts_str).strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60.0 + float(s)
        elif len(parts) == 1:
            return float(parts[0])
    except (ValueError, TypeError):
        return None
    return None


def validate_canonical_tracking(df: pd.DataFrame) -> None:
    """
    Validates a canonical player tracking DataFrame against schema v0.2.0 rules.

    Args:
        df: The tracking DataFrame to validate.

    Raises:
        ValueError: If the DataFrame does not comply with the canonical schema.
    """
    # 1. Check required columns
    required_cols = {
        "match_id",
        "frame",
        "timestamp",
        "player_id",
        "team",
        "x",
        "y",
        "confidence",
        "visible",
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required canonical tracking columns: {missing}")

    # 2. match_id validation
    if df["match_id"].isna().any() or (df["match_id"] == "").any():
        raise ValueError("match_id must be populated and non-empty for all rows.")

    # 3. frame & timestamp validation
    if not pd.api.types.is_integer_dtype(df["frame"]):
        raise ValueError("frame column must be of integer data type.")

    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        raise ValueError("timestamp column must be numeric.")

    # 4. team controlled vocabulary
    valid_teams = {"home", "away", "referee"}
    invalid_teams = set(df["team"].unique()) - valid_teams
    if invalid_teams:
        raise ValueError(f"Invalid team values found: {invalid_teams}")

    # 5. confidence in [0.0, 1.0]
    if not pd.api.types.is_numeric_dtype(df["confidence"]):
        raise ValueError("confidence column must be numeric.")
    if (df["confidence"] < 0.0).any() or (df["confidence"] > 1.0).any():
        raise ValueError("confidence values must be bounded within [0.0, 1.0].")

    # 6. visible must be boolean
    if not pd.api.types.is_bool_dtype(df["visible"]):
        raise ValueError("visible column must be boolean.")

    # 7. Coordinate numeric validation
    if not pd.api.types.is_numeric_dtype(df["x"]) or not pd.api.types.is_numeric_dtype(df["y"]):
        raise ValueError("Coordinates x and y must be numeric.")

    # 8. Duplicate player/frame detection
    dups = df.duplicated(subset=["match_id", "frame", "team", "player_id"])
    if dups.any():
        raise ValueError(f"Found {dups.sum()} duplicate player rows per frame.")


def parse_skillcorner_tracking_file(
    tracking_filepath: Union[str, Path],
    match_filepath: Union[str, Path],
    match_id: str,
    normalize_coords: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses a single SkillCorner tracking JSONL file and associated match metadata JSON.

    Args:
        tracking_filepath: Path to `{id}_tracking_extrapolated.jsonl`.
        match_filepath: Path to `{id}_match.json`.
        match_id: Canonical match identifier string (e.g. 'skillcorner_1886347').
        normalize_coords: If True, transforms metric pitch coordinates to [0, 1]
                          normalized pitch coordinates using pitch_length and pitch_width.

    Returns:
        A tuple of (players_canonical_df, ball_df).

    Raises:
        ValueError: If an unknown player ID in tracking is missing from match metadata.
    """
    tracking_path = Path(tracking_filepath)
    match_path = Path(match_filepath)

    if not tracking_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")
    if not match_path.exists():
        raise FileNotFoundError(f"Match metadata file not found: {match_path}")

    # 1. Load match metadata
    with open(match_path, "r", encoding="utf-8") as mf:
        match_meta = json.load(mf)

    home_team_id = match_meta.get("home_team", {}).get("id")
    pitch_length = float(match_meta.get("pitch_length", 105.0))
    pitch_width = float(match_meta.get("pitch_width", 68.0))

    if pitch_length <= 0 or pitch_width <= 0:
        raise ValueError(f"Invalid pitch dimensions in metadata: length={pitch_length}, width={pitch_width}")

    # Build player metadata lookup (player['id'] -> team)
    player_team_map: Dict[int, str] = {}
    for p in match_meta.get("players", []):
        pid = p.get("id")
        team_id = p.get("team_id")
        is_home = team_id == home_team_id
        player_team_map[pid] = "home" if is_home else "away"

    # 2. Parse tracking JSONL
    player_rows = []
    ball_rows = []

    with open(tracking_path, "r", encoding="utf-8") as tf:
        for line in tf:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_num = int(record["frame"])
            ts_raw = record.get("timestamp")
            timestamp = parse_timestamp_str(ts_raw) if ts_raw is not None else float(frame_num) * 0.10

            # Process ball data if present
            ball_data = record.get("ball_data")
            if ball_data is not None:
                bx = ball_data.get("x")
                by = ball_data.get("y")
                bz = ball_data.get("z")
                b_det = bool(ball_data.get("is_detected", False))

                if bx is not None and by is not None:
                    if normalize_coords:
                        norm_bx = (float(bx) + pitch_length / 2.0) / pitch_length
                        norm_by = (float(by) + pitch_width / 2.0) / pitch_width
                    else:
                        norm_bx = float(bx)
                        norm_by = float(by)
                else:
                    norm_bx = np.nan
                    norm_by = np.nan

                ball_rows.append({
                    "match_id": match_id,
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "x": norm_bx,
                    "y": norm_by,
                    "z": float(bz) if bz is not None else np.nan,
                    "confidence": 1.0 if b_det else 0.0,
                    "visible": b_det,
                })

            # Process player data
            for p_obs in record.get("player_data", []):
                pid = p_obs.get("player_id")
                if pid not in player_team_map:
                    raise ValueError(
                        f"Unknown player_id '{pid}' in match '{match_id}': not found in match metadata."
                    )
                team_val = player_team_map[pid]
                player_id_val = str(pid)

                is_detected = bool(p_obs.get("is_detected", False))
                px = p_obs.get("x")
                py = p_obs.get("y")

                if px is not None and py is not None:
                    if normalize_coords:
                        norm_px = (float(px) + pitch_length / 2.0) / pitch_length
                        norm_py = (float(py) + pitch_width / 2.0) / pitch_width
                    else:
                        norm_px = float(px)
                        norm_py = float(py)
                else:
                    norm_px = np.nan
                    norm_py = np.nan

                player_rows.append({
                    "match_id": match_id,
                    "frame": frame_num,
                    "timestamp": timestamp,
                    "player_id": player_id_val,
                    "team": team_val,
                    "x": norm_px,
                    "y": norm_py,
                    "confidence": 1.0 if is_detected else 0.0,
                    "visible": is_detected,
                })

    # Construct DataFrames
    canonical_cols = [
        "match_id",
        "frame",
        "timestamp",
        "player_id",
        "team",
        "x",
        "y",
        "confidence",
        "visible",
    ]

    if player_rows:
        players_df = pd.DataFrame(player_rows)[canonical_cols]
    else:
        players_df = pd.DataFrame(columns=canonical_cols)

    # Cast dtypes
    if not players_df.empty:
        players_df["frame"] = players_df["frame"].astype(int)
        players_df["timestamp"] = players_df["timestamp"].astype(float)
        players_df["player_id"] = players_df["player_id"].astype(str)
        players_df["team"] = players_df["team"].astype(str)
        players_df["x"] = players_df["x"].astype(float)
        players_df["y"] = players_df["y"].astype(float)
        players_df["confidence"] = players_df["confidence"].astype(float)
        players_df["visible"] = players_df["visible"].astype(bool)

    ball_cols = ["match_id", "frame", "timestamp", "x", "y", "z", "confidence", "visible"]
    if ball_rows:
        ball_df = pd.DataFrame(ball_rows)[ball_cols]
    else:
        ball_df = pd.DataFrame(columns=ball_cols)

    # Validate against canonical schema rules
    if not players_df.empty:
        validate_canonical_tracking(players_df)

    return players_df, ball_df


def load_skillcorner_match(
    match_dir: Union[str, Path],
    match_id: Union[str, int],
    normalize_coords: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Public API to load and parse a SkillCorner match into canonical tracking data.

    Args:
        match_dir: Directory containing match files (or parent `data/matches/` folder).
        match_id: The match ID (integer or string).
        normalize_coords: If True, normalize coordinates to [0, 1].

    Returns:
        Tuple of (players_canonical_df, ball_df).
    """
    m_dir = Path(match_dir)
    mid_str = str(match_id).replace("skillcorner_", "")

    # Look in match_dir or match_dir / mid_str
    if (m_dir / f"{mid_str}_tracking_extrapolated.jsonl").exists():
        tracking_path = m_dir / f"{mid_str}_tracking_extrapolated.jsonl"
        match_path = m_dir / f"{mid_str}_match.json"
    elif (m_dir / mid_str / f"{mid_str}_tracking_extrapolated.jsonl").exists():
        tracking_path = m_dir / mid_str / f"{mid_str}_tracking_extrapolated.jsonl"
        match_path = m_dir / mid_str / f"{mid_str}_match.json"
    else:
        raise FileNotFoundError(
            f"Could not locate SkillCorner match files for ID '{mid_str}' in '{match_dir}'"
        )

    canonical_match_id = f"skillcorner_{mid_str}"
    return parse_skillcorner_tracking_file(
        tracking_filepath=tracking_path,
        match_filepath=match_path,
        match_id=canonical_match_id,
        normalize_coords=normalize_coords,
    )
