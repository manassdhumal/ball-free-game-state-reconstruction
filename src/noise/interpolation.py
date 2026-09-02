"""
Missing-Track Interpolation Module

Provides linear trajectory interpolation for short tracking gaps with strict
provenance tracking (via an internal 'imputed' column) and max_gap bounds.

Design Principles:
  - Operates independently per (match_id, team, player_id).
  - Strictly enforces max_gap: gaps longer than max_gap remain missing.
  - Requires valid bounding observed points before and after missing segments (no extrapolation).
  - Preserves original observed coordinates without modification.
  - Sets 'imputed' boolean flag (True = synthetic interpolated coordinate, False = original/missing).
  - Input DataFrame is never modified in-place.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd


def interpolate_trajectory(
    df: pd.DataFrame,
    max_gap: int = 10,
    confidence_decay: bool = False,
    imputed_confidence: float = 0.5,
) -> pd.DataFrame:
    """Linearly interpolate short missing coordinate gaps for player trajectories.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame.
    max_gap : int, default=10
        Maximum length of consecutive missing frames eligible for interpolation.
        Gaps strictly greater than max_gap remain unfilled (NaN).
    confidence_decay : bool, default=False
        If True, linearly decay confidence towards the center of the gap.
        If False, assigns fixed `imputed_confidence` to filled points.
    imputed_confidence : float, default=0.5
        Base confidence score assigned to interpolated observations.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with short gaps interpolated and an 'imputed' column added.

    Raises
    ------
    ValueError
        If max_gap < 1 or input lacks required canonical columns.
    """
    if max_gap < 1:
        raise ValueError(f"max_gap must be a positive integer >= 1, got {max_gap}")

    required_cols = {"match_id", "frame", "timestamp", "player_id", "team", "x", "y", "confidence", "visible"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required canonical columns: {missing_cols}")

    out = df.copy()

    # Initialize provenance column if not present
    if "imputed" not in out.columns:
        out["imputed"] = False
    else:
        out["imputed"] = out["imputed"].astype(bool)

    if out.empty:
        return out

    # Process each entity independently
    for (_, _, _), group_indices in out.groupby(["match_id", "team", "player_id"]).groups.items():
        grp = out.loc[group_indices].sort_values("frame")
        n_rows = len(grp)
        if n_rows == 0:
            continue

        vis = grp["visible"].values.copy()
        x_vals = grp["x"].values.copy().astype(float)
        y_vals = grp["y"].values.copy().astype(float)
        conf_vals = grp["confidence"].values.copy().astype(float)
        imputed_flags = grp["imputed"].values.copy().astype(bool)
        frames = grp["frame"].values.copy()

        # Identify contiguous missing segments
        i = 0
        while i < n_rows:
            if not vis[i]:
                gap_start = i
                while i < n_rows and not vis[i]:
                    i += 1
                gap_end = i  # exclusive index
                gap_len = gap_end - gap_start

                # Eligibility check: must be <= max_gap and have valid bounding observations
                has_left = gap_start > 0 and vis[gap_start - 1]
                has_right = gap_end < n_rows and vis[gap_end]

                if gap_len <= max_gap and has_left and has_right:
                    left_idx = gap_start - 1
                    right_idx = gap_end

                    x0, y0 = x_vals[left_idx], y_vals[left_idx]
                    x1, y1 = x_vals[right_idx], y_vals[right_idx]
                    f0, f1 = frames[left_idx], frames[right_idx]

                    denom = float(f1 - f0)
                    if denom > 0:
                        for k in range(gap_start, gap_end):
                            t = (frames[k] - f0) / denom
                            x_vals[k] = x0 + t * (x1 - x0)
                            y_vals[k] = y0 + t * (y1 - y0)
                            vis[k] = True
                            imputed_flags[k] = True

                            if confidence_decay:
                                # Distance to nearest observed boundary
                                dist_to_boundary = min(k - left_idx, right_idx - k)
                                max_dist = (gap_len + 1) / 2.0
                                factor = 1.0 - (dist_to_boundary / max_dist) * 0.5
                                conf_vals[k] = float(np.clip(imputed_confidence * factor, 0.1, 1.0))
                            else:
                                conf_vals[k] = imputed_confidence
            else:
                i += 1

        # Write back interpolated arrays
        grp_idx = grp.index
        out.loc[grp_idx, "x"] = x_vals
        out.loc[grp_idx, "y"] = y_vals
        out.loc[grp_idx, "visible"] = vis
        out.loc[grp_idx, "confidence"] = conf_vals
        out.loc[grp_idx, "imputed"] = imputed_flags

    return out
