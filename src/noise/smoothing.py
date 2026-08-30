"""
Trajectory Smoothing Module

Provides low-pass filtering and trajectory smoothing for tracking data:
  1. Moving Average Filter (`smooth_moving_average`)
  2. Savitzky-Golay Filter (`smooth_savitzky_golay`)
  3. Unified Smoothing Interface (`smooth_trajectory`)

Design Guarantees:
  - Operates independently per (match_id, team, player_id).
  - Only filters contiguous visible segments; NEVER smooths across missing gaps.
  - Never fabricates coordinates for missing observations (missing remains missing).
  - Strictly preserves frame, timestamp, player_id, team, match_id.
  - Never modifies input DataFrame in-place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _smooth_segment_moving_average(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply centered moving average smoothing to a 1D coordinate array pair."""
    n = len(x)
    if n == 0 or window_size <= 1:
        return x.copy(), y.copy()

    w = min(window_size, n)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return x.copy(), y.copy()

    half = w // 2
    x_smooth = x.copy()
    y_smooth = y.copy()

    for i in range(n):
        i_start = max(0, i - half)
        i_end = min(n, i + half + 1)
        x_smooth[i] = np.mean(x[i_start:i_end])
        y_smooth[i] = np.mean(y[i_start:i_end])

    return x_smooth, y_smooth


def _smooth_segment_savgol(
    x: np.ndarray,
    y: np.ndarray,
    window_length: int,
    polyorder: int,
    mode: str = "interp",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Savitzky-Golay filter to a contiguous coordinate array pair with safe fallbacks."""
    n = len(x)
    if n == 0:
        return x.copy(), y.copy()

    if n <= polyorder:
        # Not enough points for the polynomial order: return unperturbed
        return x.copy(), y.copy()

    # Determine effective window length
    w = window_length
    if w > n:
        w = n
    if w % 2 == 0:
        w -= 1

    if w <= polyorder:
        # Fall back to largest possible odd window > polyorder, or moving average
        w = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
        if w > n:
            return _smooth_segment_moving_average(x, y, window_size=3)

    x_smooth = savgol_filter(x, window_length=w, polyorder=polyorder, mode=mode)
    y_smooth = savgol_filter(y, window_length=w, polyorder=polyorder, mode=mode)

    return x_smooth, y_smooth


def smooth_moving_average(
    df: pd.DataFrame,
    window_size: int = 5,
) -> pd.DataFrame:
    """Apply moving average smoothing to visible contiguous trajectory segments.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame.
    window_size : int, default=5
        Number of frames in the moving average filter window (at 25 FPS, 5 frames = 0.20 s).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with smoothed coordinates on visible observations.

    Raises
    ------
    ValueError
        If window_size < 1.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    return smooth_trajectory(df, method="moving_average", window_size=window_size)


def smooth_savitzky_golay(
    df: pd.DataFrame,
    window_length: int = 7,
    polyorder: int = 2,
    mode: str = "interp",
) -> pd.DataFrame:
    """Apply Savitzky-Golay polynomial smoothing to visible contiguous trajectory segments.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame.
    window_length : int, default=7
        Length of the filter window in frames (must be positive odd integer).
        At 25 FPS, 7 frames = 0.28 s; 11 frames = 0.44 s.
    polyorder : int, default=2
        Order of polynomial used to fit samples (must be < window_length).
    mode : str, default='interp'
        Boundary extension mode passed to `scipy.signal.savgol_filter`.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with smoothed coordinates on visible observations.

    Raises
    ------
    ValueError
        If window_length is even, < 3, or polyorder >= window_length.
    """
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError(f"window_length must be an odd integer >= 3, got {window_length}")
    if polyorder < 1 or polyorder >= window_length:
        raise ValueError(f"polyorder must be between 1 and window_length - 1, got {polyorder}")

    return smooth_trajectory(
        df,
        method="savgol",
        window_length=window_length,
        polyorder=polyorder,
        mode=mode,
    )


def smooth_trajectory(
    df: pd.DataFrame,
    method: str = "savgol",
    **kwargs: Any,
) -> pd.DataFrame:
    """Unified smoothing interface operating on contiguous visible trajectory segments.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical tracking DataFrame.
    method : str, default='savgol'
        Smoothing algorithm: 'savgol' or 'moving_average'.
    **kwargs : Any
        Hyperparameters passed to the underlying filter algorithm:
          - For 'savgol': window_length (int, default=7), polyorder (int, default=2), mode (str, default='interp')
          - For 'moving_average': window_size (int, default=5)

    Returns
    -------
    pd.DataFrame
        A new DataFrame with smoothed coordinates.
    """
    method_clean = method.strip().lower()
    if method_clean not in {"savgol", "moving_average", "ma"}:
        raise ValueError(f"Unsupported smoothing method '{method}'. Supported: 'savgol', 'moving_average'.")

    required_cols = {"match_id", "frame", "timestamp", "player_id", "team", "x", "y", "confidence", "visible"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required canonical columns: {missing_cols}")

    out = df.copy()
    if out.empty:
        return out

    # Extract method parameters
    if method_clean in {"savgol"}:
        window_length = kwargs.get("window_length", 7)
        polyorder = kwargs.get("polyorder", 2)
        mode = kwargs.get("mode", "interp")
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError(f"window_length must be an odd integer >= 3, got {window_length}")
        if polyorder < 1 or polyorder >= window_length:
            raise ValueError(f"polyorder must be between 1 and window_length - 1, got {polyorder}")
    else:
        window_size = kwargs.get("window_size", 5)
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

    for (_, _, _), group_indices in out.groupby(["match_id", "team", "player_id"]).groups.items():
        grp = out.loc[group_indices].sort_values("frame")
        n_rows = len(grp)
        if n_rows == 0:
            continue

        vis = grp["visible"].values
        x_vals = grp["x"].values.copy().astype(float)
        y_vals = grp["y"].values.copy().astype(float)

        # Locate contiguous visible runs
        i = 0
        while i < n_rows:
            if vis[i]:
                run_start = i
                while i < n_rows and vis[i]:
                    i += 1
                run_end = i  # exclusive index

                seg_x = x_vals[run_start:run_end]
                seg_y = y_vals[run_start:run_end]

                if method_clean == "savgol":
                    sm_x, sm_y = _smooth_segment_savgol(seg_x, seg_y, window_length, polyorder, mode)
                else:
                    sm_x, sm_y = _smooth_segment_moving_average(seg_x, seg_y, window_size)

                x_vals[run_start:run_end] = sm_x
                y_vals[run_start:run_end] = sm_y
            else:
                i += 1

        # Write back smoothed coordinates
        grp_idx = grp.index
        out.loc[grp_idx, "x"] = x_vals
        out.loc[grp_idx, "y"] = y_vals

    return out
