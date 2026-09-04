"""
Ball-Free Heuristic Possession Baseline

Infers possession from player tracking data **without** using ball
coordinates or event labels.  The pipeline has five stages:

1. Per-player heuristic possession scoring (spatial geometry + dynamics)
2. Temporal continuity (prevents frame-by-frame oscillation)
3. Canonical possession output DataFrame
4. Event inference from possession transitions
5. Evaluation against reference events

ANTI-LEAKAGE CONTRACT
---------------------
* Inference (stages 1-4) NEVER accesses Ball_X, Ball_Y, any ball-derived
  distance, event labels, future frames, or future ground-truth possession.
* Event labels are consumed ONLY in stage 5 (evaluation) AFTER predictions
  are fully produced.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.possession.spatial_graph import FrameGraph, build_frame_graph

# -----------------------------------------------------------------------
# Default heuristic weight configuration
# -----------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "opponent_proximity": 0.25,
    "teammate_proximity": 0.15,
    "local_density": 0.15,
    "centrality": 0.10,
    "movement_continuity": 0.15,
    "forward_progress": 0.10,
    "team_compactness": 0.10,
}

# -----------------------------------------------------------------------
# Required canonical tracking columns (anti-leakage whitelist)
# -----------------------------------------------------------------------
_ALLOWED_TRACKING_COLS = {
    "match_id", "frame", "timestamp", "player_id", "team",
    "x", "y", "confidence", "visible",
}

_REQUIRED_TRACKING_COLS = {
    "match_id", "frame", "timestamp", "player_id", "team",
    "x", "y", "visible",
}

# -----------------------------------------------------------------------
# Stage 1 — Per-player heuristic possession score
# -----------------------------------------------------------------------


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if constant."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def score_frame(
    graph: FrameGraph,
    prev_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    prev_velocities: Optional[Dict[str, Tuple[float, float]]] = None,
    dt: float = 0.04,
    weights: Optional[Dict[str, float]] = None,
    pitch_center: Tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """Compute heuristic possession score for each visible player in a frame.

    Parameters
    ----------
    graph : FrameGraph
        Spatial graph for the current frame.
    prev_positions : dict, optional
        {player_id: (x, y)} from the previous frame for velocity computation.
    prev_velocities : dict, optional
        {player_id: (vx, vy)} from the previous frame for acceleration /
        movement-continuity computation.
    dt : float
        Time step between frames in seconds.
    weights : dict, optional
        Feature weight overrides.  Missing keys fall back to DEFAULT_WEIGHTS.
    pitch_center : tuple
        (x, y) of the pitch centre in normalised coordinates.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, team, x, y, raw_score
        Sorted by raw_score descending.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    ndf = graph.node_df

    if len(ndf) == 0:
        return pd.DataFrame(columns=["player_id", "team", "x", "y", "raw_score"])

    n = len(ndf)

    # --- Feature 1: Opponent proximity (inverse — closer = higher) ---
    opp_dist = ndf["nearest_opponent_dist"].values.astype(float)
    # Replace inf with the max finite distance (for single-team frames)
    finite_mask = np.isfinite(opp_dist)
    if finite_mask.any():
        opp_dist[~finite_mask] = np.max(opp_dist[finite_mask]) + 0.01
    else:
        opp_dist[:] = 1.0
    f_opp = _safe_normalize(1.0 / (opp_dist + 1e-6))

    # --- Feature 2: Teammate proximity (inverse — closer = higher) ---
    tm_dist = ndf["nearest_teammate_dist"].values.astype(float)
    finite_mask = np.isfinite(tm_dist)
    if finite_mask.any():
        tm_dist[~finite_mask] = np.max(tm_dist[finite_mask]) + 0.01
    else:
        tm_dist[:] = 1.0
    f_tm = _safe_normalize(1.0 / (tm_dist + 1e-6))

    # --- Feature 3: Local density ---
    density = ndf["local_density"].values.astype(float)
    f_density = _safe_normalize(density)

    # --- Feature 4: Centrality (distance from pitch centre, inverted) ---
    cx, cy = pitch_center
    xs = ndf["x"].values.astype(float)
    ys = ndf["y"].values.astype(float)
    dist_to_center = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    f_central = _safe_normalize(1.0 / (dist_to_center + 1e-6))

    # --- Feature 5: Movement continuity (low acceleration = smooth) ---
    f_movement = np.zeros(n)
    if prev_positions is not None and prev_velocities is not None:
        for i, pid in enumerate(ndf["player_id"]):
            if pid in prev_positions and pid in prev_velocities:
                px, py = prev_positions[pid]
                vx_curr = (xs[i] - px) / dt
                vy_curr = (ys[i] - py) / dt
                vx_prev, vy_prev = prev_velocities[pid]
                # Acceleration magnitude
                ax = (vx_curr - vx_prev) / dt
                ay = (vy_curr - vy_prev) / dt
                accel_mag = np.sqrt(ax ** 2 + ay ** 2)
                # Lower acceleration → higher continuity score (inverse)
                f_movement[i] = 1.0 / (accel_mag + 1e-6)
        f_movement = _safe_normalize(f_movement)

    # --- Feature 6: Forward progress ---
    f_forward = np.zeros(n)
    if prev_positions is not None:
        for i, pid in enumerate(ndf["player_id"]):
            if pid in prev_positions:
                px, py = prev_positions[pid]
                vx = (xs[i] - px) / dt
                vy = (ys[i] - py) / dt
                team = ndf["team"].iloc[i]
                # Home attacks right (x=1), away attacks left (x=0)
                if team == "home":
                    f_forward[i] = max(0.0, vx)
                else:
                    f_forward[i] = max(0.0, -vx)
        f_forward = _safe_normalize(f_forward)

    # --- Feature 7: Team compactness ---
    f_compact = np.zeros(n)
    teams_arr = ndf["team"].values
    for team_label in np.unique(teams_arr):
        team_mask = teams_arr == team_label
        team_xs = xs[team_mask]
        team_ys = ys[team_mask]
        if len(team_xs) > 1:
            spread = np.std(team_xs) + np.std(team_ys)
            # Lower spread → higher compactness → higher score
            compactness = 1.0 / (spread + 1e-6)
        else:
            compactness = 0.0
        f_compact[team_mask] = compactness
    f_compact = _safe_normalize(f_compact)

    # --- Weighted combination ---
    raw_score = (
        w["opponent_proximity"] * f_opp
        + w["teammate_proximity"] * f_tm
        + w["local_density"] * f_density
        + w["centrality"] * f_central
        + w["movement_continuity"] * f_movement
        + w["forward_progress"] * f_forward
        + w["team_compactness"] * f_compact
    )

    result = pd.DataFrame({
        "player_id": ndf["player_id"].values,
        "team": ndf["team"].values,
        "x": xs,
        "y": ys,
        "raw_score": raw_score,
    })
    return result.sort_values("raw_score", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------
# Stage 2 + 3 — Temporal continuity & canonical output
# -----------------------------------------------------------------------


def predict_possession(
    tracking_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    density_radius: float = 0.05,
    knn_k: Optional[int] = 5,
    switch_margin: float = 0.15,
    persistence_window: int = 5,
    min_score_threshold: float = 0.10,
    dt: float = 0.04,
    pitch_center: Tuple[float, float] = (0.5, 0.5),
    frame_graphs: Optional[Dict[int, FrameGraph]] = None,
) -> pd.DataFrame:
    """Predict possession for every frame in a canonical tracking DataFrame.

    ANTI-LEAKAGE: This function accesses ONLY the columns listed in
    _REQUIRED_TRACKING_COLS.  Ball columns, event labels, and future
    frames are never read.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Canonical tracking data (may contain ball/extra columns — ignored).
    weights : dict, optional
        Feature weight overrides for score_frame().
    density_radius : float
        Radius for local density in normalised coordinates.
    knn_k : int or None
        KNN parameter for spatial graph.
    switch_margin : float
        A new possessor must exceed the incumbent by this margin
        to trigger a switch.
    persistence_window : int
        Minimum number of consecutive frames a possessor must hold
        before a switch is allowed.
    min_score_threshold : float
        Minimum score for any player to be considered a possessor.
        Below this, the frame is marked as no-possession.
    dt : float
        Time step between frames in seconds.
    pitch_center : tuple
        Pitch centre in normalised coordinates.

    Returns
    -------
    pd.DataFrame
        Canonical possession DataFrame with columns:
        match_id, frame, timestamp, possessor_id, team,
        possession_score, confidence
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    missing = _REQUIRED_TRACKING_COLS - set(tracking_df.columns)
    if missing:
        raise ValueError(
            f"tracking_df is missing required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # Extract only allowed columns to enforce anti-leakage
    # ------------------------------------------------------------------
    cols_to_use = sorted(_REQUIRED_TRACKING_COLS & set(tracking_df.columns))
    if "confidence" in tracking_df.columns:
        cols_to_use.append("confidence")
    df = tracking_df[cols_to_use].copy()

    # ------------------------------------------------------------------
    # Iterate over frames in order
    # ------------------------------------------------------------------
    frames_sorted = sorted(df["frame"].unique())
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "match_id", "frame", "timestamp", "possessor_id", 
            "team", "possession_score", "confidence"
        ])
    match_id = df["match_id"].iloc[0]

    # State for temporal continuity
    current_possessor: Optional[str] = None
    current_team: Optional[str] = None
    frames_held: int = 0

    # State for velocity / acceleration tracking
    prev_positions: Optional[Dict[str, Tuple[float, float]]] = None
    prev_velocities: Optional[Dict[str, Tuple[float, float]]] = None

    results = []

    for frame_num in frames_sorted:
        frame_data = df[df["frame"] == frame_num]
        ts = frame_data["timestamp"].iloc[0]

        # Reuse graphs when the caller has already built them for this condition.
        graph = frame_graphs.get(frame_num) if frame_graphs is not None else None
        if graph is None:
            graph = build_frame_graph(
                frame_data,
                frame=frame_num,
                timestamp=ts,
                density_radius=density_radius,
                knn_k=knn_k,
            )

        # Score players
        scores_df = score_frame(
            graph,
            prev_positions=prev_positions,
            prev_velocities=prev_velocities,
            dt=dt,
            weights=weights,
            pitch_center=pitch_center,
        )

        # Determine possessor with temporal continuity
        possessor_id = None
        possessor_team = None
        possessor_score = 0.0
        confidence = 0.0

        if len(scores_df) > 0:
            best_player = scores_df.iloc[0]
            best_score = best_player["raw_score"]

            if best_score >= min_score_threshold:
                # Check temporal continuity
                if current_possessor is not None:
                    # Find current possessor's score in this frame
                    curr_row = scores_df[
                        scores_df["player_id"] == current_possessor
                    ]
                    curr_score = (
                        curr_row["raw_score"].values[0]
                        if len(curr_row) > 0
                        else 0.0
                    )

                    if (
                        best_player["player_id"] != current_possessor
                        and frames_held < persistence_window
                    ):
                        # Persistence window not expired — keep current
                        possessor_id = current_possessor
                        possessor_team = current_team
                        possessor_score = curr_score
                    elif (
                        best_player["player_id"] != current_possessor
                        and best_score - curr_score < switch_margin
                    ):
                        # Margin not exceeded — keep current
                        possessor_id = current_possessor
                        possessor_team = current_team
                        possessor_score = curr_score
                    else:
                        # Switch to new possessor
                        possessor_id = best_player["player_id"]
                        possessor_team = best_player["team"]
                        possessor_score = best_score
                else:
                    # No previous possessor — assign best
                    possessor_id = best_player["player_id"]
                    possessor_team = best_player["team"]
                    possessor_score = best_score

                # Normalise confidence from raw score
                all_scores = scores_df["raw_score"].values
                max_score = np.max(all_scores) if len(all_scores) > 0 else 1.0
                confidence = float(
                    possessor_score / max_score if max_score > 0 else 0.0
                )
                confidence = min(confidence, 1.0)

        # Update temporal continuity state
        if possessor_id == current_possessor:
            frames_held += 1
        else:
            current_possessor = possessor_id
            current_team = possessor_team
            frames_held = 1

        results.append({
            "match_id": match_id,
            "frame": frame_num,
            "timestamp": ts,
            "possessor_id": possessor_id,
            "team": possessor_team,
            "possession_score": possessor_score,
            "confidence": confidence,
        })

        # Update velocity tracking for next frame
        visible = frame_data[frame_data["visible"] == True]  # noqa: E712
        current_positions = {}
        current_velocities = {}
        for _, row in visible.iterrows():
            pid = row["player_id"]
            cx, cy = float(row["x"]), float(row["y"])
            current_positions[pid] = (cx, cy)
            if prev_positions is not None and pid in prev_positions:
                px, py = prev_positions[pid]
                current_velocities[pid] = (
                    (cx - px) / dt,
                    (cy - py) / dt,
                )
            else:
                current_velocities[pid] = (0.0, 0.0)

        prev_positions = current_positions
        prev_velocities = current_velocities

    possession_df = pd.DataFrame(results)
    return possession_df


# -----------------------------------------------------------------------
# Stage 4 — Event inference from possession transitions
# -----------------------------------------------------------------------


def infer_events(possession_df: pd.DataFrame) -> pd.DataFrame:
    """Generate candidate events from possession changes.

    Candidate event types:
    - possession_change  : any frame where possessor_id changes
    - pass_candidate     : possessor changes, team stays
    - turnover_candidate : possessor changes, team changes
    - recovery_candidate : team regains possession after losing it

    Parameters
    ----------
    possession_df : pd.DataFrame
        Output of predict_possession().

    Returns
    -------
    pd.DataFrame
        Columns: match_id, frame, timestamp, event_type,
        from_player_id, to_player_id, from_team, to_team
    """
    if len(possession_df) < 2:
        return pd.DataFrame(columns=[
            "match_id", "frame", "timestamp", "event_type",
            "from_player_id", "to_player_id", "from_team", "to_team",
        ])

    pdf = possession_df.sort_values("frame").reset_index(drop=True)
    events = []

    prev_team_before_last_change: Optional[str] = None

    for i in range(1, len(pdf)):
        prev = pdf.iloc[i - 1]
        curr = pdf.iloc[i]

        prev_pid = prev["possessor_id"]
        curr_pid = curr["possessor_id"]

        # Skip if both None or same possessor
        if prev_pid == curr_pid:
            continue
        if prev_pid is None and curr_pid is None:
            continue

        prev_team = prev["team"]
        curr_team = curr["team"]

        # Possession change detected
        event = {
            "match_id": curr["match_id"],
            "frame": curr["frame"],
            "timestamp": curr["timestamp"],
            "from_player_id": prev_pid,
            "to_player_id": curr_pid,
            "from_team": prev_team,
            "to_team": curr_team,
        }

        if prev_team is None or curr_team is None:
            event["event_type"] = "possession_change"
        elif prev_team == curr_team:
            event["event_type"] = "pass_candidate"
        else:
            event["event_type"] = "turnover_candidate"

        events.append(event)

        # Check for recovery: team A → team B → team A
        if (
            prev_team is not None
            and curr_team is not None
            and prev_team != curr_team
            and prev_team_before_last_change is not None
            and curr_team == prev_team_before_last_change
        ):
            recovery_event = event.copy()
            recovery_event["event_type"] = "recovery_candidate"
            events.append(recovery_event)

        # Track team history for recovery detection
        if prev_team != curr_team:
            prev_team_before_last_change = prev_team

    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values("frame").reset_index(drop=True)
    return events_df


# -----------------------------------------------------------------------
# Stage 5 — Evaluation against reference events
# -----------------------------------------------------------------------

# Default temporal tolerances in seconds
DEFAULT_TOLERANCES = [0.5, 1.0, 2.0]


def _match_events_greedy(
    pred_frames: np.ndarray,
    ref_frames: np.ndarray,
    tolerance_frames: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Greedy 1-to-1temporal matching of predicted to reference events.

    Sorted by temporal proximity, each reference event matches at most
    one predicted event.

    Returns
    -------
    (n_matched, list_of_(pred_idx, ref_idx)_pairs)
    """
    if len(pred_frames) == 0 or len(ref_frames) == 0:
        return 0, []

    # Build all candidate pairs sorted by absolute temporal difference
    pairs = []
    for pi, pf in enumerate(pred_frames):
        for ri, rf in enumerate(ref_frames):
            diff = abs(int(pf) - int(rf))
            if diff <= tolerance_frames:
                pairs.append((diff, pi, ri))
    pairs.sort(key=lambda x: x[0])

    matched_pred = set()
    matched_ref = set()
    matches = []

    for diff, pi, ri in pairs:
        if pi not in matched_pred and ri not in matched_ref:
            matched_pred.add(pi)
            matched_ref.add(ri)
            matches.append((pi, ri))

    return len(matches), matches


def evaluate_predictions(
    predicted_events: pd.DataFrame,
    reference_events: pd.DataFrame,
    fps: float = 25.0,
    tolerances_sec: Optional[List[float]] = None,
    event_type_mapping: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Evaluate predicted candidate events against reference events.

    Parameters
    ----------
    predicted_events : pd.DataFrame
        Output of infer_events().
    reference_events : pd.DataFrame
        Canonical event DataFrame (from metrica_event_parser).
    fps : float
        Frames per second for converting tolerances to frames.
    tolerances_sec : list[float], optional
        Temporal tolerances in seconds.  Defaults to [0.5, 1.0, 2.0].
    event_type_mapping : dict, optional
        Mapping from predicted event types to lists of reference event
        types that count as matches.  Defaults to sensible Metrica mappings:
        - pass_candidate → ['PASS']
        - turnover_candidate → ['BALL LOST', 'BALL OUT']
        - recovery_candidate → ['RECOVERY']

    Returns
    -------
    pd.DataFrame
        Columns: event_type, tolerance_sec, tolerance_frames,
        n_predicted, n_reference, n_matched,
        precision, recall, f1,
        mean_temporal_diff_frames, median_temporal_diff_frames
    """
    if tolerances_sec is None:
        tolerances_sec = DEFAULT_TOLERANCES

    if event_type_mapping is None:
        event_type_mapping = {
            "pass_candidate": ["PASS"],
            "turnover_candidate": ["BALL LOST", "BALL OUT"],
            "recovery_candidate": ["RECOVERY"],
        }

    results = []

    for pred_type, ref_types in event_type_mapping.items():
        # Filter predicted events
        pred_mask = predicted_events["event_type"] == pred_type
        pred_subset = predicted_events[pred_mask]

        # Filter reference events
        ref_mask = reference_events["event_type"].isin(ref_types)
        ref_subset = reference_events[ref_mask]

        pred_frames = pred_subset["frame"].values if len(pred_subset) > 0 else np.array([])
        ref_frames = ref_subset["start_frame"].values if len(ref_subset) > 0 else np.array([])

        for tol_sec in tolerances_sec:
            tol_frames = int(round(tol_sec * fps))

            n_matched, match_pairs = _match_events_greedy(
                pred_frames, ref_frames, tol_frames
            )

            n_pred = len(pred_frames)
            n_ref = len(ref_frames)

            precision = n_matched / n_pred if n_pred > 0 else 0.0
            recall = n_matched / n_ref if n_ref > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Temporal difference statistics for matched pairs
            if match_pairs:
                diffs = [
                    abs(int(pred_frames[pi]) - int(ref_frames[ri]))
                    for pi, ri in match_pairs
                ]
                mean_diff = np.mean(diffs)
                median_diff = np.median(diffs)
            else:
                mean_diff = np.nan
                median_diff = np.nan

            results.append({
                "event_type": pred_type,
                "tolerance_sec": tol_sec,
                "tolerance_frames": tol_frames,
                "n_predicted": n_pred,
                "n_reference": n_ref,
                "n_matched": n_matched,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_temporal_diff_frames": round(mean_diff, 2)
                if not np.isnan(mean_diff)
                else np.nan,
                "median_temporal_diff_frames": round(median_diff, 2)
                if not np.isnan(median_diff)
                else np.nan,
            })

    return pd.DataFrame(results)


def possession_summary(possession_df: pd.DataFrame) -> Dict[str, object]:
    """Compute high-level summary statistics from possession predictions.

    Returns
    -------
    dict with keys: total_frames, possessing_frames, no_possession_frames,
    possession_pct, n_switches, home_pct, away_pct
    """
    total = len(possession_df)
    possessing = possession_df["possessor_id"].notna().sum()
    no_poss = total - possessing

    n_switches = 0
    for i in range(1, total):
        prev = possession_df.iloc[i - 1]["possessor_id"]
        curr = possession_df.iloc[i]["possessor_id"]
        if prev != curr and not (prev is None and curr is None):
            n_switches += 1

    teams = possession_df[possession_df["team"].notna()]["team"]
    home_count = (teams == "home").sum()
    away_count = (teams == "away").sum()
    team_total = home_count + away_count

    return {
        "total_frames": total,
        "possessing_frames": int(possessing),
        "no_possession_frames": int(no_poss),
        "possession_pct": round(possessing / total * 100, 1) if total > 0 else 0,
        "n_switches": n_switches,
        "home_pct": round(home_count / team_total * 100, 1) if team_total > 0 else 0,
        "away_pct": round(away_count / team_total * 100, 1) if team_total > 0 else 0,
    }
