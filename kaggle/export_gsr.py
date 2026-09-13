#!/usr/bin/env python3
"""
Canonical GSR Export and Tracking Quality Assessment Module.

Converts raw TrackLab / SoccerNet-GSR outputs into the project's Canonical
Tracking Schema (v0.2.0):
  match_id, frame, timestamp, player_id, team, x, y, confidence, visible

Calculates tracking quality metrics, creates trajectory diagnostic plots,
computes SHA256 checksums, and produces the complete export package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
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


def sha256_file(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_canonical_dataframe(df: pd.DataFrame) -> None:
    """Validates that a DataFrame conforms strictly to Canonical Tracking Schema v0.2.0."""
    missing_cols = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required canonical columns: {missing_cols}")

    if df["match_id"].isna().any() or (df["match_id"] == "").any():
        raise ValueError("match_id must be populated for all rows.")

    if not pd.api.types.is_integer_dtype(df["frame"]):
        raise ValueError("frame column must be of integer data type.")

    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
        raise ValueError("timestamp column must be numeric.")

    if not pd.api.types.is_bool_dtype(df["visible"]):
        raise ValueError("visible column must be boolean.")

    if not pd.api.types.is_numeric_dtype(df["confidence"]):
        raise ValueError("confidence column must be numeric.")

    if (df["confidence"] < 0.0).any() or (df["confidence"] > 1.0).any():
        raise ValueError("confidence values must be bounded within [0.0, 1.0].")

    valid_teams = {"home", "away", "referee"}
    invalid_teams = set(df["team"].unique()) - valid_teams
    if invalid_teams:
        raise ValueError(f"Invalid team values found: {invalid_teams}. Allowed: {valid_teams}")

    dups = df.duplicated(subset=["match_id", "frame", "team", "player_id"])
    if dups.any():
        raise ValueError(f"Found {dups.sum()} duplicate player entries per frame.")


def stitch_tracklets(
    df: pd.DataFrame,
    fps: float = 25.0,
    max_gap_frames: int = 75,
    max_speed_mps: float = 9.5,
    min_track_len: int = 3,
) -> pd.DataFrame:
    """
    Consolidates fragmented tracklets belonging to the same player entity.

    Rules:
      1. Merges non-overlapping tracklets of the same team if:
         a) Jersey numbers match (MMOCR high-confidence link).
         b) Implied spatial velocity <= max_speed_mps (physical sprinting bound).
      2. Re-maps player_ids to stable unified identifiers.
      3. Filters short transient flickers (< min_track_len) that could not be linked.
    """
    if df.empty:
        return df

    tracks: Dict[str, Dict[str, Any]] = {}
    for pid, g in df.groupby("player_id"):
        valid_coords = g.dropna(subset=["x", "y"])
        if valid_coords.empty:
            continue
        first_row = valid_coords.iloc[0]
        last_row = valid_coords.iloc[-1]
        team_val = g["team"].mode().iloc[0] if not g["team"].empty else "home"
        jersey_val = None
        if "jersey" in g.columns:
            non_null_j = g["jersey"].dropna()
            if not non_null_j.empty:
                jersey_val = str(non_null_j.mode().iloc[0])

        tracks[str(pid)] = {
            "pid": str(pid),
            "start_frame": int(g["frame"].min()),
            "end_frame": int(g["frame"].max()),
            "start_x": float(first_row["x"]),
            "start_y": float(first_row["y"]),
            "end_x": float(last_row["x"]),
            "end_y": float(last_row["y"]),
            "team": team_val,
            "jersey": jersey_val,
            "count": len(g),
        }

    # Greedily link tracklets forward in time
    mapping: Dict[str, str] = {pid: pid for pid in tracks}
    sorted_pids = sorted(tracks.keys(), key=lambda p: tracks[p]["start_frame"])

    for i, p1 in enumerate(sorted_pids):
        t1_pid = mapping[p1]
        t1 = tracks[t1_pid]
        best_match = None
        best_cost = float("inf")

        for p2 in sorted_pids[i + 1:]:
            t2_pid = mapping[p2]
            if t1_pid == t2_pid:
                continue
            t2 = tracks[t2_pid]
            # Must not overlap in time
            if t1["end_frame"] >= t2["start_frame"]:
                continue
            gap = t2["start_frame"] - t1["end_frame"]
            if gap > max_gap_frames:
                continue
            # Must be same team
            if t1["team"] != t2["team"]:
                continue

            jersey_match = (
                t1["jersey"] is not None
                and t2["jersey"] is not None
                and t1["jersey"] == t2["jersey"]
            )

            dx = (t2["start_x"] - t1["end_x"]) * 105.0
            dy = (t2["start_y"] - t1["end_y"]) * 68.0
            dist = np.sqrt(dx**2 + dy**2)
            dt = gap / fps
            speed = dist / max(dt, 0.04)

            if jersey_match or speed <= max_speed_mps:
                cost = dist * 0.2 if jersey_match else dist
                if cost < best_cost:
                    best_cost = cost
                    best_match = p2

        if best_match:
            target_pid = mapping[p1]
            source_pid = mapping[best_match]
            for k, v in mapping.items():
                if v == source_pid:
                    mapping[k] = target_pid
            tracks[target_pid]["end_frame"] = max(tracks[target_pid]["end_frame"], tracks[source_pid]["end_frame"])
            tracks[target_pid]["end_x"] = tracks[source_pid]["end_x"]
            tracks[target_pid]["end_y"] = tracks[source_pid]["end_y"]

    df_stitched = df.copy()
    df_stitched["player_id"] = df_stitched["player_id"].astype(str).map(mapping).fillna(df_stitched["player_id"])

    # Filter out unlinked short transient flickers (< min_track_len)
    if min_track_len > 1:
        stitched_counts = df_stitched["player_id"].value_counts()
        keep_pids = set(stitched_counts[stitched_counts >= min_track_len].index)
        df_stitched = df_stitched[df_stitched["player_id"].isin(keep_pids)].copy()

    return df_stitched.reset_index(drop=True)


def convert_raw_tracklab_to_canonical(
    raw_df: pd.DataFrame,
    match_id: str,
    fps: float = 25.0,
    normalize_coords: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    team_mapping: Optional[Dict[str, str]] = None,
    stitch: bool = True,
    max_gap_frames: int = 75,
    max_speed_mps: float = 9.5,
    min_track_len: int = 3,
    filter_referees: bool = False,
) -> pd.DataFrame:
    """
    Converts raw TrackLab prediction DataFrame to canonical tracking DataFrame.

    Avoids silent fallbacks:
      - Unpacks nested 'attributes' (team, role, jersey) so teams and referees are detected.
      - Resolves team labels using track-level majority voting and pitch half.
      - Preserves natural out-of-bounds player positions without artificial clamping to 0.0 or 1.0.
      - Discards true homography calibration singularities (|norm| > 1.20) as NaN (visible=False).
      - Consolidates fragmented tracklets and suppresses short transient detection flickers.
    """
    # Defensive preprocessing: filter for object detections only
    if "supercategory" in raw_df.columns:
        raw_df = raw_df[raw_df["supercategory"] == "object"].copy()
    if "track_id" in raw_df.columns:
        raw_df = raw_df[raw_df["track_id"].notna()].copy()

    # Unpack nested attributes (role, team, jersey, confidence) if present
    if "attributes" in raw_df.columns:
        def _get_attr(attr_val, key):
            if isinstance(attr_val, dict):
                return attr_val.get(key)
            return None

        for col in ("team", "role", "jersey", "confidence"):
            extracted = raw_df["attributes"].apply(lambda a: _get_attr(a, col))
            if col not in raw_df.columns or raw_df[col].isna().all():
                raw_df[col] = extracted
            else:
                raw_df[col] = raw_df[col].fillna(extracted)

    # If bbox_pitch is present as dict/list, unpack into pitch_x and pitch_y
    if "bbox_pitch" in raw_df.columns:
        def _get_x(bp):
            if isinstance(bp, dict):
                return bp.get("x_bottom_middle", bp.get("x", np.nan))
            if isinstance(bp, (list, tuple)) and len(bp) >= 1:
                return bp[0]
            return np.nan

        def _get_y(bp):
            if isinstance(bp, dict):
                return bp.get("y_bottom_middle", bp.get("y", np.nan))
            if isinstance(bp, (list, tuple)) and len(bp) >= 2:
                return bp[1]
            return np.nan

        if "pitch_x" not in raw_df.columns and "x" not in raw_df.columns:
            raw_df["pitch_x"] = raw_df["bbox_pitch"].apply(_get_x)
            raw_df["pitch_y"] = raw_df["bbox_pitch"].apply(_get_y)

    # Map column names dynamically based on TrackLab / SoccerNet output conventions
    frame_col = "frame" if "frame" in raw_df.columns else ("image_id" if "image_id" in raw_df.columns else "frame_idx")
    track_col = "track_id" if "track_id" in raw_df.columns else ("player_id" if "player_id" in raw_df.columns else "id")
    team_col = "team" if "team" in raw_df.columns else "team_id"
    role_col = "role" if "role" in raw_df.columns else "role_id"
    jersey_col = "jersey" if "jersey" in raw_df.columns else "jersey_number"
    x_col = "pitch_x" if "pitch_x" in raw_df.columns else ("x" if "x" in raw_df.columns else "x_bottom_middle")
    y_col = "pitch_y" if "pitch_y" in raw_df.columns else ("y" if "y" in raw_df.columns else "y_bottom_middle")
    
    conf_candidates = ["confidence", "detection_confidence", "bbox_confidence", "conf", "score", "det_score", "track_confidence"]
    conf_col = next((c for c in conf_candidates if c in raw_df.columns), None)

    # Domain-aware track-level team & referee consensus mapping
    track_meta: Dict[str, Dict[str, Any]] = {}
    for tid, group in raw_df.groupby(track_col):
        tid_str = str(int(tid)) if isinstance(tid, (int, float, np.integer, np.floating)) and not pd.isna(tid) else str(tid)
        t_vals = group[team_col].dropna() if team_col in group.columns else pd.Series([], dtype=object)
        r_vals = group[role_col].dropna() if role_col in group.columns else pd.Series([], dtype=object)
        
        # Calculate track-level spatial coordinates
        x_vals = group[x_col].dropna() if x_col in group.columns else pd.Series([], dtype=float)
        y_vals = group[y_col].dropna() if y_col in group.columns else pd.Series([], dtype=float)
        
        mx = float(x_vals.mean()) if not x_vals.empty else 0.5
        my = float(y_vals.mean()) if not y_vals.empty else 0.5
        # If in meters, convert to normalized [0, 1]
        if abs(mx) > 1.5:
            norm_mx = (mx + pitch_length / 2.0) / pitch_length
            norm_my = (my + pitch_width / 2.0) / pitch_width
        else:
            norm_mx, norm_my = mx, my
            
        raw_team_mode = str(t_vals.mode().iloc[0]).lower().strip() if not t_vals.empty else ""
        raw_role_mode = str(r_vals.mode().iloc[0]).lower().strip() if not r_vals.empty else ""
        
        track_meta[tid_str] = {
            "tid": tid_str,
            "mean_x": norm_mx,
            "mean_y": norm_my,
            "raw_team": raw_team_mode,
            "raw_role": raw_role_mode,
            "count": len(group),
        }

    track_team_consensus: Dict[str, str] = {}
    
    # 1. Sideline Linesmen / Assistant Referees: stationed on touchlines (|y| <= 0.05 or >= 0.95)
    for tid_str, meta in track_meta.items():
        if meta["mean_y"] < 0.05 or meta["mean_y"] > 0.95:
            track_team_consensus[tid_str] = "referee"
            
    # 2. Main On-Pitch Referee: isolated official with explicit role 'referee'
    remaining = {k: v for k, v in track_meta.items() if k not in track_team_consensus}
    explicit_refs = [k for k, v in remaining.items() if "ref" in v["raw_role"]]
    if 0 < len(explicit_refs) <= 2:
        for k in explicit_refs:
            track_team_consensus[k] = "referee"
            
    # 3. All remaining tracks MUST belong to the two competing teams (Home vs Away)
    players = {k: v for k, v in track_meta.items() if k not in track_team_consensus}
    
    # Compute average X position per cluster/raw_team
    cluster_xs: Dict[str, List[float]] = {}
    for meta in players.values():
        rt = meta["raw_team"]
        cluster_xs.setdefault(rt, []).append(meta["mean_x"])
    cluster_means = {k: float(np.mean(v)) for k, v in cluster_xs.items()}
    
    for tid_str, meta in players.items():
        rt = meta["raw_team"]
        mx = meta["mean_x"]
        if "left" in rt or "home" in rt:
            track_team_consensus[tid_str] = "home"
        elif "right" in rt or "away" in rt:
            track_team_consensus[tid_str] = "away"
        else:
            # Use cluster mean or individual mean
            c_mx = cluster_means.get(rt, mx)
            track_team_consensus[tid_str] = "home" if c_mx < 0.52 else "away"

    # Check if frame numbers are raw encoded image filenames (e.g., 2040000001)
    raw_frames = pd.to_numeric(raw_df[frame_col], errors="coerce").dropna().astype(int)
    min_frame_val = raw_frames.min() if not raw_frames.empty else 1
    is_encoded_frame = min_frame_val > 100000

    rows: List[Dict[str, Any]] = []

    for idx, r in raw_df.iterrows():
        raw_f = int(r[frame_col])
        if is_encoded_frame:
            f_val = raw_f - min_frame_val + 1
            ts_val = round((f_val - 1) / fps, 4)
        else:
            f_val = raw_f
            ts_val = round(float(f_val) / fps, 4)

        pid_raw = r[track_col]
        pid_str = str(int(pid_raw)) if isinstance(pid_raw, (int, float, np.integer, np.floating)) and not pd.isna(pid_raw) else str(pid_raw)

        # Team value from track consensus
        team_val = track_team_consensus.get(pid_str)
        if not team_val:
            team_val = _map_team_value(r.get(team_col), r.get(role_col)) or "home"

        # Coordinates
        raw_x = float(r[x_col]) if x_col in r and not pd.isna(r[x_col]) else np.nan
        raw_y = float(r[y_col]) if y_col in r and not pd.isna(r[y_col]) else np.nan

        if not np.isnan(raw_x) and not np.isnan(raw_y):
            if normalize_coords:
                # Metric pitch coordinates centered at (0, 0): [-L/2, +L/2] -> [0, 1]
                if abs(raw_x) > 1.5 or abs(raw_y) > 1.5:
                    norm_x = (raw_x + pitch_length / 2.0) / pitch_length
                    norm_y = (raw_y + pitch_width / 2.0) / pitch_width
                else:
                    norm_x = raw_x
                    norm_y = raw_y
            else:
                norm_x = raw_x
                norm_y = raw_y

            # Reject severe homography calibration divergence (e.g. extreme values like -133)
            # Allow natural out-of-bounds positions without artificial wall clamping to 0.0 or 1.0
            if norm_x < -0.20 or norm_x > 1.20 or norm_y < -0.20 or norm_y > 1.20:
                norm_x = np.nan
                norm_y = np.nan
                visible = False
            else:
                norm_x = round(float(norm_x), 6)
                norm_y = round(float(norm_y), 6)
                visible = True
        else:
            norm_x = np.nan
            norm_y = np.nan
            visible = False

        # Confidence: clamp valid scores; default to 1.0 for detected tracklets if uncalibrated
        if conf_col and conf_col in r and not pd.isna(r[conf_col]) and float(r[conf_col]) > 0:
            conf_clamped = round(max(0.0, min(1.0, float(r[conf_col]))), 4)
        else:
            conf_clamped = 1.0 if visible else 0.0

        jersey_val = str(r[jersey_col]) if jersey_col in r and pd.notna(r[jersey_col]) else None

        rows.append({
            "match_id": match_id,
            "frame": f_val,
            "timestamp": ts_val,
            "player_id": pid_str,
            "team": team_val,
            "x": norm_x,
            "y": norm_y,
            "confidence": conf_clamped,
            "visible": visible,
            "jersey": jersey_val,
        })

    canonical_df = pd.DataFrame(rows)

    # Stitch fragmented tracklets and remove short flickers
    if stitch and not canonical_df.empty:
        canonical_df = stitch_tracklets(
            canonical_df,
            fps=fps,
            max_gap_frames=max_gap_frames,
            max_speed_mps=max_speed_mps,
            min_track_len=min_track_len,
        )

    # Optional referee filtering for player-only tracking
    if filter_referees and not canonical_df.empty:
        canonical_df = canonical_df[canonical_df["team"] != "referee"].copy()

    canonical_df = canonical_df.drop_duplicates(subset=["match_id", "frame", "player_id"])
    canonical_df = canonical_df.sort_values(by=["frame", "player_id"]).reset_index(drop=True)
    canonical_df = canonical_df[CANONICAL_COLUMNS]
    validate_canonical_dataframe(canonical_df)
    return canonical_df


def compute_tracking_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive tracking quality metrics for the sequence."""
    total_obs = len(df)
    total_frames = df["frame"].nunique() if not df.empty else 0
    unique_players = df["player_id"].nunique() if not df.empty else 0

    track_lengths = df.groupby("player_id")["frame"].count() if not df.empty else pd.Series([], dtype=int)
    min_track = int(track_lengths.min()) if not track_lengths.empty else 0
    max_track = int(track_lengths.max()) if not track_lengths.empty else 0
    avg_track = float(track_lengths.mean()) if not track_lengths.empty else 0.0

    missing_coords = df["x"].isna() | df["y"].isna() if not df.empty else pd.Series([], dtype=bool)
    missingness_rate = float(missing_coords.mean()) if total_obs > 0 else 0.0
    detected_rate = float(df["visible"].mean()) if total_obs > 0 else 0.0
    mean_conf = float(df["confidence"].mean()) if total_obs > 0 else 0.0

    home_players = int(df[df["team"] == "home"]["player_id"].nunique()) if not df.empty else 0
    away_players = int(df[df["team"] == "away"]["player_id"].nunique()) if not df.empty else 0
    referees = int(df[df["team"] == "referee"]["player_id"].nunique()) if not df.empty else 0

    summary_data = [{
        "match_id": str(df["match_id"].iloc[0]) if not df.empty else "N/A",
        "total_frames": total_frames,
        "total_observations": total_obs,
        "unique_players": unique_players,
        "home_players": home_players,
        "away_players": away_players,
        "referees": referees,
        "missingness_rate": missingness_rate,
        "detection_rate": detected_rate,
        "mean_confidence": mean_conf,
        "min_track_length": min_track,
        "max_track_length": max_track,
        "avg_track_length": avg_track,
        "x_min": float(df["x"].min()) if not df.empty and not df["x"].isna().all() else np.nan,
        "x_max": float(df["x"].max()) if not df.empty and not df["x"].isna().all() else np.nan,
        "y_min": float(df["y"].min()) if not df.empty and not df["y"].isna().all() else np.nan,
        "y_max": float(df["y"].max()) if not df.empty and not df["y"].isna().all() else np.nan,
    }]
    return pd.DataFrame(summary_data)


def export_gsr_package(
    raw_output_path: Path,
    output_dir: Path,
    match_id: str = "gsr_valid_sngs04",
    fps: float = 25.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    stitch: bool = True,
    max_gap_frames: int = 75,
    min_track_len: int = 3,
    filter_referees: bool = False,
) -> Dict[str, Any]:
    """Reads raw TrackLab output, transforms to canonical schema, and produces export package."""
    exports_dir = output_dir / "exports"
    figures_dir = output_dir / "figures"
    exports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading raw TrackLab output from: {raw_output_path}")
    if str(raw_output_path).endswith(".csv"):
        raw_df = pd.read_csv(raw_output_path)
    elif str(raw_output_path).endswith(".json"):
        with open(raw_output_path, "r", encoding="utf-8") as jf:
            raw_data = json.load(jf)
        if isinstance(raw_data, list):
            raw_df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            if any(k in raw_data for k in ("run_manifest", "dataset_name", "git_commit", "system")):
                raise ValueError(
                    f"Selected file '{raw_output_path}' is a manifest/environment file, not TrackLab predictions. "
                    "Inference did not finish successfully. Please check results/gsr_kaggle/logs/ for details."
                )
            if "predictions" in raw_data:
                preds = raw_data["predictions"]
                if isinstance(preds, list):
                    raw_df = pd.DataFrame(preds)
                elif isinstance(preds, dict):
                    flat_rows = []
                    for img_id, dets in preds.items():
                        if isinstance(dets, list):
                            for d in dets:
                                if isinstance(d, dict):
                                    row = dict(d)
                                    row.setdefault("image_id", img_id)
                                    flat_rows.append(row)
                    raw_df = pd.DataFrame(flat_rows)
                else:
                    raw_df = pd.DataFrame(raw_data)
            else:
                flat_rows = []
                for k, v in raw_data.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                row = dict(item)
                                row.setdefault("frame", k)
                                flat_rows.append(row)
                raw_df = pd.DataFrame(flat_rows) if flat_rows else pd.DataFrame([raw_data])
    elif str(raw_output_path).endswith((".pklz", ".pkl")):
        import gzip, pickle
        opener = gzip.open if str(raw_output_path).endswith(".pklz") else open
        with opener(raw_output_path, "rb") as pf:
            tracker_state = pickle.load(pf)
        if hasattr(tracker_state, "detections"):
            raw_df = tracker_state.detections
        elif isinstance(tracker_state, dict) and "detections" in tracker_state:
            raw_df = tracker_state["detections"]
        else:
            raw_df = pd.DataFrame(tracker_state)
    else:
        raise ValueError(f"Unsupported file format: {raw_output_path}")

    # Convert to canonical
    canonical_df = convert_raw_tracklab_to_canonical(
        raw_df,
        match_id=match_id,
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        stitch=stitch,
        max_gap_frames=max_gap_frames,
        min_track_len=min_track_len,
        filter_referees=filter_referees,
    )
    canonical_csv_path = exports_dir / "gsr_tracking_canonical.csv"
    canonical_df.to_csv(canonical_csv_path, index=False)
    print(f"[INFO] Exported canonical tracking CSV ({len(canonical_df)} rows) to: {canonical_csv_path}")

    # Compute quality metrics
    summary_df = compute_tracking_quality_metrics(canonical_df)
    summary_csv_path = output_dir / "tracking_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[INFO] Saved tracking summary metrics to: {summary_csv_path}")

    # Generate trajectory plot
    plot_path = figures_dir / "player_trajectories.png"
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        for pid in canonical_df["player_id"].unique()[:10]:
            p_data = canonical_df[canonical_df["player_id"] == pid]
            ax.plot(p_data["x"], p_data["y"], label=f"Player {pid}", alpha=0.7)
        ax.set_title(f"Reconstructed GSR Trajectories ({match_id})")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Trajectory plot saved to: {plot_path}")
    except ImportError:
        pass

    # Build export manifest with SHA256
    manifest: Dict[str, Any] = {
        "timestamp_export": datetime.now(timezone.utc).isoformat(),
        "match_id": match_id,
        "fps": fps,
        "pitch_dimensions": {"length": pitch_length, "width": pitch_width},
        "canonical_csv": {
            "path": str(canonical_csv_path),
            "rows": len(canonical_df),
            "sha256": sha256_file(canonical_csv_path),
        },
        "tracking_summary_csv": {
            "path": str(summary_csv_path),
            "sha256": sha256_file(summary_csv_path),
        },
        "summary_metrics": summary_df.iloc[0].to_dict(),
    }
    manifest_path = exports_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"[INFO] Export manifest saved to: {manifest_path}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical GSR Export and Tracking Quality Assessment")
    parser.add_argument("--raw-output", type=str, required=True, help="Path to raw TrackLab predictions (CSV or JSON)")
    parser.add_argument("--output-dir", type=str, default="results/gsr_kaggle", help="Output directory for exports")
    parser.add_argument("--match-id", type=str, default="gsr_valid_sngs04", help="Canonical match identifier")
    parser.add_argument("--fps", type=float, default=25.0, help="Frame rate")
    parser.add_argument("--pitch-length", type=float, default=105.0, help="Field length in meters")
    parser.add_argument("--pitch-width", type=float, default=68.0, help="Field width in meters")
    parser.add_argument("--no-stitch", action="store_true", help="Disable tracklet consolidation")
    parser.add_argument("--max-gap-frames", type=int, default=75, help="Max frame gap for tracklet stitching (default: 75 / 3s)")
    parser.add_argument("--min-track-len", type=int, default=3, help="Minimum tracklet length to filter transient flickers (default: 3)")
    parser.add_argument("--filter-referees", action="store_true", help="Exclude referee observations from canonical output")
    args = parser.parse_args()

    raw_path = Path(args.raw_output)
    out_dir = Path(args.output_dir)

    if not raw_path.exists():
        print(f"[ERROR] Raw output file does not exist: {raw_path}", file=sys.stderr)
        return 1

    export_gsr_package(
        raw_output_path=raw_path,
        output_dir=out_dir,
        match_id=args.match_id,
        fps=args.fps,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        stitch=not args.no_stitch,
        max_gap_frames=args.max_gap_frames,
        min_track_len=args.min_track_len,
        filter_referees=args.filter_referees,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
