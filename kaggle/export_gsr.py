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
        raise ValueError(f"Invalid team values found: {invalid_teams}")

    dups = df.duplicated(subset=["match_id", "frame", "team", "player_id"])
    if dups.any():
        raise ValueError(f"Found {dups.sum()} duplicate player entries per frame.")


def convert_raw_tracklab_to_canonical(
    raw_df: pd.DataFrame,
    match_id: str,
    fps: float = 25.0,
    normalize_coords: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> pd.DataFrame:
    """
    Converts raw TrackLab prediction DataFrame to canonical tracking DataFrame.

    Expected raw columns from TrackLab:
      frame / image_id, track_id, bbox_ltwh / bbox, team, x_bottom_middle / pitch_x, pitch_y, confidence
    """
    rows: List[Dict[str, Any]] = []

    # Map column names dynamically
    frame_col = "frame" if "frame" in raw_df.columns else ("image_id" if "image_id" in raw_df.columns else "frame_idx")
    track_col = "track_id" if "track_id" in raw_df.columns else ("player_id" if "player_id" in raw_df.columns else "id")
    team_col = "team" if "team" in raw_df.columns else "team_id"
    x_col = "pitch_x" if "pitch_x" in raw_df.columns else ("x" if "x" in raw_df.columns else "x_bottom_middle")
    y_col = "pitch_y" if "pitch_y" in raw_df.columns else ("y" if "y" in raw_df.columns else "y_bottom_middle")
    conf_col = "confidence" if "confidence" in raw_df.columns else ("detection_confidence" if "detection_confidence" in raw_df.columns else "score")

    for _, r in raw_df.iterrows():
        f_val = int(r[frame_col])
        pid_raw = r[track_col]
        pid_str = str(int(pid_raw)) if isinstance(pid_raw, (int, float, np.integer, np.floating)) and not pd.isna(pid_raw) else str(pid_raw)

        # Team mapping
        raw_team = str(r[team_col]).lower() if team_col in r and not pd.isna(r[team_col]) else "home"
        if "away" in raw_team or raw_team in ["1", "away"]:
            team_val = "away"
        elif "ref" in raw_team or raw_team in ["2", "referee"]:
            team_val = "referee"
        else:
            team_val = "home"

        # Coordinates
        raw_x = r[x_col] if x_col in r and not pd.isna(r[x_col]) else np.nan
        raw_y = r[y_col] if y_col in r and not pd.isna(r[y_col]) else np.nan

        if not np.isnan(raw_x) and not np.isnan(raw_y):
            if normalize_coords:
                # If metric pitch coordinates centered at (0, 0): [-L/2, +L/2] -> [0, 1]
                if raw_x < 0 or raw_x > 1.0 or raw_y < 0 or raw_y > 1.0:
                    norm_x = (float(raw_x) + pitch_length / 2.0) / pitch_length
                    norm_y = (float(raw_y) + pitch_width / 2.0) / pitch_width
                else:
                    norm_x = float(raw_x)
                    norm_y = float(raw_y)
            else:
                norm_x = float(raw_x)
                norm_y = float(raw_y)
        else:
            norm_x = np.nan
            norm_y = np.nan

        conf = float(r[conf_col]) if conf_col in r and not pd.isna(r[conf_col]) else 1.0
        conf_clamped = max(0.0, min(1.0, conf))
        visible = not np.isnan(norm_x) and not np.isnan(norm_y) and conf_clamped > 0.1

        rows.append({
            "match_id": match_id,
            "frame": f_val,
            "timestamp": round(float(f_val) / fps, 4),
            "player_id": pid_str,
            "team": team_val,
            "x": norm_x,
            "y": norm_y,
            "confidence": conf_clamped,
            "visible": visible,
        })

    canonical_df = pd.DataFrame(rows)[CANONICAL_COLUMNS]
    validate_canonical_dataframe(canonical_df)
    return canonical_df


def compute_tracking_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes comprehensive tracking quality metrics for the sequence."""
    total_obs = len(df)
    total_frames = df["frame"].nunique()
    unique_players = df["player_id"].nunique()

    # Track length statistics
    track_lengths = df.groupby("player_id")["frame"].count()
    min_track = int(track_lengths.min()) if not track_lengths.empty else 0
    max_track = int(track_lengths.max()) if not track_lengths.empty else 0
    avg_track = float(track_lengths.mean()) if not track_lengths.empty else 0.0

    # Missingness and coordinate validity
    missing_coords = df["x"].isna() | df["y"].isna()
    missingness_rate = float(missing_coords.mean()) if total_obs > 0 else 0.0
    detected_rate = float(df["visible"].mean()) if total_obs > 0 else 0.0
    mean_conf = float(df["confidence"].mean()) if total_obs > 0 else 0.0

    # Summary table
    summary_data = [{
        "match_id": str(df["match_id"].iloc[0]) if not df.empty else "N/A",
        "total_frames": total_frames,
        "total_observations": total_obs,
        "unique_players": unique_players,
        "missingness_rate": missingness_rate,
        "detection_rate": detected_rate,
        "mean_confidence": mean_conf,
        "min_track_length": min_track,
        "max_track_length": max_track,
        "avg_track_length": avg_track,
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
    }]
    return pd.DataFrame(summary_data)


def export_gsr_package(
    raw_output_path: Path,
    output_dir: Path,
    match_id: str = "gsr_valid_sngs04",
    fps: float = 25.0,
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
        raw_df = pd.DataFrame(raw_data)
    else:
        raise ValueError(f"Unsupported file format: {raw_output_path}")

    # Convert to canonical
    canonical_df = convert_raw_tracklab_to_canonical(raw_df, match_id=match_id, fps=fps)
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
    args = parser.parse_args()

    raw_path = Path(args.raw_output)
    out_dir = Path(args.output_dir)

    if not raw_path.exists():
        print(f"[ERROR] Raw output file does not exist: {raw_path}", file=sys.stderr)
        return 1

    export_gsr_package(raw_output_path=raw_path, output_dir=out_dir, match_id=args.match_id, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
