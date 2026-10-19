"""Step 85: Cross-Dataset Validation Framework.

Evaluates and compares tracking dynamics, ball-free possession characteristics,
and Tactical Advantage Score (TAS) distributions across heterogeneous datasets:
  1. Metrica Sports (Optical / Broadcast Tracking, 25 FPS)
  2. SkillCorner Open Data (Computer Vision Tracking, 10 FPS)
  3. Real Video Perception Output (YOLOv8 + ByteTrack + 2D Homography, 25 FPS)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd

from src.data.metrica_parser import load_metrica_match
from src.possession.possession_baseline import predict_possession
from src.possession.spatial_graph import build_frame_graph
from src.tactics.tactical_features import extract_tactical_features
from src.tactics.tactical_score import calculate_tactical_score


def _analyze_dataset_dynamics(df: pd.DataFrame, dataset_name: str, fps: float = 25.0) -> Dict[str, Any]:
    """Extract kinematic, possession, and tactical distributions from canonical tracking."""
    # Filter valid coordinates
    valid = df.dropna(subset=["x", "y"]).copy()
    if valid.empty:
        return {"dataset": dataset_name, "error": "No valid observations"}

    # Kinematics
    valid_sorted = valid.sort_values(by=["player_id", "frame"]).copy()
    valid_sorted["dt"] = 1.0 / fps
    valid_sorted["dx"] = valid_sorted.groupby("player_id")["x"].diff()
    valid_sorted["dy"] = valid_sorted.groupby("player_id")["y"].diff()
    valid_sorted["speed"] = np.sqrt(valid_sorted["dx"]**2 + valid_sorted["dy"]**2) / valid_sorted["dt"]
    # Normal pitch width ~68m, length ~105m -> factor ~85m per pitch unit
    valid_sorted["speed_mps"] = valid_sorted["speed"] * 85.0

    mean_speed = float(valid_sorted["speed_mps"].mean(skipna=True))
    p95_speed = float(valid_sorted["speed_mps"].quantile(0.95))
    coverage = float(len(valid) / max(1, len(df)))

    # Downstream Possession & TAS across sample frames
    unique_frames = sorted(valid["frame"].unique())
    # Subsample if large
    sample_frames = unique_frames[::max(1, len(unique_frames) // 100)]
    poss_teams = []
    tas_values = []

    for f in sample_frames:
        f_rows = valid[valid["frame"] == f]
        if len(f_rows) < 4:
            continue
        try:
            ts = float(f_rows["timestamp"].iloc[0]) if "timestamp" in f_rows.columns else f / fps
            graph = build_frame_graph(f_rows, f, ts)
            pred = predict_possession(graph, f_rows)
            if not pred.empty:
                top = pred.iloc[0]
                team = str(top["team"])
                possessor_id = f"{top['team'].upper()}_{top['player_id']}"
                poss_teams.append(team)

                feats = extract_tactical_features(graph, possessor_id=possessor_id, home_attacks_x1=True)
                tas = calculate_tactical_score(feats, possessor_team=team)
                tas_values.append(tas)
        except Exception:
            continue

    home_ratio = float(poss_teams.count("home") / max(1, len(poss_teams))) if poss_teams else 0.5
    mean_tas = float(np.mean(tas_values)) if tas_values else 0.5
    std_tas = float(np.std(tas_values)) if tas_values else 0.0

    return {
        "dataset": dataset_name,
        "fps": fps,
        "total_observations": len(df),
        "unique_players": int(df["player_id"].nunique()),
        "tracking_coverage_pct": round(coverage * 100.0, 2),
        "mean_speed_mps": round(mean_speed, 2),
        "p95_speed_mps": round(p95_speed, 2),
        "home_possession_share": round(home_ratio * 100.0, 1),
        "mean_tas": round(mean_tas, 4),
        "std_tas": round(std_tas, 4),
    }


def run_cross_dataset_validation(
    metrica_df: pd.DataFrame,
    video_df: pd.DataFrame,
    skillcorner_df: Optional[pd.DataFrame] = None,
    output_dir: str = "results/step85",
) -> Dict[str, Any]:
    """Execute Step 85 Cross-Dataset Validation and emit comparative metrics."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_list = []

    # 1. Metrica
    m_res = _analyze_dataset_dynamics(metrica_df, "Metrica Optical Tracking", fps=25.0)
    metrics_list.append(m_res)

    # 2. Video Perception
    v_res = _analyze_dataset_dynamics(video_df, "YOLOv8 + ByteTrack Video Perception", fps=25.0)
    metrics_list.append(v_res)

    # 3. SkillCorner (if provided)
    if skillcorner_df is not None and not skillcorner_df.empty:
        sc_res = _analyze_dataset_dynamics(skillcorner_df, "SkillCorner Computer Vision", fps=10.0)
        metrics_list.append(sc_res)

    res_df = pd.DataFrame(metrics_list)
    csv_file = out_path / "step85_cross_dataset_comparison.csv"
    res_df.to_csv(csv_file, index=False)

    summary = {
        "step": "Step 85 — Cross-Dataset Validation",
        "timestamp": pd.Timestamp.now().isoformat(),
        "datasets_evaluated": len(metrics_list),
        "results": metrics_list,
    }
    json_file = out_path / "step85_cross_dataset_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
