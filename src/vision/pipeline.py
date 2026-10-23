"""End-to-end video pipeline: Converts gameplay video into canonical tracking and downstream insights."""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import os
import pandas as pd

from .homography import PitchHomography
from .video_tracker import VideoGameStateTracker
from src.possession.possession_baseline import score_frame
from src.possession.spatial_graph import build_frame_graph
from src.tactics.tactical_features import extract_tactical_features
from src.tactics.tactical_score import calculate_tactical_score
from src.tactics.pass_candidates import generate_pass_candidates
from src.tactics.pass_ranking import rank_counterfactual_passes
from src.noise.interpolation import interpolate_trajectory
from src.noise.smoothing import smooth_trajectory


class VideoAnalysisPipeline:
    """Orchestrates end-to-end reconstruction from video to possession and tactical analysis."""

    def __init__(
        self,
        output_dir: str | Path = "data/interim",
        homography_preset: str = "auto_detect",
        conf_threshold: float = 0.18,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.homography = PitchHomography()
        self.homography_preset = homography_preset
        self.conf_threshold = conf_threshold

    def analyze_video(
        self,
        video_path: str,
        match_id: str,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        apply_mitigation: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        start_time_sec: float = 0.0,
        end_time_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run complete perception and game-state reconstruction on an input video.

        Args:
            video_path: Path to the input video.
            match_id: Unique identifier for the match session.
            max_frames: Maximum number of frames to process.
            frame_stride: Process every N frames.
            apply_mitigation: Whether to apply interpolation and Savitzky-Golay filtering.
            progress_callback: Function reporting (progress_ratio, status_message).

        Returns:
            Dictionary containing metadata, file paths, and tactical metrics summary.
        """
        # Step 1: Video Tracking
        if progress_callback:
            progress_callback(0.05, "Initializing video tracker and YOLO weights...")

        tracker = VideoGameStateTracker(
            homography=self.homography,
            conf_threshold=self.conf_threshold,
            homography_preset=self.homography_preset,
        )

        tracking_df = tracker.process_video(
            video_path=video_path,
            match_id=match_id,
            max_frames=max_frames,
            frame_stride=frame_stride,
            progress_callback=progress_callback,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
        )

        if tracking_df.empty:
            message = (
                "No reliable full-pitch calibration was found. The video needs a visible halfway line and center-circle landmarks, or explicit pitch-corner calibration."
                if not getattr(tracker, "metadata", {}).get("pitch_calibrated", False)
                else "No players detected in video."
            )
            return {
                "match_id": match_id,
                "status": "error",
                "message": message,
                "frame_count": 0,
            }

        # Step 2: Trajectory Mitigation (Clean up detection jitter and micro-gaps)
        if apply_mitigation and len(tracking_df) > 10:
            if progress_callback:
                progress_callback(0.82, "Applying bounded interpolation and Savitzky-Golay smoothing...")
            try:
                interp_df = interpolate_trajectory(tracking_df, max_gap=10)
                tracking_df = smooth_trajectory(interp_df, method="savgol", window_length=7, polyorder=2)
            except Exception:
                # Fallback to raw tracking if sequence is too short for smoothing window
                pass

        # Step 3: Save canonical tracking table so backend services can query it directly
        output_csv = self.output_dir / f"{match_id}_canonical_tracking.csv"
        tracking_df.to_csv(output_csv, index=False)

        # Step 4: Compute match-level possession distribution & average TAS
        if progress_callback:
            progress_callback(0.90, "Computing ball-free possession and Tactical Advantage Scores...")

        unique_frames = sorted(tracking_df["frame"].unique())
        home_possession_count = 0
        away_possession_count = 0
        tas_scores = []
        frame_insights = []

        meta = getattr(tracker, "metadata", {})
        fps = meta.get("fps", 25.0)

        for f_idx in unique_frames:
            f_rows = tracking_df[tracking_df["frame"] == f_idx]
            if len(f_rows) < 4:
                continue

            # Graph & Possession
            try:
                f_time = float(f_idx / max(fps, 1.0))
                f_graph = build_frame_graph(f_rows, frame=int(f_idx), timestamp=f_time)
                scores_df = score_frame(f_graph, f_rows)
                if not scores_df.empty:
                    top = scores_df.iloc[0]
                    poss_team = str(top["team"])
                    possessor_id = str(top["player_id"])
                    if poss_team == "home":
                        home_possession_count += 1
                    elif poss_team == "away":
                        away_possession_count += 1

                    # Tactical Score
                    features = extract_tactical_features(f_graph, possessor_id=possessor_id, home_attacks_x1=True)
                    tas_val = calculate_tactical_score(features, possessor_team=poss_team or "home")
                    tas_scores.append(tas_val)

                    frame_insights.append({
                        "frame": int(f_idx),
                        "possessing_team": poss_team,
                        "possessor_player_id": possessor_id,
                        "tas": round(float(tas_val), 4),
                    })
            except Exception:
                continue

        total_poss_frames = max(1, home_possession_count + away_possession_count)
        home_pct = round(100.0 * home_possession_count / total_poss_frames, 1)
        away_pct = round(100.0 * away_possession_count / total_poss_frames, 1)
        avg_tas = round(float(sum(tas_scores) / max(1, len(tas_scores))), 4) if tas_scores else 0.500

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        min_f = int(tracking_df["frame"].min())
        max_f = int(tracking_df["frame"].max())
        total_vid_frames = meta.get("total_frames", max_f + 1)
        dur = meta.get("segment_duration_sec", round(len(unique_frames) / max(fps, 1.0), 2))
        start_t = meta.get("start_time_sec", 0.0)
        end_t = meta.get("end_time_sec", dur)

        return {
            "match_id": match_id,
            "status": "success",
            "frame_count": len(unique_frames),
            "min_frame": min_f,
            "max_frame": max_f,
            "fps": fps,
            "duration_sec": dur,
            "start_time_sec": start_t,
            "end_time_sec": end_t,
            "frame_stride": frame_stride,
            "total_video_frames": total_vid_frames,
            "tracking_rows": len(tracking_df),
            "tracking_file": str(output_csv),
            "team_colors": getattr(tracker, "team_colors", {"home": "#E62325", "away": "#034694"}),
            "possession_share": {"home": home_pct, "away": away_pct},
            "average_tas": avg_tas,
            "sample_insights": frame_insights,
        }
