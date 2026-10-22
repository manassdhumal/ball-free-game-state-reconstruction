"""Video Game State Tracker: Extracts canonical player tracking from raw football video."""

from __future__ import annotations
from typing import Callable, Dict, List, Any, Optional
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from pathlib import Path

from .homography import PitchHomography, PitchMarkingDetector, CameraMotionTracker
from .jersey_classifier import JerseyClassifier


class VideoGameStateTracker:
    """Detects, tracks, and maps players from broadcast video into canonical pitch tracking."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        homography: PitchHomography | None = None,
        conf_threshold: float = 0.08,
        tracker_config: str | None = None,
        homography_preset: str = "main_broadcast",
    ) -> None:
        """Initialize the YOLO-based video tracker.

        Args:
            model_name: Ultralytics model weights file or identifier.
            homography: PitchHomography instance. If None, default normalized mapping is used.
            conf_threshold: Minimum detection confidence score (tuned to 0.08 for high player recall).
            tracker_config: Path to tracker YAML config. Defaults to configs/bytetrack_football.yaml if present.
            homography_preset: Empirical camera preset if homography matrix is identity.
        """
        self.model = YOLO(model_name)
        self.homography = homography or PitchHomography()
        self.conf_threshold = conf_threshold
        self.homography_preset = homography_preset
        self.classifier = JerseyClassifier()
        if tracker_config is not None:
            self.tracker_config = tracker_config
        else:
            default_cfg = Path(__file__).resolve().parents[2] / "configs" / "bytetrack_football.yaml"
            self.tracker_config = str(default_cfg) if default_cfg.exists() else "bytetrack.yaml"

    def process_video(
        self,
        video_path: str,
        match_id: str = "uploaded_gameplay",
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """Process a video file and generate a canonical tracking DataFrame.

        Args:
            video_path: Absolute or relative path to the input video file.
            match_id: Identifier to assign to the reconstructed match.
            max_frames: Optional limit on the number of frames to process.
            frame_stride: Process every N-th frame (default 1 for all frames).
            progress_callback: Function accepting (progress_ratio: float, status_msg: str).

        Returns:
            DataFrame adhering to canonical tracking schema (tall table).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if homography is using identity; if so, create empirical or auto homography
        if np.allclose(self.homography.matrix, np.eye(3)):
            if self.homography_preset in ("auto", "auto_detect", "auto_broadcast"):
                # Read first valid frame to automatically estimate pitch boundaries
                ret_first, first_frame = cap.read()
                if ret_first and first_frame is not None:
                    self.homography = PitchHomography.estimate_from_frame(first_frame)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.homography = PitchHomography.from_preset("main_broadcast", width, height)
            else:
                self.homography = PitchHomography.from_preset(self.homography_preset, width, height)

        camera_tracker = CameraMotionTracker(self.homography)

        if max_frames is not None and max_frames > 0:
            frames_to_process = min(total_frames, max_frames)
        else:
            frames_to_process = total_frames

        self.metadata = {
            "fps": float(native_fps),
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(round(total_frames / max(native_fps, 1.0), 2)),
        }

        raw_detections: List[Dict[str, Any]] = []
        player_colors: List[tuple[str, np.ndarray]] = []
        frame_idx = 0
        processed_count = 0

        # Phase 1: Frame-by-frame detection, camera motion tracking, and pitch filtering
        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # Extract playable pitch surface and contours
            grass_mask, _white_lines, pitch_contour = PitchMarkingDetector.extract_grass_and_lines(frame)
            has_grass = bool(np.mean(grass_mask > 0) > 0.08)

            # High-resolution inference scale so small and distant outfield players are resolved
            inference_imgsz = max(960, min(1280, max(width, height)))
            active_conf = min(self.conf_threshold, 0.08)

            track_kwargs: Dict[str, Any] = {
                "source": frame,
                "persist": True,
                "classes": [0],
                "conf": active_conf,
                "imgsz": inference_imgsz,
                "verbose": False,
            }
            if self.tracker_config:
                track_kwargs["tracker"] = self.tracker_config

            results = self.model.track(**track_kwargs)

            # Fallback for synthetic/drawn clips where COCO assigns non-person classes
            if not results or results[0].boxes is None or len(results[0].boxes) < 4:
                fallback_kwargs = dict(track_kwargs)
                fallback_kwargs.pop("classes", None)
                fallback_kwargs["conf"] = 0.06
                fallback_res = self.model.track(**fallback_kwargs)
                if fallback_res and len(fallback_res) > 0 and fallback_res[0].boxes is not None:
                    if len(fallback_res[0].boxes) > (len(results[0].boxes) if results and results[0].boxes else 0):
                        results = fallback_res

            detected_boxes_xyxy: List[List[float]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    detected_boxes_xyxy.append(b.xyxy[0].cpu().numpy().tolist())

            # Update camera homography dynamically to track camera pan/tilt/zoom across frames
            self.homography = camera_tracker.update(frame, detected_boxes_xyxy)

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box_idx, box in enumerate(boxes):
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                    else:
                        track_id = int(box_idx + 1)
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                    w_box = float(xyxy[2] - xyxy[0])
                    h_box = float(xyxy[3] - xyxy[1])

                    # Filter out only severe non-human artifacts (keep small distant broadcast players)
                    min_h = max(5, int(height * 0.008))
                    if h_box < min_h or w_box < 3 or (w_box > 3.8 * h_box):
                        continue

                    # Foot contact point is bottom-center of bounding box
                    u_foot = float((xyxy[0] + xyxy[2]) / 2.0)
                    v_foot = float(xyxy[3])

                    # Generous pitch contour boundary to avoid dropping sideline and corner players
                    if pitch_contour is not None:
                        poly_dist = cv2.pointPolygonTest(pitch_contour, (u_foot, v_foot), True)
                        if poly_dist < -60.0:  # Permissive margin for touchline players
                            continue

                    # Safe projection into canonical pitch space
                    pitch_x, pitch_y, is_on_pitch = self.homography.project_point_safe(u_foot, v_foot)
                    if not is_on_pitch or pitch_x is None or pitch_y is None:
                        # Attempt bounded projection for valid field players standing on pitch borders
                        raw_proj = self.homography.project_point(u_foot, v_foot)
                        if raw_proj is not None and 0.0 <= raw_proj[0] <= 1.0 and 0.0 <= raw_proj[1] <= 1.0:
                            pitch_x, pitch_y = raw_proj
                        else:
                            continue

                    # Clamp strictly to pitch margins for seamless radar rendering
                    pitch_x = float(np.clip(pitch_x, 0.01, 0.99))
                    pitch_y = float(np.clip(pitch_y, 0.01, 0.99))

                    # Extract torso color for team clustering
                    color = self.classifier.extract_torso_color(frame, xyxy)
                    player_key = str(track_id)
                    if color is not None:
                        player_colors.append((player_key, color))

                    raw_detections.append({
                        "frame": frame_idx,
                        "timestamp": round(frame_idx / native_fps, 4),
                        "track_id": player_key,
                        "x": pitch_x,
                        "y": pitch_y,
                        "confidence": round(conf, 4),
                        "color": color,
                    })

            processed_count += 1
            frame_idx += 1

            if progress_callback is not None and frames_to_process > 0:
                pct = min(0.75, (frame_idx / frames_to_process) * 0.75)
                progress_callback(pct, f"Tracking players (frame {frame_idx}/{frames_to_process})")

        cap.release()

        if not raw_detections:
            # Return empty DataFrame with canonical schema
            return pd.DataFrame(columns=[
                "match_id", "frame", "timestamp", "player_id", "team",
                "x", "y", "confidence", "visible"
            ])

        # Phase 2: Fit jersey classifier across collected color samples
        if progress_callback:
            progress_callback(0.80, "Classifying player team affiliations")
        self.classifier.fit_teams(player_colors)

        # Phase 3: Construct canonical tall tracking table
        canonical_rows = []
        for det in raw_detections:
            track_id = det["track_id"]
            team = self.classifier.predict_team(track_id, det["color"])
            canonical_player_id = f"{team.upper()}_{track_id}"

            canonical_rows.append({
                "match_id": match_id,
                "frame": det["frame"],
                "timestamp": det["timestamp"],
                "player_id": canonical_player_id,
                "team": team,
                "x": det["x"],
                "y": det["y"],
                "confidence": det["confidence"],
                "visible": True,
            })

        df = pd.DataFrame(canonical_rows)

        # Phase 4: Densify player trajectories so all players maintain persistent radar dots
        if progress_callback:
            progress_callback(0.88, "Densifying player tracks to eliminate radar flickering...")
        df = self.densify_and_persist_tracks(df, total_frames=processed_count, fps=native_fps)

        # Sort canonically by frame and player_id
        df = df.sort_values(by=["frame", "player_id"]).reset_index(drop=True)

        if progress_callback:
            progress_callback(1.0, f"Completed tracking: {len(df)} player observations across {processed_count} frames")

        return df

    @staticmethod
    def densify_and_persist_tracks(
        df: pd.DataFrame,
        total_frames: int,
        fps: float = 25.0,
        max_interpolation_gap: int = 40,
        persistence_frames: int = 25,
    ) -> pd.DataFrame:
        """Densely complete player trajectories across intermediate frames and persist active pitch presence.

        Guarantees that all active outfield players and goalkeepers maintain persistent, visible dots
        on the radar throughout the game state reconstruction.
        """
        if df.empty or "player_id" not in df.columns:
            return df

        match_id = df["match_id"].iloc[0] if "match_id" in df.columns else "match"
        records: List[pd.DataFrame] = []

        for player_id, p_df in df.groupby("player_id", as_index=False):
            p_df = p_df.sort_values("frame").reset_index(drop=True)
            if p_df.empty:
                continue

            team = p_df["team"].iloc[0]
            first_f = int(p_df["frame"].min())
            last_f = int(p_df["frame"].max())

            # Bounded forward persistence: keep player active if still on pitch
            target_max_f = min(max(first_f, total_frames - 1), last_f + persistence_frames)

            # Reindex to continuous range
            full_frames = pd.DataFrame({"frame": range(first_f, target_max_f + 1)})
            merged = pd.merge(full_frames, p_df, on="frame", how="left")

            merged["match_id"] = match_id
            merged["player_id"] = player_id
            merged["team"] = team
            merged["x"] = merged["x"].interpolate(method="linear", limit=max_interpolation_gap).ffill().bfill()
            merged["y"] = merged["y"].interpolate(method="linear", limit=max_interpolation_gap).ffill().bfill()
            merged["confidence"] = merged["confidence"].interpolate(method="linear").fillna(0.70)
            merged["visible"] = True
            merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)

            # Drop any lingering NaN coordinates
            merged = merged.dropna(subset=["x", "y"])
            if not merged.empty:
                records.append(merged)

        if not records:
            return df

        dense_df = pd.concat(records, ignore_index=True)
        dense_df = dense_df.sort_values(by=["frame", "player_id"]).reset_index(drop=True)
        return dense_df
