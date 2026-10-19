"""Video Game State Tracker: Extracts canonical player tracking from raw football video."""

from __future__ import annotations
from typing import Callable, Dict, List, Any, Optional
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .homography import PitchHomography
from .jersey_classifier import JerseyClassifier


class VideoGameStateTracker:
    """Detects, tracks, and maps players from broadcast video into canonical pitch tracking."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        homography: PitchHomography | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        """Initialize the YOLO-based video tracker.

        Args:
            model_name: Ultralytics model weights file or identifier.
            homography: PitchHomography instance. If None, default normalized mapping is used.
            conf_threshold: Minimum detection confidence score.
        """
        self.model = YOLO(model_name)
        self.homography = homography or PitchHomography()
        self.conf_threshold = conf_threshold
        self.classifier = JerseyClassifier()

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

        # Check if homography is using identity; if so, create empirical broadcast homography
        if np.allclose(self.homography.matrix, np.eye(3)):
            self.homography = PitchHomography.from_preset("main_broadcast", width, height)

        if max_frames is not None:
            frames_to_process = min(total_frames, max_frames)
        else:
            frames_to_process = total_frames

        raw_detections: List[Dict[str, Any]] = []
        player_colors: List[tuple[str, np.ndarray]] = []
        frame_idx = 0
        processed_count = 0

        # Phase 1: Frame-by-frame detection and tracking
        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # Run YOLO multi-object tracking (person class = 0)
            results = self.model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=self.conf_threshold,
                verbose=False,
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # Require a valid tracking ID
                    if box.id is None:
                        continue

                    track_id = int(box.id[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                    # Foot contact point is bottom-center of bounding box
                    u_foot = float((xyxy[0] + xyxy[2]) / 2.0)
                    v_foot = float(xyxy[3])

                    # Project to pitch coordinates [0.0, 1.0]
                    pitch_x, pitch_y = self.homography.project_point(u_foot, v_foot)

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
        # Sort canonically by frame and player_id
        df = df.sort_values(by=["frame", "player_id"]).reset_index(drop=True)

        if progress_callback:
            progress_callback(1.0, f"Completed tracking: {len(df)} player observations across {processed_count} frames")

        return df
