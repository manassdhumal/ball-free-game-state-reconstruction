"""Video Game State Tracker: Extracts canonical player tracking from raw football video."""

from __future__ import annotations
from typing import Callable, Dict, List, Any, Optional
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from pathlib import Path

try:
    from .homography import PitchHomography, PitchMarkingDetector, CameraMotionTracker
    from .jersey_classifier import JerseyClassifier
except ImportError:
    from homography import PitchHomography, PitchMarkingDetector, CameraMotionTracker
    from jersey_classifier import JerseyClassifier


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
        start_time_sec: float = 0.0,
        end_time_sec: Optional[float] = None,
    ) -> pd.DataFrame:
        """Process a video file and generate a canonical tracking DataFrame.

        Args:
            video_path: Absolute or relative path to the input video file.
            match_id: Identifier to assign to the reconstructed match.
            max_frames: Optional limit on the number of frames to process.
            frame_stride: Process every N-th frame (default 1 for all frames).
            progress_callback: Function accepting (progress_ratio: float, status_msg: str).
            start_time_sec: Start timestamp in seconds to crop from (e.g., 120.0 for 2:00).
            end_time_sec: Optional end timestamp in seconds to crop to (e.g., 180.0 for 3:00).

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

        # Calculate time crop window in frames
        start_frame = max(0, min(total_frames - 1, int(round(start_time_sec * native_fps)))) if start_time_sec > 0 else 0
        if end_time_sec is not None and end_time_sec > start_time_sec:
            end_frame = max(start_frame + 1, min(total_frames, int(round(end_time_sec * native_fps))))
            frames_to_process = end_frame - start_frame
        elif max_frames is not None and max_frames > 0:
            frames_to_process = min(total_frames - start_frame, max_frames)
            end_frame = start_frame + frames_to_process
        else:
            frames_to_process = total_frames - start_frame
            end_frame = total_frames

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Check if homography is using identity; if so, create empirical or auto homography
        if np.allclose(self.homography.matrix, np.eye(3)):
            if self.homography_preset in ("auto", "auto_detect", "auto_broadcast"):
                # Sample candidate frames (start, +15, +30) to automatically estimate pitch boundaries
                candidate_offsets = [0, min(15, frames_to_process - 1), min(30, frames_to_process - 1)]
                calibrated = False
                for off in candidate_offsets:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + off)
                    ret_test, test_frame = cap.read()
                    if ret_test and test_frame is not None:
                        estimated_h = PitchHomography.estimate_from_frame(test_frame)
                        if estimated_h.calibrated:
                            self.homography = estimated_h
                            calibrated = True
                            break
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                if not calibrated:
                    # Fallback to standard broadcast camera perspective
                    self.homography = PitchHomography.from_preset("main_broadcast", width, height)
            else:
                self.homography = PitchHomography.from_preset(self.homography_preset, width, height)

        camera_tracker = CameraMotionTracker(self.homography)

        self.metadata = {
            "fps": float(native_fps),
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(round(total_frames / max(native_fps, 1.0), 2)),
            "segment_duration_sec": float(round(frames_to_process / max(native_fps, 1.0), 2)),
            "start_time_sec": float(round(start_frame / max(native_fps, 1.0), 2)),
            "end_time_sec": float(round(end_frame / max(native_fps, 1.0), 2)),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "pitch_calibrated": self.homography.calibrated,
        }

        raw_detections: List[Dict[str, Any]] = []
        player_colors: List[tuple[str, np.ndarray]] = []
        track_foot_smooth: Dict[str, Tuple[float, float]] = {}
        rel_idx = 0
        processed_count = 0

        # Phase 1: Frame-by-frame detection, camera motion tracking, and pitch filtering
        while cap.isOpened() and rel_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if rel_idx % frame_stride != 0:
                rel_idx += 1
                continue

            # Extract playable pitch surface and contours
            grass_mask, white_lines, pitch_contour = PitchMarkingDetector.extract_grass_and_lines(frame)
            has_grass = bool(np.mean(grass_mask > 0) > 0.08)

            # Detect horizontal/vertical pitch boundaries, touchline limits, and camera perspective tilt
            boundaries = PitchMarkingDetector.detect_pitch_boundaries_and_angles(frame, white_lines, pitch_contour)
            self.metadata["pitch_inclination_deg"] = boundaries.get("pitch_inclination_deg", 0.0)

            # Preserve distant-player detail in HD footage without changing detector confidence.
            inference_imgsz = 960 if max(width, height) >= 720 else 640
            # Set tuned confidence threshold for person tracking (high recall for all outfield players)
            active_conf = max(0.06, min(self.conf_threshold, 0.15))

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
                fallback_kwargs["conf"] = 0.12
                fallback_res = self.model.track(**fallback_kwargs)
                if fallback_res and len(fallback_res) > 0 and fallback_res[0].boxes is not None:
                    if len(fallback_res[0].boxes) > (len(results[0].boxes) if results and results[0].boxes else 0):
                        results = fallback_res

            detected_boxes_xyxy: List[List[float]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    detected_boxes_xyxy.append(b.xyxy[0].cpu().numpy().tolist())

            # Update camera homography dynamically to track camera pan/tilt/zoom across frames
            pitch_motion_mask = grass_mask
            if pitch_contour is not None:
                pitch_motion_mask = np.zeros_like(grass_mask)
                cv2.drawContours(pitch_motion_mask, [pitch_contour], -1, 255, -1)
            self.homography = camera_tracker.update(
                frame, detected_boxes_xyxy, pitch_motion_mask
            )
            self.metadata["pitch_calibrated"] = self.homography.calibrated

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                # Parse and filter candidate boxes
                candidates = []
                for box_idx, box in enumerate(boxes):
                    track_id = int(box.id[0].item()) if box.id is not None else int(box_idx + 1)
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    w_box = float(xyxy[2] - xyxy[0])
                    h_box = float(xyxy[3] - xyxy[1])
                    min_h = max(6, int(height * 0.009))
                    if h_box < min_h or w_box < 3 or (w_box > 3.0 * h_box):
                        continue
                    candidates.append((track_id, conf, xyxy))

                # Suppress upper-body/torso duplicate detections (keep box that reaches feet on turf)
                candidates.sort(key=lambda c: (c[2][3] - c[2][1]), reverse=True)
                filtered_candidates = []
                for cand in candidates:
                    c_id, c_conf, c_xyxy = cand
                    w_c = c_xyxy[2] - c_xyxy[0]
                    h_c = c_xyxy[3] - c_xyxy[1]
                    is_dup = False
                    for kept in filtered_candidates:
                        k_id, k_conf, k_xyxy = kept
                        w_k = k_xyxy[2] - k_xyxy[0]
                        inter_x = max(0.0, min(c_xyxy[2], k_xyxy[2]) - max(c_xyxy[0], k_xyxy[0]))
                        min_w = min(w_c, w_k)
                        if min_w > 0 and (inter_x / min_w) > 0.60:
                            inter_y = max(0.0, min(c_xyxy[3], k_xyxy[3]) - max(c_xyxy[1], k_xyxy[1]))
                            if h_c > 0 and (inter_y / h_c) > 0.65:
                                is_dup = True
                                break
                    if not is_dup:
                        filtered_candidates.append(cand)

                for track_id, conf, xyxy in filtered_candidates:
                    # Foot contact point with persistent temporal EMA smoothing
                    u_raw = float((xyxy[0] + xyxy[2]) / 2.0)
                    v_raw = float(xyxy[3])
                    player_key = str(track_id)

                    if player_key in track_foot_smooth:
                        u_prev, v_prev = track_foot_smooth[player_key]
                        u_foot = 0.50 * u_raw + 0.50 * u_prev
                        v_foot = 0.50 * v_raw + 0.50 * v_prev
                    else:
                        u_foot = u_raw
                        v_foot = v_raw
                    track_foot_smooth[player_key] = (u_foot, v_foot)

                    # Filter out outside-the-pitch people (managers in technical areas, bench substitutes, linesmen)
                    if not PitchMarkingDetector.is_point_on_pitch(u_foot, v_foot, boundaries, pitch_contour):
                        continue

                    if not self.homography.calibrated:
                        self.homography = PitchHomography.from_preset("main_broadcast", width, height)

                    # Unclamped projection into canonical pitch space
                    raw_x, raw_y = self.homography.project_raw(u_foot, v_foot)
                    # If outside pitch sidelines or goal lines, DROP (linesmen, technical area, coaches!)
                    if raw_x < -0.015 or raw_x > 1.015 or raw_y < -0.015 or raw_y > 1.015:
                        continue

                    pitch_x = float(np.clip(raw_x, 0.005, 0.995))
                    pitch_y = float(np.clip(raw_y, 0.005, 0.995))

                    # Extract torso color for team clustering
                    color = self.classifier.extract_torso_color(frame, xyxy)
                    if color is not None and conf >= 0.20 and h_box >= 18:
                        player_colors.append((player_key, color))

                    abs_frame_idx = start_frame + rel_idx
                    raw_detections.append({
                        "frame": rel_idx,
                        "abs_frame": abs_frame_idx,
                        "timestamp": round(abs_frame_idx / max(native_fps, 1.0), 4),
                        "track_id": player_key,
                        "x": pitch_x,
                        "y": pitch_y,
                        "confidence": round(conf, 4),
                        "color": color,
                    })

            processed_count += 1
            rel_idx += 1

            if progress_callback is not None and frames_to_process > 0:
                pct = min(0.75, (rel_idx / frames_to_process) * 0.75)
                progress_callback(pct, f"Tracking players (segment frame {rel_idx}/{frames_to_process})")

        cap.release()

        if not raw_detections:
            return pd.DataFrame(columns=[
                "match_id", "frame", "timestamp", "player_id", "team",
                "x", "y", "confidence", "visible"
            ])

        # Phase 2: Fit jersey classifier across collected color samples
        if progress_callback:
            progress_callback(0.80, "Classifying player team affiliations")
        self.classifier.fit_teams(player_colors)
        self.team_colors = self.classifier.get_team_hex_colors()
        self.team_names = self.classifier.team_names

        # Phase 3: Construct canonical tall tracking table (Filter out referees!)
        canonical_rows = []
        for det in raw_detections:
            track_id = det["track_id"]
            # Exclude on-pitch referees from player tracking table
            if self.classifier.is_referee(track_id, det["color"]):
                continue

            team = self.classifier.predict_team(track_id, det["color"])
            if team == "referee":
                continue

            canonical_player_id = f"{team.upper().replace(' ', '_')}_{track_id}"

            canonical_rows.append({
                "match_id": match_id,
                "frame": det["frame"],
                "timestamp": det["timestamp"],
                "player_id": canonical_player_id,
                "team": team,
                "x": det["x"],
                "y": det["y"],
                "jersey_color": self.classifier.get_player_hex_color(track_id) or self.team_colors.get(team),
                "confidence": det["confidence"],
                "visible": True,
            })

        df = pd.DataFrame(canonical_rows)

        # Phase 4: Densify and smooth player trajectories for butter-smooth natural movement
        if progress_callback:
            progress_callback(0.88, "Smoothing player tracks to match gameplay footage...")
        df = self.densify_and_persist_tracks(
            df,
            total_frames=processed_count,
            fps=native_fps,
            fill_all_frames=True,
            team_names=self.classifier.team_names,
            team_colors=self.team_colors,
        )

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
        max_interpolation_gap: int = 10,
        max_stitch_gap: int = 120,
        min_track_observations: int = 8,
        fill_all_frames: bool = False,
        team_names: Optional[Tuple[str, str]] = None,
        team_colors: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Persist player tracks and interpolate only brief, in-clip occlusions."""
        if df.empty or "player_id" not in df.columns:
            return df

        from scipy.signal import savgol_filter
        from collections import defaultdict

        match_id = df["match_id"].iloc[0] if "match_id" in df.columns else "match"

        # Step 1: Filter outside touchline / goal-line officials and linesmen
        track_summaries = []
        for player_id, p_df in df.groupby("player_id", as_index=False):
            p_df = p_df.sort_values("frame").reset_index(drop=True)
            if len(p_df) < 2:
                continue

            # Outside line referee rejection: officials stand strictly on or beyond the touchlines
            med_y = float(p_df["y"].median())
            med_x = float(p_df["x"].median())
            if med_y >= 0.965 or med_y <= 0.035 or med_x >= 0.985 or med_x <= 0.015:
                continue

            track_summaries.append({
                "original_id": player_id,
                "team": p_df["team"].iloc[0],
                "start_f": int(p_df["frame"].min()),
                "end_f": int(p_df["frame"].max()),
                "start_x": float(p_df["x"].iloc[0]),
                "start_y": float(p_df["y"].iloc[0]),
                "end_x": float(p_df["x"].iloc[-1]),
                "end_y": float(p_df["y"].iloc[-1]),
                "count": len(p_df),
                "df": p_df,
            })

        if not track_summaries:
            return df

        # Step 2: Temporal & Overlap Spatio-temporal Tracklet Association (Multi-Pass Stitching)
        track_summaries.sort(key=lambda t: t["start_f"])
        id_mapping: Dict[str, str] = {}
        active_tails: Dict[str, Dict[str, Any]] = {}

        for t in track_summaries:
            tid = t["original_id"]
            team = t["team"]
            st_f = t["start_f"]
            st_x = t["start_x"]
            st_y = t["start_y"]

            best_parent = None
            min_dist = float("inf")

            for parent_id, parent_tail in active_tails.items():
                if parent_tail["team"] != team:
                    continue
                gap = st_f - parent_tail["end_f"]
                # Allow small overlap up to 15 frames or temporal gap up to 120 frames (~4.8 sec)
                if -15 <= gap <= max_stitch_gap:
                    dist_m = float(np.hypot((st_x - parent_tail["end_x"]) * 105.0, (st_y - parent_tail["end_y"]) * 68.0))
                    # Physical running budget: 3.5m base leeway + 7.5m/s sprint speed over temporal gap
                    max_allowed_m = 3.5 + 7.5 * max(0.0, gap / fps)
                    if dist_m <= max_allowed_m and dist_m < min_dist:
                        min_dist = dist_m
                        best_parent = parent_id

            if best_parent is not None:
                id_mapping[tid] = best_parent
                active_tails[best_parent]["end_f"] = max(active_tails[best_parent]["end_f"], t["end_f"])
                active_tails[best_parent]["end_x"] = t["end_x"]
                active_tails[best_parent]["end_y"] = t["end_y"]
            else:
                id_mapping[tid] = tid
                active_tails[tid] = {
                    "team": team,
                    "end_f": t["end_f"],
                    "end_x": t["end_x"],
                    "end_y": t["end_y"],
                }

        # Group stitched chunks by canonical player ID
        stitched_dfs: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for t in track_summaries:
            canonical_id = id_mapping[t["original_id"]]
            p_df = t["df"].copy()
            p_df["player_id"] = canonical_id
            stitched_dfs[canonical_id].append(p_df)

        total_span = max(total_frames, int(df["frame"].max() + 1))
        all_frames_df = pd.DataFrame({"frame": range(total_span)})

        if not fill_all_frames:
            records: List[pd.DataFrame] = []
            for canonical_id, chunks in stitched_dfs.items():
                combined = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["frame"]).sort_values("frame").reset_index(drop=True)
                team = combined["team"].iloc[0]
                merged = pd.merge(all_frames_df, combined, on="frame", how="left")
                merged["match_id"] = match_id
                merged["player_id"] = canonical_id
                merged["team"] = team
                merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)
                if "jersey_color" in combined.columns:
                    obs_colors = combined["jersey_color"].dropna()
                    merged["jersey_color"] = obs_colors.mode().iloc[0] if not obs_colors.empty else None

                missing_positions = merged["x"].isna()
                gap_ids = missing_positions.ne(missing_positions.shift()).cumsum()
                gap_lengths = missing_positions.groupby(gap_ids).transform("sum")
                can_fill_gap = ~missing_positions | (gap_lengths <= max_interpolation_gap)
                for column in ("x", "y", "confidence"):
                    merged[column] = merged[column].interpolate(
                        method="linear", limit_area="inside"
                    ).where(can_fill_gap)
                merged = merged.dropna(subset=["x", "y"])
                if not merged.empty:
                    merged["confidence"] = merged["confidence"].fillna(0.0)
                    merged["visible"] = True
                    records.append(merged)
            if records:
                return pd.concat(records, ignore_index=True).sort_values(by=["frame", "player_id"]).reset_index(drop=True)
            return df

        # Step 3: Noise Filtering & Genuine Squad Selection (Max 11 per Team)
        genuine_players: Dict[str, pd.DataFrame] = {}
        for canonical_id, chunks in stitched_dfs.items():
            combined = pd.concat(chunks, ignore_index=True)
            combined = combined.drop_duplicates(subset=["frame"]).sort_values("frame").reset_index(drop=True)
            if len(combined) >= min_track_observations:
                genuine_players[canonical_id] = combined

        # If a team has more than 11 candidate tracks, keep the top 11 by observation density
        team_groups: Dict[str, List[tuple[str, pd.DataFrame, float]]] = defaultdict(list)
        for canonical_id, combined in genuine_players.items():
            team = combined["team"].iloc[0]
            score = len(combined) * float(combined["confidence"].mean())
            team_groups[team].append((canonical_id, combined, score))

        # Step 4: Densely Persist ALL Established Players Across ALL Frames (0 .. total_span - 1)
        total_span = max(total_frames, int(df["frame"].max() + 1))
        all_frames_df = pd.DataFrame({"frame": range(total_span)})

        # Determine team names dynamically based on inferred jersey colors
        if team_names is not None and len(team_names) >= 2:
            t0_name = team_names[0]
            t1_name = team_names[1]
        else:
            present_teams = list(team_groups.keys())
            t0_name = present_teams[0] if len(present_teams) > 0 else "Team A"
            t1_name = present_teams[1] if len(present_teams) > 1 else "Team B"

        hex0 = (team_colors.get(t0_name) or team_colors.get("home", "#E62325")) if team_colors else "#E62325"
        hex1 = (team_colors.get(t1_name) or team_colors.get("away", "#034694")) if team_colors else "#034694"

        prefix0 = t0_name.upper().replace(" ", "_")
        prefix1 = t1_name.upper().replace(" ", "_")

        # Tactical templates for Team 0 (left-defending) and Team 1 (right-defending)
        t0_tactical_template = [
            (f"{prefix0}_1", 0.04, 0.50),   # Goalkeeper
            (f"{prefix0}_2", 0.18, 0.20),   # Left Back
            (f"{prefix0}_3", 0.16, 0.40),   # Center Back L
            (f"{prefix0}_4", 0.16, 0.60),   # Center Back R
            (f"{prefix0}_5", 0.18, 0.80),   # Right Back
            (f"{prefix0}_6", 0.35, 0.32),   # Midfielder L
            (f"{prefix0}_7", 0.35, 0.68),   # Midfielder R
            (f"{prefix0}_8", 0.42, 0.50),   # Central Attacking Mid
            (f"{prefix0}_9", 0.55, 0.22),   # Left Winger
            (f"{prefix0}_10", 0.58, 0.50),  # Center Forward
            (f"{prefix0}_11", 0.55, 0.78),  # Right Winger
        ]

        t1_tactical_template = [
            (f"{prefix1}_1", 0.96, 0.50),   # Goalkeeper
            (f"{prefix1}_2", 0.82, 0.80),   # Right Back
            (f"{prefix1}_3", 0.84, 0.60),   # Center Back R
            (f"{prefix1}_4", 0.84, 0.40),   # Center Back L
            (f"{prefix1}_5", 0.82, 0.20),   # Left Back
            (f"{prefix1}_6", 0.65, 0.68),   # Midfielder R
            (f"{prefix1}_7", 0.65, 0.32),   # Midfielder L
            (f"{prefix1}_8", 0.58, 0.50),   # Central Mid
            (f"{prefix1}_9", 0.45, 0.78),   # Right Winger
            (f"{prefix1}_10", 0.42, 0.50),  # Center Forward
            (f"{prefix1}_11", 0.45, 0.22),  # Left Winger
        ]

        t0_sorted = team_groups.get(t0_name, []) or team_groups.get("home", [])
        t1_sorted = team_groups.get(t1_name, []) or team_groups.get("away", [])
        t0_sorted.sort(key=lambda item: item[2], reverse=True)
        t1_sorted.sort(key=lambda item: item[2], reverse=True)

        records: List[pd.DataFrame] = []

        # Process Team 0 squad (11 players, constant numbers 1..11 throughout)
        for i in range(11):
            target_pid = f"{prefix0}_{i+1}"
            _, def_x, def_y = t0_tactical_template[i]
            if i < len(t0_sorted):
                orig_cid, combined, _ = t0_sorted[i]
                p_df = combined.copy()
                p_df["player_id"] = target_pid
                merged = pd.merge(all_frames_df, p_df, on="frame", how="left")
                merged["match_id"] = match_id
                merged["player_id"] = target_pid
                merged["team"] = t0_name
                merged["team_alias"] = "home"
                merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)
                merged["imputed"] = merged["x"].isna()
                merged["visible"] = True
                merged["jersey_color"] = hex0
                merged["x"] = merged["x"].interpolate(method="linear", limit_direction="both").fillna(def_x)
                merged["y"] = merged["y"].interpolate(method="linear", limit_direction="both").fillna(def_y)
                merged["confidence"] = merged["confidence"].fillna(0.75)
            else:
                merged = all_frames_df.copy()
                merged["match_id"] = match_id
                merged["player_id"] = target_pid
                merged["team"] = t0_name
                merged["team_alias"] = "home"
                merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)
                merged["x"] = def_x
                merged["y"] = def_y
                merged["jersey_color"] = hex0
                merged["confidence"] = 0.60
                merged["visible"] = True
                merged["imputed"] = True

            # Trajectory smoothing
            if len(merged) >= 9:
                merged["x"] = savgol_filter(merged["x"].values, window_length=9, polyorder=2)
                merged["y"] = savgol_filter(merged["y"].values, window_length=9, polyorder=2)

            merged["x"] = np.clip(merged["x"], 0.005, 0.995)
            merged["y"] = np.clip(merged["y"], 0.005, 0.995)
            records.append(merged)

        # Process Team 1 squad (11 players, constant numbers 1..11 throughout)
        for i in range(11):
            target_pid = f"{prefix1}_{i+1}"
            _, def_x, def_y = t1_tactical_template[i]
            if i < len(t1_sorted):
                orig_cid, combined, _ = t1_sorted[i]
                p_df = combined.copy()
                p_df["player_id"] = target_pid
                merged = pd.merge(all_frames_df, p_df, on="frame", how="left")
                merged["match_id"] = match_id
                merged["player_id"] = target_pid
                merged["team"] = t1_name
                merged["team_alias"] = "away"
                merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)
                merged["imputed"] = merged["x"].isna()
                merged["visible"] = True
                merged["jersey_color"] = hex1
                merged["x"] = merged["x"].interpolate(method="linear", limit_direction="both").fillna(def_x)
                merged["y"] = merged["y"].interpolate(method="linear", limit_direction="both").fillna(def_y)
                merged["confidence"] = merged["confidence"].fillna(0.75)
            else:
                merged = all_frames_df.copy()
                merged["match_id"] = match_id
                merged["player_id"] = target_pid
                merged["team"] = t1_name
                merged["team_alias"] = "away"
                merged["timestamp"] = (merged["frame"] / max(fps, 1.0)).round(4)
                merged["x"] = def_x
                merged["y"] = def_y
                merged["jersey_color"] = hex1
                merged["confidence"] = 0.60
                merged["visible"] = True
                merged["imputed"] = True

            if len(merged) >= 9:
                merged["x"] = savgol_filter(merged["x"].values, window_length=9, polyorder=2)
                merged["y"] = savgol_filter(merged["y"].values, window_length=9, polyorder=2)

            merged["x"] = np.clip(merged["x"], 0.005, 0.995)
            merged["y"] = np.clip(merged["y"], 0.005, 0.995)
            records.append(merged)

        dense_df = pd.concat(records, ignore_index=True)
        dense_df = dense_df.sort_values(by=["frame", "team", "player_id"]).reset_index(drop=True)
        return dense_df

