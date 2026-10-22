"""Pitch homography and coordinate projection for video game state reconstruction.

Transforms pixel coordinates from a broadcast/tactical video stream into
normalized 2D pitch coordinates [0.0, 1.0] x [0.0, 1.0], using pitch markings,
structural line geometry, and camera motion tracking.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any, Optional, List
import numpy as np
import cv2


class PitchMarkingDetector:
    """Detects white pitch markings, halfway line, penalty D-arcs, and field boundaries."""

    @staticmethod
    def extract_grass_and_lines(
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Segment green grass turf and white pitch marking lines from a video frame.

        Returns:
            (grass_mask, white_line_mask, pitch_contour)
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broad HSV range for football grass across various lighting and camera grades
        grass = (
            (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 92) &
            (hsv[:, :, 1] >= 20) &
            (hsv[:, :, 2] >= 20)
        )
        grass_mask = (grass.astype(np.uint8) * 255)

        # Morphological closing to fill holes in the grass surface
        kernel_size = max(5, int(min(w, h) * 0.015))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        grass_cleaned = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)

        # Find largest grass contour (the playing surface)
        contours, _ = cv2.findContours(grass_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_contour = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 0.08 * w * h:
                pitch_contour = c

        # White lines: high intensity and low saturation within the playing field surface
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Pitch area mask
        pitch_area = (grass_cleaned > 0)
        if pitch_contour is not None:
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [pitch_contour], -1, 255, -1)
            pitch_area = (c_mask > 0)

        # White Top-Hat morphological filter to extract thin white lines even in stadium shadow
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, line_kernel)
        
        # Threshold for white markings: high brightness or strong ridge response on low saturation turf
        grass_gray = gray[pitch_area] if np.any(pitch_area) else gray
        white_thresh = max(150, int(np.percentile(grass_gray, 90))) if len(grass_gray) > 0 else 170

        white_lines = (
            ((gray >= white_thresh) | (tophat >= 24)) &
            (hsv[:, :, 1] <= 78) &
            pitch_area
        )
        white_line_mask = (white_lines.astype(np.uint8) * 255)

        return grass_cleaned, white_line_mask, pitch_contour

    @classmethod
    def detect_halfway_line(
        cls,
        frame: np.ndarray,
        white_line_mask: np.ndarray,
        pitch_contour: Optional[np.ndarray],
    ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], float]:
        """Detect the midfield halfway line (canonical X = 0.50).

        Returns:
            ((top_point, bot_point), confidence)
        """
        h, w = frame.shape[:2]
        edges = cv2.Canny(white_line_mask, 50, 150)
        min_line_len = max(40, int(h * 0.22))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_line_len, maxLineGap=25)

        if lines is None:
            return None, 0.0

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            angle = float(np.abs(np.arctan2(dy, dx) * 180.0 / np.pi))

            # Transverse line angle (approx vertical in camera perspective)
            if 62.0 <= angle <= 118.0:
                # Ensure top point has smaller y
                pt_top = (float(x1), float(y1)) if y1 < y2 else (float(x2), float(y2))
                pt_bot = (float(x2), float(y2)) if y1 < y2 else (float(x1), float(y1))
                mid_x = (pt_top[0] + pt_bot[0]) / 2.0

                # Score candidate: longer lines and lines near center circle or middle of pitch
                score = length / max(h, 1)

                # Check for center circle arc evidence near the line midpoint
                mid_y = int((pt_top[1] + pt_bot[1]) / 2.0)
                circle_region_w = int(w * 0.15)
                cx1 = max(0, int(mid_x) - circle_region_w)
                cx2 = min(w, int(mid_x) + circle_region_w)
                cy1 = max(0, mid_y - int(h * 0.15))
                cy2 = min(h, mid_y + int(h * 0.15))
                sub_mask = white_line_mask[cy1:cy2, cx1:cx2]
                if np.sum(sub_mask > 0) > 100:
                    score += 0.35  # Bonus for center circle markings

                candidates.append(((pt_top, pt_bot), score))

        if not candidates:
            return None, 0.0

        best_line, best_score = max(candidates, key=lambda c: c[1])
        conf = min(0.95, best_score)
        return best_line, conf

    @classmethod
    def detect_d_line(
        cls,
        frame: np.ndarray,
        white_line_mask: np.ndarray,
        pitch_contour: Optional[np.ndarray],
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Detect the penalty 'D' arc and its adjacent 16.5m penalty line.

        Returns:
            ({"side": "left"|"right", "apex": (u, v), "box_x": u_box}, confidence)
        """
        h, w = frame.shape[:2]
        contours, _ = cv2.findContours(white_line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        candidates = []
        for c in contours:
            if len(c) < 30:
                continue

            pts = c.reshape(-1, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]
            span_y = float(np.max(ys) - np.min(ys))
            span_x = float(np.max(xs) - np.min(xs))

            # Penalty arc covers a moderate vertical and horizontal span
            if span_y < 0.06 * h or span_y > 0.60 * h or span_x < 0.02 * w or span_x > 0.35 * w:
                continue

            # Fit 2nd degree polynomial: u = a*v^2 + b*v + c to measure curvature
            try:
                poly = np.polyfit(ys, xs, 2)
                a, b, c_val = poly[0], poly[1], poly[2]
            except Exception:
                continue

            # Check significant curvature
            if abs(a) < 1e-4:
                continue

            # Parabola vertex: v_vertex = -b / (2a)
            v_vertex = -b / (2.0 * a)
            if not (np.min(ys) <= v_vertex <= np.max(ys)):
                continue

            u_vertex = float(a * (v_vertex ** 2) + b * v_vertex + c_val)

            # Curvature direction:
            # a < 0 -> Parabola opens left, vertex has maximum u (bulges right towards midfield) -> Left Penalty D
            # a > 0 -> Parabola opens right, vertex has minimum u (bulges left towards midfield) -> Right Penalty D
            if a < 0:
                side = "left"
                conf = min(0.90, span_y / (0.35 * h))
            else:
                side = "right"
                conf = min(0.90, span_y / (0.35 * h))

            candidates.append({
                "side": side,
                "apex": (float(u_vertex), float(v_vertex)),
                "span_y": span_y,
                "conf": conf,
            })

        if not candidates:
            return None, 0.0

        best = max(candidates, key=lambda c: c["conf"])
        return best, best["conf"]

    @classmethod
    def detect_pitch_geometry(
        cls, frame: np.ndarray
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[Tuple[float, float]]]]:
        """Detect pitch markings and return source (u, v) and destination (x, y) anchor points."""
        h, w = frame.shape[:2]
        grass_mask, white_line_mask, pitch_contour = cls.extract_grass_and_lines(frame)

        # 1. Landmark detection: Halfway line (midfield) or Penalty D-arc
        half_line, half_conf = cls.detect_halfway_line(frame, white_line_mask, pitch_contour)
        d_line, d_conf = cls.detect_d_line(frame, white_line_mask, pitch_contour)

        # Detect longitudinal touchlines (sidelines)
        edges = cv2.Canny(white_line_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=max(30, int(w * 0.15)), maxLineGap=30)
        horizontal_lines = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l.flatten()
                dx = x2 - x1
                dy = y2 - y1
                length = float(np.hypot(dx, dy))
                angle = float(np.abs(np.arctan2(dy, dx) * 180.0 / np.pi))
                if angle < 28 or angle > 152:
                    horizontal_lines.append((x1, y1, x2, y2, length))

        y_top = 0.12 * h
        y_bot = 0.90 * h
        if horizontal_lines:
            y_pts = [y for _, y1, _, y2, _ in horizontal_lines for y in (y1, y2)]
            y_top = float(np.percentile(y_pts, 10))
            y_bot = float(np.percentile(y_pts, 90))

        # Case A: Halfway line is visible -> Anchor canonical X = 0.50
        if half_line is not None and half_conf >= 0.35:
            (u_top, _), (u_bot, _) = half_line
            u_mid = (u_top + u_bot) / 2.0
            # Broadcast camera typically displays a ~48% horizontal pitch window
            est_window = 0.48
            x_left = max(0.02, 0.50 - (u_mid / w) * est_window)
            x_right = min(0.98, x_left + est_window)

            src_quad = [
                (0.04 * w, y_top),
                (0.96 * w, y_top),
                (0.96 * w, y_bot),
                (0.04 * w, y_bot),
            ]
            dst_quad = [
                (x_left, 0.05),
                (x_right, 0.05),
                (x_right, 0.95),
                (x_left, 0.95),
            ]
            return src_quad, dst_quad

        # Case B: Penalty D-line is visible -> Anchor attacking/defending third
        if d_line is not None and d_conf >= 0.35:
            u_apex, _v_apex = d_line["apex"]
            if d_line["side"] == "left":
                # Left Penalty D arc apex is at canonical X ≈ 0.192
                x_apex_canonical = 0.192
                est_window = 0.45
                x_left = max(0.01, x_apex_canonical - (u_apex / w) * est_window)
                x_right = min(0.99, x_left + est_window)
            else:
                # Right Penalty D arc apex is at canonical X ≈ 0.808
                x_apex_canonical = 0.808
                est_window = 0.45
                x_left = max(0.01, x_apex_canonical - (u_apex / w) * est_window)
                x_right = min(0.99, x_left + est_window)

            src_quad = [
                (0.04 * w, y_top),
                (0.96 * w, y_top),
                (0.96 * w, y_bot),
                (0.04 * w, y_bot),
            ]
            dst_quad = [
                (x_left, 0.05),
                (x_right, 0.05),
                (x_right, 0.95),
                (x_left, 0.95),
            ]
            return src_quad, dst_quad

        # Case C: Fallback to pitch contour convex hull
        if pitch_contour is not None:
            rect = cv2.minAreaRect(pitch_contour)
            box = cv2.boxPoints(rect)
            box = sorted(box, key=lambda p: p[1])
            top_two = sorted(box[:2], key=lambda p: p[0])
            bot_two = sorted(box[2:], key=lambda p: p[0], reverse=True)
            src_quad = [
                (float(top_two[0][0]), float(top_two[0][1])),
                (float(top_two[1][0]), float(top_two[1][1])),
                (float(bot_two[0][0]), float(bot_two[0][1])),
                (float(bot_two[1][0]), float(bot_two[1][1])),
            ]
            dst_quad = [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]
            return src_quad, dst_quad

        return None, None


class CameraMotionTracker:
    """Tracks camera pan, tilt, and zoom across consecutive frames and calibrates with pitch markings."""

    def __init__(self, initial_homography: PitchHomography) -> None:
        self.current_homography = initial_homography
        self.prev_gray: Optional[np.ndarray] = None
        self.cumulative_transform = np.eye(3, dtype=np.float64)

    def update(
        self,
        frame: np.ndarray,
        player_boxes: Optional[Sequence[Sequence[float]]] = None,
    ) -> PitchHomography:
        """Update homography by computing inter-frame camera motion on the static pitch background

        and dynamically re-anchoring with detected halfway lines or penalty D markings.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.current_homography

        # Mask out player bounding boxes so moving players do not bias camera background motion
        h, w = frame.shape[:2]
        motion_mask = np.full((h, w), 255, dtype=np.uint8)
        if player_boxes:
            for box in player_boxes:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                cv2.rectangle(motion_mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 0, -1)

        # Track good features on static pitch markings and turf texture
        pts1 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=motion_mask,
            maxCorners=150,
            qualityLevel=0.015,
            minDistance=12,
        )

        if pts1 is not None and len(pts1) >= 8:
            pts2, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts1, None, winSize=(21, 21), maxLevel=2
            )
            if pts2 is not None and status is not None:
                valid1 = pts1[status.flatten() == 1]
                valid2 = pts2[status.flatten() == 1]
                if len(valid1) >= 8:
                    # Estimate affine camera motion: translation (pan/tilt) and scale (zoom)
                    matrix, inliers = cv2.estimateAffinePartial2D(valid1, valid2, method=cv2.RANSAC)
                    if matrix is not None and inliers is not None and np.sum(inliers) >= 6:
                        # Construct 3x3 inter-frame homography
                        m3x3 = np.eye(3, dtype=np.float64)
                        m3x3[:2, :] = matrix
                        try:
                            m_inv = np.linalg.inv(m3x3)
                            # Update active homography: H_t = H_{t-1} @ M_inv
                            updated_matrix = self.current_homography.matrix @ m_inv
                            self.current_homography = PitchHomography(updated_matrix)
                        except np.linalg.LinAlgError:
                            pass

        # Landmark Re-Anchoring: Check for pitch markings (halfway line or D-line) to eliminate optical flow drift
        src_quad, dst_quad = PitchMarkingDetector.detect_pitch_geometry(frame)
        if src_quad is not None and dst_quad is not None:
            try:
                landmark_homography = PitchHomography.from_four_points(src_quad, dst_quad)
                # Exponential blend: 70% motion tracking continuity + 30% landmark anchoring
                blended_matrix = 0.70 * self.current_homography.matrix + 0.30 * landmark_homography.matrix
                self.current_homography = PitchHomography(blended_matrix)
            except Exception:
                pass

        self.prev_gray = gray
        return self.current_homography



class PitchHomography:
    """Manages perspective transformation from video frames to canonical pitch space."""

    DEFAULT_PITCH_CORNERS = np.array([
        [0.0, 0.0],  # Top-left corner flag
        [1.0, 0.0],  # Top-right corner flag
        [1.0, 1.0],  # Bottom-right corner flag
        [0.0, 1.0],  # Bottom-left corner flag
    ], dtype=np.float32)

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        """Initialize with an optional 3x3 homography matrix."""
        if matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
            self.matrix = matrix.astype(np.float64)
        else:
            self.matrix = np.eye(3, dtype=np.float64)

    @classmethod
    def from_four_points(
        cls,
        src_points: Sequence[Tuple[float, float]],
        dst_points: Sequence[Tuple[float, float]] | None = None,
    ) -> PitchHomography:
        """Compute homography from 4 video pixel coordinates to 4 pitch coordinates."""
        if len(src_points) != 4:
            raise ValueError(f"Expected exactly 4 source points, got {len(src_points)}")

        src_arr = np.array(src_points, dtype=np.float32)
        if dst_points is None:
            dst_arr = cls.DEFAULT_PITCH_CORNERS
        else:
            if len(dst_points) != 4:
                raise ValueError(f"Expected exactly 4 destination points, got {len(dst_points)}")
            dst_arr = np.array(dst_points, dtype=np.float32)

        H, _ = cv2.findHomography(src_arr, dst_arr)
        if H is None:
            raise ValueError("Failed to compute homography matrix from provided points.")
        return cls(H)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        frame_width: int,
        frame_height: int,
    ) -> PitchHomography:
        """Construct an empirical homography matrix based on common broadcast camera presets."""
        w, h = float(frame_width), float(frame_height)

        if preset_name in ("auto", "auto_detect", "auto_broadcast"):
            return cls.from_preset("main_broadcast", frame_width, frame_height)

        elif preset_name == "main_broadcast":
            src = [
                (0.12 * w, 0.16 * h),
                (0.88 * w, 0.16 * h),
                (0.96 * w, 0.90 * h),
                (0.04 * w, 0.90 * h),
            ]
            dst = [
                (0.10, 0.08),
                (0.90, 0.08),
                (0.92, 0.92),
                (0.08, 0.92),
            ]
            return cls.from_four_points(src, dst)

        elif preset_name == "tactical_wide":
            src = [
                (0.06 * w, 0.10 * h),
                (0.94 * w, 0.10 * h),
                (0.98 * w, 0.92 * h),
                (0.02 * w, 0.92 * h),
            ]
            dst = [
                (0.02, 0.02),
                (0.98, 0.02),
                (0.98, 0.98),
                (0.02, 0.98),
            ]
            return cls.from_four_points(src, dst)

        elif preset_name in ("identity", "normalized"):
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            return cls.from_four_points(src, dst)

        else:
            raise ValueError(
                f"Unknown preset: '{preset_name}'. Available: 'auto_detect', 'main_broadcast', 'tactical_wide', 'normalized'."
            )

    @classmethod
    def estimate_from_frame(cls, frame: np.ndarray) -> PitchHomography:
        """Automatically detect pitch markings and structural geometry to compute homography."""
        h, w = frame.shape[:2]
        src_quad, dst_quad = PitchMarkingDetector.detect_pitch_geometry(frame)
        if src_quad is not None and dst_quad is not None:
            try:
                return cls.from_four_points(src_quad, dst_quad)
            except Exception:
                pass
        return cls.from_preset("main_broadcast", w, h)

    def project_point(self, u: float, v: float) -> Tuple[float, float]:
        """Project a single pixel coordinate (u, v) into normalized pitch space (x, y)."""
        pt = np.array([u, v, 1.0], dtype=np.float64)
        proj = self.matrix @ pt
        if abs(proj[2]) < 1e-7:
            return (0.0, 0.0)
        x = float(proj[0] / proj[2])
        y = float(proj[1] / proj[2])
        x_clamped = max(0.0, min(1.0, x))
        y_clamped = max(0.0, min(1.0, y))
        return (x_clamped, y_clamped)

    def project_point_safe(self, u: float, v: float) -> Tuple[Optional[float], Optional[float], bool]:
        """Project pixel coordinate into pitch space without artificial clamping.

        Returns:
            (pitch_x, pitch_y, is_on_pitch)
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        proj = self.matrix @ pt
        if abs(proj[2]) < 1e-7:
            return None, None, False
        x = float(proj[0] / proj[2])
        y = float(proj[1] / proj[2])

        # A valid on-pitch detection must fall within the pitch boundary (+/- 5% margin for touchlines)
        if -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:
            return max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), True
        return None, None, False

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorized projection of an (N, 2) array of pixel coordinates into pitch coordinates."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        n = points.shape[0]
        pts_homo = np.hstack([points, np.ones((n, 1), dtype=np.float64)])
        proj = (self.matrix @ pts_homo.T).T
        denom = proj[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
        res = proj[:, :2] / denom
        return np.clip(res, 0.0, 1.0)
