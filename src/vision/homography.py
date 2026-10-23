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

                has_circle_support = cls._has_center_circle_support(
                    white_line_mask, pt_top, pt_bot
                )
                if has_circle_support:
                    score += 0.35

                confidence = (
                    min(0.95, 0.55 + 0.4 * length / max(h, 1))
                    if has_circle_support
                    else min(0.44, length / max(h, 1))
                )
                candidates.append(((pt_top, pt_bot), score, confidence))

        if not candidates:
            return None, 0.0

        best_line, _best_score, conf = max(candidates, key=lambda c: c[1])
        return best_line, conf

    @staticmethod
    def _has_center_circle_support(
        white_line_mask: np.ndarray,
        pt_top: Tuple[float, float],
        pt_bot: Tuple[float, float],
    ) -> bool:
        return PitchMarkingDetector._center_circle_pair(white_line_mask, pt_top, pt_bot) is not None

    @staticmethod
    def _center_circle_pair(
        white_line_mask: np.ndarray,
        pt_top: Tuple[float, float],
        pt_bot: Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Require two opposite, similarly sized circle arcs around a candidate midfield line."""
        h, w = white_line_mask.shape
        mask = white_line_mask.copy()
        cv2.line(
            mask,
            tuple(map(int, pt_top)),
            tuple(map(int, pt_bot)),
            0,
            max(7, int(w * 0.012)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        arcs = []
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (
                len(contour) < 12
                or box_w < 0.06 * w
                or box_w > 0.30 * w
                or box_h < 0.04 * h
                or box_h > 0.32 * h
                or area < 100
                or area / max(1, box_w * box_h) < 0.045
            ):
                continue
            center = (x + box_w / 2.0, y + box_h / 2.0)
            radius = max(box_w, box_h) / 2.0
            arcs.append((center, radius, (x, y, box_w, box_h)))

        line = np.asarray(pt_bot, dtype=np.float64) - np.asarray(pt_top, dtype=np.float64)
        line_length = float(np.linalg.norm(line))
        if line_length < 1.0:
            return None

        def signed_distance(center: Tuple[float, float]) -> float:
            offset = np.asarray(center, dtype=np.float64) - np.asarray(pt_top, dtype=np.float64)
            return float((line[0] * offset[1] - line[1] * offset[0]) / line_length)

        left_candidates = []
        right_candidates = []
        for index, (center_a, radius_a, bounds_a) in enumerate(arcs):
            distance_a = signed_distance(center_a)
            for center_b, radius_b, bounds_b in arcs[index + 1:]:
                distance_b = signed_distance(center_b)
                if distance_a * distance_b >= 0:
                    continue
                smaller = min(abs(distance_a), abs(distance_b))
                larger = max(abs(distance_a), abs(distance_b))
                if smaller < 0.35 * min(radius_a, radius_b):
                    continue
                if larger > 1.45 * max(radius_a, radius_b):
                    continue
                if smaller / max(larger, 1.0) < 0.62:
                    continue
                if abs(center_a[1] - center_b[1]) > 0.12 * h:
                    continue
                circle_y = (center_a[1] + center_b[1]) / 2.0
                line_x = (
                    pt_top[0] + (circle_y - pt_top[1]) * line[0] / line[1]
                    if abs(line[1]) > 1e-6
                    else pt_top[0]
                )
                left, right = (
                    (bounds_a, bounds_b) if center_a[0] < center_b[0] else (bounds_b, bounds_a)
                )
                left_center = left[0] + left[2] / 2.0
                right_center = right[0] + right[2] / 2.0
                if left_center >= line_x or right_center <= line_x:
                    continue
                left_candidates.append(left)
                right_candidates.append(right)

        if not left_candidates:
            return None
        left = max(left_candidates, key=lambda box: box[0])
        right = min(right_candidates, key=lambda box: box[0] + box[2])
        center_y = (left[1] + left[3] / 2.0 + right[1] + right[3] / 2.0) / 2.0
        return (float(left[0]), center_y), (float(right[0] + right[2]), center_y)

    @classmethod
    def halfway_pitch_anchors(
        cls, frame: np.ndarray
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """Build full-pitch anchors only when the halfway line reaches both touchlines."""
        h, w = frame.shape[:2]
        _grass, white_lines, pitch_contour = cls.extract_grass_and_lines(frame)
        if pitch_contour is None:
            return None

        halfway, confidence = cls.detect_halfway_line(frame, white_lines, pitch_contour)
        if halfway is None or confidence < 0.65:
            return None

        pt_top, pt_bottom = halfway
        if pt_top[1] > pt_bottom[1]:
            pt_top, pt_bottom = pt_bottom, pt_top

        endpoint_tolerance = max(18.0, 0.05 * h)
        for point in (pt_top, pt_bottom):
            boundary_distance = abs(
                cv2.pointPolygonTest(pitch_contour, (float(point[0]), float(point[1])), True)
            )
            if boundary_distance > endpoint_tolerance:
                return None

        circle_pair = cls._center_circle_pair(white_lines, pt_top, pt_bottom)
        if circle_pair is None:
            return None

        circle_left, circle_right = circle_pair
        circle_diameter = circle_right[0] - circle_left[0]
        if circle_diameter < 0.05 * w:
            return None

        half_circle_normalized_radius = (9.15 / 105.0)
        src = [pt_top, circle_right, pt_bottom, circle_left]
        dst = [
            (0.50, 0.02),
            (0.50 + half_circle_normalized_radius, 0.50),
            (0.50, 0.98),
            (0.50 - half_circle_normalized_radius, 0.50),
        ]
        return src, dst

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

        halfway_anchors = cls.halfway_pitch_anchors(frame)
        if halfway_anchors is not None:
            return halfway_anchors

        # 1. Landmark detection: Halfway line (midfield) or Penalty D-arc
        half_line, half_conf = cls.detect_halfway_line(frame, white_line_mask, pitch_contour)
        d_line, d_conf = cls.detect_d_line(frame, white_line_mask, pitch_contour)

        # Baseline sideline vertical limits
        v_top = 0.14 * h
        v_bot = 0.90 * h

        # If halfway line is detected, its endpoints directly touch the far & near touchlines!
        if half_line is not None and half_conf >= 0.50:
            (u_top_pt, v_top_pt), (u_bot_pt, v_bot_pt) = half_line
            if v_top_pt > v_bot_pt:
                u_top_pt, u_bot_pt = u_bot_pt, u_top_pt
                v_top_pt, v_bot_pt = v_bot_pt, v_top_pt

            if (v_bot_pt - v_top_pt) >= 0.40 * h:
                v_top = float(v_top_pt)
                v_bot = float(v_bot_pt)
        else:
            # Fallback: Detect longitudinal touchlines (sidelines)
            edges = cv2.Canny(white_line_mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=max(30, int(w * 0.15)), maxLineGap=30)
            if lines is not None:
                top_candidates = []
                bot_candidates = []
                for l in lines:
                    x1, y1, x2, y2 = l.flatten()
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = float(np.abs(np.arctan2(dy, dx) * 180.0 / np.pi))
                    if angle < 28 or angle > 152:
                        y_mid = (y1 + y2) / 2.0
                        if y_mid <= 0.45 * h:
                            top_candidates.append(y_mid)
                        elif y_mid >= 0.60 * h:
                            bot_candidates.append(y_mid)

                if top_candidates:
                    v_top = float(np.median(top_candidates))
                if bot_candidates:
                    v_bot = float(np.median(bot_candidates))

                if (v_bot - v_top) < 0.42 * h:
                    v_top = 0.14 * h
                    v_bot = 0.90 * h

        return None, None

    @classmethod
    def detect_pitch_boundaries_and_angles(
        cls,
        frame: np.ndarray,
        white_line_mask: np.ndarray,
        pitch_contour: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Measure pitch line angles, perspective convergence, and horizontal/vertical pitch boundaries.

        Filters out managers, bench substitutes, linesmen, and spectators who stand outside the playing lines.
        """
        h, w = frame.shape[:2]
        edges = cv2.Canny(white_line_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30,
            minLineLength=max(25, int(min(w, h) * 0.08)), maxLineGap=25
        )

        top_candidates = []
        bot_candidates = []
        transverse_angles = []
        transverse_lines = []

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l.flatten()
                dx = x2 - x1
                dy = y2 - y1
                length = float(np.hypot(dx, dy))
                angle = float(np.abs(np.arctan2(dy, dx) * 180.0 / np.pi))

                # Longitudinal lines: parallel to touchlines (near-horizontal)
                if angle <= 25.0 or angle >= 155.0:
                    y_mid = (y1 + y2) / 2.0
                    if y_mid <= 0.48 * h:
                        top_candidates.append((y_mid, length))
                    elif y_mid >= 0.52 * h:
                        bot_candidates.append((y_mid, length))

                # Transverse lines: perpendicular to touchlines (halfway, penalty box, goal line)
                elif 62.0 <= angle <= 118.0:
                    transverse_angles.append(angle)
                    transverse_lines.append(((x1, y1), (x2, y2), length))

        # Determine touchline limits in image space
        v_top = 0.12 * h
        v_bot = 0.92 * h
        if top_candidates:
            top_candidates.sort(key=lambda item: item[1], reverse=True)
            v_top = float(np.median([item[0] for item in top_candidates[:5]]))
        if bot_candidates:
            bot_candidates.sort(key=lambda item: item[1], reverse=True)
            v_bot = float(np.median([item[0] for item in bot_candidates[:5]]))

        if (v_bot - v_top) < 0.38 * h:
            v_top = 0.12 * h
            v_bot = 0.92 * h

        # Calculate camera tilt and perspective inclination angle
        mean_transverse_angle = float(np.median(transverse_angles)) if transverse_angles else 90.0
        pitch_inclination = abs(90.0 - mean_transverse_angle)

        # Estimate perspective vanishing line / point if converging transverse lines exist
        vanishing_y = -float(h) * 1.5
        if len(transverse_lines) >= 2:
            try:
                # Find intersection of transverse lines
                p1a, p1b, _ = transverse_lines[0]
                p2a, p2b, _ = transverse_lines[-1]
                # Line 1: A1 u + B1 v = C1
                A1 = p1b[1] - p1a[1]
                B1 = p1a[0] - p1b[0]
                C1 = A1 * p1a[0] + B1 * p1a[1]
                # Line 2: A2 u + B2 v = C2
                A2 = p2b[1] - p2a[1]
                B2 = p2a[0] - p2b[0]
                C2 = A2 * p2a[0] + B2 * p2a[1]
                det = A1 * B2 - A2 * B1
                if abs(det) > 1e-5:
                    vanishing_y = float((A1 * C2 - A2 * C1) / det)
            except Exception:
                pass

        return {
            "top_touchline_v": float(v_top),
            "bottom_touchline_v": float(v_bot),
            "pitch_inclination_deg": float(round(pitch_inclination, 2)),
            "transverse_angle_deg": float(round(mean_transverse_angle, 2)),
            "vanishing_y": float(round(vanishing_y, 2)),
            "has_boundaries": True,
        }

    @classmethod
    def is_point_on_pitch(
        cls,
        u: float,
        v: float,
        boundaries: Dict[str, Any],
        pitch_contour: Optional[np.ndarray] = None,
    ) -> bool:
        """Check whether a player's ground contact point (u, v) is inside the playable field lines."""
        # 1. Pitch contour test (if green grass polygon was extracted)
        if pitch_contour is not None:
            dist = cv2.pointPolygonTest(pitch_contour, (float(u), float(v)), True)
            if dist < -4.0:  # Outside playable turf polygon (bench area / running track / crowd)
                return False

        # 2. Touchline boundary limits (near and far sidelines)
        v_top = boundaries.get("top_touchline_v", 0.0)
        v_bot = boundaries.get("bottom_touchline_v", 9999.0)
        # Margin of 4 pixels to account for line width
        if v < (v_top - 4.0) or v > (v_bot + 6.0):
            return False

        return True


class CameraMotionTracker:
    """Tracks camera pan, tilt, and zoom across consecutive frames with smooth pitch landmark calibration."""

    def __init__(self, initial_homography: PitchHomography) -> None:
        self.current_homography = initial_homography
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pitch_mask: Optional[np.ndarray] = None
        self.cumulative_transform = np.eye(3, dtype=np.float64)
        self._frame_count = 0

    def update(
        self,
        frame: np.ndarray,
        player_boxes: Optional[Sequence[Sequence[float]]] = None,
        pitch_mask: Optional[np.ndarray] = None,
    ) -> PitchHomography:
        """Update homography by computing inter-frame camera motion on the static pitch background

        and smoothly correcting pitch landmark drift.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pitch_mask = pitch_mask.copy() if pitch_mask is not None else None
            return self.current_homography
        if not self.current_homography.calibrated:
            if self._frame_count % 5 == 0:
                anchors = PitchMarkingDetector.halfway_pitch_anchors(frame)
                if anchors is not None:
                    self.current_homography = PitchHomography.from_four_points(*anchors)
            self.prev_gray = gray
            self.prev_pitch_mask = pitch_mask.copy() if pitch_mask is not None else None
            return self.current_homography

        h, w = frame.shape[:2]
        # Smooth pitch landmark tracking (Halfway line at X=0.50, Penalty D at X=0.192 or 0.808)
        if self._frame_count % 5 == 0:
            _grass, white_line_mask, pitch_contour = PitchMarkingDetector.extract_grass_and_lines(frame)
            anchors = PitchMarkingDetector.halfway_pitch_anchors(frame)
            if anchors is not None:
                source_points, target_points = anchors
                observed = PitchHomography.from_four_points(source_points, target_points)
                blend = 0.25
                new_mat = (
                    (1.0 - blend) * self.current_homography.matrix
                    + blend * observed.matrix
                )
                if abs(new_mat[2, 2]) > 1e-7:
                    new_mat /= new_mat[2, 2]
                self.current_homography = PitchHomography(new_mat)
            elif pitch_contour is not None:
                half_line, half_conf = PitchMarkingDetector.detect_halfway_line(frame, white_line_mask, pitch_contour)
                if half_line is not None and half_conf >= 0.40:
                    (pt_top, pt_bot) = half_line
                    mid_u = (pt_top[0] + pt_bot[0]) / 2.0
                    mid_v = (pt_top[1] + pt_bot[1]) / 2.0
                    proj_x, _ = self.current_homography.project_point(mid_u, mid_v)
                    err_x = 0.50 - proj_x
                    if abs(err_x) > 0.005:
                        shift = float(np.clip(0.18 * err_x, -0.025, 0.025))
                        trans_mat = np.array([
                            [1.0, 0.0, shift],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float64)
                        new_mat = trans_mat @ self.current_homography.matrix
                        if abs(new_mat[2, 2]) > 1e-7:
                            new_mat /= new_mat[2, 2]
                        self.current_homography = PitchHomography(new_mat)
                else:
                    d_line, d_conf = PitchMarkingDetector.detect_d_line(frame, white_line_mask, pitch_contour)
                    if d_line is not None and d_conf >= 0.40:
                        apex_u, apex_v = d_line["apex"]
                        proj_x, _ = self.current_homography.project_point(apex_u, apex_v)
                        target_x = 0.192 if d_line["side"] == "left" else 0.808
                        err_x = target_x - proj_x
                        if abs(err_x) > 0.005:
                            shift = float(np.clip(0.18 * err_x, -0.025, 0.025))
                            trans_mat = np.array([
                                [1.0, 0.0, shift],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ], dtype=np.float64)
                            new_mat = trans_mat @ self.current_homography.matrix
                            if abs(new_mat[2, 2]) > 1e-7:
                                new_mat /= new_mat[2, 2]
                            self.current_homography = PitchHomography(new_mat)

        self.prev_gray = gray
        self.prev_pitch_mask = pitch_mask.copy() if pitch_mask is not None else None
        return self.current_homography




class PitchHomography:
    """Manages perspective transformation from video frames to canonical pitch space."""

    DEFAULT_PITCH_CORNERS = np.array([
        [0.0, 0.0],  # Top-left corner flag
        [1.0, 0.0],  # Top-right corner flag
        [1.0, 1.0],  # Bottom-right corner flag
        [0.0, 1.0],  # Bottom-left corner flag
    ], dtype=np.float32)

    def __init__(self, matrix: np.ndarray | None = None, calibrated: bool = True) -> None:
        """Initialize with an optional 3x3 homography matrix."""
        if matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
            self.matrix = matrix.astype(np.float64)
        else:
            self.matrix = np.eye(3, dtype=np.float64)
        self.calibrated = calibrated

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
            return cls(np.eye(3, dtype=np.float64), calibrated=False)

        elif preset_name == "main_broadcast":
            src = [
                (0.08 * w, 0.20 * h),
                (0.92 * w, 0.20 * h),
                (0.98 * w, 0.96 * h),
                (0.02 * w, 0.96 * h),
            ]
            dst = [
                (0.06, 0.02),
                (0.94, 0.02),
                (0.94, 0.98),
                (0.06, 0.98),
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
        src_quad, dst_quad = PitchMarkingDetector.detect_pitch_geometry(frame)
        if src_quad is not None and dst_quad is not None:
            try:
                return cls.from_four_points(src_quad, dst_quad)
            except Exception:
                pass
        return cls(np.eye(3, dtype=np.float64), calibrated=False)

    def project_raw(self, u: float, v: float) -> Tuple[float, float]:
        """Project pixel coordinate into pitch space without artificial clamping."""
        pt = np.array([u, v, 1.0], dtype=np.float64)
        proj = self.matrix @ pt
        if abs(proj[2]) < 1e-7:
            return (0.0, 0.0)
        return float(proj[0] / proj[2]), float(proj[1] / proj[2])

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
