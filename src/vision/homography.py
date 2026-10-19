"""Pitch homography and coordinate projection for video game state reconstruction.

Transforms pixel coordinates from a broadcast/tactical video stream into
normalized 2D pitch coordinates [0.0, 1.0] x [0.0, 1.0].
"""

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any
import numpy as np
import cv2


class PitchHomography:
    """Manages perspective transformation from video frames to canonical pitch space."""

    # Default preset points in normalized pitch coordinates:
    # (0.0, 0.0) is top-left, (1.0, 1.0) is bottom-right.
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
        """Compute homography from 4 video pixel coordinates to 4 pitch coordinates.

        Args:
            src_points: 4 (u, v) points in video pixel space, ordered:
                        [top-left, top-right, bottom-right, bottom-left].
            dst_points: 4 (x, y) target pitch coordinates. If None, uses DEFAULT_PITCH_CORNERS.
        """
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

        if preset_name == "main_broadcast":
            # Typical elevated midfield broadcast camera covering attacking/defending thirds
            src = [
                (0.18 * w, 0.22 * h),  # Top-left pitch boundary
                (0.82 * w, 0.22 * h),  # Top-right pitch boundary
                (0.98 * w, 0.88 * h),  # Bottom-right near sideline
                (0.02 * w, 0.88 * h),  # Bottom-left near sideline
            ]
            dst = [
                (0.15, 0.05),
                (0.85, 0.05),
                (0.95, 0.95),
                (0.05, 0.95),
            ]
            return cls.from_four_points(src, dst)

        elif preset_name == "tactical_wide":
            # High tactical view covering full pitch
            src = [
                (0.08 * w, 0.15 * h),
                (0.92 * w, 0.15 * h),
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

        elif preset_name == "identity" or preset_name == "normalized":
            # Direct mapping from normalized pixel coordinates
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            return cls.from_four_points(src, dst)

        else:
            raise ValueError(
                f"Unknown preset: '{preset_name}'. Available: 'main_broadcast', 'tactical_wide', 'normalized'."
            )

    def project_point(self, u: float, v: float) -> Tuple[float, float]:
        """Project a single pixel coordinate (u, v) into normalized pitch space (x, y)."""
        pt = np.array([u, v, 1.0], dtype=np.float64)
        proj = self.matrix @ pt
        if abs(proj[2]) < 1e-7:
            return (0.0, 0.0)
        x = float(proj[0] / proj[2])
        y = float(proj[1] / proj[2])
        # Bound to pitch dimensions with slight tolerance for off-pitch lines
        x_clamped = max(0.0, min(1.0, x))
        y_clamped = max(0.0, min(1.0, y))
        return (x_clamped, y_clamped)

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorized projection of an (N, 2) array of pixel coordinates into pitch coordinates."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        n = points.shape[0]
        pts_homo = np.hstack([points, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
        proj = (self.matrix @ pts_homo.T).T  # (N, 3)
        denom = proj[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
        res = proj[:, :2] / denom
        return np.clip(res, 0.0, 1.0)
