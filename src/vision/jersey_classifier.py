"""Player team classification based on jersey color clustering."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


class JerseyClassifier:
    """Classifies detected players into 'home' and 'away' teams based on jersey color."""

    def __init__(self, team_names: Tuple[str, str] = ("home", "away")) -> None:
        self.team_names = team_names
        self.team_colors: Dict[str, np.ndarray] = {}  # RGB centers
        self._fitted = False
        self._player_team_cache: Dict[str, str] = {}

    def extract_torso_color(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        """Extract the dominant non-grass RGB color from a player's upper torso region.

        Args:
            frame: Full video frame in BGR format.
            bbox: [x1, y1, x2, y2] bounding box coordinates.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame, x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame, y2))

        w = x2 - x1
        h = y2 - y1
        if w < 2 or h < 4:
            return None

        # Torso region: vertically 10% to 55%, horizontally inner 15% to 85%
        torso_y1 = y1 + int(0.10 * h)
        torso_y2 = max(torso_y1 + 1, y1 + int(0.55 * h))
        torso_x1 = x1 + int(0.15 * w)
        torso_x2 = max(torso_x1 + 1, x1 + int(0.85 * w))

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Filter out green grass using HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass_mask = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] >= 35)
        non_grass_pixels = crop[~grass_mask]

        if len(non_grass_pixels) < 3:
            # Fallback to median color of the crop if mostly grass
            non_grass_pixels = crop.reshape(-1, 3)

        # Convert to RGB and return median color vector
        rgb_pixels = non_grass_pixels[:, ::-1]
        return np.median(rgb_pixels, axis=0)

    def fit_teams(self, player_color_samples: List[Tuple[str, np.ndarray]]) -> None:
        """Cluster extracted player colors into two distinct team color profiles.

        Args:
            player_color_samples: List of (player_id, rgb_color_array).
        """
        if len(player_color_samples) < 2:
            # Not enough samples to cluster; assign default colors
            self.team_colors[self.team_names[0]] = np.array([200.0, 50.0, 50.0])  # Red
            self.team_colors[self.team_names[1]] = np.array([50.0, 50.0, 200.0])  # Blue
            self._fitted = True
            return

        colors = np.array([c for _, c in player_color_samples], dtype=np.float32)

        # Run 2-means clustering using OpenCV
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            colors, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        self.team_colors[self.team_names[0]] = centers[0]
        self.team_colors[self.team_names[1]] = centers[1]
        self._fitted = True

        # Cache initial player assignments
        for (p_id, _), label in zip(player_color_samples, labels.flatten()):
            team = self.team_names[label]
            self._player_team_cache[p_id] = team

    def predict_team(self, player_id: str, color: np.ndarray | None) -> str:
        """Predict whether a player belongs to 'home' or 'away'."""
        if player_id in self._player_team_cache:
            return self._player_team_cache[player_id]

        if not self._fitted or color is None:
            # Deterministic hash fallback if unfitted
            team = self.team_names[hash(player_id) % 2]
            self._player_team_cache[player_id] = team
            return team

        # Nearest team center in RGB color space
        dist_0 = float(np.linalg.norm(color - self.team_colors[self.team_names[0]]))
        dist_1 = float(np.linalg.norm(color - self.team_colors[self.team_names[1]]))
        team = self.team_names[0] if dist_0 <= dist_1 else self.team_names[1]
        self._player_team_cache[player_id] = team
        return team
