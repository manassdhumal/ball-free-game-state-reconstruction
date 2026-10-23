"""Player team classification based on jersey color clustering."""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


class JerseyClassifier:
    """Classifies detected players into 'home' and 'away' teams based on jersey color."""

    def __init__(self, team_names: Tuple[str, str] = ("home", "away")) -> None:
        self.team_names = team_names
        self.team_colors: Dict[str, np.ndarray] = {}  # RGB centers
        self.player_colors: Dict[str, np.ndarray] = {}
        self._team_lab_centers: Dict[str, np.ndarray] = {}
        self._fitted = False
        self._player_team_cache: Dict[str, str] = {}
        self._referee_ids: set[str] = set()

    def extract_torso_color(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        """Extract the dominant non-grass, non-skin jersey RGB color from a player's upper torso region.

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

        # Use the central chest; red kit fabric overlaps common HSV skin ranges.
        torso_y1 = y1 + int(0.18 * h)
        torso_y2 = max(torso_y1 + 1, y1 + int(0.46 * h))
        torso_x1 = x1 + int(0.25 * w)
        torso_x2 = max(torso_x1 + 1, x1 + int(0.75 * w))

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Reject green grass
        grass_mask = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] >= 25)

        # Reject pitch shadows and extreme specular glare.
        shadow_mask = (hsv[:, :, 2] < 25)
        glare_mask = (hsv[:, :, 2] > 252) & (hsv[:, :, 1] < 15)

        invalid_mask = grass_mask | shadow_mask | glare_mask
        fabric_mask = ~invalid_mask
        fabric_pixels = crop[fabric_mask]

        if len(fabric_pixels) < 5:
            # Fallback 1: non-grass pixels
            fabric_mask = ~grass_mask
            fabric_pixels = crop[fabric_mask]

        if len(fabric_pixels) < 5:
            # Fallback 2: all pixels in crop
            fabric_mask = np.ones(hsv.shape[:2], dtype=bool)
            fabric_pixels = crop[fabric_mask]

        chromatic_pixels = crop[fabric_mask & (hsv[:, :, 1] >= 45)]
        if len(chromatic_pixels) >= max(5, int(0.06 * len(fabric_pixels))):
            fabric_pixels = chromatic_pixels

        # Convert valid pixels to RGB
        rgb_pixels = fabric_pixels[:, ::-1].astype(np.float32)

        # Extract dominant color mode (distinguish primary kit from sponsor accents/stripes)
        if len(rgb_pixels) >= 16:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                rgb_pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            c0_count = np.sum(labels == 0)
            c1_count = len(labels) - c0_count
            dominant_rgb = centers[0] if c0_count >= c1_count else centers[1]
            return dominant_rgb.astype(np.float64)

        return np.median(rgb_pixels, axis=0).astype(np.float64)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert an RGB color vector [0..255] to CIE-Lab."""
        clipped = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(clipped, cv2.COLOR_RGB2Lab)
        return lab[0, 0].astype(np.float32)

    def fit_teams(self, player_color_samples: List[Tuple[str, np.ndarray]]) -> None:
        """Cluster extracted player colors into two distinct team color profiles.

        Aggregates samples per player track and operates in CIE-Lab color space with
        outlier rejection for referees and goalkeepers.
        """
        self._referee_ids.clear()
        self._player_team_cache.clear()
        self.player_colors.clear()

        player_samples_map = defaultdict(list)
        for p_id, col in player_color_samples:
            if col is not None and not np.any(np.isnan(col)):
                player_samples_map[p_id].append(col)

        if not player_samples_map:
            self.team_colors[self.team_names[0]] = np.array([200.0, 50.0, 50.0])  # Red
            self.team_colors[self.team_names[1]] = np.array([50.0, 50.0, 200.0])  # Blue
            self._team_lab_centers[self.team_names[0]] = self._rgb_to_lab(self.team_colors[self.team_names[0]])
            self._team_lab_centers[self.team_names[1]] = self._rgb_to_lab(self.team_colors[self.team_names[1]])
            self._fitted = True
            return

        unique_players = sorted(player_samples_map)
        if len(unique_players) < 2:
            single_rgb = np.median(player_samples_map[unique_players[0]], axis=0)
            self.player_colors[unique_players[0]] = single_rgb
            self.team_colors[self.team_names[0]] = single_rgb
            contrasting = np.array([255 - single_rgb[0], 255 - single_rgb[1], 255 - single_rgb[2]])
            self.team_colors[self.team_names[1]] = contrasting
            self._team_lab_centers[self.team_names[0]] = self._rgb_to_lab(single_rgb)
            self._team_lab_centers[self.team_names[1]] = self._rgb_to_lab(contrasting)
            self._player_team_cache[unique_players[0]] = self.team_names[0]
            self._fitted = True
            return

        player_median_rgbs = []
        player_median_labs = []

        for p_id in unique_players:
            rgbs = np.array(player_samples_map[p_id], dtype=np.float32)
            med_rgb = np.median(rgbs, axis=0)
            self.player_colors[p_id] = med_rgb
            player_median_rgbs.append(med_rgb)
            player_median_labs.append(self._rgb_to_lab(med_rgb))

        player_labs = np.array(player_median_labs, dtype=np.float32)
        player_rgbs = np.array(player_median_rgbs, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        player_hsv = cv2.cvtColor(
            np.clip(player_rgbs, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
            cv2.COLOR_RGB2HSV,
        ).reshape(-1, 3)
        # Bright, nearly white fabric is often a shirt accent or lighting, not a third team.
        team_sample_indices = np.flatnonzero(
            (player_hsv[:, 1] >= 28) | (player_hsv[:, 2] < 145)
        )
        if len(team_sample_indices) < 2:
            team_sample_indices = np.arange(len(unique_players))

        team_labs = player_labs[team_sample_indices]
        team_rgbs = player_rgbs[team_sample_indices]
        cv2.setRNGSeed(0)
        k_clusters = min(2, len(team_sample_indices))
        _, labels, centers = cv2.kmeans(
            team_labs, k_clusters, None, criteria, 15, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

        c0_players_mask = labels == 0
        c1_players_mask = labels == 1 if k_clusters > 1 else labels == 0

        c0_rgb = np.median(team_rgbs[c0_players_mask], axis=0) if np.any(c0_players_mask) else team_rgbs[0]
        c1_rgb = np.median(team_rgbs[c1_players_mask], axis=0) if np.any(c1_players_mask) else team_rgbs[-1]

        # Order teams deterministically: brighter kit to team 1, darker kit to team 0 (or vice versa)
        c0_lum = 0.299 * c0_rgb[0] + 0.587 * c0_rgb[1] + 0.114 * c0_rgb[2]
        c1_lum = 0.299 * c1_rgb[0] + 0.587 * c1_rgb[1] + 0.114 * c1_rgb[2]
        if c1_lum < c0_lum:
            c0_rgb, c1_rgb = c1_rgb, c0_rgb

        self.team_colors[self.team_names[0]] = c0_rgb
        self.team_colors[self.team_names[1]] = c1_rgb
        self._team_lab_centers[self.team_names[0]] = self._rgb_to_lab(c0_rgb)
        self._team_lab_centers[self.team_names[1]] = self._rgb_to_lab(c1_rgb)
        self._fitted = True

        # Assign all players reliably to their true team
        for p_id, p_lab, p_rgb in zip(unique_players, player_labs, player_rgbs):
            dist_0 = float(np.linalg.norm(p_lab - self._team_lab_centers[self.team_names[0]]))
            dist_1 = float(np.linalg.norm(p_lab - self._team_lab_centers[self.team_names[1]]))

            is_ref = False
            # Only genuine referee kit signatures (high fluorescence neon lime, magenta, turquoise)
            if (p_rgb[0] > 140 and p_rgb[1] > 165 and p_rgb[2] < 80 and (p_rgb[1] - p_rgb[2]) > 70):
                is_ref = True
            elif (p_rgb[0] > 170 and p_rgb[1] < 80 and p_rgb[2] > 120):
                is_ref = True
            elif (p_rgb[0] < 50 and p_rgb[1] > 160 and p_rgb[2] > 170):
                is_ref = True

            if is_ref:
                self._referee_ids.add(p_id)
                self._player_team_cache[p_id] = "referee"
            else:
                team = self.team_names[0] if dist_0 <= dist_1 else self.team_names[1]
                self._player_team_cache[p_id] = team

    def is_referee(self, player_id: str, color: Optional[np.ndarray] = None) -> bool:
        """Check whether a detected entity is a referee rather than an outfield player."""
        if str(player_id) in self._referee_ids:
            return True
        if self._player_team_cache.get(str(player_id)) == "referee":
            return True
        if self._fitted and color is not None:
            # Check distinctive fluorescent referee kit colors
            if color[0] > 140 and color[1] > 165 and color[2] < 80 and (color[1] - color[2]) > 70:
                return True
            if color[0] > 170 and color[1] < 80 and color[2] > 120:
                return True
            if color[0] < 50 and color[1] > 160 and color[2] > 170:
                return True
        return False

    def predict_team(self, player_id: str, color: np.ndarray | None) -> str:
        """Predict whether a player belongs to 'home', 'away', or 'referee'."""
        pid_str = str(player_id)
        if pid_str in self._player_team_cache:
            return self._player_team_cache[pid_str]

        if not self._fitted or color is None:
            team = self.team_names[0]
            self._player_team_cache[pid_str] = team
            return team

        color_lab = self._rgb_to_lab(color)
        dist_0 = float(np.linalg.norm(color_lab - self._team_lab_centers[self.team_names[0]]))
        dist_1 = float(np.linalg.norm(color_lab - self._team_lab_centers[self.team_names[1]]))
        if self.is_referee(pid_str, color):
            self._referee_ids.add(pid_str)
            self._player_team_cache[pid_str] = "referee"
            return "referee"

        team = self.team_names[0] if dist_0 <= dist_1 else self.team_names[1]
        self._player_team_cache[pid_str] = team
        return team

    def get_team_hex_colors(self) -> Dict[str, str]:
        """Return RGB colors formatted as CSS hex strings for UI display."""
        hex_colors = {}
        for team_name in self.team_names:
            rgb = self.team_colors.get(team_name, np.array([120, 120, 120]))
            r, g, b = [int(np.clip(c, 0, 255)) for c in rgb[:3]]
            hex_colors[team_name] = f"#{r:02x}{g:02x}{b:02x}"
        return hex_colors

    def get_player_hex_color(self, player_id: str) -> str | None:
        """Return the robust observed torso color for one track as a CSS hex value."""
        rgb = self.player_colors.get(str(player_id))
        if rgb is None:
            return None
        r, g, b = [int(np.clip(c, 0, 255)) for c in rgb[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

