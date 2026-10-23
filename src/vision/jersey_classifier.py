"""Player team classification based on jersey color clustering."""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Sequence
import numpy as np
import cv2


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """Map an RGB color vector [0..255] to a human-readable football kit color name."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    clipped = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # 1. Monochromatic checks (Achromatic or near-achromatic)
    if v < 42:
        return "Black"
    if s < 32:
        if v >= 195:
            return "White"
        elif v >= 115:
            return "Grey"
        else:
            return "Black"

    # 2. Dark Navy check (often low saturation, dark blue)
    if 98 <= h <= 135 and v < 95:
        return "Navy"

    # 3. Hue ranges (in OpenCV, H is [0, 180])
    if h < 9 or h >= 172:
        if v < 90:
            return "Maroon"
        return "Red"
    elif 9 <= h < 24:
        if s > 70 and v > 150:
            return "Orange"
        return "Orange" if v >= 110 else "Brown"
    elif 24 <= h < 38:
        if s < 60 and v > 180:
            return "Cream"
        return "Yellow"
    elif 38 <= h < 84:
        if h < 46 and s > 120 and v > 180:
            return "Lime"
        return "Green" if v < 180 else "Bright Green"
    elif 84 <= h < 104:
        if s > 100 and v > 160:
            return "Turquoise"
        return "Sky Blue" if v > 150 else "Teal"
    elif 104 <= h < 134:
        if v > 175 and s < 130:
            return "Sky Blue"
        elif v < 90:
            return "Navy"
        elif s > 160 and v > 130:
            return "Royal Blue"
        return "Blue"
    elif 134 <= h < 155:
        return "Purple"
    elif 155 <= h < 172:
        return "Pink"

    return "Red"


def disambiguate_color_names(name0: str, rgb0: np.ndarray, name1: str, rgb1: np.ndarray) -> Tuple[str, str]:
    """Ensure two team color names are distinct and intuitive."""
    if name0 != name1:
        return name0, name1

    # Same base color family: disambiguate by luminance or shade
    lum0 = 0.299 * rgb0[0] + 0.587 * rgb0[1] + 0.114 * rgb0[2]
    lum1 = 0.299 * rgb1[0] + 0.587 * rgb1[1] + 0.114 * rgb1[2]

    if name0 in ("Blue", "Navy", "Sky Blue"):
        if lum0 >= lum1:
            return "Sky Blue", "Navy"
        else:
            return "Navy", "Sky Blue"

    if lum0 > lum1:
        return f"Light {name0}", f"Dark {name1}"
    else:
        return f"Dark {name0}", f"Light {name1}"


class JerseyClassifier:
    """Classifies detected players into named teams based on jersey color."""

    def __init__(self, team_names: Optional[Tuple[str, str]] = None) -> None:
        self.team_names = team_names or ("home", "away")
        self.team_colors: Dict[str, np.ndarray] = {}  # RGB centers
        self.player_colors: Dict[str, np.ndarray] = {}
        self._team_lab_centers: Dict[str, np.ndarray] = {}
        self._fitted = False
        self._player_team_cache: Dict[str, str] = {}
        self._referee_ids: set[str] = set()
        self.raw_team_names: Tuple[str, str] = self.team_names

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

        # Focus tightly on central chest (avoids shorts, arms, neck, and turf)
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

        # 2. Reject pitch shadows and specular glare
        shadow_mask = (hsv[:, :, 2] < 25)
        glare_mask = (hsv[:, :, 2] > 252) & (hsv[:, :, 1] < 15)

        invalid_mask = grass_mask | shadow_mask | glare_mask
        fabric_mask = ~invalid_mask
        fabric_pixels = crop[fabric_mask]

        if len(fabric_pixels) < 5:
            fabric_mask = ~grass_mask
            fabric_pixels = crop[fabric_mask]

        if len(fabric_pixels) < 5:
            fabric_mask = np.ones(hsv.shape[:2], dtype=bool)
            fabric_pixels = crop[fabric_mask]

        chromatic_pixels = crop[fabric_mask & (hsv[:, :, 1] >= 40)]
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
        outlier rejection for referees. Dynamically names teams after their dominant jersey color.
        """
        self._referee_ids.clear()
        self._player_team_cache.clear()
        self.player_colors.clear()

        player_samples_map = defaultdict(list)
        for p_id, col in player_color_samples:
            if col is not None and not np.any(np.isnan(col)):
                player_samples_map[p_id].append(col)

        if not player_samples_map:
            # Fallback classic kits
            c0_rgb = np.array([215.0, 35.0, 35.0])   # Red
            c1_rgb = np.array([25.0, 75.0, 210.0])   # Blue
            t0_name, t1_name = "Red", "Blue"
            self._set_team_profiles(t0_name, c0_rgb, t1_name, c1_rgb)
            return

        unique_players = sorted(player_samples_map)
        if len(unique_players) < 2:
            single_rgb = np.median(player_samples_map[unique_players[0]], axis=0)
            t0_name = rgb_to_color_name(single_rgb)
            contrasting = np.array([255 - single_rgb[0], 255 - single_rgb[1], 255 - single_rgb[2]])
            t1_name = rgb_to_color_name(contrasting)
            t0_name, t1_name = disambiguate_color_names(t0_name, single_rgb, t1_name, contrasting)
            self._set_team_profiles(t0_name, single_rgb, t1_name, contrasting)
            self._player_team_cache[unique_players[0]] = t0_name
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

        # Filter out near-white accent noise from initial centroid clustering
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

        # Step 2: Identify referee tracks (a person with a kit distinct from both team clusters)
        c0_lab = self._rgb_to_lab(c0_rgb)
        c1_lab = self._rgb_to_lab(c1_rgb)
        dists = [min(float(np.linalg.norm(p_lab - c0_lab)), float(np.linalg.norm(p_lab - c1_lab))) for p_lab in player_labs]
        median_team_dist = float(np.median(dists)) if dists else 15.0

        referee_candidates = []
        for p_id, p_lab, p_rgb in zip(unique_players, player_labs, player_rgbs):
            d0 = float(np.linalg.norm(p_lab - c0_lab))
            d1 = float(np.linalg.norm(p_lab - c1_lab))
            min_d = min(d0, d1)
            r, g, b = float(p_rgb[0]), float(p_rgb[1]), float(p_rgb[2])

            is_ref = False
            # 1. High-vis lime / fluorescent yellow referee kit
            if r > 140 and g > 165 and b < 80 and (g - b) > 70:
                is_ref = True
            # 2. Bright turquoise / cyan referee kit
            elif r < 60 and g > 150 and b > 165:
                is_ref = True
            # 3. Vivid magenta / neon pink referee kit
            elif r > 170 and g < 80 and b > 120:
                is_ref = True
            # 4. Pure black referee kit (when neither team is in black)
            elif max(r, g, b) < 45:
                h_max = max(float(np.max(c0_rgb)), float(np.max(c1_rgb)))
                if h_max > 75:
                    is_ref = True

            # 5. Statistical outlier: distinct from both team clusters
            is_outlier = (min_d >= 32.0 and min_d > 1.8 * max(median_team_dist, 10.0))

            if is_ref or is_outlier:
                confidence_score = min_d + (35.0 if is_ref else 0.0)
                referee_candidates.append((p_id, confidence_score))

        referee_candidates.sort(key=lambda c: c[1], reverse=True)
        max_refs = min(2, max(1, len(unique_players) // 8))
        for cand_id, _score in referee_candidates[:max_refs]:
            self._referee_ids.add(cand_id)
            self._player_team_cache[cand_id] = "referee"

        # Step 3: Re-calculate clean team centroids excluding referee(s)
        clean_indices = [
            i for i in team_sample_indices
            if unique_players[i] not in self._referee_ids
        ]
        if len(clean_indices) >= 2:
            clean_labs = player_labs[clean_indices]
            clean_rgbs = player_rgbs[clean_indices]
            _, clabels, _ = cv2.kmeans(
                clean_labs, 2, None, criteria, 15, cv2.KMEANS_PP_CENTERS
            )
            clabels = clabels.flatten()
            m0 = clabels == 0
            m1 = clabels == 1
            if np.any(m0) and np.any(m1):
                c0_rgb = np.median(clean_rgbs[m0], axis=0)
                c1_rgb = np.median(clean_rgbs[m1], axis=0)

        # Name teams based on their inferred jersey colors
        name0 = rgb_to_color_name(c0_rgb)
        name1 = rgb_to_color_name(c1_rgb)
        t0_name, t1_name = disambiguate_color_names(name0, c0_rgb, name1, c1_rgb)

        self._set_team_profiles(t0_name, c0_rgb, t1_name, c1_rgb)

        # Step 4: Assign all clean outfield players to one of the two jersey teams
        for p_id, p_lab in zip(unique_players, player_labs):
            if p_id in self._referee_ids:
                continue
            dist_0 = float(np.linalg.norm(p_lab - self._team_lab_centers[self.team_names[0]]))
            dist_1 = float(np.linalg.norm(p_lab - self._team_lab_centers[self.team_names[1]]))
            self._player_team_cache[p_id] = self.team_names[0] if dist_0 <= dist_1 else self.team_names[1]

    def _set_team_profiles(self, t0_name: str, c0_rgb: np.ndarray, t1_name: str, c1_rgb: np.ndarray) -> None:
        """Establish team names and their RGB/Lab centers, aliasing home/away."""
        self.team_names = (t0_name, t1_name)
        self.team_colors = {
            t0_name: c0_rgb,
            t1_name: c1_rgb,
            "home": c0_rgb,  # Backwards compatibility alias
            "away": c1_rgb,  # Backwards compatibility alias
        }
        self._team_lab_centers = {
            t0_name: self._rgb_to_lab(c0_rgb),
            t1_name: self._rgb_to_lab(c1_rgb),
            "home": self._rgb_to_lab(c0_rgb),
            "away": self._rgb_to_lab(c1_rgb),
        }
        self._fitted = True

    def is_referee(self, player_id: str, color: Optional[np.ndarray] = None) -> bool:
        """Check whether a detected entity is a referee rather than an outfield player."""
        pid_str = str(player_id)
        if pid_str in self._referee_ids:
            return True
        if self._player_team_cache.get(pid_str) == "referee":
            return True
        if self._fitted and color is not None:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            if r > 140 and g > 165 and b < 80 and (g - b) > 70:
                return True
            if r < 60 and g > 150 and b > 165:
                return True
            if r > 170 and g < 80 and b > 120:
                return True
            if max(r, g, b) < 45:
                home_max = float(np.max(self.team_colors.get(self.team_names[0], [100])))
                away_max = float(np.max(self.team_colors.get(self.team_names[1], [100])))
                if home_max > 75 and away_max > 75:
                    return True
        return False

    def predict_team(self, player_id: str, color: np.ndarray | None) -> str:
        """Predict whether a player belongs to team 1, team 2, or referee."""
        pid_str = str(player_id)
        if pid_str in self._player_team_cache:
            return self._player_team_cache[pid_str]

        if self.is_referee(pid_str, color):
            self._referee_ids.add(pid_str)
            self._player_team_cache[pid_str] = "referee"
            return "referee"

        if not self._fitted or color is None:
            team = self.team_names[0]
            self._player_team_cache[pid_str] = team
            return team

        color_lab = self._rgb_to_lab(color)
        dist_0 = float(np.linalg.norm(color_lab - self._team_lab_centers[self.team_names[0]]))
        dist_1 = float(np.linalg.norm(color_lab - self._team_lab_centers[self.team_names[1]]))
        team = self.team_names[0] if dist_0 <= dist_1 else self.team_names[1]
        self._player_team_cache[pid_str] = team
        return team

    def get_team_hex_colors(self) -> Dict[str, str]:
        """Return RGB colors formatted as CSS hex strings for UI display."""
        hex_colors = {}
        for team_name in [self.team_names[0], self.team_names[1], "home", "away"]:
            rgb = self.team_colors.get(team_name, np.array([120, 120, 120]))
            r, g, b = [int(np.clip(c, 0, 255)) for c in rgb[:3]]
            hex_colors[team_name] = f"#{r:02x}{g:02x}{b:02x}"
        return hex_colors

    def get_player_hex_color(self, player_id: str) -> str | None:
        """Return the observed torso color for one track as a CSS hex value."""
        rgb = self.player_colors.get(str(player_id))
        if rgb is None:
            return None
        r, g, b = [int(np.clip(c, 0, 255)) for c in rgb[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    print("=" * 65)
    print(" JerseyClassifier: Inferred Kit Color Detection & Team Naming")
    print("=" * 65)

    classifier = JerseyClassifier()

    # Generate synthetic player torso kit observations
    np.random.seed(42)
    # Team Red kit samples (~10 players)
    red_samples = [
        (f"track_{i}", np.array([215.0 + np.random.normal(0, 8), 35.0 + np.random.normal(0, 5), 40.0 + np.random.normal(0, 5)]))
        for i in range(1, 11)
    ]
    # Team Blue kit samples (~10 players)
    blue_samples = [
        (f"track_{i}", np.array([30.0 + np.random.normal(0, 6), 75.0 + np.random.normal(0, 6), 210.0 + np.random.normal(0, 10)]))
        for i in range(11, 21)
    ]
    # Referee (high-vis lime kit)
    ref_samples = [
        ("track_99", np.array([180.0, 230.0, 30.0]))
    ]

    all_samples = red_samples + blue_samples + ref_samples
    classifier.fit_teams(all_samples)

    print(f"\n[+] Inferred Team 1: {classifier.team_names[0]} -> Hex: {classifier.get_team_hex_colors()[classifier.team_names[0]]}")
    print(f"[+] Inferred Team 2: {classifier.team_names[1]} -> Hex: {classifier.get_team_hex_colors()[classifier.team_names[1]]}")
    print(f"[+] Detected Referees: {classifier._referee_ids}")

    print("\n--- Sample Player Predictions ---")
    for tid in ["track_1", "track_5", "track_11", "track_15", "track_99"]:
        predicted = classifier.predict_team(tid, None)
        print(f"  Player {tid:8s} -> Team: {predicted:10s} (Observed: {classifier.get_player_hex_color(tid)})")
    print("\n[+] JerseyClassifier ready for video game state reconstruction.")
