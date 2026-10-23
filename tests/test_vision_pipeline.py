"""Unit tests for the video perception and game state reconstruction vision pipeline."""

import unittest
import cv2
import numpy as np
import pandas as pd

from src.vision.homography import PitchHomography, PitchMarkingDetector, CameraMotionTracker
from src.vision.jersey_classifier import JerseyClassifier
from src.vision.video_tracker import VideoGameStateTracker


class TestPitchHomography(unittest.TestCase):
    """Test homography projection and preset generation."""

    def test_identity_projection(self):
        h = PitchHomography()
        # Identity matrix projects (u, v) -> (u, v) clamped to [0, 1]
        x, y = h.project_point(0.5, 0.75)
        self.assertAlmostEqual(x, 0.5, places=5)
        self.assertAlmostEqual(y, 0.75, places=5)

    def test_four_point_homography(self):
        # Map a 1920x1080 screen directly to [0, 1] x [0, 1]
        src = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
        dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        h = PitchHomography.from_four_points(src, dst)

        # Center point
        x, y = h.project_point(960, 540)
        self.assertAlmostEqual(x, 0.5, places=2)
        self.assertAlmostEqual(y, 0.5, places=2)

    def test_clamping_bounds(self):
        h = PitchHomography()
        # Out of bounds coordinates should clamp to [0.0, 1.0]
        x_neg, y_neg = h.project_point(-50, -20)
        self.assertEqual(x_neg, 0.0)
        self.assertEqual(y_neg, 0.0)

        x_over, y_over = h.project_point(5.0, 10.0)
        self.assertEqual(x_over, 1.0)
        self.assertEqual(y_over, 1.0)

    def test_vectorized_projection(self):
        h = PitchHomography.from_preset("normalized", 100, 100)
        pts = np.array([[0, 0], [50, 50], [100, 100]], dtype=np.float64)
        proj = h.project_points(pts)
        self.assertEqual(proj.shape, (3, 2))
        self.assertAlmostEqual(proj[1, 0], 0.5, places=2)
        self.assertAlmostEqual(proj[1, 1], 0.5, places=2)

    def test_project_point_safe(self):
        h = PitchHomography()
        # Inside bounds
        px, py, on_pitch = h.project_point_safe(0.5, 0.5)
        self.assertTrue(on_pitch)
        self.assertAlmostEqual(px, 0.5, places=5)
        self.assertAlmostEqual(py, 0.5, places=5)

        # Well outside pitch bounds
        px_out, py_out, on_pitch_out = h.project_point_safe(2.5, 3.0)
        self.assertFalse(on_pitch_out)
        self.assertIsNone(px_out)
        self.assertIsNone(py_out)


class TestPitchMarkingDetector(unittest.TestCase):
    """Test pitch segmentation and geometry detection."""

    def test_extract_grass_and_lines(self):
        # Create synthetic green turf frame
        frame = np.full((120, 160, 3), [45, 150, 45], dtype=np.uint8)
        # Add white marking line
        frame[60:64, 20:140] = [255, 255, 255]

        grass_mask, white_line_mask, pitch_contour = PitchMarkingDetector.extract_grass_and_lines(frame)
        self.assertIsNotNone(grass_mask)
        self.assertIsNotNone(white_line_mask)
        self.assertGreater(np.sum(grass_mask > 0), 100)
        self.assertGreater(np.sum(white_line_mask > 0), 20)

    def test_detect_pitch_geometry_fallback(self):
        # Plain green frame should produce bounding geometry
        frame = np.full((120, 160, 3), [45, 150, 45], dtype=np.uint8)
        src_quad, dst_quad = PitchMarkingDetector.detect_pitch_geometry(frame)
        if src_quad is not None:
            self.assertEqual(len(src_quad), 4)
            self.assertEqual(len(dst_quad), 4)

    def test_detect_halfway_line(self):
        # Green frame with a clear vertical transverse line in the middle
        h, w = 240, 320
        frame = np.full((h, w, 3), [40, 160, 40], dtype=np.uint8)
        # Draw vertical white line (halfway line) at x = 160
        frame[20:220, 158:162] = [255, 255, 255]
        # Draw center circle segment
        cv2.circle(frame, (160, 120), 35, (255, 255, 255), 2)

        _grass, white_lines, contour = PitchMarkingDetector.extract_grass_and_lines(frame)
        line, conf = PitchMarkingDetector.detect_halfway_line(frame, white_lines, contour)
        self.assertIsNotNone(line)
        self.assertGreater(conf, 0.40)
        (pt_top, pt_bot) = line
        mid_x = (pt_top[0] + pt_bot[0]) / 2.0
        self.assertAlmostEqual(mid_x, 160.0, delta=10.0)

    def test_detect_d_line_curvature(self):
        # Green frame with a penalty D arc
        h, w = 240, 320
        frame = np.full((h, w, 3), [40, 160, 40], dtype=np.uint8)
        # Draw arc curving towards the right (+u, towards midfield) -> Left Penalty D
        # cv2.ellipse: center, axes (major/minor), angle, startAngle, endAngle, color, thickness
        cv2.ellipse(frame, (120, 120), (35, 45), 0, -60, 60, (255, 255, 255), 3)

        _grass, white_lines, contour = PitchMarkingDetector.extract_grass_and_lines(frame)
        d_res, conf = PitchMarkingDetector.detect_d_line(frame, white_lines, contour)
        self.assertIsNotNone(d_res)
        self.assertEqual(d_res["side"], "left")
        self.assertGreater(conf, 0.30)


class TestCameraMotionTracker(unittest.TestCase):
    """Test inter-frame homography motion tracking."""

    def test_camera_motion_tracker_initialization(self):
        h0 = PitchHomography()
        tracker = CameraMotionTracker(h0)
        frame1 = np.full((100, 100, 3), [45, 140, 45], dtype=np.uint8)
        h_out = tracker.update(frame1)
        self.assertIsNotNone(h_out)
        np.testing.assert_array_almost_equal(h_out.matrix, h0.matrix)


class TestJerseyClassifier(unittest.TestCase):
    """Test player team classification based on jersey color."""

    def test_torso_color_extraction(self):
        classifier = JerseyClassifier()
        # Create a synthetic 100x100 BGR image with green background and a red box
        frame = np.full((100, 100, 3), [40, 180, 40], dtype=np.uint8)  # green grass
        # Draw a red player (BGR: [0, 0, 255])
        frame[20:80, 40:60] = [0, 0, 255]

        bbox = [40, 20, 60, 80]
        color = classifier.extract_torso_color(frame, bbox)
        self.assertIsNotNone(color)
        # In RGB, red should dominate
        self.assertGreater(color[0], 200)  # Red channel
        self.assertLess(color[2], 50)    # Blue channel

    def test_team_clustering(self):
        classifier = JerseyClassifier()
        # Samples of Team Red vs Team Blue
        samples = [
            ("P1", np.array([220.0, 30.0, 30.0])),  # Red
            ("P2", np.array([210.0, 40.0, 35.0])),  # Red
            ("P3", np.array([30.0, 40.0, 220.0])),  # Blue
            ("P4", np.array([25.0, 35.0, 215.0])),  # Blue
        ]
        classifier.fit_teams(samples)

        team_red = classifier.predict_team("P1", None)
        team_blue = classifier.predict_team("P3", None)
        self.assertNotEqual(team_red, team_blue)

        hex_colors = classifier.get_team_hex_colors()
        self.assertTrue(hex_colors["home"].startswith("#"))
        self.assertTrue(hex_colors["away"].startswith("#"))

    def test_referee_outlier_rejection(self):
        classifier = JerseyClassifier()
        # 5 Red players, 5 Blue players, 1 fluorescent yellow referee
        samples = []
        for i in range(5):
            samples.append((f"H_{i}", np.array([220.0, 35.0, 35.0])))
            samples.append((f"A_{i}", np.array([30.0, 40.0, 220.0])))
        # Referee wearing neon yellow (High R and High G, Low B)
        samples.append(("REF_1", np.array([245.0, 245.0, 10.0])))

        classifier.fit_teams(samples)
        # The two team colors must remain Red and Blue, not yellow!
        c_home = classifier.team_colors["home"]
        c_away = classifier.team_colors["away"]

        # One should be high-red/low-blue, the other high-blue/low-red
        is_red_blue = (
            (c_home[0] > 160 and c_away[2] > 160) or
            (c_home[2] > 160 and c_away[0] > 160)
        )
        self.assertTrue(is_red_blue, f"Clusters unexpectedly tainted: {c_home}, {c_away}")

    def test_ordinary_kit_variation_is_not_a_referee(self):
        classifier = JerseyClassifier()
        classifier.fit_teams([
            ("RED_1", np.array([220.0, 35.0, 35.0])),
            ("RED_2", np.array([210.0, 40.0, 35.0])),
            ("BLUE_1", np.array([30.0, 40.0, 220.0])),
            ("BLUE_2", np.array([25.0, 35.0, 215.0])),
        ])

        self.assertNotEqual(
            classifier.predict_team("ALT_KIT", np.array([190.0, 95.0, 45.0])),
            "referee",
        )

    def test_team_colors_are_stable_when_sample_order_changes(self):
        samples = [
            ("RED_1", np.array([220.0, 35.0, 35.0])),
            ("RED_2", np.array([210.0, 40.0, 35.0])),
            ("BLUE_1", np.array([30.0, 40.0, 220.0])),
            ("BLUE_2", np.array([25.0, 35.0, 215.0])),
        ]
        first = JerseyClassifier()
        second = JerseyClassifier()
        first.fit_teams(samples)
        second.fit_teams(list(reversed(samples)))

        self.assertEqual(first.get_team_hex_colors(), second.get_team_hex_colors())

    def test_missing_color_uses_deterministic_team_fallback(self):
        first = JerseyClassifier()
        second = JerseyClassifier()

        self.assertEqual(first.predict_team("UNKNOWN", None), "home")
        self.assertEqual(first.predict_team("UNKNOWN", None), second.predict_team("UNKNOWN", None))


class TestTrackPersistence(unittest.TestCase):
    def test_tracks_are_not_filled_outside_observations_or_long_gaps(self):
        rows = []
        for frame in [5, 6, 7, 8, 20, 21, 22, 23]:
            rows.append({
                "match_id": "m1",
                "frame": frame,
                "timestamp": frame / 25,
                "player_id": "HOME_1",
                "team": "home",
                "x": 0.2 + frame * 0.001,
                "y": 0.5,
                "confidence": 0.9,
                "visible": True,
            })

        result = VideoGameStateTracker.densify_and_persist_tracks(
            pd.DataFrame(rows), total_frames=30, fps=25
        )

        self.assertEqual(int(result["frame"].min()), 5)
        self.assertEqual(int(result["frame"].max()), 23)
        self.assertFalse(result["frame"].between(10, 19).any())


if __name__ == "__main__":
    unittest.main()

