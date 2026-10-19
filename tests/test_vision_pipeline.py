"""Unit tests for the video perception and game state reconstruction vision pipeline."""

import unittest
import numpy as np
import pandas as pd

from src.vision.homography import PitchHomography
from src.vision.jersey_classifier import JerseyClassifier


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


if __name__ == "__main__":
    unittest.main()
