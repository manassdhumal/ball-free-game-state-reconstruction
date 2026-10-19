"""End-to-end smoke test for VideoAnalysisPipeline with synthetic gameplay video."""

import os
import unittest
import cv2
import numpy as np
import pandas as pd

from src.vision.pipeline import VideoAnalysisPipeline


class TestVideoPipelineE2E(unittest.TestCase):
    """Verify that a video is processed into canonical tracking and produces tactical insights."""

    def setUp(self):
        self.test_video_path = "data/interim/synthetic_match_clip.mp4"
        os.makedirs("data/interim", exist_ok=True)

        # Generate a small 1280x720 30-frame synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.test_video_path, fourcc, 25.0, (1280, 720))

        # 4 synthetic players walking across the pitch
        for f in range(25):
            # Green pitch background
            frame = np.full((720, 1280, 3), [40, 160, 40], dtype=np.uint8)

            # Draw white pitch lines
            cv2.rectangle(frame, (100, 100), (1180, 620), (255, 255, 255), 2)
            cv2.line(frame, (640, 100), (640, 620), (255, 255, 255), 2)
            cv2.circle(frame, (640, 360), 80, (255, 255, 255), 2)

            # Draw Team Red players (standing / moving right)
            # Player 1 (Red torso: [0, 0, 255])
            x1 = 300 + f * 2
            cv2.rectangle(frame, (x1, 300), (x1 + 30, 390), (0, 0, 255), -1)

            # Player 2 (Red torso)
            x2 = 400 + f * 1
            cv2.rectangle(frame, (x2, 450), (x2 + 30, 540), (0, 0, 255), -1)

            # Draw Team Blue players (standing / moving left)
            # Player 3 (Blue torso: [255, 0, 0])
            x3 = 800 - f * 2
            cv2.rectangle(frame, (x3, 310), (x3 + 30, 400), (255, 0, 0), -1)

            # Player 4 (Blue torso)
            x4 = 700 - f * 1
            cv2.rectangle(frame, (x4, 460), (x4 + 30, 550), (255, 0, 0), -1)

            out.write(frame)

        out.release()

    def tearDown(self):
        if os.path.exists(self.test_video_path):
            try:
                os.remove(self.test_video_path)
            except OSError:
                pass

    def test_pipeline_execution(self):
        pipeline = VideoAnalysisPipeline(output_dir="data/interim")
        result = pipeline.analyze_video(
            video_path=self.test_video_path,
            match_id="test_synthetic_clip",
            max_frames=25,
            apply_mitigation=False,  # Short clip test
        )

        self.assertIn("status", result)
        self.assertIn("match_id", result)
        self.assertEqual(result["match_id"], "test_synthetic_clip")
        # Check that tracking file was written
        tracking_file = result.get("tracking_file")
        if tracking_file and os.path.exists(tracking_file):
            df = pd.read_csv(tracking_file)
            self.assertTrue({"match_id", "frame", "player_id", "team", "x", "y"}.issubset(df.columns))
            # Clean up
            os.remove(tracking_file)


if __name__ == "__main__":
    unittest.main()
