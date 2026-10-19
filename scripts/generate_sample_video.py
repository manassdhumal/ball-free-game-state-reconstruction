"""Generate a realistic synthetic broadcast match video clip for testing and demonstration."""

import os
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sample_match_clip.mp4"


def generate_sample_clip(num_frames: int = 150, fps: float = 25.0) -> str:
    """Generate a 1280x720 video clip showing 2 teams moving tactically on a pitch."""
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, fps, (width, height))

    # Base formation positions (in screen pixels)
    # Team Red (attacking left to right)
    home_players = [
        {"id": 1, "x": 300, "y": 360, "vx": 1.2, "vy": 0.0},   # Center Mid (Ball carrier)
        {"id": 2, "x": 420, "y": 240, "vx": 1.5, "vy": 0.2},   # Left Winger
        {"id": 3, "x": 440, "y": 360, "vx": 1.4, "vy": -0.1},  # Striker
        {"id": 4, "x": 410, "y": 480, "vx": 1.5, "vy": -0.2},  # Right Winger
        {"id": 5, "x": 240, "y": 280, "vx": 0.9, "vy": 0.1},   # Left Mid
        {"id": 6, "x": 230, "y": 440, "vx": 0.9, "vy": -0.1},  # Right Mid
        {"id": 7, "x": 160, "y": 200, "vx": 0.6, "vy": 0.0},   # Left Back
        {"id": 8, "x": 140, "y": 320, "vx": 0.5, "vy": 0.0},   # Center Back 1
        {"id": 9, "x": 140, "y": 400, "vx": 0.5, "vy": 0.0},   # Center Back 2
        {"id": 10, "x": 160, "y": 520, "vx": 0.6, "vy": 0.0},  # Right Back
    ]

    # Team Blue (defending right to left)
    away_players = [
        {"id": 11, "x": 580, "y": 340, "vx": -0.8, "vy": 0.2},  # Opposing Center Mid (pressing)
        {"id": 12, "x": 620, "y": 260, "vx": -0.5, "vy": 0.1},  # Opposing Mid
        {"id": 13, "x": 630, "y": 460, "vx": -0.6, "vy": -0.1}, # Opposing Mid
        {"id": 14, "x": 750, "y": 220, "vx": -0.4, "vy": 0.0},  # Right Back
        {"id": 15, "x": 730, "y": 320, "vx": -0.4, "vy": 0.1},  # Center Back 1
        {"id": 16, "x": 730, "y": 400, "vx": -0.4, "vy": -0.1}, # Center Back 2
        {"id": 17, "x": 760, "y": 500, "vx": -0.4, "vy": 0.0},  # Left Back
        {"id": 18, "x": 520, "y": 370, "vx": -0.9, "vy": 0.0},  # Pressing forward
    ]

    for f in range(num_frames):
        # Create grass turf with alternating lawn stripes
        frame = np.full((height, width, 3), (38, 140, 42), dtype=np.uint8)
        stripe_w = 90
        for s in range(0, width, stripe_w * 2):
            frame[:, s:s + stripe_w] = (44, 155, 48)

        # Draw pitch white lines
        # Sidelines
        cv2.rectangle(frame, (120, 90), (1160, 630), (255, 255, 255), 3)
        # Halfway line
        cv2.line(frame, (640, 90), (640, 630), (255, 255, 255), 3)
        # Center circle
        cv2.circle(frame, (640, 360), 85, (255, 255, 255), 3)
        # Center spot
        cv2.circle(frame, (640, 360), 4, (255, 255, 255), -1)

        # Left penalty box
        cv2.rectangle(frame, (120, 210), (320, 510), (255, 255, 255), 2)
        # Right penalty box
        cv2.rectangle(frame, (960, 210), (1160, 510), (255, 255, 255), 2)

        # Render Home players (Red jerseys, white shorts)
        for p in home_players:
            cur_x = int(p["x"] + p["vx"] * f)
            cur_y = int(p["y"] + p["vy"] * f)
            # Legs
            cv2.line(frame, (cur_x - 4, cur_y + 15), (cur_x - 4, cur_y + 35), (255, 255, 255), 3)
            cv2.line(frame, (cur_x + 4, cur_y + 15), (cur_x + 4, cur_y + 35), (255, 255, 255), 3)
            # Torso (Red BGR: [20, 25, 220])
            cv2.rectangle(frame, (cur_x - 9, cur_y - 12), (cur_x + 9, cur_y + 15), (20, 25, 220), -1)
            # Head
            cv2.circle(frame, (cur_x, cur_y - 20), 7, (180, 210, 240), -1)

        # Render Away players (Blue jerseys, white shorts)
        for p in away_players:
            cur_x = int(p["x"] + p["vx"] * f)
            cur_y = int(p["y"] + p["vy"] * f)
            # Legs
            cv2.line(frame, (cur_x - 4, cur_y + 15), (cur_x - 4, cur_y + 35), (255, 255, 255), 3)
            cv2.line(frame, (cur_x + 4, cur_y + 15), (cur_x + 4, cur_y + 35), (255, 255, 255), 3)
            # Torso (Blue BGR: [220, 60, 25])
            cv2.rectangle(frame, (cur_x - 9, cur_y - 12), (cur_x + 9, cur_y + 15), (220, 60, 25), -1)
            # Head
            cv2.circle(frame, (cur_x, cur_y - 20), 7, (180, 210, 240), -1)

        out.write(frame)

    out.release()
    print(f"Generated sample match video ({num_frames} frames): {OUTPUT_FILE}")
    return str(OUTPUT_FILE)


if __name__ == "__main__":
    generate_sample_clip(num_frames=150)
