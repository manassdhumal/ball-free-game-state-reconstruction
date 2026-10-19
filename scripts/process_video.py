"""CLI script to process football gameplay video and export annotated dual-view video."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.pipeline import VideoAnalysisPipeline
from src.vision.homography import PitchHomography


def draw_pitch_radar(
    players_in_frame: pd.DataFrame,
    possessor_id: str | None,
    possessing_team: str | None,
    tas_score: float,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Draw a 2D green pitch radar with player dots, team colors, and tactical scores."""
    # Dark grass green
    radar = np.full((height, width, 3), (35, 120, 35), dtype=np.uint8)

    # Margins
    margin = 30
    p_w = width - 2 * margin
    p_h = height - 2 * margin

    # Draw white pitch boundary & center line
    cv2.rectangle(radar, (margin, margin), (width - margin, height - margin), (240, 240, 240), 2)
    mid_x = margin + p_w // 2
    cv2.line(radar, (mid_x, margin), (mid_x, height - margin), (240, 240, 240), 2)
    cv2.circle(radar, (mid_x, margin + p_h // 2), 40, (240, 240, 240), 2)

    # Draw penalty boxes
    box_w = int(0.16 * p_w)
    box_h = int(0.40 * p_h)
    box_y1 = margin + (p_h - box_h) // 2
    box_y2 = box_y1 + box_h
    # Left box
    cv2.rectangle(radar, (margin, box_y1), (margin + box_w, box_y2), (240, 240, 240), 1)
    # Right box
    cv2.rectangle(radar, (width - margin - box_w, box_y1), (width - margin, box_y2), (240, 240, 240), 1)

    # Draw players
    for _, row in players_in_frame.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        px = int(margin + row["x"] * p_w)
        py = int(margin + row["y"] * p_h)

        is_home = (row["team"] == "home")
        color = (30, 30, 220) if is_home else (220, 100, 30)  # Red for Home, Blue for Away
        p_id = str(row["player_id"])

        # Highlight possessor
        if possessor_id and p_id == possessor_id:
            cv2.circle(radar, (px, py), 12, (0, 220, 255), 2)  # Gold aura

        cv2.circle(radar, (px, py), 7, color, -1)
        cv2.circle(radar, (px, py), 7, (255, 255, 255), 1)

        # Draw short ID
        short_id = p_id.split("_")[-1]
        cv2.putText(radar, short_id, (px - 5, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Overlay tactical HUD at top
    hud_text = f"Possession: {possessing_team.upper() if possessing_team else 'NEUTRAL'} | TAS: {tas_score:.3f}"
    cv2.putText(radar, hud_text, (margin, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return radar


def export_annotated_video(
    video_path: str,
    tracking_csv: str,
    insights: list[dict],
    output_video_path: str,
    max_frames: int | None = None,
) -> None:
    """Create a side-by-side composite video with video detections on left and pitch radar on right."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    df = pd.read_csv(tracking_csv)
    insight_dict = {item["frame"]: item for item in insights}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize source video to fit composite (640x480 each -> 1280x480 total)
    target_w, target_h = 640, 480
    composite_w = target_w * 2
    composite_h = target_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (composite_w, composite_h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        # Resize video frame
        resized_frame = cv2.resize(frame, (target_w, target_h))

        # Get tracking and insights for current frame
        frame_tracking = df[df["frame"] == frame_idx]
        cur_insight = insight_dict.get(frame_idx, {})
        poss_id = cur_insight.get("possessor_player_id")
        poss_team = cur_insight.get("possessing_team")
        tas = cur_insight.get("tas", 0.5)

        # Overlay text on video
        cv2.putText(
            resized_frame,
            f"Frame: {frame_idx} (Ball-Free Perception)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Generate 2D pitch radar
        radar_frame = draw_pitch_radar(
            players_in_frame=frame_tracking,
            possessor_id=poss_id,
            possessing_team=poss_team,
            tas_score=tas,
            width=target_w,
            height=target_h,
        )

        # Concatenate horizontally
        composite = np.hstack([resized_frame, radar_frame])
        out.write(composite)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated composite video saved: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Process football video into canonical tracking and tactical insights.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--match-id", type=str, default="cli_processed_match", help="Match identifier")
    parser.add_argument("--preset", type=str, default="main_broadcast", choices=["main_broadcast", "tactical_wide", "normalized"])
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames to process")
    parser.add_argument("--output-dir", type=str, default="data/interim", help="Directory for output files")
    parser.add_argument("--export-video", action="store_true", help="Generate side-by-side composite MP4 video")

    args = parser.parse_args()

    pipeline = VideoAnalysisPipeline(output_dir=args.output_dir, homography_preset=args.preset)

    def on_progress(pct, msg):
        sys.stdout.write(f"\r[{int(pct * 100):3d}%] {msg}")
        sys.stdout.flush()

    print(f"\nProcessing video: {args.video} (Preset: {args.preset})")
    result = pipeline.analyze_video(
        video_path=args.video,
        match_id=args.match_id,
        max_frames=args.max_frames,
        progress_callback=on_progress,
    )
    print("\n")

    if result.get("status") == "success":
        print("Reconstruction Successful!")
        print(f"  • Frames processed: {result['frame_count']}")
        print(f"  • Player observations: {result['tracking_rows']}")
        print(f"  • Possession: Home {result['possession_share']['home']}% | Away {result['possession_share']['away']}%")
        print(f"  • Average Tactical Advantage Score: {result['average_tas']:.4f}")
        print(f"  • Canonical CSV: {result['tracking_file']}")

        if args.export_video:
            out_vid = os.path.join(args.output_dir, f"{args.match_id}_annotated_dual_view.mp4")
            export_annotated_video(
                video_path=args.video,
                tracking_csv=result["tracking_file"],
                insights=result.get("sample_insights", []),
                output_video_path=out_vid,
                max_frames=args.max_frames,
            )
    else:
        print(f"Failed: {result.get('message')}")


if __name__ == "__main__":
    main()
