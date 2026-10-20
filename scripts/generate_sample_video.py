"""Generate a broadcast match video clip with realistic player silhouettes for testing and demonstration.

Players are rendered at broadcast scale (~90 px tall at 720p) with anatomical
proportions, an animated walking gait, swinging arms, shading, hair, varied
skin tones and ground shadows. Grass texture, lawn stripes and a slight camera
blur/noise are applied so that COCO-trained detectors (e.g. YOLOv8n) recognise
both teams as ``person``.
"""

import math
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sample_match_clip.mp4"

WIDTH, HEIGHT = 1280, 720
PLAYER_HEIGHT = 92  # pixels, head-top to sole

# BGR palettes
KITS = {
    "home": {"shirt": (38, 38, 200), "shirt_shade": (24, 24, 140), "shorts": (240, 240, 240),
             "shorts_shade": (190, 190, 190), "socks": (38, 38, 200), "number": (255, 255, 255)},
    "away": {"shirt": (190, 90, 20), "shirt_shade": (130, 60, 10), "shorts": (70, 35, 20),
             "shorts_shade": (45, 22, 12), "socks": (240, 240, 240), "number": (255, 255, 255)},
}
SKIN_TONES = [(150, 185, 225), (120, 160, 205), (90, 125, 170), (60, 85, 125), (170, 200, 235)]
HAIR_TONES = [(20, 20, 25), (30, 45, 70), (15, 15, 15), (60, 90, 130)]

LA = cv2.LINE_AA


def _pt(x: float, y: float) -> Tuple[int, int]:
    return int(round(x)), int(round(y))


def _limb(frame: np.ndarray, a: Tuple[float, float], b: Tuple[float, float],
          color: Tuple[int, int, int], thickness: int) -> None:
    """Draw a rounded limb segment."""
    cv2.line(frame, _pt(*a), _pt(*b), color, thickness, LA)
    r = max(1, thickness // 2)
    cv2.circle(frame, _pt(*a), r, color, -1, LA)
    cv2.circle(frame, _pt(*b), r, color, -1, LA)


def draw_humanoid_player(frame: np.ndarray, x: float, y: float, team: str, phase: float,
                         skin: Tuple[int, int, int], hair: Tuple[int, int, int],
                         number: int, facing: int = 1) -> None:
    """Draw a broadcast-scale football player whose feet touch the ground at (x, y).

    Args:
        frame: BGR image to draw on (modified in place).
        x, y: Ground contact point (bottom-centre of the player).
        team: ``"home"`` or ``"away"`` kit.
        phase: Gait phase in radians; drives leg and arm swing.
        skin, hair: BGR colours.
        number: Shirt number rendered on the back.
        facing: +1 moving right, -1 moving left (slight forward lean).
    """
    kit = KITS[team]
    s = PLAYER_HEIGHT / 92.0
    swing = math.sin(phase)
    lift = max(0.0, math.cos(phase))

    # Ground shadow (semi-transparent)
    overlay = frame.copy()
    cv2.ellipse(overlay, _pt(x + 6 * s, y), _pt(17 * s, 5 * s), 0, 0, 360, (20, 70, 22), -1, LA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    lean = 2.0 * s * facing
    hip_y = y - 44 * s
    shoulder_y = y - 70 * s
    hip_l = (x - 5 * s, hip_y)
    hip_r = (x + 5 * s, hip_y)

    # Legs: hip -> knee -> ankle, opposite phase
    def leg(hip: Tuple[float, float], sgn: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        sw = sgn * swing
        knee = (hip[0] + sw * 7 * s * facing, hip[1] + 21 * s)
        ankle = (hip[0] + sw * 11 * s * facing - (lift if sgn > 0 else 1 - lift) * 2 * s,
                 y - 4 * s - (3 * s * max(0.0, -sw)))
        return knee, ankle

    for hip, sgn in ((hip_l, -1.0), (hip_r, 1.0)):
        knee, ankle = leg(hip, sgn)
        _limb(frame, hip, knee, skin, int(8 * s))
        sock_top = (knee[0] + (ankle[0] - knee[0]) * 0.15, knee[1] + (ankle[1] - knee[1]) * 0.15)
        _limb(frame, knee, sock_top, skin, int(7 * s))
        _limb(frame, sock_top, ankle, kit["socks"], int(7 * s))
        boot_tip = (ankle[0] + 6 * s * facing, ankle[1] + 3 * s)
        _limb(frame, ankle, boot_tip, (15, 15, 15), int(5 * s))

    # Shorts
    shorts = np.array([
        _pt(x - 11 * s, hip_y - 6 * s), _pt(x + 11 * s, hip_y - 6 * s),
        _pt(x + 12 * s, hip_y + 9 * s), _pt(x + 1 * s, hip_y + 9 * s),
        _pt(x, hip_y + 4 * s), _pt(x - 1 * s, hip_y + 9 * s), _pt(x - 12 * s, hip_y + 9 * s),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [shorts], kit["shorts"], LA)
    cv2.fillPoly(frame, [shorts[[1, 2, 3, 4]]], kit["shorts_shade"], LA)

    # Arms (behind torso first), swing opposite to legs
    sh_l = (x - 12 * s + lean, shoulder_y + 3 * s)
    sh_r = (x + 12 * s + lean, shoulder_y + 3 * s)
    for sh, sgn in ((sh_l, 1.0), (sh_r, -1.0)):
        sw = sgn * swing
        elbow = (sh[0] + sw * 6 * s * facing + (-2 if sh is sh_l else 2) * s, sh[1] + 14 * s)
        hand = (elbow[0] + sw * 7 * s * facing, elbow[1] + 12 * s)
        sleeve_end = (sh[0] + (elbow[0] - sh[0]) * 0.6, sh[1] + (elbow[1] - sh[1]) * 0.6)
        _limb(frame, sleeve_end, elbow, skin, int(5 * s))
        _limb(frame, elbow, hand, skin, int(5 * s))
        _limb(frame, sh, sleeve_end, kit["shirt"], int(7 * s))

    # Torso (tapered shirt with side shading)
    torso = np.array([
        _pt(x - 13 * s + lean, shoulder_y), _pt(x + 13 * s + lean, shoulder_y),
        _pt(x + 11 * s, hip_y - 4 * s), _pt(x - 11 * s, hip_y - 4 * s),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [torso], kit["shirt"], LA)
    shade = np.array([
        _pt(x + 5 * s + lean, shoulder_y), _pt(x + 13 * s + lean, shoulder_y),
        _pt(x + 11 * s, hip_y - 4 * s), _pt(x + 4 * s, hip_y - 4 * s),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [shade], kit["shirt_shade"], LA)
    cv2.putText(frame, str(number), _pt(x - 6 * s + lean, shoulder_y + 17 * s),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, kit["number"], max(1, int(1.4 * s)), LA)

    # Neck, head and hair
    neck_top = (x + lean * 1.2, shoulder_y - 5 * s)
    _limb(frame, (x + lean, shoulder_y + 1 * s), neck_top, skin, int(6 * s))
    head_c = (x + lean * 1.4, shoulder_y - 12 * s)
    cv2.ellipse(frame, _pt(*head_c), _pt(7 * s, 8.5 * s), 0, 0, 360, skin, -1, LA)
    cv2.ellipse(frame, _pt(head_c[0], head_c[1] - 2 * s), _pt(7.4 * s, 7 * s), 0, 180, 360, hair, -1, LA)
    cv2.ellipse(frame, _pt(head_c[0] - 4 * s * facing, head_c[1]), _pt(3 * s, 5 * s), 0, 0, 360, hair, -1, LA)


def _build_pitch(rng: np.random.Generator) -> np.ndarray:
    """Static textured pitch background with lawn stripes and markings."""
    base = np.full((HEIGHT, WIDTH, 3), (38, 140, 42), dtype=np.uint8)
    stripe_w = 90
    for sx in range(0, WIDTH, stripe_w * 2):
        base[:, sx:sx + stripe_w] = (44, 155, 48)
    noise = rng.normal(0, 6, (HEIGHT, WIDTH, 1)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)

    white = (235, 240, 240)
    cv2.rectangle(base, (120, 90), (1160, 630), white, 3, LA)
    cv2.line(base, (640, 90), (640, 630), white, 3, LA)
    cv2.circle(base, (640, 360), 85, white, 3, LA)
    cv2.circle(base, (640, 360), 4, white, -1, LA)
    cv2.rectangle(base, (120, 210), (320, 510), white, 2, LA)
    cv2.rectangle(base, (960, 210), (1160, 510), white, 2, LA)
    cv2.rectangle(base, (120, 290), (180, 430), white, 2, LA)
    cv2.rectangle(base, (1100, 290), (1160, 430), white, 2, LA)
    return base


def _squad(rng: np.random.Generator) -> List[Dict]:
    # (x, y) are ground contact points; velocities in px/frame
    home = [
        (350, 400, 1.2, 0.0), (480, 285, 1.4, 0.1), (510, 400, 1.3, -0.1), (470, 520, 1.4, -0.1),
        (280, 320, 0.8, 0.1), (270, 480, 0.8, -0.1), (180, 240, 0.5, 0.0), (160, 360, 0.5, 0.0),
        (160, 440, 0.5, 0.0), (180, 560, 0.5, 0.0),
    ]
    away = [
        (620, 380, -0.8, 0.2), (680, 290, -0.5, 0.1), (690, 510, -0.5, -0.1), (820, 250, -0.4, 0.0),
        (800, 350, -0.4, 0.1), (800, 450, -0.4, -0.1), (830, 550, -0.4, 0.0), (580, 420, -0.9, 0.0),
    ]
    squad = []
    for team, rows, first_id in (("home", home, 1), ("away", away, 11)):
        numbers = rng.choice(np.arange(2, 24), size=len(rows), replace=False)
        for i, (x, y, vx, vy) in enumerate(rows):
            squad.append({
                "id": first_id + i, "team": team, "x": x, "y": y, "vx": vx, "vy": vy,
                "phase0": float(rng.uniform(0, 2 * math.pi)),
                "skin": SKIN_TONES[int(rng.integers(len(SKIN_TONES)))],
                "hair": HAIR_TONES[int(rng.integers(len(HAIR_TONES)))],
                "number": int(numbers[i]),
            })
    return squad


def iter_frames(num_frames: int = 150, seed: int = 7) -> Iterator[np.ndarray]:
    """Yield rendered BGR frames of the sample match."""
    rng = np.random.default_rng(seed)
    pitch = _build_pitch(rng)
    squad = _squad(rng)

    for f in range(num_frames):
        frame = pitch.copy()
        positions = []
        for p in squad:
            cx = p["x"] + p["vx"] * f
            cy = p["y"] + p["vy"] * f
            speed = math.hypot(p["vx"], p["vy"])
            phase = p["phase0"] + f * (0.18 + 0.12 * speed)
            positions.append((cy, cx, phase, p))
        # Painter's order: players further up the screen are drawn first
        for cy, cx, phase, p in sorted(positions, key=lambda t: t[0]):
            draw_humanoid_player(frame, cx, cy, p["team"], phase, p["skin"], p["hair"],
                                 p["number"], facing=1 if p["vx"] >= 0 else -1)

        # Mild broadcast camera softness and sensor noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
        noise = rng.normal(0, 2.5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yield frame


def generate_sample_clip(num_frames: int = 150, fps: float = 25.0) -> str:
    """Generate a 1280x720 video clip showing 2 teams moving tactically on a pitch."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, fps, (WIDTH, HEIGHT))
    for frame in iter_frames(num_frames):
        out.write(frame)
    out.release()
    print(f"Generated realistic sample match video ({num_frames} frames): {OUTPUT_FILE}")
    return str(OUTPUT_FILE)


if __name__ == "__main__":
    generate_sample_clip(num_frames=150)
