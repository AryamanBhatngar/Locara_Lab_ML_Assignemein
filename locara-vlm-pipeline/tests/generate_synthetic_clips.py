"""Generate 10 synthetic ego-centric test clips with known ground truth.

    python tests/generate_synthetic_clips.py --out data/synthetic

What these clips are for: exercising the pipeline end to end (sampling, telemetry, dedup,
routing, pre-annotation, evaluation) with controlled defects. They are rendered shapes, not
camera footage, so they cannot measure how well a VLM judges real video. Use real phone
recordings for that (README, "Evaluation").

Each clip is rendered from a large "world" image (a table with objects, walls above) viewed
through a moving crop window that imitates head motion. Hands are skin-tone forearms entering
from the bottom edge. Defects are applied per clip: motion blur and whip pans, low exposure,
a fixed third-person camera, looking away from the work, a re-encoded duplicate, hands leaving
the frame, and moderate jitter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 360, 15
SKIN = (120, 160, 215)  # BGR skin tone inside the YCrCb skin range


@dataclass
class ClipSpec:
    name: str
    duration: float
    truth: dict[str, float]
    tier: str
    seed: int = 0
    hands: int = 2  # number of visible forearms
    hand_duty: float = 1.0  # share of time hands are in frame
    bob: float = 6.0  # head-bob amplitude, px
    pan_speed: float = 0.6  # slow look-around, px per frame
    jitter: float = 0.0  # random shake, px
    blur: int = 0  # directional blur kernel length
    gain: float = 1.0  # exposure multiplier
    static_third_person: bool = False
    look_up: bool = False  # camera pointed at walls/ceiling
    walking: bool = False
    duplicate_of: str | None = None
    notes: str = ""


def _truth(ego, motion, scene, hand, light, dup, overall):
    return {
        "egocentric_framing": ego,
        "motion_quality": motion,
        "scene_diversity": scene,
        "hand_visibility": hand,
        "lighting_exposure": light,
        "duplicate_risk": dup,
        "overall_usability": overall,
    }


SPECS: list[ClipSpec] = [
    ClipSpec(
        "clip_01_kitchen_pour_ideal",
        20,
        _truth(1, 5, 4, 5, 5, 0, 5),
        "usable",
        seed=1,
        notes="Head-mounted view, two hands working, bright and stable.",
    ),
    ClipSpec(
        "clip_02_workshop_assembly_ideal",
        45,
        _truth(1, 5, 5, 5, 5, 0, 5),
        "usable",
        seed=2,
        notes="Busy workbench, two hands, stable.",
    ),
    ClipSpec(
        "clip_03_office_typing_usable",
        10,
        _truth(1, 4, 3, 4, 4, 0, 4),
        "usable",
        seed=3,
        bob=4,
        gain=0.85,
        notes="Short desk clip, good exposure.",
    ),
    ClipSpec(
        "clip_04_severe_motion_blur",
        15,
        _truth(1, 1, 3, 3, 4, 0, 1),
        "reject",
        seed=4,
        jitter=28,
        blur=31,
        bob=18,
        notes="Whip pans and directional blur in most frames.",
    ),
    ClipSpec(
        "clip_05_underexposed_dark",
        12,
        _truth(1, 3, 2, 2, 1, 0, 1),
        "reject",
        seed=5,
        gain=0.12,
        notes="Nearly black frames.",
    ),
    ClipSpec(
        "clip_06_static_tripod_non_ego",
        30,
        _truth(0, 4, 3, 1, 5, 0, 1),
        "reject",
        seed=6,
        hands=0,
        bob=0,
        pan_speed=0,
        static_third_person=True,
        notes="Fixed camera watching a person from across the room.",
    ),
    ClipSpec(
        "clip_07_no_hands_visible",
        25,
        _truth(1, 3, 1, 1, 4, 0, 1),
        "reject",
        seed=7,
        hands=0,
        look_up=True,
        walking=True,
        bob=10,
        notes="Walking, looking at walls.",
    ),
    ClipSpec(
        "clip_08_duplicate_of_clip_01",
        20,
        _truth(1, 5, 4, 5, 5, 1, 1),
        "reject",
        seed=1,
        duplicate_of="clip_01_kitchen_pour_ideal",
        notes="Re-encoded, rescaled copy of clip 01.",
    ),
    ClipSpec(
        "clip_09_borderline_partial_occlusion",
        60,
        _truth(1, 4, 4, 3, 3, 0, 3),
        "borderline",
        seed=9,
        hand_duty=0.5,
        gain=0.6,
        notes="Hands leave the frame half the time; dim.",
    ),
    ClipSpec(
        "clip_10_borderline_mild_shake",
        40,
        _truth(1, 3, 4, 4, 4, 0, 3),
        "borderline",
        seed=10,
        hands=1,
        jitter=7,
        bob=9,
        notes="One hand, moderate jitter.",
    ),
]


def _world(seed: int, look_up: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wh, ww = 900, 1400
    img = np.zeros((wh, ww, 3), np.uint8)
    g = int(rng.integers(150, 195))
    img[:] = (g + 12, g + 4, g - 6)  # neutral, slightly cool: never in the skin range
    if not look_up:
        t = int(rng.integers(70, 115))
        cv2.rectangle(img, (0, 470), (ww, wh), (t + 15, t + 8, t), -1)  # table
        for _ in range(int(rng.integers(18, 28))):  # objects
            x, y = int(rng.integers(40, ww - 120)), int(rng.integers(500, wh - 80))
            w, h = int(rng.integers(30, 110)), int(rng.integers(30, 90))
            color = [
                int(c)
                for c in rng.choice(
                    [
                        (40, 40, 200),
                        (200, 60, 40),
                        (40, 160, 60),
                        (30, 200, 230),
                        (180, 180, 180),
                        (90, 60, 30),
                    ]
                )
            ]
            if rng.random() < 0.5:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), 2)
            else:
                cv2.ellipse(img, (x, y), (w // 2, h // 2), 0, 0, 360, color, -1)
        for _ in range(8):  # shelves
            y = int(rng.integers(60, 420))
            cv2.line(
                img,
                (int(rng.integers(0, 600)), y),
                (int(rng.integers(800, ww)), y),
                (60, 60, 60),
                4,
            )
    else:
        for y in range(0, wh, 180):  # plain walls
            cv2.line(img, (0, y), (ww, y), (175, 175, 175), 1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _forearm(frame: np.ndarray, base_x: float, tip: tuple[int, int], width: int = 46) -> None:
    pts = np.array(
        [
            [base_x - width, H + 30],
            [base_x + width, H + 30],
            [tip[0] + width // 2, tip[1]],
            [tip[0] - width // 2, tip[1]],
        ],
        np.int32,
    )
    cv2.fillConvexPoly(frame, pts, SKIN)
    cv2.circle(frame, tip, width // 2 + 6, SKIN, -1)


def render(spec: ClipSpec, path: Path) -> None:
    rng = np.random.default_rng(spec.seed + 100)
    world = _world(spec.seed, spec.look_up)
    n = int(spec.duration * FPS)
    tmp = path.with_suffix(".raw.avi")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    cx, cy = 300.0, 60.0 if spec.look_up else 330.0
    for i in range(n):
        t = i / FPS
        if spec.static_third_person:
            x, y = 380, 250
        else:
            cx += spec.pan_speed * np.sin(t / 3.0)
            step = 2.5 if spec.walking else 0
            x = cx + 30 * np.sin(t / 5.0) + rng.normal(0, spec.jitter)
            y = (
                cy
                + spec.bob * np.sin(2 * np.pi * (1.6 if spec.walking else 0.6) * t)
                + rng.normal(0, spec.jitter)
                + step * np.sin(t)
            )
            if spec.blur:  # occasional whip pan
                x += 120 * np.sin(t * 2.3)
        x = int(np.clip(x, 0, world.shape[1] - W))
        y = int(np.clip(y, 0, world.shape[0] - H))
        frame = world[y : y + H, x : x + W].copy()
        if spec.static_third_person:  # a person seen from outside
            px = int(320 + 80 * np.sin(t / 2))
            cv2.circle(frame, (px, 95), 22, SKIN, -1)
            cv2.rectangle(frame, (px - 26, 118), (px + 26, 215), (90, 50, 50), -1)
        visible = spec.hands > 0 and ((t % 8.0) / 8.0) < spec.hand_duty
        if visible:
            reach = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
            _forearm(frame, 190, (int(230 + 40 * reach), int(250 - 70 * reach)))
            if spec.hands > 1:
                _forearm(frame, 460, (int(420 - 40 * reach), int(245 - 60 * reach)))
        if spec.blur:
            k = np.zeros((spec.blur, spec.blur), np.float32)
            k[spec.blur // 2, :] = 1.0 / spec.blur
            frame = cv2.filter2D(frame, -1, k)
        if spec.gain != 1.0:
            frame = np.clip(
                frame.astype(np.float32) * spec.gain + rng.normal(0, 3, frame.shape), 0, 255
            )
            frame = frame.astype(np.uint8)
        writer.write(frame)
    writer.release()
    _to_h264(tmp, path)


def _reencode_with_opencv(src: Path, dst: Path) -> None:
    cap, out = cv2.VideoCapture(str(src)), None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        out.write(frame)
    cap.release()
    if out:
        out.release()


def _to_h264(src: Path, dst: Path, extra: list[str] | None = None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            *(extra or []),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
    else:
        _reencode_with_opencv(src, dst)
    if src.name.endswith(".raw.avi"):  # left behind, the scorer would count it as another clip
        src.unlink(missing_ok=True)


def make_duplicate(src: Path, dst: Path) -> None:
    """Re-encode at lower quality and a slightly different size, with a small brightness shift."""
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                "scale=624:352,eq=brightness=0.03",
                "-c:v",
                "libx264",
                "-crf",
                "34",
                "-pix_fmt",
                "yuv420p",
                str(dst),
            ],
            check=True,
        )
    else:
        _to_h264(src, dst)


def generate(out_dir: Path, specs: list[ClipSpec] = SPECS) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    for spec in specs:
        path = out_dir / f"{spec.name}.mp4"
        if spec.duplicate_of:
            make_duplicate(out_dir / f"{spec.duplicate_of}.mp4", path)
        else:
            render(spec, path)
        truth[path.name] = {
            "tier": spec.tier,
            "scores": spec.truth,
            "duration_sec": spec.duration,
            "notes": spec.notes,
        }
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/synthetic", type=Path)
    args = parser.parse_args()
    try:
        generate(args.out)
    except OSError as exc:
        sys.exit(f"error: cannot write to {args.out}: {exc}")


if __name__ == "__main__":
    main()
