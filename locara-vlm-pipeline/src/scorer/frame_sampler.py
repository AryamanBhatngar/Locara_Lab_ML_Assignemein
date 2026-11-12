"""Frame sampling and OpenCV telemetry (Q1).

One decode pass does three jobs:

1. Sampling. The clip is split into N equal segments (8 for clips up to 10 s, 12 for 30-60 s,
   16 beyond 60 s, linear from 10 to 30 s). A few candidates are decoded per segment and the
   sharpest is kept, so one blurred frame never stands in for a segment with sharp frames.
2. Telemetry on a 160-px-wide copy at 4 fps: optical-flow magnitude and jitter, static-camera
   ratio, brightness and clipping, flicker, a skin-tone hand proxy and clutter. These ground the
   VLM (see prompts.py v3) and drive the offline baseline.
3. Perceptual hashes (64-bit dHash) of the segment-centre frames. File hashes such as SHA-256
   miss a re-encoded copy, these do not.

Sampled frames are resized to a 512 px long side and stamped with their time, so the model can
reason about order and motion between frames.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)


class VideoReadError(RuntimeError):
    """The file is missing, unreadable, or has no decodable frames."""


@dataclass
class SampledFrame:
    index: int  # frame number in the source video
    time_sec: float
    jpeg: bytes  # long side == FRAME_LONG_SIDE, timestamp stamped
    sharpness: float
    hand_sides: tuple[bool, bool]  # skin tone in the left and right of the lower frame


@dataclass
class SampledVideo:
    path: str
    duration_sec: float
    fps: float
    width: int
    height: int
    frames: list[SampledFrame]
    hashes: list[int]
    telemetry: dict[str, float]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def frames_for_duration(duration_sec: float) -> int:
    """Number of frames to send to the VLM for a clip of this length."""
    if duration_sec <= 10:
        return config.MIN_FRAMES
    if duration_sec <= 30:
        frac = (duration_sec - 10) / 20.0
        return round(config.MIN_FRAMES + frac * (config.TARGET_FRAMES - config.MIN_FRAMES))
    if duration_sec <= 60:
        return config.TARGET_FRAMES
    return config.MAX_FRAMES


def _plan(
    total_frames: int, n_segments: int, k: int
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    """Map frame index -> (segment, candidate) and segment -> centre frame index."""
    targets: dict[int, tuple[int, int]] = {}
    centres: dict[int, int] = {}
    seg_len = total_frames / n_segments
    for s in range(n_segments):
        start = s * seg_len
        for c in range(k):
            idx = int(start + (c + 1) * seg_len / (k + 1))
            idx = min(max(idx, 0), total_frames - 1)
            targets.setdefault(idx, (s, c))
        centres[s] = min(int(start + seg_len / 2), total_frames - 1)
    return targets, centres


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash. Survives re-encoding, resizing and small brightness shifts."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return img.copy()
    return cv2.resize(
        img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )


def _stamp(img: np.ndarray, t: float) -> np.ndarray:
    label = f"t={t:05.1f}s"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (tw + 8, th + 8), (0, 0, 0), -1)
    cv2.putText(
        img, label, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


SKIN_LO = np.array([0, 135, 80], dtype=np.uint8)  # YCrCb
SKIN_HI = np.array([255, 175, 128], dtype=np.uint8)


def _skin_ratio_lower(bgr_small: np.ndarray) -> float:
    """Share of skin-tone pixels in the lower 55% of the frame (the ego manipulation zone)."""
    h = bgr_small.shape[0]
    lower = bgr_small[int(h * 0.45) :]
    ycrcb = cv2.cvtColor(lower, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, SKIN_LO, SKIN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(mask.mean() / 255.0)


class _TelemetryStream:
    """Telemetry accumulated frame by frame, so memory stays flat however long the clip is."""

    def __init__(self) -> None:
        self.frames = 0
        self.flow_seconds = 0.0
        self._previous: np.ndarray | None = None
        self._width = 1
        self._flow_mag: list[float] = []
        self._global_dx: list[float] = []
        self._global_dy: list[float] = []
        self._diffs: list[float] = []
        self._brightness: list[float] = []
        self._under: list[float] = []
        self._over: list[float] = []
        self._skin: list[float] = []
        self._edges: list[float] = []

    def add(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            (config.TELEMETRY_WIDTH, max(1, round(h * config.TELEMETRY_WIDTH / w))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        value = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]
        self._brightness.append(float(value.mean()))
        self._under.append(float((value < 30).mean()))
        self._over.append(float((value > 240).mean()))
        self._skin.append(_skin_ratio_lower(small))
        self._edges.append(float((cv2.Canny(gray, 60, 150) > 0).mean()))
        if self._previous is None:
            self._width = gray.shape[1]
        elif self._previous.shape == gray.shape:  # no flow across a mid-clip resolution change
            self._add_motion(self._previous, gray)
        self._previous = gray
        self.frames += 1

    def _add_motion(self, a: np.ndarray, b: np.ndarray) -> None:
        start = time.perf_counter()
        # The stubs do not allow None for the output array, but OpenCV does.
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)  # type: ignore[call-overload]
        self._flow_mag.append(float(np.linalg.norm(flow, axis=2).mean()))
        self._global_dx.append(float(np.median(flow[..., 0])))
        self._global_dy.append(float(np.median(flow[..., 1])))
        self._diffs.append(float(cv2.absdiff(a, b).mean()))
        self.flow_seconds += time.perf_counter() - start

    def summary(self, frames: list[SampledFrame]) -> dict[str, float]:
        w = self._width
        dx, dy, diffs, skin = self._global_dx, self._global_dy, self._diffs, self._skin
        jitter = 0.0
        if len(dx) >= 2:
            jitter = float(np.mean(np.hypot(np.diff(dx), np.diff(dy)))) / w * 100
        sharp = [f.sharpness for f in frames]
        camera = [float(np.hypot(x, y)) for x, y in zip(dx, dy, strict=True)]
        b = np.array(self._brightness) if self._brightness else np.zeros(1)
        hand_frames = np.array(skin) > 0.02 if skin else np.zeros(1, bool)
        return {
            "laplacian_var_median": round(float(np.median(sharp)), 2),
            "laplacian_var_min": round(float(np.min(sharp)), 2),
            "flow_mean_pct": round(float(np.mean(self._flow_mag)) / w * 100, 3)
            if self._flow_mag
            else 0.0,
            "camera_motion_pct": round(float(np.mean(camera)) / w * 100, 3) if camera else 0.0,
            "flow_jitter_pct": round(jitter, 3),
            "static_ratio": round(float(np.mean(np.array(diffs) < 1.0)), 3) if diffs else 1.0,
            "frame_diff_mean": round(float(np.mean(diffs)), 3) if diffs else 0.0,
            "brightness_mean": round(float(b.mean()), 2),
            "under_exposed_ratio": round(float(np.mean(self._under)), 4) if self._under else 0.0,
            "over_exposed_ratio": round(float(np.mean(self._over)), 4) if self._over else 0.0,
            "flicker_cv": round(float(b.std() / max(b.mean(), 1e-6)), 4),
            "skin_ratio_lower": round(float(np.mean(skin)), 4) if skin else 0.0,
            "hand_presence_frac": round(float(np.mean(hand_frames)), 3),
            "edge_density": round(float(np.mean(self._edges)), 4) if self._edges else 0.0,
            "telemetry_frames": float(self.frames),
        }


@dataclass
class _Scan:
    """What one decode pass produced."""

    telemetry: _TelemetryStream
    decoded: int = 0
    last_sec: float = 0.0
    best: dict[int, tuple[float, int, np.ndarray]] = field(default_factory=dict)
    centre_gray: dict[int, np.ndarray] = field(default_factory=dict)


@contextmanager
def _capture(path: str) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoReadError(f"cannot open video: {path}")
        yield cap
    finally:
        cap.release()


def _usable_fps(fps: float) -> bool:
    return math.isfinite(fps) and fps > 0


def _header_frame_count(cap: cv2.VideoCapture) -> int:
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(count) if math.isfinite(count) and count > 0 else 0


def _check_length(decoded: int, path: str) -> None:
    if decoded > config.MAX_DECODED_FRAMES:
        raise VideoReadError(
            f"over {config.MAX_DECODED_FRAMES} frames in {path}: too long to score"
        )


def _check_frame_size(frame: np.ndarray, path: str) -> None:
    pixels = frame.shape[0] * frame.shape[1]
    if pixels > config.MAX_FRAME_PIXELS:
        raise VideoReadError(
            f"{pixels} pixels per frame in {path}: over the {config.MAX_FRAME_PIXELS} limit"
        )


def _count_frames(path: str) -> tuple[int, float]:
    """Decoded frames and the timestamp of the last one, for when the header is no use."""
    decoded, last_sec = 0, 0.0
    with _capture(path) as cap:
        while cap.grab():
            decoded += 1
            _check_length(decoded, path)
            last_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    return decoded, last_sec


def _checked_timing(fps: float, total: int, decoded: int, last_sec: float) -> tuple[float, int]:
    """Prefer what the decoder saw over the container header when they clearly disagree.

    Headers are often wrong: WebM and MKV can report the 1000 Hz timebase as the frame rate, and
    a truncated upload keeps the frame count of the complete file. Small differences are normal,
    so they keep the header values and avoid a second decode.
    """
    measured = (decoded - 1) / last_sec if decoded > 1 and last_sec > 0 else None
    if measured is not None and (
        not _usable_fps(fps) or abs(measured - fps) > config.FPS_TOLERANCE * fps
    ):
        fps = measured
    if abs(decoded - total) > max(2, config.FRAME_COUNT_TOLERANCE * total):
        total = decoded
    return fps, total


def _keep_sample(
    scan: _Scan, idx: int, frame: np.ndarray, segment: int | None, is_centre: bool
) -> None:
    resized = _resize_long_side(frame, config.FRAME_LONG_SIDE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if segment is not None:
        sharpness = laplacian_variance(gray)
        if segment not in scan.best or sharpness > scan.best[segment][0]:
            scan.best[segment] = (sharpness, idx, resized)
    if is_centre:
        scan.centre_gray[idx] = gray


def _scan(path: str, fps: float, total: int) -> _Scan:
    """One decode pass: sharpest frame per segment, centre frames for hashing, telemetry."""
    n_segments = min(frames_for_duration(total / fps), total)
    targets, centres = _plan(total, n_segments, config.CANDIDATES_PER_SEGMENT)
    centre_set = set(centres.values())
    # Keep at least two telemetry frames, or a very short clip has no motion to measure.
    tel_step = min(max(1, round(fps / config.TELEMETRY_FPS)), max(1, total - 1))

    scan = _Scan(_TelemetryStream())
    with _capture(path) as cap:
        while cap.grab():
            idx = scan.decoded
            scan.decoded += 1
            _check_length(scan.decoded, path)
            scan.last_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            is_centre = idx in centre_set
            if idx not in targets and not is_centre and idx % tel_step:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            _check_frame_size(frame, path)
            if idx in targets or is_centre:
                segment = targets[idx][0] if idx in targets else None
                _keep_sample(scan, idx, frame, segment, is_centre)
            if idx % tel_step == 0:
                scan.telemetry.add(frame)
    return scan


def _require_motion_frames(decoded: int, path: str) -> None:
    if decoded < 2:
        raise VideoReadError(f"{decoded} decodable frame(s) in {path}: need at least 2")


def _read_header(path: str) -> tuple[float, int, int, int]:
    with _capture(path) as cap:
        return (
            cap.get(cv2.CAP_PROP_FPS),
            _header_frame_count(cap),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )


def _scan_with_checked_timing(path: str, fps: float, total: int) -> tuple[_Scan, float]:
    """Decode the clip once, or twice if what the decoder saw contradicts the header."""
    if not _usable_fps(fps) or total <= 0:
        decoded, last_sec = _count_frames(path)
        _require_motion_frames(decoded, path)
        fps, total = _checked_timing(fps, decoded, decoded, last_sec)[0], decoded
        if not _usable_fps(fps):
            raise VideoReadError(f"no usable frame rate in {path}")

    scan = _scan(path, fps, total)
    _require_motion_frames(scan.decoded, path)
    checked = _checked_timing(fps, total, scan.decoded, scan.last_sec)
    if checked != (fps, total):
        log.warning(
            "%s: header says %.3f fps, %d frames; decoded %.3f fps, %d frames",
            path,
            fps,
            total,
            *checked,
        )
        fps, total = checked
        scan = _scan(path, fps, total)
    if not scan.best or scan.telemetry.frames < 2:
        raise VideoReadError(f"no decodable frames in {path}")
    return scan, fps


def hand_sides(img: np.ndarray) -> tuple[bool, bool]:
    """Measured on the full-quality frame, because JPEG could change the skin mask."""
    h, w = img.shape[:2]
    lower = cv2.cvtColor(img[int(h * 0.45) :], cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(lower, SKIN_LO, SKIN_HI)
    left, right = mask[:, : w // 2].mean() / 255, mask[:, w // 2 :].mean() / 255
    return bool(left > 0.03), bool(right > 0.03)


def sample_video(path: str | Path) -> SampledVideo:
    """Decode the video and return sampled frames, hashes and telemetry.

    The result crosses a process boundary, so it holds JPEG bytes rather than raw frames: about
    0.15 MB instead of 5 MB for 12 frames.
    """
    t0 = time.perf_counter()
    p = str(path)
    if not Path(p).is_file():
        raise VideoReadError(f"video not found: {p}")
    # FFmpeg also opens playlists, and a .m3u8 can make it read other local files.
    if Path(p).suffix.lower() not in config.VIDEO_EXTS:
        raise VideoReadError(f"unsupported video type: {p}")

    fps, total, width, height = _read_header(p)
    scan, fps = _scan_with_checked_timing(p, fps, total)
    t_decode = time.perf_counter()

    frames = []
    for seg in sorted(scan.best):
        sharp, fidx, img = scan.best[seg]
        t = fidx / fps
        stamped = _stamp(img.copy(), t)
        frames.append(
            SampledFrame(
                fidx, round(t, 3), encode_jpeg(stamped), round(sharp, 2), hand_sides(stamped)
            )
        )
    hashes = [dhash(scan.centre_gray[i]) for i in sorted(scan.centre_gray)]
    telemetry = scan.telemetry.summary(frames)
    t_end = time.perf_counter()
    flow = scan.telemetry.flow_seconds
    return SampledVideo(
        path=p,
        duration_sec=round(scan.decoded / fps, 3),
        fps=round(fps, 3),
        width=width,
        height=height,
        frames=frames,
        hashes=hashes,
        telemetry=telemetry,
        timings_ms={
            "decode_sample": round((t_decode - t0 - flow) * 1000, 1),
            "telemetry": round((t_end - t_decode + flow) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
        },
    )


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def telemetry_summary(t: dict[str, Any]) -> str:
    """Human-readable telemetry block injected into the v3 prompt."""
    return "\n".join(
        [
            (
                f"- sharpness (Laplacian variance, median/min over sampled frames): "
                f"{t['laplacian_var_median']:.0f} / {t['laplacian_var_min']:.0f}  "
                "(below ~60 is usually blurred)"
            ),
            (
                "- camera motion (global optical flow, % of frame width per 0.25 s): "
                f"{t['camera_motion_pct']:.2f}; "
                f"all motion incl. moving objects: {t['flow_mean_pct']:.2f}"
            ),
            (
                "- camera jitter (frame-to-frame change in global motion, %): "
                f"{t['flow_jitter_pct']:.2f}"
            ),
            (
                "- static-camera ratio (share of 0.25 s steps with almost no change): "
                f"{t['static_ratio']:.2f}"
            ),
            (
                f"- brightness (mean V, 0-255): {t['brightness_mean']:.0f}; "
                f"under-exposed pixels {t['under_exposed_ratio']:.0%}; "
                f"over-exposed {t['over_exposed_ratio']:.0%}; "
                f"flicker (CV of brightness) {t['flicker_cv']:.2f}"
            ),
            (
                f"- skin-tone pixels in the lower frame: {t['skin_ratio_lower']:.1%}; "
                f"share of moments with likely hands: {t['hand_presence_frac']:.0%}"
            ),
            f"- edge density (clutter proxy): {t['edge_density']:.3f}",
        ]
    )
