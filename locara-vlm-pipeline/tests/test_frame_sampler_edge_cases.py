"""Frame sampling on clips that are very short, damaged or carry a wrong header."""

import json
import math
import tracemalloc

import cv2
import numpy as np
import pytest

from src.scorer import frame_sampler
from src.scorer.frame_sampler import VideoReadError, frames_for_duration, sample_video

REAL_CAPTURE = cv2.VideoCapture


def textured(step: int, height: int = 360, width: int = 640) -> np.ndarray:
    """A fixed random pattern, shifted sideways so consecutive frames differ."""
    blocks = np.random.default_rng(7).integers(
        0, 255, (height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8
    )
    frame = cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.roll(frame, 3 * step, axis=1)


def write_clip(path, frames, fps=15.0, fourcc="mp4v"):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


class WrongHeader:
    """A real decoder behind a container header that reports the wrong values."""

    opened = 0

    def __init__(self, path, overrides):
        type(self).opened += 1
        self._cap = REAL_CAPTURE(path)
        self._overrides = overrides

    def get(self, prop):
        return self._overrides.get(prop, self._cap.get(prop))

    def __getattr__(self, name):
        return getattr(self._cap, name)


@pytest.fixture
def wrong_header(monkeypatch):
    def install(overrides):
        WrongHeader.opened = 0
        monkeypatch.setattr(
            frame_sampler.cv2, "VideoCapture", lambda path: WrongHeader(path, overrides)
        )

    return install


@pytest.mark.parametrize("fps", [0.0, -1.0, math.nan, math.inf, 1.0, 1000.0, 90000.0])
def test_wrong_fps_header_is_corrected_from_timestamps(wrong_header, clips, fps):
    honest = sample_video(clips["ideal10"])
    wrong_header({cv2.CAP_PROP_FPS: fps})
    sampled = sample_video(clips["ideal10"])
    assert sampled.fps == pytest.approx(honest.fps)
    assert sampled.duration_sec == pytest.approx(honest.duration_sec)
    assert sampled.frame_count == honest.frame_count
    assert sampled.telemetry == honest.telemetry


@pytest.mark.parametrize("fps", [0.0, math.nan, math.inf])
def test_no_usable_fps_and_no_timestamps_is_a_read_error(wrong_header, clips, fps):
    wrong_header({cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_POS_MSEC: 0.0})
    with pytest.raises(VideoReadError, match="frame rate"):
        sample_video(clips["ideal10"])


@pytest.mark.parametrize("count", [0.0, -1.0, math.nan, math.inf, 37.0, 600.0])
def test_wrong_frame_count_header_is_corrected_by_decoding(wrong_header, clips, count):
    honest = sample_video(clips["ideal10"])
    wrong_header({cv2.CAP_PROP_FRAME_COUNT: count})
    sampled = sample_video(clips["ideal10"])
    assert sampled.frame_count == honest.frame_count
    assert sampled.duration_sec == pytest.approx(honest.duration_sec)
    assert sampled.frames[-1].time_sec > 0.8 * sampled.duration_sec


def test_small_frame_count_error_does_not_cost_a_second_decode(wrong_header, clips):
    wrong_header({cv2.CAP_PROP_FRAME_COUNT: 151.0})
    sampled = sample_video(clips["ideal10"])
    assert sampled.duration_sec == pytest.approx(10.0)
    assert WrongHeader.opened <= 2  # one to read the header, one to decode


def test_truncated_file_reports_what_is_really_there(tmp_path):
    full = write_clip(tmp_path / "full.avi", [textured(i) for i in range(300)], fourcc="MJPG")
    cut = tmp_path / "cut.avi"
    data = full.read_bytes()
    cut.write_bytes(data[: len(data) * 2 // 5])

    capture = REAL_CAPTURE(str(cut))
    decoded = 0
    while capture.grab():
        decoded += 1
    capture.release()
    assert 0 < decoded < 300

    sampled = sample_video(cut)
    assert sampled.duration_sec == pytest.approx(decoded / 15, abs=0.1)
    assert sampled.frame_count == frames_for_duration(decoded / 15)
    assert sampled.frames[-1].time_sec > 0.8 * sampled.duration_sec


def test_single_frame_is_a_read_error(tmp_path):
    with pytest.raises(VideoReadError, match="decodable"):
        sample_video(write_clip(tmp_path / "one.mp4", [textured(0)]))


def test_file_with_no_frames_is_a_read_error(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoReadError):
        sample_video(empty)


@pytest.mark.parametrize("n_frames", [2, 3, 5])
def test_very_short_clip_still_measures_motion(tmp_path, n_frames):
    path = write_clip(tmp_path / "short.mp4", [textured(i * 4) for i in range(n_frames)], fps=30.0)
    telemetry = sample_video(path).telemetry
    assert telemetry["telemetry_frames"] >= 2
    assert telemetry["static_ratio"] < 1.0


@pytest.mark.parametrize("value", [0, 255])
def test_blank_clip_telemetry_is_finite(tmp_path, value):
    frames = [np.full((360, 640, 3), value, np.uint8)] * 30
    telemetry = sample_video(write_clip(tmp_path / "blank.mp4", frames)).telemetry
    assert all(math.isfinite(v) for v in telemetry.values())
    json.dumps(telemetry, allow_nan=False)


def test_portrait_clip_keeps_its_aspect(tmp_path):
    frames = [textured(i, 1280, 720) for i in range(45)]
    sampled = sample_video(write_clip(tmp_path / "portrait.mp4", frames))
    assert (sampled.width, sampled.height) == (720, 1280)
    image = cv2.imdecode(np.frombuffer(sampled.frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (512, 288)


class Scripted:
    """A capture that plays a fixed list of frames."""

    def __init__(self, frames, fps=15.0):
        self._frames, self._fps, self._next = frames, fps, -1

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._next / self._fps * 1000
        return float(self._frames[0].shape[1 if prop == cv2.CAP_PROP_FRAME_WIDTH else 0])

    def grab(self):
        self._next += 1
        return self._next < len(self._frames)

    def retrieve(self):
        return True, self._frames[self._next]

    def release(self):
        pass


def test_resolution_change_mid_clip_does_not_fail(monkeypatch, tmp_path):
    frames = [textured(i) for i in range(30)] + [textured(i, 480, 640) for i in range(30)]
    monkeypatch.setattr(frame_sampler.cv2, "VideoCapture", lambda path: Scripted(frames))
    path = tmp_path / "resized.mp4"
    path.write_bytes(b"placeholder")
    assert all(math.isfinite(v) for v in sample_video(path).telemetry.values())


def test_memory_does_not_grow_with_clip_length(monkeypatch, clips):
    monkeypatch.setattr(frame_sampler, "frames_for_duration", lambda duration: 8)

    def peak_bytes(path):
        tracemalloc.start()
        sample_video(path)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    peak_bytes(clips["ideal10"])  # warm up lazy allocations
    assert peak_bytes(clips["long75"]) - peak_bytes(clips["ideal10"]) < 3_000_000
