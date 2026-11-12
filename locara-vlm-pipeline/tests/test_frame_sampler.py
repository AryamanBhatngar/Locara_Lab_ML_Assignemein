import cv2
import numpy as np
import pytest

from src import config
from src.scorer.dedup import HashRegistry, sequence_distance
from src.scorer.frame_sampler import (
    VideoReadError,
    dhash,
    frames_for_duration,
    hamming,
    sample_video,
)
from src.scorer.vlm_client import offline_scores


def pixels(frame):
    return cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize(
    "duration,expected", [(5, 8), (10, 8), (20, 10), (30, 12), (45, 12), (60, 12), (90, 16)]
)
def test_frame_budget_by_duration(duration, expected):
    assert frames_for_duration(duration) == expected


def test_short_clip_gets_min_frames_in_time_order(clips):
    s = sample_video(clips["ideal10"])
    assert s.frame_count == config.MIN_FRAMES
    times = [f.time_sec for f in s.frames]
    assert times == sorted(times) and times[0] < 2.0 and times[-1] > 8.0
    assert max(pixels(s.frames[0]).shape[:2]) == config.FRAME_LONG_SIDE
    assert len(s.hashes) == config.MIN_FRAMES


def test_60s_and_75s_clips_scale_frames(clips):
    assert sample_video(clips["long60"]).frame_count == 12
    s = sample_video(clips["long75"])
    assert s.frame_count == 16
    gaps = [b.time_sec - a.time_sec for a, b in zip(s.frames, s.frames[1:])]
    assert max(gaps) < 2.5 * (75 / 16)  # coverage spread across the whole clip


def test_sampling_is_fast(clips):
    s = sample_video(clips["long60"])
    assert s.timings_ms["total"] < 15_000  # generous for CI; typically ~1 s


def test_telemetry_separates_defects(clips):
    ideal = sample_video(clips["ideal10"]).telemetry
    blur = sample_video(clips["blur10"]).telemetry
    dark = sample_video(clips["dark10"]).telemetry
    tripod = sample_video(clips["tripod10"]).telemetry
    assert blur["flow_mean_pct"] > 3 * ideal["flow_mean_pct"]
    assert dark["brightness_mean"] < 35 < ideal["brightness_mean"]
    assert tripod["camera_motion_pct"] < 0.05 < ideal["camera_motion_pct"]
    assert ideal["hand_presence_frac"] > 0.8 and tripod["hand_presence_frac"] < 0.1


def test_offline_baseline_flags_defects(clips):
    blur = offline_scores(sample_video(clips["blur10"]).telemetry)
    dark = offline_scores(sample_video(clips["dark10"]).telemetry)
    tripod = offline_scores(sample_video(clips["tripod10"]).telemetry)
    assert blur["motion_quality"]["score"] <= 2
    assert dark["lighting_exposure"]["score"] == 1
    assert tripod["egocentric_framing"]["score"] == 0


def test_dhash_basics(clips):
    s = sample_video(clips["ideal10"])
    g = cv2.cvtColor(pixels(s.frames[0]), cv2.COLOR_BGR2GRAY)
    assert hamming(dhash(g), dhash(g)) == 0
    assert hamming(dhash(g), dhash(cv2.GaussianBlur(g, (3, 3), 0))) <= 8


def test_reencoded_copy_is_duplicate_and_other_clip_is_not(clips):
    reg = HashRegistry()
    a = sample_video(clips["ideal10"]).hashes
    reg.check_and_add("a", a)
    dup = reg.best_match("b", sample_video(clips["ideal10_dup"]).hashes)
    other = reg.best_match("c", sample_video(clips["tripod10"]).hashes)
    assert dup is not None and dup.is_duplicate
    assert other is not None and not other.is_duplicate
    assert sequence_distance([], a) == 64.0


def test_registry_persists(tmp_path):
    path = tmp_path / "reg.json"
    reg = HashRegistry(path)
    reg.add("x", [1, 2, 3])
    reg.save()
    assert len(HashRegistry(path)) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoReadError):
        sample_video(tmp_path / "nope.mp4")
