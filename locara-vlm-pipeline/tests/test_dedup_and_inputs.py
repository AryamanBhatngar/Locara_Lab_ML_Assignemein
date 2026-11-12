"""Duplicate registry, input resolution, and how a batch behaves when one input is bad."""

import threading
import time

import cv2
import numpy as np
import pytest

from src import config
from src.config import Settings
from src.schemas.scoring_schema import video_id_for
from src.scorer import dedup, quality_scorer, s3_io
from src.scorer.dedup import HashRegistry, sequence_distance
from src.scorer.frame_sampler import sample_video
from src.scorer.quality_scorer import QualityScorer, VideoInput, resolve_inputs
from src.scorer.s3_io import parse_s3_uri
from src.scorer.vlm_client import OfflineClient


def solid_clip(path, value):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    for _ in range(45):
        writer.write(np.full((360, 640, 3), value, np.uint8))
    writer.release()
    return path


def test_blank_frames_are_not_evidence_of_a_duplicate(tmp_path):
    registry = HashRegistry()
    registry.add("black", sample_video(solid_clip(tmp_path / "black.mp4", 0)).hashes)
    match = registry.best_match("wall", sample_video(solid_clip(tmp_path / "wall.mp4", 200)).hashes)
    assert match is not None and not match.is_duplicate


def test_blank_frames_are_ignored_but_real_frames_still_compare():
    real = [0x0F0F0F0F0F0F0F0F] * 6
    assert sequence_distance([0, 0, *real], [0, 0, *real]) == 0.0
    assert sequence_distance([0] * 8, [0] * 8) == 64.0


def test_check_and_add_is_atomic(monkeypatch):
    registry = HashRegistry()
    registry.add("unrelated", [1 << 40] * 8)
    compare = dedup.sequence_distance

    def slow_compare(a, b):
        time.sleep(0.02)
        return compare(a, b)

    monkeypatch.setattr(dedup, "sequence_distance", slow_compare)
    hashes = [0x0F0F0F0F0F0F0F0F] * 8
    barrier = threading.Barrier(6)
    flagged = []

    def worker(i):
        barrier.wait()
        match = registry.check_and_add(f"copy{i}", hashes)
        flagged.append(bool(match and match.is_duplicate))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(flagged) == 5  # every copy but the first


def test_save_is_safe_while_clips_are_being_added(tmp_path):
    registry = HashRegistry(tmp_path / "registry.json")
    late_adds = []

    class AddsDuringSave(int):
        """Formatting this means save() is mid-iteration, so add a clip from another thread."""

        def __format__(self, spec):
            late = threading.Thread(target=registry.add, args=("late", [1]))
            late_adds.append(late)
            late.start()
            late.join(timeout=0.2)  # with the lock held this waits, and save() carries on
            return super().__format__(spec)

    registry.add("first", [AddsDuringSave(5)])
    registry.add("second", [2])
    registry.save()
    late_adds[0].join()
    assert len(registry) == 3


@pytest.mark.parametrize("content", ["not json", '["a"]', '{"clip": ["zz"]}', '{"clip": 5}'])
def test_corrupt_registry_names_the_file(tmp_path, content):
    path = tmp_path / "registry.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="registry.json"):
        HashRegistry(path)


def test_same_file_in_two_folders_shares_an_id_and_is_flagged(tmp_path, clips):
    for folder in ("first", "second"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "clip.mp4").write_bytes(clips["ideal10"].read_bytes())
    videos = resolve_inputs([str(tmp_path / "first"), str(tmp_path / "second")], Settings())
    assert videos[0].video_id == videos[1].video_id

    first, second = QualityScorer(Settings(), workers=2).score_batch(videos)
    assert second["scores"]["duplicate_risk"]["score"] == 1
    assert second["meta"]["duplicate_of"] == first["video_id"]


def test_same_path_given_twice_is_scored_once(clips):
    path = str(clips["ideal10"])
    assert len(resolve_inputs([path, path, str(clips["ideal10"].parent)], Settings())) == len(
        resolve_inputs([str(clips["ideal10"].parent)], Settings())
    )


def test_list_file_handles_bom_and_indented_comments(tmp_path, clips):
    listing = tmp_path / "videos.txt"
    rows = f"\ufeff{clips['ideal10']}\n   # a note\n\n{clips['other10']}\n"
    listing.write_text(rows, encoding="utf-8")
    assert [v.local_path for v in resolve_inputs([f"@{listing}"], Settings())] == [
        str(clips["ideal10"]),
        str(clips["other10"]),
    ]


def test_list_file_that_includes_itself_is_rejected(tmp_path):
    listing = tmp_path / "loop.txt"
    listing.write_text(f"@{listing}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"loop\.txt"):
        resolve_inputs([f"@{listing}"], Settings())


def test_failed_s3_download_only_fails_that_video(monkeypatch, clips):
    def fetch(uri, settings, info=None):
        if uri.endswith("bad.mp4"):
            raise s3_io.S3Error("access denied")
        return clips["ideal10"], False

    monkeypatch.setattr(s3_io, "head", lambda uri, settings: s3_io.ObjectInfo("etag", 10))
    monkeypatch.setattr(s3_io, "fetch", fetch)
    videos = resolve_inputs(["s3://bucket/bad.mp4", "s3://bucket/good.mp4"], Settings())
    bad, good = QualityScorer(Settings()).score_batch(videos)
    assert "access denied" in bad["error"] and bad["source"] == "s3://bucket/bad.mp4"
    assert "tier" in good


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://bucket/a/clip.mp4", ("bucket", "a/clip.mp4")),
        ("s3://bucket/take#2.mp4", ("bucket", "take#2.mp4")),
        ("s3://bucket/why?.mp4", ("bucket", "why?.mp4")),
    ],
)
def test_s3_keys_keep_url_special_characters(uri, expected):
    assert parse_s3_uri(uri) == expected


def test_only_a_limited_number_of_videos_hold_frames_at_once(monkeypatch, clips):
    monkeypatch.setattr(config, "MAX_VIDEOS_IN_FLIGHT", 2)
    lock = threading.Lock()
    sampled = scored = peak = 0

    def recording_sample(path):
        nonlocal sampled, peak
        result = sample_video(path)
        with lock:
            sampled += 1
            peak = max(peak, sampled - scored)
        return result

    class RecordingClient(OfflineClient):
        def score(self, video, prompt_version="v3", duplicate_note=""):
            nonlocal scored
            reply = super().score(video, prompt_version, duplicate_note)
            with lock:
                scored += 1
            return reply

    monkeypatch.setattr(quality_scorer, "sample_video", recording_sample)
    path = str(clips["ideal10"])
    videos = [VideoInput(path, path, video_id_for(f"copy{i}")) for i in range(6)]
    QualityScorer(Settings(), client=RecordingClient(Settings()), workers=1).score_batch(videos)
    assert sampled == 6 and peak <= 2


def test_duplicates_are_found_across_chunks(monkeypatch, clips):
    monkeypatch.setattr(config, "MAX_VIDEOS_IN_FLIGHT", 1)
    videos = [
        VideoInput(str(clips[key]), str(clips[key]), video_id_for(key))
        for key in ("ideal10", "other10", "ideal10_dup")
    ]
    first, _, copy = QualityScorer(Settings(), workers=1).score_batch(videos)
    assert copy["meta"]["duplicate_of"] == first["video_id"]
