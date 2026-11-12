"""S3 inputs stream into scoring: bounded downloads, ordered duplicate checks, a clean cache."""

import dataclasses
import json
import random
import threading
import time
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import NoCredentialsError
from moto import mock_aws

import pre_annotate
import score_videos
from src import config
from src.config import Settings
from src.scorer import quality_scorer, s3_io
from src.scorer.frame_sampler import sample_video
from src.scorer.provider_errors import AuthError
from src.scorer.quality_scorer import QualityScorer, VideoInput
from src.scorer.vlm_client import OfflineClient

HOLD_SECONDS = 8


class RecordingS3:
    """The moto client, with a hook before each download and a log of what was asked for."""

    def __init__(self, client):
        self._client = client
        self.downloads, self.heads = [], []
        self.before_download = lambda key: None
        self.fail_listing = False
        self.running = self.peak = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._client, name)

    def head_object(self, **kwargs):
        self.heads.append(kwargs["Key"])
        return self._client.head_object(**kwargs)

    def get_paginator(self, name):
        if self.fail_listing:
            raise NoCredentialsError
        return self._client.get_paginator(name)

    def download_file(self, bucket, key, path):
        with self._lock:
            self.downloads.append(key)
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            self.before_download(key)
            self._client.download_file(bucket, key, path)
        finally:
            with self._lock:
                self.running -= 1


@pytest.fixture
def s3(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        real = boto3.client("s3", region_name="us-east-1")
        real.create_bucket(Bucket="clips")
        recording = RecordingS3(real)
        monkeypatch.setattr(s3_io, "_client", lambda settings: recording)
        recording.settings = Settings(cache_dir=tmp_path / "cache", aws_region="us-east-1")
        yield recording


def upload(s3, clips, *keys):
    """Store each clip under a numbered key, in order, and return the URIs."""
    uris = []
    for number, key in enumerate(keys):
        name = f"clip{number}.mp4"
        s3.put_object(Bucket="clips", Key=name, Body=clips[key].read_bytes())
        uris.append(f"s3://clips/{name}")
    return uris


class Watching(OfflineClient):
    def __init__(self, on_score=lambda video: None):
        super().__init__(Settings())
        self.on_score = on_score

    def score(self, video, prompt_version="v3", duplicate_note=""):
        self.on_score(video)
        return super().score(video, prompt_version, duplicate_note)


def inputs(uris, settings):
    # Falls back to the old eager call, so these tests can fail for the real reason on old code.
    resolve = getattr(quality_scorer, "iter_inputs", quality_scorer.resolve_inputs)
    return resolve(uris, settings)


def cached_files(s3):
    return [p for p in s3.settings.cache_dir.rglob("*") if p.is_file()]


def test_scoring_starts_before_the_last_file_has_downloaded(s3, clips):
    uris = upload(s3, clips, "ideal10", "other10", "tripod10", "dark10")
    scoring_started = threading.Event()
    held = []

    def hold_the_last_file(key):
        if key == "clip3.mp4":
            held.append(scoring_started.wait(HOLD_SECONDS))

    s3.before_download = hold_the_last_file
    client = Watching(lambda video: scoring_started.set())
    results = QualityScorer(s3.settings, client=client).score_batch(inputs(uris, s3.settings))
    assert held == [True]
    assert len(results) == 4


def test_downloads_run_at_most_the_limit_at_once(s3, clips):
    uris = upload(s3, clips, *["ideal10"] * 6)
    s3.before_download = lambda key: time.sleep(0.2)
    scorer = QualityScorer(s3.settings, download_workers=2)
    scorer.score_batch(quality_scorer.iter_inputs(uris, s3.settings))
    assert s3.peak == 2


def test_four_downloads_run_at_once_by_default(s3, clips):
    uris = upload(s3, clips, *["ideal10"] * 8)
    s3.before_download = lambda key: time.sleep(0.3)
    scorer = QualityScorer(s3.settings, workers=4)
    assert scorer.download_workers == 4
    scorer.score_batch(quality_scorer.iter_inputs(uris, s3.settings))
    assert s3.peak == 4


def test_duplicate_check_follows_input_order_when_a_later_file_finishes_first(
    monkeypatch, s3, clips
):
    uris = upload(s3, clips, "ideal10", "ideal10")
    later_file_decoded = threading.Event()
    real_sample = quality_scorer.sample_video

    def sample(path):
        sampled = real_sample(path)
        if Path(path).name == "clip1.mp4":
            later_file_decoded.set()
        return sampled

    held = []

    def hold_the_first_file(key):
        if key == "clip0.mp4":
            held.append(later_file_decoded.wait(HOLD_SECONDS))

    monkeypatch.setattr(quality_scorer, "sample_video", sample)
    s3.before_download = hold_the_first_file
    results = QualityScorer(s3.settings).score_batch(inputs(uris, s3.settings))

    assert held == [True]
    original, copy = results
    assert original["meta"]["source"] == uris[0]
    assert original["meta"]["duplicate_of"] is None
    assert copy["meta"]["duplicate_of"] == original["video_id"]


def test_results_keep_input_order_however_downloads_finish(s3, clips):
    uris = upload(s3, clips, "ideal10", "other10", "tripod10", "dark10")
    delays = {"clip0.mp4": 0.6, "clip1.mp4": 0.4, "clip2.mp4": 0.2, "clip3.mp4": 0.0}
    s3.before_download = lambda key: time.sleep(delays[key])
    results = QualityScorer(s3.settings).score_batch(inputs(uris, s3.settings))
    assert [r["meta"]["source"] for r in results] == uris


def test_files_are_deleted_once_scored(s3, clips):
    uris = upload(s3, clips, "ideal10", "other10")
    QualityScorer(s3.settings).score_batch(inputs(uris, s3.settings))
    assert len(s3.downloads) == 2
    assert not list(s3.settings.cache_dir.rglob("*"))


def test_a_file_that_fails_is_deleted_too(s3, clips):
    s3.put_object(Bucket="clips", Key="broken.mp4", Body=b"not a video")
    results = QualityScorer(s3.settings).score_batch(inputs(["s3://clips/broken.mp4"], s3.settings))
    assert "sampling failed" in results[0]["error"]
    assert cached_files(s3) == []


def test_disk_use_stays_within_the_in_flight_limit(monkeypatch, s3, clips):
    monkeypatch.setattr(config, "MAX_VIDEOS_IN_FLIGHT", 2, raising=False)
    uris = upload(s3, clips, *["ideal10"] * 6)
    on_disk = []
    client = Watching(lambda video: on_disk.append(len(cached_files(s3))))
    results = QualityScorer(s3.settings, client=client, workers=1).score_batch(
        inputs(uris, s3.settings)
    )
    assert [r["error"] for r in results if "error" in r] == []
    assert len(on_disk) == 6
    assert max(on_disk) <= 2


def test_keep_cache_leaves_the_files_and_saves_the_next_download(s3, clips):
    uris = upload(s3, clips, "ideal10", "other10")
    scorer = QualityScorer(s3.settings, keep_cache=True)
    scorer.score_batch(quality_scorer.iter_inputs(uris, s3.settings))
    assert len(cached_files(s3)) == 2

    QualityScorer(s3.settings, keep_cache=True).score_batch(
        quality_scorer.iter_inputs(uris, s3.settings)
    )
    assert len(s3.downloads) == 2


def test_a_run_only_deletes_what_it_downloaded(s3, clips):
    uris = upload(s3, clips, "ideal10")
    QualityScorer(s3.settings, keep_cache=True).score_batch(
        quality_scorer.iter_inputs(uris, s3.settings)
    )
    QualityScorer(s3.settings).score_batch(quality_scorer.iter_inputs(uris, s3.settings))
    assert len(cached_files(s3)) == 1


def test_object_over_the_cap_is_refused_before_any_download(s3, clips):
    s3.settings = Settings(
        cache_dir=s3.settings.cache_dir, aws_region="us-east-1", s3_max_download_mb=0.5
    )
    s3.put_object(Bucket="clips", Key="small.mp4", Body=clips["ideal10"].read_bytes()[:1000])
    uris = upload(s3, clips, "ideal10")
    results = QualityScorer(s3.settings).score_batch(
        inputs(["s3://clips/small.mp4", *uris], s3.settings)
    )
    assert "S3_MAX_DOWNLOAD_MB" in results[1]["error"]
    assert "clip0.mp4" in s3.heads and "clip0.mp4" not in s3.downloads


def test_inputs_are_resolved_lazily(s3, clips):
    uris = upload(s3, clips, "ideal10", "other10", "tripod10")
    stream = quality_scorer.iter_inputs(["s3://clips/"], s3.settings)
    assert s3.heads == []
    first = next(stream)
    assert first.source == uris[0]
    assert len(s3.heads) == 1
    assert s3.downloads == []


@pytest.mark.parametrize("bad", ["nope.mp4", "@nope.txt", "s3://../x.mp4"])
def test_a_bad_input_fails_at_once_not_after_scoring_has_started(s3, clips, bad):
    with pytest.raises((OSError, ValueError)):
        quality_scorer.iter_inputs([str(clips["ideal10"]), bad], s3.settings)
    assert s3.heads == []


def test_listing_failure_keeps_what_was_scored(s3, clips):
    s3.fail_listing = True
    scorer = QualityScorer(s3.settings)
    stream = quality_scorer.iter_inputs([str(clips["ideal10"]), "s3://clips/"], s3.settings)
    results = scorer.score_batch(stream)
    assert len(results) == 1 and "tier" in results[0]
    assert "credentials" in scorer.input_failure


def test_no_more_downloads_once_the_key_is_rejected(monkeypatch, s3, clips):
    monkeypatch.setattr(config, "MAX_VIDEOS_IN_FLIGHT", 2, raising=False)
    uris = upload(s3, clips, *["ideal10"] * 8)

    def reject(video):
        raise AuthError("gemini rejected the request (HTTP 401): check GEMINI_API_KEY")

    client = Watching(reject)
    scorer = QualityScorer(s3.settings, client=client, workers=1, download_workers=1)
    results = scorer.score_batch(inputs(uris, s3.settings))
    assert len(s3.downloads) <= 2
    assert [r["error"] for r in results].count("skipped: authentication failed") >= 6
    assert cached_files(s3) == []


@pytest.fixture
def workdir(monkeypatch, s3, tmp_path):
    """Run from an empty folder, with the cache in the default place."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def default_cache_files(folder):
    return [p for p in (folder / ".cache").rglob("*") if p.is_file()]


def test_cli_scores_a_prefix_and_cleans_up(workdir, s3, clips):
    upload(s3, clips, "ideal10", "other10")
    args = ["s3://clips/", "--provider", "offline", "--download-workers", "2", "-o", "out.json"]
    assert score_videos.main(args) == 0
    assert len(json.loads(Path("out.json").read_text(encoding="utf-8"))) == 2
    assert default_cache_files(workdir) == []


def test_cli_keep_cache(workdir, s3, clips):
    upload(s3, clips, "ideal10", "other10")
    args = ["s3://clips/", "--provider", "offline", "--keep-cache", "-o", "out.json"]
    assert score_videos.main(args) == 0
    assert len(default_cache_files(workdir)) == 2


def test_cli_writes_results_before_reporting_a_listing_failure(workdir, s3, clips, capsys):
    s3.fail_listing = True
    args = [str(clips["ideal10"]), "s3://clips/", "--provider", "offline", "-o", "out.json"]
    assert score_videos.main(args) == 2
    assert "credentials" in capsys.readouterr().err
    assert json.loads(Path("out.json").read_text(encoding="utf-8"))["tier"]


def test_empty_prefix_is_reported(workdir, s3, capsys):
    assert score_videos.main(["s3://clips/none/", "--provider", "offline"]) == 2
    assert "no videos" in capsys.readouterr().err


def annotation_args(extra=()):
    return [
        "--scores",
        "s.json",
        "--provider",
        "offline",
        "--coco",
        "c.json",
        "--label-studio",
        "l.json",
        *extra,
    ]


def write_scores(uri):
    record = {"video_id": "x", "tier": "usable", "meta": {"source": uri}}
    Path("s.json").write_text(json.dumps([record]), encoding="utf-8")


def test_pre_annotate_cleans_up_what_it_downloads(workdir, s3, clips):
    (uri,) = upload(s3, clips, "ideal10")
    write_scores(uri)
    assert pre_annotate.main(annotation_args()) == 0
    assert s3.downloads == ["clip0.mp4"]
    assert default_cache_files(workdir) == []


def test_pre_annotate_keep_cache(workdir, s3, clips):
    (uri,) = upload(s3, clips, "ideal10")
    write_scores(uri)
    assert pre_annotate.main(annotation_args(["--keep-cache"])) == 0
    assert len(default_cache_files(workdir)) == 1


def test_an_unexpected_download_error_fails_that_video_only(monkeypatch, s3, clips):
    uris = upload(s3, clips, "ideal10", "other10")
    real_fetch = s3_io.fetch

    def fetch(uri, settings, info=None):
        if uri == uris[0]:
            raise RuntimeError("something unexpected")
        return real_fetch(uri, settings, info)

    monkeypatch.setattr(s3_io, "fetch", fetch)
    first, second = QualityScorer(s3.settings).score_batch(inputs(uris, s3.settings))
    assert "something unexpected" in first["error"]
    assert "tier" in second


def test_duplicate_flags_do_not_depend_on_timing(monkeypatch, clips):
    base = sample_video(clips["ideal10"])
    groups = 7
    group_hashes = [[(0x9E3779B97F4A7C15 * (g + 1)) % 2**64] * 8 for g in range(groups)]
    paths = [f"v{i}.mp4" for i in range(30)]
    videos = [VideoInput(p, p, f"{i:08x}-0000-4000-8000-000000000000") for i, p in enumerate(paths)]
    expected = [None if i < groups else videos[i % groups].video_id for i in range(30)]

    for seed in range(5):
        rng = random.Random(seed)
        delays = {p: rng.random() * 0.02 for p in paths}

        def sample(path, delays=delays):
            time.sleep(delays[path])
            return dataclasses.replace(base, hashes=group_hashes[paths.index(path) % groups])

        monkeypatch.setattr(quality_scorer, "sample_video", sample)
        results = QualityScorer(Settings(), workers=6).score_batch(list(videos))
        assert [r["meta"]["duplicate_of"] for r in results] == expected


def test_the_key_check_does_not_depend_on_what_is_on_disk(monkeypatch, s3):
    def must_not_run(self, strict=False):
        raise AssertionError("resolve() reads the disk, so cleanup elsewhere can change its answer")

    monkeypatch.setattr(Path, "resolve", must_not_run)
    with pytest.raises(s3_io.S3Error, match="unsafe"):
        s3_io.head("s3://clips/../../escaped.mp4", s3.settings)
    with pytest.raises(s3_io.S3Error, match="404|Not Found"):
        s3_io.head("s3://clips/a/b/fine.mp4", s3.settings)


def test_cleaning_up_one_file_cannot_break_a_download_next_to_it(s3):
    keys = [f"clip{i}.mp4" for i in range(8)]
    for key in keys:  # identical bytes, so they all share one ETag folder
        s3.put_object(Bucket="clips", Key=key, Body=b"same bytes" * 500)
    failures = []

    def keep_cycling(key):
        uri = f"s3://clips/{key}"
        for _ in range(40):
            try:
                path, _ = s3_io.fetch(uri, s3.settings)
                s3_io.remove_cached(path, s3.settings)
            except OSError as exc:
                failures.append(str(exc))

    threads = [threading.Thread(target=keep_cycling, args=(key,)) for key in keys]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert not list(s3.settings.cache_dir.rglob("*"))
