"""Decoding runs in worker processes, so a hung or crashed decoder costs one clip."""

import base64
import json
import pickle
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

import evaluate_pipeline
import pre_annotate
import score_videos
from src.config import Settings
from src.schemas.scoring_schema import video_id_for
from src.scorer import quality_scorer
from src.scorer.frame_sampler import VideoReadError, sample_video
from src.scorer.quality_scorer import QualityScorer, VideoInput
from src.scorer.vlm_client import OpenAIClient
from tests import fake_samplers

TIMEOUT = 4


def clip_file(clips, folder, name, key="ideal10"):
    path = folder / name
    shutil.copyfile(clips[key], path)
    return path


def video(clips, folder, name, key="ideal10"):
    path = clip_file(clips, folder, name, key)
    return VideoInput(path.as_posix(), str(path), video_id_for(name))


def scorer(workers=1, timeout=TIMEOUT):
    return QualityScorer(
        Settings(), workers=workers, decode_timeout=timeout, sampler=fake_samplers.by_name
    )


def pid_of(folder, stem):
    return int((folder / f"{stem}.pid").read_text(encoding="utf-8"))


def child_pids():
    return {child.pid for child in psutil.Process().children(recursive=True)}


def test_hung_decoder_is_killed_and_recorded(tmp_path, clips):
    results = scorer(timeout=2).score_batch([video(clips, tmp_path, "hang.mp4")])
    assert results[0]["error"] == "sampling timed out after 2 s"
    assert results[0]["source"].endswith("hang.mp4")
    assert not psutil.pid_exists(pid_of(tmp_path, "hang"))


def test_a_fractional_timeout_reads_naturally(tmp_path, clips):
    results = scorer(timeout=1.5).score_batch([video(clips, tmp_path, "hang.mp4")])
    assert results[0]["error"] == "sampling timed out after 1.5 s"


def test_other_videos_still_finish(tmp_path, clips):
    names = ["first.mp4", "hang.mp4", "second.mp4", "third.mp4"]
    videos = [video(clips, tmp_path, name) for name in names]
    results = scorer(timeout=2).score_batch(videos)
    assert ["tier" in r for r in results] == [True, False, True, True]
    assert [r.get("source") or r["meta"]["source"] for r in results] == [v.source for v in videos]


def test_the_worker_next_to_a_hung_one_keeps_running(tmp_path, clips):
    videos = [
        video(clips, tmp_path, "hang.mp4"),
        video(clips, tmp_path, "filler.mp4"),
        video(clips, tmp_path, "alongside.mp4"),
    ]
    hung, filler, alongside = scorer(workers=2).score_batch(videos)
    assert hung["error"] == f"sampling timed out after {TIMEOUT} s"
    assert "tier" in filler
    assert "tier" in alongside


def test_a_crashing_decoder_fails_that_video_only(tmp_path, clips):
    names = ["crash.mp4", "good.mp4"]
    crashed, good = scorer().score_batch([video(clips, tmp_path, name) for name in names])
    assert "crash" in crashed["error"].lower()
    assert crashed["error"].startswith("sampling failed")
    assert "tier" in good


def test_a_read_error_in_the_worker_is_recorded(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    results = scorer().score_batch([VideoInput(str(broken), str(broken), video_id_for("broken"))])
    assert results[0]["error"].startswith("sampling failed: cannot open video")


def test_workers_are_reused_between_clips(tmp_path, clips):
    names = ["a.mp4", "b.mp4", "c.mp4"]
    scorer().score_batch([video(clips, tmp_path, name) for name in names])
    assert len({pid_of(tmp_path, stem) for stem in "abc"}) == 1


def test_a_timed_out_video_is_not_registered_as_an_original(tmp_path, clips):
    timed_out = scorer(timeout=2)
    timed_out.score_batch([video(clips, tmp_path, "hang.mp4")])
    assert len(timed_out.registry) == 0


def test_no_worker_outlives_the_batch(tmp_path, clips):
    # The first batch starts any long-lived helper, such as the POSIX resource tracker.
    scorer().score_batch([video(clips, tmp_path, "warm.mp4")])
    before = child_pids()

    names = ["good.mp4", "hang.mp4", "crash.mp4", "good2.mp4"]
    results = scorer(workers=2, timeout=2).score_batch(
        [video(clips, tmp_path, name) for name in names]
    )
    assert len(results) == 4
    assert child_pids() == before


def test_no_worker_outlives_a_broken_input_stream(tmp_path, clips):
    scorer().score_batch([video(clips, tmp_path, "warm.mp4")])
    before = child_pids()
    good = video(clips, tmp_path, "good.mp4")

    def broken_stream():
        yield good
        raise OSError("listing failed")

    run = scorer()
    assert len(run.score_batch(broken_stream())) == 1
    assert "listing failed" in run.input_failure
    assert child_pids() == before


def test_closing_the_pool_kills_a_busy_worker(tmp_path, clips):
    from src.scorer.decode_pool import DecodePool

    path = clip_file(clips, tmp_path, "hang.mp4")
    pool = DecodePool(60, fake_samplers.by_name)
    errors = []

    def decode():
        try:
            pool.sample(str(path))
        except VideoReadError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=decode)
    thread.start()
    deadline = time.monotonic() + 20
    while not (tmp_path / "hang.pid").is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    pid = pid_of(tmp_path, "hang")

    pool.close()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert not psutil.pid_exists(pid)


def test_the_result_is_small_enough_to_pass_between_processes(clips):
    sampled = sample_video(clips["long75"])
    assert sampled.frame_count == 16
    assert len(pickle.dumps(sampled)) < 1_000_000
    assert all(frame.jpeg[:2] == b"\xff\xd8" for frame in sampled.frames)
    assert not any(isinstance(value, np.ndarray) for value in vars(sampled.frames[0]).values())


def test_the_model_receives_the_jpeg_bytes(monkeypatch, clips):
    import openai

    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content="{}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)

    sampled = sample_video(clips["ideal10"])
    OpenAIClient(Settings(provider="openai"), api_key="test").score(sampled)
    image = seen["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image.split(",", 1)[1]) == sampled.frames[0].jpeg


def test_hand_sides_are_measured_on_the_full_quality_frame(clips):
    sampled = sample_video(clips["ideal10"])
    assert all(len(frame.hand_sides) == 2 for frame in sampled.frames)
    assert any(any(frame.hand_sides) for frame in sampled.frames)


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_records_a_timeout_and_scores_the_rest(monkeypatch, workdir, clips):
    monkeypatch.setattr(quality_scorer, "sample_video", fake_samplers.by_name)
    hang = clip_file(clips, workdir, "hang.mp4")
    good = clip_file(clips, workdir, "good.mp4", "other10")
    args = [str(hang), str(good), "--provider", "offline", "--decode-timeout", "2", "-o", "o.json"]
    assert score_videos.main(args) == 1
    timed_out, scored = json.loads(Path("o.json").read_text(encoding="utf-8"))
    assert timed_out["error"] == "sampling timed out after 2 s"
    assert "tier" in scored


def test_cli_uses_a_worker_process_by_default(monkeypatch, workdir, clips):
    monkeypatch.setattr(quality_scorer, "sample_video", fake_samplers.by_name)
    good = clip_file(clips, workdir, "good.mp4")
    assert score_videos.main([str(good), "--provider", "offline", "-o", "o.json"]) == 0
    assert pid_of(workdir, "good") != psutil.Process().pid


def test_pre_annotate_survives_a_hung_decoder(monkeypatch, workdir, clips, capsys):
    monkeypatch.setattr(pre_annotate, "sample_video", fake_samplers.by_name)
    records = [
        {"video_id": video_id_for(name), "tier": "usable", "meta": {"source": name}}
        for name in ("hang.mp4", "good.mp4")
    ]
    clip_file(clips, workdir, "hang.mp4")
    clip_file(clips, workdir, "good.mp4")
    Path("s.json").write_text(json.dumps(records), encoding="utf-8")
    args = ["--scores", "s.json", "--provider", "offline", "--coco", "c.json"]
    args += ["--label-studio", "l.json", "--decode-timeout", "2"]
    assert pre_annotate.main(args) == 1
    assert "sampling timed out after 2 s" in capsys.readouterr().err
    assert len(json.loads(Path("c.json").read_text(encoding="utf-8"))["videos"]) == 1


def test_evaluate_accepts_a_decode_timeout(workdir, clips):
    truth = {clips["ideal10"].name: {"tier": "usable", "scores": {}}}
    Path("t.json").write_text(json.dumps(truth), encoding="utf-8")
    args = ["--videos", str(clips["ideal10"]), "--provider", "offline", "--truth", "t.json"]
    assert evaluate_pipeline.main([*args, "--decode-timeout", "30", "--out", "r.json"]) == 0


@pytest.mark.parametrize("value", ["0", "-1", "soon", "nan", "inf"])
@pytest.mark.parametrize(
    "main,required",
    [
        (score_videos.main, ["clip.mp4"]),
        (pre_annotate.main, ["--scores", "s.json"]),
        (evaluate_pipeline.main, ["--videos", "clip.mp4", "--truth", "t.json"]),
    ],
)
def test_decode_timeout_must_be_a_positive_number(main, required, value):
    with pytest.raises(SystemExit) as exit_info:
        main([*required, "--decode-timeout", value])
    assert exit_info.value.code == 2
