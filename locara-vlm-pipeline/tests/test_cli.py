"""CLI behaviour on bad input: a short message and exit code 2, never a traceback."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from botocore.exceptions import NoCredentialsError

import evaluate_pipeline
import pre_annotate
import score_videos
from src.config import get_settings
from src.schemas.scoring_schema import video_id_for
from src.scorer import s3_io
from tests import generate_synthetic_clips
from tests.generate_synthetic_clips import ClipSpec, _truth, render

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = {
    "egocentric_framing": 1,
    "motion_quality": 4,
    "scene_diversity": 4,
    "hand_visibility": 5,
    "lighting_exposure": 4,
    "duplicate_risk": 0,
    "overall_usability": 4,
}
GROUND_TRUTH = {"a.mp4": {"tier": "usable", "scores": DIMENSIONS}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Run from an empty folder: no real .env is read and nothing lands in the repo."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "VLM_PROVIDER",
        "VLM_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)


def scored(name="a.mp4", tier="usable", **meta):
    return {
        "video_id": video_id_for(name),
        "processed_at": "2026-11-06T10:00:00Z",
        "frame_count_sampled": 8,
        "scores": {
            k: {"score": v, "confidence": 0.9, "reasoning": "ok"} for k, v in DIMENSIONS.items()
        },
        "tier": tier,
        "tier_confidence": 0.9,
        "meta": {"source": name, "provider": "offline", "model": "m", **meta},
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def assert_clean_failure(capsys, code, expected):
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ") and expected in err


def test_missing_video(capsys):
    assert_clean_failure(
        capsys, score_videos.main(["nope.mp4", "--provider", "offline"]), "nope.mp4"
    )


def test_missing_list_file(capsys):
    assert_clean_failure(
        capsys, score_videos.main(["@nope.txt", "--provider", "offline"]), "nope.txt"
    )


def test_list_file_that_includes_itself(capsys):
    Path("loop.txt").write_text("@loop.txt\n", encoding="utf-8")
    assert_clean_failure(
        capsys, score_videos.main(["@loop.txt", "--provider", "offline"]), "loop.txt"
    )


def test_missing_api_key(capsys):
    assert_clean_failure(
        capsys, score_videos.main(["a.mp4", "--provider", "gemini"]), "GEMINI_API_KEY"
    )


def test_unknown_provider_in_environment(capsys, monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "bogus")
    assert_clean_failure(capsys, score_videos.main(["a.mp4"]), "VLM_PROVIDER")


def test_non_numeric_timeout(capsys, monkeypatch):
    monkeypatch.setenv("VLM_TIMEOUT_S", "soon")
    assert_clean_failure(
        capsys, score_videos.main(["a.mp4", "--provider", "offline"]), "VLM_TIMEOUT_S"
    )


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_non_finite_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VLM_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="VLM_TIMEOUT_S"):
        get_settings("offline")


def test_empty_numeric_setting_uses_the_default(monkeypatch):
    monkeypatch.setenv("VLM_TIMEOUT_S", "")
    assert get_settings("offline").request_timeout_s == 45.0


def test_corrupt_hash_registry(capsys, clips):
    Path("reg.json").write_text("not json", encoding="utf-8")
    args = [str(clips["ideal10"]), "--provider", "offline", "--hash-registry", "reg.json"]
    assert_clean_failure(capsys, score_videos.main(args), "reg.json")


def test_output_problem_is_reported_before_any_scoring(capsys, monkeypatch, clips):
    def must_not_run(self, videos):
        raise AssertionError("scored before checking the output path")

    monkeypatch.setattr(score_videos.QualityScorer, "score_batch", must_not_run)
    Path("taken").mkdir()
    args = [str(clips["ideal10"]), "--provider", "offline", "-o", "taken"]
    assert_clean_failure(capsys, score_videos.main(args), "taken")


def test_results_are_written_even_if_the_registry_cannot_be_saved(monkeypatch, clips):
    def fail(self):
        raise OSError("disk full")

    monkeypatch.setattr(score_videos.HashRegistry, "save", fail)
    args = [
        str(clips["ideal10"]),
        "--provider",
        "offline",
        "-o",
        "out.json",
        "--hash-registry",
        "r.json",
    ]
    assert score_videos.main(args) == 2
    assert json.loads(Path("out.json").read_text(encoding="utf-8"))["tier"]


class NoCredentials:
    def get_paginator(self, name):
        raise NoCredentialsError()

    def download_file(self, bucket, key, path):
        raise NoCredentialsError()

    def head_object(self, **kwargs):
        raise NoCredentialsError()


def test_s3_listing_failure_is_a_clean_error(capsys, monkeypatch):
    monkeypatch.setattr(s3_io, "_client", lambda settings: NoCredentials())
    code = score_videos.main(["s3://bucket/prefix/", "--provider", "offline"])
    assert_clean_failure(capsys, code, "credentials")


def test_s3_download_failure_is_recorded_per_video(capsys, monkeypatch):
    monkeypatch.setattr(s3_io, "_client", lambda settings: NoCredentials())
    code = score_videos.main(["s3://bucket/a.mp4", "--provider", "offline", "-o", "out.json"])
    assert code == 1
    assert "credentials" in json.loads(Path("out.json").read_text(encoding="utf-8"))["error"]


COCO = ["--coco", "coco.json", "--label-studio", "ls.json"]


def test_pre_annotate_missing_scores(capsys):
    code = pre_annotate.main(["--scores", "nope.json", "--provider", "offline", *COCO])
    assert_clean_failure(capsys, code, "nope.json")


@pytest.mark.parametrize("content", ["not json", '["a", "b"]', "{}", "[1]"])
def test_pre_annotate_rejects_files_that_are_not_scores(capsys, content):
    Path("s.json").write_text(content, encoding="utf-8")
    code = pre_annotate.main(["--scores", "s.json", "--provider", "offline", *COCO])
    assert_clean_failure(capsys, code, "s.json")


def test_pre_annotate_threshold_out_of_range(capsys):
    path = write_json("s.json", [scored()])
    code = pre_annotate.main(
        ["--scores", path, "--provider", "offline", "--threshold", "0.3", *COCO]
    )
    assert_clean_failure(capsys, code, "threshold")


def test_pre_annotate_missing_videos_folder(capsys):
    path = write_json("s.json", [scored()])
    code = pre_annotate.main(["--scores", path, "--provider", "offline", "--videos", "nope", *COCO])
    assert_clean_failure(capsys, code, "nope")


def test_pre_annotate_creates_the_raw_output_folder(clips):
    path = write_json("s.json", [scored(str(clips["ideal10"]))])
    args = ["--scores", path, "--provider", "offline", "--raw", "new/dir/raw.json", *COCO]
    assert pre_annotate.main(args) == 0
    assert len(json.loads(Path("new/dir/raw.json").read_text(encoding="utf-8"))) == 1


def test_pre_annotate_does_not_decode_rejected_clips(monkeypatch):
    decoded = []

    class RecordingPool:
        def __init__(self, timeout, sampler):
            pass

        def close(self):
            pass

        def sample(self, path):
            decoded.append(path)

    monkeypatch.setattr(pre_annotate, "DecodePool", RecordingPool)
    path = write_json(
        "s.json", [scored(tier="reject"), {"source": "x.mp4", "error": "scoring failed: x"}]
    )
    assert pre_annotate.main(["--scores", path, "--provider", "offline", *COCO]) == 0
    assert decoded == []


def test_pre_annotate_failure_keeps_earlier_outputs_and_exits_non_zero(capsys):
    Path("coco.json").write_text("earlier result", encoding="utf-8")
    path = write_json("s.json", [scored("missing.mp4")])
    assert pre_annotate.main(["--scores", path, "--provider", "offline", *COCO]) == 1
    assert Path("coco.json").read_text(encoding="utf-8") == "earlier result"
    assert "missing.mp4" in capsys.readouterr().err


def evaluate(scores, truth, *extra):
    return evaluate_pipeline.main(
        ["--scores", scores, "--truth", truth, "--out", "report.json", *extra]
    )


def test_evaluate_missing_files(capsys):
    truth = write_json("truth.json", GROUND_TRUTH)
    assert_clean_failure(capsys, evaluate("nope.json", truth), "nope.json")
    scores = write_json("s.json", [scored()])
    assert_clean_failure(capsys, evaluate(scores, "gone.json"), "gone.json")


@pytest.mark.parametrize("content", ["not json", '["a", "b"]', "{}"])
def test_evaluate_rejects_scores_of_the_wrong_shape(capsys, content):
    Path("s.json").write_text(content, encoding="utf-8")
    truth = write_json("truth.json", GROUND_TRUTH)
    assert_clean_failure(capsys, evaluate("s.json", truth), "s.json")


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '["a"]',
        '{"a.mp4": 5}',
        '{"a.mp4": {"tier": "usable"}}',
        '{"a.mp4": {"scores": {}}}',
    ],
)
def test_evaluate_rejects_truth_of_the_wrong_shape(capsys, content):
    Path("truth.json").write_text(content, encoding="utf-8")
    scores = write_json("s.json", [scored()])
    assert_clean_failure(capsys, evaluate(scores, "truth.json"), "truth.json")


def test_evaluate_names_a_bad_tier_label(capsys):
    Path("labels.csv").write_text("file,tier\na.mp4,Rejected\n", encoding="utf-8")
    scores = write_json("s.json", [scored()])
    assert_clean_failure(capsys, evaluate(scores, "labels.csv"), "Rejected")


def test_evaluate_names_a_bad_score_label(capsys):
    Path("labels.csv").write_text(
        "file,egocentric_framing,tier\na.mp4,yes,usable\n", encoding="utf-8"
    )
    scores = write_json("s.json", [scored()])
    assert_clean_failure(capsys, evaluate(scores, "labels.csv"), "a.mp4")


def test_evaluate_accepts_a_short_csv_row():
    Path("labels.csv").write_text("file,tier,motion_quality\na.mp4,usable\n", encoding="utf-8")
    assert evaluate(write_json("s.json", [scored()]), "labels.csv") == 0


def test_evaluate_with_no_matching_clips(capsys):
    scores = write_json("s.json", [scored("other.mp4")])
    truth = write_json("truth.json", GROUND_TRUTH)
    assert_clean_failure(capsys, evaluate(scores, truth), "no scored videos")


def test_evaluate_rejects_an_unknown_prompt_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        evaluate_pipeline.main(["--videos", "a.mp4", "--truth", "t.json", "--prompt-version", "v9"])
    assert exit_info.value.code == 2


def test_evaluate_creates_output_folders(clips):
    truth = write_json("truth.json", {clips["ideal10"].name: GROUND_TRUTH["a.mp4"]})
    args = ["--videos", str(clips["ideal10"]), "--provider", "offline", "--truth", truth]
    args += ["--out", "a/report.json", "--markdown", "b/report.md", "--scores-out", "c/scores.json"]
    assert evaluate_pipeline.main(args) == 0
    assert all(Path(p).is_file() for p in ("a/report.json", "b/report.md", "c/scores.json"))


def test_make_template_missing_scores(capsys):
    code = evaluate_pipeline.main(["--make-template", "nope.json", "--template-out", "t.csv"])
    assert_clean_failure(capsys, code, "nope.json")


def test_threshold_sweep_counts_cv_disagreements():
    truth = {"a.mp4": {"tier": "usable", "scores": DIMENSIONS}}
    agree = evaluate_pipeline.threshold_sweep([scored(cv_disagreements=[])], truth)
    disagree = evaluate_pipeline.threshold_sweep(
        [scored(cv_disagreements=["motion_quality"])], truth
    )

    def accepted_at(rows):
        return next(r for r in rows if r["composite_min"] == 3.75 and r["confidence_min"] == 0.9)[
            "auto_accepted"
        ]

    assert accepted_at(agree) == 1 and accepted_at(disagree) == 0


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cost_model_creates_its_output_folder():
    result = run_script("scripts/cost_model.py", "--out", "new/dir/cost.csv")
    assert result.returncode == 0, result.stderr
    assert Path("new/dir/cost.csv").is_file()


def test_benchmark_reports_bad_input_cleanly():
    result = run_script("scripts/benchmark_vllm.py", "nope", "--provider", "offline")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr and "nope" in result.stderr


def test_generator_reports_a_bad_output_path():
    Path("taken").write_text("a file", encoding="utf-8")
    result = run_script("tests/generate_synthetic_clips.py", "--out", "taken")
    assert result.returncode != 0
    assert "Traceback" not in result.stderr and "taken" in result.stderr


def test_generator_removes_its_intermediate_file_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_synthetic_clips.shutil, "which", lambda name: None)
    spec = ClipSpec("tiny", 1, _truth(1, 5, 4, 5, 5, 0, 5), "usable", seed=1)
    render(spec, tmp_path / "tiny.mp4")
    assert [p.name for p in tmp_path.glob("tiny*")] == ["tiny.mp4"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_generator_removes_its_intermediate_file_with_ffmpeg(tmp_path):
    spec = ClipSpec("tiny", 1, _truth(1, 5, 4, 5, 5, 0, 5), "usable", seed=1)
    render(spec, tmp_path / "tiny.mp4")
    assert [p.name for p in tmp_path.glob("tiny*")] == ["tiny.mp4"]


def test_missing_file_message_is_short(capsys):
    code = pre_annotate.main(["--scores", "nope.json", "--provider", "offline", *COCO])
    assert code == 2
    assert capsys.readouterr().err.strip() == "error: file not found: nope.json"


def test_summary_line_counts_each_tier(capsys, clips):
    assert score_videos.main([str(clips["tripod10"]), "--provider", "offline", "-o", "o.json"]) == 0
    assert capsys.readouterr().err.endswith("0 usable, 0 borderline, 1 reject\n")


def test_env_example_works_as_it_is_with_no_key(monkeypatch, clips):
    # A copy, because loading .env writes to it. Worker processes need the rest of the environment.
    keyless = {name: value for name, value in os.environ.items() if "API_KEY" not in name}
    monkeypatch.setattr(os, "environ", keyless)
    shutil.copyfile(ROOT / ".env.example", ".env")
    assert get_settings().provider == "offline"
    assert score_videos.main([str(clips["ideal10"]), "-o", "out.json"]) == 0


def test_env_example_has_no_active_provider_settings():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    active = [line for line in lines if line and not line.startswith("#")]
    assert not [line for line in active if line.startswith(("GEMINI", "OPENAI", "VLLM"))]
