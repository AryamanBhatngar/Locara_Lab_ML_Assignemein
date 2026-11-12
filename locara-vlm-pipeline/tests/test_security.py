"""Secrets stay out of logs and output, and hostile inputs stay inside their limits."""

import dataclasses
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import pre_annotate
import score_videos
from src import config
from src.cli import exits_cleanly
from src.config import Settings, get_settings
from src.schemas.scoring_schema import video_id_for
from src.scorer import s3_io
from src.scorer.dedup import HashRegistry
from src.scorer.frame_sampler import VideoReadError, sample_video
from src.scorer.quality_scorer import QualityScorer, VideoInput, resolve_inputs
from src.scorer.vlm_client import OpenAIClient, VLLMClient, VLMError

FAKE_KEY = "sk-" + "fakeSECRET" * 3  # built here so secret scanners do not flag a fake key
FAKE_GOOGLE_KEY = "AIza" + "SyA1234567890abcdefghijklmnop"
ERROR_PRIVILEGE_NOT_HELD = 1314


class Leaky(OpenAIClient):
    """A provider whose error message echoes the API key, as a 401 can."""

    def __init__(self):
        self.settings = Settings(max_retries=1)
        self.model = "x"

    def _call(self, prompt, video):
        raise RuntimeError(f"401 Incorrect API key provided: {FAKE_KEY}")


@pytest.fixture
def fake_openai(monkeypatch):
    import openai

    seen = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: seen.update(kwargs) or SimpleNamespace())
    return seen


def test_vllm_server_never_receives_the_openai_key(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    VLLMClient(Settings(provider="vllm", vllm_base_url="http://other-host:8000/v1"))
    assert fake_openai["api_key"] == "EMPTY"


def test_vllm_server_gets_its_own_key(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("VLLM_API_KEY", "vllm-token")
    VLLMClient(Settings(provider="vllm"))
    assert fake_openai["api_key"] == "vllm-token"


def test_openai_client_still_uses_the_openai_key(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    OpenAIClient(Settings(provider="openai"))
    assert fake_openai["api_key"] == FAKE_KEY


def test_settings_never_hold_a_credential():
    assert not [
        f.name for f in dataclasses.fields(Settings) if "key" in f.name or "secret" in f.name
    ]


@pytest.mark.parametrize(
    "text,secret",
    [
        (f"bad key {FAKE_KEY}", FAKE_KEY),
        ("key sk-abc***************xyz was rejected", "sk-abc"),
        (f"GET https://host/v1?key={FAKE_GOOGLE_KEY}", "AIzaSy"),
        ("GET https://host/v1?api_key=plainvalue&x=1", "plainvalue"),
        ("cannot reach http://admin:hunter2@vllm:8000/v1", "hunter2"),
    ],
)
def test_redact_removes_credentials(text, secret):
    from src.redact import redact

    cleaned = redact(text)
    assert secret not in cleaned and "[redacted]" in cleaned


def test_redact_removes_values_of_known_variables(monkeypatch):
    from src.redact import redact

    monkeypatch.setenv("GEMINI_API_KEY", "an-unusual-looking-value")
    assert redact("failed with an-unusual-looking-value") == "failed with [redacted]"


def test_redact_leaves_ordinary_messages_alone():
    from src.redact import redact

    text = "s3://bucket/clip.mp4: 800 MB is over the 500 MB limit"
    assert redact(text) == text


@pytest.mark.parametrize("key_in_env", [True, False])
def test_provider_error_stays_out_of_logs_and_output(monkeypatch, caplog, clips, key_in_env):
    monkeypatch.setattr("time.sleep", lambda s: None)
    if key_in_env:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = str(clips["ideal10"])
    with caplog.at_level("WARNING"):
        out = QualityScorer(Settings(), client=Leaky()).score_batch(
            [VideoInput(path, path, video_id_for("a"))]
        )
    assert "scoring failed" in out[0]["error"]
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in json.dumps(out)


def test_retry_error_does_not_chain_the_raw_exception(monkeypatch, clips):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(VLMError) as info:
        Leaky().score(sample_video(clips["ideal10"]))
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None


def test_cli_error_message_is_redacted(capsys):
    @exits_cleanly
    def main(argv):
        raise ValueError(f"connection failed for {FAKE_KEY}")

    assert main(None) == 2
    assert FAKE_KEY not in capsys.readouterr().err


class FakeS3:
    def __init__(self, size=1000, body=b"video bytes"):
        self.size, self.body, self.downloads = size, body, []

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.size, "ETag": '"abc123"'}

    def download_file(self, bucket, key, path):
        self.downloads.append(key)
        Path(path).write_bytes(self.body)


@pytest.fixture
def s3(monkeypatch, tmp_path):
    fake = FakeS3()
    monkeypatch.setattr(s3_io, "_client", lambda settings: fake)
    fake.settings = Settings(cache_dir=tmp_path / "cache", s3_max_download_mb=1)
    return fake


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/../../escaped.mp4",
        "s3://bucket/a/../../../escaped.mp4",
        "s3://bucket/..",
        pytest.param(
            "s3://bucket/..\\..\\escaped.mp4",
            marks=pytest.mark.skipif(os.name != "nt", reason="Windows separators"),
        ),
        pytest.param(
            "s3://bucket/C:/escaped.mp4",
            marks=pytest.mark.skipif(os.name != "nt", reason="Windows drive letters"),
        ),
    ],
)
def test_s3_key_cannot_leave_the_cache(s3, tmp_path, uri):
    with pytest.raises(s3_io.S3Error, match="unsafe"):
        s3_io.download(uri, s3.settings)
    assert s3.downloads == []
    assert not list(tmp_path.rglob("escaped*"))


@pytest.mark.parametrize(
    "uri", ["s3://../x.mp4", "s3://./x.mp4", "s3://a/b\\..", "s3://-bad/x.mp4"]
)
def test_unsafe_bucket_or_key_is_rejected(s3, uri):
    with pytest.raises((ValueError, OSError)):
        s3_io.download(uri, s3.settings)
    assert s3.downloads == []


def test_nested_s3_key_is_cached_under_its_bucket(s3, tmp_path):
    path = s3_io.download("s3://bucket/2026/11/clip.mp4", s3.settings)
    assert path.is_relative_to(tmp_path / "cache" / "bucket")
    assert path.parts[-3:] == ("2026", "11", "clip.mp4")
    assert path.read_bytes() == b"video bytes"


def test_unsafe_key_in_a_batch_only_fails_that_video(s3):
    videos = resolve_inputs(["s3://bucket/../../x.mp4", "s3://bucket/ok.mp4"], s3.settings)
    assert "unsafe" in videos[0].error
    assert videos[1].error is None


def test_s3_object_over_the_limit_is_not_downloaded(s3):
    s3.size = 1_000_001
    with pytest.raises(s3_io.S3Error, match="S3_MAX_DOWNLOAD_MB"):
        s3_io.download("s3://bucket/big.mp4", s3.settings)
    assert s3.downloads == []


def test_s3_object_at_the_limit_is_downloaded(s3):
    s3.size = 1_000_000
    assert s3_io.download("s3://bucket/ok.mp4", s3.settings).is_file()


def test_object_that_grew_after_the_size_check_is_discarded(s3, tmp_path):
    s3.body = b"x" * 1_000_001
    with pytest.raises(s3_io.S3Error, match="S3_MAX_DOWNLOAD_MB"):
        s3_io.download("s3://bucket/grew.mp4", s3.settings)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_oversized_file_already_in_the_cache_is_not_used(s3, tmp_path):
    cached = tmp_path / "cache" / "bucket" / "old.mp4"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x" * 1_000_001)
    s3.size = 2_000_000
    with pytest.raises(s3_io.S3Error, match="S3_MAX_DOWNLOAD_MB"):
        s3_io.download("s3://bucket/old.mp4", s3.settings)


def test_download_limit_defaults_to_500_mb_and_is_configurable(monkeypatch):
    monkeypatch.delenv("S3_MAX_DOWNLOAD_MB", raising=False)
    assert get_settings("offline").s3_max_download_mb == 500
    monkeypatch.setenv("S3_MAX_DOWNLOAD_MB", "120")
    assert get_settings("offline").s3_max_download_mb == 120


def test_failed_download_leaves_no_temp_file(s3, tmp_path):
    def fail_midway(bucket, key, path):
        Path(path).write_bytes(b"partial")
        raise OSError("connection reset")

    s3.download_file = fail_midway
    with pytest.raises(OSError, match="connection reset"):
        s3_io.download("s3://bucket/dir/clip.mp4", s3.settings)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_finished_download_leaves_no_temp_file(s3):
    path = s3_io.download("s3://bucket/dir/clip.mp4", s3.settings)
    assert [p.name for p in path.parent.iterdir()] == ["clip.mp4"]


def test_failed_registry_save_leaves_no_temp_file(monkeypatch, tmp_path):
    registry = HashRegistry(tmp_path / "registry.json")
    registry.add("a", [1])

    def fail(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        registry.save()
    assert list(tmp_path.iterdir()) == []


def test_playlist_cannot_reach_other_files_through_the_decoder(tmp_path, clips):
    playlist = tmp_path / "local.m3u8"
    playlist.write_text(
        f"#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\n{clips['ideal10'].as_posix()}\n#EXT-X-ENDLIST\n",
        encoding="utf-8",
    )
    with pytest.raises(VideoReadError, match="video type"):
        sample_video(playlist)


def test_uppercase_extension_is_accepted(tmp_path, clips):
    upper = tmp_path / "CLIP.MP4"
    upper.write_bytes(clips["ideal10"].read_bytes())
    assert sample_video(upper).frame_count == config.MIN_FRAMES


def test_oversized_frames_are_refused(monkeypatch, clips):
    monkeypatch.setattr(config, "MAX_FRAME_PIXELS", 1000, raising=False)
    with pytest.raises(VideoReadError, match="pixels"):
        sample_video(clips["ideal10"])


def test_decoding_stops_at_the_frame_limit(monkeypatch, clips):
    monkeypatch.setattr(config, "MAX_DECODED_FRAMES", 50, raising=False)
    with pytest.raises(VideoReadError, match="frames"):
        sample_video(clips["ideal10"])


def test_output_paths_use_forward_slashes(clips):
    assert "\\" not in resolve_inputs([str(clips["ideal10"])], Settings())[0].source


def scored(source):
    scores = {k: {"score": 4, "confidence": 0.9, "reasoning": "ok"} for k in ("motion_quality",)}
    return {
        "video_id": video_id_for("a"),
        "tier": "usable",
        "scores": scores,
        "meta": {"source": source},
    }


def test_videos_folder_uses_the_file_name_only(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    path = pre_annotate.video_path("../../outside/clip.mp4", str(folder))
    assert path == folder.resolve() / "clip.mp4"


@pytest.mark.parametrize("source", ["..", "a/..", ""])
def test_videos_folder_refuses_names_that_leave_it(tmp_path, source):
    with pytest.raises(ValueError, match="--videos"):
        pre_annotate.video_path(source, str(tmp_path))


def symlink_or_skip(link, target):
    """Only Windows without Developer Mode may skip. Anywhere else, a failure is a real failure."""
    try:
        link.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == ERROR_PRIVILEGE_NOT_HELD:
            pytest.skip("creating symlinks on Windows needs Developer Mode or admin rights")
        raise


def test_a_symlink_failure_is_only_skipped_for_the_windows_privilege_error(monkeypatch, tmp_path):
    def refuse(self, target):
        raise PermissionError(errno.EPERM, "not allowed")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    with pytest.raises(PermissionError):
        symlink_or_skip(tmp_path / "link", tmp_path / "target")


def test_videos_folder_refuses_a_path_that_resolves_outside_it(monkeypatch, tmp_path):
    """The same rule as the symlink test below, without needing the right to create a link."""
    folder = tmp_path / "videos"
    folder.mkdir()
    real_resolve = Path.resolve

    def resolve(self, strict=False):
        if self.name == "link.mp4":
            return tmp_path / "outside.mp4"
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="--videos"):
        pre_annotate.video_path("link.mp4", str(folder))


def test_videos_folder_refuses_symlinks_that_leave_it(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    symlink_or_skip(folder / "link.mp4", outside)
    with pytest.raises(ValueError, match="--videos"):
        pre_annotate.video_path("link.mp4", str(folder))


def test_pre_annotate_reads_s3_sources_through_the_cache(monkeypatch, tmp_path, clips):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(s3_io, "fetch", lambda uri, settings: (clips["ideal10"], False))
    Path("s.json").write_text(json.dumps([scored("s3://bucket/ideal10.mp4")]), encoding="utf-8")
    args = [
        "--scores",
        "s.json",
        "--provider",
        "offline",
        "--coco",
        "c.json",
        "--label-studio",
        "l.json",
    ]
    assert pre_annotate.main(args) == 0
    assert len(json.loads(Path("c.json").read_text(encoding="utf-8"))["videos"]) == 1


def test_output_json_contains_no_credentials(monkeypatch, tmp_path, clips):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert (
        score_videos.main([str(clips["ideal10"]), "--provider", "offline", "-o", "out.json"]) == 0
    )
    assert FAKE_KEY not in Path("out.json").read_text(encoding="utf-8")
