"""Each kind of provider failure gets the right retries, and a rejected key stops the batch.

The exceptions are the real SDK classes, built the way the SDKs build them.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
import pytest
from google import genai
from google.genai import errors as genai_errors

import evaluate_pipeline
import pre_annotate
import score_videos
from src import config
from src.config import Settings
from src.schemas.scoring_schema import video_id_for
from src.scorer import quality_scorer, vlm_client
from src.scorer.frame_sampler import sample_video
from src.scorer.quality_scorer import QualityScorer, VideoInput
from src.scorer.vlm_client import (
    BaseVLMClient,
    GeminiClient,
    OpenAIClient,
    VLLMClient,
    VLMError,
)

KEY = "sk-" + "fakeSECRET" * 3  # built here so secret scanners do not flag a fake key
SKIPPED = "skipped: authentication failed"
REQUEST = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
GOOD = {
    name: {"score": 4, "confidence": 0.9, "reasoning": "ok"}
    for name in ("motion_quality", "scene_diversity", "hand_visibility", "lighting_exposure")
}
GOOD["overall_usability"] = {"score": 4, "confidence": 0.9, "reasoning": "ok"}
GOOD["egocentric_framing"] = {"score": 1, "confidence": 0.9, "reasoning": "ok"}
GOOD["duplicate_risk"] = {"score": 0, "confidence": 0.9, "reasoning": "ok"}


def openai_error(cls, status, message="request failed", headers=None):
    response = httpx2.Response(status, headers=headers, request=REQUEST)
    return cls(message, response=response, body=None)


def gemini_error(cls, code, status, message="request failed", details=(), headers=None):
    body = {"error": {"code": code, "status": status, "message": message, "details": list(details)}}
    return cls(code, body, httpx2.Response(code, headers=headers, request=REQUEST))


def key_rejected_by_openai():
    return openai_error(openai.AuthenticationError, 401, f"Incorrect API key provided: {KEY}")


def key_rejected_by_gemini():
    reason = {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}
    message = f"API key not valid: {KEY}"
    return gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", message, [reason])


@pytest.mark.parametrize(
    "error,kind",
    [
        (lambda: key_rejected_by_openai(), "AUTH"),
        (lambda: openai_error(openai.PermissionDeniedError, 403), "AUTH"),
        (lambda: openai_error(openai.RateLimitError, 429), "RATE_LIMIT"),
        (lambda: openai_error(openai.InternalServerError, 500), "TRANSIENT"),
        (lambda: openai_error(openai.InternalServerError, 503), "TRANSIENT"),
        (lambda: openai.APITimeoutError(REQUEST), "TRANSIENT"),
        (lambda: openai.APIConnectionError(request=REQUEST), "TRANSIENT"),
        (lambda: openai_error(openai.BadRequestError, 400, "image too large"), "TRANSIENT"),
        (lambda: key_rejected_by_gemini(), "AUTH"),
        (lambda: gemini_error(genai_errors.ClientError, 401, "UNAUTHENTICATED"), "AUTH"),
        (lambda: gemini_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"), "AUTH"),
        (lambda: gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "RATE_LIMIT"),
        (lambda: gemini_error(genai_errors.ServerError, 500, "INTERNAL"), "TRANSIENT"),
        (lambda: gemini_error(genai_errors.ServerError, 503, "UNAVAILABLE"), "TRANSIENT"),
        (lambda: gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT"), "TRANSIENT"),
        (lambda: TimeoutError("slow"), "TRANSIENT"),
        (lambda: vlm_client.BadResponseError("not json"), "BAD_RESPONSE"),
    ],
)
def test_errors_are_classified(error, kind):
    assert vlm_client.classify_error(error()).name == kind


def test_invalid_json_from_the_parser_counts_as_a_bad_response():
    with pytest.raises(vlm_client.BadResponseError):
        vlm_client.parse_json_object("no json here")


class Scripted(BaseVLMClient):
    """Returns or raises the given outcomes in turn, repeating the last one."""

    provider = "gemini"
    key_name = "GEMINI_API_KEY"

    def __init__(self, *outcomes, max_retries=3):
        super().__init__(Settings(max_retries=max_retries))
        self.model = "stub-model"
        self.outcomes = list(outcomes)
        self.calls = 0
        self._lock = threading.Lock()

    def _call(self, prompt, video):
        with self._lock:
            self.calls += 1
            outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return json.loads(json.dumps(outcome)), {}


@pytest.fixture(scope="module")
def sampled(clips):
    return sample_video(clips["ideal10"])


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.mark.parametrize("make_error", [key_rejected_by_openai, key_rejected_by_gemini])
def test_auth_failure_is_never_retried(sampled, sleeps, make_error):
    client = Scripted(make_error())
    with pytest.raises(vlm_client.AuthError) as info:
        client.score(sampled)
    assert client.calls == 1 and sleeps == []
    assert "GEMINI_API_KEY" in str(info.value)
    assert KEY not in str(info.value)
    assert info.value.__cause__ is None


def test_rate_limit_is_retried(sampled, sleeps):
    client = Scripted(openai_error(openai.RateLimitError, 429), GOOD)
    assert client.score(sampled)[0] == GOOD
    assert client.calls == 2 and len(sleeps) == 1


def gemini_retry_info(seconds):
    info = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": seconds}
    return gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", details=[info])


def http_date_in(seconds):
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds), usegmt=True)


@pytest.mark.parametrize(
    "error,wait",
    [
        (lambda: openai_error(openai.RateLimitError, 429, headers={"Retry-After": "7"}), 7),
        (lambda: openai_error(openai.RateLimitError, 429, headers={"retry-after-ms": "7500"}), 7.5),
        (
            lambda: openai_error(
                openai.RateLimitError, 429, headers={"Retry-After": http_date_in(30)}
            ),
            25,
        ),
        (
            lambda: gemini_error(
                genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", headers={"Retry-After": "9"}
            ),
            9,
        ),
        (lambda: gemini_retry_info("34s"), 34),
        (lambda: gemini_retry_info("1.5s"), 1.5),
    ],
)
def test_rate_limit_waits_as_long_as_the_server_asks(sampled, sleeps, error, wait):
    client = Scripted(error(), GOOD)
    client.score(sampled)
    assert sleeps[0] >= wait


def test_retry_after_is_capped(sampled, sleeps):
    error = openai_error(openai.RateLimitError, 429, headers={"Retry-After": "99999"})
    Scripted(error, GOOD).score(sampled)
    assert sleeps[0] <= config.MAX_RETRY_AFTER_S


@pytest.mark.parametrize("value", ["soon", "-5", ""])
def test_unusable_retry_after_falls_back_to_backoff(sampled, sleeps, value):
    error = openai_error(openai.RateLimitError, 429, headers={"Retry-After": value})
    Scripted(error, GOOD).score(sampled)
    assert 0 < sleeps[0] < 3


def test_rate_limit_gives_up_after_the_retry_limit(sampled, sleeps):
    client = Scripted(openai_error(openai.RateLimitError, 429), max_retries=2)
    with pytest.raises(VLMError, match="after retries"):
        client.score(sampled)
    assert client.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        lambda: openai_error(openai.InternalServerError, 503),
        lambda: gemini_error(genai_errors.ServerError, 500, "INTERNAL"),
        lambda: TimeoutError("slow"),
    ],
)
def test_timeouts_and_server_errors_are_retried(sampled, sleeps, error):
    client = Scripted(error(), GOOD)
    assert client.score(sampled)[0] == GOOD
    assert client.calls == 2


def fake_openai_replies(monkeypatch, *contents):
    replies = list(contents)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies.pop(0) if len(replies) > 1 else replies[0]
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        message = SimpleNamespace(content=content)
        choices = [] if content is None else [SimpleNamespace(message=message)]
        return SimpleNamespace(choices=choices, usage=usage)

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    return calls


def test_bad_json_gets_exactly_one_retry(monkeypatch, sampled, sleeps):
    calls = fake_openai_replies(monkeypatch, "not json at all")
    client = OpenAIClient(Settings(provider="openai", max_retries=3), api_key="test")
    with pytest.raises(VLMError):
        client.score(sampled)
    assert len(calls) == 2


def test_bad_json_can_recover_on_the_retry(monkeypatch, sampled, sleeps):
    calls = fake_openai_replies(monkeypatch, "not json at all", json.dumps(GOOD))
    client = OpenAIClient(Settings(provider="openai", max_retries=3), api_key="test")
    assert client.score(sampled)[0] == GOOD
    assert len(calls) == 2


def test_a_reply_with_no_choices_is_a_bad_response(monkeypatch, sampled, sleeps):
    calls = fake_openai_replies(monkeypatch, None)
    client = OpenAIClient(Settings(provider="openai", max_retries=3), api_key="test")
    with pytest.raises(VLMError):
        client.score(sampled)
    assert len(calls) == 2


def test_no_retries_means_none_for_bad_json_too(sampled, sleeps):
    client = Scripted(vlm_client.BadResponseError("not json"), max_retries=0)
    with pytest.raises(VLMError):
        client.score(sampled)
    assert client.calls == 1


def batch_of(clips, count, key="ideal10"):
    path = str(clips[key])
    return [VideoInput(path, path, video_id_for(f"copy{i}")) for i in range(count)]


@pytest.mark.parametrize("workers", [1, 4])
def test_batch_of_10_makes_one_auth_call_then_stops(clips, sleeps, workers):
    client = Scripted(key_rejected_by_openai())
    scorer = QualityScorer(Settings(), client=client, workers=workers)
    results = scorer.score_batch(batch_of(clips, 10))

    assert client.calls == 1
    errors = [r["error"] for r in results]
    assert len(errors) == 10
    assert errors.count(SKIPPED) == 9
    assert sum(e.startswith("scoring failed") for e in errors) == 1
    assert "GEMINI_API_KEY" in scorer.auth_failure


def test_later_chunks_are_not_even_sampled(monkeypatch, clips, sleeps):
    monkeypatch.setattr(config, "MAX_VIDEOS_IN_FLIGHT", 2)
    decoded = []

    def recording_sample(path):
        decoded.append(path)
        return sample_video(path)

    monkeypatch.setattr(quality_scorer, "sample_video", recording_sample)
    client = Scripted(key_rejected_by_openai())
    results = QualityScorer(Settings(), client=client, workers=1).score_batch(batch_of(clips, 6))
    assert len(decoded) <= 2
    assert [r["error"] for r in results].count(SKIPPED) >= 4


def test_key_revoked_part_way_stops_the_rest(clips, sleeps):
    client = Scripted(GOOD, key_rejected_by_openai())
    results = QualityScorer(Settings(), client=client, workers=1).score_batch(batch_of(clips, 4))
    assert client.calls == 2
    assert "tier" in results[0]
    assert results[1]["error"].startswith("scoring failed")
    assert [r["error"] for r in results[2:]] == [SKIPPED, SKIPPED]


def test_model_calls_run_in_parallel_once_a_call_has_worked(clips):
    both_running = threading.Barrier(2, timeout=20)

    class NeedsCompany(Scripted):
        def _call(self, prompt, video):
            with self._lock:
                self.calls += 1
                first = self.calls == 1
            if not first:
                both_running.wait()
            return json.loads(json.dumps(GOOD)), {}

    client = NeedsCompany(GOOD)
    results = QualityScorer(Settings(), client=client, workers=4).score_batch(batch_of(clips, 5))
    assert all("tier" in r for r in results)


def test_clip_that_could_not_be_read_keeps_its_own_error(clips, tmp_path, sleeps):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    videos = [VideoInput(str(broken), str(broken), video_id_for("broken")), *batch_of(clips, 2)]
    client = Scripted(key_rejected_by_openai())
    results = QualityScorer(Settings(), client=client, workers=1).score_batch(videos)
    assert results[0]["error"].startswith("sampling failed")


def test_other_failures_do_not_stop_the_batch(clips, sleeps):
    client = Scripted(openai_error(openai.InternalServerError, 500), max_retries=0)
    scorer = QualityScorer(Settings(), client=client, workers=1)
    results = scorer.score_batch(batch_of(clips, 3))
    assert client.calls == 3
    assert all(r["error"].startswith("scoring failed") for r in results)
    assert scorer.auth_failure is None


def fake_gemini(monkeypatch, error):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        raise error

    models = SimpleNamespace(generate_content=generate_content)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=models))
    return calls


def test_gemini_client_stops_on_its_real_invalid_key_error(monkeypatch, clips, sleeps):
    calls = fake_gemini(monkeypatch, key_rejected_by_gemini())
    client = GeminiClient(Settings(provider="gemini"), api_key=KEY)
    QualityScorer(Settings(), client=client, workers=4).score_batch(batch_of(clips, 10))
    assert len(calls) == 1


@pytest.mark.parametrize("variable", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_message_names_the_gemini_key_in_use(monkeypatch, clips, sleeps, variable):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv(variable, KEY)
    fake_gemini(monkeypatch, key_rejected_by_gemini())
    client = GeminiClient(Settings(provider="gemini"))
    scorer = QualityScorer(Settings(), client=client)
    scorer.score_batch(batch_of(clips, 1))
    assert variable in scorer.auth_failure


def test_message_names_the_key_for_each_openai_style_client(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace())
    assert OpenAIClient(Settings(provider="openai"), api_key="x").key_name == "OPENAI_API_KEY"
    assert VLLMClient(Settings(provider="vllm")).key_name == "VLLM_API_KEY"


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    """An empty folder, so no real .env is read, and a fake key in the environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    return tmp_path


def test_cli_stops_and_says_which_key_to_check(monkeypatch, workdir, clips, capsys, sleeps):
    calls = fake_gemini(monkeypatch, key_rejected_by_gemini())
    inputs = [str(clips[k]) for k in ("ideal10", "other10", "tripod10", "dark10")]
    code = score_videos.main([*inputs, "--provider", "gemini", "-o", "out.json"])
    err = capsys.readouterr().err

    assert code == 2
    assert len(calls) == 1
    assert err.strip().splitlines()[-1].startswith("error: ")
    assert "GEMINI_API_KEY" in err and KEY not in err

    written = Path("out.json").read_text(encoding="utf-8")
    assert KEY not in written
    assert [r["error"] for r in json.loads(written)].count(SKIPPED) == 3


def scored_record(source):
    return {"video_id": video_id_for(source), "tier": "usable", "meta": {"source": source}}


def test_pre_annotate_stops_at_the_first_rejected_key(monkeypatch, workdir, clips, capsys, sleeps):
    calls = fake_gemini(monkeypatch, key_rejected_by_gemini())
    records = [scored_record(str(clips[k])) for k in ("ideal10", "other10", "tripod10")]
    Path("s.json").write_text(json.dumps(records), encoding="utf-8")
    args = [
        "--scores",
        "s.json",
        "--provider",
        "gemini",
        "--coco",
        "c.json",
        "--label-studio",
        "l.json",
    ]
    code = pre_annotate.main(args)
    err = capsys.readouterr().err

    assert code == 2
    assert len(calls) == 1
    assert "GEMINI_API_KEY" in err and KEY not in err


def test_evaluate_reports_a_rejected_key_instead_of_empty_results(
    monkeypatch, workdir, clips, capsys, sleeps
):
    fake_gemini(monkeypatch, key_rejected_by_gemini())
    truth = {clips["ideal10"].name: {"tier": "usable", "scores": {}}}
    Path("t.json").write_text(json.dumps(truth), encoding="utf-8")
    args = ["--videos", str(clips["ideal10"]), "--provider", "gemini", "--truth", "t.json"]
    assert evaluate_pipeline.main([*args, "--out", "r.json"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        lambda: key_rejected_by_openai(),
        lambda: key_rejected_by_gemini(),
        lambda: openai_error(openai.RateLimitError, 429, f"limit hit for {KEY}"),
        lambda: gemini_error(genai_errors.ServerError, 500, "INTERNAL", f"failed for {KEY}"),
        lambda: openai_error(openai.BadRequestError, 400, f"rejected {KEY}"),
    ],
)
def test_the_key_never_reaches_logs_or_output(monkeypatch, clips, sleeps, caplog, error):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = Scripted(error(), max_retries=1)
    scorer = QualityScorer(Settings(), client=client, workers=2)
    with caplog.at_level("INFO"):
        results = scorer.score_batch(batch_of(clips, 3))
    assert KEY not in caplog.text
    assert KEY not in json.dumps(results)
    assert KEY not in str(scorer.auth_failure)


def test_a_bad_reply_proves_the_key_works_so_calls_go_parallel(clips, sleeps):
    both_running = threading.Barrier(2, timeout=5)

    class BadThenNeedsCompany(Scripted):
        def _call(self, prompt, video):
            with self._lock:
                self.calls += 1
                first_clip = self.calls <= 2  # the first attempt and its one retry
            if first_clip:
                raise vlm_client.BadResponseError("not json")
            both_running.wait()
            return json.loads(json.dumps(GOOD)), {}

    client = BadThenNeedsCompany(GOOD)
    results = QualityScorer(Settings(), client=client, workers=4).score_batch(batch_of(clips, 5))
    assert sum("tier" in r for r in results) == 4
