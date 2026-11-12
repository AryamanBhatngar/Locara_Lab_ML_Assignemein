"""Provider adapters, tested against fake SDK clients (no network, no keys)."""

import json
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.scorer.frame_sampler import sample_video
from src.scorer.vlm_client import GeminiClient, OpenAIClient, VLMError, make_client

REPLY = {
    k: {"score": 4, "confidence": 0.8, "reasoning": "ok"}
    for k in (
        "egocentric_framing",
        "motion_quality",
        "scene_diversity",
        "hand_visibility",
        "lighting_exposure",
        "duplicate_risk",
        "overall_usability",
    )
}
REPLY["egocentric_framing"]["score"] = 1
REPLY["duplicate_risk"]["score"] = 0


def test_gemini_request_shape(monkeypatch, clips):
    calls = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.update(model=model, contents=contents, config=config)
            return SimpleNamespace(
                text=json.dumps(REPLY),
                usage_metadata=SimpleNamespace(prompt_token_count=4100, candidates_token_count=420),
            )

    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=FakeModels()))
    c = GeminiClient(Settings(provider="gemini"), api_key="test")
    video = sample_video(clips["ideal10"])
    raw, usage = c.score(video)
    assert raw == REPLY and usage["input_tokens"] == 4100
    assert calls["model"] == "gemini-2.5-flash"
    assert len(calls["contents"]) == video.frame_count + 1  # frames then prompt
    assert isinstance(calls["contents"][-1], str)
    assert calls["config"].response_mime_type == "application/json"
    assert calls["config"].temperature == 0.0


def test_openai_request_shape_and_json_mode_fallback(monkeypatch, clips):
    seen = []

    class FakeCompletions:
        def create(self, **kw):
            seen.append(kw)
            if "response_format" in kw:
                raise RuntimeError("response_format is not supported")
            msg = SimpleNamespace(content="```json\n" + json.dumps(REPLY) + "\n```")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg)],
                usage=SimpleNamespace(prompt_tokens=3300, completion_tokens=400),
            )

    import openai

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    c = OpenAIClient(Settings(provider="openai", max_retries=0), api_key="test")
    raw, usage = c.score(sample_video(clips["ideal10"]))
    assert raw == REPLY and usage["output_tokens"] == 400
    content = seen[-1]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in content[1:])
    assert "response_format" not in seen[-1]


def test_missing_keys_fail_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(VLMError, match="GEMINI_API_KEY"):
        GeminiClient(Settings(provider="gemini"))
    with pytest.raises(VLMError, match="OPENAI_API_KEY"):
        OpenAIClient(Settings(provider="openai"))


def test_offline_client_is_default():
    assert make_client(Settings()).provider == "offline"


def test_retries_then_raises(monkeypatch, clips):
    class Flaky(OpenAIClient):
        def __init__(self):
            self.settings = Settings(max_retries=1)
            self.model = "x"

        def _call(self, prompt, video):
            raise TimeoutError("slow")

    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(VLMError, match="after retries"):
        Flaky().score(sample_video(clips["ideal10"]))
