"""VLM adapters (Q2).

All providers implement the same two calls, `score()` and `annotate()`, returning a raw dict
that the orchestrator validates with the Pydantic schemas.

- GeminiClient   Google Gemini through the google-genai SDK (default gemini-2.5-flash).
- OpenAIClient   OpenAI chat completions with image inputs (default gpt-4o).
- VLLMClient     Any OpenAI-compatible server, e.g. vLLM serving Qwen2.5-VL-7B-Instruct.
- OfflineClient  NOT a VLM. Deterministic scores from the OpenCV telemetry only. Used for
                 tests, CI and as a sanity baseline; its output is always labelled
                 provider="offline".
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable
from typing import Any

from .. import config
from ..config import Settings
from ..redact import redact
from . import prompts
from .frame_sampler import SampledVideo, telemetry_summary
from .provider_errors import (
    AuthError,
    BadResponseError,
    ErrorKind,
    VLMError,
    classify_error,
    http_status,
    retry_after_seconds,
)

log = logging.getLogger(__name__)


Reply = tuple[dict[str, Any], dict[str, Any]]  # parsed model output, usage


def _reject_constant(name: str) -> float:
    raise BadResponseError(f"non-finite number in response: {name}")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _first_object(text: str) -> dict[str, Any]:
    """Try each opening brace in turn. The decoder knows strings, so braces in them are safe."""
    for match in re.finditer(r"\{", text):
        try:
            return _DECODER.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    raise BadResponseError(f"no JSON object in response: {text[:120]!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (tolerates code fences and prose)."""
    if not text:
        raise BadResponseError("empty model response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        obj = _DECODER.decode(cleaned)
    except json.JSONDecodeError:
        obj = _first_object(cleaned)
    if not isinstance(obj, dict):
        raise BadResponseError("model response is not a JSON object")
    return obj


class BaseVLMClient:
    provider = "base"
    key_name = "the API key"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.model

    def score(
        self,
        video: SampledVideo,
        prompt_version: str = prompts.PROMPT_VERSION,
        duplicate_note: str = "",
    ) -> Reply:
        prompt = prompts.build_scoring_prompt(
            prompt_version,
            telemetry_summary(video.telemetry),
            video.duration_sec,
            video.frame_count,
            duplicate_note,
        )
        return self._with_retries(lambda: self._call(prompt, video))

    def annotate(self, video: SampledVideo) -> Reply:
        prompt = prompts.build_annotation_prompt(
            video.duration_sec, [f.time_sec for f in video.frames]
        )
        return self._with_retries(lambda: self._call(prompt, video))

    def _call(self, prompt: str, video: SampledVideo) -> Reply:
        raise NotImplementedError

    def _auth_message(self, exc: Exception) -> str:
        return (
            f"{self.provider} rejected the request (HTTP {http_status(exc)}): check {self.key_name}"
        )

    def _retry_limit(self, kind: ErrorKind) -> int:
        if kind is ErrorKind.BAD_RESPONSE:
            return min(self.settings.max_retries, config.BAD_RESPONSE_RETRIES)
        return self.settings.max_retries

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        jitter = 0.8 + 0.4 * random.random()  # nosec B311 - spreads retries, not security
        delay = min(8.0, 2**attempt) * jitter
        asked_for = retry_after_seconds(exc)
        return delay if asked_for is None else max(delay, min(asked_for, config.MAX_RETRY_AFTER_S))

    def _with_retries(self, fn: Callable[[], Reply]) -> Reply:
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                kind = classify_error(exc)
                if kind is ErrorKind.AUTH:
                    raise AuthError(self._auth_message(exc)) from None
                if attempt >= self._retry_limit(kind):
                    # "from None": a chained SDK exception could print unredacted text.
                    failure = BadResponseError if kind is ErrorKind.BAD_RESPONSE else VLMError
                    raise failure(
                        f"{self.provider} call failed after retries: {redact(exc)}"
                    ) from None
                delay = self._retry_delay(exc, attempt)
                attempt += 1
                log.warning("vlm call failed (%s), retry %d in %.1fs", redact(exc), attempt, delay)
                time.sleep(delay)


def _uses_google_key(api_key: str | None) -> bool:
    return not api_key and not os.environ.get("GEMINI_API_KEY")


class GeminiClient(BaseVLMClient):
    provider = "gemini"

    def __init__(self, settings: Settings, api_key: str | None = None):
        super().__init__(settings)
        from google import genai  # lazy: only needed for this provider
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise VLMError("GEMINI_API_KEY is not set (see .env.example)")
        self.key_name = "GOOGLE_API_KEY" if _uses_google_key(api_key) else "GEMINI_API_KEY"
        self._types = types
        timeout_ms = round(settings.request_timeout_s * 1000)
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_ms))

    def _call(self, prompt: str, video: SampledVideo) -> Reply:
        t = self._types
        parts = [t.Part.from_bytes(data=f.jpeg, mime_type="image/jpeg") for f in video.frames]
        cfg = t.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            thinking_config=t.ThinkingConfig(thinking_budget=0),
        )
        start = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model, contents=[*parts, prompt], config=cfg
        )
        latency = (time.perf_counter() - start) * 1000
        um = resp.usage_metadata
        usage = {
            "latency_ms": round(latency, 1),
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
        }
        return parse_json_object(resp.text or ""), usage


class OpenAIClient(BaseVLMClient):
    provider = "openai"
    key_name = "OPENAI_API_KEY"
    detail = "high"

    def __init__(self, settings: Settings, api_key: str | None = None, base_url: str | None = None):
        super().__init__(settings)
        from openai import OpenAI

        # A self-hosted server must never be sent the real OpenAI key.
        key = api_key or os.environ.get("VLLM_API_KEY" if base_url else "OPENAI_API_KEY")
        if base_url:
            key = key or "EMPTY"
        if not key:
            raise VLMError("OPENAI_API_KEY is not set (see .env.example)")
        # max_retries=0: _with_retries is the only retry loop, otherwise the two multiply.
        self._client = OpenAI(
            api_key=key, base_url=base_url, timeout=settings.request_timeout_s, max_retries=0
        )
        self._json_mode = True

    def _call(self, prompt: str, video: SampledVideo) -> Reply:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for f in video.frames:
            b64 = base64.b64encode(f.jpeg).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": self.detail},
                }
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._json_mode and "response_format" in str(exc):
                self._json_mode = False  # some servers do not support JSON mode
                kwargs.pop("response_format")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        latency = (time.perf_counter() - start) * 1000
        usage = {
            "latency_ms": round(latency, 1),
            "input_tokens": getattr(resp.usage, "prompt_tokens", None),
            "output_tokens": getattr(resp.usage, "completion_tokens", None),
        }
        text = resp.choices[0].message.content if resp.choices else ""
        return parse_json_object(text or ""), usage


class VLLMClient(OpenAIClient):
    provider = "vllm"
    key_name = "VLLM_API_KEY"

    def __init__(self, settings: Settings):
        super().__init__(settings, base_url=settings.vllm_base_url)


def _band(value: float, edges: list[float], higher_is_better: bool = True) -> tuple[int, float]:
    """Map a value onto a 1..len(edges)+1 score, and return a 0-1 margin from the nearest edge."""
    if higher_is_better:
        score = 1 + sum(value >= e for e in edges)
    else:
        score = 1 + sum(value <= e for e in edges)
    gaps = [abs(value - e) / max(abs(e), 1e-6) for e in edges]
    return int(score), float(min(1.0, min(gaps) * 2)) if gaps else 1.0


def _dimension(score: float, margin: float, reasoning: str) -> dict[str, Any]:
    return {"score": score, "confidence": round(0.5 + 0.45 * margin, 2), "reasoning": reasoning}


def _motion_score(t: dict[str, float]) -> tuple[int, float]:
    blur, blur_margin = _band(t["laplacian_var_median"], [20, 40, 80, 150])
    shake, shake_margin = _band(t["flow_jitter_pct"], [3.0, 1.5, 0.8, 0.4], higher_is_better=False)
    motion = max(1, min(blur, shake) - (1 if blur <= 3 and shake <= 3 else 0))
    if t["flow_mean_pct"] > 5:  # whip pans
        motion = 1
    elif t["flow_mean_pct"] > 3:
        motion = min(motion, 2)
    return motion, min(blur_margin, shake_margin)


def _lighting_score(t: dict[str, float]) -> tuple[int, float]:
    brightness = t["brightness_mean"]
    if brightness < 35 or t["under_exposed_ratio"] > 0.6 or t["over_exposed_ratio"] > 0.5:
        return 1, min(1.0, abs(brightness - 35) / 35)
    light, margin = _band(brightness, [60, 90, 110])
    light += 1 if light < 5 else 0
    if t["flicker_cv"] > 0.15:
        light = min(light, 3)
    return min(light, 5), margin


def _hand_score(t: dict[str, float]) -> tuple[int, float]:
    hand, margin = _band(t["hand_presence_frac"], [0.1, 0.35, 0.65, 0.85])
    if hand == 5 and t["skin_ratio_lower"] < 0.06:
        hand = 4
    return hand, margin


def _framing_score(t: dict[str, float]) -> tuple[int, float]:
    # Optical flow cannot see camera motion in dark or textureless frames, so a "static"
    # reading there is not evidence of a fixed camera: leave framing to the other gates.
    measurable = t["brightness_mean"] >= 35 and t["edge_density"] >= 0.005
    static = measurable and (t["static_ratio"] > 0.85 or t["camera_motion_pct"] < 0.05)
    if not measurable:
        margin = 0.0
    else:
        clear_cut = t["camera_motion_pct"] < 0.02 or (not static and t["hand_presence_frac"] > 0.5)
        margin = 0.9 if clear_cut else 0.3
    return (0 if static else 1), margin


def _overall_score(ego: int, hand: int, motion: int, light: int) -> int:
    if ego == 0:
        return 1
    overall = round((hand + motion + light) / 3) - (1 if min(hand, motion, light) <= 2 else 0)
    return max(1, min(5, overall))


def offline_scores(t: dict[str, float]) -> dict[str, Any]:
    """Scores from telemetry alone: the baseline a VLM has to beat."""
    motion, motion_margin = _motion_score(t)
    light, light_margin = _lighting_score(t)
    hand, hand_margin = _hand_score(t)
    ego, ego_margin = _framing_score(t)
    scene, scene_margin = _band(t["edge_density"], [0.02, 0.04, 0.07, 0.10])
    overall = _overall_score(ego, hand, motion, light)
    framing_verdict = (
        "fixed camera, not head-mounted."
        if ego == 0
        else "camera moves continuously, consistent with head mount."
    )
    return {
        "egocentric_framing": _dimension(
            ego,
            ego_margin,
            f"Offline rule: static ratio {t['static_ratio']:.2f}, "
            f"camera motion {t.get('camera_motion_pct', 0):.2f}%; {framing_verdict}",
        ),
        "motion_quality": _dimension(
            motion,
            motion_margin,
            f"Offline rule: sharpness {t['laplacian_var_median']:.0f}, "
            f"jitter {t['flow_jitter_pct']:.2f}%, all motion {t['flow_mean_pct']:.2f}%.",
        ),
        "scene_diversity": _dimension(
            scene,
            scene_margin,
            f"Offline rule: edge density {t['edge_density']:.3f} as a clutter proxy; "
            "cannot judge across locations.",
        ),
        "hand_visibility": _dimension(
            hand,
            hand_margin,
            f"Offline rule: likely hands in {t['hand_presence_frac']:.0%} of moments "
            "(skin-tone proxy, lower frame).",
        ),
        "lighting_exposure": _dimension(
            light,
            light_margin,
            f"Offline rule: brightness {t['brightness_mean']:.0f}, "
            f"under {t['under_exposed_ratio']:.0%}, over {t['over_exposed_ratio']:.0%}, "
            f"flicker {t['flicker_cv']:.2f}.",
        ),
        "duplicate_risk": _dimension(0, 0.5, "Offline rule: set by the perceptual-hash registry."),
        "overall_usability": _dimension(
            overall,
            min(hand_margin, motion_margin, light_margin),
            "Offline rule: rounded mean of hands, motion and lighting, "
            "minus 1 if any is 2 or lower.",
        ),
    }


def _bimanual_mode(left: bool, right: bool) -> str:
    if left and right:
        return "symmetric_bimanual"
    if left:
        return "unimanual_left"
    return "unimanual_right" if right else "no_hands"


def offline_annotations(video: SampledVideo) -> dict[str, Any]:
    """Only what telemetry can support: hand presence per frame and walking. No objects."""
    t = video.telemetry
    hands = []
    for f in video.frames:
        left, right = f.hand_sides
        hands.append(
            {
                "time_sec": f.time_sec,
                "left_hand": "idle_visible" if left else "not_visible",
                "right_hand": "idle_visible" if right else "not_visible",
                "bimanual_mode": _bimanual_mode(left, right),
                "confidence": 0.6,
            }
        )
    actions = []
    if t["hand_presence_frac"] < 0.1 and t["flow_mean_pct"] > 0.3:
        actions.append(
            {
                "action": "walk_navigate",
                "start_sec": 0.0,
                "end_sec": max(0.1, video.duration_sec),
                "confidence": 0.55,
            }
        )
    view = (
        "optimal_head_mount_clear"
        if t["hand_presence_frac"] > 0.65 and t["flow_jitter_pct"] < 1.5
        else "partial_object_occlusion"
    )
    return {
        "scene": {"scene_type": "other", "confidence": 0.3},
        "viewpoint": {"viewpoint_quality": view, "confidence": 0.55},
        "actions": actions,
        "objects": [],
        "hand_states": hands,
    }


_NO_USAGE = {"latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0}


class OfflineClient(BaseVLMClient):
    provider = "offline"

    def score(
        self,
        video: SampledVideo,
        prompt_version: str = prompts.PROMPT_VERSION,
        duplicate_note: str = "",
    ) -> Reply:
        return offline_scores(video.telemetry), dict(_NO_USAGE)

    def annotate(self, video: SampledVideo) -> Reply:
        return offline_annotations(video), dict(_NO_USAGE)


def make_client(settings: Settings) -> BaseVLMClient:
    return {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
        "vllm": VLLMClient,
        "offline": OfflineClient,
    }[settings.provider](settings)
