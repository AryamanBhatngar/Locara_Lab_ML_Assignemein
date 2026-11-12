"""Malformed model replies must cost one video or one label at most, never a batch."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.annotation.exporters import to_label_studio
from src.annotation.pre_annotator import PreAnnotator
from src.config import Settings
from src.schemas.scoring_schema import video_id_for
from src.scorer.frame_sampler import sample_video
from src.scorer.quality_scorer import QualityScorer, VideoInput
from src.scorer.vlm_client import (
    BaseVLMClient,
    GeminiClient,
    OpenAIClient,
    VLMError,
    parse_json_object,
)

GOOD = {
    "egocentric_framing": {"score": 1, "confidence": 0.9, "reasoning": "Hands enter from below."},
    "motion_quality": {"score": 4, "confidence": 0.9, "reasoning": "Mostly sharp."},
    "scene_diversity": {"score": 4, "confidence": 0.9, "reasoning": "Busy kitchen."},
    "hand_visibility": {"score": 5, "confidence": 0.9, "reasoning": "Both hands pouring."},
    "lighting_exposure": {"score": 4, "confidence": 0.9, "reasoning": "Even daylight."},
    "duplicate_risk": {"score": 0, "confidence": 0.9, "reasoning": "No match."},
    "overall_usability": {"score": 4, "confidence": 0.9, "reasoning": "Good training clip."},
}
NOTE = "x"


class ByClip(BaseVLMClient):
    """Replies per clip file name; anything else gets the good reply."""

    provider = "stub"

    def __init__(self, replies):
        super().__init__(Settings())
        self.replies = replies
        self.model = "stub-model"

    def score(self, video, prompt_version="v3", duplicate_note=""):
        return copy.deepcopy(self.replies.get(Path(video.path).name, GOOD)), {}


def with_change(**changes):
    return {**copy.deepcopy(GOOD), **changes}


def score_pair(clips, bad_reply):
    """Score an original and its re-encoded copy; only the copy gets the bad reply."""
    videos = [
        VideoInput(str(clips[key]), str(clips[key]), video_id_for(key))
        for key in ("ideal10", "ideal10_dup")
    ]
    client = ByClip({clips["ideal10_dup"].name: bad_reply})
    return QualityScorer(Settings(), client=client, workers=2).score_batch(videos)


@pytest.mark.parametrize(
    "reply",
    [
        with_change(motion_quality={"score": math.inf, "confidence": 0.9, "reasoning": NOTE}),
        with_change(motion_quality={"score": math.nan, "confidence": 0.9, "reasoning": NOTE}),
        with_change(motion_quality={"score": [4], "confidence": 0.9, "reasoning": NOTE}),
        with_change(motion_quality="good"),
        {"unexpected": True},
    ],
)
def test_bad_reply_fails_one_video_and_keeps_the_rest(clips, reply):
    good, bad = score_pair(clips, reply)
    assert good["tier"] == "usable"
    assert "scoring failed" in bad["error"]


@pytest.mark.parametrize("duplicate_risk", [0.2, None, "low", ["x"]])
def test_duplicate_override_survives_an_odd_duplicate_risk_value(clips, duplicate_risk):
    original, copy_of = score_pair(clips, with_change(duplicate_risk=duplicate_risk))
    assert copy_of["tier"] == "reject"
    assert copy_of["meta"]["duplicate_of"] == original["video_id"]
    assert copy_of["scores"]["duplicate_risk"]["score"] == 1


def test_duplicate_override_keeps_reasoning_within_the_schema_limit(clips):
    long_note = {"score": 0, "confidence": 0.9, "reasoning": "y" * 1195}
    _, copy_of = score_pair(clips, with_change(duplicate_risk=long_note))
    assert copy_of["tier"] == "reject"
    assert len(copy_of["scores"]["duplicate_risk"]["reasoning"]) <= 1200


def test_unexpected_exception_from_a_client_is_recorded_per_video(clips):
    class Broken(ByClip):
        def score(self, video, prompt_version="v3", duplicate_note=""):
            if Path(video.path).name == clips["ideal10_dup"].name:
                raise ZeroDivisionError("provider bug")
            return super().score(video, prompt_version, duplicate_note)

    videos = [
        VideoInput(str(clips[key]), str(clips[key]), video_id_for(key))
        for key in ("ideal10", "ideal10_dup")
    ]
    good, bad = QualityScorer(Settings(), client=Broken({}), workers=2).score_batch(videos)
    assert "tier" in good and "provider bug" in bad["error"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ('Here: {"reasoning": "a } b", "x": 1} done', {"reasoning": "a } b", "x": 1}),
        ('Note {the} result: {"a": 1}', {"a": 1}),
        ('{"a": "{"} trailing', {"a": "{"}),
        ('prefix ```json\n{"a": 1}\n``` suffix', {"a": 1}),
    ],
)
def test_json_is_found_inside_prose(text, expected):
    assert parse_json_object(text) == expected


@pytest.mark.parametrize(
    "text", ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', 'ok {"a": NaN} ok']
)
def test_non_finite_json_numbers_are_rejected(text):
    with pytest.raises(VLMError):
        parse_json_object(text)


@pytest.mark.parametrize("text", ["", "no json", '{"a": 1', "[1, 2]", '[{"a": 1}]'])
def test_replies_without_an_object_are_rejected(text):
    with pytest.raises(VLMError):
        parse_json_object(text)


ANNOTATION = {
    "scene": {"scene_type": "home_kitchen", "confidence": 0.9},
    "viewpoint": {"viewpoint_quality": "minor_hand_self_occlusion", "confidence": 0.9},
    "actions": [{"action": "reach", "start_sec": 0, "end_sec": 2, "confidence": 0.9}],
    "objects": [],
    "hand_states": [],
}
CUP = {
    "category": "container",
    "label": "cup",
    "role": "active_manipulated",
    "bbox": [0.1, 0.1, 0.2, 0.2],
    "confidence": 0.9,
}
HAND = {
    "left_hand": "idle_visible",
    "right_hand": "idle_visible",
    "bimanual_mode": "no_hands",
    "confidence": 0.9,
}


class Annotator(BaseVLMClient):
    provider = "stub"

    def __init__(self, reply):
        super().__init__(Settings())
        self.reply = reply
        self.model = "stub-model"

    def annotate(self, video):
        return copy.deepcopy(self.reply), {}


def annotate(clips, **changes):
    record = {"video_id": video_id_for("clip"), "tier": "usable", "meta": {"source": "clip.mp4"}}
    client = Annotator({**ANNOTATION, **changes})
    return PreAnnotator(client).annotate(
        str(clips["ideal10"]), record, sample_video(clips["ideal10"])
    )


@pytest.mark.parametrize("confidence", [math.nan, math.inf, None, "high", [0.9]])
def test_unusable_confidence_drops_the_label_not_the_clip(clips, confidence):
    result = annotate(clips, scene={"scene_type": "home_kitchen", "confidence": confidence})
    assert result.scene is None
    assert result.dropped_low_confidence == 1
    assert [a.action for a in result.actions] == ["reach"]


@pytest.mark.parametrize(
    "actions",
    [
        ["reach"],
        [None],
        [{"action": "reach", "start_sec": "0:01", "end_sec": 2, "confidence": 0.9}],
        [{"action": "reach", "start_sec": None, "end_sec": 2, "confidence": 0.9}],
        [{"action": "reach", "start_sec": 0, "end_sec": math.nan, "confidence": 0.9}],
    ],
)
def test_malformed_action_drops_the_label_not_the_clip(clips, actions):
    result = annotate(clips, actions=actions)
    assert result.actions == [] and result.dropped_low_confidence == 1
    assert result.scene is not None


@pytest.mark.parametrize("field", ["actions", "objects", "hand_states"])
@pytest.mark.parametrize("value", [{"category": "container"}, "none", 3])
def test_a_field_of_the_wrong_type_is_treated_as_empty(clips, field, value):
    result = annotate(clips, **{field: value})
    assert getattr(result, field) == []
    assert result.scene is not None


def test_labels_outside_the_clip_are_dropped(clips):
    result = annotate(
        clips,
        objects=[{**CUP, "time_sec": 999.0}, {**CUP, "time_sec": 2.0}],
        hand_states=[{**HAND, "time_sec": 999.0}, {**HAND, "time_sec": 2.0}],
    )
    assert [o.time_sec for o in result.objects] == [2.0]
    assert [h.time_sec for h in result.hand_states] == [2.0]
    assert result.dropped_low_confidence == 2


def test_label_studio_url_escapes_the_file_name(clips):
    annotation = annotate(clips)
    info = {"file_name": "take 1 #2.mp4", "width": 640, "height": 360, "fps": 15.0}
    task = to_label_studio([(annotation, info)], "/data/local-files/?d=")[0]
    assert task["data"]["video"] == "/data/local-files/?d=take%201%20%232.mp4"


def test_gemini_client_applies_the_configured_timeout(monkeypatch):
    seen = {}

    def fake_client(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(models=None)

    from google import genai

    monkeypatch.setattr(genai, "Client", fake_client)
    GeminiClient(Settings(provider="gemini", request_timeout_s=12.5), api_key="test")
    assert seen["http_options"].timeout == 12_500


def test_openai_client_leaves_retries_to_our_own_backoff(monkeypatch):
    seen = {}

    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: seen.update(kwargs))
    OpenAIClient(Settings(provider="openai"), api_key="test")
    assert seen["max_retries"] == 0


def test_record_with_error_serialises_as_valid_json(clips):
    good, bad = score_pair(
        clips, with_change(motion_quality={"score": math.inf, "reasoning": NOTE})
    )
    json.dumps([good, bad], allow_nan=False)


def test_validation_failure_is_reported_on_one_line(clips):
    _, bad = score_pair(clips, with_change(motion_quality="good", scene_diversity="bad"))
    assert "\n" not in bad["error"]
    assert bad["error"].startswith("scoring failed: ") and "(and 1 more)" in bad["error"]
