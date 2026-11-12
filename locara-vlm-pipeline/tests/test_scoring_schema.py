import copy

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.schemas.scoring_schema import QualityScore, Scores, video_id_for
from src.scorer.quality_scorer import QualityScorer, VideoInput, composite_score, route
from src.scorer.vlm_client import BaseVLMClient, VLMError, parse_json_object

GOOD = {
    "egocentric_framing": {"score": 1, "confidence": 0.9, "reasoning": "Hands enter from below."},
    "motion_quality": {"score": 4, "confidence": 0.9, "reasoning": "Mostly sharp."},
    "scene_diversity": {"score": 4, "confidence": 0.9, "reasoning": "Busy kitchen."},
    "hand_visibility": {"score": 5, "confidence": 0.9, "reasoning": "Both hands pouring."},
    "lighting_exposure": {"score": 4, "confidence": 0.9, "reasoning": "Even daylight."},
    "duplicate_risk": {"score": 0, "confidence": 0.9, "reasoning": "No match."},
    "overall_usability": {"score": 4, "confidence": 0.9, "reasoning": "Good training clip."},
}


def scores(**changes):
    d = copy.deepcopy(GOOD)
    for k, v in changes.items():
        d[k]["score"] = v
    return Scores.model_validate(d)


def record(tier="usable"):
    return {
        "video_id": video_id_for("clip.mp4"),
        "processed_at": "2026-11-06T10:00:00Z",
        "frame_count_sampled": 12,
        "scores": GOOD,
        "tier": tier,
        "tier_confidence": 0.873,
    }


def test_required_output_shape():
    q = QualityScore.model_validate(record())
    out = q.to_output(include_meta=False)
    assert list(out) == [
        "video_id",
        "processed_at",
        "frame_count_sampled",
        "scores",
        "tier",
        "tier_confidence",
    ]
    assert list(out["scores"]) == list(GOOD)
    assert out["tier_confidence"] == 0.87
    assert set(out["scores"]["hand_visibility"]) == {"score", "reasoning", "confidence"}


@pytest.mark.parametrize(
    "dim,bad",
    [
        ("motion_quality", 0),
        ("motion_quality", 6),
        ("motion_quality", 3.5),
        ("egocentric_framing", 2),
        ("duplicate_risk", -0.1),
    ],
)
def test_scale_bounds(dim, bad):
    with pytest.raises(ValidationError):
        scores(**{dim: bad})


def test_rejects_bad_ids_and_tiers():
    r = record()
    r["video_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        QualityScore.model_validate(r)
    with pytest.raises(ValidationError):
        QualityScore.model_validate(record(tier="maybe"))
    r = record()
    r["scores"] = {k: v for k, v in GOOD.items() if k != "hand_visibility"}
    with pytest.raises(ValidationError):
        QualityScore.model_validate(r)


def test_video_id_is_stable_uuid4():
    a, b = video_id_for("x.mp4"), video_id_for("x.mp4")
    assert a == b and a[14] == "4"


def test_composite_weights():
    assert composite_score(scores()) == pytest.approx(
        0.3 * 5 + 0.25 * 4 + 0.2 * 4 + 0.15 * 4 + 0.1 * 4
    )


@pytest.mark.parametrize(
    "change,gate",
    [
        ({"egocentric_framing": 0}, "egocentric_framing"),
        ({"duplicate_risk": 1}, "duplicate_risk"),
        ({"hand_visibility": 1}, "hand_visibility"),
        ({"motion_quality": 1}, "motion_quality"),
        ({"lighting_exposure": 1}, "lighting_exposure"),
    ],
)
def test_hard_gates_reject(change, gate):
    r = route(scores(**change))
    assert r.tier == "reject" and gate in r.hard_gates_failed


def test_tiers_by_composite_and_confidence():
    assert route(scores()).tier == "usable"
    mid = scores(
        hand_visibility=3,
        motion_quality=3,
        lighting_exposure=3,
        scene_diversity=3,
        overall_usability=3,
    )
    assert route(mid).tier == "borderline"
    low = scores(
        hand_visibility=2,
        motion_quality=2,
        lighting_exposure=2,
        scene_diversity=2,
        overall_usability=2,
    )
    assert route(low).tier == "reject"
    assert route(scores(), disagreements=2).tier == "borderline"  # confidence below 0.80


def test_parse_json_variants():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    with pytest.raises(VLMError):
        parse_json_object("no json here")


class Stub(BaseVLMClient):
    provider = "stub"

    def __init__(self, reply):
        super().__init__(Settings())
        self.reply = reply
        self.model = "stub-model"

    def score(self, video, prompt_version="v3", duplicate_note=""):
        if isinstance(self.reply, Exception):
            raise self.reply
        return copy.deepcopy(self.reply), {"latency_ms": 1.0}


def test_scorer_end_to_end_with_stub(clips):
    reply = copy.deepcopy(GOOD)
    reply["motion_quality"]["score"] = "4.0"  # string numbers are coerced
    sc = QualityScorer(Settings(), client=Stub(reply), workers=2)
    vids = [
        VideoInput(str(clips["ideal10"]), str(clips["ideal10"]), video_id_for("a")),
        VideoInput(str(clips["ideal10_dup"]), str(clips["ideal10_dup"]), video_id_for("b")),
    ]
    a, b = sc.score_batch(vids)
    QualityScore.model_validate(a)
    assert a["scores"]["motion_quality"]["score"] == 4
    assert b["scores"]["duplicate_risk"]["score"] == 1 and b["tier"] == "reject"
    assert b["meta"]["duplicate_of"] == a["video_id"]


def test_scorer_records_errors_and_continues(clips, tmp_path):
    sc = QualityScorer(Settings(), client=Stub(VLMError("boom")))
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    out = sc.score_batch(
        [
            VideoInput(str(clips["ideal10"]), str(clips["ideal10"]), video_id_for("c")),
            VideoInput(str(bad), str(bad), video_id_for("d")),
        ]
    )
    assert "scoring failed" in out[0]["error"] and "sampling failed" in out[1]["error"]


def test_every_prompt_version_builds():
    from src.scorer import prompts
    from src.scorer.frame_sampler import telemetry_summary

    tel = {
        "laplacian_var_median": 150,
        "laplacian_var_min": 90,
        "camera_motion_pct": 0.2,
        "flow_mean_pct": 0.5,
        "flow_jitter_pct": 0.3,
        "static_ratio": 0.05,
        "brightness_mean": 140,
        "under_exposed_ratio": 0.01,
        "over_exposed_ratio": 0.0,
        "flicker_cv": 0.02,
        "skin_ratio_lower": 0.12,
        "hand_presence_frac": 0.9,
        "edge_density": 0.06,
    }
    for v in prompts.VERSIONS:
        p = prompts.build_scoring_prompt(v, telemetry_summary(tel), 45.0, 12, "")
        assert '"egocentric_framing"' in p and "{telemetry}" not in p and "{n_frames}" not in p
    v3 = prompts.build_scoring_prompt("v3", telemetry_summary(tel), 45.0, 12, "")
    assert "45.0 s, 12 frames" in v3 and "camera motion" in v3
    assert "home_kitchen" in prompts.build_annotation_prompt(10.0, [1.0, 2.0])
