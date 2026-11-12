"""Pre-annotation layer (Part B).

Takes a video plus its Part A scoring record and produces candidate labels in the taxonomy.
Confidence gating:

  confidence >= threshold (default 0.75)  -> kept as a pre-label
  REVIEW_FLOOR (0.50) <= confidence < threshold -> kept, needs_human_verification = true
  confidence < REVIEW_FLOOR                -> dropped (counted, never shown as a pre-label)

Videos in the "reject" tier are skipped. Every label on a "borderline" video is marked for
verification, because the clip itself is under review.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel, ValidationError

from .. import config
from ..schemas.taxonomy_schema import (
    ActionSegment,
    HandState,
    ObjectAnnotation,
    PreAnnotation,
    SceneLabel,
    ViewpointLabel,
)
from ..scorer.frame_sampler import SampledVideo, sample_video
from ..scorer.vlm_client import BaseVLMClient

log = logging.getLogger(__name__)


class SkipVideo(Exception):
    """The video is not eligible for pre-annotation (rejected, or scoring failed)."""


def ensure_eligible(score_record: dict[str, Any]) -> None:
    """Raise SkipVideo unless the record's tier allows pre-annotation."""
    if "error" in score_record:
        raise SkipVideo(f"scoring failed: {score_record['error']}")
    tier = score_record.get("tier")
    if tier not in ("usable", "borderline"):
        raise SkipVideo(f"tier is {tier}")


def _number(value: Any) -> float | None:
    """A finite float, or None for anything a model might send instead."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _gate(item: dict[str, Any], threshold: float, force_review: bool) -> dict[str, Any] | None:
    conf = _number(item.get("confidence", 0.0))
    if conf is None or conf < config.REVIEW_FLOOR:
        return None
    item = dict(item, confidence=round(min(1.0, max(0.0, conf)), 2))
    item["needs_human_verification"] = force_review or conf < threshold
    return item


def _clip_segment(seg: Any, duration: float) -> dict[str, Any] | None:
    if not isinstance(seg, dict):
        return None
    start, end = _number(seg.get("start_sec", 0)), _number(seg.get("end_sec", 0))
    if start is None or end is None:
        return None
    return {
        **seg,
        "start_sec": round(max(0.0, min(start, duration)), 2),
        "end_sec": round(max(0.0, min(end, duration)), 2),
    }


def _non_overlapping(segments: list[ActionSegment]) -> list[ActionSegment]:
    """Keep the more confident segment where two overlap."""
    out: list[ActionSegment] = []
    for seg in sorted(segments, key=lambda s: -s.confidence):
        if all(seg.end_sec <= o.start_sec or seg.start_sec >= o.end_sec for o in out):
            out.append(seg)
    return sorted(out, key=lambda s: s.start_sec)


class _LabelBuilder:
    """Validates one label at a time and counts what it drops."""

    def __init__(self, threshold: float, force_review: bool, duration: float):
        self.threshold = threshold
        self.force_review = force_review
        self.duration = duration
        self.dropped = 0

    def build(self, model: type[BaseModel], item: Any) -> Any:
        gated = _gate(item, self.threshold, self.force_review) if isinstance(item, dict) else None
        if gated is None:
            self.dropped += 1
            return None
        try:
            return model.model_validate(gated)
        except ValidationError as exc:
            log.warning("dropping invalid %s: %s", model.__name__, exc.errors()[0]["msg"])
            self.dropped += 1
            return None

    def build_all(self, model: type[BaseModel], items: list[Any]) -> list[Any]:
        return [label for label in (self.build(model, item) for item in items) if label]

    def inside_clip(self, labels: list[Any]) -> list[Any]:
        kept = [label for label in labels if label.time_sec <= self.duration]
        self.dropped += len(labels) - len(kept)
        return kept


class PreAnnotator:
    def __init__(
        self, client: BaseVLMClient, threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD
    ):
        if not config.REVIEW_FLOOR < threshold <= 1.0:
            raise ValueError(f"threshold must be between {config.REVIEW_FLOOR} and 1.0")
        self.client = client
        self.threshold = threshold

    def annotate(
        self, video_path: str, score_record: dict[str, Any], sampled: SampledVideo | None = None
    ) -> PreAnnotation:
        ensure_eligible(score_record)
        tier = score_record["tier"]
        sampled = sampled or sample_video(video_path)
        raw, _ = self.client.annotate(sampled)
        duration = sampled.duration_sec
        labels = _LabelBuilder(self.threshold, tier == "borderline", duration)

        scene = (
            labels.build(SceneLabel, raw["scene"]) if isinstance(raw.get("scene"), dict) else None
        )
        view = raw.get("viewpoint")
        viewpoint = labels.build(ViewpointLabel, view) if isinstance(view, dict) else None
        segments = [_clip_segment(s, duration) for s in _as_list(raw.get("actions"))]
        actions = labels.build_all(ActionSegment, segments)
        objects = labels.inside_clip(
            labels.build_all(ObjectAnnotation, _as_list(raw.get("objects"))[:6])
        )
        hands = labels.inside_clip(labels.build_all(HandState, _as_list(raw.get("hand_states"))))

        return PreAnnotation(
            video_id=score_record["video_id"],
            source=score_record.get("meta", {}).get("source", video_path),
            duration_sec=duration,
            tier=tier,
            provider=f"{self.client.provider}:{self.client.model}",
            threshold=self.threshold,
            scene=scene,
            viewpoint=viewpoint,
            actions=_non_overlapping(actions),
            objects=objects,
            hand_states=hands,
            dropped_low_confidence=labels.dropped,
        )
