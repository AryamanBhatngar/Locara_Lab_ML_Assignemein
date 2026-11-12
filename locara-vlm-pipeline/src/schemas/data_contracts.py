"""Downstream data contracts (Q10).

Each consumer gets a versioned, append-only event built from a scored video. Producers may
add optional fields in a minor version; removing or renaming a field needs a new major
version and a parallel topic, so no consumer breaks on deploy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .scoring_schema import QualityScore


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    event_id: str
    video_id: str
    emitted_at: str


class PayoutQualityEvent(_Contract):
    """Worker payment tier system. Consumed from an SQS queue; idempotent on event_id."""

    schema_version: Literal["payout.v1"] = "payout.v1"
    worker_id: str
    tier: Literal["usable", "borderline", "reject"]
    tier_confidence: float = Field(ge=0, le=1)
    overall_usability: int = Field(ge=1, le=5)
    duplicate_flag: bool
    payout_band: Literal["quality_bonus", "base", "hold_pending_review", "none"]
    reason_codes: list[str]


class AnnotationQueueTask(_Contract):
    """Annotation queue service. One task per video sent to human review."""

    schema_version: Literal["annotation_task.v1"] = "annotation_task.v1"
    video_uri: str
    priority: int = Field(ge=0, le=100, description="Higher is reviewed first")
    review_reason: Literal["borderline_quality", "low_confidence_labels", "spot_check"]
    prelabel_uri: str
    taxonomy_version: str


class ClientManifestEntry(_Contract):
    """Robotics client delivery portal. One line per clip in a versioned release manifest."""

    schema_version: Literal["delivery.v1"] = "delivery.v1"
    release: str
    video_uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_sec: float
    quality_scores: dict[str, float]
    annotation_uri: str
    annotation_format: Literal["coco_video_ext.v1"]
    label_status: Literal["human_verified", "auto_accepted"]


def payout_band(q: QualityScore) -> str:
    if q.scores.duplicate_risk.score >= 0.5:
        return "none"
    if q.tier == "usable":
        return "quality_bonus" if q.scores.overall_usability.score >= 4 else "base"
    if q.tier == "borderline":
        return "hold_pending_review"
    return "none"


def review_priority(q: QualityScore) -> int:
    """Least certain first: clips nearest the decision boundary teach the scorer the most."""
    return max(0, min(100, round((1.0 - q.tier_confidence) * 100)))
