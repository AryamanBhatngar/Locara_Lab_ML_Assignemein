"""Pydantic v2 models for the required scoring JSON (Part A).

The keys and nesting of `QualityScore` match the assignment's output schema exactly. Two
additive fields are allowed: a per-dimension `confidence` (the brief asks for "per-dimension
scores, confidence, and reasoning") and an optional `meta` block with provenance (provider,
model, latency, telemetry). Consumers that only read the required keys are unaffected.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DIMENSIONS: tuple[str, ...] = (
    "egocentric_framing",
    "motion_quality",
    "scene_diversity",
    "hand_visibility",
    "lighting_exposure",
    "duplicate_risk",
    "overall_usability",
)
UNIT_DIMENSIONS = frozenset({"egocentric_framing", "duplicate_risk"})  # 0-1 scale
FIVE_POINT_DIMENSIONS = frozenset(DIMENSIONS) - UNIT_DIMENSIONS  # 1-5 scale

MAX_REASONING_CHARS = 1200
HASH_CHUNK_BYTES = 1024 * 1024

Tier = Literal["usable", "borderline", "reject"]


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("reasoning")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("reasoning must not be empty")
        return v


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    egocentric_framing: DimensionScore
    motion_quality: DimensionScore
    scene_diversity: DimensionScore
    hand_visibility: DimensionScore
    lighting_exposure: DimensionScore
    duplicate_risk: DimensionScore
    overall_usability: DimensionScore

    @model_validator(mode="after")
    def _check_scales(self) -> Scores:
        for name in DIMENSIONS:
            s = getattr(self, name).score
            if name in UNIT_DIMENSIONS:
                if not 0.0 <= s <= 1.0:
                    raise ValueError(f"{name} must be between 0 and 1, got {s}")
            else:
                if not 1 <= s <= 5 or float(s) != int(s):
                    raise ValueError(f"{name} must be an integer from 1 to 5, got {s}")
                getattr(self, name).score = int(s)
        return self

    def value(self, name: str) -> float:
        return getattr(self, name).score


class QualityScore(BaseModel):
    """One scored video. Serialise with `to_output()` to get the submission JSON."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    processed_at: str
    frame_count_sampled: int = Field(ge=1)
    scores: Scores
    tier: Tier
    tier_confidence: float = Field(ge=0.0, le=1.0)
    meta: dict[str, Any] | None = None

    @field_validator("video_id")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        return canonical_video_id(v)

    @field_validator("processed_at")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))  # 3.10 does not accept Z
        return v

    @field_validator("tier_confidence")
    @classmethod
    def _round(cls, v: float) -> float:
        return round(v, 2)

    def to_output(self, include_meta: bool = True) -> dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if not include_meta:
            d.pop("meta", None)
        return d


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_video_id(value: str) -> str:
    """The standard form of a UUID v4, or ValueError for anything else."""
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 4:
        raise ValueError(f"video_id must be a UUID v4, got {value!r}")
    return str(parsed)


def _uuid4_shaped(raw: bytes) -> str:
    data = bytearray(raw[:16])
    data[6] = (data[6] & 0x0F) | 0x40  # version 4
    data[8] = (data[8] & 0x3F) | 0x80  # RFC 4122 variant
    return str(uuid.UUID(bytes=bytes(data)))


def video_id_for(name: str) -> str:
    """UUID v4-shaped id for a string, such as an S3 object's bucket, key, ETag and size."""
    return _uuid4_shaped(uuid.uuid5(uuid.NAMESPACE_URL, name).bytes)


def video_id_for_file(path: str | Path) -> str:
    """UUID v4-shaped id from the file's bytes, so two files only share an id if they match.

    Real uploads carry the capture app's video_id; this is for files without one.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return _uuid4_shaped(digest.digest())
