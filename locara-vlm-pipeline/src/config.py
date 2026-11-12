"""Provider names, routing weights and thresholds, sampling and input limits.

Settings come from environment variables or a .env file in the working directory. API keys are
deliberately not part of `Settings`: each client reads its own key when it is created.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments, optional quotes.

    Existing environment variables win, so a shell export always overrides the file.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a number, got {raw!r}")
    return value


# Routing (Q5). Weights sum to 1.0; the composite is on the 1-5 scale.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "hand_visibility": 0.30,
    "motion_quality": 0.25,
    "lighting_exposure": 0.20,
    "scene_diversity": 0.15,
    "overall_usability": 0.10,
}
USABLE_MIN_COMPOSITE = 3.75
BORDERLINE_MIN_COMPOSITE = 2.50
USABLE_MIN_CONFIDENCE = 0.80

# A dimension at or below this value is a hard failure (1-5 scales).
HARD_FAIL_MAX = 1

# "Bad video" definition used by the precision/recall evaluation (Q4).
BAD_MAX_1_TO_5 = 2  # score <= 2 on a 1-5 dimension counts as "bad"

# Frame sampling (Q1)
MIN_FRAMES = 8  # clips <= 10 s
TARGET_FRAMES = 12  # clips 30-60 s
MAX_FRAMES = 16  # clips > 60 s
CANDIDATES_PER_SEGMENT = 3  # frames decoded per segment; the sharpest is kept
FRAME_LONG_SIDE = 512  # resize sampled frames to this long side (keeps aspect)
TELEMETRY_FPS = 4.0  # dense low-resolution scan rate for OpenCV signals
TELEMETRY_WIDTH = 160

# How far the container header may drift from what was decoded before it is overruled.
FPS_TOLERANCE = 0.20
FRAME_COUNT_TOLERANCE = 0.02

# Inputs
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
MAX_VIDEOS_IN_FLIGHT = 16  # started but not yet scored: bounds frames in memory, files on disk
DOWNLOAD_WORKERS = 4
DECODE_TIMEOUT_S = 120.0  # per video
DECODER_START_TIMEOUT_S = 60.0  # for a worker process to come up

# A small file can decode to huge frames or a very long clip, so limit what was decoded,
# not just the file size.
MAX_FRAME_PIXELS = 7680 * 4320  # 8K
MAX_DECODED_FRAMES = 500_000  # about 4.6 hours at 30 fps

# Duplicate detection (dHash, 64-bit). Two clips are duplicates when the median Hamming
# distance between their aligned keyframe hashes is at or below this value.
DUPLICATE_MAX_HAMMING = 8

# Model calls
BAD_RESPONSE_RETRIES = 1  # temperature is 0, so more retries rarely change the answer
MAX_RETRY_AFTER_S = 120.0  # a server cannot park a worker for longer than this

# Pre-annotation gating (Part B)
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
REVIEW_FLOOR = 0.50


@dataclass(frozen=True)
class Settings:
    provider: str = "offline"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    request_timeout_s: float = 45.0
    max_retries: int = 3
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    aws_s3_bucket: str = ""
    aws_region: str = "ap-south-1"
    s3_max_download_mb: float = 500.0
    cache_dir: Path = field(default_factory=lambda: Path(".cache/videos"))

    @property
    def model(self) -> str:
        return {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "vllm": self.vllm_model,
        }.get(self.provider, "offline-cv-baseline")


def get_settings(provider: str | None = None, model: str | None = None) -> Settings:
    load_dotenv()
    prov = (provider or _env("VLM_PROVIDER", "offline")).lower()
    if prov not in {"gemini", "openai", "vllm", "offline"}:
        raise ValueError(f"Unknown VLM_PROVIDER {prov!r}: use gemini, openai, vllm or offline")
    s = Settings(
        provider=prov,
        gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o"),
        vllm_base_url=_env("VLLM_BASE_URL", "http://localhost:8000/v1"),
        vllm_model=_env("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        request_timeout_s=_env_float("VLM_TIMEOUT_S", 45.0),
        confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD),
        aws_s3_bucket=_env("AWS_S3_BUCKET", ""),
        aws_region=_env("AWS_REGION", "ap-south-1"),
        s3_max_download_mb=_env_float("S3_MAX_DOWNLOAD_MB", 500.0),
        cache_dir=Path(_env("VIDEO_CACHE_DIR", ".cache/videos")),
    )
    if model:
        key = {"gemini": "gemini_model", "openai": "openai_model", "vllm": "vllm_model"}.get(prov)
        if key:
            s = Settings(**{**s.__dict__, key: model})
    return s
