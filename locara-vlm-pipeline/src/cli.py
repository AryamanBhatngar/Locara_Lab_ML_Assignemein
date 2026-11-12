"""Helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config
from .redact import redact
from .scorer.frame_sampler import VideoReadError
from .scorer.vlm_client import VLMError

# Problems a user can fix. Anything else is a bug and should keep its traceback.
EXPECTED_ERRORS = (OSError, ValueError, ImportError, VLMError, VideoReadError)


def _message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"file not found: {exc.filename}"
    return redact(exc)


def exits_cleanly(main: Callable[[list[str] | None], int]) -> Callable[[list[str] | None], int]:
    """Turn expected errors into a one-line message and exit code 2."""

    @functools.wraps(main)
    def wrapper(argv: list[str] | None = None) -> int:
        try:
            return main(argv)
        except EXPECTED_ERRORS as exc:
            print(f"error: {_message(exc)}", file=sys.stderr)
            return 2

    return wrapper


def positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds, got {text}")
    return value


def add_decode_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decode-timeout",
        type=positive_seconds,
        default=config.DECODE_TIMEOUT_S,
        help=f"seconds to decode one video before giving up (default {config.DECODE_TIMEOUT_S:g})",
    )


def add_cache_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-cache", action="store_true", help="keep downloaded S3 files after use"
    )


def add_download_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--download-workers",
        type=int,
        default=config.DOWNLOAD_WORKERS,
        help=f"S3 downloads at once (default {config.DOWNLOAD_WORKERS})",
    )
    add_cache_option(parser)


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read the output of score_videos.py: one record, or a list of them."""
    data = read_json(path)
    records = data if isinstance(data, list) else [data]
    for record in records:
        if not isinstance(record, dict) or not ("tier" in record or "error" in record):
            raise ValueError(f"{path} does not look like score_videos.py output")
    return records


def prepare_output(*paths: str | None) -> None:
    """Check output paths up front, so a bad one fails before any paid model calls."""
    for path in paths:
        if path is None:
            continue
        target = Path(path)
        if target.is_dir():
            raise ValueError(f"output path is a folder: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
