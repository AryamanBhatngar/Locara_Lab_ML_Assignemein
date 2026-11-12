"""Sort provider failures by what they mean for retrying.

Both SDKs put the HTTP status on their exceptions (`status_code` for OpenAI, `code` for Gemini), so
classifying needs neither SDK to be imported. Their exception text is not safe to repeat: it can
include the whole response body, so `AuthError` messages are built from a template instead.
"""

from __future__ import annotations

import enum
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

AUTH_SKIP_MESSAGE = "skipped: authentication failed"

# Gemini answers an invalid or expired key with a 400, not a 401.
_BAD_KEY_MARKERS = ("api_key_invalid", "api key not valid", "api key expired", "invalid api key")


class VLMError(RuntimeError):
    """The model call failed or returned something that is not valid JSON."""


class BadResponseError(VLMError):
    """The model answered, but not with a usable JSON object."""


class AuthError(VLMError):
    """The provider rejected the credentials. Retrying cannot help and every clip would fail."""


class ErrorKind(enum.Enum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"  # timeouts, 5xx and anything unrecognised
    BAD_RESPONSE = "bad_response"


def http_status(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def classify_error(exc: Exception) -> ErrorKind:
    if isinstance(exc, BadResponseError):
        return ErrorKind.BAD_RESPONSE
    status = http_status(exc)
    if status in (401, 403):
        return ErrorKind.AUTH
    if status == 400 and any(marker in str(exc).lower() for marker in _BAD_KEY_MARKERS):
        return ErrorKind.AUTH
    if status == 429:
        return ErrorKind.RATE_LIMIT
    return ErrorKind.TRANSIENT


def _from_header(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return (when - datetime.now(timezone.utc)).total_seconds()


def _from_gemini_details(exc: Exception) -> float | None:
    """Gemini puts the wait in the body: {"error": {"details": [{"retryDelay": "34s"}]}}."""
    body: Any = getattr(exc, "details", None)
    details = body.get("error", {}).get("details") if isinstance(body, dict) else None
    for item in details if isinstance(details, list) else []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(delay))
        if match:
            return float(match.group(1))
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    """How long the server asked us to wait, or None if it did not say usefully."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    wait: float | None = None
    millis = headers.get("retry-after-ms")
    if millis:
        seconds = _from_header(str(millis))
        wait = None if seconds is None else seconds / 1000
    elif headers.get("retry-after"):
        wait = _from_header(str(headers.get("retry-after")))
    if wait is None:
        wait = _from_gemini_details(exc)
    return wait if wait is not None and wait > 0 else None
