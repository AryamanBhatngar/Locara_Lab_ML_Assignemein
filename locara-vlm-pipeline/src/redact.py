"""Keep credentials out of logs, error messages and output JSON.

Provider SDKs put request details in their exceptions: a 401 can quote the key, and a connection
error can quote a URL with credentials in it. Anything that came from an exception passes through
`redact` before it is logged, printed or stored.
"""

from __future__ import annotations

import os
import re

SECRET_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "VLLM_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
MASK = "[redacted]"
_MIN_SECRET_LENGTH = 8  # placeholders such as EMPTY are not worth masking

_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_*\-]{6,}"),  # OpenAI, including its masked echoes
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"(?i)\b(?:api[_-]?key|key|token|secret|signature)=[^&\s\"']+"),  # query strings
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),  # user:password@host
)


def redact(text: object) -> str:
    cleaned = str(text)
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= _MIN_SECRET_LENGTH:
            cleaned = cleaned.replace(value, MASK)
    for pattern in _PATTERNS:
        cleaned = pattern.sub(MASK, cleaned)
    return cleaned
