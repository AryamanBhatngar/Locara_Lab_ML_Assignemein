"""S3 input support: download a video to the local cache, or list a prefix.

boto3 is imported lazily so local-only runs do not need AWS credentials. Downloads are cached
by bucket, ETag and key, so an overwritten object is fetched again. The scorer deletes what it
downloaded once a clip is scored, unless told to keep the cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..config import Settings
from ..redact import redact
from ..schemas.scoring_schema import video_id_for

log = logging.getLogger(__name__)

_BUCKET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CACHE_LOCK = threading.Lock()  # so pruning an empty folder cannot race a download into it


class S3Error(OSError):
    """An S3 request failed."""


@dataclass(frozen=True)
class ObjectInfo:
    etag: str
    size: int


@contextmanager
def _s3_errors(uri: str) -> Iterator[None]:
    from boto3.exceptions import Boto3Error
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        yield
    except (Boto3Error, BotoCoreError, ClientError) as exc:
        raise S3Error(redact(f"{uri}: {exc}")) from None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    # Not urlparse: '#' and '?' are ordinary characters in an S3 key.
    scheme, _, rest = uri.partition("://")
    bucket, _, key = rest.partition("/")
    if scheme != "s3" or not _BUCKET_NAME.fullmatch(bucket):
        raise ValueError(f"not an s3:// URI: {uri}")
    return bucket, key.lstrip("/")


def _client(settings: Settings):
    import boto3

    return boto3.client("s3", region_name=settings.aws_region)


def _cache_path(uri: str, key: str, bucket: str, etag: str, settings: Settings) -> Path:
    """Where this version of the object is cached. S3 keys are untrusted, so '..' must not escape.

    The check works on the path text, not `Path.resolve`. That reads the disk, and on Windows its
    result changes form if another clip's cleanup removes a folder during the call, which made
    safe keys look unsafe.
    """
    # The ETag comes from the server, so it is hashed rather than trusted as a folder name.
    version = hashlib.sha256(etag.encode()).hexdigest()[:16]
    root = Path(os.path.normpath(settings.cache_dir / bucket / version))
    dest = Path(os.path.normpath(root / key))
    if dest == root or not dest.is_relative_to(root):
        raise S3Error(f"unsafe S3 key: {uri}")
    return dest


def head(uri: str, settings: Settings) -> ObjectInfo:
    """The object's ETag and size. Unsafe keys are refused before any request is made."""
    bucket, key = parse_s3_uri(uri)
    _cache_path(uri, key, bucket, "", settings)
    with _s3_errors(uri):
        reply = _client(settings).head_object(Bucket=bucket, Key=key)
    return ObjectInfo(str(reply.get("ETag", "")).strip('"'), int(reply["ContentLength"]))


def object_id(uri: str, info: ObjectInfo) -> str:
    """A video_id from the object's details, so nothing is downloaded just to name it."""
    bucket, key = parse_s3_uri(uri)
    return video_id_for(f"s3://{bucket}/{key}|{info.etag}|{info.size}")


def _too_big(uri: str, size: int, settings: Settings) -> S3Error:
    limit = settings.s3_max_download_mb
    return S3Error(
        f"{uri}: {size / 1e6:.0f} MB is over the {limit:g} MB limit (S3_MAX_DOWNLOAD_MB)"
    )


def check_size(uri: str, info: ObjectInfo, settings: Settings) -> None:
    if info.size > settings.s3_max_download_mb * 1e6:
        raise _too_big(uri, info.size, settings)


def fetch(uri: str, settings: Settings, info: ObjectInfo | None = None) -> tuple[Path, bool]:
    """Put one object in the cache. Returns its path and whether this call downloaded it."""
    info = info or head(uri, settings)
    check_size(uri, info, settings)
    bucket, key = parse_s3_uri(uri)
    dest = _cache_path(uri, key, bucket, info.etag, settings)
    if dest.is_file() and dest.stat().st_size == info.size:
        return dest, False

    with _CACHE_LOCK:
        dest.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=dest.parent, prefix=f"{dest.name}.", suffix=".part")
    os.close(handle)
    tmp = Path(name)
    try:
        with _s3_errors(uri):
            _client(settings).download_file(bucket, key, str(tmp))
        if tmp.stat().st_size > settings.s3_max_download_mb * 1e6:  # it changed after the check
            raise _too_big(uri, tmp.stat().st_size, settings)
        tmp.replace(dest)
    finally:
        with _CACHE_LOCK:
            tmp.unlink(missing_ok=True)
    return dest, True


def download(uri: str, settings: Settings, info: ObjectInfo | None = None) -> Path:
    return fetch(uri, settings, info)[0]


def remove_cached(path: Path, settings: Settings) -> None:
    """Delete a cached file and any folders it leaves empty. Every object has its own folder."""
    root = Path(os.path.abspath(settings.cache_dir))
    with _CACHE_LOCK:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete %s: %s", path, redact(exc))
            return
        for folder in Path(os.path.abspath(path)).parents:
            if folder == root or not folder.is_relative_to(root):
                break
            try:
                folder.rmdir()
            except OSError:  # not empty: another object still needs it
                break


def list_prefix(uri: str, settings: Settings) -> Iterator[str]:
    """Video URIs under a prefix, page by page, in S3's own (sorted) key order."""
    bucket, prefix = parse_s3_uri(uri)
    with _s3_errors(uri):
        paginator = _client(settings).get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(config.VIDEO_EXTS):
                    yield f"s3://{bucket}/{obj['Key']}"
