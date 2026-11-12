"""Scoring orchestrator, tier router and batch runner (Part A, Q5).

Each video moves through these stages, and different videos overlap:
  1. download, if it is on S3 (bounded pool)
  2. sample (decode, frames, telemetry, hashes), in a worker process with a time limit
  3. duplicate check, strictly in input order, so "the later copy" is the one flagged
  4. VLM call, then validation with Pydantic and routing to a tier
  5. delete the download, unless the cache is being kept

Only MAX_VIDEOS_IN_FLIGHT videos are between stages 1 and 5 at once, which bounds both the frames
in memory and the files on disk.

A failure on one video is recorded and the batch continues, with one exception: a rejected API
key would fail every clip, so the batch stops and the remaining clips are marked as skipped.
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import config
from ..config import Settings
from ..redact import redact
from ..schemas.scoring_schema import (
    DIMENSIONS,
    FIVE_POINT_DIMENSIONS,
    MAX_REASONING_CHARS,
    QualityScore,
    Scores,
    Tier,
    canonical_video_id,
    utc_now_iso,
    video_id_for,
    video_id_for_file,
)
from . import prompts, s3_io
from .decode_pool import DecodePool, DecodeTimeout
from .dedup import DuplicateMatch, HashRegistry
from .frame_sampler import SampledVideo, VideoReadError, sample_video
from .provider_errors import AUTH_SKIP_MESSAGE, AuthError, BadResponseError
from .vlm_client import BaseVLMClient, Reply, make_client, offline_scores

log = logging.getLogger(__name__)


class _SkipClip(Exception):
    """Another clip's call found the key rejected, so this clip is not sent."""


@dataclass
class VideoInput:
    source: str  # what the user passed (path or s3 URI)
    local_path: str  # where the bytes are on disk; set later for S3 objects
    video_id: str
    error: str | None = None  # set when the file could not be fetched
    info: s3_io.ObjectInfo | None = None  # set for S3 objects still to be downloaded
    downloaded: bool = False  # this run fetched the file, so it may delete it

    @property
    def origin(self) -> str:
        """Where the bytes came from, which tells a re-run from a re-upload in the registry."""
        return self.source if self.source.startswith("s3://") else str(Path(self.source).resolve())


@dataclass
class Routing:
    tier: Tier
    composite: float
    hard_gates_failed: list[str]
    tier_confidence: float


def composite_score(s: Scores) -> float:
    return round(sum(w * s.value(k) for k, w in config.COMPOSITE_WEIGHTS.items()), 3)


def hard_gate_failures(s: Scores) -> list[str]:
    failed = []
    if s.egocentric_framing.score < 0.5:
        failed.append("egocentric_framing")
    if s.duplicate_risk.score >= 0.5:
        failed.append("duplicate_risk")
    for k in ("hand_visibility", "motion_quality", "lighting_exposure"):
        if s.value(k) <= config.HARD_FAIL_MAX:
            failed.append(k)
    return failed


def tier_confidence(s: Scores, composite: float, disagreements: int = 0) -> float:
    """Mean model confidence over the routing dimensions, with penalties.

    It is reduced when the composite sits near a threshold, or when the model disagrees with the
    OpenCV measurements.
    """
    keys = list(config.COMPOSITE_WEIGHTS) + ["egocentric_framing"]
    confs = [getattr(s, k).confidence for k in keys if getattr(s, k).confidence is not None]
    base = sum(confs) / len(confs) if confs else 0.5
    margin = min(
        abs(composite - config.USABLE_MIN_COMPOSITE),
        abs(composite - config.BORDERLINE_MIN_COMPOSITE),
    )
    if margin < 0.25:
        base -= 0.10
    base -= 0.10 * disagreements
    return round(max(0.05, min(0.99, base)), 2)


def route(s: Scores, disagreements: int = 0) -> Routing:
    comp = composite_score(s)
    failed = hard_gate_failures(s)
    conf = tier_confidence(s, comp, disagreements)
    if failed:
        # A hard gate is a validity check, so confidence reflects the failing dimensions only.
        gate_conf = [
            getattr(s, k).confidence for k in failed if getattr(s, k).confidence is not None
        ]
        return Routing("reject", comp, failed, round(max(gate_conf), 2) if gate_conf else conf)
    if comp < config.BORDERLINE_MIN_COMPOSITE:
        return Routing("reject", comp, [], conf)
    if comp >= config.USABLE_MIN_COMPOSITE and conf >= config.USABLE_MIN_CONFIDENCE:
        return Routing("usable", comp, [], conf)
    return Routing("borderline", comp, [], conf)


def cv_disagreements(s: Scores, telemetry: dict[str, float]) -> list[str]:
    """Dimensions where the model and the measurement-only baseline differ by 2+ points."""
    base = offline_scores(telemetry)
    out = []
    for k in ("motion_quality", "lighting_exposure", "hand_visibility"):
        if abs(s.value(k) - base[k]["score"]) >= 2:
            out.append(k)
    return out


def _duplicate_note(match: DuplicateMatch | None) -> str:
    if match and match.is_duplicate:
        return (
            f"Duplicate signal: perceptual-hash distance {match.distance:.0f}/64 to an earlier "
            f"clip in this batch (threshold {config.DUPLICATE_MAX_HAMMING}); this is a near-copy."
        )
    return ""


Entry = tuple[str, str | None]  # an input, and the video_id supplied for it


def _is_s3_key(item: str) -> bool:
    return Path(item).suffix.lower() in config.VIDEO_EXTS


def _expand_table(listing: Path, rows: list[str], open_lists: tuple[Path, ...]) -> list[Entry]:
    """A list file with a `path,video_id` header. A blank video_id means derive one."""
    reader = csv.DictReader(rows, skipinitialspace=True)
    reader.fieldnames = [name.strip().lower() for name in reader.fieldnames or []]
    if "path" not in reader.fieldnames:
        raise ValueError(f"{listing}: the header needs a path column")
    entries: list[Entry] = []
    for row in reader:
        supplied = (row.get("video_id") or "").strip()
        try:
            video_id = canonical_video_id(supplied) if supplied else None
            entries += _expand((row["path"] or "").strip(), open_lists, video_id)
        except ValueError as exc:
            raise ValueError(f"{listing}: {exc}") from exc
    return entries


def _expand_list(item: str, open_lists: tuple[Path, ...]) -> list[Entry]:
    path = Path(item[1:])
    key = path.resolve()
    if key in open_lists:
        raise ValueError(f"list file includes itself: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    rows = [line for line in lines if line and not line.startswith("#")]
    if rows and "video_id" in [cell.strip().lower() for cell in rows[0].split(",")]:
        return _expand_table(path, rows, (*open_lists, key))
    return [entry for row in rows for entry in _expand(row, (*open_lists, key))]


def _expand(
    item: str, open_lists: tuple[Path, ...] = (), video_id: str | None = None
) -> list[Entry]:
    """Flatten @lists and check each input exists. Quick, so a typo fails before any paid work."""
    if item.startswith("@"):
        return _expand_list(item, open_lists)
    if item.startswith("s3://"):
        s3_io.parse_s3_uri(item)
        many = not _is_s3_key(item)
    else:
        path = Path(item)
        many = path.is_dir()
        if not many and not path.is_file():
            raise FileNotFoundError(f"input not found: {item}")
    if video_id and many:
        raise ValueError(f"a video_id needs a single video, not {item}")
    return [(item, video_id)]


class _InputStream:
    """Turns inputs into VideoInputs one at a time, keeping each video once.

    A video_id from the capture app wins. Without one, local files are named by their bytes and
    S3 objects by their bucket, key, ETag and size.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._seen: set[str] = set()

    def videos(self, entries: list[Entry]) -> Iterator[VideoInput]:
        for item, video_id in entries:
            if item.startswith("s3://"):
                yield from self._s3(item, video_id)
            else:
                yield from self._local(item, video_id)

    def _s3(self, item: str, video_id: str | None) -> Iterator[VideoInput]:
        for uri in [item] if _is_s3_key(item) else s3_io.list_prefix(item, self.settings):
            if uri not in self._seen:
                self._seen.add(uri)
                yield self._s3_object(uri, video_id)

    def _s3_object(self, uri: str, video_id: str | None) -> VideoInput:
        try:
            info = s3_io.head(uri, self.settings)
            s3_io.check_size(uri, info, self.settings)
        except OSError as exc:  # one bad object should not stop the batch
            failed_id = video_id or video_id_for(uri)
            return VideoInput(uri, "", failed_id, f"download failed: {redact(exc)}")
        return VideoInput(uri, "", video_id or s3_io.object_id(uri, info), info=info)

    def _local(self, item: str, video_id: str | None) -> Iterator[VideoInput]:
        path = Path(item)
        children = sorted(path.iterdir()) if path.is_dir() else [path]
        for child in children:
            if path.is_dir() and child.suffix.lower() not in config.VIDEO_EXTS:
                continue
            resolved = str(child.resolve())
            if resolved not in self._seen:
                self._seen.add(resolved)
                yield self._local_file(child, resolved, video_id)

    @staticmethod
    def _local_file(path: Path, resolved: str, video_id: str | None) -> VideoInput:
        try:
            return VideoInput(path.as_posix(), str(path), video_id or video_id_for_file(path))
        except OSError as exc:  # one unreadable file should not stop the batch
            failed_id = video_id or video_id_for(resolved)
            return VideoInput(
                path.as_posix(), str(path), failed_id, f"cannot read file: {redact(exc)}"
            )


def iter_inputs(items: list[str], settings: Settings) -> Iterator[VideoInput]:
    """Expand files, folders, @lists and s3:// URIs or prefixes into VideoInputs, as needed.

    Bad paths and list files raise here, straight away. Hashing and S3 requests happen as the
    result is consumed. Nothing is downloaded: the scorer does that, so it can overlap the work.
    """
    entries = [entry for item in items for entry in _expand(item)]
    return _InputStream(settings).videos(entries)


def resolve_inputs(items: list[str], settings: Settings) -> list[VideoInput]:
    return list(iter_inputs(items, settings))


def _short_error(exc: Exception) -> str:
    """One line for the output JSON. Pydantic's own message runs to many lines per field."""
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "reply"
        more = f" (and {exc.error_count() - 1} more)" if exc.error_count() > 1 else ""
        return redact(f"{where}: {first['msg']}{more}")
    return redact(exc)


def _validated_scores(raw: dict[str, Any], match: DuplicateMatch | None) -> Scores:
    raw = {k: raw[k] for k in DIMENSIONS if k in raw}
    for k in FIVE_POINT_DIMENSIONS:  # models sometimes return 3.0 or "3"
        if k in raw and isinstance(raw[k], dict) and raw[k].get("score") is not None:
            raw[k]["score"] = round(float(raw[k]["score"]))
    if match and match.is_duplicate:  # hash evidence overrides the model
        model_view = raw.get("duplicate_risk")
        note = model_view.get("reasoning", "") if isinstance(model_view, dict) else ""
        reasoning = (
            f"Perceptual hash matches clip {match.video_id} "
            f"(median Hamming {match.distance:.0f}/64). {note}"
        )
        raw["duplicate_risk"] = {
            "score": 1,
            "confidence": 0.95,
            "reasoning": reasoning[:MAX_REASONING_CHARS],
        }
    return Scores.model_validate(raw)


class QualityScorer:
    def __init__(
        self,
        settings: Settings,
        prompt_version: str = prompts.PROMPT_VERSION,
        registry: HashRegistry | None = None,
        workers: int = 4,
        client: BaseVLMClient | None = None,
        download_workers: int = config.DOWNLOAD_WORKERS,
        keep_cache: bool = False,
        decode_timeout: float | None = None,
        sampler: Callable[[str], SampledVideo] | None = None,
    ):
        """With no `decode_timeout`, videos decode in this process and a hang cannot be stopped."""
        self.settings = settings
        self.prompt_version = prompt_version
        # Not "registry or ...": an empty registry is falsy, and it must not be swapped out.
        self.registry = registry if registry is not None else HashRegistry()
        self.workers = max(1, workers)
        self.download_workers = max(1, download_workers)
        self.keep_cache = keep_cache
        self.decode_timeout = decode_timeout
        self.sampler = sampler
        self._decoders: DecodePool | None = None
        self.client = client or make_client(settings)
        self.auth_failure: str | None = None
        self.input_failure: str | None = None  # the input stream broke part way through
        self._download_slots = threading.Semaphore(self.download_workers)
        self._sample_slots = threading.Semaphore(self.workers)
        self._first_call = threading.Lock()
        self._key_accepted = threading.Event()

    def score_batch(self, videos: Iterable[VideoInput]) -> list[dict[str, Any]]:
        """Score a stream of videos and return their records in input order."""
        if self.decode_timeout is None:
            return self._run_batch(videos)
        with closing(DecodePool(self.decode_timeout, self._sampler())) as self._decoders:
            return self._run_batch(videos)

    def _sampler(self) -> Callable[[str], SampledVideo]:
        return self.sampler or sample_video

    def _run_batch(self, videos: Iterable[VideoInput]) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        in_flight = threading.Semaphore(max(config.MAX_VIDEOS_IN_FLIGHT, 2 * self.workers))
        started: queue.Queue[tuple[VideoInput, Future[Any]] | None] = queue.Queue()
        feed_errors: list[Exception] = []
        scored: list[Future[dict[str, Any]]] = []
        with (
            ThreadPoolExecutor(self.workers + self.download_workers) as preparing,
            ThreadPoolExecutor(self.workers) as scoring,
        ):
            feeder = threading.Thread(
                target=self._feed,
                args=(videos, started, in_flight, preparing, feed_errors),
                daemon=True,
            )
            feeder.start()
            while (item := started.get()) is not None:
                video, prepared = item
                # Waiting for the oldest video first is what keeps this check in input order.
                # Later videos that finished sooner just stay in the queue until their turn.
                sample = prepared.result()
                match = None
                if not isinstance(sample, Exception):
                    match = self.registry.check_and_add(video.video_id, sample.hashes, video.origin)
                scored.append(scoring.submit(self._finish, video, sample, match, in_flight))
            feeder.join()
        results = [future.result() for future in scored]
        if feed_errors:
            if not results:
                raise feed_errors[0]
            self.input_failure = redact(feed_errors[0])
        log.info("scored %d videos in %.1fs", len(results), time.perf_counter() - t0)
        return results

    def _feed(
        self,
        videos: Iterable[VideoInput],
        started: queue.Queue[tuple[VideoInput, Future[Any]] | None],
        in_flight: threading.Semaphore,
        preparing: ThreadPoolExecutor,
        errors: list[Exception],
    ) -> None:
        """Start videos as slots free up. Runs on its own thread, as input can block on S3."""
        try:
            for video in videos:
                in_flight.acquire()
                started.put((video, preparing.submit(self._prepare, video)))
        except Exception as exc:  # noqa: BLE001 - reported once the running videos have finished
            errors.append(exc)
        finally:
            started.put(None)

    def _prepare(self, v: VideoInput) -> SampledVideo | Exception:
        """Download and decode one video. Failures are returned, to be recorded in order."""
        if v.error:
            return VideoReadError(v.error)
        if self.auth_failure:
            return _SkipClip()
        if v.info is not None:
            try:
                with self._download_slots:
                    path, v.downloaded = s3_io.fetch(v.source, self.settings, v.info)
            except Exception as exc:  # noqa: BLE001 - recorded per video, the batch continues
                return VideoReadError(f"download failed: {redact(exc)}")
            v.local_path = str(path)
        with self._sample_slots:
            if self.auth_failure:
                return _SkipClip()
            try:
                decode = self._decoders.sample if self._decoders else self._sampler()
                return decode(v.local_path)
            except Exception as exc:  # noqa: BLE001 - recorded per video, the batch continues
                return exc

    def _finish(
        self,
        v: VideoInput,
        sample: SampledVideo | Exception,
        match: DuplicateMatch | None,
        in_flight: threading.Semaphore,
    ) -> dict[str, Any]:
        try:
            return self._score_sampled(v, sample, match)
        finally:
            if v.downloaded and not self.keep_cache:
                s3_io.remove_cached(Path(v.local_path), self.settings)
            in_flight.release()

    @staticmethod
    def _skipped(v: VideoInput) -> dict[str, Any]:
        return {"video_id": v.video_id, "source": v.source, "error": AUTH_SKIP_MESSAGE}

    def _call_model(self, s: SampledVideo, match: DuplicateMatch | None) -> Reply:
        if self.auth_failure:
            raise _SkipClip
        try:
            return self.client.score(s, self.prompt_version, _duplicate_note(match))
        except AuthError as exc:
            self.auth_failure = str(exc)
            raise

    def _ask_model(self, s: SampledVideo, match: DuplicateMatch | None) -> Reply:
        """One call at a time until the model answers, so a rejected key costs a single request.

        With parallel workers, a flag alone is too late: every worker would already be calling.
        """
        if not self._key_accepted.is_set():
            with self._first_call:
                if not self._key_accepted.is_set():
                    try:
                        reply = self._call_model(s, match)
                    except BadResponseError:  # a useless reply still proves the key works
                        self._key_accepted.set()
                        raise
                    self._key_accepted.set()
                    return reply
        return self._call_model(s, match)

    def _score_sampled(
        self, v: VideoInput, s: SampledVideo | Exception, match: DuplicateMatch | None
    ) -> dict[str, Any]:
        if isinstance(s, _SkipClip):
            return self._skipped(v)
        if isinstance(s, DecodeTimeout):
            return {"video_id": v.video_id, "source": v.source, "error": str(s)}
        if isinstance(s, Exception):
            return {
                "video_id": v.video_id,
                "source": v.source,
                "error": f"sampling failed: {redact(s)}",
            }
        try:
            return self._build_record(v, s, match)
        except _SkipClip:
            return self._skipped(v)
        except Exception as exc:  # noqa: BLE001 - one bad reply must not lose the whole batch
            return {
                "video_id": v.video_id,
                "source": v.source,
                "error": f"scoring failed: {_short_error(exc)}",
            }

    def _build_record(
        self, v: VideoInput, s: SampledVideo, match: DuplicateMatch | None
    ) -> dict[str, Any]:
        start = time.perf_counter()
        raw, usage = self._ask_model(s, match)
        scores = _validated_scores(raw, match)
        disagree = (
            [] if self.client.provider == "offline" else cv_disagreements(scores, s.telemetry)
        )
        r = route(scores, len(disagree))
        q = QualityScore(
            video_id=v.video_id,
            processed_at=utc_now_iso(),
            frame_count_sampled=s.frame_count,
            scores=scores,
            tier=r.tier,
            tier_confidence=r.tier_confidence,
            meta={
                "source": v.source,
                "provider": self.client.provider,
                "model": self.client.model,
                "prompt_version": self.prompt_version
                if self.client.provider != "offline"
                else None,
                "duration_sec": s.duration_sec,
                "frame_times_sec": [f.time_sec for f in s.frames],
                "composite_score": r.composite,
                "hard_gates_failed": r.hard_gates_failed,
                "cv_disagreements": disagree,
                "duplicate_of": match.video_id if match and match.is_duplicate else None,
                "nearest_hash_distance": None if match is None else match.distance,
                "telemetry": s.telemetry,
                "usage": usage,
                "timings_ms": {
                    **s.timings_ms,
                    "scoring": round((time.perf_counter() - start) * 1000, 1),
                },
            },
        )
        return q.to_output()
