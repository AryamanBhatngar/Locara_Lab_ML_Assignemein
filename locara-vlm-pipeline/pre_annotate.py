#!/usr/bin/env python3
"""Generate candidate annotations for scored videos (Part B).

  python pre_annotate.py --scores sample_outputs/scored_videos_batch.json \
      --coco sample_outputs/coco_annotations_sample.json \
      --label-studio sample_outputs/label_studio_tasks_sample.json

Each scored record must have `meta.source` pointing at the video (as written by
score_videos.py), or pass --videos to say where the files are. Rejected videos are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.annotation.exporters import to_coco, to_label_studio
from src.annotation.pre_annotator import PreAnnotator, SkipVideo, ensure_eligible
from src.cli import (
    add_cache_option,
    add_decode_option,
    exits_cleanly,
    load_records,
    prepare_output,
    write_json,
)
from src.config import Settings, get_settings
from src.redact import redact
from src.scorer import s3_io
from src.scorer.decode_pool import DecodePool
from src.scorer.frame_sampler import sample_video
from src.scorer.provider_errors import AUTH_SKIP_MESSAGE, AuthError
from src.scorer.vlm_client import make_client


def video_path(source: str, videos_dir: str | None) -> Path:
    """Find a record's video. With --videos, only files directly inside that folder are allowed."""
    if videos_dir:
        folder = Path(videos_dir).resolve()
        path = (folder / Path(source).name).resolve()
        if path.parent != folder:
            raise ValueError(f"{source!r} is outside the --videos folder")
        return path
    return Path(source)


@dataclass
class Run:
    annotator: PreAnnotator
    decoder: DecodePool
    settings: Settings
    videos_dir: str | None
    keep_cache: bool


def annotate_one(record: dict[str, Any], source: str, run: Run) -> tuple[Any, dict[str, Any]]:
    downloaded = False
    if source.startswith("s3://") and not run.videos_dir:
        path, downloaded = s3_io.fetch(source, run.settings)
    else:
        path = video_path(source, run.videos_dir)
    try:
        sampled = run.decoder.sample(str(path))
        annotation = run.annotator.annotate(str(path), record, sampled)
    finally:
        if downloaded and not run.keep_cache:
            s3_io.remove_cached(path, run.settings)
    return annotation, {
        "file_name": path.name,
        "width": sampled.width,
        "height": sampled.height,
        "fps": sampled.fps,
    }


def annotate_records(records: list[dict[str, Any]], run: Run):
    """Annotate each eligible record. Returns (items, skipped, failed, auth_failure)."""
    items, skipped, failed = [], [], []
    auth_failure = None
    for record in records:
        src = (record.get("meta") or {}).get("source") or record.get("source", "")
        try:
            ensure_eligible(record)  # before decoding: rejected clips need no frames
            if auth_failure:
                failed.append((Path(src).name, AUTH_SKIP_MESSAGE))
                continue
            items.append(annotate_one(record, src, run))
        except SkipVideo as exc:
            skipped.append((Path(src).name, str(exc)))
        except AuthError as exc:  # every other clip would fail the same way
            auth_failure = str(exc)
            failed.append((Path(src).name, auth_failure))
        except Exception as exc:  # noqa: BLE001 - one clip failing should not stop the rest
            failed.append((Path(src).name, redact(exc)))
    return items, skipped, failed, auth_failure


@exits_cleanly
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLM pre-annotation with confidence gating")
    parser.add_argument(
        "--scores", required=True, help="JSON from score_videos.py (object or list)"
    )
    parser.add_argument(
        "--videos", help="folder that holds the videos, if meta.source is not usable"
    )
    parser.add_argument("--provider", choices=["gemini", "openai", "vllm", "offline"])
    parser.add_argument("--model")
    parser.add_argument(
        "--threshold", type=float, help="pre-label confidence threshold (default 0.75)"
    )
    parser.add_argument("--coco", default="sample_outputs/coco_annotations_sample.json")
    parser.add_argument("--label-studio", default="sample_outputs/label_studio_tasks_sample.json")
    parser.add_argument("--raw", help="also write the gated pre-annotations as JSON")
    parser.add_argument(
        "--video-url-prefix",
        default="/data/local-files/?d=",
        help="prefix Label Studio uses to reach the videos",
    )
    add_cache_option(parser)
    add_decode_option(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    settings = get_settings(args.provider, args.model)
    threshold = args.threshold if args.threshold is not None else settings.confidence_threshold
    annotator = PreAnnotator(make_client(settings), threshold)
    records = load_records(args.scores)
    if args.videos and not Path(args.videos).is_dir():
        raise ValueError(f"--videos folder not found: {args.videos}")
    prepare_output(args.coco, args.label_studio, args.raw)

    with closing(DecodePool(args.decode_timeout, sample_video)) as decoder:
        run = Run(annotator, decoder, settings, args.videos, args.keep_cache)
        items, skipped, failed, auth_failure = annotate_records(records, run)

    if items or not failed:  # a run where everything failed must not blank earlier outputs
        write_json(args.coco, to_coco(items))
        write_json(args.label_studio, to_label_studio(items, args.video_url_prefix))
        if args.raw:
            write_json(args.raw, [annotation.model_dump() for annotation, _ in items])

    review = sum(annotation.needs_review for annotation, _ in items)
    print(
        f"pre-annotated {len(items)} videos ({review} need human verification), "
        f"skipped {len(skipped)}, failed {len(failed)}; threshold {threshold}",
        file=sys.stderr,
    )
    for name, why in skipped:
        print(f"  skipped {name}: {why}", file=sys.stderr)
    for name, why in failed:
        print(f"  FAILED {name}: {why}", file=sys.stderr)
    if auth_failure:
        print(f"error: {auth_failure}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
