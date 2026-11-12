#!/usr/bin/env python3
"""Score ego-centric videos with a VLM and route them to quality tiers (Part A).

Examples
  python score_videos.py clip.mp4                              # one video, provider from .env
  python score_videos.py data/synthetic --workers 8 -o out.json
  python score_videos.py s3://bucket/2026/11/06/ --provider gemini
  python score_videos.py @videos.txt --provider vllm           # one path or S3 URI per line,
                                                               # or a `path,video_id` CSV
  python score_videos.py data/synthetic --provider offline     # no model: OpenCV baseline only

Output: a JSON object for one input, a JSON list for several. Every record has the required
keys from the assignment; `meta` holds provenance (add --no-meta to drop it).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.cli import add_decode_option, add_download_options, exits_cleanly, prepare_output
from src.config import get_settings
from src.scorer.dedup import HashRegistry
from src.scorer.prompts import PROMPT_VERSION, VERSIONS
from src.scorer.provider_errors import AUTH_SKIP_MESSAGE
from src.scorer.quality_scorer import QualityScorer, iter_inputs

TIERS = ("usable", "borderline", "reject")


@exits_cleanly
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLM quality scorer for ego-centric video")
    parser.add_argument(
        "inputs", nargs="+", help="video files, folders, s3:// URIs or prefixes, @list.txt"
    )
    parser.add_argument(
        "--provider", choices=["gemini", "openai", "vllm", "offline"], help="overrides VLM_PROVIDER"
    )
    parser.add_argument("--model", help="overrides the provider's default model")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, choices=sorted(VERSIONS))
    parser.add_argument("--workers", type=int, default=4, help="parallel videos (default 4)")
    add_download_options(parser)
    add_decode_option(parser)
    parser.add_argument(
        "--hash-registry", help="JSON file to persist perceptual hashes across runs"
    )
    parser.add_argument("-o", "--out", help="write JSON here instead of stdout")
    parser.add_argument("--no-meta", action="store_true", help="output only the required keys")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings(args.provider, args.model)
    prepare_output(args.out, args.hash_registry)
    registry = HashRegistry(args.hash_registry)
    scorer = QualityScorer(
        settings,
        args.prompt_version,
        registry,
        args.workers,
        download_workers=args.download_workers,
        keep_cache=args.keep_cache,
        decode_timeout=args.decode_timeout,
    )
    videos = iter_inputs(args.inputs, settings)

    start = time.perf_counter()
    results = scorer.score_batch(videos)
    elapsed = time.perf_counter() - start
    if not results:
        print("error: no videos found", file=sys.stderr)
        return 2
    if args.no_meta:
        for r in results:
            r.pop("meta", None)

    payload = results[0] if len(results) == 1 else results
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    registry.save()  # after the results, so a registry problem cannot lose them

    ok = [r for r in results if "error" not in r]
    tiers = ", ".join(f"{sum(r.get('tier') == t for r in ok)} {t}" for t in TIERS)
    print(
        f"scored {len(ok)}/{len(results)} videos in {elapsed:.1f}s "
        f"with {settings.provider}:{settings.model}: {tiers}",
        file=sys.stderr,
    )
    for r in results:
        if "error" in r and r["error"] != AUTH_SKIP_MESSAGE:
            print(f"  ERROR {r['source']}: {r['error']}", file=sys.stderr)
    if scorer.auth_failure:
        skipped = sum(r.get("error") == AUTH_SKIP_MESSAGE for r in results)
        print(f"error: {scorer.auth_failure} ({skipped} videos skipped)", file=sys.stderr)
        return 2
    if scorer.input_failure:
        print(f"error: {scorer.input_failure}; later inputs were not scored", file=sys.stderr)
        return 2
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
