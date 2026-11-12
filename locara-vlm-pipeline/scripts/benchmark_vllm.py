#!/usr/bin/env python3
"""Measure scorer throughput against a running model server (feeds Q8's GPU sizing).

Start vLLM on the GPU box first, for example:
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 16384 --limit-mm-per-prompt image=16

Then, from any machine that can reach it:
    VLLM_BASE_URL=http://<gpu-host>:8000/v1 python scripts/benchmark_vllm.py data/synthetic \
        --concurrency 8 16 32 --repeat 4

Prints videos/second and latency per concurrency level. Put the best sustained figure into
`videos_per_gpu_sec` in scripts/cost_model.py and re-run the cost model.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.cli import exits_cleanly
from src.config import get_settings
from src.schemas.scoring_schema import video_id_for
from src.scorer.dedup import HashRegistry
from src.scorer.provider_errors import VLMError
from src.scorer.quality_scorer import QualityScorer, resolve_inputs


@exits_cleanly
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--provider", default="vllm", choices=["vllm", "gemini", "openai", "offline"]
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeat", type=int, default=3, help="times each video is sent per level")
    args = parser.parse_args(argv)

    settings = get_settings(args.provider)
    base = resolve_inputs(args.inputs, settings)
    for c in args.concurrency:
        videos = [
            dataclasses.replace(v, video_id=video_id_for(f"{v.source}#{i}"))
            for i in range(args.repeat)
            for v in base
        ]
        scorer = QualityScorer(
            settings,
            registry=HashRegistry(),
            workers=c,
            keep_cache=True,
            decode_timeout=config.DECODE_TIMEOUT_S,
        )
        t0 = time.perf_counter()
        out = scorer.score_batch(videos)
        wall = time.perf_counter() - t0
        if scorer.auth_failure:
            raise VLMError(scorer.auth_failure)
        ok = [r for r in out if "error" not in r]
        lat = [r["meta"]["usage"]["latency_ms"] for r in ok if r["meta"]["usage"].get("latency_ms")]
        tin = [r["meta"]["usage"].get("input_tokens") or 0 for r in ok]
        tout = [r["meta"]["usage"].get("output_tokens") or 0 for r in ok]
        mean_in = statistics.mean(tin) if tin else 0
        mean_out = statistics.mean(tout) if tout else 0
        print(
            f"concurrency {c:3d}: {len(ok)}/{len(out)} ok, {len(ok) / wall:.2f} videos/s, "
            f"latency p50 {statistics.median(lat) if lat else 0:.0f} ms, "
            f"tokens in/out {mean_in:.0f}/{mean_out:.0f}"
        )
    print("note: this includes local decoding and sampling; run it on a machine with spare CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
