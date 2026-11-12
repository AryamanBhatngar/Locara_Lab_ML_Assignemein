#!/usr/bin/env python3
"""Cost model for the 2M-hour programme (Q8, Q11).

    python scripts/cost_model.py              # writes cost_model_2M_hours.csv, prints summary
    python scripts/cost_model.py --json       # summary as JSON (used by INFRA.md)

Every input is in ASSUMPTIONS with its source. Prices were checked on 2026-11-06; change them
here and re-run, nothing else is hard-coded. Quantities marked "measure" should be replaced with
numbers from your own runs (token usage is logged in each scored record's meta.usage).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ASSUMPTIONS = {
    # Volume (assignment, Part C)
    "videos_per_day": 880_000,
    "frames_per_video": 12,
    "review_share": 0.30,
    "days_per_month": 30,
    "months": 6,
    # Tokens per video
    "gemini_tokens_per_frame": 258,  # ai.google.dev/gemini-api/docs/tokens: one 768px tile
    "openai_tokens_per_frame": 255,  # 512 px frame, high detail: 85 base + 170 per tile (verify)
    "prompt_text_tokens": 1_033,  # v3 prompt incl. telemetry, ~4 chars/token (measure)
    "output_tokens": 450,  # 7 dimensions x short reasoning (measure)
    # API prices, USD per 1M tokens
    "gemini_flash_in": 0.30,
    "gemini_flash_out": 2.50,  # gemini-2.5-flash
    "gemini_flash_lite_in": 0.10,
    "gemini_flash_lite_out": 0.40,  # gemini-2.5-flash-lite
    "gpt4o_in": 2.50,
    "gpt4o_out": 10.00,  # gpt-4o (GPT-4V successor)
    "batch_discount": 0.50,  # Gemini batch halves rates; OpenAI Batch API 50% (verify)
    # Self-hosting Qwen2.5-VL-7B on g6.2xlarge (1x NVIDIA L4 24 GB), us-east-1
    "g6_on_demand_hr": 0.9776,  # doit.com live price page
    "g6_spot_hr": 0.7838,  # lowest current AZ spot price, same page
    "spot_share": 0.70,
    "videos_per_gpu_sec": 1.0,  # ASSUMPTION: measure with scripts/benchmark_vllm.py
    "gpu_utilisation": 0.70,  # headroom for spikes and spot interruptions
    "min_gpus_always_on": 2,  # high availability floor
    "hours_per_month": 730,
    # Storage, USD per GB-month (S3 us-east-1)
    "s3_standard_gb": 0.023,
    "s3_glacier_ir_gb": 0.004,
    "s3_deep_archive_gb": 0.00099,
    "s3_put_per_1k": 0.005,
    "result_kb_per_video": 2.2,  # measured: mean scored record incl. meta (sample_outputs)
    "keyframe_kb_per_video": 360.0,  # 12 JPEG frames at 512 px, ~30 KB each (audit copy)
    "raw_video_mb_per_sec": 50 / 60,  # FS assignment: ~50 MB per 60 s clip (context only)
    # Orchestration
    "sqs_per_million": 0.40,  # standard queue requests (verify)
    "sqs_requests_per_video": 3,  # send, receive, delete
    "cpu_seconds_per_video": 1.5,  # measured locally: decode + telemetry, 640x360 (1080p ~3x)
    "cpu_vcpu_hr": 0.04,  # ASSUMPTION: Graviton spot vCPU-hour
    "db_cache_month": 1_500.0,  # ASSUMPTION: RDS Multi-AZ + ElastiCache, get an AWS quote
    # Human review
    "review_seconds_per_video": 30,  # ASSUMPTION with pre-labels (try 60 and 120)
    "annotator_usd_hr": 3.00,  # ASSUMPTION: the brief leaves $X open (try 2 and 5)
}


def _api_costs(a: dict, videos_month: int) -> dict[str, float]:
    frames = a["frames_per_video"]
    gemini_in = frames * a["gemini_tokens_per_frame"] + a["prompt_text_tokens"]
    openai_in = frames * a["openai_tokens_per_frame"] + a["prompt_text_tokens"]
    r: dict[str, float] = {
        "videos_month": videos_month,
        "gemini_in_tokens": gemini_in,
        "openai_in_tokens": openai_in,
    }
    options = {
        "gemini_flash": (gemini_in, a["gemini_flash_in"], a["gemini_flash_out"], False),
        "gemini_flash_batch": (gemini_in, a["gemini_flash_in"], a["gemini_flash_out"], True),
        "gemini_flash_lite_batch": (
            gemini_in,
            a["gemini_flash_lite_in"],
            a["gemini_flash_lite_out"],
            True,
        ),
        "gpt4o": (openai_in, a["gpt4o_in"], a["gpt4o_out"], False),
        "gpt4o_batch": (openai_in, a["gpt4o_in"], a["gpt4o_out"], True),
    }
    for key, (tokens_in, price_in, price_out, batch) in options.items():
        per_video = (tokens_in * price_in + a["output_tokens"] * price_out) / 1e6
        per_video *= 1 - a["batch_discount"] if batch else 1
        r[f"{key}_per_video"] = per_video
        r[f"{key}_month"] = per_video * videos_month
    return r


def _self_hosting_costs(a: dict, r: dict[str, float]) -> dict[str, float]:
    videos_month = r["videos_month"]
    rate = a["spot_share"] * a["g6_spot_hr"] + (1 - a["spot_share"]) * a["g6_on_demand_hr"]
    per_sec = a["videos_per_day"] / 86_400
    gpus = max(
        a["min_gpus_always_on"],
        math.ceil(per_sec / (a["videos_per_gpu_sec"] * a["gpu_utilisation"])),
    )
    floor = a["min_gpus_always_on"] * a["hours_per_month"] * a["g6_on_demand_hr"]
    costs = {
        "gpu_blended_hr": rate,
        "gpus": gpus,
        "selfhost_month": gpus * a["hours_per_month"] * rate,
        "selfhost_per_video": gpus * a["hours_per_month"] * rate / videos_month,
        "selfhost_floor_month": floor,
    }
    for key in ("gemini_flash_batch", "gpt4o"):
        costs[f"breakeven_videos_day_vs_{key}"] = floor / (
            a["days_per_month"] * r[f"{key}_per_video"]
        )
    return costs


def _storage_costs(a: dict, videos_month: int) -> dict[str, float]:
    results_gb = a["result_kb_per_video"] * videos_month / 1e6
    keyframes_gb = a["keyframe_kb_per_video"] * videos_month / 1e6
    months = a["months"]
    # Results stay in Standard. Keyframes spend one month in Standard, then move to Glacier IR.
    results_total = sum(results_gb * i * a["s3_standard_gb"] for i in range(1, months + 1))
    keyframes_total = sum(
        keyframes_gb * a["s3_standard_gb"] + keyframes_gb * (i - 1) * a["s3_glacier_ir_gb"]
        for i in range(1, months + 1)
    )
    puts = videos_month * 2 / 1000 * a["s3_put_per_1k"]
    raw_tb = a["videos_per_day"] * a["days_per_month"] * 45 * a["raw_video_mb_per_sec"] / 1e6
    return {
        "results_gb_month": results_gb,
        "keyframes_gb_month": keyframes_gb,
        "storage_month_avg": (results_total + keyframes_total) / months + puts,
        "s3_puts_month": puts,
        "raw_video_tb_month": raw_tb,
        # Month-6 run-rate if everything stayed in Standard, then the 6-month cumulative bill.
        "raw_video_standard_month6": raw_tb * 1000 * months * a["s3_standard_gb"],
        "raw_video_standard_6mo_total": sum(
            raw_tb * 1000 * i * a["s3_standard_gb"] for i in range(1, months + 1)
        ),
    }


def _operating_costs(a: dict, videos_month: int) -> dict[str, float]:
    sqs = videos_month * a["sqs_requests_per_video"] / 1e6 * a["sqs_per_million"]
    cpu = videos_month * a["cpu_seconds_per_video"] / 3600 * a["cpu_vcpu_hr"] / 0.6  # 60% busy
    review_hours = a["videos_per_day"] * a["review_share"] * a["review_seconds_per_video"] / 3600
    return {
        "sqs_month": sqs,
        "cpu_month": cpu,
        "db_cache_month": a["db_cache_month"],
        "orchestration_month": sqs + cpu + a["db_cache_month"],
        "review_videos_day": a["videos_per_day"] * a["review_share"],
        "annotator_hours_day": review_hours,
        "annotator_month": review_hours * a["days_per_month"] * a["annotator_usd_hr"],
    }


def monthly(a: dict) -> dict[str, float]:
    videos_month = a["videos_per_day"] * a["days_per_month"]
    r = _api_costs(a, videos_month)
    r.update(_self_hosting_costs(a, r))
    r.update(_storage_costs(a, videos_month))
    r.update(_operating_costs(a, videos_month))
    for scenario in ("gemini_flash_batch", "gpt4o", "selfhost"):
        r[f"total_{scenario}_month"] = (
            r[f"{scenario}_month"]
            + r["storage_month_avg"]
            + r["orchestration_month"]
            + r["annotator_month"]
        )
    return r


def levers(a: dict) -> list[tuple[str, float]]:
    """Change in the self-hosted monthly total for a 10% improvement in each driver."""
    base = monthly(a)["total_selfhost_month"]
    out = []
    for key, direction in (
        ("review_share", -1),
        ("review_seconds_per_video", -1),
        ("annotator_usd_hr", -1),
        ("videos_per_gpu_sec", +1),
        ("keyframe_kb_per_video", -1),
    ):
        b = dict(a)
        b[key] = a[key] * (1 + 0.1 * direction)
        out.append((key, base - monthly(b)["total_selfhost_month"]))
    return sorted(out, key=lambda x: -x[1])


def rows(a: dict, r: dict) -> list[list]:
    months = a["months"]
    table = []

    def add(section, item, monthly_cost, basis, six=None):
        table.append(
            [
                section,
                item,
                round(monthly_cost, 2),
                round(monthly_cost * months if six is None else six, 2),
                basis,
            ]
        )

    add(
        "inference",
        "Gemini 2.5 Flash (standard)",
        r["gemini_flash_month"],
        f"{r['gemini_in_tokens']:.0f} in + {a['output_tokens']} out tokens/video "
        "at $0.30/$2.50 per 1M",
    )
    add(
        "inference", "Gemini 2.5 Flash (batch)", r["gemini_flash_batch_month"], "batch halves rates"
    )
    add(
        "inference",
        "Gemini 2.5 Flash-Lite (batch)",
        r["gemini_flash_lite_batch_month"],
        "$0.10/$0.40 per 1M, batch",
    )
    add(
        "inference",
        "GPT-4o (standard)",
        r["gpt4o_month"],
        f"{r['openai_in_tokens']:.0f} in + {a['output_tokens']} out tokens/video "
        "at $2.50/$10 per 1M",
    )
    add("inference", "GPT-4o (batch)", r["gpt4o_batch_month"], "Batch API 50% (verify)")
    add(
        "inference",
        f"Self-hosted Qwen2.5-VL-7B, {r['gpus']} x g6.2xlarge",
        r["selfhost_month"],
        f"{a['videos_per_gpu_sec']} video/s/GPU assumed, {a['gpu_utilisation']:.0%} utilisation, "
        f"{a['spot_share']:.0%} spot at ${r['gpu_blended_hr']:.3f}/h blended",
    )
    add(
        "storage",
        "Scores + keyframes on S3 (6-month average)",
        r["storage_month_avg"],
        f"results {r['results_gb_month']:.0f} GB/mo Standard; "
        f"keyframes {r['keyframes_gb_month']:.0f} GB/mo, "
        "Standard 1 month then Glacier IR",
    )
    add(
        "storage (context)",
        "Raw video kept in S3 Standard (month-6 run-rate)",
        r["raw_video_standard_month6"],
        f"{r['raw_video_tb_month']:.0f} TB added per month at ~50 MB/min; "
        "owned by the capture pipeline; six_month_usd is the cumulative bill; "
        "a lifecycle to Glacier IR after 30 days cuts it ~80%",
        six=r["raw_video_standard_6mo_total"],
    )
    add("orchestration", "SQS", r["sqs_month"], "3 requests/video at $0.40 per 1M")
    add(
        "orchestration",
        "CPU pre-processing (decode + telemetry)",
        r["cpu_month"],
        f"{a['cpu_seconds_per_video']} vCPU-s/video at ${a['cpu_vcpu_hr']}/vCPU-h, 60% busy",
    )
    add(
        "orchestration",
        "RDS Postgres + Redis",
        r["db_cache_month"],
        "assumption; replace with AWS calculator quote",
    )
    add(
        "human review",
        f"{r['review_videos_day']:,.0f} videos/day",
        r["annotator_month"],
        f"{a['review_seconds_per_video']} s/video, {r['annotator_hours_day']:,.0f} h/day "
        f"at ${a['annotator_usd_hr']}/h",
    )
    for s, label in (
        ("gemini_flash_batch", "Gemini Flash batch"),
        ("gpt4o", "GPT-4o"),
        ("selfhost", "self-hosted Qwen"),
    ):
        add(
            "TOTAL",
            f"Total with {label} (excl. raw video)",
            r[f"total_{s}_month"],
            "sum of the rows above",
        )
    return table


def write_csv(path: str, a: dict, r: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "item", "monthly_usd", "six_month_usd", "basis"])
        w.writerows(rows(a, r))
        w.writerow([])
        w.writerow(
            ["lever", "10% improvement in", "monthly_saving_usd", "", "self-hosted scenario"]
        )
        for key, saving in levers(a):
            w.writerow(["lever", key, round(saving, 2), "", ""])
        w.writerow([])
        w.writerow(
            ["sensitivity", "review_seconds x annotator_usd_hr", "annotator_monthly_usd", "", ""]
        )
        for secs in (30, 60, 120):
            for rate in (2.0, 3.0, 5.0):
                scenario = monthly(dict(a, review_seconds_per_video=secs, annotator_usd_hr=rate))
                label = f"{secs}s @ ${rate:.0f}/h"
                w.writerow(["sensitivity", label, round(scenario["annotator_month"], 2), "", ""])
        for throughput in (0.25, 0.5, 1.0, 2.0):
            scenario = monthly(dict(a, videos_per_gpu_sec=throughput))
            label = f"GPU throughput {throughput} video/s"
            cost = round(scenario["selfhost_month"], 2)
            w.writerow(["sensitivity", label, cost, "", f"{scenario['gpus']} GPUs"])
        w.writerow([])
        for key, value in a.items():
            w.writerow(["assumption", key, value, "", ""])


def print_summary(summary: dict) -> None:
    for key in (
        "gemini_flash_month",
        "gemini_flash_batch_month",
        "gpt4o_month",
        "gpt4o_batch_month",
        "gpus",
        "selfhost_month",
        "annotator_month",
        "total_selfhost_month",
        "breakeven_videos_day_vs_gpt4o",
        "breakeven_videos_day_vs_gemini_flash_batch",
    ):
        print(f"{key:45s} {summary[key]:,}")
    print("levers:", summary["levers"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost model for the 2M-hour programme")
    parser.add_argument("--out", default="cost_model_2M_hours.csv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = monthly(ASSUMPTIONS)
    write_csv(args.out, ASSUMPTIONS, r)
    summary = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    summary["levers"] = [(k, round(v, 2)) for k, v in levers(ASSUMPTIONS)]
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)
    print(f"wrote {Path(args.out).resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
