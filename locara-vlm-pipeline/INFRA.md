# Infrastructure and scale (Part C)

All figures come from `scripts/cost_model.py`, which writes [cost_model_2M_hours.csv](cost_model_2M_hours.csv). Every input is listed there with its source or marked as an assumption. I checked prices on 2026-11-06. To change a figure, change the input and re-run the script, then copy the new output here.

The scale comes from the brief: 880,000 videos a day (26.4 M a month), 12 frames per video, and 30% sent to human review, which is 264,000 videos a day.

I send one request per video. The brief's "10.5 M VLM calls/day" counts frames. This design sends all 12 frames of a video in one request, so it is 880 K requests a day. The tokens are the same; the overhead is much lower.

## Q8. API versus self-hosted GPUs

Tokens per video:

| Model | Image tokens | Prompt text | Output | Total input |
|---|---|---|---|---|
| Gemini | 12 × 258 (one 768-px tile per 512-px frame) | 1,033 | 450 (estimate) | 4,129 |
| GPT-4o | 12 × 255 (high detail; verify with OpenAI's token counter) | 1,033 | 450 (estimate) | 4,093 |

I measured the prompt-text figure from the v3 prompt. Real token counts are logged in `meta.usage`.

Monthly inference cost at 26.4 M videos:

| Option | Price basis | Per video | Per month |
|---|---|---|---|
| GPT-4o, standard (successor of GPT-4V named in the brief) | $2.50 / $10.00 per 1M tokens | $0.0147 | **$388,938** |
| GPT-4o, Batch API (50%) | half | $0.0074 | $194,469 |
| Gemini 2.5 Flash, standard | $0.30 / $2.50 | $0.0024 | $62,402 |
| Gemini 2.5 Flash, batch | half | $0.0012 | $31,201 |
| Gemini 2.5 Flash-Lite, batch | $0.05 / $0.20 | $0.0003 | $7,826 |
| Self-hosted Qwen2.5-VL-7B, 15 × g6.2xlarge (1 × NVIDIA L4 each) | $0.842/h blended (70% spot at $0.7838, 30% on-demand at $0.9776) | $0.0003 | **$9,219** |

GPU sizing works like this: 880,000 / 86,400 = 10.2 videos per second. Assuming 1.0 video per second per L4 at 70% utilisation, that is ceil(10.2 / 0.7) = 15 GPUs, and 15 × 730 h × $0.842 = $9,219 a month.

The throughput figure is an assumption, not a measurement. At 2.0 videos/s it is 8 GPUs ($4,917); at 0.5, 30 GPUs ($18,438); at 0.25, 59 GPUs ($36,262). I would measure it with `scripts/benchmark_vllm.py` against vLLM on one g6.2xlarge before committing.

For break-even, the self-hosted floor is 2 GPUs always on, for availability: 2 × 730 h × $0.9776 = $1,427 a month. That floor handles about 120 K videos a day at 1 video/s, so below that volume self-hosting costs a flat $1,427.

| Compared with | Break-even volume |
|---|---|
| GPT-4o standard | 3,229 videos/day (about 40 hours of video a day) |
| Gemini 2.5 Flash batch | 40,256 videos/day (about 500 hours a day) |

At the programme's 880 K a day, self-hosting is cheaper than every API option except Gemini Flash-Lite batch, which comes in a little under it ($7.8 K versus $9.2 K).

The table leaves out the engineering and on-call time to run GPUs; adding one part-time MLOps engineer moves the break-even well above these numbers.

My recommendation:

1. Launch on the Gemini Flash batch API: no GPU work while the Tangible deal is pending.
2. Validate accuracy on real clips.
3. Move steady-state volume to self-hosted Qwen2.5-VL once accuracy matches Flash on the labelled set.
4. Keep the API for spikes and for second opinions on borderline clips.

## Q9. Orchestration

```
S3 upload ─► S3 event ─► SQS "to-score" (DLQ after 5 tries)
                              │
                    CPU workers (ECS/EKS, spot): download, sample frames, telemetry, dHash, Redis dup check
                              │
                     SQS "to-infer" (frames + telemetry in S3, pointer in message)
                              │
                    GPU workers: vLLM on an EC2 Auto Scaling group of g6 instances (or the Gemini batch API)
                              │
                    validate + route ─► Postgres (state, tiers) + S3 Parquet (full records)
                              │
              SQS/SNS fan-out ─► payout service · annotation queue · delivery manifest builder
```

| Option | Verdict |
|---|---|
| SQS + Lambda | Good for the CPU stage glue, but Lambda has no GPU and a 15-minute limit, so it cannot host the model |
| Celery + Redis | Works, but Redis as the only broker is a durability risk at 880 K jobs a day, and backpressure is manual |
| Ray Serve / Ray Data | Strong for batching on GPUs, but more to operate; worth it if the team already runs Ray |
| Airflow | Batch scheduling, not per-video streaming; use it for nightly reports and dataset releases only |
| **Chosen: SQS + container workers, with vLLM on GPU instances** | Durable queues, retries and DLQs come with SQS; workers are stateless; vLLM does continuous batching on each GPU |

Queues provide the backpressure. They absorb spikes: if evening uploads triple, messages wait instead of failing. Each stage scales on its own queue depth and age, through target tracking on the Auto Scaling group.

GPU workers pull only as many messages as they have batch slots. When the queue age passes a service level (for example 2 hours), the overflow is routed to the Gemini batch API instead of buying more GPUs. Spot interruptions return in-flight messages to the queue through visibility timeouts. Scoring is idempotent on `video_id`, so a retry never creates a second record.

## Q10. Storage and data contracts

Where data lives:

| Data | Store | Why |
|---|---|---|
| Video state and tiers; payout ledger | Postgres (RDS, Multi-AZ), indexed on `video_id`, `(worker_id, processed_at)`, `(tier, processed_at)` | Transactions, joins with payouts |
| Full score records with reasoning and telemetry | S3 as Parquet, partitioned by `date` and `tier`, queried with Athena | Cheap, immutable audit trail; about 2.2 KB per video, 58 GB a month |
| Perceptual hashes for dedup | Redis (hot 30 days) and Postgres (all) | Fast lookups at ingest |
| Keyframes (12 per video, 512 px) | S3 Standard for 30 days, then Glacier Instant Retrieval | Review and audits without re-decoding video; about 9.5 TB a month |

Each contract is a versioned, append-only event; the Pydantic models are in `src/schemas/data_contracts.py`.

`payout.v1` goes to the worker payment system. Its fields are `video_id`, `worker_id`, tier, tier confidence, overall usability, duplicate flag, and `payout_band` (`quality_bonus`, `base`, `hold_pending_review`, `none`), with reason codes. Duplicates never pay. Borderline clips are held until review, so pay never depends on an uncertain model call. Consumers deduplicate on `event_id`.

`annotation_task.v1` goes to the annotation queue. Its fields are video URI, pre-label URI, taxonomy version, review reason, and priority, which is (1 − tier confidence) × 100. The least certain clips go first, because their corrections teach the scorer the most.

`delivery.v1` goes to the robotics client portal, as one line per clip in a versioned release manifest: release, video URI, SHA-256, duration, the 7 scores, the annotation URI, the format (`coco_video_ext.v1`), and the label status (`human_verified` or `auto_accepted`). A release is immutable once published.

The compatibility rule is simple. New optional fields are a minor version. Renaming or removing a field needs a new major version on a parallel topic, so no consumer breaks on deploy.

## Q11. Cost model (CSV)

`cost_model_2M_hours.csv` holds monthly and six-month totals per line, three scenario totals, a lever analysis, sensitivity tables and the full list of assumptions.

| Monthly cost (raw video storage excluded) | Gemini Flash batch | GPT-4o | Self-hosted Qwen |
|---|---|---|---|
| Inference | $31,201 | $388,938 | $9,219 |
| Scores and keyframes on S3 (six-month average) | $582 | $582 | $582 |
| SQS, CPU pre-processing, Postgres and Redis | $2,265 | $2,265 | $2,265 |
| Human review: 264 K videos/day × 30 s at $3/h (2,200 h/day) | $198,000 | $198,000 | $198,000 |
| **Total** | **$232,048** | **$589,785** | **$210,067** |

The biggest lever is human review. With inference self-hosted, a 10% cut in any of review share, seconds per clip or the annotator rate saves $19,800 a month. A 10% gain in GPU throughput saves $615.

The review cost moves with its assumptions:

| Review time and rate | Monthly cost |
|---|---|
| 30 s at $2/h | $132,000 |
| 60 s at $3/h | $396,000 |
| 120 s at $5/h | $1,320,000 |

So the work that pays back most is auto-accept thresholds that keep the review share low without hurting accepted-clip precision (Q5), and pre-labels good enough that review is a quick check, not relabelling.

Raw video storage is not in the totals. It belongs to the capture pipeline, but it dwarfs everything above. At about 50 MB per minute (the capture assignment's figure), the programme adds about 990 TB a month. Kept in S3 Standard, that is a $136,620 run-rate by month 6, and $478 K over six months. A lifecycle rule moving video to Glacier Instant Retrieval after 30 days ($0.004 versus $0.023 per GB-month) cuts that by about 80%. Deep Archive ($0.00099) suits clips already delivered to the client.

Prices I used, checked 2026-11-06:

- Gemini: third-party price tracker (the official page should be checked before submission).
- Gemini image tokens: Google's token documentation.
- GPT-4o: OpenAI's model page, which marks the model deprecated.
- EC2: live pricing pages for g6.2xlarge and g5.2xlarge in us-east-1. The g5 (A10G) is $1.212/h on demand, so the newer L4-based g6 is the better choice.
- S3 class prices: a 2026 pricing guide.

The SQS price, the CPU vCPU-hour rate, the Postgres and Redis estimate, review time and annotator rate are marked as assumptions in the CSV.
