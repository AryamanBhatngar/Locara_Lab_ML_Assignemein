# Locara Labs — VLM-Assisted Video QC & Robotics Pre-Annotation Pipeline

**Author:** Aryaman Bhatnagar  
**Role Track:** ML Engineer Technical Assignment (`locaralabs.ai`)  
**Stack:** Python 3.10+ · OpenCV · Pydantic v2 · Gemini 1.5 Pro/Flash · OpenAI GPT-4o · Qwen2-VL (vLLM) · Boto3 (AWS S3) · Pytest

> **Submission Note (What I Prioritized & What I'd Improve Given More Time)**  
> For this submission, I prioritized building an end-to-end pipeline that doesn't just treat a Vision-Language Model as a black-box oracle, but grounds it with fast, deterministic OpenCV telemetry (`<0.4s` pre-filtering for Laplacian blur, optical flow, HSV exposure, and 64-bit perceptual hashing) before a single token is spent. I focused heavily on production reliability—enforcing strict Pydantic v2 contracts across all 7 scoring dimensions, implementing a two-stage hard-gate + weighted composite router for worker payout tiers, and proving out the system on a 10-clip benchmark alongside a realistic 2M-hour AWS GPU cost model. Given another week, I would replace the 2D HSV hand-presence heuristic with a lightweight 3D hand-mesh estimator (`HaMeR` or `MediaPipe Hands`) to extract explicit wrist/finger trajectories for robotics policies, switch CPU frame decoding to GPU-accelerated `NVDEC` (`decord`/`PyNvVideoCodec`), and wire up an automated nightly script that converts Label Studio human diffs directly into LoRA fine-tuning pairs for `Qwen2-VL-7B`.

## Why This Pipeline Exists

Collecting **2,000,000 hours of ego-centric (first-person) video** across thousands of field workers in India over 6 months means ingesting roughly **11,000 hours of footage every single day** (~880,000 clips/day at a 45-second average length). Manual human frame-by-frame Quality Control (QC) hits a wall after a few hundred hours per day. Worse, if a worker records 20 clips with a loose headband, a covered lens, or in a dark room, waiting days for human QC means paying for unusable data or losing worker trust.

This repository implements the two core workstreams required to run Locara Labs' collection program at scale for downstream robotics training (such as **RT-2**, **Octo**, and **OpenVLA**):

1. **Workstream A — Automated VLM Quality Scorer (`score_videos.py`)**: Accepts a local video path, folder, or `s3://` URI, samples keyframes intelligently, pairs visual frames with objective OpenCV signal telemetry, evaluates 7 robotics-critical quality dimensions in `<60 seconds` per clip, and routes each clip into `usable`, `borderline`, or `reject` tiers.
2. **Workstream B — Robotics Pre-Annotation Layer (`pre_annotate.py`)**: Takes clips that pass QC (`usable` and `borderline`), generates candidate labels across a 5-axis robotics manipulation taxonomy (atomic action primitives, manipulated objects, environment types, bimanual hand states, and viewpoint occlusion), gates them by confidence (`>= 0.75`), and exports ready-to-ingest **COCO JSON** (for model training) and **Label Studio JSON-LD tasks** (for human-in-the-loop verification).

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        UPSTREAM WORKER VIDEO INGESTION                            |
|         Local MP4 / Directory   OR   AWS S3 URI (s3://locara-raw-videos/...)      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| WORKSTREAM A: HYBRID VLM QUALITY SCORER (score_videos.py | <60s per clip)         |
|                                                                                   |
|  1. Adaptive Temporal Sampler (src/scorer/frame_sampler.py)                       |
|     - Divides clip into K windows (8 frames @ <=10s | 12 frames @ 30-60s)         |
|     - Picks sharpest local keyframe per window + scene-change peaks               |
|                                                                                   |
|  2. Fast OpenCV Telemetry Pre-Filter (<0.4s CPU execution)                        |
|     - Blur: Variance of Laplacian        - Shake/Static: Frame diff & optical flow|
|     - Exposure: HSV V-channel clipping   - Dedup: 64-bit dHash/pHash registry     |
|     - Hand ROI: Lower-frame active manipulation zone ratio                        |
|                                                                                   |
|  3. Multimodal VLM Evaluation (src/scorer/vlm_client.py)                          |
|     - 512x512 timestamped frame grid + CV telemetry injected into prompt          |
|     - Supports Gemini 1.5 Pro/Flash, GPT-4o, Qwen2-VL (vLLM), + Offline CV mode   |
|     - Enforces 7-dimension JSON output via Pydantic v2 (src/schemas/)             |
|                                                                                   |
|  4. Deterministic Hard-Gate + Weighted Composite Router                           |
|     - Hard Reject if: egocentric_framing==0, duplicate_risk==1, or severe defect  |
|     - S_comp = 0.30(Hand) + 0.25(Motion) + 0.20(Light) + 0.15(Scene) + 0.10(Use)  |
+-------------------+---------------------+---------------------+-------------------+
                    |                     |                     |
         tier = "usable"       tier = "borderline"       tier = "reject"
      (S >= 3.75, conf >= 0.8)  (2.50 <= S < 3.75)         (S < 2.50 or Gate)
                    |                     |                     |
                    v                     v                     v
      +-------------+---------------------+-----+     +---------+-------------------+
      | WORKSTREAM B: PRE-ANNOTATION ENGINE     |     | WORKER FEEDBACK & PAYOUT    |
      | (pre_annotate.py)                       |     | - Reject clip immediately   |
      | - 5-Axis Robotics Taxonomy generation   |     | - Prompt worker to re-shoot |
      | - Confidence Gating (threshold >= 0.75) |     +-----------------------------+
      +-------------+---------------------+-----+
                    |                     |
         Auto-Accepted / High Conf        | Needs Human Verification (~30% volume)
                    |                     v
                    |           +---------+-----------------------------------------+
                    |           | HUMAN-IN-THE-LOOP (Label Studio / CVAT Queue)     |
                    |           | - Annotators verify/adjust pre-populated labels   |
                    |           | - Disagreements resolved via tIoU & Kappa/Alpha   |
                    |           | - Corrections logged to Active Learning Loop      |
                    |           +---------+-----------------------------------------+
                    |                     |
                    v                     v
+-----------------------------------------------------------------------------------+
|                       DOWNSTREAM CONSUMERS & DATA CONTRACTS                       |
|  1. Robotics Client Portal (COCO JSON + RLDS episodes + SHA-256 verified manifest)|
|  2. Worker Payment Tier Ledger (High quality = bonus tier payout multiplier)      |
+-----------------------------------------------------------------------------------+
```

## Repository Structure

```text
locara-vlm-pipeline/
├── README.md                              # Project overview, setup, CLI guide, and architecture summary
├── DESIGN_DOC.md                          # In-depth engineering answers to Q1–Q7 + non-ML stakeholder brief
├── INFRA.md                               # 2M-hour scale architecture & math answering Q8–Q11
├── cost_model_2M_hours.csv                # Bonus Q11 spreadsheet modeling API vs GPU, S3, and annotator OPEX
├── requirements.txt                       # Pinned Python dependencies
├── pyproject.toml                         # Package metadata and pytest configuration
├── .env.example                           # Template for API keys, S3 bucket, and routing thresholds
├── score_videos.py                        # CLI entry point for single-clip or parallel batch QC scoring
├── pre_annotate.py                        # CLI entry point for generating COCO & Label Studio pre-annotations
├── evaluate_pipeline.py                   # CLI script running the 10-clip benchmark & Precision/Recall report
├── src/
│   ├── config.py                          # Central settings, dimension weights, and threshold constants
│   ├── schemas/
│   │   ├── scoring_schema.py              # Pydantic v2 models enforcing the required Part A JSON output
│   │   ├── taxonomy_schema.py             # Pydantic v2 models for the 5-axis Part B robotics taxonomy
│   │   └── data_contracts.py              # Downstream contracts (worker payout, HITL queue, client manifest)
│   ├── scorer/
│   │   ├── frame_sampler.py               # 3-stage adaptive frame sampler + OpenCV telemetry extractor
│   │   ├── prompts.py                     # Production VLM prompts + v1 -> v2 -> v3 prompt iteration log
│   │   ├── vlm_client.py                  # Pluggable client (Gemini, OpenAI, Qwen2-VL vLLM, Offline CV fallback)
│   │   └── quality_scorer.py              # Scoring engine, hard-gate + weighted tier router, async batch runner
│   └── annotation/
│       ├── pre_annotator.py               # Candidate action, object, scene, hand, and viewpoint label generator
│       ├── exporters.py                   # COCO JSON and Label Studio JSON-LD task exporters
│       └── label_studio_interface.xml     # Ready-to-use Label Studio XML template for ego-centric review
├── sample_outputs/
│   ├── scored_videos_batch.json           # Complete Part A JSON outputs + reasoning traces on 10 test clips
│   ├── precision_recall_report.json       # Per-dimension Precision, Recall, F1, and confusion breakdown
│   ├── coco_annotations_sample.json       # Sample COCO-compatible dataset output for robotics training
│   └── label_studio_tasks_sample.json     # Sample Label Studio pre-annotated tasks above 0.75 confidence
└── tests/
    ├── generate_synthetic_clips.py        # Synthesizes 10 test MP4s covering specific ego-centric failure modes
    ├── test_frame_sampler.py              # Unit tests for 10s vs 60s sampling and OpenCV signal calculation
    ├── test_scoring_schema.py             # Unit tests for schema bounds, hard gates, and tier routing
    └── test_pre_annotation.py             # Unit tests for taxonomy validation, confidence gating, and exports
```

## Quickstart & Native Execution Guide

This project is built as a **100% standalone native Python codebase**. Moreover, so you can test the entire pipeline immediately upon cloning without needing to configure cloud credentials or spend API tokens first, `vlm_client.py` includes a **calibrated deterministic OpenCV + heuristic inference mode (`--provider offline`)** alongside live **Gemini 1.5 Pro/Flash**, **OpenAI GPT-4o**, and **Qwen2-VL (vLLM)** adapters.

### 1. Environment Setup

```bash
# 1. Clone the repository and enter the directory
git clone <your-repo-url>
cd locara-vlm-pipeline

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the environment config template
cp .env.example .env
```

To run against a live VLM API, add your key to `.env` (for example, `GEMINI_API_KEY=your_key_here` and `VLM_PROVIDER=gemini`). If no API key is set, the pipeline automatically informs you and falls back to the calibrated local OpenCV + heuristic engine so every CLI command and test runs out of the box.

### 2. Generate the 10-Clip Test Dataset (Optional if already generated)

To make evaluation reproducible without bundling large binary videos in Git, run the synthetic clip generator. It uses OpenCV to build 10 realistic ego-centric `.mp4` test clips (`tests/data/clips/`) with a paired `ground_truth.json` covering clean bimanual manipulation, severe motion blur, dark underexposure, static non-egocentric framing, missing hands, and re-encoded duplicates:

```bash
python3 tests/generate_synthetic_clips.py
```

### 3. Run Part A — Score a Single Video or Batch Directory

**Score a single local video (or S3 URI):**

```bash
python3 score_videos.py \
  --input tests/data/clips/clip_01_kitchen_pour_ideal.mp4 \
  --output sample_outputs/single_score.json
```

**Score an S3 URI directly:**

```bash
python3 score_videos.py \
  --input s3://locara-raw-videos-prod/workers/w_1042/2026-11-06/550e8400-e29b-41d4-a716-446655440000.mp4 \
  --provider gemini
```

**Run parallel batch mode across a directory of clips (`<60s` per clip target):**

```bash
python3 score_videos.py \
  --batch-dir tests/data/clips/ \
  --workers 4 \
  --output sample_outputs/scored_videos_batch.json
```

### 4. Run Part B — Generate Robotics Pre-Annotations (COCO & Label Studio)

Pass the video directory (or single clip) and the Part A scoring JSON to generate candidate annotations across the 5-axis robotics taxonomy, gated by `--threshold 0.75`:

```bash
python3 pre_annotate.py \
  --video-dir tests/data/clips/ \
  --scores-json sample_outputs/scored_videos_batch.json \
  --threshold 0.75 \
  --coco-output sample_outputs/coco_annotations_sample.json \
  --label-studio-output sample_outputs/label_studio_tasks_sample.json
```

### 5. Run the Precision / Recall Evaluation Suite & Unit Tests

```bash
# Compute per-dimension Precision, Recall, F1, and failure mode analysis across the 10 clips
python3 evaluate_pipeline.py \
  --clips-dir tests/data/clips/ \
  --ground-truth tests/data/ground_truth.json \
  --output sample_outputs/precision_recall_report.json

# Run the automated pytest suite
pytest -v
```

## Key Engineering Highlights & Design Decisions

Full mathematical derivations and trade-off discussions live in `DESIGN_DOC.md` (`Q1–Q7`) and `INFRA.md` (`Q8–Q11`), but here is a quick summary of the core architectural choices:

### 1. Why a Hybrid OpenCV + VLM Sampler Beats Pure VLM Sampling (`Q1` & `Q3`)
When I tested my initial prompt (`v1`) by sending raw uniformly sampled frames to a VLM, two failure modes showed up immediately:
* **False-positive motion quality**: A clip with jarring camera shake often contains a few sharp frames when the wearer briefly pauses. Looking only at static images, the VLM rated `motion_quality = 4` on clips with unusable optical flow.
* **Missed re-encoded duplicates**: WhatsApp/re-uploaded duplicates fail exact `SHA-256` checks, and single-clip VLM calls have no memory of prior clips in the batch.

To fix this in `v3`, `frame_sampler.py` runs a `<0.4s` OpenCV pass over the video **before** calling the VLM, extracting Variance of the Laplacian ($\sigma^2_{\text{Lap}}$), frame-to-frame optical flow/pixel diff ($\Delta_t$), HSV V-channel clipping ratios ($r_{\text{under}}, r_{\text{over}}$), lower-frame hand ROI activity, and a 64-bit perceptual `dHash`. Injecting those exact numbers into the VLM prompt gives the model objective physical anchors while letting the VLM do what it's best at: semantic reasoning over hands, objects, and ego-centric framing.

### 2. Sample Output Schema (`Part A`)

Every scored video strictly conforms to the required JSON schema (`src/schemas/scoring_schema.py`):

```json
{
  "video_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "processed_at": "2026-11-12T12:30:15Z",
  "frame_count_sampled": 12,
  "scores": {
    "egocentric_framing": {
      "score": 1,
      "reasoning": "First-person head-mounted perspective looking down at workbench; natural head pan (mean flow 6.4 px/frame) with hands entering from lower periphery."
    },
    "motion_quality": {
      "score": 5,
      "reasoning": "Sharp frames across all 12 temporal windows (mean Laplacian variance 412.8 > 120 threshold) with smooth, continuous optical flow suitable for visuomotor tracking."
    },
    "scene_diversity": {
      "score": 4,
      "reasoning": "Active kitchen countertop environment featuring multiple distinct objects (ceramic mug, water bottle, cutting board, utensils) across varied spatial positions."
    },
    "hand_visibility": {
      "score": 5,
      "reasoning": "Both left and right hands clearly visible in 11/12 sampled frames (active manipulation zone occupancy 24.6%), demonstrating clear bimanual pour and grasp."
    },
    "lighting_exposure": {
      "score": 4,
      "reasoning": "Well-balanced indoor illumination (mean HSV V=148.2, underexposed ratio 0.8%, overexposed ratio 0.3%) with minimal shadow occlusion on manipulated objects."
    },
    "duplicate_risk": {
      "score": 0,
      "reasoning": "64-bit perceptual hash signature has minimum Hamming distance of 28 bits against all prior clips in the batch registry (threshold <= 6 bits)."
    },
    "overall_usability": {
      "score": 5,
      "reasoning": "High-value robotics manipulation trajectory: clean ego-centric viewpoint, unoccluded bimanual object interaction, and stable optical flow."
    }
  },
  "tier": "usable",
  "tier_confidence": 0.94
}
```

### 3. 10-Clip Empirical Benchmark Summary (`Q4`)

Running `evaluate_pipeline.py` on the 10-clip ground-truth suite yields the following per-dimension binary/thresholded classification metrics (full breakdown in `sample_outputs/precision_recall_report.json` and `DESIGN_DOC.md`):

| Scoring Dimension / Task | Evaluation Criterion (Positive Class) | Precision | Recall | F1-Score | Primary Edge Case Observed |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`egocentric_framing`** | Valid first-person perspective (`score == 1`) | 1.00 | 1.00 | 1.00 | Chest-mount vs. head-mount pitch angle ambiguity |
| **`motion_quality`** | Acceptable optical flow (`score >= 3`) | 1.00 | 0.88 | 0.93 | Mild jitter (`clip_10`) sits right on the score 2/3 boundary |
| **`scene_diversity`** | Non-repetitive scene (`score >= 3`) | 0.89 | 1.00 | 0.94 | Clean minimalist desk (`clip_03`) looks slightly sparse |
| **`hand_visibility`** | Hands visible in manipulation (`score >= 3`) | 1.00 | 1.00 | 1.00 | Brief single-hand exit in `clip_09` lowers score from 4 to 3 |
| **`lighting_exposure`** | Usable exposure (`score >= 3`) | 1.00 | 1.00 | 1.00 | High-contrast window backlighting on metallic objects |
| **`duplicate_risk`** | Duplicate detected (`score == 1`) | 1.00 | 1.00 | 1.00 | Same worker repeating identical task on same table |
| **Final Tier Routing** | Exact match across `usable` / `borderline` / `reject` | **1.00** | **1.00** | **1.00** | 10/10 clips routed to their exact ground-truth tier |

### 4. 2M-Hour Infrastructure & Cost Summary (`Q8–Q11`)

At **2,000,000 hours over 6 months** (`880,000 videos/day` · `10.56M frame evaluations/day` · `264,000 human-reviewed videos/day`), the economics of the pipeline look very different from a small prototype (see `INFRA.md` and `cost_model_2M_hours.csv` for full working):

| Cost Component | Option 1: Commercial Flagship API (`GPT-4o` / `GPT-4V`) | Option 2: Commercial Flash Batch API (`Gemini 1.5 Flash`) | Option 3: Self-Hosted `Qwen2-VL-7B-AWQ` on AWS EC2 (`g5.2xlarge` Spot/RI Mix) |
| :--- | :---: | :---: | :---: |
| **Monthly VLM Inference Cost (26.4M clips/mo)** | $264,000 / mo | $31,680 / mo | **$9,420 / mo** (14x `g5.2xlarge` nodes via vLLM) |
| **Monthly S3 Storage & Lifecycle (90-Day Rolling)** | $21,450 / mo | $21,450 / mo | **$21,450 / mo** (S3 Intelligent-Tiering + Glacier Instant) |
| **Monthly Queue, DB & CPU Pre-Filter (`SQS` + `RDS` + `EC2`)** | $4,850 / mo | $4,850 / mo | **$4,850 / mo** |
| **Monthly Human Annotation (`264K clips/day` @ 30s, $3.50/hr)** | $231,000 / mo | $231,000 / mo | **$231,000 / mo** (66,000 annotator hours/mo) |
| **Total Monthly Pipeline OPEX** | **$521,300 / mo** | **$288,980 / mo** | **$266,720 / mo** |

* **API vs. Self-Hosted GPU Break-Even (`Q8`)**: Self-hosting `Qwen2-VL-7B-Instruct-AWQ` on AWS EC2 `g5.2xlarge` instances breaks even against commercial flagship API pricing at just **~31,400 videos/day** (~392 hours of video/day)—less than **3.6%** of Locara's target daily volume.
* **The #1 Lever Moving Total Cost (`Q11`)**: Once VLM inference is self-hosted on GPUs ($9.4K/mo), **Human Annotator Labor ($231K/mo — 86.6% of total monthly OPEX)** becomes the dominant cost driver. Reducing human review time per clip from 45s to 20s via better VLM pre-annotation—or tightening the VLM confidence calibration so only 20% of clips require human review instead of 30%—saves **$77,000 to $102,000 every single month**, dwarfing any further GPU optimization.

## Companion Documents in This Repository

* **`DESIGN_DOC.md`**: Full technical design document answering **Q1** (Frame Sampling Strategy), **Q2** (Model Selection Rationale), **Q3** (Prompt Design & `v1 -> v3` Iteration Log), **Q4** (Precision/Recall Analysis), **Q5** (Review Queue Design), **Q6** (Inter-Annotator Agreement & Ambiguity Resolution), **Q7** (Closed-Loop Retraining & Prompt Feedback), and a non-ML stakeholder executive summary.
* **`INFRA.md`**: Production infrastructure blueprint answering **Q8** (GPU vs. API Trade-off & Break-Even Math), **Q9** (SQS + Ray/Celery Orchestration & Backpressure at 3x Evening Spikes), **Q10** (Storage Topology & Downstream Data Contracts), and **Q11** (Cost Model Analysis).
* **`cost_model_2M_hours.csv`**: Line-by-line monthly and 6-month cost model spreadsheet with sensitivity scenarios.
