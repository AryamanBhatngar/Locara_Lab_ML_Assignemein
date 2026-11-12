# Design document: VLM-assisted QC and annotation pipeline

Author: Aryaman Bhatnagar. Covers Part A (Q1 to Q4) and Part B (Q5 to Q7). Infrastructure and cost (Q8 to Q11) are in [INFRA.md](INFRA.md), summarised below.

## Summary for non-ML stakeholders

Each clip is checked on seven things a robotics buyer cares about: first-person view, steadiness, scene variety, visible hands, lighting, re-uploads and overall usefulness. Hard rules reject clips that can never be training data. Clips are accepted automatically only when the model is positive and confident, so doubtful clips go to people. Scoring looks at 8 to 16 frames and costs about 0.03 to 1.5 US cents per clip ([INFRA.md](INFRA.md)). Human review is the largest cost.

Not proven yet: no VLM has been run. The sample outputs and the Q4 numbers come from the offline OpenCV baseline.

## Part A: VLM quality scorer

### Q1. Frame sampling strategy

I budget frames by length: 8 up to 10 s, rising linearly to 12 at 30 s, 12 up to 60 s, and 16 beyond (`frames_for_duration`). The clip is split into equal segments. From three candidate frames per segment I keep the sharpest (highest Laplacian variance). Equal segments cover the whole clip, so a late defect is not missed. The sharpest candidate stops one blurred frame from standing in for a good segment. A separate segment-centre frame is hashed for duplicate detection.

Frames are resized to 512 px on the long side and stamped with their time (`t=012.5s`), so the model can reason about order. At 512 px a frame is one Gemini tile (258 tokens), so 12 frames cost about 3,100 image tokens.

Most defects (exposure, framing, duplicates, scene) show in any frame. Shake does not, because it happens between frames. So instead of adding frames, one OpenCV pass over the whole clip (160 px wide, greyscale, 4 fps) measures Farneback optical flow (overall motion, camera motion, jitter), a static-camera ratio, brightness, clipping, flicker, a skin-tone hand proxy and edge density.

In the committed run, each 10 to 60 s clip at about 640x360 took 0.9 to 3.1 s in total (`meta.timings_ms.total`), four clips at once on one machine. I have not measured 1080p. If the container header is wrong (bad frame rate or count, or a truncated upload), the sampler trusts the decoder and decodes again.

Doubling to 24 frames would double image tokens for little gain. A second pass on borderline clips is a better use of budget.

### Q2. Model selection

The default is `gemini-2.5-flash`: many images per request, enforced JSON, and thinking disabled for latency and cost. At $0.30 / $2.50 per million input / output tokens it costs about $0.0024 per clip, or $0.0012 in batch mode (INFRA Q8). The brief lists Gemini 1.5 Pro; the code uses the 2.5 generation, and the model is one setting (`GEMINI_MODEL`).

Performance on ego-centric video is not measured yet. I expect hands, lighting and scene to be strong, since they show in still frames. I expect motion and framing to be weakest, hence the telemetry.

| If the binding constraint is | Switch to | Why |
|---|---|---|
| Cost | Gemini 2.5 Flash-Lite in batch mode, then self-hosted Qwen2.5-VL-7B on L4 GPUs | Flash-Lite batch is about 4x cheaper than Flash batch; self-hosting costs a little more than Flash-Lite but keeps footage in Locara's AWS account (INFRA Q8) |
| Accuracy | Gemini 2.5 Pro or a GPT-4o-class model, only on borderline clips | An estimated 5 to 10 times the price, so spend it where the decision is uncertain |
| Data residency | Self-hosted Qwen2.5-VL | Video never leaves Locara's VPC |

All three sit behind one interface (`src/scorer/vlm_client.py`): `gemini`, `openai`, and `vllm` for any OpenAI-compatible server.

### Q3. Prompt design and iteration

The prompts are in `src/scorer/prompts.py`; pick one with `--prompt-version`.

| Version | Change | Reason |
|---|---|---|
| v1 | Dimension names and scales only | Baseline to measure against |
| v2 | Anchors for every score level, written for robotics data; first-person cues (hands from the bottom edge, head-bob parallax) and the handheld-phone pattern to reject; per-dimension confidence | Unanchored 1 to 5 scales drift between clips; confidence feeds routing |
| v3 | v2 plus measured telemetry, timestamped frames and explicit rules: cap motion at 3 when jitter is high even if frames look sharp; a static camera points to non-ego; trust eyes over the skin proxy; disagreement must lower confidence | Still frames cannot show shake; models tend to judge motion from sharpness |

Failure modes I found while building, from this repo's runs:

1. Template bug. The first v3 used `str.format`, which broke on the JSON braces in the prompt, and mocked tests missed it. It now uses placeholder replacement, and a test builds every version.
2. Blur-direction metric dropped. A gradient-direction ratio flagged every synthetic scene, because tables and shelves are strongly horizontal. Optical-flow magnitude catches whip pans instead.
3. "Static" in the dark. Optical flow reads dark or textureless frames as motionless, so the baseline called them non-first-person. Framing now defers to other gates when brightness is below 35 or edges are nearly absent.
4. False duplicate. At a Hamming threshold of 10, a different clip matched at 10 while the true duplicate sat at 6, on an earlier encode. The threshold is now 8. In the committed outputs the duplicate is at 2 and the nearest other clip at 11 (`meta.nearest_hash_distance`). The margin moves with the encoder, so it must be recalibrated on real footage.

Still to do: compare v1, v2 and v3 with a real VLM using `evaluate_pipeline.py`.

### Q4. Precision and recall

These numbers come from the offline OpenCV baseline (`--provider offline`), not from a VLM. The Gemini client is implemented and unit-tested, but it was not run for this submission, so there is no Gemini result here. To reproduce Q4 with Gemini:

```
python score_videos.py data/synthetic --provider gemini --workers 2 \
    -o sample_outputs/scored_videos_batch.json -v
python evaluate_pipeline.py --scores sample_outputs/scored_videos_batch.json \
    --truth data/synthetic/ground_truth.json \
    --out sample_outputs/precision_recall_report.json \
    --markdown sample_outputs/precision_recall_report.md
```

The positive class is the defect: a 1 to 5 dimension at 2 or below, framing at 0, or duplicate risk at 0.5 or above. Source: `sample_outputs/precision_recall_report.json`.

| Dimension | Defects | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| egocentric_framing | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |
| motion_quality | 1 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| scene_diversity | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| hand_visibility | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| lighting_exposure | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| duplicate_risk | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| overall_usability | 5 | 4 | 0 | 1 | 1.00 | 0.80 | 0.89 |

Tier accuracy is 0.70. Reject detection has precision 1.00 and recall 1.00 (5 of 5). Nothing was auto-accepted, and 5 clips went to review.

The three biggest failures are the three `tier_errors`: usable clips sent to review, with no hard gate failed. `tier_errors` has no reasoning field, so each quote is the scorer's reasoning for that clip's least confident dimension, from `scored_videos_batch.json`.

1. `clip_01_kitchen_pour_ideal`: composite 4.7, tier confidence 0.73. Scene diversity (0.65): "Offline rule: edge density 0.058 as a clutter proxy; cannot judge across locations."
2. `clip_02_workshop_assembly_ideal`: composite 4.7, tier confidence 0.75. Scene diversity (0.53): "Offline rule: edge density 0.068 as a clutter proxy; cannot judge across locations."
3. `clip_03_office_typing_usable`: composite 4.85, tier confidence 0.67. Scene diversity (0.51): "Offline rule: edge density 0.099 as a clutter proxy; cannot judge across locations."

Each clears the 3.75 composite line but misses the 0.80 confidence gate. That costs review time, not dataset quality.

The other errors, in `failures`, have clear causes. Low texture reads as blur on the dark and blank-wall clips ("sharpness 4", "sharpness 12"). The re-encoded duplicate looks static ("fixed camera, not head-mounted"). Overall usability misses the severe-blur clip (3 against a true 1) because it averages other dimensions, though the hard gates still reject it. Edge density is a poor proxy for scene variety. These are the judgements a VLM should add.

Limits: 10 synthetic clips with ground truth designed into them, so this checks the pipeline, not a model. The numbers also depend on the encoder (ffmpeg 9.0.2 with libx264 here).

## Part B: Annotation pipeline

### Taxonomy (`src/schemas/taxonomy_schema.py`, version 1.0.0)

| Axis | Labels | Granularity choice |
|---|---|---|
| Actions (temporal segments) | 17 primitives: reach, grasp, pick_up, move_transport, place, pour, open, close, push_pull, cut_slice, wipe_clean, stir_mix, type_press, assemble_insert, hand_over, walk_navigate, idle | Atomic manipulation primitives, not activities: policies are trained on primitives, and activities can be rebuilt from sequences, but not the reverse |
| Objects (boxes on keyframes) | 9 physical categories (rigid tool, container, articulated fixture, deformable cloth, food, electronics, utensil, fastener, other), a free snake_case label, and a role (manipulated, target receptacle, background) | Category drives grasp strategy; a closed list of every object in Indian homes and workshops would never be complete |
| Scene | home_kitchen, mechanical_workshop, office_desk, warehouse_packing, laundry_utility, living_room, outdoor_courtyard, other | Coarse on purpose: used for dataset balance and diversity pay, not for training |
| Hand state | per hand: power_grasp, precision_pinch, pointing_pressing, supporting_stabilizing, open_reaching, idle_visible, not_visible; plus bimanual mode | Left/right coordination ("hold and act") is a key signal for bimanual robots |
| Viewpoint quality | optimal, minor self-occlusion, partial object occlusion, extreme tilt, heavy occlusion | Tells the client which segments are usable for visual policies |

A general-purpose dataset would use activities and open-vocabulary objects with no hand detail. For a robotics client, per-hand state and primitive boundaries are worth the extra annotation time.

### Pre-annotation layer (`pre_annotate.py`)

The script takes a video and its score record, and skips rejected clips. It asks the VLM for labels from the taxonomy only and validates each with Pydantic. It drops unknown labels, boxes outside the frame, segments outside the clip and unusable confidences. Overlapping segments keep the more confident one. Labels at 0.75 or above become pre-labels, 0.50 to 0.75 are marked `needs_human_verification`, and lower ones are dropped. Every label on a borderline clip needs verification.

The COCO export (`coco_video_ext.v1`) adds `videos`, `action_segments`, `hand_states`, `scene_labels` and `viewpoint_labels` to the standard keys; COCO readers ignore them. The Label Studio tasks carry predictions that match `label_studio_interface.xml`, and tests check both against the taxonomy. The offline provider only judges hands and viewpoint, so the sample files have no objects or actions.

### Q5. Review queue design

Routing is in `quality_scorer.route`:

1. Hard gates reject: not first-person, duplicate, or motion, hands or lighting at 1. These are validity checks.
2. Composite = 0.30 hands + 0.25 motion + 0.20 lighting + 0.15 scene + 0.10 overall. Manipulation training needs hands and motion most.
3. Reject below 2.5. Auto-accept at 3.75 or above with tier confidence of 0.80 or above. Review the rest.

Tier confidence is the mean model confidence over the routing dimensions. It drops by 0.10 within 0.25 of a threshold, and by 0.10 per dimension where the model and OpenCV disagree by 2 or more.

```
  new clip
     |
     v
+---------------------------------------------------------------+
| SCORER: 8 to 16 frames + OpenCV telemetry + VLM, 7 dimensions |
| hard gates -> composite score -> tier confidence              |
+---------------------------------------------------------------+
     |                          |                          |
     v                          v                          v
+-------------+       +-------------------+       +------------------+
| AUTO-REJECT |       |   REVIEW QUEUE    |       |   AUTO-ACCEPT    |
| hard gate,  |       | everything else,  |       | composite >= 3.75|
| or composite|       | plus a 2% spot    |       | and tier conf    |
| below 2.5   |       | check of the rest |       | >= 0.80          |
+-------------+       +-------------------+       +------------------+
     |                          |                          |
     v                          +------------+-------------+
 fix-it message                              |
 to the worker                               v
                          pre-annotator: taxonomy labels, gated at
                          0.75 (pre-label) and 0.50 (verify or drop)
                                             |
                                             v
                          +-------------------------------------+
                          | LABEL STUDIO: reviewer confirms the |
                          | tier and corrects the pre-labels    |
                          +-------------------------------------+
                                             |
                                 corrections logged as deltas
                                 (predicted vs corrected)
                                  |                       |
                                  v                       v
                       +-------------------+   +----------------------+
                       | EVALUATION SET    |   | PROMPT REVISION      |
                       | frozen labelled   |   | anchors and few-shot |
                       | clips             |   | for top error groups |
                       +-------------------+   +----------------------+
                                  |                       |
                                  +-----------+-----------+
                                              |
                                              v
                          evaluate_pipeline.py: promote the new prompt
                          only if it beats the old one; recalibrate the
                          thresholds with threshold_sweep
                                              |
                                              v
                                    back to the SCORER
```

The weights and thresholds are design choices, not fitted values. `threshold_sweep` in the evaluation report gives auto-accept precision and review share for each pair (composite 3.25 to 4.25, confidence 0.5 to 0.9). Pick the loosest pair that meets the client's precision bar (say 95%), check that the review share fits annotator capacity (the brief assumes 30%), and recalibrate monthly. On the synthetic set no pair takes the offline baseline to 95%, so the baseline should never auto-accept. Hard-gate rejects go back to the worker as a fix-it message ("hold the phone at head height").

### Q6. Annotator agreement and quality

Double-label 5 to 10% of review tasks. Categorical axes (scene, hand pose, viewpoint, clip verdict) use Krippendorff's alpha, which handles missing labels and more than two annotators, plus Cohen's kappa for pairs. Action segments use temporal IoU of matched segments at 0.5 and 0.75, plus label agreement. Boxes report the share of matches above IoU 0.5. About 2% of tasks are gold tasks with known answers.

Ambiguous clips need explicit rules: label an action only while contact is visible, and mark `partial_object_occlusion` instead of guessing. Disagreement beyond a threshold goes to a senior adjudicator. Clips that stay ambiguous are tagged and left out of the release. Low agreement on one label for a week is a taxonomy or guideline problem, not an annotator problem.

### Q7. Feedback loop

1. Log each correction as a delta: the Label Studio submission diffed against its pre-label, with video ID, frame times, model, prompt version, annotator and time spent.
2. Each week, group the deltas by dimension and error type, for example "handheld phone scored as first-person" or "pour labelled as place".
3. Fix prompts first, because it is cheap: anchors or rules for the top groups, and the hardest clips as few-shot examples. Promote a version only after `evaluate_pipeline.py` on a frozen labelled set. `meta.prompt_version` keeps results traceable.
4. Recalibrate thresholds monthly with the sweep.
5. With tens of thousands of corrected clips, fine-tune a LoRA adapter on self-hosted Qwen2.5-VL and A/B it on the frozen set.

## Infrastructure and cost

At the brief's 880,000 videos a day, the monthly total is $232,048 with Gemini Flash batch, $589,785 with GPT-4o, and $210,067 with self-hosted Qwen2.5-VL on 15 L4 GPUs, excluding raw video storage. Human review is $198,000 of each total, and it is the main lever. A 10% cut in review share, review time or annotator rate saves $19,800 a month, against $615 for 10% more GPU throughput. Details are in [INFRA.md](INFRA.md) and `cost_model_2M_hours.csv`.

## Trade-offs

- Telemetry instead of more frames: cheaper, and it measures motion that frames cannot show, but it adds a signal to calibrate.
- Hard gates: one fatal defect rejects a clip, however good the rest.
- Confidence-gated auto-accept: more review (all three usable synthetic clips went to review) for a cleaner dataset.
- Hosted Gemini first: nothing to provision, but footage goes to an API until volume justifies self-hosting.
- Primitive actions and per-hand state: slower annotation, but labels a robot policy can use.
- Hand-set weights and thresholds: usable without labels, but they need calibration (Q5).

## What I would do with more time

- Run Gemini on the synthetic and real clips (README protocol), then fill Q3 and Q4 with VLM numbers.
- A second pass on borderline clips, with more frames and a stronger model.
- Cross-clip embeddings for scene diversity, since one clip cannot show whether a location repeats.
- A learned hand detector instead of the skin-tone proxy, which fails on skin-coloured surfaces and in poor light.
- Benchmarks on real 1080p footage, and GPU throughput (`scripts/benchmark_vllm.py`).
