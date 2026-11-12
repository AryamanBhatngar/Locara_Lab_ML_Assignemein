"""Prompts for the VLM scorer and the pre-annotator (Q3).

Three versions are kept so the iteration is reproducible: run any version with
`score_videos.py --prompt-version v1|v2|v3` and compare with `evaluate_pipeline.py`.

v1  Baseline. Dimension names and scales only. Expected weakness: unanchored scales, so a
    "3" means different things on different clips; motion judged from single sharp frames.
v2  Rubric-anchored. Every score level has a concrete description written for robotics data,
    plus ego-centric cues (hands entering from the bottom edge, head-bob parallax) and the
    handheld-phone pattern to reject. Adds per-dimension confidence.
v3  v2 plus OpenCV telemetry (sharpness, optical-flow jitter, static ratio, exposure,
    skin-tone hand proxy) as measured evidence, timestamped frames, and an explicit rule to
    prefer the measurement when frames look sharp but jitter is high. Strict JSON only.

The observed failure modes and fixes belong in DESIGN_DOC.md, filled from real runs of
`evaluate_pipeline.py`; nothing in this file claims results.
"""

from __future__ import annotations

from ..schemas.taxonomy_schema import taxonomy_lists

PROMPT_VERSION = "v3"

_JSON_SHAPE = """Return ONLY a JSON object, no prose, no code fences, with exactly these keys:
{"egocentric_framing": {"score": <0 or 1>, "confidence": <0-1>, "reasoning": "<1-2 sentences>"},
 "motion_quality": {"score": <1-5>, "confidence": <0-1>, "reasoning": "..."},
 "scene_diversity": {"score": <1-5>, "confidence": <0-1>, "reasoning": "..."},
 "hand_visibility": {"score": <1-5>, "confidence": <0-1>, "reasoning": "..."},
 "lighting_exposure": {"score": <1-5>, "confidence": <0-1>, "reasoning": "..."},
 "duplicate_risk": {"score": <0-1>, "confidence": <0-1>, "reasoning": "..."},
 "overall_usability": {"score": <1-5>, "confidence": <0-1>, "reasoning": "..."}}"""

V1 = (
    """You will see frames from a video. Score it on:
egocentric_framing (0-1), motion_quality (1-5), scene_diversity (1-5), hand_visibility (1-5),
lighting_exposure (1-5), duplicate_risk (0-1), overall_usability (1-5).
"""
    + _JSON_SHAPE
)

RUBRIC = """You are a quality-control reviewer for ego-centric (first-person) video that will
train robot manipulation policies. Frames are in time order and stamped with their time.

Score each dimension using these anchors:

egocentric_framing (0 or 1)
  1 = camera is head- or chest-mounted on the person doing the task: hands and forearms enter
      from the bottom edge, the view moves with the head (small continuous bob and turns),
      objects are at arm's length.
  0 = handheld phone filming forward or someone else, tripod or static placement, third-person
      view of a person, or the camera pointed away from the work. A perfectly still view
      across many seconds is a strong sign of 0.

motion_quality (1-5)
  5 = sharp frames, smooth head motion. 4 = minor blur in a few frames. 3 = noticeable shake
  or blur but actions still readable. 2 = frequent blur or jerky motion; hand detail lost.
  1 = whip pans or blur in most frames; unusable for optical flow.

scene_diversity (1-5)
  5 = rich, real working environment with many distinct objects and surfaces.
  3 = ordinary scene with a few objects. 1 = blank wall, ceiling, floor, or empty table.

hand_visibility (1-5)
  5 = both hands clearly visible manipulating objects most of the clip. 4 = one or both hands
  visible and acting most of the time. 3 = hands visible about half the time or often cut off.
  2 = hands rarely visible. 1 = no hands.

lighting_exposure (1-5)
  5 = even, natural exposure. 3 = somewhat dark or bright but details visible.
  2 = details lost in shadow or highlights. 1 = nearly black, blown out, or strong flicker.

duplicate_risk (0-1)
  Probability that this clip repeats another clip (same place, same objects, same motion).
  Use the duplicate signal below if present; otherwise judge from repetition inside the clip.

overall_usability (1-5)
  Would a robotics engineer keep this clip for manipulation training? 5 = clearly yes,
  3 = only after review, 1 = no.

confidence (0-1) for every dimension: how sure you are, given what the frames show.
Low confidence is better than a confident guess.
"""

V2 = RUBRIC + "\n" + _JSON_SHAPE

V3_RULES = """
Measured signals for this clip (computed with OpenCV on the whole clip, not only these frames):
{telemetry}
{duplicate_note}
Rules for using the measurements:
- Single frames can look sharp while the camera shakes between them. If jitter is above about
  1.5, motion_quality should not exceed 3 even when the frames look sharp.
- A static-camera ratio above about 0.85 with near-zero motion means a fixed camera, which
  points to egocentric_framing = 0 unless the frames clearly show the wearer's own hands.
- Trust your eyes over the skin-tone proxy for hands: it also fires on wood and skin-coloured
  objects. Use it only to break ties.
- If frames and measurements disagree, say so in the reasoning and lower the confidence.
Clip: {duration:.1f} s, {n_frames} frames sampled.
"""

V3 = RUBRIC + V3_RULES + "\n" + _JSON_SHAPE

VERSIONS = {"v1": V1, "v2": V2, "v3": V3}


def build_scoring_prompt(
    version: str, telemetry_text: str, duration: float, n_frames: int, duplicate_note: str = ""
) -> str:
    if version not in VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}")
    if version != "v3":
        return VERSIONS[version]
    # str.replace, not str.format: the JSON example in the prompt is full of braces.
    values = {
        "{telemetry}": telemetry_text,
        "{duplicate_note}": duplicate_note
        or "Duplicate signal: no near-duplicate found in this batch.",
        "{duration:.1f}": f"{duration:.1f}",
        "{n_frames}": str(n_frames),
    }
    out = V3
    for key, val in values.items():
        out = out.replace(key, val)
    return out


def build_annotation_prompt(duration: float, frame_times: list[float]) -> str:
    t = taxonomy_lists()
    return f"""You are pre-labelling ego-centric video for robot manipulation training.
Frames are in time order; their times in seconds are {frame_times}. Clip length {duration:.1f} s.
Use ONLY these labels.
actions: {t["actions"]}
object categories: {t["object_categories"]}; interaction roles: {t["interaction_roles"]}
scenes: {t["scenes"]}
hand poses (per hand): {t["hand_poses"]}; bimanual modes: {t["bimanual_modes"]}
viewpoint quality: {t["viewpoints"]}

Return ONLY JSON:
{{"scene": {{"scene_type": "...", "confidence": 0-1}},
 "viewpoint": {{"viewpoint_quality": "...", "confidence": 0-1}},
 "actions": [{{"action": "...", "start_sec": s, "end_sec": s, "confidence": 0-1}}],
 "objects": [{{"category": "...", "label": "short_snake_case", "role": "...", "time_sec": s,
              "bbox": [x, y, w, h] normalised 0-1 in that frame, "confidence": 0-1}}],
 "hand_states": [{{"time_sec": s, "left_hand": "...", "right_hand": "...",
                  "bimanual_mode": "...", "confidence": 0-1}}]}}
Rules: action segments must not overlap and must lie within the clip; give one hand_state per
frame time; label at most 6 objects; "left" and "right" mean the wearer's own hands; when
unsure, lower the confidence instead of guessing.
"""
