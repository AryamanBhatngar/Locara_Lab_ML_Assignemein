"""Exporters: COCO with video extensions, and Label Studio import tasks.

COCO has no native notion of video or actions, so the export keeps standard `images`,
`annotations` (boxes on keyframes) and `categories`, and adds clearly named top-level lists:
`videos`, `action_segments`, `hand_states`, `scene_labels`, `viewpoint_labels`. Standard COCO
readers ignore the extra keys; the client's loader reads them.

The Label Studio tasks match `label_studio_interface.xml` (from_name / to_name pairs).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from ..schemas.taxonomy_schema import (
    TAXONOMY_VERSION,
    ActionSegment,
    ObjectAnnotation,
    ObjectCategory,
    PreAnnotation,
)

CATEGORY_IDS = {c.value: i + 1 for i, c in enumerate(ObjectCategory)}
VideoInfo = dict[str, Any]


def _coco_video(index: int, annotation: PreAnnotation, info: VideoInfo) -> dict[str, Any]:
    return {
        "id": index,
        "video_id": annotation.video_id,
        "file_name": info["file_name"],
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "duration_sec": annotation.duration_sec,
        "quality_tier": annotation.tier,
        "needs_review": annotation.needs_review,
    }


def _coco_image(
    image_id: int, index: int, annotation: PreAnnotation, time_sec: float, info: VideoInfo
) -> dict[str, Any]:
    return {
        "id": image_id,
        "video_id": index,
        "file_name": f"{annotation.video_id}/t{time_sec:07.2f}.jpg",
        "frame_index": round(time_sec * info["fps"]),
        "time_sec": time_sec,
        "width": info["width"],
        "height": info["height"],
    }


def _coco_box(box_id: int, image_id: int, obj: ObjectAnnotation, info: VideoInfo) -> dict[str, Any]:
    x, y, w, h = obj.bbox
    width, height = info["width"], info["height"]
    return {
        "id": box_id,
        "image_id": image_id,
        "category_id": CATEGORY_IDS[obj.category],
        "bbox": [
            round(x * width, 1),
            round(y * height, 1),
            round(w * width, 1),
            round(h * height, 1),
        ],
        "area": round(w * width * h * height, 1),
        "iscrowd": 0,
        "attributes": {
            "label": obj.label,
            "role": obj.role,
            "confidence": obj.confidence,
            "needs_human_verification": obj.needs_human_verification,
        },
    }


def to_coco(items: list[tuple[PreAnnotation, VideoInfo]]) -> dict[str, Any]:
    """Each item is a pre-annotation with its video's width, height, fps and file_name."""
    images, boxes, videos, actions, hands, scenes, views = [], [], [], [], [], [], []
    image_id = box_id = 0
    for index, (annotation, info) in enumerate(items, start=1):
        videos.append(_coco_video(index, annotation, info))
        frame_ids: dict[float, int] = {}
        for obj in annotation.objects:
            if obj.time_sec not in frame_ids:
                image_id += 1
                frame_ids[obj.time_sec] = image_id
                images.append(_coco_image(image_id, index, annotation, obj.time_sec, info))
            box_id += 1
            boxes.append(_coco_box(box_id, frame_ids[obj.time_sec], obj, info))
        actions += [{"video_id": index, **a.model_dump()} for a in annotation.actions]
        hands += [{"video_id": index, **h.model_dump()} for h in annotation.hand_states]
        if annotation.scene:
            scenes.append({"video_id": index, **annotation.scene.model_dump()})
        if annotation.viewpoint:
            views.append({"video_id": index, **annotation.viewpoint.model_dump()})
    return {
        "info": {
            "description": "Locara ego-centric pre-annotations (candidate labels)",
            "version": TAXONOMY_VERSION,
            "format": "coco_video_ext.v1",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"} for name, i in CATEGORY_IDS.items()
        ],
        "images": images,
        "annotations": boxes,
        "videos": videos,
        "action_segments": actions,
        "hand_states": hands,
        "scene_labels": scenes,
        "viewpoint_labels": views,
    }


def _choice(name: str, value: str, score: float) -> dict[str, Any]:
    return {
        "from_name": name,
        "to_name": "video",
        "type": "choices",
        "value": {"choices": [value]},
        "score": score,
    }


def _action_result(action: ActionSegment, fps: float) -> dict[str, Any]:
    start = int(action.start_sec * fps) + 1
    return {
        "from_name": "actions",
        "to_name": "video",
        "type": "timelinelabels",
        "value": {
            "ranges": [{"start": start, "end": max(int(action.end_sec * fps), start)}],
            "timelinelabels": [action.action],
        },
        "score": action.confidence,
    }


def _object_result(obj: ObjectAnnotation, fps: float) -> dict[str, Any]:
    x, y, w, h = obj.bbox
    keyframe = {
        "frame": round(obj.time_sec * fps) + 1,
        "enabled": False,
        "time": obj.time_sec,
        "rotation": 0,
        "x": round(x * 100, 2),
        "y": round(y * 100, 2),
        "width": round(w * 100, 2),
        "height": round(h * 100, 2),
    }
    return {
        "from_name": "objects",
        "to_name": "video",
        "type": "videorectangle",
        "value": {"labels": [obj.category], "sequence": [keyframe]},
        "meta": {"text": [obj.label, obj.role]},
        "score": obj.confidence,
    }


def _hand_results(annotation: PreAnnotation) -> list[dict[str, Any]]:
    """Label Studio shows one choice per hand control, so use the most common state in the clip."""
    states = annotation.hand_states
    mean = round(sum(h.confidence for h in states) / len(states), 2)
    results = []
    for field in ("left_hand", "right_hand", "bimanual_mode"):
        value, _ = Counter(getattr(h, field) for h in states).most_common(1)[0]
        results.append(_choice(field, value, mean))
    return results


def _task(annotation: PreAnnotation, info: VideoInfo, video_url_prefix: str) -> dict[str, Any]:
    fps = info["fps"]
    results = [_action_result(a, fps) for a in annotation.actions]
    results += [_object_result(o, fps) for o in annotation.objects]
    confidences = [a.confidence for a in annotation.actions]
    confidences += [o.confidence for o in annotation.objects]
    if annotation.scene:
        results.append(_choice("scene", annotation.scene.scene_type, annotation.scene.confidence))
        confidences.append(annotation.scene.confidence)
    if annotation.viewpoint:
        view = annotation.viewpoint
        results.append(_choice("viewpoint", view.viewpoint_quality, view.confidence))
        confidences.append(view.confidence)
    if annotation.hand_states:
        results += _hand_results(annotation)
        confidences += [h.confidence for h in annotation.hand_states]
    return {
        "data": {
            "video": video_url_prefix + quote(info["file_name"]),
            "video_id": annotation.video_id,
            "quality_tier": annotation.tier,
            "needs_review": annotation.needs_review,
            "hand_states_per_frame": [h.model_dump() for h in annotation.hand_states],
        },
        "predictions": [
            {
                "model_version": annotation.provider,
                "score": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                "result": results,
            }
        ],
    }


def to_label_studio(
    items: list[tuple[PreAnnotation, VideoInfo]],
    video_url_prefix: str = "/data/local-files/?d=",
) -> list[dict[str, Any]]:
    return [_task(annotation, info, video_url_prefix) for annotation, info in items]
