import re
from pathlib import Path

import pytest

from src.annotation.exporters import CATEGORY_IDS, to_coco, to_label_studio
from src.annotation.pre_annotator import PreAnnotator, SkipVideo
from src.config import Settings
from src.schemas.scoring_schema import video_id_for
from src.schemas.taxonomy_schema import taxonomy_lists
from src.scorer.frame_sampler import sample_video
from src.scorer.vlm_client import BaseVLMClient

XML = Path(__file__).resolve().parents[1] / "src/annotation/label_studio_interface.xml"

REPLY = {
    "scene": {"scene_type": "home_kitchen", "confidence": 0.92},
    "viewpoint": {"viewpoint_quality": "minor_hand_self_occlusion", "confidence": 0.6},
    "actions": [
        {"action": "reach", "start_sec": 0.0, "end_sec": 2.0, "confidence": 0.9},
        {"action": "grasp", "start_sec": 1.5, "end_sec": 3.0, "confidence": 0.7},  # overlaps reach
        {"action": "pour", "start_sec": 3.0, "end_sec": 8.0, "confidence": 0.85},
        {"action": "stir_mix", "start_sec": 8.0, "end_sec": 9.0, "confidence": 0.3},  # dropped
        {"action": "fly", "start_sec": 9.0, "end_sec": 9.5, "confidence": 0.9},  # invalid
    ],
    "objects": [
        {
            "category": "container",
            "label": "steel_tumbler",
            "role": "active_manipulated",
            "time_sec": 4.0,
            "bbox": [0.4, 0.5, 0.2, 0.3],
            "confidence": 0.88,
        },
        {
            "category": "utensil",
            "label": "spoon",
            "role": "background_context",
            "time_sec": 4.0,
            "bbox": [0.9, 0.9, 0.3, 0.3],
            "confidence": 0.9,
        },  # bbox overflow
    ],
    "hand_states": [
        {
            "time_sec": 1.0,
            "left_hand": "open_reaching",
            "right_hand": "idle_visible",
            "bimanual_mode": "unimanual_left",
            "confidence": 0.8,
        },
        {
            "time_sec": 4.0,
            "left_hand": "power_grasp",
            "right_hand": "supporting_stabilizing",
            "bimanual_mode": "asymmetric_bimanual_hold_and_act",
            "confidence": 0.65,
        },
    ],
}


class StubAnnotator(BaseVLMClient):
    provider = "stub"

    def __init__(self):
        super().__init__(Settings())
        self.model = "stub-model"

    def annotate(self, video):
        import copy

        return copy.deepcopy(REPLY), {}


def score_rec(tier, path):
    return {"video_id": video_id_for(Path(path).name), "tier": tier, "meta": {"source": str(path)}}


def test_confidence_gating(clips):
    pa = PreAnnotator(StubAnnotator(), 0.75).annotate(
        str(clips["ideal10"]), score_rec("usable", clips["ideal10"])
    )
    assert pa.scene.scene_type == "home_kitchen" and not pa.scene.needs_human_verification
    assert pa.viewpoint.needs_human_verification  # 0.60 is below threshold
    assert [a.action for a in pa.actions] == ["reach", "pour"]  # overlap resolved
    assert len(pa.objects) == 1  # bbox overflow dropped
    assert [h.needs_human_verification for h in pa.hand_states] == [False, True]
    assert pa.dropped_low_confidence >= 3
    assert pa.needs_review


def test_borderline_forces_review_and_reject_is_skipped(clips):
    pa = PreAnnotator(StubAnnotator()).annotate(
        str(clips["ideal10"]), score_rec("borderline", clips["ideal10"])
    )
    assert pa.scene.needs_human_verification
    with pytest.raises(SkipVideo):
        PreAnnotator(StubAnnotator()).annotate(
            str(clips["ideal10"]), score_rec("reject", clips["ideal10"])
        )
    with pytest.raises(ValueError):
        PreAnnotator(StubAnnotator(), threshold=0.3)


def _items(clips):
    s = sample_video(clips["ideal10"])
    pa = PreAnnotator(StubAnnotator()).annotate(
        str(clips["ideal10"]), score_rec("usable", clips["ideal10"]), s
    )
    return [(pa, {"file_name": "ideal10.mp4", "width": s.width, "height": s.height, "fps": s.fps})]


def test_coco_export(clips):
    coco = to_coco(_items(clips))
    assert {
        "images",
        "annotations",
        "categories",
        "videos",
        "action_segments",
        "hand_states",
    } <= set(coco)
    ann = coco["annotations"][0]
    img = coco["images"][0]
    assert ann["image_id"] == img["id"] and ann["category_id"] == CATEGORY_IDS["container"]
    assert ann["bbox"] == [
        pytest.approx(0.4 * img["width"], abs=0.1),
        pytest.approx(0.5 * img["height"], abs=0.1),
        pytest.approx(0.2 * img["width"], abs=0.1),
        pytest.approx(0.3 * img["height"], abs=0.1),
    ]
    assert len(coco["action_segments"]) == 2


def test_label_studio_tasks_match_interface(clips):
    tasks = to_label_studio(_items(clips))
    xml = XML.read_text(encoding="utf-8")
    names = set(re.findall(r'name="([a-z_]+)"', xml))
    for r in tasks[0]["predictions"][0]["result"]:
        assert r["from_name"] in names and r["to_name"] == "video"
    assert tasks[0]["data"]["video"].endswith("ideal10.mp4")


def test_interface_labels_match_taxonomy():
    xml = XML.read_text(encoding="utf-8")
    values = set(re.findall(r'value="([a-z_]+)"', xml))
    t = taxonomy_lists()
    for key in (
        "actions",
        "object_categories",
        "scenes",
        "hand_poses",
        "bimanual_modes",
        "viewpoints",
    ):
        assert set(t[key]) <= values, key
