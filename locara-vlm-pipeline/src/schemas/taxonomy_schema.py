"""Five-axis annotation taxonomy for ego-centric robotics video (Part B).

Granularity rationale (full version in DESIGN_DOC.md):
- Actions are atomic manipulation primitives (reach, grasp, place...), not activities
  ("making tea"). Imitation-learning policies are trained on primitives and composed later;
  activity labels can be derived from primitive sequences, the reverse is not possible.
- Objects carry a coarse physical category (rigid, articulated, deformable...) plus a free
  class label. The category drives grasp strategy, which is what a robotics client needs;
  a closed list of every object would never be complete across Indian homes and workshops.
- Hand state is per hand, plus a bimanual mode, because left/right coordination is a
  first-class training signal for manipulation and is lost by a single "hands visible" flag.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(str, Enum):
    REACH = "reach"
    GRASP = "grasp"
    PICK_UP = "pick_up"
    MOVE_TRANSPORT = "move_transport"
    PLACE = "place"
    POUR = "pour"
    OPEN = "open"
    CLOSE = "close"
    PUSH_PULL = "push_pull"
    CUT_SLICE = "cut_slice"
    WIPE_CLEAN = "wipe_clean"
    STIR_MIX = "stir_mix"
    TYPE_PRESS = "type_press"
    ASSEMBLE_INSERT = "assemble_insert"
    HAND_OVER = "hand_over"
    WALK_NAVIGATE = "walk_navigate"
    IDLE = "idle"


class ObjectCategory(str, Enum):
    RIGID_TOOL = "rigid_tool"
    CONTAINER = "container"
    ARTICULATED_FIXTURE = "articulated_fixture"
    DEFORMABLE_CLOTH = "deformable_cloth"
    FOOD_ITEM = "food_item"
    ELECTRONICS = "electronics"
    UTENSIL = "utensil"
    FASTENER_PART = "fastener_part"
    OTHER = "other"


class InteractionRole(str, Enum):
    ACTIVE_MANIPULATED = "active_manipulated"
    TARGET_RECEPTACLE = "target_receptacle"
    BACKGROUND_CONTEXT = "background_context"


class Scene(str, Enum):
    HOME_KITCHEN = "home_kitchen"
    MECHANICAL_WORKSHOP = "mechanical_workshop"
    OFFICE_DESK = "office_desk"
    WAREHOUSE_PACKING = "warehouse_packing"
    LAUNDRY_UTILITY = "laundry_utility"
    LIVING_ROOM = "living_room"
    OUTDOOR_COURTYARD = "outdoor_courtyard"
    OTHER = "other"


class HandPose(str, Enum):
    POWER_GRASP = "power_grasp"
    PRECISION_PINCH = "precision_pinch"
    POINTING_PRESSING = "pointing_pressing"
    SUPPORTING_STABILIZING = "supporting_stabilizing"
    OPEN_REACHING = "open_reaching"
    IDLE_VISIBLE = "idle_visible"
    NOT_VISIBLE = "not_visible"


class BimanualMode(str, Enum):
    UNIMANUAL_LEFT = "unimanual_left"
    UNIMANUAL_RIGHT = "unimanual_right"
    SYMMETRIC_BIMANUAL = "symmetric_bimanual"
    ASYMMETRIC_BIMANUAL = "asymmetric_bimanual_hold_and_act"
    NO_HANDS = "no_hands"


class Viewpoint(str, Enum):
    OPTIMAL_CLEAR = "optimal_head_mount_clear"
    MINOR_SELF_OCCLUSION = "minor_hand_self_occlusion"
    PARTIAL_OBJECT_OCCLUSION = "partial_object_occlusion"
    EXTREME_ANGLE_TILT = "extreme_angle_tilt"
    HEAVY_OCCLUSION = "heavy_occlusion_unusable"


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_verification: bool = False


class ActionSegment(_Candidate):
    action: Action
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _order(self) -> ActionSegment:
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be after start_sec")
        return self


class ObjectAnnotation(_Candidate):
    category: ObjectCategory
    label: str = Field(min_length=1, max_length=64)
    role: InteractionRole
    time_sec: float = Field(ge=0.0)
    bbox: list[float] = Field(
        min_length=4, max_length=4, description="Normalised [x, y, width, height], each 0-1"
    )

    @model_validator(mode="after")
    def _bbox(self) -> ObjectAnnotation:
        x, y, w, h = self.bbox
        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
            and x + w <= 1.0001
            and y + h <= 1.0001
        ):
            raise ValueError(f"bbox out of range: {self.bbox}")
        return self


class HandState(_Candidate):
    time_sec: float = Field(ge=0.0)
    left_hand: HandPose
    right_hand: HandPose
    bimanual_mode: BimanualMode


class SceneLabel(_Candidate):
    scene_type: Scene


class ViewpointLabel(_Candidate):
    viewpoint_quality: Viewpoint


class PreAnnotation(BaseModel):
    """All candidate labels for one video after confidence gating."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    source: str
    duration_sec: float
    tier: str
    provider: str
    threshold: float
    scene: SceneLabel | None = None
    viewpoint: ViewpointLabel | None = None
    actions: list[ActionSegment] = []
    objects: list[ObjectAnnotation] = []
    hand_states: list[HandState] = []
    dropped_low_confidence: int = 0

    @property
    def needs_review(self) -> bool:
        items = [self.scene, self.viewpoint, *self.actions, *self.objects, *self.hand_states]
        return any(i is not None and i.needs_human_verification for i in items)


TAXONOMY_VERSION = "1.0.0"


def taxonomy_lists() -> dict[str, list[str]]:
    """Plain lists, used for prompts and the Label Studio config."""
    return {
        "actions": [a.value for a in Action],
        "object_categories": [c.value for c in ObjectCategory],
        "interaction_roles": [r.value for r in InteractionRole],
        "scenes": [s.value for s in Scene],
        "hand_poses": [h.value for h in HandPose],
        "bimanual_modes": [b.value for b in BimanualMode],
        "viewpoints": [v.value for v in Viewpoint],
    }
