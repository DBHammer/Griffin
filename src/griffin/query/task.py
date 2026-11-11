"""Query typed semantics: task definitions, objectives, output types, and reports.

Unified module covering:
  - task / task predicate (TypedTask, SelectivityTask, …)
  - objective & ground truth (Objective, ObjectiveKind, GroundTruthSpec, …)
  - output (TopKOutput, TypedOutput)
  - public report (PublicTaskKind, PublicTaskReport)
"""

from __future__ import annotations

import enum
from typing import Union

import msgspec

from griffin.model_ids import FieldId


class ResultSemanticsKind(enum.Enum):
    SCALAR_COUNT = "scalar_count"
    ENTITY_SET = "entity_set"
    RANKED_ENTITIES = "ranked_entities"


class ResultSemantics(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: ResultSemanticsKind
    k: int | None = None


# ── task & predicates ──────────────────────────────────────────────

class Hardness(enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ToleranceTarget(msgspec.Struct, frozen=True, tag="tolerance_target", tag_field="predicate_kind"):
    target: float
    tolerance: float


class PredicateEvolutionStrategyKind(enum.Enum):
    GUIDED = "guided"
    STANDARD = "standard"


class SelectivityTask(msgspec.Struct, frozen=True, tag="selectivity", tag_field="task_kind"):
    target: ToleranceTarget


class FilteredAnnTask(msgspec.Struct, frozen=True, tag="filtered_ann", tag_field="task_kind"):
    vector_field: FieldId
    selectivity: ToleranceTarget | None
    difficulty: Hardness
    gls: ToleranceTarget | None = None


TypedTask = Union[
    SelectivityTask,
    FilteredAnnTask,
]


# ── objectives ─────────────────────────────────────────────────────

class ObjectiveKind(enum.Enum):
    SELECTIVITY_BAND = "selectivity_band"
    FILTERED_ANN_DIFFICULTY_BAND = "filtered_ann_difficulty_band"
    TOPK_AVAILABLE = "topk_available"


class ObjectiveMetric(enum.Enum):
    SELECTIVITY = "selectivity"
    FILTERED_ANN_HIT_COUNT = "filtered_ann_hit_count"
    TOPK_AVAILABLE = "available_topk_count"


class FilteredAnnGLSTargetMode(enum.Enum):
    MIN = "min"
    MAX = "max"
    RANDOM = "rnd"


class Objective(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: ObjectiveKind
    metric: ObjectiveMetric
    minimum: float | None = None
    maximum: float | None = None
    k: int | None = None
    hardness: Hardness | None = None
    filtered_ann_gls_target_mode: FilteredAnnGLSTargetMode | None = None


class GroundTruthMaterialization(enum.Enum):
    NO_PUBLIC_GROUND_TRUTH = "no_public_ground_truth"
    EXACT_TOPK_ROWIDS = "exact_topk_rowids"


class GroundTruthSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    result_semantics: ResultSemantics
    materialization: GroundTruthMaterialization


# ── output ─────────────────────────────────────────────────────────

class TopKOutput(msgspec.Struct, frozen=True, tag="topk", tag_field="output_kind"):
    k: int


TypedOutput = TopKOutput


# ── public report ──────────────────────────────────────────────────

class PublicTaskKind(enum.Enum):
    SELECTIVITY = "selectivity"
    FILTERED_ANN = "filtered_ann"


class PublicTaskReport(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: PublicTaskKind
    achieved: float
    target: float
    fields: tuple[FieldId, ...] = ()
    metadata_selectivity: float | None = None


__all__ = [
    "FilteredAnnGLSTargetMode",
    "FilteredAnnTask",
    "GroundTruthMaterialization",
    "GroundTruthSpec",
    "Hardness",
    "Objective",
    "ObjectiveKind",
    "ObjectiveMetric",
    "PredicateEvolutionStrategyKind",
    "PublicTaskKind",
    "PublicTaskReport",
    "ResultSemantics",
    "ResultSemanticsKind",
    "SelectivityTask",
    "ToleranceTarget",
    "TopKOutput",
    "TypedOutput",
    "TypedTask",
]
