"""Typed contracts shared by predicate-cover search and its consumers."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, Union

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.model_ids import FieldId
from griffin.query.evaluate import (
    PredicateCoverObservation,
    SelectionCoverObservation,
)
from griffin.query.model import (
    BindingFingerprint,
    HoleId,
    ParameterValue,
    StructuralSignature,
)
from griffin.query.observation import CoverPayload, PredicateCoverHint
from griffin.query.task import PredicateEvolutionStrategyKind


class PredicateCoverTargetMetric(enum.Enum):
    SELECTIVITY = "selectivity"


class PredicateCoverConsumerKind(enum.Enum):
    COMPLETE_PREDICATE_BINDING = "complete_predicate_binding"
    FILTERED_ANN_FRONTIER_SEED = "filtered_ann_frontier_seed"


class PredicateBindingFragment(msgspec.Struct, frozen=True, omit_defaults=True):
    template_signature: StructuralSignature
    values: Mapping[HoleId, ParameterValue]
    binding_fingerprint: BindingFingerprint


PredicateCoverObservationValue: TypeAlias = Union[
    PredicateCoverObservation,
    SelectionCoverObservation,
]


class PredicateCoverRejection(msgspec.Struct, frozen=True, omit_defaults=True):
    fragment_fingerprint: BindingFingerprint
    cover: CoverPayload
    target_metric: PredicateCoverTargetMetric
    observed_value: float
    target_minimum: float
    target_maximum: float
    reason: str
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)


class PredicateCoverBestSoFar(msgspec.Struct, frozen=True):
    """搜索失败时的历史最优候选：实测选择率、到 band 的距离、以及
    最接近 GLS band 的 hit_count（无 vector 命中时为 None）。"""

    observed_value: float
    objective_distance: float
    gls_hit_count: int | None = None


class PredicateCoverCandidate(msgspec.Struct, frozen=True, omit_defaults=True):
    fragment: PredicateBindingFragment
    observation: PredicateCoverObservationValue
    hint: PredicateCoverHint
    cover: CoverPayload
    target_metric: PredicateCoverTargetMetric
    observed_value: float
    objective_distance: float = 0.0
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective_distance != 0.0:
            raise ContractViolation("PredicateCoverCandidate must be usable-only")


@dataclass(frozen=True)
class PredicateCoverPlanKey:
    consumer_kind: PredicateCoverConsumerKind
    evolution_strategy: PredicateEvolutionStrategyKind
    template_signature: StructuralSignature
    cover_node_id: str
    answer_field: FieldId
    target_metric: PredicateCoverTargetMetric
    target_minimum: float
    target_maximum: float


def predicate_target_distance(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return float(minimum - value)
    if value > maximum:
        return float(value - maximum)
    return 0.0


__all__ = [
    "PredicateBindingFragment",
    "PredicateCoverBestSoFar",
    "PredicateCoverCandidate",
    "PredicateCoverConsumerKind",
    "PredicateCoverObservationValue",
    "PredicateCoverPlanKey",
    "PredicateCoverRejection",
    "PredicateCoverTargetMetric",
    "predicate_target_distance",
]
