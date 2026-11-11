"""Typed solver lifecycle state for query parameter search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from griffin.query.model import BindingFingerprint, BindingSignature
from griffin.query.predicate_cover import (
    PredicateCoverCandidate,
    PredicateCoverPlanKey,
)
from griffin.query.predicate_evolution import PredicateEvolutionState

if TYPE_CHECKING:
    from griffin.query.domain import DomainBuildResult
    from griffin.query.vector_search import FilteredAnnPendingProposal, FilteredAnnVectorFrontier


@dataclass
class _PredicateCoverSearchRunnerState:
    evolution_state: PredicateEvolutionState = field(
        default_factory=PredicateEvolutionState
    )
    observed_fingerprints: set[BindingFingerprint] = field(default_factory=set)
    usable_candidates: list[PredicateCoverCandidate] = field(default_factory=list)
    rejection_count: int = 0
    rejection_reason: str | None = None
    budget_reason: str | None = None
    budget_exhausted: bool = False
    exhausted: bool = False


@dataclass
class PredicateCoverPlanStateStore:
    by_plan: dict[PredicateCoverPlanKey, _PredicateCoverSearchRunnerState] = field(default_factory=dict)


@dataclass
class FilteredAnnSearchState:
    frontiers: dict[BindingSignature, "FilteredAnnVectorFrontier"] = field(default_factory=dict)
    pending_proposals: dict[BindingSignature, "FilteredAnnPendingProposal"] = field(default_factory=dict)
    infeasible_reason: str | None = None
    # 搜索失败时最接近 GLS band 的 hit_count（best_so_far 的 GLS 维度）
    best_so_far_hit_count: int | None = None


@dataclass
class VectorSearchState:
    domains: "DomainBuildResult | None" = None
    filtered_ann: FilteredAnnSearchState = field(default_factory=FilteredAnnSearchState)


@dataclass
class ParameterSearchState:
    attempted_bindings: set[BindingSignature] = field(default_factory=set)
    excluded_binding_signatures: set[BindingSignature] = field(default_factory=set)
    parameter_attempts: int = 0
    predicate_cover: PredicateCoverPlanStateStore = field(default_factory=PredicateCoverPlanStateStore)
    vector: VectorSearchState = field(default_factory=VectorSearchState)


__all__ = [
    "FilteredAnnSearchState",
    "ParameterSearchState",
    "PredicateCoverPlanKey",
    "PredicateCoverPlanStateStore",
    "VectorSearchState",
]
