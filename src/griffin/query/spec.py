"""External entry point: QuerySpec and workload output types.

A QuerySpec is the typed request produced by the public config compiler.
The generation pipeline lowers it to QueryTask + QueryTemplate, then runs
binding, verification, judgement, admission, and finalization on that mainline.

Workload types (GroundTruth, AcceptedQuery, QueryWorkload) are the typed
output contract produced by the pipeline.
"""

from __future__ import annotations

import enum
from typing import Any

import msgspec

from griffin.dialects.capabilities import DialectCapabilities
from griffin.dialects.base import RenderedQuery
from griffin.model_ids import EntityId
from griffin.query.task import PredicateEvolutionStrategyKind, PublicTaskReport
from griffin.query.task import TypedOutput
from griffin.query.shape import TypedShape
from griffin.query.task import TypedTask


class ScopePayload(msgspec.Struct, frozen=True, omit_defaults=True):
    primary_entity: EntityId


class BudgetSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    max_candidates: int = 1000


class CapabilityPolicy(msgspec.Struct, frozen=True, omit_defaults=True):
    require_native: bool = False


class DialectTarget(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    capabilities: DialectCapabilities


class QuerySpec(msgspec.Struct, frozen=True, omit_defaults=True):
    """Typed external query spec.

    `shape` is a tagged TypedShape. `task` is a tagged TypedTask. `output` is
    only present for ranked retrieval queries. The whole struct round-trips
    through codec.
    """

    name: str
    scope: ScopePayload
    shape: TypedShape
    task: TypedTask
    output: TypedOutput | None = None
    budget: BudgetSpec = BudgetSpec()
    predicate_evolution_strategy: PredicateEvolutionStrategyKind = (
        PredicateEvolutionStrategyKind.STANDARD
    )
    best_so_far_fallback: bool = False


class QueryPipelineSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    seed: int
    dataset_root: str
    dialect: DialectTarget
    queries: tuple[QuerySpec, ...]
    capability_policy: CapabilityPolicy = CapabilityPolicy()


class GroundTruthAnswerKind(enum.Enum):
    FILTERED_TOPK = "filtered_topk"
    SCALAR = "scalar"


class GroundTruth(msgspec.Struct, frozen=True, omit_defaults=True):
    answer_kind: GroundTruthAnswerKind
    metric: str
    row_id_column: str
    k: int
    row_count: int
    answers: tuple[dict[str, Any], ...]
    answer_scope: dict[str, Any] = msgspec.field(default_factory=dict)


class PredicatePlanLogic(enum.Enum):
    AND = "and"
    OR = "or"


class PredicateAtomScopeKind(enum.Enum):
    ENTITY = "entity"


class EntityPredicateScope(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: PredicateAtomScopeKind
    owner: str
    alias: str


PredicateAtomScope = EntityPredicateScope
PredicateBindingSummary = dict[str, Any] | str | int | float | bool | None


class PredicateAtomRecord(msgspec.Struct, frozen=True, omit_defaults=True):
    field: str
    operator: str
    scope: PredicateAtomScope
    binding: PredicateBindingSummary
    summary: str


class AcceptedPredicatePlan(msgspec.Struct, frozen=True, omit_defaults=True):
    actual_predicate_count: int
    actual_filter_logic: PredicatePlanLogic
    atoms: tuple[PredicateAtomRecord, ...]


class CandidateAnalysis(msgspec.Struct, frozen=True, omit_defaults=True):
    accepted: bool
    score: float
    public_report: PublicTaskReport | None = None
    diagnostics: str = ""


class AcceptedQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    query_id: str
    spec: QuerySpec
    rendered: RenderedQuery
    analysis: CandidateAnalysis
    ground_truth: GroundTruth | None = None
    accepted_predicate_plan: AcceptedPredicatePlan | None = None


class GenerationStats(msgspec.Struct, frozen=True, omit_defaults=True):
    total_attempts: int
    accepted: int
    rejected: int
    canonical_evaluations: int = 0
    hint_observations: int = 0
    seed_hint_observations: int = 0
    local_search_hint_observations: int = 0
    full_search_calls: int = 0
    cover_computations: int = 0
    domain_options_scored: int = 0


class QueryWorkload(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    accepted: tuple[AcceptedQuery, ...]
    stats: GenerationStats


def ground_truth_count(workload: QueryWorkload) -> int:
    return sum(1 for query in workload.accepted if query.ground_truth is not None)


def accepted_predicate_plan_to_public_dict(plan: AcceptedPredicatePlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "actual_predicate_count": plan.actual_predicate_count,
        "actual_filter_logic": plan.actual_filter_logic.value,
        "atoms": [_predicate_atom_to_public_dict(atom) for atom in plan.atoms],
    }
    return payload


def _predicate_atom_to_public_dict(atom: PredicateAtomRecord) -> dict[str, Any]:
    return {
        "field": atom.field,
        "operator": atom.operator,
        "scope": _predicate_scope_to_public_dict(atom.scope),
        "binding": atom.binding,
        "summary": atom.summary,
    }


def _predicate_scope_to_public_dict(scope: PredicateAtomScope) -> dict[str, Any]:
    return {
        "kind": scope.kind.value,
        "owner": scope.owner,
        "alias": scope.alias,
    }


__all__ = [
    "AcceptedPredicatePlan",
    "AcceptedQuery",
    "BudgetSpec",
    "CandidateAnalysis",
    "CapabilityPolicy",
    "DialectTarget",
    "EntityPredicateScope",
    "GenerationStats",
    "GroundTruth",
    "GroundTruthAnswerKind",
    "PredicateAtomRecord",
    "PredicateAtomScopeKind",
    "PredicateBindingSummary",
    "PredicatePlanLogic",
    "QuerySpec",
    "QueryPipelineSpec",
    "QueryWorkload",
    "ScopePayload",
    "accepted_predicate_plan_to_public_dict",
    "ground_truth_count",
]
