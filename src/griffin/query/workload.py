"""Compile-time workload plan and planning helpers for query generation."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import msgspec

from griffin.model_ids import FieldId
from griffin.query.admit import AcceptedBinding
from griffin.query.dataset_resources import DatasetResources
from griffin.query.domain import (
    DomainBuildProblem,
    DomainBuildResult,
    DomainBuilder,
    parameter_space_construction_cost,
    parameter_space_is_empty,
    parameter_space_logical_capacity,
)
from griffin.query.model import (
    CategoricalSetParameterSpace,
    HoleId,
    HoleValueKind,
    QueryTask,
    QueryTemplate,
    stable_signature,
)
from griffin.query.predicate_cover import PredicateCoverBestSoFar
from griffin.query.search_core import ParameterSolverKind
from griffin.query.ground_truth import GroundTruthPlan
from griffin.query.shape import FilterLogic
from griffin.query.spec import GenerationStats, QuerySpec
from griffin.query.task import ObjectiveKind


OBSERVATION_BUDGET_MULTIPLIER = 8

_reachability_logger = logging.getLogger("griffin.query.planning")


class QueryTaskBudget(msgspec.Struct, frozen=True, omit_defaults=True):
    max_attempts: int
    max_observations: int


class EvaluationTicketBudget(msgspec.Struct, frozen=True, omit_defaults=True):
    max_attempts: int
    max_observations: int


class ResourceSpend(msgspec.Struct, frozen=True, omit_defaults=True):
    attempts: int = 0
    observations: int = 0

    def add(self, other: "ResourceSpend") -> "ResourceSpend":
        return ResourceSpend(
            attempts=int(self.attempts) + int(other.attempts),
            observations=int(self.observations) + int(other.observations),
        )


class QueryTaskIdentity(msgspec.Struct, frozen=True, omit_defaults=True):
    root_seed: int
    prototype_signature: str
    occurrence_ordinal: int
    query_seed: int


class TemplatePredicateProfileKind(enum.Enum):
    NONE = "none"
    FLAT = "flat"
    OPAQUE = "opaque"


class TemplatePredicateSlot(msgspec.Struct, frozen=True, omit_defaults=True):
    hole_id: HoleId
    field: FieldId


class TemplatePlanningSummary(msgspec.Struct, frozen=True, omit_defaults=True):
    predicate_profile_kind: TemplatePredicateProfileKind = TemplatePredicateProfileKind.NONE
    predicate_slots: tuple[TemplatePredicateSlot, ...] = ()
    filter_logic: FilterLogic | None = None


class TemplateCandidate(msgspec.Struct, frozen=True, omit_defaults=True):
    template_signature: str
    template: QueryTemplate
    solver_kind: ParameterSolverKind
    ground_truth_plan: GroundTruthPlan
    planning: TemplatePlanningSummary = TemplatePlanningSummary()


class TemplateProfileRecord(msgspec.Struct, frozen=True, omit_defaults=True):
    eligible: bool
    estimated_yield: float = 0.0
    predicate_field_reuse: int = 0
    search_space_cost: int = 0
    selectivity_reachability_penalty: float = 0.0
    diagnostics: tuple[str, ...] = ()


class SelectedTemplate(msgspec.Struct, frozen=True, omit_defaults=True):
    query_task: QueryTask
    candidate: TemplateCandidate
    profile: TemplateProfileRecord
    base_domains: DomainBuildResult
    initial_order: int


class CompiledQueryTaskPlan(msgspec.Struct, frozen=True, omit_defaults=True):
    identity: QueryTaskIdentity
    query_spec: QuerySpec
    query_task: QueryTask
    templates: tuple[TemplateCandidate, ...]
    budget: QueryTaskBudget


class CompiledWorkloadPlan(msgspec.Struct, frozen=True, omit_defaults=True):
    workload_id: str
    root_seed: int
    query_tasks: tuple[CompiledQueryTaskPlan, ...]


class TemplateRuntimeFeedbackKind(enum.Enum):
    ACCEPTED = "accepted"
    NO_ACCEPTED_CANDIDATE = "no_accepted_candidate"
    SEARCH_EXHAUSTED = "search_exhausted"


class QueryTaskTerminalKind(enum.Enum):
    ACCEPTED = "accepted"
    SEARCH_EXHAUSTED = "search_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_ELIGIBLE_TEMPLATE = "no_eligible_template"
    SHUTDOWN_REQUESTED = "shutdown_requested"


class TemplateRuntimeFeedback(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: TemplateRuntimeFeedbackKind


class CommittedQueryTaskResult(msgspec.Struct, frozen=True, omit_defaults=True):
    query_spec: QuerySpec
    query_task: QueryTask
    ground_truth_plan: GroundTruthPlan
    accepted: AcceptedBinding


class QueryTaskRunResult(msgspec.Struct, frozen=True, omit_defaults=True):
    accepted: CommittedQueryTaskResult | None
    stats: GenerationStats
    spend: ResourceSpend
    terminal_kind: QueryTaskTerminalKind
    diagnostics: tuple[str, ...] = ()
    best_so_far: PredicateCoverBestSoFar | None = None


@dataclass(frozen=True)
class SelectedTemplatePool:
    templates: tuple[SelectedTemplate, ...]
    domain_options_scored: int = 0


class TemplatePoolPlanner:
    def __init__(self, *, domain_builder: DomainBuilder | None = None) -> None:
        self._domain_builder = domain_builder or DomainBuilder()

    def plan(
        self,
        plan: CompiledQueryTaskPlan,
        *,
        resources: DatasetResources,
    ) -> SelectedTemplatePool:
        selected_templates: list[SelectedTemplate] = []
        domain_options_scored = 0
        for candidate_index, pooled_candidate in enumerate(plan.templates):
            base_domains = self._domain_builder.build(
                DomainBuildProblem(
                    task=plan.query_task,
                    template=pooled_candidate.template,
                    resources=resources,
                )
            )
            domain_options_scored += _domain_option_count(base_domains)
            profile = _template_profile_record(
                query_task=plan.query_task,
                candidate=pooled_candidate,
                base_domains=base_domains,
            )
            if not profile.eligible:
                continue
            selected_templates.append(
                SelectedTemplate(
                    query_task=plan.query_task,
                    candidate=pooled_candidate,
                    profile=profile,
                    base_domains=base_domains,
                    initial_order=candidate_index,
                )
            )
        ordered = sorted(
            selected_templates,
            key=_selected_template_order_key,
        )
        finalized = tuple(
            SelectedTemplate(
                query_task=selected.query_task,
                candidate=selected.candidate,
                profile=selected.profile,
                base_domains=selected.base_domains,
                initial_order=index,
            )
            for index, selected in enumerate(ordered)
        )
        return SelectedTemplatePool(
            templates=finalized,
            domain_options_scored=domain_options_scored,
        )


def _template_profile_record(
    *,
    query_task: QueryTask,
    candidate: TemplateCandidate,
    base_domains: DomainBuildResult,
) -> TemplateProfileRecord:
    diagnostics: list[str] = []
    empty_domains = tuple(
        domain.hole_id
        for domain in base_domains.domains
        if parameter_space_is_empty(domain)
    )
    if empty_domains:
        return TemplateProfileRecord(
            eligible=False,
            diagnostics=tuple(f"domain for {hole_id} has no options" for hole_id in empty_domains),
        )
    requires_topk_available = any(
        objective.kind is ObjectiveKind.TOPK_AVAILABLE
        for objective in query_task.objectives
    )
    if requires_topk_available and query_task.result_semantics.k is None:
        return TemplateProfileRecord(
            eligible=False,
            diagnostics=("template does not satisfy top-k availability gate",),
        )
    capacity = _binding_capacity(base_domains)
    estimated_yield = float(capacity)
    return TemplateProfileRecord(
        eligible=True,
        estimated_yield=estimated_yield,
        predicate_field_reuse=_predicate_field_reuse(candidate.planning),
        search_space_cost=_search_space_cost(base_domains),
        selectivity_reachability_penalty=_selectivity_reachability_penalty(
            query_task=query_task,
            candidate=candidate,
            base_domains=base_domains,
        ),
        diagnostics=tuple(diagnostics),
    )


def _continuation_key(selected: SelectedTemplate) -> str:
    if not _shares_vector_continuation(selected):
        return "target"
    return stable_signature(
        "support",
        {
            "template": selected.candidate.template_signature,
            "source_intent": selected.query_task.source_intent,
            "result_semantics": selected.query_task.result_semantics,
            "objectives": selected.query_task.objectives,
        },
    )


def _shares_vector_continuation(selected: SelectedTemplate) -> bool:
    if selected.candidate.solver_kind is not ParameterSolverKind.VECTOR:
        return False
    value_kinds = {domain.value_kind for domain in selected.base_domains.domains}
    return HoleValueKind.QUERY_VECTOR in value_kinds and any(
        kind is not HoleValueKind.QUERY_VECTOR for kind in value_kinds
    )


def query_task_budget_from_spec(query_spec: QuerySpec) -> QueryTaskBudget:
    max_candidates = max(int(query_spec.budget.max_candidates), 1)
    return QueryTaskBudget(
        max_attempts=max_candidates,
        max_observations=max(max_candidates * OBSERVATION_BUDGET_MULTIPLIER, 1),
    )


def _domain_option_count(domains: DomainBuildResult) -> int:
    return sum(parameter_space_construction_cost(domain) for domain in domains.domains)


def _selected_template_order_key(
    selected: SelectedTemplate,
) -> tuple[float, int, int, float, int]:
    return (
        float(selected.profile.selectivity_reachability_penalty),
        int(selected.profile.predicate_field_reuse),
        int(selected.profile.search_space_cost),
        -float(selected.profile.estimated_yield),
        int(selected.initial_order),
    )


def _predicate_field_reuse(planning: TemplatePlanningSummary) -> int:
    fields = tuple((slot.field.owner.name, slot.field.name) for slot in planning.predicate_slots)
    if len(fields) <= 1:
        return 0
    return len(fields) - len(set(fields))


def _selectivity_reachability_penalty(
    *,
    query_task: QueryTask,
    candidate: TemplateCandidate,
    base_domains: DomainBuildResult,
) -> float:
    # 可达性检测已关闭：该检测只对单字段 in/eq 模板（CategoricalSet 空间）计算
    # 理论上限，但实际谓词是多个字段联合（AND/OR 组合可达更高选择率），
    # 单字段上限不能代表组合上限，误杀大量可生成的模板。改为交给搜索本身
    # 判定（配合 per-query 时间预算：搜不到就跳过）。
    return 0.0
    logic = candidate.planning.filter_logic  # noqa: unreachable — 保留原实现备查
    logic = candidate.planning.filter_logic
    slots = candidate.planning.predicate_slots
    objective = next(
        (
            item
            for item in query_task.objectives
            if item.kind is ObjectiveKind.SELECTIVITY_BAND
            and item.minimum is not None
            and item.maximum is not None
        ),
        None,
    )
    if (
        objective is None
        or objective.minimum is None
        or objective.maximum is None
        or logic is None
        or not slots
    ):
        return 0.0
    target_minimum = float(objective.minimum)
    target_maximum = float(objective.maximum)
    bounds: list[tuple[float, float]] = []
    for slot in slots:
        space = base_domains.domain_for(slot.hole_id)
        if not isinstance(space, CategoricalSetParameterSpace):
            return 0.0
        bounds.append(_categorical_set_local_bounds(space))
    if logic is FilterLogic.AND:
        predicted_minimum = 1.0
        predicted_maximum = 1.0
        for lower, upper in bounds:
            predicted_minimum *= lower
            predicted_maximum *= upper
    else:
        minimum_complement = 1.0
        maximum_complement = 1.0
        for lower, upper in bounds:
            minimum_complement *= 1.0 - lower
            maximum_complement *= 1.0 - upper
        predicted_minimum = 1.0 - minimum_complement
        predicted_maximum = 1.0 - maximum_complement
    if predicted_maximum < target_minimum:
        _reachability_logger.warning(
            "模板谓词 %s 理论不可达目标选择率：可达区间 [%.4f, %.4f] 与 band [%.4f, %.4f] 不相交",
            ",".join(slot.field.name for slot in slots),
            predicted_minimum,
            predicted_maximum,
            target_minimum,
            target_maximum,
        )
        return target_minimum - predicted_maximum
    if predicted_minimum > target_maximum:
        _reachability_logger.warning(
            "模板谓词 %s 理论不可达目标选择率：可达区间 [%.4f, %.4f] 与 band [%.4f, %.4f] 不相交",
            ",".join(slot.field.name for slot in slots),
            predicted_minimum,
            predicted_maximum,
            target_minimum,
            target_maximum,
        )
        return predicted_minimum - target_maximum
    return 0.0


def _categorical_set_local_bounds(
    space: CategoricalSetParameterSpace,
) -> tuple[float, float]:
    if not space.atoms or space.owner_row_count <= 0:
        return (0.0, 0.0)
    counts = sorted((atom.local_count for atom in space.atoms), reverse=True)
    lower = min(counts) / float(space.owner_row_count)
    upper = sum(counts[: space.max_size]) / float(space.owner_row_count)
    return (
        min(max(lower, 0.0), 1.0),
        min(max(upper, 0.0), 1.0),
    )


def _binding_capacity(base_domains: DomainBuildResult) -> int:
    capacity = 1
    for domain in base_domains.domains:
        remaining_cap = max(1_000_000 // max(capacity, 1), 1)
        domain_capacity = max(
            parameter_space_logical_capacity(domain, cap=remaining_cap),
            1,
        )
        capacity = min(capacity * domain_capacity, 1_000_000)
    return capacity


def _search_space_cost(base_domains: DomainBuildResult) -> int:
    cost = 0
    for domain in base_domains.domains:
        if domain.value_kind is HoleValueKind.QUERY_VECTOR:
            continue
        cost = min(
            cost + max(parameter_space_construction_cost(domain), 1),
            1_000_000_000,
        )
    return max(cost, 1)


__all__ = [
    "CommittedQueryTaskResult",
    "CompiledQueryTaskPlan",
    "CompiledWorkloadPlan",
    "EvaluationTicketBudget",
    "QueryTaskBudget",
    "QueryTaskIdentity",
    "QueryTaskTerminalKind",
    "ResourceSpend",
    "TemplateRuntimeFeedback",
    "TemplateRuntimeFeedbackKind",
    "SelectedTemplate",
    "SelectedTemplatePool",
    "TemplateCandidate",
    "TemplatePlanningSummary",
    "TemplatePoolPlanner",
    "TemplatePredicateProfileKind",
    "TemplatePredicateSlot",
    "TemplateProfileRecord",
    "QueryTaskRunResult",
    "query_task_budget_from_spec",
]
