"""Filtered ANN vector search strategy."""

from __future__ import annotations

import enum
import math
import random
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field as dataclass_field
from typing import Protocol

import numpy as np

from griffin.dataset.deterministic import RegionLayoutModel, RegionRange
from griffin.dataset.exact_cover import (
    deterministic_owner_metadata,
    intersect_intervals,
    intervals_row_count,
)
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.evaluate import FilteredAnnSearchEvidence
from griffin.query.domain import (
    DomainBuildResult,
    hole_context_for_template,
    parameter_space_is_empty,
)
from griffin.query.filtered_ann_band import (
    format_hit_band_diagnostic,
    gls_loss,
    gls_target_hit_band,
    hit_band_for_gls,
    nearest_reachable_gls,
    reachable_gls_band,
    resolve_search_gls_target_for_mode,
    resolve_hit_band,
)
from griffin.query.model import (
    AggregateNode,
    BindingSignature,
    EnumeratedParameterSpace,
    HoleValueKind,
    LimitNode,
    ParameterBinding,
    ParameterSpace,
    ParameterValue,
    QueryTask,
    RankNode,
    SearchIdentity,
    SearchSeedStage,
    SelectNode,
    TemplateNode,
    VectorValue,
    binding_signature,
    derive_search_seed,
)
from griffin.query.observation import (
    CoverPayload,
    FilteredAnnHitHint,
    HintCostStage,
    ObservationRequestKind,
    ObservationRequest,
    VectorSourceHint,
)
from griffin.query.predicate_cover import (
    PredicateCoverCandidate,
    PredicateCoverConsumerKind,
    PredicateCoverTargetMetric,
)
from griffin.query.search_core import (
    BindingProblem,
    CandidateProposal,
    FilteredAnnPreparedCandidateObservations,
    ParameterFeedback,
    ParameterSearchState,
    PredicatePreparedCandidateObservations,
    PredicateCoverConsumerSpec,
    PredicateCoverSearchPlan,
    PredicateCoverSearchResult,
    apply_parameter_feedback,
    next_binding_from_domains,
    observe_search_hint,
    build_predicate_cover_plan,
    predicate_cover_search_diagnostic_reason,
    profiled_predicate_cover_search,
    proposal_domain_order,
)
from griffin.query.task import Hardness, Objective, ObjectiveKind, ObjectiveMetric
from griffin.vector_hardness_core import metric_query


FILTERED_ANN_SEED_LIMIT = 768
FILTERED_ANN_INITIAL_FRONTIER_LIMIT = 128
FILTERED_ANN_LOCAL_SEARCH_HINT_LIMIT = 128

_FILTERED_ANN_SEED_MODE_ENV = "FILTERED_ANN_SEED_MODE"


class FilteredAnnSeedMode(enum.Enum):
    """Pluggable seed-selection policy for query-vector search."""

    RANDOM = "random"  # mode 1: random start, may lie outside the vector dataset
    REGION_GUIDED = "region_guided"  # mode 2: estimated start via region filter density


def filtered_ann_seed_mode() -> FilteredAnnSeedMode:
    """Resolve the active seed mode; overridable via FILTERED_ANN_SEED_MODE."""
    raw = os.environ.get(_FILTERED_ANN_SEED_MODE_ENV, "").strip().lower()
    if not raw or raw == FilteredAnnSeedMode.REGION_GUIDED.value:
        return FilteredAnnSeedMode.REGION_GUIDED
    if raw == FilteredAnnSeedMode.RANDOM.value:
        return FilteredAnnSeedMode.RANDOM
    raise ContractViolation(
        "FILTERED_ANN_SEED_MODE must be one of "
        f"{FilteredAnnSeedMode.RANDOM.value}/{FilteredAnnSeedMode.REGION_GUIDED.value}; got {raw!r}"
    )


_FIXED_COVER_SELECTIVITY_ENV = "MMBENCHGEN_FIXED_COVER_SELECTIVITY"


def fixed_cover_selectivity() -> float | None:
    """消融（w/o 向量搜索）：读 griffin 固定 predicate 的目标 selectivity。

    非 None 时，predicate cover search 只保留实测 selectivity 与该值
    （四舍五入到整数行数）一致的 candidate，从而复用 griffin 的 predicate，
    只让 GLS 搜索环节（向量初始化 region_guided vs random）不同。
    """
    raw = os.environ.get(_FIXED_COVER_SELECTIVITY_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class FilteredAnnVectorSearchBudget:
    seed_limit: int
    initial_frontier_limit: int
    total_hint_limit: int


@dataclass(frozen=True)
class FilteredAnnVectorSearchTask:
    query_task: QueryTask
    search_identity: SearchIdentity
    field: FieldId
    vector_hole_id: str
    base_values: Mapping[str, ParameterValue]
    cover: CoverPayload
    top_k: int
    band: tuple[int, int]
    hardness: Hardness
    budget: FilteredAnnVectorSearchBudget
    source: VectorSourceHint
    seed_vectors: tuple[VectorValue, ...]
    initial_step: float


class FilteredAnnVectorSearchAlgorithm(Protocol):
    def create_frontier(self, task: FilteredAnnVectorSearchTask) -> "FilteredAnnVectorFrontier | None": ...

    def extend_frontier(
        self,
        frontier: "FilteredAnnVectorFrontier",
        current: VectorValue,
        hint: FilteredAnnHitHint,
        support: "FilteredAnnTopKSupport",
    ) -> None: ...


@dataclass
class FilteredAnnVectorFrontier:
    field: FieldId
    vector_hole_id: str
    base_values: dict[str, ParameterValue]
    cover: CoverPayload
    top_k: int
    band: tuple[int, int]
    budget: FilteredAnnVectorSearchBudget
    source: VectorSourceHint
    queue: list[tuple[VectorValue, HintCostStage]]
    seen: set[tuple[float, ...]]
    rng: np.random.Generator
    step: float
    hint_observations: int = 0
    anneal: "FilteredAnnAnnealState | None" = None
    exhausted: bool = False
    last_hints: dict[tuple[float, ...], FilteredAnnHitHint] = dataclass_field(default_factory=dict)


@dataclass
class FilteredAnnAnnealState:
    """Simulated-annealing chain state for one filtered ANN frontier."""

    phase: str = "seed"  # "seed" while evaluating the seed pool, then "anneal"
    seed_losses: list[tuple[tuple[float, ...], float, int, VectorValue]] = dataclass_field(
        default_factory=list
    )
    chain: VectorValue | None = None
    chain_loss: float = float("inf")
    chain_hit_count: int = 0
    best: VectorValue | None = None
    best_loss: float = float("inf")
    iteration: int = 0
    initial_temperature: float = 1.0
    processed: set[tuple[float, ...]] = dataclass_field(default_factory=set)


@dataclass(frozen=True)
class FilteredAnnPendingProposal:
    frontier_key: BindingSignature
    vector_components: tuple[float, ...]


@dataclass(frozen=True)
class FilteredAnnTopKSupport:
    rowids: tuple[int, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class FilteredAnnFrontierAnnealAlgorithm:
    """Simulated annealing over query vectors (paper §V-C).

    The seed pool is exactly evaluated first; the seed with the smallest
    GLS error becomes the chain start q0.  Each annealing step randomly
    perturbs the chain state; a worse candidate is accepted with
    probability exp(-ΔE/T_t) where T_t = T0/t.  The best solution ever
    seen is retained as the frontier's best.
    """

    def create_frontier(self, task: FilteredAnnVectorSearchTask) -> FilteredAnnVectorFrontier | None:
        frontier = _create_filtered_ann_frontier(task, task.seed_vectors)
        if frontier is None:
            return None
        frontier.anneal = FilteredAnnAnnealState(
            initial_temperature=max(1.0, float(task.top_k) / 2.0),
        )
        return frontier

    def extend_frontier(
        self,
        frontier: FilteredAnnVectorFrontier,
        current: VectorValue,
        hint: FilteredAnnHitHint,
        support: FilteredAnnTopKSupport,
    ) -> None:
        _extend_filtered_ann_frontier_anneal(frontier, current, hint, support)


_DEFAULT_FILTERED_ANN_VECTOR_SEARCH_ALGORITHM: FilteredAnnVectorSearchAlgorithm = FilteredAnnFrontierAnnealAlgorithm()


def _create_filtered_ann_frontier(
    task: FilteredAnnVectorSearchTask,
    vectors: tuple[VectorValue, ...],
) -> FilteredAnnVectorFrontier | None:
    seed_vectors = _dedupe_vector_values(vectors)
    if not seed_vectors:
        return None
    queued = seed_vectors[: task.budget.initial_frontier_limit]
    return FilteredAnnVectorFrontier(
        field=task.field,
        vector_hole_id=task.vector_hole_id,
        base_values=dict(task.base_values),
        cover=task.cover,
        top_k=task.top_k,
        band=task.band,
        budget=task.budget,
        source=task.source,
        queue=[(value, HintCostStage.FILTERED_ANN_SEED) for value in queued],
        seen={value.components for value in queued},
        rng=np.random.default_rng(
            _vector_option_seed(
                task.search_identity,
                task.query_task,
                task.field,
                stage=SearchSeedStage.VECTOR_FRONTIER,
                components=(task.cover.count, 0x9E3779B9),
            )
        ),
        step=task.initial_step,
    )


class VectorParameterSearchStrategy:
    """Vector parameter search strategy.

    The strategy may use FAISS-backed observations as frontier feedback for
    query-vector search. It does not produce metrics, evidence, judgement, or
    public truth; those remain owned by the evaluator.
    """

    def __init__(
        self,
        *,
        filtered_ann_algorithm: FilteredAnnVectorSearchAlgorithm | None = None,
    ) -> None:
        self._filtered_ann_algorithm = filtered_ann_algorithm or _DEFAULT_FILTERED_ANN_VECTOR_SEARCH_ALGORITHM

    def next(self, problem: BindingProblem, state: ParameterSearchState) -> CandidateProposal:
        prepared = _vector_prepared_problem(problem, state)
        if _is_filtered_ann_vector_problem(prepared):
            return _next_filtered_ann_vector_binding(prepared, state, self._filtered_ann_algorithm)
        return next_binding_from_domains(
            prepared,
            state,
            proposal_domain_order(prepared.domains.domains),
            exhausted_reason="all vector parameter bindings attempted or excluded",
        )

    def observe(
        self,
        problem: BindingProblem,
        state: ParameterSearchState,
        feedback: ParameterFeedback,
    ) -> ParameterSearchState:
        prepared = _vector_prepared_problem(problem, state)
        if _is_filtered_ann_vector_problem(prepared):
            _observe_filtered_ann_vector_feedback(prepared, state, feedback, self._filtered_ann_algorithm)
        return apply_parameter_feedback(state, feedback)


def _vector_prepared_problem(problem: BindingProblem, state: ParameterSearchState) -> BindingProblem:
    domains = state.vector.domains
    if domains is None:
        domains = _vector_domains(problem)
        state.vector.domains = domains
    return BindingProblem(
        task=problem.task,
        search_identity=problem.search_identity,
        template=problem.template,
        domains=domains,
        budget=problem.budget,
        solver_kind=problem.solver_kind,
        resources=problem.resources,
        feedback=problem.feedback,
        observation_session=problem.observation_session,
        observation_scope=problem.observation_scope,
    )


def _vector_domains(problem: BindingProblem) -> DomainBuildResult:
    return problem.domains


def _is_filtered_ann_vector_problem(problem: BindingProblem) -> bool:
    if not any(objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND for objective in problem.task.objectives):
        return False
    if not any(domain.value_kind is HoleValueKind.QUERY_VECTOR for domain in problem.domains.domains):
        return False
    return any(domain.value_kind is not HoleValueKind.QUERY_VECTOR for domain in problem.domains.domains)


def _next_filtered_ann_vector_binding(
    problem: BindingProblem,
    state: ParameterSearchState,
    algorithm: FilteredAnnVectorSearchAlgorithm,
) -> CandidateProposal:
    if problem.resources is None:
        return CandidateProposal.no_proposal("filtered ANN vector binding requires resources")
    domains = proposal_domain_order(problem.domains.domains)
    vector_domain = next((domain for domain in domains if domain.value_kind is HoleValueKind.QUERY_VECTOR), None)
    if vector_domain is None:
        return CandidateProposal.no_proposal("filtered ANN vector binding requires query vector hole")
    if not isinstance(vector_domain, EnumeratedParameterSpace):
        return CandidateProposal.no_proposal("filtered ANN query vector requires enumerated parameter space")
    if parameter_space_is_empty(vector_domain):
        return CandidateProposal.no_proposal(f"domain for {vector_domain.hole_id} has no options")
    predicate_domains = tuple(domain for domain in domains if domain.value_kind is not HoleValueKind.QUERY_VECTOR)
    for domain in predicate_domains:
        if parameter_space_is_empty(domain):
            return CandidateProposal.no_proposal(f"domain for {domain.hole_id} has no options")
    vector_field = hole_context_for_template(problem.template.root, vector_domain.hole_id).field
    if vector_field is None:
        return CandidateProposal.no_proposal("filtered ANN query vector hole is not bound to a vector field")
    vector_order_input = _vector_order_input_node(problem.template.root)
    if vector_order_input is None:
        return CandidateProposal.no_proposal("filtered ANN template does not contain vector order input")
    cover_search = _filtered_ann_cover_search_plan(
        problem=problem,
        predicate_domains=predicate_domains,
        vector_order_input=vector_order_input,
        vector_field=vector_field,
    )
    fallback_candidate: PredicateCoverCandidate | None = None
    fixed_sel = fixed_cover_selectivity()
    for cover_result in profiled_predicate_cover_search(problem, state, cover_search):
        if cover_result.candidate is None:
            continue
        if fixed_sel is not None:
            # 消融（w/o 向量搜索）：固定 griffin 的 predicate（selectivity 四舍五入
            # 整数匹配），跳过其他 candidate，只在 griffin 的 predicate 上搜 vector。
            total = cover_result.candidate.cover.total_count
            if (
                round(float(cover_result.candidate.observed_value) * total)
                != round(fixed_sel * total)
            ):
                continue
        fallback_candidate = cover_result.candidate
        with _measure_search_profile(problem, "query.search.gls"):
            base_values = dict(fallback_candidate.fragment.values)
            partial_signature = binding_signature(problem.template, base_values)
            frontier = _filtered_ann_vector_frontier_for_cover(
                problem=problem,
                state=state,
                algorithm=algorithm,
                frontier_key=partial_signature,
                vector_domain=vector_domain,
                field=vector_field,
                cover_result=cover_result,
                base_values=base_values,
            )
            binding = (
                _next_filtered_ann_frontier_binding(
                    problem=problem,
                    state=state,
                    algorithm=algorithm,
                    frontier_key=partial_signature,
                    frontier=frontier,
                )
                if frontier is not None
                else None
            )
        if binding is not None:
            if frontier is None:
                raise ContractViolation(
                    "filtered ANN binding requires an active vector frontier"
                )
            vector_value = binding.values.get(frontier.vector_hole_id)
            if not isinstance(vector_value, VectorValue):
                raise ContractViolation(
                    "filtered ANN binding requires a bound query vector"
                )
            hit_hint = frontier.last_hints.get(vector_value.components)
            if hit_hint is None:
                raise ContractViolation(
                    "filtered ANN binding requires its exact winning hit observation"
                )
            return CandidateProposal.candidate(
                binding,
                prepared_observations=FilteredAnnPreparedCandidateObservations(
                    covers=(cover_result.candidate.hint,),
                    hit=hit_hint,
                ),
            )
        if filtered_ann_seed_mode() is FilteredAnnSeedMode.RANDOM:
            # 消融（w/o 向量搜索）：random 初始化下 vector 命中失败时，不换
            # 下一个 predicate（region_guided 会换），固定第一个 selectivity
            # 命中的 predicate，使 predicate 与 region_guided 完全一致，
            # 只让 GLS 搜索环节（向量初始化）不同。
            break
    if filtered_ann_seed_mode() is FilteredAnnSeedMode.RANDOM:
        # 跳过 fallback（随机向量补位）：random 初始化在固定 predicate 上
        # 搜不到命中 GLS 的 vector 就判失败，不换 predicate、不随机补位。
        return CandidateProposal.no_proposal(
            "random seed mode: fixed predicate, vector search failed to hit GLS band"
        )
    if fallback_candidate is not None and vector_domain.values:
        fallback_rng = random.Random(
            derive_search_seed(
                problem.search_identity,
                SearchSeedStage.VECTOR_FALLBACK,
                problem.task.source_intent.public_family.value,
                state.parameter_attempts,
            )
        )
        fb_vector = fallback_rng.choice(vector_domain.values)
        fb_values = dict(fallback_candidate.fragment.values)
        fb_values[vector_domain.hole_id] = fb_vector
        fb_signature = binding_signature(problem.template, fb_values)
        if fb_signature not in state.attempted_bindings:
            fallback_binding = ParameterBinding(
                template_signature=problem.template.structural_signature,
                values=fb_values,
                complete_values=True,
                binding_signature=fb_signature,
            )
            return CandidateProposal.candidate(
                fallback_binding,
                prepared_observations=PredicatePreparedCandidateObservations(
                    covers=(fallback_candidate.hint,),
                ),
            )
    reason = state.vector.filtered_ann.infeasible_reason
    if reason:
        return CandidateProposal.no_proposal(reason)
    cover_reason = predicate_cover_search_diagnostic_reason(state, cover_search)
    if cover_reason:
        return CandidateProposal.no_proposal(cover_reason)
    return CandidateProposal.no_proposal("all filtered ANN vector bindings attempted or infeasible")


def _measure_search_profile(
    problem: BindingProblem,
    name: str,
) -> AbstractContextManager[None]:
    if problem.observation_session is None:
        return nullcontext()
    return problem.observation_session.measure_profile(name)


def _filtered_ann_cover_search_plan(
    *,
    problem: BindingProblem,
    predicate_domains: tuple[ParameterSpace, ...],
    vector_order_input: TemplateNode,
    vector_field: FieldId,
) -> PredicateCoverSearchPlan:
    selectivity_target = _selectivity_band_target(problem)
    target_minimum, target_maximum = selectivity_target if selectivity_target is not None else (0.0, 1.0)
    return build_predicate_cover_plan(
        problem=problem,
        predicate_domains=predicate_domains,
        consumer=PredicateCoverConsumerSpec(
            kind=PredicateCoverConsumerKind.FILTERED_ANN_FRONTIER_SEED,
        ),
        cover_root=vector_order_input,
        answer_field=vector_field,
        target_metric=PredicateCoverTargetMetric.SELECTIVITY,
        target_minimum=target_minimum,
        target_maximum=target_maximum,
    )


def _vector_order_input_node(root: TemplateNode) -> TemplateNode | None:
    if isinstance(root, RankNode):
        return root.input
    if isinstance(root, LimitNode):
        return _vector_order_input_node(root.input)
    if isinstance(root, (SelectNode, AggregateNode)):
        return _vector_order_input_node(root.input)
    return None


def _filtered_ann_frontiers(state: ParameterSearchState) -> dict[BindingSignature, FilteredAnnVectorFrontier]:
    return state.vector.filtered_ann.frontiers


def _filtered_ann_pending_proposals(state: ParameterSearchState) -> dict[str, FilteredAnnPendingProposal]:
    return state.vector.filtered_ann.pending_proposals


def _filtered_ann_vector_frontier_for_cover(
    *,
    problem: BindingProblem,
    state: ParameterSearchState,
    algorithm: FilteredAnnVectorSearchAlgorithm,
    frontier_key: BindingSignature,
    vector_domain: EnumeratedParameterSpace,
    field: FieldId,
    cover_result: PredicateCoverSearchResult,
    base_values: dict[str, ParameterValue],
) -> FilteredAnnVectorFrontier | None:
    frontiers = _filtered_ann_frontiers(state)
    cached = frontiers.get(frontier_key)
    if cached is not None:
        return None if cached.exhausted else cached
    if not cover_result.usable:
        return None
    if cover_result.candidate is None:
        raise ContractViolation("filtered ANN cover result requires usable candidate")
    cover_payload = cover_result.candidate.cover
    top_k = _top_k(problem)
    objective = _filtered_ann_hit_objective(problem)
    if objective is None:
        return None
    hardness = objective.hardness
    if hardness is None:
        return None
    budget = _filtered_ann_vector_search_budget(problem)
    band: tuple[int, int] | None
    if objective.filtered_ann_gls_target_mode is not None:
        point = resolve_search_gls_target_for_mode(
            mode=objective.filtered_ann_gls_target_mode,
            total_count=cover_payload.total_count,
            cover_count=cover_payload.count,
            k=top_k,
            search_identity=problem.search_identity,
        )
        if point is None:
            state.vector.filtered_ann.infeasible_reason = (
                "filtered_ann GLS target mode infeasible: "
                f"mode={objective.filtered_ann_gls_target_mode.value}, "
                f"N={int(cover_payload.total_count)}, "
                f"M={int(cover_payload.count)}, "
                f"k={int(top_k)}"
            )
            return None
        band = gls_target_hit_band(target_hit_count=int(point.hit_count), k=top_k)
    elif objective.minimum is not None and objective.maximum is not None:
        band = hit_band_for_gls(
            total_count=cover_payload.total_count,
            cover_count=cover_payload.count,
            k=top_k,
            target_gls_min=float(objective.minimum),
            target_gls_max=float(objective.maximum),
        )
        if band is None:
            reachable = reachable_gls_band(
                total_count=cover_payload.total_count,
                cover_count=cover_payload.count,
                k=top_k,
            )
            target_center = (float(objective.minimum) + float(objective.maximum)) / 2.0
            nearest = nearest_reachable_gls(
                target_gls=target_center,
                total_count=cover_payload.total_count,
                cover_count=cover_payload.count,
                k=top_k,
            )
            reachable_text = "unknown"
            if reachable is not None:
                reachable_text = f"[{reachable[0]:.6f}, {reachable[1]:.6f}]"
            nearest_text = "unknown"
            if nearest is not None:
                nearest_text = f"hit_count={nearest[0]}, gls={nearest[1]:.6f}"
            state.vector.filtered_ann.infeasible_reason = (
                "filtered_ann GLS band infeasible: "
                f"target_gls=[{float(objective.minimum):.6f}, {float(objective.maximum):.6f}], "
                f"reachable_gls={reachable_text}, "
                f"nearest={nearest_text}"
            )
            return None
    else:
        band, diagnostic = resolve_hit_band(
            total_count=cover_payload.total_count,
            cover_count=cover_payload.count,
            k=top_k,
            hardness=hardness,
        )
        if band is None:
            state.vector.filtered_ann.infeasible_reason = format_hit_band_diagnostic(
                diagnostic
            )
            return None
    state.vector.filtered_ann.infeasible_reason = None
    source = _vector_source_hint(problem, field)
    if source is None:
        return None
    if problem.resources is None:
        return None
    region_layout = _owner_region_layout(problem.resources, field.owner, vector_field_name=field.name)
    seed_vectors = _filtered_ann_seed_vectors_for_cover(
        problem=problem,
        field=field,
        source=source,
        cover=cover_payload,
        top_k=top_k,
        band=band,
        budget=budget,
        region_layout=region_layout,
    )
    search_task = FilteredAnnVectorSearchTask(
        query_task=problem.task,
        search_identity=problem.search_identity,
        field=field,
        vector_hole_id=vector_domain.hole_id,
        base_values=base_values,
        cover=cover_payload,
        top_k=top_k,
        band=band,
        hardness=hardness,
        budget=budget,
        source=source,
        seed_vectors=seed_vectors,
        initial_step=_filtered_ann_initial_step(problem=problem, field=field, source=source),
    )
    frontier = algorithm.create_frontier(search_task)
    if frontier is None:
        return None
    frontiers[frontier_key] = frontier
    return frontier


def _next_filtered_ann_frontier_binding(
    *,
    problem: BindingProblem,
    state: ParameterSearchState,
    algorithm: FilteredAnnVectorSearchAlgorithm,
    frontier_key: BindingSignature,
    frontier: FilteredAnnVectorFrontier,
) -> ParameterBinding | None:
    best_hit: int | None = None
    best_loss: float | None = None
    while frontier.queue and frontier.hint_observations < frontier.budget.total_hint_limit:
        vector, stage = frontier.queue.pop(0)
        values: dict[str, ParameterValue] = {
            **frontier.base_values,
            frontier.vector_hole_id: vector,
        }
        signature = binding_signature(problem.template, values)
        if signature in state.excluded_binding_signatures or signature in state.attempted_bindings:
            continue
        hint = _filtered_ann_hit_hint(
            problem=problem,
            field=frontier.field,
            vector=vector,
            cover=frontier.cover,
            top_k=frontier.top_k,
            band=frontier.band,
            stage=stage,
        )
        if hint is None:
            frontier.exhausted = True
            return None
        _observe_filtered_ann_frontier_hint(problem, algorithm, frontier, vector, hint)
        if hint.loss != 0.0:
            if best_loss is None or hint.loss < best_loss:
                best_loss = hint.loss
                best_hit = hint.hit_count
            continue
        _filtered_ann_pending_proposals(state)[signature] = FilteredAnnPendingProposal(
            frontier_key=frontier_key,
            vector_components=vector.components,
        )
        return ParameterBinding(
            template_signature=problem.template.structural_signature,
            values=values,
            complete_values=True,
            binding_signature=signature,
        )
    frontier.exhausted = True
    if best_hit is not None:
        state.vector.filtered_ann.best_so_far_hit_count = best_hit
    return None


def _filtered_ann_hit_objective(problem: BindingProblem) -> Objective | None:
    for objective in problem.task.objectives:
        if (
            objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND
            and objective.metric is ObjectiveMetric.FILTERED_ANN_HIT_COUNT
        ):
            return objective
    return None


def _filtered_ann_seed_vectors_for_cover(
    *,
    problem: BindingProblem,
    field: FieldId,
    source: VectorSourceHint,
    cover: CoverPayload,
    top_k: int,
    band: tuple[int, int],
    budget: FilteredAnnVectorSearchBudget,
    region_layout: RegionLayoutModel | None,
) -> tuple[VectorValue, ...]:
    seed = _vector_option_seed(
        problem.search_identity,
        problem.task,
        field,
        stage=SearchSeedStage.VECTOR_ROW_SEEDS,
        components=(cover.count,),
    )
    if filtered_ann_seed_mode() is FilteredAnnSeedMode.RANDOM or region_layout is None:
        # mode 1, or region metadata unavailable: random start vectors
        return _filtered_ann_random_seed_vectors(
            problem=problem,
            field=field,
            source=source,
            budget=budget,
            seed=seed,
        )
    rowids = _filtered_ann_region_seed_rowids(
        region_layout=region_layout,
        cover=cover,
        top_k=top_k,
        band=band,
        seed=seed,
        limit=max(1, budget.seed_limit),
    )
    values = _dedupe_vector_values(tuple(_vector_values_for_rows(problem, field, rowids)))
    if not values:
        return _filtered_ann_random_seed_vectors(
            problem=problem,
            field=field,
            source=source,
            budget=budget,
            seed=seed,
        )
    values = _filtered_ann_perturbed_seed_vectors(
        values,
        problem=problem,
        field=field,
        source=source,
        seed=seed,
    )
    return values[: max(1, budget.seed_limit)]


def _filtered_ann_perturbed_seed_vectors(
    values: tuple[VectorValue, ...],
    *,
    problem: BindingProblem,
    field: FieldId,
    source: VectorSourceHint,
    seed: int,
) -> tuple[VectorValue, ...]:
    """region seed 局部扰动：seed 离开采样数据行，但保持在数据分布的局部邻域内。

    直接用数据行向量作 seed 时，hit target 恰好被满足的行会被原样接受，
    query 贴在该行上；若该行是分布的离群点，下游 ANN 图遍历会系统性
    漏近邻。以采样数据的最近邻距离为局部尺度做一次随机扰动，让 seed
    落在数据行附近的自然位置而不是数据行本身。
    """
    if not values:
        return values
    scale = _filtered_ann_seed_perturb_scale(problem, field, source)
    if scale <= 0.0:
        return values
    rng = np.random.default_rng(seed ^ 0x3C3C3C3C)
    perturbed: list[VectorValue] = []
    for value in values:
        vec = np.asarray(value.components, dtype="float32")
        noise = rng.normal(size=vec.shape)
        noise_norm = float(np.linalg.norm(noise))
        if noise_norm <= 1e-12:
            perturbed.append(value)
            continue
        magnitude = scale * (0.5 + rng.random())
        proposal = vec + (magnitude / noise_norm) * noise
        proposal = _metric_query_vector(proposal, metric=source.metric)
        perturbed.append(VectorValue(components=tuple(float(c) for c in proposal.tolist())))
    return _dedupe_vector_values(tuple(perturbed))


def _filtered_ann_seed_perturb_scale(
    problem: BindingProblem,
    field: FieldId,
    source: VectorSourceHint,
) -> float:
    """用采样行的最近邻距离 median 估计数据的局部邻域尺度。"""
    if source.row_count <= 0:
        return 0.0
    sample = _vector_rows_for_problem(
        problem, field, _source_sample_rowids(source.row_count, limit=256)
    )
    if sample.shape[0] < 2:
        return 0.0
    norms_sq = np.sum(sample * sample, axis=1)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (sample @ sample.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    np.fill_diagonal(dist_sq, np.inf)
    nearest = np.sqrt(np.min(dist_sq, axis=1))
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.0
    return float(np.median(nearest))


def _filtered_ann_random_seed_vectors(
    *,
    problem: BindingProblem,
    field: FieldId,
    source: VectorSourceHint,
    budget: FilteredAnnVectorSearchBudget,
    seed: int,
) -> tuple[VectorValue, ...]:
    """mode 1: random start vectors, sampled outside the vector dataset.

    Direction is uniform on the sphere and magnitude matches the dataset's
    average vector norm, so the candidates share the dataset's scale while
    being unrestricted to its rows.
    """
    if source.row_count <= 0 or int(source.dimension) <= 0:
        return ()
    sample = _vector_rows_for_problem(
        problem, field, _source_sample_rowids(source.row_count, limit=256)
    )
    if sample.shape[0] == 0:
        return ()
    mean_norm = float(np.mean(np.linalg.norm(sample, axis=1)))
    if not math.isfinite(mean_norm) or mean_norm <= 0.0:
        mean_norm = 1.0
    rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    result: list[VectorValue] = []
    for _index in range(1):
        direction = rng.normal(size=int(source.dimension))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        vector = direction * (mean_norm / norm)
        vector = _metric_query_vector(vector, metric=source.metric)
        result.append(VectorValue(components=tuple(float(component) for component in vector.tolist())))
    return _dedupe_vector_values(tuple(result))


def _vector_source_hint(
    problem: BindingProblem,
    field: FieldId,
) -> VectorSourceHint | None:
    summary = observe_search_hint(
        problem,
        ObservationRequest(kind=ObservationRequestKind.VECTOR_SOURCE, field=field),
    )
    return summary.source if summary is not None else None


def _vector_rows_for_problem(problem: BindingProblem, field: FieldId, rowids: tuple[int, ...]) -> np.ndarray:
    if problem.resources is None:
        return np.empty((0, 0), dtype="float32")
    if not rowids:
        source = _vector_source_hint(problem, field)
        dimension = int(source.dimension) if source is not None else 0
        return np.empty((0, dimension), dtype="float32")
    return problem.resources.vector_oracle.vector_rows(
        dataset=problem.resources.dataset,
        field_id=field,
        rowids=rowids,
    )


def _filtered_ann_topk_support(
    problem: BindingProblem,
    field: FieldId,
    source: VectorSourceHint,
    topk_rowids: tuple[int, ...],
) -> FilteredAnnTopKSupport:
    rowids = tuple(int(rowid) for rowid in topk_rowids if 0 <= int(rowid) < int(source.row_count))
    vectors = _vector_rows_for_problem(problem, field, rowids)
    if vectors.shape[0] != len(rowids):
        return FilteredAnnTopKSupport(rowids=(), vectors=np.empty((0, int(source.dimension)), dtype="float32"))
    return FilteredAnnTopKSupport(rowids=rowids, vectors=vectors)


def _filtered_ann_vector_search_budget(problem: BindingProblem) -> FilteredAnnVectorSearchBudget:
    max_attempts = max(int(problem.task.budget.max_parameter_attempts), 1)
    if filtered_ann_seed_mode() is FilteredAnnSeedMode.RANDOM:
        # random 消融（w/o 向量搜索）：去掉内部 frontier 次数限制（128+128=256），
        # 退火链一直跑，只受外层 120s time budget 控制（run_suite 超时 SIGTERM
        # 优雅退出并写 best_so_far），保证失败也真实耗尽时间预算而非提前放弃。
        huge = 1_000_000_000
        return FilteredAnnVectorSearchBudget(
            seed_limit=huge,
            initial_frontier_limit=huge,
            total_hint_limit=huge,
        )
    initial_limit = min(FILTERED_ANN_INITIAL_FRONTIER_LIMIT, max_attempts)
    local_limit = min(FILTERED_ANN_LOCAL_SEARCH_HINT_LIMIT, max_attempts)
    return FilteredAnnVectorSearchBudget(
        seed_limit=min(FILTERED_ANN_SEED_LIMIT, max_attempts),
        initial_frontier_limit=initial_limit,
        total_hint_limit=initial_limit + local_limit,
    )


def _observe_filtered_ann_frontier_hint(
    problem: BindingProblem,
    algorithm: FilteredAnnVectorSearchAlgorithm,
    frontier: FilteredAnnVectorFrontier,
    vector: VectorValue,
    hint: FilteredAnnHitHint,
) -> None:
    frontier.hint_observations += 1
    frontier.last_hints[vector.components] = hint
    support = _filtered_ann_topk_support(
        problem,
        frontier.field,
        frontier.source,
        hint.global_topk.rowids,
    )
    algorithm.extend_frontier(frontier, vector, hint, support)


def _observe_filtered_ann_vector_feedback(
    problem: BindingProblem,
    state: ParameterSearchState,
    feedback: ParameterFeedback,
    algorithm: FilteredAnnVectorSearchAlgorithm,
) -> None:
    pending = _filtered_ann_pending_proposals(state).pop(feedback.binding_signature, None)
    if (
        pending is None
        or feedback.accepted
        or not isinstance(feedback.evidence, FilteredAnnSearchEvidence)
    ):
        return
    frontier = _filtered_ann_frontiers(state).get(pending.frontier_key)
    if frontier is None or frontier.exhausted:
        return
    previous_hint = frontier.last_hints.get(pending.vector_components)
    if previous_hint is None:
        return
    hit_count = feedback.evidence.hit_count
    canonical_hint = FilteredAnnHitHint(
        global_topk=previous_hint.global_topk,
        selection=previous_hint.selection,
        hit_count=hit_count,
        loss=gls_loss(
            hit_count=hit_count,
            band=frontier.band,
            total_count=frontier.cover.total_count,
            cover_count=frontier.cover.count,
            k=frontier.top_k,
        ),
    )
    current = VectorValue(components=pending.vector_components)
    support = _filtered_ann_topk_support(
        problem,
        frontier.field,
        frontier.source,
        canonical_hint.global_topk.rowids,
    )
    algorithm.extend_frontier(frontier, current, canonical_hint, support)


def _extend_filtered_ann_frontier_anneal(
    frontier: FilteredAnnVectorFrontier,
    current: VectorValue,
    hint: FilteredAnnHitHint,
    support: FilteredAnnTopKSupport | None = None,
) -> None:
    state = frontier.anneal
    if state is None:
        raise ContractViolation("filtered ANN anneal frontier requires anneal state")
    if current.components in state.processed:
        # canonical re-evaluation of an already-processed vector: no transition,
        # but a false zero-loss observation must be repaired so the frontier
        # does not silently exhaust with a wrong optimum.
        if (
            hint.loss > 0.0
            and state.best_loss == 0.0
            and state.best is not None
            and state.best.components == current.components
        ):
            state.best, state.best_loss = None, float("inf")
            if state.chain is not None and state.chain.components == current.components:
                state.chain_loss = hint.loss
                state.chain_hit_count = int(hint.hit_count)
            for index, (components, _loss, _hit_count, vector) in enumerate(state.seed_losses):
                if components == current.components:
                    state.seed_losses[index] = (components, hint.loss, int(hint.hit_count), vector)
                    break
            if state.phase == "anneal":
                _propose_anneal_step(frontier, support, state.chain_hit_count)
        return
    state.processed.add(current.components)
    if state.phase == "seed":
        state.seed_losses.append((current.components, hint.loss, int(hint.hit_count), current))
        if frontier.queue:
            return  # more seeds pending evaluation
        # seed pool exhausted: q0 = argmin GLS error over the pool
        if state.seed_losses:
            _components, best_loss, best_hit_count, best_vector = min(
                state.seed_losses, key=lambda item: item[1]
            )
            state.chain, state.chain_loss = best_vector, best_loss
            state.chain_hit_count = best_hit_count
            state.best, state.best_loss = best_vector, best_loss
        state.phase = "anneal"
        if state.best_loss == 0.0:
            return
        _propose_anneal_step(frontier, support, state.chain_hit_count)
        return
    # anneal phase: current is a perturbed candidate around the chain state
    delta = hint.loss - state.chain_loss
    temperature = state.initial_temperature / max(1, state.iteration + 1)
    if delta <= 0.0 or frontier.rng.random() < math.exp(-delta / temperature):
        state.chain, state.chain_loss = current, hint.loss
        state.chain_hit_count = int(hint.hit_count)
        if delta < 0.0:
            frontier.step = max(frontier.step * 0.85, 1e-6)  # accepted an improving move: shrink the perturbation
    if hint.loss < state.best_loss:
        state.best, state.best_loss = current, hint.loss
    if state.best_loss == 0.0:
        return
    state.iteration += 1
    _propose_anneal_step(frontier, support, state.chain_hit_count)


def _propose_anneal_step(
    frontier: FilteredAnnVectorFrontier,
    support: FilteredAnnTopKSupport | None = None,
    hit_count: int = 0,
) -> None:
    """Paper §V-C (Alg. 3): LOCALPERTURB — 纯随机扰动，无原型候选。

    support/hit_count 保留在签名中是为了 extend_frontier 协议的向后兼容；
    退火本身只依赖链状态的随机扰动（论文 LOCALPERTURB）。
    """
    state = frontier.anneal
    if state is None or state.chain is None:
        return
    chain = np.asarray(state.chain.components, dtype="float32")
    direction = frontier.rng.normal(size=chain.shape)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return
    scale = frontier.step * (0.5 + frontier.rng.random())
    proposal = chain + (scale / norm) * direction
    proposal = _metric_query_vector(proposal, metric=frontier.source.metric)
    components = tuple(float(component) for component in proposal.tolist())
    if components in frontier.seen:
        return
    frontier.seen.add(components)
    frontier.queue.append(
        (VectorValue(components=components), HintCostStage.FILTERED_ANN_LOCAL_SEARCH)
    )


def _filtered_ann_hit_hint(
    *,
    problem: BindingProblem,
    field: FieldId,
    vector: VectorValue,
    cover: CoverPayload,
    top_k: int,
    band: tuple[int, int],
    stage: HintCostStage | None = None,
) -> FilteredAnnHitHint | None:
    summary = observe_search_hint(
        problem,
        ObservationRequest(
            kind=ObservationRequestKind.FILTERED_ANN_HIT,
            field=field,
            query_vector=vector.components,
            k=top_k,
            cover_payload=cover,
            hit_band=band,
        ),
        stage=stage,
    )
    return summary.filtered_ann_hit if summary is not None else None


def _filtered_ann_initial_step(*, problem: BindingProblem, field: FieldId, source: VectorSourceHint) -> float:
    sample_rowids = _source_sample_rowids(source.row_count, limit=256)
    if not sample_rowids:
        return 1e-3
    sample = _vector_rows_for_problem(problem, field, sample_rowids)
    if sample.shape[0] <= 0:
        return 1e-3
    norm_mean = float(np.mean(np.linalg.norm(sample, axis=1)))
    return max(norm_mean * 0.05, 1e-3)


def _metric_query_vector(vector: np.ndarray, *, metric: VectorMetric) -> np.ndarray:
    result = metric_query(np.asarray(vector, dtype="float32"), metric=metric)
    return np.asarray(result, dtype="float32")


def _owner_region_layout(
    resources: DatasetResources,
    owner: object,
    *,
    vector_field_name: str,
) -> RegionLayoutModel | None:
    metadata = deterministic_owner_metadata(resources.dataset, owner)
    if metadata is None:
        return None
    layout = getattr(metadata, "region_layout", None)
    if layout is None:
        return None
    if not isinstance(layout, RegionLayoutModel):
        return None
    if layout.vector_field_name != vector_field_name:
        raise ContractViolation("filtered ANN region layout does not match vector field")
    return layout


def _filtered_ann_region_seed_rowids(
    *,
    region_layout: RegionLayoutModel,
    cover: CoverPayload,
    top_k: int,
    band: tuple[int, int],
    seed: int,
    limit: int,
) -> tuple[int, ...]:
    """Paper §V-C: 从 estimated GLS 最优的 regions 采样数据行作为 seed pool。

    正文 "sampled directly from the vectors within the selected regions"：
    先用 estimated GLS error 排序 region，再从最优 region（+ 邻居多样性）
    内采样数据行（representative + 随机），随后 exact top-K 评估 → argmin。
    """
    if limit <= 0 or not region_layout.regions:
        return ()
    if cover.count <= 0 or cover.total_count <= 0:
        return ()
    # Paper §V-C: rank regions by estimated GLS error.
    # σ_g = |F|/N; target σ_l from the hit band; b_i = (h_i - σ_g)/(h_i + σ_g).
    global_selectivity = cover.count / cover.total_count
    target_hit = (band[0] + band[1]) / 2.0
    target_local_selectivity = target_hit / max(top_k, 1)
    ratio = target_local_selectivity / global_selectivity
    target_gls = (ratio - 1.0) / (ratio + 1.0)
    scored_regions: list[tuple[float, int, RegionRange]] = []
    for region in region_layout.regions:
        size = int(region.public_end) - int(region.public_start) + 1
        if size <= 0:
            continue
        h_i = _cover_region_overlap_count(
            cover,
            start=int(region.public_start),
            end=int(region.public_end),
        ) / size
        b_i = (h_i - global_selectivity) / (h_i + global_selectivity)
        scored_regions.append((abs(b_i - target_gls), int(region.region_id), region))
    if not scored_regions:
        return ()
    scored_regions.sort(key=lambda item: (item[0], item[1]))
    region_by_id = {region.region_id: region for _score, _region_id, region in scored_regions}
    selected_ids: list[int] = []
    for _score, region_id, _region in scored_regions:
        selected_ids.append(region_id)
        neighbors = region_layout.region_neighbors[region_id] if region_id < len(region_layout.region_neighbors) else ()
        for neighbor_id in neighbors:
            if neighbor_id in region_by_id:
                selected_ids.append(int(neighbor_id))
        if len(dict.fromkeys(selected_ids)) >= min(len(region_by_id), max(1, limit // 2)):
            break
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    rowids: list[int] = []
    seen_regions: set[int] = set()
    for region_id in selected_ids:
        if region_id in seen_regions:
            continue
        selected_region = region_by_id.get(region_id)
        if selected_region is None:
            continue
        seen_regions.add(region_id)
        representative = int(selected_region.representative_public_ordinal)
        if selected_region.public_start <= representative <= selected_region.public_end:
            rowids.append(representative)
        remaining = limit - len(rowids)
        if remaining <= 0:
            break
        region_start = int(selected_region.public_start)
        region_size = int(selected_region.public_end) - region_start + 1
        sample_size = min(max(1, limit // max(1, len(seen_regions) * 2)), remaining, region_size)
        if sample_size > 0:
            selected = rng.choice(region_size, size=sample_size, replace=False)
            selected = selected + region_start
            rowids.extend(int(rowid) for rowid in selected.tolist())
        if len(rowids) >= limit:
            break
    return tuple(dict.fromkeys(rowids[:limit]))


def _vector_values_for_rows(problem: BindingProblem, field: FieldId, rowids: tuple[int, ...]) -> list[VectorValue]:
    vectors = _vector_rows_for_problem(problem, field, rowids)
    return [
        VectorValue(components=tuple(float(component) for component in row.tolist()))
        for row in vectors
    ]


def _cover_region_overlap_count(cover: CoverPayload, *, start: int, end: int) -> int:
    if start > end:
        return 0
    return intervals_row_count(intersect_intervals(cover.intervals, ((int(start), int(end)),)))


def _source_sample_rowids(row_count: int, *, limit: int) -> tuple[int, ...]:
    total = int(row_count)
    if total <= 0 or limit <= 0:
        return ()
    sample_size = min(total, int(limit))
    if sample_size == total:
        return tuple(range(total))
    step = total / sample_size
    return tuple(dict.fromkeys(min(total - 1, int(index * step)) for index in range(sample_size)))


def _selectivity_band_target(problem: BindingProblem) -> tuple[float, float] | None:
    for objective in problem.task.objectives:
        if (
            objective.kind is ObjectiveKind.SELECTIVITY_BAND
            and objective.metric is ObjectiveMetric.SELECTIVITY
            and objective.minimum is not None
            and objective.maximum is not None
        ):
            return float(objective.minimum), float(objective.maximum)
    return None


def _top_k(problem: BindingProblem) -> int:
    for objective in problem.task.objectives:
        if objective.kind is ObjectiveKind.TOPK_AVAILABLE and objective.k is not None:
            return int(objective.k)
    return 1


def _dedupe_vector_values(options: tuple[VectorValue, ...]) -> tuple[VectorValue, ...]:
    seen: set[tuple[float, ...]] = set()
    result: list[VectorValue] = []
    for option in options:
        if not option.components or option.components in seen:
            continue
        seen.add(option.components)
        result.append(option)
    return tuple(result)


def _vector_option_seed(
    search_identity: SearchIdentity,
    task: QueryTask,
    field: FieldId,
    *,
    stage: SearchSeedStage,
    components: tuple[str | int | float, ...] = (),
) -> int:
    return derive_search_seed(
        search_identity,
        stage,
        task.source_intent.public_family.value,
        task.source_intent.public_form or "",
        task.source_intent.source_name or "",
        field.owner.name,
        field.name,
        task.result_semantics.kind.value,
        task.result_semantics.k or 0,
        *components,
    )


__all__ = ["VectorParameterSearchStrategy"]
