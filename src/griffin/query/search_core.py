"""Core parameter search task contracts."""

from __future__ import annotations

import enum
import itertools
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, Union

import msgspec

from griffin.dataset.exact_cover import (
    deterministic_owner_metadata,
    field_model_for_name,
    intervals_row_count,
)
from griffin.infra.errors import ContractViolation
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.domain import (
    DomainBuildResult,
    enumerated_parameter_values,
    hole_context_for_template,
    parameter_space_is_empty,
)
from griffin.query.evaluate import (
    ExactSelection,
    SearchEvidence,
    SearchFeedback,
    SelectionCoverObservation,
)
from griffin.query.model import (
    BindingSignature,
    BindingFingerprint,
    EnumeratedParameterSpace,
    HoleId,
    HoleValueKind,
    ParameterBinding,
    ParameterSpace,
    ParameterValue,
    QueryTask,
    QueryTemplate,
    SearchSeedStage,
    SearchIdentity,
    TemplateNode,
    binding_signature,
    derive_search_seed,
    iter_template_nodes,
    stable_signature,
)
from griffin.query.observation import (
    CoverPayload,
    FilteredAnnHitHint,
    HintCostStage,
    ObservationScope,
    ObservationRequest,
    ObservationSummary,
    ObservationRequestKind,
    ObservationSession,
    PredicateCoverHint,
)
from griffin.query.predicate_cover import (
    PredicateBindingFragment,
    PredicateCoverBestSoFar,
    PredicateCoverCandidate,
    PredicateCoverConsumerKind,
    PredicateCoverObservationValue,
    PredicateCoverPlanKey,
    PredicateCoverRejection,
    PredicateCoverTargetMetric,
    predicate_target_distance,
)
from griffin.query.predicate_evolution import (
    PredicateAxisValueProfile,
    PredicateEvolutionCandidate,
    PredicateEvolutionFeedback,
    PredicateEvolutionPrior,
    PredicateEvolutionProblem,
    predicate_evolution_strategy,
    predicate_evolution_values,
    predicate_marginal_band,
)
from griffin.query.search_state import (
    ParameterSearchState,
    _PredicateCoverSearchRunnerState,
)
from griffin.query.task import ObjectiveKind
from griffin.query.task import ObjectiveMetric
from griffin.query.task import PredicateEvolutionStrategyKind


class CandidateProposalKind(enum.Enum):
    CANDIDATE = "candidate"
    NO_PROPOSAL = "no_proposal"


class ParameterSolverKind(enum.Enum):
    VECTOR = "vector"
    PREDICATE_COVER = "predicate_cover"
    FIXED = "fixed"


DEFAULT_PREDICATE_COVER_OBSERVATION_LIMIT = 20000
EXPERIMENT_VARIANT_ENV = "MMBENCHGEN_QUERY_EXPERIMENT_VARIANT"


def _query_experiment_variant() -> str:
    return os.environ.get(EXPERIMENT_VARIANT_ENV, "griffin").strip().lower()


def _is_baseline_experiment_variant() -> bool:
    return _query_experiment_variant() == "baseline"


def observe_search_hint(
    problem: "BindingProblem",
    request: ObservationRequest,
    *,
    stage: HintCostStage | None = None,
) -> ObservationSummary | None:
    session = problem.observation_session
    if session is None:
        return None
    return session.observe_hint(request, stage=stage, scope=problem.observation_scope)


def _record_profile_event(problem: "BindingProblem", name: str, *, count: int = 1) -> None:
    session = problem.observation_session
    if session is None:
        return
    session.record_profile_event(name, count=count)


class _PredicateCoverAttempt(msgspec.Struct, frozen=True, omit_defaults=True):
    fragment: PredicateBindingFragment
    observation: PredicateCoverObservationValue
    hint: PredicateCoverHint
    cover: CoverPayload
    target_metric: PredicateCoverTargetMetric
    observed_value: float
    objective_distance: float
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)


_PredicateCoverOutcome: TypeAlias = Union[PredicateCoverCandidate, PredicateCoverRejection]


@dataclass(frozen=True)
class PredicateCoverConsumerSpec:
    kind: PredicateCoverConsumerKind


@dataclass(frozen=True)
class PredicateCoverSearchPlan:
    consumer: PredicateCoverConsumerSpec
    answer_field: FieldId
    evolution: PredicateEvolutionProblem


@dataclass(frozen=True)
class PredicateCoverSearchBudget:
    max_cover_observations_per_solver_step: int
    max_rejections_per_search: int
    max_usable_candidates: int


@dataclass(frozen=True)
class PredicateCoverSearchResult:
    candidate: PredicateCoverCandidate | None
    rejection: PredicateCoverRejection | None

    @property
    def usable(self) -> bool:
        return self.candidate is not None


class PredicateCoverSearchRunner:
    def search(
        self,
        problem: "BindingProblem",
        state: "ParameterSearchState",
        plan: PredicateCoverSearchPlan,
    ) -> Iterator[PredicateCoverSearchResult]:
        evolution = plan.evolution
        algorithm = predicate_evolution_strategy(evolution.strategy)
        _record_profile_event(
            problem,
            f"query.predicate.evolution.strategy.{algorithm.kind.value}",
        )
        plan_key = predicate_cover_plan_key(plan)
        runner_state = _predicate_cover_search_runner_state(state, plan_key)
        for candidate in runner_state.usable_candidates:
            _record_profile_event(problem, "query.reuse.cover.search_candidate")
            yield PredicateCoverSearchResult(candidate=candidate, rejection=None)
        if runner_state.exhausted or runner_state.budget_exhausted:
            return
        budget = predicate_cover_search_budget(problem)
        observations_this_call = 0
        seed = _predicate_cover_search_seed(problem, evolution, plan_key)
        while observations_this_call < budget.max_cover_observations_per_solver_step:
            if runner_state.rejection_count >= budget.max_rejections_per_search:
                _stop_predicate_cover_for_budget(
                    runner_state,
                    reason="predicate cover budget exhausted: max rejections per search reached",
                )
                return
            if len(runner_state.usable_candidates) >= budget.max_usable_candidates:
                _stop_predicate_cover_for_budget(
                    runner_state,
                    reason="predicate cover budget exhausted: max usable candidates reached",
                )
                return
            generation_before = runner_state.evolution_state.generation
            search_candidate = algorithm.propose(
                evolution,
                runner_state.evolution_state,
                seed=seed,
                candidate_limit=_predicate_candidate_frontier_limit(
                    problem,
                    budget,
                    frontier_floor=algorithm.candidate_frontier_floor(
                        len(evolution.predicate_domains)
                    ),
                ),
            )
            generation_count = (
                runner_state.evolution_state.generation - generation_before
            )
            _record_profile_event(
                problem,
                "query.predicate.evolution.generation",
                count=generation_count,
            )
            if search_candidate is None:
                runner_state.exhausted = True
                return
            _record_profile_event(
                problem,
                "query.predicate.evolution.proposal",
            )
            runner_state.evolution_state.observed_candidates.add(search_candidate)
            values = predicate_evolution_values(evolution, search_candidate)
            fingerprint = predicate_binding_fragment(
                evolution.template,
                values,
            ).binding_fingerprint
            if fingerprint in runner_state.observed_fingerprints:
                _record_profile_event(problem, "query.avoid.cover.duplicate_binding")
                continue
            observations_this_call += 1
            result = predicate_cover_search_result(
                problem=problem,
                state=state,
                evolution=evolution,
                answer_field=plan.answer_field,
                values=values,
            )
            runner_state.observed_fingerprints.add(fingerprint)
            feedback = _predicate_cover_search_feedback(
                evolution,
                search_candidate,
                fingerprint,
                result,
            )
            algorithm.observe(
                runner_state.evolution_state,
                evolution,
                feedback,
            )
            if result.rejection is not None:
                _record_profile_event(
                    problem,
                    "query.predicate.evolution.rejection",
                )
                runner_state.rejection_count += 1
                runner_state.rejection_reason = result.rejection.reason
                continue
            if result.candidate is None:
                raise ContractViolation("predicate cover search result requires candidate or rejection")
            _record_profile_event(
                problem,
                "query.predicate.evolution.usable_candidate",
            )
            runner_state.usable_candidates.append(result.candidate)
            yield result
        reason = "predicate cover budget exhausted: max cover observations per solver step reached"
        if runner_state.rejection_count >= budget.max_rejections_per_search:
            reason = "predicate cover budget exhausted: max rejections per search reached"
        _stop_predicate_cover_for_budget(runner_state, reason=reason)


def profiled_predicate_cover_search(
    problem: "BindingProblem",
    state: "ParameterSearchState",
    plan: PredicateCoverSearchPlan,
) -> Iterator[PredicateCoverSearchResult]:
    """Advance predicate-cover search under the shared selectivity span."""
    results = iter(PredicateCoverSearchRunner().search(problem, state, plan))
    while True:
        session = problem.observation_session
        measure = (
            session.measure_profile("query.search.selectivity")
            if (session is not None and plan.evolution.target_metric is PredicateCoverTargetMetric.SELECTIVITY)
            else None
        )
        if measure is None:
            result = next(results, None)
        else:
            with measure:
                result = next(results, None)
        if result is None:
            return
        yield result


def predicate_cover_search_budget(
    problem: "BindingProblem",
) -> PredicateCoverSearchBudget:
    max_attempts = max(int(problem.task.budget.max_parameter_attempts), 1)
    default_cover_observations = max(DEFAULT_PREDICATE_COVER_OBSERVATION_LIMIT, max_attempts * 8)
    max_observations = max(
        int(problem.task.budget.max_cover_observations_per_solver_step or default_cover_observations),
        1,
    )
    max_rejections = max(
        int(problem.task.budget.max_cover_rejections_per_search or default_cover_observations),
        1,
    )
    max_usable = max(
        int(problem.task.budget.max_cover_candidates_per_search or default_cover_observations),
        1,
    )
    if _has_filtered_ann_objective(problem.task):
        max_usable = 1
    return PredicateCoverSearchBudget(
        max_cover_observations_per_solver_step=max_observations,
        max_rejections_per_search=max_rejections,
        max_usable_candidates=max_usable,
    )


def _predicate_candidate_frontier_limit(
    problem: "BindingProblem",
    budget: PredicateCoverSearchBudget,
    *,
    frontier_floor: int,
) -> int:
    ticket_limit = max(int(problem.budget), 1) * 8
    return min(
        budget.max_cover_observations_per_solver_step,
        max(frontier_floor, ticket_limit),
    )


def _has_filtered_ann_objective(task: QueryTask) -> bool:
    return any(objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND for objective in task.objectives)


def predicate_cover_search_prior(
    problem: "BindingProblem",
    *,
    predicate_domains: tuple[ParameterSpace, ...],
    target_metric: PredicateCoverTargetMetric,
    target_minimum: float,
    target_maximum: float,
) -> PredicateEvolutionPrior:
    if target_metric is PredicateCoverTargetMetric.SELECTIVITY:
        marginal_minimum, marginal_center, marginal_maximum = predicate_marginal_band(
            problem.template.root,
            predicate_domains,
            target_minimum=target_minimum,
            target_maximum=target_maximum,
        )
    else:
        marginal_minimum, marginal_center, marginal_maximum = 0.0, 0.5, 1.0
    profiles: tuple[tuple[PredicateAxisValueProfile, ...], ...] = ()
    if (
        target_metric is PredicateCoverTargetMetric.SELECTIVITY
        and problem.resources is not None
        and predicate_domains
        and not _is_baseline_experiment_variant()
    ):
        profiles = _predicate_axis_profiles(
            problem,
            predicate_domains=predicate_domains,
            marginal_center=marginal_center,
        )
    return PredicateEvolutionPrior(
        marginal_minimum=marginal_minimum,
        marginal_center=marginal_center,
        marginal_maximum=marginal_maximum,
        axis_profiles=profiles,
    )


def _predicate_axis_profiles(
    problem: "BindingProblem",
    *,
    predicate_domains: tuple[ParameterSpace, ...],
    marginal_center: float,
) -> tuple[tuple[PredicateAxisValueProfile, ...], ...]:
    resources = problem.resources
    if resources is None:
        return tuple(() for _domain in predicate_domains)
    profiles: list[tuple[PredicateAxisValueProfile, ...]] = []
    for space in predicate_domains:
        if not isinstance(space, EnumeratedParameterSpace):
            profiles.append(())
            continue
        context = hole_context_for_template(problem.template.root, space.hole_id)
        if context.field is None or not space.values:
            profiles.append(())
            continue
        metadata = deterministic_owner_metadata(resources.dataset, context.field.owner)
        if metadata is None or metadata.row_count <= 0:
            profiles.append(())
            continue
        try:
            model = field_model_for_name(metadata, context.field.name)
        except (AttributeError, ContractViolation):
            profiles.append(())
            continue
        scored: list[tuple[float, int, ParameterValue, float]] = []
        for index, value in enumerate(space.values):
            try:
                local_cover = resources.predicate_access.local_cover(model, value)
            except ContractViolation:
                continue
            selectivity = intervals_row_count(local_cover) / float(metadata.row_count)
            scored.append((abs(selectivity - marginal_center), index, value, selectivity))
        scored.sort(key=lambda item: (item[0], item[1]))
        profiles.append(
            tuple(
                PredicateAxisValueProfile(value=value, selectivity=selectivity)
                for _distance, _index, value, selectivity in scored
            )
        )
    return tuple(profiles)


def predicate_cover_search_result(
    *,
    problem: "BindingProblem",
    state: "ParameterSearchState",
    evolution: PredicateEvolutionProblem,
    answer_field: FieldId,
    values: Mapping[HoleId, ParameterValue],
) -> PredicateCoverSearchResult:
    _record_profile_event(
        problem,
        "query.predicate.evolution.exact_observation",
    )
    summary = observe_search_hint(
        problem,
        ObservationRequest(
            kind=ObservationRequestKind.PREDICATE_COVER,
            field=answer_field,
            template_signature=evolution.template.structural_signature,
            cover_node=evolution.cover_root,
            cover_values=values,
        ),
    )
    if summary is None or summary.predicate_cover is None:
        raise ContractViolation("predicate cover search requires ObservationSession predicate cover observation")
    cover = summary.predicate_cover
    cover_payload = cover.payload
    observed_value = _predicate_cover_observed_value(cover, evolution.target_metric)
    distance = predicate_target_distance(
        observed_value,
        evolution.target_minimum,
        evolution.target_maximum,
    )
    fragment = predicate_binding_fragment(evolution.template, values)
    observation = SelectionCoverObservation(
        selection=ExactSelection(
            node=_template_node_by_id(evolution.template, cover.node_id),
            scope_id=cover.scope_id,
            answer_scope=cover.answer_scope,
            cover=cover.cover,
        ),
    )
    attempt = _PredicateCoverAttempt(
        fragment=fragment,
        observation=observation,
        hint=cover,
        cover=cover_payload,
        target_metric=evolution.target_metric,
        observed_value=observed_value,
        objective_distance=distance,
    )
    outcome = _predicate_cover_outcome(
        attempt,
        target_minimum=evolution.target_minimum,
        target_maximum=evolution.target_maximum,
    )
    if isinstance(outcome, PredicateCoverCandidate):
        return PredicateCoverSearchResult(
            candidate=outcome,
            rejection=None,
        )
    return PredicateCoverSearchResult(
        candidate=None,
        rejection=outcome,
    )


def _predicate_cover_search_runner_state(
    state: "ParameterSearchState",
    plan_key: PredicateCoverPlanKey,
) -> _PredicateCoverSearchRunnerState:
    cached = state.predicate_cover.by_plan.get(plan_key)
    if cached is not None:
        return cached
    result = _PredicateCoverSearchRunnerState()
    state.predicate_cover.by_plan[plan_key] = result
    return result


def _stop_predicate_cover_for_budget(
    search_state: _PredicateCoverSearchRunnerState,
    *,
    reason: str,
) -> None:
    search_state.budget_exhausted = True
    search_state.budget_reason = reason


def predicate_cover_search_diagnostic_reason(
    state: "ParameterSearchState",
    plan: PredicateCoverSearchPlan,
) -> str | None:
    cached = state.predicate_cover.by_plan.get(predicate_cover_plan_key(plan))
    if cached is None:
        return None
    return cached.budget_reason or cached.rejection_reason


def parameter_search_best_so_far(
    state: "ParameterSearchState",
) -> PredicateCoverBestSoFar | None:
    """搜索过程中实测选择率最接近 band 的历史最优候选（附 GLS 维度 best_so_far）。"""
    best: PredicateCoverBestSoFar | None = None
    gls_hit = state.vector.filtered_ann.best_so_far_hit_count
    for runner_state in state.predicate_cover.by_plan.values():
        for score in runner_state.evolution_state.scores:
            if best is None or score.objective_distance < best.objective_distance:
                best = PredicateCoverBestSoFar(
                    observed_value=float(score.observed_value),
                    objective_distance=float(score.objective_distance),
                    gls_hit_count=gls_hit,
                )
    if best is None and gls_hit is not None:
        best = PredicateCoverBestSoFar(
            observed_value=float("nan"),
            objective_distance=float("inf"),
            gls_hit_count=gls_hit,
        )
    return best


def _predicate_cover_search_feedback(
    evolution: PredicateEvolutionProblem,
    candidate: PredicateEvolutionCandidate,
    fingerprint: BindingFingerprint,
    result: PredicateCoverSearchResult,
) -> PredicateEvolutionFeedback:
    if result.candidate is not None:
        return PredicateEvolutionFeedback(
            candidate=candidate,
            binding_fingerprint=fingerprint,
            observed_value=result.candidate.observed_value,
            objective_distance=result.candidate.objective_distance,
            usable=True,
        )
    if result.rejection is None:
        raise ContractViolation("predicate cover search result requires candidate or rejection")
    return PredicateEvolutionFeedback(
        candidate=candidate,
        binding_fingerprint=fingerprint,
        observed_value=result.rejection.observed_value,
        objective_distance=predicate_target_distance(
            result.rejection.observed_value,
            evolution.target_minimum,
            evolution.target_maximum,
        ),
        usable=False,
    )


def predicate_cover_plan_key(plan: PredicateCoverSearchPlan) -> PredicateCoverPlanKey:
    evolution = plan.evolution
    return PredicateCoverPlanKey(
        consumer_kind=plan.consumer.kind,
        evolution_strategy=evolution.strategy,
        template_signature=evolution.template.structural_signature,
        cover_node_id=evolution.cover_root.node_id,
        answer_field=plan.answer_field,
        target_metric=evolution.target_metric,
        target_minimum=float(evolution.target_minimum),
        target_maximum=float(evolution.target_maximum),
    )


def _predicate_cover_search_seed(
    problem: "BindingProblem",
    task: PredicateEvolutionProblem,
    plan_key: PredicateCoverPlanKey,
) -> int:
    return derive_search_seed(
        problem.search_identity,
        SearchSeedStage.PREDICATE_COVER,
        task.template.structural_signature,
        plan_key.consumer_kind.value,
        plan_key.cover_node_id,
        str(plan_key.answer_field),
        task.target_metric.value,
        task.target_minimum,
        task.target_maximum,
    )


def predicate_binding_fragment(
    template: QueryTemplate,
    values: Mapping[HoleId, ParameterValue],
) -> PredicateBindingFragment:
    fragment_values = dict(values)
    return PredicateBindingFragment(
        template_signature=template.structural_signature,
        values=fragment_values,
        binding_fingerprint=stable_signature(
            "predicate_fragment",
            {
                "template": template.structural_signature,
                "values": fragment_values,
            },
        ),
    )


def _predicate_cover_outcome(
    attempt: _PredicateCoverAttempt,
    *,
    target_minimum: float,
    target_maximum: float,
) -> _PredicateCoverOutcome:
    if attempt.objective_distance == 0.0:
        return PredicateCoverCandidate(
            fragment=attempt.fragment,
            observation=attempt.observation,
            hint=attempt.hint,
            cover=attempt.cover,
            target_metric=attempt.target_metric,
            observed_value=attempt.observed_value,
            objective_distance=0.0,
            diagnostics=attempt.diagnostics,
        )
    return PredicateCoverRejection(
        fragment_fingerprint=attempt.fragment.binding_fingerprint,
        cover=attempt.cover,
        target_metric=attempt.target_metric,
        observed_value=float(attempt.observed_value),
        target_minimum=float(target_minimum),
        target_maximum=float(target_maximum),
        reason=f"{attempt.target_metric.value}_out_of_band",
        diagnostics=attempt.diagnostics,
    )


def _predicate_cover_observed_value(
    cover: "PredicateCoverHint",
    metric: PredicateCoverTargetMetric,
) -> float:
    if metric is PredicateCoverTargetMetric.SELECTIVITY:
        return float(cover.selectivity)
    raise ContractViolation(f"unsupported predicate cover target metric {metric.value}")


def _template_node_by_id(template: QueryTemplate, node_id: str) -> TemplateNode:
    for node in iter_template_nodes(template.root):
        if node.node_id == node_id:
            return node
    raise ContractViolation(f"template node {node_id!r} not found")


class BindingProblem(msgspec.Struct, frozen=True, omit_defaults=True):
    task: QueryTask
    search_identity: SearchIdentity
    template: QueryTemplate
    domains: DomainBuildResult
    budget: int
    solver_kind: ParameterSolverKind
    resources: DatasetResources | None = None
    feedback: SearchFeedback | None = None
    observation_session: ObservationSession | None = None
    observation_scope: ObservationScope | None = None


class NoPreparedCandidateObservations(
    msgspec.Struct,
    frozen=True,
    tag="none",
    tag_field="prepared_observation_kind",
):
    pass


class PredicatePreparedCandidateObservations(
    msgspec.Struct,
    frozen=True,
    tag="predicate",
    tag_field="prepared_observation_kind",
):
    covers: tuple[PredicateCoverHint, ...]


class FilteredAnnPreparedCandidateObservations(
    msgspec.Struct,
    frozen=True,
    tag="filtered_ann",
    tag_field="prepared_observation_kind",
):
    covers: tuple[PredicateCoverHint, ...]
    hit: FilteredAnnHitHint


PreparedCandidateObservations = Union[
    NoPreparedCandidateObservations,
    PredicatePreparedCandidateObservations,
    FilteredAnnPreparedCandidateObservations,
]


class CandidateProposal(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: CandidateProposalKind
    binding: ParameterBinding | None = None
    prepared_observations: PreparedCandidateObservations = (
        NoPreparedCandidateObservations()
    )
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)

    @classmethod
    def candidate(
        cls,
        binding: ParameterBinding,
        *,
        prepared_observations: PreparedCandidateObservations | None = None,
    ) -> "CandidateProposal":
        return cls(
            kind=CandidateProposalKind.CANDIDATE,
            binding=binding,
            prepared_observations=(
                prepared_observations or NoPreparedCandidateObservations()
            ),
        )

    @classmethod
    def no_proposal(cls, reason: str) -> "CandidateProposal":
        return cls(
            kind=CandidateProposalKind.NO_PROPOSAL,
            diagnostics={"reason": reason},
        )


class ParameterFeedback(msgspec.Struct, frozen=True, omit_defaults=True):
    binding_signature: BindingSignature
    accepted: bool
    score: float
    evidence: SearchEvidence
    search_feedback: SearchFeedback | None = None


class ParameterSearchStrategy(Protocol):
    def next(self, problem: BindingProblem, state: ParameterSearchState) -> CandidateProposal: ...

    def observe(
        self,
        problem: BindingProblem,
        state: ParameterSearchState,
        feedback: ParameterFeedback,
    ) -> ParameterSearchState: ...


def apply_parameter_feedback(
    state: ParameterSearchState,
    feedback: ParameterFeedback,
) -> ParameterSearchState:
    state.attempted_bindings.add(feedback.binding_signature)
    if not feedback.accepted:
        state.excluded_binding_signatures.add(feedback.binding_signature)
    if feedback.search_feedback is not None:
        state.excluded_binding_signatures.update(feedback.search_feedback.excluded_binding_signatures)
    return state


class PredicateCoverFillStrategy:
    """Predicate-cover-first proposal strategy for non-rank predicate queries."""

    def next(self, problem: BindingProblem, state: ParameterSearchState) -> CandidateProposal:
        plan = predicate_cover_plan(problem)
        if isinstance(plan, str):
            return CandidateProposal.no_proposal(plan)
        for result in profiled_predicate_cover_search(
            problem,
            state,
            plan,
        ):
            if result.candidate is None:
                continue
            values = dict(result.candidate.fragment.values)
            signature = binding_signature(problem.template, values)
            if signature in state.excluded_binding_signatures:
                continue
            if signature in state.attempted_bindings:
                continue
            binding = ParameterBinding(
                template_signature=problem.template.structural_signature,
                values=values,
                complete_values=True,
                binding_signature=signature,
            )
            predicate_cover_hints = {result.candidate.hint.node_id: result.candidate.hint}
            return CandidateProposal.candidate(
                binding,
                prepared_observations=PredicatePreparedCandidateObservations(
                    covers=tuple(predicate_cover_hints.values()),
                ),
            )
        reason = predicate_cover_search_diagnostic_reason(state, plan)
        if reason:
            return CandidateProposal.no_proposal(reason)
        return CandidateProposal.no_proposal("all predicate cover candidates attempted or infeasible")

    def observe(
        self,
        problem: BindingProblem,
        state: ParameterSearchState,
        feedback: ParameterFeedback,
    ) -> ParameterSearchState:
        del problem
        return apply_parameter_feedback(state, feedback)


class FixedBindingStrategy:
    """Produce the single empty binding for templates without parameter holes."""

    def next(self, problem: BindingProblem, state: ParameterSearchState) -> CandidateProposal:
        if problem.domains.domains:
            return CandidateProposal.no_proposal("fixed binding strategy requires no domains")
        return next_binding_from_domains(
            problem,
            state,
            (),
            exhausted_reason="fixed binding already attempted or excluded",
        )

    def observe(
        self,
        problem: BindingProblem,
        state: ParameterSearchState,
        feedback: ParameterFeedback,
    ) -> ParameterSearchState:
        del problem
        return apply_parameter_feedback(state, feedback)


def next_binding_from_domains(
    problem: BindingProblem,
    state: ParameterSearchState,
    domains: tuple[ParameterSpace, ...],
    *,
    exhausted_reason: str,
) -> CandidateProposal:
    ordered_domains = proposal_domain_order(domains)
    for space in ordered_domains:
        if parameter_space_is_empty(space):
            return CandidateProposal.no_proposal(f"domain for {space.hole_id} has no options")
        if not isinstance(space, EnumeratedParameterSpace):
            return CandidateProposal.no_proposal(f"domain for {space.hole_id} requires structured predicate search")
    hole_ids = tuple(space.hole_id for space in ordered_domains)
    option_sets = tuple(enumerated_parameter_values(space) for space in ordered_domains)
    value_products = itertools.product(*option_sets) if option_sets else ((),)
    for candidate_values in value_products:
        values = dict(zip(hole_ids, candidate_values, strict=True))
        signature = binding_signature(problem.template, values)
        if signature in state.excluded_binding_signatures:
            continue
        if signature in state.attempted_bindings:
            continue
        return CandidateProposal.candidate(
            ParameterBinding(
                template_signature=problem.template.structural_signature,
                values=values,
                complete_values=True,
                binding_signature=signature,
            )
        )
    return CandidateProposal.no_proposal(exhausted_reason)


def proposal_domain_order(
    domains: tuple[ParameterSpace, ...],
) -> tuple[ParameterSpace, ...]:
    return tuple(
        sorted(
            domains,
            key=lambda domain: 0 if domain.value_kind is not HoleValueKind.QUERY_VECTOR else 1,
        )
    )


def has_predicate_cover_search(problem: BindingProblem) -> bool:
    if not _predicate_cover_domains(problem.domains.domains):
        return False
    return _predicate_cover_target(problem) is not None


def has_fixed_binding(problem: BindingProblem) -> bool:
    return not problem.domains.domains


def predicate_cover_plan(problem: BindingProblem) -> PredicateCoverSearchPlan | str:
    predicate_domains = _predicate_cover_domains(problem.domains.domains)
    if not predicate_domains:
        return "predicate cover search requires predicate domains"
    for space in predicate_domains:
        if parameter_space_is_empty(space):
            return f"domain for {space.hole_id} has no options"
    target = _predicate_cover_target(problem)
    if target is None:
        return "predicate cover search requires a selectivity objective"
    answer_field = _predicate_cover_answer_field(problem.template.root, predicate_domains)
    if answer_field is None:
        return "predicate cover search could not resolve answer field"
    consumer = PredicateCoverConsumerSpec(
        kind=PredicateCoverConsumerKind.COMPLETE_PREDICATE_BINDING,
    )
    return build_predicate_cover_plan(
        problem=problem,
        predicate_domains=predicate_domains,
        consumer=consumer,
        cover_root=problem.template.root,
        answer_field=answer_field,
        target_metric=target[0],
        target_minimum=target[1],
        target_maximum=target[2],
    )


def build_predicate_cover_plan(
    *,
    problem: BindingProblem,
    predicate_domains: tuple[ParameterSpace, ...],
    consumer: PredicateCoverConsumerSpec,
    cover_root: TemplateNode,
    answer_field: FieldId,
    target_metric: PredicateCoverTargetMetric,
    target_minimum: float,
    target_maximum: float,
) -> PredicateCoverSearchPlan:
    strategy = problem.task.predicate_evolution_strategy
    prior = (
        predicate_cover_search_prior(
            problem,
            predicate_domains=predicate_domains,
            target_metric=target_metric,
            target_minimum=target_minimum,
            target_maximum=target_maximum,
        )
        if strategy is PredicateEvolutionStrategyKind.GUIDED
        else PredicateEvolutionPrior()
    )
    evolution = PredicateEvolutionProblem(
        template=problem.template,
        cover_root=cover_root,
        predicate_domains=predicate_domains,
        target_metric=target_metric,
        target_minimum=target_minimum,
        target_maximum=target_maximum,
        strategy=strategy,
        prior=prior,
    )
    return PredicateCoverSearchPlan(
        consumer=consumer,
        answer_field=answer_field,
        evolution=evolution,
    )


def _predicate_cover_domains(
    domains: tuple[ParameterSpace, ...],
) -> tuple[ParameterSpace, ...]:
    non_predicate_values = {
        HoleValueKind.QUERY_VECTOR,
        HoleValueKind.TOP_K,
    }
    return tuple(domain for domain in proposal_domain_order(domains) if domain.value_kind not in non_predicate_values)


def _predicate_cover_target(
    problem: BindingProblem,
) -> tuple[PredicateCoverTargetMetric, float, float] | None:
    for objective in problem.task.objectives:
        if (
            objective.kind is ObjectiveKind.SELECTIVITY_BAND
            and objective.metric is ObjectiveMetric.SELECTIVITY
            and objective.minimum is not None
            and objective.maximum is not None
        ):
            return (
                PredicateCoverTargetMetric.SELECTIVITY,
                float(objective.minimum),
                float(objective.maximum),
            )
    return None


def _predicate_cover_answer_field(
    root: TemplateNode,
    predicate_domains: tuple[ParameterSpace, ...],
) -> FieldId | None:
    for domain in predicate_domains:
        context = hole_context_for_template(root, domain.hole_id)
        if context.field is not None:
            return FieldId(owner=context.field.owner, name="_rowid")
    return None
__all__ = [
    "BindingProblem",
    "CandidateProposal",
    "CandidateProposalKind",
    "FixedBindingStrategy",
    "ParameterFeedback",
    "ParameterSolverKind",
    "ParameterSearchState",
    "ParameterSearchStrategy",
    "PredicateCoverConsumerSpec",
    "PredicateCoverFillStrategy",
    "PredicateCoverSearchRunner",
    "PredicateCoverSearchBudget",
    "PredicateCoverSearchPlan",
    "PredicateCoverSearchResult",
    "profiled_predicate_cover_search",
    "apply_parameter_feedback",
    "has_fixed_binding",
    "has_predicate_cover_search",
    "next_binding_from_domains",
    "observe_search_hint",
    "predicate_binding_fragment",
    "predicate_cover_search_diagnostic_reason",
    "predicate_cover_search_budget",
    "build_predicate_cover_plan",
    "predicate_cover_plan",
    "predicate_cover_search_result",
    "proposal_domain_order",
]
