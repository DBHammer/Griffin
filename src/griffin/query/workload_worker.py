"""Evaluation ticket worker for query workload runtime."""

from __future__ import annotations

import copy
import enum
import json
import os
import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Literal

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.query.dataset_resources import DatasetResources
from griffin.query.domain import DomainBuildResult
from griffin.query.evaluate import (
    CandidateJudgement,
    Evaluator,
    FilteredAnnSearchEvidence,
    PredicateSearchEvidence,
    SearchEvidence,
    filtered_ann_search_evidence,
    judge_search_evidence,
)
from griffin.query.model import MaterializedQuery, SearchIdentity
from griffin.query.observation import (
    ObservationCostStats,
    ObservationScope,
    ObservationSession,
    PredicateCoverHint,
)
from griffin.query.search_core import (
    BindingProblem,
    FilteredAnnPreparedCandidateObservations,
    NoPreparedCandidateObservations,
    ParameterFeedback,
    PreparedCandidateObservations,
    parameter_search_best_so_far,
)
from griffin.query.predicate_cover import PredicateCoverBestSoFar
from griffin.query.search_state import ParameterSearchState
from griffin.query.solve import ParameterSolver
from griffin.query.workload import EvaluationTicketBudget, ResourceSpend, SelectedTemplate


class EvaluationTerminalKind(enum.Enum):
    SEARCH_EXHAUSTED = "search_exhausted"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    OBSERVATION_BUDGET_EXHAUSTED = "observation_budget_exhausted"


class EvaluationAttempt(msgspec.Struct, frozen=True, omit_defaults=True):
    attempt_rank: int
    materialized: MaterializedQuery
    evidence: SearchEvidence
    judgement: CandidateJudgement
    binding_signature: str
    candidate_signature: str
    prepared_observations: PreparedCandidateObservations = (
        NoPreparedCandidateObservations()
    )


class EvaluationTicket(msgspec.Struct, frozen=True, omit_defaults=True):
    ticket_id: str
    search_identity: SearchIdentity
    selected_template: SelectedTemplate
    domain_view: DomainBuildResult
    budget: EvaluationTicketBudget
    continuation_state: object | None = None


class EvaluationTicketResult(msgspec.Struct, frozen=True, omit_defaults=True):
    ticket_id: str
    template_signature: str
    attempts: tuple[EvaluationAttempt, ...]
    spend: ResourceSpend
    continuation_state: object | None = None
    terminal_kind: EvaluationTerminalKind | None = None
    diagnostics: tuple[str, ...] = ()
    best_so_far: PredicateCoverBestSoFar | None = None


class EvaluationWorker:
    def __init__(
        self,
        *,
        solver: ParameterSolver | None = None,
        evaluator: Evaluator | None = None,
        task_logger: RuntimeTaskLogger | None = None,
    ) -> None:
        self._solver = solver or ParameterSolver()
        self._evaluator = evaluator or Evaluator()
        self._task_logger = task_logger
        self._attempt_start = time.perf_counter()

    def _log_attempt(
        self,
        rank: int,
        evidence: SearchEvidence,
        judgement: CandidateJudgement,
    ) -> None:
        log_path = os.environ.get("MMBENCHGEN_GRIFFIN_ATTEMPT_LOG", "")
        if not log_path:
            return
        elapsed = time.perf_counter() - self._attempt_start
        selection = (
            evidence.selection
            if isinstance(
                evidence,
                PredicateSearchEvidence | FilteredAnnSearchEvidence,
            )
            else None
        )
        sel = None if selection is None else selection.cover.selectivity
        gls = (
            judgement.public_report.achieved
            if isinstance(evidence, FilteredAnnSearchEvidence)
            and judgement.public_report is not None
            else None
        )
        row = {
            "attempt_rank": rank,
            "elapsed_sec": round(elapsed, 4),
            "accepted": int(judgement.accepted),
            "score": round(judgement.score, 6),
            "selectivity": round(sel, 10) if sel is not None else None,
            "gls": round(gls, 10) if gls is not None else None,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def execute_ticket(
        self,
        ticket: EvaluationTicket,
        *,
        resources: DatasetResources,
        observation_session: ObservationSession,
    ) -> EvaluationTicketResult:
        search_state = _continuation_state(ticket.continuation_state)
        search_state.parameter_attempts = 0
        scope = ObservationScope(scope_id=ticket.ticket_id)
        query_task = ticket.selected_template.query_task
        problem = BindingProblem(
            task=query_task,
            search_identity=ticket.search_identity,
            template=ticket.selected_template.candidate.template,
            domains=ticket.domain_view,
            budget=max(int(ticket.budget.max_attempts), 1),
            solver_kind=ticket.selected_template.candidate.solver_kind,
            resources=resources,
            observation_session=observation_session,
            observation_scope=scope,
        )
        with _maybe_measure(self._task_logger, "query.ticket.setup"):
            solver_session = self._solver.start(problem, state=search_state)
        attempts: list[EvaluationAttempt] = []
        diagnostics: list[str] = []
        terminal_kind: EvaluationTerminalKind | None = None
        while len(attempts) < int(ticket.budget.max_attempts):
            if _observation_spend(observation_session.scope_cost_stats(scope)) >= int(ticket.budget.max_observations):
                terminal_kind = EvaluationTerminalKind.OBSERVATION_BUDGET_EXHAUSTED
                diagnostics.append("ticket observation budget exhausted")
                break
            with _maybe_measure(self._task_logger, "query.solve"):
                proposal = solver_session.next()
            if proposal.binding is None:
                terminal_kind = EvaluationTerminalKind.SEARCH_EXHAUSTED
                diagnostics.append(proposal.diagnostics.get("reason", "no proposal"))
                break
            materialized = MaterializedQuery(
                template=ticket.selected_template.candidate.template,
                binding=proposal.binding,
            )
            with _maybe_measure(self._task_logger, "query.acceptance"):
                prepared_covers = _prepared_predicate_covers(
                    proposal.prepared_observations
                )
                prepared_hit = (
                    proposal.prepared_observations.hit
                    if isinstance(
                        proposal.prepared_observations,
                        FilteredAnnPreparedCandidateObservations,
                    )
                    else None
                )
                prepared_evidence = filtered_ann_search_evidence(
                    query_task=query_task,
                    query=materialized,
                    search_identity=ticket.search_identity,
                    resources=resources,
                    predicate_covers=prepared_covers,
                    hit=prepared_hit,
                )
                if prepared_evidence is None:
                    evidence = self._evaluator.evaluate(
                        materialized,
                        resources,
                        observation_session,
                        query_task=query_task,
                        search_identity=ticket.search_identity,
                        prepared_predicate_covers=prepared_covers,
                        observation_scope=scope,
                    )
                else:
                    evidence = prepared_evidence
            with _maybe_measure(self._task_logger, "query.judge"):
                judgement = judge_search_evidence(
                    query_task.objectives,
                    evidence,
                )
            self._log_attempt(len(attempts), evidence, judgement)
            attempts.append(
                EvaluationAttempt(
                    attempt_rank=len(attempts),
                    materialized=materialized,
                    evidence=evidence,
                    judgement=judgement,
                    binding_signature=proposal.binding.binding_signature,
                    candidate_signature=proposal.binding.binding_signature,
                    prepared_observations=proposal.prepared_observations,
                )
            )
            with _maybe_measure(self._task_logger, "query.solve_feedback"):
                solver_session.observe(
                    ParameterFeedback(
                        binding_signature=proposal.binding.binding_signature,
                        accepted=judgement.accepted,
                        score=judgement.score,
                        evidence=evidence,
                        search_feedback=judgement.search_feedback,
                    )
                )
            if judgement.accepted:
                break
            if _observation_spend(observation_session.scope_cost_stats(scope)) >= int(ticket.budget.max_observations):
                terminal_kind = EvaluationTerminalKind.OBSERVATION_BUDGET_EXHAUSTED
                diagnostics.append("ticket observation budget exhausted")
                break
        if terminal_kind is None and len(attempts) >= int(ticket.budget.max_attempts):
            terminal_kind = EvaluationTerminalKind.ATTEMPT_BUDGET_EXHAUSTED
            diagnostics.append("ticket attempt budget exhausted")
        spend = _resource_spend(observation_session.scope_cost_stats(scope))
        best_so_far = (
            parameter_search_best_so_far(solver_session.state)
            if query_task.best_so_far_fallback
            else None
        )
        return EvaluationTicketResult(
            ticket_id=ticket.ticket_id,
            template_signature=ticket.selected_template.candidate.template_signature,
            attempts=tuple(attempts),
            spend=ResourceSpend(
                attempts=max(len(attempts), spend.attempts),
                observations=spend.observations,
            ),
            continuation_state=_persistable_continuation_state(solver_session.state),
            terminal_kind=terminal_kind,
            diagnostics=tuple(diagnostics),
            best_so_far=best_so_far,
        )


class _NullMeasure:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False


def _maybe_measure(task_logger: RuntimeTaskLogger | None, name: str) -> AbstractContextManager[None]:
    if task_logger is None:
        return _NullMeasure()
    return task_logger.measure(name)


def _continuation_state(raw: object | None) -> ParameterSearchState:
    if raw is None:
        return ParameterSearchState()
    if not isinstance(raw, ParameterSearchState):
        raise ContractViolation(f"unsupported continuation state {type(raw).__name__}")
    return copy.deepcopy(raw)


def _persistable_continuation_state(state: ParameterSearchState) -> ParameterSearchState:
    state.vector.domains = None
    if state.vector.filtered_ann.pending_proposals:
        raise ContractViolation("filtered ANN pending proposals must be consumed before persisting continuation")
    return state


def _prepared_predicate_covers(
    prepared: PreparedCandidateObservations,
) -> dict[str, PredicateCoverHint]:
    if isinstance(prepared, NoPreparedCandidateObservations):
        return {}
    return {cover.node_id: cover for cover in prepared.covers}


def _observation_spend(stats: ObservationCostStats) -> int:
    return int(stats.exact_calls + stats.canonical_observations + stats.cover_computations + stats.full_search_calls)


def _resource_spend(stats: ObservationCostStats) -> ResourceSpend:
    return ResourceSpend(observations=_observation_spend(stats))


__all__ = [
    "EvaluationAttempt",
    "EvaluationTerminalKind",
    "EvaluationTicket",
    "EvaluationTicketResult",
    "EvaluationWorker",
]
