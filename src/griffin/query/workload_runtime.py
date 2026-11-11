"""Task-local runtime for query generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.query.admit import accepted_binding
from griffin.query.dataset_resources import DatasetResources
from griffin.query.model import SearchIdentity
from griffin.query.observation import ObservationSession
from griffin.query.predicate_cover import PredicateCoverBestSoFar
from griffin.query.spec import GenerationStats
from griffin.query.workload import (
    CommittedQueryTaskResult,
    CompiledQueryTaskPlan,
    EvaluationTicketBudget,
    QueryTaskBudget,
    QueryTaskRunResult,
    QueryTaskTerminalKind,
    ResourceSpend,
    SelectedTemplate,
    SelectedTemplatePool,
    TemplatePoolPlanner,
    TemplateRuntimeFeedback,
    TemplateRuntimeFeedbackKind,
    _continuation_key,
)
from griffin.query.workload_worker import (
    EvaluationTerminalKind,
    EvaluationTicket,
    EvaluationTicketResult,
    EvaluationWorker,
)


@dataclass(frozen=True)
class LaneId:
    continuation_key: str
    template_signature: str

    @property
    def search_key(self) -> str:
        return f"{self.continuation_key}|{self.template_signature}"


@dataclass
class LaneState:
    epoch: int = 0
    continuation_state: object | None = None
    terminal: bool = False


@dataclass
class LaneStateStore:
    _states: dict[LaneId, LaneState] = field(default_factory=dict)

    def continuation(self, lane: LaneId) -> object | None:
        return self._states.get(lane, LaneState()).continuation_state

    def epoch(self, lane: LaneId) -> int:
        return int(self._states.get(lane, LaneState()).epoch)

    def is_terminal(self, lane: LaneId) -> bool:
        return bool(self._states.get(lane, LaneState()).terminal)

    def retire(
        self,
        *,
        lane: LaneId,
        feedback: TemplateRuntimeFeedback,
        continuation_state: object | None,
    ) -> None:
        state = self._states.get(lane, LaneState())
        state.epoch += 1
        if feedback.kind is TemplateRuntimeFeedbackKind.SEARCH_EXHAUSTED:
            state.terminal = True
            state.continuation_state = None
        else:
            state.continuation_state = continuation_state
        self._states[lane] = state


@dataclass
class QueryTaskBudgetLedger:
    budget: QueryTaskBudget
    spend: ResourceSpend = field(default_factory=ResourceSpend)

    def next_ticket_budget(self) -> EvaluationTicketBudget | None:
        remaining_attempts = max(
            int(self.budget.max_attempts) - int(self.spend.attempts),
            0,
        )
        remaining_observations = max(
            int(self.budget.max_observations) - int(self.spend.observations),
            0,
        )
        if remaining_attempts <= 0 or remaining_observations <= 0:
            return None
        return EvaluationTicketBudget(
            max_attempts=remaining_attempts,
            max_observations=remaining_observations,
        )

    def commit(self, spend: ResourceSpend) -> None:
        self.spend = self.spend.add(spend)


class QueryTaskCoordinator:
    def __init__(
        self,
        *,
        planner: TemplatePoolPlanner | None = None,
        worker: EvaluationWorker | None = None,
        task_logger: RuntimeTaskLogger | None = None,
    ) -> None:
        self._planner = planner or TemplatePoolPlanner()
        self._worker = worker or EvaluationWorker(task_logger=task_logger)
        self._shutdown_requested = False

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def run(
        self,
        plan: CompiledQueryTaskPlan,
        *,
        resources: DatasetResources,
        observation_session: ObservationSession,
        search_attempt_ordinal: int,
    ) -> QueryTaskRunResult:
        if search_attempt_ordinal < 0:
            raise ValueError("search_attempt_ordinal must be non-negative")
        selected_pool = self._planner.plan(plan, resources=resources)
        lane_store = LaneStateStore()
        budget = QueryTaskBudgetLedger(plan.budget)
        diagnostics: list[str] = []
        committed: CommittedQueryTaskResult | None = None
        best_so_far: PredicateCoverBestSoFar | None = None
        ticket_ordinal = 0
        terminal_kind = (
            QueryTaskTerminalKind.SEARCH_EXHAUSTED
            if selected_pool.templates
            else QueryTaskTerminalKind.NO_ELIGIBLE_TEMPLATE
        )

        while committed is None and not self._shutdown_requested:
            # 模板选择偏移：query_seed 保证查询间铺开 + attempt 递增保证
            # "同一查询重试时换模板"（否则 attempt N 重选同一个难搜模板，把
            # 剩余预算耗光导致整条查询被跳过）。
            n_templates = max(len(selected_pool.templates), 1)
            seed_offset = (int(plan.identity.query_seed) + search_attempt_ordinal) % n_templates
            selected = _pick_template(selected_pool, lane_store=lane_store, seed_offset=seed_offset)
            if selected is None:
                diagnostics.append("no active template remaining")
                terminal_kind = (
                    QueryTaskTerminalKind.SEARCH_EXHAUSTED
                    if selected_pool.templates
                    else QueryTaskTerminalKind.NO_ELIGIBLE_TEMPLATE
                )
                break
            ticket_budget = budget.next_ticket_budget()
            if ticket_budget is None:
                diagnostics.append("query task budget exhausted")
                terminal_kind = QueryTaskTerminalKind.BUDGET_EXHAUSTED
                break
            lane = _lane_id(selected)
            ticket = EvaluationTicket(
                ticket_id=f"ticket_{ticket_ordinal:06d}",
                search_identity=_search_identity(
                    lane,
                    lane_store,
                    query_seed=plan.identity.query_seed,
                    search_attempt_ordinal=search_attempt_ordinal,
                ),
                selected_template=selected,
                domain_view=selected.base_domains,
                budget=ticket_budget,
                continuation_state=lane_store.continuation(lane),
            )
            result = self._worker.execute_ticket(
                ticket,
                resources=resources,
                observation_session=observation_session,
            )
            if result.best_so_far is not None and (
                best_so_far is None
                or result.best_so_far.objective_distance < best_so_far.objective_distance
            ):
                best_so_far = result.best_so_far
            budget.commit(result.spend)
            committed, feedback = _commit_ticket(
                plan=plan,
                ticket=ticket,
                result=result,
            )
            lane_store.retire(
                lane=lane,
                feedback=feedback,
                continuation_state=result.continuation_state,
            )
            diagnostics.extend(result.diagnostics)
            ticket_ordinal += 1

        if committed is not None:
            terminal_kind = QueryTaskTerminalKind.ACCEPTED
        elif self._shutdown_requested:
            terminal_kind = QueryTaskTerminalKind.SHUTDOWN_REQUESTED

        return QueryTaskRunResult(
            accepted=committed,
            stats=_generation_stats(
                accepted=committed is not None,
                observation_session=observation_session,
                domain_options_scored=selected_pool.domain_options_scored,
                actual_attempts=int(budget.spend.attempts),
            ),
            spend=budget.spend,
            terminal_kind=terminal_kind,
            diagnostics=tuple(diagnostics),
            best_so_far=best_so_far,
        )


def _commit_ticket(
    *,
    plan: CompiledQueryTaskPlan,
    ticket: EvaluationTicket,
    result: EvaluationTicketResult,
) -> tuple[CommittedQueryTaskResult | None, TemplateRuntimeFeedback]:
    for attempt in sorted(
        result.attempts,
        key=lambda item: (int(item.attempt_rank), item.candidate_signature),
    ):
        judgement = attempt.judgement
        if judgement.accepted:
            return (
                CommittedQueryTaskResult(
                    query_spec=plan.query_spec,
                    query_task=ticket.selected_template.query_task,
                    ground_truth_plan=(
                        ticket.selected_template.candidate.ground_truth_plan
                    ),
                    accepted=accepted_binding(
                        materialized=attempt.materialized,
                        evidence=attempt.evidence,
                        judgement=judgement,
                    ),
                ),
                TemplateRuntimeFeedback(
                    kind=TemplateRuntimeFeedbackKind.ACCEPTED,
                ),
            )
    feedback_kind = TemplateRuntimeFeedbackKind.NO_ACCEPTED_CANDIDATE
    if result.terminal_kind is EvaluationTerminalKind.SEARCH_EXHAUSTED:
        feedback_kind = TemplateRuntimeFeedbackKind.SEARCH_EXHAUSTED
    return (
        None,
        TemplateRuntimeFeedback(
            kind=feedback_kind,
        ),
    )


def _pick_template(
    selected_pool: SelectedTemplatePool,
    *,
    lane_store: LaneStateStore,
    seed_offset: int = 0,
) -> SelectedTemplate | None:
    templates = selected_pool.templates
    if not templates:
        return None
    n = len(templates)
    # 用 query_seed 做轮换偏移：否则每条查询都从排序后的第一个模板（通常是产量
    # 最大的 n01 double between）开始选，字段/结构永远不铺开。轮换让不同 query_seed
    # 命中不同字段组合，保持负载的谓词多样性。
    for i in range(n):
        selected = templates[(i + seed_offset) % n]
        if not lane_store.is_terminal(_lane_id(selected)):
            return selected
    return None


def _lane_id(selected: SelectedTemplate) -> LaneId:
    return LaneId(
        continuation_key=_continuation_key(selected),
        template_signature=selected.candidate.template_signature,
    )


def _search_identity(
    lane: LaneId,
    lane_store: LaneStateStore,
    *,
    query_seed: int,
    search_attempt_ordinal: int,
) -> SearchIdentity:
    return SearchIdentity(
        query_seed=query_seed,
        lane_key=lane.search_key,
        epoch=lane_store.epoch(lane),
        ordinal=search_attempt_ordinal,
    )


def _generation_stats(
    *,
    accepted: bool,
    observation_session: ObservationSession,
    domain_options_scored: int,
    actual_attempts: int,
) -> GenerationStats:
    cost_stats = observation_session.total_cost_stats
    attempts = int(actual_attempts)
    accepted_count = int(accepted)
    return GenerationStats(
        total_attempts=attempts,
        accepted=accepted_count,
        rejected=max(0, attempts - accepted_count),
        canonical_evaluations=attempts,
        hint_observations=cost_stats.hint_observations,
        seed_hint_observations=cost_stats.seed_hint_observations,
        local_search_hint_observations=cost_stats.local_search_hint_observations,
        full_search_calls=cost_stats.full_search_calls,
        cover_computations=cost_stats.cover_computations,
        domain_options_scored=domain_options_scored,
    )


__all__ = [
    "LaneId",
    "LaneStateStore",
    "QueryTaskBudgetLedger",
    "QueryTaskCoordinator",
]
