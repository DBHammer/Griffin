"""Query pipeline front door for the rebuilt generation runtime."""

from __future__ import annotations

import pathlib
import os
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

from griffin.dataset.reader import Dataset
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.model_ids import FieldId
from griffin.query.compiler import compile_pipeline_spec
from griffin.query.dataset_resources import DatasetResources
from griffin.query.domain import hole_context_for_template
from griffin.query.finalize import (
    FilteredTopKGroundTruthRecord,
    accepted_predicate_plan,
    finalize_candidate,
    render_materialized_query,
)
from griffin.query.ground_truth import (
    GroundTruthMaterializer,
    GroundTruthProblem,
    GroundTruthResult,
)
from griffin.query.model import (
    HoleValueKind,
)
from griffin.query.predicate_cover import PredicateCoverBestSoFar
from griffin.query.observation import (
    DatasetObservationExecutionPort,
    ObservationSession,
)
from griffin.query.spec import (
    AcceptedQuery,
    CandidateAnalysis,
    CapabilityPolicy,
    DialectTarget,
    GenerationStats,
    GroundTruth,
    QueryPipelineSpec,
    QuerySpec,
    QueryWorkload,
)
from griffin.query.workload import (
    CompiledQueryTaskPlan,
    CompiledWorkloadPlan,
    QueryTaskBudget,
    QueryTaskRunResult,
    QueryTaskTerminalKind,
    ResourceSpend,
)
from griffin.query.workload_runtime import QueryTaskCoordinator
from griffin.query.writer import (
    invalidate_query_run,
    publish_query_run,
)

# 单条查询的时间预算（秒）：超时放弃该查询（外层跳过、继续下一条），
# 避免窄选择率窗口下拒绝采样在单条查询上无限消耗。
# 每查询搜索预算：可用环境变量覆盖（默认 30s）。runner 的 --timeout-sec
# 只是分支级兜底，不会传入引擎；需要更长搜索时必须调这里。
PER_QUERY_TIME_BUDGET_SECONDS = float(os.environ.get("MMBENCHGEN_PER_QUERY_TIME_BUDGET_SECONDS", "30"))


@dataclass(frozen=True)
class _CompletedQueryTask:
    plan: CompiledQueryTaskPlan
    result: QueryTaskRunResult
    observation_session: ObservationSession


def run_pipeline(
    *,
    name: str,
    seed: int,
    dataset_root: str | pathlib.Path,
    dialect: DialectTarget,
    queries: tuple[QuerySpec, ...],
    output_root: str | pathlib.Path,
    capability_policy: CapabilityPolicy | None = None,
) -> QueryWorkload:
    spec = QueryPipelineSpec(
        name=name,
        seed=seed,
        dataset_root=str(dataset_root),
        dialect=dialect,
        queries=queries,
        capability_policy=capability_policy or CapabilityPolicy(),
    )
    dataset = Dataset.open(spec.dataset_root)
    resources = DatasetResources.open(dataset)
    return run(spec, output_root, resources=resources)


def run(
    spec: QueryPipelineSpec,
    output_root: pathlib.Path | str,
    *,
    resources: DatasetResources,
    task_logger: RuntimeTaskLogger | None = None,
) -> QueryWorkload:
    if pathlib.Path(resources.dataset.root).resolve() != pathlib.Path(
        spec.dataset_root
    ).resolve():
        raise ValueError(
            "query resources dataset does not match QueryPipelineSpec.dataset_root"
        )
    output_path = pathlib.Path(output_root)
    run_id = output_path.resolve().name
    invalidate_query_run(
        dataset_root=spec.dataset_root,
        run_id=run_id,
        workload_dir=output_path,
    )
    with _maybe_measure(task_logger, "query.compile"):
        plan = compile_pipeline_spec(spec)
    with _maybe_stage(
        task_logger,
        "准备查询资源",
        profile_name="query.environment",
    ):
        with _maybe_measure(task_logger, "query.prepare"):
            resources.prepare_query_vector_fields(
                _required_query_vector_fields(plan)
            )
    completed_tasks: list[_CompletedQueryTask] = []
    total_queries = len(plan.query_tasks)
    with _maybe_stage(task_logger, "生成并验证查询", profile_name="query.coord"):
        for ordinal, query_task_plan in enumerate(plan.query_tasks, 1):
            print(f"[query] {ordinal}/{total_queries} 生成中...", flush=True)
            query_start = time.perf_counter()
            try:
                completed_tasks.append(
                    _run_query_task_plan(
                        query_task_plan,
                        resources=resources,
                        task_logger=task_logger,
                    )
                )
                print(
                    f"[query] {ordinal}/{total_queries} 完成 "
                    f"({time.perf_counter() - query_start:.1f}s)", flush=True,
                )
            except RuntimeError as exc:
                # 单条查询搜索失败/超时（含 per-query 时间预算耗尽）：跳过该条、
                # 继续生成下一条，保持整体吞吐（最终产物条数可能少于 count 目标）。
                print(
                    f"[skip-query] {ordinal}/{total_queries} {exc} "
                    f"({time.perf_counter() - query_start:.1f}s)",
                    flush=True, file=sys.stderr,
                )
                continue
    ground_truths: list[GroundTruthResult] = []
    with _maybe_stage(
        task_logger,
        "计算查询真实结果",
        profile_name="query.ground_truth",
    ):
        for completed in completed_tasks:
            committed = completed.result.accepted
            if committed is None:
                raise RuntimeError("completed query task lost its accepted result")
            materializer = GroundTruthMaterializer(
                observation_session=completed.observation_session,
            )
            ground_truths.append(materializer.materialize(
                GroundTruthProblem(
                    accepted=committed.accepted,
                    plan=committed.ground_truth_plan,
                )
            ))
    accepted: list[AcceptedQuery] = []
    with _maybe_measure(task_logger, "query.finalize"):
        for slot_index, (completed, ground_truth) in enumerate(
            zip(completed_tasks, ground_truths, strict=True)
        ):
            committed = completed.result.accepted
            if committed is None:
                raise RuntimeError("completed query task lost its accepted result")
            query_spec = completed.plan.query_spec
            runtime_candidate = committed.accepted
            query_id = f"q_{slot_index:04d}"
            rendered_text = render_materialized_query(
                runtime_candidate.materialized,
                dialect_name=spec.dialect.name,
                resources=resources,
            )
            finalized = finalize_candidate(
                query_id=query_id,
                accepted=runtime_candidate,
                ground_truth=ground_truth,
                rendered_query=rendered_text,
            )
            accepted.append(
                AcceptedQuery(
                    query_id=query_id,
                    spec=query_spec,
                    rendered=finalized.rendered_query,
                    analysis=CandidateAnalysis(
                        accepted=runtime_candidate.judgement.accepted,
                        score=runtime_candidate.judgement.score,
                        public_report=runtime_candidate.judgement.public_report,
                        diagnostics="; ".join(runtime_candidate.judgement.failed_constraints),
                    ),
                    ground_truth=_ground_truth_from_artifact(finalized.ground_truth),
                    accepted_predicate_plan=accepted_predicate_plan(runtime_candidate.materialized),
                )
            )
    workload = QueryWorkload(
        name=spec.name,
        accepted=tuple(accepted),
        stats=_aggregate_generation_stats(completed_tasks),
    )
    with _maybe_stage(task_logger, "写入工作负载", profile_name="query.write"):
        publish_query_run(
            dataset_root=spec.dataset_root,
            run_id=run_id,
            workload_dir=output_path,
            workload=workload,
            dialect_name=spec.dialect.name,
        )
    return workload


def _run_query_task_plan(
    plan: CompiledQueryTaskPlan,
    *,
    resources: DatasetResources,
    task_logger: RuntimeTaskLogger | None,
) -> _CompletedQueryTask:
    remaining_budget = plan.budget
    attempt_results: list[QueryTaskRunResult] = []
    attempt_diagnostics: list[str] = []
    best_so_far: PredicateCoverBestSoFar | None = None
    # per-query 时间预算：超时放弃该查询（外层捕获后跳过、继续下一条），
    # 避免窄窗口拒绝采样在单条查询上无限消耗。
    task_deadline = time.perf_counter() + PER_QUERY_TIME_BUDGET_SECONDS
    for search_attempt_ordinal in range(int(plan.budget.max_attempts)):
        if (
            remaining_budget.max_attempts <= 0
            or remaining_budget.max_observations <= 0
        ):
            break
        if time.perf_counter() > task_deadline:
            attempt_diagnostics.append(
                f"per-query time budget ({PER_QUERY_TIME_BUDGET_SECONDS:.0f}s) exhausted"
            )
            break
        if task_logger is not None:
            task_logger.record_event("query.task.search_attempt")
        observation_session = ObservationSession(
            execution_port=DatasetObservationExecutionPort(resources),
            task_logger=task_logger,
        )
        result = QueryTaskCoordinator(task_logger=task_logger).run(
            _query_task_plan_with_budget(plan, remaining_budget),
            resources=resources,
            observation_session=observation_session,
            search_attempt_ordinal=search_attempt_ordinal,
        )
        attempt_results.append(result)
        if result.best_so_far is not None and (
            best_so_far is None
            or result.best_so_far.objective_distance < best_so_far.objective_distance
        ):
            best_so_far = result.best_so_far
        if result.diagnostics:
            attempt_diagnostics.append(
                f"attempt {search_attempt_ordinal}: "
                + "; ".join(result.diagnostics)
            )
        if result.accepted is not None:
            return _CompletedQueryTask(
                plan=plan,
                result=_aggregate_query_task_attempt_results(attempt_results),
                observation_session=observation_session,
            )
        if result.terminal_kind is not QueryTaskTerminalKind.SEARCH_EXHAUSTED:
            break
        if task_logger is not None:
            task_logger.record_event("query.task.search_retry")
        remaining_budget = QueryTaskBudget(
            max_attempts=max(
                int(remaining_budget.max_attempts)
                - max(int(result.spend.attempts), 1),
                0,
            ),
            max_observations=max(
                int(remaining_budget.max_observations)
                - int(result.spend.observations),
                0,
            ),
        )
    diagnostics = "; ".join(attempt_diagnostics)
    best_so_far_suffix = ""
    if best_so_far is not None:
        sel_text = (
            f"{best_so_far.observed_value:.6f}"
            if best_so_far.observed_value == best_so_far.observed_value
            else "nan"
        )
        dist_text = (
            f"{best_so_far.objective_distance:.6f}"
            if best_so_far.objective_distance == best_so_far.objective_distance
            else "inf"
        )
        best_so_far_suffix = (
            f" best_so_far_selectivity={sel_text}"
            f" best_so_far_distance={dist_text}"
        )
        if best_so_far.gls_hit_count is not None:
            best_so_far_suffix += f" best_so_far_gls_hit={best_so_far.gls_hit_count}"
    raise RuntimeError(
        f"query {plan.query_spec.name}: no accepted candidate"
        + (f": {diagnostics}" if diagnostics else "")
        + best_so_far_suffix
    )


def _query_task_plan_with_budget(
    plan: CompiledQueryTaskPlan,
    budget: QueryTaskBudget,
) -> CompiledQueryTaskPlan:
    return CompiledQueryTaskPlan(
        identity=plan.identity,
        query_spec=plan.query_spec,
        query_task=plan.query_task,
        templates=plan.templates,
        budget=budget,
    )


def _aggregate_query_task_attempt_results(
    results: list[QueryTaskRunResult],
) -> QueryTaskRunResult:
    final = results[-1]
    spend = ResourceSpend()
    diagnostics: list[str] = []
    for result in results:
        spend = spend.add(result.spend)
        diagnostics.extend(result.diagnostics)
    return QueryTaskRunResult(
        accepted=final.accepted,
        stats=_sum_generation_stats([result.stats for result in results]),
        spend=spend,
        terminal_kind=final.terminal_kind,
        diagnostics=tuple(diagnostics),
    )


def _required_query_vector_fields(
    plan: CompiledWorkloadPlan,
) -> tuple[FieldId, ...]:
    fields: dict[tuple[str, str, str], FieldId] = {}
    for query_task in plan.query_tasks:
        for candidate in query_task.templates:
            template = candidate.template
            for hole in template.holes:
                if hole.value_kind is not HoleValueKind.QUERY_VECTOR:
                    continue
                context = hole_context_for_template(
                    template.root,
                    hole.hole_id,
                )
                if context.field is None:
                    continue
                owner = context.field.owner
                key = (type(owner).__name__, owner.name, context.field.name)
                fields[key] = context.field
    return tuple(fields[key] for key in sorted(fields))


def _aggregate_generation_stats(
    completed_tasks: list[_CompletedQueryTask],
) -> GenerationStats:
    return _sum_generation_stats(
        [completed.result.stats for completed in completed_tasks]
    )


def _sum_generation_stats(stats: list[GenerationStats]) -> GenerationStats:
    return GenerationStats(
        total_attempts=sum(item.total_attempts for item in stats),
        accepted=sum(item.accepted for item in stats),
        rejected=sum(item.rejected for item in stats),
        canonical_evaluations=sum(item.canonical_evaluations for item in stats),
        hint_observations=sum(item.hint_observations for item in stats),
        seed_hint_observations=sum(item.seed_hint_observations for item in stats),
        local_search_hint_observations=sum(
            item.local_search_hint_observations for item in stats
        ),
        full_search_calls=sum(item.full_search_calls for item in stats),
        cover_computations=sum(item.cover_computations for item in stats),
        domain_options_scored=sum(item.domain_options_scored for item in stats),
    )


def _ground_truth_from_artifact(payload: object) -> GroundTruth | None:
    if isinstance(payload, FilteredTopKGroundTruthRecord):
        return GroundTruth(
            answer_kind=payload.answer_kind,
            metric=payload.metric,
            row_id_column="_rowid",
            k=len(payload.answers),
            row_count=len(payload.answers),
            answers=payload.answers,
            answer_scope=dict(payload.answer_scope),
        )
    return None


class _NullStage:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False


def _maybe_stage(
    task_logger: RuntimeTaskLogger | None,
    name: str,
    *,
    profile_name: str | None = None,
) -> AbstractContextManager[None]:
    if task_logger is None:
        return _NullStage()
    return task_logger.stage(name, profile_name=profile_name)


def _maybe_measure(
    task_logger: RuntimeTaskLogger | None,
    name: str,
) -> AbstractContextManager[None]:
    if task_logger is None:
        return _NullStage()
    return task_logger.measure(name)


__all__ = ["QueryPipelineSpec", "run", "run_pipeline"]
