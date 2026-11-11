from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from griffin.data.pipeline import run as run_data
from griffin.dataset.deterministic import VectorAttributeCouplingModel
from griffin.dataset.reader import Dataset
from griffin.infra.errors import ContractViolation
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.ir.data_type import DoubleType, IntType, VectorType
from griffin.ir.field import Field, FieldRole
from griffin.query.pipeline import run as run_query
from griffin.query.dataset_resources import DatasetResources
from griffin.query.spec import ground_truth_count
from .compile import compile_data_job, compile_query_job
from .models import JobConfig, QueryJobConfig, RelationalDataJobConfig


def run_task_config(
    config: JobConfig,
    *,
    task_logger: RuntimeTaskLogger | None = None,
) -> dict[str, object]:
    if isinstance(config, RelationalDataJobConfig):
        if task_logger is not None:
            with task_logger.stage("编译任务"):
                spec = compile_data_job(config)
        else:
            spec = compile_data_job(config)
        output_root = Path(config.output_dir).expanduser().resolve()
        run_data(spec, output_root, task_logger=task_logger)
        if task_logger is not None:
            with task_logger.stage("打开数据集"):
                dataset = Dataset.open(output_root)
        else:
            dataset = Dataset.open(output_root)
        _log_data_dataset_summary(task_logger, dataset)
        return {
            "data": {
                "artifact": {"root": str(output_root)},
                "workspace_dir": str(output_root),
                "data_model": dataset.data_model,
                "dataset": {
                    "name": dataset.name,
                    "table_count": len(dataset.tables()),
                    "entity_count": dataset.entity_count(),
                },
            }
        }

    if not isinstance(config, QueryJobConfig):
        raise ContractViolation(f"unsupported task config type: {type(config).__name__}")
    if task_logger is not None:
        with task_logger.stage(
            "准备查询环境",
            profile_name="query.environment",
        ):
            with task_logger.measure("query.open_dataset"):
                dataset = Dataset.open(config.input_dir)
            resources = DatasetResources.open(dataset)
    else:
        dataset = Dataset.open(config.input_dir)
        resources = DatasetResources.open(dataset)
    output_root = Path(
        config.output_dir
        if config.output_dir is not None
        else Path(dataset.root) / "workloads" / f"query_{config.spec.family.value}_{config.seed}"
    ).expanduser().resolve()
    if task_logger is not None:
        with task_logger.stage("编译任务", profile_name="query.compile"):
            query_spec = compile_query_job(config, dataset)
    else:
        query_spec = compile_query_job(config, dataset)
    if task_logger is not None:
        with task_logger.stage("运行查询主线", profile_name="query.run"):
            workload = run_query(
                query_spec,
                output_root,
                resources=resources,
                task_logger=task_logger,
            )
    else:
        workload = run_query(
            query_spec,
            output_root,
            resources=resources,
        )
    return {
        "query": {
            "dataset_dir": dataset.root,
            "data_model": dataset.data_model,
            "artifact": {
                "root": str(output_root),
                "query_count": workload.stats.accepted,
                "accepted": workload.stats.accepted,
                "rejected": workload.stats.rejected,
                "total_attempts": workload.stats.total_attempts,
                "canonical_evaluations": workload.stats.canonical_evaluations,
                "search_cost": {
                    "canonical_evaluations": workload.stats.canonical_evaluations,
                    "hint_observations": workload.stats.hint_observations,
                    "seed_hint_observations": workload.stats.seed_hint_observations,
                    "local_search_hint_observations": workload.stats.local_search_hint_observations,
                    "full_search_calls": workload.stats.full_search_calls,
                    "cover_computations": workload.stats.cover_computations,
                    "domain_options_scored": workload.stats.domain_options_scored,
                },
                "ground_truth_count": ground_truth_count(workload),
            },
        }
    }


def _log_data_dataset_summary(task_logger: RuntimeTaskLogger | None, dataset: Dataset) -> None:
    if task_logger is None:
        return

    task_logger.progress(
        "数据集摘要："
        f"name={dataset.name}，模型={dataset.data_model}，"
        f"方言={dataset.dialect}，格式={dataset.output_format}"
    )
    row_counts = _data_row_counts(dataset)
    table_summary = _table_scale_summary(dataset, row_counts)
    if table_summary:
        task_logger.progress(f"表规模：{table_summary}")
    task_logger.progress(f"字段覆盖：{_field_coverage_summary(dataset)}")
    for coupling_summary in _coupling_quality_summaries(dataset):
        task_logger.progress(f"耦合质量：{coupling_summary}")


def _data_row_counts(dataset: Dataset) -> dict[str, int]:
    row_counts = {metrics.owner.name: metrics.row_count for metrics in dataset.metrics()}
    for table_meta in dataset.deterministic_tables():
        row_counts.setdefault(table_meta.table_name, table_meta.row_count)
    return row_counts


def _table_scale_summary(dataset: Dataset, row_counts: dict[str, int]) -> str:
    return _join_summary_parts(
        f"{table.id.name}={row_counts.get(table.id.name, 0)} 行" for table in dataset.tables()
    )


def _field_coverage_summary(dataset: Dataset) -> str:
    counts = {
        "numeric": 0,
        "categorical": 0,
        "vector": 0,
        "other": 0,
    }
    for table in dataset.tables():
        for field in table.fields:
            kind = _field_coverage_kind(field)
            if kind is None:
                continue
            counts[kind] += 1
    return "，".join(f"{kind}={count}" for kind, count in counts.items())


def _field_coverage_kind(field: Field) -> str | None:
    if field.role in (FieldRole.PRIMARY_KEY, FieldRole.FOREIGN_KEY):
        return None
    if field.role is FieldRole.VECTOR or isinstance(field.type, VectorType):
        return "vector"
    if field.role is FieldRole.CATEGORICAL:
        return "categorical"
    if isinstance(field.type, (IntType, DoubleType)):
        return "numeric"
    return "other"


def _coupling_quality_summaries(dataset: Dataset) -> tuple[str, ...]:
    summaries: list[str] = []
    for table_meta in dataset.deterministic_tables():
        summaries.extend(
            _format_coupling_summary(table_meta.table_name, coupling)
            for coupling in table_meta.vector_attribute_couplings
        )
    return tuple(summaries)


def _format_coupling_summary(owner_name: str, coupling: VectorAttributeCouplingModel) -> str:
    target = coupling.target_rho
    observed = coupling.observed_rho
    absolute_error = abs(observed - target)
    if abs(target) > 1e-12:
        error_text = f"relative_error={absolute_error / abs(target) * 100:.2f}%"
    else:
        error_text = f"absolute_error={absolute_error:.3f}"
    return (
        f"{owner_name}.{coupling.vector_field_name}~{coupling.attribute_field_name} "
        f"target_rho={target:.3f}，observed_rho={observed:.3f}，{error_text}"
    )


def _join_summary_parts(parts: Iterable[object]) -> str:
    return "，".join(str(part) for part in parts)


__all__ = ["run_task_config"]
