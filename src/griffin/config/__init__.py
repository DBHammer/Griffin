from collections.abc import Mapping
from pathlib import Path

from griffin.data.pipeline import DataPipelineSpec
from griffin.dataset.reader import Dataset
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.query.pipeline import QueryPipelineSpec
from .models import (
    DataJobConfig,
    DatasetOutputFormat,
    JobConfig,
    QueryJobConfig,
    RelationalDataJobConfig,
    RelationalDialect,
)


def load_job_config(path: str | Path) -> JobConfig:
    from .load import load_job_config as _impl

    return _impl(path)


def load_job_config_payload(payload: Mapping[str, object]) -> JobConfig:
    from .load import load_job_config_payload as _impl

    return _impl(payload)


def compile_data_job(config: DataJobConfig) -> DataPipelineSpec:
    from .compile import compile_data_job as _impl

    return _impl(config)


def compile_query_job(config: QueryJobConfig, dataset: Dataset) -> QueryPipelineSpec:
    from .compile import compile_query_job as _impl

    return _impl(config, dataset)


def run_task_config(
    config: JobConfig,
    *,
    task_logger: RuntimeTaskLogger | None = None,
) -> dict[str, object]:
    from .runtime import run_task_config as _impl

    return _impl(
        config,
        task_logger=task_logger,
    )


__all__ = [
    "compile_data_job",
    "compile_query_job",
    "load_job_config",
    "load_job_config_payload",
    "run_task_config",
    "DataJobConfig",
    "DatasetOutputFormat",
    "JobConfig",
    "QueryJobConfig",
    "RelationalDataJobConfig",
    "RelationalDialect",
]
