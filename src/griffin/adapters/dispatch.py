from __future__ import annotations

import logging
import time
from pathlib import Path

from griffin.config import (
    JobConfig,
    QueryJobConfig,
    RelationalDataJobConfig,
    load_job_config,
    run_task_config as _run_task_config,
)
from griffin.infra.runtime_logging import RuntimeTaskLogger
def run_task_path(
    path: str | Path,
    *,
    runtime_logger: logging.Logger | None = None,
) -> dict[str, object]:
    config_path = Path(path)
    started_at = time.perf_counter()
    if runtime_logger is not None:
        runtime_logger.info("[任务] 开始读取配置：配置=%s", str(config_path))
    try:
        config = load_job_config(config_path)
    except Exception as exc:
        if runtime_logger is not None:
            runtime_logger.error("[任务] 读取配置失败：配置=%s，原因=%s", str(config_path), exc)
        raise

    task_logger = _task_logger_for_config(config, runtime_logger)
    if task_logger is not None:
        task_logger.task_start(detail=_task_start_detail(config, config_path=config_path))
        task_logger.progress(f"读取配置完成，耗时 {time.perf_counter() - started_at:.3f} 秒")
    try:
        result = _run_task_config(
            config,
            task_logger=task_logger,
        )
    except Exception as exc:
        if task_logger is not None:
            task_logger.task_fail(exc)
        raise
    if task_logger is not None:
        task_logger.task_finish(detail=_task_finish_detail(config))
    return result


def run_task_config(
    config: JobConfig,
    *,
    runtime_logger: logging.Logger | None = None,
) -> dict[str, object]:
    task_logger = _task_logger_for_config(config, runtime_logger)
    if task_logger is not None:
        task_logger.task_start(detail=_task_start_detail(config))
    try:
        result = _run_task_config(
            config,
            task_logger=task_logger,
        )
    except Exception as exc:
        if task_logger is not None:
            task_logger.task_fail(exc)
        raise
    if task_logger is not None:
        task_logger.task_finish(detail=_task_finish_detail(config))
    return result


def _task_logger_for_config(
    config: RelationalDataJobConfig | QueryJobConfig,
    runtime_logger: logging.Logger | None,
) -> RuntimeTaskLogger | None:
    if runtime_logger is None:
        return None
    task_kind = "data"
    if isinstance(config, QueryJobConfig):
        task_kind = "query"
    return RuntimeTaskLogger(runtime_logger, task_kind)


def _task_start_detail(
    config: RelationalDataJobConfig | QueryJobConfig,
    *,
    config_path: Path | None = None,
) -> str:
    parts: list[str] = []
    if config_path is not None:
        parts.append(f"配置={config_path}")
    if isinstance(config, RelationalDataJobConfig):
        parts.append(f"输出目录={Path(config.output_dir).resolve()}")
    elif isinstance(config, QueryJobConfig):
        parts.append(f"输入目录={Path(config.input_dir).resolve()}")
        if config.output_dir is not None:
            parts.append(f"输出目录={Path(config.output_dir).resolve()}")
    else:
        parts.append(f"输出目录={Path(config.output_dir).resolve()}")
    return " ".join(parts)


def _task_finish_detail(
    config: RelationalDataJobConfig | QueryJobConfig,
) -> str:
    if isinstance(config, QueryJobConfig):
        if config.output_dir is None:
            return f"输入目录={Path(config.input_dir).resolve()}"
        return f"输出目录={Path(config.output_dir).resolve()}"
    return f"输出目录={Path(config.output_dir).resolve()}"


__all__ = ["run_task_path", "run_task_config"]
