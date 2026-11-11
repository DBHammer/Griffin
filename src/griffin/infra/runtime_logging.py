from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


_LOGGER_NAME = "griffin.task"
RUNTIME_PROFILE_ENV = "MMBENCHGEN_RUNTIME_PROFILE"

RUNTIME_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
RUNTIME_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


@dataclass(slots=True)
class _RuntimeProfileFrame:
    name: str
    started_at: float
    child_time_sec: float = 0.0


@dataclass(slots=True)
class _RuntimeProfileAggregate:
    count: int = 0
    failed_count: int = 0
    wall_time_sec: float = 0.0
    self_time_sec: float = 0.0


@dataclass(slots=True)
class _RuntimeProfileRecorder:
    path: pathlib.Path
    _aggregates: dict[str, _RuntimeProfileAggregate] = field(default_factory=dict)
    _task_started_at: float | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _task_local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _task_run_id: int = 0

    def _current_stack(self) -> list[_RuntimeProfileFrame]:
        stack = getattr(self._task_local, "stack", None)
        if stack is None:
            stack = []
            self._task_local.stack = stack
        return stack

    def _current_run_id(self) -> int | None:
        value = getattr(self._task_local, "run_id", None)
        if not isinstance(value, int):
            return None
        return value

    def _attach_thread(self) -> bool:
        with self._lock:
            task_started_at = self._task_started_at
            run_id = self._task_run_id
        if task_started_at is None:
            return False
        stack = self._current_stack()
        current_run_id = self._current_run_id()
        if current_run_id != run_id:
            stack.clear()
            stack.append(_RuntimeProfileFrame(name="task.total", started_at=task_started_at))
            self._task_local.run_id = run_id
        elif not stack:
            stack.append(_RuntimeProfileFrame(name="task.total", started_at=task_started_at))
        return True

    def task_start(self) -> None:
        started_at = time.perf_counter()
        with self._lock:
            self._aggregates.clear()
            self._task_started_at = started_at
            self._task_run_id += 1
        self._current_stack().clear()
        self._task_local.run_id = self._task_run_id
        self._current_stack().append(_RuntimeProfileFrame(name="task.total", started_at=started_at))

    def start_span(self, name: str) -> None:
        if not self._attach_thread():
            return
        self._current_stack().append(_RuntimeProfileFrame(name=name, started_at=time.perf_counter()))

    def finish_span(self, *, failed: bool) -> None:
        if not self._attach_thread():
            return
        stack = self._current_stack()
        if len(stack) <= 1:
            return
        frame = stack.pop()
        elapsed = max(time.perf_counter() - frame.started_at, 0.0)
        self._record_duration(frame.name, elapsed_sec=elapsed, failed=failed, child_time_sec=frame.child_time_sec)

    def record_duration(
        self,
        name: str,
        *,
        elapsed_sec: float,
        failed: bool = False,
        include_in_parent: bool = True,
    ) -> None:
        if not self._attach_thread():
            return
        self._record_duration(
            name,
            elapsed_sec=max(float(elapsed_sec), 0.0),
            failed=failed,
            child_time_sec=0.0,
            include_in_parent=include_in_parent,
        )

    def _record_duration(
        self,
        name: str,
        *,
        elapsed_sec: float,
        failed: bool,
        child_time_sec: float,
        include_in_parent: bool = True,
    ) -> None:
        self_time = max(elapsed_sec - child_time_sec, 0.0)
        with self._lock:
            aggregate = self._aggregates.setdefault(name, _RuntimeProfileAggregate())
            aggregate.count += 1
            aggregate.wall_time_sec += elapsed_sec
            aggregate.self_time_sec += self_time
            if failed:
                aggregate.failed_count += 1
        stack = self._current_stack()
        if include_in_parent and stack:
            stack[-1].child_time_sec += elapsed_sec

    def record_event(self, name: str, *, count: int = 1) -> None:
        if count <= 0 or not self._attach_thread():
            return
        with self._lock:
            aggregate = self._aggregates.setdefault(name, _RuntimeProfileAggregate())
            aggregate.count += int(count)

    def finish_task(self, *, task_kind: str, status: str) -> None:
        failed = status != "passed"
        while len(self._current_stack()) > 1:
            self.finish_span(failed=failed)
        stack = self._current_stack()
        if stack:
            root = stack.pop()
            elapsed = max(time.perf_counter() - root.started_at, 0.0)
            self._record_duration(
                root.name,
                elapsed_sec=elapsed,
                failed=failed,
                child_time_sec=root.child_time_sec,
                include_in_parent=False,
            )
        total_wall_time = 0.0
        with self._lock:
            task_started_at = self._task_started_at
            aggregates = {
                name: _RuntimeProfileAggregate(
                    count=aggregate.count,
                    failed_count=aggregate.failed_count,
                    wall_time_sec=aggregate.wall_time_sec,
                    self_time_sec=aggregate.self_time_sec,
                )
                for name, aggregate in self._aggregates.items()
            }
            self._task_started_at = None
        if task_started_at is not None:
            total_wall_time = max(time.perf_counter() - task_started_at, 0.0)
        payload = {
            "profile_version": 1,
            "task_kind": task_kind,
            "status": status,
            "total_wall_time_sec": round(total_wall_time, 6),
            "aggregates": [
                {
                    "name": name,
                    "count": aggregate.count,
                    "failed_count": aggregate.failed_count,
                    "wall_time_sec": round(aggregate.wall_time_sec, 6),
                    "self_time_sec": round(aggregate.self_time_sec, 6),
                }
                for name, aggregate in sorted(aggregates.items())
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_log_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=RUNTIME_LOG_FORMAT, datefmt=RUNTIME_LOG_DATEFMT)


def configure_cli_logging() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(runtime_log_formatter())
        logger.addHandler(handler)
    logger.setLevel(_runtime_log_level())
    logger.propagate = False
    return logger


def _runtime_log_level() -> int:
    configured = os.environ.get("MMBENCHGEN_LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelName(configured)
    if isinstance(level, int):
        return level
    return logging.INFO


@dataclass(slots=True)
class RuntimeTaskLogger:
    logger: logging.Logger
    task_kind: str
    task_started_at: float | None = None
    _profile: _RuntimeProfileRecorder | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        profile_path = os.environ.get(RUNTIME_PROFILE_ENV, "").strip()
        if profile_path:
            self._profile = _RuntimeProfileRecorder(pathlib.Path(profile_path).expanduser().resolve())

    def task_start(self, *, detail: str | None = None) -> None:
        self.task_started_at = time.perf_counter()
        if self._profile is not None:
            self._profile.task_start()
        message = "任务开始"
        if detail:
            message = f"{message}：{detail}"
        self._info(message)

    def task_finish(self, *, detail: str | None = None) -> None:
        elapsed = self._task_elapsed()
        message = f"任务完成，总耗时 {elapsed:.3f} 秒"
        if detail:
            message = f"{message}，{detail}"
        self._info(message)
        if self._profile is not None:
            self._profile.finish_task(task_kind=self.task_kind, status="passed")

    def task_fail(self, exc: BaseException) -> None:
        elapsed = self._task_elapsed()
        self.logger.error(
            "[%s] 任务失败，总耗时 %.3f 秒：%s",
            self.task_kind,
            elapsed,
            exc,
        )
        if self._profile is not None:
            self._profile.finish_task(task_kind=self.task_kind, status="failed")

    @contextmanager
    def stage(self, name: str, *, profile_name: str | None = None) -> Iterator[None]:
        self._info(f"进入{name}阶段")
        started_at = time.perf_counter()
        if profile_name is not None and self._profile is not None:
            self._profile.start_span(profile_name)
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            if profile_name is not None and self._profile is not None:
                self._profile.finish_span(failed=True)
            self.logger.error(
                "[%s] %s阶段失败，耗时 %.3f 秒：%s",
                self.task_kind,
                name,
                elapsed,
                exc,
            )
            raise
        else:
            elapsed = time.perf_counter() - started_at
            if profile_name is not None and self._profile is not None:
                self._profile.finish_span(failed=False)
            self._info(f"{name}阶段完成，耗时 {elapsed:.3f} 秒")

    @contextmanager
    def step(
        self,
        name: str,
        *,
        detail: str | None = None,
        profile_name: str | None = None,
    ) -> Iterator[None]:
        start_message = name
        if detail:
            start_message = f"{start_message}：{detail}"
        self._info(start_message)
        started_at = time.perf_counter()
        if profile_name is not None and self._profile is not None:
            self._profile.start_span(profile_name)
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            if profile_name is not None and self._profile is not None:
                self._profile.finish_span(failed=True)
            if detail:
                self.logger.error(
                    "[%s] %s失败：%s，耗时 %.3f 秒：%s",
                    self.task_kind,
                    name,
                    detail,
                    elapsed,
                    exc,
                )
            else:
                self.logger.error(
                    "[%s] %s失败，耗时 %.3f 秒：%s",
                    self.task_kind,
                    name,
                    elapsed,
                    exc,
                )
            raise
        else:
            elapsed = time.perf_counter() - started_at
            if profile_name is not None and self._profile is not None:
                self._profile.finish_span(failed=False)
            finish_message = f"{name}完成"
            if detail:
                finish_message = f"{finish_message}：{detail}"
            self._info(f"{finish_message}，耗时 {elapsed:.3f} 秒")

    @contextmanager
    def measure(self, name: str, *, log: bool = False, detail: str | None = None) -> Iterator[None]:
        started_at = time.perf_counter()
        if log:
            message = name if detail is None else f"{name}：{detail}"
            self._info(message)
        if self._profile is not None:
            self._profile.start_span(name)
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            if self._profile is not None:
                self._profile.finish_span(failed=True)
            if log:
                if detail:
                    self.logger.error(
                        "[%s] %s失败：%s，耗时 %.3f 秒：%s",
                        self.task_kind,
                        name,
                        detail,
                        elapsed,
                        exc,
                    )
                else:
                    self.logger.error(
                        "[%s] %s失败，耗时 %.3f 秒：%s",
                        self.task_kind,
                        name,
                        elapsed,
                        exc,
                    )
            raise
        else:
            elapsed = time.perf_counter() - started_at
            if self._profile is not None:
                self._profile.finish_span(failed=False)
            if log:
                finish = f"{name}完成"
                if detail:
                    finish = f"{finish}：{detail}"
                self._info(f"{finish}，耗时 {elapsed:.3f} 秒")

    def progress(self, message: str) -> None:
        self._info(message)

    def record_duration(
        self,
        name: str,
        elapsed_sec: float,
        *,
        failed: bool = False,
        include_in_parent: bool = True,
    ) -> None:
        if self._profile is None:
            return
        self._profile.record_duration(
            name,
            elapsed_sec=elapsed_sec,
            failed=failed,
            include_in_parent=include_in_parent,
        )

    def record_event(self, name: str, *, count: int = 1) -> None:
        if self._profile is None:
            return
        self._profile.record_event(name, count=count)

    def _task_elapsed(self) -> float:
        if self.task_started_at is None:
            return 0.0
        return time.perf_counter() - self.task_started_at

    def _info(self, message: str) -> None:
        self.logger.info("[%s] %s", self.task_kind, message)


__all__ = [
    "RUNTIME_PROFILE_ENV",
    "RUNTIME_LOG_DATEFMT",
    "RUNTIME_LOG_FORMAT",
    "RuntimeTaskLogger",
    "configure_cli_logging",
    "runtime_log_formatter",
]
