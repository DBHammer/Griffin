"""Progress reporting (typed, callback-based).

Callers pass a `ProgressSink` Protocol to long-running APIs. Production code
passes a sink from the owning session; tests use `RecordingSink`. We avoid
coupling to tqdm/loguru in the core.
"""

from __future__ import annotations

from typing import Protocol

import msgspec


class ProgressEvent(msgspec.Struct, frozen=True, omit_defaults=True):
    stage: str
    message: str
    completed: int
    total: int


class ProgressSink(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...


class RecordingSink:
    """Test helper: collects every event in order."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


__all__ = ["ProgressEvent", "ProgressSink", "RecordingSink"]
