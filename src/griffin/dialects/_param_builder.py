"""ParamBuilder — visitor-internal parameter accumulator.

The PostgreSQL renderer instantiates a ParamBuilder per render() call to
collect positional parameter bindings using `%s` placeholders.

Bindings never see this class — it lives strictly inside renderers.
"""

from __future__ import annotations

import enum
import json
from typing import Any


class ParamStyle(enum.Enum):
    PERCENT_S = "percent_s"  # PostgreSQL: %s


class ParamBuilder:
    """Append-only parameter accumulator.

    `add(value)` returns `%s` and stores `value` at the next index.
    """

    __slots__ = ("_style", "_positional")

    def __init__(self, style: ParamStyle) -> None:
        self._style = style
        self._positional: list[Any] = []

    def add(self, value: Any, *, name: str | None = None) -> str:
        if self._style == ParamStyle.PERCENT_S:
            self._positional.append(value)
            return "%s"
        raise ValueError(f"unknown param style: {self._style}")

    def freeze(self) -> str:
        """Encode the accumulated bindings as JSON for RenderedQuery."""
        return json.dumps(self._positional, ensure_ascii=False, default=list)


__all__ = ["ParamStyle", "ParamBuilder"]
