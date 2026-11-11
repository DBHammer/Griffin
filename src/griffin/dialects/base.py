"""Rendered PostgreSQL query contract."""

from __future__ import annotations

import msgspec


class RenderedQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    """The on-the-wire query text + parameter bindings."""

    text: str
    parameters_json: str  # JSON: list[Any] (positional) or dict[str,Any] (named)


__all__ = ["RenderedQuery"]
