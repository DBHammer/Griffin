"""Lowest-layer typed identity.

This module MUST NOT import from any other `griffin.*` package.
A guard test enforces this in `tests/guards/test_isolation.py`.

`FieldId`, `TableId`, `RelationId`, and `ArtifactId` are the structural
identity used by IR, Dataset, Query, Runtime, and Adapters.
"""

from __future__ import annotations

from typing import TypeAlias

import msgspec


class TableId(msgspec.Struct, frozen=True, tag="table", tag_field="id_kind"):
    name: str


class FieldId(msgspec.Struct, frozen=True, omit_defaults=True):
    owner: TableId
    name: str


class RelationId(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: str
    name: str


class ArtifactId(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: str
    owner: TableId
    name: str


EntityId: TypeAlias = TableId
ParticipantRef: TypeAlias = FieldId


__all__ = [
    "TableId",
    "EntityId",
    "FieldId",
    "RelationId",
    "ArtifactId",
    "ParticipantRef",
]
