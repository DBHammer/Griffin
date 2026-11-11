"""Relational IR schema."""

from __future__ import annotations

import enum

import msgspec

from griffin.ir.field import Field
from griffin.model_ids import FieldId, TableId


class Cardinality(enum.Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "N:1"
    MANY_TO_MANY = "N:N"


class ForeignKey(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    source_field: FieldId
    target_field: FieldId
    cardinality: Cardinality


class Table(msgspec.Struct, frozen=True, omit_defaults=True):
    id: TableId
    fields: tuple[Field, ...]
    primary_key: tuple[FieldId, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()


class RelationalSchema(msgspec.Struct, frozen=True, omit_defaults=True):
    tables: tuple[Table, ...]


__all__ = ["Cardinality", "ForeignKey", "Table", "RelationalSchema"]
