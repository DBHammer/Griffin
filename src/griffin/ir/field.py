"""Field metadata."""

from __future__ import annotations

import enum

import msgspec

from griffin.model_ids import FieldId

from .data_type import DataType


class FieldRole(enum.Enum):
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    PAYLOAD = "payload"
    VECTOR = "vector"
    CATEGORICAL = "categorical"


class Field(msgspec.Struct, frozen=True, omit_defaults=True):
    id: FieldId
    type: DataType
    role: FieldRole
    nullable: bool = False


__all__ = [
    "FieldRole",
    "Field",
]
