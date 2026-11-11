"""Typed data types and per-type metadata (e.g. VectorMetric).

Tagged-union encoding via msgspec's `tag_field` discriminator. `logical_name`
is the cross-dialect symbolic name used in dataset manifests.
"""

from __future__ import annotations

import enum
from typing import Union

import msgspec


class VectorMetric(enum.Enum):
    L2 = "l2"
    COSINE = "cosine"


class IntType(msgspec.Struct, frozen=True, tag="int", tag_field="kind"):
    bits: int = 64
    signed: bool = True

    @property
    def logical_name(self) -> str:
        return f"int{self.bits}"


class DoubleType(msgspec.Struct, frozen=True, tag="double", tag_field="kind"):
    bits: int = 64

    @property
    def logical_name(self) -> str:
        return f"float{self.bits}"


class TextType(msgspec.Struct, frozen=True, tag="text", tag_field="kind"):
    max_length: int | None = None

    @property
    def logical_name(self) -> str:
        return "text"


class VectorType(msgspec.Struct, frozen=True, tag="vector", tag_field="kind"):
    dimension: int
    metric: VectorMetric

    @property
    def logical_name(self) -> str:
        return f"vector[{self.dimension}:{self.metric.value}]"


class ListType(msgspec.Struct, frozen=True, tag="list", tag_field="kind"):
    element: "DataType"

    @property
    def logical_name(self) -> str:
        return f"list<{self.element.logical_name}>"


DataType = Union[IntType, DoubleType, TextType, VectorType, ListType]


__all__ = [
    "VectorMetric",
    "IntType",
    "DoubleType",
    "TextType",
    "VectorType",
    "ListType",
    "DataType",
]
