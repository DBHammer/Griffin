"""Statistics base + concrete kinds.

`FieldStatistics` is a tagged union of typed structs — one per data-type
family. Higher layers consume specific variants (e.g. the planner's vector
difficulty estimator reads `VectorStatistics`).
"""

from __future__ import annotations

from typing import Union

import msgspec

from griffin.model_ids import EntityId, FieldId


class NumericStatistics(msgspec.Struct, frozen=True, tag="numeric", tag_field="stat_kind"):
    field: FieldId
    count: int
    null_count: int
    min: float
    max: float
    mean: float
    stddev: float
    quantiles: tuple[tuple[float, float], ...] = ()  # (q, value) pairs


class TextStatistics(msgspec.Struct, frozen=True, tag="text", tag_field="stat_kind"):
    field: FieldId
    count: int
    null_count: int
    distinct_estimate: int
    avg_length: float


class CategoricalStatistics(msgspec.Struct, frozen=True, tag="categorical", tag_field="stat_kind"):
    field: FieldId
    count: int
    null_count: int
    distinct: int
    top_buckets: tuple[tuple[str, int], ...]  # (label, frequency)


class VectorStatistics(msgspec.Struct, frozen=True, tag="vector", tag_field="stat_kind"):
    field: FieldId
    count: int
    null_count: int
    dimension: int
    norm_mean: float
    norm_stddev: float


class ListStatistics(msgspec.Struct, frozen=True, tag="list", tag_field="stat_kind"):
    field: FieldId
    count: int
    null_count: int
    avg_length: float
    distinct_element_estimate: int


FieldStatistics = Union[
    NumericStatistics,
    TextStatistics,
    CategoricalStatistics,
    VectorStatistics,
    ListStatistics,
]


class EntityMetrics(msgspec.Struct, frozen=True, omit_defaults=True):
    owner: EntityId
    row_count: int
    field_statistics: tuple[FieldStatistics, ...]


__all__ = [
    "NumericStatistics",
    "TextStatistics",
    "CategoricalStatistics",
    "VectorStatistics",
    "ListStatistics",
    "FieldStatistics",
    "EntityMetrics",
]
