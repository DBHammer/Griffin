"""Compute statistics from materialized column data."""

from __future__ import annotations

import math
from typing import assert_never

import numpy as np

from griffin.data.entity_builders import ColumnMapping, ColumnValues
from griffin.infra.errors import ContractViolation
from griffin.dataset.statistics.base import (
    CategoricalStatistics,
    EntityMetrics,
    FieldStatistics,
    ListStatistics,
    NumericStatistics,
    TextStatistics,
    VectorStatistics,
)
from griffin.ir.data_type import DoubleType, IntType, ListType, TextType, VectorType
from griffin.ir.field import Field, FieldRole
from griffin.model_ids import EntityId
from griffin.vector_hardness_core import vector_column_matrix


def _numeric_stats(field: Field, values: list[object]) -> NumericStatistics:
    nums = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return NumericStatistics(field=field.id, count=0, null_count=len(values), min=0.0, max=0.0, mean=0.0, stddev=0.0)
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / n
    ordered = sorted(nums)
    quantile_points = tuple(
        (q, _quantile(ordered, q))
        for q in (0.1, 0.25, 0.5, 0.75, 0.9)
    )
    return NumericStatistics(
        field=field.id,
        count=n,
        null_count=len(values) - n,
        min=min(nums),
        max=max(nums),
        mean=mean,
        stddev=math.sqrt(var),
        quantiles=quantile_points,
    )


def _vector_stats(
    field: Field,
    values: ColumnValues,
    dimension: int,
) -> VectorStatistics:
    matrix = vector_column_matrix(values, field, dimension=dimension)
    norms = np.sqrt(np.sum(matrix * matrix, axis=1))
    null_count = 0
    n = norms.shape[0]
    if n == 0:
        return VectorStatistics(
            field=field.id,
            count=0,
            null_count=null_count,
            dimension=dimension,
            norm_mean=0.0,
            norm_stddev=0.0,
        )
    mean = float(np.mean(norms))
    var = float(np.var(norms))
    return VectorStatistics(
        field=field.id,
        count=n,
        null_count=null_count,
        dimension=dimension,
        norm_mean=mean,
        norm_stddev=math.sqrt(var),
    )


def _text_stats(field: Field, values: list[object]) -> TextStatistics:
    strs = [str(v) for v in values if v is not None]
    n = len(strs)
    if not strs:
        return TextStatistics(field=field.id, count=0, null_count=len(values), distinct_estimate=0, avg_length=0.0)
    distinct = len(set(strs))
    avg_len = sum(len(s) for s in strs) / n
    return TextStatistics(field=field.id, count=n, null_count=len(values) - n, distinct_estimate=distinct, avg_length=avg_len)


def _categorical_stats(field: Field, values: list[object]) -> CategoricalStatistics:
    from collections import Counter

    non_null = [v for v in values if v is not None]
    counter = Counter(str(v) for v in non_null)
    top = counter.most_common(10)
    return CategoricalStatistics(
        field=field.id,
        count=len(non_null),
        null_count=len(values) - len(non_null),
        distinct=len(counter),
        top_buckets=tuple(top),
    )


def _list_stats(field: Field, values: list[object]) -> ListStatistics:
    # Coerce to list-shaped non-null entries; ignore non-list payloads.
    non_null: list[list[object]] = [v for v in values if isinstance(v, list)]
    n = len(non_null)
    if not n:
        return ListStatistics(field=field.id, count=0, null_count=len(values), avg_length=0.0, distinct_element_estimate=0)
    avg_len = sum(len(v) for v in non_null) / n
    flat = {x for v in non_null for x in v}
    return ListStatistics(
        field=field.id,
        count=n,
        null_count=len(values) - n,
        avg_length=avg_len,
        distinct_element_estimate=len(flat),
    )


def field_statistics(
    field: Field,
    values: ColumnValues,
) -> FieldStatistics:
    scalar_values = _scalar_values(field, values) if field.role is not FieldRole.VECTOR else None
    if field.role == FieldRole.CATEGORICAL:
        if scalar_values is None:
            raise ContractViolation(f"categorical field {field.id} must be materialized as scalar values")
        return _categorical_stats(field, scalar_values)
    dt = field.type
    match dt:
        case IntType() | DoubleType():
            if scalar_values is None:
                raise ContractViolation(f"numeric field {field.id} must be materialized as scalar values")
            return _numeric_stats(field, scalar_values)
        case TextType():
            if scalar_values is None:
                raise ContractViolation(f"text field {field.id} must be materialized as scalar values")
            return _text_stats(field, scalar_values)
        case VectorType():
            return _vector_stats(field, values, dt.dimension)
        case ListType():
            if scalar_values is None:
                raise ContractViolation(f"list field {field.id} must be materialized as scalar values")
            return _list_stats(field, scalar_values)
        case _:
            assert_never(dt)


def _scalar_values(field: Field, values: ColumnValues) -> list[object]:
    if isinstance(values, np.ndarray):
        raise ContractViolation(f"field {field.id} must be materialized as scalar values")
    return values


def entity_metrics(
    owner: EntityId,
    fields: tuple[Field, ...],
    columns: ColumnMapping,
) -> EntityMetrics:
    row_count = next(iter(len(v) for v in columns.values()), 0)
    stats = tuple(
        field_statistics(f, columns[f.id.name])
        for f in fields
    )
    return EntityMetrics(owner=owner, row_count=row_count, field_statistics=stats)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return values[0]
    if q >= 1.0:
        return values[-1]
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


__all__ = ["field_statistics", "entity_metrics"]
