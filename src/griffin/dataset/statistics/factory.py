"""Closed-dispatch helpers for statistics typing."""

from __future__ import annotations

from typing import assert_never

from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import DoubleType, IntType, ListType, TextType, VectorType
from griffin.ir.field import Field, FieldRole

from .base import (
    CategoricalStatistics,
    FieldStatistics,
    ListStatistics,
    NumericStatistics,
    TextStatistics,
    VectorStatistics,
)


def empty_statistics_for(field: Field) -> FieldStatistics:
    """Return a zero-filled FieldStatistics whose tag matches `field.type`.

    Used as a placeholder by the writer when no payload has been profiled yet.
    """
    if field.role == FieldRole.CATEGORICAL:
        return CategoricalStatistics(field=field.id, count=0, null_count=0, distinct=0, top_buckets=())

    dt = field.type
    match dt:
        case IntType() | DoubleType():
            return NumericStatistics(
                field=field.id,
                count=0,
                null_count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                stddev=0.0,
            )
        case TextType():
            return TextStatistics(
                field=field.id, count=0, null_count=0, distinct_estimate=0, avg_length=0.0
            )
        case VectorType():
            return VectorStatistics(
                field=field.id,
                count=0,
                null_count=0,
                dimension=dt.dimension,
                norm_mean=0.0,
                norm_stddev=0.0,
            )
        case ListType():
            return ListStatistics(
                field=field.id, count=0, null_count=0, avg_length=0.0, distinct_element_estimate=0
            )
        case _:
            assert_never(dt)


def validate_statistics_kind(field: Field, stats: FieldStatistics) -> None:
    """Tripwire: a FieldStatistics variant must align with the field's data type."""
    dt = field.type
    match (dt, stats):
        case (IntType(), CategoricalStatistics()) if field.role == FieldRole.CATEGORICAL:
            return
        case (IntType() | DoubleType(), NumericStatistics()):
            return
        case (TextType(), TextStatistics()):
            return
        case (TextType(), CategoricalStatistics()) if field.role == FieldRole.CATEGORICAL:
            return
        case (VectorType(), VectorStatistics()):
            return
        case (ListType(), ListStatistics()):
            return
        case _:
            raise ContractViolation(
                f"statistics kind {type(stats).__name__} does not match field type {type(dt).__name__}"
            )


__all__ = ["empty_statistics_for", "validate_statistics_kind"]
