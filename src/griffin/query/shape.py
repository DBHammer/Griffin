"""Closed shape definitions for query generation."""

from __future__ import annotations

import enum
from typing import Union

import msgspec

from griffin.model_ids import FieldId, TableId
from griffin.query.model import DEFAULT_MAX_IN_VALUES


class FilterSlotKind(enum.Enum):
    SCALAR_RANGE = "scalar_range"
    SCALAR_BUCKET = "scalar_bucket"


class PredicateOperator(enum.Enum):
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    BETWEEN = "between"
    EQ = "eq"
    IN = "in"


class FilterLogic(enum.Enum):
    AND = "and"
    OR = "or"


class FilterSlotSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    field: FieldId
    kind: FilterSlotKind
    operator: PredicateOperator | None = None
    max_in_values: int | None = None

    def __post_init__(self) -> None:
        if self.max_in_values is None:
            return
        if self.operator is not PredicateOperator.IN:
            raise ValueError("max_in_values requires an IN filter slot")
        if self.max_in_values < 1:
            raise ValueError("max_in_values must be positive")


class PredicateCountRange(msgspec.Struct, frozen=True, omit_defaults=True):
    min: int
    max: int


class PredicateAtomSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    field: FieldId
    operator: PredicateOperator


class PredicateSearchSpace(msgspec.Struct, frozen=True, omit_defaults=True):
    predicate_count: PredicateCountRange
    operators: tuple[PredicateOperator, ...]
    predicate_fields: tuple[str, ...] = ()
    atoms: tuple[PredicateAtomSpec, ...] = ()
    max_in_values: int = DEFAULT_MAX_IN_VALUES

    def __post_init__(self) -> None:
        if self.max_in_values < 1:
            raise ValueError("predicate search max_in_values must be positive")


class AggregationOp(enum.Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class SelectShape(msgspec.Struct, frozen=True, tag="select", tag_field="shape_kind"):
    primary_entity: TableId
    filters: tuple[FilterSlotSpec, ...] = ()
    filter_logic: FilterLogic = FilterLogic.AND
    project: tuple[FieldId, ...] = ()
    predicate_search: PredicateSearchSpace | None = None


class AggregateShape(msgspec.Struct, frozen=True, tag="aggregate", tag_field="shape_kind"):
    primary_entity: TableId
    group_by: tuple[FieldId, ...]
    aggregations: tuple[tuple[AggregationOp, FieldId], ...]
    filters: tuple[FilterSlotSpec, ...] = ()
    filter_logic: FilterLogic = FilterLogic.AND
    predicate_search: PredicateSearchSpace | None = None


class VectorRetrieveShape(msgspec.Struct, frozen=True, tag="vector_retrieve", tag_field="shape_kind"):
    primary_entity: TableId
    target_field: FieldId
    filters: tuple[FilterSlotSpec, ...] = ()
    filter_logic: FilterLogic = FilterLogic.AND
    predicate_search: PredicateSearchSpace | None = None


TypedShape = Union[
    SelectShape,
    AggregateShape,
    VectorRetrieveShape,
]


__all__ = [
    "AggregateShape",
    "AggregationOp",
    "FilterLogic",
    "FilterSlotKind",
    "FilterSlotSpec",
    "PredicateCountRange",
    "PredicateAtomSpec",
    "PredicateOperator",
    "PredicateSearchSpace",
    "SelectShape",
    "TypedShape",
    "VectorRetrieveShape",
]
