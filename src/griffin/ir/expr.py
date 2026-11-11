"""Typed expression tree.

All expression nodes are msgspec.Struct subclasses with a discriminator tag.
Higher layers (dialect renderers, planners) match on these tags rather than
isinstance chains.
"""

from __future__ import annotations

import enum
from typing import Union

import msgspec

from griffin.model_ids import FieldId

from .data_type import VectorMetric


class BinOp(enum.Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    AND = "and"
    OR = "or"


class UnaryOpKind(enum.Enum):
    NOT = "not"


class SortDir(enum.Enum):
    ASC = "asc"
    DESC = "desc"


class NullsHandling(enum.Enum):
    DEFAULT = "default"
    FIRST = "first"
    LAST = "last"


class AggregationOp(enum.Enum):
    COUNT = "count"
    COUNT_STAR = "count_star"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class LiteralKind(enum.Enum):
    INT = "int"
    DOUBLE = "double"
    BOOL = "bool"
    TEXT = "text"
    VECTOR = "vector"


class FieldRef(msgspec.Struct, frozen=True, tag="field_ref", tag_field="op"):
    """Reference to a field bound to a specific scope variable.

    `variable` is the binding-time SQL scope alias. It is required; render
    time does not reverse-infer aliases.
    """

    field: FieldId
    variable: str


class HoleRef(msgspec.Struct, frozen=True, tag="hole_ref", tag_field="op"):
    """Reference to a query-generation parameter hole.

    HoleRef is valid in template expressions only. Materialized render IR must
    replace every HoleRef with a concrete Literal or dialect-specific parameter
    node before reaching renderers.
    """

    hole_id: str


class RowIdRef(msgspec.Struct, frozen=True, tag="row_id_ref", tag_field="op"):
    """Reference to the system stable row identity column.

    `_rowid` is a physical data-file / runtime column, but it is not a user
    FieldId and must not be confused with a table primary key.
    """

    variable: str


class Literal(msgspec.Struct, frozen=True, tag="literal", tag_field="op"):
    """Typed literal value.

    Stored as a JSON-encoded payload + a kind tag so renderers can format it
    for their dialect (e.g. '[…]'::vector cast in PG; text → quoted; bool →
    TRUE/FALSE).
    """

    kind: LiteralKind
    value_json: str


class BinaryOp(msgspec.Struct, frozen=True, tag="binary", tag_field="op"):
    operator: BinOp
    left: "Expr"
    right: "Expr"


class UnaryOp(msgspec.Struct, frozen=True, tag="unary", tag_field="op"):
    operator: UnaryOpKind
    expr: "Expr"


class FunctionCall(msgspec.Struct, frozen=True, tag="function", tag_field="op"):
    name: str
    args: tuple["Expr", ...]


class VectorDistanceTemplate(msgspec.Struct, frozen=True, tag="vector_distance_template", tag_field="op"):
    """Template-stage vector distance with a query-vector hole.

    This is the formal query-generation expression. It is materialized to
    `VectorDistance` only after parameter binding replaces the hole.
    """

    field: FieldRef
    query_vector: HoleRef


class RangePredicateTemplate(msgspec.Struct, frozen=True, tag="range_predicate_template", tag_field="op"):
    """Template-stage numeric range predicate with a range-value hole."""

    field: FieldRef
    range_value: HoleRef


class SetMembership(msgspec.Struct, frozen=True, tag="set_membership", tag_field="op"):
    expr: "Expr"
    values: tuple["Expr", ...]


class IsNull(msgspec.Struct, frozen=True, tag="is_null", tag_field="op"):
    expr: "Expr"


class VectorDistance(msgspec.Struct, frozen=True, tag="vector_distance", tag_field="op"):
    """Distance between a stored vector field and a query vector literal.

    `variable` is the SQL alias bound by the surrounding query. PostgreSQL
    renders the metric with the corresponding pgvector operator.
    """

    field: FieldId
    variable: str
    query_vector_literal: "Literal"  # kind == VECTOR
    metric: VectorMetric


class AggregateExpr(msgspec.Struct, frozen=True, tag="aggregate", tag_field="op"):
    """Aggregation function over an expression.

    For COUNT(*), use AggregationOp.COUNT_STAR with arg=None.
    """

    operator: AggregationOp
    arg: "Expr | None" = None


class StarProjection(msgspec.Struct, frozen=True, tag="star", tag_field="op"):
    """SELECT * (or RETURN *)."""

    pass


class ScoredProjection(msgspec.Struct, frozen=True, tag="scored", tag_field="op"):
    """Render an expression with an alias (typically a score column)."""

    expr: "Expr"
    alias: str


class WindowRowNumber(msgspec.Struct, frozen=True, tag="window_row_number", tag_field="op"):
    """ROW_NUMBER() OVER (ORDER BY <expr> <dir>) — windowing over the
    surrounding query. Used by ranked fusion / top-k style plans.

    Currently only ROW_NUMBER is supported (RANK / DENSE_RANK can be added
    later as additional WindowFunction variants).
    """

    order_by_expr: "Expr"
    order_by_direction: SortDir = SortDir.ASC


Expr = Union[
    FieldRef,
    RowIdRef,
    HoleRef,
    Literal,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    VectorDistanceTemplate,
    RangePredicateTemplate,
    SetMembership,
    IsNull,
    VectorDistance,
    AggregateExpr,
    StarProjection,
    ScoredProjection,
    WindowRowNumber,
]


# Marker type used by AST nodes that require a boolean expression.
Predicate = Expr


__all__ = [
    "BinOp",
    "UnaryOpKind",
    "SortDir",
    "NullsHandling",
    "AggregationOp",
    "LiteralKind",
    "FieldRef",
    "HoleRef",
    "RowIdRef",
    "Literal",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "VectorDistanceTemplate",
    "RangePredicateTemplate",
    "SetMembership",
    "IsNull",
    "VectorDistance",
    "AggregateExpr",
    "StarProjection",
    "ScoredProjection",
    "WindowRowNumber",
    "Expr",
    "Predicate",
]
