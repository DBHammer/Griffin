"""Relational AST — single-table SELECT with typed projection / ordering / CTE.

Single-table scope for the SELECT body itself. Multi-source fan-in (RRF)
goes through CteSelectQuery where each CTE is itself a single-table
SelectQuery, plus UnionAllSource for combining their outputs.

Bindings produce these AST trees; dialect renderers (visitor pattern)
consume them and emit dialect-specific text.
"""

from __future__ import annotations

from typing import Union

import msgspec

from griffin.ir.expr import AggregateExpr, Expr, FieldRef, NullsHandling, Predicate, RowIdRef, ScoredProjection, SortDir, StarProjection, WindowRowNumber
from griffin.model_ids import TableId


class TableSource(msgspec.Struct, frozen=True, tag="table", tag_field="source_kind"):
    """Single physical table. `table.name` matches a real on-disk table."""

    table: TableId
    alias: str


class CteSource(msgspec.Struct, frozen=True, tag="cte", tag_field="source_kind"):
    """Reference to a CTE name defined earlier in the same CteSelectQuery."""

    cte_name: str
    alias: str


class UnionAllSource(msgspec.Struct, frozen=True, tag="union_all", tag_field="source_kind"):
    """`(SELECT … FROM cte_0) UNION ALL (SELECT … FROM cte_1) …` as a
    derived table.

    Each branch is a SelectQuery; the resulting union is exposed as `alias`.
    """

    branches: tuple["SelectQuery", ...]
    alias: str


Source = Union[TableSource, CteSource, UnionAllSource]


class OrderBy(msgspec.Struct, frozen=True, omit_defaults=True):
    expr: Expr
    direction: SortDir | None = None
    nulls: NullsHandling = NullsHandling.DEFAULT


class Limit(msgspec.Struct, frozen=True, omit_defaults=True):
    n: int


# Allowed projection variants. WindowRowNumber is a SCORE-shaped expression
# that bindings wrap inside a ScoredProjection, so the Projection union
# stays small.
Projection = Union[StarProjection, FieldRef, RowIdRef, ScoredProjection, AggregateExpr]


class SelectQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    source: Source
    project: tuple[Projection, ...]
    where: Predicate | None = None
    group_by: tuple[Expr, ...] = ()
    order_by: tuple[OrderBy, ...] = ()
    limit: Limit | None = None


class CountQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    source: Source
    where: Predicate | None = None


class NamedSelect(msgspec.Struct, frozen=True, omit_defaults=True):
    """A named SelectQuery for use as a CTE."""

    name: str
    query: SelectQuery


class CteSelectQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    """WITH <ctes...> SELECT … — used by RRF and similar fan-in patterns."""

    ctes: tuple[NamedSelect, ...]
    final: SelectQuery


RelationalAst = Union[SelectQuery, CountQuery, CteSelectQuery]


# Re-export for binding convenience.
_ = WindowRowNumber  # noqa: F841 — referenced via expr.py


__all__ = [
    "TableSource",
    "CteSource",
    "UnionAllSource",
    "Source",
    "OrderBy",
    "Limit",
    "Projection",
    "SelectQuery",
    "CountQuery",
    "NamedSelect",
    "CteSelectQuery",
    "RelationalAst",
]
