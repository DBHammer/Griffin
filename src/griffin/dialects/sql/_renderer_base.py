"""Common SQL renderer base class.

Visitor pattern: every IR node has a corresponding `_visit_*` method on
this base, with closed `match` dispatch on the union types. PostgreSQL
overrides the vector and literal rendering hooks:

    - `_render_vector_distance` (operator differs by vector-capable dialect)
    - `_format_literal` (vector cast syntax differs)
    - `_param_style` (PostgreSQL: %s)

The base class handles the *structural* parts of SQL — SELECT/FROM/WHERE/
GROUP BY/ORDER BY/LIMIT clause assembly, expression precedence, etc.
"""

from __future__ import annotations

import json
from typing import assert_never

from griffin.dialects._param_builder import ParamBuilder, ParamStyle
from griffin.dialects.base import RenderedQuery
from griffin.dialects.capabilities import DialectCapabilities
from griffin.infra.errors import ContractViolation
from griffin.dataset.row_identity import DEFAULT_ROW_ID_COLUMN
from griffin.ir.expr import (
    AggregateExpr,
    AggregationOp,
    BinaryOp,
    BinOp,
    Expr,
    FieldRef,
    FunctionCall,
    HoleRef,
    IsNull,
    Literal,
    LiteralKind,
    NullsHandling,
    RangePredicateTemplate,
    RowIdRef,
    ScoredProjection,
    SetMembership,
    SortDir,
    StarProjection,
    UnaryOp,
    UnaryOpKind,
    VectorDistance,
    VectorDistanceTemplate,
    WindowRowNumber,
)
from griffin.ir.relational.ast import CountQuery, CteSelectQuery, CteSource, Limit, NamedSelect, OrderBy, Projection, RelationalAst, SelectQuery, Source, TableSource, UnionAllSource


_BIN_OP_SQL = {
    BinOp.EQ: "=",
    BinOp.NE: "<>",
    BinOp.LT: "<",
    BinOp.LE: "<=",
    BinOp.GT: ">",
    BinOp.GE: ">=",
    BinOp.ADD: "+",
    BinOp.SUB: "-",
    BinOp.MUL: "*",
    BinOp.DIV: "/",
    BinOp.AND: "AND",
    BinOp.OR: "OR",
}

_AGG_SQL = {
    AggregationOp.COUNT: "COUNT",
    AggregationOp.SUM: "SUM",
    AggregationOp.AVG: "AVG",
    AggregationOp.MIN: "MIN",
    AggregationOp.MAX: "MAX",
}


class SqlRendererBase:
    """Closed visitor over RelationalAst.

    Subclasses must set:
        - `name` (class attr): dialect name
        - `capabilities` (class attr): DialectCapabilities

    Subclasses may override the dialect-specific helpers:
        - `_param_style()` — default PostgreSQL positional parameters
        - `_render_vector_distance(node, params)` — required
        - `_format_vector_literal(json_str)` — default: array literal syntax
    """

    name: str = ""
    capabilities: DialectCapabilities = None  # type: ignore[assignment]

    # ─── Public API ───
    def render(self, ast: RelationalAst) -> RenderedQuery:
        params = ParamBuilder(self._param_style())
        text = self._visit_relational(ast, params)
        return RenderedQuery(text=text, parameters_json=params.freeze())

    # ─── Subclass hooks ───
    def _param_style(self) -> ParamStyle:
        return ParamStyle.PERCENT_S

    def _render_vector_distance(self, node: VectorDistance, params: ParamBuilder) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement _render_vector_distance")

    def _format_vector_literal(self, value_json: str, params: ParamBuilder) -> str:
        """Render a VECTOR-kind Literal. Vector-capable dialects override."""
        return params.add(json.loads(value_json))

    # ─── Top-level dispatch ───
    def _visit_relational(self, ast: RelationalAst, params: ParamBuilder) -> str:
        match ast:
            case SelectQuery():
                return self._visit_select(ast, params)
            case CountQuery():
                return self._visit_count(ast, params)
            case CteSelectQuery():
                return self._visit_cte_select(ast, params)
            case _:
                assert_never(ast)

    def _visit_select(self, q: SelectQuery, params: ParamBuilder) -> str:
        parts = ["SELECT", self._render_projections(q.project, params)]
        parts.append(f"FROM {self._render_source(q.source, params)}")
        if q.where is not None:
            parts.append(f"WHERE {self._visit_expr(q.where, params)}")
        if q.group_by:
            parts.append("GROUP BY " + ", ".join(self._visit_expr(e, params) for e in q.group_by))
        if q.order_by:
            # Reuse projection aliases in ORDER BY when the expressions are the same
            # object — avoids re-emitting large literals (vector queries) twice.
            alias_for_expr = {
                id(proj.expr): proj.alias
                for proj in q.project
                if isinstance(proj, ScoredProjection)
            }
            parts.append(self._render_order_by(q.order_by, params, alias_for_expr=alias_for_expr))
        if q.limit is not None:
            parts.append(self._render_limit(q.limit))
        return " ".join(parts)

    def _render_source(self, source: Source, params: ParamBuilder) -> str:
        match source:
            case TableSource():
                return f"{source.table.name} AS {source.alias}"
            case CteSource():
                return f"{source.cte_name} AS {source.alias}"
            case UnionAllSource():
                inner = " UNION ALL ".join(
                    f"({self._visit_select(b, params)})" for b in source.branches
                )
                return f"({inner}) AS {source.alias}"
            case _:
                assert_never(source)

    def _visit_count(self, q: CountQuery, params: ParamBuilder) -> str:
        text = f"SELECT COUNT(*) AS c FROM {self._render_source(q.source, params)}"
        if q.where is not None:
            text += f" WHERE {self._visit_expr(q.where, params)}"
        return text

    def _visit_cte_select(self, q: CteSelectQuery, params: ParamBuilder) -> str:
        cte_parts = [self._render_named_select(c, params) for c in q.ctes]
        with_clause = "WITH " + ", ".join(cte_parts)
        final_sql = self._visit_select(q.final, params)
        return f"{with_clause} {final_sql}"

    def _render_named_select(self, named: NamedSelect, params: ParamBuilder) -> str:
        inner = self._visit_select(named.query, params)
        return f"{named.name} AS ({inner})"

    # ─── Projections / clauses ───
    def _render_projections(
        self, projections: tuple[Projection, ...], params: ParamBuilder
    ) -> str:
        if not projections:
            return "*"
        return ", ".join(self._render_projection(p, params) for p in projections)

    def _render_projection(self, proj: Projection, params: ParamBuilder) -> str:
        match proj:
            case StarProjection():
                return "*"
            case FieldRef():
                return self._render_field_ref(proj)
            case RowIdRef():
                return self._render_rowid_ref(proj)
            case ScoredProjection():
                return f"{self._visit_expr(proj.expr, params)} AS {proj.alias}"
            case AggregateExpr():
                return self._render_aggregate(proj, params)
            case _:
                assert_never(proj)

    def _render_aggregate(self, agg: AggregateExpr, params: ParamBuilder) -> str:
        if agg.operator == AggregationOp.COUNT_STAR:
            return "COUNT(*)"
        if agg.arg is None:
            raise ContractViolation(f"non-COUNT_STAR aggregate {agg.operator} requires arg")
        op_sql = _AGG_SQL[agg.operator]
        return f"{op_sql}({self._visit_expr(agg.arg, params)})"

    def _render_order_by(
        self,
        items: tuple[OrderBy, ...],
        params: ParamBuilder,
        *,
        alias_for_expr: dict[int, str] | None = None,
    ) -> str:
        rendered = []
        for item in items:
            alias = alias_for_expr.get(id(item.expr)) if alias_for_expr else None
            base = alias if alias is not None else self._visit_expr(item.expr, params)
            if item.direction is None:
                entry = base
            else:
                direction = "ASC" if item.direction == SortDir.ASC else "DESC"
                entry = f"{base} {direction}"
            if item.nulls == NullsHandling.FIRST:
                entry += " NULLS FIRST"
            elif item.nulls == NullsHandling.LAST:
                entry += " NULLS LAST"
            rendered.append(entry)
        return "ORDER BY " + ", ".join(rendered)

    def _render_limit(self, limit: Limit) -> str:
        return f"LIMIT {limit.n}"

    # ─── Expression visitor (closed) ───
    def _visit_expr(self, e: Expr, params: ParamBuilder) -> str:
        match e:
            case FieldRef():
                return self._render_field_ref(e)
            case RowIdRef():
                return self._render_rowid_ref(e)
            case Literal():
                return self._render_literal(e, params)
            case BinaryOp():
                return self._render_binary(e, params)
            case FunctionCall():
                return self._render_function(e, params)
            case SetMembership():
                return self._render_set_membership(e, params)
            case IsNull():
                return f"({self._visit_expr(e.expr, params)} IS NULL)"
            case UnaryOp():
                if e.operator is UnaryOpKind.NOT:
                    return f"(NOT {self._visit_expr(e.expr, params)})"
                assert_never(e.operator)
            case HoleRef():
                raise ContractViolation("SQL renderer received unmaterialized HoleRef")
            case VectorDistanceTemplate() | RangePredicateTemplate():
                raise ContractViolation("SQL renderer received unmaterialized template expression")
            case VectorDistance():
                return self._render_vector_distance(e, params)
            case AggregateExpr():
                return self._render_aggregate(e, params)
            case StarProjection():
                return "*"
            case ScoredProjection():
                return f"{self._visit_expr(e.expr, params)} AS {e.alias}"
            case WindowRowNumber():
                inner = self._visit_expr(e.order_by_expr, params)
                direction = "ASC" if e.order_by_direction == SortDir.ASC else "DESC"
                return f"ROW_NUMBER() OVER (ORDER BY {inner} {direction})"
            case _:
                assert_never(e)

    def _render_field_ref(self, ref: FieldRef) -> str:
        # Variable is mandatory — bindings carry the scope.
        return f"{ref.variable}.{ref.field.name}"

    def _render_rowid_ref(self, ref: RowIdRef) -> str:
        return f"{ref.variable}.{DEFAULT_ROW_ID_COLUMN}"

    def _render_literal(self, lit: Literal, params: ParamBuilder) -> str:
        match lit.kind:
            case LiteralKind.INT | LiteralKind.DOUBLE:
                return params.add(json.loads(lit.value_json))
            case LiteralKind.BOOL:
                return params.add(json.loads(lit.value_json))
            case LiteralKind.TEXT:
                return params.add(json.loads(lit.value_json))
            case LiteralKind.VECTOR:
                return self._format_vector_literal(lit.value_json, params)
            case _:
                assert_never(lit.kind)

    def _render_binary(self, op: BinaryOp, params: ParamBuilder) -> str:
        op_sql = _BIN_OP_SQL[op.operator]
        left = self._visit_expr(op.left, params)
        right = self._visit_expr(op.right, params)
        return f"({left} {op_sql} {right})"

    def _render_function(self, fc: FunctionCall, params: ParamBuilder) -> str:
        del params
        raise ContractViolation(f"SQL renderer does not support generic function {fc.name!r}")

    def _render_set_membership(self, membership: SetMembership, params: ParamBuilder) -> str:
        if not membership.values:
            raise ContractViolation("IN predicate requires at least one value")
        expr = self._visit_expr(membership.expr, params)
        values = ", ".join(self._visit_expr(value, params) for value in membership.values)
        return f"({expr} IN ({values}))"

__all__ = ["SqlRendererBase"]
