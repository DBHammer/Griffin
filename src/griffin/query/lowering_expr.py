"""Lower current public flat predicate specs into the formal Expr union.

Public query config exposes flat filter slots. This module is the formal
lowering boundary from that config surface into recursive Expr.
"""

from __future__ import annotations

from griffin.infra.errors import ContractViolation
from griffin.ir.expr import (
    AggregateExpr,
    BinaryOp,
    BinOp,
    Expr,
    FieldRef,
    HoleRef,
    IsNull,
    RangePredicateTemplate,
    SetMembership,
    UnaryOp,
    VectorDistance,
    VectorDistanceTemplate,
)
from griffin.query.model import (
    HoleConstraints,
    HoleDecl,
    HoleId,
    HoleSemanticRole,
    HoleUsageSite,
    HoleValueKind,
    ScopeId,
)
from griffin.query.shape import (
    FilterLogic,
    FilterSlotKind,
    FilterSlotSpec,
    PredicateOperator,
)


def flat_filter_slot_to_hole(
    slot: FilterSlotSpec,
    scope_id: ScopeId,
    index: int,
) -> HoleDecl:
    value_kind = _hole_value_kind_for(slot)
    role = HoleSemanticRole.PREDICATE_VALUE
    hole_id = _filter_hole_id(scope_id, index)
    return HoleDecl(
        hole_id=hole_id,
        owner_scope=scope_id,
        value_kind=value_kind,
        semantic_role=role,
        constraints=HoleConstraints(max_cardinality=slot.max_in_values),
        usage_sites=(
            HoleUsageSite(
                node_id="",
                expr_path=(index,),
                role=role,
            ),
        ),
    )


def flat_predicate_to_expr(
    filters: tuple[FilterSlotSpec, ...],
    logic: FilterLogic,
    holes: tuple[HoleDecl, ...],
    scope_id: ScopeId,
    logics: tuple[FilterLogic, ...] | None = None,
) -> Expr:
    if len(filters) != len(holes):
        raise ContractViolation("flat predicate lowering requires one hole per filter slot")
    children = tuple(
        _slot_predicate_to_expr(slot, holes[index].hole_id, scope_id)
        for index, slot in enumerate(filters)
    )
    if not children:
        raise ContractViolation("flat predicate lowering requires at least one filter")
    if len(children) == 1:
        return children[0]
    # logics 为 connector 序列（长度 = len(children)-1），每个连接处独立 AND/OR；
    # 未提供时退化为单一 logic（向后兼容）。
    if logics is None:
        logics = (logic,) * (len(children) - 1)
    expr = children[0]
    for child, op in zip(children[1:], logics, strict=True):
        operator = BinOp.AND if op is FilterLogic.AND else BinOp.OR
        expr = BinaryOp(operator=operator, left=expr, right=child)
    return expr


def ensure_no_holes_for_render(expr: Expr) -> None:
    holes = tuple(_iter_holes(expr))
    if holes:
        raise ContractViolation(
            "materialized render expression still contains holes: "
            f"{', '.join(hole.hole_id for hole in holes)}"
        )


def _slot_predicate_to_expr(
    slot: FilterSlotSpec,
    hole_id: HoleId,
    scope_id: ScopeId,
) -> Expr:
    field = FieldRef(field=slot.field, variable=scope_id)
    hole = HoleRef(hole_id=hole_id)
    operator = slot.operator
    if operator is None:
        if slot.kind is FilterSlotKind.SCALAR_RANGE:
            operator = PredicateOperator.BETWEEN
        elif slot.kind is FilterSlotKind.SCALAR_BUCKET:
            operator = PredicateOperator.EQ
        else:
            raise ContractViolation(f"filter slot {slot.field.name} requires an operator")
    if operator is PredicateOperator.IN:
        return SetMembership(expr=field, values=(hole,))
    if operator is PredicateOperator.BETWEEN:
        return RangePredicateTemplate(field=field, range_value=hole)
    return BinaryOp(
        operator=_bin_op_for(operator),
        left=field,
        right=hole,
    )


def _bin_op_for(operator: PredicateOperator) -> BinOp:
    if operator is PredicateOperator.EQ:
        return BinOp.EQ
    if operator is PredicateOperator.LT:
        return BinOp.LT
    if operator is PredicateOperator.LE:
        return BinOp.LE
    if operator is PredicateOperator.GT:
        return BinOp.GT
    if operator is PredicateOperator.GE:
        return BinOp.GE
    raise ContractViolation(f"predicate operator {operator.value} is not a binary comparison")


def _hole_value_kind_for(slot: FilterSlotSpec) -> HoleValueKind:
    if slot.operator is PredicateOperator.IN:
        return HoleValueKind.IN_SET
    if slot.operator is PredicateOperator.BETWEEN:
        return HoleValueKind.RANGE_PAIR
    return HoleValueKind.SCALAR_LITERAL


def _filter_hole_id(scope_id: ScopeId, index: int) -> HoleId:
    return f"{scope_id}:filter[{index}]"


def _iter_holes(expr: Expr) -> tuple[HoleRef, ...]:
    if isinstance(expr, HoleRef):
        return (expr,)
    if isinstance(expr, BinaryOp):
        return _iter_holes(expr.left) + _iter_holes(expr.right)
    if isinstance(expr, UnaryOp):
        return _iter_holes(expr.expr)
    if isinstance(expr, VectorDistanceTemplate):
        return _iter_holes(expr.field) + _iter_holes(expr.query_vector)
    if isinstance(expr, RangePredicateTemplate):
        return _iter_holes(expr.field) + _iter_holes(expr.range_value)
    if isinstance(expr, SetMembership):
        holes = _iter_holes(expr.expr)
        for value in expr.values:
            holes += _iter_holes(value)
        return holes
    if isinstance(expr, IsNull):
        return _iter_holes(expr.expr)
    if isinstance(expr, VectorDistance):
        return _iter_holes(expr.query_vector_literal)
    if isinstance(expr, AggregateExpr):
        return ()
    return ()


__all__ = [
    "ensure_no_holes_for_render",
    "flat_filter_slot_to_hole",
    "flat_predicate_to_expr",
]
