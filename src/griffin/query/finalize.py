"""Artifact projection for accepted query candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import TypeAlias

import msgspec

from griffin.dialects.base import RenderedQuery
from griffin.dialects.sql.postgresql import PostgreSQLRenderer
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric, VectorType
from griffin.ir.expr import (
    BinaryOp,
    BinOp,
    Expr,
    FieldRef,
    HoleRef,
    IsNull,
    Literal,
    LiteralKind,
    RangePredicateTemplate,
    RowIdRef,
    ScoredProjection,
    SetMembership,
    SortDir,
    StarProjection,
    UnaryOp,
    VectorDistance,
    VectorDistanceTemplate,
)
from griffin.ir.relational.ast import (
    CountQuery,
    Limit,
    OrderBy,
    RelationalAst,
    SelectQuery,
    TableSource,
)
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.admit import AcceptedBinding
from griffin.query.evaluate import FilteredVectorSearchScope
from griffin.query.ground_truth import (
    FilteredTopKGroundTruthResult,
    GroundTruthResult,
    NoPublicGroundTruthResult,
)
from griffin.query.spec import (
    AcceptedPredicatePlan,
    EntityPredicateScope,
    GroundTruthAnswerKind,
    PredicateAtomRecord,
    PredicateAtomScope,
    PredicateAtomScopeKind,
    PredicateBindingSummary,
    PredicatePlanLogic,
)
from griffin.query.model import (
    AggregateNode,
    LimitNode,
    MaterializedQuery,
    ParameterBinding,
    RankNode,
    ScanNode,
    SelectNode,
    TemplateNode,
)
from griffin.query.model import (
    BucketSetValue,
    BucketValue,
    NumericComparisonValue,
    NumericRangeValue,
    ScalarValue,
    TextValue,
    VectorValue,
)


class FilteredTopKGroundTruthRecord(msgspec.Struct, frozen=True, omit_defaults=True):
    answer_kind: GroundTruthAnswerKind
    metric: str
    selection_cover_count: int
    selection_cover_fingerprint: str
    answers: tuple[dict[str, int], ...]
    answer_scope: Mapping[str, str | int] = msgspec.field(default_factory=dict)


GroundTruthArtifactRecord: TypeAlias = FilteredTopKGroundTruthRecord


class FinalizedQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    query_id: str
    rendered_query: RenderedQuery
    ground_truth: GroundTruthArtifactRecord | None = None
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)


def ground_truth_result_to_artifact(
    result: GroundTruthResult,
) -> GroundTruthArtifactRecord | None:
    if isinstance(result, NoPublicGroundTruthResult):
        return None
    if isinstance(result, FilteredTopKGroundTruthResult):
        if not isinstance(result.topk.scope, FilteredVectorSearchScope):
            raise ContractViolation(
                "filtered ground truth requires a filtered vector scope"
            )
        selection = result.topk.scope.selection
        return FilteredTopKGroundTruthRecord(
            answer_kind=GroundTruthAnswerKind.FILTERED_TOPK,
            metric=result.topk.metric.value,
            selection_cover_count=selection.cover.count,
            selection_cover_fingerprint=selection.cover.fingerprint(),
            answers=tuple(
                _topk_answer_row(answer) for answer in result.topk.answers
            ),
        )
    raise TypeError(f"unsupported ground-truth result {type(result).__name__}")


def accepted_predicate_plan(materialized: MaterializedQuery) -> AcceptedPredicatePlan | None:
    atoms: list[PredicateAtomRecord] = []
    predicate_exprs = tuple(_predicate_exprs(materialized.template.root))
    for expr in predicate_exprs:
        atoms.extend(_predicate_atoms(expr, materialized.binding))
    if not atoms:
        return None
    return AcceptedPredicatePlan(
        actual_predicate_count=len(atoms),
        actual_filter_logic=_predicate_logic(predicate_exprs, len(atoms)),
        atoms=tuple(atoms),
    )


def finalize_candidate(
    *,
    query_id: str,
    accepted: AcceptedBinding,
    ground_truth: GroundTruthResult,
    rendered_query: object | None = None,
) -> FinalizedQuery:
    return FinalizedQuery(
        query_id=query_id,
        rendered_query=rendered_query
        if isinstance(rendered_query, RenderedQuery)
        else render_materialized_query(accepted.materialized),
        ground_truth=ground_truth_result_to_artifact(ground_truth),
        diagnostics=accepted.judgement.diagnostics,
    )


def _predicate_exprs(node: TemplateNode) -> Iterable[Expr]:
    if isinstance(node, SelectNode):
        yield node.predicate_expr
        yield from _predicate_exprs(node.input)
        return
    if isinstance(node, RankNode | AggregateNode | LimitNode):
        yield from _predicate_exprs(node.input)


def _predicate_atoms(
    expr: Expr,
    binding: ParameterBinding,
) -> list[PredicateAtomRecord]:
    if isinstance(expr, BinaryOp):
        if expr.operator in {BinOp.AND, BinOp.OR}:
            return (
                _predicate_atoms(expr.left, binding)
                + _predicate_atoms(expr.right, binding)
            )
        if isinstance(expr.left, FieldRef) and isinstance(expr.right, HoleRef):
            return [_predicate_atom(expr.left, expr.operator.value, expr.right, binding)]
        return []
    if isinstance(expr, SetMembership):
        if isinstance(expr.expr, FieldRef) and len(expr.values) == 1 and isinstance(expr.values[0], HoleRef):
            return [_predicate_atom(expr.expr, "in", expr.values[0], binding)]
        return []
    if isinstance(expr, RangePredicateTemplate):
        return [_predicate_atom(expr.field, "between", expr.range_value, binding)]
    return []


def _predicate_atom(
    field_ref: FieldRef,
    operator: str,
    hole_ref: HoleRef,
    binding: ParameterBinding,
) -> PredicateAtomRecord:
    value = binding.values[hole_ref.hole_id]
    binding_summary = _binding_summary(value)
    return PredicateAtomRecord(
        field=field_ref.field.name,
        operator=operator,
        scope=_predicate_scope(field_ref),
        binding=binding_summary,
        summary=_atom_summary(field_ref, operator, binding_summary),
    )


def _predicate_scope(
    field_ref: FieldRef,
) -> PredicateAtomScope:
    return EntityPredicateScope(
        kind=PredicateAtomScopeKind.ENTITY,
        owner=field_ref.field.owner.name,
        alias=field_ref.variable,
    )


def _binding_summary(value: object) -> PredicateBindingSummary:
    if isinstance(value, NumericRangeValue):
        return {
            "kind": "range",
            "low": value.low,
            "high": value.high,
            "inclusive_low": value.inclusive_low,
            "inclusive_high": value.inclusive_high,
        }
    if isinstance(value, NumericComparisonValue):
        return {"kind": "comparison", "operator": value.operator, "value": value.value}
    if isinstance(value, BucketValue):
        return {"kind": "bucket", "label": value.label}
    if isinstance(value, BucketSetValue):
        return {"kind": "bucket_set", "labels": list(value.labels)}
    if isinstance(value, TextValue):
        return {"kind": "text", "text": value.text}
    if isinstance(value, ScalarValue):
        return {"kind": "scalar", "value": json.loads(value.json_payload)}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple):
        return {"kind": "tuple", "values": list(value)}
    return {"kind": type(value).__name__}


def _atom_summary(
    field_ref: FieldRef,
    operator: str,
    binding_summary: object,
) -> str:
    scope = _predicate_scope(field_ref)
    prefix = f"{scope.alias}.{field_ref.field.name}"
    return f"{prefix} {operator} {_summary_token(binding_summary)}"


def _summary_token(binding_summary: object) -> str:
    text = json.dumps(binding_summary, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 160 else text[:157] + "..."


def _predicate_logic(exprs: tuple[Expr, ...], atom_count: int) -> PredicatePlanLogic:
    if atom_count <= 1:
        return PredicatePlanLogic.AND
    has_or = any(_contains_bool_operator(expr, BinOp.OR) for expr in exprs)
    has_and = any(_contains_bool_operator(expr, BinOp.AND) for expr in exprs)
    if has_or and not has_and and len(exprs) == 1:
        return PredicatePlanLogic.OR
    return PredicatePlanLogic.AND


def _contains_bool_operator(expr: Expr, operator: BinOp) -> bool:
    if isinstance(expr, BinaryOp):
        return (
            expr.operator is operator
            or _contains_bool_operator(expr.left, operator)
            or _contains_bool_operator(expr.right, operator)
        )
    if isinstance(expr, VectorDistanceTemplate | RangePredicateTemplate):
        return False
    if isinstance(expr, SetMembership):
        return _contains_bool_operator(expr.expr, operator) or any(
            _contains_bool_operator(value, operator)
            for value in expr.values
        )
    if isinstance(expr, UnaryOp):
        return _contains_bool_operator(expr.expr, operator)
    if isinstance(expr, IsNull):
        return _contains_bool_operator(expr.expr, operator)
    return False


def render_materialized_query(
    materialized: MaterializedQuery,
    *,
    dialect_name: str = "postgresql",
    resources: DatasetResources | None = None,
) -> RenderedQuery:
    if dialect_name != "postgresql":
        raise ContractViolation(f"unsupported query dialect {dialect_name!r}")
    ast = _relational_ast(materialized.template.root, materialized.binding, resources)
    return PostgreSQLRenderer().render(ast)


def _relational_ast(
    node: TemplateNode,
    binding: ParameterBinding,
    resources: DatasetResources | None,
) -> RelationalAst:
    if isinstance(node, ScanNode):
        return SelectQuery(
            source=TableSource(table=node.entity, alias=node.scope),
            project=(RowIdRef(variable=node.scope), StarProjection()),
        )
    if isinstance(node, SelectNode):
        scan = _scan_input(node)
        return SelectQuery(
            source=TableSource(table=scan.entity, alias=scan.scope),
            project=(RowIdRef(variable=scan.scope), StarProjection()),
            where=_select_where(node, binding, resources),
        )
    if isinstance(node, AggregateNode):
        scan = _scan_input(node.input)
        where = _select_where(node.input, binding, resources)
        return CountQuery(source=TableSource(table=scan.entity, alias=scan.scope), where=where)
    if isinstance(node, RankNode):
        scan = _scan_input(node.input)
        where = _rank_where(node, binding, resources)
        score = _materialize_score_expr(node.score_expr, binding, resources)
        return SelectQuery(
            source=TableSource(table=scan.entity, alias=scan.scope),
            project=(
                RowIdRef(variable=scan.scope),
                StarProjection(),
                ScoredProjection(expr=score, alias="score"),
            ),
            where=where,
            order_by=(OrderBy(expr=score, direction=SortDir.ASC),),
            limit=Limit(node.top_k),
        )
    if isinstance(node, LimitNode):
        ast = _relational_ast(node.input, binding, resources)
        if isinstance(ast, SelectQuery):
            return SelectQuery(
                source=ast.source,
                project=ast.project,
                where=ast.where,
                group_by=ast.group_by,
                order_by=ast.order_by,
                limit=Limit(node.k),
            )
        return ast
    raise ContractViolation(f"SQL finalizer does not support {type(node).__name__}")


def _scan_input(node: TemplateNode) -> ScanNode:
    if isinstance(node, ScanNode):
        return node
    if isinstance(node, SelectNode):
        return _scan_input(node.input)
    raise ContractViolation(f"relational query expects scan/select input, got {type(node).__name__}")


def _select_where(node: TemplateNode, binding: ParameterBinding, resources: DatasetResources | None) -> Expr | None:
    if isinstance(node, SelectNode):
        current = _materialize_expr(node.predicate_expr, binding, resources)
        upstream = _select_where(node.input, binding, resources)
        return current if upstream is None else BinaryOp(operator=BinOp.AND, left=upstream, right=current)
    return None


def _rank_where(node: RankNode, binding: ParameterBinding, resources: DatasetResources | None) -> Expr | None:
    return _select_where(node.input, binding, resources)


def _materialize_score_expr(expr: Expr, binding: ParameterBinding, resources: DatasetResources | None) -> Expr:
    if isinstance(expr, VectorDistanceTemplate):
        value = binding.values[expr.query_vector.hole_id]
        vector = _vector_components(value)
        return VectorDistance(
            field=expr.field.field,
            variable=expr.field.variable,
            query_vector_literal=Literal(kind=LiteralKind.VECTOR, value_json=json.dumps(vector)),
            metric=_vector_metric(expr.field.field, resources),
        )
    return _materialize_expr(expr, binding, resources)


def _materialize_expr(expr: Expr, binding: ParameterBinding, resources: DatasetResources | None) -> Expr:
    if isinstance(expr, HoleRef):
        return _literal_expr(binding.values[expr.hole_id])
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            operator=expr.operator,
            left=_materialize_expr(expr.left, binding, resources),
            right=_materialize_expr(expr.right, binding, resources),
        )
    if isinstance(expr, SetMembership):
        values: list[Expr] = []
        for value in expr.values:
            if isinstance(value, HoleRef):
                values.extend(_literal_expr(item) for item in _iter_membership_values(binding.values[value.hole_id]))
            else:
                values.append(_materialize_expr(value, binding, resources))
        return SetMembership(expr=_materialize_expr(expr.expr, binding, resources), values=tuple(values))
    if isinstance(expr, UnaryOp):
        return UnaryOp(operator=expr.operator, expr=_materialize_expr(expr.expr, binding, resources))
    if isinstance(expr, IsNull):
        return IsNull(expr=_materialize_expr(expr.expr, binding, resources))
    if isinstance(expr, VectorDistanceTemplate):
        return _materialize_score_expr(expr, binding, resources)
    if isinstance(expr, RangePredicateTemplate):
        return _materialize_between(expr, binding)
    return expr


def _materialize_between(expr: RangePredicateTemplate, binding: ParameterBinding) -> Expr:
    value = binding.values[expr.range_value.hole_id]
    if not isinstance(value, NumericRangeValue):
        raise ContractViolation("between render lowering requires NumericRangeValue")
    left_op = BinOp.GE if value.inclusive_low else BinOp.GT
    right_op = BinOp.LE if value.inclusive_high else BinOp.LT
    return BinaryOp(
        operator=BinOp.AND,
        left=BinaryOp(operator=left_op, left=expr.field, right=_literal_expr(value.low)),
        right=BinaryOp(operator=right_op, left=expr.field, right=_literal_expr(value.high)),
    )


def _literal_expr(value: object) -> Literal:
    if isinstance(value, VectorValue):
        return Literal(kind=LiteralKind.VECTOR, value_json=json.dumps(list(value.components)))
    if isinstance(value, TextValue):
        return Literal(kind=LiteralKind.TEXT, value_json=json.dumps(value.text))
    if isinstance(value, NumericComparisonValue):
        return _literal_expr(value.value)
    if isinstance(value, NumericRangeValue):
        raise ContractViolation("range value must be lowered by between")
    if isinstance(value, BucketValue):
        return Literal(kind=LiteralKind.TEXT, value_json=json.dumps(value.label))
    if isinstance(value, ScalarValue):
        return Literal(kind=LiteralKind.TEXT, value_json=value.json_payload)
    if isinstance(value, bool):
        return Literal(kind=LiteralKind.BOOL, value_json=json.dumps(value))
    if isinstance(value, int):
        return Literal(kind=LiteralKind.INT, value_json=json.dumps(value))
    if isinstance(value, float):
        return Literal(kind=LiteralKind.DOUBLE, value_json=json.dumps(value))
    if isinstance(value, str):
        return Literal(kind=LiteralKind.TEXT, value_json=json.dumps(value))
    raise ContractViolation(f"cannot lower literal value {value!r}")


def _iter_membership_values(value: object) -> tuple[object, ...]:
    if isinstance(value, BucketSetValue):
        return tuple(value.labels)
    if isinstance(value, tuple):
        return value
    return (value,)


def _vector_components(value: object) -> tuple[float, ...]:
    if isinstance(value, VectorValue):
        return tuple(float(component) for component in value.components)
    if isinstance(value, tuple):
        return tuple(float(component) for component in value)
    raise ContractViolation("vector distance render lowering requires vector value")


def _vector_metric(field: FieldId, resources: DatasetResources | None) -> VectorMetric:
    if resources is None:
        raise ContractViolation("vector distance rendering requires DatasetResources")
    entity = resources.dataset.field_bearing_entity(field.owner)
    for candidate in entity.fields:
        if candidate.id == field and isinstance(candidate.type, VectorType):
            return candidate.type.metric
    raise ContractViolation(f"cannot resolve vector metric for {field}")


def _topk_answer_row(answer: object) -> dict[str, int]:
    rowid = getattr(answer, "rowid")
    return {"_rowid": int(rowid)}


__all__ = [
    "FilteredTopKGroundTruthRecord",
    "FinalizedQuery",
    "GroundTruthArtifactRecord",
    "finalize_candidate",
    "ground_truth_result_to_artifact",
    "render_materialized_query",
]
