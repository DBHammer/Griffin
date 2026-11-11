"""Relational predicate-cover computation for evaluation and search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from griffin.dataset.exact_cover import (
    OrdinalIntervalSet,
    OwnerDeterministicMetadata,
    RowCover,
)
from griffin.infra.errors import ContractViolation
from griffin.ir.expr import (
    BinaryOp,
    BinOp,
    FieldRef,
    HoleRef,
    RangePredicateTemplate,
    SetMembership,
)
from griffin.model_ids import TableId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.model import (
    AggregateNode,
    AnswerScope,
    EntityAnswerScope,
    LimitNode,
    ParameterValue,
    RankNode,
    ScanNode,
    ScopeId,
    SelectNode,
    TemplateNode,
    UnscopedAnswerScope,
    stable_signature,
)


@dataclass(frozen=True)
class PredicateCoverComputation:
    node_id: str
    scope_id: ScopeId
    cover: RowCover
    answer_scope: AnswerScope = field(default_factory=UnscopedAnswerScope)

    @property
    def total_count(self) -> int:
        return self.cover.total_count

    @property
    def selectivity(self) -> float:
        return self.cover.selectivity

    @property
    def cover_fingerprint(self) -> str:
        return self.cover.fingerprint()


def compute_predicate_cover(
    *,
    resources: DatasetResources,
    node: TemplateNode,
    values: Mapping[str, ParameterValue],
) -> PredicateCoverComputation:
    """Compute the answer-row cover for a relational template node."""
    if isinstance(node, ScanNode):
        return _scan_cover(resources=resources, node=node)
    if isinstance(node, SelectNode):
        active = compute_predicate_cover(
            resources=resources,
            node=node.input,
            values=values,
        )
        selected = predicate_cover(resources, node.predicate_expr, values, active.cover)
        return PredicateCoverComputation(
            node_id=node.node_id,
            scope_id=node.output_scope,
            cover=active.cover.intersect(selected),
            answer_scope=active.answer_scope,
        )
    if isinstance(node, RankNode | LimitNode | AggregateNode):
        return compute_predicate_cover(
            resources=resources,
            node=node.input,
            values=values,
        )
    raise ContractViolation(
        f"predicate cover search does not support node {type(node).__name__}"
    )


def predicate_cover(
    resources: DatasetResources,
    expr: object,
    values: Mapping[str, ParameterValue],
    active: RowCover,
) -> RowCover:
    if isinstance(expr, BinaryOp):
        if expr.operator is BinOp.AND:
            return predicate_cover(resources, expr.left, values, active).intersect(
                predicate_cover(resources, expr.right, values, active)
            )
        if expr.operator is BinOp.OR:
            return predicate_cover(resources, expr.left, values, active).union(
                predicate_cover(resources, expr.right, values, active)
            )
        if isinstance(expr.left, FieldRef) and isinstance(expr.right, HoleRef):
            return field_predicate_cover(
                resources,
                expr.left,
                values[expr.right.hole_id],
            ).intersect(active)
        raise ContractViolation("predicate cover leaf expects FieldRef comparison")
    if isinstance(expr, SetMembership):
        if (
            not isinstance(expr.expr, FieldRef)
            or len(expr.values) != 1
            or not isinstance(expr.values[0], HoleRef)
        ):
            raise ContractViolation(
                "predicate cover IN leaf expects FieldRef IN HoleRef"
            )
        return field_predicate_cover(
            resources,
            expr.expr,
            values[expr.values[0].hole_id],
        ).intersect(active)
    if isinstance(expr, RangePredicateTemplate):
        return field_predicate_cover(
            resources,
            expr.field,
            values[expr.range_value.hole_id],
        ).intersect(active)
    raise ContractViolation(
        f"predicate cover does not support expression {type(expr).__name__}"
    )


def field_predicate_cover(
    resources: DatasetResources,
    field: FieldRef,
    predicate: ParameterValue,
) -> RowCover:
    return RowCover.from_normalized_intervals(
        field_predicate_intervals(resources, field, predicate),
        total_count=entity_row_count(resources, field.field.owner),
    )


def field_predicate_intervals(
    resources: DatasetResources,
    field: FieldRef,
    predicate: ParameterValue,
) -> OrdinalIntervalSet:
    cache_key = stable_signature(
        "observation",
        {
            "kind": "predicate_interval_cover",
            "field": field.field,
            "predicate": predicate,
        },
    )
    return resources.predicate_interval_cover(
        cache_key,
        lambda: _field_predicate_intervals_uncached(resources, field, predicate),
    )


def _field_predicate_intervals_uncached(
    resources: DatasetResources,
    field: FieldRef,
    predicate: ParameterValue,
) -> OrdinalIntervalSet:
    owner = field.field.owner
    if not isinstance(owner, TableId):
        raise ContractViolation(
            f"predicate cover does not support field owner {owner!r}"
        )
    metadata = resources.dataset.deterministic_table(owner.name)
    if metadata is None:
        raise ContractViolation(
            f"deterministic metadata missing for field owner {owner!r}"
        )
    return resources.predicate_access.public_cover(
        metadata,
        field.field.name,
        predicate,
        dataset=resources.dataset,
        field_id=field.field,
    )


def entity_row_count(resources: DatasetResources, entity: object) -> int:
    metadata: OwnerDeterministicMetadata | None
    if not isinstance(entity, TableId):
        raise ContractViolation(f"scan does not support entity {entity!r}")
    metadata = resources.dataset.deterministic_table(entity.name)
    if metadata is None:
        raise ContractViolation(f"deterministic metadata missing for {entity!r}")
    return int(metadata.row_count)


def _scan_cover(
    *,
    resources: DatasetResources,
    node: ScanNode,
) -> PredicateCoverComputation:
    row_count = entity_row_count(resources, node.entity)
    return PredicateCoverComputation(
        node_id=node.node_id,
        scope_id=node.scope,
        cover=RowCover.full(row_count),
        answer_scope=EntityAnswerScope(entity=node.entity),
    )


__all__ = [
    "PredicateCoverComputation",
    "compute_predicate_cover",
    "entity_row_count",
    "field_predicate_cover",
    "field_predicate_intervals",
    "predicate_cover",
]
