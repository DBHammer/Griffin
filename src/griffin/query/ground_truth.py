"""Typed planning and materialization of public query ground truth."""

from __future__ import annotations

from typing import Union

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.ir.expr import VectorDistanceTemplate
from griffin.query.admit import AcceptedBinding
from griffin.query.evaluate import (
    ExactVectorTopK,
    FilteredAnnSearchEvidence,
    FilteredVectorSearchScope,
    topk_answers_from_vector_topk_hint,
)
from griffin.query.model import (
    AggregateNode,
    LimitNode,
    QueryTask,
    QueryTemplate,
    RankNode,
    SelectNode,
    TemplateNode,
)
from griffin.query.task import ObjectiveKind, ResultSemanticsKind
from griffin.query.observation import (
    CoverPayload,
    ObservationRequest,
    ObservationRequestKind,
    ObservationSession,
)


class NoPublicGroundTruthPlan(
    msgspec.Struct,
    frozen=True,
    tag="none",
    tag_field="ground_truth_plan_kind",
):
    template: QueryTemplate


class FilteredTopKGroundTruthPlan(
    msgspec.Struct,
    frozen=True,
    tag="filtered_topk",
    tag_field="ground_truth_plan_kind",
):
    template: QueryTemplate
    rank: RankNode
    selection_input: TemplateNode


GroundTruthPlan = Union[
    NoPublicGroundTruthPlan,
    FilteredTopKGroundTruthPlan,
]


class NoPublicGroundTruthResult(
    msgspec.Struct,
    frozen=True,
    tag="none",
    tag_field="ground_truth_result_kind",
):
    pass


class FilteredTopKGroundTruthResult(
    msgspec.Struct,
    frozen=True,
    tag="filtered_topk",
    tag_field="ground_truth_result_kind",
):
    topk: ExactVectorTopK


GroundTruthResult = Union[
    NoPublicGroundTruthResult,
    FilteredTopKGroundTruthResult,
]


class GroundTruthProblem(msgspec.Struct, frozen=True, omit_defaults=True):
    accepted: AcceptedBinding
    plan: GroundTruthPlan


def compile_ground_truth_plan(
    query_task: QueryTask,
    template: QueryTemplate,
) -> GroundTruthPlan:
    rank = _rank_node(template.root)
    if (
        query_task.result_semantics.kind is not ResultSemanticsKind.RANKED_ENTITIES
        or rank is None
        or not isinstance(rank.score_expr, VectorDistanceTemplate)
    ):
        return NoPublicGroundTruthPlan(template=template)
    if any(
        objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND
        for objective in query_task.objectives
    ):
        return FilteredTopKGroundTruthPlan(
            template=template,
            rank=rank,
            selection_input=rank.input,
        )
    return NoPublicGroundTruthPlan(template=template)


class GroundTruthMaterializer:
    def __init__(self, *, observation_session: ObservationSession) -> None:
        self._observation_session = observation_session

    def materialize(
        self,
        problem: GroundTruthProblem,
    ) -> GroundTruthResult:
        accepted = problem.accepted
        plan = problem.plan
        evidence = accepted.evidence
        if plan.template != accepted.materialized.template:
            raise ContractViolation(
                "ground-truth plan template does not match accepted binding"
            )
        if evidence.identity.materialized != accepted.materialized:
            raise ContractViolation(
                "search evidence candidate does not match accepted binding"
            )
        if isinstance(plan, NoPublicGroundTruthPlan):
            return NoPublicGroundTruthResult()
        if not isinstance(plan, FilteredTopKGroundTruthPlan):
            raise TypeError(f"unsupported ground-truth plan {type(plan).__name__}")
        if not isinstance(evidence, FilteredAnnSearchEvidence):
            raise ContractViolation(
                "filtered top-k ground truth requires filtered ANN search evidence"
            )
        if evidence.global_topk.rank != plan.rank:
            raise ContractViolation(
                "filtered ground-truth rank does not match search evidence"
            )
        if evidence.selection.node != plan.selection_input:
            raise ContractViolation(
                "filtered ground-truth selection does not match compiled plan"
            )
        summary = self._observation_session.observe_canonical(
            ObservationRequest(
                kind=ObservationRequestKind.VECTOR_TOPK,
                field=evidence.global_topk.field,
                query_vector=evidence.global_topk.query_vector.components,
                k=plan.rank.top_k,
                cover_payload=CoverPayload.from_cover(
                    cover=evidence.selection.cover,
                    answer_scope=evidence.selection.answer_scope,
                ),
            )
        )
        if summary.topk is None:
            raise ContractViolation(
                "filtered top-k materialization missing exact top-k payload"
            )
        topk = ExactVectorTopK(
            rank=plan.rank,
            field=evidence.global_topk.field,
            metric=evidence.global_topk.metric,
            query_vector=evidence.global_topk.query_vector,
            scope=FilteredVectorSearchScope(selection=evidence.selection),
            answers=topk_answers_from_vector_topk_hint(summary.topk),
        )
        if len(topk.answers) != plan.rank.top_k:
            raise ContractViolation(
                "accepted filtered ANN binding does not have the required exact top-k"
            )
        return FilteredTopKGroundTruthResult(topk=topk)


def _rank_node(node: TemplateNode) -> RankNode | None:
    if isinstance(node, RankNode):
        return node
    if isinstance(node, SelectNode | AggregateNode | LimitNode):
        return _rank_node(node.input)
    return None


__all__ = [
    "FilteredTopKGroundTruthPlan",
    "FilteredTopKGroundTruthResult",
    "GroundTruthMaterializer",
    "GroundTruthPlan",
    "GroundTruthProblem",
    "GroundTruthResult",
    "NoPublicGroundTruthPlan",
    "NoPublicGroundTruthResult",
    "compile_ground_truth_plan",
]
