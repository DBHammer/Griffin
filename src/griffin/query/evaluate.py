"""Evaluation, observation, evidence, and judgement contracts."""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Union

import msgspec

from griffin.dataset.exact_cover import RowCover
from griffin.dataset.reader import Dataset
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric, VectorType
from griffin.ir.expr import (
    AggregateExpr,
    AggregationOp,
    Expr,
    FieldRef,
    VectorDistanceTemplate,
)
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.model import (
    AggregateNode,
    AnswerScope,
    BindingFingerprint,
    EntityAnswerScope,
    MaterializedQuery,
    NodeId,
    ParameterBinding,
    ParameterValue,
    QueryTask,
    QueryTemplate,
    RankNode,
    ScanNode,
    SearchIdentity,
    ScopeId,
    SelectNode,
    TemplateNode,
    TemplateNodeRole,
    UnscopedAnswerScope,
    VectorValue,
    LimitNode,
    iter_template_nodes,
)
from griffin.query.predicate_cover_search import entity_row_count as _entity_row_count
from griffin.query.observation import (
    CoverPayload,
    FilteredAnnHitHint,
    ObservationScope,
    ObservationRequestKind,
    ObservationRequest,
    ObservationSummary,
    ObservationSession,
    PredicateCoverHint,
    VectorTopKHint,
)
from griffin.query.task import (
    Objective,
    ObjectiveKind,
    ObjectiveMetric,
    PublicTaskKind,
    PublicTaskReport,
)
from griffin.query.filtered_ann_band import (
    FilteredAnnHitBandDiagnostic,
    format_hit_band_diagnostic,
    gls_from_hit_count,
    gls_target_hit_band,
    hit_band,
    hit_band_diagnostics,
    hit_band_for_gls,
    hit_loss,
    resolve_search_gls_target_for_mode,
    resolve_hit_band,
    topk_hit_count,
)


class SearchFeedbackKind(enum.Enum):
    RETRY_PARAMETERS = "retry_parameters"
    RETRY_TEMPLATE = "retry_template"
    STOP = "stop"


class NodeEvalKey(msgspec.Struct, frozen=True, omit_defaults=True):
    node_id: NodeId
    subtree_binding_fingerprint: BindingFingerprint
    node_role: TemplateNodeRole | None = None


class TopKAnswer(msgspec.Struct, frozen=True, omit_defaults=True):
    rowid: int
    distance: float | None = None
    score: float | None = None


class ExactSelection(msgspec.Struct, frozen=True, omit_defaults=True):
    node: TemplateNode
    scope_id: ScopeId
    answer_scope: AnswerScope
    cover: RowCover


class GlobalVectorSearchScope(
    msgspec.Struct,
    frozen=True,
    tag="global",
    tag_field="vector_scope_kind",
):
    answer_scope: AnswerScope


class FilteredVectorSearchScope(
    msgspec.Struct,
    frozen=True,
    tag="filtered",
    tag_field="vector_scope_kind",
):
    selection: ExactSelection


VectorSearchScope = Union[GlobalVectorSearchScope, FilteredVectorSearchScope]


class ExactVectorTopK(msgspec.Struct, frozen=True, omit_defaults=True):
    rank: RankNode
    field: FieldId
    metric: VectorMetric
    query_vector: VectorValue
    scope: VectorSearchScope
    answers: tuple[TopKAnswer, ...]


class PredicateCoverObservation(msgspec.Struct, frozen=True, tag="predicate_cover", tag_field="observation_kind", omit_defaults=True):
    selection: ExactSelection
    node_role: TemplateNodeRole | None = None

    @property
    def node_id(self) -> NodeId:
        return self.selection.node.node_id

    @property
    def scope_id(self) -> ScopeId:
        return self.selection.scope_id

    @property
    def cover_count(self) -> int:
        return self.selection.cover.count

    @property
    def total_count(self) -> int:
        return self.selection.cover.total_count

    @property
    def selectivity(self) -> float:
        return self.selection.cover.selectivity

    @property
    def cover_fingerprint(self) -> str:
        return self.selection.cover.fingerprint()


class SelectionCoverObservation(msgspec.Struct, frozen=True, tag="selection_cover", tag_field="observation_kind", omit_defaults=True):
    selection: ExactSelection

    @property
    def node_id(self) -> NodeId:
        return self.selection.node.node_id

    @property
    def answer_scope_id(self) -> ScopeId:
        return self.selection.scope_id

    @property
    def cover_count(self) -> int:
        return self.selection.cover.count

    @property
    def total_count(self) -> int:
        return self.selection.cover.total_count

    @property
    def selectivity(self) -> float:
        return self.selection.cover.selectivity

    @property
    def cover_fingerprint(self) -> str:
        return self.selection.cover.fingerprint()

    @property
    def answer_scope(self) -> AnswerScope:
        return self.selection.answer_scope


class CandidateSearchIdentity(msgspec.Struct, frozen=True, omit_defaults=True):
    materialized: MaterializedQuery
    search: SearchIdentity


class PredicateSearchEvidence(
    msgspec.Struct,
    frozen=True,
    tag="predicate",
    tag_field="evidence_kind",
):
    identity: CandidateSearchIdentity
    selection: ExactSelection


class FilteredAnnSearchEvidence(
    msgspec.Struct,
    frozen=True,
    tag="filtered_ann",
    tag_field="evidence_kind",
):
    identity: CandidateSearchIdentity
    selection: ExactSelection
    global_topk: ExactVectorTopK
    hit_count: int


class NoConstraintSearchEvidence(
    msgspec.Struct,
    frozen=True,
    tag="no_constraint",
    tag_field="evidence_kind",
):
    identity: CandidateSearchIdentity


SearchEvidence = Union[
    PredicateSearchEvidence,
    FilteredAnnSearchEvidence,
    NoConstraintSearchEvidence,
]


@dataclass(frozen=True)
class FilteredVectorRankEvaluation:
    selection: ExactSelection
    global_topk: ExactVectorTopK
    hit_count: int


VectorRankEvaluation = FilteredVectorRankEvaluation


@dataclass(frozen=True)
class NodeEval:
    scope_covers: Mapping[ScopeId, RowCover]
    answer_scope: AnswerScope


@dataclass
class EvalState:
    observation_session: ObservationSession
    query_task: QueryTask
    prepared_predicate_covers: Mapping[NodeId, PredicateCoverHint] = field(
        default_factory=dict
    )
    observation_scope: ObservationScope | None = None
    node_results: dict[NodeEvalKey, NodeEval] = field(default_factory=dict)
    vector_ranks: list[VectorRankEvaluation] = field(default_factory=list)


class SearchFeedback(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: SearchFeedbackKind
    reason: str
    excluded_binding_signatures: tuple[str, ...] = ()
    excluded_structural_signatures: tuple[str, ...] = ()


class CandidateJudgement(msgspec.Struct, frozen=True, omit_defaults=True):
    accepted: bool
    score: float
    failed_constraints: tuple[str, ...] = ()
    search_feedback: SearchFeedback | None = None
    public_report: PublicTaskReport | None = None
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)


class ObjectiveScore(msgspec.Struct, frozen=True, omit_defaults=True):
    accepted: bool
    score: float
    failure: str | None = None
    diagnostics: Mapping[str, str] = msgspec.field(default_factory=dict)


class Evaluator:
    def evaluate(
        self,
        query: MaterializedQuery,
        resources: DatasetResources,
        observation_session: ObservationSession,
        *,
        query_task: QueryTask,
        search_identity: SearchIdentity,
        prepared_predicate_covers: Mapping[NodeId, PredicateCoverHint] | None = None,
        observation_scope: ObservationScope | None = None,
    ) -> SearchEvidence:
        state = EvalState(
            observation_session=observation_session,
            query_task=query_task,
            prepared_predicate_covers=dict(prepared_predicate_covers or {}),
            observation_scope=observation_scope,
        )
        root_eval = self._eval_node(query.template.root, query, resources, state)
        return _search_evidence_from_evaluation(
            query_task=query_task,
            materialized=query,
            search_identity=search_identity,
            root_eval=root_eval,
            state=state,
        )

    def _eval_node(
        self,
        node: TemplateNode,
        query: MaterializedQuery,
        resources: DatasetResources,
        state: EvalState,
    ) -> NodeEval:
        key = NodeEvalKey(
            node_id=node.node_id,
            subtree_binding_fingerprint=query.binding.binding_signature,
            node_role=_template_node_role(node),
        )
        cached = state.node_results.get(key)
        if cached is not None:
            return cached
        if isinstance(node, ScanNode):
            result = self._eval_scan(node, resources)
        elif isinstance(node, SelectNode):
            result = self._eval_select(node, query, resources, state)
        elif isinstance(node, AggregateNode):
            result = self._eval_aggregate(node, query, resources, state)
        elif isinstance(node, RankNode):
            result = self._eval_rank(node, query, resources, state)
        elif isinstance(node, LimitNode):
            result = self._eval_limit(node, query, resources, state)
        else:
            raise ContractViolation(f"evaluator does not yet support node kind {type(node).__name__}")
        state.node_results[key] = result
        return result

    def _eval_scan(self, node: ScanNode, resources: DatasetResources) -> NodeEval:
        row_count = _entity_row_count(resources, node.entity)
        cover = RowCover.full(row_count)
        return NodeEval(
            scope_covers={node.scope: cover},
            answer_scope=EntityAnswerScope(entity=node.entity),
        )

    def _eval_select(
        self,
        node: SelectNode,
        query: MaterializedQuery,
        resources: DatasetResources,
        state: EvalState,
    ) -> NodeEval:
        cover = _observe_canonical_predicate_cover(
            resources=resources,
            state=state,
            template=query.template,
            node=node,
            values=query.binding.values,
            answer_field=_selection_answer_field(node),
        )
        return NodeEval(
            scope_covers={node.output_scope: cover.cover},
            answer_scope=cover.answer_scope,
        )

    def _eval_aggregate(
        self,
        node: AggregateNode,
        query: MaterializedQuery,
        resources: DatasetResources,
        state: EvalState,
    ) -> NodeEval:
        input_eval = self._eval_node(node.input, query, resources, state)
        input_cover = _single_cover(input_eval)
        if not (
            isinstance(node.aggregate.aggregate_expr, AggregateExpr)
            and node.aggregate.aggregate_expr.operator is AggregationOp.COUNT_STAR
        ):
            raise ContractViolation("aggregate evaluator supports COUNT(*) only")
        return NodeEval(
            scope_covers={node.output_scope: input_cover},
            answer_scope=input_eval.answer_scope,
        )

    def _eval_rank(
        self,
        node: RankNode,
        query: MaterializedQuery,
        resources: DatasetResources,
        state: EvalState,
    ) -> NodeEval:
        input_eval = self._eval_node(node.input, query, resources, state)
        input_cover = _single_cover(input_eval)
        field, vector = _rank_vector_request(node.score_expr, query.binding)
        vector_metric = _vector_metric_value(resources.dataset, field.field)
        is_filtered = any(
            objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND
            for objective in state.query_task.objectives
        )
        if not is_filtered:
            raise ContractViolation(
                "rank evaluation is only supported for filtered ANN queries"
            )
        answer_scope = _answer_scope_for_rank(node)
        selection = _exact_selection(node.input, input_eval)
        unfiltered_topk = _observe_canonical_vector_topk(
            state,
            field=field.field,
            vector=vector,
            k=node.top_k,
            row_filter=None,
        )
        global_topk = ExactVectorTopK(
            rank=node,
            field=field.field,
            metric=vector_metric,
            query_vector=VectorValue(components=vector),
            scope=GlobalVectorSearchScope(answer_scope=answer_scope),
            answers=topk_answers_from_vector_topk_hint(unfiltered_topk),
        )
        hit_count = topk_hit_count(
            tuple(answer.rowid for answer in global_topk.answers),
            input_cover,
        )
        state.vector_ranks.append(
            FilteredVectorRankEvaluation(
                selection=selection,
                global_topk=global_topk,
                hit_count=hit_count,
            )
        )
        return NodeEval(
            scope_covers={node.answer_scope: input_cover},
            answer_scope=answer_scope,
        )

    def _eval_limit(
        self,
        node: LimitNode,
        query: MaterializedQuery,
        resources: DatasetResources,
        state: EvalState,
    ) -> NodeEval:
        child = self._eval_node(node.input, query, resources, state)
        return NodeEval(
            scope_covers={node.output_scope: _single_cover(child)},
            answer_scope=child.answer_scope,
        )

def _search_evidence_from_evaluation(
    *,
    query_task: QueryTask,
    materialized: MaterializedQuery,
    search_identity: SearchIdentity,
    root_eval: NodeEval,
    state: EvalState,
) -> SearchEvidence:
    identity = CandidateSearchIdentity(
        materialized=materialized,
        search=search_identity,
    )
    filtered_ranks = tuple(
        rank
        for rank in state.vector_ranks
        if isinstance(rank, FilteredVectorRankEvaluation)
    )
    if any(
        objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND
        for objective in query_task.objectives
    ):
        if len(filtered_ranks) != 1:
            raise ContractViolation(
                "filtered ANN evaluation requires exactly one filtered vector rank"
            )
        filtered_rank = filtered_ranks[0]
        return FilteredAnnSearchEvidence(
            identity=identity,
            selection=filtered_rank.selection,
            global_topk=filtered_rank.global_topk,
            hit_count=filtered_rank.hit_count,
        )

    if query_task.objectives:
        selections = tuple(
            _exact_selection(
                _template_node_by_id(materialized.template, key.node_id),
                node_eval,
            )
            for key, node_eval in state.node_results.items()
            if key.node_role
            is TemplateNodeRole.SELECTION_FILTER
        )
        if not selections:
            selections = (
                _exact_selection(materialized.template.root, root_eval),
            )
        return PredicateSearchEvidence(
            identity=identity,
            selection=selections[-1],
        )
    return NoConstraintSearchEvidence(identity=identity)


def _exact_selection(node: TemplateNode, evaluation: NodeEval) -> ExactSelection:
    covers = tuple(evaluation.scope_covers.items())
    if len(covers) != 1:
        raise ContractViolation("exact selection requires exactly one active scope")
    scope_id, cover = covers[0]
    return ExactSelection(
        node=node,
        scope_id=scope_id,
        answer_scope=evaluation.answer_scope,
        cover=cover,
    )


def filtered_ann_search_evidence(
    *,
    query_task: QueryTask,
    query: MaterializedQuery,
    search_identity: SearchIdentity,
    resources: DatasetResources,
    predicate_covers: Mapping[NodeId, PredicateCoverHint],
    hit: FilteredAnnHitHint | None,
) -> FilteredAnnSearchEvidence | None:
    """Build authoritative evidence from the exact filtered-ANN search result."""
    if hit is None:
        return None
    objective = next(
        (
            item
            for item in query_task.objectives
            if item.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND
        ),
        None,
    )
    if objective is None or objective.k is None:
        return None
    matching_covers = tuple(
        cover
        for cover in predicate_covers.values()
        if cover.cover == hit.selection.cover
    )
    if len(matching_covers) != 1:
        raise ContractViolation(
            "filtered ANN acceptance requires exactly one predicate cover matching "
            "the winning hit observation"
        )
    cover = matching_covers[0]
    cover_node = _template_node_by_id(query.template, cover.node_id)
    selection = ExactSelection(
        node=cover_node,
        scope_id=cover.scope_id,
        answer_scope=cover.answer_scope,
        cover=cover.cover,
    )
    rank = _rank_node_in_subtree(query.template.root)
    if rank is None:
        raise ContractViolation("filtered ANN acceptance requires a vector rank node")
    field, vector = _rank_vector_request(rank.score_expr, query.binding)
    global_topk = ExactVectorTopK(
        rank=rank,
        field=field.field,
        metric=_vector_metric_value(resources.dataset, field.field),
        query_vector=VectorValue(components=vector),
        scope=GlobalVectorSearchScope(answer_scope=cover.answer_scope),
        answers=topk_answers_from_vector_topk_hint(hit.global_topk),
    )
    computed_hit_count = topk_hit_count(
        tuple(answer.rowid for answer in global_topk.answers),
        selection.cover,
    )
    if computed_hit_count != hit.hit_count:
        raise ContractViolation(
            "filtered ANN search hit count does not match exact top-k and selection"
        )
    return FilteredAnnSearchEvidence(
        identity=CandidateSearchIdentity(
            materialized=query,
            search=search_identity,
        ),
        selection=selection,
        global_topk=global_topk,
        hit_count=computed_hit_count,
    )


def _template_node_by_id(template: QueryTemplate, node_id: NodeId) -> TemplateNode:
    for node in iter_template_nodes(template.root):
        if node.node_id == node_id:
            return node
    raise ContractViolation(f"template node {node_id!r} not found")


def _rank_node_in_subtree(node: TemplateNode) -> RankNode | None:
    if isinstance(node, RankNode):
        return node
    if isinstance(node, SelectNode | AggregateNode | LimitNode):
        return _rank_node_in_subtree(node.input)
    return None


_OBJECTIVE_PUBLIC_KIND: dict[ObjectiveKind, PublicTaskKind] = {
    ObjectiveKind.SELECTIVITY_BAND: PublicTaskKind.SELECTIVITY,
    ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND: PublicTaskKind.FILTERED_ANN,
}

_OBJECTIVE_METRICS: dict[ObjectiveKind, ObjectiveMetric] = {
    ObjectiveKind.SELECTIVITY_BAND: ObjectiveMetric.SELECTIVITY,
    ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND: (
        ObjectiveMetric.FILTERED_ANN_HIT_COUNT
    ),
    ObjectiveKind.TOPK_AVAILABLE: ObjectiveMetric.TOPK_AVAILABLE,
}


def judge_search_evidence(
    objectives: tuple[Objective, ...],
    evidence: SearchEvidence,
) -> CandidateJudgement:
    failed: list[str] = []
    scores: list[float] = []
    diagnostics: dict[str, str] = {}
    for objective in objectives:
        expected_metric = _OBJECTIVE_METRICS[objective.kind]
        if objective.metric is not expected_metric:
            raise ContractViolation(
                f"{objective.kind.value} objective metric must be "
                f"{expected_metric.value}, got {objective.metric}"
            )
        outcome = _typed_objective_score(objective, evidence)
        scores.append(outcome.score)
        diagnostics.update(outcome.diagnostics)
        if not outcome.accepted and outcome.failure is not None:
            failed.append(f"{objective.kind.value}:{outcome.failure}")
    accepted = not failed
    return CandidateJudgement(
        accepted=accepted,
        score=min(scores) if scores else 1.0,
        failed_constraints=tuple(failed),
        search_feedback=None
        if accepted
        else SearchFeedback(
            kind=SearchFeedbackKind.RETRY_PARAMETERS,
            reason=";".join(failed),
        ),
        public_report=_typed_public_report(objectives, evidence),
        diagnostics={**diagnostics, "failed_constraints": json.dumps(failed)},
    )


def _typed_objective_score(
    objective: Objective,
    evidence: SearchEvidence,
) -> ObjectiveScore:
    if objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND:
        if not isinstance(evidence, FilteredAnnSearchEvidence):
            return ObjectiveScore(False, 0.0, "missing filtered ANN evidence")
        return _typed_filtered_ann_score(objective, evidence)
    if objective.kind is ObjectiveKind.SELECTIVITY_BAND:
        if isinstance(evidence, FilteredAnnSearchEvidence):
            observed = evidence.selection.cover.selectivity
        elif isinstance(evidence, PredicateSearchEvidence):
            observed = evidence.selection.cover.selectivity
        else:
            return ObjectiveScore(False, 0.0, "missing predicate selection")
        return _score_objective_band(objective, observed)
    if objective.kind is ObjectiveKind.TOPK_AVAILABLE:
        if objective.k is None:
            return ObjectiveScore(False, 0.0, "top-k objective missing k")
        if isinstance(evidence, FilteredAnnSearchEvidence):
            available = min(
                evidence.selection.cover.count,
                evidence.global_topk.rank.top_k,
            )
        else:
            return ObjectiveScore(False, 0.0, "missing top-k evidence")
        accepted = available >= objective.k
        return ObjectiveScore(
            accepted,
            1.0 if accepted else 0.0,
            None if accepted else "top-k unavailable",
        )
    raise ContractViolation(f"unsupported objective kind {objective.kind.value}")


def _score_objective_band(
    objective: Objective,
    observed: float,
) -> ObjectiveScore:
    if objective.minimum is None or objective.maximum is None:
        return ObjectiveScore(
            False,
            0.0,
            f"{objective.metric.value} objective missing band",
        )
    return _score_numeric_band(
        observed,
        objective.minimum,
        objective.maximum,
        metric=objective.metric.value,
    )


def _score_numeric_band(
    observed: float,
    minimum: float,
    maximum: float,
    *,
    metric: str,
) -> ObjectiveScore:
    if observed < minimum:
        return ObjectiveScore(False, 0.0, f"{metric} below minimum")
    if observed > maximum:
        return ObjectiveScore(False, 0.0, f"{metric} above maximum")
    center = (minimum + maximum) / 2.0
    tolerance = max(maximum - center, center - minimum, 1e-9)
    return ObjectiveScore(
        True,
        max(0.0, 1.0 - abs(observed - center) / tolerance),
    )


def _typed_filtered_ann_score(
    objective: Objective,
    evidence: FilteredAnnSearchEvidence,
) -> ObjectiveScore:
    selection = evidence.selection.cover
    k = objective.k or evidence.global_topk.rank.top_k
    observed = evidence.hit_count
    if objective.filtered_ann_gls_target_mode is not None:
        target_point = resolve_search_gls_target_for_mode(
            mode=objective.filtered_ann_gls_target_mode,
            total_count=selection.total_count,
            cover_count=selection.count,
            k=k,
            search_identity=evidence.identity.search,
        )
        if target_point is None:
            return ObjectiveScore(
                False,
                0.0,
                "filtered_ann GLS target mode infeasible",
            )
        low, high = gls_target_hit_band(
            target_hit_count=target_point.hit_count,
            k=k,
        )
        diagnostics = {
            "filtered_ann_gls_target_mode": (
                objective.filtered_ann_gls_target_mode.value
            ),
            "filtered_ann_gls_target_hit_count": str(target_point.hit_count),
            "filtered_ann_gls_target": str(target_point.gls),
            "filtered_ann_gls_observed_hit_count": str(observed),
        }
        observed_gls = gls_from_hit_count(
            hit_count=observed,
            total_count=selection.total_count,
            cover_count=selection.count,
            k=k,
        )
        if observed_gls is not None:
            diagnostics["filtered_ann_gls_observed"] = str(observed_gls)
        if observed < low:
            return ObjectiveScore(
                False,
                0.0,
                "filtered_ann GLS target hits below target",
                diagnostics,
            )
        if observed > high:
            return ObjectiveScore(
                False,
                0.0,
                "filtered_ann GLS target hits above target",
                diagnostics,
            )
        return ObjectiveScore(
            True,
            1.0 if observed == target_point.hit_count else 0.5,
            diagnostics=diagnostics,
        )
    if objective.minimum is not None and objective.maximum is not None:
        band = hit_band_for_gls(
            total_count=selection.total_count,
            cover_count=selection.count,
            k=k,
            target_gls_min=objective.minimum,
            target_gls_max=objective.maximum,
        )
        if band is None:
            return ObjectiveScore(False, 0.0, "filtered_ann GLS band infeasible")
    else:
        if objective.hardness is None:
            return ObjectiveScore(
                False,
                0.0,
                "filtered_ann difficulty objective missing hardness",
            )
        band, diagnostic = resolve_hit_band(
            total_count=selection.total_count,
            cover_count=selection.count,
            k=k,
            hardness=objective.hardness,
        )
        if band is None:
            return ObjectiveScore(
                False,
                0.0,
                format_hit_band_diagnostic(diagnostic),
                hit_band_diagnostics(diagnostic),
            )
    if observed < band[0]:
        return ObjectiveScore(False, 0.0, "filtered_ann hits below minimum")
    if observed > band[1]:
        return ObjectiveScore(False, 0.0, "filtered_ann hits above maximum")
    center = (band[0] + band[1]) / 2.0
    tolerance = max(band[1] - center, center - band[0], 1.0)
    return ObjectiveScore(
        True,
        max(0.0, 1.0 - abs(observed - center) / tolerance),
    )


def _typed_public_report(
    objectives: tuple[Objective, ...],
    evidence: SearchEvidence,
) -> PublicTaskReport | None:
    for objective in sorted(objectives, key=_public_report_priority):
        public_kind = _OBJECTIVE_PUBLIC_KIND.get(objective.kind)
        if public_kind is None:
            continue
        achieved = _typed_objective_value(objective, evidence)
        target = achieved
        metadata_selectivity: float | None = None
        if (
            objective.filtered_ann_gls_target_mode is not None
            and isinstance(evidence, FilteredAnnSearchEvidence)
        ):
            # rnd/min/max GLS mode：target 是 search_identity 派生的 hit_count，
            # 与搜索侧和 acceptance 侧一致（此前误用 achieved 冒充 target）。
            point = resolve_search_gls_target_for_mode(
                mode=objective.filtered_ann_gls_target_mode,
                total_count=evidence.selection.cover.total_count,
                cover_count=evidence.selection.cover.count,
                k=objective.k or evidence.global_topk.rank.top_k,
                search_identity=evidence.identity.search,
            )
            if point is not None:
                target = float(point.hit_count)
        elif objective.minimum is not None and objective.maximum is not None:
            target = (objective.minimum + objective.maximum) / 2.0
        # 落盘 Griffin 用 metadata 算出的 selectivity（predicate cover），供
        # Fig 7(a) 与数据库实测 selectivity 对比，验证 metadata 计算准确性。
        if isinstance(evidence, FilteredAnnSearchEvidence):
            metadata_selectivity = float(evidence.selection.cover.selectivity)
        return PublicTaskReport(
            kind=public_kind,
            achieved=achieved,
            target=target,
            metadata_selectivity=metadata_selectivity,
        )
    return None


def _typed_objective_value(
    objective: Objective,
    evidence: SearchEvidence,
) -> float:
    if objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND and isinstance(
        evidence,
        FilteredAnnSearchEvidence,
    ):
        if objective.minimum is not None and objective.maximum is not None:
            gls = gls_from_hit_count(
                hit_count=evidence.hit_count,
                total_count=evidence.selection.cover.total_count,
                cover_count=evidence.selection.cover.count,
                k=objective.k or evidence.global_topk.rank.top_k,
            )
            if gls is None:
                raise ContractViolation(
                    "filtered ANN evidence could not resolve observed GLS"
                )
            return gls
        return float(evidence.hit_count)
    if objective.kind is ObjectiveKind.SELECTIVITY_BAND:
        if isinstance(evidence, FilteredAnnSearchEvidence):
            return evidence.selection.cover.selectivity
        if isinstance(evidence, PredicateSearchEvidence):
            return evidence.selection.cover.selectivity
    if objective.kind is ObjectiveKind.TOPK_AVAILABLE:
        if isinstance(evidence, FilteredAnnSearchEvidence):
            return float(
                min(
                    evidence.selection.cover.count,
                    evidence.global_topk.rank.top_k,
                )
            )
    raise ContractViolation(
        f"search evidence does not provide objective {objective.kind.value}"
    )


def _answer_scope_for_rank(node: RankNode) -> AnswerScope:
    field_ref = _rank_field_ref(node.score_expr)
    if field_ref is None:
        return UnscopedAnswerScope()
    return EntityAnswerScope(entity=field_ref.field.owner)


def _rank_field_ref(expr: Expr) -> FieldRef | None:
    if isinstance(expr, VectorDistanceTemplate):
        return expr.field
    return None


def _template_node_role(node: TemplateNode) -> TemplateNodeRole | None:
    if isinstance(node, ScanNode | SelectNode):
        return node.role
    return None


def _vector_metric_value(dataset: Dataset, field_id: object) -> VectorMetric:
    for entity in dataset.tables():
        for schema_field in entity.fields:
            if schema_field.id != field_id:
                continue
            if not isinstance(schema_field.type, VectorType):
                raise ContractViolation(f"field {field_id!r} is not a vector field")
            return schema_field.type.metric
    raise ContractViolation(f"vector field {field_id!r} not found in dataset schema")


def _observe_canonical_vector_topk(
    state: EvalState,
    *,
    field: FieldId,
    vector: tuple[float, ...],
    k: int,
    row_filter: RowCover | None,
) -> VectorTopKHint:
    summary = _observe_canonical(
        state,
        ObservationRequest(
            kind=ObservationRequestKind.VECTOR_TOPK,
            field=field,
            query_vector=vector,
            k=k,
            cover_payload=None if row_filter is None else CoverPayload.from_cover(cover=row_filter),
        ),
    )
    if summary.topk is None:
        raise ContractViolation("canonical vector top-k observation missing topk payload")
    return summary.topk


def _observe_canonical(state: EvalState, request: ObservationRequest) -> ObservationSummary:
    return state.observation_session.observe_canonical(
        request,
        scope=state.observation_scope,
    )


def topk_answers_from_vector_topk_hint(topk: VectorTopKHint) -> tuple[TopKAnswer, ...]:
    result: list[TopKAnswer] = []
    distances = topk.distances
    for index, rowid in enumerate(topk.rowids):
        distance = distances[index] if index < len(distances) else None
        result.append(
            TopKAnswer(
                rowid=int(rowid),
                distance=None if distance is None else float(distance),
            )
        )
    return tuple(result)


def _single_cover(node_eval: NodeEval) -> RowCover:
    covers = tuple(node_eval.scope_covers.values())
    if len(covers) != 1:
        raise ContractViolation("expected exactly one active scope cover")
    return covers[0]


def _rank_vector_request(expr: Expr, binding: ParameterBinding) -> tuple[FieldRef, tuple[float, ...]]:
    if isinstance(expr, VectorDistanceTemplate):
        value = _binding_value(binding, expr.query_vector.hole_id)
        if isinstance(value, VectorValue):
            return expr.field, tuple(float(component) for component in value.components)
        if not isinstance(value, tuple) or not all(isinstance(component, float | int) for component in value):
            raise ContractViolation("vector query hole must be a numeric tuple")
        return expr.field, tuple(float(component) for component in value)
    raise ContractViolation("rank evaluator currently supports vector_distance(field, hole)")


def _observe_canonical_predicate_cover(
    *,
    resources: DatasetResources,
    state: EvalState,
    template: QueryTemplate,
    node: TemplateNode,
    values: Mapping[str, ParameterValue],
    answer_field: FieldId,
) -> PredicateCoverHint:
    del resources
    cached = state.prepared_predicate_covers.get(node.node_id)
    if cached is not None:
        state.observation_session.record_profile_event("query.reuse.cover.prepared")
        return cached
    summary = _observe_canonical(
        state,
        ObservationRequest(
            kind=ObservationRequestKind.PREDICATE_COVER,
            field=answer_field,
            template_signature=template.structural_signature,
            cover_node=node,
            cover_values=values,
        ),
    )
    if summary.predicate_cover is None:
        raise ContractViolation("canonical predicate cover observation missing payload")
    return summary.predicate_cover


def _selection_answer_field(node: SelectNode) -> FieldId:
    field = _rank_field_in_subtree(node)
    if field is not None:
        return field
    return _scope_owner_field(node.output_scope, node.input)


def _rank_field_in_subtree(node: TemplateNode) -> FieldId | None:
    if isinstance(node, RankNode):
        field_ref = _rank_field_ref(node.score_expr)
        return None if field_ref is None else field_ref.field
    if isinstance(node, SelectNode | AggregateNode | LimitNode):
        return _rank_field_in_subtree(node.input)
    return None


def _scope_owner_field(scope_id: str, node: TemplateNode) -> FieldId:
    if isinstance(node, ScanNode) and node.scope == scope_id:
        return FieldId(owner=node.entity, name="_rowid")
    if isinstance(node, SelectNode | RankNode | AggregateNode | LimitNode):
        return _scope_owner_field(scope_id, node.input)
    raise ContractViolation(f"could not resolve answer field for scope {scope_id!r}")


def _binding_value(binding: ParameterBinding, hole_id: str) -> ParameterValue:
    try:
        return binding.values[hole_id]
    except KeyError as exc:
        raise ContractViolation(f"binding missing value for hole {hole_id}") from exc


def _public_report_priority(objective: Objective) -> int:
    if objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND:
        return 0
    if objective.kind is ObjectiveKind.SELECTIVITY_BAND:
        return 1
    return 2


__all__ = [
    "CandidateSearchIdentity",
    "CandidateJudgement",
    "Evaluator",
    "EvalState",
    "ExactSelection",
    "ExactVectorTopK",
    "FilteredAnnHitBandDiagnostic",
    "FilteredAnnSearchEvidence",
    "FilteredVectorSearchScope",
    "GlobalVectorSearchScope",
    "NoConstraintSearchEvidence",
    "NodeEval",
    "NodeEvalKey",
    "PredicateSearchEvidence",
    "PredicateCoverObservation",
    "SearchEvidence",
    "SearchFeedback",
    "SearchFeedbackKind",
    "SelectionCoverObservation",
    "TopKAnswer",
    "format_hit_band_diagnostic",
    "hit_band",
    "hit_loss",
    "resolve_hit_band",
    "topk_hit_count",
    "topk_answers_from_vector_topk_hint",
    "filtered_ann_search_evidence",
    "judge_search_evidence",
]
