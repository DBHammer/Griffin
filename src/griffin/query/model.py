"""Query generation semantic contracts for the mainline runtime."""

from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Literal, TypeAlias, Union

import msgspec

from griffin.ir.expr import Expr
from griffin.model_ids import EntityId
from griffin.query.task import (
    Objective,
    PredicateEvolutionStrategyKind,
    PublicTaskKind,
    ResultSemantics,
)


TaskId: TypeAlias = str
TemplateId: TypeAlias = str
NodeId: TypeAlias = str
ScopeId: TypeAlias = str
HoleId: TypeAlias = str
StructuralSignature: TypeAlias = str
SupportSignature: TypeAlias = str
BindingSignature: TypeAlias = str
BindingFingerprint: TypeAlias = str
SearchEpoch: TypeAlias = int
SearchOrdinal: TypeAlias = int
NumericValue: TypeAlias = int | float
DEFAULT_MAX_IN_VALUES = 32


class SearchIdentity(msgspec.Struct, frozen=True, omit_defaults=True):
    query_seed: int
    lane_key: str
    epoch: SearchEpoch
    ordinal: SearchOrdinal = 0


class QueryTargetSeedStage(enum.Enum):
    FILTERED_ANN_GLS = "filtered-ann-gls-target"


class SearchSeedStage(enum.Enum):
    PREDICATE_COVER = "predicate-cover"
    VECTOR_FALLBACK = "vector-fallback"
    VECTOR_FRONTIER = "vector-frontier"
    VECTOR_ROW_SEEDS = "vector-row-seeds"


def _derive_seed(*components: str | int | float) -> int:
    payload = json.dumps(
        components,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big", signed=False)


def derive_query_task_seed(
    root_seed: int,
    prototype_signature: str,
    occurrence_ordinal: int,
) -> int:
    return _derive_seed(
        root_seed,
        prototype_signature,
        occurrence_ordinal,
    )


def derive_query_target_seed(
    identity: SearchIdentity,
    stage: QueryTargetSeedStage,
    *components: str | int | float,
) -> int:
    """Derive a target seed that remains stable across search continuation."""
    return _derive_seed(
        identity.query_seed,
        identity.lane_key,
        stage.value,
        *components,
    )


def derive_search_seed(
    identity: SearchIdentity,
    stage: SearchSeedStage,
    *components: str | int | float,
) -> int:
    # JSON preserves component boundaries and primitive types. A delimiter-
    # joined string would allow values containing the delimiter (or `1` vs
    # `"1"`) to alias different search identities.
    return _derive_seed(
        identity.query_seed,
        identity.lane_key,
        identity.epoch,
        identity.ordinal,
        stage.value,
        *components,
    )


class VectorValue(msgspec.Struct, frozen=True, tag="vector", tag_field="value_kind"):
    components: tuple[float, ...]


class TextValue(msgspec.Struct, frozen=True, tag="text", tag_field="value_kind"):
    text: str


class NumericRangeValue(
    msgspec.Struct, frozen=True, tag="numeric_range", tag_field="value_kind"
):
    low: NumericValue
    high: NumericValue
    inclusive_low: bool = True
    inclusive_high: bool = True


class NumericComparisonValue(
    msgspec.Struct, frozen=True, tag="numeric_comparison", tag_field="value_kind"
):
    operator: str
    value: NumericValue


class BucketValue(msgspec.Struct, frozen=True, tag="bucket", tag_field="value_kind"):
    label: str


class BucketSetValue(msgspec.Struct, frozen=True, tag="bucket_set", tag_field="value_kind"):
    labels: tuple[str, ...]


class ScalarValue(msgspec.Struct, frozen=True, tag="scalar", tag_field="value_kind"):
    json_payload: str


ValueEntry = Union[
    VectorValue,
    TextValue,
    NumericRangeValue,
    NumericComparisonValue,
    BucketValue,
    BucketSetValue,
    ScalarValue,
]

ParameterValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    tuple[int, ...],
    tuple[float, ...],
    tuple[str, ...],
    ValueEntry,
]


class ScopeKind(enum.Enum):
    ENTITY_ROWS = "entity_rows"
    SCALAR_RESULT = "scalar_result"


class HoleValueKind(enum.Enum):
    SCALAR_LITERAL = "scalar_literal"
    RANGE_BOUND = "range_bound"
    RANGE_PAIR = "range_pair"
    IN_SET = "in_set"
    QUERY_VECTOR = "query_vector"
    TOP_K = "top_k"


class HoleSemanticRole(enum.Enum):
    PREDICATE_VALUE = "predicate_value"
    RANK_QUERY = "rank_query"
    LIMIT_VALUE = "limit_value"


class TemplateNodeRole(enum.Enum):
    SOURCE_SCAN = "source_scan"
    SELECTION_FILTER = "selection_filter"


class QuerySourceIntent(msgspec.Struct, frozen=True, omit_defaults=True):
    """Public provenance labels only.

    These fields are emitted for reports and diagnostics. Runtime routing must
    use `QueryTask`, `ResultSemantics`, template node types, and objective
    enums instead of these public strings.
    """

    public_family: PublicTaskKind
    public_form: str | None = None
    source_name: str | None = None


class QueryBudget(msgspec.Struct, frozen=True, omit_defaults=True):
    max_parameter_attempts: int = 64
    max_evaluations: int = 64
    max_cover_observations_per_solver_step: int | None = None
    max_cover_rejections_per_search: int | None = None
    max_cover_candidates_per_search: int | None = None


class QueryTask(msgspec.Struct, frozen=True, omit_defaults=True):
    task_id: TaskId
    public_name: str
    source_intent: QuerySourceIntent
    result_semantics: ResultSemantics
    objectives: tuple[Objective, ...]
    hard_constraints: tuple[Objective, ...] = ()
    budget: QueryBudget = QueryBudget()
    predicate_evolution_strategy: PredicateEvolutionStrategyKind = (
        PredicateEvolutionStrategyKind.STANDARD
    )
    best_so_far_fallback: bool = False
    report_labels: tuple[str, ...] = ()


class ScopeDecl(msgspec.Struct, frozen=True, omit_defaults=True):
    scope_id: ScopeId
    scope_kind: ScopeKind
    entity_ref: EntityId | None = None
    parent_scope: ScopeId | None = None
    variable_hint: str | None = None


class UnscopedAnswerScope(
    msgspec.Struct,
    frozen=True,
    tag="unscoped",
    tag_field="answer_scope_kind",
):
    pass


class EntityAnswerScope(
    msgspec.Struct,
    frozen=True,
    tag="entity",
    tag_field="answer_scope_kind",
):
    entity: EntityId


AnswerScope = Union[
    UnscopedAnswerScope,
    EntityAnswerScope,
]


class AggregateDecl(msgspec.Struct, frozen=True, omit_defaults=True):
    aggregate_expr: Expr
    group_by: tuple[Expr, ...] = ()


class ScanNode(msgspec.Struct, frozen=True, tag="scan", tag_field="node_kind", omit_defaults=True):
    node_id: NodeId
    entity: EntityId
    scope: ScopeId
    role: TemplateNodeRole


class SelectNode(msgspec.Struct, frozen=True, tag="select", tag_field="node_kind", omit_defaults=True):
    node_id: NodeId
    input: "TemplateNode"
    predicate_expr: Expr
    output_scope: ScopeId
    role: TemplateNodeRole | None = None


class RankNode(msgspec.Struct, frozen=True, tag="rank", tag_field="node_kind", omit_defaults=True):
    node_id: NodeId
    input: "TemplateNode"
    score_expr: Expr
    top_k: int
    answer_scope: ScopeId


class AggregateNode(msgspec.Struct, frozen=True, tag="aggregate", tag_field="node_kind", omit_defaults=True):
    node_id: NodeId
    input: "TemplateNode"
    aggregate: AggregateDecl
    output_scope: ScopeId


class LimitNode(msgspec.Struct, frozen=True, tag="limit", tag_field="node_kind", omit_defaults=True):
    node_id: NodeId
    input: "TemplateNode"
    k: int
    output_scope: ScopeId


TemplateNode = Union[
    ScanNode,
    SelectNode,
    RankNode,
    AggregateNode,
    LimitNode,
]


class HoleUsageSite(msgspec.Struct, frozen=True, omit_defaults=True):
    node_id: NodeId
    expr_path: tuple[str | int, ...] = ()
    role: HoleSemanticRole | None = None


class HoleConstraints(msgspec.Struct, frozen=True, omit_defaults=True):
    minimum: float | None = None
    maximum: float | None = None
    max_cardinality: int | None = None
    choices: tuple[ParameterValue, ...] = ()
    metric: str | None = None


class HoleDecl(msgspec.Struct, frozen=True, omit_defaults=True):
    hole_id: HoleId
    owner_scope: ScopeId
    value_kind: HoleValueKind
    semantic_role: HoleSemanticRole
    constraints: HoleConstraints = HoleConstraints()
    usage_sites: tuple[HoleUsageSite, ...] = ()


class QueryTemplate(msgspec.Struct, frozen=True, omit_defaults=True):
    root: TemplateNode
    scopes: tuple[ScopeDecl, ...]
    holes: tuple[HoleDecl, ...]
    result_semantics: ResultSemantics
    structural_signature: StructuralSignature
    support_signature: SupportSignature


class EnumeratedParameterSpace(
    msgspec.Struct,
    frozen=True,
    tag="enumerated",
    tag_field="space_kind",
    omit_defaults=True,
):
    hole_id: HoleId
    value_kind: HoleValueKind
    values: tuple[ParameterValue, ...] = ()


class CategoricalSetAtom(msgspec.Struct, frozen=True, omit_defaults=True):
    label: str
    local_count: int


class CategoricalSetParameterSpace(
    msgspec.Struct,
    frozen=True,
    tag="categorical_set",
    tag_field="space_kind",
    omit_defaults=True,
):
    hole_id: HoleId
    value_kind: HoleValueKind
    atoms: tuple[CategoricalSetAtom, ...] = ()
    owner_row_count: int = 0
    max_size: int = DEFAULT_MAX_IN_VALUES

    def __post_init__(self) -> None:
        if self.value_kind is not HoleValueKind.IN_SET:
            raise ValueError("categorical set parameter space requires IN_SET value kind")
        if self.owner_row_count < 0:
            raise ValueError("categorical set parameter space owner row count must be non-negative")
        if self.max_size < 1:
            raise ValueError("categorical set parameter space max size must be positive")
        if any(atom.local_count < 0 for atom in self.atoms):
            raise ValueError("categorical set parameter atom count must be non-negative")
        labels = tuple(atom.label for atom in self.atoms)
        if len(labels) != len(set(labels)):
            raise ValueError("categorical set parameter atom labels must be unique")


class OrderedRangeBoundary(msgspec.Struct, frozen=True, omit_defaults=True):
    value: NumericValue
    lower_count: int
    upper_count: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
            raise ValueError("ordered range boundary value must be finite numeric")
        if self.lower_count < 0 or self.upper_count < self.lower_count:
            raise ValueError("ordered range boundary counts must be non-negative and ordered")


class OrderedRangeParameterSpace(
    msgspec.Struct,
    frozen=True,
    tag="ordered_range",
    tag_field="space_kind",
    omit_defaults=True,
):
    hole_id: HoleId
    value_kind: HoleValueKind
    boundaries: tuple[OrderedRangeBoundary, ...] = ()
    owner_row_count: int = 0

    def __post_init__(self) -> None:
        if self.value_kind is not HoleValueKind.RANGE_PAIR:
            raise ValueError("ordered range parameter space requires RANGE_PAIR value kind")
        if self.owner_row_count < 0:
            raise ValueError("ordered range parameter space owner row count must be non-negative")
        previous: OrderedRangeBoundary | None = None
        for boundary in self.boundaries:
            if boundary.upper_count > self.owner_row_count:
                raise ValueError("ordered range boundary count exceeds owner row count")
            if previous is not None:
                if boundary.value <= previous.value:
                    raise ValueError("ordered range boundary values must be strictly increasing")
                if boundary.lower_count < previous.upper_count:
                    raise ValueError("ordered range boundary cumulative counts must be monotone")
            previous = boundary


ParameterSpace: TypeAlias = Union[
    EnumeratedParameterSpace,
    CategoricalSetParameterSpace,
    OrderedRangeParameterSpace,
]


class BindingTraceEvent(msgspec.Struct, frozen=True, omit_defaults=True):
    event_kind: str
    detail: Mapping[str, str] = msgspec.field(default_factory=dict)


class ParameterBinding(msgspec.Struct, frozen=True, omit_defaults=True):
    template_signature: StructuralSignature
    values: Mapping[HoleId, ParameterValue]
    complete_values: bool
    binding_signature: BindingSignature
    binding_trace: tuple[BindingTraceEvent, ...] = ()


class MaterializedQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    template: QueryTemplate
    binding: ParameterBinding
    lowered_render_ir: object | None = None
    lowered_eval_plan: object | None = None
    materialization_metadata: Mapping[str, str] = msgspec.field(default_factory=dict)


def template_node_id(node: TemplateNode) -> NodeId:
    return node.node_id


def iter_template_nodes(node: TemplateNode) -> tuple[TemplateNode, ...]:
    nodes: list[TemplateNode] = [node]
    if isinstance(node, (SelectNode, RankNode, AggregateNode, LimitNode)):
        nodes.extend(iter_template_nodes(node.input))
    return tuple(nodes)


def _canonical_payload(value: object) -> object:
    if isinstance(value, msgspec.Struct):
        return {
            key: _canonical_payload(getattr(value, key))
            for key in value.__struct_fields__
            if key != "structural_signature" and key != "support_signature"
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_payload(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    return value


def stable_signature(
    prefix: Literal["template", "binding", "support", "predicate_fragment", "observation"],
    payload: object,
) -> str:
    encoded = json.dumps(_canonical_payload(payload), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


def template_structural_signature(root: TemplateNode, holes: tuple[HoleDecl, ...]) -> StructuralSignature:
    return stable_signature("template", {"root": root, "holes": holes})


def binding_signature(template: QueryTemplate, values: Mapping[HoleId, ParameterValue]) -> BindingSignature:
    return stable_signature(
        "binding",
        {
            "template": template.structural_signature,
            "values": dict(values),
        },
    )


__all__ = [
    "AggregateDecl",
    "AggregateNode",
    "AnswerScope",
    "BucketSetValue",
    "BucketValue",
    "BindingFingerprint",
    "BindingSignature",
    "BindingTraceEvent",
    "CategoricalSetAtom",
    "CategoricalSetParameterSpace",
    "DEFAULT_MAX_IN_VALUES",
    "EnumeratedParameterSpace",
    "EntityAnswerScope",
    "HoleConstraints",
    "HoleDecl",
    "HoleId",
    "HoleSemanticRole",
    "HoleUsageSite",
    "HoleValueKind",
    "LimitNode",
    "MaterializedQuery",
    "NodeId",
    "NumericComparisonValue",
    "NumericRangeValue",
    "NumericValue",
    "OrderedRangeBoundary",
    "OrderedRangeParameterSpace",
    "ParameterBinding",
    "ParameterSpace",
    "ParameterValue",
    "QuerySourceIntent",
    "QueryBudget",
    "PredicateEvolutionStrategyKind",
    "QueryTask",
    "QueryTemplate",
    "RankNode",
    "ScalarValue",
    "ScanNode",
    "SearchIdentity",
    "QueryTargetSeedStage",
    "SearchSeedStage",
    "ScopeDecl",
    "ScopeId",
    "ScopeKind",
    "SelectNode",
    "StructuralSignature",
    "SupportSignature",
    "TaskId",
    "TemplateId",
    "TemplateNode",
    "TemplateNodeRole",
    "TextValue",
    "UnscopedAnswerScope",
    "ValueEntry",
    "VectorValue",
    "binding_signature",
    "derive_query_target_seed",
    "derive_query_task_seed",
    "derive_search_seed",
    "iter_template_nodes",
    "stable_signature",
    "template_node_id",
    "template_structural_signature",
]
