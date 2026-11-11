"""Parameter domain construction contracts."""

from __future__ import annotations

import enum
import math

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.dataset.deterministic import (
    CategoricalBlocksModel,
    DeterministicFieldModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    UniformNumericModel,
)
from griffin.dataset.exact_cover import (
    deterministic_owner_metadata,
    field_model_for_name,
    intervals_row_count,
)
from griffin.ir.expr import (
    BinaryOp,
    BinOp,
    FieldRef,
    HoleRef,
    RangePredicateTemplate,
    SetMembership,
    VectorDistanceTemplate,
)
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.model import (
    AggregateNode,
    CategoricalSetAtom,
    CategoricalSetParameterSpace,
    DEFAULT_MAX_IN_VALUES,
    EnumeratedParameterSpace,
    HoleId,
    HoleValueKind,
    LimitNode,
    OrderedRangeBoundary,
    OrderedRangeParameterSpace,
    ParameterSpace,
    ParameterValue,
    QueryTask,
    QueryTemplate,
    RankNode,
    ScanNode,
    SelectNode,
    TemplateNode,
)
from griffin.query.model import (
    BucketValue,
    NumericComparisonValue,
    VectorValue,
)

# 连续数值字段（uniform/lognormal）的分位切点数。512 时相邻切点的 cover 计数
# 间隔约 N/512 行，在 1% 等低选择率端部会超过 ±1% band 宽度，导致 band 内没有
# 可表达边界（与 int-affine 512 降采样同源的问题）。提升到 10000 后计数间隔约
# 100 行，足以表达 1% 档 ±100 行的 band。
NUMERIC_DOMAIN_CUTPOINT_COUNT = 10000


class HolePredicateOperator(enum.Enum):
    IN = "in"
    BETWEEN = "between"
    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class DomainBuildProblem(msgspec.Struct, frozen=True, omit_defaults=True):
    task: QueryTask
    template: QueryTemplate
    resources: DatasetResources | None = None


class DomainBuildResult(msgspec.Struct, frozen=True, omit_defaults=True):
    domains: tuple[ParameterSpace, ...]

    def domain_for(self, hole_id: HoleId) -> ParameterSpace:
        for domain in self.domains:
            if domain.hole_id == hole_id:
                return domain
        raise KeyError(f"no parameter domain for hole {hole_id}")


class DomainBuilder:
    """Build parameter value domains for template holes."""

    def build(self, problem: DomainBuildProblem) -> DomainBuildResult:
        domains = tuple(
            _space_for_hole(
                problem.resources,
                problem.task,
                problem.template,
                hole.hole_id,
                hole.value_kind,
                hole.constraints.choices,
                max_cardinality=hole.constraints.max_cardinality,
            )
            for hole in problem.template.holes
        )
        return DomainBuildResult(domains=domains)


def _space_for_hole(
    resources: DatasetResources | None,
    task: QueryTask,
    template: QueryTemplate,
    hole_id: HoleId,
    value_kind: HoleValueKind,
    default_values: tuple[ParameterValue, ...],
    *,
    max_cardinality: int | None,
) -> ParameterSpace:
    if default_values:
        return EnumeratedParameterSpace(
            hole_id=hole_id,
            value_kind=value_kind,
            values=default_values,
        )
    derived = _derive_space(
        resources,
        task,
        template.root,
        hole_id,
        value_kind,
        max_cardinality=max_cardinality,
    )
    if derived is not None:
        return derived
    return EnumeratedParameterSpace(hole_id=hole_id, value_kind=value_kind)


class HoleContext(msgspec.Struct, frozen=True, omit_defaults=True):
    field: FieldId | None = None
    operator: HolePredicateOperator | None = None


# 谓词域空间缓存：space 只由 (字段, 操作符, value_kind, max_cardinality, hole_id)
# 决定（model 由字段唯一决定、dataset 在进程内不变）。10000 切点的 OrderedRange
# 构建要做 20000 次 local_cover（~14 秒），此前每条查询重建一次，是生成的绝对
# 瓶颈（单条 15 秒里 ~14 秒在重建同一个域）。跨查询复用后构建只发生一次。
_DOMAIN_SPACE_CACHE: dict[tuple, ParameterSpace] = {}


def _derive_space(
    resources: DatasetResources | None,
    task: QueryTask,
    root: TemplateNode,
    hole_id: HoleId,
    value_kind: HoleValueKind,
    *,
    max_cardinality: int | None,
) -> ParameterSpace | None:
    if resources is None:
        return None
    context = hole_context_for_template(root, hole_id)
    if value_kind is HoleValueKind.QUERY_VECTOR:
        return _enumerated_space(
            hole_id,
            value_kind,
            _query_vector_options(resources, context),
        )
    if context.field is None:
        return None
    cache_key = (
        context.field.owner.name,
        context.field.name,
        context.operator,
        value_kind,
        max_cardinality,
        hole_id,
    )
    cached = _DOMAIN_SPACE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    space = _derive_space_core(
        resources=resources,
        context=context,
        hole_id=hole_id,
        value_kind=value_kind,
        max_cardinality=max_cardinality,
    )
    if space is not None:
        _DOMAIN_SPACE_CACHE[cache_key] = space
    return space


def _derive_space_core(
    *,
    resources: DatasetResources,
    context: "HoleContext",
    hole_id: HoleId,
    value_kind: HoleValueKind,
    max_cardinality: int | None,
) -> ParameterSpace | None:
    model = _deterministic_field_model(resources, context.field)
    if model is None:
        return None
    if value_kind is HoleValueKind.IN_SET:
        if not isinstance(model, (IntAffineBlocksModel, CategoricalBlocksModel)):
            return None
        return CategoricalSetParameterSpace(
            hole_id=hole_id,
            value_kind=value_kind,
            atoms=tuple(
                CategoricalSetAtom(
                    label=block.label,
                    local_count=max(int(block.end) - int(block.start) + 1, 0),
                )
                for block in model.blocks
            ),
            owner_row_count=int(model.row_count),
            max_size=(
                DEFAULT_MAX_IN_VALUES
                if max_cardinality is None
                else max_cardinality
            ),
        )
    if isinstance(model, IntAffineBlocksModel):
        if context.operator is HolePredicateOperator.BETWEEN:
            return _ordered_range_space(
                resources,
                hole_id,
                value_kind,
                model,
                _int_affine_cut_points(model),
            )
        if context.operator in {
            HolePredicateOperator.LT,
            HolePredicateOperator.LE,
            HolePredicateOperator.GT,
            HolePredicateOperator.GE,
        }:
            comparison_values = tuple(
                NumericComparisonValue(operator=context.operator.value, value=value)
                for value in _int_affine_cut_points(model)
            )
            return _enumerated_space(hole_id, value_kind, comparison_values)
        categorical_values = _categorical_options(
            tuple(block.label for block in model.blocks),
        )
        return _enumerated_space(hole_id, value_kind, categorical_values)
    if isinstance(model, CategoricalBlocksModel):
        # int 分类模型（n05~n07/n10/n11 等 int64 列）：支持 range 谓词（between/
        # lt/le/gt/ge），切点 = 全部 int 值（block label），cover 计算走
        # _categorical_int_range_intervals。text 分类模型保持 eq/in 枚举。
        if model.value_type == "int":
            cut_points = _categorical_int_cut_points(model)
            if context.operator is HolePredicateOperator.BETWEEN:
                return _ordered_range_space(
                    resources,
                    hole_id,
                    value_kind,
                    model,
                    cut_points,
                )
            if context.operator in {
                HolePredicateOperator.LT,
                HolePredicateOperator.LE,
                HolePredicateOperator.GT,
                HolePredicateOperator.GE,
            }:
                return _enumerated_space(
                    hole_id,
                    value_kind,
                    tuple(
                        NumericComparisonValue(operator=context.operator.value, value=value)
                        for value in cut_points
                    ),
                )
        categorical_values = _categorical_options(
            tuple(block.label for block in model.blocks),
        )
        return _enumerated_space(hole_id, value_kind, categorical_values)
    if isinstance(model, UniformNumericModel | LognormalNumericModel):
        operator = context.operator or HolePredicateOperator.LE
        if operator is HolePredicateOperator.BETWEEN:
            return _ordered_range_space(
                resources,
                hole_id,
                value_kind,
                model,
                _numeric_cut_points(model),
            )
        if operator in {
            HolePredicateOperator.LT,
            HolePredicateOperator.LE,
            HolePredicateOperator.GT,
            HolePredicateOperator.GE,
        }:
            numeric_comparison_values = tuple(
                NumericComparisonValue(operator=operator.value, value=value)
                for value in _numeric_cut_points(model)
            )
            return _enumerated_space(hole_id, value_kind, numeric_comparison_values)
    return None


def _enumerated_space(
    hole_id: HoleId,
    value_kind: HoleValueKind,
    values: tuple[ParameterValue, ...],
) -> EnumeratedParameterSpace:
    return EnumeratedParameterSpace(
        hole_id=hole_id,
        value_kind=value_kind,
        values=values,
    )


def _categorical_options(
    labels: tuple[str, ...],
) -> tuple[ParameterValue, ...]:
    return tuple(BucketValue(label=label) for label in labels)


def parameter_space_is_empty(space: ParameterSpace) -> bool:
    if isinstance(space, EnumeratedParameterSpace):
        return not space.values
    if isinstance(space, CategoricalSetParameterSpace):
        return not space.atoms
    return not space.boundaries


def enumerated_parameter_values(space: ParameterSpace) -> tuple[ParameterValue, ...]:
    if not isinstance(space, EnumeratedParameterSpace):
        raise ContractViolation(
            f"parameter space for {space.hole_id} requires structured candidate search"
        )
    return space.values


def parameter_space_logical_capacity(space: ParameterSpace, *, cap: int) -> int:
    if cap <= 0:
        return 0
    if isinstance(space, EnumeratedParameterSpace):
        return min(len(space.values), cap)
    if isinstance(space, OrderedRangeParameterSpace):
        boundary_count = len(space.boundaries)
        return min(boundary_count * (boundary_count + 1) // 2, cap)
    atom_count = len(space.atoms)
    total = 0
    for size in range(1, min(atom_count, space.max_size) + 1):
        total += math.comb(atom_count, size)
        if total >= cap:
            return cap
    return total


def parameter_space_construction_cost(space: ParameterSpace) -> int:
    if isinstance(space, EnumeratedParameterSpace):
        return len(space.values)
    if isinstance(space, CategoricalSetParameterSpace):
        return len(space.atoms)
    return len(space.boundaries)


def _int_affine_cut_points(model: IntAffineBlocksModel) -> tuple[int, ...]:
    # 全 block 分辨率，与 eq/in 路径（直接使用全部 block label）保持一致。
    # 旧的 512 rank 均匀降采样会把选择率计数的网格分辨率压到 ~N/512 行：
    # 在 1% 等低选择率端部，相邻切点的 cover 计数间隔（>1000 行）远大于
    # ±1% band 的宽度（200 行），band 内不存在任何可表达边界，谓词搜索
    # 结构性无解（表现为全部候选 selectivity_out_of_band）。
    return tuple(model.bias + model.step * rank for rank in range(len(model.blocks)))


def _ordered_range_space(
    resources: DatasetResources,
    hole_id: HoleId,
    value_kind: HoleValueKind,
    model: IntAffineBlocksModel | CategoricalBlocksModel | UniformNumericModel | LognormalNumericModel,
    cut_points: tuple[int, ...] | tuple[float, ...],
) -> OrderedRangeParameterSpace:
    boundaries = tuple(
        OrderedRangeBoundary(
            value=value,
            lower_count=intervals_row_count(
                resources.predicate_access.local_cover(
                    model,
                    NumericComparisonValue(operator="lt", value=value),
                )
            ),
            upper_count=intervals_row_count(
                resources.predicate_access.local_cover(
                    model,
                    NumericComparisonValue(operator="le", value=value),
                )
            ),
        )
        for value in sorted(set(cut_points))
    )
    return OrderedRangeParameterSpace(
        hole_id=hole_id,
        value_kind=value_kind,
        boundaries=boundaries,
        owner_row_count=int(model.row_count),
    )


def hole_context_for_template(root: TemplateNode, hole_id: HoleId) -> HoleContext:
    contexts: list[HoleContext] = []
    _collect_hole_contexts(root, hole_id, contexts)
    return contexts[0] if contexts else HoleContext()


def _collect_hole_contexts(node: TemplateNode, hole_id: HoleId, contexts: list[HoleContext]) -> None:
    if isinstance(node, SelectNode):
        context = _expr_hole_context(node.predicate_expr, hole_id)
        if context is not None:
            contexts.append(context)
        _collect_hole_contexts(node.input, hole_id, contexts)
        return
    if isinstance(node, RankNode):
        context = _expr_hole_context(node.score_expr, hole_id)
        if context is not None:
            contexts.append(context)
        _collect_hole_contexts(node.input, hole_id, contexts)
        return
    if isinstance(node, (AggregateNode, LimitNode)):
        _collect_hole_contexts(node.input, hole_id, contexts)
        return
    if isinstance(node, ScanNode):
        return


def _expr_hole_context(expr: object, hole_id: HoleId) -> HoleContext | None:
    if isinstance(expr, BinaryOp):
        if isinstance(expr.left, FieldRef) and isinstance(expr.right, HoleRef) and expr.right.hole_id == hole_id:
            return HoleContext(field=expr.left.field, operator=_hole_predicate_operator(expr.operator))
        return _expr_hole_context(expr.left, hole_id) or _expr_hole_context(expr.right, hole_id)
    if isinstance(expr, SetMembership):
        if any(isinstance(value, HoleRef) and value.hole_id == hole_id for value in expr.values):
            if isinstance(expr.expr, FieldRef):
                return HoleContext(field=expr.expr.field, operator=HolePredicateOperator.IN)
        return _expr_hole_context(expr.expr, hole_id)
    if isinstance(expr, VectorDistanceTemplate) and expr.query_vector.hole_id == hole_id:
        return HoleContext(field=expr.field.field)
    if isinstance(expr, RangePredicateTemplate) and expr.range_value.hole_id == hole_id:
        return HoleContext(field=expr.field.field, operator=HolePredicateOperator.BETWEEN)
    return None


def _hole_predicate_operator(operator: BinOp) -> HolePredicateOperator | None:
    if operator is BinOp.EQ:
        return HolePredicateOperator.EQ
    if operator is BinOp.LT:
        return HolePredicateOperator.LT
    if operator is BinOp.LE:
        return HolePredicateOperator.LE
    if operator is BinOp.GT:
        return HolePredicateOperator.GT
    if operator is BinOp.GE:
        return HolePredicateOperator.GE
    return None


def _deterministic_field_model(resources: DatasetResources, field: FieldId) -> DeterministicFieldModel | None:
    metadata = deterministic_owner_metadata(resources.dataset, field.owner)
    if metadata is None:
        return None
    try:
        return field_model_for_name(metadata, field.name)
    except ContractViolation:
        return None


def _categorical_int_cut_points(model: CategoricalBlocksModel) -> tuple[int, ...]:
    """int 分类模型的 range 切点：全部 int 值（block label），升序去重。

    切点数 = distinct 值数（n05≈1000、n07=100 等），全量保留（与 eq/in 的
    全 block 分辨率一致），不做降采样。
    """
    values = set()
    for block in model.blocks:
        try:
            v = int(block.label)
        except ValueError:
            continue
        if str(v) == block.label:
            values.add(v)
    return tuple(sorted(values))


def _numeric_cut_points(model: UniformNumericModel | LognormalNumericModel) -> tuple[float, ...]:
    if isinstance(model, UniformNumericModel):
        span = model.maximum - model.minimum
        denominator = float(NUMERIC_DOMAIN_CUTPOINT_COUNT + 1)
        return tuple(
            model.linear_k * (model.minimum + span * index / denominator) + model.linear_b
            for index in range(1, NUMERIC_DOMAIN_CUTPOINT_COUNT + 1)
        )
    mean = float(model.mean)
    sigma = max(float(model.sigma), 1e-9)
    denominator = float(NUMERIC_DOMAIN_CUTPOINT_COUNT + 1)
    return tuple(
        model.linear_k * math.exp(mean + sigma * (-3.0 + 6.0 * index / denominator)) + model.linear_b
        for index in range(1, NUMERIC_DOMAIN_CUTPOINT_COUNT + 1)
    )


def _entity_row_count(resources: DatasetResources, owner: object) -> int:
    owner_name = getattr(owner, "name", None)
    if owner_name is None:
        return 0
    for metrics in resources.dataset.metrics():
        if metrics.owner.name == owner_name:
            return int(metrics.row_count)
    return 0


def _query_vector_options(resources: DatasetResources, context: HoleContext) -> tuple[ParameterValue, ...]:
    if context.field is None:
        return ()
    source = resources.vector_oracle.vector_source_metadata(dataset=resources.dataset, field_id=context.field)
    if source.row_count <= 0:
        return ()
    row_count = int(source.row_count)
    limit = min(row_count, 128)
    if limit <= 0:
        return ()
    step = max(1, row_count // limit)
    selected = tuple(range(0, row_count, step))[:limit]
    rows = resources.vector_oracle.vector_rows(
        dataset=resources.dataset,
        field_id=context.field,
        rowids=selected,
    )
    return tuple(
        VectorValue(components=tuple(float(component) for component in row.tolist()))
        for row in rows
    )


__all__ = [
    "DomainBuildProblem",
    "DomainBuildResult",
    "DomainBuilder",
    "HoleContext",
    "enumerated_parameter_values",
    "hole_context_for_template",
    "parameter_space_construction_cost",
    "parameter_space_is_empty",
    "parameter_space_logical_capacity",
]
