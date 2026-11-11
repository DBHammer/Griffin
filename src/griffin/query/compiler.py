"""Internal compiler for query workload and template plans."""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass
from itertools import combinations

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.ir.expr import (
    AggregateExpr,
    AggregationOp,
    FieldRef,
    HoleRef,
    VectorDistanceTemplate,
)
from griffin.model_ids import EntityId
from griffin.query.lowering_expr import flat_filter_slot_to_hole, flat_predicate_to_expr
from griffin.query.model import (
    AggregateDecl,
    AggregateNode,
    HoleConstraints,
    HoleDecl,
    HoleSemanticRole,
    HoleValueKind,
    LimitNode,
    QueryBudget,
    QuerySourceIntent,
    QueryTask,
    QueryTemplate,
    RankNode,
    ScanNode,
    ScopeDecl,
    ScopeKind,
    SelectNode,
    TemplateNode,
    TemplateNodeRole,
    stable_signature,
    derive_query_task_seed,
    template_structural_signature,
)
from griffin.query.shape import (
    AggregateShape,
    FilterLogic,
    FilterSlotKind,
    FilterSlotSpec,
    PredicateAtomSpec,
    PredicateOperator,
    PredicateSearchSpace,
    SelectShape,
    VectorRetrieveShape,
)
from griffin.query.search_core import ParameterSolverKind
from griffin.query.ground_truth import compile_ground_truth_plan
from griffin.query.spec import QueryPipelineSpec, QuerySpec
from griffin.query.task import (
    FilteredAnnGLSTargetMode,
    FilteredAnnTask,
    Objective,
    ObjectiveKind,
    ObjectiveMetric,
    PublicTaskKind,
    SelectivityTask,
    TopKOutput,
    ResultSemantics,
    ResultSemanticsKind,
)
from griffin.query.workload import (
    CompiledQueryTaskPlan,
    CompiledWorkloadPlan,
    QueryTaskIdentity,
    TemplateCandidate,
    TemplatePlanningSummary,
    TemplatePredicateProfileKind,
    TemplatePredicateSlot,
    query_task_budget_from_spec,
)


class CompiledQuery(msgspec.Struct, frozen=True, omit_defaults=True):
    task: QueryTask
    templates: tuple[QueryTemplate, ...]


@dataclass(frozen=True)
class PredicateTemplatePlan:
    atoms: tuple[PredicateAtomSpec, ...]
    logics: tuple[FilterLogic, ...]  # connector 序列（长度 = len(atoms)-1）


@dataclass(frozen=True)
class _CompiledTemplateVariant:
    template: QueryTemplate
    planning: TemplatePlanningSummary


TEMPLATE_FRONTIER_MULTIPLIER = 4
MIN_TEMPLATE_FRONTIER_LIMIT = 32
MAX_TEMPLATE_FRONTIER_LIMIT = 512
FILTERED_ANN_PREDICATE_COVER_OBSERVATION_LIMIT = 50000
FILTERED_ANN_PREDICATE_COVER_REJECTION_LIMIT = 1000
FILTERED_ANN_GLS_TARGET_MODE_ENV = "MMBENCHGEN_FILTERED_ANN_GLS_TARGET_MODE"


def compile_pipeline_spec(spec: QueryPipelineSpec) -> CompiledWorkloadPlan:
    occurrence_by_prototype: dict[str, int] = {}
    query_tasks: list[CompiledQueryTaskPlan] = []
    for query_spec in spec.queries:
        query_task = _compile_task(query_spec)
        templates = _compile_template_candidates(query_spec, query_task)
        prototype_signature = _query_prototype_signature(
            spec=query_spec,
            query_task=query_task,
            templates=templates,
        )
        occurrence_ordinal = occurrence_by_prototype.get(prototype_signature, 0)
        occurrence_by_prototype[prototype_signature] = occurrence_ordinal + 1
        query_seed = derive_query_task_seed(
            spec.seed,
            prototype_signature,
            occurrence_ordinal,
        )
        query_tasks.append(
            CompiledQueryTaskPlan(
                identity=QueryTaskIdentity(
                    root_seed=spec.seed,
                    prototype_signature=prototype_signature,
                    occurrence_ordinal=occurrence_ordinal,
                    query_seed=query_seed,
                ),
                query_spec=query_spec,
                query_task=query_task,
                templates=templates,
                budget=query_task_budget_from_spec(query_spec),
            )
        )
    return CompiledWorkloadPlan(
        workload_id=spec.name,
        root_seed=spec.seed,
        query_tasks=tuple(query_tasks),
    )


def compile_query_spec(spec: QuerySpec) -> CompiledQuery:
    task = _compile_task(spec)
    templates = _compile_templates(spec, task.result_semantics)
    return CompiledQuery(task=task, templates=templates)


def _compile_task(spec: QuerySpec) -> QueryTask:
    result_semantics = _result_semantics_for(spec)
    objectives = _objectives_for(spec)
    public_kind = _public_task_kind(spec.task)
    public_form = _public_form_name(spec)
    query_budget = QueryBudget(
        max_parameter_attempts=spec.budget.max_candidates,
        max_evaluations=spec.budget.max_candidates,
    )
    if any(objective.kind is ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND for objective in objectives):
        query_budget = QueryBudget(
            max_parameter_attempts=spec.budget.max_candidates,
            max_evaluations=spec.budget.max_candidates,
            max_cover_observations_per_solver_step=FILTERED_ANN_PREDICATE_COVER_OBSERVATION_LIMIT,
            max_cover_rejections_per_search=FILTERED_ANN_PREDICATE_COVER_REJECTION_LIMIT,
            max_cover_candidates_per_search=1,
        )
    return QueryTask(
        task_id=spec.name,
        public_name=spec.name,
        source_intent=QuerySourceIntent(
            public_family=public_kind,
            public_form=public_form,
            source_name=_source_name(spec),
        ),
        result_semantics=result_semantics,
        objectives=objectives,
        budget=query_budget,
        predicate_evolution_strategy=spec.predicate_evolution_strategy,
        best_so_far_fallback=spec.best_so_far_fallback,
        report_labels=(public_kind.value, public_form or "unknown"),
    )


def _compile_templates(spec: QuerySpec, result_semantics: ResultSemantics) -> tuple[QueryTemplate, ...]:
    return tuple(
        variant.template
        for variant in _dedupe_template_variants(_compile_template_variants(spec, result_semantics))
    )


def _compile_template_candidates(spec: QuerySpec, query_task: QueryTask) -> tuple[TemplateCandidate, ...]:
    return tuple(
        TemplateCandidate(
            template_signature=variant.template.structural_signature,
            template=variant.template,
            solver_kind=_solver_kind_for_template(variant.template),
            ground_truth_plan=compile_ground_truth_plan(
                query_task,
                variant.template,
            ),
            planning=variant.planning,
        )
        for variant in _dedupe_template_variants(_compile_template_variants(spec, query_task.result_semantics))
    )


def _query_prototype_signature(
    *,
    spec: QuerySpec,
    query_task: QueryTask,
    templates: tuple[TemplateCandidate, ...],
) -> str:
    return stable_signature(
        "support",
        {
            "scope": spec.scope,
            "task": {
                "source_intent": query_task.source_intent,
                "result_semantics": query_task.result_semantics,
                "objectives": query_task.objectives,
                "hard_constraints": query_task.hard_constraints,
                "predicate_evolution_strategy": (
                    query_task.predicate_evolution_strategy
                ),
            },
            "budget": {
                "max_candidates": spec.budget.max_candidates,
            },
            "templates": tuple(candidate.template_signature for candidate in templates),
        },
    )


def _dedupe_template_variants(
    variants: tuple[_CompiledTemplateVariant, ...],
) -> tuple[_CompiledTemplateVariant, ...]:
    unique_variants: list[_CompiledTemplateVariant] = []
    seen: set[str] = set()
    for variant in variants:
        if variant.template.structural_signature in seen:
            continue
        seen.add(variant.template.structural_signature)
        unique_variants.append(variant)
    return tuple(unique_variants)


def _compile_template_variants(spec: QuerySpec, result_semantics: ResultSemantics) -> tuple[_CompiledTemplateVariant, ...]:
    shape = spec.shape
    template_limit = _template_frontier_limit(spec)
    # 模板池采样种子：spec.seed 每次 run 不同，保证组合爆炸时随机采样的池子
    # 逐 run 变化（同一 config 内所有查询共享一个池，池内已均匀覆盖字段）。
    seed = _stable_spec_seed(spec)
    if isinstance(shape, SelectShape):
        return tuple(
            _compile_select_template(shape, result_semantics, filters=filters, logics=logics)
            for filters, logics in _filter_variants(shape, template_limit=template_limit, seed=seed)
        )
    if isinstance(shape, AggregateShape):
        return tuple(
            _compile_aggregate_template(shape, result_semantics, filters=filters, logics=logics)
            for filters, logics in _filter_variants(shape, template_limit=template_limit, seed=seed)
        )
    if isinstance(shape, VectorRetrieveShape):
        return tuple(
            _compile_vector_template(spec, shape, result_semantics, filters=filters, logics=logics)
            for filters, logics in _filter_variants(shape, template_limit=template_limit, seed=seed)
        )
    raise ContractViolation(f"unsupported query shape {type(shape).__name__}")


def _stable_spec_seed(spec: QuerySpec) -> int:
    """从 spec 结构派生稳定采样种子（config 的 seed 字段不是 QuerySpec 的一部分）。

    用 spec.name（含 occurrence 编号）做 hash 种子：同一 run 内各查询实例的
    name 不同（xxx_0001/xxx_0002...），采样池逐查询变化，字段铺开最大化。
    """
    return int(hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:8], 16)


def _compile_select_template(
    shape: SelectShape,
    result_semantics: ResultSemantics,
    *,
    filters: tuple[FilterSlotSpec, ...],
    logics: tuple[FilterLogic, ...],
) -> _CompiledTemplateVariant:
    entity_scope = "entity"
    scan = ScanNode(
        node_id="scan",
        entity=shape.primary_entity,
        scope=entity_scope,
        role=TemplateNodeRole.SOURCE_SCAN,
    )
    holes = _holes_for_filters(filters, entity_scope)
    root: TemplateNode = scan
    if filters:
        root = SelectNode(
            node_id="select",
            input=scan,
            predicate_expr=flat_predicate_to_expr(filters, FilterLogic.AND, holes, entity_scope, logics=logics),
            output_scope=entity_scope,
            role=TemplateNodeRole.SELECTION_FILTER,
        )
    return _compiled_template_variant(
        template=_template(root, shape.primary_entity, holes, result_semantics),
        planning=_flat_planning_summary(filters=filters, holes=holes, logic=(logics[0] if logics else FilterLogic.AND)),
    )


def _compile_aggregate_template(
    shape: AggregateShape,
    result_semantics: ResultSemantics,
    *,
    filters: tuple[FilterSlotSpec, ...],
    logics: tuple[FilterLogic, ...],
) -> _CompiledTemplateVariant:
    entity_scope = "entity"
    scalar_scope = "scalar"
    scan = ScanNode(
        node_id="scan",
        entity=shape.primary_entity,
        scope=entity_scope,
        role=TemplateNodeRole.SOURCE_SCAN,
    )
    holes = _holes_for_filters(filters, entity_scope)
    input_node: TemplateNode = scan
    if filters:
        input_node = SelectNode(
            node_id="select",
            input=scan,
            predicate_expr=flat_predicate_to_expr(filters, FilterLogic.AND, holes, entity_scope, logics=logics),
            output_scope=entity_scope,
            role=TemplateNodeRole.SELECTION_FILTER,
        )
    root: TemplateNode = AggregateNode(
        node_id="aggregate",
        input=input_node,
        aggregate=AggregateDecl(
            aggregate_expr=AggregateExpr(operator=AggregationOp.COUNT_STAR),
        ),
        output_scope=scalar_scope,
    )
    return _compiled_template_variant(
        template=_template(root, shape.primary_entity, holes, result_semantics, scalar_scope=scalar_scope),
        planning=_flat_planning_summary(filters=filters, holes=holes, logic=(logics[0] if logics else FilterLogic.AND)),
    )


def _compile_vector_template(
    spec: QuerySpec,
    shape: VectorRetrieveShape,
    result_semantics: ResultSemantics,
    *,
    filters: tuple[FilterSlotSpec, ...],
    logics: tuple[FilterLogic, ...],
) -> _CompiledTemplateVariant:
    entity_scope = "entity"
    answer_scope = "answer"
    k = _top_k(spec)
    scan = ScanNode(
        node_id="scan",
        entity=shape.primary_entity,
        scope=entity_scope,
        role=TemplateNodeRole.SOURCE_SCAN,
    )
    holes = list(_holes_for_filters(filters, entity_scope))
    input_node: TemplateNode = scan
    if filters:
        input_node = SelectNode(
            node_id="select",
            input=scan,
            predicate_expr=flat_predicate_to_expr(filters, FilterLogic.AND, tuple(holes), entity_scope, logics=logics),
            output_scope=entity_scope,
            role=TemplateNodeRole.SELECTION_FILTER,
        )
    vector_hole = HoleDecl(
        hole_id=f"{entity_scope}:query_vector",
        owner_scope=entity_scope,
        value_kind=HoleValueKind.QUERY_VECTOR,
        semantic_role=HoleSemanticRole.RANK_QUERY,
        constraints=HoleConstraints(),
    )
    holes.append(vector_hole)
    rank = RankNode(
        node_id="rank",
        input=input_node,
        score_expr=VectorDistanceTemplate(
            field=FieldRef(field=shape.target_field, variable=entity_scope),
            query_vector=HoleRef(hole_id=vector_hole.hole_id),
        ),
        top_k=k,
        answer_scope=answer_scope,
    )
    root = LimitNode(
        node_id="limit",
        input=rank,
        k=k,
        output_scope=answer_scope,
    )
    predicate_holes = tuple(holes[: len(filters)])
    return _compiled_template_variant(
        template=_template(root, shape.primary_entity, tuple(holes), result_semantics, answer_scope=answer_scope),
        planning=_flat_planning_summary(filters=filters, holes=predicate_holes, logic=(logics[0] if logics else FilterLogic.AND)),
    )


def _template(
    root: TemplateNode,
    entity: EntityId,
    holes: tuple[HoleDecl, ...],
    result_semantics: ResultSemantics,
    *,
    scalar_scope: str | None = None,
    answer_scope: str | None = None,
) -> QueryTemplate:
    scopes = [
        ScopeDecl(
            scope_id="entity",
            scope_kind=ScopeKind.ENTITY_ROWS,
            entity_ref=entity,
            variable_hint="q",
        )
    ]
    if scalar_scope is not None:
        scopes.append(ScopeDecl(scope_id=scalar_scope, scope_kind=ScopeKind.SCALAR_RESULT))
    if answer_scope is not None:
        scopes.append(
            ScopeDecl(
                scope_id=answer_scope,
                scope_kind=ScopeKind.ENTITY_ROWS,
                entity_ref=entity,
            )
        )
    structural_signature = template_structural_signature(root, holes)
    return QueryTemplate(
        root=root,
        scopes=tuple(scopes),
        holes=holes,
        result_semantics=result_semantics,
        structural_signature=structural_signature,
        support_signature=stable_signature("support", {"entity": entity, "result": result_semantics.kind.value}),
    )


def _compiled_template_variant(
    *,
    template: QueryTemplate,
    planning: TemplatePlanningSummary,
) -> _CompiledTemplateVariant:
    return _CompiledTemplateVariant(template=template, planning=planning)


def _flat_planning_summary(
    *,
    filters: tuple[FilterSlotSpec, ...],
    holes: tuple[HoleDecl, ...],
    logic: FilterLogic,
) -> TemplatePlanningSummary:
    if not filters:
        return TemplatePlanningSummary()
    if len(filters) != len(holes):
        raise ContractViolation("flat planning summary requires one predicate hole per filter slot")
    return TemplatePlanningSummary(
        predicate_profile_kind=TemplatePredicateProfileKind.FLAT,
        predicate_slots=tuple(
            TemplatePredicateSlot(
                hole_id=hole.hole_id,
                field=slot.field,
            )
            for slot, hole in zip(filters, holes, strict=True)
        ),
        filter_logic=logic,
    )


def _solver_kind_for_template(template: QueryTemplate) -> ParameterSolverKind:
    hole_kinds = {hole.value_kind for hole in template.holes}
    if HoleValueKind.QUERY_VECTOR in hole_kinds:
        return ParameterSolverKind.VECTOR
    if not hole_kinds:
        return ParameterSolverKind.FIXED
    return ParameterSolverKind.PREDICATE_COVER


def _holes_for_filters(filters: tuple[FilterSlotSpec, ...], scope_id: str) -> tuple[HoleDecl, ...]:
    return tuple(
        flat_filter_slot_to_hole(slot, scope_id, index)
        for index, slot in enumerate(filters)
    )


def _filter_variants(
    shape: SelectShape | AggregateShape | VectorRetrieveShape,
    *,
    template_limit: int,
    seed: int = 0,
) -> tuple[tuple[tuple[FilterSlotSpec, ...], tuple[FilterLogic, ...]], ...]:
    if shape.filters:
        logics = (shape.filter_logic,) * max(len(shape.filters) - 1, 0)
        return ((shape.filters, logics),)
    if shape.predicate_search is None:
        return (((), (FilterLogic.AND,)),)
    variants: list[tuple[tuple[FilterSlotSpec, ...], tuple[FilterLogic, ...]]] = []
    for plan in _predicate_template_plans(
        shape.predicate_search,
        template_limit=template_limit,
        seed=seed,
    ):
        variants.append(
            (
                tuple(
                    _filter_slot_for_atom(
                        atom,
                        max_in_values=shape.predicate_search.max_in_values,
                    )
                    for atom in plan.atoms
                ),
                plan.logics,
            )
        )
    if not variants:
        raise ContractViolation("predicate search produced no fixed template variants")
    return tuple(variants)


def _predicate_template_plans(
    search_space: PredicateSearchSpace,
    *,
    template_limit: int,
    seed: int = 0,
) -> tuple[PredicateTemplatePlan, ...]:
    atoms = search_space.atoms
    if not atoms:
        raise ContractViolation("predicate_search requires a compatible predicate atom pool")
    min_count = search_space.predicate_count.min
    max_count = min(search_space.predicate_count.max, len(atoms))
    if min_count > max_count:
        raise ContractViolation("predicate_count cannot be satisfied by compatible predicate atoms")
    raw_variants = _predicate_template_plan_frontier(
        atoms,
        min_count=min_count,
        max_count=max_count,
        template_limit=template_limit,
        seed=seed,
    )
    variants: list[PredicateTemplatePlan] = []
    seen: set[tuple[tuple[tuple[str, str, str], ...], tuple[str, ...]]] = set()
    for atom_tuple, logics in raw_variants:
        key = (_predicate_plan_key(atom_tuple), tuple(lg.value for lg in logics))
        if key in seen:
            continue
        seen.add(key)
        variants.append(PredicateTemplatePlan(atoms=atom_tuple, logics=logics))
    return tuple(variants)


def _predicate_template_plan_frontier(
    atoms: tuple[PredicateAtomSpec, ...],
    *,
    min_count: int,
    max_count: int,
    template_limit: int,
    seed: int = 0,
) -> tuple[tuple[tuple[PredicateAtomSpec, ...], tuple[FilterLogic, ...]], ...]:
    limit = max(int(template_limit), 1)
    bounded: list[tuple[tuple[PredicateAtomSpec, ...], tuple[FilterLogic, ...]]] = []
    exhaustive_limit = min(limit * 8, 4096)
    total_estimate = _predicate_template_plan_count_estimate(
        atoms,
        min_count=min_count,
        max_count=max_count,
        limit=exhaustive_limit + 1,
    )
    ordered_atoms = tuple(sorted(atoms, key=_predicate_atom_order_key))
    # 按字段数均分配额：否则 count=1/2 的组合就把 limit 占满提前返回，
    # count=3~5 的模板永远不进池（实测负载字段数只有 1~2）。
    n_counts = max(max_count - min_count + 1, 1)
    per_count_limit = max(limit // n_counts, 8)
    for count in range(min_count, max_count + 1):
        if total_estimate <= exhaustive_limit:
            atom_groups: tuple[tuple[PredicateAtomSpec, ...], ...] = tuple(
                combinations(ordered_atoms, count)
            )
        elif count == 1:
            # 单原子组合数 = 原子数（不爆炸），全保留：保证每个字段×操作符
            #（含排序靠后的 int64 字段）都进模板池。
            atom_groups = tuple(combinations(ordered_atoms, count))
        else:
            atom_groups = _sampled_combinations(
                ordered_atoms, count, limit=per_count_limit, seed=seed + count
            )
        count_before = len(bounded)
        for atom_group in atom_groups:
            atom_tuple = tuple(atom_group)
            for logics in _predicate_logic_variants(atom_tuple, seed=seed + count):
                bounded.append((atom_tuple, logics))
                if len(bounded) - count_before >= per_count_limit:
                    break
            if len(bounded) - count_before >= per_count_limit:
                break
    return tuple(bounded)


def _sampled_combinations(
    atoms: tuple[PredicateAtomSpec, ...],
    count: int,
    *,
    limit: int,
    seed: int,
) -> tuple[tuple[PredicateAtomSpec, ...], ...]:
    """组合数爆炸时按 seed 确定性随机采样（替代旧的"排序取前 N"截断）。

    旧的 _bounded_combinations 按 (操作符权重, 字段名) 排序取前 limit 个组合，
    排序靠后的字段（如 int64 列的组合）被系统性排除在模板池外，负载字段
    无法铺开。随机采样让所有字段×操作符组合都有机会入池，seed 确定性保证
    同一 config 复现同一池。
    """
    import random as _random

    total = math.comb(len(atoms), count)
    if total <= limit:
        return tuple(combinations(atoms, count))
    rng = _random.Random(seed)
    picked: set[tuple[int, ...]] = set()
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    selected: list[tuple[PredicateAtomSpec, ...]] = []
    attempts = 0
    max_attempts = limit * 20
    while len(selected) < limit and attempts < max_attempts:
        attempts += 1
        indices = tuple(sorted(rng.sample(range(len(atoms)), count)))
        if indices in picked:
            continue
        picked.add(indices)
        group = tuple(atoms[i] for i in indices)
        key = _predicate_plan_key(group)
        if key in seen:
            continue
        seen.add(key)
        selected.append(group)
    return tuple(selected)


def _predicate_template_plan_count_estimate(
    atoms: tuple[PredicateAtomSpec, ...],
    *,
    min_count: int,
    max_count: int,
    limit: int,
) -> int:
    total = 0
    for count in range(min_count, max_count + 1):
        for atom_group in combinations(atoms, count):
            total += len(_predicate_logic_variants(tuple(atom_group)))
            if total >= limit:
                return total
    return total


def _predicate_atom_order_key(
    atom: PredicateAtomSpec,
) -> tuple[int, str, str]:
    return (
        _predicate_operator_rank((atom,)),
        atom.field.owner.name,
        atom.field.name,
    )


def _bounded_combinations(
    atoms: tuple[PredicateAtomSpec, ...],
    count: int,
    *,
    limit: int,
) -> tuple[tuple[PredicateAtomSpec, ...], ...]:
    if count <= 0 or len(atoms) < count or limit <= 0:
        return ()
    selected: list[tuple[PredicateAtomSpec, ...]] = []
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    for atom_group in combinations(atoms, count):
        key = _predicate_plan_key(atom_group)
        if key in seen:
            continue
        seen.add(key)
        selected.append(tuple(atom_group))
        if len(selected) >= limit:
            break
    if len(selected) < limit and len(atoms) > count:
        tail_atoms = tuple(reversed(atoms))
        for atom_group in combinations(tail_atoms, count):
            normalized = tuple(reversed(atom_group))
            key = _predicate_plan_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            selected.append(normalized)
            if len(selected) >= limit:
                break
    return tuple(selected)


def _predicate_operator_rank(atoms: tuple[PredicateAtomSpec, ...]) -> int:
    ranks = {
        PredicateOperator.BETWEEN: 0,
        PredicateOperator.LE: 1,
        PredicateOperator.GE: 1,
        PredicateOperator.LT: 2,
        PredicateOperator.GT: 2,
        PredicateOperator.EQ: 3,
        PredicateOperator.IN: 4,
    }
    return sum(ranks.get(atom.operator, 10) for atom in atoms)


def _template_frontier_limit(spec: QuerySpec) -> int:
    if os.environ.get("MMBENCHGEN_FORCE_SINGLE_TEMPLATE") == "1":
        # 对比实验：只取确定性第一个模板，保证 Griffin 与 baseline 共享同一
        # predicate 结构，仅参数填充策略不同。
        return 1
    return min(
        max(int(spec.budget.max_candidates) * TEMPLATE_FRONTIER_MULTIPLIER, MIN_TEMPLATE_FRONTIER_LIMIT),
        MAX_TEMPLATE_FRONTIER_LIMIT,
    )


def _predicate_logic_variants(
    atoms: tuple[PredicateAtomSpec, ...],
    *,
    seed: int = 0,
) -> tuple[tuple[FilterLogic, ...], ...]:
    """connector 序列变体：全 AND、全 OR、以及一个 seed 决定的混合序列。"""
    n = len(atoms)
    if n <= 1:
        return ((FilterLogic.AND,),)
    rng = random.Random(seed)
    mixed = tuple(
        rng.choice((FilterLogic.AND, FilterLogic.OR)) for _ in range(n - 1)
    )
    return (
        (FilterLogic.AND,) * (n - 1),
        (FilterLogic.OR,) * (n - 1),
        mixed,
    )


def _predicate_plan_key(atoms: tuple[PredicateAtomSpec, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (atom.field.owner.name, atom.field.name, atom.operator.value)
        for atom in atoms
    )


def _filter_slot_for_atom(
    atom: PredicateAtomSpec,
    *,
    max_in_values: int,
) -> FilterSlotSpec:
    return FilterSlotSpec(
        field=atom.field,
        kind=_filter_kind_for_operator(atom.operator),
        operator=atom.operator,
        max_in_values=max_in_values if atom.operator is PredicateOperator.IN else None,
    )


def _filter_kind_for_operator(operator: PredicateOperator) -> FilterSlotKind:
    if operator in {PredicateOperator.EQ, PredicateOperator.IN}:
        return FilterSlotKind.SCALAR_BUCKET
    return FilterSlotKind.SCALAR_RANGE


def _result_semantics_for(spec: QuerySpec) -> ResultSemantics:
    if isinstance(spec.output, TopKOutput):
        return ResultSemantics(kind=ResultSemanticsKind.RANKED_ENTITIES, k=spec.output.k)
    if isinstance(spec.shape, AggregateShape):
        return ResultSemantics(kind=ResultSemanticsKind.SCALAR_COUNT)
    return ResultSemantics(kind=ResultSemanticsKind.ENTITY_SET)


def _objectives_for(spec: QuerySpec) -> tuple[Objective, ...]:
    task = spec.task
    objectives: list[Objective] = []
    if isinstance(task, SelectivityTask):
        objectives.append(
            _band_objective(
                ObjectiveKind.SELECTIVITY_BAND,
                ObjectiveMetric.SELECTIVITY,
                task.target.target,
                task.target.tolerance,
            )
        )
    elif isinstance(task, FilteredAnnTask):
        if task.selectivity is not None:
            objectives.append(
                _band_objective(
                    ObjectiveKind.SELECTIVITY_BAND,
                    ObjectiveMetric.SELECTIVITY,
                    task.selectivity.target,
                    task.selectivity.tolerance,
                )
            )
        if task.gls is not None:
            objectives.append(
                Objective(
                    kind=ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND,
                    metric=ObjectiveMetric.FILTERED_ANN_HIT_COUNT,
                    minimum=task.gls.target - task.gls.tolerance,
                    maximum=task.gls.target + task.gls.tolerance,
                    hardness=task.difficulty,
                    k=_top_k(spec),
                )
            )
        elif (gls_mode := _filtered_ann_gls_target_mode_from_env()) is not None:
            objectives.append(
                Objective(
                    kind=ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND,
                    metric=ObjectiveMetric.FILTERED_ANN_HIT_COUNT,
                    filtered_ann_gls_target_mode=gls_mode,
                    hardness=task.difficulty,
                    k=_top_k(spec),
                )
            )
        else:
            objectives.append(
                Objective(
                    kind=ObjectiveKind.FILTERED_ANN_DIFFICULTY_BAND,
                    metric=ObjectiveMetric.FILTERED_ANN_HIT_COUNT,
                    hardness=task.difficulty,
                    k=_top_k(spec),
                )
            )
    if isinstance(spec.output, TopKOutput):
        objectives.append(Objective(kind=ObjectiveKind.TOPK_AVAILABLE, metric=ObjectiveMetric.TOPK_AVAILABLE, k=spec.output.k))
    return tuple(objectives)


def _filtered_ann_gls_target_mode_from_env() -> FilteredAnnGLSTargetMode | None:
    raw = os.getenv(FILTERED_ANN_GLS_TARGET_MODE_ENV)
    if raw is None or not raw.strip():
        return None
    token = raw.strip().lower()
    for mode in FilteredAnnGLSTargetMode:
        if mode.value == token:
            return mode
    allowed = ", ".join(mode.value for mode in FilteredAnnGLSTargetMode)
    raise ContractViolation(f"{FILTERED_ANN_GLS_TARGET_MODE_ENV} must be one of [{allowed}], got {raw!r}")


def _band_objective(
    kind: ObjectiveKind,
    metric: ObjectiveMetric,
    target: float,
    tolerance: float,
) -> Objective:
    return Objective(
        kind=kind,
        metric=metric,
        minimum=target - tolerance,
        maximum=target + tolerance,
    )


def _public_task_kind(task: object) -> PublicTaskKind:
    if isinstance(task, SelectivityTask):
        return PublicTaskKind.SELECTIVITY
    if isinstance(task, FilteredAnnTask):
        return PublicTaskKind.FILTERED_ANN
    raise ContractViolation(f"unsupported query task type {type(task).__name__}")


def _public_form_name(spec: QuerySpec) -> str | None:
    tag = getattr(spec.shape, "__struct_tag__", None)
    if tag is None:
        return None
    return str(tag)


def _top_k(spec: QuerySpec) -> int:
    if not isinstance(spec.output, TopKOutput):
        raise ContractViolation("ranked query requires TopKOutput")
    return spec.output.k


def _source_name(spec: QuerySpec) -> str | None:
    entity = spec.scope.primary_entity
    return getattr(entity, "name", None)


__all__ = [
    "CompiledQuery",
    "compile_pipeline_spec",
    "compile_query_spec",
]
