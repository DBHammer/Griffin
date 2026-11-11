from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from griffin.data.generation_request import (
    CategoricalAssociationRequest,
    CorrelationBlockRequest,
    DataGenerationRequest,
    DistributionRequest,
    FieldGenerationRequest,
    IntValueDomainRequest,
    TableGenerationRequest,
    VectorAttributeCouplingRequest,
    VectorGenerationRequest,
)
from griffin.data.schema import RelationalSchemaExport
from griffin.data.pipeline import DataPipelineSpec, SizingSpec
from griffin.dataset.entities.table import Table
from griffin.dataset.deterministic import IntAffineBlocksModel
from griffin.dataset.exact_cover import deterministic_owner_metadata, field_model_for_name
from griffin.dataset.reader import Dataset
from griffin.dataset.relations.base import RelationSpec
from griffin.query.model import DEFAULT_MAX_IN_VALUES
from griffin.dialects.sql.postgresql import POSTGRESQL_CAPABILITIES
from griffin.infra.errors import ContractViolation
from griffin.model_ids import FieldId, TableId
from griffin.query.task import (
    PredicateEvolutionStrategyKind,
    TopKOutput,
    TypedOutput,
)
from griffin.query.pipeline import QueryPipelineSpec
from griffin.query.spec import DialectTarget
from griffin.query.shape import (
    PredicateAtomSpec,
    PredicateCountRange,
    PredicateOperator,
    PredicateSearchSpace,
)
from griffin.query.shape import VectorRetrieveShape
from griffin.query.shape import SelectShape
from griffin.query.spec import BudgetSpec, QuerySpec, ScopePayload
from griffin.query.task import Hardness
from griffin.query.task import FilteredAnnTask, SelectivityTask, TypedTask
from griffin.query.task import ToleranceTarget
from griffin.ir.data_type import DataType, DoubleType, IntType, TextType, VectorMetric, VectorType
from griffin.ir.field import Field, FieldRole
from .contract import (
    correlation_pairs_to_matrix,
    normalize_type_name,
    validate_relational_spec,
)

from .models import (
    ConstraintConfig,
    ConstraintMetric,
    DataJobConfig,
    DataModelKind,
    Difficulty,
    CategoricalAssociationBlockConfig,
    NumericCorrelationBlockConfig,
    PredicateCountRangeConfig,
    QueryFamily,
    QueryFormConfig,
    QueryJobConfig,
    QueryOutputKind,
    QuerySpecConfig,
    RelationalColumnConfig,
    RelationalDataJobConfig,
    RelationalDataSpecConfig,
    RelationalTableConfig,
    VectorAttributeCouplingConfig,
    DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY,
)


MAX_QUERY_COUNT = 10_000


def compile_data_job(config: DataJobConfig) -> DataPipelineSpec:
    return _compile_relational_data_job(config, config.spec)


def compile_query_job(config: QueryJobConfig, dataset: Dataset) -> QueryPipelineSpec:
    if config.spec.count < 1:
        raise ContractViolation("spec.count must be >= 1")
    if config.spec.count > MAX_QUERY_COUNT:
        raise ContractViolation(f"spec.count must be <= {MAX_QUERY_COUNT}")
    output_root = _resolve_query_output_root(config, dataset)
    public_dialect = _public_query_dialect(dataset)
    query_specs = _compile_query_specs(config.spec, dataset)
    predicate_evolution_strategy = _query_predicate_evolution_strategy(config.spec)
    if (
        config.spec.search is not None
        and config.spec.search.predicate_evolution_strategy is not None
        and any(
            not _query_spec_uses_predicate_evolution(query_spec)
            for query_spec in query_specs
        )
    ):
        raise ContractViolation(
            "spec.search.predicate_evolution_strategy requires predicate search"
        )
    max_attempts = _query_max_attempts(config.spec)
    best_so_far_fallback = _query_best_so_far_fallback(config.spec)
    budgeted = tuple(
        QuerySpec(
            name=query_spec.name,
            scope=query_spec.scope,
            shape=query_spec.shape,
            task=query_spec.task,
            output=query_spec.output,
            budget=BudgetSpec(max_candidates=max_attempts),
            predicate_evolution_strategy=predicate_evolution_strategy,
            best_so_far_fallback=best_so_far_fallback,
        )
        for query_spec in query_specs
    )
    queries = tuple(
        _query_instance_spec(
            budgeted[index % len(budgeted)],
            index=index,
            total_count=config.spec.count,
        )
        for index in range(config.spec.count)
    )
    return QueryPipelineSpec(
        name=Path(output_root).name,
        seed=config.seed,
        dataset_root=dataset.root,
        dialect=public_dialect,
        queries=queries,
    )


def _query_instance_spec(spec: QuerySpec, *, index: int, total_count: int) -> QuerySpec:
    name = spec.name if total_count == 1 else f"{spec.name}_{index + 1:04d}"
    return QuerySpec(
        name=name,
        scope=spec.scope,
        shape=spec.shape,
        task=spec.task,
        output=spec.output,
        budget=spec.budget,
        predicate_evolution_strategy=spec.predicate_evolution_strategy,
        best_so_far_fallback=spec.best_so_far_fallback,
    )


def _query_max_attempts(config: QuerySpecConfig) -> int:
    attempts = (
        DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY
        if config.search is None
        else config.search.max_attempts_per_query
    )
    if attempts <= 0:
        raise ContractViolation("spec.search.max_attempts_per_query must be positive")
    return attempts


def _query_predicate_evolution_strategy(
    config: QuerySpecConfig,
) -> PredicateEvolutionStrategyKind:
    if (
        config.search is None
        or config.search.predicate_evolution_strategy is None
    ):
        return PredicateEvolutionStrategyKind.STANDARD
    return config.search.predicate_evolution_strategy


def _query_best_so_far_fallback(config: QuerySpecConfig) -> bool:
    if config.search is None:
        return False
    return bool(config.search.best_so_far_fallback)


def _query_spec_uses_predicate_evolution(spec: QuerySpec) -> bool:
    return spec.shape.predicate_search is not None


def _compile_relational_data_job(
    config: RelationalDataJobConfig,
    spec: RelationalDataSpecConfig,
) -> DataPipelineSpec:
    validate_relational_spec(spec)
    tables: list[Table] = []
    sizing: list[SizingSpec] = []
    relation_specs: list[RelationSpec] = []
    table_requests: list[TableGenerationRequest] = []
    for table_cfg in spec.tables:
        table_id = TableId(name=table_cfg.name)
        fields: list[Field] = []
        primary_key: list[FieldId] = []
        field_lookup: dict[str, FieldId] = {}
        for column in table_cfg.columns:
            field = _compile_relational_field(table_id, column)
            fields.append(field)
            field_lookup[field.id.name] = field.id
            if column.primary_key:
                primary_key.append(field.id)
        tables.append(
            Table(
                id=table_id,
                fields=tuple(fields),
                primary_key=tuple(primary_key),
            )
        )
        sizing.append(SizingSpec(entity_name=table_cfg.name, row_count=table_cfg.rows))
        relation_specs.extend(_compile_table_relations(table_cfg, table_id, field_lookup))
        table_requests.append(_compile_table_generation_request(table_cfg, table_id))
    return DataPipelineSpec(
        name=Path(config.output_dir).name or "dataset",
        tables=tuple(tables),
        sizing=tuple(sizing),
        data_model=DataModelKind.RELATIONAL.value,
        output_format=config.output_format.value,
        dialect=config.dialect.value,
        relation_specs=tuple(relation_specs),
        generation=DataGenerationRequest(tables=tuple(table_requests)),
        schema_export=RelationalSchemaExport(dialect=config.dialect.value),
        seed=config.seed,
    )


def _compile_relational_field(
    table_id: TableId,
    column: RelationalColumnConfig,
) -> Field:
    field_id = FieldId(owner=table_id, name=column.name)
    vector_spec = column.vector_spec
    data_type = _compile_data_type(
        column.type_name,
        vector_dim=(vector_spec.dim if vector_spec is not None else None),
        metric=(vector_spec.metric if vector_spec is not None else None),
    )
    role = _relational_role_for(column)
    return Field(id=field_id, type=data_type, role=role)


def _compile_table_relations(
    table_cfg: RelationalTableConfig,
    table_id: TableId,
    field_lookup: dict[str, FieldId],
) -> list[RelationSpec]:
    return _compile_owner_relations(table_cfg.numeric_correlation_blocks, field_lookup)


def _compile_owner_relations(
    blocks: Sequence[NumericCorrelationBlockConfig],
    field_lookup: Mapping[str, FieldId],
) -> list[RelationSpec]:
    relations: list[RelationSpec] = []
    for block in blocks:
        if len(block.columns) >= 2:
            fields = {
                column: field_lookup[column]
                for column in block.columns
                if column in field_lookup
            }
            for pair in block.pairs:
                relations.append(
                    RelationSpec(
                        kind="correlation",
                        name=f"{block.name}_{pair.left}_{pair.right}",
                        participants=(fields[pair.left], fields[pair.right]),
                    )
                )
            if len(fields) >= 2:
                relations.append(
                    RelationSpec(
                        kind="coupling",
                        name=f"{block.name}_coupling",
                        participants=tuple(fields[column] for column in block.columns),
                    )
                )
    return relations


def _compile_table_generation_request(
    table_cfg: RelationalTableConfig,
    table_id: TableId,
) -> TableGenerationRequest:
    vector_column = next(
        (column for column in table_cfg.columns if normalize_type_name(column.type_name) == "vector"),
        None,
    )
    vector_request = None
    if vector_column is not None and vector_column.vector_spec is not None:
        vector_spec = vector_column.vector_spec
        vector_request = VectorGenerationRequest(
            field_name=vector_column.name,
            dimension=vector_spec.dim,
            metric=vector_spec.metric,
            normalize=vector_spec.normalize,
            cluster_count=vector_spec.cluster_count,
            cluster_size_distribution=vector_spec.cluster_size_distribution,
            cluster_separation=vector_spec.cluster_separation,
            cluster_spread=vector_spec.cluster_spread,
            vector_length_variation=vector_spec.vector_length_variation,
            region_count=vector_spec.region_count,
            region_assignment_backend=vector_spec.region_assignment_backend,
            region_clustering_backend=vector_spec.region_clustering_backend,
        )
    return TableGenerationRequest(
        id=table_id,
        fields=tuple(
            _compile_field_request(
                column.name,
                column.distribution,
            )
            for column in table_cfg.columns
            if normalize_type_name(column.type_name) != "vector"
        ),
        vector=vector_request,
        correlation_blocks=_compile_correlation_requests(table_cfg.numeric_correlation_blocks),
        categorical_association_blocks=_compile_categorical_association_requests(table_cfg.categorical_association_blocks),
        vector_attribute_couplings=_compile_vector_couplings(table_cfg.vector_attribute_couplings),
    )


def _compile_correlation_requests(
    blocks: Sequence[NumericCorrelationBlockConfig],
) -> tuple[CorrelationBlockRequest, ...]:
    return tuple(
        CorrelationBlockRequest(
            name=block.name,
            columns=block.columns,
            matrix=correlation_pairs_to_matrix(
                columns=block.columns,
                pairs=block.pairs,
            ),
        )
        for block in blocks
    )


def _compile_categorical_association_requests(
    blocks: Sequence[CategoricalAssociationBlockConfig],
) -> tuple[CategoricalAssociationRequest, ...]:
    return tuple(
        CategoricalAssociationRequest(
            name=block.name,
            columns=(block.columns[0], block.columns[1]),
            target_cramers_v=block.target_cramers_v,
        )
        for block in blocks
    )


def _compile_vector_couplings(
    couplings: Sequence[VectorAttributeCouplingConfig],
) -> tuple[VectorAttributeCouplingRequest, ...]:
    return tuple(
        VectorAttributeCouplingRequest(
            name=coupling.name,
            vector_field=coupling.vector,
            attribute_field=coupling.attribute,
            target_rho=coupling.target_rho,
            max_groups=coupling.max_groups,
        )
        for coupling in couplings
    )


def _compile_field_request(
    name: str,
    distribution: object,
) -> FieldGenerationRequest:
    dist_request = None
    if distribution is not None:
        dist_request = _compile_distribution_request(distribution)
    return FieldGenerationRequest(
        field_name=name,
        distribution=dist_request,
    )


def _compile_distribution_request(distribution: object) -> DistributionRequest:
    value_domain = getattr(distribution, "value_domain")
    return DistributionRequest(
        kind=getattr(distribution, "kind").value,
        min=getattr(distribution, "min"),
        max=getattr(distribution, "max"),
        mean=getattr(distribution, "mean"),
        sigma=getattr(distribution, "sigma"),
        values=getattr(distribution, "values"),
        weights=getattr(distribution, "weights"),
        ndv=getattr(distribution, "ndv"),
        s=getattr(distribution, "s"),
        value_domain=(
            IntValueDomainRequest(
                min=getattr(value_domain, "min"),
                max=getattr(value_domain, "max"),
            )
            if value_domain is not None
            else None
        ),
    )


def _compile_query_specs(config: QuerySpecConfig, dataset: Dataset) -> tuple[QuerySpec, ...]:
    if config.family is QueryFamily.RELATIONAL:
        return (_compile_relational_query(config, dataset),)
    if config.family is QueryFamily.FILTERED_ANN:
        return (_compile_filtered_ann_query(config, dataset),)
    raise ContractViolation(f"unsupported query family: {config.family.value}")


def _compile_relational_query(config: QuerySpecConfig, dataset: Dataset) -> QuerySpec:
    if config.form is None or config.form.kind is not QueryFamily.RELATIONAL:
        raise ContractViolation("relational query requires spec.form.kind=relational")
    _validate_constraint_metrics(
        config,
        allowed={ConstraintMetric.SELECTIVITY},
        family="relational",
    )
    table = _require_table(dataset, config)
    _reject_non_retrieval_output(config, family="relational")
    form = config.form
    search_space = _predicate_search_space(form, family="relational")
    _validate_entity_predicate_search_space(
        dataset=dataset,
        fields=table.fields,
        search_space=search_space,
    )
    search_space = _resolve_entity_predicate_search_space(
        dataset=dataset,
        fields=table.fields,
        search_space=search_space,
    )
    shape: SelectShape = SelectShape(
        primary_entity=table.id,
        predicate_search=search_space,
    )
    task: TypedTask = SelectivityTask(
        target=_target_for_metric(config.constraints, ConstraintMetric.SELECTIVITY, default_target=0.5, default_tolerance=0.5),
    )
    return QuerySpec(
        name=f"{table.id.name}_{config.family.value}",
        scope=ScopePayload(primary_entity=table.id),
        shape=shape,
        task=task,
    )


def _compile_filtered_ann_query(config: QuerySpecConfig, dataset: Dataset) -> QuerySpec:
    _validate_filtered_ann_frontdoor(config)
    owner, fields = _require_filtered_ann_source(dataset, config)
    vector_field = _require_single_vector_field(
        fields,
        reason="filtered_ann requires exactly one vector field on the selected source",
    )
    form = config.form
    if form is None:
        raise ContractViolation("filtered_ann requires spec.form")
    search_space = _predicate_search_space(form, family="filtered_ann")
    _validate_entity_predicate_search_space(
        dataset=dataset,
        fields=fields,
        search_space=search_space,
    )
    search_space = _resolve_entity_predicate_search_space(
        dataset=dataset,
        fields=fields,
        search_space=search_space,
    )
    selectivity = _require_target_for_metric(
        config.constraints,
        ConstraintMetric.SELECTIVITY,
        reason="filtered_ann requires constraints[].metric=selectivity",
    )
    gls = _target_for_metric_or_none(config.constraints, ConstraintMetric.GLS)
    hardness = _hardness_from_form(config, QueryFamily.FILTERED_ANN)
    return QuerySpec(
        name=f"{owner.name}_filtered_ann",
        scope=ScopePayload(primary_entity=owner),
        shape=VectorRetrieveShape(
            primary_entity=owner,
            target_field=vector_field.id,
            predicate_search=search_space,
        ),
        task=FilteredAnnTask(
            vector_field=vector_field.id,
            selectivity=selectivity,
            difficulty=hardness,
            gls=gls,
        ),
        output=_resolve_retrieval_output(config, family="filtered_ann"),
    )


def _validate_constraint_metrics(
    config: QuerySpecConfig,
    *,
    allowed: set[ConstraintMetric],
    family: str,
) -> None:
    for constraint in config.constraints:
        if constraint.metric not in allowed:
            allowed_label = ", ".join(sorted(item.value for item in allowed)) or "<none>"
            raise ContractViolation(
                f"{family} query does not support constraints[].metric={constraint.metric.value}; "
                f"allowed: {allowed_label}"
            )


def _resolve_retrieval_output(config: QuerySpecConfig, *, family: str) -> TypedOutput:
    if config.output is None:
        raise ContractViolation(f"{family} query requires spec.output.kind=top_k")
    if config.output.kind is not QueryOutputKind.TOP_K:
        raise ContractViolation(f"{family} query only supports spec.output.kind=top_k")
    return TopKOutput(k=config.output.top_k)


def _reject_non_retrieval_output(config: QuerySpecConfig, *, family: str) -> None:
    if config.output is not None:
        raise ContractViolation(f"{family} query does not support spec.output")


def _resolve_query_output_root(config: QueryJobConfig, dataset: Dataset) -> str:
    if config.output_dir is not None:
        return str(Path(config.output_dir).expanduser().resolve())
    run_id = f"query_{config.spec.family.value}_{config.seed}"
    return str((Path(dataset.root) / "workloads" / run_id).resolve())


def _public_query_dialect(dataset: Dataset) -> DialectTarget:
    if dataset.dialect != "postgresql":
        raise ContractViolation(f"unsupported query dialect from dataset: {dataset.dialect}")
    return DialectTarget(name="postgresql", capabilities=POSTGRESQL_CAPABILITIES)


def _require_table(dataset: Dataset, config: QuerySpecConfig) -> Table:
    table_names = config.target_tables
    if not table_names:
        raise ContractViolation("查询配置必须指定 target_tables，请选择一个目标表")
    if len(table_names) > 1:
        raise ContractViolation(
            f"spec.target_tables only supports a single source, got {len(table_names)}"
        )
    target = table_names[0]
    for table in dataset.tables():
        if table.id.name == target:
            return table
    raise ContractViolation(f"table not found in dataset: {target}")


def _require_filtered_ann_source(
    dataset: Dataset,
    config: QuerySpecConfig,
) -> tuple[TableId, tuple[Field, ...]]:
    table = _require_table(dataset, config)
    return table.id, table.fields


def _require_single_vector_field(fields: Sequence[Field], *, reason: str) -> Field:
    vector_fields = tuple(field for field in fields if field.role is FieldRole.VECTOR)
    if len(vector_fields) != 1:
        raise ContractViolation(reason)
    return vector_fields[0]


def _relational_role_for(
    column: RelationalColumnConfig,
) -> FieldRole:
    if normalize_type_name(column.type_name) == "vector":
        return FieldRole.VECTOR
    if column.primary_key:
        return FieldRole.PRIMARY_KEY
    distribution = column.distribution
    if distribution is not None:
        if distribution.kind.value in {"categorical", "zipf"}:
            return FieldRole.CATEGORICAL
    return FieldRole.PAYLOAD

def _compile_data_type(type_name: str, *, vector_dim: int | None, metric: str | None) -> DataType:
    normalized = type_name.strip().lower()
    if normalized == "int":
        return IntType()
    if normalized == "double":
        return DoubleType()
    if normalized == "text":
        return TextType()
    if normalized == "vector":
        if vector_dim is None or metric is None:
            raise ContractViolation("vector type requires dimension and metric")
        metric_enum = {
            "l2": VectorMetric.L2,
            "cosine": VectorMetric.COSINE,
        }.get(metric.strip().lower())
        if metric_enum is None:
            raise ContractViolation(f"unsupported vector metric: {metric}")
        return VectorType(dimension=vector_dim, metric=metric_enum)
    raise ContractViolation(f"unsupported logical type: {type_name}")


def _predicate_search_space(form: QueryFormConfig | None, *, family: str) -> PredicateSearchSpace:
    if form is None:
        raise ContractViolation(f"{family} query requires spec.form")
    count = _compile_predicate_count(form.predicate_count, family=family)
    operators = _compile_predicate_operators(form.operators, family=family)
    return PredicateSearchSpace(
        predicate_count=count,
        operators=operators,
        predicate_fields=tuple(form.predicate_fields),
        max_in_values=_compile_max_in_values(form.max_in_values, operators=operators),
    )


def _form_has_predicate_search(form: QueryFormConfig | None) -> bool:
    if form is None:
        return False
    return (
        form.predicate_count is not None
        or bool(form.operators)
        or bool(form.predicate_fields)
        or form.max_in_values is not None
    )


def _compile_predicate_count(
    count: PredicateCountRangeConfig | None,
    *,
    family: str,
) -> PredicateCountRange:
    if count is None:
        raise ContractViolation(f"{family} query requires spec.form.predicate_count with min and max")
    if count.min < 1:
        raise ContractViolation("spec.form.predicate_count.min must be >= 1")
    if count.max < count.min:
        raise ContractViolation("spec.form.predicate_count.max must be >= min")
    if count.max > 8:
        raise ContractViolation("spec.form.predicate_count.max must be <= 8 in predicate search contract v1")
    return PredicateCountRange(min=count.min, max=count.max)


def _compile_predicate_operators(
    operators: Sequence[PredicateOperator],
    *,
    family: str,
) -> tuple[PredicateOperator, ...]:
    if not operators:
        raise ContractViolation(f"{family} query requires spec.form.operators")
    result: list[PredicateOperator] = []
    seen: set[PredicateOperator] = set()
    for operator in operators:
        if not isinstance(operator, PredicateOperator):
            raise ContractViolation("spec.form.operators[] must be a PredicateOperator")
        if operator in seen:
            continue
        seen.add(operator)
        result.append(operator)
    return tuple(result)


def _compile_max_in_values(
    value: int | None,
    *,
    operators: tuple[PredicateOperator, ...],
) -> int:
    if value is None:
        return DEFAULT_MAX_IN_VALUES
    if value < 1:
        raise ContractViolation("spec.form.max_in_values must be >= 1")
    if PredicateOperator.IN not in operators:
        raise ContractViolation("spec.form.max_in_values requires the in operator")
    return value


def _validate_entity_predicate_search_space(
    *,
    dataset: Dataset,
    fields: Sequence[Field],
    search_space: PredicateSearchSpace,
) -> None:
    candidates = tuple(field for field in fields if _field_can_enter_predicate_search(field))
    if search_space.predicate_fields:
        by_name = {field.id.name: field for field in candidates}
        selected: list[Field] = []
        for selector in search_space.predicate_fields:
            field = by_name.get(selector)
            if field is None:
                raise ContractViolation(f"predicate filter field {selector!r} is not available on selected source")
            selected.append(field)
        candidates = tuple(dict.fromkeys(selected))
    if not any(_field_matches_operator_pool(dataset, field, search_space.operators) for field in candidates):
        raise ContractViolation("predicate search has no compatible filter field for the selected operator pool")


def _resolve_entity_predicate_search_space(
    *,
    dataset: Dataset,
    fields: Sequence[Field],
    search_space: PredicateSearchSpace,
) -> PredicateSearchSpace:
    candidates = tuple(field for field in fields if _field_can_enter_predicate_search(field))
    if search_space.predicate_fields:
        selected_names = set(search_space.predicate_fields)
        candidates = tuple(field for field in candidates if field.id.name in selected_names)
    atoms = tuple(
        PredicateAtomSpec(field=field.id, operator=operator)
        for field in candidates
        for operator in search_space.operators
        if _field_matches_operator(dataset, field, operator)
    )
    if len(atoms) < search_space.predicate_count.min:
        raise ContractViolation(
            "predicate_count.min cannot be satisfied by compatible fields on the selected source"
        )
    resolved_fields = tuple(dict.fromkeys(field.id.name for field in candidates))
    return PredicateSearchSpace(
        predicate_count=search_space.predicate_count,
        operators=search_space.operators,
        predicate_fields=resolved_fields,
        atoms=atoms,
        max_in_values=search_space.max_in_values,
    )


def _field_can_enter_predicate_search(field: Field) -> bool:
    if field.role is FieldRole.PRIMARY_KEY or isinstance(field.type, VectorType):
        return False
    return isinstance(field.id.owner, TableId)


def _field_matches_operator_pool(
    dataset: Dataset,
    field: Field,
    operators: Sequence[PredicateOperator],
) -> bool:
    return any(_field_matches_operator(dataset, field, operator) for operator in operators)


def _field_matches_operator(
    dataset: Dataset, field: Field, operator: PredicateOperator
) -> bool:
    if operator in {
        PredicateOperator.BETWEEN,
        PredicateOperator.LT,
        PredicateOperator.LE,
        PredicateOperator.GT,
        PredicateOperator.GE,
    }:
        # range 谓词：int/double 数值列一律放行。int64 列（n05~n07/n10/n11，
        # role=CATEGORICAL 但本质是数值）此前被 IntAffine 模型门槛整类排除，
        # 导致模板池里只有 double 字段、int 字段永远无法铺开。
        return isinstance(field.type, (IntType, DoubleType))
    if operator in {PredicateOperator.EQ, PredicateOperator.IN}:
        if field.role is FieldRole.CATEGORICAL:
            return True
        return isinstance(field.type, IntType) and _is_int_affine_field(dataset, field)
    return False


def _is_int_affine_field(dataset: Dataset, field: Field) -> bool:
    metadata = deterministic_owner_metadata(dataset, field.id.owner)
    if metadata is None:
        return False
    try:
        model = field_model_for_name(metadata, field.id.name)
    except ContractViolation:
        return False
    return isinstance(model, IntAffineBlocksModel)


def _validate_filtered_ann_frontdoor(config: QuerySpecConfig) -> None:
    if config.form is None or config.form.kind is not QueryFamily.FILTERED_ANN:
        raise ContractViolation("filtered_ann query requires spec.form.kind=filtered_ann")
    if config.form.difficulty is None:
        raise ContractViolation("filtered_ann requires spec.form.difficulty")
    if not _form_has_predicate_search(config.form):
        raise ContractViolation("filtered_ann requires ordinary predicate search")
    _predicate_search_space(config.form, family="filtered_ann")
    seen_metrics = {constraint.metric for constraint in config.constraints}
    if len(seen_metrics) != len(config.constraints):
        raise ContractViolation("filtered_ann constraints must not repeat a metric")
    if ConstraintMetric.SELECTIVITY not in seen_metrics:
        raise ContractViolation("filtered_ann requires constraints[].metric=selectivity")
    for constraint in config.constraints:
        if constraint.metric not in {ConstraintMetric.SELECTIVITY, ConstraintMetric.GLS}:
            raise ContractViolation("filtered_ann only supports selectivity and gls constraints")
        if constraint.min > constraint.max:
            raise ContractViolation("filtered_ann constraint min must be <= max")
        if constraint.metric is ConstraintMetric.SELECTIVITY and not (0.0 < constraint.min <= constraint.max <= 1.0):
            raise ContractViolation("filtered_ann selectivity constraint must be within (0, 1]")
        if constraint.metric is ConstraintMetric.GLS and not (-1.0 <= constraint.min <= constraint.max < 1.0):
            raise ContractViolation("filtered_ann gls constraint must satisfy -1 <= min <= max < 1")


def _target_for_metric(
    constraints: Sequence[ConstraintConfig],
    metric: ConstraintMetric,
    *,
    default_target: float,
    default_tolerance: float,
) -> ToleranceTarget:
    for item in constraints:
        if item.metric is metric:
            center = (item.min + item.max) / 2.0
            tolerance = max((item.max - item.min) / 2.0, 1e-6)
            return ToleranceTarget(target=center, tolerance=tolerance)
    return ToleranceTarget(target=default_target, tolerance=default_tolerance)


def _require_target_for_metric(
    constraints: Sequence[ConstraintConfig],
    metric: ConstraintMetric,
    *,
    reason: str,
) -> ToleranceTarget:
    for item in constraints:
        if item.metric is metric:
            center = (item.min + item.max) / 2.0
            tolerance = max((item.max - item.min) / 2.0, 1e-6)
            return ToleranceTarget(target=center, tolerance=tolerance)
    raise ContractViolation(reason)


def _target_for_metric_or_none(
    constraints: Sequence[ConstraintConfig],
    metric: ConstraintMetric,
) -> ToleranceTarget | None:
    for item in constraints:
        if item.metric is metric:
            center = (item.min + item.max) / 2.0
            tolerance = max((item.max - item.min) / 2.0, 1e-6)
            return ToleranceTarget(target=center, tolerance=tolerance)
    return None


def _hardness_from_form(config: QuerySpecConfig, family: QueryFamily) -> Hardness:
    if config.form is None or config.form.difficulty is None:
        return Hardness.MEDIUM
    mapping = {
        Difficulty.EASY: Hardness.EASY,
        Difficulty.MEDIUM: Hardness.MEDIUM,
        Difficulty.HARD: Hardness.HARD,
    }
    return mapping[config.form.difficulty]


__all__ = [
    "compile_data_job",
    "compile_query_job",
]
