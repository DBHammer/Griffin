"""Materialize typed datasets from compiled generation requests."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Mapping
from typing import TypeAlias

import msgspec
import numpy as np

from griffin.data.entity_builders import (
    ColumnMapping,
    materialize_table_fields,
    materialize_table_vectors,
)
from griffin.data.generation_request import (
    CategoricalAssociationRequest,
    CorrelationBlockRequest,
    DataGenerationRequest,
    TableGenerationRequest,
    VectorAttributeCouplingRequest,
    VectorGenerationRequest,
)
from griffin.data.persist import persist_table

from griffin.data.vector_regions import (
    KMeansRegionConfig,
    materialize_vector_region_layout,
)
from griffin.data.vector_coupling import (
    AttributeVectorCouplingEngine,
    CouplingStrategyConfig,
    ExplainedVarianceRhoMetric,
    MatrixTransportCouplingPlanner,
    StableSegmentPlacementMaterializer,
    _score_ordered_coupled_endpoint,
    plan_independent_attribute_placement,
    plan_vector_attribute_coupling,
)
from griffin.dataset.deterministic import (
    DeterministicFieldModel,
    FieldPlacementModel,
    RegionLayoutModel,
    VectorAttributeCouplingModel,
    model_field_name,
    identity_field_placements,
)
from griffin.data.relations import materialize_correlation, materialize_coupling
from griffin.data.schema import (
    RelationalSchemaExport,
    render_relational_schema,
)
from griffin.data.statistics_writer import entity_metrics
from griffin.dataset.artifacts.base import Artifact
from griffin.dataset.entities import Table
from griffin.dataset.layout import schema_sql_path
from griffin.dataset.relations.base import Relation, RelationSpec
from griffin.dataset.writer import DatasetWriter
from griffin.infra.errors import ContractViolation
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.ir.data_type import VectorType
from griffin.ir.field import Field, FieldRole
from griffin.model_ids import FieldId

Entity: TypeAlias = Table
ScalarColumnMapping: TypeAlias = Mapping[str, list[object]]


class SizingSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    entity_name: str
    row_count: int


class DataPipelineSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    tables: tuple[Table, ...] = ()
    sizing: tuple[SizingSpec, ...] = ()
    data_model: str = "relational"
    output_format: str = "parquet"
    dialect: str = "postgresql"
    artifacts: tuple[Artifact, ...] = ()
    relations: tuple[Relation, ...] = ()
    relation_specs: tuple[RelationSpec, ...] = ()
    generation: DataGenerationRequest | None = None
    schema_export: RelationalSchemaExport | None = None
    seed: int = 0


def _sizing_for(spec: DataPipelineSpec, name: str) -> int:
    for sizing in spec.sizing:
        if sizing.entity_name == name:
            return sizing.row_count
    raise ContractViolation(f"no sizing specified for entity {name}")


def _entity_fields(entity: Entity) -> tuple[Field, ...]:
    return entity.fields


def _build_coupling_engine_from_env() -> AttributeVectorCouplingEngine | None:
    """根据环境变量 GRIFFIN_COUPLING_MODE 构建 coupling engine。

    GRIFFIN_COUPLING_MODE=griffin          → 默认（矩阵搜索 1% + 局部搜索 0.1%）
    GRIFFIN_COUPLING_MODE=only_local_search → 跳过矩阵搜索，只用局部搜索
    GRIFFIN_COUPLING_MODE=no_coupling      → 无耦合（不设置 vector_attribute_couplings 即可，此处无需处理）

    GRIFFIN_COUPLING_TIME_BUDGET_SEC 控制局部搜索时间预算（秒），默认不限时。
    """
    mode = os.environ.get("GRIFFIN_COUPLING_MODE", "griffin")
    time_budget = float(os.environ.get("GRIFFIN_COUPLING_TIME_BUDGET_SEC", "inf"))

    if mode == "only_local_search":
        metric = ExplainedVarianceRhoMetric()
        config = CouplingStrategyConfig(
            matrix_search_enabled=False,
            local_search_enabled=True,
            local_search_max_steps=1_000_000,
            local_search_initial_delta=32,
            local_search_delta_decay=0.7,
            local_search_time_budget_sec=time_budget,
        )
        planner = MatrixTransportCouplingPlanner(metric)
        return AttributeVectorCouplingEngine(planner=planner, config=config)

    if mode == "griffin" and time_budget != float("inf"):
        metric = ExplainedVarianceRhoMetric()
        config = CouplingStrategyConfig(
            local_search_max_steps=1_000_000,
            local_search_time_budget_sec=time_budget,
        )
        planner = MatrixTransportCouplingPlanner(metric)
        return AttributeVectorCouplingEngine(planner=planner, config=config)

    return None


def run(
    spec: DataPipelineSpec,
    root: pathlib.Path | str,
    *,
    task_logger: RuntimeTaskLogger | None = None,
    vector_coupling_engine: AttributeVectorCouplingEngine | None = None,
) -> pathlib.Path:
    # 若外部未传入 engine，则从环境变量构建
    if vector_coupling_engine is None:
        vector_coupling_engine = _build_coupling_engine_from_env()
    root_path = pathlib.Path(root).resolve()
    writer = DatasetWriter.create(
        root_path,
        name=spec.name,
        data_model=spec.data_model,
        output_format=spec.output_format,
        dialect=spec.dialect,
    )

    materialized: dict[str, ColumnMapping] = {}

    _run_stage(
        task_logger,
        "生成主数据",
        lambda: _generate_primary_data(
            spec,
            root_path=root_path,
            writer=writer,
            materialized=materialized,
            task_logger=task_logger,
            vector_coupling_engine=vector_coupling_engine,
        ),
    )
    if os.environ.get("GRIFFIN_SKIP_PERSIST"):
        return root_path
    _run_stage(
        task_logger,
        "计算统计信息",
        lambda: _write_metrics(
            spec,
            root_path=root_path,
            writer=writer,
            materialized=materialized,
        ),
    )

    _run_stage(
        task_logger,
        "构建检索辅助数据",
        lambda: _write_index_artifacts(
            spec,
            root_path=root_path,
            writer=writer,
            materialized=materialized,
        ),
    )
    _run_stage(
        task_logger,
        "写入关系元数据",
        lambda: _write_relations(
            spec,
            root_path=root_path,
            writer=writer,
            materialized=materialized,
        ),
    )
    _run_stage(task_logger, "生成建表语句", lambda: _write_schema(spec, root_path))

    manifest_path: pathlib.Path | None = None

    def _commit_dataset() -> None:
        nonlocal manifest_path
        manifest_path = writer.commit()

    _run_stage(task_logger, "提交数据集", _commit_dataset)
    if manifest_path is None:
        raise ContractViolation("dataset commit did not produce a manifest path")
    return manifest_path


def _run_stage(
    task_logger: RuntimeTaskLogger | None,
    name: str,
    action: Callable[[], None],
) -> None:
    if task_logger is None:
        action()
        return
    with task_logger.stage(name):
        action()


def _generate_primary_data(
    spec: DataPipelineSpec,
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
    task_logger: RuntimeTaskLogger | None,
    vector_coupling_engine: AttributeVectorCouplingEngine | None,
) -> None:
    for index, table in enumerate(spec.tables, start=1):
        _generate_table_data(
            spec,
            root_path=root_path,
            writer=writer,
            materialized=materialized,
            table=table,
            index=index,
            total=len(spec.tables),
            task_logger=task_logger,
            vector_coupling_engine=vector_coupling_engine,
        )

def _generate_table_data(
    spec: DataPipelineSpec,
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
    table: Table,
    index: int,
    total: int,
    task_logger: RuntimeTaskLogger | None,
    vector_coupling_engine: AttributeVectorCouplingEngine | None,
) -> None:
    detail = f"表={table.id.name} ({index}/{total})"
    row_count = _sizing_for(spec, table.id.name)
    request = spec.generation.table(table.id) if spec.generation is not None else None
    columns: ColumnMapping = {}
    deterministic_models: tuple[DeterministicFieldModel, ...] = ()
    region_layout: RegionLayoutModel | None = None
    placements: tuple[FieldPlacementModel, ...] | None = None
    vector_couplings: tuple[VectorAttributeCouplingModel, ...] = ()

    def _assign_models(models: tuple[DeterministicFieldModel, ...]) -> None:
        nonlocal deterministic_models
        deterministic_models = models

    _run_step(
        task_logger,
        "生成表字段数据",
        detail,
        lambda: _materialize_table_columns(
            columns_ref=columns,
            models_ref=_assign_models,
            table=table,
            row_count=row_count,
            seed=spec.seed,
            request=request,
        ),
    )
    if _has_vector_fields(table.fields):
        def _assign_vector_outputs(
            value: tuple[ColumnMapping, RegionLayoutModel | None],
        ) -> None:
            nonlocal region_layout
            vector_columns, region_layout = value
            columns.update(vector_columns)

        _run_step(
            task_logger,
            "生成表向量数据",
            detail,
            lambda: _assign_vector_outputs(
                _materialize_table_vector_columns(
                    table=table,
                    row_count=row_count,
                    seed=spec.seed,
                    request=request,
                )
            ),
        )
    if region_layout is not None or (request is not None and request.vector_attribute_couplings):
        def _assign_coupling_outputs(
            value: tuple[tuple[FieldPlacementModel, ...], tuple[VectorAttributeCouplingModel, ...]],
        ) -> None:
            nonlocal placements, vector_couplings
            placements, vector_couplings = value

        _run_step(
            task_logger,
            "应用表属性-向量放置",
            detail,
            lambda: _assign_coupling_outputs(
                _apply_vector_region_placements(
                    owner_name=table.id.name,
                    fields=table.fields,
                    columns=columns,
                    deterministic_models=deterministic_models,
                    couplings=() if request is None else request.vector_attribute_couplings,
                    correlation_blocks=() if request is None else request.correlation_blocks,
                    categorical_association_blocks=() if request is None else request.categorical_association_blocks,
                    region_layout=region_layout,
                    vector_coupling_engine=vector_coupling_engine,
                )
            ),
        )
    _run_step(
        task_logger,
        "写出表文件",
        detail,
        lambda: persist_table(
            root_path=root_path,
            writer=writer,
            materialized=materialized,
            table=table,
            columns=columns,
            deterministic_models=deterministic_models,
            row_count=row_count,
            output_format=spec.output_format,
            region_layout=region_layout,
            placements=placements,
            vector_attribute_couplings=vector_couplings,
        ) if not os.environ.get("GRIFFIN_SKIP_PERSIST") else None,
    )


def _run_step(
    task_logger: RuntimeTaskLogger | None,
    name: str,
    detail: str,
    action: Callable[[], None],
) -> None:
    if task_logger is None:
        action()
        return
    with task_logger.step(name, detail=detail):
        action()


def _apply_vector_region_placements(
    *,
    owner_name: str,
    fields: tuple[Field, ...],
    columns: ColumnMapping,
    deterministic_models: tuple[DeterministicFieldModel, ...],
    couplings: tuple[VectorAttributeCouplingRequest, ...],
    correlation_blocks: tuple[CorrelationBlockRequest, ...],
    categorical_association_blocks: tuple[CategoricalAssociationRequest, ...],
    region_layout: RegionLayoutModel | None,
    vector_coupling_engine: AttributeVectorCouplingEngine | None = None,
) -> tuple[tuple[FieldPlacementModel, ...], tuple[VectorAttributeCouplingModel, ...]]:
    if region_layout is None:
        raise ContractViolation(f"{owner_name} vector_attribute_couplings require a vector region_layout")
    fields_by_name = {field.id.name: field for field in fields}
    models_by_name = {
        model_field_name(model): model
        for model in deterministic_models
    }
    placements_by_name = {
        placement.field_name: placement
        for placement in identity_field_placements(deterministic_models)
    }
    correlation_block_by_anchor = _correlation_blocks_by_anchor(correlation_blocks, couplings)
    categorical_block_by_anchor = _categorical_association_blocks_by_anchor(categorical_association_blocks, couplings)
    placement_materializer = StableSegmentPlacementMaterializer()
    metadata: list[VectorAttributeCouplingModel] = []
    placed_fields: set[str] = set()
    vector_anchored_categorical_blocks: set[str] = set()
    # ---- 可插拔预校准：GRIFFIN_COUPLING_CALIBRATE_RHO=1 时启用 ----
    _calibrate = os.environ.get("GRIFFIN_COUPLING_CALIBRATE_RHO", "").strip() == "1"
    if _calibrate and region_layout is not None:
        _calibrated_coupling_requests = _calibrate_coupling_targets(
            requests=couplings,
            fields_by_name=fields_by_name,
            models_by_name=models_by_name,
            columns=columns,
            region_layout=region_layout,
        )
    else:
        _calibrated_coupling_requests = couplings
    # ----------------------------------------------------------------
    for request in _calibrated_coupling_requests:
        field = fields_by_name.get(request.attribute_field)
        if field is None:
            raise ContractViolation(
                f"{owner_name} vector coupling {request.name} references unknown attribute {request.attribute_field}"
            )
        model = models_by_name.get(request.attribute_field)
        if model is None:
            raise ContractViolation(
                f"{owner_name} vector coupling {request.name} missing deterministic model for {request.attribute_field}"
            )
        local_values = columns.get(request.attribute_field)
        if local_values is None:
            raise ContractViolation(
                f"{owner_name} vector coupling {request.name} missing values for {request.attribute_field}"
            )
        local_scalar_values = _require_scalar_values(local_values, request.attribute_field)
        plan = plan_vector_attribute_coupling(
            request=request,
            field=field,
            model=model,
            local_values=local_scalar_values,
            region_layout=region_layout,
            engine=vector_coupling_engine,
        )
        columns[request.attribute_field] = plan.values
        placements_by_name[request.attribute_field] = plan.placement
        placed_fields.add(request.attribute_field)
        correlation_block = correlation_block_by_anchor.get(request.attribute_field)
        if correlation_block is not None:
            _apply_correlation_block_placement(
                columns=columns,
                placements_by_name=placements_by_name,
                block=correlation_block,
                anchor_field=request.attribute_field,
                anchor_placement=plan.placement,
                materializer=placement_materializer,
            )
            placed_fields.update(column for column in correlation_block.columns if column in models_by_name)
        categorical_block = categorical_block_by_anchor.get(request.attribute_field)
        if categorical_block is not None:
            _apply_categorical_block_placement(
                columns=columns,
                placements_by_name=placements_by_name,
                block=categorical_block,
                anchor_field=request.attribute_field,
                anchor_placement=plan.placement,
                materializer=placement_materializer,
            )
            placed_fields.update(field for field in categorical_block.columns if field in models_by_name)
            vector_anchored_categorical_blocks.add(categorical_block.name)
        metadata.append(plan.metadata)

    for correlation_block in correlation_blocks:
        block_fields = tuple(column for column in correlation_block.columns if column in models_by_name)
        if not block_fields or any(column in placed_fields for column in block_fields):
            continue
        anchor_field = block_fields[0]
        anchor_placement = _apply_independent_attribute_placement(
            owner_name=owner_name,
            field_name=anchor_field,
            fields_by_name=fields_by_name,
            models_by_name=models_by_name,
            columns=columns,
            region_layout=region_layout,
        )
        placements_by_name[anchor_field] = anchor_placement
        placed_fields.add(anchor_field)
        _apply_correlation_block_placement(
            columns=columns,
            placements_by_name=placements_by_name,
            block=correlation_block,
            anchor_field=anchor_field,
            anchor_placement=anchor_placement,
            materializer=placement_materializer,
        )
        placed_fields.update(block_fields)

    for association_block in categorical_association_blocks:
        if association_block.name in vector_anchored_categorical_blocks:
            continue
        block_fields = association_block.columns
        overlapping = sorted(field for field in block_fields if field in placed_fields)
        if overlapping:
            raise ContractViolation(
                f"{owner_name} categorical association {association_block.name} fields already have vector placement: {overlapping}"
            )
        anchor_placement = _apply_independent_attribute_placement(
            owner_name=owner_name,
            field_name=block_fields[0],
            fields_by_name=fields_by_name,
            models_by_name=models_by_name,
            columns=columns,
            region_layout=region_layout,
        )
        placements_by_name[block_fields[0]] = anchor_placement
        _apply_categorical_block_placement(
            columns=columns,
            placements_by_name=placements_by_name,
            block=association_block,
            anchor_field=block_fields[0],
            anchor_placement=anchor_placement,
            materializer=placement_materializer,
        )
        placed_fields.update(block_fields)

    for model in deterministic_models:
        field_name = model_field_name(model)
        if field_name in placed_fields:
            continue
        placements_by_name[field_name] = _apply_independent_attribute_placement(
            owner_name=owner_name,
            field_name=field_name,
            fields_by_name=fields_by_name,
            models_by_name=models_by_name,
            columns=columns,
            region_layout=region_layout,
        )
        placed_fields.add(field_name)

    placements = tuple(
        placements_by_name[model_field_name(model)]
        for model in deterministic_models
    )
    return placements, tuple(metadata)


def _apply_independent_attribute_placement(
    *,
    owner_name: str,
    field_name: str,
    fields_by_name: dict[str, Field],
    models_by_name: dict[str, DeterministicFieldModel],
    columns: ColumnMapping,
    region_layout: RegionLayoutModel,
) -> FieldPlacementModel:
    field = fields_by_name.get(field_name)
    if field is None:
        raise ContractViolation(f"{owner_name} independent vector placement references unknown field {field_name}")
    model = models_by_name.get(field_name)
    if model is None:
        raise ContractViolation(f"{owner_name} independent vector placement missing deterministic model for {field_name}")
    local_values = columns.get(field_name)
    if local_values is None:
        raise ContractViolation(f"{owner_name} independent vector placement missing values for {field_name}")
    local_scalar_values = _require_scalar_values(local_values, field_name)
    public_values, placement = plan_independent_attribute_placement(
        field=field,
        model=model,
        local_values=local_scalar_values,
        region_layout=region_layout,
    )
    columns[field_name] = public_values
    return placement


def _correlation_blocks_by_anchor(
    blocks: tuple[CorrelationBlockRequest, ...],
    couplings: tuple[VectorAttributeCouplingRequest, ...],
) -> dict[str, CorrelationBlockRequest]:
    coupling_attributes = {
        request.attribute_field
        for request in couplings
    }
    by_anchor: dict[str, CorrelationBlockRequest] = {}
    for block in blocks:
        anchors = tuple(column for column in block.columns if column in coupling_attributes)
        if not anchors:
            continue
        if len(anchors) > 1:
            raise ContractViolation(
                f"correlation block {block.name} has more than one vector-coupled attribute: {anchors}"
            )
        by_anchor[anchors[0]] = block
    return by_anchor


def _categorical_association_blocks_by_anchor(
    blocks: tuple[CategoricalAssociationRequest, ...],
    couplings: tuple[VectorAttributeCouplingRequest, ...],
) -> dict[str, CategoricalAssociationRequest]:
    coupling_attributes = {
        request.attribute_field
        for request in couplings
    }
    by_anchor: dict[str, CategoricalAssociationRequest] = {}
    for block in blocks:
        anchors = tuple(column for column in block.columns if column in coupling_attributes)
        if not anchors:
            continue
        if len(anchors) > 1:
            raise ContractViolation(
                f"categorical association block {block.name} has more than one vector-coupled attribute: {anchors}"
            )
        by_anchor[anchors[0]] = block
    return by_anchor


def _apply_correlation_block_placement(
    *,
    columns: ColumnMapping,
    placements_by_name: dict[str, FieldPlacementModel],
    block: CorrelationBlockRequest,
    anchor_field: str,
    anchor_placement: FieldPlacementModel,
    materializer: StableSegmentPlacementMaterializer,
) -> None:
    for field_name in block.columns:
        if field_name == anchor_field:
            continue
        local_values = columns.get(field_name)
        if local_values is None:
            raise ContractViolation(
                f"correlation block {block.name} missing values for field {field_name}"
            )
        scalar_values = _require_scalar_values(local_values, field_name)
        placement = FieldPlacementModel(
            field_name=field_name,
            row_count=anchor_placement.row_count,
            segments=anchor_placement.segments,
        )
        columns[field_name] = materializer.apply(scalar_values, placement)
        placements_by_name[field_name] = placement


def _apply_categorical_block_placement(
    *,
    columns: ColumnMapping,
    placements_by_name: dict[str, FieldPlacementModel],
    block: CategoricalAssociationRequest,
    anchor_field: str,
    anchor_placement: FieldPlacementModel,
    materializer: StableSegmentPlacementMaterializer,
) -> None:
    for field_name in block.columns:
        if field_name == anchor_field:
            continue
        local_values = columns.get(field_name)
        if local_values is None:
            raise ContractViolation(
                f"categorical association {block.name} missing values for field {field_name}"
            )
        placement = FieldPlacementModel(
            field_name=field_name,
            row_count=anchor_placement.row_count,
            segments=anchor_placement.segments,
        )
        columns[field_name] = materializer.apply(
            _require_scalar_values(local_values, field_name),
            placement,
        )
        placements_by_name[field_name] = placement


def _materialize_table_columns(
    *,
    columns_ref: ColumnMapping,
    models_ref: Callable[[tuple[DeterministicFieldModel, ...]], None],
    table: Table,
    row_count: int,
    seed: int,
    request: object,
) -> None:
    from griffin.data.generation_request import TableGenerationRequest

    table_request = request if isinstance(request, TableGenerationRequest) else None
    columns, models = materialize_table_fields(
        table,
        row_count,
        seed=seed,
        request=table_request,
    )
    columns_ref.update(columns)
    models_ref(models)


def _materialize_table_vector_columns(
    *,
    table: Table,
    row_count: int,
    seed: int,
    request: object,
) -> tuple[ColumnMapping, RegionLayoutModel | None]:
    table_request = request if isinstance(request, TableGenerationRequest) else None
    columns = materialize_table_vectors(table, row_count, seed=seed, request=table_request)
    return _materialize_owner_vector_region_layout(
        fields=table.fields,
        columns=columns,
        row_count=row_count,
        seed=seed + 14_001,
        vector_request=_table_vector_request(table_request),
    )


def _materialize_owner_vector_region_layout(
    *,
    fields: tuple[Field, ...],
    columns: ColumnMapping,
    row_count: int,
    seed: int,
    vector_request: VectorGenerationRequest | None,
) -> tuple[ColumnMapping, RegionLayoutModel | None]:
    vector_fields = tuple(field for field in fields if field.role is FieldRole.VECTOR)
    if not vector_fields or row_count <= 0:
        return columns, None
    if len(vector_fields) > 1:
        raise ContractViolation("vector region layout requires a single vector field")
    field = vector_fields[0]
    if not isinstance(field.type, VectorType):
        raise ContractViolation(f"vector field {field.id.name} must have VectorType")
    raw_vectors = columns.get(field.id.name)
    if raw_vectors is None:
        raise ContractViolation(f"missing vector column for region layout: {field.id.name}")
    if not isinstance(raw_vectors, np.ndarray):
        raise ContractViolation(f"vector column {field.id.name} must be a numpy matrix")
    requested_region_count = (
        vector_request.region_count
        if vector_request is not None and vector_request.field_name == field.id.name
        else None
    )
    region_config = (
        KMeansRegionConfig(
            assignment_backend=vector_request.region_assignment_backend,
            clustering_backend=vector_request.region_clustering_backend,
        )
        if vector_request is not None and vector_request.field_name == field.id.name
        else None
    )
    reordered, layout = materialize_vector_region_layout(
        raw_vectors,
        vector_field_name=field.id.name,
        metric=field.type.metric,
        requested_region_count=requested_region_count,
        seed=seed,
        config=region_config,
    )
    columns[field.id.name] = reordered
    return columns, layout


def _table_vector_request(request: TableGenerationRequest | None) -> VectorGenerationRequest | None:
    return request.vector if request is not None else None


def _write_metrics(
    spec: DataPipelineSpec,
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
) -> None:
    for entity in _all_entities(spec):
        columns = materialized.get(entity.id.name)
        if columns is None:
            continue
        metrics = entity_metrics(entity.id, _entity_fields(entity), columns)
        writer.add_metrics(metrics)


def _write_index_artifacts(
    spec: DataPipelineSpec,
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
) -> None:
    for entity in _all_entities(spec):
        columns = materialized.get(entity.id.name)
        if columns is None:
            continue
        for field in _entity_fields(entity):
            if field.role is FieldRole.VECTOR:
                from .artifacts_writer import write_vector_index

                writer.add_artifact(write_vector_index(entity.id, field, columns, root_path))
    for artifact in spec.artifacts:
        writer.add_artifact(artifact)


def _all_entities(spec: DataPipelineSpec) -> tuple[Entity, ...]:
    return spec.tables


def _write_relations(
    spec: DataPipelineSpec,
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
) -> None:
    for relation in spec.relations:
        writer.add_relation(relation)
    for relation_spec in spec.relation_specs:
        writer.add_relation(
            _materialize_relation_spec(
                relation_spec,
                materialized=materialized,
            )
        )


def _write_schema(spec: DataPipelineSpec, root_path: pathlib.Path) -> None:
    export_request = spec.schema_export or RelationalSchemaExport(dialect=spec.dialect)
    schema_sql_path(root_path).write_text(
        render_relational_schema(export_request, spec.tables),
        encoding="utf-8",
    )


def _has_vector_fields(fields: tuple[Field, ...]) -> bool:
    return any(field.role is FieldRole.VECTOR for field in fields)


def _materialize_relation_spec(
    relation_spec: RelationSpec,
    *,
    materialized: dict[str, ColumnMapping],
) -> Relation:
    if relation_spec.kind == "correlation":
        fields = _require_field_participants(relation_spec, exact=2)
        columns = _require_same_entity_columns(fields, materialized)
        return materialize_correlation(relation_spec.name, fields[0], fields[1], columns)
    if relation_spec.kind == "coupling":
        fields = _require_field_participants(relation_spec, minimum=2)
        columns = _require_same_entity_columns(fields, materialized)
        return materialize_coupling(relation_spec.name, fields, columns)
    raise ContractViolation(f"unsupported relation spec kind: {relation_spec.kind}")


def _require_field_participants(
    relation_spec: RelationSpec,
    *,
    exact: int | None = None,
    minimum: int | None = None,
) -> tuple[FieldId, ...]:
    fields = tuple(
        participant for participant in relation_spec.participants if isinstance(participant, FieldId)
    )
    if exact is not None and len(fields) != exact:
        raise ContractViolation(f"relation {relation_spec.name} requires exactly {exact} field participants")
    if minimum is not None and len(fields) < minimum:
        raise ContractViolation(f"relation {relation_spec.name} requires at least {minimum} field participants")
    if not fields:
        raise ContractViolation(f"relation {relation_spec.name} requires field participants")
    return fields


def _require_same_entity_columns(
    fields: tuple[FieldId, ...],
    materialized: dict[str, ColumnMapping],
) -> ScalarColumnMapping:
    owner = fields[0].owner
    if any(field.owner != owner for field in fields[1:]):
        raise ContractViolation("cross-entity relation materialization is not supported")
    columns = materialized.get(owner.name)
    if columns is None:
        raise ContractViolation(f"entity data missing for relation participants: {owner.name}")
    return _require_scalar_columns(columns, tuple(field.name for field in fields))


def _require_scalar_columns(columns: ColumnMapping, names: tuple[str, ...]) -> ScalarColumnMapping:
    return {name: _require_scalar_column(columns, name) for name in names}


def _require_scalar_column(columns: ColumnMapping, name: str) -> list[object]:
    values = columns.get(name)
    if values is None:
        raise ContractViolation(f"missing scalar column for relation materialization: {name}")
    return _require_scalar_values(values, name)


def _require_scalar_values(values: object, name: str) -> list[object]:
    if not isinstance(values, list):
        raise ContractViolation(f"column {name} must be materialized as scalar values")
    return values


def _calibrate_coupling_targets(
    *,
    requests: tuple[VectorAttributeCouplingRequest, ...],
    fields_by_name: dict[str, Field],
    models_by_name: dict[str, DeterministicFieldModel],
    columns: ColumnMapping,
    region_layout: RegionLayoutModel,
) -> tuple[VectorAttributeCouplingRequest, ...]:
    """快速预校准：按 score-ordered endpoint 收紧当前搜索支持的 target rho。

    通过环境变量 GRIFFIN_COUPLING_CALIBRATE_RHO=1 启用。
    预校准只跑快速矩阵搜索（跳过 local_search），O(region_count × group_count × iterations)，
    然后对 target_rho 做 clamped 降级。
    """
    from griffin.data.vector_coupling import (
        CouplingTask,
        DeterministicAttributeGroupingStrategy,
        ExplainedVarianceRhoMetric,
    )
    import logging as _logging
    _log = _logging.getLogger("griffin.task")
    metric = ExplainedVarianceRhoMetric()
    grouping_strategy = DeterministicAttributeGroupingStrategy()
    calibrated: list[VectorAttributeCouplingRequest] = []
    for request in requests:
        field = fields_by_name.get(request.attribute_field)
        model = models_by_name.get(request.attribute_field)
        local_values = columns.get(request.attribute_field)
        if field is None or model is None or local_values is None:
            calibrated.append(request)
            continue
        local_scalar = _require_scalar_values(local_values, request.attribute_field)
        task = CouplingTask(
            name=request.name,
            vector_field_name=request.vector_field,
            attribute_field_name=request.attribute_field,
            target_rho=request.target_rho,
            max_groups=request.max_groups,
            field=field,
            model=model,
            local_values=tuple(local_scalar),
            region_layout=region_layout,
        )
        grouping = grouping_strategy.group(task)
        if not grouping.groups:
            calibrated.append(request)
            continue
        region_counts = tuple(region.public_end - region.public_start + 1 for region in region_layout.regions)
        coupled = _score_ordered_coupled_endpoint(region_counts, grouping)
        coupled_rho = metric.from_matrix(coupled, grouping)
        if request.target_rho <= coupled_rho:
            calibrated.append(request)
        else:
            safe_rho = max(0.0, coupled_rho * 0.95)
            _log.info(
                "[calibrate] %s: target_rho=%.4f > coupled_endpoint=%.4f → clamped to %.4f",
                request.name,
                request.target_rho,
                coupled_rho,
                safe_rho,
            )
            calibrated.append(
                VectorAttributeCouplingRequest(
                    name=request.name,
                    vector_field=request.vector_field,
                    attribute_field=request.attribute_field,
                    target_rho=safe_rho,
                    max_groups=request.max_groups,
                )
            )
    return tuple(calibrated)


__all__ = ["DataPipelineSpec", "SizingSpec", "run"]
