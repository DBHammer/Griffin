"""Typed deterministic metadata for relational tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union

import msgspec


class NullBlock(msgspec.Struct, frozen=True, omit_defaults=True):
    start: int
    end: int


class CategoryBlock(msgspec.Struct, frozen=True, omit_defaults=True):
    label: str
    start: int
    end: int


class UniformNumericModel(msgspec.Struct, frozen=True, omit_defaults=True, tag="uniform_numeric", tag_field="model_kind"):
    field_name: str
    value_type: Literal["double"]
    row_count: int
    non_null_row_count: int
    nulls_first: int = 0
    minimum: float = 0.0
    maximum: float = 1.0
    linear_k: float = 1.0
    linear_b: float = 0.0


class LognormalNumericModel(msgspec.Struct, frozen=True, omit_defaults=True, tag="lognormal_numeric", tag_field="model_kind"):
    field_name: str
    value_type: Literal["int", "double"]
    row_count: int
    non_null_row_count: int
    nulls_first: int = 0
    mean: float = 0.0
    sigma: float = 1.0
    linear_k: float = 1.0
    linear_b: float = 0.0


class CategoricalBlocksModel(msgspec.Struct, frozen=True, omit_defaults=True, tag="categorical_blocks", tag_field="model_kind"):
    field_name: str
    value_type: Literal["int", "text"]
    row_count: int
    non_null_row_count: int
    nulls_first: int = 0
    blocks: tuple[CategoryBlock, ...] = ()


class IntAffineBlocksModel(msgspec.Struct, frozen=True, omit_defaults=True, tag="int_affine_blocks", tag_field="model_kind"):
    field_name: str
    distribution_kind: Literal["uniform", "zipf"]
    row_count: int
    non_null_row_count: int
    domain_minimum: int
    domain_maximum: int
    step: int
    bias: int
    blocks: tuple[CategoryBlock, ...]
    nulls_first: int = 0


DeterministicFieldModel = Union[
    UniformNumericModel,
    LognormalNumericModel,
    CategoricalBlocksModel,
    IntAffineBlocksModel,
]


def model_field_name(model: DeterministicFieldModel) -> str:
    if isinstance(model, (UniformNumericModel, LognormalNumericModel, CategoricalBlocksModel, IntAffineBlocksModel)):
        return model.field_name
    raise TypeError(f"unsupported deterministic model type: {type(model).__name__}")


def model_value_type(model: DeterministicFieldModel) -> str:
    if isinstance(model, (UniformNumericModel, LognormalNumericModel, CategoricalBlocksModel)):
        return model.value_type
    if isinstance(model, IntAffineBlocksModel):
        return "int"
    raise TypeError(f"unsupported deterministic model type: {type(model).__name__}")


class PlacementSegment(msgspec.Struct, frozen=True, omit_defaults=True):
    local_start: int
    local_end: int
    public_start: int
    public_end: int
    region_id: int | None = None


class FieldPlacementModel(msgspec.Struct, frozen=True, omit_defaults=True):
    field_name: str
    row_count: int
    segments: tuple[PlacementSegment, ...] = ()


def identity_field_placement(field_name: str, row_count: int) -> FieldPlacementModel:
    if row_count <= 0:
        return FieldPlacementModel(field_name=field_name, row_count=row_count)
    return FieldPlacementModel(
        field_name=field_name,
        row_count=row_count,
        segments=(
            PlacementSegment(
                local_start=0,
                local_end=row_count - 1,
                public_start=0,
                public_end=row_count - 1,
            ),
        ),
    )


def identity_field_placements(
    models: Iterable[DeterministicFieldModel],
) -> tuple[FieldPlacementModel, ...]:
    return tuple(
        identity_field_placement(
            field_name=model_field_name(model),
            row_count=model.row_count,
        )
        for model in models
    )


class RegionRange(msgspec.Struct, frozen=True, omit_defaults=True):
    region_id: int
    public_start: int
    public_end: int
    representative_public_ordinal: int


class RegionLayoutModel(msgspec.Struct, frozen=True, omit_defaults=True):
    vector_field_name: str
    row_count: int
    region_method: str = "kmeans"
    region_clustering_backend: str = "faiss"
    region_assignment_backend: str = "faiss"
    requested_region_count: int | None = None
    resolved_region_count: int | None = None
    region_centers: tuple[tuple[float, ...], ...] = ()
    region_neighbor_limit: int | None = None
    region_neighbors: tuple[tuple[int, ...], ...] = ()
    region_inertia: float | None = None
    region_iterations: int | None = None
    regions: tuple[RegionRange, ...] = ()


class CouplingGroup(msgspec.Struct, frozen=True, omit_defaults=True):
    group_id: int
    label: str
    local_start: int
    local_end: int
    representative_value: float | None = None


class VectorAttributeCouplingModel(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    vector_field_name: str
    attribute_field_name: str
    target_rho: float
    observed_rho: float
    max_groups: int | None = None
    groups: tuple[CouplingGroup, ...] = ()
    region_group_counts: tuple[tuple[int, ...], ...] = ()


class TableDeterministicMetadata(msgspec.Struct, frozen=True, omit_defaults=True):
    table_name: str
    row_ordinal_start: int = 0
    row_count: int = 0
    supported_fields: tuple[DeterministicFieldModel, ...] = ()
    field_placements: tuple[FieldPlacementModel, ...] = ()
    region_layout: RegionLayoutModel | None = None
    vector_attribute_couplings: tuple[VectorAttributeCouplingModel, ...] = ()


__all__ = [
    "CategoryBlock",
    "CategoricalBlocksModel",
    "CouplingGroup",
    "DeterministicFieldModel",
    "FieldPlacementModel",
    "IntAffineBlocksModel",
    "LognormalNumericModel",
    "NullBlock",
    "PlacementSegment",
    "RegionLayoutModel",
    "RegionRange",
    "TableDeterministicMetadata",
    "UniformNumericModel",
    "VectorAttributeCouplingModel",
    "model_field_name",
    "model_value_type",
    "identity_field_placement",
    "identity_field_placements",
]
