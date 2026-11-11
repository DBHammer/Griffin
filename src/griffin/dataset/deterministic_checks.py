"""Shared invariant checks for deterministic dataset metadata."""

from __future__ import annotations

from griffin.dataset.deterministic import (
    CategoricalBlocksModel,
    DeterministicFieldModel,
    model_field_name,
    model_value_type,
    FieldPlacementModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    PlacementSegment,
    RegionLayoutModel,
    TableDeterministicMetadata,
    UniformNumericModel,
    VectorAttributeCouplingModel,
    identity_field_placements,
)
from griffin.dataset.exact_cover import coupling_replay_placement
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import (
    DoubleType,
    IntType,
    TextType,
    VectorMetric,
    VectorType,
)
from griffin.ir.field import Field, FieldRole
from griffin.vector_hardness_core import exact_self_first_neighbor_labels


def normalize_table_metadata(
    metadata: TableDeterministicMetadata,
) -> TableDeterministicMetadata:
    placements = metadata.field_placements or identity_field_placements(
        metadata.supported_fields
    )
    return TableDeterministicMetadata(
        table_name=metadata.table_name,
        row_ordinal_start=metadata.row_ordinal_start,
        row_count=metadata.row_count,
        supported_fields=metadata.supported_fields,
        field_placements=placements,
        region_layout=metadata.region_layout,
        vector_attribute_couplings=metadata.vector_attribute_couplings,
    )


def validate_deterministic_fields(
    models: tuple[object, ...],
    *,
    row_count: int,
    fields: tuple[Field, ...],
    owner: str,
) -> None:
    if row_count < 0:
        raise ContractViolation(f"{owner} metadata row_count must be >= 0")
    allowed_fields = {
        field.id.name: field
        for field in fields
        if field.role not in {FieldRole.PRIMARY_KEY, FieldRole.VECTOR}
    }
    for model in models:
        field_name = getattr(model, "field_name", None)
        if field_name not in allowed_fields:
            raise ContractViolation(
                f"{owner} model references unsupported deterministic field: {field_name}"
            )
        if getattr(model, "row_count", None) != row_count:
            raise ContractViolation(
                f"{owner} model row_count must match metadata row_count"
            )
        validate_deterministic_model_type(
            model, field=allowed_fields[field_name], owner=owner
        )


def validate_deterministic_model_type(
    model: object, *, field: Field, owner: str
) -> None:
    if not isinstance(
        model,
        (
            UniformNumericModel,
            LognormalNumericModel,
            CategoricalBlocksModel,
            IntAffineBlocksModel,
        ),
    ):
        raise ContractViolation(
            f"{owner} uses unsupported deterministic model type: {type(model).__name__}"
        )
    value_type = model_value_type(model)
    if value_type == "int" and not isinstance(field.type, IntType):
        raise ContractViolation(
            f"{owner} model type does not match field {field.id.name}"
        )
    if value_type == "double" and not isinstance(field.type, DoubleType):
        raise ContractViolation(
            f"{owner} model type does not match field {field.id.name}"
        )
    if value_type == "text" and not isinstance(field.type, TextType):
        raise ContractViolation(
            f"{owner} model type does not match field {field.id.name}"
        )
    if isinstance(model, IntAffineBlocksModel):
        _validate_int_affine_blocks(model, owner=owner)


def _validate_int_affine_blocks(model: IntAffineBlocksModel, *, owner: str) -> None:
    if model.distribution_kind not in {"uniform", "zipf"}:
        raise ContractViolation(
            f"{owner} int affine distribution_kind must be uniform or zipf"
        )
    if model.row_count != model.nulls_first + model.non_null_row_count:
        raise ContractViolation(
            f"{owner} int affine row counts must cover the full ordinal domain"
        )
    if model.non_null_row_count < 1:
        raise ContractViolation(f"{owner} int affine non_null_row_count must be >= 1")
    if model.domain_minimum > model.domain_maximum:
        raise ContractViolation(f"{owner} int affine domain minimum must be <= maximum")
    if not model.blocks:
        raise ContractViolation(f"{owner} int affine blocks must be non-empty")
    if len(model.blocks) == 1:
        if model.step != 0:
            raise ContractViolation(f"{owner} single-value int affine step must be 0")
    elif model.step < 1:
        raise ContractViolation(f"{owner} multi-value int affine step must be >= 1")
    if model.domain_maximum - model.domain_minimum < len(model.blocks) - 1:
        raise ContractViolation(
            f"{owner} int affine value_domain cannot hold all support values"
        )

    expected_start = 0
    for index, block in enumerate(model.blocks):
        if block.start != expected_start or block.end < block.start:
            raise ContractViolation(
                f"{owner} int affine blocks must be contiguous and non-empty"
            )
        expected_value = model.bias + model.step * index
        if block.label != str(expected_value):
            raise ContractViolation(
                f"{owner} int affine block labels must follow bias + step * index"
            )
        if (
            expected_value < model.domain_minimum
            or expected_value > model.domain_maximum
        ):
            raise ContractViolation(
                f"{owner} int affine block labels must stay inside value_domain"
            )
        expected_start = block.end + 1
    if expected_start != model.non_null_row_count:
        raise ContractViolation(
            f"{owner} int affine blocks must cover all non-null rows"
        )
    if model.distribution_kind == "uniform":
        counts = tuple(block.end - block.start + 1 for block in model.blocks)
        if max(counts) - min(counts) > 1:
            raise ContractViolation(
                f"{owner} int uniform block counts must differ by at most 1"
            )


def validate_field_placements(
    placements: tuple[FieldPlacementModel, ...],
    *,
    row_count: int,
    models: tuple[object, ...],
    owner: str,
) -> None:
    supported_models = _deterministic_models(models, owner=owner)
    expected_fields = {model_field_name(model) for model in supported_models}
    placement_fields = {placement.field_name for placement in placements}
    if placement_fields != expected_fields:
        raise ContractViolation(
            f"{owner} must cover exactly the deterministic fields/properties"
        )
    for placement in placements:
        if placement.row_count != row_count:
            raise ContractViolation(f"{owner} row_count must match metadata row_count")
        validate_placement_segments(
            placement.segments,
            row_count=row_count,
            owner=f"{owner} {placement.field_name}",
        )


def validate_placement_segments(
    segments: tuple[PlacementSegment, ...],
    *,
    row_count: int,
    owner: str,
) -> None:
    if row_count == 0:
        if segments:
            raise ContractViolation(
                f"{owner} must not contain placement segments when row_count=0"
            )
        return
    if not segments:
        raise ContractViolation(
            f"{owner} must contain placement segments when row_count>0"
        )
    expected_local = 0
    for segment in segments:
        if segment.local_start < 0 or segment.public_start < 0:
            raise ContractViolation(f"{owner} placement segments must be zero-based")
        if segment.local_end < segment.local_start:
            raise ContractViolation(f"{owner} local segment must be non-decreasing")
        if segment.public_end < segment.public_start:
            raise ContractViolation(f"{owner} public segment must be non-decreasing")
        if segment.local_start != expected_local:
            raise ContractViolation(f"{owner} local segments must be contiguous")
        local_length = segment.local_end - segment.local_start
        public_length = segment.public_end - segment.public_start
        if local_length != public_length:
            raise ContractViolation(f"{owner} local/public segment lengths must match")
        expected_local = segment.local_end + 1
    if expected_local != row_count:
        raise ContractViolation(
            f"{owner} local segments must cover full ordinal domain"
        )

    expected_public = 0
    for segment in sorted(segments, key=lambda item: item.public_start):
        if segment.public_start != expected_public:
            raise ContractViolation(
                f"{owner} public segments must cover the full ordinal domain"
            )
        expected_public = segment.public_end + 1
    if expected_public != row_count:
        raise ContractViolation(
            f"{owner} public segments must cover the full ordinal domain"
        )


def validate_region_layout(
    region_layout: RegionLayoutModel | None,
    *,
    row_count: int,
    fields: tuple[Field, ...],
    owner: str,
) -> None:
    vector_fields = {
        field.id.name for field in fields if field.role is FieldRole.VECTOR
    }
    if vector_fields and row_count > 0 and region_layout is None:
        raise ContractViolation(
            f"{owner} is required for vector-bearing owners with row_count>0"
        )
    if region_layout is None:
        return
    if region_layout.row_count != row_count:
        raise ContractViolation(f"{owner} row_count must match metadata row_count")
    if region_layout.vector_field_name not in vector_fields:
        raise ContractViolation(
            f"{owner} references unknown vector field: {region_layout.vector_field_name}"
        )
    if row_count == 0:
        if region_layout.regions:
            raise ContractViolation(
                f"{owner} must not contain regions when row_count=0"
            )
        if region_layout.region_centers:
            raise ContractViolation(
                f"{owner} must not contain region_centers when row_count=0"
            )
        if region_layout.region_neighbor_limit is not None:
            raise ContractViolation(
                f"{owner} must not contain region_neighbor_limit when row_count=0"
            )
        if region_layout.region_neighbors:
            raise ContractViolation(
                f"{owner} must not contain region_neighbors when row_count=0"
            )
        return
    if not region_layout.regions:
        raise ContractViolation(f"{owner} must contain regions when row_count>0")
    if region_layout.region_method != "kmeans":
        raise ContractViolation(f"{owner} region_method must be kmeans")
    if region_layout.region_clustering_backend not in {"faiss", "numpy", "sklearn"}:
        raise ContractViolation(
            f"{owner} region_clustering_backend must be one of: faiss, numpy, sklearn"
        )
    if region_layout.region_assignment_backend not in {"faiss", "numpy", "sklearn"}:
        raise ContractViolation(
            f"{owner} region_assignment_backend must be one of: "
            "faiss, numpy, sklearn"
        )
    if (
        region_layout.requested_region_count is not None
        and region_layout.requested_region_count < 1
    ):
        raise ContractViolation(f"{owner} requested_region_count must be >= 1")
    if (
        region_layout.resolved_region_count is not None
        and region_layout.resolved_region_count != len(region_layout.regions)
    ):
        raise ContractViolation(f"{owner} resolved_region_count must match regions")
    if not region_layout.region_centers:
        raise ContractViolation(f"{owner} region_centers must be present")
    if len(region_layout.region_centers) != len(region_layout.regions):
        raise ContractViolation(f"{owner} region_centers row count must match regions")
    vector_dimension: int | None = None
    for field in fields:
        if field.id.name == region_layout.vector_field_name and isinstance(
            field.type, VectorType
        ):
            vector_dimension = field.type.dimension
            break
    if vector_dimension is not None:
        for center in region_layout.region_centers:
            if len(center) != vector_dimension:
                raise ContractViolation(
                    f"{owner} region_centers dimension must match vector field"
                )
    _validate_region_neighbors(region_layout, owner=owner)
    expected_public = 0
    for expected_region_id, region in enumerate(region_layout.regions):
        if region.region_id != expected_region_id:
            raise ContractViolation(f"{owner} region_id must be contiguous from 0")
        if region.public_end < region.public_start:
            raise ContractViolation(f"{owner} region range must be non-decreasing")
        if region.public_start != expected_public:
            raise ContractViolation(
                f"{owner} regions must be contiguous on public ordinal"
            )
        if not (
            region.public_start
            <= region.representative_public_ordinal
            <= region.public_end
        ):
            raise ContractViolation(
                f"{owner} representative_public_ordinal must fall inside its region"
            )
        expected_public = region.public_end + 1
    if expected_public != row_count:
        raise ContractViolation(f"{owner} regions must cover full ordinal domain")


def _validate_region_neighbors(region_layout: RegionLayoutModel, *, owner: str) -> None:
    region_count = len(region_layout.regions)
    if region_layout.region_neighbor_limit is None:
        raise ContractViolation(f"{owner} region_neighbor_limit must be present")
    neighbor_limit = int(region_layout.region_neighbor_limit)
    if neighbor_limit < 1 or neighbor_limit > region_count:
        raise ContractViolation(
            f"{owner} region_neighbor_limit must be between 1 and region count"
        )
    if not region_layout.region_neighbors:
        raise ContractViolation(f"{owner} region_neighbors must be present")
    if len(region_layout.region_neighbors) != region_count:
        raise ContractViolation(
            f"{owner} region_neighbors row count must match regions"
        )
    for region_id, neighbors in enumerate(region_layout.region_neighbors):
        if len(neighbors) != neighbor_limit:
            raise ContractViolation(
                f"{owner} region_neighbors row length must match region_neighbor_limit"
            )
        if not neighbors:
            raise ContractViolation(f"{owner} region_neighbors rows must not be empty")
        if int(neighbors[0]) != region_id:
            raise ContractViolation(
                f"{owner} region_neighbors first entry must be the region itself"
            )
        seen: set[int] = set()
        for neighbor in neighbors:
            neighbor_id = int(neighbor)
            if neighbor_id < 0 or neighbor_id >= region_count:
                raise ContractViolation(
                    f"{owner} region_neighbors contains out-of-range region id"
                )
            if neighbor_id in seen:
                raise ContractViolation(
                    f"{owner} region_neighbors rows must not contain duplicates"
                )
            seen.add(neighbor_id)
    expected = exact_self_first_neighbor_labels(
        region_layout.region_centers,
        metric=VectorMetric.L2,
        limit=neighbor_limit,
    )
    if region_layout.region_neighbors != expected:
        raise ContractViolation(
            f"{owner} region_neighbors must match center exact KNN order"
        )


def validate_vector_attribute_couplings(
    couplings: tuple[VectorAttributeCouplingModel, ...],
    *,
    row_count: int,
    models: tuple[object, ...],
    placements: tuple[FieldPlacementModel, ...],
    region_layout: RegionLayoutModel | None,
    fields: tuple[Field, ...],
    owner: str,
) -> None:
    if not couplings:
        return
    if region_layout is None:
        raise ContractViolation(f"{owner} require region_layout")
    supported_models = _deterministic_models(models, owner=owner)
    model_names = {model_field_name(model) for model in supported_models}
    placement_names = {placement.field_name for placement in placements}
    placements_by_name = {placement.field_name: placement for placement in placements}
    vector_fields = {
        field.id.name for field in fields if field.role is FieldRole.VECTOR
    }
    seen_names: set[str] = set()
    seen_attributes: set[str] = set()
    region_counts = tuple(
        region.public_end - region.public_start + 1 for region in region_layout.regions
    )
    for coupling in couplings:
        if coupling.name in seen_names:
            raise ContractViolation(
                f"{owner} contains duplicate coupling name: {coupling.name}"
            )
        seen_names.add(coupling.name)
        if coupling.attribute_field_name in seen_attributes:
            raise ContractViolation(
                f"{owner} contains duplicate attribute: {coupling.attribute_field_name}"
            )
        seen_attributes.add(coupling.attribute_field_name)
        if coupling.vector_field_name != region_layout.vector_field_name:
            raise ContractViolation(
                f"{owner} vector_field_name must match region_layout"
            )
        if coupling.vector_field_name not in vector_fields:
            raise ContractViolation(
                f"{owner} references unknown vector field: {coupling.vector_field_name}"
            )
        if coupling.attribute_field_name not in model_names:
            raise ContractViolation(
                f"{owner} references field/property without deterministic model: "
                f"{coupling.attribute_field_name}"
            )
        if coupling.attribute_field_name not in placement_names:
            raise ContractViolation(
                f"{owner} references field/property without placement: "
                f"{coupling.attribute_field_name}"
            )
        if coupling.max_groups is not None and coupling.max_groups < 2:
            raise ContractViolation(f"{owner} max_groups must be >= 2")
        if not 0.0 <= coupling.target_rho <= 1.0:
            raise ContractViolation(f"{owner} target_rho must be within [0,1]")
        if not 0.0 <= coupling.observed_rho <= 1.0:
            raise ContractViolation(f"{owner} observed_rho must be within [0,1]")
        if len(coupling.region_group_counts) != len(region_counts):
            raise ContractViolation(
                f"{owner} region_group_counts row count must match region_layout"
            )
        if not coupling.groups:
            raise ContractViolation(f"{owner} groups must be non-empty")
        group_count = len(coupling.groups)
        group_counts = [
            group.local_end - group.local_start + 1 for group in coupling.groups
        ]
        if any(count <= 0 for count in group_counts):
            raise ContractViolation(f"{owner} group ranges must be non-empty")
        for row_index, row in enumerate(coupling.region_group_counts):
            if len(row) != group_count:
                raise ContractViolation(
                    f"{owner} region_group_counts column count must match groups"
                )
            if any(count < 0 for count in row):
                raise ContractViolation(
                    f"{owner} region_group_counts must be non-negative"
                )
            if sum(row) != region_counts[row_index]:
                raise ContractViolation(
                    f"{owner} region_group_counts row sums must match region sizes"
                )
        for column_index, expected_count in enumerate(group_counts):
            observed_count = sum(
                row[column_index] for row in coupling.region_group_counts
            )
            if observed_count != expected_count:
                raise ContractViolation(
                    f"{owner} region_group_counts column sums must match group sizes"
                )
        if sum(group_counts) != row_count:
            raise ContractViolation(f"{owner} groups must cover metadata row_count")
        try:
            replayed_placement = coupling_replay_placement(coupling, region_layout)
        except ContractViolation as exc:
            raise ContractViolation(
                f"{owner} cannot replay coupling placement: {exc}"
            ) from exc
        persisted_placement = placements_by_name[coupling.attribute_field_name]
        if _placement_coordinates(replayed_placement) != _placement_coordinates(
            persisted_placement
        ):
            raise ContractViolation(
                f"{owner} coupling replay must match persisted field placement"
            )


def _placement_coordinates(
    placement: FieldPlacementModel,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            segment.local_start,
            segment.local_end,
            segment.public_start,
            segment.public_end,
        )
        for segment in placement.segments
    )


def _deterministic_models(
    models: tuple[object, ...], *, owner: str
) -> tuple[DeterministicFieldModel, ...]:
    supported: list[DeterministicFieldModel] = []
    for model in models:
        if not isinstance(
            model,
            (
                UniformNumericModel,
                LognormalNumericModel,
                CategoricalBlocksModel,
                IntAffineBlocksModel,
            ),
        ):
            raise ContractViolation(
                f"{owner} uses unsupported deterministic model type: {type(model).__name__}"
            )
        supported.append(model)
    return tuple(supported)


__all__ = [
    "normalize_table_metadata",
    "validate_deterministic_fields",
    "validate_deterministic_model_type",
    "validate_field_placements",
    "validate_placement_segments",
    "validate_region_layout",
    "validate_vector_attribute_couplings",
]
