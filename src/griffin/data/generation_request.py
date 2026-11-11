from __future__ import annotations

import enum

import msgspec

from griffin.model_ids import TableId


class VectorRegionAssignmentBackend(enum.Enum):
    SKLEARN = "sklearn"


class VectorRegionClusteringBackend(enum.Enum):
    SKLEARN = "sklearn"


class IntValueDomainRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    min: int
    max: int


class DistributionRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: str
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    sigma: float | None = None
    values: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    ndv: int | None = None
    s: float | None = None
    value_domain: IntValueDomainRequest | None = None


class FieldGenerationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    field_name: str
    distribution: DistributionRequest | None = None


class VectorGenerationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    field_name: str
    dimension: int
    metric: str
    normalize: str | None = None
    cluster_count: int | None = None
    cluster_size_distribution: str | None = None
    cluster_separation: float | None = None
    cluster_spread: float | None = None
    vector_length_variation: str | None = None
    region_count: int | None = None
    region_assignment_backend: VectorRegionAssignmentBackend = (
        VectorRegionAssignmentBackend.SKLEARN
    )
    region_clustering_backend: VectorRegionClusteringBackend = (
        VectorRegionClusteringBackend.SKLEARN
    )


class CorrelationBlockRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    columns: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]


class CategoricalAssociationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    columns: tuple[str, str]
    target_cramers_v: float


class VectorAttributeCouplingRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    vector_field: str
    attribute_field: str
    target_rho: float
    max_groups: int | None = None


class TableGenerationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    id: TableId
    fields: tuple[FieldGenerationRequest, ...] = ()
    vector: VectorGenerationRequest | None = None
    correlation_blocks: tuple[CorrelationBlockRequest, ...] = ()
    categorical_association_blocks: tuple[CategoricalAssociationRequest, ...] = ()
    vector_attribute_couplings: tuple[VectorAttributeCouplingRequest, ...] = ()

    def field(self, name: str) -> FieldGenerationRequest | None:
        for request in self.fields:
            if request.field_name == name:
                return request
        return None


class DataGenerationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    tables: tuple[TableGenerationRequest, ...] = ()

    def table(self, table_id: TableId) -> TableGenerationRequest | None:
        for request in self.tables:
            if request.id == table_id:
                return request
        return None

__all__ = [
    "CategoricalAssociationRequest",
    "CorrelationBlockRequest",
    "DataGenerationRequest",
    "DistributionRequest",
    "FieldGenerationRequest",
    "IntValueDomainRequest",
    "TableGenerationRequest",
    "VectorGenerationRequest",
    "VectorRegionAssignmentBackend",
    "VectorRegionClusteringBackend",
    "VectorAttributeCouplingRequest",
]
