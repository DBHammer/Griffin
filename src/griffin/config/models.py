from __future__ import annotations

import enum
from typing import TypeAlias

import msgspec

from griffin.data.generation_request import (
    VectorRegionAssignmentBackend,
    VectorRegionClusteringBackend,
)
from griffin.query.shape import PredicateOperator
from griffin.query.task import PredicateEvolutionStrategyKind


class JobType(enum.Enum):
    DATA = "data"
    QUERY = "query"


class DataModelKind(enum.Enum):
    RELATIONAL = "relational"


class DatasetOutputFormat(enum.Enum):
    PARQUET = "parquet"
    CSV = "csv"


class RelationalDialect(enum.Enum):
    POSTGRESQL = "postgresql"


class DistributionKind(enum.Enum):
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    CATEGORICAL = "categorical"
    ZIPF = "zipf"


class IntValueDomainConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    min: int
    max: int


class QueryFamily(enum.Enum):
    RELATIONAL = "relational"
    FILTERED_ANN = "filtered_ann"


class QueryOutputKind(enum.Enum):
    TOP_K = "top_k"


class ConstraintMetric(enum.Enum):
    SELECTIVITY = "selectivity"
    GLS = "gls"


class Difficulty(enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY = 120


class DistributionConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: DistributionKind
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    sigma: float | None = None
    values: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    ndv: int | None = None
    s: float | None = None
    value_domain: IntValueDomainConfig | None = None


class RelationalColumnConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    type_name: str
    primary_key: bool = False
    distribution: DistributionConfig | None = None
    vector_spec: VectorColumnSpec | None = None


class VectorColumnSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    dim: int
    metric: str
    normalize: str
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


class NumericCorrelationPairConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    left: str
    right: str
    rho: float


class NumericCorrelationBlockConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    columns: tuple[str, ...]
    pairs: tuple[NumericCorrelationPairConfig, ...]


class CategoricalAssociationBlockConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    columns: tuple[str, ...]
    target_cramers_v: float


class VectorAttributeCouplingConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    vector: str
    attribute: str
    target_rho: float
    max_groups: int | None = None


class RelationalTableConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    rows: int
    columns: tuple[RelationalColumnConfig, ...]
    numeric_correlation_blocks: tuple[NumericCorrelationBlockConfig, ...] = ()
    categorical_association_blocks: tuple[CategoricalAssociationBlockConfig, ...] = ()
    vector_attribute_couplings: tuple[VectorAttributeCouplingConfig, ...] = ()


class RelationalDataSpecConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    tables: tuple[RelationalTableConfig, ...]


class BaseDataJobConfig(msgspec.Struct, frozen=True, omit_defaults=True, kw_only=True):
    type: JobType = JobType.DATA
    seed: int
    output_dir: str
    output_format: DatasetOutputFormat = DatasetOutputFormat.PARQUET


class RelationalDataJobConfig(BaseDataJobConfig, frozen=True, omit_defaults=True, kw_only=True):
    data_model: DataModelKind = DataModelKind.RELATIONAL
    dialect: RelationalDialect
    spec: RelationalDataSpecConfig

    def __post_init__(self) -> None:
        if self.type is not JobType.DATA:
            raise ValueError("relational data job must use type=data")
        if self.data_model is not DataModelKind.RELATIONAL:
            raise ValueError("relational data job must use data_model=relational")


class ConstraintConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    metric: ConstraintMetric
    min: float
    max: float


class QueryOutputConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: QueryOutputKind
    top_k: int


class PredicateCountRangeConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    min: int
    max: int


class QueryFormConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: QueryFamily
    predicate_count: PredicateCountRangeConfig | None = None
    operators: tuple[PredicateOperator, ...] = ()
    predicate_fields: tuple[str, ...] = ()
    max_in_values: int | None = None
    difficulty: Difficulty | None = None


class SearchConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    max_attempts_per_query: int = DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY
    predicate_evolution_strategy: PredicateEvolutionStrategyKind | None = None
    best_so_far_fallback: bool = False


class QuerySpecConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    family: QueryFamily
    count: int
    target_tables: tuple[str, ...] = ()
    form: QueryFormConfig | None = None
    constraints: tuple[ConstraintConfig, ...] = ()
    output: QueryOutputConfig | None = None
    search: SearchConfig | None = None


class QueryJobConfig(msgspec.Struct, frozen=True, omit_defaults=True):
    type: JobType
    seed: int
    input_dir: str
    spec: QuerySpecConfig
    output_dir: str | None = None


DataJobConfig: TypeAlias = RelationalDataJobConfig
JobConfig: TypeAlias = DataJobConfig | QueryJobConfig


__all__ = [
    "BaseDataJobConfig",
    "ConstraintConfig",
    "ConstraintMetric",
    "CategoricalAssociationBlockConfig",
    "DataJobConfig",
    "DataModelKind",
    "DatasetOutputFormat",
    "Difficulty",
    "DistributionConfig",
    "DistributionKind",
    "IntValueDomainConfig",
    "JobConfig",
    "JobType",
    "NumericCorrelationBlockConfig",
    "NumericCorrelationPairConfig",
    "PredicateEvolutionStrategyKind",
    "VectorAttributeCouplingConfig",
    "QueryFamily",
    "QueryFormConfig",
    "QueryJobConfig",
    "QueryOutputConfig",
    "QueryOutputKind",
    "QuerySpecConfig",
    "RelationalDialect",
    "RelationalColumnConfig",
    "RelationalDataSpecConfig",
    "RelationalDataJobConfig",
    "RelationalTableConfig",
    "SearchConfig",
    "VectorColumnSpec",
]
