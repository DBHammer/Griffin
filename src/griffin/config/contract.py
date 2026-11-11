from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from griffin.data.deterministic import categorical_counts
from griffin.infra.errors import ContractViolation

from .models import (
    DistributionConfig,
    DistributionKind,
    CategoricalAssociationBlockConfig,
    NumericCorrelationBlockConfig,
    RelationalColumnConfig,
    RelationalDataSpecConfig,
    RelationalTableConfig,
    VectorColumnSpec,
    VectorAttributeCouplingConfig,
)

RELATIONAL_LOGICAL_TYPES = frozenset({"int", "double", "text", "vector"})
NUMERIC_LOGICAL_TYPES = frozenset({"int", "double"})
CATEGORICAL_ASSOCIATION_LOGICAL_TYPES = frozenset({"int", "text"})
VECTOR_METRICS = frozenset({"l2", "cosine"})
VECTOR_NORMALIZE = frozenset({"none", "l2"})
VECTOR_CLUSTER_SIZE_DISTRIBUTION = frozenset({"uniform", "zipf_0_5", "zipf_1_0"})
VECTOR_LENGTH_VARIATION = frozenset({"none", "low", "medium", "high"})
MAX_CATEGORICAL_ASSOCIATION_POSITIVE_VALUES = 128


_OwnerFieldConfig = RelationalColumnConfig


def validate_relational_spec(spec: RelationalDataSpecConfig) -> None:
    seen_tables: set[str] = set()
    for table in spec.tables:
        if table.name in seen_tables:
            raise ContractViolation(f"duplicate table name: {table.name}")
        seen_tables.add(table.name)
        validate_relational_table(table)


def validate_relational_table(table: RelationalTableConfig) -> None:
    if table.rows < 1:
        raise ContractViolation(
            f"table {table.name}.rows must be >= 1 (got {table.rows})"
        )
    if not table.columns:
        raise ContractViolation(
            f"table {table.name} must declare at least one column"
        )
    seen_columns: set[str] = set()
    columns_by_name: dict[str, RelationalColumnConfig] = {}
    vector_columns = 0
    for column in table.columns:
        if column.name in seen_columns:
            raise ContractViolation(
                f"duplicate column name in table {table.name}: {column.name}"
            )
        seen_columns.add(column.name)
        columns_by_name[column.name] = column
        normalized_type = validate_relational_logical_type(column.type_name, field="column.type")
        if normalized_type == "vector":
            vector_columns += 1
            if column.primary_key:
                raise ContractViolation(
                    f"column.primary_key is not allowed for vector column {table.name}.{column.name}"
                )
            if column.distribution is not None:
                raise ContractViolation(
                    f"column.distribution is not allowed for vector column {table.name}.{column.name}"
                )
            if column.vector_spec is None:
                raise ContractViolation(
                    f"column.vector_spec must be set for vector column {table.name}.{column.name}"
                )
            validate_vector_contract(
                column.vector_spec,
                field=f"{table.name}.{column.name}.vector_spec",
            )
            validate_vector_region_count(
                column.vector_spec,
                row_count=table.rows,
                field=f"{table.name}.{column.name}.vector_spec",
            )
            continue
        if column.vector_spec is not None:
            raise ContractViolation(
                f"column.vector_spec is only allowed for vector column {table.name}.{column.name}"
            )
        if column.primary_key:
            if column.distribution is not None:
                raise ContractViolation(
                    f"column.distribution is not allowed for primary_key column {table.name}.{column.name}"
                )
        if column.distribution is not None:
            validate_distribution_contract(
                column.distribution,
                field=f"{table.name}.{column.name}.distribution",
            )
            validate_distribution_compatibility(column, table_name=table.name)
            validate_distribution_cardinality(
                column.distribution,
                row_count=table.rows,
                field=f"{table.name}.{column.name}.distribution",
            )
    if vector_columns > 1:
        raise ContractViolation(f"table {table.name} may declare at most one vector column")

    correlation_blocks = validate_owner_correlation_blocks(
        owner_name=f"table {table.name}",
        blocks=table.numeric_correlation_blocks,
        fields_by_name=columns_by_name,
        type_name=lambda column: column.type_name,
        distribution=lambda column: column.distribution,
    )
    association_fields = validate_owner_categorical_associations(
        owner_name=f"table {table.name}",
        blocks=table.categorical_association_blocks,
        fields_by_name=columns_by_name,
        type_name=lambda column: column.type_name,
        distribution=lambda column: column.distribution,
        row_count=table.rows,
    )
    validate_categorical_association_conflicts(
        owner_name=f"table {table.name}",
        association_fields=association_fields,
        correlation_blocks=correlation_blocks,
        couplings=table.vector_attribute_couplings,
    )
    validate_owner_vector_attribute_couplings(
        owner_name=f"table {table.name}",
        couplings=table.vector_attribute_couplings,
        fields_by_name=columns_by_name,
        type_name=lambda column: column.type_name,
        distribution=lambda column: column.distribution,
        correlation_blocks=correlation_blocks,
    )


def validate_distribution_contract(
    distribution: DistributionConfig,
    *,
    field: str,
) -> None:
    kind = distribution.kind
    if kind not in {DistributionKind.UNIFORM, DistributionKind.ZIPF}:
        require_absent(
            distribution.value_domain,
            field=f"{field}.value_domain",
            kind=kind.value,
        )
    if kind is DistributionKind.UNIFORM:
        require_absent(distribution.mean, field=f"{field}.mean", kind=kind.value)
        require_absent(distribution.sigma, field=f"{field}.sigma", kind=kind.value)
        require_absent(distribution.values, field=f"{field}.values", kind=kind.value)
        require_absent(distribution.weights, field=f"{field}.weights", kind=kind.value)
        require_absent(distribution.s, field=f"{field}.s", kind=kind.value)
        continuous = distribution.min is not None or distribution.max is not None
        affine = distribution.ndv is not None or distribution.value_domain is not None
        if continuous == affine:
            raise ContractViolation(
                f"{field} must declare exactly one uniform domain: min/max or ndv/value_domain"
            )
        if continuous:
            require_present(distribution.min, field=f"{field}.min")
            require_present(distribution.max, field=f"{field}.max")
            if distribution.min is None or distribution.max is None:
                raise ContractViolation(f"{field}.min and {field}.max must be set")
            if distribution.min > distribution.max:
                raise ContractViolation(f"{field}.min must be <= {field}.max")
        else:
            require_present(distribution.ndv, field=f"{field}.ndv")
            require_present(distribution.value_domain, field=f"{field}.value_domain")
            if distribution.ndv is None or distribution.value_domain is None:
                raise ContractViolation(
                    f"{field}.ndv and {field}.value_domain must be set"
                )
            if distribution.ndv < 1:
                raise ContractViolation(f"{field}.ndv must be >= 1")
            if not _is_strict_int(distribution.value_domain.min):
                raise ContractViolation(f"{field}.value_domain.min must be an integer")
            if not _is_strict_int(distribution.value_domain.max):
                raise ContractViolation(f"{field}.value_domain.max must be an integer")
            if distribution.value_domain.min > distribution.value_domain.max:
                raise ContractViolation(
                    f"{field}.value_domain.min must be <= {field}.value_domain.max"
                )
    elif kind is DistributionKind.LOGNORMAL:
        require_absent(distribution.min, field=f"{field}.min", kind=kind.value)
        require_absent(distribution.max, field=f"{field}.max", kind=kind.value)
        require_present(distribution.mean, field=f"{field}.mean")
        require_present(distribution.sigma, field=f"{field}.sigma")
        require_absent(distribution.values, field=f"{field}.values", kind=kind.value)
        require_absent(distribution.weights, field=f"{field}.weights", kind=kind.value)
        require_absent(distribution.ndv, field=f"{field}.ndv", kind=kind.value)
        require_absent(distribution.s, field=f"{field}.s", kind=kind.value)
        if distribution.sigma is None:
            raise ContractViolation(f"{field}.sigma must be set")
        if distribution.sigma <= 0.0:
            raise ContractViolation(f"{field}.sigma must be > 0")
    elif kind is DistributionKind.CATEGORICAL:
        require_absent(distribution.min, field=f"{field}.min", kind=kind.value)
        require_absent(distribution.max, field=f"{field}.max", kind=kind.value)
        require_absent(distribution.mean, field=f"{field}.mean", kind=kind.value)
        require_absent(distribution.sigma, field=f"{field}.sigma", kind=kind.value)
        if not distribution.values:
            raise ContractViolation(f"{field}.values must be non-empty")
        if not distribution.weights:
            raise ContractViolation(f"{field}.weights must be non-empty")
        if len(distribution.values) != len(distribution.weights):
            raise ContractViolation(f"{field}.weights must align with {field}.values")
        if any(weight <= 0.0 for weight in distribution.weights):
            raise ContractViolation(f"{field}.weights must all be > 0")
        require_absent(distribution.ndv, field=f"{field}.ndv", kind=kind.value)
        require_absent(distribution.s, field=f"{field}.s", kind=kind.value)
    elif kind is DistributionKind.ZIPF:
        require_absent(distribution.min, field=f"{field}.min", kind=kind.value)
        require_absent(distribution.max, field=f"{field}.max", kind=kind.value)
        require_absent(distribution.mean, field=f"{field}.mean", kind=kind.value)
        require_absent(distribution.sigma, field=f"{field}.sigma", kind=kind.value)
        require_absent(distribution.values, field=f"{field}.values", kind=kind.value)
        require_absent(distribution.weights, field=f"{field}.weights", kind=kind.value)
        require_present(distribution.ndv, field=f"{field}.ndv")
        require_present(distribution.s, field=f"{field}.s")
        if distribution.ndv is None or distribution.s is None:
            raise ContractViolation(f"{field}.ndv and {field}.s must be set")
        if distribution.ndv < 1:
            raise ContractViolation(f"{field}.ndv must be >= 1")
        if distribution.s <= 0.0:
            raise ContractViolation(f"{field}.s must be > 0")
        if distribution.value_domain is not None:
            if not _is_strict_int(distribution.value_domain.min):
                raise ContractViolation(f"{field}.value_domain.min must be an integer")
            if not _is_strict_int(distribution.value_domain.max):
                raise ContractViolation(f"{field}.value_domain.max must be an integer")
            if distribution.value_domain.min > distribution.value_domain.max:
                raise ContractViolation(f"{field}.value_domain.min must be <= {field}.value_domain.max")
    else:
        raise ContractViolation(f"{field}.type is unsupported: {kind.value}")


def validate_distribution_compatibility(
    column: RelationalColumnConfig,
    *,
    table_name: str,
) -> None:
    distribution = column.distribution
    if distribution is None:
        return
    field = f"{table_name}.{column.name}.distribution"
    normalized_type = normalize_type_name(column.type_name)
    kind = distribution.kind

    if normalized_type == "int":
        if kind is DistributionKind.LOGNORMAL:
            return
        if kind is DistributionKind.UNIFORM:
            domain = distribution.value_domain
            if domain is None or distribution.ndv is None:
                raise ContractViolation(
                    f"{field}.ndv and value_domain must be set for int uniform columns"
                )
            if distribution.min is not None or distribution.max is not None:
                raise ContractViolation(
                    f"{field}.min/max are not allowed for int uniform columns"
                )
            _validate_int_affine_domain(
                domain_min=domain.min,
                domain_max=domain.max,
                ndv=distribution.ndv,
                field=field,
            )
            return
        if kind is DistributionKind.CATEGORICAL:
            invalid = [value for value in distribution.values if not is_int_label(value)]
            if invalid:
                raise ContractViolation(
                    f"{field}.values must contain only integer labels for int categorical columns"
                )
            return
        if kind is DistributionKind.ZIPF:
            domain = distribution.value_domain
            if domain is None:
                raise ContractViolation(f"{field}.value_domain must be set for int zipf columns")
            if distribution.ndv is None:
                raise ContractViolation(f"{field}.ndv must be set for int zipf columns")
            _validate_int_affine_domain(
                domain_min=domain.min,
                domain_max=domain.max,
                ndv=distribution.ndv,
                field=field,
            )
            return
        raise ContractViolation(
            f"{field}.type is not supported for int; allowed: uniform, lognormal, categorical, zipf"
        )

    if normalized_type == "double":
        if kind is DistributionKind.UNIFORM:
            if distribution.min is None or distribution.max is None:
                raise ContractViolation(
                    f"{field}.min and max must be set for double uniform columns"
                )
            if distribution.ndv is not None or distribution.value_domain is not None:
                raise ContractViolation(
                    f"{field}.ndv/value_domain are not allowed for double uniform columns"
                )
            return
        if kind is DistributionKind.LOGNORMAL:
            return
        raise ContractViolation(
            f"{field}.type is not supported for double; allowed: uniform, lognormal"
        )

    if normalized_type == "text":
        if kind in {DistributionKind.CATEGORICAL, DistributionKind.ZIPF}:
            if kind is DistributionKind.ZIPF and distribution.value_domain is not None:
                raise ContractViolation(f"{field}.value_domain is not allowed for text zipf columns")
            return
        raise ContractViolation(
            f"{field}.type is not supported for text; allowed: categorical, zipf"
        )


def validate_distribution_cardinality(
    distribution: DistributionConfig,
    *,
    row_count: int,
    field: str,
) -> None:
    if distribution.kind not in {DistributionKind.UNIFORM, DistributionKind.ZIPF}:
        return
    if distribution.ndv is None:
        return
    if row_count < distribution.ndv:
        raise ContractViolation(
            f"{field}.ndv must be <= owner row_count; got ndv={distribution.ndv}, row_count={row_count}"
        )


def _validate_int_affine_domain(
    *,
    domain_min: int,
    domain_max: int,
    ndv: int,
    field: str,
) -> None:
    if domain_max - domain_min < ndv - 1:
        raise ContractViolation(
            f"{field}.value_domain must contain at least ndv distinct integers; "
            f"got min={domain_min}, max={domain_max}, ndv={ndv}"
        )


def validate_vector_contract(spec: VectorColumnSpec, *, field: str) -> None:
    if spec.dim <= 0:
        raise ContractViolation(f"{field}.dim must be > 0")
    if spec.metric not in VECTOR_METRICS:
        allowed = ", ".join(sorted(VECTOR_METRICS))
        raise ContractViolation(f"{field}.metric must be one of: {allowed}")
    if spec.normalize is None:
        raise ContractViolation(f"{field}.normalize must be set")
    if spec.normalize not in VECTOR_NORMALIZE:
        allowed = ", ".join(sorted(VECTOR_NORMALIZE))
        raise ContractViolation(f"{field}.normalize must be one of: {allowed}")
    if spec.cluster_count is not None and spec.cluster_count < 1:
        raise ContractViolation(f"{field}.cluster_count must be >= 1")
    if spec.region_count is not None and spec.region_count < 1:
        raise ContractViolation(f"{field}.region_count must be >= 1")
    if (
        spec.cluster_size_distribution is not None
        and spec.cluster_size_distribution not in VECTOR_CLUSTER_SIZE_DISTRIBUTION
    ):
        allowed = ", ".join(sorted(VECTOR_CLUSTER_SIZE_DISTRIBUTION))
        raise ContractViolation(
            f"{field}.cluster_size_distribution must be one of: {allowed}"
        )
    if spec.cluster_separation is not None and spec.cluster_separation <= 0.0:
        raise ContractViolation(f"{field}.cluster_separation must be > 0")
    if spec.cluster_spread is not None and spec.cluster_spread <= 0.0:
        raise ContractViolation(f"{field}.cluster_spread must be > 0")
    if (
        spec.vector_length_variation is not None
        and spec.vector_length_variation not in VECTOR_LENGTH_VARIATION
    ):
        allowed = ", ".join(sorted(VECTOR_LENGTH_VARIATION))
        raise ContractViolation(
            f"{field}.vector_length_variation must be one of: {allowed}"
        )


def validate_vector_region_count(spec: VectorColumnSpec, *, row_count: int, field: str) -> None:
    if row_count < 0:
        raise ContractViolation(f"{field}.row_count must be >= 0")
    if spec.region_count is not None and spec.region_count > row_count:
        raise ContractViolation(f"{field}.region_count must be <= row_count")


def validate_correlation_block(
    table_name: str,
    block: NumericCorrelationBlockConfig,
    columns_by_name: dict[str, RelationalColumnConfig],
) -> None:
    validate_correlation_block_for_owner(
        owner_name=f"table {table_name}",
        block=block,
        fields_by_name=columns_by_name,
        type_name=lambda column: column.type_name,
        distribution=lambda column: column.distribution,
    )


def validate_owner_correlation_blocks(
    *,
    owner_name: str,
    blocks: Sequence[NumericCorrelationBlockConfig],
    fields_by_name: Mapping[str, _OwnerFieldConfig],
    type_name: Callable[[_OwnerFieldConfig], str],
    distribution: Callable[[_OwnerFieldConfig], DistributionConfig | None],
) -> tuple[frozenset[str], ...]:
    seen_blocks: set[str] = set()
    seen_fields: set[str] = set()
    column_sets: list[frozenset[str]] = []
    for block in blocks:
        if block.name in seen_blocks:
            raise ContractViolation(
                f"duplicate numeric_correlation_blocks[].name in {owner_name}: {block.name}"
            )
        seen_blocks.add(block.name)
        overlapping = sorted(seen_fields & set(block.columns))
        if overlapping:
            raise ContractViolation(
                "numeric_correlation_blocks[].columns must not overlap across "
                f"blocks in {owner_name}: {overlapping}"
            )
        validate_correlation_block_for_owner(
            owner_name=owner_name,
            block=block,
            fields_by_name=fields_by_name,
            type_name=type_name,
            distribution=distribution,
        )
        seen_fields.update(block.columns)
        column_sets.append(frozenset(block.columns))
    return tuple(column_sets)


def validate_correlation_block_for_owner(
    *,
    owner_name: str,
    block: NumericCorrelationBlockConfig,
    fields_by_name: Mapping[str, _OwnerFieldConfig],
    type_name: Callable[[_OwnerFieldConfig], str],
    distribution: Callable[[_OwnerFieldConfig], DistributionConfig | None],
) -> None:
    if len(block.columns) < 2:
        raise ContractViolation(
            f"numeric_correlation_blocks[].columns must contain at least 2 columns in {owner_name}"
        )
    if len(set(block.columns)) != len(block.columns):
        raise ContractViolation(
            f"numeric_correlation_blocks[].columns must not contain duplicates in {owner_name}"
        )
    for column_name in block.columns:
        column = fields_by_name.get(column_name)
        if column is None:
            raise ContractViolation(
                "numeric_correlation_blocks[].columns references unknown "
                f"column/field/property in {owner_name}: {column_name}"
            )
        if normalize_type_name(type_name(column)) not in NUMERIC_LOGICAL_TYPES:
            raise ContractViolation(
                "numeric_correlation_blocks[].columns must reference numeric "
                f"columns/fields/properties in {owner_name}: {column_name}"
            )
        field_distribution = distribution(column)
        if field_distribution is None or field_distribution.kind not in {
            DistributionKind.UNIFORM,
            DistributionKind.LOGNORMAL,
        }:
            raise ContractViolation(
                "numeric_correlation_blocks[].columns must reference numeric fields/properties "
                f"with distribution.type=uniform or lognormal in {owner_name}: {column_name}"
            )
    column_set = set(block.columns)
    expected_pairs = {
        frozenset((left, right))
        for index, left in enumerate(block.columns)
        for right in block.columns[index + 1 :]
    }
    seen_pairs: set[frozenset[str]] = set()
    for pair in block.pairs:
        if pair.left == pair.right:
            raise ContractViolation(
                f"numeric_correlation_blocks[].pairs must reference two distinct columns in {owner_name}: {pair.left}"
            )
        if pair.left not in column_set or pair.right not in column_set:
            raise ContractViolation(
                "numeric_correlation_blocks[].pairs must reference columns listed in "
                f"numeric_correlation_blocks[].columns in {owner_name}: {pair.left}, {pair.right}"
            )
        key = frozenset((pair.left, pair.right))
        if key in seen_pairs:
            raise ContractViolation(
                f"numeric_correlation_blocks[].pairs contains duplicate pair in {owner_name}: {pair.left}, {pair.right}"
            )
        seen_pairs.add(key)
        if not -1.0 <= pair.rho <= 1.0:
            raise ContractViolation(
                f"numeric_correlation_blocks[].pairs rho must be within [-1,1] in {owner_name}: {pair.rho:g}"
            )
    if seen_pairs != expected_pairs:
        missing = sorted(
            tuple(sorted(pair))
            for pair in expected_pairs - seen_pairs
        )
        raise ContractViolation(
            f"numeric_correlation_blocks[].pairs must cover every unordered column pair in {owner_name}: missing {missing}"
        )
    matrix = correlation_pairs_to_matrix(columns=block.columns, pairs=block.pairs)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) < -1e-8:
        raise ContractViolation(
            f"numeric_correlation_blocks[].pairs must form a positive semidefinite correlation matrix in {owner_name}"
        )


def correlation_pairs_to_matrix(
    *,
    columns: Sequence[str],
    pairs: Sequence[object],
) -> tuple[tuple[float, ...], ...]:
    index_by_column = {column: index for index, column in enumerate(columns)}
    size = len(columns)
    matrix = np.eye(size, dtype=float)
    for pair in pairs:
        left = getattr(pair, "left")
        right = getattr(pair, "right")
        rho = float(getattr(pair, "rho"))
        left_index = index_by_column[left]
        right_index = index_by_column[right]
        matrix[left_index, right_index] = rho
        matrix[right_index, left_index] = rho
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def validate_owner_categorical_associations(
    *,
    owner_name: str,
    blocks: Sequence[CategoricalAssociationBlockConfig],
    fields_by_name: Mapping[str, _OwnerFieldConfig],
    type_name: Callable[[_OwnerFieldConfig], str],
    distribution: Callable[[_OwnerFieldConfig], DistributionConfig | None],
    row_count: int,
) -> tuple[frozenset[str], ...]:
    seen_names: set[str] = set()
    seen_fields: set[str] = set()
    field_sets: list[frozenset[str]] = []
    for block in blocks:
        if block.name in seen_names:
            raise ContractViolation(
                f"duplicate categorical_association_blocks[].name in {owner_name}: {block.name}"
            )
        seen_names.add(block.name)
        block_fields = frozenset(block.columns)
        if len(block.columns) != 2 or len(block_fields) != 2:
            raise ContractViolation(
                f"categorical_association_blocks[].columns must contain exactly two distinct fields in {owner_name}: {block.name}"
            )
        overlapping = sorted(seen_fields & block_fields)
        if overlapping:
            raise ContractViolation(
                "categorical_association_blocks[].columns must not overlap across "
                f"blocks in {owner_name}: {overlapping}"
            )
        validate_categorical_association_block(
            owner_name=owner_name,
            block=block,
            fields_by_name=fields_by_name,
            type_name=type_name,
            distribution=distribution,
            row_count=row_count,
        )
        seen_fields.update(block_fields)
        field_sets.append(block_fields)
    return tuple(field_sets)


def validate_categorical_association_block(
    *,
    owner_name: str,
    block: CategoricalAssociationBlockConfig,
    fields_by_name: Mapping[str, _OwnerFieldConfig],
    type_name: Callable[[_OwnerFieldConfig], str],
    distribution: Callable[[_OwnerFieldConfig], DistributionConfig | None],
    row_count: int,
) -> None:
    if not 0.0 <= block.target_cramers_v <= 1.0:
        raise ContractViolation(
            f"categorical_association_blocks[].target_cramers_v must be within [0,1] in {owner_name}: {block.name}"
        )
    for column_name in block.columns:
        field = fields_by_name.get(column_name)
        if field is None:
            raise ContractViolation(
                "categorical_association_blocks[].columns references unknown "
                f"field/property in {owner_name}: {column_name}"
            )
        normalized_type = normalize_type_name(type_name(field))
        if normalized_type not in CATEGORICAL_ASSOCIATION_LOGICAL_TYPES:
            raise ContractViolation(
                "categorical_association_blocks[].columns must reference int/text "
                f"categorical fields/properties in {owner_name}: {column_name}"
            )
        field_distribution = distribution(field)
        if field_distribution is None or field_distribution.kind is not DistributionKind.CATEGORICAL:
            raise ContractViolation(
                "categorical_association_blocks[].columns must declare distribution.type=categorical "
                f"in {owner_name}: {column_name}"
            )
        if len(field_distribution.values) < 2:
            raise ContractViolation(
                "categorical_association_blocks[].columns categorical values must contain at least "
                f"2 values in {owner_name}: {column_name}"
            )
        positive_count = sum(
            1
            for count in categorical_counts(
                labels=field_distribution.values,
                weights=field_distribution.weights,
                row_count=row_count,
            )
            if count > 0
        )
        if positive_count < 2:
            raise ContractViolation(
                "categorical_association_blocks[].columns categorical values must keep at least "
                f"2 positive-count values after weight rounding in {owner_name}: {column_name}"
            )
        if positive_count > MAX_CATEGORICAL_ASSOCIATION_POSITIVE_VALUES:
            raise ContractViolation(
                "categorical_association_blocks[].columns categorical values must keep at most "
                f"{MAX_CATEGORICAL_ASSOCIATION_POSITIVE_VALUES} positive-count values after "
                f"weight rounding in {owner_name}: {column_name}"
            )


def validate_categorical_association_conflicts(
    *,
    owner_name: str,
    association_fields: Sequence[frozenset[str]],
    correlation_blocks: Sequence[frozenset[str]],
    couplings: Sequence[VectorAttributeCouplingConfig],
) -> None:
    association_flat = set().union(*association_fields) if association_fields else set()
    for block in correlation_blocks:
        overlapping = sorted(block & association_flat)
        if overlapping:
            raise ContractViolation(
                "categorical_association_blocks fields must not participate in "
                f"numeric_correlation_blocks in {owner_name}: {overlapping}"
            )
    coupling_attributes = {coupling.attribute for coupling in couplings}
    for block in association_fields:
        coupled = sorted(block & coupling_attributes)
        if len(coupled) > 1:
            raise ContractViolation(
                "categorical_association_blocks columns may include at most one "
                f"vector_attribute_couplings[].attribute in {owner_name}: {coupled}"
            )


def validate_owner_vector_attribute_couplings(
    *,
    owner_name: str,
    couplings: Sequence[VectorAttributeCouplingConfig],
    fields_by_name: Mapping[str, _OwnerFieldConfig],
    type_name: Callable[[_OwnerFieldConfig], str],
    distribution: Callable[[_OwnerFieldConfig], DistributionConfig | None],
    correlation_blocks: Sequence[frozenset[str]],
) -> None:
    seen_names: set[str] = set()
    seen_attributes: set[str] = set()
    for coupling in couplings:
        if coupling.name in seen_names:
            raise ContractViolation(
                f"duplicate vector_attribute_couplings[].name in {owner_name}: {coupling.name}"
            )
        seen_names.add(coupling.name)
        vector_field = fields_by_name.get(coupling.vector)
        if vector_field is None:
            raise ContractViolation(
                "vector_attribute_couplings[].vector references unknown "
                f"field/property in {owner_name}: {coupling.vector}"
            )
        if normalize_type_name(type_name(vector_field)) != "vector":
            raise ContractViolation(
                "vector_attribute_couplings[].vector must reference a vector "
                f"field/property in {owner_name}: {coupling.vector}"
            )
        attribute = fields_by_name.get(coupling.attribute)
        if attribute is None:
            raise ContractViolation(
                "vector_attribute_couplings[].attribute references unknown "
                f"field/property in {owner_name}: {coupling.attribute}"
            )
        attribute_type = normalize_type_name(type_name(attribute))
        if attribute_type == "vector":
            raise ContractViolation(
                "vector_attribute_couplings[].attribute must reference a scalar "
                f"field/property in {owner_name}: {coupling.attribute}"
            )
        if coupling.attribute in seen_attributes:
            raise ContractViolation(
                f"vector_attribute_couplings[].attribute must be unique in {owner_name}: {coupling.attribute}"
            )
        seen_attributes.add(coupling.attribute)
        if coupling.max_groups is not None and coupling.max_groups < 2:
            raise ContractViolation(
                f"vector_attribute_couplings[].max_groups must be >= 2 in {owner_name}: {coupling.name}"
            )
        if not has_supported_deterministic_model(attribute_type, distribution(attribute)):
            raise ContractViolation(
                "vector_attribute_couplings[].attribute must use a supported "
                f"deterministic distribution in {owner_name}: {coupling.attribute}"
            )
    coupling_attributes = {
        coupling.attribute
        for coupling in couplings
    }
    for block in correlation_blocks:
        coupled = sorted(block & coupling_attributes)
        if len(coupled) > 1:
            raise ContractViolation(
                "numeric_correlation_blocks columns may include at most one "
                f"vector_attribute_couplings[].attribute in {owner_name}: {coupled}"
            )


def has_supported_deterministic_model(
    normalized_type: str,
    distribution: DistributionConfig | None,
) -> bool:
    if distribution is None:
        return False
    if normalized_type in {"int", "double"} and distribution.kind in {
        DistributionKind.UNIFORM,
        DistributionKind.LOGNORMAL,
    }:
        return True
    if normalized_type in {"int", "text"} and distribution.kind in {
        DistributionKind.CATEGORICAL,
        DistributionKind.ZIPF,
    }:
        return True
    return False


def validate_relational_logical_type(type_name: str, *, field: str) -> str:
    normalized = normalize_type_name(type_name)
    if normalized not in RELATIONAL_LOGICAL_TYPES:
        allowed = ", ".join(sorted(RELATIONAL_LOGICAL_TYPES))
        raise ContractViolation(f"{field} must be one of: {allowed}")
    return normalized


def normalize_type_name(type_name: str) -> str:
    return type_name.strip().lower()


def require_present(value: object, *, field: str) -> None:
    if value is None:
        raise ContractViolation(f"{field} must be set")


def require_absent(value: object, *, field: str, kind: str) -> None:
    if value is None:
        return
    if isinstance(value, tuple) and not value:
        return
    raise ContractViolation(f"{field} is not allowed for {kind}")


def is_int_label(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        parsed = float(text)
    except ValueError:
        return False
    return parsed.is_integer()


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "correlation_pairs_to_matrix",
    "validate_categorical_association_block",
    "validate_categorical_association_conflicts",
    "validate_correlation_block",
    "validate_correlation_block_for_owner",
    "validate_distribution_compatibility",
    "validate_distribution_contract",
    "validate_owner_correlation_blocks",
    "validate_owner_vector_attribute_couplings",
    "validate_relational_logical_type",
    "validate_relational_spec",
    "validate_vector_contract",
    "has_supported_deterministic_model",
    "normalize_type_name",
]
