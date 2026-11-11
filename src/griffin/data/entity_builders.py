"""Relational table materialization."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
from typing import Protocol, TypeAlias, cast

import numpy as np

from griffin.data.generation_request import (
    CategoricalAssociationRequest,
    CorrelationBlockRequest,
    FieldGenerationRequest,
    TableGenerationRequest,
    VectorAttributeCouplingRequest,
    VectorGenerationRequest,
)
from griffin.data.deterministic import deterministic_model_for_field, materialize_supported_field
from griffin.dataset.deterministic import (
    CategoricalBlocksModel,
    CategoryBlock,
    DeterministicFieldModel,
)
from griffin.data.values import (
    generate_correlation_block,
    generate_field_values,
    generate_vector_values,
    generator_for,
)
from griffin.dataset.entities import Table
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import IntType, TextType
from griffin.ir.field import Field, FieldRole

ColumnValues: TypeAlias = list[object] | np.ndarray
ColumnMapping: TypeAlias = dict[str, ColumnValues]

CATEGORICAL_ASSOCIATION_TOLERANCE = 0.08
_EXACT_ASSOCIATION_TOTAL_LIMIT = 200
_EXACT_ASSOCIATION_TABLE_LIMIT = 200_000
_IPF_MAX_ITERATIONS = 512
_IPF_THETA_SEARCH_ITERATIONS = 48
_IPF_THETA_MAX = 256.0
_LOCAL_REPAIR_MAX_STEPS = 128
_LOCAL_REPAIR_SWAP_LIMIT = 50_000
_LOCAL_REPAIR_CANDIDATE_LIMIT = 3


@dataclass(frozen=True, slots=True)
class _AssociationCandidate:
    matrix: tuple[tuple[int, ...], ...]
    observed: float
    preference: float


class _FieldOwnerRequest(Protocol):
    correlation_blocks: tuple[CorrelationBlockRequest, ...]
    vector_attribute_couplings: tuple[VectorAttributeCouplingRequest, ...]
    categorical_association_blocks: tuple[CategoricalAssociationRequest, ...]

    def field(self, name: str) -> FieldGenerationRequest | None: ...


class _TableFieldOwnerRequest(_FieldOwnerRequest, Protocol):
    vector: VectorGenerationRequest | None


def _materialize_vectors(
    fields: tuple[Field, ...],
    row_count: int,
    *,
    seed: int,
    get_vector_request: Callable[[str], VectorGenerationRequest | None],
    structured_seed_base: int,
    generator_seed_base: int,
) -> ColumnMapping:
    out: ColumnMapping = {}
    for offset, field in enumerate(fields):
        if field.role is not FieldRole.VECTOR:
            continue
        vector_request = get_vector_request(field.id.name)
        if vector_request is not None:
            out[field.id.name] = generate_vector_values(
                row_count,
                request=vector_request,
                seed=seed + structured_seed_base + offset,
            )
            continue
        generated = generator_for(field.type, seed=seed + generator_seed_base + offset).generate(row_count)
        out[field.id.name] = np.asarray(generated, dtype="float32")
    return out


def materialize_table_fields(
    table: Table,
    row_count: int,
    *,
    seed: int = 0,
    request: TableGenerationRequest | None = None,
) -> tuple[ColumnMapping, tuple[DeterministicFieldModel, ...]]:
    return materialize_owner_scalar_fields(
        table.fields,
        row_count,
        seed=seed,
        request=_table_field_owner_request(request),
    )


def materialize_table_vectors(
    table: Table,
    row_count: int,
    *,
    seed: int = 0,
    request: TableGenerationRequest | None = None,
) -> ColumnMapping:
    single = request.vector if request is not None else None
    def _get(name: str) -> VectorGenerationRequest | None:
        if single is not None and single.field_name == name:
            return single
        return None
    return _materialize_vectors(
        table.fields, row_count, seed=seed,
        get_vector_request=_get,
        structured_seed_base=4001,
        generator_seed_base=5000,
    )


def _materialize_scalar_fields(
    fields: tuple[Field, ...],
    row_count: int,
    *,
    seed: int,
    handled: set[str],
    request_field_lookup: Callable[[str], FieldGenerationRequest | None],
) -> tuple[ColumnMapping, list[DeterministicFieldModel]]:
    out: ColumnMapping = {}
    deterministic_models: list[DeterministicFieldModel] = []
    for offset, field in enumerate(fields):
        name = field.id.name
        if field.role is FieldRole.VECTOR:
            continue
        if name in handled:
            continue
        if field.role == FieldRole.PRIMARY_KEY and isinstance(field.type, IntType):
            out[name] = [int(value) for value in range(row_count)]
            continue
        field_request = request_field_lookup(name)
        model = (
            deterministic_model_for_field(field, row_count, request=field_request, seed=seed)
            if field_request is not None
            else None
        )
        if model is not None and field_request is not None:
            out[name] = materialize_supported_field(field, row_count, model=model)
            deterministic_models.append(model)
            continue
        out[name] = generate_field_values(field, row_count, seed=seed + offset, request=field_request)
    return out, deterministic_models


def materialize_owner_scalar_fields(
    fields: tuple[Field, ...],
    row_count: int,
    *,
    seed: int = 0,
    request: _FieldOwnerRequest | None = None,
) -> tuple[ColumnMapping, tuple[DeterministicFieldModel, ...]]:
    out: ColumnMapping = {}
    handled: set[str] = set()
    fields_by_name = {field.id.name: field for field in fields}
    deterministic_models: list[DeterministicFieldModel] = []

    if request is not None:
        for association_block in request.categorical_association_blocks:
            generated, models = _generate_categorical_association_block(
                block=association_block,
                fields_by_name=fields_by_name,
                request=request,
                row_count=row_count,
            )
            out.update(generated)
            deterministic_models.extend(models)
            handled.update(association_block.columns)

        for offset, correlation_block in enumerate(request.correlation_blocks):
            block_fields = tuple(
                fields_by_name[column]
                for column in correlation_block.columns
                if column in fields_by_name
            )
            if len(block_fields) != len(correlation_block.columns):
                continue
            block_requests = tuple(request.field(column) for column in correlation_block.columns)
            block_models = _correlation_block_models(
                block_fields=block_fields,
                block_requests=block_requests,
                row_count=row_count,
                seed=seed,
            )
            anchor_field = _correlation_vector_anchor_field(correlation_block, request)
            out.update(
                generate_correlation_block(
                    block_fields,
                    row_count,
                    block=correlation_block,
                    models=block_models,
                    seed=seed + 101 * (offset + 1),
                    anchor_field=anchor_field,
                )
            )
            handled.update(correlation_block.columns)
            deterministic_models.extend(block_models)

    field_out, scalar_models = _materialize_scalar_fields(
        fields,
        row_count,
        seed=seed,
        handled=handled,
        request_field_lookup=(lambda name: request.field(name)) if request is not None else (lambda _: None),
    )
    out.update(field_out)
    deterministic_models.extend(scalar_models)
    return out, tuple(deterministic_models)


def _correlation_block_models(
    *,
    block_fields: tuple[Field, ...],
    block_requests: tuple[FieldGenerationRequest | None, ...],
    row_count: int,
    seed: int,
) -> tuple[DeterministicFieldModel, ...]:
    return tuple(
        _correlation_numeric_model(
            field,
            row_count,
            request=field_request,
            seed=seed,
        )
        for field, field_request in zip(block_fields, block_requests, strict=True)
    )


def _correlation_numeric_model(
    field: Field,
    row_count: int,
    *,
    request: FieldGenerationRequest | None,
    seed: int,
) -> DeterministicFieldModel:
    if request is None:
        raise ContractViolation(
            f"correlation block field {field.id.name} requires an explicit numeric distribution"
        )
    model = deterministic_model_for_field(
        field,
        row_count,
        request=request,
        seed=seed,
    )
    if model is None:
        raise ContractViolation(
            f"correlation block field {field.id.name} requires a numeric deterministic model"
        )
    return model


def _correlation_vector_anchor_field(
    block: CorrelationBlockRequest,
    request: _FieldOwnerRequest,
) -> str | None:
    coupling_attributes = [
        getattr(coupling, "attribute_field")
        for coupling in request.vector_attribute_couplings
        if getattr(coupling, "attribute_field") in block.columns
    ]
    if not coupling_attributes:
        return None
    return str(coupling_attributes[0])


def _generate_categorical_association_block(
    *,
    block: CategoricalAssociationRequest,
    fields_by_name: dict[str, Field],
    request: _FieldOwnerRequest,
    row_count: int,
) -> tuple[ColumnMapping, tuple[DeterministicFieldModel, ...]]:
    left_name, right_name = block.columns
    left_field = fields_by_name.get(left_name)
    right_field = fields_by_name.get(right_name)
    if left_field is None or right_field is None:
        raise ContractViolation(f"categorical association {block.name} references unknown fields")
    left_request = request.field(left_name)
    right_request = request.field(right_name)
    if left_request is None or right_request is None:
        raise ContractViolation(f"categorical association {block.name} requires categorical generation requests")
    left_model = deterministic_model_for_field(left_field, row_count, request=left_request)
    right_model = deterministic_model_for_field(right_field, row_count, request=right_request)
    if not isinstance(left_model, CategoricalBlocksModel) or not isinstance(right_model, CategoricalBlocksModel):
        raise ContractViolation(f"categorical association {block.name} requires categorical deterministic models")
    matrix = categorical_association_matrix(
        row_counts=_category_counts(left_model),
        column_counts=_category_counts(right_model),
        target_cramers_v=block.target_cramers_v,
    )
    anchor_field = _categorical_association_vector_anchor(block, request.vector_attribute_couplings)
    left_values, right_values, left_blocks, right_blocks = _materialize_categorical_joint_values(
        matrix=matrix,
        left_field=left_field,
        right_field=right_field,
        left_labels=tuple(category.label for category in left_model.blocks),
        right_labels=tuple(category.label for category in right_model.blocks),
        row_count=row_count,
        anchor_field=anchor_field,
    )
    return (
        {
            left_field.id.name: left_values,
            right_field.id.name: right_values,
        },
        (
            _categorical_model_like(left_model, blocks=left_blocks),
            _categorical_model_like(right_model, blocks=right_blocks),
        ),
    )


def categorical_association_matrix(
    *,
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
    target_cramers_v: float,
) -> tuple[tuple[int, ...], ...]:
    _validate_marginals(row_counts, column_counts)
    target = max(0.0, min(1.0, target_cramers_v))
    exact_candidate = _exact_association_candidate(
        row_counts=row_counts,
        column_counts=column_counts,
        target_cramers_v=target,
    )
    if exact_candidate is not None:
        if target < 1.0 and abs(exact_candidate.observed - target) > CATEGORICAL_ASSOCIATION_TOLERANCE:
            raise ContractViolation(
                "categorical association target_cramers_v cannot be satisfied with current "
                f"marginals: target={target:g}, closest={exact_candidate.observed:.6g}"
            )
        return exact_candidate.matrix

    candidate = _ipf_association_candidate(
        row_counts=row_counts,
        column_counts=column_counts,
        target_cramers_v=target,
    )
    if target < 1.0 and abs(candidate.observed - target) > CATEGORICAL_ASSOCIATION_TOLERANCE:
        raise ContractViolation(
            "categorical association target_cramers_v cannot be satisfied within tolerance: "
            f"target={target:g}, observed={candidate.observed:.6g}, "
            f"tolerance={CATEGORICAL_ASSOCIATION_TOLERANCE:g}"
        )
    return candidate.matrix


def cramers_v(matrix: tuple[tuple[int, ...], ...]) -> float:
    total = sum(sum(row) for row in matrix)
    if total <= 0 or len(matrix) < 2 or not matrix or len(matrix[0]) < 2:
        return 0.0
    row_counts = [sum(row) for row in matrix]
    column_count = len(matrix[0])
    column_counts = [sum(row[column] for row in matrix) for column in range(column_count)]
    chi_square = 0.0
    for row_index, row in enumerate(matrix):
        for column_index, observed in enumerate(row):
            expected = row_counts[row_index] * column_counts[column_index] / total
            if expected <= 0.0:
                continue
            chi_square += ((observed - expected) ** 2) / expected
    denominator = total * min(len(matrix) - 1, column_count - 1)
    if denominator <= 0:
        return 0.0
    return float(max(0.0, min(1.0, np.sqrt(chi_square / denominator))))


def _validate_marginals(row_counts: tuple[int, ...], column_counts: tuple[int, ...]) -> None:
    if len(row_counts) < 2 or len(column_counts) < 2:
        raise ContractViolation("categorical association requires at least two categories on each side")
    if any(count < 0 for count in row_counts) or any(count < 0 for count in column_counts):
        raise ContractViolation("categorical association counts must be non-negative")
    if sum(row_counts) != sum(column_counts):
        raise ContractViolation("categorical association marginal counts must have the same total")


def _category_counts(model: CategoricalBlocksModel) -> tuple[int, ...]:
    return tuple(category.end - category.start + 1 for category in model.blocks)


def _exact_association_candidate(
    *,
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
    target_cramers_v: float,
) -> _AssociationCandidate | None:
    if sum(row_counts) > _EXACT_ASSOCIATION_TOTAL_LIMIT:
        return None
    estimated_tables = _estimate_contingency_table_count(row_counts, column_counts)
    if estimated_tables > _EXACT_ASSOCIATION_TABLE_LIMIT:
        return None

    preference = _association_preference_scores(
        row_count=len(row_counts),
        column_count=len(column_counts),
    )
    best: _AssociationCandidate | None = None
    table_count = 0
    for matrix in _enumerate_matrices(row_counts, column_counts):
        table_count += 1
        if table_count > _EXACT_ASSOCIATION_TABLE_LIMIT:
            return None
        candidate = _candidate_for_matrix(matrix, preference=preference)
        if best is None or _candidate_sort_key(candidate, target_cramers_v) < _candidate_sort_key(best, target_cramers_v):
            best = candidate
    return best


def _estimate_contingency_table_count(row_counts: tuple[int, ...], column_counts: tuple[int, ...]) -> int:
    estimate = 1
    columns = len(column_counts)
    for row_count in row_counts[:-1]:
        estimate *= math.comb(row_count + columns - 1, columns - 1)
        if estimate > _EXACT_ASSOCIATION_TABLE_LIMIT:
            return estimate
    return estimate


def _enumerate_matrices(
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    row_total = len(row_counts)
    column_total = len(column_counts)
    matrix = [[0 for _ in range(column_total)] for _ in range(row_total)]
    column_remaining = list(column_counts)

    def fill_cell(row_index: int, column_index: int, row_remaining: int) -> Iterator[tuple[tuple[int, ...], ...]]:
        if row_index == row_total:
            if all(remaining == 0 for remaining in column_remaining):
                yield tuple(tuple(row) for row in matrix)
            return
        if column_index == column_total - 1:
            value = row_remaining
            if 0 <= value <= column_remaining[column_index]:
                matrix[row_index][column_index] = value
                column_remaining[column_index] -= value
                next_row = row_index + 1
                next_remaining = row_counts[next_row] if next_row < row_total else 0
                yield from fill_cell(next_row, 0, next_remaining)
                column_remaining[column_index] += value
                matrix[row_index][column_index] = 0
            return
        maximum = min(row_remaining, column_remaining[column_index])
        for value in range(maximum + 1):
            matrix[row_index][column_index] = value
            column_remaining[column_index] -= value
            yield from fill_cell(row_index, column_index + 1, row_remaining - value)
            column_remaining[column_index] += value
            matrix[row_index][column_index] = 0

    if not row_counts:
        return iter(())
    return fill_cell(0, 0, row_counts[0])


def _candidate_for_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    preference: tuple[tuple[float, ...], ...],
) -> _AssociationCandidate:
    return _AssociationCandidate(
        matrix=matrix,
        observed=cramers_v(matrix),
        preference=_matrix_preference(matrix, preference),
    )


def _candidate_sort_key(candidate: _AssociationCandidate, target_cramers_v: float) -> tuple[float, int, float]:
    return _association_sort_key(candidate.observed, candidate.preference, target_cramers_v)


def _association_sort_key(
    observed_cramers_v: float,
    preference_score: float,
    target_cramers_v: float,
) -> tuple[float, int, float]:
    return (
        abs(observed_cramers_v - target_cramers_v),
        0 if observed_cramers_v <= target_cramers_v + 1.0e-12 else 1,
        -preference_score,
    )


def _ipf_association_candidate(
    *,
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
    target_cramers_v: float,
) -> _AssociationCandidate:
    preference = _association_preference_scores(
        row_count=len(row_counts),
        column_count=len(column_counts),
    )
    theta_low = 0.0
    theta_high = 1.0
    high_matrix = _ipf_matrix(row_counts, column_counts, preference, theta_high)
    while _continuous_cramers_v(high_matrix) < target_cramers_v and theta_high < _IPF_THETA_MAX:
        theta_low = theta_high
        theta_high *= 2.0
        high_matrix = _ipf_matrix(row_counts, column_counts, preference, theta_high)

    continuous_candidates = [
        _ipf_matrix(row_counts, column_counts, preference, 0.0),
        high_matrix,
    ]
    for _ in range(_IPF_THETA_SEARCH_ITERATIONS):
        theta = (theta_low + theta_high) / 2.0
        matrix = _ipf_matrix(row_counts, column_counts, preference, theta)
        continuous_candidates.append(matrix)
        if _continuous_cramers_v(matrix) <= target_cramers_v:
            theta_low = theta
        else:
            theta_high = theta

    rounded_candidates: list[_AssociationCandidate] = []
    for continuous in continuous_candidates:
        rounded = _rounded_matrix(continuous.tolist(), row_counts, column_counts)
        rounded_candidates.append(_candidate_for_matrix(rounded, preference=preference))

    candidates = _dedupe_candidates(rounded_candidates)
    repair_inputs = sorted(
        candidates,
        key=lambda candidate: _candidate_sort_key(candidate, target_cramers_v),
    )[:_LOCAL_REPAIR_CANDIDATE_LIMIT]
    for candidate in repair_inputs:
        candidates.append(
            _repair_matrix(
                candidate.matrix,
                target_cramers_v=target_cramers_v,
                preference=preference,
            )
        )

    return min(candidates, key=lambda candidate: _candidate_sort_key(candidate, target_cramers_v))


def _dedupe_candidates(candidates: list[_AssociationCandidate]) -> list[_AssociationCandidate]:
    unique: list[_AssociationCandidate] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for candidate in candidates:
        if candidate.matrix in seen:
            continue
        seen.add(candidate.matrix)
        unique.append(candidate)
    return unique


def _ipf_matrix(
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
    preference: tuple[tuple[float, ...], ...],
    theta: float,
) -> np.ndarray:
    row_targets = np.asarray(row_counts, dtype=float)
    column_targets = np.asarray(column_counts, dtype=float)
    scores = np.asarray(preference, dtype=float)
    shifted = scores - float(np.max(scores))
    matrix = np.exp(theta * shifted)
    matrix[row_targets <= 0.0, :] = 0.0
    matrix[:, column_targets <= 0.0] = 0.0
    positive_rows = row_targets > 0.0
    positive_columns = column_targets > 0.0
    if not np.any(matrix[positive_rows][:, positive_columns]):
        matrix[np.ix_(positive_rows, positive_columns)] = 1.0
    for _ in range(_IPF_MAX_ITERATIONS):
        row_sums = matrix.sum(axis=1)
        for row_index, row_sum in enumerate(row_sums):
            if row_targets[row_index] <= 0.0:
                continue
            if row_sum <= 0.0:
                matrix[row_index, positive_columns] = row_targets[row_index] / float(np.sum(positive_columns))
            else:
                matrix[row_index, :] *= row_targets[row_index] / row_sum
        column_sums = matrix.sum(axis=0)
        for column_index, column_sum in enumerate(column_sums):
            if column_targets[column_index] <= 0.0:
                continue
            if column_sum <= 0.0:
                matrix[positive_rows, column_index] = column_targets[column_index] / float(np.sum(positive_rows))
            else:
                matrix[:, column_index] *= column_targets[column_index] / column_sum
        if (
            np.max(np.abs(matrix.sum(axis=1) - row_targets)) <= 1.0e-6
            and np.max(np.abs(matrix.sum(axis=0) - column_targets)) <= 1.0e-6
        ):
            break
    return matrix


def _continuous_cramers_v(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    if total <= 0.0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0
    row_counts = matrix.sum(axis=1)
    column_counts = matrix.sum(axis=0)
    expected = np.outer(row_counts, column_counts) / total
    mask = expected > 0.0
    chi_square = float(np.sum(((matrix[mask] - expected[mask]) ** 2) / expected[mask]))
    denominator = total * min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, math.sqrt(chi_square / denominator))))


def _repair_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    target_cramers_v: float,
    preference: tuple[tuple[float, ...], ...],
) -> _AssociationCandidate:
    current = _candidate_for_matrix(matrix, preference=preference)
    row_count = len(matrix)
    column_count = len(matrix[0])
    swap_space = math.comb(row_count, 2) * math.comb(column_count, 2) * 2
    if swap_space > _LOCAL_REPAIR_SWAP_LIMIT:
        return current
    row_counts = tuple(sum(row) for row in matrix)
    column_counts = tuple(
        sum(row[column] for row in matrix)
        for column in range(column_count)
    )
    total = sum(row_counts)
    if total <= 0:
        return current
    denominator = total * min(row_count - 1, column_count - 1)
    if denominator <= 0:
        return current
    expected = tuple(
        tuple(row_counts[row] * column_counts[column] / total for column in range(column_count))
        for row in range(row_count)
    )
    chi_square = _chi_square_from_matrix(current.matrix, expected)
    for _ in range(_LOCAL_REPAIR_MAX_STEPS):
        current_key = _candidate_sort_key(current, target_cramers_v)
        best_key = current_key
        best_chi_square = chi_square
        best_observed = current.observed
        best_preference = current.preference
        best_swap: tuple[int, int, int, int, int] | None = None
        for row_a in range(row_count):
            for row_b in range(row_a + 1, row_count):
                for column_a in range(column_count):
                    for column_b in range(column_a + 1, column_count):
                        for delta in (1, -1):
                            next_chi_square = _swap_chi_square(
                                current.matrix,
                                expected,
                                chi_square,
                                row_a,
                                row_b,
                                column_a,
                                column_b,
                                delta,
                            )
                            if next_chi_square is None:
                                continue
                            observed = math.sqrt(next_chi_square / denominator)
                            bounded_observed = float(max(0.0, min(1.0, observed)))
                            next_preference = _swap_preference(
                                current.preference,
                                preference,
                                row_a,
                                row_b,
                                column_a,
                                column_b,
                                delta,
                            )
                            next_key = _association_sort_key(
                                bounded_observed,
                                next_preference,
                                target_cramers_v,
                            )
                            if next_key < best_key:
                                best_key = next_key
                                best_chi_square = next_chi_square
                                best_observed = bounded_observed
                                best_preference = next_preference
                                best_swap = (row_a, row_b, column_a, column_b, delta)
        if best_swap is None or best_key >= current_key:
            return current
        row_a, row_b, column_a, column_b, delta = best_swap
        current = _AssociationCandidate(
            matrix=_apply_swap_unchecked(
                current.matrix,
                row_a,
                row_b,
                column_a,
                column_b,
                delta,
            ),
            observed=best_observed,
            preference=best_preference,
        )
        chi_square = best_chi_square
    return current


def _chi_square_from_matrix(
    matrix: tuple[tuple[int, ...], ...],
    expected: tuple[tuple[float, ...], ...],
) -> float:
    return float(
        sum(
            _chi_square_cell(matrix[row][column], expected[row][column])
            for row in range(len(matrix))
            for column in range(len(matrix[0]))
        )
    )


def _swap_chi_square(
    matrix: tuple[tuple[int, ...], ...],
    expected: tuple[tuple[float, ...], ...],
    chi_square: float,
    row_a: int,
    row_b: int,
    column_a: int,
    column_b: int,
    delta: int,
) -> float | None:
    next_values = (
        matrix[row_a][column_a] + delta,
        matrix[row_b][column_b] + delta,
        matrix[row_a][column_b] - delta,
        matrix[row_b][column_a] - delta,
    )
    if any(value < 0 for value in next_values):
        return None
    previous = (
        (row_a, column_a, matrix[row_a][column_a]),
        (row_b, column_b, matrix[row_b][column_b]),
        (row_a, column_b, matrix[row_a][column_b]),
        (row_b, column_a, matrix[row_b][column_a]),
    )
    updated = (
        (row_a, column_a, next_values[0]),
        (row_b, column_b, next_values[1]),
        (row_a, column_b, next_values[2]),
        (row_b, column_a, next_values[3]),
    )
    next_chi_square = chi_square
    for row, column, value in previous:
        next_chi_square -= _chi_square_cell(value, expected[row][column])
    for row, column, value in updated:
        next_chi_square += _chi_square_cell(value, expected[row][column])
    return max(0.0, next_chi_square)


def _chi_square_cell(observed: int, expected: float) -> float:
    if expected <= 0.0:
        return 0.0
    return ((observed - expected) ** 2) / expected


def _swap_preference(
    current_preference: float,
    preference: tuple[tuple[float, ...], ...],
    row_a: int,
    row_b: int,
    column_a: int,
    column_b: int,
    delta: int,
) -> float:
    return current_preference + delta * (
        preference[row_a][column_a]
        + preference[row_b][column_b]
        - preference[row_a][column_b]
        - preference[row_b][column_a]
    )


def _apply_swap_unchecked(
    matrix: tuple[tuple[int, ...], ...],
    row_a: int,
    row_b: int,
    column_a: int,
    column_b: int,
    delta: int,
) -> tuple[tuple[int, ...], ...]:
    mutable = [list(row) for row in matrix]
    mutable[row_a][column_a] += delta
    mutable[row_b][column_b] += delta
    mutable[row_a][column_b] -= delta
    mutable[row_b][column_a] -= delta
    if (
        mutable[row_a][column_a] < 0
        or mutable[row_b][column_b] < 0
        or mutable[row_a][column_b] < 0
        or mutable[row_b][column_a] < 0
    ):
        raise ContractViolation("categorical association swap produced a negative cell")
    return tuple(tuple(row) for row in mutable)


def _association_preference_scores(
    *,
    row_count: int,
    column_count: int,
) -> tuple[tuple[float, ...], ...]:
    denominator = max(row_count, column_count) - 1
    if denominator <= 0:
        return tuple(tuple(1.0 for _ in range(column_count)) for _ in range(row_count))
    scores: list[list[float]] = []
    for row_index in range(row_count):
        row_score: list[float] = []
        for column_index in range(column_count):
            row_position = row_index / max(row_count - 1, 1)
            column_position = column_index / max(column_count - 1, 1)
            distance = abs(row_position - column_position)
            row_score.append(1.0 - distance)
        scores.append(row_score)
    return tuple(tuple(row) for row in scores)


def _matrix_preference(
    matrix: tuple[tuple[int, ...], ...],
    preference: tuple[tuple[float, ...], ...],
) -> float:
    return float(
        sum(
            count * preference[row_index][column_index]
            for row_index, row in enumerate(matrix)
            for column_index, count in enumerate(row)
        )
    )


def _rounded_matrix(
    values: list[list[float]],
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    matrix = [[int(np.floor(value)) for value in row] for row in values]
    row_deficit = [row_counts[row] - sum(matrix[row]) for row in range(len(row_counts))]
    column_deficit = [
        column_counts[column] - sum(matrix[row][column] for row in range(len(row_counts)))
        for column in range(len(column_counts))
    ]
    remaining = sum(row_deficit)
    if remaining != sum(column_deficit):
        raise ContractViolation("categorical association rounding deficits are inconsistent")
    if remaining > 0:
        for row, column in _rounding_assignments(values, matrix, row_deficit, column_deficit):
            matrix[row][column] += 1
    result = tuple(tuple(row) for row in matrix)
    _validate_matrix(result, row_counts, column_counts)
    return result


def _rounding_assignments(
    values: list[list[float]],
    matrix: list[list[int]],
    row_deficit: list[int],
    column_deficit: list[int],
) -> tuple[tuple[int, int], ...]:
    from ortools.graph.python import min_cost_flow  # type: ignore[import-untyped]

    row_count = len(row_deficit)
    column_count = len(column_deficit)
    required = sum(row_deficit)
    source = 0
    row_offset = 1
    column_offset = row_offset + row_count
    sink = column_offset + column_count
    solver = min_cost_flow.SimpleMinCostFlow()
    for row, deficit in enumerate(row_deficit):
        solver.add_arc_with_capacity_and_unit_cost(source, row_offset + row, deficit, 0)
    for row in range(row_count):
        for column in range(column_count):
            fraction = values[row][column] - matrix[row][column]
            solver.add_arc_with_capacity_and_unit_cost(
                row_offset + row,
                column_offset + column,
                1,
                -int(round(fraction * 1_000_000_000)),
            )
    for column, deficit in enumerate(column_deficit):
        solver.add_arc_with_capacity_and_unit_cost(column_offset + column, sink, deficit, 0)
    for node in range(sink + 1):
        if node == source:
            solver.set_node_supply(node, required)
        elif node == sink:
            solver.set_node_supply(node, -required)
        else:
            solver.set_node_supply(node, 0)
    if solver.solve() != solver.OPTIMAL:
        raise ContractViolation("categorical association rounding flow is infeasible")
    assignments: list[tuple[int, int]] = []
    for arc in range(solver.num_arcs()):
        tail = solver.tail(arc)
        head = solver.head(arc)
        row = tail - row_offset
        column = head - column_offset
        if 0 <= row < row_count and 0 <= column < column_count:
            if solver.flow(arc) > 0:
                assignments.append((row, column))
    if len(assignments) != required:
        raise ContractViolation("categorical association rounding could not assign all deficits")
    return tuple(assignments)


def _validate_matrix(
    matrix: tuple[tuple[int, ...], ...],
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
) -> None:
    if len(matrix) != len(row_counts):
        raise ContractViolation("categorical association matrix row count mismatch")
    for row in matrix:
        if len(row) != len(column_counts):
            raise ContractViolation("categorical association matrix column count mismatch")
        if any(count < 0 for count in row):
            raise ContractViolation("categorical association matrix counts must be non-negative")
    observed_rows = tuple(sum(row) for row in matrix)
    observed_columns = tuple(sum(row[column] for row in matrix) for column in range(len(column_counts)))
    if observed_rows != row_counts or observed_columns != column_counts:
        raise ContractViolation("categorical association matrix must preserve marginal counts")


def _materialize_categorical_joint_values(
    *,
    matrix: tuple[tuple[int, ...], ...],
    left_field: Field,
    right_field: Field,
    left_labels: tuple[str, ...],
    right_labels: tuple[str, ...],
    row_count: int,
    anchor_field: str | None = None,
) -> tuple[list[object], list[object], tuple[CategoryBlock, ...], tuple[CategoryBlock, ...]]:
    left_values: list[object] = []
    right_values: list[object] = []
    left_blocks: list[CategoryBlock] = []
    right_blocks: list[CategoryBlock] = []

    if anchor_field == right_field.id.name:
        cells = (
            (row_index, column_index, matrix[row_index][column_index])
            for column_index in range(len(right_labels))
            for row_index in range(len(left_labels))
        )
    else:
        cells = (
            (row_index, column_index, matrix[row_index][column_index])
            for row_index in range(len(left_labels))
            for column_index in range(len(right_labels))
        )

    for row_index, column_index, count in cells:
        if count <= 0:
            continue
        left_label = left_labels[row_index]
        right_label = right_labels[column_index]
        start = len(left_values)
        end = start + count - 1
        left_values.extend([_coerce_association_value(left_field, left_label)] * count)
        right_values.extend([_coerce_association_value(right_field, right_label)] * count)
        left_blocks.append(CategoryBlock(label=left_label, start=start, end=end))
        right_blocks.append(CategoryBlock(label=right_label, start=start, end=end))
    if len(left_values) != row_count or len(right_values) != row_count:
        raise ContractViolation("categorical association did not materialize the expected row count")
    return left_values, right_values, tuple(left_blocks), tuple(right_blocks)


def _categorical_association_vector_anchor(
    block: CategoricalAssociationRequest,
    couplings: tuple[VectorAttributeCouplingRequest, ...],
) -> str | None:
    anchors = tuple(
        request.attribute_field
        for request in couplings
        if request.attribute_field in block.columns
    )
    if len(anchors) > 1:
        raise ContractViolation(
            f"categorical association {block.name} has more than one vector-coupled attribute: {anchors}"
        )
    return anchors[0] if anchors else None


def _coerce_association_value(field: Field, label: str) -> object:
    if isinstance(field.type, IntType):
        return int(float(label))
    if isinstance(field.type, TextType):
        return label
    raise ContractViolation(f"categorical association requires int/text field, got {type(field.type).__name__}")


def _categorical_model_like(
    model: CategoricalBlocksModel,
    *,
    blocks: tuple[CategoryBlock, ...],
) -> CategoricalBlocksModel:
    return CategoricalBlocksModel(
        field_name=model.field_name,
        value_type=model.value_type,
        row_count=model.row_count,
        non_null_row_count=model.non_null_row_count,
        nulls_first=model.nulls_first,
        blocks=blocks,
    )


def _table_field_owner_request(request: TableGenerationRequest | None) -> _TableFieldOwnerRequest | None:
    if request is None:
        return None
    return cast(_TableFieldOwnerRequest, request)


__all__ = [
    "ColumnMapping",
    "ColumnValues",
    "materialize_owner_scalar_fields",
    "materialize_table_fields",
    "materialize_table_vectors",
]
